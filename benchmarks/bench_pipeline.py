#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Where one block assembly spends its time, stage by stage.

``bench_assembly.py`` answers "how fast is an assembly"; this answers "which part of
it".  The two questions need different measurements: the stages overlap, and JAX
dispatches asynchronously, so timing them separately double-counts.  Each stage here
is therefore a *pipeline* run from the same starting point, one step longer than the
last, ending in a blocking read.  The increments are then real costs.

The stages follow :mod:`hippymfem.fem.csrassemble`:

=========================  ===================================================
kernel                     dof gather and the differentiated element kernel
scatter                    reduction of element entries into the local CSR
build ldof matrix          that CSR as a block-diagonal ``HypreParMatrix``
``P^T A P``                the parallel reduction to true dofs
eliminate                  essential rows and columns, and the diagonal
=========================  ===================================================

This is the measurement that showed the AD layer was not the bottleneck: at four
ranks the kernel was 10% of an assembly while the triple product was 55% and the
elimination 25%.  See ``NOTES.md`` for what changed as a result.

Usage::

    HIPPYMFEM_DEVICE=gpu mpirun -n 4 python benchmarks/bench_pipeline.py
    HIPPYMFEM_DEVICE=gpu mpirun -n 4 python benchmarks/bench_pipeline.py \\
        --kind hex --n 30 --order 2 --out results/pipeline_004.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hp                                          # noqa: E402
from hippymfem.fem import csrassemble as C                      # noqa: E402
from hippymfem.fem import elimination                           # noqa: E402
from hippymfem.fem.kernel import set_device, device             # noqa: E402
from hippymfem.modeling.variables import ADJOINT, STATE         # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size

STAGES = ["kernel", "+ scatter", "+ build ldof matrix", "+ P^T A P",
          "+ eliminate = full"]


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def build(kind, n, order):
    mesh = (mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON)
            if kind == "hex" else
            mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.QUADRILATERAL)
            if kind == "quad" else
            mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE))
    pm = mfem.ParMesh(COMM, mesh)
    Vu = hp.FunctionSpace.H1(pm, order)
    Vm = hp.FunctionSpace.H1(pm, 1)

    def varf(u, m, p, x):
        return jnp.exp(m.val) * jnp.dot(u.grad, p.grad) - p.val

    bc = hp.DirichletBC(Vu, None, "all")
    batches = hp.MeshBatches(pm, 2 * order + 2, COMM)
    kern = hp.QuadratureKernel(varf, [Vu, Vm, Vu], batches)
    hp.parRandom.set_seed(21)
    vecs = [Vu.vector(), Vm.vector(), Vu.vector()]
    for v, s in zip(vecs, (1.0, 0.4, 1.0)):
        hp.parRandom.normal(s, v)
    loc = [Vu.local_values(vecs[0]), Vm.local_values(vecs[1]),
           Vu.local_values(vecs[2])]
    return pm, Vu, Vm, batches, kern, loc, bc


def timed(fn, reps):
    fn()
    COMM.Barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn()
        del r
    COMM.Barrier()
    return (time.perf_counter() - t0) / reps


def pipelines(kind, n, order, dev, reps):
    pm, Vu, Vm, batches, kern, loc, bc = build(kind, n, order)
    set_device(dev)
    NE = pm.GetNE()
    pat = C.get_pattern(Vu, Vu, batches.groups)
    P, ident = C._prolongation(Vu)
    fold = elimination.FOLD_ELIMINATION and C._boolean_prolongation(Vu)
    kill = (pat.masked_slots(C._ess_ldof_mask(Vu, bc.ess_tdof),
                             C._ess_ldof_mask(Vu, bc.ess_tdof))
            if fold else None)
    # Which route the library takes decides which stages exist.  On one rank P is
    # the identity and the ldof matrix is the result; with a boolean P on several
    # ranks the entries go straight into true-dof rows and there is neither an ldof
    # matrix nor a triple product; otherwise the triple product is formed.
    tdof = (not ident) and C._tdof_route(Vu, Vu, True)
    tp = None
    if tdof:
        from hippymfem.fem.tdofassemble import get_tdof_pattern

        tp = get_tdof_pattern(pat, Vu, Vu)

    def E():
        return kern.element_matrices(ADJOINT, STATE, loc)

    def p_kernel():
        mats = E()
        for a in mats:                  # a blocking read: JAX dispatch is async
            np.asarray(a[:1, :1, :1])
        return mats

    def p_full():
        return C.assemble_matrix_csr(Vu, Vu, batches.groups, E(), NE,
                                     test_ess=bc.ess_tdof)

    if tdof:
        zero = tp.kill(kill)
        stages = ["kernel", "+ scatter (true-dof rows + send buffer)",
                  "+ exchange ghost rows + build matrix", "+ eliminate = full"]

        def p_scatter():
            return tp.target.data(E(), zero_slots=zero)

        def p_build():
            return tp.finish(tp.target.data(E(), zero_slots=zero))

        fns = (p_kernel, p_scatter, p_build, p_full)
    else:
        stages = list(STAGES)

        def p_scatter():
            return pat.data(E(), zero_slots=kill)

        def p_build():
            return C.local_par_matrix(pat, pat.data(E(), zero_slots=kill), Vu, Vu,
                                      reuse=not ident)

        def p_triple():
            A = C.local_par_matrix(pat, pat.data(E(), zero_slots=kill), Vu, Vu,
                                   reuse=not ident)
            if ident:
                return A
            return C._triple(A, None, P, COMM, (id(Vu.fes),))

        fns = (p_kernel, p_scatter, p_build, p_triple, p_full)
    t = [timed(f, reps) for f in fns]
    return {
        "kind": kind, "n": n, "order": order, "ranks": NP,
        "device": "gpu" if str(device()).lower().startswith("cuda") else "cpu",
        "requested_device": dev,
        "elements_local": NE, "elements": COMM.allreduce(NE),
        "state_dofs": Vu.GlobalTrueVSize(),
        "prolongation": "identity" if ident else "real",
        "route": ("identity" if ident else "true-dof" if tdof else "triple product"),
        "folded_elimination": bool(fold),
        "triple_form": C._TRIPLE_CHOICE.get((id(Vu.fes),), "n/a"),
        "stages": stages,
        "cumulative_s": dict(zip(stages, t)),
    }


def report(rows):
    """Print one table per device, with the increments and their share."""
    for row in rows:
        stages = row.get("stages", STAGES)
        t = [row["cumulative_s"][s] for s in stages]
        total = t[-1]
        say("%s order %d | %d rank(s) | %d elem/rank | %d dofs | route: %s | "
            "triple=%s | folded=%s"
            % (row["kind"], row["order"], row["ranks"], row["elements_local"],
               row["state_dofs"], row.get("route", row["prolongation"]),
               row["triple_form"], row["folded_elimination"]))
        say("  %-40s %11s %11s %8s" % ("stage", "cumul. ms", "stage ms", "share"))
        prev = 0.0
        for name, v in zip(stages, t):
            say("  %-40s %11.1f %11.1f %7.1f%%"
                % (name, 1e3 * v, 1e3 * (v - prev),
                   100.0 * (v - prev) / max(total, 1e-30)))
            prev = v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="hex", choices=("hex", "quad", "tri"))
    ap.add_argument("--n", type=int, default=30, help="cells per side")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--devices", default="cpu,gpu",
                    help="comma-separated subset of cpu,gpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for dev in args.devices.split(","):
        dev = dev.strip()
        if not dev:
            continue
        try:
            rows.append(pipelines(args.kind, args.n, args.order, dev, args.reps))
        except RuntimeError as exc:          # no GPU visible: say so, do not fail
            say("  %s skipped: %s" % (dev, exc))
    set_device("cpu")
    report(rows)
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"experiment": "pipeline", "rows": rows}, fh, indent=1)
        say("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

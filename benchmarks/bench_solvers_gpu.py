#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""hypre's solvers on the CPU against the GPU, in the configuration hIPPyMFEM uses.

With a CUDA-enabled PyMFEM there are two independent questions, and conflating them
gives a misleading answer:

1. **Does MFEM's own assembly get faster on a GPU?**  Its *legacy* full assembly does
   not -- it is 50 times slower on the device in our measurement, because building a
   sparse matrix element by element is not what the device path is designed for;
   MFEM's GPU story is partial assembly with a matrix-free operator.  hIPPyMFEM does
   not use MFEM's assembly at all: element arrays come from JAX and are scattered
   directly into the sparse structure, so this does not apply to it.

2. **Do hypre's solves get faster on a GPU?**  That is the question here.  The matrix
   is built the way hIPPyMFEM builds it, and only the solve is timed, swept over
   problem size, because GPU algebraic multigrid needs enough work per device to pay
   for its setup.

Run the two configurations separately -- hypre's execution policy is fixed once, by
``Hypre::Init()`` from MFEM's device::

    PYTHONPATH=<cuda-pymfem> mpirun -n 4 python benchmarks/bench_solvers_gpu.py --out results/solve_cpu.json
    PYTHONPATH=<cuda-pymfem> mpirun -n 4 python benchmarks/bench_solvers_gpu.py --gpu --out results/solve_gpu.json
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true",
                    help="run hypre on the device (needs a CUDA PyMFEM)")
    ap.add_argument("--orders", default="1,2")
    ap.add_argument("--cells", default="16,24,32,40,48",
                    help="cells per side, comma separated")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--nrhs", type=int, default=8,
                    help="right-hand sides for the steady-state solve timing; a "
                         "Newton-CG inner loop amortizes the AMG setup this way")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # The device, and hence hypre's memory and execution policy, is set once here.
    # **One GPU per rank.**  mfem.Device("cuda") with no device id puts every rank on
    # device 0; that does not announce itself, it just looks like the device scaling
    # badly.
    import hippymfem as hp                                          # noqa: E402

    dev = hp.configure_device("cuda" if args.gpu else "cpu", quiet=True)
    mfem.Hypre.Init()
    if RANK == 0:
        dev.Print()
    if args.gpu:
        # The direct-CSR route builds the matrix from numpy arrays, which a
        # device-configured hypre reads as device pointers.  MFEM's own assembly is
        # the route that survives that; it is slower, which is the price of having
        # the solves on the device at all.
        from hippymfem.fem import assemble as _asm

        _asm.set_assembly_backend("integrator")

    cfg = hp.mfem_config()
    if args.gpu and not cfg.get("MFEM_USE_CUDA"):
        say("  this PyMFEM has no CUDA (MFEM %s); nothing to measure."
            % cfg.get("version"))
        return 0

    rows = []
    say("hypre solves on %s, %d ranks, MFEM %s (CUDA %s)"
        % ("GPU" if args.gpu else "CPU", NP, cfg.get("version"),
           cfg.get("MFEM_USE_CUDA")))
    say("%6s %5s %12s %11s %9s %11s %11s %11s"
        % ("cells", "order", "true dofs", "build (s)", "CG its",
           "AMG setup", "cold solve", "steady solve"))

    for order in [int(o) for o in args.orders.split(",")]:
        for n in [int(c) for c in args.cells.split(",")]:
            r = one(n, order, args.tol, args.gpu, args.nrhs)
            if r is None:
                continue
            rows.append(r)
            say("%6d %5d %12d %11.2f %9d %11.2f %11.2f %11.4f"
                % (n, order, r["tdofs"], r["t_build"], r["cg_its"],
                   r["t_setup"], r["t_solve"], r["t_solve_steady"]))

    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"gpu": bool(args.gpu), "ranks": NP,
                   "host": platform.node(),
                   "mfem": cfg.get("version"),
                   "mfem_cuda": cfg.get("MFEM_USE_CUDA"),
                   "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rows": rows}, open(args.out, "w"), indent=1)
        say("wrote %s" % args.out)
    return 0


def one(n, order, tol, gpu, nrhs=8):
    """Build the operator the way hIPPyMFEM does, then time only the solve."""
    import jax.numpy as jnp

    import hippymfem as hp
    from hippymfem.fem.elementbatch import MeshBatches
    from hippymfem.fem.kernel import QuadratureKernel
    from hippymfem.fem.assemble import assemble_matrix
    from hippymfem.modeling.variables import ADJOINT, STATE

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(
        n, n, n, mfem.Element.HEXAHEDRON))
    Vu = hp.FunctionSpace.H1(pm, order)
    Vm = hp.FunctionSpace.H1(pm, 1)
    batches = MeshBatches(pm, 2 * order + 2, COMM)
    kern = QuadratureKernel(
        lambda u, m, p, x: jnp.exp(m.val) * jnp.dot(u.grad, p.grad),
        [Vu, Vm, Vu], batches)
    hp.parRandom.set_seed(3)
    mv = Vm.vector()
    hp.parRandom.normal(0.3, mv)
    loc = [Vu.local_values(Vu.vector()), Vm.local_values(mv),
           Vu.local_values(Vu.vector())]
    bc = hp.DirichletBC(Vu, None, "all")

    COMM.Barrier()
    t0 = time.perf_counter()
    A = assemble_matrix(Vu, Vu, batches.groups,
                        kern.element_matrices(ADJOINT, STATE, loc),
                        pm.GetNE(), test_ess=bc.ess_tdof)
    COMM.Barrier()
    t_build = time.perf_counter() - t0

    b = Vu.vector()
    hp.parRandom.normal(1.0, b)
    bc.zero(b)
    x = Vu.vector()

    COMM.Barrier()
    t0 = time.perf_counter()
    amg = mfem.HypreBoomerAMG(A)
    amg.SetPrintLevel(0)
    amg.SetOperator(A)
    # force the setup to happen now rather than inside the first solve
    tmp = Vu.vector()
    amg.Mult(b.hypre, tmp.hypre)
    COMM.Barrier()
    t_setup = time.perf_counter() - t0

    cg = mfem.CGSolver(COMM)
    cg.SetOperator(A)
    cg.SetPreconditioner(amg)
    cg.SetRelTol(tol)
    cg.SetMaxIter(500)
    cg.SetPrintLevel(-1)
    COMM.Barrier()
    t0 = time.perf_counter()
    cg.Mult(b.hypre, x.hypre)
    COMM.Barrier()
    t_solve = time.perf_counter() - t0

    # The steady state is the number that matters: a Newton-CG inner loop builds the
    # multigrid hierarchy once per linearization point and then solves with it tens
    # of times, so the setup is amortized and timing one cold solve understates the
    # device by a wide margin.
    rhs = [Vu.vector() for _ in range(nrhs)]
    for v in rhs:
        hp.parRandom.normal(1.0, v)
        bc.zero(v)
    COMM.Barrier()
    t0 = time.perf_counter()
    its = 0
    for v in rhs:
        x.zero()
        cg.Mult(v.hypre, x.hypre)
        its += int(cg.GetNumIterations())
    COMM.Barrier()
    t_steady = (time.perf_counter() - t0) / max(nrhs, 1)
    its_steady = its / max(nrhs, 1)
    cg.Mult(b.hypre, x.hypre)          # restore x for the residual check below

    # the residual, so a "fast" solve that did not solve cannot pass unnoticed
    r = Vu.vector()
    A.Mult(x.hypre, r.hypre)
    r.axpy(-1.0, b)
    rel = r.norm("l2") / max(b.norm("l2"), 1e-300)
    return {"n": n, "order": order, "elements": COMM.allreduce(pm.GetNE()),
            "tdofs": Vu.GlobalTrueVSize(), "t_build": t_build,
            "t_setup": t_setup, "t_solve": t_solve,
            "t_solve_steady": t_steady, "cg_its_steady": its_steady,
            "cg_its": int(cg.GetNumIterations()),
            "residual": rel, "gpu": bool(gpu)}


if __name__ == "__main__":
    raise SystemExit(main())

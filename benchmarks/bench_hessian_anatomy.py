#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What a reduced-Hessian action does besides its two Krylov solves.

``bench_scaling.py`` times the nine operations of an action one by one and completes each
with a norm.  This script times the action as the library runs it (``H.mult``) and, with
the library of ``tools/gpuprof.c`` preloaded, counts what it asks of the driver: how many copies between
CPU and GPU memory, how many bytes and how long.  It then repeats the two incremental
solves with the vectors left on the GPU (MFEM's solver called directly), so that the
difference is the cost of the library's own vector handling: the copy of the right-hand
side, the zeroing of the boundary entries and of the solution, and the updates between
the operations, which run on the CPU.

The problem is that of ``bench_scaling.py`` and ``bench_newton_device.py``.
"""
import argparse
import ctypes
import json
import os
import platform
import sys
import time

import numpy as np

HM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HM)

import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

import mfem.par as mfem                                              # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

from hippymfem.modeling.variables import PARAMETER, STATE, ADJOINT   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def gpu_name():
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        n = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        return n[0] if n else "none"
    except Exception:                                                # noqa: BLE001
        return "unknown"


class Prof:
    """The counters of libgpuprof.so, if it is preloaded."""

    def __init__(self):
        self.lib = None
        try:
            lib = ctypes.CDLL(None)
            lib.gpuprof_nkinds.restype = ctypes.c_int
            self.n = lib.gpuprof_nkinds()
            lib.gpuprof_name.restype = ctypes.c_char_p
            self.names = [lib.gpuprof_name(k).decode() for k in range(self.n)]
            self.lib = lib
        except (AttributeError, OSError):
            self.n, self.names = 0, []

    def reset(self):
        if self.lib is not None:
            self.lib.gpuprof_reset()

    def read(self):
        """{name: (count, seconds, bytes)} since the last reset, for the kinds that occurred."""
        if self.lib is None:
            return {}
        c = (ctypes.c_long * self.n)()
        t = (ctypes.c_double * self.n)()
        b = (ctypes.c_double * self.n)()
        self.lib.gpuprof_get(c, t, b)
        return {self.names[k]: (c[k], t[k], b[k]) for k in range(self.n) if c[k]}


PROF = Prof()
COPIES = ("copy_d2h", "copy_h2d", "copy_d2d", "dev_alloc", "dev_free")


def copies(d):
    """'copy_d2h 5 (21.3 ms, 86.1 MB)  copy_h2d ...' for the copies and allocations."""
    out = []
    for k in COPIES:
        if k in d:
            c, t, b = d[k]
            out.append("%s %d (%.2f ms%s)" % (k, c, 1e3 * t, ", %.1f MB" % (b / 1e6) if b else ""))
    return "  ".join(out) if out else "(none)"


def timed(fn, sync=None):
    """Seconds for ``fn`` between barriers; ``sync`` is a ParVector whose norm completes
    the device work (it is read the way the library would read it next)."""
    COMM.Barrier()
    t0 = time.perf_counter()
    fn()
    if sync is not None:
        sync.norm("l2")
    COMM.Barrier()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"))
    ap.add_argument("--cg-tol", type=float, default=1e-12)
    ap.add_argument("--cart-part", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))
    N, ORDER = args.n, args.order
    serial = mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON)
    if args.cart_part:
        p = COMM.size
        grid = min(((a, b, p // (a * b)) for a in range(1, p + 1) if p % a == 0
                    for b in range(a, p // a + 1) if (p // a) % b == 0 and b <= p // (a * b)),
                   key=lambda g: max(g) / min(g))
        nxyz = mfem.intArray(list(grid))
        pmesh = mfem.ParMesh(COMM, serial, serial.CartesianPartitioning(nxyz.GetData()))
    else:
        pmesh = mfem.ParMesh(COMM, serial)
    del serial
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    ndof, mdof = int(Vu.GlobalTrueVSize()), int(Vm.GlobalTrueVSize())
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, tree %s, counters %s (%s)"
        % (N, ORDER, ndof, mdof, COMM.size, HM, "on" if PROF.lib is not None else "off", gpu_name()))
    say("  hypre matrix-vector kernel: %s" % getattr(hm.common.mfemconfig, "HYPRE_SPMV", "n/a"))

    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=args.cg_tol,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, args.ntargets) for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    model = hm.Model(pde, prior, hm.DiscreteStateObservation(B, data, nstd ** 2))

    x = [model.generate_vector(STATE), prior.mean.copy(), model.generate_vector(ADJOINT)]
    t_fwd = timed(lambda: model.solveFwd(x[STATE], x), x[STATE])
    t_adj = timed(lambda: model.solveAdj(x[ADJOINT], x), x[ADJOINT])
    g = model.generate_vector(PARAMETER)
    t_grad = timed(lambda: model.evalGradientParameter(x, g), g)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    x[PARAMETER].axpy(1e-3, prior.mean)
    t_blocks = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
    say("  forward %.3f s | adjoint %.3f s | gradient %.3f s | blocks %.3f s"
        % (t_fwd, t_adj, t_grad, t_blocks))

    H = hm.ReducedHessian(model)
    v = model.generate_vector(PARAMETER)
    hm.parRandom.normal(1.0, v)
    w = model.generate_vector(PARAMETER)
    H.mult(v, w)                                                   # warm
    H.mult(v, w)
    wnorm = w.norm("l2")

    # ---- the action as the library runs it
    ts_action, action_prof = [], {}
    for _ in range(args.repeats):
        PROF.reset()
        ts_action.append(timed(lambda: H.mult(v, w), w))
        action_prof = PROF.read()
    t_action = float(np.median(ts_action))
    it_f, it_a = int(pde.solver_fwd_inc.iterations), int(pde.solver_adj_inc.iterations)
    say("  Hessian action %.4f s (min %.4f, max %.4f); incremental solves %d and %d iterations; |Hv| %.12e"
        % (t_action, min(ts_action), max(ts_action), it_f, it_a, wnorm))
    say("      driver, per action: " + copies(action_prof))

    # ---- the nine operations and the three updates, each with its copies
    rhs_fwd, rhs_adj, rhs_adj2 = (model.generate_vector(STATE), model.generate_vector(ADJOINT),
                                  model.generate_vector(ADJOINT))
    uhat, phat, yhelp = (model.generate_vector(STATE), model.generate_vector(ADJOINT),
                         model.generate_vector(PARAMETER))
    steps = [
        ("applyC", lambda: model.applyC(v, rhs_fwd)),
        ("solveFwdInc", lambda: model.solveFwdIncremental(uhat, rhs_fwd)),
        ("applyWuu", lambda: model.applyWuu(uhat, rhs_adj)),
        ("applyWum", lambda: model.applyWum(v, rhs_adj2)),
        ("axpy rhs_adj", lambda: rhs_adj.axpy(-1.0, rhs_adj2)),
        ("solveAdjInc", lambda: model.solveAdjIncremental(phat, rhs_adj)),
        ("applyWmm", lambda: model.applyWmm(v, w)),
        ("applyCt", lambda: model.applyCt(phat, yhelp)),
        ("axpy y", lambda: w.axpy(1.0, yhelp)),
        ("applyWmu", lambda: model.applyWmu(uhat, yhelp)),
        ("axpy y 2", lambda: w.axpy(-1.0, yhelp)),
        ("applyR", lambda: model.applyR(v, yhelp)),
        ("axpy y 3", lambda: w.axpy(1.0, yhelp)),
    ]
    rows = {name: [] for name, _ in steps}
    last = {}
    for rep in range(args.repeats + 1):
        for name, fn in steps:
            PROF.reset()
            COMM.Barrier()
            t0 = time.perf_counter()
            fn()
            t = time.perf_counter() - t0          # no completion: the next step waits for it
            last[name] = PROF.read()
            if rep:
                rows[name].append(t)
    op = {name: float(np.median(ts_)) for name, ts_ in rows.items()}
    say("  step by step (the device work of a step may complete in the next one):")
    for name, _ in steps:
        say("      %-13s %8.2f ms   %s" % (name, 1e3 * op[name], copies(last[name])))
    say("      sum %.4f s" % sum(op.values()))

    # ---- the two solves with the vectors left on the GPU: MFEM's solver called directly
    raw = {}
    for name, solver, rhs in (("fwd", pde.solver_fwd_inc, rhs_fwd), ("adj", pde.solver_adj_inc, rhs_adj)):
        s = solver._solver
        b = rhs.copy()
        pde.bc0.zero(b)
        xs = model.generate_vector(STATE)
        s.Mult(b.hypre, xs.hypre)                  # uploads b, flags xs for the device
        xs.hypre.Norml2()
        ts, prof = [], {}
        for _ in range(args.repeats):
            COMM.Barrier()
            PROF.reset()
            t0 = time.perf_counter()
            s.Mult(b.hypre, xs.hypre)
            xs.hypre.Norml2()                      # a reduction on the device
            COMM.Barrier()
            ts.append(time.perf_counter() - t0)
            prof = PROF.read()
        its = int(s.GetNumIterations())
        raw[name] = {"t": float(np.median(ts)), "iterations": its}
        say("  MFEM's solver alone, %s: %.4f s, %d iterations, %.3f ms per iteration; %s"
            % (name, raw[name]["t"], its, 1e3 * raw[name]["t"] / max(its, 1), copies(prof)))
    solves = raw["fwd"]["t"] + raw["adj"]["t"]
    products = sum(op[k] for k in ("applyC", "applyWuu", "applyWum", "applyWmm", "applyCt",
                                   "applyWmu", "applyR"))
    say("  action %.4f s = solvers alone %.4f s (%.1f %%) + the rest %.4f s (%.1f %%)"
        % (t_action, solves, 100 * solves / t_action, t_action - solves,
           100 * (t_action - solves) / t_action))

    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N, "order": ORDER,
           "tdofs": ndof, "mdofs": mdof, "tree": HM, "tag": args.tag,
           "hypre_spmv": getattr(hm.common.mfemconfig, "HYPRE_SPMV", None),
           "cart_part": bool(args.cart_part), "t_fwd": t_fwd, "t_adj": t_adj, "t_grad": t_grad,
           "t_blocks": t_blocks, "t_action": t_action, "t_action_all": ts_action, "wnorm": wnorm,
           "it_fwd_inc": it_f, "it_adj_inc": it_a, "action_prof": action_prof, "op": op,
           "op_prof": last, "raw_solves": raw, "products": products, "repeats": args.repeats,
           "counters": PROF.lib is not None}
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
    COMM.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

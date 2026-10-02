#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The steps of a forward solve at a new parameter, timed one by one.

``bench_scaling.py`` times a forward solve as a whole.  This script runs the same steps
as ``PDEVariationalProblem.solveFwd`` does for a linear residual and times each between
synchronizations of the device: the release of the previous operators, the residual, the
Jacobian with its symmetry test, the hand-over of the operator to the solver, the solve
with its BoomerAMG setup, and the update with the second residual.  With the library of
``tools/gpuprof.c`` preloaded it also reports, per step, what the process asked of the
driver: copies between CPU and GPU memory with their bytes, and device allocations and
frees with their time, which is where a forward solve on many GPUs of one node loses
(``DESIGN_NOTES.md``, section 11).  At the end it prints the size of JAX's arena, the
device memory in use and what hypre's pool holds.

The problem is that of ``bench_scaling.py`` and ``bench_newton_device.py``::

    mpicc -O2 -fPIC -shared -o libgpuprof.so tools/gpuprof.c -ldl
    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n 16 -x LD_PRELOAD=$PWD/libgpuprof.so \\
        tools/mpirun_pinned.sh python benchmarks/bench_forward_steps.py --n 161 --cart-part
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
    from hippymfem.common import devicebridge as bridge
    from hippymfem.modeling.PDEVariationalProblem import _set_operator_once

    def sync():
        if bridge.available():
            bridge.synchronize()
        else:
            import jax
            (jax.device_put(0.0) + 0).block_until_ready()

    model.solveFwd(x[STATE], x)                       # warm
    rows = {}
    for rep in range(args.repeats + 1):
        x[PARAMETER].axpy(1e-3, mtrue)
        m = x[PARAMETER]
        t = {}
        COMM.Barrier(); t0 = time.perf_counter()
        pde.release_linearization_point() if pde._lin_point is not None else None
        u = x[STATE]; u.zero(); pde.bc.apply(u)
        p = pde.Vh[ADJOINT].vector(); du = pde.Vh[STATE].vector()
        solver = pde._get_solver("solver")
        sync(); t["vectors and release"] = time.perf_counter() - t0; t0 = time.perf_counter()
        PROF.reset()
        r = pde._residual([u, m, p], ADJOINT, ess=pde.bc0.ess); r0 = r.norm("l2")
        sync(); t["residual"] = time.perf_counter() - t0; t0 = time.perf_counter()
        prof_res = PROF.read(); PROF.reset()
        if rep == args.repeats and os.environ.get("JAC_PROFILE"):
            import cProfile, pstats
            pr = cProfile.Profile(); pr.enable()
            J, _Jt = pde._jacobian([u, m, p], u=u)
            sync(); pr.disable()
            if RANK == 0:
                pstats.Stats(pr).sort_stats("tottime").print_stats(14)
        else:
            J, _Jt = pde._jacobian([u, m, p], u=u)
        sync(); t["Jacobian (assembly, symmetry test)"] = time.perf_counter() - t0; t0 = time.perf_counter()
        prof_jac = PROF.read(); PROF.reset()
        _set_operator_once(solver, J)
        sync(); t["set operator"] = time.perf_counter() - t0; t0 = time.perf_counter()
        r.scale(-1.0)
        solver.solve(du, r)
        sync(); t["solve (with the AMG setup)"] = time.perf_counter() - t0; t0 = time.perf_counter()
        prof_sol = PROF.read(); PROF.reset()
        u.axpy(1.0, du)
        r = pde._residual([u, m, p], ADJOINT, ess=pde.bc0.ess); rn = r.norm("l2")
        sync(); t["update and second residual"] = time.perf_counter() - t0
        if rep:
            for k, v in t.items():
                rows.setdefault(k, []).append(v)
    tot = 0.0
    for k, v in rows.items():
        tot += float(np.median(v))
        say("  %-36s %8.1f ms" % (k, 1e3 * float(np.median(v))))
    say("  %-36s %8.1f ms   (%d iterations; bridge %s)" % ("sum", 1e3 * tot, int(solver.iterations), bridge.available()))
    from hippymfem.fem import kernel as K
    st = K.device().memory_stats() or {}
    say("  JAX arena %.0f MB (peak in use %.0f MB, in use %.0f MB, largest allocation %.0f MB); chunk setting %s"
        % (st.get("pool_bytes", 0) / 2 ** 20, st.get("peak_bytes_in_use", 0) / 2 ** 20,
           st.get("bytes_in_use", 0) / 2 ** 20, st.get("largest_alloc_size", 0) / 2 ** 20,
           os.environ.get("HIPPYMFEM_ELEMENT_CHUNK", "planner")))
    # what the card holds at the end, and what of it is hypre's pool
    try:
        from hippymfem.common import mfemconfig as cfg
        rt = None
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart.so" in line:
                    rt = ctypes.CDLL(line.split(None, 5)[-1].strip())
                    break
        free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
        rt.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))
        used = COMM.gather((total_b.value - free_b.value) / 2 ** 20, root=0)
        pool = cfg.HYPRE_POOL
        held = COMM.gather((pool.cached / 2 ** 20) if pool is not None and pool.installed else 0.0, root=0)
        ps = pool.stats() if pool is not None and pool.installed else {}
        say("  device memory in use: %.0f MB (largest rank %.0f MB); hypre's pool holds %.0f MB (largest %.0f MB), "
            "limits %s / %s MB, served %d of %d requests"
            % (used[0], max(used), held[0], max(held),
               "%.0f" % (pool.limit / 2 ** 20) if ps else "-", "%.0f" % (pool.keep / 2 ** 20) if ps and hasattr(pool, "keep") else "-",
               ps.get("from_pool", 0), ps.get("requests", 0)))
    except Exception as exc:                                         # noqa: BLE001
        say("  device memory: not read (%s)" % (exc,))
    say("      residual, driver: " + copies(prof_res))
    say("      Jacobian, driver: " + copies(prof_jac))
    say("      solve,    driver: " + copies(prof_sol) + "  amg_setup %.3f s" % prof_sol.get("amg_setup", (0, 0, 0))[1])
    if args.out and RANK == 0:
        rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N, "order": ORDER,
               "tdofs": ndof, "tree": HM, "tag": args.tag,
               "steps_ms": {k: 1e3 * float(np.median(v)) for k, v in rows.items()},
               "t_forward": tot, "iterations": int(solver.iterations),
               "jax_arena_mb": st.get("pool_bytes", 0) / 2 ** 20,
               "jax_peak_mb": st.get("peak_bytes_in_use", 0) / 2 ** 20,
               "driver": {"residual": prof_res, "jacobian": prof_jac, "solve": prof_sol},
               "env": {k: v for k, v in os.environ.items() if k.startswith("HIPPYMFEM_")}}
        try:
            rec["device_used_mb"] = list(used)
            rec["hypre_pool_held_mb"] = list(held)
            rec["hypre_pool"] = dict(ps, limit_mb=pool.limit / 2 ** 20, keep_mb=getattr(pool, "keep", 0) / 2 ** 20) if ps else {}
        except NameError:
            pass
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
    COMM.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

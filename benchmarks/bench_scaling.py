#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Strong-scaling profile: where a Newton-CG step loses parallel efficiency.

``bench_newton_device.py`` times the stages; this one takes a reduced-Hessian action
apart into the nine operations it is made of and records, for every Krylov solve, the
number of iterations it took.  That separates the two ways a rank count can cost
efficiency:

* **iteration count** -- BoomerAMG is a weaker preconditioner on more subdomains, so a
  solve takes more iterations.  Visible as ``it_*``.
* **time per iteration** -- launch latency, halo exchanges and the global reductions of
  CG, which do not shrink with the subdomain.  Visible as ``t_*`` divided by ``it_*``.

Run the same problem at 1, 2 and 4 devices and compare::

    for r in 1 2 4; do
      HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n $r tools/mpirun_pinned.sh \
        python benchmarks/bench_scaling.py --n 64 --device cuda \
        --out results/local/scaling_l40s_n64_r$r.json
    done

The problem is that of ``bench_newton_device.py``: same mesh, PDE, prior, observations
and solver settings, so the stage times are comparable with those records.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

import mfem.par as mfem                                              # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

from hippymfem.fem import kernel as kernel_mod                       # noqa: E402
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


def timed(fn, sync=None):
    """Time ``fn`` with a barrier on each side.

    ``sync`` is a vector whose norm is taken before the closing barrier: an
    MPI reduction has to have the local value on the host, so it completes the device
    work that ``fn`` queued.  Without it a device-side product is timed as the launch.
    """
    COMM.Barrier()
    t0 = time.perf_counter()
    out = fn()
    if sync is not None:
        sync.norm("l2")
    COMM.Barrier()
    return time.perf_counter() - t0, out


def iters(solver):
    """Krylov iterations of the last solve on a hIPPyMFEM solver (0 if direct)."""
    return int(getattr(solver, "iterations", 0) or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="Hessian actions to time (the median is reported)")
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"), help="mfem.Device kind; 'cpu' keeps hypre on the host")
    ap.add_argument("--amg-relax-type", type=int, default=-1, help="hypre relax type; 16 = Chebyshev")
    ap.add_argument("--amg-max-levels", type=int, default=-1, help="hypre BoomerAMG max levels; -1 = MFEM default (25)")
    ap.add_argument("--amg-agg-levels", type=int, default=-1, help="levels of aggressive coarsening; -1 = MFEM default (1)")
    ap.add_argument("--amg-theta", type=float, default=-1.0, help="strength threshold; -1 = MFEM default (0.25)")
    ap.add_argument("--tag", default="", help="label written into the record")
    ap.add_argument("--cg-tol", type=float, default=1e-12, help="relative tolerance of every Krylov solve")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))

    N, ORDER = args.n, args.order
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, %d elem/rank, kernels on %s (%s)"
        % (N, ORDER, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), COMM.size, pmesh.GetNE(),
           kernel_mod.device(), gpu_name()))

    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(), is_fwd_linear=True)
    amg = {}
    if args.amg_relax_type >= 0:
        amg["amg_relax_type"] = args.amg_relax_type
    if args.amg_max_levels > 0:
        amg["amg_max_levels"] = args.amg_max_levels
    if args.amg_agg_levels >= 0:
        amg["amg_agg_levels"] = args.amg_agg_levels
    if args.amg_theta >= 0.0:
        amg["amg_strength_threshold"] = args.amg_theta
    # the three solves the records were taken with; the adjoint keeps its default
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=args.cg_tol,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"),
                    **amg)
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
    t_fwd, _ = timed(lambda: model.solveFwd(x[STATE], x), x[STATE])
    it_fwd = iters(pde.solver)
    t_adj, _ = timed(lambda: model.solveAdj(x[ADJOINT], x), x[ADJOINT])
    it_adj = iters(pde.solver)
    g = model.generate_vector(PARAMETER)
    t_grad, _ = timed(lambda: model.evalGradientParameter(x, g), g)
    t_blocks_cold, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
    x[PARAMETER].axpy(1e-3, prior.mean)
    t_blocks, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))

    # ---- a reduced-Hessian action, operation by operation (hm.ReducedHessian.TrueHessian)
    H = hm.ReducedHessian(model)
    v = model.generate_vector(PARAMETER)
    hm.parRandom.normal(1.0, v)
    w = model.generate_vector(PARAMETER)
    H.mult(v, w)                                                   # warm: compile, set up
    rhs_fwd, rhs_adj, rhs_adj2 = (model.generate_vector(STATE), model.generate_vector(ADJOINT),
                                  model.generate_vector(ADJOINT))
    uhat, phat, yhelp = (model.generate_vector(STATE), model.generate_vector(ADJOINT),
                         model.generate_vector(PARAMETER))
    names = ["applyC", "solveFwdInc", "applyWuu", "applyWum", "solveAdjInc",
             "applyWmm", "applyCt", "applyWmu", "applyR"]
    runs, it_runs = [], []
    for _ in range(args.repeats):
        t = {}
        t["applyC"], _ = timed(lambda: model.applyC(v, rhs_fwd), rhs_fwd)
        t["solveFwdInc"], _ = timed(lambda: model.solveFwdIncremental(uhat, rhs_fwd), uhat)
        it_f = iters(pde.solver_fwd_inc)
        t["applyWuu"], _ = timed(lambda: model.applyWuu(uhat, rhs_adj), rhs_adj)
        t["applyWum"], _ = timed(lambda: model.applyWum(v, rhs_adj2), rhs_adj2)
        rhs_adj.axpy(-1.0, rhs_adj2)
        t["solveAdjInc"], _ = timed(lambda: model.solveAdjIncremental(phat, rhs_adj), phat)
        it_a = iters(pde.solver_adj_inc)
        t["applyWmm"], _ = timed(lambda: model.applyWmm(v, w), w)
        t["applyCt"], _ = timed(lambda: model.applyCt(phat, yhelp), yhelp)
        t["applyWmu"], _ = timed(lambda: model.applyWmu(uhat, yhelp), yhelp)
        t["applyR"], _ = timed(lambda: model.applyR(v, yhelp), yhelp)
        runs.append(t)
        it_runs.append((it_f, it_a))
    op = {k: float(np.median([r[k] for r in runs])) for k in names}
    it_fwd_inc = int(np.median([a for a, _ in it_runs]))
    it_adj_inc = int(np.median([b for _, b in it_runs]))
    t_action = sum(op.values())

    say("  forward %.3f s (%d it) | adjoint %.3f s (%d it) | gradient %.3f s | blocks %.3f s (cold %.3f s)"
        % (t_fwd, it_fwd, t_adj, it_adj, t_grad, t_blocks, t_blocks_cold))
    say("  Hessian action %.4f s = %s" % (t_action, " + ".join("%s %.4f" % (k, op[k]) for k in names)))
    say("  incremental solves: fwd %d it (%.2f ms/it), adj %d it (%.2f ms/it)"
        % (it_fwd_inc, 1e3 * op["solveFwdInc"] / max(it_fwd_inc, 1),
           it_adj_inc, 1e3 * op["solveAdjInc"] / max(it_adj_inc, 1)))
    solves = op["solveFwdInc"] + op["solveAdjInc"]
    say("  the two solves are %.1f %% of the action; the eight products and R are %.1f %%"
        % (100 * solves / t_action, 100 * (t_action - solves) / t_action))

    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N, "order": ORDER,
           "tdofs": int(Vu.GlobalTrueVSize()), "mdofs": int(Vm.GlobalTrueVSize()),
           "mfem_device": args.device, "amg_relax_type": args.amg_relax_type, "cg_tol": args.cg_tol,
           "amg_max_levels": args.amg_max_levels, "amg_agg_levels": args.amg_agg_levels,
           "amg_theta": args.amg_theta, "tag": args.tag,
           "t_fwd": t_fwd, "it_fwd": it_fwd, "t_adj": t_adj, "it_adj": it_adj, "t_grad": t_grad,
           "t_blocks": t_blocks, "t_blocks_cold": t_blocks_cold,
           "t_action": t_action, "op": op, "it_fwd_inc": it_fwd_inc, "it_adj_inc": it_adj_inc,
           "repeats": args.repeats}
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

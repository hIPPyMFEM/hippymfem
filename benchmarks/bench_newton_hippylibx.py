#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""hIPPYlibx (https://github.com/hIPPyMFEM/hippylibx) on the problem of
``bench_newton_device.py``, for the CPU comparison.

Same mesh (``n^3`` hexahedra, second-order state, first-order parameter), same PDE
(``exp(m) grad u . grad p``), same Dirichlet data (``u = z`` on the bottom and top
faces), same BiLaplacian prior (gamma 0.1, delta 0.5, Robin), a prior sample as the
true parameter, 200 pointwise observations at 1% noise, CG + BoomerAMG for every
solve (through PETSc here, through MFEM there, with the same BoomerAMG settings;
hIPPYlibx's linear forward solve is LU by construction and is redirected, see
:func:`_krylov_forward_solves`), and the same Newton-CG settings.
The two libraries draw different random numbers, so the *data* differ; the cost per
stage and per Newton step is what is compared, with the CG counts reported so a
difference in iteration counts is visible rather than folded into the time.

Run in an environment that has dolfinx and hIPPYlibx::

    python benchmarks/bench_newton_hippylibx.py --n 64 --steps 2
    mpirun -n 4 python benchmarks/bench_newton_hippylibx.py --n 64 --steps 2
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh
from petsc4py import PETSc

import hippylibx as hpx

if os.environ.get("PETSC_TRACEBACK"):
    # petsc4py raises a bare error code; this makes PETSc print its own stack and the
    # message (for an allocation failure, the size it asked for and where).
    PETSc.Sys.pushErrorHandler("traceback")

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def timed(fn):
    COMM.Barrier()
    t0 = time.perf_counter()
    out = fn()
    COMM.Barrier()
    return time.perf_counter() - t0, out


#: hypre BoomerAMG set up the way MFEM's ``HypreBoomerAMG`` sets it up by default
#: (``SetDefaultOptions``: HMIS coarsening, one level of aggressive coarsening,
#: extended+i interpolation with P_max 4, l1-scaled hybrid Gauss-Seidel, theta 0.25,
#: 25 levels).  PETSc's own defaults are Falgout coarsening with classical
#: interpolation and no aggressive coarsening, whose coarse operators fill in badly on
#: a 3D second-order problem: the 64^3 run failed its first solve with a hypre
#: allocation error at one rank and had not finished that solve in eight minutes at
#: four.  With these the two libraries run the same preconditioner.
MFEM_AMG = {
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_agg_nl": 1,
    "pc_hypre_boomeramg_agg_num_paths": 1,
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_P_max": 4,
    "pc_hypre_boomeramg_relax_type_all": "l1scaled-SOR/Jacobi",
    "pc_hypre_boomeramg_strong_threshold": 0.25,
    "pc_hypre_boomeramg_max_levels": 25,
}


def _krylov_forward_solves():
    """Make hIPPYlibx's linear forward solve use CG + BoomerAMG instead of LU.

    ``PDEVariationalProblem.solveFwd`` builds a dolfinx ``LinearProblem`` with
    ``petsc_options={"ksp_type": "preonly", "pc_type": "lu"}`` written into the
    options database under its own prefix, so the solver attached to ``pde.solver``
    serves only the adjoint and incremental solves.  A serial LU of the 64^3 matrix
    overflowed its own size arithmetic (``MatLUFactorSymbolic_SeqAIJ`` asked for
    1.8e19 bytes) and MUMPS at four ranks failed the same way, which is what the bare
    "error code 55" was.  This substitutes the options for that one problem; the rest
    of dolfinx is untouched.
    """
    from dolfinx.fem import petsc as fem_petsc

    base = fem_petsc.LinearProblem
    opts = {"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-12, "ksp_max_it": 2000}
    opts.update(MFEM_AMG)

    class KrylovLinearProblem(base):
        def __init__(self, *a, petsc_options=None, **kw):
            if kw.get("petsc_options_prefix") == "hippylibx_fwd":
                petsc_options = dict(opts)
            super().__init__(*a, petsc_options=petsc_options, **kw)

    fem_petsc.LinearProblem = KrylovLinearProblem


def amg_like_mfem(solver):
    """Apply :data:`MFEM_AMG` to a hippylibx PETSc solver wrapper (if it is hypre)."""
    ksp = solver.ksp()
    pc = ksp.getPC()
    if pc.getType() == "hypre":
        pc.setFromOptions()
    return solver


def krylov(rtol=1e-12, max_it=2000):
    s = hpx.PETScKrylovSolver(COMM, "cg", "hypre_amg")
    s.parameters["relative_tolerance"] = rtol
    s.parameters["maximum_iterations"] = max_it
    return amg_like_mfem(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--cg-max", type=int, default=25)
    ap.add_argument("--cg-tol", type=float, default=1e-6)
    ap.add_argument("--gn-iter", type=int, default=0)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--petsc-amg", action="store_true",
                    help="keep PETSc's default BoomerAMG settings instead of MFEM's")
    args = ap.parse_args()
    N, ORDER = args.n, args.order
    if not args.petsc_amg:
        opts = PETSc.Options()
        for k, v in MFEM_AMG.items():
            opts[k] = v
    _krylov_forward_solves()

    msh = dmesh.create_box(COMM, [np.zeros(3), np.ones(3)], [N, N, N],
                           dmesh.CellType.hexahedron)
    Vu = fem.functionspace(msh, ("Lagrange", ORDER))
    Vm = fem.functionspace(msh, ("Lagrange", 1))
    Vh = [Vu, Vm, Vu]
    ndof = [V.dofmap.index_map.size_global * V.dofmap.index_map_bs for V in (Vu, Vm)]
    ne_local = msh.topology.index_map(3).size_local
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, %d elem/rank"
        % (N, ORDER, ndof[0], ndof[1], COMM.size, ne_local))
    say("  hIPPYlibx %s, dolfinx %s, PETSc CG + BoomerAMG (%s settings), everything on the host"
        % (getattr(hpx, "__version__", "?"), __import__("dolfinx").__version__,
           "PETSc default" if args.petsc_amg else "MFEM's default"))

    def pde_varf(u, m, p):
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx

    u_bdr = fem.Function(Vu)
    u_bdr.interpolate(lambda x: x[2])
    u_bdr0 = fem.Function(Vu)
    u_bdr0.x.array[:] = 0.0
    facets = dmesh.locate_entities_boundary(
        msh, 2, lambda x: np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0))
    dofs = fem.locate_dofs_topological(Vu, 2, facets)
    bc = fem.dirichletbc(u_bdr, dofs)
    bc0 = fem.dirichletbc(u_bdr0, dofs)
    pde = hpx.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    pde.solver = krylov()
    pde.solver_fwd_inc = krylov()
    pde.solver_adj_inc = krylov()

    prior = hpx.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    # The prior's R-solve is a composite (two A-solves and an M-solve); the PETSc
    # solvers are the attributes, and only the hypre ones take the options.
    for name in ("Asolver", "Msolver", "Rsolver"):
        sol = getattr(prior, name, None)
        if sol is not None and hasattr(sol, "ksp"):
            amg_like_mfem(sol)
    noise = PETSc.Vec().create(COMM)
    prior.init_vector(noise, "noise")
    hpx.parRandom.normal(1.0, noise)
    mtrue = prior.mean.copy()
    prior.sample(noise, mtrue)

    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, args.ntargets) for _ in range(3)])
    B = hpx.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm(PETSc.NormType.INFINITY), 1e-30)
    hpx.parRandom.normal_perturb(nstd, data)
    misfit = hpx.DiscreteStateObservation(B, data, nstd ** 2)
    model = hpx.Model(pde, prior, misfit)

    # ---- the pieces of a Newton-CG iteration, at the prior mean
    x = [model.generate_vector(hpx.STATE), prior.mean.copy(), model.generate_vector(hpx.ADJOINT)]
    t_fwd, _ = timed(lambda: model.solveFwd(x[hpx.STATE], x))
    t_adj, _ = timed(lambda: model.solveAdj(x[hpx.ADJOINT], x))
    g = model.generate_vector(hpx.PARAMETER)
    t_grad, _ = timed(lambda: model.evalGradientParameter(x, g))
    t_hess, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
    x[hpx.PARAMETER].axpy(1e-3, prior.mean)
    t_hess_warm, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
    H = hpx.ReducedHessian(model, misfit_only=False)
    v = model.generate_vector(hpx.PARAMETER)
    hpx.parRandom.normal(1.0, v)
    w = model.generate_vector(hpx.PARAMETER)
    # The first apply is timed too: it pays whatever was deferred to first use (solver
    # setup, compilation), so the cold and the warm apply are both recorded.
    t_apply_cold, _ = timed(lambda: H.mult(v, w))
    t_apply, _ = timed(lambda: H.mult(v, w))
    say("  forward solve %.3f s | adjoint solve %.3f s | gradient %.3f s | Hessian blocks %.3f s "
        "(warm %.3f s) | reduced-Hessian apply %.3f s (cold %.3f s)"
        % (t_fwd, t_adj, t_grad, t_hess, t_hess_warm, t_apply, t_apply_cold))

    params = hpx.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = args.steps
    params["globalization"] = "LS"
    params["GN_iter"] = args.gn_iter
    params["cg_max_iter"] = args.cg_max
    params["cg_coarse_tolerance"] = args.cg_tol
    params["print_level"] = -1
    solver = hpx.ReducedSpaceNewtonCG(model, params)
    t_step, xs = timed(lambda: solver.solve([None, prior.mean.copy(), None]))
    diff = mtrue.copy()
    diff.axpy(-1.0, xs[hpx.PARAMETER])
    err = diff.norm() / max(mtrue.norm(), 1e-300)
    reasons = getattr(solver, "termination_reasons", None)
    reason = reasons[solver.reason] if reasons else str(getattr(solver, "reason", "?"))
    say("  %d Newton step(s) %.2f s  newton %d  cg %d  J %.8e  |grad| %.4e  err(m) %.8f  (%s)"
        % (args.steps, t_step, solver.it, solver.total_cg_iter, solver.final_cost,
           solver.final_grad_norm, err, reason))
    rec = {"library": "hippylibx", "amg": "petsc-default" if args.petsc_amg else "mfem-default", "host": platform.node(), "ranks": COMM.size, "n": N,
           "order": ORDER, "tdofs": ndof[0], "mdofs": ndof[1], "mfem_device": "host",
           "t_fwd": t_fwd, "t_adj": t_adj, "t_grad": t_grad, "t_hess_blocks": t_hess,
           "t_hess_blocks_warm": t_hess_warm, "t_hess_apply": t_apply, "t_hess_apply_cold": t_apply_cold, "steps": args.steps,
           "t_steps": t_step, "newton_it": solver.it, "cg_it": solver.total_cg_iter,
           "J": float(solver.final_cost), "gradnorm": float(solver.final_grad_norm),
           "err_m": float(err)}
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

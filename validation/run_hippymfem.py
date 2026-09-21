#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Solve the shared benchmark with hIPPyMFEM and write the results as JSON.

Run with the conda ``base`` interpreter (PyMFEM lives there)::

    python validation/run_hippymfem.py --out validation/out/mfem.json

If ``--data`` names an existing file the observations are read from it, so that
both libraries see the identical noise realization; otherwise the data is
generated and written there.
"""

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shared_case as sc                                     # noqa: E402
import hippymfem as hm                                       # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402
from hippymfem.common.linalg import operator_to_dense       # noqa: E402


def dense_spectrum(model, x, Vm, comm, nkeep=None, resolve=True):
    """Exact generalized spectrum of the data-misfit Hessian w.r.t. the prior.

    Densifies both operators by applying them to unit vectors, then solves the
    small dense problem.  Costs one Hessian application (two linear solves) per
    parameter dof, so it is only for the coarse validation case -- but it is
    deterministic and dof-ordering independent, which is exactly what is needed
    to compare two libraries.
    """
    import scipy.linalg as sla

    nkeep = nkeep or sc.N_DENSE_EIG
    n = Vm.GlobalTrueVSize()
    if resolve:                       # make the state and adjoint match x[PARAMETER]
        model.solveFwd(x[STATE], x)
        model.solveAdj(x[ADJOINT], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    Hd = operator_to_dense(Hmisfit, n, comm)
    Rd = operator_to_dense(model.prior.R, n, comm)
    Hd = 0.5 * (Hd + Hd.T)
    Rd = 0.5 * (Rd + Rd.T)
    vals = sla.eigh(Hd, Rd, eigvals_only=True)
    return np.sort(vals)[::-1][:nkeep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation/out/mfem.json")
    ap.add_argument("--data", default="validation/out/data.json")
    ap.add_argument("--nx", type=int, default=sc.NX)
    ap.add_argument("--no-dense", action="store_true",
                    help="skip the dense spectrum (it is the slow part)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    mfem.Hypre.Init()

    # ---------------------------------------------------------------- mesh
    mesh_path = "/tmp/_hippymfem_validation_n%d.mesh" % args.nx
    if rank == 0:
        sc.write_mfem_mesh(mesh_path, args.nx)
    comm.Barrier()
    pmesh = mfem.ParMesh(comm, mfem.Mesh(mesh_path, 1, 1))

    Vu = hm.FunctionSpace.H1(pmesh, sc.ORDER_STATE)
    Vm = hm.FunctionSpace.H1(pmesh, sc.ORDER_PARAM)
    Vh = [Vu, Vm, Vu]
    ndofs = [Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), Vu.GlobalTrueVSize()]

    # ------------------------------------------------------------ forward PDE
    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    bc = hm.DirichletBC(Vu, sc.u_boundary, bdr_attributes=[1, 3])
    bc0 = bc.homogeneous()
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True,
                                   quadrature_degree=sc.QUADRATURE_DEGREE)
    for attr in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if comm.size == 1:
            setattr(pde, attr, hm.LUSolver(comm))
        else:
            s = hm.KrylovSolver(comm, "cg", "amg")
            s.parameters["rel_tolerance"] = 1e-14
            s.parameters["max_iter"] = 3000
            setattr(pde, attr, s)

    # ------------------------------------------------------------------ prior
    prior = hm.BiLaplacianPrior(
        Vm, sc.GAMMA, sc.DELTA, Theta=sc.theta_matrix(), robin_bc=sc.ROBIN_BC,
        solver_type="lu" if comm.size == 1 else "krylov",
        quadrature_degree=2 * sc.ORDER_PARAM,
    )

    # ------------------------------------------------------------ observations
    targets = sc.targets()
    B = hm.assemblePointwiseObservation(Vu, targets)

    mtrue = Vm.project(sc.m_true)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    clean = B.createVecLeft()
    B.mult(utrue, clean)
    clean_full = B.gather(clean)

    if os.path.exists(args.data):
        rec = sc.read_json(args.data)
        data_full = np.asarray(rec["data"], dtype=float)
        noise_std = float(rec["noise_std"])
        if rank == 0:
            print("read data from %s" % args.data, flush=True)
    else:
        noise_std = sc.REL_NOISE * float(np.abs(clean_full).max())
        rng = np.random.default_rng(12345)
        data_full = clean_full + noise_std * rng.standard_normal(clean_full.size)
        if rank == 0:
            sc.write_json(args.data, {
                "data": data_full, "noise_std": noise_std,
                "clean_mfem": clean_full, "targets": targets,
            })
            print("wrote data to %s" % args.data, flush=True)
    data = B.scatter(data_full)
    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    model = hm.Model(pde, prior, misfit)

    out = {
        "library": "hippymfem",
        "nranks": comm.size,
        "nx": args.nx,
        "ndofs": {"state": ndofs[0], "parameter": ndofs[1], "adjoint": ndofs[2]},
        "noise_std": noise_std,
        "clean_data": clean_full,
    }

    # ------------------------------------------------- cost and gradient at m0
    m0 = Vm.project(sc.m_init)
    x0 = model.generate_vector()
    x0[PARAMETER] = m0
    model.solveFwd(x0[STATE], x0)
    model.solveAdj(x0[ADJOINT], x0)
    c_tot, c_reg, c_mis = model.cost(x0)
    g = Vm.vector()
    gnorm = model.evalGradientParameter(x0, g)
    out["at_m0"] = {"cost_total": c_tot, "cost_reg": c_reg, "cost_misfit": c_mis,
                    "grad_norm_Rinv": gnorm}

    # a fixed, library-independent directional derivative: the gradient paired
    # with the analytic direction m_true, so no random vector has to be shared
    dm = Vm.project(sc.m_true)
    out["at_m0"]["grad_dot_mtrue"] = g.inner(dm)

    # Hessian action in the same fixed direction
    model.setPointForHessianEvaluations(x0, gauss_newton_approx=False)
    H = hm.ReducedHessian(model, misfit_only=False)
    Hd = Vm.vector()
    H.mult(dm, Hd)
    out["at_m0"]["mtrue_H_mtrue"] = Hd.inner(dm)
    Hgn = hm.ReducedHessian(
        hm.Model(pde, prior, misfit), misfit_only=False)
    model.setPointForHessianEvaluations(x0, gauss_newton_approx=True)
    Hgn = hm.ReducedHessian(model, misfit_only=False)
    Hd.zero()
    Hgn.mult(dm, Hd)
    out["at_m0"]["mtrue_HGN_mtrue"] = Hd.inner(dm)

    # prior diagnostics
    out["prior"] = {
        "cost_mtrue": prior.cost(mtrue),
        "trace_exact": prior.trace("Exact"),
    }

    # --------------------------------------------------------------- MAP point
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = sc.NEWTON_REL_TOL
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = sc.NEWTON_MAX_ITER
    params["globalization"] = "LS"
    params["GN_iter"] = sc.GN_ITER
    params["print_level"] = 0 if rank == 0 else -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, prior.mean.copy(), None])
    c_tot, c_reg, c_mis = model.cost(x)
    out["map"] = {
        "converged": bool(solver.converged),
        "reason": solver.termination_reasons[solver.reason],
        "newton_iterations": solver.it,
        "total_cg_iterations": solver.total_cg_iter,
        "final_cost": c_tot, "final_reg": c_reg, "final_misfit": c_mis,
        "final_grad_norm": solver.final_grad_norm,
    }

    # --------------------------------------------------- Laplace approximation
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k, p = sc.N_EIG, sc.N_OVERSAMPLE
    Omega = hm.MultiVector(x[PARAMETER], k + p)
    for j, fn in enumerate(sc.sketch_functions(k + p)):
        Omega[j].assign(Vm.project(fn))      # deterministic, library-independent
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=sc.N_POWER)
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    out["eigenvalues"] = d
    out["kl_from_prior"] = post.klDistanceFromPrior()
    pv, prv, corr = post.pointwise_variance(method="Exact")
    tr_post, tr_pr, tr_corr = post.trace(method="Exact")
    out["traces"] = {"posterior": tr_post, "prior": tr_pr, "correction": tr_corr}

    # ----------------------------------------------- fields at shared points
    pts = sc.sample_points()
    Bm = hm.assemblePointwiseObservation(Vm, pts)
    Bu = hm.assemblePointwiseObservation(Vu, pts)

    def sample_m(v):
        o = Bm.createVecLeft()
        Bm.mult(v, o)
        return Bm.gather(o)

    def sample_u(v):
        o = Bu.createVecLeft()
        Bu.mult(v, o)
        return Bu.gather(o)

    out["fields"] = {
        "sample_points": pts,
        "m_true": sample_m(mtrue),
        "m_map": sample_m(x[PARAMETER]),
        "u_map": sample_u(x[STATE]),
        "u_true": sample_u(utrue),
        "post_variance": sample_m(pv),
        "prior_variance": sample_m(prv),
    }

    # ------------------------------------------------ exact (dense) spectrum
    # Tests the Hessian operator itself, with no randomized solver in the way.
    if not args.no_dense:
        # At the fixed analytic point m0 this is a clean test of the Hessian
        # operator: it does not depend on where either optimizer happened to
        # stop.  The MAP-point spectrum is reported too, but it inherits the
        # ~1e-8 difference between the two MAP points.
        out["dense_eigenvalues_at_m0"] = dense_spectrum(model, x0, Vm, comm)
        out["dense_eigenvalues"] = dense_spectrum(model, x, Vm, comm)

    if rank == 0:
        sc.write_json(args.out, out)
        print("\nwrote %s" % args.out, flush=True)
        print("  MAP: cost %.10e, misfit %.10e, reg %.10e"
              % (out["map"]["final_cost"], out["map"]["final_misfit"],
                 out["map"]["final_reg"]), flush=True)
        print("  leading eigenvalues: %s"
              % np.array2string(np.asarray(d[:5]), precision=6), flush=True)


if __name__ == "__main__":
    main()

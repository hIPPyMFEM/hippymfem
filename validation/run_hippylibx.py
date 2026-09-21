#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Solve the shared benchmark with hIPPYlibx and write the results as JSON.

Run with an interpreter that has dolfinx and hIPPYlibx::

    python validation/run_hippylibx.py --out validation/out/hippylibx.json

The mesh, the observation targets, the true parameter, the data and the
quadrature degree all come from ``shared_case``, so this solves the *same*
discrete problem as ``run_hippymfem.py`` rather than a nearby one.
"""

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

import basix
import ufl
from dolfinx import fem, mesh as dmesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get("HIPPYLIBX_DIR"):            # a source checkout of hIPPYlibx
    sys.path.insert(0, os.environ["HIPPYLIBX_DIR"])

import shared_case as sc                    # noqa: E402
from hippylibx import *                     # noqa: E402,F401,F403
import hippylibx as hpx                     # noqa: E402


def build_mesh(comm, nx):
    """The shared mesh, built from the same vertex/cell arrays as the MFEM side."""
    verts, cells, _bdr = sc.unit_square_triangles(nx)
    dom = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    return dmesh.create_mesh(comm, cells, dom, verts)


def top_bottom(x):
    return np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


def facet_dofs(V, marker):
    fdim = V.mesh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(V.mesh, fdim, marker)
    return fem.locate_dofs_topological(V, fdim, facets)


def sample_at(V, pts, vec):
    """Evaluate a dof vector at ``pts`` using hIPPYlibx's own observation operator."""
    B = hpx.assemblePointwiseObservation(V, pts)
    out = B.createVecLeft()
    B.mult(vec, out)
    loc = out.array
    gathered = V.mesh.comm.allgather(loc)
    return np.concatenate(gathered)


def operator_to_dense(op, n, comm):
    """Dense matrix of a matrix-free operator, by applying it to unit vectors."""
    cols = []
    for j in range(n):
        e = _unit_vector(op, n, j, comm)
        y = _like(op, comm)
        op.mult(e, y)
        cols.append(np.concatenate(comm.allgather(y.array)))
    return np.array(cols).T


def _like(op, comm):
    from petsc4py import PETSc

    v = PETSc.Vec().create(comm=comm)
    op.init_vector(v, 0)
    v.set(0.0)
    return v


def _unit_vector(op, n, j, comm):
    v = _like(op, comm)
    lo, hi = v.getOwnershipRange()
    if lo <= j < hi:
        v.array[j - lo] = 1.0
    v.assemble()
    return v


def dense_spectrum(model, x, comm, nkeep, resolve=True):
    """Exact generalized spectrum of the data-misfit Hessian w.r.t. the prior."""
    import scipy.linalg as sla

    n = model.prior.M.getSize()[0]
    if resolve:
        model.solveFwd(x[hpx.STATE], x)
        model.solveAdj(x[hpx.ADJOINT], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hpx.ReducedHessian(model, misfit_only=True)
    Hd = operator_to_dense(Hmisfit, n, comm)
    Rd = operator_to_dense(model.prior.R, n, comm)
    Hd = 0.5 * (Hd + Hd.T)
    Rd = 0.5 * (Rd + Rd.T)
    vals = sla.eigh(Hd, Rd, eigvals_only=True)
    return np.sort(vals)[::-1][:nkeep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation/out/hippylibx.json")
    ap.add_argument("--data", default="validation/out/data.json")
    ap.add_argument("--nx", type=int, default=sc.NX)
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    msh = build_mesh(comm, args.nx)

    Vh2 = fem.functionspace(msh, ("Lagrange", sc.ORDER_STATE))
    Vh1 = fem.functionspace(msh, ("Lagrange", sc.ORDER_PARAM))
    Vh = [Vh2, Vh1, Vh2]
    ndofs = [V.dofmap.index_map.size_global * V.dofmap.index_map_bs for V in Vh]

    # ------------------------------------------------------------ forward PDE
    dx_q = ufl.dx(metadata={"quadrature_degree": sc.QUADRATURE_DEGREE})

    def pde_varf(u, m, p):
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * dx_q

    u_bdr = fem.Function(Vh[hpx.STATE])
    u_bdr.interpolate(lambda x: x[1])
    u_bdr0 = fem.Function(Vh[hpx.STATE])
    u_bdr0.x.array[:] = 0.0
    dofs = facet_dofs(Vh[hpx.STATE], top_bottom)
    bc = fem.dirichletbc(u_bdr, dofs)
    bc0 = fem.dirichletbc(u_bdr0, dofs)

    pde = hpx.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    pde.solver = hpx.PETScLUSolver(msh.comm)
    pde.solver_fwd_inc = hpx.PETScLUSolver(msh.comm)
    pde.solver_adj_inc = hpx.PETScLUSolver(msh.comm)

    # ------------------------------------------------------------------ prior
    Th = sc.theta_matrix()
    anis = ufl.as_matrix(((Th[0, 0], Th[0, 1]), (Th[1, 0], Th[1, 1])))
    prior = hpx.BiLaplacianPrior(Vh[hpx.PARAMETER], sc.GAMMA, sc.DELTA, anis,
                                 robin_bc=sc.ROBIN_BC, solver_type="lu")

    # ------------------------------------------------------------ observations
    targets = sc.targets()
    B = hpx.assemblePointwiseObservation(Vh[hpx.STATE], targets)

    mtrue_f = fem.Function(Vh[hpx.PARAMETER])
    mtrue_f.interpolate(lambda x: sc.m_true(x))
    mtrue = prior.M.createVecRight()
    owned = (Vh[hpx.PARAMETER].dofmap.index_map.size_local
             * Vh[hpx.PARAMETER].dofmap.index_map_bs)
    mtrue.array[:] = mtrue_f.x.array[:owned]
    mtrue.assemble()

    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    clean = B.createVecLeft()
    B.mult(utrue, clean)
    clean_full = np.concatenate(comm.allgather(clean.array))

    rec = sc.read_json(args.data)
    data_full = np.asarray(rec["data"], dtype=float)
    noise_std = float(rec["noise_std"])
    data = B.createVecLeft()
    lo, hi = data.getOwnershipRange()
    data.array[:] = data_full[lo:hi]
    data.assemble()
    misfit = hpx.DiscreteStateObservation(B, data, noise_std ** 2)
    model = hpx.Model(pde, prior, misfit)

    out = {
        "library": "hippylibx",
        "nranks": comm.size,
        "nx": args.nx,
        "ndofs": {"state": ndofs[0], "parameter": ndofs[1], "adjoint": ndofs[2]},
        "noise_std": noise_std,
        "clean_data": clean_full,
    }

    # ------------------------------------------------- cost and gradient at m0
    m0_f = fem.Function(Vh[hpx.PARAMETER])
    m0_f.interpolate(lambda x: sc.m_init(x))
    m0 = model.generate_vector(hpx.PARAMETER)
    m0.array[:] = m0_f.x.array[:owned]
    m0.assemble()

    x0 = model.generate_vector()
    x0[hpx.PARAMETER] = m0
    model.solveFwd(x0[hpx.STATE], x0)
    model.solveAdj(x0[hpx.ADJOINT], x0)
    c_tot, c_reg, c_mis = model.cost(x0)
    g = model.generate_vector(hpx.PARAMETER)
    gnorm = model.evalGradientParameter(x0, g)
    out["at_m0"] = {"cost_total": c_tot, "cost_reg": c_reg, "cost_misfit": c_mis,
                    "grad_norm_Rinv": gnorm,
                    "grad_dot_mtrue": float(g.dot(mtrue))}

    model.setPointForHessianEvaluations(x0, gauss_newton_approx=False)
    H = hpx.ReducedHessian(model, misfit_only=False)
    Hd = model.generate_vector(hpx.PARAMETER)
    H.mult(mtrue, Hd)
    out["at_m0"]["mtrue_H_mtrue"] = float(Hd.dot(mtrue))
    model.setPointForHessianEvaluations(x0, gauss_newton_approx=True)
    Hgn = hpx.ReducedHessian(model, misfit_only=False)
    Hd.set(0.0)
    Hgn.mult(mtrue, Hd)
    out["at_m0"]["mtrue_HGN_mtrue"] = float(Hd.dot(mtrue))

    out["prior"] = {"cost_mtrue": prior.cost(mtrue),
                    "trace_exact": float(prior.trace("Exact"))}

    # --------------------------------------------------------------- MAP point
    params = hpx.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = sc.NEWTON_REL_TOL
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = sc.NEWTON_MAX_ITER
    params["globalization"] = "LS"
    params["GN_iter"] = sc.GN_ITER
    params["print_level"] = 0 if rank == 0 else -1
    solver = hpx.ReducedSpaceNewtonCG(model, params)
    m_start = prior.mean.copy()
    x = solver.solve([None, m_start, None])
    c_tot, c_reg, c_mis = model.cost(x)
    out["map"] = {
        "converged": bool(solver.converged),
        "newton_iterations": solver.it,
        "total_cg_iterations": solver.total_cg_iter,
        "final_cost": c_tot, "final_reg": c_reg, "final_misfit": c_mis,
        "final_grad_norm": float(solver.final_grad_norm),
    }

    # --------------------------------------------------- Laplace approximation
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hpx.ReducedHessian(model, misfit_only=True)
    k, p = sc.N_EIG, sc.N_OVERSAMPLE
    Omega = hpx.MultiVector(x[hpx.PARAMETER], k + p)
    sketch = fem.Function(Vh[hpx.PARAMETER])
    for j, fn in enumerate(sc.sketch_functions(k + p)):
        sketch.interpolate(lambda xx, fn=fn: fn(xx))
        Omega[j].array[:] = sketch.x.array[:owned]   # deterministic sketch
        Omega[j].assemble()
    d, U = hpx.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=sc.N_POWER)
    post = hpx.GaussianLRPosterior(prior, d, U)
    post.mean = x[hpx.PARAMETER]
    out["eigenvalues"] = np.asarray(d)
    out["kl_from_prior"] = float(post.klDistanceFromPrior())
    pv, prv, corr = post.pointwise_variance(method="Exact")
    tr_post, tr_pr, tr_corr = post.trace(method="Exact")
    out["traces"] = {"posterior": float(tr_post), "prior": float(tr_pr),
                     "correction": float(tr_corr)}

    # ----------------------------------------------- fields at shared points
    pts = sc.sample_points()
    out["fields"] = {
        "sample_points": pts,
        "m_true": sample_at(Vh[hpx.PARAMETER], pts, mtrue),
        "m_map": sample_at(Vh[hpx.PARAMETER], pts, x[hpx.PARAMETER]),
        "u_map": sample_at(Vh[hpx.STATE], pts, x[hpx.STATE]),
        "u_true": sample_at(Vh[hpx.STATE], pts, utrue),
        "post_variance": sample_at(Vh[hpx.PARAMETER], pts, pv),
        "prior_variance": sample_at(Vh[hpx.PARAMETER], pts, prv),
    }

    if not args.no_dense:
        out["dense_eigenvalues_at_m0"] = dense_spectrum(model, x0, comm,
                                                        sc.N_DENSE_EIG)
        out["dense_eigenvalues"] = dense_spectrum(model, x, comm, sc.N_DENSE_EIG)

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

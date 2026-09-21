#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Subsurface flow: infer a log-permeability field from pressure observations.

The benchmark of hIPPYlib's subsurface-flow example, written with hIPPyMFEM.  Infer :math:`m` in

.. math:: -\nabla\cdot(e^{m}\nabla u) = 0 \quad\text{in }\Omega=(0,1)^2,
          \qquad u = y \text{ on the top and bottom edges},

from pointwise observations of :math:`u`, with an anisotropic Matern
(bi-Laplacian) prior.  Computes the MAP point by inexact Newton-CG and the
Laplace approximation of the posterior by a randomized generalized eigensolver,
then writes the fields for ParaView and samples from prior and posterior.

Run::

    python applications/poisson/model_subsurf.py
    mpirun -n 4 python applications/poisson/model_subsurf.py
"""

import argparse
import math
import os
import sys

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import hippymfem as hm                                              # noqa: E402
from hippymfem.modeling.variables import PARAMETER, STATE  # noqa: E402

SEP = "\n" + "#" * 78 + "\n"


def anisotropic_tensor(theta0=2.0, theta1=0.5, alpha=math.pi / 4):
    """Constant anisotropic diffusion tensor for the prior."""
    sa, ca = math.sin(alpha), math.cos(alpha)
    return np.array([
        [theta0 * sa * sa + theta1 * ca * ca, (theta0 - theta1) * sa * ca],
        [(theta0 - theta1) * sa * ca, theta0 * ca * ca + theta1 * sa * sa],
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--order", type=int, default=2, help="state polynomial degree")
    ap.add_argument("--ntargets", type=int, default=50)
    ap.add_argument("--rel-noise", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--nsamples", type=int, default=5,
                    help="prior/posterior samples to write")
    ap.add_argument("--neig", type=int, default=50,
                    help="eigenpairs for the Laplace approximation")
    ap.add_argument("--verify", action="store_true",
                    help="run the finite-difference gradient/Hessian checks")
    ap.add_argument("--out", default="applications/poisson/results")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    mfem.Hypre.Init()

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    # ------------------------------------------------- mesh and spaces
    pmesh = mfem.ParMesh(comm, mfem.Mesh.MakeCartesian2D(
        args.nx, args.ny, mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, args.order)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    log(SEP + "Mesh and finite element spaces" + SEP)
    log("Number of dofs: STATE=%d, PARAMETER=%d, ADJOINT=%d"
        % (Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), Vu.GlobalTrueVSize()))
    log("MPI ranks: %d" % comm.size)

    # ------------------------------------------------- forward problem
    def pde_varf(u, m, p, x):
        """Weak residual density of -div(exp(m) grad u) = 0."""
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    # MakeCartesian2D boundary attributes: 1 bottom, 2 right, 3 top, 4 left
    bc = hm.DirichletBC(Vu, lambda x: x[1], bdr_attributes=[1, 3])
    bc0 = bc.homogeneous()
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Chosen by problem size rather than by rank count: an exact solve is worth
    # having while it is affordable, and a one-rank run on a fine mesh is exactly
    # where it stops being so.
    pde.set_solvers(hm.auto_solver, Vu, comm, rel_tolerance=1e-13, max_iter=3000)
    log("Linear solver: %s" % type(pde.solver).__name__)

    # ------------------------------------------------- prior
    prior = hm.BiLaplacianPrior(
        Vm, args.gamma, args.delta, Theta=anisotropic_tensor(), robin_bc=True,
        solver_type="lu" if comm.size == 1 else "krylov")
    log(SEP + "Prior" + SEP)
    log("Matern (bi-Laplacian): gamma=%g, delta=%g, anisotropic, Robin bc"
        % (args.gamma, args.delta))
    log("correlation length ~ sqrt(gamma/delta) = %.3f"
        % math.sqrt(args.gamma / args.delta))

    # ------------------------------------------------- synthetic truth and data
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)

    rng = np.random.default_rng(1)
    targets = np.column_stack((rng.uniform(0.1, 0.9, args.ntargets),
                               rng.uniform(0.1, 0.5, args.ntargets)))
    B = hm.assemblePointwiseObservation(Vu, targets)
    log(SEP + "Synthetic observations" + SEP)
    log("Number of observation points: %d" % args.ntargets)

    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    noise_std = args.rel_noise * max(data.norm("linf"), 1e-30)
    B.perturb(data, noise_std)   # keyed on target index: rank-count independent
    log("Relative noise: %g  (std %.4e)" % (args.rel_noise, noise_std))

    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    model = hm.Model(pde, prior, misfit)

    # ------------------------------------------------- optional FD checks
    if args.verify:
        log(SEP + "Finite-difference gradient and Hessian checks" + SEP)
        m0 = Vm.project(lambda x: np.sin(x[0]))
        eps, eg, eH = hm.modelVerify(model, m0, verbose=(rank == 0))
        log("observed slopes: gradient %.3f, Hessian %.3f"
            % (hm.best_slope(eps, eg), hm.best_slope(eps, eH)))

    # ------------------------------------------------- MAP point
    log(SEP + "MAP point (inexact Newton-CG)" + SEP)
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = 30
    params["globalization"] = "LS"
    params["GN_iter"] = 5
    params["print_level"] = 0 if rank == 0 else -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, prior.mean.copy(), None])

    log("\n%s" % solver.termination_reasons[solver.reason])
    log("Newton iterations: %d, total CG iterations: %d"
        % (solver.it, solver.total_cg_iter))
    log("Final cost %.6e, gradient norm %.6e"
        % (solver.final_cost, solver.final_grad_norm))
    err = (mtrue.copy().axpy(-1.0, x[PARAMETER]).norm("l2")
           / max(mtrue.norm("l2"), 1e-300))
    log("Relative error in the parameter: %.4f" % err)

    # ------------------------------------------------- Laplace approximation
    log(SEP + "Laplace approximation of the posterior" + SEP)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k, p = args.neig, 20
    Omega = hm.MultiVector(x[PARAMETER], k + p)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1)
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    log("Computed %d eigenpairs; d[0]=%.4e, d[%d]=%.4e"
        % (k, d[0], k - 1, d[-1]))
    log("KL divergence of the Laplace posterior from the prior: %.6e"
        % post.klDistanceFromPrior())
    tr_post, tr_pr, tr_corr = post.trace(method="Randomized", r=200)
    log("Traces: posterior %.6e, prior %.6e, correction %.6e"
        % (tr_post, tr_pr, tr_corr))
    pv, prv, corr = post.pointwise_variance(method="Randomized", r=200)

    # ------------------------------------------------- output
    os.makedirs(args.out, exist_ok=True)
    log(SEP + "Output" + SEP)
    fields = {
        "m_true": (Vm, mtrue),
        "m_map": (Vm, x[PARAMETER]),
        "u_true": (Vu, utrue),
        "u_map": (Vu, x[STATE]),
        "prior_variance": (Vm, prv),
        "post_variance": (Vm, pv),
    }
    hm.write_paraview(os.path.join(args.out, "subsurf"), pmesh, fields)
    log("wrote %s" % os.path.join(args.out, "subsurf"))

    if args.nsamples > 0:
        s_pr, s_po = Vm.vector(), Vm.vector()
        samples = {}
        for i in range(args.nsamples):
            prior.sample_noise(1.0, noise)
            post.sample(noise, s_pr, s_po)
            samples["prior_sample_%d" % i] = (Vm, s_pr.copy())
            samples["post_sample_%d" % i] = (Vm, s_po.copy())
        hm.write_paraview(os.path.join(args.out, "samples"), pmesh, samples)
        log("wrote %d prior and posterior samples" % args.nsamples)

    hm.exportPointwiseObservation(
        B, data, os.path.join(args.out, "observations.csv"), comm)
    log("wrote observations.csv")
    log("\nSolver call counts: %s" % pde.n_calls)


if __name__ == "__main__":
    main()

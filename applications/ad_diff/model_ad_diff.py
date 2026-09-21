#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Advection-diffusion: infer the initial condition from later observations.

The benchmark of hIPPYlib's ``ad_diff`` example.  The state solves

.. math:: u_t + v\cdot\nabla u - \kappa\,\Delta u = 0,\qquad u(0) = m,

and the parameter is the initial condition, observed only at later times.  This
exercises the whole time-dependent machinery: a forward march, an adjoint march
backwards in time, and a space-time reduced Hessian.

Run::

    python applications/ad_diff/model_ad_diff.py
    mpirun -n 4 python applications/ad_diff/model_ad_diff.py
"""

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import hippymfem as hm                                              # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE  # noqa: E402

SEP = "\n" + "#" * 78 + "\n"


class InitialConditionProblem(hm.PDEProblem):
    r"""Presents a time-dependent problem as ``m \mapsto`` trajectory with ``u(0) = m``.

    The parameter enters only through the initial condition, so the residual has
    no explicit ``m`` dependence and the whole gradient is
    :math:`B_1^{\!\top}p_1`, where :math:`B_1 = \partial_{u_0}\partial_p r_1`
    couples the first step to the initial level.  That block is also the ``C``
    block of the reduced Hessian.
    """

    def __init__(self, pde):
        self.pde = pde
        self.Vh = pde.Vh
        self.times = pde.times

    def generate_state(self):
        return self.pde.generate_state()

    def generate_parameter(self):
        return self.pde.generate_parameter()

    def generate_adjoint(self):
        return self.pde.generate_adjoint()

    def init_parameter(self, m):
        return m

    def solveFwd(self, out, x):
        self.pde.u0 = x[PARAMETER].copy()
        self.pde.bc.zero(self.pde.u0)
        return self.pde.solveFwd(out, x)

    def solveAdj(self, out, x, adj_rhs):
        return self.pde.solveAdj(out, x, adj_rhs)

    def _B1(self, x):
        t1, t0 = self.times[1], self.times[0]
        return self.pde._step_block(
            3, 1, x[STATE].view(t1), x[STATE].view(t0), x[PARAMETER],
            x[ADJOINT].view(t1) if x[ADJOINT] is not None
            else self.pde.generate_static_adjoint(),
            t1, test_ess=self.pde.bc0.ess_tdof)

    def evalGradientParameter(self, x, out):
        out.zero()
        B = self._B1(x)
        B.MultTranspose(x[ADJOINT].view(self.times[1]).hypre, out.hypre)
        self.pde.bc0.zero(out)
        self._keep = B
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        return self.pde.setLinearizationPoint(x, gauss_newton_approx)

    def solveIncremental(self, out, rhs, is_adj):
        return self.pde.solveIncremental(out, rhs, is_adj)

    def apply_ij(self, i, j, dir, out):
        if i == ADJOINT and j == PARAMETER:
            out.zero()
            self.pde._blocks(1)["B"].Mult(dir.hypre,
                                          out.view(self.times[1]).hypre)
            return out
        if i == PARAMETER and j == ADJOINT:
            out.zero()
            self.pde._blocks(1)["B"].MultTranspose(
                dir.view(self.times[1]).hypre, out.hypre)
            self.pde.bc0.zero(out)
            return out
        return self.pde.apply_ij(i, j, dir, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--nt", type=int, default=16)
    ap.add_argument("--t-final", type=float, default=0.4)
    ap.add_argument("--kappa", type=float, default=0.02)
    ap.add_argument("--ntargets", type=int, default=100)
    ap.add_argument("--order", type=int, default=1)
    # The prior is specified by what it means -- marginal variance and
    # correlation length -- rather than by the raw bi-Laplacian coefficients.
    # Getting this wrong is not a cosmetic matter: a prior whose pointwise
    # standard deviation is 30x the signal amplitude leaves the Newton system
    # with a condition number around 1e9, and CG cannot solve it.
    ap.add_argument("--sigma2", type=float, default=0.25,
                    help="prior marginal variance")
    ap.add_argument("--rho", type=float, default=0.15,
                    help="prior correlation length")
    ap.add_argument("--rel-noise", type=float, default=0.01)
    ap.add_argument("--neig", type=int, default=30)
    ap.add_argument("--cg-max-iter", type=int, default=400)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default="applications/ad_diff/results")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    mfem.Hypre.Init()

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    pmesh = mfem.ParMesh(comm, mfem.Mesh.MakeCartesian2D(
        args.nx, args.nx, mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, args.order)
    Vh = [Vu, Vu, Vu]               # the parameter is the initial state
    dt = args.t_final / args.nt
    vel = jnp.array([1.0, 0.4])
    kappa = args.kappa

    def varf(u, u_old, m, p, x, t, dt_):
        """Implicit Euler for u_t + v.grad u - kappa lap u = 0."""
        return ((u.val - u_old.val) / dt_ * p.val
                + kappa * hm.inner(u.grad, p.grad)
                + jnp.dot(vel, u.grad) * p.val)

    bc = hm.DirichletBC(Vu, 0.0, bdr_attributes="all")
    pde = hm.TimeDependentPDEVariationalProblem(
        Vh, varf, bc, bc.homogeneous(), Vu.vector(), 0.0, args.t_final, dt,
        is_fwd_linear=True, quadrature_degree=2 * args.order + 2)
    # By problem size, not rank count: see the note in poisson/model_subsurf.py.
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        s = hm.auto_solver(Vu, comm, method="gmres")
        if isinstance(s, hm.KrylovSolver):
            s.parameters["rel_tolerance"] = 1e-13
        setattr(pde, a, s)
    log("Linear solver: %s" % type(pde.solver).__name__)

    prob = InitialConditionProblem(pde)
    log(SEP + "Time-dependent advection-diffusion" + SEP)
    log("dofs per level: %d ; time levels: %d ; dt = %g"
        % (Vu.GlobalTrueVSize(), pde.nt, dt))
    log("kappa = %g, velocity = %s" % (kappa, np.asarray(vel)))

    # ------------------------------------------------- observations
    rng = np.random.default_rng(1)
    targets = rng.uniform(0.1, 0.9, size=(args.ntargets, 2))
    B = hm.assemblePointwiseObservation(Vu, targets)
    misfits = [None] * pde.nt
    for k in range(1, pde.nt):
        misfits[k] = hm.DiscreteStateObservation(B, B.createVecLeft(), 1.0)
    misfit = hm.MisfitTD(misfits, pde.times)
    log("Observations: %d points at each of %d time levels"
        % (args.ntargets, pde.nt - 1))

    gamma, delta = hm.BiLaplacianComputeCoefficients(args.sigma2, args.rho, 2)
    prior = hm.BiLaplacianPrior(Vu, gamma, delta,
                                solver_type="lu" if comm.size == 1 else "krylov")
    prior_sd = float(np.sqrt(prior.pointwise_variance("Exact").max()))
    log("Prior: sigma^2 = %g, correlation length %g  ->  gamma = %.5g, delta = %.5g"
        % (args.sigma2, args.rho, gamma, delta))
    log("       pointwise standard deviation %.3f (the truth has amplitude 1)"
        % prior_sd)
    model = hm.Model(prob, prior, misfit)

    # ------------------------------------------------- synthetic data
    mtrue = Vu.project(lambda x: np.exp(-60.0 * ((x[0] - 0.3) ** 2
                                                 + (x[1] - 0.4) ** 2)))
    bc.zero(mtrue)
    utrue = prob.generate_state()
    prob.solveFwd(utrue, [utrue, mtrue, None])
    # One noise level for the whole data set, scaled by the largest observed
    # value over all times: a per-level scale would make each level's weight
    # depend on how much the solution had already decayed.
    hm.parRandom.set_seed(1)
    clean_max = 0.0
    for k in range(1, pde.nt):
        B.mult(utrue.view(pde.times[k]), misfits[k].d)
        clean_max = max(clean_max, misfits[k].d.norm("linf"))
    noise_std = args.rel_noise * max(clean_max, 1e-30)
    for k in range(1, pde.nt):
        B.perturb(misfits[k].d, noise_std)
        misfits[k].noise_variance = noise_std ** 2
    log("Noise: relative %g, std %.4e (largest clean value %.4e)"
        % (args.rel_noise, noise_std, clean_max))

    if args.verify:
        log(SEP + "Finite-difference checks" + SEP)
        m0 = Vu.project(lambda x: 0.3 * np.exp(-40.0 * ((x[0] - 0.45) ** 2
                                                        + (x[1] - 0.45) ** 2)))
        bc.zero(m0)
        eps, eg, eH = hm.modelVerify(model, m0, is_quadratic=True,
                                     verbose=(rank == 0))
        log("gradient slope %.3f; max Hessian error %.3e (the cost is quadratic "
            "in m, so this should be round-off)"
            % (hm.best_slope(eps, eg), eH.max()))

    # ------------------------------------------------- MAP point
    log(SEP + "MAP point" + SEP)
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-8
    params["max_iter"] = 30
    params["GN_iter"] = 30          # the problem is linear in m: GN is exact
    # Observing every level at 1% noise makes the posterior sharp, so the
    # prior-preconditioned Hessian still spans several decades and the Newton
    # system needs more than the default 100 CG iterations.
    params["cg_max_iter"] = args.cg_max_iter
    params["print_level"] = 0 if rank == 0 else -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, prior.mean.copy(), None])
    log("\n%s" % solver.termination_reasons[solver.reason])
    log("Newton iterations %d, CG iterations %d"
        % (solver.it, solver.total_cg_iter))
    err = (mtrue.copy().axpy(-1.0, x[PARAMETER]).norm("l2")
           / max(mtrue.norm("l2"), 1e-300))
    log("Relative error in the initial condition: %.4f" % err)

    # ------------------------------------------------- Laplace approximation
    log(SEP + "Laplace approximation" + SEP)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k = args.neig
    Omega = hm.MultiVector(x[PARAMETER], k + 15)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1)
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    log("eigenvalues: d[0] = %.4e, d[%d] = %.4e" % (d[0], k - 1, d[-1]))
    pv, prv, corr = post.pointwise_variance(method="Randomized", r=150)

    # ------------------------------------------------- output
    os.makedirs(args.out, exist_ok=True)
    hm.write_paraview(os.path.join(args.out, "ic"), pmesh, {
        "m_true": (Vu, mtrue),
        "m_map": (Vu, x[PARAMETER]),
        "prior_variance": (Vu, prv),
        "post_variance": (Vu, pv),
    })
    pde.exportState(x[STATE], os.path.join(args.out, "u_map"))
    pde.exportState(utrue, os.path.join(args.out, "u_true"))
    log(SEP + "Output" + SEP)
    log("wrote fields and trajectories under %s" % args.out)
    log("Solver call counts: %s" % pde.n_calls)


if __name__ == "__main__":
    main()

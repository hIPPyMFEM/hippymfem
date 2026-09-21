#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Infer a log-conductivity field from boundary flux data, with a Robin condition.

A demonstration of the two capabilities the AD route gained in the second pass:
a **boundary density** and an **exact parallel solve**.  Infer :math:`m` in

.. math::

    -\nabla\cdot(e^{m}\nabla u) = f \quad\text{in }\Omega=(0,1)^2, \qquad
    e^{m}\,\partial_n u + \kappa\,u = g \quad\text{on }\partial\Omega,

from pointwise observations of :math:`u`.  Nothing here is essential-boundary
constrained: the Robin term is what makes the forward problem solvable, so a
mistake in the boundary block is not a small error, it is a singular matrix.  The
weak residual is the sum of two densities,

.. code-block:: python

    def pde_varf(u, m, p, x):                     # the dx term
        return jnp.exp(m.val) * inner(u.grad, p.grad) - f(x) * p.val

    def bdr_varf(u, m, p, x, n):                  # the ds term
        return (KAPPA * u.val - g(x)) * p.val

and every operator the inverse problem needs is differentiated from both, so the
boundary term enters ``A``, ``C``, ``W_uu``, ``W_um`` and ``W_mm`` without being
mentioned again.

The run also reports the **normal flux** :math:`e^{m}\partial_n u` on the boundary
at the MAP point, computed as a boundary functional of the recovered field.  That
is the quantity a boundary element's own basis cannot produce -- its shape
functions have no normal derivative -- and it is available because the boundary
kernels map each face quadrature point back into the adjacent volume element.

Solves are exact (``hm.LUSolver``, replicated SuperLU) on any number of ranks, so
the Hessian spectrum below is the discrete one and not an artifact of a Krylov
tolerance.

Run::

    python applications/boundary/model_robin.py
    mpirun -n 4 python applications/boundary/model_robin.py
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
from hippymfem.modeling.variables import PARAMETER, STATE           # noqa: E402

SEP = "\n" + "#" * 78 + "\n"
KAPPA = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=32, help="cells per side")
    ap.add_argument("--order", type=int, default=2, help="state polynomial degree")
    ap.add_argument("--ntargets", type=int, default=80)
    ap.add_argument("--noise", type=float, default=0.01,
                    help="relative noise on the data")
    ap.add_argument("--nmodes", type=int, default=40,
                    help="eigenpairs for the Laplace approximation")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--iterative", action="store_true",
                    help="use Krylov+AMG instead of exact solves")
    args = ap.parse_args()

    mfem.Hypre.Init()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    root = rank == 0

    def say(*a):
        if root:
            print(*a, flush=True)

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "results")
    if root:
        os.makedirs(outdir, exist_ok=True)
    comm.Barrier()

    # ------------------------------------------------------------------ spaces
    mesh = mfem.ParMesh(comm, mfem.Mesh.MakeCartesian2D(
        args.nx, args.nx, mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(mesh, args.order)
    Vm = hm.FunctionSpace.H1(mesh, 1)
    say(SEP + "Robin-boundary conductivity inversion")
    say("  state %s, parameter %s, %d ranks"
        % (Vu, Vm, comm.size))

    # ------------------------------------------------------------------ forward
    def source(x):
        return 1.0 + 4.0 * jnp.exp(-20.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))

    def boundary_data(x):
        return 0.5 * x[1] + 0.25

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad) - source(x) * p.val

    def bdr_varf(u, m, p, x, n):
        return (KAPPA * u.val - boundary_data(x)) * p.val

    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, None, None,
                                   is_fwd_linear=True, bdr_varf=bdr_varf,
                                   bdr_attributes="all")
    if args.iterative:
        def make():
            s = hm.KrylovSolver(comm, "cg", "amg")
            s.parameters["rel_tolerance"] = 1e-13
            s.parameters["max_iter"] = 3000
            return s
        kind = "krylov"
    else:
        def make():
            return hm.LUSolver(comm)
        kind = "lu"
    pde.set_solvers(make)
    say("  solves: %s (%s)" % (kind, "exact" if kind == "lu" else "iterative"))

    # -------------------------------------------------------------------- prior
    prior = hm.BiLaplacianPrior(Vm, gamma=0.12, delta=0.6, robin_bc=True,
                                solver_type=kind)
    say("  prior: bi-Laplacian, pointwise std %.3f, correlation length %.3f"
        % (prior.pointwise_std() if hasattr(prior, "pointwise_std") else float("nan"),
           math.sqrt(0.12 / 0.6)))

    # ------------------------------------------------------- synthetic data
    hm.parRandom.set_seed(17)
    mtrue = Vm.project(lambda z: 0.8 * np.sin(2.5 * z[0]) * np.cos(2.0 * z[1])
                       - 0.4 * z[0])
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])

    rng = np.random.default_rng(17)
    targets = np.column_stack((rng.uniform(0.05, 0.95, args.ntargets),
                               rng.uniform(0.05, 0.95, args.ntargets)))
    B = hm.assemblePointwiseObservation(Vu, targets)
    data = B.createVecLeft()
    B.mult(utrue, data)
    noise_std = args.noise * max(data.norm("linf"), 1e-30)
    B.perturb(data, noise_std)
    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    say("  data: %d pointwise observations, noise std %.4e"
        % (args.ntargets, noise_std))

    model = hm.Model(pde, prior, misfit)

    # ------------------------------------------------- the boundary term matters
    # Without the Robin term the forward operator is singular (pure Neumann data
    # with no essential condition), so this is not a refinement of the answer, it
    # is the difference between a solvable problem and an unsolvable one.
    flux_kernel = _flux_functional(Vu, Vm, pde)
    say("  check: the forward solve is well posed only because of the ds term")
    say("         ||u_true|| = %.6f" % utrue.norm("l2"))

    # ---------------------------------------------------------- gradient check
    m0 = prior.mean.copy()
    hm.parRandom.normal(0.2, m0)
    eps, err_grad, err_H = hm.modelVerify(model, m0, verbose=False,
                                          eps=np.logspace(-1, -6, 11))
    say("  modelVerify: gradient FD slope %.4f, Hessian FD slope %.4f"
        % (hm.best_slope(eps, err_grad), hm.best_slope(eps, err_H, lo=1e-4, hi=1e-1)))

    # -------------------------------------------------------------- MAP point
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["max_iter"] = 30
    params["print_level"] = 0 if root else -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    say(SEP + "Newton-CG")
    x = solver.solve([None, prior.mean.copy(), None])
    say("  %s in %d iterations; ||g||/||g_0|| = %.3e"
        % (solver.termination_reasons[solver.reason], solver.it,
           solver.final_grad_norm / max(solver.initial_grad_norm, 1e-300)))

    err = x[PARAMETER].copy().axpy(-1.0, mtrue)
    say("  relative L2 error in m: %.4f (prior mean would give %.4f)"
        % (err.norm("l2") / max(mtrue.norm("l2"), 1e-300),
           prior.mean.copy().axpy(-1.0, mtrue).norm("l2")
           / max(mtrue.norm("l2"), 1e-300)))

    # ------------------------------------------------ boundary flux at the MAP
    total, absflux = flux_kernel(x[STATE], x[PARAMETER])
    t_true, a_true = flux_kernel(utrue, mtrue)
    say(SEP + "Boundary normal flux  int_dOmega exp(m) du/dn ds")
    say("  at the MAP point : net %+.6f   absolute %.6f" % (total, absflux))
    say("  at the truth     : net %+.6f   absolute %.6f" % (t_true, a_true))
    say("  (the net flux must balance the source: int_Omega f dx = %.6f)"
        % _source_integral(Vu, Vm, pde, source))

    # ------------------------------------------------- Laplace approximation
    say(SEP + "Laplace approximation of the posterior")
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k = min(args.nmodes, Vm.GlobalTrueVSize() - 2)
    Omega = hm.MultiVector(x[PARAMETER], k + 10)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)
    post = hm.GaussianLRPosterior(prior, d, U)
    post.mean = x[PARAMETER]
    say("  %d eigenvalues, range [%.3e, %.3e]" % (k, d[-1], d[0]))
    say("  log|I + H_misfit| (posterior vs prior) = %.6f"
        % post.logdet_Hessian() if hasattr(post, "logdet_Hessian") else "")

    # ---------------------------------------------------------------- output
    hm.write_paraview(os.path.join(outdir, "robin_map"), mesh,
                      {"m_map": (Vm, x[PARAMETER]), "m_true": (Vm, mtrue),
                       "u_map": (Vu, x[STATE]), "u_true": (Vu, utrue)})
    say(SEP + "wrote %s" % os.path.join(outdir, "robin_map"))
    return 0


def _flux_functional(Vu, Vm, pde):
    """A callable returning the net and absolute boundary normal flux.

    Built as a boundary density of the state and parameter, which is the only way
    to get ``du/dn`` on the boundary: the integrand is ``exp(m) grad(u) . n``.
    """
    from hippymfem.fem.boundary import BoundaryKernel, get_boundary_batches
    from hippymfem.modeling.variables import ADJOINT

    bb = get_boundary_batches(Vu.mesh, pde.quadrature_degree, "all",
                              Vu.comm, space=Vu)
    net = BoundaryKernel(
        lambda u, m, p, x, n: jnp.exp(m.val) * jnp.dot(u.grad, n),
        [Vu, Vm, Vu], bb)
    absol = BoundaryKernel(
        lambda u, m, p, x, n: jnp.abs(jnp.exp(m.val) * jnp.dot(u.grad, n)),
        [Vu, Vm, Vu], bb)
    from hippymfem.fem.assemble import assemble_scalar

    def evaluate(u, m):
        loc = [Vu.local_values(u), Vm.local_values(m),
               Vu.local_values(Vu.vector())]
        return (assemble_scalar(Vu.comm, net.element_values(loc)),
                assemble_scalar(Vu.comm, absol.element_values(loc)))

    return evaluate


def _source_integral(Vu, Vm, pde, source):
    """``int_Omega f dx``, for the flux balance check."""
    from hippymfem.fem.assemble import assemble_scalar
    from hippymfem.fem.elementbatch import get_batches
    from hippymfem.fem.kernel import QuadratureKernel

    batches = get_batches(Vu.mesh, pde.quadrature_degree)
    K = QuadratureKernel(lambda u, m, p, x: source(x), [Vu, Vm, Vu], batches)
    loc = [Vu.local_values(Vu.vector()), Vm.local_values(Vm.vector()),
           Vu.local_values(Vu.vector())]
    return assemble_scalar(Vu.comm, K.element_values(loc))


if __name__ == "__main__":
    raise SystemExit(main())

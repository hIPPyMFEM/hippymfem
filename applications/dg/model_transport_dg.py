#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Infer a diffusivity field from a downstream plume, discretised with DG.

A transport problem at a Peclet number where continuous Galerkin does not belong.
A tracer enters on the left, is carried by a uniform wind and spreads by diffusion:

.. math::

    b\cdot\nabla u - \nabla\cdot(\kappa e^{m}\nabla u) = 0 \quad\text{in }(0,1)^2,
    \qquad u = g \ \text{on the inflow}, \qquad
    \kappa e^{m}\partial_n u = 0 \ \text{on the outflow},

and the inverse problem is to recover the log-diffusivity :math:`m` from
concentration measurements taken downstream of the inlet.  The plume's width is
what the data see, and the diffusivity is what sets it.

**Why discontinuous Galerkin.**  Advection dominates: the mesh Peclet number is in
the tens, where an unstabilised continuous discretisation has no mechanism to pick the
upwind direction.  The upwind flux below is that mechanism, and it lives on the
interior faces.  It also makes the scheme conservative element by element, so the
tracer entering the inlet leaves through the outflow to machine precision; the run
reports that balance.

**What the discretisation costs to write.**  Three densities, and nothing else:

.. code-block:: python

    def pde_varf(u, m, p, x):                       # dx
        return kappa(m) * inner(u.grad, p.grad) - u.val * dot(b(x), p.grad)

    def facet_varf(u, m, p, x, n, h):               # dS: upwind flux, then SIPG
        bn = dot(b(x), n)
        return ((bn * avg(u) + 0.5 * abs(bn) * jump(u)) * jump(p)
                - kappa_f(m) * dot(avg_grad(u), n) * jump(p)
                - kappa_f(m) * jump(u) * dot(avg_grad(p), n)
                + penalty * kappa_f(m) * avg_inv_h(h) * jump(u) * jump(p))

    def bdr_varf(u, m, p, x, n, h):                 # ds: inflow data, outflow flux
        ...

They are differentiated once, so the facet term reaches the Jacobian, the parameter
Jacobian and every second-order block without being mentioned again: the diffusivity
multiplies the interior-face flux and its penalty, so a facet block that was wrong
would show up as a wrong gradient.  ``modelVerify``'s finite differences below are
the check.

Run::

    python applications/dg/model_transport_dg.py
    mpirun -n 4 python applications/dg/model_transport_dg.py --nx 48 --order 2
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
from hippymfem.fem.facets import (avg, avg_grad, facet_values,      # noqa: E402
                                  get_facet_batches, jump)
from hippymfem.modeling.variables import PARAMETER, STATE           # noqa: E402

SEP = "\n" + "#" * 78 + "\n"
WIND = np.array([1.0, 0.3])          # uniform, so inflow is the left and bottom sides
KAPPA = 1e-3                         # diffusivity scale; e^m multiplies it
PENALTY = 10.0                       # interior penalty, in units of {kappa / h}


def inflow(x):
    """The tracer profile carried in at the inlet: a band with sharp edges."""
    return 0.5 * (1.0 + jnp.tanh((0.14 - jnp.abs(x[1] - 0.55)) / 0.02))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=32, help="cells per side")
    ap.add_argument("--order", type=int, default=1, help="DG polynomial degree")
    ap.add_argument("--ntargets", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--nmodes", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iterative", action="store_true",
                    help="Krylov + AMG instead of exact solves")
    ap.add_argument("--no-map", action="store_true",
                    help="stop after the derivative checks")
    args = ap.parse_args()

    mfem.Hypre.Init()
    comm = MPI.COMM_WORLD
    root = comm.rank == 0

    def say(*a):
        if root:
            print(*a, flush=True)

    outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "results")
    if root:
        os.makedirs(outdir, exist_ok=True)
    comm.Barrier()

    mesh = mfem.ParMesh(comm, mfem.Mesh.MakeCartesian2D(
        args.nx, args.nx, mfem.Element.QUADRILATERAL))
    Vu = hm.FunctionSpace.L2(mesh, args.order)
    Vm = hm.FunctionSpace.H1(mesh, 1)
    say(SEP + "Transport at high Peclet number, discontinuous Galerkin")
    say("  wind (%.2f, %.2f), diffusivity %.1e, mesh Peclet %.1f"
        % (WIND[0], WIND[1], KAPPA,
           float(np.linalg.norm(WIND)) / (args.nx * KAPPA)))
    say("  state %s, parameter %s, %d ranks" % (Vu, Vm, comm.size))

    # --------------------------------------------------------------- the form
    def kappa(m):
        return KAPPA * jnp.exp(m.val)

    def pde_varf(u, m, p, x):
        return (kappa(m) * hm.inner(u.grad, p.grad)
                - u.val * jnp.dot(jnp.asarray(WIND), p.grad))

    def facet_varf(u, m, p, x, n, h):
        bn = jnp.dot(jnp.asarray(WIND), n)
        k = KAPPA * jnp.exp(avg(m))
        upwind = (bn * avg(u) + 0.5 * jnp.abs(bn) * jump(u)) * jump(p)
        sipg = (-k * jnp.dot(avg_grad(u), n) * jump(p)
                - k * jump(u) * jnp.dot(avg_grad(p), n)
                + PENALTY * k * 0.5 * (1.0 / h[0] + 1.0 / h[1]) * jump(u) * jump(p))
        return upwind + sipg

    def bdr_varf(u, m, p, x, n, h):
        """One density for both halves of the boundary: the normal decides which."""
        bn = jnp.dot(jnp.asarray(WIND), n)
        k = kappa(m)
        d = u.val - inflow(x)
        # outflow carries the state out; inflow carries the data in
        advect = jnp.where(bn > 0.0, bn * u.val, bn * inflow(x)) * p.val
        nitsche = (-k * jnp.dot(u.grad, n) * p.val - k * d * jnp.dot(p.grad, n)
                   + PENALTY * k * (1.0 / h) * d * p.val)
        return advect + jnp.where(bn < 0.0, nitsche, 0.0)

    pde = hm.PDEVariationalProblem(
        [Vu, Vm, Vu], pde_varf, None, None, is_fwd_linear=True,
        quadrature_degree=2 * args.order + 2, facet_varf=facet_varf,
        bdr_varf=bdr_varf, bdr_attributes="all")
    if args.iterative:
        def make():
            s = hm.KrylovSolver(comm, "gmres", "amg")
            s.parameters["rel_tolerance"] = 1e-12
            s.parameters["max_iter"] = 2000
            return s
        kind = "krylov"
    else:
        def make():
            return hm.LUSolver(comm)
        kind = "lu"
    pde.set_solvers(make)
    say("  no essential dofs: the inflow data is imposed weakly (%s solves)" % kind)

    # -------------------------------------------------------- the true state
    hm.parRandom.set_seed(21)
    mtrue = Vm.project(lambda z: 0.9 * np.sin(3.0 * z[0]) * np.cos(2.0 * z[1])
                       - 0.5 * z[1])
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    lo, hi = _range(Vu, utrue)
    jumps = _jump_norm(Vu, Vm, pde)
    jn = jumps(utrue, mtrue)
    flux = _flux_report(Vu, Vm, pde, bdr_varf)
    tin, tout, imbalance = flux(utrue, mtrue)
    say(SEP + "Forward solve")
    say("  state range [%+.4f, %+.4f]; the tracer enters in [0, 1]" % (lo, hi))
    say("  interior jump ||[u]||_dS = %.3e, %.2f %% of ||u||: the solution is"
        " discontinuous\n   where it has to be and continuous where it does not"
        % (jn, 100 * jn / max(utrue.norm("l2"), 1e-300)))
    say("  tracer in %.6f, out %.6f, imbalance %.2e relative: the constant is in the"
        "\n   test space, so an upwind DG solve balances the tracer exactly"
        % (tin, tout, abs(imbalance) / max(abs(tin), 1e-300)))

    # ------------------------------------------------------------------- data
    rng = np.random.default_rng(21)
    targets = np.column_stack((rng.uniform(0.35, 0.95, args.ntargets),
                               rng.uniform(0.15, 0.95, args.ntargets)))
    B = hm.assemblePointwiseObservation(Vu, targets)
    data = B.createVecLeft()
    B.mult(utrue, data)
    noise_std = args.noise * max(data.norm("linf"), 1e-30)
    B.perturb(data, noise_std)
    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    prior = hm.BiLaplacianPrior(Vm, gamma=0.1, delta=0.5, robin_bc=True,
                                solver_type=kind)
    model = hm.Model(pde, prior, misfit)
    say("  %d downstream observations, noise std %.3e" % (args.ntargets, noise_std))

    # -------------------------------------------------- the facet blocks matter
    m0 = prior.mean.copy()
    hm.parRandom.normal(0.2, m0)
    eps, err_grad, err_H = hm.modelVerify(model, m0, verbose=False,
                                          eps=np.logspace(-2, -7, 11))
    say(SEP + "Derivatives of a residual with a dS term")
    say("  modelVerify: gradient FD slope %.4f, Hessian FD slope %.4f"
        % (hm.best_slope(eps, err_grad), hm.best_slope(eps, err_H)))
    say("  (the diffusivity multiplies the interior-face flux and its penalty, so"
        " these slopes\n   are the check that C, W_uu, W_um and W_mm carry their"
        " facet part)")
    if args.no_map:
        return 0

    # --------------------------------------------------------------- inversion
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-8
    params["max_iter"] = 25
    params["print_level"] = 0 if root else -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    say(SEP + "Newton-CG")
    x = solver.solve([None, prior.mean.copy(), None])
    say("  %s in %d iterations; ||g||/||g_0|| = %.3e"
        % (solver.termination_reasons[solver.reason], solver.it,
           solver.final_grad_norm / max(solver.initial_grad_norm, 1e-300)))
    err = x[PARAMETER].copy().axpy(-1.0, mtrue).norm("l2")
    base = prior.mean.copy().axpy(-1.0, mtrue).norm("l2")
    say("  relative L2 error in m: %.4f (the prior mean gives %.4f)"
        % (err / max(mtrue.norm("l2"), 1e-300),
           base / max(mtrue.norm("l2"), 1e-300)))

    # ------------------------------------------------------ Laplace posterior
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k = min(args.nmodes, Vm.GlobalTrueVSize() - 2)
    Omega = hm.MultiVector(x[PARAMETER], k + 10)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    tr_post, tr_prior, _ = post.trace()
    say(SEP + "Laplace approximation")
    say("  %d eigenvalues in [%.3e, %.3e]; posterior trace %.4f of the prior's %.4f"
        % (k, d[-1], d[0], tr_post, tr_prior))

    hm.write_paraview(os.path.join(outdir, "transport_dg"), mesh,
                      {"m_map": (Vm, x[PARAMETER]), "m_true": (Vm, mtrue),
                       "u_map": (Vu, x[STATE]), "u_true": (Vu, utrue)})
    say(SEP + "wrote %s" % os.path.join(outdir, "transport_dg"))
    return 0


def _range(V, v):
    """The state's smallest and largest dof values, over every rank."""
    a = V.local_values(v)
    comm = V.comm
    return (comm.allreduce(float(np.min(a)), op=MPI.MIN),
            comm.allreduce(float(np.max(a)), op=MPI.MAX))


def _jump_norm(Vu, Vm, pde):
    """``sqrt(int_F [u]^2 dS)`` as a facet functional of the state."""
    from hippymfem.fem.kernel import QuadratureKernel

    batches = get_facet_batches(Vu.mesh, pde.quadrature_degree, space=Vu)
    K = QuadratureKernel(lambda u, m, p, x, n, h: jump(u) ** 2,
                         [Vu, Vm, Vu], batches)

    def evaluate(u, m):
        loc = [facet_values(Vu, u), facet_values(Vm, m),
               facet_values(Vu, Vu.vector())]
        # a face shared with another rank is seen from both sides, with the same
        # value: half from each counts it once
        local = sum(float(np.dot(np.asarray(v, dtype=float).reshape(-1),
                                 np.where(g.shared, 0.5, 1.0)))
                    for g, v in zip(batches.groups, K.element_values(loc)))
        return float(np.sqrt(max(Vu.comm.allreduce(local, op=MPI.SUM), 0.0)))

    return evaluate


def _flux_report(Vu, Vm, pde, bdr_varf):
    """``(in, out, imbalance)``: the tracer crossing the boundary, from the ds density.

    Testing the residual with the constant function leaves only the boundary term, so
    the same density the forward problem was written with *is* the flux balance; a
    solve that satisfies it to rounding is one whose upwind flux and weak inflow
    condition agree.
    """
    from hippymfem.fem.assemble import assemble_scalar
    from hippymfem.fem.boundary import BoundaryKernel, get_boundary_batches

    bb = get_boundary_batches(Vu.mesh, pde.quadrature_degree, "all", Vu.comm,
                              space=Vu)
    wind = jnp.asarray(WIND)
    kernels = {
        "in": BoundaryKernel(
            lambda u, m, p, x, n: -jnp.minimum(jnp.dot(wind, n), 0.0) * inflow(x),
            [Vu, Vm, Vu], bb),
        "out": BoundaryKernel(
            lambda u, m, p, x, n: jnp.maximum(jnp.dot(wind, n), 0.0) * u.val,
            [Vu, Vm, Vu], bb),
        "net": BoundaryKernel(bdr_varf, [Vu, Vm, Vu], bb),
    }
    one = Vu.vector()
    one.array[:] = 1.0                       # p = 1: the residual's flux balance

    def evaluate(u, m):
        loc = [Vu.local_values(u), Vm.local_values(m), Vu.local_values(one)]
        got = {k: assemble_scalar(Vu.comm, kern.element_values(loc))
               for k, kern in kernels.items()}
        return got["in"], got["out"], got["net"]

    return evaluate


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Forward UQ: the effective permeability of a random medium.

Propagates the prior uncertainty in the log-permeability through the flow problem
to a scalar quantity of interest -- here the dissipated power
:math:`\int_\Omega e^{m}|\nabla u|^2`, which is what sets the effective
permeability.

Three estimates of its mean and variance are compared:

1. **Taylor**, analytic, from the eigenvalues of the prior-preconditioned Hessian
   of the parameter-to-QoI map: no sampling at all;
2. **plain Monte Carlo**, one PDE solve per sample;
3. **Monte Carlo with the Taylor model as a control variate**, which samples the
   *difference* between the map and its quadratic model.

The third is the point of the exercise: when the map is nearly quadratic, the
difference has a much smaller variance, so the same error is reached with far
fewer PDE solves.

Run::

    python applications/forward_uq/model_subsurf_effperm.py
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
from hippymfem.modeling.variables import STATE  # noqa: E402

SEP = "\n" + "#" * 78 + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--neig", type=int, default=40)
    ap.add_argument("--nsamples", type=int, default=300)
    ap.add_argument("--out", default="applications/forward_uq/results")
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
    Vm = hm.FunctionSpace.H1(pmesh, 1)

    def pde_varf(u, m, p, x):
        """-div(exp(m) grad u) = 0 with a unit pressure drop left to right."""
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    def qoi_varf(u, m, x):
        """Dissipated power, integral of exp(m) |grad u|^2."""
        return jnp.exp(m.val) * hm.inner(u.grad, u.grad)

    # attributes: 1 bottom, 2 right, 3 top, 4 left
    bc = hm.DirichletBC(Vu, lambda x: 1.0 - x[0], bdr_attributes=[2, 4])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    # By problem size, not rank count: see the note in poisson/model_subsurf.py.
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        s = hm.auto_solver(Vu, comm)
        if isinstance(s, hm.KrylovSolver):
            s.parameters["rel_tolerance"] = 1e-13
        setattr(pde, a, s)

    prior = hm.BiLaplacianPrior(
        Vm, args.gamma, args.delta,
        solver_type="krylov" if isinstance(pde.solver, hm.KrylovSolver) else "lu")
    qoi = hm.VariationalQoi([Vu, Vm, Vu], qoi_varf)
    p2q = hm.Parameter2QoiMap(pde, qoi)

    log(SEP + "Problem" + SEP)
    log("state dofs %d, parameter dofs %d, ranks %d"
        % (Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), comm.size))
    log("prior: gamma %g, delta %g (correlation length ~ %.3f)"
        % (args.gamma, args.delta, np.sqrt(args.gamma / args.delta)))

    # ------------------------------------------------- Taylor approximation
    log(SEP + "Taylor approximation of the QoI" + SEP)
    hm.parRandom.set_seed(2026)
    k = args.neig
    Omega = hm.MultiVector(prior.mean, k)
    hm.parRandom.normal_multivector(1.0, Omega)
    tay = hm.TaylorApproximationQoi(p2q, prior)
    d, U = tay.computeLowRankFactorization(Omega, k=k, s=1)
    log("QoI at the prior mean: %.8e" % tay.q_bar)
    log("Hessian eigenvalues: d[0] = %.4e, d[%d] = %.4e" % (d[0], k - 1, d[-1]))
    log("")
    log("%-28s %16s %16s" % ("", "mean", "variance"))
    for order in (1, 2):
        log("%-28s %16.8e %16.8e"
            % ("Taylor, order %d" % order, tay.expectedValue(order),
               tay.variance(order)))

    # ------------------------------------------------- Monte Carlo
    log(SEP + "Monte Carlo with the Taylor control variate" + SEP)
    res = hm.varianceReductionMC(prior, p2q, tay, args.nsamples, order=2,
                                 verbose=False)
    log("%-28s %16.8e  (standard error %.2e)"
        % ("plain Monte Carlo", res["mc_mean"], res["mc_stderr"]))
    log("%-28s %16.8e  (standard error %.2e)"
        % ("Taylor + Monte Carlo", res["reduced_mean"], res["reduced_stderr"]))
    log("")
    log("sample standard deviation: Q %.4e, Q - Q_taylor %.4e"
        % (res["sd_q"], res["sd_diff"]))
    log("variance reduction factor: %.1f" % res["variance_reduction"])
    log("  -> the same accuracy needs about %.0fx fewer PDE solves"
        % res["variance_reduction"])
    log("")
    log("sample variance of Q: %.8e" % (res["sd_q"] ** 2))
    log("Taylor variance     : %.8e" % res["taylor_variance"])

    # ------------------------------------------------- output
    os.makedirs(args.out, exist_ok=True)
    u = p2q.generate_vector(STATE)
    p2q.solveFwd(u, [u, prior.mean.copy(), None])
    hm.write_paraview(os.path.join(args.out, "effperm"), pmesh, {
        "m_mean": (Vm, prior.mean),
        "u_mean": (Vu, u),
        "qoi_gradient": (Vm, tay.g_bar),
    })
    modes = {("eigenvector_%02d" % i): (Vm, U[i]) for i in range(min(6, U.nvec()))}
    hm.write_paraview(os.path.join(args.out, "modes"), pmesh, modes)
    if rank == 0:
        np.savetxt(os.path.join(args.out, "samples.csv"),
                   np.column_stack((res["samples_q"], res["samples_taylor"])),
                   delimiter=",", header="Q,Q_taylor", comments="")
        np.savetxt(os.path.join(args.out, "eigenvalues.csv"), d, delimiter=",",
                   header="eigenvalue", comments="")
    log("\nwrote fields, samples and eigenvalues under %s" % args.out)


if __name__ == "__main__":
    main()

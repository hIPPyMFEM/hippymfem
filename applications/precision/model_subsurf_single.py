#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Subsurface flow in 3D: the MAP point with single-precision solves, the Laplace approximation in double.

The model problem of the GPU guide's measurements (``benchmarks/bench_precision.py``):
infer :math:`m` in

.. math:: -\nabla\cdot(e^{m}\nabla u) = 0 \quad\text{in }\Omega=(0,1)^3,
          \qquad u = z \text{ on the bottom and top faces},

from pointwise observations of :math:`u`, with a BiLaplacian prior, on :math:`n^3`
hexahedra (P2 state, P1 parameter).

With a single-precision build of hypre named (``HIPPYMFEM_HYPRE_SINGLE``, built by
``tools/build_hypre_single.sh``) the Jacobian and the CG solves with it run in that
library while Newton-CG computes the MAP point, with the incremental solves of the
Hessian actions at 1e-6.  A single-precision solve reaches a relative residual near
1e-5, which is enough for the Newton directions but not for the small eigenvalues of
the Laplace approximation, so the single-precision solves are switched off after the
MAP point, and the eigenpairs, a posterior sample and the pointwise variance are
computed with double-precision solves.  ``HIPPYMFEM_PRECISION=mixed`` (single-precision
element matrices) goes with the single-precision solves and can stay: at 8^3 the
eigenvalues then agreed with a run in double precision throughout to 2e-7, less than
incremental solves to 1e-8 move them.  ``--fp64-kernels`` switches the element matrices
back as well (4e-10), at the price of compiling their kernels again: at 24^3 on a
Blackwell MIG instance the eigenpairs then took 109 s, against 3 s with the kernels
already compiled.  Without a single-precision library every solve is in double
precision, and the script says so.

Run::

    python applications/precision/model_subsurf_single.py --n 8
    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 HIPPYMFEM_PRECISION=mixed \
        HIPPYMFEM_HYPRE_SINGLE=/path/to/hypre_single/libHYPRE_single.so \
        python applications/precision/model_subsurf_single.py --n 24

The first run on a GPU compiles the element kernels, and on a Blackwell MIG instance
that is most of its time: at 24^3 the MAP point took 46 s.  With JAX's persistent
compilation cache (``JAX_COMPILATION_CACHE_DIR=<directory>``) filled by an earlier run
of the same size, it took 8.5 s and the eigenpairs 2.7 s, in the same 12 Newton and 145
CG iterations.
"""

import argparse
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import hippymfem as hm                                              # noqa: E402
from hippymfem.algorithms import singlesolve                        # noqa: E402
from hippymfem.modeling.variables import PARAMETER, STATE           # noqa: E402

SEP = "\n" + "#" * 78 + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=16, help="elements per edge of the cube")
    ap.add_argument("--order", type=int, default=2, help="state polynomial degree")
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--rel-noise", type=float, default=0.01)
    ap.add_argument("--newton-tol", type=float, default=1e-6,
                    help="relative gradient norm at which Newton-CG stops")
    ap.add_argument("--inc-tol", type=float, default=1e-6,
                    help="relative tolerance of the incremental solves during Newton-CG")
    ap.add_argument("--laplace-inc-tol", type=float, default=1e-8,
                    help="relative tolerance of the incremental solves after the MAP point")
    ap.add_argument("--neig", type=int, default=20,
                    help="eigenpairs for the Laplace approximation")
    ap.add_argument("--nsamples", type=int, default=32,
                    help="Monte Carlo samples of the pointwise variance")
    ap.add_argument("--fp64-kernels", action="store_true",
                    help="after the MAP point, switch HIPPYMFEM_PRECISION=mixed element matrices "
                    "back to double precision as well (compiles their kernels again)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    # ------------------------------------------------- precisions
    # library() loads the library named by HIPPYMFEM_HYPRE_SINGLE, or returns None (with
    # a RuntimeWarning when the library is named but cannot be used): the solves then
    # stay in double precision and nothing else changes.
    single = singlesolve.library() is not None
    log(SEP + "Precisions" + SEP)
    log("element kernels: %s (HIPPYMFEM_PRECISION)" % hm.config.precision)
    if single:
        log("linear solves of the PDE: single-precision hypre %s" % singlesolve.library().path)
    else:
        log("linear solves of the PDE: double precision (%s)"
            % (singlesolve.why_not() or "HIPPYMFEM_HYPRE_SINGLE is not set"))

    # ------------------------------------------------- mesh and spaces
    pmesh = mfem.ParMesh(comm, mfem.Mesh.MakeCartesian3D(
        args.n, args.n, args.n, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, args.order)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    log(SEP + "Mesh and finite element spaces" + SEP)
    log("%d^3 hexahedra: STATE=%d, PARAMETER=%d dofs on %d MPI rank(s)"
        % (args.n, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), comm.size))

    # ------------------------------------------------- forward problem
    def pde_varf(u, m, p, x):
        """Weak residual density of -div(exp(m) grad u) = 0."""
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    # MakeCartesian3D boundary attributes: 1 bottom (z = 0), 6 top (z = 1)
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    # CG with BoomerAMG for the forward and the two incremental solves (max_direct=0:
    # never a direct solve, which has no single-precision counterpart)
    pde.set_solvers(hm.auto_solver, Vu, comm, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))

    # ------------------------------------------------- prior, truth and data
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
    noise_std = args.rel_noise * max(data.norm("linf"), 1e-30)
    B.perturb(data, noise_std)
    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    model = hm.Model(pde, prior, misfit)
    log("%d observation points, relative noise %g" % (args.ntargets, args.rel_noise))

    # ------------------------------------------------- MAP point
    log(SEP + "MAP point (inexact Newton-CG)" + SEP)
    # The CG of a Newton step keeps its residuals orthogonal (cg_reorthogonalize, the
    # default), so the incremental solves need a loose tolerance only, and a
    # single-precision solve, which stops near 1e-5 whatever it is asked, is good enough.
    for name in ("solver_fwd_inc", "solver_adj_inc"):
        getattr(pde, name).parameters["rel_tolerance"] = args.inc_tol
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = args.newton_tol
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = 25
    params["globalization"] = "LS"
    params["GN_iter"] = 5
    params["cg_max_iter"] = 50
    params["print_level"] = 0 if rank == 0 else -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    comm.Barrier()
    t0 = time.perf_counter()
    x = solver.solve([None, prior.mean.copy(), None])
    comm.Barrier()
    t_map = time.perf_counter() - t0
    log("\n%s" % solver.termination_reasons[solver.reason])
    log("Newton iterations: %d, CG iterations: %d, %.1f s (%s solves)"
        % (solver.it, solver.total_cg_iter, t_map, "single-precision" if single else "double-precision"))
    log("Final cost %.10e, gradient norm %.3e"
        % (solver.final_cost, solver.final_grad_norm))
    err = (mtrue.copy().axpy(-1.0, x[PARAMETER]).norm("l2")
           / max(mtrue.norm("l2"), 1e-300))
    log("Relative error in the parameter: %.4f" % err)

    # ------------------------------------------------- back to double precision
    # The eigenpairs come out to about the accuracy of the incremental solves, relative
    # to the largest eigenvalue, so the stages after the MAP point solve in double
    # precision, to 1e-8.  Without a single-precision library this changes nothing but
    # the tolerance.
    pde.single_solves = False          # the Jacobian is assembled in double precision again
    if args.fp64_kernels and hm.config.precision != "fp64":
        hm.config.precision = "fp64"   # and so are its element matrices
    pde.invalidate_jacobian()
    for name in ("solver_fwd_inc", "solver_adj_inc"):
        getattr(pde, name).parameters["rel_tolerance"] = args.laplace_inc_tol

    # ------------------------------------------------- Laplace approximation
    log(SEP + "Laplace approximation of the posterior (double-precision solves)" + SEP)
    comm.Barrier()
    t0 = time.perf_counter()
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k, p = args.neig, 10
    Omega = hm.MultiVector(x[PARAMETER], k + p)
    hm.parRandom.set_seed(99)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1)
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    comm.Barrier()
    t_eig = time.perf_counter() - t0
    log("%d eigenpairs in %.1f s (%d Hessian actions), element kernels %s; "
        "%d eigenvalues above 1" % (k, t_eig, 2 * (k + p), hm.config.precision, int((d > 1.0).sum())))
    log("eigenvalues: %s ... %.6e" % (" ".join("%.6e" % v for v in d[:4]), d[-1]))
    log("KL divergence of the Laplace posterior from the prior: %.6e"
        % post.klDistanceFromPrior())

    s_pr, s_po = Vm.vector(), Vm.vector()
    prior.sample_noise(1.0, noise)
    post.sample(noise, s_pr, s_po)
    log("a posterior sample and the prior sample of the same noise: |m_post - m_MAP| = %.4e, "
        "|m_prior - m_prior_mean| = %.4e"
        % (s_po.copy().axpy(-1.0, x[PARAMETER]).norm("l2"),
           s_pr.copy().axpy(-1.0, prior.mean).norm("l2")))

    comm.Barrier()
    t0 = time.perf_counter()
    pv, prv, _ = post.pointwise_variance(method="MonteCarlo", n=args.nsamples)
    comm.Barrier()
    t_var = time.perf_counter() - t0
    log("pointwise variance, Monte Carlo with %d samples, %.1f s: mean %.4e posterior, "
        "%.4e prior" % (args.nsamples, t_var, pv.sum() / Vm.GlobalTrueVSize(),
                        prv.sum() / Vm.GlobalTrueVSize()))
    log("\nSolver call counts: %s" % pde.n_calls)


if __name__ == "__main__":
    main()

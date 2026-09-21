#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The Laplace approximation of the posterior, stage by stage, with a solver breakdown.

Same problem as ``bench_newton_device.py`` (``n^3`` second-order hexahedra,
``exp(m) grad u . grad p``, BiLaplacian prior, 200 pointwise observations).  After the
MAP point:

1. the randomized generalized eigensolver ``doublePassG`` for the prior-preconditioned
   data-misfit Hessian, ``k`` eigenpairs with ``p`` oversampling columns;
2. ``N`` posterior samples through ``GaussianLRPosterior.sample``;
3. the randomized pointwise variance with ``r`` probes, and the traces.

Every solver and operator the stages call is wrapped, so the time of each stage is
split into Hessian applies (two incremental solves each), prior ``R^{-1}`` applies
(two parameter-space AMG solves each), the B-orthogonalization, and the rest.  The
usual sanity checks are made as well: ``U`` is R-orthonormal, the spectrum is
sorted, and the sample variance of the ``N`` samples tracks the low-rank pointwise
variance.

    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 PYTHONPATH=<cuda pymfem> \\
        mpirun -n 4 tools/mpirun_pinned.sh python benchmarks/bench_laplace.py --n 64 --k 50
"""

import argparse
import collections
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

from hippymfem import _jaxconfig                                     # noqa: E402
from hippymfem.fem import kernel as kernel_mod                       # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402

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


class Meter:
    """Accumulates wall time and call counts of the methods it wraps."""

    def __init__(self):
        self.t = collections.defaultdict(float)
        self.n = collections.defaultdict(int)
        self._undo = []

    def wrap(self, obj, name, label):
        f = getattr(obj, name)

        def g(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                self.t[label] += time.perf_counter() - t0
                self.n[label] += 1

        setattr(obj, name, g)
        self._undo.append((obj, name, f))
        return self

    def snapshot(self):
        return {k: (self.t[k], self.n[k]) for k in self.t}

    @staticmethod
    def delta(after, before):
        return {k: (after[k][0] - before.get(k, (0.0, 0))[0], after[k][1] - before.get(k, (0.0, 0))[1])
                for k in after if after[k][1] - before.get(k, (0.0, 0))[1] > 0}

    def restore(self):
        for obj, name, f in reversed(self._undo):
            setattr(obj, name, f)
        self._undo = []


def report(label, total, parts, tops):
    """A stage's time split into the named top-level categories (which do not overlap);
    every other wrapped call that ran inside them is listed underneath, unsummed."""
    tops = [t for t in tops if t in parts]
    acc = sum(parts[t][0] for t in tops)
    say("  %-40s %8.2f s" % (label, total))
    for t in sorted(tops, key=lambda t: -parts[t][0]):
        tt, n = parts[t]
        say("    %-38s %8.2f s  %5.1f%%  (%d calls, %.1f ms each)"
            % (t, tt, 100.0 * tt / max(total, 1e-12), n, 1e3 * tt / max(n, 1)))
    say("    %-38s %8.2f s  %5.1f%%" % ("everything else", total - acc, 100.0 * (total - acc) / max(total, 1e-12)))
    inner = [(k, v) for k, v in parts.items() if k not in tops]
    for k, (tt, n) in sorted(inner, key=lambda kv: -kv[1][0]):
        say("      inside the above: %-20s %8.2f s  (%d calls, %.1f ms each)" % (k, tt, n, 1e3 * tt / max(n, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--k", type=int, default=50, help="eigenpairs")
    ap.add_argument("--p", type=int, default=20, help="oversampling columns")
    ap.add_argument("--passes", type=int, default=1, help="power iterations s in doublePassG")
    ap.add_argument("--single-pass", action="store_true", help="singlePassG: half the Hessian applies, less accurate")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--r", type=int, default=64, help="probes for the randomized variance")
    ap.add_argument("--newton-max", type=int, default=25)
    ap.add_argument("--newton-tol", type=float, default=1e-6)
    ap.add_argument("--cg-max", type=int, default=50)
    ap.add_argument("--gauss-newton", action="store_true", help="Gauss-Newton Hessian at the MAP")
    ap.add_argument("--inc-tol", type=float, default=1e-8,
                    help="relative tolerance of the incremental solves for the Laplace stages; 1e-8 moves the "
                    "leading eigenvalues by a few 1e-6 relative and saves about a quarter of the time of 1e-12")
    ap.add_argument("--prior-tol", type=float, default=1e-8,
                    help="relative tolerance of the prior's solves; 1e-8 costs a few 1e-6 on the eigenvalues "
                    "and 1e-9 in R-orthonormality for prior solves several times faster than 1e-12")
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"),
                    help="MFEM device; the default follows HIPPYMFEM_HYPRE_DEVICE")
    ap.add_argument("--symmetric-jacobian", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="save the sample and pointwise variances (rank 0, 1 rank only)")
    args = ap.parse_args()
    N, ORDER = args.n, args.order

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks" %
        (N, ORDER, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), COMM.size))
    say("  kernels on %s, MFEM on %s, k=%d p=%d passes=%d, %d samples, %d variance probes"
        % (kernel_mod.device(), args.device, args.k, args.p, args.passes, args.samples, args.r))

    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc.homogeneous(), is_fwd_linear=True,
                                   symmetric_jacobian=(True if args.symmetric_jacobian
                                                       else "auto"))
    # the three solves the records were taken with; the adjoint keeps its default
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    if args.prior_tol is not None:
        for sol in (getattr(prior, "Asolver", None), getattr(prior, "Msolver", None)):
            if sol is not None and hasattr(sol, "parameters"):
                sol.parameters["rel_tolerance"] = args.prior_tol
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
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)

    # ---- MAP
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = args.newton_tol
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = args.newton_max
    params["globalization"] = "LS"
    params["GN_iter"] = 5
    params["cg_max_iter"] = args.cg_max
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    t_map, x = timed(lambda: solver.solve([None, prior.mean.copy(), None]))
    say("  MAP: %.2f s, %d Newton its, %d CG its, %s" % (
        t_map, solver.it, solver.total_cg_iter, solver.termination_reasons[solver.reason]))

    if args.inc_tol is not None:
        for attr in ("solver_fwd_inc", "solver_adj_inc"):
            getattr(pde, attr).parameters["rel_tolerance"] = args.inc_tol
    # ---- instruments
    meter = Meter()
    meter.wrap(pde.solver_fwd_inc, "solve", "incremental solve (fwd)")
    meter.wrap(pde.solver_adj_inc, "solve", "incremental solve (adj)")
    for name, label in (("Asolver", "prior A solve"), ("Msolver", "prior M solve")):
        sol = getattr(prior, name, None)
        if sol is not None and hasattr(sol, "solve"):
            meter.wrap(sol, "solve", label)
    meter.wrap(prior.R, "mult", "R matvec")
    meter.wrap(prior.Rsolver, "solve", "prior R^-1 apply")
    meter.wrap(prior, "sample", "prior sample")
    meter.wrap(prior, "sample_noise", "noise draw")
    if hasattr(prior, "sqrtM"):
        meter.wrap(prior.sqrtM, "mult", "sqrtM matvec")
    meter.wrap(hm.MultiVector, "Borthogonalize", "B-orthogonalization")
    meter.wrap(hm.MultiVector, "dot", "MultiVector dot")

    # ---- eigensolver
    t_hess, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=args.gauss_newton))
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    meter.wrap(Hmisfit, "mult", "Hessian apply")
    Omega = hm.MultiVector(x[PARAMETER], args.k + args.p)
    hm.parRandom.set_seed(99)
    hm.parRandom.normal_multivector(1.0, Omega)
    before = meter.snapshot()
    eig = hm.singlePassG if args.single_pass else hm.doublePassG
    t_eig, (d, U) = timed(lambda: eig(Hmisfit, prior.R, prior.Rsolver, Omega, args.k, s=args.passes))
    eig_parts = Meter.delta(meter.snapshot(), before)
    BU = hm.MultiVector(U[0], U.nvec())
    hm.MatMvMult(prior.R, U, BU)
    orth = float(np.abs(U.dot_mv(BU) - np.eye(U.nvec())).max())
    say("  Hessian blocks at the MAP: %.2f s" % t_hess)
    report("eigensolver (k=%d, p=%d)" % (args.k, args.p), t_eig, eig_parts,
           ["Hessian apply", "prior R^-1 apply", "B-orthogonalization", "MultiVector dot"])
    say("    eigenvalues: %.3e .. %.3e, %d above 1, R-orthonormality %.1e" % (d[0], d[-1], int((d > 1).sum()), orth))

    # ---- sampling
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    meter.wrap(post.sampler, "sample", "low-rank correction")
    meter.wrap(post, "sample", "posterior sample")
    s_pr, s_po = Vm.vector(), Vm.vector()
    acc = np.zeros(Vm.GetTrueVSize() if hasattr(Vm, "GetTrueVSize") else s_po.local_size)
    hm.parRandom.set_seed(7)

    def draw():
        acc[:] = 0.0
        for _ in range(args.samples):
            prior.sample_noise(1.0, noise)
            post.sample(noise, s_pr, s_po, add_mean=False)
            acc[:] += s_po.array ** 2
        acc[:] /= args.samples
    before = meter.snapshot()
    t_samp, _ = timed(draw)
    samp_parts = Meter.delta(meter.snapshot(), before)
    report("%d posterior samples" % args.samples, t_samp, samp_parts, ["noise draw", "posterior sample"])
    say("    per sample %.1f ms" % (1e3 * t_samp / args.samples))

    # ---- variance and traces
    before = meter.snapshot()
    t_var, (pv, prv, corr) = timed(lambda: post.pointwise_variance(method="Randomized", r=args.r))
    var_parts = Meter.delta(meter.snapshot(), before)
    report("pointwise variance, randomized (r=%d)" % args.r, t_var, var_parts, ["prior R^-1 apply", "MultiVector dot"])
    before = meter.snapshot()
    t_mc, (pv_mc, prv_mc, _c) = timed(lambda: post.pointwise_variance(method="MonteCarlo", n=args.samples))
    mc_parts = Meter.delta(meter.snapshot(), before)
    report("pointwise variance, Monte Carlo (n=%d)" % args.samples, t_mc, mc_parts, ["noise draw", "prior sample"])
    say("    prior variance, mean: randomized %.4e  Monte Carlo %.4e  (ratio %.3f; the randomized one is a truncation)"
        % (prv.array.mean() if prv.local_size else 0.0, prv_mc.array.mean() if prv_mc.local_size else 0.0,
           prv.sum() / max(prv_mc.sum(), 1e-300)))
    t_tr, (tr_post, tr_pr, tr_corr) = timed(lambda: post.trace(method="Randomized", r=args.r))
    say("  traces (r=%d): %.2f s  posterior %.4e prior %.4e correction %.4e" % (args.r, t_tr, tr_post, tr_pr, tr_corr))
    # sample variance against the low-rank pointwise variance (a loose statistical check)
    num = (np.abs(acc - pv_mc.array).max() / max(np.abs(pv_mc.array).max(), 1e-300)) if pv_mc.local_size else 0.0
    num = COMM.allreduce(num, op=MPI.MAX)
    below = COMM.allreduce(int(np.all(pv.array <= prv.array + 1e-12)), op=MPI.MIN) == 1
    say("    sample variance vs pointwise variance: max rel dev %.3f over %d samples; posterior <= prior: %s"
        % (num, args.samples, below))
    meter.restore()
    if args.dump and COMM.size == 1:
        np.savez(args.dump, sample_var=acc, pointwise_var=pv.array, prior_var=prv.array, d=np.asarray(d),
                 U0=U[0].array, mean=x[PARAMETER].array)

    rec = {"host": platform.node(), "ranks": COMM.size, "n": N, "order": ORDER, "mfem_device": args.device,
           "kernels": str(kernel_mod.device()), "tdofs": Vu.GlobalTrueVSize(), "mdofs": Vm.GlobalTrueVSize(),
           "k": args.k, "p": args.p, "passes": args.passes, "single_pass": bool(args.single_pass), "samples": args.samples, "r": args.r,
           "gauss_newton": bool(args.gauss_newton), "prior_tol": args.prior_tol, "inc_tol": args.inc_tol,
           "t_map": t_map, "newton_it": solver.it, "cg_it": solver.total_cg_iter,
           "t_hess_blocks": t_hess, "t_eig": t_eig, "eig_parts": {k: list(v) for k, v in eig_parts.items()},
           "d_max": float(d[0]), "d_min": float(d[-1]), "n_above_one": int((d > 1).sum()), "orth": orth,
           "t_samples": t_samp, "samp_parts": {k: list(v) for k, v in samp_parts.items()},
           "t_variance": t_var, "var_parts": {k: list(v) for k, v in var_parts.items()}, "t_variance_mc": t_mc, "t_trace": t_tr,
           "tr_post": tr_post, "tr_prior": tr_pr, "tr_corr": tr_corr, "sample_var_rel_dev": num,
           "posterior_below_prior": bool(below)}
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

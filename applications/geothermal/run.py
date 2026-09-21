#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The geothermal inversion, end to end, with timings and records.

    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 PYTHONPATH=<cuda pymfem> \\
      mpirun -n 4 tools/mpirun_pinned.sh python -m applications.geothermal.run --n 64 --out results/local/geothermal_n64_r4.json

Stages: synthetic truth and data, MAP by Newton-CG, Laplace approximation (doublePassG),
posterior samples, Monte Carlo and randomized pointwise variance, traces, KL divergence,
the mean-temperature QoI at the MAP with its linearized posterior std and its spread over
posterior samples pushed through the nonlinear forward solve.  Rank 0 dumps the truth, MAP
and posterior std on the parameter grid for the figures.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

from hippymfem.fem import kernel as kernel_mod                       # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402
from applications.geothermal.model import Geothermal, T_SCALE, ANOMALY   # noqa: E402
from applications.geothermal.qoi import target_mean_qoi, linearized_posterior_std, sampled_qoi, taylor_qoi  # noqa: E402

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


def gathered(space, v):
    """Coordinates and values of a P1 field on rank 0 (None elsewhere)."""
    xyz = space.coordinates()
    vals = v.array.copy()
    X = COMM.gather(xyz, root=0)
    V = COMM.gather(vals, root=0)
    if RANK == 0:
        return np.concatenate(X), np.concatenate(V)
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--boreholes", type=int, default=60)
    ap.add_argument("--noise-kelvin", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--qoi-samples", type=int, default=32)
    ap.add_argument("--taylor-k", type=int, default=30, help="rank of the QoI-Hessian factorization (0 skips the Taylor moments)")
    ap.add_argument("--skip-qoi", action="store_true", help="stop after the posterior std (no QoI stage)")
    ap.add_argument("--gauss-newton", action="store_true",
                    help="Gauss-Newton throughout: the MAP's every iteration and the Laplace Hessian drop the "
                         "second-order blocks (W_uu, W_um, W_mm), which is what lets 128^3 fit four 45 GB cards")
    ap.add_argument("--newton-max", type=int, default=30)
    ap.add_argument("--newton-tol", type=float, default=1e-6)
    ap.add_argument("--cg-max", type=int, default=50)
    ap.add_argument("--inc-tol", type=float, default=1e-8)
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"),
                    help="MFEM device; the default follows HIPPYMFEM_HYPRE_DEVICE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="npz with truth / MAP / posterior std on the P1 grid (rank 0)")
    ap.add_argument("--paraview", default=None, help="basename for a ParaView collection")
    args = ap.parse_args()

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))
    t_build, G = timed(lambda: Geothermal(args.n, COMM, order=args.order, nboreholes=args.boreholes,
                                          noise_kelvin=args.noise_kelvin))
    info = G.summary()
    say("geothermal %d^3: %d state dofs, %d parameter dofs, %d observations in %d boreholes, noise %.2f K, "
        "T_max(true) %.0f K, forward Newton its %d, %d ranks, kernels on %s, MFEM on %s, built in %.1f s"
        % (args.n, info["state_dofs"], info["param_dofs"], info["observations"], info["boreholes"],
           info["noise_kelvin"], info["u_true_max_kelvin"], info["forward_newton_iterations"], COMM.size,
           kernel_mod.device(), args.device, t_build))
    model, prior, pde = G.model, G.prior, G.pde
    for attr in ("solver_fwd_inc", "solver_adj_inc"):
        getattr(pde, attr).parameters["rel_tolerance"] = args.inc_tol

    # ---- prior spread, so the numbers below can be read
    t_pv, prv = timed(lambda: prior.pointwise_variance(method="MonteCarlo", n=32))
    say("  prior pointwise std (MC, 32): mean %.3f, max %.3f in log k (%.1f s)"
        % (float(np.sqrt(prv.sum() / prv.global_size)), float(np.sqrt(prv.max())), t_pv))

    # ---- MAP
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = args.newton_tol
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = args.newton_max
    params["globalization"] = "LS"
    params["GN_iter"] = args.newton_max + 1 if args.gauss_newton else 5
    params["cg_max_iter"] = args.cg_max
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    t_map, x = timed(lambda: solver.solve([None, prior.mean.copy(), None]))
    err_m = G.mtrue.copy().axpy(-1.0, x[PARAMETER]).norm("l2") / max(G.mtrue.norm("l2"), 1e-300)
    err_pr = G.mtrue.copy().axpy(-1.0, prior.mean).norm("l2") / max(G.mtrue.norm("l2"), 1e-300)
    c_tot, c_reg, c_mis = model.cost(x)
    say("  MAP: %.1f s, %d Newton its, %d CG its, %s; J %.4e (misfit %.4e, reg %.4e); "
        "rel error of m %.3f (prior mean: %.3f)" % (t_map, solver.it, solver.total_cg_iter,
                                                    solver.termination_reasons[solver.reason], c_tot, c_mis, c_reg, err_m, err_pr))
    # the anomaly is what the data can see: the error inside the target box, and the
    # correlation of MAP and truth there and everywhere (the global L2 error is dominated
    # by the prior sample's fine structure, which 60 boreholes were never going to resolve)
    xyz = G.Vm.coordinates()
    cc, rr = ANOMALY["centre"], 1.5 * ANOMALY["radius"]
    box = (np.abs(xyz[:, 0] - cc[0]) < rr) & (np.abs(xyz[:, 1] - cc[1]) < rr) & (np.abs(xyz[:, 2] - cc[2]) < 0.6 * rr)

    def stats(mask):
        t, m_ = G.mtrue.array[mask], x[PARAMETER].array[mask]
        s = np.array([np.sum((t - m_) ** 2), np.sum(t ** 2), np.sum(t * m_), np.sum(m_ ** 2), np.sum(t), np.sum(m_), mask.sum()], float)
        s = COMM.allreduce(s, op=MPI.SUM)
        nn = max(s[6], 1.0)
        cov = s[2] / nn - (s[4] / nn) * (s[5] / nn)
        vt, vm = s[1] / nn - (s[4] / nn) ** 2, s[3] / nn - (s[5] / nn) ** 2
        return float(np.sqrt(s[0] / max(s[1], 1e-300))), float(cov / max(np.sqrt(vt * vm), 1e-300)), float(s[4] / nn), float(s[5] / nn)

    err_box, corr_box, mean_true_box, mean_map_box = stats(box)
    err_all, corr_all, _, _ = stats(np.ones(box.size, bool))
    say("  anomaly box: mean log k truth %.3f, MAP %.3f; rel error %.3f, correlation %.3f (whole domain: %.3f, %.3f)"
        % (mean_true_box, mean_map_box, err_box, corr_box, err_all, corr_all))

    # ---- Laplace approximation
    t_hess, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=args.gauss_newton))
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    Omega = hm.MultiVector(x[PARAMETER], args.k + args.p)
    hm.parRandom.set_seed(99)
    hm.parRandom.normal_multivector(1.0, Omega)
    t_eig, (d, U) = timed(lambda: hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, args.k, s=1))
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    say("  Laplace: Hessian blocks %.1f s, eigensolver %.1f s; eigenvalues %.3e .. %.3e, %d above 1"
        % (t_hess, t_eig, d[0], d[-1], int((d > 1).sum())))

    # ---- samples, variance, traces, KL
    noise = prior.noise_vector()
    s_pr, s_po = G.Vm.vector(), G.Vm.vector()
    hm.parRandom.set_seed(7)
    acc = np.zeros(s_po.local_size)

    def draw():
        acc[:] = 0.0
        for _ in range(args.samples):
            prior.sample_noise(1.0, noise)
            post.sample(noise, s_pr, s_po, add_mean=False)
            acc[:] += s_po.array ** 2
        acc[:] /= args.samples
    t_samp, _ = timed(draw)
    t_var, (pv, prv2, corr) = timed(lambda: post.pointwise_variance(method="MonteCarlo", n=args.samples))
    t_tr, (tr_post, tr_pr, tr_corr) = timed(lambda: post.trace(method="Randomized", r=min(args.samples, 64)))
    kld = post.klDistanceFromPrior()
    below = COMM.allreduce(int(np.all(pv.array <= prv2.array + 1e-12)), op=MPI.MIN) == 1
    say("  samples: %d in %.1f s (%.0f ms each); MC variance %.1f s; traces %.1f s (posterior %.3e, prior %.3e); KL %.3e; posterior <= prior: %s"
        % (args.samples, t_samp, 1e3 * t_samp / args.samples, t_var, t_tr, tr_post, tr_pr, kld, below))
    std_post = float(np.sqrt(pv.sum() / pv.global_size))
    std_pr = float(np.sqrt(prv2.sum() / prv2.global_size))
    say("  pointwise std of log k: prior %.3f -> posterior %.3f (mean over dofs)" % (std_pr, std_post))
    # calibration of the Laplace posterior against the synthetic truth: fraction of dofs
    # where the truth lies within two posterior standard deviations of the MAP
    dev = np.abs(G.mtrue.array - x[PARAMETER].array)
    inside = COMM.allreduce(int(np.sum(dev <= 2.0 * np.sqrt(np.maximum(pv.array, 0.0)))), op=MPI.SUM)
    total = COMM.allreduce(int(dev.size), op=MPI.SUM)
    say("  truth within 2 posterior std of the MAP at %.1f%% of dofs" % (100.0 * inside / max(total, 1)))

    # The record and the field dump are written now, before the QoI stage, and again
    # after it, so a failure in that last stage cannot lose an hour-long MAP.
    q_true = q_map = q_std_lin = q_mean = q_std = q_t1 = q_t2 = s_t1 = s_t2 = q_th = e_th = float("nan")
    t_lin = t_qs = t_taylor = 0.0
    qvals = np.zeros(0)
    data_all = G.B.gather(G.data)          # in target order, on every rank

    def write_outputs():
        rec = {"host": platform.node(), "ranks": COMM.size, "device": args.device, "kernels": str(kernel_mod.device()),
               "gauss_newton": bool(args.gauss_newton),
               "problem": info, "k": args.k, "p": args.p, "samples": args.samples, "qoi_samples": args.qoi_samples,
               "t_build": t_build, "t_map": t_map, "newton_it": solver.it, "cg_it": solver.total_cg_iter,
               "J": c_tot, "misfit": c_mis, "reg": c_reg, "err_m": float(err_m), "err_prior_mean": float(err_pr),
               "err_box": err_box, "corr_box": corr_box, "corr_all": corr_all, "mean_true_box": mean_true_box, "mean_map_box": mean_map_box,
               "t_hess_blocks": t_hess, "t_eig": t_eig, "d_max": float(d[0]), "d_min": float(d[-1]), "n_above_one": int((d > 1).sum()),
               "eigenvalues": [float(v) for v in d], "t_samples": t_samp, "t_variance_mc": t_var, "t_trace": t_tr,
               "tr_post": tr_post, "tr_prior": tr_pr, "kl": float(kld), "std_prior": std_pr, "std_post": std_post,
               "coverage_2std": inside / max(total, 1), "qoi_true_K": T_SCALE * q_true, "qoi_map_K": T_SCALE * q_map,
               "qoi_std_linearized_K": T_SCALE * q_std_lin, "qoi_sample_mean_K": T_SCALE * q_mean,
               "qoi_sample_std_K": T_SCALE * q_std, "t_qoi_linearized": t_lin, "t_qoi_samples": t_qs,
               "qoi_taylor1_mean_K": T_SCALE * q_t1, "qoi_taylor1_std_K": T_SCALE * s_t1, "qoi_taylor2_mean_K": T_SCALE * q_t2,
               "qoi_taylor2_std_K": T_SCALE * s_t2, "taylor_k": args.taylor_k, "t_qoi_taylor": t_taylor,
               "qoi_taylor2_hutchinson_mean_K": T_SCALE * q_th, "qoi_taylor2_hutchinson_stderr_K": T_SCALE * e_th}
        if args.out and RANK == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(rec, f, indent=2)
        if args.dump:
            X, mt = gathered(G.Vm, G.mtrue)
            _, mm = gathered(G.Vm, x[PARAMETER])
            _, sd = gathered(G.Vm, pv.copy().assign(np.sqrt(np.maximum(pv.array, 0.0))))
            _, pstd = gathered(G.Vm, prv2.copy().assign(np.sqrt(np.maximum(prv2.array, 0.0))))
            if RANK == 0:
                np.savez(args.dump, xyz=X, mtrue=mt, mmap=mm, std_post=sd, std_prior=pstd, d=np.asarray(d),
                         targets=G.targets, data=data_all, qoi=np.asarray(qvals) * T_SCALE, n=args.n, ranks=COMM.size,
                         qoi_map_K=T_SCALE * q_map, qoi_std_lin_K=T_SCALE * q_std_lin, qoi_true_K=T_SCALE * q_true,
                         qoi_taylor2_mean_K=T_SCALE * q_th, qoi_taylor2_std_K=T_SCALE * s_t1)
    write_outputs()
    if args.skip_qoi:
        say("  (QoI stage skipped)")
        return 0

    # ---- the QoI: mean temperature in the target volume
    qoi = target_mean_qoi(G.Vu)
    t_lin, (q_std_lin, q_map, g) = timed(lambda: linearized_posterior_std(model, pde, qoi, post, x))
    q_true = qoi.eval([G.utrue, G.mtrue, None])
    t_qs, (q_mean, q_std, qvals) = timed(lambda: sampled_qoi(pde, qoi, post, x, args.qoi_samples, noise, s_pr, s_po))
    say("  QoI (mean temperature in the target volume): truth %.2f K, MAP %.2f K; linearized posterior std %.2f K (%.1f s); "
        "over %d posterior samples %.2f +- %.2f K (%.1f s)"
        % (T_SCALE * q_true, T_SCALE * q_map, T_SCALE * q_std_lin, t_lin, args.qoi_samples, T_SCALE * q_mean, T_SCALE * q_std, t_qs))
    if args.taylor_k > 0:
        t_taylor, (q_t1, q_t2, s_t1, s_t2, d_q, q_th, e_th) = timed(lambda: taylor_qoi(pde, qoi, post, k=args.taylor_k, n_trace=args.qoi_samples))
        say("  QoI second-order Taylor mean under the Laplace posterior (%.1f s): low-rank trace (k %d, eigenvalues %.1e .. %.1e) "
            "%.2f K, i.e. %+.3f K over the MAP; Hutchinson trace (%d posterior samples, %d QoI-Hessian applies) %.2f +- %.2f K, "
            "i.e. %+.2f K over the MAP; the sample mean was %+.2f K over the MAP"
            % (t_taylor, args.taylor_k, d_q[0], d_q[-1], T_SCALE * q_t2, T_SCALE * (q_t2 - q_t1), args.qoi_samples, args.qoi_samples,
               T_SCALE * q_th, T_SCALE * e_th, T_SCALE * (q_th - q_t1), T_SCALE * (q_mean - q_map)))

    write_outputs()
    if args.paraview:
        from hippymfem.fem.io import write_paraview
        write_paraview(args.paraview, G.pmesh, {"m_true": (G.Vm, G.mtrue), "m_map": (G.Vm, x[PARAMETER]),
                                                "std_post": (G.Vm, pv.copy().assign(np.sqrt(np.maximum(pv.array, 0.0)))),
                                                "u_map": (G.Vu, x[STATE])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

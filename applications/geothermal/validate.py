#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Checks of the geothermal application, each a subcommand.

    python -m applications.geothermal.validate fd        --n 16   # FD slopes, Hessian symmetry
    python -m applications.geothermal.validate forward   --n 32   # Newton convergence of the forward solve
    python -m applications.geothermal.validate variance  --n 16   # MC and sample variance against the exact one
    python -m applications.geothermal.validate partition a.npz b.npz ...   # dumps of run.py at different rank counts

The first three build the model (mesh, truth, data) exactly as ``run.py`` does and run
under MPI; ``partition`` is a serial numpy comparison of ``run.py --dump`` files.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _say(comm, *a):
    if comm.rank == 0:
        print(*a, flush=True)


def check_fd(args, comm):
    import hippymfem as hm
    from hippymfem.modeling.modelVerify import modelVerify, best_slope
    from hippymfem.modeling.variables import PARAMETER
    from applications.geothermal.model import Geothermal

    G = Geothermal(args.n, comm, order=args.order)
    model = G.model
    # The reduced Hessian is symmetric only to the accuracy of the incremental solves
    # (GMRES, 1e-10 in the model: measured asymmetry 4.2e-10 at 16^3), so they are
    # tightened here to make the check a statement about the derivatives.
    for attr in ("solver_fwd_inc", "solver_adj_inc"):
        getattr(G.pde, attr).parameters["rel_tolerance"] = args.inc_tol
    # at the truth: the forward map is nonlinear there (T_max ~ 120 K, k(T) 15% below k(0))
    m0 = G.mtrue.copy()
    eps = np.power(0.5, np.arange(3, 27))
    t0 = time.perf_counter()
    eps, eg, eh = modelVerify(model, m0, is_quadratic=False, misfit_only=False, verbose=(comm.rank == 0), eps=eps)
    sg, sh = best_slope(eps, eg), best_slope(eps, eh)
    # symmetry of the full reduced Hessian at the same point
    x = model.generate_vector()
    x[PARAMETER] = m0
    model.solveFwd(x[0], x)
    model.solveAdj(x[2], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    H = hm.ReducedHessian(model, misfit_only=False)
    hm.parRandom.set_seed(3)
    h1, h2 = model.generate_vector(PARAMETER), model.generate_vector(PARAMETER)
    hm.parRandom.normal(1.0, h1)
    hm.parRandom.normal(1.0, h2)
    Hh1, Hh2 = h1.duplicate(), h2.duplicate()
    H.mult(h1, Hh1)
    H.mult(h2, Hh2)
    a, b = h2.inner(Hh1), h1.inner(Hh2)
    sym = abs(a - b) / max(abs(a), abs(b), 1e-300)
    ok = abs(sg - 1.0) < 0.05 and abs(sh - 1.0) < 0.05 and sym < 1e-9
    _say(comm, "fd %d^3: gradient slope %.4f, Hessian slope %.4f, Hessian asymmetry %.2e (%.1f s): %s"
         % (args.n, sg, sh, sym, time.perf_counter() - t0, "OK" if ok else "FAILED"))
    return {"check": "fd", "n": args.n, "slope_grad": sg, "slope_hess": sh, "hess_asym": sym, "ok": bool(ok),
            "eps": eps.tolist(), "err_grad": eg.tolist(), "err_hess": eh.tolist()}


def check_forward(args, comm):
    from hippymfem.modeling.variables import ADJOINT
    from applications.geothermal.model import Geothermal, T_SCALE

    G = Geothermal(args.n, comm, order=args.order)
    pde = G.pde
    pde.newton_parameters["print_level"] = 0 if comm.rank == 0 else -1
    u = pde.generate_state()
    # the residual at the initial guess (zero state with the boundary data), for the reduction
    u0 = pde.generate_state()
    pde.bc.apply(u0)
    r0 = pde._residual([u0, G.mtrue, None], ADJOINT, ess=pde.bc0.ess).norm("l2")
    comm.Barrier()
    t0 = time.perf_counter()
    pde.solveFwd(u, [u, G.mtrue, None])
    comm.Barrier()
    t = time.perf_counter() - t0
    r1 = pde._residual([u, G.mtrue, None], ADJOINT, ess=pde.bc0.ess).norm("l2")
    its = int(pde.fwd_iterations)
    ok = its <= 6 and r1 <= 1e-9 * r0
    _say(comm, "forward %d^3: %d Newton iterations, ||r|| %.3e -> %.3e (%.1e relative), T_max %.0f K, %.1f s: %s"
         % (args.n, its, r0, r1, r1 / r0, T_SCALE * u.norm("linf"), t, "OK" if ok else "FAILED"))
    return {"check": "forward", "n": args.n, "newton_its": its, "r0": r0, "r1": r1, "t": t, "ok": bool(ok)}


def check_variance(args, comm):
    import hippymfem as hm
    from mpi4py import MPI
    from hippymfem.modeling.variables import PARAMETER
    from applications.geothermal.model import Geothermal

    G = Geothermal(args.n, comm, order=args.order)
    model, prior = G.model, G.prior
    # a Laplace approximation at the truth (the MAP is not needed for the estimator check)
    x = model.generate_vector()
    x[PARAMETER] = G.mtrue.copy()
    model.solveFwd(x[0], x)
    model.solveAdj(x[2], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hm = hm.ReducedHessian(model, misfit_only=True)
    Omega = hm.MultiVector(x[PARAMETER], args.k + args.p)
    hm.parRandom.set_seed(99)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hm, prior.R, prior.Rsolver, Omega, args.k, s=1)
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    t0 = time.perf_counter()
    ex, ex_pr, corr = post.pointwise_variance(method="Exact")
    t_ex = time.perf_counter() - t0
    t0 = time.perf_counter()
    mc, mc_pr, _ = post.pointwise_variance(method="MonteCarlo", n=args.samples)
    t_mc = time.perf_counter() - t0
    # the sample variance of actual posterior draws
    noise = prior.noise_vector()
    s_pr, s_po = G.Vm.vector(), G.Vm.vector()
    hm.parRandom.set_seed(7)
    acc = np.zeros(s_po.local_size)
    t0 = time.perf_counter()
    for _ in range(args.samples):
        prior.sample_noise(1.0, noise)
        post.sample(noise, s_pr, s_po, add_mean=False)
        acc += s_po.array ** 2
    acc /= args.samples
    t_s = time.perf_counter() - t0

    def rel(a, b):
        num = comm.allreduce(float(np.max(np.abs(a - b))), op=MPI.MAX)
        den = comm.allreduce(float(np.max(np.abs(b))), op=MPI.MAX)
        return num / max(den, 1e-300)

    def mean(v):
        return comm.allreduce(float(np.sum(v)), op=MPI.SUM) / comm.allreduce(int(np.size(v)), op=MPI.SUM)

    r_mc = rel(mc.array, ex.array)
    r_s = rel(acc, ex.array)
    ratio_mc = mean(mc.array) / mean(ex.array)
    ratio_s = mean(acc) / mean(ex.array)
    # 3 sigma of the mean of n chi-square(1) variables, relative
    tol_ratio = 3.0 * np.sqrt(2.0 / args.samples)
    ok = abs(ratio_mc - 1.0) < tol_ratio and abs(ratio_s - 1.0) < tol_ratio and r_s < 0.35 and r_mc < 0.35
    _say(comm, "variance %d^3 (k %d, %d samples): exact %.1f s; MC/exact mean ratio %.4f, max rel %.3f (%.1f s); "
         "sample/exact mean ratio %.4f, max rel %.3f (%.1f s); prior->posterior mean var %.4f -> %.4f: %s"
         % (args.n, args.k, args.samples, t_ex, ratio_mc, r_mc, t_mc, ratio_s, r_s, t_s,
            mean(ex_pr.array), mean(ex.array), "OK" if ok else "FAILED"))
    return {"check": "variance", "n": args.n, "k": args.k, "samples": args.samples, "ratio_mc": ratio_mc, "maxrel_mc": r_mc,
            "ratio_sample": ratio_s, "maxrel_sample": r_s, "t_exact": t_ex, "t_mc": t_mc, "t_samples": t_s,
            "var_prior_mean": mean(ex_pr.array), "var_post_mean": mean(ex.array), "ok": bool(ok)}


def check_partition(files, tol=1e-12):
    """Compare ``run.py --dump`` files: truth, data, MAP, posterior std, eigenvalues."""
    ref = None
    z0 = np.load(files[0])
    out = {"check": "partition", "files": list(files), "rows": [], "n": int(z0["n"]),
           "where": "host ranks" if "host" in os.path.basename(files[0]) else "cards"}
    for f in files:
        z = np.load(f)
        # sort on integer grid keys: the coordinates carry rank-dependent round-off, and
        # an exact lexsort on them scrambled equal rows (a 1.0 "difference" at 2 ranks)
        n = int(z["n"])
        g = np.rint(z["xyz"] * n * 8).astype(np.int64)
        order = np.lexsort((g[:, 0], g[:, 1], g[:, 2]))
        cur = {"xyz": z["xyz"][order], "mtrue": z["mtrue"][order], "mmap": z["mmap"][order],
               "std_post": z["std_post"][order], "d": z["d"], "data": z["data"] if "data" in z.files else None,
               "ranks": int(z["ranks"]) if "ranks" in z.files else -1}
        if ref is None:
            ref = cur
            print("%-40s %6s %12s %12s %12s %12s %12s" % ("file", "ranks", "xyz", "data", "truth", "MAP", "post std / eig"))
            print("%-40s %6d %12s" % (os.path.basename(f), cur["ranks"], "reference"))
            continue
        row = {"file": f, "ranks": cur["ranks"]}
        for key in ("xyz", "data", "mtrue", "mmap", "std_post", "d"):
            a, b = cur[key], ref[key]
            if a is None or b is None or np.shape(a) != np.shape(b):
                row[key] = float("nan")
                continue
            row[key] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
        print("%-40s %6d %12.2e %12.2e %12.2e %12.2e %12.2e / %.2e"
              % (os.path.basename(f), cur["ranks"], row["xyz"], row["data"], row["mtrue"], row["mmap"], row["std_post"], row["d"]))
        # The random streams are partition independent, the Krylov solves are not: the
        # prior sample (an A-solve at 1e-8) and the data agree to that tolerance, the MAP
        # (Newton-CG on top) to about 1e-5, and the truncated eigen-decomposition and the
        # posterior std it gives only to the eigensolver's own accuracy (0.1 at k = 20,
        # 16^3, where more than k eigenvalues are above one).  The pass criterion is the
        # stream and the solves; the rest is reported.
        row["ok"] = bool(all(np.isnan(row[k]) or row[k] <= tol for k in ("xyz", "data", "mtrue")) and
                         (np.isnan(row["mmap"]) or row["mmap"] <= 1e3 * tol))
        out["rows"].append(row)
    out["ok"] = bool(all(r["ok"] for r in out["rows"]))
    print("partition check (data and truth to %.0e, MAP to %.0e): %s" % (tol, 1e3 * tol, "OK" if out["ok"] else "FAILED"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("check", choices=["fd", "forward", "variance", "partition"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--tol", type=float, default=1e-6, help="partition: data and truth (Krylov solves at 1e-8)")
    ap.add_argument("--inc-tol", type=float, default=1e-13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.check == "partition":
        rec = check_partition(args.files, args.tol)
        rank = 0
    else:
        import hippymfem as hm
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
        hm.configure_device(args.device, comm, quiet=(rank != 0))
        rec = {"fd": check_fd, "forward": check_forward, "variance": check_variance}[args.check](args, comm)
        rec["ranks"] = comm.size
        rec["device"] = args.device
    if args.out and rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
    return 0 if rec.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

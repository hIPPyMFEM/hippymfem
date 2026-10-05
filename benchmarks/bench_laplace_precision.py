#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Laplace stages with single- against double-precision solves, at one MAP point.

The problem of ``benchmarks/bench_laplace.py`` (n^3 second-order hexahedra,
``exp(m) grad u . grad p``, BiLaplacian prior, 200 pointwise observations, 1 % noise).
The MAP point is computed once (or loaded from ``--map-file``) and every configuration
runs the Laplace stages at it:

  name:precision:inc_tol:single:reps      e.g.  ref:fp64:1e-10:0:1  sgl:mixed:1e-8:1:3

``precision`` is the kernels' (fp64 or mixed), ``single`` whether the solves run in the
single-precision hypre (HIPPYMFEM_HYPRE_SINGLE must name it), ``inc_tol`` the relative
tolerance of the incremental solves (floored at 1e-5 in single precision).  The stages:
doublePassG (k, p), posterior samples, pointwise variance (randomized and Monte Carlo),
traces; every random draw is seeded the same in every configuration, so differences
come from the eigenpairs alone.  The first configuration is the reference for the
errors; at the end the reference operator is applied to every other configuration's
eigenvectors (U^T H_ref U against diag(d)).  One rank.
"""

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hm  # noqa: E402
from hippymfem.fem import kernel as km
from hippymfem.algorithms import singlesolve
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

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


class Counter:
    """Wall time, calls and returned iteration counts of a wrapped method."""

    def __init__(self):
        self.t = collections.defaultdict(float)
        self.n = collections.defaultdict(int)
        self.its = collections.defaultdict(int)

    def wrap(self, obj, name, label, count_its=False):
        f = getattr(obj, name)

        def g(*a, **k):
            t0 = time.perf_counter()
            out = None
            try:
                out = f(*a, **k)
                return out
            finally:
                self.t[label] += time.perf_counter() - t0
                self.n[label] += 1
                if count_its and isinstance(out, (int, np.integer)):
                    self.its[label] += int(out)

        setattr(obj, name, g)

    def snap(self):
        return {k: (self.t[k], self.n[k], self.its[k]) for k in self.t}

    @staticmethod
    def delta(a, b):
        return {k: [a[k][0] - b.get(k, (0, 0, 0))[0], a[k][1] - b.get(k, (0, 0, 0))[1],
                    a[k][2] - b.get(k, (0, 0, 0))[2]]
                for k in a if a[k][1] - b.get(k, (0, 0, 0))[1] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--prior-tol", type=float, default=1e-8)
    ap.add_argument("--map-file", default=None, help="npy of the MAP parameter (1 rank); computed and saved if absent")
    ap.add_argument("--map-mode", default="mixed", help="kernels' precision for the MAP computation (single solves "
                    "if the library is named and mode is mixed)")
    ap.add_argument("--configs", default="ref:fp64:1e-10:0:1,dbl:fp64:1e-8:0:3,sgl:mixed:1e-8:1:3,dbl5:fp64:1e-5:0:1")
    ap.add_argument("--diag", default="dbl,sgl,dbl5", help="configurations whose U meet the reference operator")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="npz with d, correction and variance fields of every configuration")
    args = ap.parse_args()
    assert COMM.size == 1, "one rank"

    hm.configure_device("cuda" if hm.config.hypre_device else "cpu", COMM, quiet=(RANK != 0))
    N = args.n
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, 2)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    say("%d^3: %d state dofs, %d parameter dofs; single library %s"
        % (N, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), singlesolve.HYPRE_SINGLE or "-"))
    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    for sol in (getattr(prior, "Asolver", None), getattr(prior, "Msolver", None)):
        if sol is not None and hasattr(sol, "parameters"):
            sol.parameters["rel_tolerance"] = args.prior_tol
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, 200) for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    # the data in double precision, as in the benchmarks
    km.set_precision("fp64")
    pde.single_solves = False
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)

    def activate(prec, single, inc_tol):
        km.set_precision(prec)
        pde.single_solves = bool(single)
        pde.invalidate_jacobian()
        pde.release_linearization_point()
        pde._release_operators("solver", "solver_adj")
        for attr in ("solver_fwd_inc", "solver_adj_inc"):
            getattr(pde, attr).parameters["rel_tolerance"] = inc_tol

    rec = {"n": N, "k": args.k, "p": args.p, "samples": args.samples, "r": args.r, "prior_tol": args.prior_tol,
           "single_library": singlesolve.HYPRE_SINGLE, "configs": {}}
    # ---- MAP
    m = Vm.vector()
    if args.map_file and os.path.exists(args.map_file):
        m.array[:] = np.load(args.map_file)
        say("MAP loaded from %s" % args.map_file)
        rec["map"] = {"loaded": args.map_file}
    else:
        activate(args.map_mode, args.map_mode == "mixed", 1e-6)
        params = hm.ReducedSpaceNewtonCG_ParameterList()
        params["rel_tolerance"] = 1e-6
        params["abs_tolerance"] = 1e-12
        params["max_iter"] = 25
        params["globalization"] = "LS"
        params["GN_iter"] = 5
        params["cg_max_iter"] = 50
        params["print_level"] = -1
        solver = hm.ReducedSpaceNewtonCG(model, params)
        t_map, xs = timed(lambda: solver.solve([None, prior.mean.copy(), None]))
        m.assign(xs[PARAMETER])
        say("MAP (%s): %.1f s, %d Newton, %d CG, J %.10e, %s"
            % (args.map_mode, t_map, solver.it, solver.total_cg_iter, solver.final_cost,
               solver.termination_reasons[solver.reason]))
        rec["map"] = {"mode": args.map_mode, "t": t_map, "newton": solver.it, "cg": solver.total_cg_iter,
                      "J": float(solver.final_cost)}
        if args.map_file:
            os.makedirs(os.path.dirname(os.path.abspath(args.map_file)), exist_ok=True)
            np.save(args.map_file, m.array.copy())
        del xs, solver

    cnt = Counter()
    cnt.wrap(pde.solver_fwd_inc, "solve", "inc fwd", count_its=True)
    cnt.wrap(pde.solver_adj_inc, "solve", "inc adj", count_its=True)
    cnt.wrap(prior.Rsolver, "solve", "prior R^-1")

    def linearize():
        x = [model.generate_vector(STATE), m.copy(), model.generate_vector(ADJOINT)]
        model.solveFwd(x[STATE], x)
        model.solveAdj(x[ADJOINT], x)
        model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        H = hm.ReducedHessian(model, misfit_only=True)
        cnt.wrap(H, "mult", "Hessian apply")
        return x, H

    results = {}
    dump = {}
    for spec in [s for s in args.configs.split(",") if s]:
        name, prec, tol, single, reps = spec.split(":")
        tol, single, reps = float(tol), single == "1", int(reps)
        activate(prec, single, tol)
        t_lin, (x, H) = timed(linearize)
        A = pde.solver_fwd_inc.A
        kind = type(A).__name__
        say("=== %s: kernels %s, solves %s (%s), incremental tolerance %.0e, %d repeats; linearization %.2f s"
            % (name, prec, "single" if single else "double", kind, tol, reps, t_lin))
        out = {"precision": prec, "single": single, "inc_tol": tol, "matrix": kind, "t_linearize": t_lin,
               "reps": reps, "t_eig": [], "t_samples": [], "t_var_rand": [], "t_var_mc": [], "t_trace": []}
        for rep in range(reps):
            hm.parRandom.set_seed(99)
            Omega = hm.MultiVector(x[PARAMETER], args.k + args.p)
            hm.parRandom.normal_multivector(1.0, Omega)
            b0 = cnt.snap()
            t_eig, (d, U) = timed(lambda: hm.doublePassG(H, prior.R, prior.Rsolver, Omega, args.k, s=1))
            parts = Counter.delta(cnt.snap(), b0)
            out["t_eig"].append(t_eig)
            out["eig_parts"] = parts
            post = hm.GaussianLRPosterior(prior, d, U, mean=m)
            s_pr, s_po = Vm.vector(), Vm.vector()
            acc = np.zeros(m.local_size)

            def draw():
                acc[:] = 0.0
                hm.parRandom.set_seed(7)
                for _ in range(args.samples):
                    prior.sample_noise(1.0, noise)
                    post.sample(noise, s_pr, s_po, add_mean=False)
                    acc[:] += s_po.array ** 2
                acc[:] /= args.samples
            t_s, _ = timed(draw)
            out["t_samples"].append(t_s)

            def var_rand():
                hm.parRandom.set_seed(8)
                return post.pointwise_variance(method="Randomized", r=args.r)
            t_v, (pv, prv, corr) = timed(var_rand)
            out["t_var_rand"].append(t_v)

            def var_mc():
                hm.parRandom.set_seed(9)
                return post.pointwise_variance(method="MonteCarlo", n=args.samples)
            t_mc, (pv_mc, prv_mc, corr_mc) = timed(var_mc)
            out["t_var_mc"].append(t_mc)

            def traces():
                hm.parRandom.set_seed(10)
                return post.trace(method="Randomized", r=args.r)
            t_tr, (tr_post, tr_pr, tr_corr) = timed(traces)
            out["t_trace"].append(t_tr)
            say("  rep %d: eigensolver %.2f s (Hessian applies %d, %.1f ms each; incremental solves %.1f + %.1f "
                "its each), samples %.2f s, variance %.2f s (randomized) %.2f s (MC), traces %.2f s"
                % (rep, t_eig, parts["Hessian apply"][1], 1e3 * parts["Hessian apply"][0] / parts["Hessian apply"][1],
                   parts["inc fwd"][2] / max(parts["inc fwd"][1], 1), parts["inc adj"][2] / max(parts["inc adj"][1], 1),
                   t_s, t_v, t_mc, t_tr))
        for key in ("t_eig", "t_samples", "t_var_rand", "t_var_mc", "t_trace"):
            out[key + "_median"] = float(np.median(out[key]))
        out.update({"d": [float(v) for v in d], "tr_post": tr_post, "tr_prior": tr_pr, "tr_corr": tr_corr,
                    "inc_its_fwd": parts["inc fwd"][2] / max(parts["inc fwd"][1], 1),
                    "inc_its_adj": parts["inc adj"][2] / max(parts["inc adj"][1], 1),
                    "t_hessian_apply": parts["Hessian apply"][0] / parts["Hessian apply"][1]})
        BU = hm.MultiVector(U[0], U.nvec())
        hm.MatMvMult(prior.R, U, BU)
        out["R_orthonormality"] = float(np.abs(U.dot_mv(BU) - np.eye(U.nvec())).max())
        say("  eigenvalues %.4e .. %.4e; traces post %.6e prior %.6e corr %.6e; R-orthonormality %.1e"
            % (d[0], d[-1], tr_post, tr_pr, tr_corr, out["R_orthonormality"]))
        results[name] = {"d": np.asarray(d).copy(), "U": U, "corr": corr.array.copy(), "pv": pv.array.copy(),
                         "prv": prv.array.copy(), "pv_mc": pv_mc.array.copy(), "acc": acc.copy(),
                         "tr": (tr_post, tr_pr, tr_corr)}
        dump.update({name + "_d": np.asarray(d), name + "_corr": corr.array.copy(), name + "_pv": pv.array.copy(),
                     name + "_pvmc": pv_mc.array.copy()})
        rec["configs"][name] = out
        del x, H, post

    # ---- errors against the first configuration
    names = list(results)
    ref = results[names[0]]
    for name in names[1:]:
        r = results[name]
        o = rec["configs"][name]
        rel_d = np.abs(r["d"] - ref["d"]) / np.abs(ref["d"])
        dcorr = r["corr"] - ref["corr"]
        pv_ref = ref["pv"]
        relv = np.abs(dcorr) / np.maximum(np.abs(pv_ref), 1e-300)
        relv_mc = np.abs(r["pv_mc"] - ref["pv_mc"]) / np.maximum(np.abs(ref["pv_mc"]), 1e-300)
        o["err"] = {"eig_rel_max": float(rel_d.max()), "eig_rel_last": float(rel_d[-1]),
                    "eig_rel_first": float(rel_d[0]), "eig_rel": [float(v) for v in rel_d],
                    "var_rel_max": float(relv.max()), "var_rel_rms": float(np.sqrt(np.mean(relv ** 2))),
                    "var_mc_rel_max": float(relv_mc.max()), "var_mc_rel_rms": float(np.sqrt(np.mean(relv_mc ** 2))),
                    "corr_rel_max": float(np.abs(dcorr).max() / np.abs(ref["corr"]).max()),
                    "tr_post_rel": abs(r["tr"][0] - ref["tr"][0]) / abs(ref["tr"][0]),
                    "tr_corr_rel": abs(r["tr"][2] - ref["tr"][2]) / abs(ref["tr"][2])}
        e = o["err"]
        say("%s against %s: eigenvalues rel. error max %.2e (first %.2e, last %.2e); posterior variance rel. "
            "error max %.2e rms %.2e (Monte Carlo field: max %.2e rms %.2e); traces: posterior %.2e, correction %.2e"
            % (name, names[0], e["eig_rel_max"], e["eig_rel_first"], e["eig_rel_last"], e["var_rel_max"],
               e["var_rel_rms"], e["var_mc_rel_max"], e["var_mc_rel_rms"], e["tr_post_rel"], e["tr_corr_rel"]))
        say("  eigenvalue rel. errors: " + " ".join("%.1e" % v for v in rel_d))

    # ---- the reference operator against the other configurations' eigenvectors
    diag_names = [n for n in args.diag.split(",") if n in results and n != names[0]]
    if diag_names:
        name0 = names[0]
        _, prec, tol, single, _ = [s for s in args.configs.split(",") if s.startswith(name0 + ":")][0].split(":")
        activate(prec, single == "1", float(tol))
        x, H = linearize()
        for name in [name0] + diag_names:
            U = results[name]["U"]
            d = results[name]["d"]
            HU = hm.MultiVector(U[0], U.nvec())
            hm.MatMvMult(H, U, HU)
            E = HU.dot_mv(U)
            E = 0.5 * (E + E.T)
            dd = np.diag(E)
            off = E - np.diag(dd)
            scale = np.sqrt(np.outer(np.abs(d), np.abs(d)))
            o = rec["configs"][name].setdefault("err", {})
            o["rayleigh_rel"] = [float(v) for v in np.abs(dd - d) / np.abs(d)]
            o["rayleigh_rel_max"] = float((np.abs(dd - d) / np.abs(d)).max())
            o["offdiag_rel_max"] = float((np.abs(off) / scale).max())
            o["diagonalization"] = float(np.abs(E - np.diag(d)).max() / np.abs(d).max())
            # eigenvalues of the reference operator projected on this U (Rayleigh-Ritz)
            ritz = np.sort(np.linalg.eigvalsh(E))[::-1]
            o["ritz_vs_ref_rel_max"] = float((np.abs(ritz - results[name0]["d"]) / results[name0]["d"]).max())
            say("%s: U^T H_ref U against diag(d): diagonal rel. max %.2e, off-diagonal (scaled by sqrt(d_i d_j)) max "
                "%.2e, max|E - D|/d_max %.2e; Ritz values of H_ref on span(U) against the reference eigenvalues: "
                "max rel %.2e" % (name, o["rayleigh_rel_max"], o["offdiag_rel_max"], o["diagonalization"],
                                  o["ritz_vs_ref_rel_max"]))
    for name in names:
        o = rec["configs"][name]
        say("%-6s eig %.2f s, samples %.2f s, variance %.2f s / %.2f s (MC), traces %.2f s (medians); Hessian apply "
            "%.1f ms; incremental its %.1f / %.1f"
            % (name, o["t_eig_median"], o["t_samples_median"], o["t_var_rand_median"], o["t_var_mc_median"],
               o["t_trace_median"], 1e3 * o["t_hessian_apply"], o["inc_its_fwd"], o["inc_its_adj"]))
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        say("wrote %s" % args.out)
    if args.dump and RANK == 0:
        np.savez(args.dump, **dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

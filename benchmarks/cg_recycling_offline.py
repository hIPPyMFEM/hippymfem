#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Reusing the CG of Newton step k in step k+1: feasibility, offline (64^3, one rank).

Newton-CG on the problem of ``bench_precision.py`` runs as usual and takes its own steps.
The CG of every step records its coefficients and its (reorthogonalized) preconditioned
residuals.  From them, at the end of step k, the Lanczos relation gives Ritz pairs
(theta_i, w_i) of the prior-preconditioned Hessian, R-orthonormal.  At step k+1 the same
Newton system (same operator, right-hand side, tolerance and preconditioner) is solved
again, its result discarded, with

* ``lmp_<c>``: the spectral limited-memory preconditioner
  P r = R^{-1} r + sum_i (1/theta_i - 1) w_i (w_i . r), over the Ritz pairs of step k
  whose residual estimate is below c * theta_i (and theta_i > 1.05);
* ``init_<c>``: R^{-1} alone from the deflated initial guess x0 = W Theta^{-1} W^T b;

each stopping on the R^{-1}-norm of the residual as the step's own CG does (the part of
P r that is R^{-1} r gives it without another solve).  Reported per step: the CG count of
the step and of each variant, the number of Ritz pairs used, and how far each variant's
solution is from the step's (R-norm, relative).
"""

import argparse
import json
import math
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
from hippymfem.algorithms import NewtonCG as ncg
from hippymfem.algorithms.cgsolverSteihaug import CGSolverSteihaug
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

COMM = MPI.COMM_WORLD


def say(*a):
    if COMM.rank == 0:
        print(*a, flush=True)


LOG = []
STATE_RITZ = {"pairs": None}
CUTS = (1e-2, 1e-1)
PRIOR = {}


class Counted:
    """An operator whose applications are counted."""

    def __init__(self, A):
        self.A, self.n = A, 0

    def mult(self, x, y):
        self.n += 1
        return self.A.mult(x, y)


def pcg(A, Binv, b, tol, maxit, W=(), theta=(), x0=None, reorth=True):
    """Preconditioned CG, residuals orthogonalized; preconditioner R^{-1} plus the
    spectral correction over (W, theta); stops when r.R^{-1}r <= tol^2 b.R^{-1}b."""
    A = Counted(A)
    coef = [1.0 / t - 1.0 for t in theta]

    def prec(r):
        z = r.duplicate()
        Binv.solve(z, r)
        plain = r.inner(z)
        for w, c in zip(W, coef):
            z.axpy(c * w.inner(r), w)
        return z, plain

    x = b.duplicate()
    x.zero()
    r = b.copy()
    zb, ref = prec(b)
    if x0 is not None:
        x.axpy(1.0, x0)
        Ax = b.duplicate()
        A.mult(x0, Ax)
        r.axpy(-1.0, Ax)
        z, plain = prec(r)
    else:
        z, plain = zb, ref
    target = tol * tol * ref
    if plain <= target:
        return x, A.n
    d = z.copy()
    nom = r.inner(z)
    basis = [(r.copy(), z.copy(), nom)]
    Ad = b.duplicate()
    it = 0
    while it < maxit:
        A.mult(d, Ad)
        den = d.inner(Ad)
        if den <= 0.0:
            break
        alpha = nom / den
        x.axpy(alpha, d)
        r.axpy(-alpha, Ad)
        if reorth:
            for rj, zj, rho in basis:
                r.axpy(-r.inner(zj) / rho, rj)
        z, plain = prec(r)
        betanom = r.inner(z)
        it += 1
        if plain <= target:
            break
        basis.append((r.copy(), z.copy(), betanom))
        d.scale(betanom / nom)
        d.axpy(1.0, z)
        nom = betanom
    return x, A.n


def ritz(alphas, rhos, Z):
    """Ritz pairs of B^{-1}A from a CG run: T from the coefficients, V = z_j / sqrt(rho_j);
    returns theta (descending), the residual estimates, and a function giving w_i."""
    m = len(alphas)
    if m == 0:
        return np.zeros(0), np.zeros(0), None
    betas = [rhos[j] / rhos[j - 1] for j in range(1, len(rhos))]       # beta_1 .. beta_m
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j > 0 else 0.0)
        if j > 0:
            T[j, j - 1] = T[j - 1, j] = -math.sqrt(betas[j - 1]) / alphas[j - 1]
    th, Q = np.linalg.eigh(T)
    order = np.argsort(th)[::-1]
    th, Q = th[order], Q[:, order]
    nxt = math.sqrt(betas[m - 1]) / alphas[m - 1] if len(betas) >= m else 0.0
    res = np.abs(nxt * Q[m - 1, :])

    def vec(i):
        w = Z[0].duplicate()
        w.zero()
        for j in range(m):
            w.axpy(Q[j, i] / math.sqrt(rhos[j]), Z[j])
        return w
    return th, res, vec


class RecordingCG(CGSolverSteihaug):
    """The CG of a Newton step (line search, no trust region), recording its
    coefficients, then the offline variants on the same system."""

    def _solve(self, x, b):
        self.iter = 0
        self.converged = False
        self.reasonid = 0
        alphas, rhos, Z = [], [], []
        self.r.zero()
        self.r.axpy(1.0, b)
        x.zero()
        self.z.zero()
        self.B_solver.solve(self.z, self.r)
        self.d.zero()
        self.d.axpy(1.0, self.z)
        nom0 = self.d.inner(self.r)
        nom = nom0
        rhos.append(nom)
        Z.append(self.z.copy())
        basis = [(self.r.copy(), self.z.copy(), nom)]
        rtol2 = nom * self.parameters["rel_tolerance"] ** 2
        r0 = max(rtol2, self.parameters["abs_tolerance"] ** 2)
        self._rec = (alphas, rhos, Z)
        if nom <= r0:
            self.converged, self.reasonid = True, 1
            return self.iter
        self.A.mult(self.d, self.Ad)
        den = self.Ad.inner(self.d)
        if den <= 0.0:
            self._negative_curvature(x, first=True)
            return self.iter
        self.iter = 1
        while True:
            alpha = nom / den
            alphas.append(alpha)
            x.axpy(alpha, self.d)
            self.r.axpy(-alpha, self.Ad)
            for rj, zj, rho in basis:
                self.r.axpy(-self.r.inner(zj) / rho, rj)
            self.B_solver.solve(self.z, self.r)
            betanom = self.r.inner(self.z)
            rhos.append(betanom)
            if betanom < r0:
                self.converged, self.reasonid = True, 1
                self.final_norm = math.sqrt(max(betanom, 0.0))
                break
            self.iter += 1
            if self.iter > self.parameters["max_iter"]:
                self.converged, self.reasonid = False, 0
                self.final_norm = math.sqrt(max(betanom, 0.0))
                break
            basis.append((self.r.copy(), self.z.copy(), betanom))
            Z.append(self.z.copy())
            self.d.scale(betanom / nom)
            self.d.axpy(1.0, self.z)
            self.A.mult(self.d, self.Ad)
            den = self.d.inner(self.Ad)
            if den <= 0.0:
                self._negative_curvature(x, first=False)
                break
            nom = betanom
        return self.iter

    def solve(self, x, b):
        t0 = time.perf_counter()
        n0 = self.A.ncalls if hasattr(self.A, "ncalls") else 0
        it = super().solve(x, b)
        base = (self.A.ncalls - n0) if hasattr(self.A, "ncalls") else it
        t_base = time.perf_counter() - t0
        tol = float(self.parameters["rel_tolerance"])
        maxit = int(self.parameters["max_iter"])
        entry = {"step": len(LOG) + 1, "cg": base, "tol": tol, "reason": self.reasonid, "t_cg": t_base}
        R = PRIOR["R"]

        def rdist(y):
            dy = y.copy().axpy(-1.0, x)
            Rd, Rx = x.duplicate(), x.duplicate()
            R.mult(dy, Rd)
            R.mult(x, Rx)
            return math.sqrt(max(dy.inner(Rd), 0.0) / max(x.inner(Rx), 1e-300))

        pairs = STATE_RITZ["pairs"]
        if pairs is not None:
            th, res, W = pairs
            for c in CUTS:
                sel = [i for i in range(len(th)) if res[i] <= c * th[i] and th[i] > 1.05]
                Ws, ths = [W[i] for i in sel], [th[i] for i in sel]
                y, n = pcg(self.A, self.B_solver, b, tol, maxit, Ws, ths)
                entry["lmp_%g" % c] = {"cg": n, "pairs": len(sel), "dist": rdist(y)}
                x0 = b.duplicate()
                x0.zero()
                for w, t in zip(Ws, ths):
                    x0.axpy(w.inner(b) / t, w)
                y, n = pcg(self.A, self.B_solver, b, tol, maxit, x0=x0)
                entry["init_%g" % c] = {"cg": n, "pairs": len(sel), "dist": rdist(y)}
        # Ritz pairs of this step for the next (only from a CG that converged on its test)
        alphas, rhos, Z = self._rec
        if self.reasonid == 1 and len(alphas) >= 2:
            th, res, vec = ritz(alphas, rhos, Z)
            keep = [i for i in range(len(th)) if res[i] <= max(CUTS) * th[i] and th[i] > 1.05]
            W = {i: vec(i) for i in keep}
            STATE_RITZ["pairs"] = (th, res, W)
            entry["ritz"] = {"m": len(alphas), "theta_max": float(th[0]), "theta_min": float(th[-1]),
                             "converged_1e-2": int(sum(1 for i in range(len(th)) if res[i] <= 1e-2 * th[i])),
                             "converged_1e-1": int(sum(1 for i in range(len(th)) if res[i] <= 1e-1 * th[i])),
                             "top": [float(v) for v in th[:8]]}
        else:
            STATE_RITZ["pairs"] = None
        say("step %2d: CG %3d (tol %.2e)  %s  %s" % (
            entry["step"], base, tol,
            "  ".join("%s %d (%d pairs, dist %.1e)" % (k, v["cg"], v["pairs"], v["dist"])
                      for k, v in entry.items() if isinstance(v, dict) and "cg" in v),
            ("Ritz: m %d, %d/%d converged (1e-2/1e-1), theta %.3g..%.3g"
             % (entry["ritz"]["m"], entry["ritz"]["converged_1e-2"], entry["ritz"]["converged_1e-1"],
                entry["ritz"]["theta_max"], entry["ritz"]["theta_min"])) if "ritz" in entry else ""))
        LOG.append(entry)
        # the operator's count must be that of the step's own CG (NewtonCG adds it up)
        if hasattr(self.A, "ncalls"):
            self.A.ncalls = n0 + base
        return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--mode", default="mixed")
    ap.add_argument("--inc-tol", type=float, default=1e-6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    hm.configure_device("cuda" if hm.config.hypre_device else "cpu", COMM, quiet=True)
    km.set_precision(args.mode)
    N = args.n
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, 2)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem(Vh, lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad),
                                   bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    for attr in ("solver_fwd_inc", "solver_adj_inc"):
        getattr(pde, attr).parameters["rel_tolerance"] = args.inc_tol
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    PRIOR["R"] = prior.R
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, 200) for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    model = hm.Model(pde, prior, hm.DiscreteStateObservation(B, data, nstd ** 2))
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-6
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = 25
    params["globalization"] = "LS"
    params["GN_iter"] = 5
    params["cg_max_iter"] = 50
    params["print_level"] = 0
    ncg.CGSolverSteihaug = RecordingCG
    solver = hm.ReducedSpaceNewtonCG(model, params)
    t0 = time.perf_counter()
    solver.solve([None, prior.mean.copy(), None])
    say("Newton-CG: %d Newton, %d CG, J %.10e, |g| %.4e, %.1f s (with the offline solves)"
        % (solver.it, solver.total_cg_iter, solver.final_cost, solver.final_grad_norm, time.perf_counter() - t0))
    keys = sorted({k for e in LOG for k, v in e.items() if isinstance(v, dict) and "cg" in v})
    tot = {k: sum(e[k]["cg"] for e in LOG if k in e) for k in keys}
    base_cmp = sum(e["cg"] for e in LOG if keys and keys[0] in e)
    say("steps 2..: own CG %d; %s" % (base_cmp, ", ".join("%s %d" % kv for kv in tot.items())))
    if args.out and COMM.rank == 0:
        with open(args.out, "w") as f:
            json.dump({"n": N, "mode": args.mode, "inc_tol": args.inc_tol, "newton": solver.it,
                       "cg": solver.total_cg_iter, "J": float(solver.final_cost), "steps": LOG,
                       "totals": tot, "own_from_step2": base_cmp}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

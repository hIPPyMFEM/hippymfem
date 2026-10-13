# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What a low rank of the misfit Hessian leaves out, and how good its eigenpairs are.

    python tail_check.py <folder for the record> <starts> <steps> <the run's options, with --out the run's folder>

In coordinates in which the prior is the identity the Hessian is Ht = R^-1 H, self-adjoint in the R inner
product, and the run's eigenvectors U (U^T R U = I) span what the low rank holds; P = I - U U^T R takes
that out.  From each start z (a sample of the prior, E z z^T = R^-1), m Lanczos steps with P Ht P give Ritz
values theta_j and weights w_j with
    z^T R f(P Ht P) z  ~  |P z|_R^2 sum_j w_j f(theta_j),
whose mean over the starts is the trace of f over the complement: with f(x) = x the curvature left out
(exact for any m), with f(x) = x / (1 + x) the variance reduction left out (the low rank itself holds
sum lambda / (1 + lambda)), with f(x) = log(1 + x) twice the information left out.  The largest Ritz value
is a lower bound on the largest eigenvalue left out.  The starts go through the Hessian together, a batch
to a pair of sweeps.

The eigenpairs themselves: for a few of them the Rayleigh quotient u^T H u against the eigenvalue on
file, and the residual |Ht u - lambda u|_R.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as rn

dest, K, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
args = rn.parse(sys.argv[4:] + ["--stages", "map,eig", "--resume"])
import hippymfem as hm
from mpi4py import MPI
import mfem.par as mfem
from model import MisfitHessian

comm = MPI.COMM_WORLD
mfem.Hypre.Init()
P = rn.Problem(args, comm, writer=comm.rank == 0)
os.makedirs(dest, exist_ok=True)
z = rn.stage_data(P)
model = rn.make_model(P, z)
x, hist = rn.stage_map(P, model, z)                     # done: the MAP point is read, its forward problem solved
assert x is not None and hist.get("done"), "the run has no MAP point"
d, U = rn.stage_eig(P, model, x)                        # read
Ua = np.asarray(U.data)                                 # the eigenvectors as rows, R-orthonormal
k, n = Ua.shape
model.setPointForHessianEvaluations(x)
H = MisfitHessian(model)
prior = P.prior
B = args.batch


def new():
    return P.Vm.vector()


def Rmult(a):
    out = new()
    prior.R.mult(a, out)
    return out


def Rsolve(a):
    out = new()
    prior.Rsolver.solve(out, a)
    return out


def products(vs):
    out = []
    for a in range(0, len(vs), B):
        out += H.mult_many(vs[a:a + B])
    return out


def deflate(v, Rv):
    """``v - U U^T R v`` and the coefficients taken out."""
    c = Ua @ np.asarray(Rv.array)
    v.array[:] = np.asarray(v.array) - c @ Ua
    return c


rec = dict(rank=int(k), parameters=int(n), eigenvalues_sum_ratio=float((d / (1 + d)).sum()), above_one=int((d > 1).sum()))
# ---- the eigenpairs on file
t0 = time.time()
idx = sorted(set(int(i) for i in np.unique(np.round(np.geomspace(1, k, 16))).astype(int) - 1))
us = []
for i in idx:
    u = new()
    u.array[:] = Ua[i]
    us.append(u)
Hu = products(us)
pairs = []
for i, u, hu in zip(idx, us, Hu):
    rq = float(u.inner(hu))                              # u^T H u (u^T R u = 1)
    y = Rsolve(hu)                                       # Ht u
    y.axpy(-d[i], u)
    res = float(np.sqrt(max(y.inner(Rmult(y)), 0.0)))
    pairs.append(dict(index=i, eigenvalue=float(d[i]), rayleigh=rq, residual=res))
    P.say("tail: eigenpair %4d: eigenvalue on file %.4g, Rayleigh quotient %.4g (%.2f of it), residual %.3g (%.2f of the eigenvalue)"
          % (i, d[i], rq, rq / d[i], res, res / d[i]))
rec["pairs"] = pairs
P.say("tail: %d eigenpairs checked in %.0f s" % (len(idx), time.time() - t0))
# ---- Lanczos on the deflated operator, K starts side by side
hm.parRandom.set_seed(4242)
Q, RQ, beta0 = [[] for _ in range(K)], [[] for _ in range(K)], np.zeros(K)
for s in range(K):
    q = new()
    prior.sample(prior.sample_noise(1.0), q, add_mean=False)
    Rq = Rmult(q)
    deflate(q, Rq)
    Rq = Rmult(q)
    beta0[s] = np.sqrt(q.inner(Rq))
    q.scale(1.0 / beta0[s]); Rq.scale(1.0 / beta0[s])
    Q[s].append(q); RQ[s].append(Rq)
al, be, leak = np.zeros((K, M)), np.zeros((K, M)), np.zeros((K, M))
for j in range(M):
    t0 = time.time()
    HQ = products([Q[s][j] for s in range(K)])
    for s in range(K):
        al[s, j] = Q[s][j].inner(HQ[s])
        w = Rsolve(HQ[s])
        Rw = Rmult(w)
        full = np.sqrt(max(w.inner(Rw), 0.0))
        c = deflate(w, Rw)
        leak[s, j] = np.linalg.norm(c) / max(full, 1e-300)   # the part of Ht q that falls back into the span
        for _ in range(2):                                   # against every earlier vector, twice
            for q, Rq in zip(Q[s], RQ[s]):
                w.axpy(-float(Rq.inner(w)), q)
        Rw = Rmult(w)
        be[s, j] = np.sqrt(max(w.inner(Rw), 0.0))
        if j + 1 < M:
            w.scale(1.0 / be[s, j]); Rw.scale(1.0 / be[s, j])
            Q[s].append(w); RQ[s].append(Rw)
    fs = {"x": lambda t: t, "x2": lambda t: t * t, "ratio": lambda t: t / (1.0 + t), "log": lambda t: np.log1p(np.maximum(t, -0.999))}
    est = {name: np.zeros(K) for name in fs}
    top = np.zeros(K)
    for s in range(K):
        T = np.diag(al[s, :j + 1]) + np.diag(be[s, :j], 1) + np.diag(be[s, :j], -1)
        th, V = np.linalg.eigh(T)
        wts = V[0] ** 2
        top[s] = th[-1]
        for name, f in fs.items():
            est[name][s] = beta0[s] ** 2 * float((wts * f(th)).sum())
    line = {name: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(K)) if K > 1 else 0.0) for name, v in est.items()}
    rec["steps"] = rec.get("steps", []) + [dict(step=j + 1, estimates=line, largest_ritz=top.tolist(), leak=leak[:, j].tolist(),
                                                seconds=time.time() - t0)]
    rec.update(alpha=al[:, :j + 1].tolist(), beta=be[:, :j + 1].tolist(), beta0=beta0.tolist())
    if P.writer:
        with open(os.path.join(dest, "tail_check.json.tmp"), "w") as f:
            json.dump(rec, f)
        os.replace(os.path.join(dest, "tail_check.json.tmp"), os.path.join(dest, "tail_check.json"))
    P.say("tail: %2d Lanczos steps from %d starts: left out, curvature %.1f +- %.1f, variance reduction %.1f +- %.1f (the low rank holds %.1f), "
          "log(1 + x) %.1f +- %.1f, sum of squares %.1f; largest Ritz value %.3g (mean %.3g); %.2f of Ht q falls back into the span; %.0f s"
          % (j + 1, K, line["x"][0], line["x"][1], line["ratio"][0], line["ratio"][1], rec["eigenvalues_sum_ratio"], line["log"][0], line["log"][1],
             line["x2"][0], top.max(), top.mean(), leak[:, j].mean(), time.time() - t0))
    if P.out_of_time():
        break
P.say("tail: done")

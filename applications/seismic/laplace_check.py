# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""How good the Laplace approximation of a finished seismic run is: the true negative log posterior at draws of
the Laplace posterior against its quadratic model, and along a few eigenvectors.

    python laplace_check.py <folder for the record> <draws> <the run's options, with --out the run's folder>
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as rn

dest, K = sys.argv[1], int(sys.argv[2])
args = rn.parse(sys.argv[3:] + ["--stages", "map,eig", "--resume"])
import hippymfem as hm
from mpi4py import MPI
import mfem.par as mfem
from hippymfem.modeling.variables import PARAMETER, STATE

comm = MPI.COMM_WORLD
mfem.Hypre.Init()
P = rn.Problem(args, comm, writer=comm.rank == 0)
os.makedirs(dest, exist_ok=True)
z = rn.stage_data(P)
model = rn.make_model(P, z)
x, hist = rn.stage_map(P, model, z)                     # done: the MAP point is read, its forward problem solved
assert x is not None and hist.get("done"), "the run has no MAP point"
d, U = rn.stage_eig(P, model, x)                        # read
c0 = [float(v) for v in model.cost(x)]
P.say("laplace: at the MAP point J = %.6e (regularization %.6e, misfit %.6e)" % tuple(c0))
post = hm.GaussianLRPosterior(P.prior, d, U, mean=x[PARAMETER])
Ua = np.asarray(U.data)
mmap = np.array(x[PARAMETER].array)
n = mmap.size
hm.parRandom.set_seed(2026)
dirs = [0, int(np.searchsorted(-d, -100.0)), int(np.searchsorted(-d, -10.0)), min(int(np.searchsorted(-d, -1.0)), d.size - 1)]
steps = [-2.0, 2.0]
D, label = [], []
for i in range(K):
    sp, so = P.Vm.vector(), P.Vm.vector()
    post.sample(P.prior.sample_noise(1.0), sp, so, add_mean=False)
    D.append(np.array(so.array)); label.append(("sample", i, 0.0))
for kk in dirs:
    for t in steps:
        D.append(t / np.sqrt(1.0 + d[kk]) * Ua[kk]); label.append(("eigenvector %d (eigenvalue %.3g)" % (kk, d[kk]), kk, t))
v, w = P.Vm.vector(), P.Vm.vector()
q_r, q_lr = [], []
for dl in D:
    v.array[:] = dl
    P.prior.R.mult(v, w)
    q_r.append(0.5 * float(v.inner(w)))
    q_lr.append(0.5 * float(((Ua @ np.asarray(w.array)) ** 2 * d).sum()))
q_r, q_lr = np.array(q_r), np.array(q_lr)
# the Gauss-Newton quadratic form itself for a few of the draws (two sweeps each)
model.setPointForHessianEvaluations(x)
H = hm.ReducedHessian(model, misfit_only=True)
q_gn = np.full(len(D), np.nan)
for i in list(range(min(K, 4))):
    v.array[:] = D[i]
    H.mult(v, w)
    q_gn[i] = 0.5 * float(v.inner(w))
P.say("laplace: %d draws; expected 1/2 delta^T H delta = %.1f; drawn (low rank) %.1f +- %.1f; Gauss-Newton against low rank for the first: %s"
      % (K, 0.5 * (d / (1 + d)).sum(), q_lr[:K].mean(), q_lr[:K].std(), ", ".join("%.2f/%.2f" % (q_gn[i], q_lr[i]) for i in range(min(K, 4)))))
rec = dict(J_map=[c0[0], c0[1], c0[2]], q_gn=[None if not np.isfinite(q) else float(q) for q in q_gn], q_r=q_r.tolist(), q_lr=q_lr.tolist(),
           label=label, eigenvalues=[float(d[kk]) for kk in dirs], rank=int(d.size), above_one=int((d > 1).sum()), parameters=int(n),
           J=[], newton=[], seconds=[])
m = P.Vm.vector()
for i, dl in enumerate(D):
    t0 = time.time()
    m.array[:] = mmap + dl
    xs = model.generate_vector()
    xs[PARAMETER] = m
    model.solveFwd(xs[STATE], xs)
    c = [float(v_) for v_ in model.cost(xs)]
    rec["J"].append(c); rec["newton"].append(0); rec["seconds"].append(time.time() - t0)
    if P.writer:
        with open(os.path.join(dest, "laplace_check.json.tmp"), "w") as f:
            json.dump(rec, f)
        os.replace(os.path.join(dest, "laplace_check.json.tmp"), os.path.join(dest, "laplace_check.json"))
    dJ = c[0] - c0[0]
    P.say("laplace: %-34s t %+.0f: J - J_MAP = %9.3f; quadratic model %9.3f (low rank); difference %+8.3f  (%.0f s)"
          % (label[i][0] if label[i][0] != "sample" else "draw %d" % i, label[i][2], dJ, q_lr[i] + q_r[i], dJ - q_lr[i] - q_r[i], rec["seconds"][-1]))
P.say("laplace: done")

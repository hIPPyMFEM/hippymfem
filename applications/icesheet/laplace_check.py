# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""How good the Laplace approximation of a finished run is: the true negative log posterior at samples of the
Laplace posterior against its quadratic model, and along a few eigenvectors.

    python laplace_check.py <run folder> <folder for the record> <samples> [--res 8 --nz 3 ...]

(the run's own mesh options after the three arguments; the run must have its low rank).

For a draw m = m_MAP + delta of N(m_MAP, (H_lr + R)^-1):
    J(m)     = misfit(m) + 1/2 |m - m_prior|^2_R          (one nonlinear forward solve)
    J_gn(m)  = J(m_MAP) + 1/2 delta^T (H_gn + R) delta      (the Gauss-Newton Hessian at the MAP point, by a block solve)
    J_lr(m)  = J(m_MAP) + 1/2 delta^T (H_lr + R) delta      (the low rank the draws come from)
The importance weight of the draw is exp(-(J - J_lr)).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as rn
import lowrank

out, dest, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = rn.parse(sys.argv[4:] + ["--out", out, "--stages", "map"])
R = rn.Run(a)
os.makedirs(dest, exist_ok=True)
z = rn.stage_data(R)
model, prior = rn.make_model(R, z)
x, hist = rn.stage_map(R, model, prior, z)
assert x is not None and hist.get("done"), "the run has no MAP point"
import hippymfem as hm
from hippymfem.modeling.variables import PARAMETER, STATE

pb = R.pb
c0 = [float(v) for v in model.cost(x)]                      # total, regularization, misfit at the MAP point
R.say("laplace: at the MAP point J = %.6e (misfit %.6e, regularization %.6e)" % (c0[0], c0[2], c0[1]))
model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
H = rn.BlockHessian(hm.ReducedHessian(model, misfit_only=True), pb, model.misfit)
lr = lowrank.GrowingLowRank(H, prior, x[PARAMETER], R.path("lowrank"))
k = lr.kept(rn.OVERSAMPLE)
d, U = lr.eig(k)
Ua = np.asarray(U.data)                                     # the eigenvectors as rows, R-orthonormal
post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
mmap = np.array(x[PARAMETER].array)
n = mmap.size
hm.parRandom.set_seed(2026)
noise = prior.noise_vector()
sp, so = pb.Vm.vector(), pb.Vm.vector()
ND = 4                                                       # eigenvectors scanned: the first, two in between, one near eigenvalue 1
dirs = [0, int(np.searchsorted(-d, -100.0)), int(np.searchsorted(-d, -10.0)), int(np.searchsorted(-d, -1.0))]
steps = [-2.0, 2.0]
D = np.zeros((n, K + ND * len(steps)))
for i in range(K):
    prior.sample_noise(1.0, noise)
    post.sample(noise, sp, so, add_mean=False)
    D[:, i] = so.array
label = [("sample", i, 0.0) for i in range(K)]
for q, kk in enumerate(dirs):
    for s_, t in enumerate(steps):
        D[:, K + q * len(steps) + s_] = t / np.sqrt(1.0 + d[kk]) * Ua[kk]
        label.append(("eigenvector %d (eigenvalue %.3g)" % (kk, d[kk]), kk, t))
HD = pb.misfit_hessian_block(model.misfit, D)               # the Gauss-Newton misfit Hessian at the MAP point
v, w = pb.Vm.vector(), pb.Vm.vector()
RD = np.zeros_like(D)
for i in range(D.shape[1]):
    v.array[:] = D[:, i]
    prior.R.mult(v, w)
    RD[:, i] = w.array
q_gn = 0.5 * np.einsum("ij,ij->j", D, HD)
q_r = 0.5 * np.einsum("ij,ij->j", D, RD)
q_lr = 0.5 * ((Ua @ RD) ** 2 * d[:, None]).sum(axis=0)
# the misfit's gradient at the MAP point is -R (m_MAP - m_prior): its part in J is cancelled by the prior's
rec = dict(J_map=c0, q_gn=q_gn.tolist(), q_r=q_r.tolist(), q_lr=q_lr.tolist(), label=label, eigenvalues=[float(d[kk]) for kk in dirs],
           rank=int(k), above_one=int((d > 1).sum()), parameters=int(n), J=[], newton=[], seconds=[])
R.say("laplace: %d draws; expected 1/2 delta^T H delta = %.1f (low rank: 1/2 sum lambda/(1+lambda)); drawn: Gauss-Newton %.1f +- %.1f, low rank %.1f +- %.1f"
      % (K, 0.5 * (d / (1 + d)).sum(), q_gn[:K].mean(), q_gn[:K].std(), q_lr[:K].mean(), q_lr[:K].std()))
for i in range(D.shape[1]):
    t0 = time.time()
    so.array[:] = mmap + D[:, i]
    xs = [pb.generate_state(), so, None]
    pb.start_from(x)
    try:
        pb.solveFwd(xs[STATE], xs)
        c = [float(v_) for v_ in model.cost(xs)]
    except RuntimeError as e:
        R.say("laplace: %s %d: %s" % (label[i][0], i, e))
        c = [float("nan")] * 3
    rec["J"].append(c); rec["newton"].append(int(pb.last_forward["newton"])); rec["seconds"].append(time.time() - t0)
    with open(os.path.join(dest, "laplace_check.json.tmp"), "w") as f:
        json.dump(rec, f)
    os.replace(os.path.join(dest, "laplace_check.json.tmp"), os.path.join(dest, "laplace_check.json"))
    dJ = c[0] - c0[0]
    R.say("laplace: %-34s t %+.0f: J - J_MAP = %10.2f; quadratic model %10.2f (Gauss-Newton), %10.2f (low rank); difference %+9.2f, %+9.2f  (%d Newton steps, %.0f s)"
          % (label[i][0] if label[i][0] != "sample" else "draw %d" % i, label[i][2], dJ, q_gn[i] + q_r[i], q_lr[i] + q_r[i],
             dJ - q_gn[i] - q_r[i], dJ - q_lr[i] - q_r[i], rec["newton"][-1], rec["seconds"][-1]))
    if R.late():
        break
pb.start_from(x)
R.say("laplace: done")

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""How good the Laplace approximation is for a run whose posterior comes from the sparse Hessian
(``run.py --hessian local``): the true negative log posterior at draws of that posterior against its
quadratic model, and along the first draws at minus, half and twice their length.

    python laplace_check_local.py <run folder> <folder for the record> <draws> [the run's options]

(the run's own options after the three arguments, ``--hessian local --reach`` among them; the run must
have its sparse Hessian).

For a draw m = m_MAP + delta of N(m_MAP, (H_s + R)^-1):
    J(m)     = misfit(m) + 1/2 |m - m_prior|^2_R          (one nonlinear forward solve)
    J_gn(m)  = J(m_MAP) + 1/2 delta^T (H_gn + R) delta      (the Gauss-Newton Hessian at the MAP point, its own products)
    J_s(m)   = J(m_MAP) + 1/2 delta^T (H_s + R) delta       (the sparse matrix the draws come from)
A Gaussian posterior has 1/2 delta^T (H + R) delta = n / 2 on average over its draws, n the number of
parameters.  The importance weight of a draw is exp(-(J - J_s)).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as rn
import localhessian as lh

out, dest, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = rn.parse(sys.argv[4:] + ["--out", out, "--stages", "map"])
R = rn.Run(a)
os.makedirs(dest, exist_ok=True)
z = rn.stage_data(R)
model, prior = rn.make_model(R, z)
x, hist = (rn.stage_map_local if a.map_solver == "local" else rn.stage_map)(R, model, prior, z)
assert x is not None and hist.get("done"), "the run has no MAP point"
import hippymfem as hm
from hippymfem.common.linalg import hypre_to_scipy
from hippymfem.modeling.variables import PARAMETER, STATE

pb = R.pb
c0 = [float(v) for v in model.cost(x)]                      # total, regularization, misfit at the MAP point
R.say("laplace: at the MAP point J = %.6e (misfit %.6e, regularization %.6e)" % (c0[0], c0[2], c0[1]))
Hs, local = rn.stage_local(R, model, prior, x)             # read
assert Hs is not None, "the run has no sparse Hessian"
model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
blocks = hasattr(pb.solver, "solve_many")
post = lh.LocalPosterior(Hs, hypre_to_scipy(prior.A).tocsr(), hypre_to_scipy(prior.M).tocsr(), pb, model.misfit, prior,
                         x[PARAMETER], R.comm, precision=a.factors, method=a.local_solver,
                         draws=a.draws if a.draws != "auto" else ("adjoint" if blocks else "lanczos"))
mmap = np.array(x[PARAMETER].array)
n = mmap.size
hm.parRandom.set_seed(2026)
_, So = post.draw(K)
ND = min(K, 3 if blocks else 1)                             # draws scanned along their line
steps = [-1.0, 0.5, 2.0]
D = np.zeros((n, K + ND * len(steps)))
D[:, :K] = So
label = [("draw", i, 1.0) for i in range(K)]
for q in range(ND):
    for s_, t in enumerate(steps):
        D[:, K + q * len(steps) + s_] = t * So[:, q]
        label.append(("draw %d times" % q, q, t))
# the Gauss-Newton misfit Hessian at the MAP point on every direction: a block solve of the factorization, or a product each
if blocks:
    HD = pb.misfit_hessian_block(model.misfit, D)
else:
    Hop = hm.ReducedHessian(model, misfit_only=True)
    HD = np.zeros_like(D)
    v, w = pb.Vm.vector(), pb.Vm.vector()
    for i in range(K + ND * len(steps)):
        if label[i][0] != "draw":                           # a multiple of a draw: its product is the draw's
            HD[:, i] = label[i][2] * HD[:, label[i][1]]
            continue
        v.array[:] = D[:, i]
        Hop.mult(v, w)
        HD[:, i] = w.array
v, w = pb.Vm.vector(), pb.Vm.vector()
RD = np.zeros_like(D)
for i in range(D.shape[1]):
    v.array[:] = D[:, i]
    prior.R.mult(v, w)
    RD[:, i] = w.array
q_gn = 0.5 * np.einsum("ij,ij->j", D, HD)
q_s = 0.5 * np.einsum("ij,ij->j", D, Hs @ D)
q_r = 0.5 * np.einsum("ij,ij->j", D, RD)
rec = dict(J_map=c0, q_gn=q_gn.tolist(), q_s=q_s.tolist(), q_r=q_r.tolist(), label=label, parameters=int(n), draws=int(K),
           products=int(local["products"]), reach=float(local["reach"]), J=[], newton=[], seconds=[])
R.say("laplace: %d draws of the posterior with the sparse Hessian (%d products), %d parameters: 1/2 delta^T (H + R) delta = %.1f +- %.1f "
      "with the Gauss-Newton Hessian, %.1f +- %.1f with the sparse one (%.1f for a Gaussian); its misfit part %.1f and %.1f"
      % (K, local["products"], n, (q_gn + q_r)[:K].mean(), (q_gn + q_r)[:K].std(), (q_s + q_r)[:K].mean(), (q_s + q_r)[:K].std(), 0.5 * n,
         q_gn[:K].mean(), q_s[:K].mean()))
so = pb.Vm.vector()
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
    R.say("laplace: %-14s %+4.1f: J - J_MAP = %11.2f; quadratic model %11.2f (Gauss-Newton), %11.2f (sparse); difference %+9.2f, %+9.2f  (%d Newton steps, %.0f s)"
          % (label[i][0] if label[i][0] != "draw" else "draw %d" % i, label[i][2], dJ, q_gn[i] + q_r[i], q_s[i] + q_r[i],
             dJ - q_gn[i] - q_r[i], dJ - q_s[i] - q_r[i], rec["newton"][-1], rec["seconds"][-1]))
    if R.late():
        break
pb.start_from(x)
m = len(rec["J"])
dJ = np.array([c[0] for c in rec["J"]]) - c0[0]
k = min(m, K)
if k:
    e_gn, e_s = dJ[:k] - (q_gn + q_r)[:k], dJ[:k] - (q_s + q_r)[:k]
    lw = -(e_s - e_s.min())
    wts = np.exp(lw)
    R.say("laplace: over %d draws J - J_MAP = %.1f +- %.1f; above the Gauss-Newton quadratic by %+.1f +- %.1f, above the sparse one by %+.1f +- %.1f "
          "(standard deviations over the draws); effective sample size of the draws as importance samples %.1f of %d"
          % (k, dJ[:k].mean(), dJ[:k].std(), e_gn.mean(), e_gn.std(), e_s.mean(), e_s.std(), wts.sum() ** 2 / (wts ** 2).sum(), k))
R.say("laplace: done")

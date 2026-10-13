# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Samples of the Laplace posterior itself, without a low rank of the Hessian: what the pointwise standard
deviation of a low-rank posterior is to be measured against.

    python exact_samples.py <folder for the record> <samples> <CG iterations> <seed> <the run's options, with --out the run's folder>

A sample of N(0, (H + R)^-1) is x = (H + R)^-1 b with b ~ N(0, H + R):
    b = R s + J^T W^T eta / sigma,    s a sample of the prior, eta white noise on the data
(one backward sweep gives the second part for a batch of samples).  The system is solved by conjugate
gradients preconditioned with the run's low-rank posterior covariance R^-1 - U D U^T, D = lambda / (1 + lambda):
the preconditioned matrix is the identity plus what the low rank leaves out, so a few iterations do, each one
Hessian product per sample, a batch of samples to a pair of sweeps.  (``EXACT_RANK=k`` in the environment
keeps only the first ``k`` eigenpairs in the preconditioner and in the low rank's samples.)

With every sample goes the low-rank posterior's sample from the same prior sample.  Both sets are saved
(``exact_<seed>.npy``, ``lowrank_<seed>.npy``, canonical vertex order, float32, zero mean), and the record has
the residuals of the iteration.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as rn

dest, N, ITS, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
args = rn.parse(sys.argv[5:] + ["--stages", "map,eig", "--resume"])
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
keep = d > 0.0
d, Ua = d[keep], Ua[keep]
if os.environ.get("EXACT_RANK"):                        # a poorer preconditioner, to see what the iteration does with it
    d, Ua = d[:int(os.environ["EXACT_RANK"])], Ua[:int(os.environ["EXACT_RANK"])]
model.setPointForHessianEvaluations(x)
H = MisfitHessian(model)
prior = P.prior
B = args.batch
shrink = d / (1.0 + d)                                  # the low rank's part of the covariance: U shrink U^T
pull = 1.0 - 1.0 / np.sqrt(1.0 + d)                     # the low-rank posterior sample of a prior sample


def new():
    return P.Vm.vector()


def Rmult(a):
    out = new()
    prior.R.mult(a, out)
    return out


def precondition(r):
    """``(R^-1 - U D U^T) r``."""
    out = new()
    prior.Rsolver.solve(out, r)
    out.array[:] = np.asarray(out.array) - (shrink * (Ua @ np.asarray(r.array))) @ Ua
    return out


hm.parRandom.set_seed(7000 + SEED)
rng = np.random.default_rng(7000 + SEED)
lin = model._at_linearization()
exact, lowrank = [], []
rec = dict(samples=0, iterations=ITS, seed=SEED, rank=int(d.size), residuals=[], seconds=[])
t_all = time.time()
for a0 in range(0, N, B):
    nb = min(B, N - a0)
    t0 = time.time()
    s = []
    for _ in range(nb):
        v = new()
        prior.sample(prior.sample_noise(1.0), v, add_mean=False)
        s.append(v)
    eta = rng.standard_normal((nb,) + model.d_obs.shape)
    q = model.data.apply_transpose(eta) / model.noise_std
    bH = model.engine.adjoint(q, final=lin.final)         # J^T W^T eta / sigma: covariance H
    b = []
    for v, g in zip(s, bH):
        w = Rmult(v)
        w.axpy(1.0, g)
        b.append(w)
    # conjugate gradients for every sample of the batch, the products taken together
    X = [new() for _ in range(nb)]
    for v in X:
        v.zero()
    Rr = [w.copy() for w in b]
    Z = [precondition(r) for r in Rr]
    Pd = [v.copy() for v in Z]
    rz = np.array([r.inner(v) for r, v in zip(Rr, Z)])
    rz0 = rz.copy()
    hist_b = [np.ones(nb).tolist()]
    for it in range(ITS):
        HP = H.mult_many(Pd)
        for k in range(nb):
            KP = Rmult(Pd[k])
            KP.axpy(1.0, HP[k])
            alpha = rz[k] / Pd[k].inner(KP)
            X[k].axpy(alpha, Pd[k])
            Rr[k].axpy(-alpha, KP)
            Z[k] = precondition(Rr[k])
            new_rz = Rr[k].inner(Z[k])
            beta = new_rz / rz[k]
            rz[k] = new_rz
            Pd[k].scale(beta)
            Pd[k].axpy(1.0, Z[k])
        hist_b.append(np.sqrt(np.maximum(rz / rz0, 0.0)).tolist())
        P.say("exact: samples %d to %d, iteration %d: residual (in the preconditioner's norm) %.2e at the most, %.0f s"
              % (a0, a0 + nb - 1, it + 1, max(hist_b[-1]), time.time() - t0))
    for k in range(nb):
        exact.append(np.array(X[k].array))
        sl = np.asarray(s[k].array) - (pull * (Ua @ np.asarray(Rmult(s[k]).array))) @ Ua
        lowrank.append(sl)
    rec["samples"] = len(exact)
    rec["residuals"].append(hist_b)
    rec["seconds"].append(time.time() - t0)
    E, L = P.canon.gather(np.array(exact)), P.canon.gather(np.array(lowrank))
    if P.writer:
        np.save(os.path.join(dest, "exact_%d.npy" % SEED), E.astype(np.float32))
        np.save(os.path.join(dest, "lowrank_%d.npy" % SEED), L.astype(np.float32))
        with open(os.path.join(dest, "exact_%d.json" % SEED), "w") as f:
            json.dump(rec, f)
    se, sl_ = float(np.sqrt(np.mean(E ** 2))), float(np.sqrt(np.mean(L ** 2)))
    P.say("exact: %d samples in %.0f s; rms over vertices and samples: %.4f, the low rank's from the same prior samples %.4f (ratio %.3f)"
          % (len(exact), time.time() - t_all, se, sl_, se / sl_))
    if P.out_of_time():
        break
P.say("exact: done")

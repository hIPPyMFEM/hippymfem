# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The double pass of hIPPyMFEM's ``doublePassG``, grown a block of random columns at a
time until the generalized eigenvalues ``H u = lambda R u`` it finds have gone below a
threshold and their count above it has settled (the same code as the geothermal
animation's ``movie_data.py``).  Columns already B-orthonormalized do not change when
more are appended, so the basis is saved after every block and a stopped job resumes."""
import json
import os
import time

import numpy as np
from mpi4py import MPI

import hippymfem as hm

COMM = MPI.COMM_WORLD
RANK = COMM.rank

try:
    import cupy as cp
except ImportError:                                          # pragma: no cover
    cp = None


def _on_card(fn, *arrays):
    """``fn`` of host arrays computed on the card, when there is one and it has the room,
    else on the host.  The dense algebra of a basis of 10^4 columns (the projected
    eigenproblem, the products with the basis) takes minutes on the one core an MPI rank
    is bound to and seconds on a card."""
    if cp is not None and os.environ.get("HIPPYMFEM_DEVICE", "") in ("gpu", "cuda"):
        try:
            out = fn(cp, *[cp.asarray(a) for a in arrays])
            return tuple(cp.asnumpy(o) for o in out) if isinstance(out, tuple) else cp.asnumpy(out)
        except cp.cuda.memory.OutOfMemoryError:
            pass
        finally:
            cp.get_default_memory_pool().free_all_blocks()
    return fn(np, *arrays)


def _eigvalsh(T):
    return np.sort(_on_card(lambda xp, A: xp.linalg.eigvalsh(A), T))[::-1]


#: bytes of a basis put on the card at a time: the basis of a fine grid (5000 columns of
#: half a million parameters are 20 GB) does not fit beside a factorization whole, and on
#: the host its products take the one core of the rank
PIECE = 2 ** 31


def _rows(A, width=None):
    """The rows of ``A`` (or of an array of rows ``width`` long) in ranges of at most ``PIECE`` bytes."""
    step = max(1, int(PIECE // (8 * max(A.shape[1] if width is None else width, 1))))
    return [slice(a, min(a + step, A.shape[0])) for a in range(0, A.shape[0], step)]


def _in_pieces(card, host):
    """``card()`` if there is a card and it has the room, else ``host()``."""
    if cp is not None and os.environ.get("HIPPYMFEM_DEVICE", "") in ("gpu", "cuda"):
        try:
            return card()
        except cp.cuda.memory.OutOfMemoryError:
            pass
        finally:
            cp.get_default_memory_pool().free_all_blocks()
    return host()


def _abt(A, B):
    """``A @ B.T`` for a basis ``A`` (rows) and a block ``B`` (rows)."""
    def card():
        Bt = cp.asarray(B).T
        out = np.empty((A.shape[0], B.shape[0]))
        for s in _rows(A):
            out[s] = cp.asnumpy(cp.asarray(A[s]) @ Bt)
        return out
    return _in_pieces(card, lambda: A @ B.T)


def _ctq(C, Q):
    """``C.T @ Q`` for the coefficients ``C`` (one row per row of the basis ``Q``)."""
    def card():
        out = cp.zeros((C.shape[1], Q.shape[1]))
        for s in _rows(Q):
            out += cp.asarray(C[s]).T @ cp.asarray(Q[s])
        return cp.asnumpy(out)
    return _in_pieces(card, lambda: C.T @ Q)


def _vtq(V, Q):
    """``V.T @ Q`` for many combinations ``V`` (columns) of the basis ``Q``: the result is
    as large as the basis, and is made a range of its columns at a time."""
    def card():
        Vt = cp.asarray(np.ascontiguousarray(V.T))
        out = np.empty((V.shape[1], Q.shape[1]))
        step = max(1, int(PIECE // (8 * max(Q.shape[0], V.shape[1], 1))))
        for a in range(0, Q.shape[1], step):
            out[:, a:a + step] = cp.asnumpy(Vt @ cp.asarray(np.ascontiguousarray(Q[:, a:a + step])))
        return out
    return _in_pieces(card, lambda: V.T @ Q)


def say(*a):
    if RANK == 0:
        print(" ".join(str(x) for x in a), flush=True)


def allsum(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    COMM.Allreduce(MPI.IN_PLACE, a, op=MPI.SUM)
    return a


class GrowingLowRank:
    """The double pass, a block of columns at a time.

    Keeps ``Q`` (B-orthonormal, ``Q^T R Q = I``), ``B Q`` and ``T = Q^T H Q`` and saves
    each block's rows of ``Q`` and ``B Q`` per rank, ``T`` and the progress on rank 0.
    """

    def __init__(self, H, prior, template, folder):
        self.H, self.prior, self.template, self.folder = H, prior, template, folder
        self.n = int(template.global_size)                 # the dimension of the parameter
        self.Q = np.zeros((0, template.local_size))
        self.BQ = np.zeros((0, template.local_size))
        self.T = np.zeros((0, 0))
        self.history = []
        os.makedirs(folder, exist_ok=True) if RANK == 0 else None
        COMM.Barrier()
        state = os.path.join(folder, "state.json")
        if os.path.exists(state):
            st = COMM.bcast(json.load(open(state)) if RANK == 0 else None)
            for b in range(st["blocks"]):
                q = np.load(os.path.join(folder, "Q_r%03d_b%03d.npy" % (RANK, b)))
                bq = np.load(os.path.join(folder, "BQ_r%03d_b%03d.npy" % (RANK, b)))
                if q.shape[1] != template.local_size:
                    raise RuntimeError("the saved basis was made with another partition")
                self.Q = np.vstack([self.Q, q])
                self.BQ = np.vstack([self.BQ, bq])
            self.T = np.load(os.path.join(folder, "T.npy")) if RANK == 0 else None
            # its leading block: the projection onto the columns that were read (a run
            # killed between the two files leaves T one block ahead)
            self.T = COMM.bcast(self.T)[:self.Q.shape[0], :self.Q.shape[0]]
            self.history = st["history"]
            say("lowrank: resumed with %d columns in %d blocks" % (self.Q.shape[0], st["blocks"]))

    @property
    def ncols(self):
        return self.Q.shape[0]

    def kept(self, p):
        """How many eigenpairs of the projected problem to keep: all but ``p`` columns of
        oversampling, or all of them once the basis spans the parameter space, where the
        projected problem is the Hessian itself."""
        return self.ncols if self.ncols >= self.n else self.ncols - p

    def _project(self, Qb):
        if self.ncols:
            C = allsum(_abt(self.BQ, Qb))                                        # (ncols, nb)
            Qb -= _ctq(C, self.Q)

    def _borth(self, Qb):
        mv = hm.MultiVector.from_array(COMM, Qb)
        Bq, _ = mv.Borthogonalize(self.prior.R)
        return np.array(mv.data), np.array(Bq.data)

    def add_block(self, Omega):
        """Append the block of random vectors ``Omega`` (a MultiVector)."""
        t0 = time.time()
        nb = Omega.nvec()
        Y = hm.MultiVector(self.template, nb)
        if hasattr(self.H, "mult_block"):          # a batched Hessian: block solves
            Y.data[:] = self.H.mult_block(np.asarray(Omega.data).T).T
        else:
            for i in range(nb):
                self.H.mult(Omega[i], Y[i])
        t_h1 = time.time() - t0
        Qraw = hm.MultiVector(self.template, nb)
        for i in range(nb):
            self.prior.Rsolver.solve(Qraw[i], Y[i])
        del Y
        Qb = np.array(Qraw.data)
        del Qraw
        # B-orthogonal to the columns already there, then among themselves; twice
        for _ in range(2):
            self._project(Qb)
            self._project(Qb)
            Qb, BQb = self._borth(Qb)
        t1 = time.time()
        Qmv = hm.MultiVector.from_array(COMM, Qb.copy())
        AQ = hm.MultiVector(self.template, nb)
        if hasattr(self.H, "mult_block"):
            AQ.data[:] = self.H.mult_block(np.asarray(Qmv.data).T).T
        else:
            for i in range(nb):
                self.H.mult(Qmv[i], AQ[i])
        t_h2 = time.time() - t1
        AQb = np.array(AQ.data)
        del AQ, Qmv
        n0 = self.ncols
        self.Q = np.vstack([self.Q, Qb])
        self.BQ = np.vstack([self.BQ, BQb])
        Tc = allsum(_abt(self.Q, AQb))                                        # (n0 + nb, nb) = Q^T H Q_new
        T = np.zeros((n0 + nb, n0 + nb))
        T[:n0, :n0] = self.T
        T[:, n0:] = Tc
        T[n0:, :n0] = Tc[:n0].T
        T[n0:, n0:] = 0.5 * (Tc[n0:] + Tc[n0:].T)
        self.T = T
        b = len(self.history)
        np.save(os.path.join(self.folder, "Q_r%03d_b%03d.npy" % (RANK, b)), Qb)
        np.save(os.path.join(self.folder, "BQ_r%03d_b%03d.npy" % (RANK, b)), BQb)
        COMM.Barrier()                                       # every rank's block is on disk
        self.w = _eigvalsh(T)
        rec = dict(block=b, columns=n0 + nb, seconds=time.time() - t0, seconds_products=t_h1 + t_h2,
                   products=2 * nb)
        self.history.append(rec)
        if RANK == 0:
            # each written whole, the count of blocks last
            np.save(os.path.join(self.folder, "T.tmp.npy"), T)
            os.replace(os.path.join(self.folder, "T.tmp.npy"), os.path.join(self.folder, "T.npy"))
            with open(os.path.join(self.folder, "state.json.tmp"), "w") as f:
                json.dump(dict(blocks=b + 1, history=self.history), f, indent=1)
            os.replace(os.path.join(self.folder, "state.json.tmp"), os.path.join(self.folder, "state.json"))
        COMM.Barrier()
        return self.w

    def eigenvalues(self):
        """The eigenvalues of the projected problem, largest first."""
        if getattr(self, "w", None) is None or len(self.w) != self.ncols:
            self.w = _eigvalsh(self.T)
        return self.w

    def eig(self, k):
        """The top ``k`` eigenpairs ``(d, U)`` of the projected problem."""
        def top(xp, T):
            w, V = xp.linalg.eigh(T)
            order = xp.argsort(w)[::-1][:k]
            return w[order], V[:, order]
        d, V = _on_card(top, self.T)
        U = hm.MultiVector.from_array(COMM, np.ascontiguousarray(_vtq(V, self.Q)))
        return d, U




def grow(lr, template, first=70, block=150, p=20, stop=1.0, settle=0.02, max_cols=3000,
         late=None, seed=99, log=say):
    """Add blocks to ``lr`` until its smallest kept eigenvalue (``p`` columns of
    oversampling aside) is below ``stop`` and the number above ``stop`` grew by less than
    ``settle`` in the last block, or ``max_cols`` columns, or as many columns as there
    are parameters; returns ``(done, counts)``.  ``late(ahead)`` says, the same on every
    rank, whether the run has to end within ``ahead`` seconds: no block is started that
    would not be finished by then."""
    above = []
    if len(lr.history) > 1:
        # resumed: the count before the last block, from the leading block of T (the
        # projection onto the basis as it was then)
        n0 = lr.history[-2]["columns"]
        w0 = _eigvalsh(lr.T[:n0, :n0])
        above.append(int((w0[:n0 - p] > stop).sum()))
    while True:
        if lr.ncols:
            w = lr.eigenvalues()
            k = lr.kept(p)
            above.append(int((w[:k] > stop).sum()))
            settled = len(above) > 1 and above[-1] <= (1.0 + settle) * above[-2]
            done = (w[k - 1] < stop and settled) or lr.ncols >= min(max_cols, lr.n)
            log("lowrank: %d columns, k %d: eigenvalues %.4g .. %.4g, %d of them above %g -> %s"
                % (lr.ncols, k, w[0], w[k - 1], above[-1], stop, "enough" if done else "more"))
            if done:
                return True, above
        nb = min(first if lr.ncols == 0 else block, lr.n - lr.ncols)     # never more columns than parameters
        if late is not None and lr.history:
            h = lr.history[-1]
            prev = lr.history[-2]["columns"] if len(lr.history) > 1 else 0
            per_col = h["seconds"] / max(h["columns"] - prev, 1)
            if late(1.2 * per_col * nb):
                log("lowrank: stopping before the next block (time limit)")
                return False, above
        b = len(lr.history)
        hm.parRandom.set_seed(seed if b == 0 else 1000 + b)
        Omega = hm.MultiVector(template, nb)
        hm.parRandom.normal_multivector(1.0, Omega)
        lr.add_block(Omega)
        h = lr.history[-1]
        log("lowrank: block %d (%d columns) in %.1f s, %.2f s a Hessian product"
            % (b, nb, h["seconds"], h["seconds_products"] / h["products"]))

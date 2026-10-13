# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The misfit Hessian as a sparse matrix, for data that see the parameter locally.

The surface velocity over a point of the bed answers to the friction within a few ice
thicknesses of it, so a column of the misfit Hessian ``H = J^T W J`` is a bump around its
own node: at 20 km it falls to 1.5 % of its diagonal entry 100 km away, 99 % of it lies
within 130 km, and it is positive.  Such a matrix has as many eigenvalues above one as the
data resolve patches of the bed (thousands, and more on every finer grid), which is what
makes its low rank expensive, and it has few entries that matter in a row, which is what
this module uses.

Its columns are read off products with combs: the sum of the unit vectors of every
``s``-th node in x and in y is one vector, and where the nodes of a comb are further apart
than twice the reach of a column their columns do not overlap, so that one product gives
all of them.  ``s^2`` combs (one per offset) cover every node: ``(2 reach / step)^2``
products, whatever the number of parameters and the rank.  (They do grow as the grid is
refined under a fixed reach.  The point spread function method of Alger, Hartland, Petra
and Ghattas, SIAM J. Sci. Comput. 2024, takes the columns at scattered points and
interpolates them between, which does not.)

The posterior precision ``H_s + R`` with the prior's ``R = A M^-1 A`` is not sparse, but

    [ H_s   A ] [ y ]   [ b ]
    [ A    -M ] [ w ] = [ c ]        gives        (H_s + A M^-1 A) y = b + A M^-1 c,

is: one factorization of it (cuDSS) solves with the posterior covariance, and with
``c ~ N(0, M)`` (the prior's own noise) and ``b ~ N(0, H)`` (one adjoint solve) it draws
samples.  Pointwise variances are mean squares of such samples.

On a fine grid the rows of ``H_s`` are long (800 entries at a reach of 80 km on nodes 5 km
apart) and that system's factors do not fit a card.  The solves are then conjugate
gradients on ``H_s + A M^-1 A`` preconditioned by the same system with ``H_s`` lumped onto
its diagonal, ``D = diag(H_s 1)``: its rows are those of the prior's own matrices, so it
factorizes at any size, and since the columns of ``H_s`` are positive bumps ``D`` acts
like ``H_s`` on everything smoother than a bump.
"""
import time

import numpy as np
import scipy.sparse as sp


def lattice(xy, origin, step):
    """The integer coordinates of dofs on their lattice of spacing ``step``."""
    return np.rint((np.asarray(xy)[:, :2] - np.asarray(origin)) / step).astype(np.int64)


def spacing(reach, step):
    """Nodes between the teeth of a comb, so that columns of ``reach`` do not overlap."""
    return max(2, int(np.ceil(2.0 * reach / step - 1e-9)))


def comb_products(apply_block, IJ, s, chunk=256, log=None, only=None, late=None):
    """``H`` times each of the ``s^2`` combs (or those numbered ``only``): ``(n, combs)``
    float32 in their order, each dof's comb, and how many were done (``late()`` true ends
    the loop early)."""
    n = IJ.shape[0]
    color = (IJ[:, 0] % s) * s + (IJ[:, 1] % s)
    which = np.arange(s * s) if only is None else np.asarray(only)
    HM = np.zeros((n, which.size), dtype=np.float32)
    place = -np.ones(s * s, dtype=np.int64)
    place[which] = np.arange(which.size)
    rows = np.arange(n)
    t0 = time.time()
    done = 0
    for a in range(0, which.size, chunk):
        b = min(a + chunk, which.size)
        col = place[color]
        mine = (col >= a) & (col < b)
        M = np.zeros((n, b - a))
        M[rows[mine], col[mine] - a] = 1.0
        HM[:, a:b] = apply_block(M)
        done = b
        if log is not None:
            log("local: %d of %d products, %.1f s" % (b, which.size, time.time() - t0))
        if late is not None and late():
            break
    return HM, color, done


def lanczos_sqrt(matvec, Z, k=60):
    """``H^(1/2) Z`` column by column by ``k`` Lanczos steps on the card (``matvec`` applies
    the symmetric matrix to a block; negative Ritz values, of a matrix that is positive
    only up to its approximation, count as zero).  Vectors of covariance ``H`` from white
    ones without a factor of ``H``."""
    import cupy as cp

    n, m = Z.shape
    b0 = cp.linalg.norm(Z, axis=0)
    V = [Z / b0]
    al, be = [], []
    w = matvec(V[0])
    a = (w * V[0]).sum(axis=0)
    w -= V[0] * a
    al.append(a)
    for j in range(1, k):
        b = cp.linalg.norm(w, axis=0)
        b = cp.where(b > 0, b, 1.0)
        be.append(b)
        V.append(w / b)
        w = matvec(V[j]) - V[j - 1] * b
        a = (w * V[j]).sum(axis=0)
        w -= V[j] * a
        al.append(a)
    al, be = cp.asnumpy(cp.stack(al)), cp.asnumpy(cp.stack(be))             # (k, m), (k - 1, m)
    out = cp.zeros_like(Z)
    coef = np.zeros((k, m))
    for c in range(m):
        ew, ev = np.linalg.eigh(np.diag(al[:, c]) + np.diag(be[:, c], 1) + np.diag(be[:, c], -1))
        coef[:, c] = ev @ (np.sqrt(np.maximum(ew, 0.0)) * ev[0])
    coef = cp.asarray(coef)
    for j in range(k):
        out += V[j] * coef[j]
    return out * b0


def sparse_from_combs(IJ, HM, color, s, step, reach):
    """The entries ``H_ij`` with ``|x_i - x_j| < reach``, each read off the response of the
    comb of ``j`` at ``i`` (``reach <= s step / 2``: no other tooth of that comb is as
    near), symmetrized.

    The rest of every row is dropped.  Its sum is known (the combs' responses add up to
    ``H 1``), and three ways of keeping it were tried: the tail on the diagonal, the rows
    scaled, and a taper to zero at the reach with what it takes on the diagonal.  All are
    exact on a constant and worse on samples of the prior, which are not smooth across a
    column of this matrix."""
    if reach > 0.5 * s * step + 1e-9:
        raise ValueError("a reach of %g km needs combs more than %d nodes apart" % (reach, s))
    n = IJ.shape[0]
    lo = IJ.min(axis=0)
    span = IJ.max(axis=0) - lo + 1
    idx = -np.ones(tuple(span), dtype=np.int64)
    a0, b0 = IJ[:, 0] - lo[0], IJ[:, 1] - lo[1]
    idx[a0, b0] = np.arange(n)
    r = int(np.ceil(reach / step))
    rows, cols, vals = [], [], []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if (dx * dx + dy * dy) * step * step >= reach * reach * (1.0 - 1e-12):
                continue
            a, b = a0 + dx, b0 + dy
            ok = (a >= 0) & (a < span[0]) & (b >= 0) & (b < span[1])
            j = np.full(n, -1, dtype=np.int64)
            j[ok] = idx[a[ok], b[ok]]
            i = np.nonzero(j >= 0)[0]
            j = j[i]
            rows.append(i.astype(np.int32)); cols.append(j.astype(np.int32)); vals.append(HM[i, color[j]].astype(np.float64))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    H = (0.5 * (H + H.T)).tocsr()
    H.sort_indices()
    return H


class LocalPosterior:
    """``N(mean, (H_s + A M^-1 A)^-1)`` through the sparse symmetric system above.

    ``draw(k)`` returns ``k`` zero-mean samples of the prior and of the posterior made from
    the same noise; ``solve(B)`` applies the posterior covariance to the columns of ``B``.
    ``Hlr.solve`` and ``sample`` are those of the library's ``GaussianLRPosterior``, for the
    code that takes either.
    """

    #: entries of ``H_s`` up to which its system is factorized whole; above, conjugate gradients
    DIRECT_NNZ = 2.0e7

    def __init__(self, Hs, A, M, problem, misfit, prior, mean, comm, precision="double", seed=20261004, method="auto",
                 rtol=1e-7, maxit=300, draws="adjoint"):
        import gpu_direct

        self.n = n = Hs.shape[0]
        self.pb, self.misfit, self.prior, self.mean = problem, misfit, prior, mean
        self.method = method if method != "auto" else ("direct" if Hs.nnz <= self.DIRECT_NNZ else "cg")
        #: where a draw's misfit part comes from: an adjoint solve with white noise on the data (the
        #: Hessian itself), or the square root of the sparse matrix by Lanczos (no solve of the PDE)
        self.draws = draws
        if draws == "lanczos" and self.method != "cg":
            import cupyx.scipy.sparse as csp

            self.Hs = csp.csr_matrix(Hs)
        self.rtol, self.maxit, self.iterations = rtol, maxit, []
        A = sp.csr_matrix(A)
        A.sort_indices()
        lead = Hs
        if self.method == "cg":
            import cupyx.scipy.sparse as csp

            # the preconditioner: H_s lumped onto its diagonal (H_s applied to 1), in the place of H_s
            lead = sp.diags(np.maximum(np.asarray(Hs.sum(axis=1)).ravel(), 0.0)).tocsr()
            self.Hs = csp.csr_matrix(Hs)
            self.A = csp.csr_matrix(A)
            Mc = sp.csr_matrix(M)
            Mc.sort_indices()
            self.mass_solver = gpu_direct.CuDSSDirect(comm, spd=True)
            self.mass_solver.factorize_csr(n, Mc.indptr, Mc.indices, Mc.data)
        K = sp.bmat([[lead, A], [A, -M]], format="csr")
        K.sort_indices()
        self.nnz = int(K.nnz)
        self.solver = gpu_direct.CuDSSDirect(comm, precision=precision)
        self.solver.factorize_csr(2 * n, K.indptr, K.indices, K.data)
        del K
        # the system has a negative eigenvalue for every row of the mass matrix, and one more
        # for every negative eigenvalue of H_s + A M^-1 A: a matrix cut off at a reach too
        # short for its grid is not positive, and what its solves give is no posterior
        inertia = self.solver.inertia() if self.method == "direct" else None
        self.negative = None if inertia is None else inertia[1] - n
        if self.negative:
            raise RuntimeError("the sparse Hessian plus the prior is not positive definite (%d negative eigenvalues): "
                               "take a longer reach" % self.negative)
        self.prior_solver = gpu_direct.CuDSSDirect(comm, spd=True)
        self.prior_solver.factorize_csr(n, A.indptr, A.indices, A.data)
        self.nd = len(misfit.scipy_BW()[1])
        self.rng = np.random.default_rng(seed)
        self._noise = prior.noise_vector()
        self._rhs = problem.Vm.vector()
        self.Hlr = _Solve(self)

    def _system(self, B, C=None):
        """The first block of the solution of the sparse system with right-hand sides ``(B, C)``."""
        import cupy as cp

        n, k = self.n, B.shape[1]
        rhs = cp.zeros((2 * n, k), order="F")
        rhs[:n] = B
        if C is not None:
            rhs[n:] = C
        return self.solver.solve_many(rhs, refine=1)[:n]

    def solve(self, B, C=None):
        """``(H_s + A M^-1 A)^-1 (B + A M^-1 C)`` for the columns of ``B`` (and ``C``)."""
        import cupy as cp

        B = cp.asarray(B)
        C = None if C is None else cp.asarray(C)
        if self.method == "direct":
            Y = self._system(B, C)
        else:
            if C is not None:
                B = B + self.A @ self.mass_solver.solve_many(cp.asfortranarray(C), refine=1)
            Y = self._cg(B)
        Y = cp.asnumpy(Y)
        cp.get_default_memory_pool().free_all_blocks()
        return Y

    def _cg(self, B):
        """Conjugate gradients on ``H_s + A M^-1 A``, every column its own iteration."""
        import cupy as cp

        def apply(P):
            return self.Hs @ P + self.A @ self.mass_solver.solve_many(cp.asfortranarray(self.A @ P), refine=1)

        X = cp.zeros_like(B)
        Rr = B.copy()
        b = cp.maximum(cp.linalg.norm(B, axis=0), 1e-300)
        Z = self._system(Rr)
        P = Z.copy()
        rz = (Rr * Z).sum(axis=0)
        for it in range(self.maxit):
            KP = apply(P)
            den = (P * KP).sum(axis=0)
            if bool((den[cp.linalg.norm(P, axis=0) > 0] <= 0).any()):
                raise RuntimeError("the sparse Hessian plus the prior is not positive definite: take a longer reach")
            alpha = cp.where(den > 0, rz / cp.where(den > 0, den, 1.0), 0.0)
            X += P * alpha
            Rr -= KP * alpha
            res = float((cp.linalg.norm(Rr, axis=0) / b).max())
            if res < self.rtol:
                break
            Z = self._system(Rr)
            rz_new = (Rr * Z).sum(axis=0)
            P = Z + P * cp.where(rz > 0, rz_new / cp.where(rz > 0, rz, 1.0), 0.0)
            rz = rz_new
        self.iterations.append((it + 1, res))
        return X

    def prior_solve(self, C):
        """``A^-1 C``: with ``C ~ N(0, M)`` samples of the prior."""
        import cupy as cp

        Y = cp.asnumpy(self.prior_solver.solve_many(cp.asarray(C, order="F"), refine=1))
        cp.get_default_memory_pool().free_all_blocks()
        return Y

    def _prior_noise(self, k, noises=None):
        """``k`` columns ``sqrt(M) xi``, from the library's noise stream (or the given noises)."""
        C = np.zeros((self.n, k))
        for i in range(k):
            noise = noises[i] if noises is not None else self.prior.sample_noise(1.0, self._noise)
            self.prior.sqrtM.mult(noise, self._rhs)
            C[:, i] = self._rhs.array
        return C

    def draw(self, k, noises=None):
        """``k`` zero-mean samples of the prior and of the posterior, as columns."""
        C = self._prior_noise(k, noises)
        if self.draws == "lanczos":
            import cupy as cp

            B = lanczos_sqrt(lambda P: self.Hs @ P, cp.asarray(self.rng.standard_normal((self.n, k))))
        else:
            Z = self.rng.standard_normal((self.nd, k))
            B = self.pb.misfit_hessian_sqrt_block(self.misfit, Z)   # covariance: the misfit Hessian itself
        return self.prior_solve(C), self.solve(B, C)

    def sample(self, noise, s_prior, s_post, add_mean=True):
        sp_, so = self.draw(1, noises=[noise])
        s_prior.array[:] = sp_[:, 0]
        s_post.array[:] = so[:, 0]
        if add_mean:
            s_prior.axpy(1.0, self.prior.mean)
            s_post.axpy(1.0, self.mean)

    def describe(self):
        out = dict(method=self.method, draws=self.draws, system_nonzeros=self.nnz, solver=self.solver.describe(),
                   prior_solver=self.prior_solver.describe())
        if self.iterations:
            its = np.array([i for i, _ in self.iterations])
            out.update(cg_solves=len(its), cg_iterations_mean=float(its.mean()), cg_iterations_max=int(its.max()),
                       cg_residual_max=float(max(r for _, r in self.iterations)))
        return out


class _Solve:
    def __init__(self, post):
        self.post = post

    def solve(self, out, rhs):
        out.array[:] = self.post.solve(np.asarray(rhs.array)[:, None])[:, 0]

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Randomized SVD.

Needed for non-symmetric reduced operators, above all the parameter-to-observable
map :math:`B A^{-1} C` of a linear inverse problem, whose singular values say how
much of the parameter the data can see.
"""

import numpy as np

from .multivector import MatMvMult, MatMvTranspmult, MultiVector, MvDSmatMult


def accuracyEnhancedSVD(A, Omega, k, s=1, check=False):
    r"""Randomized SVD of a (possibly non-symmetric) operator.

    Returns ``(U, sigma, V)`` with ``U`` in the range, ``V`` in the domain, and
    :math:`A \approx U \,\mathrm{diag}(\sigma)\, V^{\!\top}`.

    ``s`` power iterations are applied with re-orthogonalization at every step,
    which is what keeps the small singular values from being swamped.
    """
    nvec = Omega.nvec()
    if nvec < k:
        raise ValueError("Omega needs at least k = %d columns, has %d" % (k, nvec))

    Ybar = MultiVector(Omega[0], nvec)
    Q = MultiVector(Omega)
    Q.orthogonalize()
    for _ in range(s):
        MatMvMult(A, Q, Ybar)               # range side
        Ybar.orthogonalize()
        MatMvTranspmult(A, Ybar, Q)         # back to the domain
        Q.orthogonalize()

    # Q spans the dominant right singular subspace
    AQ = MultiVector(Ybar[0], nvec)
    MatMvMult(A, Q, AQ)
    B = AQ.dot_mv(AQ)                        # (A Q)^T (A Q)
    B = 0.5 * (B + B.T)
    evals, evecs = np.linalg.eigh(B)
    order = evals.argsort()[::-1][:k]
    sigma = np.sqrt(np.maximum(evals[order], 0.0))
    W = evecs[:, order]

    V = MultiVector(Omega[0], k)
    MvDSmatMult(Q, W, V)
    U = MultiVector(Ybar[0], k)
    MvDSmatMult(AQ, W, U)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(sigma > 0, 1.0 / sigma, 0.0)
    U.scale(scale)

    if check:
        return U, sigma, V, check_SVD(A, U, sigma, V)
    return U, sigma, V


def singlePassSVD(A, Omega, k, check=False):
    """Randomized SVD: :func:`accuracyEnhancedSVD` with ``s=1``, its default."""
    return accuracyEnhancedSVD(A, Omega, k, s=1, check=check)


def check_SVD(A, U, sigma, V):
    r"""Diagnostics: ``(||A v_i - s_i u_i||, ||U^T U - I||, ||V^T V - I||)``."""
    k = U.nvec()
    AV = MultiVector(U[0], k)
    MatMvMult(A, V, AV)
    res = np.zeros(k)
    for i in range(k):
        r = AV[i].copy().axpy(-sigma[i], U[i])
        res[i] = r.norm("l2")
    eU = np.linalg.norm(U.dot_mv(U) - np.eye(k), "fro")
    eV = np.linalg.norm(V.dot_mv(V) - np.eye(k), "fro")
    return res, eU, eV


#: snake_case spellings (see :mod:`hippymfem.common.naming`)
accuracy_enhanced_svd = accuracyEnhancedSVD
single_pass_svd = singlePassSVD

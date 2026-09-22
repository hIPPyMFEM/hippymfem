# Derived from hIPPYlib / hIPPYlibx (https://hippylib.github.io):
# Copyright (c) 2016-2018, The University of Texas at Austin & University of
# California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin, University of
# California--Merced, Washington University in St. Louis.
# Copyright (c) 2025-, Georgia Institute of Technology.
# Modified in 2026 for MFEM, hypre and JAX by Peng Chen, Georgia Institute of
# Technology.  See the file COPYRIGHT for details.
#
# hIPPyMFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software
# Foundation) version 2.0 dated June 1991.  See the file LICENSE.
r"""Randomized eigensolvers for the (generalized) Hermitian eigenvalue problem.

The Laplace approximation needs the dominant eigenpairs of the prior-preconditioned
data misfit Hessian, i.e. :math:`H_{\rm misfit} u = \lambda R u`.  The dominant
spectrum decays fast for ill-posed inverse problems, which is exactly the regime
where randomized methods are appropriate.

References
----------
1. N. Halko, P.G. Martinsson, J.A. Tropp, *Finding structure with randomness*,
   SIAM Review 53(2), 2011.
2. A.K. Saibaba, J. Lee, P.K. Kitanidis, *Randomized algorithms for generalized
   Hermitian eigenvalue problems*, NLAA 23(2), 2016.
"""

import numpy as np

from ..common.operators import Solver2Operator
from .multivector import MatMvMult, MultiVector, MvDSmatMult


def _top_k(T, k):
    """The ``k`` largest eigenpairs of a small symmetric matrix, descending."""
    d, V = np.linalg.eigh(T)
    order = d.argsort()[::-1][:k]
    return d[order], V[:, order]


def singlePass(A, Omega, k, s=1, check=False):
    r"""Dominant eigenpairs of ``A`` from one pass over the operator.

    Returns ``(d, U)`` with :math:`U^{\!\top}U = I_k`.
    """
    nvec = Omega.nvec()
    if nvec < k:
        raise ValueError("Omega needs at least k = %d columns, has %d" % (k, nvec))

    Y_pr = MultiVector(Omega)
    Y = MultiVector(Omega)
    for i in range(s):
        Y_pr.swap(Y)
        if i:
            Y_pr.orthogonalize()          # see doublePass: keeps the tail through the power iterations
        MatMvMult(A, Y_pr, Y)

    Q = MultiVector(Y)
    Q.orthogonalize()

    Zt = Y_pr.dot_mv(Q)
    Wt = Y.dot_mv(Q)
    Tt = np.linalg.solve(Zt, Wt)
    T = 0.5 * (Tt + Tt.T)

    d, V = _top_k(T, k)
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    if check:
        check_std(A, U, d)
    return d, U


def doublePass(A, Omega, k, s=1, check=False):
    r"""Dominant eigenpairs of ``A`` with ``s`` power iterations and two passes.

    More accurate than :func:`singlePass` at the cost of ``k`` extra operator
    applications; this is the default for the Laplace approximation.
    """
    nvec = Omega.nvec()
    if nvec < k:
        raise ValueError("Omega needs at least k = %d columns, has %d" % (k, nvec))

    Q = MultiVector(Omega)
    Y = MultiVector(Omega[0], nvec)
    for _ in range(s):
        MatMvMult(A, Q, Y)
        Q.swap(Y)
        # Orthonormalize after *every* application, not once at the end (subspace
        # iteration): the s-th power raises the spectral ratio to its s-th power,
        # and once that passes 1/eps the trailing directions are gone from Y before
        # any orthonormalization can save them.  Measured on the validation case
        # (ratio 3e6 over 40 eigenvalues): with one orthonormalization at the end a
        # third iteration put the last twenty eigenvalues off by 30 to 90 %; with
        # one per application every extra iteration helps.  s = 1 is unchanged.
        Q.orthogonalize()

    AQ = MultiVector(Omega[0], nvec)
    MatMvMult(A, Q, AQ)
    T = AQ.dot_mv(Q)

    d, V = _top_k(T, k)
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    if check:
        check_std(A, U, d)
    return d, U


def singlePassG(A, B, Binv, Omega, k, s=1, check=False):
    r"""Single-pass solver for :math:`A u = \lambda B u`, with :math:`U^{\!\top}BU=I`."""
    nvec = Omega.nvec()
    if nvec < k:
        raise ValueError("Omega needs at least k = %d columns, has %d" % (k, nvec))

    Ybar = MultiVector(Omega[0], nvec)
    Y_pr = MultiVector(Omega)
    Q = MultiVector(Omega)
    for i in range(s):
        Y_pr.swap(Q)
        if i:
            Y_pr.Borthogonalize(B)        # see doublePass
        MatMvMult(A, Y_pr, Ybar)
        MatMvMult(_as_operator(Binv), Ybar, Q)

    BQ, _ = Q.Borthogonalize(B)
    Xt = Y_pr.dot_mv(BQ)
    Wt = Ybar.dot_mv(Q)
    Tt = np.linalg.solve(Xt, Wt)
    T = 0.5 * (Tt + Tt.T)

    d, V = _top_k(T, k)
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    if check:
        check_g(A, B, U, d)
    return d, U


def doublePassG(A, B, Binv, Omega, k, s=1, check=False):
    r"""Double-pass solver for :math:`A u = \lambda B u`, with :math:`U^{\!\top}BU=I`.

    ``A`` is the data-misfit Hessian, ``B`` the prior precision ``R`` and ``Binv``
    a solver for it.
    """
    nvec = Omega.nvec()
    if nvec < k:
        raise ValueError("Omega needs at least k = %d columns, has %d" % (k, nvec))

    Ybar = MultiVector(Omega[0], nvec)
    Q = MultiVector(Omega)
    for _ in range(s):
        MatMvMult(A, Q, Ybar)
        MatMvMult(_as_operator(Binv), Ybar, Q)
        Q.Borthogonalize(B)               # after every application; see doublePass

    AQ = MultiVector(Omega[0], nvec)
    MatMvMult(A, Q, AQ)
    T = AQ.dot_mv(Q)

    d, V = _top_k(T, k)
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    if check:
        check_g(A, B, U, d)
    return d, U


def _as_operator(S):
    """Accept either an operator (``mult``) or a solver (``solve``)."""
    if hasattr(S, "mult"):
        return S
    return Solver2Operator(S)


# ------------------------------------------------------------------- diagnostics
def check_std(A, U, d):
    """Orthogonality and residual diagnostics for a standard eigenproblem.

    Returns ``(err_AU_UD, err_UtU_I, err_UtAU_D)``; all three should be small.
    """
    nvec = U.nvec()
    AU = MultiVector(U[0], nvec)
    MatMvMult(A, U, AU)

    # columnwise residual ||A u_i - d_i u_i||
    err = np.zeros(nvec)
    for i in range(nvec):
        r = AU[i].copy().axpy(-d[i], U[i])
        err[i] = r.norm("l2")

    UtU = U.dot_mv(U)
    err_Bortho = np.linalg.norm(UtU - np.eye(nvec), "fro")
    V = U.dot_mv(AU)
    err_QtQ = np.linalg.norm(V - np.diag(d), "fro") / max(np.linalg.norm(d), 1e-300)
    return err, err_Bortho, err_QtQ


def check_g(A, B, U, d):
    """Diagnostics for a generalized eigenproblem: residual and ``B``-orthogonality."""
    nvec = U.nvec()
    AU = MultiVector(U[0], nvec)
    BU = MultiVector(U[0], nvec)
    MatMvMult(A, U, AU)
    MatMvMult(B, U, BU)

    err = np.zeros(nvec)
    for i in range(nvec):
        r = AU[i].copy().axpy(-d[i], BU[i])
        err[i] = r.norm("l2")

    UtBU = U.dot_mv(BU)
    err_Bortho = np.linalg.norm(UtBU - np.eye(nvec), "fro")
    V = U.dot_mv(AU)
    err_AV = np.linalg.norm(V - np.diag(d), "fro") / max(np.linalg.norm(d), 1e-300)
    return err, err_Bortho, err_AV


#: snake_case spellings (see :mod:`hippymfem.common.naming`)
single_pass = singlePass
double_pass = doublePass
single_pass_g = singlePassG
double_pass_g = doublePassG

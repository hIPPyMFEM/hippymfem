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
r"""Low-rank operators built from a spectral decomposition.

Represents :math:`U D U^{\!\top}` (optionally in a :math:`B` inner product, where
the columns of ``U`` are :math:`B`-orthonormal).  Used by the Laplace
approximation, where ``d`` and ``U`` come from the randomized generalized
eigensolver.
"""

import numpy as np

from ..common.operators import Operator, init_vector_like
from .multivector import MultiVector


class LowRankOperator(Operator):
    r""":math:`A = U\,\mathrm{diag}(d)\,U^{\!\top} B`.

    Parameters
    ----------
    d : ndarray
    U : MultiVector
        Columns :math:`B`-orthonormal: :math:`U^{\!\top}BU = I`.
    my_init_vector : callable, optional
        Supplies the layout, when ``U``'s own is not what the caller wants.
    """

    def __init__(self, d, U, my_init_vector=None):
        self.d = np.asarray(d, dtype=float)
        self.U = U
        self.my_init_vector = my_init_vector

    def init_vector(self, x, dim):
        if self.my_init_vector is not None:
            return self.my_init_vector(x, dim)
        return init_vector_like(x, self.U[0])

    def mult(self, x, y):
        """``y = U diag(d) U^T x``."""
        Utx = self.U.dot_v(x)
        y.zero()
        self.U.reduce(y, self.d * Utx)
        return y

    multTranspose = mult

    def inner(self, x, y):
        Utx = self.U.dot_v(x)
        Uty = self.U.dot_v(y)
        return float(np.sum(self.d * Utx * Uty))

    def solve(self, sol, rhs):
        """``sol = U diag(1/d) U^T rhs`` (the pseudo-inverse on the range of U)."""
        Utr = self.U.dot_v(rhs)
        sol.zero()
        with np.errstate(divide="ignore", invalid="ignore"):
            coeff = np.where(np.abs(self.d) > 0, Utr / self.d, 0.0)
        self.U.reduce(sol, coeff)
        return sol

    def get_diagonal(self, diag):
        r"""Diagonal of :math:`U \mathrm{diag}(d) U^{\!\top}`."""
        diag.zero()
        for i in range(self.U.nvec()):
            diag.array[:] += self.d[i] * self.U[i].array ** 2
        return diag

    def trace(self, W=None):
        r"""Trace of :math:`U D U^{\!\top} W`; ``W = None`` means the identity."""
        if W is None:
            UtU = self.U.dot_mv(self.U)
            return float(np.sum(self.d * np.diag(UtU)))
        WU = MultiVector(self.U[0], self.U.nvec())
        for i in range(self.U.nvec()):
            W.mult(self.U[i], WU[i])
        UtWU = self.U.dot_mv(WU)
        return float(np.sum(self.d * np.diag(UtWU)))

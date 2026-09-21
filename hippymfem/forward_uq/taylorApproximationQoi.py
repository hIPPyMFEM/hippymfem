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
r"""Taylor approximations of the parameter-to-QoI map, and their moments.

Around the mean :math:`\bar m` of a Gaussian :math:`m \sim N(\bar m, R^{-1})`,

.. math:: \mathcal{Q}(m) \approx \bar q + g^{\!\top}\delta m
          + \tfrac12 \delta m^{\!\top}\mathcal{H}\,\delta m .

Both moments of that quadratic model are available in closed form from the
eigenvalues :math:`d_i` of the prior-preconditioned Hessian:

.. math:: \mathbb{E}[\mathcal{Q}] \approx \bar q + \tfrac12\sum_i d_i, \qquad
          \mathrm{Var}[\mathcal{Q}] \approx g^{\!\top}R^{-1}g
          + \tfrac12\sum_i d_i^2 .

That is the point of the construction: a handful of Hessian eigenpairs replaces a
Monte Carlo sum, and the same quadratic model then serves as a control variate
for a small sample (see :mod:`.varianceReductionMC`).
"""

import numpy as np

from ..common.naming import sync_spellings
from ..algorithms.randomizedEigensolver import doublePassG
from ..modeling.variables import ADJOINT, PARAMETER, STATE


class TaylorApproximationQoi:
    """First- and second-order Taylor model of a parameter-to-QoI map.

    Parameters
    ----------
    p2qoimap : Parameter2QoiMap
    distribution : a prior, or a :class:`~hippymfem.modeling.posterior.GaussianLRPosterior`
        The Gaussian the parameter is drawn from; the expansion point is its mean.
    """

    def __init__(self, p2qoimap, distribution):
        self.p2qoimap = p2qoimap
        self.distribution = distribution
        self.x_bar = p2qoimap.generate_vector()
        self.x_bar[PARAMETER].axpy(1.0, distribution.mean)
        self.q_bar = 0.0
        self.g_bar = p2qoimap.generate_vector(PARAMETER)
        self.d = None
        self.U = None
        self.H = None

    # -------------------------------------------------------------- factorize
    def computeLowRankFactorization(self, Omega, k=None, s=1, check=False):
        """Solve at the mean and factor the prior-preconditioned Hessian.

        Must be called before the moment methods.  ``Omega`` supplies the
        randomized sketch; ``k`` defaults to all of its columns.
        """
        p = self.p2qoimap
        p.solveFwd(self.x_bar[STATE], self.x_bar)
        p.solveAdj(self.x_bar[ADJOINT], self.x_bar)
        self.q_bar = p.eval(self.x_bar)
        p.evalGradientParameter(self.x_bar, self.g_bar)
        self.H = p.hessian(x=self.x_bar)
        k = Omega.nvec() if k is None else int(k)
        if hasattr(self.distribution, "R"):
            B, Binv = self.distribution.R, self.distribution.Rsolver
        else:
            # A low-rank posterior plays the same role, but its precision operator
            # carries both the precision (``mult``) and the covariance (``solve``).
            # The eigensolver takes anything with ``mult`` at face value, so the
            # covariance has to be handed over as an operator of its own; passing
            # the precision twice applies it where the covariance is meant, and the
            # eigenvalues come back near zero.
            from ..common.operators import Solver2Operator

            B = self.distribution.Hlr
            Binv = Solver2Operator(B)
        self.d, self.U = doublePassG(self.H, B, Binv, Omega, k, s=s, check=check)
        return self.d, self.U

    def _require(self):
        if self.d is None:
            raise RuntimeError("computeLowRankFactorization must be called first")

    # ----------------------------------------------------------------- moments
    def expectedValue(self, order=2):
        """Expected value of the Taylor model (analytic)."""
        self._require()
        if order == 1:
            return self.q_bar
        if order == 2:
            return self.q_bar + 0.5 * float(np.sum(self.d))
        raise ValueError("order must be 1 or 2")

    def variance(self, order=2):
        """Variance of the Taylor model (analytic)."""
        self._require()
        Rinv_g = self.p2qoimap.generate_vector(PARAMETER)
        if hasattr(self.distribution, "Rsolver"):
            self.distribution.Rsolver.solve(Rinv_g, self.g_bar)
        else:
            self.distribution.Hlr.solve(Rinv_g, self.g_bar)
        lin_var = Rinv_g.inner(self.g_bar)
        if order == 1:
            return lin_var
        if order == 2:
            return lin_var + 0.5 * float(np.sum(self.d ** 2))
        raise ValueError("order must be 1 or 2")

    # ------------------------------------------------------------------ eval
    def eval(self, m, order=2):
        """Evaluate the Taylor model at a realization ``m``."""
        self._require()
        dm = m.copy().axpy(-1.0, self.x_bar[PARAMETER])
        q = self.q_bar + self.g_bar.inner(dm)
        if order == 1:
            return q
        if order == 2:
            return q + 0.5 * self.H.inner(dm, dm)
        raise ValueError("order must be 1 or 2")


sync_spellings(TaylorApproximationQoi)

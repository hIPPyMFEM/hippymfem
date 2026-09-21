# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Hutchinson trace estimation with a running error estimate.

Used for the integrated prior variance when the exact diagonal would cost one
solve per degree of freedom.
"""

import math

from ..common.operators import make_vector
from ..common.random import parRandom


class TraceEstimator:
    r"""Estimate :math:`\mathrm{tr}(A)` by averaging :math:`z^{\!\top}Az` over
    Rademacher vectors ``z``.

    Parameters
    ----------
    A : operator
    solve_mode : bool
        Use ``A.solve`` instead of ``A.mult`` (so the trace of the inverse).
    accuracy : float
        Target relative standard error; iteration stops once reached.
        (``accurancy``, hIPPYlib's spelling, is accepted too.)
    init_vector : callable, optional
    """

    def __init__(self, A, solve_mode=False, accurancy=1e-1, init_vector=None,
                 random_engine=None, accuracy=None):
        self.A = A
        self.solve_mode = bool(solve_mode)
        self.accuracy = float(accurancy if accuracy is None else accuracy)
        self.random_engine = random_engine if random_engine is not None else parRandom
        if init_vector is None:
            self.z = A.generate_vector(0)
            self.Az = A.generate_vector(0)
        else:
            holder = _InitVectorHolder(init_vector)
            self.z = make_vector(holder, 0)
            self.Az = make_vector(holder, 0)

    def _apply(self):
        if self.solve_mode:
            self.Az.zero()
            self.A.solve(self.Az, self.z)
        else:
            self.A.mult(self.z, self.Az)
        return self.Az.inner(self.z)

    def __call__(self, min_iter=5, max_iter=100):
        """Return ``(estimate, standard_error)``."""
        sum_q = 0.0
        sum_q2 = 0.0
        for i in range(max_iter):
            self.random_engine.rademacher(self.z)
            q = self._apply()
            sum_q += q
            sum_q2 += q * q
            n = i + 1
            if n >= min_iter:
                mean = sum_q / n
                var = max(sum_q2 / n - mean * mean, 0.0)
                err = math.sqrt(var / n)
                if err < self.accuracy * abs(mean) or abs(mean) < 1e-300:
                    return mean, err
        n = max_iter
        mean = sum_q / n
        var = max(sum_q2 / n - mean * mean, 0.0)
        return mean, math.sqrt(var / n)


class _InitVectorHolder:
    """Adapt a bare ``init_vector`` callable to the object protocol."""

    def __init__(self, fn):
        self.init_vector = fn


#: hIPPYlib's spelling of the attribute, kept readable and assignable
TraceEstimator.accurancy = property(lambda self: self.accuracy,
                                    lambda self, v: setattr(self, "accuracy", float(v)))

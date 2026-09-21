# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Chain diagnostics."""

import numpy as np


def _acorr(mean_free, lag, norm=1.0):
    n = mean_free.size
    if lag >= n:
        return 0.0
    return float(np.dot(mean_free[: n - lag], mean_free[lag:]) / (n * norm))


def _acorr_vs_lag(samples, max_lag):
    s = np.asarray(samples, dtype=float).ravel()
    mean_free = s - s.mean()
    norm = _acorr(mean_free, 0, 1.0)
    if norm <= 0.0:
        return np.zeros(max_lag + 1)
    return np.array([_acorr(mean_free, k, norm) for k in range(max_lag + 1)])


def integratedAutocorrelationTime(samples, max_lag=None):
    """Integrated autocorrelation time, by the initial-positive-sequence rule.

    Returns ``(iact, lags, autocorrelation)``.  The sum is truncated at the first
    negative autocorrelation, which is the standard way to keep the estimator
    from accumulating noise at large lags.
    """
    s = np.asarray(samples, dtype=float).ravel()
    if max_lag is None:
        max_lag = min(s.size - 1, max(10, s.size // 10))
    ac = _acorr_vs_lag(s, int(max_lag))
    neg = np.nonzero(ac[1:] < 0.0)[0]
    cut = int(neg[0]) + 1 if neg.size else ac.size
    iact = 1.0 + 2.0 * float(np.sum(ac[1:cut]))
    return iact, np.arange(ac.size), ac


def effective_sample_size(samples, max_lag=None):
    """Number of effectively independent samples in the chain."""
    s = np.asarray(samples, dtype=float).ravel()
    iact, _, _ = integratedAutocorrelationTime(s, max_lag)
    return s.size / max(iact, 1e-300)


def chain_summary(samples, max_lag=None):
    """Mean, standard error, IACT and effective sample size."""
    s = np.asarray(samples, dtype=float).ravel()
    iact, _, _ = integratedAutocorrelationTime(s, max_lag)
    ess = s.size / max(iact, 1e-300)
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "iact": float(iact),
        "ess": float(ess),
        "standard_error": float(s.std(ddof=1) / np.sqrt(max(ess, 1e-300)))
        if s.size > 1 else 0.0,
    }


#: snake_case spellings (see :mod:`hippymfem.common.naming`)
integrated_autocorrelation_time = integratedAutocorrelationTime

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The Armijo backtracking step shared by the line-search optimizers."""

import numpy as np

from ..modeling.variables import PARAMETER, STATE


def armijo_backtrack(model, x, x_star, mhat, mg_mhat, cost_old, alpha=1.0,
                     c_armijo=1e-4, max_backtracking=10, gdm_tolerance=None,
                     bounds=None, failures=(RuntimeError,)):
    """Try ``x + alpha * mhat``, halving ``alpha`` until the cost decreases enough.

    The trial point is formed in ``x_star`` (parameter and state; the state is
    started from ``x``'s), solved forward, and accepted when
    ``cost < cost_old + alpha * c_armijo * mg_mhat`` (``mg_mhat`` is the directional
    derivative, negative for a descent direction), or when ``gdm_tolerance`` is given
    and ``-mg_mhat`` is already below it.  A forward solve that raises one of
    ``failures`` counts as no decrease.  ``bounds = (lo, hi)`` clips the trial
    parameter first.

    Returns ``(accepted, alpha, n_backtrack, cost, fwd_failed)``: on acceptance
    ``x`` holds the new parameter and state and ``cost`` is ``model.cost`` there;
    otherwise ``x`` is unchanged, ``n_backtrack == max_backtracking`` and ``cost``
    is the last trial's (its total ``cost_old + 1`` and the rest ``nan`` when that
    trial's forward solve failed).
    """
    n_back = 0
    fwd_failed = False
    cost = None
    while n_back < max_backtracking:
        x_star[PARAMETER].assign(x[PARAMETER]).axpy(alpha, mhat)
        if bounds is not None:
            lo, hi = bounds
            np.clip(x_star[PARAMETER].array, lo.array, hi.array,
                    out=x_star[PARAMETER].array)
        x_star[STATE].assign(x[STATE])
        try:
            model.solveFwd(x_star[STATE], x_star)
            cost = list(model.cost(x_star))
            fwd_failed = False
        except failures:
            cost = [cost_old + 1.0, float("nan"), float("nan")]   # no sufficient decrease
            fwd_failed = True
        if (cost[0] < cost_old + alpha * c_armijo * mg_mhat
                or (gdm_tolerance is not None and -mg_mhat <= gdm_tolerance)):
            x[PARAMETER].assign(x_star[PARAMETER])
            x[STATE].assign(x_star[STATE])
            return True, alpha, n_back, cost, fwd_failed
        n_back += 1
        alpha *= 0.5
    return False, alpha, n_back, cost, fwd_failed

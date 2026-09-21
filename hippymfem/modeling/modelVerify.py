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
r"""Finite-difference verification of the gradient and the Hessian.

Two checks, both from hIPPYlib:

* the cost should satisfy
  :math:`J(m+h\,\tilde m) = J(m) + h\,g^{\!\top}\tilde m + O(h^2)`, so the
  error in the first-order model falls like :math:`h`;
* the gradient should satisfy
  :math:`g(m+h\tilde m) = g(m) + h\,H\tilde m + O(h^2)`, likewise.

Seeing those slopes shows that the adjoint gradient and the second-order blocks
are consistent with the cost.
"""

import numpy as np

from ..common.random import parRandom
from .reducedHessian import ReducedHessian
from .variables import ADJOINT, PARAMETER, STATE


def modelVerify(model, m0, is_quadratic=False, misfit_only=False, verbose=True,
                eps=None, plotting=False, filename=None):
    """Run the finite-difference checks at ``m0``.

    Parameters
    ----------
    model : Model
    m0 : ParVector
        The parameter at which the derivatives are checked.
    is_quadratic : bool
        The cost is quadratic in ``m``, so the Hessian error should be round-off.
    misfit_only : bool
        Check the misfit part of the cost only.
    verbose : bool
        Print the table of errors and observed slopes on rank 0.
    eps : array_like, optional
        Finite-difference steps; halvings from 1 down to 1e-10 by default.
    plotting : bool
        Draw both error curves against a first-order reference.  The figure is
        left open, so a notebook shows it inline.
    filename : str, optional
        Also save the figure there, and close it.

    Returns
    -------
    eps : ndarray
        Step sizes used.
    err_grad : ndarray
        :math:`|(J(m+h\\tilde m) - J(m))/h - g^{\\!\\top}\\tilde m|`.
    err_H : ndarray
        :math:`\\|(g(m+h\\tilde m) - g(m))/h - H \\tilde m\\|_\\infty`.
    """
    index = 2 if misfit_only else 0

    h = model.generate_vector(PARAMETER)
    parRandom.normal(1.0, h)

    x = model.generate_vector()
    x[PARAMETER] = m0
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)
    cx = model.cost(x)

    grad_x = model.generate_vector(PARAMETER)
    model.evalGradientParameter(x, grad_x, misfit_only=misfit_only)
    grad_xh = grad_x.inner(h)

    model.setPointForHessianEvaluations(x)
    H = ReducedHessian(model, misfit_only=misfit_only)
    Hh = model.generate_vector(PARAMETER)
    H.mult(h, Hh)

    if eps is None:
        n_eps = 32
        eps = np.power(0.5, np.arange(n_eps))
        eps = eps[eps > 1e-10]
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    err_grad = np.zeros(eps.shape)
    err_H = np.zeros(eps.shape)

    comm = m0.comm
    if verbose and comm.rank == 0:
        print("%9s %13s %13s %13s %13s"
              % ("eps", "||err grad||", "||err H||", "J(m+h)", "slope g"), flush=True)

    prev = None
    for i, e in enumerate(eps):
        my_eps = float(e)
        x_plus = model.generate_vector()
        x_plus[PARAMETER] = m0.copy().axpy(my_eps, h)
        model.solveFwd(x_plus[STATE], x_plus)
        model.solveAdj(x_plus[ADJOINT], x_plus)

        # model.cost is collective (the misfit reduces over ranks and the prior
        # solves a linear system): call it on every rank, not inside the rank-0
        # print below, which would deadlock.
        cost_plus = model.cost(x_plus)
        dc = cost_plus[index] - cx[index]
        err_grad[i] = abs(dc / my_eps - grad_xh)

        grad_xplus = model.generate_vector(PARAMETER)
        model.evalGradientParameter(x_plus, grad_xplus, misfit_only=misfit_only)
        err = grad_xplus.copy().axpy(-1.0, grad_x).scale(1.0 / my_eps)
        err.axpy(-1.0, Hh)
        err_H[i] = err.norm("linf")

        if verbose and comm.rank == 0:
            slope = ("%13.4f" % (np.log(err_grad[i] / prev) / np.log(0.5))
                     if prev not in (None, 0.0) and err_grad[i] > 0 else "%13s" % "-")
            print("%9.2e %13.6e %13.6e %13.6e %s"
                  % (my_eps, err_grad[i], err_H[i], cost_plus[index], slope),
                  flush=True)
        prev = err_grad[i]

    if is_quadratic:
        # For a quadratic cost the Hessian error is only the solver and round-off
        # error of the two gradients, divided by the step, so the tolerance scales
        # with |g| / eps + |H h|: an absolute one would fail a correct Hessian when
        # the gradient is large, while a wrong block still leaves an error of order
        # |H h| at the largest step.
        gnorm = grad_x.norm("linf")              # collective: all ranks
        hnorm = Hh.norm("linf")
        if verbose and comm.rank == 0:
            ok = bool(np.all(err_H <= 1e-8 * (gnorm / eps + hnorm)))
            print("Quadratic cost: the Hessian error should be ~0 everywhere: %s"
                  % ("yes" if ok else "NO"), flush=True)

    if plotting:
        _plot(eps, err_grad, err_H, comm, filename)

    return eps, err_grad, err_H


def fd_slopes(eps, err):
    """Observed convergence rates ``log2(err[i]/err[i+1])`` between steps."""
    eps = np.asarray(eps, float)
    err = np.asarray(err, float)
    good = (err[:-1] > 0) & (err[1:] > 0)
    out = np.full(err.size - 1, np.nan)
    out[good] = np.log(err[:-1][good] / err[1:][good]) / np.log(
        eps[:-1][good] / eps[1:][good])
    return out


def best_slope(eps, err, lo=1e-7, hi=1e-2):
    """Median observed slope over the step sizes where round-off has not taken over."""
    eps = np.asarray(eps, float)
    mask = (eps >= lo) & (eps <= hi)
    if mask.sum() < 3:
        mask = np.ones_like(eps, dtype=bool)
    s = fd_slopes(eps[mask], np.asarray(err, float)[mask])
    s = s[np.isfinite(s)]
    return float(np.median(s)) if s.size else float("nan")


def _plot(eps, err_grad, err_H, comm, filename):
    """Draw both error curves against O(h); save them when ``filename`` is given.

    The figure is left open otherwise, so a notebook shows it inline.  The backend
    is never switched here: forcing Agg from inside a library call would silently
    stop every later inline plot of the session.
    """
    if comm.rank != 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, y, lbl in ((ax[0], err_grad, "gradient"), (ax[1], err_H, "Hessian")):
        a.loglog(eps, y, "o-", label="error")
        ref = y[0] * (np.asarray(eps) / eps[0])
        a.loglog(eps, ref, "k--", label="O(h)")
        a.set_xlabel("h")
        a.set_ylabel("error")
        a.set_title(lbl)
        a.legend(fontsize=8)
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=130)
        plt.close(fig)
    return fig


#: snake_case spellings (see :mod:`hippymfem.common.naming`)
model_verify = modelVerify

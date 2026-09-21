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
r"""The parameter-to-QoI map and its derivatives.

For a QoI :math:`q(u, m)` constrained by the forward PDE, the reduced map is
:math:`\mathcal{Q}(m) = q(u(m), m)`.  Its gradient comes from the same adjoint
machinery as the inverse problem's, and its Hessian from the same incremental
solves:

.. math:: \nabla\mathcal{Q} = \partial_m q + C^{\!\top}p, \qquad
          \mathcal{H} = \text{(the reduced Hessian of }q).

The sign convention differs from the inverse problem's in one place: the adjoint
source is :math:`-\partial_u q` rather than :math:`-\partial_u\Phi`, and there is
no regularization term.
"""

import numpy as np

from ..modeling.model import ReducedMap
from ..modeling.reducedHessian import ReducedHessian
from ..modeling.variables import ADJOINT, PARAMETER, STATE


class Parameter2QoiMap(ReducedMap):
    """The reduced map ``m -> q(u(m), m)`` with adjoint derivatives.

    A :class:`~hippymfem.modeling.model.ReducedMap` whose functional is the QoI:
    the same solves and blocks as the inverse problem's :class:`~.model.Model`,
    without a prior.
    """

    def __init__(self, problem, qoi):
        super(Parameter2QoiMap, self).__init__(problem, qoi)
        self.qoi = qoi

    def eval(self, x):
        return self.qoi.eval(x)

    def evalGradientParameter(self, x, mg):
        r"""``mg`` = :math:`\partial_m q + C^{\!\top}p`."""
        return self._reduced_gradient(x, mg)

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        super(Parameter2QoiMap, self).setLinearizationPoint(x, gauss_newton_approx)
        self.x_lin = x
        return self

    # ------------------------------------------------------------- convenience
    def reduced_eval(self, m):
        """``Q(m)``: solve forward, then evaluate."""
        u = self.generate_vector(STATE)
        self.solveFwd(u, [u, m, None])
        return self.eval([u, m, None])

    def reduced_gradient(self, m, g=None):
        """``grad Q(m)``; returns ``(value, gradient)``."""
        x = self.generate_vector()
        x[PARAMETER] = m
        self.solveFwd(x[STATE], x)
        q = self.eval(x)
        self.solveAdj(x[ADJOINT], x)
        if g is None:
            g = self.generate_vector(PARAMETER)
        self.evalGradientParameter(x, g)
        self._last_x = x
        return q, g

    def hessian(self, m=None, x=None):
        """A :class:`Parameter2QoiHessian` at ``m`` (or at a given ``x``)."""
        if x is None:
            if m is None:
                raise ValueError("hessian needs m or x")
            _q, _g = self.reduced_gradient(m)
            x = self._last_x
        self.setLinearizationPoint(x)
        return Parameter2QoiHessian(self)


class Parameter2QoiHessian(ReducedHessian):
    """Matrix-free Hessian of the parameter-to-QoI map: the reduced Hessian of the
    map's functional, with no prior term."""

    def __init__(self, p2qoimap):
        super(Parameter2QoiHessian, self).__init__(p2qoimap, misfit_only=True)
        self.map = p2qoimap


def parameter2QoiMapVerify(p2qoimap, m0, eps=None, verbose=True):
    """Finite-difference check of the reduced gradient and Hessian.

    Returns ``(eps, err_grad, err_H)``, each of which should fall like ``h``.
    """
    from ..common.random import parRandom

    comm = m0.comm
    h = p2qoimap.generate_vector(PARAMETER)
    parRandom.normal(1.0, h)

    q0, g0 = p2qoimap.reduced_gradient(m0)
    gh = g0.inner(h)
    H = p2qoimap.hessian(x=p2qoimap._last_x)
    Hh = p2qoimap.generate_vector(PARAMETER)
    H.mult(h, Hh)

    if eps is None:
        eps = np.power(0.5, np.arange(2, 20))
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    err_grad = np.zeros(eps.size)
    err_H = np.zeros(eps.size)
    if verbose and comm.rank == 0:
        print("%9s %14s %14s" % ("eps", "||err grad||", "||err H||"), flush=True)
    for i, e in enumerate(eps):
        mp = m0.copy().axpy(float(e), h)
        qp, gp = p2qoimap.reduced_gradient(mp)
        err_grad[i] = abs((qp - q0) / e - gh)
        d = gp.copy().axpy(-1.0, g0).scale(1.0 / e)
        d.axpy(-1.0, Hh)
        err_H[i] = d.norm("linf")
        if verbose and comm.rank == 0:
            print("%9.2e %14.6e %14.6e" % (e, err_grad[i], err_H[i]), flush=True)
    return eps, err_grad, err_H


#: hIPPYlib's name for the verification helper
qoiVerify = parameter2QoiMapVerify

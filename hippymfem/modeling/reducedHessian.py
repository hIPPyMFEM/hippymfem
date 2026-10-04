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
r"""The reduced Hessian, applied matrix-free.

For a direction :math:`\tilde m`, the Gauss-Newton Hessian is

.. math::
   H_{GN}\tilde m = C^{\!\top} A^{-\top} W_{uu} A^{-1} C \tilde m + R \tilde m,

and the full Hessian adds the :math:`W_{um}`, :math:`W_{mu}` and :math:`W_{mm}`
terms.  The Gauss-Newton product needs no explicit sign: the incremental forward
solve returns :math:`\hat u = A^{-1}C\tilde m = -\,\mathrm{d}u/\mathrm{d}m[\tilde m]`,
and that sign cancels against the one in the adjoint source.  Each application
costs one incremental forward and one incremental adjoint solve, which is why
those solves dominate Newton-CG.
"""

from ..common.operators import Operator, init_vector_like
from .variables import ADJOINT, PARAMETER, STATE


class ReducedHessian(Operator):
    """Matrix-free reduced Hessian at the point set on the model.

    Parameters
    ----------
    model : Model
        ``setPointForHessianEvaluations`` must have been called.
    misfit_only : bool
        Drop the ``R`` term, giving the Hessian of the data misfit alone (the
        operator whose dominant eigenvectors define the Laplace approximation).
    """

    def __init__(self, model, misfit_only=False):
        self.model = model
        self.gauss_newton_approx = model.gauss_newton_approx
        self.misfit_only = bool(misfit_only)
        self.ncalls = 0

        self.rhs_fwd = model.generate_vector(STATE)
        self.rhs_adj = model.generate_vector(ADJOINT)
        self.rhs_adj2 = model.generate_vector(ADJOINT)
        self.uhat = model.generate_vector(STATE)
        self.phat = model.generate_vector(ADJOINT)
        self.yhelp = model.generate_vector(PARAMETER)
        #: relative tolerance of the two incremental solves of an action, where looser
        #: than the solvers' own (:meth:`set_accuracy`); ``None``: their own
        self.accuracy = None
        try:
            import inspect

            self._relaxable = "rel_tolerance" in inspect.signature(
                model.problem.solveIncremental).parameters
        except (AttributeError, TypeError, ValueError):
            self._relaxable = False

    def init_vector(self, x, dim):
        return init_vector_like(x, self.model.generate_vector(PARAMETER))

    def set_accuracy(self, tolerance):
        """Ask the next actions for incremental solves stopped at the relative
        tolerance ``tolerance``, if that is looser than the solvers' own (``None``:
        their own again).  A Krylov iteration can use products that become less
        accurate as it converges (``CGSolverSteihaug``, ``relax_operator``).  The prior's
        part of an action keeps its accuracy: a relative error of its mass solve is the
        same relative error of ``R x``, with nothing to damp it, and at the accuracies
        that the incremental solves tolerate it changed the path of the model problem."""
        self.accuracy = float(tolerance) if (tolerance and self._relaxable) else None
        return self

    def _incremental(self, adjoint, sol, rhs):
        solve = self.model.solveAdjIncremental if adjoint else self.model.solveFwdIncremental
        if self.accuracy is None:
            return solve(sol, rhs)
        return solve(sol, rhs, rel_tolerance=self.accuracy)

    def mult(self, x, y):
        if self.gauss_newton_approx:
            self.GNHessian(x, y)
        else:
            self.TrueHessian(x, y)
        self.ncalls += 1
        return y

    multTranspose = mult

    def inner(self, x, y):
        Hx = self.model.generate_vector(PARAMETER)
        self.mult(x, Hx)
        return Hx.inner(y)

    def GNHessian(self, x, y):
        # uhat = A^{-1} C x = -du/dm[x]; that sign cancels the adjoint source's,
        # so neither right-hand side is negated
        self.model.applyC(x, self.rhs_fwd)
        self._incremental(False, self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self._incremental(True, self.phat, self.rhs_adj)
        self.model.applyCt(self.phat, y)
        if not self.misfit_only:
            self.model.applyR(x, self.yhelp)
            y.axpy(1.0, self.yhelp)
        return y

    def TrueHessian(self, x, y):
        self.model.applyC(x, self.rhs_fwd)
        self._incremental(False, self.uhat, self.rhs_fwd)
        self.model.applyWuu(self.uhat, self.rhs_adj)
        self.model.applyWum(x, self.rhs_adj2)
        self.rhs_adj.axpy(-1.0, self.rhs_adj2)
        self._incremental(True, self.phat, self.rhs_adj)
        self.model.applyWmm(x, y)
        self.model.applyCt(self.phat, self.yhelp)
        y.axpy(1.0, self.yhelp)
        self.model.applyWmu(self.uhat, self.yhelp)
        y.axpy(-1.0, self.yhelp)
        if not self.misfit_only:
            self.model.applyR(x, self.yhelp)
            y.axpy(1.0, self.yhelp)
        return y


class FDHessian(Operator):
    """Reduced Hessian by central finite differences of the gradient.

    Slow, and only as accurate as the step size, but it needs nothing but
    ``solveFwd``/``solveAdj``/``evalGradientParameter``, which makes it the right
    reference for checking :class:`ReducedHessian`.
    """

    def __init__(self, model, m0, h, misfit_only=False):
        self.model = model
        self.m0 = m0.copy()
        self.h = float(h)
        self.misfit_only = bool(misfit_only)
        self.ncalls = 0

        self.state_plus = model.generate_vector(STATE)
        self.adj_plus = model.generate_vector(ADJOINT)
        self.g_plus = model.generate_vector(PARAMETER)
        self.state_minus = model.generate_vector(STATE)
        self.adj_minus = model.generate_vector(ADJOINT)
        self.g_minus = model.generate_vector(PARAMETER)

    def init_vector(self, x, dim):
        return init_vector_like(x, self.model.generate_vector(PARAMETER))

    def mult(self, x, y):
        h = self.h
        mp = self.m0.copy().axpy(h, x)
        self.model.solveFwd(self.state_plus, [self.state_plus, mp, self.adj_plus])
        self.model.solveAdj(self.adj_plus, [self.state_plus, mp, self.adj_plus])
        self.model.evalGradientParameter(
            [self.state_plus, mp, self.adj_plus], self.g_plus,
            misfit_only=self.misfit_only)

        mm = self.m0.copy().axpy(-h, x)
        self.model.solveFwd(self.state_minus, [self.state_minus, mm, self.adj_minus])
        self.model.solveAdj(self.adj_minus, [self.state_minus, mm, self.adj_minus])
        self.model.evalGradientParameter(
            [self.state_minus, mm, self.adj_minus], self.g_minus,
            misfit_only=self.misfit_only)

        y.assign(self.g_plus).axpy(-1.0, self.g_minus).scale(0.5 / h)
        self.ncalls += 1
        return y

    multTranspose = mult

    def inner(self, x, y):
        Hx = self.model.generate_vector(PARAMETER)
        self.mult(x, Hx)
        return Hx.inner(y)

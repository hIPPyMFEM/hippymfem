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
r"""The inverse problem: forward model + prior + misfit.

The cost functional is

.. math:: J(m) = \underbrace{\Phi(u(m))}_{\text{misfit}}
                 + \underbrace{\tfrac12\|m-\bar m\|_R^2}_{\text{regularization}},

and :class:`Model` assembles the adjoint-based gradient and the action of the
reduced Hessian from the blocks the PDE problem and misfit provide.  Signatures
follow hIPPYlib exactly.
"""

from ..common.naming import SnakeCamel, sync_spellings
from .variables import ADJOINT, PARAMETER, STATE


class ReducedMap(SnakeCamel):
    r"""A PDE problem and a functional of ``(u, m)`` reduced over the state.

    Both the inverse problem (:class:`Model`, whose functional is the data misfit)
    and the parameter-to-QoI map (:class:`~hippymfem.forward_uq.Parameter2QoiMap`,
    whose functional is the QoI) reduce a functional :math:`f(u(m), m)` through the
    same adjoint machinery: the adjoint source is :math:`-\partial_u f`, the reduced
    gradient :math:`\partial_m f + C^{\!\top}p`, and the reduced Hessian action
    combines the PDE's blocks with the functional's.  This class holds that
    machinery once; the subclasses add what differs (the prior and the cost for the
    inverse problem, the value for the QoI map).

    The functional supplies ``grad(i, x, out)``, ``setLinearizationPoint`` and
    ``apply_ij(i, j, d, out)``; the problem is a :class:`~.PDEProblem.PDEProblem`.
    """

    def __init__(self, problem, functional):
        self.problem = problem
        self.functional = functional
        self.gauss_newton_approx = False
        self.n_fwd_solve = 0
        self.n_adj_solve = 0
        self.n_inc_solve = 0

    # ------------------------------------------------------------------ vectors
    def generate_vector(self, component="ALL"):
        """A vector (or the triple) in the shape of the problem's variables."""
        return self.problem.generate_vector(component)

    def init_parameter(self, m):
        return self.problem.init_parameter(m)

    # ------------------------------------------------------------------- solves
    def solveFwd(self, out, x):
        self.n_fwd_solve += 1
        return self.problem.solveFwd(out, x)

    def solveAdj(self, out, x):
        r"""Solve the adjoint problem with right-hand side :math:`-\partial_u f`."""
        self.n_adj_solve += 1
        rhs = self.problem.generate_state()
        self.functional.grad(STATE, x, rhs)
        rhs.scale(-1.0)
        return self.problem.solveAdj(out, x, rhs)

    def _reduced_gradient(self, x, mg):
        r"""``mg`` = :math:`\partial_m f + C^{\!\top}p`, the PDE's and the functional's
        parameter derivatives; returns ``mg``."""
        self.problem.evalGradientParameter(x, mg)
        tmp = self.problem.generate_parameter()
        self.functional.grad(PARAMETER, x, tmp)
        mg.axpy(1.0, tmp)
        return mg

    def _set_functional_point(self, x):
        self.functional.setLinearizationPoint(x)

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """Fix the linearization point of the reduced Hessian."""
        self.gauss_newton_approx = bool(gauss_newton_approx)
        self.problem.setLinearizationPoint(x, self.gauss_newton_approx)
        self._set_functional_point(x)
        return self

    def solveFwdIncremental(self, sol, rhs):
        self.n_inc_solve += 1
        return self.problem.solveIncremental(sol, rhs, False)

    def solveAdjIncremental(self, sol, rhs):
        self.n_inc_solve += 1
        return self.problem.solveIncremental(sol, rhs, True)

    # ------------------------------------------------------------------- blocks
    def applyC(self, dm, out):
        return self.problem.apply_ij(ADJOINT, PARAMETER, dm, out)

    def applyCt(self, dp, out):
        return self.problem.apply_ij(PARAMETER, ADJOINT, dp, out)

    def applyWuu(self, du, out):
        """``W_uu`` from the functional plus, unless Gauss-Newton, from the PDE."""
        self.functional.apply_ij(STATE, STATE, du, out)
        if not self.gauss_newton_approx:
            tmp = out.duplicate()
            self.problem.apply_ij(STATE, STATE, du, tmp)
            out.axpy(1.0, tmp)
        return out

    def _second_order(self, i, j, d, out):
        """PDE block plus functional block, or zero under Gauss-Newton."""
        if self.gauss_newton_approx:
            out.zero()
            return out
        self.problem.apply_ij(i, j, d, out)
        tmp = out.duplicate()
        self.functional.apply_ij(i, j, d, tmp)
        out.axpy(1.0, tmp)
        return out

    def applyWum(self, dm, out):
        return self._second_order(STATE, PARAMETER, dm, out)

    def applyWmu(self, du, out):
        return self._second_order(PARAMETER, STATE, du, out)

    def applyWmm(self, dm, out):
        return self._second_order(PARAMETER, PARAMETER, dm, out)

    def apply_ij(self, i, j, d, out):
        """Dispatch to the corresponding ``applyXY``."""
        table = {
            (ADJOINT, PARAMETER): self.applyC,
            (PARAMETER, ADJOINT): self.applyCt,
            (STATE, STATE): self.applyWuu,
            (STATE, PARAMETER): self.applyWum,
            (PARAMETER, STATE): self.applyWmu,
            (PARAMETER, PARAMETER): self.applyWmm,
        }
        if (i, j) not in table:
            raise ValueError("no reduced block (%d, %d)" % (i, j))
        return table[(i, j)](d, out)


class Model(ReducedMap):
    """Full description of a PDE-constrained Bayesian inverse problem.

    Parameters
    ----------
    problem : PDEProblem
    prior : a prior from :mod:`hippymfem.modeling.prior`
    misfit : a misfit from :mod:`hippymfem.modeling.misfit`
    """

    def __init__(self, problem, prior, misfit, gradient_norm="M"):
        super(Model, self).__init__(problem, misfit)
        self.prior = prior
        self.misfit = misfit
        #: Riesz map used to measure the gradient: ``"M"`` (the default) gives the
        #: discrete L2 norm ``sqrt(g^T M^{-1} g)`` that hIPPYlib reports, so
        #: iteration histories, Newton-CG's stopping test and its Eisenstat-Walker
        #: tolerances match hIPPYlib's; ``"R"`` gives the prior-preconditioned
        #: norm ``sqrt(g^T R^{-1} g)``.  Both are mesh-independent.
        self.gradient_norm = gradient_norm

    # --------------------------------------------------------------------- cost
    def cost(self, x):
        """``[total, regularization, misfit]`` at ``x = [u, m, p]``."""
        misfit_cost = self.misfit.cost(x)
        reg_cost = self.prior.cost(x[PARAMETER])
        return [misfit_cost + reg_cost, reg_cost, misfit_cost]

    def evalGradientParameter(self, x, mg, misfit_only=False):
        r"""Gradient of the cost in ``mg``; returns its norm.

        The gradient is a dual-space object, so its norm needs a Riesz map, set by
        :attr:`gradient_norm` (by default the discrete :math:`L^2` norm
        :math:`\sqrt{g^{\!\top}M^{-1}g}` that hIPPYlib reports).
        """
        self._reduced_gradient(x, mg)
        if not misfit_only:
            tmp = self.problem.generate_parameter()
            self.prior.grad(x[PARAMETER], tmp)
            mg.axpy(1.0, tmp)
        g = self.problem.generate_parameter()
        solver = (self.prior.Msolver if self.gradient_norm == "M"
                  else self.prior.Rsolver)
        solver.solve(g, mg)
        return max(g.inner(mg), 0.0) ** 0.5

    def _set_functional_point(self, x):
        self.misfit.setLinearizationPoint(x, self.gauss_newton_approx)

    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """Fix the linearization point of the reduced Hessian."""
        return self.setLinearizationPoint(x, gauss_newton_approx)

    def applyR(self, dm, out):
        return self.prior.R.mult(dm, out)

    def Rsolver(self):
        return self.prior.Rsolver


sync_spellings(ReducedMap)
sync_spellings(Model)

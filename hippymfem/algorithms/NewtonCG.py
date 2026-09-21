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
r"""Inexact Newton-CG in the reduced parameter space.

At each outer iteration the Newton system :math:`H \hat m = -g` is solved
inexactly by preconditioned CG (preconditioner :math:`R`), truncated by an
Eisenstat-Walker tolerance :math:`\min(\tau_0, \sqrt{\|g\|/\|g_0\|})`.  The first
``GN_iter`` iterations use the Gauss-Newton Hessian, which is positive definite
far from the minimum; the full Hessian takes over once the iterate is close
enough for its indefiniteness not to matter.

Globalization is either an Armijo line search or a trust region; both are ported
from hIPPYlib, with the same parameters, the same termination codes and the same
printed table, so runs can be compared iteration by iteration.
"""

import math

from ..common.parameterList import ParameterList
from ..modeling.reducedHessian import ReducedHessian
from ..modeling.variables import ADJOINT, PARAMETER, STATE
from .cgsolverSteihaug import CGSolverSteihaug
from .linesearch import armijo_backtrack


class ModelConvergenceError(RuntimeError):
    """Raised by a forward solve that fails, so the line search can back off."""


def LS_ParameterList():
    return ParameterList({
        "c_armijo": [1e-4, "Armijo constant for sufficient reduction"],
        "max_backtracking_iter": [10, "maximum number of backtracking iterations"],
    })


def TR_ParameterList():
    return ParameterList({
        "eta": [0.05, "reject the step if actual/predicted reduction < eta"],
    })


def ReducedSpaceNewtonCG_ParameterList():
    return ParameterList({
        "rel_tolerance": [1e-6, "converge when ||g||/||g_0|| <= rel_tolerance"],
        "abs_tolerance": [1e-12, "converge when ||g|| <= abs_tolerance"],
        "gdm_tolerance": [1e-18, "converge when (g, dm) <= gdm_tolerance"],
        "max_iter": [20, "maximum number of outer iterations"],
        "globalization": ["LS", "line search (LS) or trust region (TR)"],
        "print_level": [0, "verbosity; -1 silent"],
        "GN_iter": [5, "Gauss-Newton iterations before switching to full Newton"],
        "cg_coarse_tolerance": [0.5, "coarsest CG tolerance (Eisenstat-Walker)"],
        "cg_max_iter": [100, "maximum CG iterations per Newton step"],
        "LS": [LS_ParameterList(), "line search parameters"],
        "TR": [TR_ParameterList(), "trust region parameters"],
    })


class ReducedSpaceNewtonCG:
    """Inexact Newton-CG for the reduced (parameter-space) problem."""

    termination_reasons = [
        "Maximum number of Iteration reached",                        # 0
        "Norm of the gradient less than tolerance",                   # 1
        "Maximum number of backtracking reached",                     # 2
        "Norm of (g, dm) less than tolerance",                        # 3
        "Forward solve failed during backtracking",                   # 4
    ]

    def __init__(self, model, parameters=None, callback=None):
        self.model = model
        self.parameters = (parameters if parameters is not None
                           else ReducedSpaceNewtonCG_ParameterList())
        self.callback = callback
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0.0
        self.final_cost = 0.0
        self.fwd_failed = False
        #: gradient norm at the starting point, so callers can judge the
        #: reduction achieved independently of whether a requested tolerance was
        #: met: with inexact forward solves the attainable floor is set by the
        #: linear solver, not by the optimizer
        self.initial_grad_norm = float("nan")
        #: ||g||/||g_0|| when a line search exhausted its backtracks
        self.grad_reduction = float("nan")

    @property
    def comm(self):
        return self.model.prior.comm

    def _rank0(self):
        return self.comm is None or self.comm.rank == 0

    def solve(self, x):
        """Minimize the cost from the initial guess ``x = [u, m, p]``."""
        if self.model is None:
            raise TypeError("model cannot be None")
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
        g = self.parameters["globalization"]
        if g == "LS":
            return self._solve_ls(x)
        if g == "TR":
            return self._solve_tr(x)
        raise ValueError("unknown globalization %r" % (g,))

    # ------------------------------------------------------------ line search
    def _solve_ls(self, x):
        p = self.parameters
        rel_tol, abs_tol = p["rel_tolerance"], p["abs_tolerance"]
        max_iter, print_level = p["max_iter"], p["print_level"]
        GN_iter = p["GN_iter"]
        cg_coarse_tolerance, cg_max_iter = p["cg_coarse_tolerance"], p["cg_max_iter"]
        c_armijo = p["LS"]["c_armijo"]
        max_backtracking_iter = p["LS"]["max_backtracking_iter"]

        self.model.solveFwd(x[STATE], x)
        self.it = 0
        self.converged = False
        self.ncalls += 1

        mhat = self.model.generate_vector(PARAMETER)
        mg = self.model.generate_vector(PARAMETER)
        x_star = [self.model.generate_vector(STATE),
                  self.model.generate_vector(PARAMETER),
                  None] + list(x[3:])

        cost_old, reg_old, misfit_old = self.model.cost(x)
        cost_new, reg_new, misfit_new = cost_old, reg_old, misfit_old
        gradnorm = gradnorm_ini = float("nan")
        tol = abs_tol

        while self.it < max_iter and not self.converged:
            self.model.solveAdj(x[ADJOINT], x)
            self.model.setPointForHessianEvaluations(
                x, gauss_newton_approx=(self.it < GN_iter))
            gradnorm = self.model.evalGradientParameter(x, mg)

            if self.it == 0:
                gradnorm_ini = gradnorm
                self.initial_grad_norm = gradnorm
                tol = max(abs_tol, gradnorm_ini * rel_tol)
                if gradnorm_ini == 0.0:
                    self.converged = True
                    self.reason = 1
                    self.final_grad_norm = 0.0
                    self.final_cost = cost_old
                    return x

            if gradnorm < tol and self.it > 0:
                self.converged = True
                self.reason = 1
                break

            self.it += 1
            tolcg = (cg_coarse_tolerance if gradnorm_ini == 0.0
                     else min(cg_coarse_tolerance, math.sqrt(gradnorm / gradnorm_ini)))

            HessApply = ReducedHessian(self.model)
            solver = CGSolverSteihaug(comm=self.comm)
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["max_iter"] = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level - 1
            solver.solve(mhat, mg.copy().scale(-1.0))
            self.total_cg_iter += HessApply.ncalls

            mg_mhat = mg.inner(mhat)
            accepted, alpha, n_backtrack, (cost_new, reg_new, misfit_new), self.fwd_failed = \
                armijo_backtrack(self.model, x, x_star, mhat, mg_mhat, cost_old,
                                 c_armijo=c_armijo, max_backtracking=max_backtracking_iter,
                                 gdm_tolerance=p["gdm_tolerance"],
                                 failures=(ModelConvergenceError, RuntimeError))
            if accepted:
                cost_old = cost_new

            if print_level >= 0 and self._rank0():
                if self.it == 1:
                    print("\n%3s %5s %15s %15s %15s %15s %14s %14s %14s"
                          % ("It", "cg_it", "cost", "misfit", "reg", "(g,dm)",
                             "||g||", "alpha", "tolcg"), flush=True)
                print("%3d %5d %15e %15e %15e %15e %14e %14e %14e"
                      % (self.it, HessApply.ncalls, cost_new, misfit_new, reg_new,
                         mg_mhat, gradnorm, alpha, tolcg), flush=True)

            if self.callback:
                self.callback(self.it, x)

            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 4 if self.fwd_failed else 2
                # A line search that cannot improve may mean the iterate is stuck
                # far from a minimum, or that it is *at* one and the requested
                # tolerance is below what finite precision allows.  Record the
                # gradient reduction achieved so the two are distinguishable.
                self.grad_reduction = (gradnorm / gradnorm_ini
                                       if gradnorm_ini else float("nan"))
                if print_level >= 0 and self._rank0():
                    print("  line search exhausted with ||g||/||g_0|| = %.3e "
                          "(tolerance %.3e); the iterate may already be at a "
                          "minimum" % (self.grad_reduction,
                                       tol / max(gradnorm_ini, 1e-300)),
                          flush=True)
                break
            if -mg_mhat <= p["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break

        self.final_grad_norm = gradnorm
        self.final_cost = cost_new
        return x

    # ----------------------------------------------------------- trust region
    def _solve_tr(self, x):
        p = self.parameters
        rel_tol, abs_tol = p["rel_tolerance"], p["abs_tolerance"]
        max_iter, print_level = p["max_iter"], p["print_level"]
        GN_iter = p["GN_iter"]
        cg_coarse_tolerance, cg_max_iter = p["cg_coarse_tolerance"], p["cg_max_iter"]
        eta_TR = p["TR"]["eta"]

        self.model.solveFwd(x[STATE], x)
        self.it = 0
        self.converged = False
        self.ncalls += 1

        mhat = self.model.generate_vector(PARAMETER)
        R_mhat = self.model.generate_vector(PARAMETER)
        mg = self.model.generate_vector(PARAMETER)
        x_star = [self.model.generate_vector(STATE),
                  self.model.generate_vector(PARAMETER), None] + list(x[3:])

        cost_old, reg_old, misfit_old = self.model.cost(x)
        cost_new, reg_new, misfit_new = cost_old, reg_old, misfit_old
        delta_TR = None
        gradnorm = gradnorm_ini = float("nan")
        tol = abs_tol

        while self.it < max_iter and not self.converged:
            self.model.solveAdj(x[ADJOINT], x)
            self.model.setPointForHessianEvaluations(
                x, gauss_newton_approx=(self.it < GN_iter))
            gradnorm = self.model.evalGradientParameter(x, mg)

            if self.it == 0:
                gradnorm_ini = gradnorm
                self.initial_grad_norm = gradnorm
                tol = max(abs_tol, gradnorm_ini * rel_tol)
            if gradnorm < tol and self.it > 0:
                self.converged = True
                self.reason = 1
                break

            self.it += 1
            tolcg = min(cg_coarse_tolerance,
                        math.sqrt(gradnorm / max(gradnorm_ini, 1e-300)))

            HessApply = ReducedHessian(self.model)
            solver = CGSolverSteihaug(comm=self.comm)
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            if delta_TR is not None:
                solver.set_TR(delta_TR, self.model.prior.R)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["max_iter"] = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level - 1
            solver.solve(mhat, mg.copy().scale(-1.0))
            self.total_cg_iter += HessApply.ncalls

            if delta_TR is None:       # first step sets the initial radius
                self.model.applyR(mhat, R_mhat)
                delta_TR = max(math.sqrt(max(R_mhat.inner(mhat), 0.0)) * 5.0, 1.0)

            x_star[PARAMETER].zero()
            x_star[PARAMETER].axpy(1.0, x[PARAMETER])
            x_star[PARAMETER].axpy(1.0, mhat)
            x_star[STATE].zero()
            x_star[STATE].axpy(1.0, x[STATE])
            self.model.solveFwd(x_star[STATE], x_star)
            cost_star, reg_star, misfit_star = self.model.cost(x_star)

            # predicted reduction from the quadratic model
            Hmhat = self.model.generate_vector(PARAMETER)
            HessApply.mult(mhat, Hmhat)
            pred_reduction = -(mg.inner(mhat) + 0.5 * Hmhat.inner(mhat))
            actual_reduction = cost_old - cost_star
            rho_TR = (actual_reduction / pred_reduction
                      if pred_reduction != 0.0 else 0.0)

            self.model.applyR(mhat, R_mhat)
            mhat_Rnorm = math.sqrt(max(R_mhat.inner(mhat), 0.0))
            accept = rho_TR > eta_TR
            if accept:
                x[PARAMETER].zero()
                x[PARAMETER].axpy(1.0, x_star[PARAMETER])
                x[STATE].zero()
                x[STATE].axpy(1.0, x_star[STATE])
                cost_old, reg_new, misfit_new = cost_star, reg_star, misfit_star
                cost_new = cost_star
            if rho_TR < 0.25:
                delta_TR *= 0.5
            elif rho_TR > 0.75 and mhat_Rnorm >= 0.99 * delta_TR:
                delta_TR *= 2.0

            if print_level >= 0 and self._rank0():
                if self.it == 1:
                    print("\n%3s %5s %15s %15s %15s %15s %14s %14s"
                          % ("It", "cg_it", "cost", "misfit", "reg", "||g||",
                             "TR radius", "rho_TR"), flush=True)
                print("%3d %5d %15e %15e %15e %14e %14e %14e%s"
                      % (self.it, HessApply.ncalls, cost_new, misfit_new, reg_new,
                         gradnorm, delta_TR, rho_TR,
                         "" if accept else "  (rejected)"), flush=True)

            if self.callback:
                self.callback(self.it, x)
            if delta_TR < 1e-12:
                self.converged = False
                self.reason = 2
                break

        self.final_grad_norm = gradnorm
        self.final_cost = cost_new
        return x

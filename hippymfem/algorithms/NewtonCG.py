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
printed table.

One default departs from hIPPYlib: ``cg_reorthogonalize``.  The CG iteration of a Newton
step keeps its residuals orthogonal explicitly instead of by its recurrence
(:mod:`~hippymfem.algorithms.cgsolverSteihaug`).  On the model problem of the benchmarks
with 2.1 million state dofs this took 131 CG iterations for twelve Newton steps where
the recurrence took 193 to 210, a number that changed from run to run and from one GPU
to another, and the 131 did not change when the incremental solves of the Hessian
action were stopped at a relative residual of 1e-6 instead of 1e-12; with the
recurrence, incremental solves stopped at 1e-8 already cost a third more CG iterations.
The gradient stays exact (its forward and adjoint solves keep their tolerance), so the
MAP point is the same; only the Newton directions come from a Hessian of lower accuracy.
``cg_reorthogonalize = False`` gives hIPPYlib's iteration, for a comparison step by step.

Two parameters make the solves inside that CG cheaper.  ``cg_preconditioner_tolerance``
(1e-6 by default) stops the prior's solves where they precondition it; ``0`` leaves
them at their own tolerance, as hIPPYlib does.  ``cg_hessian_relaxation`` (off by
default) lets the incremental solves of a Hessian action lose accuracy as the CG
converges.  Both are described, with what they save and why they cannot go further,
in the guide (``docs/source/guide/optimization.rst``).
"""

import math

from ..common.parameterList import ParameterList
from ..fem.kernel import gradient_floor
from ..modeling.reducedHessian import ReducedHessian
from ..modeling.variables import ADJOINT, PARAMETER, STATE
from .cgsolverSteihaug import CGSolverSteihaug
from .linesearch import armijo_backtrack


_UNSET = object()


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
        "cg_reorthogonalize": [True, "the CG of a Newton step keeps its residuals orthogonal "
                                     "explicitly (CGSolverSteihaug, reorthogonalize): fewer "
                                     "Hessian actions, the same count in every run, and "
                                     "incremental solves that need a loose tolerance only; "
                                     "False is hIPPYlib's iteration"],
        "cg_preconditioner_tolerance": [1e-6, "relative tolerance of the prior's solves "
                                               "where they precondition the CG of a Newton "
                                               "step (line search with cg_reorthogonalize), "
                                               "if looser than their own and at most a "
                                               "thousandth of that CG's tolerance; 0: "
                                               "their own"],
        "cg_hessian_relaxation": [0.0, "the incremental solves of a Hessian action stop "
                                        "at this times (the CG's tolerance) times |r_0| / "
                                        "|r_k| at CG iteration k, where that is looser "
                                        "than their own tolerance (line search); 0: "
                                        "their own tolerance throughout"],
        "single_refine_goal": [0.0, "with the solves in a single-precision hypre, the "
                                    "relative residual at which the refinement of the "
                                    "forward and the adjoint solve may stop while this "
                                    "solver runs (PDEVariationalProblem."
                                    "SINGLE_REFINE_GOAL, if the problem leaves it at 0), "
                                    "never more than 1e3 * rel_tolerance**2; 1e-9 is two "
                                    "passes instead of three at a tolerance of 1e-6: "
                                    "about 5 % less time while the iteration keeps its "
                                    "path, and a Newton step more where it does not (one "
                                    "solve in five at 128^3); 0, the default: the "
                                    "problem's own"],
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

    def _cg_preconditioner(self, tolcg):
        """The preconditioner of the CG of a Newton step that stops at ``tolcg``.

        The prior's precision solver, with its Krylov solves stopped at
        ``cg_preconditioner_tolerance`` where that is looser than their own
        (``prior.getHessianPreconditioner``).  A solve stopped at a tolerance is not
        the same linear map at every application, and what the reorthogonalized CG
        then removes from a residual is not all error of the iterate: the residual it
        stops on is off by about that tolerance times the residual it started from.
        The tolerance is therefore never more than a thousandth of ``tolcg``, and the
        solves are left as they are without ``cg_reorthogonalize``."""
        p = self.parameters
        tol = float(p["cg_preconditioner_tolerance"] or 0.0) if p["cg_reorthogonalize"] else 0.0
        tol = min(tol, 1e-3 * float(tolcg))
        if tol > 0.0:
            try:
                return self.model.Rsolver(tol)
            except TypeError:              # a model whose Rsolver takes no argument
                pass
        return self.model.Rsolver()

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
        if g not in ("LS", "TR"):
            raise ValueError("unknown globalization %r" % (g,))
        # On request (single_refine_goal, 0 by default) the refinement of a
        # single-precision forward or adjoint solve stops early while this solver runs
        # (a problem that sets its own goal keeps it): at 1e-9, two passes instead of
        # three.  The decrease the line search must see near the end shrinks with the
        # square of the gradient, so the goal does too: 1e-9 at a tolerance of 1e-6,
        # 1e-13 at 1e-8, where it is tighter than the solvers' own and changes nothing
        # (at 1e-11 a Newton-CG run to 1e-8 ended in a line search that found no
        # decrease).  It is not the default because the last steps can need the digits:
        # at 128^3 one solve in five backtracked in its last line search and took a
        # fourteenth step with the goal at 1e-9, and never with full refinement.
        pde = getattr(self.model, "problem", None)
        goal = min(float(self.parameters["single_refine_goal"] or 0.0),
                   1e3 * float(self.parameters["rel_tolerance"]) ** 2)
        own = None
        if goal > 0.0 and pde is not None and not getattr(pde, "SINGLE_REFINE_GOAL", 1.0):
            own = pde.__dict__.get("SINGLE_REFINE_GOAL", _UNSET)
            pde.SINGLE_REFINE_GOAL = goal
        try:
            return self._solve_ls(x) if g == "LS" else self._solve_tr(x)
        finally:
            if own is _UNSET:
                del pde.SINGLE_REFINE_GOAL
            elif own is not None:
                pde.SINGLE_REFINE_GOAL = own

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
                # (no tighter than single-precision element vectors allow, if in use)
                tol = max(abs_tol, gradnorm_ini * max(rel_tol, gradient_floor()))
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
            solver.set_preconditioner(self._cg_preconditioner(tolcg))
            solver.parameters["relax_operator"] = float(p["cg_hessian_relaxation"] or 0.0)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["max_iter"] = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["reorthogonalize"] = bool(p["cg_reorthogonalize"])
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
                tol = max(abs_tol, gradnorm_ini * max(rel_tol, gradient_floor()))
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
            solver.parameters["reorthogonalize"] = bool(p["cg_reorthogonalize"])
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

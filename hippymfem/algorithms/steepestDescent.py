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
"""Steepest descent in the prior-preconditioned metric.

The descent direction is :math:`-R^{-1}g`, not :math:`-g`: gradients live in the
dual space, and using the Euclidean gradient on a mesh-dependent discretization
gives steps whose size changes with refinement.
"""

from ..common.parameterList import ParameterList
from ..modeling.variables import ADJOINT, PARAMETER, STATE
from .linesearch import armijo_backtrack


def SteepestDescent_ParameterList():
    return ParameterList({
        "rel_tolerance": [1e-6, "converge when ||g||/||g_0|| <= rel_tolerance"],
        "abs_tolerance": [1e-12, "converge when ||g|| <= abs_tolerance"],
        "max_iter": [500, "maximum number of iterations"],
        "c_armijo": [1e-4, "Armijo constant for sufficient reduction"],
        "max_backtracking_iter": [10, "maximum backtracking iterations"],
        "print_level": [0, "verbosity; -1 silent"],
        "alpha": [1.0, "initial step length"],
    })


class SteepestDescent:
    termination_reasons = [
        "Maximum number of Iteration reached",
        "Norm of the gradient less than tolerance",
        "Maximum number of backtracking reached",
    ]

    def __init__(self, model, parameters=None, callback=None):
        self.model = model
        self.parameters = (parameters if parameters is not None
                           else SteepestDescent_ParameterList())
        self.callback = callback
        self.it = 0
        self.converged = False
        self.reason = 0
        self.final_grad_norm = 0.0
        self.final_cost = 0.0

    def solve(self, x):
        p = self.parameters
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
        comm = self.model.prior.comm

        self.model.solveFwd(x[STATE], x)
        cost_old, _, _ = self.model.cost(x)
        cost_new = cost_old
        mg = self.model.generate_vector(PARAMETER)
        mhat = self.model.generate_vector(PARAMETER)
        x_star = [self.model.generate_vector(STATE),
                  self.model.generate_vector(PARAMETER), None]
        alpha = p["alpha"]
        self.it = 0
        self.converged = False
        gradnorm_ini = None
        gradnorm = float("nan")

        while self.it < p["max_iter"] and not self.converged:
            self.model.solveAdj(x[ADJOINT], x)
            gradnorm = self.model.evalGradientParameter(x, mg)
            if gradnorm_ini is None:
                gradnorm_ini = gradnorm
                tol = max(p["abs_tolerance"], gradnorm_ini * p["rel_tolerance"])
            if gradnorm < tol and self.it > 0:
                self.converged = True
                self.reason = 1
                break
            self.it += 1

            self.model.prior.Rsolver.solve(mhat, mg)
            mhat.scale(-1.0)
            mg_mhat = mg.inner(mhat)

            alpha *= 2.0
            accepted, alpha, n_back, (cost_new, _, _), _ = armijo_backtrack(
                self.model, x, x_star, mhat, mg_mhat, cost_old, alpha=alpha,
                c_armijo=p["c_armijo"], max_backtracking=p["max_backtracking_iter"])
            if accepted:
                cost_old = cost_new

            if p["print_level"] >= 0 and comm.rank == 0:
                if self.it == 1:
                    print("\n%3s %15s %15s %14s %14s"
                          % ("It", "cost", "(g,dm)", "||g||", "alpha"), flush=True)
                print("%3d %15e %15e %14e %14e"
                      % (self.it, cost_new, mg_mhat, gradnorm, alpha), flush=True)
            if self.callback:
                self.callback(self.it, x)
            if n_back == p["max_backtracking_iter"]:
                self.converged = False
                self.reason = 2
                break

        self.final_grad_norm = gradnorm
        self.final_cost = cost_new
        return x

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
r"""(Limited-memory) BFGS for the reduced problem.

An alternative to Newton-CG when second-order information is unavailable or too
expensive: it needs only gradients, but typically many more iterations.  The
inverse-Hessian approximation is applied by the two-loop recursion, with Powell
damping so that the curvature condition :math:`s^{\!\top}y > 0` holds even when
the line search returns a poor step.

``H0inv`` is the initial inverse-Hessian guess.  Using the prior solver
:math:`R^{-1}` there, rather than a scaled identity, makes the iteration
mesh-independent, for the same reason it preconditions Newton-CG.
"""

import numpy as np

from ..common.parameterList import ParameterList
from ..modeling.variables import ADJOINT, PARAMETER, STATE
from .NewtonCG import LS_ParameterList
from .linesearch import armijo_backtrack


def BFGSoperator_ParameterList():
    return ParameterList({
        "BFGS_damping": [0.2, "Powell damping parameter"],
        "memory_limit": [np.inf, "number of stored vector pairs (inf = full BFGS)"],
    })


def BFGS_ParameterList():
    ls = LS_ParameterList()
    ls["max_backtracking_iter"] = 25
    return ParameterList({
        "rel_tolerance": [1e-6, "converge when ||g||/||g_0|| <= rel_tolerance"],
        "abs_tolerance": [1e-12, "converge when ||g|| <= abs_tolerance"],
        "gdm_tolerance": [1e-18, "converge when (g, dm) <= gdm_tolerance"],
        "max_iter": [500, "maximum number of iterations"],
        "globalization": ["LS", "line search (LS)"],
        "print_level": [0, "verbosity; -1 silent"],
        "LS": [ls, "line search parameters"],
        "BFGS_op": [BFGSoperator_ParameterList(), "BFGS operator parameters"],
    })


class RescaledIdentity(object):
    r"""Default ``H0inv``: multiplication by a scalar :math:`d_0`.

    The scalar is refreshed from the oldest stored secant pair, which is the
    standard scaling that keeps the first steps from being wildly wrong.
    """

    def __init__(self, init_vector=None):
        self.d0 = 1.0
        self._init_vector = init_vector

    def init_vector(self, x, dim):
        if self._init_vector is None:
            raise RuntimeError("RescaledIdentity has no init_vector")
        return self._init_vector(x, dim)

    def solve(self, x, b):
        x.zero()
        x.axpy(self.d0, b)
        return x


class BFGS_operator:
    """The BFGS inverse-Hessian approximation, applied by the two-loop recursion."""

    def __init__(self, parameters=None):
        self.S, self.Y, self.R = [], [], []
        self.H0inv = None
        self.help = None
        #: the two-loop recursion's work vector; :meth:`update` writes ``H y`` into
        #: :attr:`help`, so the recursion must not work in that one too
        self._work = None
        self.update_scaling = True
        self.parameters = (parameters if parameters is not None
                           else BFGSoperator_ParameterList())

    def set_H0inv(self, H0inv):
        self.H0inv = H0inv
        return self

    def solve(self, x, b):
        r"""``x = H_k b``, the current approximation to :math:`H^{-1}b`."""
        A = []
        if self._work is None:
            self._work = b.copy()
        else:
            self._work.zero()
            self._work.axpy(1.0, b)
        q = self._work

        for s, y, r in zip(reversed(self.S), reversed(self.Y), reversed(self.R)):
            a = r * s.inner(q)
            A.append(a)
            q.axpy(-a, y)

        self.H0inv.solve(x, q)

        for s, y, r, a in zip(self.S, self.Y, self.R, reversed(A)):
            bb = r * y.inner(x)
            x.axpy(a - bb, s)
        return x

    def update(self, s, y):
        r"""Add the secant pair ``(s, y)``, damping it if curvature is too small.

        The damping needs :math:`H y`, written into :attr:`help`.  hIPPYlib's
        version (and this one in 0.1.0) computed it with :meth:`solve` working in
        :attr:`help` as well, so ``H0inv.solve`` got its input as its output: the
        default rescaled identity zeroed it, :math:`y^{\!\top} H y` came out 0,
        and a pair that needed damping was damped to ``s = 0`` and raised.
        """
        damp = self.parameters["BFGS_damping"]
        memlim = self.parameters["memory_limit"]
        if self.help is None:
            self.help = y.copy()
        else:
            self.help.zero()

        sy = s.inner(y)
        self.solve(self.help, y)
        yHy = y.inner(self.help)
        theta = 1.0
        if sy < damp * yHy:
            theta = (1.0 - damp) * yHy / (yHy - sy)
            s.scale(theta)
            s.axpy(1.0 - theta, self.help)
            sy = s.inner(y)
        if sy <= 0.0:
            raise FloatingPointError(
                "BFGS update failed the curvature condition: s^T y = %g" % sy)
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.R.append(1.0 / sy)

        if len(self.S) > memlim:
            self.S.pop(0)
            self.Y.pop(0)
            self.R.pop(0)
            self.update_scaling = True

        if hasattr(self.H0inv, "d0") and self.update_scaling:
            s0, y0 = self.S[0], self.Y[0]
            self.H0inv.d0 = s0.inner(y0) / y0.inner(y0)
            self.update_scaling = False
        return theta


class BFGS:
    """BFGS (or L-BFGS) for the reduced inverse problem."""

    termination_reasons = [
        "Maximum number of Iteration reached",
        "Norm of the gradient less than tolerance",
        "Maximum number of backtracking reached",
        "Norm of (g, dm) less than tolerance",
    ]

    def __init__(self, model, parameters=None):
        self.model = model
        self.parameters = parameters if parameters is not None else BFGS_ParameterList()
        self.BFGSop = BFGS_operator(self.parameters["BFGS_op"])
        self.it = 0
        self.converged = False
        self.reason = 0
        self.ncalls = 0
        self.final_grad_norm = 0.0
        self.final_cost = 0.0

    def solve(self, x, H0inv=None, bounds_xPARAM=None):
        """Minimize the cost from the initial guess ``x``.

        ``H0inv`` defaults to a rescaled identity.  ``bounds_xPARAM``, if given,
        is a ``(lower, upper)`` pair of ParVectors, clipped after each step.
        """
        p = self.parameters
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
        if H0inv is None:
            H0inv = RescaledIdentity(self.model.prior.init_vector)
        self.BFGSop.set_H0inv(H0inv)
        comm = self.model.prior.comm

        self.model.solveFwd(x[STATE], x)
        self.it = 0
        self.converged = False
        self.ncalls += 1

        mhat = self.model.generate_vector(PARAMETER)
        mg = self.model.generate_vector(PARAMETER)
        mg_old = self.model.generate_vector(PARAMETER)
        x_star = [self.model.generate_vector(STATE),
                  self.model.generate_vector(PARAMETER), None]

        cost_old, reg_old, misfit_old = self.model.cost(x)
        cost_new, reg_new, misfit_new = cost_old, reg_old, misfit_old
        gradnorm_ini = None
        gradnorm = float("nan")
        tol = p["abs_tolerance"]
        #: step length accepted by the previous iteration's line search; the
        #: secant pair is (alpha * direction, gradient change), so it has to
        #: carry across iterations
        alpha = 1.0

        while self.it < p["max_iter"] and not self.converged:
            self.model.solveAdj(x[ADJOINT], x)
            gradnorm = self.model.evalGradientParameter(x, mg)
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(p["abs_tolerance"], gradnorm_ini * p["rel_tolerance"])
            else:
                secant_s = mhat.copy().scale(alpha)
                secant_y = mg.copy().axpy(-1.0, mg_old)
                self.BFGSop.update(secant_s, secant_y)
            if gradnorm < tol and self.it > 0:
                self.converged = True
                self.reason = 1
                break
            self.it += 1

            self.BFGSop.solve(mhat, mg)
            mhat.scale(-1.0)
            mg_old.assign(mg)
            mg_mhat = mg.inner(mhat)

            accepted, alpha, n_back, (cost_new, reg_new, misfit_new), _ = armijo_backtrack(
                self.model, x, x_star, mhat, mg_mhat, cost_old,
                c_armijo=p["LS"]["c_armijo"],
                max_backtracking=p["LS"]["max_backtracking_iter"],
                gdm_tolerance=p["gdm_tolerance"], bounds=bounds_xPARAM)
            if accepted:
                cost_old = cost_new

            if p["print_level"] >= 0 and comm.rank == 0:
                if self.it == 1:
                    print("\n%3s %15s %15s %15s %15s %14s %14s"
                          % ("It", "cost", "misfit", "reg", "(g,dm)", "||g||",
                             "alpha"), flush=True)
                print("%3d %15e %15e %15e %15e %14e %14e"
                      % (self.it, cost_new, misfit_new, reg_new, mg_mhat,
                         gradnorm, alpha), flush=True)

            if n_back == p["LS"]["max_backtracking_iter"]:
                self.converged = False
                self.reason = 2
                break
            if -mg_mhat <= p["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break

        self.final_grad_norm = gradnorm
        self.final_cost = cost_new
        return x

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
r"""Preconditioned CG with the Steihaug trust-region stopping rule.

Ported from hIPPYlib, including the termination codes, so that iteration counts
and stopping behaviour are directly comparable between the two libraries, with one
departure: with a trust region, a direction of nonpositive curvature is followed to
the boundary, as Steihaug's method prescribes.  hIPPYlib takes the full first
direction wherever that lands (outside the region when the radius is small) and, at
a later iteration, stops at the current iterate.  Without a trust region the two
agree: the first direction in full, or the current iterate.

Termination reasons:

0. the iteration limit was reached (no convergence);
1. the preconditioned residual met the tolerance (convergence);
2. a direction of nonpositive curvature was found (the operator is not SPD);
3. the trust-region boundary was reached.

The stopping test is on the :math:`B^{-1}`-norm of the residual, relative to its
initial value or against an absolute floor.
"""

import math

from ..common.parameterList import ParameterList


def CGSolverSteihaug_ParameterList():
    return ParameterList({
        "rel_tolerance": [1e-9, "relative tolerance for the stopping criterion"],
        "abs_tolerance": [1e-12, "absolute tolerance for the stopping criterion"],
        "max_iter": [1000, "maximum number of iterations"],
        "zero_initial_guess": [True, "start from 0; otherwise use the incoming x"],
        "print_level": [0, "-1 silent; 0 final residual or reason; 1 every iteration"],
    })


class CGSolverSteihaug:
    r"""Solve :math:`Ax=b` with preconditioner :math:`B` and Steihaug truncation.

    ``A`` must provide ``mult(x, y)`` and ``init_vector(x, dim)``; ``B_solver``
    must provide ``solve(x, b)``.
    """

    reason = [
        "Maximum Number of Iterations Reached",
        "Relative/Absolute residual less than tol",
        "Reached a negative direction",
        "Reached trust region boundary",
    ]

    def __init__(self, parameters=None, comm=None):
        # Built here, not as a default argument: a default is evaluated once, and
        # every solver made without parameters would share one ParameterList.
        self.parameters = (parameters if parameters is not None
                           else CGSolverSteihaug_ParameterList())
        self.comm = comm
        self.A = None
        self.B_solver = None
        self.B_op = None
        self.converged = False
        self.iter = 0
        self.reasonid = 0
        self.final_norm = 0.0
        self.TR_radius_2 = None
        self.update_x = self.update_x_without_TR
        self.r = self.z = self.d = self.Ad = self.Bx = None

    def set_operator(self, A):
        self.A = A
        self.r = A.generate_vector(0)
        self.z = self.r.duplicate()
        self.d = self.r.duplicate()
        self.Ad = self.r.duplicate()
        self.Bx = self.r.duplicate()
        if self.comm is None:
            self.comm = self.r.comm
        return self

    def set_preconditioner(self, B_solver):
        self.B_solver = B_solver
        return self

    def set_TR(self, radius, B_op):
        """Restrict steps to the ball ``x^T B_op x <= radius^2``."""
        if not self.parameters["zero_initial_guess"]:
            raise ValueError("a trust region requires zero_initial_guess = True")
        self.TR_radius_2 = radius * radius
        self.update_x = self.update_x_with_TR
        self.B_op = B_op
        self.Bx = self.r.duplicate()
        return self

    # ---------------------------------------------------------------- stepping
    def update_x_without_TR(self, x, alpha, d):
        x.axpy(alpha, d)
        return False

    def update_x_with_TR(self, x, alpha, d):
        """Step, and if that leaves the trust region, stop on its boundary."""
        x_bk = x.copy()
        x.axpy(alpha, d)
        self.Bx.zero()
        self.B_op.mult(x, self.Bx)
        if self.Bx.inner(x) < self.TR_radius_2:
            return False
        self.Bx.zero()
        self.B_op.mult(x_bk, self.Bx)
        x_Bnorm2 = self.Bx.inner(x_bk)
        Bd = self.d.duplicate()
        self.B_op.mult(self.d, Bd)
        d_Bnorm2 = Bd.inner(d)
        d_Bx = Bd.inner(x_bk)
        a_tau = alpha * alpha * d_Bnorm2
        b_tau_half = alpha * d_Bx
        c_tau = x_Bnorm2 - self.TR_radius_2
        tau = (-b_tau_half
               + math.sqrt(max(b_tau_half * b_tau_half - a_tau * c_tau, 0.0))) / a_tau
        x.zero()
        x.axpy(1.0, x_bk)
        x.axpy(tau * alpha, d)
        return True

    def _tau_to_boundary(self, x, d, rd, dAd):
        r"""The step :math:`\tau` along ``d`` from ``x`` (inside the region) to the
        trust-region boundary, of the two (one each way) the one the quadratic
        model :math:`-\tau\, r^{\!\top} d + \tfrac12 \tau^2 d^{\!\top} A d`
        prefers."""
        Bd = d.duplicate()
        self.B_op.mult(d, Bd)
        a = Bd.inner(d)
        b_half = Bd.inner(x)
        self.Bx.zero()
        self.B_op.mult(x, self.Bx)
        c = self.Bx.inner(x) - self.TR_radius_2
        disc = math.sqrt(max(b_half * b_half - a * c, 0.0))
        best = None
        for tau in ((-b_half + disc) / a, (-b_half - disc) / a):
            m = -tau * rd + 0.5 * tau * tau * dAd
            if best is None or m < best[0]:
                best = (m, tau)
        return best[1]

    def _negative_curvature(self, x, first):
        """Stop on a direction of nonpositive curvature (``self.d``, with ``self.Ad``).

        With a trust region, step to its boundary along the direction; without one,
        take the whole direction at the first iteration and stop where the iterate
        is at a later one (hIPPYlib's rule)."""
        self.converged = True
        self.reasonid = 2
        if self.TR_radius_2 is not None:
            tau = self._tau_to_boundary(x, self.d, self.r.inner(self.d),
                                        self.d.inner(self.Ad))
        elif first:
            tau = 1.0
        else:
            tau = 0.0
        if tau != 0.0:
            x.axpy(tau, self.d)
            self.r.axpy(-tau, self.Ad)
            self.B_solver.solve(self.z, self.r)     # else z is B^-1 r already
        self.final_norm = math.sqrt(max(self.r.inner(self.z), 0.0))
        self._report()

    # ------------------------------------------------------------------- solve
    def _report(self, extra=None):
        if self.parameters["print_level"] >= 0 and (
            self.comm is None or self.comm.rank == 0
        ):
            print(" " + self.reason[self.reasonid], flush=True)
            if self.converged:
                print(" Converged in %d iterations with final norm %.6e"
                      % (self.iter, self.final_norm), flush=True)
            else:
                print(" Not converged. Final residual norm %.6e"
                      % self.final_norm, flush=True)

    def _trace(self, it, nom):
        if self.parameters["print_level"] == 1 and (
            self.comm is None or self.comm.rank == 0
        ):
            print("   CG %3d  (B r, r) = %.6e" % (it, nom), flush=True)

    def solve(self, x, b):
        self.iter = 0
        self.converged = False
        self.reasonid = 0
        betanom = 0.0

        if self.parameters["zero_initial_guess"]:
            self.r.zero()
            self.r.axpy(1.0, b)
            x.zero()
        else:
            if self.TR_radius_2 is not None:
                raise ValueError("a nonzero initial guess is incompatible with a "
                                 "trust region")
            self.A.mult(x, self.r)
            self.r.scale(-1.0)
            self.r.axpy(1.0, b)

        self.z.zero()
        self.B_solver.solve(self.z, self.r)
        self.d.zero()
        self.d.axpy(1.0, self.z)

        nom0 = self.d.inner(self.r)
        nom = nom0
        self._trace(0, nom)

        rtol2 = nom * self.parameters["rel_tolerance"] ** 2
        atol2 = self.parameters["abs_tolerance"] ** 2
        r0 = max(rtol2, atol2)

        if nom <= r0:
            self.converged = True
            self.reasonid = 1
            self.final_norm = math.sqrt(max(nom, 0.0))
            self._report()
            return self.iter

        self.A.mult(self.d, self.Ad)
        den = self.Ad.inner(self.d)

        if den <= 0.0:
            self._negative_curvature(x, first=True)
            return self.iter

        self.iter = 1
        while True:
            alpha = nom / den
            if self.update_x(x, alpha, self.d):
                self.converged = True
                self.reasonid = 3
                self.final_norm = math.sqrt(max(betanom, 0.0))
                self._report()
                break

            self.r.axpy(-alpha, self.Ad)
            self.B_solver.solve(self.z, self.r)
            betanom = self.r.inner(self.z)
            self._trace(self.iter, betanom)

            if betanom < r0:
                self.converged = True
                self.reasonid = 1
                self.final_norm = math.sqrt(max(betanom, 0.0))
                self._report()
                break

            self.iter += 1
            if self.iter > self.parameters["max_iter"]:
                self.converged = False
                self.reasonid = 0
                self.final_norm = math.sqrt(max(betanom, 0.0))
                self._report()
                break

            beta = betanom / nom
            self.d.scale(beta)
            self.d.axpy(1.0, self.z)
            self.A.mult(self.d, self.Ad)
            den = self.d.inner(self.Ad)

            if den <= 0.0:
                self._negative_curvature(x, first=False)
                break

            nom = betanom
        return self.iter

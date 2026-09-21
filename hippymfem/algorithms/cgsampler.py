# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Sampling from a Gaussian using conjugate gradients.

Draws :math:`x \sim N(0, A^{-1})` by accumulating the Lanczos/CG search
directions, which avoids ever forming a square root of :math:`A^{-1}`:
if :math:`d_i` are the CG directions for a random right-hand side, then
:math:`\sum_i (z_i / \sqrt{d_i^{\!\top} A d_i})\, d_i` has covariance
:math:`A^{-1}` restricted to the Krylov subspace explored.

Reference: Parker & Fox, *Sampling Gaussian distributions in Krylov spaces with
conjugate gradients*, SIAM J. Sci. Comput. 34(3), 2012.
"""

import math

import numpy as np

from ..common.parameterList import ParameterList
from ..common.random import parRandom


def CGSampler_ParameterList():
    return ParameterList({
        "rel_tolerance": [1e-9, "relative tolerance for the CG residual"],
        "abs_tolerance": [1e-12, "absolute tolerance for the CG residual"],
        "max_iter": [1000, "maximum number of iterations"],
        "print_level": [-1, "verbosity; -1 silent"],
    })


class CGSampler:
    r"""Sample :math:`N(0, A^{-1})` with conjugate gradients."""

    def __init__(self, parameters=None):
        self.parameters = (parameters if parameters is not None
                           else CGSampler_ParameterList())
        self.A = None
        self.converged = False
        self.iter = 0
        self.final_norm = 0.0
        self.b = self.r = self.p = self.Ap = None

    def set_operator(self, A):
        self.A = A
        self.b = A.generate_vector(0)
        self.r = self.b.duplicate()
        self.p = self.b.duplicate()
        self.Ap = self.b.duplicate()
        return self

    def sample(self, noise, s):
        r"""``s`` receives a sample; ``noise`` supplies the standard normals.

        ``noise`` must have at least as many entries as CG takes iterations; an
        exhausted supply raises rather than reusing numbers.  A numpy array must be the
        same on every rank.  A ParVector contributes its global entries, in global
        order, so every rank uses the same variate at each iteration.
        """
        # a ParVector (what every other ``sample`` takes) or a numpy array; anything
        # else means "draw from parRandom"
        if isinstance(noise, np.ndarray):
            noise_arr = np.asarray(noise, dtype=float).reshape(-1)
        elif hasattr(noise, "array"):
            # Each CG iteration scales the whole distributed vector by one variate, so
            # every rank must read the same one: the local slices, gathered in rank
            # order, are the global entries.  Reading only the local slice would give
            # each rank different coefficients, and the rank that ran out first would
            # raise while the others wait in the next collective.
            comm = getattr(noise, "comm", None)
            local = np.asarray(noise.array, dtype=float).reshape(-1)
            noise_arr = (np.concatenate(comm.allgather(local))
                         if comm is not None and comm.size > 1 else local)
        else:
            noise_arr = None
        s.zero()
        self.iter = 0
        self.converged = False

        parRandom.normal(1.0, self.b)
        self.r.assign(self.b)
        self.p.assign(self.r)
        d = self.r.inner(self.r)
        tol = max(self.parameters["abs_tolerance"] ** 2,
                  d * self.parameters["rel_tolerance"] ** 2)
        nom0 = d

        k = 0
        while k < self.parameters["max_iter"]:
            self.A.mult(self.p, self.Ap)
            gamma = self.Ap.inner(self.p)
            if gamma <= 0.0:
                break
            if noise_arr is not None:
                if k >= noise_arr.size:
                    raise ValueError(
                        "CGSampler needs at least %d normal variates, got %d"
                        % (k + 1, noise_arr.size))
                z = float(noise_arr[k])
            else:
                z = parRandom.scalar_normal(1.0)
            s.axpy(z / math.sqrt(gamma), self.p)

            alpha = d / gamma
            self.r.axpy(-alpha, self.Ap)
            dnew = self.r.inner(self.r)
            k += 1
            self.iter = k
            if dnew < tol:
                self.converged = True
                break
            beta = dnew / d
            self.p.scale(beta)
            self.p.axpy(1.0, self.r)
            d = dnew
        self.final_norm = math.sqrt(max(d, 0.0))
        if self.parameters["print_level"] >= 0 and s.comm.rank == 0:
            print(" CGSampler: %d iterations, residual %.3e (from %.3e)"
                  % (self.iter, self.final_norm, math.sqrt(nom0)), flush=True)
        return s

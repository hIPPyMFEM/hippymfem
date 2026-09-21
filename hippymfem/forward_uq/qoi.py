# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Quantities of interest: scalar functionals of the state and parameter.

A QoI plays the role the misfit plays in the inverse problem: it supplies a
value, its first derivatives, and the action of its second derivatives.  The
forward-UQ machinery then treats :math:`m \mapsto q(u(m), m)` as a
parameter-to-QoI map and propagates uncertainty through it.
"""

import mfem.par as mfem

from ..common.keepalive import KeepAlive
from ..common.linalg import as_matrix
from ..fem.assemble import assemble_native_matrix, mass_functional  # noqa: F401
from ..common.naming import SnakeCamel, sync_spellings
from ..fem.coefficients import attribute_indicator
from ..fem.spaces import as_space
from ..modeling.variables import STATE


class Qoi(SnakeCamel):
    """Abstract scalar functional of ``x = [u, m, p]``."""

    def eval(self, x):
        raise NotImplementedError

    def grad(self, i, x, g):
        raise NotImplementedError

    def setLinearizationPoint(self, x):
        raise NotImplementedError

    def apply_ij(self, i, j, dir, out):
        raise NotImplementedError


class NullQoi(Qoi):
    """Always zero; useful as a placeholder."""

    def eval(self, x):
        return 0.0

    def grad(self, i, x, g):
        g.zero()
        return g

    def setLinearizationPoint(self, x):
        return self

    def apply_ij(self, i, j, dir, out):
        out.zero()
        return out


class LinearStateQoi(Qoi):
    r""":math:`q = \ell^{\!\top} u` for a fixed dual vector :math:`\ell`.

    Covers pointwise and averaged observations of the state: with
    :math:`\ell = B^{\!\top}e_k` it is the value at target ``k``, and with
    :math:`\ell = M\mathbf{1}/|\Omega|` the spatial average.
    """

    def __init__(self, ell):
        self.ell = ell

    def eval(self, x):
        return self.ell.inner(x[STATE])

    def grad(self, i, x, g):
        if i == STATE:
            g.assign(self.ell)
        else:
            g.zero()
        return g

    def setLinearizationPoint(self, x):
        return self

    def apply_ij(self, i, j, dir, out):
        out.zero()            # linear in u, so no second derivatives
        return out


class QuadraticStateQoi(Qoi, KeepAlive):
    r""":math:`q = \tfrac12 u^{\!\top} W u` for a symmetric ``W``.

    With ``W`` the mass matrix restricted to a subdomain this is half the
    squared :math:`L^2` norm of the state there.
    """

    def __init__(self, W, comm=None):
        self.W = as_matrix(W)
        self.keep(W)
        self._Wu = None

    def _tmp(self, like):
        if self._Wu is None:
            self._Wu = like.duplicate()
        return self._Wu

    def eval(self, x):
        Wu = self._tmp(x[STATE])
        self.W.Mult(x[STATE].hypre, Wu.hypre)
        return 0.5 * Wu.inner(x[STATE])

    def grad(self, i, x, g):
        if i == STATE:
            self.W.Mult(x[STATE].hypre, g.hypre)
        else:
            g.zero()
        return g

    def setLinearizationPoint(self, x):
        return self

    def apply_ij(self, i, j, dir, out):
        if i == STATE and j == STATE:
            self.W.Mult(dir.hypre, out.hypre)
        else:
            out.zero()
        return out


def l2_norm_qoi(Vh, attributes=None, comm=None):
    r"""``q = 1/2 ||u||^2`` over the domain or over given element attributes."""
    Vh = as_space(Vh)
    keep = []
    coeff = None
    if attributes is not None:
        coeff, held = attribute_indicator(Vh.mesh, attributes)
        keep += held + [coeff]
    integ = (mfem.VectorMassIntegrator if Vh.vdim > 1 else mfem.MassIntegrator)
    it = integ() if coeff is None else integ(coeff)
    W = assemble_native_matrix(Vh, [it])
    q = QuadraticStateQoi(W)
    q.keep(it, *keep)
    return q


def weighted_mean_qoi(Vh, w=None):
    r"""``q(u)`` = the ``w``-weighted mean of the state, ``(w^T M u) / (w^T M 1)``.

    ``w`` is a field on ``Vh`` (an indicator of a target region, say); ``None`` is
    the spatial mean.  The functional is built by :func:`mass_functional`, so no
    matrix is formed; the normalizing volume is kept as ``q.volume``.
    """
    Vh = as_space(Vh)
    ell = mass_functional(Vh, w)
    one = Vh.vector()
    one.set(1.0)
    vol = ell.inner(one)
    ell.scale(1.0 / max(vol, 1e-300))
    q = LinearStateQoi(ell)
    q.volume = vol
    return q


def mean_state_qoi(Vh):
    r"""``q`` = the spatial average of the state, ``(1^T M u) / (1^T M 1)``."""
    return weighted_mean_qoi(Vh)


sync_spellings(Qoi)

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
r"""Misfit (data likelihood) functionals.

A misfit exposes ``cost``, ``grad``, ``setLinearizationPoint`` and ``apply_ij``
with hIPPYlib's signatures, so the model and the algorithms do not care which one
is in use.
"""

import numpy as np

import mfem.par as mfem

from ..common.keepalive import KeepAlive
from ..fem.assemble import assemble_native_matrix
from ..fem.bcs import as_bcset
from ..common.naming import SnakeCamel, sync_spellings
from ..fem.coefficients import attribute_indicator
from ..fem.spaces import as_space
from .pointwiseObservation import assemblePointwiseObservation
from .variables import PARAMETER, STATE


class Misfit(SnakeCamel):
    """Abstract misfit term of the cost functional."""

    def cost(self, x):
        raise NotImplementedError

    def grad(self, i, x, out):
        raise NotImplementedError

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        raise NotImplementedError

    def apply_ij(self, i, j, dir, out):
        raise NotImplementedError


def _check_noise_variance(nv):
    if nv is None:
        raise ValueError("noise_variance must be specified")
    if nv == 0:
        raise ZeroDivisionError(
            "noise_variance must not be 0; use 1.0 for a deterministic inverse problem"
        )
    return nv


class DiscreteStateObservation(Misfit, KeepAlive):
    r"""Gaussian misfit for a linear observation operator:
    :math:`\tfrac{1}{2\sigma^2}\|Bu - d\|^2`.
    """

    def __init__(self, B, data=None, noise_variance=None):
        self.B = B
        self.d = data if data is not None else B.createVecLeft()
        self.Bu = B.createVecLeft()
        self.noise_variance = noise_variance
        self.keep(B)

    def cost(self, x):
        nv = _check_noise_variance(self.noise_variance)
        self.B.mult(x[STATE], self.Bu)
        self.Bu.axpy(-1.0, self.d)
        return (0.5 / nv) * self.Bu.inner(self.Bu)

    def grad(self, i, x, out):
        nv = _check_noise_variance(self.noise_variance)
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            self.Bu.axpy(-1.0, self.d)
            self.B.multTranspose(self.Bu, out)
            out.scale(1.0 / nv)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError("grad index %r" % (i,))
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        return self          # already quadratic

    def apply_ij(self, i, j, dir, out):
        nv = _check_noise_variance(self.noise_variance)
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            self.B.multTranspose(self.Bu, out)
            out.scale(1.0 / nv)
        else:
            out.zero()
        return out


def PointwiseStateObservation(Vh, obs_points, data=None, noise_variance=None):
    """Misfit for pointwise observations of the state at ``obs_points``."""
    B = assemblePointwiseObservation(Vh, obs_points)
    return DiscreteStateObservation(B, data, noise_variance)


class MultDiscreteStateObservation(Misfit, KeepAlive):
    r"""Multiplicative Gamma(M, M) noise model.

    The negative log-likelihood is
    :math:`M \sum_t \left(\log (Bu)_t + d_t / (Bu)_t\right)`, so the state must
    stay strictly positive; a nonpositive observed value is reported rather than
    producing a silent NaN.
    """

    def __init__(self, B, data=None, Mpar=1.0):
        self.B = B
        self.d = data if data is not None else B.createVecLeft()
        self.Bu = B.createVecLeft()
        self.Bu_lin = B.createVecLeft()
        self.help = B.createVecLeft()
        self.Mpar = float(Mpar)
        self.keep(B)

    def _positive(self, a, where):
        if a.size and np.min(a) <= 0.0:
            raise FloatingPointError(
                "multiplicative noise model needs Bu > 0; got min %g in %s"
                % (float(np.min(a)), where)
            )

    def cost(self, x):
        self.B.mult(x[STATE], self.Bu)
        bu = self.Bu.array
        self._positive(bu, "cost")
        self.help.array[:] = np.log(bu) + self.d.array / bu
        ones = self.B.createVecLeft()
        ones.set(1.0)
        return self.Mpar * self.help.inner(ones)

    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            self.B.mult(x[STATE], self.Bu)
            bu = self.Bu.array
            self._positive(bu, "grad")
            self.help.array[:] = 1.0 / bu - self.d.array / (bu * bu)
            self.B.multTranspose(self.help, out)
            out.scale(self.Mpar)
        elif i != PARAMETER:
            raise IndexError("grad index %r" % (i,))
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        self.B.mult(x[STATE], self.Bu_lin)
        return self

    def apply_ij(self, i, j, dir, out):
        out.zero()
        if i == STATE and j == STATE:
            self.B.mult(dir, self.Bu)
            bd = self.Bu.array
            bl = self.Bu_lin.array
            self._positive(bl, "apply_ij")
            self.help.array[:] = (-bd * bl ** -2
                                  + 2.0 * self.d.array * bd * bl ** -3)
            self.B.multTranspose(self.help, out)
            out.scale(self.Mpar)
        return out


def MultPointwiseStateObservation(Vh, obs_points, Mpar, data=None):
    """Multiplicative-noise misfit for pointwise observations."""
    B = assemblePointwiseObservation(Vh, obs_points)
    return MultDiscreteStateObservation(B, data, Mpar)


class ContinuousStateObservation(Misfit, KeepAlive):
    r"""Misfit in a weighted :math:`L^2` norm over the domain or a subdomain:
    :math:`\tfrac{1}{2\sigma^2}(u-d)^{\!\top} W (u-d)`.

    Parameters
    ----------
    Vh : FunctionSpace
        State space.
    attributes : sequence of int, optional
        Mesh element attributes defining the observed subdomain; the default is
        the whole domain.
    bcs : optional
        Essential conditions whose rows and columns are removed from ``W``.
    boundary : bool
        Observe on the boundary (``ds``) rather than the interior (``dx``).
    """

    def __init__(self, Vh, attributes=None, bcs=None, data=None,
                 noise_variance=None, boundary=False, coefficient=None):
        self.Vh = as_space(Vh)
        self.bcs = as_bcset(bcs, self.Vh)
        coeff = coefficient
        keep = []
        if attributes is not None:
            pw, held = attribute_indicator(self.Vh.mesh, attributes)
            keep += held + [pw]
            coeff = pw if coeff is None else mfem.ProductCoefficient(coeff, pw)

        # An H(curl)/H(div) space has vdim == 1 but a vector-valued basis, so the
        # weighting operator is the vector FE mass matrix, not the scalar one.
        from ..fem.vectorfe import is_vector_space

        if is_vector_space(self.Vh):
            integ = mfem.VectorFEMassIntegrator
        elif self.Vh.vdim > 1:
            integ = mfem.VectorMassIntegrator
        else:
            integ = mfem.MassIntegrator
        it = integ() if coeff is None else integ(coeff)
        if boundary:
            self.W = assemble_native_matrix(self.Vh, [], [it],
                                            ess=self.bcs.ess_tdof,
                                            diag_policy="zero")
        else:
            self.W = assemble_native_matrix(self.Vh, [it],
                                            ess=self.bcs.ess_tdof,
                                            diag_policy="zero")
        self.keep(self.W, it, *keep)
        self.d = data if data is not None else self.Vh.vector()
        self.noise_variance = noise_variance
        self._r = self.Vh.vector()
        self._Wr = self.Vh.vector()

    def createVecLeft(self):
        return self.Vh.vector()

    def cost(self, x):
        nv = _check_noise_variance(self.noise_variance)
        self._r.assign(x[STATE]).axpy(-1.0, self.d)
        self.W.Mult(self._r.hypre, self._Wr.hypre)
        return (0.5 / nv) * self._Wr.inner(self._r)

    def grad(self, i, x, out):
        nv = _check_noise_variance(self.noise_variance)
        if i == STATE:
            self._r.assign(x[STATE]).axpy(-1.0, self.d)
            self.W.Mult(self._r.hypre, out.hypre)
            out.scale(1.0 / nv)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError("grad index %r" % (i,))
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        return self

    def apply_ij(self, i, j, dir, out):
        nv = _check_noise_variance(self.noise_variance)
        if i == STATE and j == STATE:
            self.W.Mult(dir.hypre, out.hypre)
            out.scale(1.0 / nv)
        else:
            out.zero()
        return out


class MultiStateMisfit(Misfit):
    """Sum of misfits over several independent states (experiments)."""

    def __init__(self, misfits=None):
        self.misfits = list(misfits) if misfits else []

    def append(self, misfit):
        self.misfits.append(misfit)
        return self

    def cost(self, x):
        return float(sum(m.cost([x[STATE][i], x[PARAMETER], None])
                         for i, m in enumerate(self.misfits)))

    def grad(self, i, x, out):
        if i == STATE:
            for k, m in enumerate(self.misfits):
                m.grad(i, [x[STATE][k], x[PARAMETER], None], out[k])
        else:
            out.zero()
            tmp = out.duplicate()
            for k, m in enumerate(self.misfits):
                m.grad(i, [x[STATE][k], x[PARAMETER], None], tmp)
                out.axpy(1.0, tmp)
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        for k, m in enumerate(self.misfits):
            m.setLinearizationPoint([x[STATE][k], x[PARAMETER], None],
                                    gauss_newton_approx)
        return self

    def apply_ij(self, i, j, dir, out):
        if i == STATE and j == STATE:
            for k, m in enumerate(self.misfits):
                m.apply_ij(i, j, dir[k], out[k])
        elif i == STATE:
            for k, m in enumerate(self.misfits):
                m.apply_ij(i, j, dir, out[k])
        elif j == STATE:
            out.zero()
            tmp = out.duplicate()
            for k, m in enumerate(self.misfits):
                m.apply_ij(i, j, dir[k], tmp)
                out.axpy(1.0, tmp)
        else:
            out.zero()
            tmp = out.duplicate()
            for m in self.misfits:
                m.apply_ij(i, j, dir, tmp)
                out.axpy(1.0, tmp)
        return out


class MisfitTD(Misfit):
    """Time-dependent misfit: a list of stationary misfits, one per observation time."""

    def __init__(self, misfits, sim_times):
        self.misfits = list(misfits)
        self.sim_times = np.asarray(sim_times, dtype=float)
        if len(self.misfits) != self.sim_times.size:
            raise ValueError("one misfit per simulation time is required")

    def cost(self, x):
        total = 0.0
        for t, m in zip(self.sim_times, self.misfits):
            if m is None:
                continue
            total += m.cost([x[STATE].view(t), x[PARAMETER], None])
        return total

    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            for t, m in zip(self.sim_times, self.misfits):
                if m is None:
                    continue
                m.grad(i, [x[STATE].view(t), x[PARAMETER], None], out.view(t))
        else:
            tmp = out.duplicate()
            for t, m in zip(self.sim_times, self.misfits):
                if m is None:
                    continue
                m.grad(i, [x[STATE].view(t), x[PARAMETER], None], tmp)
                out.axpy(1.0, tmp)
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        for t, m in zip(self.sim_times, self.misfits):
            if m is None:
                continue
            m.setLinearizationPoint([x[STATE].view(t), x[PARAMETER], None],
                                    gauss_newton_approx)
        return self

    def apply_ij(self, i, j, dir, out):
        out.zero()
        if i == STATE and j == STATE:
            for t, m in zip(self.sim_times, self.misfits):
                if m is None:
                    continue
                m.apply_ij(i, j, dir.view(t), out.view(t))
        elif i == STATE:
            for t, m in zip(self.sim_times, self.misfits):
                if m is None:
                    continue
                m.apply_ij(i, j, dir, out.view(t))
        elif j == STATE:
            tmp = out.duplicate()
            for t, m in zip(self.sim_times, self.misfits):
                if m is None:
                    continue
                m.apply_ij(i, j, dir.view(t), tmp)
                out.axpy(1.0, tmp)
        return out


sync_spellings(Misfit)

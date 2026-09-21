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
r"""Low-rank Gaussian (Laplace) approximation of the posterior.

If :math:`(d_i, u_i)` are the dominant generalized eigenpairs of

.. math:: H_{\rm misfit}\, u_i = d_i\, R\, u_i, \qquad U^{\!\top} R U = I,

then the Laplace approximation at the MAP point has

* precision apply: :math:`(R + R U D U^{\!\top} R)\,x`;
* covariance apply: :math:`\left(R^{-1} - U (I + D^{-1})^{-1} U^{\!\top}\right) x`;
* sampling: :math:`y = (I - U S U^{\!\top} R) x` with
  :math:`S = I - (I+D)^{-1/2}` and :math:`x \sim N(0, R^{-1})`.

Once the eigenpairs are in hand, all three need only prior operations and ``k``
inner products, with no further PDE solves.
"""

import numpy as np

from ..common.operators import as_operator
from ..common.naming import sync_spellings
from ..algorithms.lowRankOperator import LowRankOperator
from ..common.keepalive import KeepAlive
from ..common.operators import Operator, make_vector


class LowRankHessian(Operator):
    r"""Low-rank posterior precision and its inverse."""

    def __init__(self, prior, d, U):
        self.prior = prior
        self.d = np.asarray(d, dtype=float)
        self.U = U
        self.LowRankH = LowRankOperator(self.d, U, prior.init_vector)
        dsolve = self.d / (1.0 + self.d)
        self.LowRankHinv = LowRankOperator(dsolve, U, prior.init_vector)
        self.help = make_vector(prior, 0)
        self.help1 = make_vector(prior, 0)

    def init_vector(self, x, dim):
        return self.prior.init_vector(x, dim)

    def inner(self, x, y):
        Hx = self.help.duplicate()
        self.mult(x, Hx)
        return Hx.inner(y)

    def mult(self, x, y):
        r""":math:`y = (R + R U D U^{\!\top} R) x`."""
        self.prior.R.mult(x, y)
        self.LowRankH.mult(y, self.help)
        self.prior.R.mult(self.help, self.help1)
        y.axpy(1.0, self.help1)
        return y

    multTranspose = mult

    def solve(self, sol, rhs):
        r""":math:`\mathrm{sol} = (R^{-1} - U (I + D^{-1})^{-1} U^{\!\top})\,\mathrm{rhs}`."""
        self.prior.Rsolver.solve(sol, rhs)
        self.LowRankHinv.mult(rhs, self.help)
        sol.axpy(-1.0, self.help)
        return sol


class LowRankPosteriorSampler(KeepAlive):
    r"""Turn a prior sample into a posterior sample: :math:`y = (I - U S U^{\!\top} R)x`."""

    def __init__(self, prior, d, U):
        self.prior = prior
        d = np.asarray(d, dtype=float)
        self.d = 1.0 - np.power(1.0 + d, -0.5)
        self.lrsqrt = LowRankOperator(self.d, U, prior.init_vector)
        self.help = make_vector(prior, 0)

    def init_vector(self, x, dim):
        return self.prior.init_vector(x, dim)

    def sample(self, noise, s):
        """``noise`` is a prior sample (zero mean); ``s`` becomes a posterior sample."""
        self.prior.R.mult(noise, self.help)
        self.lrsqrt.mult(self.help, s)
        s.axpy(-1.0, noise)
        s.scale(-1.0)
        return s


class GaussianLRPosterior(KeepAlive):
    """Laplace approximation of the posterior at the MAP point.

    Parameters
    ----------
    prior : a prior object
    d, U : the dominant generalized eigenpairs of the data-misfit Hessian
    mean : ParVector, optional
        The MAP point.
    """

    def __init__(self, prior, d, U, mean=None):
        self.prior = prior
        self.d = np.asarray(d, dtype=float)
        self.U = U
        self.Hlr = LowRankHessian(prior, self.d, U)
        self.sampler = LowRankPosteriorSampler(prior, self.d, U)
        self.mean = mean

    def init_vector(self, x, dim):
        return self.prior.init_vector(x, dim)

    def cost(self, m):
        if self.mean is None:
            return 0.5 * self.Hlr.inner(m, m)
        dm = m.copy().axpy(-1.0, self.mean)
        return 0.5 * self.Hlr.inner(dm, dm)

    def sample(self, *args, noise=None, s_prior=None, s_post=None, add_mean=True,
               rng=None):
        """Draw a posterior sample; returns ``s_post``.

        hIPPYlib's two positional forms work: ``sample(noise, s_prior, s_post)``
        with white noise, filling the prior and the posterior sample, and
        ``sample(s_prior, s_post)`` from an existing zero-mean prior sample.  The
        keyword form makes the roles explicit, and every vector may be left out:
        ``post.sample()`` draws the noise (from ``rng`` or ``parRandom``) and returns
        a new posterior sample, ``post.sample(s_prior=s)`` transforms a given prior
        sample.
        """
        if len(args) == 3:
            noise, s_prior, s_post = args
        elif len(args) == 2:
            s_prior, s_post = args
        elif args:
            raise TypeError("sample() takes 2 or 3 positional arguments, or keywords")
        if s_post is None:
            s_post = self.mean.duplicate() if self.mean is not None else self.prior.mean.duplicate()
        if s_prior is None and noise is None:
            noise = self.prior.sample_noise(1.0, rng=rng)
        if noise is not None:
            if s_prior is None:
                s_prior = self.prior.mean.duplicate()
            self._sample_given_white_noise(noise, s_prior, s_post)
        else:
            self._sample_given_prior(s_prior, s_post)
        if add_mean and self.mean is not None:
            s_prior.axpy(1.0, self.prior.mean)
            s_post.axpy(1.0, self.mean)
        return s_post

    def _sample_given_white_noise(self, noise, s_prior, s_post):
        self.prior.sample(noise, s_prior, add_mean=False)
        self.sampler.sample(s_prior, s_post)

    def _sample_given_prior(self, s_prior, s_post):
        self.sampler.sample(s_prior, s_post)

    # -------------------------------------------------------------- statistics
    def trace(self, **kwargs):
        """``(posterior, prior, correction)`` traces of the covariance."""
        pr_trace = self.prior.trace(**kwargs)
        corr_trace = self.trace_update()
        return pr_trace - corr_trace, pr_trace, corr_trace

    def trace_update(self):
        r"""Trace of the data-informed correction, :math:`\mathrm{tr}(U(I+D^{-1})^{-1}U^{\!\top}M)`."""
        return self.Hlr.LowRankHinv.trace(as_operator(self.prior.M, self.prior.comm))

    def pointwise_variance(self, **kwargs):
        """``(posterior, prior, correction)`` pointwise variance fields."""
        pr = self.prior.pointwise_variance(**kwargs)
        corr = pr.duplicate()
        self.Hlr.LowRankHinv.get_diagonal(corr)
        post = pr.copy().axpy(-1.0, corr)
        return post, pr, corr

    def klDistanceFromPrior(self, sub_comp=False):
        r"""Kullback-Leibler divergence of the Laplace posterior from the prior."""
        dplus1 = self.d + 1.0
        c_logdet = 0.5 * float(np.sum(np.log(dplus1)))
        c_trace = -0.5 * float(np.sum(self.d / dplus1))
        c_shift = self.prior.cost(self.mean) if self.mean is not None else 0.0
        kld = c_logdet + c_trace + c_shift
        if sub_comp:
            return kld, c_logdet, c_trace, c_shift
        return kld


sync_spellings(GaussianLRPosterior)

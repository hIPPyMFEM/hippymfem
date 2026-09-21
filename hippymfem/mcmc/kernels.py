# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""MCMC transition kernels for the posterior over the parameter field.

All of these are dimension-robust: their acceptance rates do not degrade as the
mesh is refined, because the proposals are built from the prior rather than from
a fixed step in the discrete parameter space.

=========  ===============================================================
``pCN``    preconditioned Crank-Nicolson: propose
           :math:`m' = \bar m + s\,\xi + \sqrt{1-s^2}(m-\bar m)` with
           :math:`\xi \sim N(0, R^{-1})`, which leaves the prior invariant,
           so the acceptance ratio involves only the misfit.
``gpCN``   the same, but with the *Laplace approximation* in place of the
           prior, so the proposal already knows where the posterior mass is;
           far more efficient when the data is informative.
``MALA``   Metropolis-adjusted Langevin: a gradient-informed proposal.
``IS``     importance sampling from the Laplace approximation (independent
           proposals, weights returned rather than accepted/rejected).
=========  ===============================================================

References
----------
* S.L. Cotter, G.O. Roberts, A.M. Stuart, D. White, *MCMC methods for functions*,
  Statistical Science 28(3), 2013 (pCN).
* F.J. Pinski, G. Simpson, A.M. Stuart, H. Weber, *Algorithms for
  Kullback-Leibler approximation of probability measures in infinite
  dimensions*, SIAM J. Sci. Comput. 37(6), 2015, Alg. 5.2 (gpCN).
"""

import math

from ..common.naming import sync_spellings
from ..common.operators import make_vector
from ..common.parameterList import ParameterList
from ..common.random import parRandom
from ..modeling.variables import PARAMETER


class _KernelBase:
    """Shared plumbing: parameters, the noise vector, and the uniform draw."""

    def __init__(self, model):
        self.model = model
        self.parameters = ParameterList({"s": [0.1, "step size of the proposal"]})
        self.rng = parRandom

    def name(self):
        raise NotImplementedError

    def derivativeInfo(self):
        """Highest derivative the kernel needs (0 = cost only, 1 = gradient)."""
        return 0

    def init_sample(self, sample):
        """Solve the forward problem and record the misfit at ``sample.m``."""
        self.model.solveFwd(sample.u, [sample.u, sample.m, None])
        sample.cost = self.model.cost([sample.u, sample.m, None])[2]
        return sample

    def _log_uniform(self):
        """log U(0,1), identical on every rank so the chain does not diverge."""
        return math.log(max(self.rng.scalar_uniform(0.0, 1.0), 1e-300))

    def consume_random(self):
        """Advance the stream as one step would, without taking the step."""
        raise NotImplementedError


class pCNKernel(_KernelBase):
    r"""Preconditioned Crank-Nicolson with the prior as reference measure."""

    def __init__(self, model):
        super(pCNKernel, self).__init__(model)
        self.noise = make_vector(model.prior, "noise")
        self._w = model.generate_vector(PARAMETER)
        self._tmp = model.generate_vector(PARAMETER)

    def name(self):
        return "pCN"

    def proposal(self, current):
        prior = self.model.prior
        prior.sample_noise(1.0, self.noise, self.rng)
        prior.sample(self.noise, self._w, add_mean=False)
        s = self.parameters["s"]
        self._w.scale(s)
        self._w.axpy(1.0, prior.mean)
        self._tmp.assign(current.m).axpy(-1.0, prior.mean)
        self._w.axpy(math.sqrt(max(1.0 - s * s, 0.0)), self._tmp)
        return self._w

    def sample(self, current, proposed):
        """One Metropolis-Hastings step; returns 1 if accepted, 0 otherwise."""
        proposed.m.assign(self.proposal(current))
        self.init_sample(proposed)
        # the pCN proposal preserves the prior, so only the misfit enters
        log_alpha = current.cost - proposed.cost
        if log_alpha > self._log_uniform():
            current.assign(proposed)
            return 1
        return 0

    def consume_random(self):
        self.model.prior.sample_noise(1.0, self.noise, self.rng)
        self.rng.scalar_uniform()


class gpCNKernel(_KernelBase):
    r"""pCN with the Laplace approximation ``nu`` as reference measure."""

    def __init__(self, model, nu):
        super(gpCNKernel, self).__init__(model)
        self.nu = nu
        self.prior = model.prior
        self.noise = make_vector(nu, "noise")
        self._w = model.generate_vector(PARAMETER)
        self._w_prior = model.generate_vector(PARAMETER)
        self._tmp = model.generate_vector(PARAMETER)

    def name(self):
        return "gpCN"

    def delta(self, sample):
        r"""The log-density difference that the gpCN ratio needs.

        :math:`\Phi(m) + \tfrac12\|m\|_R^2 - \tfrac12\|m-\bar m_\nu\|^2_{H_\nu}`:
        the true negative log-posterior minus the reference measure's, so the
        ratio corrects the proposal's approximation of the posterior.
        """
        self._tmp.assign(sample.m).axpy(-1.0, self.nu.mean)
        return (sample.cost + self.prior.cost(sample.m)
                - 0.5 * self.nu.Hlr.inner(self._tmp, self._tmp))

    def proposal(self, current):
        self.prior.sample_noise(1.0, self.noise, self.rng)
        self.nu.sample(self.noise, self._w_prior, self._w, add_mean=False)
        s = self.parameters["s"]
        self._w.scale(s)
        self._w.axpy(1.0, self.nu.mean)
        self._tmp.assign(current.m).axpy(-1.0, self.nu.mean)
        self._w.axpy(math.sqrt(max(1.0 - s * s, 0.0)), self._tmp)
        return self._w

    def sample(self, current, proposed):
        proposed.m.assign(self.proposal(current))
        self.init_sample(proposed)
        log_alpha = self.delta(current) - self.delta(proposed)
        if log_alpha > self._log_uniform():
            current.assign(proposed)
            return 1
        return 0

    def consume_random(self):
        self.prior.sample_noise(1.0, self.noise, self.rng)
        self.rng.scalar_uniform()


class MALAKernel(_KernelBase):
    r"""Metropolis-adjusted Langevin, preconditioned by the prior covariance.

    The proposal is
    :math:`m' = m + \tfrac{\tau}{2}\,(-R^{-1}g) + \sqrt{\tau}\,\xi`, with
    :math:`g` the gradient of the negative log-posterior and
    :math:`\xi\sim N(0,R^{-1})`; the Metropolis correction makes the chain exact
    despite the discretization of the Langevin dynamics.
    """

    def __init__(self, model):
        super(MALAKernel, self).__init__(model)
        self.parameters = ParameterList({"delta_t": [0.25, "MALA time step"]})
        self.noise = make_vector(model.prior, "noise")
        self._w = model.generate_vector(PARAMETER)
        self._tmp = model.generate_vector(PARAMETER)

    def name(self):
        return "MALA"

    def derivativeInfo(self):
        return 1

    def init_sample(self, sample):
        """Forward and adjoint solves, the cost, and the preconditioned gradient."""
        m = sample.m
        self.model.solveFwd(sample.u, [sample.u, m, None])
        self.model.solveAdj(sample.p, [sample.u, m, sample.p])
        self.model.evalGradientParameter([sample.u, m, sample.p], sample.g)
        self.model.prior.Rsolver.solve(sample.Cg, sample.g)
        sample.cost = self.model.cost([sample.u, m, sample.p])[0]
        return sample

    def proposal(self, current):
        dt = self.parameters["delta_t"]
        self.model.prior.sample_noise(1.0, self.noise, self.rng)
        self.model.prior.sample(self.noise, self._w, add_mean=False)
        self._w.scale(math.sqrt(dt))
        self._w.axpy(1.0, current.m)
        self._w.axpy(-0.5 * dt, current.Cg)
        return self._w

    def _log_q(self, origin, destination):
        r"""log of the proposal density ``q(origin -> destination)``, up to a
        constant independent of both."""
        dt = self.parameters["delta_t"]
        self._tmp.assign(destination.m).axpy(-1.0, origin.m)
        self._tmp.axpy(0.5 * dt, origin.Cg)
        Rd = self.model.generate_vector(PARAMETER)
        self.model.prior.R.mult(self._tmp, Rd)
        return -0.5 / dt * Rd.inner(self._tmp)

    def sample(self, current, proposed):
        proposed.m.assign(self.proposal(current))
        self.init_sample(proposed)
        log_alpha = ((current.cost - proposed.cost)
                     + self._log_q(proposed, current)
                     - self._log_q(current, proposed))
        if log_alpha > self._log_uniform():
            current.assign(proposed)
            return 1
        return 0

    def consume_random(self):
        self.model.prior.sample_noise(1.0, self.noise, self.rng)
        self.rng.scalar_uniform()


class ISKernel(_KernelBase):
    r"""Importance sampling from the Laplace approximation.

    Proposals are independent draws from ``nu``, and each carries the weight
    :math:`\exp(-\delta(m))`.  The chain is not Markov: every "step" is accepted
    and the weight is what corrects the approximation, so the usual acceptance
    rate is meaningless and always reported as 1.
    """

    def __init__(self, model, nu):
        super(ISKernel, self).__init__(model)
        self.nu = nu
        self.prior = model.prior
        self.noise = make_vector(nu, "noise")
        self._w = model.generate_vector(PARAMETER)
        self._w_prior = model.generate_vector(PARAMETER)
        self._tmp = model.generate_vector(PARAMETER)

    def name(self):
        return "IS"

    def delta(self, sample):
        self._tmp.assign(sample.m).axpy(-1.0, self.nu.mean)
        return (sample.cost + self.prior.cost(sample.m)
                - 0.5 * self.nu.Hlr.inner(self._tmp, self._tmp))

    def proposal(self, current):
        self.prior.sample_noise(1.0, self.noise, self.rng)
        self.nu.sample(self.noise, self._w_prior, self._w, add_mean=False)
        self._w.axpy(1.0, self.nu.mean)
        return self._w

    def sample(self, current, proposed):
        proposed.m.assign(self.proposal(current))
        self.init_sample(proposed)
        current.assign(proposed)
        current.weight = math.exp(-self.delta(proposed))
        return 1

    def consume_random(self):
        self.prior.sample_noise(1.0, self.noise, self.rng)


sync_spellings(_KernelBase)
sync_spellings(MALAKernel)

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Monte Carlo with a Taylor-model control variate.

Plain Monte Carlo estimates :math:`\mathbb{E}[\mathcal{Q}]` with an error falling
like :math:`\sigma/\sqrt{N}`, and each sample costs a PDE solve.  Using the
quadratic Taylor model :math:`\mathcal{Q}_2` as a control variate,

.. math:: \mathbb{E}[\mathcal{Q}] \approx \mathbb{E}[\mathcal{Q}_2]
          + \frac{1}{N}\sum_i \left(\mathcal{Q}(m_i) - \mathcal{Q}_2(m_i)\right),

the sampled quantity is the *difference*, whose variance is small whenever the
map is nearly quadratic.  :math:`\mathbb{E}[\mathcal{Q}_2]` is known analytically,
so nothing is lost and the sample variance is what shrinks.
"""

import numpy as np

from ..common.random import parRandom
from ..modeling.variables import PARAMETER, STATE


def varianceReductionMC(prior, p2qoimap, taylor_qoi, nsamples, order=2,
                        rng=None, verbose=False, comm=None):
    """Estimate the mean of the QoI with and without the Taylor control variate.

    Returns a dict with the plain Monte Carlo estimate, the corrected estimate,
    the two sample standard deviations, and the observed variance reduction
    factor.
    """
    rng = rng if rng is not None else parRandom
    comm = comm if comm is not None else prior.comm
    noise = prior.noise_vector()
    m = p2qoimap.generate_vector(PARAMETER)
    u = p2qoimap.generate_vector(STATE)

    q = np.zeros(int(nsamples))
    qtay = np.zeros(int(nsamples))
    for i in range(int(nsamples)):
        prior.sample_noise(1.0, noise, rng)
        prior.sample(noise, m)
        u.zero() if hasattr(u, "zero") else None
        p2qoimap.solveFwd(u, [u, m, None])
        q[i] = p2qoimap.eval([u, m, None])
        qtay[i] = taylor_qoi.eval(m, order=order)
        if verbose and comm.rank == 0 and (i + 1) % max(nsamples // 10, 1) == 0:
            print("  sample %d/%d: Q = %.6e, Q_taylor = %.6e"
                  % (i + 1, nsamples, q[i], qtay[i]), flush=True)

    diff = q - qtay
    e_taylor = taylor_qoi.expectedValue(order=order)
    mc_mean = float(q.mean())
    vr_mean = e_taylor + float(diff.mean())
    sd_q = float(q.std(ddof=1)) if q.size > 1 else 0.0
    sd_d = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return {
        "nsamples": int(nsamples),
        "mc_mean": mc_mean,
        "mc_stderr": sd_q / np.sqrt(max(q.size, 1)),
        "taylor_mean": e_taylor,
        "taylor_variance": taylor_qoi.variance(order=order),
        "reduced_mean": vr_mean,
        "reduced_stderr": sd_d / np.sqrt(max(diff.size, 1)),
        "sd_q": sd_q,
        "sd_diff": sd_d,
        "variance_reduction": (sd_q / sd_d) ** 2 if sd_d > 0 else float("inf"),
        "samples_q": q,
        "samples_taylor": qtay,
    }


#: snake_case spellings (see :mod:`hippymfem.common.naming`)
variance_reduction_mc = varianceReductionMC

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The quantity of interest: mean temperature in the target volume around the anomaly,
and its posterior spread under the Laplace approximation, two ways."""

import numpy as np

import hippymfem as hm
from hippymfem.forward_uq.qoi import weighted_mean_qoi
from hippymfem.modeling.variables import STATE, PARAMETER, ADJOINT

from .model import ANOMALY, T_SCALE


def target_mean_qoi(Vu, centre=None, radius=None):
    """``q(u) = mean of u over the target box`` as a linear functional of the state."""
    c = centre or ANOMALY["centre"]
    r = radius or 1.5 * ANOMALY["radius"]
    # the indicator, interpolated at the nodes (the array form serves the whole node
    # set at once); the functional M w is assembled as a linear form, no mass matrix
    w = Vu.project(lambda z: ((np.abs(z[..., 0] - c[0]) < r) & (np.abs(z[..., 1] - c[1]) < r)
                              & (np.abs(z[..., 2] - c[2]) < 0.6 * r)).astype(np.float64))
    return weighted_mean_qoi(Vu, w)


def linearized_posterior_std(model, pde, qoi, post, x_map):
    """``sqrt(g^T Gamma_post g)`` with ``g = dq/dm`` at the MAP: the first-order estimate."""
    p2q = hm.Parameter2QoiMap(pde, qoi)
    x = p2q.generate_vector()
    x[PARAMETER].assign(x_map[PARAMETER])
    p2q.solveFwd(x[STATE], x)
    p2q.solveAdj(x[ADJOINT], x)
    g = p2q.generate_vector(PARAMETER)
    p2q.evalGradientParameter(x, g)
    v = g.duplicate()
    post.Hlr.solve(v, g)               # Gamma_post g
    var = g.inner(v)
    return float(np.sqrt(max(var, 0.0))), float(qoi.eval(x)), g


def taylor_qoi(pde, qoi, post, k=30, p=10, n_trace=32, seed=11):
    """First- and second-order Taylor moments of the QoI under the Laplace posterior.

    The second-order mean adds ``1/2 tr(Gamma_post H_q)`` to the plug-in value: what the
    linearization misses when the forward map is convex in ``m`` (a random conductivity
    conducts worse on average than its mean field, so the samples run hotter than the
    MAP).  Two estimates of the trace: the low-rank factorization of the
    posterior-preconditioned QoI Hessian (``k`` eigenpairs), which is right when that
    Hessian is low rank, and a Hutchinson estimate ``mean_i s_i^T H_q s_i`` over
    ``n_trace`` zero-mean posterior samples, which is right whatever the spectrum.  Here
    the spectrum is flat (every dof's ``exp(m)`` adds a little curvature), the low-rank
    trace is nothing and the Hutchinson one is the sample shift.  Returns
    ``(mean1, mean2_lowrank, std1, std2_lowrank, d, mean2_hutchinson, hutch_stderr)``.
    """
    from hippymfem.forward_uq.taylorApproximationQoi import TaylorApproximationQoi

    p2q = hm.Parameter2QoiMap(pde, qoi)
    taylor = TaylorApproximationQoi(p2q, post)
    Omega = hm.MultiVector(p2q.generate_vector(PARAMETER), k + p)
    hm.parRandom.set_seed(seed)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, _ = taylor.computeLowRankFactorization(Omega, k)
    # Hutchinson: tr(Gamma H) = E[s^T H s], s ~ N(0, Gamma_post)
    noise = post.prior.noise_vector()
    s_pr, s_po = p2q.generate_vector(PARAMETER), p2q.generate_vector(PARAMETER)
    hm.parRandom.set_seed(seed + 1)
    vals = []
    for _ in range(n_trace):
        post.prior.sample_noise(1.0, noise)
        post.sample(noise, s_pr, s_po, add_mean=False)
        vals.append(taylor.H.inner(s_po, s_po))
    vals = 0.5 * np.asarray(vals)
    return (float(taylor.expectedValue(order=1)), float(taylor.expectedValue(order=2)),
            float(np.sqrt(max(taylor.variance(order=1), 0.0))), float(np.sqrt(max(taylor.variance(order=2), 0.0))), np.asarray(d),
            float(taylor.expectedValue(order=1) + vals.mean()), float(vals.std(ddof=1) / np.sqrt(n_trace)) if n_trace > 1 else 0.0)


def sampled_qoi(pde, qoi, post, x_map, nsamples, noise, s_prior, s_post, rng_seed=5):
    """The QoI over posterior samples, each through the nonlinear forward solve."""
    hm.parRandom.set_seed(rng_seed)
    vals = []
    u = pde.generate_state()
    # The MAP's linearization point (its Jacobian, blocks and hierarchies) is not needed
    # for forward solves at sampled parameters, and holding it beside each sample's
    # Jacobian would raise the peak by a whole linearization point.
    if hasattr(pde, "release_linearization_point"):
        pde.release_linearization_point()
    for _ in range(nsamples):
        pde.problem_prior = None
        post.prior.sample_noise(1.0, noise)
        post.sample(noise, s_prior, s_post, add_mean=True)
        pde.solveFwd(u, [u, s_post, None])
        vals.append(qoi.eval([u, s_post, None]))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1)) if nsamples > 1 else 0.0, vals

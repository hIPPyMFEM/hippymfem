# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""The Bayesian inverse problem for the Earth's wave speed, in the shape hIPPyMFEM's
solvers expect.

The library's solvers (inexact Newton-CG, the reduced Hessian, the randomized
eigensolvers, the low-rank posterior) talk to a *model*: an object that solves the
forward and the adjoint problem, evaluates the cost and its gradient, and applies the
blocks the Hessian is made of.  :class:`SeismicModel` is such an object for the wave
equation, built on the explicit propagator of :mod:`sem` instead of on a variational
problem with a linear solve.  What differs from a stationary problem is what stands
in for the state vector:

* a forward solve leaves the **seismograms** at the receivers and the **last two
  states** of the field, not its history.  The leapfrog scheme is reversible, so the
  history is recomputed backward from those two states whenever a sweep needs it;
* the adjoint "vector" is the result of the backward sweep, which correlates the
  adjoint field with the forward one as it goes: the misfit's gradient with respect
  to the parameter;
* a Gauss-Newton Hessian action is two sweeps of two fields each (the forward field
  with its increment, then the reconstructed forward field with the incremental
  adjoint), and several directions share the forward field of each sweep.

The data are those of the paper: the leading Fourier coefficients of the ground
motion at each receiver, with independent Gaussian noise.
"""

import time

import numpy as np

import hippymfem as hm
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE


# ------------------------------------------------------------------- the data
class FourierData:
    r"""Seismograms to data: the first ``nfreq`` Fourier coefficients of the ground
    velocity at each receiver.

    The propagator records :math:`y = \rho^{-1}\,\hat r\cdot\nabla P` at the receivers,
    which is the vertical acceleration of the ground (:math:`\rho v_t = \nabla P`, and
    at a free surface, where :math:`P = 0`, the gradient is vertical).  The velocity
    is its integral in time by the trapezoid rule, and the data are

    .. math:: c_k = \frac1T \int_0^T v(t)\, e^{-2\pi i k t/T}\,dt, \qquad k = 0,\dots,
              n_{\rm freq}-1,

    again by the trapezoid rule, stored as their real parts followed by the imaginary
    parts of :math:`k \ge 1`: ``2 nfreq - 1`` numbers per receiver.  The whole map is
    one matrix ``W`` of shape ``(2 nfreq - 1, nsteps + 1)`` applied along the time
    axis; its transpose turns data residuals into the series the adjoint injects.
    """

    def __init__(self, nsteps, dt, nfreq=61):
        N = int(nsteps)
        T = N * dt
        t = dt * np.arange(N + 1)
        w = np.full(N + 1, dt)
        w[0] = w[-1] = 0.5 * dt
        k = np.arange(int(nfreq))
        E = np.exp(-2j * np.pi * np.outer(k, t) / T) * (w / T)[None, :]
        F = np.concatenate([E.real, E.imag[1:]])                 # coefficients of v
        # v_n = dt (y_0/2 + y_1 + ... + y_{n-1} + y_n/2), v_0 = 0, folded into F
        tail = np.concatenate([np.cumsum(F[:, ::-1], axis=1)[:, ::-1][:, 1:],
                               np.zeros((F.shape[0], 1))], axis=1)      # sum over n > j
        W = dt * (tail + 0.5 * F)
        W[:, 0] = 0.5 * dt * tail[:, 0]
        self.W = np.ascontiguousarray(W)
        self.nfreq, self.nsteps, self.dt, self.T = int(nfreq), N, float(dt), T
        self.ndata = self.W.shape[0]
        self.frequencies = k / T

    def apply(self, y):
        """``(..., nsteps + 1, nrec)`` series to ``(..., ndata, nrec)`` data."""
        return np.einsum("kn,...nr->...kr", self.W, y)

    def apply_transpose(self, r):
        """``(..., ndata, nrec)`` to ``(..., nsteps + 1, nrec)``."""
        return np.einsum("kn,...kr->...nr", self.W, r)

    def velocity(self, y):
        """The ground velocity series of recorded series ``y`` (for plotting)."""
        y = np.asarray(y)
        v = np.zeros_like(y)
        v[..., 1:, :] = self.dt * np.cumsum(0.5 * (y[..., 1:, :] + y[..., :-1, :]), axis=-2)
        return v


# ------------------------------------------------------------ the "state vector"
class WaveSolution:
    """What a sweep leaves behind, in the place the solvers keep a state or an
    adjoint vector.  Only the fields the sweep in question produces are set:

    ``y, final, m`` : a forward solve's seismograms, last two states and parameter;
    ``g`` : a backward sweep's parameter-space result;
    ``dm, dy, q`` : a direction, its seismogram sensitivity, an adjoint source.
    """

    __slots__ = ("y", "final", "m", "g", "dm", "dy", "q")

    def __init__(self):
        for k in self.__slots__:
            setattr(self, k, None)

    def assign(self, other):
        for k in self.__slots__:
            setattr(self, k, getattr(other, k))
        return self

    def zero(self):
        for k in self.__slots__:
            setattr(self, k, None)
        return self

    def copy(self):
        return WaveSolution().assign(self)

    duplicate = copy


# ------------------------------------------------------------------- the model
class SeismicModel(hm.Model):
    r"""Wave speed from seismograms: cost, gradient and Gauss-Newton Hessian.

    .. math:: J(m) = \frac{1}{2\sigma^2}\,\|W y(m) - d\|^2
                     + \frac12\,\|m - \bar m\|_{R}^2 ,

    with ``y(m)`` the receivers' series of the propagator at the parameter ``m``
    (the wave speed's departure from the reference model, km/s), ``W`` the data map,
    ``d`` the observed data and ``sigma`` the noise standard deviation.

    Parameters
    ----------
    engine : sem.SpectralWave
        With sources, receivers, ``dt`` and ``nsteps`` set.
    data : FourierData
    prior : a hIPPyMFEM prior on the parameter space
    d_obs : ndarray, ``(ndata, nrec)``
    noise_std : float

    The Hessian is always the Gauss-Newton one, as in the paper, which is the Hessian
    itself where the data are fitted exactly.
    """

    def __init__(self, engine, data, prior, d_obs, noise_std, gradient_norm="M"):
        hm.Model.__init__(self, None, prior, None, gradient_norm=gradient_norm)
        self.engine = engine
        self.data = data
        self.d_obs = np.asarray(d_obs, dtype=np.float64)
        self.noise_std = float(noise_std)
        self.gauss_newton_approx = True
        self._lin = None
        self.n_hess = 0

    # ------------------------------------------------------------------ vectors
    def generate_vector(self, component="ALL"):
        if component == "ALL":
            return [WaveSolution(), self.engine.Vm.vector(), WaveSolution()]
        if component == PARAMETER:
            return self.engine.Vm.vector()
        return WaveSolution()

    def init_parameter(self, m):
        m.zero()
        return m

    # ------------------------------------------------------------------- solves
    def _residual(self, y):
        return (self.data.apply(y) - self.d_obs) / self.noise_std

    def solveFwd(self, out, x):
        """Propagate at ``x[PARAMETER]``; ``out`` keeps the seismograms and the last
        two states."""
        self.n_fwd_solve += 1
        m = x[PARAMETER]
        self.engine.set_parameter(m)
        out.y = self.engine.forward()
        out.final = self.engine.final_state()
        out.m = m.copy()
        return out

    def solveAdj(self, out, x):
        """The backward sweep for the misfit's gradient at ``x``: ``out.g`` is
        :math:`J_y^{\\top} W^{\\top} (W y - d)/\\sigma^2` as a parameter dual vector."""
        self.n_adj_solve += 1
        st = x[STATE]
        self.engine.set_parameter(st.m)
        q = self.data.apply_transpose(self._residual(st.y)) / self.noise_std
        out.g = self.engine.adjoint(q[None], final=st.final)[0]
        return out

    # --------------------------------------------------------------------- cost
    def misfit_cost(self, y):
        r = self._residual(y)
        return 0.5 * float(np.sum(r * r))

    def cost(self, x):
        """``[total, regularization, misfit]`` at ``x = [u, m, p]``."""
        misfit = self.misfit_cost(x[STATE].y)
        reg = self.prior.cost(x[PARAMETER])
        return [misfit + reg, reg, misfit]

    def evalGradientParameter(self, x, mg, misfit_only=False):
        mg.zero()
        mg.axpy(1.0, x[ADJOINT].g)
        if not misfit_only:
            tmp = self.engine.Vm.vector()
            self.prior.grad(x[PARAMETER], tmp)
            mg.axpy(1.0, tmp)
        g = self.engine.Vm.vector()
        solver = self.prior.Msolver if self.gradient_norm == "M" else self.prior.Rsolver
        solver.solve(g, mg)
        return max(g.inner(mg), 0.0) ** 0.5

    # ------------------------------------------------------------------ Hessian
    def setLinearizationPoint(self, x, gauss_newton_approx=True, matrix_free=False):
        self.gauss_newton_approx = True
        self._lin = x[STATE].copy()
        return self

    def setPointForHessianEvaluations(self, x, gauss_newton_approx=True, matrix_free=False):
        return self.setLinearizationPoint(x)

    def _at_linearization(self):
        if self._lin is None:
            raise RuntimeError("setPointForHessianEvaluations has not been called")
        self.engine.set_parameter(self._lin.m)
        return self._lin

    def applyC(self, dm, out):
        out.dm = dm.copy()
        return out

    def solveFwdIncremental(self, sol, rhs):
        self.n_inc_solve += 1
        self._at_linearization()
        sol.dy = self.engine.incremental([rhs.dm])[0]
        return sol

    def applyWuu(self, du, out):
        out.q = self.data.apply_transpose(self.data.apply(du.dy)) / self.noise_std ** 2
        return out

    def solveAdjIncremental(self, sol, rhs):
        self.n_inc_solve += 1
        lin = self._at_linearization()
        sol.g = self.engine.adjoint(rhs.q[None], final=lin.final)[0]
        return sol

    def applyCt(self, dp, out):
        out.zero()
        out.axpy(1.0, dp.g)
        return out

    def _second_order(self, i, j, d, out):
        raise NotImplementedError("the seismic model applies the Gauss-Newton Hessian only")

    def hessian_misfit(self, directions):
        """The Gauss-Newton Hessian of the misfit applied to several directions at
        once: two sweeps, each carrying the forward field once for all of them.
        Returns one parameter dual vector per direction."""
        lin = self._at_linearization()
        dy = self.engine.incremental(list(directions))
        q = self.data.apply_transpose(self.data.apply(dy)) / self.noise_std ** 2
        self.n_hess += len(directions)
        return self.engine.adjoint(q, final=lin.final)


class MisfitHessian:
    """The misfit's Gauss-Newton Hessian as an operator, with a batched product."""

    def __init__(self, model):
        self.model = model
        self.ncalls = 0

    def init_vector(self, x, dim):
        return self.model.engine.Vm.vector()

    def mult(self, x, y):
        y.zero()
        y.axpy(1.0, self.model.hessian_misfit([x])[0])
        self.ncalls += 1
        return y

    def mult_many(self, xs):
        self.ncalls += len(xs)
        return self.model.hessian_misfit(xs)


# ------------------------------------------------------- low-rank approximation
def low_rank_hessian(model, Omega, k, batch=8, double_pass=False, products=None,
                     ndone=0, on_batch=None, log=None, ensemble=None):
    r"""Dominant eigenpairs of :math:`H_{\rm misfit}\,u = \lambda R\,u`, with
    :math:`U^{\top} R U = I`, from products of the Hessian with the random vectors
    ``Omega``, ``batch`` of them per pair of sweeps.

    The single-pass algorithm (the paper's) uses one product per column of ``Omega``;
    ``double_pass`` spends as many again on the orthonormalized range and is more
    accurate in the tail.  Both are hIPPyMFEM's (``singlePassG``/``doublePassG``, one
    power iteration), with the products taken in batches.

    ``products``, if given, is a :class:`hippymfem.MultiVector` whose first ``ndone``
    columns already hold ``H Omega`` (a resumed run); ``on_batch(Y, ndone)`` is
    called after every batch so they can be saved.

    ``ensemble``: a communicator whose ranks each hold the whole problem.  Rank ``r``
    then takes the products of the columns ``r, r + size, ...`` (still ``batch`` to a
    pair of sweeps) and the solves with the prior of the same columns, and the columns
    are exchanged (hIPPyMFEM's ``MatMvMult`` with ``ensemble=``), so every rank returns
    the same eigenpairs; the orthogonalization is done by every rank.

    Returns ``(d, U, Y)``.
    """
    prior = model.prior
    nvec = Omega.nvec()
    if nvec < k:
        raise ValueError("Omega needs at least k = %d columns, has %d" % (k, nvec))
    H = MisfitHessian(model)
    say = log if log is not None else (lambda *a: None)

    shared = ensemble is not None and ensemble.size > 1

    def products_of(X, Y, start=0, tag="H Omega"):
        t0 = time.time()
        if shared:
            from hippymfem.algorithms.multivector import ensemble_columns, ensemble_exchange

            mine = [i for i in ensemble_columns(ensemble, nvec) if i >= start]
        else:
            mine = list(range(start, nvec))
        for a in range(0, len(mine), batch):
            cols = mine[a:a + batch]
            outs = H.mult_many([X[i] for i in cols])
            for i, g in zip(cols, outs):
                Y[i] = g
            if on_batch is not None and tag == "H Omega":
                on_batch(Y, cols[-1] + 1)
            say("  %s: %d of %d products%s, %.1f s" % (tag, a + len(cols) + (0 if shared else start),
                                                        len(mine) + (0 if shared else start),
                                                        " on each of %d ranks" % ensemble.size if shared else "",
                                                        time.time() - t0))
        if shared:
            ensemble_exchange(ensemble, Y)
        return Y

    Rinv = hm.Solver2Operator(prior.Rsolver, init_vector=prior.init_vector)
    Y = products if products is not None else hm.MultiVector(Omega[0], nvec)
    products_of(Omega, Y, int(ndone) if products is not None else 0)
    Q = hm.MultiVector(Omega[0], nvec)
    if shared:
        from hippymfem.algorithms.multivector import MatMvMult

        MatMvMult(Rinv, Y, Q, ensemble)
    else:
        for i in range(nvec):
            Rinv.mult(Y[i], Q[i])
    if not double_pass:
        BQ, _ = Q.Borthogonalize(prior.R)
        Xt = Omega.dot_mv(BQ)
        Wt = Y.dot_mv(Q)
        Tt = np.linalg.solve(Xt, Wt)
        T = 0.5 * (Tt + Tt.T)
    else:
        Q.Borthogonalize(prior.R)
        AQ = products_of(Q, hm.MultiVector(Omega[0], nvec), 0, tag="H Q")
        T = AQ.dot_mv(Q)
        T = 0.5 * (T + T.T)
    d, V = np.linalg.eigh(T)
    order = d.argsort()[::-1][:k]
    d, V = d[order], V[:, order]
    U = hm.MultiVector(Omega[0], k)
    hm.MvDSmatMult(Q, V, U)
    return d, U, Y

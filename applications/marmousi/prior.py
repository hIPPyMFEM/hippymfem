# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""The library's bi-Laplacian prior on the lattice: a stencil, and three ways to solve.

``hm.BiLaplacianPrior(Vm, gamma, delta, Theta, lumped_mass=True)`` without a Robin term has
the precision :math:`R = A L^{-1} A` with :math:`A = \delta M + \gamma K_\Theta` the mass and
the stiffness matrix of trilinear elements and :math:`L` the mass by the rule at the
vertices.  On equal bricks all three are tensor products of three-point matrices, and
:class:`LatticePrior` applies the same :math:`R` (to 1e-14 of the library's, on meshes with
three different steps) as two passes of one stencil of 27 numbers over a rank's block.
The library's own costs what a general mesh costs: its constructor 33 to 38 microseconds
and 1.6 to 2.7 kB an element on one host core (five minutes and 13 to 21 GB a rank at
eight million elements), and its solves are multigrid, the part of the library's scaling
study that kept least of one GPU on 4096.

What makes the stencil serve every rank is the boundary.  With the natural boundary
condition a boundary row of :math:`A` is the interior stencil applied to the field
continued evenly across the face, times the weight of the node in :math:`L` (a half for
each face it lies on).  So with :math:`T = L^{-1} A`,

.. math:: T u = S(\mathrm{mirror}\,u), \qquad A = L\,T, \qquad R = L\,T^2 ,

where :math:`S` is the interior stencil of :math:`T` and the mirror image is written into a
rank's halo where its array leaves the lattice.  :math:`T u` is again even across the
faces, so two passes need one halo of two nodes and no second exchange.

:math:`T` is a convolution, which gives three solves; the object takes the first that
holds:

* one rank holds the lattice: on the even continuation of the field, periodic with
  twice the elements a direction, :math:`T` is diagonalized by the Fourier transform
  (the type-1 cosine transform of the lattice, written as the transform of its
  continuation).  Exact.
* several ranks: a solution decays away from its source by :math:`e^{-\mu}` a node,
  :math:`\cosh\mu = (1 + (\kappa h)^2/3) / (1 - (\kappa h)^2/6)`,
  :math:`\kappa^2 = \delta/(\gamma\theta)`, so a rank's own transform of its block with a
  halo of :math:`W` nodes, periodic on that array, is the solve on its own nodes to
  :math:`e^{-\mu W}`: one exchange and two transforms, no global transform and no
  multigrid.  At 3.75 nodes a correlation length :math:`\mu = 0.54`, 35 nodes for 1e-8.
* where that halo would be wider than the caller allows (a correlation length of many
  nodes): the Chebyshev polynomial of :math:`T` on the interval of its symbol, which is
  known in closed form.  One linear, symmetric, positive map of a fixed number of
  stencil passes.

The symbol also says what the prior's variance on the lattice is, which is not what the
continuous operator's formula gives when a correlation length is a few nodes (740 for 300
asked at 3.75 nodes): :meth:`LatticePrior.coefficients` sets :math:`\gamma` and
:math:`\delta` from it.

Lengths, steps and the tensor's diagonal are in the order x, y, z; arrays are ``[k, j, i]``
(see :mod:`lattice`).  The device functions take and return a rank's owned block.
"""
import itertools
import math

import numpy as np

from hippymfem.common.operators import Operator, init_vector_like
from hippymfem.common.parvector import ParVector
from hippymfem.common.random import parRandom
from hippymfem.fem import kernel as kernel_mod

import jax
import jax.numpy as jnp

from lattice import DeviceHalo, Halo


def stencil(h, gamma, delta, theta=(1.0, 1.0, 1.0)):
    """The 27 weights ``[dk, dj, di]`` of :math:`T = L^{-1} A` at a node inside the
    lattice: of the mass, a sixth of ``(1, 4, 1)`` a direction, and of the stiffness
    ``(-1, 2, -1) / h^2`` in one direction times the mass in the two others."""
    m = np.array([1.0, 4.0, 1.0]) / 6.0
    k = [th * np.array([-1.0, 2.0, -1.0]) / (hd * hd) for hd, th in zip(h, theta)]

    def tensor(z, y, x):
        return z[:, None, None] * y[None, :, None] * x[None, None, :]

    return delta * tensor(m, m, m) + gamma * (tensor(m, m, k[0]) + tensor(m, k[1], m) + tensor(k[2], m, m))


def _symbol(c, h, gamma, delta, theta):
    """The multiplier of :math:`T` at the cosines ``c = (cz, cy, cx)`` of the angles a
    node (arrays that broadcast, NumPy's or JAX's): what :func:`stencil` sums to on a
    wave."""
    m = [(2.0 + ca) / 3.0 for ca in c]
    k = [2.0 * th * (1.0 - ca) / (hd * hd) for ca, hd, th in zip(c, h[::-1], theta[::-1])]
    return delta * m[0] * m[1] * m[2] + gamma * (k[0] * m[1] * m[2] + m[0] * k[1] * m[2] + m[0] * m[1] * k[2])


def decay(h, gamma, delta, theta=(1.0, 1.0, 1.0)):
    """``(mu_z, mu_y, mu_x)``: a solution of :math:`T u = f` falls by at least ``exp(-mu)`` a
    node along an axis away from where ``f`` is.

    A wave across the axis falls along it at the rate of ``cosh(mu) = |1 + q/3| / |1 - q/6|``
    with ``q = (kappa h)^2`` for a field that is constant across, and larger for one that
    is not: up to ``q + 12 sum_e theta_e h^2 / (theta h_e^2)`` over the two other
    directions.  The rate rises with ``q`` up to 6, where a node does not see its
    neighbours along the axis, and falls beyond (the field then alternates in sign), to
    ``acosh(2) = 1.32``.  So the slowest wave is the constant one for correlation lengths
    above 1.3 steps (equal steps, no anisotropy) and the most oscillating one below: the
    lesser of the two ends.
    """
    def rate(q):
        return math.inf if q == 6.0 else math.acosh(abs((1.0 + q / 3.0) / (1.0 - q / 6.0)))

    out = []
    for d in (2, 1, 0):
        scale = h[d] * h[d] / (gamma * theta[d])
        across = 12.0 * gamma * sum(theta[e] / (h[e] * h[e]) for e in range(3) if e != d)
        out.append(min(rate(delta * scale), rate((delta + across) * scale)))
    return tuple(out)


def interval(h, gamma, delta, theta=(1.0, 1.0, 1.0)):
    """The least and the greatest value of the symbol of :math:`T`: it is linear in each
    of the three cosines, so they are among its values at the eight corners."""
    values = [_symbol(c, h, gamma, delta, theta) for c in itertools.product((1.0, -1.0), repeat=3)]
    return min(values), max(values)


def variance(h, gamma, delta, theta=(1.0, 1.0, 1.0), rtol=1e-10):
    """The pointwise variance of the prior at a node far from the lattice's faces: the
    mean of :math:`1/\\hat R` over the angles, :math:`\\hat R = h_x h_y h_z \\hat T^2`.

    The mean over ``M`` equal angles a direction is the variance on a periodic lattice of
    ``M`` nodes, which differs from the unbounded one's by ``exp(-mu M)``; the symbol is
    even, so half of the angles are computed.
    """
    grids = []
    for mu in decay(h, gamma, delta, theta):
        half = max(8, int(math.ceil(0.5 * (5.0 - math.log(rtol)) / mu))) if math.isfinite(mu) else 8
        weight = np.full(half + 1, 2.0)
        weight[0] = weight[-1] = 1.0
        grids.append((np.cos(np.pi * np.arange(half + 1) / half), weight / (2 * half)))
    (cz, wz), (cy, wy), (cx, wx) = grids
    total, rows = 0.0, max(1, 2 ** 22 // (cy.size * cx.size))
    for a in range(0, cz.size, rows):               # a few million angles at a time
        c = (cz[a:a + rows, None, None], cy[None, :, None], cx[None, None, :])
        t = _symbol(c, h, gamma, delta, theta)
        total += float(np.einsum("k,j,i,kji->", wz[a:a + rows], wy, wx, 1.0 / (t * t)))
    return total / (h[0] * h[1] * h[2])


def coefficients(sigma, rho, h, theta=(1.0, 1.0, 1.0)):
    """``(gamma, delta)`` for a pointwise standard deviation ``sigma`` and a correlation
    length ``rho`` on a lattice of steps ``h``: ``delta / gamma = kappa^2`` with
    ``kappa = 2 / rho`` as for the continuous operator in three dimensions, and the size
    of the two such that the LATTICE's variance (:func:`variance`) is ``sigma^2``.  The
    continuous operator's formula, ``hm.BiLaplacianComputeCoefficients``, gives a prior
    whose samples on the lattice have 2.5 times the variance asked at 3.75 nodes a
    correlation length and 1.3 times at 20.

    What the lattice adds is at its own scale, so a variance set this way is not the
    smooth part's: at 3.75 nodes a correlation length two neighbours have the correlation
    0.15 where ``exp(-kappa h)`` is 0.59, and nodes 4 and 8 steps apart 0.047 and 0.005
    where it is 0.118 and 0.014, the covariance of the continuous field of a variance 2.5
    times smaller.  At 1.5 nodes neighbours have -0.12: the samples are close to white."""
    kappa2 = 4.0 / (rho * rho)
    # the precision is of second degree in (gamma, delta): scaling both by s divides the variance by s^2
    s = math.sqrt(variance(h, 1.0, kappa2, theta)) / sigma
    return s, s * kappa2


class _Product(Operator):
    """A map of lattice arrays as an operator of the library, on vectors of the parameter
    space: a copy to the device and one back."""

    def __init__(self, prior, fn):
        self.prior, self.fn = prior, fn

    def init_vector(self, x, dim):
        return self.prior.init_vector(x, dim)

    def mult(self, x, y):
        return self.prior.vector(self.fn(self.prior.lattice(x)), y)


class _Solve:
    """The same as a solver of the library.  (Not one class with :class:`_Product`: an
    object with both ``mult`` and ``solve`` is taken for an operator where an inverse is
    meant, ``randomizedEigensolver._as_operator``.)"""

    def __init__(self, prior, fn):
        self.prior, self.fn = prior, fn

    def init_vector(self, x, dim):
        return self.prior.init_vector(x, dim)

    def solve(self, x, b):
        self.prior.vector(self.fn(self.prior.lattice(b)), x)
        return 0


class LatticePrior:
    """The bi-Laplacian prior with a lumped mass and natural boundary conditions on the
    block of ``lat`` that ``rank`` holds.

    ``h = (hx, hy, hz)`` are the steps, ``theta`` the diagonal of the diffusion tensor,
    ``dtype`` the precision of the device arrays.  ``tol`` is what a solve on several
    ranks is accurate to; ``solver`` names the way (``"cosine"``, ``"fft"``,
    ``"chebyshev"``) or leaves it to the object (``"auto"``: the cosine transform on one
    rank, else the local transform where its halo is at most ``halo`` nodes, else the
    polynomial, which exchanges a halo of ``steps`` nodes every ``steps`` passes).  What
    was chosen is in ``solver``, ``width`` (the transform's halo ``(wz, wy, wx)``),
    ``degree`` (the polynomial's) and ``bound`` (the error it stands for: 0,
    ``exp(-mu W)``, the Chebyshev bound).

    On the device, own block in and own block out: :meth:`apply_A`, :meth:`apply_R`,
    :meth:`solve_A`, :meth:`solve_R`, :meth:`sample`, :meth:`noise`, and ``dot(a, b)``, the
    inner product reduced over the ranks.  From the symbol: :meth:`symbols`,
    :meth:`variance`, :meth:`coefficients`.

    With ``lmap`` (a :class:`lattice.LatticeMap`) it is also a prior of the library, on
    ``ParVector`` s of ``lmap.Vm``: ``R``, ``Rsolver``, ``M``, ``Msolver``, ``mean`` (a
    vector, zero by default), ``Vh``, ``cost``, ``grad``, ``init_vector``,
    ``noise_vector``, ``sample_noise``, ``sample``, ``getHessianPreconditioner``.  Each
    call copies its vector to the device and the result back, which is right for the
    outer iteration; an inner solve calls the device functions.

    The variance is :meth:`variance` away from the faces and twice, four and eight times
    that at a face, an edge and a corner of the box (the mirror image of a node on a face
    is the node itself): the price of the natural boundary condition, on which the mirror
    image, and so the stencil at the boundary and the two transforms, rest.
    """

    def __init__(self, lat, rank, comm, h, gamma, delta, theta=(1.0, 1.0, 1.0), dtype=np.float64,
                 mean=None, lmap=None, tol=1e-8, solver="auto", halo=64, steps=8, device=None):
        self.lat, self.rank, self.comm = lat, int(rank), comm
        self.h = tuple(float(x) for x in h)
        self.theta = tuple(float(t) for t in theta)
        self.gamma, self.delta, self.tol = float(gamma), float(delta), float(tol)
        if min(self.h + self.theta + (self.gamma, self.delta)) <= 0.0:
            raise ValueError("the lattice prior needs positive steps, gamma, delta and theta (h %s, "
                             "gamma %g, delta %g, theta %s)" % (self.h, self.gamma, self.delta, self.theta))
        self.dtype = np.dtype(dtype)
        self.device = device if device is not None else kernel_mod.device()
        self.steps = max(1, int(steps))
        self.mu = decay(self.h, self.gamma, self.delta, self.theta)
        self.interval = interval(self.h, self.gamma, self.delta, self.theta)
        self._variance = None
        self._set_stencil()
        self._set_solver(solver, int(halo))
        self.lmap = lmap
        if lmap is not None:
            self.Vh = lmap.Vm
            self.mean = mean if mean is not None else lmap.Vm.vector()
            self.R, self.Rsolver = _Product(self, self.apply_R), _Solve(self, self.solve_R)
            self.M, self.Msolver = _Product(self, self._lump), _Solve(self, self._unlump)
        elif mean is not None:
            raise ValueError("a mean is a vector of the library: it needs lmap")

    # ----------------------------------------------------------------- set-up
    def _halo(self, W, fft=False):
        return DeviceHalo(Halo(self.lat, self.rank, W, fft=fft), self.comm, self.dtype, self.device)

    def _set_stencil(self):
        S = stencil(self.h, self.gamma, self.delta, self.theta).tolist()

        def T(a):
            """The interior stencil: an array of n nodes a side gives n - 2."""
            n0, n1, n2 = a.shape
            out = None
            for dk, dj, di in itertools.product(range(3), repeat=3):
                term = S[dk][dj][di] * a[dk:n0 - 2 + dk, dj:n1 - 2 + dj, di:n2 - 2 + di]
                out = term if out is None else out + term
            return out

        self._T = T
        self.halo1, self.halo2 = self._halo(1), self._halo(2)
        self.own_shape, self.dot = self.halo2.own_shape, self.halo2.dot
        # the lumped mass on the owned block, a factor an axis: the step, halved on a face of the lattice
        lump = []
        for ax, ((lo, hi), g, hd) in enumerate(zip(self.lat.owned(self.lat.block(self.rank)), self.lat.shape,
                                                    self.h[::-1])):
            i = np.arange(lo, hi)
            w = hd * np.where((i == 0) | (i == g - 1), 0.5, 1.0)
            lump.append(w.reshape([-1 if a == ax else 1 for a in range(3)]))

        def put(x):
            return jax.device_put(x.astype(self.dtype), self.device)

        lz, ly, lx = (put(w) for w in lump)
        uz, uy, ux = (put(1.0 / w) for w in lump)
        qz, qy, qx = (put(1.0 / np.sqrt(w)) for w in lump)
        # where the owned block lies in the closed one, which is what the passes leave of a halo array
        own = tuple(slice(s.start - 1, s.stop - 1) for s in self.halo1.own_slices)
        self._lump = jax.jit(lambda a: a * lz * ly * lx)
        self._unlump = jax.jit(lambda a: a * uz * uy * ux)
        self._unroot = jax.jit(lambda a: a * qz * qy * qx)
        self._A = jax.jit(lambda ext: T(ext)[own] * lz * ly * lx)
        self._R = jax.jit(lambda ext: T(T(ext))[own] * lz * ly * lx)

    def _set_solver(self, solver, halo):
        one = self.lat.size == 1
        width = tuple(max(1, int(math.ceil(-math.log(self.tol) / mu))) if math.isfinite(mu) else 1
                      for mu in self.mu)
        if solver == "auto":
            solver = "cosine" if one else "fft" if max(width) <= halo else "chebyshev"
        self.solver, self.width, self.degree, self.bound = solver, None, None, 0.0
        if solver == "cosine":
            if not one:
                raise ValueError("the cosine transform solves on one rank; the lattice is in %s blocks"
                                 % (self.lat.P,))
            self._fourier = self._cosine()
        elif solver == "fft":
            if max(width) > halo:
                raise ValueError(
                    "a rank's own transform is accurate to exp(-mu W): the tolerance %.1e with a decay of "
                    "%s a node (z, y, x) asks a halo of %s nodes, and %d are allowed; take the "
                    "polynomial (solver=\"chebyshev\"), a looser tolerance or a wider halo"
                    % (self.tol, tuple(round(m, 3) for m in self.mu), width, halo))
            self.width = width
            self.bound = max(math.exp(-mu * w) for mu, w in zip(self.mu, width))
            self.haloW = self._halo(width, fft=True)
            self._fourier = self._transform(self.haloW.shape, self.haloW.own_slices)
        elif solver == "chebyshev":
            a, b = self.interval
            centre, half = 0.5 * (a + b), 0.5 * (b - a)
            sigma = centre / half
            self.degree = k = max(1, int(math.ceil(math.acosh(1.0 / self.tol) / math.acosh(sigma))))
            self.bound = 1.0 / math.cosh(k * math.acosh(sigma))
            # Chebyshev's iteration (Saad, Iterative Methods, algorithm 12.1): the step j + 1 is
            # d <- rho_j rho_(j+1) d + (2 rho_(j+1) / half) r
            rho, coef = 1.0 / sigma, []
            for _ in range(k - 1):
                new = 1.0 / (2.0 * sigma - rho)
                coef.append((new * rho, 2.0 * new / half))
                rho = new
            self._coef = jax.device_put(np.array(coef, dtype=self.dtype).reshape(-1, 2), self.device)
            self._first, self._chunks = 1.0 / centre, {}
        else:
            raise ValueError("solver is \"auto\", \"cosine\", \"fft\" or \"chebyshev\", not %r" % (solver,))

    def _divide(self, angles):
        """The division of a transform by the symbol of ``T`` or by its square, for the
        angles of the transform's entries along each axis.  The symbol is formed where it
        is used, from three vectors of cosines: no array of the transform's size is kept."""
        c = [jax.device_put(np.cos(t).astype(self.dtype).reshape([-1 if a == ax else 1 for a in range(3)]),
                            self.device) for ax, t in enumerate(angles)]
        h, gamma, delta, theta = self.h, self.gamma, self.delta, self.theta

        def divide(F, power):
            t = _symbol(c, h, gamma, delta, theta)
            return F / (t if power == 1 else t * t)

        return divide

    def _cosine(self):
        """Compiled ``T^-1`` and ``T^-2`` on the whole lattice, exactly: the cosines
        ``cos(pi q i / n)`` of a direction of ``n`` elements are eigenvectors of ``T`` with
        the symbol at the angle ``pi q / n`` as the eigenvalue.

        The type-1 cosine transform is taken a direction at a time, as the real part of
        the transform of the direction's even continuation (the nodes followed by the
        inner ones in reverse, ``2 n`` long), and is its own inverse up to that length.
        Six passes over twice the lattice: 3.0 ms for 201 x 201 x 121 nodes in double on
        an MI210, where one transform of the continuation in all three directions, eight
        times the lattice, and back took 9.4.
        """
        g = self.lat.shape
        divide = self._divide([np.pi * np.arange(n) / (n - 1) for n in g])
        scale = 1.0 / float(np.prod([2 * (n - 1) for n in g]))

        def cosine(a, ax):
            inner = tuple(slice(1, -1) if b == ax else slice(None) for b in range(3))
            return jnp.fft.rfft(jnp.concatenate([a, jnp.flip(a, ax)[inner]], axis=ax), axis=ax).real

        def inverse(a, power):
            for ax in range(3):
                a = cosine(a, ax)
            a = divide(a, power)
            for ax in range(3):
                a = cosine(a, ax)
            return a * scale

        return {p: jax.jit(lambda a, p=p: inverse(a, p)) for p in (1, 2)}

    def _transform(self, shape, cut):
        """Compiled ``T^-1`` and ``T^-2`` of a field on a periodic array of ``shape`` (a
        rank's halo array, mirrored where it leaves the lattice), the entries ``cut`` of the
        result."""
        freq = (np.fft.fftfreq, np.fft.fftfreq, np.fft.rfftfreq)
        divide = self._divide([2.0 * np.pi * f(n) for f, n in zip(freq, shape)])

        def inverse(a, power):
            F = divide(jnp.fft.rfftn(a, axes=(0, 1, 2)), power)
            return jnp.fft.irfftn(F, s=shape, axes=(0, 1, 2))[cut]

        return {p: jax.jit(lambda a, p=p: inverse(a, p)) for p in (1, 2)}

    def _chunk(self, m):
        """A halo of ``m`` nodes and ``m`` compiled steps of Chebyshev's iteration on it:
        the residual and the direction come with their halos, every pass of the stencil
        takes a node off each side, and they leave as owned blocks."""
        if m not in self._chunks:
            halo, T = self._halo(m), self._T
            own = halo.own_slices

            def run(x, r, d, coef):
                for j in range(m):
                    x = x + d[tuple(slice(s.start - j, s.stop - j) for s in own)]
                    r = r[1:-1, 1:-1, 1:-1] - T(d)
                    d = coef[j, 0] * d[1:-1, 1:-1, 1:-1] + coef[j, 1] * r
                core = tuple(slice(s.start - m, s.stop - m) for s in own)
                return x, r[core], d[core]

            self._chunks[m] = (halo, jax.jit(run))
        return self._chunks[m]

    def _chebyshev(self, g):
        """The polynomial of ``T`` that stands for ``T^-1``, applied to ``g``."""
        x, r, d = jnp.zeros_like(g), g, g * self._first
        done, passes = 0, self.degree - 1
        while done < passes:
            m = min(self.steps, passes - done)
            halo, run = self._chunk(m)
            x, r, d = run(x, halo.mirror(halo.fill(r)), halo.mirror(halo.fill(d)), self._coef[done:done + m])
            done += m
        return x + d

    def _inverse(self, g, power):
        """``T^-power g``, by the way the object chose."""
        if self.solver == "cosine":
            return self._fourier[power](g)
        if self.solver == "fft":
            return self._fourier[power](self.haloW.mirror(self.haloW.fill(g)))
        for _ in range(power):
            g = self._chebyshev(g)
        return g

    # ----------------------------------------------------------- on the device
    def apply_A(self, a):
        """``A a``."""
        return self._A(self.halo1.mirror(self.halo1.fill(a)))

    def apply_R(self, a):
        """``R a = A L^-1 A a``: one halo of two nodes, two passes of the stencil."""
        return self._R(self.halo2.mirror(self.halo2.fill(a)))

    def solve_A(self, f):
        """``A^-1 f``."""
        return self._inverse(self._unlump(f), 1)

    def solve_R(self, f):
        """``R^-1 f``, the prior's covariance applied."""
        return self._inverse(self._unlump(f), 2)

    def noise(self, sigma=1.0, rng=None):
        """Independent ``N(0, sigma^2)`` on the owned block.  A node's number is drawn at
        its global index in the stream of ``rng`` (the library's ``parRandom`` by
        default), so the field is the same on any number of ranks, and nothing is sent."""
        return jax.device_put(self._noise(sigma, rng).astype(self.dtype), self.device)

    def _noise(self, sigma, rng):
        rng = parRandom if rng is None else rng
        (k0, k1), (j0, j1), (i0, i1) = self.lat.owned(self.lat.block(self.rank))
        gz, gy, gx = self.lat.shape
        # a row of the block along x is a run of the stream
        k, j = np.arange(k0, k1, dtype=np.int64)[:, None], np.arange(j0, j1, dtype=np.int64)[None, :]
        xi = rng.normal_blocks(sigma, (gx * (j + gy * k) + i0).reshape(-1), i1 - i0)
        rng.advance(gz * gy * gx)
        return np.asarray(xi).reshape(self.own_shape)

    def sample(self, noise=None, s=None, add_mean=True, rng=None):
        """A sample of the prior, in one of two forms.

        Given a lattice array ``noise`` of independent standard normals (:meth:`noise`):
        the device array ``A^-1 sqrt(L) noise``, whose covariance is ``R^-1``; no mean is
        added.  Given a vector of the library, or nothing (the noise is then drawn): the
        library's ``prior.sample(noise, s, add_mean)``, which fills and returns ``s``."""
        if noise is not None and not isinstance(noise, ParVector):
            return self._inverse(self._unroot(noise), 1)
        if noise is None:
            noise = self.sample_noise(1.0, rng=rng)
        s = self.vector(self.sample(self.lattice(noise)), s)
        if add_mean:
            s.axpy(1.0, self.mean)
        return s

    # --------------------------------------------------------- from the symbol
    def symbols(self, M):
        """``(A^, R^)`` on a periodic lattice of ``M = (mz, my, mx)`` nodes (or one number
        for the three): what the two matrices multiply the unnormalized transform of a
        field on such a box by, real, in the layout of ``rfftn``, ``(mz, my, mx // 2 + 1)``.
        The stencils are convolutions, so this is exact there, and ``R^`` adds to any
        operator that is given by its symbol on the same box."""
        M = (int(M),) * 3 if np.ndim(M) == 0 else tuple(int(m) for m in M)
        c = [np.cos(2.0 * np.pi * f(n)).reshape([-1 if a == ax else 1 for a in range(3)])
             for ax, (f, n) in enumerate(zip((np.fft.fftfreq, np.fft.fftfreq, np.fft.rfftfreq), M))]
        t = _symbol(c, self.h, self.gamma, self.delta, self.theta)
        volume = self.h[0] * self.h[1] * self.h[2]
        return volume * t, volume * t * t

    def variance(self):
        """The pointwise variance away from the faces of the box (:func:`variance`)."""
        if self._variance is None:
            self._variance = variance(self.h, self.gamma, self.delta, self.theta)
        return self._variance

    coefficients = staticmethod(coefficients)

    # ------------------------------------------------------ the library's face
    def lattice(self, v):
        """The owned block of a vector of the library, on the device."""
        return self.lmap.to_lattice(v, device=self.device, dtype=self.dtype)

    def vector(self, a, v=None):
        """The vector of the library (``v``, or a new one) of an owned block."""
        return self.lmap.from_lattice(a, v)

    def init_vector(self, x, dim):
        """As the library's: ``x`` made a vector of the parameter space.  The noise is one
        number a node, so ``dim="noise"`` is the same."""
        return init_vector_like(x, self.Vh.vector())

    def noise_vector(self):
        return self.Vh.vector()

    def sample_noise(self, sigma=1.0, out=None, rng=None):
        """A vector of independent ``N(0, sigma^2)``, the same on any number of ranks."""
        return self.vector(self._noise(sigma, rng), out)

    def cost(self, m):
        """``1/2 (m - mean)^T R (m - mean)``."""
        d = self.lattice(m) - self.lattice(self.mean)
        return 0.5 * self.dot(d, self.apply_R(d))

    def grad(self, m, out):
        """``R (m - mean)`` in ``out``."""
        return self.vector(self.apply_R(self.lattice(m) - self.lattice(self.mean)), out)

    def getHessianPreconditioner(self, tolerance=None, cycles=None):
        """``Rsolver``: it is one linear map at the cost of two transforms already, so
        there is nothing to loosen and no cycle to count."""
        return self.Rsolver

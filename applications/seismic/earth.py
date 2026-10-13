# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The Earth of the global seismic inverse problem: reference model, truth, mesh.

Three things the application needs before any wave is propagated:

* **PREM** (Dziewonski and Anderson, 1981), the radially symmetric reference model:
  density and compressional wave speed as polynomials of the radius in each layer.
  The crust and the ocean are replaced by the upward continuation of the lid, as a
  mesh with 100 km elements cannot hold a 24 km crust.
* **S20RTS** (Ritsema, van Heijst and Woodhouse, 1999), the lateral variations that
  serve as the synthetic truth: shear speed anomalies in spherical harmonics up to
  degree 20 on 21 radial splines between the core-mantle boundary and the Moho,
  converted to compressional speed with ``dln vp = 0.5 dln vs`` (Ritsema and van
  Heijst, 2004).  The coefficient file is the one distributed with SPECFEM3D_GLOBE.
* **The mesh**: a ball of hexahedra in 13 warped cubes (a central cube, six blocks
  from it to the core-mantle boundary, six more to the surface), conforming, with a
  layer boundary on every first-order discontinuity of PREM in the mantle.  It is
  built as a straight-sided mesh in a *cube configuration* and warped by an
  analytic map applied to the nodes of whatever order, after any refinement, so the
  nodes of the core-mantle boundary and of the surface lie on their spheres exactly.

Lengths are in km, times in s, wave speeds in km/s, densities in g/cm^3.
"""

import os

import numpy as np

#: S20RTS's coefficients: the file distributed with SPECFEM3D_GLOBE, fetched into data/
S20RTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "S20RTS.dat")
S20RTS_URL = "https://raw.githubusercontent.com/SPECFEM/specfem3d_globe/master/DATA/s20rts/S20RTS.dat"


def fetch():
    """Download S20RTS's coefficients into ``data/`` (115 kB)."""
    import urllib.request

    os.makedirs(os.path.dirname(S20RTS_FILE), exist_ok=True)
    if not os.path.exists(S20RTS_FILE):
        urllib.request.urlretrieve(S20RTS_URL, S20RTS_FILE)
    return S20RTS_FILE

R_EARTH = 6371.0
R_CMB = 3480.0
R_ICB = 1221.5
R_MOHO = 6346.6

# ----------------------------------------------------------------------------- PREM
#: ``(r_top, rho coefficients, vp coefficients)`` from the centre outward, polynomials
#: in ``x = r / 6371`` (isotropic PREM; the last layer is the lid continued to the surface)
_PREM = (
    (1221.5, (13.0885, 0.0, -8.8381, 0.0), (11.2622, 0.0, -6.3640, 0.0)),
    (3480.0, (12.5815, -1.2638, -3.6426, -5.5281), (11.0487, -4.0362, 4.8023, -13.5732)),
    (3630.0, (7.9565, -6.4761, 5.5283, -3.0807), (15.3891, -5.3181, 5.5242, -2.5514)),
    (5600.0, (7.9565, -6.4761, 5.5283, -3.0807), (24.9520, -40.4673, 51.4832, -26.6419)),
    (5701.0, (7.9565, -6.4761, 5.5283, -3.0807), (29.2766, -23.6027, 5.5242, -2.5514)),
    (5771.0, (5.3197, -1.4836, 0.0, 0.0), (19.0957, -9.8672, 0.0, 0.0)),
    (5971.0, (11.2494, -8.0298, 0.0, 0.0), (39.7027, -32.6166, 0.0, 0.0)),
    (6151.0, (7.1089, -3.8045, 0.0, 0.0), (20.3926, -12.2569, 0.0, 0.0)),
    (6371.0, (2.6910, 0.6924, 0.0, 0.0), (4.1875, 3.9382, 0.0, 0.0)),
)
_PREM_TOPS = np.array([row[0] for row in _PREM])
_PREM_RHO = np.array([row[1] for row in _PREM])
_PREM_VP = np.array([row[2] for row in _PREM])
#: radii of the first-order discontinuities of this model, the surface included
PREM_DISCONTINUITIES = (1221.5, 3480.0, 5701.0, 5971.0, 6151.0, 6371.0)


def prem_layer(r):
    """Index of the PREM layer containing radius ``r`` (the lower layer at an interface)."""
    return np.minimum(np.searchsorted(_PREM_TOPS, np.asarray(r, dtype=float), side="left"),
                      len(_PREM) - 1)


def _poly(coef, x):
    return coef[..., 0] + x * (coef[..., 1] + x * (coef[..., 2] + x * coef[..., 3]))


def prem(r, layer=None):
    """``(rho, vp)`` of PREM at radius ``r``.

    ``layer`` selects the layer's polynomial explicitly, which is how a node on an
    interface gets the value of the element it is seen from; by default the layer is
    the one the radius falls in.
    """
    r = np.asarray(r, dtype=float)
    lay = prem_layer(r) if layer is None else np.broadcast_to(np.asarray(layer), r.shape)
    x = r / R_EARTH
    return _poly(_PREM_RHO[lay], x), _poly(_PREM_VP[lay], x)


# --------------------------------------------------------------------------- S20RTS
_SPKNT = np.array([-1.00000, -0.78631, -0.59207, -0.41550, -0.25499, -0.10909, 0.02353,
                   0.14409, 0.25367, 0.35329, 0.44384, 0.52615, 0.60097, 0.66899,
                   0.73081, 0.78701, 0.83810, 0.88454, 0.92675, 0.96512, 1.00000])


def _three_point_slope(x, y, end):
    """Slope at an end knot of the parabola through the three knots next to it."""
    if end == 0:
        h, h2 = x[1] - x[0], x[2] - x[0]
        return (y[0] * (h * h - h2 * h2) + y[1] * h2 * h2 - y[2] * h * h) / (h * h2 * (h2 - h))
    h, ha = x[-1] - x[-2], x[-3] - x[-1]
    y0 = -h * ha * (ha + h)
    return (y[-1] * (h * h - ha * ha) + y[-2] * ha * ha - y[-3] * h * h) / y0


class S20RTS:
    """The S20RTS shear speed anomaly ``dln vs(r, colatitude, longitude)``.

    The radial basis is the set of cardinal cubic splines on Ritsema's 21 knots (C2,
    with the three-point slope at both ends, which is what the ``rspln`` routine of
    the reference implementation computes); coefficient ``k`` multiplies the spline
    of knot ``21 - k``, so ``k = 0`` is the Moho and ``k = 20`` the core-mantle
    boundary.  The angular basis is the fully normalized harmonics with the
    Condon-Shortley phase, real form ``a cos(m phi) + b sin(m phi)``.
    """

    NK, NS = 20, 20

    def __init__(self, path=None):
        from scipy.interpolate import CubicSpline

        if path is None:
            path = S20RTS_FILE
        if not os.path.exists(path):
            raise FileNotFoundError("%s is missing: python earth.py fetch" % path)
        vals = np.array(open(path).read().split(), dtype=float)
        nk, ns = self.NK, self.NS
        if vals.size != (nk + 1) * (ns + 1) ** 2:
            raise ValueError("%s: %d numbers, expected %d" % (path, vals.size, (nk + 1) * (ns + 1) ** 2))
        self.a = np.zeros((nk + 1, ns + 1, ns + 1))
        self.b = np.zeros((nk + 1, ns + 1, ns + 1))
        p = 0
        for k in range(nk + 1):
            for l in range(ns + 1):
                self.a[k, l, 0] = vals[p]
                self.a[k, l, 1:l + 1] = vals[p + 1:p + 2 * l + 1:2]
                self.b[k, l, 1:l + 1] = vals[p + 2:p + 2 * l + 2:2]
                p += 2 * l + 1
        self.splines = []
        for j in range(nk + 1):
            y = np.zeros(nk + 1)
            y[j] = 1.0
            self.splines.append(CubicSpline(
                _SPKNT, y, bc_type=((1, _three_point_slope(_SPKNT, y, 0)),
                                    (1, _three_point_slope(_SPKNT, y, 1)))))

    def radial_basis(self, r):
        """``(npts, 21)``: the 21 radial functions at radii ``r`` (km), zero in the core.

        Above the Moho the value at the Moho is kept, so the lid that stands in for
        the crust here carries the anomaly of the uppermost mantle.
        """
        r = np.asarray(r, dtype=float)
        xr = -1.0 + 2.0 * (np.minimum(r, R_MOHO) - R_CMB) / (R_MOHO - R_CMB)
        out = np.stack([self.splines[self.NK - k](np.clip(xr, -1.0, 1.0))
                        for k in range(self.NK + 1)], axis=-1)
        out[r <= R_CMB * (1.0 + 1e-12)] = 0.0
        return out

    def dlnvs(self, xyz, lmax=None):
        """The anomaly at the points ``xyz`` (``(n, 3)``, km), as a fraction."""
        xyz = np.asarray(xyz, dtype=float)
        r = np.linalg.norm(xyz, axis=1)
        rs = np.maximum(r, 1e-300)
        theta = np.arccos(np.clip(xyz[:, 2] / rs, -1.0, 1.0))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        rb = self.radial_basis(r)                               # (n, 21)
        out = np.zeros(xyz.shape[0])
        active = np.nonzero(r > R_CMB * (1.0 + 1e-12))[0]
        if active.size == 0:
            return out
        th, ph, rba = theta[active], phi[active], rb[active]
        acc = np.zeros(active.size)
        lmax = self.NS if lmax is None else int(lmax)
        X = colatitude_harmonics(lmax, th)
        for l in range(lmax + 1):
            al = rba @ self.a[:, l, :]                          # (n, 21): radial sums per m
            bl = rba @ self.b[:, l, :]
            for m in range(l + 1):
                x = X[(l, m)]                                   # X_lm(theta), phase included
                if m == 0:
                    acc += al[:, 0] * x
                else:
                    acc += (al[:, m] * np.cos(m * ph) + bl[:, m] * np.sin(m * ph)) * x
        out[active] = acc
        return out


def colatitude_harmonics(lmax, theta):
    r"""The colatitude part of the fully normalized spherical harmonics,
    :math:`X_l^m(\theta) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}\,
    P_l^m(\cos\theta)` with the Condon-Shortley phase, for ``0 <= m <= l <= lmax``,
    as a dictionary ``(l, m) -> array``.  By the standard three-term recurrences, which
    are stable for these degrees."""
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    X = {(0, 0): np.full(theta.shape, np.sqrt(0.25 / np.pi))}
    for m in range(1, lmax + 1):
        X[(m, m)] = -np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * s * X[(m - 1, m - 1)]
    for m in range(lmax):
        X[(m + 1, m)] = np.sqrt(2.0 * m + 3.0) * c * X[(m, m)]
        for l in range(m + 2, lmax + 1):
            a = np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
            b = np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
            X[(l, m)] = a * (c * X[(l - 1, m)] - b * X[(l - 2, m)])
    return X


# ----------------------------------------------------------------------------- mesh
def mantle_levels(h):
    """Radii of the layer boundaries from the core-mantle boundary to the surface.

    Every first-order discontinuity of PREM in the mantle is a boundary, and each
    interval between two of them is divided evenly into layers of about ``h`` km.
    """
    marks = [R_CMB] + [d for d in PREM_DISCONTINUITIES if R_CMB < d <= R_EARTH]
    out = [R_CMB]
    for a, b in zip(marks[:-1], marks[1:]):
        n = max(1, int(round((b - a) / h)))
        out += list(a + (b - a) * np.arange(1, n + 1) / n)
    return np.array(out)


class BallMesh:
    """A conforming hexahedral mesh of the Earth in 13 warped cubes.

    Parameters
    ----------
    n : int
        Elements along an edge of the central cube, which is also the number across
        each of the six faces of the cubed sphere (a face spans 90 degrees).
    n_core : int
        Layers between the central cube and the core-mantle boundary.
    r_levels : sequence of float, optional
        Radii of the layer boundaries in the mantle, from 3480 to 6371 km;
        :func:`mantle_levels` with the surface element size by default.
    cube : float
        Half-width of the central cube in km.  Its ``n^3`` elements are the smallest
        of the mesh and sit where the waves are fastest, so they set the time step;
        1700 km is about as large as the cube gets before the layers between its
        corners and the core-mantle boundary become thinner than its own elements.

    In the cube configuration the mesh fills ``[-1, 1]^3``: a lattice of ``n^3``
    cubes of half-width ``s0`` at the centre, and around it layers that are the
    surfaces of concentric cubes, each split in ``6 n^2`` cells.  The elements are
    straight there, so uniform refinement is exact, and :meth:`warp` takes any point
    of the configuration to its place in the ball.
    """

    S0 = 0.25

    def __init__(self, n, n_core=None, r_levels=None, cube=1700.0):
        self.n = int(n)
        #: lateral size of an element at the surface, km
        self.h_surface = 0.5 * np.pi * R_EARTH / self.n
        if r_levels is None:
            r_levels = mantle_levels(self.h_surface)
        self.r_levels = np.asarray(r_levels, dtype=float)
        if abs(self.r_levels[0] - R_CMB) > 1e-9 or abs(self.r_levels[-1] - R_EARTH) > 1e-9:
            raise ValueError("r_levels must run from %g to %g km" % (R_CMB, R_EARTH))
        self.cube = float(cube)
        if self.cube * np.sqrt(3.0) >= R_CMB:
            raise ValueError("the central cube's corners must lie inside the core")
        if n_core is None:
            # Layers about as thick as the elements are wide at the surface.  The
            # elements of the core are narrower than that already (their width shrinks
            # with the radius), and it is the central cube that limits the time step:
            # thinner layers here would add elements and shorten the step for nothing.
            n_core = max(1, int((R_CMB - self.cube) / self.h_surface))
        self.n_core = int(n_core)
        self.n_mantle = self.r_levels.size - 1
        self.nlev = self.n_core + self.n_mantle
        #: blend parameter of the inner blocks at their levels, 0 on the cube, 1 on the CMB
        self.t_levels = np.arange(self.n_core + 1) / float(self.n_core)

    # ------------------------------------------------------------ cube configuration
    def level_s(self, k):
        """Half-width of the concentric cube of level ``k`` in the cube configuration."""
        return self.S0 + (1.0 - self.S0) * np.asarray(k, dtype=float) / self.nlev

    def build(self):
        """``(vertices, hexes, attributes, boundary quads)`` of the cube configuration.

        Attribute 1 is the central cube, 2 the blocks of the core, 3 the mantle; the
        boundary quadrilaterals are the surface, attribute 1.
        """
        n, L = self.n, self.nlev
        n1 = n + 1
        I, J, K = np.meshgrid(np.arange(n1), np.arange(n1), np.arange(n1), indexing="ij")
        cid = (I + n1 * J + n1 * n1 * K)                       # lattice point index
        lat = np.stack([I, J, K], axis=-1).reshape(-1, 3)[np.argsort(cid.ravel())]
        on = ((lat == 0) | (lat == n)).any(axis=1)
        ns = int(on.sum())
        sid = np.full(n1 ** 3, -1, dtype=np.int64)
        sid[on] = np.arange(ns)
        unit = 2.0 * lat / n - 1.0                              # in [-1, 1]^3
        ncen = n1 ** 3
        verts = np.empty((ncen + L * ns, 3))
        verts[:ncen] = self.S0 * unit
        for k in range(1, L + 1):
            verts[ncen + (k - 1) * ns: ncen + k * ns] = self.level_s(k) * unit[on]

        def vid(k, c):
            return c if k == 0 else ncen + (k - 1) * ns + sid[c]

        def c(i, j, k):
            return i + n1 * j + n1 * n1 * k

        a = np.arange(n)
        i, j, k = [x.ravel() for x in np.meshgrid(a, a, a, indexing="ij")]
        central = np.stack([c(i, j, k), c(i + 1, j, k), c(i + 1, j + 1, k), c(i, j + 1, k),
                            c(i, j, k + 1), c(i + 1, j, k + 1), c(i + 1, j + 1, k + 1),
                            c(i, j + 1, k + 1)], axis=1)
        p, q = [x.ravel() for x in np.meshgrid(a, a, indexing="ij")]
        z, e = np.zeros_like(p), np.full_like(p, n)
        # the six faces, each quadrilateral counterclockwise seen from outside
        quads = np.concatenate([
            np.stack([c(z, p, q), c(z, p, q + 1), c(z, p + 1, q + 1), c(z, p + 1, q)], axis=1),
            np.stack([c(e, p, q), c(e, p + 1, q), c(e, p + 1, q + 1), c(e, p, q + 1)], axis=1),
            np.stack([c(p, z, q), c(p + 1, z, q), c(p + 1, z, q + 1), c(p, z, q + 1)], axis=1),
            np.stack([c(p, e, q), c(p, e, q + 1), c(p + 1, e, q + 1), c(p + 1, e, q)], axis=1),
            np.stack([c(p, q, z), c(p, q + 1, z), c(p + 1, q + 1, z), c(p + 1, q, z)], axis=1),
            np.stack([c(p, q, e), c(p + 1, q, e), c(p + 1, q + 1, e), c(p, q + 1, e)], axis=1),
        ])
        hexes, attrs = [central], [np.full(central.shape[0], 1, dtype=np.int32)]
        for lev in range(1, L + 1):
            hexes.append(np.concatenate([vid(lev - 1, quads), vid(lev, quads)], axis=1))
            attrs.append(np.full(quads.shape[0], 2 if lev <= self.n_core else 3, dtype=np.int32))
        return (verts, np.concatenate(hexes).astype(np.int64), np.concatenate(attrs),
                vid(L, quads).astype(np.int64))

    def vertex_key(self, q, refine=0):
        """An integer naming each vertex of the (refined) cube configuration, from its
        position ``q`` there: ``((level * m + I) * m + J) * m + K`` with ``m = n + 1``
        lattice points along an edge, ``level`` 0 for the central lattice and the layer
        number outside it.  The same on every rank and for every partition, so a field
        saved in the order of its keys can be read back by any run.
        """
        q = np.asarray(q, dtype=float)
        f = 2 ** int(refine)
        n, L = self.n * f, self.nlev * f
        s = np.abs(q).max(axis=1)
        lev = np.rint((s - self.S0) / (1.0 - self.S0) * L).astype(np.int64)
        lev = np.where(s <= self.S0 * (1.0 + 1e-12), 0, np.maximum(lev, 0))
        scale = np.where(lev == 0, self.S0, s)
        ijk = np.rint((q / scale[:, None] + 1.0) * 0.5 * n).astype(np.int64)
        m = n + 1
        return ((lev * m + ijk[:, 0]) * m + ijk[:, 1]) * m + ijk[:, 2]

    def node_keys(self, order, refine=0):
        """The naming of the nodes of degree ``order`` that :meth:`sem.Collocation.from_keys`
        asks for: ``f(qv, W) -> (keys, boundary)``.

        A node's key is :meth:`vertex_key`'s on a lattice ``order`` times finer in
        every direction, the layers included.  The lattice position of a node is the
        trilinear interpolant of those of its element's vertices, which is exact: an
        element is a unit cube of the lattice (in a layer, the cube of a cell of the
        cubed sphere and of the layer's two levels).
        """
        f = 2 ** int(refine)
        n, L, p = self.n * f, self.nlev * f, int(order)
        m = p * n + 1

        def keys(qv, W, block=1 << 20):
            ne, nq = qv.shape[0], W.shape[0]
            key = np.empty((ne, nq), dtype=np.int64)
            boundary = np.empty((ne, nq), dtype=bool)
            Wt = np.ascontiguousarray(p * W.T)
            for a in range(0, ne, block):
                b = min(a + block, ne)
                comps = self.decode_key(self.vertex_key(qv[a:b].reshape(-1, 3), refine), refine)
                for c, comp in enumerate(comps):                  # level, then the three lattice indices
                    t = np.rint(comp.reshape(b - a, 8).astype(np.float64) @ Wt).astype(np.int64)
                    if c == 0:
                        boundary[a:b] = t == p * L
                        key[a:b] = t
                    else:
                        key[a:b] *= m
                        key[a:b] += t
            return key, boundary

        return keys

    def decode_key(self, key, refine=0):
        """``(level, I, J, K)`` of vertex keys (see :meth:`vertex_key`)."""
        m = self.n * 2 ** int(refine) + 1
        key = np.asarray(key, dtype=np.int64)
        return key // m ** 3, (key // m ** 2) % m, (key // m) % m, key % m

    def level_radius(self, level, refine=0):
        """Radius of the mantle layer boundary numbered ``level`` (as in the keys);
        ``nan`` for the levels of the core, whose layers are not spheres."""
        f = 2 ** int(refine)
        kf = np.asarray(level, dtype=float) / f - self.n_core
        k0 = np.clip(np.floor(kf).astype(np.int64), 0, self.n_mantle - 1)
        r = self.r_levels[k0] + (kf - k0) * (self.r_levels[k0 + 1] - self.r_levels[k0])
        return np.where(kf < -1e-9, np.nan, r)

    # ------------------------------------------------------------------------ the map
    def warp(self, q):
        """Positions in the ball (km) of points ``q`` of the cube configuration."""
        q = np.asarray(q, dtype=float)
        aq = np.abs(q)
        s = aq.max(axis=1)
        x = np.empty_like(q)
        inner = s <= self.S0 * (1.0 + 1e-13)
        x[inner] = (self.cube / self.S0) * q[inner]
        o = np.nonzero(~inner)[0]
        if o.size == 0:
            return x
        qo, so = q[o], s[o]
        d = qo / so[:, None]                                    # on the unit cube's surface
        ax = np.argmax(aq[o], axis=1)
        rows = np.arange(o.size)
        e = np.tan(0.25 * np.pi * np.clip(d, -1.0, 1.0))        # equiangular on each face
        e[rows, ax] = np.sign(qo[rows, ax])
        nrm = e / np.linalg.norm(e, axis=1)[:, None]
        d[rows, ax] = np.sign(qo[rows, ax])
        kf = np.clip((so - self.S0) / (1.0 - self.S0) * self.nlev, 0.0, float(self.nlev))
        k0 = np.minimum(kf.astype(np.int64), self.nlev - 1)
        th = kf - k0
        xo = np.empty_like(qo)
        core = k0 < self.n_core
        if core.any():
            kc = k0[core]
            t = self.t_levels[kc] + th[core] * (self.t_levels[kc + 1] - self.t_levels[kc])
            xo[core] = ((1.0 - t)[:, None] * self.cube * d[core]
                        + (t * R_CMB)[:, None] * nrm[core])
        man = ~core
        if man.any():
            km = k0[man] - self.n_core
            r = self.r_levels[km] + th[man] * (self.r_levels[km + 1] - self.r_levels[km])
            xo[man] = r[:, None] * nrm[man]
        x[o] = xo
        return x

    def warp_jax(self):
        """:meth:`warp` as a JAX function of ``(..., 3)`` arrays, for evaluating the
        geometry on the device (see :class:`sem.Collocation`)."""
        import jax.numpy as jnp

        S0, nlev, n_core, n_mantle, cube = self.S0, self.nlev, self.n_core, self.n_mantle, self.cube
        t_levels = jnp.asarray(self.t_levels)
        r_levels = jnp.asarray(self.r_levels)

        def warp(q):
            aq = jnp.abs(q)
            s = aq.max(axis=-1)
            inner = s <= S0 * (1.0 + 1e-13)
            so = jnp.where(inner, 1.0, s)
            onax = jnp.arange(3) == jnp.argmax(aq, axis=-1)[..., None]
            sgn = jnp.sign(q)
            d = jnp.where(onax, sgn, q / so[..., None])            # on the unit cube's surface
            e = jnp.where(onax, sgn, jnp.tan(0.25 * jnp.pi * jnp.clip(d, -1.0, 1.0)))
            nrm = e / jnp.linalg.norm(e, axis=-1, keepdims=True)
            kf = jnp.clip((so - S0) / (1.0 - S0) * nlev, 0.0, float(nlev))
            k0 = jnp.minimum(kf.astype(jnp.int32), nlev - 1)
            th = kf - k0
            kc = jnp.minimum(k0, n_core - 1)
            t = t_levels[kc] + th * (t_levels[kc + 1] - t_levels[kc])
            x_core = (1.0 - t)[..., None] * cube * d + (t * R_CMB)[..., None] * nrm
            km = jnp.clip(k0 - n_core, 0, n_mantle - 1)
            r = r_levels[km] + th * (r_levels[km + 1] - r_levels[km])
            x_out = jnp.where((k0 < n_core)[..., None], x_core, r[..., None] * nrm)
            return jnp.where(inner[..., None], (cube / S0) * q, x_out)

        return warp

    # ----------------------------------------------------------------------- MFEM mesh
    def serial_mesh(self):
        """The cube configuration as a serial ``mfem.Mesh`` (straight hexahedra)."""
        import mfem.par as mfem

        verts, hexes, attrs, bdr = self.build()
        mesh = mfem.Mesh(3, verts.shape[0], hexes.shape[0], bdr.shape[0], 3)
        for v in verts.tolist():
            mesh.AddVertex(v[0], v[1], v[2])
        for h, a in zip(hexes.tolist(), attrs.tolist()):
            mesh.AddHex(h, a)
        for b in bdr.tolist():
            mesh.AddBdrQuad(b, 1)
        mesh.FinalizeHexMesh(1, 0, False)
        return mesh, hexes, verts

    def partition(self, verts, hexes, nparts):
        """Recursive coordinate bisection of the elements by their warped centroids.

        Deterministic, so every rank computes the same partition, and balanced to one
        element whatever the number of parts.
        """
        cen = self.warp(verts[hexes].mean(axis=1))
        part = np.zeros(hexes.shape[0], dtype=np.int32)

        def split(idx, lo, n):
            if n == 1:
                part[idx] = lo
                return
            nl = n // 2
            c = cen[idx]
            axis = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
            order = np.argsort(c[:, axis], kind="stable")
            cut = int(round(idx.size * nl / float(n)))
            split(idx[order[:cut]], lo, nl)
            split(idx[order[cut:]], lo + nl, n - nl)

        split(np.arange(hexes.shape[0]), 0, int(nparts))
        return part

    def cube_mesh(self, comm, refine=0):
        """The cube configuration as a parallel mesh, partitioned and refined ``refine``
        times (exactly, the elements being straight there), not yet warped.

        A space built on it now sees the vertices at their lattice positions, which is
        where :meth:`vertex_key` reads them; :meth:`warp_mesh` then moves the mesh
        under it.
        """
        import mfem.par as mfem

        serial, hexes, verts = self.serial_mesh()
        if comm.size > 1:
            part = mfem.intArray(self.partition(verts, hexes, comm.size).tolist())
            pmesh = mfem.ParMesh(comm, serial, part.GetData())
            pmesh._keep_part = part
        else:
            pmesh = mfem.ParMesh(comm, serial)
        del serial
        for _ in range(int(refine)):
            pmesh.UniformRefinement()
        return pmesh

    def warp_mesh(self, pmesh, order):
        """Give the cube configuration nodes of degree ``order`` and move them into the
        ball.  Returns the nodes' positions in the cube configuration."""
        return warp_nodes(pmesh, self.warp, order)


def warp_nodes(pmesh, fn, order):
    """Give ``pmesh`` nodes of degree ``order`` and move them by ``fn`` (``(n, 3)`` to
    ``(n, 3)``); returns their positions before the move.  Collective: the nodal grid
    function is built on every rank."""
    import mfem.par as mfem

    pmesh.SetCurvature(int(order), False, 3, mfem.Ordering.byNODES)
    nodes = pmesh.GetNodes()
    arr = nodes.GetDataArray()
    before = np.array(arr)
    nn = arr.size // 3
    arr[:] = fn(before.reshape(3, nn).T).T.reshape(-1)
    return before


# -------------------------------------------------------------- sources and receivers
def sph_to_xyz(r, lat_deg, lon_deg):
    """Cartesian position (km) of radius ``r``, latitude and longitude in degrees."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon),
                     r * np.sin(lat)], axis=-1)


def paper_sources(depth=10.0):
    """The five simultaneous sources of the paper: the North pole and four points on
    the equator 90 degrees apart, all ``depth`` km down."""
    lat = np.array([90.0, 0.0, 0.0, 0.0, 0.0])
    lon = np.array([0.0, 0.0, 90.0, 180.0, 270.0])
    return sph_to_xyz(R_EARTH - depth, lat, lon)


def paper_receivers(nlat=10, nlon=10):
    """``nlat * nlon`` receivers on the surface over the Northern and Eastern
    hemispheres, on parallels 9 degrees apart from 4.5 N to 85.5 N and meridians 18
    degrees apart from 9 E to 171 E (the paper gives 100 receivers on zonal lines
    over these hemispheres, not their coordinates).  Returns ``(xyz, lat, lon)``."""
    lat = 90.0 * (np.arange(nlat) + 0.5) / nlat
    lon = 180.0 * (np.arange(nlon) + 0.5) / nlon
    LAT, LON = [x.ravel() for x in np.meshgrid(lat, lon, indexing="ij")]
    return sph_to_xyz(R_EARTH, LAT, LON), LAT, LON


# ------------------------------------------------------------ the model at the nodes
def nodal_prem(X):
    """``(rho, vp)`` of PREM at element nodes ``X`` (``(ne, nq, 3)``).

    Each element takes the polynomial of the layer its centroid is in, so a node on a
    discontinuity gets, in each element, that element's side of it.
    """
    r = np.linalg.norm(X, axis=2)
    layer = prem_layer(np.linalg.norm(X.mean(axis=1), axis=1))
    return prem(r, layer=layer[:, None])


def prem_mean_vp(r):
    """PREM's compressional speed at radius ``r``, the mean of the two sides at a
    discontinuity: the value a continuous field can take there."""
    r = np.asarray(r, dtype=float)
    eps = 1e-6
    return 0.5 * (prem(np.maximum(r - eps, 0.0))[1] + prem(np.minimum(r + eps, R_EARTH))[1])


def s20rts_perturbation(xyz, model=None, scale=0.5, lmax=None):
    """The synthetic truth as a perturbation of the wave speed, km/s, at points
    ``xyz``: ``dln vp = scale * dln vs`` of S20RTS times PREM's speed (zero in the
    core, where S20RTS says nothing)."""
    model = S20RTS() if model is None else model
    xyz = np.asarray(xyz, dtype=float)
    r = np.linalg.norm(xyz, axis=1)
    return scale * model.dlnvs(xyz, lmax=lmax) * prem_mean_vp(r)


def prem_aux():
    """:func:`nodal_prem` as a JAX function of a block's node positions
    ``(nb, nq, 3)``, returning ``(rho, vp)`` there: the auxiliary values of the wave
    equation's density, evaluated on the device with the coefficients."""
    import jax.numpy as jnp

    tops, rho_c, vp_c = jnp.asarray(_PREM_TOPS), jnp.asarray(_PREM_RHO), jnp.asarray(_PREM_VP)

    def aux(X):
        r = jnp.linalg.norm(X, axis=-1)
        rc = jnp.linalg.norm(X.mean(axis=1), axis=-1)
        lay = jnp.minimum(jnp.searchsorted(tops, rc, side="left"), tops.shape[0] - 1)
        x = r / R_EARTH
        cr, cv = rho_c[lay][:, None, :], vp_c[lay][:, None, :]
        rho = cr[..., 0] + x * (cr[..., 1] + x * (cr[..., 2] + x * cr[..., 3]))
        vp = cv[..., 0] + x * (cv[..., 1] + x * (cv[..., 2] + x * cv[..., 3]))
        return rho, vp

    return aux


def ricker(t, f0, t0=None):
    """A Ricker wavelet of peak frequency ``f0`` centred at ``t0`` (``1.5 / f0`` by
    default); its spectrum is negligible above about ``2.5 f0``."""
    t0 = 1.5 / f0 if t0 is None else t0
    a = (np.pi * f0 * (np.asarray(t, dtype=float) - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["fetch"]:
        print(fetch())
    else:
        sys.exit("usage: python earth.py fetch")

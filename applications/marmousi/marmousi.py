# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The Marmousi model in three dimensions: the data, the speed, the window, the survey.

What the application needs before any wave is propagated:

* **The section** of Hu, Alger, Nammour and Ghattas (arXiv 2507.10804): 9200 m by 2400 m
  on a lattice of 20 m, 461 x 121 speeds between 1500 and 4000 m/s.  It is 400 m of water
  on the original Marmousi model (Institut Francais du Petrole, 1988), every fifth of its
  4 m samples from 40 to 2040 m of depth, cut off at 4000 m/s.  ``python marmousi.py
  fetch`` gets the paper's own file, its wavelet, and the original as Madagascar's server
  has it, from which the section is rebuilt bit for bit; none of them is distributed here.
* **The third dimension**, which Marmousi does not have.  The speed at a point is the
  section's at a place that moves with ``y``::

      v(x, y, z) = V2(X, Z),      V2 the bilinear interpolant of the section's lattice

      s = LX - |(x mod 2 LX) - LX|                 the section continued by mirror images
      X = s + a(x, y) sin(pi s / LX)               stretched along x, both ends fixed
      Z = z + b(x, y) sin^2(pi (z - ZW) / (LZ - ZW))    layers moved up and down under the sea bed

      a = STRETCH (sin(2 pi y / STRETCH_Y[0]) + sin(2 pi y / STRETCH_Y[1]) / 2) m(x)
      b = HEAVE sin(2 pi y / HEAVE_Y) cos(pi s / LX) m(x)
      m = 1 + eps cos(2 pi x / TILE_X)

  A formula of the point, so that any rank evaluates its own block at any resolution and no
  array of the whole box exists anywhere.  It is made to keep what the method relies on:
  the plane ``y = 0`` is the paper's section itself (``a = b = 0`` there); the water stays
  the flat layer it is (``Z = z`` above the sea bed), so the unknown can be frozen in it;
  ``x -> X`` and ``z -> Z`` are monotone maps of the section onto itself, so every speed
  lies between the section's least and greatest and the time step is known before a mesh
  exists; with ``eps = 0`` the speed is even about ``x = 0`` and ``x = LX``.  The strike
  turns by up to 27 degrees and the layers dip across the line by up to 9.  ``eps = 1/2``
  is for surveys of many tiles: no two tiles then hold the same medium.
* **The initial model**: the section smoothed by a Gaussian of 120 m under the sea bed, the
  water kept, carried through the same map.  On ``y = 0`` it is the initial model of the
  paper's runs with the water fixed.
* **The window** of the paper, which multiplies the unknown: one inside, falling to zero
  as ``exp(-(2 (1 - d / margin))^3)`` within a margin of the four sides and of the bottom,
  zero on those faces, and zero in the water.
* **The source's time function**: a cosine under a Gaussian whose spectrum has the paper's
  band, or the paper's own from its file.
* **The survey**: sources and receivers on grids at one depth, over the box or repeated
  over tiles of it.

Lengths are in m, times in s, speeds in m/s; ``x`` runs along the section, ``y`` across
it, ``z`` down from the free surface ``z = 0``.  Lattice arrays are ``[k, j, i]``: ``z``
first, ``x`` fastest.  A driver on a lattice of ``h`` metres, for the whole box or for a
rank's ``block`` of it (``lattice.Lattice.closed(B)``: ranges of nodes along ``z, y, x``)::

    F = on_lattice(box, h, block, comm=comm)      # F.c_true, F.c0, F.window, F.q_true
    aux = lattice_aux(F.c0, box, h, block)        # the initial speed for the propagator
    sources, receivers = survey(box)
    w = wavelet(dt * numpy.arange(nsteps))

    python marmousi.py fetch              # 7.2 MB into data/ (or $MARMOUSI_DATA), on a login node
    python marmousi.py figures OUT        # the model, the window, the wavelets, the survey
"""

import collections
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
#: the folder of the three files: ``data`` beside this file, or the one ``MARMOUSI_DATA``
#: names (a scratch file system)
DATA = os.environ.get("MARMOUSI_DATA") or os.path.join(HERE, "data")
#: the paper's repository (github.com/mathewgaohu/lrs-psido, MIT licence) at the commit this
#: application was written against: an address whose files cannot change
PAPER_URL = ("https://raw.githubusercontent.com/mathewgaohu/lrs-psido/"
             "777081f4f562fe9852cb1bf86e9944c6e3f299e8/marmousi_model/")
#: ``name: (address, bytes, SHA-256)``.  ``mt.npy`` is the paper's section and ``src.rsf@`` its
#: wavelet; ``marmvel.hh`` is the original Marmousi speed at 4 m on Madagascar's data server,
#: from which the section can be made without the paper's repository.
FILES = {
    "mt.npy": (PAPER_URL + "mt.npy", 223252,
               "b2d0fe3ce1f5f9b9f0711d4997828f78e66b960658bf80e41d4d8382deb6d912"),
    "src.rsf@": (PAPER_URL + "src.rsf%40", 16004,
                 "3265e5bd80d15cf377f32cb5ad572fda9692523f996fcc5f4bc1a406d9fa286c"),
    "marmvel.hh": ("https://www.ahay.org/data/marm/marmvel.hh", 6913301,
                   "644597dd0789f16b832c61e3f7c35fc64633e54a54ac26419970b6daa92c99b0"),
}

#: the section's length and depth, the sea bed, the least and the greatest speed
LX, LZ, ZW, CMIN, CMAX = 9200.0, 2400.0, 400.0, 1500.0, 4000.0
#: the step of the section's lattice, and its rows of water: rows 0 to 19, ``z <= 380`` m
D = 20.0
NW = 20
#: the depth down to which the speed is the water's to the bit.  Between it and the first
#: row of rock at ``ZW`` the interpolant climbs from the one to the other.
Z_WATER = ZW - D
#: the base box of the application, 9200 x 2400 x 2400 m with the paper's section in the
#: middle plane: what a ``Box`` is made with
BOX_SIZE = (LX, 2400.0, LZ)
BOX_ORIGIN = (0.0, -1200.0, 0.0)

_GOLDEN = 0.5 * (1.0 + 5.0 ** 0.5)
#: m, the stretch along ``x`` (at most 1.5 times it; ``1.5 STRETCH pi / LX = 0.15 < 1`` keeps
#: ``x -> X`` monotone) and its two wavelengths in ``y``, whose ratio is irrational: the
#: pattern never repeats across the line
STRETCH = 300.0
STRETCH_Y = (4800.0, 4800.0 * _GOLDEN)
#: m, how far the layers move up and down (``HEAVE pi / (LZ - ZW) = 0.24 < 1`` keeps
#: ``z -> Z`` monotone) and the wavelength of that in ``y``
HEAVE = 150.0
HEAVE_Y = 6000.0
#: m, the wavelength of the modulation along ``x`` that ``eps`` switches on: no multiple of
#: the mirror images' period ``2 LX``, so that no two tiles of a survey are alike
TILE_X = 2.0 * LX * _GOLDEN

#: the paper's taper ``exp(-(ll (1 - d / margin))^ex)`` (its ``window_config``)
TAPER_LL, TAPER_EX = 2.0, 3.0
#: the paper's wavelet file: 4001 samples every 2 ms, largest at 0.998 s
PAPER_NT, PAPER_DT = 4001, 0.002
#: m: two positions nearer than this are the same (a node on a face, a node on the last row
#: of water).  Far above the rounding of a position in a survey of hundreds of km, far
#: below any lattice step.
TOL = 1e-6

#: what a driver gets on a lattice (:func:`on_lattice`): arrays ``[k, j, i]`` and the nodes'
#: coordinates along each axis
Fields = collections.namedtuple("Fields", "c_true c0 window q_true x y z")


# ------------------------------------------------------------------ the data
def fetch(folder=DATA):
    """Download the three files of :data:`FILES` into ``folder`` (7.2 MB) and check each
    by its size and its SHA-256: they are other sites' files, and what follows relies on
    their being the ones this application was written for.  A file that is there already
    is checked and kept.  To be run once on a login node; compute nodes have no network."""
    import urllib.request

    # through a link: ``data`` is one to a scratch file system, and after a purge there it
    # names a folder that is gone, which ``makedirs`` of the link itself refuses to make
    os.makedirs(os.path.realpath(folder), exist_ok=True)
    for name, (url, size, sha) in FILES.items():
        path = os.path.join(folder, name)
        there = os.path.exists(path)
        if there:
            with open(path, "rb") as f:
                raw = f.read()
        else:
            with urllib.request.urlopen(url, timeout=120) as r:
                raw = r.read()
        got = hashlib.sha256(raw).hexdigest()
        if len(raw) != size or got != sha:
            raise RuntimeError("%s: %d bytes with SHA-256 %s, not the %d bytes with %s that this "
                               "application was written for"
                               % (path if there else url, len(raw), got, size, sha))
        if not there:
            # under another name until it is whole, so that a download cut short never
            # stands where the next run looks
            with open(path + ".part", "wb") as f:
                f.write(raw)
            os.replace(path + ".part", path)
    return folder


def _first_rank(make, comm):
    """What ``make()`` gives on the first rank of ``comm``, on every rank: 4096 ranks do not
    read one small file each.  An error of the first rank is raised on all of them, so that
    none is left waiting for the others."""
    if comm is None:
        return make()
    out = None
    if comm.rank == 0:
        try:
            out = make()
        except Exception as e:  # noqa: BLE001 (whatever it is, every rank has to hear of it)
            out = e
    out = comm.bcast(out, root=0)
    if isinstance(out, Exception):
        raise out
    return out


def read_marmvel(path):
    """The original Marmousi speed ``[x, z]``, 2301 x 751 values 4 m apart, from Madagascar's
    ``marmvel.hh``: a text header, then 2301 traces of 751 big-endian ``float32``."""
    with open(path, "rb") as f:
        raw = f.read()
    n = 4 * 751 * 2301
    if len(raw) < n:
        raise ValueError("%s: %d bytes, too few for 2301 x 751 float32" % (path, len(raw)))
    return np.frombuffer(raw, dtype=">f4", offset=len(raw) - n).reshape(2301, 751).astype(np.float64)


def paper_section(marmvel):
    """The paper's section from the original model: 20 rows of water over every fifth
    sample of its depths 40 to 2040 m, cut off at 4000 m/s.  Equal to ``mt.npy`` to the
    bit."""
    rock = np.clip(marmvel[::5, 10:515:5], CMIN, CMAX)
    return np.concatenate([np.full((rock.shape[0], NW), CMIN), rock], axis=1)


def _stored_section(folder):
    """The section as ``folder`` has it: the paper's file, or the original cut as the paper
    cut it."""
    path = os.path.join(folder, "mt.npy")
    if os.path.exists(path):
        V2 = np.load(path, allow_pickle=False).astype(np.float64)
    elif os.path.exists(os.path.join(folder, "marmvel.hh")):
        V2 = paper_section(read_marmvel(os.path.join(folder, "marmvel.hh")))
    else:
        raise FileNotFoundError("neither mt.npy nor marmvel.hh in %s (the folder that MARMOUSI_DATA "
                                "names, or data beside marmousi.py): python marmousi.py fetch gets them"
                                % folder)
    if V2.shape != (int(LX / D) + 1, int(LZ / D) + 1):
        raise ValueError("the section in %s is %s, not 461 x 121" % (folder, V2.shape))
    return V2


def section(folder=DATA, smooth=0.0, comm=None):
    """The paper's section ``V2[i, k]``, the speed at ``x = 20 i``, ``z = 20 k``: 461 x 121
    values in double precision, from ``mt.npy`` or, without it, rebuilt from
    ``marmvel.hh``.

    ``smooth`` (m) is the standard deviation of a Gaussian that the rock is smoothed by,
    the water kept: for a lattice coarser than the section's 20 m, whose nodes would
    otherwise take a thin layer here and miss it there, and fall by turns on a node of
    the section and midway between two, a sample and then a mean (half the lattice's step
    is enough).  With the communicator ``comm`` of a run on several ranks the first rank
    reads and the others are sent what it read.
    """
    V2 = _first_rank(lambda: _stored_section(folder), comm)
    return _smoothed(V2, smooth, True) if smooth > 0.0 else V2


def _gaussian(V, sigma, mode):
    """``V`` filtered along both axes by a Gaussian of ``sigma`` lattice steps cut at four
    of them, the array continued by ``numpy.pad``'s ``mode``: what
    ``scipy.ndimage.gaussian_filter`` computes (``edge`` is its ``nearest``, ``symmetric``
    its ``reflect``)."""
    r = int(4.0 * sigma + 0.5)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    out = np.asarray(V, dtype=np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        P = np.pad(out, pad, mode=mode)
        n = out.shape[axis]
        out = sum(w[j] * np.take(P, np.arange(j, j + n), axis=axis) for j in range(2 * r + 1))
    return out


def _smoothed(V2, sigma, keep_water):
    """The section smoothed by a Gaussian of ``sigma`` metres: the rock alone, mirrored at
    the sea bed, or all of it with its edge values continued."""
    V2 = np.asarray(V2, dtype=np.float64)
    if sigma <= 0.0:
        # no smoothing: a Gaussian of no width has no weights to divide by
        return V2.copy()
    if keep_water:
        out = V2.copy()
        out[:, NW:] = _gaussian(V2[:, NW:], sigma / D, "symmetric")
    else:
        out = _gaussian(V2, sigma / D, "edge")
    # a weighted mean lies between the least and the greatest of what it averages; the
    # rounding of the sums does not know that, and the time step counts on the greatest
    return np.clip(out, V2.min(), V2.max())


def initial_section(V2, sigma=120.0, keep_water=True):
    """The section smoothed by a Gaussian of ``sigma`` metres: the initial model.

    With ``keep_water`` the rock alone is smoothed, mirrored at the sea bed, and the water
    stays 1500 m/s to the bit: the initial model of the paper's runs with the water fixed
    (its file ``m02.npy``).  Without, the whole section is smoothed with its edge values
    continued, as the paper's inversion does, which carries the sea bed into the water
    (a mean of 1597 m/s at 380 m of depth).
    """
    return _smoothed(V2, sigma, keep_water)


# ------------------------------------------------------------------ the speed
def _warp(xp, x, y, z, eps):
    """Where in the section the point ``(x, y, z)`` takes its speed from: ``(X, Z)``."""
    s = LX - xp.abs(xp.mod(x, 2.0 * LX) - LX)
    m = 1.0 + eps * xp.cos(2.0 * np.pi * x / TILE_X)
    a = STRETCH * m * (xp.sin(2.0 * np.pi * y / STRETCH_Y[0]) + 0.5 * xp.sin(2.0 * np.pi * y / STRETCH_Y[1]))
    b = HEAVE * m * xp.sin(2.0 * np.pi * y / HEAVE_Y) * xp.cos(np.pi * s / LX)
    t = xp.clip((z - ZW) / (LZ - ZW), 0.0, 1.0)
    return s + a * xp.sin(np.pi * s / LX), z + b * xp.sin(np.pi * t) ** 2


def _bilinear(xp, V, X, Z, least, greatest):
    """The bilinear interpolant of the section's lattice ``V`` at ``(X, Z)``; outside the
    section the nearest value of its edge."""
    nx, nz = V.shape
    fx = xp.clip(X / D, 0.0, nx - 1.0)
    fz = xp.clip(Z / D, 0.0, nz - 1.0)
    i = xp.minimum(xp.floor(fx), nx - 2.0).astype(xp.int32)
    k = xp.minimum(xp.floor(fz), nz - 2.0).astype(xp.int32)
    s, t = fx - i, fz - k
    # by differences: four equal corners give their value to the bit (the water stays 1500.0),
    # and a point on a lattice node takes the node's
    v0 = V[i, k] + s * (V[i + 1, k] - V[i, k])
    v1 = V[i, k + 1] + s * (V[i + 1, k + 1] - V[i, k + 1])
    # a mean of the four corners, which rounding can put an ulp outside them: the time step
    # is set by the greatest speed of the section and must not meet a greater one
    return xp.clip(v0 + t * (v1 - v0), least, greatest)


def speed(V2, x, y, z, eps=0.0):
    """The three-dimensional speed made of the section ``V2`` at the points ``(x, y, z)``
    (arrays that broadcast), by the formula of the module's text; ``z`` is the depth under
    the free surface.

    It is defined at every point: beyond the ends of the section along ``x`` its mirror
    images follow each other, below ``LZ`` the bottom row continues, and ``y`` has no end.
    ``eps`` modulates the map along ``x`` so that the tiles of a large survey differ.
    """
    V2 = np.asarray(V2, dtype=np.float64)
    x, y, z = (np.asarray(c, dtype=np.float64) for c in (x, y, z))
    X, Z = _warp(np, x, y, z, eps)
    return _bilinear(np, V2, X, Z, V2.min(), V2.max())


def _jnp():
    """JAX's NumPy as the library sets it up: on its device and in double precision, which
    JAX by itself is not."""
    from hippymfem.fem import kernel  # noqa: F401
    import jax.numpy as jnp

    return jnp


def speed_fn(V2, eps=0.0, dtype=None):
    """:func:`speed` as a JAX function of positions ``(..., 3)``: the speed at a block's
    nodes computed on the device, from the 56 thousand values of the section kept there.
    ``dtype`` is the precision of the result (double by default)."""
    jnp = _jnp()
    V = jnp.asarray(np.asarray(V2, dtype=np.float64))
    least, greatest = float(np.min(V2)), float(np.max(V2))

    def fn(X):
        Xs, Zs = _warp(jnp, X[..., 0], X[..., 1], X[..., 2], eps)
        v = _bilinear(jnp, V, Xs, Zs, least, greatest)
        return v if dtype is None else v.astype(dtype)

    return fn


# ------------------------------------------------------------------ the window
def _ends(box):
    lo = np.asarray(box.origin, dtype=np.float64).reshape(3)
    return lo, lo + np.asarray(box.size, dtype=np.float64).reshape(3)


def _taper(xp, d, margin):
    """The window's factor of one face at the distance ``d`` inside it (negative outside
    the box): zero on the face and outside, the paper's exponential within ``margin``, one
    beyond.  A margin of zero is no taper: one up to the face and on it."""
    if margin <= 0.0:
        return xp.where(d > -TOL, 1.0, 0.0)
    t = TAPER_LL * xp.maximum(1.0 - d / margin, 0.0)
    return xp.where(d > TOL, xp.exp(-t ** TAPER_EX), 0.0)


def _window(xp, x, y, z, lo, hi, margins, water):
    mx, my, mz = (float(m) for m in margins)
    w = (_taper(xp, x - lo[0], mx) * _taper(xp, hi[0] - x, mx)
         * _taper(xp, y - lo[1], my) * _taper(xp, hi[1] - y, my) * _taper(xp, hi[2] - z, mz))
    return w * (xp.where(z > Z_WATER + TOL, 1.0, 0.0) if water else _taper(xp, z - lo[2], ZW))


def window(x, y, z, box, margins=(600.0, 600.0, 400.0), water=True):
    """The paper's window ``W`` at the points ``(x, y, z)`` (arrays that broadcast) of the
    box ``box`` (anything with ``size`` and ``origin``): what multiplies the unknown, so
    that the speed is the initial model's wherever ``W`` is zero.

    A product of one factor a face.  At the distance ``d`` inside a side (``margins[0]``
    for the two faces across ``x``, ``margins[1]`` across ``y``) or above the bottom
    (``margins[2]``) the factor is ``exp(-(2 (1 - d / margin))^3)`` for ``d < margin`` and
    1 beyond: 0.093, 0.37, 0.74, 0.96 at a third, a half, two thirds, five sixths of the
    margin, and exactly 1 from the margin inwards.  It is 0 on the face itself (the
    paper zeroes its boundary lines; here the absorbing faces keep the speed their
    condition was made for) and outside the box.  A margin of 0 is no taper at those
    faces.

    At the top, with ``water`` (the default), ``W`` is 0 where the speed is the water's
    to the bit, ``z <= ZW - 20 = 380`` m, and has no taper below: at the sea bed
    ``z = ZW``, the first row of rock, it is 1, as in the paper's runs with the water
    fixed, whose unknowns are the rows from 400 m down.  Between 380 and 400 m, where the
    speed climbs from the water's to the rock's, it is 1 as well, so that the true speed
    equals the initial one wherever the window is zero at the top.  Without ``water`` the
    top is tapered like a side over the 400 m of the water, the window of the paper's
    inversion.

    On the section's own lattice this is the paper's window to rounding, the defaults
    being its margins of 30, 30 and 20 points.
    """
    lo, hi = _ends(box)
    x, y, z = (np.asarray(c, dtype=np.float64) for c in (x, y, z))
    return _window(np, x, y, z, lo, hi, margins, water)


def window_fn(box, margins=(600.0, 600.0, 400.0), water=True, dtype=None):
    """:func:`window` as a JAX function of positions ``(..., 3)``."""
    jnp = _jnp()
    lo, hi = _ends(box)

    def fn(X):
        w = _window(jnp, X[..., 0], X[..., 1], X[..., 2], lo, hi, margins, water)
        return w if dtype is None else w.astype(dtype)

    return fn


# ------------------------------------------------------------------ on a lattice
def nodes(box, h, block=None):
    """The coordinates ``(x, y, z)`` of the nodes of the lattice of step ``h`` (one number,
    or one a direction, ``(hx, hy, hz)``) in ``box``, one array a direction.

    ``block`` is a rank's part of the lattice as ``lattice.py`` names it, in the order of
    the arrays' axes: ``((lo, hi), (lo, hi), (lo, hi))`` in node numbers along ``z, y, x``,
    ``hi`` excluded, which is what ``Lattice.closed(B)``, ``Lattice.owned(B)`` and
    ``Halo.box`` are.  It may reach outside the box (a halo).  Without it, all the nodes.
    """
    lo, hi = _ends(box)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), (3,))
    n = np.rint((hi - lo) / h).astype(np.int64)
    if np.any(np.abs(n * h - (hi - lo)) > TOL):
        raise ValueError("the box %s is not a whole number of steps of %s m"
                         % ((hi - lo).tolist(), h.tolist()))
    if block is None:
        spans = [(0, int(g) + 1) for g in n]
    else:
        if len(block) != 3:
            raise ValueError("a block is three ranges of nodes, along z, y and x: %r" % (block,))
        spans = [block[2], block[1], block[0]]               # x, y, z of a box of nodes (z, y, x)
    return tuple(lo[d] + h[d] * np.arange(a, b) for d, (a, b) in enumerate(spans))


def on_lattice(box, h, block=None, folder=DATA, comm=None, eps=0.0, smooth=None, sigma=120.0,
               margins=(600.0, 600.0, 400.0), water=True, dtype=np.float64):
    """What a driver sets a problem up with, at the nodes of the lattice of step ``h`` in
    ``box`` or of a rank's ``block`` of it (see :func:`nodes`): a :data:`Fields` of

    * ``c_true``, the true speed: the section (smoothed over ``smooth`` metres, by default
      half the step if the lattice is coarser than the section's 20 m, see :func:`section`)
      in three dimensions;
    * ``c0``, the initial speed: the section smoothed over ``sigma`` metres, the water
      kept, through the same map;
    * ``window``, the window ``W`` (:func:`window`);
    * ``q_true = c_true - c0``, the true offset of the speed from the initial model;

    arrays ``[k, j, i]`` of ``dtype``, and the nodes' coordinates ``x``, ``y``, ``z``.
    Every value is a function of the node's position alone: the blocks of the ranks are
    parts of one field, and nothing is exchanged to make them.  With ``comm`` the call is
    collective (the first rank reads the section for all).

    ``q_true`` is zero wherever the window is zero in the water.  On the sides and the
    bottom, where the window is zero as well, it is not, as in the paper, whose true model
    differs from the initial one on its boundary lines: data made with ``q_true`` hold a
    part that no unknown ``m`` of ``c = c0 + W m`` gives back, and data made with ``W
    q_true`` are those of a model the unknown can reach (``m = q_true``).
    """
    V2 = section(folder, comm=comm)
    step = float(np.max(h))
    smooth = (0.5 * step if step > D else 0.0) if smooth is None else smooth
    x, y, z = nodes(box, h, block)
    X, Y, Z = x[None, None, :], y[None, :, None], z[:, None, None]
    c_true = speed(_smoothed(V2, smooth, True) if smooth > 0.0 else V2, X, Y, Z, eps)
    c0 = speed(initial_section(V2, sigma), X, Y, Z, eps)
    W = window(X, Y, Z, box, margins, water)
    return Fields(c_true.astype(dtype), c0.astype(dtype), W.astype(dtype), (c_true - c0).astype(dtype),
                  x, y, z)


def _trilinear(xp, C, first, step, X):
    """The trilinear interpolant of the lattice array ``C[k, j, i]``, whose first node is at
    ``first`` and whose step is ``step``, at positions ``X (..., 3)``; outside the array
    the nearest value of its faces."""
    f = [xp.clip((X[..., d] - first[d]) / step[d], 0.0, C.shape[2 - d] - 1.0) for d in range(3)]
    i, j, k = (xp.minimum(xp.floor(f[d]), C.shape[2 - d] - 2.0).astype(xp.int32) for d in range(3))
    rx, ry, rz = f[0] - i, f[1] - j, f[2] - k

    def along_x(k, j):
        return C[k, j, i] + rx * (C[k, j, i + 1] - C[k, j, i])

    def along_xy(k):
        low = along_x(k, j)
        return low + ry * (along_x(k, j + 1) - low)

    low = along_xy(k)
    return low + rz * (along_xy(k + 1) - low)


def lattice_aux(c0, box, h, block=None, dtype=None):
    """The initial speed for the propagator's ``aux``: a JAX function of a block's node
    positions ``(..., 3)`` that returns ``(c0 there,)``, the trilinear interpolant of the
    lattice array ``c0[k, j, i]`` of :func:`on_lattice` (of the same ``box``, ``h`` and
    ``block``).  The block has to hold the nodes of all the rank's elements, which
    ``Lattice.closed(B)`` does and ``Lattice.owned(B)`` does not: a position outside the
    array takes the nearest value of its faces, without a word.

    The parameter is trilinear on the elements, so with this the speed of the wave
    equation, ``c0 + q``, is at every node of every element the trilinear interpolant of
    the lattice's speeds: within their range, and the water's in every element of water.
    The formula itself, ``speed_fn(initial_section(V2), eps)``, has features of 20 m that
    a coarser lattice of offsets cannot follow: with it and the lattice of 50 m the true
    speed ``c0 + q_true`` goes down to 1413 m/s in the elements on the sea bed and up to
    4001 m/s.
    """
    jnp = _jnp()
    x, y, z = nodes(box, h, block)
    if np.shape(c0) != (z.size, y.size, x.size) or min(np.shape(c0)) < 2:
        raise ValueError("the lattice array is %s and the block has %d x %d x %d nodes (z, y, x); a cell "
                         "to interpolate in takes two a direction" % (np.shape(c0), z.size, y.size, x.size))
    first = (float(x[0]), float(y[0]), float(z[0]))
    step = tuple(float(s) for s in np.broadcast_to(np.asarray(h, dtype=np.float64), (3,)))
    C = jnp.asarray(np.asarray(c0, dtype=np.float64))

    def aux(X):
        v = _trilinear(jnp, C, first, step, X)
        return (v if dtype is None else v.astype(dtype),)

    return aux


# ------------------------------------------------------------------ the source in time
def resample(w, dt, t, points=8):
    """The samples ``w`` taken every ``dt`` from time 0, at the times ``t``: the polynomial
    through the ``points`` samples around each time, and zero outside the samples' span.
    For a signal of at most 12 Hz sampled every 2 ms eight points are right to 1e-9 of its
    size; at a sample's own time the sample is returned to the bit."""
    w = np.asarray(w, dtype=np.float64)
    if w.size < points:
        raise ValueError("%d samples are too few for a polynomial through %d" % (w.size, points))
    u = np.asarray(t, dtype=np.float64) / dt
    # ``k dt / dt`` is not always ``k``: a time within rounding of a sample's is that sample's
    u = np.where(np.abs(u - np.rint(u)) < 1e-9, np.rint(u), u)
    first = np.clip(np.floor(u).astype(np.int64) - (points // 2 - 1), 0, w.size - points)
    r = u - first
    out = np.zeros(u.shape)
    for a in range(points):
        lagrange = np.ones(u.shape)
        for b in range(points):
            if b != a:
                lagrange *= (r - b) / (a - b)
        out += lagrange * w[first + a]
    return np.where((u >= 0.0) & (u <= w.size - 1.0), out, 0.0)


def paper_wavelet(folder=DATA, comm=None):
    """The paper's source time function, the 4001 samples of ``src.rsf@`` every
    :data:`PAPER_DT`: the raw part of a Madagascar file whose header the paper's
    repository does not have, little-endian ``float32``.  A band-pass flat from 5 to 10 Hz
    with half its amplitude at 4.5 and 11 Hz and 1 % of it at 12 Hz, even about its
    largest sample, 0.0259 at 0.998 s."""
    def read():
        path = os.path.join(folder, "src.rsf@")
        if not os.path.exists(path):
            raise FileNotFoundError("no src.rsf@ in %s: python marmousi.py fetch gets it" % folder)
        w = np.fromfile(path, dtype="<f4").astype(np.float64)
        if w.size != PAPER_NT:
            raise ValueError("%s holds %d samples, not %d" % (path, w.size, PAPER_NT))
        return w

    return _first_rank(read, comm)


def wavelet(t, kind="compact", fc=7.0, bw=1.5, t0=0.5, folder=DATA, comm=None):
    """The source's time function at the times ``t``.

    ``compact`` (the default) is ``exp(-(t - t0)^2 / (2 st^2)) cos(2 pi fc (t - t0))`` with
    ``st = 1 / (2 pi bw)``: its spectrum is a Gaussian of standard deviation ``bw`` about
    ``fc``, so with 7 and 1.5 Hz it has 13.5 % of its peak at the ends of the paper's band,
    4 and 10 Hz, and 1 % at 11.6 Hz, which is what the mesh has to carry.  It is below
    1e-3 of its largest value, 1, outside ``t0 -+ 0.38`` s, so a record may start at 0
    with ``t0 = 0.5``.

    ``paper`` is the paper's own (:func:`paper_wavelet`) at the times ``t``
    (:func:`resample`), with its amplitude and its delay of 1 s: it is still 1e-3 of its
    peak at 2.6 s, and a record has to be a second longer for the same arrivals.  ``fc``,
    ``bw`` and ``t0`` do not apply to it.
    """
    t = np.asarray(t, dtype=np.float64)
    if kind == "compact":
        tau = t - t0
        return np.exp(-0.5 * (2.0 * np.pi * bw * tau) ** 2) * np.cos(2.0 * np.pi * fc * tau)
    if kind == "paper":
        return resample(paper_wavelet(folder, comm), PAPER_DT, t)
    raise ValueError("wavelet kind %r: compact or paper" % (kind,))


# ------------------------------------------------------------------ the survey
def _line(length, step):
    """Points ``step`` apart, centred in ``[0, length]``, as many as stay a quarter of a
    step from its ends (one in the middle of a length too short for that)."""
    n = max(int(np.floor((length - 0.5 * step) / step + 1e-9)) + 1, 1)
    return 0.5 * (length - (n - 1) * step) + step * np.arange(n)


def survey(box, ds=400.0, dr=100.0, depth=25.0, tile=None, only=None):
    """Sources ``(ns, 3)`` and receivers ``(nr, 3)`` of the box ``box`` (anything with
    ``size`` and ``origin``): grids of step ``ds`` and ``dr`` in plan, centred in the box,
    ``depth`` under the free surface.  The base box gets 23 x 6 = 138 sources and 92 x 24
    = 2208 receivers.

    The depth is not zero because the field is zero on the free surface, where a source
    radiates nothing: at 25 m the surface's mirror image multiplies what goes down by 0.81
    at 4 Hz and 1.73 at 10 Hz.  With the default steps no receiver stands on a source
    (sources at odd multiples of 200 m from the box's corner, receivers of 50 m).

    ``tile = (tx, ty)`` cuts the box into ``tx`` by ``ty`` equal tiles in plan and gives
    each the same survey, for shots that fire one source in every tile at once: a survey
    64 times as wide then costs a rank no more fields.  It returns ``(sources, receivers,
    tile_of, place)``: for every source the tile it is in, ``I + tx J`` (the rank of that
    column in a grid ``(tx, ty, 1)`` of blocks), and its place in the tile, the same number
    for the sources that stand at the same point of their tiles.  Shot ``k`` is the
    sources with ``place == k``.  Sources are ordered by tile, then by place; in a tile
    ``x`` runs fastest.

    ``only`` (tile numbers) gives the sources and the receivers of those tiles alone, in
    the order asked: a rank's own, of a survey whose whole no rank should make (4096 tiles
    of 4.8 km have 9.4 million receivers).  Every tile has the same number ``K`` of sources
    (``place`` runs from 0 to ``K - 1``) and of receivers, so a source's number in the
    whole survey is ``K tile_of + place``, and a tile's receivers are numbered likewise.
    """
    lo, hi = _ends(box)
    tx, ty = (1, 1) if tile is None else (int(tile[0]), int(tile[1]))
    side = (hi[:2] - lo[:2]) / (tx, ty)
    which = np.arange(tx * ty) if only is None else np.atleast_1d(np.asarray(only, dtype=np.int64))
    if which.size and (which.min() < 0 or which.max() >= tx * ty):
        raise ValueError("tiles %s of a survey of %d x %d" % (which.tolist(), tx, ty))
    corners = lo[:2] + side * np.stack([which % tx, which // tx], axis=1)

    def grid(step):
        gx, gy = np.meshgrid(_line(side[0], step), _line(side[1], step))
        one = np.stack([gx.ravel(), gy.ravel()], axis=1)
        xy = (corners[:, None, :] + one[None, :, :]).reshape(-1, 2)
        return np.concatenate([xy, np.full((xy.shape[0], 1), lo[2] + depth)], axis=1), one.shape[0]

    sources, shots = grid(ds)
    receivers, _ = grid(dr)
    if tile is None:
        return sources, receivers
    return sources, receivers, np.repeat(which, shots), np.tile(np.arange(shots), which.size)


# ------------------------------------------------------------------ figures
def figures(out, folder=DATA):
    """Write ``model.png``, ``window.png``, ``wavelet.png`` and ``survey.png`` into ``out``
    and return their paths: sections and maps of the speed and the initial model, the
    window, the two wavelets with their spectra, the survey of the base box and one of
    tiles.  A speed and a window are magnitudes and take one hue from light to dark;
    lines and points take the first hues of a fixed order; text and axes are ink and gray.
    """
    import types

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, ink2, muted, axis, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
    blue, orange = "#2a78d6", "#eb6834"
    ramp = LinearSegmentedColormap.from_list("blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                      "#256abf", "#184f95", "#0d366b", "#06203f"])
    style = {
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": muted,
        "ytick.color": muted, "xtick.labelcolor": ink2, "ytick.labelcolor": ink2, "text.color": ink,
        "axes.titlecolor": ink, "axes.titlelocation": "left", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "legend.fontsize": 8.5, "grid.color": grid, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
    os.makedirs(out, exist_ok=True)
    V2 = section(folder)
    V0 = initial_section(V2)
    base = types.SimpleNamespace(size=BOX_SIZE, origin=BOX_ORIGIN)
    x = np.arange(0.0, LX + 1.0, D)
    z = np.arange(0.0, LZ + 1.0, D)
    km = 1e-3
    left, width = 0.75, 12.5                                  # inches: the left margin, the page
    paths = []

    def canvas(height, title):
        """A page and the function that puts axes on it by inches from its left and its top:
        boxes cut to the proportions of what they show line up, which a grid of cells
        that each centre an image of another shape does not."""
        fig = plt.figure(figsize=(width, height))
        fig.suptitle(title, x=left / width, y=1.0 - 0.3 / height, ha="left", va="top", fontsize=11)

        def place(x0, top, w, h):
            return fig.add_axes([x0 / width, 1.0 - (top + h) / height, w / width, h / height])

        return fig, place

    def save(fig, name):
        paths.append(os.path.join(out, name))
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)

    def image(ax, values, extent, title, xlabel=None, ylabel=None, yticks=None, **kw):
        im = ax.imshow(values, extent=extent, cmap=ramp, aspect="auto", interpolation="nearest", **kw)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if yticks is not None:
            ax.set_yticks(yticks)
        return im

    def model_figure():
        a = 6.5                                               # a section along the line, 9.2 x 2.4 km
        ha = a * LZ / LX
        b0, b = left + a + 0.8, 2.0 * ha                      # one across it, 4.8 x 2.4 km
        m = 0.5 * (a + b)                                     # a map, 9.2 x 4.8 km
        hm = m * 4800.0 / LX
        tops = [0.95 + (ha + 0.62) * r for r in range(3)]
        top_maps = tops[2] + ha + 0.95
        top_last = top_maps + hm + 0.95
        fig, place = canvas(top_last + ha + 0.65, "The Marmousi section in three dimensions: stretched along "
                            "x and its layers moved up and down, as functions of y")
        y = np.arange(-2400.0, 2400.0 + 1.0, D)
        depth = [0, 0.4, 1.2, 2.4]
        kw = dict(vmin=CMIN, vmax=CMAX)
        for r, yy in enumerate((-1200.0, 0.0, 1200.0)):
            im = image(place(left, tops[r], a, ha), speed(V2, x[:, None], yy, z[None, :]).T,
                       [0, LX * km, LZ * km, 0], "along the line at y = %+.1f km" % (yy * km) if yy else
                       "along the line at y = 0: the paper's section", "x (km)" if r == 2 else None,
                       "depth (km)", depth, **kw)
        for r, xx in enumerate((3000.0, 4600.0, 6200.0)):
            ax = place(b0, tops[r], b, ha)
            image(ax, speed(V2, xx, y[:, None], z[None, :]).T, [-2.4, 2.4, LZ * km, 0],
                  "across it at x = %.1f km" % (xx * km), "y (km)" if r == 2 else None, None, depth, **kw)
            ax.set_xticks([-2.4, -1.2, 0, 1.2, 2.4])
        for c, zz in enumerate((800.0, 1600.0)):
            ax = place(left + c * (m + 0.8), top_maps, m, hm)
            note = "; between the lines, the base box" if c == 0 else ""
            image(ax, speed(V2, x[:, None], y[None, :], zz).T, [0, LX * km, -2.4, 2.4],
                  "map at %.1f km of depth%s" % (zz * km, note), "x (km)", "y (km)" if c == 0 else None,
                  [-2.4, -1.2, 0, 1.2, 2.4], origin="lower", **kw)
            for yy in (-1.2, 1.2):
                ax.axhline(yy, color=surface, lw=0.8)
        image(place(left, top_last, a, ha), speed(V0, x[:, None], 0.0, z[None, :]).T,
              [0, LX * km, LZ * km, 0], "the initial model at y = 0: the section smoothed over 120 m under "
              "the sea bed", "x (km)", "depth (km)", depth, **kw)
        ax = place(b0, top_last, b, ha)
        yb = np.arange(-1200.0, 1200.0 + 1.0, 100.0)
        v = speed(V2, x[:, None, None], yb[None, :, None], z[None, None, :])
        ax.fill_betweenx(z * km, v.min(axis=(0, 1)), v.max(axis=(0, 1)), color=blue, alpha=0.12, lw=0)
        ax.plot(v.mean(axis=(0, 1)), z * km, color=blue)
        ax.set_ylim(LZ * km, 0)
        ax.set_xlim(1400, 4100)
        ax.set_yticks(depth)
        ax.set_title("mean and range over the base box")
        ax.set_xlabel("speed (m/s)")
        ax.grid(True)
        cb = fig.colorbar(im, cax=place(b0 + b + 0.22, tops[0], 0.14, tops[2] + ha - tops[0]))
        cb.set_label("wave speed (m/s)")
        cb.outline.set_edgecolor(axis)
        save(fig, "model.png")

    def window_figure():
        a = 7.4
        ha = a * LZ / LX
        b0, b = left + a + 1.4, width - (left + a + 1.4) - 0.3
        tops = [0.95, 0.95 + ha + 1.05]
        fig, place = canvas(tops[1] + ha + 0.65, "The window that multiplies the unknown: "
                            "exp(-(2 (1 - d / margin))^3) within 600 m of the sides and 400 m of the bottom")
        yb = np.arange(-1200.0, 1200.0 + 1.0, D)
        im = image(place(left, tops[0], a, ha), window(x[:, None], 0.0, z[None, :], base).T,
                   [0, LX * km, LZ * km, 0], "along the line at y = 0: zero in the water, one from the sea "
                   "bed down", "x (km)", "depth (km)", [0, 0.4, 1.2, 2.0, 2.4], vmin=0.0, vmax=1.0)
        image(place(left, tops[1], a, ha), window(x[:, None], yb[None, :], 1200.0, base).T,
              [0, LX * km, -1.2, 1.2], "map at 1.2 km of depth: one where the sides are more than 600 m "
              "away", "x (km)", "y (km)", [-1.2, -0.6, 0, 0.6, 1.2], origin="lower", vmin=0.0, vmax=1.0)
        cb = fig.colorbar(im, cax=place(left + a + 0.2, tops[0], 0.14, tops[1] + ha - tops[0]))
        cb.outline.set_edgecolor(axis)
        ax = place(b0, tops[0], b, ha)
        d = np.linspace(0.0, 800.0, 401)
        marks = np.array([200.0, 300.0, 400.0, 500.0])
        ax.plot(d, window(d, 0.0, 1200.0, base), color=blue)
        ax.plot(marks, window(marks, 0.0, 1200.0, base), "o", color=blue, ms=7, mec=surface, mew=1.6)
        for dm, wm in zip(marks, window(marks, 0.0, 1200.0, base)):
            ax.annotate("%.2g" % wm, (dm, wm), textcoords="offset points", xytext=(8, -5), color=ink2,
                        fontsize=8)
        ax.set_title("against the distance from a side")
        ax.set_xlabel("m; marked, the paper's nodes 10, 15, 20, 25")
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax = place(b0, tops[1], b, ha)
        fine = np.arange(0.0, LZ + 0.5, 1.0)
        ax.plot(window(4600.0, 0.0, fine, base, water=False), fine * km, color=orange,
                label="tapered over the water")
        ax.plot(window(4600.0, 0.0, fine, base), fine * km, color=blue, label="water frozen (the default)")
        ax.set_ylim(LZ * km, 0)
        ax.set_xlim(0, 1.05)
        ax.set_yticks([0, 0.4, 1.2, 2.0, 2.4])
        ax.set_title("against depth in the middle of the box")
        ax.set_xlabel("window")
        ax.set_ylabel("depth (km)")
        ax.legend(loc="center left", bbox_to_anchor=(0.03, 0.55), fontsize=8)
        ax.grid(True)
        save(fig, "window.png")

    def wavelet_figure():
        dt = 0.001
        t = dt * np.arange(8001)
        height = 3.6
        fig, place = canvas(height, "The source's time function")
        axes = [place(left, 0.95, 4.0, 2.05), place(left + 4.75, 0.95, 2.9, 2.05),
                place(left + 8.4, 0.95, 2.9, 2.05)]
        for label, w, color in (("compact: 7 Hz under a Gaussian, centred at 0.5 s", wavelet(t), blue),
                                ("the paper's file, centred at 1.0 s", wavelet(t, "paper", folder=folder),
                                 orange)):
            f = np.fft.rfftfreq(1 << 17, dt)
            W = np.abs(np.fft.rfft(w, 1 << 17))
            axes[0].plot(t, w / np.abs(w).max(), color=color, label=label)
            axes[1].plot(f, W / W.max(), color=color)
            axes[2].semilogy(f, W / W.max(), color=color)
        axes[0].set_xlim(0, 2.6)
        axes[0].set_title("in time, over its largest value")
        axes[0].set_xlabel("time (s)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2,
                   bbox_to_anchor=(1.0 - 0.4 / width, 1.0 - 0.2 / height))
        axes[1].set_title("spectrum over its peak; marked, 4 and 10 Hz")
        axes[2].set_title("the same, logarithmic")
        axes[2].set_ylim(1e-5, 2)
        for ax in axes[1:]:
            ax.set_xlim(0, 16)
            ax.set_xlabel("frequency (Hz)")
            for fq in (4.0, 10.0):
                ax.axvline(fq, color=muted, lw=1.0, zorder=1)
        for ax in axes:
            ax.grid(True)
        save(fig, "wavelet.png")

    def survey_figure():
        a = width - left - 0.3
        ha = a * 2400.0 / LX
        side = 6.4
        fig, place = canvas(0.95 + ha + 1.0 + side + 0.65, "The survey in plan")
        ax = place(left, 0.95, a, ha)
        src, rec = survey(base)
        ax.fill([0.6, 8.6, 8.6, 0.6], [-0.6, -0.6, 0.6, 0.6], color=blue, alpha=0.1, lw=0,
                label="where the window is one, from 0.4 to 2.0 km of depth")
        ax.scatter(rec[:, 0] * km, rec[:, 1] * km, s=2.5, color=muted, lw=0, label="receivers")
        ax.scatter(src[:, 0] * km, src[:, 1] * km, s=40, color=blue, edgecolor=surface, lw=1.4, zorder=3,
                   label="sources")
        ax.set_title("the base box, 9.2 x 2.4 km: %d sources every 400 m, %d receivers every 100 m, 25 m deep"
                     % (src.shape[0], rec.shape[0]))
        ax.set_xlim(0, LX * km)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1.2, -0.6, 0, 0.6, 1.2])
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        handles, labels = ax.get_legend_handles_labels()
        ax = place(left, 0.95 + ha + 1.0, side, side)
        tiles = types.SimpleNamespace(size=(9600.0, 9600.0, LZ), origin=(0.0, -4800.0, 0.0))
        src, rec, tile_of, place_of = survey(tiles, ds=1200.0, dr=200.0, tile=(2, 2))
        shot = place_of == 5
        ax.scatter(rec[:, 0] * km, rec[:, 1] * km, s=2.5, color=muted, lw=0)
        ax.scatter(src[~shot, 0] * km, src[~shot, 1] * km, s=40, color=blue, edgecolor=surface, lw=1.4,
                   zorder=3)
        ax.scatter(src[shot, 0] * km, src[shot, 1] * km, s=80, color=orange, edgecolor=surface, lw=1.6,
                   zorder=4, label="one shot of a survey of tiles:\nthe sources at one place of every tile")
        ax.axvline(4.8, color=axis, lw=0.8, zorder=0)
        ax.axhline(0.0, color=axis, lw=0.8, zorder=0)
        ax.set_title("2 x 2 tiles of 4.8 km: %d sources every 1200 m in %d shots of %d, %d receivers every "
                     "200 m" % (src.shape[0], place_of.max() + 1, tile_of.max() + 1, rec.shape[0]))
        ax.set_xlim(0, 9.6)
        ax.set_ylim(-4.8, 4.8)
        ax.set_xticks([0, 2.4, 4.8, 7.2, 9.6])
        ax.set_yticks([-4.8, -2.4, 0, 2.4, 4.8])
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        more, more_labels = ax.get_legend_handles_labels()
        ax.legend(handles + more, labels + more_labels, loc="upper left", bbox_to_anchor=(1.06, 1.0),
                  labelspacing=1.0)
        save(fig, "survey.png")

    with plt.rc_context(style):
        for make in (model_figure, window_figure, wavelet_figure, survey_figure):
            make()
    return paths


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        print(fetch())
    elif len(sys.argv) == 3 and sys.argv[1] == "figures":
        print("\n".join(figures(sys.argv[2])))
    else:
        sys.exit("usage: python marmousi.py fetch | figures OUT")

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""The misfit Hessian as a point spread function operator on the parameter's lattice.

A column of the Gauss-Newton Hessian of a seismic inverse problem is the image of a point:
an oscillating blob a few wavelengths wide whose shape changes slowly from place to place.
The point spread function method (Hu, Alger, Nammour and Ghattas, arXiv 2507.10804; Alger,
Hartland, Petra and Ghattas, SIAM J. Sci. Comput. 2024) reads such columns off products
with combs of unit vectors, as ``applications/icesheet/localhessian.py`` does, but at a
sparse lattice of sample points, and interpolates between them.  With ``W`` the known taper
of the inverse problem (``H = W H0 W``, ones if there is none):

.. code-block:: text

    p_k(y) = (H0 comb_c(k))(x_k + y) win(y),  |y| <= rho      impulse response at sample point x_k
    s_k(xi) = max(Re FFT p_k, 0)                              its symbol row, real and positive part
    g_k = cut(IFFT sqrt(s_k))                                 an even kernel of radius rho_g
    S u = sum_k w_k ((g_k + eta G) * u)                       w_k: weights that sum to one
    H~  = W S S^T W                                           symmetric positive semi-definite

What is the paper's and what is not:

* The combs, the mask ``win``, the real positive part and the square root are the reference
  code's.  One product with the Hessian gives every sample point's row when the teeth are
  ``2 rho`` apart (``teeth = 1``); ``teeth^d`` products give a lattice ``teeth`` times
  finer than the window allows in one.
* The reference applies its operator with transforms of the whole grid, one a sample
  point.  Here a rank transforms boxes of its own: a box for every lattice CELL, the cell
  and a margin of the kernel's radius, in which the ``2^d`` weights of the cell's corners
  are the only ones that are not zero.  ``S u`` on a cell is one forward transform of the
  box of ``u``, a product with each corner's symbol and an inverse transform; ``S^T v`` is
  the same backwards with an overlap-add.  The circular convolution of a box is exact on
  the cell, because the box holds every value a kernel reaches from it; so the operator is
  defined without reference to boxes or ranks, a rank needs its neighbours' values within
  the kernel's radius and nothing else, and its transpose is exact.  The boxes of every
  ``ncol``-th cell tile the field: a batch of them is a slice and a reshape of it.
* The weights are the tensor hats of the sample lattice.  The reference's are Gaussians
  normalized to sum to one, which are not compact; quadratic B-splines are within 0.02 of
  those and are the option ``weights="bspline"``, at 27 symbols a cell in the place of 8.
  On the synthetic Hessian of the tests the hats are the more accurate: 3.5 % against
  6.4 % with the B-splines.
* **The taper is applied, not interpolated**: the combs go through ``H0``, and ``W`` stands
  on both sides of ``S S^T``.  A taper that changes within a lattice step is not the
  interpolant of its values at the sample points: with the combs through ``W H0 W`` the
  error on that Hessian is 60 %.  Where ``W`` is zero on every node of the outermost cells
  of a rank, ``H~`` does not transform their boxes: their part of ``S^T (W u)`` is a
  transform of zeros and their part of ``W S`` a product with zeros.  A rank that owns
  both end planes of its block (the first of a direction) has a layer of cells more than
  the others, which is one plane of nodes on a face of the lattice: with a taper that
  vanishes there every rank transforms as many boxes.
* The reference adds a low-frequency term to the square root, ``S = A^ + 0.1 G`` with ``G``
  a Gaussian blur, before it forms the preconditioner.  Because the weights sum to one,
  ``eta G`` is a term of every kernel and costs nothing; ``eta`` is given as a fraction of
  the largest square-root symbol (the reference's 0.1 is in the units of its own).
* **The size can be applied like the taper, and is not the paper's** (:func:`amplitude`, the
  argument ``amplitude`` of :func:`build`).  The Hessian of a survey at the surface falls
  with depth by orders of magnitude within one step of the sample lattice, and no sample
  point lies above the first one.  With ``a(x)`` the square root of the Hessian's diagonal,
  estimated from products with white vectors, the kernels are those of ``p_k(y) / (a(x_k +
  y) a(x_k))`` and the caller gives ``W a`` where it gave ``W``: ``H~ = W a S S^T a W``,
  symmetric positive as before.  On the Hessian of the small Marmousi case the diagonal of
  ``H~`` over the Hessian's runs with depth from 0.40 under the sea bed to 4.5 at the bottom
  without it and from 0.95 to 1.6 with it.

The Hessian of the posterior is approximated by ``H~ + R`` (:class:`ApproximateHessian`):
a product, a solve by conjugate gradients on device arrays, a square root for samples, and
on top of them the methods the library's algorithms call.  The symbol of the inverse,
``(W_k^2 s_k + r)^-1`` with ``r`` the prior's symbol, does not replace that solve: as the
only preconditioner of the true system it takes 278 iterations on the test Hessian
(largest eigenvalue against the prior 1e4) where the solve takes 27.  It preconditions the
solve: 138 to 167 iterations where the prior takes 579.

A product is a loop over batches of boxes, and what a batch costs beside its transforms is
what the loop costs: run from Python, a product with ``H~`` is about a hundred dispatches of
small compiled functions and updates of whole arrays, 52 ms on an MI210 on a lattice of 30 625
nodes, where its 144 transforms take 6.  So a product is ONE compiled call (:data:`WHOLE`,
``HIPPYMFEM_PSF_WHOLE``): ``S`` and ``S^T`` each, and on a rank that holds the whole lattice
(every rank has the box on its own card and one of them does the lattice's algebra) ``H~`` is
one call and an iteration of the solve with ``H~ + R`` one call and one transfer of three
numbers: 6.2 ms an iteration with the prior there, where the pieces took 55.  The same
operations in the same order: the same bits as the loop.

Everything here works on a rank's OWNED block of the lattice, as arrays ``[k, j, i]`` on
the device, in two or three dimensions.  What it asks of the rest of the application:

* a halo object (``lattice.DeviceHalo``; :class:`OneRankHalo` here is its one-rank case)
  with ``own_shape``, ``shape``, ``own_slices``, ``origin``, ``global_shape``,
  ``fill(a_own) -> ext``, ``add_back(ext) -> a_own`` and ``dot(a_own, b_own)``;
* a prior with ``apply_R(a)`` and ``solve_R(a)`` on owned arrays and, for the two
  symbol-space preconditioners, ``symbols(M) -> (A^, R^)``: the symbols of its two
  matrices on a periodic box of ``M`` nodes a direction, in the layout of ``rfftn``
  (``prior.LatticePrior``);
* for an iteration of the solve in one call, on a rank that holds the whole lattice: a halo
  whose ``fill`` and ``add_back`` can be traced (both of the above can where nothing is
  exchanged) and whose inner product is shown as a compiled function (:func:`_device_dot`),
  and a prior whose two functions can be traced; where one of them cannot, the iteration
  runs in its pieces as it does on several ranks;
* for the library's face, a map with ``to_lattice(v)`` and ``from_lattice(a, v=None)``
  between a parameter vector and the owned block (``lattice.LatticeMap``).
"""
import itertools
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import hippymfem as hm
from hippymfem.common.parvector import allreduce
from hippymfem.fem import kernel as kernel_mod

import jax
import jax.numpy as jnp

# the comb loop is the ice sheet application's.  Its folder goes to the END of the path: both
# folders have a run.py, a figures.py and a scale.py, and this application's must be found first
_ICESHEET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icesheet")
if _ICESHEET not in sys.path:
    sys.path.append(_ICESHEET)
import localhessian                                                          # noqa: E402

#: corners of a cell, a dimension, that carry a weight
CORNERS = {"hat": 2, "bspline": 3}

#: How much of a product is one compiled call: ``HIPPYMFEM_PSF_WHOLE``, and the argument ``whole`` of
#: :class:`CellPC`, :class:`PSFOperator` and :class:`ApproximateHessian`, which holds over it.
#:
#: ``0`` (``loops``)
#:     a batch of boxes: the batches of a product are run from Python one after the other, each a
#:     compiled kernel and an update of the whole array outside any compiled function.  About a
#:     hundred dispatches a product with ``H~``: some 45 ms on an MI210 whatever the lattice's size.
#: ``products``
#:     ``S`` and ``S^T`` each: every batch of a product traced by one ``jax.jit``.
#: ``1`` (``whole``)
#:     that, and on a rank that holds the whole lattice (nothing to exchange) ``H~ = W S S^T W`` is
#:     one call and an iteration of the solve with ``H~ + R`` one call and one transfer to the host.
#: ``auto`` (the default)
#:     ``1`` where the temporaries of a whole call fit the card beside what it holds when the
#:     operator is made (:func:`whole_fits`), ``0`` where they would not.
#:
#: The results have the same bits whatever the setting.
WHOLE = os.environ.get("HIPPYMFEM_PSF_WHOLE", "auto").strip().lower() or "auto"

#: Box points of a product, every corner's part counted, up to which a whole call transforms ALL its boxes
#: in one batched transform each way and all the corners' parts in one (:meth:`CellPC._a1`):
#: ``HIPPYMFEM_PSF_JOIN``, and the argument ``join`` of :class:`CellPC` and :class:`PSFOperator` (a bound, or
#: ``True`` / ``False``).  A transform costs some 40 microseconds on an MI210 whatever its size, and a product
#: batch by batch is 72 of them (8 colours of boxes, a transform of a batch and one a corner): on a small
#: lattice that is all a product costs.  On 30 625 nodes (112 boxes of 16, 3.7 million such points) an
#: iteration of the solve with the prior takes 1.14 ms joined and 6.3 batch by batch, on 444 185 nodes (192
#: boxes of 32, 50 million) 7.0 and 12.1; the card holds 0.14 and 1.75 GiB at most where the batches hold
#: 0.05 and 0.30.  The default takes the second lattice and nothing much larger: the gain falls with the size
#: (the transforms' own time takes over) and the memory rises with it, 17 GiB and more on 14 million nodes.
#: ``0``: never.  The same bits either way.
JOIN = int(float(os.environ.get("HIPPYMFEM_PSF_JOIN", "") or 2 ** 26))
#: real arrays of ALL the boxes of a product, every corner's part counted, that a joined call holds at its most
JOIN_ARRAYS = 5.0

_MODES = {"0": "loops", "no": "loops", "false": "loops", "off": "loops", "loops": "loops", "products": "products",
          "1": "whole", "yes": "whole", "true": "whole", "on": "whole", "whole": "whole", "auto": "auto"}


def _mode(whole=None):
    """``loops``, ``products``, ``whole`` or ``auto``: of the argument ``whole`` of a constructor, or
    of :data:`WHOLE` where it is ``None``."""
    name = WHOLE if whole is None else str(int(whole)) if isinstance(whole, (bool, np.bool_)) else whole
    try:
        return _MODES[str(name).strip().lower()]
    except KeyError:
        raise ValueError("HIPPYMFEM_PSF_WHOLE and the argument whole are 0, products, 1 or auto, not %r"
                         % (name,)) from None


def _put(x, dtype=None):
    """A device array as it is, or a COPY of a host array, on this rank's device in ``dtype``.
    (On the host JAX takes a NumPy array's memory for its own without copying it: what the
    caller writes into that array afterwards would be read by whatever was made of it.)"""
    if isinstance(x, jax.Array):
        return x if dtype is None else x.astype(dtype)
    return jax.device_put(np.array(x, dtype=dtype), kernel_mod.device())


def _zeros(shape, dtype, like=None):
    """Zeros on this rank's device; where ``like`` is an array being traced, the operation that
    makes them inside the compiled function (not an array of zeros made now and kept in it)."""
    if isinstance(like, jax.core.Tracer):
        return jnp.zeros(tuple(shape), dtype)
    return jnp.zeros(tuple(shape), dtype, device=kernel_mod.device())


def _ready(x):
    """Wait for a device array: a time read after it is the time of what made it."""
    return jax.block_until_ready(x)


def _across(rows, d):
    """One-dimensional arrays, one a dimension, as the factors of a tensor product."""
    return [np.asarray(a).reshape([-1 if j == k else 1 for j in range(d)]) for k, a in enumerate(rows)]


def _product(rows):
    out = 1.0
    for a in _across(rows, len(rows)):
        out = out * a
    return out


def _own(halo):
    """The global nodes a rank owns, ``[(lo, hi)]`` a dimension."""
    return [(int(o) + s.start, int(o) + s.stop) for o, s in zip(halo.origin, halo.own_slices)]


# ------------------------------------------------------------------ sample points and combs
class SamplePoints:
    """The sample points of a lattice of ``shape`` nodes: per dimension the nodes
    ``first + step * i``, ``i = 0 .. K - 1``.  The comb of a sample point is ``i mod teeth``
    in every dimension: ``teeth^d`` combs whose teeth are ``teeth * step`` nodes apart."""

    def __init__(self, shape, step, first, K, teeth):
        self.shape = tuple(int(n) for n in shape)
        self.d = len(self.shape)
        self.step, self.teeth = int(step), int(teeth)
        self.first, self.K = tuple(int(a) for a in first), tuple(int(k) for k in K)
        self.ncomb = self.teeth ** self.d

    def nodes(self, k):
        """The sample points' node numbers in dimension ``k``."""
        return self.first[k] + self.step * np.arange(self.K[k])

    def color(self, index):
        """The comb of the sample point of lattice index ``index``: the number
        ``localhessian.comb_products`` gives it."""
        return int(np.ravel_multi_index(tuple(i % self.teeth for i in index), (self.teeth,) * self.d))


def combs(shape, step, anchor=None, teeth=1, constraint=None):
    """The sample points ``anchor + step * i`` of a lattice of ``shape`` nodes that lie
    inside it and inside ``constraint`` (``[(lo, hi)]`` a dimension, both ends included;
    ``None`` for no bound), and their combs.

    The constraint is where the taper of the inverse problem is one, less the window's
    radius: an impulse response taken where the taper falls would be spread by the
    interpolation over a cell in which it does not hold.  ``anchor`` is half a step by
    default.  Everything is in GLOBAL node numbers, so that the ranks agree on the combs."""
    d = len(shape)
    anchor = np.broadcast_to(np.asarray(step // 2 if anchor is None else anchor, dtype=np.int64), (d,))
    first, K = [], []
    for k in range(d):
        lo, hi = 0, int(shape[k]) - 1
        if constraint is not None and constraint[k] is not None:
            lo, hi = max(lo, int(constraint[k][0])), min(hi, int(constraint[k][1]))
        a = int(anchor[k]) + step * -((int(anchor[k]) - lo) // step)        # the first one at or after lo
        if a > hi:
            raise ValueError("no sample point in dimension %d: anchor %d, step %d, nodes %d to %d"
                             % (k, anchor[k], step, lo, hi))
        first.append(a)
        K.append((hi - a) // step + 1)
    return SamplePoints(shape, step, first, K, teeth)


def comb_products(apply_block, sample, halo, filter_sigma=0.0, dtype=np.float64, chunk=256, log=None,
                  only=None, late=None):
    """The responses of ``apply_block`` to the combs of ``sample`` on this rank's owned
    nodes, ``(combs, *own_shape)`` float32, and how many were done: the loop of
    ``localhessian.comb_products`` on the lattice of sample points.

    ``apply_block`` takes and returns ``(n_own, k)`` host arrays, a column a field in the
    order ``[k, j, i]`` of the owned block: the products with ``H0``, the Hessian WITHOUT
    the taper.  The loop makes the same calls on every rank.  With ``filter_sigma`` the
    combs see ``F H0 F`` (:func:`highpass`): the comb is filtered here before the product,
    the response in :func:`build` after it.  ``chunk``, ``log``, ``only`` and ``late`` are
    the ice sheet application's (shares of the combs kept by several jobs, a time limit).
    """
    d = sample.d
    own = _own(halo)
    IJ = np.stack(np.meshgrid(*[np.arange(lo, hi) for lo, hi in own], indexing="ij"), axis=-1).reshape(-1, d)
    # localhessian's teeth are every node first + step * i of the lattice; the sample points are those
    # of them inside the constraint, so the others are pulled out of the combs
    last = np.asarray(sample.first) + sample.step * (np.asarray(sample.K) - 1)
    inside = ((IJ >= np.asarray(sample.first)) & (IJ <= last)).all(axis=1)

    def block(M):
        M = M * inside[:, None]
        if filter_sigma:
            fields = [_put(M[:, j].reshape(halo.own_shape), dtype) for j in range(M.shape[1])]
            M = np.stack([np.asarray(highpass(halo, a, filter_sigma)).reshape(-1) for a in fields], axis=1)
        return apply_block(M)

    HM, _, done = localhessian.comb_products(block, IJ, sample.teeth, chunk=chunk, log=log, only=only,
                                             late=late, stride=sample.step, anchor=sample.first)
    return HM.T.reshape((HM.shape[1],) + tuple(halo.own_shape)), done


def comb_vectors(sample, halo, filter_sigma=0.0, dtype=np.float64):
    """The combs themselves on this rank's owned nodes, ``(combs, *own_shape)``: the sums of
    the unit vectors of the sample points of each colour (high-pass filtered if asked), for
    a caller that computes the block of products its own way."""
    return comb_products(lambda M: M, sample, halo, filter_sigma, dtype)[0]


def highpass(halo, a, sigma):
    """``a`` minus its Gaussian blur of ``sigma`` nodes: the paper's filter, which takes the
    smooth, non-local part of the Hessian (transmission) out of what the combs see, where
    every tooth's window would otherwise collect every other tooth's far field.

    A separable stencil of radius ``4 sigma`` on the halo's array, zero beyond the lattice.
    (The reference repeats the edge value there, which is not symmetric; the two differ
    within ``4 sigma`` of the lattice's edge, where the taper is not one.)"""
    r = int(4.0 * sigma + 0.5)
    own = _own(halo)
    for k, (lo, hi) in enumerate(own):
        if halo.origin[k] > lo - r or halo.origin[k] + halo.shape[k] < hi + r:
            raise ValueError("the filter of sigma %g needs a halo of %d nodes" % (sigma, r))
    y = np.arange(-r, r + 1)
    g1 = np.exp(-0.5 * (y / float(sigma)) ** 2)
    g1 = g1 / g1.sum()
    ext = halo.fill(a)
    for k in range(ext.ndim):
        n = ext.shape[k] - 2 * r
        ext = sum(float(g1[j]) * jax.lax.slice_in_dim(ext, j, n + j, axis=k) for j in range(2 * r + 1))
    return a - ext[tuple(slice(s.start - r, s.stop - r) for s in halo.own_slices)]


# --------------------------------------------------------------------------------- windows
def cut_window(rho, d, kind="reference", smooth=0.4, alpha=3.0):
    """A window ``(2 rho + 1)^d`` around a point: what cuts an impulse response out of a
    comb's response, and what cuts a kernel to a compact one.

    ``reference``: the reference code's mask, of the maximum norm: 1 up to ``(1 - smooth)
    rho``, then ``exp(-alpha t^2)`` with ``t`` from 0 to 1 at ``rho`` (0.05 there for
    ``alpha`` 3), nothing beyond.
    ``cosine``: a product of one-dimensional profiles, 1 up to ``(1 - smooth) rho``, then a
    raised cosine that reaches zero at ``rho + 1``.
    ``kaiser:B``: a product of Kaiser windows of parameter ``B``.  Low side lobes: what the
    cut of a kernel whose symbol spans orders of magnitude needs, since with a cosine
    taper the cut symbol of an inverse changes sign (``B`` 5 is 37 dB down).
    ``box``: ones.
    """
    y = np.abs(np.arange(-rho, rho + 1)).astype(np.float64)
    if kind == "box":
        return np.ones((2 * rho + 1,) * d)
    if kind == "reference":
        r = np.max(np.stack(np.meshgrid(*([y] * d), indexing="ij")), axis=0)
        t = np.clip((r / rho - (1.0 - smooth)) / smooth, 0.0, 1.0)
        return np.exp(-alpha * t ** 2)
    if kind.startswith("kaiser:"):
        w1 = np.kaiser(2 * rho + 1, float(kind.split(":")[1]))
    elif kind == "cosine":
        a = (1.0 - smooth) * rho
        w1 = 0.5 * (1.0 + np.cos(np.pi * np.clip((y - a) / (rho + 1 - a), 0.0, 1.0)))
    else:
        raise ValueError("no window %r" % (kind,))
    return _product([w1] * d)


def gaussian(sigma, rho, d):
    """The kernel of the low-frequency term, ``(2 rho + 1)^d``: a Gaussian of ``sigma``
    nodes cut at ``4 sigma`` (and at ``rho``, the kernels' radius), of sum one."""
    y = np.arange(-rho, rho + 1)
    g1 = np.where(np.abs(y) <= int(4.0 * sigma + 0.5), np.exp(-0.5 * (y / float(sigma)) ** 2), 0.0)
    return _product([g1 / g1.sum()] * d)


# ----------------------------------------------------- impulse responses, symbols, kernels
def extract_psfs(resp, origin, sample, first, count, rho, win):
    """The impulse responses around sample points, from the responses of the combs.

    ``resp``: ``(combs, ...)`` host array, the responses on an array whose first node is
    the global node ``origin`` and which is zero beyond the lattice (a rank's halo array).
    Returns ``(count_1, .., count_d, (2 rho + 1)^d)``: for the sample point of lattice index
    ``first + i`` the response of ITS comb at ``x + y``, ``|y| <= rho``, times ``win``.
    The shift of the reference code is the indexing by ``y``."""
    d = sample.d
    rp = np.pad(resp, [(0, 0)] + [(rho, rho)] * d)
    out = np.zeros(tuple(count) + (2 * rho + 1,) * d, dtype=resp.dtype)
    for idx in itertools.product(*[range(n) for n in count]):
        index = [first[k] + idx[k] for k in range(d)]
        at = [sample.first[k] + sample.step * index[k] - origin[k] for k in range(d)]
        out[idx] = rp[(sample.color(index),) + tuple(slice(x, x + 2 * rho + 1) for x in at)] * win
    return out


def _embed(ker, M, d):
    """Centred kernels ``(..., (2 r + 1)^d)`` on periodic boxes of ``M`` with the centre at
    index 0."""
    r = (ker.shape[-1] - 1) // 2
    if M < 2 * r + 1:
        raise ValueError("a kernel of radius %d does not fit a box of %d" % (r, M))
    out = jnp.pad(ker, [(0, 0)] * (ker.ndim - d) + [(0, M - 2 * r - 1)] * d)
    return jnp.roll(out, (-r,) * d, axis=tuple(range(-d, 0)))


def symbol(ker, M, d):
    """The symbol rows of centred kernels on a box of ``M``, real ``(..., M, .., M // 2 +
    1)``: of an even kernel the real part is all there is, of any other it is the symbol of
    its even part (the reference: ``rows.real``)."""
    return jnp.fft.rfftn(_embed(ker, M, d), axes=tuple(range(-d, 0))).real


def kernel_of(sym, M, r, d, win=None):
    """The centred kernels of radius ``r`` of real symbol rows on a box of ``M``, times
    ``win``."""
    axes = tuple(range(-d, 0))
    k = jnp.fft.irfftn(sym.astype(jnp.result_type(sym.dtype, jnp.complex64)), s=(M,) * d, axes=axes)
    k = jnp.roll(k, (r,) * d, axis=axes)[(Ellipsis,) + (slice(0, 2 * r + 1),) * d]
    return k if win is None else k * win


def batch_bound(m, d, max_boxes=None):
    """Boxes of ``m^d`` nodes in a batch: ``max_boxes`` if given, else what keeps a batch at
    ``2^25`` box points (65 boxes of ``80^3``, 1024 of ``32^3``).  Beside the symbols the
    device holds one batch's temporaries, about ten real arrays of its boxes with hats (the
    transforms of a cell's 8 corners are held together): 41 MB a box of ``80^3`` in double
    precision, 2.5 GiB a batch in one piece (measured: :func:`whole_bytes`).  A transform
    whose work area does not fit ends the process on an AMD card (it is not a Python
    exception)."""
    return max(1, 2 ** 25 // m ** d) if max_boxes is None else int(max_boxes)


#: real arrays of a batch's boxes (by the corners of a cell) and of the padded array that a whole
#: call holds at its most (:func:`whole_bytes`).  Measured for 8 corners; the others by their count.
WHOLE_BATCH, WHOLE_ARRAYS = {4: 6.5, 8: 10.5, 9: 11.5, 27: 30.0}, 7.0


def whole_bytes(padded_shape, m, corners, dtype, max_boxes=None):
    """What one compiled call for a product over the batches of an array of ``padded_shape`` nodes
    asks of a card in ONE piece beside its arguments, in bytes, from above.

    XLA takes the temporaries of a compiled function as one piece: a batch's (the transforms of
    its corners held together, 10.1 real arrays of the batch's boxes: :func:`batch_bound`) and
    the arrays of the product.  Measured on an MI210 on 241^3 nodes in double precision: 2.29
    GiB with boxes of 32 (batches of 512), 2.65 with boxes of 80 (batches of 64), where the
    largest piece of the loop is one batch's, 1.37 and 2.47 GiB; and the most the card held in
    a product, a transform's work area with it, 3.79 and 4.58 GiB against the loop's 3.56 and
    5.18.  That is with the batches made to wait for each other (:meth:`CellPC._after`):
    without, 8.96 GiB in one piece and 10.23 at most with boxes of 32."""
    d = len(padded_shape)
    batch = min(batch_bound(m, d, max_boxes) * m ** d, int(np.prod(padded_shape)))
    return int(np.dtype(dtype).itemsize * (WHOLE_BATCH[corners] * batch
                                           + WHOLE_ARRAYS * float(np.prod(padded_shape))))


def _room(nbytes, share=0.5):
    """Whether ``nbytes`` are within ``share`` of what this rank's device has not given out now.  The
    host always has room."""
    try:
        stats = kernel_mod.device().memory_stats()
        room = int(stats["bytes_limit"]) - int(stats["bytes_in_use"])
    except Exception:                                            # noqa: BLE001  (the host has none)
        return True
    return nbytes <= share * room


def whole_fits(padded_shape, m, corners, dtype, max_boxes=None, share=0.5):
    """Whether a product as one compiled call has room on this rank's device: its piece
    (:func:`whole_bytes`) within ``share`` of what the device's allocator has not given out when
    the operator is made (the transforms' work areas come on top).  The host always has.  A
    call that does not fit is an error of the allocator at best, and where it is a transform's
    work area that does not fit, the end of the process on an AMD card: so the loops are kept
    where there is doubt (:data:`WHOLE`, ``auto``)."""
    return _room(whole_bytes(padded_shape, m, corners, dtype, max_boxes), share)


def _in_batches(kernels, M, d, max_boxes, f):
    """``f`` of the kernels ``(.., (2 r + 1)^d)``, a batch of boxes of ``M`` at a time on the
    device."""
    flat = kernels.reshape((-1,) + kernels.shape[-d:])
    each = batch_bound(M, d, max_boxes)
    return [f(a, _put(flat[a:a + each], np.float64)) for a in range(0, flat.shape[0], each)]


def sqrt_kernels(psf, M, rho_g, d, win_g=None, max_boxes=None):
    """Kernels ``g_k`` with ``g_k * g_k`` the even, positive part of ``p_k``: the symbol rows
    of the impulse responses on a box of ``M``, their real positive part, the square root,
    back, cut to the radius ``rho_g`` with the window ``win_g``.  Returns them (host) and the
    largest square-root symbol, the unit of the low-frequency term.

    The square root of a compact kernel's symbol is not the symbol of a compact kernel, but
    the cut costs nothing in accuracy: 3.5 % on the test Hessian with it and without it,
    3.6 % at half the radius (7.0 % with a Kaiser window in the place of the cosine taper,
    which is for inverses).  A compact kernel is what makes the local form exact."""
    win = None if win_g is None else _put(win_g)

    def one(a, p):
        root = jnp.sqrt(jnp.maximum(symbol(p, M, d), 0.0))
        return np.asarray(kernel_of(root, M, rho_g, d, win)), float(root.max())

    parts = _in_batches(psf, M, d, max_boxes, one)
    g = np.concatenate([p[0] for p in parts]).reshape(psf.shape[:-d] + (2 * rho_g + 1,) * d)
    return g, max(p[1] for p in parts)


def inverse_sqrt_kernels(g, w2, r, M, rho_q, d, win_q=None, split=False, max_boxes=None):
    """Kernels ``q_k`` of the symbol ``(w_k^2 t_k^2 + r)^(-1/2)``, with ``t_k`` the symbol of
    ``g_k`` on a box of ``M``, ``w2`` the squared taper at the points ``(K_1, .., K_d)`` and
    ``r`` the prior's symbol on the box: ``sum_k w_k q_k *`` is then a square root of an
    approximate inverse of ``W S S^T W + R``.

    ``split``: of ``(1 + w_k^2 t_k^2 / r)^(-1/2)`` instead, as a unit impulse minus a compact
    kernel: the factor that stands beside an exact solve with the prior.  It is one where
    the data say nothing, so its cut loses less."""
    win = None if win_q is None else _put(win_q)
    w2 = np.asarray(w2, dtype=np.float64).reshape(-1)
    r = _put(r, np.float64)

    def one(a, gk):
        s = _put(w2[a:a + gk.shape[0]]).reshape((-1,) + (1,) * d) * symbol(gk, M, d) ** 2
        if not split:
            return np.asarray(kernel_of(1.0 / jnp.sqrt(s + r), M, rho_q, d, win))
        q = -np.asarray(kernel_of(1.0 - 1.0 / jnp.sqrt(1.0 + s / r), M, rho_q, d, win))
        q[(Ellipsis,) + (rho_q,) * d] += 1.0
        return q

    parts = _in_batches(g, M, d, max_boxes, one)
    return np.concatenate(parts).reshape(g.shape[:-d] + (2 * rho_q + 1,) * d)


def stencil_symbol(apply, shape, M, r=2):
    """The symbol on a box of ``M`` of a translation-invariant operator on a whole lattice of
    ``shape`` nodes, from its response to ONE unit vector at the middle (radius ``r``): the
    check of a prior's ``symbols``, and a symbol for a prior that has none."""
    d = len(shape)
    c = tuple(n // 2 for n in shape)
    e = np.zeros(shape)
    e[c] = 1.0
    y = apply(_put(e))[tuple(slice(ci - r, ci + r + 1) for ci in c)]
    return np.asarray(symbol(_put(y, np.float64), M, d))


# ------------------------------------------------------------------ cells, weights, tiles
class _Tiles:
    """One dimension of the cells a rank works on: ``count`` tiles of ``step`` nodes that
    cover its nodes ``[lo, hi)``, the first starting at the global node ``start``; the
    weights of a tile's ``nc`` corners at its nodes, ``table (nc, step)``; and the lattice
    index ``corner`` of the first tile's first corner.

    A tile is a cell of the sample lattice, ``[x_i, x_i + step)`` for hats.  A node ON a
    lattice plane has the weight one of that plane and zero of the next, so it may as well
    be the last node of the cell before: ``(x_i, x_i + step]``.  The choice with the fewer
    tiles is taken (with as many, the one whose outermost corners lie nearer the block).  A
    rank owns the high faces of its block, and with the block's faces on lattice planes the
    second choice has no tile of a single plane of nodes: 6 tiles where the first has 7, for
    240 nodes and a step of 40."""

    def __init__(self, sample, k, lo, hi, weights):
        L, a = sample.step, sample.first[k]
        nc = CORNERS[weights]
        half = L // 2 if weights == "bspline" else 0             # a B-spline's tile is centred on its point
        best = None
        for shift in ((0, 1) if (weights == "hat" or L % 2 == 0) else (0,)):
            off = shift - half
            i0, i1 = (lo - off - a) // L, (hi - 1 - off - a) // L
            # With as many tiles, the choice whose outermost corners lie nearer the block: a rank makes the
            # kernels of its corners from its own halo, so they are what the build's halo has to reach
            # (_reach).  The first block of a direction owns both its end planes and has a tile more either
            # way; with the single plane at its LOW end that tile's other corner is beyond the lattice, with
            # it at the high end it is a sample point a whole step further out.
            ends = np.array([i0 - (nc - 2), i1 + 1])
            x = a + L * np.clip(ends, 0, sample.K[k] - 1)
            at = np.clip(a + L * ends, 0, sample.shape[k] - 1)
            cost = (i1 - i0 + 1, int(max(lo - min(x[0], at[0]), max(x[1], at[1]) - (hi - 1))))
            if best is None or cost < best[0]:
                best = (cost, i0, off)
        (self.count, _), i0, off = best
        self.start = a + i0 * L + off
        t = (np.arange(L) + off) / float(L)                      # a node's place from its tile's point
        if weights == "hat":
            self.table = np.stack([1.0 - t, t])
        else:
            self.table = np.stack([0.5 * (0.5 - t) ** 2, 0.75 - t ** 2, 0.5 * (0.5 + t) ** 2])
        self.corner = i0 - (nc - 2)
        self.corners = self.count + nc - 1                       # the lattice indices from ``corner`` on


def _reach(sample, block, rho, weights):
    """The nodes ``(lo, hi)`` a dimension, both included, that the kernels of a block's cells are
    made of: the windows of the sample points at the cells' corners (the nearest sample point
    for a corner beyond them), and the corners themselves, where the taper is read (at the
    lattice's edge for a corner beyond it)."""
    out = []
    for k, (lo, hi) in enumerate(block):
        t = _Tiles(sample, k, lo, hi, weights)
        ends = np.array([t.corner, t.corner + t.corners - 1])
        x = sample.first[k] + sample.step * np.clip(ends, 0, sample.K[k] - 1)
        at = np.clip(sample.first[k] + sample.step * ends, 0, sample.shape[k] - 1)
        out.append((int(min(max(x[0] - rho, 0), at[0])),
                    int(max(min(x[1] + rho, sample.shape[k] - 1), at[1]))))
    return out


def build_width(sample, blocks, rho, weights="hat"):
    """The halo, in nodes a dimension, that :func:`build` needs on ranks owning ``blocks``
    (``[(lo, hi)] * d`` each, global nodes): every rank computes the kernels of the corners
    of its own cells from the comb responses within ``rho`` of them, and no kernel travels.
    ``rho`` with the blocks' faces on lattice planes (``rho + 1`` where a block owns the nodes
    from the one after its low face, as ``lattice.Lattice``'s do: ``rho`` around the closed
    range of its elements), up to ``step + rho`` otherwise, more for a block beyond the
    outermost sample points.  The same on every rank that computes it from the same blocks."""
    width = [0] * sample.d
    for block in blocks:
        for k, ((lo, hi), (need_lo, need_hi)) in enumerate(zip(block, _reach(sample, block, rho, weights))):
            width[k] = max(width[k], lo - need_lo, need_hi - (hi - 1))
    return tuple(width)


class Kernels:
    """What a rank keeps of the construction.

    ``g``: ``(n_1, .., n_d, (2 rho_g + 1)^d)`` host array, the kernels of the sample points
    of lattice indices ``first_d .. first_d + n_d - 1``; a cell corner beyond them takes the
    nearest one's.  ``smax``: the largest square-root symbol of all ranks.  ``psf``: the
    impulse responses the kernels were made of.  ``corner_window``: the taper at the corners
    of the rank's cells, from the lattice index ``corner_first`` (``None``: ones).  ``box``:
    the box of the symbol rows."""

    def __init__(self, sample, weights, g, first, smax=0.0, psf=None, corner_window=None, corner_first=None,
                 box=None):
        self.sample, self.weights, self.g, self.first = sample, weights, g, tuple(int(a) for a in first)
        self.smax, self.psf, self.box = float(smax), psf, box
        self.corner_window, self.corner_first = corner_window, corner_first
        self.rho_g = (g.shape[-1] - 1) // 2


def _fast_length(n):
    """The first even length from ``n`` on whose factors are 2, 3 and 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1 and n % 2 == 0:
            return n
        n += 1


def _blur(halo, a, sigma):
    """``a`` smoothed by a Gaussian of ``sigma`` nodes (one number a dimension; 0: not along
    that one): a separable stencil cut at ``4 sigma`` on the halo's array, zero beyond the
    lattice, and not divided by the part of it that lies inside: a caller divides two fields
    smoothed alike."""
    radius = [int(4.0 * s + 0.5) for s in sigma]
    for k, ((lo, hi), r) in enumerate(zip(_own(halo), radius)):
        if halo.origin[k] > lo - r or halo.origin[k] + halo.shape[k] < hi + r:
            raise ValueError("a Gaussian of %g nodes needs a halo of %d nodes (dimension %d)" % (sigma[k], r, k))
    ext = halo.fill(a)
    for k, r in enumerate(radius):
        if r:
            y = np.arange(-r, r + 1)
            g1 = np.exp(-0.5 * (y / float(sigma[k])) ** 2)
            g1 = g1 / g1.sum()
            n = ext.shape[k] - 2 * r
            ext = sum(float(g1[j]) * jax.lax.slice_in_dim(ext, j, n + j, axis=k) for j in range(2 * r + 1))
    return ext[tuple(slice(s.start - r, s.stop - r) for s, r in zip(halo.own_slices, radius))]


def amplitude_width(d, sigma, depth=None):
    """The widths ``(depth, sigma, .., sigma)`` of :func:`amplitude`'s Gaussian a dimension, and the
    halo they ask for, in nodes."""
    sigma = float(sigma)
    widths = (sigma / 3.0 if depth is None else float(depth),) + (sigma,) * (d - 1)
    return widths, tuple(int(4.0 * s + 0.5) for s in widths)


def amplitude(halo, vectors, products, window=None, sigma=2.0, depth=None, floor=0.1, dtype=np.float64,
              comm=None):
    r"""The local size of the Hessian on this rank's owned nodes: ``a(x)``, the square root of an
    estimate of the diagonal of ``H0``, from products with white vectors.

    ``vectors``: fields ``v`` of independent standard normals on the owned nodes; ``products``:
    ``p = W H0 (W v)`` for each (the check products of a driver, taken through the taper
    ``window``).  ``v p`` has the mean ``W^2 diag(H0)`` at a node, with a scatter of the size
    of a whole row of the Hessian, so it is averaged: ``sum_v v p`` and ``sum_v W^2 v^2`` are
    smoothed by the same Gaussian, ``sigma`` nodes along every dimension but the first
    (depth, where the size changes fastest) and ``depth`` nodes along that one (a third of
    ``sigma``), and divided.  The Gaussian is zero beyond the lattice and the taper is in the
    weights, so nothing is averaged across the taper's edge.  An estimate below ``floor`` of
    its depth plane's own (the plane's two sums, divided) is set to that: with two vectors
    and two nodes the scatter is a fifth of the size, a noisy node could come out near zero
    or below it, and the operator divides by ``a``.  Where the taper is zero all around a
    node the plane's value stands.

    One white vector is enough (the kernels are local, so the scatter averages out over
    the nodes of the Gaussian); a field that depends on depth alone (a large ``sigma``)
    gives half of what the full one gives.  ``halo``: at least as wide as
    :func:`amplitude_width` says.  Collective: every rank fills the halo twice a vector,
    and with ``comm`` (None on one rank) the planes' sums are reduced.  Returns a host array
    in double precision, positive everywhere."""
    d = len(halo.own_shape)
    widths, _ = amplitude_width(d, sigma, depth)
    if len(vectors) == 0 or len(vectors) != len(products):
        raise ValueError("the amplitude takes white vectors and their products, as many of each: %d and %d"
                         % (len(vectors), len(products)))
    W2 = None if window is None else np.asarray(window, dtype=np.float64) ** 2
    num, den = np.zeros(halo.own_shape), np.zeros(halo.own_shape)
    for v, p in zip(vectors, products):
        v, p = np.asarray(v, dtype=np.float64), np.asarray(p, dtype=np.float64)
        num += v * p
        den += v * v if W2 is None else W2 * v * v
    # a depth plane's own estimate, from its two sums over all ranks
    lo, hi = _own(halo)[0]
    planes = np.zeros((2, halo.global_shape[0]))
    across = tuple(range(1, d))
    planes[0, lo:hi], planes[1, lo:hi] = num.sum(axis=across), den.sum(axis=across)
    if comm is not None:
        mine, planes = planes, np.empty_like(planes)
        comm.Allreduce(mine, planes, op=MPI.SUM)
    total = planes[0].sum() / planes[1].sum() if planes[1].sum() > 0.0 else 0.0
    if not total > 0.0:
        raise ValueError("the products with the white vectors have no positive mean of v . H v: no amplitude")
    plane = np.where(planes[1] > 0.0, planes[0] / np.where(planes[1] > 0.0, planes[1], 1.0), 0.0)
    plane = np.maximum(plane, 1e-3 * total)[lo:hi].reshape((-1,) + (1,) * (d - 1))
    # the two sums smoothed alike, and divided
    num = np.asarray(_blur(halo, _put(num, dtype), widths), dtype=np.float64)
    den = np.asarray(_blur(halo, _put(den, dtype), widths), dtype=np.float64)
    top = float(den.max())
    if comm is not None:
        top = allreduce(comm, top, op=MPI.MAX)
    have = den > 1e-8 * top
    est = np.where(have, num / np.where(have, den, 1.0), plane)
    return np.sqrt(np.maximum(est, floor * plane))


def build(responses, sample, halo, rho, rho_g=None, weights="hat", mask="reference", filter_sigma=0.0,
          window=None, taper="cosine", box=None, dtype=np.float64, comm=None, max_boxes=None, amplitude=None):
    """The kernels of this rank from the combs' responses on its owned nodes.

    ``responses``: ``(combs, *own_shape)``, the products of ``H0`` with the combs
    (:func:`comb_products`).  ``rho``: the radius of the mask that cuts an impulse response
    out, at most half the distance of the teeth.  ``rho_g``: the radius of the kernels
    (``rho``).  ``mask``, ``taper``: the windows of the two cuts (:func:`cut_window`).
    ``filter_sigma``: the second half of the high-pass filter.  ``window``: the taper ``W``
    on the owned nodes, of which the values at the cells' corners are kept for the
    symbol-space preconditioners.  ``box``: the box of the symbol rows (about ``3 rho``).
    ``halo``: wide enough for :func:`build_width`, and for the filter; arrays go through it
    in ``dtype``.  ``comm``: the ranks' communicator (``None`` on one rank), for the largest
    square-root symbol of them all.

    ``amplitude``: the local size ``a`` of the Hessian on the owned nodes, positive
    (:func:`amplitude`).  The kernels are then those of the responses divided by ``a`` where
    they are read and by ``a`` at their sample point, kernels of a Hessian of unit size, and
    the operator of them wants ``W a`` for its ``window``, here and in :class:`PSFOperator`:
    ``H~ = W a S S^T a W``.  The amplitudes at the sample points of the kernels are kept as
    the result's ``amplitude`` (None without).

    A rank computes the kernel of every corner of its cells from its own halo array, so
    corners that several ranks share are computed by each from the same numbers and nothing
    is exchanged but the responses.  Collective: every rank fills the halo ``combs`` times
    (once more with a window, once more with an amplitude), and with ``comm`` two numbers
    are reduced (three with an amplitude).
    """
    d = sample.d
    rho_g = rho if rho_g is None else int(rho_g)
    if 2 * rho > sample.teeth * sample.step:
        raise ValueError("a mask of radius %d reaches the next tooth of its comb (%d nodes apart)"
                         % (rho, sample.teeth * sample.step))
    if len(responses) != sample.ncomb:
        raise ValueError("%d responses for %d combs" % (len(responses), sample.ncomb))
    own, origin, shape = _own(halo), halo.origin, halo.shape
    # a halo too narrow for one rank stops them all, before any of them waits in an exchange
    short = any(need_lo < o or need_hi >= o + n
                for (need_lo, need_hi), o, n in zip(_reach(sample, own, rho, weights), origin, shape))
    if comm is not None:
        short = allreduce(comm, int(short), op=MPI.MAX)
    if short:
        raise ValueError("the halo does not hold the impulse responses of the cells of every rank: here %s "
                         "nodes are needed beyond the block (psf.build_width)"
                         % (build_width(sample, [own], rho, weights),))
    tiles = [_Tiles(sample, k, lo, hi, weights) for k, (lo, hi) in enumerate(own)]
    first = [int(np.clip(t.corner, 0, sample.K[k] - 1)) for k, t in enumerate(tiles)]
    last = [int(np.clip(t.corner + t.corners - 1, 0, sample.K[k] - 1)) for k, t in enumerate(tiles)]
    ext = []
    for c in range(sample.ncomb):
        a = _put(responses[c], dtype)
        if filter_sigma:
            a = highpass(halo, a, filter_sigma)
        ext.append(np.asarray(halo.fill(a)))
    count = [b - a + 1 for a, b in zip(first, last)]
    at_points = None
    if amplitude is not None:
        # the responses over the size where they are read (nothing beyond the lattice, where both are
        # zero), and below over the size at the sample point each belongs to
        size = np.asarray(halo.fill(_put(amplitude, dtype)))
        ext = [np.divide(e, size, out=np.zeros_like(e), where=size > 0.0) for e in ext]
        at = [sample.first[k] + sample.step * (first[k] + np.arange(count[k])) - origin[k] for k in range(d)]
        at_points = size[np.ix_(*at)].astype(np.float64)
        bad = int(not at_points.min() > 0.0)
        if comm is not None:
            bad = allreduce(comm, bad, op=MPI.MAX)
        if bad:
            raise ValueError("the amplitude is not positive at every sample point")
    psf = extract_psfs(np.stack(ext), origin, sample, first, count, rho, cut_window(rho, d, mask))
    del ext
    if at_points is not None:
        psf = psf / at_points.reshape(at_points.shape + (1,) * d).astype(psf.dtype)
    box = _fast_length(3 * max(rho, rho_g) + 4) if box is None else int(box)
    g, smax = sqrt_kernels(psf, box, rho_g, d, cut_window(rho_g, d, taper), max_boxes)
    if comm is not None:
        smax = allreduce(comm, smax, op=MPI.MAX)
    corner_window = None
    if window is not None:
        # at a corner beyond the lattice the taper of the nearest node: zero where it falls to the edge
        at = [np.clip(sample.first[k] + sample.step * (t.corner + np.arange(t.corners)), 0,
                      sample.shape[k] - 1) - origin[k] for k, t in enumerate(tiles)]
        corner_window = np.asarray(halo.fill(_put(window, dtype)))[np.ix_(*at)].astype(np.float64)
    out = Kernels(sample, weights, g, first, smax, psf, corner_window, [t.corner for t in tiles], box)
    out.amplitude = at_points
    return out


# ----------------------------------------------------- the operator, local batched FFTs
class _Batches(list):
    """The batches of a range of tiles (:meth:`CellPC._batches`), ``(box, tiles, count, corners)``
    each, and the compiled calls of the two products over all of them, made when first asked for
    (:meth:`CellPC._call`): they go when the list does."""

    def __init__(self, batches=()):
        super().__init__(batches)
        self.calls = {}


class CellPC:
    """``u -> sum_k w_k (g_k * u)`` on the nodes ``own`` of a rank, and its transpose ``v ->
    sum_k g_k * (w_k v)``, by transforms of boxes of one size.

    ``kernels``: ``(n_1, .., n_d, (2 rho + 1)^d)`` even kernels of the sample points of
    lattice indices ``first_d .. first_d + n_d - 1`` (a corner beyond takes the nearest
    one's); ``own``: ``[(lo, hi)]`` a dimension, global nodes; ``frame``: the global node of
    the first entry, and the shape, of the array the field comes in and its transpose goes
    out in (the halo's), which holds ``rho`` nodes around ``own`` at least.

    A box is a tile (:class:`_Tiles`) and a margin: ``m = ncol * step >= step + 2 rho``
    nodes a side.  The boxes of the tiles with the same index modulo ``ncol`` in every
    dimension tile the array that holds all of them, so a batch of boxes is a slice and a
    reshape of it, and the overlap-add of the transpose a reshape back and an add on a
    slice.  A batch is at most ``max_boxes`` boxes (:func:`batch_bound`; cut along the first
    dimension).  The two kernels are compiled once for every shape of a batch, with the
    symbols as an argument.

    Both products take a list of batches in the place of all of them: :meth:`tapered` gives
    the one of a product that a taper multiplies, without the outermost tiles on which the
    taper is zero.

    ``join`` (:data:`JOIN`): where the boxes of a product are few, a whole call does not go batch
    by batch at all: it gathers every box, transforms them in one batched transform, all the
    corners' parts of all of them in one back, and puts the tiles where the batches would have
    (:meth:`_a1`, :meth:`_t1`).  ``joined`` says whether.

    ``whole`` (:data:`WHOLE`): a product is ONE compiled call, the loop over its batches traced
    by one ``jax.jit`` with the symbols as an argument, or the loop itself run from Python.  The
    same operations in the same order, and the same bits; what differs is where the arrays of a
    product live (inside a compiled function XLA writes a batch's result into the array it has,
    outside each update is a new array of the whole size) and the dispatches: on an MI210 in
    double precision ``S`` and ``S^T`` take 33.2 and 18.1 ms as loops and 3.8 and 2.4 ms as one
    call each on a lattice of 30 625 nodes (112 boxes of 16), and 131.9 and 105.7 against 114.6
    and 92.5 ms on 14 million nodes (4096 boxes of 32).

    Holds ``sym`` ``(n_1 .. n_d, m, .., m // 2 + 1)`` real, the kernels' symbols on the box one
    after the other, and ``hats`` ``(nc^d, step^d)``, the weights of a tile's corners.
    """

    def __init__(self, sample, weights, kernels, first, own, frame, dtype=np.float64, max_boxes=None,
                 whole=None, join=None):
        d = self.d = sample.d
        L = self.L = sample.step
        self.dtype = np.dtype(dtype)
        self.rho = (kernels.shape[-1] - 1) // 2
        self.ncol = 1 + -(-2 * self.rho // L)
        m = self.m = self.ncol * L
        self.h0 = (m - L) // 2                                   # the margin before a tile; the rest after it
        origin, shape = frame
        tiles = [_Tiles(sample, k, lo, hi, weights) for k, (lo, hi) in enumerate(own)]
        self.Nc = [t.count for t in tiles]
        self.own_shape = tuple(hi - lo for lo, hi in own)
        #: the array the boxes are cut from: every tile and the margins of the outermost ones
        self.padded_shape = tuple((n + self.ncol - 1) * L for n in self.Nc)
        self._in, self._in_pad, self._out, self._out_pad = [], [], [], []
        for k in range(d):
            lo, hi = own[k]
            if origin[k] > lo - self.rho or origin[k] + shape[k] < hi + self.rho:
                raise ValueError("kernels of radius %d need a halo of as many nodes around the block "
                                 "(dimension %d)" % (self.rho, k))
            # the part that the frame and the padded array share, in each one's own numbering
            p0 = tiles[k].start - self.h0
            a, b = max(p0, origin[k]), min(p0 + self.padded_shape[k], origin[k] + shape[k])
            self._in.append(slice(a - origin[k], b - origin[k]))
            self._in_pad.append((a - p0, p0 + self.padded_shape[k] - b))
            self._out.append(slice(a - p0, b - p0))
            self._out_pad.append((a - origin[k], origin[k] + shape[k] - b))
        self._in, self._out = tuple(self._in), tuple(self._out)
        self._own = tuple(slice(lo - t.start, hi - t.start) for (lo, hi), t in zip(own, tiles))
        self._own_pad = [(lo - t.start, t.start + t.count * L - hi) for (lo, hi), t in zip(own, tiles)]
        nc = CORNERS[weights]
        self.corners = list(itertools.product(range(nc), repeat=d))
        # On the host XLA runs the transforms of a compiled function that do not wait for each other
        # at the same time, and its FFT (DUCC, jaxlib 0.11.1) can leave every thread of the pool
        # waiting for a free one: with 4 cores a batch's 8 transforms hung.  A loop over the corners
        # makes each wait for the one before.  A card runs them one after the other as it is, and
        # the loop costs 10 % there (143 against 130 ms for S on 240^3 nodes), so it is the host's.
        self._loop = not kernel_mod.on_gpu()
        self.hats = _put(np.stack([_product([tiles[k].table[p[k]] for k in range(d)]) for p in self.corners]),
                         self.dtype)
        self.sym = jnp.concatenate(_in_batches(kernels, m, d, max_boxes,
                                               lambda a, g: symbol(g, m, d).astype(self.dtype)))
        # what the batches of a range of tiles are made from (_batches): the corners of a tile, the
        # lattice index of the first tile's first corner, the stored kernels' first index and their
        # count, the boxes of a batch
        self._stored = (nc, [t.corner for t in tiles], list(first), kernels.shape[:d],
                        batch_bound(m, d, max_boxes))
        #: the batches of all the tiles: where the boxes lie in the padded array, the tiles they are the
        #: boxes of, how many a dimension, and the stored kernel of each corner of each tile, ``(nc^d,
        #: T_1, .., T_d)``
        self.pieces, self.fft_points = self._batches([0] * d, self.Nc)
        self._ka, self._kt = jax.jit(self._kernel_a), jax.jit(self._kernel_t)
        #: ``loops``, ``products`` or ``whole`` (:data:`WHOLE`; ``auto`` is decided here, once)
        self.mode = _mode(whole)
        if self.mode == "auto":
            self.mode = "whole" if whole_fits(self.padded_shape, m, len(self.corners), self.dtype,
                                              max_boxes) else "loops"
        # what makes the batches of a whole call wait for each other (_after)
        self._live = _put(np.array(True))
        #: whether a whole call transforms every box of the product in one batch (:data:`JOIN`): where the
        #: boxes are few enough, and (unless asked for outright) what they take fits the card as it is now
        points = len(self.corners) * int(np.prod(self.Nc)) * m ** d
        if isinstance(join, (bool, np.bool_)):
            self.joined = bool(join) and self.mode != "loops"
        else:
            self.joined = (self.mode != "loops" and points <= (JOIN if join is None else join)
                           and _room(JOIN_ARRAYS * points * self.dtype.itemsize))

    def _batches(self, lo, hi):
        """The batches of the tiles ``lo_k <= t < hi_k`` and the box points they transform a product.

        A tile's colour is its index modulo ``ncol`` among ALL the tiles, and the colours come in one
        order whatever the range: the boxes that overlap at a node of the transpose are added there a
        colour after the other, so with a range they are added in the order they have without one."""
        d, L, m, ncol = self.d, self.L, self.m, self.ncol
        nc, corner, first, have, bound = self._stored
        out, points = [], 0
        for color in itertools.product(range(ncol), repeat=d):
            s = [lo[k] + (color[k] - lo[k]) % ncol for k in range(d)]        # the colour's first tile there
            T = [len(range(s[k], hi[k], ncol)) for k in range(d)]
            if min(T) == 0:
                continue
            rows = max(1, bound // int(np.prod(T[1:])))
            for a in range(0, T[0], rows):
                Tp = [min(a + rows, T[0]) - a] + T[1:]
                t0 = [s[0] + a * ncol] + s[1:]
                box = tuple(slice(t0[k] * L, t0[k] * L + Tp[k] * m) for k in range(d))
                cells = tuple(slice(t0[k], t0[k] + Tp[k] * ncol, ncol) for k in range(d))
                at = []
                for k in range(d):
                    t = t0[k] + ncol * np.arange(Tp[k])
                    c = corner[k] + t[None, :] + np.arange(nc)[:, None]
                    at.append(np.clip(c, first[k], first[k] + have[k] - 1) - first[k])
                ix = [np.ravel_multi_index(np.ix_(*[at[k][p[k]] for k in range(d)]), have)
                      for p in self.corners]
                out.append((box, cells, Tp, _put(np.stack(ix).astype(np.int32))))
                points += (1 + len(self.corners)) * int(np.prod(Tp)) * m ** d
        return _Batches(out), points

    def tapered(self, nonzero):
        """The batches for a product with a taper ``W`` around it (``W S``, ``S^T W``) and how many boxes
        they are: every tile's but those of the outermost planes of tiles, a dimension, in which ``W`` is
        zero on every node this rank owns.  ``nonzero``: where ``W`` is not zero on the rank's own nodes
        (host).  ``(None, all the boxes)`` if there is no such plane.

        A tile on which ``W`` is zero is a tile of zeros for ``S^T W``, whose box then adds zeros to the
        result, and a tile of ``S`` that ``W`` multiplies by zero.  Left out of both, ``W S S^T W u`` has
        the same bits wherever ``W`` is not zero (the boxes kept are added in the order they had,
        :meth:`_batches`) and is zero where it is, for any ``u`` that is finite.  Which tiles a rank
        leaves out is its own affair: the arrays it exchanges are what they were.

        Only planes at the ends, so that the tiles kept are still slices of the arrays; a tile of zeros
        in between is transformed as before.  With lattice planes on the blocks' faces the lattice's
        first plane of nodes is a plane of tiles of its own (:class:`_Tiles`), which the first block of
        a direction has more than the others."""
        d, L = self.d, self.L
        nonzero = np.asarray(nonzero, dtype=bool)
        if nonzero.shape != self.own_shape:                      # a taper that broadcasts multiplies as well
            try:
                nonzero = np.broadcast_to(nonzero, self.own_shape)
            except ValueError:
                raise ValueError("a taper of shape %s on a rank that owns %s nodes: it is given on the OWNED "
                                 "block" % (nonzero.shape, self.own_shape)) from None
        a = np.pad(nonzero, self._own_pad)
        a = a.reshape([n for k in range(d) for n in (self.Nc[k], L)]).any(axis=tuple(range(1, 2 * d, 2)))
        lo, hi = [], []
        for k in range(d):
            used = np.flatnonzero(a.any(axis=tuple(j for j in range(d) if j != k)))
            lo.append(int(used[0]) if used.size else 0)
            hi.append(int(used[-1]) + 1 if used.size else 0)
        if all(b - a == n for a, b, n in zip(lo, hi, self.Nc)):
            return None, int(np.prod(self.Nc))
        return self._batches(lo, hi)[0], int(np.prod([b - a for a, b in zip(lo, hi)]))

    # ---- one batch
    def _sum(self, term, zero):
        """``term(i)`` summed over the corners of a cell, a transform each."""
        n = len(self.corners)
        if self._loop:
            return jax.lax.fori_loop(0, n, lambda i, acc: acc + term(i), zero())
        acc = term(0)
        for i in range(1, n):
            acc = acc + term(i)
        return acc

    def _kernel_a(self, boxes, sym, hats, ix):
        """Boxes of ``u`` ``(T.., m..)`` to the tiles' parts of the result ``(T.., step..)``."""
        d, L, m, h0 = self.d, self.L, self.m, self.h0
        axes = tuple(range(-d, 0))
        crop = (Ellipsis,) + (slice(h0, h0 + L),) * d
        U = jnp.fft.rfftn(boxes, axes=axes)

        def corner(i):
            return jnp.fft.irfftn(U * jnp.take(sym, ix[i], axis=0), s=(m,) * d, axes=axes)[crop] * hats[i]

        return self._sum(corner, lambda: jnp.zeros(boxes.shape[:d] + (L,) * d, boxes.dtype))

    def _kernel_t(self, tiles, sym, hats, ix):
        """The tiles' parts of ``v`` ``(T.., step..)`` to boxes of the transpose ``(T.., m..)``."""
        d, L, m, h0 = self.d, self.L, self.m, self.h0
        axes = tuple(range(-d, 0))
        pad = [(0, 0)] * d + [(h0, m - L - h0)] * d

        def corner(i):
            return jnp.fft.rfftn(jnp.pad(tiles * hats[i], pad), axes=axes) * jnp.take(sym, ix[i], axis=0)

        acc = self._sum(corner, lambda: jnp.zeros(tiles.shape[:d] + (m,) * (d - 1) + (m // 2 + 1,),
                                                  jnp.result_type(tiles.dtype, jnp.complex64)))
        return jnp.fft.irfftn(acc, s=(m,) * d, axes=axes)

    def _to_boxes(self, a, T, size):
        """``(T_1 size, .., T_d size)`` to ``(T_1, .., T_d, size, .., size)``."""
        d = self.d
        a = a.reshape([n for k in range(d) for n in (T[k], size)])
        return a.transpose([2 * k for k in range(d)] + [2 * k + 1 for k in range(d)])

    def _from_boxes(self, a, T, size):
        d = self.d
        return a.transpose([j for k in range(d) for j in (k, d + k)]).reshape([T[k] * size for k in range(d)])

    # ---- the two products
    @staticmethod
    def _after(a, done, live):
        """``a``, made to wait for ``done``: its first entry is written again, as it is, by an
        expression that reads ``done``'s (``live`` is a traced value that is true, which the
        compiler cannot know).  ``a`` to the bit; ``None`` for ``live``: ``a`` itself.

        What a whole call's batches read is made to wait for the batch before, for two reasons.
        Their transforms do not depend on each other, and XLA keeps the temporaries of what may
        run side by side apart: without the wait one call for ``S^T`` on 14 million nodes took
        the temporaries of its 8 batches in one piece of 8.96 GiB, with it 2.29, where the loop
        holds one batch's (1.37).  And on the host XLA does run such transforms at once, and as
        many of them as its pool has threads never return (:meth:`_sum`'s reason, between the
        batches here)."""
        if live is None:
            return a
        first = jnp.where(live, a[(0,) * a.ndim], done[(0,) * done.ndim].astype(a.dtype))
        return jax.lax.dynamic_update_slice(a, first.reshape((1,) * a.ndim), (0,) * a.ndim)

    def _a(self, ext, sym, hats, ixs, pieces, live=None):
        """:meth:`apply` on its arrays, for a list of batches and their corners' kernels ``ixs``:
        what runs from Python as it stands and inside one compiled call when traced (``live``:
        :meth:`_after`)."""
        L, m = self.L, self.m
        up = jnp.pad(ext[self._in], self._in_pad)
        cells = _zeros(tuple(self.Nc) + (L,) * self.d, self.dtype, ext)
        for (box, where, T, _), ix in zip(pieces, ixs):
            cells = cells.at[where].set(self._ka(self._to_boxes(up[box], T, m), sym, hats, ix))
            up = self._after(up, cells, live)
        return self._from_boxes(cells, self.Nc, L)[self._own]

    def _t(self, v, sym, hats, ixs, pieces, live=None):
        """:meth:`apply_t` on its arrays, as :meth:`_a`."""
        L, m = self.L, self.m
        cells = self._to_boxes(jnp.pad(v, self._own_pad), self.Nc, L)
        z = _zeros(self.padded_shape, self.dtype, v)
        for (box, where, T, _), ix in zip(pieces, ixs):
            z = z.at[box].add(self._from_boxes(self._kt(cells[where], sym, hats, ix), T, m))
            cells = self._after(cells, z, live)
        return jnp.pad(z[self._out], self._out_pad)

    def _a1(self, ext, sym, hats, ixs, pieces, live=None):
        """:meth:`_a` with every box of the product in ONE batch: a transform of all the boxes, one
        back of all the corners' parts of all of them, where :meth:`_a` makes 9 a batch.  The boxes are
        gathered and the tiles put back batch by batch, the corners summed in the same order: the
        same operations on every box, and the same bits."""
        d, L, m, h0 = self.d, self.L, self.m, self.h0
        if not pieces:
            return self._a(ext, sym, hats, ixs, pieces)
        axes = tuple(range(-d, 0))
        crop = (Ellipsis,) + (slice(h0, h0 + L),) * d
        up = jnp.pad(ext[self._in], self._in_pad)
        boxes = jnp.concatenate([self._to_boxes(up[box], T, m).reshape((-1,) + (m,) * d) for box, _, T, _ in pieces])
        ix = jnp.concatenate([i.reshape(i.shape[0], -1) for i in ixs], axis=1)             # (corners, boxes)
        U = jnp.fft.rfftn(boxes, axes=axes)
        Y = jnp.fft.irfftn(U[None] * jnp.take(sym, ix, axis=0), s=(m,) * d, axes=axes)[crop]
        parts = self._sum(lambda i: Y[i] * hats[i], lambda: jnp.zeros(boxes.shape[:1] + (L,) * d, boxes.dtype))
        cells, at = _zeros(tuple(self.Nc) + (L,) * d, self.dtype, ext), 0
        for _, where, T, _ in pieces:
            n = int(np.prod(T))
            cells = cells.at[where].set(parts[at:at + n].reshape(tuple(T) + (L,) * d))
            at += n
        return self._from_boxes(cells, self.Nc, L)[self._own]

    def _t1(self, v, sym, hats, ixs, pieces, live=None):
        """:meth:`_t` with every box of the product in one batch, as :meth:`_a1`."""
        d, L, m, h0 = self.d, self.L, self.m, self.h0
        if not pieces:
            return self._t(v, sym, hats, ixs, pieces)
        axes = tuple(range(-d, 0))
        cells = self._to_boxes(jnp.pad(v, self._own_pad), self.Nc, L)
        tiles = jnp.concatenate([cells[where].reshape((-1,) + (L,) * d) for _, where, _, _ in pieces])
        ix = jnp.concatenate([i.reshape(i.shape[0], -1) for i in ixs], axis=1)
        pad = [(0, 0)] * 2 + [(h0, m - L - h0)] * d
        X = jnp.fft.rfftn(jnp.pad(tiles[None] * hats[:, None], pad), axes=axes)             # (corners, boxes, ..)
        acc = self._sum(lambda i: X[i] * jnp.take(sym, ix[i], axis=0), lambda: jnp.zeros(X.shape[1:], X.dtype))
        Y = jnp.fft.irfftn(acc, s=(m,) * d, axes=axes)
        z, at = _zeros(self.padded_shape, self.dtype, v), 0
        for box, _, T, _ in pieces:
            n = int(np.prod(T))
            z = z.at[box].add(self._from_boxes(Y[at:at + n].reshape(tuple(T) + (m,) * d), T, m))
            at += n
        return jnp.pad(z[self._out], self._out_pad)

    def product(self, which, x, sym, hats, ixs, pieces, live):
        """One of the two products as it is traced into a compiled call: batch by batch, or every box
        in one batch (:data:`JOIN`)."""
        body = ((self._a1, self._a), (self._t1, self._t))[which == "t"][not self.joined]
        return body(x, sym, hats, ixs, pieces, live)

    def _call(self, which, pieces):
        """One of the two products over all the batches of ``pieces`` as ONE compiled call ``(x, sym,
        hats, ixs, live)``, or ``None`` where the batches run from Python (:data:`WHOLE`; a list that
        :meth:`_batches` did not make)."""
        calls = getattr(pieces, "calls", None)
        if self.mode == "loops" or calls is None:
            return None
        if which not in calls:
            calls[which] = jax.jit(lambda x, sym, hats, ixs, live: self.product(which, x, sym, hats, ixs, pieces, live))
        return calls[which]

    def apply(self, ext, pieces=None):
        """``sum_k w_k (g_k * u)`` on the rank's own nodes from ``u`` in the frame: its own values
        and its neighbours' within ``rho`` of them, zero beyond the lattice.  With ``pieces``
        (:meth:`tapered`) on their tiles only, and zero on the others."""
        pieces = self.pieces if pieces is None else pieces
        ixs, call = [p[3] for p in pieces], self._call("a", pieces)
        if call is None:
            return self._a(ext, self.sym, self.hats, ixs, pieces)
        return call(ext, self.sym, self.hats, ixs, self._live)

    def apply_t(self, v, pieces=None):
        """``sum_k g_k * (w_k v)`` of ``v`` on the rank's own nodes, in the frame: what lies around
        the block belongs to the neighbours, to be sent and added there.  With ``pieces``
        (:meth:`tapered`) of the part of ``v`` on their tiles: all of it if ``v`` is zero on the
        others."""
        pieces = self.pieces if pieces is None else pieces
        ixs, call = [p[3] for p in pieces], self._call("t", pieces)
        if call is None:
            return self._t(v, self.sym, self.hats, ixs, pieces)
        return call(v, self.sym, self.hats, ixs, self._live)

    def memory(self):
        size = self.sym.dtype.itemsize
        return dict(symbols_bytes=int(np.prod(self.sym.shape)) * size,
                    per_sample_point_bytes=int(np.prod(self.sym.shape[1:])) * size,
                    box=self.m, cell=self.L, radius=self.rho, sample_points=self.sym.shape[0],
                    boxes=int(np.prod(self.Nc)), batches=len(self.pieces),
                    fft_points_per_node=self.fft_points / float(np.prod(self.own_shape)), calls=self.mode,
                    joined=self.joined)


class OneRankHalo:
    """The halo of a lattice that one rank holds whole: the members of ``lattice.DeviceHalo``
    this module uses, for the layout in which every rank has the whole box on its card, and
    for tests.  ``fill`` pads with zeros and nothing is sent."""

    def __init__(self, shape, width):
        d = len(shape)
        self.W = tuple(int(w) for w in np.broadcast_to(width, (d,)))
        self.global_shape = self.own_shape = tuple(int(n) for n in shape)
        self.shape = tuple(n + 2 * w for n, w in zip(self.own_shape, self.W))
        self.origin = tuple(-w for w in self.W)
        self.own_slices = tuple(slice(w, w + n) for n, w in zip(self.own_shape, self.W))

    def fill(self, a):
        return jnp.pad(a, [(w, w) for w in self.W])

    def add_back(self, ext):
        return ext[self.own_slices]

    def dot(self, a, b):
        return float(_dot(a, b))


#: an inner product accumulated in double precision whatever the arrays' own
_dot = jax.jit(lambda a, b: jnp.sum(a * b, dtype=jnp.float64))


def _holds_all(halo):
    """Whether the halo's rank owns the whole lattice: its ``fill`` and ``add_back`` then exchange
    nothing (a pad and a cut), and what works through them can be traced into one compiled call."""
    return tuple(halo.own_shape) == tuple(halo.global_shape)


def _device_dot(halo):
    """The inner product of a halo that holds the whole lattice as a function that can be traced
    (``halo.dot`` returns a number of the host), or ``None`` if the halo does not show one: this
    module's, and ``lattice.DeviceHalo``'s compiled sum, which is not the same sum in single
    precision (the products in double there, in single here)."""
    if isinstance(halo, OneRankHalo):
        return _dot
    return getattr(halo, "_dot", None)


def _same_bits(a, b):
    """Whether two device arrays, or two sequences of them, are the same to the bit."""
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same_bits(x, y) for x, y in zip(a, b))
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    kind = {4: jnp.uint32, 8: jnp.uint64}[a.dtype.itemsize]
    return bool(jnp.all(jax.lax.bitcast_convert_type(a, kind) == jax.lax.bitcast_convert_type(b, kind)))


class _Whole:
    """A function of device arrays as ONE compiled call that stands for the same made in pieces, and is
    taken for them only once it has been seen to give their bits.

    ``ok`` is ``None`` until the first comparison, then whether the call is in use.  :meth:`same` makes
    the call beside a result of the pieces (made first, so that whatever the pieces' parts create at
    their first call, a table, a compiled function, exists before anything is traced) and compares;
    a call that cannot be traced (a halo or a prior of another kind), whose temporaries do not fit, or
    whose result is not the pieces' to the bit (a part that is not itself a compiled function, whose
    operations a compiled whole rounds otherwise) is given up for good, and ``error`` says why."""

    def __init__(self, body):
        self.fn, self.ok, self.error = jax.jit(body), None, None

    def __bool__(self):
        return self.ok is not False

    def same(self, args, pieces, last=True):
        """Whether the call of ``args`` gives ``pieces`` (an array or a tuple of arrays).  If not, the
        call is given up; if so and this is the ``last`` comparison, it is in use from here on."""
        try:
            out = _ready(self.fn(*args))
        except Exception as error:                               # noqa: BLE001
            self.ok, self.error = False, error
            return False
        if not _same_bits(out, pieces):
            self.ok, self.error = False, "not the bits of the pieces"
            return False
        self.ok = True if last else None
        return True

    def __call__(self, *args):
        return self.fn(*args)


class PSFOperator:
    """``S``, ``S^T`` and ``H~ = W S S^T W`` on a rank's owned lattice arrays, through its halo.

    ``kernels``: :func:`build`'s.  ``window``: the taper ``W`` on the owned nodes (ones if
    ``None``).  ``eta``: the weight of the low-frequency term as a fraction of the largest
    square-root symbol, ``blur_sigma`` its Gaussian's width in nodes; the term is added to
    every kernel (the weights sum to one), so ``S`` with ``eta = 0`` is the reference's
    ``A^`` and with ``eta > 0`` its ``A^ + eta G``, at the same cost.  The Gaussian is cut at
    the kernels' radius: keep ``4 blur_sigma`` within it.  ``dtype``: the precision of the
    symbols and of the fields, which has to be the halo's.

    A product with ``S^T`` ends with one exchange (what a rank computed on its neighbours'
    nodes is sent and added), a product with ``S`` begins with one (the neighbours' values
    around the block); ``H~`` is one of each.  ``timers`` are the seconds in the transforms
    (``S``, ``St``) and inside the exchanges; ``counts`` the products and the boxes they
    transformed.

    ``H~`` and ``sqrt`` do not transform the outermost tiles of the rank on which ``W`` is
    zero at every node, an exact zero in ``dtype`` (:meth:`CellPC.tapered`): the lattice's
    first plane of nodes where the taper vanishes on its faces, a layer of boxes that the
    first block of a direction has more than the others (4096 boxes for 3375 with blocks
    of 240 elements and a step of 16, and every rank waits for the slowest in the
    exchanges).  The result has the bits of the product over every tile where ``W`` is not
    zero and is zero where it is, and the exchanges are the same.  ``S`` and ``S^T`` alone
    transform every tile, and so does an operator without a window.  ``every_tile``: so do
    ``H~`` and ``sqrt`` (the same numbers: for a check, and for the time the tiles cost).

    ``whole`` (:data:`WHOLE`): ``S`` and ``S^T`` are each one compiled call, and on a rank that
    holds the whole lattice, where the halo's ``fill`` and ``add_back`` are a pad and a cut,
    ``H~`` and ``sqrt`` are one call each, the taper's two products in it (``whole`` says whether;
    the first product is made both ways, and the one call is kept if it gave the same bits:
    :class:`_Whole`).  The same bits as the batches run from Python.  A product in one call
    cannot be timed in its parts: ``H~``'s seconds are counted half to ``S`` and half to ``St``.
    """

    def __init__(self, halo, kernels, window=None, eta=0.0, blur_sigma=4.0, dtype=np.float64, max_boxes=None,
                 every_tile=False, whole=None, join=None):
        self.halo, self.kernels = halo, kernels
        self.dtype = np.dtype(dtype)
        # A halo exchanges ONE precision (lattice.DeviceHalo.dtype) and reads its buffers by that size.
        # A product with S of another precision is refused there; one with S^T is not, and what the
        # neighbours then add is the bytes of one precision read as the other's.  (The same on every
        # rank, so they all stop here.)
        have = getattr(halo, "dtype", None)
        if have is not None and np.dtype(have) != self.dtype:
            raise ValueError("the halo exchanges %s and the operator is asked for %s: give both the same dtype"
                             % (np.dtype(have).name, self.dtype.name))
        d = kernels.sample.d
        if eta and not kernels.smax > 0.0:
            raise ValueError("eta is a fraction of the largest square-root symbol, which these kernels "
                             "do not have")
        #: the weight of the low-frequency term in the units of the kernels
        self.eta = float(eta) * kernels.smax
        #: the kernels of ``S``: what the symbol-space preconditioners are made of
        self.g = kernels.g + self.eta * gaussian(blur_sigma, kernels.rho_g, d) if eta else kernels.g
        self.cell = CellPC(kernels.sample, kernels.weights, self.g, kernels.first, _own(halo),
                           (halo.origin, halo.shape), self.dtype, max_boxes, whole, join)
        self.window = None if window is None else _put(window, self.dtype)
        #: the boxes of a product with ``S`` or ``S^T`` alone, and of each of the two in ``H~`` and of
        #: ``sqrt``, whose batches are ``_tapered`` (``None``: every tile's)
        self.boxes = self.boxes_tapered = int(np.prod(self.cell.Nc))
        self._tapered = None
        if self.window is not None and not every_tile:
            # decided from W as it multiplies (in the operator's precision), read on the host: a tile
            # is left out for zeros in the array, whatever a card makes of a number near zero
            self._tapered, self.boxes_tapered = self.cell.tapered(np.asarray(self.window) != 0)
        self.timers = dict(S=0.0, St=0.0, exchange=0.0)
        self.counts = dict(S=0, St=0, boxes=0)
        #: whether ``H~`` and ``sqrt`` are one compiled call each (the rank holds the whole lattice)
        self.whole = self.cell.mode == "whole" and _holds_all(halo)
        self._H_call = _Whole(self._H) if self.whole else None
        self._sqrt_call = _Whole(self._sqrt) if self.whole and self.window is not None else None

    # ---- inside a compiled call (a rank that holds the whole lattice)
    def _arrays(self):
        """What a compiled call over this operator is given beside its fields: the symbols, the
        weights, the kernels of the corners of every batch (of all the tiles, and of those of a
        product that the taper multiplies), the taper, the host's flag (:meth:`CellPC._after`)."""
        cell = self.cell
        return dict(sym=cell.sym, hats=cell.hats, every=[p[3] for p in cell.pieces],
                    tapered=[p[3] for p in (cell.pieces if self._tapered is None else self._tapered)],
                    window=self.window, live=cell._live)

    def _batches(self, c, tapered):
        if tapered and self._tapered is not None:
            return self._tapered, c["tapered"]
        return self.cell.pieces, c["every"]

    def _S(self, u, c, tapered=False):
        """:meth:`S` as it is traced: ``c`` is :meth:`_arrays`."""
        pieces, ixs = self._batches(c, tapered)
        return self.cell.product("a", self.halo.fill(u), c["sym"], c["hats"], ixs, pieces, c["live"])

    def _St(self, v, c, tapered=False):
        pieces, ixs = self._batches(c, tapered)
        return self.halo.add_back(self.cell.product("t", v, c["sym"], c["hats"], ixs, pieces, c["live"]))

    def _H(self, u, c):
        if c["window"] is None:
            return self._S(self._St(u, c), c)
        return c["window"] * self._S(self._St(c["window"] * u, c, True), c, True)

    def _sqrt(self, xi, c):
        return c["window"] * self._S(xi, c, True)

    def _counted(self, products, seconds=0.0):
        """Count ``products`` products with ``H~`` made inside a compiled call."""
        self.timers["S"] += 0.5 * seconds
        self.timers["St"] += 0.5 * seconds
        self.counts["S"] += products
        self.counts["St"] += products
        self.counts["boxes"] += 2 * products * self.boxes_tapered

    # ---- the products
    def S(self, u, tapered=False):
        """``S u``.  ``tapered``: what ``W`` multiplies afterwards, zero on the tiles left out for it."""
        t0 = time.perf_counter()
        ext = _ready(self.halo.fill(u))
        t1 = time.perf_counter()
        out = _ready(self.cell.apply(ext, self._tapered if tapered else None))
        self.timers["exchange"] += t1 - t0
        self.timers["S"] += time.perf_counter() - t1
        self.counts["S"] += 1
        self.counts["boxes"] += self.boxes_tapered if tapered else self.boxes
        return out

    def St(self, v, tapered=False):
        """``S^T v``.  ``tapered``: of a ``v`` that ``W`` has multiplied, zero on the tiles left out."""
        t0 = time.perf_counter()
        ext = _ready(self.cell.apply_t(v, self._tapered if tapered else None))
        t1 = time.perf_counter()
        out = _ready(self.halo.add_back(ext))
        self.timers["St"] += t1 - t0
        self.timers["exchange"] += time.perf_counter() - t1
        self.counts["St"] += 1
        self.counts["boxes"] += self.boxes_tapered if tapered else self.boxes
        return out

    def H(self, u):
        """``W S S^T W u``."""
        call = self._H_call
        if call and call.ok:
            t0 = time.perf_counter()
            out = _ready(call(_ready(u), self._arrays()))
            self._counted(1, time.perf_counter() - t0)
            return out
        if self.window is None:
            out = self.S(self.St(_ready(u)))
        else:
            out = _ready(self.window * self.S(self.St(_ready(self.window * u), True), True))
        if call:                                                 # the first product: is the one call the same?
            call.same((u, self._arrays()), out)
        return out

    def sqrt(self, xi):
        """``W S xi``: of covariance ``H~`` for a white ``xi`` on the nodes."""
        if self.window is None:
            return self.S(_ready(xi))
        call = self._sqrt_call
        if call and call.ok:
            t0 = time.perf_counter()
            out = _ready(call(_ready(xi), self._arrays()))
            self.timers["S"] += time.perf_counter() - t0
            self.counts["S"] += 1
            self.counts["boxes"] += self.boxes_tapered
            return out
        out = _ready(self.window * self.S(_ready(xi), True))
        if call:
            call.same((xi, self._arrays()), out)
        return out

    def memory(self):
        """:meth:`CellPC.memory`, whose ``boxes`` and ``batches`` are those of a product with ``S`` or
        ``S^T`` alone, and ``boxes_tapered``, ``batches_tapered``: those of each of the two in ``H~``
        and of ``sqrt`` (fewer where tiles are left out: what a driver prints for every rank)."""
        batches = len(self.cell.pieces if self._tapered is None else self._tapered)
        return dict(self.cell.memory(), boxes_tapered=self.boxes_tapered, batches_tapered=batches,
                    calls="whole" if self._H_call else "loops" if self.cell.mode == "loops" else "products")


# ------------------------------------------------- a factor that follows the taper: levels
#: ``a + w y`` as one compiled operation: what the pieces and a compiled whole both round the same way
_axpy = jax.jit(lambda a, w, y: a + w * y)


def level_weights(window, levels, past=True):
    """The weights of the levels at every node of a taper ``W`` (host), ``(len(levels) + 1, ..)``:
    the hats in ``u = 1 / W`` of the node's own ``u`` between the levels ``u_l`` (1 at a node
    where ``W`` is one), and last the prior's level: the nodes where ``W`` is zero, and those
    beyond the last level if the levels reach past the knees (``past``), which otherwise take
    the last level.  They sum to one at every node."""
    W = np.asarray(window, dtype=np.float64)
    u = np.where(W > 0, 1.0 / np.where(W > 0, W, 1.0), np.inf)
    us = [float(v) for v in levels]
    V = np.zeros((len(us) + 1,) + W.shape)
    inside = u <= us[-1]
    V[0][inside & (u <= us[0])] = 1.0
    for l in range(1, len(us)):
        m = inside & (u > us[l - 1]) & (u <= us[l])
        V[l][m] = (u[m] - us[l - 1]) / (us[l] - us[l - 1])
        V[l - 1][m] = 1.0 - V[l][m]
    V[-1] = np.isinf(u) | (~inside & bool(past))
    V[len(us) - 1] += ~inside & ~np.isinf(u) & (not past)
    return V


class LevelFactor:
    r"""The factor of the preconditioner ``"levels"``: a square root ``Bq`` of an approximate inverse
    of ``W S S^T W + R`` whose symbol follows the taper ``W`` from node to node.

    The local symbol of the inverse's root is ``q(x, xi) = (w(x)^2 s_k(xi) + r(xi))^(-1/2)``.  In
    ``u = 1 / w`` it is ``u (s + r u^2)^(-1/2)``: linear in ``u`` below the knee ``u* = sqrt(s /
    r)``, where the data hold the parameter and ``q = 1 / (w sqrt(s))``, and constant above it,
    where the prior does.  The factor of ``"symbol"`` interpolates ``q`` between the corners of a
    tile by the hats of the sample lattice with the taper read AT the corners; between a corner
    where ``w`` is zero and one where it is one that hands the prior's variance to nodes the data
    hold, and the solve is not much better for it than with the prior alone.  Here

    .. code-block:: text

        Bq u = sum_k sum_l w_k(x) v_l(u(x)) (q_(k,l) * u) + v_oo(x) F u

    with levels ``u_l = 1, ratio, ratio^2, ..`` up to the first above ``top`` times the largest
    knee, ``q_(k,l)`` the kernel of ``(s_k / u_l^2 + r)^(-1/2)`` (cut as the other factors'),
    ``v_l`` the hats in ``u`` of the node's own ``u(x)`` (:func:`level_weights`): exact below the
    knee and above it, low at it by 16 % at most for a ratio of 4, the safe side.  A node where
    the taper is zero, or beyond the last level, has the prior's level, which is no cut kernel
    but the prior's own half solve, exactly: ``F = A^-1 L^(1/2)`` for ``R = A L^-1 A``, as
    ``solve_A(lhalf *)`` with ``lhalf`` the square root of the lumped mass, read off the prior
    (``apply_A(sample(1))``); its transpose is ``lhalf * solve_A()``.  ``Bq Bq^T`` is symmetric
    positive by construction.

    A tile's terms are the pairs (kernel, level) of the kernels that DIFFER among its corners (a
    corner beyond the outermost sample points has the nearest one's) and of the levels with a
    weight on its nodes; a term's weight is the sum of the hats of the corners that share the
    kernel times the level's hat.  A product transforms the boxes of the tiles that have a term
    once, every term's part once the other way, and adds a tile's terms in a fixed order; the
    terms of ``chunk`` tiles are transformed together (:func:`batch_bound`).  A tile on which the
    taper is one has a level and its 8 corners, as a tile of ``"symbol"``'s factor has; one on
    which it is zero has no term at all.

    It has the members of :class:`PSFOperator` that the solve uses: ``S``, ``St``, their traced
    forms, ``counts``, ``timers``, ``memory()``.  On several ranks every rank has the terms of
    its own tiles and the exchanges are ``S``'s and ``S^T``'s; the levels are the same on all of
    them (the largest knee is reduced over the ranks when the factor is made: collective), and
    the prior's level is the prior's own solve on its ranks.
    """

    def __init__(self, psf, prior, rho_q=None, kaiser=5.0, ratio=4.0, top=2.0, max_boxes=None):
        k, cell, halo = psf.kernels, psf.cell, psf.halo
        d = self.d = cell.d
        self.psf, self.prior, self.halo, self.cell, self.dtype = psf, prior, halo, cell, psf.dtype
        L, m = cell.L, cell.m
        rho_q = k.rho_g if rho_q is None else int(rho_q)
        if rho_q > min(cell.h0, m - L - cell.h0):
            raise ValueError("kernels of radius %d do not fit the margin of the operator's boxes (%d of %d nodes "
                             "around a cell of %d)" % (rho_q, min(cell.h0, m - L - cell.h0), m, L))
        for name in ("sample", "solve_A", "apply_A"):
            if not hasattr(prior, name):
                raise ValueError("the preconditioner \"levels\" asks the prior for sample, solve_A and apply_A "
                                 "on owned arrays: its level is the prior's own half solve")
        box = k.box if k.box else _fast_length(3 * max(k.rho_g, rho_q) + 4)
        r_box = np.asarray(prior.symbols(box)[1], dtype=np.float64)
        g_flat = psf.g.reshape((-1,) + psf.g.shape[-d:])
        # ---- the levels, the same on every rank: from the largest knee of all the kernels
        knee = max([0.0] + _in_batches(g_flat, box, d, max_boxes,
                                       lambda a, g: float(jnp.max(symbol(g, box, d) ** 2 / _put(r_box)))))
        knee = float(np.sqrt(knee))
        comm = getattr(halo, "comm", None)
        if not _holds_all(halo) and comm is not None:
            knee = allreduce(comm, knee, op=MPI.MAX)
        W = np.ones(cell.own_shape) if psf.window is None else np.asarray(psf.window, dtype=np.float64)
        # The first level is the least u = 1 / W of all ranks' nodes, one for a taper (which is at most
        # one), less where the window is larger (an amplitude in it, --psf-amplitude): a node below the
        # first level would take that level's kernel, larger than its own.
        wmax = float(np.max(np.broadcast_to(W, cell.own_shape))) if W.size else 0.0
        if not _holds_all(halo) and comm is not None:
            wmax = allreduce(comm, wmax, op=MPI.MAX)
        first = min(1.0, 1.0 / wmax) if wmax > 0.0 else 1.0
        #: the largest knee ``sqrt(s / r)`` of all the kernels, and the levels
        self.knee, self.levels = knee, [first]
        while self.levels[-1] < top * knee:
            self.levels.append(self.levels[-1] * float(ratio))
        nl = len(self.levels)
        V = level_weights(W, self.levels)
        #: the share of the rank's nodes that each level has a weight on, the prior's last
        self.share = [float((v != 0).mean()) for v in V]
        # ---- the prior's level
        self.prior_level = bool(V[-1].any())
        one = _put(np.ones(cell.own_shape), self.dtype)
        lhalf = _ready(prior.apply_A(prior.sample(one)))
        # (collective, and made on every rank whether it has such a node or not)
        check = prior.apply_A(prior.sample(lhalf)) - lhalf * lhalf
        self.mass_error = float(np.sqrt(halo.dot(check, check) / halo.dot(lhalf * lhalf, lhalf * lhalf)))
        if not self.mass_error < 1e-5:
            raise ValueError("apply_A(sample(.)) of this prior is not a diagonal (%.1e off): the preconditioner "
                             "\"levels\" reads the square root of the lumped mass off it" % self.mass_error)

        # ---- the terms of every tile, in the order of the cell's batches
        def tiles_of(a):                                         # (own..) to (Nc.., L..), zero beyond the rank's nodes
            a = np.pad(a, cell._own_pad).reshape([n for j in range(d) for n in (cell.Nc[j], L)])
            return a.transpose([2 * j for j in range(d)] + [2 * j + 1 for j in range(d)])

        Vt = [tiles_of(V[l]) for l in range(nl)]
        hats = np.asarray(cell.hats, dtype=np.float64)
        hat_tbl, hat_ids, v_tbl, v_ids, pair_ids, terms = [], {}, [], {}, {}, []
        self._place, offset = [], 0
        for box_at, where, T, ix in cell.pieces:
            ix = np.asarray(ix)
            n = int(np.prod(T))
            self._place.append((box_at, where, tuple(T), offset, n))
            starts = [range(s.start, s.stop, s.step) for s in where]
            for f, pos in enumerate(np.ndindex(*T)):
                t = tuple(starts[j][pos[j]] for j in range(d))
                present = [l for l in range(nl) if Vt[l][t].any()]
                if not present:
                    continue
                kid = [int(ix[(c,) + pos]) for c in range(ix.shape[0])]
                for K in sorted(set(kid)):
                    mask = tuple(c for c in range(len(kid)) if kid[c] == K)
                    if mask not in hat_ids:
                        hat_ids[mask] = len(hat_tbl)
                        hat_tbl.append(hats[list(mask)].sum(axis=0))
                    for l in present:
                        key = Vt[l][t].tobytes()
                        if key not in v_ids:
                            v_ids[key] = len(v_tbl)
                            v_tbl.append(Vt[l][t])
                        terms.append((offset + f, pair_ids.setdefault((l, K), len(pair_ids)), hat_ids[mask], v_ids[key]))
            offset += n
        terms = np.array(terms, dtype=np.int64).reshape(-1, 4)
        #: the tiles of the rank's cells, those with a term, the terms, the (kernel, level) pairs that differ
        self.tiles, self.active, self.terms, self.pairs = offset, int(np.unique(terms[:, 0]).size), terms.shape[0], len(pair_ids)
        #: the boxes a product transforms: a tile's once, every term's once
        self.boxes = self.boxes_tapered = self.active + self.terms
        # ---- the symbols of the pairs on the operator's box, level by level
        win = cut_window(rho_q, d, "kaiser:%g" % kaiser)
        rows = [None] * len(pair_ids)
        for l in range(nl):
            Ks = sorted(K for (ll, K) in pair_ids if ll == l)
            if Ks:
                q = inverse_sqrt_kernels(g_flat[Ks], np.full(len(Ks), 1.0 / self.levels[l] ** 2), r_box, box, rho_q, d, win,
                                         False, max_boxes)
                s = np.concatenate(_in_batches(q, m, d, max_boxes, lambda a, g: np.asarray(symbol(g, m, d))))
                for K, row in zip(Ks, s):
                    rows[pair_ids[(l, K)]] = row
        shape = (m,) * (d - 1) + (m // 2 + 1,)
        sym = _put(np.stack(rows) if rows else np.zeros((1,) + shape), self.dtype)
        # ---- chunks of tiles whose terms are transformed together
        tiles_with = np.unique(terms[:, 0])
        count = np.bincount(terms[:, 0], minlength=max(self.tiles, 1))
        terms = terms[np.argsort(terms[:, 0], kind="stable")]
        bound, chunks, a = batch_bound(m, d, max_boxes), [], 0
        while a < tiles_with.size:
            b, n = a, 0
            while b < tiles_with.size and (b == a or n + count[tiles_with[b]] <= bound):
                n += count[tiles_with[b]]
                b += 1
            ids = tiles_with[a:b]
            sel = terms[np.isin(terms[:, 0], ids)]
            rel = np.searchsorted(ids, sel[:, 0])
            pos = np.full((ids.size, int(count[ids].max())), sel.shape[0], dtype=np.int64)       # (past the last: a zero)
            filled = np.zeros(ids.size, dtype=np.int64)
            for j, r in enumerate(rel):
                pos[r, filled[r]] = j
                filled[r] += 1
            chunks.append(tuple(_put(x.astype(np.int32)) for x in (ids, rel, sel[:, 1], sel[:, 2], sel[:, 3], pos)))
            a = b
        # what a compiled call is given beside its field (not kept in it)
        hat_tbl = np.stack(hat_tbl) if hat_tbl else np.zeros((1,) + (L,) * d)
        v_tbl = np.stack(v_tbl) if v_tbl else np.zeros((1,) + (L,) * d)
        self._c = dict(sym=sym, hat=_put(hat_tbl, self.dtype), v=_put(v_tbl, self.dtype), chunks=chunks,
                       V_inf=_put(V[-1], self.dtype), lhalf=lhalf, live=cell._live)
        self._ca, self._ct = jax.jit(self._a), jax.jit(self._t)
        self.timers = dict(S=0.0, St=0.0, exchange=0.0)
        self.counts = dict(S=0, St=0, boxes=0)

    # ---- a rank's own part, as it is traced
    def _a(self, ext, c):
        """The tiles' part of ``Bq u`` on the rank's own nodes from ``u`` in the halo's array."""
        cell, d = self.cell, self.d
        L, m, h0 = cell.L, cell.m, cell.h0
        axes = tuple(range(-d, 0))
        crop = (Ellipsis,) + (slice(h0, h0 + L),) * d
        up = jnp.pad(ext[cell._in], cell._in_pad)
        boxes = jnp.concatenate([cell._to_boxes(up[box], T, m).reshape((-1,) + (m,) * d) for box, _, T, _, _ in self._place])
        parts = jnp.zeros((max(self.tiles, 1),) + (L,) * d, ext.dtype)
        for ids, rel, pair, hat, v, pos in c["chunks"]:
            U = jnp.fft.rfftn(boxes[ids], axes=axes)
            Y = jnp.fft.irfftn(U[rel] * c["sym"][pair], s=(m,) * d, axes=axes)[crop] * (c["hat"][hat] * c["v"][v])
            # (the terms are made, then added: a compiled whole may otherwise work a term's last product into
            # the sum, one rounding for two, in one place and not in another)
            Y = jax.lax.optimization_barrier(Y)
            Y = jnp.concatenate([Y, jnp.zeros((1,) + Y.shape[1:], Y.dtype)])
            acc = Y[pos[:, 0]]
            for j in range(1, pos.shape[1]):
                acc = acc + Y[pos[:, j]]
            parts = parts.at[ids].set(acc, unique_indices=True)
            boxes = cell._after(boxes, parts, c["live"])         # the next chunk waits for this one (CellPC._after)
        cells = jnp.zeros(tuple(cell.Nc) + (L,) * d, ext.dtype)
        for _, where, T, at, n in self._place:
            cells = cells.at[where].set(parts[at:at + n].reshape(T + (L,) * d))
        return cell._from_boxes(cells, cell.Nc, L)[cell._own]

    def _t(self, v, c):
        """The tiles' part of ``Bq^T v`` of ``v`` on the rank's own nodes, in the halo's array."""
        cell, d = self.cell, self.d
        L, m, h0 = cell.L, cell.m, cell.h0
        axes = tuple(range(-d, 0))
        pad = [(0, 0)] + [(h0, m - L - h0)] * d
        cells = cell._to_boxes(jnp.pad(v, cell._own_pad), cell.Nc, L)
        tiles = jnp.concatenate([cells[where].reshape((-1,) + (L,) * d) for _, where, _, _, _ in self._place])
        back = jnp.zeros((max(self.tiles, 1),) + (m,) * d, v.dtype)
        for ids, rel, pair, hat, w, pos in c["chunks"]:
            X = jnp.fft.rfftn(jnp.pad(tiles[ids][rel] * (c["hat"][hat] * c["v"][w]), pad), axes=axes) * c["sym"][pair]
            X = jax.lax.optimization_barrier(X)
            X = jnp.concatenate([X, jnp.zeros((1,) + X.shape[1:], X.dtype)])
            acc = X[pos[:, 0]]
            for j in range(1, pos.shape[1]):
                acc = acc + X[pos[:, j]]
            back = back.at[ids].set(jnp.fft.irfftn(acc, s=(m,) * d, axes=axes), unique_indices=True)
            tiles = cell._after(tiles, back, c["live"])
        z = jnp.zeros(cell.padded_shape, v.dtype)
        for box, _, T, at, n in self._place:
            z = z.at[box].add(cell._from_boxes(back[at:at + n].reshape(T + (m,) * d), T, m))
        return jnp.pad(z[cell._out], cell._out_pad)

    def _arrays(self):
        return self._c

    def _S(self, u, c, tapered=False):
        """:meth:`S` as it is traced (a rank that holds the whole lattice)."""
        out = self._a(self.halo.fill(u), c)
        if not self.prior_level:
            return out
        # The prior's transforms wait for the tiles' (CellPC._after).  Its result is not worked into the sum,
        # nor the sum into what is made of it next: both are arrays of their own, as they are for the pieces.
        y = self.prior.solve_A(self.cell._after(c["lhalf"] * u, out, c["live"]))
        out, y = jax.lax.optimization_barrier((out, y))
        return jax.lax.optimization_barrier(out + c["V_inf"] * y)

    def _St(self, v, c, tapered=False):
        out = self.halo.add_back(self._t(v, c))
        if not self.prior_level:
            return out
        y = self.prior.solve_A(self.cell._after(c["V_inf"] * v, out, c["live"]))
        out, y = jax.lax.optimization_barrier((out, y))
        return jax.lax.optimization_barrier(out + c["lhalf"] * y)

    def _counted(self, products, seconds=0.0):
        self.timers["S"] += 0.5 * seconds
        self.timers["St"] += 0.5 * seconds
        self.counts["S"] += products
        self.counts["St"] += products
        self.counts["boxes"] += 2 * products * self.boxes

    # ---- the two products, through the halo
    def apply(self, ext):
        """The compiled call of the tiles' part of ``Bq``."""
        return self._ca(ext, self._c)

    def apply_t(self, v):
        return self._ct(v, self._c)

    def S(self, u, tapered=False):
        """``Bq u``."""
        c = self._c
        t0 = time.perf_counter()
        ext = _ready(self.halo.fill(u))
        t1 = time.perf_counter()
        out = _ready(self.apply(ext))
        if self.prior_level:
            out = _ready(_axpy(out, c["V_inf"], self.prior.solve_A(_ready(c["lhalf"] * u))))
        self.timers["exchange"] += t1 - t0
        self.timers["S"] += time.perf_counter() - t1
        self.counts["S"] += 1
        self.counts["boxes"] += self.boxes
        return out

    def St(self, v, tapered=False):
        """``Bq^T v``."""
        c = self._c
        t0 = time.perf_counter()
        ext = _ready(self.apply_t(v))
        t1 = time.perf_counter()
        out = _ready(self.halo.add_back(ext))
        t2 = time.perf_counter()
        if self.prior_level:
            out = _ready(_axpy(out, c["lhalf"], self.prior.solve_A(_ready(c["V_inf"] * v))))
        self.timers["St"] += t1 - t0 + time.perf_counter() - t2
        self.timers["exchange"] += t2 - t1
        self.counts["St"] += 1
        self.counts["boxes"] += self.boxes
        return out

    def memory(self):
        size = self.dtype.itemsize
        return dict(symbols_bytes=int(np.prod(self._c["sym"].shape)) * size, box=self.cell.m, cell=self.cell.L,
                    tiles=self.tiles, tiles_with_a_term=self.active, terms=self.terms, pairs=self.pairs,
                    boxes=self.boxes, chunks=len(self._c["chunks"]), levels=list(self.levels), knee=self.knee,
                    share=list(self.share), prior_level=self.prior_level)


# ------------------------------------------------------------------------------- H~ + R
@jax.jit
def _step(x, r, p, Ap, alpha):
    return x + alpha * p, r - alpha * Ap


@jax.jit
def _direction(z, p, beta):
    return z + beta * p


class ApproximateHessian(hm.Operator):
    """``H~ + R``: the Hessian of the negative log posterior with the point spread function
    operator in the place of the misfit's part.

    On owned lattice arrays on the device: ``apply(a)``, ``solve_lattice(b) -> (x,
    iterations)`` by conjugate gradients, ``sqrt(xi)`` (:func:`draw`).  A parameter vector
    is converted at the two ends only, through ``lmap``, in the methods the library's
    algorithms call: ``init_vector``, ``mult``, ``solve`` (it overwrites its output),
    ``inner``, ``sqrt_mult``.  So one object is the initial inverse Hessian of L-BFGS, the
    preconditioner of Newton-CG and the Hessian of the Laplace approximation.

    ``precondition`` of the conjugate gradients:

    * ``"prior"``: a solve with ``R``.
    * ``"symbol"``: ``Bq Bq^T``, ``Bq = sum_k w_k q_k *`` with ``q_k`` the kernel of ``(W_k^2
      s_k + r)^(-1/2)`` cut at ``rho_q`` by a Kaiser window of parameter ``kaiser`` (with a
      cosine taper its symbol changes sign and the iteration stalls).  No solve with the
      prior.
    * ``"split"``: ``N R^-1 N^T``, ``N``'s kernels those of ``(1 + W_k^2 s_k / r)^(-1/2)``: the
      prior's part exactly, a solve with it an iteration.  (With the prior's half solves on
      the outside, ``R^-1/2 N N^T R^-T/2``, the count is 118 for 138; that form asks the prior
      for its factor.)
    * ``"none"``: the reference code's way.
    * ``"levels"``: ``Bq Bq^T`` with a factor whose symbol follows the taper from node to node
      (:class:`LevelFactor`; ``levels`` is the ratio of its levels, ``top`` where they end).  What
      the symbol form is where there is no taper, where there is one: on the kernels of two
      inversions (30 625 and 444 185 parameters, tapers that are one on 41 and 30 % of the nodes)
      the solve to 1e-6 takes 57 iterations with it where the prior takes 856, ``"symbol"`` 758
      and ``"split"`` 589 on the first.  It asks the prior for ``sample``, ``solve_A`` and
      ``apply_A`` on owned arrays.

    On the test Hessian (largest eigenvalue against the prior 1e4) the solve to 1e-6 takes
    1960, 579, 167 and 138 iterations with none, the prior, the symbol and the split form
    on 115 thousand nodes in two dimensions, and 611, 171 and 140 with the last three on
    4.8 million in three: the count is set by that eigenvalue, not by the lattice's size.
    In single precision the two symbol forms take as many iterations as in double and the
    prior a fifth more.

    ``counts`` and ``timers``: solves, iterations, products with ``H~ + R``, and the seconds
    in the products with ``H~``, with ``R``, in the preconditioner and in the solves whole.

    ``whole`` (:data:`WHOLE`; the operator's by default): on a rank that holds the whole lattice
    an iteration of the solve is ONE compiled call (the preconditioner, the new direction, the
    product with ``H~ + R``, the step and the three inner products) and one transfer of three
    numbers to the host, which decides as before.  The same iterates to the bit and the same
    count: the first two iterations of the first solve are made both ways, and the one call is
    kept if it gave the bits of the pieces (:meth:`_try_whole`; it does with
    ``prior.LatticePrior``, whose functions are compiled ones; with a prior whose solve is a
    sequence of operations run from Python a compiled whole rounds otherwise, and the pieces
    stay).  The seconds of such iterations are in ``timers["whole"]`` and not in ``H``, ``R``,
    ``precondition``, which one call does not tell apart (``HIPPYMFEM_PSF_WHOLE=products`` for
    those).
    """

    def __init__(self, psf, prior, precondition="prior", rtol=1e-8, maxit=2000, lmap=None, rho_q=None,
                 kaiser=5.0, max_boxes=None, whole=None, levels=4.0, top=2.0):
        self.psf, self.prior, self.lmap = psf, prior, lmap
        self.rtol, self.maxit = float(rtol), int(maxit)
        self.precondition = precondition
        self.counts = dict(solves=0, iterations=0, products=0)
        self.timers = dict(H=0.0, R=0.0, precondition=0.0, solve=0.0, whole=0.0)
        self._whole = psf.whole if whole is None else _mode(whole) == "whole" and _holds_all(psf.halo)
        if precondition in ("symbol", "split"):
            self.factor = self._factor(psf.kernels.rho_g if rho_q is None else int(rho_q), kaiser, max_boxes)
        elif precondition == "levels":
            self.factor = LevelFactor(psf, prior, rho_q, kaiser, levels, top, max_boxes)
        elif precondition not in ("prior", "none"):
            raise ValueError("no preconditioner %r" % (precondition,))
        self._template = None if lmap is None else lmap.from_lattice(np.zeros(psf.halo.own_shape))
        self._dot = _device_dot(psf.halo) if self._whole else None
        self._step_call = _Whole(self._iteration) if self._dot is not None else None
        self._fac = getattr(self, "factor", None)

    def _factor(self, rho_q, kaiser, max_boxes):
        """``Bq`` or ``N`` as an operator of the same kind as ``S``, on the lattice of EVERY corner
        of this rank's cells: beyond the outermost sample points the nearest one's symbol with
        the taper at the corner's own place, so that the factor becomes the prior's where the
        taper has taken the data away.

        It is an operator without a window, and its products transform every tile: what it is
        applied to is a residual of ``H~ + R``, which is not zero where the taper is, and where the
        taper is zero the factor is the prior's inverse root (``"symbol"``) or the identity
        (``"split"``), not zero.  So the tiles that ``H~`` leaves out (:meth:`CellPC.tapered`) stay in
        the preconditioner, and the first block of a direction has its layer of boxes more there."""
        psf, k = self.psf, self.psf.kernels
        d = k.sample.d
        tiles = [_Tiles(k.sample, j, lo, hi, k.weights) for j, (lo, hi) in enumerate(_own(psf.halo))]
        g = psf.g
        for j, t in enumerate(tiles):
            at = np.clip(t.corner + np.arange(t.corners), k.first[j], k.first[j] + g.shape[j] - 1)
            g = np.take(g, at - k.first[j], axis=j)
        n = tuple(t.corners for t in tiles)
        w2 = np.ones(n) if k.corner_window is None else k.corner_window ** 2
        q = inverse_sqrt_kernels(g, w2, self.prior.symbols(k.box)[1], k.box, rho_q, d,
                                 cut_window(rho_q, d, "kaiser:%g" % kaiser), self.precondition == "split",
                                 max_boxes)
        return PSFOperator(psf.halo, Kernels(k.sample, k.weights, q, [t.corner for t in tiles]),
                           dtype=psf.dtype, max_boxes=max_boxes,
                           whole="loops" if psf.cell.mode == "loops" else "whole" if self._whole else "products",
                           join=psf.cell.joined)

    # ---- on owned lattice arrays
    def apply(self, a):
        """``(H~ + R) a``."""
        a = _ready(a)
        t0 = time.perf_counter()
        Ha = self.psf.H(a)
        t1 = time.perf_counter()
        out = _ready(Ha + self.prior.apply_R(a))
        self.timers["H"] += t1 - t0
        self.timers["R"] += time.perf_counter() - t1
        self.counts["products"] += 1
        return out

    def _precondition(self, r):
        if self.precondition == "none":
            return r
        t0 = time.perf_counter()
        if self.precondition == "prior":
            z = self.prior.solve_R(r)
        elif self.precondition in ("symbol", "levels"):
            z = self.factor.S(self.factor.St(r))
        else:
            z = self.factor.S(self.prior.solve_R(self.factor.St(r)))
        z = _ready(z)
        self.timers["precondition"] += time.perf_counter() - t0
        return z

    def _iteration(self, x, r, p, rz, first, c, cf):
        """An iteration of :meth:`solve_lattice` as it is traced, on a rank that holds the whole
        lattice: from the iterate, the residual, the direction and ``r.z`` of the iteration before
        (``first``: there is none) to the new three and ``(r.z, p.(H~ + R) p, r.r)`` in double.
        ``c``, ``cf``: the arrays of the operator and of the preconditioner's factor
        (:meth:`PSFOperator._arrays`).

        The same operations as the pieces, and the same bits: a result that is a product is kept
        from being worked into the sum that follows it (a fused multiply-add rounds once where
        the pieces round twice), which is what the barriers are for."""
        psf, prior, dot, pre = self.psf, self.prior, self._dot, self.precondition
        if pre == "none":
            z = r
        elif pre == "prior":
            z = prior.solve_R(r)
        elif pre in ("symbol", "levels"):
            z = self.factor._S(self.factor._St(r, cf), cf)
        else:
            z = self.factor._S(prior.solve_R(self.factor._St(r, cf)), cf)
        z = jax.lax.optimization_barrier(z)
        rz_new = dot(r, z)
        p = jnp.where(first, z, z + (rz_new / rz).astype(z.dtype) * p)
        Hp, Rp = jax.lax.optimization_barrier((psf._H(p, c), prior.apply_R(p)))
        Ap = Hp + Rp
        pAp = dot(p, Ap)
        alpha = (rz_new / pAp).astype(z.dtype)
        x, r = x + alpha * p, r - alpha * Ap
        return x, r, p, jnp.stack([rz_new, pAp, dot(r, r)])

    def _try_whole(self, b):
        """Whether an iteration in one compiled call may stand for the pieces: the first two iterations
        from ``b``, made in pieces and by the call, each from the pieces' state, are the same to the
        bit, the three numbers with them (the device's inner product is then the halo's).  Decided
        once; nothing is decided where the curvature of ``b`` is not positive.  The pieces go first, so
        that what the prior and the operators make at their first call is not made inside a trace
        (a table made there would be an array of the trace, of no use to the call after).  Counts and
        timers are left as they were: these four iterations are no part of a solve."""
        call, dot, factor = self._step_call, self.psf.halo.dot, self._fac
        kept = [(m, dict(m)) for o in (self, self.psf, factor) if o is not None for m in (o.counts, o.timers)]
        c = self.psf._arrays()
        cf = None if factor is None else factor._arrays()
        x = p = jnp.zeros_like(b)
        r, rz, first = b, 1.0, True
        try:
            for _ in range(2):
                z = self._precondition(r)
                new_rz = dot(r, z)
                new_p = z if first else _direction(z, p, new_rz / rz)
                Ap = self.apply(new_p)
                pAp = dot(new_p, Ap)
                if not pAp > 0.0:
                    return
                new_x, new_r = _step(x, r, new_p, Ap, new_rz / pAp)
                numbers = jnp.asarray([new_rz, pAp, dot(new_r, new_r)])
                if not call.same((x, r, p, np.float64(rz), np.bool_(first), c, cf), (new_x, new_r, new_p, numbers),
                                 last=not first):
                    return
                x, r, p, rz, first = new_x, new_r, new_p, new_rz, False
        finally:
            for mapping, was in kept:
                mapping.update(was)

    def _solve_whole(self, b, floor, maxit):
        """The iterations of :meth:`solve_lattice`, each one compiled call: ``(x, iterations)``."""
        c = self.psf._arrays()
        cf = None if self._fac is None else self._fac._arrays()
        x, r, p, rz, it = jnp.zeros_like(b), b, jnp.zeros_like(b), 1.0, 0
        t0 = time.perf_counter()
        for it in range(1, maxit + 1):
            new_x, new_r, p, numbers = self._step_call(x, r, p, np.float64(rz), np.bool_(it == 1), c, cf)
            rz, pAp, rr = (float(v) for v in np.asarray(numbers))
            self.counts["products"] += 1
            if not pAp > 0.0:
                break
            x, r = new_x, new_r
            if np.sqrt(rr) <= floor or it == maxit:
                break
        x = _ready(x)
        self.timers["whole"] += time.perf_counter() - t0
        self.psf._counted(it)
        if cf is not None:
            self._fac._counted(it)
        return x, it

    def solve_lattice(self, b, rtol=None, maxit=None):
        """``(H~ + R) x = b`` by preconditioned conjugate gradients from zero, to a residual of
        ``rtol`` of ``b`` in the 2-norm (the same test whatever the preconditioner).  Returns
        ``x`` and the iterations.  It stops where the curvature is not positive, which in
        exact arithmetic it always is: single precision at its floor."""
        rtol = self.rtol if rtol is None else rtol
        maxit = self.maxit if maxit is None else maxit
        dot = self.psf.halo.dot
        t0 = time.perf_counter()
        b = _ready(b)
        x, r = jnp.zeros_like(b), b
        floor = rtol * np.sqrt(dot(b, b))
        it = 0
        if self._step_call and self._step_call.ok is None and floor > 0.0 and maxit > 0:
            self._try_whole(b)
        if self._step_call and self._step_call.ok and floor > 0.0 and maxit > 0:
            x, it = self._solve_whole(b, floor, maxit)
        elif floor > 0.0:
            z = self._precondition(r)
            p, rz = z, dot(r, z)
            for it in range(1, maxit + 1):
                Ap = self.apply(p)
                pAp = dot(p, Ap)
                if not pAp > 0.0:
                    break
                x, r = _step(x, r, p, Ap, rz / pAp)
                if np.sqrt(dot(r, r)) <= floor or it == maxit:
                    break
                z = self._precondition(r)
                rz, old = dot(r, z), rz
                p = _direction(z, p, rz / old)
        x = _ready(x)
        self.timers["solve"] += time.perf_counter() - t0
        self.counts["solves"] += 1
        self.counts["iterations"] += it
        return x, it

    def sqrt(self, xi):
        """``W S xi``, of covariance ``H~``: the misfit's half of the right-hand side of a draw."""
        return self.psf.sqrt(xi)

    # ---- the library's face
    def _lattice(self, v):
        return _put(self.lmap.to_lattice(v), self.psf.dtype)

    def _vector(self, a, v):
        self.lmap.from_lattice(np.asarray(a, dtype=np.float64), v)

    def init_vector(self, x, dim):
        hm.init_vector_like(x, self._template)

    def mult(self, x, y):
        self._vector(self.apply(self._lattice(x)), y)

    def solve(self, x, b):
        a, it = self.solve_lattice(self._lattice(b))
        self._vector(a, x)
        return it

    def sqrt_mult(self, xi, y):
        self._vector(self.sqrt(self._lattice(xi)), y)


def draw(hessian, s_prior, xi):
    """A sample of ``N(0, (H~ + R)^-1)`` and the iterations of its solve, from a sample
    ``s_prior`` of the prior and a white field ``xi`` on the nodes: ``x = (H~ + R)^-1 (R
    s_prior + W S xi)``.  The right-hand side has the covariance ``R + H~``, so ``x`` has
    that of the inverse exactly: the recipe of the ice sheet application's
    ``LocalPosterior.draw``, with no adjoint solve for the misfit's half because ``H~`` is
    given as a square.  (The symbol-space square root alone, ``Bq z``, is no sample: with the
    true Hessian's products its ``x^T (H + R) x / n`` is 1.36 on the test Hessian where a
    draw's is 0.993 and the exact posterior's is 1.)"""
    return hessian.solve_lattice(hessian.prior.apply_R(s_prior) + hessian.sqrt(xi))


def check_action(apply, vectors, products, dot):
    """``|H~ v - H v| / |H v|`` for vectors the approximation was not made from, given with
    the Hessian's own products (fields of the same sweeps as the combs): the ice sheet
    application's test of its sparse Hessian.  White vectors weigh every scale, samples of
    the prior the scales that matter."""
    out = []
    for v, Hv in zip(vectors, products):
        e = apply(v) - Hv
        out.append(float(np.sqrt(dot(e, e) / dot(Hv, Hv))))
    return out


# ------------------------------------------------- the same operator, global transforms
def weight_fields(sample, weights="hat"):
    """The weights as fields, ``(K_1, .., K_d, *shape)``, by their formula: the tensor hats
    (or quadratic B-splines) of the sample lattice, a point beyond the outermost ones
    counted as the nearest of them.  For :class:`GlobalPC`."""
    rows = []
    for k in range(sample.d):
        t = (np.arange(sample.shape[k]) - sample.first[k]) / float(sample.step)
        w = np.zeros((sample.K[k], sample.shape[k]))
        for j in range(int(np.floor(t.min())) - 2, int(np.ceil(t.max())) + 3):
            y = np.abs(t - j)
            if weights == "hat":
                b = np.maximum(0.0, 1.0 - y)
            else:
                b = np.where(y <= 0.5, 0.75 - y ** 2, 0.5 * np.maximum(0.0, 1.5 - y) ** 2)
            w[min(max(j, 0), sample.K[k] - 1)] += b
        rows.append(w)
    out = np.ones(sample.K + sample.shape)
    for k, w in enumerate(rows):
        out = out * w.reshape([sample.K[j] if j == k else 1 for j in range(sample.d)]
                              + [sample.shape[j] if j == k else 1 for j in range(sample.d)])
    return out


class GlobalPC:
    """``sum_k w_k (g_k * u)`` and its transpose on a whole lattice by transforms of the whole
    (zero-padded) field, a forward and an inverse one a sample point: the reference code's
    ``RealPDOLowRankSymbol`` with any weights, on the host.  What :class:`CellPC` is checked
    against; the padding makes the convolution the one that does not wrap around."""

    def __init__(self, shape, weights, kernels):
        import scipy.fft as sfft

        self.fft, self.shape = sfft, tuple(shape)
        d = len(shape)
        rho = (kernels.shape[-1] - 1) // 2
        self.big = tuple(sfft.next_fast_len(n + rho, real=True) for n in shape)
        self.w = weights.reshape((-1,) + self.shape)
        self.sym = []
        for ker in kernels.reshape((-1,) + kernels.shape[d:]):
            box = np.zeros(self.big)
            box[(slice(0, 2 * rho + 1),) * d] = ker
            self.sym.append(sfft.rfftn(np.roll(box, (-rho,) * d, axis=tuple(range(d)))).real)
        self.crop = tuple(slice(0, n) for n in shape)

    def apply(self, u):
        U = self.fft.rfftn(u, s=self.big)
        out = np.zeros(self.shape)
        for w, s in zip(self.w, self.sym):
            out += w * self.fft.irfftn(U * s, s=self.big)[self.crop]
        return out

    def apply_t(self, v):
        acc = 0.0
        for w, s in zip(self.w, self.sym):
            acc = acc + self.fft.rfftn(w * v, s=self.big) * s
        return self.fft.irfftn(acc, s=self.big)[self.crop]

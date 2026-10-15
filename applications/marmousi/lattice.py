# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The nodes of a box of equal hexahedra as a lattice over a grid of ranks.

The parameter of this application is trilinear on equal bricks, so its unknowns are the
nodes of a lattice, and what works on it (the point spread functions, the prior, the solve
with their sum) is stencils, slices and Fourier transforms of an array ``[k, j, i]`` on the
device.  MFEM's numbering of the same unknowns has nothing of the lattice in it: on a mesh
made by cuts two neighbours along x are 2.5 million apart in the dof number on average, of
8.1 million.  So nothing here runs in MFEM's order, and one permutation,
:class:`LatticeMap`, takes the library's vectors to a rank's block and back at the few
places where the two meet (the propagator's parameter, the library's algorithms).

On a box everything a parallel run asks of a mesh is arithmetic on a rank's number.  The
box is cut into ``px py pz`` equal blocks, rank ``I + px (J + py K)`` holding block
``(I, J, K)``, which is the numbering of MFEM's ``Mesh::CartesianPartitioning``; a node on a
face between blocks belongs to the block with the lower index, which is MFEM's rule for
true dofs (a shared dof is the lowest rank's).  Who sends which nodes to whom for a halo of
any width is then the meet of two boxes that both ends compute from the same integers: no
search, no message to set an exchange up, and no size that one rank decides and another
has to be told (:class:`Lattice`, :class:`Halo`).  :class:`LatticeMap` holds this
arithmetic against MFEM on every rank when it is made, and stops them all where the two
disagree.

Conventions
-----------
* What counts along the mesh's axes is in the order x, y, z, as MFEM has it: the elements
  ``N = (Nx, Ny, Nz)``, the blocks ``P = (px, py, pz)``, a block ``(I, J, K)``, lengths and
  an origin in metres.
* What describes an array is in the array's order z, y, x (C order, x fastest): a shape,
  a width ``W = (wz, wy, wx)``, an origin in nodes, slices, and a *box* of nodes
  ``((lo, hi), (lo, hi), (lo, hi))`` in global node numbers, ``hi`` excluded.
* Block ``I`` of a direction with ``e = N / p`` elements a block holds the elements
  ``[I e, (I + 1) e)``, so the nodes ``[I e, (I + 1) e]`` (its *closed* range), and owns the
  nodes ``(I e, (I + 1) e]``, and node 0 when ``I = 0``.

A rank's halo array is its closed range widened by ``W`` nodes a side, so that it has one
shape on every rank (the owned range of the first block of a direction is a node longer
than the others'): one compiled stencil and one transform serve all of them.
"""
import itertools
import os
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from hippymfem.common.parvector import allreduce
from hippymfem.fem import kernel as kernel_mod

import jax
import jax.numpy as jnp
from jax import lax

#: Whether a halo travels from and into the cards' own memory, as the wave field's shared
#: unknowns do (``sem.DEVICE_MPI``, the same variable read the same way): ``auto`` is where
#: Cray MPICH runs with its GPU support and the arrays are not the host's.
#: ``HIPPYMFEM_DEVICE_MPI`` (``1``, ``0`` or ``auto``).
DEVICE_MPI = os.environ.get("HIPPYMFEM_DEVICE_MPI", "auto").strip().lower() or "auto"

#: Bytes of the pieces in which a card's values are brought to the host where MPI cannot
#: take them from the card.  JAX makes a new host array for every array it brings from a
#: card and glibc maps new memory for one above 32 MiB (20 to 35 ms for 34 to 57 MB on an
#: EPYC 7763, measured for the wave field's exchange: ``sem.HOST_PIECE``); pieces below
#: that, copied one after the other into an array that is kept, each take the block the
#: one before left.
HOST_PIECE = 16 * 2 ** 20

#: The tags of the two exchanges, so that a message of one is never taken for the other's.
TAG_FILL, TAG_BACK = 73, 74


def good_size(n):
    """The least length of at least ``n`` that has no prime factor above 7: what a fast
    Fourier transform of a halo array wants (321 = 3 x 107 nodes becomes 324)."""
    m = max(int(n), 1)
    while True:
        r = m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def _meet(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi > lo else None


def _slices(box, frame):
    """Where ``box`` lies in an array whose first node is ``frame`` (three node numbers)."""
    return tuple(slice(lo - f, hi - f) for (lo, hi), f in zip(box, frame))


def _count(box):
    return int(np.prod([hi - lo for lo, hi in box], dtype=np.int64))


class Lattice:
    """``N = (Nx, Ny, Nz)`` elements in ``P = (px, py, pz)`` equal blocks, one a rank.

    ``p_d`` has to divide ``N_d``: ``Mesh::CartesianPartitioning`` puts an element where
    its centre lies, and a centre on a cut is decided by round-off (7 elements in 2 blocks
    were cut at 4, not at 3).  ``P = (1, 1, 1)`` is one rank that holds the whole lattice.

    ``shape`` is the lattice's nodes ``(Nz + 1, Ny + 1, Nx + 1)``, ``e`` the elements of a
    block ``(ex, ey, ez)``, ``size`` the number of blocks.
    """

    def __init__(self, N, P=(1, 1, 1)):
        self.N = tuple(int(n) for n in N)
        self.P = tuple(int(p) for p in P)
        if len(self.N) != 3 or len(self.P) != 3 or min(self.N + self.P) < 1:
            raise ValueError("a lattice needs three numbers of elements and of blocks, each at "
                             "least one (%s elements, %s blocks)" % (self.N, self.P))
        if any(n % p for n, p in zip(self.N, self.P)):
            raise ValueError("the blocks %s do not divide the elements %s" % (self.P, self.N))
        self.e = tuple(n // p for n, p in zip(self.N, self.P))
        self.size = self.P[0] * self.P[1] * self.P[2]
        self.shape = tuple(n + 1 for n in self.N[::-1])

    def block(self, rank):
        """The block ``(I, J, K)`` of a rank."""
        px, py, _ = self.P
        return (rank % px, (rank // px) % py, rank // (px * py))

    def rank(self, B):
        """The rank of block ``B = (I, J, K)``."""
        return B[0] + self.P[0] * (B[1] + self.P[1] * B[2])

    def closed(self, B):
        """The box of the nodes of block ``B``'s elements."""
        return tuple((I * e, (I + 1) * e + 1) for I, e in zip(B[::-1], self.e[::-1]))

    def owned(self, B):
        """The box of the nodes block ``B`` owns: its closed range without the low faces,
        which are the blocks' before it."""
        return tuple((I * e + (1 if I > 0 else 0), (I + 1) * e + 1)
                     for I, e in zip(B[::-1], self.e[::-1]))

    def owner(self, d, g):
        """The block index, in direction ``d`` (0 for x), of the owner of the nodes ``g``
        of that direction (an array of node numbers)."""
        return np.maximum((np.asarray(g) - 1) // self.e[d], 0)

    def need(self, B, W, core="closed", fft=False):
        """What block ``B`` has a field on for a halo of ``W = (wz, wy, wx)`` nodes: its
        closed range (or, with ``core="owned"``, the nodes it owns) widened by ``W`` a
        side, as ``(box, need)``: the array's box, which can reach outside the lattice,
        and its part inside the lattice, which is what there is to fill.

        With ``fft`` the box is lengthened at its high end to a length a fast transform
        likes (:func:`good_size`); what it gains is filled as the rest is."""
        c = self.closed(B) if core == "closed" else self.owned(B)
        box = []
        for (lo, hi), w in zip(c, W):
            n = hi - lo + 2 * w
            box.append((lo - w, hi + w + (good_size(n) - n if fft else 0)))
        return tuple(box), tuple((max(lo, 0), min(hi, g)) for (lo, hi), g in zip(box, self.shape))


class Halo:
    """The plan of one rank's exchanges for a halo of ``W`` nodes: who sends it which
    nodes, and which of its own it sends to whom.

    ``W`` is ``(wz, wy, wx)`` or one number for the three.  With ``fft`` the array is
    lengthened to transform-friendly lengths (:meth:`Lattice.need`), the same on every
    rank.

    What a rank takes from rank ``r`` is ``owned(r) & need(me)`` and what it gives is
    ``owned(me) & need(r)``: a halo wider than a neighbour's block only makes the lists
    longer.  ``recv`` and ``send`` are ``[(rank, box)]`` in ascending rank, one box a rank.
    The values of a box travel in the C order of ``[k, j, i]``; a rank's send buffer is its
    boxes one after the other, and so is its receive buffer: ``send_spans`` and
    ``recv_spans`` give ``(rank, start, stop)`` in them, ``nsend`` and ``nrecv`` their
    lengths.  (With the closed range as the core and no halo in a direction a rank takes
    a face from the block before it and gives that block nothing: the two lists need not
    name the same ranks.)

    The two exchanges are :meth:`pack` then :meth:`unpack` (*fill*: a rank's owned block
    becomes the field on its array, zero outside the lattice) and :meth:`pack_T` then
    :meth:`unpack_T` (*add back*, the transpose: what every rank has on the nodes of this
    one, summed onto its owned block), the second with the two buffers' roles exchanged.
    These four are the plan in NumPy, which is what the compiled ones of
    :class:`DeviceHalo` are held against.

    Members that describe the array, all in its order: ``W``, ``shape``, ``own_shape``,
    ``own_slices`` (where the owned block lies in the array), ``origin`` (the global node
    of its first entry, negative where it leaves the lattice), ``global_shape``.
    """

    def __init__(self, lat, rank, W, core="closed", fft=False):
        self.lat, self.rank, self.core, self.fft = lat, int(rank), core, bool(fft)
        self.W = (int(W),) * 3 if np.ndim(W) == 0 else tuple(int(w) for w in W)
        B = lat.block(self.rank)
        self.own = lat.owned(B)
        self.box, self.need = lat.need(B, self.W, core, self.fft)
        self.shape = tuple(hi - lo for lo, hi in self.box)
        self.own_shape = tuple(hi - lo for lo, hi in self.own)
        self.origin = tuple(lo for lo, _ in self.box)
        self.global_shape = lat.shape
        self.own_slices = _slices(self.own, self.origin)
        take, give = [], []
        for a in range(3):                       # an axis of the arrays: direction 2 - a of the mesh
            t, g = [], []
            for J in range(lat.P[2 - a]):
                other = [0, 0, 0]
                other[2 - a] = J
                m = _meet(lat.owned(other)[a], self.need[a])
                if m:
                    t.append((J, m))
                m = _meet(self.own[a], lat.need(other, self.W, core, self.fft)[1][a])
                if m:
                    g.append((J, m))
            take.append(t)
            give.append(g)
        self.recv, self.send = self._boxes(take), self._boxes(give)
        self.recv_spans, self.nrecv = self._spans(self.recv)
        self.send_spans, self.nsend = self._spans(self.send)
        self.neighbors = sorted(set(r for r, _ in self.recv) | set(r for r, _ in self.send))
        self._recv_at = [_slices(box, self.origin) for _, box in self.recv]
        self._send_at = [_slices(box, [lo for lo, _ in self.own]) for _, box in self.send]

    def _boxes(self, per_axis):
        out = []
        for (K, bz), (J, by), (I, bx) in itertools.product(*per_axis):
            r = self.lat.rank((I, J, K))
            if r != self.rank:
                out.append((r, (bz, by, bx)))
        out.sort(key=lambda t: t[0])
        return out

    @staticmethod
    def _spans(boxes):
        out, a = [], 0
        for r, box in boxes:
            out.append((r, a, a + _count(box)))
            a += _count(box)
        return out, a

    def mirror_index(self):
        """For each axis of the array, the entry whose value an entry takes when the field
        is continued evenly across the lattice's faces: itself inside the lattice, its
        mirror image about the end node outside (the end node is not repeated; a halo
        wider than the lattice is reflected again at the far face).

        The array holds the image of everything within ``W`` of its core.  Of what a
        transform-friendly length added beyond that it may not, on a block shorter than
        the addition: such an entry is its own image and stays zero.
        """
        out = []
        for (lo, hi), g in zip(self.box, self.global_shape):
            p = 2 * (g - 1)
            i = np.mod(np.arange(lo, hi), p)
            image = np.where(i > g - 1, p - i, i) - lo
            here = np.arange(hi - lo)
            out.append(np.where((image >= 0) & (image < hi - lo), image, here))
        return out

    # ------------------------------------------------------------------- fill
    def pack(self, x):
        """The send buffer of an owned block ``x``."""
        out = np.empty(self.nsend, dtype=x.dtype)
        for s, (_, a, b) in zip(self._send_at, self.send_spans):
            out[a:b] = x[s].reshape(-1)
        return out

    def unpack(self, x, inc):
        """The halo array of an owned block ``x`` and the receive buffer ``inc``."""
        ext = np.zeros(self.shape, dtype=x.dtype)
        ext[self.own_slices] = x
        for s, (_, a, b) in zip(self._recv_at, self.recv_spans):
            ext[s] = inc[a:b].reshape(ext[s].shape)
        return ext

    # --------------------------------------------------------------- add back
    def pack_T(self, ext):
        """What a halo array holds on the others' nodes, laid out as the receive buffer of
        a fill."""
        out = np.empty(self.nrecv, dtype=ext.dtype)
        for s, (_, a, b) in zip(self._recv_at, self.recv_spans):
            out[a:b] = ext[s].reshape(-1)
        return out

    def unpack_T(self, ext, inc):
        """The owned part of ``ext`` plus what the others hold on this rank's nodes
        (``inc``, laid out as the send buffer of a fill), added in ascending rank."""
        y = np.array(ext[self.own_slices])
        for s, (_, a, b) in zip(self._send_at, self.send_spans):
            y[s] += inc[a:b].reshape(y[s].shape)
        return y


class DeviceHalo:
    """A :class:`Halo` at work on device arrays, bound to a communicator: what the point
    spread functions and the prior ask of a halo.

    ``fill(a_own) -> ext``
        the field on the rank's array: its own block, the neighbours' values around it,
        zeros outside the lattice.
    ``add_back(ext) -> a_own``
        the transpose: what every rank computed on this rank's nodes, summed.
    ``mirror(ext) -> ext``
        the field continued evenly across the lattice's faces, written where the array
        leaves the lattice (the natural boundary condition of the prior's stencil).
    ``dot(a_own, b_own) -> float``
        the inner product of two fields, reduced over the ranks.
    ``W``, ``shape``, ``own_shape``, ``own_slices``, ``origin``, ``global_shape``
        the plan's, in the array's order.

    A lattice of one block (``P = (1, 1, 1)``) sends and reduces nothing, and ``comm`` is
    then not used at all: a rank that holds the whole box beside others that hold it too
    is that case.

    The packing and the unpacking are compiled, every slice a constant.  The buffers go
    from card to card where MPI takes device addresses (:data:`DEVICE_MPI`; the pattern of
    ``sem.Interface.swap_device``): the neighbours' values are then written behind JAX's
    back into an array this object keeps for each of the two exchanges, which is why
    :meth:`fill` and :meth:`add_back` wait for their result before they return it (the
    next exchange overwrites that array).  Otherwise they go through host arrays that are
    kept.  ``seconds`` and ``exchanges`` count the time between a buffer being ready and
    the neighbours' being on the device, and the exchanges.
    """

    def __init__(self, halo, comm, dtype=np.float64, device=None):
        self.plan, self.comm = halo, comm
        self.dtype = np.dtype(dtype)
        self.device = device if device is not None else kernel_mod.device()
        lat = halo.lat
        if lat.size > 1 and comm.size != lat.size:
            raise ValueError("a lattice of %s blocks on a communicator of %d ranks"
                             % (lat.P, comm.size))
        self.W, self.shape, self.own_shape = halo.W, halo.shape, halo.own_shape
        self.own_slices, self.origin, self.global_shape = halo.own_slices, halo.origin, halo.global_shape
        self.seconds, self.exchanges = 0.0, 0
        self.on_device = (DEVICE_MPI in ("1", "yes", "true", "on")
                          or (DEVICE_MPI == "auto" and self.device.platform != "cpu"
                              and os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0") == "1"))
        self._incoming, self._host = {}, {}

        own_at, shape = halo.own_slices, halo.shape
        pad = [(s.start, n - s.stop) for s, n in zip(own_at, shape)]
        send_at, recv_at = halo._send_at, halo._recv_at
        send_spans, recv_spans = halo.send_spans, halo.recv_spans

        def start_size(s):
            return tuple(sl.start for sl in s), tuple(sl.stop - sl.start for sl in s)

        def gather(x, where):
            if not where:
                return jnp.zeros((0,), x.dtype)
            return jnp.concatenate([x[s].reshape(-1) for s in where])

        def unpack(x, inc):
            ext = jnp.pad(x, pad)
            for s, (_, a, b) in zip(recv_at, recv_spans):
                start, size = start_size(s)
                ext = lax.dynamic_update_slice(ext, inc[a:b].reshape(size), start)
            return ext

        def unpack_T(ext, inc):
            y = ext[own_at]
            for s, (_, a, b) in zip(send_at, send_spans):
                start, size = start_size(s)
                y = lax.dynamic_update_slice(y, y[s] + inc[a:b].reshape(size), start)
            return y

        self._pack = jax.jit(lambda x: gather(x, send_at))
        self._pack_T = jax.jit(lambda ext: gather(ext, recv_at))
        self._unpack, self._unpack_T = jax.jit(unpack), jax.jit(unpack_T)
        self._pad = jax.jit(lambda x: jnp.pad(x, pad))
        self._own = jax.jit(lambda ext: ext[own_at])
        # (summed in double whatever the fields' precision: an iteration's inner products
        # decide its steps, and the conversion is part of the one pass over the two arrays)
        self._dot = jax.jit(lambda a, b: jnp.sum(a.astype(jnp.float64) * b.astype(jnp.float64)))
        index = halo.mirror_index()
        self._mirror = None
        if any(not np.array_equal(ix, np.arange(ix.size)) for ix in index):
            moved = [(ax, jax.device_put(ix.astype(np.int32), self.device))
                     for ax, ix in enumerate(index) if not np.array_equal(ix, np.arange(ix.size))]

            def mirror(ext):
                for ax, ix in moved:
                    ext = jnp.take(ext, ix, axis=ax)
                return ext

            self._mirror = jax.jit(mirror)

    # ------------------------------------------------------------ the interface
    def _held(self, a, shape, what):
        """Stop an array that is not what this rank exchanges, on any number of ranks.  The
        compiled slices cut whatever they are given: the closed block where the owned one
        is meant is the same array on one rank and a node longer on the others, where the
        result came out shifted by a node without a word.  And MPI is told the object's
        precision: an array of another one passes on one rank, and on several its buffer
        is read with the wrong item size before anything objects."""
        if tuple(a.shape) != tuple(shape) or a.dtype != self.dtype:
            raise ValueError("the halo of rank %d takes %s, an array of shape %s and of %s, not one of "
                             "shape %s and of %s" % (self.plan.rank, what, tuple(shape), self.dtype.name,
                                                    tuple(a.shape), a.dtype))

    def fill(self, a):
        """The halo array of the owned block ``a``."""
        h = self.plan
        self._held(a, self.own_shape, "the rank's owned block")
        if not h.neighbors:
            return self._pad(a)
        inc = self._swap(self._pack(a), h.send_spans, h.recv_spans, h.nrecv, TAG_FILL)
        return jax.block_until_ready(self._unpack(a, inc))

    def add_back(self, ext):
        """The owned block of the sum of all ranks' halo arrays ``ext``."""
        h = self.plan
        self._held(ext, self.shape, "the rank's halo array")
        if not h.neighbors:
            return self._own(ext)
        inc = self._swap(self._pack_T(ext), h.recv_spans, h.send_spans, h.nsend, TAG_BACK)
        return jax.block_until_ready(self._unpack_T(ext, inc))

    def mirror(self, ext):
        """``ext`` with the mirror image where it leaves the lattice (itself on a rank
        whose array does not)."""
        return ext if self._mirror is None else self._mirror(ext)

    def dot(self, a, b):
        """The inner product of two fields given by their owned blocks, over all ranks."""
        value = float(self._dot(a, b))
        return value if self.plan.lat.size == 1 else allreduce(self.comm, value)

    # -------------------------------------------------------------- the exchange
    def _swap(self, out, snd, rcv, n, tag):
        """Send the device array ``out`` in the spans ``snd`` and return the neighbours'
        ``n`` values, laid out by the spans ``rcv``, as a device array.  All receives are
        posted, then all sends, then one wait."""
        comm = self.comm
        out = jax.block_until_ready(out)
        t0 = time.perf_counter()
        if self.on_device:
            inc = self._incoming.get(tag)
            if inc is None:
                # an array of this object's own, made by a computation: not a constant
                # that JAX may share with another array (devicebridge.copy_to_jax)
                inc = jax.device_put(np.zeros(n, self.dtype), self.device) + self.dtype.type(0)
                inc = self._incoming[tag] = jax.block_until_ready(inc)
            size, typ = self.dtype.itemsize, MPI._typedict[self.dtype.char]
            at = getattr(MPI, "buffer", None) or MPI.memory
            reqs = []
            if rcv:
                p = inc.unsafe_buffer_pointer()
                reqs += [comm.Irecv([at.fromaddress(p + a * size, (b - a) * size), typ], source=r, tag=tag)
                         for r, a, b in rcv]
            if snd:
                p = out.unsafe_buffer_pointer()
                reqs += [comm.Isend([at.fromaddress(p + a * size, (b - a) * size), typ], dest=r, tag=tag)
                         for r, a, b in snd]
            MPI.Request.Waitall(reqs)
        else:
            send, recv = self._to_host(out, tag), self._kept(("receive", tag), n)
            reqs = [comm.Irecv(recv[a:b], source=r, tag=tag) for r, a, b in rcv]
            reqs += [comm.Isend(send[a:b], dest=r, tag=tag) for r, a, b in snd]
            MPI.Request.Waitall(reqs)
            # on the device and waited for before the kept array is written again, and
            # before a compiled call is launched with it (sem.PUT_FIRST)
            inc = jax.block_until_ready(jax.device_put(recv, self.device))
        self.seconds += time.perf_counter() - t0
        self.exchanges += 1
        return inc

    def _kept(self, key, n):
        if key not in self._host:
            self._host[key] = np.zeros(n, self.dtype)
        return self._host[key]

    def _to_host(self, x, tag):
        """A device array in host memory: the array JAX makes of it, or, where that would
        be a large one on a card, its pieces in an array that is kept (:data:`HOST_PIECE`)."""
        if self.device.platform == "cpu" or x.nbytes <= 2 * HOST_PIECE:
            return np.asarray(x)
        out = self._kept(("send", tag), x.shape[0])
        m = HOST_PIECE // self.dtype.itemsize
        for a in range(0, x.shape[0], m):
            out[a:a + m] = np.asarray(x[a:a + m])
        return out


class LatticeMap:
    """The true dofs of the trilinear space ``Vm`` and this rank's owned block of ``lat``.

    ``size`` and ``origin`` are the box's lengths and its corner ``(x, y, z)``.  A dof's
    node is its vertex's coordinates rounded (they are off an integer by 1e-13 of a step
    after the cuts that made the mesh), and MFEM numbers a rank's true dofs as its owned
    local dofs in their order, so the table is a cumulative sum.  Built before anything
    gives the mesh nodes: a prior of the library does, and the vertices are then not
    where the geometry is read from.

    ``tdof_of_owned``
        the true dof at each node of the owned block, in the block's order ``[k, j, i]``.
    ``ldof_of_closed``
        the local dof (MFEM's vertex) at each node of the closed block: what
        ``Vm.local_values`` is indexed with to give the block the elements see.
    ``own_shape``, ``closed_shape``, ``h`` (the steps ``(hx, hy, hz)``)

    Making it is collective.  Six things are checked on every rank and the worst code is
    reduced, so that all ranks raise together: 1 the mesh has no nodes; 2 its vertices are
    the lattice's, as many as the rank's closed block has; 3 they span that block, so the
    rank holds the elements the arithmetic gives it; 4 no node twice; 5 the rank owns as
    many dofs as MFEM says; 6 (``check``) MFEM's own word on who owns what and in which
    order: a vector of true-dof numbers through the prolongation is the table's on the
    owned dofs and another rank's number on the others.
    """

    def __init__(self, Vm, lat, size, origin=(0.0, 0.0, 0.0), check=True):
        comm, fes, mesh = Vm.comm, Vm.fes, Vm.mesh
        self.Vm, self.lat, self.comm = Vm, lat, comm
        B = lat.block(comm.rank)
        closed, own = lat.closed(B), lat.owned(B)
        self.closed_shape = tuple(hi - lo for lo, hi in closed)
        self.own_shape = tuple(hi - lo for lo, hi in own)
        self.h = tuple(float(s) / n for s, n in zip(size, lat.N))
        bad, nv = 0, int(np.prod(self.closed_shape))
        if comm.size != lat.size or mesh.GetNodes() is not None:
            bad = 1
        else:
            v = mfem.Vector()
            mesh.GetVertices(v)                   # all x, then all y, then all z
            q = (np.array(v.GetDataArray()).reshape(3, -1).T - np.asarray(origin, float)) / self.h
            kji = np.rint(q).astype(np.int64)[:, ::-1]
            if kji.shape[0] != nv or nv != fes.GetVSize() or np.abs(q[:, ::-1] - kji).max() > 1e-6:
                bad = 2
        if not bad:
            lo, hi = np.array([c[0] for c in closed]), np.array([c[1] for c in closed])
            if (kji.min(axis=0) != lo).any() or (kji.max(axis=0) != hi - 1).any():
                bad = 3
        if not bad:
            place = np.ravel_multi_index(tuple((kji - lo).T), self.closed_shape)
            ldof_of_closed = np.empty(nv, dtype=np.int32)
            ldof_of_closed[place] = np.arange(nv, dtype=np.int32)
            if not np.array_equal(place[ldof_of_closed], np.arange(nv)):
                bad = 4
        if not bad:
            olo = np.array([c[0] for c in own])
            owned = (kji >= olo).all(axis=1)                    # the lattice's rule
            tdof = np.cumsum(owned) - 1                         # MFEM's: the owned local dofs in their order
            ntrue = int(owned.sum())
            if ntrue != fes.GetTrueVSize() or ntrue != int(np.prod(self.own_shape)):
                bad = 5
        # The sixth check is a product with the prolongation, in which a rank waits for its
        # neighbours' values: it is made by all ranks or by none.  (Made by those that had
        # passed the five others, a rank whose block was right waited in it for a
        # neighbour whose block was not, which had gone on to the reduction: no rank
        # raised and none returned.)
        worst = allreduce(comm, int(bad), MPI.MAX)
        if not worst and check:
            first = int(fes.GetMyTDofOffset())
            label = Vm.vector()
            label._host_write()[:] = first + np.arange(ntrue, dtype=np.float64)
            gid = np.rint(Vm.local_values(label)).astype(np.int64)
            mine = (gid >= first) & (gid < first + ntrue)
            if not np.array_equal(gid[owned], first + tdof[owned]) or mine[~owned].any():
                bad = 6
            worst = allreduce(comm, int(bad), MPI.MAX)
        if worst:
            raise RuntimeError(
                "the lattice %s in %s blocks and MFEM's mesh disagree (code %d on some rank, %d on "
                "rank %d): 1 the mesh has nodes or the ranks are not the blocks, 2 the vertices are not "
                "the lattice's, 3 another block of elements, 4 a node twice, 5 another number of "
                "owned nodes, 6 other owners or another order of the true dofs"
                % (lat.N, lat.P, worst, bad, comm.rank))
        self.ldof_of_closed = ldof_of_closed
        sel = np.nonzero(owned)[0]
        oplace = np.ravel_multi_index(tuple((kji[sel] - olo).T), self.own_shape)
        self.tdof_of_owned = np.empty(ntrue, dtype=np.int32)
        self.tdof_of_owned[oplace] = tdof[sel]
        self._on_device = {}

    def _tables(self, dev):
        """The table and its inverse on a device, made when an array there is first asked
        for or given, with the two compiled gathers."""
        if dev not in self._on_device:
            n = self.tdof_of_owned.size
            back = np.empty(n, dtype=np.int32)
            back[self.tdof_of_owned] = np.arange(n, dtype=np.int32)
            shape = self.own_shape
            self._on_device[dev] = (
                jax.device_put(self.tdof_of_owned, dev), jax.device_put(back, dev),
                jax.jit(lambda x, t, dtype: x[t].astype(dtype).reshape(shape), static_argnums=2),
                jax.jit(lambda a, t: a.reshape(-1)[t].astype(jnp.float64)))
        return self._on_device[dev]

    def to_lattice(self, v, device=False, dtype=np.float64):
        """The owned block ``[k, j, i]`` of the vector ``v`` on ``Vm``: a NumPy array, or
        with ``device`` (``True`` for the library's, or a device of JAX) an array of
        ``dtype`` there.  The permutation is then done on the device: one copy of the
        vector to the card."""
        x = v._host_read()
        if device is False:
            return x[self.tdof_of_owned].reshape(self.own_shape)
        dev = kernel_mod.device() if device is True else device
        table, _, forth, _ = self._tables(dev)
        # Waited for.  JAX launches the gather and returns; until it has run it reads the
        # vector's own host memory (which on the host JAX may take without a copy), and the
        # caller is free to write its vector again once this returns: of 20 vectors
        # overwritten right after the call, 9 to 18 came back with the new values in them
        # (host, a million entries).
        return jax.block_until_ready(forth(jax.device_put(x, dev), table, np.dtype(dtype)))

    def from_lattice(self, a, v=None):
        """The vector on ``Vm`` (``v``, or a new one) of an owned block ``a``, a NumPy or a
        device array of any precision."""
        v = self.Vm.vector() if v is None else v
        if isinstance(a, np.ndarray):
            v._host_write()[self.tdof_of_owned] = a.reshape(-1)
        else:
            if int(np.prod(a.shape)) != self.tdof_of_owned.size:
                # (NumPy refuses the assignment above; a gather on the device takes what it finds)
                raise ValueError("an array of shape %s is not this rank's owned block %s"
                                 % (tuple(a.shape), self.own_shape))
            _, table, _, back = self._tables(next(iter(a.devices())))
            v._host_write()[:] = np.asarray(back(a, table))
        return v

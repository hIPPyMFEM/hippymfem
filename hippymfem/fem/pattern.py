# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Sparsity patterns and the scatter of element arrays into them.

:class:`ScatterPattern` is the value-independent part of a matrix assembly (the
CSR graph, the slot of every element entry, the masks and the device copies) and
:class:`VectorPattern` its counterpart for dual vectors; both are cached per
(space pair, element groups) in an :class:`~hippymfem.common.identitycache.IdentityCache`.
"""

import os
import time
import warnings

import numpy as np
from ..common.identitycache import IdentityCache
from ..common.keepalive import KeepAlive
from ..common.linalg import _diag_block, alias
from ..common.parvector import _HYPRE_INT
from .elementbatch import group_tables
from .spaces import as_space
from .parmat import _TRANSPOSE, _TRIPLE_CHOICE
from .prolongation import _BOOLEAN_P, _ESS_LDOF, _IDENTITY_P


#: When true, the device-side scatter gathers into each CSR slot in a fixed order
#: instead of using atomics, so repeated GPU assemblies are bit-identical.  Costs
#: ``nnz * maxc`` doubles of working memory.  Set ``HIPPYMFEM_GPU_DETERMINISTIC=1``.
DETERMINISTIC = os.environ.get("HIPPYMFEM_GPU_DETERMINISTIC", "").lower() not in (
    "", "0", "no", "false", "off")


#: Keep the ``nnz``-sized accumulator of a fused assembly and reset it in place
#: instead of allocating one per assembly.  JAX's arena never returns memory, so the
#: kept buffer costs nothing that freeing it would give back, while asking for a fresh
#: one costs a *contiguous* ``nnz`` block: at two million P2 hexahedra on a rank that
#: is 7.7 GiB, more than half the arena at a quarter-card cap, and once the arena has
#: grown to its cap and small allocations have landed in the freed block, the next
#: assembly fails (RESOURCE_EXHAUSTED inside the scatter; 400^3 on 32 MIG slices).
#: Buffers are kept per (device, ``nnz``) and borrowed, so what is held is the peak an
#: assembly needs at once, never more.  Set ``HIPPYMFEM_FUSED_KEEP=0`` to allocate per
#: assembly.
FUSED_KEEP = os.environ.get("HIPPYMFEM_FUSED_KEEP", "1").lower() not in (
    "0", "no", "false", "off")

#: Share of JAX's arena above which an accumulator is kept.  Only the large ones are
#: worth it: a small buffer is one the allocator can fit in the holes it already has,
#: while a buffer of a third of the arena or more cannot be found again once the arena
#: is full and its old block has been split (the assemblies that failed that way sat at
#: 0.34 and 0.58 of their arenas).  Keeping the small ones as well raises the
#: high-water mark for nothing, since A, C and the W blocks are each assembled in a
#: pass of their own: at 128^3 on two L40S, where the accumulator is a fifth of the
#: arena, keeping it takes JAX's peak from 7.6 to 11.7 GiB.  Set
#: ``HIPPYMFEM_FUSED_KEEP_SHARE``.
FUSED_KEEP_SHARE = float(os.environ.get("HIPPYMFEM_FUSED_KEEP_SHARE", "0.25") or 0.25)

#: Doubles of zeros written per in-place update when an accumulator is reset.
_ZERO_BLOCK = 1 << 22

_ACC_FREE = {}                     # (device, nnz) -> kept accumulators
_ACC_ZERO = {}                     # (device, block) -> a block of zeros
_ACC_PUT = {}                      # the jitted in-place block write


def set_deterministic(flag=True):
    """Turn the order-independent device scatter on or off; returns the old value."""
    global DETERMINISTIC
    old, DETERMINISTIC = DETERMINISTIC, bool(flag)
    return old


def _device_key():
    from .kernel import device

    return device()


def host_writable(acc):
    """A writeable host array from an accumulator that may still be a device array.

    ``np.asarray`` of a JAX array is the one device-to-host copy; JAX hands it back
    read-only but owning its buffer.  Every accumulator here is consumed by its
    caller, which zeroes the eliminated slots, adds the boundary entries and builds
    the matrix on top of it, so the flag is lifted rather than paying a second
    ``nnz``-sized copy.  Where it cannot be lifted (a view of a host JAX buffer),
    the copy is made.  ``np.add.at`` does not check the flag, so a read-only
    accumulator would fail only at its first indexed assignment.
    """
    out = np.asarray(acc)
    if not out.flags.writeable:
        try:
            out.flags.writeable = True
        except ValueError:
            out = out.copy()
    return out


def _fused_add(acc, flat, slot, sign):
    """``acc[slot] += flat`` on the device, writing into ``acc`` rather than a copy.

    ``donate_argnums`` is what makes this affordable: without it every chunk would
    allocate and copy a fresh ``nnz``-sized accumulator, gigabytes per chunk on a
    large mesh.
    """
    import jax

    key = (sign is not None,)
    fn = _FUSED_ADD.get(key)
    if fn is None:
        def step(a, v, idx, sg):
            return a.at[idx].add(v if sg is None else v * sg)

        fn = _FUSED_ADD[key] = jax.jit(step, donate_argnums=(0,))
    return fn(acc, flat, slot, sign)


_FUSED_ADD = {}


def _acc_keepable(d, n):
    """Whether an ``n``-double accumulator on ``d`` is worth keeping.

    Two conditions.  The allocator must have a fixed arena: under
    ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` every buffer is a driver allocation that
    a free really does return, so a kept one would take memory from hypre rather than
    from JAX's own high-water mark, and a host device (which reports no budget) may
    hand back a *view* of the buffer from ``np.asarray``, which must not be reused
    under the caller.  And the buffer must be a large share of that arena
    (:data:`FUSED_KEEP_SHARE`).
    """
    try:
        stats = d.memory_stats() or {}
    except Exception:                                        # noqa: BLE001
        return False
    limit = stats.get("bytes_limit")
    return bool(limit) and n * 8 >= FUSED_KEEP_SHARE * limit


def _acc_reset(acc, n):
    """Zero ``acc`` in place, a block at a time.

    ``jnp.zeros_like``, ``a * 0`` and ``a.at[:].set(0)`` each allocate a second
    accumulator even with the first donated, because XLA copies rather than aliases
    when the output does not come from an in-place write: measured on JAX 0.11, a 4
    GiB accumulator under a 6.2 GiB cap fails on all three.  A donated
    ``dynamic_update_slice`` does write in place (peak 4.06 GiB for the same case),
    so the zeros go in through one.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    fn = _ACC_PUT.get("put")
    if fn is None:
        fn = _ACC_PUT["put"] = jax.jit(
            lambda a, z, i: lax.dynamic_update_slice(a, z, (i,)),
            donate_argnums=(0,))
    blk = min(max(1, _ZERO_BLOCK), n)
    key = (_device_key(), blk)
    z = _ACC_ZERO.get(key)
    if z is None:
        z = _ACC_ZERO[key] = jnp.zeros(blk, dtype=jnp.float64)
    for start in range(0, n, blk):
        m = min(blk, n - start)
        acc = fn(acc, z if m == blk else z[:m], start)
    return acc


def _acc_take(d, n):
    """A zeroed accumulator of ``n`` doubles on ``d``, kept from an earlier assembly
    where there is one."""
    import jax.numpy as jnp

    free = _ACC_FREE.get((d, n))
    if free:
        return _acc_reset(free.pop(), n)
    return jnp.zeros(n, dtype=jnp.float64)


def _acc_give(d, n, acc):
    """Keep ``acc`` for the next assembly of this size."""
    _ACC_FREE.setdefault((d, n), []).append(acc)


def _warn_if_growth_arena(d, n):
    """Say so once when a large accumulator is about to be asked of a *growing* arena.

    JAX's arena grows by carving regions from the driver, and every region it has
    carved counts against the cap even while most of it is free, so a request for a
    large contiguous block fails with the arena reading nearly empty: at 128^3 on an
    L40S, a 4.0 GiB accumulator cannot be had from an 11.1 GiB arena grown in pieces,
    while the same arena preallocated as one region serves it with 5.6 GiB to spare.
    The message is what the failure otherwise never says.
    """
    if n * 8 < _GROWTH_WARN_SHARE * (_arena_limit(d) or 0) or d in _WARNED:
        return
    _WARNED.add(d)
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() in (
            "1", "true", "yes", "on"):
        return
    warnings.warn(
        "this assembly needs %.1f GiB in one piece, %.0f%% of JAX's arena, and the "
        "arena is grown on demand: the regions it has already taken count against "
        "the cap even when they are free, so the allocation can fail with the arena "
        "reading almost empty. Set XLA_PYTHON_CLIENT_PREALLOCATE=true (with "
        "HIPPYMFEM_GPU_MEM_FRACTION sized for hypre's share) to take the arena as "
        "one region, or use more ranks."
        % (n * 8 / 2 ** 30, 100.0 * n * 8 / max(_arena_limit(d) or 1, 1)),
        RuntimeWarning, stacklevel=4)


def _arena_limit(d):
    """Bytes JAX's allocator may hold on ``d``, or None when it reports no cap."""
    try:
        return (d.memory_stats() or {}).get("bytes_limit")
    except Exception:                                        # noqa: BLE001
        return None


#: Share of the arena at which :func:`_warn_if_growth_arena` speaks up.
_GROWTH_WARN_SHARE = 0.15
_WARNED = set()


def clear_accumulators():
    """Drop every kept accumulator (for tests, and after a mesh is refined)."""
    _ACC_FREE.clear()
    _ACC_ZERO.clear()
    _WARNED.clear()


# --------------------------------------------------------------------- pattern
class ScatterPattern(KeepAlive):
    """CSR graph of the local ``ldof x ldof`` element-assembly map.

    Built once per (test space, trial space, element groups) triple.  Holds

    * ``indptr``, ``indices``: the CSR structure over **local** dof indices.  Within
      a row the columns ascend, except that on a square pattern the diagonal entry
      comes **first**, the order hypre keeps a square matrix in, so hypre has
      nothing to permute and the matrix can alias these arrays without a copy;
    * ``slot``: for every entry of every element array, the position of its
      target in the CSR ``data`` array.  Assembly is then one reduction over
      ``slot``, with no Python-level loop;
    * ``sign``: the product of the test and trial dof signs, or ``None`` when
      every sign is ``+1`` (always, for H1/L2; not for H(curl)/H(div)).

    Parameters
    ----------
    test_tables, trial_tables : sequence of SpaceTables
        One per element group, in the same order as the element arrays will be.
    """

    def _diag_first(self):
        """Whether rows lead with their diagonal: on a square local shape, as hypre wants."""
        return self.nrow == self.ncol

    def _encode(self, rows, cols):
        """Packed ``(row, col)`` keys in this pattern's order, diagonal first if square."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if self._diag_first():
            cols = np.where(cols == rows, np.int64(0), cols + 1)
        return (rows << self._KEY_SHIFT) | cols

    #: Bits given to the column in a packed ``(row, col)`` key.  A rank's local dof
    #: count is bounded by its share of the mesh and never reaches 2**32; the guard in
    #: :meth:`_build` says so rather than packing a key that would silently wrap.
    _KEY_SHIFT = np.int64(32)
    _KEY_MASK = np.int64((1 << 32) - 1)

    def __init__(self, test_tables, trial_tables, test_space=None,
                 trial_space=None):
        if len(test_tables) != len(trial_tables):
            raise ValueError("test and trial tables must cover the same groups")
        # The matrix shape comes from the *spaces*, never from the tables.  A rank
        # may own no elements of a group (with a boundary term on one attribute,
        # most ranks own none), and a shape from the first table would give it a
        # zero-row matrix while the others build full ones: a hang in the
        # collective construction that follows, not an error.
        self.nrow = int((test_space if test_space is not None
                         else test_tables[0].space).fes.GetVSize())
        self.ncol = int((trial_space if trial_space is not None
                         else trial_tables[0].space).fes.GetVSize())
        self.shapes = [(int(tt.group.ne), int(tt.nd_total), int(tr.nd_total))
                       for tt, tr in zip(test_tables, trial_tables)]
        self.sizes = [ne * a * b for ne, a, b in self.shapes]

        rows, cols, signs, elems = [], [], [], []
        any_sign = False
        for tt, tr in zip(test_tables, trial_tables):
            r = tt.edofs[:, :, None]                  # (ne, nd_test, 1)
            c = tr.edofs[:, None, :]                  # (ne, 1, nd_trial)
            shape = (tt.group.ne, tt.nd_total, tr.nd_total)
            rows.append(np.broadcast_to(r, shape).reshape(-1))
            cols.append(np.broadcast_to(c, shape).reshape(-1))
            elems.append(np.repeat(tt.group.elems, tt.nd_total * tr.nd_total))
            if tt.signs.min() < 0 or tr.signs.min() < 0:
                any_sign = True
                s = tt.signs[:, :, None] * tr.signs[:, None, :]
                signs.append(np.broadcast_to(s, shape).reshape(-1))
            else:
                signs.append(None)

        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
        else:                                   # this rank owns no such elements
            rows = np.zeros(0, np.int64)
            cols = np.zeros(0, np.int64)
        self.sign = (np.concatenate([
            s if s is not None else np.ones(n) for s, n in zip(signs, self.sizes)
        ]) if any_sign else None)

        # The element arrays arrive grouped by geometry, but MFEM's assembly loop
        # visits elements in mesh order.  Summing in a different order differs at
        # round-off, which is enough to move a long optimizer run, so on a
        # multi-geometry mesh the entries are permuted back into element order.  A
        # single-geometry mesh is already in that order and pays nothing.
        self.perm = None
        self._hypre = {}
        self._masked = IdentityCache()
        self._dev_zero = IdentityCache()
        self._fused = {}
        self._reuse = {}
        self._dev = {}
        self._dev_pad = {}
        self._pad = None
        self.maxc = None
        self._lookup = None
        if len(self.shapes) > 1:
            eidx = np.concatenate(elems)
            perm = np.argsort(eidx, kind="stable")
            if not np.array_equal(perm, np.arange(perm.size)):
                # as for slot: one entry per element-matrix entry, so the width
                # matters on the device
                self.perm = perm.astype(np.int32, copy=False) \
                    if perm.size <= np.iinfo(np.int32).max else perm
                rows = rows[perm]
                cols = cols[perm]
                if self.sign is not None:
                    self.sign = self.sign[perm]

        self._build(rows, cols)

    def _build(self, rows, cols):
        """Unique (row, col) pairs, in CSR order, and the entry -> slot map."""
        n = rows.size
        if max(self.nrow, self.ncol) > int(self._KEY_MASK):
            raise ValueError(
                "a rank holds %d rows and %d columns, which does not fit the packed "
                "(row, column) key: that needs more than 2**32 local dofs on one rank"
                % (self.nrow, self.ncol))
        if n == 0:
            self.indptr = np.zeros(self.nrow + 1, dtype=np.int32)
            self.indices = np.zeros(0, dtype=_HYPRE_INT)
            self.slot = np.zeros(0, dtype=np.intp)
            self.nnz = 0
            return
        # A large pattern is built without a global sort (patternbuild: group the
        # entries by row, order each short row on its own), which gives the same arrays
        # several times faster and with no device memory.  The sort below remains the
        # route for small patterns, and wherever numba is not importable.
        from . import patternbuild

        if patternbuild.wanted(n):
            t0 = time.perf_counter() if PATTERN_TIMING else None
            self.indptr, self.indices, self.slot, self.nnz = patternbuild.build(
                rows, cols, self.nrow, self.ncol, self._diag_first(), _HYPRE_INT)
            if PATTERN_TIMING:
                _say_phase("sort-free (%d thr)" % patternbuild.threads(),
                           time.perf_counter() - t0, n)
            return
        # One integer key per entry orders by row then column, which is CSR order.
        t_key = time.perf_counter() if PATTERN_TIMING else None
        # The two fields are packed by a shift rather than a multiply by ``ncol``, so
        # that unpacking them below is a shift and a mask: an integer division over
        # every nonzero is one of the more expensive things this build does, and a
        # local dof index never approaches the 2**32 the shift allows.
        #
        # On a square pattern each row must lead with its diagonal, the order hypre
        # keeps a square matrix in.  Rather than sort by column and then rotate every
        # row, the column is encoded so that the sort produces that order directly:
        # the diagonal is code 0 and any other column its index plus one.  Built in
        # slices, which keeps the int64 temporaries of the conversion off the peak.
        diag_first = self._diag_first()
        key = np.empty(n, dtype=np.int64)
        step = 1 << 26
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            r = rows[lo:hi].astype(np.int64)
            c = cols[lo:hi].astype(np.int64)
            if diag_first:
                on_diag = c == r
                c += 1
                c[on_diag] = 0
            r <<= self._KEY_SHIFT                      # in place: no temporaries
            r |= c
            key[lo:hi] = r
        del r, c
        if PATTERN_TIMING:
            _say_phase("key", time.perf_counter() - t_key, n)
        # Any valid argsort will do (equal keys get the same slot), so with the
        # kernels on a GPU it runs there.  On the host it is numpy's *stable* sort,
        # for speed rather than stability: its merge sort exploits the runs the
        # element order leaves in the keys, about twice as fast as an introsort.
        from .devsort import argsort_keys

        clock = time.perf_counter if PATTERN_TIMING else None
        t0 = clock() if clock else None
        order = argsort_keys(key)
        if clock:
            _say_phase("argsort", clock() - t0, n)
            t0 = clock()
        skey = key[order]
        del key                                  # peak memory: free as we go
        first = np.empty(n, dtype=bool)
        first[0] = True
        np.not_equal(skey[1:], skey[:-1], out=first[1:])
        # ``cumsum`` of a bool defaults to int64; the running index is bounded by the
        # nonzero count, so where that fits an int32 the array is half the bytes and
        # the scatter below moves half as much.  At a billion entries that is 4 GB of
        # traffic saved on a one-time cost that is mostly memory bandwidth.
        nnz_guess = int(np.count_nonzero(first))
        cum_t = np.int32 if nnz_guess <= np.iinfo(np.int32).max else np.int64
        slot_sorted = np.cumsum(first, dtype=cum_t)
        slot_sorted -= 1
        self.nnz = int(slot_sorted[-1]) + 1
        if clock:
            _say_phase("gather + unique", clock() - t0, n)
            t0 = clock()
        # int32 where it fits: with one entry per element-matrix entry this is the
        # largest array the pattern keeps and copies to the GPU, and unlike the
        # kernel's working set it cannot be chunked away.
        idx_t = np.int32 if self.nnz <= np.iinfo(np.int32).max else np.intp

        uk = skey[first]
        t_csr = time.perf_counter() if PATTERN_TIMING else None
        ucol = uk & self._KEY_MASK
        urow = uk >> self._KEY_SHIFT
        if diag_first:
            ucol = np.where(ucol == 0, urow, ucol - 1)
        counts = np.bincount(urow, minlength=self.nrow)
        indptr = np.zeros(self.nrow + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        self.indices = np.ascontiguousarray(ucol).astype(_HYPRE_INT, copy=False)
        self.indptr = indptr.astype(np.int32)
        if PATTERN_TIMING:
            _say_phase("CSR + diagonal first", time.perf_counter() - t_csr, n)
            t0 = clock()

        # Last, so that the diagonal-first order is already folded into the values written:
        # this is the array with one entry per element-matrix entry, and it is written
        # once and never rewritten.
        slot = np.empty(n, dtype=idx_t)
        slot[order] = slot_sorted
        del order, slot_sorted
        self.slot = slot
        if clock:
            _say_phase("scatter into slot", clock() - t0, n)

    # -------------------------------------------------------------- lookup
    def slots_of(self, rows, cols):
        """CSR slots of the local ``(row, col)`` pairs, which must all be in the graph.

        For entries that are not element-array entries of this pattern's groups:
        a boundary residual's element matrices, whose dofs are those of the
        adjacent volume element, land in the domain block's own slots this way,
        so one matrix is built with the boundary terms already in it (instead of
        two matrices added and eliminated afterwards).  One sort of the graph's
        keys per pattern, then one ``searchsorted`` per call.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if self._lookup is None:
            row_of = np.repeat(np.arange(self.nrow, dtype=np.int64),
                               np.diff(self.indptr.astype(np.int64)))
            keys = self._encode(row_of, np.asarray(self.indices, dtype=np.int64))
            order = np.argsort(keys, kind="stable")      # the diagonal-first rows
            self._lookup = (keys[order], order.astype(np.intp))
        skeys, order = self._lookup
        q = self._encode(rows, cols)
        pos = np.searchsorted(skeys, q)
        if pos.size and (pos.max() >= skeys.size or not np.array_equal(skeys[np.minimum(pos, skeys.size - 1)], q)):
            raise ValueError("%d of %d entries are not in the sparsity pattern"
                             % (int(np.count_nonzero(skeys[np.minimum(pos, skeys.size - 1)] != q)), q.size))
        return order[pos]

    # -------------------------------------------------------------- masking
    def masked_slots(self, row_mask, col_mask):
        """Indices of the CSR slots in the masked rows and columns.

        Zeroing these in the ``data`` array before the triple product is what
        essential-dof elimination does afterwards, and costs an indexed write of
        the eliminated entries rather than a second parallel matrix.  Cached on
        the identity of the two marker arrays, which are themselves cached per
        (space, dof set).
        """
        got = self._masked.get((row_mask, col_mask))
        if got is not None:
            return got
        if self.nnz == 0:
            kill = np.zeros(0, dtype=np.intp)
        else:
            cols = np.asarray(self.indices, dtype=np.int64)
            hit = (np.repeat(row_mask, np.diff(self.indptr.astype(np.int64)))
                   if row_mask is not None else np.zeros(self.nnz, dtype=bool))
            if col_mask is not None:
                hit = hit | col_mask[cols]
            kill = np.flatnonzero(hit).astype(np.intp)
        return self._masked.put((row_mask, col_mask), kill)

    # ------------------------------------------------------------------ reuse
    def reusable(self, roff, coff, same):
        """The cached ldof matrix for this partition, or ``None`` on first use."""
        return self._reuse.get((int(roff), int(coff), bool(same)))

    def make_reusable(self, roff, coff, same, A):
        """Record ``A`` so later assemblies can overwrite its values in place.

        hypre keeps its own copy of the arrays, and the order it stores a row in
        may differ from this pattern's.  The permutation is derived once by
        reading the structure back, one ``searchsorted`` over the row-major keys
        mapping stored positions to ours; ``A`` is not recorded if its structure
        cannot be addressed that way.
        """
        # ``blk`` is _diag_block's shared scratch wrapper (a private one crashes a
        # CUDA build, see _diag_block), so the kept view must not refer to it; the
        # buffer belongs to ``A``, which the reuse entry keeps.
        blk = _diag_block(A)
        n = blk.Height()
        I = np.asarray(blk.GetIArray())[:n + 1].copy()
        nnz = int(I[-1])
        stored_j = np.asarray(blk.GetJArray())[:nnz].copy()
        view = alias(np.asarray(blk.GetDataArray()))[:nnz]
        ours_j = np.asarray(self.indices, dtype=np.int64)
        if nnz != self.nnz or not np.array_equal(I, self.indptr):
            return                              # structure we cannot address
        if np.array_equal(stored_j, ours_j):
            order = None
        else:
            rowof = np.repeat(np.arange(n, dtype=np.int64),
                              np.diff(I.astype(np.int64)))
            ncol = np.int64(max(self.ncol, 1))
            order = np.searchsorted(rowof * ncol + ours_j,
                                    rowof * ncol + stored_j.astype(np.int64))
            if not np.array_equal(ours_j[order], stored_j):
                return                          # not a within-row permutation
        self._reuse[(int(roff), int(coff), bool(same))] = (A, view, order)
        self.keep(A)

    # ------------------------------------------------------------------ hypre
    def hypre_indices(self, roff, coff, same):
        """``(indptr, shifted indices, row starts, col starts)`` for hypre.

        Cached on ``(roff, coff, same)``: the pattern and a rank's ldof offsets
        are both fixed for the life of the spaces, so these arrays serve every one
        of an inverse problem's hundreds of assemblies.  ``None`` is returned for
        the column starts of a square block, where hypre must be handed one
        pointer for both partitions.
        """
        key = (int(roff), int(coff), bool(same))
        got = self._hypre.get(key)
        if got is None:
            I = np.ascontiguousarray(self.indptr, dtype=np.int32)
            J = np.ascontiguousarray(self.indices.astype(np.int64) + coff,
                                     dtype=_HYPRE_INT)
            rows = np.array([roff, roff + self.nrow], dtype=_HYPRE_INT)
            cols = (None if same else
                    np.array([coff, coff + self.ncol], dtype=_HYPRE_INT))
            got = self._hypre[key] = (I, J, rows, cols)
        return got

    # ------------------------------------------------------------------ values
    def _device_maps(self):
        """``slot``, ``perm`` and ``sign`` as device arrays, built once."""
        from .kernel import _put, device

        d = device()
        if d not in self._dev:
            self._dev[d] = (_put(self.slot),
                            None if self.perm is None else _put(self.perm),
                            None if self.sign is None else _put(self.sign))
        return self._dev[d]

    def data(self, element_matrices, zero_slots=None):
        """CSR ``data`` array for one set of element matrices.

        The reduction runs where the element matrices are: when the kernels ran on
        a GPU the scatter happens there and only the ``nnz`` assembled values come
        back, so the ``(ne, nd, nd)`` array never touches the host.
        """
        if len(element_matrices) != len(self.shapes):
            raise ValueError("expected %d element arrays, got %d"
                             % (len(self.shapes), len(element_matrices)))
        if self.nnz == 0:                       # no elements on this rank
            return np.zeros(0)
        on_device = not all(isinstance(E, np.ndarray) for E in element_matrices)
        if on_device:
            return host_writable(self._data_device(element_matrices, zero_slots))
        if len(element_matrices) == 1:
            flat = np.asarray(element_matrices[0], dtype=np.float64).reshape(-1)
        else:
            flat = np.concatenate([
                np.asarray(E, dtype=np.float64).reshape(-1)
                for E in element_matrices])
        if flat.size != self.slot.size:
            raise ValueError("element arrays hold %d entries, pattern expects %d"
                             % (flat.size, self.slot.size))
        if self.perm is not None:
            flat = flat[self.perm]
        if self.sign is not None:
            flat = flat * self.sign
        out = np.bincount(self.slot, weights=flat, minlength=self.nnz)
        if zero_slots is not None and zero_slots.size:
            out[zero_slots] = 0.0
        return out

    def fusable(self):
        """Whether chunks can be scattered as they are produced.

        A chunk of elements must map onto a slice of ``slot``, which holds unless
        the entries were permuted into mesh-element order across several geometry
        groups, the only case ``perm`` is set for.
        """
        return self.perm is None and self.nnz > 0

    def data_fused(self, chunks, zero_slots=None):
        """Scatter each chunk of element matrices as it arrives.

        The chunks are never glued into one ``(ne, nd, nd)`` array, which only the
        scatter would read and which would be the largest resident allocation of
        the assembly.

        ``chunks`` yields ``(group, start, stop, array)``.  Written as
        :meth:`fused_begin` / :meth:`fused_add` / :meth:`fused_end` so that a loop
        over chunks that carry *several* blocks can feed several patterns at once.
        """
        acc, maps = self.fused_begin()
        for g, a, bnd, arr in chunks:
            acc = self.fused_add(acc, maps, g, a, bnd, arr)
        return self.fused_end(acc, zero_slots)

    def fused_begin(self):
        """A zeroed accumulator on the current device, and the maps to feed it.

        The accumulator is the largest allocation an assembly makes, so it is
        borrowed from the buffers earlier assemblies kept and reset in place rather
        than allocated afresh (:data:`FUSED_KEEP`).
        """
        import jax.numpy as jnp

        from .kernel import device

        d = device()
        maps = self._fused_maps(d)
        if FUSED_KEEP and _acc_keepable(d, self.nnz):
            return _acc_take(d, self.nnz), maps
        _warn_if_growth_arena(d, self.nnz)
        return jnp.zeros(self.nnz, dtype=jnp.float64), maps

    def fused_add(self, acc, maps, g, a, bnd, arr):
        """Add one chunk's element matrices into the accumulator; returns it.

        The accumulator passed in is donated to the update (see :func:`_fused_add`),
        so only the returned one may be used afterwards.
        """
        import jax.numpy as jnp

        base, nent = maps["base"][g], maps["nent"][g]
        lo, hi = base + a * nent, base + bnd * nent
        flat = jnp.reshape(arr, (-1,))
        if flat.shape[0] != hi - lo:
            raise ValueError("chunk [%d, %d) of group %d holds %d entries, "
                             "pattern expects %d"
                             % (a, bnd, g, flat.shape[0], hi - lo))
        return _fused_add(acc, flat, maps["slot"][lo:hi],
                          None if maps["sign"] is None else maps["sign"][lo:hi])

    def fused_end(self, acc, zero_slots=None):
        """Bring the accumulator to the host and zero the eliminated slots there.

        Zeroing on the device would allocate a second accumulator, eagerly and even
        in a jitted update that donates the buffer, while the host copy is made
        anyway and zeros are exact wherever they are written.  The copy comes back
        writeable (:func:`host_writable`), as :meth:`data`'s does.
        """
        out = host_writable(acc)
        if zero_slots is not None and zero_slots.size:
            out[zero_slots] = 0.0
        if FUSED_KEEP:
            from .kernel import device

            d = device()
            if _acc_keepable(d, self.nnz):
                # The host copy is made; the device buffer goes back for the next
                # assembly of this pattern (or any other of the same nnz).
                _acc_give(d, self.nnz, acc)
        return out

    def _fused_maps(self, d):
        """Entry offsets, and the slot and sign maps a chunk is sliced from.

        These stay on the **host**.  Like the geometry, they are indexed one chunk
        at a time, so keeping them resident on the device buys nothing and costs a
        great deal: ``slot`` is one int32 per element-matrix entry, gigabytes on a
        large mesh, while a chunk's share transfers in milliseconds beside a chunk
        that takes seconds.
        """
        got = self._fused.get(d)
        if got is None:
            base, n = [], 0
            for size in self.sizes:
                base.append(n)
                n += int(size)
            got = self._fused[d] = {
                "slot": self.slot,
                "sign": self.sign,
                "base": base,
                "nent": [int(a) * int(b) for _, a, b in self.shapes],
            }
        return got

    def _data_device(self, element_matrices, zero_slots=None):
        """The same reduction, on the device the element matrices live on.

        ``zero_slots`` is applied here, on the device, rather than to the host copy.
        """
        import jax.numpy as jnp
        from jax.ops import segment_sum

        slot, perm, sign = self._device_maps()
        if len(element_matrices) == 1:
            flat = jnp.reshape(element_matrices[0], (-1,))
        else:
            flat = jnp.concatenate([jnp.reshape(E, (-1,))
                                    for E in element_matrices])
        if flat.size != self.slot.size:
            raise ValueError("element arrays hold %d entries, pattern expects %d"
                             % (flat.size, self.slot.size))
        if perm is not None:
            flat = flat[perm]
        if sign is not None:
            flat = flat * sign
        if DETERMINISTIC:
            out = self._gather_sum(flat)
        else:
            out = segment_sum(flat, slot, num_segments=self.nnz,
                              indices_are_sorted=False)
        if zero_slots is not None and zero_slots.size:
            out = out.at[self._device_zero(zero_slots)].set(0.0)
        return out

    def _device_zero(self, zero_slots):
        """``zero_slots`` on the current device, built once per (device, set)."""
        from .kernel import _put, device

        got = self._dev_zero.get((zero_slots,), device())
        if got is None:
            got = self._dev_zero.put((zero_slots,), _put(zero_slots), device())
        return got

    def _gather_sum(self, flat):
        """A reduction that does not depend on the order atomics happen to run in.

        ``segment_sum`` lowers to a scatter-add, which on a GPU is implemented with
        atomics: correct, but the summation order varies between runs, so two
        identical assemblies differ in the last bits.  Here each CSR slot instead
        *gathers* its own contributions from a fixed, padded index list and sums
        them in a fixed order, so repeated assemblies are bit-identical.

        The cost is ``nnz * maxc`` doubles of working memory, where ``maxc`` is the
        largest number of element entries landing in one slot (4 for P1
        quadrilaterals, 9 for P2, and bounded by the number of elements meeting at
        a dof).  Selected by ``HIPPYMFEM_GPU_DETERMINISTIC=1``.
        """
        import jax.numpy as jnp
        from .kernel import _put

        d = _device_key()
        if d not in self._dev_pad:
            self._dev_pad[d] = _put(self._pad_index())
        idx = self._dev_pad[d]
        ext = jnp.concatenate([flat, jnp.zeros((1,), dtype=flat.dtype)])
        return jnp.sum(ext[idx], axis=1)

    def _pad_index(self):
        """``(nnz, maxc)`` gather map; unused positions point at an appended zero."""
        if self._pad is None:
            order = np.argsort(self.slot, kind="stable")
            counts = np.bincount(self.slot, minlength=self.nnz)
            maxc = int(counts.max()) if counts.size else 0
            pad = np.full((self.nnz, maxc), self.slot.size, dtype=np.int64)
            starts = np.zeros(self.nnz + 1, dtype=np.int64)
            np.cumsum(counts, out=starts[1:])
            within = np.arange(self.slot.size) - np.repeat(starts[:-1], counts)
            pad[self.slot[order], within] = order
            self._pad = pad
            self.maxc = maxc
        return self._pad

    def __repr__(self):
        return "ScatterPattern(%dx%d, nnz=%d, entries=%d)" % (
            self.nrow, self.ncol, self.nnz, self.slot.size)


#: Print how long each phase of a pattern build took, on rank 0.  The build is the
#: one-time cost a large run pays before its first assembly (115 s of the 121 s a
#: 256^3 linearization point took on 8 H100), and which phase carries it depends on
#: where the sort runs, so it is worth being able to ask.  Set
#: ``HIPPYMFEM_PATTERN_TIMING=1``.
PATTERN_TIMING = os.environ.get("HIPPYMFEM_PATTERN_TIMING", "").lower() in (
    "1", "true", "yes", "on")


def _say_phase(what, seconds, n):
    """One line per phase of a pattern build, on rank 0 only."""
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.rank:
            return
    except Exception:                                        # noqa: BLE001
        pass
    print("  pattern %-22s %8.3f s  (%6.1f ns/entry, %.2f G entries)"
          % (what, seconds, 1e9 * seconds / max(n, 1), n / 1e9), flush=True)


_PATTERN_CACHE = IdentityCache()


def get_pattern(test_space, trial_space, groups):
    """Cached :class:`ScatterPattern` for a space pair on a set of element groups.

    The cache is keyed on the identity of the two spaces and the groups, and
    building a pattern involves no communication, so a hit on one rank and a miss
    on another is harmless.  (A cache that wraps a collective cannot be keyed this
    way; see :class:`~hippymfem.common.parvector.Layout`.)
    """
    test_space = as_space(test_space)
    trial_space = as_space(trial_space)
    objs = (test_space.fes, trial_space.fes) + tuple(groups)
    pat = _PATTERN_CACHE.get(objs)
    if pat is None:
        pat = ScatterPattern(
            [group_tables(test_space, g) for g in groups],
            [group_tables(trial_space, g) for g in groups],
            test_space=test_space, trial_space=trial_space,
        )
        pat.keep(test_space, trial_space, *groups)
        _PATTERN_CACHE.put(objs, pat)
    return pat


def clear_pattern_cache():
    """Drop every cached pattern (for tests, and after a mesh is refined)."""
    _PATTERN_CACHE.clear()
    _VECTOR_CACHE.clear()
    # Everything keyed on a space goes too.  (A freed space takes its own
    # IdentityCache entries with it; this is for a refined mesh, whose objects
    # live on, and for the tests.)
    _IDENTITY_P.clear()
    _BOOLEAN_P.clear()
    _ESS_LDOF.clear()
    _TRIPLE_CHOICE.clear()
    _TRANSPOSE.clear()
    clear_accumulators()
    from .tdofassemble import clear_tdof_cache

    clear_tdof_cache()


class VectorPattern(KeepAlive):
    """Local dof indices of every element-vector entry, in mesh-element order."""

    def __init__(self, tables, space=None):
        # As in ScatterPattern: the length is the space's, not the first table's,
        # because a rank may own no elements of this group at all.
        self.n = int((space if space is not None else tables[0].space).fes.GetVSize())
        self.sizes = [int(t.group.ne) * int(t.nd_total) for t in tables]
        edofs = [t.edofs.reshape(-1) for t in tables]
        signs = [t.signs.reshape(-1) if t.signs.min() < 0 else None
                 for t in tables]
        self.edofs = (np.concatenate(edofs) if edofs
                      else np.zeros(0, np.int64))
        self.sign = (np.concatenate([
            s if s is not None else np.ones(k) for s, k in zip(signs, self.sizes)
        ]) if any(s is not None for s in signs) else None)
        self.perm = None
        self._dev = {}
        if len(tables) > 1:
            eidx = np.concatenate([np.repeat(t.group.elems, t.nd_total)
                                   for t in tables])
            perm = np.argsort(eidx, kind="stable")
            if not np.array_equal(perm, np.arange(perm.size)):
                # int32 where it fits, as in ScatterPattern
                self.perm = perm.astype(np.int32, copy=False) \
                    if perm.size <= np.iinfo(np.int32).max else perm
                self.edofs = self.edofs[perm]
                if self.sign is not None:
                    self.sign = self.sign[perm]

    def scatter(self, element_vectors):
        """Local dof array holding the summed element vectors.

        Reduced where the element vectors are, as in :meth:`ScatterPattern.data`.
        """
        if not element_vectors:                 # this rank owns no such elements
            return np.zeros(self.n)
        if not all(isinstance(v, np.ndarray) for v in element_vectors):
            return np.asarray(self._scatter_device(element_vectors))
        if len(element_vectors) == 1:
            flat = np.asarray(element_vectors[0], dtype=np.float64).reshape(-1)
        else:
            flat = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1)
                                   for v in element_vectors])
        if flat.size != self.edofs.size:
            raise ValueError("element vectors hold %d entries, pattern expects %d"
                             % (flat.size, self.edofs.size))
        if self.perm is not None:
            flat = flat[self.perm]
        if self.sign is not None:
            flat = flat * self.sign
        return np.bincount(self.edofs, weights=flat, minlength=self.n)

    def _scatter_device(self, element_vectors):
        import jax.numpy as jnp
        from jax.ops import segment_sum

        from .kernel import _put, device

        d = device()
        if d not in self._dev:
            self._dev[d] = (_put(self.edofs),
                            None if self.perm is None else _put(self.perm),
                            None if self.sign is None else _put(self.sign))
        idx, perm, sign = self._dev[d]
        if len(element_vectors) == 1:
            flat = jnp.reshape(element_vectors[0], (-1,))
        else:
            flat = jnp.concatenate([jnp.reshape(v, (-1,))
                                    for v in element_vectors])
        if perm is not None:
            flat = flat[perm]
        if sign is not None:
            flat = flat * sign
        return segment_sum(flat, idx, num_segments=self.n,
                           indices_are_sorted=False)


_VECTOR_CACHE = IdentityCache()


def get_vector_pattern(space, groups):
    """Cached :class:`VectorPattern` for a space on a set of element groups."""
    space = as_space(space)
    objs = (space.fes,) + tuple(groups)
    pat = _VECTOR_CACHE.get(objs)
    if pat is None:
        pat = VectorPattern([group_tables(space, g) for g in groups], space=space)
        pat.keep(space, *groups)
        _VECTOR_CACHE.put(objs, pat)
    return pat

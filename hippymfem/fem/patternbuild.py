# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""A sparsity pattern built without a global sort.

:class:`~hippymfem.fem.pattern.ScatterPattern` needs the CSR structure of the union of
all element couplings and, for every element-matrix entry, its position in the CSR data
array.  The general route sorts one key per entry to bring equal ``(row, col)`` pairs
together: at 256\\ :sup:`3` on 8 H100 that is 1.5 billion keys a rank, and it is most of
the time a large run spends before its first assembly.

The sort is not needed.  An element matrix contributes, for each of its rows, a
contiguous *run* of entries sharing that row, so grouping the entries by row is a
counting sort over the runs (one pass, about ``n / nd`` items).  A row then holds some
ninety entries, and those are ordered by one short sort of packed values, ``(code << 16)
| q``: ``q`` is the entry's place in the row and ``code`` its column, encoded so that the
diagonal of a square pattern is 0 and any other column its index plus one.  That sort
puts the row in exactly the order hypre keeps a square matrix in, diagonal first and the
rest ascending, and carries every entry's identity with it, so one linear scan yields the
row's distinct columns and each entry's final position.  Nothing global is sorted and
nothing is searched.

The output is identical to the sort route's, array for array (``test_assembly`` checks
it).  Measured on one host at 96\\ :sup:`3` (645 million entries): 20.0 s on one thread,
6.6 s on eight, 2.8 s on thirty-two, against 27.3 s for the sort route with the sort on
an L40S.  It uses no device memory at all, and about half the host memory of the sort
route, since it never forms the 64-bit key, its argsort or the sorted copy.

The true-dof pattern (:mod:`~hippymfem.fem.tdofassemble`) merges its own entries with
those other ranks send through the same grouping (:func:`unique_inverse_rows`), and lays
out hypre's diagonal and off-diagonal blocks with one compiled pass over the rows
(:func:`block_slots`); at 128\\ :sup:`3` on two L40S those took 17.0 s and 13.7 s as sorts
and mask passes, and take 5.7 s and 0.42 s.

It needs numba, and is used when numba is importable and the pattern is large enough to
repay the compilation (:data:`MIN_ENTRIES`); otherwise the sort route runs, unchanged.
"""

import os

import numpy as np

#: ``auto`` uses this builder when numba is importable and the pattern has at least
#: :data:`MIN_ENTRIES` entries, ``numba`` forces it, ``sort`` never uses it.  Set
#: ``HIPPYMFEM_PATTERN_BUILDER``.
MODE = os.environ.get("HIPPYMFEM_PATTERN_BUILDER", "auto").strip().lower()
#: Below this many element-matrix entries the sort route is quicker than compiling
#: this one; above it the compilation is a rounding error.  Set
#: ``HIPPYMFEM_PATTERN_BUILDER_MIN``.
MIN_ENTRIES = int(os.environ.get("HIPPYMFEM_PATTERN_BUILDER_MIN", str(2 ** 24)) or 0)
#: Threads for the build; ``0``, the default, takes this rank's share of the node (see
#: :func:`threads`).  Set ``HIPPYMFEM_PATTERN_THREADS``.
THREADS = int(os.environ.get("HIPPYMFEM_PATTERN_THREADS", "0") or 0)

#: Bits given to an entry's place within its row in the packed per-row key.
QBITS = 16

_KERNELS = None
_USABLE = None


def usable():
    """Whether numba is importable here, asked once."""
    global _USABLE
    if _USABLE is None:
        if MODE == "sort":
            _USABLE = False
        else:
            try:
                import numba  # noqa: F401

                _USABLE = True
            except Exception:                                    # noqa: BLE001
                _USABLE = False
    return _USABLE


def wanted(n):
    """Whether a pattern of ``n`` entries should be built here rather than by sorting."""
    if MODE == "sort" or not usable():
        return False
    return MODE == "numba" or n >= MIN_ENTRIES


def threads():
    """This rank's share of the cores, never more than its affinity mask allows.

    numba's default is every core of the machine, which on a node shared by 32 ranks
    would put 2048 threads on 64 cores; the element kernels' XLA pool did exactly that
    and a 64^3 run on 32 ranks never finished.  So the count is the smaller of the cores
    this process may run on and the node's cores divided by the ranks on it.
    """
    if THREADS > 0:
        return THREADS
    try:
        allowed = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = os.cpu_count() or 1
    from .kernel import _node_ranks

    share = max((os.cpu_count() or 1) // max(_node_ranks(), 1), 1)
    return max(min(allowed, share), 1)


def _kernels():
    """The compiled passes, from :mod:`._patternkernels` (numba caches them on disk)."""
    global _KERNELS
    if _KERNELS is None:
        from . import _patternkernels as pk

        _KERNELS = (pk.runs, pk.group_runs, pk.order_rows, pk.place_rows)
    return _KERNELS


def build(rows, cols, nrow, ncol, diag_first, index_type, indptr_type=np.int32):
    """``(indptr, indices, slot, nnz)`` for flat ``rows``/``cols``, without a global sort.

    ``indptr`` is ``indptr_type`` (int32, as the scatter pattern keeps it) and ``slot`` of
    the width the nonzero count needs, as the sort route produces them; ``indices`` has
    ``index_type`` (hypre's integer).  With ``diag_first`` false the rows come out in
    plain column order, so ``slot`` is exactly the inverse ``np.unique`` would return
    for the packed keys and ``indices`` the unique columns in key order.
    """
    import numba

    runs, group_runs, order_rows, place_rows = _kernels()
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    n = rows.size
    col_t = np.int32 if ncol <= np.iinfo(np.int32).max else np.int64
    cols = np.ascontiguousarray(cols, dtype=col_t)
    old = numba.get_num_threads()
    numba.set_num_threads(min(threads(), numba.config.NUMBA_NUM_THREADS))
    try:
        rstart = runs(rows)
        rcount, ecount, runs_of_row = group_runs(rows, rstart, int(nrow))
        maxlen = int(np.max(np.diff(ecount))) if nrow else 1
        if maxlen >= (1 << QBITS):
            raise ValueError("a row holds %d entries, more than the packed per-row key "
                             "allows (%d)" % (maxlen, (1 << QBITS) - 1))
        # positions are bounded by the entry count until the nonzeros are known
        wide = np.int32 if n <= np.iinfo(np.int32).max else np.int64
        scratch = np.empty(n, col_t)
        ucount = np.zeros(int(nrow), np.int64)
        slot = np.empty(n, wide)
        order_rows(cols, rstart, rcount, ecount, runs_of_row, int(nrow), bool(diag_first),
                   scratch, ucount, slot, max(maxlen, 1), 4096)
        indptr = np.zeros(int(nrow) + 1, np.int64)
        np.cumsum(ucount, out=indptr[1:])
        nnz = int(indptr[-1])
        indices = np.empty(nnz, col_t)
        place_rows(rstart, rcount, ecount, runs_of_row, indptr, int(nrow), scratch,
                   indices, slot)
    finally:
        numba.set_num_threads(old)
    idx_t = np.int32 if nnz <= np.iinfo(np.int32).max else np.intp
    if slot.dtype != idx_t:
        slot = slot.astype(idx_t)
    return (indptr.astype(indptr_type, copy=False), indices.astype(index_type, copy=False),
            slot, nnz)


def unique_inverse_rows(rows, cols, nrow, ncol):
    """``(indptr, ucol, inverse)`` of the ``(row, col)`` pairs, without a global sort.

    What ``np.unique(row * ncol + col, return_inverse=True)`` gives, decoded: the
    distinct pairs in row-then-column order (``indptr`` over rows, ``ucol`` their
    columns) and, for each input pair, the index of its distinct pair.  The true-dof
    pattern's merge of own and received entries is this, and on a large mesh it is the
    largest sort a build does after the scatter pattern's.
    """
    indptr, ucol, inv, _ = build(rows, cols, nrow, ncol, False, np.int64,
                                 indptr_type=np.int64)
    return indptr, ucol, inv


def block_slots(urow, rowstart, in_diag, lead, nrow):
    """The slot layout of :func:`~hippymfem.fem.tdofassemble._slot_order`, compiled."""
    import numba

    from . import _patternkernels as pk

    newslot = np.empty(urow.size, np.int64)
    old = numba.get_num_threads()
    numba.set_num_threads(min(threads(), numba.config.NUMBA_NUM_THREADS))
    try:
        pk.block_slots(np.ascontiguousarray(urow, dtype=np.int64),
                       np.ascontiguousarray(rowstart, dtype=np.int64),
                       np.ascontiguousarray(in_diag, dtype=np.bool_),
                       np.ascontiguousarray(lead, dtype=np.bool_), int(nrow), newslot)
    finally:
        numba.set_num_threads(old)
    return newslot

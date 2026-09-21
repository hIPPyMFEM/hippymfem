# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Assembly straight into true-dof rows when the prolongation is boolean.

For a boolean ``P`` the triple product ``P^T A P`` is a permutation with a sum over
shared dofs: entry ``(i, j)`` of the ldof matrix lands at ``(t(i), t(j))``, ``t``
being the ldof-to-true-dof map.  hypre's ``RAP`` does symbolic work over every
nonzero instead, several times the element kernel's cost, although on a few ranks
only a percent or two of the entries sit in ghost rows.

So the pattern is re-targeted at true-dof numbering, once, and every assembly then

* scatters the entries whose *row* this rank owns straight into the true-dof CSR,
  the same reduction as the ldof scatter with a different slot map;
* scatters the entries whose row belongs to another rank into a send buffer that is
  contiguous per destination, and exchanges it with one ``Alltoallv`` of values;
* adds what arrived through a slot map fixed once;
* builds the ``HypreParMatrix`` from hypre's diagonal and off-diagonal blocks, which
  the slot layout makes two contiguous views of the accumulator.

**Two things it does not do.**  It does not handle a ``P`` with real interpolation
weights, for which the permutation picture is false: a nonconforming ``P``, or that
of a space with a ``DofTransformation`` (H(curl) at order two and up on tetrahedra,
prisms and pyramids), into which MFEM folds a shared face's transformation on the
rank that does not own it.  The caller decides with
:func:`hippymfem.fem.prolongation._boolean_prolongation`, collectively, and falls
back to the triple product for both.  And it does not reproduce the triple product's
summation order at shared dofs (own contributions are summed first, received ones
after), so on more than one rank it agrees with ``RAP`` to round-off, not bit for
bit.  On one rank ``P`` is the identity and this route is never taken.  The
measurements behind the route and its slot layout are in
``benchmarks/DESIGN_NOTES.md``, section 3.
"""

import copy
import os
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from ..common.identitycache import IdentityCache
from ..common.linalg import own
from ..common.parvector import _HYPRE_INT
from . import pattern as _pattern, patternbuild
from .devsort import unique_inverse
from .pattern import host_writable

#: How a parallel matrix is built with hypre on a device.  ``"block"``, the default,
#: builds hypre's two blocks directly from this pattern; ``"copy"``, the fallback,
#: hands MFEM one row-major CSR with global columns and lets it re-derive them.  The
#: block constructor makes a forward solve and a warm Hessian assembly about a third
#: faster, with identical results.  Every rank must agree, so set it through the
#: launcher's environment (``HIPPYMFEM_PARMAT_DEVICE``).
DEVICE_PARMAT = os.environ.get("HIPPYMFEM_PARMAT_DEVICE", "block").lower()


def set_device_parmat(mode):
    """Choose the device matrix constructor, returning the previous choice."""
    global DEVICE_PARMAT

    if mode not in ("copy", "block"):
        raise ValueError("HIPPYMFEM_PARMAT_DEVICE must be copy or block, got %r" % (mode,))
    old, DEVICE_PARMAT = DEVICE_PARMAT, mode
    return old


def _slot_order(urow, ucol, in_diag, lead, nrow):
    """The block-major, diagonal-leading slot of each entry of a row-sorted pattern.

    ``urow``/``ucol`` are sorted by row then column (a ``np.unique`` of row-major
    keys).  Returns ``newslot`` such that ``newslot[i]`` is where entry ``i`` goes in
    the order ``np.lexsort((ucol, ~lead, urow, ~in_diag))`` would produce: every
    diagonal-block entry first, row by row, each row's ``lead`` entry ahead of the
    rest of its row, then every off-diagonal entry row by row; within each of those
    groups the input (column) order.  Linear passes: bincounts for the group sizes,
    cumulative counts for the rank of an entry within its row's group.
    """
    nnz = int(urow.size)
    if nnz == 0:
        return np.zeros(0, dtype=np.int64)
    rowstart = np.zeros(nrow + 1, dtype=np.int64)
    np.cumsum(np.bincount(urow, minlength=nrow), out=rowstart[1:])
    ndiag = np.bincount(urow[in_diag], minlength=nrow)
    nnz_diag = int(ndiag.sum())
    dstart = np.zeros(nrow, dtype=np.int64)
    np.cumsum(ndiag[:-1], out=dstart[1:])
    ostart = np.full(nrow, nnz_diag, dtype=np.int64)
    if nrow > 1:
        ostart[1:] += np.cumsum(np.bincount(urow[~in_diag], minlength=nrow)[:-1])
    haslead = np.bincount(urow[lead], minlength=nrow)
    newslot = np.empty(nnz, dtype=np.int64)
    rest = in_diag & ~lead
    for mask, start, extra in ((rest, dstart, haslead), (~in_diag, ostart, None)):
        cum = np.cumsum(mask)                          # class entries up to and including i
        before = np.concatenate(([0], cum))[rowstart[:-1]]   # class entries before each row
        rank = (cum - 1) - before[urow]                # rank within the row's group
        pos = start[urow] + rank
        if extra is not None:
            pos += extra[urow]
        newslot[mask] = pos[mask]
        del cum, before, rank, pos
    newslot[lead] = dstart[urow[lead]]
    return newslot


class TrueDofPattern:
    """An ldof :class:`~hippymfem.fem.csrassemble.ScatterPattern` re-targeted at true dofs.

    ``target`` is a shallow copy of the ldof pattern whose ``slot`` map sends every
    element entry either to a true-dof CSR slot (row owned here) or to a position in
    the send buffer (row owned elsewhere), so the pattern's unchanged ``data`` and
    ``data_fused`` produce both in one reduction.  The copy has its own device
    caches, since its slot map differs.  :meth:`finish` then exchanges the send
    buffer and builds the matrix.

    The symbolic work (dof maps, exchange pattern, merged true-dof graph) is done
    here, once per (pattern, spaces), at the same order of cost as building the ldof
    pattern.
    """

    def __init__(self, pattern, test_space, trial_space):
        from .prolongation import _prolongation

        comm = test_space.comm
        size = comm.size
        fes_t, fes_r = test_space.fes, trial_space.fes
        same = fes_t is fes_r
        nld_t = int(fes_t.GetVSize())
        gt_t, lt_t = _tdof_maps(fes_t)
        if same:
            gt_r = gt_t
        else:
            gt_r = _tdof_maps(fes_r)[0]
        owned = lt_t >= 0
        # Partitions from the prolongations themselves, so this matrix and the one
        # the triple product would have built cannot disagree about who owns what.
        Pt, _ = _prolongation(test_space)
        Pr = Pt if same else _prolongation(trial_space)[0]
        cpt = Pt.GetColPartArray()
        cpr = Pr.GetColPartArray()
        self.t0, self.t1 = int(cpt[0]), int(cpt[1])
        self.c0, self.c1 = int(cpr[0]), int(cpr[1])
        self.gnr, self.gnc = int(Pt.GetGlobalNumCols()), int(Pr.GetGlobalNumCols())
        self.ntd = self.t1 - self.t0
        starts = np.array(comm.allgather(self.t0) + [self.gnr], dtype=np.int64)

        I = np.asarray(pattern.indptr, dtype=np.int64)
        J = np.asarray(pattern.indices, dtype=np.int64)
        rows_l = np.repeat(np.arange(nld_t, dtype=np.int64), np.diff(I))
        rg, cg, mine = gt_t[rows_l], gt_r[J], owned[rows_l]
        del rows_l, I, J

        # Ghost entries, grouped by the rank that owns their row.  The order fixed
        # here is the order the send buffer is filled in on every assembly.
        gidx = np.flatnonzero(~mine)
        dest = np.searchsorted(starts, rg[gidx], side="right") - 1
        o = np.argsort(dest, kind="stable")
        gidx, dest = gidx[o], dest[o]
        del o
        scnt = np.bincount(dest, minlength=size).astype(np.int64)
        rcnt = np.empty(size, dtype=np.int64)
        comm.Alltoall([scnt, MPI.INT64_T], [rcnt, MPI.INT64_T])
        sdsp = np.concatenate([[0], np.cumsum(scnt)[:-1]]).astype(np.int64)
        rdsp = np.concatenate([[0], np.cumsum(rcnt)[:-1]]).astype(np.int64)
        nsend, nrecv = int(scnt.sum()), int(rcnt.sum())
        r_rg = np.empty(nrecv, dtype=np.int64)
        r_cg = np.empty(nrecv, dtype=np.int64)
        comm.Alltoallv([np.ascontiguousarray(rg[gidx]), scnt, sdsp, MPI.INT64_T],
                       [r_rg, rcnt, rdsp, MPI.INT64_T])
        comm.Alltoallv([np.ascontiguousarray(cg[gidx]), scnt, sdsp, MPI.INT64_T],
                       [r_cg, rcnt, rdsp, MPI.INT64_T])

        # The true-dof graph: this rank's rows, global columns, from its own entries
        # and the ones that arrived.  One key per entry orders by row then column.
        n_own = int(mine.sum())
        t0 = time.perf_counter()
        rowstart = None
        if patternbuild.wanted(n_own + nrecv):
            # Without a sort: an own row's entries are already distinct (the ldof ->
            # true-dof map is injective on this rank), so duplicates come only from
            # rows other ranks sent, and grouping by row then ordering each short row
            # (patternbuild) gives np.unique's pairs and inverse exactly.
            rowstart, ucol, inv = patternbuild.unique_inverse_rows(
                np.concatenate([rg[mine] - self.t0, r_rg - self.t0]),
                np.concatenate([cg[mine], r_cg]), self.ntd, self.gnc)
            del rg, cg, r_rg, r_cg
            nnz_t = int(ucol.size)
            urow = np.repeat(np.arange(self.ntd, dtype=np.int64), np.diff(rowstart))
            ucol = ucol.astype(np.int64, copy=False)
        else:
            keys = np.concatenate([
                (rg[mine] - self.t0) * np.int64(self.gnc) + cg[mine],
                (r_rg - self.t0) * np.int64(self.gnc) + r_cg])
            del rg, cg, r_rg, r_cg
            # np.unique on the host, whose inverse uses a stable (merge) sort: the keys
            # arrive grouped by row, where merge sort is several times faster than
            # introsort.  With the kernels on a GPU the sort runs on the device in chunks.
            uniq, inv = unique_inverse(keys)
            del keys
            nnz_t = int(uniq.size)
            urow = uniq // np.int64(self.gnc)
            ucol = uniq % np.int64(self.gnc)
            del uniq
        if _pattern.PATTERN_TIMING:
            _pattern._say_phase("true-dof unique", time.perf_counter() - t0, n_own + nrecv)
        # Slot layout.  hypre keeps a parallel matrix as two CSR blocks (the columns
        # this rank owns, and the rest), so the slots are block-major: every
        # diagonal-block entry first, row by row, then every off-diagonal one.  Both
        # blocks are then contiguous views of one accumulator, which MFEM's block
        # constructor aliases without a copy on the host.  Within a row of the
        # diagonal block the diagonal entry leads: hypre reorders a square matrix to
        # that order and writes the permutation back, but leaves alone a matrix that
        # already has it.
        in_diag = (ucol >= self.c0) & (ucol < self.c1)
        lead = np.zeros(nnz_t, dtype=bool)
        if self.c1 - self.c0 == self.ntd:            # a locally square diagonal block
            lead = in_diag & (ucol - self.c0 == urow)
        # Block, row, lead, column order without a sort: ``urow``/``ucol`` are already
        # sorted by row and column, so the only moves are the off-diagonal entries to
        # the second block and the diagonal to the front of its row, which
        # ``_slot_order`` makes in linear passes (a four-key lexsort costs about half
        # the pattern build).
        t0 = time.perf_counter()
        if patternbuild.wanted(nnz_t):
            if rowstart is None:
                rowstart = np.zeros(self.ntd + 1, dtype=np.int64)
                np.cumsum(np.bincount(urow, minlength=self.ntd), out=rowstart[1:])
            newslot = patternbuild.block_slots(urow, rowstart, in_diag, lead, self.ntd)
        else:
            newslot = _slot_order(urow, ucol, in_diag, lead, self.ntd)
        del rowstart
        if _pattern.PATTERN_TIMING:
            _pattern._say_phase("true-dof slot order", time.perf_counter() - t0, nnz_t)
        nnz_diag = int(in_diag.sum())
        r2 = np.empty(nnz_t, dtype=np.int64); r2[newslot] = urow
        c2 = np.empty(nnz_t, dtype=np.int64); c2[newslot] = ucol
        del in_diag, lead
        cmap = np.unique(c2[nnz_diag:])
        self.n_offd = int(cmap.size)
        # Which constructor, agreed by all ranks.  The copying 9-argument constructor
        # and the diag/offd one communicate differently (one allreduces the global
        # nonzero count, the other builds its comm package first), so ranks choosing
        # by their *own* ``n_offd`` would deadlock inside MFEM.  The copy route is
        # taken only when no rank has an off-diagonal column.
        self.all_local = not bool(comm.allreduce(int(self.n_offd), op=MPI.MAX))
        # The copying constructor (``_finish_copy``) takes the graph as one row-major
        # CSR with global columns, which ``urow``/``ucol`` already are, with
        # ``newslot`` giving each entry's slot.  These arrays are kept only where that
        # constructor is the route now (the host with no off-diagonal column anywhere,
        # or a device with ``HIPPYMFEM_PARMAT_DEVICE=copy``), and otherwise rebuilt
        # from the two blocks on first use: ``J_t`` alone can be a gigabyte per rank,
        # and the block constructor never reads it.
        from ..common.parvector import device_active

        if self.all_local or (device_active() and DEVICE_PARMAT != "block"):
            indptr = np.zeros(self.ntd + 1, dtype=np.int64)
            np.cumsum(np.bincount(urow, minlength=self.ntd), out=indptr[1:])
            self._rm = (indptr.astype(np.int32),
                        np.ascontiguousarray(ucol).astype(_HYPRE_INT, copy=False),
                        None if np.array_equal(newslot, np.arange(nnz_t)) else
                        newslot.astype(np.int32 if nnz_t < 2 ** 31 else np.int64))
            del indptr
        else:
            self._rm = None
        del urow, ucol
        self.I_diag = np.zeros(self.ntd + 1, dtype=np.int32)
        np.cumsum(np.bincount(r2[:nnz_diag], minlength=self.ntd), out=self.I_diag[1:])
        self.J_diag = np.ascontiguousarray(c2[:nnz_diag] - self.c0).astype(np.int32)
        self.I_offd = np.zeros(self.ntd + 1, dtype=np.int32)
        np.cumsum(np.bincount(r2[nnz_diag:], minlength=self.ntd), out=self.I_offd[1:])
        self.J_offd = np.ascontiguousarray(np.searchsorted(cmap, c2[nnz_diag:])).astype(np.int32)
        self.nnz_diag = nnz_diag
        del r2, c2
        # Pointers MFEM's block constructor wants, built by index: an intArray made
        # from a list or an array can hand back a pointer into storage that does not
        # outlive the call, and the matrix built on it would be wrong.
        self.ia_rows = mfem.intArray(2); self.ia_rows[0] = self.t0; self.ia_rows[1] = self.t1
        self.ia_cols = mfem.intArray(2); self.ia_cols[0] = self.c0; self.ia_cols[1] = self.c1
        self.ia_cmap = mfem.intArray(max(self.n_offd, 1))
        for k in range(self.n_offd):
            self.ia_cmap[k] = int(cmap[k])
        self.cmap = cmap.astype(np.int64)

        # ldof CSR slot -> target: a true-dof slot for an own row, a send-buffer
        # position (past the true-dof slots) for a ghost row.
        tslot = np.empty(pattern.nnz, dtype=np.int64)
        tslot[np.flatnonzero(mine)] = newslot[inv[:n_own]]
        tslot[gidx] = nnz_t + np.arange(nsend, dtype=np.int64)
        self.slot_recv = np.ascontiguousarray(newslot[inv[n_own:]])
        del inv, gidx, mine, newslot
        self.tslot = tslot
        self.nnz_t, self.nsend, self.nrecv = nnz_t, nsend, nrecv
        # Whether *any* rank exchanges, agreed by all of them: skipping the
        # ``Alltoallv`` when *this* rank has nothing to send or receive would be a
        # collective behind a rank-local condition (a rank without boundary elements
        # would skip it and the others would wait in it forever).
        self.exchange = bool(comm.allreduce(nsend + nrecv, op=MPI.MAX))
        self.scnt, self.sdsp, self.rcnt, self.rdsp = scnt, sdsp, rcnt, rdsp
        self.comm, self.same = comm, same
        self.rows_t = np.array([self.t0, self.t1], dtype=_HYPRE_INT)
        self.cols_t = np.array([self.c0, self.c1], dtype=_HYPRE_INT)
        self._kill = IdentityCache()
        self._diag = IdentityCache()

        # The scatter target: the ldof pattern with the composed slot map, the
        # accumulator widened by the send buffer, and its per-device caches reset.
        tp = copy.copy(pattern)
        width = nnz_t + nsend
        idx_t = np.int32 if width <= np.iinfo(np.int32).max else np.intp
        tp.slot = tslot[pattern.slot].astype(idx_t, copy=False)
        tp.nnz = width
        tp.nrow, tp.ncol = self.ntd, self.gnc
        tp.indptr, tp.indices = self.I_diag, self.J_diag   # the diagonal block's
        for name in ("_hypre", "_fused", "_reuse", "_dev", "_dev_pad"):
            setattr(tp, name, {})
        tp._masked, tp._dev_zero = IdentityCache(), IdentityCache()
        tp._pad = None
        tp.maxc = None
        self.target = tp

    def kill(self, ldof_kill):
        """Target slots of a set of ldof CSR slots, for the folded elimination.

        Masking an entry before it is sent is the same as masking it after it lands,
        since ``P`` is boolean here.  Cached on the identity of the ldof set.
        """
        if ldof_kill is None:
            return None
        got = self._kill.get((ldof_kill,))
        if got is None:
            got = self._kill.put((ldof_kill,), self.tslot[ldof_kill])
        return got

    def diagonal_slots(self, ess):
        """Accumulator slots of the diagonal entries of the local true-dof rows ``ess``.

        For the folded elimination: the eliminated rows' diagonal is written into the
        accumulator before the matrix exists, because writing into a built matrix
        moves the whole matrix between host and device when hypre runs on one.  The
        diagonal leads its row in this layout (see ``__init__``), so the slot is
        ``I_diag[i]`` after one check; a row where it does not is searched, and a row
        with no diagonal entry at all raises ``RuntimeError`` here rather than
        leaving a singular matrix.  Cached on the identity of ``ess``.
        """
        got = self._diag.get((ess,))
        if got is None:
            idx = np.asarray(ess.ToList(), dtype=np.int64)
            slots = np.zeros(0, dtype=np.int64)
            if idx.size:
                lo = self.I_diag[idx].astype(np.int64)
                hi = self.I_diag[idx + 1].astype(np.int64)
                slots = lo.copy()
                lead = hi > lo
                lead[lead] = self.J_diag[lo[lead]] == idx[lead]
                missing = 0
                for k in np.flatnonzero(~lead):
                    hit = np.flatnonzero(self.J_diag[lo[k]:hi[k]] == idx[k])
                    if hit.size:
                        slots[k] = lo[k] + int(hit[0])
                    else:
                        missing += 1
                if missing:
                    raise RuntimeError(
                        "%d of %d eliminated rows have no diagonal entry in the "
                        "pattern; the folded elimination relies on it.  Set "
                        "HIPPYMFEM_FOLD_ELIMINATION=0." % (missing, idx.size))
            got = self._diag.put((ess,), slots)
        return got

    def finish(self, acc, diagonal=None):
        """Exchange the send buffer of a scattered accumulator and build the matrix.

        ``diagonal`` is ``(slots, value)``, the slots from :meth:`diagonal_slots`; it
        is written into the accumulator after the exchange, so no route has to write
        into a built matrix.

        On the host the matrix *aliases* ``acc``: its two blocks are the leading
        views of it, and the matrix keeps the accumulator alive.  Each ``data()``
        call makes a fresh accumulator, so no two matrices share one.  With hypre on
        a device each matrix gets private copies of the index arrays instead: MFEM
        aliases them there and keys the device mirror on the host pointer, so two live
        matrices must not share them.
        """
        acc = host_writable(acc)
        if self.exchange:                  # collective decision, see __init__
            rbuf = np.empty(self.nrecv, dtype=np.float64)
            self.comm.Alltoallv(
                [np.ascontiguousarray(acc[self.nnz_t:], dtype=np.float64),
                 self.scnt, self.sdsp, MPI.DOUBLE],
                [rbuf, self.rcnt, self.rdsp, MPI.DOUBLE])
            if self.nrecv:
                acc[:self.nnz_t] += np.bincount(self.slot_recv, weights=rbuf,
                                                minlength=self.nnz_t)
        if diagonal is not None:
            slots, value = diagonal
            acc[slots] = value
        from ..common.parvector import device_active

        if device_active() and DEVICE_PARMAT == "block":
            # Private index arrays per matrix: MFEM aliases them and keys the device
            # mirror on the host pointer, so two live matrices must not share them
            # (see ``parmat._as_sparse``), and the pattern's own arrays are shared
            # with its scatter target.
            return self._finish_block(acc, self.I_diag.copy(), self.J_diag.copy(),
                                      self.I_offd.copy(), self.J_offd.copy())
        if self.all_local or device_active():
            # ``all_local`` is a collective decision (see __init__).  On a device every
            # rank reaching here takes the copying constructor: the diag/offd one below
            # aliases the wrappers' arrays, MFEM registers them by host pointer, and a
            # Python proxy clears its ``__dict__`` (the kept wrappers and arrays)
            # before the C++ destructor runs, so hypre's destroy would find its device
            # mirrors already freed.  Copying into hypre-owned memory removes the
            # aliasing, the shared registry entry and the ordering problem at once.
            return self._finish_copy(acc)
        return self._finish_block(acc, self.I_diag, self.J_diag,
                                  self.I_offd, self.J_offd)

    def _finish_block(self, acc, I_d, J_d, I_o, J_o):
        """The diag/offd constructor: the two blocks as views of the accumulator.

        Copies nothing on the host, and on a device aliases the host arrays and lets
        MFEM's memory manager hold the mirrors (see ``parmat._as_sparse``).  The
        index arrays are the caller's to choose: on the host every matrix of a pattern
        may share the pattern's own, and on a device none may (see ``finish``).
        """
        # An empty block gets a private one-element array, not an empty view of
        # ``acc``: two empty views start at the same address, which MFEM's memory
        # manager (keyed on the host pointer) would register twice.  A rank owning
        # no element of a boundary pattern has such a block.
        d_diag = acc[:self.nnz_diag] if self.nnz_diag else np.zeros(1)
        d_offd = (acc[self.nnz_diag:self.nnz_t] if self.nnz_t > self.nnz_diag
                  else np.zeros(1))
        diag = mfem.SparseMatrix([I_d, J_d, d_diag, self.ntd, self.c1 - self.c0],
                                 False, False, True)
        offd = mfem.SparseMatrix([I_o, J_o, d_offd, self.ntd, self.n_offd],
                                 False, False, True)
        A = mfem.HypreParMatrix(self.comm, self.gnr, self.gnc, self.ia_rows.GetData(),
                                self.ia_cols.GetData(), diag, offd,
                                self.ia_cmap.GetData())
        A.CopyRowStarts()
        A.CopyColStarts()
        return own(A, acc, d_diag, d_offd, diag, offd, I_d, J_d, I_o, J_o,
                   self.ia_rows, self.ia_cols, self.ia_cmap)


    def _rowmajor(self):
        """``(I_t, J_t, perm_rm)``: the graph as one row-major CSR with global columns,
        and the slot of each of its entries (``None`` when the slots are already in
        that order).  Built in ``__init__`` where the copying constructor is the
        route in force, otherwise here on first use, from the two blocks."""
        if self._rm is None:
            ntd, nnz_t = self.ntd, self.nnz_t
            rows = np.concatenate([np.repeat(np.arange(ntd), np.diff(self.I_diag)),
                                   np.repeat(np.arange(ntd), np.diff(self.I_offd))])
            cols = np.concatenate([self.J_diag.astype(np.int64) + self.c0,
                                   self.cmap[self.J_offd]])
            order = np.lexsort((cols, rows))
            indptr = np.zeros(ntd + 1, dtype=np.int64)
            np.cumsum(np.bincount(rows, minlength=ntd), out=indptr[1:])
            self._rm = (indptr.astype(np.int32),
                        np.ascontiguousarray(cols[order]).astype(_HYPRE_INT, copy=False),
                        None if np.array_equal(order, np.arange(nnz_t)) else
                        order.astype(np.int32 if nnz_t < 2 ** 31 else np.int64))
        return self._rm

    @property
    def I_t(self):
        return self._rowmajor()[0]

    @property
    def J_t(self):
        return self._rowmajor()[1]

    @property
    def perm_rm(self):
        return self._rowmajor()[2]

    def _finish_copy(self, acc):
        """The copying constructor: one row-major CSR with global columns.

        MFEM splits it into hypre's two blocks and copies into hypre-owned memory,
        so nothing of ours is aliased or registered; the cost is the split and one
        gather of the values from the block-major slot order.  Taken on a device with
        ``HIPPYMFEM_PARMAT_DEVICE=copy``, and on the host only for a pattern with no
        off-diagonal column on any rank (``all_local``).
        """
        I_t, J_t, perm_rm = self._rowmajor()
        D = acc[:self.nnz_t]
        if perm_rm is not None:
            D = D[perm_rm]
        D = np.ascontiguousarray(D, dtype=np.float64)
        args = ([I_t, J_t, D, self.rows_t] if self.same
                else [I_t, J_t, D, self.rows_t, self.cols_t])
        A = mfem.HypreParMatrix(self.comm, self.ntd, self.gnr, self.gnc, args)
        A.CopyRowStarts()
        A.CopyColStarts()
        return A


def _tdof_maps(fes):
    """``(global_tdof, local_tdof)`` for every local dof of ``fes``; ``-1`` where the
    dof is owned elsewhere.

    On the host both come in bulk from the prolongation's local rows, an order of
    magnitude faster than one SWIG call per dof: a boolean ``P`` has one entry per
    row, whose column is the global true dof, and the local one is that column less
    the rank's first.  A non-boolean ``P`` (which the true-dof route never takes) or
    hypre's memory on a device (where the merged rows cannot be read from the host)
    falls back to one SWIG call per dof.
    """
    from ..common.parvector import device_active

    nld = int(fes.GetVSize())
    if not device_active() and nld:
        from ..common.linalg import hypre_to_scipy

        P = fes.Dof_TrueDof_Matrix()
        Pl = hypre_to_scipy(P)
        if Pl.shape[0] == nld and np.array_equal(np.diff(Pl.indptr), np.ones(nld, dtype=Pl.indptr.dtype)):
            gt = np.asarray(Pl.indices[Pl.indptr[:-1]], dtype=np.int64)
            starts = np.asarray(mfem.intArray((P.GetColStarts(), 2)).GetDataArray())
            c0, c1 = int(starts[0]), int(starts[1])
            lt = np.where((gt >= c0) & (gt < c1), gt - c0, -1)
            return gt, lt
    gt = np.fromiter((fes.GetGlobalTDofNumber(i) for i in range(nld)), dtype=np.int64,
                     count=nld)
    lt = np.fromiter((fes.GetLocalTDofNumber(i) for i in range(nld)), dtype=np.int64,
                     count=nld)
    return gt, lt


_TDOF = IdentityCache()


def get_tdof_pattern(pattern, test_space, trial_space):
    """The :class:`TrueDofPattern` for a pattern and its spaces, built once.

    Keyed on the identities of all three (an :class:`~.identitycache.IdentityCache`,
    so the entry goes when any of them does); cleared with the pattern cache.
    """
    objs = (pattern, test_space.fes, trial_space.fes)
    got = _TDOF.get(objs)
    if got is None:
        got = _TDOF.put(objs, TrueDofPattern(pattern, test_space, trial_space))
    return got


def clear_tdof_cache():
    _TDOF.clear()

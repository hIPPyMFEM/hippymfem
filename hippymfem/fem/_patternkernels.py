# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The compiled passes of :mod:`hippymfem.fem.patternbuild`.

Kept apart so that importing the policy (whether to use them) never imports numba, and
defined at module level so that numba's on-disk cache can hold them: a nested definition
would be compiled afresh in every process.
"""

import numba as nb
import numpy as np

#: Bits given to an entry's place within its row in the packed per-row key.
QBITS = 16
MASK = (1 << QBITS) - 1

@nb.njit(cache=True)
def runs(rows):
    n = rows.size
    nrun = 0
    for k in range(n):
        if k == 0 or rows[k] != rows[k - 1]:
            nrun += 1
    rstart = np.empty(nrun + 1, np.int64)
    j = 0
    for k in range(n):
        if k == 0 or rows[k] != rows[k - 1]:
            rstart[j] = k
            j += 1
    rstart[nrun] = n
    return rstart

@nb.njit(cache=True)
def group_runs(rows, rstart, nrow):
    nrun = rstart.size - 1
    rcount = np.zeros(nrow + 1, np.int64)
    ecount = np.zeros(nrow + 1, np.int64)
    for t in range(nrun):
        r = rows[rstart[t]]
        rcount[r + 1] += 1
        ecount[r + 1] += rstart[t + 1] - rstart[t]
    for r in range(nrow):
        rcount[r + 1] += rcount[r]
        ecount[r + 1] += ecount[r]
    fill = rcount[:-1].copy()
    runs_of_row = np.empty(nrun, np.int64)
    for t in range(nrun):
        r = rows[rstart[t]]
        runs_of_row[fill[r]] = t
        fill[r] += 1
    return rcount, ecount, runs_of_row

@nb.njit(parallel=True, cache=True)
def order_rows(cols, rstart, rcount, ecount, runs_of_row, nrow, diag_first, scratch,
               ucount, slot, maxlen, block):
    nblk = (nrow + block - 1) // block
    for bi in nb.prange(nblk):
        key = np.empty(maxlen, np.int64)
        ent = np.empty(maxlen, np.int64)
        for r in range(bi * block, min(nrow, (bi + 1) * block)):
            lo = ecount[r]
            m = ecount[r + 1] - lo
            if m == 0:
                continue
            q = 0
            for p in range(rcount[r], rcount[r + 1]):
                t = runs_of_row[p]
                for k in range(rstart[t], rstart[t + 1]):
                    c = cols[k]
                    code = 0 if (diag_first and c == r) else c + 1
                    key[q] = (np.int64(code) << QBITS) | q
                    ent[q] = k
                    q += 1
            kk = key[:m]
            kk.sort()
            pos = -1
            prev = -1
            for s in range(m):
                code = kk[s] >> QBITS
                if code != prev:
                    pos += 1
                    scratch[lo + pos] = r if code == 0 else code - 1
                    prev = code
                slot[ent[kk[s] & MASK]] = pos
            ucount[r] = pos + 1

@nb.njit(parallel=True, cache=True)
def place_rows(rstart, rcount, ecount, runs_of_row, indptr, nrow, scratch, indices, slot):
    for r in nb.prange(nrow):
        lo = ecount[r]
        base = indptr[r]
        c = indptr[r + 1] - base
        for q in range(c):
            indices[base + q] = scratch[lo + q]
        for p in range(rcount[r], rcount[r + 1]):
            t = runs_of_row[p]
            for k in range(rstart[t], rstart[t + 1]):
                slot[k] += base


# ---------------------------------------------------------------- from the dof tables
# The same graph for the entries of one group of elements, read from the two dof tables
# instead of from flat ``rows``/``cols`` arrays with one index per element-matrix entry
# (16 bytes an entry, 12 GB for a million quadratic hexahedra, which the passes above
# only read).  Entry ``k = (e * ndt + a) * ndr + b`` is row ``tdofs[e, a]`` and column
# ``cdofs[e, b]``; run ``t = e * ndt + a`` is row ``a`` of element ``e``.

@nb.njit(cache=True)
def group_table(tdofs, nrow):
    """``(rcount, runs_of_row)`` as :func:`group_runs` gives them, for the runs of a
    table: every run has the trial table's width, so the entry counts are ``rcount``
    times that."""
    ne, ndt = tdofs.shape
    rcount = np.zeros(nrow + 1, np.int64)
    for e in range(ne):
        for a in range(ndt):
            rcount[tdofs[e, a] + 1] += 1
    for r in range(nrow):
        rcount[r + 1] += rcount[r]
    fill = rcount[:-1].copy()
    runs_of_row = np.empty(ne * ndt, np.int64)
    for e in range(ne):
        for a in range(ndt):
            r = tdofs[e, a]
            runs_of_row[fill[r]] = e * ndt + a
            fill[r] += 1
    return rcount, runs_of_row

@nb.njit(parallel=True, cache=True)
def order_table(cdofs, ndt, rcount, runs_of_row, nrow, diag_first, ucount, slot, maxlen,
                block):
    """:func:`order_rows` for a table: each entry's place within its row goes to
    ``slot`` and the row's number of distinct columns to ``ucount``.  The columns
    themselves are not kept: :func:`place_table` reads them from the table again."""
    ndr = cdofs.shape[1]
    nblk = (nrow + block - 1) // block
    for bi in nb.prange(nblk):
        key = np.empty(maxlen, np.int64)
        ent = np.empty(maxlen, np.int64)
        for r in range(bi * block, min(nrow, (bi + 1) * block)):
            m = (rcount[r + 1] - rcount[r]) * ndr
            if m == 0:
                continue
            q = 0
            for p in range(rcount[r], rcount[r + 1]):
                t = runs_of_row[p]
                e = t // ndt
                k0 = t * ndr
                for b in range(ndr):
                    c = cdofs[e, b]
                    code = 0 if (diag_first and c == r) else c + 1
                    key[q] = (np.int64(code) << QBITS) | q
                    ent[q] = k0 + b
                    q += 1
            kk = key[:m]
            kk.sort()
            pos = -1
            prev = -1
            for s in range(m):
                code = kk[s] >> QBITS
                if code != prev:
                    pos += 1
                    prev = code
                slot[ent[kk[s] & MASK]] = pos
            ucount[r] = pos + 1

@nb.njit(parallel=True, cache=True)
def place_table(cdofs, ndt, rcount, runs_of_row, indptr, nrow, indices, slot):
    """:func:`place_rows` for a table: every entry writes its column at its place (the
    entries of one place all hold the same column) and takes its row's offset."""
    ndr = cdofs.shape[1]
    for r in nb.prange(nrow):
        base = indptr[r]
        for p in range(rcount[r], rcount[r + 1]):
            t = runs_of_row[p]
            e = t // ndt
            k0 = t * ndr
            for b in range(ndr):
                k = k0 + b
                indices[base + slot[k]] = cdofs[e, b]
                slot[k] += base


# ------------------------------------------------------------- the true-dof graph
# The graph of a rank's true-dof rows (tdofassemble.TrueDofPattern) from the ldof graph,
# row by row: a true-dof row is one ldof row of this rank, its columns mapped to global
# true dofs, and the entries other ranks sent for it.  Each row is ordered by one short
# sort, as above, so nothing of the size of the graph is formed beside what is kept: the
# slot of every ldof entry and the columns of the diagonal block.

@nb.njit(parallel=True, cache=True)
def tdof_count(ldof_of_row, indptr, indices, gcol, rptr, rcol, rorder, c0, c1, square,
               tslot, slot_recv, ndiag, noff, maxlen, block):
    """First pass.  For true-dof row ``r``: its distinct global columns in ascending
    order, counted inside ``[c0, c1)`` (``ndiag``) and outside (``noff``), and in each
    entry's slot its place within the row's part of its block: ``p >= 0`` in the
    diagonal block, where the row's own diagonal leads (``square``), ``-1 - p`` in the
    other."""
    ntd = ldof_of_row.size
    nblk = (ntd + block - 1) // block
    for bi in nb.prange(nblk):
        key = np.empty(maxlen, np.int64)
        for r in range(bi * block, min(ntd, (bi + 1) * block)):
            i = ldof_of_row[r]
            s0 = indptr[i]
            m_own = indptr[i + 1] - s0
            q0 = rptr[r]
            m = m_own + rptr[r + 1] - q0
            if m == 0:
                ndiag[r] = 0
                noff[r] = 0
                continue
            for q in range(m_own):
                key[q] = (np.int64(gcol[indices[s0 + q]]) << QBITS) | q
            for q in range(m_own, m):
                key[q] = (np.int64(rcol[q0 + q - m_own]) << QBITS) | q
            kk = key[:m]
            kk.sort()
            lead_col = c0 + r if square else -1
            haslead = 0
            prev = -1
            for s in range(m):
                g = kk[s] >> QBITS
                if g != prev:
                    prev = g
                    if g == lead_col:
                        haslead = 1
            d = 0
            o = 0
            cur = 0
            prev = -1
            for s in range(m):
                g = kk[s] >> QBITS
                if g != prev:
                    prev = g
                    if g >= c0 and g < c1:
                        if g == lead_col:
                            cur = 0
                        else:
                            cur = haslead + d
                            d += 1
                    else:
                        cur = -1 - o
                        o += 1
                q = kk[s] & MASK
                if q < m_own:
                    tslot[s0 + q] = cur
                else:
                    slot_recv[rorder[q0 + q - m_own]] = cur
            ndiag[r] = haslead + d
            noff[r] = o

@nb.njit(parallel=True, cache=True)
def tdof_place(ldof_of_row, indptr, indices, gcol, rptr, rcol, rorder, c0, nnz_diag,
               dstart, ostart, tslot, slot_recv, J_diag, offd_cols):
    """Second pass: the slots (block-major: the diagonal block row by row, then the
    other) and the columns, each entry writing the column of its place."""
    ntd = ldof_of_row.size
    for r in nb.prange(ntd):
        i = ldof_of_row[r]
        db = dstart[r]
        ob = ostart[r]
        for s in range(indptr[i], indptr[i + 1]):
            p = tslot[s]
            g = gcol[indices[s]]
            if p >= 0:
                J_diag[db + p] = g - c0
                tslot[s] = db + p
            else:
                offd_cols[ob - 1 - p] = g
                tslot[s] = nnz_diag + ob - 1 - p
        for q in range(rptr[r], rptr[r + 1]):
            j = rorder[q]
            p = slot_recv[j]
            g = rcol[q]
            if p >= 0:
                J_diag[db + p] = g - c0
                slot_recv[j] = db + p
            else:
                offd_cols[ob - 1 - p] = g
                slot_recv[j] = nnz_diag + ob - 1 - p


@nb.njit(parallel=True, cache=True)
def block_slots(urow, rowstart, in_diag, lead, nrow, newslot):
    """The block-major, diagonal-leading slot of each entry of a row-sorted graph.

    The layout :func:`hippymfem.fem.tdofassemble._slot_order` defines (every
    diagonal-block entry first, row by row, each row's ``lead`` entry ahead of the rest
    of its row, then every off-diagonal entry row by row, the input order within each
    group), in one pass over each row instead of whole-array mask passes.
    """
    ndiag = np.zeros(nrow, np.int64)
    noff = np.zeros(nrow, np.int64)
    for r in nb.prange(nrow):
        d = 0
        for i in range(rowstart[r], rowstart[r + 1]):
            if in_diag[i]:
                d += 1
        ndiag[r] = d
        noff[r] = rowstart[r + 1] - rowstart[r] - d
    dstart = np.zeros(nrow, np.int64)
    ostart = np.zeros(nrow, np.int64)
    acc = 0
    for r in range(nrow):
        dstart[r] = acc
        acc += ndiag[r]
    for r in range(nrow):
        ostart[r] = acc
        acc += noff[r]
    for r in nb.prange(nrow):
        haslead = 0
        for i in range(rowstart[r], rowstart[r + 1]):
            if lead[i]:
                haslead = 1
        dr = 0
        orank = 0
        for i in range(rowstart[r], rowstart[r + 1]):
            if lead[i]:
                newslot[i] = dstart[r]
            elif in_diag[i]:
                newslot[i] = dstart[r] + haslead + dr
                dr += 1
            else:
                newslot[i] = ostart[r] + orank
                orank += 1

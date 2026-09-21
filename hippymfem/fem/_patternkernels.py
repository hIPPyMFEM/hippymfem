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

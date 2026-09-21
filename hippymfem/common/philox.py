# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Philox4x64-10 as a vectorized pure function of (key, counter), bit-identical to numpy's.

:class:`~hippymfem.common.random.Random` ties every random word to a global index so
that a sample does not depend on the partition.  For element-indexed noise the
indices are sparse hash keys, and one ``np.random.Philox`` object per element,
advanced to its offset, would dominate the cost of posterior sampling.  Since the
generator is a pure function of its key and counter, the words for every element can
be computed at once.  This reproduces numpy's ``Philox(key=seed)`` word for word: the
same constants, rounds and key schedule (Random123's ``philox4x64_R(10)``), a 128-bit
key as two little-endian 64-bit words, a 256-bit counter advanced by whole blocks of
four words and incremented before a block is generated, and doubles taken as
``(word >> 11) * 2**-53``.
"""

import numpy as np

_M0 = np.uint64(0xD2E7470EE14C6C93)
_M1 = np.uint64(0xCA5A826395121157)
_W0 = np.uint64(0x9E3779B97F4A7C15)
_W1 = np.uint64(0xBB67AE8584CAA73B)
_MASK32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)
_TWO53 = 9007199254740992.0
_MASK64 = (1 << 64) - 1


def _mulhilo(a, b, xp=np):
    """``(hi, lo)`` of the 128-bit product of two uint64 arrays."""
    m32, s32 = xp.uint64(0xFFFFFFFF), xp.uint64(32)
    a_lo, a_hi = a & m32, a >> s32
    b_lo, b_hi = b & m32, b >> s32
    lo_lo = a_lo * b_lo
    hi_lo = a_hi * b_lo
    lo_hi = a_lo * b_hi
    hi_hi = a_hi * b_hi
    cross = (lo_lo >> s32) + (hi_lo & m32) + lo_hi
    hi = hi_hi + (hi_lo >> s32) + (cross >> s32)
    lo = (cross << s32) | (lo_lo & m32)
    return hi, lo


def philox4x64(c0, c1, c2, c3, k0, k1, rounds=10, xp=np):
    """One block of four words per counter; all arguments are uint64 arrays (numpy or jax)."""
    k0 = xp.asarray(k0, dtype=xp.uint64).reshape(-1)     # 1-d: wrapping adds, no scalar warning
    k1 = xp.asarray(k1, dtype=xp.uint64).reshape(-1)
    W0, W1, M0, M1 = (xp.asarray(np.array([v], dtype=np.uint64)) for v in
                      (0x9E3779B97F4A7C15, 0xBB67AE8584CAA73B, 0xD2E7470EE14C6C93, 0xCA5A826395121157))
    for r in range(rounds):
        if r:
            k0 = k0 + W0
            k1 = k1 + W1
        hi0, lo0 = _mulhilo(M0, c0, xp)
        hi1, lo1 = _mulhilo(M1, c2, xp)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


_JAX_PHILOX = None


def _philox_jax(c0, c1, c2, c3, k0, k1):
    """The block function compiled once by JAX: ten rounds as one kernel, not two hundred."""
    global _JAX_PHILOX
    if _JAX_PHILOX is None:
        import jax
        import jax.numpy as jnp

        _JAX_PHILOX = jax.jit(lambda a, b, c, d, e, f: philox4x64(a, b, c, d, e, f, xp=jnp))
    return _JAX_PHILOX(c0, c1, c2, c3, k0, k1)


def _split_counter(blocks):
    """A 1-D array of Python-int block indices as four uint64 counter words."""
    out = []
    for shift in (0, 64, 128, 192):
        out.append(np.array([(int(b) >> shift) & _MASK64 for b in blocks], dtype=np.uint64))
    return out


def _words(c0, c1, c2, c3, k0, k1, xp):
    """The four Philox output words, on the device ``xp`` computes on."""
    if xp is np:
        return philox4x64(c0, c1, c2, c3, k0, k1)
    return _philox_jax(c0, c1, c2, c3, k0, k1)


def _uniforms(words, ne, nb, rem, n, xp):
    """Row ``i``'s ``n`` uniforms from its ``nb`` blocks of four words, skipping the
    first ``rem[i]``: the shared tail of every branch of :func:`uniforms_at`."""
    words = xp.stack(list(words), axis=1).reshape(ne, nb * 4)
    cols = xp.asarray(rem[:, None] + np.arange(n)[None, :])
    picked = xp.take_along_axis(words, cols, axis=1)
    u = (picked >> xp.uint64(11)).astype(xp.float64) * (1.0 / _TWO53)
    return xp.clip(u, 1e-300, 1.0 - 1e-16)


def uniforms_at(seed, offsets, n, xp=np):
    """``(len(offsets), n)`` uniforms in (0, 1): word ``offsets[i] + j`` of stream ``seed``.

    ``offsets`` may be Python ints of any size (the counter is 256 bits, like
    numpy's).  Row ``i`` is what the scalar path draws:
    ``np.random.Generator(Philox(key=seed))`` advanced by ``offsets[i] // 4`` blocks,
    its first ``offsets[i] % 4`` doubles discarded, then ``n`` doubles, clipped the
    same way (``ndtri(0)`` is ``-inf``).
    """
    n = int(n)
    seed = int(seed)
    if isinstance(offsets, tuple):
        # (lo, hi): the offsets as 128-bit numbers in two uint64 arrays, for keys past
        # 2^63 (element keys pack three 21-bit coordinates, times the stride).  The
        # counter is built with an explicit carry so everything stays array
        # arithmetic; Python ints would cost more than the words themselves.
        lo, hi = (np.asarray(offsets[0], dtype=np.uint64), np.asarray(offsets[1], dtype=np.uint64))
        ne = int(lo.size)
        if ne == 0 or n == 0:
            return np.zeros((ne, n))
        k0 = xp.asarray(np.array([seed & _MASK64], dtype=np.uint64))
        k1 = xp.asarray(np.array([(seed >> 64) & _MASK64], dtype=np.uint64))
        two = np.uint64(2)
        first_lo = (lo >> two) | (hi << np.uint64(62))       # offset // 4, low word
        first_hi = hi >> two                                   # offset // 4, high word
        rem = (lo & np.uint64(3)).astype(np.int64)
        nb = int(np.max((rem + n + 3) // 4))
        j = np.arange(1, nb + 1, dtype=np.uint64)[None, :]     # counter = block + 1
        c0 = (first_lo[:, None] + j)
        carry = (c0 < first_lo[:, None]).astype(np.uint64)
        c1 = first_hi[:, None] + carry
        c0, c1 = xp.asarray(c0.reshape(-1)), xp.asarray(c1.reshape(-1))
        c2 = c3 = xp.zeros_like(c0)
        return _uniforms(_words(c0, c1, c2, c3, k0, k1, xp), ne, nb, rem, n, xp)
    if isinstance(offsets, np.ndarray) and offsets.dtype.kind == "i":
        big = int(offsets.max()) if offsets.size else 0
    else:
        offsets = [int(o) for o in offsets]
        big = max(offsets) if offsets else 0
    ne = len(offsets)
    if ne == 0 or n == 0:
        return np.zeros((ne, n))
    k0 = np.uint64(seed & _MASK64)
    k1 = np.uint64((seed >> 64) & _MASK64)
    # the blocks each row needs: from its first word's block to its last word's.
    # numpy increments the counter *before* it generates a block, so the block a
    # generator advanced by b steps emits is the one at counter b + 1.
    if big + n + 4 < (1 << 63):
        # every counter fits one word: pure array arithmetic, no Python-int
        # bookkeeping (which would be most of the draw's cost).  ``xp`` is numpy on
        # the host and ``jax.numpy`` when the element kernels run on a GPU, where the
        # same uint64 expressions run on the card and the words never leave it.
        off = np.asarray(offsets, dtype=np.int64)
        first = off // 4
        rem = off % 4
        nb = int(np.max((rem + n + 3) // 4))
        blocks = (first[:, None] + np.arange(nb, dtype=np.int64)[None, :] + 1).reshape(-1)
        c0 = xp.asarray(blocks.astype(np.uint64))
        c1 = c2 = c3 = xp.zeros_like(c0)
        k0 = xp.asarray(np.array([seed & _MASK64], dtype=np.uint64))
        k1 = xp.asarray(np.array([(seed >> 64) & _MASK64], dtype=np.uint64))
        return _uniforms(_words(c0, c1, c2, c3, k0, k1, xp), ne, nb, rem, n, xp)
    else:
        offsets = [int(o) for o in offsets]
        first = [o // 4 for o in offsets]
        rem = np.array([o % 4 for o in offsets], dtype=np.int64)
        nb = int(np.max((rem + n + 3) // 4))
        blocks = [f + j + 1 for f in first for j in range(nb)]  # row-major, Python ints
        c0, c1, c2, c3 = _split_counter(blocks)
    return _uniforms(philox4x64(c0, c1, c2, c3, k0, k1), ne, nb, rem, n, np)

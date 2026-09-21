# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Partition-independent parallel random numbers.

Reproducibility across MPI decompositions is a test instrument, not a convenience:
the test suite runs on several rank counts and checks results that must not depend
on the partition, which is meaningless if the random input itself does.

So every entry of a random vector is tied to its **global** index.  We use a
counter-based generator (Philox) advanced to the global offset and draw exactly
one 64-bit word per entry, mapping uniforms to normals with the inverse normal
CDF rather than a ziggurat.  Both properties are needed: a fixed word budget per
entry, and a generator that can jump ahead in O(1).
"""

import numpy as np
from scipy.special import ndtri
from .philox import uniforms_at


def _on_gpu():
    """Whether the element kernels run on a GPU, which is where a draw then runs too."""
    try:
        from ..fem.kernel import device

        return any(k in str(device()).lower() for k in ("cuda", "gpu"))
    except Exception:                                            # noqa: BLE001
        return False


class Random:
    """Counter-based parallel RNG.

    Parameters
    ----------
    seed : int
        Stream key.  Two :class:`Random` objects with the same seed produce the
        same global sequence regardless of how vectors are partitioned.
    """

    def __init__(self, seed=1):
        self.seed = int(seed)
        #: counter of 64-bit words consumed, so successive draws differ
        self._stream = 0

    # ------------------------------------------------------------------ core
    #: Philox4x64 emits 4 words per counter step and ``BitGenerator.advance``
    #: counts those steps, not words, so a word offset is split into a block
    #: index and a remainder that is drawn and discarded.
    _WORDS_PER_BLOCK = 4

    def _words(self, offset, n):
        """``n`` uniform doubles in (0,1) starting at global word ``offset``.

        One double consumes exactly one 64-bit word, so the value at global
        index ``i`` is the same no matter how the vector is partitioned.
        """
        n = int(n)
        if n == 0:
            return np.zeros(0)
        # offsets can be large (geometric element keys), so keep Python ints
        total = int(self._stream) + int(offset)
        block, rem = divmod(total, self._WORDS_PER_BLOCK)
        bg = np.random.Philox(key=self.seed)
        if block:
            bg.advance(block)
        u = np.random.Generator(bg).random(rem + n)[rem:]
        # ndtri(0) is -inf; Generator.random() can return exactly 0.
        return np.clip(u, 1e-300, 1.0 - 1e-16)

    def _advance_stream(self, global_size):
        # consume the words just drawn so consecutive draws cannot overlap
        self._stream = int(self._stream) + int(global_size)

    def set_seed(self, seed):
        self.seed = int(seed)
        self._stream = 0

    # ------------------------------------------------------------- interface
    def normal(self, sigma, out, add=False):
        """Fill ``out`` with N(0, sigma^2) samples (or add them if ``add``)."""
        lo, _ = out.owner_range
        u = self._words(lo, out.local_size)
        vals = sigma * ndtri(u)
        if add:
            out.array[:] += vals
        else:
            out.array[:] = vals
        self._advance_stream(out.global_size)
        return out

    def normal_perturb(self, sigma, out):
        """``out += N(0, sigma^2)``; the hIPPYlib name for additive noise."""
        return self.normal(sigma, out, add=True)

    def uniform(self, a, b, out):
        """Fill ``out`` with Uniform(a, b) samples."""
        lo, _ = out.owner_range
        u = self._words(lo, out.local_size)
        out.array[:] = a + (b - a) * u
        self._advance_stream(out.global_size)
        return out

    def rademacher(self, out):
        """Fill ``out`` with independent +-1 values."""
        lo, _ = out.owner_range
        u = self._words(lo, out.local_size)
        out.array[:] = np.where(u < 0.5, -1.0, 1.0)
        self._advance_stream(out.global_size)
        return out

    def normal_multivector(self, sigma, mv):
        """Fill every column of a :class:`MultiVector` with N(0, sigma^2)."""
        for i in range(mv.nvec()):
            self.normal(sigma, mv[i])
        return mv

    def normal_blocks(self, sigma, global_offsets, n):
        """``(len(offsets), n)`` normals, row ``i`` at global words ``[offsets[i], +n)``.

        The same numbers :meth:`normal_block` gives row by row, computed for every
        row at once (see :mod:`hippymfem.common.philox`); like it, does not advance
        the stream.  This is what element-indexed noise draws: one call per sample
        instead of one generator per element.
        """

        stream = int(self._stream)
        if isinstance(global_offsets, tuple):
            # (lo, hi) uint64 word pairs (see philox.uniforms_at): add the stream with
            # a carry and stay in array arithmetic whatever the key width
            lo = np.asarray(global_offsets[0], dtype=np.uint64)
            hi = np.asarray(global_offsets[1], dtype=np.uint64)
            s_lo, s_hi = np.uint64(stream & ((1 << 64) - 1)), np.uint64(stream >> 64)
            lo2 = lo + s_lo
            hi2 = hi + s_hi + (lo2 < lo).astype(np.uint64)
            offs = (lo2, hi2)
            big = None
        elif isinstance(global_offsets, np.ndarray) and global_offsets.dtype.kind == "i" \
                and global_offsets.size and int(global_offsets.max()) + stream + int(n) + 8 < (1 << 63):
            offs = global_offsets.astype(np.int64) + np.int64(stream)     # no Python ints
            big = int(offs.max())
        else:
            offs = [stream + int(o) for o in global_offsets]
            big = max(offs) if offs else 0
        if _on_gpu() and (big is None or big + int(n) + 4 < (1 << 63)):
            # the words and the inverse normal CDF on the card: on the host, scipy's
            # ndtri over a sample's normals is most of the sample's cost
            import jax
            import jax.numpy as jnp
            from jax.scipy.special import ndtri as jndtri

            u = uniforms_at(self.seed, offs, n, xp=jnp)
            return np.asarray(jax.device_get(sigma * jndtri(u)))
        return sigma * ndtri(uniforms_at(self.seed, offs, n))

    def normal_block(self, sigma, global_offset, n):
        """Normals for the global index range ``[offset, offset+n)``.

        Does **not** advance the stream: the caller decides how much of the
        global stream this draw consumed, which lets element-indexed noise be
        drawn without a collective.
        """
        return sigma * ndtri(self._words(global_offset, n))

    def advance(self, nwords):
        """Mark ``nwords`` of the global stream as consumed (no communication)."""
        self._stream = int(self._stream) + int(nwords)
        return self

    def normal_array(self, sigma, comm, local_size, global_offset):
        """Normals for one locally-owned block, advancing the stream globally.

        Kept for callers that have no cheaper way to know the global stream
        length; prefer :meth:`normal_block` plus one :meth:`advance`.
        """
        vals = self.normal_block(sigma, global_offset, local_size)
        self._advance_stream(comm.allreduce(int(local_size)))
        return vals

    def scalar_normal(self, sigma=1.0, comm=None):
        """One N(0, sigma^2) draw, identical on every rank."""
        u = self._words(0, 1)
        self._advance_stream(1)
        return float(sigma * ndtri(u)[0])

    def scalar_uniform(self, a=0.0, b=1.0, comm=None):
        """One Uniform(a, b) draw, identical on every rank."""
        u = self._words(0, 1)
        self._advance_stream(1)
        return float(a + (b - a) * u[0])


#: module-level instance, mirroring ``hippylibx.parRandom``
parRandom = Random(seed=1)

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The two big sorts of a pattern build, on the kernel device when there is one.

``ScatterPattern`` argsorts one row-major key per element-matrix entry and
``TrueDofPattern`` takes the unique keys of the true-dof graph with their inverse.
On a large mesh those are hundreds of millions of keys, tens of seconds of one-time
setup on the host against a fraction of a second per hundred million keys on a
GPU, so when the kernels are on a GPU the sorts go there too.

The order among equal keys reaches neither caller: equal keys are the same
``(row, col)`` and get the same slot whichever comes first, and the inverse of a
unique is the same for every copy of a key.  So any valid argsort will do, and the
device one is used chunked when the keys exceed :func:`chunk_limit`: the keys are
bucketed by value range on the host (a linear radix sort of 16-bit bucket ids),
each bucket is sorted on the device in a buffer of one fixed size, so that a
handful of compiled shapes serve every rank and every space pair, and the buckets
are concatenated.  A bucket over the limit (a skewed key range) is bucketed again.
Peak device memory is that of one chunk's sort.

``HIPPYMFEM_PATTERN_SORT`` is ``auto`` (the device when the kernels are on a GPU and
the array has at least ``HIPPYMFEM_PATTERN_SORT_MIN`` entries), ``host`` or
``device``.
"""

import os
import time

import numpy as np

from . import kernel as kernel_mod

MODE = os.environ.get("HIPPYMFEM_PATTERN_SORT", "auto").strip().lower()
MIN_DEVICE = int(os.environ.get("HIPPYMFEM_PATTERN_SORT_MIN", str(2 ** 22)) or 0)
#: Keys per device sort; ``0``, the default, sizes it from the free device memory.
#: Set ``HIPPYMFEM_PATTERN_SORT_CHUNK``.
CHUNK = int(os.environ.get("HIPPYMFEM_PATTERN_SORT_CHUNK", "0") or 0)
MAX_CHUNK = 2 ** 28
PAD = 2 ** 22
#: Device bytes per key at the peak of a sort (measured about 35, with margin).
BYTES_PER_KEY = 40
#: The same for CuPy's radix sort, which sorts the array as it is rather than a
#: padded buffer and keeps one temporary beside the keys and the indices: measured
#: about 20, taken with the same margin.  A bigger chunk means fewer buckets, and
#: the bucketing is host work, so this is worth distinguishing.
BYTES_PER_KEY_CUPY = 24
_SORT = {}


def chunk_limit():
    """Keys one device sort may hold: a power of two, from what the budget has free.

    The bucketed path costs about what the host sort costs (its gathers and the
    bucketing are host work), while one device call is a few times faster, so the
    limit is as large as the budget allows.  The budget is JAX's share of the card,
    and the pattern is built before the kernels take theirs.
    """
    if CHUNK:
        return int(CHUNK)
    try:
        stats = kernel_mod.device().memory_stats() or {}
        free = int(stats["bytes_limit"]) - int(stats.get("bytes_in_use", 0))
    except Exception:
        return 2 ** 26
    per_key = BYTES_PER_KEY_CUPY if _cupy_usable() else BYTES_PER_KEY
    m = 2 ** 24
    while m * 2 * per_key <= 0.8 * free and m * 2 <= MAX_CHUNK:
        m *= 2
    return m


def use_device(n):
    """Whether a sort of ``n`` keys goes to the device."""
    if MODE == "host" or n == 0:
        return False
    if MODE == "device":
        return True
    return n >= MIN_DEVICE and kernel_mod.on_gpu()


def _padded_size(n, fixed=0):
    """The buffer size a chunk of ``n`` keys is sorted at.

    Every distinct size is a compile of a second or two, so a bucketed sort uses one
    size for every bucket (``fixed``, the chunk limit) and a single array the
    smallest power-of-two multiple of ``PAD`` that holds it: a run compiles a
    handful of shapes at most, whatever the ranks' and spaces' entry counts are.
    """
    if fixed:
        return int(fixed)
    m = PAD
    while m < n:
        m *= 2
    return m


#: Which sorting kernel the device path uses: ``auto`` prefers CuPy's radix sort
#: (CUB) and falls back to XLA's, ``cupy`` and ``xla`` force one.  XLA's sort is a
#: bitonic network, and on 100 M int64 keys on an L40S it measures 24 ns a key
#: against CuPy's 9.2 (transfers included), which matters because a 256\ :sup:`3`
#: linearization point sorts 1.5 billion of them.  Set ``HIPPYMFEM_PATTERN_SORT_KERNEL``.
SORT_KERNEL = os.environ.get("HIPPYMFEM_PATTERN_SORT_KERNEL", "auto").strip().lower()
_CUPY_OK = None


def _cupy_usable():
    """Whether the CuPy path is available, asked once and remembered.

    :func:`chunk_limit` needs the answer before the first sort, since the chunk it
    picks depends on how much the sorting kernel holds per key.
    """
    global _CUPY_OK
    if _CUPY_OK is None:
        if SORT_KERNEL == "xla":
            _CUPY_OK = False
        else:
            try:
                import cupy  # noqa: F401

                _CUPY_OK = True
            except Exception:                                # noqa: BLE001
                _CUPY_OK = False
    return _CUPY_OK


def _cupy_argsort(keys):
    """CUB's radix sort through CuPy, or ``None`` if it is not usable here.

    Returns ``None`` rather than raising for every reason it might not work (CuPy
    absent, a driver it will not talk to, no room on the card beside JAX and hypre),
    so the caller simply takes the XLA path instead.  CuPy allocates outside JAX's
    arena, which is why the chunk limit still applies and why the pool is released
    immediately: the sort is a one-time cost and must not hold memory a solve needs.
    """
    if not _cupy_usable():
        return None
    import cupy as cp
    pool = None
    try:
        dev = getattr(kernel_mod.device(), "id", 0)
        with cp.cuda.Device(int(dev)):
            pool = cp.get_default_memory_pool()
            order = cp.asnumpy(cp.argsort(cp.asarray(keys)))
        return order
    except Exception:                                        # noqa: BLE001
        if SORT_KERNEL == "cupy":
            raise
        return None
    finally:
        if pool is not None:
            try:
                pool.free_all_blocks()
            except Exception:                                # noqa: BLE001
                pass


def _device_argsort(keys, fixed=0):
    """Argsort of one chunk on the device, padded to :func:`_padded_size`.

    CuPy's radix sort takes the array as it is; XLA's is compiled per shape, so the
    chunk is padded to one of a handful of sizes to keep the compilation count down.
    """
    order = _cupy_argsort(keys)
    if order is not None:
        return order

    import jax
    import jax.numpy as jnp

    n = int(keys.size)
    m = _padded_size(n, fixed)
    fn = _SORT.get(m)
    if fn is None:
        fn = _SORT[m] = jax.jit(lambda k: jnp.argsort(k))
    buf = np.empty(m, dtype=np.int64)
    buf[:n] = keys
    buf[n:] = np.iinfo(np.int64).max          # the padding sorts last
    order = np.asarray(fn(jax.device_put(buf, kernel_mod.device())))
    del buf
    return order if m == n else order[order < n]


def argsort_keys(keys):
    """A valid argsort of int64 ``keys`` (ties in any order), device or host."""
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    n = int(keys.size)
    if not use_device(n):
        return np.argsort(keys, kind="stable")
    chunk = chunk_limit()
    if n <= chunk:
        return _device_argsort(keys)
    kmin, kmax = int(keys.min()), int(keys.max())
    if kmax == kmin:
        return np.arange(n, dtype=np.int64)
    # Buckets close to a chunk, so that each one fills the sorting buffer, with a width
    # rounded up to a power of two: the bucket of a key is then a subtraction and a
    # shift rather than an integer division, which at a billion keys is worth having.
    # Rounding up can only make the buckets fewer and larger, so a bucket over the
    # chunk limit is still handled by the recursion below.
    # Aim at four tenths of a chunk: the power-of-two width below can double a bucket,
    # and keys are not uniform over their range (rows differ in length), so a target
    # near the limit sends the largest buckets through the recursion, which buckets
    # them all over again.  More, smaller buckets cost a device call each and nothing
    # else, and they hold no more device memory than one chunk.
    nb = int(min(-(-n // int(0.4 * chunk)) + 1, 32767))
    width = -(-(kmax - kmin + 1) // nb)
    shift = max(int(width - 1).bit_length(), 0)
    nb = ((kmax - kmin) >> shift) + 1
    # in slices, so that the intermediate of the subtraction never exists whole: at a
    # billion keys that temporary alone is 8 GB of host memory on every rank
    bucket = np.empty(n, dtype=np.int16 if nb <= 32767 else np.int32)
    step = 1 << 26
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        np.right_shift(keys[lo:hi] - kmin, shift, out=bucket[lo:hi], casting="unsafe")
    pb = np.argsort(bucket, kind="stable")     # 16-bit keys: numpy's radix sort, linear
    counts = np.bincount(bucket, minlength=nb)
    del bucket
    order = np.empty(n, dtype=np.int64)
    start = 0
    from .pattern import PATTERN_TIMING

    t_gather = t_sort = 0.0
    nchunk = 0
    for cnt in counts:
        cnt = int(cnt)
        if cnt == 0:
            continue
        idx = pb[start:start + cnt]
        t0 = time.perf_counter() if PATTERN_TIMING else 0.0
        sub = keys[idx]
        if PATTERN_TIMING:
            t_gather += time.perf_counter() - t0
            t0 = time.perf_counter()
        loc = _device_argsort(sub, fixed=chunk) if cnt <= chunk else argsort_keys(sub)
        if PATTERN_TIMING:
            t_sort += time.perf_counter() - t0
            t0 = time.perf_counter()
        order[start:start + cnt] = idx[loc]
        if PATTERN_TIMING:
            t_gather += time.perf_counter() - t0
        nchunk += 1
        start += cnt
    if PATTERN_TIMING:
        from .pattern import _say_phase

        _say_phase("  of which device sort", t_sort, n)
        _say_phase("  of which host gathers", t_gather, n)
        _say_phase("  chunks: %d" % nchunk, 0.0, n)
    return order


def unique_inverse(keys):
    """``np.unique(keys, return_inverse=True)``, through :func:`argsort_keys`."""
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    n = int(keys.size)
    if not use_device(n):
        return np.unique(keys, return_inverse=True)
    order = argsort_keys(keys)
    sk = keys[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(sk[1:], sk[:-1], out=first[1:])
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.cumsum(first) - 1
    return sk[first], inv

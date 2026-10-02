# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What this PyMFEM was built with, read rather than probed.

MFEM's own way of answering "do you have CUDA?" is to try to configure a device and
``MFEM_ABORT`` if not, which ends the MPI job instead of returning false.  The build
configuration is recorded in the installed ``config/_config.hpp``, so it can be read
directly and without side effects.
"""

import math
import os
import re
import warnings

import mfem.par as mfem

from .mpiutil import local_rank
from .._jaxconfig import hypre_on_device

__all__ = ["mfem_config", "mfem_has", "mfem_version",
           "configure_device"]

_CONFIG = None

#: Features worth reporting; absence means the build does not have them.
_FLAGS = (
    "MFEM_USE_MPI", "MFEM_USE_CUDA", "MFEM_USE_HIP", "MFEM_USE_OPENMP",
    "MFEM_USE_METIS", "MFEM_USE_METIS_5", "MFEM_USE_LAPACK",
    "MFEM_USE_SUITESPARSE", "MFEM_USE_MUMPS", "MFEM_USE_SUPERLU",
    "MFEM_USE_STRUMPACK", "MFEM_USE_PETSC", "MFEM_USE_SLEPC",
    "MFEM_USE_ZLIB", "MFEM_USE_GSLIB", "MFEM_USE_CEED", "MFEM_USE_ENZYME",
    "MFEM_USE_SINGLE", "MFEM_USE_DOUBLE", "MFEM_USE_MEMALLOC",
)


def _config_paths():
    root = os.path.dirname(os.path.abspath(mfem.__file__))
    root = os.path.dirname(root)                      # .../site-packages/mfem/..
    cands = []
    for sub in ("mfem/external/par/include/mfem/config",
                "mfem/external/ser/include/mfem/config"):
        for name in ("_config.hpp", "config.hpp"):
            cands.append(os.path.join(root, sub, name))
    return cands


def mfem_config(refresh=False):
    """A dict of the build flags that are defined, plus ``version``."""
    global _CONFIG
    if _CONFIG is not None and not refresh:
        return _CONFIG
    out = {}
    text = ""
    for p in _config_paths():
        if os.path.exists(p):
            try:
                text += open(p).read()
            except Exception:
                pass
    for f in _FLAGS:
        out[f] = bool(re.search(r"^\s*#define\s+%s\b" % f, text, re.M))
    m = re.search(r'^\s*#define\s+MFEM_VERSION_STRING\s+"([^"]+)"', text, re.M)
    out["version"] = m.group(1) if m else None
    try:
        import mfem as _m

        out["pymfem_version"] = getattr(_m, "__version__", None)
    except Exception:
        out["pymfem_version"] = None
    _CONFIG = out
    return out


def mfem_has(flag):
    """``mfem_has("MFEM_USE_CUDA")``, without touching an MFEM device."""
    if not flag.startswith("MFEM_"):
        flag = "MFEM_USE_" + flag.upper()
    return bool(mfem_config().get(flag, False))


def mfem_gpu_backend():
    """The GPU backend this PyMFEM's MFEM was built for: ``"cuda"``, ``"hip"`` or None.

    NVIDIA builds carry ``MFEM_USE_CUDA``, AMD builds ``MFEM_USE_HIP``
    (``tools/build_pymfem_hip.sh``); MFEM refuses both at once.
    """
    if mfem_has("MFEM_USE_CUDA"):
        return "cuda"
    if mfem_has("MFEM_USE_HIP"):
        return "hip"
    return None


def mfem_version():
    """MFEM's version string, e.g. ``"4.8.0"``."""
    return mfem_config().get("version")

def configure_device(kind="gpu", comm=None, quiet=False):
    """Put MFEM (and therefore hypre) on a device, **one per rank**.

    ``mfem.Device("cuda")`` with no device id puts every rank on device 0, which
    nothing reports and which shows up only as poor GPU scaling.  This picks
    ``local_rank % n_devices`` instead, the rule the element kernels use, so a rank's
    matrix and kernels share a card.

    Only useful with a PyMFEM built for a GPU: CUDA (``mfem_has("MFEM_USE_CUDA")``) or
    HIP (``MFEM_USE_HIP``).  ``kind`` may name the vendor (``"cuda"``, ``"hip"``) or
    just ``"gpu"``; each means the backend the build has, so a script runs unchanged
    on either vendor.  Both assembly routes work with hypre on a device: the true-dof
    scatter builds the parallel matrix from hypre's two blocks
    (:mod:`hippymfem.fem.tdofassemble`), and the triple-product fallback builds its
    local matrix with the constructor that copies into hypre's own memory
    (:func:`hippymfem.fem.parmat.local_par_matrix`), which is what it does on a device
    whatever ``HIPPYMFEM_PARMAT`` says.  An explicit ``HIPPYMFEM_PARMAT=direct`` is
    still refused here.

    **This is the configuration that matters.**  The element kernels are a small share
    of a Newton step, so moving only them to the GPU gains little; moving the solves
    as well is worth more than an order of magnitude (see the GPU guide).

    Returns the ``mfem.Device`` it built, which must be kept alive as long as MFEM is
    used; this module holds a reference to it as well.
    """
    import mfem.par as mfem

    if _DEVICE:
        # MFEM allows one Device per process, so a second call (say, after the
        # import-time configuration) returns the first.
        return _DEVICE[-1]
    if kind in ("cpu", None):
        _DEVICE.append(mfem.Device("cpu"))
        return _DEVICE[-1]
    backend = mfem_gpu_backend()
    if backend is None:
        raise RuntimeError(
            "this PyMFEM was built without CUDA or HIP (MFEM_USE_CUDA and MFEM_USE_HIP "
            "are false), so MFEM and hypre cannot use a device; "
            "tools/build_pymfem_cuda.sh (NVIDIA) and tools/build_pymfem_hip.sh (AMD) "
            "build one that can. The element kernels are a separate matter and run on "
            "a GPU through JAX with HIPPYMFEM_DEVICE=gpu on any build.")
    if kind in ("gpu", "cuda", "hip", "rocm"):
        kind = backend

    if not hypre_on_device():
        raise RuntimeError(
            "hypre is about to run on a GPU that JAX has already been told it may "
            "fill. JAX's allocator keeps its high-water mark and never returns it, "
            "so hypre would be left with whatever JAX happened not to touch.\n"
            "  Set HIPPYMFEM_HYPRE_DEVICE=1 *before* importing hippymfem, which "
            "halves JAX's share so the two fit together. The cap is read once, "
            "when JAX is imported, so setting it afterwards has no effect.")
    from ..fem import parmat as _parmat

    if _parmat.PARMAT_MODE == "direct":
        raise RuntimeError(
            "hypre cannot run on a device while HIPPYMFEM_PARMAT=direct: that route "
            "builds the matrix from numpy arrays, which a device-configured hypre "
            "reads as device pointers, and it segfaults rather than failing "
            "cleanly.\n"
            "  Unset HIPPYMFEM_PARMAT (or set it to \"auto\") to let MFEM's own "
            "ParallelAssemble build the parallel matrix from the same local CSR. "
            "It is bit-identical and it keeps the vectorized scatter, so nothing is "
            "given up by doing so.")
    # The element kernels also pick their device by node-local rank, and the two must
    # agree: with MFEM on device 0 and JAX on device 1, hypre gets a pointer from the
    # wrong context and thrust aborts with "invalid device ordinal".
    n = _device_count()
    # Asked once, unconditionally: without launcher variables naming the node-local
    # rank this is a collective (``local_rank`` splits the communicator), so it must
    # not sit behind ``not quiet`` below, which a caller may make rank-local
    # (``quiet=(rank != 0)``).  The launcher's variables need no communication, which
    # is what lets this run at import.
    local = _node_rank(comm)
    idx = local % max(n, 1)
    dev = mfem.Device(kind, idx)
    _DEVICE.append(dev)
    set_hypre_spmv(os.environ.get("HIPPYMFEM_HYPRE_SPMV", "auto"), comm)
    try:
        megabytes = float(os.environ.get("HIPPYMFEM_HYPRE_POOL", "0") or 0)
    except ValueError:
        megabytes = 0.0
    if megabytes > 0:
        set_hypre_pool(megabytes)
    if not quiet and local == 0:
        print("hippymfem: MFEM on %s device %d of %d" % (kind, idx, n), flush=True)
        from .._jaxconfig import PIN_SKIPPED

        if PIN_SKIPPED and n > 1:
            # rank 0 only (no collective here): nothing else reports the stray contexts
            print("hippymfem: GPU not pinned (%s); each rank holds a context on all %d "
                  "cards" % (PIN_SKIPPED, n), flush=True)
    return dev


_DEVICE = []


def _hypre_library():
    """A ``ctypes`` handle of the hypre library this process has loaded, or ``None``."""
    import ctypes

    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libHYPRE" in line:
                    return ctypes.CDLL(line.split(None, 5)[-1].strip())
    except OSError:
        pass
    return None


#: The kernel of hypre's matrix-vector products that :func:`set_hypre_spmv` chose:
#: ``"vendor"``, ``"hypre"``, or ``None`` when nothing was set.
HYPRE_SPMV = None


def set_hypre_spmv(kernel="auto", comm=None):
    """Choose the kernel of hypre's sparse matrix-vector products on a GPU.

    hypre stores a parallel matrix as two blocks per rank: the diagonal block, which
    couples the rank's own dofs, and the off-diagonal block, which couples them to the
    dofs of its neighbours.  The off-diagonal block has as many rows as the diagonal one
    and nonzeros in the few rows next to an interface only, and so have the blocks of
    the interpolation matrices of BoomerAMG.  By default hypre hands every block to the
    vendor's library, and with cuSPARSE the time of the product with such a block is
    unrelated to its nonzeros and grows with its rows: measured with hypre 2.32 and CUDA 12.9 on two
    ranks with 2.2 million dofs each, 0.6 to 0.9 ms for the off-diagonal block of the
    Jacobian (840 thousand nonzeros) and 5.4 ms for a block of an interpolation matrix
    with 701 nonzeros, against 2.2 ms for the diagonal block with 136 million.  hypre's
    own kernel takes 0.16 ms or less for the same blocks and 2.6 ms for the diagonal
    one.  On one rank there are no off-diagonal blocks and the vendor's kernel is the
    faster one (12 % per Krylov iteration); on two to sixteen ranks hypre's kernel made
    an iteration 11 to 36 % faster, and on two H100 it took 6.2 ms where cuSPARSE took
    16.5 (the GPU guide, "Several GPUs"; ``benchmarks/krylov_anatomy.py`` measures it).

    ``kernel`` is ``"vendor"``, ``"hypre"``, or ``"auto"``: hypre's kernel when ``comm``
    has more than one rank and the build is a CUDA one, the vendor's otherwise.  The
    environment variable ``HIPPYMFEM_HYPRE_SPMV`` sets it for :func:`configure_device`.
    The choice changes the time of a product and nothing else.  Returns the kernel now
    in use, or ``None`` when hypre is not on a device or the switch is not available.
    """
    global HYPRE_SPMV
    kernel = (kernel or "auto").lower()
    if kernel not in ("auto", "vendor", "hypre"):
        raise ValueError("HIPPYMFEM_HYPRE_SPMV must be auto, vendor or hypre, not %r" % kernel)
    if not _DEVICE or mfem_gpu_backend() is None:
        return None
    if kernel == "auto":
        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        kernel = "hypre" if (comm.Get_size() > 1 and mfem_gpu_backend() == "cuda") else "vendor"
    lib = _hypre_library()
    if lib is None or not hasattr(lib, "HYPRE_SetSpMVUseVendor"):
        return None
    import ctypes

    lib.HYPRE_SetSpMVUseVendor(ctypes.c_int(1 if kernel == "vendor" else 0))
    HYPRE_SPMV = kernel
    return kernel


class HyprePool:
    """A recycling pool for hypre's device memory; see :func:`set_hypre_pool`.

    Attributes (all in bytes or counts, for this rank): ``cached`` and ``peak_cached``,
    the freed memory the pool holds; ``in_use`` and ``peak_in_use``, the memory of the
    blocks it handed out; ``requests``, ``from_pool``, ``driver_allocs`` and
    ``driver_frees``.
    """

    #: size classes per factor of two; a recycled block is at most 2**(2/8) = 1.19
    #: times the size asked for
    CLASSES = 8.0

    def __init__(self, hypre, runtime, max_cached, max_block=None):
        import ctypes

        self._ct = ctypes
        self._dev_malloc = runtime.cudaMalloc
        self._dev_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._dev_malloc.restype = ctypes.c_int
        self._dev_free = runtime.cudaFree
        self._dev_free.argtypes = [ctypes.c_void_p]
        self._dev_free.restype = ctypes.c_int
        self.max_cached = int(max_cached)
        self.max_block = int(max_block if max_block is not None else max_cached)
        self._bins = {}                    # size class -> [(pointer, size), ...]
        self._size = {}                    # pointer -> size, for the blocks handed out
        self.cached = self.peak_cached = self.in_use = self.peak_in_use = 0
        self.requests = self.from_pool = self.driver_allocs = self.driver_frees = 0
        # hypre keeps the two function pointers: the callback objects must outlive it
        self._cb_malloc = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_size_t)(self._malloc)
        self._cb_free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self._free)
        self._hypre = hypre
        hypre.hypre_SetUserDeviceMalloc(self._cb_malloc)
        hypre.hypre_SetUserDeviceMfree(self._cb_free)
        self.installed = True

    def _class(self, size):
        return int(self.CLASSES * math.log2(size)) if size > 1 else 0

    def _driver_malloc(self, size):
        p = self._ct.c_void_p()
        if self._dev_malloc(self._ct.byref(p), size) != 0 or not p.value:
            # out of device memory: give back what the pool holds and try once more
            self.trim()
            p = self._ct.c_void_p()
            self._dev_malloc(self._ct.byref(p), size)
        self.driver_allocs += 1
        return p.value

    def _malloc(self, out, size):
        # called by hypre: nothing may escape, or hypre is left without a pointer
        try:
            self._take(out, size)
        except Exception:                                        # noqa: BLE001
            out[0] = self._driver_malloc(size) if size else None

    def _take(self, out, size):
        self.requests += 1
        if not size:
            out[0] = None
            return
        c = self._class(size)
        for k in (c, c + 1):
            blocks = self._bins.get(k)
            if blocks:
                for i in range(len(blocks) - 1, -1, -1):
                    p, s = blocks[i]
                    if s >= size:
                        blocks[i] = blocks[-1]
                        blocks.pop()
                        self.cached -= s
                        self._size[p] = s
                        self.in_use += s
                        if self.in_use > self.peak_in_use:
                            self.peak_in_use = self.in_use
                        self.from_pool += 1
                        out[0] = p
                        return
        p = self._driver_malloc(size)
        if p:
            self._size[p] = size
            self.in_use += size
            if self.in_use > self.peak_in_use:
                self.peak_in_use = self.in_use
        out[0] = p

    def _free(self, p):
        if not p:
            return
        s = self._size.pop(p, 0)
        if s:
            self.in_use -= s
            if s <= self.max_block and self.cached + s <= self.max_cached:
                self._bins.setdefault(self._class(s), []).append((p, s))
                self.cached += s
                if self.cached > self.peak_cached:
                    self.peak_cached = self.cached
                return
        # not handed out by this pool, too large, or the pool is full
        self._dev_free(p)
        self.driver_frees += 1

    def trim(self):
        """Return every block the pool holds to the driver; returns the bytes freed."""
        freed = self.cached
        for blocks in self._bins.values():
            for p, _s in blocks:
                self._dev_free(p)
                self.driver_frees += 1
        self._bins.clear()
        self.cached = 0
        return freed

    def uninstall(self):
        """Give hypre its own allocator back and release what the pool holds.  Blocks
        still in use are freed by hypre's own ``cudaFree`` afterwards, which takes any
        device pointer."""
        if self.installed:
            self._hypre.hypre_SetUserDeviceMalloc(None)
            self._hypre.hypre_SetUserDeviceMfree(None)
            self.installed = False
            self.trim()

    def stats(self):
        return {k: getattr(self, k) for k in (
            "requests", "from_pool", "driver_allocs", "driver_frees", "cached",
            "peak_cached", "in_use", "peak_in_use")}


#: The pool :func:`set_hypre_pool` installed, or ``None``.
HYPRE_POOL = None


def set_hypre_pool(megabytes=1024.0, max_block_megabytes=None):
    """Let hypre recycle its device memory through a pool that may hold ``megabytes``.

    A hypre built without Umpire and without its own device pool, which is what the
    PyMFEM build scripts produce, takes every device array from ``cudaMalloc`` and
    returns it with ``cudaFree``.  One BoomerAMG setup makes about two thousand such
    pairs of calls.  Each is a trip into the driver that costs tens of microseconds
    when one process uses the node and several hundred when sixteen do: measured on
    sixteen MIG instances of one node with 2.1 million dofs each, 2,206 allocations
    took 467 ms and 1,993 frees 617 ms of a setup of 1.47 s, against 21 ms and 91 ms
    of 0.18 s for one process alone (the GPU guide, "Several GPUs").

    hypre lets the application supply the two functions, and this supplies a pair that
    keeps freed blocks and hands them out again.  A new block has exactly the size
    asked for; a freed one is kept unless the pool already holds ``megabytes`` or the
    block is larger than ``max_block_megabytes`` (default: the same), in which case it
    goes back to the driver; a request is served by a kept block of at most 1.19 times
    its size.  So the pool costs at most ``megabytes`` of device memory per rank beyond
    what hypre is using, and :meth:`HyprePool.trim` returns that at any time.  If the
    driver refuses an allocation the pool is emptied and the allocation retried.

    With 1024 MB the setup above took 0.62 s instead of 1.47 s, and with 4096 MB
    0.44 s.  Nothing but the time of an allocation changes.  Off unless called, or
    unless ``HIPPYMFEM_HYPRE_POOL`` gives the megabytes for :func:`configure_device`;
    NVIDIA builds only.  Returns the :class:`HyprePool`, or ``None`` when hypre is not
    on a CUDA device or the hooks are not available.
    """
    global HYPRE_POOL
    if HYPRE_POOL is not None and HYPRE_POOL.installed:
        HYPRE_POOL.max_cached = int(megabytes * 2 ** 20)
        HYPRE_POOL.max_block = int((max_block_megabytes or megabytes) * 2 ** 20)
        return HYPRE_POOL
    if not _DEVICE or mfem_gpu_backend() != "cuda":
        return None
    import atexit
    import ctypes

    hypre = _hypre_library()
    runtime = None
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart.so" in line:
                    runtime = ctypes.CDLL(line.split(None, 5)[-1].strip())
                    break
    except OSError:
        pass
    if (hypre is None or runtime is None
            or not hasattr(hypre, "hypre_SetUserDeviceMalloc")
            or not hasattr(hypre, "hypre_SetUserDeviceMfree")):
        return None
    HYPRE_POOL = HyprePool(hypre, runtime, megabytes * 2 ** 20,
                           None if max_block_megabytes is None
                           else max_block_megabytes * 2 ** 20)
    # hypre frees its last arrays while the interpreter shuts down, when a callback
    # into Python is no longer safe: hand the allocator back before that
    atexit.register(HYPRE_POOL.uninstall)
    return HYPRE_POOL


def hypre_pool_megabytes():
    """Megabytes the installed pool may hold; 0 without one."""
    if HYPRE_POOL is None or not HYPRE_POOL.installed:
        return 0.0
    return HYPRE_POOL.max_cached / 2 ** 20


def set_hypre_pool_megabytes(megabytes):
    """:func:`set_hypre_pool` for a positive number, and for zero the removal of the
    pool, after which hypre allocates through the driver again."""
    megabytes = float(megabytes)
    if megabytes > 0:
        set_hypre_pool(megabytes)
    elif HYPRE_POOL is not None:
        HYPRE_POOL.uninstall()


def _node_rank(comm=None):
    """This rank's index within its node: from the launcher's variables when they
    name it, by the collective :func:`~.mpiutil.local_rank` otherwise."""
    from .._jaxconfig import _node

    rank, _, known = _node()
    return rank if known else local_rank(comm)


def auto_configure_device():
    """Configure MFEM's device from the environment, once, at import.

    ``HIPPYMFEM_HYPRE_DEVICE=1`` with a GPU build of PyMFEM (CUDA or HIP) puts MFEM
    and hypre on this rank's card, exactly as :func:`configure_device` would; on a host
    build it warns and does nothing, without the variable it does nothing, and
    ``HIPPYMFEM_AUTO_DEVICE=0`` turns it off.  Every script thus takes its configuration from the environment set
    before the import, without calling anything; an explicit ``configure_device``
    afterwards is a no-op.  Returns the device, or ``None`` when nothing was
    configured.
    """

    if os.environ.get("HIPPYMFEM_AUTO_DEVICE", "1").lower() in ("0", "no", "false", "off"):
        return None

    if not hypre_on_device() or _DEVICE:
        return None
    if mfem_gpu_backend() is None:
        # Nothing to configure, but the variable was set for a reason: say so, or a
        # whole run goes by with hypre on the host (a forward solve ten times slower)
        # and JAX still held to the share it leaves hypre on the card.
        warnings.warn(
            "HIPPYMFEM_HYPRE_DEVICE is set, but this PyMFEM (%s) was built without "
            "CUDA or HIP, so MFEM and hypre stay on the host; JAX's share of each card "
            "is still reduced as if hypre shared it. Put a GPU build of PyMFEM first "
            "on PYTHONPATH (tools/build_pymfem_cuda.sh or tools/build_pymfem_hip.sh), "
            "or unset the variable." % os.path.dirname(mfem.__file__),
            RuntimeWarning, stacklevel=2)
        return None
    return configure_device("gpu", quiet=False)


def _device_count():
    """GPUs this process may use.

    The visible-device variable first (``CUDA_VISIBLE_DEVICES`` or AMD's), and the
    vendor tool only when none is set: ``nvidia-smi -L`` lists every card on the node
    whatever the variable says, so trusting it would index past the end of the list
    once a scheduler or :func:`hippymfem._jaxconfig._pin_visible_device` has narrowed
    it.
    """
    from .._jaxconfig import _count_cards, _visible_var

    vis = _visible_var()
    if vis is not None:
        n = len([t for t in vis.split(",") if t.strip() not in ("", "-1")])
        return max(n, 1)
    return max(_count_cards(), 1)



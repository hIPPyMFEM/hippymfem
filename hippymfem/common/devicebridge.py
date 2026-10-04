# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Copies between JAX arrays and MFEM or hypre memory that stay on the GPU.

The element kernels are JAX programs and the matrices and vectors belong to MFEM and
hypre.  When both live on the same GPU, what one produces for the other used to travel
through CPU memory: an assembled matrix was copied to the host as a numpy array, handed
to MFEM there, and uploaded again when hypre first used it.  For the Jacobian of 32,768
quadratic hexahedra on an H100 that passage was 61 of the 71 ms of an assembly
(``benchmarks/DESIGN_NOTES.md``, section 11).

Nothing in JAX writes into memory it does not own, and nothing in PyMFEM builds a matrix
from a device array, so the two are joined one level down: a JAX array gives the address
of its buffer (``unsafe_buffer_pointer``), MFEM gives the address of the device copy of
a vector or of a block of a matrix (``Write``), and the runtime's ``cudaMemcpy`` or
``hipMemcpy`` copies between them on the device.  The runtime library is the one the
process has already loaded, found in ``/proc/self/maps`` as the hypre pool finds its
allocator (:mod:`.mfemconfig`).

The bridge is used only when :func:`available` says so: MFEM configured on a GPU, hypre
on it, the element kernels on the same card, the copy function found, and a first copy
of eight numbers read back correctly.  ``HIPPYMFEM_DEVICE_BRIDGE=0`` turns it off, and
everything then takes the route through the host as before.
"""

import ctypes
import os
import warnings

import numpy as np

__all__ = ["available", "why_not", "jax_address", "copy_from_jax", "copy_from_host",
           "copy_to_jax", "address", "synchronize", "set_device_bridge"]

#: ``"auto"`` (use the bridge when :func:`available`), ``"0"`` (never), or ``"1"``
#: (as ``auto``, but say why when it cannot be used).  ``HIPPYMFEM_DEVICE_BRIDGE``.
DEVICE_BRIDGE = os.environ.get("HIPPYMFEM_DEVICE_BRIDGE", "auto").lower()

_H2D, _D2H, _D2D = 1, 2, 3          # cudaMemcpyKind and hipMemcpyKind agree on these

_STATE = {"checked": False, "ok": False, "why": "not checked", "memcpy": None,
          "sync": None, "copies": 0, "bytes": 0}


def set_device_bridge(mode):
    """``"auto"``, ``"0"`` or ``"1"``; returns the previous setting."""
    global DEVICE_BRIDGE
    mode = str(mode).lower()
    if mode in ("true", "yes", "on"):
        mode = "1"
    if mode in ("false", "no", "off"):
        mode = "0"
    if mode not in ("auto", "0", "1"):
        raise ValueError("HIPPYMFEM_DEVICE_BRIDGE must be auto, 0 or 1, not %r" % (mode,))
    old, DEVICE_BRIDGE = DEVICE_BRIDGE, mode
    _STATE["checked"] = False
    return old


def _runtime(backend):
    """``(memcpy, synchronize, set_device)`` of the GPU runtime this process has loaded."""
    name, prefix = (("libcudart.so", "cuda") if backend == "cuda"
                    else ("libamdhip64.so", "hip"))
    path = None
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if name in line:
                    path = line.split(None, 5)[-1].strip()
                    break
    except OSError:
        return None
    if path is None:
        return None
    lib = ctypes.CDLL(path)
    try:
        memcpy = getattr(lib, prefix + "Memcpy")
        sync = getattr(lib, prefix + "DeviceSynchronize")
        setdev = getattr(lib, prefix + "SetDevice")
    except AttributeError:
        return None
    memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    memcpy.restype = ctypes.c_int
    sync.argtypes = []
    sync.restype = ctypes.c_int
    setdev.argtypes = [ctypes.c_int]
    setdev.restype = ctypes.c_int
    return memcpy, sync, setdev


def _check():
    """Decide once whether the bridge can be used, and remember why not."""
    _STATE["checked"] = True
    _STATE["ok"] = False
    if DEVICE_BRIDGE == "0":
        _STATE["why"] = "HIPPYMFEM_DEVICE_BRIDGE=0"
        return
    from . import mfemconfig
    from .parvector import device_active

    backend = mfemconfig.mfem_gpu_backend()
    if backend is None or not mfemconfig._DEVICE or not device_active():
        _STATE["why"] = "MFEM is not configured on a GPU"
        return
    try:
        from ..fem.kernel import device

        d = device()
    except Exception as exc:                                     # noqa: BLE001
        _STATE["why"] = "the kernels' device could not be read (%s)" % (exc,)
        return
    if getattr(d, "platform", "cpu") not in ("gpu", "cuda", "rocm"):
        _STATE["why"] = "the element kernels run on the host"
        return
    index = getattr(mfemconfig, "DEVICE_INDEX", None)
    if index is None or int(getattr(d, "id", -1)) != int(index):
        _STATE["why"] = ("the kernels are on GPU %s and MFEM is on GPU %s"
                         % (getattr(d, "id", "?"), index))
        return
    rt = _runtime(backend)
    if rt is None:
        _STATE["why"] = "the %s runtime's copy function was not found" % backend
        return
    memcpy, sync, setdev = rt
    if setdev(int(index)) != 0:
        _STATE["why"] = "the runtime refused device %d" % int(index)
        return
    _STATE["memcpy"], _STATE["sync"] = memcpy, sync
    # One copy of eight numbers each way, read back through MFEM: the two libraries
    # must be in one context of one card for an address of one to mean anything to
    # the other, and this is the cheapest way to be sure of it.
    try:
        import jax.numpy as jnp
        import mfem.par as mfem

        probe = jnp.asarray(np.arange(1.0, 9.0))
        v = mfem.Vector(8)
        v.UseDevice(True)
        v.Assign(0.0)
        _STATE["ok"] = True                    # for the two calls below
        copy_from_jax(address(v.Write()), probe)
        synchronize()
        v.HostRead()
        got = np.array(v.GetDataArray(), dtype=np.float64, copy=True)
        back = copy_to_jax(address(v.Read()), 8)
        _STATE["ok"] = bool(np.array_equal(got, np.arange(1.0, 9.0))
                            and np.array_equal(np.asarray(back), np.arange(1.0, 9.0)))
        if not _STATE["ok"]:
            _STATE["why"] = "a test copy did not arrive (got %s)" % (got,)
    except Exception as exc:                                     # noqa: BLE001
        _STATE["ok"] = False
        _STATE["why"] = "a test copy failed (%s: %s)" % (type(exc).__name__, exc)
    if _STATE["ok"]:
        _STATE["why"] = ""
    elif DEVICE_BRIDGE == "1":
        warnings.warn("HIPPYMFEM_DEVICE_BRIDGE=1, but the bridge cannot be used: "
                      + _STATE["why"], RuntimeWarning, stacklevel=3)


def available():
    """Whether JAX arrays and MFEM memory can be copied into each other on the GPU."""
    if not _STATE["checked"]:
        _check()
    return _STATE["ok"]


def why_not():
    """The reason :func:`available` is false, or ``""``."""
    if not _STATE["checked"]:
        _check()
    return _STATE["why"]


def stats(reset=False):
    """``(copies, bytes)`` moved over the bridge by this rank so far."""
    out = (_STATE["copies"], _STATE["bytes"])
    if reset:
        _STATE["copies"] = _STATE["bytes"] = 0
    return out


def address(pointer):
    """The integer address behind a pointer MFEM's Python interface returned."""
    return int(pointer)


def jax_address(x):
    """``(address, bytes)`` of the device buffer of a JAX array, once it is computed."""
    x.block_until_ready()
    return int(x.unsafe_buffer_pointer()), int(x.size) * int(x.dtype.itemsize)


def _copy(dst, src, nbytes, kind):
    if nbytes <= 0:
        return
    rc = _STATE["memcpy"](ctypes.c_void_p(dst), ctypes.c_void_p(src),
                          ctypes.c_size_t(nbytes), kind)
    if rc != 0:
        raise RuntimeError("the GPU runtime's copy failed with error %d" % rc)
    _STATE["copies"] += 1
    _STATE["bytes"] += int(nbytes)


def copy_from_jax(dst, x, start=0, count=None):
    """Copy ``count`` entries of the JAX array ``x``, from entry ``start``, to the
    device address ``dst``.  The copy is queued on the device; :func:`synchronize`
    waits for it."""
    base, total = jax_address(x)
    item = int(x.dtype.itemsize)
    n = int(x.size) - int(start) if count is None else int(count)
    if start < 0 or n < 0 or (start + n) * item > total:
        raise ValueError("entries [%d, %d) are outside an array of %d"
                         % (start, start + n, int(x.size)))
    _copy(int(dst), base + int(start) * item, n * item, _D2D)


def copy_device(dst, src, nbytes):
    """Copy ``nbytes`` from the device address ``src`` to the device address ``dst``."""
    _copy(int(dst), int(src), int(nbytes), _D2D)


def copy_from_host(dst, array):
    """Copy a contiguous numpy array to the device address ``dst``."""
    array = np.ascontiguousarray(array)
    _copy(int(dst), array.ctypes.data, array.nbytes, _H2D)


def copy_to_jax(src, count, dtype=np.float64):
    """A new JAX array holding ``count`` entries read from the device address ``src``.

    The array is created by JAX and filled before anything else has seen it; the
    producer of ``src`` must have finished (:func:`synchronize`)."""
    import jax.numpy as jnp

    dtype = np.dtype(dtype)
    # ``+ 0`` runs a computation, so the buffer is one JAX allocated for this array and
    # not one it shares with a constant
    out = jnp.zeros(int(count), dtype=dtype) + dtype.type(0)
    base, total = jax_address(out)
    _copy(base, int(src), int(count) * dtype.itemsize, _D2D)
    synchronize()
    return out


def synchronize():
    """Wait for the copies queued so far (and everything else on the device)."""
    rc = _STATE["sync"]()
    if rc != 0:
        raise RuntimeError("the GPU runtime's synchronize failed with error %d" % rc)

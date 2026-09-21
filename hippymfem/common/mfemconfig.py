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
    if not quiet and local == 0:
        print("hippymfem: MFEM on %s device %d of %d" % (kind, idx, n), flush=True)
        from .._jaxconfig import PIN_SKIPPED

        if PIN_SKIPPED and n > 1:
            # rank 0 only (no collective here): nothing else reports the stray contexts
            print("hippymfem: GPU not pinned (%s); each rank holds a context on all %d "
                  "cards" % (PIN_SKIPPED, n), flush=True)
    return dev


_DEVICE = []


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



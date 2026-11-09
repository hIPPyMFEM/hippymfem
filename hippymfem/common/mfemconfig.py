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
from ..config import env_flag
import re
import time
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

def _pool_settings():
    """What ``HIPPYMFEM_HYPRE_POOL`` and ``HIPPYMFEM_HYPRE_POOL_KEEP`` ask for:
    ``("auto", megabytes kept between set-ups)`` or ``(megabytes, None)``.

    A value that is neither ``auto`` nor a number raises, with the variable's name.  It
    was taken for 0: ``HIPPYMFEM_HYPRE_POOL=on`` ran without a pool, the opposite of what
    was typed, and a ``HIPPYMFEM_HYPRE_POOL_KEEP`` that could not be read kept the
    default, both without a word.  An empty value is the default."""
    name = "HIPPYMFEM_HYPRE_POOL"
    text = (os.environ.get(name, "auto") or "auto").strip().lower() or "auto"
    try:
        if text != "auto":
            return float(text), None
        name = "HIPPYMFEM_HYPRE_POOL_KEEP"
        text = os.environ.get(name, "").strip()
        return "auto", float(text) if text else HYPRE_POOL_KEEP_MB
    except ValueError:
        raise ValueError("%s must be %smegabytes, not %r"
                         % (name, "" if name.endswith("KEEP") else '"auto" or ', text)) from None


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
    (:func:`hippymfem.fem.parmat.local_par_matrix`).

    **This is the configuration that matters.**  The element kernels are a small share
    of a Newton step, so moving only them to the GPU gains little; moving the solves
    as well is worth more than an order of magnitude (see the GPU guide).

    Returns the ``mfem.Device`` it built, which must be kept alive as long as MFEM is
    used; this module holds a reference to it as well.
    """
    import mfem.par as mfem

    global DEVICE_INDEX
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
    # The kernel named in the environment is checked before anything is made (without a
    # device the call reads the name and sets nothing).  A mistyped name used to be refused
    # after the device was registered: a caller that caught the error, or a second import,
    # which returns the registered device as it is, then went on with one that had had no
    # check of the MPI library and no pool.
    spmv = os.environ.get("HIPPYMFEM_HYPRE_SPMV", "auto")
    set_hypre_spmv(spmv, comm)
    pool, keep = _pool_settings()           # (likewise: a value that cannot be read stops here)
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
    # Never destroyed from Python: MFEM's memory manager goes with the Device, and at the
    # interpreter's exit this module may be cleared before a script's own MFEM objects,
    # whose hypre matrices then read the memory manager in their destructor and fault
    # (HypreParMatrix::Destroy).  It lives until the process ends.
    dev.thisown = False
    _DEVICE.append(dev)
    DEVICE_INDEX = idx
    # (the build's own flag, not the backend that was asked for: a test that fakes the
    # backend on a host build must not reach MFEM's memory manager at exit; and once)
    if kind == "cuda" and mfem_has("MFEM_USE_CUDA") and not _TEARDOWN[0]:
        import atexit

        _TEARDOWN[0] = True
        atexit.register(_late_teardown)
    set_hypre_spmv(spmv, comm)
    global GPU_AWARE_SOLVES
    want = os.environ.get("HIPPYMFEM_GPU_AWARE_MPI", "").strip().lower()
    asked = None                           # what the variable asks of every exchange, if it does
    if want in ("0", "1", "on", "off", "true", "false", "yes", "no"):
        asked = want in ("1", "on", "true", "yes")
        set_hypre_gpu_aware_mpi(asked)
    elif want == "solve":
        # only while a Krylov solve runs on a hierarchy that is already set up
        # (linSolvers): checked now as if it were on throughout
        set_hypre_gpu_aware_mpi(True)
        GPU_AWARE_SOLVES = bool(_check_gpu_aware_mpi(comm, quiet))
        if set_hypre_gpu_aware_mpi(False):
            # (a hypre that is compiled to hand MPI its buffers, 2.31 or 2.32, does so
            # throughout: there is nothing to switch around a solve)
            GPU_AWARE_SOLVES = False
    aware = _check_gpu_aware_mpi(comm, quiet)
    if not quiet and local == 0 and _gpu_aware_fixed(_hypre_library()):
        # (a hypre whose exchange is compiled one way: what the variable asks beyond that
        # is said.  It used to be written into a field that this exchange does not read,
        # and the check of the MPI library passed or refused by that field.)
        if want == "solve" or (asked is not None and asked != bool(aware)):
            warnings.warn(
                "HIPPYMFEM_GPU_AWARE_MPI=%s changes nothing with this hypre: before hypre "
                "2.33 the exchange is compiled one way, and this library %s whatever it is "
                "asked at run time." % (want, "hands MPI its buffers on the GPU" if aware
                                        else "stages its buffers through CPU memory"),
                RuntimeWarning, stacklevel=2)
    # hypre's device memory: "auto" (the default) recycles up to HYPRE_POOL_SETUP_MB of
    # it while a BoomerAMG setup runs and keeps HYPRE_POOL_KEEP_MB of the smallest blocks
    # between setups (HIPPYMFEM_HYPRE_POOL_KEEP changes that; 0 keeps nothing); a number
    # of megabytes keeps that much at all times; "0" leaves every allocation to the
    # driver.
    if pool == "auto":
        set_hypre_pool(HYPRE_POOL_SETUP_MB, scoped=True, keep_megabytes=max(keep, 0.0))
    elif pool > 0:
        set_hypre_pool(pool)
    if not quiet and local == 0:
        print("hippymfem: MFEM on %s device %d of %d%s"
              % (kind, idx, n, "; hypre hands MPI its buffers on the GPU" if aware else ""),
              flush=True)
        from .._jaxconfig import PIN_SKIPPED

        if PIN_SKIPPED and n > 1:
            # rank 0 only (no collective here): nothing else reports the stray contexts
            print("hippymfem: GPU not pinned (%s); each rank holds a context on all %d "
                  "cards" % (PIN_SKIPPED, n), flush=True)
    return dev


_DEVICE = []

#: The index of the GPU that :func:`configure_device` put MFEM and hypre on, among the
#: GPUs this process may use; ``None`` on the host.  The element kernels pick the same
#: one, which :mod:`.devicebridge` checks before it copies between the two on the device.
DEVICE_INDEX = None


#: Path of a second, single-precision hypre library this process has loaded
#: (:mod:`hippymfem.algorithms.singlesolve`), which :func:`_hypre_library` must not
#: mistake for the one MFEM uses.
SINGLE_LIBRARY_PATH = None

_HYPRE_HANDLE = None


def _hypre_library():
    """A ``ctypes`` handle of the hypre library MFEM uses in this process, or ``None``."""
    global _HYPRE_HANDLE
    if _HYPRE_HANDLE is not None:
        return _HYPRE_HANDLE
    import ctypes

    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libHYPRE" in line:
                    path = line.split(None, 5)[-1].strip()
                    if SINGLE_LIBRARY_PATH and os.path.realpath(path) == SINGLE_LIBRARY_PATH:
                        continue
                    _HYPRE_HANDLE = ctypes.CDLL(path)
                    return _HYPRE_HANDLE
    except OSError:
        pass
    return None


def _single_library():
    """A ``ctypes`` handle of the single-precision hypre that this process loaded beside
    MFEM's (:data:`SINGLE_LIBRARY_PATH`), or ``None``."""
    if not SINGLE_LIBRARY_PATH:
        return None
    import ctypes

    try:
        return ctypes.CDLL(SINGLE_LIBRARY_PATH)
    except OSError:
        return None


_MFEM_HANDLE = None

#: BoomerAMG settings beyond the dozen that MFEM's wrapper makes and PyMFEM passes on, by
#: name, with the call of hypre's C interface each one is (:func:`set_boomeramg_options`).
BOOMERAMG_SETTERS = {
    "coarsen": ("HYPRE_BoomerAMGSetCoarsenType", int),
    "agg": ("HYPRE_BoomerAMGSetAggNumLevels", int),
    "agg_interp": ("HYPRE_BoomerAMGSetAggInterpType", int),
    "agg_pmax": ("HYPRE_BoomerAMGSetAggPMaxElmts", int),
    "relax": ("HYPRE_BoomerAMGSetRelaxType", int),
    "relax_wt": ("HYPRE_BoomerAMGSetRelaxWt", float),
    "outer_wt": ("HYPRE_BoomerAMGSetOuterWt", float),
    "sweeps": ("HYPRE_BoomerAMGSetNumSweeps", int),
    "theta": ("HYPRE_BoomerAMGSetStrongThreshold", float),
    "interp": ("HYPRE_BoomerAMGSetInterpType", int),
    "pmax": ("HYPRE_BoomerAMGSetPMaxElmts", int),
    "trunc": ("HYPRE_BoomerAMGSetTruncFactor", float),
    "levels": ("HYPRE_BoomerAMGSetMaxLevels", int),
    "coarse_size": ("HYPRE_BoomerAMGSetMaxCoarseSize", int),
    "min_coarse_size": ("HYPRE_BoomerAMGSetMinCoarseSize", int),
    # a hierarchy ends at its first level of at most this many rows and one process
    # finishes it, gathered (check_seq_threshold: a device needs a patched hypre)
    "seq_threshold": ("HYPRE_BoomerAMGSetSeqThreshold", int),
    "redundant": ("HYPRE_BoomerAMGSetRedundant", int),
    "coarse_relax": ("HYPRE_BoomerAMGSetCycleRelaxType", int),     # (type, 3): the coarsest level
    "coarse_sweeps": ("HYPRE_BoomerAMGSetCycleNumSweeps", int),    # (count, 3)
    "keep_transpose": ("HYPRE_BoomerAMGSetKeepTranspose", int),
    "rap2": ("HYPRE_BoomerAMGSetRAP2", int),
    "mod_rap2": ("HYPRE_BoomerAMGSetModuleRAP2", int),
    "cheby_order": ("HYPRE_BoomerAMGSetChebyOrder", int),
    "cheby_fraction": ("HYPRE_BoomerAMGSetChebyFraction", float),
    "print": ("HYPRE_BoomerAMGSetPrintLevel", int),
    "debug": ("HYPRE_BoomerAMGSetDebugFlag", int),         # 1: the time of each phase of a setup, by level
}


def parse_boomeramg_options(text, what="HIPPYMFEM_AMG"):
    """``"relax=7,pmax=6"`` as a dict of the names of :data:`BOOMERAMG_SETTERS`."""
    out = {}
    for item in (text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in BOOMERAMG_SETTERS:
            raise ValueError("%s: unknown setting %r (known: %s)"
                             % (what, key, ", ".join(sorted(BOOMERAMG_SETTERS))))
        out[key] = BOOMERAMG_SETTERS[key][1](val)
    return out


def seq_on_device(library):
    """Whether a hypre (a ``ctypes`` handle) can finish a device hierarchy with a gathered
    level: one built with ``tools/hypre-3.2.0-seqdevice.patch``, which marks itself with
    a function of its own.  The function returns the patch's version, and the third is
    asked for.  In the first every rank solved the gathered system with a hierarchy of
    its own coarsening, and the ranks' corrections did not fit together; the second
    still gathered a level that the coarsening could not make smaller, whatever its
    size, and solved the gathered level of a system with the wrong functions."""
    if library is None or not hasattr(library, "hypre_BoomerAMGSeqOnDevice"):
        return False
    try:
        return int(library.hypre_BoomerAMGSeqOnDevice()) >= 3
    except Exception:                                            # noqa: BLE001
        return False


def seq_solvers(pc, comm=None, library=None):
    """Who solves the gathered level of a BoomerAMG that is set up (``seq_threshold``, hypre
    on a device): ``"all"`` every rank that holds rows of it, ``"first"`` the first of
    them, ``None`` where there is no such level or the library cannot tell.  ``pc`` is an
    ``mfem.HypreBoomerAMG`` or hypre's handle of one (of ``library``, default the hypre
    of this process); with ``comm`` the answer is one for all its ranks."""
    import ctypes

    H = library if library is not None else _hypre_library()
    if H is None or not hasattr(H, "hypre_BoomerAMGSeqSolvers"):
        return None
    handle = pc if isinstance(pc, (int, ctypes.c_void_p)) else boomeramg_handle(pc)
    if not handle:
        return None
    fn = H.hypre_BoomerAMGSeqSolvers
    fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_int
    mode = int(fn(handle if isinstance(handle, ctypes.c_void_p) else ctypes.c_void_p(handle)))
    if comm is not None:
        mode = max(comm.allgather(mode))
    return {1: "all", 0: "first"}.get(mode)


def check_seq_threshold(options, library, device):
    """Refuse ``seq_threshold`` where hypre would fault on it, with the reason.

    ``seq_threshold=n`` ends a hierarchy at its first level of at most ``n`` rows (in
    all, not a rank) and gathers that level on the ranks that hold rows of it, where a
    hierarchy of one process finishes it.  On thousands of ranks the last levels are a
    few rows a rank that talk to many ranks, and each costs four exchanges a cycle: a
    third to a half of a cycle of a linear-element hierarchy on 512 MI250X.  hypre's
    own code for this reads device memory through host pointers; with the patch the
    gathered level is solved on a device: the first rank's, which sends the solution to
    the others, or with ``redundant=1`` every rank's (each keeps its own rows; the patch
    makes the ranks' coarsenings draw the same random numbers, compares the solvers on
    one right-hand side, and lets the first rank solve where they differ)."""
    if int(options.get("seq_threshold", 0) or 0) > 0 and device and not seq_on_device(library):
        raise RuntimeError("the BoomerAMG option seq_threshold needs, with hypre on a device, a "
                           "hypre built with tools/hypre-3.2.0-seqdevice.patch (its third "
                           "version); this one would fault in its set-up, give corrections "
                           "that differ from rank to rank, or gather a level that it cannot "
                           "coarsen whatever its size")


def _mfem_library():
    """A ``ctypes`` handle of the MFEM library of this process, or ``None``."""
    global _MFEM_HANDLE
    if _MFEM_HANDLE is not None:
        return _MFEM_HANDLE
    import ctypes

    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libmfem" in line:
                    _MFEM_HANDLE = ctypes.CDLL(line.split(None, 5)[-1].strip())
                    return _MFEM_HANDLE
    except OSError:
        pass
    return None


#: whether :func:`_late_teardown` is registered (once a process)
_TEARDOWN = [False]


def _late_teardown():
    """Register MFEM's ``Hypre::Finalize`` and ``MemoryManager::Destroy`` as exit handlers
    of the process, from a Python ``atexit`` handler (CUDA builds, :func:`configure_device`).

    The Device is never freed from Python (above), so MFEM's memory manager lives until
    the static destructors of the process, which run after the CUDA runtime's own
    teardown, registered when CUDA started.  Device memory still held by MFEM or hypre
    is then freed into a runtime that is shutting down: hypre prints "driver shutting
    down" and MFEM aborts, a finished run ending with status 134 (``test_vectorfe`` on
    Blackwell cards; HIP tolerates it).  Exit handlers run last registered first, and
    these are registered during the interpreter's finalization, after everything else:
    they run after every Python object is gone and before CUDA's teardown.  Afterwards
    MFEM's own destructors find the manager gone and the library finalized, and do
    nothing.  A library without these symbols is left as it is.  A script that makes an
    ``mfem.Device`` of its own beside :func:`configure_device` has it destroyed during
    the finalization, after the check below, and MFEM then stops at the second
    ``Destroy``: keep to the one Device that :func:`configure_device` makes."""
    import ctypes

    lib = _mfem_library()
    if lib is None:
        return
    try:
        # (a manager already destroyed, by a Device of the script's own, is left alone:
        # MFEM stops the process on a second Destroy)
        if not ctypes.c_bool.in_dll(lib, "_ZN4mfem13MemoryManager6existsE").value:
            return
        destroy = ctypes.cast(getattr(lib, "_ZN4mfem13MemoryManager7DestroyEv"),
                              ctypes.c_void_p).value
        finalize = ctypes.cast(getattr(lib, "_ZN4mfem5Hypre8FinalizeEv"), ctypes.c_void_p).value
        manager = ctypes.addressof(ctypes.c_char.in_dll(lib, "_ZN4mfem2mmE"))
        register = ctypes.CDLL(None).__cxa_atexit
    except (AttributeError, ValueError, OSError):
        return
    register.restype = ctypes.c_int
    register.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    register(destroy, manager, None)        # runs second: MemoryManager::Destroy on mfem::mm
    register(finalize, None, None)          # runs first: Hypre::Finalize


def boomeramg_handle(pc):
    """hypre's own handle (``HYPRE_Solver``) of an ``mfem.HypreBoomerAMG``, or ``None``.

    MFEM converts its preconditioner to that handle with a conversion operator, which a
    shared MFEM library exports and PyMFEM does not wrap; it is called here by its symbol.
    With the handle every option of BoomerAMG is within reach, not only those that MFEM's
    class has a method for (:func:`set_boomeramg_options`).
    """
    lib = _mfem_library()
    name = "_ZNK4mfem14HypreBoomerAMGcvP19hypre_Solver_structEv"
    if lib is None or not hasattr(lib, name):
        return None
    import ctypes

    try:
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
        return fn(int(pc.this)) or None
    except Exception:                                            # noqa: BLE001
        return None


def set_boomeramg_options(pc, options):
    """Apply ``options`` (a dict of the names of :data:`BOOMERAMG_SETTERS`) to an
    ``mfem.HypreBoomerAMG`` that is not set up yet, through hypre's C interface.

    Raises ``RuntimeError`` where the handle cannot be reached (a static MFEM, a build
    whose symbols differ) and ``ValueError`` for a value hypre refuses.
    """
    if not options:
        return pc
    import ctypes

    from .parvector import device_active

    H, amg = _hypre_library(), boomeramg_handle(pc)
    if H is None or amg is None:
        raise RuntimeError("BoomerAMG options %s need hypre's handle of the preconditioner, which "
                           "this MFEM library does not give" % ", ".join(sorted(options)))
    # (whether MFEM is on a card, not whether a device was configured: the "cpu" one,
    # which the drivers configure on a host run, leaves hypre in host memory)
    check_seq_threshold(options, H, device_active())
    amg = ctypes.c_void_p(amg)
    # What a setter returns is hypre's error flag, which is the process's and stays up
    # after the call that raised it, on the ranks where it did (a set-up that met a zero
    # row; a solve of hypre's PCG that ended at its limit, where nothing read it).
    # Cleared first, it is the setter's own verdict, the same on every rank.
    if hasattr(H, "HYPRE_ClearAllErrors"):
        H.HYPRE_ClearAllErrors()
    # hypre's setters of the relaxation and of its sweeps reset those of the coarsest
    # level, so the coarsest level's own come after them, wherever they were written
    for key, val in sorted(options.items(),
                           key=lambda item: item[0] in ("coarse_relax", "coarse_sweeps")):
        name, kind = BOOMERAMG_SETTERS[key]
        fn = getattr(H, name)
        if key in ("coarse_relax", "coarse_sweeps"):
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
            rc = fn(amg, int(val), 3)
        else:
            # (the hypre that MFEM uses has double-precision reals)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_double if kind is float else ctypes.c_int]
            fn.restype = ctypes.c_int
            rc = fn(amg, val)
        if rc:
            if hasattr(H, "HYPRE_ClearAllErrors"):
                H.HYPRE_ClearAllErrors()
            raise ValueError("hypre refused the BoomerAMG option %s=%r (%s returned %d)" % (key, val, name, rc))
    return pc


def hypre_gpu_aware_mpi():
    """Whether the hypre of this process hands MPI its buffers on the GPU.

    hypre exchanges the values of shared dofs at every product with a parallel matrix.
    As PyMFEM builds it, it copies them to CPU memory, sends them from there and copies
    what it receives back.  Configured with ``HYPRE_WITH_GPU_AWARE_MPI``
    (``tools/rebuild_hypre.sh --gpu-aware-mpi``) it gives MPI the address of the buffer
    on the GPU instead, which needs an MPI that can read one (:func:`mpi_gpu_support`).
    Before hypre 2.33 the option is fixed when hypre is compiled, and this reads what the
    loaded library was built with; from 2.33 on the option presets a setting of the
    library (:func:`set_hypre_gpu_aware_mpi`), and this reads what is in force.
    """
    lib = _hypre_library()
    if lib is None or not hasattr(lib, "hypre_GetGpuAwareMPI"):
        return False
    import ctypes

    try:
        lib.hypre_GetGpuAwareMPI.restype = ctypes.c_int
        return bool(lib.hypre_GetGpuAwareMPI())
    except Exception:                                            # noqa: BLE001
        return False


def _gpu_aware_fixed(lib):
    """Whether the hypre ``lib`` has the switch of :func:`set_hypre_gpu_aware_mpi` and an
    exchange that does not follow it: hypre 2.31 and 2.32.

    The switch and the field of the handle that it writes came with 2.31, and the exchange
    (``hypre_ParCSRCommHandleCreate_v2``) reads the field from 2.33 on.  In between it is
    compiled one way or the other by ``HYPRE_WITH_GPU_AWARE_MPI`` alone, as MFEM notes
    where it prepares one (``linalg/hypre.cpp``): the field says nothing of what a library
    does once something has written it.  The build option presets it, so the field of such
    a library is never written here and goes on saying what the library was built to do.
    """
    if (lib is None or not hasattr(lib, "hypre_GetGpuAwareMPI")
            or not hasattr(lib, "HYPRE_VersionNumber")):
        return False
    import ctypes

    number = ctypes.c_longlong(0)                   # (room for hypre's integer, 4 or 8 bytes)
    try:
        lib.HYPRE_VersionNumber(None, None, None, ctypes.byref(number))
    except Exception:                                            # noqa: BLE001
        return False
    return 0 < number.value < 23300


#: ``HIPPYMFEM_GPU_AWARE_MPI=solve``: hypre hands MPI its device buffers during the Krylov
#: solves on a hierarchy that is set up, and stages them through CPU memory otherwise.  A
#: product with a parallel matrix exchanges through two buffers that the matrix keeps, which
#: an MPI library registers once; a BoomerAMG setup and a matrix product exchange through
#: buffers of their own every time, and with Cray MPICH each of those costs a registration:
#: the setup of 0.8 s at 128^3 on 8 MI250X had not ended after 8 minutes.
#:
#: Experimental, and off unless asked for.  On Frontier (Cray MPICH 9.1.0, October 2026) it
#: took 10 to 20 % off an iteration on up to 64 MI250X and at 512^3 on 512, and deadlocked or
#: faulted every time at 320^3 on 128 and at 400^3 on 256, with ranks of one node left in
#: different exchanges of a V-cycle, whatever the MPI was told (no transfers between GPUs of
#: a node, the signalled handle cache, no asynchronous copies, software matching).  hypre's
#: own staging with page-locked buffers (tools/hypre-3.2.0-staging.patch) gains as much.
GPU_AWARE_SOLVES = False


def set_hypre_gpu_aware_mpi(on=True):
    """Make hypre hand MPI its buffers on the GPU (``True``) or stage them through CPU
    memory (``False``), whatever it was configured with.

    Since hypre 2.33 the choice is a field of its handle that the build option only
    presets, so one library serves both.  hypre 2.31 and 2.32 have the field and this
    switch, but their exchange is compiled one way and does not read it
    (:func:`_gpu_aware_fixed`): such a library is left as it was built, and what is
    returned says which way that is.  The MPI library has to read device memory
    (:func:`mpi_gpu_support`): Cray MPICH does with ``MPICH_GPU_SUPPORT_ENABLED=1`` and
    its GPU transport library in the process (``libmpi_gtl_hsa.so`` on AMD cards, linked
    or preloaded).  ``HIPPYMFEM_GPU_AWARE_MPI=1`` or ``0`` sets it for
    :func:`configure_device`.  Returns what is now in force, or ``None`` when the
    library has no such switch.
    """
    lib = _hypre_library()
    if lib is None or not hasattr(lib, "hypre_SetGpuAwareMPI"):
        return None
    import ctypes

    if not _gpu_aware_fixed(lib):
        lib.hypre_SetGpuAwareMPI(ctypes.c_int(1 if on else 0))
    return hypre_gpu_aware_mpi()


def mpi_gpu_support():
    """Whether the MPI library of this process says it can send from GPU memory:
    ``True``, ``False``, or ``None`` when it has no way of saying.

    Open MPI answers through ``MPIX_Query_cuda_support``, and MPICH 4 and Cray MPICH
    through that and ``MPIX_Query_hip_support``, one call per vendor: the one for the
    cards of this build is asked.  An older MPICH has no such call; its support is
    switched on by ``MPICH_GPU_SUPPORT_ENABLED=1``, which is what is read for it.
    ``True`` says that the library can, not that the transport chosen for this run does:
    Open MPI 4 with a UCX that was built without CUDA still answers yes, and needs
    ``--mca pml ob1 --mca btl self,smcuda``.
    """
    import ctypes

    try:
        with open("/proc/self/maps") as f:
            names = {line.split(None, 5)[-1].strip() for line in f if "/libmpi" in line}
    except OSError:
        names = set()
    query = "MPIX_Query_hip_support" if mfem_gpu_backend() == "hip" else "MPIX_Query_cuda_support"
    for name in sorted(names):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        if hasattr(lib, query):
            ask = getattr(lib, query)
            ask.restype = ctypes.c_int
            return bool(ask())
    v = os.environ.get("MPICH_GPU_SUPPORT_ENABLED")
    if v is not None:
        return v.strip() == "1"
    return None


def _check_gpu_aware_mpi(comm=None, quiet=False):
    """Stop, with the reason, a run whose hypre would hand device addresses to an MPI
    that reads them as CPU memory: the alternative is a segmentation fault inside MPI
    at the first product with a parallel matrix."""
    if not hypre_gpu_aware_mpi():
        return False
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        return True
    ok = mpi_gpu_support()
    if ok is False:
        # hypre goes back to staging before the run is stopped.  The device is configured
        # by now and a second configure_device returns it without this check, so a caller
        # that catches the error would otherwise go on with the exchange it was refused.
        # A hypre before 2.33 that is compiled to hand MPI its buffers cannot go back, and
        # the variable is no way out of this for it: the advice is then the other library.
        compiled = set_hypre_gpu_aware_mpi(False)
        raise RuntimeError(
            "this hypre gives MPI the addresses of its buffers on the GPU (built with "
            "HYPRE_WITH_GPU_AWARE_MPI, or HIPPYMFEM_GPU_AWARE_MPI=1), and the MPI library "
            "of this run says it cannot read this card's memory, so it would read a "
            "device address as CPU memory and crash at the first product with a parallel "
            "matrix.\n"
            "  Run with a GPU-aware MPI (Open MPI built --with-cuda; with Open MPI 4 "
            "and a UCX without CUDA, 'mpirun --mca pml ob1 --mca btl self,smcuda'; Cray "
            "MPICH with MPICH_GPU_SUPPORT_ENABLED=1 and its GTL library linked or "
            "preloaded), or leave the exchange to hypre, which then stages its buffers "
            "through CPU memory (%s)."
            % ("the libHYPRE.so that the PyMFEM build scripts produce: this hypre is older "
               "than 2.33 and compiled to hand MPI its buffers, which HIPPYMFEM_GPU_AWARE_MPI "
               "does not change" if compiled else
               "HIPPYMFEM_GPU_AWARE_MPI=0, or the libHYPRE.so that the PyMFEM build scripts "
               "produce"))
    if ok is None and not quiet and comm.Get_rank() == 0:
        import warnings

        warnings.warn(
            "this hypre hands MPI its buffers on the GPU (HYPRE_WITH_GPU_AWARE_MPI), "
            "and this MPI library does not say whether it can send from GPU memory. "
            "If the run crashes inside MPI at its first parallel matrix-vector "
            "product, that is why.", RuntimeWarning, stacklevel=3)
    return True


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
    vendor's library, and with cuSPARSE the time of the product with such a block grows
    with its rows, not its nonzeros: 5.4 ms for a block of an interpolation matrix with
    701 nonzeros in 2.2 million rows, where hypre's own kernel takes 0.16 ms or less.
    On one rank there are no off-diagonal blocks and the vendor's kernel is the faster
    one; on two to sixteen ranks hypre's kernel made a Krylov iteration 11 to 36 %
    faster (the GPU guide, "Several GPUs").

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
    ``driver_frees``; ``driver_bytes``, the sum of the blocks the driver was asked for, and
    ``driver_seconds``, the time its two functions took; ``peak_footprint``, the most the
    blocks in use and the kept ones came to when a block was taken from the driver;
    ``trims``, the times the pool was emptied (the device was out of memory, or a caller
    asked), and ``refused``, the requests the driver refused even then; ``min_free``, the
    least free memory the card had when hypre took a block of 16 MiB or more from the
    driver, and ``device_total``, the card's memory.
    """

    #: size classes per factor of two; a recycled block is at most 2**(2/8) = 1.19
    #: times the size asked for
    CLASSES = 8.0

    #: Between setups a scoped pool holds at most this share of the most hypre has had
    #: in use at once (and at most ``keep``): a small problem keeps little.
    KEEP_SHARE = 0.25

    #: The card's free memory is asked after a block of at least this many bytes.
    _MEM_INFO_FROM = 1 << 24

    def __init__(self, hypre, runtime, max_cached, max_block=None, scoped=False, keep=0):
        import ctypes

        #: ``True``: the pool keeps up to ``limit`` bytes of freed blocks between
        #: :meth:`open` and :meth:`close`, that is, while a BoomerAMG setup runs, and up
        #: to ``keep`` bytes outside it.  ``False``: up to ``max_cached`` at all times.
        self.scoped = bool(scoped)
        self.limit = int(max_cached)
        self.keep = int(keep) if self.scoped else int(max_cached)

        self._ct = ctypes
        # the runtime's own pair, cudaMalloc and cudaFree or hipMalloc and hipFree
        vendor = "cuda" if hasattr(runtime, "cudaMalloc") else "hip"
        self._dev_malloc = getattr(runtime, vendor + "Malloc")
        self._dev_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._dev_malloc.restype = ctypes.c_int
        self._dev_free = getattr(runtime, vendor + "Free")
        self._dev_free.argtypes = [ctypes.c_void_p]
        self._dev_free.restype = ctypes.c_int
        # reads the runtime's last error, which resets it
        self._last_error = getattr(runtime, vendor + "GetLastError", None)
        # the card's free and total memory, asked after each block taken from the driver
        self._mem_info = getattr(runtime, vendor + "MemGetInfo", None)
        if self._mem_info is not None:
            self._mem_info.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.POINTER(ctypes.c_size_t)]
            self._mem_info.restype = ctypes.c_int
        self._mem_free, self._mem_total = ctypes.c_size_t(), ctypes.c_size_t()
        #: the least free memory of the card seen when hypre took a block from the driver,
        #: and the card's memory, in bytes (0 until asked)
        self.min_free = self.device_total = 0
        #: how many requests the driver refused even after the pool was emptied
        self.refused = 0
        self.max_cached = self.keep
        self.max_block = int(max_block if max_block is not None else max_cached)
        self._bins = {}                    # size class -> [(pointer, size), ...]
        self._size = {}                    # pointer -> size, for the blocks handed out
        self.cached = self.peak_cached = self.in_use = self.peak_in_use = 0
        self.requests = self.from_pool = self.driver_allocs = self.driver_frees = 0
        self.driver_bytes, self.driver_seconds = 0, 0.0
        self.peak_footprint = self.trims = 0
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
        t0 = time.perf_counter()
        p = self._ct.c_void_p()
        if self._dev_malloc(self._ct.byref(p), size) != 0 or not p.value:
            # out of device memory: give back what the pool holds and try once more
            self.trim()
            p = self._ct.c_void_p()
            if self._dev_malloc(self._ct.byref(p), size) != 0 or not p.value:
                p = self._ct.c_void_p()
                self._refuse(size)
            # The runtime keeps a refusal as its last error, and MFEM reads the last
            # error after its next kernel and takes it for that kernel's.  Read it here.
            if self._last_error is not None:
                self._last_error()
        self.driver_allocs += 1
        self.driver_bytes += size
        self.driver_seconds += time.perf_counter() - t0
        if self._mem_info is not None and size >= self._MEM_INFO_FROM:
            # (how near the card came to full, which no counter of the pool's own shows:
            # JAX and MFEM hold memory of the same card)
            if self._mem_info(self._ct.byref(self._mem_free),
                              self._ct.byref(self._mem_total)) == 0:
                free = int(self._mem_free.value)
                if not self.device_total or free < self.min_free:
                    self.min_free = free
                self.device_total = int(self._mem_total.value)
        return p.value

    def _driver_free(self, p):
        t0 = time.perf_counter()
        self._dev_free(p)
        self.driver_frees += 1
        self.driver_seconds += time.perf_counter() - t0

    def _refuse(self, size):
        """The card has no room for a block hypre asks for.  hypre stops the run at a
        null pointer without a word, so say what happened, once."""
        self.refused += 1
        if self.refused == 1:
            import sys

            print("hippymfem: the GPU is out of memory: hypre asked for %.2f GiB with %.2f GiB of its "
                  "own in use.  Use more ranks, or leave hypre more of the card "
                  "(HIPPYMFEM_GPU_MEM_FRACTION)." % (size / 2 ** 30, self.in_use / 2 ** 30),
                  file=sys.stderr, flush=True)

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
            if self.in_use + self.cached > self.peak_footprint:
                self.peak_footprint = self.in_use + self.cached
        out[0] = p

    def _free(self, p):
        if not p:
            return
        s = self._size.pop(p, 0)
        if s:
            self.in_use -= s
            if s <= self.max_block:
                if self.cached + s > self.max_cached:
                    self._make_room(s)
                if self.cached + s <= self.max_cached:
                    self._bins.setdefault(self._class(s), []).append((p, s))
                    self.cached += s
                    if self.cached > self.peak_cached:
                        self.peak_cached = self.cached
                    return
        # not handed out by this pool, too large, or the pool is full
        self._driver_free(p)

    def _largest(self):
        """The size class of the largest block held, or ``None``."""
        top = None
        for k, blocks in self._bins.items():
            if blocks and (top is None or k > top):
                top = k
        return top

    def _make_room(self, s):
        """The pool is full: give larger blocks back to the driver, largest first, until
        a block of ``s`` bytes fits.  Blocks no larger than ``s`` stay, so the pool ends
        up holding the small blocks, which are most of the calls and little of the
        memory."""
        c = self._class(s)
        while self.cached + s > self.max_cached:
            top = self._largest()
            if top is None or top <= c:
                return
            p, big = self._bins[top].pop()
            self.cached -= big
            self._driver_free(p)

    def _shrink(self):
        """Give blocks back to the driver, largest first, down to ``max_cached``."""
        while self.cached > self.max_cached:
            top = self._largest()
            if top is None:
                break
            p, big = self._bins[top].pop()
            self.cached -= big
            self._driver_free(p)

    def trim(self):
        """Return every block the pool holds to the driver; returns the bytes freed."""
        freed = self.cached
        if freed:
            self.trims += 1
        # The bins are taken out before the first call of the driver.  Python's collector
        # can start inside this loop, at one of those calls, and where it finalizes a
        # dropped matrix, hypre frees that matrix's arrays through this pool: such a block
        # goes into bins of its own, not into the table that the loop is reading.
        bins, self._bins, self.cached = self._bins, {}, 0
        for blocks in bins.values():
            for p, _s in blocks:
                self._driver_free(p)
        return freed

    def open(self):
        """A setup is about to run: a scoped pool may now hold ``limit`` bytes."""
        if self.scoped:
            self.max_cached = self.limit

    def close(self):
        """A setup is over: a scoped pool goes back to the ``keep`` bytes it may hold
        between setups, or to a quarter of the most hypre has had in use if that is
        less, and returns the rest, largest blocks first.  The calls are not
        counted against :meth:`open`: a solver that was given an operator and never
        solved must not leave the pool at its larger limit, so the first solve of any
        solver ends the scope, and a setup still pending then runs with ``keep``."""
        if self.scoped:
            self.max_cached = min(self.keep, int(self.KEEP_SHARE * self.peak_in_use))
            self._shrink()

    def uninstall(self):
        """Give hypre its own allocator back and release what the pool holds.  Blocks
        still in use are freed by hypre's own ``cudaFree`` afterwards, which takes any
        device pointer."""
        if self.installed:
            # A single-precision hypre beside MFEM's allocates through the module's pool
            # as well (singlesolve._Library).  Left with the pool's two functions it would
            # go on filling a pool that nothing opens, closes or trims any more.
            single = _single_library() if self is HYPRE_POOL else None
            for library in (self._hypre, single):
                if library is not None and hasattr(library, "hypre_SetUserDeviceMalloc"):
                    library.hypre_SetUserDeviceMalloc(None)
                    library.hypre_SetUserDeviceMfree(None)
            self.installed = False
            self.trim()

    def stats(self):
        return {k: getattr(self, k) for k in (
            "requests", "from_pool", "driver_allocs", "driver_frees", "driver_bytes",
            "driver_seconds", "cached", "peak_cached", "in_use", "peak_in_use",
            "peak_footprint", "trims", "refused", "min_free", "device_total")}


#: The pool :func:`set_hypre_pool` installed, or ``None``.
HYPRE_POOL = None


#: Megabytes the default pool may hold while a BoomerAMG setup is running.
HYPRE_POOL_SETUP_MB = 1024.0

#: Megabytes the default pool may hold between setups (``HIPPYMFEM_HYPRE_POOL_KEEP``).
HYPRE_POOL_KEEP_MB = 512.0


def hypre_pool_open():
    """A BoomerAMG setup is about to run: let a scoped pool recycle its arrays."""
    if HYPRE_POOL is not None and HYPRE_POOL.installed:
        HYPRE_POOL.open()


def hypre_pool_close():
    """The setup is over: a scoped pool returns all but what it keeps between setups."""
    if HYPRE_POOL is not None and HYPRE_POOL.installed:
        HYPRE_POOL.close()


def hypre_pool_trim():
    """Return everything the pool holds to the driver (for a caller that is short of
    device memory); returns the bytes freed."""
    if HYPRE_POOL is not None and HYPRE_POOL.installed:
        return HYPRE_POOL.trim()
    return 0


def set_hypre_pool(megabytes=1024.0, max_block_megabytes=None, scoped=False,
                   keep_megabytes=0.0):
    """Let hypre recycle its device memory through a pool that may hold ``megabytes``.

    A hypre built without Umpire and without its own device pool, which is what the
    PyMFEM build scripts produce, takes every device array from ``cudaMalloc`` and
    returns it with ``cudaFree`` (``hipMalloc`` and ``hipFree`` on AMD cards).  One BoomerAMG setup makes about two thousand such
    pairs of calls.  Each is a trip into the driver that costs tens of microseconds
    when one process uses the node and several hundred when sixteen do, more than a
    second of a setup of 1.5 s (the GPU guide, "Several GPUs").

    hypre lets the application supply the two functions, and this supplies a pair that
    keeps freed blocks and hands them out again.  A new block has exactly the size
    asked for; a freed one is kept unless it is larger than ``max_block_megabytes``
    (default: ``megabytes``) or the pool already holds ``megabytes``, in which case
    larger blocks are given back to the driver to make room for it, and it goes back
    itself when there are none; a request is served by a kept block of at most 1.19
    times its size.  So the pool costs at most ``megabytes`` of device memory per rank
    beyond what hypre is using, and :meth:`HyprePool.trim` returns that at any time.
    If the driver refuses an allocation the pool is emptied and the allocation retried,
    and the element kernels empty it when they run out of device memory.

    Nothing but the time of an allocation changes.

    With ``scoped=True`` the pool has two limits: ``megabytes`` while a BoomerAMG setup
    runs (between :func:`hypre_pool_open`, which a solver calls when it is given an
    operator, and :func:`hypre_pool_close`, after its first solve) and
    ``keep_megabytes`` otherwise, or a quarter of the most hypre has had in use at
    once when that is less, so that a small problem keeps little.  The first serves
    the setup's own temporaries.  The second carries blocks from one setup to the
    next, and from the hierarchy a solver gives up to the one it builds.  When the
    pool is full a freed block displaces larger ones, so what it keeps are the
    smallest, which are the many: nine in ten of a forward solve's allocations are
    under 1 MB.

    :func:`configure_device` installs the scoped pool by default
    (``HIPPYMFEM_HYPRE_POOL=auto``: :data:`HYPRE_POOL_SETUP_MB` during a setup and
    :data:`HYPRE_POOL_KEEP_MB` between setups, the latter changed by
    ``HIPPYMFEM_HYPRE_POOL_KEEP``, in megabytes, where 0 keeps nothing).
    ``HIPPYMFEM_HYPRE_POOL=<megabytes>`` asks for one limit at all times, and ``0`` for
    no pool.  CUDA and HIP builds.  Returns the :class:`HyprePool`, or ``None`` when
    hypre is not on a device or the hooks are not available.
    """
    global HYPRE_POOL
    if HYPRE_POOL is not None and HYPRE_POOL.installed:
        HYPRE_POOL.scoped = bool(scoped)
        HYPRE_POOL.limit = int(megabytes * 2 ** 20)
        HYPRE_POOL.keep = int(keep_megabytes * 2 ** 20) if scoped else HYPRE_POOL.limit
        HYPRE_POOL.max_block = int((max_block_megabytes or megabytes) * 2 ** 20)
        HYPRE_POOL.max_cached = HYPRE_POOL.keep
        HYPRE_POOL._shrink()
        return HYPRE_POOL
    backend = mfem_gpu_backend()
    if not _DEVICE or backend not in ("cuda", "hip"):
        return None
    import atexit
    import ctypes

    hypre = _hypre_library()
    runtime = None
    name = "libcudart.so" if backend == "cuda" else "libamdhip64.so"
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if name in line:
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
                           else max_block_megabytes * 2 ** 20, scoped=scoped,
                           keep=keep_megabytes * 2 ** 20)
    # hypre frees its last arrays while the interpreter shuts down, when a callback
    # into Python is no longer safe: hand the allocator back before that
    atexit.register(HYPRE_POOL.uninstall)
    return HYPRE_POOL


def hypre_pool_megabytes():
    """Megabytes the installed pool may hold between setups; 0 without a pool."""
    if HYPRE_POOL is None or not HYPRE_POOL.installed:
        return 0.0
    return HYPRE_POOL.keep / 2 ** 20


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

    if not env_flag("HIPPYMFEM_AUTO_DEVICE", True):
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



# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""JAX configuration, applied before JAX is imported anywhere.

Import this module first from anything that touches JAX.  These settings have to
be in place before the first ``import jax``, because JAX reads them once:

``JAX_ENABLE_X64``
    Always on.  ``HIPPYMFEM_PRECISION=fp32`` (:data:`hippymfem.fem.kernel.PRECISION`)
    is a tool for a preconditioner or a Gauss-Newton approximation, not for the
    solve path: it speeds up the kernel but barely the assembly, and the forward
    solve's own linearity check rejects the result (``benchmarks/DESIGN_NOTES.md``,
    section 1).

``JAX_PLATFORMS``
    Defaults to ``"cpu"``.  Set ``HIPPYMFEM_DEVICE=gpu``, or ``auto`` for a GPU
    whenever the process can see one (or ``JAX_PLATFORMS``
    directly) to run the element kernels on the GPU; see below for why it is not
    the default and what the GPU path covers.

``XLA_PYTHON_CLIENT_PREALLOCATE`` / ``XLA_PYTHON_CLIENT_MEM_FRACTION``
    JAX otherwise claims 75% of GPU memory *per process* on import, so N ranks
    sharing one GPU each try to take three quarters of it and the run dies with
    ``CUDA_ERROR_OUT_OF_MEMORY``.  Preallocation is off, and the per-process
    fraction is capped.

Why the GPU is opt-in
---------------------

``HIPPYMFEM_DEVICE=gpu`` puts the **AD layer** on the card: the batched element
kernels, the dof gather and the scatter into the CSR structure, end to end, so one
assembly costs one host-to-device copy of the local dof vectors and one
device-to-host copy of the CSR values.  With a GPU build of PyMFEM (CUDA or HIP),
``HIPPYMFEM_HYPRE_DEVICE=1`` puts MFEM's matrices and hypre's solvers there too
(:func:`hippymfem.common.mfemconfig.configure_device`); with a host build they stay
on the host regardless.  Whether the kernels win depends on the card's
double-precision rate, which varies a hundredfold between cards:
``hippymfem/test/test_gpu.py`` measures the card present, and
``benchmarks/DESIGN_NOTES.md``, section 1, has reference numbers.
"""

import os
import sys


def _want_gpu():
    v = os.environ.get("HIPPYMFEM_DEVICE", "").strip().lower()
    if v == "auto":
        return _auto_gpu()
    return v in ("gpu", "cuda", "rocm", "gpu:0")


_AUTO_GPU = None


def _auto_gpu():
    """``HIPPYMFEM_DEVICE=auto``: a GPU when this process can see one, the host
    otherwise.

    Decided once and before JAX is imported (it reads its platform list once): from
    the launcher's visible-device variable when one is set, else from the vendor
    tool.  A node without a card, or a job that hid them, runs on the host without
    a word, so one environment serves a laptop and a GPU node.
    """
    global _AUTO_GPU
    if _AUTO_GPU is None:
        _AUTO_GPU = _vendor() is not None
        for name in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                     "HIP_VISIBLE_DEVICES"):
            v = os.environ.get(name)
            if v is not None:
                _AUTO_GPU = any(t.strip() not in ("", "-1") for t in v.split(","))
                break
    return _AUTO_GPU


def _platforms():
    """The backend priority list for ``JAX_PLATFORMS``.

    The host is always last, so a script (or the test suite) can compare the two in
    one process; ``set_device`` picks between them, and the accelerator is the
    default when one was asked for.

    Exactly one accelerator is named: JAX raises when a listed platform cannot be
    initialized instead of skipping it, so naming both vendors would fail on any
    machine that has only one.  ``HIPPYMFEM_DEVICE=gpu`` names no vendor, so it is
    settled from the node: the visible-device variable a launcher set, then the
    vendor's own tool, then NVIDIA when neither answers.
    """
    v = os.environ.get("HIPPYMFEM_DEVICE", "").strip().lower()
    if not _want_gpu():
        return "cpu"
    if v in ("rocm", "cuda"):
        return v + ",cpu"
    return (_vendor() or "cuda") + ",cpu"


def _vendor():
    """``"cuda"``, ``"rocm"`` or ``None``: which accelerator this node carries.

    Free when a launcher has set a visible-device variable (only one vendor's is ever
    set); otherwise it asks the vendor tools in a subprocess.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return "cuda"
    if any(os.environ.get(n) is not None
           for n in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")):
        return "rocm"
    for cmd, token, name in ((["nvidia-smi", "-L"], "GPU ", "cuda"),
                             (["rocm-smi", "--showid"], "GPU[", "rocm")):
        if _probe(cmd, token):
            return name
    return None


def _probe(cmd, token):
    """How many devices a vendor tool reports, 0 when it is not installed."""
    import subprocess

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:                                             # noqa: BLE001
        return 0
    if token == "GPU[":
        # rocm-smi prints several lines per card (name, id, revision, ...), each
        # starting GPU[k]: count the distinct k, not the lines
        import re

        return len(set(re.findall(r"GPU\[(\d+)\]", out.stdout)))
    return out.stdout.count(token)


os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", _platforms())
#: Why the platform list could not be applied, when jax was already in use.
PLATFORMS_LATE = None
if "jax" in sys.modules:
    # jax reads JAX_PLATFORMS when it is imported, so a jax imported before hippymfem
    # never saw the value above and would bring up every backend it finds at its first
    # computation (a CUDA context on every card of the node).  The config can still
    # be set until that first computation.
    try:
        import jax

        jax.config.update("jax_platforms", os.environ["JAX_PLATFORMS"])
    except Exception as e:                                        # noqa: BLE001
        PLATFORMS_LATE = str(e)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "rocm" in os.environ["JAX_PLATFORMS"] and "command_buffer" not in os.environ.get("XLA_FLAGS", ""):
    # XLA's command buffers (ROCm's counterpart of CUDA graphs) segfault inside
    # libamdhip64 once an element batch passes a block-dependent size, so no chunk size
    # is a safe cap; with them off every batch runs whole.  XLA reads the flag when its
    # client starts, which is after this point even if jax was imported first.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_gpu_enable_command_buffer=").strip()


#: The GPU ids this process could use before :func:`_pin_visible_device` narrowed the
#: list, cached on first use (before the narrowing) so the per-card sharing
#: arithmetic does not see its own effect.
_ALL_VISIBLE = None
_NODE = None


#: What a scheduler or a launcher uses to narrow a process to some of the node's cards.
#: NVIDIA reads the first, AMD's runtime the other two, and a job may be given either.
_VISIBLE_VARS = ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def _visible_var():
    """The device list a launcher set, whichever vendor's variable carries it."""
    for name in _VISIBLE_VARS:
        v = os.environ.get(name)
        if v is not None:
            return v
    return None


def _count_cards():
    """Cards on this node, asked of whichever vendor tool is installed."""
    for cmd, token in ((["nvidia-smi", "-L"], "GPU "),
                       (["rocm-smi", "--showid"], "GPU[")):
        n = _probe(cmd, token)
        if n:
            return n
    return 0


def _all_visible():
    """GPU ids this process may use, as a list, before any narrowing here."""
    global _ALL_VISIBLE
    if _ALL_VISIBLE is None:
        vis = _visible_var()
        if vis is not None:
            _ALL_VISIBLE = [t.strip() for t in vis.split(",")
                            if t.strip() not in ("", "-1")]
        else:
            _ALL_VISIBLE = [str(i) for i in range(_count_cards())]
    return _ALL_VISIBLE


def _visible_devices():
    """Number of GPUs this process can see, without importing JAX (too early)."""
    return max(len(_all_visible()), 1)


#: Launcher variables giving this process's rank and count *within its node*.
_LOCAL_RANK_VARS = ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID",
                    "MV2_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                    "PMI_LOCAL_RANK")
_LOCAL_SIZE_VARS = ("OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE",
                    "MV2_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS",
                    "PMI_LOCAL_SIZE")
#: Any of these means *some* launcher started us, even if it named no local rank.
_LAUNCHER_MARKS = ("OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS", "SLURM_JOB_ID",
                   "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "MPI_LOCALNRANKS")


def _leading_int(v):
    """First integer in a launcher value; ``SLURM_NTASKS_PER_NODE`` can say 4(x2)."""
    n = ""
    for ch in str(v).strip():
        if ch.isdigit():
            n += ch
        else:
            break
    return int(n) if n else None


def _node():
    """``(rank, size, known)`` for this process within its node, from the launcher.

    **Read from the environment, never from MPI.**  :func:`_pin_visible_device` needs
    it at import, and with a CUDA-aware MPI, ``MPI_Init`` enumerates the CUDA
    devices, after which ``CUDA_VISIBLE_DEVICES`` has no effect.  Once MPI is up,
    use :func:`hippymfem.common.mpiutil.local_rank`.

    ``known`` is false when a launcher started the process but named no node-local
    rank.  The ``(0, 1)`` returned then is a placeholder, not to be used to pick a
    device: it would put every rank on device 0.  A process started without a
    launcher gets ``(0, 1, True)``.
    """
    global _NODE
    if _NODE is None:
        rank = size = None
        for v in _LOCAL_RANK_VARS:
            if os.environ.get(v) is not None:
                rank = _leading_int(os.environ[v])
                break
        for v in _LOCAL_SIZE_VARS:
            if os.environ.get(v) is not None:
                size = _leading_int(os.environ[v])
                break
        if rank is not None and size:
            _NODE = (rank, size, True)
        elif any(os.environ.get(v) for v in _LAUNCHER_MARKS):
            _NODE = (0, 1, False)      # launched, but we cannot place ourselves
        else:
            _NODE = (0, 1, True)       # no launcher: one process, and it is us
    return _NODE


def _pin_visible_device():
    """Restrict this process to the one GPU it will use, before anything touches CUDA.

    JAX opens a context on every device it can see, not only the one it computes on,
    and those contexts hold memory the allocator's fraction does not account for.
    The visible-device variable gets one entry, chosen by node-local rank: the rule
    the element kernels and :func:`~hippymfem.common.mfemconfig.configure_device`
    use, so a rank's kernels, matrix and solves all land on one card.

    **The pin only lands if hippymfem is imported before mpi4py, MFEM and JAX.**  The
    CUDA driver latches the variable at the process's first CUDA call, which with a
    CUDA-aware MPI is ``MPI_Init``.  If one of those is already imported the pin is
    skipped and :data:`PIN_SKIPPED` says why; ``tools/mpirun_pinned.sh`` sets the
    variable at the launcher and works whatever the import order.
    ``JAX_CUDA_VISIBLE_DEVICES`` is no alternative: the ``jax-cuda12-plugin`` backend
    ignores it.  Measurements: ``benchmarks/DESIGN_NOTES.md``, section 6.

    Set ``HIPPYMFEM_PIN_GPU=0`` to leave the list alone.
    """
    global PIN_SKIPPED
    ids = _all_visible()
    rank, _, known = _node()
    if len(ids) <= 1:
        return None
    if not known:
        PIN_SKIPPED = ("a launcher started this process but named no node-local "
                       "rank, and guessing would put every rank on the same GPU")
        return None
    late = [m for m in ("mpi4py.MPI", "mfem._par.mfem", "mfem.par", "jax")
            if m in sys.modules]
    if late:
        PIN_SKIPPED = ("%s was imported before hippymfem, and the CUDA driver fixes "
                       "the visible device list at the first CUDA call; import "
                       "hippymfem first, or launch through tools/mpirun_pinned.sh"
                       % late[0])
        return None
    pick = ids[rank % len(ids)]
    # NVIDIA's variable, or AMD's: ROCR_VISIBLE_DEVICES narrows what the ROCm runtime
    # enumerates, beneath HIP and anything built on it
    os.environ["ROCR_VISIBLE_DEVICES" if _vendor() == "rocm" else "CUDA_VISIBLE_DEVICES"] = pick
    return pick


def _ranks_per_device():
    """How many ranks on this node will share each visible GPU.

    Read from the launcher, like :func:`_node`, never from MPI: the memory fraction is
    settled at import, and ``MPI_Init`` would latch the CUDA device list.
    """
    local = _node()[1]
    # tools/mpirun_pinned.sh narrows the visible list to one card per rank before
    # python starts, so four ranks on four cards look like four ranks on one; it passes
    # the node's card count in HIPPYMFEM_NODE_GPUS so the share is divided only among
    # ranks that really share a card.
    node = os.environ.get("HIPPYMFEM_NODE_GPUS", "")
    ndev = int(node) if node.isdigit() and int(node) > 0 else max(_visible_devices(), 1)
    return max(1, -(-local // ndev))            # ceil(local / ndev)


def hypre_on_device():
    """Whether hypre will share this GPU, which halves what JAX may take.

    It must be known before JAX is imported: JAX reads its memory cap once, at
    import, and never gives memory back (its allocator keeps its high-water mark), so
    an assembly that filled the card would leave hypre almost nothing.  Set
    ``HIPPYMFEM_HYPRE_DEVICE=1`` before importing hippymfem to run the solves on the
    card too (see :func:`~hippymfem.common.mfemconfig.configure_device`).
    """
    return os.environ.get("HIPPYMFEM_HYPRE_DEVICE", "").lower() not in (
        "", "0", "no", "false", "off")


def _mem_fraction():
    """Share of each GPU this process may use.

    90% of the card divided by the ranks sharing it, or 45% when hypre shares the
    card as well (its hierarchy needs room that JAX's allocator would otherwise
    reserve and hold); ``HIPPYMFEM_GPU_MEM_FRACTION`` overrides.  Why the cap follows
    the ranks and hypre: ``benchmarks/DESIGN_NOTES.md``, section 6.
    """
    env = os.environ.get("HIPPYMFEM_GPU_MEM_FRACTION")
    if env:
        return max(0.05, min(0.95, float(env)))
    base = 0.45 if hypre_on_device() else 0.90
    return round(base / _ranks_per_device(), 4)


#: Which GPU this process was pinned to, or ``None`` if it was left alone.
PINNED_DEVICE = None
#: Why the pin was skipped, or ``None`` when it was not attempted or succeeded.  Worth
#: printing from a benchmark: an unpinned rank opens a CUDA context on every card of
#: the node, memory nothing accounts for.
PIN_SKIPPED = None
#: True when this process hid every GPU from itself because it asked for none.
HIDDEN_GPUS = False

if _want_gpu():
    # The memory fraction first, while the whole device list is still visible: it
    # divides by how many ranks share a card.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(_mem_fraction()))
    if os.environ.get("HIPPYMFEM_PIN_GPU", "1").lower() not in ("0", "no", "false", "off"):
        PINNED_DEVICE = _pin_visible_device()
elif (os.environ.get("HIPPYMFEM_PIN_GPU", "1").lower() not in ("0", "no", "false", "off")
      and not any(v in os.environ for v in _VISIBLE_VARS) and not hypre_on_device()
      and not any(m in sys.modules for m in ("mpi4py.MPI", "mfem._par.mfem", "mfem.par"))):
    # A run that asked for no device holds none: the platform list keeps JAX off the
    # cards, and this hides them from everything else.  It must precede the first CUDA
    # call, and MPI_Init counts (it opens no context but fixes the visible list), so it
    # is skipped once mpi4py or MFEM is imported; tools/mpirun_pinned.sh applies the
    # same rule at the launcher.  An explicit visible-device list is honored.  AMD's
    # runtime reads ROCR_VISIBLE_DEVICES instead.
    os.environ["ROCR_VISIBLE_DEVICES" if _vendor() == "rocm" else "CUDA_VISIBLE_DEVICES"] = ""
    HIDDEN_GPUS = True


def platforms():
    """The ``JAX_PLATFORMS`` value in force."""
    return os.environ.get("JAX_PLATFORMS")


def gpu_requested():
    """True when the environment asked for GPU element kernels."""
    return _want_gpu() or "cuda" in (platforms() or "") or "gpu" in (platforms() or "")

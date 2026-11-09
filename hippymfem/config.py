# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Every knob of the library in one object, ``hippymfem.config``.

The library reads its ``HIPPYMFEM_*`` environment variables in a dozen modules.
This object names each one as an attribute with a one-line
description: ``hm.config.parmat_device`` reads the current setting,
``hm.config.parmat_device = "copy"`` changes it (through the module's own setter,
so it takes effect exactly as the environment variable would), and
``hm.config.show()`` prints them all with their values and sources, which is what a
bug report should carry.

Some settings are read by XLA or MPI before the library can act and are therefore
**import-time only**: the device, hypre on the device, JAX's memory fraction, GPU
pinning, the node's GPU count, PETSc.  They appear here read-only, with the
environment variable to set before ``import hippymfem``; ``tools/mpirun_pinned.sh``
sets the per-rank ones at the launcher.  The owning modules are imported on first
use, so importing this module loads nothing.
"""

import importlib
import os


def _module_attr(module, attr):
    def get():
        return getattr(importlib.import_module("hippymfem." + module), attr)
    get.module = "hippymfem." + module
    return get


def _module_call(module, fn):
    def get():
        return getattr(importlib.import_module("hippymfem." + module), fn)()
    get.module = "hippymfem." + module
    return get


def _module_setter(module, fn, cast=None):
    def put(value):
        getattr(importlib.import_module("hippymfem." + module), fn)(
            value if cast is None else cast(value))
    return put


def _module_assign(module, attr, cast):
    def put(value):
        setattr(importlib.import_module("hippymfem." + module), attr, cast(value))
    return put


def _class_attr(module, cls, attr):
    def get():
        return getattr(getattr(importlib.import_module("hippymfem." + module), cls), attr)
    get.module = "hippymfem." + module
    return get


def _class_assign(module, cls, attr, cast):
    def put(value):
        setattr(getattr(importlib.import_module("hippymfem." + module), cls), attr, cast(value))
    return put


def _single_amg_setter(value):
    """``HIPPYMFEM_SINGLE_AMG``: kept in the environment and parsed again into the options of
    the single-precision solves where that module is loaded (it reads them at import)."""
    import sys

    text = "" if value is None else str(value)
    mod = sys.modules.get("hippymfem.algorithms.singlesolve")
    if mod is not None:
        options = mod.parse_amg_options(text)           # (refuses an unknown name first)
        mod.AMG_OPTIONS = options
    os.environ["HIPPYMFEM_SINGLE_AMG"] = text


def _env_only(env, default=""):
    def get():
        return os.environ.get(env, default)
    return get


def _env_setter(env, cast=None):
    def put(value):
        os.environ[env] = str(value if cast is None else cast(value))
    return put


class _Knob(object):
    __slots__ = ("name", "env", "doc", "get", "put", "when")

    def __init__(self, name, env, doc, get, put=None, when="runtime"):
        self.name, self.env, self.doc, self.get, self.put, self.when = (
            name, env, doc, get, put, when)


#: the spellings of an on/off setting, in any case
FLAG_TRUE = ("1", "yes", "true", "on")
FLAG_FALSE = ("0", "no", "false", "off")


def parse_flag(value, default, name="the setting"):
    """The one reading of an on/off setting, at import and by the setters alike: ``True``
    for 1, yes, true or on, ``False`` for 0, no, false or off (any case, spaces around
    ignored), ``default`` for an empty value or ``None``.  Anything else raises, naming
    ``name``, so that a mistyped value does not quietly select one of the two."""
    if isinstance(value, bool):
        return value
    v = "" if value is None else str(value).strip().lower()
    if v == "":
        return bool(default)
    if v in FLAG_TRUE:
        return True
    if v in FLAG_FALSE:
        return False
    raise ValueError("%s must be one of %s, or empty for the default, not %r"
                     % (name, ", ".join(FLAG_TRUE + FLAG_FALSE), value))


def env_flag(name, default):
    """:func:`parse_flag` of the environment variable ``name``."""
    return parse_flag(os.environ.get(name), default, name)


def parse_tristate(value, name="the setting"):
    """``"auto"``, ``"1"`` or ``"0"`` for a setting that is on, off, or left to the library:
    auto or an empty value, a spelling of on, a spelling of off (:func:`parse_flag`'s).
    Anything else raises, naming ``name``."""
    if isinstance(value, bool):
        return "1" if value else "0"
    v = "" if value is None else str(value).strip().lower()
    if v in ("", "auto"):
        return "auto"
    if v in FLAG_TRUE:
        return "1"
    if v in FLAG_FALSE:
        return "0"
    raise ValueError("%s must be auto, one of %s, or empty for auto, not %r"
                     % (name, ", ".join(FLAG_TRUE + FLAG_FALSE), value))


def env_tristate(name):
    """:func:`parse_tristate` of the environment variable ``name``."""
    return parse_tristate(os.environ.get(name), name)


def _flag(name, default):
    """The setter's cast of an on/off knob: :func:`parse_flag`, empty for the default."""
    def cast(value):
        return parse_flag(value, default, name)
    return cast


def _tristate(name):
    """The setter's cast of an auto/on/off knob: :func:`parse_tristate`."""
    def cast(value):
        return parse_tristate(value, name)
    return cast


def _int(v):
    """A cast for a whole number that lives in the environment and is read back with
    ``int``: ``True`` is written as 1 and ``2.0`` as 2, where ``str`` alone would write
    ``"True"`` and ``"2.0"``, which ``int`` does not take and the readers refuse by the
    variable's name (before they did, the default ran in its place without a word).  An
    empty string stays: the default."""
    if isinstance(v, str) and not v.strip():
        return ""
    n = int(v)
    if not isinstance(v, str) and n != v:
        raise ValueError("must be a whole number, not %r" % (v,))
    return n


def _choice(*allowed):
    """A cast that lowercases a setting and refuses a value that is not ``allowed``."""
    def cast(value):
        v = str(value).strip().lower()
        if v not in allowed:
            raise ValueError("must be one of %s, not %r" % (", ".join(allowed), value))
        return v
    return cast


def env_choice(name, default, allowed):
    """The environment variable ``name``, lowercased, or ``default``.  A value that is
    not ``allowed`` raises, so that a mistyped route name does not select another one."""
    raw = os.environ.get(name, "")
    v = raw.strip().lower() or default
    if v not in allowed:
        raise ValueError("%s must be one of %s, not %r" % (name, ", ".join(allowed), raw))
    return v


_KNOBS = [
    # ---------------------------------------------------------- import-time only
    _Knob("device", "HIPPYMFEM_DEVICE",
          "where the element kernels run: unset/cpu, gpu, or auto (a GPU when one is visible; JAX reads the platform list once, at import)",
          _env_only("HIPPYMFEM_DEVICE", "cpu"), when="import"),
    _Knob("hypre_device", "HIPPYMFEM_HYPRE_DEVICE",
          "MFEM and hypre on the device too (CUDA build of PyMFEM; configured at import)",
          _module_call("_jaxconfig", "hypre_on_device"), when="import"),
    _Knob("auto_device", "HIPPYMFEM_AUTO_DEVICE",
          "configure MFEM's device at import when hypre_device is set (0 to call configure_device yourself)",
          lambda: env_flag("HIPPYMFEM_AUTO_DEVICE", True), when="import"),
    _Knob("gpu_mem_fraction", "HIPPYMFEM_GPU_MEM_FRACTION",
          "JAX's share of each card (default 0.90 per rank on the card, 0.45 when hypre shares it)",
          _env_only("HIPPYMFEM_GPU_MEM_FRACTION", "(default)"), when="import"),
    _Knob("pin_gpu", "HIPPYMFEM_PIN_GPU",
          "restrict each rank to one card before CUDA initializes (needs import before mpi4py/MFEM)",
          lambda: env_flag("HIPPYMFEM_PIN_GPU", True), when="import"),
    _Knob("node_gpus", "HIPPYMFEM_NODE_GPUS",
          "how many cards the node has, when the launcher hid all but one (set by mpirun_pinned.sh)",
          _env_only("HIPPYMFEM_NODE_GPUS", "(from the visible devices)"), when="import"),
    _Knob("profile", "HIPPYMFEM_PROFILE",
          "a named set of defaults that fills in what the environment leaves unset: default, "
          "gpu, gpu-large (hm.config.profiles() describes them; hm.config.use_profile after import)",
          lambda: _ACTIVE_PROFILE[0], when="import"),
    _Knob("abort_on_error", "HIPPYMFEM_ABORT_ON_ERROR",
          "on several ranks, an uncaught exception on one of them aborts every rank (default) "
          "instead of leaving the others waiting in a collective",
          lambda: env_flag("HIPPYMFEM_ABORT_ON_ERROR", True), when="import"),
    _Knob("jax_cache", "HIPPYMFEM_JAX_CACHE",
          "a directory for JAX's persistent compilation cache (off when unset); on a file "
          "system every rank can write",
          _env_only("HIPPYMFEM_JAX_CACHE", ""), when="import"),
    _Knob("petsc", "HIPPYMFEM_PETSC",
          "import petsc4py before PyMFEM so the PETSc solvers are available",
          lambda: env_flag("HIPPYMFEM_PETSC", False), when="import"),
    # ------------------------------------------------------------------ assembly
    _Knob("parmat", "HIPPYMFEM_PARMAT",
          "how the parallel matrix is built: auto (true-dof route where P is boolean), tdof, mfem",
          _module_attr("fem.parmat", "PARMAT_MODE"), _module_setter("fem.parmat", "set_parmat_mode")),
    _Knob("parmat_device", "HIPPYMFEM_PARMAT_DEVICE",
          "with hypre on a device: block (hypre's two blocks directly, default) or copy",
          _module_attr("fem.tdofassemble", "DEVICE_PARMAT"), _module_setter("fem.tdofassemble", "set_device_parmat")),
    _Knob("tdof_identity", "HIPPYMFEM_TDOF_IDENTITY",
          "true-dof route also where the prolongations are the identity (one rank): auto (with hypre on a device), 1, 0",
          _module_attr("fem.csrassemble", "TDOF_IDENTITY"),
          _module_assign("fem.csrassemble", "TDOF_IDENTITY", _tristate("HIPPYMFEM_TDOF_IDENTITY"))),
    _Knob("device_bridge", "HIPPYMFEM_DEVICE_BRIDGE",
          "with the kernels and hypre on one GPU, matrix values and assembled vectors cross between "
          "JAX's memory and MFEM's on the device instead of through the host: auto, 1, 0",
          _module_attr("common.devicebridge", "DEVICE_BRIDGE"),
          _module_setter("common.devicebridge", "set_device_bridge")),
    _Knob("device_vectors", "HIPPYMFEM_DEVICE_VECTORS",
          "with MFEM on a GPU, a vector last handed to MFEM or hypre is updated, copied and reduced "
          "on the device instead of with numpy on the host: auto, 1, 0",
          _module_attr("common.parvector", "DEVICE_VECTORS"),
          _module_setter("common.parvector", "set_device_vectors")),
    _Knob("device_pattern", "HIPPYMFEM_DEVICE_PATTERN",
          "keep a pattern's index arrays on the device (4 bytes per element-matrix entry and per "
          "nonzero) so that an assembly uploads nothing but the row pointers",
          _module_attr("fem.pattern", "DEVICE_PATTERN"), _module_setter("fem.pattern", "set_device_pattern",
                                                         _flag("HIPPYMFEM_DEVICE_PATTERN", False))),
    _Knob("fold_elimination", "HIPPYMFEM_FOLD_ELIMINATION",
          "fold the essential-dof elimination into the scatter (default) or use MFEM's calls after",
          _module_attr("fem.elimination", "FOLD_ELIMINATION"),
          _module_setter("fem.elimination", "set_fold_elimination",
                         _flag("HIPPYMFEM_FOLD_ELIMINATION", True))),
    _Knob("gpu_deterministic", "HIPPYMFEM_GPU_DETERMINISTIC",
          "device scatter in a fixed order instead of atomics (bit-identical repeats, more memory)",
          _module_attr("fem.pattern", "DETERMINISTIC"), _module_setter("fem.pattern", "set_deterministic",
                                                        _flag("HIPPYMFEM_GPU_DETERMINISTIC", False))),
    _Knob("keep_geometric_factors", "HIPPYMFEM_KEEP_GEOMETRIC_FACTORS",
          "keep MFEM's GeometricFactors alive after the batches are built",
          _module_attr("fem.elementbatch", "KEEP_GEOMETRIC_FACTORS"),
          _module_assign("fem.elementbatch", "KEEP_GEOMETRIC_FACTORS",
                         _flag("HIPPYMFEM_KEEP_GEOMETRIC_FACTORS", False))),
    _Knob("direct_transposes", "HIPPYMFEM_DIRECT_TRANSPOSES",
          "C^T and W_um^T assembled as matrices of their own from the transposed element matrices "
          "instead of formed by MFEM at the first transposed product: auto (with hypre on a "
          "device, default), 1, 0",
          _module_attr("modeling.PDEVariationalProblem", "DIRECT_TRANSPOSES"),
          _module_assign("modeling.PDEVariationalProblem", "DIRECT_TRANSPOSES",
                         _tristate("HIPPYMFEM_DIRECT_TRANSPOSES"))),
    _Knob("share_hessian", "HIPPYMFEM_SHARE_HESSIAN",
          "one differentiation pass for all blocks of a linearization point (default)",
          _module_attr("modeling.PDEVariationalProblem", "SHARE_HESSIAN_PASS"),
          _module_setter("modeling.PDEVariationalProblem", "set_share_hessian_pass",
                         _flag("HIPPYMFEM_SHARE_HESSIAN", True))),
    # ------------------------------------------------------------------- kernels
    _Knob("precision", "HIPPYMFEM_PRECISION",
          "precision of the element kernels: fp64 (default), mixed (matrices in single "
          "precision, vectors in double, solves refined: for GPUs with slow double "
          "precision) or fp32 (everything single, nothing corrected)",
          _module_attr("fem.kernel", "PRECISION"), _module_setter("fem.kernel", "set_precision")),
    _Knob("hessian", "HIPPYMFEM_HESSIAN",
          "how element Hessians are differentiated: element (default), quadrature "
          "(second derivatives at the quadrature points, faster on a GPU) or auto",
          _module_attr("fem.kernel", "HESSIAN_MODE"), _module_setter("fem.kernel", "set_hessian_mode")),
    _Knob("element_chunk", "HIPPYMFEM_ELEMENT_CHUNK",
          "elements per kernel launch; 0 (default) plans the chunk from the device's free memory",
          _module_attr("fem.kernel", "ELEMENT_CHUNK"), _module_assign("fem.kernel", "ELEMENT_CHUNK", int)),
    _Knob("host_batch", "HIPPYMFEM_HOST_BATCH",
          "elements per step of an element kernel on the host (2048, default); 0 maps the "
          "whole batch at once",
          _module_attr("fem.kernel", "HOST_BATCH"), _module_assign("fem.kernel", "HOST_BATCH", int)),
    _Knob("ad_working_set", "HIPPYMFEM_AD_WORKING_SET",
          "doubles per tangent and quadrature point the chunk planner assumes (16, with a margin)",
          _module_attr("fem.kernel", "AD_DOUBLES_PER_TANGENT_QP"),
          _module_assign("fem.kernel", "AD_DOUBLES_PER_TANGENT_QP", float)),
    _Knob("ad_vector_tangents", "HIPPYMFEM_AD_VECTOR_TANGENTS",
          "tangents a chunk of an element vector (a residual, a gradient) is planned for (8; the "
          "matrices carry one for every dof); 0 plans a vector as a matrix",
          _module_attr("fem.kernel", "AD_VECTOR_TANGENTS"),
          _module_assign("fem.kernel", "AD_VECTOR_TANGENTS", float)),
    _Knob("geometry_stream", "HIPPYMFEM_GEOMETRY_STREAM",
          "share of the device budget above which a group's geometry stays on the host and is streamed",
          _module_attr("fem.kernel", "GEOMETRY_STREAM_FRACTION"),
          _module_assign("fem.kernel", "GEOMETRY_STREAM_FRACTION", float)),
    _Knob("affine_geometry", "HIPPYMFEM_AFFINE_GEOMETRY",
          "keep the inverse Jacobians of a group once an element where every element's map is "
          "affine (a sixty-fourth of the array with 64 points, and what is left of a large "
          "group's geometry stays on the card): on by default; the element arrays change at "
          "round-off; for the batches built afterwards",
          _module_attr("fem.elementbatch", "AFFINE_GEOMETRY"),
          _module_assign("fem.elementbatch", "AFFINE_GEOMETRY", _flag("HIPPYMFEM_AFFINE_GEOMETRY", True))),
    _Knob("geometry_slice", "HIPPYMFEM_GEOMETRY_SLICE",
          "elements per slice when the quadrature geometry is built here rather than by "
          "MFEM's batched call; a positive value also forces that path",
          _module_attr("fem.elementbatch", "GEOMETRY_SLICE"),
          _module_assign("fem.elementbatch", "GEOMETRY_SLICE", int)),
    _Knob("fused_keep", "HIPPYMFEM_FUSED_KEEP",
          "keep a large assembly accumulator and zero it in place, rather than allocating one "
          "per assembly",
          _module_attr("fem.pattern", "FUSED_KEEP"),
          _module_assign("fem.pattern", "FUSED_KEEP", _flag("HIPPYMFEM_FUSED_KEEP", True))),
    _Knob("fused_keep_share", "HIPPYMFEM_FUSED_KEEP_SHARE",
          "share of JAX's arena above which an accumulator is worth keeping",
          _module_attr("fem.pattern", "FUSED_KEEP_SHARE"),
          _module_assign("fem.pattern", "FUSED_KEEP_SHARE", float)),
    _Knob("gpu_mem_reserve", "HIPPYMFEM_GPU_MEM_RESERVE",
          "GiB of device memory the chunk planner leaves untouched, for whatever else shares the card",
          _module_attr("fem.kernel", "GPU_MEM_RESERVE"), _module_assign("fem.kernel", "GPU_MEM_RESERVE", float)),
    _Knob("pattern_builder", "HIPPYMFEM_PATTERN_BUILDER",
          "how a sparsity pattern is built: auto (sort-free when numba is importable and "
          "the pattern is large), numba (always sort-free), sort (the global sort)",
          _module_attr("fem.patternbuild", "MODE"),
          _module_assign("fem.patternbuild", "MODE", _choice("auto", "numba", "sort"))),
    _Knob("pattern_builder_min", "HIPPYMFEM_PATTERN_BUILDER_MIN",
          "smallest pattern, in element-matrix entries, the sort-free builder is used for",
          _module_attr("fem.patternbuild", "MIN_ENTRIES"),
          _module_assign("fem.patternbuild", "MIN_ENTRIES", int)),
    _Knob("pattern_lean", "HIPPYMFEM_PATTERN_LEAN",
          "build a large pattern's index tables from the dof tables and row by row, and keep "
          "no map twice: half the host memory of a rank (the maps are the same; one given up "
          "is made again when asked for)",
          _module_attr("fem.pattern", "LEAN"), _module_assign("fem.pattern", "LEAN",
                                                          _flag("HIPPYMFEM_PATTERN_LEAN", True))),
    _Knob("pattern_threads", "HIPPYMFEM_PATTERN_THREADS",
          "threads for a sort-free pattern build; 0 takes this rank's share of the node",
          _module_attr("fem.patternbuild", "THREADS"),
          _module_assign("fem.patternbuild", "THREADS", int)),
    _Knob("host_mem_fraction", "HIPPYMFEM_HOST_MEM_FRACTION",
          "share of the node's available memory, split between its ranks, that the chunk "
          "planner may use on the host; 0 takes the whole element group in one launch",
          _module_attr("fem.kernel", "HOST_MEM_FRACTION"),
          _module_assign("fem.kernel", "HOST_MEM_FRACTION", float)),
    _Knob("pattern_sort", "HIPPYMFEM_PATTERN_SORT",
          "where the pattern build sorts its keys: auto, host, device",
          _module_attr("fem.devsort", "MODE"),
          _module_assign("fem.devsort", "MODE", _choice("auto", "host", "device"))),
    _Knob("pattern_sort_min", "HIPPYMFEM_PATTERN_SORT_MIN",
          "smallest key array the device sort is used for",
          _module_attr("fem.devsort", "MIN_DEVICE"), _module_assign("fem.devsort", "MIN_DEVICE", int)),
    _Knob("pattern_sort_chunk", "HIPPYMFEM_PATTERN_SORT_CHUNK",
          "keys per device sort chunk; 0 sizes it from the free memory, up to 2^26 in JAX's arena",
          _module_attr("fem.devsort", "CHUNK"), _module_assign("fem.devsort", "CHUNK", int)),
    _Knob("pattern_sort_kernel", "HIPPYMFEM_PATTERN_SORT_KERNEL",
          "the kernel of the device sort: auto (CuPy's where importable), cupy, xla",
          _module_attr("fem.devsort", "SORT_KERNEL"),
          _module_assign("fem.devsort", "SORT_KERNEL", _choice("auto", "cupy", "xla"))),
    _Knob("pattern_timing", "HIPPYMFEM_PATTERN_TIMING",
          "print what each phase of a sparsity-pattern build took, on rank 0",
          _module_attr("fem.pattern", "PATTERN_TIMING"),
          _module_assign("fem.pattern", "PATTERN_TIMING", _flag("HIPPYMFEM_PATTERN_TIMING", False))),
    # ------------------------------------------------------------------- solvers
    _Knob("hypre_spmv", "HIPPYMFEM_HYPRE_SPMV",
          "hypre's matrix-vector kernel on a GPU: auto (its own on several ranks of a CUDA build), vendor, hypre",
          _module_attr("common.mfemconfig", "HYPRE_SPMV"), _module_setter("common.mfemconfig", "set_hypre_spmv")),
    _Knob("hypre_pcg", "HIPPYMFEM_HYPRE_PCG",
          "CG solves with a hypre preconditioner in hypre's own PCG (1, the default) or in MFEM's CG (0): "
          "the same iteration; hypre's saves a matrix-vector product per V-cycle",
          _module_attr("algorithms.linSolvers", "HYPRE_PCG"), _module_setter("algorithms.linSolvers", "set_hypre_pcg")),
    _Knob("hypre_single", "HIPPYMFEM_HYPRE_SINGLE",
          "path of a single-precision build of hypre (tools/build_hypre_single.sh): the Jacobian, its "
          "BoomerAMG hierarchy and the CG solves with it then live in that library; empty: double precision",
          _module_attr("algorithms.singlesolve", "HYPRE_SINGLE"),
          _module_setter("algorithms.singlesolve", "set_hypre_single")),
    _Knob("hypre_pool", "HIPPYMFEM_HYPRE_POOL",
          "megabytes of freed device memory a recycling pool for hypre may hold per rank between "
          "BoomerAMG setups (the default pool, HIPPYMFEM_HYPRE_POOL=auto, may hold 1024 during a "
          "setup and HIPPYMFEM_HYPRE_POOL_KEEP, 512, between setups; 0 removes the pool)",
          _module_call("common.mfemconfig", "hypre_pool_megabytes"),
          _module_setter("common.mfemconfig", "set_hypre_pool_megabytes")),
    _Knob("hypre_pool_keep", "HIPPYMFEM_HYPRE_POOL_KEEP",
          "megabytes the default hypre pool keeps between BoomerAMG setups (512)",
          _env_only("HIPPYMFEM_HYPRE_POOL_KEEP", "512"), when="import"),
    _Knob("gpu_aware_mpi", "HIPPYMFEM_GPU_AWARE_MPI",
          "hypre hands MPI its buffers on the GPU (1), stages them through CPU memory (0), or hands "
          "them over only inside Krylov solves (solve, experimental); empty: as the library was built "
          "(a hypre before 2.33 does what it was built to do, whatever is set)",
          _env_only("HIPPYMFEM_GPU_AWARE_MPI", ""), when="import"),
    _Knob("amg_relax", "HIPPYMFEM_AMG_RELAX",
          "default BoomerAMG relaxation type for new solvers; -1 keeps MFEM's (16 is Chebyshev, SPD only)",
          _env_only("HIPPYMFEM_AMG_RELAX", "-1"), _env_setter("HIPPYMFEM_AMG_RELAX", _int)),
    _Knob("amg_max_levels", "HIPPYMFEM_AMG_MAX_LEVELS",
          "default BoomerAMG level cap for new solvers; -1 keeps hypre's",
          _env_only("HIPPYMFEM_AMG_MAX_LEVELS", "-1"), _env_setter("HIPPYMFEM_AMG_MAX_LEVELS", _int)),
    _Knob("amg", "HIPPYMFEM_AMG",
          "further BoomerAMG settings of new solvers by hypre's names, such as relax_wt=1.8,pmax=6 "
          "(mfemconfig.BOOMERAMG_SETTERS); empty keeps MFEM's",
          _env_only("HIPPYMFEM_AMG", ""), _env_setter("HIPPYMFEM_AMG")),
    _Knob("amg_cg", "HIPPYMFEM_AMG_CG",
          "further BoomerAMG settings of new CG solvers alone, after HIPPYMFEM_AMG: for symmetric "
          "positive definite operators, such as relax_wt=1.8 (GMRES on a convection-dominated "
          "Jacobian stalls with it); empty adds none",
          _env_only("HIPPYMFEM_AMG_CG", ""), _env_setter("HIPPYMFEM_AMG_CG")),
    _Knob("pc_reuse", "HIPPYMFEM_PC_REUSE",
          "default for reusing a solver's AMG hierarchy across operators (0: rebuild each time)",
          _env_only("HIPPYMFEM_PC_REUSE", "0"), _env_setter("HIPPYMFEM_PC_REUSE", _int)),
    _Knob("pmg", "HIPPYMFEM_PMG",
          "default of new solvers for a p-multigrid in the single-precision solves of a space of "
          "order two or more (0: BoomerAMG on the matrix itself)",
          _env_only("HIPPYMFEM_PMG", "0"), _env_setter("HIPPYMFEM_PMG", _int)),
    _Knob("pmg_h", "HIPPYMFEM_PMG_H",
          "default of new solvers for the meshes under the linear elements of a p-multigrid: a V-cycle "
          "over up to this many of the meshes that the mesh was refined from (0: BoomerAMG on the "
          "linear elements)",
          _env_only("HIPPYMFEM_PMG_H", "0"), _env_setter("HIPPYMFEM_PMG_H", _int)),
    _Knob("prior_h", "HIPPYMFEM_PRIOR_H",
          "default of priors for the meshes that the multigrid cycles of their Hessian preconditioner go "
          "over, in the single-precision hypre, where the mesh was last cut in two (0: BoomerAMG on the "
          "prior's operator)",
          _env_only("HIPPYMFEM_PRIOR_H", "0"), _env_setter("HIPPYMFEM_PRIOR_H", _int)),
    _Knob("pmg_h_host", "HIPPYMFEM_PMG_H_HOST",
          "default of new solvers for the levels of those meshes that live in host memory with hypre on "
          "a device: the ones of at most this many rows a rank, and what is under them, where the "
          "gathered last matrix has no more rows than that in all (0: none)",
          _env_only("HIPPYMFEM_PMG_H_HOST", "0"), _env_setter("HIPPYMFEM_PMG_H_HOST", _int)),
    _Knob("single_blocks", "HIPPYMFEM_SINGLE_BLOCKS",
          "the blocks of a linearization point are matrices of the single-precision hypre where it is "
          "loaded: two thirds of their memory, products to single precision (in gpu-large)",
          _module_attr("fem.tdofassemble", "SINGLE_BLOCKS"),
          _module_assign("fem.tdofassemble", "SINGLE_BLOCKS", _flag("HIPPYMFEM_SINGLE_BLOCKS", False))),
    _Knob("single_amg", "HIPPYMFEM_SINGLE_AMG",
          "BoomerAMG settings of the single-precision solves by hypre's names, such as "
          "pmax=8,interp=18 (applied after HIPPYMFEM_AMG)",
          _env_only("HIPPYMFEM_SINGLE_AMG", ""), _single_amg_setter),
    _Knob("single_row_sums", "HIPPYMFEM_SINGLE_ROW_SUMS",
          "a Jacobian of the single-precision hypre keeps the defect of its row sums for its solver "
          "to add in every product (default; the rounding of a uniform mesh's rows adds up)",
          _module_attr("fem.tdofassemble", "SINGLE_ROW_SUMS"),
          _module_assign("fem.tdofassemble", "SINGLE_ROW_SUMS", _flag("HIPPYMFEM_SINGLE_ROW_SUMS", True))),
    _Knob("single_give_up", "HIPPYMFEM_SINGLE_GIVE_UP",
          "a forward solve that single precision cannot carry: auto (default: solved again in "
          "double precision where that Jacobian fits on the device, raised for the line search "
          "to backtrack where it does not), double, or raise",
          _class_attr("modeling.PDEVariationalProblem", "PDEVariationalProblem", "SINGLE_GIVE_UP"),
          _class_assign("modeling.PDEVariationalProblem", "PDEVariationalProblem", "SINGLE_GIVE_UP",
                        _choice("auto", "double", "raise"))),
]
_BY_NAME = {k.name: k for k in _KNOBS}
_BY_ENV = {k.env: k for k in _KNOBS}

#: Named sets of defaults, by environment variable.  A profile fills in what the
#: environment leaves unset, so an explicit setting always wins: ``HIPPYMFEM_PROFILE=<name>``
#: before ``import hippymfem`` reaches every knob, the import-time ones included;
#: ``hm.config.use_profile(name)`` afterwards sets the runtime ones.  A value given as
#: ``(value, needs)`` is used only where the environment variable ``needs`` is set (mixed
#: precision only with a single-precision hypre named).  What changes the discrete problem
#: (a lumped prior mass) or belongs to a solver call (tolerances, Newton-CG parameters) is
#: not in a profile: see the GPU guide's "Recommended settings".
PROFILES = {
    "default": ("the library's own defaults", {}),
    "gpu": ("element Hessians at the quadrature points (faster on a GPU) and, where a "
            "single-precision hypre is named in HIPPYMFEM_HYPRE_SINGLE, the Jacobians and their "
            "solves in single precision with the chunk planner set for single-precision "
            "tangents",
            {"HIPPYMFEM_HESSIAN": "quadrature",
             "HIPPYMFEM_PRECISION": ("mixed", "HIPPYMFEM_HYPRE_SINGLE"),
             "HIPPYMFEM_AD_WORKING_SET": ("8", "HIPPYMFEM_HYPRE_SINGLE")}),
    "gpu-large": ("'gpu' and the solver settings of the weak-scaling series on Frontier to 4096 "
                  "GPUs: BoomerAMG with l1-Jacobi weight 1.8 and six interpolation entries a row "
                  "for the CG solvers (HIPPYMFEM_AMG_CG; GMRES keeps MFEM's settings, with "
                  "which it converges on convection-dominated problems where that weight "
                  "stalls it), in the single-precision solves the p-multigrid over up to seven meshes "
                  "of a uniformly refined mesh (BoomerAMG where there are none) with its small "
                  "levels in host memory, the prior's cycles over the meshes, and with a "
                  "single-precision hypre the blocks of a linearization point in single "
                  "precision; with a hypre "
                  "built with tools/hypre-3.2.0-seqdevice.patch add seq_threshold=30000,"
                  "redundant=1 to HIPPYMFEM_SINGLE_AMG",
                  {"HIPPYMFEM_HESSIAN": "quadrature",
                   "HIPPYMFEM_PRECISION": ("mixed", "HIPPYMFEM_HYPRE_SINGLE"),
                   "HIPPYMFEM_AD_WORKING_SET": ("8", "HIPPYMFEM_HYPRE_SINGLE"),
                   "HIPPYMFEM_AMG_CG": "relax_wt=1.8,pmax=6",
                   "HIPPYMFEM_SINGLE_AMG": "pmax=8,interp=18",
                   "HIPPYMFEM_PMG": "1",
                   "HIPPYMFEM_PMG_H": "7",
                   "HIPPYMFEM_PMG_H_HOST": "5000",
                   "HIPPYMFEM_PRIOR_H": "7",
                   "HIPPYMFEM_SINGLE_BLOCKS": ("1", "HIPPYMFEM_HYPRE_SINGLE")}),
}
#: The profile in force and the environment variables it set (``show`` marks them).
_ACTIVE_PROFILE = ["default"]
PROFILE_SET = {}
#: The environment variables of the knobs set through ``hippymfem.config`` in this process:
#: a profile applied afterwards leaves them, and ``show`` reports them as set.
EXPLICIT = set()


def _profile(name):
    key = str(name if name is not None else "default").strip().lower() or "default"
    if key not in PROFILES:
        raise ValueError("HIPPYMFEM_PROFILE must be one of %s, not %r"
                         % (", ".join(sorted(PROFILES)), name))
    return key, PROFILES[key][1]


def _condition_holds(env, need):
    """Whether the variable ``need`` that a profile's entry ``(value, need)`` asks for is
    set: not empty, and not a spelling of off (``HIPPYMFEM_HYPRE_DEVICE=0`` is hypre on
    the host, not on the device).  A path, as in ``HIPPYMFEM_HYPRE_SINGLE``, is set."""
    v = str(env.get(need, "") or "").strip().lower()
    return v != "" and v not in FLAG_FALSE


def apply_profile_env(name=None, environ=None):
    """Fill in the environment variables of a profile (default: ``HIPPYMFEM_PROFILE``) that
    ``environ`` (default ``os.environ``) leaves unset or empty, and return them.  ``import
    hippymfem`` calls it first, before any module reads its variables."""
    env = os.environ if environ is None else environ
    key, values = _profile(env.get("HIPPYMFEM_PROFILE", "") if name is None else name)
    out = {}
    for var, val in values.items():
        need = None
        if isinstance(val, tuple):
            val, need = val
        if env.get(var, "") != "" or (need and not _condition_holds(env, need)):
            continue
        env[var] = val
        out[var] = val
    if environ is None:
        _ACTIVE_PROFILE[0] = key
        PROFILE_SET.update(dict.fromkeys(out, key))
    return out





class Config(object):
    """The knobs as attributes; see the module docstring.  ``show()`` prints them."""

    def __getattr__(self, name):
        knob = _BY_NAME.get(name)
        if knob is None:
            raise AttributeError("hippymfem.config has no knob %r; see config.show()" % name)
        return knob.get()

    def __setattr__(self, name, value):
        knob = _BY_NAME.get(name)
        if knob is None:
            raise AttributeError("hippymfem.config has no knob %r; see config.show()" % name)
        if knob.put is None:
            raise AttributeError(
                "%s is read before the import (set %s in the environment, before "
                "importing hippymfem)" % (name, knob.env))
        knob.put(value)
        EXPLICIT.add(knob.env)
        PROFILE_SET.pop(knob.env, None)

    def __dir__(self):
        return sorted(_BY_NAME) + ["show", "as_dict", "knobs"]

    @property
    def knobs(self):
        """The knob descriptions, in display order."""
        return list(_KNOBS)

    def as_dict(self, load=True):
        """``{name: value}``; with ``load=False`` the knobs whose module is not
        imported yet report ``"(unloaded)"`` instead of importing it (and JAX)."""
        import sys

        out = {}
        for k in _KNOBS:
            mod = getattr(k.get, "module", None)
            if not load and mod is not None and mod not in sys.modules:
                out[k.name] = "(unloaded)"
                continue
            try:
                out[k.name] = k.get()
            except Exception as exc:                        # noqa: BLE001
                out[k.name] = "(unavailable: %s)" % exc
        return out

    def profiles(self):
        """``{name: description}`` of the named sets of defaults (:data:`PROFILES`)."""
        return {k: v[0] for k, v in PROFILES.items()}

    def use_profile(self, name):
        """Apply a profile after ``import hippymfem``: each runtime knob it names that the
        environment leaves unset (or that another profile set), and that was not set
        through ``hippymfem.config``, takes its value; an import-time knob is left as it
        is, with a warning, since it is read before.  Returns ``{knob: value}`` of what
        was set."""
        import warnings

        key, values = _profile(name)
        out = {}
        for var, val in values.items():
            need = None
            if isinstance(val, tuple):
                val, need = val
            if need and not _condition_holds(os.environ, need):
                continue
            if var in EXPLICIT or (os.environ.get(var, "") != "" and var not in PROFILE_SET):
                continue                         # set by the user: an explicit setting wins
            knob = _BY_ENV.get(var)
            if knob is None or knob.put is None or knob.when == "import":
                warnings.warn("profile %r: %s is read at import; set HIPPYMFEM_PROFILE=%s "
                              "before importing hippymfem" % (key, var, key), RuntimeWarning)
                continue
            knob.put(val)
            os.environ[var] = str(val)
            PROFILE_SET[var] = key
            out[knob.name] = val
        _ACTIVE_PROFILE[0] = key
        return out

    def show(self, file=None, load=True):
        """Print every knob: its value; where the value comes from (``set`` through this
        object, ``profile <name>``, ``env`` or ``default``); when it is read; what it
        does; and its environment variable.  Put the output in a bug report."""
        import sys

        vals = self.as_dict(load=load)
        out = file if file is not None else sys.stdout
        print("hippymfem configuration", file=out)
        for k in _KNOBS:
            src = ("set" if k.env in EXPLICIT
                   else "profile %s" % PROFILE_SET[k.env] if k.env in PROFILE_SET
                   else "env" if os.environ.get(k.env) not in (None, "") else "default")
            print("  %-24s = %-14r [%s, %s] %s" % (k.name, vals[k.name], src, k.when, k.doc),
                  file=out)
            print("  %-24s   %s" % ("", k.env), file=out)


config = Config()

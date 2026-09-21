# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Every knob of the library in one object, ``hippymfem.config``.

The library reads twenty-odd ``HIPPYMFEM_*`` environment variables in a dozen
modules.  This object names each one as a typed attribute with a one-line
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


def _module_setter(module, fn):
    def put(value):
        getattr(importlib.import_module("hippymfem." + module), fn)(value)
    return put


def _module_assign(module, attr, cast):
    def put(value):
        setattr(importlib.import_module("hippymfem." + module), attr, cast(value))
    return put


def _env_only(env, default=""):
    def get():
        return os.environ.get(env, default)
    return get


def _env_setter(env):
    def put(value):
        os.environ[env] = str(value)
    return put


class _Knob(object):
    __slots__ = ("name", "env", "doc", "get", "put", "when")

    def __init__(self, name, env, doc, get, put=None, when="runtime"):
        self.name, self.env, self.doc, self.get, self.put, self.when = (
            name, env, doc, get, put, when)


def _bool(v):
    return str(v).lower() in ("1", "yes", "true", "on") if not isinstance(v, bool) else v


_KNOBS = [
    # ---------------------------------------------------------- import-time only
    _Knob("device", "HIPPYMFEM_DEVICE",
          "where the element kernels run: unset/cpu, or gpu (JAX reads the platform list once, at import)",
          _env_only("HIPPYMFEM_DEVICE", "cpu"), when="import"),
    _Knob("hypre_device", "HIPPYMFEM_HYPRE_DEVICE",
          "MFEM and hypre on the device too (CUDA build of PyMFEM; configured at import)",
          lambda: _bool(os.environ.get("HIPPYMFEM_HYPRE_DEVICE", "")), when="import"),
    _Knob("auto_device", "HIPPYMFEM_AUTO_DEVICE",
          "configure MFEM's device at import when hypre_device is set (0 to call configure_device yourself)",
          lambda: _bool(os.environ.get("HIPPYMFEM_AUTO_DEVICE", "1")), when="import"),
    _Knob("gpu_mem_fraction", "HIPPYMFEM_GPU_MEM_FRACTION",
          "JAX's share of each card (default 0.90 per rank on the card, 0.45 when hypre shares it)",
          _env_only("HIPPYMFEM_GPU_MEM_FRACTION", "(default)"), when="import"),
    _Knob("pin_gpu", "HIPPYMFEM_PIN_GPU",
          "restrict each rank to one card before CUDA initializes (needs import before mpi4py/MFEM)",
          lambda: _bool(os.environ.get("HIPPYMFEM_PIN_GPU", "1")), when="import"),
    _Knob("node_gpus", "HIPPYMFEM_NODE_GPUS",
          "how many cards the node has, when the launcher hid all but one (set by mpirun_pinned.sh)",
          _env_only("HIPPYMFEM_NODE_GPUS", "(from the visible devices)"), when="import"),
    _Knob("petsc", "HIPPYMFEM_PETSC",
          "import petsc4py before PyMFEM so the PETSc solvers are available",
          lambda: _bool(os.environ.get("HIPPYMFEM_PETSC", "")), when="import"),
    # ------------------------------------------------------------------ assembly
    _Knob("assembly_backend", "HIPPYMFEM_ASSEMBLY",
          "csr (direct scatter, default) or integrator (MFEM's per-element callback, the reference)",
          _module_attr("fem.assemble", "_BACKEND"), _module_setter("fem.assemble", "set_assembly_backend")),
    _Knob("parmat", "HIPPYMFEM_PARMAT",
          "how the parallel matrix is built: auto (true-dof route where P is boolean), tdof, mfem, direct",
          _module_attr("fem.parmat", "PARMAT_MODE"), _module_setter("fem.parmat", "set_parmat_mode")),
    _Knob("parmat_device", "HIPPYMFEM_PARMAT_DEVICE",
          "with hypre on a device: block (hypre's two blocks directly, default) or copy",
          _module_attr("fem.tdofassemble", "DEVICE_PARMAT"), _module_setter("fem.tdofassemble", "set_device_parmat")),
    _Knob("triple", "HIPPYMFEM_TRIPLE",
          "the triple product's form where one is formed: auto (timed once), rap, split",
          _module_attr("fem.parmat", "TRIPLE_MODE"), _module_setter("fem.parmat", "set_triple_mode")),
    _Knob("fold_elimination", "HIPPYMFEM_FOLD_ELIMINATION",
          "fold the essential-dof elimination into the scatter (default) or use MFEM's calls after",
          _module_attr("fem.elimination", "FOLD_ELIMINATION"), _module_setter("fem.elimination", "set_fold_elimination")),
    _Knob("gpu_deterministic", "HIPPYMFEM_GPU_DETERMINISTIC",
          "device scatter in a fixed order instead of atomics (bit-identical repeats, more memory)",
          _module_attr("fem.pattern", "DETERMINISTIC"), _module_setter("fem.pattern", "set_deterministic")),
    _Knob("keep_geometric_factors", "HIPPYMFEM_KEEP_GEOMETRIC_FACTORS",
          "keep MFEM's GeometricFactors alive after the batches are built",
          _module_attr("fem.elementbatch", "KEEP_GEOMETRIC_FACTORS"),
          _module_assign("fem.elementbatch", "KEEP_GEOMETRIC_FACTORS", _bool)),
    _Knob("share_hessian", "HIPPYMFEM_SHARE_HESSIAN",
          "one differentiation pass for all blocks of a linearization point (default)",
          _module_attr("modeling.PDEVariationalProblem", "SHARE_HESSIAN_PASS"),
          _module_setter("modeling.PDEVariationalProblem", "set_share_hessian_pass")),
    # ------------------------------------------------------------------- kernels
    _Knob("precision", "HIPPYMFEM_PRECISION",
          "fp64 (default) or fp32 for the element kernels (a tool, not the solve path)",
          _module_attr("fem.kernel", "PRECISION"), _module_setter("fem.kernel", "set_precision")),
    _Knob("element_chunk", "HIPPYMFEM_ELEMENT_CHUNK",
          "elements per kernel launch; 0 (default) plans the chunk from the device's free memory",
          _module_attr("fem.kernel", "ELEMENT_CHUNK"), _module_assign("fem.kernel", "ELEMENT_CHUNK", int)),
    _Knob("ad_working_set", "HIPPYMFEM_AD_WORKING_SET",
          "doubles per tangent and quadrature point the chunk planner assumes (16, with a margin)",
          _module_attr("fem.kernel", "AD_DOUBLES_PER_TANGENT_QP"),
          _module_assign("fem.kernel", "AD_DOUBLES_PER_TANGENT_QP", float)),
    _Knob("chunk_plan", "HIPPYMFEM_CHUNK_PLAN",
          "how a split batch's chunk is sized: estimate (default; the per-tangent estimate) "
          "or xla (XLA's memory analysis of the compiled pass: fewer chunks, more device memory)",
          _module_attr("fem.kernel", "CHUNK_PLAN"), _module_assign("fem.kernel", "CHUNK_PLAN", str)),
    _Knob("chunk_share", "HIPPYMFEM_CHUNK_SHARE",
          "share of the free device memory a chunk sized by XLA's analysis may take (0.6)",
          _module_attr("fem.kernel", "CHUNK_SHARE"), _module_assign("fem.kernel", "CHUNK_SHARE", float)),
    _Knob("chunk_probe", "HIPPYMFEM_CHUNK_PROBE",
          "elements in the probe chunk XLA's memory analysis is compiled at (2048)",
          _module_attr("fem.kernel", "CHUNK_PROBE"), _module_assign("fem.kernel", "CHUNK_PROBE", int)),
    _Knob("geometry_stream", "HIPPYMFEM_GEOMETRY_STREAM",
          "share of the device budget above which a group's geometry stays on the host and is streamed",
          _module_attr("fem.kernel", "GEOMETRY_STREAM_FRACTION"),
          _module_assign("fem.kernel", "GEOMETRY_STREAM_FRACTION", float)),
    _Knob("geometry_slice", "HIPPYMFEM_GEOMETRY_SLICE",
          "elements per slice when the quadrature geometry is built here rather than by "
          "MFEM's batched call; a positive value also forces that path",
          _module_attr("fem.elementbatch", "GEOMETRY_SLICE"),
          _module_assign("fem.elementbatch", "GEOMETRY_SLICE", int)),
    _Knob("fused_keep", "HIPPYMFEM_FUSED_KEEP",
          "keep a large assembly accumulator and zero it in place, rather than allocating one "
          "per assembly",
          _module_attr("fem.pattern", "FUSED_KEEP"),
          _module_assign("fem.pattern", "FUSED_KEEP", _bool)),
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
          _module_attr("fem.patternbuild", "MODE"), _module_assign("fem.patternbuild", "MODE", str)),
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
          _module_attr("fem.devsort", "MODE"), _module_assign("fem.devsort", "MODE", str)),
    _Knob("pattern_sort_min", "HIPPYMFEM_PATTERN_SORT_MIN",
          "smallest key array the device sort is used for",
          _module_attr("fem.devsort", "MIN_DEVICE"), _module_assign("fem.devsort", "MIN_DEVICE", int)),
    _Knob("pattern_sort_chunk", "HIPPYMFEM_PATTERN_SORT_CHUNK",
          "keys per device sort chunk; 0 sizes it from the free memory",
          _module_attr("fem.devsort", "CHUNK"), _module_assign("fem.devsort", "CHUNK", int)),
    # ------------------------------------------------------------------- solvers
    _Knob("amg_relax", "HIPPYMFEM_AMG_RELAX",
          "default BoomerAMG relaxation type for new solvers; -1 keeps MFEM's (16 is Chebyshev, SPD only)",
          _env_only("HIPPYMFEM_AMG_RELAX", "-1"), _env_setter("HIPPYMFEM_AMG_RELAX")),
    _Knob("amg_max_levels", "HIPPYMFEM_AMG_MAX_LEVELS",
          "default BoomerAMG level cap for new solvers; -1 keeps hypre's",
          _env_only("HIPPYMFEM_AMG_MAX_LEVELS", "-1"), _env_setter("HIPPYMFEM_AMG_MAX_LEVELS")),
    _Knob("pc_reuse", "HIPPYMFEM_PC_REUSE",
          "default for reusing a solver's AMG hierarchy across operators (0: rebuild each time)",
          _env_only("HIPPYMFEM_PC_REUSE", "0"), _env_setter("HIPPYMFEM_PC_REUSE")),
]
_BY_NAME = {k.name: k for k in _KNOBS}


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

    def show(self, file=None, load=True):
        """Print every knob: value, its environment variable (marked when set), when it
        is read, and what it does.  Put the output in a bug report."""
        import sys

        vals = self.as_dict(load=load)
        out = file if file is not None else sys.stdout
        print("hippymfem configuration", file=out)
        for k in _KNOBS:
            src = "env" if os.environ.get(k.env) not in (None, "") else "default"
            print("  %-24s = %-14r [%s, %s] %s" % (k.name, vals[k.name], src, k.when, k.doc),
                  file=out)
            print("  %-24s   %s" % ("", k.env), file=out)


config = Config()

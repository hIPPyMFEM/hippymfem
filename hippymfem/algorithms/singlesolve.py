# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Linear solves in single precision: a second hypre next to the one MFEM uses.

hypre is compiled for one precision, and the hypre that MFEM and PyMFEM are built on is
a double-precision one.  A CG iteration with BoomerAMG is bound by memory traffic: a
matrix entry is twelve bytes in double precision (value and column index) and eight in
single, and the vectors and the hierarchy shrink the same way.  This module loads a
single-precision build of the same hypre (``tools/build_hypre_single.sh``) next to the
double-precision one and runs the solves of a PDE problem in it:

* the Jacobian is assembled into the single-precision library and exists there only
  (:class:`SingleParMatrix`, from the scattered values of the element kernels:
  :meth:`~hippymfem.fem.tdofassemble.TrueDofPattern.finish`), and so does its BoomerAMG
  hierarchy: two thirds of the memory of both, and no double-precision copy at any
  time;
* a :class:`~hippymfem.algorithms.linSolvers.KrylovSolver` given such a matrix runs
  hypre's PCG with BoomerAMG in that library (:class:`SingleEngine`), on vectors
  converted on the device;
* a single-precision solve reaches a relative residual near ``1e-5``
  (:data:`SINGLE_TOL_FLOOR`).  The forward and adjoint solves are therefore refined
  against residuals in double precision, which the element kernels compute
  (:meth:`PDEVariationalProblem._refined_solve`): three passes and the iterations of one
  double-precision solve in all.  The incremental solves of a Hessian action are used
  as they are, which the reorthogonalized CG of a Newton step allows
  (:mod:`~hippymfem.algorithms.cgsolverSteihaug`).

The two libraries define the same symbols.  The single-precision one is linked with
``-Bsymbolic`` and loaded with ``RTLD_LOCAL``, so that its calls stay inside it, and it
is reached through its ``ctypes`` handle only.  The few fields of hypre's structures
read here (the arrays of a matrix's two blocks, the data of a vector) are found at the
offsets that ``tools/hypre_offsets.c``, compiled against the headers of that very build,
wrote next to the library.

``HIPPYMFEM_HYPRE_SINGLE`` names the library and turns the mode on
(``hm.config.hypre_single``).  CG with BoomerAMG on a symmetric Jacobian only; anything
else keeps the double-precision solve.
"""

import ctypes
import json
import os

import numpy as np
from mpi4py import MPI

from ..common import devicebridge as bridge
from ..common import mfemconfig
from ..common.parvector import device_active

#: Path of the single-precision libHYPRE, or ``""``: the mode is off.
HYPRE_SINGLE = os.environ.get("HIPPYMFEM_HYPRE_SINGLE", "").strip()

#: The relative residual (in the norm of the preconditioner, as CG measures it) below
#: which a single-precision solve is not asked to go.  A tolerance below it is raised
#: to it: the solve would stagnate above the tolerance and run to its iteration limit.
SINGLE_TOL_FLOOR = 1e-5

#: Entries converted per pass between the two precisions (a transient of twelve bytes
#: each in the element kernels' device memory).
CHUNK = 1 << 24

HOST, DEVICE = 0, 1

#: BoomerAMG settings of the single-precision solves that replace MFEM's defaults, by
#: name: ``coarsen``, ``agg``, ``agg_interp``, ``agg_pmax``, ``relax``, ``sweeps``,
#: ``theta``, ``interp``, ``pmax``, ``levels``, ``coarse_size``, ``coarse_relax``,
#: ``keep_transpose``, ``rap2``, ``cheby_order``, ``trunc`` (the hypre call each one
#: makes is in :data:`_AMG_SETTERS`).  From ``HIPPYMFEM_SINGLE_AMG``, a comma-separated
#: list such as ``"theta=0.5,pmax=3"``; matrices set up before a change keep theirs.
_AMG_SETTERS = {
    "coarsen": ("HYPRE_BoomerAMGSetCoarsenType", int),
    "agg": ("HYPRE_BoomerAMGSetAggNumLevels", int),
    "agg_interp": ("HYPRE_BoomerAMGSetAggInterpType", int),
    "agg_pmax": ("HYPRE_BoomerAMGSetAggPMaxElmts", int),
    "relax": ("HYPRE_BoomerAMGSetRelaxType", int),
    "sweeps": ("HYPRE_BoomerAMGSetNumSweeps", int),
    "theta": ("HYPRE_BoomerAMGSetStrongThreshold", float),
    "interp": ("HYPRE_BoomerAMGSetInterpType", int),
    "pmax": ("HYPRE_BoomerAMGSetPMaxElmts", int),
    "levels": ("HYPRE_BoomerAMGSetMaxLevels", int),
    "coarse_size": ("HYPRE_BoomerAMGSetMaxCoarseSize", int),
    "coarse_relax": ("HYPRE_BoomerAMGSetCycleRelaxType", int),     # (type, 3): the coarsest level
    "keep_transpose": ("HYPRE_BoomerAMGSetKeepTranspose", int),
    "rap2": ("HYPRE_BoomerAMGSetRAP2", int),
    "cheby_order": ("HYPRE_BoomerAMGSetChebyOrder", int),
    "trunc": ("HYPRE_BoomerAMGSetTruncFactor", float),
}


def parse_amg_options(text):
    """``"theta=0.5,pmax=3"`` as a dict of the names of :data:`_AMG_SETTERS`."""
    out = {}
    for item in (text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in _AMG_SETTERS:
            raise ValueError("HIPPYMFEM_SINGLE_AMG: unknown setting %r (known: %s)"
                             % (key, ", ".join(sorted(_AMG_SETTERS))))
        out[key] = _AMG_SETTERS[key][1](val)
    return out


AMG_OPTIONS = parse_amg_options(os.environ.get("HIPPYMFEM_SINGLE_AMG", ""))

_LIB = None
_WHY = ""


def set_hypre_single(path):
    """Name the single-precision hypre library (``""`` turns the mode off); returns the
    old path.  Matrices and solvers that already exist keep what they were built with."""
    global HYPRE_SINGLE, _LIB, _WHY
    old = HYPRE_SINGLE
    path = (path or "").strip()
    if path != HYPRE_SINGLE:
        HYPRE_SINGLE, _LIB, _WHY = path, None, ""
    return old


def why_not():
    """Why :func:`library` returned ``None``, or ``""``."""
    return _WHY


def library():
    """The single-precision library, loaded and initialized at the first call, or
    ``None`` when none is named or it cannot be used (:func:`why_not`)."""
    global _LIB, _WHY
    if _LIB is not None:
        return _LIB
    if not HYPRE_SINGLE or _WHY:
        return None
    try:
        _LIB = _Library(HYPRE_SINGLE)
    except Exception as exc:                                     # noqa: BLE001
        _WHY = "%s: %s" % (type(exc).__name__, exc)
        import warnings

        warnings.warn("HIPPYMFEM_HYPRE_SINGLE=%s cannot be used (%s); the solves stay in "
                      "double precision" % (HYPRE_SINGLE, _WHY), RuntimeWarning, stacklevel=2)
        return None
    return _LIB


def action_tolerance(tol, single=1e-4):
    """A tolerance for a comparison that involves a reduced-Hessian action: ``tol`` when
    the solves run in double precision.  In a single-precision hypre the incremental
    solves stop near :data:`SINGLE_TOL_FLOOR`, so an action is exact, and symmetric, to
    about that and no such comparison holds tighter than ``single``."""
    return max(float(tol), float(single)) if (HYPRE_SINGLE and library() is not None) else float(tol)


def _comm_type():
    return ctypes.c_int if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int) else ctypes.c_void_p


class _Library:
    """A single-precision libHYPRE: its handle, the offsets of its structures and the
    settings of the double-precision one."""

    def __init__(self, path):
        path = os.path.realpath(path)
        with open(os.path.splitext(path)[0] + ".json") as f:
            self.off = json.load(f)
        o = self.off
        if (o["sizeof_real"], o["sizeof_complex"], o["sizeof_int"], o["sizeof_bigint"]) != (4, 4, 4, 4):
            raise ValueError("needs a build with 4-byte reals and integers, this one has %s"
                             % {k: v for k, v in o.items() if k.startswith("sizeof")})
        self.device = bool(device_active())
        if self.device and not bridge.available():
            raise RuntimeError("the device bridge is not available (%s)" % bridge.why_not())
        # the double-precision library must be found by path before this one is mapped
        mfemconfig._hypre_library()
        self.path = path
        self.H = H = ctypes.CDLL(path)                 # RTLD_LOCAL
        mfemconfig.SINGLE_LIBRARY_PATH = path
        p, i, r, C = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, _comm_type()
        P = ctypes.POINTER
        sig = {
            "HYPRE_Initialize": ([], i), "HYPRE_SetMemoryLocation": ([i], i),
            "HYPRE_SetExecutionPolicy": ([i], i), "HYPRE_SetSpGemmUseVendor": ([i], i),
            "HYPRE_SetUseGpuRand": ([i], i), "HYPRE_SetSpMVUseVendor": ([i], i),
            "HYPRE_ClearAllErrors": ([], i),
            "hypre_ParCSRMatrixCreate": ([C, i, i, p, p, i, i, i], p),
            "hypre_ParCSRMatrixInitialize_v2": ([p, i], i),
            "hypre_ParCSRMatrixSetNumNonzeros": ([p], i),
            "hypre_ParCSRMatrixSetDNumNonzeros": ([p], i),
            "hypre_MatvecCommPkgCreate": ([p], i),
            "hypre_ParCSRMatrixDestroy": ([p], i),
            "hypre_ParCSRMatrixMatvec": ([r, p, p, r, p], i),
            "hypre_ParCSRMatrixMatvecT": ([r, p, p, r, p], i),
            "hypre_ParVectorCreate": ([C, i, p], p),
            "hypre_ParVectorInitialize_v2": ([p, i], i),
            "hypre_ParVectorDestroy": ([p], i),
            "hypre_ParVectorSetConstantValues": ([p, r], i),
            "HYPRE_BoomerAMGCreate": ([P(p)], i), "HYPRE_BoomerAMGDestroy": ([p], i),
            "HYPRE_BoomerAMGSetCoarsenType": ([p, i], i), "HYPRE_BoomerAMGSetAggNumLevels": ([p, i], i),
            "HYPRE_BoomerAMGSetRelaxType": ([p, i], i), "HYPRE_BoomerAMGSetNumSweeps": ([p, i], i),
            "HYPRE_BoomerAMGSetStrongThreshold": ([p, r], i), "HYPRE_BoomerAMGSetInterpType": ([p, i], i),
            "HYPRE_BoomerAMGSetPMaxElmts": ([p, i], i), "HYPRE_BoomerAMGSetPrintLevel": ([p, i], i),
            "HYPRE_BoomerAMGSetMaxLevels": ([p, i], i), "HYPRE_BoomerAMGSetMaxIter": ([p, i], i),
            "HYPRE_BoomerAMGSetTol": ([p, r], i),
            "HYPRE_ParCSRPCGCreate": ([C, P(p)], i), "HYPRE_ParCSRPCGDestroy": ([p], i),
            "HYPRE_PCGSetMaxIter": ([p, i], i), "HYPRE_PCGSetTol": ([p, r], i),
            "HYPRE_PCGSetAbsoluteTol": ([p, r], i), "HYPRE_PCGSetTwoNorm": ([p, i], i),
            "HYPRE_PCGSetPrintLevel": ([p, i], i), "HYPRE_PCGSetLogging": ([p, i], i),
            "HYPRE_PCGSetPrecond": ([p, p, p, p], i),
            "HYPRE_ParCSRPCGSetup": ([p, p, p, p], i), "HYPRE_ParCSRPCGSolve": ([p, p, p, p], i),
            "HYPRE_PCGGetNumIterations": ([p, P(i)], i),
            "HYPRE_PCGGetFinalRelativeResidualNorm": ([p, P(r)], i),
            "hypre_Free": ([p, i], None),
        }
        for name, (args, res) in sig.items():
            fn = getattr(H, name)
            fn.argtypes, fn.restype = args, res
        H.HYPRE_Initialize()
        if self.device:
            # what MFEM sets in the double-precision library (Hypre::InitDevice,
            # Hypre::SetDefaultOptions) and what the library chose for its products
            H.HYPRE_SetMemoryLocation(DEVICE)
            H.HYPRE_SetExecutionPolicy(DEVICE)
            # (MFEM: hypre's own sparse product on CUDA, the vendor's on HIP)
            H.HYPRE_SetSpGemmUseVendor(0 if mfemconfig.mfem_gpu_backend() == "cuda" else 1)
            H.HYPRE_SetUseGpuRand(1)
            H.HYPRE_SetSpMVUseVendor(0 if mfemconfig.HYPRE_SPMV == "hypre" else 1)
            pool = mfemconfig.HYPRE_POOL if os.environ.get("HIPPYMFEM_SINGLE_POOL", "1") != "0" else None
            if (pool is not None and pool.installed and hasattr(H, "hypre_SetUserDeviceMalloc")
                    and hasattr(H, "hypre_SetUserDeviceMfree")):
                # one recycling pool for both libraries: a block is device memory
                H.hypre_SetUserDeviceMalloc(pool._cb_malloc)
                H.hypre_SetUserDeviceMfree(pool._cb_free)
                import atexit

                atexit.register(self._unhook)        # before the pool's own handler

    def _unhook(self):
        try:
            self.H.hypre_SetUserDeviceMalloc(None)
            self.H.hypre_SetUserDeviceMfree(None)
        except Exception:                                        # noqa: BLE001
            pass

    # ----------------------------------------------------------------- memory
    def pointer(self, base, name):
        """The pointer stored at the field ``name`` of the structure at ``base``."""
        return ctypes.c_void_p.from_address(int(base) + self.off[name]).value

    def set_pointer(self, base, name, value):
        """Store the pointer ``value`` (``None`` for NULL) at the field ``name``."""
        ctypes.c_void_p.from_address(int(base) + self.off[name]).value = value

    def integer(self, base, name):
        return ctypes.c_int.from_address(int(base) + self.off[name]).value

    def comm(self, comm):
        return _comm_type()(MPI._handleof(comm))

    def put(self, dst, array):
        """A contiguous numpy array to the library's memory at ``dst``."""
        if array.nbytes <= 0:
            return
        if self.device:
            bridge.copy_from_host(dst, array)
        else:
            ctypes.memmove(dst, array.ctypes.data, array.nbytes)

    def convert(self, dst, src, count, dtype_from, dtype_to):
        """``count`` entries at ``src`` to the other precision at ``dst``."""
        if count <= 0:
            return
        a, b = np.dtype(dtype_from), np.dtype(dtype_to)
        if not self.device:
            view = np.ctypeslib.as_array((np.ctypeslib.as_ctypes_type(a) * count).from_address(int(src)))
            out = np.ctypeslib.as_array((np.ctypeslib.as_ctypes_type(b) * count).from_address(int(dst)))
            out[:] = view
            return
        import jax.numpy as jnp

        for s in range(0, count, CHUNK):
            c = min(CHUNK, count - s)
            x = bridge.copy_to_jax(int(src) + s * a.itemsize, c, a)
            y = x.astype(jnp.dtype(b))
            bridge.copy_from_jax(int(dst) + s * b.itemsize, y)
            bridge.synchronize()                    # before ``y`` is released

    def check(self, code, what, comm=None):
        """Raise for a failed call of the library.  hypre keeps its error flag per
        process, so the verdict on a call that all ranks of ``comm`` make together is
        reduced over them: a rank that raised alone would leave the others waiting in
        their next exchange with it."""
        code = int(code)
        if comm is not None:
            code = int(comm.allreduce(abs(code), op=MPI.MAX))
        if code:
            self.H.HYPRE_ClearAllErrors()
            raise RuntimeError("the single-precision hypre: %s returned %d" % (what, code))

    @staticmethod
    def agree(comm, error):
        """Raise on every rank of ``comm`` when one of them holds ``error``, an exception
        of work each rank did on its own (``None`` on a rank where it went well)."""
        said = comm.allgather(None if error is None else str(error))
        first = next((m for m in said if m is not None), None)
        if error is not None:
            raise RuntimeError(str(error)) from error
        if first is not None:
            raise RuntimeError("%s (on another rank)" % first)


#: Every matrix of the single-precision library made from one true-dof pattern uses one
#: copy of its column indices (:class:`_SharedColumns`): the first matrix's, which the
#: pattern keeps when that matrix goes.  A new matrix then needs no upload of four bytes
#: a nonzero from the host (52 ms of the 187 ms of an assembly at 2.1 million dofs on an
#: H100), and two live matrices of one pattern (the Jacobian of the last point, still
#: held by the incremental solvers, and that of the next) hold one copy instead of two.
#: hypre neither reorders nor sorts the columns of a matrix that it solves with
#: BoomerAMG and PCG (the diagonal leads every row of the pattern already, which is all
#: its reordering does), so the copy stays as it was made.  Set
#: ``HIPPYMFEM_SINGLE_SHARE_COLUMNS=0`` to give every matrix its own.
SHARE_COLUMNS = os.environ.get("HIPPYMFEM_SINGLE_SHARE_COLUMNS", "1").lower() not in (
    "0", "no", "false", "off")


class _SharedColumns:
    """The column indices of a pattern's two blocks in the single-precision library,
    owned here and lent to every matrix made from the pattern.  Freed when the pattern
    and the last matrix that uses them are gone."""

    def __init__(self, lib, pointers, counts):
        self.lib, self.pointers, self.counts = lib, tuple(pointers), tuple(counts)

    def matches(self, lib, counts):
        return self.lib is lib and self.counts == tuple(counts)

    def __del__(self):
        try:
            location = DEVICE if self.lib.device else HOST
            for ptr in self.pointers:
                if ptr:
                    self.lib.H.hypre_Free(ctypes.c_void_p(ptr), location)
        except Exception:                                        # noqa: BLE001
            pass
        self.pointers = ()


_FINITE = []


def _all_finite(a):
    """Whether every entry of a device array is finite, as one reduction on the device
    (no array of flags is made)."""
    if not _FINITE:
        import jax
        import jax.numpy as jnp

        _FINITE.append(jax.jit(lambda v: jnp.all(jnp.isfinite(v))))
    return _FINITE[0](a)


class SingleParMatrix:
    """A parallel matrix in the single-precision library.

    Made from a true-dof pattern and the scattered values of an assembly
    (:meth:`from_pattern`): the two blocks of hypre's format, with the values rounded
    to single precision.  It answers what a PDE problem asks of its Jacobian:
    ``Height``, ``Width``, ``Mult`` and ``MultTranspose`` on the library's
    (double-precision) vectors, which are converted on the way in and out.
    """

    def __init__(self):
        raise TypeError("use SingleParMatrix.from_pattern")

    @classmethod
    def from_pattern(cls, tp, acc, lib=None):
        """The matrix of the true-dof pattern ``tp`` with the values ``acc``: the entries
        of the diagonal block, then those of the off-diagonal one, in the pattern's
        order (what :meth:`TrueDofPattern.finish` holds after its exchange), as a JAX
        array on the device or a numpy array."""
        lib = lib if lib is not None else library()
        if lib is None:
            raise RuntimeError("no single-precision hypre (%s)"
                               % (why_not() or "HIPPYMFEM_HYPRE_SINGLE is not set"))
        self = object.__new__(cls)
        self.lib, H = lib, lib.H
        self.comm = tp.comm
        self.par = None
        self._vecs = None
        self._columns = None
        n, nd, no = int(tp.ntd), int(tp.nnz_diag), int(tp.nnz_t - tp.nnz_diag)
        n_offd = int(tp.n_offd)
        self._height, self._width = n, int(tp.c1 - tp.c0)
        self.global_rows, self.global_cols = int(tp.gnr), int(tp.gnc)
        self.row_starts = np.array([tp.t0, tp.t1], dtype=np.int32)
        self.col_starts = np.array([tp.c0, tp.c1], dtype=np.int32)
        self.nnz = nd + no
        par = H.hypre_ParCSRMatrixCreate(lib.comm(self.comm), self.global_rows, self.global_cols,
                                         self.row_starts.ctypes.data, self.col_starts.ctypes.data,
                                         n_offd, nd, no)
        self.par = ctypes.c_void_p(par) if par else None
        # What follows is each rank's own work, up to the exchanges at the end.  A rank
        # that fails keeps its exception until all have finished, and then all raise.
        failed, vals = None, None
        try:
            if not par:
                raise RuntimeError("the single-precision hypre could not create a matrix")
            shared = getattr(tp, "_single_columns", None) if SHARE_COLUMNS else None
            if shared is not None and shared.matches(lib, (nd, no)):
                # the column indices of an earlier matrix of this pattern: set before
                # the arrays are allocated, so that hypre allocates none for them
                for name, ptr in zip(("par_diag", "par_offd"), shared.pointers):
                    if ptr:
                        lib.set_pointer(lib.pointer(par, name), "csr_j", ptr)
                self._columns = shared
            lib.check(H.hypre_ParCSRMatrixInitialize_v2(self.par, DEVICE if lib.device else HOST),
                      "ParCSRMatrixInitialize")
            on_host = isinstance(acc, np.ndarray)
            if on_host:
                with np.errstate(over="ignore", invalid="ignore"):
                    vals = np.ascontiguousarray(acc[:nd + no], dtype=np.float32)
                finite = bool(np.isfinite(vals).all())
            else:
                import jax.numpy as jnp

                # accumulated in single precision (tdofassemble.SINGLE_ACCUMULATE): the
                # accumulator itself holds the values, and nothing is copied
                vals = acc if acc.dtype == jnp.float32 else acc[:nd + no].astype(jnp.float32)
                finite = bool(_all_finite(vals))
            # An entry beyond the range of single precision (a coefficient spanning
            # dozens of decades, as a line search may propose) becomes infinite, and
            # hypre does not return from a setup with such a matrix.  A RuntimeError,
            # which a line search takes for a failed solve and answers by backtracking.
            if not finite:
                raise RuntimeError("the matrix has entries beyond the range of single "
                                   "precision (or not finite)")
            for name, block, I, J, start, count in (("par_diag", "diag", tp.I_diag, tp.J_diag, 0, nd),
                                                    ("par_offd", "offd", tp.I_offd, tp.J_offd, nd, no)):
                csr = lib.pointer(par, name)
                if (lib.integer(csr, "csr_num_rows") != n
                        or lib.integer(csr, "csr_num_nonzeros") != count):
                    raise RuntimeError("the offsets file does not describe this library")
                lib.put(lib.pointer(csr, "csr_i"), np.ascontiguousarray(I, dtype=np.int32))
                if not count:
                    continue
                if self._columns is None:
                    # (a device copy the pattern already keeps is used, none is made for
                    # this: the matrix's own copy is the one kept from now on)
                    dev_J = tp.device_columns(block, create=False) if lib.device else None
                    if dev_J is not None:
                        bridge.copy_from_jax(lib.pointer(csr, "csr_j"), dev_J)
                    else:
                        lib.put(lib.pointer(csr, "csr_j"), np.ascontiguousarray(J[:count], dtype=np.int32))
                if on_host:
                    lib.put(lib.pointer(csr, "csr_data"), vals[start:start + count])
                else:
                    bridge.copy_from_jax(lib.pointer(csr, "csr_data"), vals, start, count)
            if n_offd:
                cm = np.ascontiguousarray(np.asarray(tp.ia_cmap.GetDataArray())[:n_offd], dtype=np.int32)
                ctypes.memmove(lib.pointer(par, "par_col_map_offd"), cm.ctypes.data, cm.nbytes)
            if lib.device:
                bridge.synchronize()               # before ``vals`` is released
            if self._columns is None and SHARE_COLUMNS:
                # the first matrix of the pattern: its column indices become the
                # pattern's, lent to it and to every later matrix
                self._columns = tp._single_columns = _SharedColumns(
                    lib, (lib.pointer(lib.pointer(par, "par_diag"), "csr_j") if nd else None,
                          lib.pointer(lib.pointer(par, "par_offd"), "csr_j") if no else None),
                    (nd, no))
        except Exception as e:                                   # noqa: BLE001
            failed = e
        del vals
        try:
            lib.agree(self.comm, failed)
            lib.check(H.hypre_ParCSRMatrixSetNumNonzeros(self.par), "ParCSRMatrixSetNumNonzeros",
                      self.comm)
            H.hypre_ParCSRMatrixSetDNumNonzeros(self.par)
            lib.check(H.hypre_MatvecCommPkgCreate(self.par), "MatvecCommPkgCreate", self.comm)
        except Exception:
            self.destroy()
            raise
        return self

    # ------------------------------------------------------------- interface
    def Height(self):
        return self._height

    def Width(self):
        return self._width

    def GetComm(self):
        return self.comm

    def GetGlobalNumRows(self):
        return self.global_rows

    def NNZ(self):
        return self.nnz

    def vector(self, columns=False):
        """A vector of the single-precision library on this matrix's rows (or columns)."""
        return SingleVector(self.lib, self.comm, self.global_cols if columns else self.global_rows,
                            self.col_starts if columns else self.row_starts)

    def _work(self):
        if self._vecs is None:
            self._vecs = (self.vector(columns=True), self.vector())
        return self._vecs

    def Mult(self, x, y):
        """``y = A x`` on two of MFEM's vectors, the product in single precision."""
        vx, vy = self._work()
        vx.set(x)
        self.lib.check(self.lib.H.hypre_ParCSRMatrixMatvec(1.0, self.par, vx.par, 0.0, vy.par),
                       "Matvec", self.comm)
        vy.get(y)

    def MultTranspose(self, x, y):
        vx, vy = self._work()
        vy.set(x)
        self.lib.check(self.lib.H.hypre_ParCSRMatrixMatvecT(1.0, self.par, vy.par, 0.0, vx.par),
                       "MatvecT", self.comm)
        vx.get(y)

    def destroy(self):
        if self._vecs is not None:
            for v in self._vecs:
                v.destroy()
            self._vecs = None
        if self.par is not None:
            shared = getattr(self, "_columns", None)
            if shared is not None:
                # the borrowed column indices are not hypre's to free
                for name, ptr in zip(("par_diag", "par_offd"), shared.pointers):
                    if ptr:
                        csr = self.lib.pointer(self.par.value, name)
                        if self.lib.pointer(csr, "csr_j") == ptr:
                            self.lib.set_pointer(csr, "csr_j", None)
                self._columns = None
            self.lib.H.hypre_ParCSRMatrixDestroy(self.par)
            self.par = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:                                        # noqa: BLE001
            pass


class SingleVector:
    """A vector of the single-precision library, filled from and read into MFEM's."""

    def __init__(self, lib, comm, global_size, starts):
        self.lib = lib
        starts = np.ascontiguousarray(starts, dtype=np.int32)
        self.n = int(starts[1] - starts[0])
        par = lib.H.hypre_ParVectorCreate(lib.comm(comm), int(global_size), starts.ctypes.data)
        if not par:
            raise RuntimeError("the single-precision hypre could not create a vector")
        self.par = ctypes.c_void_p(par)
        lib.check(lib.H.hypre_ParVectorInitialize_v2(self.par, DEVICE if lib.device else HOST),
                  "ParVectorInitialize")
        self.data = lib.pointer(lib.pointer(par, "parvec_local"), "vec_data")

    def set(self, x):
        """From one of MFEM's vectors (double precision)."""
        self.lib.convert(self.data, bridge.address(x.Read(self.lib.device)), self.n, np.float64, np.float32)

    def get(self, y):
        """Into one of MFEM's vectors."""
        self.lib.convert(bridge.address(y.Write(self.lib.device)), self.data, self.n, np.float32, np.float64)

    def zero(self):
        self.lib.H.hypre_ParVectorSetConstantValues(self.par, 0.0)

    def destroy(self):
        if self.par is not None:
            self.lib.H.hypre_ParVectorDestroy(self.par)
            self.par = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:                                        # noqa: BLE001
            pass


class SingleEngine:
    """hypre's PCG with BoomerAMG on a :class:`SingleParMatrix`, set up once and shared
    by the solvers of that matrix (the forward and the two incremental ones)."""

    def __init__(self, S, parameters):
        self.matrix, self.lib = S, S.lib
        H, lib, p = S.lib.H, S.lib, parameters
        self.amg = self.pcg = None
        self.b, self.x = S.vector(), S.vector(columns=True)
        amg = ctypes.c_void_p()
        H.HYPRE_BoomerAMGCreate(ctypes.byref(amg))
        self.amg = amg
        # MFEM's defaults for the device and for the host, then the solver's own, then
        # AMG_OPTIONS.  On a device HIPPYMFEM_SINGLE_AMG="relax=7,pmax=6" (Jacobi
        # relaxation instead of l1-Jacobi, which divides by the sum of the magnitudes of
        # a row, a few times the diagonal for quadratic elements, and six interpolation
        # entries a row instead of four) took the solves of the model problem at 64^3
        # (Blackwell instance, to 1e-5) from 11 to 6 iterations and from 76 to 48 ms at
        # the MAP point, the prior mean and the truth alike, and Newton-CG to 1e-6 from
        # 38.5 to 29.0 s (H100: 19.8 to 15.7 s) with the same Newton and CG counts.  Opt-in
        # until it is checked on more problems: plain Jacobi needs no diagonal dominance
        # to be defined, but it is not guaranteed to smooth without it.
        gpu = lib.device
        relax = int(p["amg_relax_type"])
        levels = int(p["amg_max_levels"])
        agg = int(p["amg_agg_levels"])
        theta = float(p["amg_strength_threshold"])
        H.HYPRE_BoomerAMGSetCoarsenType(amg, 8 if gpu else 10)
        H.HYPRE_BoomerAMGSetAggNumLevels(amg, agg if agg >= 0 else (0 if gpu else 1))
        H.HYPRE_BoomerAMGSetRelaxType(amg, relax if relax >= 0 else (18 if gpu else 8))
        H.HYPRE_BoomerAMGSetNumSweeps(amg, 1)
        H.HYPRE_BoomerAMGSetStrongThreshold(amg, theta if theta >= 0.0 else 0.25)
        H.HYPRE_BoomerAMGSetInterpType(amg, 6)
        H.HYPRE_BoomerAMGSetPMaxElmts(amg, 4)
        H.HYPRE_BoomerAMGSetPrintLevel(amg, 0)
        H.HYPRE_BoomerAMGSetMaxLevels(amg, levels if levels > 0 else 25)
        H.HYPRE_BoomerAMGSetMaxIter(amg, 1)
        H.HYPRE_BoomerAMGSetTol(amg, 0.0)
        for key, val in AMG_OPTIONS.items():
            name, kind = _AMG_SETTERS[key]
            fn = getattr(H, name)
            if kind is float:
                fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_float], ctypes.c_int
            elif key == "coarse_relax":
                fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
            else:
                fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
            rc = fn(amg, val, 3) if key == "coarse_relax" else fn(amg, val)
            if rc:
                H.HYPRE_ClearAllErrors()
                raise ValueError("the single-precision hypre refused %s=%r (%s returned %d)"
                                 % (key, val, name, rc))
        pcg = ctypes.c_void_p()
        H.HYPRE_ParCSRPCGCreate(lib.comm(S.comm), ctypes.byref(pcg))
        self.pcg = pcg
        H.HYPRE_PCGSetTwoNorm(pcg, 0)
        H.HYPRE_PCGSetPrintLevel(pcg, 0)
        H.HYPRE_PCGSetLogging(pcg, 0)
        H.HYPRE_PCGSetPrecond(pcg, ctypes.cast(H.HYPRE_BoomerAMGSolve, ctypes.c_void_p),
                              ctypes.cast(H.HYPRE_BoomerAMGSetup, ctypes.c_void_p), amg)
        mfemconfig.hypre_pool_open()
        try:
            # the setup of one rank can fail alone (its part of a hierarchy overflows)
            lib.check(H.HYPRE_ParCSRPCGSetup(pcg, S.par, self.b.par, self.x.par), "PCGSetup",
                      S.comm)
            if lib.device:
                bridge.synchronize()
        finally:
            mfemconfig.hypre_pool_close()

    def solve(self, x, b, rel_tol, abs_tol, max_iter):
        """``x = A^{-1} b`` for two of MFEM's vectors, from a zero guess; returns the
        iterations and the final relative residual in the preconditioner's norm."""
        H, lib = self.lib.H, self.lib
        H.HYPRE_PCGSetTol(self.pcg, max(float(rel_tol), SINGLE_TOL_FLOOR))
        H.HYPRE_PCGSetAbsoluteTol(self.pcg, float(abs_tol))
        H.HYPRE_PCGSetMaxIter(self.pcg, int(max_iter))
        self.b.set(b)
        self.x.zero()
        rc = H.HYPRE_ParCSRPCGSolve(self.pcg, self.matrix.par, self.b.par, self.x.par)
        if rc:
            H.HYPRE_ClearAllErrors()                # not converged sets an error
        self.x.get(x)
        its, norm = ctypes.c_int(), ctypes.c_float()
        H.HYPRE_PCGGetNumIterations(self.pcg, ctypes.byref(its))
        H.HYPRE_PCGGetFinalRelativeResidualNorm(self.pcg, ctypes.byref(norm))
        return int(its.value), float(norm.value)

    def destroy(self):
        H = self.lib.H
        if self.pcg is not None:
            H.HYPRE_ParCSRPCGDestroy(self.pcg)
            self.pcg = None
        if self.amg is not None:
            H.HYPRE_BoomerAMGDestroy(self.amg)
            self.amg = None
        for v in (getattr(self, "b", None), getattr(self, "x", None)):
            if v is not None:
                v.destroy()
        self.b = self.x = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:                                        # noqa: BLE001
            pass

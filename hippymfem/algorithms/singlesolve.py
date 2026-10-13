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
else keeps the double-precision solve.  On a GPU it runs with CUDA and with HIP builds
(NVIDIA and AMD cards); on any other device the library is refused before it is loaded
(:func:`unsupported`), with a warning, and the solves stay in double precision.  The
library has four-byte values and four-byte indices within a rank, and global indices
of either width: one built ``--enable-mixedint`` holds a Jacobian of more than
2 147 483 647 rows, and one with 32-bit global indices refuses such a matrix.
"""

import ctypes
import json
import os
import weakref

import numpy as np
from mpi4py import MPI

from ..common import devicebridge as bridge
from ..common import mfemconfig
from ..common.parvector import allreduce as _allreduce, device_active

#: Path of the single-precision libHYPRE, or ``""``: the mode is off.
HYPRE_SINGLE = os.environ.get("HIPPYMFEM_HYPRE_SINGLE", "").strip()

#: The relative residual (in the norm of the preconditioner, as CG measures it) below
#: which a single-precision solve is not asked to go.  A tolerance below it is raised
#: to it: the solve would stagnate above the tolerance and run to its iteration limit.
SINGLE_TOL_FLOOR = 1e-5

#: The bit of hypre's error flag for an iteration that ended without meeting its tolerance.
HYPRE_ERROR_CONV = 256

#: The functions of hypre's PCG for a parallel matrix, in the order of
#: ``hypre_PCGFunctionsCreate`` (``HYPRE_ParCSRPCGCreate`` passes these).
_PCG_TABLE = ("hypre_ParKrylovCAlloc", "hypre_ParKrylovFree", "hypre_ParKrylovCommInfo",
              "hypre_ParKrylovCreateVector", "hypre_ParKrylovDestroyVector",
              "hypre_ParKrylovMatvecCreate", "hypre_ParKrylovMatvec",
              "hypre_ParKrylovMatvecDestroy", "hypre_ParKrylovInnerProd",
              "hypre_ParKrylovCopyVector", "hypre_ParKrylovClearVector",
              "hypre_ParKrylovScaleVector", "hypre_ParKrylovAxpy",
              "hypre_ParKrylovIdentitySetup", "hypre_ParKrylovIdentity")

#: Entries converted per pass between the two precisions (a transient of twelve bytes
#: each in the element kernels' device memory).
CHUNK = 1 << 24

HOST, DEVICE = 0, 1

#: BoomerAMG settings of the single-precision solves alone, by the names of
#: :data:`hippymfem.common.mfemconfig.BOOMERAMG_SETTERS` (``coarsen``, ``agg``, ``relax``,
#: ``relax_wt``, ``sweeps``, ``theta``, ``interp``, ``pmax``, ``levels``, ``coarse_size``,
#: ``coarse_relax``, ``keep_transpose``, ``rap2``, ``cheby_order``, ``trunc`` and the
#: others of that table, with the hypre call each one makes).  From
#: ``HIPPYMFEM_SINGLE_AMG``, a comma-separated list such as ``"theta=0.5,pmax=3"``;
#: matrices set up before a change keep theirs.  They come after the settings that a
#: solver has for both precisions (its parameters and its ``amg_options``, which
#: ``HIPPYMFEM_AMG`` sets): a hierarchy in single precision is otherwise the one the same
#: solver would build in double.
_AMG_SETTERS = mfemconfig.BOOMERAMG_SETTERS


def parse_amg_options(text):
    """``"theta=0.5,pmax=3"`` as a dict of the names of :data:`_AMG_SETTERS`."""
    return mfemconfig.parse_boomeramg_options(text, "HIPPYMFEM_SINGLE_AMG")


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


def unsupported():
    """Why this build cannot run the single-precision solves at all, or ``""``.

    They are built (``tools/build_hypre_single.sh``) and tested with CUDA and HIP builds
    of MFEM and hypre and with host builds.  On any other device the code of the library
    is untried, and :func:`library` refuses it before loading it."""
    backend = mfemconfig.mfem_gpu_backend()
    if device_active() and backend not in (None, "cuda", "hip"):
        return ("the single-precision solves run with CUDA, HIP and host builds only, and "
                "this MFEM and hypre run on %s" % backend)
    return ""


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
        why = unsupported()
        if why:
            raise RuntimeError(why)
        path = os.path.realpath(path)
        with open(os.path.splitext(path)[0] + ".json") as f:
            self.off = json.load(f)
        o = self.off
        if ((o["sizeof_real"], o["sizeof_complex"], o["sizeof_int"]) != (4, 4, 4)
                or o["sizeof_bigint"] not in (4, 8)):
            raise ValueError("needs a build with 4-byte reals and 4-byte indices within a rank, "
                             "this one has %s" % {k: v for k, v in o.items() if k.startswith("sizeof")})
        #: the type of a global index (``HYPRE_BigInt``): 64 bits in a build
        #: ``--enable-mixedint``, which numbers spaces of more than 2 147 483 647 unknowns
        self.bigint = np.dtype(np.int64 if o["sizeof_bigint"] == 8 else np.int32)
        self.device = bool(device_active())
        if self.device and not bridge.available():
            raise RuntimeError("the device bridge is not available (%s)" % bridge.why_not())
        # the double-precision library must be found by path before this one is mapped
        mfemconfig._hypre_library()
        self.path = path
        self.H = H = ctypes.CDLL(path)                 # RTLD_LOCAL
        mfemconfig.SINGLE_LIBRARY_PATH = path
        p, i, r, C = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, _comm_type()
        B = ctypes.c_longlong if self.bigint.itemsize == 8 else ctypes.c_int
        P = ctypes.POINTER
        sig = {
            "HYPRE_Initialize": ([], i), "HYPRE_SetMemoryLocation": ([i], i),
            "HYPRE_SetExecutionPolicy": ([i], i), "HYPRE_SetSpGemmUseVendor": ([i], i),
            "HYPRE_SetUseGpuRand": ([i], i), "HYPRE_SetSpMVUseVendor": ([i], i),
            "HYPRE_ClearAllErrors": ([], i),
            "hypre_ParCSRMatrixCreate": ([C, B, B, p, p, i, i, i], p),
            "hypre_ParCSRMatrixInitialize_v2": ([p, i], i),
            "hypre_ParCSRMatrixSetNumNonzeros": ([p], i),
            "hypre_ParCSRMatrixSetDNumNonzeros": ([p], i),
            "hypre_MatvecCommPkgCreate": ([p], i),
            "hypre_ParCSRMatrixDestroy": ([p], i),
            "hypre_ParCSRMatrixMatvec": ([r, p, p, r, p], i),
            "hypre_ParCSRMatrixMatvecT": ([r, p, p, r, p], i),
            "hypre_ParVectorCreate": ([C, B, p], p),
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
            "HYPRE_PCGGetConverged": ([p, P(i)], i),
            "HYPRE_PCGGetFinalRelativeResidualNorm": ([p, P(r)], i),
            "hypre_Free": ([p, i], None),
        }
        for name, (args, res) in sig.items():
            fn = getattr(H, name)
            fn.argtypes, fn.restype = args, res
        #: forms and keeps the transposes of a matrix's two blocks (hypre on a device has
        #: no transposed product of its own: without them it forms one at every call)
        self.local_transpose = getattr(H, "hypre_ParCSRMatrixLocalTranspose", None)
        if self.local_transpose is not None:
            self.local_transpose.argtypes, self.local_transpose.restype = [p], i
        # What the p-multigrid calls beside these (PTransfer, PMGEngine).  The Galerkin
        # product is hypre's own of a coarsening on a device, hypre_ParCSRMatrixRAPKT with
        # the arguments it has had since hypre 3.2, and before that the two products MFEM
        # forms it with.  A library without one of the calls keeps BoomerAMG (no_pmg).
        self.version = tuple(int(t) for t in "".join(
            c if c.isdigit() else " " for c in str(o.get("version", "0"))).split()[:3])
        pmg = {
            "hypre_ParCSRMatrixMatvecOutOfPlace": ([r, p, p, r, p, p], i),
            "hypre_ParCSRMatrixTranspose": ([p, P(p), i], i),
            "hypre_ParCSRMatMat": ([p, p], p),
            "hypre_ParCSRTMatMatKT": ([p, p, i], p),
            "hypre_CSRMatrixReorder": ([p], i),
            "hypre_ParCSRComputeL1Norms": ([p, i, p, P(p)], i),
            "hypre_CSRMatrixExtractDiagonal": ([p, p, i], None),
            "hypre_ParVectorScale": ([r, p], i),
            "hypre_ParVectorCopy": ([p, p], i),
            "hypre_ParVectorAxpy": ([r, p, p], i),
            "HYPRE_BoomerAMGSetup": ([p, p, p, p], i),
            "HYPRE_BoomerAMGSolve": ([p, p, p, p], i),
        }
        missing = [name for name in pmg if not hasattr(H, name)]
        for name, (args, res) in pmg.items():
            if name not in missing:
                fn = getattr(H, name)
                fn.argtypes, fn.restype = args, res
        #: ``y += x / b`` entry by entry (hypre's name for it changed)
        self.divide_add = next((getattr(H, name) for name in (
            "hypre_ParVectorPointwiseDivpy", "hypre_ParVectorElmdivpy") if hasattr(H, name)), None)
        if self.divide_add is None:
            missing.append("hypre_ParVectorPointwiseDivpy")
        else:
            self.divide_add.argtypes, self.divide_add.restype = [p, p, p], i
        #: a vector to zero, marked as zero for a relaxation that then saves a product
        self.set_zero = getattr(H, "hypre_ParVectorSetZeros", None)
        if self.set_zero is not None:
            self.set_zero.argtypes, self.set_zero.restype = [p], i
        #: whether levels of a p-multigrid can live in host memory beside the device
        #: (:data:`PMG_H_HOST`): a copy of a matrix in the other memory, and a copy of a
        #: vector's entries between the two
        self.host_levels = bool(self.device) and all(
            hasattr(H, name) for name in ("hypre_ParCSRMatrixClone_v2", "hypre_Memcpy"))
        if self.host_levels:
            H.hypre_ParCSRMatrixClone_v2.argtypes, H.hypre_ParCSRMatrixClone_v2.restype = [p, i, i], p
            H.hypre_Memcpy.argtypes, H.hypre_Memcpy.restype = [p, p, ctypes.c_size_t, i, i], None
        self.rapkt = None
        if self.version >= (3, 2) and hasattr(H, "hypre_ParCSRMatrixRAPKT"):
            self.rapkt = H.hypre_ParCSRMatrixRAPKT
            self.rapkt.argtypes, self.rapkt.restype = [p, p, p, i, i], p
        # What a PCG whose product adds the defect of the matrix's row sums is made of
        # (pcg, SingleParMatrix.row_defect): hypre's function table for a parallel matrix
        # with one entry replaced.  A library without one of the names keeps hypre's PCG.
        self.own_product = (self.divide_add is not None and hasattr(H, "hypre_ParVectorAxpy")
                            and all(hasattr(H, name) for name in _PCG_TABLE + (
                                "hypre_PCGFunctionsCreate", "hypre_PCGCreate")))
        if self.own_product:
            H.hypre_PCGFunctionsCreate.argtypes = [p] * len(_PCG_TABLE)
            H.hypre_PCGFunctionsCreate.restype = p
            H.hypre_PCGCreate.argtypes, H.hypre_PCGCreate.restype = [p], p
            H.hypre_ParVectorAxpy.argtypes, H.hypre_ParVectorAxpy.restype = [r, p, p], i
        #: why this library cannot run the p-multigrid, or ``""``
        self.no_pmg = ("hypre %s lacks %s" % (o.get("version", "?"), ", ".join(missing))
                       if missing else "")
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
        elif hasattr(H, "HYPRE_SetMemoryLocation"):
            # a GPU build under MFEM on the host: what MFEM tells the double-precision
            # library then (Hypre::InitDevice), where such a build would otherwise take
            # the device for everything it allocates itself (a HIP build of hypre 3.2.0
            # faults in its first setup on a host matrix without this)
            H.HYPRE_SetMemoryLocation(HOST)
            H.HYPRE_SetExecutionPolicy(HOST)

    def _unhook(self):
        try:
            self.H.hypre_SetUserDeviceMalloc(None)
            self.H.hypre_SetUserDeviceMfree(None)
        except Exception:                                        # noqa: BLE001
            pass

    # ------------------------------------------------------------ p-multigrid
    def galerkin(self, A, P, products=None, keep=False, host=False):
        """``P^T A P`` as a new matrix of the library (a pointer; ``None`` where the
        product failed), the diagonal first in its rows.  ``products=True`` takes the two
        products also where hypre's own routine for the three is at hand.  ``keep=True``
        takes them as well and returns ``(P^T A P, A P)``, both the caller's to destroy.
        ``host=True``: of two matrices in host memory with the library on a device (the
        caller has called :meth:`to_host`)."""
        H = self.H
        Q = None
        if self.rapkt is not None and not products and not keep:
            C = self.rapkt(P, A, P, 0, 1)
        else:
            Q = H.hypre_ParCSRMatMat(A, P)
            if not Q:
                return (None, None) if keep else None
            try:
                C = H.hypre_ParCSRTMatMatKT(P, Q, 0)
            finally:
                if not keep:
                    H.hypre_ParCSRMatrixDestroy(Q)
        if C and (host or not self.device):
            # (on a device hypre's products move the diagonal to the front themselves;
            # on the host they leave the rows as the products made them, and a hierarchy
            # reads the first entry of a row as its diagonal)
            H.hypre_CSRMatrixReorder(self.pointer(C, "par_diag"))
        return (C, Q) if keep else C

    # ------------------------------------------- host memory beside a device
    def to_host(self):
        """From here on hypre makes what it makes in host memory and runs it there: for
        the levels of a p-multigrid that live in host memory with the library on a
        device (:data:`PMG_H_HOST`).  hypre runs an operation where its operands are;
        what an operation allocates itself (a product, a hierarchy, the buffers of an
        exchange) goes where these two settings say.  :meth:`to_device` ends it.  The two
        nest: an object that is taken down while another runs in host memory (a collection
        of Python inside a cycle) ends its own stretch and not the other's."""
        self._host_depth = getattr(self, "_host_depth", 0) + 1
        if self._host_depth == 1:
            self.H.HYPRE_SetMemoryLocation(HOST)
            self.H.HYPRE_SetExecutionPolicy(HOST)

    def to_device(self):
        self._host_depth = max(getattr(self, "_host_depth", 1) - 1, 0)
        if self._host_depth == 0:
            self.H.HYPRE_SetMemoryLocation(DEVICE)
            self.H.HYPRE_SetExecutionPolicy(DEVICE)

    def host_copy(self, A):
        """A copy in host memory of the matrix ``A`` of the library (a pointer; the
        caller's to destroy), the diagonal first in its rows."""
        B = self.H.hypre_ParCSRMatrixClone_v2(A, 1, HOST)
        if B:
            self.H.hypre_CSRMatrixReorder(self.pointer(B, "par_diag"))
        return B

    def zero(self, v):
        """Set a vector of the library to zero."""
        if self.set_zero is not None:
            self.set_zero(v)
        else:
            self.H.hypre_ParVectorSetConstantValues(v, 0.0)

    def pcg(self, S):
        """hypre's PCG for the matrix ``S``, created and not set up, and what has to live
        as long as it does.

        For a matrix that carries the defect of its row sums
        (:attr:`SingleParMatrix.row_defect`) the product of the iteration is the matrix's
        plus that vector times the argument, entry by entry: the PCG is then built on
        hypre's table of functions for a parallel matrix with the product replaced by a
        function of this module, and everything else in it is hypre's own.
        """
        H = self.H
        rd = getattr(S, "row_defect", None)
        if rd is None or not self.own_product:
            pcg = ctypes.c_void_p()
            H.HYPRE_ParCSRPCGCreate(self.comm(S.comm), ctypes.byref(pcg))
            return pcg, None
        product, divide_add, axpy = H.hypre_ParCSRMatrixMatvec, self.divide_add, H.hypre_ParVectorAxpy
        plus, minus, work, zero = rd.plus.par, rd.minus.par, rd.work.par, self.zero

        def matvec(_data, alpha, A, x, beta, y):
            # y = alpha (A + diag(c)) x + beta y.  hypre's PCG asks with alpha 1 (the
            # product with a direction) and -1 (a residual).
            if rd.zero_guess and alpha == -1.0 and beta == 1.0:
                # The residual that a solve starts with, of a guess that is zero (the
                # engine says so before it calls the PCG): the right-hand side as it
                # is, and no product (9 ms of a solve of 230 at 14 million unknowns).
                rd.zero_guess = False
                return 0
            rc = product(alpha, A, x, beta, y)
            if alpha == 1.0:
                divide_add(x, plus, y)
            elif alpha == -1.0:
                divide_add(x, minus, y)
            elif alpha != 0.0:
                zero(work)
                divide_add(x, plus, work)
                axpy(alpha, work, y)
            return rc

        call = _MATVEC(matvec)
        table = [ctypes.cast(getattr(H, name), ctypes.c_void_p) for name in _PCG_TABLE]
        table[_PCG_TABLE.index("hypre_ParKrylovMatvec")] = ctypes.cast(call, ctypes.c_void_p)
        functions = H.hypre_PCGFunctionsCreate(*table)
        pcg = ctypes.c_void_p(H.hypre_PCGCreate(functions))
        if not pcg:
            raise RuntimeError("the single-precision hypre could not create a PCG")
        return pcg, (call, rd)

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
            code = _allreduce(comm, abs(code), MPI.MAX)
        if code:
            self.H.HYPRE_ClearAllErrors()
            raise RuntimeError("the single-precision hypre: %s returned %d" % (what, code))

    @staticmethod
    def agree(comm, error):
        """Raise on every rank of ``comm`` when one of them holds ``error``, an exception
        of work each rank did on its own (``None`` on a rank where it went well)."""
        # (one flag first: gathering every rank's verdict, None on all of them where
        # all went well, is a pickle a rank on every rank, milliseconds a solve on
        # thousands of ranks)
        if not _allreduce(comm, 0 if error is None else 1, MPI.MAX):
            return
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
    and the last matrix that uses them are gone.

    A matrix that takes them says so (:meth:`borrow`) and says when it has let go of them
    (:meth:`give_back`, after it has taken the pointers out of its blocks): the arrays are
    freed when nothing refers to this object any more *and* no matrix holds them.  The
    two are not the same at the end of a process or of a reference cycle, where Python
    runs the finalizers of a group of objects in no particular order: freed here first,
    the arrays would be freed again by hypre with a matrix that still had their
    pointers."""

    def __init__(self, lib, pointers, counts):
        self.lib, self.pointers, self.counts = lib, tuple(pointers), tuple(counts)
        self.borrowers, self.dropped = 0, False

    def matches(self, lib, counts):
        return self.lib is lib and self.counts == tuple(counts) and bool(self.pointers)

    def borrow(self):
        self.borrowers += 1
        return self

    def give_back(self):
        self.borrowers -= 1
        if self.dropped and self.borrowers <= 0:
            self._free()

    def _free(self):
        pointers, self.pointers = self.pointers, ()
        try:
            location = DEVICE if self.lib.device else HOST
            for ptr in pointers:
                if ptr:
                    self.lib.H.hypre_Free(ctypes.c_void_p(ptr), location)
        except Exception:                                        # noqa: BLE001
            pass

    def __del__(self):
        self.dropped = True
        if self.borrowers <= 0:
            self._free()


_FINITE = []


def _all_finite(a):
    """Whether every entry of a device array is finite, as one reduction on the device
    (no array of flags is made)."""
    if not _FINITE:
        import jax
        import jax.numpy as jnp

        _FINITE.append(jax.jit(lambda v: jnp.all(jnp.isfinite(v))))
    return _FINITE[0](a)


_SINGLE = {}

#: Entries of a double-precision accumulator rounded per pass into a matrix of the library
#: (:func:`_put_single`: a transient of sixteen bytes each), or checked per pass
#: (:func:`_all_single`).
PIECE = 1 << 26


def _all_single(a):
    """Whether every entry of a double-precision device array is within the range of
    single precision (and finite).  A piece at a time: the reduction over the whole
    array made a copy of it on the way, the largest allocation of an assembly once more
    (the peak of JAX's memory on an MI250X at 14 million unknowns: 16.5 GiB instead of
    12.0, enough to make its arena grow to the cap)."""
    import jax
    import jax.numpy as jnp

    n = int(a.shape[0])
    w = int(min(PIECE, n))
    if w <= 0:
        return True
    fn = _SINGLE.get(w)
    if fn is None:
        top = float(np.finfo(np.float32).max)
        fn = _SINGLE[w] = jax.jit(
            lambda v, k: jnp.all(jnp.abs(jax.lax.dynamic_slice(v, (k,), (w,))) <= top))
    for s in range(0, n, w):
        if not bool(fn(a, np.int64(min(s, n - w)))):
            return False
    return True


_PIECES = {}


def _put_single(dst, acc, start, count, residual=False):
    """``count`` entries of the double-precision device array ``acc``, from entry
    ``start``, rounded to single precision into the device address ``dst``, a piece at
    a time (:data:`PIECE` entries: the transient is theirs).  With ``residual``, what
    the rounding of each entry took from it instead (itself rounded, which loses a
    ten-millionth of it)."""
    import jax
    import jax.numpy as jnp

    w = int(min(PIECE, count))
    if w <= 0:
        return
    fns = _PIECES.get(w)
    if fns is None:
        # Two programs, the second given the first one's result: in one, a compiler that
        # takes the conversion there and back for the identity leaves no residual
        # (the MI250X's, 11 October 2026: every residual zero).
        fns = _PIECES[w] = (
            jax.jit(lambda a, k: jax.lax.dynamic_slice(a, (k,), (w,)).astype(jnp.float32)),
            jax.jit(lambda a, k, r: (jax.lax.dynamic_slice(a, (k,), (w,))
                                     - r.astype(jnp.float64)).astype(jnp.float32)))
    for s in range(0, count, w):
        k = np.int64(start + min(s, count - w))     # (the last piece overlaps the one before)
        y = fns[0](acc, k)
        if residual:
            y = fns[1](acc, k, y)
        bridge.copy_from_jax(int(dst) + 4 * int(k - start), y)
        bridge.synchronize()                        # before ``y`` is released


class RowDefect:
    """What the rounding of a matrix's values to single precision took from the sum of
    each row: ``c[i] = sum_j (a_ij - fl(a_ij))``, a vector of the library, with ``1 / c``
    and ``-1 / c`` for hypre's entrywise ``y += x / b`` (infinite where ``c`` is zero)
    and a vector to work in.

    An entry is rounded by up to half a unit in its last place, and a row of a
    discretized differential operator sums to much less than its entries: to zero for a
    diffusion operator away from the boundary.  After the rounding it sums to a few
    hundred-millionths of its diagonal.  On a mesh of like elements with a constant
    coefficient every like row is rounded alike, so the matrix has gained a term of
    order zero with one sign over the whole domain, ``A + diag(d)``, and the solution of
    a system with it is off by ``A^{-1} (d x)``: relative to ``x``, the rounding unit
    times the square of the number of nodes across the domain (measured on the model
    problem at the prior mean, second-order hexahedra: the residual a refinement's
    second pass leaves, 3.7e-10 of the first at 120^3 and 2.0e-9 at 240^3).  The
    element matrices are exact on the constants (``GroupKernel.consistent_block``); it
    is their storage in single precision that is not.  ``A + diag(c)`` has the row sums
    of the double-precision values again, and it is what hypre's PCG multiplies by where
    a matrix has its defect (:meth:`_Library.pcg`).
    """

    def __init__(self, S):
        self.lib = S.lib
        self.c, self.plus, self.minus, self.work = (S.vector() for _ in range(4))
        #: set by an engine before a solve from a zero guess, and read once by the
        #: product of its PCG (:meth:`_Library.pcg`): the first residual is the
        #: right-hand side
        self.zero_guess = False

    def finish(self):
        """``plus`` and ``minus`` from ``c``."""
        lib, n = self.lib, self.c.n
        if not n:
            return
        if lib.device:
            import jax.numpy as jnp

            bridge.synchronize()
            c = bridge.copy_to_jax(self.c.data, n, np.float32)
            inv = jnp.where(c != 0, 1.0 / jnp.where(c != 0, c, 1.0), jnp.inf).astype(jnp.float32)
            neg = -inv
            bridge.copy_from_jax(self.plus.data, inv)
            bridge.copy_from_jax(self.minus.data, neg)
            bridge.synchronize()                    # before the two are released
        else:
            c = np.ctypeslib.as_array((ctypes.c_float * n).from_address(int(self.c.data)))
            with np.errstate(divide="ignore"):
                inv = np.where(c != 0, np.float32(1.0) / np.where(c != 0, c, np.float32(1.0)),
                               np.float32(np.inf)).astype(np.float32)
            lib.put(self.plus.data, inv)
            lib.put(self.minus.data, np.ascontiguousarray(-inv))

    def values(self):
        """``c`` as a numpy array (for a test or a report)."""
        n = self.c.n
        if not n:
            return np.zeros(0, dtype=np.float32)
        if self.lib.device:
            bridge.synchronize()
            return np.asarray(bridge.copy_to_jax(self.c.data, n, np.float32))
        return np.array(np.ctypeslib.as_array((ctypes.c_float * n).from_address(int(self.c.data))))

    def destroy(self):
        for name in ("c", "plus", "minus", "work"):
            v = getattr(self, name, None)
            if v is not None:
                v.destroy()
            setattr(self, name, None)


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
    def from_pattern(cls, tp, acc, lib=None, row_sums=False):
        """The matrix of the true-dof pattern ``tp`` with the values ``acc``: the entries
        of the diagonal block, then those of the off-diagonal one, in the pattern's
        order (what :meth:`TrueDofPattern.finish` holds after its exchange), as a JAX
        array on the device or a numpy array.

        With ``row_sums`` and values in double precision the matrix also keeps what
        their rounding takes from the sum of each row (:attr:`row_defect`)."""
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
        #: with the library on a device: whether the first transposed product forms the
        #: transposes of the two blocks and the matrix keeps them (as MFEM's matrices
        #: do), at the memory of a second matrix; ``False``: hypre forms one at every
        #: transposed product.  For a matrix that is applied transposed many times.
        self.keep_transpose = False
        self._transposed = False
        #: :class:`RowDefect`, or ``None``
        self.row_defect = None
        n, nd, no = int(tp.ntd), int(tp.nnz_diag), int(tp.nnz_t - tp.nnz_diag)
        n_offd = int(tp.n_offd)
        self._height, self._width = n, int(tp.c1 - tp.c0)
        self.global_rows, self.global_cols = int(tp.gnr), int(tp.gnc)
        if max(self.global_rows, self.global_cols) > np.iinfo(lib.bigint).max:
            # (the same on every rank: nothing has been exchanged yet)
            raise ValueError("a matrix of %d by %d is beyond the 32-bit global indices of the "
                             "single-precision hypre %s: build it --enable-mixedint"
                             % (self.global_rows, self.global_cols, lib.path))
        self.row_starts = np.array([tp.t0, tp.t1], dtype=lib.bigint)
        self.col_starts = np.array([tp.c0, tp.c1], dtype=lib.bigint)
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
                self._columns = shared.borrow()
            lib.check(H.hypre_ParCSRMatrixInitialize_v2(self.par, DEVICE if lib.device else HOST),
                      "ParCSRMatrixInitialize")
            on_host = isinstance(acc, np.ndarray)
            defect = None
            if on_host:
                with np.errstate(over="ignore", invalid="ignore"):
                    vals = np.ascontiguousarray(acc[:nd + no], dtype=np.float32)
                finite = bool(np.isfinite(vals).all())
                if row_sums and finite and acc.dtype == np.float64:
                    # what the rounding took, row by row (the sums of small numbers of
                    # either sign, in double precision)
                    defect = np.zeros(n)
                    for I, start, count in ((tp.I_diag, 0, nd), (tp.I_offd, nd, no)):
                        if count:
                            I = np.asarray(I[:n + 1], dtype=np.int64)
                            run = np.concatenate(([0.0], np.cumsum(
                                acc[start:start + count] - vals[start:start + count])))
                            defect += run[I[1:]] - run[I[:-1]]
            else:
                import jax.numpy as jnp

                if acc.dtype == jnp.float32:
                    # accumulated in single precision (tdofassemble.SINGLE_ACCUMULATE):
                    # the accumulator itself holds the values, and nothing is copied
                    vals = acc
                    finite = bool(_all_finite(vals))
                else:
                    # in double precision: rounded a piece at a time (_put_single)
                    vals = None
                    finite = bool(_all_single(acc))
                    defect = bool(row_sums and finite)
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
                elif vals is not None:
                    bridge.copy_from_jax(lib.pointer(csr, "csr_data"), vals, start, count)
                else:
                    # (with a defect to find, what the rounding took comes first: the
                    # values follow when the matrix has multiplied that, below)
                    _put_single(lib.pointer(csr, "csr_data"), acc, start, count,
                                residual=bool(defect))
            if n_offd:
                cm = np.ascontiguousarray(tp.cmap[:n_offd], dtype=lib.bigint)
                ctypes.memmove(lib.pointer(par, "par_col_map_offd"), cm.ctypes.data, cm.nbytes)
            if lib.device:
                bridge.synchronize()               # before ``vals`` is released
            if self._columns is None and SHARE_COLUMNS:
                # the first matrix of the pattern: its column indices become the
                # pattern's, lent to it and to every later matrix
                self._columns = tp._single_columns = _SharedColumns(
                    lib, (lib.pointer(lib.pointer(par, "par_diag"), "csr_j") if nd else None,
                          lib.pointer(lib.pointer(par, "par_offd"), "csr_j") if no else None),
                    (nd, no)).borrow()
        except Exception as e:                                   # noqa: BLE001
            failed = e
        del vals
        try:
            lib.agree(self.comm, failed)
            lib.check(H.hypre_ParCSRMatrixSetNumNonzeros(self.par), "ParCSRMatrixSetNumNonzeros",
                      self.comm)
            H.hypre_ParCSRMatrixSetDNumNonzeros(self.par)
            lib.check(H.hypre_MatvecCommPkgCreate(self.par), "MatvecCommPkgCreate", self.comm)
            if row_sums:
                # (collective: every rank is here with the same ``row_sums``, and a rank
                # whose values were given in single precision has no defect to add)
                self._set_row_defect(acc, defect, nd, no)
        except Exception:
            self.destroy()
            raise
        return self

    def _set_row_defect(self, acc, defect, nd, no):
        """Make :attr:`row_defect`.  ``defect`` is the vector on the host, or ``True``
        where the matrix holds, in place of its values, what the rounding of the device
        array ``acc`` took from each: the vector is then the product of that matrix with
        the vector of ones (an exchange with the other ranks, who are all on this branch:
        the route of an assembly is one for all), and the values are written afterwards.
        ``None``: the values came in single precision, and there is nothing to keep."""
        lib, H = self.lib, self.lib.H
        if defect is None or defect is False:
            return
        rd = RowDefect(self)
        try:
            if defect is True:
                ones = self.vector(columns=True)
                try:
                    H.hypre_ParVectorSetConstantValues(ones.par, 1.0)
                    lib.check(H.hypre_ParCSRMatrixMatvec(1.0, self.par, ones.par, 0.0, rd.c.par),
                              "Matvec", self.comm)
                    if lib.device:
                        bridge.synchronize()
                finally:
                    ones.destroy()
                for name, start, count in (("par_diag", 0, nd), ("par_offd", nd, no)):
                    if count:
                        _put_single(lib.pointer(lib.pointer(self.par.value, name), "csr_data"),
                                    acc, start, count)
            else:
                lib.put(rd.c.data, np.ascontiguousarray(defect, dtype=np.float32))
            rd.finish()
        except Exception:
            rd.destroy()
            raise
        self.row_defect = rd

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

    def vector(self, columns=False, host=False):
        """A vector of the single-precision library on this matrix's rows (or columns);
        ``host=True``: in host memory also with the library on a device."""
        return SingleVector(self.lib, self.comm, self.global_cols if columns else self.global_rows,
                            self.col_starts if columns else self.row_starts, host=host)

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
        """``y = A^T x`` on two of MFEM's vectors, the product in single precision."""
        lib = self.lib
        vx, vy = self._work()
        vy.set(x)
        if self.keep_transpose and not self._transposed and lib.device and lib.local_transpose is not None:
            # (every rank is here together, as for the product below)
            lib.check(lib.local_transpose(self.par), "ParCSRMatrixLocalTranspose", self.comm)
            self._transposed = True
        lib.check(lib.H.hypre_ParCSRMatrixMatvecT(1.0, self.par, vy.par, 0.0, vx.par),
                  "MatvecT", self.comm)
        vx.get(y)

    def destroy(self):
        if self._vecs is not None:
            for v in self._vecs:
                v.destroy()
            self._vecs = None
        if getattr(self, "row_defect", None) is not None:
            self.row_defect.destroy()
            self.row_defect = None
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
            if shared is not None:
                shared.give_back()             # (freed with the last of its matrices)

    def __del__(self):
        try:
            self.destroy()
        except Exception:                                        # noqa: BLE001
            pass


class _MatrixBlocks:
    """What :meth:`SingleParMatrix.from_pattern` reads of a true-dof pattern, for a copy of
    one of MFEM's parallel matrices (:func:`matrix_blocks`)."""

    def device_columns(self, block, create=True):
        return None


def matrix_blocks(A, comm):
    """The rows of one of MFEM's parallel matrices ``A`` (double precision) on this rank
    as :meth:`SingleParMatrix.from_pattern` takes them, with their values in single
    precision (``values``): for a copy of a matrix that stays, such as a prior's
    elliptic operator.  The two blocks are read in host memory (twelve bytes an entry
    there for the time of the call).  A square matrix of hypre has the diagonal first
    in its rows, and the copy keeps it there."""
    from ..common.linalg import _diag_block, _offd_block
    from ..common.parvector import host_sync

    tb = _MatrixBlocks()
    tb.comm = comm
    rp, cp = A.GetRowPartArray(), A.GetColPartArray()
    tb.t0, tb.t1, tb.c0, tb.c1 = int(rp[0]), int(rp[1]), int(cp[0]), int(cp[1])
    tb.gnr, tb.gnc = int(A.GetGlobalNumRows()), int(A.GetGlobalNumCols())
    tb.ntd = tb.t1 - tb.t0
    blk = _diag_block(A)
    host_sync(blk)
    tb.I_diag = np.array(blk.GetIArray(), dtype=np.int32)
    tb.nnz_diag = int(tb.I_diag[-1]) if tb.I_diag.size else 0
    tb.J_diag = np.array(blk.GetJArray(), dtype=np.int32)[:tb.nnz_diag]
    values = [np.array(blk.GetDataArray(), dtype=np.float32)[:tb.nnz_diag]]
    blk, cmap = _offd_block(A)
    tb.cmap, tb.n_offd = cmap, int(cmap.size)
    if tb.n_offd:
        host_sync(blk)
        tb.I_offd = np.array(blk.GetIArray(), dtype=np.int32)
        no = int(tb.I_offd[-1])
        tb.J_offd = np.array(blk.GetJArray(), dtype=np.int32)[:no]
        values.append(np.array(blk.GetDataArray(), dtype=np.float32)[:no])
    else:
        tb.I_offd, tb.J_offd, no = np.zeros(tb.ntd + 1, dtype=np.int32), np.zeros(0, dtype=np.int32), 0
    if tb.I_diag.size != tb.ntd + 1 or tb.I_offd.size != tb.ntd + 1:
        raise RuntimeError("the blocks of the matrix have %d and %d rows, the matrix %d on this rank"
                           % (tb.I_diag.size - 1, tb.I_offd.size - 1, tb.ntd))
    tb.nnz_t = tb.nnz_diag + no
    tb.values = np.concatenate(values)
    return tb


class SingleVector:
    """A vector of the single-precision library, filled from and read into MFEM's."""

    def __init__(self, lib, comm, global_size, starts, host=False):
        self.lib = lib
        #: in host memory (always, without a device; with one, where asked: a level of a
        #: p-multigrid that lives there)
        self.host = bool(host) or not lib.device
        starts = np.ascontiguousarray(starts, dtype=lib.bigint)
        self.n = int(starts[1] - starts[0])
        par = lib.H.hypre_ParVectorCreate(lib.comm(comm), int(global_size), starts.ctypes.data)
        if not par:
            raise RuntimeError("the single-precision hypre could not create a vector")
        self.par = ctypes.c_void_p(par)
        lib.check(lib.H.hypre_ParVectorInitialize_v2(self.par, HOST if self.host else DEVICE),
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


def _boomeramg(lib, p, gather=True):
    """A BoomerAMG of the single-precision library with the settings of a solver's
    parameters ``p``, not set up: MFEM's defaults for the device and for the host, then the
    solver's own, then :data:`AMG_OPTIONS`.  ``gather=False`` leaves ``seq_threshold`` and
    ``redundant`` out of them (a hierarchy in host memory, whose last matrix the engine
    gathers itself or which goes on between the ranks: :class:`_HostGathered`).

    On a device HIPPYMFEM_SINGLE_AMG="relax=7,pmax=6" (Jacobi relaxation instead of
    l1-Jacobi, which divides by the sum of the magnitudes of a row, a few times the
    diagonal for quadratic elements, and six interpolation entries a row instead of four)
    took the solves of the model problem at 64^3 (Blackwell instance, to 1e-5) from 11 to 6
    iterations and from 76 to 48 ms at the MAP point, the prior mean and the truth alike,
    and Newton-CG to 1e-6 from 38.5 to 29.0 s (H100: 19.8 to 15.7 s) with the same Newton
    and CG counts.  Opt-in for good: plain Jacobi smooths only where the largest eigenvalue
    of D^-1 A is below 2 (1.71 on the model problem; 2.38 on quadratic tetrahedra, 6.85
    with a parameter three times as large).  Above, the preconditioner is not positive
    definite and hypre's PCG stops after a few iterations, which the solver reports as a
    failure (KrylovSolver._solve_single).  "relax=16,cheby_order=1,pmax=6" takes the weight
    from an eigenvalue estimate on every level and converged in every case tried, with half
    the gain on the model problem."""
    H = lib.H
    amg = ctypes.c_void_p()
    H.HYPRE_BoomerAMGCreate(ctypes.byref(amg))
    try:
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
        # the solver's options for both precisions (HIPPYMFEM_AMG), then those of the
        # single-precision solves alone
        both = dict(mfemconfig.parse_boomeramg_options(str(p["amg_options"] or "")))
        both.update(AMG_OPTIONS)
        if gather:
            mfemconfig.check_seq_threshold(both, H, gpu)
        else:
            both = {k: v for k, v in both.items() if k not in ("seq_threshold", "redundant")}
        for key, val in both.items():
            name, kind = _AMG_SETTERS[key]
            fn = getattr(H, name)
            if key in ("coarse_relax", "coarse_sweeps"):
                fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
                rc = fn(amg, int(val), 3)
            else:
                fn.argtypes = [ctypes.c_void_p, ctypes.c_float if kind is float else ctypes.c_int]
                fn.restype = ctypes.c_int
                rc = fn(amg, val)
            if rc:
                H.HYPRE_ClearAllErrors()
                raise ValueError("the single-precision hypre refused %s=%r (%s returned %d)"
                                 % (key, val, name, rc))
    except Exception:
        H.HYPRE_BoomerAMGDestroy(amg)
        raise
    return amg


def gather_threshold(p):
    """The ``seq_threshold`` of the hierarchies of a solver's parameters ``p`` (0: none):
    a hierarchy ends at its first level of at most that many rows, which is gathered and
    solved by one process a rank (:func:`hippymfem.common.mfemconfig.check_seq_threshold`)."""
    both = dict(mfemconfig.parse_boomeramg_options(str(p["amg_options"] or "")))
    both.update(AMG_OPTIONS)
    return int(both.get("seq_threshold", 0) or 0)


def _relaxation(lib, p):
    """The relaxation of a hierarchy's levels under the settings of ``p``, as
    :func:`_boomeramg` reads them: hypre's type and the weight."""
    relax = int(p["amg_relax_type"])
    relax, weight = (relax if relax >= 0 else (18 if lib.device else 8)), 1.0
    for options in (mfemconfig.parse_boomeramg_options(str(p["amg_options"] or "")), AMG_OPTIONS):
        relax, weight = int(options.get("relax", relax)), float(options.get("relax_wt", weight))
    return relax, weight


class SingleEngine:
    """hypre's PCG with BoomerAMG on a :class:`SingleParMatrix`, set up once and shared
    by the solvers of that matrix (the forward and the two incremental ones)."""

    def __init__(self, S, parameters):
        self.matrix, self.lib = S, S.lib
        H, lib = S.lib.H, S.lib
        self.amg = self.pcg = self._product = None
        self.converged, self.refused, self.error = False, False, 0
        self.b, self.x = S.vector(), S.vector(columns=True)
        self.amg = amg = _boomeramg(lib, parameters)
        pcg, self._product = lib.pcg(S)
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
        iterations and the final relative residual in the preconditioner's norm.

        :attr:`converged` is hypre's own verdict on the solve and :attr:`error` its
        error flag.  The count says nothing, and the verdict is not enough: with a
        preconditioner ``C`` that is not positive definite, ``<C r, r>`` turns negative,
        hypre's test ``<C r, r> < eps <C b, b>`` then holds, and the solve returns as
        converged after a few iterations with a residual that is not a number (and a
        true one of order one).

        :attr:`refused`: hypre did not iterate.  Where ``<C b, b>`` is not positive in
        single precision it takes the right-hand side for zero, and where it is not
        finite for a wrong input; it returns at once, without a verdict and with the
        count and the residual of the solve before.  0 and not-a-number are returned
        then.  (``KrylovSolver._solve_single`` solves such a system, and one on which
        the iteration breaks off, in units that suit the right-hand side.)"""
        H, lib = self.lib.H, self.lib
        H.HYPRE_PCGSetTol(self.pcg, max(float(rel_tol), SINGLE_TOL_FLOOR))
        H.HYPRE_PCGSetAbsoluteTol(self.pcg, float(abs_tol))
        H.HYPRE_PCGSetMaxIter(self.pcg, int(max_iter))
        self.b.set(b)
        self.x.zero()
        defect = self._product[1] if self._product is not None else None
        if defect is not None:
            defect.zero_guess = True
        try:
            rc = H.HYPRE_ParCSRPCGSolve(self.pcg, self.matrix.par, self.b.par, self.x.par)
        finally:
            if defect is not None:
                defect.zero_guess = False            # (a PCG that returned before its residual)
        if rc:
            H.HYPRE_ClearAllErrors()                # the flag is the process's and stays up
        self.x.get(x)
        its, norm, conv = ctypes.c_int(), ctypes.c_float(), ctypes.c_int()
        H.HYPRE_PCGGetNumIterations(self.pcg, ctypes.byref(its))
        H.HYPRE_PCGGetFinalRelativeResidualNorm(self.pcg, ctypes.byref(norm))
        H.HYPRE_PCGGetConverged(self.pcg, ctypes.byref(conv))
        self.converged, self.error, self.refused = bool(conv.value), int(rc), False
        if not conv.value:
            # An iteration that fails, by a breakdown or at its limit, raises the bit.
            # The verdict is one for all ranks, but the flag is a process's: what is
            # read from it is reduced over them, or one rank would go on alone.
            failed = _allreduce(self.matrix.comm, 1 if rc & HYPRE_ERROR_CONV else 0, MPI.MAX)
            self.refused = not failed
        if self.refused:
            return 0, float("nan")
        return int(its.value), float(norm.value)

    def destroy(self):
        H = self.lib.H
        if self.pcg is not None:
            H.HYPRE_ParCSRPCGDestroy(self.pcg)
            self.pcg = None
        self._product = None                       # (the product's callback, after its PCG)
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


#: Whether a :class:`PMGEngine` keeps the product ``A P`` of the matrix with the
#: interpolation, which the Galerkin product passes through anyway.  The residual after the
#: coarse correction is then ``t - (A P) e`` instead of ``b - A x``: a product with a
#: quarter of the matrix's entries in place of one of the three products with the matrix
#: that an iteration makes, for 2 bytes of device memory an entry of the matrix
#: (1.7 GiB with 14 million quadratic-hexahedron unknowns).  ``HIPPYMFEM_PMG_KEEP_AP=0``
#: recomputes the residual instead.
PMG_KEEP_AP = os.environ.get("HIPPYMFEM_PMG_KEEP_AP", "1").lower() not in ("0", "no", "false", "off")

#: Whether the two products of a p-multigrid, ``P^T A P`` and ``A P``, are made from the
#: element matrices in the pass that assembles the Jacobian (``P_e^T A_e P_e``, 8 by 8 for
#: a quadratic hexahedron, and ``A_e P_e``, scattered into their own patterns) where the
#: problem can (``PDEVariationalProblem._jacobian_with_products``), instead of by hypre's
#: sparse products of the assembled matrices.  The same two matrices; the element
#: products cost a third of the Jacobian's scatter more and nothing that grows with the
#: ranks, where hypre's products take 0.20 s on one MI250X and 0.37 s on eight at 14
#: million unknowns each.  ``HIPPYMFEM_PMG_ELEMENT_PRODUCTS=0`` leaves them to hypre.
PMG_ELEMENT_PRODUCTS = os.environ.get("HIPPYMFEM_PMG_ELEMENT_PRODUCTS", "1").lower() not in (
    "0", "no", "false", "off")

#: The relaxation on a mesh of a p-multigrid's geometric levels whose next mesh is ``f``
#: times coarser a side, ``f`` three or more: ``"chebyshev"`` (the default), ``f`` steps of
#: the Chebyshev iteration on the matrix scaled by the l1 norms of its rows, for the
#: eigenvalues from ``1.2 / f^2`` to one (those norms bound them by one, and the mesh below
#: carries the error under ``1 / f^2`` of the largest): its error on that band is at most
#: 0.20 for a cut by three and 0.21 for one by five, at ``f - 1`` products with the level's
#: matrix before the correction and as many after it.  ``"sweep"``: the one weighted sweep
#: of every other level.  A mesh cut in two keeps the sweep.  ``HIPPYMFEM_PMG_H_SMOOTH``.
PMG_H_SMOOTH = os.environ.get("HIPPYMFEM_PMG_H_SMOOTH", "chebyshev").strip().lower()
if PMG_H_SMOOTH not in ("chebyshev", "sweep"):
    raise ValueError("HIPPYMFEM_PMG_H_SMOOTH must be chebyshev or sweep, not %r" % (PMG_H_SMOOTH,))

#: With the library on a device, the geometric levels of a p-multigrid that live in host
#: memory: a mesh's matrix of at most this many rows a rank, and everything under it (the
#: matrices of the coarser meshes, their relaxations and BoomerAMG on the last).  Not where
#: the last matrix is one that the hierarchy gathers and has more rows than this in all:
#: every rank solves the whole of it, on one core in host memory, and the levels above it
#: then stay on the device with it.  Nor the last matrix alone, which has no exchange
#: between ranks to save.  A level costs a device its kernel launches and, around each of
#: its exchanges between ranks, a copy from the card and one back, whatever its size:
#: 0.16 ms an exchange and more on 512 MI250X, and a cycle over the cuts 5, 3, 2, 2, 2 of
#: 120^3 elements a rank has 20 of its 38 exchanges on the two meshes of 3375 and of 125
#: elements a rank.  On the host such a level takes microseconds and its exchange is a
#: message; the cycle hands the restricted residual to the host at the first such level
#: and takes the correction back.  The matrices of the geometric levels have the
#: neighbours of the mesh at every size, which is why this pays where the same for the
#: last levels of an algebraic hierarchy, each rank of which talks to many, did not
#: (``docs/source/guide/gpu.rst``).  0, the default: every level on the device.  The
#: solver parameter ``pmg_h_host``; ``HIPPYMFEM_PMG_H_HOST``.
PMG_H_HOST = int(os.environ.get("HIPPYMFEM_PMG_H_HOST", "0") or 0)

#: A probe's clock on the coarser meshes of a p-multigrid: ``HIPPYMFEM_PMG_TIMERS=k`` times
#: the correction from level ``k`` of them down (0: the linear elements of the mesh itself;
#: the number of levels: the last matrix alone), with the device at rest before and after,
#: into :data:`TIMERS`.  Several levels, as ``2,3,4``, are timed one inside the other, so
#: that the differences are what each level takes by itself.  A clock costs two waits for
#: the device a cycle: a run with it is for the clock, not for the time of the steps.
PMG_TIMERS = tuple(sorted(set(int(t) for t in os.environ.get("HIPPYMFEM_PMG_TIMERS", "").replace(";", ",").split(",")
                              if t.strip().lstrip("-").isdigit() and int(t) >= 0)))
#: what the clocks of :data:`PMG_TIMERS` measured in this process: by the kind of cycle
#: (``"PMGEngine"``: a Jacobian's p-multigrid; ``"HCycle"``: a prior's cycle over the meshes)
#: and the level, the seconds and the calls
TIMERS = {}

#: ``HIPPYMFEM_PMG_TRACE=1``: the first and the middle rank say where a p-multigrid with
#: levels in host memory is, in its set-up and in its first cycles (to find a rank that
#: waits; the lines are flushed as they are written).
PMG_TRACE = os.environ.get("HIPPYMFEM_PMG_TRACE", "").strip() not in ("", "0")
_TRACED = {"cycles": 0}


def _trace(comm, text):
    if PMG_TRACE and comm.rank in (0, comm.size // 2):
        print("[pmg, rank %d] %s" % (comm.rank, text), flush=True)


#: What the last engine of this process put in host memory (:data:`PMG_H_HOST`): the index
#: of the first such level among the coarser meshes and the rows of its matrix; empty
#: where it put nothing there.  For a benchmark's record.
HOST_LEVELS = {}

#: a preconditioner as hypre's PCG calls it: (its data, the matrix, the residual, the result)
_PRECOND = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_void_p)
#: a product as hypre's Krylov solvers call it: (its data, alpha, the matrix, x, beta, y)
_MATVEC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p)


class PTransfer:
    """The interpolation of a p-multigrid and its transpose in the single-precision library:
    from the linear elements of a mesh to a space of higher order
    (:func:`hippymfem.fem.ptransfer.interpolation`).  Made once for a space and its
    essential dofs, and used by every :class:`PMGEngine` of a matrix on that space."""

    def __init__(self, blocks, lib=None):
        lib = lib if lib is not None else library()
        if lib is None:
            raise RuntimeError("no single-precision hypre (%s)"
                               % (why_not() or "HIPPYMFEM_HYPRE_SINGLE is not set"))
        self.lib, self.comm = lib, blocks.comm
        self.P = self.R = None
        #: the two in host memory, where a level of theirs lives there (:meth:`host`)
        self._host = None
        if lib.no_pmg:
            raise RuntimeError("no p-multigrid in this single-precision hypre: %s" % lib.no_pmg)
        self.nnz = int(blocks.global_nnz)
        self.row_sum_error = float(blocks.row_sum_error)
        #: for the products from element matrices (:data:`PMG_ELEMENT_PRODUCTS`): the
        #: coarse space, its essential true dofs and the interpolation of an element
        self.coarse = getattr(blocks, "coarse", None)
        self.coarse_ess = getattr(blocks, "coarse_ess", None)
        self.local = getattr(blocks, "local", None)
        #: between two meshes (:func:`hippymfem.fem.ptransfer.refinement`): how many times
        #: coarser a side the coarse one is
        self.cut = getattr(blocks, "cut", None)
        #: (the blocks own the column indices that the matrix borrows)
        self.blocks = blocks
        self.P = SingleParMatrix.from_pattern(blocks, blocks.values, lib)
        blocks.values = None
        # the transpose as a matrix of its own: a product with the transpose of a matrix
        # transposes it at every call on a device
        R = ctypes.c_void_p()
        lib.check(lib.H.hypre_ParCSRMatrixTranspose(self.P.par, ctypes.byref(R), 1),
                  "ParCSRMatrixTranspose", self.comm)
        self.R = R
        if lib.device:
            bridge.synchronize()

    def host(self):
        """The interpolation and its transpose in host memory, with the library on a
        device (two pointers, made at the first call and kept): for a level that lives
        there (:data:`PMG_H_HOST`).  Collective."""
        if self._host is None:
            lib, H = self.lib, self.lib.H
            P = H.hypre_ParCSRMatrixClone_v2(self.P.par, 1, HOST)
            R = H.hypre_ParCSRMatrixClone_v2(self.R, 1, HOST) if P else None
            self._host = (ctypes.c_void_p(P) if P else None, ctypes.c_void_p(R) if R else None)
            lib.check(0 if (P and R) else 1, "ParCSRMatrixClone (host)", self.comm)
        return self._host

    def destroy(self):
        for M in getattr(self, "_host", None) or ():
            if M is not None:
                self.lib.H.hypre_ParCSRMatrixDestroy(M)
        self._host = None
        if self.R is not None:
            self.lib.H.hypre_ParCSRMatrixDestroy(self.R)
            self.R = None
        if self.P is not None:
            self.P.destroy()
            self.P = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:                                        # noqa: BLE001
            pass


class _HostGathered:
    """The last matrix of a p-multigrid's coarser meshes, in host memory, as one system on
    every rank: its rows are gathered from all the ranks at the set-up and a BoomerAMG of
    one process is set up on them, the same on every rank; a cycle gathers the right-hand
    side, solves, and each rank keeps its own entries of its own solution.

    hypre's gathered solve (``seq_threshold``) does this for a matrix on a device, with the
    patch of ``tools/``.  In host memory its own code does not gather the matrix it is
    given but coarsens it once between the ranks and gathers what that leaves, on the ranks
    that hold rows of it; a rank that holds none takes the coarsest level for its own to
    relax, which is a collective of all the ranks, while the others gather among
    themselves.  A last matrix of 125 rows on 64 ranks (one or two a rank) stopped in its
    first cycle that way, and one of 729 rows on 512.  The matrices that reach this class
    are a few thousand rows at the most (the solver's ``seq_threshold``, and
    :data:`PMG_H_HOST` beside a device), so a gather of the whole of them is small.
    """

    def __init__(self, lib, comm, A, first_row, parameters):
        H = lib.H
        self.lib, self.comm = lib, comm
        self.amg = self.par = self.F = self.U = None
        diag, offd = lib.pointer(A.value, "par_diag"), lib.pointer(A.value, "par_offd")
        n = int(lib.integer(diag, "csr_num_rows"))

        def view(ptr, count, ctype):
            if not count or not ptr:
                return np.zeros(0, dtype=ctype)
            return np.ctypeslib.as_array((ctype * int(count)).from_address(int(ptr)))

        big = ctypes.c_longlong if lib.bigint.itemsize == 8 else ctypes.c_int
        blocks = []
        for csr in (diag, offd):
            i = view(lib.pointer(csr, "csr_i"), n + 1, ctypes.c_int) if n else np.zeros(1, dtype=np.int32)
            nnz = int(i[-1]) if n else 0
            blocks.append((np.asarray(i, dtype=np.int64), view(lib.pointer(csr, "csr_j"), nnz, ctypes.c_int),
                           view(lib.pointer(csr, "csr_data"), nnz, ctypes.c_float)))
        (di, dj, dv), (oi, oj, ov) = blocks
        cmap = view(lib.pointer(A.value, "par_col_map_offd"), int(lib.integer(offd, "csr_num_cols")), big)
        # this rank's rows with global columns, the diagonal block's entries first in a row
        # (the diagonal leads them: the copy in host memory was reordered)
        dn, on = np.diff(di), np.diff(oi)
        start = np.concatenate(([0], np.cumsum(dn + on)))
        cols = np.empty(int(start[-1]), dtype=np.int32)
        vals = np.empty(int(start[-1]), dtype=np.float32)
        rows = np.repeat(np.arange(n), dn)
        at = start[rows] + (np.arange(dj.size) - di[rows])
        cols[at], vals[at] = dj + int(first_row), dv
        rows = np.repeat(np.arange(n), on)
        at = start[rows] + dn[rows] + (np.arange(oj.size) - oi[rows])
        cols[at], vals[at] = cmap[oj] if oj.size else oj, ov
        # the whole matrix on every rank (the ranks hold consecutive rows, in their order)
        size = comm.size
        sizes = np.empty((size, 2), dtype=np.int32)
        comm.Allgather(np.array([n, cols.size], dtype=np.int32), sizes)
        nrows, nnzs = np.ascontiguousarray(sizes[:, 0]), np.ascontiguousarray(sizes[:, 1])
        rdsp = np.concatenate(([0], np.cumsum(nrows)[:-1])).astype(np.int32)
        ndsp = np.concatenate(([0], np.cumsum(nnzs)[:-1])).astype(np.int32)
        N, NNZ = int(nrows.sum()), int(nnzs.sum())
        per_row = np.empty(N, dtype=np.int32)
        J, V = np.empty(NNZ, dtype=np.int32), np.empty(NNZ, dtype=np.float32)
        comm.Allgatherv([np.ascontiguousarray((dn + on).astype(np.int32)), MPI.INT], [per_row, (nrows, rdsp), MPI.INT])
        comm.Allgatherv([cols, MPI.INT], [J, (nnzs, ndsp), MPI.INT])
        comm.Allgatherv([vals, MPI.FLOAT], [V, (nnzs, ndsp), MPI.FLOAT])
        I = np.zeros(N + 1, dtype=np.int32)
        np.cumsum(per_row, out=I[1:])
        self.n, self.first, self.N = n, int(rdsp[comm.rank]), N
        self._counts, self._displs = nrows, rdsp
        self._starts = np.array([0, N], dtype=lib.bigint)
        one = lib.comm(MPI.COMM_SELF)
        device = lib.device
        if device:
            lib.to_host()
        try:
            par = H.hypre_ParCSRMatrixCreate(one, N, N, self._starts.ctypes.data, self._starts.ctypes.data, 0, NNZ, 0)
            if not par:
                raise RuntimeError("the single-precision hypre could not create the gathered matrix")
            self.par = ctypes.c_void_p(par)
            lib.check(H.hypre_ParCSRMatrixInitialize_v2(self.par, HOST), "ParCSRMatrixInitialize")
            csr = lib.pointer(par, "par_diag")
            for name, a in (("csr_i", I), ("csr_j", J), ("csr_data", V)):
                if a.nbytes:
                    ctypes.memmove(lib.pointer(csr, name), a.ctypes.data, a.nbytes)
            lib.check(H.hypre_ParCSRMatrixSetNumNonzeros(self.par), "ParCSRMatrixSetNumNonzeros")
            H.hypre_ParCSRMatrixSetDNumNonzeros(self.par)
            self.F = SingleVector(lib, MPI.COMM_SELF, N, self._starts, host=True)
            self.U = SingleVector(lib, MPI.COMM_SELF, N, self._starts, host=True)
            self._f = view(self.F.data, N, ctypes.c_float)
            self._u = view(self.U.data, N, ctypes.c_float)
            self.amg = _boomeramg(lib, parameters, gather=False)
            self.code = int(H.HYPRE_BoomerAMGSetup(self.amg, self.par, self.F.par, self.U.par)) if N else 0
            if PMG_TRACE and N and not self.code:
                # what one cycle leaves of the residual of a right-hand side without structure
                rng = np.random.RandomState(7)
                f0 = (rng.rand(N) - 0.5).astype(np.float32)
                self._f[:] = f0
                lib.zero(self.U.par)
                H.HYPRE_BoomerAMGSolve(self.amg, self.par, self.F.par, self.U.par)
                R = SingleVector(lib, MPI.COMM_SELF, N, self._starts, host=True)
                H.hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, self.par, self.U.par, 1.0, self.F.par, R.par)
                left = float(np.linalg.norm(view(R.data, N, ctypes.c_float))) / float(np.linalg.norm(f0))
                R.destroy()
                levels = ctypes.c_int()
                if hasattr(H, "HYPRE_BoomerAMGGetNumLevels"):
                    H.HYPRE_BoomerAMGGetNumLevels(self.amg, ctypes.byref(levels))
                _trace(comm, "the gathered matrix: %d rows, %d entries, a hierarchy of %d levels; one cycle leaves "
                             "%.3f of a residual; the diagonal leads %d of the rows; sum of the entries %.6e"
                       % (N, NNZ, levels.value, left, int((J[I[:-1]] == np.arange(N)).sum()), float(V.astype(np.float64).sum())))
        finally:
            if device:
                lib.to_device()

    def solve(self, b, x, n):
        """The correction for the right-hand side whose entries on this rank are the ``n``
        numbers at ``b`` (host memory), into the ``n`` at ``x``.  Collective; with the
        library on a device the caller has called :meth:`_Library.to_host`."""
        H = self.lib.H
        mine = (np.ctypeslib.as_array((ctypes.c_float * n).from_address(int(b))) if n
                else np.zeros(0, dtype=np.float32))
        self.comm.Allgatherv([mine, MPI.FLOAT], [self._f, (self._counts, self._displs), MPI.FLOAT])
        if not self.N:
            return
        self.lib.zero(self.U.par)
        H.HYPRE_BoomerAMGSolve(self.amg, self.par, self.F.par, self.U.par)
        if n:
            ctypes.memmove(int(x), self._u.ctypes.data + 4 * self.first, 4 * n)

    def destroy(self):
        H, device = self.lib.H, self.lib.device
        if device:
            self.lib.to_host()
        try:
            if self.amg is not None:
                H.HYPRE_BoomerAMGDestroy(self.amg)
                self.amg = None
            if self.par is not None:
                H.hypre_ParCSRMatrixDestroy(self.par)
                self.par = None
        finally:
            if device:
                self.lib.to_device()
        self._f = self._u = None
        for name in ("F", "U"):
            if getattr(self, name, None) is not None:
                getattr(self, name).destroy()
            setattr(self, name, None)

    def __del__(self):
        try:
            self.destroy()
        except Exception:                                        # noqa: BLE001
            pass


class PMGEngine(SingleEngine):
    """hypre's PCG on a :class:`SingleParMatrix` of elements of higher order with a
    p-multigrid as its preconditioner, in place of BoomerAMG on that matrix; the interface
    of :class:`SingleEngine`.

    One application relaxes on the matrix, corrects from the linear elements of the same
    mesh, and relaxes again.  The coarse matrix is the Galerkin product ``P^T A P`` with the
    interpolation ``P`` of a :class:`PTransfer`: the matrix of the same form on the linear
    elements, by the same quadrature, with the identity on the essential dofs.  The
    correction is one V-cycle of BoomerAMG on it, with the settings the solver has for its
    hierarchies, and the relaxation is the one those settings give a finest level: one sweep
    of l1-Jacobi (of Jacobi with ``relax=7``) with the weight ``relax_wt``.

    A set-up is then the product and a hierarchy of a matrix with an eighth of the rows and
    a twentieth of the entries (quadratic hexahedra), where BoomerAMG on the matrix itself
    coarsens it, builds an interpolation for it and forms such a product with that.  The
    cycle costs what BoomerAMG's does, two products with the matrix and the coarse levels,
    and the iteration counts were the same (``docs/source/guide/gpu.rst``, Single
    precision).  The steps of the cycle are calls of the library made from Python, a dozen
    an application.

    Where the transfer comes with interpolations from coarser meshes (``transfer.h``, a
    list of :class:`PTransfer` from :func:`hippymfem.fem.ptransfer.refinement`: the solver
    parameter ``pmg_h`` on a mesh that knows the meshes it was refined from), the correction
    from the linear elements is a V-cycle over those meshes instead: on each, the same
    relaxation, the Galerkin product with the interpolation from the next one (made by the
    library at every set-up, with its product ``A P`` kept for the residual), and BoomerAMG
    on the matrix of the coarsest mesh only.  Those matrices have the 27 entries a row of
    the linear elements at every level, and their ranks the neighbours of the mesh.

    A mesh whose next one is three or more times coarser a side (a cut by three or by five
    in :func:`applications.geothermal.model.cube_mesh`) relaxes by a polynomial instead of
    one sweep (:data:`PMG_H_SMOOTH`): the correction from such a mesh leaves more of the
    error to the relaxation, and one sweep before and after a cut by five took the
    Jacobian's solves from 6 to 9 iterations on 512 GCDs.
    """

    def __init__(self, S, transfer, parameters):
        self.matrix, self.lib, self.transfer = S, S.lib, transfer
        H, lib = S.lib.H, S.lib
        self.amg = self.pcg = self.coarse = self.ap = self._product = None
        #: the levels of the coarser meshes (see the class), and the matrix of the hierarchy
        self._h, self._bottom = [], None
        #: the first of those levels that lives in host memory with the library on a device
        #: (:data:`PMG_H_HOST`; the number of levels where the hierarchy alone does, ``None``
        #: where nothing does), the copy of its matrix there, the two vectors that carry a
        #: cycle across, and where the device has the same two
        self._host_at = self._host_top = self._host_b = self._host_x = self._host_io = None
        self._in_host, self._timing = False, set()
        #: the matrix's own products, where it came with them (not this engine's to destroy)
        self._given = None
        if lib.no_pmg:
            raise RuntimeError("no p-multigrid in this single-precision hypre: %s" % lib.no_pmg)
        self.converged, self.refused, self.error = False, False, 0
        #: an exception of the cycle, kept until the solve is over (nothing may leave a
        #: function that the library calls)
        self.failure = None
        self.b, self.x = S.vector(), S.vector(columns=True)
        self._t, self._d = S.vector(), S.vector()
        #: the first cycle of a solve, kept for the second (:meth:`_cycle`), and which of
        #: the two is next (0: neither)
        self._kept, self._nth = S.vector(columns=True), 0
        P = transfer.P
        self._rc, self._ec = P.vector(columns=True), P.vector(columns=True)
        relax, weight = _relaxation(lib, parameters)
        #: the last matrix gathered by the engine, where it is in host memory and the
        #: solver's ``seq_threshold`` gathers it (:class:`_HostGathered`), and the entries
        #: of the two vectors of a cycle on this rank
        self._gathered = self._gathered_io = None
        mfemconfig.hypre_pool_open()
        try:
            given = getattr(S, "products", None)
            if given is not None and given.get("C") is not None:
                # made with the matrix, from its element matrices
                self._given = given
                self.coarse = given["C"].par
                self.ap = given["Q"].par if PMG_KEEP_AP and given.get("Q") is not None else None
            else:
                if PMG_KEEP_AP:
                    coarse, ap = lib.galerkin(S.par, P.par, keep=True)
                    self.ap = ctypes.c_void_p(ap) if ap else None
                else:
                    coarse = lib.galerkin(S.par, P.par)
                self.coarse = ctypes.c_void_p(coarse) if coarse else None
                # (the product of one rank can fail alone)
                lib.check(0 if coarse else 1, "ParCSRMatrixRAPKT", S.comm)
            # the levels of the coarser meshes, where the transfer has them: each with its
            # matrix, the product of that matrix with the interpolation from the next
            # mesh, its relaxation and its vectors; the hierarchy is then of the last
            # Galerkin product
            self._levels(self.coarse, (self._rc, self._ec), P, getattr(transfer, "h", None) or (),
                         parameters, relax, weight)
            # the relaxation: D / w, with D the l1 norms of the rows or the diagonal
            self._relaxation_of(S.par, self._d, relax, weight)
            pcg, self._product = lib.pcg(S)
            self.pcg = pcg
            H.HYPRE_PCGSetTwoNorm(pcg, 0)
            H.HYPRE_PCGSetPrintLevel(pcg, 0)
            H.HYPRE_PCGSetLogging(pcg, 0)
            # (through a weak reference: a callback that held the engine would keep it,
            # its matrix and its hierarchy until a collection of cycles)
            me = weakref.ref(self)
            self._call = _PRECOND(lambda _data, A, b, x: me()._cycle(A, b, x))
            self._none = _PRECOND(lambda *a: 0)
            H.HYPRE_PCGSetPrecond(pcg, ctypes.cast(self._call, ctypes.c_void_p),
                                  ctypes.cast(self._none, ctypes.c_void_p), None)
            lib.check(H.HYPRE_ParCSRPCGSetup(pcg, S.par, self.b.par, self.x.par), "PCGSetup",
                      S.comm)
            if lib.device:
                bridge.synchronize()
        finally:
            mfemconfig.hypre_pool_close()

    def _levels(self, top, below, above, transfers, parameters, relax, weight):
        """The levels of the coarser meshes under the matrix ``top`` of the linear elements
        (a pointer of the library), and the hierarchy of the last matrix: each level with
        its matrix, the product of that matrix with the interpolation from the next mesh
        (``transfers``, the next coarser first), its relaxation and its vectors.  ``below``
        are the two vectors that a cycle hands the first level, ``above`` what knows their
        layout (the interpolation, or the matrix, whose columns they are)."""
        lib, H, comm = self.lib, self.lib.H, self.matrix.comm
        transfers = list(transfers)
        host_rows = int(parameters["pmg_h_host"] or 0) if "pmg_h_host" in parameters else PMG_H_HOST
        if not (lib.device and lib.host_levels):
            host_rows = 0
        gathered, ranks = gather_threshold(parameters), comm.size
        # What host memory saves a small level is the device's latency around its exchanges
        # between ranks; what it costs is one core's arithmetic.  So: the meshes of at most
        # host_rows rows a rank go there, with everything under them -- and not where the
        # matrix under them that the hierarchy gathers, which every rank then solves whole
        # on its core and without an exchange, has more rows than that in all (with a
        # gathered matrix of 15 625 rows an iteration took 1.0 to 1.4 ms longer than with
        # every level on the device, on 32 and on 256 MI250X GCDs; with one of 729 rows
        # 1.1 to 1.2 ms less, on 128 and on 512; with one of 4913 rows 0.6 to 0.7 ms less,
        # on 1024 to 4096).  The last matrix alone stays on the device: it has no
        # exchange to save.
        last = int((transfers[-1].P if transfers else above).global_cols)
        if 0 < last <= gathered and last > host_rows:
            host_rows = 0

        def on_host(rows):
            return host_rows > 0 and rows <= host_rows * ranks

        bottom, host = top, False
        for tr in transfers:
            if not host and on_host(int(above.global_cols)):
                _trace(comm, "set-up: level %d (%d rows) to host memory" % (len(self._h), int(above.global_cols)))
                bottom, below, host = self._enter_host(bottom, above, below), None, True
            if host:
                _trace(comm, "set-up: level %d, the interpolation to host memory" % len(self._h))
                Pk, Rk = tr.host()
                _trace(comm, "set-up: level %d, the Galerkin product in host memory" % len(self._h))
                lib.to_host()
                try:
                    C, Q = lib.galerkin(bottom, Pk, keep=True, host=True)
                finally:
                    lib.to_device()
                _trace(comm, "set-up: level %d, the product is made" % len(self._h))
            else:
                Pk, Rk = tr.P.par, tr.R
                C, Q = lib.galerkin(bottom, Pk, keep=True)
            lib.check(0 if C else 1, "ParCSRMatrixRAPKT", comm)
            level = {"transfer": tr, "P": Pk, "R": Rk, "A": bottom, "host": host,
                     "ap": ctypes.c_void_p(Q) if Q else None, "C": ctypes.c_void_p(C),
                     "t": tr.P.vector(host=host), "d": tr.P.vector(host=host),
                     "rc": tr.P.vector(columns=True, host=host),
                     "ec": tr.P.vector(columns=True, host=host), "steps": 0}
            self._h.append(level)
            # how many times coarser a side the next mesh is
            f = int(getattr(tr, "cut", None) or 2)
            if PMG_H_SMOOTH == "chebyshev" and f >= 3 and relax != 7:
                # a polynomial on the matrix scaled by the l1 norms of its rows
                level.update(steps=f, band=(1.2 / (f * f), 1.0), p=tr.P.vector(host=host),
                             z=tr.P.vector(host=host))
                self._relaxation_of(bottom, level["d"], relax, 1.0, host=host)
            else:
                self._relaxation_of(bottom, level["d"], relax, weight, host=host)
            bottom, below, above = level["C"], (level["rc"], level["ec"]), tr.P
        self._bottom = bottom
        rows = int(above.global_cols)
        if (host or not lib.device) and 0 < rows <= gathered:
            # in host memory and gathered: by the engine, not by hypre's own code for
            # it (see _HostGathered)
            _trace(comm, "set-up: the last matrix (%d rows) gathered in host memory" % rows)
            self._gathered = _HostGathered(lib, comm, bottom, int(above.col_starts[0]), parameters)
            self._gathered_io = (below[0].data, below[1].data, below[1].par.value, int(below[0].n))
            code = self._gathered.code
        else:
            # (beside a device, a hierarchy in host memory goes on between the ranks
            # to its end: hypre's gathering there is the code that _HostGathered is
            # in place of)
            self.amg = _boomeramg(lib, parameters, gather=not host)
            if host:
                _trace(comm, "set-up: BoomerAMG on the last matrix (%d rows), in host memory" % rows)
                lib.to_host()
            try:
                code = H.HYPRE_BoomerAMGSetup(self.amg, bottom, below[0].par, below[1].par)
            finally:
                if host:
                    lib.to_device()
        if host:
            _trace(comm, "set-up: the last matrix is set up (code %d)" % int(code))
        lib.check(code, "BoomerAMGSetup", comm)

    def _relaxation_of(self, A, d, relax, weight, host=False):
        """Into the vector ``d`` of the library: what a relaxation on the matrix ``A``
        divides by, the l1 norms of the rows (the diagonal with ``relax=7``) over the
        weight.  ``host=True``: for a matrix and a vector in host memory with the library
        on a device."""
        lib, H = self.lib, self.lib.H
        n, device = d.n, lib.device and not host
        if host:
            lib.to_host()
        try:
            if relax == 7:
                H.hypre_CSRMatrixExtractDiagonal(lib.pointer(A.value, "par_diag"), d.data, 0)
            else:
                norms = ctypes.c_void_p()
                code = H.hypre_ParCSRComputeL1Norms(A, 1, None, ctypes.byref(norms))
                if n and norms.value:
                    if device:
                        bridge.copy_device(d.data, norms.value, 4 * n)
                        bridge.synchronize()
                    else:
                        ctypes.memmove(d.data, norms.value, 4 * n)
                if norms.value:
                    H.hypre_Free(norms, DEVICE if device else HOST)
                lib.check(code, "ParCSRComputeL1Norms", self.matrix.comm)
            H.hypre_ParVectorScale(1.0 / weight, d.par)
        finally:
            if host:
                lib.to_device()

    def _enter_host(self, bottom, above, below):
        """The matrix ``bottom`` of a level, and with it everything under it, from the
        device to host memory (:data:`PMG_H_HOST`): returns the copy there.  ``above`` is
        the interpolation whose columns are that level, ``below`` the two vectors of the
        device that a cycle hands it and takes its correction in."""
        lib = self.lib
        copy = lib.host_copy(bottom)
        lib.check(0 if copy else 1, "ParCSRMatrixClone (host)", self.matrix.comm)
        self._host_top = ctypes.c_void_p(copy)
        self._host_b = above.vector(columns=True, host=True)
        self._host_x = above.vector(columns=True, host=True)
        self._host_at = len(self._h)
        self._host_io = (below[0].data, below[1].data, 4 * below[0].n)
        HOST_LEVELS.update(level=self._host_at, rows=int(above.global_cols))
        return self._host_top

    def _correct(self, k, b, x):
        """``x`` = the correction for the right-hand side ``b`` on level ``k`` of the
        coarser meshes (0: the linear elements of the mesh itself); ``x`` arrives zero.
        A V-cycle from there down: relax, correct from the next mesh, relax; under the
        last mesh, one V-cycle of the hierarchy."""
        lib, H = self.lib, self.lib.H
        if k in PMG_TIMERS and k not in self._timing:
            # (a probe's clock on this level and everything under it)
            import time

            if lib.device:
                bridge.synchronize()
            t0 = time.perf_counter()
            self._timing.add(k)
            try:
                self._correct(k, b, x)
            finally:
                self._timing.discard(k)
            if lib.device:
                bridge.synchronize()
            tally = TIMERS.setdefault((type(self).__name__, k), [0.0, 0])
            tally[0] += time.perf_counter() - t0
            tally[1] += 1
            return
        if k == self._host_at and not self._in_host:
            # the levels from here down live in host memory: the right-hand side goes
            # there, the cycle runs there, and the correction comes back
            src, dst, nbytes = self._host_io
            hb, hx = self._host_b, self._host_x
            say = PMG_TRACE and _TRACED["cycles"] < 3
            if say:
                _TRACED["cycles"] += 1
                _trace(self.matrix.comm, "cycle %d: the right-hand side of level %d to the host" % (_TRACED["cycles"], k))
            if nbytes:
                H.hypre_Memcpy(hb.data, src, nbytes, HOST, DEVICE)
            self._in_host = True
            lib.to_host()
            try:
                lib.zero(hx.par)
                self._correct(k, hb.par, hx.par)
            finally:
                lib.to_device()
                self._in_host = False
            if say:
                _trace(self.matrix.comm, "cycle %d: the correction back to the device" % _TRACED["cycles"])
            if nbytes:
                H.hypre_Memcpy(dst, hx.data, nbytes, DEVICE, HOST)
            if say:
                _trace(self.matrix.comm, "cycle %d: done" % _TRACED["cycles"])
            return
        say = PMG_TRACE and self._in_host and _TRACED["cycles"] <= 1
        if k == len(self._h):
            if say:
                _trace(self.matrix.comm, "  BoomerAMG on the last matrix")
            if self._gathered is not None:
                bp, xp, xpar, n = self._gathered_io
                self._gathered.solve(bp, xp, n)
                # (the entries were written past hypre, which had the vector down as zero)
                zeros = lib.off.get("parvec_all_zeros")
                if zeros is not None:
                    ctypes.c_int.from_address(int(xpar) + int(zeros)).value = 0
            else:
                H.HYPRE_BoomerAMGSolve(self.amg, self._bottom, b, x)
            if say:
                _trace(self.matrix.comm, "  BoomerAMG returned")
            return
        level = self._h[k]
        t, d, rc, ec = level["t"].par, level["d"].par, level["rc"].par, level["ec"].par
        if say:
            _trace(self.matrix.comm, "  level %d: relaxation (%d steps)" % (k, level["steps"]))
        if level["steps"]:
            H.hypre_ParVectorCopy(b, t)
            self._polynomial(level, t, x)
        else:
            lib.divide_add(b, d, x)
        if say:
            _trace(self.matrix.comm, "  level %d: residual" % k)
        H.hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, level["A"], x, 1.0, b, t)
        if say:
            _trace(self.matrix.comm, "  level %d: restriction" % k)
        H.hypre_ParCSRMatrixMatvec(1.0, level["R"], t, 0.0, rc)
        lib.zero(ec)
        self._correct(k + 1, rc, ec)
        if say:
            _trace(self.matrix.comm, "  level %d: interpolation and the rest" % k)
        H.hypre_ParCSRMatrixMatvec(1.0, level["P"], ec, 1.0, x)
        if level["ap"] is not None:
            H.hypre_ParCSRMatrixMatvec(-1.0, level["ap"], ec, 1.0, t)
        else:
            H.hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, level["A"], x, 1.0, b, t)
        if level["steps"]:
            self._polynomial(level, t, x)
        else:
            lib.divide_add(t, d, x)

    def _polynomial(self, level, r, x):
        """``x += q(D^-1 A) D^-1 r`` for the residual ``r`` of ``x`` on a level, ``q`` the
        polynomial of the level's Chebyshev steps on its band of ``D^-1 A`` (``D`` the l1
        norms of the rows): one map, the same before and after the correction, so the
        cycle stays symmetric.  ``r`` is used up."""
        lib, H = self.lib, self.lib.H
        A, d, p, z = level["A"], level["d"].par, level["p"].par, level["z"].par
        lo, hi = level["band"]
        theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
        sigma = theta / delta
        rho = 1.0 / sigma
        lib.zero(p)
        lib.divide_add(r, d, p)
        H.hypre_ParVectorScale(1.0 / theta, p)
        H.hypre_ParVectorAxpy(1.0, p, x)
        for _ in range(1, int(level["steps"])):
            H.hypre_ParCSRMatrixMatvec(-1.0, A, p, 1.0, r)
            new = 1.0 / (2.0 * sigma - rho)
            H.hypre_ParVectorScale(new * rho, p)
            lib.zero(z)
            lib.divide_add(r, d, z)
            H.hypre_ParVectorAxpy(2.0 * new / delta, z, p)
            H.hypre_ParVectorAxpy(1.0, p, x)
            rho = new

    def _cycle(self, A, b, x):
        """``x = C b``, on vectors of the library, as hypre's PCG asks for it.  ``x``
        arrives zero (the PCG clears it), so the first relaxation needs no product."""
        try:
            lib, H, t, d = self.lib, self.lib.H, self._t.par, self._d.par
            if self._nth == 2:
                # hypre's PCG applies the preconditioner to the right-hand side, for the
                # norm that it stops on, and then to the first residual, which from a
                # zero guess is that same vector: the cycle before this one, as it was
                # (15 to 20 ms of a solve of 230 to 300)
                self._nth = 0
                H.hypre_ParVectorCopy(self._kept.par, x)
                return 0
            lib.divide_add(b, d, x)
            H.hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, x, 1.0, b, t)
            H.hypre_ParCSRMatrixMatvec(1.0, self.transfer.R, t, 0.0, self._rc.par)
            lib.zero(self._ec.par)
            self._correct(0, self._rc.par, self._ec.par)
            H.hypre_ParCSRMatrixMatvec(1.0, self.transfer.P.par, self._ec.par, 1.0, x)
            if self.ap is not None:
                # the residual of the corrected iterate from that of the relaxed one
                H.hypre_ParCSRMatrixMatvec(-1.0, self.ap, self._ec.par, 1.0, t)
            else:
                H.hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, x, 1.0, b, t)
            lib.divide_add(t, d, x)
            if self._nth == 1:
                self._nth = 2
                H.hypre_ParVectorCopy(x, self._kept.par)
        except BaseException as e:                               # noqa: BLE001
            self.failure = e
            return 1
        return 0

    def solve(self, x, b, rel_tol, abs_tol, max_iter):
        # (the first two cycles of a solve are one: see _cycle.  The guess is zero in
        # every solve of an engine, and the PCG's norm is the preconditioner's.)
        self._nth = 1
        try:
            out = super(PMGEngine, self).solve(x, b, rel_tol, abs_tol, max_iter)
        finally:
            self._nth = 0
        failed, self.failure = self.failure, None
        self.lib.agree(self.matrix.comm, failed)
        return out

    def destroy(self):
        H = self.lib.H
        if self.pcg is not None:
            H.HYPRE_ParCSRPCGDestroy(self.pcg)
            self.pcg = None
        self._product = None                       # (the product's callback, after its PCG)
        if self.amg is not None:
            # (a hierarchy in host memory is taken down as it was set up)
            host = getattr(self, "_host_at", None) is not None
            if host:
                self.lib.to_host()
            try:
                H.HYPRE_BoomerAMGDestroy(self.amg)
            finally:
                if host:
                    self.lib.to_device()
            self.amg = None
        if getattr(self, "_gathered", None) is not None:
            self._gathered.destroy()
        self._gathered = self._gathered_io = None
        if getattr(self, "_host_top", None) is not None:
            H.hypre_ParCSRMatrixDestroy(self._host_top)
        for name in ("_host_b", "_host_x"):
            if getattr(self, name, None) is not None:
                getattr(self, name).destroy()
            setattr(self, name, None)
        self._host_at = self._host_top = self._host_io = None
        for level in getattr(self, "_h", None) or ():
            # (the level's own matrix "A" is the one above's "C", or the engine's "coarse")
            for name in ("C", "ap"):
                if level.get(name) is not None:
                    H.hypre_ParCSRMatrixDestroy(level[name])
                level[name] = None
            for name in ("t", "d", "rc", "ec", "p", "z"):
                if level.get(name) is not None:
                    level[name].destroy()
                level[name] = None
        self._h, self._bottom = [], None
        for name in ("coarse", "ap"):
            if getattr(self, name, None) is not None and getattr(self, "_given", None) is None:
                H.hypre_ParCSRMatrixDestroy(getattr(self, name))
            setattr(self, name, None)
        self._given = None
        for name in ("b", "x", "_t", "_d", "_rc", "_ec", "_kept"):
            v = getattr(self, name, None)
            if v is not None:
                v.destroy()
            setattr(self, name, None)


class HCycle(PMGEngine):
    """One multigrid V-cycle over the meshes that a mesh was refined from, for a matrix on
    the linear elements of that mesh, in the single-precision library: the levels that a
    :class:`PMGEngine` has under its linear elements, on their own.  For a matrix that
    stays, such as a prior's elliptic operator: the Galerkin products are made once.

    ``S`` is the matrix (a :class:`SingleParMatrix`, for instance of
    :func:`matrix_blocks`), ``transfers`` the interpolations from the coarser meshes, the
    next coarser first (:class:`PTransfer` of :func:`hippymfem.fem.ptransfer.refinement`),
    and ``parameters`` a solver's: its relaxation, the hierarchy of the last matrix, the
    levels in host memory (:data:`PMG_H_HOST`).

    :meth:`Mult` applies the cycle to one of MFEM's vectors, from zero, as a preconditioner
    of MFEM's does.  :meth:`polynomial` applies a fixed Chebyshev polynomial of cycles,
    wholly in the library: its residuals are products with the single-precision matrix,
    and a solve converts two vectors where a cycle at a time converts two each.  Such a
    polynomial is one linear map up to the rounding of single precision, about 1e-7 of
    its result: a preconditioner, not a solve to tolerance.
    """

    #: (as MFEM's solvers have it; this one starts from zero whatever it is given)
    iterative_mode = False

    def __init__(self, S, transfers, parameters):
        transfers = list(transfers)
        if not transfers:
            raise ValueError("HCycle needs at least one coarser mesh")
        self.matrix, self.lib, self.transfer = S, S.lib, None
        lib = S.lib
        if lib.no_pmg:
            raise RuntimeError("no multigrid over meshes in this single-precision hypre: %s" % lib.no_pmg)
        self.amg = self.pcg = self.coarse = self.ap = self._product = None
        self._h, self._bottom = [], None
        self._host_at = self._host_top = self._host_b = self._host_x = self._host_io = None
        self._in_host, self._timing = False, set()
        self._gathered = self._gathered_io = None
        #: (the matrix is the caller's: nothing of it is this cycle's to destroy)
        self._given = {}
        self.converged, self.refused, self.error, self.failure = True, False, 0, None
        self.b, self.x = S.vector(), S.vector(columns=True)
        #: the vectors of :meth:`polynomial`: the step, the iterate, the product
        self._poly = [S.vector() for _ in range(3)]
        self._transfers = transfers
        relax, weight = _relaxation(lib, parameters)
        mfemconfig.hypre_pool_open()
        try:
            self._levels(S.par, (self.b, self.x), S, transfers, parameters, relax, weight)
            if lib.device:
                bridge.synchronize()
        finally:
            mfemconfig.hypre_pool_close()

    @property
    def rows(self):
        """The global rows of the coarser meshes in use, the next coarser first."""
        return [int(lv["transfer"].P.global_cols) for lv in self._h]

    def Mult(self, b, x):
        """``x = C b`` for two of MFEM's vectors."""
        self.b.set(b)
        self.lib.zero(self.x.par)
        self._correct(0, self.b.par, self.x.par)
        self.x.get(x)

    def polynomial(self, x, b, degree, interval):
        """``x = q(C A) C b`` for two of MFEM's vectors: ``degree`` steps of the Chebyshev
        iteration on ``interval``, the eigenvalues of the cycle ``C`` times the matrix
        (:class:`~hippymfem.algorithms.linSolvers.ChebyshevSolver`, which estimates the
        interval and hands its solves here)."""
        lib, H, A = self.lib, self.lib.H, self.matrix.par
        r, z = self.b.par, self.x.par
        d, y, t = (v.par for v in self._poly)
        lo, hi = interval
        theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
        self.b.set(b)
        lib.zero(z)
        self._correct(0, r, z)
        H.hypre_ParVectorCopy(z, d)
        H.hypre_ParVectorScale(1.0 / theta, d)
        H.hypre_ParVectorCopy(d, y)
        if delta > 0.0 and int(degree) > 1:
            sigma = theta / delta
            rho = 1.0 / sigma
            for _ in range(1, int(degree)):
                H.hypre_ParCSRMatrixMatvec(-1.0, A, d, 1.0, r)
                lib.zero(z)
                self._correct(0, r, z)
                new = 1.0 / (2.0 * sigma - rho)
                H.hypre_ParVectorScale(new * rho, d)
                H.hypre_ParVectorAxpy(2.0 * new / delta, z, d)
                H.hypre_ParVectorAxpy(1.0, d, y)
                rho = new
        self._poly[1].get(x)

    def solve(self, x, b, rel_tol, abs_tol, max_iter):
        raise NotImplementedError("HCycle is a cycle, not a solver: Mult or polynomial")

    def destroy(self):
        for v in getattr(self, "_poly", None) or ():
            v.destroy()
        self._poly = []
        super(HCycle, self).destroy()
        self._transfers = []

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Parallel direct (exact) linear solvers.

hIPPYlib uses ``PETScLUSolver``, a distributed MUMPS factorization, whenever an
exact forward, adjoint or incremental solve is wanted: an exact Hessian action
lets a randomized eigensolver or a Newton-CG tolerance study measure the
discretization rather than the Krylov tolerance.  PyMFEM's standard parallel
build links none of MUMPS, SuperLU_dist, STRUMPACK or PETSc, so this module
supplies the capability from the Python side, by two exact routes:

:class:`ReplicatedLUSolver`
    Gathers the distributed matrix onto every rank and factorizes it with
    scipy's SuperLU.  Needs nothing beyond scipy, works on any rank count, and
    matches a distributed direct solve to round-off.  The factorization is
    redundant: it costs one serial factorization of the whole matrix in time
    and ``O(nnz(L+U))`` per rank in memory, the standard "redundant coarse
    solve" trade and the right one up to a few hundred thousand unknowns.
    Beyond that use a Krylov solver.

:class:`PETScLUSolver`
    A distributed factorization when petsc4py is importable, picking the best
    package the local PETSc was built with (MUMPS, SuperLU_dist, STRUMPACK,
    PaStiX) and falling back to PETSc's own ``redundant`` + ``lu``, which is
    exact as well.  :class:`PETScKrylovSolver` exposes PETSc's Krylov methods
    and preconditioners for the same matrices.

**Import order matters for the PETSc route.**  petsc4py and PyMFEM are linked
against MPI libraries that must be resolved in that order; importing petsc4py
after ``mfem`` raises ``ImportError: libmpi_mpifh.so.40: undefined symbol:
mpi_conversion_fn_null_``.  Set ``HIPPYMFEM_PETSC=1`` before importing
``hippymfem`` (which then imports petsc4py first), or import petsc4py yourself at
the top of the script.  :func:`petsc_available` reports what happened instead of
guessing.
"""

import numpy as np
from mpi4py import MPI

from .._petsc import petsc_available, preload_petsc
from ..common.linalg import as_matrix, hypre_to_scipy
from ..common.parvector import ParVector
from .linSolvers import _SolverBase, _all_finite

__all__ = [
    "ReplicatedLUSolver",
    "PETScLUSolver",
    "PETScKrylovSolver",
    "petsc_available",
    "preload_petsc",
    "gather_matrix",
]

#: Global size past which a replicated factorization is refused rather than
#: quietly allocating the whole of ``L+U`` on every rank.  Override per solver
#: with ``max_global_size``.
DEFAULT_MAX_GLOBAL_SIZE = 400000


# ------------------------------------------------------------------- gathering
def gather_matrix(A, comm=None, format="csc"):
    """Assemble a distributed ``HypreParMatrix`` into a scipy matrix on every rank.

    Blocks are ordered by their true row offset rather than by rank, so this does
    not assume that hypre's partition runs in rank order.
    """
    import scipy.sparse as sp

    A = as_matrix(A)
    comm = comm if comm is not None else MPI.COMM_WORLD
    loc = hypre_to_scipy(A)
    ncols = int(A.GetGlobalNumCols())
    first = int(A.GetRowPartArray()[0])
    blocks = comm.allgather(
        (first, loc.indptr.copy(), loc.indices.copy(), loc.data.copy(),
         loc.shape[0]))
    blocks.sort(key=lambda t: t[0])
    mats = [sp.csr_matrix((d, j, ip), shape=(nr, ncols))
            for _, ip, j, d, nr in blocks]
    return sp.vstack(mats, format=format)


class _Base(_SolverBase):
    """The shared solver base with the direct solvers' defaults: an exact solve
    counts as one iteration, and the Krylov tolerance (PETSc's solver) is tight."""

    def __init__(self, comm=None):
        super(_Base, self).__init__(comm)
        self.parameters["rel_tolerance"] = 1e-14
        self.iterations = 1


# ------------------------------------------------------------- replicated LU
class ReplicatedLUSolver(_Base):
    """Exact sparse LU on every rank, of the gathered matrix.

    A drop-in stand-in for hIPPYlib's ``PETScLUSolver`` that needs no external
    package.  ``solve`` gathers the right-hand side, applies the factorization
    and keeps this rank's slice, so the result is bit-identical on every rank and
    independent of the rank count.

    Parameters
    ----------
    comm : mpi4py communicator
    max_global_size : int
        Refuse matrices larger than this.  The default keeps a typo from turning
        into an out-of-memory kill; raise it deliberately if the fill fits.
    """

    def __init__(self, comm=None, max_global_size=None, permc_spec=None):
        super(ReplicatedLUSolver, self).__init__(comm)
        self.max_global_size = int(max_global_size if max_global_size is not None
                                   else DEFAULT_MAX_GLOBAL_SIZE)
        self.permc_spec = permc_spec
        self._lu = None
        self._n = None

    def set_operator(self, A):
        import scipy.sparse.linalg as sla

        A = as_matrix(A)
        self.A = A
        n = int(A.GetGlobalNumRows())
        if n != int(A.GetGlobalNumCols()):
            raise ValueError("a direct solve needs a square matrix (%d x %d)"
                             % (n, A.GetGlobalNumCols()))
        if n > self.max_global_size:
            raise RuntimeError(
                "ReplicatedLUSolver refuses a %d x %d matrix: the factorization "
                "is stored on every rank.  Raise max_global_size if the fill "
                "fits, or use KrylovSolver(comm, 'cg', 'amg')." % (n, n))
        self._n = n
        G = gather_matrix(A, self.comm, format="csc")
        kw = {} if self.permc_spec is None else {"permc_spec": self.permc_spec}
        self._lu = sla.splu(G, **kw)
        self._template = ParVector(self.comm, A.Height())
        self.iterations = 1
        self.converged = True
        return self

    SetOperator = set_operator

    def _solve(self, x, b, trans):
        if self._lu is None:
            raise RuntimeError("set_operator must be called before solve")
        full = b.allgather()
        sol = self._lu.solve(full, trans=trans)
        if not np.isfinite(sol).all():
            self.converged = False
            if self.parameters["error_on_nonconvergence"]:
                raise RuntimeError("the LU solve produced non-finite values; the "
                                   "matrix is probably singular")
        else:
            self.converged = True
        lo, hi = x.owner_range
        x.array[:] = sol[lo:hi]
        return 1

    def solve(self, x, b):
        """``x = A^{-1} b``."""
        return self._solve(x, b, "N")

    def solveTranspose(self, x, b):
        """``x = A^{-T} b``."""
        return self._solve(x, b, "T")

    def mult(self, x, y):
        """``y = A^{-1} x``, so the solver can be used as an operator."""
        self.solve(y, x)
        return y

    def __repr__(self):
        return "ReplicatedLUSolver(n=%s, ranks=%d)" % (self._n, self.comm.size)


# -------------------------------------------------------------------- PETSc
def _require_petsc():
    mod = preload_petsc()
    if mod is None:
        raise RuntimeError("petsc4py is not usable here: %s" % petsc_available()[1])
    return mod


def to_petsc_matrix(A, comm=None):
    """A PETSc ``MPIAIJ``/``SEQAIJ`` view of a ``HypreParMatrix``.

    hypre hands out local rows with global column indices, which is exactly the
    CSR form ``MatMPIAIJSetPreallocationCSR`` wants, so this is a copy and no
    index translation.
    """
    PETSc = _require_petsc()
    A = as_matrix(A)
    comm = comm if comm is not None else MPI.COMM_WORLD
    csr = hypre_to_scipy(A)
    M = PETSc.Mat().createAIJ(
        size=((A.Height(), int(A.GetGlobalNumRows())),
              (A.Width(), int(A.GetGlobalNumCols()))),
        csr=(csr.indptr.astype(PETSc.IntType),
             csr.indices.astype(PETSc.IntType),
             np.ascontiguousarray(csr.data)),
        comm=comm)
    M.assemble()
    return M


#: Factor packages tried in order; the first the local PETSc can set up wins.
PETSC_PARALLEL_PACKAGES = ("mumps", "superlu_dist", "strumpack", "pastix")
PETSC_SERIAL_PACKAGES = ("mumps", "umfpack", "klu", "superlu", "petsc")


class _PETScBase(_Base):
    def __init__(self, comm=None):
        super(_PETScBase, self).__init__(comm)
        self._PETSc = _require_petsc()
        self._ksp = None
        self._mat = None
        self._prefix = "hippymfem_%d_" % id(self)

    def _vec(self, arr, glob):
        return self._PETSc.Vec().createWithArray(
            arr, size=(arr.size, int(glob)), comm=self.comm)

    def solve(self, x, b):
        if self._ksp is None:
            raise RuntimeError("set_operator must be called before solve")
        if not self.parameters["nonzero_initial_guess"]:
            x.zero()
        else:
            self._ksp.setInitialGuessNonzero(True)
        n = int(self.A.GetGlobalNumRows())
        bv = self._vec(np.ascontiguousarray(b.array), n)
        xv = self._vec(x.array, n)
        self._ksp.solve(bv, xv)
        reason = self._ksp.getConvergedReason()
        self.iterations = int(self._ksp.getIterationNumber()) or 1
        self.converged = reason > 0
        if self.converged and not _all_finite(x):
            self.converged = False
        if not self.converged and self.parameters["error_on_nonconvergence"]:
            raise RuntimeError("PETSc solve failed: converged reason %d after "
                               "%d iterations" % (reason, self.iterations))
        return self.iterations

    def mult(self, x, y):
        self.solve(y, x)
        return y


class PETScLUSolver(_PETScBase):
    """Distributed exact LU through PETSc.

    ``package="auto"`` tries the parallel factorizations the local PETSc was
    built with and, if none is available, falls back to ``redundant`` with an
    inner ``lu``: every rank factorizes the gathered matrix, which is still an
    exact solve.  :attr:`package_used` records what was chosen, so a script can
    report it rather than assume it.
    """

    def __init__(self, comm=None, package="auto"):
        super(PETScLUSolver, self).__init__(comm)
        self.package = package
        self.package_used = None

    def _candidates(self):
        if self.package != "auto":
            return [self.package]
        if self.comm.size > 1:
            return list(PETSC_PARALLEL_PACKAGES) + ["redundant"]
        return list(PETSC_SERIAL_PACKAGES)

    def _try(self, name):
        """Build a KSP for one package; ``None`` when the build does not have it.

        Whether a package exists is a property of the PETSc build, so every rank
        reaches the same verdict; the result is reduced anyway so that a
        disagreement cannot leave the ranks on different code paths.
        """
        PETSc = self._PETSc
        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOperators(self._mat)
        ksp.setType("preonly")
        ok = True
        try:
            if name == "redundant":
                opts = PETSc.Options()
                opts[self._prefix + "pc_type"] = "redundant"
                opts[self._prefix + "redundant_ksp_type"] = "preonly"
                opts[self._prefix + "redundant_pc_type"] = "lu"
                ksp.setOptionsPrefix(self._prefix)
                ksp.setFromOptions()
            else:
                pc = ksp.getPC()
                pc.setType("lu")
                pc.setFactorSolverType(name)
            ksp.setUp()
        except Exception:
            ok = False
        if not bool(self.comm.allreduce(int(ok), op=MPI.MIN)):
            try:
                ksp.destroy()
            except Exception:
                pass
            return None
        return ksp

    def set_operator(self, A):
        A = as_matrix(A)
        self.A = A
        self._mat = to_petsc_matrix(A, self.comm)
        tried = []
        for name in self._candidates():
            ksp = self._try(name)
            if ksp is not None:
                self._ksp = ksp
                self.package_used = name
                break
            tried.append(name)
        if self._ksp is None:
            raise RuntimeError("no PETSc direct factorization is available; "
                               "tried %s" % (tried,))
        self._template = ParVector(self.comm, A.Height())
        self.iterations = 1
        self.converged = True
        return self

    SetOperator = set_operator

    def solveTranspose(self, x, b):
        """``x = A^{-T} b``."""
        if self._ksp is None:
            raise RuntimeError("set_operator must be called before solve")
        n = int(self.A.GetGlobalNumRows())
        x.zero()
        bv = self._vec(np.ascontiguousarray(b.array), n)
        xv = self._vec(x.array, n)
        self._ksp.solveTranspose(bv, xv)
        return 1

    def __repr__(self):
        return "PETScLUSolver(package=%s, ranks=%d)" % (self.package_used,
                                                        self.comm.size)


class PETScKrylovSolver(_PETScBase):
    """PETSc Krylov solver, the counterpart of hIPPYlib's ``PETScKrylovSolver``.

    ``method`` and ``precond`` are PETSc names (``"cg"``, ``"gmres"``; ``"gamg"``,
    ``"hypre"``, ``"ilu"``, ``"jacobi"``, ``"none"``).  Extra PETSc options can be
    passed as keywords, e.g. ``pc_hypre_type="boomeramg"``.
    """

    def __init__(self, comm=None, method="cg", precond="gamg", **options):
        super(PETScKrylovSolver, self).__init__(comm)
        self.method = method
        self.precond = precond
        self.options = options

    def set_operator(self, A):
        PETSc = self._PETSc
        A = as_matrix(A)
        self.A = A
        self._mat = to_petsc_matrix(A, self.comm)
        opts = PETSc.Options()
        for k, v in self.options.items():
            opts[self._prefix + k] = v
        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOperators(self._mat)
        ksp.setOptionsPrefix(self._prefix)
        ksp.setType(self.method)
        pc = ksp.getPC()
        pc.setType("none" if self.precond in (None, "none") else self.precond)
        ksp.setFromOptions()
        p = self.parameters
        ksp.setTolerances(rtol=float(p["rel_tolerance"]),
                          atol=float(p["abs_tolerance"]),
                          max_it=int(p["max_iter"]))
        self._ksp = ksp
        self._template = ParVector(self.comm, A.Height())
        return self

    SetOperator = set_operator

    def __repr__(self):
        return "PETScKrylovSolver(%s/%s)" % (self.method, self.precond)



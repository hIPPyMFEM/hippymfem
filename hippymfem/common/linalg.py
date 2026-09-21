# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Matrix utilities: products, transposes, diagonals, and scipy/hypre bridges."""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from .parvector import ParVector, _HYPRE_INT, host_readwrite, host_sync

_parcsr = None

#: Reusable wrappers for the MFEM calls that overwrite the matrix they are given with
#: a block (``GetDiag``, ``MergeDiagAndOffd``).  Built on first use, not at import: on
#: a CUDA-enabled PyMFEM an ``mfem.SparseMatrix`` cannot be constructed before an
#: ``mfem::Device`` exists.
_BLOCKS = {}


def _scratch(name):
    """One scratch ``SparseMatrix`` per role, built the first time it is needed.

    Built with the one-row constructor, not the default one: on the CUDA build of
    PyMFEM the default constructor segfaults in a process that has configured the
    device but not yet used it, while the sized one always works, and ``GetDiag``
    replaces its contents anyway.
    """
    blk = _BLOCKS.get(name)
    if blk is None:
        blk = _BLOCKS[name] = mfem.SparseMatrix(1)
    return blk


def _diag_block(A):
    """``A``'s diagonal block, wrapped in a ``SparseMatrix`` that is reused.

    ``GetDiag`` goes through MFEM's ``MakeWrapper``, which swaps a temporary into the
    object handed over and frees the previous contents, so one long-lived wrapper is
    the intended way to call it repeatedly.  A fresh ``mfem.SparseMatrix`` per call
    segfaults on a CUDA-enabled PyMFEM: its destructor reaches
    ``SparseMatrix::ClearGPUSparse`` and ``cusparseDestroySpMat`` on a descriptor it
    should not have, and the crash surfaces at some *later*, unrelated wrapper
    construction.

    The returned matrix is scratch, valid only until the next call: read or write it,
    do not keep it.
    """
    blk = _scratch("diag")
    A.GetDiag(blk)
    return blk


def _merged_block(A):
    """``A``'s local rows, both hypre blocks merged, in a wrapper that is reused.

    Scratch, like :func:`_diag_block`, and for the same reason.  **Host only**:
    ``MergeDiagAndOffd`` has hypre allocate the merged matrix in hypre's own memory
    location and then deep-copies it from the host, so with hypre on a device, and
    once ``A`` has migrated there, it reads device memory through a host pointer.
    """
    blk = _scratch("merged")
    A.MergeDiagAndOffd(blk)
    return blk


def _extra():
    """Lazily import PyMFEM's hypre/scipy helpers."""
    global _parcsr
    if _parcsr is None:
        import mfem.common.parcsr_extra as pe

        _parcsr = pe
    return _parcsr


def alias(arr):
    """A numpy view of ``arr``'s buffer that holds no reference to ``arr``.

    PyMFEM's array accessors make the owning MFEM object the numpy base.  This is for
    a view that must outlive a scratch wrapper (see :func:`_diag_block`) while
    aliasing memory owned by something else, such as hypre's own matrix.  The caller
    must keep that owner alive; nothing here can check it.
    """
    addr, ro = arr.__array_interface__["data"]
    if ro:
        raise ValueError("cannot alias a read-only array")
    ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
    return np.ctypeslib.as_array((ctype * arr.size).from_address(addr))


# --------------------------------------------------------------------- ownership
def take_ownership(mat):
    """Hand a C++-allocated ``HypreParMatrix`` to Python's garbage collector.

    PyMFEM marks only ``ParMult`` and ``Transpose`` with SWIG's ``%newobject``; the
    annotations for ``RAP``, ``Add``, ``ParAdd`` and the ``Eliminate*`` methods are
    commented out in ``mfem/_par/hypre.i``.  Their results arrive with
    ``thisown == False`` and are never freed, a matrix-sized leak per call that an
    optimizer, assembling and eliminating on every iteration, cannot afford.

    Each of those functions returns a freshly ``new``-ed ``HypreParMatrix`` that owns
    its hypre data and that nothing on the C++ side tracks, so transferring ownership
    to Python is safe.
    """
    if mat is not None:
        try:
            mat.thisown = True
        except AttributeError:
            pass
    return mat


# --------------------------------------------------------------------- products
def MatMatMult(A, B):
    """``A @ B`` for ``HypreParMatrix`` operands."""
    return mfem.ParMult(A, B)


def Transpose(A):
    """``A^T`` as a new ``HypreParMatrix``."""
    return A.Transpose()


def own(mat, *keepers):
    """Tie the lifetime of ``keepers`` to ``mat`` (appending; ``None`` is skipped).

    MFEM objects hold raw pointers into one another and PyMFEM does not always
    transfer ownership: a matrix assembled by a form belongs to the form, a
    ``ParMult`` result reads its factors, a matrix built on numpy arrays reads them,
    a grid function reads its space.  Attaching the owners to the object that needs
    them makes the Python reference graph match the C++ one, so the usual
    ``A = assemble_matrix(...)`` just works and dropping the helpers is safe.  This
    is the one place that does it; the owners are found under ``_hippymfem_owners``.
    """
    held = list(getattr(mat, "_hippymfem_owners", ()))
    held.extend(k for k in keepers if k is not None)
    mat._hippymfem_owners = tuple(held)
    return mat


def MatAtB(A, B):
    """``A^T @ B``."""
    At = A.Transpose()
    return own(mfem.ParMult(At, B), At)          # At must outlive the product


def MatPtAP(A, P):
    """``P^T @ A @ P``."""
    return take_ownership(mfem.RAP(A, P))


def MatMatMatMult(A, B, C):
    """``A @ B @ C``."""
    AB = mfem.ParMult(A, B)
    return own(mfem.ParMult(AB, C), AB)


def ParAdd(A, B, alpha=1.0, beta=1.0):
    """``alpha*A + beta*B`` as a new ``HypreParMatrix``."""
    if alpha == 1.0 and beta == 1.0:
        return take_ownership(mfem.ParAdd(A, B))
    return take_ownership(mfem.Add(alpha, A, beta, B))


# ------------------------------------------------------------------- diagonals
def _local_diag(A):
    """Diagonal of the locally owned block of ``A`` as a numpy array.

    ``HypreParMatrix::GetDiag(Vector&)`` assumes every row stores its diagonal
    entry first and reads out of bounds on an empty row, which a legitimately
    all-zero block (``W_uu`` of a linear forward problem, say) produces.  Going
    through the diagonal sparse block is safe for any sparsity pattern.
    """
    import scipy.sparse as sp

    blk = _diag_block(A)
    host_sync(blk)
    n, m = blk.Height(), blk.Width()
    csr = sp.csr_matrix(
        (blk.GetDataArray().copy(), blk.GetJArray().copy(), blk.GetIArray().copy()),
        shape=(n, m),
    )
    out = np.zeros(n)
    k = min(n, m)
    out[:k] = csr.diagonal()[:k]
    return out


def set_diagonal_entries(A, idx, value, report_missing=False):
    """Set ``A[i, i] = value`` for the local row indices in ``idx``, in place.

    ``value`` is a scalar, or one value per entry of ``idx``.  A row with no
    structural diagonal entry cannot be written and is skipped; with
    ``report_missing`` the count of such rows is returned instead of ``A``, for
    callers that depend on the entry being there.

    Needed because MFEM's ``DiagonalPolicy`` is honored only on the serial
    ``SparseMatrix`` path: in parallel, ``EliminateRowsCols`` always leaves 1.0 behind.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return 0 if report_missing else A
    vals = np.broadcast_to(np.asarray(value, dtype=np.float64), idx.shape)
    # Stage the write on A, not on the wrapper: GetDiag wraps MFEM's host-side copy of
    # hypre's arrays, so with hypre's memory on a device a write through the wrapper
    # alone never reaches the copy hypre computes with, and nothing fails.
    # A.HostReadWrite() makes the host copy valid and invalidates the device one;
    # A.HypreReadWrite() afterwards puts the edits back where hypre reads them.  Both
    # are no-ops on a host-only build.
    A.HostReadWrite()
    blk = _diag_block(A)
    host_readwrite(blk)
    I = np.asarray(blk.GetIArray())
    J = np.asarray(blk.GetJArray())
    D = blk.GetDataArray()
    lo = I[idx].astype(np.int64)
    hi = I[idx + 1].astype(np.int64)
    # The diagonal entry leads its row in every pattern this library builds (hypre
    # puts it first in a square matrix too), so one vectorized comparison settles
    # nearly every row and only the rest take the Python loop below.
    lead = np.zeros(idx.size, dtype=bool)
    nonempty = hi > lo
    lead[nonempty] = J[lo[nonempty]] == idx[nonempty]
    D[lo[lead]] = vals[lead]
    missing = 0
    for k in np.flatnonzero(~lead):        # the diagonal is elsewhere in the row, or absent
        hit = np.flatnonzero(J[lo[k]:hi[k]] == idx[k])
        if hit.size:
            D[lo[k] + int(hit[0])] = vals[k]
        else:
            missing += 1
    del I, J, D
    A.HypreReadWrite()
    return missing if report_missing else A


def get_diagonal(A, d=None, comm=None):
    """Extract ``diag(A)`` into a :class:`ParVector`."""
    comm = comm if comm is not None else MPI.COMM_WORLD
    if d is None:
        d = ParVector(comm, A.Height())
    d.array[:] = _local_diag(A)
    return d


def trace(A, comm=None):
    """Global trace of a square ``HypreParMatrix``."""
    comm = comm if comm is not None else MPI.COMM_WORLD
    return comm.allreduce(float(_local_diag(A).sum()), op=MPI.SUM)


def estimate_diagonal_inv2(Asolver, k, d):
    """Probing estimate of ``diag(A^{-1})`` from ``k`` Rademacher vectors.

    This is hIPPYlib's estimator: with ``z`` having independent +-1 entries,
    ``E[ z .* A^{-1} z ] = diag(A^{-1})``.
    """
    from .random import parRandom

    x = d.duplicate()
    b = d.duplicate()
    d.zero()
    for _ in range(k):
        parRandom.rademacher(b)
        x.zero()
        Asolver.solve(x, b)
        d.array[:] += x.array * b.array
    d.scale(1.0 / k)
    return d


# -------------------------------------------------------------- dense / scipy
def hypre_to_scipy(A):
    """Local rows of ``A`` as a scipy CSR matrix with **global** column indices.

    PyMFEM's ``ToScipyCSR`` does the same but builds a fresh ``mfem.SparseMatrix``
    per call, which a CUDA-enabled build does not survive repeatedly (see
    :func:`_diag_block`).  Here the merge goes through the shared wrapper and the
    arrays are copied out, so the result does not alias MFEM memory and stays valid
    after the next call.
    """
    import scipy.sparse as sp

    blk = _merged_block(A)
    host_sync(blk)
    n = blk.Height()
    indptr = np.asarray(blk.GetIArray())[:n + 1].copy()
    nnz = int(indptr[-1])
    return sp.csr_matrix(
        (np.asarray(blk.GetDataArray())[:nnz].copy(),
         np.asarray(blk.GetJArray())[:nnz].copy(), indptr),
        shape=(n, blk.Width()))


def scipy_to_hypre(csr, comm=None, col_starts=None):
    """Build a ``HypreParMatrix`` from per-rank CSR blocks with global columns.

    ``csr`` holds this rank's rows; the row partition is inferred from the block
    heights.  ``col_starts`` must be given whenever the matrix is not square
    with identical row and column partitions.
    """
    if comm is not None and comm != MPI.COMM_WORLD:
        raise NotImplementedError(
            "PyMFEM's CSR->hypre helper is hard-wired to COMM_WORLD"
        )
    return _extra().ToHypreParCSR(
        csr.tocsr(),
        col_starts=None if col_starts is None else np.asarray(col_starts, dtype=_HYPRE_INT),
        assert_non_square_no_col_starts=False,
    )


def to_dense(A, comm=None):
    """Gather ``A`` into a dense numpy array, replicated on every rank.

    For small matrices only; used by the test suite to compare against dense
    reference computations.
    """
    comm = comm if comm is not None else MPI.COMM_WORLD
    loc = hypre_to_scipy(A)
    blocks = comm.allgather(
        (loc.indptr.copy(), loc.indices.copy(), loc.data.copy(), loc.shape)
    )
    ncols = A.GetGlobalNumCols()
    rows_total = sum(b[3][0] for b in blocks)
    out = np.zeros((rows_total, ncols))
    r0 = 0
    for indptr, indices, data, shape in blocks:
        for i in range(shape[0]):
            sl = slice(indptr[i], indptr[i + 1])
            out[r0 + i, indices[sl]] = data[sl]
        r0 += shape[0]
    return out


def operator_to_dense(op, domain_size, comm=None):
    """Dense matrix of a matrix-free operator by applying it to unit vectors.

    ``domain_size`` is the *global* domain dimension.  Used only in tests.
    """
    comm = comm if comm is not None else MPI.COMM_WORLD
    x = op.generate_vector(1)
    y = op.generate_vector(0)
    lo, hi = x.owner_range
    cols = []
    for j in range(domain_size):
        x.zero()
        if lo <= j < hi:
            x.array[j - lo] = 1.0
        op.mult(x, y)
        full = comm.allgather(y.array.copy())
        cols.append(np.concatenate(full))
    return np.array(cols).T


# --------------------------------------------------------------------- helpers
def amg_method(amg_type="boomer"):
    """Name of the default algebraic multigrid preconditioner.

    Kept for source compatibility with hIPPYlib drivers, where this selects
    between PETSc's AMG implementations.  Here hypre's BoomerAMG is the only
    option, so the argument is accepted and ignored.
    """
    return "boomer"


def as_matrix(A):
    """Return the underlying ``HypreParMatrix`` of ``A``, or ``A`` itself."""
    return getattr(A, "A", A)

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""From a local CSR to the parallel ``HypreParMatrix``: the constructors, the
triple product ``P^T A P``, and the route selection (``HIPPYMFEM_PARMAT``).
"""

import numpy as np
import mfem.par as mfem
from ..common.identitycache import IdentityCache
from ..common.linalg import own, take_ownership
from ..common.parvector import _HYPRE_INT
from ..config import env_choice
from .prolongation import _boolean_prolongation, _ldof_offset, _ldof_starts
from ..common.parvector import device_active


# --------------------------------------------------- the route through MFEM
#: How the local CSR becomes a ``HypreParMatrix``.  ``"tdof"`` scatters straight into
#: true-dof rows and exchanges the few entries in ghost rows
#: (:mod:`hippymfem.fem.tdofassemble`), which needs boolean prolongations and is two
#: to three times faster than a triple product on a few ranks.  ``"mfem"`` builds the
#: block-diagonal matrix over the local dofs and forms the parallel triple product
#: (:func:`_triple`), which is right for every space.  ``"auto"``, the default, is
#: ``"tdof"`` wherever the prolongations are boolean and ``"mfem"`` otherwise.  Set
#: ``HIPPYMFEM_PARMAT``.
PARMAT_MODE = env_choice("HIPPYMFEM_PARMAT", "auto", ("auto", "mfem", "tdof"))

#: Build the local matrix on the host with the constructor that copies the arrays into
#: hypre's memory, as a run with hypre on a device always does, instead of the one that
#: aliases them.  For the test suite, which holds the two to exact equality.
COPYING_CONSTRUCTOR = False


def set_parmat_mode(mode):
    """Choose how the parallel matrix is built; returns the old mode."""
    global PARMAT_MODE
    if mode not in ("auto", "mfem", "tdof"):
        raise ValueError("PARMAT_MODE must be auto, mfem or tdof, got %r" % (mode,))
    old, PARMAT_MODE = PARMAT_MODE, mode
    return old


def _tdof_route(test_space, trial_space, same):
    """Whether to assemble straight into true-dof rows and skip the triple product.

    Needs boolean prolongations on both sides, decided **collectively** by
    :func:`_boolean_prolongation` so every rank takes the same branch; the only
    other condition is the mode, a global.  Nothing rank-local may be consulted
    ahead of that collective.
    """
    if PARMAT_MODE == "mfem":
        return False
    if not _boolean_prolongation(test_space):
        return False
    return same or _boolean_prolongation(trial_space)


def _as_sparse(pattern, data):
    """Wrap the local CSR as an ``mfem.SparseMatrix`` that owns none of it.

    Returns the matrix and the arrays to keep referenced while it lives.  PyMFEM
    details that decide how this has to be written: a five-element list resolves to
    ``SparseMatrix(i, j, data, m, n)``, which wraps *and owns*, so MFEM would
    ``delete[]`` numpy's buffers; the ``ownij, owna, issorted`` overload leaves them
    alone.  The typemap clears ``NPY_ARRAY_OWNDATA`` on the values array, so a base
    array passed there would be freed by nobody, hence the view.

    **Nothing is copied on the host.**  A ``HypreParMatrix`` built from this
    *aliases* these arrays (its values are ``data``), so the caller must keep what
    this returns alive for as long as the matrix.  That is safe because hypre has
    nothing to write back: a square pattern already stores each row's diagonal
    first (:meth:`ScatterPattern._build`), so ``hypre_CSRMatrixReorder`` leaves it
    in place, and a rectangular block is never reordered.  Every
    ``pattern.data()`` call returns a fresh buffer, so no two matrices share one.

    A buffer that came back from the device is flagged read-only although numpy
    owns it; the flag is lifted so the elimination's in-place writes stay legal.

    **With hypre on a device the matrix aliases them too**, and MFEM registers the
    aliased host pointers with its memory manager, which owns the device mirror:
    two live matrices built from the same arrays would share one registry entry,
    and destroying either would free the mirror of both.  So on a device ``I`` and
    ``J`` are copied per matrix; ``data`` already is a fresh device-to-host
    transfer per assembly.  See ``benchmarks/DESIGN_NOTES.md``, section 2.
    """
    I = np.ascontiguousarray(pattern.indptr, dtype=np.int32)
    J = np.ascontiguousarray(pattern.indices, dtype=np.int32)

    if device_active():
        # private copies per matrix (see the docstring)
        I = I.copy()
        J = J.copy()
    D = np.ascontiguousarray(data, dtype=np.float64)
    if not D.flags.writeable:
        try:
            D.flags.writeable = True
        except ValueError:
            D = D.copy()
    keep = [I, J, D]
    return mfem.SparseMatrix([I, J, D.view(), pattern.nrow, pattern.ncol],
                             False, False, True), keep


def local_par_matrix(pattern, data, test_space, trial_space):
    """Block-diagonal ``HypreParMatrix`` over the ldof partition.

    The local CSR carries local column indices; they are shifted into the global
    ldof numbering here.  Nothing is communicated: every column of a rank's
    element matrices is one of that rank's own local dofs.

    Two constructors can do this, with bit-identical results.  On the host the
    arrays are wrapped as an ``mfem.SparseMatrix`` (:func:`_as_sparse`) that MFEM's
    constructor aliases, copying nothing.  With hypre on a device the raw arrays go
    to the constructor that copies them into hypre's memory: the aliasing one
    registers the arrays with MFEM's memory manager, and a Python proxy drops the
    arrays it keeps before its C++ destructor runs, so hypre's destroy would find
    the device mirrors already freed (see ``TrueDofPattern.finish``).
    """
    on_device = device_active()
    comm = test_space.comm
    roff, rglob = _ldof_offset(test_space)
    same = test_space.fes is trial_space.fes
    if same:
        coff, cglob = roff, rglob
    else:
        coff, cglob = _ldof_offset(trial_space)

    if not (on_device or COPYING_CONSTRUCTOR):
        # The SparseMatrix takes the *local* column indices, not the shifted ones:
        # it is the rank's diagonal block, and the partition arrays say where it
        # sits.  Square blocks take the 4-argument constructor so MFEM sees one
        # partition for both sides.
        sp, keep = _as_sparse(pattern, data)
        if same:
            A = mfem.HypreParMatrix(comm, rglob, _ldof_starts(test_space), sp)
        else:
            A = mfem.HypreParMatrix(comm, rglob, cglob, _ldof_starts(test_space),
                                    _ldof_starts(trial_space), sp)
        # A aliases our arrays, so they have to outlive it.
        own(A, sp, *keep)
    else:
        # cached per partition (see ScatterPattern.hypre_indices)
        I, J, rows, cols = pattern.hypre_indices(roff, coff, same)
        D = np.ascontiguousarray(data, dtype=np.float64)
        if on_device:
            # This constructor's typemaps want HYPRE_Int row pointers and
            # HYPRE_BigInt columns and partitions, whose widths differ between the
            # host and device builds; ``hypre_indices`` is sized for the host build.
            # The casts are no-ops where the widths agree.
            I = np.ascontiguousarray(I, dtype=np.int32)
            J = np.ascontiguousarray(J, dtype=_HYPRE_INT)
            rows = np.ascontiguousarray(rows, dtype=_HYPRE_INT)
            if cols is not None:              # None for a square block, by design
                cols = np.ascontiguousarray(cols, dtype=_HYPRE_INT)
        if same:
            # A 4-element list makes PyMFEM pass one pointer for both partitions,
            # which is how MFEM detects "square, same partition" and reorders each
            # row so that its diagonal entry comes first.
            args = [I, J, D, rows]
        else:
            args = [I, J, D, rows, cols]
        A = mfem.HypreParMatrix(comm, pattern.nrow, rglob, cglob, args)
    A.CopyRowStarts()
    A.CopyColStarts()
    return A


_TRANSPOSE = IdentityCache()


def _transposed(P, owner):
    """``P^T`` for the prolongation of the space ``owner``, built once.

    ``P`` belongs to the space and never changes, and hypre is handed the
    transpose on every assembly, so it is built once, owned and kept.  Keyed on
    the space, not on ``P``: every ``Dof_TrueDof_Matrix`` call returns a new
    proxy, on which a cache would always miss.
    """
    got = _TRANSPOSE.get((owner,))
    if got is None:
        got = _TRANSPOSE.put((owner,), take_ownership(P.Transpose()))
    return got


def _triple(A, Rt, P, spaces):
    """``R^T A P`` as two sparse products, with the transpose taken once.

    ``spaces`` are the ``ParFiniteElementSpace`` objects the prolongations belong
    to (the test space first); the first one keys the cached transpose.  The result
    is the matrix MFEM's fused ``RAP`` gives, nonzero counts and the explicitly
    stored zeros of the folded elimination included (``test_assembly`` holds the two
    to exact equality).  On the host the two products take a half to a sixth of the
    fused form's time; on a device the two forms are within a quarter of each other.
    """
    tmp = take_ownership(mfem.ParMult(_transposed(P if Rt is None else Rt, spaces[0]), A))
    try:
        return take_ownership(mfem.ParMult(tmp, P))
    finally:
        del tmp

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""From a local CSR to the parallel ``HypreParMatrix``: the constructors, the
triple product ``P^T A P`` in its two forms, and the route selection
(``HIPPYMFEM_PARMAT``, ``HIPPYMFEM_TRIPLE``).
"""

import os
import numpy as np
import mfem.par as mfem
from mpi4py import MPI
from ..common.identitycache import IdentityCache
from ..common.linalg import own, take_ownership
from ..common.parvector import _HYPRE_INT
from .prolongation import _boolean_prolongation, _ldof_offset, _ldof_starts
import time
from ..common.parvector import device_active


def _check_host_hypre():
    """Raise ``RuntimeError`` if MFEM has put hypre's memory on a device.

    Called by :func:`local_par_matrix` only on its host path in ``"direct"`` mode
    (:data:`PARMAT_MODE`); with hypre on a device that function takes the copying
    constructor without calling this.
    """

    if device_active():
        raise RuntimeError(
            "MFEM is configured on a GPU device (mfem.Device(\"cuda\")), which puts "
            "hypre's memory on the device, and HIPPYMFEM_PARMAT=direct builds the "
            "matrix from host CSR arrays, which hypre then reads as device "
            "pointers.\n"
            "  Unset HIPPYMFEM_PARMAT (or set it to \"auto\") to let MFEM's own "
            "ParallelAssemble build the parallel matrix from the same local CSR, "
            "which is bit-identical and works on the device.\n"
            "  Whichever route builds the matrix, give each rank its own GPU or "
            "they all land on device 0:\n"
            "      mfem.Device(\"cuda\", rank % n_devices)\n"
            "  See docs/source/guide/gpu.rst for what each configuration measured.")


# --------------------------------------------------- the route through MFEM
#: How the local CSR becomes a ``HypreParMatrix``.  ``"direct"`` hands hypre the
#: numpy arrays through PyMFEM's CSR constructor, which copies them; ``"mfem"`` wraps
#: them as an ``mfem.SparseMatrix`` and lets MFEM's constructor alias them, about a
#: hundred times cheaper.  Both then form the parallel triple product.  ``"tdof"``
#: does not: for boolean prolongations it scatters straight into true-dof rows and
#: exchanges the few entries in ghost rows (:mod:`hippymfem.fem.tdofassemble`), two
#: to three times faster than the triple product on a few ranks.  ``"auto"``, the
#: default, is ``"tdof"`` wherever the prolongations are boolean and ``"mfem"``
#: otherwise; ``"direct"`` is the reference the test suite compares against.  With
#: hypre on a device the triple-product route always uses the copying constructor
#: (see :func:`local_par_matrix`).  Set ``HIPPYMFEM_PARMAT``.
PARMAT_MODE = os.environ.get("HIPPYMFEM_PARMAT", "auto").lower()


def set_parmat_mode(mode):
    """Choose how the parallel matrix is built; returns the old mode."""
    global PARMAT_MODE
    if mode not in ("auto", "direct", "mfem", "tdof"):
        raise ValueError("PARMAT_MODE must be auto, direct, mfem or tdof, got %r"
                         % (mode,))
    old, PARMAT_MODE = PARMAT_MODE, mode
    return old


def _via_mfem():
    """Whether to let MFEM build the parallel matrix from our local CSR."""
    return PARMAT_MODE != "direct"


def _tdof_route(test_space, trial_space, same):
    """Whether to assemble straight into true-dof rows and skip the triple product.

    Needs boolean prolongations on both sides, decided **collectively** by
    :func:`_boolean_prolongation` so every rank takes the same branch; the only
    other condition is the mode, a global.  Nothing rank-local may be consulted
    ahead of that collective.
    """
    if PARMAT_MODE in ("mfem", "direct"):
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


def local_par_matrix(pattern, data, test_space, trial_space, reuse=False):
    """Block-diagonal ``HypreParMatrix`` over the ldof partition.

    The local CSR carries local column indices; they are shifted into the global
    ldof numbering here.  Nothing is communicated: every column of a rank's
    element matrices is one of that rank's own local dofs.

    Two constructors can do this, with bit-identical results.  On the host the
    default wraps the arrays as an ``mfem.SparseMatrix`` (:func:`_as_sparse`) that
    MFEM's constructor aliases, copying nothing; ``HIPPYMFEM_PARMAT=direct`` hands
    the raw arrays to the constructor that copies them into hypre's memory.  With
    hypre on a device the copying constructor is always used: the aliasing one
    registers the arrays with MFEM's memory manager, and a Python proxy drops the
    arrays it keeps before its C++ destructor runs, so hypre's destroy would find
    the device mirrors already freed (see ``TrueDofPattern.finish``).

    With ``reuse``, one matrix per (pattern, partition) is built on the first call
    and later calls overwrite its values in place.  That is only correct when the
    matrix does not outlive the call, which is the case exactly when a triple
    product follows and consumes it; a caller that receives this matrix as the
    assembled result must not ask for it.  Only the copying constructor on the
    host reuses; the aliasing route, and any build with hypre on a device, makes a
    new matrix each time.
    """

    # On a device: the copying constructor, and no reuse, whose in-place writes go
    # through a host alias of hypre's values.
    on_device = device_active()
    via_mfem = _via_mfem() and not on_device
    reuse = reuse and not on_device
    if not via_mfem and not on_device:
        _check_host_hypre()
    comm = test_space.comm
    roff, rglob = _ldof_offset(test_space)
    same = test_space.fes is trial_space.fes
    if same:
        coff, cglob = roff, rglob
    else:
        coff, cglob = _ldof_offset(trial_space)

    # cached per partition (see ScatterPattern.hypre_indices)
    I, J, rows, cols = pattern.hypre_indices(roff, coff, same)
    if reuse and not via_mfem:
        got = pattern.reusable(roff, coff, same)
        if got is not None:
            A, view, order = got
            if order is None:
                np.copyto(view, data, casting="unsafe")
            else:
                np.take(data, order, out=view)
            return A
    if via_mfem:
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
    if reuse and not via_mfem:
        pattern.make_reusable(roff, coff, same, A)
    return A


#: How to form the parallel triple product ``R^T A P``.  ``"rap"`` is MFEM's fused
#: :func:`mfem.RAP`; ``"split"`` is the same thing as two sparse products with the
#: transpose taken once; ``"auto"`` times both on the first assembly for a given
#: pair of spaces and keeps the faster.  They are bit-identical, so this is a
#: speed choice only.  Set ``HIPPYMFEM_TRIPLE``.
TRIPLE_MODE = os.environ.get("HIPPYMFEM_TRIPLE", "auto").lower()


_TRIPLE_CHOICE = IdentityCache()


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


def _triple_fused(A, Rt, P, owner=None):
    return take_ownership(mfem.RAP(A, P) if Rt is None else mfem.RAP(Rt, A, P))


def _triple_split(A, Rt, P, owner=None):
    """``R^T A P`` as two sparse products instead of one fused triple product.

    The output is identical to the fused form's, nonzero counts and the explicitly
    stored zeros of the folded elimination included.  Which is faster depends on
    the rank count and the problem size, so :func:`_triple` measures.
    """
    tmp = take_ownership(mfem.ParMult(_transposed(P if Rt is None else Rt, owner), A))
    try:
        return take_ownership(mfem.ParMult(tmp, P))
    finally:
        del tmp


def _triple(A, Rt, P, comm, spaces):
    """``R^T A P`` by whichever form is faster here, decided once per space pair.

    ``spaces`` are the ``ParFiniteElementSpace`` objects the prolongations belong
    to (the test space first), which key the decision and the cached transpose.

    The decision has to be the same on every rank, since both forms communicate:
    the two timings are reduced with MAX (the slowest rank is what a collective
    costs) before they are compared, so every rank compares the same numbers.
    """
    mode = TRIPLE_MODE
    if mode == "auto":
        mode = _TRIPLE_CHOICE.get(spaces)
        if mode is None:

            t = {}
            for name, fn in (("rap", _triple_fused), ("split", _triple_split)):
                fn(A, Rt, P, spaces[0])                    # warm-up, dropped at once
                comm.Barrier()
                t0 = time.perf_counter()
                fn(A, Rt, P, spaces[0])
                t[name] = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
            mode = "rap" if t["rap"] <= t["split"] else "split"
            _TRIPLE_CHOICE.put(spaces, mode)
    return (_triple_fused if mode == "rap" else _triple_split)(A, Rt, P, spaces[0])


def set_triple_mode(mode):
    """Choose the triple-product form (``"auto"``, ``"rap"``, ``"split"``); returns the
    old one.  The tests use it to compare the two."""
    global TRIPLE_MODE
    if mode not in ("auto", "rap", "split"):
        raise ValueError("TRIPLE_MODE must be auto, rap or split, got %r" % (mode,))
    old, TRIPLE_MODE = TRIPLE_MODE, mode
    return old


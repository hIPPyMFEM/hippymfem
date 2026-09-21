# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Essential-dof elimination: folded into the scatter where the prolongation is
boolean (``HIPPYMFEM_FOLD_ELIMINATION``), through MFEM's own calls otherwise.
"""

import os
import numpy as np
from ..common.linalg import _local_diag, own, set_diagonal_entries, take_ownership
from .prolongation import _boolean_prolongation


#: When true (the default), essential-dof elimination is folded into the scatter
#: as a mask on the local CSR slots, instead of a second parallel matrix from
#: ``EliminateRowsCols``, which is a large share of a GPU assembly.
#: ``HIPPYMFEM_FOLD_ELIMINATION=0`` uses MFEM's calls instead; the test suite
#: compares the two entry by entry.
FOLD_ELIMINATION = os.environ.get("HIPPYMFEM_FOLD_ELIMINATION", "1").lower() not in (
    "0", "no", "false", "off")


def set_fold_elimination(flag=True):
    """Turn the folded essential-dof elimination on or off; returns the old value."""
    global FOLD_ELIMINATION
    old, FOLD_ELIMINATION = FOLD_ELIMINATION, bool(flag)
    return old


def _foldable(test_space, trial_space, test_ess, trial_ess, diag_policy, same):
    """Whether the essential-dof elimination can be folded into the scatter.

    Three conditions, all of which every rank decides identically: there is
    something to eliminate; the prolongations are boolean, so masking local rows
    and columns is equivalent to eliminating true ones
    (:func:`_boolean_prolongation`, which reduces its local test across ranks);
    and the diagonal policy is one this path can reproduce.  ``"keep"`` is not,
    because masking destroys the assembled diagonal it would have kept, so it
    goes through :func:`_eliminate`.
    """
    if not FOLD_ELIMINATION:
        return False
    if test_ess is None and trial_ess is None:
        return False
    if diag_policy not in ("one", "zero"):
        return False
    if not _boolean_prolongation(test_space):
        return False
    return same or _boolean_prolongation(trial_space)


def _set_eliminated_diagonal(A, ess, value):
    """Write the diagonal of the eliminated rows, and insist the slot is there.

    The masked rows reach hypre as structural zeros, which its triple product
    keeps because the symbolic phase is value-independent (the test suite checks
    this).  Were a future hypre to prune them, the diagonal would silently stay
    zero and the matrix be singular, so a miss raises ``RuntimeError`` here rather
    than giving a wrong answer downstream.
    """
    idx = np.asarray(ess.ToList(), dtype=np.int64)
    if idx.size == 0:
        return A
    missed = set_diagonal_entries(A, idx, value, report_missing=True)
    if missed:
        raise RuntimeError(
            "%d of %d eliminated rows have no diagonal entry in the assembled "
            "matrix; hypre pruned the structural zeros that the folded "
            "elimination relies on. Set HIPPYMFEM_FOLD_ELIMINATION=0."
            % (missed, idx.size))
    return A


_keep_with = own          # an alias, re-exported by csrassemble


def _eliminate(A, test_space, trial_space, test_ess, trial_ess, diag_policy,
               same):
    """Apply essential-dof elimination exactly as MFEM's Form*SystemMatrix does.

    The eliminated part ``Ae`` is **kept alive alongside A**, as MFEM keeps it
    (``p_mat_e``, for the lifetime of the form), and this is not optional:
    ``hypre_ParCSRMatrixEliminateAAe`` builds ``Ae`` sharing state with ``A``, so
    destroying ``Ae`` silently damages ``A``.  The damage shows only when
    something walks ``A``'s parallel structure (a transpose, say) and only once a
    rank has no essential dofs, so it hides on small rank counts and segfaults on
    large ones.
    """
    if same:
        ess = test_ess
        # Branch on whether the caller asked for elimination at all, which every
        # rank decides alike, and NEVER on ess.Size(): with essential conditions on
        # part of the boundary some ranks own none, and skipping the collective
        # EliminateRowsCols on those ranks while the others call it desynchronizes
        # the job (unnoticed on a few ranks, a segfault on more).
        if ess is None:
            return A
        idx = np.asarray(ess.ToList(), dtype=np.int64)
        keep = _local_diag(A)[idx] if diag_policy == "keep" else None
        # HypreParMatrix::EliminateRowsCols zeroes the rows and columns and
        # leaves 1.0 on the diagonal, whatever MFEM's DiagonalPolicy says; the
        # serial DiagonalPolicy never reaches the parallel path.
        Ae = take_ownership(A.EliminateRowsCols(ess))
        _keep_with(A, Ae)
        if diag_policy == "zero":
            set_diagonal_entries(A, idx, 0.0)
        elif diag_policy == "keep":
            set_diagonal_entries(A, idx, keep)
        return A

    # EliminateCols communicates, so it is called on the same condition everywhere;
    # EliminateRows only touches local rows and is safe either way.
    if trial_ess is not None:
        _keep_with(A, take_ownership(A.EliminateCols(trial_ess)))
    if test_ess is not None and test_ess.Size():
        A.EliminateRows(test_ess)
    return A

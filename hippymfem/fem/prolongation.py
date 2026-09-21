# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The prolongation ``P`` of a space: is it the identity, is it boolean, and the
essential-dof mask it induces on local dofs.

Every answer here selects between code paths with *different* collectives, so the
ones that matter are decided collectively and memoized on the space
(:class:`~hippymfem.common.identitycache.IdentityCache`).
"""

import numpy as np
import mfem.par as mfem
from mpi4py import MPI
from ..common.identitycache import IdentityCache
from ..common.linalg import _diag_block, _merged_block
from ..common.parvector import host_readwrite, host_sync
from ..common.parvector import device_active


# ------------------------------------------------------------- local -> hypre
def _ldof_offset(space):
    """This rank's first global ldof index, and the global ldof count.

    Taken from the row partition of ``Dof_TrueDof_Matrix`` rather than recomputed,
    so that the matrix built here and the ``P`` it is multiplied by cannot
    disagree about the partition.
    """
    P = space.fes.Dof_TrueDof_Matrix()
    part = P.GetRowPartArray()
    return int(part[0]), int(P.GetGlobalNumRows())


def _ldof_starts(space):
    """The ldof row partition of ``P`` as the pointer hypre's constructors want.

    The constructors that take an ``mfem.SparseMatrix`` have no numpy typemap for
    ``HYPRE_BigInt *row_starts``, so the pointer is taken from
    ``Dof_TrueDof_Matrix``, which also makes it the partition the triple product
    uses (see :func:`_ldof_offset`).
    """
    return space.fes.Dof_TrueDof_Matrix().GetRowStarts()


def _is_identity(P):
    """True when ``P`` is the identity, so ``P^T A P`` can be skipped.

    This is the common serial/conforming case: each true dof is one local dof.
    Checked, not assumed: a non-conforming mesh has a genuine ``P`` even on one
    rank.
    """
    if P.GetGlobalNumRows() != P.GetGlobalNumCols():
        return False
    if P.Height() != P.Width():
        return False
    blk = _diag_block(P)
    n = blk.Height()
    if blk.NumNonZeroElems() != n:
        return False
    I = blk.GetIArray()
    J = blk.GetJArray()
    D = blk.GetDataArray()
    if I[n] != n:
        return False
    return (np.array_equal(np.asarray(J[:n]), np.arange(n))
            and np.all(np.asarray(D[:n]) == 1.0))


_IDENTITY_P = IdentityCache()


_BOOLEAN_P = IdentityCache()


_ESS_LDOF = IdentityCache()


def _is_boolean(P):
    r"""True when every row of ``P`` has exactly one entry and it is ``1``.

    That is the conforming case, where each local dof is a copy of exactly one
    true dof.  It is what makes the essential-dof mask foldable into the scatter:
    for such a ``P``, masking local rows and columns before the triple product
    gives the same matrix as eliminating true rows and columns after it, because
    the local mask is the pullback of the true one and
    :math:`P^{\top}(DAD)P = D_t (P^{\top}\!AP) D_t`.  A non-conforming mesh has a
    ``P`` with real interpolation weights, for which that identity is false.

    **Host only.**  ``MergeDiagAndOffd`` puts the rank's rows of both hypre blocks
    into one local CSR (the two blocks separately would need ``GetOffd``, whose
    column-map out-parameter is not callable from Python), but it has hypre
    allocate the merged matrix in hypre's *own* memory location and then
    deep-copies it from the host.  With hypre on a device, once ``P`` has migrated
    there, the copy reads device memory through a host pointer and segfaults.
    :func:`_boolean_local` decides when this may be called.
    """
    merged = _merged_block(P)
    host_sync(merged)
    # The columns never matter: one entry per row, equal to one, is the whole test.
    I = np.asarray(merged.GetIArray(), dtype=np.int64)
    D = np.asarray(merged.GetDataArray())
    return bool(np.all(np.diff(I) == 1) and np.all(D == 1.0))


def _has_dof_transformation(space):
    """Whether the space applies a per-element ``DofTransformation``.

    That is H(curl) on simplices, prisms and pyramids at order two and up, where a
    face carries more than one dof and the element's view of the face is a dense
    transformation of the face's own.  It is a property of the collection and the
    element geometry, so one element per geometry is asked, and the loop stops once
    every geometry in the mesh has been seen (one call on a single-geometry mesh).
    ``IsIdentity`` marks order one, where the transformation exists but has
    nothing to act on.
    """
    fes = space.fes
    get = getattr(fes, "GetElementDofTransformation", None)
    if get is None:
        return False
    mesh = fes.GetParMesh()
    ngeom = mesh.GetNumGeometries(mesh.Dimension())
    seen = set()
    for e in range(mesh.GetNE()):
        g = mesh.GetElementBaseGeometry(e)
        if g in seen:
            continue
        seen.add(g)
        dt = get(e)
        if dt is not None and not dt.IsIdentity():
            return True
        if len(seen) >= ngeom:
            break
    return False


def _boolean_local(space, P):
    """Whether this rank's ``P`` is boolean, by the cheapest test that is safe here.

    For a space with a ``DofTransformation`` the answer is no before anything is
    looked at: MFEM folds the face transformation of a shared face into ``P`` on
    the rank that does not own it, and neither the fold nor the true-dof route holds
    for such a ``P`` (``_is_boolean`` would catch only some of the symptoms, and only
    on the ranks that have them).  After that,
    ``ParFiniteElementSpace::Conforming()`` is *sufficient* and needs no matrix
    traversal; it is not *necessary* (an ``NCMesh`` with no hanging nodes answers
    False while its ``P`` is still boolean), so when it says no the entries are
    inspected with :func:`_is_boolean`.  See ``benchmarks/DESIGN_NOTES.md``,
    section 2.

    With hypre's memory on a device :func:`_is_boolean` cannot run, and the
    conservative answer is taken.  That gives up only the folded elimination on a
    nonconforming mesh whose ``P`` happens to be boolean; the result is correct
    either way, since :func:`_eliminate` then does the work with MFEM's own calls.
    """
    if _has_dof_transformation(space):
        return False
    if space.fes.Conforming():
        return True

    if device_active():
        return False
    return _is_boolean(P)


def _boolean_prolongation(space):
    """Whether this space's ``P`` is boolean, decided **collectively**.

    The answer selects between two code paths with *different* collectives, so
    every rank has to reach the same one: ``EliminateRowsCols`` communicates, and
    a rank that skipped it while the others called it would hang the job.  The
    local test is therefore reduced with a logical AND before it is used, and the
    result is memoized so the reduction happens once per space.
    """
    got = _BOOLEAN_P.get((space.fes,))
    if got is None:
        P, ident = _prolongation(space)
        local = True if ident else _boolean_local(space, P)
        got = _BOOLEAN_P.put((space.fes,),
                             bool(space.comm.allreduce(bool(local), op=MPI.LAND)))
    return got


def _ess_ldof_mask(space, ess):
    """Local-dof marker for a set of essential *true* dofs.

    Obtained by pushing a true-dof indicator through the space's own prolongation
    (``SetFromTrueDofs``), so the marker is by construction the pullback MFEM
    would use, including on the ranks that merely hold a copy of a shared dof.
    Collective, and memoized on the (space, dof set) pair.
    """
    got = _ESS_LDOF.get((space.fes, ess))
    if got is not None:
        return got
    gf = mfem.ParGridFunction(space.fes)
    tv = mfem.Vector(space.fes.GetTrueVSize())
    tv.Assign(0.0)
    idx = np.asarray(ess.ToList(), dtype=np.int64)
    if idx.size:
        # host_readwrite, not a bare GetDataArray: under a device-configured MFEM
        # the write would reach only the host copy, SetFromTrueDofs would prolong
        # the zeros still on the device, and the mask would come back empty, with
        # no error: the matrix would silently keep the rows it should clear.
        host_readwrite(tv)
        tv.GetDataArray()[idx] = 1.0
    gf.SetFromTrueDofs(tv)
    host_sync(gf)
    mask = np.asarray(gf.GetDataArray()).copy() > 0.5
    return _ESS_LDOF.put((space.fes, ess), mask)


def _prolongation(space):
    """``(P, is_identity)`` for a space, with the identity test memoized.

    ``Dof_TrueDof_Matrix`` hands back a new Python proxy of the same matrix on
    every call, so anything cached about ``P`` is keyed on the space, not on it.
    """
    P = space.fes.Dof_TrueDof_Matrix()
    got = _IDENTITY_P.get((space.fes,))
    if got is None:
        got = _IDENTITY_P.put((space.fes,), _is_identity(P))
    return P, got

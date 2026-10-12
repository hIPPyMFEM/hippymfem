# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The interpolation of a p-multigrid: from the linear elements of a mesh to its elements
of higher order, on true dofs.

A function of the low-order space is a function of the high-order one, and its dofs there
are its values at that space's nodes: the row of a high-order dof holds the low-order shape
functions at its node, one to eight numbers for a quadratic hexahedron (a vertex takes its
own value, the middle of an edge the mean of its two ends, and so on).  With ``P`` this
matrix and ``A`` the matrix of a form on the high-order space, ``P^T A P`` is the matrix of
the same form on the low-order space, by the same quadrature: a coarse problem that needs
nothing of the form but ``A``.

:func:`interpolation` builds ``P`` in the two blocks of hypre's layout of a parallel matrix,
for :class:`hippymfem.algorithms.singlesolve.PTransfer`.

Under the linear elements the same engine can go on with the meshes that the mesh was
refined from, where it has them (:func:`set_coarser_meshes`): :func:`refinement` is the
interpolation from the linear elements of a mesh to those of the mesh refined from it, the
values of a coarse function at the fine vertices (its own value at a coarse vertex, the
mean of two, four or eight at the middle of an edge, a face or an element cut in two a
side).  Its Galerkin products have the 27 entries a row of the fine matrix at every level
and an eighth of the rows each time, where an algebraic coarsening of that matrix has 100
to 170 entries a row from its second level on and a level more for every eight times the
ranks.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from ..common.parvector import host_sync
from .elementbatch import group_tables
from .prolongation import _boolean_prolongation, _ess_ldof_mask, _prolongation
from .spaces import as_space
from .tdofassemble import _tdof_maps


class TransferBlocks:
    """The interpolation of :func:`interpolation`: what
    :meth:`~hippymfem.algorithms.singlesolve.SingleParMatrix.from_pattern` reads of a
    true-dof pattern, with the values (``values``: the diagonal block's, then the
    off-diagonal one's).

    ``row_sum_error``: how far the rows are from summing to one before the essential dofs
    are taken out (over all ranks), which they do for an interpolation of nodal elements.
    """

    def device_columns(self, block, create=True):
        return None


def why_not(fine, coarse=None):
    """Why no interpolation to ``fine`` can be built here (from ``coarse``, when given), or
    ``""``.  It is built for the nodal H1 spaces of ``H1_FECollection`` on a conforming
    mesh, from linear elements to elements of a higher order."""
    fine = as_space(fine)
    spaces = [fine] if coarse is None else [fine, as_space(coarse)]
    for s in spaces:
        if not s.fec.Name().startswith("H1_"):
            return "the space %s is not a nodal H1 space" % s.fec.Name()
        if not _boolean_prolongation(s):
            return "the mesh is nonconforming"
    if fine.order < 2:
        return "the space has linear elements already"
    if coarse is not None:
        coarse = spaces[1]
        if coarse.mesh is not fine.mesh:
            return "the two spaces are on different meshes"
        if coarse.vdim != fine.vdim:
            return "the two spaces have %d and %d components" % (coarse.vdim, fine.vdim)
        if coarse.order != 1:
            return "the coarse space is of order %d, not of linear elements" % coarse.order
    return ""


def interpolation(fine, coarse, groups, ess=None):
    """The interpolation from ``coarse`` to ``fine``, two nodal H1 spaces on one conforming
    mesh, on true dofs.

    ``groups``: the element groups of the mesh (``MeshBatches.groups``).  ``ess``: the
    essential true dofs of ``fine``, an ``mfem.intArray`` as a boundary condition holds it
    (``bc.ess_tdof``).  A coarse dof at the node of an essential fine dof then keeps that
    dof alone, with weight one, and nothing else: the Galerkin product has the identity
    there, as the fine matrix has on its essential dofs, and a coarse correction leaves the
    essential dofs alone.  (A fine dof at another node of the essential boundary has no
    entry: it is interpolated from essential nodes only.)

    Collective.  Returns a :class:`TransferBlocks`.
    """
    fine, coarse = as_space(fine), as_space(coarse)
    why = why_not(fine, coarse)
    if why:
        raise NotImplementedError("no p-multigrid interpolation: %s" % why)
    fes2, fes1 = fine.fes, coarse.fes
    comm, vdim = fine.comm, fine.vdim
    gt2, lt2 = _tdof_maps(fes2)
    gt1, _ = _tdof_maps(fes1)
    nld2, nld1 = int(fes2.GetVSize()), int(fes1.GetVSize())
    rows, cols, vals = [], [], []
    done = np.zeros(nld2, dtype=bool)
    locals_ = []
    for group in groups:
        t2, t1 = group_tables(fine, group), group_tables(coarse, group)
        if not group.ne:
            locals_.append(np.zeros((t2.nd_total, t1.nd_total)))
            continue
        e0 = int(group.elems[0])
        fe2, fe1 = fes2.GetFE(e0), fes1.GetFE(e0)
        nodes = fe2.GetNodes()
        sh = mfem.Vector(t1.nd)
        local = np.zeros((t2.nd, t1.nd))
        for i in range(t2.nd):
            fe1.CalcShape(nodes.IntPoint(i), sh)
            host_sync(sh)
            local[i, :] = sh.GetDataArray()
        local[np.abs(local) < 1e-12] = 0.0
        # (for the element dofs, which are component-major: every component alike)
        locals_.append(np.kron(np.eye(vdim), local))
        # every fine dof of the group from the first element that holds it (all of them
        # give its row the same numbers)
        flat = t2.edofs.ravel()
        first = np.full(nld2, -1, dtype=np.int64)
        first[flat[::-1]] = np.arange(flat.size - 1, -1, -1, dtype=np.int64)
        todo = np.flatnonzero((first >= 0) & ~done)
        done[todo] = True
        el, at = first[todo] // t2.nd_total, first[todo] % t2.nd_total
        del first, flat
        comp, node = at // t2.nd, at % t2.nd                  # (element dofs are component-major)
        for j in range(t1.nd):
            w = local[node, j]
            keep = np.flatnonzero(w != 0.0)
            rows.append(todo[keep])
            cols.append(t1.edofs[el[keep], comp[keep] * t1.nd + j])
            vals.append(w[keep])
        del el, at, comp, node, todo
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    return _blocks(fine, coarse, rows, cols, vals, done, ess, locals_)


def _blocks(fine, coarse, rows, cols, vals, done, ess, locals_):
    """The :class:`TransferBlocks` of an interpolation given by its entries on local dofs
    (``rows`` of ``fine``, ``cols`` of ``coarse``; ``done``: the fine local dofs that have
    a row), with the essential true dofs ``ess`` of ``fine`` taken out as
    :func:`interpolation` says.  Collective."""
    fes2, fes1 = fine.fes, coarse.fes
    comm = fine.comm
    gt2, lt2 = _tdof_maps(fes2)
    gt1, _ = _tdof_maps(fes1)
    nld2, nld1 = int(fes2.GetVSize()), int(fes1.GetVSize())
    sums = np.bincount(rows, weights=vals, minlength=nld2)
    off = float(np.abs(sums[done] - 1.0).max()) if done.any() else 0.0
    del sums
    ess1 = np.zeros(nld1, dtype=bool)
    if ess is not None and comm.allreduce(int(ess.Size()), op=MPI.MAX):
        # the coarse dofs at essential nodes: those whose own fine dof is essential
        mask2 = _ess_ldof_mask(fine, ess)
        own = np.abs(vals - 1.0) < 1e-12
        ess1[cols[own]] = mask2[rows[own]]
        keep = ~ess1[cols] | own
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        del mask2, own, keep
    # the essential true dofs of the coarse space, as a boundary condition holds them
    _, lt1 = _tdof_maps(fes1)
    coarse_ess = mfem.intArray([int(i) for i in np.sort(lt1[ess1 & (lt1 >= 0)])])
    mine = lt2[rows] >= 0
    r, g, v = lt2[rows[mine]], gt1[cols[mine]], vals[mine]
    del rows, cols, vals, mine
    o = np.lexsort((g, r))
    r, g, v = r[o], g[o], v[o]
    del o
    Pt, Pr = _prolongation(fine)[0], _prolongation(coarse)[0]
    cpt, cpr = Pt.GetColPartArray(), Pr.GetColPartArray()
    tb = TransferBlocks()
    tb.comm = comm
    tb.t0, tb.t1, tb.c0, tb.c1 = int(cpt[0]), int(cpt[1]), int(cpr[0]), int(cpr[1])
    tb.gnr, tb.gnc = int(Pt.GetGlobalNumCols()), int(Pr.GetGlobalNumCols())
    tb.ntd = tb.t1 - tb.t0
    ind = (g >= tb.c0) & (g < tb.c1)
    tb.nnz_diag, tb.nnz_t = int(ind.sum()), int(g.size)
    tb.I_diag = np.zeros(tb.ntd + 1, dtype=np.int32)
    np.cumsum(np.bincount(r[ind], minlength=tb.ntd), out=tb.I_diag[1:])
    tb.J_diag = (g[ind] - tb.c0).astype(np.int32)
    tb.I_offd = np.zeros(tb.ntd + 1, dtype=np.int32)
    np.cumsum(np.bincount(r[~ind], minlength=tb.ntd), out=tb.I_offd[1:])
    cmap = np.unique(g[~ind])
    tb.cmap, tb.n_offd = cmap.astype(np.int64), int(cmap.size)
    tb.J_offd = np.searchsorted(cmap, g[~ind]).astype(np.int32)
    tb.values = np.concatenate([v[ind], v[~ind]])
    #: per element group, the interpolation of one element (fine element dofs by coarse
    #: ones), for the products with element matrices (``P_e^T A_e P_e``, ``A_e P_e``)
    tb.local = locals_
    tb.coarse, tb.coarse_ess = coarse, coarse_ess
    tb.row_sum_error = float(comm.allreduce(off, op=MPI.MAX))
    tb.global_nnz = int(comm.allreduce(tb.nnz_t, op=MPI.SUM))
    return tb


def set_coarser_meshes(mesh, coarser):
    """Tell the library the meshes that ``mesh`` was refined from: ``coarser[0]`` the mesh
    that ``mesh`` is a refinement of (``ParMesh.MakeRefined(coarser[0], f, ...)``, which
    keeps the map of every fine element into its coarse one), ``coarser[1]`` the one that
    ``coarser[0]`` is a refinement of, and so on.  A p-multigrid asked for geometric levels
    (the solver parameter ``pmg_h``, ``HIPPYMFEM_PMG_H``) takes them from here.  The meshes
    are kept alive with ``mesh``."""
    mesh.hippymfem_coarser = list(coarser)


def coarser_meshes(mesh):
    """The meshes of :func:`set_coarser_meshes` for ``mesh``, the next coarser first; empty
    where there are none."""
    return list(getattr(mesh, "hippymfem_coarser", None) or ())


def refinement(fine, coarse, ess=None):
    """The interpolation from ``coarse`` to ``fine``, the linear elements of a conforming
    mesh and of the mesh refined from it (``fine.mesh`` made by ``MakeRefined`` of
    ``coarse.mesh``), on true dofs: the values of a coarse function at the fine vertices.

    ``ess``: the essential true dofs of ``fine``, as in :func:`interpolation`, with the
    same rule: a coarse dof at the vertex of an essential fine dof keeps that dof alone,
    so that the Galerkin product has the identity there and a coarse correction leaves
    the essential dofs alone.  The essential true dofs of ``coarse`` that this gives are
    the result's ``coarse_ess``, for the level below.

    Collective.  Returns a :class:`TransferBlocks` (without ``local``: there is no one
    interpolation of an element here; with ``cut``, how many times coarser a side the coarse
    mesh is).
    """
    fine, coarse = as_space(fine), as_space(coarse)
    for s in (fine, coarse):
        if not s.fec.Name().startswith("H1_") or s.order != 1:
            raise NotImplementedError("no interpolation between meshes for %s of order %d: it is built for linear "
                                      "nodal elements" % (s.fec.Name(), s.order))
        if not _boolean_prolongation(s):
            raise NotImplementedError("no interpolation between meshes: the mesh is nonconforming")
    if fine.vdim != 1 or coarse.vdim != 1:
        raise NotImplementedError("no interpolation between meshes for %d components" % fine.vdim)
    # MFEM's own, on the local dofs of the two spaces (it reads the map of the fine
    # elements into the coarse ones that the refinement left with the fine mesh)
    T = mfem.OperatorHandle(mfem.Operator.MFEM_SPARSEMAT)
    fine.fes.GetTransferOperator(coarse.fes, T)
    sp = mfem.Opr2SparseMat(T.Ptr())
    if sp is None:
        raise RuntimeError("MFEM gave no matrix for the interpolation between the two meshes")
    nld2 = int(fine.fes.GetVSize())
    if int(sp.Height()) != nld2 or int(sp.Width()) != int(coarse.fes.GetVSize()):
        raise RuntimeError("the interpolation between the two meshes is %d by %d, the spaces have %d and %d local dofs"
                           % (sp.Height(), sp.Width(), nld2, coarse.fes.GetVSize()))
    I = np.array(sp.GetIArray(), dtype=np.int64)
    cols = np.array(sp.GetJArray(), dtype=np.int64)
    vals = np.array(sp.GetDataArray(), dtype=np.float64)
    rows = np.repeat(np.arange(nld2, dtype=np.int64), np.diff(I))
    del sp, T, I
    keep = np.abs(vals) > 1e-12                    # (an element's whole row is stored: 8 numbers, most of them zero)
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    done = np.zeros(nld2, dtype=bool)
    done[rows] = True
    tb = _blocks(fine, coarse, rows, cols, vals, done, ess, None)
    #: how many times coarser a side the coarse mesh is (2 for a mesh cut in two a side)
    dim = int(fine.mesh.Dimension())
    tb.cut = int(round((float(fine.mesh.GetGlobalNE()) / max(float(coarse.mesh.GetGlobalNE()), 1.0)) ** (1.0 / dim)))
    return tb

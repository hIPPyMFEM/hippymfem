# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Batched element data: quadrature geometry, basis tables, and dof maps.

The element kernels are vectorized over elements, so everything they need is
precomputed once and stored as contiguous arrays:

* per quadrature point, the reference basis values ``N[q, i]`` and reference
  gradients ``G[q, i, k]``, identical for every element of a given geometry
  and polynomial order, so computed once;
* per element and quadrature point, ``Jinv[e, q, k, d] = dxi_k / dx_d`` and the
  weight ``wdet[e, q] = w_q |det J|``, from MFEM's batched ``GeometricFactors``;
* per element, the (signed) dof indices into the *local* dof vector.

Physical gradients are **not** stored: they are formed inside the JIT'd kernel
as ``G @ Jinv``, which costs one fused contraction and saves a factor of
``ndof/sdim`` in memory.

MFEM's ``GeometricFactors`` uses a single integration rule for the whole mesh,
so meshes with more than one element geometry are split into groups and handled
one group at a time.
"""

import os

import numpy as np

from mpi4py import MPI

import mfem.par as mfem

from ..common.identitycache import IdentityCache
from ..common.keepalive import KeepAlive
from ..common.parvector import host_sync
from .spaces import as_space

#: Leave MFEM's cached ``GeometricFactors`` in place after the element geometry has
#: been copied out.  Off by default, since nothing in hIPPyMFEM reads them again;
#: set ``HIPPYMFEM_KEEP_GEOMETRIC_FACTORS=1`` if your own code holds a pointer from
#: ``mesh.GetGeometricFactors`` on a mesh hIPPyMFEM also batches.
KEEP_GEOMETRIC_FACTORS = os.environ.get(
    "HIPPYMFEM_KEEP_GEOMETRIC_FACTORS", "0").lower() in ("1", "yes", "true", "on")

#: MFEM sizes the vectors of ``GetGeometricFactors`` with an ``int``: the Jacobians
#: are ``nq * sdim * dim * NE``, which passes 2^31 at four million hexahedra and 64
#: quadrature points, and the kernel then reads out of bounds (an illegal memory
#: access, not an error: job 5856681, 400^3 on 16 H100).  Above the limit the
#: geometry is built here instead, in element slices, from the mesh nodes.  That
#: path also keeps the geometry off the card, where MFEM's costs ``nq*sdim*dim*NE``
#: doubles for something read once.
_MFEM_VECTOR_LIMIT = 2 ** 31 - 1

#: Elements per slice when the geometry is built here.  ``0`` sizes a slice at about
#: 128 MB of Jacobians; a positive value pins it and also *forces* this path, which
#: is how the two routes are compared.  Set ``HIPPYMFEM_GEOMETRY_SLICE``.
GEOMETRY_SLICE = int(os.environ.get("HIPPYMFEM_GEOMETRY_SLICE", "0") or 0)


def default_quadrature_degree(spaces, extra=2):
    """A deliberately generous default: twice the highest degree present, plus ``extra``.

    The residual density may be non-polynomial (``exp(m)``, ``1/m``, ...) so no
    exact rule exists.  What matters for correctness of the inverse problem is
    that *every* derivative block uses the *same* rule, which is automatic
    because they are all differentiated from one quadrature sum.
    """
    return 2 * max(s.order for s in spaces) + extra


class ElementGroup(KeepAlive):
    """Quadrature geometry for the elements of one geometry type.

    Attributes
    ----------
    geom : int
        ``mfem.Geometry`` type.
    elems : ndarray of int
        Mesh element indices in this group, ascending.
    ir : mfem.IntegrationRule
    w : ndarray, shape (nq,)
        Reference quadrature weights.
    wdet : ndarray, shape (ne, nq)
        ``w_q * |det J|``.
    Jinv : ndarray, shape (ne, nq, dim, sdim)
    X : ndarray, shape (ne, nq, sdim)
        Physical coordinates of the quadrature points.
    """

    def __init__(self, mesh, geom, elems, quadrature_degree, batched=None,
                 sliced=None):
        self.mesh = mesh
        self.geom = geom
        self.elems = np.asarray(elems, dtype=np.int64)
        self.dim = mesh.Dimension()
        self.sdim = mesh.SpaceDimension()
        self.ir = mfem.IntRules.Get(geom, int(quadrature_degree))
        self.keep(mesh, self.ir)
        self.nq = self.ir.GetNPoints()
        self.ne = int(self.elems.size)
        self.w = np.array([self.ir.IntPoint(q).weight for q in range(self.nq)])
        # Whether to use MFEM's batched GeometricFactors.  The caller decides (see
        # MeshBatches): that call is collective on a ParMesh (it builds the nodal
        # grid function), while the number of geometries is rank-LOCAL (on a mixed
        # mesh one rank may hold only triangles), so deciding here could deadlock.
        self.batched = batched
        # Whether to build the geometry here rather than with MFEM's batched call.
        # Like ``batched``, the caller decides (see MeshBatches): the nodal grid
        # function this path needs is built collectively on a ParMesh, while the
        # element count that motivates it is rank-local.
        self.sliced = sliced
        self._build_geometry()

    def _build_geometry(self):
        mesh, nq, dim, sdim = self.mesh, self.nq, self.dim, self.sdim
        batched = self.batched
        if batched is None:            # standalone use: decide locally
            batched = mesh.GetNumGeometries(dim) == 1
        sliced = self.sliced
        if sliced is None:             # standalone use: decide locally
            sliced = bool(GEOMETRY_SLICE) or (
                nq * sdim * dim * int(mesh.GetNE()) > _MFEM_VECTOR_LIMIT)
        if sliced:
            self._geometry_sliced()
            return
        if batched:
            gf = mesh.GetGeometricFactors(
                self.ir,
                mfem.GeometricFactors.JACOBIANS
                | mfem.GeometricFactors.DETERMINANTS
                | mfem.GeometricFactors.COORDINATES,
            )
            # With MFEM configured on a device these live in device memory, and
            # GetDataArray() would hand back stale host values.
            for arr in (gf.J, gf.detJ, gf.X):
                host_sync(arr)
            NE = mesh.GetNE()
            # MFEM column-major layouts: J is (NQ, SDIM, DIM, NE),
            # detJ is (NQ, NE), X is (NQ, SDIM, NE).
            J = gf.J.GetDataArray().reshape(nq, sdim, dim, NE, order="F")
            det = gf.detJ.GetDataArray().reshape(nq, NE, order="F")
            X = gf.X.GetDataArray().reshape(nq, sdim, NE, order="F")
            sel = self.elems
            J = np.ascontiguousarray(np.transpose(J, (3, 0, 1, 2))[sel])
            det = np.ascontiguousarray(np.transpose(det, (1, 0))[sel])
            self.X = np.ascontiguousarray(np.transpose(X, (2, 0, 1))[sel])
            # Everything above is a copy (fancy indexing copies), so MFEM's own cache
            # of the factors is dropped: on a device it holds J, detJ and X twice,
            # on host and device, gigabytes on a large mesh.  MFEM drops every cached
            # factor of the mesh at once; a later group on this mesh recomputes its own.
            del gf
            if not KEEP_GEOMETRIC_FACTORS:
                mesh.DeleteGeometricFactors()
        else:
            J, det, self.X = self._geometry_elementwise()

        # Jinv[k, d] = dxi_k / dx_d.  Square J is the common case; a manifold
        # mesh (sdim > dim) needs the pseudo-inverse.
        if sdim == dim:
            self.Jinv = np.ascontiguousarray(np.linalg.inv(J))
        else:
            self.Jinv = np.ascontiguousarray(np.linalg.pinv(J))
        self.detJ = det
        self.wdet = np.ascontiguousarray(self.w[None, :] * np.abs(det))

    def _geometry_sliced(self, slice_ne=None):
        """``Jinv``, ``detJ``, ``X`` and ``wdet`` from the mesh nodes, in slices.

        The geometry of a quadrature point is a product against the nodal basis,
        ``J(q) = sum_i x_i grad phi_i(q)`` and ``X(q) = sum_i x_i phi_i(q)``, so a
        slice of elements is one ``dgemm`` and the results go straight into the
        arrays the kernels stream from.  Nothing of the size of the whole geometry is
        ever held twice, which is what MFEM's batched call cannot avoid: it computes
        the Jacobians on the device, copies them to the host, and is then inverted
        into a second array of the same size.  See :data:`_MFEM_VECTOR_LIMIT` for why
        the large case has to come here.
        """
        mesh, nq, dim, sdim, ne = self.mesh, self.nq, self.dim, self.sdim, self.ne
        nodes = mesh.GetNodes()
        if nodes is None:
            mesh.EnsureNodes()                        # collective on a ParMesh
            nodes = mesh.GetNodes()
        nfes = nodes.FESpace()
        vdim = nfes.GetVDim()
        if vdim != sdim:
            raise NotImplementedError(
                "the nodal space has vdim %d on a mesh of space dimension %d"
                % (vdim, sdim))
        fe = nfes.GetFE(int(self.elems[0]))
        nd = fe.GetDof()
        N, G = _basis_tables(fe, self.ir, dim, nd)
        edofs, signs = _element_dofs(nfes, self.elems, vdim, nd)
        host_sync(nodes)
        xall = nodes.GetDataArray()
        # (nd, nq*dim) and (nd, nq): one matrix product per slice, BLAS-backed
        Gm = np.ascontiguousarray(np.transpose(G, (1, 0, 2)).reshape(nd, nq * dim))
        Nm = np.ascontiguousarray(N.T)
        self.Jinv = np.empty((ne, nq, dim, sdim))
        self.detJ = np.empty((ne, nq))
        self.X = np.empty((ne, nq, sdim))
        self.wdet = np.empty((ne, nq))
        signed = signs is not None and signs.size and signs.min() < 0
        m = int(slice_ne or GEOMETRY_SLICE
                or max(256, 2 ** 27 // max(nq * sdim * dim * 8, 1)))
        for a in range(0, ne, m):
            bnd = min(a + m, ne)
            x = xall[edofs[a:bnd]]
            if signed:
                x = x * signs[a:bnd]
            x = x.reshape((bnd - a) * vdim, nd)
            J = (x @ Gm).reshape(bnd - a, sdim, nq, dim).transpose(0, 2, 1, 3)
            self.X[a:bnd] = (x @ Nm).reshape(bnd - a, sdim, nq).transpose(0, 2, 1)
            if sdim == dim:
                det = np.linalg.det(J)
                self.Jinv[a:bnd] = np.linalg.inv(J)
            else:
                # a manifold mesh: MFEM's weight is sqrt(det(J^T J))
                det = np.sqrt(np.linalg.det(
                    np.einsum("...ij,...ik->...jk", J, J)))
                self.Jinv[a:bnd] = np.linalg.pinv(J)
            self.detJ[a:bnd] = det
            self.wdet[a:bnd] = self.w[None, :] * np.abs(det)

    def _geometry_elementwise(self):
        """Fallback for mixed-geometry meshes: loop over elements in Python."""
        mesh, nq, dim, sdim = self.mesh, self.nq, self.dim, self.sdim
        ne = self.ne
        J = np.zeros((ne, nq, sdim, dim))
        det = np.zeros((ne, nq))
        X = np.zeros((ne, nq, sdim))
        pt = mfem.Vector(sdim)
        for a, e in enumerate(self.elems):
            T = mesh.GetElementTransformation(int(e))
            for q in range(nq):
                ip = self.ir.IntPoint(q)
                T.SetIntPoint(ip)
                Jm = T.Jacobian()
                host_sync(Jm)
                J[a, q, :, :] = Jm.GetDataArray().reshape(sdim, dim, order="F")
                det[a, q] = T.Weight()
                T.Transform(ip, pt)
                host_sync(pt)
                X[a, q, :] = pt.GetDataArray()
        return J, det, X

    def __repr__(self):
        return "ElementGroup(geom=%d, ne=%d, nq=%d)" % (self.geom, self.ne, self.nq)

    def make_tables(self, space):
        """Basis tables of ``space`` on this group (a boundary group overrides this)."""
        from .vectorfe import VectorSpaceTables, is_vector_space

        if is_vector_space(space):
            return VectorSpaceTables(space, self)
        return SpaceTables(space, self)


class SpaceTables(KeepAlive):
    """Basis tables and element dof maps of one space on one element group.

    Attributes
    ----------
    N : ndarray, shape (nq, nd)
    G : ndarray, shape (nq, nd, dim)
    edofs : ndarray of int, shape (ne, vdim*nd)
        Indices into the space's *local* dof vector, component-major.
    signs : ndarray, shape (ne, vdim*nd)
        +-1 multipliers decoded from MFEM's signed vdof encoding.
    """

    def __init__(self, space, group):
        space = as_space(space)
        self.space = space
        self.group = group
        fes = space.fes
        self.vdim = space.vdim
        fe = fes.GetFE(int(group.elems[0]))
        self.nd = fe.GetDof()
        self.order = fe.GetOrder()
        self._check_uniform(fes, group)
        self.N, self.G = _basis_tables(fe, group.ir, group.dim, self.nd)
        self.edofs, self.signs = _element_dofs(fes, group.elems, self.vdim, self.nd)
        self.nd_total = self.vdim * self.nd
        self._dev_maps = {}

    def _check_uniform(self, fes, group):
        """All elements of a group must share one FE; variable order is not supported."""
        for e in group.elems[:: max(1, group.ne // 32)]:
            if fes.GetFE(int(e)).GetDof() != self.nd:
                raise NotImplementedError(
                    "variable-order spaces are not supported by the batched "
                    "element kernels (element %d has %d dofs, expected %d)"
                    % (int(e), fes.GetFE(int(e)).GetDof(), self.nd)
                )

    def gather(self, local_array):
        """Element dof values, shape ``(ne, vdim*nd)``, from a local dof array."""
        return local_array[self.edofs] * self.signs

    def gather_device(self, local_array):
        """Element dof values as a device array, gathered where the kernels run.

        ``local_array`` may be numpy (it is copied to the device once) or already
        a device array.  The dof map and the signs are cached on the device, so
        repeated assemblies upload only the dof values.
        """
        from .kernel import _put

        from .kernel import device

        d = device()
        if d not in self._dev_maps:
            sg = None if self.signs.min() > 0 else _put(self.signs)
            self._dev_maps[d] = (_put(self.edofs), sg)
        idx, sg = self._dev_maps[d]
        vals = _put(local_array)[idx]
        return vals if sg is None else vals * sg

    # ------------------------------------------------------- kernel interface
    def jax_tables(self):
        """Arrays the field evaluator needs: the reference basis and its gradient."""
        return (self.N, self.G)

    def jax_axes(self):
        """``in_axes`` for :meth:`jax_tables`; both are shared by the whole group."""
        return (None, None)

    def evaluator(self):
        """A function ``(dofs, tables, Jinv) -> Field`` for this slot."""
        vdim, nd = self.vdim, self.nd
        from .kernel import _eval_field

        def ev(dofs, tabs, Jinv):
            N, G = tabs
            return _eval_field(dofs, N, G, Jinv, vdim, nd)

        return ev

    def transform_dual_matrix(self, mats, trial_tables):
        """Map element matrices from the reference dof basis to the global one.

        A no-op for H1/L2, where the two coincide; H(curl) on simplices needs
        MFEM's ``DofTransformation``.
        """
        return mats

    def transform_dual_vector(self, vecs):
        """Map element vectors from the reference dof basis to the global one."""
        return vecs

    def __repr__(self):
        return "SpaceTables(nd=%d, vdim=%d, ne=%d)" % (self.nd, self.vdim, self.group.ne)


def _basis_tables(fe, ir, dim, nd):
    """Reference shape values and gradients at the quadrature points."""
    nq = ir.GetNPoints()
    N = np.zeros((nq, nd))
    G = np.zeros((nq, nd, dim))
    sh = mfem.Vector(nd)
    dsh = mfem.DenseMatrix(nd, dim)
    for q in range(nq):
        ip = ir.IntPoint(q)
        fe.CalcShape(ip, sh)
        fe.CalcDShape(ip, dsh)
        host_sync(sh)
        host_sync(dsh)
        N[q, :] = sh.GetDataArray()
        # DenseMatrix data is column-major
        G[q, :, :] = dsh.GetDataArray().reshape(nd, dim, order="F")
    return np.ascontiguousarray(N), np.ascontiguousarray(G)


def _element_dofs(fes, elems, vdim, nd):
    """Signed element vdof maps for a list of elements, ``(ne, vdim*nd)`` each.

    Read in bulk from the space's element-to-dof table, not with one
    ``GetElementVDofs`` call per element (seconds of SWIG calls on a fine mesh).
    The table holds the scalar dofs; the vdofs follow MFEM's ``DofToVDof``
    (``dof + k*ndofs`` by nodes, ``dof*vdim + k`` by vdim).  MFEM encodes a sign
    flip as a negative index ``-1-i``, and so does the table; H1/L2 spaces never
    produce one, but decoding costs nothing and keeps the gather correct in general.
    """
    elems = np.asarray(elems, dtype=np.int64)
    ne = elems.size
    if ne == 0:
        return (np.zeros((0, vdim * nd), dtype=np.int64),
                np.zeros((0, vdim * nd), dtype=np.float64))
    table = fes.GetElementToDofTable()
    nrows = table.Size()
    I = np.asarray(mfem.intArray((table.GetI(), nrows + 1)).GetDataArray(),
                   dtype=np.int64)
    J = np.asarray(mfem.intArray((table.GetJ(), int(I[-1]))).GetDataArray(),
                   dtype=np.int64)
    start = I[elems]
    if not np.array_equal(I[elems + 1] - start, np.full(ne, nd)):
        raise ValueError("the elements do not all carry %d dofs" % nd)
    raw = J[start[:, None] + np.arange(nd)[None, :]]              # (ne, nd), signed
    neg = raw < 0
    dof = np.where(neg, -1 - raw, raw)
    if vdim > 1:
        ndofs = fes.GetNDofs()
        k = np.arange(vdim)
        if fes.GetOrdering() == mfem.Ordering.byNODES:
            vdof = dof[:, None, :] + (k * ndofs)[None, :, None]
        else:
            vdof = dof[:, None, :] * vdim + k[None, :, None]
        dof = vdof.reshape(ne, vdim * nd)
        neg = np.broadcast_to(neg[:, None, :], vdof.shape).reshape(ne, vdim * nd)
    signs = np.where(neg, -1.0, 1.0)
    return np.ascontiguousarray(dof), np.ascontiguousarray(signs)


_TABLE_CACHE = IdentityCache()


def group_tables(space, group):
    """Cached :class:`SpaceTables` for one space on one element group.

    Memoized on the identity of the space's ``ParFiniteElementSpace`` and the
    group.  Building tables is purely rank-local (shape evaluation and the
    element-to-dof table), so a hit on one rank and a miss on another is harmless.
    """
    space = as_space(space)
    tab = _TABLE_CACHE.get((space.fes, group))
    if tab is None:
        tab = _TABLE_CACHE.put((space.fes, group), group.make_tables(space))
    return tab


def clear_table_cache():
    """Drop every cached :class:`SpaceTables` (for tests, and after refinement)."""
    _TABLE_CACHE.clear()


class MeshBatches(KeepAlive):
    """All element groups of a mesh for one quadrature degree, with table caching.

    Construction is collective: whether the batched ``GeometricFactors`` path can
    be used is decided from the **global** number of element geometries, because
    that call is collective on a ParMesh and the local geometry count is not the
    same on every rank.
    """

    def __init__(self, mesh, quadrature_degree, comm=None):
        self.mesh = mesh
        self.quadrature_degree = int(quadrature_degree)
        self.keep(mesh)
        self.comm = comm if comm is not None else _mesh_comm(mesh)
        dim = mesh.Dimension()
        local_ngeom = mesh.GetNumGeometries(dim)
        self.ngeom = self.comm.allreduce(local_ngeom, op=MPI.MAX)
        batched = self.ngeom == 1
        geoms = {}
        if local_ngeom == 1 and mesh.GetNE():
            # one geometry: no per-element query
            geoms[mesh.GetElementGeometry(0)] = list(range(mesh.GetNE()))
        else:
            for e in range(mesh.GetNE()):
                geoms.setdefault(mesh.GetElementGeometry(e), []).append(e)
        sliced = self._sliced(mesh, geoms)
        self.groups = [
            ElementGroup(mesh, g, np.array(v, dtype=np.int64),
                         self.quadrature_degree, batched=batched, sliced=sliced)
            for g, v in sorted(geoms.items())
        ]
        self._tables = IdentityCache()

    def _sliced(self, mesh, geoms):
        """Whether to build the geometry in element slices, decided collectively.

        MFEM's batched call would overflow its int-sized vectors on a rank with a few
        million elements (:data:`_MFEM_VECTOR_LIMIT`), and the slice path needs the
        nodal grid function, whose construction is collective, so the answer must be
        the same on every rank: the local test is reduced with MAX.  The vectors are
        sized from the *mesh* element count, not the group's, so every geometry
        present is measured against the whole mesh.
        """
        dim, sdim = mesh.Dimension(), mesh.SpaceDimension()
        ne = int(mesh.GetNE())
        worst = 0
        for g in geoms:
            nq = mfem.IntRules.Get(g, self.quadrature_degree).GetNPoints()
            worst = max(worst, nq * sdim * dim * ne)
        local = bool(GEOMETRY_SLICE) or worst > _MFEM_VECTOR_LIMIT
        return self.comm.allreduce(int(local), op=MPI.MAX) > 0

    def tables(self, space):
        """Cached :class:`SpaceTables` for ``space``, one per group."""
        fes = as_space(space).fes
        got = self._tables.get((fes,))
        if got is None:
            got = self._tables.put((fes,), [group_tables(space, g) for g in self.groups])
        return got

    @property
    def nelem(self):
        return self.mesh.GetNE()

    def __repr__(self):
        return "MeshBatches(qdeg=%d, groups=%d, NE=%d)" % (
            self.quadrature_degree,
            len(self.groups),
            self.mesh.GetNE(),
        )


def _mesh_comm(mesh):
    """MPI communicator of a ParMesh, falling back to ``COMM_WORLD``."""
    get = getattr(mesh, "GetComm", None)
    if get is not None:
        try:
            c = get()
            if isinstance(c, MPI.Comm):
                return c
        except Exception:
            pass
    return MPI.COMM_WORLD


_BATCH_CACHE = IdentityCache()


def get_batches(mesh, quadrature_degree, comm=None):
    """Cached :class:`MeshBatches` for ``(mesh, quadrature_degree)``.

    The cache key is the mesh identity and the degree, both of which every rank
    agrees on, so the cached construction is hit or missed uniformly and its
    internal collective stays synchronized.
    """
    got = _BATCH_CACHE.get((mesh,), int(quadrature_degree))
    if got is None:
        got = _BATCH_CACHE.put((mesh,), MeshBatches(mesh, quadrature_degree, comm),
                               int(quadrature_degree))
    return got

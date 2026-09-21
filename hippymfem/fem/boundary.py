# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Boundary integrals in the AD route.

The domain kernels in :mod:`.kernel` cover ``\\int_\\Omega`` only, which leaves out
Neumann and Robin conditions, boundary sources, and any misfit or quantity of
interest that lives on the boundary.  UFL gets those from ``ds``; this module is
the equivalent here.

A boundary density is a function of the same fields plus the outward unit normal::

    def bdr_varf(u, m, p, x, n):
        return kappa * (u.val - g(x)) * p.val          # Robin
        # or jnp.exp(m.val) * jnp.dot(u.grad, n) * p.val   (flux)

and it is differentiated exactly like the domain density, so every block (``A``,
``C``, ``W_uu``, ``W_um``, ``W_mm``, third derivatives) picks up its boundary part
automatically.

**The fields carry their full gradient, not just their trace.**  A boundary
element's own finite element can only give tangential derivatives, so instead each
face quadrature point is mapped back into the *adjacent volume element* with
MFEM's ``FaceElementTransformations``; the shape functions and gradients are
evaluated there, and the element dofs are the volume element's.  That is what makes
``u.grad``, and therefore the normal flux ``dot(u.grad, n)``, available.  The
price is that the basis tables vary per boundary element instead of being shared,
which costs ``nbe * nq * nd`` doubles: boundary elements are a lower-dimensional
set, so this is small next to the domain tables.

Geometry is built with a Python loop over boundary elements for the same reason
(``nbe`` grows only as ``N^((d-1)/d)``), and the loop avoids MFEM's face-geometry
batching, whose availability depends on rank-local face counts.
"""

import numpy as np

from mpi4py import MPI

import mfem.par as mfem

from ..common.identitycache import IdentityCache
from ..common.keepalive import KeepAlive
from .spaces import as_space
from .elementbatch import _element_dofs, group_tables
from .csrassemble import assemble_matrix_csr, assemble_vector_csr

__all__ = [
    "BoundaryGroup",
    "BoundarySpaceTables",
    "BoundaryBatches",
    "BoundaryKernel",
    "assemble_boundary_matrix",
    "assemble_boundary_vector",
    "boundary_marker",
]


def boundary_marker(mesh, bdr_attributes="all"):
    """An MFEM attribute marker array from ``"all"``, ``None``, or a list of attributes."""
    nattr = int(mesh.bdr_attributes.Max()) if mesh.bdr_attributes.Size() else 0
    m = mfem.intArray(nattr)
    if bdr_attributes is None or (isinstance(bdr_attributes, str)
                                  and bdr_attributes == "all"):
        m.Assign(1)
        return m
    m.Assign(0)
    for a in np.atleast_1d(bdr_attributes):
        a = int(a)
        if not 1 <= a <= nattr:
            raise ValueError("boundary attribute %d outside 1..%d" % (a, nattr))
        m[a - 1] = 1
    return m


class BoundaryGroup(KeepAlive):
    """Face quadrature data for boundary elements sharing one geometry pair.

    Attributes
    ----------
    elems : ndarray
        Boundary element indices, ascending.  Named ``elems`` so that the
        assembly pattern, which orders entries by ``group.elems`` to match
        MFEM's own loop, needs no special case.
    vol_elems : ndarray
        Index of the volume element adjacent to each boundary element; the dof
        maps come from these.
    wdet : ndarray, shape (ne, nq)
        ``w_q`` times the **face** measure.
    Jinv : ndarray, shape (ne, nq, dim, sdim)
        Inverse Jacobian of the *volume* element at the mapped point.
    X : ndarray, shape (ne, nq, sdim)
    nor : ndarray, shape (ne, nq, sdim)
        Outward **unit** normal.
    eip : ndarray, shape (ne, nq, dim)
        Volume reference coordinates of the face quadrature points.
    h : ndarray, shape (ne, nq)
        The adjacent element's measure divided by the face measure, which is what a
        Nitsche or DG penalty is scaled by, and the same quantity MFEM's
        ``DGDiffusionIntegrator`` uses on a boundary face.  A density that takes it
        after the normal is given it; one that does not is left alone.
    """

    def __init__(self, mesh, bdr_elems, quadrature_degree, geom=None):
        self.mesh = mesh
        self.dim = mesh.Dimension()
        self.sdim = mesh.SpaceDimension()
        bdr_elems = np.asarray(bdr_elems, dtype=np.int64)
        self.geom = (int(geom) if geom is not None
                     else int(mesh.GetBdrElementGeometry(int(bdr_elems[0]))))
        self.ir = mfem.IntRules.Get(self.geom, int(quadrature_degree))
        self.nq = self.ir.GetNPoints()
        self.w = np.array([self.ir.IntPoint(q).weight for q in range(self.nq)])
        self.keep(mesh, self.ir)
        self._build(bdr_elems)
        self.ne = int(self.elems.size)

    def _build(self, bdr_elems):
        mesh, nq, dim, sdim = self.mesh, self.nq, self.dim, self.sdim
        keep, vol = [], []
        for b in bdr_elems:
            ftr = mesh.GetBdrFaceTransformations(int(b))
            if ftr is None:
                continue             # a non-conforming slave face; reported below
            keep.append(int(b))
            vol.append(int(ftr.Elem1No))
        if len(keep) != len(bdr_elems):
            raise NotImplementedError(
                "%d of %d boundary elements have no face transformation "
                "(non-conforming boundary faces are not supported by the "
                "boundary kernels)" % (len(bdr_elems) - len(keep), len(bdr_elems)))
        self.elems = np.array(keep, dtype=np.int64)
        self.vol_elems = np.array(vol, dtype=np.int64)
        ne = len(keep)

        self.wdet = np.zeros((ne, nq))
        self.Jinv = np.zeros((ne, nq, dim, sdim))
        self.X = np.zeros((ne, nq, sdim))
        self.nor = np.zeros((ne, nq, sdim))
        self.eip = np.zeros((ne, nq, dim))
        self.detJ = np.zeros((ne, nq))
        self.h = np.zeros((ne, nq))

        eip = mfem.IntegrationPoint()
        nor = mfem.Vector(sdim)
        coords = mfem.DenseMatrix()
        for a, b in enumerate(self.elems):
            ftr = mesh.GetBdrFaceTransformations(int(b))
            # physical coordinates of the whole face rule in one call; the
            # per-point Transform overload does not fill its output here
            ftr.Face.Transform(self.ir, coords)
            self.X[a] = coords.GetDataArray().reshape(
                sdim, nq, order="F").T
            T1 = ftr.GetElement1Transformation()
            for q in range(nq):
                ip = self.ir.IntPoint(q)
                ftr.Loc1.Transform(ip, eip)
                self.eip[a, q, 0] = eip.x
                if dim > 1:
                    self.eip[a, q, 1] = eip.y
                if dim > 2:
                    self.eip[a, q, 2] = eip.z
                ftr.Face.SetIntPoint(ip)
                w = ftr.Face.Weight()
                self.wdet[a, q] = self.w[q] * w
                self.detJ[a, q] = w
                if sdim == dim:
                    mfem.CalcOrtho(ftr.Face.Jacobian(), nor)
                    v = nor.GetDataArray().copy()
                    self.nor[a, q] = v / max(np.linalg.norm(v), 1e-300)
                T1.SetIntPoint(eip)
                self.h[a, q] = T1.Weight() / max(w, 1e-300)
                J = T1.Jacobian().GetDataArray().reshape(sdim, dim, order="F")
                self.Jinv[a, q] = (np.linalg.inv(J) if sdim == dim
                                   else np.linalg.pinv(J))

    def __repr__(self):
        return "BoundaryGroup(geom=%d, ne=%d, nq=%d)" % (self.geom, self.ne,
                                                         self.nq)

    def make_tables(self, space):
        return BoundarySpaceTables(space, self)


class BoundarySpaceTables(KeepAlive):
    """Shape values and gradients of the adjacent volume element, per boundary element.

    The layout matches :class:`~.elementbatch.SpaceTables` except that ``N`` and
    ``G`` carry a leading element axis, because the face-to-volume map differs
    from face to face.
    """

    def __init__(self, space, group):
        space = as_space(space)
        self.space = space
        self.group = group
        fes = space.fes
        self.vdim = space.vdim
        fe = fes.GetFE(int(group.vol_elems[0]))
        self.nd = fe.GetDof()
        self.order = fe.GetOrder()
        self.nd_total = self.vdim * self.nd
        self._dev_maps = {}
        self._build(fes, group)

    def _build(self, fes, group):
        ne, nq, dim, nd = group.ne, group.nq, group.dim, self.nd
        self.N = np.zeros((ne, nq, nd))
        self.G = np.zeros((ne, nq, nd, dim))
        sh = mfem.Vector(nd)
        dsh = mfem.DenseMatrix(nd, dim)
        ip = mfem.IntegrationPoint()
        for a in range(ne):
            fe = fes.GetFE(int(group.vol_elems[a]))
            if fe.GetDof() != nd:
                raise NotImplementedError(
                    "variable-order spaces are not supported by the boundary "
                    "kernels (element %d has %d dofs, expected %d)"
                    % (int(group.vol_elems[a]), fe.GetDof(), nd))
            for q in range(nq):
                e = group.eip[a, q]
                ip.Set3(float(e[0]),
                        float(e[1]) if dim > 1 else 0.0,
                        float(e[2]) if dim > 2 else 0.0)
                fe.CalcShape(ip, sh)
                fe.CalcDShape(ip, dsh)
                self.N[a, q] = sh.GetDataArray()
                self.G[a, q] = dsh.GetDataArray().reshape(nd, dim, order="F")

        self.edofs, self.signs = _element_dofs(fes, group.vol_elems, self.vdim, nd)

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

    def jax_tables(self):
        return (self.N, self.G)

    def jax_axes(self):
        """Per boundary element: the face-to-volume map differs from face to face."""
        return (0, 0)

    def evaluator(self):
        vdim, nd = self.vdim, self.nd
        from .kernel import _eval_field

        def ev(dofs, tabs, Jinv):
            N, G = tabs
            return _eval_field(dofs, N, G, Jinv, vdim, nd)

        return ev

    def transform_dual_matrix(self, mats, trial_tables):
        return mats

    def transform_dual_vector(self, vecs):
        return vecs

    def __repr__(self):
        return "BoundarySpaceTables(nd=%d, vdim=%d, ne=%d)" % (
            self.nd, self.vdim, self.group.ne)


class BoundaryBatches(KeepAlive):
    """Boundary element groups of a mesh for one quadrature degree and marker.

    Grouped by boundary geometry *and* by the number of dofs of the adjacent
    volume element, so that every group has uniform array shapes even on a mixed
    mesh where a triangle and a quadrilateral share an edge of the boundary.
    """

    def __init__(self, mesh, quadrature_degree, bdr_attributes="all", comm=None,
                 space=None):
        self.mesh = mesh
        self.quadrature_degree = int(quadrature_degree)
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.keep(mesh)
        self.marker = boundary_marker(mesh, bdr_attributes)
        sel = []
        for b in range(mesh.GetNBE()):
            attr = mesh.GetBdrAttribute(b)
            if 1 <= attr <= self.marker.Size() and self.marker[attr - 1]:
                sel.append(b)
        buckets = {}
        for b in sel:
            key = int(mesh.GetBdrElementGeometry(b))
            if space is not None:
                ftr = mesh.GetBdrFaceTransformations(int(b))
                if ftr is None:
                    continue
                key = (key, as_space(space).fes.GetFE(int(ftr.Elem1No)).GetDof())
            buckets.setdefault(key, []).append(b)
        self.groups = [
            BoundaryGroup(mesh, np.array(v, dtype=np.int64),
                          self.quadrature_degree,
                          geom=k[0] if isinstance(k, tuple) else k)
            for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        ]
        self._tables = IdentityCache()

    def tables(self, space):

        fes = as_space(space).fes
        got = self._tables.get((fes,))
        if got is None:
            got = self._tables.put((fes,), [group_tables(space, g) for g in self.groups])
        return got

    @property
    def nelem(self):
        return self.mesh.GetNBE()

    def __repr__(self):
        return "BoundaryBatches(qdeg=%d, groups=%d, NBE=%d)" % (
            self.quadrature_degree, len(self.groups), self.mesh.GetNBE())


_BDR_CACHE = IdentityCache()


def get_boundary_batches(mesh, quadrature_degree, bdr_attributes="all", comm=None,
                         space=None):
    """Cached :class:`BoundaryBatches`.

    Keyed on values every rank agrees on, so the construction is hit or missed
    uniformly.
    """
    attrs = ("all" if isinstance(bdr_attributes, str) or bdr_attributes is None
             else tuple(int(a) for a in np.atleast_1d(bdr_attributes)))
    objs = (mesh, None if space is None else as_space(space).fes)
    extra = (int(quadrature_degree), attrs)
    got = _BDR_CACHE.get(objs, extra)
    if got is None:
        got = _BDR_CACHE.put(objs, BoundaryBatches(mesh, quadrature_degree,
                                                   bdr_attributes, comm, space), extra)
    return got


def clear_boundary_cache():
    """Drop every cached :class:`BoundaryBatches`."""
    _BDR_CACHE.clear()


# ----------------------------------------------------------------- assembly
def assemble_boundary_matrix(test_space, trial_space, groups, element_matrices,
                             test_ess=None, trial_ess=None, diag_policy="one"):
    """Assemble a boundary block, always through the direct-CSR path.

    The callback route cannot serve these arrays: its integrators are domain
    integrators, and the element arrays here are indexed by boundary element but
    carry the adjacent *volume* element's dofs.
    """

    return assemble_matrix_csr(test_space, trial_space, groups, element_matrices,
                               0, test_ess=test_ess, trial_ess=trial_ess,
                               diag_policy=diag_policy)


def assemble_boundary_vector(space, groups, element_vectors, ess=None, out=None):
    """Assemble a boundary dual vector through the direct-CSR path."""

    return assemble_vector_csr(space, groups, element_vectors, 0, ess=ess,
                               out=out)


# ------------------------------------------------------------------- kernels


def __getattr__(name):
    """:class:`~.kernel.BoundaryKernel` lives with the kernels (it is the
    :class:`~.kernel.QuadratureKernel` over :class:`BoundaryBatches`) and is
    resolved on first use, so importing this module does not load JAX."""
    if name == "BoundaryKernel":
        import importlib

        return importlib.import_module("hippymfem.fem.kernel").BoundaryKernel
    raise AttributeError(name)

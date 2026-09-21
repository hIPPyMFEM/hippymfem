# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Interior facet terms: the ``dS`` of a discontinuous Galerkin form.

:mod:`.boundary` covers :math:`\int_{\partial\Omega}`, where a face has one adjacent
element.  An interior face has two, and a DG form is written in their traces: the jump
:math:`[\![u]\!] = u^- - u^+` and the average :math:`\{u\} = (u^- + u^+)/2`, with
:math:`-` the side the face normal points out of (MFEM's ``Elem1``).  A facet density is
a function of those::

    def facet_varf(u, m, p, x, n, h):
        return (-jnp.dot(avg_grad(u), n) * jump(p)
                - jnp.dot(avg_grad(p), n) * jump(u)
                + kappa * 0.5 * (1.0 / h[0] + 1.0 / h[1]) * jump(u) * jump(p))

and it is differentiated exactly like a domain density, so every block picks up its
facet part.  Each field arrives as a :class:`FacetField` carrying the two traces, and
``h`` holds the two elements' measures, each divided by the face measure (see
:class:`InteriorFacetGroup`).

**One slot, two elements.**  The dof vector of a slot on a facet is the two elements'
dofs concatenated, the layout of MFEM's ``AssembleFaceMatrix``, so a second derivative
of the density *is* a face matrix, with no rearrangement.  The geometry carries both
elements' inverse Jacobians stacked, which is the only thing the group kernels of
:mod:`.kernel` need to serve two sides.

**Faces shared between ranks.**  The second element of a shared face lives on a
neighbour, so a facet block couples dofs this rank does not have.  It is assembled the
way MFEM assembles its own DG forms: over local dofs, with the neighbour's columns
carrying their global numbers, and folded to true dofs by the prolongation afterwards.
Each rank takes the rows of the element on its side of a shared face and the other rank
takes the rest, so the face is counted once.  :mod:`.csrassemble`'s pattern cannot be
reused for this, since it holds the couplings within one element and a facet couples
two.
"""

import numpy as np
from typing import NamedTuple

import mfem.par as mfem

from ..common.identitycache import IdentityCache
from ..common.keepalive import KeepAlive
from .elementbatch import _element_dofs, default_quadrature_degree
from .spaces import as_space


class FacetField(NamedTuple):
    """A field at an interior facet point, from both sides.

    ``minus`` is the element the face normal points out of, ``plus`` the other; each is
    a :class:`~hippymfem.fem.kernel.Field` with ``val`` and ``grad``.  Use
    :func:`jump` and :func:`avg` on it rather than reaching for the sides, so a density
    reads like the form it comes from.
    """

    minus: object
    plus: object


def jump(f):
    r""":math:`[\![f]\!] = f^- - f^+`, for a field or either of its parts."""
    if isinstance(f, FacetField):
        return f.minus.val - f.plus.val
    return f.minus - f.plus


def avg(f):
    r""":math:`\{f\} = (f^- + f^+)/2`."""
    if isinstance(f, FacetField):
        return 0.5 * (f.minus.val + f.plus.val)
    return 0.5 * (f.minus + f.plus)


def jump_grad(f):
    r""":math:`[\![\nabla f]\!]`."""
    return f.minus.grad - f.plus.grad


def avg_grad(f):
    r""":math:`\{\nabla f\}`."""
    return 0.5 * (f.minus.grad + f.plus.grad)


#: Whether this PyMFEM build's two-argument ``GetFaceNbrElementVDofs`` works; ``None``
#: until the first shared face settles it (see ``_face_nbr_vdofs``).
_FNBR_TWO_ARG = None


def _face_nbr_vdofs(fes, k):
    """The signed vdofs of a face-neighbour element, whichever wrapper this build has.

    PyMFEM's one-argument wrapper for this call reads an undefined name in some builds
    (it passes ``i`` where it means ``args[0]``), and the two-argument form, which
    fills an ``intArray``, is missing from others.  The first call finds out which one
    works and the rest follow it.
    """
    global _FNBR_TWO_ARG

    if _FNBR_TWO_ARG is not False:
        try:
            vdofs = mfem.intArray()
            fes.GetFaceNbrElementVDofs(int(k), vdofs)
            _FNBR_TWO_ARG = True
            return np.asarray(vdofs.ToList(), dtype=np.int64)
        except (TypeError, NotImplementedError):
            if _FNBR_TWO_ARG:
                raise
            _FNBR_TWO_ARG = False
    return np.asarray(fes.GetFaceNbrElementVDofs(int(k)), dtype=np.int64)


class InteriorFacetGroup(KeepAlive):
    """Quadrature data of interior faces sharing one geometry and one dof count.

    Attributes
    ----------
    elems : ndarray
        Mesh face indices of the local interior faces, then ``-1`` for each face
        shared with another rank.
    e1, e2 : ndarray
        The two adjacent elements.  For a shared face ``e2`` is
        ``mesh.GetNE() + k``, the face-neighbour numbering MFEM uses.
    wdet : ndarray, shape (nf, nq)
        ``w_q`` times the face measure.
    Jinv : ndarray, shape (nf, nq, 2, dim, sdim)
        Inverse Jacobians of the two elements at the mapped points, ``minus`` first.
    X : ndarray, shape (nf, nq, sdim)
    nor : ndarray, shape (nf, nq, sdim)
        Unit normal, pointing out of ``e1``.
    h : ndarray, shape (nf, nq, 2)
        Both elements' measures divided by the face measure, the quantity MFEM's DG
        integrators penalise with: their ``{h^{-1}}`` is the average
        ``(1/h[0] + 1/h[1])/2``.  A density receives it as a pair, so any other
        convention (the smaller of the two, say) is the user's to write.
    eip1, eip2 : ndarray, shape (nf, nq, dim)
        Reference coordinates of the face points inside each element.
    """

    #: the face measure reaches every facet density, which cannot penalise without it
    h_always = True

    def __init__(self, mesh, quadrature_degree, geom=None):
        # A hanging node splits one side of a face into several, so MFEM visits it as a
        # master and its slaves, and the pairing built below (one element on each side,
        # one quadrature rule for both) does not describe it.  Assembling anyway produces
        # a facet term that is quietly inconsistent with the residual, and the first sign
        # of it is a Newton step that fails to solve a linear problem, which says nothing
        # about the cause.  Refuse it here instead.  Domain and boundary integrals on a
        # non-conforming mesh are supported and unaffected.
        nonconforming = getattr(mesh, "Nonconforming", None)
        if nonconforming is not None and bool(nonconforming()):
            raise ValueError(
                "interior facet densities (facet_varf) are not supported on a "
                "non-conforming mesh: its hanging nodes make a face a master with "
                "slaves, which the facet kernels do not assemble.  Domain and boundary "
                "densities are supported there; for facet terms use a conforming mesh.")
        self.mesh = mesh
        self.dim = mesh.Dimension()
        self.sdim = mesh.SpaceDimension()
        self.quadrature_degree = int(quadrature_degree)
        self._tables = IdentityCache()
        self._build(geom)
        self.ne = int(self.e1.size)
        self.keep(mesh)

    # ------------------------------------------------------------------ build
    def _faces(self):
        """``(face, e1, e2, shared)`` for every interior face this rank assembles.

        Local interior faces first, in face order, then the shared faces in MFEM's
        shared-face order, which is the order ``ParBilinearForm`` visits them in.
        """
        mesh = self.mesh
        # Collective: every rank reaches it, even one with no shared face, before
        # anything asks about a shared face or a face-neighbour element.
        if hasattr(mesh, "ExchangeFaceNbrData"):
            mesh.ExchangeFaceNbrData()
        out = []
        for f in range(mesh.GetNumFaces()):
            ftr = mesh.GetInteriorFaceTransformations(f)
            if ftr is None:
                continue
            out.append((f, int(ftr.Elem1No), int(ftr.Elem2No), False))
        ns = mesh.GetNSharedFaces() if hasattr(mesh, "GetNSharedFaces") else 0
        for k in range(ns):
            ftr = mesh.GetSharedFaceTransformations(k)
            out.append((-1, int(ftr.Elem1No), int(ftr.Elem2No), True))
        return out

    def _transformation(self, face, shared, index):
        return (self.mesh.GetSharedFaceTransformations(index) if shared
                else self.mesh.GetInteriorFaceTransformations(face))

    def _build(self, geom):
        dim, sdim = self.dim, self.sdim
        faces = self._faces()
        self.faces = faces
        nf = len(faces)
        self.elems = np.array([f for f, _, _, _ in faces], dtype=np.int64)
        self.e1 = np.array([a for _, a, _, _ in faces], dtype=np.int64)
        self.e2 = np.array([b for _, _, b, _ in faces], dtype=np.int64)
        self.shared = np.array([s for _, _, _, s in faces], dtype=bool)
        self.index = np.arange(nf, dtype=np.int64)
        if nf == 0:                       # a rank with no interior face of its own
            self.geom = int(geom) if geom is not None else -1
            self.nq = 0
            for name, shape in (("wdet", (0, 0)), ("X", (0, 0, sdim)),
                                ("nor", (0, 0, sdim)), ("h", (0, 0, 2)),
                                ("eip1", (0, 0, dim)), ("eip2", (0, 0, dim))):
                setattr(self, name, np.zeros(shape))
            self.Jinv = np.zeros((0, 0, 2, dim, sdim))
            self.w = np.zeros(0)
            return
        f0, _, _, s0 = faces[0]
        ftr0 = self._transformation(f0, s0, 0)
        self.geom = int(ftr0.GetGeometryType()) if geom is None else int(geom)
        self.ir = mfem.IntRules.Get(self.geom, self.quadrature_degree)
        self.nq = self.ir.GetNPoints()
        self.w = np.array([self.ir.IntPoint(q).weight for q in range(self.nq)])
        self.keep(self.ir)
        nq = self.nq
        self.wdet = np.zeros((nf, nq))
        self.Jinv = np.zeros((nf, nq, 2, dim, sdim))
        self.X = np.zeros((nf, nq, sdim))
        self.nor = np.zeros((nf, nq, sdim))
        self.h = np.zeros((nf, nq, 2))
        self.eip1 = np.zeros((nf, nq, dim))
        self.eip2 = np.zeros((nf, nq, dim))
        eip = mfem.IntegrationPoint()
        nor = mfem.Vector(sdim)
        coords = mfem.DenseMatrix()
        shared_seen = 0
        for a, (face, _, _, shared) in enumerate(faces):
            idx = shared_seen if shared else 0
            ftr = self._transformation(face, shared, idx)
            if shared:
                shared_seen += 1
            ftr.Face.Transform(self.ir, coords)
            self.X[a] = coords.GetDataArray().reshape(sdim, nq, order="F").T
            T1, T2 = ftr.GetElement1Transformation(), ftr.GetElement2Transformation()
            for q in range(nq):
                ip = self.ir.IntPoint(q)
                ftr.Face.SetIntPoint(ip)
                w = ftr.Face.Weight()
                self.wdet[a, q] = self.w[q] * w
                if sdim == dim:
                    mfem.CalcOrtho(ftr.Face.Jacobian(), nor)
                    v = nor.GetDataArray().copy()
                    self.nor[a, q] = v / max(np.linalg.norm(v), 1e-300)
                for side, (loc, T, eips) in enumerate(
                        ((ftr.Loc1, T1, self.eip1), (ftr.Loc2, T2, self.eip2))):
                    loc.Transform(ip, eip)
                    eips[a, q, 0] = eip.x
                    if dim > 1:
                        eips[a, q, 1] = eip.y
                    if dim > 2:
                        eips[a, q, 2] = eip.z
                    T.SetIntPoint(eip)
                    J = T.Jacobian().GetDataArray().reshape(sdim, dim, order="F")
                    self.Jinv[a, q, side] = (np.linalg.inv(J) if sdim == dim
                                             else np.linalg.pinv(J))
                    self.h[a, q, side] = T.Weight() / max(w, 1e-300)

    def make_tables(self, space):
        """Cached :class:`FacetSpaceTables` for ``space`` on this group."""
        space = as_space(space)
        got = self._tables.get((space.fes,))
        if got is None:
            got = self._tables.put((space.fes,), FacetSpaceTables(space, self))
        return got

    def __repr__(self):
        return "InteriorFacetGroup(geom=%d, nf=%d, nq=%d)" % (self.geom, self.ne,
                                                              self.nq)


class FacetSpaceTables(KeepAlive):
    """Shape values and gradients of both adjacent elements, per interior face.

    The layout matches :class:`~.boundary.BoundarySpaceTables` twice over: ``N1``,
    ``G1`` for the ``minus`` side and ``N2``, ``G2`` for ``plus``, each with a leading
    face axis, because the face-to-element map differs from face to face.  A slot's dof
    vector is the two elements' dofs concatenated, so ``nd_total`` is their sum.
    """

    def __init__(self, space, group):
        space = as_space(space)
        self.space = space
        self.group = group
        self.vdim = space.vdim
        fes = space.fes
        self.ne_local = fes.GetParMesh().GetNE() if hasattr(fes, "GetParMesh") else None
        if group.ne:
            fe = fes.GetFE(int(group.e1[0]))
            self.nd = fe.GetDof()
        else:
            self.nd = 0
        self.nd_total = self.vdim * 2 * self.nd
        self._dev_maps = IdentityCache()
        self._build(fes, group)

    def _side_fe(self, fes, e):
        """The element's finite element; a face-neighbour element on a shared face."""
        ne = self.ne_local
        if ne is not None and e >= ne:
            return fes.GetFaceNbrFE(int(e - ne))
        return fes.GetFE(int(e))

    def _build(self, fes, group):
        # Collective: the face-neighbour dofs and elements of a shared face live on
        # the other rank until this is called, and every rank calls it.
        if hasattr(fes, "ExchangeFaceNbrData"):
            fes.ExchangeFaceNbrData()
        nf, nq, dim, nd = group.ne, group.nq, group.dim, self.nd
        self.N1 = np.zeros((nf, nq, nd))
        self.G1 = np.zeros((nf, nq, nd, dim))
        self.N2 = np.zeros((nf, nq, nd))
        self.G2 = np.zeros((nf, nq, nd, dim))
        if nf == 0:
            self.edofs = np.zeros((0, 0), dtype=np.int64)
            self.signs = np.ones((0, 0))
            return
        sh = mfem.Vector(nd)
        dsh = mfem.DenseMatrix(nd, dim)
        ip = mfem.IntegrationPoint()
        for a in range(nf):
            for side, (e, eips, N, G) in enumerate(
                    ((group.e1[a], group.eip1, self.N1, self.G1),
                     (group.e2[a], group.eip2, self.N2, self.G2))):
                fe = self._side_fe(fes, int(e))
                if fe.GetDof() != nd:
                    raise NotImplementedError(
                        "the facet kernels need one dof count per group (element %d "
                        "has %d dofs, expected %d)" % (int(e), fe.GetDof(), nd))
                for q in range(nq):
                    c = eips[a, q]
                    ip.Set3(float(c[0]), float(c[1]) if dim > 1 else 0.0,
                            float(c[2]) if dim > 2 else 0.0)
                    fe.CalcShape(ip, sh)
                    fe.CalcDShape(ip, dsh)
                    N[a, q] = sh.GetDataArray()
                    G[a, q] = dsh.GetDataArray().reshape(nd, dim, order="F")
        # Dof maps into the extended numbering of facet_values: this rank's local
        # dofs, then MFEM's face-neighbour values, where the far side of a shared face
        # lives.  Indexing both as one array keeps one gather.
        self.nlocal = int(fes.GetVSize())
        local = group.e2 < (self.ne_local if self.ne_local is not None
                            else group.e2.max() + 1)
        d1, s1 = _element_dofs(fes, group.e1, self.vdim, nd)
        d2 = np.zeros_like(d1)
        s2 = np.ones_like(s1)
        if local.any():
            d2[local], s2[local] = _element_dofs(fes, group.e2[local], self.vdim, nd)
        for a in np.nonzero(~local)[0]:
            raw = _face_nbr_vdofs(fes, int(group.e2[a] - self.ne_local))
            neg = raw < 0
            d2[a] = self.nlocal + np.where(neg, -1 - raw, raw)
            s2[a] = np.where(neg, -1.0, 1.0)
        self.edofs = np.concatenate([d1, d2], axis=1)
        self.signs = np.concatenate([s1, s2], axis=1)
        self.is_local = local

    # ------------------------------------------------------- kernel interface
    def jax_tables(self):
        """The four per-face tables the two-sided evaluator needs."""
        return (self.N1, self.G1, self.N2, self.G2)

    def jax_axes(self):
        """Every table carries the face axis: the map differs from face to face."""
        return (0, 0, 0, 0)

    def evaluator(self):
        """``(dofs, tables, Jinv) -> FacetField``; ``dofs`` is both sides concatenated."""
        vdim, nd = self.vdim, self.nd
        from .kernel import _eval_field

        def ev(dofs, tabs, Jinv):
            N1, G1, N2, G2 = tabs
            half = vdim * nd
            return FacetField(
                _eval_field(dofs[:half], N1, G1, Jinv[:, 0], vdim, nd),
                _eval_field(dofs[half:], N2, G2, Jinv[:, 1], vdim, nd))

        return ev

    def gather(self, values):
        """Both elements' dof values, ``(nf, 2*vdim*nd)``.

        ``values`` is what :func:`facet_values` returns: this rank's local dof values
        followed by the face-neighbour ones, which is where the far side of a shared
        face lives.  A purely local batch may pass the local array alone.
        """
        values = np.asarray(values)
        if values.shape[0] < self.nlocal + 1 and np.any(self.edofs >= self.nlocal):
            raise RuntimeError(
                "this batch has faces shared with another rank, whose far side needs "
                "the face-neighbour values; gather from facet_values(space, v)")
        return values[self.edofs] * self.signs

    def gather_device(self, local_array):
        from .kernel import _put
        import jax.numpy as jnp

        idx = self._dev_maps.get((self,), extra=("idx",))
        if idx is None:
            idx = self._dev_maps.put((self,), (_put(self.edofs), _put(self.signs)),
                                     extra=("idx",))
        e, s = idx
        arr = local_array if hasattr(local_array, "device") else _put(local_array)
        return jnp.take(arr, e, axis=0) * s

    def transform_dual_matrix(self, mats, trial_tables):
        """No reference-to-global map: the facet kernels cover H1 and L2 spaces."""
        return mats

    def transform_dual_vector(self, vecs):
        return vecs

    def __repr__(self):
        return "FacetSpaceTables(nd=%d, vdim=%d, nf=%d)" % (self.nd, self.vdim,
                                                            self.group.ne)


class FacetBatches(KeepAlive):
    """The interior faces of a mesh, in the shape the group kernels take.

    One group: interior faces of a conforming mesh share a geometry, and the dof count
    is the element's.  Built like :class:`~.boundary.BoundaryBatches` so that
    :class:`~hippymfem.fem.kernel.QuadratureKernel` takes either without noticing.
    """

    def __init__(self, mesh, quadrature_degree, comm=None):
        self.mesh = mesh
        self.quadrature_degree = int(quadrature_degree)
        self.groups = [InteriorFacetGroup(mesh, self.quadrature_degree)]
        self._tables = IdentityCache()
        self.keep(mesh)

    def tables(self, space):
        """Cached :class:`FacetSpaceTables` for ``space``, one per group."""
        space = as_space(space)
        got = self._tables.get((space.fes,))
        if got is None:
            got = self._tables.put((space.fes,),
                                   [g.make_tables(space) for g in self.groups])
        return got

    @property
    def nfaces(self):
        return sum(g.ne for g in self.groups)


_FACET_BATCHES = IdentityCache()


def get_facet_batches(mesh, quadrature_degree=None, space=None, comm=None):
    """Cached :class:`FacetBatches` for a mesh and a quadrature degree."""
    if quadrature_degree is None:
        quadrature_degree = default_quadrature_degree([space] if space is not None
                                                      else [])
    quadrature_degree = int(quadrature_degree)
    got = _FACET_BATCHES.get((mesh,), extra=(quadrature_degree,))
    if got is None:
        got = _FACET_BATCHES.put((mesh,), FacetBatches(mesh, quadrature_degree, comm),
                                 extra=(quadrature_degree,))
    return got


def clear_facet_cache():
    """Drop every cached facet batch, pattern and dof map (after refinement, and in tests).

    The assembly skeleton and the global dof numbers are cached on the spaces and the
    mesh they were built from, so a refinement that keeps those objects alive must say
    so; :func:`~hippymfem.fem.pattern.clear_pattern_cache` is the domain counterpart.
    """
    _FACET_BATCHES.clear()
    _FACET_PATTERNS.clear()
    _GLOBAL_LDOFS.clear()


# ------------------------------------------------------------------- assembly
_GLOBAL_LDOFS = IdentityCache()


def global_ldof_map(space):
    """Global ldof numbers of the extended dofs: this rank's own, then its neighbours'.

    :class:`FacetSpaceTables` numbers the far side of a shared face after this rank's
    own dofs, and this map turns that numbering into the global one a parallel matrix
    takes as its columns.  They are *ldof* numbers, one per rank per shared dof, not
    true dofs; the facet block is folded onto true dofs by the prolongation, exactly as
    MFEM folds its own DG blocks.  MFEM keeps the same map in
    ``face_nbr_glob_dof_map`` and hands it back as a bare pointer, so it is rebuilt
    here by exchanging the numbers themselves.  Collective; cached per space.
    """
    space = as_space(space)
    got = _GLOBAL_LDOFS.get((space.fes,))
    if got is not None:
        return got
    from .prolongation import _ldof_offset
    from ..common.parvector import host_sync, host_readwrite

    fes = space.fes
    # The offset comes from the prolongation's row partition, so that the columns
    # written here and the P they are folded by cannot disagree about the partition.
    glob = _ldof_offset(space)[0] + np.arange(int(fes.GetVSize()), dtype=np.int64)
    if not hasattr(fes, "ExchangeFaceNbrData"):
        return _GLOBAL_LDOFS.put((space.fes,), glob)
    fes.ExchangeFaceNbrData()                      # collective
    gf = mfem.ParGridFunction(fes)
    host_readwrite(gf)
    gf.GetDataArray()[:] = glob                    # exact in double below 2^53 dofs
    gf.ExchangeFaceNbrData()                       # collective
    nbr = gf.FaceNbrData()
    host_sync(nbr)
    extra = (np.rint(np.asarray(nbr.GetDataArray(), dtype=np.float64)).astype(np.int64)
             if nbr.Size() else np.zeros(0, dtype=np.int64))
    return _GLOBAL_LDOFS.put((space.fes,), np.concatenate([glob, extra]))


class _FacetPattern(KeepAlive):
    """The CSR skeleton of a facet block, built once and refilled on every assembly.

    Rows are this rank's test ldofs; columns are global ldof numbers of the trial
    space, its own dofs and its face neighbours' (:func:`global_ldof_map`).  A face
    shared with another rank contributes only the rows of the element on this side,
    and the neighbour assembles the rest, as ``ParBilinearForm::AssembleSharedFaces``
    divides the work.  Rows that belong to the neighbour are sent to a sentinel row
    and dropped, which is why the skeleton is built row-sorted.
    """

    def __init__(self, groups, test_tables, trial_tables, gcols, nrow):
        self.nrow = int(nrow)
        self.sizes = []
        rows, cols, sign = [], [], []
        for g, tt, ut in zip(groups, test_tables, trial_tables):
            nr, nc = tt.edofs.shape[1], ut.edofs.shape[1]
            self.sizes.append(g.ne * nr * nc)
            if g.ne == 0:
                continue
            r = tt.edofs.astype(np.int64, copy=True)
            if g.shared.any():
                r[g.shared, nr // 2:] = self.nrow         # the neighbour's rows
            rows.append(np.repeat(r, nc, axis=1).reshape(-1))
            cols.append(np.tile(gcols[ut.edofs], (1, nr)).reshape(-1))
            sign.append((tt.signs[:, :, None] * ut.signs[:, None, :]).reshape(-1))
        self.sign = (np.concatenate(sign) if sign else np.zeros(0))
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        order = np.lexsort((cols, rows))
        r, c = rows[order], cols[order]
        first = np.ones(r.size, dtype=bool)
        if r.size:
            first[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
        self.slot = np.empty(r.size, dtype=np.int64)
        self.slot[order] = np.cumsum(first) - 1
        self.nslot = int(first.sum())
        self.indices = c[first]
        self.indptr = np.zeros(self.nrow + 1, dtype=np.int32)
        self.indptr[1:] = np.cumsum(np.bincount(r[first], minlength=self.nrow + 1)
                                    [:self.nrow])
        self.nnz = int(self.indptr[-1])              # the sentinel row's entries are last

    def data(self, element_matrices):
        """The CSR values of one assembly, from one face-matrix array per group."""
        flat = []
        for m, n in zip(element_matrices, self.sizes):
            a = np.asarray(m, dtype=np.float64).reshape(-1)
            if a.size != n:
                raise ValueError("facet matrices of %d entries, expected %d: the batch "
                                 "and the pattern disagree" % (a.size, n))
            flat.append(a)
        flat = np.concatenate(flat) if flat else np.zeros(0)
        vals = np.bincount(self.slot, weights=flat * self.sign, minlength=self.nslot)
        return vals[:self.nnz]


_FACET_PATTERNS = IdentityCache()


def _facet_pattern(test, trial, groups, test_tables, trial_tables):
    """Cached :class:`_FacetPattern` for a space pair on one set of facet groups."""
    test_tables = test_tables or [g.make_tables(test) for g in groups]
    trial_tables = (test_tables if trial.fes is test.fes
                    else trial_tables or [g.make_tables(trial) for g in groups])
    key = (test_tables[0], trial_tables[0]) if groups else (test.fes, trial.fes)
    got = _FACET_PATTERNS.get(key)
    if got is None:
        got = _FACET_PATTERNS.put(
            key, _FacetPattern(groups, test_tables, trial_tables,
                               global_ldof_map(trial), test.fes.GetVSize()))
    return got


def _facet_ldof_matrix(pattern, values, test, trial):
    """The facet block as a ``HypreParMatrix`` over the two spaces' ldof partitions.

    The columns are already global, so nothing is communicated here; hypre splits them
    into its diagonal and off-diagonal blocks and the constructor copies every array it
    is given.  MFEM's ``ParBilinearForm`` builds the same kind of matrix for a form with
    interior face integrators.
    """
    from .prolongation import _ldof_offset
    from ..common.parvector import _HYPRE_INT
    from ..common.linalg import own

    roff, rglob = _ldof_offset(test)
    same = trial.fes is test.fes
    coff, cglob = (roff, rglob) if same else _ldof_offset(trial)
    I = np.ascontiguousarray(pattern.indptr, dtype=np.int32)
    J = np.ascontiguousarray(pattern.indices, dtype=_HYPRE_INT)
    D = np.ascontiguousarray(values, dtype=np.float64)
    rows = np.array([roff, roff + pattern.nrow], dtype=_HYPRE_INT)
    # One partition array for a square block, as hypre wants it: it compares the two
    # pointers to decide whether to bring each row's diagonal entry to the front.
    args = ([I, J, D, rows] if same else
            [I, J, D, rows, np.array([coff, coff + trial.fes.GetVSize()],
                                     dtype=_HYPRE_INT)])
    A = mfem.HypreParMatrix(test.comm, pattern.nrow, rglob, cglob, args)
    A.CopyRowStarts()
    A.CopyColStarts()
    return own(A, *args)


def assemble_facet_matrix(test_space, groups, element_matrices, trial_space=None,
                          test_tables=None, trial_tables=None):
    """Assemble interior facet terms into a parallel true-dof matrix.

    ``element_matrices`` is one ``(nf, nd_test1 + nd_test2, nd_trial1 + nd_trial2)``
    array per group, as :meth:`element_matrices
    <hippymfem.fem.kernel.QuadratureKernel.element_matrices>` gives for a facet batch:
    rows the test dofs of both elements, columns the trial dofs of both, ``minus`` side
    first.  With ``trial_space`` the block may be rectangular, as the parameter blocks
    of an inverse problem are; square and rectangular blocks go the same way.
    """
    from .parmat import _triple
    from .prolongation import _prolongation

    test = as_space(test_space)
    trial = test if trial_space is None else as_space(trial_space)
    pattern = _facet_pattern(test, trial, groups, test_tables, trial_tables)
    A = _facet_ldof_matrix(pattern, pattern.data(element_matrices), test, trial)
    same = trial.fes is test.fes
    P, identity = _prolongation(test)
    if same:
        return A if identity else _triple(A, None, P, test.comm, (test.fes,))
    Pr, ridentity = _prolongation(trial)
    if identity and ridentity:
        return A                       # two discontinuous spaces: ldofs are true dofs
    return _triple(A, P, Pr, test.comm, (test.fes, trial.fes))


def assemble_facet_vector(space, groups, element_vectors, tables=None, out=None):
    """Assemble a facet residual into a dual (true-dof) vector.

    ``element_vectors`` is one ``(nf, 2*nd)`` array per group, the two elements'
    entries concatenated, as a first derivative of a facet density gives them.

    **No communication.**  Each rank scatters the entries of the element the normal
    leaves, which it owns, for every face it sees; a face shared with another rank is
    seen by both, once from each side, so between them the face's whole contribution
    lands.  A face interior to this rank is seen once and both halves are scattered
    here.  That is the same division of labour MFEM's own DG assembly makes, and it is
    why the matrix needs MFEM's face-neighbour dofs while the vector does not.
    """
    space = as_space(space)
    nl = space.fes.GetVSize()
    local = np.zeros(nl)
    for g, tabs, v in zip(groups, tables or [None] * len(groups), element_vectors):
        if g.ne == 0:
            continue
        if tabs is None:
            tabs = g.make_tables(space)
        v = np.asarray(v, dtype=np.float64)
        half = v.shape[1] // 2
        edofs, signs = tabs.edofs, tabs.signs
        np.add.at(local, edofs[:, :half].reshape(-1),
                  (v[:, :half] * signs[:, :half]).reshape(-1))
        own = ~g.shared                      # the far side is this rank's as well
        if own.any():
            np.add.at(local, edofs[own][:, half:].reshape(-1),
                      (v[own][:, half:] * signs[own][:, half:]).reshape(-1))
    return space.assemble_dual(local, out)


def facet_values(space, v, gf=None):
    """Local dof values followed by MFEM's face-neighbour values, for a facet gather.

    The far side of a face shared with another rank is not in this rank's dof vector;
    MFEM carries those values in ``ParGridFunction.FaceNbrData`` once
    ``ExchangeFaceNbrData`` has run, and :class:`FacetSpaceTables` indexes the two as
    one array.  Collective.
    """
    space = as_space(space)
    gf = space.to_gridfunction(v, gf)
    from ..common.parvector import host_sync

    host_sync(gf)
    local = np.asarray(gf.GetDataArray(), dtype=np.float64).copy()
    if not hasattr(gf, "ExchangeFaceNbrData"):
        return local
    gf.ExchangeFaceNbrData()                       # collective, every rank reaches it
    nbr = gf.FaceNbrData()
    host_sync(nbr)
    extra = np.asarray(nbr.GetDataArray(), dtype=np.float64).copy() if nbr.Size() else np.zeros(0)
    return np.concatenate([local, extra])

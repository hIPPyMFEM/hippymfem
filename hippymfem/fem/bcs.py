# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Essential (Dirichlet) boundary conditions.

hIPPYlib takes two boundary condition lists: ``bc`` with the real data, used for
the forward solve, and ``bc0``, its homogeneous counterpart, used for the
adjoint and incremental solves.  The same pair appears here.

The discrete convention, which matches what hIPPYlibx does through dolfinx, is a
single uniform rule:

    for any operator block, zero the essential entries of the **input** when the
    trial space carries essential conditions, and of the **output** when the
    test space does.

Matrices follow suit, block by block.  This table is the one statement of the
convention; :mod:`.assemble` implements it and
:class:`~hippymfem.modeling.PDEVariationalProblem.PDEVariationalProblem` relies on it.

======================  =========================================================
block                   essential-dof treatment
======================  =========================================================
``A``, ``A^T``          rows *and* columns eliminated, **unit** diagonal, so a
                        solve against a right-hand side that is zero on the
                        essential dofs returns an increment that is zero there
``W_uu``                rows *and* columns eliminated, **zero** diagonal
``C``, ``W_um``         rows of the test space eliminated
``W_mm``                untouched
``C^T``, ``W_mu``       the transposes, which carry the condition to the
                        correct side
======================  =========================================================

With those eliminations baked into the matrices, ``apply_ij`` is a plain matrix
product and needs no masking, and the result agrees with hIPPYlib's matrix-free
``set_bc`` bookkeeping block for block.
"""

import numpy as np

import mfem.par as mfem

from ..common.keepalive import KeepAlive
from .spaces import as_space


class DirichletBC(KeepAlive):
    """Essential boundary condition on part of the boundary of a space.

    Parameters
    ----------
    space : FunctionSpace or ParFiniteElementSpace
    value : float, callable, or None
        Boundary data.  A callable receives physical coordinates as a numpy
        array and returns a float (or a length-``vdim`` array).  On a vector
        space a single number applies to every component.  ``None`` means
        homogeneous.
    bdr_attributes : sequence of int, "all", or None
        Mesh boundary attributes (MFEM's 1-based numbering) where the condition
        applies.  ``"all"`` selects the whole boundary.
    marker : callable, optional
        Alternative to ``bdr_attributes``: ``marker(x) -> bool`` applied to the
        coordinates of boundary element vertices; a boundary element is selected
        when all of its vertices satisfy it.
    component : int, optional
        For vector spaces, restrict the condition to one component.
    """

    def __init__(self, space, value=None, bdr_attributes="all", marker=None,
                 component=-1):
        self.space = as_space(space)
        self.value = value
        self.component = int(component)
        fes = self.space.fes
        mesh = self.space.mesh
        nattr = mesh.bdr_attributes.Size() and mesh.bdr_attributes.Max() or 0
        self.nattr = int(nattr)

        if marker is not None:
            self.bdr_marker = self._marker_to_attributes(marker)
        else:
            self.bdr_marker = self._attributes_to_marker(bdr_attributes)

        ess = mfem.intArray()
        if self.component >= 0:
            fes.GetEssentialTrueDofs(self.bdr_marker, ess, self.component)
        else:
            fes.GetEssentialTrueDofs(self.bdr_marker, ess)
        self.ess_tdof = ess
        self.keep(ess, self.bdr_marker)
        self.ess = np.array(ess.ToList(), dtype=np.int64) if ess.Size() else np.zeros(0, np.int64)

    # ------------------------------------------------------------ selection
    def _attributes_to_marker(self, attrs):
        m = mfem.intArray(self.nattr)
        if attrs is None or (isinstance(attrs, str) and attrs == "all"):
            m.Assign(1)
            return m
        m.Assign(0)
        for a in np.atleast_1d(attrs):
            a = int(a)
            if not 1 <= a <= self.nattr:
                raise ValueError(
                    "boundary attribute %d outside 1..%d" % (a, self.nattr)
                )
            m[a - 1] = 1
        return m

    def _marker_to_attributes(self, marker):
        """Select boundary elements by coordinate predicate.

        Implemented by re-attributing: boundary elements whose vertices all
        satisfy ``marker`` get a fresh attribute number, so that MFEM's
        attribute machinery can be used unchanged.  The mesh is modified, which
        is why ``bdr_attributes`` is the preferred route.
        """
        mesh = self.space.mesh
        new_attr = self.nattr + 1
        selected = []
        for b in range(mesh.GetNBE()):
            verts = mesh.GetBdrElementVertices(b)
            ok = True
            for v in verts:
                x = np.array(mesh.GetVertexArray(int(v)))
                if not marker(x):
                    ok = False
                    break
            if ok:
                selected.append(b)
        for b in selected:
            mesh.SetBdrAttribute(b, new_attr)
        mesh.SetAttributes()
        self.nattr = int(mesh.bdr_attributes.Max()) if mesh.bdr_attributes.Size() else 0
        m = mfem.intArray(self.nattr)
        m.Assign(0)
        if new_attr <= self.nattr:
            m[new_attr - 1] = 1
        return m

    # ------------------------------------------------------------- homogeneous
    def homogeneous(self):
        """The same condition with zero data (hIPPYlib's ``bc0``)."""
        out = DirichletBC.__new__(DirichletBC)
        out.space = self.space
        out.value = None
        out.component = self.component
        out.nattr = self.nattr
        out.bdr_marker = self.bdr_marker
        out.ess_tdof = self.ess_tdof
        out.ess = self.ess
        out.keep(self.bdr_marker, self.ess_tdof)
        return out

    # ------------------------------------------------------------- application
    def apply(self, v):
        """Set the essential entries of the true-dof vector ``v`` to the data.

        The boundary values depend only on the data, the marker and the space, so
        they are projected once and reused; projecting into a fresh
        ``ParGridFunction`` on every call and keeping it would leak one state-sized
        array (with its device mirror) per forward solve.  Reassigning ``value``
        invalidates the cache.
        """
        if self.value is None:
            return self.zero(v)
        got = getattr(self, "_ess_values", None)
        if got is None or got[0] is not self.value:
            gf = mfem.ParGridFunction(self.space.fes)
            gf.Assign(0.0)
            coeff = _as_coefficient(self.value, self.space.vdim)
            gf.ProjectBdrCoefficient(coeff, self.bdr_marker)
            tv = self.space.vector()
            gf.GetTrueDofs(tv.hypre)
            got = self._ess_values = (self.value, tv.array[self.ess].copy())
            del gf, coeff, tv                  # nothing downstream points at them
        if self.ess.size:
            v.array[self.ess] = got[1]
        return v

    def zero(self, v):
        """Set the essential entries of ``v`` to zero."""
        if self.ess.size:
            v.array[self.ess] = 0.0
        return v


class BCSet(KeepAlive):
    """A collection of :class:`DirichletBC` on one space, acting as one."""

    def __init__(self, bcs=None, space=None):
        if bcs is None:
            bcs = []
        elif isinstance(bcs, DirichletBC):
            bcs = [bcs]
        else:
            bcs = list(bcs)
        self.bcs = bcs
        self.space = as_space(space) if space is not None else (
            bcs[0].space if bcs else None
        )
        if bcs:
            ess = np.unique(np.concatenate([b.ess for b in bcs])) if any(
                b.ess.size for b in bcs
            ) else np.zeros(0, np.int64)
        else:
            ess = np.zeros(0, np.int64)
        self.ess = ess.astype(np.int64)
        self.ess_tdof = mfem.intArray(self.ess.tolist())
        self.keep(self.ess_tdof, *bcs)

    def __bool__(self):
        return bool(self.ess.size) or bool(self.bcs)

    def __len__(self):
        return len(self.bcs)

    def __iter__(self):
        return iter(self.bcs)

    def homogeneous(self):
        return BCSet([b.homogeneous() for b in self.bcs], self.space)

    def apply(self, v):
        for b in self.bcs:
            b.apply(v)
        return v

    def zero(self, v):
        if self.ess.size:
            v.array[self.ess] = 0.0
        return v

    def zero_copy(self, v, out=None):
        """A copy of ``v`` with essential entries zeroed (leaves ``v`` alone)."""
        if out is None:
            out = v.copy()
        else:
            out.assign(v)
        return self.zero(out)


def as_bcset(bcs, space=None):
    """Coerce ``bcs`` (None / DirichletBC / list / BCSet) to a :class:`BCSet`."""
    if isinstance(bcs, BCSet):
        return bcs
    return BCSet(bcs, space)


def _as_coefficient(value, vdim):
    if isinstance(value, mfem.Coefficient) or isinstance(value, mfem.VectorCoefficient):
        return value
    if callable(value):
        if vdim == 1:
            return _ScalarBdr(value)
        return _VectorBdr(value, vdim)
    if vdim == 1:
        return mfem.ConstantCoefficient(float(value))
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if arr.size == 1:
        # one number on a vector space means the same value in every component
        arr = np.repeat(arr, vdim)
    if arr.size != vdim:
        raise ValueError(
            "boundary value has %d components but the space has vdim %d"
            % (arr.size, vdim))
    v = mfem.Vector(np.ascontiguousarray(arr))
    return mfem.VectorConstantCoefficient(v)


class _ScalarBdr(mfem.PyCoefficient):
    def __init__(self, fn):
        super(_ScalarBdr, self).__init__()
        self.fn = fn

    def EvalValue(self, x):
        return float(self.fn(np.asarray(x)))


class _VectorBdr(mfem.VectorPyCoefficient):
    def __init__(self, fn, vdim):
        super(_VectorBdr, self).__init__(vdim)
        self.fn = fn

    def EvalValue(self, x):
        return np.asarray(self.fn(np.asarray(x)), dtype=np.float64)

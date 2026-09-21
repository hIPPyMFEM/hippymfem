# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Finite element space wrappers.

A thin layer over ``mfem.ParFiniteElementSpace`` whose main job is lifetime
management: MFEM keeps raw pointers to the mesh and the finite element
collection, so a space built from a temporary collection crashes the moment it
is used.  :class:`FunctionSpace` owns its pieces.
"""

import numpy as np

import mfem.par as mfem

from ..common.linalg import own
from ..common.keepalive import KeepAlive
from ..common.parvector import ParVector, _fes_comm, host_readwrite, host_sync

#: collections the element kernels cover.  H1/L2 carry scalar (or vdim-replicated)
#: Lagrange dofs; ND/RT are vector valued, Piola mapped, and may carry MFEM
#: DofTransformations, all handled in :mod:`hippymfem.fem.vectorfe`.
_SAFE_FAMILIES = ("H1", "L2", "H1Pos", "H1_Trace", "L2_T", "ND", "RT")


class FunctionSpace(KeepAlive):
    """A parallel finite element space together with everything it points at.

    Parameters
    ----------
    mesh : mfem.ParMesh
    fec : mfem.FiniteElementCollection
    vdim : int
        Number of solution components.
    ordering : int
        ``mfem.Ordering.byNODES`` (default) or ``byVDIM``.  The element vdof
        array is component-major in both cases, which is what the element
        kernels assume.
    """

    def __init__(self, mesh, fec, vdim=1, ordering=None):
        if ordering is None:
            ordering = mfem.Ordering.byNODES
        self.mesh = mesh
        self.fec = fec
        self.vdim = int(vdim)
        self.fes = mfem.ParFiniteElementSpace(mesh, fec, self.vdim, ordering)
        self.keep(mesh, fec, self.fes)
        self.comm = _fes_comm(self.fes)
        self._check_family()
        # Build the true-dof layout here, where construction is unambiguously
        # collective, so that every later vector() is communication-free.
        self._layout = ParVector.from_fes(self.fes, self.comm).layout

    # ----------------------------------------------------------- constructors
    @classmethod
    def H1(cls, mesh, order, vdim=1, ordering=None):
        """Continuous Lagrange space of degree ``order``."""
        fec = mfem.H1_FECollection(order, mesh.Dimension())
        return cls(mesh, fec, vdim, ordering)

    @classmethod
    def L2(cls, mesh, order, vdim=1, ordering=None):
        """Discontinuous Lagrange space of degree ``order``."""
        fec = mfem.L2_FECollection(order, mesh.Dimension())
        return cls(mesh, fec, vdim, ordering)

    @classmethod
    def ND(cls, mesh, order):
        """Nedelec H(curl) space of degree ``order``.

        Fields from this space carry ``val`` (a vector) and ``curl``, not ``grad``;
        see :mod:`hippymfem.fem.vectorfe`.
        """
        fec = mfem.ND_FECollection(order, mesh.Dimension())
        return cls(mesh, fec, 1)

    @classmethod
    def RT(cls, mesh, order):
        """Raviart-Thomas H(div) space of degree ``order``.

        Fields from this space carry ``val`` (a vector) and ``div``.  Note MFEM's
        numbering: ``RT(0)`` is the lowest-order space.
        """
        fec = mfem.RT_FECollection(order, mesh.Dimension())
        return cls(mesh, fec, 1)

    @classmethod
    def wrap(cls, fes, mesh=None, fec=None):
        """Adopt an existing ``ParFiniteElementSpace`` without rebuilding it."""
        obj = cls.__new__(cls)
        obj.fes = fes
        obj.mesh = mesh if mesh is not None else fes.GetParMesh()
        obj.fec = fec if fec is not None else fes.FEColl()
        obj.vdim = fes.GetVDim()
        obj.comm = _fes_comm(fes)
        obj.keep(fes, obj.mesh, obj.fec)
        obj._check_family()
        obj._layout = ParVector.from_fes(fes, obj.comm).layout
        return obj

    def _check_family(self):
        name = self.fec.Name()
        if not any(name.startswith(f) for f in _SAFE_FAMILIES):
            raise NotImplementedError(
                "hIPPyMFEM element kernels support H1, L2, ND (H(curl)) and RT "
                "(H(div)) collections; got %r." % name
            )

    # ------------------------------------------------------------- properties
    @property
    def dim(self):
        """Reference (topological) dimension of the mesh."""
        return self.mesh.Dimension()

    @property
    def sdim(self):
        """Space (physical) dimension of the mesh."""
        return self.mesh.SpaceDimension()

    @property
    def order(self):
        """Polynomial degree (of element 0)."""
        return self.fes.GetOrder(0)

    def GetTrueVSize(self):
        return self.fes.GetTrueVSize()

    def GlobalTrueVSize(self):
        return self.fes.GlobalTrueVSize()

    def GetVSize(self):
        return self.fes.GetVSize()

    def __repr__(self):
        return "FunctionSpace(%s, vdim=%d, tdofs=%d)" % (
            self.fec.Name(),
            self.vdim,
            self.fes.GlobalTrueVSize(),
        )

    # ------------------------------------------------------------- conversion
    def vector(self):
        """A zero true-dof :class:`ParVector` on this space.

        The layout is built once and shared, so repeated calls inside a loop do
        not communicate.
        """
        return ParVector(self.comm, self.fes.GetTrueVSize(), layout=self._layout)

    def gridfunction(self, name=None):
        """A zero ``ParGridFunction`` on this space.

        The function keeps the space alive, not the other way round: a space
        keeping every grid function it made would leak one state-sized array per
        call.
        """
        gf = mfem.ParGridFunction(self.fes)
        gf.Assign(0.0)
        return own(gf, self)                 # the function reads the space's fes

    def to_gridfunction(self, v, gf=None):
        """Scatter a true-dof vector into a ``ParGridFunction`` (with ghosts)."""
        if gf is None:
            gf = own(mfem.ParGridFunction(self.fes), self)
        gf.SetFromTrueDofs(v.hypre)
        return gf

    def from_gridfunction(self, gf, v=None):
        """Restrict a ``ParGridFunction`` to a true-dof vector."""
        if v is None:
            v = self.vector()
        gf.GetTrueDofs(v.hypre)
        return v

    def local_values(self, v, out=None):
        """Local dof values of a true-dof vector, **including ghosts**, as numpy.

        This is ``P v``, the gather element kernels need.  A numpy array is
        returned rather than an ``mfem.Vector`` so that callers never have to
        reach for ``GetDataArray()``, whose views do not keep their owner alive
        (see :func:`hippymfem.common.parvector.to_numpy`).
        """
        n = self.fes.GetVSize()
        scratch = mfem.Vector(n)
        P = self.fes.GetProlongationMatrix()
        if P is None:
            scratch.GetDataArray()[:] = v.array
        else:
            P.Mult(v.hypre, scratch)
        if out is None:
            out = np.empty(n, dtype=np.float64)
        host_sync(scratch)
        out[:] = scratch.GetDataArray()
        return out

    def assemble_dual(self, local, v=None):
        """Assemble a local dual (residual) vector to true dofs: ``P^T local``.

        ``local`` may be a numpy array of local dof values or an ``mfem.Vector``.
        """
        if v is None:
            v = self.vector()
        if isinstance(local, np.ndarray):
            scratch = mfem.Vector(np.ascontiguousarray(local, dtype=np.float64))
        else:
            scratch = local
        P = self.fes.GetProlongationMatrix()
        if P is None:
            v.array[:] = scratch.GetDataArray()
        else:
            P.MultTranspose(scratch, v.hypre)
        return v

    def project(self, fn, v=None):
        """Project a python callable ``fn(x) -> float or array`` onto the space.

        On a nodal (Lagrange) space the projection is interpolation at the dof
        nodes, and ``fn`` is evaluated there directly: once on the whole
        ``(n, sdim)`` array of node coordinates when it accepts one, point by
        point otherwise, with no MFEM callback per point either way.  Other
        spaces (H(curl), H(div)) go through MFEM's ``ProjectCoefficient``.  The
        point-by-point evaluation reproduces the callback route bit for bit; a
        vectorized ``fn`` runs numpy's array kernels, whose transcendental
        functions can differ from the scalar ones in the last bit.
        """
        if not self.is_nodal:
            gf = mfem.ParGridFunction(self.fes)
            coeff = _ScalarPy(fn) if self.vdim == 1 else _VectorPy(fn, self.vdim)
            gf.ProjectCoefficient(coeff)
            return self.from_gridfunction(gf, v)   # gf and coeff are done with here
        X = self._node_coordinates()                # (nldof_scalar, sdim)
        vals = _evaluate_at(fn, X, self.vdim)       # (n,) or (n, vdim)
        gf = mfem.ParGridFunction(self.fes)
        host_readwrite(gf)                          # the write must be seen (see coordinates)
        arr = gf.GetDataArray()
        if self.vdim == 1:
            arr[:] = vals
        elif self.fes.GetOrdering() == mfem.Ordering.byNODES:
            arr[:] = vals.T.reshape(-1)             # component-major
        else:
            arr[:] = vals.reshape(-1)
        return self.from_gridfunction(gf, v)

    @property
    def is_nodal(self):
        """Whether the dofs are point values at nodes (H1 and L2 Lagrange spaces)."""
        name = self.fec.Name()
        return name.startswith("H1") or name.startswith("L2")

    def _node_coordinates(self):
        """Physical coordinates of every local scalar dof, ``(nldof, sdim)``.

        One element transformation per element (its reference nodes mapped in one
        call) rather than one Python callback per node and coordinate: at P2 on
        hexahedra that is 3 SWIG calls per element instead of 81.
        """
        fes, mesh, sdim = self.fes, self.mesh, self.sdim
        nl = fes.GetNDofs()
        X = np.zeros((nl, sdim))
        from .elementbatch import _element_dofs

        ne = mesh.GetNE()
        if ne:
            nd = fes.GetFE(0).GetDof()
            dofs, _ = _element_dofs(fes, np.arange(ne), 1, nd)
            pm = mfem.DenseMatrix()
            for e in range(ne):
                fe = fes.GetFE(e)
                tr = mesh.GetElementTransformation(e)
                tr.Transform(fe.GetNodes(), pm)     # (sdim, nd)
                X[dofs[e], :] = np.asarray(pm.GetDataArray()).reshape(sdim, -1).T[:, :sdim]
        return X

    def coordinates(self):
        """Physical coordinates of the scalar dofs as an ``(ntdof, sdim)`` array.

        Only meaningful for Lagrange (nodal) spaces with ``vdim == 1``.
        """
        if self.vdim != 1:
            raise ValueError("coordinates() is for scalar spaces")
        X = self._node_coordinates()
        out = np.zeros((self.fes.GetTrueVSize(), self.sdim))
        gf = mfem.ParGridFunction(self.fes)
        tmp = self.vector()
        for d in range(self.sdim):
            # host_readwrite, not a bare GetDataArray (see prolongation.py): under
            # a device-configured MFEM the first GetTrueDofs leaves the device copy
            # the valid one, and a bare write to the host copy is ignored.
            host_readwrite(gf)
            gf.GetDataArray()[:] = X[:, d]
            out[:, d] = self.from_gridfunction(gf, tmp).array
        return out


def _evaluate_at(fn, X, vdim):
    """``fn`` at the points ``X`` (``(n, sdim)``): vectorized when ``fn`` takes the
    whole array, else point by point.  Returns ``(n,)`` or ``(n, vdim)``."""
    n = X.shape[0]
    want = (n,) if vdim == 1 else (n, vdim)
    try:
        vals = np.asarray(fn(X), dtype=np.float64)
        if vals.shape == want:
            return vals
        if vdim > 1 and vals.shape == (vdim, n):
            return vals.T
    except Exception:                                     # noqa: BLE001
        pass
    vals = np.empty(want)
    for k in range(n):
        vals[k] = fn(X[k])
    return vals


class _ScalarPy(mfem.PyCoefficient):
    def __init__(self, fn):
        super(_ScalarPy, self).__init__()
        self.fn = fn

    def EvalValue(self, x):
        return float(self.fn(np.asarray(x)))


class _VectorPy(mfem.VectorPyCoefficient):
    def __init__(self, fn, vdim):
        super(_VectorPy, self).__init__(vdim)
        self.fn = fn

    def EvalValue(self, x):
        return np.asarray(self.fn(np.asarray(x)), dtype=np.float64)


def as_space(v):
    """Coerce ``v`` (FunctionSpace or ParFiniteElementSpace) to a FunctionSpace."""
    if isinstance(v, FunctionSpace):
        return v
    if isinstance(v, mfem.ParFiniteElementSpace):
        return FunctionSpace.wrap(v)
    raise TypeError("expected FunctionSpace or ParFiniteElementSpace, got %s"
                    % type(v).__name__)

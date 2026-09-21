# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""H(curl) and H(div) spaces: Nedelec and Raviart-Thomas.

Two things set these families apart from the Lagrange ones in the AD route, and
both are handled here.

**The basis is vector valued and mapped.**  A Nedelec basis function is pushed
forward covariantly (``J^{-T} w_ref``) and a Raviart-Thomas one contravariantly
(``J w_ref / det J``), so unlike Lagrange shape functions they cannot be tabulated
once for the whole element group, since the map depends on the element.  MFEM's
``CalcVShape(ElementTransformation&, ...)`` applies the right map, and it is used
here rather than a hand-rolled Piola transform, so the convention is MFEM's by
construction.  The derivative a field of this kind carries is not its gradient
(which is not in the space) but its **curl** for H(curl) and its **divergence**
for H(div), so the densities are written with ``u.curl`` / ``u.div`` instead of
``u.grad``:

.. code-block:: python

    def varf(u, m, p, x):                 # curl-curl with a varying coefficient
        return jnp.exp(m.val) * u.curl * p.curl + u.val @ p.val

**Element dofs may need a transformation, not just a sign.**  MFEM encodes edge
and face orientation partly as a sign on the dof index (which the batched gather
already decodes) and, for H(curl) of order above one on a geometry that is not a
tensor product (triangles, tetrahedra, wedges), partly as a small dense
``DofTransformation`` ``T`` per element.  The convention, from MFEM's own assembly
loop, is

* gather:  ``u_ref = T^{-1} (signs * u_global)``
* matrix:  ``A_global = T^T A_ref T``, then the signs
* vector:  ``b_global = T^T b_ref``, then the signs

and this module applies all three.  ``T`` is built only when MFEM reports it is not
the identity, which is never for H(div), for H(curl) on quadrilaterals and
hexahedra, or for lowest-order H(curl) anywhere.
"""

from typing import NamedTuple

import numpy as np

import mfem.par as mfem

from ..common.keepalive import KeepAlive
from .spaces import as_space
from .elementbatch import _element_dofs

__all__ = [
    "HCurlField",
    "HDivField",
    "VectorSpaceTables",
    "is_vector_space",
    "vector_kind",
]


class HCurlField(NamedTuple):
    """An H(curl) field at one quadrature point: a vector value and its curl.

    ``curl`` is a scalar in 2D and a ``(3,)`` vector in 3D, as in MFEM.
    """

    val: "np.ndarray"
    curl: "np.ndarray"


class HDivField(NamedTuple):
    """An H(div) field at one quadrature point: a vector value and its divergence."""

    val: "np.ndarray"
    div: "np.ndarray"


def vector_kind(space):
    """``"hcurl"``, ``"hdiv"`` or ``None`` for the element family of ``space``."""
    space = as_space(space)
    fe = space.fes.GetFE(0)
    if fe is None or fe.GetRangeType() != mfem.FiniteElement.VECTOR:
        return None
    d = fe.GetDerivType()
    if d == mfem.FiniteElement.CURL:
        return "hcurl"
    if d == mfem.FiniteElement.DIV:
        return "hdiv"
    raise NotImplementedError(
        "vector finite element %r has derivative type %d, which the element "
        "kernels do not cover (H(curl) and H(div) are supported)"
        % (space.fec.Name(), d))


def is_vector_space(space):
    """True for an H(curl) or H(div) space, whose basis functions are vectors."""
    return vector_kind(space) is not None


class VectorSpaceTables(KeepAlive):
    """Mapped vector basis tables and dof maps of one vector space on one group.

    Attributes
    ----------
    V : ndarray, shape (ne, nq, nd, sdim)
        Physical (Piola-mapped) vector shape functions.
    D : ndarray
        Physical curl, ``(ne, nq, nd)`` in 2D and ``(ne, nq, nd, 3)`` in 3D, or
        physical divergence, ``(ne, nq, nd)``.
    edofs, signs : ndarray, shape (ne, nd)
    T, Tinv : ndarray or None, shape (ne, nd, nd)
        MFEM's per-element ``DofTransformation``, present only when it is not the
        identity.
    """

    def __init__(self, space, group):
        space = as_space(space)
        self.space = space
        self.group = group
        self.kind = vector_kind(space)
        fes = space.fes
        if space.vdim != 1:
            raise NotImplementedError(
                "H(curl)/H(div) spaces carry their vector character in the basis, "
                "so vdim must be 1 (got %d)" % space.vdim)
        self.vdim = 1
        fe = fes.GetFE(int(group.elems[0]))
        self.nd = fe.GetDof()
        self.nd_total = self.nd
        self.order = fe.GetOrder()
        self.sdim = group.sdim
        self.dim = group.dim
        self._dev_maps = {}
        self._build(fes, group)

    # ------------------------------------------------------------------ build
    def _build(self, fes, group):
        ne, nq, nd = group.ne, group.nq, self.nd
        sdim, dim = self.sdim, self.dim
        mesh = group.mesh
        self.V = np.zeros((ne, nq, nd, sdim))
        curl3 = self.kind == "hcurl" and dim == 3
        self.D = (np.zeros((ne, nq, nd, 3)) if curl3
                  else np.zeros((ne, nq, nd)))

        Vsh = mfem.DenseMatrix(nd, sdim)
        Csh = mfem.DenseMatrix(nd, 3 if curl3 else 1)
        Dsh = mfem.Vector(nd)
        # the reference curl/divergence depends only on the quadrature point, so
        # it is tabulated once per point and mapped with numpy
        ref = np.zeros((nq, nd, 3 if curl3 else 1))
        fe0 = fes.GetFE(int(group.elems[0]))
        for q in range(nq):
            ip = group.ir.IntPoint(q)
            if self.kind == "hcurl":
                fe0.CalcCurlShape(ip, Csh)
                ref[q] = Csh.GetDataArray().reshape(
                    nd, 3 if curl3 else 1, order="F")
            else:
                fe0.CalcDivShape(ip, Dsh)
                ref[q, :, 0] = Dsh.GetDataArray()

        for a, e in enumerate(group.elems):
            fe = fes.GetFE(int(e))
            if fe.GetDof() != nd:
                raise NotImplementedError(
                    "variable-order spaces are not supported by the batched "
                    "element kernels (element %d has %d dofs, expected %d)"
                    % (int(e), fe.GetDof(), nd))
            T = mesh.GetElementTransformation(int(e))
            for q in range(nq):
                ip = group.ir.IntPoint(q)
                T.SetIntPoint(ip)
                fe.CalcVShape(T, Vsh)
                self.V[a, q] = Vsh.GetDataArray().reshape(nd, sdim, order="F")
                det = T.Weight()
                if curl3:
                    # physical curl = J curl_ref / det J, which is what MFEM's
                    # CurlCurlIntegrator forms as MultABt(curlshape, J) / Weight
                    J = T.Jacobian().GetDataArray().reshape(sdim, dim, order="F")
                    self.D[a, q] = (ref[q] @ J.T) / det
                else:
                    self.D[a, q] = ref[q, :, 0] / det


        self.edofs, self.signs = _element_dofs(fes, group.elems, 1, nd)
        self._build_transformation(fes, group)

    def _build_transformation(self, fes, group):
        """Materialize MFEM's ``DofTransformation`` per element, if there is one."""
        self.T = None
        self.Tinv = None
        get = getattr(fes, "GetElementDofTransformation", None)
        if get is None:
            return
        nd = self.nd
        eye = np.eye(nd)
        mats = None
        for a, e in enumerate(group.elems):
            dt = get(int(e))
            if dt is None or dt.IsIdentity():
                continue
            if mats is None:
                mats = np.repeat(eye[None, :, :], group.ne, axis=0)
            M = np.empty((nd, nd))
            for k in range(nd):
                v = mfem.Vector(eye[k].copy())
                dt.TransformPrimal(v)
                M[:, k] = v.GetDataArray()
            mats[a] = M
        if mats is not None:
            self.T = mats
            self.Tinv = np.linalg.inv(mats)

    # ------------------------------------------------------------- interface
    def gather(self, local_array):
        """Element dof values in the **reference** dof basis.

        The signs come first and the inverse transformation second, which is the
        order MFEM's ``GridFunction::GetElementDofValues`` uses.
        """
        vals = local_array[self.edofs] * self.signs
        if self.Tinv is None:
            return vals
        return np.einsum("eij,ej->ei", self.Tinv, vals)

    def gather_device(self, local_array):
        """The same gather, performed where the kernels run."""
        import jax.numpy as jnp

        from .kernel import _put

        from .kernel import device

        d = device()
        if d not in self._dev_maps:
            sg = None if self.signs.min() > 0 else _put(self.signs)
            ti = None if self.Tinv is None else _put(self.Tinv)
            self._dev_maps[d] = (_put(self.edofs), sg, ti)
        idx, sg, ti = self._dev_maps[d]
        vals = _put(local_array)[idx]
        if sg is not None:
            vals = vals * sg
        if ti is None:
            return vals
        return jnp.einsum("eij,ej->ei", ti, vals)

    def jax_tables(self):
        return (self.V, self.D)

    def jax_axes(self):
        """Per element: the Piola map depends on the element Jacobian."""
        return (0, 0)

    def evaluator(self):
        """A function ``(dofs, tables, Jinv) -> HCurlField`` or ``HDivField``."""
        if self.kind == "hcurl":
            curl3 = self.dim == 3

            def ev(dofs, tabs, Jinv):
                import jax.numpy as jnp

                V, D = tabs
                val = jnp.einsum("qid,i->qd", V, dofs)
                if curl3:
                    return HCurlField(val, jnp.einsum("qik,i->qk", D, dofs))
                return HCurlField(val, jnp.einsum("qi,i->q", D, dofs))
        else:
            def ev(dofs, tabs, Jinv):
                import jax.numpy as jnp

                V, D = tabs
                return HDivField(jnp.einsum("qid,i->qd", V, dofs),
                                 jnp.einsum("qi,i->q", D, dofs))
        return ev

    def _dev_T(self):
        from .kernel import _put, device

        if not hasattr(self, "_devT"):
            self._devT = {}
        d = device()
        if d not in self._devT:
            self._devT[d] = None if self.T is None else _put(self.T)
        return self._devT[d]

    def transform_dual_matrix(self, mats, trial_tables):
        """``A_global = T_test^T A_ref T_trial``, MFEM's ``TransformDual``.

        Applied on whichever side the element matrices already live, so a device
        result is not brought to the host just to be multiplied.
        """
        Tt = self.T
        Tr = getattr(trial_tables, "T", None)
        if Tt is None and Tr is None:
            return mats
        host = isinstance(mats, np.ndarray)
        if host:
            ein, out = np.einsum, mats
            Tt_, Tr_ = Tt, Tr
        else:
            import jax.numpy as jnp

            ein, out = jnp.einsum, mats
            Tt_ = None if Tt is None else self._dev_T()
            Tr_ = None if Tr is None else trial_tables._dev_T()
        if Tt_ is not None:
            out = ein("eki,ekl->eil", Tt_, out)
        if Tr_ is not None:
            out = ein("eil,elj->eij", out, Tr_)
        return out

    def transform_dual_vector(self, vecs):
        """``b_global = T^T b_ref``."""
        if self.T is None:
            return vecs
        if isinstance(vecs, np.ndarray):
            return np.einsum("eki,ek->ei", self.T, vecs)
        import jax.numpy as jnp

        return jnp.einsum("eki,ek->ei", self._dev_T(), vecs)

    def __repr__(self):
        return "VectorSpaceTables(%s, nd=%d, ne=%d, doftrans=%s)" % (
            self.kind, self.nd, self.group.ne, self.T is not None)

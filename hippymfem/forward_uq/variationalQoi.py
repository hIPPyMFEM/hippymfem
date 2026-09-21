# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Quantities of interest defined by a pointwise density.

The same machinery as the forward problem: the user writes
``q(u, m, x) -> scalar`` at one quadrature point, and the value, first and second
derivatives all come from differentiating it.  Use this when the QoI is an
integral such as :math:`\int_\Omega e^{m}|\nabla u|^2`, which no fixed matrix
represents.
"""

from ..common.keepalive import KeepAlive
from ..fem.assemble import assemble_scalar, assemble_vector, assemble_matrix
from ..fem.elementbatch import default_quadrature_degree, get_batches
from ..fem.spaces import as_space
from ..modeling.variables import PARAMETER, STATE
from .qoi import Qoi


class VariationalQoi(Qoi, KeepAlive):
    """QoI from a pointwise density ``q(u, m, x)``.

    Parameters
    ----------
    Vh : sequence of 3 spaces
        ``[STATE, PARAMETER, ADJOINT]``.
    qoi_varf : callable
        ``qoi_varf(u, m, x) -> scalar``, JAX-traceable, with
        :class:`~hippymfem.fem.kernel.Field` arguments.
    """

    #: slots of the QoI kernel
    _U, _M = 0, 1

    def __init__(self, Vh, qoi_varf, quadrature_degree=None):
        self.Vh = [as_space(v) for v in Vh]
        self.qoi_varf = qoi_varf
        self.comm = self.Vh[STATE].comm
        self.mesh = self.Vh[STATE].mesh
        if quadrature_degree is None:
            quadrature_degree = default_quadrature_degree(self.Vh)
        self.batches = get_batches(self.mesh, int(quadrature_degree))
        self.nelem = self.mesh.GetNE()

        from ..fem.kernel import QuadratureKernel

        # the QoI depends on (u, m) only; the adjoint never enters
        self.kernel = QuadratureKernel(
            lambda u, m, x: qoi_varf(u, m, x),
            [self.Vh[STATE], self.Vh[PARAMETER]], self.batches)
        self._x_lin = None

    def _locals(self, x):
        return [self.Vh[STATE].local_values(x[STATE]),
                self.Vh[PARAMETER].local_values(x[PARAMETER])]

    def eval(self, x):
        vals = self.kernel.element_values(self._locals(x))
        return assemble_scalar(self.comm, vals)

    def grad(self, i, x, g):
        slot = {STATE: self._U, PARAMETER: self._M}.get(i)
        if slot is None:
            g.zero()
            return g
        vecs = self.kernel.element_vectors(slot, self._locals(x))
        space = self.Vh[STATE] if slot == self._U else self.Vh[PARAMETER]
        v = assemble_vector(space, self.batches.groups, vecs, self.nelem)
        g.assign(v)
        return g

    def setLinearizationPoint(self, x):
        self._x_lin = [x[STATE].copy(), x[PARAMETER].copy(), None]
        loc = self._locals(self._x_lin)
        # the three blocks are slices of one element Hessian: one pass, not three
        pairs = ((self._U, self._U), (self._U, self._M), (self._M, self._M))
        mats = self.kernel.element_matrices_many(pairs, loc)
        space = {self._U: self.Vh[STATE], self._M: self.Vh[PARAMETER]}
        self._blocks = {
            (a, b): assemble_matrix(space[a], space[b], self.batches.groups,
                                    mats[(a, b)], self.nelem)
            for (a, b) in pairs}
        return self

    def apply_ij(self, i, j, dir, out):
        if self._x_lin is None:
            raise RuntimeError("setLinearizationPoint must be called first")
        table = {
            (STATE, STATE): ((self._U, self._U), False),
            (STATE, PARAMETER): ((self._U, self._M), False),
            (PARAMETER, STATE): ((self._U, self._M), True),
            (PARAMETER, PARAMETER): ((self._M, self._M), False),
        }
        if (i, j) not in table:
            out.zero()
            return out
        key, transpose = table[(i, j)]
        mat = self._blocks[key]
        if transpose:
            mat.MultTranspose(dir.hypre, out.hypre)
        else:
            mat.Mult(dir.hypre, out.hypre)
        return out

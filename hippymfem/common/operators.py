# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Operator and solver protocols, and adapters between them and MFEM.

Operators are duck-typed exactly as in hIPPYlib:

``mult(x, y)``
    ``y = A x``
``multTranspose(x, y)``
    ``y = A^T x`` (optional)
``init_vector(x, dim)``
    ``dim == 0`` -> a vector in the range of ``A``; ``dim == 1`` -> the domain.
    Accepts ``"noise"`` for priors, as hIPPYlib does.
``inner(x, y)``
    ``y^T A x`` (optional; supplied by :class:`Operator` when ``mult`` exists)

Solvers expose ``solve(x, b)`` meaning ``x = A^{-1} b``.

``dim`` follows hIPPYlib's convention throughout, including the quirk that the
same integer means "range" for operators and "solution" for solvers.
"""

import numpy as np

import mfem.par as mfem

from .keepalive import KeepAlive
from .naming import SnakeCamel, sync_spellings
from .parvector import ParVector, host_readwrite, host_sync


class Operator(SnakeCamel, KeepAlive):
    """Base class supplying ``inner`` and vector generation from ``mult``."""

    def mult(self, x, y):
        raise NotImplementedError

    def multTranspose(self, x, y):
        raise NotImplementedError(
            "%s does not implement multTranspose" % type(self).__name__
        )

    def init_vector(self, x, dim):
        raise NotImplementedError

    def generate_vector(self, dim=1):
        """Return a fresh vector in the range (``dim=0``) or domain (``dim=1``)."""
        x = _Slot()
        self.init_vector(x, dim)
        return x.value

    def inner(self, x, y):
        """``y^T A x``."""
        tmp = self.generate_vector(0)
        self.mult(x, tmp)
        return tmp.inner(y)

    def createVecLeft(self):
        return self.generate_vector(0)

    def createVecRight(self):
        return self.generate_vector(1)


class _Slot:
    """Sink used by ``generate_vector`` to capture what ``init_vector`` builds."""

    value = None

    def __setattr__(self, name, v):
        object.__setattr__(self, name, v)


def make_vector(obj, dim=0):
    """Create a vector compatible with ``obj.init_vector(x, dim)``.

    hIPPYlib's ``init_vector`` *fills* a target instead of returning one, which
    is awkward when a fresh vector is wanted.  This adapts the convention in one
    place, so callers never hand-roll a sink object (and never accidentally hand
    one that :func:`init_vector_like` does not recognise).
    """
    slot = _Slot()
    obj.init_vector(slot, dim)
    if slot.value is None:
        raise RuntimeError(
            "%s.init_vector did not produce a vector for dim=%r"
            % (type(obj).__name__, dim)
        )
    return slot.value


def init_vector_like(x, template):
    """hIPPYlib's ``init_vector`` idiom: resize ``x`` in place to match ``template``.

    ``x`` may be a ``ParVector`` (resized only if the layout differs), a ``_Slot``
    (used by :meth:`Operator.generate_vector`), or a list whose element 0 is set.
    """
    if isinstance(x, ParVector):
        if x.local_size == template.local_size:
            x.zero()
            return x
        raise ValueError(
            "init_vector: target has local size %d, need %d"
            % (x.local_size, template.local_size)
        )
    if isinstance(x, _Slot):
        x.value = template.duplicate()
        return x.value
    if isinstance(x, list):
        x[0] = template.duplicate()
        return x[0]
    raise TypeError("init_vector cannot fill a %s" % type(x).__name__)


def as_operator(obj, comm=None):
    """``obj`` as an :class:`Operator`: itself when it has ``mult``, else a
    :class:`MatrixOperator` around the (hypre) matrix.  The one place the library
    tells the two apart."""
    if hasattr(obj, "mult"):
        return obj
    return MatrixOperator(obj, comm)


class MatrixOperator(Operator):
    """Wrap an ``mfem.HypreParMatrix`` in the hIPPYlib operator protocol."""

    def __init__(self, A, comm=None, range_vec=None, domain_vec=None):
        self.A = A
        self.keep(A)
        self.comm = comm if comm is not None else _matrix_comm(A)
        self._range = range_vec
        self._domain = domain_vec

    # -- layout --------------------------------------------------------------
    def _range_template(self):
        if self._range is None:
            self._range = ParVector(self.comm, self.A.Height())
        return self._range

    def _domain_template(self):
        if self._domain is None:
            self._domain = ParVector(self.comm, self.A.Width())
        return self._domain

    def init_vector(self, x, dim):
        tpl = self._range_template() if dim == 0 else self._domain_template()
        return init_vector_like(x, tpl)

    # -- action --------------------------------------------------------------
    def mult(self, x, y):
        self.A.Mult(x.hypre, y.hypre)
        return y

    def multTranspose(self, x, y):
        self.A.MultTranspose(x.hypre, y.hypre)
        return y

    transpmult = multTranspose

    def getSize(self):
        return self.A.GetGlobalNumRows()

    def getComm(self):
        return self.comm

    def __repr__(self):
        return "MatrixOperator(%d x %d)" % (
            self.A.GetGlobalNumRows(),
            self.A.GetGlobalNumCols(),
        )


class TransposeOperator(Operator):
    """``A^T`` as an operator, without forming the transpose."""

    def __init__(self, A):
        self.A = A
        self.keep(A)

    def init_vector(self, x, dim):
        return self.A.init_vector(x, 1 - dim)

    def mult(self, x, y):
        return self.A.multTranspose(x, y)

    def multTranspose(self, x, y):
        return self.A.mult(x, y)


class IdentityOperator(Operator):
    def __init__(self, template):
        self.template = template

    def init_vector(self, x, dim):
        return init_vector_like(x, self.template)

    def mult(self, x, y):
        y.assign(x)
        return y

    multTranspose = mult

    def solve(self, x, b):
        x.assign(b)
        return x


class ScaledOperator(Operator):
    def __init__(self, A, alpha):
        self.A = A
        self.alpha = alpha
        self.keep(A)

    def init_vector(self, x, dim):
        return self.A.init_vector(x, dim)

    def mult(self, x, y):
        self.A.mult(x, y)
        y.scale(self.alpha)
        return y

    def multTranspose(self, x, y):
        self.A.multTranspose(x, y)
        y.scale(self.alpha)
        return y


class SumOperator(Operator):
    """``alpha*A + beta*B`` with matching layouts."""

    def __init__(self, A, B, alpha=1.0, beta=1.0):
        self.A, self.B = A, B
        self.alpha, self.beta = alpha, beta
        self.keep(A, B)
        self._tmp = None

    def init_vector(self, x, dim):
        return self.A.init_vector(x, dim)

    def mult(self, x, y):
        if self._tmp is None:
            self._tmp = self.A.generate_vector(0)
        self.A.mult(x, y)
        y.scale(self.alpha)
        self.B.mult(x, self._tmp)
        y.axpy(self.beta, self._tmp)
        return y

    def multTranspose(self, x, y):
        if self._tmp is None:
            self._tmp = self.A.generate_vector(1)
        self.A.multTranspose(x, y)
        y.scale(self.alpha)
        self.B.multTranspose(x, self._tmp)
        y.axpy(self.beta, self._tmp)
        return y


class ProductOperator(Operator):
    """``A @ B``: applies ``B`` then ``A``."""

    def __init__(self, A, B):
        self.A, self.B = A, B
        self.keep(A, B)
        self._mid = None

    def init_vector(self, x, dim):
        return self.A.init_vector(x, 0) if dim == 0 else self.B.init_vector(x, 1)

    def _middle(self):
        if self._mid is None:
            self._mid = self.B.generate_vector(0)
        return self._mid

    def mult(self, x, y):
        mid = self._middle()
        self.B.mult(x, mid)
        self.A.mult(mid, y)
        return y

    def multTranspose(self, x, y):
        mid = self._middle()
        self.A.multTranspose(x, mid)
        self.B.multTranspose(mid, y)
        return y


class DiagonalOperator(Operator):
    """Multiplication by a vector, entrywise."""

    def __init__(self, d):
        self.d = d

    def init_vector(self, x, dim):
        return init_vector_like(x, self.d)

    def mult(self, x, y):
        np.multiply(x.array, self.d.array, out=y.array)
        return y

    multTranspose = mult

    def solve(self, x, b):
        np.divide(b.array, self.d.array, out=x.array)
        return x

    def inner(self, x, y):
        return x.comm.allreduce(
            float(np.dot(x.array * self.d.array, y.array)), op=__import__("mpi4py").MPI.SUM
        )


class Solver2Operator(Operator):
    """Present a solver's ``solve`` as an operator's ``mult``."""

    def __init__(self, S, init_vector=None):
        self.S = S
        self.keep(S)
        self._init = init_vector

    def init_vector(self, x, dim):
        if self._init is not None:
            return self._init(x, dim)
        return self.S.init_vector(x, dim)

    def mult(self, x, y):
        self.S.solve(y, x)
        return y

    def multTranspose(self, x, y):
        tr = getattr(self.S, "solveTranspose", None)
        if tr is None:
            raise NotImplementedError("wrapped solver has no transpose solve")
        tr(y, x)
        return y


class Operator2Solver:
    """Present an operator's ``mult`` as a solver's ``solve``."""

    def __init__(self, op, init_vector=None):
        self.op = op
        self._init = init_vector

    def init_vector(self, x, dim):
        if self._init is not None:
            return self._init(x, dim)
        return self.op.init_vector(x, dim)

    def solve(self, x, b):
        self.op.mult(b, x)
        return 0

    def inner(self, x, y):
        tmp = self.op.generate_vector(0)
        self.op.mult(y, tmp)
        return tmp.inner(x)


class MFEMOperator(mfem.PyOperatorBase):
    """Adapt a hIPPyMFEM operator so MFEM's own solvers can drive it.

    MFEM hands us ``mfem.Vector`` objects; we copy through ``ParVector`` scratch
    space because MFEM does not guarantee its vectors carry a hypre partition.  The
    copies go through the host synchronization helpers: with MFEM on a device the
    vector it hands over may be current only on the card, and the result we write
    has to be marked as the host copy being the valid one.
    """

    def __init__(self, op, height, width=None):
        width = height if width is None else width
        super(MFEMOperator, self).__init__(height, width)
        self.op = op
        self._x = op.generate_vector(1)
        self._y = op.generate_vector(0)

    def Mult(self, x, y):
        host_sync(x)
        self._x.array[:] = x.GetDataArray()
        self.op.mult(self._x, self._y)
        host_readwrite(y)
        y.GetDataArray()[:] = self._y.array

    def MultTranspose(self, x, y):
        host_sync(x)
        self._y.array[:] = x.GetDataArray()
        self.op.multTranspose(self._y, self._x)
        host_readwrite(y)
        y.GetDataArray()[:] = self._x.array


def _matrix_comm(A):
    get = getattr(A, "GetComm", None)
    if get is not None:
        try:
            from mpi4py import MPI

            c = get()
            if isinstance(c, MPI.Comm):
                return c
        except Exception:
            pass
    from mpi4py import MPI

    return MPI.COMM_WORLD


sync_spellings(Operator)

# Derived from hIPPYlib / hIPPYlibx (https://hippylib.github.io):
# Copyright (c) 2016-2018, The University of Texas at Austin & University of
# California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin, University of
# California--Merced, Washington University in St. Louis.
# Copyright (c) 2025-, Georgia Institute of Technology.
# Modified in 2026 for MFEM, hypre and JAX by Peng Chen, Georgia Institute of
# Technology.  See the file COPYRIGHT for details.
#
# hIPPyMFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software
# Foundation) version 2.0 dated June 1991.  See the file LICENSE.
"""A collection of distributed vectors with BLAS-3 reductions.

hIPPYlib stores a ``MultiVector`` as a Python list of vectors, so ``V^T W``
costs ``nvec^2`` separate global reductions.  Here the columns are rows of a
single ``(nvec, n_local)`` C-contiguous array, so ``V^T W`` is one local GEMM
plus one ``Allreduce``.  Each column is still exposed as a
:class:`~hippymfem.common.parvector.ParVector` aliasing its row, so code written
against the hIPPYlib interface works unchanged.
"""

import numpy as np
from mpi4py import MPI

from ..common.naming import SnakeCamel, sync_spellings
from ..common.parvector import ParVector


class MultiVector(SnakeCamel):
    """``nvec`` distributed vectors sharing one layout.

    Parameters
    ----------
    v : ParVector or MultiVector, optional
        Layout template, or a MultiVector to deep-copy.
    nvec : int, optional
        Number of columns; required when ``v`` is a ParVector.
    """

    def __init__(self, v=None, nvec=None):
        self.comm = None
        self._data = None
        self._vecs = []
        self._layout = None
        if v is None:
            return
        if isinstance(v, MultiVector):
            self.comm = v.comm
            self._set_data(v._data.copy(), getattr(v, "_layout", None))
            return
        if nvec is None:
            raise ValueError("nvec is required when initializing from a vector")
        self.comm = v.comm
        self._set_data(np.zeros((int(nvec), v.local_size), dtype=np.float64),
                       v.layout)

    # ------------------------------------------------------------- internals
    def _set_data(self, data, layout=None):
        self._data = np.ascontiguousarray(data, dtype=np.float64)
        if layout is None and self._data.shape[0]:
            layout = ParVector.from_array(self.comm, self._data[0]).layout
        self._layout = layout
        self._vecs = [
            ParVector.from_array(self.comm, self._data[i], layout=layout)
            for i in range(self._data.shape[0])
        ]

    def _host_data(self):
        """The backing array with every column current on the host, for read and write.

        The columns are ``ParVector`` objects aliasing the rows of ``_data``, and hypre may
        have written one of them on a device (``B.mult`` into a column): the row is
        then stale until the vector syncs, and a reduction reading it directly would
        be silently wrong.  Every raw access goes through here; on a host build it is
        a flag test per column.
        """
        for v in self._vecs:
            v._host()
        return self._data

    @property
    def data(self):
        """The ``(nvec, n_local)`` backing array, current on the host."""
        return self._host_data()

    @classmethod
    def from_array(cls, comm, data):
        """Adopt a ``(nvec, n_local)`` array (no copy)."""
        mv = cls()
        mv.comm = comm
        mv._set_data(data)
        return mv

    # -------------------------------------------------------------- sequence
    def nvec(self):
        return 0 if self._data is None else int(self._data.shape[0])

    def __len__(self):
        return self.nvec()

    def __getitem__(self, i):
        return self._vecs[i]

    def __setitem__(self, i, v):
        self._vecs[i]._host()[:] = v.array

    def __iter__(self):
        return iter(self._vecs)

    def setSizeFromVector(self, v, nvec):
        """Reset to ``nvec`` zero columns with ``v``'s layout."""
        self.comm = v.comm
        self._set_data(np.zeros((int(nvec), v.local_size), dtype=np.float64),
                       v.layout)
        return self

    def copy(self):
        return MultiVector(self)

    # ----------------------------------------------------------- reductions
    def dot(self, v):
        """``self^T v``.

        Returns a 1-D array of length ``nvec`` for a vector argument and a
        **flattened** ``nvec x v.nvec`` array for a MultiVector argument, which
        is hIPPYlib's convention; :meth:`dot_mv` reshapes it.
        """
        if isinstance(v, MultiVector):
            loc = self._host_data() @ v._host_data().T
            out = np.zeros_like(loc)
            self.comm.Allreduce(np.ascontiguousarray(loc), out, op=MPI.SUM)
            return out.reshape(-1, order="C")
        loc = self._host_data() @ v.array
        out = np.zeros_like(loc)
        self.comm.Allreduce(np.ascontiguousarray(loc), out, op=MPI.SUM)
        return out

    def dot_v(self, v):
        return self.dot(v)

    def dot_mv(self, mv):
        return self.dot(mv).reshape((self.nvec(), mv.nvec()), order="C")

    def norm(self, norm_type="l2"):
        """Per-column norms as a 1-D array."""
        if norm_type in ("l2", 2):
            d = self._host_data()
            loc = np.einsum("ij,ij->i", d, d)
            out = np.zeros_like(loc)
            self.comm.Allreduce(np.ascontiguousarray(loc), out, op=MPI.SUM)
            return np.sqrt(np.maximum(out, 0.0))
        if norm_type in ("linf", "inf", np.inf):
            d = self._host_data()
            loc = np.abs(d).max(axis=1) if d.shape[1] else np.zeros(self.nvec())
            out = np.zeros_like(loc)
            self.comm.Allreduce(np.ascontiguousarray(loc), out, op=MPI.MAX)
            return out
        raise ValueError("unknown norm type %r" % (norm_type,))

    # -------------------------------------------------------------- algebra
    def reduce(self, v, alpha):
        """``v += sum_i alpha[i] * self[i]``."""
        a = np.asarray(alpha, dtype=np.float64).ravel()
        if a.size != self.nvec():
            raise ValueError("reduce: expected %d coefficients, got %d" % (self.nvec(), a.size))
        v.array[:] += a @ self._host_data()
        return v

    def axpy(self, a, y):
        """``self[i] += a[i]*y[i]`` for a MultiVector ``y``, or ``self[i] += a*y``."""
        if isinstance(y, MultiVector):
            coeff = np.asarray(a, dtype=np.float64).ravel()
            if y.nvec() != self.nvec():
                raise ValueError("axpy: MultiVector sizes differ")
            d, yd = self._host_data(), y._host_data()
            if coeff.size == 1:
                d += float(coeff[0]) * yd
            elif coeff.size == self.nvec():
                d += coeff[:, None] * yd
            else:
                raise ValueError("axpy: expected 1 or %d coefficients" % self.nvec())
        else:
            self._host_data()[...] += float(a) * y.array[None, :]
        return self

    def scale(self, k, a=None):
        """Scale every column by ``k[i]``, or column ``k`` by ``a``."""
        if a is None:
            coeff = np.asarray(k, dtype=np.float64).ravel()
            d = self._host_data()
            if coeff.size == 1:
                d *= float(coeff[0])
            elif coeff.size == self.nvec():
                d *= coeff[:, None]
            else:
                raise ValueError("scale: expected 1 or %d coefficients" % self.nvec())
        else:
            self[k]._host()[:] *= float(a)
        return self

    def zero(self):
        self._host_data()[:] = 0.0
        return self

    def swap(self, other):
        self._data, other._data = other._data, self._data
        self._vecs, other._vecs = other._vecs, self._vecs
        self._layout, other._layout = other._layout, self._layout
        return self

    # --------------------------------------------------- orthogonalization
    def orthogonalize(self):
        """Euclidean QR in place; ``self`` becomes ``Q``, returns ``R``."""
        return self._mgs()

    def Borthogonalize(self, B):
        """``B``-orthogonal QR in place.

        Returns ``(Bq, r)`` where ``Bq`` holds ``B @ Q`` and ``r`` is the
        triangular factor, with ``Q^T B Q = I``.  ``self`` becomes ``Q``.
        """
        return self._mgs(B)

    def _mgs(self, B=None):
        """Modified Gram-Schmidt with Rutishauser re-orthogonalization, in the
        ``B`` inner product when one is given and the Euclidean one otherwise.

        Returns ``(Bq, r)`` with ``B``, ``r`` alone without.
        """
        n = self.nvec()
        Bq = None if B is None else MultiVector(self[0], n)
        r = np.zeros((n, n))
        eps = np.finfo(np.float64).eps

        def image(k):
            """``B q_k`` (refreshed), or ``q_k`` itself."""
            if B is None:
                return self[k]
            B.mult(self[k], Bq[k])
            return Bq[k]

        for k in range(n):
            t = np.sqrt(max(image(k).inner(self[k]), 0.0))
            nach = 1
            while nach:
                for i in range(k):
                    s = (self[i] if B is None else Bq[i]).inner(self[k])
                    r[i, k] += s
                    self[k].axpy(-s, self[i])
                tt = np.sqrt(max(image(k).inner(self[k]), 0.0))
                if t * 10.0 * eps < tt < t / 10.0:
                    nach, t = 1, tt
                else:
                    nach = 0
                    if tt < 10.0 * eps * t:
                        tt = 0.0
            r[k, k] = tt
            inv = 1.0 / tt if abs(tt * eps) > 0.0 else 0.0
            self[k].scale(inv)
            if B is not None:
                Bq[k].scale(inv)
        return r if B is None else (Bq, r)

    def __repr__(self):
        return "MultiVector(nvec=%d, local=%d)" % (
            self.nvec(),
            0 if self._data is None else self._data.shape[1],
        )


# ------------------------------------------------------------------- helpers
def MatMvMult(A, x, y):
    """``y[i] = A x[i]`` for every column."""
    if x.nvec() != y.nvec():
        raise ValueError("MatMvMult: column counts differ")
    for i in range(x.nvec()):
        A.mult(x[i], y[i])
    return y


def MatMvTranspmult(A, x, y):
    """``y[i] = A^T x[i]`` for every column."""
    if x.nvec() != y.nvec():
        raise ValueError("MatMvTranspmult: column counts differ")
    for i in range(x.nvec()):
        A.multTranspose(x[i], y[i])
    return y


def MvDSmatMult(X, A, Y):
    """``Y = X A`` for a small dense ``A`` (``X.nvec x Y.nvec``)."""
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (X.nvec(), Y.nvec()):
        raise ValueError(
            "MvDSmatMult: A has shape %s, expected (%d, %d)"
            % (A.shape, X.nvec(), Y.nvec())
        )
    Y.data[:] = A.T @ X.data          # both properties sync their columns
    return Y


sync_spellings(MultiVector)

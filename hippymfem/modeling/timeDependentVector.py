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
"""A vector with one spatial field per discrete time level.

Time-dependent inverse problems treat the whole trajectory as the state, so the
state vector is a list of spatial vectors indexed by time.  Lookup is by time
value rather than index, with a tolerance, because the misfit and the time
stepper refer to times, not to positions in an array.
"""

import numpy as np
from mpi4py import MPI

from ..common.operators import as_operator
from ..common.keepalive import KeepAlive
from ..common.parvector import ParVector


class TimeDependentVector(KeepAlive):
    """A trajectory: one :class:`ParVector` per entry of ``times``.

    Parameters
    ----------
    times : array_like
        Discrete times, ascending.
    tol : float
        Tolerance for matching a time to a level.
    comm : mpi4py communicator
    """

    def __init__(self, times, tol=1e-10, comm=None):
        self.times = np.asarray(times, dtype=float)
        self.tol = float(tol)
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.nsteps = self.times.size
        self.data = [None] * self.nsteps

    # ------------------------------------------------------------------ setup
    def initialize(self, space_or_vector):
        """Allocate a zero field at every time level."""
        if hasattr(space_or_vector, "vector"):
            tpl = space_or_vector.vector()
        elif isinstance(space_or_vector, ParVector):
            tpl = space_or_vector
        else:
            tpl = ParVector.from_fes(space_or_vector)
        self.data = [tpl.duplicate() for _ in range(self.nsteps)]
        return self

    def copy(self, other=None):
        """A deep copy; with ``other`` given, copy that trajectory into ``self``."""
        if other is None:
            out = TimeDependentVector(self.times, self.tol, self.comm)
            out.data = [v.copy() if v is not None else None for v in self.data]
            return out
        for i, v in enumerate(other.data):
            if v is None:
                continue
            if self.data[i] is None:
                self.data[i] = v.copy()
            else:
                self.data[i].assign(v)
        return self

    # ----------------------------------------------------------------- access
    def index(self, t):
        """Index of the level nearest ``t``; raises if none is within ``tol``."""
        k = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[k] - t) > self.tol:
            raise KeyError("no time level within %g of t = %g" % (self.tol, t))
        return k

    def view(self, t):
        """The spatial vector at time ``t`` (a reference, not a copy)."""
        return self.data[self.index(t)]

    def store(self, u, t):
        """Copy the spatial vector ``u`` into the level at time ``t``."""
        k = self.index(t)
        if self.data[k] is None:
            self.data[k] = u.copy()
        else:
            self.data[k].assign(u)
        return self

    def retrieve(self, u, t):
        """Copy the level at time ``t`` into ``u``."""
        u.assign(self.data[self.index(t)])
        return u

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, v):
        self.data[i] = v

    def __len__(self):
        return self.nsteps

    def __iter__(self):
        return iter(self.data)

    # ------------------------------------------------------------------- math
    def zero(self):
        for v in self.data:
            if v is not None:
                v.zero()
        return self

    def set(self, alpha):
        for v in self.data:
            if v is not None:
                v.set(alpha)
        return self

    def scale(self, alpha):
        for v in self.data:
            if v is not None:
                v.scale(alpha)
        return self

    def axpy(self, a, other):
        for v, w in zip(self.data, other.data):
            if v is not None and w is not None:
                v.axpy(a, w)
        return self

    def assign(self, other):
        return self.copy(other)

    def duplicate(self):
        out = TimeDependentVector(self.times, self.tol, self.comm)
        out.data = [v.duplicate() if v is not None else None for v in self.data]
        return out

    def inner(self, other):
        """Sum over time levels of the spatial inner products."""
        return float(sum(v.inner(w) for v, w in zip(self.data, other.data)
                         if v is not None and w is not None))

    dot = inner

    def element_wise_inner(self, other):
        """Per-time-level inner products, as an array."""
        return np.array([v.inner(w) if v is not None and w is not None else 0.0
                         for v, w in zip(self.data, other.data)])

    def norm(self, time_norm="linf", space_norm="l2"):
        """Norm over space at each level, combined over time.

        ``time_norm`` is ``"linf"``, ``"l2"`` or ``"l1"``.
        """
        vals = np.array([v.norm(space_norm) if v is not None else 0.0
                         for v in self.data])
        if time_norm in ("linf", "inf"):
            return float(vals.max()) if vals.size else 0.0
        if time_norm == "l2":
            return float(np.sqrt(np.sum(vals ** 2)))
        if time_norm == "l1":
            return float(np.sum(vals))
        raise ValueError("unknown time norm %r" % (time_norm,))

    def matmul(self, mat, out):
        """Apply a spatial operator or matrix at every level."""
        for v, w in zip(self.data, out.data):
            if v is None or w is None:
                continue
            as_operator(mat, self.comm).mult(v, w)
        return out

    # ---------------------------------------------------------------- sugar
    def __iadd__(self, other):
        return self.axpy(1.0, other)

    def __isub__(self, other):
        return self.axpy(-1.0, other)

    def __imul__(self, a):
        return self.scale(a)

    def __add__(self, other):
        return self.copy().axpy(1.0, other)

    def __sub__(self, other):
        return self.copy().axpy(-1.0, other)

    def __mul__(self, a):
        return self.copy().scale(a)

    __rmul__ = __mul__

    def __neg__(self):
        return self.copy().scale(-1.0)

    def __repr__(self):
        return "TimeDependentVector(%d levels, t in [%g, %g])" % (
            self.nsteps, self.times[0], self.times[-1])

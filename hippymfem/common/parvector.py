# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Distributed vectors.

hIPPyMFEM represents every distributed vector as a :class:`ParVector`: a
contiguous ``numpy`` array of the locally owned entries, wrapped by an
``mfem.HypreParVector`` that aliases the same memory.  That gives three things
at once:

* ``v.array`` is a plain numpy view, so kernels and MPI calls are direct;
* ``v.hypre`` is accepted by every hypre solver and ``HypreParMatrix``;
* because ``HypreParVector`` derives from ``mfem::Vector``, the same object is
  also accepted by MFEM's own ``CGSolver``/``GMRESSolver`` and by ``Operator``.

Vectors on a finite element space live on **true dofs**.  MFEM numbers true
dofs contiguously by rank, so the partition is the exclusive prefix sum of
``GetTrueVSize()``; this is computed by one ``allgather`` and matches the row
partition that ``ParBilinearForm.ParallelAssemble()`` produces.  That
compatibility is asserted in the test suite rather than assumed.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from .keepalive import KeepAlive

_HYPRE_INT = np.int32 if mfem.sizeof_HYPRE_Int() == 4 else np.int64


def partition(comm, local_size):
    """Return ``(offsets, global_size)`` for a contiguous by-rank partition.

    ``offsets`` has ``comm.size + 1`` entries; ``offsets[r]`` is the global
    index of rank ``r``'s first entry.
    """
    counts = comm.allgather(int(local_size))
    offs = np.zeros(comm.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offs[1:])
    return offs, int(offs[-1])


class Layout:
    """The global index layout of a distributed vector.

    Building one costs a single ``allgather``, so it is built **once** and shared: a
    vector derived from an existing one (``duplicate``, ``copy``, a ``MultiVector``
    column, anything from ``FunctionSpace.vector``) reuses its parent's layout and
    does not communicate, which keeps allocation cheap inside sampling and Krylov
    loops.

    It is deliberately **not** memoized on ``(comm, local_size)``: local sizes differ
    between ranks, so one rank could hit the cache and skip the ``allgather`` while
    another misses and performs it.  The ranks would then desynchronize, and a
    *later*, unrelated collective would receive the wrong message.  Constructing a
    layout always communicates; only reuse is free.
    """

    __slots__ = ("comm", "local_size", "offsets", "global_size", "col_starts")

    def __init__(self, comm, local_size):
        self.comm = comm
        self.local_size = int(local_size)
        self.offsets, self.global_size = partition(comm, self.local_size)
        rank = comm.rank
        self.col_starts = np.array(
            [self.offsets[rank], self.offsets[rank + 1], self.global_size],
            dtype=_HYPRE_INT,
        )


class ParVector(KeepAlive):
    """A distributed vector of real numbers.

    Parameters
    ----------
    comm : mpi4py communicator
    local_size : int
        Number of locally owned entries.  Ignored if ``array`` is given.
    array : numpy.ndarray, optional
        Existing contiguous float64 buffer to adopt (not copied).
    """

    def __init__(self, comm, local_size=None, array=None, layout=None):
        self.comm = comm
        if array is None:
            if local_size is None:
                raise ValueError("ParVector needs local_size or array")
            array = np.zeros(int(local_size), dtype=np.float64)
        else:
            array = np.ascontiguousarray(array, dtype=np.float64).reshape(-1)
        self._array = array
        if layout is None:
            layout = Layout(comm, array.shape[0])        # one allgather
        elif layout.local_size != array.shape[0]:
            raise ValueError("layout has local size %d, array has %d"
                             % (layout.local_size, array.shape[0]))
        self._layout = layout
        self._offsets = layout.offsets
        self._global_size = layout.global_size
        self._col_starts = layout.col_starts
        # hypre does not own this array; both it and the partitioning must be
        # retained for the lifetime of the HypreParVector.  With MFEM configured on
        # a device the buffer is registered with its memory manager and mirrored
        # on the card, and every access from Python goes through ``_host`` so the
        # two copies stay in step (see ``host_sync`` / ``host_readwrite`` below).
        self._hv = mfem.HypreParVector(
            comm, self._global_size, [self._array, self._col_starts]
        )
        self.keep(self._array, layout, self._hv)

    def _host(self):
        """The locally owned entries as numpy, valid on the host for read *and* write.

        Every vector operation goes through this rather than touching ``_array``
        directly, so the synchronization happens once per operation.  On a host-only
        build it is a single flag test.
        """
        host_readwrite(self._hv)
        return self._array

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_fes(cls, fes, comm=None, layout=None):
        """A true-dof vector on the parallel finite element space ``fes``."""
        if comm is None:
            comm = _fes_comm(fes)
        return cls(comm, fes.GetTrueVSize(), layout=layout)

    @classmethod
    def from_array(cls, comm, array, layout=None):
        """Adopt ``array`` (no copy) as the local part of a new vector."""
        return cls(comm, array=array, layout=layout)

    @property
    def layout(self):
        """The shared :class:`Layout` of this vector."""
        return self._layout

    def duplicate(self):
        """A new zero vector with the same layout; communicates nothing."""
        return ParVector(self.comm, self._layout.local_size, layout=self._layout)

    # hippylib / petsc4py aliases used by ported algorithms
    createVecLeft = duplicate
    createVecRight = duplicate

    def copy(self):
        """A new vector with the same layout and the same values."""
        out = self.duplicate()
        out._host()[:] = self._host()
        return out

    # ------------------------------------------------------------------ views
    @property
    def array(self):
        """numpy view of the locally owned entries.

        With MFEM configured on a GPU the current values may be only in device
        memory, so this makes the host copy current and valid for read *and* write
        (callers do both through this view) before handing it over.  Without that the
        symptom is not an error but stale values, such as a matvec that returns
        zeros.  With MFEM on the host the check is a single flag test.
        """
        return self._host()

    @property
    def hypre(self):
        """The ``mfem.HypreParVector`` aliasing this vector's memory."""
        return self._hv

    @property
    def offsets(self):
        """Global offsets array, ``comm.size + 1`` entries."""
        return self._offsets

    @property
    def owner_range(self):
        """``(first, last)`` global indices owned by this rank."""
        r = self.comm.rank
        return int(self._offsets[r]), int(self._offsets[r + 1])

    @property
    def local_size(self):
        # from the layout, not the buffer: asking the buffer would synchronize it
        return int(self._layout.local_size)

    @property
    def global_size(self):
        return self._global_size

    def getSize(self):
        return self._global_size

    def getLocalSize(self):
        return self.local_size

    def getComm(self):
        return self.comm

    def __len__(self):
        return self.local_size

    # ------------------------------------------------------------------- math
    # Each of these takes the host view once, into a local: on a device build every
    # access costs a synchronization check, and ``self.array *= a`` would try to
    # assign to the read-only property.
    def zero(self):
        self._host()[:] = 0.0
        return self

    def set(self, alpha):
        """Set every entry to the scalar ``alpha``."""
        self._host()[:] = alpha
        return self

    def scale(self, alpha):
        a = self._host()
        a *= alpha
        return self

    def axpy(self, alpha, y):
        """``self += alpha * y``."""
        self._check(y)
        a, b = self._host(), y._host()
        if alpha == 1.0:
            a += b
        elif alpha == -1.0:
            a -= b
        else:
            a += alpha * b
        return self

    def aypx(self, alpha, y):
        """``self = alpha * self + y``."""
        self._check(y)
        a, b = self._host(), y._host()
        a *= alpha
        a += b
        return self

    def axpby(self, alpha, beta, y):
        """``self = alpha * y + beta * self``."""
        self._check(y)
        a, b = self._host(), y._host()
        a *= beta
        a += alpha * b
        return self

    def assign(self, other):
        """Copy values from ``other`` (a ParVector or a scalar)."""
        if isinstance(other, ParVector):
            self._check(other)
            self._host()[:] = other._host()
        else:
            self._host()[:] = other
        return self

    def pointwise_mult(self, y):
        self._check(y)
        a = self._host()
        a *= y._host()
        return self

    def inner(self, y):
        """Global Euclidean inner product.

        Reads through ``_host`` like every other method: reading ``_array`` directly
        after a hypre matvec on a device would return a stale value.
        """
        self._check(y)
        loc = float(np.dot(self._host(), y._host()))
        return self.comm.allreduce(loc, op=MPI.SUM)

    dot = inner

    def norm(self, norm_type="l2"):
        """Global norm; ``norm_type`` in ``{'l2', 'linf', 'l1'}``."""
        if norm_type in ("l2", 2, "NORM_2"):
            return float(np.sqrt(max(self.inner(self), 0.0)))
        if norm_type in ("linf", "inf", np.inf, "NORM_INFINITY"):
            loc = float(np.abs(self._host()).max()) if self.local_size else 0.0
            return self.comm.allreduce(loc, op=MPI.MAX)
        if norm_type in ("l1", 1, "NORM_1"):
            loc = float(np.abs(self._host()).sum())
            return self.comm.allreduce(loc, op=MPI.SUM)
        raise ValueError("unknown norm type %r" % (norm_type,))

    def sum(self):
        return self.comm.allreduce(float(self._host().sum()), op=MPI.SUM)

    def max(self):
        loc = float(self._host().max()) if self.local_size else -np.inf
        return self.comm.allreduce(loc, op=MPI.MAX)

    def min(self):
        loc = float(self._host().min()) if self.local_size else np.inf
        return self.comm.allreduce(loc, op=MPI.MIN)

    # ------------------------------------------------------------- operators
    def __iadd__(self, other):
        if isinstance(other, ParVector):
            return self.axpy(1.0, other)
        a = self._host()
        a += other
        return self

    def __isub__(self, other):
        if isinstance(other, ParVector):
            return self.axpy(-1.0, other)
        a = self._host()
        a -= other
        return self

    def __imul__(self, alpha):
        return self.scale(alpha)

    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __mul__(self, alpha):
        return self.copy().scale(alpha)

    __rmul__ = __mul__

    def __neg__(self):
        return self.copy().scale(-1.0)

    def _check(self, y):
        if not isinstance(y, ParVector):
            raise TypeError("expected ParVector, got %s" % type(y).__name__)
        if y.local_size != self.local_size:
            raise ValueError(
                "incompatible local sizes %d vs %d" % (self.local_size, y.local_size)
            )

    def __repr__(self):
        return "ParVector(local=%d, global=%d)" % (self.local_size, self.global_size)

    # ------------------------------------------------------------------- misc
    def gather_to_zero(self):
        """Gather the whole vector onto rank 0 as a numpy array (``None`` elsewhere)."""
        parts = self.comm.gather(self._host(), root=0)
        if self.comm.rank == 0:
            return np.concatenate(parts)
        return None

    def allgather(self, out=None):
        """The whole vector as a numpy array on **every** rank.

        Uses ``Allgatherv`` with the layout's own counts, so it is one collective
        with no Python-level concatenation.  This sits inside the per-solve path
        of :class:`~hippymfem.algorithms.directSolvers.ReplicatedLUSolver`.
        """
        n = self.global_size
        if out is None:
            out = np.empty(n, dtype=np.float64)
        off = self.layout.offsets
        counts = np.diff(off).astype("i")
        self.comm.Allgatherv(
            [np.ascontiguousarray(self._host()), MPI.DOUBLE],
            [out, counts, off[:-1].astype("i"), MPI.DOUBLE])
        return out

    def scatter_from_zero(self, full):
        """Inverse of :meth:`gather_to_zero`; ``full`` is read on rank 0 only."""
        full = self.comm.bcast(full if self.comm.rank == 0 else None, root=0)
        lo, hi = self.owner_range
        self._host()[:] = np.asarray(full, dtype=np.float64)[lo:hi]
        return self


def device_active():
    """Whether MFEM is *currently* configured on a GPU, not merely built for one.

    A PyMFEM built for a GPU keeps everything on the host until a GPU
    ``mfem.Device`` is constructed, and until then none of the synchronization below
    is needed.  ``Device::IsEnabled()`` answers this cheaply enough for the vector
    access path.
    """
    try:
        return bool(mfem.Device.IsEnabled())
    except Exception:
        return False


def host_sync(obj):
    """Make an MFEM object's data valid on the host before it is read as numpy.

    When MFEM is configured on a device, an ``mfem::Vector`` may hold its current
    values only in device memory, and ``GetDataArray()`` silently hands back the
    *host* pointer regardless.  Reading it gives stale or zero data rather than an
    error, and the failure surfaces far from the cause (a singular matrix, say).

    ``HostRead`` performs the copy and marks the host copy valid; a ``SparseMatrix``
    keeps its three arrays separately and needs all three.  On a host-only build this
    returns immediately.
    """
    if obj is None or not device_active():
        return obj
    return _sync(obj, ("HostRead",),
                 ("HostReadI", "HostReadJ", "HostReadData"))


def host_readwrite(obj):
    """Make an MFEM object's data valid on the host **and** invalidate the device copy.

    Use this when the host side is about to *write*: marking the data
    host-read-only would let a later device operation read the pre-write values.
    :class:`ParVector` goes through this, because its numpy array and its
    ``HypreParVector`` alias the same memory and either side may write.
    """
    if obj is None or not device_active():
        return obj
    return _sync(obj, ("HostReadWrite",),
                 ("HostReadWriteI", "HostReadWriteJ", "HostReadWriteData"))


def _sync(obj, single, triple):
    for name in single:
        fn = getattr(obj, name, None)
        if fn is not None:
            _call_sync(fn, obj, name)
            return obj
    for name in triple:
        fn = getattr(obj, name, None)
        if fn is not None:
            _call_sync(fn, obj, name)
    return obj


def _call_sync(fn, obj, name):
    """Run one synchronization call; if it fails, warn and re-raise.

    Swallowing the exception would turn a failed copy into silently stale data, the
    failure mode this module exists to prevent.
    """
    try:
        fn()
    except Exception as exc:
        import warnings

        warnings.warn("%s.%s() failed during host/device synchronization: %s"
                      % (type(obj).__name__, name, exc), RuntimeWarning, stacklevel=4)
        raise


def to_numpy(v, copy=True):
    """Local data of an MFEM ``Vector`` as a numpy array.

    ``mfem.Vector.GetDataArray()`` returns a view that does **not** hold a
    Python reference to the vector that owns the memory.  So

    .. code-block:: python

        form.ParallelAssemble().GetDataArray().sum()    # WRONG

    reads freed memory: nothing keeps the temporary ``HypreParVector`` alive
    past ``GetDataArray()``.  The symptom is a plausible but slightly wrong
    number that changes from run to run, not a crash.  Always go through this
    helper, which copies while the owner is still referenced by its argument.
    """
    host_sync(v)
    a = v.GetDataArray()
    return np.array(a, dtype=np.float64, copy=True) if copy else a


def _fes_comm(fes):
    """MPI communicator of a ``ParFiniteElementSpace`` (falls back to COMM_WORLD)."""
    mesh = fes.GetParMesh() if hasattr(fes, "GetParMesh") else None
    for obj in (fes, mesh):
        if obj is None:
            continue
        get = getattr(obj, "GetComm", None)
        if get is None:
            continue
        try:
            c = get()
        except Exception:
            continue
        if isinstance(c, MPI.Comm):
            return c
    return MPI.COMM_WORLD


def as_parvector(x, comm=None):
    """Coerce ``x`` (ParVector, numpy array, or mfem.Vector) to a ParVector."""
    if isinstance(x, ParVector):
        return x
    if comm is None:
        comm = MPI.COMM_WORLD
    if isinstance(x, np.ndarray):
        return ParVector.from_array(comm, x.copy())
    if isinstance(x, mfem.Vector):
        return ParVector.from_array(comm, np.array(x.GetDataArray()))
    raise TypeError("cannot interpret %s as a ParVector" % type(x).__name__)

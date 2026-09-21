# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Integrators that serve precomputed element arrays to MFEM's assembly loop.

This is the ``"integrator"`` assembly route
(:func:`~hippymfem.fem.assemble.set_assembly_backend`), the reference for the
default direct scatter of :mod:`.csrassemble`.  The element matrices are produced
in one batched JAX call; MFEM then walks the elements, asks for them one at a time,
and does the rest itself: it combines local dofs into true dofs across ranks,
handles non-conforming interfaces and eliminates essential boundary conditions.
The price is one Python callback per element.

Lookup is by ``Trans.ElementNo`` rather than call order, so nothing depends on
the order in which MFEM visits elements.
"""

import numpy as np

import mfem.par as mfem


class ElementLookup:
    """Maps a mesh element index to its precomputed array.

    Handles mixed-geometry meshes, where different element groups have
    differently shaped arrays.
    """

    def __init__(self, groups, arrays, nelem):
        # MFEM reads these one element at a time, on the host, so arrays the kernels
        # left on a device are copied over in bulk here: indexing them per element
        # would cost one transfer each, and ``elmat.Assign`` needs a numpy array (it
        # raises inside the SWIG director, where the error message is lost).
        self.arrays = [a if isinstance(a, np.ndarray) else np.asarray(a)
                       for a in arrays]
        if len(groups) == 1 and groups[0].ne == nelem:
            g = groups[0]
            if np.array_equal(g.elems, np.arange(nelem, dtype=np.int64)):
                self._gid = None           # fast path: direct indexing
                return
        self._gid = np.full(nelem, -1, dtype=np.int32)
        self._pos = np.zeros(nelem, dtype=np.int32)
        for gi, g in enumerate(groups):
            self._gid[g.elems] = gi
            self._pos[g.elems] = np.arange(g.ne, dtype=np.int32)

    def __call__(self, e):
        if self._gid is None:
            return self.arrays[0][e]
        return self.arrays[self._gid[e]][self._pos[e]]


class CachedMatrixIntegrator(mfem.PyBilinearFormIntegrator):
    """Serves cached element matrices.

    ``AssembleElementMatrix`` is used by ``BilinearForm`` (square blocks) and
    ``AssembleElementMatrix2`` by ``MixedBilinearForm`` (rectangular blocks);
    both read from the same lookup.
    """

    def __init__(self, lookup):
        super(CachedMatrixIntegrator, self).__init__()
        self.lookup = lookup
        self.ncalls = 0

    def AssembleElementMatrix(self, el, Trans, elmat):
        E = self.lookup(Trans.ElementNo)
        elmat.SetSize(E.shape[0], E.shape[1])
        elmat.Assign(E)
        self.ncalls += 1

    def AssembleElementMatrix2(self, trial_fe, test_fe, Trans, elmat):
        E = self.lookup(Trans.ElementNo)
        elmat.SetSize(E.shape[0], E.shape[1])
        elmat.Assign(E)
        self.ncalls += 1


class CachedVectorIntegrator(mfem.PyLinearFormIntegrator):
    """Serves cached element vectors to a ``LinearForm``."""

    def __init__(self, lookup):
        super(CachedVectorIntegrator, self).__init__()
        self.lookup = lookup
        self.ncalls = 0

    def AssembleRHSElementVect(self, el, Trans, elvect):
        v = self.lookup(Trans.ElementNo)
        elvect.SetSize(v.shape[0])
        elvect.Assign(v)
        self.ncalls += 1

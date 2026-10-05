# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""MFEM's own assembly of the element arrays: the reference of the test suite.

The library scatters its element arrays straight into the parallel matrix
(:mod:`hippymfem.fem.csrassemble`).  Here the same arrays are handed to MFEM one
element at a time through a ``PyBilinearFormIntegrator``, and MFEM does the rest
itself: it combines local dofs into true dofs across ranks, handles non-conforming
interfaces and eliminates essential boundary conditions.  The tests hold the two to
exact equality on every block, and ``benchmarks/bench_assembly.py`` times one against
the other; the price of this route is one Python callback per element.

Lookup is by ``Trans.ElementNo`` rather than call order, so nothing depends on
the order in which MFEM visits elements.
"""

import numpy as np

import mfem.par as mfem

from ..common.linalg import own, set_diagonal_entries
from ..common.parvector import to_numpy
from ..fem.assemble import _policy, empty_ess
from ..fem.spaces import as_space


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


def _materialize(chunks):
    """Glue ``(group, start, stop, array)`` chunks back into one array per group."""
    parts = {}
    for g, _a, _b, arr in chunks:
        parts.setdefault(g, []).append(np.asarray(arr))
    for g in sorted(parts):
        yield np.concatenate(parts[g], axis=0) if len(parts[g]) > 1 else parts[g][0]


def assemble_matrix(test_space, trial_space, groups, element_matrices, nelem,
                    test_ess=None, trial_ess=None, diag_policy="one"):
    """:func:`hippymfem.fem.assemble.assemble_matrix`, by MFEM's element loop.

    A ``ParBilinearForm`` when the two spaces are the same object, otherwise a
    ``ParMixedBilinearForm``; ``nelem`` is the number of mesh elements, for the
    element lookup.
    """
    test_space = as_space(test_space)
    trial_space = as_space(trial_space)
    if callable(element_matrices):
        # MFEM's callback wants the arrays, so chunks are realized here
        element_matrices = list(_materialize(element_matrices()))
    lookup = ElementLookup(groups, element_matrices, nelem)
    integ = CachedMatrixIntegrator(lookup)

    same = test_space.fes is trial_space.fes
    A = mfem.HypreParMatrix()
    if same:
        form = mfem.ParBilinearForm(test_space.fes)
        form.SetDiagonalPolicy(_policy("one" if diag_policy == "zero" else diag_policy))
        form.AddDomainIntegrator(integ)
        form.Assemble()
        form.Finalize()
        ess = test_ess if test_ess is not None else empty_ess()
        form.FormSystemMatrix(ess, A)
        if diag_policy == "zero" and ess.Size():
            # MFEM's DiagonalPolicy reaches only the serial SparseMatrix path;
            # hypre's EliminateRowsCols always writes 1.0, so fix it up here.
            set_diagonal_entries(A, np.asarray(ess.ToList(), dtype=np.int64), 0.0)
        return own(A, form, integ, lookup)
    form = mfem.ParMixedBilinearForm(trial_space.fes, test_space.fes)
    form.AddDomainIntegrator(integ)
    form.Assemble()
    form.Finalize()
    form.FormRectangularSystemMatrix(
        trial_ess if trial_ess is not None else empty_ess(),
        test_ess if test_ess is not None else empty_ess(),
        A,
    )
    return own(A, form, integ, lookup)


def assemble_vector(space, groups, element_vectors, nelem, ess=None, out=None):
    """:func:`hippymfem.fem.assemble.assemble_vector`, by MFEM's element loop."""
    space = as_space(space)
    lookup = ElementLookup(groups, element_vectors, nelem)
    integ = CachedVectorIntegrator(lookup)
    form = mfem.ParLinearForm(space.fes)
    form.AddDomainIntegrator(integ)
    form.Assemble()
    hv = form.ParallelAssemble()
    if out is None:
        out = space.vector()
    out.array[:] = to_numpy(hv, copy=False)
    del hv
    if ess is not None and len(ess):
        out.array[np.asarray(ess, dtype=np.int64)] = 0.0
    return out

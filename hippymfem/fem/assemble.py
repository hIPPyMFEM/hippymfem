# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Assembly of AD-generated element arrays into parallel matrices and vectors.

By default the element arrays are scattered directly (:mod:`.csrassemble`).  The
``"integrator"`` route instead builds a short-lived MFEM form, attaches a
:class:`~.integrators.CachedMatrixIntegrator` holding the already-computed
element arrays, and lets MFEM assemble.  Either way essential boundary conditions
are applied block by block as the table in :mod:`.bcs` states, so that the
matrices behave exactly as hIPPYlib's do.
"""

import os

import numpy as np

import mfem.par as mfem

from ..common.linalg import own, set_diagonal_entries  # noqa: F401  (own: re-exported)
from ..common.parvector import host_sync, to_numpy
from .integrators import CachedMatrixIntegrator, CachedVectorIntegrator, ElementLookup
from .spaces import as_space
from .csrassemble import assemble_matrix_csr, assemble_vector_csr

_EMPTY = None

#: Which assembly route :func:`assemble_matrix` and :func:`assemble_vector` take.
#: ``"csr"`` scatters the element arrays directly (see :mod:`.csrassemble`);
#: ``"integrator"`` hands them to MFEM one element at a time.  The two agree to
#: round-off, which the test suite checks on every block; the integrator route is
#: the reference, and a fallback for element families the direct scatter does not
#: cover.
_BACKEND = os.environ.get("HIPPYMFEM_ASSEMBLY", "csr")


def set_assembly_backend(name):
    """Select ``"csr"`` (default) or ``"integrator"`` assembly; returns the old one."""
    global _BACKEND
    if name not in ("csr", "integrator"):
        raise ValueError("assembly backend must be 'csr' or 'integrator', got %r"
                         % (name,))
    old, _BACKEND = _BACKEND, name
    return old


def assembly_backend():
    """The assembly route currently in use."""
    return _BACKEND


def empty_ess():
    """A shared empty essential-dof list."""
    global _EMPTY
    if _EMPTY is None:
        _EMPTY = mfem.intArray()
    return _EMPTY


def _materialize(chunks):
    """Glue ``(group, start, stop, array)`` chunks back into one array per group."""
    import numpy as _np

    parts = {}
    for g, _a, _b, arr in chunks:
        parts.setdefault(g, []).append(_np.asarray(arr))
    for g in sorted(parts):
        yield _np.concatenate(parts[g], axis=0) if len(parts[g]) > 1 else parts[g][0]


def assemble_matrix(test_space, trial_space, groups, element_matrices, nelem,
                    test_ess=None, trial_ess=None, diag_policy="one"):
    """Assemble a block from per-group element matrices.

    Parameters
    ----------
    test_space, trial_space : FunctionSpace
        Row and column spaces.  When they are the same object a
        ``ParBilinearForm`` is used, otherwise a ``ParMixedBilinearForm``.
    groups : sequence of ElementGroup
    element_matrices : sequence of ndarray, or callable
        One ``(ne, nd_test, nd_trial)`` array per group.  A callable returning an
        iterator of ``(group, start, stop, array)`` chunks lets the direct-CSR
        route scatter each chunk as it is produced, so the full array is never
        formed; the callback route materializes them.
    nelem : int
        Total number of mesh elements (for the element lookup).
    test_ess, trial_ess : mfem.intArray, optional
        Essential true dofs to eliminate on each side.
    diag_policy : {"one", "zero", "keep"}
        Diagonal entry left on eliminated rows of a square block.  ``"one"`` for
        the forward Jacobian (so that the essential block is the identity and
        solves work on a right-hand side with zeroed essential entries);
        ``"zero"`` for second-order blocks, which must annihilate the essential
        subspace.

    Returns
    -------
    mfem.HypreParMatrix
        With whatever it references (the MFEM form, on the integrator route)
        retained through :func:`own`.
    """
    if _BACKEND == "csr":

        return assemble_matrix_csr(
            test_space, trial_space, groups, element_matrices, nelem,
            test_ess=test_ess, trial_ess=trial_ess, diag_policy=diag_policy)

    test_space = as_space(test_space)
    trial_space = as_space(trial_space)
    if callable(element_matrices):
        # Only the direct-CSR route can consume chunks lazily; MFEM's callback
        # wants the arrays, so realize them here.
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
    else:
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
    """Assemble a dual (residual) vector from per-group element vectors.

    The result lives on true dofs: MFEM's ``ParallelAssemble`` applies ``P^T``,
    which is the correct reduction for a linear functional.
    """
    if _BACKEND == "csr":

        return assemble_vector_csr(space, groups, element_vectors, nelem,
                                   ess=ess, out=out)

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


def assemble_scalar(comm, element_values):
    """Sum per-element integrals into one global scalar.

    The per-group sum is taken where the values are, so a device result returns one
    scalar rather than one value per element.
    """
    from mpi4py import MPI

    loc = 0.0
    for v in element_values:
        if isinstance(v, np.ndarray):
            loc += float(np.sum(v))
        else:
            import jax.numpy as jnp

            loc += float(jnp.sum(v))
    return comm.allreduce(float(loc), op=MPI.SUM)


def _policy(name):
    return {
        "one": mfem.Operator.DIAG_ONE,
        "zero": mfem.Operator.DIAG_ZERO,
        "keep": mfem.Operator.DIAG_KEEP,
    }[name]



def mass_functional(space, w=None, coeff=None, out=None):
    r"""``M w`` as a linear form, without building ``M``.

    :math:`\ell_i = \int c\,w_h\,\phi_i`, with :math:`w_h` the finite element field
    ``w`` (a true-dof vector on ``space``; ``None`` means the constant 1) and ``c`` an
    optional :class:`mfem.Coefficient` multiplying it.  This *is* the mass matrix's
    action on ``w``, to round-off: the rule is ``MassIntegrator``'s own,
    ``2p + Trans.OrderW()`` (MFEM adds nothing for a coefficient), so the two agree on
    curved elements as well as affine ones, where the linear form's default rule of
    ``2p + 1`` would not.  Building ``M`` for a single product costs a matrix and
    hypre's assembly temporaries; this costs one element loop and one vector.  Scalar
    H1/L2 spaces only.
    """
    space = as_space(space)
    if space.vdim != 1 or not space.is_nodal:
        raise ValueError("mass_functional is for scalar H1/L2 spaces, not %s"
                         % space.fec.Name())
    keep = []
    if w is None:
        c = mfem.ConstantCoefficient(1.0)
    else:
        gf = space.to_gridfunction(w)
        host_sync(gf)                       # the coefficient is evaluated on the host
        c = mfem.GridFunctionCoefficient(gf)
        keep.append(gf)
    if coeff is not None:
        keep.append(c)
        c = mfem.ProductCoefficient(coeff, c)
    ne = space.mesh.GetNE()
    ob = int(space.mesh.GetElementTransformation(0).OrderW()) if ne else 1
    integ = mfem.DomainLFIntegrator(c, 2, ob)
    form = mfem.ParLinearForm(space.fes)
    form.AddDomainIntegrator(integ)
    form.Assemble()
    hv = form.ParallelAssemble()
    out = space.vector() if out is None else out
    out.array[:] = to_numpy(hv, copy=False)
    del hv, form, integ, c, keep
    return out


# ---------------------------------------------------------------------- native
def assemble_native_matrix(space, integrators, bdr_integrators=(), ess=None,
                           diag_policy="one", trial_space=None, trial_ess=None):
    """Assemble a block from hand-written MFEM integrators instead of AD kernels.

    Used by the priors (whose forms are fixed bilinear forms that MFEM already
    has integrators for) and available to users through ``LinearPDEProblem``.
    """
    space = as_space(space)
    if trial_space is None or trial_space is space or trial_space.fes is space.fes:
        form = mfem.ParBilinearForm(space.fes)
        form.SetDiagonalPolicy(_policy(diag_policy))
        for it in integrators:
            form.AddDomainIntegrator(it)
        for it in bdr_integrators:
            form.AddBoundaryIntegrator(it)
        form.Assemble()
        form.Finalize()
        A = mfem.HypreParMatrix()
        form.FormSystemMatrix(ess if ess is not None else empty_ess(), A)
        return own(A, form, *integrators, *bdr_integrators)

    trial_space = as_space(trial_space)
    form = mfem.ParMixedBilinearForm(trial_space.fes, space.fes)
    for it in integrators:
        form.AddDomainIntegrator(it)
    for it in bdr_integrators:
        form.AddBoundaryIntegrator(it)
    form.Assemble()
    form.Finalize()
    A = mfem.HypreParMatrix()
    form.FormRectangularSystemMatrix(
        trial_ess if trial_ess is not None else empty_ess(),
        ess if ess is not None else empty_ess(),
        A,
    )
    return own(A, form, *integrators, *bdr_integrators)

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Direct CSR assembly: scatter element arrays without MFEM's per-element callback.

The callback route (:mod:`.integrators`) hands each element array to MFEM through a
``PyBilinearFormIntegrator``.  It is the reference, since MFEM owns the dof
combination, the essential-bc elimination and the parallel reduction, but its
Python call per element is a large share of an assembly on a fine mesh
(``benchmarks/DESIGN_NOTES.md``, section 2).  This module, the default route of
:func:`hippymfem.fem.assemble.assemble_matrix`, removes that call and does what
``ParBilinearForm::ParallelAssemble`` does, one level up:

1. scatter the element arrays into the local ``ldof x ldof`` CSR graph with one
   vectorized reduction (no Python loop over elements);
2. turn that local CSR into the true-dof ``HypreParMatrix``.  When both
   prolongations are boolean (every conforming space without a
   ``DofTransformation``, decided collectively so all ranks take one branch), the
   ldof entries are mapped straight onto true-dof slots (:mod:`.tdofassemble`):
   the rank's own rows land in the diagonal and off-diagonal blocks directly, and
   the rows of shared dofs it does not own go to their owner in one ``Alltoallv``,
   so ``P^T A P`` is never formed, which is a few times cheaper.  Otherwise the CSR is
   wrapped as a block-diagonal ``HypreParMatrix`` over the **ldof** partition and
   hypre forms ``P^T A_local P`` with :func:`mfem.RAP`, which is right for
   non-conforming interfaces and for faces that carry a dof transformation alike.
   ``HIPPYMFEM_PARMAT`` forces either (:func:`set_parmat_mode`).  With hypre on a
   device every live matrix gets its own copy of the pattern's index arrays, for
   the reason in :func:`_as_sparse`;
3. eliminate essential dofs with the same ``HypreParMatrix`` calls MFEM uses
   (``EliminateRowsCols`` for square blocks, ``EliminateCols`` then
   ``EliminateRows`` for rectangular ones) or, when the prolongation is boolean,
   by masking the ldof rows and columns before step 2, which gives the same matrix
   without a second pass.

**One deliberate difference from the callback route.**  MFEM inserts element
entries with ``skip_zeros=1``, so a slot that is *exactly* zero in the element
matrices never enters the sparsity pattern.  This module keeps it, because the
pattern is built once and reused: an entry that vanishes for one parameter can be
nonzero for the next (P1 stiffness matrices on right triangles have exact zeros
from orthogonal gradients; an anisotropic coefficient fills them in).  The cost is
a few percent more nonzeros and, because hypre then sums matvec rows in a
different order, results that differ from the callback route at round-off (which a
long BFGS run can amplify into visibly different iterates).  The assembled
matrices themselves are *identical*: the test suite compares them entry by entry
on 1, 2 and 4 ranks.

The sparsity pattern and the scatter map depend only on the mesh and the spaces,
so they are built once per pair of spaces and reused by every later assembly.  That
is what pays off in an inverse problem, where a Newton-CG run reassembles ``A``,
``C``, ``W_uu`` and the other blocks hundreds of times on a fixed mesh.  Set
``HIPPYMFEM_ASSEMBLY=integrator`` or call
:func:`hippymfem.fem.assemble.set_assembly_backend` to use the callback route.
"""

import numpy as np
from .spaces import as_space
from .prolongation import (  # noqa: F401  (re-exported, see below)
    _BOOLEAN_P,
    _ESS_LDOF,
    _IDENTITY_P,
    _boolean_local,
    _boolean_prolongation,
    _ess_ldof_mask,
    _has_dof_transformation,
    _is_boolean,
    _is_identity,
    _ldof_offset,
    _ldof_starts,
    _prolongation,
)
from .parmat import (  # noqa: F401  (re-exported, see below)
    _TRANSPOSE,
    _TRIPLE_CHOICE,
    _as_sparse,
    _check_host_hypre,
    _tdof_route,
    _transposed,
    _triple,
    _triple_fused,
    _triple_split,
    _via_mfem,
    local_par_matrix,
    set_parmat_mode,
    set_triple_mode,
)
from .elimination import (  # noqa: F401  (re-exported, see below)
    _eliminate,
    _foldable,
    _keep_with,
    _set_eliminated_diagonal,
    set_fold_elimination,
)
from .pattern import (  # noqa: F401  (re-exported, see below)
    ScatterPattern,
    host_writable,
    VectorPattern,
    _FUSED_ADD,
    _PATTERN_CACHE,
    _VECTOR_CACHE,
    _device_key,
    _fused_add,
    clear_pattern_cache,
    get_pattern,
    get_vector_pattern,
    set_deterministic,
)

# This module is a facade: the four concerns live in their own modules and are
# re-exported here, so ``from hippymfem.fem.csrassemble import ...`` reaches them all:
# :mod:`.pattern` (the CSR graphs and the scatter), :mod:`.prolongation` (what ``P``
# is), :mod:`.parmat` (the parallel matrix and the triple product) and
# :mod:`.elimination` (essential dofs).  The route switches (``PARMAT_MODE``,
# ``TRIPLE_MODE``, ``FOLD_ELIMINATION``, ``DETERMINISTIC``) are read through their
# modules, since a name copied here would not follow ``set_*_mode``.


__all__ = [
    "ScatterPattern",
    "set_deterministic",
    "set_fold_elimination",
    "VectorPattern",
    "get_vector_pattern",
    "get_pattern",
    "assemble_matrix_csr",
    "add_boundary_entries",
    "assemble_vector_csr",
    "clear_pattern_cache",
]


# ------------------------------------------------------------------- assembly
class BlockPlan:
    """Everything about a block's assembly that does not depend on the values.

    Built by :func:`plan_block` from the spaces, groups and boundary conditions:
    the scatter pattern, whether the essential-dof elimination folds into the
    scatter and which slots it zeroes, the prolongations, and whether the block
    goes straight into true-dof rows.  ``target`` is the pattern the element
    entries are scattered into and ``zero`` the slots to clear in it.  Given the
    scattered values, :func:`finish_block` makes the matrix.  The split lets one
    pass over chunks feed several blocks (:func:`scatter_many`).
    """

    __slots__ = ("test_space", "trial_space", "pattern", "same", "test_ess",
                 "trial_ess", "diag_policy", "fold", "kill", "Pt", "Pr", "escapes",
                 "tpat", "target", "zero")


def plan_block(test_space, trial_space, groups, test_ess=None, trial_ess=None,
               diag_policy="one"):
    """The value-independent part of assembling a block; collective."""
    test_space = as_space(test_space)
    trial_space = as_space(trial_space)
    p = BlockPlan()
    p.test_space, p.trial_space = test_space, trial_space
    p.test_ess, p.trial_ess, p.diag_policy = test_ess, trial_ess, diag_policy
    p.pattern = get_pattern(test_space, trial_space, groups)
    p.same = test_space.fes is trial_space.fes
    p.fold = _foldable(test_space, trial_space, test_ess, trial_ess, diag_policy,
                       p.same)
    p.kill = None
    if p.fold:
        row_mask = (None if test_ess is None
                    else _ess_ldof_mask(test_space, test_ess))
        col_mask = (row_mask if p.same and test_ess is not None else
                    None if trial_ess is None
                    else _ess_ldof_mask(trial_space, trial_ess))
        p.kill = p.pattern.masked_slots(row_mask, col_mask)
    p.Pt, t_ident = _prolongation(test_space)
    if p.same:
        p.Pr, r_ident = p.Pt, t_ident
    else:
        p.Pr, r_ident = _prolongation(trial_space)
    p.escapes = t_ident and r_ident
    p.tpat = None
    if not p.escapes and _tdof_route(test_space, trial_space, p.same):
        from .tdofassemble import get_tdof_pattern

        p.tpat = get_tdof_pattern(p.pattern, test_space, trial_space)
    p.target = p.tpat.target if p.tpat is not None else p.pattern
    p.zero = p.tpat.kill(p.kill) if p.tpat is not None else p.kill
    return p


def finish_block(p, acc):
    """The matrix of a planned block from its scattered values.

    ``acc`` is what ``p.target.data`` or ``p.target.data_fused`` returned: the
    true-dof CSR plus send buffer on the true-dof route, the ldof CSR otherwise.
    """
    if p.tpat is not None:
        # Straight into true-dof rows; neither the ldof matrix nor the triple
        # product is formed.  The eliminated rows' diagonal goes into the
        # accumulator, so the matrix is complete when built
        # (TrueDofPattern.diagonal_slots).
        diagonal = None
        if p.fold and p.same and p.test_ess is not None:
            diagonal = (p.tpat.diagonal_slots(p.test_ess),
                        0.0 if p.diag_policy == "zero" else 1.0)
        A = p.tpat.finish(acc, diagonal)
        if p.fold:
            return A
    else:
        # A triple product consumes the ldof matrix and hands back a new one, so
        # that matrix never reaches the caller and can be built once and refilled.
        # When the prolongation is the identity it *is* the result, and must be
        # fresh.
        Aloc = local_par_matrix(p.pattern, acc, p.test_space, p.trial_space,
                                reuse=not p.escapes)
        if p.escapes:
            A = Aloc
        elif p.same:
            A = _triple(Aloc, None, p.Pt, p.test_space.comm, (p.test_space.fes,))
        else:
            A = _triple(Aloc, p.Pt, p.Pr, p.test_space.comm,
                        (p.test_space.fes, p.trial_space.fes))
        del Aloc
    if p.fold:
        if p.same and p.test_ess is not None:
            _set_eliminated_diagonal(A, p.test_ess,
                                     0.0 if p.diag_policy == "zero" else 1.0)
        return A
    return _eliminate(A, p.test_space, p.trial_space, p.test_ess, p.trial_ess,
                      p.diag_policy, p.same)


def scatter_many(plans, chunks):
    """One pass over chunks carrying several blocks, into every plan's target.

    ``plans`` maps a key to a :class:`BlockPlan`; ``chunks`` yields
    ``(group, start, stop, {key: array})``.  Returns ``{key: accumulator}`` ready
    for :func:`finish_block`.  Beyond the accumulators, nothing larger than one
    chunk's element matrices is resident, so a linearization point that needs five
    blocks is assembled without concatenating any of them.
    """
    state = {k: p.target.fused_begin() for k, p in plans.items()}
    for g, a, bnd, mats in chunks:
        for k, p in plans.items():
            acc, maps = state[k]
            state[k] = (p.target.fused_add(acc, maps, g, a, bnd, mats[k]), maps)
    return {k: p.target.fused_end(state[k][0], p.zero) for k, p in plans.items()}


def add_boundary_entries(p, acc, boundary):
    """Add a boundary residual's element matrices into a domain block's accumulator.

    ``boundary`` is ``(test_tables, trial_tables, element_matrices)`` over the
    boundary groups.  The dofs are the adjacent volume elements', so every entry
    has a slot in the domain pattern (:meth:`ScatterPattern.slots_of`), mapped on
    to the true-dof target where that route is taken.  The slots the folded
    elimination zeroes are zeroed again afterwards, so the elimination holds for
    the sum and one matrix comes out, instead of two to add and eliminate.  The
    streamed pass of a linearization point works the same way
    (``PDEVariationalProblem._blocks_streamed``).
    """
    test_tables, trial_tables, mats = boundary
    acc = host_writable(acc)        # the device routes return a device array
    for tt, tr, m in zip(test_tables, trial_tables, mats):
        if int(tt.group.ne) == 0:
            continue
        m = np.asarray(m, dtype=np.float64)
        shape = (int(tt.group.ne), int(tt.nd_total), int(tr.nd_total))
        rows = np.broadcast_to(tt.edofs[:, :, None], shape).reshape(-1)
        cols = np.broadcast_to(tr.edofs[:, None, :], shape).reshape(-1)
        vals = m.reshape(-1)
        if tt.signs.min() < 0 or tr.signs.min() < 0:
            vals = vals * np.broadcast_to(tt.signs[:, :, None] * tr.signs[:, None, :],
                                          shape).reshape(-1)
        slots = p.pattern.slots_of(rows, cols)
        if p.tpat is not None:
            slots = p.tpat.tslot[slots]
        np.add.at(acc, slots, vals)
    if p.zero is not None and len(p.zero):
        acc[p.zero] = 0.0
    return acc


def assemble_matrix_csr(test_space, trial_space, groups, element_matrices, nelem,
                        test_ess=None, trial_ess=None, diag_policy="one",
                        boundary=None):
    """Assemble a block from per-group element matrices without the callback.

    Signature-compatible with :func:`hippymfem.fem.assemble.assemble_matrix`;
    ``nelem`` is accepted and unused (the pattern already knows which elements
    each group holds).  Runs :func:`plan_block`, the scatter and
    :func:`finish_block`.  With
    ``boundary = (test_tables, trial_tables, element_matrices)`` over the boundary
    groups, those entries are added into the same matrix
    (:func:`add_boundary_entries`).
    """
    p = plan_block(test_space, trial_space, groups, test_ess, trial_ess,
                   diag_policy)
    if callable(element_matrices):
        # A thunk instead of the arrays: the caller is letting the scatter drive
        # the kernel, so the full (ne, nd, nd) array is never formed.
        acc = p.target.data_fused(element_matrices(), zero_slots=p.zero)
    else:
        acc = p.target.data(element_matrices, zero_slots=p.zero)
    if boundary is not None:
        acc = add_boundary_entries(p, acc, boundary)
    return finish_block(p, acc)


def assemble_vector_csr(space, groups, element_vectors, nelem, ess=None,
                        out=None):
    """Assemble a dual (residual) vector by direct scatter plus ``P^T``.

    The element vectors are summed into the local dof array with one
    :func:`numpy.bincount`, then reduced to true dofs by the space's
    prolongation transpose, which is the same ``P^T`` that
    ``ParLinearForm::ParallelAssemble`` applies.
    """
    space = as_space(space)
    pat = get_vector_pattern(space, groups)
    out = space.assemble_dual(pat.scatter(element_vectors), out)
    if ess is not None and len(ess):
        out.array[np.asarray(ess, dtype=np.int64)] = 0.0
    return out

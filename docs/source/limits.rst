Limits, stated up front
=======================

What the AD route does not cover
--------------------------------

**Non-conforming interior faces.**  Interior facet terms (UFL's ``dS``, and with them
DG formulations) are supported on conforming faces only; a face with a hanging node is
not.  The facet kernels refuse a non-conforming mesh outright, with an error that says
so, because assembling one anyway produces a facet term quietly inconsistent with the
residual.  Domain and boundary densities on the same mesh are unaffected.  See
:doc:`guide/facets`.

**Mixed spaces of different families** are outside
:class:`~hippymfem.modeling.PDEVariationalProblem.PDEVariationalProblem`, which gives
each of its three variables one space: a Taylor-Hood velocity-pressure pair or an RT-L2
mixed Darcy formulation cannot be handed to it.  The differentiation is not restricted
in the same way.  A density written over four slots in two families yields all four
Jacobian blocks, which glue into one operator with MFEM's ``HypreParMatrixFromBlocks``;
`tutorial 11 <tutorials/11_VectorAndMixedFields.html>`_ solves mixed Darcy that way, and
``test_vectorfe.py`` checks the blocks of mixed Darcy and of poroelasticity against
MFEM's own integrators.  What is missing is the wrapper, not the derivatives.  A
vector-valued space of one family (``vdim > 1`` Lagrange, H(curl), H(div)) goes through
the wrapper unchanged.

**Variable-order spaces** are rejected with a clear error; the batched element
kernels assume one finite element per group.  A boundary element with no face
transformation, which is how a non-conforming boundary face would arrive, is refused by
the boundary kernels.  Non-conforming interiors are fine for domain integrals.

Performance
-----------

A full assembly costs a few microseconds per element on a CPU core at P1 and more
at higher order; see :doc:`guide/performance` for the measured table.  dolfinx's
FFCx-generated C kernels are somewhat faster per element on a CPU: for the Jacobian of
the reference density on P2/P1 quadrilaterals at quadrature degree 6, a dolfinx assembly
takes 3.2 microseconds per element against 4.4 for a full assembly here, on one host
rank.  The GPU path closes and reverses that gap when the card has usable double
precision, which varies widely between GPU models, so measure rather than assume.

**How much a rank can hold.**  Not memory, but 32-bit indices: hypre addresses a rank's
nonzeros with an ``int``, and four million P2 hexahedra on a rank carry 2.06e9 of them,
96 % of the ceiling.  Above roughly that, add ranks.  MFEM does not accept a hypre built
with 64-bit local indices (``--enable-bigint``), only one with 64-bit *global* indices
(``--enable-mixedint``), which is what a run past 2\ :sup:`31` unknowns in all needs.
Large assemblies also want JAX's arena preallocated rather than grown
(``XLA_PYTHON_CLIENT_PREALLOCATE=true``, sized by ``HIPPYMFEM_GPU_MEM_FRACTION``; see
:ref:`gpu-memory`).

**The first assembly** of a pair of spaces builds the sparsity pattern and compiles the
kernel, about three warm assemblies in all.

Solvers
-------

``LUSolver`` is exact on any rank count, but the factorization is serial: the
matrix is gathered onto rank 0 and factorized there (on every rank with
``replicate=True``), so it is refused above 400 000 unknowns by default.  A distributed
direct solve needs a PETSc with MUMPS or SuperLU_dist (through ``PETScLUSolver``); the
petsc4py on PyPI has neither, and ``tools/install_petsc_mumps.sh`` builds a PETSc with
MUMPS and petsc4py against it.  The factorization is the cost: 105 s for 274 625
unknowns of a 3D P2 problem on four ranks, against 0.14 s a solve and 0.84 s for a
CG+BoomerAMG solve to 1e-12 (:doc:`guide/solvers`).

Randomized eigenvector tails
----------------------------

The last of the ``k`` computed eigenpairs are the least accurate, and what limits
them is the subspace the sketch spans, not round-off.  Measured on the validation
case (P2/P1 on a 12 x 12 mesh, 169 parameter dofs, so the dense generalized
spectrum is exact; ``k = 40``, a spectral ratio of 3.2e6 from the first eigenvalue
to the 40th), the relative error of the 40th eigenvalue with exact inner solves is

==================  =======  =======  =======  =======
oversampling ``p``  s = 1    s = 2    s = 3    s = 4
==================  =======  =======  =======  =======
10                  2.1e-1   7.1e-2   2.4e-2   7.8e-3
25                  6.6e-2   2.0e-3   5.9e-5   2.4e-6
60                  6.0e-3   1.4e-5   1.7e-8   4.0e-11
==================  =======  =======  =======  =======

with ``s`` the power iterations of ``doublePassG``, orthonormalized after every
application; the 20th eigenvalue (ratio 1.4e5) is at 2.7e-4 with ``p = 25, s = 1``
already and at round-off from ``s = 3``.  The inner solver's tolerance is a floor under
all of this: at ``1e-9`` the leading eigenvalues stop at about 1e-9 and the 40th at
6e-6, whatever ``p`` and ``s``.  The posterior trace, pointwise variance and KL
divergence inherit the tail's error; the dense spectrum on a coarse mesh
(``dense_spectrum`` in ``validation/run_hippymfem.py``) is the way to check.

Reproducibility
---------------

Random vectors are bit-identical across rank counts, and what follows a linear solve
agrees to that solve's tolerance (round-off with the exact solvers; see
:doc:`guide/parallel`).  GPU results agree with CPU results
at round-off (4e-16 measured) and, with the default atomics-based scatter, vary in
the last bits between runs; ``HIPPYMFEM_GPU_DETERMINISTIC=1`` removes that at the
cost of working memory.

Things that will bite you
-------------------------

**Object lifetime.**  PyMFEM hands raw pointers to MFEM, so a garbage-collected
wrapper is a segfault, not an exception.  Conversely several PyMFEM functions return
objects Python never frees.  Both are handled inside the library; code that assembles
through MFEM directly has to handle them too.  See :doc:`guide/parallel`.

**Collectives and rank-local state.**  Several MFEM calls are collective without
looking it, and several properties are rank-local without looking it.  Letting one
decide the other deadlocks the job, and because MPI busy-waits inside collectives,
the hang looks like computation.  ``tools/check_collectives.py`` scans for the
easiest instance to write by accident.

**A failed solve must not end the job.**  MFEM's hypre wrappers abort the whole MPI
job when a preconditioner setup fails.  Inside an optimizer that is wrong, and
``KrylovSolver`` turns it into a catchable error instead.

Maturity
--------

This library is young.  The argument for trusting it is not seniority but that every
claim above is attached to a check that runs in ``./run_tests.sh``.

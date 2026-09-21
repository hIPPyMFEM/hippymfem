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
`tutorial 11 <tutorials/11_VectorAndMixedFields.html>`_ solves mixed Darcy that way and
checks the pressure against the primal solution, and ``test_vectorfe.py`` assembles the
same four blocks for mixed Darcy and for poroelasticity, holding the diagonal ones
against MFEM's own integrators, the off-diagonal pair to being each other's negative
transpose, and the glued operator to acting as the four blocks do.  What is missing is
the wrapper, not the derivatives.  A vector-valued space of one family (``vdim > 1``
Lagrange, H(curl), H(div)) goes through the wrapper unchanged.

**Variable-order spaces** are rejected with a clear error; the batched element
kernels assume one finite element per group.  Non-conforming interiors are fine for
domain integrals, which the facet test exercises.  **A boundary element with no face
transformation**, which is how a non-conforming slave boundary face would arrive, is
refused by the boundary kernels with a message naming the count; that guard has never
fired here, and four attempts to provoke it (2D and 3D, isotropic and anisotropic
refinement, at the boundary and in the interior) produced none, so it is a guard rather
than a measured limit.

Performance
-----------

A full assembly costs a few microseconds per element on a CPU core at P1 and more
at higher order; see :doc:`guide/performance` for the measured table.  dolfinx's
FFCx-generated C kernels are faster per element on a CPU: for the Jacobian of the
reference density on P2/P1 quadrilaterals at quadrature degree 6, a dolfinx assembly
takes 3.2 microseconds per element against 19.4 for a full assembly here, on one host
core.  The GPU path closes and reverses that gap when the card has usable double
precision, and double-precision throughput varies by two orders of magnitude between GPU
models, so measure rather than assume.

**How much a rank can hold.**  Not memory, but 32-bit indices: hypre addresses a rank's
nonzeros with an ``int``, and four million P2 hexahedra on a rank carry 2.06e9 of them,
96 % of the ceiling.  Above roughly that, add ranks.  A second limit, MFEM's int-sized
vectors in its batched ``GetGeometricFactors``, is handled by building the quadrature
geometry here in element slices instead (:ref:`gpu-memory`).  Large assemblies also need
JAX's arena preallocated rather than grown, since one array of a sixth of the arena or
more can fail against a cap that reads empty.

Building the sparsity pattern costs a fraction of one warm assembly (0.4 to 0.7,
measured at P1 and P2 on quadrilaterals and hexahedra), and the whole one-time cost of
the first assembly in a fresh process is about three warm assemblies, most of it
compiling the kernel rather than building the pattern.  A block assembled only once is
therefore no worse off on the default route: measured cold, one assembly takes 0.18 s
against the integrator route's 0.24 s at P1 quadrilaterals, and the two are within 5 %
on P2 hexahedra.  ``HIPPYMFEM_ASSEMBLY=integrator`` remains the reference implementation
and the fallback for element families the direct scatter does not cover.

Solvers
-------

``LUSolver`` is exact on any rank count but replicates the factorization, so it is
refused above 400 000 unknowns by default.  A genuinely distributed direct solve
needs either a PETSc with MUMPS/SuperLU_dist (through ``PETScLUSolver``) or an MFEM
built with one.

Randomized eigenvector tails
----------------------------

Past a spectral ratio of about 1e5 the trailing eigenpairs are round-off limited, and
the posterior trace, pointwise variance and KL divergence inherit that.  Use more
power iterations, or the exact dense spectrum on a coarse mesh, when the tail
matters.

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

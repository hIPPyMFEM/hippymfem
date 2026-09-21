Linear solvers
==============

Every solve in the library goes through a small protocol, ``set_operator(A)`` then
``solve(x, b)``, so the forward, adjoint, incremental and prior solves can each be
given a different solver.

.. code-block:: python

   for attr in ("solver", "solver_fwd_inc", "solver_adj_inc"):
       setattr(pde, attr, hm.KrylovSolver(comm, "cg", "amg"))

Letting the size choose
-----------------------

:func:`~hippymfem.algorithms.linSolvers.auto_solver` picks a direct solve where one
is affordable and Krylov where it is not:

.. code-block:: python

   for attr in ("solver", "solver_fwd_inc", "solver_adj_inc"):
       setattr(pde, attr, hm.auto_solver(Vu, comm))

The criterion is the **global unknown count**, not the rank count.  A rule like
``comm.size == 1`` has it backwards: a one-rank run on a fine mesh is exactly where
a replicated factorization runs out of memory, while a four-rank run on a coarse one
is where an exact solve is both affordable and worth having.  The threshold is the
replicated solver's own refusal size, so the choice never lands on a solver that
would then decline; pass ``max_direct`` to change it.  Where petsc4py is importable
the distributed factorization is preferred, since it is exact without replicating
the fill.

Krylov with a hypre preconditioner
----------------------------------

When the Jacobian is symmetric positive definite, say so instead of setting the
solvers by hand: ``PDEVariationalProblem(..., spd_jacobian=True)`` (also on the
time-dependent problem) makes every default solver CG with BoomerAMG.  Measured on
the Poisson problem with the same iteration counts, a warm solve is 5-8 % cheaper on
the host at 48^3-64^3 and 27-30 % cheaper with hypre on a device (0.178 against
0.244 s at 64^3, 0.604 against 0.859 s at 96^3 on four L40S), because CG holds no
Krylov basis to orthogonalize against.  It is opt-in: the symmetry probe behind
``symmetric_jacobian="auto"`` cannot tell definite from indefinite, and CG on an
indefinite Jacobian fails where GMRES converges.

:class:`~hippymfem.algorithms.linSolvers.KrylovSolver` wraps MFEM's CG, GMRES,
FGMRES, BiCGSTAB and MINRES with BoomerAMG, ILU, Jacobi, Gauss-Seidel, ParaSails or
Euclid.  For vector-valued problems pass ``systems_dim`` (and ``elasticity=True``
with the space) so BoomerAMG uses its systems-of-PDEs mode.  ``parameters["amg_relax_type"]``
selects hypre's relaxation (``-1``, the default, keeps MFEM's choice, l1-Jacobi on a
device); ``16``, Chebyshev, measured 12 % faster per CG + AMG solve of a 2.1 M-dof
SPD operator on L40S cards, on one card and on four alike, with half the CG
iterations.  It is not the default because it needs an SPD operator (the geothermal
model's Jacobian is not) and the gain is a constant, not a change in scaling.

One behaviour differs deliberately from MFEM's default.  MFEM's hypre wrappers use
``ABORT_HYPRE_ERRORS``, so a failed AMG setup ends the whole MPI job.  Inside an
optimizer that is the wrong response: a line search routinely proposes a parameter
for which the operator is hopeless (``exp(m)`` spanning hundreds of decades is
finite but defeats AMG's coarsening), and the right answer is to reject that step.
``KrylovSolver`` sets ``WARN_HYPRE_ERRORS`` and treats a "converged" result that
contains non-finite values as non-convergence, so the step becomes a catchable
``RuntimeError`` the line search handles.

Exact solves, including in parallel
-----------------------------------

:class:`~hippymfem.algorithms.linSolvers.LUSolver` is the stand-in for hIPPYlib's
``PETScLUSolver``.  On one rank it factorizes in place with scipy's SuperLU; in
parallel it gathers the matrix onto every rank and factorizes it there.  That is
exact and rank-count independent, verified to 1e-15 by comparing a functional of
the solution across 1, 2 and 4 rank runs, at the price of a redundant
factorization: one serial factorization in time, ``O(nnz(L+U))`` per rank in memory.
It refuses matrices above ``max_global_size`` (400 000 unknowns by default) rather
than silently exhausting memory.

Why bother: an exact solve makes the reduced Hessian exact, so a spectrum or a
tolerance study measures the discretization instead of a Krylov tolerance.  Before
this, that was only possible on one rank.

Genuinely distributed direct solvers
------------------------------------

:class:`~hippymfem.algorithms.directSolvers.PETScLUSolver` uses the best
factorization the local PETSc was built with (MUMPS, SuperLU_dist, STRUMPACK or
PaStiX), and otherwise PETSc's ``redundant`` with an inner ``lu``, which is exact as
well.  :class:`~hippymfem.algorithms.directSolvers.PETScKrylovSolver` exposes
PETSc's Krylov methods and preconditioners on the same matrices.

.. warning::

   **petsc4py must be imported before PyMFEM.**  In the reference environment the
   reverse order fails with ``libmpi_mpifh.so.40: undefined symbol:
   mpi_conversion_fn_null_``, because the system MPI Fortran library is found first.
   Set ``HIPPYMFEM_PETSC=1`` before importing hippymfem, or import petsc4py at the
   top of the script.  ``hm.petsc_available()`` returns ``(True, "")`` or
   ``(False, reason)`` so a script can report the reason rather than fail
   mysteriously.

If MFEM itself was built with MUMPS, SuperLU_dist or STRUMPACK, those are reachable
through ``assemble_native_matrix`` and MFEM's own solver classes;
``hm.mfem_config()`` says which are present.

Choosing
--------

=========================================  ==================================
situation                                  solver
=========================================  ==================================
large problem, well-conditioned operator   ``KrylovSolver(comm, "cg", "amg")``
exact Hessian or spectrum needed           ``LUSolver`` up to ~4e5 unknowns
exact and larger                           ``PETScLUSolver`` with MUMPS
mass-matrix-like operator, cheap            ``LumpedMassSolver``
=========================================  ==================================

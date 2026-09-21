Parallel correctness
====================

The rule
--------

**Never let a collective, or a process-ending abort, or a tolerance, depend on
state that only one rank or one solver knows.**

Every parallel defect found while building this library was an instance of that
rule, and none of them announced itself.  OpenMPI busy-waits inside collectives, so
a deadlocked rank shows 95% CPU and looks like slow computation rather than a hang.

Ways to break it
----------------

**A collective inside a rank-0 block.**  ``ParVector.inner``, ``.norm`` and
``.sum`` are collectives, as is ``Model.cost`` (the misfit reduces and the prior
solves).  Calling one inside ``if rank == 0:`` deadlocks.  ``tools/check_collectives.py``
scans for the version of this that is easiest to write by accident.

.. code-block:: python

   # wrong
   if comm.rank == 0:
       print("cost", model.cost(x))          # cost() is collective

   # right
   c = model.cost(x)
   if comm.rank == 0:
       print("cost", c)

**Branching on a rank-local mesh property before a collective.**
``GetGeometricFactors`` is collective on a ``ParMesh`` (it builds the nodal grid
function) while ``GetNumGeometries`` is rank-local: on a mixed triangle and
quadrilateral mesh one rank can hold only triangles and see one geometry while
another holds both and sees two.  The two then take different code paths and the job
hangs.  ``MeshBatches`` decides once from ``allreduce(..., MAX)``.

**Memoizing on rank-local state.**  Caching a layout on ``(comm, local_size)``
looks obviously safe and is a correctness bug: local sizes differ, so one rank hits
the cache and skips the ``allgather`` while another misses and performs it.  The
ranks desynchronize and a *later, unrelated* collective receives the wrong message.
Caches in this library are keyed only on values every rank agrees on, or guard work
that involves no communication at all.

**A tolerance that depends on the solver.**  Asserting an absolute gradient
tolerance is not meaningful when the available accuracy floor is 1e-13 with an
exact solve and 1e-7 with an iterative one.  Assert a *reduction* instead.

Partition independence
----------------------

The random inputs of a run do not depend on the number of ranks:

* random vectors from :data:`~hippymfem.common.random.parRandom` are bit-identical on
  1, 2 and 4 ranks (the suite checks it): the stream is a counter-based Philox
  generator indexed by global dof rather than by local position;
* the white noise behind a prior sample is keyed on a hash of the element centroid
  quantized to 20 bits per dimension (``ParMesh.GetGlobalElementNum`` is *partition
  dependent*, so keying on it would not do), so it is the same field on any partition.

What follows a linear solve agrees to that solve's tolerance: a prior sample, the
synthetic data and the MAP point are the same across rank counts to round-off with the
exact solvers and to the Krylov tolerance otherwise (1e-14 measured for a prior sample
solved to that tolerance).

A vector keyed on the **global true dof index** is *not* a partition-independent
object: hypre numbers true dofs rank by rank, so index 5 is a different mesh entity
at a different rank count.  Compare functions, or functionals of them, not dof
vectors.

Assembly in parallel
--------------------

Two routes turn the local element arrays into a parallel matrix, and
``HIPPYMFEM_PARMAT`` (``auto`` by default) chooses between them per block,
collectively.

Where **both prolongations are boolean**, which is every conforming space without a
``DofTransformation``, the entries go straight into true-dof rows: for a boolean ``P``
the triple product is a permutation with a sum over shared dofs, so the rank's own rows
land in hypre's diagonal and off-diagonal blocks directly and the rows it does not own
reach their owner in one ``Alltoallv``.  ``P^T A P`` is never formed.

Otherwise, a non-conforming space or one with a ``DofTransformation``, the local
``ldof x ldof`` matrix is wrapped as a block-diagonal ``HypreParMatrix`` over the
**ldof** partition and ``P^T A P`` is formed with ``RAP`` against
``Dof_TrueDof_Matrix``, which is exactly what ``ParBilinearForm::ParallelAssemble``
does.  All communication, including the non-conforming interfaces, then stays inside
hypre, and nothing about that reduction is reimplemented.

The two routes agree to round-off on every block, which ``test_assembly`` checks by
assembling the same problem both ways (``tdof`` and ``mfem``), and the result of either
is bit-identical to MFEM's own route.

Object lifetime
---------------

PyMFEM hands raw pointers to MFEM, so a garbage-collected wrapper is a segfault
rather than an exception, and in the opposite direction several PyMFEM functions
return objects Python never frees (``RAP``, ``Add``, ``ParAdd`` and the
``Eliminate*`` methods: ``%newobject`` is commented out for them).  Both are handled
by ``KeepAlive``, ``fem.assemble.own`` and ``common.linalg.take_ownership``, and both
will bite code that assembles through MFEM directly.

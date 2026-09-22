Running on a GPU
================

What runs where
---------------

Two things can move to a GPU, and they are switched on separately.

===============================  ================================================
setting                          what moves
===============================  ================================================
``HIPPYMFEM_DEVICE=gpu``         the element kernels, the dof gather and the
                                 scatter into the CSR structure, through JAX.
                                 Works with any PyMFEM.  ``auto`` does the same
                                 when the process can see a GPU and nothing
                                 otherwise, so one environment serves a laptop
                                 and a GPU node.
``HIPPYMFEM_HYPRE_DEVICE=1``     MFEM's matrices and every hypre solve as well.
                                 Needs PyMFEM built for CUDA
                                 (``tools/build_pymfem_cuda.sh``) or for HIP
                                 (``tools/build_pymfem_hip.sh``, see `AMD cards`_).
===============================  ================================================

The second is where the large factors are: the solves are most of a Newton step.  Check
what your PyMFEM can do rather than assume it:

.. code-block:: python

   import hippymfem as hm
   c = hm.mfem_config()
   print(c["version"], "CUDA:", c["MFEM_USE_CUDA"], "HIP:", c["MFEM_USE_HIP"])

Setting ``HIPPYMFEM_HYPRE_DEVICE=1`` with a PyMFEM built for neither raises a
``RuntimeWarning`` at import and leaves MFEM and hypre on the host.

Turning it on
-------------

JAX fixes its platform list when it is imported, so both settings are environment
variables read at ``import hippymfem``:

.. code-block:: bash

   HIPPYMFEM_DEVICE=gpu python my_script.py
   HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 \
       mpirun -n 4 tools/mpirun_pinned.sh python my_script.py

The script itself does not change.  ``tools/mpirun_pinned.sh`` gives each rank one card
before CUDA initializes (see `Taking exactly the GPUs and cores you asked for`_).
``HIPPYMFEM_DEVICE=gpu`` sets ``JAX_PLATFORMS`` to one accelerator and the host
(``cuda,cpu`` on an NVIDIA node, ``rocm,cpu`` on an AMD one), so both backends are live
and :func:`~hippymfem.fem.kernel.set_device` can switch between them in one process,
which is how the test suite compares them on identical inputs:

.. code-block:: python

   from hippymfem.fem import kernel as K
   K.set_device("gpu")           # one GPU per rank, by node-local rank
   print(K.device(), K.on_gpu())

Two memory settings are handled for you.  JAX's preallocation is switched off, and each
process's share of its card is capped at 0.90 divided by the ranks sharing the card
(0.45 when hypre is on the card too), because JAX otherwise claims 75 % of the device
*per process*.  ``HIPPYMFEM_GPU_MEM_FRACTION`` overrides the share; :ref:`gpu-memory`
says when to change it.

.. _mfem-device:

What to expect
--------------

**Moving the kernels alone is worth little; moving the solves is worth a lot.**  One
Newton step at 531 441 state dofs on one L40S, built from measured components (one
assembly of the linearization point, a forward and an adjoint solve, and twenty CG
iterations at two incremental solves each), every configuration reaching the same state
to eleven digits:

======================  ===========  ===========  ============  ==============
where the work runs     assemble     fwd solve    incr solve    Newton step
======================  ===========  ===========  ============  ==============
all on the host         28.3 s       4.41 s       3.910 s       193.4 s
GPU kernels only        1.8 s        3.93 s       3.910 s       166.0 s
GPU kernels and hypre   5.9 s        0.51 s       0.066 s       **9.6 s**
host kernels, GPU       31.7 s       0.99 s       0.063 s       36.2 s
hypre
======================  ===========  ===========  ============  ==============

The kernels-only configuration is worth 1.17x on a step and the full one 20x.

**Two Newton-CG steps of the benchmark problem** (P2 hexahedra for the state and adjoint,
P1 for the parameter, 25 CG iterations allowed per step and 8 taken, CG with BoomerAMG
for every solve), on one node with four L40S and two AMD EPYC 9334:

=============================  =============  =====================  =====================  ================
64\ :sup:`3`, 4.6 M unknowns   forward solve  Hessian blocks (warm)  reduced-Hessian apply  two Newton steps
=============================  =============  =====================  =====================  ================
1 L40S                         3.4 s          1.73 s                 0.62 s                 **27.9 s**
2 L40S                         1.5 s          0.67 s                 0.47 s                 14.4 s
4 L40S                         0.9 s          0.39 s                 0.32 s                 **9.6 s**
4 ranks, all host              12.5 s         1.5 s                  10.9 s                 179.5 s
hIPPYlibx, 4 ranks, host       29.4 s         8.7 s                  13.8 s                 402.5 s
=============================  =============  =====================  =====================  ================

=============================  =============  =====================  =====================  ================
128\ :sup:`3`, 36 M unknowns   forward solve  Hessian blocks (warm)  reduced-Hessian apply  two Newton steps
=============================  =============  =====================  =====================  ================
2 L40S                         10.7 s         5.2 s                  3.9 s                  87.7 s
4 L40S                         6.1 s          2.8 s                  2.2 s                  **49.5 s**
4 ranks, all host              108.5 s        11.9 s                 101.3 s                1409.3 s
hIPPYlibx, 4 ranks, host       241.0 s        71.4 s                 140.7 s                2669.0 s
=============================  =============  =====================  =====================  ================

==============================  =============  =====================  =====================  ================
256\ :sup:`3`, 287 M unknowns   forward solve  Hessian blocks (warm)  reduced-Hessian apply  two Newton steps
==============================  =============  =====================  =====================  ================
32 Blackwell slices (16 cards)  8.5 s          4.2 s                  2.6 s                  **77.4 s**
32 ranks, all host              not timed      not timed              not timed              2859.0 s
hIPPYlibx, 32 ranks, host       423.9 s        102.4 s                378.9 s                6187.2 s
==============================  =============  =====================  =====================  ================

The 256\ :sup:`3` rows are rank for rank as well, on 32 ranks: the GPU ranks are 48 GB MIG
slices of 16 RTX PRO 6000 Blackwell cards of a cluster, the host ranks 32 cores of the node
above, one core per rank (with two cores per rank the hIPPyMFEM host run takes 2456 s).

Four cards against four host ranks of the same library are 19x at 64\ :sup:`3` and 28x at
128\ :sup:`3`, and 32 slices against 32 host ranks 37x at 256\ :sup:`3`, with the same cost
functional to eight digits and the same CG counts.  The
hIPPYlibx rows are the same problem (mesh, spaces, PDE, data model, prior, Newton-CG
settings) solved by `hIPPYlibx <https://github.com/hIPPyMFEM/hippylibx>`_ on dolfinx 0.10,
with PETSc's CG and BoomerAMG given the same BoomerAMG options.  The two libraries draw
different random data, so their CG counts differ (11 and 6 against 8): four cards are 42x
and 54x faster than hIPPYlibx as measured, and 37x and 60x when it is charged the same CG
counts.  At 256\ :sup:`3` every run takes 9 CG iterations, and the 32 slices are 80x faster.
The host rows use four of the node's 64 cores (32 at 256\ :sup:`3`), the same rank count as
the GPU rows, so they are a device swap at a fixed rank count and not the machine's best
host time.

The first linearization point of a run pays the sparsity patterns and JAX's compilation
once (5 s at 64\ :sup:`3` on one card, 24 s at 128\ :sup:`3` on two); a Newton run pays
the warm figure from its second step.

**Other cards and larger problems**, same benchmark:

==============  ===========  ==================================  =====================
mesh            unknowns     GPUs                                two Newton-CG steps
==============  ===========  ==================================  =====================
64\ :sup:`3`    4.57 M       1 AMD MI210                         18.7 s
128\ :sup:`3`   36.1 M       1 H100 (80 GB)                      109 s
128\ :sup:`3`   36.1 M       4 AMD MI210                         31.9 s
256\ :sup:`3`   287 M        8 H100                              141 s
256\ :sup:`3`   287 M        8 RTX PRO 6000 Blackwell            140 s
256\ :sup:`3`   287 M        16 RTX PRO 6000 Blackwell           77 s
400\ :sup:`3`   1.09 B       24 RTX PRO 6000 Blackwell           197 s for one step,
                                                                 compilation included
==============  ===========  ==================================  =====================

The Blackwell cards were split into two 48 GB MIG slices each, one rank per slice.
Doubling them at 256\ :sup:`3` is 1.81x, 90 % of linear.

.. _smallest-kernel:

Where the device loses
----------------------

**With hypre on the host, the solves do not move.**  A reduced-Hessian application is
two linear solves and a few matvecs that cost the same whatever the kernels run on.
Measured on a 274 625-dof problem at one rank: assembly 7.72 s on the CPU against 0.65 s
with GPU kernels, and the Hessian application 3.62 s against 3.68 s.

**There has to be enough arithmetic per element.**  The full-assembly speedup grows with
the work an element kernel does, from about 9x on a P1 quadrilateral to about 15x on a P2
hexahedron, and more than that on the kernel alone (:doc:`performance`).

**There has to be enough of the mesh on each rank** to cover the fixed per-assembly
cost.  For P1 quadrilaterals, the smallest kernel in the library, the measured
full-assembly speedup on an L40S is

==========================  ===========  ==========
elements per rank           ranks        speedup
==========================  ===========  ==========
582                         4            1.3x
1154                        2            0.4x
2304                        1            0.7x
32400                       1            9.5x
==========================  ===========  ==========

The first three are reproducible, not noise: both devices carry a fixed per-assembly
cost and which one dominates depends on the local element count.  Above about
:math:`10^4` elements per rank the device wins on every case measured.  With the solves
on the card the same holds for strong scaling: at 32\ :sup:`3` four cards are barely
faster than one, at 128\ :sup:`3` two to four cards scale at 89 %, so plan for about a
million state dofs per GPU.

**Double precision varies by two orders of magnitude across cards.**  Element kernels
are double precision throughout.  Measured with a 4096-cubed matrix multiply:

==================  ===================  ===================
card                fp64                 fp32
==================  ===================  ===================
L40S                1.25 TFLOP/s         125 TFLOP/s
H100 80GB HBM3      349 TFLOP/s          381 TFLOP/s
==================  ===================  ===================

On the same code the kernel speedup over a host core is 11x to 26x on an L40S and 11x to
114x on an H100 (quadrilateral P1 to hexahedral P3).  The library therefore does not
promise a speedup; :mod:`hippymfem.test.test_gpu` measures one for whatever card is
present.

.. _gpu-memory:

How large a problem fits
------------------------

A Newton step holds about 2 kB of device memory per unknown (state, parameter and
adjoint together): the assembled blocks and the AMG hierarchies.  That puts
128\ :sup:`3` (36 M unknowns) on four 45 GB L40S at 17 to 19 GiB per card, on two of them
at 29 to 35 GiB, or on one 80 GB H100 at 69 GiB.  What decides whether a given run fits:

**JAX's share of the card.**  The default (0.45 with hypre on the card) is the faster
operating point; a quarter gives memory back for a few percent of time, because it only
makes the element kernels run in more, smaller chunks.  Two Newton-CG steps on four L40S:

.. table::
   :widths: auto

   ========================  ==========  ================  =========  ==========
   ..                        assembly    two Newton steps  JAX peak   card peak
   ========================  ==========  ================  =========  ==========
   64\ :sup:`3`, 0.45         1.030 s     14.20 s           2.3 GB     7.1 GiB
   64\ :sup:`3`, 0.25         1.029 s     14.36 s           2.2 GB     7.2 GiB
   128\ :sup:`3`, 0.45        7.42 s      71.8 s            10.5 GB    30.2 GiB
   128\ :sup:`3`, 0.25        7.49 s      74.7 s            3.8 GB     21.2 GiB
   ========================  ==========  ================  =========  ==========

(These rows predate later speedups; the comparison within each pair is what they show.)
Set ``HIPPYMFEM_GPU_MEM_FRACTION=0.25`` when a run is short of card memory.
``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` is not a substitute: it reports no budget for
the chunk planner to work from, and its card peak was higher.

**What the problem class keeps.**  Three settings of
:class:`~hippymfem.modeling.PDEVariationalProblem.PDEVariationalProblem` decide how much
a Newton step holds:

* A symmetric Jacobian is detected by a probe at every assembly
  (``symmetric_jacobian="auto"``, the default), and the adjoint solves then reuse ``A``
  and its AMG hierarchy instead of a transposed copy and a second hierarchy: 3 GB plus
  6 GB per card at 128\ :sup:`3`.
* ``transpose_free_adjoint=True`` does the same for a non-symmetric Jacobian: it solves
  ``A^T p = b`` through ``A``'s transpose action with ``A``'s own hierarchy as the
  preconditioner.  The geothermal application at 128\ :sup:`3` needed it to fit four
  L40S.
* ``release_linearization_on_move=True`` drops the previous linearization point at the
  first forward solve for a new parameter.  It is safe for line-search Newton-CG and
  opt-in because other algorithms may use the old point after a trial solve.

The library itself holds only what is current: a solver keeps its present operator and
hierarchy and no other, an operator is released before its replacement is assembled, and
MFEM's cached geometric factors are freed once copied
(``HIPPYMFEM_KEEP_GEOMETRIC_FACTORS=1`` keeps them, if your own code holds a pointer to
them).  ``test_device`` checks that card memory is flat across operator resets.

**How the element batch is split.**  The element kernels run the batch in chunks sized
before the first launch from the kernel's shape (forward-mode AD carries every tangent
through every quadrature point, measured at 7 to 13 doubles per pair) against a third of
what is free, where free is the smaller of JAX's remaining budget and what the driver
reports (``HIPPYMFEM_GPU_MEM_RESERVE``, in GiB, is subtracted for whatever else shares the
card).  On the host the budget is this rank's share of the node's available memory
(``HIPPYMFEM_HOST_MEM_FRACTION``).  An out-of-memory error shrinks the chunk and retries,
and ``HIPPYMFEM_ELEMENT_CHUNK`` pins a size.  Once a batch is split, nothing that is only
read a chunk at a time stays resident: each chunk of element matrices is scattered as it
is produced, and the per-element geometry and the scatter map stay on the host and are
sliced per chunk (above ``HIPPYMFEM_GEOMETRY_STREAM``, 0.25 of the budget, for the
geometry).  Chunking changes results at round-off only (about 1e-15 relative), and an
unsplit batch stays bit-identical.

The chunk count is a memory choice, not a speed one.  A launch costs about 5 ms: at
1 906 624 P2 hexahedra on one L40S a warm Jacobian assembly takes 9.3 s in 128 chunks and
9.1 s in 16, while the 16 chunks raise JAX's peak from 10.8 GiB to 26.2.
``HIPPYMFEM_CHUNK_PLAN=xla`` sizes the chunk from XLA's memory analysis of the compiled
pass instead of the estimate (143 kB an element against the estimate's 664 for the P2
hexahedral Hessian pass), which runs 8 to 17 times fewer chunks and is for a card with
memory to spare: at a million elements on one L40S it took a warm linearization point
from 6.9 s to 6.5 s and JAX's peak from 6.1 GiB to 14.4.

With these in place a P2 hexahedral Jacobian assembly fits on one 45 GB L40S up to:

=====================  ====================  ======================
elements per rank      state dofs            card memory
=====================  ====================  ======================
343 000                2 803 221             17.1 GB
884 736                7 189 057             20.1 GB
1 331 000              10 793 861            22.5 GB
1 906 624              15 438 249            36.1 GB
=====================  ====================  ======================

**One large array against a grown arena.**  JAX's allocator takes its arena from the
driver in regions and counts every region against the cap even while it is free, so a
request for one large *contiguous* array can fail against a cap that reads almost empty.
The array that provokes it is the assembly's accumulator, one double per nonzero: 7.7 GiB
at two million P2 hexahedra on a rank.  At 128\ :sup:`3` on two L40S with the share at a
quarter of the card, a 4.0 GiB accumulator cannot be allocated on the first assembly,
while the same arena taken as one region serves it; 256\ :sup:`3` on eight H100 behaves
the same way.  When one array is more than roughly a sixth of the arena, set
``XLA_PYTHON_CLIENT_PREALLOCATE=true`` and size ``HIPPYMFEM_GPU_MEM_FRACTION`` for
hypre's share; the assembly says so in a ``RuntimeWarning`` that names the size and the
share.  A large accumulator is then kept between assemblies and zeroed in place
(``HIPPYMFEM_FUSED_KEEP``, ``HIPPYMFEM_FUSED_KEEP_SHARE``).

**Two ceilings past a few million elements a rank**, neither of them the card.  MFEM
sizes the vectors of its batched ``GetGeometricFactors`` with an ``int``, which overflows
at four million hexahedra with 64 quadrature points; above that limit the geometry is
built here instead, from the mesh nodes, one slice of elements at a time, and agrees with
MFEM's factors to 2.4e-15 (``HIPPYMFEM_GEOMETRY_SLICE`` forces that path).  What remains
is hypre's 32-bit indices: a rank holding four million P2 hexahedra carries 2.06e9
nonzeros in the state Jacobian, 96 % of what they can address, so add ranks past that.

The sparsity patterns
---------------------

Before its first assembly a space pair builds two patterns once: the scatter pattern
(every element-matrix entry mapped to a CSR slot) and, in parallel, the true-dof pattern
(this rank's rows merged with those other ranks send, laid out as hypre's diagonal and
off-diagonal blocks).  At scale this is the largest one-time cost of a run: 1.5 billion
entries a rank at 256\ :sup:`3` on eight GPUs.

**Large patterns are built without a global sort** (:mod:`hippymfem.fem.patternbuild`).
An element matrix contributes, for each of its rows, a contiguous run of entries sharing
that row, so the entries are grouped by row in one counting pass and each row's ninety or
so entries are ordered on their own, diagonal first as hypre wants it.  The true-dof
pattern is merged and laid out the same way.  The arrays are identical to the sort
route's, dtypes included, which ``test_assembly`` checks on every element family at one,
two and four ranks.  It runs on the host with numba, uses no device memory and about half
the host memory of the sort, and takes this rank's share of the node's cores as its
thread count (``HIPPYMFEM_PATTERN_THREADS``).  Measured:

=============================================  ==============  ==============
..                                             sort route      sort-free
=============================================  ==============  ==============
scatter pattern, 96\ :sup:`3`, one rank        27.3 s          9.3 s (8 threads),
(645 M entries; the sort on an L40S)                           5.0 s (64)
true-dof merge, 128\ :sup:`3` on two L40S      17.0 s          5.7 s
true-dof block layout, same run                13.7 s          0.42 s
first forward solve, same run                  96.2 s          50.3 s
cold Hessian blocks, same run                  31.0 s          23.7 s
=============================================  ==============  ==============

It is used when numba is importable and the pattern has at least
``HIPPYMFEM_PATTERN_BUILDER_MIN`` entries (2\ :sup:`24`; below that the sort is quicker
than compiling).  ``HIPPYMFEM_PATTERN_BUILDER=sort`` restores the sort route, whose sort
runs on the card when the kernels do, through CuPy's radix sort where CuPy is importable
and XLA's otherwise (9.2 ns a key against 24; ``HIPPYMFEM_PATTERN_SORT_KERNEL`` forces
either).  ``HIPPYMFEM_PATTERN_TIMING=1`` prints what each phase of a build cost.

How the parallel matrix is built
--------------------------------

One assembly costs one host-to-device copy of the local dof vectors and one
device-to-host copy of the assembled CSR values; the gather, the batched kernel, any
``DofTransformation`` and the scatter all run on the device, and the ``(ne, nd, nd)``
element-matrix array never reaches the host.

``HIPPYMFEM_PARMAT`` selects how the local CSR becomes the parallel matrix.  ``auto``,
the default, assembles straight into true-dof rows whenever both prolongations are
boolean (every conforming space without a ``DofTransformation``, decided collectively):
the rank's own rows land in hypre's diagonal and off-diagonal blocks directly and the
rows of shared dofs it does not own reach their owner in one ``Alltoallv``, so
``P^T A P`` is never formed.  Otherwise MFEM's ``HypreParMatrix`` constructor forms the
triple product, which is the route ``mfem`` forces.  ``direct`` builds from raw pointers,
is host-only, and under a device-configured MFEM raises a ``RuntimeError`` naming the fix.

With hypre on a device, ``HIPPYMFEM_PARMAT_DEVICE=block`` (the default) hands hypre its
two blocks as the pattern laid them out; ``copy`` hands MFEM one row-major CSR and lets it
re-derive the split at every assembly, which costs 2.66 s per assembly at 128\ :sup:`3`
and is kept as a fallback.  Every rank must agree, so set it in the environment.

Reproducibility and correctness
-------------------------------

The default device scatter is a scatter-add, which a GPU implements with atomics:
correct, but the summation order varies between runs, so two identical assemblies differ
in the last bits (measured spread 5e-17 relative).  ``HIPPYMFEM_GPU_DETERMINISTIC=1``
switches to a scatter in which each CSR slot sums its contributions in a fixed order,
making repeated assemblies **bit-identical**, at a cost of ``nnz * maxc`` doubles of
working memory (``maxc`` is 4 for P1 quadrilaterals, 9 for P2).  The two modes agree to
1e-16.

``HIPPYMFEM_DEVICE=gpu python -m hippymfem.test.test_gpu`` assembles the same problems on
both devices in one process and requires every block, every assembled parallel matrix
and every residual to agree at round-off (worst case 4.3e-16 on element arrays, 3.6e-16
on assembled operators), and solves a whole inverse problem on each.  With no GPU visible
it prints the reason and skips.  ``hippymfem/test/test_device.py`` is the suite that puts
hypre on the device; ``run_tests.sh`` runs it, through the pinning wrapper, when told
where that build is:

.. code-block:: bash

   HIPPYMFEM_CUDA_PYMFEM=/path/to/pymfem-cuda/site-packages ./run_tests.sh 1 2

Under a device-configured MFEM, host reads and writes of MFEM data go through
:func:`~hippymfem.common.parvector.host_sync` and ``host_readwrite``.  Code that calls
MFEM directly has to do the same: MFEM keeps returning the host pointer from
``GetDataArray()`` whether or not it is current, and the symptom is wrong numbers, not an
error.

The Laplace approximation on the cards
--------------------------------------

``benchmarks/bench_laplace.py`` times the stages after the MAP point: the randomized
generalized eigensolver (``doublePassG``, k = 50, p = 20), posterior sampling, the
pointwise variance and the traces.

=======================================  ===========================  ====================  ====================  =====================
Laplace stage                            32\ :sup:`3`, host, 4 ranks  32\ :sup:`3`, 1 L40S  64\ :sup:`3`, 4 L40S  128\ :sup:`3`, 4 L40S
=======================================  ===========================  ====================  ====================  =====================
eigensolver (140 Hessian applies)        199.1 s                      12.1 s                34.9 s                234.2 s
one posterior sample                     165 ms                       63 ms                 60 ms                 229 ms
pointwise variance, randomized, r = 64   4.6 s                        2.7 s                 6.5 s                 19.4 s
pointwise variance, Monte Carlo, n = 64  --                           3.8 s                 3.0 s                 11.4 s
traces, r = 64                           5.8 s                        3.2 s                 7.2 s                 21.7 s
=======================================  ===========================  ====================  ====================  =====================

The random stream is a vectorized Philox generator, bit-identical to numpy's and on the
card when the kernels are, which is what makes a sample tens of milliseconds.  The
``"Randomized"`` pointwise variance is a truncated spectrum and 20-25 % low at r = 64;
the ``"MonteCarlo"`` method (one solve per sample) is unbiased and is what the benchmark
reports.  The incremental and prior solves can run at 1e-8 for these stages: the
eigenvalues move by 6e-6 at most and the eigensolver is 1.4x faster.  At 128\ :sup:`3`
the whole workflow, MAP included, runs in under 20 minutes on four L40S.

Taking exactly the GPUs and cores you asked for
-----------------------------------------------

**GPUs.**  JAX opens a context on every device it can see when its backend comes up, not
only on the one it computes on: on a four-card node a one-rank job held 439 MB on each of
the three idle cards.  The cure is one visible device per rank, set before the process's
first CUDA call, which is ``MPI_Init`` when the MPI is CUDA-aware.
``tools/mpirun_pinned.sh`` sets it at the launcher::

   mpirun -n 4 tools/mpirun_pinned.sh python script.py

The library also pins itself (``HIPPYMFEM_PIN_GPU``, on by default) when it is imported
before ``mpi4py`` and MFEM; imported after them it is too late, it says so on rank 0, and
the wrapper is the way.  With no device id, ``mfem.Device("cuda")`` puts **every rank on
GPU 0**; :func:`~hippymfem.common.mfemconfig.configure_device`, which the import runs
for you, picks ``local_rank % n_devices``, the same rule the element kernels use, so a
rank's matrix and its kernels share a card.

**A host run holds nothing on the cards.**  JAX reads ``JAX_PLATFORMS`` when *it* is
imported, so a script that imports ``jax`` before ``hippymfem`` used to bring up the CUDA
backend on every card.  hippymfem now also sets the platform list through ``jax.config``
when JAX is already imported, and when it is imported first it empties the visible-device
list for a host run, so nothing else in the process can open a context either.

**Cores.**  With the element kernels on the CPU, XLA's worker pool takes about two cores
per rank whatever ``OMP_NUM_THREADS`` says.  OpenMPI's default ``--bind-to core``
confines each rank to one core; a run started without ``mpirun`` should be given
``taskset`` for the same reason.

AMD cards
---------

Everything runs on an AMD card: the element kernels through JAX, and MFEM and hypre
through a HIP build of PyMFEM.  Measured on an AMD Instinct MI210 (64 GB, ROCm 7.2,
``jax-rocm7-plugin`` 0.11.1, hypre 3.2.0, MFEM 4.9):

===========================================  ======  ======  ======
two Newton-CG steps                          MI210   H100    L40S
===========================================  ======  ======  ======
:math:`32^{3}`, one GPU                      1.80 s  1.6 s   2.0 s
:math:`64^{3}`, one GPU                      18.7 s  18.5 s  27.9 s
:math:`128^{3}`, four GPUs                   31.9 s  --      49.5 s
:math:`64^{3}`, kernels only, hypre on host  720 s   --      --
===========================================  ======  ======  ======

The answers are the NVIDIA ones: the same cost functional to nine digits and the same CG
counts at every size, and the geothermal application at :math:`32^{3}` reproduces its
L40S row.  :math:`128^{3}` does not fit on one 64 GB card but runs on two: 65.9 s,
against 89.4 s on two L40S with the same flags.

**Kernels.**  ``HIPPYMFEM_DEVICE=gpu`` names ``rocm,cpu`` on a node whose card
``rocm-smi`` reports.  One XLA setting is changed for ROCm: command buffers (the
counterpart of CUDA graphs) segfault inside ``libamdhip64`` once an element batch passes
a size that depends on the block, so ``--xla_gpu_enable_command_buffer=`` is added to
``XLA_FLAGS`` when ROCm is selected.

**MFEM and hypre.**  PyMFEM's build system knows only CUDA, so
``CPU_PYMFEM=<a CPU PyMFEM tree> tools/build_pymfem_hip.sh <prefix> gfx90a`` builds the
three pieces itself:

#. hypre **3.2** with ``--with-hip --without-umpire``.  The version matters: hypre added
   ROCm 7 support in 3.1.0, and 2.32, the version PyMFEM pins, faults in every device
   kernel on ROCm 7.
#. MFEM with ``MFEM_USE_HIP``, ROCm's ``clang++`` under OpenMPI's wrapper, and two
   patches: ``sparsemat.hpp`` guards ``SparseMatrix``'s hipSPARSE members on a macro the
   host-compiled wrappers do not see, so the two sides would disagree on the object's
   size; and ``SparseMatrix::AddMult`` stays on MFEM's own device kernel, because
   hipSPARSE's SpMV on ROCm 7 is right on a matrix's first call and wrong on every later
   one.
#. PyMFEM's parallel wrappers recompiled against that MFEM.

Then ``HIPPYMFEM_HYPRE_DEVICE=1`` puts MFEM on ``mfem.Device("hip")``:
:func:`~hippymfem.common.mfemconfig.configure_device` reads the backend from the build,
so ``"cuda"`` and ``"gpu"`` in a script mean HIP on this build and nothing written for
NVIDIA changes.  Rank pinning, in the library and in ``tools/mpirun_pinned.sh``, writes
``ROCR_VISIBLE_DEVICES`` on an AMD node.  On the MI210, ``run_tests.sh 1 2`` with the HIP
build passes every suite, ``test_gpu`` and ``test_device`` included.

Going further
-------------

``tools/build_pymfem_cuda.sh`` has the CUDA build recipe, including the upstream problems
it works around.  ``benchmarks/DESIGN_NOTES.md`` records the measurements behind the
design decisions summarized here (the chunk planner, the true-dof route, the device
matrix constructors, the memory fixes), and :doc:`performance` has the host-side picture.

Performance
===========

Every number on this page was measured by ``benchmarks/bench_assembly.py`` or
``benchmarks/bench_pipeline.py`` on the machine it names, and is reproduced by
running it.  Nothing here is a projection.

Where the time goes
-------------------

After the forward and adjoint solves, an inverse problem spends its time assembling:
a Newton-CG run reassembles the Jacobian and the second-order blocks at every
linearization point, and a nonlinear forward solve or an MCMC chain reassembles at
every step.  An assembly has three parts:

#. the **dof gather**: local dof vector to element dof arrays;
#. the **element kernel**: the JAX program that differentiates the density;
#. the **scatter**: element arrays into the parallel matrix, then ``P^T A P`` and
   essential-dof elimination.

Which stage costs what
----------------------

``benchmarks/bench_pipeline.py`` runs the assembly as a sequence of pipelines, each
one stage longer than the last and each ending in a blocking read, so the increments
are real costs rather than overlapping measurements.  That is what showed the AD
layer was not the bottleneck: at four ranks on 27 000 hexahedral P2 elements per rank
the kernel was 10% of an assembly, the parallel triple product 55% and the
essential-dof elimination 25%.

Three changes followed, each of which leaves the assembled matrix **bit-identical**
(``test_assembly`` compares it densely against MFEM's own calls on 1, 2 and 4 ranks):

* the essential-dof elimination is applied as a mask on the local CSR slots before
  the triple product, instead of as a second parallel matrix from
  ``EliminateRowsCols`` afterwards.  The two agree because the prolongation is
  boolean, so the local mask is the pullback of the true-dof one;
  ``HIPPYMFEM_FOLD_ELIMINATION=0`` restores the old path;
* the triple product is formed either by MFEM's fused ``RAP`` or by two sparse
  products with the transpose taken once.  Neither is always faster, so the library
  times both on the first assembly for a pair of spaces and keeps the winner;
  ``HIPPYMFEM_TRIPLE=rap`` or ``split`` overrides;
* the block-diagonal local matrix is consumed by the triple product and never
  reaches the caller, so it is built once and its values overwritten.

Assembly route
--------------

The original route handed each element array back to MFEM through a
``PyBilinearFormIntegrator``.  That costs a Python call per element, measured at
3.0 microseconds, against 0.44 for MFEM's own native integrator doing the whole
job.  The direct route scatters into the CSR structure itself, reusing a sparsity
map built once:

=========================  ================  ================  =========
stage (32k P1 quads)       callback route    direct route      speedup
=========================  ================  ================  =========
scatter into the matrix    3.90 us/elem      **0.13 us/elem**  31x
JAX kernel                 5.47 us/elem      5.47 us/elem      --
full assembly              8.36 us/elem      **5.47 us/elem**  1.5x
=========================  ================  ================  =========

Up to 61x on the scatter, on P1 triangles.

The direct route is faster than MFEM's native baseline because the pattern and the
entry-to-slot map are built once and reused; MFEM rebuilds them on every assembly.
Building them is cheap against an assembly: the pattern costs 0.4 to 0.7 of a warm
assembly (P1 and P2, quadrilaterals and hexahedra), and the first assembly in a fresh
process costs about three, most of which is compiling the kernel.  Even a block
assembled exactly once is therefore no worse off here: cold, one assembly takes 0.18 s
on the direct route against 0.24 s on the integrator route at P1 quadrilaterals, and the
two are within 5 % at P2 hexahedra.  ``HIPPYMFEM_ASSEMBLY=integrator`` is the reference
implementation and the fallback for families the direct scatter does not cover, rather
than an optimization for one-shot assembly.

After this change **the remaining cost is the kernel, not the assembly**, which is
what makes the GPU worth using.

CPU against GPU
---------------

Measured on one NVIDIA L40S against one CPU core's share, double precision
throughout, full assembly including ``P^T A P`` and elimination:

=================  =============  =============  =========
case               CPU us/elem    GPU us/elem    speedup
=================  =============  =============  =========
quad P1            5.47           0.58           9.5x
quad P2            11.7           1.41           8.3x
quad P3            37.9           4.53           8.4x
hex P1             31.6           1.98           15.9x
hex P2             180            12.1           14.9x
=================  =============  =============  =========

The pattern is the useful result: **the speedup grows with the work per element.**
At P1 in 2D an element matrix is 4x4 and the kernel is dominated by overheads that a
GPU does not remove; at P2 in 3D it is 27x27 and the batch is large enough to fill the
device.  Inverse problems tend to live at the expensive end, because the parameter
field is what the mesh has to resolve.

Note that double-precision throughput is what is being used, and it varies by two
orders of magnitude between GPUs: 1.25 TFLOP/s measured on this L40S against 125
TFLOP/s single precision on the same card.  A compute-class card runs fp64 at about
half its fp32 rate, so these speedups are a lower bound for A100/H100-class hardware
and an upper bound for inference-class cards.

What the GPU does not accelerate
--------------------------------

With the default PyMFEM the matrix and every linear solve stay on the host, so the
end-to-end speedup obeys Amdahl's law on the assembly fraction.  How much that costs
is directly visible: over a fixed 274 625-dof problem on one rank, moving the kernels
to the device takes an assembly from 7.72 s to 0.65 s and a reduced-Hessian
application, which is two solves and a few matvecs, from 3.62 s to 3.68 s.

So for a linear forward problem with AMG the solves dominate and the gain is modest;
for a nonlinear forward problem, a high-order discretization, or an MCMC chain, all
of which reassemble constantly, assembly is the larger share.  ``hm.mfem_config()``
reports whether this PyMFEM has CUDA or HIP at all.  With such a build,
``HIPPYMFEM_HYPRE_DEVICE=1`` puts hypre on the device as well, and then the solves are
the part that speeds up most: a 2.1e6-dof Newton step runs on one card.  That is under
:ref:`mfem-device`.

Scaling
-------

Assembly is linear in elements per rank and was measured to 1.6e5 elements per rank
(1.09 s per assembly on P1 tetrahedra, 6.7 us/element).  The per-element cost *falls* with batch size until the
batch is large enough to amortize the fixed overheads, which happens around 1e4
elements per rank on a CPU and later on a GPU, so a GPU needs a larger local
problem to pay for itself.  On the smallest kernel the library has, P1
quadrilaterals, that crossover is the whole story: 0.4x to 1.3x below 2500 elements
per rank against 9.5x at 32400.  See :ref:`smallest-kernel`.

With hypre on the device the picture is strong scaling across cards.  At
64\ :sup:`3` P2 (2.1e6 state dofs) two Newton-CG steps take 27.9 s on one L40S,
14.4 s on two and 9.6 s on four, with the warm Hessian-block assembly at 1.73, 0.67
and 0.39 s: 2.9x and 4.4x from four cards.  What does not scale is the cold first
linearization point, which pays the sparsity patterns and JAX's compilation once.  The
same two steps all on the host take 786 s on one core-bound rank and 179.5 s on four:
one card is worth 28 ranks' worth of that, four cards 19 times four ranks.  At
128\ :sup:`3` (17e6 state dofs) the same four cards take 49.5 s for two steps and the
host 1409 s on four ranks (28x), with the same J and CG count.
`hIPPYlibx <https://github.com/hIPPyMFEM/hippylibx>`_ on the same host, with the same
BoomerAMG settings, takes 402.5 s at 64\ :sup:`3` and 2669 s at 128\ :sup:`3`
(``benchmarks/bench_newton_hippylibx.py``).  At 256\ :sup:`3` (287 M unknowns) on 32 ranks,
32 Blackwell slices take 77 s against 2859 s for the host and 6187 s for hIPPYlibx on 32
cores (37x and 80x), all at 9 CG iterations; :doc:`gpu` has the per-stage tables.

.. _hessian-blocks:

The Hessian blocks: one slot pass per column
--------------------------------------------

A linearization point assembles ``C``, ``W_uu``, ``W_um`` and ``W_mm``, and the
differentiation pass behind them used to be one ``jacfwd(grad(R))`` over every
element dof: 62 forward tangents for a P2/P1 hexahedron, of which a point uses the
``u`` and ``m`` columns, 35.  The pass is now one ``jacfwd`` per column slot
(``GroupKernel._hess_slot``), and ``hess_block``, ``hess_cols`` and ``hess_all`` are all
built from it.  That is what keeps them bit-identical to each other: XLA rounds a
forward pass differently for a different number of tangents (a 35-tangent pass was
measured 4e-16 from the 62-tangent one), and the suite asserts zero.  The Jacobian
pass drops from 62 to 27 tangents by the same change.

A residual that is linear in the state (``is_fwd_linear=True``, which ``solveFwd``
checks) has ``W_uu = p . d2R/du2 = 0``; the block is then not assembled at all, which
removes the largest of the five (the state-state pattern, the size of ``A``) and
leaves the pass with the ``m`` columns: 8 tangents.  ``apply_ij`` returns zero for the
absent block, as it already did for Gauss-Newton.

Measured back to back on L40S cards, two Newton steps, same load, ``J`` to nine digits:

======================  ==================  ========================  ==================
run                     two Newton steps    Hessian blocks warm       card peak
======================  ==================  ========================  ==================
64\ :sup:`3`, 1 card     42.7 -> 29.4 s      6.35 -> 1.81 s            18.0 -> 16.5 GiB
64\ :sup:`3`, 4 cards    17.0 -> 12.9 s      2.04 -> 0.60 s            8.1 -> 5.7 GiB
128\ :sup:`3`, 4 cards   90.0 -> 63.4 s      15.4 -> 4.6 s             22.2 -> 16.6 GiB
128\ :sup:`3`, 2 cards   168.3 -> 116.8 s    29.2 -> 8.6 s             35.4 -> 29.4 GiB
======================  ==================  ========================  ==================

For a residual that is nonlinear in the state (the geothermal model) ``W_uu`` stays and
the pass pushes 35 tangents; the gain is then roughly 62/35 on the differentiation.
After this the blocks are dominated by the scatter and the hypre builds, not by the
differentiation.

.. _pattern-build:

The one-time pattern build
--------------------------

The first assembly of a space pair builds two patterns: the ``ScatterPattern`` (one
row-major key per element-matrix entry, argsorted, the unique keys, the entry-to-slot
map) and the ``TrueDofPattern`` (the true-dof graph from the ldof one and the ghost
exchange: a ``np.unique`` of the keys with its inverse, and the block-major,
diagonal-leading slot order).  At 64\ :sup:`3` hexahedral P2 on one rank those are
1.9e8 and 1.35e8 keys; profiled on the H100 node at the per-rank load of a 400\ :sup:`3`
run, the three sorts in them were 186 s of a 296 s pattern build.  Two changes:

* **The slot order without a sort.**  ``TrueDofPattern`` ordered its slots with a
  four-key ``np.lexsort`` (block, row, diagonal-first, column), 98 s at 6.8e8
  entries.  Its input is already sorted by row and column, so the only moves are the
  off-diagonal entries to the second block and each row's diagonal entry to its front;
  ``_slot_order`` does that in linear passes (bincounts and cumulative counts) and
  gives the identical permutation, which the suite checks against the lexsort on
  random patterns.  At 64\ :sup:`3` the true-dof pattern goes from 36.7 to 23.3 s on
  the host.
* **The two remaining sorts on the device.**  When the kernels are on a GPU, the
  key argsort and the unique-with-inverse go through ``devsort.argsort_keys``: the
  keys are sorted in one device call when JAX's budget holds them (a 1.9e8-key sort is
  3 s that way against 7.9 s on the host), padded to a power of two so that a run
  compiles a handful of shapes; larger arrays are bucketed by key range on the host
  (a radix sort of 16-bit bucket ids) and sorted a bucket at a time at a fixed shape.
  The order among equal keys reaches neither caller, so any valid argsort gives the
  same pattern, which the GPU suite checks array for array against the host build.
  ``HIPPYMFEM_PATTERN_SORT=host`` turns it off.  At 64\ :sup:`3` on one L40S the
  three patterns of a Newton problem (state-state, its true-dof pattern, state-parameter)
  build in 23.2 s instead of 41.9 s.

What the sorts leave is host work: the key construction, the slot scatter, the
bincounts and the ghost exchange, about 9 s of the 23 s at 64\ :sup:`3`.

Measuring it yourself
---------------------

.. code-block:: bash

   HIPPYMFEM_DEVICE=gpu python benchmarks/bench_assembly.py --out results/asm.json

Pitfalls worth knowing about when you do:

* discard the first call, since JAX compiles and the sparsity pattern is built;
* compare CPU and GPU **in one process** so the mesh, the dof numbering and the
  inputs are the same objects;
* XLA's CPU backend uses about two threads for these kernels regardless of
  ``OMP_NUM_THREADS``, so per-rank CPU throughput does not improve with more cores;
  node throughput comes from more MPI ranks;
* use ``mpicxx -showme:incdirs``-style facts rather than assumptions about the
  machine, and record what the numbers were measured on.

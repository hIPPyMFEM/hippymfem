Performance
===========

The assembly numbers on this page are those of ``benchmarks/bench_assembly.py`` on one
NVIDIA L40S and one MPI rank of an AMD EPYC 9334 host, in double precision; running it
gives the same table for your machine.

Where the time goes
-------------------

After the forward and adjoint solves, an inverse problem spends its time assembling:
a Newton-CG run reassembles the Jacobian and the second-order blocks at every
linearization point, and a nonlinear forward solve or an MCMC chain reassembles at
every step.  An assembly has three parts:

#. the **dof gather**: local dof vector to element dof arrays;
#. the **element kernel**: the JAX program that differentiates the density;
#. the **scatter**: element arrays into the parallel matrix, with the essential dofs
   eliminated.

How an assembly is built
------------------------

The element arrays are scattered straight into the CSR structure of the parallel
matrix, through a sparsity pattern and an entry-to-slot map that are built once per pair
of spaces and reused.  Handing each element matrix back to MFEM through a Python
integrator callback instead, which the test suite keeps as its reference
(``hippymfem/test/reference_assembly.py``), costs 3.9 microseconds per element for the
scatter on P1 quadrilaterals and 54 on P2 hexahedra; the direct scatter costs 0.03 and
2.9.

Three more steps are folded into the scatter:

* the essential-dof elimination is a mask on the local CSR slots, not a second parallel
  matrix from ``EliminateRowsCols`` afterwards;
* where the prolongation is boolean (every conforming space without a
  ``DofTransformation``), a rank's rows go directly into hypre's diagonal and
  off-diagonal blocks and the rows of shared dofs to their owner in one exchange, so
  ``P^T A P`` is never formed;
* otherwise ``P^T A P`` is formed as two sparse products with the transpose taken once,
  which on the host takes a half to a sixth of the time of hypre's fused ``RAP`` and gives
  the same matrix.

The assembled matrix is the one MFEM's own calls give: ``test_assembly`` compares the
two densely, for every block and geometry, on 1, 2 and 4 ranks.  After this **the
remaining cost is the kernel, not the assembly**, which is what makes the GPU worth
using.

CPU against GPU
---------------

A full assembly of the Jacobian of ``exp(m) grad u . grad p``, with its elimination:

=================  =============  =============  =========
case               CPU us/elem    GPU us/elem    speedup
=================  =============  =============  =========
quad P1            1.35           0.10           13x
quad P2            4.39           0.38           11x
quad P3            16.5           1.44           11x
hex P1             10.8           0.79           14x
hex P2             91.6           4.58           20x
hex P3             518            26.5           20x
tet P2             11.6           0.92           13x
=================  =============  =============  =========

**The gain is largest where an element does the most work.**  At P1 in 2D an element
matrix is 4x4 and the kernel is dominated by overheads that a GPU does not remove; at P2
in 3D it is 27x27 and the batch fills the device.  Inverse problems tend to live at the
expensive end, because the parameter field is what the mesh has to resolve.

The table is a complete assembly on meshes of a few thousand to a hundred thousand
elements.  On meshes that fill a card (0.03 to 2 million elements, ten element types;
``benchmarks/bench_assembly_sweep.py``, version 0.1.0, November 2026) the element kernel
alone ran 23 to 102 times faster on an L40S and 81 to 548 times faster on an H100 than on
one core of the same node, and 28 to 120 times faster on a GPU of Frontier (one die of
an AMD MI250X), from linear triangles at the low end to cubic hexahedra at the high end.
The card matters: the kernels run 3.1 to 6.9 times faster on an H100 than on an L40S.
At these speeds the kernel is no longer the only cost of an assembly: the scatter into
the hypre matrix takes 9 to 33 % of it on the GPU of Frontier and 29 to 69 % on the H100
(:doc:`gpu`).

The reference of these factors, the same generic kernel on one core, is not a slow one.
For the diffusion operator on quadratic hexahedra with a coefficient given by a
first-order field, the kernel computed the element matrices in 70 to 105 microseconds an
element on a core of three machines, 0.66 to 1.0 times what MFEM's ``DiffusionIntegrator``
took with a ``GridFunctionCoefficient`` and the same quadrature.

On the host a kernel steps through its batch 2 048 elements at a time
(``hm.config.host_batch``), which keeps its intermediates in cache, and XLA uses about
two threads per rank whatever ``OMP_NUM_THREADS`` says: node throughput comes from more
MPI ranks.

What the GPU does not accelerate
--------------------------------

With the default PyMFEM the matrix and every linear solve stay on the host, so the
end-to-end speedup obeys Amdahl's law on the assembly fraction: over a 274 625-dof
problem on one rank, moving the kernels to the device takes an assembly from 7.72 s to
0.65 s and leaves a reduced-Hessian application, which is two solves and a few matvecs,
at 3.6 s.

So for a linear forward problem with AMG the solves dominate and the gain is modest;
for a nonlinear forward problem, a high-order discretization, or an MCMC chain, all
of which reassemble constantly, assembly is the larger share.  ``hm.mfem_config()``
reports whether this PyMFEM has CUDA or HIP at all.  With such a build,
``HIPPYMFEM_HYPRE_DEVICE=1`` puts hypre on the device as well, and then the solves are
the part that speeds up most.  That is under :ref:`mfem-device`.

Scaling
-------

Assembly is linear in the elements of a rank once the batch is large enough to cover
the fixed cost of a kernel launch, which takes more elements on a GPU than on a host
core.  P2 quadrilaterals, microseconds per element:

==================  ========  ========  =========
elements per rank   CPU       GPU       speedup
==================  ========  ========  =========
256                 6.3       8.9       0.7x
1 024               4.6       2.3       2.0x
4 096               4.8       0.63      7.6x
16 384              4.3       0.37      12x
102 400             5.1       0.49      11x
==================  ========  ========  =========

With hypre on the device the picture is strong scaling across cards, for problems that
are large enough: two Newton-CG steps at 128\ :sup:`3` P2 hexahedra (17 million state
dofs) take 34.6 s on two GPUs of Frontier and 4.6 s on 64, while at 64\ :sup:`3` (2.1
million) they take 8.3 s on one and still 3.3 s on 16.  :doc:`gpu` has the tables,
against host ranks and against hIPPYlibx.

What a linearization point assembles
------------------------------------

A linearization point assembles ``C``, ``W_uu``, ``W_um`` and ``W_mm`` from one
forward-mode pass per column slot, so every block comes from the same differentiation
and they are bit-identical however they are requested.  A residual that is linear in
the state (``is_fwd_linear=True``, which ``solveFwd`` checks) has ``W_uu = 0``; the
block is then not assembled at all, which removes the largest of them.  The first
assembly of a pair of spaces also builds their sparsity pattern, once
(:ref:`gpu-memory`).

Measuring it yourself
---------------------

.. code-block:: bash

   HIPPYMFEM_DEVICE=gpu python benchmarks/bench_assembly.py --out results/asm.json

Pitfalls worth knowing about when you do:

* discard the first call, since JAX compiles and the sparsity pattern is built;
* compare CPU and GPU **in one process** so the mesh, the dof numbering and the
  inputs are the same objects;
* record what the numbers were measured on;
* bind a host run to the cores it is meant to use (one rank to one core for a per-core
  time): XLA's threads otherwise use the cores next to it;
* if the first assembly of a large mesh takes minutes where it took seconds, look at the
  machine's memory.  numpy asks the kernel for huge pages for large arrays, and on a
  machine whose memory is full of file cache and fragmented each such allocation can
  stall for tens of seconds: the sparsity patterns of a 64\ :sup:`3` mesh then took ten
  minutes instead of one on such a server.  ``NUMPY_MADVISE_HUGEPAGE=0`` in the
  environment, set before numpy is imported, restores it; the timed stages on a GPU are
  not affected.

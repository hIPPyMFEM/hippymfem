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
400\ :sup:`3`   1.09 B       24 RTX PRO 6000 Blackwell           191 s
==============  ===========  ==================================  =====================

These rows, and the tables above them, were measured in September 2026.  Since then the
assembled matrices and the vectors stay on the card, hypre multiplies with its own
kernel on several ranks and recycles its device memory (`Several GPUs`_, `How the
parallel matrix is built`_), and the same benchmark, with the same cost functional and
CG counts, takes:

==============  ==================================  ===========  =========
mesh            GPUs                                before       now
==============  ==================================  ===========  =========
64\ :sup:`3`    1 L40S                              27.9 s       13.6 s
64\ :sup:`3`    4 L40S                              9.6 s        5.7 s
128\ :sup:`3`   4 L40S                              49.5 s       29.1 s
64\ :sup:`3`    1 H100                              18.5 s       6.1 s
128\ :sup:`3`   1 H100                              109 s        42.9 s
64\ :sup:`3`    1 AMD MI210                         18.7 s       11.2 s
128\ :sup:`3`   4 AMD MI210                         31.9 s       23.6 s
256\ :sup:`3`   8 RTX PRO 6000 Blackwell            140 s        101 s
256\ :sup:`3`   16 RTX PRO 6000 Blackwell           77 s         46.3 s
400\ :sup:`3`   24 RTX PRO 6000 Blackwell           191 s        134 s
==============  ==================================  ===========  =========

The L40S rows of September are from a workstation with four L40S; the new ones are from
a cluster on which four L40S were two on each of two nodes.  On the H100 at
128\ :sup:`3` the forward solve went from 21.4 to 5.6 s, the warm Hessian blocks from 9.6
to 2.4 s, a reduced-Hessian action from 2.0 to 1.6 s and the peak of the card from 68.8
to 58.7 GiB.  At 400\ :sup:`3` the setup up to the synthetic data's forward solve takes
377 instead of 480 s and the first Hessian-block build 85 instead of 116 s, with 37.2
instead of 38.5 GiB per slice.  Against four host ranks of the same library (measured in
September on the workstation), four L40S are now 31x at 64\ :sup:`3` and 48x at
128\ :sup:`3`, and 32 slices against 32 host ranks 62x at 256\ :sup:`3`.  The run on
eight H100 at 256\ :sup:`3` has not been repeated.

The Blackwell cards were split into two 48 GB MIG slices each, one rank per slice.
Doubling them at 256\ :sup:`3` is 1.81x, 90 % of linear.  At 400\ :sup:`3` the two steps
are warm, as every row is, and take 190.9 s with 3 and 9 CG iterations: forward solve
20.1 s, warm Hessian blocks 9.5 s, a reduced-Hessian action 6.3 s, 38.5 GiB per slice.  The
first Hessian-block build carries the JAX compilation and takes 116 s; a run that skips the
separately timed stages (``--newton-only``) pays it inside its first step.  The setup up to
the synthetic data's forward solve takes 480 s.

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
million state dofs per GPU (:ref:`several-gpus` says where the rest goes).

**The speedup depends on the card and on the mesh.**  Element kernels are double
precision unless asked otherwise (:ref:`single-precision`), and so are the figures of
this section.  On meshes with 0.3 to 2.1 million state dofs the kernel speedup
over a host core (one core of a Xeon Platinum 8462Y+) is 17x to 89x on an L40S and 43x to
537x on an H100 (P1 triangles to P3 hexahedra, measured on 2026-10-02 with
``benchmarks/bench_assembly_sweep.py``, the smaller of two runs on the card; in three
dimensions 41x to 89x and 95x to 537x), so the kernels run 2.3 to 6.3 times faster on
the H100 than on the L40S.  Small meshes mislead in both directions.  A kernel call
costs at least about 1.5 ms on a card, so 65 000 P1 triangles show 15x on an H100 where
two million show 43x.  And the host core's time per element is not constant either: P3
hexahedra took 1 025 us per element at 2 744 elements and 606 us at 32 768 (793, 1 029,
659 and 643 us at 512, 2 744, 8 000 and 17 576 elements on another node), so the
2 744-element mesh that this guide used before showed 737x where the larger mesh shows
537x.  The library therefore does not promise a speedup;
:mod:`hippymfem.test.test_gpu` measures one for whatever card is present.  (An earlier version of this guide gave matrix-multiply rates in double
precision for the two cards.  The H100 figure had been taken with JAX's 64-bit mode off,
so it was a single-precision rate, and the single-precision figures were TF32 rates; the
table has been removed.)

**A complete assembly costs little more than its kernel.**  With hypre on the card the
assembled values never leave it: the scatter writes them into an accumulator in JAX's
memory, and they are copied from there into the two blocks of the hypre matrix on the
device (`How the parallel matrix is built`_).  The same cases assemble 18x to 84x faster
on an L40S and 46x to 313x faster on an H100 than on a host core (40x to 84x and 92x to
313x in three dimensions).  Taken apart on an H100 for 32 768 Q2 hexahedra, a matrix of
17 million entries (``benchmarks/bench_assembly_profile.py``): the kernel 9.8 ms, the
step from the element matrices to the hypre matrix 10.9 ms, a complete assembly timed as
a whole 12.7 ms.  What the second step still spends is mostly the upload of the column
indices, which the pattern keeps on the host.  For six of the ten cases a complete
assembly takes 4 to 20 % longer than its kernel; for Q2 and Q3 hexahedra and Q3 and Q4
quadrilaterals, whose matrices have the most entries per element, 1.5 to 2.1 times as
long.

Until October 2026 the values went through the host: copied down as a numpy array, handed
to MFEM there and uploaded again when hypre first used them.  That passage was 61 of the
71 ms of the assembly above (the copy down 32 ms, 136 MB at 4.2 GB/s; the upload about
20 ms), and a complete assembly was 11x to 36x faster than a core on an L40S and 15x to
73x on an H100, whatever the kernel gained.  ``HIPPYMFEM_DEVICE_BRIDGE=0`` restores that
route.  A block whose prolongations are the identity takes the true-dof route as well
when hypre is on a device (``HIPPYMFEM_TDOF_IDENTITY=auto``); before that, one rank
built its matrix through the constructor that copies and splits a row-major CSR on the
host, 171 ms for the same matrix.

**From P2 up, differentiate at the quadrature points.**  ``HIPPYMFEM_HESSIAN=quadrature``
(or ``hm.config.hessian = "quadrature"``) builds each element Hessian block from the
density's second derivatives at the quadrature points, with respect to the fields'
values and gradients, contracted with the basis, where the default pushes one forward
tangent per element dof through the whole element.  The parts of the pointwise Hessian
that are identically zero are skipped, and the blocks agree with the default's to
round-off.  The element kernels on one L40S and hypre on the host, a linearization point
counting its Jacobian, the other blocks, their assembly and the AMG setup:

=====================================  ==========  =====================  =====================
element (default) / quadrature         elements    Jacobian               linearization point
=====================================  ==========  =====================  =====================
Poisson, P1 tetrahedra                 384 000     0.054 / 0.081 s        0.218 / 0.259 s
Poisson, P2 tetrahedra                 82 944      0.064 / 0.040 s        0.179 / 0.154 s
nonlinear density, P2 tetrahedra       82 944      0.124 / 0.064 s        0.387 / 0.257 s
elasticity, P2 tetrahedra              24 576      0.121 / 0.046 s        0.262 / 0.152 s
elasticity, Q2 hexahedra               4 096       0.139 / 0.032 s        0.305 / 0.176 s
=====================================  ==========  =====================  =====================

``auto`` times both routes once per column slot and keeps the faster; here it keeps the
element route for P1 and for the parameter column of the P2 Poisson problem.  The default
stays ``element``: its results are reproducible bit for bit, and on a host it is the
faster route except for vector-valued Jacobians.

.. _single-precision:

Single precision
----------------

Three things can run in single precision, each on its own switch, and none of them
changes what is computed: the state, the adjoint, the gradient and the MAP point come
out as in double precision.

**The element matrices** (``HIPPYMFEM_PRECISION=mixed``, ``hm.config.precision``).
The matrix kernels run in single precision and the vector kernels (residuals, gradients,
matrix-free products) in double.  A single-precision element matrix is wrong by the same
rounding in every element of a mesh of like elements, an error that a solve multiplies by
the condition number, so the Jacobian's element matrices are corrected to act on the
constants as the double-precision ones do, by one or two double-precision tangents per
element, and symmetric blocks are made symmetric to the last bit.  The forward and the
adjoint solve are then refined against double-precision residuals from the vector
kernels: two passes, the iterations of one solve.  With 2.1 million state dofs the state
differed from the double-precision one by 3e-13, the gradient by 3e-13 and a Hessian
action by 4e-9 to 1e-7, with the same Newton and CG counts.  ``fp32`` puts the vector
kernels in single precision too and is for experiments only: the state is then wrong by
1e-6 and the gradient by 4e-5 on a mesh of 16\ :sup:`3` elements, more on a finer one,
and the optimizers stop at that floor.

**The linear solves** (``HIPPYMFEM_HYPRE_SINGLE=/path/to/libHYPRE_single.so``,
``hm.config.hypre_single``).  hypre is compiled for one precision, and MFEM and PyMFEM
use a double-precision one.  ``tools/build_hypre_single.sh <PyMFEM tree> <directory>``
builds the same hypre in single precision, in about three minutes, with the options of
the installed build; the library loads it next to the other one.  The Jacobian of a PDE
problem is then assembled into that library and exists there alone, with its BoomerAMG
hierarchy, and the CG solves with it run there
(:mod:`hippymfem.algorithms.singlesolve`).  A single-precision solve reaches a relative
residual near 1e-5.  The forward and the adjoint solve are therefore refined against
double-precision residuals, which the element kernels compute, to the solver's own
tolerance: three passes for 1e-12, with the iterations of one double-precision solve
in all, the last pass not followed by another evaluation of the residual when the
earlier ones predict that it reaches the goal.  The state then agreed with the
double-precision one to 1e-12.  ``PDEVariationalProblem.SINGLE_REFINE_GOAL = 1e-9``
stops after two passes and one evaluation of the residual, with the state exact to
4e-10: Newton-CG with a tolerance of 1e-6 then took the same steps to the same cost
functional to nine digits, while BFGS run to 1e-8 ended in a line search that found no
decrease, which is why it is not the default.  The incremental solves of a Hessian
action are used as they are, which the
reorthogonalized CG of a Newton step allows (:doc:`optimization`).  It applies when the
three solvers that hold the Jacobian are CG with BoomerAMG and the Jacobian is
symmetric; any other problem keeps its double-precision solves.

**The tolerance of the incremental solves** is the third and the largest: with the
reorthogonalized CG they need 1e-6 where the recurrence needed round-off.

Newton-CG to a relative gradient norm of 1e-6 on the model problem with 2.1 million state
dofs (64\ :sup:`3` Q2 hexahedra), one GPU or MIG instance unless said otherwise.  Every
row but the first took twelve Newton steps and 131 CG iterations and gave the same cost
functional to nine digits; the first took 193 to 210 CG iterations:

.. table::
   :widths: auto

   ==================================================  ========  ========  ===========  ================
   ..                                                  H100      L40S      Blackwell    four Blackwell
                                                                           instance     instances
   ==================================================  ========  ========  ===========  ================
   the library of 2 October 2026                       63.8 s    161.5 s   163.3 s      81.7 s
   reorthogonalized CG, incremental solves to 1e-6     32.0 s    73.3 s    83.8 s       39.0 s
   and hypre's own PCG (:doc:`solvers`)                29.7 s    64.8 s    77.7 s       36.3 s
   and single-precision element matrices               29.1 s    61.0 s    61.7 s       32.5 s
   single-precision solves, double-precision kernels   25.4 s    50.6 s    59.9 s       29.9 s
   single-precision solves and element matrices        24.5 s    46.3 s    45.4 s       26.1 s
   and forward and adjoint solves refined to 1e-9      23.5 s    43.5 s    42.4 s       25.6 s
   ==================================================  ========  ========  ===========  ================

The last row is 2.7, 3.7, 3.9 and 3.2 times faster than the first, the one before it
2.6, 3.5, 3.6 and 3.1 times.  Remeasured on 4 October with the solves that the CG of a
Newton step needs made cheaper (:doc:`optimization`; the median of three solves after
a first one; double precision throughout against single-precision element matrices and
solves):

.. table::
   :widths: auto

   ==================================================  ==================  ==================
   ..                                                  H100                L40S
   ==================================================  ==================  ==================
   the prior's solves to 1e-12 in the preconditioner   29.9 / 25.3 s       \- / 46.1 s
   to 1e-6 (``cg_preconditioner_tolerance``, default)  27.6 / 23.6 s       62.6 / 43.2 s
   and the index arrays on the device                  25.0 / 21.1 s       \- / 39.9 s
   and Hessian actions relaxed (``1e-2``)              \- / 18.0 s         \- / 32.4 s
   Hessian actions relaxed, index arrays uploaded      22.8 / 20.1 s       \- / \-
   ==================================================  ==================  ==================

All with twelve Newton steps and 131 CG iterations in the last solve of each run (a
change of a fraction of a percent in the path can add a thirteenth step to this
problem, :doc:`optimization`).  So on the L40S single precision solved for the MAP
point 1.45 times faster than double precision with the same settings, with all three
of these 1.9 times faster than double precision with the default ones, and 5.0 times
faster than the library of 2 October; on the H100 1.17, 1.5 and 3.5 times.

What single precision itself gives, stage by stage: double
precision throughout against single-precision element matrices and solves, the solves
in both by hypre's PCG, the incremental ones to 1e-6 and to the 1e-5 that single
precision reaches:

.. table::
   :widths: auto

   =========================================  ================  ================  ====================
   ..                                         H100              L40S              Blackwell instance
   =========================================  ================  ================  ====================
   one CG iteration with BoomerAMG            3.69 / 3.19 ms    9.46 / 7.24 ms    9.52 / 6.93 ms
   BoomerAMG setup                            0.069 / 0.075 s   0.156 / 0.156 s   0.151 / 0.124 s
   Jacobian element kernel                    67 / 39 ms        339 / 124 ms      587 / 111 ms
   complete Jacobian assembly                 104 / 86 ms       380 / 165 ms      634 / 146 ms
   forward solve at a new parameter           0.41 / 0.37 s     1.00 / 0.75 s     1.33 / 0.67 s
   adjoint solve                              0.100 / 0.119 s   0.243 / 0.255 s   0.248 / 0.257 s
   blocks of a linearization point            0.27 / 0.23 s     0.39 / 0.29 s     1.06 / 0.26 s
   Hessian action                             0.107 / 0.077 s   0.260 / 0.162 s   0.263 / 0.160 s
   =========================================  ================  ================  ====================

So single precision pays most where the card has least double-precision throughput
(the kernels, five times on a Blackwell instance) and where an iteration is bound by
memory traffic (a CG iteration, 1.16 to 1.37 times), and it does not help a BoomerAMG
setup.  The Hessian action gains most from the solves (1.4 to 1.6 times), because its
two solves need no refinement.  The forward and the adjoint solve gain least from them:
a refinement needs the double-precision residual from the element kernels, which on
the H100 costs as much as seven CG iterations.  The forward solve owes its gain to the
assembly, and the adjoint solve, three passes with two evaluations of the residual, is
a little slower than in double precision.  Refined to 1e-9 the forward solve took 0.34, 0.65
and 0.60 s and the adjoint solve 0.093, 0.192 and 0.191 s, 1.1 to 1.3 times faster than
in double precision.  The first refined adjoint solve of a process also compiles the
kernel of its residual, once (9 to 33 s in the runs at 128\ :sup:`3` below); an adjoint
solve in double precision evaluates no residual and does not pay it.

**Memory.**  A matrix entry is twelve bytes in double precision (value and column index)
and eight in single, so the Jacobian and its hierarchy take two thirds of what they
took, and no double-precision copy of the Jacobian is made at any time.  The card's
memory during that Newton-CG solve, at its peak, with double precision throughout and
with single-precision element matrices and solves (``benchmarks/bench_precision.py
--solves-only``):

.. table::
   :widths: auto

   ==================================  =================  =================  =======================
   ..                                  H100               L40S               busiest of four
                                                                             Blackwell instances
   ==================================  =================  =================  =======================
   the card                            18.9 / 17.5 GiB    18.7 / 17.3 GiB    6.4 / 5.8 GiB
   outside the element kernels' pool   10.4 / 9.0 GiB     10.2 / 8.8 GiB     4.2 / 3.7 GiB
   the pool's largest use              5.4 / 4.4 GiB      4.4 / 4.4 GiB      1.8 / 1.4 GiB
   ==================================  =================  =================  =======================

What lies outside the pool is hypre's matrices, hierarchies and vectors, MFEM and the
CUDA context; a seventh of it goes.  The other blocks of a linearization point stay in
double precision (for a forward problem that is linear in the state these are ``C`` and
``W_um``, 0.4 GB each at this size).  The Jacobian's accumulator does not.  Where the
elements are assembled a chunk at a time, the route of a mesh that does not fit the
card whole, a matrix that goes to the single-precision library is accumulated in single
precision (``HIPPYMFEM_SINGLE_ACCUMULATE``, on): the accumulator, the largest
allocation of an assembly, takes four bytes a nonzero instead of eight, and its values
pass into the matrix as they are, with no rounded copy in between.  Summed term by term
in single precision instead of rounded once, the matrix differs in the last bit of some
entries (a product by 4e-8).

**At a larger size.**  At 128\ :sup:`3` (17.0 million state dofs) on four L40S the same
solve took 222 s in double precision and 141 s with single-precision element matrices
and solves, 1.57 times faster, in the same 13 Newton and 191 CG iterations and to the
same cost in ten digits (111 s with the Hessian actions relaxed, :doc:`optimization`;
before the changes of 4 October 240 s and 162 s).  A Hessian action took 0.71 and
0.42 s, a forward solve at a new parameter 2.26 and 1.41 s.  The busiest card held
39.2 GiB at its peak in double precision and 33.5 GiB in single, 22.2 and 16.5 GiB of it
outside the element kernels' pool.

A run that releases each linearization point before the next
(``release_linearization_on_move``) holds less at its peak.  In the two Newton steps of
``benchmarks/bench_newton_device.py --release-linearization`` at 128\ :sup:`3` the
busiest of four Blackwell instances held 20.4 GiB in double precision and 18.3 GiB with
single-precision element matrices and solves, and the two steps took 28.3 and 13.2 s.
With the accumulator in double precision and a rounded copy of it, as at first, the same
run held 21.9 GiB, more than in double precision: the copy took the element kernels'
arena past the size it had, and the arena grows by a whole region at a time, so that a
few hundred megabytes more can show as gigabytes on the card.

With the memory that single precision frees, the index arrays of the assembly can stay
on the device (``HIPPYMFEM_DEVICE_PATTERN=1``, :ref:`gpu-memory`), which saves their
upload from the CPU at every assembly: at 64\ :sup:`3` on the H100 the forward solve took
0.26 s instead of 0.39 s and the Newton-CG solve 21.1 s instead of 23.6 s (25.0 s
instead of 27.6 s in double precision), on the L40S 39.9 s instead of 43.2 s.  As it
was first, with a four-byte scatter map for every block and a JAX copy of the
Jacobian's column indices, it took the busiest of four Blackwell instances at
128\ :sup:`3` from 18.3 to 26.2 GiB: those 2.3 GiB took the use of the element kernels'
arena to 8.9 of the 9.0 GiB it had, and the arena, which grows by whole regions, took
another 8 GiB.  Now the scatter
map is compact (below) and the single-precision matrices share their column indices
with no copy in JAX, and the device pattern adds 1.0 GiB to the arena's use, which stays
inside it: the busiest instance holds 18.0 GiB with it and without it, against 19.9 GiB
in double precision (``benchmarks/bench_newton_device.py --release-linearization``,
traced with ``fp32_1003/mem_trace.py``).

Two things make every assembly into the single-precision library cheaper, with or
without the device pattern.  Its matrices share one copy of the column indices of their
pattern (``HIPPYMFEM_SINGLE_SHARE_COLUMNS``, on): a new matrix uploads only its row
pointers and values, where it uploaded four more bytes a nonzero from the host (52 ms of
an assembly of 187 ms at 64\ :sup:`3` on the H100), and two that live at once (the
Jacobians of the last and the next point) hold one copy.  And the scatter map that every
fused assembly, in either precision, takes from the host slice by slice is compact
(``HIPPYMFEM_COMPACT_PATTERN``, on): for every element row two base slots, and for
every entry one byte that picks one and adds an offset below 128, which is 1.3 bytes an
entry for quadratic hexahedra instead of 4.  The slots rebuilt on the device are the
same, so the matrices are the same to the last bit on a CPU (on a GPU the scatter's
atomics add in an order of their own anyway).  At 64\ :sup:`3` on the H100 the two took
a forward solve at a new parameter from 0.39 to 0.29 s and the Newton-CG solve from
23.7 to 22.3 s; in double precision the compact map alone took a forward solve from
0.41 to 0.36 s.  In the two Newton steps at 128\ :sup:`3` on four Blackwell instances the
two took 13.2 s to 12.4 s without the device pattern, which gave 12.2 s.

The 1e-5 of the incremental solves is also the accuracy of a Laplace approximation
computed with them, which is more than a posterior needs.  At the MAP point of the model
problem at 64\ :sup:`3` on a Blackwell instance (``doublePassG`` with k = 50 and p = 20,
eigenvalues from 3.2e4 down to 4.5), mixed kernels and single-precision solves against
double precision with the incremental solves at 1e-10: the eigenvalues agreed to
1.0e-5 (the smallest kept to 3.8e-6), the pointwise posterior variance to 5.7e-6
(1.4e-6 rms over the dofs), the traces to 2e-7, and the eigenvectors diagonalized the
double-precision Hessian to 1.1e-6 of the largest eigenvalue.  Double precision with
the incremental solves stopped at 1e-5 gave the same errors: they are those of the
tolerance, not of the arithmetic.  The eigensolver took 26.0 s instead of 52.3 s
(double precision, incremental solves at 1e-8, the default of ``bench_laplace.py``;
36.5 s at 1e-5); the samples, the variances and the traces do not solve with the
Jacobian and took the same 23 s.  So the single-precision solves stay on for the
stages after the MAP point.  Where eigenvalues are wanted to more digits than that,
switch them off (in ``test_uq`` the eigenvectors diagonalized the Hessian to 3e-7 of the
largest eigenvalue instead of 1e-9):

.. code-block:: python

   pde.set_single_solves(False)       # the Jacobian is assembled in double precision again
   model.setPointForHessianEvaluations(x)

The 1e-5 also shows in the iterates on the way.  In the first Newton step of
``bench_newton_device.py``, from the prior mean and with the full Hessian, CG stopped
after three or four iterations in both precisions, and the directional derivative of
the step differed by 0.2 %.  Two steps in, far from the minimum, the costs differed by
6 % (128\ :sup:`3`) and 30 % (64\ :sup:`3`).  The runs to a tolerance agreed, as above,
so compare such runs and not a fixed number of steps.

What to set, for a symmetric problem solved by CG with BoomerAMG:

.. code-block:: bash

   tools/build_hypre_single.sh /path/to/PyMFEM /path/to/hypre_single     # once
   export HIPPYMFEM_HYPRE_SINGLE=/path/to/hypre_single/libHYPRE_single.so
   export HIPPYMFEM_PRECISION=mixed

and a loose tolerance on the two incremental solvers (:doc:`optimization`).  The
single-precision library is used on a host build as well (``tools/build_hypre_single.sh``
on that PyMFEM tree).  There the suites take about as long with it as without
(``test_optimization`` on two ranks 54 s against 48 s); no problem of a size that
matters was timed.  ``test_solvers`` and ``test_device`` check the single-precision
solves against the double-precision ones when the variable is set.

.. _several-gpus:

Several GPUs
------------

Three things cost parallel efficiency on GPUs, and none of them is the launch latency of
a V-cycle on a small subdomain, which is what this guide used to say: one card needs
0.3 ms plus 5.3 ns per state dof for a CG iteration with a BoomerAMG V-cycle (0.49 ms at
36 000 dofs, 11.6 ms at 2.1 million, on half an RTX PRO 6000).  All numbers below are for
the Jacobian of the model problem on MIG instances of that card, hypre 2.32 and CUDA
12.9, measured with ``benchmarks/krylov_anatomy.py`` and the counters of
``tools/gpuprof.c``.  The matrix of these measurements is assembled by MFEM with the
coefficient interpolated at the nodes, and the right-hand side is random.  (Until
2026-10-02 the benchmark read the parameter from a host copy that the device had not
filled, so the coefficient was 1; the iteration times below were measured again with the
coefficient ``exp(m)`` and differ from the earlier ones by 3 % or less, except at
64\ :sup:`3` on two and four instances, where they differ by up to 13 %.)  The Krylov
iterations on different instances do not disturb one another: eight independent copies
of the 64\ :sup:`3` problem needed the same time per iteration together as one alone
(their BoomerAMG setups did disturb one another, see below).

**cuSPARSE on hypre's off-diagonal blocks.**  hypre keeps a parallel matrix as a diagonal
block and an off-diagonal block per rank, and by default multiplies with both through
cuSPARSE.  For a block with few nonzeros the time of that product does not shrink with
the nonzeros, and an off-diagonal block has every row and almost no nonzeros: among the
off-diagonal blocks with 2.2 million rows a product took 0.5 to 5.4 ms, the longest for
the block with the fewest nonzeros.  What cuSPARSE spends the time on was not determined
(the descriptor that hypre creates for every product costs under 0.03 ms per iteration);
the times were the same on an H100.  On two ranks with 2.2 million rows each:

==========================================  ============  ============  ==============
block                                       nonzeros      cuSPARSE      hypre's kernel
==========================================  ============  ============  ==============
diagonal block of the Jacobian              136 million   2.2 ms        2.6 ms
its off-diagonal block                      842 000       0.6-0.9 ms    0.16 ms
an off-diagonal block of an interpolation   701           5.4 ms        under 0.2 ms
==========================================  ============  ============  ==============

One rank has no off-diagonal blocks, so the loss appears at the step from one card to
two.  ``HIPPYMFEM_HYPRE_SPMV`` chooses the kernel (:func:`~hippymfem.common.mfemconfig.set_hypre_spmv`):
``auto``, the default, is hypre's own on more than one rank of a CUDA build and the
vendor's on one, where it is the faster at this size (12 % per iteration here, none on
an H100 at 2 million dofs, 18 % at 8.6 million; at 36 000 dofs hypre's kernel is the
faster on one rank too, 0.31 against 0.49 ms).  It is a run-time switch of hypre and
changes nothing but the time.  Time per CG iteration in ms, cuSPARSE / hypre's kernel:

=========  ==========================  ============================================
instances  64\ :sup:`3` (2.1 M dofs)   2.1 M dofs per instance (mesh)
=========  ==========================  ============================================
1          11.5 / 12.9                 11.5 / 12.9 (64\ :sup:`3`)
2          12.3 / 8.4                  23.4 / 15.6 (81\ :sup:`3`)
4          5.5 / 5.0                   22.7 / 16.0 (102\ :sup:`3`)
8          3.9 / 3.3                   22.2 / 16.4 (128\ :sup:`3`)
16         3.2 / 2.6                   21.0 / 17.6 (161\ :sup:`3`)
=========  ==========================  ============================================

and a whole reduced-Hessian application (``bench_scaling.py --both-kernels``): 0.60 s on
one instance; on two, four, eight and sixteen at 64\ :sup:`3` 0.61, 0.30, 0.21 and 0.17 s
with cuSPARSE against 0.43, 0.26, 0.17 and 0.14 s; at 2.1 million dofs per instance 1.20,
1.09, 1.14 and 1.13 s against 0.79, 0.81, 0.84 and 0.91 s.  Two Newton-CG steps at
128\ :sup:`3` on eight instances took 29.7 s with cuSPARSE and 26.1 s with hypre's
kernel, same cost functional and CG counts (both before the changes described under
`How the parallel matrix is built`_; 18.1 s now).  On full H100 cards the difference is larger, because the diagonal products are
quicker and the off-diagonal ones are not: 16.5 against 6.2 ms per iteration on two cards
with 2.2 million dofs each (4.4 ms on one), and at 64\ :sup:`3` two cards were *slower*
than one with cuSPARSE (8.1 against 4.4 ms) and faster with hypre's kernel (3.8 ms).
hypre's kernel is slower than cuSPARSE on the big diagonal blocks, so a hypre that chose
per block would gain more.  A prototype of that choice (cuSPARSE for blocks with at least
16 nonzeros per row, a change in ``seq_mv/csr_matvec_device.c``) took 14.2 to 16.3 ms per
iteration on two to sixteen instances at 2.1 million dofs each, where hypre's kernel takes
15.7 to 17.7 ms, and 18.8 against 22.4 ms on two H100 at 8.5 million dofs each (no gain
at 2.2 million each, 6.3 against 6.2 ms); it is not part of this library, and one of its
fifteen runs ended in a non-finite vector that did not recur and is not explained.  The
HIP build, which has a later hypre and multiplies through rocSPARSE, shows no such cost:
two MI210 at 2.2 million dofs each took 12.5 ms per iteration against 9.2 ms on one, and
rocSPARSE was faster than hypre's kernel in the three cases measured, which is why
``auto`` leaves HIP builds alone.

**The halo exchange, at every level, through the host.**  An iteration makes 23 to 31
exchanges (one per product with a level matrix or an interpolation matrix, on six to
eight levels), and without a GPU-aware MPI each one packs on the card, copies to the
host, sends, and copies back.  The exchanges were not timed one by one.  With hypre's
kernel an iteration on several instances takes 1.0 to 2.2 ms longer than on one instance
with the same dofs when an instance holds up to 1.1 million dofs, and 2.5 to 5.0 ms
longer at 2.1 million: 40 to 160 us per exchange, the off-diagonal products included.
That is what is left of the weak-scaling loss (66 to 74 % efficiency at 2.1 million dofs
per instance, against the cuSPARSE time on one) and what bounds strong scaling: at 64\ :sup:`3` on sixteen instances an
iteration takes 2.6 ms where a sixteenth of one instance's time is 0.7 ms.  Chebyshev
smoothing (16 iterations instead of 25, each 1.6 times the cost) and a dense solve of a
coarsest level of 500 unknowns changed a solve by 2 % or less at 2.1 million dofs per
instance; ``UCX_RNDV_THRESH=inf`` changed nothing.  So keep a million state dofs or more
on each GPU.

Two builds of hypre make the exchange itself cheaper, and ``tools/rebuild_hypre.sh``
produces either from the hypre source of an existing PyMFEM build, to be preloaded or
copied over the installed library.  Neither is what ``tools/build_pymfem_cuda.sh``
installs by default, and every time in this guide outside this paragraph is that of the
unpatched library; ``HYPRE_PINNED_STAGING=1 tools/build_pymfem_cuda.sh ...`` adds the
second variant at the end of a build and keeps the first library as
``libHYPRE.so.stock``.

* ``--gpu-aware-mpi`` configures hypre with ``HYPRE_WITH_GPU_AWARE_MPI``, so that it
  gives MPI its buffers on the device and a CUDA-aware MPI moves them.  On two H100 of
  one node with Open MPI 4.1.8 built ``--with-cuda`` (``--mca pml ob1 --mca btl
  self,smcuda``, which uses CUDA IPC) an iteration took 5.4 instead of 6.2 ms at 2.2
  million dofs per card and 20.9 instead of 22.4 ms at 8.5 million.  With IPC switched
  off, so that Open MPI stages the buffers through shared memory itself, it took 5.5 and
  21.1 ms: most of the gain is in how the staging is done.  **Not for MIG instances.**
  They cannot use CUDA IPC, and Open MPI's shared-memory transport is slower than the
  UCX one the runs above use: 15.3 against 16.9 ms on two instances, 16.5 against 16.7
  on four, 19.7 against 18.8 on eight, 21.2 against 18.1 on sixteen (2.1 million dofs
  each), and 8.9 against 2.8 ms on sixteen at 64\ :sup:`3`.  The option is fixed when
  hypre is compiled, and an MPI without CUDA support reads a device address as host
  memory, so :func:`~hippymfem.common.mfemconfig.configure_device` asks the loaded hypre
  (:func:`~hippymfem.common.mfemconfig.hypre_gpu_aware_mpi`) and the MPI library
  (:func:`~hippymfem.common.mfemconfig.mpi_gpu_support`) and raises with the reason on
  more than one rank instead of letting the run die inside MPI.  An Open MPI 4 whose UCX
  was built without CUDA answers yes and still needs ``--mca pml ob1``.
* ``--pinned-staging`` keeps hypre's own route through the host and changes its buffers
  (``tools/hypre-2.32.0-pinned-staging.patch``).  hypre allocates two pageable host
  buffers for every exchange and frees them at its end; the patch keeps page-locked
  ones and reuses them.  With the MPI and transport of the runs above, the patch on and
  off in alternation, an iteration took 17.0 instead of 18.1 ms on sixteen instances at
  2.1 million dofs each (-5.7 %), -3.1 % on eight, -3.3 % on two, and -5.1 % on sixteen
  at 64\ :sup:`3`.  ``HYPRE_STAGE_PINNED=0`` switches it off at run time.

**cudaMalloc and cudaFree, in the BoomerAMG setup and around it.**  A hypre built
without Umpire or its own device pool, which is what the build scripts produce, takes
every device array from the driver.  A forward solve at a new parameter on sixteen
instances made 4 963 such calls without a pool, and they took 1.28 s of its 3.21 s.
Recorded one by one (``benchmarks/hypre_pool_trace.py``), a rank makes 2 300 allocations
per forward solve with its Hessian blocks; 89 % of them are under 1 MB and take 6 to
70 us each, and the 256 above 1 MB take 1.7 to 2.3 ms each on sixteen instances (0.2 to
0.3 ms when two processes use an H100), with frees that cost as much again.  A call
takes longer the more processes of a node make them at once: eight independent
single-rank setups at once spent 0.31 s in them where one alone spends 0.11 s.

The library gives hypre a recycling pool through its hook for user allocators
(:func:`~hippymfem.common.mfemconfig.set_hypre_pool`): freed blocks are kept and handed
out again to requests of at most 1.19 times less.  By default
(``HIPPYMFEM_HYPRE_POOL=auto``) it may hold 1 GiB while a setup runs, which serves the
setup's own temporaries, and 512 MB between setups (``HIPPYMFEM_HYPRE_POOL_KEEP``, in
megabytes, and never more than a quarter of the most hypre has had in use), which
carries blocks from the hierarchy a solver gives up to the one it builds next.  When the
pool is full a freed block displaces larger ones, so that it keeps the many small
blocks.  The BoomerAMG setup inside a forward solve, and the forward solve, at 2.1
million dofs per instance (``benchmarks/bench_forward_steps.py``):

=========  ===================  ====================  ===================  ====================
instances  setup, no pool       setup, default pool   forward, no pool     forward, default
=========  ===================  ====================  ===================  ====================
1          0.17 s               0.15 s                1.47 s               1.47 s
2          0.31 s               0.24 s                1.77 s               1.65 s
4          0.46 s               0.27 s                1.94 s               1.74 s
8          0.74 s               0.36 s                2.29 s               1.93 s
16         1.45 s               0.55 s                3.21 s               2.37 s
=========  ===================  ====================  ===================  ====================

On sixteen instances a forward solve took 2.67 s with nothing kept between setups,
2.31 s with 512 MB and 2.26 s with 1 GiB (one job; ``HIPPYMFEM_HYPRE_POOL_KEEP=0`` is
the first).  ``HIPPYMFEM_HYPRE_POOL=<megabytes>`` asks for one limit at all times and
``0`` for no pool.  The pool holds what it keeps, 0.5 GB per rank here,
:meth:`~hippymfem.common.mfemconfig.HyprePool.trim` returns it, and it is returned by
itself when the driver refuses hypre an allocation or the element kernels run out of
device memory.  What is left on sixteen instances is 200 driver calls and 0.5 s per
forward solve: the largest temporaries of the setup (90 allocations of 10 GB in all),
which only a pool of several GB would keep, and MFEM's own arrays, which do not pass
through hypre's allocator.

**What is left outside the Krylov iterations.**  A reduced-Hessian application is its two
incremental solves almost entirely (``benchmarks/bench_hessian_anatomy.py``, which times
the application as the library runs it and then MFEM's solver alone):

============================================  ===========  =============  =====
configuration                                 application  solvers alone  share
============================================  ===========  =============  =====
1 instance, 64\ :sup:`3` (2.1 M dofs)         0.601 s      0.579 s        96 %
16 instances, 161\ :sup:`3` (2.1 M each)      0.912 s      0.879 s        96 %
16 instances, 64\ :sup:`3` (134 k each)       0.139 s      0.130 s        93 %
1 H100, 64\ :sup:`3`                          0.226 s      0.215 s        95 %
============================================  ===========  =============  =====

The shares were 91, 91 and 89 % on the instances while
:class:`~hippymfem.common.parvector.ParVector` did its updates, copies and inner products
with numpy on the host and the pointwise observation operator was applied there: a
vector that a solve had left on the card was copied down and up again around every
incremental solve.  A vector now follows its data.  Once hypre has used it on the device
its arithmetic runs there through MFEM, the essential entries are set there, and the
observation operator is a hypre matrix; ``.array`` remains the way to the host and the
point where the values are brought down.  ``HIPPYMFEM_DEVICE_VECTORS=0`` restores the
numpy route.  Of the 33 ms that remain on sixteen instances, 26 are the prior's
precision with its mass solve.

A forward solve at a new parameter is assembly, a BoomerAMG setup and one CG solve
(``benchmarks/bench_forward_steps.py``; 2.1 million dofs per instance):

==================================  ==========  ============  ========
step                                1 instance  16 instances  1 H100
==================================  ==========  ============  ========
residual, twice                     58 ms       126 ms        26 ms
Jacobian with its symmetry test     976 ms      1 153 ms      327 ms
solve with the BoomerAMG setup      430 ms      1 075 ms      173 ms
the rest                            4 ms        15 ms         4 ms
forward solve                       1.47 s      2.37 s        0.53 s
==================================  ==========  ============  ========

On an instance the element kernel is most of the Jacobian (half a card, double
precision); on the H100 the kernel and its synchronization are 176 ms, and the scatter
with the upload of its map and the upload of the column indices about 110 ms.  Before
the assembled values stayed on the card and the pool kept blocks between setups, the
same forward solve took 3.22 s on one instance, 4.75 s on sixteen and 1.34 s on the H100
(``bench_scaling.py``, which now gives 1.48 and 2.38 s on the instances).

.. _gpu-memory:

How large a problem fits
------------------------

A Newton step holds about 2 kB of device memory per unknown (state, parameter and
adjoint together): the assembled blocks and the AMG hierarchies.  That puts
128\ :sup:`3` (36 M unknowns) on four 45 GB L40S at 17 to 19 GiB per card, on two of them
at 29 to 35 GiB, or on one 80 GB H100 at 69 GiB.  What decides whether a given run fits:

**What the one-time setup leaves behind.**  JAX keeps what its arena has grown to, so
the largest thing that ever ran in it sets its size for the rest of the run.  Without
numba the two sorts of a pattern build run in the arena (`The sparsity patterns`_), and
with one chunk as large as the budget allowed they left it at 16.9 GB on an H100 with
2.1 million state dofs on one rank, where the element kernels need 8.7 GB: the card
stood at 21.0 GB.  The sort now takes chunks of 2\ :sup:`26` keys at most there
(:data:`hippymfem.fem.devsort.MAX_CHUNK_ARENA`), and the same run stands at 12.8 GB with
the same forward solve and a pattern build that is 8 s longer; a MIG instance with the
same dofs went from 12.7 to 8.7 GB, and two Newton-CG steps at 64\ :sup:`3` on one H100
peak at 14.6 GiB.  With numba or CuPy installed nothing changes: the sort-free builder
uses no device memory and CuPy's sort returns its own.

**JAX's share of the card.**  The share (0.45 of the card with hypre on it, above) is a
ceiling, not a reservation: the pool grows as the element kernels need it, and the chunk
planner sizes their batches from the room left under the ceiling, so a lower share means
smaller chunks and a smaller pool, at some cost in time.  Matrix-free linearization points
(``setLinearizationPoint(x, matrix_free=True)``; ``--matrix-free`` in
``benchmarks/bench_newton_device.py``) leave ``C``, ``W_um`` and ``W_mm`` unassembled and
compute their products at every Hessian apply instead.  Two Newton-CG steps at
128\ :sup:`3` on four L40S, the peaks of the card, of what lies outside JAX (hypre's
matrices and hierarchies, MFEM, the CUDA context) and of JAX's pool, with the same cost
functional and CG count in every row:

.. table::
   :widths: auto

   =====================================  ==========  ============  ==========  ================
   ..                                     card peak   outside JAX   JAX pool    two Newton steps
   =====================================  ==========  ============  ==========  ================
   default                                19.1 GiB    10.1 GiB      9.0 GiB     49.1 s
   matrix-free points                     17.2 GiB    8.2 GiB       9.0 GiB     51.3 s
   JAX share 0.20                         15.1 GiB    10.1 GiB      5.0 GiB     52.7 s
   JAX share 0.20, matrix-free points     13.2 GiB    8.2 GiB       5.0 GiB     63.6 s
   =====================================  ==========  ============  ==========  ================

That table is of September 2026.  With the matrices and vectors kept on the card
(2026-10-02, four L40S of a cluster, two per node, peaks measured in the process), the
same four runs take 29.2, 30.3, 33.1 and 48.0 s and peak at 20.0, 19.4, 18.1 and
17.3 GiB (JAX pool 9.0, 9.0, 6.0 and 6.0 GiB): the lower share now buys 1.9 GiB a card
for 13 % more time, matrix-free points 0.6 GiB for 4 %, and both 2.7 GiB for 64 %.

Set ``HIPPYMFEM_GPU_MEM_FRACTION=0.20`` when a run is short of card memory.  Matrix-free
points save a little more, but under the lower share every one of their products runs
in smaller chunks as well, so they are for a run that still does not fit.  ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` is not a substitute: it reports no
budget for the chunk planner to work from, and its card peak was higher.

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

A streamed geometry is copied to the card once per element pass, and how it is copied
decides what such a pass costs.  JAX moves a numpy array through a staging buffer of its
own, at 5 to 11 GB/s on an H100, whose PCIe 5 link moves 55 GB/s from memory locked for
the device.  Where the device bridge can be used (hypre and the kernels on one card), a
group that streams its geometry has it locked once (``cudaHostRegister``) and each
chunk's slice is copied by the CUDA runtime, of the arrays the kernel reads only (the
coordinates ``X`` of the quadrature points not at all where the density does not use
them); ``HIPPYMFEM_PINNED_STREAM=0`` restores JAX's copy.  The element arrays are the
same to the last bit.  At 64\ :sup:`3` on the H100 with the streaming forced
(``HIPPYMFEM_ELEMENT_CHUNK=23000``), the gradient took 0.063 s instead of 0.169 s
(0.027 s with the geometry kept on the card), a forward solve 0.48 s instead of 0.73 s.
At 128\ :sup:`3` on one H100, where the geometry (13.9 GB) has to stream, the forward
solve took 2.86 s instead of 5.51 s, the adjoint solve 1.25 s instead of 2.52 s, the
gradient 0.40 s instead of 1.05 s, and the Newton-CG solve with single-precision solves
197 s instead of 326 s, in the same 13 Newton and 191 CG iterations.

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
either).  XLA's sort runs in JAX's arena and takes chunks of 2\ :sup:`26` keys at most,
so that it does not leave the arena larger than the kernels need (`How large a problem
fits`_; ``HIPPYMFEM_PATTERN_SORT_CHUNK`` pins a size).  ``HIPPYMFEM_PATTERN_TIMING=1``
prints what each phase of a build cost.

How the parallel matrix is built
--------------------------------

The gather, the batched kernel, any ``DofTransformation`` and the scatter all run on the
device, and the ``(ne, nd, nd)`` element-matrix array never reaches the host.  With
hypre on the host, one assembly then costs one host-to-device copy of the local dof
vectors and one device-to-host copy of the assembled CSR values.

With hypre on the same card nothing of that crosses.  JAX and MFEM each own their device
memory and neither can write into the other's, so the library joins them one level down
(:mod:`hippymfem.common.devicebridge`): a JAX array gives the address of its buffer, MFEM
the address of the device copy of a vector or of a matrix block, and the runtime's
``cudaMemcpy`` or ``hipMemcpy`` copies between the two on the device.  MFEM's
constructor for a matrix given as two blocks does not copy them; it aliases their memory
and uploads only what is not yet valid on the device.  So the blocks are created over
host arrays that are never filled, the assembled values are copied into their device
copies from JAX's accumulator, and the constructor uploads the row pointers and nothing
else.  The dof values that the kernels read and the residual and gradient vectors they
produce pass over the same bridge.  What still leaves the card in an assembly: the
values of rows of shared dofs that a rank does not own, which go to their owner in one
``Alltoallv`` (4.8 MB per Jacobian at 2.1 million dofs on sixteen ranks), and an
accumulator that needs boundary-face entries.  What still arrives from the host: the
row pointers, the column indices (0.55 GB per Jacobian at 2.1 million dofs; a matrix of
the single-precision library takes those of the earlier matrices of its pattern
instead, :ref:`single-precision`) and the scatter map, slice by slice, in a compact form
(one byte an entry and two bases an element row: 0.24 GB instead of 0.76 GB,
``HIPPYMFEM_COMPACT_PATTERN``).  ``HIPPYMFEM_DEVICE_PATTERN=1`` keeps the last two on the
device as well: with the full map that was 1.3 GB at that size, and a forward solve went
from 551 to 534 ms on one H100 and from 2.31 to 2.19 s on sixteen MIG instances, where
sixteen ranks upload at once.  It is off by default because of the memory.

The bridge is used when MFEM is on a GPU, the element kernels are on the same card, the
runtime's copy function is found among the libraries the process has loaded, and eight
numbers copied each way come back right; otherwise, and with
``HIPPYMFEM_DEVICE_BRIDGE=0``, the values go through the host as above.  Matrices built
the two ways agree to round-off (the device scatter does not add in a fixed order).

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
error.  A :class:`~hippymfem.common.parvector.ParVector` that hypre has used has its
current values on the device; its ``array`` brings them to the host and is the place to
read or write them from Python.

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
reports.  (The table is of September 2026.  With the present library, on L40S cards of a
cluster, the eigensolver takes 10.1 s at 32\ :sup:`3`, 28.4 s at 64\ :sup:`3` and 175 s
at 128\ :sup:`3`, and the whole workflow at 128\ :sup:`3`, MAP included, 14.4 minutes
instead of 19.5.)  The incremental and prior solves can run at 1e-8 for these stages: the
eigenvalues move by 6e-6 at most and the eigensolver is 1.4x faster.  At 128\ :sup:`3`
the whole workflow, MAP included, runs in under 20 minutes on four L40S.

**Several GPUs: an ensemble instead of a domain decomposition.**  Every one of these
stages is a set of independent solves (140 Hessian applications in the eigensolver, one
or two prior solves per sample or probe).  Dividing the *mesh* over the GPUs makes each
solve faster by the factor of `Several GPUs`_ and no more: at 64\ :sup:`3` that is 2.0x
on four MIG instances and 2.9x on eight for the eigensolver, and nothing at all for the
prior solves, which have 275 000 unknowns.  When the problem fits on one GPU, divide the
*vectors* instead.  Build the problem on ``MPI.COMM_SELF`` on every rank, so that each
GPU holds all of it, and pass the world communicator as ``ensemble``::

   d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, ensemble=MPI.COMM_WORLD)
   post.pointwise_variance(method="MonteCarlo", n=64, ensemble=MPI.COMM_WORLD)
   post.trace(method="Randomized", r=64, ensemble=MPI.COMM_WORLD)

Rank ``r`` applies the operator to the columns ``r, r + size, ...`` and the columns are
exchanged, so every rank ends with the same eigenpairs; the random streams are
partition-independent, so the ranks start from the same ``Omega``, and the Monte Carlo
samples are the ones a single rank would draw (:meth:`~hippymfem.common.random.Random.seek`).
``benchmarks/bench_laplace.py --ensemble`` does all of this.  Measured at 64\ :sup:`3`
on MIG instances of RTX PRO 6000 Blackwell cards, incremental solves to 1e-8:

==========================================  ==========  ===================  ===================  ===============  ===============
after the MAP point                         1 instance  4, domain decomp.    8, domain decomp.    4, ensemble      8, ensemble
==========================================  ==========  ===================  ===================  ===============  ===============
eigensolver (k = 50, p = 20)                62.0 s      30.4 s               21.2 s               17.6 s           9.9 s
64 posterior samples                        5.7 s       3.0 s                2.7 s                2.1 s            0.9 s
pointwise variance, randomized, r = 64      5.3 s       6.0 s                5.3 s                3.7 s            3.6 s
pointwise variance, Monte Carlo, n = 64     4.2 s       2.5 s                2.2 s                1.3 s            0.8 s
traces, r = 64                              6.3 s       6.4 s                5.8 s                3.2 s            2.7 s
all of them                                 83.5 s      48.2 s               37.2 s               27.8 s           17.9 s
==========================================  ==========  ===================  ===================  ===============  ===============

The eigenvalues of the ensemble agree with those of one instance to 2e-6 (the MAP
points of two runs differ by that much).  At 32\ :sup:`3` the domain decomposition on
four instances is slower than one instance for these stages (18.1 against 16.2 s); the
ensemble takes 7.8 s.  What an ensemble does not speed up: the orthogonalizations, which
are sequential in the columns and which every rank repeats (1.4 s of the eigensolver
above, most of the randomized variance and of the trace), and the MAP point, whose CG
iterations depend on one another.  Every rank also has to set the problem up and hold
it, so the mesh has to fit on one GPU, and a run that computes the MAP point with a
domain decomposition and then switches to an ensemble sets the problem up twice.

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
L40S row.  (The table is of September 2026.  With the matrices and vectors kept on the
card, which works through ``hipMemcpy`` on this build as it does through ``cudaMemcpy``
on the NVIDIA ones, one MI210 takes 11.2 s at :math:`64^{3}` and four take 23.6 s at
:math:`128^{3}`, with an in-process peak of 14.4 and 19.9 GiB per card where it was 22.7
and 28.4; ``test_device`` passes on one and two ranks.  The recycling pool for hypre's
memory is for CUDA builds only.)  :math:`128^{3}` does not fit on one 64 GB card but runs on two: 65.9 s,
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

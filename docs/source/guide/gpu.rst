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

.. _gpu-recommended:

Recommended settings
--------------------

What to set, with the sections that have the measurements.  The numbers are for the
model problem of this guide (``exp(m)`` diffusion on second-order hexahedra, a
BiLaplacian prior, pointwise observations), its MAP point computed by Newton-CG to a
relative gradient norm of 1e-6.  ``applications/precision/model_subsurf_single.py`` does
all of it.

**Profiles.**  ``HIPPYMFEM_PROFILE=gpu``, set before ``import hippymfem``, takes the element
Hessians at the quadrature points and, where ``HIPPYMFEM_HYPRE_SINGLE`` names a
single-precision hypre, the Jacobians and their solves in single precision with the chunk
planner set for single-precision tangents.  ``HIPPYMFEM_PROFILE=gpu-large`` adds the
solver settings of the weak-scaling runs on Frontier (BoomerAMG with l1-Jacobi weight 1.8
and six interpolation entries a row; in the single-precision solves the p-multigrid over
up to seven meshes of a uniformly refined mesh, with its small levels in host memory; the
prior's cycles over the meshes).  A profile fills in only what the environment leaves
unset, and ``hm.config.show()`` marks what it set; ``hm.config.use_profile(name)`` applies
the runtime part after import.  The tolerances and the Newton-CG parameters below remain
the script's, as does a lumped prior mass, which changes the discrete problem.  The
compact geometry of affine elements, and with a single-precision hypre the row sums kept,
are on without a profile; ``gpu-large`` also puts the blocks of a linearization point in
single precision, whose identities then hold to single precision only.

**The MAP point.**

* The defaults need no change: the CG of a Newton step keeps its residuals orthogonal,
  CG with a hypre preconditioner runs in hypre's own PCG, and the prior's solves that
  precondition that CG stop at 1e-6 (:doc:`optimization`, :doc:`solvers`).
* Stop the two incremental solvers at 1e-6 and leave the forward and the adjoint solver
  at their tolerance.  With the orthogonalized residuals the Newton and CG counts stay
  those of incremental solves to 1e-12, and at 64\ :sup:`3` on an H100 the solve took
  32.0 s instead of the 63.8 s of the recurrence with incremental solves to 1e-12.
  1e-4 is too loose: one of three GPUs then took a thirteenth Newton step.
* Set ``HIPPYMFEM_PRECISION=mixed`` together with ``HIPPYMFEM_HYPRE_SINGLE``, the path of
  a single-precision build of hypre (:ref:`hypre-single-install`).  The Newton and CG
  counts and the cost functional stay those of double precision; at 64\ :sup:`3` the
  solve was 1.29 times faster on an H100, 1.51 times on an L40S and 1.81 times on a
  Blackwell instance, and at 128\ :sup:`3` on four L40S 1.57 times, with 11 % less memory
  on the busiest card (:ref:`single-precision`).  The single-precision solves need a
  Jacobian solved by CG with BoomerAMG where it is symmetric, or by GMRES with BoomerAMG
  where it is not (its transpose is then assembled in single precision too, for the
  adjoint); other solvers keep their double-precision solves.
* On a GPU, ``HIPPYMFEM_SINGLE_AMG="relax=7,pmax=6"`` (Jacobi relaxation and six
  interpolation entries a row in the BoomerAMG of the single-precision solves) took the
  Newton-CG solve at 64\ :sup:`3` from 38.5 to 29.0 s on a Blackwell instance and from
  19.8 to 15.7 s on an H100, with the same Newton and CG counts, also on the problems with
  ten times more observations and with noise ten times smaller.  It is for problems like
  this one only, first- or second-order hexahedra on a regular mesh with a moderate
  contrast in the coefficient: on quadratic tetrahedra, a stretched mesh, an anisotropic
  coefficient or a strong contrast the solves fail, with an error.
  ``"relax=16,cheby_order=1,pmax=6"`` (Chebyshev relaxation of order one) converged on
  all of these and has half the gain here (:ref:`single-precision`).

**The Laplace approximation.**  Keep the single-precision solves for the stages after the
MAP point.  Their incremental solves stop near 1e-5, and the eigenpairs come out to about
that: at 64\ :sup:`3` on a Blackwell instance (``doublePassG``, k = 50, p = 20, eigenvalues
from 3.2e4 down to 4.5) the eigenvalues agreed with those of double-precision solves to
1e-10 within 1.0e-5, the smallest kept within 3.8e-6, and the pointwise posterior variance
within 5.7e-6, three orders of magnitude below what a posterior needs; the eigensolver took
26.0 s instead of 52.3 s with double-precision solves to 1e-8 (:ref:`single-precision`).
Where eigenvalues are wanted to more digits than that, switch the single-precision solves
off after the MAP point and run the incremental solves to about 1e-8 (below).  On several
GPUs, if the problem fits on one, run these stages as an ensemble: at 64\ :sup:`3` the
eigensolver took 15.1 s on four MIG instances as an ensemble and 28.1 s as a domain
decomposition (`The Laplace approximation on the cards`_).

**Several GPUs.**  Start the ranks through ``tools/mpirun_pinned.sh``, one card or MIG
instance each.  Two things follow the rank count by themselves: the share of a card that
JAX may use is divided among the ranks on it, and on more than one rank of a CUDA build
hypre multiplies with its own kernel instead of cuSPARSE.  Keep a million state dofs or
more on each GPU: below that, the halo exchanges through the host set the time of a
Krylov iteration (:ref:`several-gpus`).  The settings above stay as they are.

**Short of memory**, in this order: single precision as above, and
``release_linearization_on_move=True`` for a line-search Newton-CG, which drops a
linearization point before the next one is assembled (with both, the Newton-CG run at
128\ :sup:`3`, 17.0 million state dofs, peaked at 59,785 MiB on one 80 GB H100, and at
65,683 MiB in double precision with the release); ``HIPPYMFEM_GPU_MEM_FRACTION=0.20``,
which saved 2.8 GiB a card for 14 % more time in two Newton steps at 128\ :sup:`3` on four
L40S; matrix-free linearization points last (:ref:`gpu-memory`).

.. code-block:: python

   # HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 HIPPYMFEM_PRECISION=mixed
   # HIPPYMFEM_HYPRE_SINGLE=/path/to/libHYPRE_single.so
   pde.set_solvers(hm.auto_solver, Vu, comm, max_direct=0, rel_tolerance=1e-12)
   for name in ("solver_fwd_inc", "solver_adj_inc"):
       getattr(pde, name).parameters["rel_tolerance"] = 1e-6
   x = hm.ReducedSpaceNewtonCG(model, params).solve([None, prior.mean.copy(), None])

   model.setPointForHessianEvaluations(x)   # the Laplace approximation: single-precision solves

   # only where eigenvalues are wanted to more than five digits:
   # pde.set_single_solves(False)
   # for name in ("solver_fwd_inc", "solver_adj_inc"):
   #     getattr(pde, name).parameters["rel_tolerance"] = 1e-8
   # model.setPointForHessianEvaluations(x)

``max_direct=0`` keeps :func:`~hippymfem.algorithms.linSolvers.auto_solver` from a direct
solve on a small mesh, which has no single-precision counterpart.  Measured and left
opt-in: the index arrays of the assembly kept on the device
(``HIPPYMFEM_DEVICE_PATTERN=1``, memory permitting), Hessian actions relaxed as the CG
converges (``cg_hessian_relaxation``, which lost on a more informative problem), Jacobi
relaxation in MFEM's BoomerAMG of the double-precision solves (``HIPPYMFEM_AMG_RELAX=7``,
which took one problem from 329 to 364 CG iterations), and a refinement of the forward
and the adjoint solve that stops at 1e-9 while Newton-CG runs (``single_refine_goal``:
at 128\ :sup:`3` on an H100 187 s instead of 199 s while the iteration kept its path, and
223 s the one time in five that it did not).

.. _mfem-device:

What to expect
--------------

**Moving the kernels alone is worth little; moving the solves is worth a lot.**  The
benchmark problem below at 40\ :sup:`3` (531 441 state dofs) on one rank and one L40S,
with the same cost functional and CG counts in every configuration:

======================  ==============  =============  =============  ================
where the work runs     Hessian blocks  forward solve  Hessian apply  two Newton steps
======================  ==============  =============  =============  ================
all on the host         1.73 s          10.9 s         8.90 s         97.3 s
GPU kernels only        0.24 s          5.6 s          8.35 s         73.9 s
GPU kernels and hypre   0.24 s          0.45 s         0.16 s         **3.2 s**
======================  ==============  =============  =============  ================

The kernels alone are worth 1.3x on the two steps, the kernels and the solves 31x.

**Two Newton-CG steps of the benchmark problem** (P2 hexahedra for the state and the
adjoint, P1 for the parameter, CG with BoomerAMG for every solve;
``benchmarks/bench_newton_device.py``, and ``benchmarks/bench_newton_hippylibx.py`` for
the hIPPYlibx rows):

==============  =========  ===========================  =======  =======  =======  ==========
mesh            unknowns   where it runs                forward  Hessian  Hessian  two Newton
                                                        solve    blocks   apply    steps
==============  =========  ===========================  =======  =======  =======  ==========
64\ :sup:`3`    4.6 M      1 L40S                       0.94 s   0.29 s   0.49 s   10.9 s
64\ :sup:`3`    4.6 M      1 H100                       0.40 s   0.16 s   0.20 s   4.7 s
64\ :sup:`3`    4.6 M      1 AMD MI210                  0.84 s   0.23 s   0.38 s   9.2 s
64\ :sup:`3`    4.6 M      4 L40S                       0.39 s   0.09 s   0.22 s   4.9 s
64\ :sup:`3`    4.6 M      2 Blackwell cards            0.52 s   0.25 s   0.26 s   6.2 s
64\ :sup:`3`    4.6 M      4 host ranks                 12.5 s   1.5 s    10.9 s   179.5 s
64\ :sup:`3`    4.6 M      hIPPYlibx, 4 host ranks      29.4 s   8.7 s    13.8 s   402.5 s
128\ :sup:`3`   36 M       1 H100                       3.5 s    1.5 s    1.3 s    29.4 s
128\ :sup:`3`   36 M       4 L40S                       2.3 s    0.56 s   1.4 s    23.4 s
128\ :sup:`3`   36 M       4 AMD MI210                  2.1 s    0.48 s   1.1 s    19.6 s
128\ :sup:`3`   36 M       2 Blackwell cards            2.9 s    1.9 s    1.4 s    28.1 s
128\ :sup:`3`   36 M       8 Blackwell cards            1.2 s    0.60 s   0.45 s   10.9 s
128\ :sup:`3`   36 M       4 host ranks                 108.5 s  11.9 s   101.3 s  1409 s
128\ :sup:`3`   36 M       hIPPYlibx, 4 host ranks      241.0 s  71.4 s   140.7 s  2669 s
256\ :sup:`3`   287 M      8 Blackwell cards            8.7 s    4.7 s    2.9 s    82.6 s
256\ :sup:`3`   287 M      16 Blackwell cards           3.6 s    2.0 s    1.5 s    38.5 s
256\ :sup:`3`   287 M      32 host ranks                --       --       --       2859 s
256\ :sup:`3`   287 M      hIPPYlibx, 32 host ranks     423.9 s  102.4 s  378.9 s  6187 s
400\ :sup:`3`   1.09 B     24 Blackwell cards           13.1 s   6.4 s    4.8 s    134 s
==============  =========  ===========================  =======  =======  =======  ==========

The GPU rows were measured in October 2026 on a cluster: the four L40S are two on each
of two nodes, and the Blackwell cards (RTX PRO 6000) were split into two 48 GB MIG slices
each, one rank per slice, so two cards are four ranks and 16 cards are 32.  The host and
hIPPYlibx rows were measured in September 2026 on one node with two AMD EPYC 9334, with
as many ranks as the GPU rows beside them: they are a device swap at a fixed rank count,
not the node's best host time.  Against four host ranks of the same library, the four
slices of two Blackwell cards are 29 and 50 times faster at 64\ :sup:`3` and
128\ :sup:`3` and four L40S 37 and 60 times; at 256\ :sup:`3` the 16 cards are 74 times
faster than 32 host ranks.  The cost functional is the same to seven digits and the CG
counts are the same.

The hIPPYlibx rows are the same problem (mesh, spaces, PDE, data model, prior,
Newton-CG settings) solved by `hIPPYlibx <https://github.com/hIPPyMFEM/hippylibx>`_ on
dolfinx 0.10, with PETSc's CG and BoomerAMG given the same BoomerAMG options.  The two
libraries draw different random data, so their CG counts differ at the two smaller
sizes (11 and 6 against 8): two Blackwell cards are 65 and 95 times faster as measured,
and 58 and 105 times when hIPPYlibx is charged the same CG counts (four L40S 82 and 114
times as measured).  At 256\ :sup:`3` every run takes 9 CG iterations, and the GPUs are
161 times faster.

Every row is warm.  The first linearization point of a run pays the sparsity patterns
and JAX's compilation once (5 s at 64\ :sup:`3` on one card, 85 s at 400\ :sup:`3`,
where the setup up to the first forward solve takes another 377 s).

.. _smallest-kernel:

Where the device loses
----------------------

**With hypre on the host, the solves do not move.**  A reduced-Hessian application is
two linear solves and a few matvecs that cost the same whatever the kernels run on: on
a 274 625-dof problem, GPU kernels take the assembly of a linearization point from 0.88 s
to 0.13 s and leave the Hessian application at 4.1 s.

**There has to be enough arithmetic per element.**  The full-assembly speedup grows
with the work an element kernel does, from 11 to 13x on quadrilaterals to 20x on P2 and
P3 hexahedra on an L40S (:doc:`performance`).

**There has to be enough of the mesh on each rank** to cover the fixed cost of a kernel
launch.  On P2 quadrilaterals the device is slower than the host at 256 elements per
rank, twice as fast at 1 024 and 8 to 13 times faster from 4 096.  Above about
:math:`10^4` elements per rank the device wins on every case measured.

**The speedup depends on the card.**  Element kernels are double precision unless asked
otherwise (:ref:`single-precision`), and they run 2.3 to 6.3 times faster on an H100 than
on an L40S.  The library therefore does not promise a speedup;
:mod:`hippymfem.test.test_gpu` measures one for whatever card is present.

**From P2 up, differentiate at the quadrature points.**  ``HIPPYMFEM_HESSIAN=quadrature``
(or ``hm.config.hessian = "quadrature"``) builds each element Hessian block from the
density's second derivatives at the quadrature points, contracted with the basis, where
the default pushes one forward tangent per element dof through the whole element.  The
blocks agree with the default's to round-off, and on a GPU a Jacobian assembles 1.6 to
4.4 times faster for P2 and Q2 elements (and slower for P1).  ``auto`` times both routes
once per column slot and keeps the faster.  The default stays ``element``: its results
are reproducible bit for bit, and on a host it is the faster route except for
vector-valued Jacobians.

.. _single-precision:

Single precision
----------------

Two things can run in single precision, each on its own switch, and a third switch, a
loose tolerance on the incremental solves, is what makes the second usable.  None of
the three changes what is computed: the state, the adjoint, the gradient and the MAP
point come out as in double precision.  :ref:`gpu-recommended` says what to set.

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
action by 4e-9 to 1e-7, with the same Newton and CG counts.  Since a symmetrized matrix
passes any symmetry probe, ``symmetric_jacobian="auto"`` decides here on the first
Jacobian and keeps the verdict: a residual that is symmetric at some parameters and not
at others has to declare ``symmetric_jacobian=False``.  A refined solve that ends far
above its goal, because the matrix is not the operator of its residual or because the
parameter makes the Jacobian too ill-conditioned for single precision (a line search
may propose such a point), is solved again with the Jacobian in double precision, with
a warning.  That second Jacobian and its hierarchy are on the card beside the
single-precision ones, three times their size, and a problem sized for the mixed mode
may have no room for them: ``PDEVariationalProblem.SINGLE_GIVE_UP = "raise"`` makes
such a *forward* solve raise instead, which a line search takes for a failed trial and
answers with a shorter step (an adjoint solve, at a point the line search accepted, is
solved in double precision either way; a forward solve outside a line search then
fails where it would have been solved, which is why ``"double"`` is the default).
``fp32`` puts the vector
kernels in single precision too and is for experiments only: the state is then wrong by
1e-6 and the gradient by 4e-5 on a mesh of 16\ :sup:`3` elements, more on a finer one,
and the optimizers stop at that floor.

**The linear solves** (``HIPPYMFEM_HYPRE_SINGLE=/path/to/libHYPRE_single.so``,
``hm.config.hypre_single``).  hypre is compiled for one precision, and MFEM and PyMFEM
use a double-precision one.  ``tools/build_hypre_single.sh <PyMFEM tree> <directory>``
builds the same hypre in single precision, in about three minutes, with the options of
the installed build (:ref:`hypre-single-install`); the library loads it next to the
other one.  The Jacobian of a PDE problem is then assembled into that library and exists
there alone, with its BoomerAMG hierarchy, and the CG solves with it run there
(:mod:`hippymfem.algorithms.singlesolve`).  A single-precision solve reaches a relative
residual near 1e-5, for a right-hand side of any size: hypre's PCG works with squares
that leave the range of single precision where the right-hand side is far from one
(it breaks off at a size of 1e-16), and such a system is solved for the right-hand side
scaled by a power of two.  The forward and the adjoint solve are therefore refined against
double-precision residuals, which the element kernels compute, to the solver's own
tolerance: three passes for 1e-12, with the iterations of one double-precision solve
in all, the last pass not followed by another evaluation of the residual when the
earlier ones predict that it reaches the goal.  The state then agreed with the
double-precision one to 1e-12.  ``PDEVariationalProblem.SINGLE_REFINE_GOAL = 1e-9``
stops after two passes and one evaluation of the residual, with the state exact to
4e-10: Newton-CG with a tolerance of 1e-6 then took the same steps to the same cost
functional to nine digits, while BFGS run to 1e-8 ended in a line search that found no
decrease, which is why it is not the problem's default.  Newton-CG can set it while it
runs (``single_refine_goal``, never above 1e3 times the square of its own tolerance,
which leaves a run to 1e-8 as it is), and that is off by default as well.  With 1e-9 the
solve at 64\ :sup:`3` took 3 to 5 % less time on an H100, an L40S and Blackwell instances,
with the same twelve steps and 131 CG iterations (20.2 s instead of 20.8 s on the H100).
At 128\ :sup:`3` it saves 6 % while the iteration keeps its path, 187 s instead of
199 s on an H100 with thirteen steps and 191 CG iterations, but it did not always keep
it: in one solve of five the last line search backtracked and a fourteenth step followed
(223 s, 232 CG iterations), and on four L40S one solve took 178 s and the next 139 s,
where every solve with full refinement took 143 s.  With 1e-10 one solve of three went
the same way.  The incremental solves of a Hessian action are used as they are, which
the reorthogonalized CG of a Newton step allows (:doc:`optimization`).  All of this
applies when the three solvers that hold the Jacobian are CG with BoomerAMG and the
Jacobian is symmetric; any other problem keeps its double-precision solves.  It runs with
CUDA and with HIP builds (`AMD cards`_ has the measurements on MI250X), and with global
indices of either width: the library of a stack built ``HYPRE_MIXEDINT=1`` has four-byte
values, four-byte indices within a rank and eight-byte ones across ranks, and holds a
Jacobian of more than 2 147 483 647 rows; one with 32-bit global indices refuses such a
matrix.

.. _single-row-sums:

**The sums of the Jacobian's rows** (``HIPPYMFEM_SINGLE_ROW_SUMS=1``).  A row of a
discretized diffusion operator sums to zero away from the boundary, and what the operator
returns for a smooth function is what is left when entries thousands to millions of times
larger cancel.  The element matrices of the mixed precision are therefore made exact on
the constants (:meth:`~hippymfem.fem.kernel.GroupKernel.consistent_block`).  The matrix
of the single-precision library then rounds every entry to 24 bits, and the sum of a row
is off again, by about 3e-8 of its diagonal.  On a mesh of like elements with a constant
coefficient every like row is rounded alike: the matrix has gained a term of order zero
with one sign over the whole domain, and a solve with it is off, relative to a smooth
solution, by the rounding unit times the square of the number of nodes across the domain.
The refined solves remove that (their residuals are the double-precision operator's),
at a price that grows with the mesh, since every pass leaves that fraction of what it
corrected; the incremental solves of a Hessian action, which are not refined, keep it.
A prior mean is often such a coefficient, and a Newton iteration starts there.

With the switch on, the Jacobian is accumulated in double precision, what the rounding of
its values takes from each row's sum is kept as a vector ``c`` of the library
(:class:`~hippymfem.algorithms.singlesolve.RowDefect`: the rounding residuals are put in
the matrix, multiplied by the vector of ones, and the values written after), and the
product of hypre's PCG adds ``c x`` entry by entry
(:meth:`~hippymfem.algorithms.singlesolve._Library.pcg`): the iteration then solves with
a matrix that has the row sums of the double-precision values.  On the MI250X cards of
Frontier, with second-order hexahedra at the prior mean of the model problem, the
residual that the second pass of a refined forward solve leaves went from 3.7e-10 of the
first to 1e-10 at 120\ :sup:`3` elements and from 2.1e-9 to 1e-10 at 240\ :sup:`3`, where
1e-10 is what two passes of the solver give, and a sample of the prior, whose elements
are rounded each in its own way, was there already.  Without it a refined solve took
three passes up to 240\ :sup:`3` and 4.3 on average at 1920\ :sup:`3`, and two Newton-CG
steps ended at a cost 0.5 to 1.7 times that of double precision, depending on the size;
with it three passes at every size, and the cost of double precision to within about
2 %, which is what the tolerance of the incremental solves leaves (`AMD cards`_).  It costs
0.1 s an assembly at 14 million unknowns a card: the accumulator in double precision for
the duration of the assembly (3.3 GiB more), two passes over it and one product.  Off by
default.

One thing about that card's compiler: in a single compiled program it takes a conversion
to single precision and back for the identity, so ``x - x.astype(float32).astype(float64)``
is zero there.  The residual of the rounding is computed in two programs, the second given
the rounded values.

The BoomerAMG of that library takes MFEM's defaults for a device, then the solver's own
settings, its ``amg_options`` included (``HIPPYMFEM_AMG``, which a hierarchy in double
precision takes the same way), and ``HIPPYMFEM_SINGLE_AMG`` sets any of hypre's BoomerAMG
options by name for the single-precision solves alone
(:data:`hippymfem.algorithms.singlesolve.AMG_OPTIONS`).  ``"relax=7,pmax=6"``, Jacobi
relaxation (hypre's type 7) instead of l1-Jacobi (18) and six interpolation entries a
row instead of four, is the one worth setting on the model problem.  l1-Jacobi divides
by the sum of the magnitudes of a row, a few times the diagonal for quadratic elements,
and smooths that much less.  On the Jacobian at 64\ :sup:`3` (Blackwell instance, the
right-hand sides of incremental solves, solved to 1e-5) it took the iterations from 11
to 6 and a solve from 76 to 48 ms with the same setup time, at the MAP point, at the
prior mean and at the true parameter alike, and Newton-CG to 1e-6 from 38.5 to 29.0 s on
a Blackwell instance and from 19.8 to 15.7 s on an H100, with the same twelve Newton steps
and 131 CG iterations.  It kept the steps and CG iterations of the problem with ten times
more observations (68.3 to 52.0 s, 15 Newton steps and 273 CG iterations) and of the one
with noise ten times smaller, whose CG runs into its cap of 50 iterations in most late
steps (113.1 to 84.7 s, 18 and 488).

It is not the default, and it is not for every problem.  Plain Jacobi smooths only where
the largest eigenvalue of D\ :sup:`-1`\ A is below 2: it is 1.50 for trilinear and 1.71
for triquadratic hexahedra on the model problem's mesh and 1.98 for linear tetrahedra,
but 2.47 for cubic hexahedra, 2.38 for quadratic tetrahedra, 2.27 on a mesh stretched by
ten, 3.69 with an anisotropy of 100 and 6.85 with a parameter three times as large (a
contrast of 1e14).  Above 2 the preconditioner is not positive definite.  The cubic
hexahedra converged all the same (one system in 24 iterations instead of 21, Newton-CG
at 32\ :sup:`3` on an H100 in 13.8 s instead of 16.3 s).  In the other cases hypre's PCG
stops after two or three iterations with a residual of order one: the solver then raises
an error (``the preconditioner is not positive definite on this matrix``), and a forward
or an adjoint solve is solved with the Jacobian in double precision instead, with a
warning that says the same.

``"relax=16,cheby_order=1,pmax=6"`` is the setting that is safe on all of them.
Chebyshev relaxation of order one is a Jacobi whose weight hypre takes from an estimate
of that eigenvalue on every level.  On one small system of each kind (9,000 to 36,000
rows on a host, solved to 1e-5) it converged in every case above and never in more
iterations than l1-Jacobi: 13 against 19 on the model problem's hexahedra, 15 against 21
on cubic hexahedra, 17 against 22 on quadratic tetrahedra, 18 against 37 with the
parameter three times as large, 72 against 105 on the stretched mesh.  On a GPU its
iteration costs half as much again (4.85 ms against 3.26 ms at 64\ :sup:`3` on an H100),
so it pays where it removes more than a third of the iterations.  On the model problem
it does: six iterations instead of eleven, a solve in 29.1 ms against 35.9 ms with the
defaults and 22.9 ms with plain Jacobi, and in one series of Newton-CG runs on an H100
18.0 s against 20.0 s and 16.0 s.

Of the other settings tried, Chebyshev relaxation of order two (hypre's default order)
halved the iterations at more than twice their cost, a strength threshold of 0.5 and
fewer interpolation entries cost iterations, HMIS coarsening set up on the host (2.2 s),
aggressive coarsening cost iterations, and its extended+i interpolation does not run on
a device (it crashes).
In MFEM's BoomerAMG, which PyMFEM lets set the relaxation but not the interpolation
entries, ``HIPPYMFEM_AMG_RELAX=7`` (read by every AMG solver, the prior's too) took
Newton-CG in double precision throughout from 73.0 to 56.1 s on a Blackwell instance with
the same counts, and from 199.8 to 148.5 s with noise ten times smaller; with ten times
more observations from 158.6 to 129.6 s, but with 364 CG iterations instead of 329 in the
same eighteen Newton steps.

``HIPPYMFEM_AMG`` (the solvers' parameter ``amg_options``) sets the rest in MFEM's
BoomerAMG as well, by the names of ``HIPPYMFEM_SINGLE_AMG`` and a few more
(:data:`hippymfem.common.mfemconfig.BOOMERAMG_SETTERS`).  MFEM's class converts to
hypre's own handle of the preconditioner, a shared MFEM library exports that conversion,
and through the handle hypre's C interface takes every option
(:func:`~hippymfem.common.mfemconfig.set_boomeramg_options`; with an MFEM library that
does not export it the options raise an error that says so).  ``"relax_wt=1.8,pmax=6"``
keeps l1-Jacobi and gives it a weight.  The sums that l1-Jacobi divides by bound the
eigenvalues of the scaled matrix by one, so any weight below two leaves the
preconditioner positive definite: most of plain Jacobi's gain without its condition.  On
the MI250X cards of Frontier (double precision, 2.1 million state unknowns a GPU) an
incremental solve took 17 iterations instead of 26 at 64\ :sup:`3` on one GPU and 17.5
instead of 31 at 512\ :sup:`3` on 512, and two Newton-CG steps there 10.8 s instead of
14.3 s (plain Jacobi with the same six entries: 14 iterations at 128\ :sup:`3` on eight,
where this takes 17).  With a parameter three times as large it converged in 22
iterations at 64\ :sup:`3` against 32 with MFEM's settings, where plain Jacobi's solves
ended after 10.  It is not the default either.

.. _gpu-pmg:

**A p-multigrid in place of BoomerAMG** (``HIPPYMFEM_PMG=1``, the solvers' parameter
``pmg``; in the single-precision solves).  BoomerAMG on the matrix of quadratic elements
coarsens a matrix with 64 entries a row, builds an interpolation for it and forms the
Galerkin product with that interpolation: a set-up, which is a third of a forward solve at
a new parameter.  The linear elements of the same mesh are a coarse space that is there
already.  Their interpolation ``P`` to the state space follows from the two spaces alone
(:mod:`hippymfem.fem.ptransfer`: a vertex keeps its value, the middle of an edge takes
the mean of its ends), ``P^T A P`` is the matrix of the same form on them, with an
eighth of the rows and a twentieth of the entries, and BoomerAMG is set up on that
(:class:`~hippymfem.algorithms.singlesolve.PMGEngine`).  One application of the
preconditioner relaxes on the Jacobian as the finest level of BoomerAMG would (l1-Jacobi
with the weight ``relax_wt``, Jacobi with ``relax=7``), takes one V-cycle of that
hierarchy on the restricted residual, and relaxes again.  The Galerkin product passes
through ``A P``, which the engine keeps: the residual after the coarse correction is then
``t - (A P) e``, a product with a quarter of the Jacobian's entries in place of the third
product with the Jacobian that an iteration would make (``HIPPYMFEM_PMG_KEEP_AP=0``
computes that residual with the Jacobian again and saves the 2 bytes a Jacobian entry
that the product takes on the card).  For a scalar state on a domain term the two
products are not formed by hypre at all: ``A P`` and ``P^T A P`` are sums over the
elements of ``A_e P_e`` and ``P_e^T A_e P_e``, with the interpolation ``P_e`` of one
element, and the pass that assembles the Jacobian scatters them into their own patterns
(``PDEVariationalProblem._jacobian_with_products``; ``HIPPYMFEM_PMG_ELEMENT_PRODUCTS=0``
leaves them to hypre).  The same two matrices, for a third of the Jacobian's scatter
more, and without the part of a set-up that grows with the ranks.

On the MI250X cards of Frontier, with 14 million state unknowns a GPU and
``relax_wt=1.8,pmax=6``, at the eight parameters of the forward solves of two Newton-CG
steps (a line search's trial points among them): six or seven iterations to 1e-5 at
every one, as with BoomerAMG, of 31 ms instead of 43 ms on one GPU and 34 ms instead of
48 ms on eight (40 and 44 ms without the kept product), and a set-up of 0.39 s instead of
1.18 s on one GPU (the product 0.20 s, the hierarchy 0.17 s) and of 0.67 s instead of
1.60 s on eight.  A forward solve at a new parameter then takes 2.5 s instead of 3.5 s on
one GPU and 3.0 s instead of 4.1 s on eight, and the two steps, charged for the same
operations, 21.8 s instead of 29.4 s and 26.3 s instead of 34.4 s.  The state, the
adjoint and the gradient at the prior mean agree with BoomerAMG's to 6e-11, and hypre
holds 19.5 GiB of a card at its fullest instead of 21.8 GiB.  With the products from the
element matrices the set-up is 0.18 s on one GPU and 0.29 s on eight (0.42 s on 4096),
an assembly 0.10 and 0.16 s longer, and the two steps take 3 to 4 % less.  The
interpolation is built once for a problem, on the host, for
the nodal H1 spaces of order two or more on a conforming mesh; a problem whose state
space allows none keeps BoomerAMG, with a warning, and so does a library that lacks one
of the calls (it was built with hypre 3.2, and forms the product with an older one as
MFEM does).  Off by default.

.. _gpu-pmg-h:

**Geometric levels under the linear elements** (the solvers' parameter ``pmg_h``,
``HIPPYMFEM_PMG_H``; with ``pmg``).  BoomerAMG on the linear elements' matrix still
coarsens a matrix, builds interpolations and forms their products at every set-up, and
what it builds has 100 to 170 entries a row from its second level on and one level more
for every eight times the ranks, each with its exchanges whatever its size: the cost of
a cycle and of a set-up grows with the ranks.  A mesh that was made by refining
coarser meshes has a hierarchy without any of that.  Tell the library the meshes
(:func:`hippymfem.fem.ptransfer.set_coarser_meshes`: the mesh that ``ParMesh.MakeRefined``
cut into the mesh at hand, the one that was cut into that, and so on;
``applications.geothermal.model.cube_mesh(n, comm, levels=...)`` does it for the mesh of
the benchmarks, ``--h-levels`` there), and with ``pmg_h=k`` the correction from the
linear elements is a V-cycle over up to ``k`` of them (for a state of one component: a
state of several keeps BoomerAMG under its linear elements, with a warning).  The
interpolation between the
linear elements of two meshes is MFEM's own on local dofs
(:func:`~hippymfem.fem.ptransfer.refinement`: the values of a coarse function at the
fine vertices), with the essential dofs taken out as in the interpolation from the
linear elements, so that every Galerkin product has the identity where the Jacobian has
it; the matrices are those products level by level, made by the library at every set-up
with ``A P`` kept for the residual; the relaxation is the fine level's; and BoomerAMG is
set up on the last product alone, which ``seq_threshold`` gathers
(:class:`~hippymfem.algorithms.singlesolve.PMGEngine`).  Every level has the 27 entries a
row of linear hexahedra and the neighbours of the mesh, at any number of ranks.  The
levels end at the first matrix that the hierarchy would gather anyway.  Where the next
mesh is three or five times coarser a side (a cut by three or five, which a mesh of
120\ :sup:`3` elements a rank needs under its three cuts in two) the correction from it
leaves more to the relaxation than one sweep does away with, and that level relaxes by
as many steps of a Chebyshev polynomial on the matrix scaled by its rows' l1 norms
instead (``HIPPYMFEM_PMG_H_SMOOTH=sweep`` keeps the sweep).  The interpolation is built
for linear elements on a conforming mesh that MFEM refines in parallel (hexahedra and
quadrilaterals in MFEM 4.9); a mesh that knows no coarser one keeps BoomerAMG on the
linear elements, with a warning.  Off by default.

**The small levels in host memory** (``pmg_h_host``, ``HIPPYMFEM_PMG_H_HOST``: rows a
rank).  A level of a few thousand rows a rank costs a device almost nothing to compute
and all of its latency: a kernel launch for every operation and, around each exchange
between ranks, a copy from the card and one back.  The meshes of at most that many rows
a rank, and everything under them, can live in host memory beside the device instead:
their matrices are the same Galerkin products (the first is copied from the device at
every set-up, the others are made in host memory), their relaxation the same, and a cycle
hands the restricted residual to the host at the first such level and takes the
correction back (:class:`~hippymfem.algorithms.singlesolve.PMGEngine`,
``_Library.to_host``).  On the host such a level takes microseconds and its exchange is a
message.  It pays for these levels where it did not for the last levels of an algebraic
hierarchy (tried on 11 October, below): a geometric level has the neighbours of the mesh
at every size, 26 for a box, where a rank of an algebraic one talks to more and more
ranks as it gets coarser, and messages cost the host what they cost the device.  The
last matrix, where the solver's ``seq_threshold`` gathers it, is gathered by the engine
itself and solved by a BoomerAMG of one process on every rank
(:class:`~hippymfem.algorithms.singlesolve._HostGathered`).  hypre's own code for that
in host memory coarsens the matrix once between the ranks and gathers what is left on
the ranks that hold rows of it; a rank that holds none goes on to a collective of all
the ranks, and both sides wait: a last matrix of 125 rows on 64 ranks stopped in its
first cycle that way, and one of 729 rows on 512.  A gathered matrix has no exchange to
save, and every rank solves the whole of it on one core, so the setting bounds it too:
where the last matrix has more rows than that in all, the levels above it stay on
the device with it (with 15 625 rows an iteration took 1.0 to 1.4 ms longer in host
memory, on 32 and on 256 GCDs; with 729 rows 1.1 to 1.2 ms less, on 128 and on 512; with
4913 rows 0.6 to 0.7 ms less, on 1024 to 4096), and the last matrix alone is never
moved.  Off by default.

**A prior's cycles over the same meshes** (a prior's ``h_levels``, ``prior_h``,
``HIPPYMFEM_PRIOR_H``).  Where the multigrid cycles of a prior's elliptic operator
precondition the Hessian (``cg_preconditioner_cycles``), they can be V-cycles over the
meshes that the mesh was refined from, in the single-precision library, in place of
cycles of BoomerAMG on the operator in double precision
(:class:`~hippymfem.algorithms.singlesolve.HCycle`): the operator is copied there once
(:func:`~hippymfem.algorithms.singlesolve.matrix_blocks`), its Galerkin products are made
once, since the operator stays, and the fixed polynomial of cycles runs wholly in that
library, with two conversions a solve.  It is one linear map up to the rounding of
single precision, about 1e-7 of its result, which a preconditioner whose own error is
1e-4 to 1e-3 of the solve does not see.  For a parameter space of linear nodal elements
with one component, on a mesh that was last cut in two a side: after a cut by three the
cycle relaxes on the finest matrix by a polynomial of three steps and costs more than
BoomerAMG's (0.061 to 0.079 s for the preconditioner of one CG iteration against 0.047 to
0.063 s, on 2, 16 and 128 GCDs; 0.033 to 0.058 s against 0.052 to 0.072 s after a cut
in two).  A prior on such a mesh, or on one that knows no coarser mesh, keeps BoomerAMG,
with a warning.  Off by default.

Keeping the hierarchy of a linearization point for the trial points of its line search
instead (hypre's PCG on the new matrix with the older hierarchy, which takes its finest
matrix from the solve) does not work on this problem.  Its first Newton steps move the
parameter by 1 to 54 in the maximum norm, a coefficient that changes by a factor of
three at the least, and at each of the fourteen trial points of two steps on one GPU and
on eight the solve either ran to 200 iterations or stopped after a few on a
preconditioner that was no longer positive definite (the weights of the finest
relaxation are the old matrix's), with a residual of the size of the right-hand side.

**The blocks of a linearization point in single precision** (``single_blocks``,
``HIPPYMFEM_SINGLE_BLOCKS``).  The CG of a Newton step applies ``C``, ``W_um`` and their
transposes in every iteration.  hypre on a device has no transposed product: it needs
the transpose of a matrix, which MFEM forms and keeps at the first transposed product
after each linearization point.  At 120\ :sup:`3` quadratic hexahedra a rank the two
blocks and their transposes are 9.7 GiB of a card in double precision (217 million
nonzeros each, twelve bytes a nonzero, four times), and at 128\ :sup:`3` a card of 64 GiB
has 1.0 GiB left with them.  A product with these blocks does not need double precision:
the incremental solves on either side of it stop at 1e-5 to 1e-6, and in the mixed
precision the element matrices of the blocks are single precision's already.  With the
setting, the blocks whose elimination folds into the scatter are matrices of the
single-precision library: four bytes a value, one copy of the column indices for the
blocks of a pattern (which the pattern keeps, so that finishing a block uploads no index),
and the two transposes in single precision as well.  On one node of MI250X GPUs the
least free memory of a card goes from 15.6 to 20.3 GiB at 120\ :sup:`3` elements a GPU
and from 1.0 to 6.8 GiB at 128\ :sup:`3`, the assembly of the blocks from 1.04 to 0.84 s
and from 1.22 to 1.01 s, and the operations of two Newton-CG steps by 0.43 and 0.53 s
(2.6 %); a product costs 1.3 to 2 ms more, for the two conversions of a vector around
it, so a Newton step of many CG iterations gains less.  ``W_mm`` and a block with a facet part stay in double precision.
Off by default.

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

The library as it is now, measured on 5 October at 64\ :sup:`3` and at 128\ :sup:`3`
(17.0 million state dofs) with the settings of :ref:`gpu-recommended` (incremental solves
to 1e-6, the forward and the adjoint solve refined to the solver's tolerance): double
precision throughout against single-precision element matrices and solves, the time of
a solve after a first one and the memory of the busiest card at its peak:

.. table::
   :widths: auto

   ==============  ========================  ===================  ===================  =======
   mesh            GPUs                      double precision     single precision     faster
   ==============  ========================  ===================  ===================  =======
   64\ :sup:`3`    1 H100                    26.7 s, 18.9 GiB     20.8 s, 13.0 GiB     1.29x
   64\ :sup:`3`    1 L40S                    61.2 s, 18.7 GiB     40.6 s, 12.8 GiB     1.51x
   64\ :sup:`3`    1 Blackwell instance      71.9 s, 14.7 GiB     39.8 s, 12.8 GiB     1.81x
   64\ :sup:`3`    4 Blackwell instances     33.0 s, 6.3 GiB      22.7 s, 5.7 GiB      1.45x
   128\ :sup:`3`   4 L40S                    224.2 s, 22.2 GiB    142.9 s, 19.8 GiB    1.57x
   128\ :sup:`3`   4 Blackwell instances     252.5 s, 22.2 GiB    144.6 s, 19.8 GiB    1.75x
   128\ :sup:`3`   1 H100                    249.9 s, 64.1 GiB    198.7 s, 58.4 GiB    1.26x
   ==============  ========================  ===================  ===================  =======

Every run at 64\ :sup:`3` took twelve Newton steps and 131 CG iterations and every run
at 128\ :sup:`3` thirteen and 191, in both precisions; the runs at 128\ :sup:`3` release
each linearization point before the next (``release_linearization_on_move``).

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
a little slower than in double precision.  Refined to 1e-9 only (``SINGLE_REFINE_GOAL``)
the forward solve took 0.34, 0.65 and 0.60 s and the adjoint solve 0.093, 0.192 and
0.191 s, which makes the adjoint solve too 1.1 to 1.3 times faster than in double
precision.  The first refined adjoint solve of a process also compiles the
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
CUDA context; a seventh of it goes.  Since a Gauss-Newton point assembles its single
block a chunk at a time, the H100's card in single precision peaks at 13.0 GiB instead
of the 17.5 GiB of this table, the L40S's at 12.8 GiB and the busiest of four Blackwell
instances at 5.7 GiB (the table of 5 October above; double precision unchanged).  The
other blocks of a linearization point stay in double precision (for a forward problem
that is linear in the state these are ``C`` and ``W_um``, 0.4 GB each at this size).  The Jacobian's accumulator does not.  Where the
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
Jacobian and took the same 23 s.  With ``HIPPYMFEM_SINGLE_AMG="relax=7,pmax=6"`` (above)
the eigensolver took 17.4 s with the same errors.  So the single-precision solves stay on for the
stages after the MAP point.  Where eigenvalues are wanted to more digits than that,
switch them off (in ``test_uq`` the eigenvectors diagonalized the Hessian to 3e-7 of the
largest eigenvalue instead of 1e-9):

.. code-block:: python

   pde.set_single_solves(False)       # the Jacobian is assembled in double precision again
   hm.config.precision = "fp64"       # optional: and its element matrices
   model.setPointForHessianEvaluations(x)

``applications/precision/model_subsurf_single.py`` does this.  At 8\ :sup:`3` on one host
core (4 913 state dofs; Newton-CG to 1e-8, twenty eigenpairs, incremental solves to 1e-12
after the MAP point), against a run in double precision throughout, the eigenvalues
differed by up to 1.3e-5 relative with the single-precision solves left on, by 2e-7 with
the solves switched off and the element matrices left in single precision, and by 4e-10
with both switched, the level at which the two MAP points agreed (2e-9).  Switching the
element matrices compiles their kernels again (at 24\ :sup:`3` on a Blackwell MIG instance
twenty eigenpairs then took 109 s, against 3 s with the kernels already compiled), and
incremental solves to 1e-8 move the eigenvalues by more than 2e-7 anyway
(`The Laplace approximation on the cards`_).

The 1e-5 also shows in the iterates on the way.  In the first Newton step of
``bench_newton_device.py``, from the prior mean and with the full Hessian, CG stopped
after three or four iterations in both precisions, and the directional derivative of
the step differed by 0.2 %.  Two steps in, far from the minimum, the costs differed by
6 % (128\ :sup:`3`) and 30 % (64\ :sup:`3`).  The runs to a tolerance agreed, as above,
so compare such runs and not a fixed number of steps.

What to set, for a symmetric problem solved by CG with BoomerAMG (:ref:`gpu-recommended`
has the rest, :ref:`hypre-single-install` the build):

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

**Keep a million state dofs or more on each GPU.**  One card (half an RTX PRO 6000
Blackwell) needs 0.3 ms plus 5.3 ns per state dof for a CG iteration with a BoomerAMG
V-cycle.  On several cards every iteration adds 23 to 31 halo exchanges, one per
product with a level matrix or an interpolation matrix, and without a GPU-aware MPI
each of them goes through the host: 1 to 2 ms per iteration at up to 1.1 million dofs
per card, 2.5 to 5 ms at 2.1 million.  So at 32\ :sup:`3` four Blackwell slices are
hardly faster than one (1.47 against 1.55 s for two Newton steps) and at 64\ :sup:`3`
four L40S 2.2 times, while at 128\ :sup:`3` four L40S are twice as fast as two, and
sixteen Blackwell cards at 256\ :sup:`3` 2.1 times as fast as eight.

Three settings matter on several cards.  Their defaults are the ones to use:

* ``HIPPYMFEM_HYPRE_SPMV=auto``.  hypre keeps a parallel matrix as a diagonal and an
  off-diagonal block per rank, and by default multiplies with both through cuSPARSE,
  which takes up to 5 ms for an off-diagonal block that is almost empty: two cards were
  slower than one.  ``auto`` uses hypre's own kernel on more than one rank of a CUDA
  build and the vendor's on one rank (:func:`~hippymfem.common.mfemconfig.set_hypre_spmv`).
  It changes nothing but the time.
* ``HIPPYMFEM_HYPRE_POOL=auto``.  A BoomerAMG setup takes thousands of device arrays
  from the driver, and on sixteen ranks of a node those calls were 1.28 s of a 3.21 s
  forward solve.  The library gives hypre a recycling pool
  (:func:`~hippymfem.common.mfemconfig.set_hypre_pool`) that may hold 1 GiB while a
  setup runs and 512 MB between setups (``HIPPYMFEM_HYPRE_POOL_KEEP``, in megabytes).
  ``HIPPYMFEM_HYPRE_POOL=<megabytes>`` asks for one limit at all times and ``0`` for no
  pool; :meth:`~hippymfem.common.mfemconfig.HyprePool.trim` returns what it holds, and
  it is returned by itself when the card runs out of memory.  On AMD cards the pool was
  worth 0 to 2 % of two Newton steps on up to 32 MI250X.  Its counters (``driver_seconds``
  and ``driver_bytes`` of :class:`~hippymfem.common.mfemconfig.HyprePool`) show seconds
  inside the driver there, 7 to 19 ms for each GiB with tens of GiB a setup, but nearly all
  of them in the call that gives a block back, which returns in 0.1 ms on an idle card:
  half of that time is the wait for the kernels still running.  A pool of 24 GiB, which
  keeps every large block from one setup to the next, took the driver's share of a forward
  solve from 0.26 s to 0.10 s and the solve from 4.36 s to 4.28 s.  What the large
  temporaries of a setup cost is the copies and sorts that fill them, and that is dealt
  with in hypre (``tools/hypre-3.2.0-spgemm.patch``, below).
* ``HIPPYMFEM_DEVICE_VECTORS=auto``.  A :class:`~hippymfem.common.parvector.ParVector`
  follows its data: once hypre has used it on the device, its arithmetic runs there and
  the observation operator is a hypre matrix, so nothing is copied down and up around a
  solve.  ``.array`` remains the way to the host and the point where the values are
  brought down.  ``0`` restores numpy on the host.

Two optional builds of hypre make the exchange itself cheaper, and
``tools/rebuild_hypre.sh`` produces either from the hypre source of an existing PyMFEM
build: ``--pinned-staging`` keeps page-locked host buffers for the exchange (3 to 6 %
per Krylov iteration), and ``--gpu-aware-mpi`` hands the device buffers to a CUDA-aware
MPI (13 % on two H100; not for MIG slices, which cannot use CUDA IPC).  Neither is what
``tools/build_pymfem_cuda.sh`` installs by default, and every time in this guide is
that of the unpatched library.

The HIP build has the first of the two as ``HYPRE_PINNED_STAGING=1
tools/build_pymfem_hip.sh``, which applies ``tools/hypre-3.2.0-staging.patch`` to its
hypre 3.2.0: the same page-locked buffers, and an exchange that starts before the product
with the local block instead of after it, so that the copy to the host and the sends
overlap with that kernel.  At 128\ :sup:`3` on eight MI250X an iteration took 9.9 ms
instead of 11.2 ms (10.7 with the buffers alone, 10.6 with the early start alone), and
two Newton-CG steps at 512\ :sup:`3` on 512 took 10.0 s instead of 10.8 s.
``HYPRE_STAGE_PINNED=0`` and ``HYPRE_STAGE_EARLY=0`` in the environment of a run switch
either off.  hypre 3 chooses between its staging and a GPU-aware MPI at run time (hypre
2.31 and 2.32, the CUDA stack's, have the switch and an exchange that is compiled one
way: the setting changes nothing there, which is said), and
``HIPPYMFEM_GPU_AWARE_MPI=1`` or ``0`` sets it whatever the library was built with, for
MFEM's hypre: the single-precision library (``HIPPYMFEM_HYPRE_SINGLE``) stages its
exchanges always, so with the solves in single precision the setting reaches the
double-precision ones alone.  With
Cray MPICH 9.1.0 and its GTL library on Frontier that exchange took 10 to 20 % off an
iteration where it ran, and deadlocked or faulted every time at 320\ :sup:`3` on 128 and
at 400\ :sup:`3` on 256 GPUs, whatever the MPI library was told: the staging is what to
use there.

``HYPRE_SORTED_SPGEMM=1 tools/build_pymfem_hip.sh`` applies
``tools/hypre-3.2.0-spgemm.patch``, which is about the setup of a hierarchy and not its
use.  hypre computes the coarse operators of BoomerAMG with rocSPARSE's sparse product and
hands it both operands with their rows in ascending order: before every product it copies
each operand and sorts the copy, with a permutation and a second copy of the values beside
it.  Most operands of a setup are sorted as they come (a transpose, the result of the
product before), and the largest that is not, the fine matrix, is sorted but for the
diagonal entry, which hypre keeps first in each row.  With the patch a product checks the
rows of its operands in one pass and takes a sorted one as it is; the sorted copy of a
matrix with only its first entries out of place is written in one pass, a warp a row; the
matrix that joins a diagonal block with its off-processor one is written in order in the
first place; and the rows a rank receives are sorted before they are stacked under its
own.  That saves the copies and the sorts, and 10 to 16 GiB of device memory at the peak
of a setup.  With 8.6 million state unknowns on each MI250X (quadratic elements, 545
million nonzeros a GPU) a setup takes 0.81 s instead of 1.53 s on one GPU and 1.0 to 1.2 s
instead of 2.3 to 2.5 s on two to eight, a forward solve 4.5 s instead of 5.4 s and 5.1 s
instead of 6.5 s, and hypre's device memory peaks at 13 to 15 GiB instead of 23 to 31; the
cost functional agrees to nine digits and the CG counts are the same.
``HYPRE_SPGEMM_SKIP_SORT=0`` in the environment of a run has hypre copy and sort every
operand again.
From one GPU to 1024 (1024\ :sup:`3` elements, 18.3 billion unknowns) the two steps of the
Newton benchmark with that load take 6 to 15 % less time with the patch (30.2 s on 1024
GPUs instead of 33.0 s), and 9.5 to 14 % less when every size is charged for the same
operations.  1024 GPUs then keep 78 % of the rate of one, where they kept 76 %, and 88 % of
the rate of one node, where they kept 90 %: a setup lost 0.7 s on one GPU and 1.1 to 1.4 s
on any more, the same on a node as on 128.  With hypre at half its memory, that load also no longer needs JAX held to a
fifth of the card (``HIPPYMFEM_GPU_MEM_FRACTION=0.2``): with the library's own share the
operations charged take 27 s on one GPU and 32 s on eight, where they took 44 and 51 s.

A reduced-Hessian application is then its two incremental solves almost entirely (93 to
96 % of it).  For the stages after the MAP point there is a second way to use several
cards, which scales better when the problem fits on one: see `The Laplace approximation
on the cards`_.

.. _gpu-memory:

How large a problem fits
------------------------

A Newton step holds about 2 kB of device memory per unknown (state, parameter and
adjoint together): the assembled blocks and the AMG hierarchies.  That puts
128\ :sup:`3` (36 M unknowns) on four 45 GB L40S at 20 GiB per card, on two of them at
34 GiB, or on one 80 GB H100 at 58 GiB; a Newton-CG run to its tolerance peaked there at
64.1 GiB in double precision (253.8 s) and at 58.4 GiB with single-precision element
matrices and solves (197.7 s), both with ``release_linearization_on_move``.  At
400\ :sup:`3` a 48 GB slice holds 37 GiB.  What decides whether a given run fits:

**JAX's share of the card.**  The share (0.45 of the card with hypre on it) is a
ceiling, not a reservation: the pool grows as the element kernels need it, and their
batches are sized from the room left under the ceiling, so a lower share means smaller
chunks and a smaller pool, at some cost in time.  Set
``HIPPYMFEM_GPU_MEM_FRACTION=0.20`` when a run is short of card memory: at
128\ :sup:`3` on four L40S it saves 2.8 GiB per card for 14 % more time.
``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` is not a substitute: it reports no budget to
size the chunks from, and its card peak was higher.

**Matrix-free linearization points.**  ``setLinearizationPoint(x, matrix_free=True)``
leaves ``C``, ``W_um`` and ``W_mm`` unassembled and computes their products at every
Hessian application instead: 1.5 GiB per card for 8 % more time in the same run.  With
the lower share as well it is 3.2 GiB for 50 %, so it is for a run that still does not
fit.

**What the problem class keeps.**  Three settings of
:class:`~hippymfem.modeling.PDEVariationalProblem.PDEVariationalProblem` decide how much
a Newton step holds:

* A symmetric Jacobian is detected by a probe at every assembly
  (``symmetric_jacobian="auto"``, the default), and the adjoint solves then reuse ``A``
  and its AMG hierarchy instead of a transposed copy and a second hierarchy: 3 GB plus
  6 GB per card at 128\ :sup:`3`.
* ``transpose_free_adjoint=True`` does the same for a non-symmetric Jacobian: it solves
  ``A^T p = b`` through ``A``'s transpose action with ``A``'s own hierarchy as the
  preconditioner.  The geothermal application uses it.
* ``release_linearization_on_move=True`` drops the previous linearization point at the
  first forward solve for a new parameter.  It is safe for line-search Newton-CG and
  opt-in because other algorithms may use the old point after a trial solve.

The library itself holds only what is current: a solver keeps its present operator and
hierarchy and no other, and an operator is released before its replacement is
assembled.  ``test_device`` checks that card memory is flat across operator resets.

**How the element batch is split.**  The element kernels run the batch in chunks sized
before the first launch from the kernel's shape against a third of what is free, where
free is the smaller of JAX's remaining budget and what the driver reports
(``HIPPYMFEM_GPU_MEM_RESERVE``, in GiB, is subtracted for whatever else shares the
card).  On the host the budget is this rank's share of the node's available memory
(``HIPPYMFEM_HOST_MEM_FRACTION``).  An out-of-memory error shrinks the chunk and
retries, and ``HIPPYMFEM_ELEMENT_CHUNK`` pins a size.  The chunk count is a memory
choice, not a speed one: a launch costs about 5 ms.  Chunking changes results at
round-off only (about 1e-15 relative), and an unsplit batch stays bit-identical.

Once a batch is split, the per-element geometry and the scatter map stay on the host
and are sliced per chunk (above ``HIPPYMFEM_GEOMETRY_STREAM``, 0.25 of the budget, for
the geometry).  A streamed geometry is copied to the card once per element pass, and how
it is copied decides what such a pass costs.  JAX moves a numpy array through a staging
buffer of its own, measured at 3.4 to 5.6 GB/s on an H100 (8.8 GB/s from JAX's own pinned
host arrays), whose PCIe 5 link moves 55 GB/s from memory locked for the device.  Where
the device bridge can be used (hypre and the kernels on one card), a group that streams
its geometry has it locked once (``cudaHostRegister``) and each chunk's slice is copied
by the CUDA runtime, of the arrays the kernel reads only (the coordinates ``X`` of the
quadrature points not at all where the density does not use them);
``HIPPYMFEM_PINNED_STREAM=0`` restores JAX's copy.  The element arrays are the same to
the last bit.  At 64\ :sup:`3` on the H100 in double precision with the
streaming forced (``HIPPYMFEM_ELEMENT_CHUNK=23000``, ``HIPPYMFEM_GEOMETRY_STREAM=0.001``),
the gradient took 0.063 s instead of 0.15 to 0.17 s (0.027 s with the geometry kept on
the card), a forward solve 0.48 s instead of 0.73 to 0.93 s (0.36 s).  At 128\ :sup:`3`
on one H100 with single-precision element matrices and solves, where the geometry
(13.9 GB) has to stream, the forward solve took 2.86 s instead of 5.51 s, the adjoint
solve 1.25 s instead of 2.52 s, the gradient 0.40 s instead of 1.05 s, and the Newton-CG
solve 197 s instead of 326 s, in the same 13 Newton and 191 CG iterations; in double
precision, which did not fit the card before the Gauss-Newton points were assembled a
chunk at a time, the Newton-CG solve took 253.8 s.

**An affine group has little to stream.**  Where every element of a group is mapped
affinely (straight simplices, parallelepipeds: any uniformly refined box) its inverse
Jacobian is the same at all of its quadrature points, and
``HIPPYMFEM_AFFINE_GEOMETRY=1`` (``hm.config.affine_geometry``, for the batches built
afterwards) keeps it once an element: 72 bytes instead of 4.6 kB with 64 points.  The
inverse Jacobians and the weights of such a group then stay on the card (1.0 GB for 1.7
million quadratic hexahedra, where the whole geometry is 11.5 GB), and only the
coordinates are left on the host, for the kernels whose density reads them.  A pass over
a streamed geometry is mostly the stream: on an MI250X with those elements a residual
takes 0.46 s with the geometry streamed and 0.09 s with it on the card.  And the eight
ranks of a node stream at once: with a million elements each their element passes were
5 to 33 % slower than those of a rank alone (a residual 5 %, the scatter of the Hessian's
blocks 20 %), and with the geometry on the card a residual is within 3 % and the scatters
within 5 %.  With the compact geometry, on eight MI250X in double precision, two
Newton-CG steps of the benchmark above take 59.8 s instead of 72.2 s, and a rank holds
34.5 GiB of host memory instead of 41.9 GiB.  It is off by default: the one inverse
Jacobian is the mean over the points, which differ in their last digits where the map is
affine, so the element arrays change at round-off (the norms of the state and of the
gradient in the twelfth and the eleventh digit there).  A group with one curved element
keeps an inverse Jacobian a point.

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
driver in regions, so a request for one large *contiguous* array can fail against a cap
that reads almost empty.  The array that provokes it is the assembly's accumulator, one
double per nonzero: 7.7 GiB at two million P2 hexahedra on a rank.  When one array is
more than roughly a sixth of the arena, set ``XLA_PYTHON_CLIENT_PREALLOCATE=true`` and
size ``HIPPYMFEM_GPU_MEM_FRACTION`` for hypre's share; the assembly says so in a
``RuntimeWarning`` that names the size and the share.

**A rank holds at most about four million P2 hexahedra.**  hypre addresses a rank's
nonzeros with 32-bit indices, and four million P2 hexahedra carry 2.06e9 nonzeros in
the state Jacobian, 96 % of what they can address.  Add ranks past that.

**The one-time setup.**  Before its first assembly a space pair builds its sparsity
patterns once, the largest one-time cost of a large run (1.5 billion entries a rank at
256\ :sup:`3` on eight GPUs).  With numba installed (the ``perf`` extra) they are built
on the host without a global sort, three times faster and with no device memory
(:mod:`hippymfem.fem.patternbuild`; ``HIPPYMFEM_PATTERN_THREADS`` sets the thread
count, by default this rank's share of the node's cores).  numba keeps the compiled
passes on disk, and the library points it to a directory of the node,
``hippymfem-numba-<uid>`` under the system's temporary directory, unless
``NUMBA_CACHE_DIR`` names another: numba's own place is next to the sources, and on a
file system shared by the nodes one of 32 ranks started from a fresh checkout stopped
with ``Stale file handle`` while the others rewrote the cache.  The first run on a node
compiles them, in three seconds.  The tables are what a rank's host memory goes to: with
a million P2 hexahedra and the Jacobian, the two rectangular blocks and the parameter
block of a Newton step, 13 GiB of 27, the element geometry 7.  They are built from the
dof tables and one row at a time, and a map with one entry per element-matrix entry is
given up once the map made from it exists (``HIPPYMFEM_PATTERN_LEAN``, on; a map given up
is made again if something asks for it).  Before that the same tables took 30 GiB, and a
rank 50 GiB at the peak of their builds, which eight ranks of a node with 512 GB could not
hold for more than a million elements each; 1.7 million a rank now take 41 to 43 GiB, and
the Jacobian's tables are built in 16 s instead of 29 s.  In an assembly on the card
the scatter map arrives from the host slice by slice in a compact form (one byte an entry
and two bases an element row, a third of the four-byte map, which
``HIPPYMFEM_COMPACT_PATTERN=0`` restores), and ``HIPPYMFEM_DEVICE_PATTERN=1`` keeps it
and the column indices on the device, at the price of their memory.

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
eigensolver (140 Hessian applies)        199.1 s                      9.2 s                 25.1 s                147.3 s
one posterior sample                     165 ms                       24 ms                 46 ms                 285 ms
pointwise variance, randomized, r = 64   4.6 s                        1.7 s                 4.5 s                 14.2 s
pointwise variance, Monte Carlo, n = 64  --                           1.2 s                 2.4 s                 14.6 s
traces, r = 64                           5.8 s                        1.9 s                 4.9 s                 16.2 s
=======================================  ===========================  ====================  ====================  =====================

The random stream is a vectorized Philox generator, bit-identical to numpy's and on the
card when the kernels are, which is what makes a sample tens of milliseconds.  The
``"Randomized"`` pointwise variance is a truncated spectrum and 20-25 % low at r = 64;
the ``"MonteCarlo"`` method (one solve per sample) is unbiased.  The incremental and
prior solves can run at 1e-8 for these stages: the eigenvalues move by 6e-6 at most and
the eigensolver is 1.4x faster.  At 128\ :sup:`3` the whole workflow, MAP included,
runs in 10.4 minutes on four L40S (two on each of two nodes, as above), and in 8.3 minutes
with the incremental solves of the MAP point stopped at 1e-6 (:ref:`gpu-recommended`).

**Several GPUs: an ensemble instead of a domain decomposition.**  Every one of these
stages is a set of independent solves (140 Hessian applications in the eigensolver, one
or two prior solves per sample or probe).  Dividing the *mesh* over the GPUs makes each
solve faster by the factor of `Several GPUs`_ and no more.  When the problem fits on one
GPU, divide the *vectors* instead.  Build the problem on ``MPI.COMM_SELF`` on every
rank, so that each GPU holds all of it, and pass the world communicator as
``ensemble``::

   d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, ensemble=MPI.COMM_WORLD)
   post.pointwise_variance(method="MonteCarlo", n=64, ensemble=MPI.COMM_WORLD)
   post.trace(method="Randomized", r=64, ensemble=MPI.COMM_WORLD)

Rank ``r`` applies the operator to the columns ``r, r + size, ...`` and the columns are
exchanged, so every rank ends with the same eigenpairs; the random streams are
partition-independent, so the ranks start from the same ``Omega``, and the Monte Carlo
samples are the ones a single rank would draw (:meth:`~hippymfem.common.random.Random.seek`).
``benchmarks/bench_laplace.py --ensemble`` does all of this.  At 64\ :sup:`3` on MIG
slices of RTX PRO 6000 Blackwell cards, incremental solves to 1e-8:

==========================================  ========  ===================  ===========
after the MAP point                         1 slice   8, domain decomp.    8, ensemble
==========================================  ========  ===================  ===========
eigensolver (k = 50, p = 20)                52.6 s    22.0 s               8.6 s
64 posterior samples                        5.7 s     3.1 s                0.9 s
pointwise variance, Monte Carlo, n = 64     4.0 s     2.6 s                0.7 s
all stages                                  73.3 s    40.8 s               16.4 s
==========================================  ========  ===================  ===========

The eigenvalues of the ensemble agree with those of one slice to 2e-8.  What an
ensemble does not speed up: the orthogonalizations, which are sequential in the columns
and which every rank repeats, and the MAP point, whose CG iterations depend on one
another.  Every rank also has to set the problem up and hold it, so the mesh has to fit
on one GPU.

Taking exactly the GPUs and cores you asked for
-----------------------------------------------

**GPUs.**  JAX opens a context on every device it can see when its backend comes up, not
only on the one it computes on.  The cure is one visible device per rank, set before the
process's first CUDA call, which is ``MPI_Init`` when the MPI is CUDA-aware.
``tools/mpirun_pinned.sh`` sets it at the launcher::

   mpirun -n 4 tools/mpirun_pinned.sh python script.py

The library also pins itself (``HIPPYMFEM_PIN_GPU``, on by default) when it is imported
before ``mpi4py`` and MFEM; imported after them it is too late, it says so on rank 0, and
the wrapper is the way.  With no device id, ``mfem.Device("cuda")`` puts **every rank on
GPU 0**; :func:`~hippymfem.common.mfemconfig.configure_device`, which the import runs
for you, picks ``local_rank % n_devices``, the same rule the element kernels use, so a
rank's matrix and its kernels share a card.

**A host run holds nothing on the cards**, whether ``jax`` or ``hippymfem`` is imported
first.

**Cores.**  With the element kernels on the CPU, XLA's worker pool takes about two cores
per rank whatever ``OMP_NUM_THREADS`` says.  OpenMPI's default ``--bind-to core``
confines each rank to one core; a run started without ``mpirun`` should be given
``taskset`` for the same reason.

AMD cards
---------

Everything runs on an AMD card: the element kernels through JAX, and MFEM and hypre
through a HIP build of PyMFEM.  On an AMD Instinct MI210 (64 GB, ROCm 7.2,
``jax-rocm7-plugin`` 0.11.1, hypre 3.2.0, MFEM 4.9), two Newton-CG steps of the
benchmark above:

===========================================  ======  ======  ======
two Newton-CG steps                          MI210   H100    L40S
===========================================  ======  ======  ======
:math:`64^{3}`, one GPU                      9.2 s   4.7 s   10.9 s
:math:`128^{3}`, four GPUs                   19.6 s  --      23.4 s
===========================================  ======  ======  ======

The answers are the NVIDIA ones: the same cost functional to seven digits and the same CG
counts at every size.  ``run_tests.sh 1 2`` with the HIP build passes every suite,
``test_gpu`` and ``test_device`` included.

**Kernels.**  ``HIPPYMFEM_DEVICE=gpu`` names ``rocm,cpu`` on a node whose card
``rocm-smi`` reports.  One XLA setting is changed for ROCm: command buffers (the
counterpart of CUDA graphs) segfault inside ``libamdhip64`` once an element batch passes
a size that depends on the block, so ``--xla_gpu_enable_command_buffer=`` is added to
``XLA_FLAGS`` when ROCm is selected.

**The kernel cache.**  The HIP runtime compiles through ``libamd_comgr``, XLA's kernels
and the code of a HIP build of MFEM and hypre alike, and keeps every result on disk, 12 MB
a kernel on average: by default in ``comgr`` under ``$XDG_CACHE_HOME`` or ``~/.cache``.
The library points it to a directory of the node, ``hippymfem-comgr-<uid>`` under the
system's temporary directory, unless ``AMD_COMGR_CACHE_DIR`` names another.  The home
directory is shared by the nodes of a cluster, and ranks that compile the same kernel
replace one another's file there while others have it open: of 32 ranks on four nodes of
Frontier, where home is NFS projected through DVS, ten waited there for ever and the
others for them.  The first run on a node compiles into an empty cache, at about half a
second a kernel.

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

**More than 2 147 483 647 unknowns in a space.**  hypre numbers the unknowns of a space with
``HYPRE_BigInt``, which is 32 bits wide in the build above as in PyMFEM's own, and MFEM stops
with ``overflow in offsets`` at the first space whose dofs, summed over the ranks before the
shared ones are removed, pass that range: with quadratic hexahedra a mesh of 640\ :sup:`3`.
``HYPRE_MIXEDINT=1 tools/build_pymfem_hip.sh <prefix>`` builds hypre ``--enable-mixedint``
(64-bit indices of a space and 32-bit ones within a rank; a GPU build of hypre has no
``--enable-bigint``) and has SWIG write PyMFEM's wrappers again, which otherwise convert a
global size as an ``int``.  The library reads the two widths from the build
(``parvector._HYPRE_INT`` and ``_HYPRE_BIGINT``) and runs on either: ``run_tests.sh 1 2`` passes
every suite on both, and at every size that both can run (four 64\ :sup:`3` blocks of elements
a GPU, 1 to 256 GPUs) the two builds gave the same cost functional to seven digits, the same
Newton and CG counts, and times within 1.3 %.  With it the Newton benchmark above
ran at 1024\ :sup:`3` on 1024 MI250X of Frontier, 8.6 billion state unknowns and 18.3 billion
in all: two Newton-CG steps in 33 s (44 s with the library's own BoomerAMG settings), 90 % of
the rate of one node of eight with the same 8.4 million state unknowns a GPU.  With hypre
built with both of its patches (``HYPRE_PINNED_STAGING=1 HYPRE_SORTED_SPGEMM=1``) and the
index tables of ``HIPPYMFEM_PATTERN_LEAN``, which let a rank of a node hold 1.7 million
elements, it ran at 1920\ :sup:`3` on 4096 MI250X (512 nodes): 56.7 billion state
unknowns, a parameter of 7.1 billion, itself past the 32-bit range, and 120.4 billion
unknowns in all.  Two Newton-CG steps took 66 s (4 forward and 2 adjoint solves, 8 CG
iterations) and a forward solve 9.1 s, of which the BoomerAMG setup is 2.7 s; charged for
the same operations at every size, that is 88 % of the rate of one node and 76 % of the
rate of one GPU with the same 14 million state unknowns a GPU, and a linearized solve took
20 iterations where it takes 17 on one GPU.

Then ``HIPPYMFEM_HYPRE_DEVICE=1`` puts MFEM on ``mfem.Device("hip")``:
:func:`~hippymfem.common.mfemconfig.configure_device` reads the backend from the build,
so ``"cuda"`` and ``"gpu"`` in a script mean HIP on this build and nothing written for
NVIDIA changes.  Rank pinning, in the library and in ``tools/mpirun_pinned.sh``, writes
``ROCR_VISIBLE_DEVICES`` on an AMD node.

**Single precision.**  The solves in a single-precision hypre (`Single precision`_) run
on AMD cards as on NVIDIA ones, and ``tools/build_hypre_single.sh <prefix> <directory>``
builds the library from the stack's own hypre source, patches and index widths included
(:ref:`hypre-single-install`).  On the MI250X of Frontier, with the Newton benchmark
above at 14 million state unknowns a GPU (120\ :sup:`3` quadratic hexahedra a GPU,
``HIPPYMFEM_AMG=relax_wt=1.8,pmax=6``, the incremental solves to 1e-5 in both
precisions), an iteration of a linearized solve takes 47 ms on eight GPUs instead of
61 ms and a BoomerAMG setup 1.7 s instead of 2.4 s, and a card has 16.6 GiB free at its
fullest instead of 7.3 GiB.  A refined forward or adjoint solve takes two more residuals
from the element kernels, half a second each at that size, so with the same settings two
Newton-CG steps take as long as in double precision: what the mode gives there is the
memory.  Given to JAX (``HIPPYMFEM_GPU_MEM_FRACTION=0.35`` instead of 0.25, which double
precision cannot spare at that load) it lets the element matrices run in chunks twice as
large, and the two steps, charged for the same operations, then take 49.7 s on eight GPUs
where double precision takes 62.0 s, and 43.7 s against 54.4 s on one.  The chunk planner
counts the tangents of the element matrices as doubles in either mode; with
``HIPPYMFEM_AD_WORKING_SET=8`` instead of 16, which is their size in the mixed mode, the
chunks double once more and the two steps take 45.8 s on eight GPUs, with 16.6 GiB of a
card free at its fullest.  With the compact geometry of an affine mesh on the card as
well (``HIPPYMFEM_AFFINE_GEOMETRY=1``, :ref:`gpu-memory`; JAX at 0.45) they take 34.4 s on eight
GPUs and 29.2 s on one, and 51.3 and 43.4 s in double precision; eight GPUs then keep
85 % of the rate of one in either precision, and what they lose is hypre's: a BoomerAMG
setup of 1.6 s where one GPU takes 1.2 s, 47 ms an iteration instead of 43 ms, the
prior's solves.  On 512 GPUs (960\ :sup:`3` elements, 15.1 billion unknowns) the same
operations take 41.1 s in that configuration and 80.9 s in the double-precision series
above: half the time at every size from 1 to 512 GPUs, and a lower ratio, 71 % of the
rate of one GPU and 84 % of that of one node where the slower series keeps 79 % and
92 %.  From one node to 64 both lose the same 6.7 s, which is the solver's (an
iteration 51 ms, a setup 1.7 s, eight iterations a linearized solve where one GPU takes
seven) and a larger share of the shorter time.  At that load a card has no room for a
double-precision Jacobian beside the others: on 128 GPUs a trial of a line search that
single precision could not carry ended a run out of device memory, and with
``SINGLE_GIVE_UP = "raise"`` (`Single precision`_) the trial was refused and the step
halved.  With the p-multigrid in place of BoomerAMG on the matrix (``HIPPYMFEM_PMG=1``,
:ref:`the paragraph on it <gpu-pmg>`) the operations take 24.3 s on one GPU, 28.4 s on
eight and 35.1 s on 512, 15 to 19 % less at every size, and the two steps at
960\ :sup:`3` take 33.3 s.  The ratio stays where it was, 69 % of the rate of one GPU and
81 % of that of one node on 512: from one node to 64 this configuration loses the 6.7 s
that the two before it lose, half of them in Krylov iterations whose number grows with
the mesh (a linearized solve 7.0 on eight GPUs and 8.0 on 512).  The state, the
adjoint and the gradient at the prior mean agree with the double-precision ones to
3e-11, 4e-10 and 3e-10 (237 million unknowns on eight GPUs).  The cost functional after
a few Newton steps did not (1.30e3 in the mixed mode and 2.11e3 in double precision
after two steps, 115 and 131 after eight, where two double-precision runs whose
incremental solves stop at 1e-12 and at 1e-5 are 0.2 to 1.2 % apart after two), and from
64 GPUs on a refined solve took more than its three passes: the sums of the Jacobian's
rows, rounded away in the single-precision library (:ref:`the paragraph on them
<single-row-sums>`).  With ``HIPPYMFEM_SINGLE_ROW_SUMS=1``, the products of the
p-multigrid from the element matrices and the element Hessians by the quadrature route
(``HIPPYMFEM_HESSIAN=quadrature``, which halves the Jacobian's kernel on this card: an
assembly 0.73 s instead of 1.23 s), and the rows of other ranks exchanged by messages to
the neighbours (``HIPPYMFEM_DEVICE_EXCHANGE``, the default: 18 ms a Jacobian's assembly on
64 GPUs where one collective on fresh host arrays took 166 ms), the operations take 19.4 s
on one GPU, 22.6 s on eight, 26.6 s on 512 and 29.2 s on 4096, at 1920\ :sup:`3`
elements and 120.4 billion unknowns: 2.9 times less than the double-precision series
above at that size, and 66 % of the rate of one GPU, 77 % of that of one node (90 % of
that of 256 GPUs).  The two steps there take 23.6 s, with the four forward solves and the
eight CG iterations of the double-precision run and its cost functional after them to
2e-4 (within about 2 % at every size from one GPU up).  What the 4096 GPUs lose against
one is by then the solver's Krylov iterations, a third more of them and each a third
longer, the set-up of the p-multigrid's hierarchy, the prior's solves, all three the small
levels of an algebraic hierarchy on that many ranks, and the wait of every assembly for
the slowest card (0.04 to 0.08 s a Jacobian in two runs, 0.5 s in a third).  A prior with
a lumped mass matrix (``lumped_mass``, :doc:`priors`) takes another 1.1 s off the 19.4 s
on one GPU, 1.0 s off the 24.9 s on 64 and 0.8 s off the 26.5 s on 512: the mass solves
of its products, a cost of every size alike.

Three settings more, with that prior (the fourth series of 11 October 2026): the
refinement of a forward or an adjoint solve ended at a relative residual of 1e-9
(``single_refine_goal``: two passes and one residual in double precision instead of three
and three), and eight interpolation entries a row instead of six in the two hierarchies
of linear elements, the p-multigrid's (``HIPPYMFEM_SINGLE_AMG=pmax=8``) and the prior's
(its solver's ``amg_options``).  The p-multigrid's iteration count follows the strength
of its one coarse cycle closely (aggressive coarsening there takes it from 7 to 11), and
that cycle weakens with the size: with eight entries the count goes from 6.5 on one GPU
to 7.1 on 4096 where it went from 6.8 to 8.9.  The operations then take 16.7 s on one
GPU, 20.3 s on eight, 22.5 s on 512 and 24.6 s on 4096, and the two steps at
1920\ :sup:`3` 19.8 s: 3.4 times less than double precision, 68 % of the rate of one GPU
and 82 % of that of one node.  The percentages of the double-precision series are higher
(76 and 88 %) and its losses larger: 20.3 s between one GPU and 4096 against 7.9 s here,
of a single-GPU time 3.8 times as long.  What is left is spread over every level of the
two hierarchies, 90 to 100 microseconds for each exchange of each level (a kernel that
gathers, a copy from the card, the messages, a copy back) and nine or ten levels on 512
to 4096 GPUs;
continuing a hierarchy on the host below some size, by a patch to hypre, was tried and
gains nothing there, since the small levels cost messages between many ranks either way.
``interp=18``, the matrix-based form of the same interpolation, sets these hierarchies up
a seventh faster with the same iterations (0.3 s of the 21.4 s on 64 GPUs).

The last levels of such a hierarchy can be gathered (``seq_threshold``, a BoomerAMG option
by its name: the hierarchy ends at its first level of at most that many rows in all, and
one process finishes it).  On 512 GPUs the levels below 83 000 rows have 7 300, 700, 75
and 9 rows, a few rows a rank that talk to many ranks at four exchanges a cycle each;
gathered with ``seq_threshold=30000`` they are one collective and a hierarchy of one
process, and an iteration of CG with the p-multigrid's linear-element hierarchy takes
7.1 ms instead of 8.8 ms there (6.3 instead of 7.3 ms on 64), a solve of the prior
0.088 s instead of 0.101 s, in the same iterations and with the same set-up.  A level
of 83 000 rows gathered takes no less a cycle and 0.2 to 0.6 s more a set-up, since
every rank receives the level's matrix, and one of a million rows twice the cycle.
hypre's own code for this reads device memory through host pointers;
``tools/hypre-3.2.0-seqdevice.patch`` (``HYPRE_SEQ_DEVICE=1 tools/build_pymfem_hip.sh``)
has the device's, and without it the library refuses the option on a device.  The first
of the ranks that hold rows of the level solves it and sends the solution to the others;
with ``redundant=1`` every one of them solves it and keeps its own rows, which saves
that broadcast.  The second is one linear map only where the ranks' solvers are the same,
and in the first version of the patch they were not: hypre's coarsening on a device draws
its random numbers from one generator a process, which stands wherever the process's
earlier hierarchies left it, so every rank built another hierarchy of the same matrix
(three to five levels), the ranks' solutions of one system differed by 3 to 41 %, and the
pieces did not fit at the seams.  CG then took 78 iterations where it takes 7 on the
smooth right-hand side of an incremental forward solve, or stalled, while a random
right-hand side showed nothing.  The patch now sets the generator back before each rank's
set-up and compares the solvers on one right-hand side, and the first rank solves where
they differ (:func:`~hippymfem.common.mfemconfig.seq_solvers` says which it was); the
cycle is symmetric to rounding either way, which ``test_device`` checks.  Its third
version (``hypre_BoomerAMGSeqOnDevice()`` returns 3) closes two cases that hypre's own
code has in host memory as well.  hypre's set-up also ends at a level on which the
coarsening finds no coarse point (every row diagonally dominant: a mass matrix, or a
prior at the level whose mesh width passes the correlation length), and it gathered
that level whatever its size, the whole of it on every rank that holds rows, with 32-bit
counts; such a level is now relaxed, as it is without the setting.  And the gathered
solver of a system (``systems_dim``) was told the number of functions and not which row
is which, so it paired rows of different functions below the first level (34 CG
iterations where the hierarchy takes 12, on vector diffusion); it is handed the level's
own labels.  A hierarchy of one equation that coarsens to its threshold, as every one of
the scaling runs does, is the same in the second and the third version.

Each CG iteration of a Newton step applies the prior's covariance as its preconditioner:
two solves with the prior's operator, by CG to 1e-6 (``cg_preconditioner_tolerance``), 10
iterations on one GPU and 13 on 4096 at 4.4 and 10.1 ms, which is 0.09 s and 0.26 s a CG
iteration and was the largest single loss to the size that was left.  With
``cg_preconditioner_cycles=5`` (:doc:`optimization`) each solve is five multigrid cycles
under a fixed polynomial, one linear map by construction: 0.07 s a CG iteration on 64
GPUs instead of 0.16 s, and the cost functional after two steps is that of the Krylov
solves to 4e-4.  With four cycles it was 1.3 % from it and with three 4 %, with a
gradient half as large again, although the polynomial's bound is 0.3 and 1.4 % there: the
Newton steps are CG stopped after four iterations, and such an iterate moves with the
preconditioner.
With these three, the gathered levels, ``interp=18`` and five cycles for each of the
prior's solves (the fifth series of 11 October 2026), the operations take 16.7 s on one
GPU, 19.6 s on eight, 21.6 s on 512 and 23.3 s on 4096, and the two steps at
1920\ :sup:`3` 18.6 s: 72 % of the rate of one GPU and 84 % of that of one node, 88 % from
64 GPUs on, the first size whose ranks have neighbours on every side.  The 6.6 s that
4096 GPUs lose against one are 3.0 s of Krylov iterations of the Jacobian's solves (an
iteration 40.5 ms against 31.2 ms, and a tenth more of them), 1.1 s of the five set-ups
of the p-multigrid's hierarchy, 0.6 s of the preconditioner, and the assemblies; three of
them are lost by the eighth GPU of the first node, where every product gains an exchange
and the set-up its parallel form.
Two things in that series were not hypre's.  A vector's arithmetic runs where the vector
was last used, on the device once it has been handed to MFEM or hypre and on the host
until then, with every operand that is current on the device copied back first; the CG
of a Newton step never handed its iterate over, and with the polynomial preconditioner,
which copied its right-hand side instead of passing it on, not its residual either.  Its
updates and inner products then ran on the host: 26 ms a CG iteration on one GPU and
87 ms on 4096, where host arithmetic has a long tail over the ranks.  And hypre's PCG
begins a solve with the preconditioner applied to the right-hand side, for its norm, a
product with the guess, and the preconditioner applied to the residual, which from the
zero guess of these solves is the same vector: one p-multigrid cycle and one product of
the Jacobian in 32 solves that did nothing.  With the CG's vectors handed to MFEM at
the start, the polynomial's cycles applied from the iterate (no product beside them),
the first cycle of a solve kept for the second and the product of a zero guess left out,
the operations take 15.2 s on one GPU, 18.2 s on eight, 18.9 s on 64 and 19.8 s on 512
(16.7, 19.6, 20.4 and 21.6 s in the series), with the same cost functional.
The series in that tree (the sixth of 11 October 2026) has 15.4 s on one GPU, 18.1 s on
eight, 19.9 s on 512 and **20.9 s on 4096**, and the two steps at 1920\ :sup:`3`, 120.4
billion unknowns, take **16.4 s**: 3.8 to 4.2 times less than double precision at
every size, 74 % of the rate of one GPU and 87 % of that of one node, 90 % from 64 GPUs on (double
precision: 76 and 88 %).  The single GPU of that series' own job took 15.9 s, 4.5 % more
than in two other runs, by which measure the 4096 keep 76 %: one card's time scatters
by that much, and every percentage here with it.  Between one GPU and 4096 the
operations now lose 5.5 s where the double-precision ones lose 20.3 s: 2.1 s in the
Krylov iterations of the Jacobian's solves (35.1 ms against 27.8 ms, and 225 against
210), 1.1 s in the five set-ups of the p-multigrid's hierarchy, 1.0 s in the five
assemblies, 0.4 s in the Hessian's blocks and 0.3 s in the preconditioner.
An iteration of those solves was timed operation by operation on one GPU, in the tree
before the first cycle of a solve was kept: a cycle of the p-multigrid is 15.2 ms, of
which 9.1 ms the product with the Jacobian, 2.3 ms the product with the Jacobian times
the interpolation, 2.3 ms the cycle of the linear elements (4.1 ms on 64 GPUs), 0.5 and
0.7 ms restriction and interpolation and 0.2 ms each relaxation; the PCG's own product,
inner products and updates are 11.5 ms.  The two products with the assembled matrix,
0.9 billion entries each, are two thirds of an iteration and run at the card's memory
bandwidth, and assembling that matrix (0.85 s, five times) is another quarter of the
operations.  An operator applied element by element would spare most of the assembly;
whether it also applies faster than the matrix does at this order has not been measured
here.
Eight 64\ :sup:`3` blocks of elements a GPU (17 million state unknowns each), which the
eight GPUs of a node do not hold in double precision, fit in the mixed mode:
256\ :sup:`3` on eight GPUs, 287 million unknowns, in 57 s for two steps, with 6 GiB of
each card left and 50 GiB of host memory a rank.  Name the library together with
``HIPPYMFEM_PRECISION=mixed``:
from element matrices in double precision the Jacobian is rounded into it entry by entry
and not corrected on the constants, and at 102\ :sup:`3` on one GPU the cost functional
after two steps was 1.6 % from the double-precision one, where the mixed mode was 3e-4
from it.

Eight blocks a GPU are also the largest load that fits in the configuration of the last
series, and a larger load would not raise the percentages above.  With these
128\ :sup:`3` elements a GPU (2.1 million, a fifth more than the 120\ :sup:`3` of the
series) the same operations
take 19.0, 21.6 and 23.2 s on 1, 8 and 64 GPUs: 88 and 82 % of one GPU, where
120\ :sup:`3` keeps 85 and 82 %, and the three points on eight GPUs are the line
searches of those runs and not the load.  By element every operation costs what it
does at 120\ :sup:`3`: an iteration of the Jacobian's solves 16.0, 17.6 and 18.4 ns an
element against 15.9, 17.7 and 18.5, and only the set-up of the hierarchy 3 to 6 %
less.  What the larger sizes lose grows with the load, then.  In double precision it
was the same from a million elements a GPU: with one, two and four blocks a GPU 256
GPUs kept 61, 71 and 80 % of one, and after that four blocks and 120\ :sup:`3` kept 82
and 79 % on 512.  At 128\ :sup:`3` a card has 2 GiB of its 64 free at the worst moment
(16 GiB at 120\ :sup:`3`) and a rank 42 GB of host memory.  136\ :sup:`3` and
144\ :sup:`3` run out of device memory: with JAX's share of the card at 0.45 after the
first forward solve, in hypre's pool or in an allocation of MFEM, and with 0.30 in the
first assembly, where JAX cannot have the 11.4 GiB of the Jacobian's accumulator in one
piece from an arena of 19 GiB.  152\ :sup:`3` on the eight GPUs of a node runs out of
the node's 512 GB (a rank takes about 8 GB and 16 kB an element).  Past both, the
Jacobian of n\ :sup:`3` elements on a rank has (8 n + 1)\ :sup:`3` entries, more than
the 2\ :sup:`31` that the library counts a rank's entries with from n = 162.  On the
card the Jacobian is held more than once: its values and column indices in the library
(8 bytes an entry, 8 GiB at 128\ :sup:`3`) and the accumulator of its assembly in JAX,
which is in double precision where the row sums are kept
(``HIPPYMFEM_SINGLE_ROW_SUMS``, 8 bytes an entry again) and which JAX keeps between
assemblies once it is a quarter of its arena (``HIPPYMFEM_FUSED_KEEP_SHARE``: not at
120\ :sup:`3`, where JAX holds 2 GiB between assemblies, and at 128\ :sup:`3`, where it
holds 10 GiB).

What the ranks of a run differ by was measured on 12 October, with every rank's own times
gathered after the steps (``bench_newton_device.py --rank-table``).  The longest element
pass of a rank (0.74 s) varies with a standard deviation of 1.2 % on 64 to 4096 GPUs, the
slowest rank is 3 to 6 % behind the median, and every rank waits for it at the collective
after an assembly: 41, 51 and 66 ms on 64, 512 and 4096 GPUs.  It is not slow nodes: the
same benchmark on the eight nodes ranked fastest, on the eight slowest and on eight around
the median took 18.97, 18.89 and 18.95 s.  It is the cards.  The two GPUs of an MI250X
are slow or fast together (correlation 0.69 on 4096 GPUs; four fifths of the variance
between ranks is between cards), 4 % of the cards are more than 2 % behind the median
and the slowest 5 %, and a card's time in that pass follows neither its time in a shorter
pass nor two kernels timed before the run.  A run on more ranks waits for the slowest of
more cards, and with equal boxes it cannot give that card less.

What does grow with the ranks in the code's own hands is BoomerAMG's hierarchy of the
linear elements: 0.28 s a set-up on one node and 0.35 to 0.45 s on 512 to 4096 GPUs, an
iteration of the Jacobian's solves from 30.5 ms on one node to 35.1 ms on 4096, and one
level more for every eight times the ranks, the newest of a few rows a rank and with
its exchanges all the same.  With the geometric levels in its place (:ref:`described above
<gpu-pmg-h>`; the mesh from one element a rank by cuts in 5, 3, 2, 2, 2 a side,
``HIPPYMFEM_PMG_H=7``) a
set-up is 0.07 s on one GPU, 0.12 to 0.13 s on 8 to 512 and 0.16 s on 4096 (0.38 s
there), a forward linearized solve takes 4.1 iterations on 4096 GPUs where it took 7.0
and the same 0.17 s from one node on, and the operations of two Newton-CG steps take
15.0, 17.0, 17.8, 18.3 and 19.4 s on 1, 8, 64, 512 and 4096 GPUs where the series above
takes 15.4, 18.1, 18.9, 19.9 and 20.9 s (the seventh series, one job on 512 nodes).
1920\ :sup:`3` on 4096 GPUs, 120 billion unknowns: two steps in 15.3 s, 77.5 % of one GPU
and 87.6 % of one node (the series above 73.7 and 86.7 %, double precision 75.9 and
87.8 %), 91.6 % from 64 GPUs; 960\ :sup:`3` on 512 GPUs keeps 82 and 93 %.  The levels
are the same on any number of ranks, with a gathered matrix of 729 rows on 512 GPUs and
of 4913 on 4096.  What still grows with the ranks is the time of an iteration (one more
costs 24 ms on one GPU, 27 ms on a node and 32 ms on 4096: the cycle has 38 products with
an exchange between ranks, 20 of them on its two coarsest meshes, and the adjoint
linearized solves keep their seven iterations), the wait of an assembly for the slowest
card, and the prior's preconditioner (0.05 s a CG iteration on one node, 0.09 s on 4096),
whose cycles are BoomerAMG's.

With the small levels in host memory and the prior's cycles over the meshes
(``HIPPYMFEM_PMG_H_HOST=5000``, ``HIPPYMFEM_PRIOR_H=7``; :ref:`described above <gpu-pmg-h>`; the eighth
series, one job on 512 nodes that afternoon) the operations take 17.0, 17.7, 17.7 and
19.0 s on 8, 64, 512 and 4096 GPUs: 3.0 % less on 512 and 1.9 % less on 4096, where two
steps at 1920\ :sup:`3` take 15.1 s and keep 89 % of one node (88 % above).  An
iteration of the Jacobian's solves is 1.2 ms shorter where the gathered matrix has 729
rows (128 and 512 GPUs) and 0.6 to 0.7 ms where it has 4913 (1024 to 4096); the solve of
that matrix itself takes 0.42 ms in host memory and 0.47 ms on the device, so the
difference is not there, and is not yet accounted for.  The prior's preconditioner takes 0.034 s a CG iteration on one node and 0.076 s on 4096
(0.052 and 0.089 s).  That series' percentage against one GPU, 81 %, does not compare
with the 77.5 % above: its single GPU took 15.42 s, 2.6 % more than the one before it at
the same settings but for the prior's; against that earlier run the 4096 keep 79 %.
In the same configuration 2048\ :sup:`3` elements ran on 4096 GPUs, 128\ :sup:`3` a
GPU and 146 billion unknowns: two Newton-CG steps in 15.2 s, the operations in 23.0 s,
82 % of one GPU and 88 % of one node at that load, with 1.0 GiB of a card left at the
tightest.  The largest load, 128\ :sup:`3` elements a GPU
by seven cuts in two, takes 18.4, 20.4, 21.5 and 22.5 s on 1, 8, 64 and 512 GPUs
(19.0, 21.6 and 23.2 s on the first three with BoomerAMG on the linear elements):
1024\ :sup:`3` on 512 GPUs, 18.3 billion unknowns, at 82 % of one GPU and 91 % of one
node.  The single GPU that the seventh series' percentages are against took 15.03 s,
sharing its node with the runs on two and four GPUs; the same run took 14.85 s alone on a
node and 15.53 s in an earlier job with the same sharing, so a percentage against one GPU
is known to 3 points (the single-GPU runs differ in the pass that the cards differ in).
The measure of the tsunami work that won the Gordon Bell
prize of 2025 (arXiv 2504.16344) is a different one: one explicit time step, without a
linear solve, against the smallest run of 85 nodes, 92 % on 128 times the GPUs.  By that
kind of measure the operations here keep 90 % from 4 nodes to 512 (128 times the GPUs),
92 % from 8 nodes and 94 % from 64; the assemblies and the gradient, which like that step
have no solve, 95 to 96 % from 8 nodes and 97 to 99 % from 64.

One failure on these cards is below the library.  With ROCm 7.2.0, ``test_gpu`` on eight
ranks ends in a segmentation fault about one run in four (9 of 39 runs on two nodes of
Frontier on 11 October 2026, a tree of the day before among them): a rank dies inside the
runtime (``libhsa-runtime64``, under the ``hipStreamWaitEvent`` by which a JAX program
waits for a transfer on another stream), in another test each time and with no check
failed, and once the runtime reported a malformed packet of a queue instead and the rank
hung.  That suite runs JAX alone on a card, with thousands of small transfers; with hypre
on the card, in ``test_device`` and in the benchmarks, it has not happened.  A run of the
suite that ends without its count of failures is to be read with this, and run again.

Going further
-------------

``tools/build_pymfem_cuda.sh`` has the CUDA build recipe, including the upstream problems
it works around.  :doc:`configuration` lists every setting, :doc:`performance` has the
host-side picture, and ``benchmarks/DESIGN_NOTES.md`` explains the design decisions
summarized here.

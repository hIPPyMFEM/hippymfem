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
  solve was 1.17 times faster on an H100 and 1.45 times on an L40S, and at
  128\ :sup:`3` on four L40S 1.57 times, with 14 % less memory on the busiest card
  (:ref:`single-precision`).  The single-precision solves need a Jacobian that is
  symmetric and solved by CG with BoomerAMG; other problems keep their double-precision
  solves.
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
eigensolver took 17.6 s on four MIG instances as an ensemble and 30.4 s as a domain
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
65,683 MiB in double precision with the release); ``HIPPYMFEM_GPU_MEM_FRACTION=0.20``, which saved 1.9 GiB a card for 13 %
more time in two Newton steps at 128\ :sup:`3` on four L40S; matrix-free linearization
points last (:ref:`gpu-memory`).

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
64\ :sup:`3`    4.6 M      1 L40S                       1.19 s   0.47 s   0.59 s   13.6 s
64\ :sup:`3`    4.6 M      1 H100                       0.54 s   0.31 s   0.23 s   6.1 s
64\ :sup:`3`    4.6 M      1 AMD MI210                  1.02 s   0.37 s   0.46 s   11.2 s
64\ :sup:`3`    4.6 M      4 L40S                       0.45 s   0.13 s   0.26 s   5.7 s
64\ :sup:`3`    4.6 M      4 host ranks                 12.5 s   1.5 s    10.9 s   179.5 s
64\ :sup:`3`    4.6 M      hIPPYlibx, 4 host ranks      29.4 s   8.7 s    13.8 s   402.5 s
128\ :sup:`3`   36 M       1 H100                       5.6 s    2.4 s    1.6 s    42.9 s
128\ :sup:`3`   36 M       4 L40S                       2.9 s    0.97 s   1.6 s    29.1 s
128\ :sup:`3`   36 M       4 AMD MI210                  2.5 s    0.82 s   1.2 s    23.6 s
128\ :sup:`3`   36 M       4 host ranks                 108.5 s  11.9 s   101.3 s  1409 s
128\ :sup:`3`   36 M       hIPPYlibx, 4 host ranks      241.0 s  71.4 s   140.7 s  2669 s
256\ :sup:`3`   287 M      8 Blackwell cards            10.7 s   5.2 s    3.5 s    101 s
256\ :sup:`3`   287 M      16 Blackwell cards           4.3 s    2.6 s    1.8 s    46.3 s
256\ :sup:`3`   287 M      32 host ranks                --       --       --       2859 s
256\ :sup:`3`   287 M      hIPPYlibx, 32 host ranks     423.9 s  102.4 s  378.9 s  6187 s
400\ :sup:`3`   1.09 B     24 Blackwell cards           13.1 s   6.4 s    4.8 s    134 s
==============  =========  ===========================  =======  =======  =======  ==========

The GPU rows were measured in October 2026 on a cluster: the four L40S are two on each
of two nodes, and the Blackwell cards (RTX PRO 6000) were split into two 48 GB MIG slices
each, one rank per slice, so 16 cards are 32 ranks.  The host and hIPPYlibx rows were
measured in September 2026 on one node with two AMD EPYC 9334, with as many ranks as the
GPU rows beside them: they are a device swap at a fixed rank count, not the node's best
host time.  Against host ranks of the same library the GPUs are 31, 48 and 62 times
faster at the three sizes, with the same cost functional to eight digits and the same CG
counts.

The hIPPYlibx rows are the same problem (mesh, spaces, PDE, data model, prior,
Newton-CG settings) solved by `hIPPYlibx <https://github.com/hIPPyMFEM/hippylibx>`_ on
dolfinx 0.10, with PETSc's CG and BoomerAMG given the same BoomerAMG options.  The two
libraries draw different random data, so their CG counts differ at the two smaller
sizes (11 and 6 against 8): four cards are 70 and 92 times faster as measured, and 63
and 101 times when hIPPYlibx is charged the same CG counts.  At 256\ :sup:`3` every run
takes 9 CG iterations, and the GPUs are 134 times faster.

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
a warning.  ``fp32`` puts the vector
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
Jacobian is symmetric; any other problem keeps its double-precision solves.  On a GPU it
is CUDA only: with a HIP build (AMD GPUs) the library is refused with a warning and the
solves stay in double precision.

The BoomerAMG of that library takes MFEM's defaults for a device, and
``HIPPYMFEM_SINGLE_AMG`` sets any of hypre's BoomerAMG options by name
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
of the 17.5 GiB of the table (double precision unchanged; the other columns not yet
measured again).  The other blocks of a linearization point stay in
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
per card, 2.5 to 5 ms at 2.1 million.  So at 32\ :sup:`3` four L40S are 1.35 times
faster than one and at 64\ :sup:`3` 2.4 times, while at 128\ :sup:`3` four are twice as
fast as two, and sixteen Blackwell cards at 256\ :sup:`3` 2.2 times as fast as eight.

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
  it is returned by itself when the card runs out of memory.  CUDA builds only.
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
31 GiB, or on one 80 GB H100 at 59 GiB; a Newton-CG run to its tolerance peaked there at
64.1 GiB in double precision (253.8 s) and at 58.4 GiB with single-precision element
matrices and solves (197.7 s), both with ``release_linearization_on_move``.  At
400\ :sup:`3` a 48 GB slice holds 37 GiB.  What decides whether a given run fits:

**JAX's share of the card.**  The share (0.45 of the card with hypre on it) is a
ceiling, not a reservation: the pool grows as the element kernels need it, and their
batches are sized from the room left under the ceiling, so a lower share means smaller
chunks and a smaller pool, at some cost in time.  Set
``HIPPYMFEM_GPU_MEM_FRACTION=0.20`` when a run is short of card memory: at
128\ :sup:`3` on four L40S it saves 1.9 GiB per card for 13 % more time.
``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` is not a substitute: it reports no budget to
size the chunks from, and its card peak was higher.

**Matrix-free linearization points.**  ``setLinearizationPoint(x, matrix_free=True)``
leaves ``C``, ``W_um`` and ``W_mm`` unassembled and computes their products at every
Hessian application instead: 0.6 GiB per card for 4 % more time in the same run.  With
the lower share as well it is 2.7 GiB for 64 %, so it is for a run that still does not
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
compiles them, in three seconds.  In an assembly on the card
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
eigensolver (140 Hessian applies)        199.1 s                      10.1 s                28.4 s                174.5 s
one posterior sample                     165 ms                       24 ms                 46 ms                 274 ms
pointwise variance, randomized, r = 64   4.6 s                        1.6 s                 4.4 s                 15.3 s
pointwise variance, Monte Carlo, n = 64  --                           1.2 s                 2.3 s                 14.0 s
traces, r = 64                           5.8 s                        1.9 s                 4.9 s                 17.1 s
=======================================  ===========================  ====================  ====================  =====================

The random stream is a vectorized Philox generator, bit-identical to numpy's and on the
card when the kernels are, which is what makes a sample tens of milliseconds.  The
``"Randomized"`` pointwise variance is a truncated spectrum and 20-25 % low at r = 64;
the ``"MonteCarlo"`` method (one solve per sample) is unbiased.  The incremental and
prior solves can run at 1e-8 for these stages: the eigenvalues move by 6e-6 at most and
the eigensolver is 1.4x faster.  At 128\ :sup:`3` the whole workflow, MAP included,
runs in 14.4 minutes on four L40S (two on each of two nodes, as above).

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
eigensolver (k = 50, p = 20)                62.0 s    21.2 s               9.9 s
64 posterior samples                        5.7 s     2.7 s                0.9 s
pointwise variance, Monte Carlo, n = 64     4.2 s     2.2 s                0.8 s
all stages                                  83.5 s    37.2 s               17.9 s
==========================================  ========  ===================  ===========

The eigenvalues of the ensemble agree with those of one slice to 2e-6.  What an
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
:math:`64^{3}`, one GPU                      11.2 s  6.1 s   13.6 s
:math:`128^{3}`, four GPUs                   23.6 s  --      29.1 s
===========================================  ======  ======  ======

The answers are the NVIDIA ones: the same cost functional to nine digits and the same CG
counts at every size.  ``run_tests.sh 1 2`` with the HIP build passes every suite,
``test_gpu`` and ``test_device`` included.  The recycling pool for hypre's memory is for
CUDA builds only.

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

Then ``HIPPYMFEM_HYPRE_DEVICE=1`` puts MFEM on ``mfem.Device("hip")``:
:func:`~hippymfem.common.mfemconfig.configure_device` reads the backend from the build,
so ``"cuda"`` and ``"gpu"`` in a script mean HIP on this build and nothing written for
NVIDIA changes.  Rank pinning, in the library and in ``tools/mpirun_pinned.sh``, writes
``ROCR_VISIBLE_DEVICES`` on an AMD node.

Going further
-------------

``tools/build_pymfem_cuda.sh`` has the CUDA build recipe, including the upstream problems
it works around.  :doc:`configuration` lists every setting, :doc:`performance` has the
host-side picture, and ``benchmarks/DESIGN_NOTES.md`` explains the design decisions
summarized here.

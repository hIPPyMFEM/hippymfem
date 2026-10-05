# Changelog

## Unreleased

### Added

- **Single precision** in three places, none of which changes the state, the gradient or
  the MAP point (`docs/source/guide/gpu.rst`, "Single precision").
  `HIPPYMFEM_PRECISION=mixed` computes the element matrices in single precision, corrects
  the Jacobian's to act on the constants as the double-precision ones do, and refines
  the forward and adjoint solves against double-precision residuals: the Jacobian's
  kernel is 1.7 (H100), 2.7 (L40S) and 5.3 (Blackwell instance) times faster, a complete
  assembly 1.2, 2.3 and 4.2 times. `HIPPYMFEM_HYPRE_SINGLE=<libHYPRE_single.so>`
  (`tools/build_hypre_single.sh`, `hippymfem.algorithms.singlesolve`) loads a
  single-precision build of hypre next to the double-precision one: the Jacobian is
  assembled into it and exists there alone, with its BoomerAMG hierarchy, the forward
  and adjoint solves are refined against double-precision residuals to the solver's
  tolerance, and the incremental solves of a Hessian action stop at 1e-5 and are used
  as they are. A CG iteration is 1.16 (H100), 1.31 (L40S) and 1.37 (Blackwell instance)
  times faster, a Hessian action 1.4 to 1.6 times, a BoomerAMG setup no faster, and the
  card holds 1.4 GiB less at 2.1 million state dofs (18.9 -> 17.5 GiB on the H100).
  With both, the Newton-CG run above takes 24.5 s on the H100, 46.3 s on the L40S,
  45.4 s on a Blackwell instance and 26.1 s on four: 2.6, 3.5, 3.6 and 3.1 times faster
  than before these three changes; with the forward and adjoint solves refined to 1e-9
  only (`PDEVariationalProblem.SINGLE_REFINE_GOAL`, enough for Newton-CG to 1e-6 but
  not for BFGS to 1e-8) 23.5, 43.5, 42.4 and 25.6 s.
  At 17.0 million state dofs on four L40S the run takes 162 s against 240 s in double
  precision, in the same 13 Newton and 191 CG iterations, and the busiest card holds
  34.5 GiB against 39.2 GiB; a run that releases each linearization point before the
  next holds more with the single-precision solves (21.9 to 23.4 GiB against 19.7 GiB
  on the busiest of four Blackwell instances). A failure that one rank meets alone
  (hypre's error flag is per process) stops every rank: before, a BFGS line search on
  two ranks could wait forever. The Jacobian of the single-precision library is
  accumulated in single precision where the elements are assembled in chunks
  (`HIPPYMFEM_SINGLE_ACCUMULATE`, on): half the largest allocation of an assembly and no
  rounded copy of it. In the two Newton steps at 128^3 on four Blackwell instances that
  release each linearization point the busiest instance held 18.3 GiB with
  single-precision kernels and solves against 20.4 GiB in double precision (21.9 GiB
  with the accumulator in double precision and a rounded copy, as before).
  The single-precision solves run with CUDA builds (H100, L40S, RTX PRO 6000 Blackwell)
  and host builds; with a HIP build (MI210) the library is refused before it is loaded,
  with a warning, and the solves stay in double precision (`singlesolve.unsupported`).
  A non-symmetric Jacobian (GMRES, `transpose_free_adjoint`) keeps its double-precision
  solves. With single-precision element matrices or solves `symmetric_jacobian="auto"`
  keeps the verdict of its first probe; a residual symmetric at the first parameter and
  not later got the Jacobian for its transpose, without an error: on an
  advection-diffusion test the gradient was off by 3e-5 (mixed element matrices) and
  1e-2 (single-precision solves). With the solves alone the verdict is now checked on
  every new Jacobian (four products in single precision); the symmetrized matrices of
  the mixed mode cannot be probed, and such a residual has to declare
  `symmetric_jacobian=False` there. A refined forward or adjoint solve that ends far
  above its goal (`PDEVariationalProblem.REFINE_STALL`: above 1e3 times the goal and 1e-8
  of its first residual) is solved once more with the Jacobian assembled in double
  precision, with a `RuntimeWarning` once per process, where it used to return what it
  had: single precision cannot carry a solve at a parameter that makes the Jacobian too
  ill-conditioned (BFGS from the prior mean in `test_optimization` passes m in [-50, 52],
  where a single-precision forward solve diverged and an adjoint solve stopped at 0.84
  of its first residual), and the double-precision Jacobian drops a kept verdict of
  symmetry that it contradicts.
  `HIPPYMFEM_PRECISION=fp32`, everything in single precision, remains a tool for
  experiments: the optimizers now stop at the floor of that precision instead of
  failing in a line search.
- **Newton-CG can set the refinement goal of the single-precision solves**
  (`single_refine_goal` of `ReducedSpaceNewtonCG`, off by default; at most 1e3 times the
  square of its tolerance, so a run to 1e-8 is unchanged): with 1e-9 the forward and adjoint
  solves stop after two passes instead of three. That is 3 to 5 % less time at 64^3 with the
  same counts. At 128^3 on an H100 it is 187 s instead of 199 s while the iteration keeps
  its thirteen steps, and in one solve of five the last line search backtracked and a
  fourteenth step followed (223 s); with full refinement every solve took the same path,
  which is why that is the default. The problem's own `SINGLE_REFINE_GOAL` stays 0 too. **The
  stages after the MAP point may keep the single-precision solves**
  (`PDEVariationalProblem.set_single_solves`): at 64^3 the eigenvalues came out within 1e-5
  of double precision and the pointwise posterior variance within 6e-6, and the eigensolver
  took 26.0 s instead of 52.3 s. **BoomerAMG options of the single-precision solves**
  (`HIPPYMFEM_SINGLE_AMG`, opt-in) and the relaxation of MFEM's BoomerAMG
  (`HIPPYMFEM_AMG_RELAX`, opt-in): `relax=7,pmax=6` took Newton-CG at 64^3 from 19.8 to
  15.7 s on an H100 with the same counts; checked on the model problem only.
- **Independent solves as an ensemble over the GPUs**: `ensemble=` on `MatMvMult`, the
  randomized eigensolvers and the `trace` and `pointwise_variance` of the prior and the
  posterior. When the problem fits on one GPU, every rank builds it on `MPI.COMM_SELF`
  and the columns or the Monte Carlo samples are divided among the ranks; every rank
  returns what one rank alone computes (`benchmarks/bench_laplace.py --ensemble`).
- **Matrix-free linearization points**: `setLinearizationPoint(x, matrix_free=True)`
  assembles the Jacobian alone and takes the products with the other blocks from the
  element kernels, which saves their memory. `PDEVariationalProblem.apply_ij_at(i, j, x,
  dir, out)` applies a second-derivative block at any point without assembling it.
- `HIPPYMFEM_HESSIAN=quadrature` (`hm.config.hessian`): element Hessian blocks from the
  density's second derivatives at the quadrature points, the faster route on a GPU from
  P2 up. The default stays `element`, whose results are reproducible bit for bit.
- `PDEProblem.apply_third_dir`: the weighted second directional derivatives of a gradient
  along several directions in one kernel pass.
- `TimeDependentPDEVariationalProblem` takes a boundary density (`bdr_varf`).
- `HIPPYMFEM_DEVICE=auto`: the element kernels take a GPU when the process can see one
  and the host otherwise.
- A problem with an interior-facet density checks once that the density is unchanged when
  the two sides of a face trade places, and warns if it is not
  (`hippymfem.fem.facets.check_facet_symmetry`).
- `tools/install_petsc_mumps.sh` builds PETSc with MUMPS and petsc4py against it, so
  `PETScLUSolver` factorizes in parallel. `tools/rebuild_hypre.sh` rebuilds hypre with
  page-locked staging buffers or for a CUDA-aware MPI.
- Tests for the exported classes that no suite exercised: the multiplicative-noise and
  multi-state misfits, the vector and mollified priors, importance sampling, the full
  tracer, steepest descent, and the lumped-mass and transpose solvers.

### Changed

- **Newton-CG keeps the residuals of its CG orthogonal explicitly** (`cg_reorthogonalize`
  of `ReducedSpaceNewtonCG`, on by default; `reorthogonalize` of `CGSolverSteihaug`). The
  recurrence of CG loses that orthogonality to rounding on a prior-preconditioned
  Hessian, and to the error of the incremental solves when these are inexact: the model
  problem with 2.1 million state dofs took 193 to 210 CG iterations for its twelve Newton
  steps, depending on the GPU and the rank count, and a third more with incremental
  solves stopped at 1e-8. With every residual made orthogonal to the earlier ones it
  takes 131 on every GPU and rank count, and still 131 with incremental solves stopped
  at 1e-6, which halves their iterations. One H100: 63.8 s -> 47.4 s -> 32.0 s with
  incremental solves to 1e-6; one L40S 161.5 -> 110.5 -> 73.3 s; four MIG instances of
  two RTX PRO 6000 Blackwell 81.7 -> 39.0 s. The cost functional agrees to nine digits.
  `cg_reorthogonalize = False` gives hIPPYlib's iteration. **Iteration counts and times
  of a Newton-CG run to its tolerance change with this default.** The benchmark of two
  Newton-CG iterations (`bench_newton_device.py --steps 2`) keeps its eight CG
  iterations at 64^3 and takes 13.6 -> 11.5 s on an L40S with hypre's PCG below and the
  device geometry of the chunked assemblies.
- **CG with a hypre preconditioner runs in hypre's own PCG** (`HIPPYMFEM_HYPRE_PCG`,
  `hm.config.hypre_pcg`, on by default). The iteration is the same as MFEM's `CGSolver`
  from a zero initial guess. hypre's PCG marks the vector it hands to BoomerAMG as zero,
  which saves the first relaxation of a V-cycle its matrix-vector product on the finest
  level: a solve of 2.1 million dofs to 1e-12 in 24 iterations took 0.103 -> 0.089 s on
  an H100 and 0.274 -> 0.227 s on an L40S; the Newton-CG run above 32.0 -> 29.7 s and
  73.3 -> 64.8 s. The solvers of one operator share one PCG object.
- **Cheaper solves inside the CG of a Newton step** (`docs/source/guide/optimization.rst`,
  "The CG of a Newton step"). The prior's solves, where they precondition that CG, stop
  at 1e-6 (`cg_preconditioner_tolerance` of `ReducedSpaceNewtonCG`, the default; never
  more than a thousandth of the CG's own tolerance; `0` leaves them as they are; the
  line search with `cg_reorthogonalize` only): eleven iterations per solve instead of
  twenty-one on the model problem, the same Newton and CG counts and gradient norms. With
  `cg_hessian_relaxation = c` (off by default) the incremental solves of a Hessian action
  at CG iteration k stop at c times the CG's tolerance times |r_0| / |r_k| where that is
  looser than their own tolerance (`relax_operator` of `CGSolverSteihaug`,
  `ReducedHessian.set_accuracy`, `rel_tolerance` of `solveIncremental`); with c = 1e-2
  6.5 iterations instead of eleven, the same counts. Newton-CG to 1e-6 at 64^3 with
  single-precision solves: H100 25.3 -> 23.6 s with the first, 18.0 s with the second and
  the index arrays on the device (`HIPPYMFEM_DEVICE_PATTERN=1`); L40S 46.1 -> 43.2 ->
  32.4 s; in double precision on the H100 29.9 -> 27.6 s. **This default changes the times
  of every Newton-CG run, not its counts.**
- **A streamed geometry is copied to the device at the rate of the bus**
  (`HIPPYMFEM_PINNED_STREAM`, on, where the device bridge can be used). A group whose
  geometry is too large to stay on the card (`HIPPYMFEM_GEOMETRY_STREAM`) had JAX move
  each chunk's slice from a numpy array, through a staging buffer of its own: 3.4 to
  5.6 GB/s measured on an H100 (8.8 GB/s from JAX's pinned host arrays), against the
  55 GB/s of its PCIe 5 link. The geometry is now locked
  in RAM once (`devicebridge.pin`) and each slice is copied by the CUDA runtime
  (`devicebridge.host_to_jax`), of the arrays the kernel reads only. The element arrays
  are bit for bit the same. 128^3 on one H100 with single-precision solves: forward solve
  5.51 -> 2.86 s, adjoint 2.52 -> 1.25 s, gradient 1.05 -> 0.40 s, Newton-CG 326 -> 197 s
  in the same 13 Newton and 191 CG iterations.
- **The scatter map of a fused assembly in a compact form** (`HIPPYMFEM_COMPACT_PATTERN`,
  on): for every element row two base slots and for every entry one byte (two where an
  offset does not fit in seven bits) that picks one and adds an offset, 1.3 bytes an
  entry for quadratic hexahedra instead of 4. The slots are rebuilt in the scatter on the
  device and are the same, so the matrices are bit for bit the same on a CPU. It is a
  third of the map that an assembly uploads from the host slice by slice, and a third of
  what `HIPPYMFEM_DEVICE_PATTERN=1` keeps on the device. **The matrices of the
  single-precision library share the column indices of their pattern**
  (`HIPPYMFEM_SINGLE_SHARE_COLUMNS`, on): no upload of four bytes a nonzero per matrix
  (52 ms of a 187 ms assembly at 64^3 on an H100), and one copy for two matrices alive at
  once. With both the device pattern no longer grows the element kernels' arena by a
  region: at 128^3 on four Blackwell instances in the two-iteration benchmark the busiest
  instance held 26.2 GiB with it before and 18.0 GiB now, as much as without it (double
  precision: 19.9 GiB).
- **Newton-CG in double precision at 128^3 fits one H100.** With 2.1 million quadratic
  hexahedra on one rank (17.0 million state dofs) it ran out of the card's memory in its
  first Gauss-Newton step, at every share of JAX tried. A linearization point that needs one
  block (`C`, at a Gauss-Newton step) took the glued route of the element kernels: a split
  batch held every row block of the slot pass at once and all of it again while the chunks
  were joined, and the scatter of the joined arrays kept the pattern's full map on the
  device, 7.8, 7.8 and 1.7 GiB at that size. In single precision, which took the same route,
  that took the kernels' pool from 11.2 to 21.2 GiB in use and its arena to the cap; in
  double precision the 44 GiB needed outside the pool did not fit beside it. Such a block
  is now assembled a chunk at a time through the fused scatter, as the Jacobian and a full
  point are; an unsplit batch is computed whole, into the same arrays as before. 128^3 on
  one H100 at the default share, Newton-CG to 1e-6 with `release_linearization_on_move`:
  253.8 s in double precision, 13 Newton and 191 CG iterations, the card at 64.1 GiB at its
  peak (the kernels' pool 12.6 GiB in use); with single-precision solves 197.7 s as before,
  the card at 58.4 GiB instead of 74.0. 64^3 is unchanged within the noise of the H100
  (double 27.2 to 27.9 s, single 21.4 to 22.2 s, before and after). **An out-of-memory error
  inside a chunk loop is retried again.** JAX 0.11 reports an allocation that fails while a
  launch runs at the next synchronization, under the name of the program waited on; in the
  chunk loops that was the finiteness check of the element arrays (`kernel.all_finite`,
  where the failures above showed as `jit__reduce_all`), past the retry. The loops now wait
  for their results inside it, and the size of the failed request is read as JAX 0.11 words
  it.
- **Assembled matrices and vectors stay on the GPU** (`HIPPYMFEM_DEVICE_BRIDGE`,
  `HIPPYMFEM_DEVICE_VECTORS`). With the element kernels and hypre on one card, assembled
  values and the vectors between two hypre calls no longer pass through the host. Two
  Newton-CG steps of the benchmark take 29.1 s instead of 49.5 at 128^3 on four L40S and
  134 s instead of 191 at 400^3 on 24 Blackwell cards, with the same cost functional and
  CG counts.
- **hypre on several GPUs.** hypre multiplies with its own kernel on more than one rank
  of a CUDA build (`HIPPYMFEM_HYPRE_SPMV`), and recycles its device memory through a pool
  that is on by default (`HIPPYMFEM_HYPRE_POOL`, `HIPPYMFEM_HYPRE_POOL_KEEP`).
- One rank with hypre on a device assembles through the true-dof route
  (`HIPPYMFEM_TDOF_IDENTITY`).
- A boundary term adds its entries to a block only where it has any: a term that does not
  depend on the block's variables, as a prescribed flux in the Jacobian, no longer takes
  that block's assembly through the host on a GPU.
- The kernels of one mesh share one device copy of its geometry and of each space's
  tables.
- The sorts of a pattern build take bounded chunks of JAX's arena, which lowers the
  device memory a run holds afterwards.
- On a CPU the element kernels step through the batch 2 048 elements at a time
  (`HIPPYMFEM_HOST_BATCH`) and the element dof gather is compiled: large host assemblies
  are two to three times faster.
- `LUSolver` and `ReplicatedLUSolver` factorize on rank 0 and scatter the solution;
  `replicate=True` restores the factorization on every rank.
- The randomized eigensolvers orthonormalize after every power iteration, so more
  iterations sharpen the tail of the spectrum.
- `CGSolverSteihaug` with a trust region follows a direction of nonpositive curvature to
  the boundary, as Steihaug's method prescribes.
- A route switch refuses a value it does not know, in the environment (at import) and
  through `hm.config`, instead of falling back to another route; `hm.config` names every
  setting (`pattern_builder_min`, `pattern_sort_kernel`, `pattern_timing` and
  `hypre_pool_keep` are new there).
- The documentation and the benchmarks folder are shorter: the user guide keeps what a
  run needs, `benchmarks/README.md` maps each published table to its script,
  `docs/source/guide/configuration.rst` lists every setting, and the profiling scripts of
  finished investigations are gone from `benchmarks/` and `tools/`.
- `tools/build_pymfem_hip.sh` builds where ROCm and MPI come from modules. It took ROCm
  from `/opt/rocm` and the MPI include directories from OpenMPI's wrapper; it now reads
  `ROCM_PATH`, takes the include directories from an MPICH wrapper too and hands them to
  hypre's `hipcc`, and drives ROCm's clang through `MPICH_CXX` beside `OMPI_CXX`. A
  device source of hypre that the compiler cannot build at `-O2` is built at `-Os`: the
  clang of ROCm 7.2.0 stops in its AMDGPU backend on three SpGEMM kernels. The build
  was run on Frontier (Cray MPICH 9.1.0, ROCm 7.2.0, MI250X).

### Removed

- The snake_case aliases of hIPPYlib's method names (`solve_fwd` for `solveFwd`, and so
  on) and of the module-level functions (`double_pass_g` for `doublePassG`), with
  `hippymfem.common.naming`. Nothing used them; the names of hIPPYlib are the interface.
- Four switches whose other setting was a slower way to the same result:
  `HIPPYMFEM_ASSEMBLY=integrator` (assembly through MFEM's per-element callback, with
  `hippymfem.fem.integrators`; the tests keep it as their reference in
  `hippymfem/test/reference_assembly.py`), `HIPPYMFEM_TRIPLE` (the triple product is two
  sparse products, which the tests hold against hypre's `RAP`),
  `HIPPYMFEM_CHUNK_PLAN`, `_SHARE` and `_PROBE` (chunks are sized from the estimate), and
  `HIPPYMFEM_PARMAT=direct`.

### Fixed

- **A single-precision solve that hypre's PCG abandons is no longer taken for converged.**
  A solve counted as converged when it had used fewer than `max_iter` iterations. With a
  preconditioner that is not positive definite hypre's PCG stops after two or three
  iterations and reports convergence, with a residual that is not a number and a true one
  of order one. BoomerAMG with plain Jacobi relaxation (`HIPPYMFEM_SINGLE_AMG="relax=7,pmax=6"`)
  is such a preconditioner outside first- and second-order hexahedra on a regular mesh with
  moderate contrast (quadratic tetrahedra, a stretched mesh, an anisotropic coefficient, a
  strong contrast); the default smoother never is. The verdict is now hypre's own together
  with a finite residual: the solver raises, and a forward or adjoint solve is solved with
  the Jacobian in double precision instead (`PDEVariationalProblem.REFINE_STALL`). The
  guide says where `relax=7` is safe and names `relax=16,cheby_order=1,pmax=6` (Chebyshev
  relaxation of order one), which converged in every case tried, at half the gain on the
  model problem.
- **Small and large right-hand sides in the single-precision solves.** hypre's PCG works
  with the squares of the right-hand side's and the residual's size, and in single precision
  those leave the range of numbers long before the vectors do. On a right-hand side of size
  1e-16 it broke off after two to four iterations with a residual of 4e-5 to 2e-2, one below
  about 1e-23 it took for zero, and one above 1e19 for a wrong input; all three came back as
  converged (seven of the 2,225 incremental solves of the Taylor approximation in `test_uq`
  on two ranks). Such a system is now solved again for the right-hand side scaled by a
  power of two to a size near one, which changes nothing but exponents, and a zero
  right-hand side returns zero without an iteration.
- numba's cache of the compiled pattern passes was next to the sources (numba's default).
  On a file system shared by the nodes the ranks of a run rewrite it under one another: of
  32 ranks started from a fresh checkout one stopped with `OSError: [Errno 116] Stale file
  handle` and the others waited for it. The cache is now in a directory of the node and of
  the user, `hippymfem-numba-<uid>` under the system's temporary directory
  (`patternbuild.cache_dir`), unless `NUMBA_CACHE_DIR` names another.
- **A script that holds MFEM objects at module level no longer ends with a segmentation
  fault on a GPU.** MFEM's memory manager goes with the `mfem.Device` that
  `mfemconfig.configure_device` creates, and at the interpreter's exit the module holding
  it could be cleared before the script's own objects, whose hypre matrices then faulted
  in their destructor (`HypreParMatrix::Destroy`), after all the work was done (exit code
  139). The device is now never destroyed from Python. Seen on RTX PRO 6000 Blackwell
  instances with the library of 2 October as well.
- The matrix-free second-derivative products and the third-derivative kernels carried
  zero tangents through the whole element for the slots without a direction.
- `MultiVector(other)` copied the backing array without syncing it from the device.
- `ParVector.norm("linf")`, `max()`, `min()` and `MultiVector.norm("linf")` lost a NaN held
  by a rank other than the first.
- `BFGS_operator.update` computed `H y` in the output vector of the two-loop recursion, so
  a pair that needed Powell damping raised.
- `test_kernels`' chunk-planner checks and `test_device`'s accumulator check failed on
  four ranks, where their batches did not split.
- `release()` left `LUSolver` on several ranks and the PETSc solvers holding the
  factorization it is called to free.
- `tools/mpirun_pinned.sh` hid the GPUs from a run with `HIPPYMFEM_DEVICE=auto`.
- `hm.config` took `"0"` as true for `fold_elimination`, `gpu_deterministic`,
  `device_pattern` and `share_hessian`, and reported `hypre_device` by another rule than
  the one the import applies.
- `hm.NullQoi` was the MCMC module's class, without the derivative methods of a QoI; the
  two are one class now.
- The MCMC guide's example used arguments the classes do not take.
- `VectorBiLaplacianPrior` assembled the mass matrix of its vector space with the scalar
  mass integrator, which left the matrix wrong and the prior unusable.
- `MultDiscreteStateObservation` (the multiplicative noise model) and `LumpedMassSolver`
  raised their errors on the ranks that saw the bad value only, which hangs a parallel run.
- hypre's device memory pool: when the driver refused a block and gave it after the pool
  was emptied, the runtime still remembered the refusal and MFEM stopped at its next
  kernel with an out-of-memory error; and when the card was full, hypre ended the run
  without a message. The refusal is read now, and a full card is reported.
- Two test suites under `srun` with Cray MPICH. `test_nb` added one entry to its results
  on rank 0 only, so rank 0 made one `check()` broadcast more than the other ranks:
  OpenMPI let the mismatched collectives pass, Cray MPICH's collectives over shared
  memory waited for ever on two ranks and more. `test_vectors` starts a fresh interpreter
  for its run without a visible GPU and strips the launcher's variables from the
  environment; `PALS_*` were not among them, and with those the child took itself for a
  rank of the step and waited in `MPI_Init` for the others.

## Version 0.1.0, released on September 21, 2026

First public release.

### The library

- The forward PDE is written as a residual density, a JAX function of the fields at one
  quadrature point, `pde_varf(u, m, p, x)`, with boundary densities
  `bdr_varf(u, m, p, x, n)` and interior facet densities `facet_varf(u, m, p, x, n, h)`.
  Every derivative block (the Jacobian and its transpose, the parameter Jacobian, the
  second-order blocks of the Lagrangian, third derivatives) comes from differentiating it.
- Discontinuous Galerkin formulations: a facet density is written in the jump and average
  of the two traces a face has, with the face measure of each side for the penalty term.
  An interior-penalty matrix agrees with MFEM's `DGDiffusionIntegrator` to 4.5e-16 on one,
  two and four ranks, boundary faces included, so Dirichlet data is imposed weakly; a face
  shared between ranks is assembled as MFEM assembles its own, each rank taking the rows of
  the element on its side. Non-conforming interior faces are not supported.
  `applications/dg/model_transport_dg.py` is a worked example: a diffusivity field
  inferred from a downstream plume at a mesh Peclet number in the tens, with an upwind
  flux on the interior faces and the inflow data imposed weakly.
- H1, L2, H(curl) and H(div) spaces on MFEM meshes (triangles, quadrilaterals, tetrahedra,
  hexahedra), and vector-valued H1 spaces.
- Assembly by a direct scatter of element arrays into hypre's parallel CSR matrices, with
  the sparsity pattern reused across assemblies.
- The inverse-problem layer: `PDEVariationalProblem`, `TimeDependentPDEVariationalProblem`,
  `Model`, `ReducedHessian` and `modelVerify`; Laplacian, BiLaplacian (anisotropic, with
  Robin conditions) and finite-dimensional Gaussian priors; pointwise, continuous and
  time-dependent misfits. It is adapted from hIPPYlib and keeps its interface conventions,
  in camelCase (`solveFwd`) and snake_case (`solve_fwd`) alike.
- Inexact Newton-CG with line search or trust region, BFGS, and steepest descent.
- The Laplace approximation: randomized single- and double-pass eigensolvers, and
  `GaussianLRPosterior` with sampling, traces and pointwise variance (exact, randomized or
  Monte Carlo).
- MCMC with pCN, gpCN, MALA and importance-sampling kernels, tracers, and the integrated
  autocorrelation time and effective sample size.
- Forward UQ: variational, linear and quadratic QoIs, the parameter-to-QoI map, Taylor
  approximations, and variance-reduced Monte Carlo.
- Linear solvers: MFEM's Krylov methods with hypre preconditioners, an exact replicated
  direct solver on any number of ranks, and PETSc solvers when petsc4py is available.
- GPUs: element kernels on the GPU through JAX (`HIPPYMFEM_DEVICE=gpu`), and MFEM and
  hypre on the device with a CUDA or HIP build of PyMFEM (`HIPPYMFEM_HYPRE_DEVICE=1`),
  which `tools/build_pymfem_cuda.sh` and `tools/build_pymfem_hip.sh` produce. The same
  scripts run on NVIDIA and AMD cards with the same cost functional and CG counts;
  `docs/source/guide/gpu.rst` has the measurements, the memory settings and the build
  notes.
- Sparsity patterns built once per space pair and reused, without a global sort when numba
  is available; element batches split to fit the device or the host memory.
- `hm.config`: every setting of the library in one object, with its value and source.
- Random streams that do not depend on the number of ranks, so prior samples, synthetic
  data and MAP points agree across rank counts to the tolerance of the solves.
- `hippymfem.nb`: plots of fields, meshes, observations, eigenvalues, eigenvectors and
  trajectories, for notebooks.

### Tutorials, applications and validation

- Twelve tutorials. Seven cover the standard workflow and are adapted from hIPPYlib's:
  MFEM101, a deterministic inversion written by hand, Bayesian subsurface flow, initial
  condition inversion for advection-diffusion, the spectrum of the Hessian, MCMC, and
  Gaussian priors. Five cover what is specific to this library: residuals as programs, the
  GPU path, facet terms and DG, vector and mixed-family spaces, and forward UQ.
- Applications: subsurface flow, advection-diffusion, a Robin boundary coefficient, MCMC,
  forward UQ, DG transport at high Peclet number, and a 3D basin-scale geothermal inversion
  with a tabulated, temperature-dependent conductivity.
- Cross-validation against hIPPYlibx on a shared discrete problem (`validation/`).

### Testing and infrastructure

- Thirteen test suites, each holding the library against an external reference, run by
  `run_tests.sh` or pytest on any number of ranks, plus GPU and device suites for machines
  that have the hardware.
- GitHub Actions CI (test suites, tutorials, applications, documentation), a Dockerfile,
  and a script that builds PyMFEM with MPI from source.
- Sphinx documentation with a user guide, the rendered tutorials and the API reference; it
  builds without PyMFEM, as on Read the Docs.

### Design notes

The measurements behind the design decisions (the chunk planner, the direct CSR assembly,
the true-dof route, the device matrix constructors, the solver and memory choices) are in
`benchmarks/DESIGN_NOTES.md`.

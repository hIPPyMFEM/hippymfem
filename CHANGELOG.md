# Changelog

## Unreleased

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
  of a Newton-CG run change with this default.**
- **CG with a hypre preconditioner runs in hypre's own PCG** (`HIPPYMFEM_HYPRE_PCG`,
  `hm.config.hypre_pcg`, on by default). The iteration is the same as MFEM's `CGSolver`
  from a zero initial guess. hypre's PCG marks the vector it hands to BoomerAMG as zero,
  which saves the first relaxation of a V-cycle its matrix-vector product on the finest
  level: a solve of 2.1 million dofs to 1e-12 in 24 iterations took 0.103 -> 0.089 s on
  an H100 and 0.274 -> 0.227 s on an L40S; the Newton-CG run above 32.0 -> 29.6 s and
  73.3 -> 65.0 s. The solvers of one operator share one PCG object.
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
  and adjoint solves are refined against double-precision residuals, and the
  incremental solves of a Hessian action stop at 1e-5 and are used as they are. A CG
  iteration is 1.16 (H100), 1.31 (L40S) and 1.37 (Blackwell instance) times faster, a
  Hessian action 1.4 and 1.6 times, a BoomerAMG setup no faster, and the card holds
  1.3 GiB less at 2.1 million state dofs. With both, the Newton-CG run above takes
  24.8 s on the H100, 45.9 s on the L40S, 46.8 s on a Blackwell instance and 26.8 s on
  four: 2.6, 3.5, 3.5 and 3.0 times faster than before these three changes.
  `HIPPYMFEM_PRECISION=fp32`, everything in single precision, remains a tool for
  experiments: the optimizers now stop at the floor of that precision instead of
  failing in a line search.
- **Independent solves as an ensemble over the GPUs** (`ensemble=` of `MatMvMult`,
  `singlePass`, `doublePass`, `singlePassG`, `doublePassG`, and of the prior's and the
  posterior's `trace` and `pointwise_variance`; `benchmarks/bench_laplace.py --ensemble`).
  When the problem fits on one GPU, every rank builds it on `MPI.COMM_SELF` and the
  columns of a MultiVector, or the Monte Carlo samples, are divided among the ranks of
  the communicator passed as `ensemble`; the results are exchanged and every rank returns
  what one rank alone computes (`Random.tell` / `Random.seek` keep the samples the same).
  The stages of the Laplace approximation are sets of independent solves, and a domain
  decomposition of a small mesh speeds each of them up very little: at 64^3 the
  eigensolver takes 62.0 s on one MIG instance of an RTX PRO 6000 Blackwell, 30.4 s on
  four and 21.2 s on eight with the mesh divided, and 17.6 s and 9.9 s as an ensemble;
  all stages after the MAP point 83.5 s, 48.2 / 37.2 s and 27.8 / 17.9 s. The MAP point
  itself cannot be computed this way.
- **Assembled matrices and vectors stay on the GPU** (`hippymfem.common.devicebridge`;
  `HIPPYMFEM_DEVICE_BRIDGE`, `HIPPYMFEM_DEVICE_VECTORS`, `hm.config.device_bridge`,
  `hm.config.device_vectors`). With the element kernels and hypre on one card, the
  assembled values used to be copied to the host, handed to MFEM there and uploaded again,
  and every vector the library touched between two hypre calls was copied down and up.
  JAX and MFEM cannot write into each other's memory, so the library now takes the address
  of a JAX array's buffer and of MFEM's device copy of a vector or matrix block and copies
  between them on the device with the runtime's `cudaMemcpy` or `hipMemcpy`. The two
  blocks of a hypre matrix are filled that way and MFEM's constructor uploads the row
  pointers only; a `ParVector` that hypre has used does its arithmetic through MFEM on
  the device; essential entries, the kernels' dof values, assembled residual and gradient
  vectors and the pointwise observation operator stay there too. A complete Jacobian
  assembly of 32 768 Q2 hexahedra on an H100 went from 71 to 12.7 ms (kernel 9.8 ms); on
  meshes of 0.3 to 2.1 million state dofs a complete assembly is 18x to 84x faster than a
  host core on an L40S and 46x to 313x on an H100 (11x to 36x and 15x to 73x before). A
  forward solve at a new parameter with 2.1 million state dofs: 3.22 -> 1.48 s on one MIG
  instance, 4.75 -> 2.38 s on sixteen with 2.1 million each, 1.34 -> 0.53 s on an H100. A reduced-Hessian application is its two solves
  for 96 % at 2.1 million dofs per instance (91 % before). Two Newton-CG steps, against the times of
  September: 128^3 on four L40S 49.5 -> 29.1 s, on one H100 109 -> 42.9 s, on four MI210
  31.9 -> 23.6 s; 256^3 on 16 Blackwell cards 77.4 -> 46.3 s; 400^3 on 24 Blackwell cards
  191 -> 134 s, with the same cost functional to nine digits and the same CG counts.  The
  copy on the device also works on the HIP build (MI210). Matrices agree with the host route to round-off.
  Both switches fall back to the host route, which is also taken when the kernels and
  hypre are on different cards or the runtime's copy function is not found.
- **hypre's recycling pool is on by default and keeps blocks between setups**
  (`HIPPYMFEM_HYPRE_POOL=auto`, `HIPPYMFEM_HYPRE_POOL_KEEP`, `set_hypre_pool(...,
  scoped=True, keep_megabytes=...)`, `hypre_pool_trim`). Without a pool a forward solve on
  sixteen MIG instances made 4 963 device allocations and frees, 1.28 s of its 3.21 s; the
  ones that cost are the few hundred blocks above 1 MB, at 1.7 to 2.3 ms each when sixteen
  processes share a node. The pool may hold 1 GiB while a BoomerAMG setup runs and 512 MB
  between setups (never more than a quarter of the most hypre has had in use), and a
  freed block displaces larger ones when it is full. The forward solve took 2.37 s and its
  setup 0.55 instead of 1.45 s. `HIPPYMFEM_HYPRE_POOL=<megabytes>` keeps one limit at all
  times as before, `0` removes the pool, and the element kernels empty it before they
  retry after running out of device memory. NVIDIA builds only.
- **Less device memory after the pattern build.** Without numba or CuPy the sorts of a
  pattern build run in JAX's arena, which never shrinks; they now take chunks of 2^26
  keys at most (`hippymfem.fem.devsort.MAX_CHUNK_ARENA`). 2.1 million Q2 state dofs on one
  H100: 21.0 -> 12.8 GB on the card for 8 s more of one-time setup; a MIG instance with
  the same dofs 12.7 -> 8.7 GB.
- `HIPPYMFEM_DEVICE_PATTERN=1` (`hm.config.device_pattern`) keeps the scatter map and the
  column indices of the patterns on the device, so that an assembly uploads nothing but
  row pointers: 1.3 GB at 2.1 million state dofs for 3 % of a forward solve on one H100
  and 5 % on sixteen MIG instances. Off by default.
- **hypre with a faster exchange between GPUs** (`tools/rebuild_hypre.sh`,
  `tools/hypre-2.32.0-pinned-staging.patch`; `mfemconfig.hypre_gpu_aware_mpi`,
  `mpi_gpu_support`). The script rebuilds the hypre of an existing PyMFEM build in one of
  two variants. `--gpu-aware-mpi` hands MPI the device buffers: with a CUDA-aware Open MPI
  a CG iteration on two H100 took 5.4 instead of 6.2 ms at 2.2 million dofs per card and
  20.9 instead of 22.4 ms at 8.5 million; on MIG instances, which cannot use CUDA IPC, it
  is slower than the default from four instances up. `--pinned-staging` keeps the route
  through the host with page-locked buffers that are reused: 3 to 6 % per iteration on
  two to sixteen MIG instances (`HYPRE_PINNED_STAGING=1 tools/build_pymfem_cuda.sh`
  installs it at the end of a build; off by default). `configure_device` raises, with the reason, when the
  loaded hypre hands over device buffers and the MPI library reports no CUDA support.
- `benchmarks/bench_forward_steps.py` (the steps of a forward solve with the driver calls
  of each), `hypre_pool_trace.py` and `hypre_pool_replay.py` (hypre's device allocations
  recorded and replayed under pool rules). `krylov_anatomy.py` had two defects: with
  `--assembly mfem` it read the parameter from a host copy that the device had not
  filled, so its matrices had the coefficient 1, and on several ranks its variant without
  a kernel was no longer cuSPARSE once the library chose hypre's. The iteration times of
  the GPU guide were measured again and changed by 3 % or less, except at 64^3 on two and
  four instances (up to 13 %).
  `benchmarks/DESIGN_NOTES.md`, section 11, has the measurements.
- **hypre's matrix-vector kernel on several GPUs** (`HIPPYMFEM_HYPRE_SPMV`,
  `hippymfem.common.mfemconfig.set_hypre_spmv`, `hm.config.hypre_spmv`). hypre multiplies
  with the off-diagonal blocks of its parallel matrices through cuSPARSE, whose product
  does not get cheaper with fewer nonzeros: up to 5.4 ms for a block with 2.2 million rows
  and 701 nonzeros, against 2.2 ms for the diagonal block with 136 million. `auto`, the default,
  switches hypre to its own kernel when the communicator has more than one rank (CUDA
  builds) and keeps the vendor's on one rank, where it is the faster. A CG iteration on two
  to sixteen MIG instances became 11 to 36 % faster and a reduced-Hessian application 1.2
  to 1.5 times; on two H100 an iteration went from 16.5 to 6.2 ms. Results are unchanged.
- **A recycling pool for hypre's device memory** (`HIPPYMFEM_HYPRE_POOL=<megabytes>`,
  `set_hypre_pool`, `hm.config.hypre_pool`; see above for the default). A BoomerAMG setup makes about
  2 200 `cudaMalloc` and 2 000 `cudaFree`, 60 to 75 % of its time, and a call takes longer
  the more processes of a node make them: 0.16 s on one MIG instance, 1.47 s on sixteen.
  With a pool that may hold 1 GiB per rank, sixteen instances took 0.63 s. The pool goes
  through hypre's hook for user allocators, hands out exact new blocks and recycled ones
  of at most 1.19 times the request, and empties itself when the driver refuses an
  allocation. NVIDIA builds only.
- **One rank with hypre on a device assembles through the true-dof route**
  (`HIPPYMFEM_TDOF_IDENTITY`, default `auto`). A block with identity prolongations used the
  constructor that copies and splits a row-major CSR on the host; it now hands MFEM
  hypre's two blocks, as several ranks do. A complete Jacobian assembly on one H100 or
  L40S became 1.4 to 3.2 times faster (hex P2: 5.2 -> 1.7 us per element on an H100), with
  matrices identical to round-off. The host is unchanged.
- The GPU guide's assembly factors are now those of meshes with 0.3 to 2.1 million state
  dofs (kernels 17x to 89x on an L40S and 43x to 537x on an H100 against one host core;
  the complete assembly is in the first entry). The smaller meshes used before understated
  the cheap elements (a kernel call costs about 1.5 ms on a card) and overstated the P3
  hexahedra (the host core is 1.7 times slower per element at 2 744 elements than at
  32 768).
- `benchmarks/bench_hessian_anatomy.py`: a reduced-Hessian application as the library
  runs it against MFEM's solver alone, with the copies between host and device counted.
  The two solves were 89 to 91 % of an application on one and on sixteen MIG instances
  while the vector arithmetic and the observation operator ran on the host (first entry).
  `benchmarks/bench_assembly_profile.py` now also counts the transfers of the matrix (the
  upload of the finished matrix is about 20 of the 71 ms of an assembly of 32 768 Q2
  hexahedra on an H100).
- `benchmarks/krylov_anatomy.py` and `tools/gpuprof.c`: one preconditioned Krylov
  iteration taken apart per rank count and BoomerAMG variant, with a preloaded library
  that counts driver allocations, copies, kernel launches, cuSPARSE products and MPI calls
  and times every kernel on the device. `benchmarks/bench_assembly_sweep.py` and
  `bench_assembly_profile.py`: time per element against the number of elements, and where
  a complete assembly on a GPU spends it. `bench_scaling.py --cart-part --both-kernels`.
  The GPU guide's new section "Several GPUs" and `benchmarks/DESIGN_NOTES.md`, section 10,
  have the measurements.

- `PDEVariationalProblem.apply_ij_at(i, j, x, dir, out)`: a second-derivative block at `x`
  applied to a direction, from the element kernels and without assembling the block
  (`QuadratureKernel.element_hvp`, one forward tangent through the element gradient where
  the assembled block takes one per element dof). Equal to `apply_ij` after
  `setLinearizationPoint(x)` to round-off, the essential rows and columns and the
  Jacobian's identity rows included; the cheaper route for a block applied once or twice at
  a point (SOUPyMFEM's sample-average Hessians). As for the residual, an interior-facet
  density must be unchanged when the sides of a face trade places (every DG form is).
- The kernels of one mesh share a single device copy of its geometry and of each space's
  tables; each kernel held its own, so a problem with a quadrature-kernel objective or
  penalty (SOUPyMFEM's QoIs) held two or three copies of the largest arrays on the device
  (873 against 454 MB of JAX arrays at 68 921 P1 unknowns).
- `setLinearizationPoint(x, matrix_free=True)` (and `matrix_free=` on
  `Model.setPointForHessianEvaluations` and `ReducedMap.setLinearizationPoint`) assembles
  the Jacobian alone, which the incremental solves need, and `apply_ij` then takes the
  products with the other blocks from the element kernels at `x` (`apply_ij_at`);
  `matrix_free=(PARAMETER,)` does that for the blocks that involve the parameter and
  assembles the rest. A point then costs almost nothing and holds less, and every product
  costs a kernel pass. On the two Newton-CG steps of the GPU guide (four L40S, hypre on the
  cards, `benchmarks/bench_newton_device.py --matrix-free`) the card peak fell from 4.9 to
  4.6 GiB at 64^3 and from 19.1 to 17.2 GiB at 128^3 (hypre's `C`, `W_um` and `W_mm`; JAX's
  pool and the host are unchanged), and the steps took 9.9 s instead of 9.6 and 51.3 s
  instead of 49.1, with the same cost functional: a reduced-Hessian apply is about 40 %
  slower and a point 0.4 s and 2.7 s cheaper, so it breaks even at about three applies per
  point. On a CPU a reduced-Hessian apply costs about nine times the assembled one (a P1
  Poisson inverse problem at 68 921 unknowns: 1.63 s against 0.19, the point 1.0 s against
  2.1), so there it is for memory alone. It suits a point applied once or twice (a
  sample-average Hessian) or a problem short of memory; the default stays assembled. A
  residual declared linear in the state (`is_fwd_linear`) has no `W_uu` work at a
  matrix-free point either.
- A problem with an interior-facet density checks once that the density is unchanged when
  the two sides of a face trade places (traces swapped, normal reversed, element measures
  exchanged) and warns if it is not (`hippymfem.fem.facets.check_facet_symmetry`). A face
  shared by two ranks is assembled by each from its own side, so such a density gives
  results that depend on the partition; every form written in jumps, averages and the
  normal passes. `test_modeling`'s third-derivative check used one that did not (an odd
  power of the jumps; it compared like with like, so it passed) and now uses one that does.
- `HIPPYMFEM_HESSIAN=quadrature` (or `hm.config.hessian`): the element Hessian blocks from
  the density's second derivatives at the quadrature points, with respect to the fields'
  values and gradients, contracted with the basis values and physical gradients; the
  element route, the default, pushes one forward tangent per element dof through the whole
  element. The parts of the pointwise Hessian that are identically zero are found once
  from the density's trace and skipped (a residual linear in the state has no state-state
  part, and most couple few of the others). Equal to the element route to round-off. On
  an L40S it is the faster route from P2 up: a Jacobian 1.6x (Poisson, P2 tetrahedra),
  1.9x (a nonlinear density), 2.6x (elasticity, P2 tetrahedra) and 4.4x (elasticity, Q2
  hexahedra) faster, and a whole linearization point 1.2x to 1.7x. On P1 the element route
  stays the faster (the quadrature route's Jacobian runs at 0.7x), and so it does on the
  host, except for vector-valued Jacobians (1.4x there). `auto` times both routes once per
  kernel program, column slot and device and keeps the faster, so kernels built alike take
  the same route within a run. The default stays `element`, whose results are reproducible
  bit for bit. Spaces with a Piola map and interior facets always take the element route.
- Fixed: the matrix-free second-derivative products (`hvp_block`, behind `element_hvp` and
  `apply_ij_at`) and the third-derivative kernels (`third_block`, `third_dir_block`) gave
  the slots without a direction zero tangents, which JAX carries through the whole element,
  where differentiating in the direction's slots alone lets it drop the rest: a product
  into the state or adjoint row paid for every field's evaluation even where it is
  identically zero. On SOUPyMFEM's control problem at 82 944 P1 tetrahedra such products
  take 4.2 and 2.4 ms instead of 34 and 32, and a sample-average Hessian action with 8
  samples at 15 625 unknowns takes 1.02 s instead of 1.22 (2.16 s with assembled blocks).
  The inverse problem of `bench_newton_device.py`, whose density couples every slot, is
  unchanged (two Newton steps at 64^3 on one L40S: 27.1 s matrix-free, 27.5 s assembled).
- The GPU guide's table of JAX's share of the card is re-measured on the current code, with
  matrix-free linearization points beside it (128^3 on four L40S): a share of 0.20 takes the
  card peak from 19.1 to 15.1 GiB for 7 % more time, and matrix-free points on top take it
  to 13.2 GiB for 30 % more.
- On a CPU the element kernels step through the batch 2 048 elements at a time inside the
  compiled program (`HIPPYMFEM_HOST_BATCH`, `0` for the whole batch), which keeps each
  step's intermediates in cache: 2.2x on a P1-tetrahedron Jacobian and 3x on a third
  derivative at 196 608 elements, with the same element arrays. The GPU path is unchanged.
  The CPU timings quoted elsewhere in the documentation predate it.
- The element dof gather in front of every kernel is compiled; eager indexing spent more
  time checking the index array than gathering (a residual-vector assembly at 64^3
  tetrahedra: 1.11 s, now 0.35 s). The values are unchanged.
- `PDEProblem.apply_third_dir(i, x, dirs, weights, out)`: the weighted second directional
  derivatives of the slot-`i` gradient along directions spanning every variable, the sum of
  the `apply_ijk` pairs of each direction in one kernel pass (`QuadratureKernel.
  element_third_dir`); the base class sums the pairs. SOUPyMFEM's second-order adjoint uses
  it: its quadratic Taylor gradient at 32^3 takes half the time.
- Fixed: `MultiVector(other)` copied the backing array without syncing it, so columns hypre
  had written on a device were copied as zeros.
- Fixed: `ParVector.norm("linf")`, `max()`, `min()` and `MultiVector.norm("linf")` lost a
  NaN held by a rank other than the first (MPI's MAX and MIN compare, and NaN compares
  false); they now return NaN wherever a rank holds one.
- Fixed: `BFGS_operator.update` computed `H y` with the two-loop recursion working in the
  output vector, so `H0inv.solve` got its input as its output; with the default rescaled
  identity `y^T H y` came out 0 and a pair that needed Powell damping raised. hIPPYlib has
  the same code.
- `CGSolverSteihaug` with a trust region follows a direction of nonpositive curvature to the
  boundary, as Steihaug's method prescribes; it took the whole first direction wherever that
  landed (outside a small region) and stopped inside the region at a later iteration, as
  hIPPYlib does. Without a trust region nothing changes.
- `test_kernels`' chunk-planner checks size their mesh by the number of ranks, and
  `test_device`'s accumulator check its pinned chunk: on four ranks the batches did not
  split (250 elements a rank against the planner's floor of 256, 54 against a chunk of
  64) and the checks failed.
- `TimeDependentPDEVariationalProblem` takes a boundary density (`bdr_varf`), so a Robin
  condition or a prescribed flux enters the one-step residual as it does the stationary
  one; checked against MFEM's boundary mass matrix and through an inversion.
- `LUSolver` and `ReplicatedLUSolver` factorize on rank 0 and scatter the solution, so a
  node's memory is charged once rather than once per rank; `replicate=True` restores the
  factorization on every rank. A factorization that fails raises on every rank.
- `HIPPYMFEM_DEVICE=auto`: the element kernels take a GPU when the process can see one and
  the host otherwise, so one environment serves a laptop and a GPU node.
- The randomized eigensolvers orthonormalize after every power iteration, so more
  iterations sharpen the tail of the spectrum instead of losing it past 1/eps: at a
  spectral ratio of 3e6, the 40th of 40 eigenvalues goes from 84 % off to 6e-5 with three
  iterations and 25 extra vectors; one iteration is unchanged.
- `tools/install_petsc_mumps.sh` builds PETSc with MUMPS, ScaLAPACK, METIS and ParMETIS
  against the system MPI and petsc4py against it, so `PETScLUSolver` factorizes in parallel
  (`package_used == "mumps"`); the solvers guide gives the measured cost.
- The 400^3 row (1.09 billion unknowns, 24 Blackwell GPUs) now reports two warm Newton-CG
  steps like every other row: 190.9 s with 3 + 9 CG iterations (it quoted one step with the
  first Hessian-block build and its JAX compilation inside, 197 s, before).
- `benchmarks/bench_laplace.py` gains `--fields` (truth, MAP, prior and posterior pointwise
  std and the exact variance reduction on the parameter grid, for plotting),
  `--release-linearization` (128^3 then fits on two 45 GiB cards), and records the setup and
  end-to-end times, the KL divergence and the fraction of dofs where the truth lies within
  two posterior standard deviations of the MAP.
- The README figure and the GPU and performance guides carry the 256^3 comparison on 32
  ranks: 77 s on 32 Blackwell slices against 6187 s for hIPPYlibx and 2859 s for hIPPyMFEM
  on 32 CPU cores.

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

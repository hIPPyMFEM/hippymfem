# Changelog

## Unreleased

### Added

- **Every suite on a GPU.**  `DEVICE_SUITES=1 ./run_tests.sh 1 2` runs every suite once
  more with the element kernels and hypre on the card (`HIPPYMFEM_DEVICE=gpu
  HIPPYMFEM_HYPRE_DEVICE=1`), as a GPU user runs a script.  The checks that compare two
  routes for identical results allow round-off there (`hippymfem/test/_device.py`), and
  the checks that read MFEM vectors through raw host views synchronize them first.
- **An initial density for time-dependent problems.**
  `TimeDependentPDEVariationalProblem(..., init_varf=rho0)` defines the initial state by a
  weak equation of its own, `R_0(u_0, m; p) = integral of rho0(u0, m, p, x) = 0`, written
  as a density like the step's: `(u0.val - m.val) * p.val` sets the initial state to the
  parameter, `(u0.val - g(m.val)) * p.val` to a map of it, and any density JAX can
  differentiate twice is allowed (`init_linear=False` for one that is nonlinear in `u0`).
  It is solved and differentiated by a `PDEVariationalProblem` of its own (`init_problem`,
  with the state's conditions), and its terms join those of the steps in the solves, the
  gradient and every Hessian block.  The hand-written wrapper of the tests and of
  `applications/ad_diff` (`u0 = m` with the boundary values zeroed) is no longer needed;
  the wrapper ignores the parameter's boundary values, where the density's projection
  keeps them, and the two agree at the interior dofs to round-off.  `test_timedependent`
  checks that agreement and the first-order finite-difference checks of the gradient and
  the Hessian action for `(u0 - tanh(m) / 2) p`.
- **Single-precision solves for Jacobians that are not symmetric.**  Where the solvers of a
  problem are GMRES with BoomerAMG, a Jacobian that the symmetry probe (or
  `symmetric_jacobian=False`) finds not symmetric is assembled into the single-precision
  hypre as well, and so is its transpose, from the (state, adjoint) block of the same
  residual; each is solved by hypre's GMRES with a BoomerAMG of its own there
  (`singlesolve.SingleGMRESEngine`), the adjoint by a solver of its own (`solver_adj`,
  cloned from the forward solver where it is not set).  The forward and the adjoint solve
  are refined against double-precision residuals as in the symmetric case.  Before, such a
  Jacobian went back to double precision.  The defect of the row sums that the
  single-precision PCG adds is left to the refinement.  A problem that asks for
  `transpose_free_adjoint=True` keeps such a Jacobian in double precision: there is no
  transpose wrapper of a matrix of the single-precision hypre.
- **The transposed blocks of a linearization point as matrices of their own.**  With hypre on
  a device, `C^T` and `W_um^T`, which every CG iteration applies, are assembled from the
  transposed element matrices of the same differentiation pass into a pattern of their own,
  with the essential columns eliminated (`HIPPYMFEM_DIRECT_TRANSPOSES=auto|1|0`).  MFEM used to
  form them at the first transposed product of each linearization point by sorting, which
  asked the driver for 4.9 GiB a block at 120^3 quadratic hexahedra a GCD, kept 2.4 GiB, and
  cost 1 to 2.4 % of two Newton-CG steps; on a host MFEM multiplies with a transpose without
  forming it, and nothing changes there.  A problem with interior-facet terms keeps MFEM's.
- **A GPU workflow and a performance check.**  `benchmarks/perf_check.py` runs two
  Newton-CG steps of the model problem at a few sizes and compares them with the baseline of
  the machine (`benchmarks/perf_baselines/<machine>.json`, written with `--record`): a size
  fails when it is more than 15 % slower, or when its Newton and CG counts or its cost
  functional differ; a baseline recorded under another configuration (device, profile,
  precision, solver settings) is refused.  `.github/workflows/gpu.yml` runs the suites and
  that check on a self-hosted runner labelled `gpu` (on pushes to `main`, by hand and weekly;
  never on pull requests), with the interpreter and the GPU build of PyMFEM named in
  repository variables.
- **Robustness on many ranks and a faster start.**  On more than one rank, `import
  hippymfem` wraps `sys.excepthook` so that one rank's uncaught exception prints its
  traceback and aborts every rank (`HIPPYMFEM_ABORT_ON_ERROR=0` keeps the old behavior):
  the other ranks used to wait in their next collective until the job's time limit.
  `HIPPYMFEM_JAX_CACHE=<directory>` turns on JAX's persistent compilation cache, so that a
  run of a problem compiled before takes its element kernels from it (programs that
  compiled in under `HIPPYMFEM_JAX_CACHE_MIN_SECONDS`, 1 s, are not kept).  The mesh that
  every rank builds its own part of, from a coarse mesh it partitions and cuts, is the
  library's: `hm.box_mesh(n, comm, coarse=None, levels=0)` (from the geothermal
  application's `cube_mesh`, which stays as a name): 0.7 s and 0.3 GB a rank at 256^3 on 32
  ranks where `ParMesh(comm, Mesh.MakeCartesian3D(...))` takes 42 s and 10 GB, and with
  `levels` the meshes between for the p-multigrid's geometric levels.
- **Profiles: named sets of defaults** (`HIPPYMFEM_PROFILE=gpu|gpu-large` before `import
  hippymfem`, or `hm.config.use_profile(name)` after it; `hm.config.profiles()` describes
  them).  A profile fills in the environment variables it names that the environment leaves
  unset, so an explicit setting always wins, and `hm.config.show()` marks what it set.
  `gpu` takes the element Hessians at the quadrature points and, where a single-precision
  hypre is named (`HIPPYMFEM_HYPRE_SINGLE`), the Jacobians and their solves in single
  precision with the chunk planner set for single-precision tangents.  `gpu-large` adds the
  solver settings of the weak-scaling series on Frontier to 4096 GPUs that run on any hypre:
  BoomerAMG with l1-Jacobi weight 1.8 and six interpolation entries a row, the p-multigrid
  over up to seven meshes of a uniformly refined mesh with its small levels in host memory,
  the prior's cycles over the meshes, and with a single-precision hypre the blocks of a
  linearization point in single precision (with `tools/hypre-3.2.0-seqdevice.patch`, add
  `seq_threshold=30000,redundant=1` to `HIPPYMFEM_SINGLE_AMG`).  What changes the discrete
  problem (a lumped prior mass) and what belongs to a solver call (tolerances, Newton-CG
  parameters) is not in a profile.  New knobs `profile`, `single_amg`, `single_row_sums`,
  `single_give_up`.

- **Two applications that follow published inversions.**  `applications/seismic/` infers
  the wave speed of the Earth from seismograms, the problem of Bui-Thanh et al. (SC12,
  2012): the acoustic wave equation as a density at a point, continuous spectral
  elements with leapfrog and time reversal for the adjoint and Hessian sweeps, and the
  library's `Model`, `ReducedSpaceNewtonCG`, `MultiVector` and `GaussianLRPosterior`
  unchanged.  With 987 387 parameters and 26.4 million wave unknowns the MAP point takes
  12 Gauss-Newton iterations and the low rank 800 Hessian products; a forward solve was
  run on 24.2 billion unknowns.  `applications/icesheet/` infers the basal friction of
  the Antarctic ice sheet from the velocity of its surface, the problem of Isaac et al.
  (J. Comput. Phys. 296, 2015): full Stokes with Glen's law as one residual density over
  the velocity, the pressure and the friction, a class that implements the library's
  `PDEProblem` interface with a direct solver on the card (cuDSS) or without any matrix,
  a posterior from a low rank or from the misfit Hessian as a sparse matrix, and a
  forward solve on 1.09 billion unknowns.  Both fetch their data on first use
  (`earth.py fetch`, `antarctica.py fetch`); their READMEs have the commands, the
  results and what was checked.  They are not part of the continuous integration, which
  has neither their data nor a GPU.
- **A third application: the Marmousi model in three dimensions with a point spread
  function Hessian.**  `applications/marmousi/` takes the inversion of Hu, Alger, Nammour
  and Ghattas (arXiv:2507.10804, 2025) from the paper's two-dimensional section to a box
  of 9.2 x 2.4 x 2.4 km: acoustic waves with absorbing faces on the seismic application's
  propagator, the wave speed's offset trilinear on the elements as the unknown, the
  library's bi-Laplacian prior as a stencil on that lattice, and the misfit Hessian
  approximated by the paper's point spread functions, read off products with combs of
  unit vectors (the ice sheet's `localhessian.comb_products`, extended to steps, anchors
  and any dimension) and applied as a product-convolution by Fourier transforms of small
  boxes on the GPUs, with a halo from the neighbouring ranks and no global transform.
  `(H~ + R)^-1` is the initial inverse Hessian of the driver's L-BFGS, the preconditioner
  of `ReducedSpaceNewtonCG` and the Laplace approximation that `gpCNKernel` proposes from.
  On a small case where the data dominate the prior, L-BFGS with it reaches a gradient of
  1e-4 of the first in 95 to 111 propagations of all shots, where L-BFGS without a
  preconditioner has 1.1e-4 after 310 and with the prior needs 353.  `run.py` runs the
  inversion in stages that resume (a farm of shots, or one box cut into blocks);
  `scale.py` times the sweeps and the operator at any size.  The README has the commands,
  the results.  `--psf-amplitude` scales the operator by the Hessian's local size, estimated
  from products with white vectors, and `--inner-precondition levels` preconditions the
  inner solve by a factor whose symbol follows the window from node to node (about 60
  iterations where the prior's takes 750): with both, L-BFGS reaches 1e-6 of the first
  gradient in 204 propagations and 425 s on 32 GPUs, where without a preconditioner it
  takes 852 propagations and 1544 s.  With the data of the true model everywhere on the
  paper's box (138 GPUs), where the windowed unknown cannot explain all of them, plain
  L-BFGS in single precision with the time step at 0.95 of its bound reaches its best model
  1.76 times sooner than in double precision with the default step, and the point spread
  function preconditioner does not help; `--max-step`, `--start-from` and `--psf-weight`
  are the driver's options that were tried for it.
- **The seismic propagator for boxes** (`applications/seismic/sem.py`, defaults unchanged
  for the globe): first-order absorbing faces whose forward boundary values are kept for
  the backward sweep, a stiffness applied through slices of a 3D array of the unknowns on
  a mesh of equal bricks (0.097 ms a step a million unknowns on an MI250X GCD, 7.5 times
  the globe's kernel), shots as matrices of source amplitudes so that nothing is compiled
  from one shot to the next, and the exchange of a step behind its update
  (`HIPPYMFEM_SEM_OVERLAP`, `auto` by default where the exchange goes from GPU memory to
  GPU memory and a rank holds at least 16 million unknowns times the fields): the same
  bits as before, and on 4096 GPUs of Frontier (783 billion unknowns) 93 to 96 % of one
  GPU's step for cubes of 192 million unknowns a rank (87 to 92 % without it), on 512 GPUs
  99 to 100 % for columns (95 to 98 %).  The globe of the seismic application on 8 GPUs takes the same 154 ms a
  step with it, and 1 % less with two fields.
- **The two applications on AMD cards, and their weak scaling to 4096 GPUs of Frontier**
  (MI250X; a rank has one of a card's two GPUs).  The seismic propagator is JAX and ran
  there once the fault of the launch queue below was cured.  The ice sheet's solver
  without a matrix (`--solver columns`) runs with CuPy's ROCm build beside JAX; its two
  solvers that factorize need cuDSS and have no counterpart there.
  `seismic/scale.py --n 64 --keys` with one refinement more for eight times the ranks has
  7 798 784 elements and 211 million unknowns on every rank: on 4096 ranks, 862 596 390 337
  unknowns in double precision, a forward step takes 154.6 ms and the two-field sweeps of
  a Hessian action 186.3 and 194.9 ms, 99.7, 99.2 and 99.4 % of one GPU (the mean of eight
  one-rank runs on the GPUs of a node), with the adjoint identity to 1.9e-15; in single
  precision at `--n 80`, 1 710 194 718 145 unknowns, 140.5 ms, 100.2 %.
  `icesheet/scale.py --measure` with `--res R --nz 3 --refine 3` on 4 (32/R)^2 ranks has
  7 500 columns of 24 layers and 4.6 million unknowns on every rank: on 4096 ranks,
  18 927 600 204 unknowns, a product with the Jacobian takes 0.066 s (93 % of one GPU),
  the preconditioner's set-up 6.6 to 6.8 s (80 %), a Krylov iteration 0.488 s (47 %: the
  collapsed problem's conjugate gradients on the host take 19 iterations where they take
  one on one rank) and 0.289 s with `--coarse-maxit 4` (79 %), which leaves the outer
  iterations of a solve as they are (70 and 70 on 64 ranks, 251 against 257 on 256).
  With 88 layers 68 330 755 212 unknowns take 0.80 s an iteration, 90 % of one GPU, and
  133 s a set-up.  A solve to 1e-6 from the same state takes 17, 23, 34, 70 and 257
  Krylov iterations at 40, 20, 10, 5 and 2.5 km: the limit the application's README
  names, measured further.  Both READMEs have the tables.
- **The seismic propagator's exchange from card to card** (`HIPPYMFEM_DEVICE_MPI`,
  `sem.DEVICE_MPI`, `Interface.swap_device`, `SpectralWave._swap_device`).  The values a
  rank shares are gathered on its card, and MPI is handed the addresses of that array and
  of a device array kept for the neighbours' (JAX's `unsafe_buffer_pointer`), where the
  library knows device memory: `auto` takes Cray MPICH's `MPICH_GPU_SUPPORT_ENABLED=1`,
  which it does not start with unless its transport library for the cards is loaded.  On
  Frontier 22 MB a rank to 2 to 12 neighbours take 0.6 to 1.3 ms so, on one node and
  between two, against 3.5 ms in MPI and 3.6 ms back to the card through host arrays,
  after the 3.6 ms a run takes to bring them to the host; a step of 211 million unknowns a
  rank on 512 GPUs takes 153.8 ms in the place of 165.0 (one field) and 185.0 in the place
  of 213.8 (two).  With that library loaded the cards' runtime is started by `MPI_Init`: a
  program that imports mpi4py before hippymfem has to have the setting of the entry on the
  launch queue under Fixed in its environment (`_jaxconfig.PROFILER_LATE` says when it
  came too late).  The ice sheet's solver keeps its exchanges on the host (a megabyte
  each): from CuPy's arrays runs with that MPI hung now and then on 8, 256 and 1024 ranks,
  and why was not found out.
- **`icesheet/scale.py --measure`**: what a job of a few minutes can time of the solver
  without a matrix, the same work at every size and with every rank's own clock (the
  minimum, the mean and the maximum over the ranks in the record): a residual, the
  preconditioner's set-up with its parts, a product with the Jacobian, an application of
  the preconditioner, `--krylov-its` Krylov iterations whatever the residual does, and
  with `--krylov-count` the iterations of one solve; at the solution of `--start-from` or
  at the shallow-ice state of the grid (`IceSheet.shallow_ice`), which takes no solve.
  Its timers wait for the card (`ColumnStokes(wait=True)`, `settle`), and the solver's
  record has the seconds of its exchanges and of its orthogonalization.  `ICESHEET_DATA`
  names the folder of the grids (rank 0 reads and broadcasts; a missing grid is an error,
  not a rebuild on every rank), `--coarse amg` and `--checkpoint-every` are new options,
  and CuPy and the element kernels are checked to be on one card when the solver is built.
- **The blocks of a linearization point in the single-precision hypre**
  (`HIPPYMFEM_SINGLE_BLOCKS=1`, `single_blocks`; `fem.tdofassemble.SINGLE_BLOCKS`,
  `PDEVariationalProblem._single_blocks`, `singlesolve.SingleParMatrix.keep_transpose`).
  The CG of a Newton step applies `C` and `W_um` and their transposes in every iteration,
  and with hypre on a device a transposed product needs the transpose stored: MFEM forms
  and keeps it at the first such product after each linearization point.  With 120^3
  quadratic hexahedra on a rank the two blocks have 217 million nonzeros each, and they
  and their transposes took 9.7 GiB of a card in double precision, each transpose asking
  the driver for 4.9 GiB when it was formed; with 128^3 a card had 1.0 GiB left, and
  forming the second transpose on it took three times as long as the first.  A product
  with these blocks has no use for more than single precision beside the incremental
  solves around it (1e-5 to 1e-6), and in the mixed precision their element matrices are
  computed in it.  With the setting on, and the single-precision library loaded, the
  blocks whose elimination folds into the scatter are made as matrices of that library:
  four bytes a value, one copy of the column indices for the blocks of one pattern, which
  the pattern keeps from point to point so that finishing a block uploads nothing, and
  the two that are applied transposed keep their transposes there.  On one node of
  MI250X GCDs, in the benchmark's configuration: at 120^3 elements a GCD the least free
  memory of a card rises from 15.6 to 20.3 GiB and at 128^3 from 1.0 to 6.8 GiB; the
  assembly of the blocks takes 0.84 s in place of 1.04 s and 1.01 s in place of 1.22 s;
  a product takes 1.3 to 2 ms more (two conversions of a vector), and at 128^3 the two
  transposed products no longer wait for the driver (18 ms in place of 47 ms for the
  slower one, averaged over the calls of two steps); the operations of two Newton-CG
  steps take 16.6 s in place of 17.1 s and 19.7 s in place of 20.4 s, with the same
  iterations (0.43 and 0.53 s of that are the blocks' and the products', 2.6 %; the rest
  is the line searches' excess, which differs between two runs).  The products agree
  with the double-precision ones to 1e-7.  `W_mm`, which
  has no elimination to fold, and a block with a facet part stay MFEM's.  Off by default.
- **The small geometric levels of a p-multigrid in host memory beside the device** (the
  solvers' parameter `pmg_h_host`, `HIPPYMFEM_PMG_H_HOST`, in rows a rank, with `pmg_h`;
  `singlesolve.PMGEngine._levels`, `_enter_host`, `_HostGathered`, `_Library.to_host`,
  `PTransfer.host`).  A level of a few thousand rows a rank costs a device almost nothing
  to compute and all of its latency: a kernel launch for every operation and, around each
  exchange between ranks, a copy from the card and one back.  On 512 MI250X GCDs of
  Frontier the two coarsest meshes of the cycle and the gathered matrix under them (3460
  and 135 rows a rank, 729 rows) took 3.1 ms of an iteration's 35 ms and held 20 of its 38
  exchanges; on 64 GCDs the one such mesh took 1.4 ms.  The meshes of at most `pmg_h_host`
  rows a rank, and what is under them, now can live in host memory: their matrices are
  the same Galerkin products (the first copied from the device at every set-up, the
  others made there), their relaxation the same, and a cycle hands the restricted
  residual to the host at the first such level and takes the correction back.  There the
  same levels take 1.6 to 1.7 ms on 512 GCDs.  It pays for these levels where it did not
  for the last levels of an algebraic hierarchy (11 October): a geometric level has the
  26 neighbours of the mesh at every size.  A last matrix that the solver's
  `seq_threshold` gathers is gathered in host memory by the library itself and solved by
  a BoomerAMG of one process on every rank; hypre's own code for that in host memory
  leaves the ranks without a row of the gathered level in another collective than the
  others, and a matrix of 729 rows on 512 ranks stopped in its first cycle.  A gathered
  matrix has no exchange to save, and every rank solves the whole of it on one core, so
  the setting bounds it too: where it has more rows than `pmg_h_host` in all, the levels
  above it stay on the device with it (15 625 rows: an iteration 1.0 to 1.4 ms longer in
  host memory; 729 rows: 1.1 to 1.2 ms shorter; 4913 rows: 0.6 to 0.7 ms shorter), and
  the last matrix alone is never moved.  With `HIPPYMFEM_PMG_H_HOST=5000`, 960^3 on 512
  GCDs: 17.7 s for the operations of two Newton-CG steps where every level on the device
  takes 18.3 s, with the same iterations.  With it and the prior's cycles of the next
  entry, in one job on 1 to 4096 GCDs (the eighth series): 17.0, 17.7, 17.7 and 19.0 s
  on 8, 64, 512 and 4096 GCDs where the series of the entry below takes 17.0, 17.8, 18.3
  and 19.4 s; **1920^3 on 4096 GCDs, 120 billion unknowns: 1.9 % less time, 89.4 % of
  one node (87.6 %)**, and 81 % of that job's single GCD, which was itself 2.6 % slower
  than the one before it (79 % against that one; 77.5 % before).  An iteration is 1.2 ms
  shorter where the gathered matrix has 729 rows (128 and 512 GCDs) and 0.6 to 0.7 ms
  where it has 4913 (1024 to 4096; the solve of that matrix takes 0.42 ms in host memory
  and 0.47 ms on the device, so the difference is elsewhere, and not yet found).  In that
  configuration 2048^3 elements ran on 4096 GCDs (128^3 a GCD, 146 billion unknowns):
  two Newton-CG steps in 15.2 s, the operations in 23.0 s, 82 % of one GCD and 88 % of
  one node at that load.  Off by default.
- **A prior's multigrid cycles over the meshes, in the single-precision library** (a
  prior's `h_levels`, `prior_h`, `HIPPYMFEM_PRIOR_H`; `singlesolve.HCycle` and
  `matrix_blocks`, `linSolvers.CycleOf`, `prior._geometric_cycle`).  Where a fixed
  polynomial of multigrid cycles stands for each of the prior's elliptic solves in the
  CG's preconditioner (`cg_preconditioner_cycles`), the cycles can be V-cycles over the
  meshes that the mesh was refined from, in the single-precision hypre, in place of
  BoomerAMG's on the operator in double precision.  The operator is copied there once,
  its Galerkin products are made once, since it stays, and the whole polynomial runs in
  that library: a solve converts two vectors.  The rounding, 1e-7 of the result, is far
  below the polynomial's own error.  On MI250X GCDs the preconditioner of one CG
  iteration takes 0.035 s in place of 0.052 s on one node, 0.054 s in place of 0.071 s
  on 512 GCDs and 0.076 s in place of 0.089 s on 4096 (the cycle's eigenvalues lie in
  [0.60, 1.05] where BoomerAMG's lie in [0.54, 1.05]).  For linear nodal elements with
  one component on a mesh that was last cut in two a side; after a cut by three the
  cycle relaxes on the finest matrix by a polynomial of three steps and costs more than
  BoomerAMG's (0.061 to 0.079 s against 0.047 to 0.063 s), so such a prior, and one on a
  mesh that knows no coarser mesh, keeps BoomerAMG, with a warning.  Off by default.
- **Each rank's side of its card in the benchmark's table** (`bench_newton_device.py
  --rank-table`): JAX's allocator, the card's memory in use, hypre's pool with the
  seconds it spent in the driver, and the driver's seconds inside each stage, beside the
  rank's times.  Twice in two days every assembly of a Jacobian in a run waited half a
  second for one rank; in the run whose ranks were recorded that rank took 1.2 s where
  the others took 0.7 s, with its every other time and the kernels timed on its card
  unremarkable, and not in the next run on the same nodes.  The table is for reading the
  next one.
- **Geometric levels under the linear elements of a p-multigrid** (the solvers' parameter
  `pmg_h`, `HIPPYMFEM_PMG_H`, with `pmg`; `fem.ptransfer.refinement`, `set_coarser_meshes`,
  `coarser_meshes`; `singlesolve.PMGEngine` with `transfer.h`, `HIPPYMFEM_PMG_H_SMOOTH`;
  `applications.geothermal.model.cube_mesh(levels=...)`, `bench_newton_device.py
  --h-levels`).  BoomerAMG on the linear elements' matrix coarsens it, builds
  interpolations and forms their products at every set-up, and what it builds has 100 to
  170 entries a row from its second level on and one level more for every eight times the
  ranks, each with its exchanges whatever its size.  A mesh that was made by cutting
  coarser meshes (`ParMesh.MakeRefined`) has a hierarchy that needs none of it.  Told the
  meshes, the library corrects from the linear elements by a V-cycle over up to `pmg_h` of
  them: the interpolation between the linear elements of two meshes from MFEM's own on
  local dofs, with the essential dofs taken out as in the interpolation of the
  p-multigrid; the Galerkin products level by level at every set-up, by hypre's products
  with `A P` kept; the fine level's relaxation on each mesh, or as many steps of a
  Chebyshev polynomial as the next mesh is times coarser a side where that is three or
  more (one sweep around a cut by five took the Jacobian's solves from 6 to 9 iterations);
  BoomerAMG on the last product alone.  Every level has the 27 entries a row of linear
  hexahedra and the neighbours of the mesh, and the levels end at the first matrix that
  `seq_threshold` gathers.  `cube_mesh(n, comm, levels=(5, 3, 2, 2, 2))` builds the
  benchmark's mesh from one element a rank by those cuts and keeps the meshes between.
  On the MI250X GCDs of Frontier, 120^3 elements a GCD, in the configuration of the sixth
  series (below) with these levels, in one job on 1 to 4096 GCDs: a set-up of 0.07 s on
  one GCD, 0.12 to 0.13 s on 8 to 512 and 0.16 s on 4096 (0.16 s, 0.28 to 0.35 s and
  0.38 s); a forward linearized solve of 4.1 iterations on 4096 GCDs where there were 7.0,
  in the same 0.17 s from one node on (0.18 to 0.25 s), while the adjoint ones keep their
  7 iterations; and the operations of two Newton-CG steps in 15.0, 17.0, 17.8, 18.3 and
  19.4 s on 1, 8, 64, 512 and 4096 GCDs where they took 15.4, 18.1, 18.9, 19.9 and 20.9 s.
  **On 4096 GCDs (1920^3, 120 billion unknowns): 77.5 % of one GCD and 87.6 % of one node
  instead of 73.7 and 86.7 %, 91.6 % from 64 GCDs instead of 90.4 %**; on 512 GCDs 82 and
  93 % instead of 77.5 and 91 %.  The levels are the same on any number of ranks (the
  gathered matrix has 729 rows on 512 GCDs and 4913 on 4096).  What still grows with the
  ranks is the time of an iteration (24 ms for one more on one GCD, 32 ms on 4096: the
  cycle has 38 products with an exchange between ranks, 20 of them on its two coarsest
  meshes), the wait of an assembly for the slowest card, and the prior's preconditioner,
  whose cycles are BoomerAMG's.  With 128^3 elements a GCD, seven cuts in two: 18.4, 20.4
  and 21.5 s on 1, 8 and 64 GCDs instead of 19.0, 21.6 and 23.2 s, and 22.5 s on 512
  (1024^3, 18.3 billion unknowns: 82 % of one GCD, 91 % of one node).
  For linear elements on a conforming mesh that MFEM refines in parallel (hexahedra and
  quadrilaterals in MFEM 4.9); a mesh that knows no coarser one keeps BoomerAMG on the
  linear elements, with a warning.  Off by default.
- **Every rank's own times after a benchmark** (`bench_newton_device.py --rank-table`):
  the stages of the steps on every rank, its waits at the collectives, and two kernels
  that each rank runs alone on its card, gathered into `OUT.ranks.json` with their spread
  in the record.  The times a benchmark prints are rank 0's, and a rank that is done
  waits at the next collective for the slowest one.  On 512 MI250X GCDs of Frontier the
  element pass of a rank varies with a standard deviation of 1.2 % (9 ms of 0.74 s), the
  slowest rank is 65 ms behind the fastest at the collective after the Jacobian's pass,
  and the cards' own kernels are within 3 to 5 % of their medians.  It is not slow nodes:
  the same benchmark on the eight nodes ranked fastest, on the eight slowest and on eight
  around the median took 18.97, 18.89 and 18.95 s.  It is the cards, in their longest
  pass: on 4096 GCDs the two GCDs of an MI250X are slow or fast together (correlation
  0.69; four fifths of the variance between ranks is between cards), 4 % of the cards
  are more than 2 % behind the median and the slowest 5 %, and an assembly waits 65 ms
  for it; a card's time in that pass does not follow the two kernels timed before the run.
- **The last levels of a hierarchy gathered, with hypre on a device** (`seq_threshold`
  and `redundant` among the BoomerAMG options of `HIPPYMFEM_AMG`, `HIPPYMFEM_SINGLE_AMG`
  and a solver's `amg_options`; `tools/hypre-3.2.0-seqdevice.patch`, `HYPRE_SEQ_DEVICE=1
  tools/build_pymfem_hip.sh`; `mfemconfig.seq_on_device`, `mfemconfig.seq_solvers`).
  `seq_threshold=n` ends a hierarchy at its first level of at most `n` rows in all, and
  one process finishes that level, gathered.  On hundreds of ranks the last levels are a
  few rows a rank that talk to many ranks, at four exchanges a cycle each: on 512 MI250X
  GCDs of Frontier an iteration of CG with a linear-element hierarchy takes 7.1 ms
  instead of 8.8 ms with `seq_threshold=30000`, and a solve of the prior 0.088 s instead
  of 0.101 s, in the same iterations.  hypre's own code for this reads device memory
  through host pointers and faults; the patch reads the level from a host copy, gathers
  it, and sets a hierarchy of one process up on the device.  The first rank that holds
  rows of the level solves it and sends the solution to the others, or, with
  `redundant=1`, every such rank solves it and keeps its own rows.  A first version did
  only the second, and was wrong: hypre's device coarsening draws its random numbers from
  one generator a process, which stands wherever earlier hierarchies left it, so each
  rank had another hierarchy of the same matrix, the ranks' solutions differed by 3 to
  41 %, and the pieces did not fit together (CG took 78 iterations for 7 on a smooth
  right-hand side, or stalled, where a random right-hand side showed nothing).  Now the
  generator is set back before each rank's set-up and the solvers are compared on one
  right-hand side; where they differ the first rank solves.  `test_device` checks that a
  cycle with a gathered level is symmetric to rounding and solves for a smooth solution
  in the default cycle's iterations.  A library without the patch, or with its first
  version, is refused the option on a device with the reason.  Gathering a larger level
  does not pay (83 000 rows: the same cycle and 0.2 to 0.6 s more a set-up, since every
  rank receives the level's matrix).  Not a default.
- **A fixed polynomial of multigrid cycles in place of the prior's solves, where they
  precondition the CG of a Newton step** (`ChebyshevSolver`, `cg_preconditioner_cycles`
  of `ReducedSpaceNewtonCG`, `prior.getHessianPreconditioner(cycles=k)`,
  `model.Rsolver(cycles=k)`, `--cg-prec-cycles` of `bench_newton_device.py`).  The
  preconditioner of that CG is the prior's covariance, two solves with the prior's
  operator, which were Krylov solves stopped at 1e-6: ten to thirteen iterations each,
  0.09 s a CG iteration on one MI250X GCD and 0.26 s on 4096, the largest loss to the
  size that the Newton benchmark had left.  `ChebyshevSolver(krylov, k)` applies the
  solver's multigrid cycle `k` times, each from the iterate before it (so that the
  residual is formed inside the cycle and no product is made beside it), with the
  coefficients of the Chebyshev iteration on the interval of the cycle's eigenvalues,
  which it estimates once from twelve CG iterations: one linear map, symmetric and
  positive definite, within `1 / T_k(...)` of the solve (`error_bound`), where a Krylov
  solve stopped at a tolerance is another map for every right-hand side.  With five
  cycles the cost functional after two Newton steps at 480^3 on 64 GCDs is the one of
  the Krylov solves to 4e-4 (two such runs differ by 2e-4) and the prior's part of a CG
  iteration takes 0.07 s instead of 0.16 s; with four cycles the functional is 1.3 % off
  and with three 4 %, so the count is to be measured and not read off the bound.  Off by
  default (`cg_preconditioner_cycles = 0`).
- **BoomerAMG options that MFEM's class does not have, and the Newton benchmark on 1 to
  512 GPUs of Frontier** (`HIPPYMFEM_AMG`, the solvers' `amg_options`,
  `mfemconfig.set_boomeramg_options`).  PyMFEM passes on the settings that
  `mfem.HypreBoomerAMG` has a method for.  The class converts to hypre's own handle of
  the preconditioner, a shared MFEM library exports that conversion, and through the
  handle hypre's C interface takes every option, by the names of `HIPPYMFEM_SINGLE_AMG`
  and eight more (`mfemconfig.BOOMERAMG_SETTERS`); an MFEM library without the symbol
  gets an error that says so.  The benchmark of the README's "How it scales"
  (`bench_newton_device.py`, two Newton-CG steps) ran on OLCF Frontier with one 64^3
  block of elements on each MI250X GCD, from 64^3 on one to 512^3 on 512 (2.29 billion
  unknowns, 64 nodes): 8.6 s on one and 14.3 s on 512 with MFEM's settings.
  `HIPPYMFEM_AMG=relax_wt=1.8,pmax=6` (l1-Jacobi with a weight, which stays positive
  definite for any weight below two, and six interpolation entries a row) keeps an
  incremental solve at 17 iterations from one GCD to 512, where MFEM's settings take 26
  to 31: 7.3 s and 10.8 s.  With a parameter three times as large it took 22 iterations
  at 64^3 against 32, where plain Jacobi (`HIPPYMFEM_AMG_RELAX=7`) stopped after 10.  The
  two steps are not the same work at every size (3 to 11 forward solves, 6 to 12 CG
  iterations); with every size charged for the operations of the 64^3 run at its own cost
  of each, 512 GCDs keep 48 % of one GCD's rate with MFEM's settings and 54 % with these
  (71 and 76 % of one node's), and 256 GCDs with two blocks each 61 and 66 %.  What is
  lost is in the BoomerAMG hierarchy: it gains a level with every eightfold size, and a
  level below the third costs 0.4 to 0.6 ms a cycle on several GCDs whatever its size,
  against a few hundredths on one, where it has no exchange (4.6 ms of a cycle of 12.8 ms
  at 512^3, 0.9 of 7.3 ms at 64^3); the element kernels take the same time at every size.
  Not the default.
- **Spaces of more than 2 147 483 647 unknowns** (`HYPRE_MIXEDINT=1
  tools/build_pymfem_hip.sh`, `parvector._HYPRE_BIGINT`).  hypre numbers the unknowns of a
  space with `HYPRE_BigInt`, 32 bits wide as PyMFEM and the HIP script build it, and MFEM
  stops at the first space whose dofs, summed over the ranks, pass that range: 640^3 quadratic
  hexahedra.  The script now builds hypre `--enable-mixedint` on request (64-bit indices of a
  space, 32-bit ones within a rank) and has SWIG write PyMFEM's wrappers again from the
  headers of that build (with `-DHYPRE_MIXEDINT`: SWIG reads `HYPRE_utilities.h`, where the
  types are chosen, and not the `HYPRE_config.h` it includes, and wrappers written without
  it compile and convert a global size as an `int`).  The library had one integer type for a
  rank's indices and a space's (`_HYPRE_INT`); global columns, partitions and column maps now
  take `_HYPRE_BIGINT`, the arrays that MFEM wants as `HYPRE_BigInt *` come from
  `parvector.bigint_array`, `scipy_to_hypre` builds its matrix itself where the widths differ
  (PyMFEM's helper sizes every array as `HYPRE_Int`), and `hypre_to_scipy` refuses a matrix
  of more columns than MFEM's `SparseMatrix` can index.  Every suite passes on a build of
  either kind (`test_solvers` has the two widths and the round trip), and at the sizes both
  can run (1 to 256 MI250X GCDs with four 64^3 blocks each) the two gave the same cost
  functional to seven digits, the same Newton and CG counts and times within 1.3 %.
  The Newton benchmark with four 64^3 blocks of elements a GCD then ran on Frontier at 816^3
  on 512 MI250X GCDs (4.35 billion state unknowns) and at 1024^3 on 1024 (8.6 billion, 18.3
  billion with the adjoint and the parameter): two Newton-CG steps in 33.0 s with
  `HIPPYMFEM_AMG=relax_wt=1.8,pmax=6` and the patched staging, 34.8 s with hypre's own
  staging, 44.3 s with the library's settings (35 iterations a linearized solve instead of
  19).  Charged for the same operations, 128 nodes keep 90 % of the rate of one node and
  76 % of one GCD's (81 and 65 % with the library's settings).
  With the two hypre patches and the index tables of the next entries, 120^3 elements a GCD
  (14 million state unknowns, the most that the eight ranks of a node hold) ran from one GCD
  to 1920^3 on 4096, 512 nodes: 56.7 billion state unknowns, 7.1 billion of the parameter,
  itself past the 32-bit range, and 120.4 billion in all.  Two Newton-CG steps took 66.0 s
  (4 forward and 2 adjoint solves, 8 CG iterations), a forward solve 9.1 s with a BoomerAMG
  setup of 2.7 s, and a linearized solve 20 iterations where one GCD takes 17.  Charged for
  the same operations, 512 nodes keep 88 % of the rate of one node and 76 % of one GCD's.
- **The index tables of an assembly in half the host memory** (`fem.pattern.LEAN`,
  `HIPPYMFEM_PATTERN_LEAN`, on).  A rank held 12.5 GB of host memory for every 262 144
  quadratic hexahedra, two thirds of it index tables: 50 GiB at the peak of their builds
  for a million elements, so that the eight ranks of a node with 512 GB could not take
  more.  The graph of a block and the slot of every element-matrix entry now come from the
  two dof tables of its element group (`patternbuild.build_tables`: the flat row and column
  of every entry, 16 bytes each, are not formed); the true-dof graph is laid out one row at
  a time from the rank's own row and the entries that arrived for it
  (`patternbuild.tdof_graph`: the global row and column of every nonzero, their sorted
  copies and the permutations between them are not formed, and the slot of a nonzero is
  kept in 32 bits below two billion); the compact scatter map is made in slabs; and a map
  with one entry per element-matrix entry is given up when what is made from it exists, the
  ldof pattern's once its true-dof target has its own and the target's once the compact map
  is made (`ScatterPattern.release_slot`; a map given up is made again when asked for).
  Every array is the same (`test_assembly` compares them on 1 to 8 ranks, with the maps
  given up and made again), and two Newton-CG steps give the same cost functional and
  counts.  With a million elements a rank the peak is 28 GiB instead of 50 and the tables
  13 GiB instead of 30; the Jacobian's pattern is built in 11 s instead of 15 s and its
  true-dof layout in 5 s instead of 14 s.  1.7 million elements a rank (120^3) take 41 to
  43 GiB, on one GCD as on 4096.  `HIPPYMFEM_PATTERN_LEAN=0` builds and keeps the tables as
  before.
- **`tools/hypre-3.2.0-staging.patch`** (`HYPRE_PINNED_STAGING=1
  tools/build_pymfem_hip.sh`): for the hypre of the HIP build, the page-locked staging
  buffers of the 2.32.0 patch and a staged exchange that starts before the product with
  the local block instead of after it, so that the copy to the host and the sends overlap
  with that kernel.  An iteration at 128^3 on 8 MI250X GCDs takes 9.9 ms instead of
  11.2 ms (10.7 with the buffers alone, 10.6 with the early start alone), and the two
  steps at 512^3 on 512 take 10.0 s instead of 10.8 s: 59 % of one GCD's rate, 77 % of
  one node's, and 71 % with two blocks a GCD on 256.  `HYPRE_STAGE_PINNED=0` and
  `HYPRE_STAGE_EARLY=0` in the environment of a run switch either off.  The suites pass
  with the patched library preloaded.
- **`tools/hypre-3.2.0-spgemm.patch`** (`HYPRE_SORTED_SPGEMM=1 tools/build_pymfem_hip.sh`):
  the sparse products of a BoomerAMG setup on AMD cards without their sorts.  hypre hands
  rocSPARSE both operands of a product with their rows in ascending order, and gets there
  by copying each operand and sorting the copy, with a permutation and a second copy of the
  values beside it, before every product.  Most operands of a setup are sorted as they come
  (a transpose, the result of the product before), and the fine matrix is sorted but for
  the diagonal entry that hypre keeps first in each row.  The patch checks the rows of an
  operand in one pass and takes a sorted one as it is, writes the sorted copy of a matrix
  with only its first entries out of place in one pass (a warp a row; a first attempt with
  a thread a row was slower than rocSPARSE's sort), writes the matrix that joins a diagonal
  block with its off-processor one in order in the first place, and sorts the rows a rank
  receives before it stacks them under its own.  Any other matrix is copied and sorted as
  before.  It saves those copies and sorts and the memory they took: in two Newton-CG steps
  with 8.6 million state unknowns a GCD the hypre of one rank took 163 to 311 GiB from the
  driver and now takes 91 to 165 GiB.  A setup at that size takes 0.81 s instead of 1.53 s
  on one GCD and 1.04 to 1.17 s instead of 2.31 to 2.52 s on two to eight; a forward solve
  4.49 s instead of 5.37 s and 5.12 s instead of 6.45 s; hypre's device memory peaks at
  13 GiB instead of 23 on one GCD and at 15 instead of 31 on several; the cost functional
  is the same to nine digits and the CG counts are the same.  The series with four 64^3
  blocks a GCD, 102^3 on one GCD to 1024^3 on 1024, takes 6 to 15 % less time with it as
  measured (two steps at 1024^3 in 30.2 s instead of 33.0 s) and 9.5 to 14 % less charged
  for the same operations at every size; 1024 GCDs then keep 78 % of one GCD's rate instead
  of 76 %, and 88 % of one node's instead of 90 %, since a setup lost 0.7 s on one GCD and
  1.1 to 1.4 s on any more, on one node as on 128.  With hypre at half its memory that load
  no longer needs JAX held to a fifth of the card and the linearization released: with the
  library's own share the operations charged take 27 s on one GCD and 32 s on eight, where
  they took 44 and 51 s.  `HYPRE_SPGEMM_SKIP_SORT=0` in the environment of a run has hypre
  copy and sort every operand again (`1` leaves out the one-pass copy and the ordered join,
  `2` the join).  Every suite passes on a build with it.
- **hypre's device memory pool on AMD cards** (`hipMalloc`; worth 0 to 2 % of two Newton
  steps on up to 32 MI250X.  `HyprePool` now counts the driver's time and bytes,
  `driver_seconds` and `driver_bytes`, and the card's least free memory, `min_free`, and
  the Newton benchmark records them for each stage: the driver's time is nearly all in
  giving blocks back, half of it the wait for kernels still running, and a pool of 24 GiB
  that keeps the large blocks takes it from 0.26 to 0.10 s a forward solve and the solve
  from 4.36 to 4.28 s), the MPI library's own answer about device memory on a HIP build
  (`MPIX_Query_hip_support`), and `HIPPYMFEM_GPU_AWARE_MPI=0|1`, which sets hypre's
  GPU-aware exchange whatever the library was built with (from hypre 2.33 on: see "Fixed"
  for 2.31 and 2.32).  `=solve` (experimental) turns it
  on only inside Krylov solves on a hierarchy that is set up, since a setup's exchanges
  each cost Cray MPICH a registration.  With Cray MPICH 9.1.0 it takes 10 to 20 % off an
  iteration and deadlocks or faults every time at 320^3 on 128 and at 400^3 on 256 GCDs, so
  nothing above uses it.
- **Benchmarks.**  `bench_newton_device.py --coarse`, `--amg`, `--prior-amg`, and in its
  record the calls and the seconds of every solver and stage of the two steps, which is
  what the charging above reads; `--single-refine-goal` and `--single-give-up`; the
  precision of a run, whether
  its solves were in the single-precision hypre, JAX's share of the card and what its
  allocator held at the most; and the norms of the state, the adjoint, the gradient and
  one reduced-Hessian action at the prior mean, which two runs of a problem agree on
  whatever their precision (the cost functional after two steps is not such a number: a
  line search that backtracks takes another path for a direction that differs in a late
  digit).  `bench_scaling.py --coarse`, `--m-scale`, `--amg`, and `--profile-vcycle`, a
  V-cycle on hierarchies cut at 1, 2, ... levels.
- **The geothermal application builds its mesh in parallel** (`model.cube_mesh`,
  `run.py --coarse`).  Every rank used to build and partition the whole mesh: 635 bytes
  an element while METIS partitions it, which is 42 s and 10 GB a rank at 256^3 on 32
  ranks and would be 79 GB a rank at 512^3, where the eight ranks of a Frontier node have
  512 GB between them.  Now every rank builds a coarse mesh that keeps 512 elements a
  rank, takes its own box of it (the ranks as a grid of equal boxes, 4 x 8 x 8 on 256;
  METIS when they make no grid) and cuts its elements into up to 8^3 with
  `ParMesh.MakeRefined`: 0.6 s and 0.3 GB a rank at 256^3, 4 s at 512^3 on 256 ranks.
  The vertices are the same lattice to the last bit and the boundary attributes the same
  faces; `validate.py mesh` checks it on any number of ranks, by sums (each lattice point
  once, the volume of every element, the faces of each attribute).  At 12^3 the dumps on
  1, 2 and 4 ranks agree with that of the whole mesh (the data to 7e-16, the MAP point to
  1e-7).  At 256^3 on 32 MI250X every rank holds 524 288 elements, the build takes 160 s
  instead of 207 s, and the MAP the same 17 Newton and 359 CG iterations in 643 s instead
  of 649 s; with METIS on the coarse mesh the largest part was 2.9 % above the mean and
  the MAP took 659 s.  With it the application ran at 512^3 on 256 MI250X, 1.08 billion
  state unknowns, in 59 minutes end to end.  `--coarse N`, with `N` the size of the mesh,
  builds the whole mesh on every rank as before.
- **The geothermal application draws its pictures and its animation** from the dump of a
  run, without repeating the MAP solve (`applications/geothermal/README.md`).
  `figures.py` also writes `<out>_block.png`, the truth, the MAP estimate and the
  posterior standard deviation on the block with a quarter cut away, which is the
  picture of the application's README.  `movie.py` draws the animation of the
  repository's README with PyVista: an MP4, an animated WebP and a still, for a light and
  for a dark page, with an opening (`--reveal`), the camera's swing or a full turn
  (`--turn`), and lines and sentences under the pictures (`--hardware`, `--say`).  From a
  dump alone it shows the true rock, the MAP estimate, the truth minus the MAP and the
  posterior standard deviation.  `movie_data.py fields` adds what the README's animation
  shows beyond a dump, the heat flowing through the true rock and samples of the
  posterior: it rebuilds the problem, takes the MAP point from the dump and computes the
  eigenpairs there as `run.py` does (at 12^3, on as many ranks as the run, its
  eigenvalues are the dump's to 3e-14).  `movie_data.py forward` solves the
  forward problem alone on a finer mesh and keeps its temperature on the inversion's
  grid, for the large picture.  `movie.py` needs neither MFEM nor JAX; it was run with
  PyVista 0.46 on a GPU and with PyVista 0.49 and the software renderer of `vtk-osmesa`
  9.3 on a node without one, and it takes ffmpeg from the path or from `imageio-ffmpeg`.
- **The geometry of an affine element group once an element** (`HIPPYMFEM_AFFINE_GEOMETRY=1`,
  `hm.config.affine_geometry`, `fem.elementbatch.AFFINE_GEOMETRY`; off by default).  A
  kernel reads the inverse Jacobians at every quadrature point, 4.6 kB an element with 64
  points, and a group too large to keep them on the card streams them to it with every
  chunk of every pass: with 1.7 million quadratic hexahedra on an MI250X that stream was
  four fifths of a residual (0.46 s, 0.09 s with the geometry on the card), and the eight
  ranks of a node, streaming at once, were 5 to 33 % slower in their element passes than a
  rank alone.  Where every element of a group is mapped affinely (straight simplices,
  parallelepipeds) the inverse Jacobian is one an element: `ElementGroup` keeps it as
  `(ne, 1, dim, sdim)` (the mean over the points, which agree to 1e-12; `group.affine`,
  `group.full_Jinv()`), the evaluators apply the one matrix, and an affine group keeps on
  the card what is small (the inverse Jacobians and the weights, 1.0 GB there instead of
  11.5 GB) and leaves the coordinates on the host for the kernels that read them
  (`GroupKernel._kept_on_device`).  On eight MI250X in double precision two Newton-CG
  steps of the benchmark take 59.8 s instead of 72.2 s and a rank holds 34.5 GiB of host
  memory instead of 41.9 GiB; in the mixed mode, with the freed card given to JAX
  (`HIPPYMFEM_GPU_MEM_FRACTION=0.45`), their operations take 34.4 s on eight GCDs and
  29.2 s on one, where the series to 4096 GCDs of the same morning took 73.6 and 62.9 s.
  Eight 64^3 blocks a GCD then run on a node in 39 s for two steps with 11 GiB of a card
  left.  The element arrays change at round-off (the state's norm in the twelfth digit),
  which is why it is opt-in; every suite passes with it on (`test_kernels` compares the
  two forms by either route to the second derivatives and in the mixed precision, and
  that a curved group keeps a Jacobian a point).
- **The single-precision solves on AMD cards, and with 64-bit global indices**
  (`algorithms.singlesolve`, `tools/build_hypre_single.sh <prefix of build_pymfem_hip.sh>
  <directory>`).  The library refused a single-precision hypre on a HIP build, its build
  script knew PyMFEM's CMake build of hypre only, and its matrices had 32-bit global
  indices.  The script now also takes the prefix of the HIP stack: it copies that stack's
  hypre source, patches and all, configures the copy with the options the installed build
  recorded and `--enable-single`, and links it `-Bsymbolic` under a soname of its own.
  `singlesolve` reads the width of `HYPRE_BigInt` from the offsets file and passes global
  sizes, partitions and column maps in it (a library with 32-bit global indices refuses a
  matrix of more than 2 147 483 647 rows), sets the host policy in a GPU build that runs
  under MFEM on the host, and gives a hierarchy the solver's `amg_options`
  (`HIPPYMFEM_AMG`) before `HIPPYMFEM_SINGLE_AMG`, by one table of names
  (`mfemconfig.BOOMERAMG_SETTERS`).  Every suite passes with the library named on HIP
  stacks of both index widths, on eight ranks too, which two checks had to learn: the
  chunk that `single_case` forces is now smaller than a rank's elements (125 of them on
  eight ranks did not split at 128), and the eigen-residual of `test_uq`'s
  posterior-preconditioned pairs has the tolerance of an action under single-precision
  solves (1.9e-6 on eight ranks, 7e-11 in double precision).  On the MI250X of Frontier at 120^3
  quadratic hexahedra a GCD (14 million state unknowns, incremental solves to 1e-5 in both
  precisions): an iteration of a linearized solve 47 ms instead of 61 ms on eight GCDs, a
  BoomerAMG setup 1.7 s instead of 2.4 s, 16.6 GiB of a card free at its fullest instead
  of 7.3 GiB; the state, the adjoint and the gradient within 4e-10 of double precision.
  A refined forward or adjoint solve takes two more residuals from the element kernels, so
  with the same settings two Newton-CG steps take as long as in double precision; with the
  freed memory given to JAX (`HIPPYMFEM_GPU_MEM_FRACTION=0.35`) their operations take
  49.7 s on eight GCDs instead of 62.0 s and 43.7 s on one instead of 54.4 s (45.8 s on
  eight with `HIPPYMFEM_AD_WORKING_SET=8`, the size of the matrices' tangents in that mode,
  which the chunk planner does not assume by itself).  Eight 64^3
  blocks a GCD, which a node does not hold in double precision, fit: 256^3 on eight GCDs
  (287 million unknowns) in 57 s for two steps.  The library is to be named together with
  `HIPPYMFEM_PRECISION=mixed`: from double-precision element matrices the Jacobian is not
  corrected on the constants, and the cost functional after two steps at 102^3 was 1.6 %
  off where the mixed mode was 3e-4 off.  In the mixed mode with the compact geometry of
  an affine mesh (the entry above) the benchmark's operations take half the time of the
  double-precision series at every size from 1 to 512 GCDs (41.1 s instead of 80.9 s at
  960^3, 15.1 billion unknowns), with a lower ratio to one GCD, 71 % instead of 79 %:
  both series lose the same 6.7 s from one node to 64, in the solver.
- **A p-multigrid in the Jacobian's single-precision solves** (`HIPPYMFEM_PMG=1`, the
  solvers' parameter `pmg`, `hm.config.pmg`; `algorithms.singlesolve.PMGEngine` and
  `PTransfer`, `fem.ptransfer.interpolation`; off by default).  BoomerAMG on the matrix of
  quadratic elements coarsens it, builds an interpolation for it and forms the Galerkin
  product: a third of a forward solve at a new parameter.  The linear elements of the same
  mesh are a coarse space that needs none of that: their interpolation `P` to the state
  space follows from the two spaces, `P^T A P` is the matrix of the same form on them (an
  eighth of the rows, a twentieth of the entries; the identity on the essential dofs), and
  BoomerAMG is set up on it.  An application relaxes on the Jacobian as the finest level
  of BoomerAMG would (l1-Jacobi with the weight `relax_wt`, Jacobi with `relax=7`), takes
  one V-cycle of that hierarchy on the restricted residual, and relaxes again; the steps
  are calls of the library from Python inside hypre's PCG.  The product `A P`, which the
  Galerkin product passes through, is kept (`singlesolve.PMG_KEEP_AP`,
  `HIPPYMFEM_PMG_KEEP_AP=0` to give its 2 bytes a matrix entry back): the residual after
  the coarse correction is `t - (A P) e`, a product with a quarter of the matrix's entries
  in place of the third product with the matrix that an iteration would make.  On the
  MI250X of Frontier at 14 million state unknowns a GCD: the same six or seven iterations
  to 1e-5 as BoomerAMG at every parameter of two Newton-CG steps, each 31 ms instead of
  43 ms on one GCD and 34 instead of 48 ms on eight (40 and 44 ms without the kept
  product); a set-up of 0.39 s instead of 1.18 s and of 0.67 s instead of 1.60 s; the two
  steps' operations in 21.8 s instead of 29.4 s on one GCD and 26.3 s instead of 34.4 s
  on eight, with the state, the adjoint and the gradient at the prior mean within 6e-11
  and 2.3 GiB less of a card held by hypre.  To 512 GCDs (960^3, 15.1 billion unknowns),
  measured before the product was kept, the operations took 15 to 19 % less time at every
  size, 35.1 s instead of 41.1 s there, with the ratio to one GCD where it was (69 %; 72 %
  with BoomerAMG on the matrix).  `interpolation` is for the
  nodal H1 spaces of order two or more on a conforming mesh, hexahedra, tetrahedra,
  quadrilaterals and triangles (`single_case` checks that it carries a linear function to
  itself for orders two and three, that the coarse matrix is the Jacobian of the form on
  the linear elements, and that the forward solve and Newton-CG give the state and the
  MAP point of double precision); a problem whose state space allows none keeps
  BoomerAMG, with a warning.  The hierarchy's settings are in one function for both
  engines (`singlesolve._boomeramg`).  Tried beside it and not kept: the hierarchy of a
  linearization point as the preconditioner of the trial points of its line search,
  which failed at every one of fourteen (the parameter moves by 1 to 54 in the maximum
  norm; 200 iterations, or a preconditioner that is no longer positive definite).
- **The products of a p-multigrid from the Jacobian's element matrices**
  (`singlesolve.PMG_ELEMENT_PRODUCTS`, on; `HIPPYMFEM_PMG_ELEMENT_PRODUCTS=0`;
  `PDEVariationalProblem._jacobian_with_products`, `fem.csrassemble.scatter_derived`).
  `A P` and `P^T A P` are sums over the elements of `A_e P_e` and `P_e^T A_e P_e`, with
  the interpolation `P_e` of one element (27 by 8 for a quadratic hexahedron).  For a
  scalar state on a domain term the pass that assembles the Jacobian scatters the three
  matrices, each into its own pattern, with the essential dofs eliminated alike, and the
  engine takes the two products with the matrix instead of asking hypre for them.  They
  are the matrices hypre's products give (`single_case` compares the two engines and the
  products on a vector).  On the MI250X of Frontier at 14 million state unknowns a GCD an
  assembly takes 0.10 s more on one GCD and 0.16 s more on eight, a set-up 0.18 s
  instead of 0.39 s and 0.29 s instead of 0.68 s (0.42 s on 4096 GCDs, where hypre's
  products made it 0.80 s), and two Newton-CG steps 3 to 4 % less.
- **The sums of the rows of a single-precision Jacobian** (`HIPPYMFEM_SINGLE_ROW_SUMS=1`,
  `fem.tdofassemble.SINGLE_ROW_SUMS`, off by default;
  `algorithms.singlesolve.RowDefect`, `_Library.pcg`).  The element matrices of the mixed
  precision are exact on the constants, but the matrix of the single-precision library
  rounds every entry to 24 bits and the sum of a row is off again, by about 3e-8 of its
  diagonal.  On a mesh of like elements with a constant coefficient every like row is
  rounded alike, the matrix has a term of order zero with one sign over the domain, and a
  solve with it is off, relative to a smooth solution, by the rounding unit times the
  square of the number of nodes across the domain: about 7e-3 at 240^3 second-order
  hexahedra, and of order one half at 1920^3.  The refined solves remove it at a price
  that grows with the mesh (3 passes up to 240^3, 4.3 on average at 1920^3 on 4096 GCDs
  of Frontier), the incremental solves of a Hessian action keep it, and two Newton-CG
  steps from a constant prior mean ended at 0.5 to 1.7 times the cost that double
  precision reaches.  With the switch on the Jacobian is accumulated in double precision,
  what the rounding takes from each row's sum is kept as a vector of the library (the
  rounding residuals put in the matrix, multiplied by ones, and the values written
  after), and hypre's PCG is built on its table of functions for a parallel matrix with
  the product replaced by one that adds that vector times the argument.  Measured: the
  residual after a refinement's second pass at the prior mean from 2.1e-9 of the first to
  1e-10 at 240^3 (to the solver's own limit; a sample of the prior was there already);
  three passes at every size to 1920^3, and the cost of double precision after two steps
  to within about 2 % at every size (579.22 against 579.35 at 1920^3, 120 billion unknowns,
  where the run without it had 972).  It costs 0.1 s an assembly at 14 million unknowns
  a card: the accumulator in double precision for the duration of the assembly, two
  passes over it and one product.  `bench_newton_device.py` records the switch, the
  passes of its two stage solves (`stage_passes`), the route of the element Hessians and
  whether the p-multigrid's products came from the element matrices.
  On an MI250X a conversion to single precision and back inside one compiled program is
  taken for the identity (`x - x.astype(float32).astype(float64)` is zero there): the
  rounding residual is computed in two programs.
- **A matrix of the single-precision library from a double-precision device
  accumulator is rounded a piece at a time** (`singlesolve._put_single`, `PIECE`), where
  one expression copied the accumulator and its rounded values whole, and its range is
  checked a piece at a time (`_all_single`).
- **A prior with a lumped mass matrix** (`BiLaplacianPrior(lumped_mass=True)`,
  `SqrtPrecisionPDE_Prior`; `fem.elementbatch.VERTEX_RULE`, `integration_rule`; off by
  default).  `R = A M^-1 A` solves with the mass matrix at every product, which the
  cost, the gradient and each Hessian action take: 61 to 70 CG iterations with the
  consistent mass of first-order hexahedra, 50 to 80 ms on an MI250X at 1.8 million
  parameter unknowns, as much as a solve with `R` to 1e-9.  With the option the mass
  matrix is integrated by the rule at the vertices of an element, which makes it diagonal
  (the sums of the rows of the consistent one on affine elements: the lumped mass of the
  SPDE approach of Lindgren, Rue and Lindstrom), its solve is one iteration, and the
  square root that samples are drawn with is by the same rule, so that `R^-1` is their
  covariance exactly.  Another discretization of the same prior, with another matrix
  `R`; for a parameter space of first order on a mesh of one kind of element
  (`test_modeling.test_lumped_mass_prior`, hexahedra and tetrahedra, on one, two and
  eight ranks).  `bench_newton_device.py --prior-lumped-mass`: the operations of two
  Newton-CG steps on Frontier in 18.3 s instead of 19.4 s on one MI250X, 22.2 s instead
  of 22.5 s on eight, 23.9 s instead of 24.9 s on 64 and 25.7 s instead of 26.5 s on 512
  (960^3 elements, the two run side by side).
- **The rows of other ranks leave a device assembly by messages to the neighbours**
  (`fem.tdofassemble.DEVICE_EXCHANGE`, `HIPPYMFEM_DEVICE_EXCHANGE`: `neighbors`, the
  default, or `alltoallv`; `TrueDofPattern._exchange_messages`,
  `devicebridge.copy_to_host`).  The entries an assembly computes for rows that another
  rank owns went from the device through a slice and a fresh host array into one
  `Alltoallv` in double precision, and back through a fresh array.  They now go from the
  accumulator's own memory into a page-locked buffer that the pattern keeps, to each
  owner in a message of its own, and back from a second kept buffer, in the
  accumulator's precision.  On 64 MI250X of Frontier with 14 million quadratic-hexahedron
  unknowns a rank (66 MB to send for the Jacobian) the exchange of the Jacobian and of
  the two products of its p-multigrid takes 18 ms instead of 166 ms an assembly, and the
  operations of two Newton-CG steps 24.9 s instead of 26.4 s (22.5 instead of 23.1 s on
  eight; the same on one).  On 4096 MI250X (1920^3 elements, 120.4 billion unknowns) they
  take 29.2 s instead of 31.2 s, a forward solve at a new parameter 2.75 s instead of
  3.17 s, and the two steps 23.6 s: 66 % of the rate of one MI250X and 77 % of that of
  one node, where the series before it kept 62 and 74 %.  The ranks of a pattern agree
  on the way when it is made (`TrueDofPattern.messages`: messages where every rank
  finishes its matrices on the device, the collective otherwise, as on the host), and
  both ways are open to a host and to a device accumulator: in an assembly over
  elements that some ranks do not have, such as a boundary's, those ranks hold host
  accumulators beside the others' device ones.  A message carries its sender's
  precision, which the receiver reads from its length
  (`test_device.test_exchange_routes_agree`, `test_exchange_of_mixed_accumulators`).
- **The true-dof maps of a space are kept** (`fem.tdofassemble._tdof_maps`): every pattern
  of a problem asked for those of the same two or three spaces again, on a device by one
  SWIG call a dof, ten seconds for fourteen million dofs.
- **A forward solve that single precision cannot carry may raise instead of going to
  double precision** (`PDEVariationalProblem.SINGLE_GIVE_UP = "raise"`; `"double"`, the
  default, is the behaviour so far; `bench_newton_device.py --single-give-up`).  The
  fallback assembles the Jacobian in double precision and builds its hierarchy beside the
  single-precision ones, three times their size on the card, and a problem sized for the
  mixed mode may not have that room: with 1.7 million quadratic hexahedra a rank on a
  64 GiB MI250X (600^3 on 128 GCDs) a trial point of a line search at which the
  single-precision solver gave up ended the run out of device memory.  With `"raise"`
  the forward solve raises a `RuntimeError`, on every rank, which the line searches take
  for a failed trial and answer by backtracking; the same run then finished with one
  trial refused.  An adjoint solve is at a point the line search accepted and goes to
  double precision either way.  `single_case` (in `test_solvers` and `test_device`)
  checks both ways out on a preconditioner made indefinite.
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

- **One reading of the on/off settings.**  Every on/off `HIPPYMFEM_*` variable is read by
  `config.parse_flag`, at import and by its `hm.config` setter alike: 1, yes, true, on and
  0, no, false, off in any case, an empty value for the default, and anything else refused
  with the variable's name.  The auto/on/off ones (`HIPPYMFEM_TDOF_IDENTITY`,
  `_DEVICE_BRIDGE`, `_DEVICE_VECTORS`, `_DIRECT_TRANSPOSES`) are read by
  `config.parse_tristate`.  Before, an empty `HIPPYMFEM_SINGLE_ROW_SUMS` or
  `HIPPYMFEM_AFFINE_GEOMETRY` turned the setting off although both are on by default, an
  unknown word was read as on by some modules and as off by others, and
  `HIPPYMFEM_DEVICE_VECTORS=off` left the device vectors on.
- **A knob set through `hm.config` outranks a profile.**  `hm.config.use_profile` applied
  afterwards no longer overrides it, and `hm.config.show()` reports it as `set` instead of
  `default`.
- **`benchmarks/perf_check.py` keeps one record per configuration.**  A record's
  configuration is every `HIPPYMFEM_*` variable but those that only say where things are
  or what to print, `XLA_FLAGS`, and the benchmark's options; a run is compared only with
  the record of its own configuration and rank count.  With none to compare with it exits
  with 2, which the GPU workflow reports as a warning instead of passing without a word.
- **Defaults from the scaling work on Frontier.**  The compact geometry of affine element
  groups (`HIPPYMFEM_AFFINE_GEOMETRY`) is on: one inverse Jacobian an element where every
  element of a group is mapped affinely, which on a node of eight MI250X took two
  Newton-CG steps at 120^3 elements a GCD from 62.0 to 51.3 s in double precision and
  from 45.8 to 34.4 s in mixed precision, with 7 GiB less host memory a rank; the element
  arrays change at round-off (`=0` for the arrays of before).  Where the single-precision
  hypre is loaded, the Jacobian keeps the defect of its row sums
  (`HIPPYMFEM_SINGLE_ROW_SUMS`; without it the refined solves needed more passes from 64
  GCDs on); the blocks of a linearization point as its matrices (`HIPPYMFEM_SINGLE_BLOCKS`;
  1.6 % less time on 4096 GCDs and 4.8 GiB more of a card free) are in the profile
  `gpu-large`, since their identities then hold to single precision only.  A forward solve that single precision cannot carry is solved again in double
  precision only where that Jacobian and its hierarchy fit on the device, and raised for
  the line search to backtrack where they do not (`SINGLE_GIVE_UP = "auto"`,
  `HIPPYMFEM_SINGLE_GIVE_UP`): the old default, `"double"`, ran a trial of 600^3 on 128
  GCDs out of device memory.

- **The seismic propagator numbers a rank's unknowns in the order its elements meet them**
  (`sem.first_touch`, with `--keys`), not in the order of their keys.  The gather and
  the scatter of every step read and write the unknowns of neighbouring elements, and
  the elements of a refined mesh come in an order that keeps neighbours together.  On an
  MI250X a step of 211 million unknowns takes 154.1 ms in the place of 161.3 (two fields
  184.9 for 196.4); numbered at random a step of 27 million takes twice as long (22 and
  44 ms on an MI210).  The
  ranks of a run, whose pieces of the mesh lie differently across the keys, also differ
  less: on 512 GPUs a rank's own time was 157.6 to 167.0 ms and is 146.9 to 152.6, which
  took the weak scaling there from 96 to 100 % of one GPU.  Results are the same numbers
  in another order (the receivers' extreme and the shared unknowns of every size are
  unchanged, and the adjoint identity holds to round-off as before).
- **The seismic application's set-up and its exchange through the host.**  The 64 keys of
  each element come from a compiled loop where the library would use one (40 s a rank at
  7.8 million elements became 4); the search for the elements of the sources and
  receivers looks at the rank's own box first and then at one coordinate (29 s became
  4); the coarse mesh is handed to MFEM as arrays; `scale.py --dt-limit` takes the
  stability limit from a run before.  The set-up of a run of 211 million unknowns a rank
  on 512 GPUs takes 70 s where it took 141.  Through host arrays the
  receives are posted before the element work, into arrays that are kept, in the fields'
  own precision, and what a rank sends comes from its card in pieces of 16 MiB
  (`HIPPYMFEM_SEM_PIECE`; glibc maps new memory for every array above 32 MiB): the
  two-field sweeps on 512 GPUs take 213.8 and 222.7 ms where they took 256.7 and 266.0
  (11 ms of that is the numbering of the entry above).
  The mass perturbations and the fold of the backward sweep are computed a piece of
  elements at a time, the power iteration of the stable step stays in the fields'
  precision, and `SeismicModel` gives up the final states of a linearization point it has
  left.  `scale.py` prints and records, over the ranks, a rank's own time, its wait for
  the element work, its copies and its time in MPI, with the neighbours and the values
  sent.
- **The ice sheet's solver without a matrix on AMD cards** (`matrixfree.ColumnStokes.dots`,
  `_probe`, `probe_bytes`).  The two classical Gram-Schmidt passes of FGMRES multiplied
  the Krylov vectors with the new one through the card's BLAS, and hipBLAS takes a thread
  a row for a few long rows: 170 ms for 4 to 40 rows of 4.7 million, 8 ms for one dot
  product (on an MI210; on an MI250X the change took 0.37 s off an iteration).  Sums
  over stretches of 1024 entries by one reduction kernel, then the sum of those, take 0.3
  to 2 ms and 0.2 ms, on any card.  The probing of the Schur complement
  takes several levels of a colour in one product (as many as `probe_bytes` allow, the
  same number on every rank): the pass over the element matrices of `B` and the launches
  of the solves on the columns serve them all.  At 24 levels and 4.7 million unknowns on
  one MI250X GPU a set-up takes 5.4 s in the place of 11.5 and a Krylov iteration 0.23 s
  in the place of 0.60, with the Krylov counts of a forward solve unchanged step by
  step; at 88 levels the probing is still 124 of a set-up's 133 s.  The reductions of an
  iteration are MPI's own.
- **The ice sheet's set-up** (`antarctica.Columns.mesh3d`, `mesh2d`;
  `IceSheet._footprint_map`, `set_checkpoint`; and in the library
  `FunctionSpace._node_coordinates`, `BoundaryGroup._build`).  The serial meshes are
  handed to MFEM as arrays (27 s became 4 at 5.3 million hexahedra); the footprint is cut
  as the columns are and its map to the bed is made from the rank's own nodes, where every
  rank had gathered and sorted every footprint node of the mesh (495 MB to each of 4096
  ranks); the coordinates of a space's nodes are one product for all elements and are kept
  by the space until the mesh moves (identical to the last bit, 6 to 11 times faster, and
  computed once a space however often they are asked for); the inverses and the normals of
  a group of boundary faces are taken for all points at once; a checkpointed solve writes
  its iterate when it stops or ends, not after every Newton step (`--checkpoint-every` for
  more).  The set-up of a problem of 4.6 million unknowns a rank takes 10 s where it took
  27, and 25 s on 4096 ranks.
- **The Newton benchmark takes the small geometric levels to host memory and the prior's
  cycles over the meshes by default** (`bench_newton_device.py --pmg-h-host`, `--prior-h`).
  For a mesh built by cuts (`--h-levels`) the solvers' `pmg_h_host` is 5000 and the prior's
  `h_levels` is the number of cuts, the latter where the last cut is in two and the
  single-precision hypre is loaded: the configuration of the eighth series on Frontier.
  `HIPPYMFEM_PMG_H_HOST` and `HIPPYMFEM_PRIOR_H` in the environment and the two options
  override it, and `0` for both is the configuration of the seventh series.  The
  library's own defaults are as they were: both off.  The record has what was in force
  (`pmg_h_host_setting`, `prior_h_setting`).
- **The clock on a p-multigrid's cycle takes several levels** (`HIPPYMFEM_PMG_TIMERS=0,1,2,3`):
  one clock inside the other, so that the differences are what each level takes by
  itself, and the cycles of a Jacobian's p-multigrid and of a prior are tallied apart
  (`singlesolve.TIMERS`, the record's `pmg_h_timer`, now a list).
- **The reductions and gathers of the Newton steps are MPI's own** (`parvector.allreduce`,
  `Layout.of_range`, `ParVector.offsets`; `singlesolve._Library.check` and `agree`;
  `linSolvers`, `PDEVariationalProblem`, `fem.assemble`).  mpi4py's `allreduce` and
  `allgather` pickle what they send, a tree of messages up and down and one object a
  rank: 79 µs a reduction and 289 µs a gather on 512 ranks where MPI's own on a buffer
  take 34 and 65 µs.  Two Newton-CG steps made some 450 of them: every inner product and
  norm of a `ParVector`, the check after every call of the single-precision library that
  each rank's succeeded, an agreement on a failure at every single-precision solve, and a
  gather of every rank's size for each vector made without a layout.  They are now
  reductions on buffers; a vector made for a matrix takes its layout from the matrix's
  row range, with one gather of the sizes on a buffer; and an agreement reduces one flag
  and gathers the messages only when a
  rank has failed.  On 4096 ranks a pickled gather takes 1.64 ms where MPI's own takes
  0.13 ms, and the pickled calls of two steps came to 0.11 s.
- **The vectors of a Newton step stay on the device** (`CGSolverSteihaug._solve`,
  `ChebyshevSolver.solve`, `PDEVariationalProblem.solveIncremental`).  A `ParVector`'s
  arithmetic runs where the vector was last used: on the device once it has been handed
  to MFEM or hypre, on the host until then, where every operand that is current on the
  device is first copied back.  The CG of a Newton step never handed its iterate to
  MFEM, and its residual only through the preconditioner; the polynomial preconditioner
  of the entry above copied that residual instead of passing it on, so with it every
  update and inner product of the reorthogonalization ran on the host (42 inner
  products at 2.5 ms and 42 updates at 2.2 ms in two steps on eight MI250X GCDs, where
  the same on the device take 0.1 ms; `tools/trace_vectors.py`, run around a script,
  lists every vector operation of the Newton steps by where it ran and who called it).
  Host arithmetic has a long tail over the ranks of a large run: what the CG did besides
  the Hessian and the preconditioner was 26 ms an iteration on one GCD and 87 ms on
  4096.  The CG now hands its six vectors to MFEM when it starts, the polynomial its
  right-hand side, and the incremental solve its output before it is zeroed (27 ms for a
  host vector of 14 million unknowns, and an upload); two parameter vectors are compared
  by the Euclidean norm of their difference, which is reduced on the device.  Nothing
  changes without a device.  With the polynomial's cycles applied from the iterate, the
  charged operations of the Newton benchmark went from 16.7, 19.6, 20.4 and 21.6 s to
  16.0, 18.9, 19.9 and 20.9 s on 1, 8, 64 and 512 GCDs, with the same cost functional.
- **One cycle and one product less in every single-precision solve with the p-multigrid**
  (`PMGEngine`, `RowDefect.zero_guess`).  hypre's PCG applies the preconditioner to the
  right-hand side, for the norm that it stops on, then forms the residual of its guess
  with a product, then applies the preconditioner to that residual.  An engine's guess
  is zero, so the residual is the right-hand side and the second application repeats the
  first: 15 ms for the cycle and 9 ms for the product at 14 million unknowns on an
  MI250X, of a solve of 230 ms, in each of the 32 solves that two Newton-CG steps charge.
  The engine now keeps the first cycle's result for the second, and the product that a
  matrix with its row sums kept brings to the PCG returns at once for a zero guess.  The
  iterates are the same to the last bit.  The benchmark's operations: 15.2, 18.2, 18.9
  and 19.8 s on 1, 8, 64 and 512 GCDs.  The series in that tree, with the gathered
  levels and the polynomial preconditioner of the two entries above: 15.4 s on one GCD
  and **20.9 s on 4096, at 1920^3 elements and 120.4 billion unknowns, where two
  Newton-CG steps take 16.4 s** (24.6 s and 19.8 s in the series before those two
  entries, 84.4 s and 66.0 s in double precision): 74 % of the rate of one GCD and 87 %
  of that of one node (68 and 82 % before; 76 and 88 % in double precision), with 5.5 s
  lost between one GCD and 4096 where there were 7.9 s, and 20.3 s in double precision.
- **Element vectors run in larger chunks** (`fem.kernel.AD_VECTOR_TANGENTS`,
  `HIPPYMFEM_AD_VECTOR_TANGENTS`, 8).  The planner sized every chunk for the element
  matrices, which carry a tangent for every dof of every variable: 508 kB an element with
  quadratic hexahedra.  A residual or a gradient is one reverse sweep and was measured at
  16 kB an element, so with the card shared with hypre it ran in hundreds of launches of
  1.5 ms each.  A vector is now planned for eight tangents (a margin of four on that
  measurement, never more than the matrices' plan, and with the same retry when an
  estimate is too small).  With 1.7 million quadratic hexahedra on an MI250X and a quarter
  of the card for JAX a residual takes 0.47 s instead of 0.70 s and a gradient 0.56 s
  instead of 1.02 s, and the operations of two Newton-CG steps 5 to 7 % less, in double
  precision and in the mixed mode alike; the results are those of the smaller chunks to
  round-off.  `0` plans a vector as a matrix, as before.
- **The measurements quoted in the README and in the GPU guide are those of 5 October 2026**,
  taken with the library as it is now (hypre's own PCG, the orthogonalized CG of a Newton
  step, the compact scatter map). Two Newton-CG steps of the benchmark take 10.9 s at 64^3
  on one L40S (13.6 s before), 23.4 s at 128^3 on four (29.1 s) and 38.5 s at 256^3 on 16
  RTX PRO 6000 Blackwell cards (46.3 s); there the GPUs are 161 times faster than hIPPYlibx
  and 74 times faster than hIPPyMFEM on 32 CPU ranks. The comparison chart of the README is
  now on Blackwell cards at every size, four CPU ranks against the four MIG slices of two
  cards up to 128^3, and its ratios are those of the measured wall times (24, 65, 95 and
  161 over hIPPYlibx; 10, 29, 50 and 74 over hIPPyMFEM on CPU ranks). The guide's tables of
  the Laplace stages, of the ensemble and of the memory settings follow, and it has a table
  of the Newton-CG solve in double and in single precision on seven sets of GPUs. The 400^3
  row and the CPU rows are as they were. The README opens with an animation of the
  geothermal application in the place of the still picture, on a light or on a dark
  background after the theme of the page (`docs/images/geothermal_turn_light.webp` and
  `geothermal_turn_dark.webp`).
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
  it. The element matrices of a split batch whose chunks are joined (`element_matrices`)
  are waited for until their chunk size has got through with no more free device memory
  than there is now (`kernel._room`); after that they are returned while they are
  computed, as those of an unsplit batch are, so that an assembly prepares its scatter
  while the kernel runs. Waiting every time cost that overlap: 9 ms of a complete assembly
  of 262,144 quadratic hexahedra on an L40S (919 against 910 ms) and of 518,400 quadratic
  quadrilaterals (166 against 157 ms). The element vectors are always waited for: their
  callers check them at once, so that returning early would gain nothing and lose the
  retry.
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
- A nonlinear forward solve ends by its Newton correction as well as by its residual
  (`newton_parameters["step_tolerance"]`, 1e-12 of the state). The residual has a floor of
  round-off that grows with the mesh, 4.8e-15 at 16^3 and 1.2e-14 at 256^3 in the
  geothermal application, and with a tolerance under it a solve ran to `max_iter` and
  raised although its state had all its digits: 30 iterations with up to twelve step
  lengths each, for every trial of a line search of Newton-CG, about twelve minutes at
  256^3 before that search gave up. And a solve whose own line search finds no step that
  lowers the residual, with a correction that is not negligible, raises at once and no
  longer at `max_iter`.
- The geothermal application stops the incremental solves of its Hessian actions at 1e-6
  and lets a Newton step take up to 200 CG iterations (`--inc-tol`, `--cg-max`; they were
  1e-8 and 50). Its data then grew with the mesh, one sample per element layer in every
  borehole, and the CG of a Newton step with them: at 256^3 the cap of 50 ended 24 of 35
  steps before the Eisenstat-Walker tolerance was met, and the gradient came down by 0.7 a
  step. With the new values the same MAP point takes 18 steps and 552 CG iterations
  instead of 35 and 1302, 851 s instead of 2190 s on 32 MI250X (at 128^3 on 8 of them 361 s
  instead of 747 s), and the whole run 36 minutes instead of 59; the cost functional agrees
  to 5e-11 and the 50 eigenvalues to 1.5e-5. The tolerance alone keeps the iterations and
  takes a fifth off each. The same 660 observations on 16^3 and on 24^3 take 250 and 251 CG
  iterations: it is the data that the cap has to follow, not the mesh.
- The data and the synthetic truth of the geothermal application no longer depend on the
  mesh. The boreholes had one temperature sample per element layer, 660 observations at
  16^3 and 10 740 at 256^3 with the same noise, and the truth away from the anomaly was a
  prior sample drawn on the mesh: a finer mesh solved another inverse problem, a more
  informative one on another rock (the largest eigenvalue of the prior-preconditioned
  misfit Hessian was 16.7 times larger with the 256^3 data than with the 16^3 data, on one
  mesh). Every borehole is now logged every 62.5 m, 45 samples and 2 700 observations in
  all, and the truth is a function of the point, 1 024 plane waves with the prior's
  spectrum and none shorter than 0.4 km (`model.background_truth`). On Frontier 64^3 on
  one MI250X, 128^3 on four and 256^3 on 32 take 16, 16 and 17 Newton steps with 363, 358
  and 359 CG iterations (213, 554 and 649 s); their largest eigenvalues are 1.008e6,
  1.009e6 and 1.010e6, about 500 are above one on each (506 of the 600 computed, 491 and
  483 of 566, in 393, 992 and 1 171 s), and the target temperature at the MAP is 71.58,
  71.57 and 71.59 K. The application's README and the guide hold these records, and
  `docs/images/geothermal_posterior.png` is the 128^3 run.
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

- **A chunked assembly of a space with a dof transformation failed.**  Second-order Nedelec
  elements on tetrahedra carry MFEM's `DofTransformation`, one matrix per element; an
  assembly split into chunks applied the whole group's transformations to each chunk and
  stopped with a shape mismatch.  Each chunk now takes its own.
- **`box_mesh` refused coarse meshes its levels allow.**  It limited the cut of a coarse
  element to eight a side before counting the halvings of `levels` (24^3 from 2^3 with two
  halvings), and with `levels` and no `coarse` it kept a default coarse mesh that the
  halvings do not divide (16^3 with three).  A default that was valid is kept.
- **Single-precision GMRES: its verdict, small right-hand sides, and which problems use it.**
  - hypre's GMRES can report convergence with a final relative residual far above its
    tolerance. On a nearly singular operator it stopped after 20 iterations with 3.6 for a
    tolerance of 1e-5, and the solver took the result. `KrylovSolver` now takes such a
    solve for converged only where hypre's own final residual is below the relative
    tolerance (or below the absolute one, times `|b|`). Otherwise the solve fails as a
    single-precision PCG that breaks off does: the forward and adjoint solves fall back on
    double precision (`SINGLE_GIVE_UP`), and the error names GMRES. The final residual is
    kept in `KrylovSolver.final_rel_norm`.
  - A right-hand side below the absolute tolerance gives zero, as in double precision.
    hypre's GMRES took it for zero and returned without a verdict, and the solve raised
    an error about a preconditioner that is not positive definite.
  - With `HIPPYMFEM_HYPRE_SINGLE` named, a Jacobian found symmetric went to single
    precision also where its solvers are GMRES, as they are by default (`spd_jacobian`
    unset). The Hessian actions of such a problem then dropped from about 3e-10 to 3e-5
    against finite differences. As before GMRES could run there, a symmetric Jacobian
    goes to the single-precision hypre only where all three of its solvers are CG;
    GMRES solvers take it only for a Jacobian that is not symmetric. A symmetric verdict
    is kept from one assembly to the next only where the Jacobian goes to single
    precision. Otherwise it is probed again, as without a single-precision hypre.
  - The adjoint's solver, cloned from the forward solver for a Jacobian of that library
    that is not symmetric, was cloned once. It now follows settings made on the forward
    solver after its first solve, as the forward solver does when it solves the adjoint
    itself. A `solver_adj` the user sets is left as it is.

  New checks in `single_case` (run by `test_solvers` and `test_device` with a
  single-precision hypre named) cover each of these.
- **The abort hook ended runs it had no reason to end.**  `sys.excepthook` also receives
  errors that are handled where they are shown, and errors that are only reported. In
  each of the following cases the hook now prints the traceback as before and does not
  call `MPI.COMM_WORLD.Abort`.
  - A console of the standard library (pdb's `interact`, the REPL of Python 3.13) shows
    an error typed into it through the hook and goes on. One typo there ended every rank.
  - `weakref.finalize` reports through the hook an exception of a finalizer that it runs
    at exit. A run that had finished its work then ended with an abort and status 1.
  - In an interactive session (`python -i`, the REPL), ending the run is the user's call.
  - After `MPI.Finalize`, `Abort` cannot reach the other ranks, and the traceback was lost
    in its error.

  `abort.handled_elsewhere` makes the decision. `test_vectors` checks it with a recorder
  in place of the abort.
- **Reading `ParVector.offsets` on one rank hung, and `parvector.allreduce` could reduce
  bits.**
  - A layout made from a matrix's row range (`Layout.of_range`, the vectors of a solver)
    gathered every rank's offsets at their first read. A read made on one rank only, as a
    debugging print makes, waited forever for the others. The offsets are now gathered
    when the layout is made, by one collective on a buffer (0.13 ms on 4096 ranks, once
    for an operator).
  - `allreduce` chose the MPI type from each rank's own value. An integer on one rank,
    such as 0 from an empty sum, beside doubles on the others summed their bits:
    4609434218613702656 on rank 0 for 0 and 1.5. Every value now goes as a double, which
    is exact for integers up to 2^53. An integer comes back where the value was one and
    the result is whole.

  `test_vectors` checks both.
- **A misspelt `HIPPYMFEM_SINGLE_GIVE_UP` was accepted at import.** It was found only when a
  forward solve gave up, as a `ValueError` that a line search does not take for a failed
  trial. It is now read with `config.env_choice`, as the other settings of the environment
  are, and stops `import hippymfem` (checked by `test_vectors` in a fresh interpreter).
- **`FunctionSpace.coordinates()` and `project()` missed nodes moved in place.** The
  coordinates of the dofs are kept between calls, until MFEM counts a change. MFEM does
  not count nodes written in place before `mesh.NodesUpdated()`, so a curved mesh moved
  that way gave the old coordinates. The key of the kept array now holds a fingerprint
  of the nodes: the exclusive or of their 64-bit words and their sum, one pass over them
  (0.02 s for 400 MB). The vertices of a mesh without nodes were already part of the key.
  Before the array was kept, such a move was seen at once, and it is again
  (`test_assembly.test_coordinates_after_moving_nodes`).
- **`KrylovSolver(fes=...)` with a `FunctionSpace`, and a systems ordering left unsaid.**
  - Given one of the library's `FunctionSpace`s as `fes`, the solver raised
    `AttributeError`: it asked the space for `GetOrdering`, which only MFEM's space has. A
    `FunctionSpace` is now read through its MFEM space, for the ordering and for the
    elasticity options.
  - With `systems_dim` and neither `fes` nor `systems_bynodes`, the solver takes the
    components one after another, as the library's spaces are ordered. On an interleaved
    (`byVDIM`) space, BoomerAMG then pairs rows of different components, and a solve takes
    15 to 50 times the iterations. That default stays, and a `RuntimeWarning` now says it
    once. The library's own priors always name the ordering.

  `test_solvers.test_systems_ordering` checks both.
- **The square root of a Laplacian prior was slower with the compact affine geometry.** It
  contracted the inverse Jacobians repeated at every quadrature point (`full_Jinv`, a view
  of stride zero), which numpy's optimized `einsum` copies out for its products. On
  13,824 linear hexahedra, `S^T` took 12.9 ms where it takes 7.8 ms without the compact
  geometry. On an affine group the compact array (one inverse Jacobian an element) is now
  contracted as it is: 8.0 ms for `S^T`, and 4.3 ms for `S` where it took 5.8, with the
  same results to the last digit (`prior._jinv_operand`).
- **A finished run on CUDA could end with status 134.** MFEM's Device is never freed from
  Python, so its memory manager lives until the static destructors of the process. Those
  run after the CUDA runtime's own teardown, which was registered when CUDA started.
  Device memory still held then was freed into a runtime that was shutting down: hypre
  printed "driver shutting down", and MFEM aborted (`test_vectorfe` on Blackwell cards;
  HIP tolerates it, and `main` did the same). On a CUDA build, `configure_device` now
  registers a Python `atexit` handler. During the interpreter's finalization it registers
  MFEM's `Hypre::Finalize` and `MemoryManager::Destroy` as exit handlers of the process
  (`__cxa_atexit`). Registered last, they run first: after every Python object is gone and
  before CUDA's teardown. MFEM's destructors then find the manager gone and do nothing
  (`mfemconfig._late_teardown`). This happens once a process, only on a build of MFEM with
  CUDA, whatever backend a test fakes, and not where the manager is already gone. MFEM
  stops the process on a second `Destroy`, as a first version of this change showed in
  `test_solvers`.
- **Columns named alone were eliminated on one route only.** `assemble_matrix(V, V, ...,
  trial_ess=ess)` with the same space on both sides and `symmetric` at its default
  eliminates those columns on the folded route. MFEM's route
  (`HIPPYMFEM_FOLD_ELIMINATION=0`, or a diagonal policy of `"keep"`) took the symmetric
  branch, which eliminates by `test_ess`, and returned the matrix untouched
  (`elimination._eliminate`; `test_assembly.test_trial_ess_only_both_routes`).
- **A parameter in the state's own space lost its boundary values in the assembled Hessian
  blocks.**  The assembler eliminated a block's essential dofs on both sides whenever its
  two sides were the same space object, as the Jacobian needs.  With `Vh = [V, V, V]` and
  essential conditions on the state, that made `C` and `W_um` drop the parameter's columns
  at the state's essential dofs (and put ones on the diagonal of their essential rows), so
  the Hessian action ignored the parameter's values there: 10 to 17 % off the matrix-free
  products in the new check, 38 % off finite differences for a nonlinear initial density.
  The gradient (assembled as a vector) and the matrix-free products (`apply_ij_at`) were
  right.  The assembler now eliminates on both sides only where the caller asks
  (`plan_block`, `assemble_matrix` and `assemble_matrix_csr` take `symmetric=`; the default
  keeps the old rule).  `PDEVariationalProblem` and the step blocks of
  `TimeDependentPDEVariationalProblem` pass `symmetric=False` for every block with a
  parameter side, which keeps the parameter's columns and zeroes the essential rows.  With
  MFEM's elimination (`HIPPYMFEM_FOLD_ELIMINATION=0`) the directly assembled transposes of
  such blocks also kept columns they should have lost; they lose them now.  Of the
  tutorials, `05_HessianSpectrum` was affected (a source term in the state's space and
  `u = 0` on one edge): its gradient and MAP point were right, but its stored eigenvalues
  and CG counts came from the old Hessian.  The applications put the parameter in a space of
  its own, or (`ad_diff`) zero its boundary values in the initial-condition wrapper, and do
  not change.  `test_modeling` compares the assembled blocks with the matrix-free products
  for a parameter in the state's space, with the elimination folded and with MFEM's.

- **On AMD cards a program that queued more kernel launches than the runtime's queue holds
  ended in a segmentation fault** (`_jaxconfig`: `ROCPROFILER_REGISTER_ENABLED=0` unless
  the environment says otherwise).  JAX's ROCm plugin is linked with AMD's profiler
  library, and the runtime, finding it in the process, hands it every launch on its way
  to the card whether or not anything is profiled.  That layer does not survive a full
  launch queue (16 384 packets): the process falls inside `libhsa-runtime64`, or the
  queue aborts with "The AQL packet is malformed".  JAX hands launches over without
  waiting, so a loop of compiled calls whose results nothing reads gets there: forty
  time steps of the seismic propagator on one rank did every time (3 of 3 runs on an
  MI210; with the setting 3 of 3 pass, at the same speed), and this was the fault that
  ended `test_gpu` on one rank now and then (5 of 82 runs before; none of 52 with the
  setting, none of 52 with a queue of 2^20 packets, which only moves the limit).  Without
  the profiler's layer the runtime waits for room in its queue.  The runtime reads the
  variable at its first use: a program whose MPI starts it in `MPI_Init` (Cray MPICH with
  its GPU support) and that imports mpi4py before hippymfem has to have the variable in
  its environment, and `_jaxconfig.PROFILER_LATE` says so then.
- **The seismic propagator lost sources and receivers on fine meshes, and the copies of
  an unknown shared by three or more ranks drifted apart** (`sem.locate`, `_merge_fn`).
  The Newton iteration that finds a point's element stopped only below 1e-13 in
  reference units, which round-off does not reach in elements of a few kilometres: with
  the paper's mesh refined four and five times (512 and 4096 ranks) 2 and about 20 of
  110 points were left without an element and the run stopped after its whole set-up.
  It now also ends when the increment, already below 1e-8, has stopped falling.  And XLA
  rewrote the sum of the sharers' values so that a rank third or later among them added
  its own first: such copies differed in their last bits and, the field never being
  exchanged, drifted (5e-10 of the field in double precision after 16 000 steps on
  made-up ranks, 16 % in single).  An optimization barrier holds the order.  And a point
  near a cut between ranks went to the lowest rank that found it within the search's
  tolerance of 2 % of an element, also where another rank held the element it lies in:
  on three made-up ranks two of 105 receivers were read 1.1 % of an element from their
  place.  A point now goes to the rank whose element it is least outside of, which is the
  element one rank alone gives it.
- **`icesheet/scale.py --steps` with `--solver columns` handed the solver a matrix**, and
  `misfit_hessian_sqrt_block` imported CuPy where it needed none; `run.py` on one rank
  asks for the factorization of the collapsed problem on the card only where cuDSS is
  installed, and `--solver dd` or `cudss` without it ends in one line.
- **A matrix assembled chunk by chunk on a mesh of several kinds of element was wrong**
  (`fem.pattern.ScatterPattern._fused_maps`, `_array_order`).  Where the kinds of element
  of a mesh interleave (triangles among quadrilaterals) a pattern keeps its slots in the
  order of the mesh's elements, and the scatter that takes each chunk of element matrices
  as it is produced read them in the order of the element arrays, one kind after the
  other: every entry went into another entry's slot.  The matrix had the right sparsity
  and finite values and was off by its own size.  That scatter is the route of a batch that
  a device or the host splits, of the streamed pass of a linearization point, and of every
  Jacobian with the element products of the p-multigrid.  A linear forward solve said so
  (its one step leaves the residual where it was, and the single-precision solve gives up
  and is repeated in double precision); the blocks of a linearization point and the
  incremental solves said nothing.  One kind of element, which is every run on boxes,
  takes the same arrays as before.  `test_assembly.test_fused_scatter` and
  `test_compact_scatter_map` have the mixed mesh.
- **On a mesh of several kinds of element a density that reads the coordinates got
  none** (`fem.elementbatch.ElementGroup._geometry_elementwise`; older than the scaling
  work).  PyMFEM's `ElementTransformation.Transform` hands the point back for an
  integration point and does not write the vector it is given; the group kept that vector,
  which nothing had written, as the coordinates of its quadrature points.  The integral of
  `x` over the unit square came out as 1e-315.  Weights and Jacobians were right, so a
  density without `x` was too.  `test_kernels.test_mixed_geometry` integrates `x`, `y`,
  `x y` and `x^2` now.
- **The rule at the vertices had the geometry of two vertices exchanged on quadrilaterals
  and hexahedra** (`fem.elementbatch._tensor_places`; `BiLaplacianPrior(lumped_mass=True)`).
  MFEM's batched geometric factors return the points of an element with a tensor basis
  in the order of a tensor grid, whatever the order of the rule, and the rule at the
  vertices follows the element's vertices, (1, 1) before (0, 1): its third and fourth
  points (and its seventh and eighth) had each other's coordinates, Jacobian and weight.
  The square root of a lumped mass matrix then weighed those vertices by each other's
  |det J|, so that on elements that are no parallelograms or parallelepipeds
  `sqrtM sqrtM^T` was not `M` (4 to 18 % of a diagonal entry on the meshes tried) and a
  sample of the prior did not have the covariance of its precision; with the geometry
  built in slices (`HIPPYMFEM_GEOMETRY_SLICE`) it was right.  Each point of such a rule now
  takes the geometry from its place in MFEM's grid.  On uniform boxes |det J| is one number
  an element and nothing but round-off moves (the benchmark's truth by 1e-15 of its size);
  rules in MFEM's own order, which are all the others, are untouched to the last bit.
  `test_kernels.test_vertex_rule_geometry`, and distorted quadrilaterals and hexahedra in
  `test_modeling.test_lumped_mass_prior`.
- **A mesh of two kinds of element whose ranks each hold one kind passed for a mesh of
  one** (`fem.elementbatch.MeshBatches`, the lumped mass matrix of `BiLaplacianPrior`).
  Each rank counted the kinds of its own elements.  MFEM's batched geometry does not take
  such a mesh (it stops in a null pointer), and the lumped mass matrix was assembled with
  the rule of the other kind.  The kinds of all ranks are put together now.  MFEM's own
  partition of the suite's mixed mesh leaves a rank with both kinds on 2 to 16 ranks, which
  is why it had not shown.
- **A block whose ranks split their element batches differently hung**
  (`PDEVariationalProblem._domain_block`; older than the scaling work).  Whether a batch is
  split is each rank's own answer: its element count, its memory, a chunk it remembers
  from an out-of-memory retry.  A rank with a whole batch reduced its overflow flag before
  the assembly and a rank with a split one after it, so the two kinds waited in different
  collectives: on eight ranks the forward solve did not return, on two one rank raised
  "produced non-finite element values" for finite values (a flag paired with another
  rank's reduction of something else).  One flag and one collective after the assembly
  now, on both routes.  The scaling runs were not exposed (equal boxes, every rank splits,
  and their Jacobian is streamed on every rank).  `test_assembly.test_split_and_whole_ranks`
  puts the ranks on different sides, in host memory and on a device.
- **`require_finite` let a vector with entries that are not numbers pass where MFEM holds
  it on a GPU.**  It asked for the Euclidean norm, and MFEM 4.9's `Vector::Norml2` passes
  over an entry that is not a number (`if (n > 0)`), on the host as on a device; an
  infinite entry it finds.  The sum of the magnitudes is asked now.
  `test_modeling.test_require_finite`.
- **An error that hypre left up is no longer taken for a refusal of the next option, or
  for the failure of the next set-up** (`mfemconfig.set_boomeramg_options`,
  `singlesolve._boomeramg`, `linSolvers._pcg_verdict`).  hypre has one error flag a
  process, every setter of an option returns it, and MFEM's `HyprePCG` neither reads nor
  clears it.  After a `cg(amg)` solve that ended at its limit the next solver with
  `amg_options` (the default is `HIPPYMFEM_AMG`) raised `ValueError: hypre refused the
  BoomerAMG option relax_wt=1.8 (... returned 256)`, which no line search catches: a
  Newton run died where it should have backtracked (seen in one probe of the scaling
  work).  After a set-up that met a zero row the flag was up on the rank that holds the
  row alone, and that rank alone raised (12) while the others went on.  The setters'
  verdict is now read from a flag that was cleared first, and a solve in hypre's PCG
  reads and clears the flag when it ends.
- **A double-precision solve that hypre's PCG abandons is no longer taken for converged**
  (`KrylovSolver.solve`), the counterpart of the single-precision entry below.  The verdict
  was "fewer iterations than allowed, or the tolerance met".  hypre's PCG ends early in
  four ways that are not convergence: a preconditioner that is not positive definite
  (`<C r, r>` turns negative, its test passes, the residual is the root of a negative
  number: BoomerAMG with `relax=7` and a weight above what the matrix allows), a
  breakdown, a matrix that is not positive definite (it takes the right-hand side for
  zero and returns it as the solution), and a right-hand side that is not a number (it
  returns zero).  Each came back as converged, with the count and the residual of the
  solve before in the last two.  The verdict is now hypre's own (`HYPRE_PCGGetConverged`,
  read through the solver's handle as `boomeramg_handle` reads BoomerAMG's) together with
  a residual that is a number; a zero right-hand side gives zero in no iterations.  A
  solve that meets its tolerance at the last iteration allowed is converged as before,
  and one that makes the iterations it is allowed with both tolerances zero keeps their
  count (hypre raises nothing at its limit where it has no test to miss).
  `test_solvers.test_hypre_pcg_verdict`.
- **The coarsest level's relaxation and sweeps hold wherever they are written in a list
  of options** (`coarse_relax`, `coarse_sweeps`; both precisions).  hypre's setters of the
  relaxation and of its sweeps reset those of the coarsest level, so
  `coarse_relax=7,relax=18` gave Gaussian elimination on the coarsest level, and with the
  two names in `HIPPYMFEM_AMG` and `HIPPYMFEM_SINGLE_AMG` the order could not be chosen.
- **`seq_threshold` no longer gathers a level that cannot be coarsened, and gathers the
  level of a system with the functions of its rows** (`tools/hypre-3.2.0-seqdevice.patch`,
  its third version; `hypre_BoomerAMGSeqOnDevice()` returns 3).  hypre's set-up also ends
  at a level on which coarsening finds no coarse point (every row diagonally dominant: a
  mass matrix; a prior at the level whose mesh width passes the correlation length), and
  it gathered that level whatever its size, in host memory too: the whole of it on every
  rank that holds rows, with 32-bit counts.  Such a level is relaxed now, as without the
  setting.  And the gathered solver of a system was told the number of functions and not
  which row is which, so it paired rows of different functions below the first level (34
  CG iterations where the hierarchy takes 12, on vector diffusion); it gets the level's
  own labels.  A hierarchy of one equation that coarsens to its threshold, which is every
  scaling run, is the same.  The build scripts try each patch whole and without fuzz
  before applying it, say which switch a kept library ignores, and
  `build_hypre_single.sh` no longer ends without a word when a build fails or its first
  argument is a relative path.  `seq_threshold` in host memory under MFEM's "cpu" device
  is no longer refused for the lack of the patch.  On a device the setting now asks for
  the third version (`mfemconfig.seq_on_device`): a library that carries the second, which
  ran every series of the scaling study, is refused with the reason, so that neither
  defect can come back with an older build.
- **BoomerAMG is told how the components of a vector space are ordered**
  (`KrylovSolver(systems_dim=..., systems_bynodes=...)`; older than the scaling work).
  With the number of components alone hypre takes the rows for components in turn, which
  is the interleaved ordering; this library's spaces are ordered component after component
  unless another ordering is asked for, so every systems hierarchy of a default vector
  space paired rows of different components: 58 CG iterations where 12 do on vector
  diffusion with the coarsening and relaxation of a device, 21 where 6 do with the
  host's, 15 where 5 do for `VectorBiLaplacianPrior`.  The solver now passes the
  ordering (by nodes unless told otherwise, read from `fes` where one is given, and from
  the space by the priors).  `test_solvers.test_systems_ordering`.
- **In the single-precision library, what a matrix makes of the defect of its row sums is
  one for all ranks** (`SingleParMatrix._set_row_defect`), and a vector without entries
  asks MFEM for no pointer (`SingleVector.set`, `get`).  The kind of values a rank hands
  over is its own (a host array on a rank without elements, single precision from a batch
  that was not split), and the product that finds a card's defect, its reduced check and
  the kind of PCG followed it; a rank without rows on a device took `int()` of a pointer
  that PyMFEM returns as `None`.  Latent both: another failure came first.
- **A state of several components with `pmg_h` keeps BoomerAMG under the linear elements,
  with a warning** (`PDEVariationalProblem._p_transfer`), where the first Jacobian raised
  `NotImplementedError` (taken by a line search for a failed trial) and the next call went
  on with a half-built transfer and no levels, without a word.
- **`HIPPYMFEM_HESSIAN=auto` no longer keeps the route that was traced faster**
  (`fem.kernel`): a chunk loop that streams geometry traces a kernel before its first call,
  and the timing of the two routes ran on tracers.
- **A rank that holds no element of a block's groups** assembles the rows it is sent
  (`ScatterPattern.data` raised on an empty list of arrays where its target had slots for
  rows of other ranks: the suite's own boundary test at 6, 7 and 16 ranks), and two sets
  of groups that are both empty on a rank no longer share one pattern there
  (`get_pattern`: the other ranks built theirs collectively while that rank went on).
- **Settings assigned through `hm.config` as `True` or as a float** (`pmg`, `pmg_h`,
  `pmg_h_host`, `prior_h`, `pc_reuse`, `amg_relax`, `amg_max_levels`) were written as
  text that the readers could not parse, and the solvers took the default without a word;
  they are written as whole numbers, and a value that is none is refused.
- **hypre's pool**: `trim` took the bins out before it called the driver (a block freed by
  a finalizer during the call was listed twice or raised), `uninstall` gives the
  single-precision library its allocator back as well, and a refused GPU-aware hypre is
  left staging its buffers when the check raises.  The single-precision library stages
  its exchanges whatever it was built for.  In the experimental mode
  `HIPPYMFEM_GPU_AWARE_MPI=solve` the first solve by one of MFEM's solvers on a hierarchy
  that hypre's PCG set up is a set-up too (MFEM builds it again) and is staged, and so is
  the mirror, the first solve by hypre's PCG on a hierarchy that MFEM's CG has used (a
  solve from zero after one from a given vector, or with a PCG lent before it solved).
- **The benchmarks.**  `bench_scaling.py --profile-setup` reported as its V-cycle a tenth
  of a second set-up plus a cycle (MFEM sets a preconditioner up again at its own first
  application); `--profile-vcycle` with `--both-kernels` stopped before the record
  (`H` rebound); both profiles were attempted on a Jacobian of the single-precision hypre.
  `bench_newton_device.py --rank-table` read the card's memory after its two probe kernels
  had taken 1.5 GiB (four smoke records are off by that; no series record), and its stages
  now count the plans, the scatter and the matrices of a Jacobian with element products
  under names of their own (`+jacobian plan_block` and so on: they were in no stage but
  the inclusive `+jacobian`).  What an engine put in host memory is recorded by the kind
  of cycle (`singlesolve.HOST_LEVELS`), so a prior's cycle no longer answers for the
  Jacobian's.
- **Checks that a NaN could not fail.**  Python's `max` keeps its first argument against a
  NaN, mpi4py's reduction does the same between ranks, and MFEM's `Norml2` passes over
  one: a check `worst < tol` printed `[ok]` for a NaN in any case but the first or on one
  rank of several.  The maxima of the test suites go through a helper that carries a NaN
  (`largest`), the reductions through `parvector.allreduce_extreme`, the norms through the
  library's own, in every test file.  Conjuncts that could not fail are gone, and the
  check of `cg_preconditioner_cycles` fails when the preconditioner is never applied.
- Smaller ones: the warning about a growing arena no longer fires on a device that
  reports no cap (the host: "448800 % of JAX's arena"); `FunctionSpace.order` answers on a
  rank without elements (MFEM stops the process at the question for element 0); the
  true-dof pattern built without the compiled passes sorts by two keys where a row times
  the columns passes 64 bits (17 million rows a rank beside 5.5e11 columns).
- **A setting that cannot be read is refused by its name, not replaced by the default.**
  `HIPPYMFEM_PMG=on` or `HIPPYMFEM_PMG_H=two` in a job script ran the default configuration
  without a word; the seven whole-number variables of the solvers and `HIPPYMFEM_PRIOR_H`
  now raise `ValueError` with the variable's name at the first solver
  (`linSolvers._env_default`), as assignments through `hm.config` already did.  A mistyped
  `HIPPYMFEM_HYPRE_SPMV` is refused before the device is made (it was refused after, and a
  caller that caught the error went on with a device that had had no check of the MPI
  library and no pool), and so is a `HIPPYMFEM_HYPRE_POOL` that is neither `auto` nor
  megabytes (`on` was taken for 0, no pool at all) or a `HIPPYMFEM_HYPRE_POOL_KEEP` that
  is no number.  A solver parameter that hypre refuses (`amg_strength_threshold`
  above one, which it dropped for its own 0.25) raises at `set_operator` with the
  parameter's name (`KrylovSolver._make_pc`: MFEM's setters do not pass hypre's verdict on).
  `test_solvers.test_refused_settings`.
- **A systems hierarchy keeps the solver's own settings** (`KrylovSolver._make_pc`).  With
  the number of components MFEM writes two settings of its own, no aggressive levels and a
  strength threshold of 0.5, and they were written after the solver's: `amg_agg_levels`,
  `amg_strength_threshold` and a `theta` or `agg` of `amg_options` were replaced for every
  vector problem without a word.  The components are set first now.
- **The trust-region method takes `cg_preconditioner_cycles`** (`NewtonCG`), which the
  line search honoured and it ignored without saying so.
- **An error inside a prior is no longer taken for an argument it does not know**
  (`model._takes`; `Model.Rsolver`, `NewtonCG._cg_preconditioner`).  The call that builds
  the prior's preconditioner sat in `except TypeError`, for priors and models written
  before the arguments `tolerance` and `cycles` existed, and every `TypeError` raised
  inside went the same way: the caller fell back on the plain solver, on the rank that
  raised alone.  The signature is asked before the call.
- **Row sums in single precision of a batch that is not split** (`ScatterPattern.data`,
  `csrassemble.assemble_matrix_csr`).  With `HIPPYMFEM_SINGLE_ROW_SUMS=1` and
  single-precision element matrices, a device summed a batch that fits whole in the
  matrices' precision: the sums had their rounding in them and the matrix kept no defect,
  without a word, while a split batch kept it.  Such a batch is summed in double precision
  now.  The runs with the element products of the p-multigrid do not come there.
- **GPU-aware MPI with hypre 2.31 and 2.32** (`mfemconfig._gpu_aware_fixed`; the CUDA
  stack, whose PyMFEM brings hypre 2.32).  Those versions have the run-time switch and
  the field it writes, and an exchange that is compiled one way and does not read the
  field (MFEM says so where it prepares one).  A library compiled to hand MPI its buffers
  under an MPI that cannot read device memory was refused with the advice
  `HIPPYMFEM_GPU_AWARE_MPI=0`, and with that the check passed, by the field, while hypre
  went on handing over device addresses.  The field of such a library is no longer
  written, so that it goes on saying what the library was built to do; a setting it cannot
  follow is warned of, and the refusal advises the other library.  Read from MFEM's source
  and the tree's patch for 2.32; not run on that machine yet.
- **Smaller ones of the second round.**  `PDEVariationalProblem._same` joins the other
  ranks' reduction where its own sizes differ (it returned ahead of it);
  `ptransfer`'s largest row-sum error carries a rank's NaN, and a weight of MFEM's
  transfer matrix that is not a number is kept, so that the library refuses the matrix,
  where it was left out with the zeros; `element_matrices_or_chunks` asks whether a batch
  is split at the weight of the pass that follows (asked at weight 1, a batch between the
  two was handed over whole and then split and joined); `fem.kernel.RETRIES` counts the
  element passes that ran out of memory and were tried again in smaller chunks, and the
  Newton benchmark records it over all ranks (`element_retries`) beside how far apart the
  ranks' solvers of a gathered last matrix were (`pmg_h_gathered`); that comparison says
  when a rank's solution is not a number; `vectorfe.vector_kind` asks MFEM for the typical
  element, which a rank without elements has too; `modelVerify.best_slope` is NaN where an
  error inside its window is not a finite number (it left the step out, and a cost that
  was NaN at one step passed every slope check); `tools/build_pymfem_hip.sh` says which
  step failed and where its log is, where it ended with status 1 and no word;
  `bench_scaling.py` names what it measures with a Jacobian in the single-precision hypre
  (the product with its two conversions; the kernel that those solves keep under
  `--both-kernels`); the checks of `seq_threshold` on one rank, where hypre drops the
  setting, say that nothing is gathered instead of comparing the default hierarchy with
  itself under another name.
- **`set_hessian_mode` also drops the consistent blocks of the mixed precision**
  (`fem.kernel`): a Jacobian compiled by one route kept that route when the mode was
  changed afterwards, since the block that is exact on the constants is cached under a key
  of its own.
- **The column indices that single-precision matrices share are freed once, whatever the
  order of the finalizers** (`singlesolve._SharedColumns`).  Every matrix of a pattern in
  the single-precision hypre borrows the pattern's one copy of its column indices and
  takes the pointers out of its blocks before hypre destroys it.  At the end of a process,
  or of a reference cycle, Python finalizes a group of objects in no particular order:
  where the shared copy went first, it freed the arrays, and hypre freed them again with
  a matrix that still had their pointers.  `test_gpu` with the library named ended that
  way on two and on eight ranks of a HIP build, after its last check ("double free or
  corruption", "corrupted size vs. prev_size while consolidating").  A matrix now says
  when it takes the arrays and when it has let go of them, and they are freed when
  nothing refers to the shared copy *and* no matrix holds them.
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
- The cache of AMD's code object manager was under the home directory (the manager's
  default, `~/.cache/comgr`). The HIP runtime keeps every kernel it compiles there, XLA's
  and those of a HIP build of MFEM and hypre, and on a file system shared by the nodes the
  ranks of a run replace one another's files while others have them open: of 32 ranks on
  four nodes of Frontier (NFS projected through DVS) ten waited for ever and the others
  for them. On an AMD card the cache is now in a directory of the node and of the user,
  `hippymfem-comgr-<uid>` under the system's temporary directory
  (`_jaxconfig.COMGR_CACHE`), unless `AMD_COMGR_CACHE_DIR` names another.
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
- Two checks whose outcome depended on the rank count. `test_vectorfe` asserted that
  Newton-CG reports convergence on its H(curl) problem, where the predicted decrease
  after the first step is about 1e-18: under `gdm_tolerance` on most partitions, and
  2.2e-18 on four ranks, where the line search was then exhausted at the same point. The
  reduction of the gradient is asserted now, as for the reference MAP point of
  `test_uq`. `test_gpu` compared an assembly on the device and on the host on a fixed
  mesh, of which four ranks hold a quarter each: against the device's fixed cost per
  assembly the host won there (0.8x for first-order hexahedra on four MI250X). The
  meshes grow with the rank count now, so that a rank holds the elements of the one-rank
  run (4.8x).
- The geothermal application at 256^3. Its forward Newton solve kept the library's
  absolute tolerance of 1e-14, which the residual of a mesh that size does not reach in
  double precision: it stops at 1.2e-14 (at 4.8e-15 at 16^3). A solve that a line search
  warm-starts has a relative tolerance smaller still, so it ran its 30 iterations and
  failed, every trial of the line search with it, and Newton-CG stopped at its sixth step
  with "Forward solve failed during backtracking". The application asks for 1e-13 now.

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

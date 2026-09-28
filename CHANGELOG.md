# Changelog

## Unreleased

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

# Changelog

## Unreleased

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

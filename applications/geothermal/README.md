# Basin-scale geothermal heat-flow inversion

Steady heat conduction in an 8 km x 8 km x 4 km block of layered rock, with a
conductivity that depends on temperature through a *tabulated* law per lithology, is
inverted for the rock's reference conductivity from borehole temperature logs.  It is
chosen to exercise what is specific to hIPPyMFEM: a residual density that no form language
expresses (a table read with a Gaussian kernel, a lithology map with dipping interfaces,
an anisotropic tensor), differentiated exactly by JAX at the quadrature points, and the
whole Bayesian workflow (MAP, Laplace approximation, posterior samples, a quantity of
interest) on GPUs.

![The true log conductivity, the MAP estimate from 60 boreholes and the posterior standard deviation at 128^3](../../docs/images/geothermal_posterior.png)

## The model (`model.py`)

- **PDE.** `-div(k(m, T, x) D grad T) = q(x)` on the unit cube (the block in scaled
  coordinates, `D = diag(1, 1, (L/H)^2)`), `T` in units of 100 K above the surface.
  Dirichlet `T = 0` on the top face, a basal heat flux of 66 mW/m^2 on the bottom face
  (a boundary density), no flux on the sides, radiogenic heat production per lithology.
- **Conductivity.** `k = exp(m) f_lith(x)(T)`, where `f` is the Vosteen & Schellschmidt
  (2003) law `k0 / (0.99 + T (a - b/k0))` for sediments, carbonates and basement,
  *tabulated* at 16 temperatures and read with a Gaussian kernel (smooth in `T`, so the
  Hessian is exact and the Newton solves converge quadratically).  The unknown `m` is
  the log of the reference conductivity relative to 2.5 W/(m K).
- **Data.** 60 vertical boreholes at random positions, a temperature sample every 62.5 m
  from the surface to 2.8 km depth (45 per borehole, 2 700 in all, on every mesh), noise
  0.5 K.
- **Prior.** BiLaplacian, `gamma = 0.3`, `delta = 4.8`, `Theta = diag(1, 1, 0.25)`,
  Robin boundary: marginal std 0.50 in log k, correlation length 2 km horizontally and
  0.5 km vertically.
- **Truth.** A random field with the prior's spectrum and standard deviation, without
  its shortest waves (1 024 plane waves, none shorter than 1.6 km horizontally and 0.4 km
  vertically; `model.background_truth`), plus a buried high-conductivity body (log k + 1,
  radius 1 km, centred at 2.3 km depth).  Both are functions of the point, so every mesh
  holds the same rock, and with the same logs a finer mesh solves the same inverse
  problem more accurately.  With a Dirichlet top and a flux bottom the thermal signature
  of a conductive body lives at and below it, so the logs reach its depth.
- **Solvers.** The Jacobian `dR/dT` carries `k'(T) dT grad T . grad p` and is not
  symmetric, so the forward and incremental solves use GMRES with BoomerAMG, and the
  adjoint applies the true transpose with the forward operator's AMG hierarchy
  (`symmetric_jacobian=False, transpose_free_adjoint=True`).

## The workflow (`run.py`)

From the repository's root:

    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 PYTHONPATH=<cuda pymfem> \
      mpirun -n 4 tools/mpirun_pinned.sh python -m applications.geothermal.run --n 64 --k 200 \
        --out results/geothermal_n64.json --dump results/geothermal_n64.npz \
        --paraview results/paraview/geothermal_n64
    python -m applications.geothermal.figures results/geothermal_n64.npz \
        --json results/geothermal_n64.json --out results/figures/geothermal_n64

1. synthetic truth and data; the prior's pointwise std by Monte Carlo;
2. the MAP by inexact Newton-CG (five Gauss-Newton iterations, then full Newton,
   backtracking line search);
3. the Laplace approximation: `doublePassG` with `k` eigenpairs and `p` oversampling;
4. posterior samples; the pointwise variance by Monte Carlo (unbiased, unlike the
   randomized estimator); traces; the KL divergence from the prior; the fraction of dofs
   at which the truth lies within two posterior standard deviations of the MAP;
5. the quantity of interest, the mean temperature in the target volume around the
   anomaly, at the MAP with its linearized posterior standard deviation
   (`sqrt(g^T Gamma_post g)`) and over posterior samples pushed through the nonlinear
   forward solve.

Rank 0 writes a JSON record of every timing and number, and with `--dump` the truth,
the MAP, the prior and posterior std on the P1 grid, the eigenvalues, the borehole
coordinates, the data and the sampled QoI values.  `figures.py` turns a dump into the
slice, spectrum, QoI and profile figures.  Without a GPU, drop the two variables and
the launcher wrapper: `--n 16` takes seven minutes on one host core.

## The checks (`validate.py`)

    python -m applications.geothermal.validate fd       --n 16    # FD slopes 1 +- 0.05, Hessian symmetric to 1e-10
    python -m applications.geothermal.validate forward  --n 32    # <= 6 Newton iterations to 1e-9
    python -m applications.geothermal.validate variance --n 16    # MC and sample variance vs the exact posterior variance
    python -m applications.geothermal.validate partition a.npz b.npz c.npz   # dumps at 1, 2, 4 ranks agree

Each check prints its verdict.

## What to expect

On the 64 GB AMD MI250X cards of Frontier, with `--gauss-newton --k 566` (`--k 600` at 64³):

| mesh | state unknowns | GPUs | MAP (Newton, CG iterations) | Laplace (eigenpairs above one) | 64 posterior samples | QoI: truth, at the MAP, linearized std |
|------|---------------:|-----:|-----------------------------|--------------------------------|----------------------|----------------------------------------|
| 64³  | 2 146 689      | 1    | 3.6 min (16, 363)           | 6.6 min (506)                  | 19 s                 | 71.60, 71.58, 0.08 K                   |
| 128³ | 16 974 593     | 4    | 9.2 min (16, 358)           | 16.5 min (491)                 | 67 s                 | 71.60, 71.57, 0.09 K                   |
| 256³ | 135 005 697    | 32   | 10.8 min (17, 359)          | 19.5 min (483)                 | 77 s                 | 71.62, 71.59, 0.10 K                   |

The data (2 700 observations) and the truth are the same on every mesh, and so are the
Newton and CG iterations, the eigenvalues (the largest 1.008e6, 1.009e6 and 1.010e6, the
hundredth 61.9, 62.3 and 61.8) and the target temperature.  The times are from October
2026; 128³ on four cards and 256³ on 32 carry the same 4.2 million state unknowns a card.
The Laplace column is the eigensolver for all the `--k` pairs with 20 more for
oversampling, of which the pairs above one are counted.

- **How many eigenpairs.**  About 500 eigenvalues of the prior-preconditioned misfit
  Hessian are above one, on every mesh: these are the directions the data inform more
  than the prior.  The default `--k 50` keeps the 50 most informed of them (the 50th
  eigenvalue is 2.7e2); `--k 566` passes one, and is what to use when the posterior
  variance itself is the result.
- **Samples against the MAP.**  The QoI over posterior samples lies above its value at
  the MAP, by 0.4 to 0.7 K, several linearized standard deviations (with a fixed basal
  flux the temperature at depth is convex in the log conductivity).  `qoi.taylor_qoi`
  estimates that shift from the second-order term, and the record holds it as
  `qoi_taylor2_hutchinson_mean_K` (72.08, 72.22 and 72.29 K against sample means of
  71.99, 72.14 and 72.27 K).
- **Solver settings.**  The incremental solves of a Hessian action stop at 1e-6 and a
  Newton step may take 200 CG iterations (`--inc-tol`, `--cg-max`): its CG keeps its
  residuals orthogonal, so the looser solves leave the iterations as they are, and the
  Eisenstat-Walker tolerance, which asks for up to 68 iterations in the last steps,
  decides every step.
- **Memory.**  `--gauss-newton` drops the second-order blocks of the Hessian; with it
  the MAP and the Laplace approximation at 128³ (17 million state unknowns) fit four
  48 GB cards (`--gauss-newton --skip-qoi`).

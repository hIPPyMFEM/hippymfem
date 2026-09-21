# Basin-scale geothermal heat-flow inversion

Steady heat conduction in an 8 km x 8 km x 4 km block of layered rock, with a
conductivity that depends on temperature through a *tabulated* law per lithology, is
inverted for the rock's reference conductivity from borehole temperature logs.  It is
chosen to exercise what is specific to hIPPyMFEM: a residual density that no form language
expresses (a table read with a Gaussian kernel, a lithology map with dipping interfaces,
an anisotropic tensor), differentiated exactly by JAX at the quadrature points, and the
whole Bayesian workflow (MAP, Laplace approximation, posterior samples, a quantity of
interest) on GPUs.

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
- **Data.** 60 vertical boreholes at random positions, one temperature sample per
  element layer from the surface to 2.8 km depth (45 per borehole at 64^3), noise 0.5 K.
- **Prior.** BiLaplacian, `gamma = 0.3`, `delta = 4.8`, `Theta = diag(1, 1, 0.25)`,
  Robin boundary: marginal std 0.50 in log k, correlation length 2 km horizontally and
  0.5 km vertically (measured by Monte Carlo on the 16^3 mesh; the first calibration,
  gamma 0.004 / delta 1, gave a std of 16 and conductivity contrasts of exp(60), and a
  forward solve that could not converge).
- **Truth.** A prior sample plus a buried high-conductivity body (log k + 1, radius
  1 km, centred at 2.3 km depth).  With a Dirichlet top and a flux bottom the thermal
  signature of a conductive body lives at and below it, so the logs have to reach its
  depth: the first 32^3 run, with the body at 2.6 km under 2 km logs, recovered
  nothing of it (MAP -0.06 against a truth of 0.93 in the box).
- **Solvers.** The Jacobian `dR/dT` carries `k'(T) dT grad T . grad p` and is not
  symmetric, so the forward and incremental solves use GMRES with BoomerAMG and the
  adjoint uses the true transpose (`symmetric_jacobian=False`; the memory saving of the
  shared operator is not available here).  Newton on the forward problem: 4 iterations
  to 1e-14 at 16^3 (the residual goes 1.9e-2, 2.0e-4, 1.2e-8, 4.8e-15).

## The workflow (`run.py`)

    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 PYTHONPATH=<cuda pymfem> \
      mpirun -n 4 tools/mpirun_pinned.sh python -m applications.geothermal.run --n 64 --k 100 \
        --out results/geothermal_n64_r4.json --dump results/geothermal_n64_r4.npz \
        --paraview results/paraview/geothermal_n64

1. synthetic truth and data; the prior's pointwise std by Monte Carlo (so the numbers
   below can be read);
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
slice, spectrum, QoI and profile figures.

## The checks (`validate.py`)

    python -m applications.geothermal.validate fd       --n 16    # FD slopes 1 +- 0.05, Hessian symmetric to 1e-10
    python -m applications.geothermal.validate forward  --n 32    # <= 6 Newton iterations to 1e-9
    python -m applications.geothermal.validate variance --n 16    # MC and sample variance vs the exact posterior variance
    python -m applications.geothermal.validate partition a.npz b.npz c.npz   # dumps at 1, 2, 4 ranks agree

Each check prints its verdict; the numbers below come from the records `run.py`
writes.

## Numbers (L40S cards)

| run            | state dofs | obs   | build | MAP (Newton / CG)        | Laplace (k)    | sample | box corr. | coverage | QoI: truth / MAP / std lin. / std sampled |
|----------------|-----------:|------:|------:|--------------------------|----------------|-------:|----------:|---------:|-------------------------------------------|
| 32³, 1 card    | 274 625    | 1 320 | 18 s  | 236 s (24 / 790)         | 18 s (50)      | 26 ms  | 0.82      | 96.5 %   | 85.09 / 85.17 / 0.45 / 0.46 K             |
| 32³, 2 cards   | 274 625    | 1 320 | 17 s  | 222 s (24 / 784)         | 21 s (50)      | 40 ms  | 0.82      | 96.4 %   | 85.09 / 85.17 / 0.41 / 0.40 K             |
| 32³, 4 cards   | 274 625    | 1 320 | 22 s  | 191 s (27 / 932)         | 17 s (50)      | 44 ms  | 0.82      | 96.5 %   | 85.09 / 85.17 / 0.44 / 0.46 K             |
| 64³, 4 cards   | 2 146 689  | 2 700 | 41 s  | 506 s (19 / 690)         | 147 s (200)    | 75 ms  | 0.72      | 92.5 %   | 45.00 / 44.92 / 0.10 / 0.14 K             |
| 64³, 4 cards, k = 100 | 2 146 689 | 2 700 | 48 s | 556 s (20 / 745)     | 83 s (100)     | 71 ms  | 0.72      | 94.6 %   | 45.00 / 44.92 / 0.16 / 0.19 K             |
| 96³, 4 cards   | 7 189 057  | 4 020 | 123 s | 2177 s (30 / 882)        | 220 s (100)    | 129 ms | 0.81      | 92.5 %   | 80.95 / 80.97 / 0.30 / 0.44 K (8 samples) |
| 128³, 4 cards, Gauss-Newton | 16 974 593 | 5 400 | 236 s | 4220 s (30 / 1013) | 266 s (50)     | 247 ms | 0.75      | 93.0 %   | not reached (see below)                  |

"Box corr." is the correlation of MAP and truth inside the body's box; "coverage" the
fraction of dofs at which the truth lies within two posterior standard deviations of
the MAP; the QoI is the mean temperature above the surface in the target volume, with
its linearized posterior standard deviation and the standard deviation over 32 posterior
samples pushed through the nonlinear forward solve.

**The samples run hotter than the MAP.**  The sample mean of the QoI is above its value
at the MAP at every size, by more than the linearized std: +0.72 K at 32³ (1.6 σ) and
+0.35 K at 64³ (3.6 σ, 14 standard errors of the sample mean).  The sign is the physics:
with a fixed basal flux the temperature at depth integrates q/k, and
E[1/k] = E[exp(−m)] = exp(−μ + σ²/2) > 1/E[k], so a random conductivity conducts worse
on average than its mean field.  The first-order posterior of the QoI cannot see it; the
second-order Taylor term ½ tr(Γ_post H_q) can, and `qoi.taylor_qoi` computes it two
ways: the low-rank factorization of the posterior-preconditioned QoI Hessian
(`forward_uq.TaylorApproximationQoi`, k = 30) finds eigenvalues of 1e-9 and a shift of
0.000 K, because that Hessian's spectrum is flat (every dof's exp(m) adds a little
curvature and no direction dominates); a Hutchinson estimate ½ mean(sᵀ H_q s) over 32
zero-mean posterior samples gives +0.39 ± 0.02 K at 64³ against the sampled +0.35 K, in
58 s against 289 s for the 32 nonlinear forward solves, and +0.77 ± 0.05 K at 32³
against the sampled +0.72 K (21 s against 120 s), and +0.93 ± 0.08 K at 96³ against
+0.99 K over 8 samples (137 s against 237 s).  Both are in the JSON
(`qoi_taylor2_mean_K`, `qoi_taylor2_hutchinson_mean_K`) and the dashed curve of the QoI
figure is the second-order mean with the linearized width.

The truth is a different prior
sample at every resolution (the stream is indexed by dof), and at 64³ its domain mean of
0.33 in log k makes the block 1.4 times more conductive than at 32³, hence the 45 K
target temperature against 85 K.  At 64³ all 100, and then all 200, eigenvalues asked
for are above one (the smallest 2.6 at k = 200): 2 700 borehole temperatures at 0.5 K
inform more than 200 directions, and the posterior std keeps falling with k (0.378 to
0.364 mean over dofs), the coverage with it (94.6 to 92.5 %).  The sampled QoI std
exceeds the linearized one by a third at 64³ (0.14 against 0.10 K, 32 samples) where
the two agreed at 32³: the forward map's nonlinearity in the target volume.  The 32³ MAP
does not scale from one to four cards (236 to 191 s) because 274 k dofs do not fill a
card.  Newton-CG iteration counts vary between device runs of the same problem (27/943
and 24/790 at 32³ on one card: hypre's device reductions are not deterministic and the
line search and CG stopping are sensitive to the last bits); J agrees to five digits.  The 96³ run (7.2 M state dofs, 33 GB per card with `HIPPYMFEM_GPU_MEM_FRACTION=0.4`;
at 0.25 the JAX pool ran dry in the Hessian-block assembly) stopped the MAP at the
30-iteration cap with the gradient still above the tolerance, so its MAP numbers are for
30 iterations, not for convergence.  At 128³ (17.0 M state dofs) the build and the truth's forward solve went through
(290 s) and the MAP ran out of card memory in hypre: the non-symmetric Jacobian means
A, its transpose and two AMG hierarchies per step, more than the 44.5 GB of a L40S; it
needs 80 GB cards for full Newton.  With Gauss-Newton (`--gauss-newton`) and the
transpose-free adjoint (`transpose_free_adjoint=True`, the adjoint through A's transpose
action with A's hierarchy, no transposed copy and no second hierarchy) it runs on the
four L40S at 28 GB per card: MAP in 4220 s at the 30-iteration cap (1013 CG iterations),
Laplace with k = 50 in 266 s, a sample in 247 ms (the first run measured 4080 s, 945 CG
iterations, 254 s and 213 ms: device runs differ in their Newton-CG path), posterior std 0.478 → 0.374, coverage
93.0 %, correlation 0.75 in the body's box (0.74 over the domain).  Both runs died in the QoI stage after all of that, at 42.7 GB per card, the second one
with the MAP's operators released before the sampled forward solves, so the extra 15 GB
of that stage is not the held linearization point and is not yet explained; the record
and the field dump (`geothermal_n128_r4.{json,npz}`) are written before the stage, and
`--skip-qoi` stops there.  The QoI at 128³ is therefore not reported.  The 16³ host rows (`geothermal_n16_host_r*.json`) exist for the partition check; `geothermal_k100_n64_r4.json` is the 64³ run with k = 100.

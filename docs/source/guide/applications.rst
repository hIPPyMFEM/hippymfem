An application: basin-scale geothermal inversion
================================================

``applications/geothermal/`` is a complete Bayesian inversion built on the library, with
a residual that no form language writes: a conductivity whose temperature dependence is
a *table* per lithology, a lithology map with dipping interfaces, an anisotropic tensor,
a basal heat flux as a boundary density.  It runs end to end on GPUs and is the worked
example of what the quadrature-point AD design is for.

The problem
-----------

Steady heat conduction in an 8 km x 8 km x 4 km block, scaled to the unit cube:

.. math::

   -\nabla\cdot\big(k(m, u, x)\, D\, \nabla u\big) = q(x), \qquad
   k = e^{m(x)}\, f_{\ell(x)}(T_{\mathrm{scale}}\, u), \qquad D = \mathrm{diag}(1, 1, (L/H)^2),

with the surface temperature as Dirichlet data on the top face, a basal heat flux of
66 mW/m^2 on the bottom face and no flux on the sides.  The unknown ``m`` is the log of
the rock's reference conductivity relative to 2.5 W/(m K), on P1; the temperature is P2.
``f_ℓ`` is the Vosteen–Schellschmidt law of lithology ``ℓ`` tabulated at 16 temperatures
and read with a Gaussian kernel, so it is smooth in ``u`` and the Hessian is exact.  The
residual density, in full:

.. code-block:: python

   def varf(u, m, p, x):
       li = lithology_index(x)                       # jnp.where on two dipping interfaces
       T = T_SCALE * u.val
       wts = jnp.exp(-0.5 * ((T - nodes) / w) ** 2)
       f = jnp.sum(wts * jnp.take(tabs, li, axis=0)) / jnp.sum(wts)
       k = jnp.exp(m.val) * f
       return k * jnp.dot(u.grad * D, p.grad) - jnp.take(q_rad, li) * p.val

   def bdr_varf(u, m, p, x, n):                     # bottom face: heat enters from below
       return -Q_BASAL * p.val

The data are temperature logs from 60 boreholes to 2.8 km depth (one sample per element
layer) at 0.5 K noise; the prior is a BiLaplacian with a 2 km horizontal and 0.5 km
vertical correlation length and a marginal standard deviation of 0.5 in log k; the
synthetic truth is a prior sample plus a buried conductive body (log k + 1, radius 1 km,
2.3 km deep).

Two things about the solvers follow from the physics.  ``dR/du`` carries
``k'(u) du grad u . grad p`` and is not symmetric, so the forward and incremental solves
are GMRES with BoomerAMG and the problem is built with ``symmetric_jacobian=False`` (the
adjoint uses the true transpose; the shared-operator memory saving of the symmetric case
is not available).  And the forward problem is nonlinear: Newton converges in four
iterations from the boundary data (residual 1.9e-2, 2.0e-4, 1.2e-8, 4.8e-15 at 16^3).

Running it
----------

.. code-block:: bash

   HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 PYTHONPATH=<cuda pymfem> \
     mpirun -n 4 tools/mpirun_pinned.sh python -m applications.geothermal.run --n 64 --k 100 \
       --out results/geothermal_n64_r4.json --dump results/geothermal_n64_r4.npz \
       --paraview results/paraview/geothermal_n64
   python -m applications.geothermal.figures results/geothermal_n64_r4.npz \
       --json results/geothermal_n64_r4.json --out results/figures/geothermal_n64

The stages, each timed and recorded in the JSON: the synthetic truth and data; the
prior's pointwise standard deviation by Monte Carlo (so the numbers can be read); the MAP
by inexact Newton-CG; the Laplace approximation (``doublePassG`` with ``k`` eigenpairs);
posterior samples, the Monte Carlo pointwise variance, traces and the KL divergence from
the prior; the fraction of dofs at which the truth lies within two posterior standard
deviations of the MAP; and a quantity of interest, the mean temperature in a target
volume around the body, at the MAP with its linearized posterior standard deviation
:math:`\sqrt{g^{\top} \Gamma_{\mathrm{post}} g}` and over posterior samples pushed through the
nonlinear forward solve.

What it gives
-------------

At :math:`32^{3}` on one L40S (274 625 state dofs, 1 320 observations):

====================================  ================================================
stage                                 result
====================================  ================================================
build (truth, data, one forward)      18 s
MAP                                   236 s, 24 Newton iterations, 790 CG iterations
Laplace (blocks + eigensolver, k=50)  18 s; 50 eigenvalues above 1
posterior sample                      26 ms
recovery in the body's box            correlation 0.82 (0.72 over the domain)
truth within 2 posterior std          96.5 % of dofs
QoI: mean target temperature          truth 85.1 K, MAP 85.2 K; std 0.45 K linearized,
                                      0.46 K over 32 samples through the forward solve
====================================  ================================================

At :math:`64^{3}` on four L40S (2 146 689 state dofs, 2 700 observations): built in 41 s,
the MAP in 506 s (19 Newton and 690 CG iterations), the Laplace approximation with
k = 200 in 147 s (all 200 eigenvalues above one: the logs inform more directions than
that), a posterior sample in 75 ms, the correlation of MAP and truth 0.72 in the body's
box, the truth within two posterior standard deviations of the MAP at 92.5 % of the
dofs, and the target temperature 45.00 K in truth against 44.92 K at the MAP with a
linearized posterior standard deviation of 0.10 K and 0.14 K over 32 samples through the
nonlinear forward solve (the truth is a different prior sample at each resolution, 1.4
times more conductive at 64^3, hence the lower temperature).  The full table is in
``applications/geothermal/README.md``, with the :math:`96^{3}` row (7.2 M state dofs,
33 GB per card, the MAP at its 30-iteration cap in 2177 s) and the :math:`128^{3}`
attempt, which ran out of card memory at the MAP because the non-symmetric Jacobian
holds two operators and two hierarchies per step.

The samples run hotter than the MAP: the sample mean of the QoI is +0.35 K above its
value at the MAP at 64^3, 3.6 linearized standard deviations, because with a fixed basal
flux the temperature integrates :math:`q/k` and :math:`E[1/k] > 1/E[k]`.  The first-order
posterior of the QoI cannot see this; the second-order Taylor term
:math:`\tfrac12 \mathrm{tr}(\Gamma_{\mathrm{post}} H_q)` can, and the application computes it
two ways (``qoi.taylor_qoi``): the low-rank factorization of
:class:`~hippymfem.forward_uq.TaylorApproximationQoi` finds eigenvalues of 1e-9 and no
shift, because that Hessian's spectrum is flat, while a Hutchinson estimate over 32
zero-mean posterior samples gives +0.39 +- 0.02 K in 58 s, against the sampled +0.35 K
in 289 s of nonlinear forward solves.  Which trace estimator fits is a property of the
QoI, and both are one call.

The figure the run produces (``figures.py``) shows the body recovered where the logs
reach it and the posterior standard deviation falling from the prior's 0.5 to 0.2 along
the logs and returning to the prior below them.

The checks (``validate.py``)
----------------------------

``fd`` (finite-difference slopes 1.0000 for the gradient and 1.0000 for the Hessian, a
Hessian asymmetry of 7e-10 with GMRES solves), ``forward`` (four Newton iterations to
3e-14 relative), ``variance`` (the Monte Carlo posterior variance is 1.010 of the exact
one on average over the dofs at 600 samples; the sample variance 1.008) and
``partition``: the dumps of ``run.py`` at 1, 2 and 4 ranks (16^3, host).  The random
streams are partition independent (see :doc:`parallel`); the Krylov solves are not, and
that is what the comparison shows: the synthetic data agree to 2e-9 and the truth (a prior
sample, an A-solve at 1e-8) to 1e-8, the MAP to 3e-5, the eigenvalues to 4e-4, and the
Monte Carlo posterior std to 0.13, because at k = 20 with more than 20 eigenvalues above
one the truncated decomposition is only as reproducible as the randomized eigensolver.
hIPPYlibx, whose generator is seeded per rank, draws a different truth on every rank
count (relative difference 1.2 and 1.7 at 2 and 4 ranks, ``validation/partition_hippylibx.py``).

Three things the first runs taught
----------------------------------

- **Measure the prior's std in 3D.**  The first BiLaplacian numbers gave a marginal std
  of 16 in log k: conductivity contrasts of :math:`e^{60}` and a forward solve that could
  not converge at the truth while every diagnostic at ``m = 0`` was healthy.  The
  variance scales as :math:`1/(\gamma^{3/2}\delta^{1/2})` at fixed :math:`\Theta`;
  ``prior.pointwise_variance(method="MonteCarlo", n=48)`` takes seconds at 16^3.
- **Scale the sources with the geometry.**  Dividing the physical weak form by
  :math:`H k_{\mathrm{ref}} T_{\mathrm{scale}}`, a basal flux :math:`q` becomes
  :math:`q L^{2}/(H k_{\mathrm{ref}} T_{\mathrm{scale}})`; the first value was off by the
  aspect-ratio factor and gave a 35 K contrast over 4 km instead of 105 K.
- **Know where the data can see.**  With a Dirichlet top and a flux bottom, the
  temperature above a conductive body is set by the resistance to the surface, so a body
  at 2.6 km under 2 km logs left no trace in the data (correlation 0.41 in its box).  The
  logs now reach 2.8 km and the body sits at 2.3 km.

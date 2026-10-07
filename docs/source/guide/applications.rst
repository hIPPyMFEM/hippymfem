An application: basin-scale geothermal inversion
================================================

``applications/geothermal/`` is a complete Bayesian inversion built on the library, with
a residual that no form language writes: a conductivity whose temperature dependence is
a *table* per lithology, a lithology map with dipping interfaces, an anisotropic tensor,
a basal heat flux as a boundary density.  It runs on GPUs and is the worked example of
what the quadrature-point AD design is for.

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

The data are temperature logs from 60 boreholes to 2.8 km depth, a sample every 62.5 m
(2 700 in all, whatever the mesh), at 0.5 K noise; the prior is a BiLaplacian with a 2 km
horizontal and 0.5 km vertical correlation length and a marginal standard deviation of
0.5 in log k; the synthetic truth is a random field with the prior's spectrum, without
its shortest waves, plus a buried conductive body (log k + 1, radius 1 km, 2.3 km deep),
both functions of the point.  Neither the data nor the truth depend on the mesh, so a
finer mesh solves the same inverse problem more accurately.

Two things about the solvers follow from the physics.  ``dR/du`` carries
``k'(u) du grad u . grad p`` and is not symmetric, so the forward and incremental solves
are GMRES with BoomerAMG and the problem is built with ``symmetric_jacobian=False`` and
``transpose_free_adjoint=True``: the adjoint applies the true transpose with the forward
operator's AMG hierarchy.  And the forward problem is nonlinear: Newton converges in
four iterations from the boundary data.

Running it
----------

.. code-block:: bash

   HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 PYTHONPATH=<cuda pymfem> \
     mpirun -n 4 tools/mpirun_pinned.sh python -m applications.geothermal.run --n 64 --k 200 \
       --out results/geothermal_n64.json --dump results/geothermal_n64.npz \
       --paraview results/paraview/geothermal_n64
   python -m applications.geothermal.figures results/geothermal_n64.npz \
       --json results/geothermal_n64.json --out results/figures/geothermal_n64

The stages, each timed and recorded in the JSON: the synthetic truth and data; the
prior's pointwise standard deviation by Monte Carlo; the MAP by inexact Newton-CG; the
Laplace approximation (``doublePassG`` with ``k`` eigenpairs); posterior samples, the
Monte Carlo pointwise variance, traces and the KL divergence from the prior; the
fraction of dofs at which the truth lies within two posterior standard deviations of
the MAP; and a quantity of interest, the mean temperature in a target volume around the
body, at the MAP with its linearized posterior standard deviation
:math:`\sqrt{g^{\top} \Gamma_{\mathrm{post}} g}` and over posterior samples pushed through the
nonlinear forward solve.  ``validate.py`` holds the application's own checks:
finite-difference slopes, the forward Newton convergence, the Monte Carlo variance
against the exact one, and the agreement of runs on 1, 2 and 4 ranks.

What it gives
-------------

At :math:`64^{3}` on one AMD MI250X (2 146 689 state dofs, 2 700 observations), with
``--gauss-newton --k 600``:

====================================  ================================================
stage                                 result
====================================  ================================================
build (truth, data, one forward)      82 s
MAP                                   213 s, 16 Newton iterations, 363 CG iterations
Laplace (eigensolver, 600 pairs)      393 s; 506 eigenvalues above one
posterior sample                      0.29 s
recovery in the body's box            correlation 0.96 (0.83 over the domain)
QoI: mean target temperature          truth 71.60 K, MAP 71.58 K; std 0.08 K linearized,
                                      0.19 K over 32 samples through the forward solve
====================================  ================================================

The data and the truth are the same on every mesh, so a finer mesh repeats the inversion:
at :math:`128^{3}` on four MI250X and at :math:`256^{3}` on 32 (135 million state dofs)
the MAP takes 16 and 17 Newton iterations with 358 and 359 CG iterations, in 554 s and
649 s, and the eigensolver finds 491 and 483 eigenvalues above one among 566, in 992 s
and 1 171 s; the largest eigenvalue is 1.008e6, 1.009e6 and 1.010e6 on the three meshes
and the target temperature at the MAP 71.58, 71.57 and 71.59 K.  The body is recovered
where the logs reach it, and the posterior standard deviation falls from the prior's
0.5 to 0.2 along the logs and returns to the prior below them.

The QoI over posterior samples lies above its value at the MAP: with a fixed basal flux
the temperature integrates :math:`q/k`, and :math:`E[1/k] > 1/E[k]`.  The first-order
posterior of the QoI cannot see this; the second-order Taylor term
:math:`\tfrac12 \mathrm{tr}(\Gamma_{\mathrm{post}} H_q)` can, and ``qoi.taylor_qoi``
computes it with a Hutchinson estimate over posterior samples, at a fifth of the cost of
the sampled forward solves.

Three things to check in a problem of your own
----------------------------------------------

- **Measure the prior's std in 3D.**  The variance of a BiLaplacian prior scales as
  :math:`1/(\gamma^{3/2}\delta^{1/2})` at fixed :math:`\Theta`, and a marginal std of 16
  in log k means conductivity contrasts of :math:`e^{60}` and a forward solve that cannot
  converge at a prior sample while every diagnostic at ``m = 0`` is healthy.
  ``prior.pointwise_variance(method="MonteCarlo", n=48)`` takes seconds at 16^3.
- **Scale the sources with the geometry.**  Dividing the physical weak form by
  :math:`H k_{\mathrm{ref}} T_{\mathrm{scale}}`, a basal flux :math:`q` becomes
  :math:`q L^{2}/(H k_{\mathrm{ref}} T_{\mathrm{scale}})`: the aspect ratio enters.
- **Know where the data can see.**  With a Dirichlet top and a flux bottom, the
  temperature above a conductive body is set by the resistance to the surface, so a body
  below the logs leaves no trace in the data.  Here the logs reach 2.8 km and the body
  sits at 2.3 km.

Forward uncertainty propagation
===============================

Given a distribution on the parameter, what is the distribution of a quantity of
interest?

A quantity of interest
----------------------

.. code-block:: python

   qoi = hm.VariationalQoi(Vh, lambda u, m, x: u.val * u.val)

:class:`~hippymfem.forward_uq.variationalQoi.VariationalQoi` takes a density exactly
like the PDE residual does, so its derivatives with respect to the state and the
parameter are automatic.

The parameter-to-QoI map
------------------------

.. code-block:: python

   p2q  = hm.Parameter2QoiMap(pde, qoi)
   q    = p2q.reduced_eval(m)          # solve forward, then evaluate
   q, g = p2q.reduced_gradient(m)      # ... and one adjoint solve for the gradient
   H    = p2q.hessian(m)               # matrix-free QoI Hessian at that point

Solving the forward problem and one adjoint problem per QoI gives the gradient; the
Hessian action costs an incremental pair.  Third derivatives of the residual enter
here, which is why the kernel provides them.
:func:`~hippymfem.forward_uq.parameter2QoiMap.parameter2QoiMapVerify` (hIPPYlib's
``qoiVerify``) checks both against finite differences.

Taylor approximations with variance reduction
---------------------------------------------

.. code-block:: python

   Omega = hm.MultiVector(prior.mean, 40)
   hm.parRandom.normal_multivector(1.0, Omega)

   taylor = hm.TaylorApproximationQoi(p2q, prior)
   d, U = taylor.computeLowRankFactorization(Omega, k=40)   # required first
   mean2, var2 = taylor.expectedValue(order=2), taylor.variance(order=2)

   mc = hm.varianceReductionMC(prior, p2q, taylor, 300, order=2)
   mc["reduced_mean"], mc["reduced_stderr"], mc["variance_reduction"]

A second-order Taylor expansion of the QoI around the mean of the distribution gives a
closed-form mean and variance, from the eigenvalues :math:`d_i` of the QoI Hessian
preconditioned by the covariance:
:math:`\mathbb{E} \approx \bar q + \tfrac12\sum_i d_i` and
:math:`\mathrm{Var} \approx g^{\top}\mathcal{C}g + \tfrac12\sum_i d_i^2`.  The
factorization is therefore the cost, and it has to be computed before the moments are
asked for.

It is cheap and biased.  Using it as a control variate in a Monte Carlo estimate removes
the bias while keeping most of the variance reduction, so a few hundred samples can match
what plain Monte Carlo needs thousands for.  Report both, with the measured variance
reduction factor, rather than the Taylor moments alone.

After the data
--------------

:class:`~hippymfem.modeling.posterior.GaussianLRPosterior` is a Gaussian too, so it stands
in for the prior everywhere above: ``TaylorApproximationQoi(p2q, post)`` expands the map at
the MAP point and preconditions the QoI Hessian with the posterior covariance, which is what
makes a prediction after data as cheap as one before it.  ``varianceReductionMC`` takes a
prior (it draws through ``sample_noise``); with a posterior, draw with ``post.sample()`` and
average ``Q - Q_taylor`` yourself, as
`tutorial 12 <../tutorials/12_ForwardUQ.html>`_ does.

The posterior
=============

The Laplace approximation
-------------------------

At the MAP point the posterior is approximated by a Gaussian whose covariance is the
inverse Hessian.  That Hessian is never formed: the misfit part is numerically low
rank, since data informs finitely many directions, so a randomized generalized
eigensolver finds the dominant subspace from a few dozen Hessian applications.

.. code-block:: python

   model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
   Hmisfit = hm.ReducedHessian(model, misfit_only=True)
   Omega = hm.MultiVector(x[PARAMETER], k + 10)
   hm.parRandom.normal_multivector(1.0, Omega)
   d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)
   post = hm.GaussianLRPosterior(prior, d, U)
   post.mean = x[PARAMETER]

``d`` and ``U`` solve the generalized problem :math:`H_{\text{misfit}} u = \lambda R u`,
so the approximation is made in the prior-weighted inner product, which is the one in
which the truncation is meaningful.

What the posterior gives
------------------------

==========================================  =================================
``post.sample(noise, s)``                   a posterior sample
``post.pointwise_variance(method=...)``     pointwise variance field
``post.trace()``                            trace of the covariance
``post.klDistanceFromPrior()``              KL divergence from the prior
``post.cost(m)``                            the Gaussian potential
==========================================  =================================

Choosing ``k``
--------------

Look at the spectrum.  The eigenvalues decay, and ``k`` should reach past where
:math:`\lambda_i` falls below about 1, because directions with :math:`\lambda \ll 1` are
prior-dominated and contribute almost nothing to the update.  Taking ``k`` far past
that point buys little, and the trailing eigenpairs are the least accurate ones:
their error is set by the oversampling and the power iterations (at a spectral ratio
of 3e6 the 40th eigenvalue is off by 7 % with ``p = 10, s = 2`` and by 6e-5 with
``p = 25, s = 3``; :doc:`../limits` has the table), and the trace, pointwise variance
and KL divergence inherit it.  More oversampling and more power iterations both help,
every iteration (they are re-orthonormalized); the exact dense spectrum on a coarse
mesh is the way to check.

Solvers matter here
-------------------

Each Hessian application is an incremental forward solve and an incremental adjoint
solve.  With an iterative solver at tolerance :math:`\tau`, the Hessian action carries
an error of that order and the computed spectrum cannot be cleaner.  Use
``LUSolver``, exact on any rank count, when the spectrum itself is the object of
study.

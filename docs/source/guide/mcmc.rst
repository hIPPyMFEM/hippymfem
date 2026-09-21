MCMC
====

The samplers here are dimension independent: their acceptance rate does not
degrade as the mesh is refined, because the proposals respect the prior.

.. code-block:: python

   kernel = hm.pCNKernel(model, s=0.1)
   chain  = hm.MCMC(kernel)
   chain.parameters["number_of_samples"] = 5000
   chain.parameters["burn_in"] = 500
   tracer = hm.FullTracer(Vm, chain.parameters["number_of_samples"])
   n_accept = chain.run(m0, qoi=qoi, tracer=tracer)

================================================  ============================
:class:`~hippymfem.mcmc.kernels.pCNKernel`        preconditioned Crank-Nicolson;
                                                  proposes along the prior
:class:`~hippymfem.mcmc.kernels.gpCNKernel`       pCN around the Laplace
                                                  approximation, with much better
                                                  mixing when it is a good fit
:class:`~hippymfem.mcmc.kernels.MALAKernel`       Metropolis-adjusted Langevin,
                                                  using the gradient
:class:`~hippymfem.mcmc.kernels.ISKernel`         importance sampling from the
                                                  Laplace approximation
================================================  ============================

Reporting a chain honestly
--------------------------

An acceptance rate is not a convergence diagnostic.  Report the integrated
autocorrelation time and the effective sample size:

.. code-block:: python

   iact, lags, acorr = hm.integratedAutocorrelationTime(tracer.data[:, 0])
   ess = len(tracer.data) / iact

A chain with 1500 samples and an IACT of 70 has an effective sample size of about
21, and a mean computed from it is not a result.  Tune the step size to an acceptance
rate in the 20-40% range first, then run long enough that the effective sample size
justifies the precision being quoted.  :func:`~hippymfem.mcmc.diagnostics.integratedAutocorrelationTime`
is checked against an AR(1) process whose IACT is known analytically.

Cost
----

Every proposal requires a forward solve, and for a nonlinear problem that means
reassembling the Jacobian.  This is where the element kernels dominate the runtime
and where :doc:`gpu` pays off most directly.

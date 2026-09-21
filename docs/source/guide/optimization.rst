Finding the MAP point
=====================

Inexact Newton-CG
-----------------

.. code-block:: python

   params = hm.ReducedSpaceNewtonCG_ParameterList()
   params["rel_tolerance"] = 1e-9
   params["max_iter"] = 30
   solver = hm.ReducedSpaceNewtonCG(model, params)
   x = solver.solve([None, prior.mean.copy(), None])
   print(solver.termination_reasons[solver.reason], solver.it,
         solver.final_grad_norm / solver.initial_grad_norm)

The Newton system is solved with conjugate gradients against the matrix-free reduced
Hessian, truncated by the Eisenstat-Walker criterion so early iterations do not pay
for accuracy they cannot use.  The prior precision is the preconditioner, which is
what makes the iteration count depend on the information content of the data rather
than on the mesh.

Two globalizations, selected by ``globalization``:

==================  ==============================================================
``"LS"``            Armijo backtracking line search (the default)
``"TR"``            Steihaug trust region, for strongly indefinite Hessians
==================  ==============================================================

The Gauss-Newton approximation, the full Hessian with the terms involving the
second derivative of the forward operator dropped, is used for the first
``GN_iter`` iterations, because it is positive definite everywhere while the full
Hessian is legitimately indefinite away from a minimum.

Gradient norms
--------------

``gradient_norm`` selects the norm the stopping test uses: ``"M"``, the discrete
:math:`L^2` norm through the mass matrix, is the default and matches hIPPYlib;
``"R"`` uses the prior precision.  This is not cosmetic: the iteration history
differs visibly between the two, and matching hIPPYlib's choice is what makes the
two libraries' histories agree step for step.

BFGS
----

.. code-block:: python

   params = hm.BFGS_ParameterList()
   params["BFGS_op"]["memory_limit"] = 25
   solver = hm.BFGS(model, params)
   x = solver.solve([None, prior.mean.copy(), None], H0inv=prior.Rsolver)

Limited-memory BFGS with the prior solver as the initial inverse Hessian.  It needs
only the cost and the gradient, so agreement between BFGS and Newton-CG on the
minimizer is strong evidence that **both** the gradient and the Hessian are right:
two optimizers with nothing in common beyond the cost and the gradient cannot agree
by accident.  The test suite requires them to agree to 2e-3 and observes 1e-5 to
1e-6.

Expect BFGS to need hundreds of iterations where Newton-CG needs tens.  That is a
statement about preconditioning, not correctness, and the iteration count moves by
about 10% under round-off-level changes, so do not treat it as a regression signal.

Verifying a new problem
-----------------------

Before trusting a gradient on a new residual:

.. code-block:: python

   eps, err_grad, err_H = hm.modelVerify(model, m0, eps=np.logspace(-1, -6, 11))
   print(hm.best_slope(eps, err_grad), hm.best_slope(eps, err_H))

Both slopes should be 1.0.  ``best_slope`` takes the median over the step sizes where
round-off has not taken over; choose the window from the measured error curve rather
than leaving the default, because the Hessian difference reaches its floor much
earlier than the gradient's when the misfit is nearly quadratic.

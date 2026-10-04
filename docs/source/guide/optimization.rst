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
for accuracy they cannot use (`The CG of a Newton step`_).  The prior precision is the preconditioner, which is
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

The CG of a Newton step
-----------------------

The CG iteration keeps every new residual orthogonal to all the earlier ones
explicitly (``cg_reorthogonalize``, on by default;
:class:`~hippymfem.algorithms.cgsolverSteihaug.CGSolverSteihaug`,
``reorthogonalize``).  The recurrence of CG gives that orthogonality only for one
symmetric operator in exact arithmetic.  A prior-preconditioned Hessian has a few
large, well separated eigenvalues, for which rounding destroys it within a few tens of
iterations, and a Hessian action computed with inexact solves is not one symmetric
operator at all.  In both cases the recurrence alone is delayed, and by an amount that
changes with the last digit of the operands: the model problem of the benchmarks with
2.1 million state dofs took 193 to 210 CG iterations for its twelve Newton steps,
depending on the GPU and the number of ranks, and with the incremental solves stopped
at ``1e-8`` instead of ``1e-12`` a third more.  With the residuals orthogonalized it
took 131 on every GPU and rank count, and still 131 with the incremental solves stopped
at ``1e-6``.  The cost is ``k`` inner products and updates of parameter vectors at
iteration ``k`` and two stored vectors per iteration, nothing next to a Hessian action.

Two consequences.  The incremental solves of a Hessian action need a loose tolerance
only, which halves their iterations:

.. code-block:: python

   pde.set_solvers(hm.auto_solver, Vu, comm, max_direct=0, rel_tolerance=1e-12)
   for name in ("solver_fwd_inc", "solver_adj_inc"):
       getattr(pde, name).parameters["rel_tolerance"] = 1e-6

The forward and adjoint solves keep their tolerance: they give the gradient, which must
be exact for Newton to converge to the MAP point, while the Hessian only shapes the
steps.  Newton-CG to a relative gradient norm of ``1e-6`` on that problem, the same
twelve Newton steps and the same cost functional to nine digits in every column:

==============================================  ========  ========  =====================
..                                              H100      L40S      Blackwell instance
==============================================  ========  ========  =====================
recurrence alone, incremental solves to 1e-12   63.8 s    161.5 s   163.3 s
residuals orthogonalized                        47.4 s    110.5 s   122.2 s
and incremental solves to 1e-6                  32.0 s    73.3 s    83.8 s
==============================================  ========  ========  =====================

(one GPU each, or one MIG instance of an RTX PRO 6000 Blackwell, with MFEM's CG for the
solves in all three rows; hypre's own PCG, :doc:`solvers`, takes another 7 to 12 % off
the last.  At ``1e-4`` the H100 took 27.8 s, but one of the three GPUs then needed a
thirteenth Newton step, so ``1e-6`` is the tolerance to use.)  And single precision becomes usable for those
solves (:ref:`single-precision` in the GPU guide), since their rounding no longer
disturbs the iteration.

``cg_reorthogonalize = False`` gives hIPPYlib's iteration, for a comparison step by
step.  The low-rank eigensolvers of the Laplace approximation are not Krylov
recurrences and never had this sensitivity; their Hessian actions want incremental
solves to about ``1e-8`` for the accuracy of the small eigenvalues.

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

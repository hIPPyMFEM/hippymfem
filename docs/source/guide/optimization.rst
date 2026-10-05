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

Two more solves inside that CG are cheaper than their tolerances say.

**The prior's solves, where they precondition the CG, stop at 1e-6**
(``cg_preconditioner_tolerance``, the default).  On the model problem that is eleven CG
iterations for each of the two solves of :math:`R^{-1} = A^{-1} M A^{-1}` instead of
twenty-one, with the same Newton and CG counts and the same gradient norm after every
Newton step to four digits; with single-precision solves the Newton-CG solve took
23.6 s instead of 25.3 s on an H100 and 43.2 s instead of 46.1 s on an L40S.  It cannot
be much looser.  A solve stopped at a tolerance is not one linear map of its right-hand
side, so what the orthogonalization removes from a residual is then not all error of
the iterate, and the residual the CG stops on is off by about that tolerance times the
first residual (times the iteration count to the power 1.5): at 1e-5 the twelfth
Newton step of the model problem no longer met its tolerance.  The library never uses
more than a thousandth of the CG's own tolerance, and it leaves the solves as they are
without ``cg_reorthogonalize`` and in the trust-region method; ``0`` leaves them as
they are everywhere.

**The Hessian actions may lose accuracy as the CG converges**
(``cg_hessian_relaxation``, off by default).  The error of a product enters the
residual in proportion to the step it multiplies, and the steps shrink with the
residual (inexact Krylov methods).  With ``cg_hessian_relaxation = c`` the two
incremental solves of the action at CG iteration ``k`` stop at ``c`` times the CG's
tolerance times :math:`\|r_0\| / \|r_k\|`, where that is looser than their own
tolerance.  The prior's part of the action keeps its accuracy: a relative error of its
mass solve is the same relative error of :math:`R x`, with nothing to damp it.  With
``c = 1e-2`` the incremental solves of the model problem took 6.5 iterations on average
instead of eleven, in the same Newton and CG counts at 32\ :sup:`3` and 64\ :sup:`3` with
the gradient norm after each Newton step within two percent, and the Newton-CG solve
with single-precision solves took 32.4 s instead of 39.9 s on an L40S; with
``c = 1e-1`` the iteration took a different path (twelve Newton steps instead of eleven
at 32\ :sup:`3`).  It is off by default because it was not as safe on other problems.
With ten times more observations the steps stayed fifteen at ``c = 1e-2`` and became
sixteen at ``1e-3``.  With noise ten times smaller, a more informative problem whose CG
runs into ``cg_max_iter`` in most late steps, the eighteen Newton steps and 488 CG
iterations became 23 and 597 at ``1e-2`` (20 and 514 at ``1e-3``), slower overall.  Linear
elements at 128\ :sup:`3` kept their twelve steps and 149 CG iterations at ``1e-2`` and
took 9 % less time.  The incremental solves' error enters through the misfit Hessian,
whose largest eigenvalues grow with the information in the data, so the ``c`` that is
safe shrinks with it, and nothing here knows them in advance.

**Counting the last Newton step.**  The twelve steps of the model problem at
64\ :sup:`3` end with a gradient norm of 0.0976 against a tolerance of 0.1158, and the CG
of the twelfth step passes its own test at its 33rd iteration by less than a percent.
A change of the path of that size (the prior's solves at 1e-4, a relaxation of 3e-3,
another GPU) ends that CG at its 32nd iteration and the step at 0.127, and a thirteenth
step of 37 CG iterations follows.  Thirteen steps and 167 CG iterations in a table are
then not a slower method: compare the gradient norms step by step.

``cg_reorthogonalize = False`` gives hIPPYlib's iteration, for a comparison step by
step.  The low-rank eigensolvers of the Laplace approximation are not Krylov
recurrences and never had this sensitivity; the accuracy of their eigenpairs is that of
the incremental solves: at 64\ :sup:`3` the 50 leading eigenvalues came out to 1e-8 with
the solves at ``1e-8`` and to 1e-5 (the pointwise posterior variance to 6e-6) with the
solves at ``1e-5``, in either precision (:ref:`single-precision`).

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

Time-dependent problems
=======================

:class:`~hippymfem.modeling.TimeDependentPDEVariationalProblem.TimeDependentPDEVariationalProblem`
solves an initial-condition or parameter inversion for a time-dependent PDE.  The
residual density takes **four** fields: the new time level, the previous one, the
parameter and the adjoint.

.. code-block:: python

   def varf(u_new, u_old, m, p, x, t, dt):
       theta = 0.5                                   # Crank-Nicolson
       u = theta * u_new.val + (1 - theta) * u_old.val
       g = theta * u_new.grad + (1 - theta) * u_old.grad
       return ((u_new.val - u_old.val) / dt * p.val
               + jnp.exp(m.val) * jnp.dot(g, p.grad))

   pde = hm.TimeDependentPDEVariationalProblem(
       [Vu, Vm, Vu], varf, bc, bc0, t_init=0.0, t_final=1.0, dt=0.01)

Why the previous level is a differentiation slot
------------------------------------------------

The adjoint of a time-stepping scheme needs
:math:`\partial r_{n+1}/\partial u_n`: the derivative of one step's residual with
respect to the *previous* state.  Treating the old level as a frozen coefficient
would force a finite difference there and make the gradient inexact.  Making it a
real slot keeps every block exact, and the stationary problem is then just the
three-slot case.

The per-step blocks are ``A``, ``B`` (the cross-time coupling), ``C``, and the
second-order blocks ``W_nn``, ``W_no``, ``W_oo``, ``W_nm``, ``W_om``, ``W_mm``.
``applyWuu`` accumulates into both time levels, which is what makes the
time-dependent reduced Hessian symmetric; the test suite checks that
``W_um`` and ``W_mu`` are exact transposes and that the Hessian is symmetric to
1e-15 on a quasilinear problem where all four cross-time blocks are nonzero.

Operators along the trajectory
------------------------------

Nothing is assembled, and no multigrid hierarchy set up, inside a solve loop.
The step operators of a trajectory (``A_n``, its transpose and ``B_n``) are built
once when the adjoint is first solved at that state and parameter, kept while the
point does not change, and completed with ``C_n`` and the ``W`` blocks by
``setLinearizationPoint``; every block of a step comes from one differentiation
pass.  The forward solve of a residual that is linear in the new level keeps
its Jacobian across calls at the same parameter.

The common case of a step Jacobian that does not change in time (a linear PDE
with coefficients independent of ``t`` and a fixed ``dt``) is detected rather
than assumed: the operators of the first two steps are compared through their
action on a random vector, and when they agree one matrix, one transpose and one
solver with its hierarchy serve the whole march.  Otherwise every step keeps its
own, cloned from the solver you set (``solver_fwd_inc``, ``solver_adj_inc``;
``solver_adj`` defaults to a clone of ``solver``).  On the 16-step
advection-diffusion test problem this takes a forward or adjoint solve from 16
BoomerAMG setups to one, and a Hessian action from 32 to none.  The comparison
tolerance is ``TimeDependentPDEVariationalProblem.INVARIANCE_PROBE_TOL``
(``1e-13``); a problem whose Jacobian changes only after the second step must set
it to ``-1`` to force per-step operators.

Scalar parameters
-----------------

``t`` and ``dt`` arrive as **traced** values rather than constants, so changing the
step size between steps reuses the compiled kernel instead of triggering a
recompile.  Declare them with ``nparams`` when building a kernel directly.

Time-dependent vectors
----------------------

:class:`~hippymfem.modeling.timeDependentVector.TimeDependentVector` holds a
snapshot per time level with the arithmetic the algorithms need, and
:class:`~hippymfem.modeling.misfit.MisfitTD` compares snapshots against data at
chosen times.

.. note::

   ``apply_ijk``, third derivatives, is not implemented for the time-dependent
   problem, because Newton-CG does not use it.  A time-dependent Taylor
   approximation of a QoI would need it.

   The time-dependent problem takes a **domain density only**: there is no
   ``bdr_varf`` or ``facet_varf`` argument, so a Neumann or Robin condition has to
   enter through the stationary problem class or be imposed on the space.

Concepts
========

The residual density
--------------------

Everything starts from one function.  Write the weak residual of the forward PDE
as an integrand evaluated at a single quadrature point:

.. code-block:: python

   def pde_varf(u, m, p, x):
       return jnp.exp(m.val) * hm.inner(u.grad, p.grad) - f(x) * p.val

The forward problem is :math:`\partial R/\partial p = 0`, where
:math:`R = \int_\Omega \text{density}\,dx`.  Reading the residual as the PDE part of
a Lagrangian is what makes the rest automatic.

``u``, ``m`` and ``p`` are the state, the parameter and the adjoint.  Each arrives
as a :class:`~hippymfem.fem.kernel.Field`, a small record holding the field's value
and gradient at that point:

.. list-table::
   :header-rows: 1
   :widths: 20 35 35

   * - attribute
     - scalar space (``vdim == 1``)
     - vector space (``vdim == d``)
   * - ``.val``
     - scalar
     - shape ``(d,)``
   * - ``.grad``
     - shape ``(sdim,)``
     - shape ``(d, sdim)``

``x`` is the physical position of the quadrature point, as an array of length
``sdim``.  For H(curl) and H(div) spaces the record carries ``.curl`` or ``.div``
instead of ``.grad``, because the gradient is not in the space; see
:doc:`spaces`.

The density must be traceable by JAX, which in practice means: use ``jax.numpy``
rather than ``numpy``, and do not branch on the *value* of a field (use
``jnp.where``).  Within those rules it is ordinary Python; see
:ref:`beyond-a-form-language`.

The blocks, and where they come from
------------------------------------

A form-language library builds the operators of the inverse problem by applying
``ufl.derivative`` to a UFL form.  Here they come from differentiating the
quadrature sum with JAX, vectorized over elements:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - block
     - what it is
   * - :math:`A = \partial_u \partial_p R`
     - forward Jacobian
   * - :math:`C = \partial_m \partial_p R`
     - parameter Jacobian
   * - :math:`W_{uu} = \partial_u\partial_u R`
     - second-order state block
   * - :math:`W_{um} = \partial_u\partial_m R`
     - mixed block
   * - :math:`W_{mm} = \partial_m\partial_m R`
     - second-order parameter block
   * - third derivatives
     - contracted with two directions, for Taylor expansions of a QoI

Because all of them are derivatives of *one* object evaluated on *one* quadrature
rule, they are mutually consistent by construction.  That is what makes the adjoint
gradient exact for the discretized problem rather than merely close to the
continuous one, and it is why ``modelVerify``'s finite-difference slopes come out at
1.000 rather than 0.98.

Variables and the state vector
------------------------------

A point in the problem is a list ``x = [u, m, p]`` indexed by the constants
:data:`~hippymfem.modeling.variables.STATE`,
:data:`~hippymfem.modeling.variables.PARAMETER` and
:data:`~hippymfem.modeling.variables.ADJOINT`.  Entries are
:class:`~hippymfem.common.parvector.ParVector` objects: a numpy array of this
rank's true dofs, aliased by an ``mfem.HypreParVector`` so MFEM and hypre can act
on the same memory without a copy.

.. code-block:: python

   x = model.generate_vector()             # [u, m, p], all zero
   x[PARAMETER] = prior.mean.copy()
   model.solveFwd(x[STATE], x)
   model.solveAdj(x[ADJOINT], x)
   cost, reg, misfit = model.cost(x)

The three objects of an inverse problem
---------------------------------------

:class:`~hippymfem.modeling.PDEVariationalProblem.PDEVariationalProblem`
    The forward problem, built from the residual density and the essential boundary
    conditions.  Supplies the solves and all the blocks above.

A prior
    :class:`~hippymfem.modeling.prior.BiLaplacianPrior` and relatives: a
    Matern-class Gaussian measure whose precision is a differential operator, so
    sampling and applying :math:`R^{-1}` are both linear solves rather than dense
    factorizations.  See :doc:`priors`.

A misfit
    What the data says, as a functional of the state:
    :class:`~hippymfem.modeling.misfit.DiscreteStateObservation` for pointwise
    data, :class:`~hippymfem.modeling.misfit.ContinuousStateObservation` for a
    weighted :math:`L^2` norm over a region or the boundary.  See
    :doc:`observations`.

:class:`~hippymfem.modeling.model.Model` ties the three together and is what the
optimizers, the Hessian and the samplers consume.

.. _beyond-a-form-language:

Densities a form language cannot express
----------------------------------------

Because the density is differentiated rather than compiled from a symbolic form,
its body may contain anything JAX can handle.  Two cases that are awkward or
impossible in UFL and ordinary here:

**A learned closure.**  A small neural network inside the residual, with its
weights as ordinary captured arrays:

.. code-block:: python

   def pde_varf(u, m, p, x):
       z = jnp.stack([u.val, m.val])
       h = jnp.tanh(W1 @ z + b1)
       kappa = jnp.exp(W2 @ h + b2)[0]          # a learned conductivity
       return kappa * hm.inner(u.grad, p.grad)

**An inner solve for a local constitutive law.**  A few Newton steps on a scalar
equation, differentiated through:

.. code-block:: python

   def pde_varf(u, m, p, x):
       def residual(s):                         # s solves a local nonlinear law
           return s ** 3 + s - jnp.exp(m.val) * u.val
       s = 0.0
       for _ in range(3):                       # unrolled Newton
           s = s - residual(s) / (3.0 * s ** 2 + 1.0)
       return s * hm.inner(u.grad, p.grad)

Both give exact gradients and Hessians: the finite-difference slopes are 1.000 and
1.001 respectively, measured.  The reduced Hessian is still assembled from exact
second derivatives, so Newton-CG converges at the same rate it does for a
textbook density.

What you do not write
---------------------

Not the adjoint equation, not the parameter Jacobian, not the second-order blocks,
not the boundary-condition bookkeeping for any of them, and not a separate
Gauss-Newton approximation, which is the same object with one term dropped, as
:class:`~hippymfem.modeling.reducedHessian.ReducedHessian` does with a flag.

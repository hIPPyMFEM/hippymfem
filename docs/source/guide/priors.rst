Priors
======

The prior is a Gaussian measure on the parameter field whose precision is a
differential operator, so sampling from it and applying :math:`R^{-1}` are linear
solves rather than dense factorizations, which is what makes the whole approach
work at mesh-independent cost.

.. code-block:: python

   prior = hm.BiLaplacianPrior(Vm, gamma=0.1, delta=0.5, robin_bc=True)

:math:`R = A M^{-1} A` with :math:`A = \gamma(-\nabla\cdot\Theta\nabla) + \delta I`,
giving a Matern-class covariance.  ``robin_bc=True`` adds the boundary term that
removes the spurious variance inflation a homogeneous Neumann condition produces
near the boundary.

===========================  ===============================================
parameter                    effect
===========================  ===============================================
``gamma``, ``delta``         pointwise variance scales as
                             :math:`1/(4\pi\gamma\delta)` in 2D; the
                             correlation length as :math:`\sqrt{\gamma/\delta}`
``Theta``                    an anisotropic tensor for the differential
                             operator, as a ``(d, d)`` array
``mean``                     the prior mean, a
                             :class:`~hippymfem.common.parvector.ParVector`
``robin_bc``                 the boundary term above
``solver_type``              ``"lu"`` or ``"krylov"`` for the internal solves
``lumped_mass``              the mass matrix :math:`M` by the rule at the
                             vertices: diagonal (below)
===========================  ===============================================

**A lumped mass.**  Every product with :math:`R`, which the cost, the gradient and each
Hessian action take, solves a system with the mass matrix; with the consistent one that
is some seventy CG iterations, 50 to 80 ms on a GPU at two million parameter unknowns,
as much as a solve with :math:`R` itself.  ``lumped_mass=True`` integrates the mass
matrix by the rule at the vertices of an element, which makes it diagonal (on affine
elements the sums of the rows of the consistent one): the lumped mass of the SPDE
approach to Matern fields (Lindgren, Rue and Lindstrom, 2011).  The square root for
sampling is by the same rule, so samples have the covariance :math:`R^{-1}` of that
precision exactly.  It is another discretization of the same prior, with another
matrix :math:`R`, which is why it is not the default; it is for a parameter space of
first order on a mesh of one kind of element.  On the MI250X cards of Frontier, with the
Newton benchmark at 1.8 million parameter unknowns a GPU, the operations of two
Newton-CG steps took 18.3 s instead of 19.4 s on one GPU and 23.9 s instead of 24.9 s on
64.

Use :func:`~hippymfem.modeling.prior.BiLaplacianComputeCoefficients` to go the other
way, from a desired pointwise standard deviation and correlation length to
``gamma`` and ``delta``.  Do this rather than guessing: a prior whose pointwise
standard deviation is 28 for a field of amplitude 1 gives a Hessian with condition
number :math:`5\times10^8`, and Newton-CG will not converge on it.

Sampling
--------

.. code-block:: python

   noise = prior.noise_vector()
   prior.sample_noise(1.0, noise)      # white noise, partition independent
   m = Vm.vector()
   prior.sample(noise, m)              # m ~ N(mean, R^{-1})

Sampling needs :math:`\sqrt{M}`, the square root of the mass matrix, which is built
in closed form at quadrature points rather than by a factorization: the element mass
matrix of a single quadrature point is rank one, so its square root is explicit, and
assembling those gives a matrix with :math:`\sqrt{M}\sqrt{M}^{\top} = M` exactly.
The test suite checks that identity to round-off rather than to a tolerance.

The white noise behind a sample is the same field on any number of ranks, so prior
samples agree across rank counts to the tolerance of the prior's solve; see
:doc:`parallel`.

Other priors
------------

.. list-table::
   :widths: 55 45

   * - :class:`~hippymfem.modeling.prior.LaplacianPrior`
     - :math:`R = A`, a rougher field
   * - :class:`~hippymfem.modeling.prior.BiLaplacianPrior`
     - Matern, the usual choice
   * - :class:`~hippymfem.modeling.prior.VectorBiLaplacianPrior`
     - a vector-valued parameter
   * - :class:`~hippymfem.modeling.prior.GaussianRealPrior`
     - a finite-dimensional Gaussian

``VectorBiLaplacianPrior`` applies one scalar ``gamma``/``delta`` pair to all
components, because MFEM's vector integrators take one scalar coefficient;
component-dependent values raise rather than silently averaging.

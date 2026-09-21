Boundary terms
==============

A second density, taking the outward unit normal, supplies what UFL writes with
``ds``: Neumann and Robin conditions, boundary sources, penalty terms, boundary
misfits and boundary quantities of interest.

.. code-block:: python

   def pde_varf(u, m, p, x):                      # the dx term
       return jnp.exp(m.val) * hm.inner(u.grad, p.grad) - f(x) * p.val

   def bdr_varf(u, m, p, x, n):                   # the ds term
       return (kappa * u.val - g(x)) * p.val      # Robin

   pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                  is_fwd_linear=True,
                                  bdr_varf=bdr_varf, bdr_attributes=[2, 4])

It is differentiated exactly like the domain density, so every block (``A``,
``C``, ``W_uu``, ``W_um``, ``W_mm``, third derivatives) picks up its boundary part
without being mentioned again.  The domain and boundary parts are summed **before**
essential-dof elimination; adding them afterwards would leave 2.0 on the essential
diagonal.

The full gradient is available on the boundary
----------------------------------------------

A boundary element's own finite element lives on a lower-dimensional reference
element, so it can only produce tangential derivatives.  To make ``u.grad``
meaningful on the boundary, each face quadrature point is mapped back into the
**adjacent volume element** with MFEM's ``FaceElementTransformations``; the shape
functions and gradients are evaluated there and the element dofs are the volume
element's.  A normal flux is then just

.. code-block:: python

   def flux(u, m, p, x, n):
       return jnp.exp(m.val) * jnp.dot(u.grad, n) * p.val

and it is correct: compared against MFEM's ``BoundaryNormalLFIntegrator`` on a field
whose gradient the space represents exactly, the difference is 2.5e-16.

The price is that the basis tables vary per boundary element instead of being shared
by a group.  Boundary elements are a lower-dimensional set, with ``nbe`` growing
like :math:`N^{(d-1)/d}`, so this costs little next to the domain tables.

Selecting part of the boundary
------------------------------

``bdr_attributes`` takes MFEM's 1-based attribute numbers, or ``"all"``.  The
measure is exact: summing a unit boundary density over attributes 1 to 4 of the unit
square gives 4.000000000000, and the four sides sum to the whole boundary to 1e-13.

Boundary functionals on their own
---------------------------------

The machinery is usable without a PDE problem, for a boundary quantity of interest
or a diagnostic:

.. code-block:: python

   from hippymfem.fem.boundary import BoundaryKernel, get_boundary_batches
   from hippymfem.fem.assemble import assemble_scalar

   bb = get_boundary_batches(mesh, qdeg, "all", comm, space=Vu)
   flux = BoundaryKernel(lambda u, m, p, x, n: jnp.exp(m.val) * jnp.dot(u.grad, n),
                         [Vu, Vm, Vu], bb)
   total = assemble_scalar(comm, flux.element_values(
       [Vu.local_values(u), Vm.local_values(m), Vu.local_values(Vu.vector())]))

Not supported
-------------

**Non-conforming boundary faces** are rejected rather than mis-assembled.  Interior
facet terms, UFL's ``dS``, are a separate density and have their own page:
:doc:`facets`.

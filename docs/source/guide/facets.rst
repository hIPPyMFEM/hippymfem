Interior facet terms and DG
===========================

A third density supplies what UFL writes with ``dS``: the interior-face integrals of a
discontinuous Galerkin form.  An interior face has two adjacent elements, so every field
arrives carrying both traces, and the form is written in their jump
:math:`[\![u]\!] = u^- - u^+` and average :math:`\{u\} = (u^- + u^+)/2`, with :math:`-`
the side the face normal points out of.

.. code-block:: python

   from hippymfem.fem.facets import avg, avg_grad, jump

   def pde_varf(u, m, p, x):                          # the dx term
       return jnp.exp(m.val) * hm.inner(u.grad, p.grad) - f(x) * p.val

   def facet_varf(u, m, p, x, n, h):                  # the dS term: SIPG
       k = jnp.exp(avg(m))
       return (-k * jnp.dot(avg_grad(u), n) * jump(p)
               - k * jump(u) * jnp.dot(avg_grad(p), n)
               + kappa * k * 0.5 * (1 / h[0] + 1 / h[1]) * jump(u) * jump(p))

   def bdr_varf(u, m, p, x, n, h):                    # the ds term: Dirichlet data
       k = jnp.exp(m.val)
       return (-k * jnp.dot(u.grad, n) * p.val - k * (u.val - g(x)) * jnp.dot(p.grad, n)
               + kappa * k * (1 / h) * (u.val - g(x)) * p.val)

   Vu = hm.FunctionSpace.L2(mesh, 2)                  # a discontinuous state
   pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, None, None,
                                  is_fwd_linear=True, facet_varf=facet_varf,
                                  bdr_varf=bdr_varf)

A DG problem has no essential dofs: the Dirichlet data is imposed weakly on boundary
faces, which is why ``bc`` and ``bc0`` are ``None`` above.  The facet density is
differentiated exactly like the other two, so the residual and every block (``A``,
``C``, ``W_uu``, ``W_um``, ``W_mm``, third derivatives) picks up its facet part without
being mentioned again.

What the density is given
-------------------------

``x`` is the physical coordinate on the face and ``n`` the unit normal out of the first
element.  ``h`` is the pair of face measures, each element's measure divided by the
face's, which is what an interior penalty is scaled by; MFEM's ``DGDiffusionIntegrator``
penalises with the average ``0.5 * (1 / h[0] + 1 / h[1])``, and any other convention
(the smaller of the two, say) is yours to write.  A boundary density is
offered the same measure as a scalar, as ``bdr_varf(u, m, p, x, n, h)``; one written
without it, ``bdr_varf(u, m, p, x, n)``, keeps working unchanged.

Each field is a ``FacetField`` with the two traces in ``.minus`` and ``.plus``, so
``jump``, ``avg``, ``jump_grad`` and ``avg_grad`` are the whole vocabulary:
``avg(m)`` is a value, ``avg_grad(u)`` a vector, and a field's own side is
``u.minus.val`` if a one-sided term is what you want (an upwind flux, say).

Faces shared between ranks
--------------------------

The second element of a face on a rank boundary lives on a neighbour.  Facet terms are
assembled the way MFEM assembles its own DG forms: over local dofs, with the
neighbour's columns carrying their global numbers, folded onto true dofs afterwards.
Each rank takes the rows of the element on its side of a shared face and the neighbour
takes the rest, so the face is counted once and no rank needs the other's matrix.  The
values of the far side arrive through MFEM's face-neighbour exchange, which the
assembly performs; nothing about the form changes with the rank count, and the
assembled matrix does not either.

Everything is conforming here: a non-conforming interior face (one with a hanging node)
is not supported by the facet kernels, which refuse such a mesh with an error rather than
assemble a face term inconsistent with the residual, and a variable-order space is
rejected, as one finite element per group is what the batched kernels assume.  Domain and
boundary densities on a non-conforming mesh are supported.

A worked example
----------------

``applications/dg/model_transport_dg.py`` infers a diffusivity field from a downstream
plume at a mesh Peclet number in the tens: an upwind flux on the interior faces, the
inflow data imposed weakly, and the diffusivity multiplying both the interior-face flux
and its penalty, so the facet term reaches every derivative block.  It reports the
tracer balance through the boundary, which an upwind DG solve satisfies to rounding
because the constant is in the test space.

What is checked
---------------

``hippymfem.test.test_facets``, on 1, 2 and 4 ranks:

* an interior-penalty matrix against MFEM's ``DGDiffusionIntegrator``, to
  :math:`4.5\times10^{-16}`, and the Nitsche boundary term against the same integrator
  on boundary faces;
* the facet residual against the matrix it differentiates to, which is the test of the
  scatter, since the two reach the shared faces differently;
* a rectangular facet block, a DG state against a continuous parameter, against a
  finite difference of the residual;
* a facet term whose test space is continuous, where the rows are folded onto true dofs
  rather than being them already;
* a DG Poisson problem assembled against MFEM's assembly of the same problem, to
  :math:`10^{-15}`, solved and converging at order :math:`p+1`;
* a DG inverse problem, where the finite-difference checks of the gradient and the
  Hessian are the test that the facet term reaches ``C``, ``W_uu``, ``W_um`` and
  ``W_mm``;
* interior-penalty stabilisation of a continuous problem, where the facet block meets
  essential conditions and the essential rows come out as the identity exactly.

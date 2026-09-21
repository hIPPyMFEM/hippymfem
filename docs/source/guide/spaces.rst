Finite element spaces
=====================

:class:`~hippymfem.fem.spaces.FunctionSpace` wraps
``mfem.ParFiniteElementSpace`` together with everything it points at.  That
ownership is not cosmetic: MFEM stores raw pointers, so a space built from a
temporary collection crashes the next time it is used.

.. code-block:: python

   Vh1  = hm.FunctionSpace.H1(mesh, order=2)             # continuous Lagrange
   Vl2  = hm.FunctionSpace.L2(mesh, order=1)             # discontinuous Lagrange
   Vvec = hm.FunctionSpace.H1(mesh, order=2, vdim=2)     # vector Lagrange
   Vnd  = hm.FunctionSpace.ND(mesh, order=1)             # Nedelec, H(curl)
   Vrt  = hm.FunctionSpace.RT(mesh, order=0)             # Raviart-Thomas, H(div)

What each family gives the density
----------------------------------

==========================  =====================  ==========================
space                       ``field.val``          derivative attribute
==========================  =====================  ==========================
H1/L2, ``vdim == 1``        scalar                 ``.grad``, shape ``(sdim,)``
H1/L2, ``vdim == d``        shape ``(d,)``         ``.grad``, ``(d, sdim)``
ND (H(curl))                shape ``(sdim,)``      ``.curl``: scalar in 2D,
                                                   ``(3,)`` in 3D
RT (H(div))                 shape ``(sdim,)``      ``.div``, scalar
==========================  =====================  ==========================

A vector-element field has no ``.grad`` attribute, deliberately: the full gradient
is not a member of the space, so offering one would mean silently returning
something that is not the derivative of the discrete field.

An essential condition on a vector space applies to every component when its value is a
single number or ``None``, and to one component when ``component=`` names it, which is how
a roller support (normal displacement fixed, tangential free) is written::

   hm.DirichletBC(Vvec, 0.0, bdr_attributes=[4])                   # clamped
   hm.DirichletBC(Vvec, 0.0, bdr_attributes=[1], component=1)      # roller

Observing a vector field is also a choice rather than a default:
:func:`~hippymfem.modeling.pointwiseObservation.assemblePointwiseObservation` needs
``component`` or a ``components`` weight per target, and
``assemblePointwiseLOSObservation`` is the line-of-sight spelling of the latter.

.. code-block:: python

   def curl_curl(u, m, p, x):                    # H(curl) state
       return jnp.exp(m.val) * u.curl * p.curl + jnp.dot(u.val, p.val)

   def darcy(u, m, p, x):                        # H(div) state
       return jnp.exp(-m.val) * jnp.dot(u.val, p.val) + u.div * p.div

Why vector elements need more than a sign
-----------------------------------------

Nedelec and Raviart-Thomas basis functions are Piola mapped, covariantly
(:math:`J^{-T} w`) for H(curl) and contravariantly (:math:`J w / \det J`) for
H(div), so unlike Lagrange shape functions they cannot be tabulated once for a
whole group of elements.  MFEM's ``CalcVShape`` applies the right map and is used
directly, so the convention is MFEM's rather than a reimplementation.

Building those per-element tables costs a measured 4 to 8.5 microseconds per
element-quadrature-point.  They are built once per space and cached, so this is a
setup cost and not a per-assembly one.

MFEM also encodes element orientation partly as a sign on the dof index and partly,
for H(curl) above lowest order on simplices, as a small dense
``DofTransformation`` per element.  hIPPyMFEM applies it in the three places MFEM
does and in the same order:

* gather: :math:`u_{\mathrm{ref}} = T^{-1}(\mathrm{signs} \cdot u)`
* element matrix: :math:`A = T_{\mathrm{test}}^{\top} A_{\mathrm{ref}} T_{\mathrm{trial}}`, then the signs
* element vector: :math:`b = T^{\top} b_{\mathrm{ref}}`, then the signs

Getting that order wrong does not crash; it produces a wrong matrix that is still
symmetric.  The element matrices are therefore checked against
``VectorFEMassIntegrator``, ``CurlCurlIntegrator`` and ``DivDivIntegrator`` on
triangles, quadrilaterals, tetrahedra and hexahedra, at orders that do and do not
need a transformation, and agree to 1.4e-15.

Quadrature
----------

The integration rule degree defaults to ``2 * max(order) + 2``.  It is deliberately
generous: the density may be non-polynomial (``exp(m)``, ``1/m``), so no exact rule
exists, and what matters for the inverse problem is not that each integral be
exact but that **every derivative block use the same rule**, which is automatic,
because they are all differentiated from one quadrature sum.

Pass ``quadrature_degree`` to
:class:`~hippymfem.modeling.PDEVariationalProblem.PDEVariationalProblem` to choose
it explicitly.  When comparing an assembled block against an MFEM integrator, set
the same rule on the MFEM side with ``integrator.SetIntRule(...)``, or the
comparison measures the difference between two quadrature choices rather than the
kernel.

Limitations
-----------

* Variable-order spaces are rejected with a clear error; the batched element
  kernels assume one finite element per group.
* Mixed/block spaces built from *different* families (Taylor-Hood, mixed Darcy) are not
  supported by
  :class:`~hippymfem.modeling.PDEVariationalProblem.PDEVariationalProblem`, which gives
  each of its three variables one space.  A vector-valued space of a single family is.
  The kernel itself is not restricted: a density written over four slots in two families
  yields all four Jacobian blocks, which glue into one operator with MFEM's
  ``HypreParMatrixFromBlocks``.  `Tutorial 11 <../tutorials/11_VectorAndMixedFields.html>`_
  solves mixed Darcy that way, and ``test_vectorfe.py`` does the same for poroelasticity.
* Non-conforming meshes work for domain integrals, whose reduction falls back to
  hypre's ``P^T A P`` because the prolongation is no longer boolean (see
  :doc:`parallel`).  Non-conforming *boundary* faces are rejected by the boundary
  kernels, and an interior facet density is refused on a non-conforming mesh
  altogether (:doc:`facets`).

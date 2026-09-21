Coming from hIPPYlib
====================

The inverse-problem classes keep the names and signatures of
`hIPPYlib <https://hippylib.github.io>`_, from which that layer is adapted, so a driver
script written for it usually ports by changing the imports and the space construction.
This page is the map.

**Two spellings.**  hIPPYlib's methods are camelCase (``solveFwd``,
``setLinearizationPoint``, ``applyWuu``); the library's own names are snake_case.
Every class of the programming model answers to both (``solve_fwd`` and
``solveFwd`` are the same method, and a subclass may override either), and the
module-level functions have both too (``model_verify`` / ``modelVerify``,
``double_pass_g`` / ``doublePassG``).  The snake_case spelling is the preferred
one in this documentation; the camelCase one is what a ported driver already
contains.  See :mod:`hippymfem.common.naming` for the list.

=====================================  =========================================
hIPPYlib / hIPPYlibx                   hIPPyMFEM
=====================================  =========================================
``pde_varf(u, m, p)`` in UFL           ``pde_varf(u, m, p, x)`` as a JAX density,
                                       fields carrying ``.val`` and ``.grad``
a ``ds`` term in the form              ``bdr_varf(u, m, p, x, n)``, passed as
                                       ``PDEVariationalProblem(..., bdr_varf=...)``
a ``dS`` term in the form              ``facet_varf(u, m, p, x, n, h)``, written
                                       in ``jump`` and ``avg``
``dolfinx.fem.functionspace``          ``hm.FunctionSpace.H1`` / ``.L2`` /
                                       ``.ND`` / ``.RT``
``fem.dirichletbc``                    ``hm.DirichletBC(space, value,
                                       bdr_attributes=...)``
``PDEVariationalProblem``              same name, same role
``BiLaplacianPrior``                   same name
``PointwiseStateObservation``          ``DiscreteStateObservation`` (and a
                                       ``PointwiseStateObservation`` wrapper)
``ContinuousStateObservation``         same name
``Model``, ``ReducedHessian``          same names
``modelVerify``                        same name; returns
                                       ``(eps, err_grad, err_H)``
``ReducedSpaceNewtonCG``               same name
``BFGS``, ``SteepestDescent``          same names
``doublePass``, ``doublePassG``        same names
``GaussianLRPosterior``                same name
``MCMC``, ``pCN``, ``gpCN``, ``MALA``  same names
``PETScKrylovSolver``                  ``hm.KrylovSolver`` (MFEM Krylov + hypre),
                                       or ``hm.PETScKrylovSolver``
``PETScLUSolver``                      ``hm.LUSolver``, exact on any rank count,
                                       or ``hm.PETScLUSolver``
``petsc4py.PETSc.Vec``                 ``hm.ParVector``
``amg_method()``                       accepted and ignored; hypre BoomerAMG is
                                       the only option
=====================================  =========================================

Differences that change results
-------------------------------

**The quadrature rule.**  hIPPYlib's rule is chosen by FFCx from the form's
estimated degree; here it is ``2*max(order) + 2`` by default.  For a non-polynomial
density neither is exact, so the two libraries agree only when given the same rule.
The cross-validation harness (:doc:`validation`) sets them equal, which is why it
reaches 1e-13 rather than 1e-6.

**The gradient norm.**  Both default to the discrete :math:`L^2` norm through the
mass matrix.  Selecting the prior-weighted norm changes the iteration history.

**Randomized eigensolver draws.**  The random matrices differ, so the computed
eigenvalues agree to the accuracy of the randomized method rather than to round-off.
For an exact comparison use the dense spectrum on a coarse mesh, which is what the
validation harness does.

Things hIPPyMFEM has that hIPPYlib does not
-------------------------------------------

* a residual density that may contain anything JAX can differentiate, such as a
  network, a table or an inner Newton solve, with exact derivatives;
* element kernels that run on a GPU without the residual being rewritten, and, with a
  CUDA or HIP build of PyMFEM, the matrices and the solves on the card as well, from
  the same script;
* direct-CSR assembly with a reused sparsity pattern, which is faster than assembling
  through MFEM's own integrator interface;
* MFEM's meshes, element families and I/O.

Things hIPPYlib has that hIPPyMFEM does not
-------------------------------------------

* a form language, and with it automatic quadrature degree estimation, and mixed or
  block spaces of different families in the problem class itself; here such a system is
  assembled from its blocks, which come from one density all the same (see
  `tutorial 11 <tutorials/11_VectorAndMixedFields.html>`_);
* compiled C element kernels, which are faster per element on a CPU;
* years of use by many people.

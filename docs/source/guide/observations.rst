Observations and misfits
========================

A misfit is a functional of the state with a gradient and a second derivative.  Two
are used in practice.

Pointwise data
--------------

.. code-block:: python

   targets = np.column_stack((xs, ys))               # (ntargets, sdim)
   B = hm.assemblePointwiseObservation(Vu, targets)
   d = B.createVecLeft()
   B.mult(utrue, d)
   B.perturb(d, noise_std)                           # noise keyed on target index
   misfit = hm.DiscreteStateObservation(B, d, noise_std ** 2)

``B`` is the interpolation operator: a sparse parallel matrix whose rows evaluate
the finite element field at the target points, built by MFEM's point locator so a
target is found on whichever rank owns its element.  Its adjoint is exact and the
test suite checks ``B`` against direct evaluation and ``B^T`` against ``B``.

``perturb`` keys the noise on the **target index** rather than on the row's local
position, so the synthetic data is the same on any number of ranks.

On a vector-valued state a point observation has to say *what* it reads: pass
``component=i`` for one component, or ``components``, a weight per component per
target, for a line-of-sight combination (``assemblePointwiseLOSObservation`` is that
spelling).  A scalar space needs neither.  Observing every component at a point means
repeating the target once per component, which is what
`tutorial 11 <../tutorials/11_VectorAndMixedFields.html>`_ does with displacement data.

Distributed data
----------------

.. code-block:: python

   misfit = hm.ContinuousStateObservation(Vu, noise_variance=sigma2)
   misfit.d = d

A weighted :math:`L^2` misfit :math:`\frac{1}{2\sigma^2}(u-d)^{\top}W(u-d)`.  Pass
``attributes`` to observe a subdomain, ``boundary=True`` to observe on the boundary,
or ``coefficient`` to weight it.  For H(curl)/H(div) states ``W`` is the vector
finite element mass matrix, not the scalar one.

Others
------

.. list-table::
   :widths: 60 40

   * - :func:`~hippymfem.modeling.misfit.PointwiseStateObservation`
     - convenience wrapper
   * - :class:`~hippymfem.modeling.misfit.MultDiscreteStateObservation`
     - several data sets
   * - :class:`~hippymfem.modeling.misfit.MultiStateMisfit`
     - independent experiments
   * - :class:`~hippymfem.modeling.misfit.MisfitTD`
     - time-dependent data

The noise covariance
--------------------

``noise_variance`` is :math:`\sigma^2`, not :math:`\sigma`.  It enters the cost, the
gradient and the Hessian, so getting it wrong rescales the posterior rather than
producing an error.  Set it from the noise actually added to the data, and for
synthetic studies add the noise with ``B.perturb`` so the two cannot disagree.

Configuration
=============

One rule: **the environment says where the library runs, before the import**, and
everything else is a setting you can read and change at runtime through one object.
On a CPU nothing needs to be set.  On GPUs:

.. code-block:: bash

   HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 \
       mpirun -n 4 tools/mpirun_pinned.sh python my_script.py

``HIPPYMFEM_DEVICE=gpu`` puts the element kernels on the card, and
``HIPPYMFEM_HYPRE_DEVICE=1`` (with a CUDA or HIP build of PyMFEM) puts MFEM and hypre there
too, configured when ``hippymfem`` is imported.  ``tools/mpirun_pinned.sh`` gives each
rank one card before CUDA initializes.  Every other setting below has a default chosen
from measurements, and most runs never touch one.

Set before the import
---------------------

These are read by XLA or MPI before the library can act.  ``hm.config`` shows them
read-only.

===============================  ====================================================
variable                         what it decides
===============================  ====================================================
``HIPPYMFEM_DEVICE``             where the element kernels run: unset, ``gpu``, or
                                 ``auto`` (a GPU when the process can see one)
``HIPPYMFEM_HYPRE_DEVICE``       MFEM and hypre on the device too (CUDA or HIP PyMFEM)
``HIPPYMFEM_GPU_MEM_FRACTION``   JAX's share of each card (the default follows the
                                 ranks per card and whether hypre shares it);
                                 :ref:`gpu-memory`
``HIPPYMFEM_PIN_GPU``            restrict each rank to one card before CUDA starts
                                 (on by default; needs ``hippymfem`` imported before
                                 ``mpi4py`` and MFEM, otherwise use the wrapper)
``HIPPYMFEM_PETSC``              import petsc4py first, for the PETSc solvers
                                 (:doc:`solvers`)
``HIPPYMFEM_AUTO_DEVICE``        ``0``: do not configure MFEM's device at import; call
                                 :func:`~hippymfem.common.mfemconfig.configure_device`
                                 yourself
``HIPPYMFEM_NODE_GPUS``          the node's card count; written by
                                 ``tools/mpirun_pinned.sh``, not by you
===============================  ====================================================

Every other setting, one object
-------------------------------

.. code-block:: python

   import hippymfem as hm

   hm.config.hessian                 # "element": how element Hessians are differentiated
   hm.config.hessian = "quadrature"  # the same as HIPPYMFEM_HESSIAN=quadrature
   hm.config.show()                  # every setting, its value and source: put it in a bug report

:data:`hippymfem.config` names each setting as an attribute, with a one-line description
and the environment variable that seeds it (``HIPPYMFEM_`` and the name in capitals,
except where the table says otherwise).  Assigning an attribute takes effect as the
environment variable would have, and a value the setting does not know is refused, in
the environment as well.  Neither ``import hippymfem`` nor
``hm.config.as_dict(load=False)`` loads JAX; reading a kernel setting imports the kernel
module, as using it would.

The settings a run may want to change:

==========================  ============  ==================================================
``hm.config``               default       what it decides
==========================  ============  ==================================================
``hessian``                 ``element``   how element Hessian blocks are differentiated:
                                          ``quadrature`` is faster on a GPU from P2 up,
                                          ``auto`` times both (:ref:`smallest-kernel`)
``precision``               ``fp64``      precision of the element kernels: ``mixed`` puts
                                          the element matrices in single precision and
                                          refines the solves (:ref:`single-precision`);
                                          ``fp32``, everything single, is for experiments
``hypre_single``            empty         path of a single-precision build of hypre: the
                                          Jacobian of a PDE problem and the CG solves with
                                          it then run in that library
                                          (:ref:`single-precision`)
``gpu_deterministic``       off           device scatter in a fixed order: bit-identical
                                          repeats, more working memory
``element_chunk``           0             elements per kernel launch; 0 plans the chunk from
                                          the free memory (:ref:`gpu-memory`)
``gpu_mem_reserve``         0             GiB of card memory the chunk planner leaves to
                                          whatever else shares the card
``host_mem_fraction``       0.5           share of the node's available memory, split
                                          between its ranks, for host element batches
``hypre_spmv``              ``auto``      hypre's matrix-vector kernel on a GPU: its own on
                                          several ranks of a CUDA build, the vendor's on
                                          one (:ref:`several-gpus`)
``hypre_pool``              ``auto``      megabytes of freed device memory a recycling pool
                                          keeps for hypre; ``0`` for no pool.
                                          ``hypre_pool_keep`` (512) is what the default
                                          pool keeps between BoomerAMG setups
``device_vectors``          ``auto``      vectors that hypre has used are updated on the
                                          device; ``0`` restores numpy on the host
``pattern_threads``         0             threads of a sparsity-pattern build; 0 takes this
                                          rank's share of the node's cores
==========================  ============  ==================================================

The remaining ones select a reference implementation that the tests compare the default
against, or tune a threshold.  Their defaults are the measured best, and they are listed
for completeness:

==========================  ==============  ================================================
``hm.config``               default         what it decides
==========================  ==============  ================================================
``parmat``                  ``auto``        how the parallel matrix is built: straight into
                                            true-dof rows where the prolongation is boolean
                                            (``auto``, ``tdof``), or by the triple product
                                            with the prolongation for every space (``mfem``)
``parmat_device``           ``block``       with hypre on a device: hand hypre its two
                                            blocks, or ``copy`` one row-major CSR
``tdof_identity``           ``auto``        the true-dof route also on one rank (with hypre
                                            on a device)
``fold_elimination``        on              essential dofs eliminated in the scatter rather
                                            than by MFEM's calls afterwards
``share_hessian``           on              one differentiation pass for all blocks of a
                                            linearization point
``device_bridge``           ``auto``        with the kernels and hypre on one GPU, values
                                            cross between JAX's memory and MFEM's on the
                                            device instead of through the host
``device_pattern``          off             keep a pattern's index arrays on the device
``keep_geometric_factors``  off             keep MFEM's ``GeometricFactors`` alive after the
                                            element batches are built
``host_batch``              2048            elements per step of an element kernel on the
                                            host; 0 maps the whole batch at once
``ad_working_set``          16              doubles per tangent and quadrature point the
                                            chunk planner assumes
``geometry_stream``         0.25            share of the device budget above which element
                                            geometry stays on the host and is streamed
``geometry_slice``          0               elements per slice when the quadrature geometry
                                            is built here; a positive value forces that path
``fused_keep``              on              keep a large assembly accumulator between
                                            assemblies (above ``fused_keep_share``, 0.25 of
                                            JAX's arena)
``pattern_builder``         ``auto``        sparsity patterns without a global sort when
                                            numba is importable and the pattern has
                                            ``pattern_builder_min`` (2\ :sup:`24`) entries;
                                            ``numba`` always, ``sort`` never
``pattern_sort``            ``auto``        where the sort route sorts: ``host`` or
                                            ``device`` (from ``pattern_sort_min``,
                                            2\ :sup:`22` keys, in chunks of
                                            ``pattern_sort_chunk``, with the kernel
                                            ``pattern_sort_kernel``: CuPy's or XLA's)
``pattern_timing``          off             print what each phase of a pattern build took
``hypre_pcg``               on              CG with a hypre preconditioner runs in hypre's
                                            own PCG, the same iteration with one product
                                            fewer per V-cycle; off: MFEM's CG
``amg_relax``               -1              default BoomerAMG relaxation type of new
                                            solvers; -1 keeps MFEM's
``amg_max_levels``          -1              default BoomerAMG level cap; -1 keeps hypre's
``pc_reuse``                0               reuse a solver's AMG hierarchy across operators
==========================  ==============  ================================================

Six switches of the device assembly and of the single-precision solves are read from the
environment alone, when their module is imported.  Each is on unless set to ``0``, except
the BoomerAMG options, which are a string and empty by default:

==================================  ======================================================
variable                            what it decides
==================================  ======================================================
``HIPPYMFEM_COMPACT_PATTERN``       the scatter map of a fused assembly in its compact
                                    form, a third of the four-byte map
                                    (:ref:`gpu-memory`)
``HIPPYMFEM_PINNED_STREAM``         streamed element geometry goes to the card from
                                    pinned host memory instead of through JAX's copy
                                    (:ref:`gpu-memory`)
``HIPPYMFEM_SINGLE_ACCUMULATE``     a matrix for the single-precision hypre is
                                    accumulated in single precision; ``0`` accumulates
                                    in double and rounds once
``HIPPYMFEM_SINGLE_SHARE_COLUMNS``  single-precision matrices of one pattern share one
                                    copy of its column indices on the device
``HIPPYMFEM_SINGLE_POOL``           the single-precision hypre allocates device memory
                                    from the recycling pool of the double-precision one
``HIPPYMFEM_SINGLE_AMG``            BoomerAMG options of the single-precision solves by
                                    name, such as ``relax=7,pmax=6``
                                    (:ref:`single-precision`)
==================================  ======================================================

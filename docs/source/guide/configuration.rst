Configuration
=============

One rule: **the environment says where the library runs, before the import**, and
everything else is a setting you can read and change at runtime through one object.

.. code-block:: bash

   HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 \
       mpirun -n 4 tools/mpirun_pinned.sh python my_script.py

``HIPPYMFEM_DEVICE=gpu`` puts the element kernels on the card, and
``HIPPYMFEM_HYPRE_DEVICE=1`` (with a CUDA or HIP build of PyMFEM) puts MFEM and hypre there
too, configured when ``hippymfem`` is imported.  ``tools/mpirun_pinned.sh`` gives each
rank one card before CUDA initializes; it is the documented mechanism, and the
import-time pinning inside the library is a best-effort fallback that says what it
skipped (``hippymfem._jaxconfig.PIN_SKIPPED``).  A script that imports ``hippymfem``
before ``mpi4py``, MFEM and JAX needs nothing else.

Every knob, one object
----------------------

.. code-block:: python

   import hippymfem as hm

   hm.config.parmat_device           # "block": how the device matrix is built
   hm.config.triple = "split"        # the same as hm.fem.parmat.set_triple_mode("split")
   hm.config.show()                  # every knob, its value and source: put it in a bug report

:data:`hippymfem.config` names each of the library's settings as a typed attribute with
a one-line description and the environment variable that seeds it.  Assigning an
attribute goes through the owning module's own setter, so it takes effect exactly as
the environment variable would have.  Seven settings are read by XLA or MPI before the
library can act and are therefore **import-time only**; they appear read-only, and
assigning them says which variable to set:

===============================  ====================================================
variable                         what it decides
===============================  ====================================================
``HIPPYMFEM_DEVICE``             where the element kernels run (unset or ``gpu``)
``HIPPYMFEM_HYPRE_DEVICE``       MFEM and hypre on the device too (CUDA or HIP PyMFEM)
``HIPPYMFEM_AUTO_DEVICE``        configure MFEM's device at import (``0`` to opt out)
``HIPPYMFEM_GPU_MEM_FRACTION``   JAX's share of each card (the default follows the
                                 ranks per card and whether hypre shares it)
``HIPPYMFEM_PIN_GPU``            restrict each rank to one card before CUDA starts
``HIPPYMFEM_NODE_GPUS``          the node's card count when the launcher hid the rest
``HIPPYMFEM_PETSC``              import petsc4py first, for the PETSc solvers
===============================  ====================================================

The runtime settings are listed by ``hm.config.show()``; the assembly route, the
parallel-matrix constructors, the elimination, the kernel precision and chunking, the
pattern sort, and the solver defaults are among them.  Neither ``import hippymfem`` nor
``hm.config.as_dict(load=False)`` loads JAX; reading a kernel setting imports the
kernel module, as using it would.

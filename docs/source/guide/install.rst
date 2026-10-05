Installation
============

Requirements
------------

==================  ========================================================
Python              3.10 or later (tested with 3.12)
PyMFEM              4.8 or 4.9, built **with MPI** (``mfem.par``).  A CUDA build
                    (``tools/build_pymfem_cuda.sh``) or a HIP one
                    (``tools/build_pymfem_hip.sh``) is optional and moves MFEM and
                    hypre to the card; the element kernels need neither.  See
                    :doc:`gpu`.
mpi4py              built against the MPI that PyMFEM links
JAX                 0.4.30 or later (tested with 0.5 and 0.11), for the
                    residual densities and their derivatives
NumPy, SciPy        NumPy 1.26 or 2.x; SciPy supplies the sparse direct solves
Matplotlib          optional: plotting (``hippymfem.nb``) and the tutorials
petsc4py            optional; adds distributed direct solvers, see
                    :doc:`solvers`; ``tools/install_petsc_mumps.sh`` builds a
                    PETSc with MUMPS for it
numba               optional; builds large sparsity patterns without a global
                    sort, several times faster (:doc:`gpu`)
CuPy                optional; a faster device sort where the sort route runs
hypre, single       optional; a single-precision build of the hypre that PyMFEM
precision           uses (``tools/build_hypre_single.sh``, three minutes), loaded next
                    to it for the linear solves of a PDE problem
                    (`A single-precision hypre`_; :ref:`single-precision` in
                    the GPU guide)
==================  ========================================================

Everything except PyMFEM is a ``pip install``.  The ``mfem`` wheel on PyPI is serial
only, so PyMFEM is built from source.

Step by step
------------

.. code-block:: bash

   sudo apt-get install libopenmpi-dev openmpi-bin      # or load your cluster's MPI
   python -m pip install "numpy>=2" scipy
   python -m pip install --no-binary mpi4py mpi4py      # tied to that MPI
   tools/install_pymfem_parallel.sh                     # PyMFEM 4.8 with MPI
   python -m pip install -e ".[all]"                    # from the source root

``tools/install_pymfem_parallel.sh`` downloads the PyMFEM 4.8.0.1 source distribution,
applies the two patches a current toolchain needs, pins SWIG to 4.3.1, and lets PyMFEM
download and compile MFEM, hypre and METIS; it takes 20 to 60 minutes and finishes by
importing ``mfem.par``.  Given a directory (``tools/install_pymfem_parallel.sh
wheelhouse``) it builds a wheel there once and installs from it afterwards.

The extras of the ``pip install`` are ``ad`` (JAX), ``viz`` (Matplotlib), ``tutorial``
(JAX, Matplotlib, Jupyter), ``test``, ``docs``, and ``all``.  Putting the source root on
``PYTHONPATH`` works instead of installing.

Check the installation with

.. code-block:: bash

   python -c "import hippymfem as hm; print(hm.__version__, hm.mfem_config()['version'])"

``hm.mfem_config()`` reports how PyMFEM was built (MPI, CUDA, which direct solvers) by
reading its configuration header.  Do not probe this by configuring an MFEM ``Device``:
on a build without the requested backend MFEM responds with ``MFEM_ABORT``, which ends
the MPI job instead of returning false.

.. _hypre-single-install:

A single-precision hypre
------------------------

Optional.  The linear solves of a PDE problem whose Jacobian is symmetric and solved by
CG with BoomerAMG can run in a single-precision build of the hypre that PyMFEM uses,
loaded next to it in the same process; :ref:`single-precision` in the GPU guide says
what that gives.  ``tools/build_hypre_single.sh`` configures the same hypre source again
with the options of PyMFEM's build of it (compilers and flags, MPI, CUDA architecture,
hypre's own options) and single precision, so it needs ``cmake`` and the compiler, MPI
and CUDA toolkit of that build:

.. code-block:: bash

   module load gcc/12.3.0 openmpi/4.1.8        # the compiler and MPI PyMFEM was built with
   tools/build_hypre_single.sh <prefix>/PyMFEM /path/to/hypre_single
   export HIPPYMFEM_HYPRE_SINGLE=/path/to/hypre_single/libHYPRE_single.so

The first argument is a PyMFEM source tree in which PyMFEM's build compiled hypre with
CMake: the script reads ``external/hypre/src/cmbuild/CMakeCache.txt`` there.
``tools/build_pymfem_cuda.sh`` keeps that tree under its prefix (``<prefix>/PyMFEM``).
``tools/install_pymfem_parallel.sh`` builds in a temporary directory and removes it, so
for a host build PyMFEM has to be built from a checkout that is kept.  The HIP build of
``tools/build_pymfem_hip.sh`` configures its hypre without CMake and is not supported.
``MODULES="gcc/12.3.0 openmpi/4.1.8"`` makes the script load the modules itself, and
``JOBS`` (default 12) sets the parallel build, which takes about three minutes with
twelve for a CUDA build and half a minute for a host build.  The script stops if the new library links another MPI than the installed one,
since it would not load next to it.  Into the output directory it writes
``libHYPRE_single.so``; ``libHYPRE_single.json``, the offsets of the few fields of
hypre's structures that the library reads, taken from this build's own headers (keep it
next to the ``.so``); and the build itself (``build/``, ``initial_cache.cmake``,
``hypre_offsets``, ``configure.log``, ``build.log``).
It has been built and used with CUDA builds for H100, L40S and RTX PRO 6000 Blackwell
cards and with a host build.

Set the variable before ``import hippymfem``, or set ``hm.config.hypre_single`` before
the problem is built.  To see whether the library loads, run in the environment of the
runs (on a GPU build with ``HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1``, since the
library is set up for the device MFEM uses):

.. code-block:: bash

   python -c "
   import hippymfem
   from hippymfem.algorithms import singlesolve
   lib = singlesolve.library()
   print('loaded', lib.path) if lib else print('not loaded:', singlesolve.why_not() or 'HIPPYMFEM_HYPRE_SINGLE is not set')"

A library that is named but cannot be used (no ``libHYPRE_single.json`` next to it, a
build that is not single precision, no device bridge on a GPU) gives a
``RuntimeWarning`` at its first use, and the solves stay in double precision.  With the
variable set, ``test_solvers`` and ``test_device`` also compare the solves in the
single-precision library with the double-precision ones (`Running the tests`_).

Docker
------

.. code-block:: bash

   docker build -t hippymfem .
   docker run --rm -p 8888:8888 hippymfem           # JupyterLab on the tutorials
   docker run --rm hippymfem ./run_tests.sh 1 2     # the test suites

The image is Ubuntu 24.04 with OpenMPI, PyMFEM built from source, JAX on the CPU and
JupyterLab.

Running the tests
-----------------

.. code-block:: bash

   ./run_tests.sh            # one rank
   ./run_tests.sh 1 2 4      # and on two and four ranks
   python -m pytest          # the same suites through pytest, on one and two ranks

Each suite is checked against something external rather than against itself: MFEM's own
integrators, exact Gaussian posteriors, dense eigendecompositions, finite differences.
The GPU and PETSc suites skip with a printed reason when those are unavailable; they
never pass silently.  ``HIPPYMFEM_PRECISION=mixed ./run_tests.sh`` runs every suite with
single-precision element matrices, and with ``HIPPYMFEM_HYPRE_SINGLE`` set
``test_solvers`` and ``test_device`` check the solves in the single-precision hypre
against the double-precision ones.

Cross-validation against hIPPYlibx
----------------------------------

.. code-block:: bash

   # in the FEniCSx environment (dolfinx 0.10):
   python -m pip install git+https://github.com/hIPPyMFEM/hippylibx.git
   # then, from the hIPPyMFEM source root:
   MFEM_PY=python FENICSX_PY=/path/to/dolfinx/python ./validation/run_validation.sh 12 1

`hIPPYlibx <https://github.com/hIPPyMFEM/hippylibx>`_ is hIPPYlib on FEniCSx.  The two
libraries usually live in different environments, so each driver runs under its own
interpreter on the same discrete problem: same mesh arrays, same quadrature degree,
same observation points, same noise realization read from a shared file.  See
:doc:`../validation`.

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
                    :doc:`solvers`
numba               optional; builds large sparsity patterns without a global
                    sort, several times faster (:doc:`gpu`)
CuPy                optional; a faster device sort where the sort route runs
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
never pass silently.

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

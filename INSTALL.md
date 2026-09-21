# Installing hIPPyMFEM

hIPPyMFEM is a Python package on top of [MFEM](https://mfem.org). It needs:

| package | version | used for |
|---|---|---|
| Python | 3.10 or later (tested with 3.12) | |
| [PyMFEM](https://github.com/mfem/PyMFEM) | 4.8 or 4.9, **built with MPI** (`mfem.par`) | meshes, spaces, assembly, hypre |
| an MPI implementation and [mpi4py](https://mpi4py.readthedocs.io) | mpi4py built against the MPI that PyMFEM links | parallel runs |
| [JAX](https://docs.jax.dev) | 0.4.30 or later (tested with 0.5 and 0.11) | residual densities and their derivatives |
| NumPy, SciPy | NumPy 1.26 or 2.x | |
| Matplotlib | optional | plotting (`hippymfem.nb`) and the tutorials |
| petsc4py | optional | distributed direct solvers |
| numba | optional (`pip install -e ".[perf]"`) | builds large sparsity patterns without a global sort, several times faster |

Everything except PyMFEM is a `pip install`. The `mfem` wheel on PyPI is serial only, and
hIPPyMFEM needs the parallel wrappers, so PyMFEM is built from source.

## 1. MPI and mpi4py

On Ubuntu or Debian:

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
python -m pip install --no-binary mpi4py mpi4py
```

Building mpi4py from source ties it to the MPI on your `PATH`, which is the one PyMFEM will
be built against. On a cluster, load the MPI module first.

## 2. PyMFEM with MPI

```bash
python -m pip install "numpy>=2" scipy
tools/install_pymfem_parallel.sh
```

The script downloads the PyMFEM 4.8.0.1 source distribution, applies the two patches a
current toolchain needs (`tools/patch_pymfem.py`), pins SWIG to 4.3.1, and lets PyMFEM's
build download and compile MFEM, hypre and METIS. Expect 20 to 60 minutes. It finishes by
importing `mfem.par`, which fails for a serial build.

To build once and reuse the result, give the script a directory:
`tools/install_pymfem_parallel.sh wheelhouse` builds a wheel there and installs it, and
later runs install that wheel directly.

## 3. hIPPyMFEM

From the root of the source tree:

```bash
python -m pip install -e ".[all]"
```

`all` adds JAX (CPU), Matplotlib, JupyterLab and pytest. The smaller extras are `ad`
(JAX), `viz` (Matplotlib), `tutorial` (JAX, Matplotlib and Jupyter), `perf` (numba), `test`
and `docs`.
Putting the source root on `PYTHONPATH` instead of installing works too, and the tutorials
do that for themselves.

Check the installation:

```bash
python -c "import hippymfem as hm; print(hm.__version__, hm.mfem_config()['version'])"
./run_tests.sh            # every suite on one rank, a few minutes
./run_tests.sh 1 2 4      # and on two and four ranks
python -m pytest          # every suite on one and two ranks, through pytest
```

## Docker

The `Dockerfile` builds all of the above on Ubuntu 24.04: OpenMPI, PyMFEM from source, JAX
on the CPU, and JupyterLab.

```bash
docker build -t hippymfem .
docker run --rm -p 8888:8888 hippymfem              # JupyterLab on the tutorials
docker run --rm hippymfem ./run_tests.sh 1 2        # the test suites
```

Building PyMFEM takes most of the build time, and Docker caches that stage, so rebuilding
after a change to hIPPyMFEM is quick.

## GPUs

Two parts of hIPPyMFEM can run on a GPU, independently of each other:

- **the element kernels**, through JAX: install JAX with CUDA (`pip install "jax[cuda12]"`)
  and set `HIPPYMFEM_DEVICE=gpu` before importing hippymfem;
- **the linear solves**, through MFEM and hypre: this needs PyMFEM built with CUDA, which
  `tools/build_pymfem_cuda.sh` does, and `HIPPYMFEM_HYPRE_DEVICE=1`.

On AMD cards both parts work the same way: `pip install "jax[rocm]"` for the kernels
(the accelerator is chosen from what the node carries, so nothing in a script changes),
and `CPU_PYMFEM=<a CPU PyMFEM tree> tools/build_pymfem_hip.sh <prefix> <arch>` for the
solves, which builds hypre, MFEM and PyMFEM's wrappers for HIP because PyMFEM's own
build system has no HIP option. The GPU guide's "AMD cards" section has the versions
that work and the patches it applies.

Launch multi-GPU runs through `tools/mpirun_pinned.sh`, which gives every rank its own
card. The GPU guide (`docs/source/guide/gpu.rst`) has the details and the measurements, and
`run_tests.sh` runs the GPU suites where a GPU is visible.

## Documentation

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

The build does not need PyMFEM: missing heavy dependencies are mocked, and the tutorials are
rendered with the outputs stored in the notebooks.

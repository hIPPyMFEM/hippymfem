# hIPPyMFEM in a container: PyMFEM with MPI, JAX on the CPU, Jupyter for the tutorials.
#
#   docker build -t hippymfem .
#   docker run --rm -p 8888:8888 hippymfem                 # the tutorials in JupyterLab
#   docker run --rm hippymfem ./run_tests.sh 1 2           # the test suites on 1 and 2 ranks
#
# PyMFEM is compiled from source (the PyPI wheel is serial only), which takes most
# of an hour; that stage does not depend on the hIPPyMFEM sources, so Docker caches
# it across rebuilds.  `docker build --target env .` stops after the environment.

FROM ubuntu:24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git \
        python3 python3-dev python3-venv \
        libopenmpi-dev openmpi-bin \
 && rm -rf /var/lib/apt/lists/*
# OpenMPI refuses to run as root, or with more ranks than cores, without these.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "numpy>=2" scipy \
 && pip install --no-cache-dir --no-binary mpi4py mpi4py

# Build PyMFEM into a wheel ...
FROM base AS pymfem
COPY tools/install_pymfem_parallel.sh tools/patch_pymfem.py /tmp/
RUN bash /tmp/install_pymfem_parallel.sh /wheelhouse

# ... and install it in a stage that has only the wheel, so the image does not
# carry the build tree and a wheel that is not self-contained fails here.
FROM base AS env
COPY --from=pymfem /wheelhouse /wheelhouse
RUN pip install --no-cache-dir /wheelhouse/mfem-*.whl \
 && pip install --no-cache-dir "jax>=0.4.30" matplotlib pytest \
        jupyterlab nbconvert nbformat ipykernel \
 && python -c "import mfem.par, jax; print('mfem.par and jax', jax.__version__)"

FROM env
COPY . /opt/hippymfem
WORKDIR /opt/hippymfem
RUN pip install --no-cache-dir -e ".[all]"
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--notebook-dir=/opt/hippymfem/tutorial"]

#!/bin/bash
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see LICENSE and COPYRIGHT.
#
# Build PETSc with MUMPS (a distributed sparse direct solver) and petsc4py against
# it, so that ``hm.PETScLUSolver`` factorizes in parallel instead of falling back to
# PETSc's ``redundant`` serial LU.
#
# The petsc4py on PyPI configures PETSc without MUMPS, and a conda-forge PETSc
# brings its own MPI, which cannot share a process with the MPI PyMFEM and mpi4py
# link.  So PETSc is built here from its source distribution against the system
# MPI (the same mpicc, mpicxx and mpifort PyMFEM used), with MUMPS, ScaLAPACK,
# METIS and ParMETIS downloaded and built by PETSc's configure.  A Fortran
# compiler is required (MUMPS and ScaLAPACK are Fortran).  Expect 10 to 20
# minutes on a workstation.
#
#   tools/install_petsc_mumps.sh                 # into $PETSC_PREFIX, petsc4py into $PYTHON
#   PETSC_PREFIX=/opt/petsc-mumps PYTHON=venv/bin/python tools/install_petsc_mumps.sh
#
# Environment: PYTHON (default python3; a venv made with --system-site-packages on
# top of the interpreter that has PyMFEM keeps the PyPI petsc4py untouched),
# PETSC_VERSION (default 3.23.5, the petsc4py version it must match),
# PETSC_PREFIX (default $HOME/petsc-mumps), JOBS (default 16).
#
# Afterwards: set HIPPYMFEM_PETSC=1 (petsc4py must be imported before PyMFEM) and
# ``hm.PETScLUSolver(comm).package_used`` reads "mumps".
set -euo pipefail
PYTHON="${PYTHON:-python3}"
VERSION="${PETSC_VERSION:-3.23.5}"
PREFIX="${PETSC_PREFIX:-$HOME/petsc-mumps}"
JOBS="${JOBS:-16}"

for tool in mpicc mpicxx mpifort; do
  command -v "$tool" >/dev/null || {
    echo "$tool not found: install an MPI implementation with Fortran bindings first" >&2
    exit 1
  }
done
"$PYTHON" -c "import mpi4py" 2>/dev/null || {
  echo "mpi4py is not importable from $PYTHON" >&2
  exit 1
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
# the source distribution behind the petsc package on PyPI; the same tarball as
# https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-$VERSION.tar.gz
"$PYTHON" -m pip download "petsc==$VERSION" --no-binary :all: --no-deps --no-build-isolation -d .
tar xzf "petsc-$VERSION.tar.gz"
cd "petsc-$VERSION"
export PETSC_DIR="$PWD" PETSC_ARCH=arch-mumps-opt
./configure --prefix="$PREFIX" --with-shared-libraries=1 --with-c2html=0 --with-debugging=0 \
  --with-cc="$(command -v mpicc)" --with-cxx="$(command -v mpicxx)" --with-fc="$(command -v mpifort)" \
  --download-scalapack --download-mumps --download-metis --download-parmetis \
  COPTFLAGS="-O2" CXXOPTFLAGS="-O2" FOPTFLAGS="-O2"
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all -j "$JOBS"
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" install

# petsc4py of the same version, built against this PETSc (Cython is its build dependency)
# petsc4py 3.23 builds with neither Cython 3.1+ nor setuptools 80+ (distutils' execute()
# lost its dry_run argument); both pins are for its build only
"$PYTHON" -m pip install "cython<3.1" "setuptools<80"
PETSC_DIR="$PREFIX" PETSC_ARCH= "$PYTHON" -m pip install --ignore-installed --no-binary :all: --no-deps --no-build-isolation "petsc4py==$VERSION"
"$PYTHON" - <<'PY'
from petsc4py import PETSc
print("petsc4py", PETSc.Sys.getVersion(), "mumps:", PETSc.Sys.hasExternalPackage("mumps"))
PY
echo "done: set HIPPYMFEM_PETSC=1 and use hm.PETScLUSolver(comm)"

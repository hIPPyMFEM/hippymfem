#!/bin/bash
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see LICENSE and COPYRIGHT.
#
# Build and install PyMFEM with MPI support (the ``mfem.par`` wrappers).
#
# hIPPyMFEM needs the parallel wrappers and the mfem wheel on PyPI is serial only,
# so this builds PyMFEM from its source distribution.  PyMFEM's build system
# downloads and compiles MFEM, hypre and METIS itself; what it needs from the
# system is a C++ compiler, an MPI implementation with mpicc and mpicxx on the
# PATH, and mpi4py built against that MPI.  Expect 20 to 60 minutes.
#
#   tools/install_pymfem_parallel.sh               # build and install
#   tools/install_pymfem_parallel.sh wheelhouse    # build a wheel into wheelhouse/
#                                                  # once, install from it afterwards
#
# Environment: PYTHON (default python3), PYMFEM_VERSION (default 4.8.0.1).
set -euo pipefail
PYTHON="${PYTHON:-python3}"
VERSION="${PYMFEM_VERSION:-4.8.0.1}"
WHEELHOUSE="${1:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for tool in mpicc mpicxx; do
  command -v "$tool" >/dev/null || {
    echo "$tool not found: install an MPI implementation (e.g. libopenmpi-dev) first" >&2
    exit 1
  }
done
"$PYTHON" -c "import mpi4py" 2>/dev/null || {
  echo "mpi4py is not importable: pip install --no-binary mpi4py mpi4py first" >&2
  exit 1
}

if [ -n "$WHEELHOUSE" ] && ls "$WHEELHOUSE"/mfem-"$VERSION"-*.whl >/dev/null 2>&1; then
  "$PYTHON" -m pip install "$WHEELHOUSE"/mfem-"$VERSION"-*.whl
else
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  curl -fsSL "https://files.pythonhosted.org/packages/source/m/mfem/mfem-$VERSION.tar.gz" \
    | tar -xz -C "$WORK"
  SRC="$WORK/mfem-$VERSION"

  # Two upstream problems stop the release from compiling on a current toolchain:
  # the SWIG wrappers need -fpermissive, and their setup.py imports distutils,
  # which Python 3.12 removed.  patch_pymfem.py fixes both (and skips MFEM's
  # examples, which the wrappers never use).
  "$PYTHON" "$HERE/patch_pymfem.py" "$SRC"

  # The SWIG version matters in both directions: PyMFEM 4.8 needs at least 4.3,
  # and SWIG 4.5 generates Python 2 calls (PyString_Check, PyInt_AsLong) that do
  # not compile.  pip's isolated build environment would take the newest SWIG, so
  # the build dependencies are installed here, pinned, and isolation is turned off.
  "$PYTHON" -m pip install "setuptools>=80.0.1" "numpy>=2" "cmake>=4" "swig==4.3.1"

  if [ -n "$WHEELHOUSE" ]; then
    mkdir -p "$WHEELHOUSE"
    "$PYTHON" -m pip wheel --no-deps --no-build-isolation "$SRC" \
        -C"with-parallel=Yes" -w "$WHEELHOUSE"
    "$PYTHON" -m pip install "$WHEELHOUSE"/mfem-"$VERSION"-*.whl
  else
    "$PYTHON" -m pip install --no-build-isolation "$SRC" -C"with-parallel=Yes"
  fi
fi

# A serial-only build imports mfem.ser but fails here, which is the point.
"$PYTHON" -c "
import mfem, mfem.par
from mpi4py import MPI
print('PyMFEM', mfem.__version__, 'with', MPI.Get_library_version().splitlines()[0])
"

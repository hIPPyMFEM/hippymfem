#!/bin/bash
# Build PyMFEM with CUDA for both MFEM and hypre, into a private prefix.
#
# hIPPyMFEM does not need this: its GPU path is the AD layer, which runs through JAX
# and needs nothing from MFEM.  A CUDA PyMFEM answers a different question -- whether
# the *solves* should move to the device -- and lets `hm.mfem_config()` report a real
# answer instead of a guess.  See docs/source/guide/gpu.rst for what it measured.
#
# Four upstream problems have to be worked around, each of which fails in a way that
# does not name its cause.  They are applied here as sed patches so the recipe is
# reproducible.
#
#   1. pyproject.toml uses the PEP 639 `license = "..."` string form, which older
#      setuptools rejects with a schema error.
#   2. With MFEM_USE_CUDA every .cpp is compiled by nvcc, but PyMFEM configures MFEM
#      with CMAKE_CXX_COMPILER=mpicxx, so cmake never records an MPI include directory
#      -- the wrapper supplied it -- and nvcc cannot find mpi.h.  The include path has
#      to go into CMAKE_CUDA_FLAGS.  Note that the cmake build directories are
#      cmbuild_par / cmbuild_ser: a stale cache silently ignores the new flags.
#   3. MFEM's own example binaries fail to link (the CUDA link line has no MPI).  They
#      are not used by the wrapper, so they are turned off.
#   4. The SWIG wrappers need -fpermissive (SWIG 4.2 generates SWIG_init returning
#      PyObject* while numpy's import_array1 returns int), and their setup.py imports
#      distutils, which Python 3.12 removed -- while importing setuptools instead
#      pulls in the standard library's socket, which does `import array` and finds
#      PyMFEM's own array.py.  The script's directory has to leave sys.path first.
#
# Usage:  tools/build_pymfem_cuda.sh [prefix] [cuda-arch]
#   default prefix     $HOME/pymfem-cuda
#   default cuda-arch  89   (L40S; A100 = 80, H100 = 90)
#
# Select the result with:
#   PYTHONPATH=<prefix>/PyMFEM/lib/python3.12/site-packages python ...
set -eu

PREFIX="${1:-$HOME/pymfem-cuda}"
ARCH="${2:-89}"
PY="${PYTHON:-python3}"

# Toolchain locations.  Defaults suit a workstation; a cluster that provides CUDA and MPI
# through modules exports these first instead of editing the script.  CUDA_HOME is
# taken from the environment if set, and from the usual place otherwise, because on a
# module system /usr/local/cuda either is absent or is the wrong version.
CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-/usr/local/cuda}}"
MPICC_BIN="${MPICC:-$(command -v mpicc || echo /usr/bin/mpicc)}"
MPICXX_BIN="${MPICXX:-$(command -v mpicxx || echo /usr/bin/mpicxx)}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_HOME
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v g++ || echo /usr/bin/g++)}"
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
command -v "$CUDACXX" >/dev/null || { echo "no nvcc at $CUDACXX" >&2; exit 1; }
echo "nvcc:   $CUDACXX ($("$CUDACXX" --version | tail -1))"
echo "mpi:    $MPICC_BIN / $MPICXX_BIN"
echo "host++: $CUDAHOSTCXX"

mkdir -p "$PREFIX"
cd "$PREFIX"
[ -d PyMFEM ] || git clone --depth 1 https://github.com/mfem/PyMFEM.git
SRC="$PREFIX/PyMFEM"

echo "=== patch 1: PEP 639 license field ==="
$PY - "$SRC" <<'EOF'
import re, sys, os
p = os.path.join(sys.argv[1], "pyproject.toml")
s = open(p).read()
s2 = re.sub(r'^license\s*=\s*"([^"]+)"\s*$', r'license = {text = "\1"}', s, flags=re.M)
s2 = re.sub(r'^license-files\s*=.*$\n', '', s2, flags=re.M)
if s2 != s:
    open(p, "w").write(s2)
    print("  patched")
else:
    print("  already fine")
EOF

echo "=== patch 2: MPI include path for nvcc; patch 3: examples off ==="
$PY - "$SRC" <<'EOF'
import sys, os
p = os.path.join(sys.argv[1], "_build_system", "build_mfem.py")
s = open(p).read()
if "HIPPYMFEM_EXTRA_CUDA_FLAGS" not in s:
    old = """    if bglb.enable_cuda:
        cmake_opts['DMFEM_USE_CUDA'] = '1'
        if bglb.cuda_arch != '':
            cmake_opts['DCMAKE_CUDA_ARCHITECTURES'] = bglb.cuda_arch"""
    new = """    if bglb.enable_cuda:
        cmake_opts['DMFEM_USE_CUDA'] = '1'
        if bglb.cuda_arch != '':
            cmake_opts['DCMAKE_CUDA_ARCHITECTURES'] = bglb.cuda_arch
        # nvcc compiles every .cpp when CUDA is on, and it is not the mpicxx wrapper
        # cmake was told to use, so it never sees an MPI include directory.
        if not serial:
            import subprocess as _sp
            inc = []
            try:
                out = _sp.run([os.environ.get('MPICXX', 'mpicxx'), '-showme:incdirs'], capture_output=True,
                              text=True, timeout=30)
                if out.returncode == 0:
                    inc = out.stdout.split()
            except Exception:
                pass
            extra = os.environ.get('HIPPYMFEM_EXTRA_CUDA_FLAGS', '')
            cmake_opts['DCMAKE_CUDA_FLAGS'] = (
                ' '.join('-I' + d for d in inc) + ' ' + extra).strip()"""
    assert s.count(old) == 1, "build_mfem.py layout changed upstream"
    s = s.replace(old, new)
    if not s.lstrip().startswith("import os"):
        s = "import os\n" + s
old = """                  'DMFEM_ENABLE_EXAMPLES': '1',
                  'DMFEM_ENABLE_MINIAPPS': '1',"""
new = """                  # not used by the wrapper, and they fail to link under CUDA
                  # (the parallel examples get no MPI on the CUDA link line)
                  'DMFEM_ENABLE_EXAMPLES': '0',
                  'DMFEM_ENABLE_MINIAPPS': '0',"""
if old in s:
    s = s.replace(old, new)
open(p, "w").write(s)
print("  patched")
EOF

echo "=== patch 4: SWIG wrapper build (-fpermissive, distutils, sys.path) ==="
for sub in _par _ser; do
  f="$SRC/mfem/$sub/setup.py"
  [ -f "$f" ] || continue
  $PY - "$f" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "    extra_compile_args = [cxxstdflag, '-DSWIG_TYPE_TABLE=PyMFEM']"
new = ("    # SWIG 4.2 generates SWIG_init returning PyObject* while numpy's\n"
       "    # import_array1 returns int; g++ names -fpermissive as the fix, and the\n"
       "    # return value is only used on the import-failure path.\n"
       "    extra_compile_args = [cxxstdflag, '-DSWIG_TYPE_TABLE=PyMFEM',\n"
       "                          '-fpermissive']")
if old in s:
    s = s.replace(old, new)
old2 = "sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))"
new2 = ("# This directory holds array.py; importing setuptools pulls in the standard\n"
        "# library's socket, which does `import array` and would find it.  Removing\n"
        "# sys.argv[0]'s directory is not enough: '' is on the path as well.\n"
        "_here = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))\n"
        "sys.path[:] = [_p for _p in sys.path\n"
        "               if os.path.abspath(_p if _p else os.getcwd()) != _here]")
if old2 in s:
    s = s.replace(old2, new2)
s = s.replace("from distutils.core import Extension, setup",
              "from setuptools import Extension, setup   # distutils is gone in 3.12")
open(p, "w").write(s)
print("  patched", p)
EOF
done

echo "=== configure and build ==="
MPI_INC=$("$MPICXX_BIN" -showme:incdirs 2>/dev/null | tr ' ' '\n' | sed 's#^#-I#' | tr '\n' ' ')
export MPICXX="$MPICXX_BIN"
export CUDAFLAGS="$MPI_INC -allow-unsupported-compiler"
export HIPPYMFEM_EXTRA_CUDA_FLAGS="-allow-unsupported-compiler"
# a stale cmake cache ignores the new CUDA flags, and the directories are *_par/*_ser
rm -rf "$SRC/external/mfem/cmbuild_par" "$SRC/external/mfem/cmbuild_ser"

cd "$SRC"
"$PY" setup.py install \
    --with-parallel --no-serial \
    --with-cuda --with-cuda-hypre --cuda-arch="$ARCH" \
    --prefix="$PREFIX/install" \
    --MPICC="$MPICC_BIN" --MPICXX="$MPICXX_BIN"

echo
cd "$PREFIX"   # not $SRC: the checkout's mfem/ directory would shadow the installed package
echo "=== verify ==="
SITE="$SRC/lib/python3.12/site-packages"
PYTHONPATH="$SITE" "$PY" - <<'EOF'
import os, re
import mfem
print("  mfem", mfem.__version__, "from", os.path.dirname(mfem.__file__))
cfg = open(os.path.join(os.path.dirname(mfem.__file__),
                        "external/par/include/mfem/config/_config.hpp")).read()
for f in ("MFEM_USE_MPI", "MFEM_USE_CUDA", "MFEM_USE_METIS"):
    print("  %-16s %s" % (f, bool(re.search(r"^\s*#define\s+%s\b" % f, cfg, re.M))))
import mfem.par as m
d = m.Device("cuda")
d.Print()
EOF
echo
echo "Select this build with:"
echo "  PYTHONPATH=$SITE python ..."

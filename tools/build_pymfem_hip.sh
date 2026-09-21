#!/bin/bash
# Build hypre and MFEM for AMD GPUs (ROCm/HIP) and PyMFEM's parallel wrappers on top.
#
# PyMFEM's build system knows only CUDA (--with-cuda, --with-cuda-hypre), so this does
# the three pieces by hand and hands PyMFEM an MFEM it did not build:
#
#   1. hypre with --with-hip (autotools; its CMake has no HIP switch at 2.32);
#   2. MFEM with MFEM_USE_HIP through CMake's HIP language, shared, MPI, METIS;
#   3. the SWIG wrappers of an existing CPU PyMFEM tree of the same MFEM version,
#      recompiled against the new headers and libraries (no swig rerun).
#
# One source patch is needed, and it is the HIP twin of the CUDA trap recorded in
# NOTES.md: sparsemat.hpp guards SparseMatrix's hipSPARSE members on
# MFEM_USE_CUDA_OR_HIP, which hip.hpp defines only when the HIP compiler is running
# (__HIP__).  libmfem.so is compiled by it, the wrappers by mpicxx, and the two would
# disagree on sizeof(SparseMatrix).  The guard is widened to MFEM_USE_HIP here, in the
# private source copy, so the installed header is right from the start.
#
# Usage:  CPU_PYMFEM=<tree> tools/build_pymfem_hip.sh <prefix> [gpu-arch]   (MI210: gfx90a)
# Needs:  /opt/rocm, an MPI compiler on PATH, and CPU_PYMFEM pointing at a CPU PyMFEM
#         tree of the same MFEM version, whose wrappers are recompiled here.  PYTHON
#         names the interpreter to build for (python3 by default).
set -eu
PREFIX="$1"
ARCH="${2:-gfx90a}"
CPU_PYMFEM="${CPU_PYMFEM:-}"
if [ -z "$CPU_PYMFEM" ] || [ ! -d "$CPU_PYMFEM/external/mfem" ]; then
  echo "set CPU_PYMFEM to a CPU PyMFEM tree of the same MFEM version:" >&2
  echo "  CPU_PYMFEM=/path/to/PyMFEM $0 $*" >&2
  exit 1
fi
PY="${PYTHON:-${PY:-python3}}"
NJ="${NJ:-$(nproc)}"
HYPRE_TAG="${HYPRE_TAG:-v3.2.0}"
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
MPICXX=$(command -v mpicxx)
MPICC=$(command -v mpicc)
MPIINC=$($MPICXX -showme:incdirs | tr ' ' '\n' | sed 's/^/-I/' | tr '\n' ' ')
echo "prefix $PREFIX  arch $ARCH  jobs $NJ  mpicxx $MPICXX"
mkdir -p "$PREFIX/src" "$PREFIX/logs"
LOG="$PREFIX/logs"
PYVER=$("$PY" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')
EXT="$CPU_PYMFEM/lib/$PYVER/site-packages/mfem/external"     # metis and zlib flags come from here

# ---------------------------------------------------------------- 1. hypre
if [ ! -f "$PREFIX/hypre/lib/libHYPRE.so" ]; then
  echo "=== hypre (HIP) $(date +%T)"
  # hypre 3.1 is the first with ROCm 7 support (hypre issue 1490): ROCm >= 6.5 has real
  # warp sync builtins with 64-bit masks, and 2.32 -- the version PyMFEM pins -- compiles
  # against ROCm 7 only after patches and then faults in every device kernel (hardware
  # exception 0x1016 in hypreGPUKernel_CSRMoveDiagFirst, hypre's own ij driver included)
  if [ ! -d "$PREFIX/src/hypre" ]; then
    curl -sL --retry 5 -o "$PREFIX/src/hypre-$HYPRE_TAG.tar.gz" \
        "https://github.com/hypre-space/hypre/archive/refs/tags/$HYPRE_TAG.tar.gz"
    mkdir -p "$PREFIX/src/hypre"
    tar xzf "$PREFIX/src/hypre-$HYPRE_TAG.tar.gz" -C "$PREFIX/src/hypre" --strip-components=1
  fi
  (cd "$PREFIX/src/hypre/src"
   # ROCm 7's rocprim headers need C++17 (std::variant); older hypre hard-codes C++14
   # for its HIP sources and --with-cxxstandard does not reach HIPCXXFLAGS
   sed -i 's/HIPCXXFLAGS="-x hip -std=c++14 /HIPCXXFLAGS="-x hip -std=c++17 /' configure
   # hypre 3 turns Umpire on for GPU builds; MFEM manages device memory itself
   ./configure --prefix="$PREFIX/hypre" --with-hip --with-gpu-arch="$ARCH" --enable-shared --without-umpire \
       CC="$MPICC" CXX="$MPICXX" > "$LOG/hypre_configure.log" 2>&1
   make clean > /dev/null 2>&1 || true
   make -j "$NJ" > "$LOG/hypre_make.log" 2>&1
   make install > "$LOG/hypre_install.log" 2>&1)
fi
ls "$PREFIX/hypre/lib/libHYPRE.so"

# ---------------------------------------------------------------- 2. MFEM
if [ ! -f "$PREFIX/mfem/par/lib/libmfem.so" ]; then
  echo "=== MFEM (HIP) $(date +%T)"
  [ -d "$PREFIX/src/mfem" ] || cp -r "$CPU_PYMFEM/external/mfem" "$PREFIX/src/mfem"
  rm -rf "$PREFIX/src/mfem/cmbuild_par" "$PREFIX/src/mfem/cmbuild_ser"
  "$PY" - "$PREFIX/src/mfem/linalg/sparsemat.hpp" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "   void InitGPUSparse();\n\n#ifdef MFEM_USE_CUDA_OR_HIP\n"
new = ("   void InitGPUSparse();\n\n"
       "// widened from MFEM_USE_CUDA_OR_HIP, which needs __HIP__: host-compiled code\n"
       "// (PyMFEM's wrappers) must see the same SparseMatrix layout as libmfem.so\n"
       "#if defined(MFEM_USE_CUDA_OR_HIP) || defined(MFEM_USE_HIP)\n")
if old in s:
    s = s.replace(old, new)
    open(p, "w").write(s)
    print("  patched", p)
elif "defined(MFEM_USE_CUDA_OR_HIP) || defined(MFEM_USE_HIP)" in s:
    print("  already patched")
else:
    sys.exit("sparsemat.hpp layout changed; patch not applied")
EOF
  # hipSPARSE's SpMV on ROCm 7 is right on a matrix's first call and wrong on every
  # later one, even with the dense-vector descriptors rebuilt per call (measured on an
  # MI210: error of the size of the result from the second call on), while MFEM's own
  # device kernel is right on every call.  Under HIP, SparseMatrix::AddMult takes the
  # native kernel.
  "$PY" - "$PREFIX/src/mfem/linalg/sparsemat.cpp" <<'SPMV'
import sys
p = sys.argv[1]
s = open(p).read()
old = "   if ((Device::Allows(Backend::CUDA_MASK | Backend::HIP_MASK)) && useGPUSparse)\n"
new = ("   // hipSPARSE SpMV is wrong after a matrix's first call on ROCm 7: native kernel\n"
       "#ifdef MFEM_USE_HIP\n"
       "   if (Device::Allows(Backend::CUDA_MASK) && useGPUSparse)\n"
       "#else\n" + old + "#endif\n")
if "hipSPARSE SpMV is wrong after a matrix's first call" in s:
    print("  sparsemat.cpp already patched")
else:
    assert s.count(old) == 1, "sparsemat.cpp SpMV layout changed"
    open(p, "w").write(s.replace(old, new))
    print("  patched", p)
SPMV
  # MFEM's HIP build compiles its C++ with the HIP flags (--offload-arch), so the C++
  # compiler must be ROCm's clang; OpenMPI's wrapper drives it through OMPI_CXX, which
  # keeps cmake's MPI detection working
  export OMPI_CXX="$ROCM_PATH/lib/llvm/bin/clang++"
  cmake -S "$PREFIX/src/mfem" -B "$PREFIX/src/mfem/cmbuild_par" \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX="$PREFIX/mfem/par" \
      -DCMAKE_CXX_COMPILER="$MPICXX" \
      -DMFEM_USE_MPI=ON -DMFEM_USE_METIS=ON -DMFEM_USE_METIS_5=ON -DMFEM_USE_ZLIB=ON \
      -DMFEM_USE_EXCEPTIONS=OFF -DMFEM_ENABLE_EXAMPLES=OFF -DMFEM_ENABLE_MINIAPPS=OFF \
      -DMFEM_ENABLE_TESTING=OFF \
      -DHYPRE_DIR="$PREFIX/hypre" -DMETIS_DIR="$EXT" \
      -DMFEM_USE_HIP=ON -DHIP_ARCH="$ARCH" -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
      -DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++" \
      -DCMAKE_HIP_FLAGS="$MPIINC" \
      > "$LOG/mfem_cmake.log" 2>&1
  cmake --build "$PREFIX/src/mfem/cmbuild_par" -j "$NJ" > "$LOG/mfem_make.log" 2>&1
  cmake --install "$PREFIX/src/mfem/cmbuild_par" > "$LOG/mfem_install.log" 2>&1
  unset OMPI_CXX
fi
ls "$PREFIX/mfem/par/lib/libmfem.so"

# ---------------------------------------------------------------- 3. wrappers
echo "=== PyMFEM parallel wrappers $(date +%T)"
if [ ! -d "$PREFIX/PyMFEM" ]; then
  mkdir -p "$PREFIX/PyMFEM"
  # the wrapper sources and python modules only: not the external builds or site-packages
  (cd "$CPU_PYMFEM" && tar cf - --exclude=./external --exclude=./lib --exclude=./build \
       --exclude='*.so' --exclude='*.o' .) | (cd "$PREFIX/PyMFEM" && tar xf -)
fi
TPL=$(grep -E "^MFEM_TPLFLAGS" "$PREFIX/mfem/par/share/mfem/config.mk" | cut -d= -f2-)
"$PY" - "$PREFIX" "$CPU_PYMFEM" "$TPL -D__HIP_PLATFORM_AMD__ -I$ROCM_PATH/include" <<'EOF'
import os, re, sys
prefix, cpu, tpl = sys.argv[1:4]
src = open(os.path.join(cpu, "setup_local.py")).read()
def put(key, val):
    global src
    src, n = re.subn(r'^%s = ".*"$' % key, '%s = "%s"' % (key, val), src, flags=re.M)
    assert n == 1, key
put("hypreinc", prefix + "/hypre/include")
put("hyprelib", prefix + "/hypre/lib")
put("mfem_outside", "1")
put("mfembuilddir", prefix + "/mfem/par/include")
put("mfemincdir", prefix + "/mfem/par/include/mfem")
put("mfemlnkdir", prefix + "/mfem/par/lib")
put("mfemsrcdir", prefix + "/src/mfem")
# config.mk records MFEM's flags as CMake wrote them: quoted paths, -isystem pairs and
# generator expressions carrying the HIP compile flags (-x hip, --offload-arch) that
# must not reach the host compiler.  Keep include paths and defines, once each.
toks, keep = re.sub(r"\$<[^>]*>", " ", tpl).replace('"', "").split(), []
i = 0
while i < len(toks):
    t = toks[i]
    if t == "-isystem" and i + 1 < len(toks):
        t = "-I" + toks[i + 1]
        i += 1
    if (t.startswith("-I") or t.startswith("-D")) and len(t) > 2 and t not in keep and ">" not in t:
        keep.append(t)
    i += 1
put("mfemptpl", " ".join(keep))
open(os.path.join(prefix, "PyMFEM", "setup_local.py"), "w").write(src)
print("  setup_local.py written")
EOF
# the wrapper setup.py keeps only the -I part of MFEM's flags; the HIP headers also need
# the platform define (__HIP_PLATFORM_AMD__), so the -D flags are passed through too
"$PY" - "$PREFIX/PyMFEM/mfem/_par/setup.py" <<'PATCH'
import sys
p = sys.argv[1]
s = open(p).read()
if "extra_tpl_defines" not in s:
    old = '        if x.startswith("-I"):\n            tpl_include.append(x[2:])\n'
    new = old + ('        elif x.startswith("-D") and len(x) > 2:\n'
                 '            extra_tpl_defines.append(x)\n')
    assert s.count(old) == 1, "setup.py layout changed"
    s = s.replace(old, new).replace("    tpl_include = []\n",
                                    "    tpl_include = []\n    extra_tpl_defines = []\n")
    old2 = "                          '-fpermissive']"
    assert s.count(old2) == 1, "setup.py compile args changed"
    s = s.replace(old2, "                          '-fpermissive'] + sorted(set(extra_tpl_defines))")
    open(p, "w").write(s)
    print("  patched", p)
PATCH
(cd "$PREFIX/PyMFEM/mfem/_par" && "$PY" setup.py build_ext --inplace --force --parallel "$NJ" \
     > "$LOG/wrappers.log" 2>&1)
SITE="$PREFIX/PyMFEM/lib/$PYVER/site-packages"
mkdir -p "$SITE"
rm -rf "$SITE/mfem"
cp -r "$PREFIX/PyMFEM/mfem" "$SITE/mfem"
# the layout a PyMFEM-built MFEM has, so that tools reading the build's configuration
# from site-packages (hippymfem.common.mfemconfig) find this one
mkdir -p "$SITE/mfem/external"
ln -sfn "$PREFIX/mfem/par" "$SITE/mfem/external/par"
ln -sfn "$PREFIX/hypre/include" "$SITE/mfem/external/include"
ln -sfn "$PREFIX/hypre/lib" "$SITE/mfem/external/lib64"
echo "=== done $(date +%T): PYTHONPATH=$SITE"

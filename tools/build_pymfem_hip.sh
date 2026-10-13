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
# One source patch is needed: sparsemat.hpp guards SparseMatrix's hipSPARSE members on
# MFEM_USE_CUDA_OR_HIP, which hip.hpp defines only when the HIP compiler is running
# (__HIP__).  libmfem.so is compiled by it, the wrappers by mpicxx, and the two would
# disagree on sizeof(SparseMatrix).  The guard is widened to MFEM_USE_HIP here, in the
# private source copy, so the installed header is right from the start.
#
# Usage:  CPU_PYMFEM=<tree> tools/build_pymfem_hip.sh <prefix> [gpu-arch]   (MI210: gfx90a)
# Needs:  ROCm (ROCM_PATH, which a cluster's module exports; /opt/rocm otherwise), an MPI
#         compiler on PATH (OpenMPI, or an MPICH such as Cray's), and CPU_PYMFEM pointing
#         at a CPU PyMFEM tree of the same MFEM version, whose wrappers are recompiled
#         here.  PYTHON names the interpreter to build for (python3 by default).
#         HYPRE_PINNED_STAGING=1 builds hypre with tools/hypre-3.2.0-staging.patch (below).
#         HYPRE_SORTED_SPGEMM=1 builds hypre with tools/hypre-3.2.0-spgemm.patch (below).
#         HYPRE_SEQ_DEVICE=1 builds hypre with tools/hypre-3.2.0-seqdevice.patch (below).
#         HYPRE_MIXEDINT=1 builds hypre with 64-bit global indices (--enable-mixedint), for
#         spaces of more than 2 147 483 647 unknowns, and has SWIG write the wrappers again
#         (below); it needs swig on PATH, or SWIG naming it.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
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
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
MPICXX=$(command -v mpicxx || true)
MPICC=$(command -v mpicc || true)
[ -n "$MPICXX" ] && [ -n "$MPICC" ] || { echo "no mpicxx or mpicc on PATH (load the MPI to build with first)" >&2; exit 1; }
# the MPI include directories, for the two compilers that are not the MPI wrapper (hipcc
# and CMake's HIP language): OpenMPI's wrapper lists them, an MPICH one shows its command
MPIINC=$($MPICXX -showme:incdirs 2>/dev/null | tr ' ' '\n' | sed '/^$/d; s/^/-I/' | tr '\n' ' ')
[ -n "$MPIINC" ] || MPIINC=$($MPICXX -show 2>/dev/null | tr ' ' '\n' | grep '^-I' | tr '\n' ' ')
[ -n "$MPIINC" ] || { echo "cannot read the MPI include directories from $MPICXX" >&2; exit 1; }
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
        "https://github.com/hypre-space/hypre/archive/refs/tags/$HYPRE_TAG.tar.gz" \
        || { echo "hypre $HYPRE_TAG was not downloaded (curl ended with $?)" >&2; exit 1; }
    mkdir -p "$PREFIX/src/hypre"
    tar xzf "$PREFIX/src/hypre-$HYPRE_TAG.tar.gz" -C "$PREFIX/src/hypre" --strip-components=1
  fi
  (cd "$PREFIX/src/hypre/src"
   # ROCm 7's rocprim headers need C++17 (std::variant); older hypre hard-codes C++14
   # for its HIP sources and --with-cxxstandard does not reach HIPCXXFLAGS
   sed -i 's/HIPCXXFLAGS="-x hip -std=c++14 /HIPCXXFLAGS="-x hip -std=c++17 /' configure
   # HYPRE_PINNED_STAGING=1: the exchange of shared dofs through page-locked CPU buffers
   # that are kept, started before the product with the local block instead of after it
   # (tools/hypre-3.2.0-staging.patch; two Newton-CG steps at 512^3 on 512 MI250X of
   # Frontier in 10.0 s instead of 10.8 s).  HYPRE_STAGE_PINNED=0 and HYPRE_STAGE_EARLY=0
   # in the environment of a run switch either part off.
   # (Each of the three patches is tried whole before it is applied, and without fuzz.  One
   # that failed part of the way left the hunks that had applied, the first of them with
   # the name the next run takes for "patched already", and that run built the tree as it
   # was; and patch's default fuzz put a hunk whose context had changed somewhere else.)
   if [ "${HYPRE_PINNED_STAGING:-0}" = 1 ] && ! grep -q hm_pin_get parcsr_mv/par_csr_communication.c; then
     patch -p1 --forward --fuzz=0 --dry-run < "$HERE/hypre-3.2.0-staging.patch" > "$LOG/hypre_patch.log" 2>&1 \
       && patch -p1 --forward --fuzz=0 < "$HERE/hypre-3.2.0-staging.patch" > "$LOG/hypre_patch.log" 2>&1 \
       || { echo "tools/hypre-3.2.0-staging.patch did not apply to hypre $HYPRE_TAG: $LOG/hypre_patch.log" >&2; exit 1; }
   fi
   # HYPRE_SORTED_SPGEMM=1: hypre copies and sorts both operands of every sparse product
   # for rocSPARSE.  With tools/hypre-3.2.0-spgemm.patch an operand whose rows are sorted
   # already (a transpose, the result of the product before) is used as it is; the sorted
   # copy of one with only the first entry of each row out of place (the fine matrix,
   # diagonal first) is written in one pass; the matrix that joins a diagonal block with
   # its off-processor one is written in order; and the rows a rank receives are sorted
   # before they are stacked under its own.  A BoomerAMG setup for 8.6 million unknowns of
   # quadratic elements on an MI250X of Frontier then takes 0.81 s instead of 1.53 s, and
   # 1.0 to 1.2 s instead of 2.3 to 2.5 s a GCD on 2 to 8 of them, with the same iterates
   # and half the device memory at its peak (the GPU guide, "Several GPUs").
   # HYPRE_SPGEMM_SKIP_SORT=0 in the environment of a run has every operand sorted again.
   if [ "${HYPRE_SORTED_SPGEMM:-0}" = 1 ] && ! grep -q hypre_CSRSpGemmRowsAscend seq_mv/csr_spgemm_device_rocsparse.c; then
     patch -p1 --forward --fuzz=0 --dry-run < "$HERE/hypre-3.2.0-spgemm.patch" > "$LOG/hypre_patch_spgemm.log" 2>&1 \
       && patch -p1 --forward --fuzz=0 < "$HERE/hypre-3.2.0-spgemm.patch" > "$LOG/hypre_patch_spgemm.log" 2>&1 \
       || { echo "tools/hypre-3.2.0-spgemm.patch did not apply to hypre $HYPRE_TAG: $LOG/hypre_patch_spgemm.log" >&2; exit 1; }
   fi
   # HYPRE_SEQ_DEVICE=1: BoomerAMG's seq_threshold (a hierarchy ended at a level of a few
   # thousand rows, which one process finishes, gathered) faults with hypre on a device:
   # it reads the level through host pointers.  With tools/hypre-3.2.0-seqdevice.patch
   # the level is read from a host copy and solved on a device: the first rank's, which
   # sends the solution to the others, or with redundant=1 every rank's, each keeping its
   # own rows (the patch gives the ranks' coarsenings the same random numbers and compares
   # the solvers, since corrections of different hierarchies do not fit together).  The
   # last levels of a hierarchy on hundreds of ranks are a few rows a rank that talk to
   # many ranks; gathered at 30 000 rows an iteration of CG with a linear-element
   # hierarchy takes 7.1 ms instead of 8.8 ms on 512 MI250X of Frontier, and a solve of
   # the prior 0.088 s instead of 0.101 s (the GPU guide, "AMD cards").  Nothing changes
   # for a solver that does not set seq_threshold.
   # (The source is known for patched by a line of the patch's third version, not by the
   # name of its first function: one that carries an older version is not built as it is.)
   if [ "${HYPRE_SEQ_DEVICE:-0}" = 1 ] && ! grep -q 'coarse_size <= (HYPRE_BigInt) seq_threshold' parcsr_ls/par_amg_setup.c; then
     patch -p1 --forward --fuzz=0 --dry-run < "$HERE/hypre-3.2.0-seqdevice.patch" > "$LOG/hypre_patch_seqdevice.log" 2>&1 \
       && patch -p1 --forward --fuzz=0 < "$HERE/hypre-3.2.0-seqdevice.patch" > "$LOG/hypre_patch_seqdevice.log" 2>&1 \
       || { echo "tools/hypre-3.2.0-seqdevice.patch did not apply to hypre $HYPRE_TAG: $LOG/hypre_patch_seqdevice.log" \
                 "(a source that carries an older version of the patch is not patched over: remove" \
                 "$PREFIX/src/hypre, which is then unpacked and patched again, and give every HYPRE_* switch of the first build)" >&2; exit 1; }
   fi
   # hypre 3 turns Umpire on for GPU builds; MFEM manages device memory itself.  Its
   # device sources go to hipcc, which is not the MPI wrapper and finds mpi.h only if
   # it is told where (an MPI module that sets CPATH hides this; Cray's does not)
   # HYPRE_MIXEDINT=1: the unknowns of a space are numbered with 64-bit integers
   # (HYPRE_BigInt) and those of a rank stay 32-bit (HYPRE_Int).  A GPU build of hypre
   # has no --enable-bigint: its device kernels and the vendor's take 32-bit local indices.
   MIXED=""; [ "${HYPRE_MIXEDINT:-0}" = 1 ] && MIXED="--enable-mixedint"
   ./configure --prefix="$PREFIX/hypre" --with-hip --with-gpu-arch="$ARCH" --enable-shared --without-umpire $MIXED \
       CC="$MPICC" CXX="$MPICXX" HIPCXXFLAGS="$MPIINC" > "$LOG/hypre_configure.log" 2>&1 \
       || { echo "hypre's configure failed: $LOG/hypre_configure.log" >&2; exit 1; }
   make clean > /dev/null 2>&1 || true
   # A compiler that cannot build a device source at -O2 gets that one source at -Os.
   # The clang of ROCm 7.2.0 on Frontier (22.0.0git, roc-7.2.0 26014) stops in its AMDGPU
   # backend on three SpGEMM kernels (csr_spgemm_device_numer1 to 3: "Illegal
   # instruction detected: Operand has incorrect register class"); they compile at -Os
   # and -O1, and at -O2 with the compilers of ROCm 7.1.1 and 7.14.1.
   n=0
   until make -j "$NJ" > "$LOG/hypre_make.log" 2>&1; do
     bad=$(grep -o '[A-Za-z0-9_]*\.obj\] Error' "$LOG/hypre_make.log" | sed 's/\] Error//' | sort -u | tr '\n' ' ')
     n=$((n + 1))
     [ -n "$bad" ] && [ "$n" -le 8 ] || { echo "hypre did not build: $LOG/hypre_make.log" >&2; exit 1; }
     echo "  the compiler failed at -O2, building at -Os: $bad"
     echo "$bad: CUFLAGS += -Os" >> config/Makefile.config
   done
   make install > "$LOG/hypre_install.log" 2>&1 \
     || { echo "hypre did not install: $LOG/hypre_install.log" >&2; exit 1; })
else
  # hypre is built and stays as it is, so a patch asked for now is not applied.  It is said
  # where the source of the library does not carry it: what is built on top of this prefix,
  # and tools/build_hypre_single.sh, which copies that source, would take it for there.
  for sw in "HYPRE_PINNED_STAGING|hm_pin_get|parcsr_mv/par_csr_communication.c" \
            "HYPRE_SORTED_SPGEMM|hypre_CSRSpGemmRowsAscend|seq_mv/csr_spgemm_device_rocsparse.c" \
            "HYPRE_SEQ_DEVICE|coarse_size <= (HYPRE_BigInt) seq_threshold|parcsr_ls/par_amg_setup.c"; do
    IFS='|' read -r name mark file <<< "$sw"
    if [ "${!name:-0}" = 1 ] && ! grep -qs "$mark" "$PREFIX/src/hypre/src/$file"; then
      echo "$name=1 is not applied: $PREFIX/hypre/lib/libHYPRE.so is built already and is kept, and" \
           "the patch as it stands is not in $PREFIX/src/hypre (to build hypre again remove $PREFIX/hypre," \
           "and $PREFIX/src/hypre too where it carries an older version of the patch)" >&2
    fi
  done
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
  # compiler must be ROCm's clang; the MPI wrapper drives it (OMPI_CXX for OpenMPI's,
  # MPICH_CXX for an MPICH one), which keeps cmake's MPI detection working
  export OMPI_CXX="$ROCM_PATH/lib/llvm/bin/clang++"
  export MPICH_CXX="$OMPI_CXX"
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
      > "$LOG/mfem_cmake.log" 2>&1 \
      || { echo "MFEM's cmake failed: $LOG/mfem_cmake.log" >&2; exit 1; }
  cmake --build "$PREFIX/src/mfem/cmbuild_par" -j "$NJ" > "$LOG/mfem_make.log" 2>&1 \
      || { echo "MFEM did not build: $LOG/mfem_make.log" >&2; exit 1; }
  cmake --install "$PREFIX/src/mfem/cmbuild_par" > "$LOG/mfem_install.log" 2>&1 \
      || { echo "MFEM did not install: $LOG/mfem_install.log" >&2; exit 1; }
  unset OMPI_CXX MPICH_CXX
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
# HYPRE_MIXEDINT=1: the wrappers of the CPU tree were written by SWIG for a hypre with one
# integer width, so they convert a global size or offset as an int.  SWIG writes them
# again, from the MFEM headers of the CPU tree (the same interface as the wrappers that are
# known to compile here) with the hypre headers of this build in front of that tree's.
if [ "${HYPRE_MIXEDINT:-0}" = 1 ] && [ ! -f "$PREFIX/PyMFEM/.swig_mixedint" ]; then
  SWIG="${SWIG:-$(command -v swig || true)}"
  [ -n "$SWIG" ] && [ -x "$SWIG" ] || { echo "HYPRE_MIXEDINT=1 needs swig (pip install swig), or SWIG=<path>" >&2; exit 1; }
  MPI4PY_INC=$("$PY" -c 'import mpi4py; print(mpi4py.get_include())')
  echo "  swig: $("$SWIG" -version | grep -i version | head -1)"
  (cd "$PREFIX/PyMFEM/mfem/_par"
   # (SWIG reads HYPRE_utilities.h, where the two types are chosen, and not the
   # HYPRE_config.h that it includes: the choice is named here)
   ls *_wrap.cxx | sed 's/_wrap\.cxx$//' | xargs -P "$NJ" -I{} "$SWIG" -c++ -python -std=c++17 -fastproxy -olddefs -keyword -DHYPRE_MIXEDINT \
       -I"$PREFIX/hypre/include" -I"$EXT/par/include" -I"$EXT/par/include/mfem" -I"$CPU_PYMFEM/external/mfem" \
       -I"$EXT/include" -I"$MPI4PY_INC" {}.i) > "$LOG/swig.log" 2>&1 \
      || { echo "swig failed: $LOG/swig.log" >&2; exit 1; }
  touch "$PREFIX/PyMFEM/.swig_mixedint"
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
     > "$LOG/wrappers.log" 2>&1) \
     || { echo "the wrappers did not build: $LOG/wrappers.log" >&2; exit 1; }
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

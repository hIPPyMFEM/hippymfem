#!/bin/bash
# Build a single-precision libHYPRE next to the double-precision one of a PyMFEM build,
# for hippymfem.algorithms.singlesolve (HIPPYMFEM_HYPRE_SINGLE): the Jacobian of a PDE
# problem, its BoomerAMG hierarchy and the CG solves with it then live in this library.
#
#   tools/build_hypre_single.sh <PyMFEM source tree> <output directory>
#   tools/build_hypre_single.sh <prefix of tools/build_pymfem_hip.sh> <output directory>
#
# The options are those of the installed build (its CMakeCache.txt under
# external/hypre/src/cmbuild) with HYPRE_ENABLE_SINGLE=ON, so the library is for the
# same GPU architecture, the same MPI and the same CUDA.  For an AMD (HIP) stack, whose
# hypre is configured with autotools, they are the options that its config.status
# recorded with --enable-single, in a copy of its source tree with the patches it was
# built with (the build is in the source, and that tree stays as it is): the same index
# widths, so 64-bit global indices where the stack has them (HYPRE_MIXEDINT=1).  Two
# things make it usable next to the double-precision library, which defines the same
# symbols:
#
#   * it is linked with -Bsymbolic, so that its own calls stay inside it, and it is
#     loaded with RTLD_LOCAL (ctypes' default) and reached through its handle only;
#   * tools/hypre_offsets.c, compiled against the headers and the HYPRE_config.h of
#     this very build, writes the offsets of the few structure fields the library reads
#     (the arrays of a matrix's two blocks, the data of a vector) to
#     libHYPRE_single.json, next to libHYPRE_single.so.
#
# Load the compiler and MPI modules of the PyMFEM build first (MODULES="gcc/12.3.0
# openmpi/4.1.8" loads them here): a library linked against another MPI does not load
# next to the installed one.  About three minutes with twelve jobs (JOBS).  Built and
# used with CUDA builds for H100, L40S and RTX PRO 6000 Blackwell cards, with a host
# build, and with the HIP build of tools/build_pymfem_hip.sh for MI250X cards (hypre
# 3.2.0, ROCm 7.2.0).
set -eo pipefail
PYMFEM=${1:?PyMFEM source tree, or the prefix of tools/build_pymfem_hip.sh}
NEW=${2:?output directory}
TOOLS=$(cd "$(dirname "$0")" && pwd)
if [ -n "${MODULES:-}" ]; then
  source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
  module load $MODULES > /dev/null 2>&1
fi
set -u
mkdir -p "$NEW"
NEW=$(cd "$NEW" && pwd)

# ------------------------------------------------- an autotools build (the HIP stack)
if [ ! -f "$PYMFEM/external/hypre/src/cmbuild/CMakeCache.txt" ] && [ -x "$PYMFEM/src/hypre/src/config.status" ]; then
  OLD=$PYMFEM/src/hypre
  rm -rf "$NEW/hypre"
  mkdir -p "$NEW/hypre"
  (cd "$OLD" && tar cf - --exclude='*.o' --exclude='*.obj' --exclude='*.so' --exclude='*.a' .) | (cd "$NEW/hypre" && tar xf -)
  cd "$NEW/hypre/src"
  # the options of the installed build, as its configure was given them
  eval "set -- $(./config.status --config)"
  ./configure "$@" --enable-single --prefix="$NEW/install" LDFLAGS=-Wl,-Bsymbolic > "$NEW/configure.log" 2>&1 \
      || { tail -20 "$NEW/configure.log"; exit 1; }
  grep -q "define HYPRE_SINGLE 1" HYPRE_config.h || { echo "HYPRE_SINGLE did not reach HYPRE_config.h"; exit 1; }
  # (the sources that the compiler of the installed build could not take at -O2)
  grep ': CUFLAGS += ' "$OLD/src/config/Makefile.config" >> config/Makefile.config || true
  make clean > /dev/null 2>&1 || true
  n=0
  until make -j "${JOBS:-12}" > "$NEW/build.log" 2>&1; do
    bad=$(grep -o '[A-Za-z0-9_]*\.obj\] Error' "$NEW/build.log" | sed 's/\] Error//' | sort -u | tr '\n' ' ')
    n=$((n + 1))
    [ -n "$bad" ] && [ "$n" -le 8 ] || { tail -30 "$NEW/build.log"; exit 1; }
    echo "  the compiler failed at -O2, building at -Os: $bad"
    echo "$bad: CUFLAGS += -Os" >> config/Makefile.config
  done
  # the library linked once more under its own soname (the last step of the build, in lib/):
  # two libraries of one soname in a process are told apart by path only
  rm -f lib/libHYPRE*
  make -C lib SONAME=libHYPRE_single.so >> "$NEW/build.log" 2>&1 || { tail -30 "$NEW/build.log"; exit 1; }
  want=$(ldd "$PYMFEM/hypre/lib/libHYPRE.so" | awk '/libmpi/ {print $1}')
  got=$(ldd lib/libHYPRE_single.so | awk '/libmpi/ {print $1}')
  [ "$want" = "$got" ] || { echo "linked against $got, the installed library against $want: load the modules of the stack"; exit 1; }
  cp lib/libHYPRE_single.so "$NEW/libHYPRE_single.so"
  MPICC=$(./config.status --config | tr ' ' '\n' | sed -n "s/^'\{0,1\}CC=//p" | tr -d "'")
  "${MPICC:-mpicc}" -I. -Iutilities -Iseq_mv -Iparcsr_mv -Imultivector -Iseq_block_mv \
      ${ROCM_PATH:+-I"$ROCM_PATH/include"} -D__HIP_PLATFORM_AMD__ "$TOOLS/hypre_offsets.c" -o "$NEW/hypre_offsets"
  "$NEW/hypre_offsets" > "$NEW/libHYPRE_single.json"
  grep -q '"sizeof_real": 4' "$NEW/libHYPRE_single.json" || { echo "the offsets program did not see a single-precision build"; exit 1; }
  echo "built $NEW/libHYPRE_single.so ($got) and libHYPRE_single.json"
  echo "use it with  HIPPYMFEM_HYPRE_SINGLE=$NEW/libHYPRE_single.so"
  exit 0
fi

# ------------------------------------------------- a CMake build (PyMFEM's own)
SRC=$PYMFEM/external/hypre/src
OLD=$SRC/cmbuild
[ -f "$OLD/CMakeCache.txt" ] || { echo "no hypre build under $OLD, and no $PYMFEM/src/hypre/src/config.status"; exit 1; }
INIT=$NEW/initial_cache.cmake
: > "$INIT"
while IFS= read -r line; do
  name=${line%%:*}; value=${line#*=}
  printf 'set(%s "%s" CACHE STRING "" FORCE)\n' "$name" "$value" >> "$INIT"
done < <(grep -E '^(CMAKE_BUILD_TYPE|CMAKE_C_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_HIP_COMPILER|CMAKE_HIP_ARCHITECTURES|CMAKE_HIP_FLAGS|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS|CUDA_TOOLKIT_ROOT_DIR|ROCM_PATH|HIP_PATH|MPI_C_COMPILER|MPI_CXX_COMPILER|HYPRE_ENABLE_[A-Z_]+|HYPRE_WITH_[A-Z_]+|HYPRE_CUDA_SM):' "$OLD/CMakeCache.txt" \
           | grep -vE '^HYPRE_ENABLE_SINGLE:')
cat >> "$INIT" <<EOT
set(HYPRE_ENABLE_SINGLE "ON" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic" CACHE STRING "" FORCE)
EOT
cmake -C "$INIT" -S "$SRC" -B "$NEW/build" > "$NEW/configure.log" 2>&1 || { tail -20 "$NEW/configure.log"; exit 1; }
grep -q "define HYPRE_SINGLE 1" "$NEW/build/HYPRE_config.h" || { echo "HYPRE_SINGLE did not reach HYPRE_config.h"; exit 1; }
cmake --build "$NEW/build" -j "${JOBS:-12}" > "$NEW/build.log" 2>&1 || { tail -30 "$NEW/build.log"; exit 1; }
want=$(ldd "$OLD/libHYPRE.so" | awk '/libmpi/ {print $1}')
got=$(ldd "$NEW/build/libHYPRE.so" | awk '/libmpi/ {print $1}')
[ "$want" = "$got" ] || { echo "linked against $got, the installed library against $want: load the modules of the PyMFEM build"; exit 1; }
cp "$NEW/build/libHYPRE.so" "$NEW/libHYPRE_single.so"
# the offsets of the structure fields, from this build's own headers
MPICC=$(awk -F= '/^MPI_C_COMPILER:/ {print $2}' "$OLD/CMakeCache.txt")
CUDAINC=$(awk -F= '/^CUDA_TOOLKIT_ROOT_DIR:/ {print $2}' "$OLD/CMakeCache.txt")
"${MPICC:-mpicc}" -I"$NEW/build" -I"$SRC" -I"$SRC/utilities" -I"$SRC/seq_mv" -I"$SRC/parcsr_mv" \
    -I"$SRC/multivector" -I"$SRC/seq_block_mv" ${CUDAINC:+-I"$CUDAINC/include"} -I/usr/local/cuda/include \
    "$TOOLS/hypre_offsets.c" -o "$NEW/hypre_offsets"
"$NEW/hypre_offsets" > "$NEW/libHYPRE_single.json"
grep -q '"sizeof_real": 4' "$NEW/libHYPRE_single.json" || { echo "the offsets program did not see a single-precision build"; exit 1; }
echo "built $NEW/libHYPRE_single.so ($got) and libHYPRE_single.json"
echo "use it with  HIPPYMFEM_HYPRE_SINGLE=$NEW/libHYPRE_single.so"

#!/bin/bash
# Build a single-precision libHYPRE next to the double-precision one of a PyMFEM build,
# for hippymfem.algorithms.singlesolve (HIPPYMFEM_HYPRE_SINGLE): the Jacobian of a PDE
# problem, its BoomerAMG hierarchy and the CG solves with it then live in this library.
#
#   tools/build_hypre_single.sh <PyMFEM source tree> <output directory>
#
# The options are those of the installed build (its CMakeCache.txt under
# external/hypre/src/cmbuild) with HYPRE_ENABLE_SINGLE=ON, so the library is for the
# same GPU architecture, the same MPI and the same CUDA.  Two things make it usable
# next to the double-precision library, which defines the same symbols:
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
# used with CUDA builds for H100, L40S and RTX PRO 6000 Blackwell cards and with a host
# build; not tried with a HIP build.
set -eo pipefail
PYMFEM=${1:?PyMFEM source tree}
NEW=${2:?output directory}
TOOLS=$(cd "$(dirname "$0")" && pwd)
if [ -n "${MODULES:-}" ]; then
  source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
  module load $MODULES > /dev/null 2>&1
fi
set -u
SRC=$PYMFEM/external/hypre/src
OLD=$SRC/cmbuild
[ -f "$OLD/CMakeCache.txt" ] || { echo "no hypre build under $OLD"; exit 1; }
mkdir -p "$NEW"
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

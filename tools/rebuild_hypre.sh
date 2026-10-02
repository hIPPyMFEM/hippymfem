#!/bin/bash
# Build a second libHYPRE.so for a PyMFEM that tools/build_pymfem_cuda.sh built, with a
# faster exchange of shared dofs between GPUs.
#
# hypre exchanges the values of shared dofs at every product with a parallel matrix: 23
# to 31 times in one CG iteration with a BoomerAMG V-cycle.  As PyMFEM builds it, hypre
# packs the values on the GPU, allocates a buffer in CPU memory, copies them into it,
# sends them, and copies what it receives back through a second buffer; both buffers
# are pageable and freed at the end of the exchange.  Two variants, one per option:
#
#   --pinned-staging    The same route through CPU memory with page-locked buffers that
#                       are kept from one exchange to the next
#                       (tools/hypre-2.32.0-pinned-staging.patch, applied to a copy of
#                       the source).  Works with any MPI.  Measured per CG iteration on
#                       MIG instances of RTX PRO 6000 Blackwell cards with 2.1 million
#                       dofs each: 17.0 instead of 18.1 ms on sixteen (-5.7 %), -3.1 %
#                       on eight, -3.3 % on two; -5.1 % on sixteen with 134 thousand
#                       dofs each.  HYPRE_STAGE_PINNED=0 in the environment of a run
#                       gives hypre's own buffers back.
#   --gpu-aware-mpi     hypre gives MPI the device buffers (HYPRE_WITH_GPU_AWARE_MPI) and
#                       a CUDA-aware MPI moves them between the GPUs itself.  Measured
#                       on two H100 of one node with Open MPI 4.1.8 built --with-cuda:
#                       5.4 instead of 6.2 ms with 2.2 million dofs on each card and
#                       20.9 instead of 22.4 ms with 8.5 million.  Not for MIG
#                       instances: they cannot share memory through CUDA IPC, Open MPI
#                       then stages the buffers itself, and that was slower than hypre's
#                       own staging on four to sixteen instances (21.2 against 18.1 ms
#                       on sixteen).
#
# The options change nothing in hypre's interface, so the library built here can take
# the place of the installed one:
#
#   module load <the MPI to build against>     # for --gpu-aware-mpi, one with CUDA support
#   tools/rebuild_hypre.sh --pinned-staging <PyMFEM source tree> [<build directory>]
#
# and then either preload it,
#
#   mpirun -n 16 -x LD_PRELOAD=<build>/libHYPRE.so tools/mpirun_pinned.sh python script.py
#
# or copy it over <site-packages>/mfem/external/lib64/libHYPRE.so (keep the old one).
#
# What --gpu-aware-mpi needs at run time:
#   * the same MPI for mpirun, mpi4py and MFEM (the major version PyMFEM was built with;
#     LD_PRELOAD its libmpi.so as well if their run paths name another build);
#   * a transport of that MPI that understands device pointers.  With Open MPI 4:
#     --mca pml ob1 --mca btl self,smcuda (one node), unless its UCX was built with CUDA.
#     A transport that does not will read a device address as CPU memory and crash.
# hippymfem checks what it can (hippymfem.common.mfemconfig.hypre_gpu_aware_mpi and
# mpi_gpu_support) and stops with the reason, since the alternative is a segmentation
# fault inside MPI.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
AWARE=0; PINNED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-aware-mpi)  AWARE=1; shift ;;
    --pinned-staging) PINNED=1; shift ;;
    -h|--help) sed -n '2,52p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
USAGE="usage: $0 [--pinned-staging] [--gpu-aware-mpi] <PyMFEM source tree> [<build directory>]"
PYMFEM=${1:?$USAGE}
[ $AWARE = 1 ] || [ $PINNED = 1 ] || { echo "$USAGE"; echo "  name at least one of the two options"; exit 1; }
[ $AWARE = 1 ] && [ $PINNED = 1 ] && echo "note: with --gpu-aware-mpi nothing is staged through CPU memory, so --pinned-staging changes nothing"
SRC=$PYMFEM/external/hypre/src
OLD=$SRC/cmbuild
NEW=${2:-$SRC/cmbuild_exchange}
[ -f "$OLD/CMakeCache.txt" ] || { echo "no hypre build under $OLD"; exit 1; }
command -v mpicc >/dev/null || { echo "no mpicc in PATH: load the MPI to build against first"; exit 1; }
MPICC=$(command -v mpicc)
MPICXX=$(command -v mpicxx)
MPIINC=$(dirname "$(dirname "$MPICC")")/include
if [ $AWARE = 1 ] && command -v ompi_info >/dev/null; then
  ompi_info --parsable --all 2>/dev/null | grep -q "opal_built_with_cuda_support:value:true" \
    || echo "WARNING: $(command -v ompi_info) does not report CUDA support; the library built here needs an MPI that has it"
fi
mkdir -p "$NEW"
if [ $PINNED = 1 ]; then
  # never the PyMFEM tree itself: a copy of the source, patched
  rm -rf "$NEW/src"
  cp -r "$SRC" "$NEW/src"
  rm -rf "$NEW/src/cmbuild" "$NEW/src/cmbuild_exchange"
  ( cd "$NEW/src" && patch -p1 --forward < "$HERE/hypre-2.32.0-pinned-staging.patch" ) \
    || { echo "the patch is for hypre 2.32.0 and did not apply to $SRC"; exit 1; }
  SRC=$NEW/src
fi
# the options of the installed build, with the MPI of this shell
INIT=$NEW/initial_cache.cmake
: > "$INIT"
while IFS= read -r line; do
  name=${line%%:*}
  value=${line#*=}
  case "$name" in
    CMAKE_CUDA_FLAGS)
      # the old MPI's include directory is in the CUDA flags: name this one
      value=$(echo "$value" | sed -E "s#-I[^ ]*openmpi[^ ]*/include#-I$MPIINC#")
      ;;
  esac
  printf 'set(%s "%s" CACHE STRING "" FORCE)\n' "$name" "$value" >> "$INIT"
done < <(grep -E '^(CMAKE_BUILD_TYPE|CMAKE_C_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS|CUDA_TOOLKIT_ROOT_DIR|HYPRE_ENABLE_[A-Z_]+|HYPRE_WITH_[A-Z_]+|HYPRE_CUDA_SM):' "$OLD/CMakeCache.txt" \
           | grep -vE '^(HYPRE_WITH_GPU_AWARE_MPI|HYPRE_WITH_MPI):')
cat >> "$INIT" <<EOF
set(HYPRE_WITH_MPI "ON" CACHE STRING "" FORCE)
set(HYPRE_WITH_GPU_AWARE_MPI "$([ $AWARE = 1 ] && echo ON || echo OFF)" CACHE STRING "" FORCE)
set(MPI_C_COMPILER "$MPICC" CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER "$MPICXX" CACHE STRING "" FORCE)
EOF
cmake -C "$INIT" -S "$SRC" -B "$NEW/build" > "$NEW/configure.log" 2>&1 || { tail -20 "$NEW/configure.log"; exit 1; }
if [ $AWARE = 1 ]; then
  grep -q "define HYPRE_WITH_GPU_AWARE_MPI 1" "$NEW/build/HYPRE_config.h" || { echo "the option did not reach HYPRE_config.h"; exit 1; }
fi
cmake --build "$NEW/build" -j "${JOBS:-8}" > "$NEW/build.log" 2>&1 || { tail -20 "$NEW/build.log"; exit 1; }
cp "$NEW/build/libHYPRE.so" "$NEW/libHYPRE.so"
echo "built $NEW/libHYPRE.so$([ $PINNED = 1 ] && echo ' (page-locked staging buffers)')$([ $AWARE = 1 ] && echo ' (device buffers handed to MPI)')"
echo "  preload it:  mpirun ... -x LD_PRELOAD=$NEW/libHYPRE.so ..."
echo "  or replace:  $PYMFEM/lib/python*/site-packages/mfem/external/lib64/libHYPRE.so"

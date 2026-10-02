#!/bin/bash
# Build a second libHYPRE.so that hands its device buffers to MPI, for a PyMFEM that
# tools/build_pymfem_cuda.sh built.
#
# hypre exchanges the values of shared dofs at every product with a parallel matrix: 23
# to 31 times in one CG iteration with a BoomerAMG V-cycle.  As PyMFEM builds it, hypre
# packs the values on the GPU, copies them to CPU memory, sends them, and copies what it
# receives back.  Configured with HYPRE_WITH_GPU_AWARE_MPI it gives MPI the device
# buffers, and a CUDA-aware MPI moves them between the GPUs itself.  Measured on two
# H100 of one node with Open MPI 4.1.8 built --with-cuda (docs: "Several GPUs"): a CG
# iteration took 5.4 instead of 6.2 ms with 2.2 million dofs on each card and 20.9
# instead of 22.4 ms with 8.5 million.
#
# The option is a compile-time one and changes nothing in hypre's interface, so the
# library this script builds can take the place of the installed one:
#
#   module load <an MPI built with CUDA support>          # its mpicc must be first in PATH
#   tools/rebuild_hypre_gpu_aware_mpi.sh <PyMFEM source tree> [<build directory>]
#
# and then either preload it,
#
#   mpirun -n 2 --mca pml ob1 --mca btl self,smcuda -x LD_PRELOAD=<build>/libHYPRE.so \
#       tools/mpirun_pinned.sh python script.py
#
# or copy it over <site-packages>/mfem/external/lib64/libHYPRE.so (keep the old one).
#
# What it needs at run time:
#   * the same MPI for mpirun, mpi4py and MFEM (same major version as the one PyMFEM
#     was built with; LD_PRELOAD of its libmpi.so if their run paths name another);
#   * a transport of that MPI that understands device pointers.  With Open MPI 4:
#     --mca pml ob1 --mca btl self,smcuda (one node), unless its UCX was built with CUDA.
#     A transport that does not will read a device address as CPU memory and crash;
#   * on MIG instances, --mca btl_smcuda_use_cuda_ipc 0: they cannot share memory
#     through CUDA IPC, and Open MPI then stages the buffers through shared CPU memory.
# hippymfem checks the first two as far as it can (hippymfem.common.mfemconfig) and
# stops with an explanation, since the alternative is a segmentation fault in MPI.
set -euo pipefail
PYMFEM=${1:?usage: $0 <PyMFEM source tree> [<build directory>]}
SRC=$PYMFEM/external/hypre/src
OLD=$SRC/cmbuild
NEW=${2:-$SRC/cmbuild_gpu_aware_mpi}
[ -f "$OLD/CMakeCache.txt" ] || { echo "no hypre build under $OLD"; exit 1; }
command -v mpicc >/dev/null || { echo "no mpicc in PATH: load the CUDA-aware MPI first"; exit 1; }
MPICC=$(command -v mpicc)
MPICXX=$(command -v mpicxx)
MPIINC=$(dirname "$(dirname "$MPICC")")/include
if command -v ompi_info >/dev/null; then
  ompi_info --parsable --all 2>/dev/null | grep -q "opal_built_with_cuda_support:value:true" \
    || echo "WARNING: $(command -v ompi_info) does not report CUDA support; the library built here needs an MPI that has it"
fi
mkdir -p "$NEW"
# the options of the installed build, with the MPI replaced and the one option added
INIT=$NEW/initial_cache.cmake
: > "$INIT"
while IFS= read -r line; do
  name=${line%%:*}
  value=${line#*=}
  case "$name" in
    CMAKE_CUDA_FLAGS)
      # the old MPI's include directory is in the CUDA flags: name the new one
      value=$(echo "$value" | sed -E "s#-I[^ ]*openmpi[^ ]*/include#-I$MPIINC#")
      ;;
  esac
  printf 'set(%s "%s" CACHE STRING "" FORCE)\n' "$name" "$value" >> "$INIT"
done < <(grep -E '^(CMAKE_BUILD_TYPE|CMAKE_C_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_C_FLAGS|CMAKE_CXX_FLAGS|CUDA_TOOLKIT_ROOT_DIR|HYPRE_ENABLE_[A-Z_]+|HYPRE_WITH_[A-Z_]+|HYPRE_CUDA_SM):' "$OLD/CMakeCache.txt" \
           | grep -vE '^(HYPRE_WITH_GPU_AWARE_MPI|HYPRE_WITH_MPI):')
cat >> "$INIT" <<EOF
set(HYPRE_WITH_MPI "ON" CACHE STRING "" FORCE)
set(HYPRE_WITH_GPU_AWARE_MPI "ON" CACHE STRING "" FORCE)
set(MPI_C_COMPILER "$MPICC" CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER "$MPICXX" CACHE STRING "" FORCE)
EOF
cmake -C "$INIT" -S "$SRC" -B "$NEW" > "$NEW/configure.log" 2>&1 || { tail -20 "$NEW/configure.log"; exit 1; }
grep -q "define HYPRE_WITH_GPU_AWARE_MPI 1" "$NEW/HYPRE_config.h" || { echo "the option did not reach HYPRE_config.h"; exit 1; }
cmake --build "$NEW" -j "${JOBS:-8}" > "$NEW/build.log" 2>&1 || { tail -20 "$NEW/build.log"; exit 1; }
echo "built $NEW/libHYPRE.so"
echo "  preload it:  mpirun ... -x LD_PRELOAD=$NEW/libHYPRE.so ..."
echo "  or replace:  $PYMFEM/lib/python*/site-packages/mfem/external/lib64/libHYPRE.so"

#!/bin/bash
# One GPU per rank, and nothing else.
#
# JAX opens a CUDA context on every device it can see when its backend comes up, not
# only on the one it computes on: on a four-card node a one-rank job held 439 MB on each
# of the three idle cards, and at four ranks every card carried three stray contexts.
# CUDA_VISIBLE_DEVICES removes that, but it has to be set before the interpreter's first
# CUDA call -- which is MPI_Init here, since the MPI is CUDA-aware -- so the reliable
# place is the launcher, before python starts.  Rank k of the node gets device k % ndev.
#
# Measured with hypre on the device at two ranks: only the two assigned cards carried
# memory (7977 and 7947 MB at peak), the other two stayed at 4 MB, same answer.
#
# Cores need nothing extra: OpenMPI's default --bind-to core already confines each rank
# to one core (measured 1.00 cores busy per rank), which is what stops XLA's CPU worker
# pool taking two.  A run started without mpirun should get taskset for the same reason.
#
# Usage:
#   mpirun -n 4 tools/mpirun_pinned.sh python script.py args...
#   HIPPYMFEM_DEVICE=gpu mpirun -n 2 tools/mpirun_pinned.sh python benchmarks/bench_newton_step.py --device cuda
#
# Honors a CUDA_VISIBLE_DEVICES already set by a scheduler: it subsets that list rather
# than the node's, so a SLURM allocation of two cards is split between two ranks, not
# overwritten with device ids the job was never given.
# A run that asked for no device gets none.  What put an all-host benchmark on all
# four cards (426 MB each) was JAX bringing up its CUDA backend because the script
# imported jax before hippymfem could confine it to the CPU; hippymfem now sets the
# platform list either way, and this empties the device list before python starts so
# that nothing else in the process can open a context either.  An explicit
# CUDA_VISIBLE_DEVICES is the caller's choice and is honored; HIPPYMFEM_PIN_GPU=0 too.
case "${HIPPYMFEM_DEVICE:-}" in gpu|cuda|rocm|gpu:0) WANT_GPU=1 ;; *) WANT_GPU=0 ;; esac
case "${HIPPYMFEM_HYPRE_DEVICE:-}" in 1|yes|true|on) WANT_GPU=1 ;; esac
case "${HIPPYMFEM_PIN_GPU:-1}" in 0|no|false|off) PIN=0 ;; *) PIN=1 ;; esac
# The variable of the vendor this node carries: NVIDIA's, or AMD's ROCm runtime's.  A
# scheduler that already set one decides it; otherwise whichever vendor tool answers.
if [ -n "${ROCR_VISIBLE_DEVICES+x}" ] || [ -n "${HIP_VISIBLE_DEVICES+x}" ]; then
  VAR=ROCR_VISIBLE_DEVICES
  [ -n "${ROCR_VISIBLE_DEVICES+x}" ] || ROCR_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
elif [ -z "${CUDA_VISIBLE_DEVICES+x}" ] && ! nvidia-smi -L >/dev/null 2>&1 \
     && rocm-smi --showid 2>/dev/null | grep -q "GPU\["; then
  VAR=ROCR_VISIBLE_DEVICES
else
  VAR=CUDA_VISIBLE_DEVICES
fi
if [ "$WANT_GPU" = 0 ] && [ "$PIN" = 1 ] && [ -z "${!VAR+x}" ]; then
  export "$VAR"=""
  exec "$@"
fi
if [ -n "${!VAR:-}" ]; then
  IFS=',' read -r -a DEVS <<< "${!VAR}"
elif [ "$VAR" = ROCR_VISIBLE_DEVICES ]; then
  mapfile -t DEVS < <(rocm-smi --showid 2>/dev/null | sed -n 's/^GPU\[\([0-9]*\)\].*Device Name.*/\1/p' | sort -un)
else
  mapfile -t DEVS < <(nvidia-smi -L 2>/dev/null | sed -n 's/^GPU \([0-9]*\):.*/\1/p')
fi
NDEV=${#DEVS[@]}
LOCAL=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${MV2_COMM_WORLD_LOCAL_RANK:-0}}}
if [ "$NDEV" -gt 0 ]; then
  # Tell the library how many cards the node really has.  Once the list below is one
  # entry, the library cannot tell four ranks on four cards from four ranks on one,
  # and it sized JAX's share as if they shared: 11% of a card instead of 45% at 128^3
  # with hypre on the device, 913-element chunks, and an out-of-memory on a 2 GB
  # accumulator while the card held 10 GB.
  export HIPPYMFEM_NODE_GPUS="$NDEV"
  export "$VAR"="${DEVS[$(( LOCAL % NDEV ))]}"
fi
exec "$@"

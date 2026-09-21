#!/bin/bash
# Cross-validate hIPPyMFEM against hIPPYlibx on the shared benchmark.
#
#   ./validation/run_validation.sh [nx] [ranks]
#
# The two libraries live in different conda environments, so each driver runs
# under its own interpreter.  The data file is written by whichever driver runs
# first and read by the other, so the noise realization is bit-identical.
set -u
cd "$(dirname "$0")/.."
NX="${1:-12}"
RANKS="${2:-1}"
MFEM_PY="${MFEM_PY:-python}"          # an interpreter with PyMFEM (MPI) and hIPPyMFEM
FENICSX_PY="${FENICSX_PY:-python}"    # an interpreter with dolfinx and hIPPYlibx
OUT=validation/out

mkdir -p "$OUT"
rm -f "$OUT"/data.json "$OUT"/mfem.json "$OUT"/hippylibx.json

run() {  # run <interpreter> <script> <outfile>
  if [ "$RANKS" = "1" ]; then "$1" "$2" --out "$3" --nx "$NX"
  else mpirun -n "$RANKS" "$1" "$2" --out "$3" --nx "$NX"; fi
}

echo "### hIPPyMFEM (nx=$NX, ranks=$RANKS) ###"
run "$MFEM_PY" validation/run_hippymfem.py "$OUT/mfem.json" || exit 1
echo
echo "### hIPPYlibx (nx=$NX, ranks=$RANKS) ###"
run "$FENICSX_PY" validation/run_hippylibx.py "$OUT/hippylibx.json" || exit 1
echo
"$MFEM_PY" validation/compare.py "$OUT/mfem.json" "$OUT/hippylibx.json" \
    | tee "$OUT/report_nx${NX}_np${RANKS}.txt"

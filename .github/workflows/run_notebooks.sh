#!/bin/bash
# Execute every tutorial notebook, writing the executed copies to a scratch directory:
# a cell that raises fails the run.  Used by CI; runs the same way locally.
set -euo pipefail
cd "$(dirname "$0")/../../tutorial"
OUT="$(mktemp -d)"
# The names are zero padded (01_ ... 12_), so they sort in order; sort -V keeps that true
# if an unpadded name ever appears, and the glob is *.ipynb so nothing is silently skipped
# (the old [0-9]_*.ipynb glob skipped 10_FacetsAndDG without a word).
for nb in $(ls *.ipynb | sort -V); do
  echo "=== $nb ==="
  start=$(date +%s)
  jupyter nbconvert --to notebook --execute --output-dir "$OUT" \
    --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 "$nb"
  echo "    $(( $(date +%s) - start )) s"
done

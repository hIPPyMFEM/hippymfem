#!/bin/bash
# Run every application on small inputs, on one and two ranks.  A smoke test that the
# drivers still run end to end; the test suites check the numbers.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="$(mktemp -d)"
MPIEXEC="${MPIEXEC:-mpirun}"
export MPLBACKEND=Agg

run() {   # run <ranks> <python arguments...>
  local n=$1; shift
  echo "=== $* on $n rank(s) ==="
  local start=$(date +%s)
  if [ "$n" = "1" ]; then
    python "$@" > "$OUT/log" 2>&1 || { tail -40 "$OUT/log"; exit 1; }
  else
    $MPIEXEC -n "$n" ${MPIEXEC_FLAGS:-} python "$@" > "$OUT/log" 2>&1 || { tail -40 "$OUT/log"; exit 1; }
  fi
  echo "    ok, $(( $(date +%s) - start )) s"
}

for n in 1 2; do
  run $n applications/poisson/model_subsurf.py --nx 16 --ny 16 --neig 10 --nsamples 1 --out "$OUT/poisson"
  run $n applications/ad_diff/model_ad_diff.py --nx 16 --nt 8 --ntargets 40 --neig 10 --out "$OUT/ad_diff"
  run $n applications/boundary/model_robin.py --nx 12 --ntargets 20 --nmodes 10
  run $n applications/forward_uq/model_subsurf_effperm.py --nx 12 --neig 10 --nsamples 20 --out "$OUT/forward_uq"
  run $n applications/mcmc/model_subsurf_mcmc.py --nx 12 --ntargets 20 --neig 10 --nsamples 100 \
      --burn-in 20 --tune 20 --tune-steps 0.3 --out "$OUT/mcmc"
  run $n applications/dg/model_transport_dg.py --nx 12 --ntargets 20 --nmodes 10 --out "$OUT/dg"
done

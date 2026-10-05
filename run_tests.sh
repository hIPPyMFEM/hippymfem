#!/bin/bash
# Run the hIPPyMFEM test suite.  Usage: ./run_tests.sh [ranks...] (default: 1)
#
#   ./run_tests.sh              # 1 rank, all suites
#   ./run_tests.sh 1 2 4        # and on 2 and 4 ranks
#   PHASES=test_kernels ./run_tests.sh 4     # one suite only
#   HIPPYMFEM_CUDA_PYMFEM=/path/to/cuda/site-packages ./run_tests.sh 1 2   # + hypre on the device
PY="${PY:-python}"   # the interpreter that has PyMFEM: PY=/path/to/python ./run_tests.sh
cd "$(dirname "$0")"
PHASES="${PHASES:-test_vectors test_kernels test_solves test_modeling test_optimization test_timedependent test_uq test_assembly test_solvers test_boundary test_facets test_vectorfe test_nb}"
RANKS="${@:-1}"
WRAP=""
status=0

# run_suite LABEL SUITE NRANKS KEEP [ENV=VALUE ...]: run one suite and print its failed
# checks, its count of failures and the lines matching KEEP.  $WRAP goes between the
# launcher and python.
run_suite() {
  local label=$1 ph=$2 n=$3 keep=$4
  shift 4
  local launch=()
  [ "$n" != "1" ] && launch=(mpirun -n "$n")
  echo "=== $label on $n rank(s) ==="
  env "$@" timeout 2400 "${launch[@]}" $WRAP $PY -m hippymfem.test.$ph 2>&1 | grep -E "^  \[FAIL|FAILURES$keep"
  [ ${PIPESTATUS[0]} -ne 0 ] && { echo "  (suite did not finish cleanly)"; status=1; }
}

for n in $RANKS; do
  for ph in $PHASES; do run_suite $ph $ph $n ""; done
done
# test_solvers skips the PETSc bridge unless petsc4py is imported before PyMFEM,
# so run it once more with that enabled.  A PETSc-less environment still reports
# the skip and its reason rather than passing silently.
if [ -z "$SKIP_PETSC" ]; then
  for n in $RANKS; do run_suite "test_solvers (PETSc)" test_solvers $n "|skipped:" HIPPYMFEM_PETSC=1; done
fi
# The GPU suite needs both JAX backends live, which requires the environment
# variable to be set before import; it skips with a printed reason when no GPU is
# visible, so running it unconditionally is safe.
if [ -z "$SKIP_GPU" ]; then
  for n in $RANKS; do run_suite test_gpu test_gpu $n "|skipped:" HIPPYMFEM_DEVICE=gpu; done
fi
# hypre on the device needs the CUDA build of PyMFEM, which lives outside the
# default environment: name its site-packages in HIPPYMFEM_CUDA_PYMFEM to run that
# suite.  The pinning wrapper gives each rank one card, and HIPPYMFEM_HYPRE_DEVICE
# has to be set before import, which is why both are set here and not in the test.
if [ -n "$HIPPYMFEM_CUDA_PYMFEM" ]; then
  WRAP=tools/mpirun_pinned.sh
  for n in $RANKS; do
    run_suite test_device test_device $n "|nothing to test" \
      PYTHONPATH="$HIPPYMFEM_CUDA_PYMFEM${PYTHONPATH:+:$PYTHONPATH}" HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1
  done
fi
if [ $status -eq 0 ]; then echo "ALL SUITES PASSED"; else echo "SOME SUITES FAILED"; fi
exit $status

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Run hIPPyMFEM's test suites under pytest.

Each suite in ``hippymfem/test`` is an MPI program that holds the library against an
external reference (MFEM's own integrators, dense eigendecompositions, exact Gaussian
posteriors, finite differences) and ends with a ``FAILURES: n`` line.  This module
runs every suite as a subprocess, on each rank count in ``HIPPYMFEM_TEST_RANKS``
(default ``1,2``), and fails with the suite's own report when any check fails.

    pytest                                   # every suite on 1 and 2 ranks
    HIPPYMFEM_TEST_RANKS=1,2,4 pytest -k kernels
    MPIEXEC_FLAGS="--oversubscribe" pytest   # more ranks than cores

The GPU and hypre-on-device suites need hardware and builds that CI does not have;
``run_tests.sh`` runs them where they apply.
"""

import os
import re
import shutil
import subprocess
import sys

import pytest

SUITES = [
    "test_vectors",         # vectors, operators, the partition-independent RNG
    "test_kernels",         # AD blocks against MFEM's integrators
    "test_solves",          # forward, adjoint and incremental solves
    "test_modeling",        # priors, observations, misfits, modelVerify, Hessian
    "test_optimization",    # Steihaug CG, randomized eigensolvers, BFGS, Laplace
    "test_timedependent",   # time-dependent inversion
    "test_uq",              # MCMC against exact posteriors, forward UQ
    "test_assembly",        # direct-CSR assembly against the callback route
    "test_solvers",         # exact parallel solves
    "test_boundary",        # boundary densities against MFEM's integrators
    "test_facets",          # interior facet terms and DG
    "test_vectorfe",        # H(curl) and H(div)
    "test_nb",              # plotting helpers
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = [int(r) for r in os.environ.get("HIPPYMFEM_TEST_RANKS", "1,2").split(",")
         if r.strip()]
TIMEOUT = int(os.environ.get("HIPPYMFEM_TEST_TIMEOUT", "2400"))


def _launcher(nranks):
    if nranks == 1:
        return []
    mpiexec = (os.environ.get("MPIEXEC") or shutil.which("mpiexec")
               or shutil.which("mpirun"))
    if mpiexec is None:
        pytest.skip("no mpiexec on PATH for a %d-rank run" % nranks)
    return [mpiexec, "-n", str(nranks)] + os.environ.get("MPIEXEC_FLAGS", "").split()


@pytest.mark.parametrize("nranks", RANKS, ids=lambda n: "np%d" % n)
@pytest.mark.parametrize("suite", SUITES)
def test_suite(suite, nranks):
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = _launcher(nranks) + [sys.executable, "-m", "hippymfem.test." + suite]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=TIMEOUT)
    out = proc.stdout + proc.stderr
    failed = [line for line in out.splitlines() if line.startswith("  [FAIL")]
    summary = re.findall(r"^FAILURES: (\d+)", out, flags=re.M)
    ok = proc.returncode == 0 and summary and int(summary[-1]) == 0 and not failed
    assert ok, ("%s on %d rank(s) (exit %d):\n%s"
                % (suite, nranks, proc.returncode,
                   "\n".join(failed) if failed else out[-6000:]))

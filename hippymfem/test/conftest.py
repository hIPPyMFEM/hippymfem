# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Make a suite run under pytest fail when one of its checks does.

The suites are written to be run as scripts, under ``mpirun``: a check that fails is
recorded and printed and the suite carries on, so that one run reports every failure
rather than the first.  ``tests/test_suites.py`` launches them that way and reads the
summary.  Pointing pytest at a suite module instead collects its ``test_`` functions
directly, and those return normally whether their checks passed or not, which would
report a pass on a failing suite.  This fixture closes that gap by looking at what the
module recorded while the function ran.

The failure surfaces as an error in the test's teardown, naming the checks that were
recorded, which is what a fixture can do on any pytest since 7.  One rank only: a real
run still goes through ``tests/test_suites.py``.
"""

import pytest


@pytest.fixture(autouse=True)
def _recorded_checks_must_pass(request):
    module = request.module
    before = len(getattr(module, "FAILS", ()))
    yield
    failures = getattr(module, "FAILS", ())
    new = list(failures)[before:]
    if new:
        pytest.fail("%d check(s) failed: %s" % (len(new), "; ".join(map(str, new))))

#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Scan for MPI collectives called inside rank-0-only blocks.

That combination deadlocks: rank 0 enters the reduction and no other rank joins
it.  The symptom is a run that prints nothing and sits at partial CPU -- never an
exception -- so it is worth checking mechanically.  Three real instances of this
were found in hIPPyMFEM by running this scan, one of them in the library itself.

The guard is matched in any boolean combination (``if verbose and rank == 0:``
counts), and the collectives include the non-obvious ones: ``Model.cost`` reduces
the misfit and solves with the prior, ``PointwiseObservation.mult`` applies the
prolongation, and every ``KrylovSolver.solve`` is collective.

Usage::

    python tools/check_collectives.py [paths...]      # default: the whole tree

Exits nonzero if anything is found.
"""

import os
import re
import sys

GUARD = re.compile(r"^(\s*)if\s+[^:]*\b(?:RANK|rank)\s*==\s*0\b[^:]*:\s*$")

COLLECTIVE = re.compile(
    r"\.(inner|norm|sum|max|min|dot)\s*\(" r"|\.cost\s*\(" r"|\.solve\s*\("
    r"|\.mult\s*\(" r"|\.multTranspose\s*\(" r"|allreduce|allgather"
    r"|Allreduce|Allgather|Barrier|gather_to_zero|local_values"
    r"|evalGradient|pointwise_variance|\.trace\s*\(|\.sample\s*\("
)

#: expressions that merely look like collectives (numpy, dicts, plain python)
SAFE = re.compile(
    r"np\.|numpy\.|math\.|float\(|int\(|len\(|\.size\b|\.shape\b"
    r"|\.items\(|\.keys\(|\.values\(|\[[\"']"
)

DEFAULT_PATHS = ("hippymfem", "applications", "validation", "tools")


def scan_file(path):
    findings = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        m = GUARD.match(lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            code = line.split("#")[0]
            if COLLECTIVE.search(code) and not SAFE.search(code):
                findings.append((j + 1, line.strip()))
            j += 1
        i = j
    return findings


def main(argv):
    paths = argv[1:] or list(DEFAULT_PATHS)
    total = 0
    for root in paths:
        if os.path.isfile(root):
            files = [root]
        else:
            files = [os.path.join(d, f)
                     for d, _, fs in os.walk(root) for f in sorted(fs)
                     if f.endswith(".py")]
        for f in sorted(files):
            for lineno, text in scan_file(f):
                print("%s:%d: %s" % (f, lineno, text[:100]))
                total += 1
    if total:
        print("\n%d possible collective(s) inside a rank-0 guard." % total)
        print("Compute the value on every rank, then print it on one.")
        return 1
    print("clean: no collectives found inside rank-0 guards")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

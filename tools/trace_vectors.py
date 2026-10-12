#!/usr/bin/env python
"""Where the vector arithmetic of a Newton solve runs: on the device or on the host.

    mpirun ... python tools/trace_vectors.py SCRIPT.py [the script's arguments]

(for instance ``benchmarks/bench_newton_device.py --steps 2``: any script that calls
``ReducedSpaceNewtonCG.solve``).

A ParVector's arithmetic runs on the device once the vector has been handed to MFEM or hypre
(its ``hypre`` property) and on the host until then, where every operand that is current on
the device only is first copied back.  This times every operation of ParVector inside the
Newton steps (ReducedSpaceNewtonCG.solve) on rank 0, by operation, by where it ran and by the
line that called it, and prints the largest at the end.  With work queued on the device the
time of a copy to the host includes the wait for what was queued before it.
"""
import collections
import os
import runpy
import sys
import time

import hippymfem as hm                                              # noqa: E402 (before mpi4py and jax)
from mpi4py import MPI                                              # noqa: E402

from hippymfem.common import parvector as pv

T = collections.defaultdict(lambda: [0, 0.0, 0])
ACTIVE = [False]
DEPTH = [0]


def wrap(name):
    real = getattr(pv.ParVector, name)

    def traced(self, *a, **k):
        if not ACTIVE[0]:
            return real(self, *a, **k)
        dev = bool(self._dev)
        outer = DEPTH[0] == 0
        DEPTH[0] += 1
        t0 = time.perf_counter()
        try:
            return real(self, *a, **k)
        finally:
            dt = time.perf_counter() - t0
            DEPTH[0] -= 1
            fr = sys._getframe(1)
            while fr is not None and fr.f_code.co_filename.endswith(("parvector.py", "trace_vectors.py")):
                fr = fr.f_back
            where = "%s:%d %s" % (os.path.basename(fr.f_code.co_filename), fr.f_lineno, fr.f_code.co_name) if fr else "?"
            t = T[(name, "device" if dev else "host", "outer" if outer else "inner", where)]
            t[0] += 1
            t[1] += dt
            t[2] = max(t[2], int(self._layout.local_size))

    setattr(pv.ParVector, name, traced)


for _name in ("_host", "_host_read", "_host_write", "axpy", "aypx", "axpby", "inner", "norm", "scale", "assign",
              "copy", "zero", "set", "pointwise_mult", "duplicate"):
    wrap(_name)

_solve = hm.ReducedSpaceNewtonCG.solve


def solve(self, *a, **k):
    ACTIVE[0] = True
    t0 = time.perf_counter()
    try:
        return _solve(self, *a, **k)
    finally:
        ACTIVE[0] = False
        if MPI.COMM_WORLD.rank == 0:
            total = time.perf_counter() - t0
            rows = sorted(T.items(), key=lambda kv: -kv[1][1])
            outer = sum(v[1] for k, v in rows if k[2] == "outer")
            host = sum(v[1] for k, v in rows if k[2] == "outer" and k[1] == "host")
            print("vector operations in the Newton steps (rank 0): %.3f s of %.2f s, %.3f s of them on the host"
                  % (outer, total, host), flush=True)
            print("  %-14s %-6s %-5s %7s %9s %9s %10s  %s" % ("operation", "where", "level", "calls", "seconds", "ms a call",
                                                              "entries", "called from"))
            for (name, loc, lev, where), (n, s, size) in rows[:60]:
                print("  %-14s %-6s %-5s %7d %9.3f %9.3f %10d  %s" % (name, loc, lev, n, s, 1e3 * s / n, size, where), flush=True)


hm.ReducedSpaceNewtonCG.solve = solve

if __name__ == "__main__":
    bench = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(bench, run_name="__main__")

#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""A performance check against a baseline of this machine: two Newton-CG steps of the model
problem (``bench_newton_device.py``) at a few sizes, each compared with the record of a run
that was right.

    python benchmarks/perf_check.py --record             # write the baseline of this machine
    python benchmarks/perf_check.py                      # compare with it (exit 1 if worse)

A size fails when its two steps take more than ``1 + tolerance`` times the baseline, when the
Newton or CG counts differ from it, or when the cost functional differs from it by more than
1e-6 of its size (the same mesh, data and settings give the same path).  Faster is reported,
not failed; record again to keep it.  The baseline is ``perf_baselines/<machine>.json``, the
machine named by ``--machine``, ``HIPPYMFEM_PERF_MACHINE`` or the host name; a timing is
the least of ``--repeat`` runs.  It holds one record per configuration and rank count, and
``--record`` adds or replaces the one of this run.

The runs are started by ``--launcher`` (default ``HIPPYMFEM_LAUNCHER`` or ``mpirun -n {ranks}``;
on one rank without one, the interpreter itself), with the environment of this process:
set the device there, as for the benchmark (``HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1``,
a profile, the single-precision hypre).  A record keeps the configuration it was made in:
every ``HIPPYMFEM_*`` variable but those that only say where things are or what to print
(:data:`NOT_CONFIG`), ``XLA_FLAGS``, and the options after ``--``.  A run is compared with
the record of the same configuration only.

Exit status: 0 when every size is within the tolerance on the recorded path, 1 when one is
not, 2 when there is nothing to compare with (no baseline, or no record of this
configuration and rank count), which a scheduled job can report without failing.
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_newton_device.py")
#: ``HIPPYMFEM_*`` variables that say where things are, how a run is launched or what it
#: prints, not what it computes: every other one is part of a record's configuration
NOT_CONFIG = ("HIPPYMFEM_ABORT_ON_ERROR", "HIPPYMFEM_CUDA_PYMFEM", "HIPPYMFEM_GPU_ENV",
              "HIPPYMFEM_GPU_PYTHON", "HIPPYMFEM_JAX_CACHE", "HIPPYMFEM_JAX_CACHE_MIN_SECONDS",
              "HIPPYMFEM_LAUNCHER", "HIPPYMFEM_NODE_GPUS", "HIPPYMFEM_PERF_MACHINE",
              "HIPPYMFEM_PATTERN_TIMING", "HIPPYMFEM_PMG_TIMERS", "HIPPYMFEM_PMG_TRACE",
              "HIPPYMFEM_TEST_RANKS")
#: the exit status when there is no record to compare with
NO_RECORD = 2


def configuration(bench_args):
    """What makes a run comparable with a record: the variables that change what the
    benchmark computes or how, and the benchmark's own options."""
    env = {k: v for k, v in sorted(os.environ.items())
           if (k.startswith("HIPPYMFEM_") and k not in NOT_CONFIG) or k == "XLA_FLAGS"}
    return {"env": env, "bench_args": list(bench_args)}


def records_of(base):
    """The records of a baseline file; one written before records were kept per
    configuration is a single record whose configuration matches nothing now."""
    if "records" in base:
        return list(base["records"])
    return [{"ranks": base.get("ranks"), "config": {"legacy": base.get("config")},
             "runs": base.get("runs", {})}]


def run_bench(n, ranks, launcher, extra):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "r.json")
        cmd = [sys.executable, BENCH, "--n", str(n), "--steps", "2", "--out", out] + extra
        if launcher or ranks > 1:
            cmd = shlex.split((launcher or "mpirun -n {ranks}").format(ranks=ranks)) + cmd
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0 or not os.path.exists(out):
            sys.stdout.write(p.stdout[-3000:])
            raise SystemExit("perf_check: the benchmark at %d^3 on %d rank(s) failed (exit %d)"
                             % (n, ranks, p.returncode))
        with open(out) as f:
            return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--machine", default=os.environ.get("HIPPYMFEM_PERF_MACHINE") or socket.gethostname().split(".")[0])
    ap.add_argument("--sizes", default="32,64", help="mesh sizes n of n^3 elements, comma-separated")
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--launcher", default=os.environ.get("HIPPYMFEM_LAUNCHER", ""))
    ap.add_argument("--record", action="store_true", help="write the baseline instead of comparing")
    ap.add_argument("--baseline-dir", default=os.path.join(HERE, "perf_baselines"))
    ap.add_argument("bench_args", nargs="*", help="after --: options for bench_newton_device.py")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if not sizes:
        raise SystemExit("perf_check: --sizes names no size")
    config = configuration(args.bench_args)
    path = os.path.join(args.baseline_dir, "%s.json" % args.machine)
    records = []
    if os.path.exists(path):
        with open(path) as f:
            records = records_of(json.load(f))
    base = None
    if not args.record:
        if not records:
            print("perf_check: no baseline %s; record one with --record" % path)
            return NO_RECORD
        mine = [r for r in records if r.get("config") == config and r.get("ranks") == args.ranks]
        if not mine:
            print("perf_check: %s has no record of this configuration on %d rank(s); it has %d "
                  "other(s).  This run: %s" % (path, args.ranks, len(records), json.dumps(config)))
            return NO_RECORD
        base = mine[0]
    runs, failed = {}, []
    for n in sizes:
        best = None
        for _ in range(max(1, args.repeat)):
            r = run_bench(n, args.ranks, args.launcher, args.bench_args)
            if best is None or r["t_steps"] < best["t_steps"]:
                best = r
        rec = {k: best[k] for k in ("t_steps", "t_fwd", "t_adj", "t_hess_blocks_warm", "t_hess_apply",
                                    "newton_it", "cg_it", "J", "tdofs") if k in best}
        runs[str(n)] = rec
        if base is None:
            print("%4d^3: two steps %.3f s, %d Newton, %d CG, J %.10e" % (n, rec["t_steps"], rec["newton_it"],
                                                                          rec["cg_it"], rec["J"]))
            continue
        b = base["runs"].get(str(n))
        if b is None:
            failed.append("%d^3: not in the baseline" % n)
            continue
        ratio = rec["t_steps"] / b["t_steps"]
        same_path = rec["newton_it"] == b["newton_it"] and rec["cg_it"] == b["cg_it"]
        same_cost = abs(rec["J"] - b["J"]) <= 1e-6 * max(abs(b["J"]), 1e-300)
        verdict = "ok" if (ratio <= 1.0 + args.tolerance and same_path and same_cost) else "FAIL"
        print("%4d^3: two steps %.3f s against %.3f s (%+.1f %%), %d/%d Newton, %d/%d CG, J %.10e against "
              "%.10e  [%s]" % (n, rec["t_steps"], b["t_steps"], 100.0 * (ratio - 1.0), rec["newton_it"],
                               b["newton_it"], rec["cg_it"], b["cg_it"], rec["J"], b["J"], verdict))
        if verdict == "FAIL":
            failed.append("%d^3" % n)
    if args.record:
        os.makedirs(args.baseline_dir, exist_ok=True)
        kept = [r for r in records if not (r.get("config") == config and r.get("ranks") == args.ranks)]
        kept.append({"ranks": args.ranks, "config": config, "runs": runs})
        with open(path, "w") as f:
            json.dump({"machine": args.machine, "records": kept}, f, indent=1)
        print("perf_check: wrote %s (%d record(s))" % (path, len(kept)))
        return 0
    if failed:
        print("perf_check: FAILED at %s" % ", ".join(failed))
        return 1
    print("perf_check: every size within %.0f %% of the baseline, with the same path" % (100 * args.tolerance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

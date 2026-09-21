#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Tables from the records that ``bench_scaling.py`` writes.

    python benchmarks/analyze_scaling.py                 # scaling_* records
    python benchmarks/analyze_scaling.py --knobs         # knob_* records as well

Strong scaling is reported two ways, because the two hardware kinds lose efficiency for
opposite reasons: the *iteration count* of an incremental solve (how much the
preconditioner weakens on more subdomains) and the *time per iteration* (launch latency,
halo exchanges and CG's reductions, none of which shrink with the subdomain).
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "local", pattern))):
        d = json.load(open(f))
        out[(d["n"], d["ranks"], d.get("tag", ""))] = d
    return out


def ms_per_iter(d):
    return 1e3 * d["op"]["solveFwdInc"] / max(d["it_fwd_inc"], 1)


def scaling(recs, label, unit, counts):
    if not recs:
        return
    print("\n%s\n%s" % (label, "-" * len(label)))
    print("%-5s %-6s %12s %10s %5s %10s %9s %9s"
          % ("n", unit, "dofs/" + unit[:-1], "action[s]", "it", "ms/iter", "eff(t/it)", "eff(act)"))
    for n in sorted({k[0] for k in recs}):
        base = None
        for r in counts:
            d = recs.get((n, r, ""))
            if d is None:
                continue
            if base is None:
                base, br = d, r
            e_it = (ms_per_iter(base) / ms_per_iter(d)) / (r / br) * 100
            e_ac = (base["t_action"] / d["t_action"]) / (r / br) * 100
            print("%-5d %-6d %12s %10.4f %5d %10.2f %8.0f%% %8.0f%%"
                  % (n, r, "{:,}".format((2 * n + 1) ** 3 // r), d["t_action"], d["it_fwd_inc"],
                     ms_per_iter(d), e_it, e_ac))


def composition(recs, label):
    if not recs:
        return
    print("\n%s\n%s" % (label, "-" * len(label)))
    for k in sorted(recs):
        if k[2]:
            continue
        d = recs[k]
        s = d["op"]["solveFwdInc"] + d["op"]["solveAdjInc"]
        print("  n=%-4d %d: action %9.4f s | two incremental solves %5.1f %% | prior R %4.1f %% | "
              "seven sparse products %4.1f %%"
              % (k[0], k[1], d["t_action"], 100 * s / d["t_action"],
                 100 * d["op"]["applyR"] / d["t_action"],
                 100 * (d["t_action"] - s - d["op"]["applyR"]) / d["t_action"]))


def knobs(recs):
    if not recs:
        return
    print("\nBoomerAMG settings against the default, one incremental solve\n"
          "-----------------------------------------------------------")
    print("%-5s %-6s %-10s %6s %10s %11s %10s" % ("n", "cards", "setting", "it", "ms/iter",
                                                  "solve[s]", "vs base"))
    for n in sorted({k[0] for k in recs}):
        for r in sorted({k[1] for k in recs if k[0] == n}):
            base = recs.get((n, r, "base"))
            for k in sorted(recs):
                if k[0] != n or k[1] != r:
                    continue
                d = recs[k]
                rel = base["op"]["solveFwdInc"] / d["op"]["solveFwdInc"] if base else float("nan")
                print("%-5d %-6d %-10s %6d %10.2f %11.4f %9.2fx"
                      % (n, r, k[2], d["it_fwd_inc"], ms_per_iter(d), d["op"]["solveFwdInc"], rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--knobs", action="store_true", help="also table the knob_* records")
    args = ap.parse_args()
    gpu = load("scaling_l40s_n*_r*.json")
    host = load("scaling_host_n*_r*.json")
    scaling(gpu, "Strong scaling on L40S GPUs", "cards", (1, 2, 4, 8))
    scaling(host, "Strong scaling on host cores", "ranks", (1, 2, 4, 8, 16, 32))
    print("\nIterations of one incremental solve (how the preconditioner itself scales)")
    for recs, what in ((host, "host"), (gpu, "L40S")):
        for n in sorted({k[0] for k in recs}):
            row = "  n=%-4d %-5s: " % (n, what)
            row += "  ".join("%d -> %d it" % (k[1], recs[k]["it_fwd_inc"])
                             for k in sorted(recs) if k[0] == n and not k[2])
            print(row)
    composition(gpu, "What a reduced-Hessian action is made of, on the GPUs")
    composition(host, "What a reduced-Hessian action is made of, on the host")
    if args.knobs:
        knobs(load("knob_l40s_n*_r*_*.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

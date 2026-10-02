#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Replay a trace of hypre's device allocations (``hypre_pool_trace.py``) under pool policies and
count what each would ask of the driver.

A policy says, when a block is freed, whether the pool keeps it: up to ``scope_cap`` bytes
of blocks of at most ``scope_block`` bytes while a BoomerAMG setup is pending (between the
solver's open and close), and up to ``keep_cap`` bytes of blocks of at most ``keep_block``
bytes otherwise.  At the end of a setup the pool is cut down to what it may keep,
smallest blocks first.  With ``evict`` a freed block displaces larger ones when the pool
is full.  A request is served by a kept block of the same size class or the next (at
most 1.19 times the size asked for).  These are the rules of
``hippymfem.common.mfemconfig.HyprePool``, whose default is the policy marked so below.

The trace has the time of every driver call it recorded (no pool, so every request was
one).  The cost of a call that a policy still makes is taken from the mean time of the
recorded calls of its size class, allocations and frees separately.  Reported per policy,
for the steady-state solves (all but the first), per forward solve with its Hessian point:

  calls      driver allocations + frees
  ms         their time by the table above
  held       the most the pool holds outside a setup
  over       how far (blocks in use + blocks kept) exceeds the peak of the blocks in use
             without a pool: the pool's cost at the memory peak
"""
import argparse
import collections
import math
import pickle

CLASSES = 8.0


def cls(size):
    return int(CLASSES * math.log2(size)) if size > 1 else 0


def phase_of(e):
    return e[3] if e[0] == "a" else (e[2] if e[0] == "f" else e[1])


def cost_tables(events):
    """Mean seconds of a driver allocation and of a driver free, per size class."""
    size, a, f = {}, collections.defaultdict(list), collections.defaultdict(list)
    for e in events:
        if e[0] == "a":
            size[e[1]] = e[2]
            if len(e) > 4:
                a[cls(e[2])].append(e[4])
        elif e[0] == "f" and len(e) > 3 and e[1] in size:
            f[cls(size[e[1]])].append(e[3])
    mean = lambda d: {k: sum(v) / len(v) for k, v in d.items()}      # noqa: E731
    return mean(a), mean(f)


class Sim:
    def __init__(self, costs, scope_cap=0, keep_cap=0, keep_block=None, scope_block=None,
                 evict=False, keep_share=None):
        self.ca, self.cf = costs
        self.evict = evict
        self.keep_share = keep_share       # of the most in use: HyprePool.KEEP_SHARE
        self.scope_cap, self.keep_cap = scope_cap, keep_cap
        self.keep_block = keep_cap if keep_block is None else keep_block
        self.scope_block = scope_cap if scope_block is None else scope_block
        self.depth = 0
        self.bins = collections.defaultdict(list)
        self.size, self.req = {}, {}
        self.cached = self.in_use = self.req_in_use = 0
        self.peak_total = self.peak_req = self.held = 0
        self.calls = collections.Counter()  # phase -> driver calls
        self.time = collections.Counter()   # phase -> seconds of them

    def _driver(self, phase, table, size):
        self.calls[phase] += 1
        c = cls(size)
        if c not in table and table:
            c = min(table, key=lambda k: abs(k - c))
        self.time[phase] += table.get(c, 0.0)

    def alloc(self, i, size, phase):
        c = cls(size)
        got = None
        for k in (c, c + 1):
            blocks = self.bins.get(k)
            if blocks:
                for j in range(len(blocks) - 1, -1, -1):
                    if blocks[j] >= size:
                        got = blocks[j]
                        blocks[j] = blocks[-1]
                        blocks.pop()
                        break
            if got is not None:
                break
        if got is None:
            got = size
            self._driver(phase, self.ca, size)
        else:
            self.cached -= got
        self.size[i], self.req[i] = got, size
        self.in_use += got
        self.req_in_use += size
        self.peak_total = max(self.peak_total, self.in_use + self.cached)
        self.peak_req = max(self.peak_req, self.req_in_use)

    def free(self, i, phase):
        s = self.size.pop(i)
        self.in_use -= s
        self.req_in_use -= self.req.pop(i)
        cap, block = ((self.scope_cap, self.scope_block) if self.depth > 0
                      else (self.keep_cap, self.keep_block))
        if s <= block and self.cached + s > cap and self.evict:
            # full: give up larger blocks for this one, largest first
            c = cls(s)
            while self.cached + s > cap:
                top = max((k for k, b in self.bins.items() if b), default=None)
                if top is None or top <= c:
                    break
                big = self.bins[top].pop()
                self.cached -= big
                self._driver(phase, self.cf, big)
        if s <= block and self.cached + s <= cap:
            self.bins[cls(s)].append(s)
            self.cached += s
            if self.depth == 0:
                self.held = max(self.held, self.cached)
        else:
            self._driver(phase, self.cf, s)

    def open(self):
        self.depth += 1

    def close(self, phase):
        # as the library: the first solve of any solver ends the scope
        self.depth = 0
        cap = self.keep_cap
        if self.keep_share is not None:
            cap = min(cap, int(self.keep_share * self.peak_req))
        blocks = sorted(s for b in self.bins.values() for s in b)
        self.bins.clear()
        kept = 0
        for s in blocks:
            if s <= self.keep_block and kept + s <= cap:
                self.bins[cls(s)].append(s)
                kept += s
            else:
                self._driver(phase, self.cf, s)
        self.cached = kept
        self.held = max(self.held, self.cached)


def run(events, costs, **policy):
    sim = Sim(costs, **policy)
    for e in events:
        if e[0] == "a":
            sim.alloc(e[1], e[2], e[3])
        elif e[0] == "f":
            sim.free(e[1], e[2])
        elif e[0] == "open":
            sim.open()
        else:
            sim.close(e[1])
    return sim


MB, GB = 2 ** 20, 2 ** 30
POLICIES = [
    ("no pool", dict()),
    ("during a setup, 1 GiB", dict(scope_cap=GB)),
    ("during a setup, 4 GiB", dict(scope_cap=4 * GB)),
    ("always, 1 GiB", dict(scope_cap=GB, keep_cap=GB)),
    ("always, 4 GiB", dict(scope_cap=4 * GB, keep_cap=4 * GB)),
    ("setup 1 GiB + keep 128 MB of blocks <= 1 MB", dict(scope_cap=GB, keep_cap=128 * MB, keep_block=MB)),
    ("setup 1 GiB + keep 256 MB of blocks <= 4 MB", dict(scope_cap=GB, keep_cap=256 * MB, keep_block=4 * MB)),
    ("setup 1 GiB + keep 512 MB of blocks <= 16 MB", dict(scope_cap=GB, keep_cap=512 * MB, keep_block=16 * MB)),
    ("setup 2 GiB + keep 128 MB of blocks <= 1 MB", dict(scope_cap=2 * GB, keep_cap=128 * MB, keep_block=MB)),
    ("setup 4 GiB + keep 128 MB of blocks <= 1 MB", dict(scope_cap=4 * GB, keep_cap=128 * MB, keep_block=MB)),
    ("setup 4 GiB + keep 1 GiB", dict(scope_cap=4 * GB, keep_cap=GB)),
    ("setup 8 GiB + keep 128 MB of blocks <= 1 MB", dict(scope_cap=8 * GB, keep_cap=128 * MB, keep_block=MB)),
    ("setup 1 GiB + keep 128 MB <= 16 MB, evict", dict(scope_cap=GB, keep_cap=128 * MB, keep_block=16 * MB, evict=True)),
    ("setup 1 GiB + keep 256 MB <= 16 MB, evict", dict(scope_cap=GB, keep_cap=256 * MB, keep_block=16 * MB, evict=True)),
    ("setup 1 GiB + keep 256 MB <= 64 MB, evict", dict(scope_cap=GB, keep_cap=256 * MB, keep_block=64 * MB, evict=True)),
    ("setup 1 GiB + keep 512 MB <= 16 MB, evict", dict(scope_cap=GB, keep_cap=512 * MB, keep_block=16 * MB, evict=True)),
    ("setup 1 GiB + keep 512 MB <= 64 MB, evict", dict(scope_cap=GB, keep_cap=512 * MB, keep_block=64 * MB, evict=True)),
    ("setup 1 GiB + keep 512 MB, evict", dict(scope_cap=GB, keep_cap=512 * MB, evict=True)),
    ("the library's default (the line above, 1/4 of the peak)",
     dict(scope_cap=GB, keep_cap=512 * MB, evict=True, keep_share=0.25)),
    ("setup 1 GiB + keep 1 GiB, evict", dict(scope_cap=GB, keep_cap=GB, evict=True)),
    ("setup 2 GiB + keep 512 MB <= 64 MB, evict", dict(scope_cap=2 * GB, keep_cap=512 * MB, keep_block=64 * MB, evict=True)),
    ("setup 2 GiB + keep 1 GiB, evict", dict(scope_cap=2 * GB, keep_cap=GB, evict=True)),
    ("always, 16 GiB", dict(scope_cap=16 * GB, keep_cap=16 * GB)),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    with open(args.trace, "rb") as f:
        d = pickle.load(f)
    events = d["events"][args.rank]
    solves = sorted({phase_of(e).split(":")[0] for e in events if phase_of(e).startswith("solve")})
    steady = solves[1:]
    costs = cost_tables(events)
    print("%d^3 on %d ranks, rank %d: %d events; steady-state solves: %s"
          % (d["n"], d["ranks"], args.rank, len(events), ", ".join(steady)))
    size = {e[1]: e[2] for e in events if e[0] == "a"}
    al = [(e[2], e[4] if len(e) > 4 else 0.0) for e in events
          if e[0] == "a" and e[3].split(":")[0] in steady]
    fr = [(size.get(e[1], 0), e[3] if len(e) > 3 else 0.0) for e in events
          if e[0] == "f" and e[2].split(":")[0] in steady]
    n, tot = len(al), sum(s for s, _ in al)
    ta, tf = sum(t for _, t in al), sum(t for _, t in fr)
    print("  per solve: %d allocations (%.1f GB, %.1f ms) and %d frees (%.1f ms); by size:"
          % (n / len(steady), tot / 2 ** 30 / len(steady), 1e3 * ta / len(steady),
             len(fr) / len(steady), 1e3 * tf / len(steady)))
    print("    %23s %7s %7s %9s %9s %10s %10s" % ("bytes", "allocs", "share", "MB", "alloc ms", "free ms", "us/alloc"))
    for lo, hi in ((0, 4096), (4096, 65536), (65536, 2 ** 20), (2 ** 20, 2 ** 24),
                   (2 ** 24, 2 ** 28), (2 ** 28, 2 ** 40)):
        sa = [(s, t) for s, t in al if lo <= s < hi]
        sf = [(s, t) for s, t in fr if lo <= s < hi]
        if not sa:
            continue
        print("    %10d to %-10d %7d %6.1f%% %9.1f %9.1f %10.1f %10.0f"
              % (lo, hi, len(sa) / len(steady), 100.0 * len(sa) / n,
                 sum(s for s, _ in sa) / 2 ** 20 / len(steady),
                 1e3 * sum(t for _, t in sa) / len(steady), 1e3 * sum(t for _, t in sf) / len(steady),
                 1e6 * sum(t for _, t in sa) / len(sa)))
    print("\n  %-46s %7s %8s %8s %8s | ms by phase" % ("policy", "calls", "ms", "held MB", "over MB"))
    for name, pol in POLICIES:
        sim = run(events, costs, **pol)
        per, tper = collections.Counter(), collections.Counter()
        for phase, c in sim.calls.items():
            s, _, ph = phase.partition(": ")
            if s in steady:
                per[ph] += c
                tper[ph] += sim.time[phase]
        k = len(steady)
        detail = "  ".join("%s %.0f" % (ph, 1e3 * t / k) for ph, t in sorted(tper.items(), key=lambda kv: -kv[1])[:4])
        print("  %-46s %7.0f %8.1f %8.0f %8.0f | %s"
              % (name, sum(per.values()) / k, 1e3 * sum(tper.values()) / k, sim.held / MB,
                 (sim.peak_total - sim.peak_req) / MB, detail))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Element kernel and assembly: time per element against the number of elements.

``benchmarks/bench_assembly.py`` compares one CPU core with one GPU at one mesh per
element type, with 2,744 to 162,000 elements.  This script repeats its measurement (it
calls the same ``measure``) over a range of mesh sizes, so that the two regimes can be
told apart: small batches, where the GPU time per element is the fixed cost of a kernel
call divided by the number of elements, and large batches, where it is the throughput of
the card.  The CPU time per element does not depend on the mesh, so the CPU is measured
up to ``--cpu-max`` elements only.

``--hypre-device`` configures MFEM and hypre on the GPU first, as the GPU runs of the
library do, so that "complete assembly" ends with the matrix in GPU memory; without it
the matrix is created in CPU memory, as in ``bench_assembly.py``.
"""
import argparse
import json
import os
import sys
import time

HM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HM)
sys.path.insert(0, os.path.join(HM, "benchmarks"))

import hippymfem as hm                                               # noqa: E402
import bench_assembly as B                                           # noqa: E402
import mfem.par as mfem                                              # noqa: E402

COMM = B.COMM


def parse_cases(text):
    cases = []
    for item in filter(None, text.split(";")):
        kind, order, sizes = item.split(":")
        cases.append((kind, int(order), [int(v) for v in sizes.split(",")]))
    return cases


def nelem(kind, n):
    return {"quad": n * n, "tri": 2 * n * n, "hex": n ** 3, "tet": 6 * n ** 3}[kind]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default="hex:2:8,16,24,32,48,64")
    ap.add_argument("--devices", default="cpu,gpu")
    ap.add_argument("--cpu-max", type=int, default=40000, help="largest mesh (elements) timed on the CPU")
    ap.add_argument("--gpu-min", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--density", default="diffusion")
    ap.add_argument("--hypre-device", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.hypre_device:
        hm.configure_device("cuda", COMM)
    else:
        mfem.Hypre.Init()
    info = B.machine()
    info["hypre_device"] = bool(args.hypre_device)
    info["hessian_route"] = str(getattr(hm.config, "hessian", ""))
    # what sizes the kernels' chunks: JAX's share of the card and the library's settings
    info["env"] = {k: v for k, v in os.environ.items()
                   if k.startswith("HIPPYMFEM_") or k.startswith("XLA_")}
    try:
        from hippymfem.fem import kernel as _K

        info["jax_bytes_limit"] = int((_K.device().memory_stats() or {}).get("bytes_limit", 0))
    except Exception:                                                # noqa: BLE001
        info["jax_bytes_limit"] = 0
    B.say("host %s, GPUs %s, hypre on %s, density %s, route %s"
          % (info["host"], info["gpus"], "device" if args.hypre_device else "CPU",
             args.density, info["hessian_route"]))
    B.say("%-8s %5s %9s %-4s %12s %12s %12s %10s"
          % ("element", "n", "elements", "dev", "kernel us/e", "scatter us/e", "full us/e", "wall s"))
    rows = []
    for kind, order, sizes in parse_cases(args.cases):
        for n in sizes:
            ne = nelem(kind, n)
            for dev in args.devices.split(","):
                if dev == "cpu" and ne > args.cpu_max:
                    continue
                if dev == "gpu" and ne < args.gpu_min:
                    continue
                t0 = time.perf_counter()
                try:
                    r = B.measure(kind, n, order, "csr", dev, args.reps, density=args.density)
                except Exception as e:                               # noqa: BLE001
                    B.say("%-8s %5d %9d %-4s failed: %s" % ("%s P%d" % (kind, order), n, ne, dev,
                                                           str(e).splitlines()[0][:120]))
                    continue
                r["wall"] = time.perf_counter() - t0
                rows.append(r)
                B.say("%-8s %5d %9d %-4s %12.3f %12.3f %12.3f %10.1f"
                      % ("%s P%d" % (kind, order), n, r["NE_global"], dev, r["us_per_elem_kernel"],
                         r["us_per_elem_scatter"], r["us_per_elem_full"], r["wall"]))
                if args.out and B.RANK == 0:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    with open(args.out, "w") as f:
                        json.dump({"machine": info, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

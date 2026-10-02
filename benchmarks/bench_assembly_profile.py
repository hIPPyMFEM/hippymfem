#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Where the time of a complete assembly on a GPU goes, and what a scatter costs.

For one element type and mesh: the element kernel, the reduction of the element matrices
to the values of the CSR matrix by several methods, the copy of the values to CPU memory,
and the complete assembly as the library does it (with the matrix in CPU memory, or on
the GPU with ``--hypre-device``).  With the library of ``tools/gpuprof.c`` preloaded it
also counts the transfers between CPU and GPU memory that MFEM and hypre make in the
step from the element matrices to the hypre matrix, and it times that step with the
matrix kept: its construction, its first two products and its destruction.

Reductions compared, all on the GPU:

* ``segment_sum``   the library's default: a scatter-add with unsorted indices
* ``at_add``        ``acc.at[slot].add(values)``, as the streamed route does
* ``sorted``        the entries permuted so that the slots ascend, then a segment sum
                    that is told the indices are sorted
* ``padded``        every slot gathers its contributions from a padded index list and
                    sums them (the library's deterministic option)
* ``bucketed``      the same gather, with the slots grouped by their number of
                    contributions, so that nothing is padded
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HM)
sys.path.insert(0, os.path.join(HM, "benchmarks"))

import hippymfem as hm                                               # noqa: E402
import bench_assembly as B                                           # noqa: E402
import mfem.par as mfem                                              # noqa: E402
import jax                                                           # noqa: E402
import jax.numpy as jnp                                              # noqa: E402
from jax.ops import segment_sum                                      # noqa: E402

from hippymfem.fem import assemble as asm                            # noqa: E402
from hippymfem.fem import kernel as K                                # noqa: E402
from hippymfem.fem.csrassemble import get_pattern                    # noqa: E402
from hippymfem.modeling.variables import ADJOINT, STATE              # noqa: E402


def bench(fn, reps):
    """Median seconds of ``fn`` over ``reps`` calls after one warm-up; ``fn`` returns
    something to wait for."""
    out = fn()
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="hex")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="gpu", help="where the kernels and reductions run")
    ap.add_argument("--hypre-device", action="store_true")
    ap.add_argument("--cprofile", type=int, default=0,
                    help="profile the library's scatter-to-matrix step and print this many lines")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.hypre_device:
        hm.configure_device("cuda", B.COMM)
    else:
        mfem.Hypre.Init()
    pm, Vu, Vm, b, kern, loc, bc = B.make(args.kind, args.n, args.order)
    NE = pm.GetNE()
    asm.set_assembly_backend("csr")
    K.set_device(args.device)
    dev = K.device()
    pat = get_pattern(Vu, Vu, b.groups)
    slot = np.asarray(pat.slot)
    nnz, nent = int(pat.nnz), int(slot.size)
    print("%s P%d n=%d: %d elements, %d element entries, %d matrix entries (%.2f per slot), "
          "perm %s, sign %s, hypre on %s"
          % (args.kind, args.order, args.n, NE, nent, nnz, nent / nnz, pat.perm is not None,
             pat.sign is not None, "device" if args.hypre_device else "CPU"), flush=True)
    rec = {"kind": args.kind, "n": args.n, "order": args.order, "NE": NE, "entries": nent,
           "nnz": nnz, "hypre_device": bool(args.hypre_device), "gpu": B.machine()["gpus"]}
    us = lambda t: 1e6 * t / NE                                      # noqa: E731

    # ---- the kernel
    def kernel():
        return kern.element_matrices(ADJOINT, STATE, loc)

    t_kernel, mats = bench(kernel, args.reps)
    print("  kernel                      %9.3f ms  %8.3f us/element" % (1e3 * t_kernel, us(t_kernel)), flush=True)
    rec["kernel"] = t_kernel
    flat = jnp.reshape(mats[0], (-1,)) if len(mats) == 1 else jnp.concatenate(
        [jnp.reshape(E, (-1,)) for E in mats])
    jax.block_until_ready(flat)

    # ---- reductions
    d_slot = jax.device_put(jnp.asarray(slot), dev)

    @jax.jit
    def r_segment(v):
        return segment_sum(v, d_slot, num_segments=nnz, indices_are_sorted=False)

    @jax.jit
    def r_at(v):
        return jnp.zeros(nnz, dtype=v.dtype).at[d_slot].add(v)

    order = np.argsort(slot, kind="stable")
    counts = np.bincount(slot, minlength=nnz)
    d_order = jax.device_put(jnp.asarray(order), dev)
    d_sorted = jax.device_put(jnp.asarray(slot[order]), dev)

    @jax.jit
    def r_sorted(v):
        return segment_sum(v[d_order], d_sorted, num_segments=nnz, indices_are_sorted=True)

    maxc = int(counts.max())
    starts = np.zeros(nnz + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    pad = np.full((nnz, maxc), nent, dtype=np.int64)
    within = np.arange(nent) - np.repeat(starts[:-1], counts)
    pad[slot[order], within] = order
    d_pad = jax.device_put(jnp.asarray(pad), dev)

    @jax.jit
    def r_padded(v):
        ext = jnp.concatenate([v, jnp.zeros((1,), dtype=v.dtype)])
        return jnp.sum(ext[d_pad], axis=1)

    classes = [int(m) for m in np.unique(counts) if m > 0]
    groups, where = [], np.empty(nnz, dtype=np.int64)
    at = 0
    for m in classes:
        s = np.flatnonzero(counts == m)
        idx = order[starts[s][:, None] + np.arange(m)[None, :]]          # (n_m, m)
        groups.append(jax.device_put(jnp.asarray(idx), dev))
        where[s] = at + np.arange(s.size)
        at += s.size
    d_where = jax.device_put(jnp.asarray(where), dev)

    @jax.jit
    def r_bucketed(v):
        parts = [jnp.sum(v[g], axis=1) for g in groups]
        return jnp.concatenate(parts)[d_where]

    print("  slots by number of contributions: %s"
          % ", ".join("%d: %.1f%%" % (m, 100.0 * np.count_nonzero(counts == m) / nnz) for m in classes),
          flush=True)
    ref = None
    for name, fn in (("segment_sum", r_segment), ("at_add", r_at), ("sorted", r_sorted),
                     ("padded", r_padded), ("bucketed", r_bucketed)):
        try:
            t, out = bench(lambda: fn(flat), args.reps)
        except Exception as e:                                       # noqa: BLE001
            print("  %-27s failed: %s" % (name, str(e).splitlines()[0][:100]), flush=True)
            continue
        if ref is None:
            ref = out
        err = float(jnp.max(jnp.abs(out - ref)) / jnp.max(jnp.abs(ref)))
        print("  reduction %-17s %9.3f ms  %8.3f us/element  %6.2f ns/entry  (difference %.1e)"
              % (name, 1e3 * t, us(t), 1e9 * t / nent, err), flush=True)
        rec["reduce_" + name] = t

    # ---- the copy of the values to CPU memory, and back
    t0 = time.perf_counter()
    host = np.asarray(ref)
    t_d2h = time.perf_counter() - t0
    t0 = time.perf_counter()
    back = jax.device_put(host, dev)
    jax.block_until_ready(back)
    t_h2d = time.perf_counter() - t0
    print("  values to CPU memory        %9.3f ms  %8.3f us/element  (%.0f MB, %.2f GB/s); back %.3f ms"
          % (1e3 * t_d2h, us(t_d2h), host.nbytes / 1e6, host.nbytes / 1e9 / t_d2h, 1e3 * t_h2d), flush=True)
    rec["d2h"], rec["h2d"] = t_d2h, t_h2d

    # ---- the library's assembly from element matrices that exist, and complete
    def scatter_only():
        A = asm.assemble_matrix(Vu, Vu, b.groups, mats, NE, test_ess=bc.ess_tdof)
        del A

    def full():
        A = asm.assemble_matrix(Vu, Vu, b.groups, kern.element_matrices(ADJOINT, STATE, loc), NE,
                                test_ess=bc.ess_tdof)
        del A

    from bench_hessian_anatomy import PROF, copies

    def build_only():
        return asm.assemble_matrix(Vu, Vu, b.groups, mats, NE, test_ess=bc.ess_tdof)

    for name, fn in (("library: scatter to matrix", scatter_only), ("library: complete", full)):
        fn()
        ts = []
        for _ in range(args.reps):
            PROF.reset()
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
            prof = PROF.read()
        t = float(np.median(ts))
        print("  %-27s %9.3f ms  %8.3f us/element" % (name, 1e3 * t, us(t)), flush=True)
        print("      driver: " + copies(prof), flush=True)
        rec[name] = t
        rec[name + " prof"] = prof
    # the same step with the matrix kept: its construction, then its first product (which
    # would move it to the GPU if the construction had not), then its destruction
    xv, yv = Vu.vector(), Vu.vector()
    xv.set(1.0)
    A0 = build_only()
    A0.Mult(xv.hypre, yv.hypre)
    yv.norm("l2")
    del A0
    rows = {"build": [], "first product": [], "second product": [], "delete": []}
    profs = {}
    for _ in range(args.reps):
        PROF.reset()
        t0 = time.perf_counter()
        A1 = build_only()
        rows["build"].append(time.perf_counter() - t0)
        profs["build"] = PROF.read()
        for key in ("first product", "second product"):
            PROF.reset()
            t0 = time.perf_counter()
            A1.Mult(xv.hypre, yv.hypre)
            yv.hypre.Norml2()
            rows[key].append(time.perf_counter() - t0)
            profs[key] = PROF.read()
        PROF.reset()
        t0 = time.perf_counter()
        del A1
        rows["delete"].append(time.perf_counter() - t0)
        profs["delete"] = PROF.read()
    for key in rows:
        t = float(np.median(rows[key]))
        print("  matrix kept: %-15s %9.3f ms   %s" % (key, 1e3 * t, copies(profs[key])), flush=True)
        rec["kept " + key] = t
        rec["kept " + key + " prof"] = profs[key]
    if args.cprofile:
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        scatter_only()
        pr.disable()
        st = pstats.Stats(pr)
        st.sort_stats("cumulative").print_stats(args.cprofile)
        st.sort_stats("tottime").print_stats(12)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

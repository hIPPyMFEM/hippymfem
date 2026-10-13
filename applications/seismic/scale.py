# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Throughput of the wave propagator and of its sweeps, at any size.

What an inversion spends its time in, measured over a number of time steps rather
than a whole observation window so that the largest meshes fit a short job:

* the forward sweep (one field);
* the incremental forward sweep (the field and its sensitivity in ``--directions``
  directions);
* the backward sweep (the reconstructed field and as many adjoint fields).

With ``--check`` the three are tied together by the identity
``<J dm, q> = <dm, J^T q>`` on the steps taken, which holds to round-off whatever
the size, the number of ranks and the precision of the geometric factors.

    mpirun -n 4 python scale.py --n 32 --refine 1 --steps 50 --out scale.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--refine", type=int, default=0)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--cube", type=float, default=1700.0)
    p.add_argument("--cfl", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=50,
                   help="time steps each sweep is timed over; 0 takes the whole window --T with the source of --f0")
    p.add_argument("--directions", type=int, default=1)
    p.add_argument("--T", type=float, default=1000.0, help="window the per-solve estimates are quoted for")
    p.add_argument("--f0", type=float, default=0.028)
    p.add_argument("--float32", action="store_true")
    p.add_argument("--geometry-float32", action="store_true")
    p.add_argument("--element-chunk", type=int, default=10000,
                   help="element-fields per pass of the stiffness kernel (0: all the elements at once)")
    p.add_argument("--element-block", type=int, default=0,
                   help="elements per block of the per-element device arrays (0: one array)")
    p.add_argument("--check", action="store_true", help="verify the adjoint identity on the steps taken")
    p.add_argument("--keys", action="store_true",
                   help="number the wave unknowns by keys instead of by an MFEM space (needed above 2^31 of them)")
    p.add_argument("--mfem-nodes", action="store_true",
                   help="take the geometry from MFEM's curved mesh on the host instead of the analytic map on the device")
    p.add_argument("--forward-only", action="store_true",
                   help="time the forward sweep alone, and keep nothing for the others on the device")
    p.add_argument("--dt-iters", type=int, default=40)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    if args.check and args.forward_only:
        p.error("--check needs the adjoint sweep, which --forward-only leaves out")
    if args.keys and args.mfem_nodes:
        p.error("--keys and --mfem-nodes are two ways to place the nodes: give one")
    return args


def main(argv=None):
    args = parse(argv)
    import hippymfem as hm
    from mpi4py import MPI
    import mfem.par as mfem
    import jax.numpy as jnp
    import earth
    import sem
    from run import density

    comm = MPI.COMM_WORLD
    rank0 = comm.rank == 0
    mfem.Hypre.Init()

    def say(*a):
        if rank0:
            print(*a, flush=True)

    T = {}
    t_all = time.time()

    def lap(name, t0):
        comm.Barrier()
        T[name] = time.time() - t0
        return time.time()

    t0 = time.time()
    bm = earth.BallMesh(args.n, cube=args.cube)
    pmesh = bm.cube_mesh(comm, args.refine)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    t0 = lap("mesh", t0)
    if args.mfem_nodes:
        # the reference path: MFEM's curved mesh, its geometry evaluated on the host
        bm.warp_mesh(pmesh, args.order)
        t0 = lap("warp", t0)
        Vu = hm.FunctionSpace.H1(pmesh, args.order)
        t0 = lap("space", t0)
        col = sem.Collocation(Vu)
        aux = earth.nodal_prem(col.X)
    elif args.keys:
        # no MFEM space of the wave field's degree: its unknowns are numbered by keys
        Vu = None
        col = sem.Collocation.from_keys(Vm, args.order, sem.vertex_coordinates(pmesh), bm.warp_jax(),
                                        bm.node_keys(args.order, args.refine))
        aux = earth.prem_aux()
    else:
        Vu = hm.FunctionSpace.H1(pmesh, args.order)
        t0 = lap("space", t0)
        col = sem.Collocation(Vu, mapped=(Vm, sem.vertex_coordinates(pmesh), bm.warp_jax()))
        aux = earth.prem_aux()
    t0 = lap("geometry", t0)
    # dt = 0 for now: the stability limit is estimated below, with --dt-iters
    eng = sem.SpectralWave(
        Vu, Vm, density, aux=aux, col=col, cfl=args.cfl, dt=0.0,
        dtype=jnp.float32 if args.float32 else jnp.float64,
        geometry_dtype=jnp.float32 if (args.geometry_float32 or args.float32) else None,
        element_chunk=args.element_chunk, element_block=args.element_block,
        sensitivities=not args.forward_only)
    del aux
    t0 = lap("engine", t0)
    eng.set_parameter(Vm.vector())
    t0 = lap("coefficients", t0)
    dt_lim = eng.stable_dt(iters=args.dt_iters)
    t0 = lap("stable dt", t0)
    ne, nd, nm = int(pmesh.GetGlobalNE()), int(eng.global_size), int(Vm.GlobalTrueVSize())
    nfull = int(np.ceil(args.T / (args.cfl * dt_lim)))
    whole = args.steps <= 0
    N = nfull if whole else args.steps
    eng.dt, eng.nsteps = (args.T / nfull if whole else args.cfl * dt_lim), N
    t = eng.dt * np.arange(N + 1)
    src = earth.paper_sources()
    rec, _, _ = earth.paper_receivers()
    if args.check and not whole:
        # a few steps do not take a wave from a source to a receiver of the experiment
        # (anything travels one element a step at most, and what arrives first is far
        # below round-off): the identity is tested at the five points of the surface
        # above the sources as well, which the waves do reach
        rec = np.concatenate([rec, earth.R_EARTH * src / np.linalg.norm(src, axis=1)[:, None]])
    if whole:
        wav = np.tile(earth.ricker(t, args.f0)[:, None], (1, src.shape[0]))
    else:
        # a source that is active during the steps taken, whatever their number
        wav = np.tile(earth.ricker(t, 4.0 / (N * eng.dt), 0.4 * N * eng.dt)[:, None], (1, src.shape[0]))
    eng.set_sources(sem.value_operator(col, src, comm), wav)
    rhat = rec / np.linalg.norm(rec, axis=1)[:, None]
    eng.set_receivers(sem.derivative_operator(col, rec, rhat, 1.0 / float(earth.prem(earth.R_EARTH)[0]), comm))
    t0 = lap("sources and receivers", t0)
    iface = comm.allreduce(int(eng.iface.iface.size))
    say("seismic scale: n = %d (refine %d), order %d: %d elements, %d wave unknowns, %d parameters; %d ranks, %s, %s%s%s"
        % (args.n, args.refine, args.order, ne, nd, nm, comm.size, hm.fem.kernel.device(),
           "fp32" if args.float32 else "fp64", ", geometry fp32" if args.geometry_float32 else "",
           ", numbered by keys" if args.keys else ""))
    say("  setup: " + ", ".join("%s %.1f s" % kv for kv in T.items()))
    say("  stable dt %.5f s, dt %.5f s: %d steps for %.0f s; %d shared unknowns (%.2f %% of all); volume error %.1e"
        % (dt_lim, eng.dt, nfull, args.T, iface, 100.0 * iface / max(nd, 1),
           abs(eng.volume / (4.0 / 3.0 * np.pi * earth.R_EARTH ** 3) - 1.0)))

    def timed(fn):
        comm.Barrier()
        w0, e0 = eng.t_wave, eng.t_exchange
        t0 = time.perf_counter()
        out = fn()
        comm.Barrier()
        dt_ = time.perf_counter() - t0
        return out, dt_, eng.t_wave - w0, comm.allreduce(eng.t_exchange - e0, op=MPI.MAX)

    def finish(res, extra):
        dev = hm.fem.kernel.device()
        peak, limit = 0.0, 0.0
        if hasattr(dev, "memory_stats"):
            ms = dev.memory_stats() or {}
            peak = ms.get("peak_bytes_in_use", 0) / 2 ** 30
            limit = ms.get("bytes_limit", 0) / 2 ** 30
        peaks = comm.gather(peak)
        limit = comm.allreduce(limit, op=MPI.MIN)
        try:
            import resource

            rss = comm.gather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
        except Exception:                                         # noqa: BLE001
            rss = None
        if rank0:
            say("  device memory in use at peak: %.2f GiB per rank at most (of %.2f it may take), %.1f GiB in all; "
                "host %.1f GiB per rank at most" % (max(peaks), limit, sum(peaks), max(rss) if rss else 0.0))
        out = dict(args=vars(args), elements=ne, wave_dofs=nd, parameters=nm, ranks=comm.size,
                   device=str(dev), dt_limit=dt_lim, dt=eng.dt, steps_per_window=nfull, shared_dofs=iface,
                   setup_seconds=T, sweeps=res, device_peak_gib=peaks, device_limit_gib=limit, host_rss_gib=rss,
                   seconds=time.time() - t_all, whole_window=whole, **(extra or {}))
        if args.out and rank0:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
        say("seismic scale: done in %.1f s" % (time.time() - t_all))
        return 0

    res = {}
    # every sweep is compiled on a few steps before it is timed
    eng.forward(limit=3)
    y, tt, tw, tx = timed(eng.forward)
    res["forward"] = dict(fields=1, seconds=tt, ms_per_step=1e3 * tw / N, exchange_ms_per_step=1e3 * tx / N,
                          dof_updates_per_s=nd * N / tw)
    say("  forward (1 field):            %8.2f ms/step (%5.2f in the exchange), %7.2f G unknown-updates/s; "
        "%.0f s per %.0f s solve" % (1e3 * tw / N, 1e3 * tx / N, 1e-9 * nd * N / tw, tw / N * nfull, args.T))
    nb = args.directions
    if args.forward_only:
        return finish(res, None)
    hm.parRandom.set_seed(11)
    dms = []
    for k in range(nb):
        d = Vm.vector()
        hm.parRandom.normal(1.0, d)
        d.array[:] *= 0.01
        dms.append(d)
    final = eng.final_state()
    eng.incremental(dms, limit=3)
    dy, tt, tw, tx = timed(lambda: eng.incremental(dms))
    res["incremental"] = dict(fields=1 + nb, seconds=tt, ms_per_step=1e3 * tw / N,
                              exchange_ms_per_step=1e3 * tx / N, dof_updates_per_s=(1 + nb) * nd * N / tw)
    say("  incremental forward (%d fields): %6.2f ms/step (%5.2f in the exchange), %7.2f G unknown-updates/s"
        % (1 + nb, 1e3 * tw / N, 1e3 * tx / N, 1e-9 * (1 + nb) * nd * N / tw))
    rng = np.random.default_rng(5)
    q = rng.standard_normal((nb, N + 1, rec.shape[0]))
    q[:, 0] = 0.0
    eng.adjoint(q, final=final, limit=3)
    g, tt, tw, tx = timed(lambda: eng.adjoint(q, final=final))
    res["adjoint"] = dict(fields=1 + nb, seconds=tt, ms_per_step=1e3 * tw / N,
                          exchange_ms_per_step=1e3 * tx / N, dof_updates_per_s=(1 + nb) * nd * N / tw)
    say("  backward sweep (%d fields):      %6.2f ms/step (%5.2f in the exchange), %7.2f G unknown-updates/s"
        % (1 + nb, 1e3 * tw / N, 1e3 * tx / N, 1e-9 * (1 + nb) * nd * N / tw))
    per_hess = (res["incremental"]["ms_per_step"] + res["adjoint"]["ms_per_step"]) * 1e-3 * nfull
    per_grad = (res["forward"]["ms_per_step"] + res["adjoint"]["ms_per_step"]) * 1e-3 * nfull
    say("  one Gauss-Newton Hessian action on %d direction%s over %.0f s: %s%.0f s; a gradient: %s%.0f s"
        % (nb, "" if nb == 1 else "s", args.T, "" if whole else "about ", per_hess,
           "" if whole else "about ", per_grad))
    if args.check and not np.abs(dy).max() > 0.0:
        say("  <J dm, q> = <dm, J^T q>: not tested, nothing has reached a receiver in %d steps" % N)
    elif args.check:
        worst = 0.0
        for k in range(nb):
            lhs = float(np.sum(dy[k] * q[k]))
            rhs = g[k].inner(dms[k])
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-300))
        res["adjoint_identity"] = worst
        say("  <J dm, q> = <dm, J^T q> to %.2e (relative), |J dm| max %.3e" % (worst, np.abs(dy).max()))
    return finish(res, dict(hessian_action_seconds=per_hess, gradient_seconds=per_grad))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import sem

        sem.abort_all()
        raise

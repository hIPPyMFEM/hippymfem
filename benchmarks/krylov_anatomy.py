#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Anatomy of one preconditioned Krylov iteration of the model problem.

``benchmarks/bench_scaling.py`` shows that a reduced-Hessian action on GPUs is its two
incremental solves and that their time per iteration stops falling as ranks are added.
This script takes the iteration itself apart.  It assembles the Jacobian of the model
problem (the same mesh, PDE, prior sample and boundary conditions as the benchmarks),
and then times, with hypre and MFEM only:

* a matrix-vector product, an inner product and a vector update,
* one BoomerAMG V-cycle,
* a preconditioned CG solve to the tolerance of the incremental solves,

for one or more settings of BoomerAMG (``--variant``).  When ``libgpuprof.so``
(``tools/gpuprof.c``) is preloaded it also counts, per operation, what the process asks
of the CUDA driver and of MPI: device allocations, copies between CPU and GPU memory,
kernel launches, synchronizations, sends and reductions, and the time the calling thread
spends in each; ``--kernel-table`` adds the device time of every kernel.  Run it as the
other benchmarks::

    mpicc -O2 -fPIC -shared -o libgpuprof.so tools/gpuprof.c -ldl
    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n 8 -x LD_PRELOAD=$PWD/libgpuprof.so \\
        tools/mpirun_pinned.sh python benchmarks/krylov_anatomy.py --n 128 --assembly mfem \\
        --cart-part --variant default: --variant hypre:vendor=0 --out anatomy_n128_r8.json

A variant is ``name:key=value,key=value``.  Keys: ``relax``, ``coarsen``, ``interp``,
``agg``, ``theta``, ``maxlev``, ``sweeps`` (MFEM setters), ``vendor`` (cuSPARSE or hypre's
own matrix-vector product, a global switch), and any hypre name ``X`` as ``amg.X=int`` or
``amgr.X=float`` or ``amg2.X=int:int``, applied by the preloaded library just before the
setup (``HYPRE_BoomerAMGSetX``), e.g. ``amg.PMaxElmts=2`` or ``amg2.CycleRelaxType=199:3``.

``--independent`` runs one copy of the problem per rank, each on its own communicator,
with the timed loops started together.  It measures whether GPUs (or MIG instances of one
GPU) slow each other down when they compute at the same time without exchanging anything.
"""
import argparse
import ctypes
import json
import os
import platform
import sys
import time

import numpy as np

HM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HM)

import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

import mfem.par as mfem                                              # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

WORLD = MPI.COMM_WORLD
INDEPENDENT = "--independent" in sys.argv
COMM = MPI.COMM_SELF if INDEPENDENT else WORLD
RANK = WORLD.rank


def say(*a):
    if INDEPENDENT:
        print("[rank %d]" % RANK, *a, flush=True)
    elif RANK == 0:
        print(*a, flush=True)


def gpu_name():
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        n = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        return n[0] if n else "none"
    except Exception:                                                # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- the preloaded counters
class Prof:
    """The counters of libgpuprof.so, if it is preloaded; otherwise every delta is empty."""

    def __init__(self):
        self.lib = None
        try:
            lib = ctypes.CDLL(None)
            lib.gpuprof_nkinds.restype = ctypes.c_int
            self.n = lib.gpuprof_nkinds()
            lib.gpuprof_name.restype = ctypes.c_char_p
            self.names = [lib.gpuprof_name(k).decode() for k in range(self.n)]
            self.lib = lib
        except (AttributeError, OSError):
            self.n, self.names = 0, []

    def reset(self):
        if self.lib is not None:
            self.lib.gpuprof_reset()

    def read(self, per=1.0):
        """{name: (count, seconds, bytes)} since the last reset, each divided by ``per``,
        for the kinds that occurred."""
        if self.lib is None:
            return {}
        c = (ctypes.c_long * self.n)()
        t = (ctypes.c_double * self.n)()
        b = (ctypes.c_double * self.n)()
        self.lib.gpuprof_get(c, t, b)
        return {self.names[k]: (c[k] / per, t[k] / per, b[k] / per)
                for k in range(self.n) if c[k]}

    def read_ranks(self, per=1.0):
        """The counters of every rank, as a list indexed by rank (on rank 0; ``None``
        elsewhere).  Call it after the timed region: the gather is itself MPI."""
        mine = self.read(per)
        if INDEPENDENT:
            return [mine]
        return WORLD.gather(mine, root=0)

    def kernel_table(self, fn, path):
        """Run ``fn`` with every kernel bracketed by a synchronization and write this
        rank's table of device times per kernel to ``path``."""
        if self.lib is None or not hasattr(self.lib, "gpuprof_sync_mode"):
            return
        self.lib.gpuprof_sync_mode(ctypes.c_int(1))
        try:
            fn()
        finally:
            self.lib.gpuprof_dump(path.encode())
            self.lib.gpuprof_sync_mode(ctypes.c_int(0))

    def hooked(self):
        if self.lib is None:
            return 0
        self.lib.gpuprof_hooked.restype = ctypes.c_int
        return self.lib.gpuprof_hooked()


PROF = Prof()


def fmt_prof(d, keys=None):
    """One line: 'launch 212 (1.31 ms)  copy_d2h 14 (0.21 ms, 3.2 kB) ...'."""
    out = []
    for k, (c, t, b) in d.items():
        if keys is not None and k not in keys:
            continue
        s = "%s %.4g (%.3f ms" % (k, c, 1e3 * t)
        if b:
            s += ", %.4g kB" % (b / 1e3)
        out.append(s + ")")
    return "  ".join(out) if out else "(no counters)"


class Pool:
    """The library's recycling pool for hypre's device memory, if asked for."""

    def __init__(self, megabytes):
        self.py = None
        if megabytes > 0:
            self.py = hm.common.mfemconfig.set_hypre_pool(megabytes)
            if self.py is None:
                raise SystemExit("the pool could not be installed (needs hypre on a CUDA device)")

    def stats(self, reset=False):
        if self.py is None:
            return {}
        st = self.py.stats()
        if reset:
            p = self.py
            p.requests = p.from_pool = p.driver_allocs = p.driver_frees = 0
            p.peak_cached, p.peak_in_use = p.cached, p.in_use
        return st


def hypre_lib():
    """ctypes handle of the libHYPRE this process has loaded."""
    with open("/proc/self/maps") as f:
        for line in f:
            if "libHYPRE" in line:
                return ctypes.CDLL(line.split()[-1])
    return None


def flush_c():
    ctypes.CDLL(None).fflush(None)


# ---------------------------------------------------------------- timing
def done(v):
    """Complete the device work queued on ``v``: a norm computed on the device (the
    library's own ``ParVector.norm`` would copy the vector to CPU memory)."""
    return v.hypre.Norml2()


def timed(fn, sync):
    """Seconds for ``fn``, between barriers, with the device work completed by a norm."""
    WORLD.Barrier()
    t0 = time.perf_counter()
    fn()
    done(sync)
    if not INDEPENDENT:
        WORLD.Barrier()
    return time.perf_counter() - t0


def loop_time(fn, count, sync, reps):
    """Median over ``reps`` of the time of one call of ``fn`` in a loop of ``count``,
    and the counters of the last loop per call."""
    ts = []
    prof = {}
    for _ in range(reps):
        done(sync)
        PROF.reset()

        def body():
            for _ in range(count):
                fn()
        t = timed(body, sync)
        prof = PROF.read(per=count)
        ts.append(t / count)
    return float(np.median(ts)), prof


def parse_variant(text):
    name, _, rest = text.partition(":")
    opts = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        opts[k.strip()] = v.strip()
    return name.strip(), opts


def make_amg(A, opts, print_level):
    for k in [k for k in os.environ if k.startswith(("GPUPROF_AMG_", "GPUPROF_AMGR_", "GPUPROF_AMG2_"))]:
        del os.environ[k]
    pc = mfem.HypreBoomerAMG(A)
    pc.SetPrintLevel(print_level)
    for k, v in opts.items():
        if k == "relax":
            pc.SetRelaxType(int(v))
        elif k == "coarsen":
            pc.SetCoarsening(int(v))
        elif k == "interp":
            pc.SetInterpolation(int(v))
        elif k == "agg":
            pc.SetAggressiveCoarsening(int(v))
        elif k == "theta":
            pc.SetStrengthThresh(float(v))
        elif k == "maxlev":
            pc.SetMaxLevels(int(v))
        elif k == "sweeps":
            pc.SetCycleNumSweeps(int(v), int(v))
        elif k == "vendor":
            pass
        elif k.startswith("amg."):
            os.environ["GPUPROF_AMG_" + k[4:]] = v
        elif k.startswith("amgr."):
            os.environ["GPUPROF_AMGR_" + k[5:]] = v
        elif k.startswith("amg2."):
            os.environ["GPUPROF_AMG2_" + k[5:]] = v
        else:
            raise SystemExit("unknown variant key %r" % k)
    return pc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"))
    ap.add_argument("--variant", action="append", default=[],
                    help="name:key=value,...; repeatable; 'default:' is MFEM's setting")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--count", type=int, default=20, help="calls per timed loop")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--print-hierarchy", action="store_true",
                    help="have BoomerAMG print its setup statistics")
    ap.add_argument("--assembly", default="jax", choices=("jax", "mfem"),
                    help="jax: the library's Jacobian (compiles the element kernel, a "
                         "minute or more); mfem: MFEM's DiffusionIntegrator with the "
                         "coefficient exp(m) interpolated at the nodes, no JAX")
    ap.add_argument("--pool", type=int, default=0,
                    help="megabytes of freed device memory that the library's recycling pool "
                         "for hypre may hold (0: no pool, hypre calls cudaMalloc and cudaFree)")
    ap.add_argument("--setup-repeats", type=int, default=1,
                    help="set up the preconditioner this many times per variant (a Newton "
                         "iteration sets it up once per forward solve)")
    ap.add_argument("--kernel-table", action="store_true",
                    help="one more CG solve per variant with every kernel synchronized, and "
                         "its device time per kernel written next to --out, one file per rank")
    ap.add_argument("--independent", action="store_true",
                    help="one copy of the problem per rank, timed at the same moment")
    ap.add_argument("--cart-part", action="store_true",
                    help="partition the box as a Cartesian grid of ranks instead of with METIS")
    ap.add_argument("--unit-coefficient", action="store_true",
                    help="with --assembly mfem: the coefficient 1 instead of exp(m).  The "
                         "records of this script up to 2026-10-02 were taken on that "
                         "matrix, because the script read the parameter from a CPU copy "
                         "that the GPU had not filled")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    variants = [parse_variant(v) for v in (args.variant or ["default:"])]

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))
    pool = Pool(args.pool if args.device != "cpu" else 0)
    N, ORDER = args.n, args.order
    t_start = time.perf_counter()
    serial = mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON)
    if args.cart_part:
        # the most cubic grid of ranks that divides the rank count (bench_newton_device.py)
        p = COMM.size
        grid = min(((a, b, p // (a * b)) for a in range(1, p + 1) if p % a == 0
                    for b in range(a, p // a + 1) if (p // a) % b == 0 and b <= p // (a * b)),
                   key=lambda g: max(g) / min(g))
        nxyz = mfem.intArray(list(grid))
        pmesh = mfem.ParMesh(COMM, serial, serial.CartesianPartitioning(nxyz.GetData()))
        say("  Cartesian partition %s" % (grid,))
    else:
        grid = None
        pmesh = mfem.ParMesh(COMM, serial)
    del serial
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    ndof = int(Vu.GlobalTrueVSize())
    say("%d^3 hex order %d: %d state dofs, %d ranks (%.0f per rank), hypre on %s (%s), counters %s"
        % (N, ORDER, ndof, COMM.size, ndof / COMM.size, args.device, gpu_name(),
           "on" if PROF.lib is not None else "off"))

    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=args.tol,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    u = pde.generate_state()
    PROF.reset()
    keep = []
    if args.assembly == "jax":
        t_fwd = timed(lambda: pde.solveFwd(u, [u, mtrue, None]), u)
        it_fwd = int(pde.solver.iterations)
        A = pde.solver.A
        say("  set up in %.1f s; the library's forward solve %.3f s, %d iterations"
            % (time.perf_counter() - t_start, t_fwd, it_fwd))
    else:
        t0 = time.perf_counter()
        m_gf = Vm.to_gridfunction(mtrue)
        kappa = mfem.ParGridFunction(Vm.fes)
        if args.unit_coefficient:
            kappa.Assign(1.0)
        else:
            # the prolongation leaves the values on the GPU: bring them to the CPU before
            # GetDataArray(), which returns the CPU copy whether or not it is current
            m_gf.HostRead()
            kappa.Assign(np.exp(np.array(m_gf.GetDataArray(), copy=True)))
        coef = mfem.GridFunctionCoefficient(kappa)
        form = mfem.ParBilinearForm(Vu.fes)
        form.AddDomainIntegrator(mfem.DiffusionIntegrator(coef))
        form.Assemble()
        form.Finalize()
        A = mfem.HypreParMatrix()
        form.FormSystemMatrix(bc.ess_tdof, A)
        keep += [m_gf, kappa, coef, form]
        t_fwd, it_fwd = 0.0, 0
        say("  set up in %.1f s, of which MFEM's assembly %.1f s"
            % (time.perf_counter() - t_start, time.perf_counter() - t0))
    say("  driver entry points wrapped: %d" % PROF.hooked())

    x, y, b = pde.generate_state(), pde.generate_state(), pde.generate_state()
    hm.parRandom.normal(1.0, b)
    x.assign(b)
    y.zero()
    H = hypre_lib()

    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N,
           "order": ORDER, "tdofs": ndof, "device": args.device, "tol": args.tol,
           "tag": args.tag, "assembly": args.assembly,
           "coefficient": ("unit" if (args.assembly == "mfem" and args.unit_coefficient)
                           else "exp(m)"),
           "vendor_set": True,
           "pool_mb": args.pool, "grid": list(grid) if grid else None, "t_fwd": t_fwd, "it_fwd": it_fwd,
           "counters": PROF.lib is not None,
           "variants": {}}

    def kernels(label):
        # MFEM's own operations, which are what its CG solver calls
        r = {}
        r["spmv"], p = loop_time(lambda: A.Mult(x.hypre, y.hypre), args.count, y, args.reps)
        r["spmv_prof"] = p
        r["dot"], p = loop_time(lambda: mfem.InnerProduct(x.hypre, y.hypre), args.count, y,
                                args.reps)
        r["dot_prof"] = p
        r["axpy"], p = loop_time(lambda: y.hypre.Add(1e-3, x.hypre), args.count, y, args.reps)
        r["axpy_prof"] = p
        # the library's vector class, which the outer algorithms use
        r["lib_dot"], p = loop_time(lambda: x.inner(y), 3, y, 1)
        r["lib_dot_prof"] = p
        say("  [%s] matrix-vector %.3f ms | inner product %.3f ms | update %.3f ms"
            " | the library's inner product %.3f ms"
            % (label, 1e3 * r["spmv"], 1e3 * r["dot"], 1e3 * r["axpy"], 1e3 * r["lib_dot"]))
        say("      matrix-vector: " + fmt_prof(r["spmv_prof"]))
        say("      inner product: " + fmt_prof(r["dot_prof"]))
        say("      update:        " + fmt_prof(r["axpy_prof"]))
        say("      library inner: " + fmt_prof(r["lib_dot_prof"]))
        return r

    # The library chooses hypre's own kernel on more than one rank
    # (mfemconfig.set_hypre_spmv), so the vendor's is set here explicitly: a variant
    # without "vendor" is then cuSPARSE on any number of ranks.
    vendor_now = 1
    if H is not None and args.device != "cpu":
        H.HYPRE_SetSpMVUseVendor(ctypes.c_int(1))
    rec["kernels"] = kernels("cuSPARSE" if args.device != "cpu" else "host")

    for name, opts in variants:
        vendor = int(opts.get("vendor", 1))
        if vendor != vendor_now and H is not None and args.device != "cpu":
            H.HYPRE_SetSpMVUseVendor(ctypes.c_int(vendor))
            vendor_now = vendor
            rec["kernels_vendor%d" % vendor] = kernels("vendor=%d" % vendor)
        z = pde.generate_state()
        z.zero()
        say("--- variant %s %s" % (name, opts))
        setups = []
        pc = None
        for k in range(max(args.setup_repeats, 1)):
            if pc is not None:
                del pc                                  # frees the hierarchy
            pc = make_amg(A, opts, 1 if (args.print_hierarchy and k == 0) else 0)
            flush_c()
            pool.stats(reset=True)
            PROF.reset()
            t_first = timed(lambda: pc.Mult(b.hypre, z.hypre), z)
            setup_prof = PROF.read()
            flush_c()
            t_setup = setup_prof.get("amg_setup", (0, t_first, 0))[1]
            g = lambda kk, i=1: setup_prof.get(kk, (0, 0, 0))[i]    # noqa: E731
            setups.append({"t_setup": t_setup, "t_first": t_first, "dev_alloc": g("dev_alloc", 0),
                           "t_dev_alloc": g("dev_alloc"), "dev_free": g("dev_free", 0),
                           "t_dev_free": g("dev_free"), "pool": pool.stats()})
            if args.setup_repeats > 1 or pool.py is not None:
                ps = setups[-1]["pool"]
                say("  setup %d: %.3f s; driver allocations %d (%.0f ms), frees %d (%.0f ms)%s"
                    % (k + 1, t_setup, g("dev_alloc", 0), 1e3 * g("dev_alloc"), g("dev_free", 0),
                       1e3 * g("dev_free"),
                       "" if not ps else "; pool served %d of %d requests, holds %.0f MB "
                       "(peak %.0f MB), in use %.0f MB (peak %.0f MB)"
                       % (ps["from_pool"], ps["requests"], ps["cached"] / 2 ** 20,
                          ps["peak_cached"] / 2 ** 20, ps["in_use"] / 2 ** 20,
                          ps["peak_in_use"] / 2 ** 20)))
        t_cyc, cyc_prof = loop_time(lambda: pc.Mult(b.hypre, z.hypre), args.count, z, args.reps)

        cg = mfem.CGSolver(COMM)
        cg.SetOperator(A)
        cg.SetPreconditioner(pc)
        cg.SetRelTol(args.tol)
        cg.SetAbsTol(1e-20)
        cg.SetMaxIter(2000)
        cg.SetPrintLevel(-1)
        ts, its, cg_prof, cg_ranks = [], 0, {}, None
        for _ in range(args.reps):
            z.zero()
            done(z)
            PROF.reset()
            t = timed(lambda: cg.Mult(b.hypre, z.hypre), z)
            its = int(cg.GetNumIterations())
            cg_ranks = PROF.read_ranks(per=max(its, 1))
            cg_prof = PROF.read(per=max(its, 1)) if cg_ranks is None else cg_ranks[0]
            ts.append(t)
        t_cg = float(np.median(ts))
        if args.kernel_table and args.out:
            z.zero()
            done(z)
            WORLD.Barrier()
            PROF.kernel_table(lambda: (cg.Mult(b.hypre, z.hypre), done(z)),
                              args.out.replace(".json", "_%s_p%d.ktab" % (name, RANK)))
            WORLD.Barrier()
        # the residual actually reached, as a check that every variant solves the system
        A.Mult(z.hypre, y.hypre)
        y.axpy(-1.0, b)
        relres = y.norm("l2") / b.norm("l2")
        v = {"opts": opts, "t_setup": t_setup, "t_first": t_first, "setups": setups,
             "t_vcycle": t_cyc,
             "t_cg": t_cg, "iterations": its, "t_per_iteration": t_cg / max(its, 1),
             "relres": relres, "vcycle_prof": cyc_prof, "cg_prof_per_iteration": cg_prof,
             "cg_prof_ranks": cg_ranks, "setup_prof": setup_prof}
        rec["variants"][name] = v
        say("  setup %.3f s | V-cycle %.3f ms | CG %d iterations, %.3f s, %.3f ms per iteration"
            " | residual %.1e" % (t_setup, 1e3 * t_cyc, its, t_cg, 1e3 * t_cg / max(its, 1), relres))
        say("      V-cycle:          " + fmt_prof(cyc_prof))
        say("      per CG iteration: " + fmt_prof(cg_prof))
        say("      setup:            " + fmt_prof(setup_prof, ("dev_alloc", "dev_free", "launch",
                                                              "copy_d2h", "copy_h2d", "sync")))
        if cg_ranks and len(cg_ranks) > 1:
            for kind in ("mpi_waitall", "mpi_allreduce", "sync", "copy_d2h"):
                say("      %-13s ms per iteration, by rank: %s"
                    % (kind, " ".join("%.2f" % (1e3 * r.get(kind, (0, 0, 0))[1]) for r in cg_ranks)))
        del cg, pc

    rec["independent"] = INDEPENDENT
    rec["world_ranks"] = WORLD.size
    rec["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.out and (RANK == 0 or INDEPENDENT):
        out = args.out.replace(".json", "_p%d.json" % RANK) if INDEPENDENT else args.out
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1)
    WORLD.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

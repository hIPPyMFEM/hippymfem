#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Newton steps of a 3D inverse problem with MFEM, hypre and the kernels on the GPU.

The configuration that matters: element kernels, matrices and solves all on the
card, one card per rank.  Times the three things a Newton-CG iteration is made of --
assembling every Hessian block at a linearization point, applying the reduced
Hessian (two incremental solves), and the step itself -- and prints the cost
functional and parameter error so two cards or two rank counts can be checked for
the same answer, not only compared for speed.

Run it through ``tools/mpirun_pinned.sh`` so a rank holds a context on its own
card only.

Usage::

    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n 4 tools/mpirun_pinned.sh \\
        python benchmarks/bench_newton_device.py --n 64 --steps 2 --out results/x.json
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# hippymfem first: it pins (or, for a host run, hides) this process's GPUs and fixes
# JAX's platform list, which only takes if it comes before mpi4py and jax.
import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

import mfem.par as mfem
import jax.numpy as jnp

from hippymfem import _jaxconfig                                     # noqa: E402
from hippymfem.algorithms import singlesolve as single_mod           # noqa: E402
from hippymfem.fem import kernel as kernel_mod                       # noqa: E402
from hippymfem.fem import tdofassemble as tdof_mod                   # noqa: E402
from hippymfem.modeling.variables import PARAMETER, STATE, ADJOINT   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def gpu_name():
    """The card's product name, from whichever vendor's tool the node has."""
    import subprocess

    for cmd, pick in ((["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                       lambda line: line.strip()),
                      (["rocm-smi", "--showproductname"],
                       lambda line: line.split("Card Series:")[1].strip()
                       if "Card Series:" in line else "")):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except Exception:
            continue
        names = [pick(l) for l in out.stdout.splitlines()]
        names = [x for x in names if x]
        if out.returncode == 0 and names:
            return names[0]
    return "unknown"


def since_launch():
    """Seconds this process has existed: what the launch and the imports took, on Linux."""
    try:
        with open("/proc/self/stat") as f:
            ticks = float(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as f:
            return float(f.read().split()[0]) - ticks / os.sysconf("SC_CLK_TCK")
    except Exception:                                            # noqa: BLE001
        return float("nan")


def card_probe(reps=7):
    """Seconds of two kernels of this rank alone on its card, the least of ``reps`` each, or
    ``None`` on a host: a triad over 2^27 single-precision numbers (the card's memory
    bandwidth, which a product with a matrix is bound by) and a pass of exponentials over
    them (its arithmetic, as in an element pass).  Run by every rank at the same time after
    the steps, when every card has been busy alike: what the ranks' cards differ by."""
    if not kernel_mod.on_gpu():
        return None
    try:
        import jax
        import jax.numpy as jnp

        dev = kernel_mod.device()
        n = 1 << 27
        x = jax.device_put(jnp.linspace(0.0, 1.0, n, dtype=jnp.float32), dev)
        y = jax.device_put(jnp.full(n, 0.5, dtype=jnp.float32), dev)
        triad = jax.jit(lambda a, b: a + 1.0001 * b)
        expo = jax.jit(lambda a: jnp.sum(jnp.exp(-a * a) * jnp.tanh(a)))
        out = {}
        for name, fn, argv in (("triad", triad, (x, y)), ("exp", expo, (x,))):
            fn(*argv).block_until_ready()                    # compiled, and the card awake
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                fn(*argv).block_until_ready()
                best = min(best, time.perf_counter() - t0)
            out[name] = best
        del x, y
        return out
    except Exception as e:                                       # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}


def card_memory():
    """This rank's side of its card after the steps, for ``--rank-table``: JAX's allocator
    (what it holds, its peak, its cap), everything on the card (used, total), and the
    counts of hypre's pool with the seconds it spent asking the driver.  A rank that is
    late in a pass and not in the kernels timed on its card may show here."""
    out = {}
    try:
        import jax

        st = jax.devices()[0].memory_stats() or {}
        out["jax"] = {k: int(st[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "pool_bytes",
                                                "peak_pool_bytes", "largest_alloc_size") if k in st}
    except Exception:                                            # noqa: BLE001
        pass
    try:
        card = hm.common.devicebridge.device_memory()
        if card:
            out["card"] = [int(card[0]), int(card[1])]
        pool = hm.common.mfemconfig.HYPRE_POOL
        if pool is not None:
            st = pool.stats()
            out["pool"] = {k: st[k] for k in ("requests", "from_pool", "driver_allocs", "driver_frees", "driver_bytes",
                                              "driver_seconds", "peak_in_use", "min_free") if k in st}
    except Exception:                                            # noqa: BLE001
        pass
    return out


def spread_summary(table):
    """What the ranks of a run differ by, from the table ``rank_table`` gathers: for a
    rank's own kernels and for some stages of the steps, the least, the median and the
    largest value over the ranks with the host of the largest; and the ranks' waits at the
    collective after an element pass (the stage ``+check``: a rank that is done waits there
    for the slowest one, so the largest wait is how far the fastest rank is ahead)."""
    def stat(vals):
        v = sorted((x, h) for x, h in vals if x == x)
        if not v:
            return None
        xs = [x for x, _ in v]
        q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]   # noqa: E731
        return {"min": xs[0], "median": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1], "mean": sum(xs) / len(xs),
                "host_of_max": v[-1][1], "host_of_min": v[0][1]}

    out = {"ranks": len(table)}
    for k in ("triad", "exp"):
        out["probe_" + k] = stat([(float((t.get("probe") or {}).get(k, float("nan"))), t["host"]) for t in table])
    for k in ("+scatter_many", "+check", "+residual", "+finish_block", "+set_operator", "+require_finite_arrays"):
        out[k] = stat([(t["stages"][k][1] / max(t["stages"][k][0], 1), t["host"]) for t in table if k in t["stages"]])
    return out


def timed(fn):
    COMM.Barrier()
    t0 = time.perf_counter()
    out = fn()
    COMM.Barrier()
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--cg-max", type=int, default=25)
    ap.add_argument("--cg-tol", type=float, default=1e-6)
    ap.add_argument("--cg-reorth", action=argparse.BooleanOptionalAction, default=None,
                    help="the CG of a Newton step keeps its residuals orthogonal explicitly (the library's "
                    "default since October 2026); --no-cg-reorth is the recurrence alone, with which the "
                    "records of the paper were taken")
    ap.add_argument("--cg-prec-tol", type=float, default=None,
                    help="accuracy of the prior's solves where they precondition the CG of a Newton step, "
                    "as a fraction of that CG's tolerance (cg_preconditioner_tolerance; 0 = the solves as "
                    "they are, as in the records of the paper)")
    ap.add_argument("--cg-prec-cycles", type=int, default=None,
                    help="each of the prior's solves where they precondition the CG of a Newton step is a "
                    "fixed polynomial of this many multigrid cycles instead of a Krylov solve "
                    "(cg_preconditioner_cycles; 0 or absent = Krylov solves)")
    ap.add_argument("--cg-relax", type=float, default=None,
                    help="the incremental solves stop at this times the CG's tolerance times |r_0|/|r_k| "
                    "where that is looser than --inc-tol (cg_hessian_relaxation; 0 = --inc-tol throughout, "
                    "as in the records of the paper)")
    ap.add_argument("--inc-tol", type=float, default=None,
                    help="relative tolerance of the incremental solves during the Newton-CG iteration "
                    "(default: that of the forward solve, 1e-12; 1e-6 is enough with --cg-reorth)")
    ap.add_argument("--gn-iter", type=int, default=0)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"), help="mfem.Device kind; 'cpu' keeps hypre on the host")
    ap.add_argument("--out", default=None)
    ap.add_argument("--symmetric-jacobian", action="store_true",
                    help="declare dR/du symmetric (it is, for this PDE): adjoint solves reuse A and its AMG")
    ap.add_argument("--single-refine-goal", type=float, default=None,
                    help="with the Jacobian in a single-precision hypre, the relative residual at which the "
                    "refinement of the forward and the adjoint solve may stop (PDEVariationalProblem."
                    "SINGLE_REFINE_GOAL; default: the solver's own tolerance, three passes for 1e-12; 1e-9 "
                    "stops after two passes and one evaluation of the residual)")
    ap.add_argument("--single-give-up", choices=("double", "raise"), default=None,
                    help="what a forward solve does where single precision cannot carry it "
                    "(PDEVariationalProblem.SINGLE_GIVE_UP): solve again in double precision, the "
                    "default, or raise, which the line search answers by backtracking (for a load "
                    "that leaves no room for a double-precision Jacobian beside the single one)")
    ap.add_argument("--release-linearization", action="store_true",
                    help="drop the linearization point at a forward solve for a new parameter "
                         "(safe for this line-search Newton-CG; needed for 128^3 on one 80 GB card)")
    ap.add_argument("--matrix-free", action="store_true",
                    help="matrix-free linearization points: assemble the Jacobian alone and take "
                         "the products with the other Hessian blocks from the element kernels")
    ap.add_argument("--cart-part", action="store_true",
                    help="partition the box as a Cartesian grid of ranks instead of METIS on every rank")
    ap.add_argument("--coarse", type=int, default=None,
                    help="build the mesh in parallel: every rank builds a mesh of COARSE^3 elements and cuts "
                         "its own part into (n / COARSE)^3, in equal boxes when the ranks make a grid "
                         "(hippymfem.fem.meshes.box_mesh); 0 chooses COARSE.  Without it every "
                         "rank builds and partitions the whole mesh, 635 bytes an element")
    ap.add_argument("--amg", default=None,
                    help="BoomerAMG settings of the problem's solves alone, such as \"relax_wt=1.8,pmax=6\" "
                         "(the names of mfemconfig.BOOMERAMG_SETTERS); HIPPYMFEM_AMG sets every solver's")
    ap.add_argument("--prior-lumped-mass", action="store_true",
                    help="the prior's mass matrix by the rule at the vertices (BiLaplacianPrior, "
                    "lumped_mass): a product with its precision then solves with a diagonal matrix. "
                    "Another discretization of the prior: not the problem of the runs without it")
    ap.add_argument("--prior-amg", default=None,
                    help="BoomerAMG settings of the prior's solves alone (a linear-element operator, whose "
                         "V-cycle is bound by its levels and not by its size)")
    ap.add_argument("--profile-setup", default=None,
                    help="after the steps: BoomerAMG setups of the last Jacobian with other settings, each "
                         "timed with a solve (bench_scaling.profile_setup: 'name:settings;name:settings')")
    ap.add_argument("--h-levels", default="0",
                    help="with --coarse: build the mesh by this many halvings at the end and keep the meshes between "
                         "them, for the geometric levels of a p-multigrid (HIPPYMFEM_PMG_H); or the cuts themselves "
                         "from the first to the last, as 5,3,2,2,2 (the coarse mesh is then N over their product "
                         "unless --coarse names it)")
    ap.add_argument("--pmg-h-host", type=int, default=None,
                    help="with the geometric levels of a p-multigrid (HIPPYMFEM_PMG_H) and hypre on a device: the "
                         "levels of at most this many rows a rank live in host memory (the solvers' pmg_h_host).  "
                         "Default: HIPPYMFEM_PMG_H_HOST where it is set, else 5000 for a mesh built by cuts "
                         "(--h-levels); 0: every level on the device")
    ap.add_argument("--prior-h", type=int, default=None,
                    help="the multigrid cycles of the prior's preconditioner (--cg-prec-cycles) go over up to this "
                         "many of the meshes that the mesh was cut from, in the single-precision hypre (a prior's "
                         "h_levels).  Default: HIPPYMFEM_PRIOR_H where it is set, else every mesh of a mesh built "
                         "by cuts whose last cut is in two, with that hypre loaded; 0: BoomerAMG on the operator")
    ap.add_argument("--rank-table", action="store_true",
                    help="after the steps: every rank's own times (its stages, its waits at the collectives, two "
                         "kernels on its card) and its side of the card's memory gathered into OUT.ranks.json, "
                         "and their spread into the record")
    ap.add_argument("--newton-only", action="store_true",
                    help="skip the separately timed stages; synthetic data, then the Newton steps")
    ap.add_argument("--warm-up", action="store_true",
                    help="with --newton-only: build every kernel and pattern once before the step timer, "
                         "as the timed stages of a full run do, so the Newton steps are warm")
    args = ap.parse_args()

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))
    # The benchmark's own defaults for a mesh built by cuts, since the eighth series on Frontier
    # (12 October 2026): the small geometric levels of the p-multigrid in host memory, and the
    # prior's cycles over the meshes.  Both are settings of the library that are off there; the
    # solvers and the prior read them from the environment when they are made, below.
    cuts = [f for f in str(args.h_levels).split(",") if f.strip()]
    by_cuts = args.coarse is not None and (len(cuts) > 1 or (len(cuts) == 1 and int(cuts[0]) > 0))
    last_in_two = by_cuts and (len(cuts) == 1 or int(cuts[-1]) == 2)
    if args.pmg_h_host is not None:
        os.environ["HIPPYMFEM_PMG_H_HOST"] = str(int(args.pmg_h_host))
    elif by_cuts and os.environ.get("HIPPYMFEM_PMG_H_HOST", "") == "":
        os.environ["HIPPYMFEM_PMG_H_HOST"] = "5000"
    if args.prior_h is not None:
        os.environ["HIPPYMFEM_PRIOR_H"] = str(int(args.prior_h))
    elif last_in_two and os.environ.get("HIPPYMFEM_PRIOR_H", "") == "" and single_mod.HYPRE_SINGLE:
        os.environ["HIPPYMFEM_PRIOR_H"] = str(len(cuts) if len(cuts) > 1 else int(cuts[0]))
    # what MFEM actually runs on: "cuda" or "gpu" means the backend the build has (hip on AMD)
    from hippymfem.common.mfemconfig import mfem_gpu_backend
    mfem_dev = args.device if args.device in ("cpu", None) else (mfem_gpu_backend() or args.device)
    N, ORDER = args.n, args.order
    t_start = time.perf_counter()
    t_launch = since_launch()
    serial = None
    if args.coarse is not None:
        from hippymfem.fem.meshes import box_mesh as cube_mesh

        h_cuts = [int(f) for f in str(args.h_levels).split(",") if f.strip()]
        h_cuts = h_cuts if len(h_cuts) > 1 else [2] * (h_cuts[0] if h_cuts else 0)
        pmesh, n0, grid = cube_mesh(N, COMM, coarse=(args.coarse or None), levels=h_cuts)
        say("  mesh built in parallel from %d^3, %s%s"
            % (n0, "equal boxes %s" % (grid,) if grid else "METIS on that mesh",
               ", cut by %s one after the other (the meshes between kept)" % " x ".join("%d" % f for f in h_cuts) if h_cuts else ""))
    else:
        serial = mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON)
    if serial is None:
        pass
    elif args.cart_part:
        # the most cubic grid of ranks that divides the rank count
        p = COMM.size
        grid = min(((a, b, p // (a * b)) for a in range(1, p + 1) if p % a == 0
                    for b in range(a, p // a + 1) if (p // a) % b == 0 and b <= p // (a * b)),
                   key=lambda g: max(g) / min(g))
        nxyz = mfem.intArray(list(grid))
        pmesh = mfem.ParMesh(COMM, serial, serial.CartesianPartitioning(nxyz.GetData()))
        say("  Cartesian partition %s" % (grid,))
    else:
        pmesh = mfem.ParMesh(COMM, serial)
    del serial
    t_mesh = time.perf_counter() - t_start
    ne = COMM.gather(pmesh.GetNE(), root=0)
    say("  [%.0f s] parallel mesh built: %d to %d elements a rank; the clock started %.0f s into the process"
        % (t_mesh, min(ne or [0]), max(ne or [0]), t_launch))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, %d elem/rank"
        % (N, ORDER, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), COMM.size, pmesh.GetNE()))
    say("  kernels on %s (%s), MFEM on %s, parmat %s, pin %s%s"
        % (kernel_mod.device(), gpu_name(), mfem_dev,
           os.environ.get("HIPPYMFEM_PARMAT", "auto"),
           "rank->%s" % _jaxconfig.PINNED_DEVICE if _jaxconfig.PINNED_DEVICE is not None
           else (_jaxconfig.PIN_SKIPPED or "launcher/single device"),
           ", matrix-free linearization points" if args.matrix_free else ""))

    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc.homogeneous(), is_fwd_linear=True,
                                   symmetric_jacobian=(True if args.symmetric_jacobian
                                                       else "auto"),
                                   release_linearization_on_move=args.release_linearization)
    if args.single_refine_goal is not None:
        pde.SINGLE_REFINE_GOAL = float(args.single_refine_goal)
    if args.single_give_up is not None:
        pde.SINGLE_GIVE_UP = args.single_give_up
    # the three solves the records were taken with; the adjoint keeps its default
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"),
                    **({"amg_options": args.amg} if args.amg is not None else {}))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov",
                                lumped_mass=bool(args.prior_lumped_mass))
    if args.prior_amg is not None:
        # the prior built its hierarchy with the defaults: once more, with these
        As = prior.Rsolver.Asolver
        As.parameters["amg_options"] = args.prior_amg
        As.set_operator(As.A)
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, args.ntargets) for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)
    if args.matrix_free:
        # every linearization point, the timed stages' and the Newton-CG solver's alike
        import functools

        model.setPointForHessianEvaluations = functools.partial(
            model.setPointForHessianEvaluations, matrix_free=True)

    # ---- the pieces of a Newton-CG iteration, at the prior mean
    t_data = time.perf_counter() - t_start
    say("  [%.0f s] synthetic data forward solve done" % t_data)
    stage_norms = stage_passes = None
    if args.newton_only:
        t_fwd = t_adj = t_grad = t_hess = t_hess_warm = t_apply = t_apply_cold = float("nan")
        if args.warm_up:
            # the pieces a full run times before its Newton steps, run once and kept out of
            # the step timer, so the steps are warm like every other row of a comparison
            x = [model.generate_vector(STATE), prior.mean.copy(), model.generate_vector(ADJOINT)]
            t_fwd, _ = timed(lambda: model.solveFwd(x[STATE], x))
            t_adj, _ = timed(lambda: model.solveAdj(x[ADJOINT], x))
            g = model.generate_vector(PARAMETER)
            t_grad, _ = timed(lambda: model.evalGradientParameter(x, g))
            t_hess, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
            H = hm.ReducedHessian(model)
            v = model.generate_vector(PARAMETER); hm.parRandom.normal(1.0, v)
            w = model.generate_vector(PARAMETER)
            t_apply_cold, _ = timed(lambda: H.mult(v, w))
            say("  [%.0f s] warm-up, outside the step timer: forward solve %.3f s | adjoint solve %.3f s | "
                "gradient %.3f s | Hessian blocks %.3f s (cold) | reduced-Hessian apply %.3f s (cold)"
                % (time.perf_counter() - t_start, t_fwd, t_adj, t_grad, t_hess, t_apply_cold))
            del H, v, w, g, x
            pde.release_linearization_point()       # the steps assemble their own blocks
    else:
        x = [model.generate_vector(STATE), prior.mean.copy(), model.generate_vector(ADJOINT)]
        t_fwd, _ = timed(lambda: model.solveFwd(x[STATE], x))
        t_adj, _ = timed(lambda: model.solveAdj(x[ADJOINT], x))
        # (the passes of the two solves where they are refined: three, unless single
        # precision makes less of a pass at this size)
        stage_passes = {"forward": int(getattr(pde, "fwd_iterations", 0) or 0),
                        "adjoint": int(getattr(pde, "adj_iterations", 0) or 0)}
        g = model.generate_vector(PARAMETER)
        t_grad, _ = timed(lambda: model.evalGradientParameter(x, g))
        t_hess, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
        # The first linearization point pays one-time symbolic work -- on more than one
        # rank, the true-dof exchange pattern of every block -- so the cold and the warm
        # cost are both reported; a Newton run pays the warm one from its second step.
        x[PARAMETER].axpy(1e-3, prior.mean)          # move the point, so nothing is reused
        t_hess_warm, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
        H = hm.ReducedHessian(model)
        v = model.generate_vector(PARAMETER); hm.parRandom.normal(1.0, v)
        w = model.generate_vector(PARAMETER)
        # The first apply is timed too: it pays whatever was deferred to first use (solver
        # setup, compilation), so the cold and the warm apply are both recorded.
        t_apply_cold, _ = timed(lambda: H.mult(v, w))
        t_apply, _ = timed(lambda: H.mult(v, w))
        say("  forward solve %.3f s | adjoint solve %.3f s | gradient %.3f s | Hessian blocks %.3f s "
            "(warm %.3f s) | reduced-Hessian apply %.3f s (cold %.3f s) | passes of the two solves %d, %d"
            % (t_fwd, t_adj, t_grad, t_hess, t_hess_warm, t_apply, t_apply_cold,
               stage_passes["forward"], stage_passes["adjoint"]))
        # what two runs of one problem must agree on whatever their precision or the tolerance
        # of their incremental solves (the two steps do not: a line search that backtracks takes
        # another path for a direction that differs in its fifth digit): the state, the adjoint
        # and the gradient at the prior mean, and one reduced-Hessian action on a fixed vector
        stage_norms = {"state": x[STATE].norm("l2"), "adjoint": x[ADJOINT].norm("l2"), "gradient": g.norm("l2"),
                       "action": w.norm("l2"), "curvature": v.inner(w), "direction": v.norm("l2")}
        say("  at the prior mean: |u| %.12e  |p| %.12e  |g| %.12e  |Hv| %.8e  v.Hv %.8e"
            % (stage_norms["state"], stage_norms["adjoint"], stage_norms["gradient"], stage_norms["action"],
               stage_norms["curvature"]))
        say("  [%.0f s] the stages done" % (time.perf_counter() - t_start))
    if not args.newton_only and any(k in str(kernel_mod.device()).lower() for k in ("cuda", "gpu", "rocm")):
        # JAX's side of the card, so the split between it and hypre is measured: the
        # allocator keeps its high-water mark, so "peak" is what hypre never gets back.
        try:
            import jax

            st = jax.devices()[0].memory_stats() or {}
            say("  JAX device memory: %.2f GB in use, %.2f GB peak, cap %.2f GB"
                % (st.get("bytes_in_use", 0) / 2 ** 30, st.get("peak_bytes_in_use", 0) / 2 ** 30,
                   st.get("bytes_limit", 0) / 2 ** 30))
        except Exception:                                        # noqa: BLE001
            pass

    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = args.steps
    params["globalization"] = "LS"
    params["GN_iter"] = args.gn_iter
    params["cg_max_iter"] = args.cg_max
    params["cg_coarse_tolerance"] = args.cg_tol
    params["print_level"] = -1
    if args.cg_reorth is not None:
        params["cg_reorthogonalize"] = bool(args.cg_reorth)
    if args.cg_prec_tol is not None:
        params["cg_preconditioner_tolerance"] = args.cg_prec_tol
    if args.cg_prec_cycles is not None:
        params["cg_preconditioner_cycles"] = args.cg_prec_cycles
    if args.cg_relax is not None:
        params["cg_hessian_relaxation"] = args.cg_relax
    if args.inc_tol is not None:
        for attr in ("solver_fwd_inc", "solver_adj_inc"):
            getattr(pde, attr).parameters["rel_tolerance"] = args.inc_tol
    solver = hm.ReducedSpaceNewtonCG(model, params)
    # what the steps are made of: every solve of the problem's solvers, counted and timed on
    # rank 0, so that two runs whose line searches or CG counts differ can still be compared
    tally = {}

    def tallied(name, s):
        if s is None or getattr(s, "_tallied", False):
            return
        real = s.solve

        def solve(x, b):
            # a solve that sets its preconditioner up first is counted apart: the setup
            # is most of it, and how many there are depends on the line searches
            key = name + ("+setup" if getattr(s, "_setup_pending", False) and getattr(s, "_setup_due", True) else "")
            t0 = time.perf_counter()
            out = real(x, b)
            t = tally.setdefault(key, [0, 0, 0.0])
            t[0] += 1
            t[1] += int(getattr(s, "iterations", 0) or 0)
            t[2] += time.perf_counter() - t0
            return out

        s.solve, s._tallied = solve, True

    for name in ("solver", "solver_adj", "solver_fwd_inc", "solver_adj_inc"):
        tallied(name, getattr(pde, name, None))
    # the prior's Krylov solver, twice in every application of the CG's preconditioner
    tallied("prior", getattr(getattr(prior, "Rsolver", None), "Asolver", None))
    # and the model's own stages, whose time beyond the solves is assembly and products; a
    # call that raises (a forward solve at a point the line search then rejects) is counted
    # apart, so that the time of the steps adds up
    stages = {}

    import functools

    def staged(owner, attr, name=None):
        real = getattr(owner, attr)
        name = name or attr

        @functools.wraps(real)       # (the library reads set_operator's signature: _takes_donor)
        def call(*a, **k):
            t0, ok = time.perf_counter(), False
            pool = hm.common.mfemconfig.HYPRE_POOL
            d0 = (pool.driver_seconds, pool.driver_bytes) if pool is not None else (0.0, 0)
            try:
                out = real(*a, **k)
                ok = True
                return out
            finally:
                # (calls, seconds, calls that raised, their seconds, and of hypre's device
                # memory inside the stage: the driver's seconds and the bytes it was asked for)
                t = stages.setdefault(name, [0, 0.0, 0, 0.0, 0.0, 0])
                t[0 if ok else 2] += 1
                t[1 if ok else 3] += time.perf_counter() - t0
                if pool is not None:
                    t[4] += pool.driver_seconds - d0[0]
                    t[5] += pool.driver_bytes - d0[1]

        setattr(owner, attr, call)

    for name in ("solveFwd", "solveAdj", "evalGradientParameter", "setPointForHessianEvaluations", "cost",
                 "solveFwdIncremental", "solveAdjIncremental"):
        staged(model, name)
    # an application of the reduced Hessian (the two incremental solves are inside it) and one
    # of the CG's preconditioner, the prior's covariance: together a CG iteration
    staged(hm.ReducedHessian, "mult", "hessian")
    if getattr(prior, "Rsolver", None) is not None:
        staged(prior.Rsolver, "solve", "preconditioner")
    # with --cg-prec-cycles the preconditioner is another object of the prior, built here (its
    # interval is estimated once, outside the steps) and counted under the same names
    cheb, prior_h_rows = None, []
    if int(params["cg_preconditioner_cycles"] or 0) > 0:
        pc = prior.getHessianPreconditioner(cycles=int(params["cg_preconditioner_cycles"]))
        if pc is not prior.Rsolver and isinstance(getattr(pc, "Asolver", None), hm.ChebyshevSolver):
            cheb = pc.Asolver
            tallied("prior", cheb)
            staged(pc, "solve", "preconditioner")
            say("  the CG's preconditioner: %d cycles a solve of the prior, eigenvalues of the cycle in "
                "[%.3f, %.3f] (Ritz values %.3f, %.3f), error bound %.1e"
                % ((cheb.degree,) + cheb.interval + cheb.ritz + (cheb.error_bound,)))
            geo = getattr(pc, "geometric", None)
            if geo is not None:
                # (HIPPYMFEM_PRIOR_H: the cycles over the meshes, in the single-precision library)
                prior_h_rows = [int(r) for r in geo.rows]
                say("  the prior's cycles: over the coarser meshes, %s rows of linear elements, in single "
                    "precision%s" % (", ".join("%d" % r for r in prior_h_rows),
                                     "" if geo._host_at is None else "; in host memory from level %d on" % geo._host_at))
    # what a forward solve and an assembly of the blocks are made of, by the library's own
    # functions (names with a +: they run inside the stages above and inside one another).
    # On a device a function that only queues work is charged little, and the wait goes to
    # the next one that reads a result: the checks for finite values and the collectives.
    import gc as _gc
    from hippymfem.algorithms.linSolvers import KrylovSolver as _Krylov

    _pvp = sys.modules[type(pde).__module__]          # (the package exports the class under the module's name)

    for owner, attr in ((pde, "_jacobian"), (pde, "_residual"), (pde, "_block"), (pde, "_blocks_streamed"),
                        (pde, "_probe_symmetry"), (pde, "_release_operators"), (pde, "_locals"),
                        (_pvp, "assemble_matrix"), (_pvp, "assemble_vector"), (_pvp, "assemble_scalar"),
                        (_pvp, "scatter_many"), (_pvp, "finish_block"), (_pvp, "plan_block"),
                        (_pvp, "require_finite_arrays"), (getattr(_pvp, "_FiniteFlag", None), "check"),
                        (_Krylov, "set_operator"), (_gc, "collect"),
                        (model, "applyC"), (model, "applyCt"), (model, "applyWuu"), (model, "applyWum"),
                        (model, "applyWmu"), (model, "applyWmm"), (model, "applyR")):
        if owner is not None and hasattr(owner, attr):
            staged(owner, attr, "+" + attr.lstrip("_"))
    # The Jacobian with the products of a p-multigrid (pde._jacobian_with_products) takes its
    # three functions from the module that defines them, at every call, and not by the names of
    # the problem's module staged above: those count the blocks of a linearization point.  The
    # Jacobian's are counted under names of their own, staged in that module while such a call
    # runs and no longer (assemble_matrix plans and finishes its block through the same names,
    # inside its own stage).
    if hasattr(pde, "_jacobian_with_products"):
        from hippymfem.fem import csrassemble as _csr

        plain = {attr: getattr(_csr, attr) for attr in ("plan_block", "scatter_derived", "finish_block")}
        for attr in plain:
            staged(_csr, attr, "+jacobian " + attr)
        counted = {attr: getattr(_csr, attr) for attr in plain}
        vars(_csr).update(plain)
        with_products = pde._jacobian_with_products

        @functools.wraps(with_products)
        def with_stages(*a, **k):
            vars(_csr).update(counted)
            try:
                return with_products(*a, **k)
            finally:
                vars(_csr).update(plain)

        pde._jacobian_with_products = with_stages
        staged(pde, "_jacobian_with_products", "+jacobian_with_products")
    staged(misfit, "cost", "+misfit cost")
    staged(prior, "cost", "+prior cost")
    calls0 = dict(pde.n_calls)
    _pool = hm.common.mfemconfig.HYPRE_POOL
    pool0 = _pool.stats() if _pool is not None else None
    t_step, xs = timed(lambda: solver.solve([None, prior.mean.copy(), None]))
    # what hypre asked of the pool and of the driver in the steps (this rank)
    pool_steps = ({k: _pool.stats()[k] - pool0[k] for k in ("requests", "from_pool", "driver_allocs", "driver_frees",
                                                             "driver_bytes", "driver_seconds", "trims")}
                  if pool0 is not None else None)
    say("  [%.0f s] the Newton steps done" % (time.perf_counter() - t_start))
    calls = {k: pde.n_calls[k] - calls0.get(k, 0) for k in pde.n_calls}
    say("  in the steps: %s; %s"
        % (", ".join("%d %s" % (v, k) for k, v in calls.items()),
           "; ".join("%s %d solves, %d iterations, %.2f s" % (k, v[0], v[1], v[2]) for k, v in tally.items())))
    say("  stages of the model: %s" % "; ".join(
        "%s %d, %.2f s%s" % (k, v[0], v[1], " (and %d that raised, %.2f s)" % (v[2], v[3]) if v[2] else "")
        for k, v in stages.items() if not k.startswith("+")))
    say("  inside them: %s" % "; ".join("%s %d, %.2f s" % (k[1:], v[0], v[1]) for k, v in stages.items() if k.startswith("+")))
    err = (mtrue.copy().axpy(-1.0, xs[PARAMETER]).norm("l2") / max(mtrue.norm("l2"), 1e-300))
    # the meshes under the linear elements of the p-multigrid, where it has them (pmg_h)
    h_rows = [int(t.P.global_cols) for t in (getattr(getattr(pde, "_ptransfer", None), "h", None) or ())]
    h_host, h_timer, h_gathered = {}, None, {}
    if h_rows or prior_h_rows:
        from hippymfem.algorithms import singlesolve as _ss

        # (the ranks' solvers of a last matrix gathered in host memory, compared at each set-up:
        # the last one's.  Zero by construction; cycles that were weaker in one job alone left
        # nothing to tell why, "Probe 98" of the scaling study.)
        h_gathered = {kind: dict(g) for kind, g in sorted(_ss.GATHERED.items())}
        for kind, g in h_gathered.items():
            say("  the ranks' solvers of the gathered last matrix of %s (%d rows): their solutions of one right-hand "
                "side lie %.1e of their size apart"
                % ("the p-multigrid" if kind == "PMGEngine" else "the prior's cycles" if kind == "HCycle" else kind,
                   g["rows"], g["spread"]))
    if h_rows:
        h_host = dict(_ss.HOST_LEVELS.get("PMGEngine") or {})    # (the prior's cycle has its own, above)
        say("  the p-multigrid's coarser meshes: %s rows of linear elements, BoomerAMG on the last%s"
            % (", ".join("%d" % r for r in h_rows),
               "; in host memory from the matrix of %d rows on" % h_host["rows"] if h_host else ""))
        if _ss.TIMERS:
            # (HIPPYMFEM_PMG_TIMERS: a probe's clocks on the corrections from some levels down,
            # rank 0's, for the Jacobian's p-multigrid and for the prior's cycles apart)
            h_timer = [{"cycle": kind, "level": int(k), "calls": int(v[1]), "seconds": float(v[0])}
                       for (kind, k), v in sorted(_ss.TIMERS.items()) if v[1]]
            for kind in sorted(set(t["cycle"] for t in h_timer)):
                say("  the corrections of %s from a level down, timed with the device at rest around each: %s"
                    % ("the p-multigrid" if kind == "PMGEngine" else "the prior's cycles" if kind == "HCycle" else kind,
                       "; ".join("level %d: %d, %.3f ms each" % (t["level"], t["calls"], 1e3 * t["seconds"] / t["calls"])
                                 for t in h_timer if t["cycle"] == kind)))
    say("  %d Newton step(s) %.2f s  newton %d  cg %d  J %.8e  |grad| %.4e  err(m) %.8f  (%s)"
        % (args.steps, t_step, solver.it, solver.total_cg_iter, solver.final_cost,
           solver.final_grad_norm, err, solver.termination_reasons[solver.reason]))
    # the card at the end of the steps, on rank 0: what JAX's allocator holds (it gives nothing
    # back, so its peak is what hypre never gets) and what everything on the card holds; read
    # before the kernels of --rank-table, which take memory of their own
    jax_mem, card = None, None
    try:
        import jax

        st = jax.devices()[0].memory_stats() or {}
        jax_mem = {k: int(st[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved", "peak_bytes_reserved",
                                           "bytes_limit", "pool_bytes", "peak_pool_bytes", "largest_alloc_size") if k in st}
        card = hm.common.devicebridge.device_memory()
    except Exception:                                            # noqa: BLE001
        pass
    # Every rank's own times.  The stages above are rank 0's, and a rank that is done waits
    # at the next collective for the slowest one: so the times of a run are those of its
    # slowest card, and "+check" (the collective after an element pass) on a rank is how
    # far behind it the slowest rank was.
    rank_spread = None
    if args.rank_table:
        # (the memory first: the probe's arrays are 1.5 GiB that JAX's allocator then keeps)
        memory = card_memory()
        probe = card_probe()
        table = COMM.gather({"host": platform.node(),
                             "gpu": os.environ.get("ROCR_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")),
                             # (calls, seconds, and of those the seconds and bytes of the pool's
                             # requests to the driver)
                             "stages": {k: [v[0], v[1], v[4], v[5]] for k, v in stages.items()}, "probe": probe,
                             "memory": memory}, root=0)
        if RANK == 0:
            rank_spread = spread_summary(table)
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(os.path.splitext(args.out)[0] + ".ranks.json", "w") as f:
                    json.dump(table, f)
            for k in ("probe_triad", "probe_exp", "+scatter_many", "+check"):
                v = rank_spread.get(k)
                if v:
                    say("  over the ranks, %s: least %.4f s, median %.4f, largest %.4f (%+.1f %% of the median, on %s)"
                        % (k, v["min"], v["median"], v["max"], 100.0 * (v["max"] / v["median"] - 1.0) if v["median"] else 0.0,
                           v["host_of_max"]))
    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N, "order": ORDER,
           "jax_memory": jax_mem, "card_used_total": list(card) if card else None, "stage_norms": stage_norms,
           "tdofs": Vu.GlobalTrueVSize(), "mdofs": Vm.GlobalTrueVSize(), "mfem_device": mfem_dev,
           "symmetric_jacobian": bool(args.symmetric_jacobian),
           "release_linearization": bool(args.release_linearization), "parmat": os.environ.get("HIPPYMFEM_PARMAT", "auto"),
           "matrix_free": bool(args.matrix_free),
           "t_fwd": t_fwd, "t_adj": t_adj, "t_grad": t_grad, "t_hess_blocks": t_hess,
           "t_hess_blocks_warm": t_hess_warm,
           "t_hess_apply": t_apply, "t_hess_apply_cold": t_apply_cold, "steps": args.steps, "t_steps": t_step,
           "newton_it": solver.it, "cg_it": solver.total_cg_iter, "J": solver.final_cost,
           "cg_reorthogonalize": bool(params["cg_reorthogonalize"]), "inc_tol": args.inc_tol,
           "cg_preconditioner_cycles": int(params["cg_preconditioner_cycles"] or 0),
           "cg_preconditioner_interval": list(cheb.interval) if cheb is not None else None,
           "gradnorm": solver.final_grad_norm, "err_m": err,
           "cart_part": bool(args.cart_part), "newton_only": bool(args.newton_only),
           "coarse": args.coarse, "t_mesh": t_mesh, "t_to_data": t_data, "t_total": time.perf_counter() - t_start,
           "t_launch": t_launch, "amg_options": os.environ.get("HIPPYMFEM_AMG", ""),
           # the precision of the element matrices, and whether the Jacobian and its solves
           # were in the single-precision hypre (a library named, loaded and taken by the PDE)
           "precision": hm.fem.kernel.PRECISION,
           "hypre_single": bool(pde._wants_single()),
           "single_amg": os.environ.get("HIPPYMFEM_SINGLE_AMG", ""),
           # the preconditioner of the Jacobian's single-precision solves: BoomerAMG on the
           # matrix (SingleEngine) or the p-multigrid (PMGEngine, with HIPPYMFEM_PMG=1)
           "single_engine": type(getattr(pde.solver, "_pc", None)).__name__ if pde._wants_single() else None,
           "single_refine_goal": float(pde.SINGLE_REFINE_GOAL), "single_give_up": pde.SINGLE_GIVE_UP,
           # the geometric levels of the p-multigrid: the halvings the mesh was built by and
           # kept, and the rows of the linear elements on the coarser meshes in use
           "h_levels": str(args.h_levels), "pmg_h_rows": h_rows, "pmg_h_host": h_host, "pmg_h_timer": h_timer,
           "pmg_h_gathered": h_gathered,
           "prior_h_rows": prior_h_rows,
           "single_blocks": bool(tdof_mod.SINGLE_BLOCKS),
           # element passes that ran out of memory on a card and were tried again in smaller
           # chunks, over all ranks: the rank keeps the smaller chunk and is slower from then on
           "element_retries": int(COMM.allreduce(int(kernel_mod.RETRIES), op=MPI.SUM)),
           "pmg_h_host_setting": int(os.environ.get("HIPPYMFEM_PMG_H_HOST", "0") or 0),
           "prior_h_setting": int(os.environ.get("HIPPYMFEM_PRIOR_H", "0") or 0),
           # how the element Hessians were differentiated, and two choices of the Jacobian in
           # the single-precision hypre: the defect of its row sums kept for the solver's
           # product, and the products of the p-multigrid made from its element matrices
           "hessian_mode": hm.fem.kernel.HESSIAN_MODE, "stage_passes": stage_passes,
           "single_row_sums": bool(tdof_mod.SINGLE_ROW_SUMS),
           "pmg_element_products": bool(single_mod.PMG_ELEMENT_PRODUCTS),
           "gpu_mem_fraction": os.environ.get("HIPPYMFEM_GPU_MEM_FRACTION", ""),
           "amg": args.amg, "prior_amg": args.prior_amg, "prior_lumped_mass": bool(args.prior_lumped_mass),
           "device_exchange": tdof_mod.DEVICE_EXCHANGE,
           "amg_relax_env": os.environ.get("HIPPYMFEM_AMG_RELAX", ""),
           "calls_in_steps": calls, "solves_in_steps": {k: {"solves": v[0], "iterations": v[1], "seconds": v[2]} for k, v in tally.items()},
           "gpu_aware_solves": bool(hm.common.mfemconfig.GPU_AWARE_SOLVES),
           "stages_in_steps": {k: {"calls": v[0], "seconds": v[1], "failed": v[2], "failed_seconds": v[3],
                                   "driver_seconds": v[4], "driver_bytes": v[5]}
                               for k, v in stages.items()},
           "elements_per_rank": [int(min(ne)), int(max(ne))] if ne else None,
           "gpu_aware_mpi": bool(hm.common.mfemconfig.hypre_gpu_aware_mpi()),
           "hypre_spmv": hm.common.mfemconfig.HYPRE_SPMV,
           "hypre_pool": (hm.common.mfemconfig.HYPRE_POOL.stats()
                          if hm.common.mfemconfig.HYPRE_POOL is not None else None),
           "hypre_pool_steps": pool_steps,
           "rank_spread": rank_spread,
           "warm_up": bool(args.warm_up)}
    def write():
        if args.out and RANK == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(rec, f, indent=2)

    write()
    if args.profile_setup:
        # after the record is written: a setup that runs out of device memory ends the run
        from bench_scaling import profile_setup

        A = getattr(pde.solver_fwd_inc, "A", None) or getattr(pde.solver, "A", None)
        if A is not None and not isinstance(A, mfem.HypreParMatrix):
            # (the setups are of MFEM's BoomerAMG, which has no matrix of its own to take here)
            say("  no profile of BoomerAMG: the Jacobian is in the single-precision hypre")
        elif A is not None and getattr(pde.solver, "precond_type", None) == "amg":
            b = model.generate_vector(STATE)
            hm.parRandom.normal(1.0, b)
            rec["setup_profile"] = profile_setup(args.profile_setup, A, pde.solver._make_pc, b,
                                                 model.generate_vector(STATE))
            write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

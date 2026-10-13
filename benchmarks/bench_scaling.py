#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Strong-scaling profile: where a Newton-CG step loses parallel efficiency.

``bench_newton_device.py`` times the stages; this one takes a reduced-Hessian action
apart into the nine operations it is made of and records, for every Krylov solve, the
number of iterations it took.  That separates the two ways a rank count can cost
efficiency:

* **iteration count** -- BoomerAMG is a weaker preconditioner on more subdomains, so a
  solve takes more iterations.  Visible as ``it_*``.
* **time per iteration** -- launch latency, halo exchanges and the global reductions of
  CG, which do not shrink with the subdomain.  Visible as ``t_*`` divided by ``it_*``.

Run the same problem at 1, 2 and 4 devices and compare::

    for r in 1 2 4; do
      HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n $r tools/mpirun_pinned.sh \
        python benchmarks/bench_scaling.py --n 64 --device cuda \
        --out results/scaling_n64_r$r.json
    done

The problem is that of ``bench_newton_device.py``: same mesh, PDE, prior, observations
and solver settings, so the stage times are comparable with those records.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

import mfem.par as mfem                                              # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

from hippymfem.fem import kernel as kernel_mod                       # noqa: E402
from hippymfem.modeling.variables import PARAMETER, STATE, ADJOINT   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def say(*a):
    if RANK == 0:
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


def timed(fn, sync=None):
    """Time ``fn`` with a barrier on each side.

    ``sync`` is a vector whose norm is taken before the closing barrier: an
    MPI reduction has to have the local value on the host, so it completes the device
    work that ``fn`` queued.  Without it a device-side product is timed as the launch.
    """
    COMM.Barrier()
    t0 = time.perf_counter()
    out = fn()
    if sync is not None:
        sync.norm("l2")
    COMM.Barrier()
    return time.perf_counter() - t0, out


def iters(solver):
    """Krylov iterations of the last solve on a hIPPyMFEM solver (0 if direct)."""
    return int(getattr(solver, "iterations", 0) or 0)


def profile_setup(spec, A, make_pc, b, xw):
    """BoomerAMG setups of the matrix ``A`` with other settings, each timed with a solve.

    ``spec`` is ``"name:settings;name:settings"``, the settings as for ``--amg`` on top of
    what ``make_pc`` sets (a solver's own), with ``spgemm=0|1`` for hypre's own sparse
    product or the vendor's, which stays in force for the sets after it, and ``debug=1``
    for hypre's printout of the phases of a setup by level.  For each set, twice: a new
    preconditioner in hypre's PCG, the first solve to 1e-8 (which sets the hierarchy up,
    with hypre's memory pool open as a solver of the library has it) and a second one; the
    setup is their difference.  Then ten V-cycles, after one that is not timed: MFEM sets a
    preconditioner up at the first application of its own whatever hypre's PCG did before.
    Returns ``{name: {"settings", "runs"}}`` and prints the faster of the two setups of each
    set.
    """
    import ctypes

    mc = hm.common.mfemconfig
    H = mc._hypre_library()
    sprof, sets = {}, []
    for item in spec.split(";"):                     # (read them all first: a mistyped one stops here)
        name, _, text = item.partition(":")
        toks = [t.strip() for t in text.split(",") if t.strip()]
        sets.append((name.strip(), text, [int(t.split("=")[1]) for t in toks if t.startswith("spgemm=")],
                     mc.parse_boomeramg_options(",".join(t for t in toks if not t.startswith("spgemm=")),
                                                "--profile-setup")))
    for name, text, vendor, opts in sets:
        for v in vendor:
            H.HYPRE_SetSpGemmUseVendor(ctypes.c_int(v))
        runs = []
        try:
            for _ in range(2):
                pc = make_pc(A)
                mc.set_boomeramg_options(pc, opts)
                cg = mfem.HyprePCG(A)
                cg.SetTol(1e-8)
                cg.SetMaxIter(200)
                cg.SetPrintLevel(0)
                cg.SetPreconditioner(pc)
                xw.zero()
                mc.hypre_pool_open()
                t1, _ = timed(lambda: cg.Mult(b.hypre, xw.hypre), xw)
                mc.hypre_pool_close()
                xw.zero()
                t2, _ = timed(lambda: cg.Mult(b.hypre, xw.hypre), xw)
                its = mfem.intp()
                cg.GetNumIterations(its)
                # MFEM sets a preconditioner up at the first application of its own, whatever
                # hypre's PCG did before (ChebyshevSolver.setup): that application is made here,
                # outside the clock of the ten, with the pool open as for any setup.
                mc.hypre_pool_open()
                timed(lambda: pc.Mult(b.hypre, xw.hypre), xw)
                mc.hypre_pool_close()
                t10, _ = timed(lambda: [pc.Mult(b.hypre, xw.hypre) for _ in range(10)], xw)
                runs.append({"setup": t1 - t2, "solve": t2, "its": int(its.value()), "cycle": t10 / 10.0,
                             "finite": bool(np.isfinite(xw.norm("l2")))})
                del cg, pc
        except Exception as e:                                           # noqa: BLE001
            say("  setup %-12s FAILED: %s" % (name, " ".join(str(e).split())[:160]))
            continue
        best = min(runs, key=lambda r: r["setup"])
        sprof[name] = {"settings": text, "runs": runs}
        say("  setup %-12s %6.3f s (the other %.3f) | solve %.3f s, %d it, %.2f ms/it | V-cycle %.2f ms%s | %s"
            % (name, best["setup"], max(r["setup"] for r in runs), best["solve"], best["its"],
               1e3 * best["solve"] / max(best["its"], 1), 1e3 * best["cycle"],
               "" if all(r["finite"] for r in runs) else " | NOT FINITE", text))
    return sprof


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3, help="Hessian actions to time (the median is reported)")
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"), help="mfem.Device kind; 'cpu' keeps hypre on the host")
    ap.add_argument("--amg-relax-type", type=int, default=-1, help="hypre relax type; 16 = Chebyshev")
    ap.add_argument("--amg-max-levels", type=int, default=-1, help="hypre BoomerAMG max levels; -1 = MFEM default (25)")
    ap.add_argument("--amg-agg-levels", type=int, default=-1, help="levels of aggressive coarsening; -1 = MFEM default (1)")
    ap.add_argument("--amg-theta", type=float, default=-1.0, help="strength threshold; -1 = MFEM default (0.25)")
    ap.add_argument("--amg", default=None,
                    help="BoomerAMG settings of the problem's solves alone, such as \"relax_wt=1.8,pmax=6\" (the names "
                         "of mfemconfig.BOOMERAMG_SETTERS); HIPPYMFEM_AMG sets every solver's")
    ap.add_argument("--tag", default="", help="label written into the record")
    ap.add_argument("--cg-tol", type=float, default=1e-12, help="relative tolerance of every Krylov solve")
    ap.add_argument("--both-kernels", action="store_true",
                    help="time the Hessian action a second time with hypre's other "
                         "matrix-vector kernel (vendor library or hypre's own)")
    ap.add_argument("--cart-part", action="store_true",
                    help="partition the box as a Cartesian grid of ranks instead of with METIS")
    ap.add_argument("--coarse", type=int, default=None,
                    help="build the mesh in parallel from a COARSE^3 mesh, in equal boxes when the ranks make "
                         "a grid (bench_newton_device.py); 0 chooses COARSE")
    ap.add_argument("--m-scale", type=float, default=0.0,
                    help="take every operation at this multiple of the synthetic truth instead of the prior "
                         "mean (0): the coefficient exp(m) is then rough, as at the iterates of an inversion, "
                         "and 3 is where plain Jacobi relaxation stops being a smoother")
    ap.add_argument("--profile-vcycle", action="store_true",
                    help="time a V-cycle of the Jacobian's BoomerAMG with 1, 2, ... levels (what a level adds), "
                         "with the exchange staged and, where it can be, on the device; and a dot product and a "
                         "vector update, the rest of a CG iteration")
    ap.add_argument("--profile-setup", default=None,
                    help="BoomerAMG setups of the Jacobian with other settings, each timed with a solve "
                         "to 1e-8 and its V-cycle: 'name:settings;name:settings', the settings as for "
                         "--amg and on top of the solver's own (debug=1 has hypre print the phases of "
                         "a setup by level), and spgemm=0|1 for hypre's own sparse product or the "
                         "vendor's, which stays in force for the sets after it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))

    N, ORDER = args.n, args.order
    t_start = time.perf_counter()
    serial = None
    if args.coarse is not None:
        from applications.geothermal.model import cube_mesh

        pmesh, n0, grid = cube_mesh(N, COMM, coarse=(args.coarse or None))
        say("  mesh built in parallel from %d^3, %s" % (n0, "equal boxes %s" % (grid,) if grid else "METIS on that mesh"))
    else:
        serial = mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON)
    if serial is None:
        pass
    elif args.cart_part:
        # the most cubic grid of ranks that divides the rank count (bench_newton_device.py)
        p = COMM.size
        grid = min(((a, b, p // (a * b)) for a in range(1, p + 1) if p % a == 0
                    for b in range(a, p // a + 1) if (p // a) % b == 0 and b <= p // (a * b)),
                   key=lambda g: max(g) / min(g))
        nxyz = mfem.intArray(list(grid))
        pmesh = mfem.ParMesh(COMM, serial, serial.CartesianPartitioning(nxyz.GetData()))
    else:
        pmesh = mfem.ParMesh(COMM, serial)
    del serial
    t_mesh = time.perf_counter() - t_start
    ne = COMM.gather(pmesh.GetNE(), root=0)
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, %d elem/rank, kernels on %s (%s)"
        % (N, ORDER, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), COMM.size, pmesh.GetNE(),
           kernel_mod.device(), gpu_name()))
    say("  hypre matrix-vector kernel: %s" % hm.common.mfemconfig.HYPRE_SPMV)

    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(), is_fwd_linear=True)
    amg = {}
    if args.amg_relax_type >= 0:
        amg["amg_relax_type"] = args.amg_relax_type
    if args.amg_max_levels > 0:
        amg["amg_max_levels"] = args.amg_max_levels
    if args.amg_agg_levels >= 0:
        amg["amg_agg_levels"] = args.amg_agg_levels
    if args.amg_theta >= 0.0:
        amg["amg_strength_threshold"] = args.amg_theta
    if args.amg is not None:
        amg["amg_options"] = args.amg
    # the three solves the records were taken with; the adjoint keeps its default
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=args.cg_tol,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"),
                    **amg)
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
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
    model = hm.Model(pde, prior, hm.DiscreteStateObservation(B, data, nstd ** 2))

    x = [model.generate_vector(STATE), prior.mean.copy(), model.generate_vector(ADJOINT)]
    if args.m_scale:
        x[PARAMETER].axpy(args.m_scale, mtrue)
        say("  the point is %g times the truth: |m| up to %.2f" % (args.m_scale, x[PARAMETER].norm("linf")))
    try:
        t_fwd, _ = timed(lambda: model.solveFwd(x[STATE], x), x[STATE])
    except RuntimeError as e:
        # (every rank raises: the solver's verdict is collective)
        say("  forward solve FAILED: %s" % " ".join(str(e).split())[:200])
        return 3
    it_fwd = iters(pde.solver)
    t_adj, _ = timed(lambda: model.solveAdj(x[ADJOINT], x), x[ADJOINT])
    it_adj = iters(pde.solver)
    g = model.generate_vector(PARAMETER)
    t_grad, _ = timed(lambda: model.evalGradientParameter(x, g), g)
    t_blocks_cold, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
    x[PARAMETER].axpy(1e-3, prior.mean)
    t_blocks, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))

    # ---- a reduced-Hessian action, operation by operation (hm.ReducedHessian.TrueHessian)
    H = hm.ReducedHessian(model)
    v = model.generate_vector(PARAMETER)
    hm.parRandom.normal(1.0, v)
    w = model.generate_vector(PARAMETER)
    H.mult(v, w)                                                   # warm: compile, set up
    rhs_fwd, rhs_adj, rhs_adj2 = (model.generate_vector(STATE), model.generate_vector(ADJOINT),
                                  model.generate_vector(ADJOINT))
    uhat, phat, yhelp = (model.generate_vector(STATE), model.generate_vector(ADJOINT),
                         model.generate_vector(PARAMETER))
    names = ["applyC", "solveFwdInc", "applyWuu", "applyWum", "solveAdjInc",
             "applyWmm", "applyCt", "applyWmu", "applyR"]

    def action_profile():
        """The nine operations of a Hessian action, timed ``--repeats`` times: the
        medians, and the iteration counts of the two incremental solves."""
        rr, ii = [], []
        for _ in range(args.repeats):
            t = {}
            t["applyC"], _ = timed(lambda: model.applyC(v, rhs_fwd), rhs_fwd)
            t["solveFwdInc"], _ = timed(lambda: model.solveFwdIncremental(uhat, rhs_fwd), uhat)
            it_f = iters(pde.solver_fwd_inc)
            t["applyWuu"], _ = timed(lambda: model.applyWuu(uhat, rhs_adj), rhs_adj)
            t["applyWum"], _ = timed(lambda: model.applyWum(v, rhs_adj2), rhs_adj2)
            rhs_adj.axpy(-1.0, rhs_adj2)
            t["solveAdjInc"], _ = timed(lambda: model.solveAdjIncremental(phat, rhs_adj), phat)
            it_a = iters(pde.solver_adj_inc)
            t["applyWmm"], _ = timed(lambda: model.applyWmm(v, w), w)
            t["applyCt"], _ = timed(lambda: model.applyCt(phat, yhelp), yhelp)
            t["applyWmu"], _ = timed(lambda: model.applyWmu(uhat, yhelp), yhelp)
            t["applyR"], _ = timed(lambda: model.applyR(v, yhelp), yhelp)
            rr.append(t)
            ii.append((it_f, it_a))
        return ({k: float(np.median([r[k] for r in rr])) for k in names},
                int(np.median([a for a, _ in ii])), int(np.median([b for _, b in ii])))

    runs, it_runs = [], []
    for _ in range(args.repeats):
        t = {}
        t["applyC"], _ = timed(lambda: model.applyC(v, rhs_fwd), rhs_fwd)
        t["solveFwdInc"], _ = timed(lambda: model.solveFwdIncremental(uhat, rhs_fwd), uhat)
        it_f = iters(pde.solver_fwd_inc)
        t["applyWuu"], _ = timed(lambda: model.applyWuu(uhat, rhs_adj), rhs_adj)
        t["applyWum"], _ = timed(lambda: model.applyWum(v, rhs_adj2), rhs_adj2)
        rhs_adj.axpy(-1.0, rhs_adj2)
        t["solveAdjInc"], _ = timed(lambda: model.solveAdjIncremental(phat, rhs_adj), phat)
        it_a = iters(pde.solver_adj_inc)
        t["applyWmm"], _ = timed(lambda: model.applyWmm(v, w), w)
        t["applyCt"], _ = timed(lambda: model.applyCt(phat, yhelp), yhelp)
        t["applyWmu"], _ = timed(lambda: model.applyWmu(uhat, yhelp), yhelp)
        t["applyR"], _ = timed(lambda: model.applyR(v, yhelp), yhelp)
        runs.append(t)
        it_runs.append((it_f, it_a))
    op = {k: float(np.median([r[k] for r in runs])) for k in names}
    it_fwd_inc = int(np.median([a for a, _ in it_runs]))
    it_adj_inc = int(np.median([b for _, b in it_runs]))
    t_action = sum(op.values())

    say("  forward %.3f s (%d it) | adjoint %.3f s (%d it) | gradient %.3f s | blocks %.3f s (cold %.3f s)"
        % (t_fwd, it_fwd, t_adj, it_adj, t_grad, t_blocks, t_blocks_cold))
    say("  Hessian action %.4f s = %s" % (t_action, " + ".join("%s %.4f" % (k, op[k]) for k in names)))
    say("  incremental solves: fwd %d it (%.2f ms/it), adj %d it (%.2f ms/it)"
        % (it_fwd_inc, 1e3 * op["solveFwdInc"] / max(it_fwd_inc, 1),
           it_adj_inc, 1e3 * op["solveAdjInc"] / max(it_adj_inc, 1)))
    # ---- one product with the Jacobian, the unit a V-cycle is made of: the diagonal block's
    # product plus one halo exchange.  With HIPPYMFEM_GPU_AWARE_MPI=solve it is timed both
    # ways, hypre staging the exchange through CPU memory and handing MPI the device buffers.
    mc = hm.common.mfemconfig
    A = getattr(pde.solver_fwd_inc, "A", None) or getattr(pde.solver, "A", None)
    # A Jacobian that lives in the single-precision hypre (HIPPYMFEM_HYPRE_SINGLE) is multiplied
    # from here by SingleParMatrix.Mult: the conversion of its two vectors and a reduction of
    # hypre's verdict over the ranks around the product, which is not the unit of a V-cycle.  It
    # is timed under the name of what it is, once: that library always stages its exchange.
    in_single = A is not None and not isinstance(A, mfem.HypreParMatrix)
    t_prod = {}
    if A is not None:
        for name in (["single, with the conversion of its two vectors"] if in_single
                     else ["staged", "aware"] if mc.GPU_AWARE_SOLVES else ["staged"]):
            if name == "aware":
                mc.set_hypre_gpu_aware_mpi(True)
            try:
                A.Mult(uhat.hypre, rhs_fwd.hypre)
                rhs_fwd.norm("l2")
                COMM.Barrier()
                t0 = time.perf_counter()
                for _ in range(50):
                    A.Mult(uhat.hypre, rhs_fwd.hypre)
                rhs_fwd.norm("l2")
                COMM.Barrier()
                t_prod[name] = (time.perf_counter() - t0) / 50.0
            finally:
                if name == "aware":
                    mc.set_hypre_gpu_aware_mpi(False)
        say("  a product with the Jacobian: %s" % ", ".join("%.3f ms (%s)" % (1e3 * v, k) for k, v in t_prod.items()))
    # ---- a V-cycle, level by level: hierarchies of 1, 2, ... levels on the Jacobian with the
    # solver's own settings, each set up once and applied 20 times.  The difference of two
    # consecutive ones is what a level adds (on that level and on the one above it).  The list
    # ends two entries after the hierarchy stops growing (its nonzeros are recorded), since
    # the last levels have a handful of rows and add nothing that the sum can show.
    vprof = None
    # (the two profiles are of MFEM's BoomerAMG: a Jacobian that lives in the single-precision
    # hypre, HIPPYMFEM_HYPRE_SINGLE, is no matrix of MFEM's to build one on)
    of_mfem = isinstance(A, mfem.HypreParMatrix)
    if (args.profile_vcycle or args.profile_setup) and A is not None and not of_mfem:
        say("  no profile of BoomerAMG: the Jacobian is in the single-precision hypre")
    if args.profile_vcycle and of_mfem and getattr(pde.solver, "precond_type", None) == "amg":
        import ctypes

        hypre = mc._hypre_library()                 # (H is the reduced Hessian, applied again below)
        modes = ["staged", "aware"] if mc.GPU_AWARE_SOLVES else ["staged"]
        vprof = {"levels": [], "setup": [], "cum_nnz": [], **{m: [] for m in modes}}
        b, xw = rhs_fwd, model.generate_vector(STATE)
        last, same = None, 0
        for L in range(1, 16):
            pc = pde.solver._make_pc(A)
            mc.set_boomeramg_options(pc, {"levels": L})
            h = ctypes.c_void_p(mc.boomeramg_handle(pc))
            hypre.HYPRE_BoomerAMGSetCumNnzAP.argtypes = [ctypes.c_void_p, ctypes.c_double]
            hypre.HYPRE_BoomerAMGSetCumNnzAP(h, 1.0)
            t_first, _ = timed(lambda: pc.Mult(b.hypre, xw.hypre), xw)      # the setup and one cycle
            nnz = ctypes.c_double(0.0)
            hypre.HYPRE_BoomerAMGGetCumNnzAP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
            hypre.HYPRE_BoomerAMGGetCumNnzAP(h, ctypes.byref(nnz))
            same = same + 1 if (last is not None and nnz.value <= last) else 0
            if same > 2:
                del pc
                break
            last = nnz.value
            cycle = {}
            for name in modes:
                if name == "aware":
                    mc.set_hypre_gpu_aware_mpi(True)
                try:
                    pc.Mult(b.hypre, xw.hypre)
                    t, _ = timed(lambda: [pc.Mult(b.hypre, xw.hypre) for _ in range(20)], xw)
                    cycle[name] = t / 20.0
                finally:
                    if name == "aware":
                        mc.set_hypre_gpu_aware_mpi(False)
            vprof["levels"].append(L)
            vprof["setup"].append(t_first - cycle["staged"])
            vprof["cum_nnz"].append(nnz.value)
            for name in modes:
                vprof[name].append(cycle[name])
            del pc
        t, _ = timed(lambda: [b.inner(xw) for _ in range(50)])
        vprof["t_dot"] = t / 50.0
        t, _ = timed(lambda: [xw.axpy(1e-300, b) for _ in range(50)], xw)
        vprof["t_axpy"] = t / 50.0
        for name in modes:
            say("  a V-cycle with 1, 2, ... levels (%s): %s ms" % (name, " ".join("%.2f" % (1e3 * v) for v in vprof[name])))
        say("  its setup: %s s; nonzeros of all levels over the matrix's: %s"
            % (" ".join("%.3f" % v for v in vprof["setup"]),
               " ".join("%.3f" % (v / vprof["cum_nnz"][0]) for v in vprof["cum_nnz"])))
        say("  a dot product %.3f ms, a vector update %.3f ms" % (1e3 * vprof["t_dot"], 1e3 * vprof["t_axpy"]))
    # ---- the setup of the Jacobian's hierarchy with other settings (profile_setup)
    sprof = None
    if args.profile_setup and of_mfem and getattr(pde.solver, "precond_type", None) == "amg":
        sprof = profile_setup(args.profile_setup, A, pde.solver._make_pc, rhs_fwd, model.generate_vector(STATE))
    solves = op["solveFwdInc"] + op["solveAdjInc"]
    say("  the two solves are %.1f %% of the action; the eight products and R are %.1f %%"
        % (100 * solves / t_action, 100 * (t_action - solves) / t_action))

    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N, "order": ORDER,
           "tdofs": int(Vu.GlobalTrueVSize()), "mdofs": int(Vm.GlobalTrueVSize()),
           "mfem_device": args.device, "amg_relax_type": args.amg_relax_type, "cg_tol": args.cg_tol,
           "amg_max_levels": args.amg_max_levels, "amg_agg_levels": args.amg_agg_levels,
           "amg_theta": args.amg_theta, "amg": args.amg, "tag": args.tag,
           "hypre_spmv": hm.common.mfemconfig.HYPRE_SPMV, "cart_part": bool(args.cart_part),
           "coarse": args.coarse, "t_mesh": t_mesh, "elements_per_rank": [int(min(ne)), int(max(ne))] if ne else None,
           "t_product": t_prod, "vcycle": vprof, "setup_profile": sprof, "gpu_aware_solves": bool(mc.GPU_AWARE_SOLVES),
           "gpu_aware_mpi": bool(hm.common.mfemconfig.hypre_gpu_aware_mpi()),
           # (whether the Jacobian and its solves were in the single-precision hypre)
           "hypre_single": bool(in_single),
           "hypre_pool": (hm.common.mfemconfig.HYPRE_POOL.stats()
                          if hm.common.mfemconfig.HYPRE_POOL is not None else None),
           "t_fwd": t_fwd, "it_fwd": it_fwd, "t_adj": t_adj, "it_adj": it_adj, "t_grad": t_grad,
           "t_blocks": t_blocks, "t_blocks_cold": t_blocks_cold,
           "t_action": t_action, "op": op, "it_fwd_inc": it_fwd_inc, "it_adj_inc": it_adj_inc,
           "repeats": args.repeats, "m_scale": args.m_scale, "amg_options": os.environ.get("HIPPYMFEM_AMG", ""),
           "amg_relax_env": os.environ.get("HIPPYMFEM_AMG_RELAX", "")}
    if args.both_kernels and hm.common.mfemconfig.HYPRE_SPMV is not None:
        # the same Hessian action with hypre's other matrix-vector kernel: a global
        # switch, so the matrices, the hierarchies and the vectors stay as they are
        first = hm.common.mfemconfig.HYPRE_SPMV
        other = "hypre" if first == "vendor" else "vendor"
        hm.common.mfemconfig.set_hypre_spmv(other, COMM)
        H.mult(v, w)                                               # warm the other kernel
        op2, itf2, ita2 = action_profile()
        t2 = sum(op2.values())
        hm.common.mfemconfig.set_hypre_spmv(first, COMM)
        # (the switch is of MFEM's hypre.  A single-precision hypre beside it took its kernel
        # when it was loaded and keeps it, so the two solves of a Jacobian that lives there,
        # most of the action, were not on the other kernel: the line and the record say so)
        kept = first if in_single else None
        say("  with the %s kernel%s: Hessian action %.4f s; incremental solves fwd %d it "
            "(%.2f ms/it), adj %d it (%.2f ms/it)"
            % (other, " in MFEM's hypre (the solves, in the single-precision hypre, keep the %s one)"
               % kept if kept else "", t2, itf2, 1e3 * op2["solveFwdInc"] / max(itf2, 1),
               ita2, 1e3 * op2["solveAdjInc"] / max(ita2, 1)))
        rec["other_kernel"] = {"hypre_spmv": other, "t_action": t2, "op": op2,
                               "it_fwd_inc": itf2, "it_adj_inc": ita2, "single_hypre_spmv": kept}
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

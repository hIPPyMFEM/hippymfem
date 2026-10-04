#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The three precisions of the element kernels: what each costs and what each keeps.

``HIPPYMFEM_PRECISION`` (:data:`hippymfem.fem.kernel.PRECISION`):

* ``fp64``: everything in double precision;
* ``mixed``: element matrices in single precision, element vectors in double, the
  forward and the adjoint solve refined against the double-precision residual;
* ``fp32``: matrices and vectors in single precision, nothing corrected.

For the model problem (``n^3`` second-order hexahedra, ``exp(m) grad u . grad p``,
BiLaplacian prior, pointwise observations) this measures, for each of them and on
whatever device is present:

* the element kernel of the Jacobian and its complete assembly, warm;
* the stages of an iteration, warm: a forward solve at a new parameter, an adjoint
  solve, a gradient, the Hessian blocks at a new linearization point, a Hessian action;
* how far the state, the adjoint, the reduced gradient and a Hessian action are from
  the double-precision ones at the same parameter;
* a Newton-CG solve to its tolerance: iterations, time, the gradient norm reached, and
  the distance of the MAP point from the double-precision one.

Usage::

    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 python benchmarks/bench_precision.py --n 32
    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n 4 tools/mpirun_pinned.sh \\
        python benchmarks/bench_precision.py --n 64 --out results/precision_n64_r4.json
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hm                                               # noqa: E402
from hippymfem.fem import assemble as asm                            # noqa: E402
from hippymfem.fem import kernel as km                               # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402

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
    except Exception:
        return "unknown"


def clock(fn, reps=1):
    """Seconds per call of ``fn``, over ``reps`` calls, all ranks."""
    COMM.Barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    COMM.Barrier()
    return (time.perf_counter() - t0) / reps, out


def rel(a, b):
    """``|a - b| / |b|`` for two vectors."""
    d = a.copy().axpy(-1.0, b)
    return d.norm("l2") / max(b.norm("l2"), 1e-300)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3, help="repetitions of each timed stage")
    ap.add_argument("--modes", default="fp64,mixed,fp32")
    ap.add_argument("--newton-max", type=int, default=25)
    ap.add_argument("--newton-tol", type=float, default=1e-6)
    ap.add_argument("--cg-max", type=int, default=50)
    ap.add_argument("--no-newton", action="store_true", help="the stages and the accuracies only")
    ap.add_argument("--newton-repeats", type=int, default=2,
                    help="the Newton-CG solve is run this many times and the last is reported: the "
                    "first compiles the kernels that only an optimization reaches")
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"))
    ap.add_argument("--symmetric-jacobian", action="store_true")
    ap.add_argument("--newton-print", action="store_true", help="print the Newton-CG iterations")
    ap.add_argument("--print-refinement", action="store_true",
                    help="print the residual after every pass of a refined solve")
    ap.add_argument("--probe-tol-single", type=float, default=None,
                    help="tolerance of the symmetry probe for single-precision matrices (experiments)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    modes = [m for m in args.modes.split(",") if m]
    if modes[0] != "fp64":
        raise SystemExit("the first mode must be fp64: the others are compared with it")

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))
    N, ORDER = args.n, args.order
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    NE = pmesh.GetNE()
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, %d elem/rank, kernels on %s (%s)"
        % (N, ORDER, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), COMM.size, NE, km.device(), gpu_name()))

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    bc0 = bc.homogeneous()
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True,
                                   symmetric_jacobian=(True if args.symmetric_jacobian else "auto"))
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    if args.probe_tol_single is not None:
        pde.SYMMETRY_PROBE_TOL_SINGLE = args.probe_tol_single
    if args.print_refinement:
        pde.newton_parameters["print_level"] = 0
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
    pde.solveFwd(utrue, [utrue, mtrue, None])           # the data, in double precision
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)

    # the point of the comparisons: halfway to the truth, so that the coefficient varies
    mpt = prior.mean.copy().axpy(0.5, mtrue.copy().axpy(-1.0, prior.mean))
    direction = Vm.vector()
    hm.parRandom.set_seed(5)
    hm.parRandom.normal(1.0, direction)

    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N, "order": ORDER,
           "NE_local": NE, "tdofs": Vu.GlobalTrueVSize(), "mdofs": Vm.GlobalTrueVSize(),
           "device": str(km.device()), "mfem_device": args.device, "modes": {}}
    ref = {}
    batches = pde.batches
    loc = None
    for mode in modes:
        km.set_precision(mode)
        pde.invalidate_jacobian()
        pde.release_linearization_point()
        out = rec["modes"][mode] = {}
        say("--- %s" % mode)

        # ---- the Jacobian's element kernel and its complete assembly
        x = [model.generate_vector(STATE), mpt.copy(), model.generate_vector(ADJOINT)]
        loc = pde._locals(x) + pde._aux_locals()

        def kernel_only():
            mats = pde.kernel.element_matrices(ADJOINT, STATE, loc)
            m0 = mats[0]
            return float(jnp.sum(m0[0, 0])) if not isinstance(m0, np.ndarray) else float(m0[0, 0, 0])

        def assembly():
            A = asm.assemble_matrix(Vu, Vu, batches.groups,
                                    pde.kernel.element_matrices(ADJOINT, STATE, loc), NE,
                                    test_ess=bc0.ess_tdof)
            del A
        kernel_only()
        out["t_kernel"], _ = clock(kernel_only, args.reps)
        assembly()
        out["t_assembly"], _ = clock(assembly, args.reps)

        # ---- the stages, warm: every one is run once before it is timed
        def forward():
            pde.invalidate_jacobian()
            model.solveFwd(x[STATE], x)
            return getattr(pde, "fwd_iterations", 1)

        def adjoint():
            model.solveAdj(x[ADJOINT], x)
            return getattr(pde, "adj_iterations", 1)

        g = model.generate_vector(PARAMETER)

        def gradient():
            return model.evalGradientParameter(x, g)

        def blocks():
            pde.release_linearization_point()
            model.setPointForHessianEvaluations(x)

        forward()
        forward()
        out["t_forward"], out["forward_passes"] = clock(forward, args.reps)
        adjoint()
        out["t_adjoint"], out["adjoint_passes"] = clock(adjoint, args.reps)
        gradient()
        out["t_gradient"], _ = clock(gradient, args.reps)
        blocks()
        out["t_blocks"], _ = clock(blocks, args.reps)
        H = hm.ReducedHessian(model)
        Hd = model.generate_vector(PARAMETER)
        H.mult(direction, Hd)
        out["t_action"], _ = clock(lambda: H.mult(direction, Hd), args.reps)
        cur = {"u": x[STATE].copy(), "p": x[ADJOINT].copy(), "g": g.copy(), "Hd": Hd.copy()}
        if mode == "fp64":
            ref = cur
        out.update({"err_" + k: (0.0 if mode == "fp64" else rel(cur[k], ref[k])) for k in cur})
        out["cost"] = [float(c) for c in model.cost(x)]
        say("  kernel %.4f s | assembly %.4f s | forward %.3f s (%s passes) | adjoint %.3f s (%s) | gradient %.4f s"
            " | Hessian blocks %.3f s | Hessian action %.3f s"
            % (out["t_kernel"], out["t_assembly"], out["t_forward"], out["forward_passes"], out["t_adjoint"],
               out["adjoint_passes"], out["t_gradient"], out["t_blocks"], out["t_action"]))
        say("  against fp64: state %.2e | adjoint %.2e | gradient %.2e | Hessian action %.2e"
            % (out["err_u"], out["err_p"], out["err_g"], out["err_Hd"]))

        # ---- Newton-CG to its tolerance
        if not args.no_newton:
            pde.invalidate_jacobian()
            pde.release_linearization_point()
            params = hm.ReducedSpaceNewtonCG_ParameterList()
            params["rel_tolerance"] = args.newton_tol
            params["abs_tolerance"] = 1e-12
            params["max_iter"] = args.newton_max
            params["globalization"] = "LS"
            params["GN_iter"] = 5
            params["cg_max_iter"] = args.cg_max
            params["print_level"] = 0 if args.newton_print else -1
            try:
                for _ in range(max(1, args.newton_repeats)):
                    pde.invalidate_jacobian()
                    pde.release_linearization_point()
                    solver = hm.ReducedSpaceNewtonCG(model, params)
                    calls0 = dict(pde.n_calls)
                    first = out.get("newton_first_wall")
                    wall, xs = clock(lambda: solver.solve([None, prior.mean.copy(), None]))
                    if first is None:
                        out["newton_first_wall"] = wall
            except Exception as exc:                                  # noqa: BLE001
                out["newton"] = {"failed": str(exc).splitlines()[0][:200]}
                say("  Newton-CG: FAILED -- %s" % out["newton"]["failed"])
                continue
            mmap = xs[PARAMETER].copy()
            if mode == "fp64":
                ref["m"] = mmap
            out["newton"] = {
                "wall": wall, "newton": solver.it, "cg": solver.total_cg_iter, "J": float(solver.final_cost),
                "gradnorm": float(solver.final_grad_norm), "reason": solver.termination_reasons[solver.reason],
                "converged": bool(solver.converged),
                "err_map": 0.0 if mode == "fp64" else (rel(mmap, ref["m"]) if "m" in ref else None),
                "err_truth": rel(mmap, mtrue),
                "calls": {k: pde.n_calls[k] - calls0.get(k, 0) for k in pde.n_calls}}
            r = out["newton"]
            say("  Newton-CG: %.2f s, %d Newton and %d CG iterations, J %.10e, |g| %.3e (%s); "
                "MAP against fp64 %s; %s"
                % (r["wall"], r["newton"], r["cg"], r["J"], r["gradnorm"], r["reason"],
                   "--" if r["err_map"] is None else "%.2e" % r["err_map"],
                   ", ".join("%s %d" % kv for kv in sorted(r["calls"].items()))))
    km.set_precision("fp64")

    base = rec["modes"]["fp64"]
    for mode in modes[1:]:
        o = rec["modes"][mode]
        say("%s against fp64: kernel %.2fx, assembly %.2fx, forward %.2fx, adjoint %.2fx, blocks %.2fx, "
            "action %.2fx%s"
            % (mode, base["t_kernel"] / o["t_kernel"], base["t_assembly"] / o["t_assembly"],
               base["t_forward"] / o["t_forward"], base["t_adjoint"] / o["t_adjoint"],
               base["t_blocks"] / o["t_blocks"], base["t_action"] / o["t_action"],
               "" if "newton" not in o or "failed" in o["newton"] or "newton" not in base
               else ", Newton-CG %.2fx" % (base["newton"]["wall"] / o["newton"]["wall"])))
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        say("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

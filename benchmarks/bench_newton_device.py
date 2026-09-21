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
from hippymfem.fem import assemble as asm                            # noqa: E402
from hippymfem.fem import kernel as kernel_mod                       # noqa: E402
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
    ap.add_argument("--gn-iter", type=int, default=0)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--route", default=None, help="assembly backend: csr or integrator")
    ap.add_argument("--device", default=("cuda" if hm.config.hypre_device else "cpu"), help="mfem.Device kind; 'cpu' keeps hypre on the host")
    ap.add_argument("--out", default=None)
    ap.add_argument("--symmetric-jacobian", action="store_true",
                    help="declare dR/du symmetric (it is, for this PDE): adjoint solves reuse A and its AMG")
    ap.add_argument("--release-linearization", action="store_true",
                    help="drop the linearization point at a forward solve for a new parameter "
                         "(safe for this line-search Newton-CG; needed for 128^3 on one 80 GB card)")
    ap.add_argument("--cart-part", action="store_true",
                    help="partition the box as a Cartesian grid of ranks instead of METIS on every rank")
    ap.add_argument("--newton-only", action="store_true",
                    help="skip the separately timed stages; synthetic data, then the Newton steps")
    ap.add_argument("--warm-up", action="store_true",
                    help="with --newton-only: build every kernel and pattern once before the step timer, "
                         "as the timed stages of a full run do, so the Newton steps are warm")
    args = ap.parse_args()

    hm.configure_device(args.device, COMM, quiet=(RANK != 0))
    # what MFEM actually runs on: "cuda" or "gpu" means the backend the build has (hip on AMD)
    from hippymfem.common.mfemconfig import mfem_gpu_backend
    mfem_dev = args.device if args.device in ("cpu", None) else (mfem_gpu_backend() or args.device)
    if args.route:
        asm.set_assembly_backend(args.route)
    N, ORDER = args.n, args.order
    t_start = time.perf_counter()
    serial = mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON)
    if args.cart_part:
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
    say("  [%.0f s] parallel mesh built" % (time.perf_counter() - t_start))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, %d elem/rank"
        % (N, ORDER, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(), COMM.size, pmesh.GetNE()))
    say("  kernels on %s (%s), MFEM on %s, assembly %s, parmat %s, pin %s"
        % (kernel_mod.device(), gpu_name(), mfem_dev, asm.assembly_backend(),
           os.environ.get("HIPPYMFEM_PARMAT", "auto"),
           "rank->%s" % _jaxconfig.PINNED_DEVICE if _jaxconfig.PINNED_DEVICE is not None
           else (_jaxconfig.PIN_SKIPPED or "launcher/single device")))

    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc.homogeneous(), is_fwd_linear=True,
                                   symmetric_jacobian=(True if args.symmetric_jacobian
                                                       else "auto"),
                                   release_linearization_on_move=args.release_linearization)
    # the three solves the records were taken with; the adjoint keeps its default
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
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
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)

    # ---- the pieces of a Newton-CG iteration, at the prior mean
    say("  [%.0f s] synthetic data forward solve done" % (time.perf_counter() - t_start))
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
            "(warm %.3f s) | reduced-Hessian apply %.3f s (cold %.3f s)"
            % (t_fwd, t_adj, t_grad, t_hess, t_hess_warm, t_apply, t_apply_cold))
    if not args.newton_only and any(k in str(kernel_mod.device()).lower() for k in ("cuda", "gpu")):
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
    solver = hm.ReducedSpaceNewtonCG(model, params)
    t_step, xs = timed(lambda: solver.solve([None, prior.mean.copy(), None]))
    err = (mtrue.copy().axpy(-1.0, xs[PARAMETER]).norm("l2") / max(mtrue.norm("l2"), 1e-300))
    say("  %d Newton step(s) %.2f s  newton %d  cg %d  J %.8e  |grad| %.4e  err(m) %.8f  (%s)"
        % (args.steps, t_step, solver.it, solver.total_cg_iter, solver.final_cost,
           solver.final_grad_norm, err, solver.termination_reasons[solver.reason]))
    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size, "n": N, "order": ORDER,
           "tdofs": Vu.GlobalTrueVSize(), "mdofs": Vm.GlobalTrueVSize(), "mfem_device": mfem_dev,
           "backend": asm.assembly_backend(), "symmetric_jacobian": bool(args.symmetric_jacobian),
           "release_linearization": bool(args.release_linearization), "parmat": os.environ.get("HIPPYMFEM_PARMAT", "auto"),
           "t_fwd": t_fwd, "t_adj": t_adj, "t_grad": t_grad, "t_hess_blocks": t_hess,
           "t_hess_blocks_warm": t_hess_warm,
           "t_hess_apply": t_apply, "t_hess_apply_cold": t_apply_cold, "steps": args.steps, "t_steps": t_step,
           "newton_it": solver.it, "cg_it": solver.total_cg_iter, "J": solver.final_cost,
           "gradnorm": solver.final_grad_norm, "err_m": err,
           "cart_part": bool(args.cart_part), "newton_only": bool(args.newton_only),
           "warm_up": bool(args.warm_up)}
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

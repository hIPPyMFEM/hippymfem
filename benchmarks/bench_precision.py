#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Single-precision element kernels: what they buy, and what they cost.

``_jaxconfig`` says single precision "would destroy the Newton convergence this library
depends on".  That was asserted, not measured, and the reason to measure it is that the
gap is enormous on a workstation card: 1.25 TFLOP/s fp64 against 125 TFLOP/s fp32 on an
L40S, a factor of 100 in peak throughput.

``HIPPYMFEM_PRECISION=fp32`` casts the kernel's inputs to single precision inside the
functional every derivative is built on, so the quadrature-point evaluation and the AD
run in fp32 while the derivative arrays come back float64 and hypre still gets a
double-precision matrix.  This measures, on whatever card is present:

* the kernel and the full assembly, both precisions, steady state;
* how far the assembled operator moves, as a relative matvec difference;
* how far the reduced gradient moves, which is what an optimizer actually consumes;
* and a Newton solve in each precision, which is the claim under test -- the gradient
  norm it can reach, the cost functional, and the parameter error.

Usage::

    HIPPYMFEM_DEVICE=gpu python benchmarks/bench_precision.py --n 24
    HIPPYMFEM_DEVICE=gpu mpirun -n 4 python benchmarks/bench_precision.py --n 32 \
        --out results/precision_l40s.json
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


def timed(fn, reps):
    fn()
    COMM.Barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    COMM.Barrier()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6, help="Newton iterations")
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    N, ORDER = args.n, args.order
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(
        N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    batches = hm.MeshBatches(pmesh, 2 * ORDER + 2, COMM)
    NE = pmesh.GetNE()
    say("%d^3 hex order %d: %d state dofs, %d ranks, %d elem/rank, kernels on %s (%s)"
        % (N, ORDER, Vu.GlobalTrueVSize(), COMM.size, NE, km.device(), gpu_name()))

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    K = hm.QuadratureKernel(pde_varf, Vh, batches)
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    bc0 = bc.homogeneous()
    hm.parRandom.set_seed(4)
    uv = Vu.vector(); hm.parRandom.normal(1.0, uv)
    mv = Vm.vector(); hm.parRandom.normal(0.3, mv)
    loc = [Vu.local_values(uv), Vm.local_values(mv), Vu.local_values(uv)]

    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size,
           "n": N, "order": ORDER, "NE_local": NE,
           "tdofs": Vu.GlobalTrueVSize(), "device": str(km.device())}

    # ---------------------------------------------------- kernel and assembly
    probe = Vu.vector(); hm.parRandom.set_seed(99); hm.parRandom.normal(1.0, probe)
    mat, tk, ta = {}, {}, {}
    for mode in ("fp64", "fp32"):
        old = km.set_precision(mode)
        try:
            def kernel_only():
                m = K.element_matrices(ADJOINT, STATE, loc)
                m0 = m[0]
                (float(jnp.sum(m0[0, 0])) if not isinstance(m0, np.ndarray)
                 else float(m0[0, 0, 0]))
            tk[mode] = timed(kernel_only, args.reps)

            def full():
                A = asm.assemble_matrix(Vu, Vu, batches.groups,
                                        K.element_matrices(ADJOINT, STATE, loc), NE,
                                        test_ess=bc0.ess_tdof)
                del A
            ta[mode] = timed(full, args.reps)

            A = asm.assemble_matrix(Vu, Vu, batches.groups,
                                    K.element_matrices(ADJOINT, STATE, loc), NE,
                                    test_ess=bc0.ess_tdof)
            y = Vu.vector(); y.zero()
            A.Mult(probe.hypre, y.hypre)
            mat[mode] = np.asarray(y.array).copy()
            del A
        finally:
            km.set_precision(old)
    num = COMM.allreduce(float(((mat["fp64"] - mat["fp32"]) ** 2).sum()))
    den = COMM.allreduce(float((mat["fp64"] ** 2).sum()))
    rel_mat = (num / den) ** 0.5 if den else float("nan")
    say("  kernel    fp64 %8.4f s   fp32 %8.4f s   %5.2fx"
        % (tk["fp64"], tk["fp32"], tk["fp64"] / max(tk["fp32"], 1e-12)))
    say("  assembly  fp64 %8.4f s   fp32 %8.4f s   %5.2fx"
        % (ta["fp64"], ta["fp32"], ta["fp64"] / max(ta["fp32"], 1e-12)))
    say("  assembled operator: ||A32 x - A64 x|| / ||A64 x|| = %.3e" % rel_mat)
    rec.update({"t_kernel": tk, "t_assembly": ta, "rel_matvec": rel_mat,
                "kernel_speedup": tk["fp64"] / max(tk["fp32"], 1e-12),
                "assembly_speedup": ta["fp64"] / max(ta["fp32"], 1e-12)})

    # ------------------------------------------------- gradient and a Newton run
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    # the three solves the records were taken with; the adjoint keeps its default
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector(); prior.sample_noise(1.0, noise)
    mtrue = Vm.vector(); prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, args.ntargets) for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state(); pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft(); B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30); B.perturb(data, nstd)
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)

    grads, runs = {}, {}
    for mode in ("fp64", "fp32"):
        old = km.set_precision(mode)
        try:
            x = [model.generate_vector(STATE), prior.mean.copy(),
                 model.generate_vector(ADJOINT)]
            # A failure here is a result, not an accident: the forward solve asserts
            # that one Newton step on a linear residual actually lands, and a Jacobian
            # assembled in single precision does not satisfy that.  Recording the
            # message is the whole point of running this.
            model.solveFwd(x[STATE], x)
            model.solveAdj(x[ADJOINT], x)
            g = model.generate_vector(PARAMETER)
            model.evalGradientParameter(x, g)
            grads[mode] = np.asarray(g.array).copy()   # noqa: E501

            params = hm.ReducedSpaceNewtonCG_ParameterList()
            params["rel_tolerance"] = 1e-9
            params["abs_tolerance"] = 1e-12
            params["max_iter"] = args.steps
            params["globalization"] = "LS"
            params["GN_iter"] = 0
            params["cg_max_iter"] = 30
            params["cg_coarse_tolerance"] = 1e-6
            params["print_level"] = -1
            solver = hm.ReducedSpaceNewtonCG(model, params)
            COMM.Barrier(); t0 = time.perf_counter()
            xs = solver.solve([None, prior.mean.copy(), None])
            COMM.Barrier(); wall = time.perf_counter() - t0
            err = (mtrue.copy().axpy(-1.0, xs[PARAMETER]).norm("l2")
                   / max(mtrue.norm("l2"), 1e-300))
            runs[mode] = {"wall": wall, "newton": solver.it,
                          "cg": solver.total_cg_iter, "J": solver.final_cost,
                          "gradnorm": solver.final_grad_norm, "err_m": err,
                          "reason": solver.termination_reasons[solver.reason]}
        except Exception as exc:
            msg = str(exc).splitlines()[0]
            say("  Newton %s: REFUSED -- %s" % (mode, msg[:140]))
            runs[mode] = {"failed": msg}
            grads.setdefault(mode, None)
        finally:
            km.set_precision(old)
    if grads.get("fp32") is not None and grads.get("fp64") is not None:
        gn = COMM.allreduce(float(((grads["fp64"] - grads["fp32"]) ** 2).sum()))
        gd = COMM.allreduce(float((grads["fp64"] ** 2).sum()))
        rel_g = (gn / gd) ** 0.5 if gd else float("nan")
        say("  reduced gradient: ||g32 - g64|| / ||g64|| = %.3e" % rel_g)
    else:
        rel_g = float("nan")
        say("  reduced gradient: not comparable (one precision refused to solve)")
    for mode in ("fp64", "fp32"):
        r = runs[mode]
        if "failed" in r:
            continue
        say("  Newton %s: wall %7.2f s  newton %2d  cg %4d  J %.8e  |grad| %.4e  "
            "err(m) %.6f  (%s)"
            % (mode, r["wall"], r["newton"], r["cg"], r["J"], r["gradnorm"],
               r["err_m"], r["reason"]))
    rec.update({"rel_gradient": rel_g, "newton": runs})

    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        say("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

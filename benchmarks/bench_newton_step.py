#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What a Newton step of a 3D inverse problem costs, and how much of it is the GPU's.

``bench_assembly.py`` times one assembly and ``bench_pipeline.py`` splits it into
stages.  Neither answers the question an optimizer asks: of the time a Newton step
takes, how much does moving the element kernels and the scatter to a GPU actually
remove?  That needs the whole step -- linearization point, forward and adjoint solves,
gradient, and the Hessian applications inside CG -- which is what this measures.

The step is timed twice over, deliberately:

* ``t_step``, the wall time of ``--steps`` Newton iterations, which includes the linear
  solves.  With hypre on the host those solves are a *constant* across GPUs, so they
  dilute the comparison rather than confounding it, and the dilution is reported.
* ``t_assembly``, one full assembly of the forward Jacobian in the same process, which
  is the part a faster card changes.

Both are reported so a speedup can be attributed instead of guessed at.  The cost
functional and the parameter error are printed as well: two cards must reach the same
answer, and a run that does not is a measurement of nothing.

Usage::

    HIPPYMFEM_DEVICE=gpu mpirun -n 4 python benchmarks/bench_newton_step.py --n 32
    HIPPYMFEM_DEVICE=gpu python benchmarks/bench_newton_step.py --n 24 --steps 2 \
        --out results/newton_h100.json
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
from hippymfem.fem import kernel as kernel_mod                       # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def gpu_name():
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        names = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        return names[0] if names else "none"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=32, help="cells per side (hex mesh)")
    ap.add_argument("--order", type=int, default=2, help="state polynomial degree")
    ap.add_argument("--steps", type=int, default=1, help="Newton iterations to time")
    ap.add_argument("--cg-max", type=int, default=20)
    ap.add_argument("--cg-tol", type=float, default=1e-6,
                    help="coarsest CG tolerance; small forces real solve work")
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3,
                    help="repeats for the assembly timing")
    ap.add_argument("--device", default=None,
                    help="mfem.Device kind; omit to leave MFEM on the host")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.device:
        hm.configure_device(args.device, COMM, quiet=(RANK != 0))

    N, ORDER = args.n, args.order
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(
        N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    batches = hm.MeshBatches(pmesh, 2 * ORDER + 2, COMM)
    NE = pmesh.GetNE()
    say("%d^3 hex order %d: %d state dofs, %d parameter dofs, %d ranks, "
        "%d elem/rank" % (N, ORDER, Vu.GlobalTrueVSize(), Vm.GlobalTrueVSize(),
                          COMM.size, NE))
    from hippymfem import _jaxconfig as _jc
    say("  kernels on %s (%s), MFEM on %s, assembly %s"
        % (kernel_mod.device(), gpu_name(), args.device or "host",
           asm.assembly_backend()))
    say("  GPU pinning: %s"
        % ("this rank pinned to CUDA device %s" % _jc.PINNED_DEVICE
           if _jc.PINNED_DEVICE is not None
           else "not pinned (%s)" % (_jc.PIN_SKIPPED or "single visible device")))

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
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
    targets = np.column_stack([rng.uniform(0.1, 0.9, args.ntargets)
                               for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)

    # ---- the part a faster card changes: one full assembly, steady state
    uv = Vu.vector()
    hm.parRandom.normal(1.0, uv)
    loc = [Vu.local_values(uv), Vm.local_values(mtrue), Vu.local_values(uv)]
    K = hm.QuadratureKernel(pde_varf, Vh, batches)

    def one_assembly():
        A = asm.assemble_matrix(Vu, Vu, batches.groups,
                                K.element_matrices(ADJOINT, STATE, loc), NE,
                                test_ess=bc.homogeneous().ess_tdof)
        del A

    one_assembly()
    COMM.Barrier()
    t0 = time.perf_counter()
    for _ in range(args.reps):
        one_assembly()
    COMM.Barrier()
    t_asm = (time.perf_counter() - t0) / args.reps

    # ---- the whole step
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = args.steps
    params["globalization"] = "LS"
    params["GN_iter"] = 0
    params["cg_max_iter"] = args.cg_max
    params["cg_coarse_tolerance"] = args.cg_tol
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    COMM.Barrier()
    t0 = time.perf_counter()
    x = solver.solve([None, prior.mean.copy(), None])
    COMM.Barrier()
    t_step = time.perf_counter() - t0

    err = (mtrue.copy().axpy(-1.0, x[PARAMETER]).norm("l2")
           / max(mtrue.norm("l2"), 1e-300))
    say("  assembly      %8.4f s   (%7.2f us/elem)" % (t_asm, 1e6 * t_asm / NE))
    say("  %d Newton step(s) %8.4f s   newton %d, cg %d"
        % (args.steps, t_step, solver.it, solver.total_cg_iter))
    say("  J %.8e   |grad| %.4e   err(m) %.8f   reason: %s"
        % (solver.final_cost, solver.final_grad_norm, err,
           solver.termination_reasons[solver.reason]))

    rec = {"host": platform.node(), "gpu": gpu_name(), "ranks": COMM.size,
           "n": N, "order": ORDER, "NE_local": NE,
           "tdofs": Vu.GlobalTrueVSize(), "mdofs": Vm.GlobalTrueVSize(),
           "kernel_device": str(kernel_mod.device()),
           "mfem_device": args.device or "host",
           "backend": asm.assembly_backend(),
           "t_assembly": t_asm, "us_per_elem": 1e6 * t_asm / NE,
           "steps": args.steps, "t_step": t_step,
           "newton_it": solver.it, "cg_it": solver.total_cg_iter,
           "final_cost": solver.final_cost,
           "final_grad_norm": solver.final_grad_norm,
           "err_m": err}
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        say("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

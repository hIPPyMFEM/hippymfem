#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Large-scale demonstrations: a 3D Bayesian inverse problem, scaling, and the GPU share.

Four experiments, each writing a JSON record:

``map``
    A 3D subsurface inversion -- infer a log-conductivity field from pointwise
    pressure data -- solved to the MAP point by inexact Newton-CG, with a breakdown of
    where the wall time went (assembly, forward/adjoint solves, Hessian applications).
``laplace``
    The low-rank Laplace approximation of the posterior at that point: the spectrum,
    the pointwise variance field, and the cost in Hessian applications.
``scaling``
    Strong scaling of one assembly and of one Newton iteration, from 1 rank upward.
``throughput``
    Node-level assembly throughput: many CPU ranks against a few GPU-backed ranks,
    which is the comparison that matters on a node with both.

Run the pieces separately -- they have very different resource profiles::

    mpirun -n 32 python benchmarks/bench_largescale.py --exp map --n 48 --out results/map32.json
    HIPPYMFEM_DEVICE=gpu mpirun -n 4 python benchmarks/bench_largescale.py --exp throughput
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

import hippymfem as hp                                              # noqa: E402
from hippymfem.fem import assemble as asm                           # noqa: E402
from hippymfem.fem import kernel as K                               # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


class Timers(dict):
    """Wall time accumulated per label, summed over calls."""

    def time(self, label):
        return _Timer(self, label)


class _Timer(object):
    def __init__(self, store, label):
        self.store, self.label = store, label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.store[self.label] = self.store.get(self.label, 0.0) + (
            time.perf_counter() - self.t0)
        return False


def machine():
    info = {"host": platform.node(), "ranks": NP, "cpu_count": os.cpu_count(),
            "jax_device": str(K.device()),
            "mfem": hp.mfem_config().get("version"),
            "mfem_cuda": hp.mfem_config().get("MFEM_USE_CUDA"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        info["gpus"] = sorted(set(l.strip() for l in out.stdout.splitlines()
                                  if l.strip()))
    except Exception:
        info["gpus"] = []
    return info


# ------------------------------------------------------------------ the problem
def subsurface3d(n=32, order=2, ntargets=400, seed=7, exact=False, gpu=False):
    """``-div(e^m grad u) = f`` on the unit cube, pressure observed at points."""
    if gpu:
        K.set_device("gpu")
    mesh = mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON)
    pm = mfem.ParMesh(COMM, mesh)
    Vu = hp.FunctionSpace.H1(pm, order)
    Vm = hp.FunctionSpace.H1(pm, 1)

    def varf(u, m, p, x):
        src = 1.0 + 8.0 * jnp.exp(-30.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2
                                           + (x[2] - 0.5) ** 2))
        return jnp.exp(m.val) * jnp.dot(u.grad, p.grad) - src * p.val

    bc = hp.DirichletBC(Vu, lambda z: z[2], bdr_attributes=[1, 6])
    pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if exact:
            setattr(pde, a, hp.LUSolver(COMM, max_global_size=10 ** 9))
        else:
            s = hp.KrylovSolver(COMM, "cg", "amg")
            s.parameters["rel_tolerance"] = 1e-10
            s.parameters["max_iter"] = 500
            setattr(pde, a, s)

    prior = hp.BiLaplacianPrior(Vm, gamma=0.15, delta=0.8, robin_bc=True,
                                solver_type="krylov")
    rng = np.random.default_rng(seed)
    targets = np.column_stack((rng.uniform(0.08, 0.92, ntargets),
                               rng.uniform(0.08, 0.92, ntargets),
                               rng.uniform(0.08, 0.92, ntargets)))
    B = hp.assemblePointwiseObservation(Vu, targets)
    mtrue = Vm.project(
        lambda z: 1.2 * np.sin(2.5 * z[0]) * np.cos(2.0 * z[1])
        * np.cos(1.5 * z[2]) - 0.5 * z[0])
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    std = 0.01 * max(data.norm("linf"), 1e-30)
    hp.parRandom.set_seed(seed)
    B.perturb(data, std)
    misfit = hp.DiscreteStateObservation(B, data, std ** 2)
    return pm, Vu, Vm, pde, hp.Model(pde, prior, misfit), prior, mtrue, utrue


# ---------------------------------------------------------------- experiments
def exp_map(args):
    """MAP point of a 3D inversion, with a wall-time breakdown."""
    t = Timers()
    with t.time("setup"):
        pm, Vu, Vm, pde, model, prior, mtrue, utrue = subsurface3d(
            args.n, args.order, args.ntargets, gpu=args.gpu)
    say("3D subsurface inversion")
    say("  mesh %d^3 hexes = %d elements, state %s, parameter %s"
        % (args.n, COMM.allreduce(pm.GetNE()), Vu, Vm))
    say("  %d observations, %d ranks, JAX on %s"
        % (args.ntargets, NP, K.device()))

    params = hp.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = args.tol
    params["max_iter"] = args.maxit
    params["print_level"] = 0 if RANK == 0 else -1
    params["GN_iter"] = 3
    solver = hp.ReducedSpaceNewtonCG(model, params)
    with t.time("newton"):
        x = solver.solve([None, prior.mean.copy(), None])
    err = x[PARAMETER].copy().axpy(-1.0, mtrue).norm("l2") / max(
        mtrue.norm("l2"), 1e-300)
    err0 = prior.mean.copy().axpy(-1.0, mtrue).norm("l2") / max(
        mtrue.norm("l2"), 1e-300)
    say("  %s after %d iterations, %d CG its total"
        % (solver.termination_reasons[solver.reason], solver.it,
           getattr(solver, "total_cg_iter", -1)))
    say("  ||g||/||g0|| = %.3e,  parameter error %.4f (prior mean %.4f)"
        % (solver.final_grad_norm / max(solver.initial_grad_norm, 1e-300),
           err, err0))
    say("  wall: setup %.1f s, Newton %.1f s" % (t["setup"], t["newton"]))
    say("  PDE calls: %s" % dict(pde.n_calls))

    out = {"experiment": "map", "n": args.n, "order": args.order,
           "elements": COMM.allreduce(pm.GetNE()),
           "state_dofs": Vu.GlobalTrueVSize(),
           "param_dofs": Vm.GlobalTrueVSize(),
           "ntargets": args.ntargets, "newton_its": solver.it,
           "converged": bool(solver.converged),
           "grad_reduction": solver.final_grad_norm
           / max(solver.initial_grad_norm, 1e-300),
           "param_error": err, "prior_error": err0,
           "t_setup": t["setup"], "t_newton": t["newton"],
           "pde_calls": dict(pde.n_calls)}

    if args.laplace:
        out.update(exp_laplace(model, prior, x, Vm, args))
    if args.save and RANK == 0:
        os.makedirs(args.save, exist_ok=True)
    if args.save:
        hp.write_paraview(os.path.join(args.save, "map3d"), Vm.mesh,
                          {"m_map": (Vm, x[PARAMETER]), "m_true": (Vm, mtrue),
                           "u_map": (Vu, x[STATE])})
    return out


def exp_laplace(model, prior, x, Vm, args):
    """Low-rank Laplace approximation at the MAP point."""
    k = args.nmodes
    say("\nLaplace approximation (k = %d)" % k)
    t0 = time.perf_counter()
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hp.ReducedHessian(model, misfit_only=True)
    Omega = hp.MultiVector(x[PARAMETER], k + 10)
    hp.parRandom.normal_multivector(1.0, Omega)
    d, U = hp.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)
    post = hp.GaussianLRPosterior(prior, d, U)
    post.mean = x[PARAMETER]
    wall = time.perf_counter() - t0
    say("  %d eigenvalues in [%.3e, %.3e], %.1f s"
        % (k, float(d[-1]), float(d[0]), wall))
    nabove = int(np.sum(np.asarray(d) > 1.0))
    say("  %d eigenvalues above 1 (data-informed directions)" % nabove)
    try:
        var = post.pointwise_variance(method="Randomized", r=min(k, 200))
        pv = (float(var.min()), float(var.max()))
    except Exception as exc:
        pv = None
        say("  pointwise variance unavailable: %s" % str(exc)[:80])
    return {"laplace_k": k, "eig_max": float(d[0]), "eig_min": float(d[-1]),
            "eig_above_one": nabove, "t_laplace": wall,
            "pointwise_variance_range": pv,
            "eigenvalues": [float(v) for v in np.asarray(d)]}


def exp_scaling(args):
    """Strong scaling of one assembly and one Hessian application."""
    pm, Vu, Vm, pde, model, prior, mtrue, utrue = subsurface3d(
        args.n, args.order, args.ntargets, gpu=args.gpu)
    x = model.generate_vector()
    x[PARAMETER] = prior.mean.copy()
    t0 = time.perf_counter()
    model.solveFwd(x[STATE], x)
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.solveAdj(x[ADJOINT], x)
    t_adj = time.perf_counter() - t0

    # one assembly of the Jacobian, steady state
    pde.invalidate_jacobian()
    pde._jacobian(x)
    COMM.Barrier()
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        pde.invalidate_jacobian()
        pde._jacobian(x)
    COMM.Barrier()
    t_asm = (time.perf_counter() - t0) / reps

    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    H = hp.ReducedHessian(model, misfit_only=True)
    v = model.generate_vector(PARAMETER)
    hp.parRandom.normal(1.0, v)
    Hv = model.generate_vector(PARAMETER)
    H.mult(v, Hv)
    COMM.Barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        H.mult(v, Hv)
    COMM.Barrier()
    t_hess = (time.perf_counter() - t0) / reps

    ne = COMM.allreduce(pm.GetNE())
    say("scaling at %d ranks: NE=%d, state dofs=%d" % (NP, ne, Vu.GlobalTrueVSize()))
    say("  assembly %.3f s (%.2f us/elem/rank), fwd %.2f s, adj %.2f s, "
        "Hessian apply %.3f s"
        % (t_asm, 1e6 * t_asm / max(pm.GetNE(), 1), t_fwd, t_adj, t_hess))
    return {"experiment": "scaling", "ranks": NP, "n": args.n,
            "device": "gpu" if args.gpu else "cpu",
            "order": args.order, "elements": ne,
            "state_dofs": Vu.GlobalTrueVSize(),
            "param_dofs": Vm.GlobalTrueVSize(),
            "t_assembly": t_asm, "t_forward": t_fwd, "t_adjoint": t_adj,
            "t_hessian_apply": t_hess,
            "us_per_elem": 1e6 * t_asm / max(pm.GetNE(), 1)}


def exp_throughput(args):
    """Node-level assembly throughput on whatever this rank's device is."""
    from hippymfem.fem.elementbatch import MeshBatches
    from hippymfem.fem.kernel import QuadratureKernel

    if args.gpu:
        K.set_device("gpu")
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(
        args.n, args.n, args.n, mfem.Element.HEXAHEDRON))
    Vu = hp.FunctionSpace.H1(pm, args.order)
    Vm = hp.FunctionSpace.H1(pm, 1)
    b = MeshBatches(pm, 2 * args.order + 2, COMM)
    kern = QuadratureKernel(
        lambda u, m, p, x: jnp.exp(m.val) * jnp.dot(u.grad, p.grad),
        [Vu, Vm, Vu], b)
    hp.parRandom.set_seed(1)
    uv, mv, pv = Vu.vector(), Vm.vector(), Vu.vector()
    hp.parRandom.normal(0.3, mv)
    loc = [Vu.local_values(uv), Vm.local_values(mv), Vu.local_values(pv)]
    bc = hp.DirichletBC(Vu, None, "all")
    NE = pm.GetNE()

    def kernel_only():
        mats = kern.element_matrices(ADJOINT, STATE, loc)
        # force completion without a device-to-host copy of the whole array
        float(np.asarray(mats[0][0, 0, 0]))

    def once():
        A = asm.assemble_matrix(Vu, Vu, b.groups,
                                kern.element_matrices(ADJOINT, STATE, loc), NE,
                                test_ess=bc.ess_tdof)
        del A

    def timeit(fn, reps=5):
        fn()
        COMM.Barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        COMM.Barrier()
        return (time.perf_counter() - t0) / reps

    t_kern = timeit(kernel_only)
    el = timeit(once)
    ne_tot = COMM.allreduce(NE)
    # which device each rank actually got, so a silent collapse onto device 0
    # cannot be mistaken for poor GPU scaling
    devs = sorted(set(COMM.allgather(str(K.device()))))
    say("throughput: %d ranks on %s, %d elements total, %.3f s per assembly"
        % (NP, ",".join(devs), ne_tot, el))
    say("  %.3g elements/s for the node, %.2f us/elem/rank"
        % (ne_tot / el, 1e6 * el / NE))
    say("  of which kernel %.2f us/elem (%.0f%%), host-side assembly "
        "(P^T A P, elimination, hypre) %.2f us/elem"
        % (1e6 * t_kern / NE, 100.0 * t_kern / el, 1e6 * (el - t_kern) / NE))
    return {"experiment": "throughput", "ranks": NP, "device": str(K.device()),
            "devices": devs, "on_gpu": K.on_gpu(), "n": args.n,
            "order": args.order, "elements": ne_tot, "t_assembly": el,
            "t_kernel": t_kern,
            "elements_per_sec": ne_tot / el,
            "kernel_elements_per_sec": ne_tot / t_kern,
            "us_per_elem_per_rank": 1e6 * el / NE,
            "us_per_elem_kernel": 1e6 * t_kern / NE,
            "state_dofs": Vu.GlobalTrueVSize()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="map",
                    choices=("map", "scaling", "throughput"))
    ap.add_argument("--n", type=int, default=32, help="cells per side")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=400)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxit", type=int, default=30)
    ap.add_argument("--nmodes", type=int, default=100)
    ap.add_argument("--laplace", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None, help="directory for ParaView output")
    args = ap.parse_args()

    mfem.Hypre.Init()
    info = machine()
    say("hIPPyMFEM large-scale benchmark: %s" % args.exp)
    say("  %s, %d ranks, GPUs %s, MFEM %s (CUDA %s)"
        % (info["host"], NP, info["gpus"] or "none", info["mfem"],
           info["mfem_cuda"]))
    out = {"map": exp_map, "scaling": exp_scaling,
           "throughput": exp_throughput}[args.exp](args)
    out["machine"] = info
    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)
        say("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

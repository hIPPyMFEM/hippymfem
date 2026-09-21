#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What the Hessian blocks cost: one linearization point plus N reduced-Hessian applies (E3).

hIPPyMFEM assembles the Hessian blocks (C, Wuu, Wum, Wmm) at ``setPointForHessianEvaluations``
and every apply is sparse matrix-vector products plus two solves.  This times the point and
then ``max(applies)`` applies one by one, so the cost of "one point plus N applies" can be
read for any N, and the per-apply time after the first (the cold one JIT-compiles or
sets up the solvers) is the asymptotic figure.  ``--library hippylibx`` takes the same
measurement in hIPPYlibx, in its own environment.

    HIPPYMFEM_DEVICE=cpu mpirun -n 4 tools/mpirun_pinned.sh python benchmarks/bench_apply_count.py --library hippymfem --n 64 --device cpu
    mpirun -n 4 python benchmarks/bench_apply_count.py --library hippylibx --n 64    # FEniCSx environment

The problem is that of ``bench_newton_device.py`` / ``bench_newton_hippylibx.py``: the
same mesh, PDE, prior, observations and solver settings.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", choices=["hippymfem", "hippylibx"], default="hippymfem")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--applies", default="1,5,25,100")
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    applies = sorted(int(v) for v in args.applies.split(","))
    nmax = applies[-1]
    if args.library == "hippymfem":
        rec = run_hippymfem(args, nmax)
    else:
        rec = run_hippylibx(args, nmax)
    t = np.asarray(rec["t_applies"])
    rec["applies"] = applies
    rec["totals"] = {str(N): float(rec["t_point"] + t[:N].sum()) for N in applies}
    rec["t_apply_asymptotic"] = float(np.median(t[1:])) if t.size > 1 else float(t[0])
    if rec["rank"] == 0:
        print("  %s %d^3 %d ranks: point %.3f s, first apply %.3f s, then %.3f s per apply; one point + N applies: %s"
              % (args.library, args.n, rec["ranks"], rec["t_point"], t[0], rec["t_apply_asymptotic"],
                 ", ".join("N=%d %.1f s" % (N, rec["totals"][str(N)]) for N in applies)), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(rec, f, indent=2)
    return 0


def run_hippymfem(args, nmax):
    import hippymfem as hm
    import jax.numpy as jnp
    import mfem.par as mfem
    from mpi4py import MPI
    from hippymfem.modeling.variables import STATE, PARAMETER, ADJOINT
    from hippymfem.fem import kernel as kernel_mod

    COMM = MPI.COMM_WORLD
    hm.configure_device(args.device, COMM, quiet=(COMM.rank != 0))

    def timed(fn):
        COMM.Barrier()
        t0 = time.perf_counter()
        out = fn()
        COMM.Barrier()
        return time.perf_counter() - t0, out

    N, ORDER = args.n, args.order
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(), is_fwd_linear=True)
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
    model = hm.Model(pde, prior, hm.DiscreteStateObservation(B, data, nstd ** 2))
    x = [model.generate_vector(STATE), prior.mean.copy(), model.generate_vector(ADJOINT)]
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)
    t_point, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
    H = hm.ReducedHessian(model)
    v = model.generate_vector(PARAMETER)
    hm.parRandom.normal(1.0, v)
    w = model.generate_vector(PARAMETER)
    ts = []
    for _ in range(nmax):
        t, _ = timed(lambda: H.mult(v, w))
        ts.append(t)
    return {"library": "hippymfem", "host": platform.node(), "ranks": COMM.size, "rank": COMM.rank, "n": N, "order": ORDER,
            "tdofs": int(Vu.GlobalTrueVSize()), "mdofs": int(Vm.GlobalTrueVSize()), "device": args.device,
            "kernels": str(kernel_mod.device()), "t_point": t_point, "t_applies": ts}


def run_hippylibx(args, nmax):
    import ufl
    from dolfinx import fem, mesh as dmesh
    from mpi4py import MPI
    from petsc4py import PETSc
    import hippylibx as hpx
    from benchmarks.bench_newton_hippylibx import MFEM_AMG, _krylov_forward_solves, amg_like_mfem, krylov

    COMM = MPI.COMM_WORLD
    opts = PETSc.Options()
    for k, val in MFEM_AMG.items():
        opts[k] = val
    _krylov_forward_solves()

    def timed(fn):
        COMM.Barrier()
        t0 = time.perf_counter()
        out = fn()
        COMM.Barrier()
        return time.perf_counter() - t0, out

    N, ORDER = args.n, args.order
    msh = dmesh.create_box(COMM, [np.zeros(3), np.ones(3)], [N, N, N], dmesh.CellType.hexahedron)
    Vu = fem.functionspace(msh, ("Lagrange", ORDER))
    Vm = fem.functionspace(msh, ("Lagrange", 1))
    ndof = [V.dofmap.index_map.size_global * V.dofmap.index_map_bs for V in (Vu, Vm)]

    def pde_varf(u, m, p):
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx

    u_bdr = fem.Function(Vu)
    u_bdr.interpolate(lambda x: x[2])
    u_bdr0 = fem.Function(Vu)
    u_bdr0.x.array[:] = 0.0
    facets = dmesh.locate_entities_boundary(msh, 2, lambda x: np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0))
    dofs = fem.locate_dofs_topological(Vu, 2, facets)
    pde = hpx.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, fem.dirichletbc(u_bdr, dofs), fem.dirichletbc(u_bdr0, dofs),
                                    is_fwd_linear=True)
    pde.solver = krylov()
    pde.solver_fwd_inc = krylov()
    pde.solver_adj_inc = krylov()
    prior = hpx.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    for name in ("Asolver", "Msolver", "Rsolver"):
        sol = getattr(prior, name, None)
        if sol is not None and hasattr(sol, "ksp"):
            amg_like_mfem(sol)
    noise = PETSc.Vec().create(COMM)
    prior.init_vector(noise, "noise")
    hpx.parRandom.normal(1.0, noise)
    mtrue = prior.mean.copy()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, args.ntargets) for _ in range(3)])
    B = hpx.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm(PETSc.NormType.INFINITY), 1e-30)
    hpx.parRandom.normal_perturb(nstd, data)
    model = hpx.Model(pde, prior, hpx.DiscreteStateObservation(B, data, nstd ** 2))
    x = [model.generate_vector(hpx.STATE), prior.mean.copy(), model.generate_vector(hpx.ADJOINT)]
    model.solveFwd(x[hpx.STATE], x)
    model.solveAdj(x[hpx.ADJOINT], x)
    t_point, _ = timed(lambda: model.setPointForHessianEvaluations(x, gauss_newton_approx=False))
    H = hpx.ReducedHessian(model, misfit_only=False)
    v = model.generate_vector(hpx.PARAMETER)
    hpx.parRandom.normal(1.0, v)
    w = model.generate_vector(hpx.PARAMETER)
    ts = []
    for _ in range(nmax):
        t, _ = timed(lambda: H.mult(v, w))
        ts.append(t)
    return {"library": "hippylibx", "host": platform.node(), "ranks": COMM.size, "rank": COMM.rank, "n": N, "order": ORDER,
            "tdofs": int(ndof[0]), "mdofs": int(ndof[1]), "device": "cpu", "kernels": "ufl/ffcx", "t_point": t_point, "t_applies": ts}


if __name__ == "__main__":
    raise SystemExit(main())

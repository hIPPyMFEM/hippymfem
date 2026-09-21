#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What hIPPYlibx's random stream does across MPI partitions (E5, the counterpart).

hIPPYlibx seeds a numpy generator per rank from ``SeedSequence(seed).spawn(nproc)``, so
the prior sample, the synthetic data and therefore the MAP depend on the number of
ranks.  This draws the benchmark problem's truth and data at the current rank count and
dumps the prior sample sorted by dof coordinates, the data norm and the first Newton
steps' cost; ``compare`` prints the differences between dumps.

    mpirun -n 2 python validation/partition_hippylibx.py --n 16 --out results/partition_hippylibx_n16_r2.npz
    python validation/partition_hippylibx.py compare results/partition_hippylibx_n16_r*.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare(files, out=None):
    ref = None
    rows = []
    for f in files:
        z = np.load(f)
        # sort on integer grid keys: the coordinates carry rank-dependent round-off, and
        # an exact lexsort on them scrambled equal rows (a 1.0 "difference" at 2 ranks)
        n = int(z["n"])
        g = np.rint(z["xyz"] * n * 8).astype(np.int64)
        order = np.lexsort((g[:, 0], g[:, 1], g[:, 2]))
        cur = {"xyz": z["xyz"][order], "mtrue": z["mtrue"][order], "data_norm": float(z["data_norm"]),
               "J": float(z["J"]), "ranks": int(z["ranks"])}
        if ref is None:
            ref = cur
            print("%-36s %6s %12s %12s %12s %12s" % ("file", "ranks", "xyz", "prior sample", "|data|", "J"))
            print("%-36s %6d %12s" % (os.path.basename(f), cur["ranks"], "reference"))
            continue
        same_grid = cur["xyz"].shape == ref["xyz"].shape
        dx = float(np.max(np.abs(cur["xyz"] - ref["xyz"]))) if same_grid else float("nan")
        dm = (float(np.max(np.abs(cur["mtrue"] - ref["mtrue"])) / max(np.max(np.abs(ref["mtrue"])), 1e-300))
              if same_grid else float("nan"))
        dd = abs(cur["data_norm"] - ref["data_norm"]) / max(abs(ref["data_norm"]), 1e-300)
        dj = abs(cur["J"] - ref["J"]) / max(abs(ref["J"]), 1e-300)
        print("%-36s %6d %12.2e %12.2e %12.2e %12.2e" % (os.path.basename(f), cur["ranks"], dx, dm, dd, dj))
        rows.append({"file": f, "ranks": cur["ranks"], "xyz": dx, "mtrue": dm, "data_norm": dd, "J": dj})
    if out:
        with open(out, "w") as fh:
            json.dump({"check": "partition_hippylibx", "reference": files[0], "rows": rows}, fh, indent=2)
    return 0


def run(args):
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
    N, ORDER = args.n, args.order
    msh = dmesh.create_box(COMM, [np.zeros(3), np.ones(3)], [N, N, N], dmesh.CellType.hexahedron)
    Vu = fem.functionspace(msh, ("Lagrange", ORDER))
    Vm = fem.functionspace(msh, ("Lagrange", 1))

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
    params = hpx.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = args.steps
    params["globalization"] = "LS"
    params["GN_iter"] = 0
    params["cg_max_iter"] = 25
    params["print_level"] = -1
    J = float("nan")
    if args.steps > 0:
        solver = hpx.ReducedSpaceNewtonCG(model, params)
        solver.solve([None, prior.mean.copy(), None])
        J = float(solver.final_cost)
    # collectives first (a norm inside the rank-0 print below deadlocked at 2 ranks)
    m_norm, d_norm = mtrue.norm(), data.norm()
    # the prior sample on the owned dofs, with coordinates, gathered on rank 0
    nloc = Vm.dofmap.index_map.size_local
    xyz = Vm.tabulate_dof_coordinates()[:nloc]
    vals = mtrue.getArray()[:nloc]
    X = COMM.gather(xyz, root=0)
    V = COMM.gather(vals, root=0)
    if COMM.rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, xyz=np.concatenate(X), mtrue=np.concatenate(V), data_norm=d_norm, J=J,
                 ranks=COMM.size, n=N)
        print("%d ranks: |m_true| %.10e, |data| %.10e, J after %d steps %.10e" % (COMM.size, m_norm, d_norm, args.steps, J),
              flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="run")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "compare":
        return compare(args.files, args.out)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

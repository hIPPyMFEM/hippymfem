#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""BoomerAMG settings for the single-precision incremental solves, 64^3.

The problem of ``benchmarks/bench_precision.py``.  At the MAP point (``--map-file``, from
``bench_laplace_precision.py``; a late Newton point), with mixed kernels and the Jacobian in the
single-precision hypre, the right-hand sides of the incremental solves of a few Hessian
actions are captured (directions R^{-1} xi: smooth, like the CG's).  Every setting then
gets a fresh PCG + BoomerAMG on the same matrix (``singlesolve.AMG_OPTIONS``) and solves
every captured right-hand side to ``--tol`` (1e-5, the floor of the single solves) from
zero: setup time, iterations, time per solve (median of ``--reps`` passes), and the error
against double-precision solutions to 1e-12 (2-norm, relative).  One rank.

    --settings "default;theta=0.5;coarsen=10;..."   (';'-separated, each a HIPPYMFEM_SINGLE_AMG string)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hm  # noqa: E402
from hippymfem.fem import kernel as km
from hippymfem.algorithms import singlesolve
from hippymfem.common import devicebridge as bridge
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

COMM = MPI.COMM_WORLD


def say(*a):
    print(*a, flush=True)


DEFAULT_SETTINGS = ";".join([
    "default",
    "coarsen=10",
    "theta=0.5", "theta=0.6", "theta=0.4",
    "pmax=2", "pmax=3", "pmax=6",
    "interp=14", "interp=18",
    "relax=16", "relax=16,cheby_order=3", "relax=7", "sweeps=2",
    "keep_transpose=1",
    "levels=10", "levels=6",
    "coarse_size=500", "coarse_size=5000",
    "trunc=0.1",
    "theta=0.5,pmax=3", "theta=0.5,keep_transpose=1", "coarsen=10,theta=0.5",
    "agg=1,agg_interp=5", "agg=1,agg_interp=6", "agg=1,agg_interp=7", "agg=1",
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--map-file", required=True)
    ap.add_argument("--directions", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--settings", default=DEFAULT_SETTINGS)
    ap.add_argument("--points", default="map", help="comma-separated: map (the MAP file), mean (m = 0), truth")
    ap.add_argument("--settings-other", default=None, help="settings at the points other than map (default: --settings)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hm.configure_device("cuda" if hm.config.hypre_device else "cpu", COMM, quiet=True)
    singlesolve._AMG_SETTERS.setdefault("relax_wt", ("HYPRE_BoomerAMGSetRelaxWt", float))
    N = args.n
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON))
    Vu = hm.FunctionSpace.H1(pmesh, 2)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]
    pde = hm.PDEVariationalProblem(Vh, lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad),
                                   hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6]),
                                   hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6]).homogeneous(),
                                   is_fwd_linear=True)
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, 200) for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    km.set_precision("fp64")
    pde.single_solves = False
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    misfit = hm.DiscreteStateObservation(B, data, nstd ** 2)
    model = hm.Model(pde, prior, misfit)
    rec_all = {}
    for point in [q.strip() for q in args.points.split(",") if q.strip()]:
        m = Vm.vector()
        if point == "map":
            m.array[:] = np.load(args.map_file)
        elif point == "truth":
            m.assign(mtrue)
        settings_text = args.settings if (point == "map" or not args.settings_other) else args.settings_other
        say("===== point %s" % point)

        def activate(prec, single, inc_tol):
            km.set_precision(prec)
            pde.single_solves = bool(single)
            pde.invalidate_jacobian()
            pde.release_linearization_point()
            pde._release_operators("solver", "solver_adj")
            for attr in ("solver_fwd_inc", "solver_adj_inc"):
                getattr(pde, attr).parameters["rel_tolerance"] = inc_tol

        # the right-hand sides of the incremental solves of a few Hessian actions, captured
        rhs = []

        def capture(solver, label):
            f = solver.solve

            def g(x, b):
                rhs.append((label, b.copy()))
                return f(x, b)
            solver.solve = g
            return f

        activate("mixed", True, 1e-6)
        x = [model.generate_vector(STATE), m.copy(), model.generate_vector(ADJOINT)]
        model.solveFwd(x[STATE], x)
        model.solveAdj(x[ADJOINT], x)
        model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        H = hm.ReducedHessian(model)
        S = pde.solver_fwd_inc.A
        say("matrix %s, %d rows, %d nonzeros; settings before the sweep: %s"
            % (type(S).__name__, S.GetGlobalNumRows(), S.NNZ(), singlesolve.AMG_OPTIONS))
        f_fwd = capture(pde.solver_fwd_inc, "fwd")
        f_adj = capture(pde.solver_adj_inc, "adj")
        hm.parRandom.set_seed(5)
        for i in range(args.directions):
            xi, d, Hd = Vm.vector(), Vm.vector(), Vm.vector()
            hm.parRandom.normal(1.0, xi)
            prior.Rsolver.solve(d, xi)
            H.mult(d, Hd)
        pde.solver_fwd_inc.solve, pde.solver_adj_inc.solve = f_fwd, f_adj
        say("captured %d right-hand sides (%s)" % (len(rhs), ", ".join(l for l, _ in rhs)))

        # references: double precision throughout, 1e-12
        activate("fp64", False, 1e-12)
        xr = [model.generate_vector(STATE), m.copy(), model.generate_vector(ADJOINT)]
        model.solveFwd(xr[STATE], xr)
        model.solveAdj(xr[ADJOINT], xr)
        model.setPointForHessianEvaluations(xr, gauss_newton_approx=False)
        refs = []
        for label, b in rhs:
            y = Vu.vector()
            s = pde.solver_fwd_inc if label == "fwd" else pde.solver_adj_inc
            its = s.solve(y, b)
            refs.append(y)
        say("double-precision references: done")
        activate("mixed", True, 1e-6)
        x = [model.generate_vector(STATE), m.copy(), model.generate_vector(ADJOINT)]
        model.solveFwd(x[STATE], x)
        model.solveAdj(x[ADJOINT], x)
        model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        S = pde.solver_fwd_inc.A
        params = pde.solver_fwd_inc.parameters

        rec = rec_all[point] = {"n": N, "tol": args.tol, "nrhs": len(rhs), "rows": int(S.GetGlobalNumRows()),
                                "nnz": int(S.NNZ()), "settings": {}}
        y = Vu.vector()
        for text in [t.strip() for t in settings_text.split(";") if t.strip()]:
            opts = {} if text == "default" else singlesolve.parse_amg_options(text)
            singlesolve.AMG_OPTIONS = opts
            out = rec["settings"][text] = {}
            try:
                bridge.synchronize()
                t0 = time.perf_counter()
                eng = singlesolve.SingleEngine(S, params)
                bridge.synchronize()
                t_setup = time.perf_counter() - t0
                # a first solve (the engine's buffers), untimed
                eng.solve(y.hypre, rhs[0][1].hypre, args.tol, 0.0, 1000)
                walls, its, errs, finals = [], [], [], []
                for rep in range(args.reps):
                    bridge.synchronize()
                    t0 = time.perf_counter()
                    its_rep = []
                    for (label, b), yr in zip(rhs, refs):
                        k, fin = eng.solve(y.hypre, b.hypre, args.tol, 0.0, 1000)
                        its_rep.append(k)
                        if rep == 0:
                            finals.append(fin)
                            errs.append(y.copy().axpy(-1.0, yr).norm("l2") / yr.norm("l2"))
                    bridge.synchronize()
                    walls.append(time.perf_counter() - t0)
                    its = its_rep
                # setup again, timed: the first setup of a process pays for allocations
                eng.destroy()
                bridge.synchronize()
                t0 = time.perf_counter()
                eng = singlesolve.SingleEngine(S, params)
                bridge.synchronize()
                t_setup2 = time.perf_counter() - t0
                eng.destroy()
                del eng
                nf = sum(1 for l, _ in rhs if l == "fwd")
                out.update({"t_setup": t_setup, "t_setup2": t_setup2, "walls": walls,
                            "t_solve": float(np.median(walls)) / len(rhs), "its": its,
                            "its_mean": float(np.mean(its)), "its_fwd": float(np.mean(its[:nf] if nf else [0])),
                            "err_max": float(max(errs)), "err_mean": float(np.mean(errs)),
                            "final_max": float(max(finals))})
                say("%-34s setup %.3f / %.3f s | its %5.1f (%s) | %.1f ms per solve, %.2f ms per it | error vs double "
                    "mean %.1e max %.1e" % (text, t_setup, t_setup2, out["its_mean"], " ".join(str(v) for v in its),
                                            1e3 * out["t_solve"], 1e3 * out["t_solve"] / max(out["its_mean"], 1e-9),
                                            out["err_mean"], out["err_max"]))
            except Exception as exc:                                       # noqa: BLE001
                out["failed"] = str(exc).splitlines()[0][:200]
                say("%-34s FAILED: %s" % (text, out["failed"]))
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(rec_all, f, indent=2)
    singlesolve.AMG_OPTIONS = {}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

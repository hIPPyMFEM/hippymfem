#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The nonlinear Stokes solve of the Antarctic ice sheet at any size, on any number of
ranks: the forward problem from rest (a continuation in the strain-rate floor, then
Newton), then one adjoint solve and one Gauss-Newton Hessian action, with timings.

    mpirun -n 16 tools/mpirun_pinned.sh python scale.py --res 2 --nz 4 --out x.json
    # a solve longer than a job: the same command again continues it
    mpirun -n 64 tools/mpirun_pinned.sh python scale.py --res 1 --nz 4 --checkpoint ck --time-limit 540 --out x.json
    # more levels: from the solution with fewer, interpolated along the columns (a few Newton steps)
    mpirun -n 64 tools/mpirun_pinned.sh python scale.py --res 1 --nz 4 --no-derivatives --save-state s4
    mpirun -n 64 tools/mpirun_pinned.sh python scale.py --res 1 --nz 8 --start-from s4 --out y.json

The friction is the one built from the driving stress and the observed speed (the truth
of ``run.py``'s synthetic problem).  ``--solver dd``, on any number of ranks, one GPU each:
FGMRES with every rank's velocity block factorized by cuDSS (the mesh is cut into whole
columns); ``--solver columns``: FGMRES without any matrix (``matrixfree.py``), for meshes
past the memory of those factorizations; ``--solver cudss``: one direct solve, on one rank.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def column_interpolate(Xc, Vc, Xf, order):
    """Fields ``Vc`` (``(k, n)``) at the nodes ``Xc`` of a mesh of columns, at the nodes
    ``Xf`` of a mesh with the same columns and other levels: along each column, the
    piecewise polynomial of degree ``order`` through the nodes.  Both sets are those of
    one rank, which holds the same columns on either mesh."""
    def columns(X):
        ix = np.rint(X[:, 0] * 1e4).astype(np.int64)
        iy = np.rint(X[:, 1] * 1e4).astype(np.int64)
        key = (ix - ix.min()) * (int(iy.max() - iy.min()) + 1) + (iy - iy.min())
        o = np.lexsort((X[:, 2], iy, ix))
        ncol = np.unique(key).size
        if X.shape[0] % max(ncol, 1):
            raise RuntimeError("the nodes do not form whole columns")
        return o, ncol, X.shape[0] // max(ncol, 1), np.stack([ix[o], iy[o]], axis=1)

    if Xf.shape[0] == 0:
        return np.zeros((Vc.shape[0], 0))
    oc, nc, Lc, kc = columns(Xc)
    of, nf, Lf, kf = columns(Xf)
    if nc != nf or not np.array_equal(kc[::Lc], kf[::Lf]):
        raise RuntimeError("the two meshes do not have the same columns on this rank (%d and %d)" % (nc, nf))
    zc, zf = Xc[oc, 2].reshape(nc, Lc), Xf[of, 2].reshape(nf, Lf)
    # heights as fractions of the column, which are the same in every column
    span = np.maximum(zc[:, -1:] - zc[:, :1], 1e-300)
    tc, tf = ((zc - zc[:, :1]) / span).mean(axis=0), ((zf - zf[:, :1]) / span).mean(axis=0)
    W = np.zeros((Lf, Lc))
    ends = tc[::order]                                            # the levels between the elements
    for l, t in enumerate(tf):
        e = int(np.clip(np.searchsorted(ends, t, side="right") - 1, 0, ends.size - 2))
        nodes = np.arange(order * e, order * e + order + 1)
        for a_ in nodes:                                          # Lagrange's polynomials on the element's nodes
            w = 1.0
            for b_ in nodes:
                if b_ != a_:
                    w *= (t - tc[b_]) / (tc[a_] - tc[b_])
            W[l, a_] = w
    out = np.empty((Vc.shape[0], Xf.shape[0]))
    for k in range(Vc.shape[0]):
        out[k, of] = (Vc[k, oc].reshape(nc, Lc) @ W.T).ravel()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))
    p.add_argument("--res", type=int, default=4, help="grid cell = 5 km x res")
    p.add_argument("--nz", type=int, default=4, help="hexahedra per column")
    p.add_argument("--solver", choices=("dd", "columns", "cudss"), default="dd",
                   help="dd: every rank's velocity block factorized; columns: no matrix at all (matrixfree.py), "
                        "for meshes past the memory of the factorizations; cudss: one direct solve, one rank")
    p.add_argument("--factors", choices=("double", "single"), default="double",
                   help="cudss: the precision of the factors (single halves their memory; the solutions are refined "
                        "against the double-precision matrix either way)")
    p.add_argument("--forcing", action="store_true",
                   help="columns: the tolerance of each Newton step's linear solve between 1e-2 and --rtol, as the "
                        "residual's fall allows")
    p.add_argument("--restart", type=int, default=40, help="columns: Krylov vectors before a restart")
    p.add_argument("--sweeps", type=int, default=1,
                   help="columns: Chebyshev steps of the solves on the columns before and after the collapsed problem "
                        "(more where the cells are no wider than the ice is thick)")
    p.add_argument("--smooth-range", type=float, default=30.0,
                   help="columns: the Chebyshev steps take down the spectrum from its top to 1 / this of it")
    p.add_argument("--coarse-maxit", type=int, default=20, help="columns: iterations of a solve of the collapsed problem at most")
    p.add_argument("--coarse-rtol", type=float, default=1e-2, help="columns: tolerance of a solve of the collapsed problem")
    p.add_argument("--coarse", choices=("ilu", "direct"), default="ilu",
                   help="columns: the collapsed problem by conjugate gradients with an incomplete factorization on "
                        "the host, or (one rank) by a factorization on the card")
    p.add_argument("--schur-every", type=int, default=1,
                   help="columns: the patches of the Schur complement made anew at every this many Newton steps")
    p.add_argument("--good-step", type=float, default=1.0,
                   help="a whole Newton step is taken without trying shorter ones if it leaves this much of the residual "
                        "or less (1: backtracking as usual)")
    p.add_argument("--refine", type=int, default=0,
                   help="uniform refinements of the mesh of columns (each halves the cells and the layers)")
    p.add_argument("--rtol", type=float, default=1e-6, help="Krylov tolerance of every linear solve (dd)")
    p.add_argument("--no-derivatives", action="store_true", help="the forward solve only")
    p.add_argument("--gradient-only", action="store_true", help="after the forward solve the gradient, not the Hessian action")
    p.add_argument("--steps", type=int, default=0,
                   help="time only this many Newton steps from rest (sizes where the whole solve does not fit a "
                        "job): the Jacobian, the factorizations, the solve")
    p.add_argument("--checkpoint", default=None,
                   help="a folder: the forward solve saves its iterate there after every Newton step and continues "
                        "from the one that is there (a solve longer than a job)")
    p.add_argument("--save-state", default=None,
                   help="a folder: the solution, with the positions of its nodes, one file per rank")
    p.add_argument("--start-from", default=None,
                   help="the --save-state folder of a run on the same grid and number of ranks with other levels: "
                        "the forward solve starts from that solution, interpolated along every column")
    p.add_argument("--time-limit", type=float, default=0.0,
                   help="with --checkpoint: stop after the first Newton step that ends this many seconds into the run; "
                        "the derivatives are not started after it either")
    p.add_argument("--time-budget", type=float, default=0.0,
                   help="with --checkpoint: a Newton step is not started if, judged by the last one, it would end "
                        "later than this many seconds into the run")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    from mpi4py import MPI                  # before hippymfem: then it does not pin one GPU per process
    import hippymfem as hm
    import antarctica as an
    import stokes as st
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    comm = MPI.COMM_WORLD
    rank0 = comm.rank == 0
    T0 = time.time()

    def say(*x):
        if rank0:
            print("[%7.1f s] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)

    def rank_file(folder):
        return os.path.join(folder, "state_r%04d_of%04d.npz" % (comm.rank, comm.size))

    carry = {}

    def through_refinement(mesh, step):
        """The solution to start from, put on the mesh of the columns before it is refined
        (interpolated along the columns there) and carried through every refinement by
        MFEM; the positions of the nodes go along, to check the numbering afterwards."""
        import mfem.par as mfem

        if step == 0:
            z = np.load(rank_file(a.start_from))
            carry["nz"] = int(z["nz"])
            fec2, fec1 = mfem.H1_FECollection(2, 3), mfem.H1_FECollection(1, 3)
            fu = mfem.ParFiniteElementSpace(mesh, fec2, 3, mfem.Ordering.byNODES)
            fp = mfem.ParFiniteElementSpace(mesh, fec1)
            X2, X1 = hm.FunctionSpace.H1(mesh, 2).coordinates(), hm.FunctionSpace.H1(mesh, 1).coordinates()
            fields = {"u": (fu, column_interpolate(z["X2"], z["u"], X2, 2).ravel()), "x2": (fu, X2.T.ravel()),
                      "p": (fp, column_interpolate(z["X1"], z["p"][None], X1, 1)[0]), "x1": (fp, X1[:, 2].copy())}
            for k, (fes, vals) in fields.items():
                gf = mfem.ParGridFunction(fes)
                gf.SetFromTrueDofs(mfem.Vector(np.ascontiguousarray(vals)))
                carry[k] = gf
            carry["keep"] = (fec2, fec1, fu, fp)
        else:
            carry["keep"][2].Update()
            carry["keep"][3].Update()
            for k in ("u", "x2", "p", "x1"):
                carry[k].Update()

    # a solution saved on the mesh of the columns goes through the refinements; one saved
    # on a mesh refined as this one is only interpolated along the columns
    saved_refine = None
    # a later job of a checkpointed solve continues from its checkpoint: the solution to
    # start from is not read again
    resumed = comm.bcast(bool(a.start_from and a.checkpoint and os.path.exists(os.path.join(a.checkpoint, "forward.json")))
                         if rank0 else None)
    if a.start_from and not resumed:
        with np.load(rank_file(a.start_from)) as z:
            saved_refine = comm.bcast(int(z["refine"]) if "refine" in z.files else 0 if rank0 else None)
        if saved_refine not in (0, a.refine):
            raise SystemExit("the solution in %s is on a mesh refined %d times, this one %d" % (a.start_from, saved_refine, a.refine))
    carried = bool(a.start_from) and not resumed and a.refine > 0 and saved_refine == 0
    t0 = time.time()
    grid = an.ModelGrid.load(a.res)
    cols = an.Columns(grid, a.nz)
    pb = st.IceSheet(cols, comm, refine=a.refine, solver=None, verbose=rank0,
                     partition="columns" if a.solver in ("dd", "columns") else "metis",
                     on_mesh=through_refinement if carried else None)
    if a.solver == "cudss":
        import gpu_direct
        pb.solver = gpu_direct.CuDSSDirect(comm, precision=a.factors)
    elif a.solver == "columns":
        import matrixfree
        pb.solver = matrixfree.ColumnStokes(pb, rtol=a.rtol, forcing=a.forcing, restart=a.restart, coarse_maxit=a.coarse_maxit,
                                            coarse_rtol=a.coarse_rtol, schur_every=a.schur_every, sweeps=a.sweeps,
                                            smooth_range=a.smooth_range, coarse=a.coarse)
    else:
        pb.solver = st.BlockFGMRES(pb, rtol=a.rtol)
    pb.good_step = a.good_step
    t_setup = time.time() - t0
    sz = pb.summary()
    unknowns = sz["velocity_dofs"] + sz["pressure_dofs"]
    say("icesheet scale: %g km grid, %d levels: %d elements, %d unknowns (%d velocity, %d pressure), "
        "%d parameters, %d ranks; setup %.1f s"
        % (grid.h / 2 ** a.refine, a.nz * 2 ** a.refine, sz["elements"], unknowns, sz["velocity_dofs"],
           sz["pressure_dofs"], sz["parameters"], comm.size, t_setup))
    m = pb.Vm.vector()
    xy = pb.Vm.coordinates()
    # the friction at the nodes of the grid, between them (a refined mesh) interpolated in its log
    logf = np.log(grid.driving_stress_friction())
    fx, fy = (xy[:, 0] - grid.xn[0]) / grid.h, (xy[:, 1] - grid.yn[0]) / grid.h
    I, J = np.floor(fx + 1e-9).astype(int), np.floor(fy + 1e-9).astype(int)
    tx, ty = np.clip(fx - I, 0.0, 1.0), np.clip(fy - J, 0.0, 1.0)
    I1, J1 = I + (tx > 1e-9), J + (ty > 1e-9)
    tx, ty = np.where(tx > 1e-9, tx, 0.0), np.where(ty > 1e-9, ty, 0.0)
    m.array[:] = ((1 - tx) * (1 - ty) * logf[J, I] + tx * (1 - ty) * logf[J, I1]
                  + (1 - tx) * ty * logf[J1, I] + tx * ty * logf[J1, I1])
    if not np.all(np.isfinite(m.array)):
        raise RuntimeError("the friction is not defined at a node of the footprint")
    x = [pb.generate_state(), m, pb.generate_state()]
    if a.steps:
        m3 = pb.to_bed(m)
        u, p_ = pb.Vv.vector(), pb.Vs.vector()
        rows = []
        for k in range(a.steps):
            t0 = time.time()
            ru, rp = pb.residual(u, p_, m3, 1e-2)
            t1 = time.time()
            J = pb.jacobian(u, p_, m3, 1e-2)
            t2 = time.time()
            pb._factorize(J)
            t3 = time.time()
            rhs = pb.generate_state(); pb.join(ru, rp, rhs); rhs.scale(-1.0)
            d = pb.generate_state()
            pb._solve(d, rhs)
            t4 = time.time()
            du, dp = pb.split(d)
            u.axpy(1.0, du); p_.axpy(1.0, dp)
            rn = np.sqrt(ru.inner(ru) + rp.inner(rp))
            rows.append(dict(residual_s=t1 - t0, assemble_s=t2 - t1, setup_s=t3 - t2, solve_s=t4 - t3, residual=float(rn)))
            say("step %d: |r| %.3e; residual %.1f s, Jacobian %.1f s, %s %.1f s, solve %.1f s"
                % (k, rn, t1 - t0, t2 - t1, "factorization" if a.solver == "cudss" else "block factorizations", t3 - t2, t4 - t3))
        rec = dict(args=vars(a), sizes=sz, unknowns=unknowns, grid_km=grid.h, setup_s=t_setup, steps=rows,
                   solver=pb.solver.describe() if hasattr(pb.solver, "describe") else None)
        if a.out and rank0:
            with open(a.out, "w") as fh:
                json.dump(rec, fh, indent=1)
        say("done in %.1f s" % (time.time() - T0))
        return 0
    def write(rec):
        if a.out and rank0:
            with open(a.out + ".tmp", "w") as fh:
                json.dump(rec, fh, indent=1)
            os.replace(a.out + ".tmp", a.out)

    def nodes():
        # the nodes of the scalar Q2 space, which number each velocity component, and of Q1
        return hm.FunctionSpace.H1(pb.mesh, pb.order).coordinates(), pb.Vs.coordinates()

    if resumed:
        pb.start_from([pb.generate_state(), m, None])       # a warm solve: its iterate is in the checkpoint
        say("forward: continuing the solve started from %s" % a.start_from)
    elif a.start_from:
        import mfem.par as mfem

        X2, X1 = nodes()
        s0 = pb.generate_state()
        u0, p0 = pb.Vv.vector(), pb.Vs.vector()
        if carried:
            got = {}
            for k, n in (("u", pb.nv), ("x2", pb.nv), ("p", pb.np), ("x1", pb.np)):
                tv = mfem.Vector(n)
                carry[k].GetTrueDofs(tv)
                got[k] = np.array(tv.GetDataArray())
            off = max(float(np.abs(got["x2"].reshape(3, -1).T - X2).max()) if X2.size else 0.0,
                      float(np.abs(got["x1"] - X1[:, 2]).max()) if X1.size else 0.0)
            if comm.allreduce(off, op=MPI.MAX) > 1e-8:
                raise RuntimeError("the refined spaces are not numbered as the problem's (%.1e km apart)" % off)
            u0.array[:], p0.array[:] = got["u"], got["p"]
            nz0 = carry["nz"]
            carry.clear()
        else:
            z = np.load(rank_file(a.start_from))
            u0.array.reshape(3, -1)[:] = column_interpolate(z["X2"], z["u"], X2, pb.order)
            p0.array[:] = column_interpolate(z["X1"], z["p"][None], X1, 1)[0]
            nz0 = int(z["nz"]) * 2 ** saved_refine
            del z
        pb.join(u0, p0, s0)
        pb.start_from([s0, m, None])
        say("forward: starting from the solution in %s (%d levels there), interpolated along the columns%s"
            % (a.start_from, comm.bcast(nz0 if rank0 else None), " and through the refinement" if carried else ""))
        del X2, X1, u0, p0, s0
    deadline = T0 + a.time_limit if a.time_limit else None
    if a.checkpoint:
        pb.set_checkpoint(a.checkpoint, deadline, warm=bool(a.start_from),
                          budget_end=T0 + a.time_budget if a.time_budget else None)
    t0 = time.time()
    try:
        pb.solveFwd(x[STATE], x)
    except st.ForwardStopped as e:
        say("forward: %s (%d linear solves so far, %.1f s assembling, %.1f s in solver setup, %.1f s solving); "
            "run again to continue" % (e, pb.counters["solves"], pb.counters["assemble_s"], pb.counters["factor_s"],
                                       pb.counters["solve_s"]))
        return 0
    t_fwd = pb.last_forward["seconds"] if a.checkpoint else time.time() - t0
    flux = pb.mass_flux(x[STATE])
    rec = dict(args=vars(a), sizes=sz, unknowns=unknowns, grid_km=grid.h, setup_s=t_setup, forward_s=t_fwd,
               forward=pb.last_forward, counters=dict(pb.counters), mass_flux_gt_a=flux)
    # what an earlier job of the same run already measured (the derivatives, one at a time)
    prev = comm.bcast(json.load(open(a.out)) if rank0 and a.out and a.checkpoint and os.path.exists(a.out) else {})
    for k in ("linearization_s", "gradient_s", "gradient_norm", "gradient_krylov", "hessian_action_s",
              "hessian_krylov", "hessian_inner"):
        if k in prev:
            rec[k] = prev[k]
    if hasattr(pb.solver, "describe"):
        rec["forward_solver"] = pb.solver.describe()
    if a.save_state:
        if rank0:
            os.makedirs(a.save_state, exist_ok=True)
        comm.Barrier()
        us, ps = pb.split(x[STATE])
        X2, X1 = nodes()
        np.savez(rank_file(a.save_state), X2=X2, X1=X1, u=np.asarray(us.array).reshape(3, -1), p=np.asarray(ps.array),
                 nz=a.nz, res=a.res, refine=a.refine, unknowns=unknowns)
        comm.Barrier()
        say("forward: the solution and its nodes saved in %s" % a.save_state)
        del us, ps, X2, X1
    u = pb.velocity(x[STATE])
    sp = comm.allreduce(float(np.max(np.sqrt((u.array.reshape(3, -1) ** 2).sum(axis=0))) if u.local_size else 0.0), op=MPI.MAX)
    # where the fastest ice is (the velocity's true dofs are ordered by component, each
    # block in the numbering of the scalar Q2 space)
    import mfem.par as mfem
    if pb.Vv.fes.GetOrdering() == mfem.Ordering.byNODES:
        X = hm.FunctionSpace.H1(pb.mesh, pb.order).coordinates()
        speed = np.sqrt((u.array.reshape(3, -1) ** 2).sum(axis=0))
        top = np.argsort(speed)[::-1][:5]
        mine = [dict(speed_m_a=1e3 * float(speed[i]), x=float(X[i, 0]), y=float(X[i, 1]), z=float(X[i, 2]),
                     u=[1e3 * float(c) for c in u.array.reshape(3, -1)[:, i]]) for i in top]
        allv = sorted(sum(comm.allgather(mine), []), key=lambda d: -d["speed_m_a"])[:8]
        for d in allv:
            Jn = int(np.rint((d["y"] - grid.yn[0]) / grid.h)); In = int(np.rint((d["x"] - grid.xn[0]) / grid.h))
            d.update(surface_km=float(grid.s[Jn, In]), bed_km=float(grid.b[Jn, In]))
        say("fastest ice: " + "; ".join("%.0f m/a at (%.0f, %.0f, %.3f) km, s %.3f b %.3f, u %s"
                                       % (d["speed_m_a"], d["x"], d["y"], d["z"], d["surface_km"], d["bed_km"],
                                          ", ".join("%.0f" % c for c in d["u"])) for d in allv[:4]))
        rec["fastest"] = allv
    say("forward: %.1f s, %d Newton steps in %d stages (%d linear solves, %.1f s assembling, %.1f s in solver setup, "
        "%.1f s solving); fastest ice %.0f m/a; mass flux %.0f Gt/a; residual %.1e of the loads"
        % (t_fwd, pb.last_forward["newton"], pb.last_forward["stages"], pb.counters["solves"], pb.counters["assemble_s"],
           pb.counters["factor_s"], pb.counters["solve_s"], 1e3 * sp, flux,
           pb.last_forward["residual_norm"] / pb.last_forward["loads_norm"]))
    write(rec)

    def late():
        # rank 0's clock: every rank takes the same branch
        return comm.bcast(deadline is not None and time.time() > deadline if comm.rank == 0 else None, root=0)

    def krylov(n):
        return [int(i) for i in getattr(pb.solver, "iterations", [])[-n:]]

    if not a.no_derivatives and not ("gradient_s" in rec and ("hessian_action_s" in rec or a.gradient_only)) and not late():
        # the Jacobian at the solution and its factorization (or the preconditioner's)
        if a.time_budget and hasattr(pb.solver, "deadline"):
            pb.solver.deadline = T0 + a.time_budget - 30.0          # a solve returns what it has by then
        t0 = time.time()
        pb._at(x)
        rec["linearization_s"] = time.time() - t0
        say("derivatives: the Jacobian at the solution and its %s, %.1f s"
            % ("factorization" if a.solver == "cudss" else "preconditioner", rec["linearization_s"]))
        if "gradient_s" not in rec:
            # a gradient: one adjoint solve with a misfit-like source (the surface velocity
            # against 0.9 of itself), then C^T lambda
            rhs = pb.generate_state()
            rhs.array[:pb.nv] = 0.1 * u.array
            t0 = time.time()
            pb.solveAdj(x[ADJOINT], x, rhs)
            g = pb.generate_parameter()
            pb.evalGradientParameter(x, g)
            rec.update(gradient_s=time.time() - t0, gradient_norm=float(g.norm()), gradient_krylov=krylov(1),
                       gradient_complete=not getattr(pb.solver, "timed_out", False))
            say("derivatives: gradient (adjoint solve%s) %.1f s"
                % (", %d iterations" % rec["gradient_krylov"][0] if rec["gradient_krylov"] else "", rec["gradient_s"]))
            write(rec)
        if "hessian_action_s" not in rec and not late() and not a.gradient_only:
            # one Gauss-Newton Hessian action: C dm, two incremental solves, C^T
            t0 = time.time()
            pb.setLinearizationPoint(x, True)
            dm = pb.generate_parameter()
            dm.array[:] = np.sin(xy[:, 0] / 300.0) * np.cos(xy[:, 1] / 300.0)
            cu = pb.generate_state()
            pb.apply_ij(ADJOINT, PARAMETER, dm, cu)
            cu.scale(-1.0)
            uh = pb.generate_state()
            pb.solveIncremental(uh, cu, False)
            lam = pb.generate_state()
            uh.scale(-1.0)
            pb.solveIncremental(lam, uh, True)
            hdm = pb.generate_parameter()
            pb.apply_ij(PARAMETER, ADJOINT, lam, hdm)
            rec.update(hessian_action_s=time.time() - t0, hessian_krylov=krylov(2), hessian_inner=float(hdm.inner(dm)))
            say("derivatives: Gauss-Newton Hessian action (two incremental solves%s) %.1f s"
                % (", %s iterations" % " and ".join(str(i) for i in rec["hessian_krylov"]) if rec["hessian_krylov"] else "",
                   rec["hessian_action_s"]))
        if hasattr(pb.solver, "describe"):
            rec["solver"] = pb.solver.describe()
    rec["seconds"] = time.time() - T0
    write(rec)
    say("done in %.1f s" % rec["seconds"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        # one rank's error must end the run: the others would wait for it until the job's limit
        import traceback

        from mpi4py import MPI

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise

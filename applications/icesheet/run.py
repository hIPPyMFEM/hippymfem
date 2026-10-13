#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The Antarctic ice sheet inverse problem of Isaac, Petra, Stadler and Ghattas (J.
Comput. Phys. 296, 2015), end to end, with hIPPyMFEM.

Two problems (``--data``):

``synthetic``  a test with a known answer: the data are the model's own surface velocity
               at a friction built from the driving stress and the observed speed, with
               noise of 10 % of the speed, at the surface nodes with an observed velocity;
``observed``   the paper's: the observed velocities (ITS_LIVE, and MEaSUREs around the
               pole), its misfit (an integral over the surface) and its prior.

Stages (``--stages``), each saving what the next needs in ``--out``; a run started again
resumes where it stopped:

``data``    the observations;
``verify``  the gradient and the linearized state against finite differences, and the
            symmetry of the Gauss-Newton Hessian;
``map``     the MAP point by inexact Gauss-Newton-CG (``hm.ReducedSpaceNewtonCG``);
``eig``     the eigenvalues of the misfit Hessian against the prior at the MAP point, a
            block of products at a time until they have gone below one;
``post``    the Laplace approximation: pointwise variance, prior and posterior samples;
``qoi``     the paper's predictions: the ice mass flux into the ocean through all the
            margins, East Antarctica's and the Totten Glacier's, with the linearized
            posterior standard deviation, and with ``--qoi-samples`` over posterior
            samples pushed through the nonlinear model.
"""
import argparse
import glob
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

#: what depends on the problem, unless given: (synthetic, observed)
DEFAULTS = dict(init=("mean", "fit:6"), rtol=(1e-5, 1e-3), max_newton=(40, 80), cg_max=(200, 300),
                friction_cap=(0.0, 1e6))
#: the speed added to the observed one in the noise's standard deviation, m/a: 1 m/a in the
#: synthetic problem; the paper's epsilon = 1e-9 km^2/a^2 for the observed velocities
EPS_OBS = dict(synthetic=1.0, observed=1e3 * np.sqrt(1e-9))
FIRST, OVERSAMPLE = 70, 20                  # the low rank's first block and its extra columns


def parse(argv=None):
    p = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))
    g = p.add_argument_group("mesh")
    g.add_argument("--res", type=int, default=8, help="grid cell = 5 km x res (8: 40 km)")
    g.add_argument("--nz", type=int, default=3, help="hexahedra per column")
    g.add_argument("--friction-order", type=int, choices=(1, 2), default=1,
                   help="the friction on the bed: bilinear on the grid's cells, or (2) biquadratic, the trace of the "
                        "velocity's space and the paper's choice: four times the parameters on the same mesh")
    g = p.add_argument_group("data")
    g.add_argument("--data", choices=("synthetic", "observed"), default="synthetic")
    g.add_argument("--noise", type=float, default=0.1, help="noise std as a fraction of the speed")
    g.add_argument("--obs-spacing", type=float, default=80.0,
                   help="the distance (km) at which observations are independent; the noise is --noise of the speed "
                        "per obs-spacing^2 of surface.  1 is the paper's statement taken literally, and gives "
                        "eigenvalues a few thousand times those of its Figure 5; 80 puts them at its level")
    g.add_argument("--misfit", choices=("integral", "nodes"), default="integral",
                   help="synthetic: integral is the misfit of the observed problem (the paper's integral over the "
                        "surface, noise per obs-spacing^2), so what the data say does not grow with the mesh; nodes "
                        "is one independent observation at every node of the grid, with which it does (four times "
                        "the data and twice to four times the eigenvalues above one at every halving of the grid)")
    g.add_argument("--seed", type=int, default=1, help="synthetic: of the noise")
    g = p.add_argument_group("prior")
    g.add_argument("--prior-std", type=float, default=1.5, help="synthetic: pointwise std of beta")
    g.add_argument("--prior-corr", type=float, default=150.0, help="synthetic: correlation length, km")
    g.add_argument("--prior-gamma", type=float, default=10.0, help="observed: the prior is (-gamma Laplacian + delta)^-2")
    g.add_argument("--prior-delta", type=float, default=1e-5)
    g.add_argument("--prior-bc", choices=("robin", "natural"), default="robin",
                   help="boundary condition of the prior's operator; the paper's natural one leaves islands of a few "
                        "cells a standard deviation above 1000")
    g = p.add_argument_group("inversion")
    g.add_argument("--init", default=None,
                   help="where Gauss-Newton starts: mean (the prior's), driving (the friction of the driving stress and "
                        "the observed speed), fit:N (driving, then N steps of beta += log(modelled speed / observed "
                        "speed)), or from:DIR (the MAP point of the run in DIR, on any grid).  Default: mean for the "
                        "synthetic problem, fit:6 for the observed one")
    g.add_argument("--max-newton", type=int, default=None, help="0: take the starting point, or a stopped run's iterate")
    g.add_argument("--cg-max", type=int, default=None)
    g.add_argument("--rtol", type=float, default=None, help="gradient reduction of the MAP point (1e-5; observed 1e-3)")
    g.add_argument("--g-ref", type=float, default=None,
                   help="the gradient norm --rtol is relative to (default: the first); for a start from another run's "
                        "MAP point, about that run's first gradient")
    g.add_argument("--friction-cap", type=float, default=None,
                   help="the largest friction (MPa a/km), approached smoothly; 0 for none (see stokes.friction).  "
                        "Default: 1e6 for the observed problem, whose prior does not hold beta where the bed is frozen")
    g.add_argument("--hessian", choices=("lowrank", "local"), default="lowrank",
                   help="the misfit Hessian in the posterior: its dominant eigenpairs (lowrank), or the sparse matrix "
                        "of its entries within --reach of each node, from (2 reach / node spacing)^2 products, for "
                        "data that see the friction locally (local; one GPU, the direct solver)")
    g.add_argument("--reach", type=float, default=120.0,
                   help="local: km within which the entries of the misfit Hessian are kept (at 20 km a column is "
                        "1.5 %% of its diagonal 100 km away)")
    g.add_argument("--local-solver", choices=("auto", "direct", "cg"), default="auto",
                   help="local: the solves with the posterior precision, by a factorization of its sparse system or "
                        "by conjugate gradients (auto: the factorization while it is small)")
    g.add_argument("--map-solver", choices=("newton", "local"), default="newton",
                   help="the MAP point by inexact Gauss-Newton-CG, or (local) by Newton's method with the sparse misfit "
                        "Hessian of another run on the same footprint (--local-from) in the place of this one's: a "
                        "forward and an adjoint solve an iteration, no Hessian products")
    g.add_argument("--local-from", default=None, help="the folder of a run with --hessian local, for --map-solver local")
    g.add_argument("--local-share", default="0/1", metavar="K/N",
                   help="local: take the combs whose number is K modulo N; N runs, one GPU each, share the products, and "
                        "the run that finds them all there goes on")
    g.add_argument("--draws", choices=("auto", "adjoint", "lanczos"), default="auto",
                   help="local: the misfit's part of a posterior draw from an adjoint solve (the Hessian itself; auto "
                        "with the direct solver), or from the sparse matrix's square root by Lanczos (no solve)")
    g.add_argument("--block", type=int, default=150, help="columns added to the low rank at a time")
    g.add_argument("--max-cols", type=int, default=4000)
    g.add_argument("--var-samples", type=int, default=1000)
    g.add_argument("--nsamples", type=int, default=4, help="prior and posterior samples saved")
    g.add_argument("--qoi-samples", type=int, default=0,
                   help="posterior samples pushed through the nonlinear forward problem in the qoi stage")
    g.add_argument("--qoi-share", default="0/1", metavar="K/N",
                   help="take the samples whose number is K modulo N: N runs of the qoi stage, one GPU each, share them")
    g = p.add_argument_group("run")
    g.add_argument("--solver", choices=("cudss", "scipy", "columns"), default="cudss",
                   help="scipy: a host solve, for small meshes; columns: no matrix at all (matrixfree.py), for meshes "
                        "whose factors do not fit a card (10 km with five layers)")
    g.add_argument("--krylov-rtol", type=float, default=1e-8, help="columns: tolerance of the linear solves")
    g.add_argument("--forward-rtol", type=float, default=1e-8,
                   help="the fall of the residual that ends a forward solve (with --solver columns a tighter one, and a "
                        "tighter --krylov-rtol, take the gradient's accuracy further)")
    g.add_argument("--factors", choices=("double", "single"), default="double",
                   help="cudss: the precision of the factors; single halves their memory, and the solutions are "
                        "refined against the double-precision matrix either way")
    g.add_argument("--stages", default="data,map,eig,post,qoi")
    g.add_argument("--time-limit", type=float, default=0.0, help="seconds after which a stage stops where it can resume")
    g.add_argument("--out", required=True)
    a = p.parse_args(argv)
    for k, v in DEFAULTS.items():
        if getattr(a, k) is None:
            setattr(a, k, v[a.data == "observed"])
    a.friction_cap = a.friction_cap or None
    try:
        k, n = (int(v) for v in a.qoi_share.split("/"))
    except ValueError:
        p.error("--qoi-share is K/N")
    if not 0 <= k < n:
        p.error("--qoi-share K/N needs 0 <= K < N")
    a.qoi_share = [k, n]
    try:
        k, n = (int(v) for v in a.local_share.split("/"))
    except ValueError:
        p.error("--local-share is K/N")
    if not 0 <= k < n:
        p.error("--local-share K/N needs 0 <= K < N")
    a.local_share = [k, n]
    return a


class Run:
    def __init__(self, a):
        import hippymfem as hm  # noqa: F401  (before mpi4py: it sets up the devices)
        from mpi4py import MPI
        import antarctica as an
        import stokes as st

        self.a, self.comm = a, MPI.COMM_WORLD
        self.rank0 = self.comm.rank == 0
        self.t0 = time.time()
        os.makedirs(a.out, exist_ok=True) if self.rank0 else None
        self.comm.Barrier()
        self.grid = an.ModelGrid.load(a.res)
        self.cols = an.Columns(self.grid, a.nz)
        if a.solver == "cudss":
            import gpu_direct
            solver = gpu_direct.CuDSSDirect(self.comm, precision=a.factors)
        elif a.solver == "columns":
            solver = None
        else:
            solver = st.ScipyDirect(self.comm)
        self.pb = st.IceSheet(self.cols, self.comm, solver=solver, verbose=False, friction_cap=a.friction_cap,
                              friction_order=a.friction_order, partition="columns" if a.solver == "columns" else "metis")
        if a.solver == "columns":
            import matrixfree
            self.pb.solver = matrixfree.ColumnStokes(self.pb, rtol=a.krylov_rtol, forcing=True, restart=80,
                                                     coarse="direct" if self.comm.size == 1 else "ilu")
        self.pb.forward_rtol = a.forward_rtol
        self.fhalf = a.friction_order == 2                # the friction has dofs half way between the grid's nodes
        self.sizes = dict(self.pb.summary(), grid=self.grid.summary())
        self.dump("sizes.json", self.sizes)
        self.say("icesheet: %s" % json.dumps({k: v for k, v in self.sizes.items() if k != "grid"}))

    def late(self, ahead=0.0):
        """Whether the time limit will have passed ``ahead`` seconds from now, or a STOP
        file asks the run to end: rank 0's answer, the same on every rank."""
        a, mine = self.a, None
        if self.rank0:
            mine = bool(a.time_limit and time.time() + ahead > self.t0 + a.time_limit) or stop_requested(a.out)
        return self.comm.bcast(mine, root=0)

    def say(self, *a):
        if self.rank0:
            print("[%7.1f s] " % (time.time() - self.t0) + " ".join(str(x) for x in a), flush=True)

    def path(self, name):
        return os.path.join(self.a.out, name)

    def _tmp(self, name):
        # a name of this process's own: runs that share a folder (--qoi-share) write the same files
        return self.path("%s.%s.%d.tmp" % (name, socket.gethostname(), os.getpid()))

    def dump(self, name, obj):
        """Written whole: a run killed while it writes leaves the file as it was."""
        if self.rank0:
            with open(self._tmp(name), "w") as f:
                json.dump(obj, f, indent=1)
            os.replace(self._tmp(name), self.path(name))

    def keys(self, xy, half=False):
        """Integer keys of positions on the grid's nodes (``half``: on the Q2 nodes too),
        to match fields saved on rank 0 with each rank's dofs."""
        g = self.grid
        step = g.h / 2 if half else g.h
        k = np.rint((np.asarray(xy)[:, :2] - np.array([g.xn[0], g.yn[0]])) / step).astype(np.int64)
        return k[:, 0] * 10 ** 7 + k[:, 1]

    def node_values(self, field2d):
        """A field on the grid's nodes at the footprint dofs: its value at a node, and at a dof
        between nodes (a quadratic friction's) the mean of the nodes it lies between."""
        g = self.grid
        xy = self.pb.Vm.coordinates()
        fx, fy = (xy[:, 0] - g.xn[0]) / g.h, (xy[:, 1] - g.yn[0]) / g.h
        I, J = np.floor(fx + 1e-9).astype(int), np.floor(fy + 1e-9).astype(int)
        tx, ty = fx - I, fy - J
        on_x, on_y = tx < 1e-9, ty < 1e-9
        # a dof on a node (or on a line of nodes) takes nothing from beyond it, where there may be no ice
        I1, J1 = I + ~on_x, J + ~on_y
        tx, ty = np.where(on_x, 0.0, tx), np.where(on_y, 0.0, ty)
        return ((1 - tx) * (1 - ty) * field2d[J, I] + tx * (1 - ty) * field2d[J, I1]
                + (1 - tx) * ty * field2d[J1, I] + tx * ty * field2d[J1, I1])

    def save_footprint(self, name, v):
        """A footprint vector as rows (x, y, value), on rank 0."""
        X = self.comm.gather(self.pb.Vm.coordinates()[:, :2], root=0)
        V = self.comm.gather(np.asarray(v.array), root=0)
        if self.rank0:
            np.save(self._tmp(name) + ".npy", np.column_stack((np.concatenate(X), np.concatenate(V))))
            os.replace(self._tmp(name) + ".npy", self.path(name))

    def load_footprint(self, name):
        """The values of a saved footprint field at this rank's dofs."""
        arr = np.load(self.path(name))
        ks = self.keys(arr, half=self.fhalf)
        order = np.argsort(ks)
        mine = self.keys(self.pb.Vm.coordinates(), half=self.fhalf)
        pos = np.minimum(np.searchsorted(ks[order], mine), len(ks) - 1)
        if not np.all(ks[order[pos]] == mine):
            raise ValueError("%s was saved on another grid" % name)
        return arr[order[pos], 2]


# ------------------------------------------------------------------ the data
def log_friction(g):
    """The log of the friction built from the driving stress and the observed speed, on the
    grid's nodes (the truth of the synthetic problem); NaN where there is no ice."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(g.driving_stress_friction())


def stage_data(R):
    if os.path.exists(R.path("data.npz")):
        z = dict(np.load(R.path("data.npz")))
        R.say("data: read %d observation points" % z["targets"].shape[0])
        return z
    return data_synthetic(R) if R.a.data == "synthetic" else data_observed(R)


def data_synthetic(R):
    """The model's own surface velocity at the true friction, with noise: the field of the
    observed problem's integral misfit (every Q2 node of the surface where a velocity is
    observed, noise per ``obs_spacing^2`` of surface), or with ``--misfit nodes`` one
    independent observation at every node of the grid."""
    from hippymfem.modeling.variables import STATE
    import stokes as st

    a, pb, g = R.a, R.pb, R.grid
    if a.misfit == "integral":
        return _synthetic_field(R)
    J, I = np.nonzero(g.node_used & np.isfinite(g.uobs).all(axis=-1))
    # just under the surface: a point on it can be missed by point location
    targets = np.column_stack((g.xn[I], g.yn[J], g.s[J, I] - 0.01 * (g.s[J, I] - g.b[J, I]) / a.nz))
    mtrue = pb.Vm.vector()
    mtrue.array[:] = R.node_values(log_friction(g))
    x = [pb.generate_state(), mtrue, None]
    t0 = time.time()
    pb.solveFwd(x[STATE], x)
    R.say("data: the forward solve at the true friction, %d Newton steps, %.1f s" % (pb.last_forward["newton"], time.time() - t0))
    mis = st.SurfaceVelocityMisfit(pb, targets, np.zeros((len(targets), 2)), np.ones(len(targets)), missing="ignore")
    clean = mis.prediction(x[STATE])
    R.save_footprint("truth.npy", mtrue)
    out = dict(targets=targets, clean=R.comm.bcast(clean), mtrue_mean=np.array(mtrue.sum() / mtrue.global_size))
    truth = R.comm.bcast(np.load(R.path("truth.npy")) if R.rank0 else None)
    out.update(mtrue_xy=truth[:, :2], mtrue=truth[:, 2])
    for r in st.REGIONS:
        out["flux_true_" + r] = np.array(pb.mass_flux(x[STATE], r))
    speed = np.hypot(out["clean"][:, 0], out["clean"][:, 1])
    sigma = a.noise * np.sqrt(speed ** 2 + (EPS_OBS["synthetic"] / 1e3) ** 2)
    out.update(d=out["clean"] + sigma[:, None] * np.random.default_rng(a.seed).standard_normal(out["clean"].shape),
               sigma=sigma)
    # the mesh's point location misses some of the nodes in thin, steep columns by the
    # margins: they are left out (the noise is drawn first, so the others keep theirs)
    found = mis.located()
    for k in ("targets", "clean", "d", "sigma"):
        out[k] = out[k][found]
    speed = speed[found]
    if R.rank0:
        np.savez(R.path("data.npz"), **out)
    R.say("data: %d points%s, the model's velocities at the true friction, noise %.0f %% of the speed; speed median "
          "%.1f m/a, max %.0f m/a" % (int(found.sum()), " (%d more that the mesh does not locate are left out)"
                                      % int((~found).sum()) if not found.all() else "",
                                      100 * a.noise, 1e3 * np.median(speed), 1e3 * speed.max()))
    return out


def _synthetic_field(R):
    """The synthetic data of the integral misfit: the model's velocity at the true friction at
    the surface nodes of the observed problem, with the noise that misfit assumes, independent
    at the nodes with variance ``(obs_spacing sigma)^2 / area``."""
    from hippymfem.modeling.variables import STATE
    import stokes as st

    a, pb, g = R.a, R.pb, R.grid
    out = _surface_nodes(R)                                       # targets, area, filled (rank 0)
    out = R.comm.bcast(out)
    mtrue = pb.Vm.vector()
    mtrue.array[:] = R.node_values(log_friction(g))
    x = [pb.generate_state(), mtrue, None]
    t0 = time.time()
    pb.solveFwd(x[STATE], x)
    R.say("data: the forward solve at the true friction, %d Newton steps, %.1f s" % (pb.last_forward["newton"], time.time() - t0))
    n = len(out["targets"])
    mis = _integral_misfit(R, dict(targets=out["targets"], d=np.zeros((n, 2)), sigma=np.ones(n)))
    clean = R.comm.bcast(mis.prediction(x[STATE]))
    R.save_footprint("truth.npy", mtrue)
    truth = R.comm.bcast(np.load(R.path("truth.npy")) if R.rank0 else None)
    out.update(clean=clean, mtrue_mean=np.array(mtrue.sum() / mtrue.global_size), mtrue_xy=truth[:, :2], mtrue=truth[:, 2])
    for r in st.REGIONS:
        out["flux_true_" + r] = np.array(pb.mass_flux(x[STATE], r))
    speed = np.hypot(clean[:, 0], clean[:, 1])
    sigma = a.noise * np.sqrt(speed ** 2 + (EPS_OBS["synthetic"] / 1e3) ** 2)
    node = a.obs_spacing * sigma / np.sqrt(out["area"])           # the noise of one node
    out.update(d=clean + node[:, None] * np.random.default_rng(a.seed).standard_normal(clean.shape), sigma=sigma)
    out.pop("d_observed", None)
    if R.rank0:
        np.savez(R.path("data.npz"), **out)
    R.say("data: the model's velocities at the true friction over %.4g km^2 of surface (%d Q2 nodes), noise %.0f %% of "
          "the speed per %g km x %g km (%.0f %% at a node, median); speed median %.1f m/a, max %.0f m/a"
          % (out["area"].sum(), n, 100 * a.noise, a.obs_spacing, a.obs_spacing, 100 * np.median(node / np.maximum(speed, 1e-12)),
             1e3 * np.median(speed), 1e3 * speed.max()))
    return out


def _surface_nodes(R):
    """On rank 0: the Q2 nodes of the surface where a velocity is observed, in an order that
    does not depend on the partition, with their areas and the observed velocity
    (``d_observed``, bilinear between the grid's nodes)."""
    pb, g = R.pb, R.grid
    sn = pb.surface_nodes()
    parts = R.comm.gather((sn["xyz"], sn["area"]), root=0)
    if not R.rank0:
        return None
    xyz, area = (np.concatenate([q[k] for q in parts]) for k in (0, 1))
    order = np.argsort(R.keys(xyz, half=True))
    xyz, area = xyz[order], area[order]
    # bilinear between the grid's nodes; NaN where a node it needs has no observation
    fx, fy = (xyz[:, 0] - g.xn[0]) / g.h, (xyz[:, 1] - g.yn[0]) / g.h
    i0 = np.clip(np.floor(fx + 1e-9).astype(int), 0, g.uobs.shape[1] - 2)
    j0 = np.clip(np.floor(fy + 1e-9).astype(int), 0, g.uobs.shape[0] - 2)
    tx, ty = fx - i0, fy - j0
    d = np.zeros((len(xyz), 2))
    filled = np.zeros(len(xyz), bool)                             # the velocity is MEaSUREs' (ITS_LIVE has none there)
    for wgt, (jj, ii) in (((1 - tx) * (1 - ty), (j0, i0)), (tx * (1 - ty), (j0, i0 + 1)),
                          ((1 - tx) * ty, (j0 + 1, i0)), (tx * ty, (j0 + 1, i0 + 1))):
        use = wgt > 1e-9
        d[use] += wgt[use, None] * np.where(g.node_used[jj, ii][:, None], g.uobs[jj, ii], np.nan)[use]
        filled[use] |= g.ufill[jj, ii][use]
    keep = np.isfinite(d).all(axis=1)
    return dict(targets=xyz[keep], area=area[keep], d_observed=d[keep], filled=filled[keep])


def _integral_misfit(R, z):
    """The integral misfit on this rank's surface nodes, matched to the data by their place;
    the weights are area / (spacing sigma)^2."""
    import stokes as st

    pb = R.pb
    sn = pb.surface_nodes()
    keys = R.keys(z["targets"], half=True)
    order = np.argsort(keys)
    mine = R.keys(sn["xyz"], half=True)
    rows = order[np.minimum(np.searchsorted(keys[order], mine), len(keys) - 1)]
    have = keys[rows] == mine
    return st.SurfaceIntegralMisfit(pb, z["d"][rows], R.a.obs_spacing * np.where(have, z["sigma"][rows], 1.0), have=have,
                                    rows=rows, n_global=len(keys))


def data_observed(R):
    """The observed velocity at every Q2 node of the surface (bilinear between the grid's
    nodes), with the node's area: the data of the paper's integral misfit."""
    a = R.a
    out = _surface_nodes(R)
    if R.rank0:
        d = out.pop("d_observed")
        out.update(d=d, sigma=a.noise * np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + (EPS_OBS["observed"] / 1e3) ** 2))
        np.savez(R.path("data.npz"), **out)
    out = R.comm.bcast(out)
    speed = np.hypot(out["d"][:, 0], out["d"][:, 1])
    R.say("data: the observed velocity over %.4g km^2 of surface (%d Q2 nodes, %d of them from MEaSUREs), noise "
          "%.0f %% of the speed per %g km x %g km; speed median %.1f m/a, max %.0f m/a"
          % (out["area"].sum(), len(speed), int(out["filled"].sum()), 100 * a.noise, a.obs_spacing, a.obs_spacing,
             1e3 * np.median(speed), 1e3 * speed.max()))
    return out


def make_model(R, z):
    """The misfit, the prior and ``hm.Model``."""
    import hippymfem as hm
    import stokes as st

    a, pb = R.a, R.pb
    # the misfit by the data (a field with areas: the integral), the prior by the problem
    mis = _integral_misfit(R, z) if "area" in z else st.SurfaceVelocityMisfit(pb, z["targets"], z["d"], z["sigma"])
    if a.data == "observed":
        gamma, delta, mean = a.prior_gamma, a.prior_delta, 0.0
    else:
        gamma, delta = hm.BiLaplacianComputeCoefficients(a.prior_std ** 2, a.prior_corr, 2)
        mean = float(z["mtrue_mean"])
    m0 = pb.Vm.vector()
    m0.array[:] = mean
    prior = hm.BiLaplacianPrior(pb.Vm, gamma, delta, mean=m0, robin_bc=a.prior_bc == "robin")
    R.prior = dict(gamma=float(gamma), delta=float(delta), mean=mean, bc=a.prior_bc)
    return hm.Model(pb, prior, mis), prior


def stage_verify(R, model, prior):
    """Finite-difference checks of the gradient and of the linearized forward map."""
    import hippymfem as hm
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    pb = R.pb
    hm.parRandom.set_seed(5)
    m = prior.mean.copy()
    s = pb.Vm.vector()
    prior.sample(prior.sample_noise(1.0), s, add_mean=False)
    m.axpy(0.3, s)
    dm = pb.Vm.vector()
    prior.sample(prior.sample_noise(1.0), dm, add_mean=False)
    x = model.generate_vector()
    x[PARAMETER] = m
    model.solveFwd(x[STATE], x)
    c0 = model.cost(x)[0]
    model.solveAdj(x[ADJOINT], x)
    g = pb.Vm.vector()
    model.evalGradientParameter(x, g)
    gdm = g.inner(dm)
    rows = []
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    # linearized forward map: u_hat = -J^{-1} C dm
    rhs = pb.generate_state()
    model.applyC(dm, rhs)
    rhs.scale(-1.0)
    uh = pb.generate_state()
    model.solveFwdIncremental(uh, rhs)
    for h in (1e-1, 1e-2, 1e-3, 1e-4):
        xh = model.generate_vector()
        xh[PARAMETER] = m.copy().axpy(h, dm)
        model.solveFwd(xh[STATE], xh)
        ch = model.cost(xh)[0]
        du = xh[STATE].copy().axpy(-1.0, x[STATE])
        du.scale(1.0 / h)
        e_state = du.copy().axpy(-1.0, uh).norm() / max(uh.norm(), 1e-300)
        e_grad = abs((ch - c0) / h - gdm) / max(abs(gdm), 1e-300)
        rows.append(dict(h=h, grad=e_grad, state=e_state))
        R.say("verify: h %.0e: gradient %.2e, linearized state %.2e" % (h, e_grad, e_state))
    # symmetry of the Gauss-Newton Hessian
    H = hm.ReducedHessian(model, misfit_only=True)
    d1, d2 = pb.Vm.vector(), pb.Vm.vector()
    prior.sample(prior.sample_noise(1.0), d1, add_mean=False)
    prior.sample(prior.sample_noise(1.0), d2, add_mean=False)
    h1, h2 = pb.Vm.vector(), pb.Vm.vector()
    H.mult(d1, h1)
    H.mult(d2, h2)
    sym = abs(h1.inner(d2) - h2.inner(d1)) / max(abs(h1.inner(d2)), 1e-300)
    R.say("verify: (H d1, d2) = %.8e, (H d2, d1) = %.8e, relative difference %.1e" % (h1.inner(d2), h2.inner(d1), sym))
    pb.forget()
    return dict(fd=rows, hessian_symmetry=sym)


# ------------------------------------------------------------- the MAP point
def initial_guess(R, model, prior, z):
    """Where Gauss-Newton starts (``--init``)."""
    from mpi4py import MPI
    from hippymfem.modeling.variables import PARAMETER, STATE

    a, pb, g = R.a, R.pb, R.grid
    m0 = prior.mean.copy()
    if a.init == "mean":
        return m0
    if a.init.startswith("from:"):
        # another run's MAP point: linear between its nodes, the nearest of them outside
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        src = np.load(os.path.join(a.init[5:], "map.npy"))
        xy = pb.Vm.coordinates()[:, :2]
        v = LinearNDInterpolator(src[:, :2], src[:, 2])(xy)
        far = ~np.isfinite(v)
        v[far] = NearestNDInterpolator(src[:, :2], src[:, 2])(xy[far])
        m0.array[:] = v
        R.say("map: starting from the MAP point of %s (%d nodes)" % (a.init[5:], len(src)))
        return m0
    m0.array[:] = R.node_values(log_friction(g))
    if not a.init.startswith("fit:"):
        return m0
    # beta += log(modelled speed / observed speed) at the footprint nodes that have a datum
    # above them: the friction that carries the observed speed where the ice slides.  The
    # best of the iterates is kept (the update looks at the vertices only, and the misfit,
    # which integrates over all the surface nodes, can go up again)
    half = "area" in z                                  # the integral's data are on the Q2 nodes
    keys = R.keys(z["targets"], half=half)
    order = np.argsort(keys)
    mine = R.keys(pb.Vm.coordinates(), half=half)
    rows = order[np.minimum(np.searchsorted(keys[order], mine), len(keys) - 1)]
    have = keys[rows] == mine
    obs = np.hypot(z["d"][rows, 0], z["d"][rows, 1])
    lo, hi, floor = np.log(1e-3), np.log(1e4), 1e-4               # MPa a/km; 0.1 m/a
    x = model.generate_vector()
    best, best_cost, worse = m0.copy(), np.inf, 0
    for it in range(int(a.init.split(":")[1]) + 1):
        x[PARAMETER] = m0
        model.solveFwd(x[STATE], x)
        c = model.cost(x)
        pred = R.comm.bcast(model.misfit.prediction(x[STATE]))
        mod = np.hypot(pred[rows, 0], pred[rows, 1])
        step = np.where(have, np.clip(np.log((mod + floor) / (obs + floor)), -1.0, 1.0), 0.0)
        R.say("map: fit %2d: misfit %.4e, regularization %.4e; rms log speed ratio %.3f"
              % (it, c[2], c[1], np.sqrt(R.comm.allreduce(float(np.sum(step ** 2)), op=MPI.SUM)
                                         / max(R.comm.allreduce(int(have.sum()), op=MPI.SUM), 1))))
        worse = worse + 1 if c[0] >= best_cost else 0
        if c[0] < best_cost:
            best, best_cost = m0.copy(), c[0]
        if worse == 2:
            break
        m0.array[:] = np.clip(m0.array + step, lo, hi)
    return best


def state_file(R):
    return R.path("map_state_r%03d_of%03d.npy" % (R.comm.rank, R.comm.size))


def finish_map(R, model, x, hist):
    """Save the MAP point, its state (a later job on as many ranks starts from it instead
    of a forward solve from rest) and the model's velocity at the data."""
    from hippymfem.modeling.variables import PARAMETER, STATE

    R.save_footprint("map.npy", x[PARAMETER])
    np.save(state_file(R), np.asarray(x[STATE].array))
    pred = model.misfit.prediction(x[STATE])
    if R.rank0:
        np.save(R.path("map_velocity.npy"), pred)
    hist["flux_map"] = R.pb.mass_flux(x[STATE])
    R.dump("map.json", hist)


def stage_map_local(R, model, prior, z):
    """The MAP point by Newton's method with a sparse misfit Hessian from another run on the same
    footprint (``--local-from``: fewer layers, the factorization's block solves) in the place
    of this run's Hessian: ``m <- m - alpha (H_s + R)^-1 g`` with ``alpha`` halved while the
    cost does not fall.  An iteration is a forward and an adjoint solve; the Hessian of the
    other mesh is close to this one's, so the gradient falls by a constant factor a step."""
    import glob
    import scipy.sparse as sp
    import localhessian as lh
    from hippymfem.common.linalg import hypre_to_scipy
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    a, pb = R.a, R.pb
    saved = os.path.exists(R.path("map.json")) and os.path.exists(R.path("map.npy"))
    hist = json.load(open(R.path("map.json"))) if saved else {"iterations": []}
    if hist.get("done"):
        return stage_map(R, model, prior, z)                 # read, with its state
    if saved:
        m = prior.mean.copy()
        m.array[:] = R.load_footprint("map.npy")
        R.say("map: resuming after %d iterations" % len(hist["iterations"]))
    else:
        m = initial_guess(R, model, prior, z)
    files = sorted(glob.glob(os.path.join(a.local_from or "", "local_reach*.npz")))
    if not files:
        raise SystemExit("--map-solver local needs --local-from, the folder of a run with --hessian local")
    Hs = sp.load_npz(files[-1])
    if Hs.shape[0] != m.local_size:
        raise SystemExit("the sparse Hessian in %s is of another footprint (%d parameters, here %d)" % (a.local_from, Hs.shape[0], m.local_size))
    K = lh.LocalPosterior(Hs, hypre_to_scipy(prior.A).tocsr(), hypre_to_scipy(prior.M).tocsr(), pb, model.misfit, prior, m,
                          R.comm, method=a.local_solver)
    del Hs
    t0 = time.time()
    last = hist["iterations"][-1] if hist["iterations"] else dict(seconds=0.0)

    def evaluate(mv):
        xs = model.generate_vector()
        xs[PARAMETER] = mv
        model.solveFwd(xs[STATE], xs)
        return xs, model.cost(xs)

    xs, c = evaluate(m)
    g, d = pb.Vm.vector(), pb.Vm.vector()
    converged, alpha, reason = False, 1.0, "the iterations ran out"
    if a.g_ref and not hist.get("g0"):
        hist["g0"] = float(a.g_ref)                          # the gradient the tolerance is measured against
    while True:
        model.solveAdj(xs[ADJOINT], xs)
        gn = float(model.evalGradientParameter(xs, g))
        hist.setdefault("g0", gn)
        it = len(hist["iterations"]) + 1
        hist["iterations"].append(dict(it=it, cg=0, cost=c[0], reg=c[1], misfit=c[2], g=gn, alpha=alpha,
                                       seconds=last["seconds"] + time.time() - t0, **pb.counters))
        R.save_footprint("map.npy", xs[PARAMETER])
        R.dump("map.json", hist)
        R.say("map: %2d: cost %.6e (misfit %.6e, regularization %.4e), |g| %.3e (%.1e of the first), step %.3g, %.0f s"
              % (it, c[0], c[2], c[1], gn, gn / hist["g0"], alpha, hist["iterations"][-1]["seconds"]))
        converged = gn <= a.rtol * hist["g0"]
        if converged:
            reason = "the gradient fell by --rtol"
        if converged or it >= a.max_newton:
            break
        if R.late():
            R.say("map: stopped (time limit or STOP file) after %d iterations" % it)
            return None, hist
        d.array[:] = -K.solve(np.asarray(g.array)[:, None])[:, 0]
        alpha, lower = 1.0, False
        while alpha >= 1.0 / 64.0:
            mt = xs[PARAMETER].copy()
            mt.axpy(alpha, d)
            xt, ct = evaluate(mt)
            lower = ct[0] < c[0]
            if lower:
                break
            alpha *= 0.5
        if not lower:
            # the gradient is at the accuracy of the solves: no step along it lowers the cost
            reason = "no step lowers the cost any more"
            break
        xs, c = xt, ct
    hist.update(done=True, converged=bool(converged), newton=len(hist["iterations"]), cg=0, reason=reason,
                seconds=hist["iterations"][-1]["seconds"], g=hist["iterations"][-1]["g"], solver="local")
    finish_map(R, model, xs, hist)
    R.say("map: %s; %d iterations with the sparse Hessian of %s, |g|/|g0| %.2e, %.1f s"
          % (hist["reason"], hist["newton"], a.local_from, hist["g"] / hist["g0"], hist["seconds"]))
    return xs, hist


def stage_map(R, model, prior, z):
    import hippymfem as hm
    from mpi4py import MPI
    from hippymfem.modeling.variables import PARAMETER, STATE

    a, pb = R.a, R.pb
    saved = os.path.exists(R.path("map.json")) and os.path.exists(R.path("map.npy"))
    hist = json.load(open(R.path("map.json"))) if saved else {"iterations": []}
    m0 = prior.mean.copy()
    if saved:
        m0.array[:] = R.load_footprint("map.npy")
    if hist.get("done"):
        x = model.generate_vector()
        x[PARAMETER] = m0
        f = state_file(R)
        have = R.comm.allreduce(os.path.exists(f), op=MPI.LAND)
        if have:
            x[STATE].array[:] = np.load(f)
        if have and pb.is_solution(x):
            R.say("map: read, with its state")
        else:
            model.solveFwd(x[STATE], x)
            np.save(f, np.asarray(x[STATE].array))
            R.say("map: read")
        return x, hist
    if saved:
        # a stopped run: Gauss-Newton continues from the iterate it saved, with the
        # tolerance measured against the first gradient
        R.say("map: resuming after %d Gauss-Newton iterations" % len(hist["iterations"]))
    else:
        m0 = initial_guess(R, model, prior, z)
    n_base = len(hist["iterations"])
    last = hist["iterations"][-1] if n_base else dict(seconds=0.0, cg=0)
    if a.max_newton == 0:
        # no (more) inversion: the starting point, or the iterate a stopped run saved
        x = model.generate_vector()
        x[PARAMETER] = m0
        model.solveFwd(x[STATE], x)
        c = model.cost(x)
        hist.update(done=True, converged=False, newton=n_base, cg=last["cg"], seconds=last["seconds"], cost=c[0],
                    reason="taken as it is (--max-newton 0) after %d Gauss-Newton iterations" % n_base)
        finish_map(R, model, x, hist)
        R.say("map: %s; misfit %.4e, regularization %.4e" % (hist["reason"], c[2], c[1]))
        return x, hist
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["max_iter"] = a.max_newton
    params["GN_iter"] = a.max_newton + 1
    params["cg_max_iter"] = a.cg_max
    params["rel_tolerance"] = a.rtol
    params["abs_tolerance"] = 0.0
    params["print_level"] = 0 if R.rank0 else -1
    if a.g_ref and not hist.get("g0"):
        hist["g0"] = float(a.g_ref)
    if hist.get("g0"):
        params["rel_tolerance"] = 0.0
        params["abs_tolerance"] = a.rtol * hist["g0"]
    solver = hm.ReducedSpaceNewtonCG(model, params)
    t0 = time.time()

    def callback(it, x):
        c = model.cost(x)
        hist["iterations"].append(dict(it=n_base + it, cg=last["cg"] + int(solver.total_cg_iter), cost=c[0], reg=c[1],
                                       misfit=c[2], seconds=last["seconds"] + time.time() - t0, **pb.counters))
        hist.setdefault("g0", float(solver.initial_grad_norm))
        R.save_footprint("map.npy", x[PARAMETER])
        R.dump("map.json", hist)
        if R.late():
            raise TimeoutError
    solver.callback = callback
    try:
        x = solver.solve([None, m0, None])
    except TimeoutError:
        R.say("map: stopped (time limit or STOP file) after %d Gauss-Newton iterations" % len(hist["iterations"]))
        return None, hist
    hist.update(done=True, converged=bool(solver.converged), reason=solver.termination_reasons[solver.reason],
                newton=len(hist["iterations"]), cg=last["cg"] + int(solver.total_cg_iter),
                seconds=last["seconds"] + time.time() - t0, g=float(solver.final_grad_norm))
    hist.setdefault("g0", float(solver.initial_grad_norm))
    finish_map(R, model, x, hist)
    R.say("map: %s; %d Gauss-Newton iterations, %d CG, |g|/|g0| %.2e, %.1f s"
          % (hist["reason"], hist["newton"], hist["cg"], hist["g"] / hist["g0"], hist["seconds"]))
    return x, hist


# ------------------------------------------------ the posterior and the predictions
class BlockHessian:
    """The misfit Hessian with a product for a block of directions at once
    (``IceSheet.misfit_hessian_block``) beside the library's one-vector one."""

    def __init__(self, H, pb, misfit):
        self.H, self.pb, self.misfit = H, pb, misfit

    def mult(self, x, y):
        return self.H.mult(x, y)

    def mult_block(self, M):
        return self.pb.misfit_hessian_block(self.misfit, M)


def stage_eig(R, model, prior, x):
    import hippymfem as hm
    from hippymfem.modeling.variables import PARAMETER
    import lowrank

    a, pb = R.a, R.pb
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    H = hm.ReducedHessian(model, misfit_only=True)
    if a.solver == "cudss" and R.comm.size == 1:
        H = BlockHessian(H, pb, model.misfit)          # block solves of the factorization
    lr = lowrank.GrowingLowRank(H, prior, x[PARAMETER], R.path("lowrank"))
    t0 = time.time()
    done, counts = lowrank.grow(lr, x[PARAMETER], first=FIRST, block=a.block, p=OVERSAMPLE, max_cols=a.max_cols,
                                late=R.late, log=R.say)
    if not done:
        return None, None, dict(done=False, history=lr.history)
    k = lr.kept(OVERSAMPLE)
    d, U = lr.eig(k)
    rec = dict(done=True, k=k, columns=lr.ncols, d=[float(v) for v in d], above_one=int((d > 1).sum()),
               counts_above_one=counts, history=lr.history, seconds=time.time() - t0)
    R.dump("eig.json", rec)
    R.say("eig: %d eigenpairs, %.4g .. %.4g, %d above 1" % (k, d[0], d[-1], rec["above_one"]))
    return d, U, rec


def stage_local(R, model, prior, x):
    """The misfit Hessian at the MAP point as a sparse matrix (``localhessian.py``), from
    products with combs of unit vectors, checked against true products with other vectors."""
    import scipy.sparse as sp
    import hippymfem as hm
    import localhessian as lh

    a, pb, g = R.a, R.pb, R.grid
    step = g.h / a.friction_order
    name = "local_reach%g" % a.reach
    if os.path.exists(R.path(name + ".json")) and os.path.exists(R.path(name + ".npz")):
        rec = json.load(open(R.path(name + ".json")))
        R.say("local: read the sparse Hessian (%d products, %.0f entries a row)" % (rec["products"], rec["entries_per_row"]))
        return sp.load_npz(R.path(name + ".npz")), rec
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    IJ = lh.lattice(pb.Vm.coordinates(), [g.xn[0], g.yn[0]], step)
    s = lh.spacing(a.reach, step)
    n = IJ.shape[0]
    t0 = time.time()
    saved = R.path("local_products_s%d.npy" % s)
    if hasattr(pb.solver, "solve_many"):
        apply = lambda M: pb.misfit_hessian_block(model.misfit, M)           # block solves of the factorization
        chunk = 256
    else:
        Hop = hm.ReducedHessian(model, misfit_only=True)
        vin, vout = pb.Vm.vector(), pb.Vm.vector()

        def apply(M):                                                         # a product a column
            out = np.zeros_like(M)
            for k in range(M.shape[1]):
                vin.array[:] = M[:, k]
                Hop.mult(vin, vout)
                out[:, k] = vout.array
            return out
        chunk = 4
    K, N = a.local_share
    if os.path.exists(saved):
        HM, color = np.load(saved), (IJ[:, 0] % s) * s + (IJ[:, 1] % s)
        R.say("local: read the %d products with combs %d nodes apart" % (s * s, s))
    elif N > 1 or not hasattr(pb.solver, "solve_many"):
        # this run's share of the combs, kept as it grows; then all the shares, if they are there
        mine = np.arange(K, s * s, N)
        f = R.path("local_products_s%d.%dof%d.npy" % (s, K, N))
        have = np.load(f) if os.path.exists(f) else np.zeros((n, 0), dtype=np.float32)
        if have.shape[1] < mine.size:
            def keep(part):
                np.save(R._tmp("share") + ".npy", part)
                os.replace(R._tmp("share") + ".npy", f)
            got = [have]
            for c0 in range(have.shape[1], mine.size, chunk):
                part, color, _ = lh.comb_products(apply, IJ, s, chunk=chunk, only=mine[c0:c0 + chunk])
                got.append(part)
                keep(np.hstack(got))
                R.say("local: share %d of %d: %d of %d products, %.1f s" % (K, N, min(c0 + chunk, mine.size), mine.size, time.time() - t0))
                if R.late():
                    R.say("local: stopping (time limit or STOP file)")
                    return None, dict(done=False)
        files = [R.path("local_products_s%d.%dof%d.npy" % (s, k, N)) for k in range(N)]
        sizes = [np.arange(k, s * s, N).size for k in range(N)]
        if not all(os.path.exists(g) and np.load(g, mmap_mode="r").shape[1] == m for g, m in zip(files, sizes)):
            R.say("local: share %d of %d is saved; the others are not all there yet" % (K, N))
            return None, dict(done=False)
        HM = np.zeros((n, s * s), dtype=np.float32)
        for k, g in enumerate(files):
            HM[:, k::N] = np.load(g)
        color = (IJ[:, 0] % s) * s + (IJ[:, 1] % s)
        np.save(saved, HM)
    else:
        HM, color, _ = lh.comb_products(apply, IJ, s, chunk=chunk, log=R.say)
        np.save(saved, HM)
    t1 = time.time()
    Hs = lh.sparse_from_combs(IJ, HM, color, s, step, a.reach)
    del HM
    # against the Hessian itself: white vectors (every scale) and samples of the prior (the scales that matter)
    hm.parRandom.set_seed(53)
    V = np.random.default_rng(53).standard_normal((n, 8))
    v = pb.Vm.vector()
    for k in range(4, 8):
        prior.sample(prior.sample_noise(1.0), v, add_mean=False)
        V[:, k] = v.array
    HV = apply(V)
    err = np.linalg.norm(Hs @ V - HV, axis=0) / np.linalg.norm(HV, axis=0)
    sp.save_npz(R.path(name + ".npz"), Hs)
    rec = dict(done=True, reach=a.reach, spacing=int(s), step_km=float(step), products=int(s * s), check_products=8,
               entries_per_row=float(Hs.nnz / n), seconds=time.time() - t0, seconds_products=t1 - t0,
               action_error_white=float(err[:4].mean()), action_error_prior=float(err[4:].mean()))
    R.dump(name + ".json", rec)
    R.say("local: %d products with combs %d nodes (%g km) apart in %.1f s; entries within %g km, %.0f a row; "
          "against the Hessian's own products: %.1f %% on white vectors, %.1f %% on prior samples"
          % (s * s, s, s * step, t1 - t0, a.reach, rec["entries_per_row"], 100 * rec["action_error_white"],
             100 * rec["action_error_prior"]))
    return Hs, rec


def stage_post_local(R, model, prior, x, Hs):
    """Samples and pointwise variances of the Laplace posterior with the sparse Hessian."""
    import hippymfem as hm
    import localhessian as lh
    from hippymfem.common.linalg import hypre_to_scipy
    from hippymfem.modeling.variables import PARAMETER

    a, pb = R.a, R.pb
    t0 = time.time()
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    post = lh.LocalPosterior(Hs, hypre_to_scipy(prior.A).tocsr(), hypre_to_scipy(prior.M).tocsr(), pb, model.misfit, prior,
                             x[PARAMETER], R.comm, precision=a.factors, method=a.local_solver,
                             draws=a.draws if a.draws != "auto" else ("adjoint" if hasattr(pb.solver, "solve_many") else "lanczos"))
    t1 = time.time()
    n = Hs.shape[0]
    hm.parRandom.set_seed(37)
    acc = np.zeros((2, n))
    batch = max(8, min(128, int(6.0e7 // n)))
    if post.draws == "lanczos":
        batch = min(batch, 48)                         # sixty Lanczos vectors a column are kept on the card
    for k0 in range(0, a.var_samples, batch):
        Sp, So = post.draw(min(batch, a.var_samples - k0))
        acc += np.stack(((Sp ** 2).sum(axis=1), (So ** 2).sum(axis=1)))
    var = [pb.Vm.vector(), pb.Vm.vector()]
    for v, s, name in zip(var, acc, ("var_prior.npy", "var_post.npy")):
        v.array[:] = s / a.var_samples
        R.save_footprint(name, v)
    # data cannot widen a Gaussian: a variance above the prior's (the same draws give both, so
    # 20 % is far outside their noise) is the sparse matrix's doing
    ratio = acc[1] / np.maximum(acc[0], 1e-300)
    above = dict(above_prior=float((ratio > 1.2).mean()), above_prior_max=float(ratio.max()))
    if above["above_prior"] > 0:
        R.say("post: the variance is above the prior's at %.2f %% of the nodes, up to %.3g times: the sparse Hessian is "
              "not positive there; take a longer reach or a finer grid" % (100 * above["above_prior"], above["above_prior_max"]))
    hm.parRandom.set_seed(7)
    sp_, so = pb.Vm.vector(), pb.Vm.vector()
    if a.nsamples:
        Sp, So = post.draw(a.nsamples)
        for i in range(a.nsamples):
            sp_.array[:] = Sp[:, i] + prior.mean.array
            so.array[:] = So[:, i] + x[PARAMETER].array
            R.save_footprint("sample_prior_%d.npy" % i, sp_)
            R.save_footprint("sample_post_%d.npy" % i, so)
    rec = dict(seconds=time.time() - t0, seconds_factorization=t1 - t0, hessian="local", reach=a.reach,
               std_prior=float(np.sqrt(var[0].sum() / var[0].global_size)),
               std_post=float(np.sqrt(var[1].sum() / var[1].global_size)), **above, **post.describe())
    R.say("post: std of beta %.3f -> %.3f (rms over the nodes) with the sparse Hessian; factorization %.1f s, %d draws %.1f s%s"
          % (rec["std_prior"], rec["std_post"], t1 - t0, a.var_samples + a.nsamples, time.time() - t1,
             "; conjugate gradients, %.1f iterations a block (%d at most)" % (rec["cg_iterations_mean"], rec["cg_iterations_max"])
             if "cg_iterations_mean" in rec else ""))
    return post, rec


def stage_post(R, model, prior, x, d, U):
    import hippymfem as hm
    from hippymfem.modeling.variables import PARAMETER

    a, pb = R.a, R.pb
    t0 = time.time()
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])
    # the variances as mean squares of samples.  (The library's estimator, the prior's
    # variance from samples minus the exact low-rank correction, is noisy where the data
    # shrink the variance most.)
    hm.parRandom.set_seed(37)
    acc = np.zeros((2, pb.Vm.vector().local_size))
    noise = prior.noise_vector()
    sp, so = pb.Vm.vector(), pb.Vm.vector()
    for _ in range(a.var_samples):
        prior.sample_noise(1.0, noise)
        post.sample(noise, sp, so, add_mean=False)
        acc += np.stack((sp.array ** 2, so.array ** 2))
    var = [pb.Vm.vector(), pb.Vm.vector()]
    for v, s, name in zip(var, acc, ("var_prior.npy", "var_post.npy")):
        v.array[:] = s / a.var_samples
        R.save_footprint(name, v)
    hm.parRandom.set_seed(7)
    for i in range(a.nsamples):
        post.sample(prior.sample_noise(1.0), sp, so, add_mean=True)
        R.save_footprint("sample_prior_%d.npy" % i, sp)
        R.save_footprint("sample_post_%d.npy" % i, so)
    rec = dict(seconds=time.time() - t0, std_prior=float(np.sqrt(var[0].sum() / var[0].global_size)),
               std_post=float(np.sqrt(var[1].sum() / var[1].global_size)))
    R.say("post: std of beta %.3f -> %.3f (rms over the nodes), %.1f s" % (rec["std_prior"], rec["std_post"], rec["seconds"]))
    return post, rec


def stage_qoi(R, model, prior, x, post, z):
    """The three fluxes Q(m) with their linearized posterior std: g = dQ/dm by one adjoint
    solve each, var = g^T Gamma_post g."""
    import stokes as st
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    pb = R.pb
    t0 = time.time()
    # the adjoint of Q: J^T v = -dQ/du, then g = C^T v; every region reuses the
    # factorization of the Jacobian at the MAP point
    adj = {}
    for r in st.REGIONS:
        rhs = pb.generate_state()
        rhs.array[:pb.nv] = -st.RHO_ICE * pb.flux_vector(r).array
        adj[r] = pb.generate_state()
        pb.solveAdj(adj[r], x, rhs)
    pb.setLinearizationPoint(x, True)
    rec = dict(regions={})
    for r in st.REGIONS:
        g, Hg, Hp = pb.Vm.vector(), pb.Vm.vector(), pb.Vm.vector()
        pb.apply_ij(PARAMETER, ADJOINT, adj[r], g)
        post.Hlr.solve(Hg, g)
        prior.Rsolver.solve(Hp, g)
        var = max(g.inner(Hg), 0.0)
        rec["regions"][r] = dict(flux_map=pb.mass_flux(x[STATE], r), flux_std_post=float(np.sqrt(var)),
                                 flux_std_prior=float(np.sqrt(max(g.inner(Hp), 0.0))))
        if "flux_true_" + r in z:
            rec["regions"][r]["flux_true"] = float(z["flux_true_" + r])
        R.save_footprint("qoi_gradient_%s.npy" % r, g)
        # the paper's inference-for-prediction direction (its Figure 10, after Lieberman
        # and Willcox): Gamma_post g / sqrt(g^T Gamma_post g), the one parameter mode that
        # the data inform and the prediction depends on
        Hg.scale(1.0 / max(np.sqrt(var), 1e-300))
        R.save_footprint("qoi_ifp_%s.npy" % r, Hg)
    rec["seconds"] = time.time() - t0
    if R.a.qoi_samples:
        rec["sampled"] = sample_predictions(R, prior, x, post)
    for r in st.REGIONS:
        q = rec["regions"][r]
        smp = rec.get("sampled", {}).get(r)
        R.say("qoi: mass flux into the ocean, %-6s %8.2f Gt/a, linearized std %.2f Gt/a (prior %.2f Gt/a)%s%s"
              % (r, q["flux_map"], q["flux_std_post"], q["flux_std_prior"],
                 ", truth %.2f" % q["flux_true"] if "flux_true" in q else "",
                 "; over %d samples %.2f +- %.2f (std %.2f)" % (rec["sampled"]["n"], smp["mean"], smp["stderr"], smp["std"])
                 if smp else ""))
    return rec


def sample_predictions(R, prior, x, post):
    """The fluxes over posterior samples, each through the nonlinear forward problem: the
    check of the linearized prediction that the paper does not make.  Saved after every
    sample (``qoi_samples.json``), resumed, and stopped at the time limit.  With
    ``--qoi-share K/N`` the run takes the samples whose number is K modulo N and saves
    them in a file of its own (``qoi_samples.KofN.json``); the samples of every file in
    the folder count as done, and all of them go into the statistics."""
    import hippymfem as hm
    import stokes as st
    from hippymfem.modeling.variables import STATE

    pb, a = R.pb, R.a
    k, n = a.qoi_share
    name = "qoi_samples.json" if n == 1 else "qoi_samples.%dof%d.json" % (k, n)

    def everyones():
        # the samples in every file of the folder, by number: this run's and the others'
        out = {}
        if R.rank0:
            for g in sorted(glob.glob(R.path("qoi_samples*.json"))):
                for d in json.load(open(g)):
                    out.setdefault(d["sample"], d)
        return R.comm.bcast(out)

    done = R.comm.bcast(json.load(open(R.path(name))) if R.rank0 and os.path.exists(R.path(name)) else [])
    hm.parRandom.set_seed(101)
    noise = prior.noise_vector()
    sp, so = pb.Vm.vector(), pb.Vm.vector()
    t_last = None
    for i in range(a.qoi_samples):
        prior.sample_noise(1.0, noise)                       # the stream, whether or not done
        post.sample(noise, sp, so, add_mean=True)
        if i % n != k or i in everyones():
            continue
        if R.late(ahead=1.5 * t_last if t_last else 0.0):
            R.say("qoi: stopping the samples (time limit or STOP file)")
            break
        t0 = time.time()
        xs = [pb.generate_state(), so, None]
        pb.start_from(x)                                     # from the MAP point and its state
        try:
            pb.solveFwd(xs[STATE], xs)
        except RuntimeError as e:
            R.say("qoi: sample %d: %s" % (i, e))
            done.append(dict(sample=i, failed=str(e)))       # not tried again by a later run
            R.dump(name, done)
            continue
        done.append(dict(sample=i, newton=pb.last_forward["newton"], seconds=time.time() - t0,
                         **{r: pb.mass_flux(xs[STATE], r) for r in st.REGIONS}))
        t_last = time.time() - t0
        R.dump(name, done)
        R.say("qoi: sample %d: %s (%d Newton steps, %.0f s)" % (i, ", ".join("%s %.1f" % (r, done[-1][r]) for r in st.REGIONS),
                                                               done[-1]["newton"], t_last))
    pb.start_from(x)
    done = [d for _, d in sorted(everyones().items()) if "failed" not in d]
    out = dict(n=len(done))
    for r in st.REGIONS:
        v = np.array([d[r] for d in done])
        if len(v) > 1:
            out[r] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), stderr=float(v.std(ddof=1) / np.sqrt(len(v))))
    return out


#: options that say how a run proceeds, not what problem it solves: the others must be the
#: same when a folder is taken up again
RUN_ONLY = ("out", "stages", "time_limit", "solver", "factors", "init", "max_newton", "cg_max", "rtol", "g_ref", "block",
            "max_cols", "var_samples", "nsamples", "qoi_samples", "qoi_share", "hessian", "reach", "local_solver",
            "krylov_rtol", "map_solver", "local_from", "local_share", "draws", "forward_rtol")


def stop_requested(out):
    """A file named STOP in the output directory asks the run to end: before it builds
    anything, or after the Gauss-Newton iteration under way (the iterate is saved)."""
    return os.path.exists(os.path.join(out, "STOP"))


def main(argv=None):
    a = parse(argv)
    if stop_requested(a.out):
        print("icesheet: %s exists; remove it to run" % os.path.join(a.out, "STOP"), flush=True)
        return 0
    # a folder holds one problem: what an earlier run saved there is taken up again
    f = os.path.join(a.out, "args.json")
    was = json.load(open(f)) if os.path.exists(f) else {}
    diff = ["%s %r, now %r" % (k, was[k], v) for k, v in sorted(vars(a).items())
            if k not in RUN_ONLY and k in was and was[k] != v]
    if diff:
        print("icesheet: %s holds a run with other settings (%s); give the same ones or another --out"
              % (a.out, ", ".join(diff)), flush=True)
        return 1
    R = Run(a)
    R.dump("args.json", vars(a))
    stages = a.stages.split(",")
    z = stage_data(R)
    model, prior = make_model(R, z)
    summary = dict(args=vars(a), sizes=R.sizes, prior=R.prior)
    if "verify" in stages:
        summary["verify"] = stage_verify(R, model, prior)
    x = d = U = post = None
    if any(s in stages for s in ("map", "eig", "post", "qoi")):
        x, summary["map"] = (stage_map_local if a.map_solver == "local" else stage_map)(R, model, prior, z)
    if x is not None and a.hessian == "local" and any(s in stages for s in ("eig", "post", "qoi")):
        Hs, summary["local"] = stage_local(R, model, prior, x)
        if Hs is not None and any(s in stages for s in ("post", "qoi")):
            post, summary["post"] = stage_post_local(R, model, prior, x, Hs)
    elif x is not None and any(s in stages for s in ("eig", "post", "qoi")):
        d, U, summary["eig"] = stage_eig(R, model, prior, x)
    if d is not None and any(s in stages for s in ("post", "qoi")):
        post, summary["post"] = stage_post(R, model, prior, x, d, U)
    if post is not None and "qoi" in stages:
        summary["qoi"] = stage_qoi(R, model, prior, x, post, z)
    summary["counters"] = dict(R.pb.counters)
    summary["solver"] = R.pb.solver.describe() if hasattr(R.pb.solver, "describe") else None
    summary["seconds"] = time.time() - R.t0
    R.dump("summary.json", summary)
    R.say("icesheet: done in %.1f s" % summary["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

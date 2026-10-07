#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What the animation of ``movie.py`` shows beyond a ``run.py --dump`` file.

A dump holds the truth, the MAP point and the standard deviations.  The animation also
shows the heat flowing through the true rock and the posterior in motion, which need the
true temperature and samples of the posterior.  Two commands make them, both from the
same launcher and with the same environment as ``run.py``:

    mpirun -n 4 python -m applications.geothermal.movie_data fields --dump results/geothermal_n128.npz \\
        --gauss-newton --out results/fields_n128.npz

rebuilds the problem of the dump, takes the MAP point from it (no Newton iteration is
repeated), computes the eigenpairs at it as ``run.py`` does (give ``--gauss-newton`` if
the run had it; ``--k`` is by default the number of eigenvalues in the dump), and writes
the truth, the MAP point, the prior and the posterior standard deviation, the true
temperature with the conductivity it flows through, and pairs of samples of the prior
and of the posterior made from the same noise.

    mpirun -n 64 python -m applications.geothermal.movie_data forward --n 512 --onto 128 \\
        --out results/forward_n512.npz

solves the forward problem alone on a finer mesh than an inversion can afford.  The
truth is a function of the point, so that mesh holds the rock of every inversion; its
temperature is kept at the vertices of the ``--onto`` mesh (the inversion's), where
``movie.py --temperature`` puts it in the place of the inversion's own.

Every field is on the vertex grid ``(n + 1)^3`` of the Cartesian mesh, as 32-bit floats
in the order x fastest, then y, then z.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402
from applications.geothermal.model import Geothermal, K_REF, T_SCALE, lithology_index_np   # noqa: E402

COMM = MPI.COMM_WORLD


def say(*a):
    if COMM.rank == 0:
        print(*a, flush=True)


def grid_index(xyz, n):
    """The place of each vertex ``xyz`` on the grid of spacing ``1/n`` (x fastest, then y, z)."""
    ijk = np.rint(np.asarray(xyz) * n).astype(np.int64)
    return (ijk[:, 2] * (n + 1) + ijk[:, 1]) * (n + 1) + ijk[:, 0]


_PLACES = {}


def on_vertices(space, values, n):
    """The nodal field ``values`` of ``space`` at the vertices of the grid of spacing
    ``1/n``, as one array on rank 0 (None elsewhere).  ``n`` is the mesh's own size or
    divides it: of a quadratic field, and of a field on a finer mesh, the nodes between
    those vertices are left out."""
    if (id(space), n) not in _PLACES:                     # the nodes of a space are asked for once
        xyz = space.coordinates()
        at = np.all(np.abs(xyz * n - np.rint(xyz * n)) < 1e-6, axis=1)
        _PLACES[id(space), n] = at, grid_index(xyz[at], n)
    at, idx = _PLACES[id(space), n]
    I = COMM.gather(idx, root=0)
    V = COMM.gather(np.asarray(values)[at].astype(np.float32), root=0)
    if COMM.rank:
        return None
    out = np.full((n + 1) ** 3, np.nan, np.float32)
    out[np.concatenate(I)] = np.concatenate(V)
    if np.isnan(out).any():
        raise RuntimeError("the nodes of the space do not cover the grid of spacing 1/%d" % n)
    return out


def conductivity(G, m, u, n):
    """``k(m, T, x)`` of the model in W/(m K) on the vertex grid, from ``m`` and the
    temperature ``u`` there: the heat flows along ``-k grad T``."""
    g = np.linspace(0.0, 1.0, n + 1)
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    li = lithology_index_np(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))
    w = 1.2 * (G.nodes[1] - G.nodes[0])                   # the kernel of model.make_densities
    out = np.empty(u.size, np.float32)
    for a in range(0, u.size, 1 << 20):
        s = slice(a, a + (1 << 20))
        wts = np.exp(-0.5 * ((T_SCALE * u[s, None] - G.nodes[None, :]) / w) ** 2)
        out[s] = K_REF * np.exp(m[s]) * np.sum(wts * G.tabs[li[s]], axis=1) / np.sum(wts, axis=1)
    return out


def save(path, **arrays):
    if COMM.rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, T_SCALE=T_SCALE, **arrays)


def fields(args):
    """The fields of the animation for the inversion of a dump."""
    z = np.load(args.dump)
    n = int(z["n"])
    hm.configure_device(args.device, COMM, quiet=(COMM.rank != 0))
    t0 = time.perf_counter()
    G = Geothermal(n, COMM, order=args.order, nboreholes=len(np.unique(np.round(z["targets"][:, :2], 9), axis=0)),
                   noise_kelvin=args.noise_kelvin)
    model, prior, pde = G.model, G.prior, G.pde
    for attr in ("solver_fwd_inc", "solver_adj_inc"):
        getattr(pde, attr).parameters["rel_tolerance"] = args.inc_tol
    say("geothermal %d^3 on %d ranks, built in %.1f s" % (n, COMM.size, time.perf_counter() - t0))

    # ---- the problem is the dump's, and its MAP point is taken from it
    mine = grid_index(G.Vm.coordinates(), n)

    def of_dump(key):
        grid = np.full((n + 1) ** 3, np.nan)
        grid[grid_index(z["xyz"], n)] = z[key]
        return grid[mine]
    truth = COMM.allreduce(float(np.max(np.abs(of_dump("mtrue") - G.mtrue.array), initial=0.0)), op=MPI.MAX)
    data = float(np.max(np.abs(z["data"] - G.B.gather(G.data)))) if G.targets.shape == z["targets"].shape else np.inf
    say("  against the dump: the truth differs by %.1e, the data by %.1e" % (truth, data))
    if truth > 1e-6 or data > 1e-6:
        raise SystemExit("%s was made for another problem (another version of model.py, other boreholes or noise)" % args.dump)
    x = model.generate_vector()
    x[PARAMETER].array[:] = of_dump("mmap")
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)
    g = model.evalGradientParameter(x, model.generate_vector(PARAMETER))
    x0 = model.generate_vector()
    x0[PARAMETER].assign(prior.mean)
    model.solveFwd(x0[STATE], x0)
    model.solveAdj(x0[ADJOINT], x0)
    g0 = model.evalGradientParameter(x0, model.generate_vector(PARAMETER))
    say("  the MAP point of the dump: gradient %.3e, %.1e of that at the prior mean" % (g, g / g0))

    # ---- the eigenpairs, as run.py computes them
    k = args.k or int(z["d"].size)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=args.gauss_newton)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    Omega = hm.MultiVector(x[PARAMETER], k + args.p)
    hm.parRandom.set_seed(99)
    hm.parRandom.normal_multivector(1.0, Omega)
    t0 = time.perf_counter()
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k, s=1)
    m = min(k, int(z["d"].size))
    say("  eigensolver: %d pairs in %.1f s, %.3e .. %.3e, %d above 1; against the dump's the first %d differ by %.1e (relative)"
        % (k, time.perf_counter() - t0, d[0], d[-1], int((d > 1).sum()), m, float(np.max(np.abs(d[:m] / z["d"][:m] - 1.0)))))
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])

    # ---- the standard deviations, and pairs of samples from the same noise, without their means
    t0 = time.perf_counter()
    hm.parRandom.set_seed(31)
    pv, prv, _ = post.pointwise_variance(method="MonteCarlo", n=args.var_samples)
    grid = lambda space, values: on_vertices(space, values, n)
    noise, s_pr, s_po = prior.noise_vector(), G.Vm.vector(), G.Vm.vector()
    S_pr, S_po = [], []
    for i in range(args.samples):
        hm.parRandom.set_seed(700 + i)
        prior.sample_noise(1.0, noise)
        post.sample(noise, s_pr, s_po, add_mean=False)
        S_pr.append(grid(G.Vm, s_pr.array))
        S_po.append(grid(G.Vm, s_po.array))
    out = dict(mtrue=grid(G.Vm, G.mtrue.array), mmap=grid(G.Vm, x[PARAMETER].array),
               std_prior=grid(G.Vm, np.sqrt(np.maximum(prv.array, 0.0))), std_post=grid(G.Vm, np.sqrt(np.maximum(pv.array, 0.0))),
               u_true=grid(G.Vu, G.utrue.array), u_map=grid(G.Vu, x[STATE].array))
    if COMM.rank == 0:
        save(args.out, n=n, s_prior=np.array(S_pr), s_post=np.array(S_po), d=np.asarray(d), targets=G.targets,
             k_true=conductivity(G, out["mtrue"], out["u_true"], n), forward_n=n, forward_dofs=int(G.Vu.GlobalTrueVSize()), **out)
    say("  variance from %d samples and %d pairs of samples in %.1f s; wrote %s"
        % (args.var_samples, args.samples, time.perf_counter() - t0, args.out))
    return 0


def forward(args):
    """The forward problem on the ``--n`` mesh; its temperature on the ``--onto`` grid."""
    if args.n % args.onto:
        raise SystemExit("--onto %d does not divide --n %d" % (args.onto, args.n))
    hm.configure_device(args.device, COMM, quiet=(COMM.rank != 0))
    t0 = time.perf_counter()
    G = Geothermal(args.n, COMM, order=args.order)
    seconds = time.perf_counter() - t0
    info = G.summary()
    mtrue = on_vertices(G.Vm, G.mtrue.array, args.onto)
    u_true = on_vertices(G.Vu, G.utrue.array, args.onto)
    if COMM.rank == 0:
        save(args.out, n=args.onto, mtrue=mtrue, u_true=u_true, k_true=conductivity(G, mtrue, u_true, args.onto),
             forward_n=args.n, forward_dofs=info["state_dofs"], ranks=COMM.size, seconds=seconds,
             newton_iterations=info["forward_newton_iterations"])
    say("geothermal %d^3: forward solve with %d state dofs on %d ranks, %d Newton iterations, T_max %.0f K; built and "
        "solved in %.1f s; wrote %s (the temperature on the %d^3 grid)"
        % (args.n, info["state_dofs"], COMM.size, info["forward_newton_iterations"], info["u_true_max_kelvin"],
           seconds, args.out, args.onto))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    device = dict(default=("cuda" if hm.config.hypre_device else "cpu"), help="MFEM device; the default follows HIPPYMFEM_HYPRE_DEVICE")
    f = sub.add_parser("fields", help="the fields of the animation for the inversion of a run.py dump")
    f.add_argument("--dump", required=True, help="the file of run.py --dump")
    f.add_argument("--out", required=True)
    f.add_argument("--k", type=int, default=0, help="eigenpairs (default: as many as the dump has eigenvalues)")
    f.add_argument("--p", type=int, default=20, help="oversampling, as run.py --p")
    f.add_argument("--gauss-newton", action="store_true", help="as run.py --gauss-newton: give it if the run had it")
    f.add_argument("--inc-tol", type=float, default=1e-6, help="as run.py --inc-tol")
    f.add_argument("--noise-kelvin", type=float, default=0.5, help="as run.py --noise-kelvin")
    f.add_argument("--samples", type=int, default=12, help="pairs of a prior and a posterior sample; the animation turns through them")
    f.add_argument("--var-samples", type=int, default=800,
                   help="Monte Carlo samples of the prior's pointwise variance: the posterior std has 2.5 %% of noise "
                        "with 800 and 9 %% with the 64 of a dump")
    f.add_argument("--order", type=int, default=2)
    f.add_argument("--device", **device)
    f.set_defaults(run=fields)
    f = sub.add_parser("forward", help="the forward problem alone, its temperature on a coarser grid")
    f.add_argument("--n", type=int, required=True, help="the mesh of the forward solve")
    f.add_argument("--onto", type=int, required=True, help="the mesh of the inversion that is drawn (divides --n)")
    f.add_argument("--order", type=int, default=2)
    f.add_argument("--device", **device)
    f.add_argument("--out", required=True)
    f.set_defaults(run=forward)
    args = ap.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())

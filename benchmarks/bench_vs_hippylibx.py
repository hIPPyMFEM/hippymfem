#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""What hIPPyMFEM can do that a form language cannot, measured rather than asserted.

The interesting comparison with a form-language library is not speed (dolfinx's
FFCx-generated C kernels are faster per element on a CPU, and the documentation says so).
It is **what can be written down at all**, and whether the derivatives stay exact when
it is.  Each experiment here is a residual density that has no UFL equivalent,
together with the evidence that the gradient and Hessian obtained from it are right:
finite-difference slopes of 1, a symmetric reduced Hessian, and an optimizer that
converges.

=====  =======================================================================
A      a neural-network closure inside the residual
B      an inner Newton solve for a local constitutive law, differentiated through
C      a non-smooth (regularized) law with a data-dependent branch
D      a residual depending on a field through a table lookup / interpolation
E      third derivatives of the residual, used by a Taylor-expanded QoI
F      the same density executed on a GPU, which compiled C kernels cannot be
=====  =======================================================================

Usage::

    python benchmarks/bench_vs_hippylibx.py --out results/pros.json
    HIPPYMFEM_DEVICE=gpu python benchmarks/bench_vs_hippylibx.py --gpu
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hp                                              # noqa: E402
from hippymfem.fem import kernel as K                               # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
RESULTS = []


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def record(name, **kw):
    kw["name"] = name
    RESULTS.append(kw)


# ------------------------------------------------------------------- the densities
def mlp_density(seed=0, width=16):
    """A two-layer network inside the residual, acting as a learned conductivity."""
    rng = np.random.default_rng(seed)
    W1 = jnp.asarray(rng.standard_normal((width, 2)) / np.sqrt(2.0))
    b1 = jnp.asarray(rng.standard_normal(width) * 0.1)
    W2 = jnp.asarray(rng.standard_normal((1, width)) / np.sqrt(width))
    b2 = jnp.asarray(rng.standard_normal(1) * 0.1)

    def varf(u, m, p, x):
        z = jnp.stack([u.val, m.val])
        h = jnp.tanh(W1 @ z + b1)
        kappa = jnp.exp((W2 @ h + b2)[0])
        return kappa * jnp.dot(u.grad, p.grad) - p.val

    return varf, "A: neural-network closure (2x%d tanh MLP in the conductivity)" % width


def local_newton_density(steps=3):
    """A local constitutive law solved by an unrolled Newton iteration."""
    def varf(u, m, p, x):
        target = jnp.exp(m.val) * (1.0 + u.val ** 2)

        def res(s):
            return s ** 3 + s - target

        s = target / 2.0
        for _ in range(steps):
            s = s - res(s) / (3.0 * s ** 2 + 1.0)
        return s * jnp.dot(u.grad, p.grad) - p.val

    return varf, "B: %d-step inner Newton solve for a local law" % steps


def regularized_branch_density(eps=1e-2):
    """A non-smooth law, regularized, with a branch on the field value.

    ``jnp.where`` keeps it differentiable as a program; a form language has no way
    to express the branch at all.
    """
    def varf(u, m, p, x):
        g2 = jnp.dot(u.grad, u.grad)
        # a Carreau-like shear-thinning viscosity with a regularized cutoff
        nu = jnp.where(g2 > eps ** 2,
                       jnp.exp(m.val) * (g2 + eps ** 2) ** (-0.25),
                       jnp.exp(m.val) * eps ** (-0.5))
        return nu * jnp.dot(u.grad, p.grad) - p.val

    return varf, "C: regularized non-smooth law with a branch (jnp.where)"


def table_density(n=33, seed=1):
    """A coefficient read from a table by **linear** interpolation in the parameter.

    Included because it is instructive rather than because it works: a piecewise
    linear interpolant is continuous but not continuously differentiable, so its
    second derivative is zero almost everywhere and a delta at the knots.  The
    gradient is right and the Hessian is not, and the finite-difference test says so.
    """
    rng = np.random.default_rng(seed)
    grid = jnp.asarray(np.linspace(-3.0, 3.0, n))
    table = jnp.asarray(np.exp(0.5 * np.linspace(-3.0, 3.0, n))
                        + 0.1 * rng.standard_normal(n) ** 2)

    def varf(u, m, p, x):
        kappa = jnp.interp(m.val, grid, table)
        return kappa * jnp.dot(u.grad, p.grad) - p.val

    return varf, "D1: tabulated coefficient, LINEAR interpolation (only C0)"


def smooth_table_density(n=33, seed=1, width=None):
    r"""The same table, read with a $C^\infty$ kernel so the Hessian is meaningful.

    A Gaussian radial-basis expansion over the same knots: smooth, so the second
    derivative exists, and still a table lookup that no form language can express.
    """
    rng = np.random.default_rng(seed)
    nodes = np.linspace(-3.0, 3.0, n)
    h = nodes[1] - nodes[0]
    w = width if width is not None else 1.2 * h
    vals = np.exp(0.5 * nodes) + 0.1 * rng.standard_normal(n) ** 2
    nodes_j = jnp.asarray(nodes)
    vals_j = jnp.asarray(vals)

    def varf(u, m, p, x):
        wts = jnp.exp(-0.5 * ((m.val - nodes_j) / w) ** 2)
        kappa = jnp.sum(wts * vals_j) / jnp.sum(wts)
        return kappa * jnp.dot(u.grad, p.grad) - p.val

    return varf, "D2: same table, smooth (Gaussian-kernel) interpolation"


def reference_density():
    """The textbook density, for the cost comparison."""
    def varf(u, m, p, x):
        return jnp.exp(m.val) * jnp.dot(u.grad, p.grad) - p.val

    return varf, "reference: exp(m) grad u . grad p (expressible in UFL)"


# ------------------------------------------------------------------------ harness
def build(varf, n=16, order=2, ntargets=50, seed=3):
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(
        n, n, mfem.Element.TRIANGLE))
    Vu = hp.FunctionSpace.H1(pm, order)
    Vm = hp.FunctionSpace.H1(pm, 1)
    bc = hp.DirichletBC(Vu, lambda z: z[1], bdr_attributes=[1, 3])
    pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(),
                                   is_fwd_linear=False)
    pde.newton_parameters["max_iter"] = 40
    # An exact solve where the problem is small enough for one, so that the
    # gradient and Hessian checks below measure the derivatives and not a Krylov
    # tolerance; iterative once it is not, so that --n can be raised.
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        s = hp.auto_solver(Vu, COMM, method="gmres")
        if isinstance(s, hp.KrylovSolver):
            s.parameters["rel_tolerance"] = 1e-12
            s.parameters["max_iter"] = 3000
        setattr(pde, a, s)
    prior = hp.BiLaplacianPrior(
        Vm, 0.2, 1.0, robin_bc=True,
        solver_type="krylov" if isinstance(pde.solver, hp.KrylovSolver) else "lu")
    rng = np.random.default_rng(seed)
    targets = np.column_stack((rng.uniform(0.1, 0.9, ntargets),
                               rng.uniform(0.1, 0.9, ntargets)))
    B = hp.assemblePointwiseObservation(Vu, targets)
    mtrue = Vm.project(lambda z: 0.5 * np.sin(2.0 * z[0]) * np.cos(2.0 * z[1]))
    ut = pde.generate_state()
    pde.solveFwd(ut, [ut, mtrue, None])
    data = B.createVecLeft()
    B.mult(ut, data)
    std = 0.01 * max(data.norm("linf"), 1e-30)
    hp.parRandom.set_seed(seed)
    B.perturb(data, std)
    misfit = hp.DiscreteStateObservation(B, data, std ** 2)
    return pm, Vu, Vm, pde, hp.Model(pde, prior, misfit), prior, mtrue


def verify(label, varf, n=16, order=2):
    """Finite-difference slopes, Hessian symmetry, and whether Newton-CG converges."""
    pm, Vu, Vm, pde, model, prior, mtrue = build(varf, n, order)
    m0 = prior.mean.copy()
    hp.parRandom.set_seed(11)
    hp.parRandom.normal(0.2, m0)
    eps, eg, eh = hp.modelVerify(model, m0, verbose=False,
                                 eps=np.logspace(-1, -7, 13))
    gs = hp.best_slope(eps, eg, lo=1e-6, hi=1e-1)
    hs = hp.best_slope(eps, eh, lo=1e-4, hi=1e-1)

    x = model.generate_vector()
    x[PARAMETER] = m0.copy()
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    H = hp.ReducedHessian(model, misfit_only=True)
    a = model.generate_vector(PARAMETER)
    b = model.generate_vector(PARAMETER)
    hp.parRandom.normal(1.0, a)
    hp.parRandom.normal(1.0, b)
    Ha, Hb = model.generate_vector(PARAMETER), model.generate_vector(PARAMETER)
    H.mult(a, Ha)
    H.mult(b, Hb)
    sym = abs(b.inner(Ha) - a.inner(Hb)) / max(abs(b.inner(Ha)), 1e-300)

    p = hp.ReducedSpaceNewtonCG_ParameterList()
    p["rel_tolerance"] = 1e-8
    p["max_iter"] = 40
    p["print_level"] = -1
    solver = hp.ReducedSpaceNewtonCG(model, p)
    t0 = time.perf_counter()
    xm = solver.solve([None, prior.mean.copy(), None])
    wall = time.perf_counter() - t0
    err = xm[PARAMETER].copy().axpy(-1.0, mtrue).norm("l2") / max(
        mtrue.norm("l2"), 1e-300)

    say("  %-62s grad %.4f  hess %.4f  sym %.1e  %2d its %6.2fs  err %.3f"
        % (label, gs, hs, sym, solver.it, wall, err))
    record(label, grad_slope=gs, hess_slope=hs, hess_symmetry=sym,
           newton_its=solver.it, converged=bool(solver.converged),
           wall=wall, param_error=err, tdofs=Vu.GlobalTrueVSize(),
           nelem=pm.GetNE() * NP)
    ok = abs(gs - 1.0) < 0.12 and abs(hs - 1.0) < 0.15 and sym < 1e-8
    return ok


def test_expressiveness(n, order):
    say("\n=== Densities with no UFL equivalent: are the derivatives still exact? ===")
    say("  %-62s %-12s %-12s %-10s" % ("density", "grad slope", "hess slope",
                                       "symmetry"))
    allok = True
    expect_bad_hessian = {"D1"}
    for maker in (reference_density, mlp_density, local_newton_density,
                  regularized_branch_density, table_density,
                  smooth_table_density):
        varf, label = maker()
        ok = verify(label, varf, n, order)
        tag = label.split(":", 1)[0]
        if tag in expect_bad_hessian:
            # A C0 interpolant has no meaningful second derivative; the gradient
            # slope must still be 1, and the Hessian slope must NOT be, which is the
            # point of including it.
            r = RESULTS[-1]
            ok = abs(r["grad_slope"] - 1.0) < 0.12 and abs(
                r["hess_slope"] - 1.0) > 0.2
            say("      (expected: gradient exact, Hessian not -- a C0 interpolant "
                "has no second derivative)")
        allok &= ok
    say("  every case behaved as expected: %s" % allok)
    return allok


def test_third_derivatives(n, order):
    """Third derivatives of a non-UFL density, through a Taylor-expanded QoI.

    A second-order Taylor expansion of a quantity of interest needs the third
    derivative of the residual.  Here the residual contains an inner Newton solve, so
    the third derivative is a third derivative of that iteration -- which is
    available because the density is a program, not a form.
    """
    say("\n=== Third derivatives of a density containing an inner solve ===")
    varf, label = local_newton_density()
    pm, Vu, Vm, pde, model, prior, mtrue = build(varf, n, order)
    m0 = prior.mean.copy()
    hp.parRandom.set_seed(5)
    hp.parRandom.normal(0.2, m0)

    x = model.generate_vector()
    x[PARAMETER] = m0.copy()
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)

    jdir = Vu.vector()
    kdir = Vu.vector()
    hp.parRandom.normal(1.0, jdir)
    hp.parRandom.normal(1.0, kdir)
    # The two paths treat essential dofs differently on purpose: W_uu has its
    # essential rows *and columns* eliminated, while the third-derivative vector has
    # only its rows zeroed.  Directions with no essential component make the
    # comparison about the derivative rather than about that convention.
    pde.bc0.zero(jdir)
    pde.bc0.zero(kdir)
    out = Vu.vector()
    pde.apply_ijk(STATE, STATE, STATE, x, jdir, kdir, out)
    third = out.norm("l2")

    # A central difference of W_uu in the direction kdir, contracted with jdir.
    # apply_ij uses the blocks assembled at the stored linearization point, so the
    # point is moved by re-calling setLinearizationPoint rather than passed in.
    def Wuu_times(state, d):
        pde.setLinearizationPoint([state, x[PARAMETER], x[ADJOINT]],
                                  gauss_newton_approx=False)
        res = Vu.vector()
        pde.apply_ij(STATE, STATE, d, res)
        return res

    epsv = 1e-5
    up = x[STATE].copy().axpy(epsv, kdir)
    um = x[STATE].copy().axpy(-epsv, kdir)
    fd = Wuu_times(up, jdir)
    fd.axpy(-1.0, Wuu_times(um, jdir))
    fd.scale(0.5 / epsv)
    pde.bc0.zero(fd)
    pde.bc0.zero(out)
    d = fd.copy()
    d.axpy(-1.0, out)
    rel = d.norm("l2") / max(fd.norm("l2"), 1e-300)
    say("  third derivative vs a central difference of the second: rel %.3e "
        "(||T|| = %.4e, eps = %.0e)" % (rel, third, epsv))
    record("E: third derivative of an inner-solve density",
           third_norm=third, fd_rel=rel, fd_eps=epsv)
    return rel < 1e-4


def test_cost_of_expressiveness(n, order, gpu):
    """What the non-UFL densities cost, against the textbook one and on a GPU."""
    say("\n=== What the freedom costs: assembly time per element ===")
    from hippymfem.fem import assemble as asm
    from hippymfem.fem.elementbatch import MeshBatches
    from hippymfem.fem.kernel import QuadratureKernel

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(
        96, 96, mfem.Element.QUADRILATERAL))
    Vu = hp.FunctionSpace.H1(pm, 2)
    Vm = hp.FunctionSpace.H1(pm, 1)
    batches = MeshBatches(pm, 6, COMM)
    NE = pm.GetNE()
    hp.parRandom.set_seed(2)
    uv, mv, pv = Vu.vector(), Vm.vector(), Vu.vector()
    hp.parRandom.normal(1.0, uv)
    hp.parRandom.normal(0.3, mv)
    hp.parRandom.normal(1.0, pv)
    loc = [Vu.local_values(uv), Vm.local_values(mv), Vu.local_values(pv)]
    bc = hp.DirichletBC(Vu, None, "all")

    devices = ["cpu"] + (["gpu"] if gpu else [])
    say("  %-56s %s" % ("density", "  ".join("%11s" % ("%s us/elem" % d)
                                             for d in devices)))
    ref = {}
    for maker in (reference_density, mlp_density, local_newton_density,
                  regularized_branch_density, smooth_table_density):
        varf, label = maker()
        kern = QuadratureKernel(varf, [Vu, Vm, Vu], batches)
        times = {}
        for dev in devices:
            old = K.device()
            K.set_device(dev)
            try:
                def once():
                    A = asm.assemble_matrix(
                        Vu, Vu, batches.groups,
                        kern.element_matrices(ADJOINT, STATE, loc), NE,
                        test_ess=bc.ess_tdof)
                    del A
                once()
                COMM.Barrier()
                t0 = time.perf_counter()
                for _ in range(3):
                    once()
                COMM.Barrier()
                times[dev] = (time.perf_counter() - t0) / 3
            finally:
                K._DEVICE = old
        say("  %-56s %s" % (label[:56], "  ".join(
            "%11.2f" % (1e6 * times[d] / NE) for d in devices)))
        rec = {"us_per_elem_%s" % d: 1e6 * times[d] / NE for d in devices}
        if gpu:
            rec["gpu_speedup"] = times["cpu"] / times["gpu"]
        record("F: assembly cost, " + label[:40], NE=NE, **rec)
        ref[label] = times
    base = ref[reference_density()[1]]["cpu"]
    say("  relative to the UFL-expressible reference: %s"
        % ", ".join("%.2fx" % (v["cpu"] / base) for v in ref.values()))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--gpu", action="store_true",
                    help="also time the densities on a GPU")
    ap.add_argument("--skip-cost", action="store_true")
    args = ap.parse_args()

    mfem.Hypre.Init()
    gpu = args.gpu and K.device().platform != "cpu" or (
        args.gpu and _gpu_available())
    say("hIPPyMFEM: what a differentiated program can express that a form cannot")
    say("  %d ranks, JAX device %s" % (NP, K.device()))

    ok = test_expressiveness(args.n, args.order)
    ok &= test_third_derivatives(args.n, args.order)
    if not args.skip_cost:
        ok &= test_cost_of_expressiveness(args.n, args.order, gpu)

    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"ranks": NP, "results": RESULTS}, open(args.out, "w"), indent=1)
        say("\nwrote %s" % args.out)
    say("\nall checks passed: %s" % ok)
    return 0 if ok else 1


def _gpu_available():
    try:
        return len(jax.devices("cuda")) > 0
    except RuntimeError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())

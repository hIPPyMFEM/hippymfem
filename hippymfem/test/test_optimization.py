# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Newton-CG, BFGS, randomized eigensolvers and the Laplace approximation.

Run with ``mpirun -n N python -m hippymfem.test.test_optimization``.
"""

import math

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hm
from hippymfem.common.operators import Operator, init_vector_like
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
FAILS = []


def check(name, ok, detail=""):
    if RANK == 0:
        print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


# ----------------------------------------------------------------- CG solver
class _DenseOp(Operator):
    """A small SPD operator from an explicit dense matrix, for exact references."""

    def __init__(self, A, comm):
        self.Adense = np.asarray(A, dtype=float)
        self.comm = comm
        n = self.Adense.shape[0]
        self.template = hm.ParVector(comm, n if comm.rank == 0 else 0)

    def init_vector(self, x, dim):
        return init_vector_like(x, self.template)

    def mult(self, x, y):
        full = np.concatenate(self.comm.allgather(x.array))
        r = self.Adense @ full
        lo, hi = y.owner_range
        y.array[:] = r[lo:hi]
        return y

    multTranspose = mult


class _Identity:
    def __init__(self, op):
        self.op = op

    def solve(self, x, b):
        x.assign(b)
        return 1

    def init_vector(self, x, dim):
        return self.op.init_vector(x, dim)


def test_cg_steihaug():
    if RANK == 0:
        print("CGSolverSteihaug")
    n = 30
    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    lam = np.linspace(1.0, 100.0, n)
    A = Q @ np.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    op = _DenseOp(A, COMM)

    b = op.generate_vector(0)
    hm.parRandom.set_seed(1)
    hm.parRandom.normal(1.0, b)
    x = op.generate_vector(0)

    solver = hm.CGSolverSteihaug(comm=COMM)
    solver.set_operator(op)
    solver.set_preconditioner(_Identity(op))
    solver.parameters["rel_tolerance"] = 1e-12
    solver.parameters["max_iter"] = 200
    solver.parameters["print_level"] = -1
    solver.solve(x, b)

    bfull = np.concatenate(COMM.allgather(b.array))
    exact = np.linalg.solve(A, bfull)
    got = np.concatenate(COMM.allgather(x.array))
    e = np.linalg.norm(got - exact) / np.linalg.norm(exact)
    check("CG solves an SPD system", e < 1e-9,
          "(rel %.2e in %d iters, reason %d)" % (e, solver.iter, solver.reasonid))
    check("CG reports convergence", solver.converged and solver.reasonid == 1)

    # indefinite operator must terminate with reason 2
    lam2 = lam.copy()
    lam2[0] = -5.0
    A2 = Q @ np.diag(lam2) @ Q.T
    op2 = _DenseOp(0.5 * (A2 + A2.T), COMM)
    s2 = hm.CGSolverSteihaug(comm=COMM)
    s2.set_operator(op2)
    s2.set_preconditioner(_Identity(op2))
    s2.parameters["print_level"] = -1
    s2.solve(op2.generate_vector(0), b)
    check("CG detects a negative direction", s2.reasonid == 2,
          "(reason %d)" % s2.reasonid)

    # trust region must stop on the boundary
    s3 = hm.CGSolverSteihaug(comm=COMM)
    s3.set_operator(op)
    s3.set_preconditioner(_Identity(op))
    s3.parameters["print_level"] = -1
    s3.parameters["rel_tolerance"] = 1e-14
    radius = 1e-3
    s3.set_TR(radius, op)
    xtr = op.generate_vector(0)
    s3.solve(xtr, b)
    Bx = op.generate_vector(0)
    op.mult(xtr, Bx)
    nrm = math.sqrt(max(Bx.inner(xtr), 0.0))
    check("trust region respected", s3.reasonid == 3 and nrm <= radius * (1 + 1e-8),
          "(reason %d, ||x||_B = %.3e <= %.1e)" % (s3.reasonid, nrm, radius))

    # negative curvature inside a trust region: Steihaug steps to the boundary along
    # the direction, at the first iteration (hIPPYlib took the whole direction, far
    # outside a small region) and at a later one (hIPPYlib stopped inside)
    def model_value(A_, xv):
        xf = np.concatenate(COMM.allgather(xv.array))
        return -float(bfull @ xf) + 0.5 * float(xf @ A_ @ xf)

    for label, lam_neg in (("the first iteration", lambda l: -np.abs(l) - 1.0),
                           ("a later iteration", None)):
        if lam_neg is None:
            # positive along b's main components, one negative eigenvalue that CG
            # meets after a few steps
            lam3 = lam.copy()
            lam3[-1] = -50.0
        else:
            lam3 = lam_neg(lam)
        A3 = Q @ np.diag(lam3) @ Q.T
        A3 = 0.5 * (A3 + A3.T)
        op3 = _DenseOp(A3, COMM)
        for radius in (1e-2, 1e2):
            s4 = hm.CGSolverSteihaug(comm=COMM)
            s4.set_operator(op3)
            s4.set_preconditioner(_Identity(op3))
            s4.parameters["print_level"] = -1
            s4.set_TR(radius, op)
            x4 = op3.generate_vector(0)
            s4.solve(x4, b)
            B4 = op.generate_vector(0)
            op.mult(x4, B4)
            nrm = math.sqrt(max(B4.inner(x4), 0.0))
            ok = (s4.reasonid in (2, 3) and abs(nrm - radius) <= 1e-8 * radius
                  and model_value(A3, x4) < 0.0)
            check("negative curvature at %s ends on the boundary (radius %g)"
                  % (label, radius), ok,
                  "(reason %d after %d its, ||x||_B = %.6e, m = %.3e)"
                  % (s4.reasonid, s4.iter, nrm, model_value(A3, x4)))
    # without a trust region the first direction is taken whole (hIPPYlib's rule)
    A5 = Q @ np.diag(-np.abs(lam) - 1.0) @ Q.T
    op5 = _DenseOp(0.5 * (A5 + A5.T), COMM)
    s5 = hm.CGSolverSteihaug(comm=COMM)
    s5.set_operator(op5)
    s5.set_preconditioner(_Identity(op5))
    s5.parameters["print_level"] = -1
    x5 = op5.generate_vector(0)
    s5.solve(x5, b)
    e5 = x5.copy().axpy(-1.0, b).norm("l2") / b.norm("l2")
    check("without a trust region, negative curvature at once returns the direction",
          s5.reasonid == 2 and e5 < 1e-15, "(reason %d, |x - b|/|b| %.1e)" % (s5.reasonid, e5))


def test_bfgs_operator():
    """The damped BFGS update needs ``H y``, and the two-loop recursion computing it
    used the output vector as its own work vector, so ``H0inv.solve`` got its input as
    its output: the default rescaled identity then returned 0, a pair that needed
    damping was damped to ``s = 0``, and the update raised.  After the fix the update
    damps as Powell's rule says, and ``H`` satisfies the secant equation."""
    if RANK == 0:
        print("BFGS operator: damping and the secant equation")
    n = 6
    op = hm.BFGS_operator()
    H0 = hm.RescaledIdentity()
    H0.d0 = 2.0
    op.set_H0inv(H0)
    rng = np.random.default_rng(7)

    def vec(values):
        v = hm.ParVector(COMM, n if RANK == 0 else 0)
        if RANK == 0:
            v.array[:] = values
        return v

    s = vec(np.eye(n)[0])
    y = vec(np.r_[-0.5, 1.0, np.zeros(n - 2)])       # s^T y < 0: must be damped
    yHy = 2.0 * y.inner(y)
    try:
        theta = op.update(s.copy(), y.copy())
        sy = 1.0 / op.R[-1]
        want = (1.0 - 0.2) * yHy / (yHy - s.inner(y))
        ok = abs(theta - want) <= 1e-14 and abs(sy - 0.2 * yHy) <= 1e-14 * yHy
        detail = "(theta %.6f, want %.6f; s^T y %.6f, want %.6f)" % (
            theta, want, sy, 0.2 * yHy)
    except FloatingPointError as e:
        ok, detail = False, "(raised: %s)" % e
    check("a pair with s^T y < 0 is damped to s^T y = 0.2 y^T H y", ok, detail)

    for _ in range(3):
        sv = rng.standard_normal(n)
        s = vec(sv)
        y = vec(rng.standard_normal(n) + 3.0 * sv)
        op.update(s.copy(), y.copy())
    Hy = vec(np.zeros(n))
    op.solve(Hy, op.Y[-1])
    e = Hy.copy().axpy(-1.0, op.S[-1]).norm("l2") / op.S[-1].norm("l2")
    check("H y = s for the newest pair", e < 1e-12, "(rel %.1e)" % e)


# ------------------------------------------------- randomized eigensolvers
def test_randomized_eig():
    if RANK == 0:
        print("randomized eigensolvers")
    n = 60
    rng = np.random.default_rng(11)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    lam = np.array([10.0 ** (-0.35 * i) for i in range(n)])   # fast decay
    A = Q @ np.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    op = _DenseOp(A, COMM)

    k = 10
    Omega = hm.MultiVector(op.generate_vector(0), k + 10)
    hm.parRandom.set_seed(21)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePass(op, Omega, k, s=2)
    e = np.abs(d - lam[:k]).max() / lam[0]
    check("doublePass eigenvalues", e < 1e-8, "(max rel err %.2e)" % e)
    UtU = U.dot_mv(U)
    check("doublePass eigenvectors orthonormal",
          np.abs(UtU - np.eye(k)).max() < 1e-10)
    res, eB, eA = hm.check_std(op, U, d)
    check("doublePass residuals", res.max() / lam[0] < 1e-8,
          "(max %.2e)" % (res.max() / lam[0]))

    Omega2 = hm.MultiVector(op.generate_vector(0), k + 10)
    hm.parRandom.normal_multivector(1.0, Omega2)
    d1, U1 = hm.singlePass(op, Omega2, k, s=2)
    e = np.abs(d1 - lam[:k]).max() / lam[0]
    check("singlePass eigenvalues", e < 1e-3, "(max rel err %.2e)" % e)

    # generalized problem: A u = lam B u with B SPD
    Bm = Q @ np.diag(np.linspace(1.0, 3.0, n)) @ Q.T
    Bm = 0.5 * (Bm + Bm.T)
    Bop = _DenseOp(Bm, COMM)
    Binv = _DenseOp(np.linalg.inv(Bm), COMM)

    class _S:
        def __init__(self, o):
            self.o = o

        def solve(self, x, b):
            self.o.mult(b, x)
            return 1

        def init_vector(self, x, dim):
            return self.o.init_vector(x, dim)

    lam_g = np.sort(np.linalg.eigvalsh(np.linalg.solve(Bm, A)))[::-1]
    Omega3 = hm.MultiVector(op.generate_vector(0), k + 12)
    hm.parRandom.normal_multivector(1.0, Omega3)
    dg, Ug = hm.doublePassG(op, Bop, _S(Binv), Omega3, k, s=2)
    e = np.abs(dg - lam_g[:k]).max() / lam_g[0]
    check("doublePassG generalized eigenvalues", e < 1e-7,
          "(max rel err %.2e)" % e)
    BU = hm.MultiVector(Ug[0], k)
    hm.MatMvMult(Bop, Ug, BU)
    UtBU = Ug.dot_mv(BU)
    check("doublePassG B-orthonormal", np.abs(UtBU - np.eye(k)).max() < 1e-9,
          "(%.2e)" % np.abs(UtBU - np.eye(k)).max())


def test_low_rank_operator():
    if RANK == 0:
        print("LowRankOperator")
    n = 20
    rng = np.random.default_rng(5)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    k = 6
    d = np.linspace(5.0, 1.0, k)
    tpl = hm.ParVector(COMM, n if RANK == 0 else 0)
    U = hm.MultiVector(tpl, k)
    for i in range(k):
        if RANK == 0:
            U[i].array[:] = Q[:, i]
    lro = hm.LowRankOperator(d, U)

    x = tpl.duplicate()
    hm.parRandom.set_seed(4)
    hm.parRandom.normal(1.0, x)
    y = tpl.duplicate()
    lro.mult(x, y)
    xf = np.concatenate(COMM.allgather(x.array))
    want = Q[:, :k] @ (d * (Q[:, :k].T @ xf))
    got = np.concatenate(COMM.allgather(y.array))
    check("U D U^T action", np.abs(got - want).max() < 1e-12,
          "(%.2e)" % np.abs(got - want).max())

    z = tpl.duplicate()
    lro.solve(z, y)
    gotz = np.concatenate(COMM.allgather(z.array))
    proj = Q[:, :k] @ (Q[:, :k].T @ xf)
    check("pseudo-inverse on the range", np.abs(gotz - proj).max() < 1e-11,
          "(%.2e)" % np.abs(gotz - proj).max())

    diag = tpl.duplicate()
    lro.get_diagonal(diag)
    gd = np.concatenate(COMM.allgather(diag.array))
    wd = np.einsum("ij,j,ij->i", Q[:, :k], d, Q[:, :k])
    check("get_diagonal", np.abs(gd - wd).max() < 1e-12)
    check("trace", abs(lro.trace() - d.sum()) < 1e-10,
          "(%.10f vs %.10f)" % (lro.trace(), d.sum()))


def test_randomized_svd():
    if RANK == 0:
        print("randomized SVD")
    n = 40
    rng = np.random.default_rng(17)
    Qa = np.linalg.qr(rng.standard_normal((n, n)))[0]
    Qb = np.linalg.qr(rng.standard_normal((n, n)))[0]
    sig = np.array([2.0 ** (-0.7 * i) for i in range(n)])
    A = Qa @ np.diag(sig) @ Qb.T
    op = _DenseOp(A, COMM)
    op.multTranspose = lambda x, y, A=A, op=op: _apply(A.T, x, y, COMM)

    k = 8
    Omega = hm.MultiVector(op.generate_vector(0), k + 10)
    hm.parRandom.set_seed(31)
    hm.parRandom.normal_multivector(1.0, Omega)
    U, s, V = hm.accuracyEnhancedSVD(op, Omega, k, s=2)
    e = np.abs(s - sig[:k]).max() / sig[0]
    check("randomized SVD singular values", e < 1e-7, "(max rel err %.2e)" % e)
    check("left singular vectors orthonormal",
          np.abs(U.dot_mv(U) - np.eye(k)).max() < 1e-9)
    check("right singular vectors orthonormal",
          np.abs(V.dot_mv(V) - np.eye(k)).max() < 1e-9)


def _apply(M, x, y, comm):
    full = np.concatenate(comm.allgather(x.array))
    r = M @ full
    lo, hi = y.owner_range
    y.array[:] = r[lo:hi]
    return y


def test_trace_estimator():
    if RANK == 0:
        print("TraceEstimator")
    n = 50
    rng = np.random.default_rng(8)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    lam = np.linspace(1.0, 4.0, n)
    A = Q @ np.diag(lam) @ Q.T
    op = _DenseOp(0.5 * (A + A.T), COMM)
    hm.parRandom.set_seed(55)
    est = hm.TraceEstimator(op, False, 1e-2)
    tr, err = est(20, 4000)
    rel = abs(tr - lam.sum()) / lam.sum()
    check("trace estimate", rel < 0.05,
          "(%.4f vs %.4f, rel %.4f, reported se %.4f)" % (tr, lam.sum(), rel, err))


# ------------------------------------------------------ full inverse problem
def build_problem(n=16, order=2, ntargets=50, seed=1, gamma=0.1, delta=0.5,
                  truth=None):
    """The hIPPYlib subsurface benchmark: -div(exp(m) grad u) = 0.

    ``truth=None`` draws the true parameter from the prior, as hIPPYlib's demo
    does; passing a callable uses a smooth analytic field instead, which is the
    only case where pointwise data can actually recover the parameter and so the
    only case where an L2 error reduction is a meaningful assertion.
    """
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, order)
    Vm = hm.FunctionSpace.H1(pmesh, 1)

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    bc = hm.DirichletBC(Vu, lambda x: x[1], bdr_attributes=[1, 3])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    kind = "lu" if NP == 1 else "krylov"
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if NP == 1:
            setattr(pde, a, hm.LUSolver(COMM))
        else:
            s = hm.KrylovSolver(COMM, "cg", "amg")
            s.parameters["rel_tolerance"] = 1e-13
            s.parameters["max_iter"] = 2000
            setattr(pde, a, s)

    th0, th1, al = 2.0, 0.5, math.pi / 4
    sa, ca = math.sin(al), math.cos(al)
    Theta = np.array([[th0 * sa * sa + th1 * ca * ca, (th0 - th1) * sa * ca],
                      [(th0 - th1) * sa * ca, th0 * ca * ca + th1 * sa * sa]])
    prior = hm.BiLaplacianPrior(Vm, gamma, delta, Theta=Theta, robin_bc=True,
                                solver_type=kind)

    rng = np.random.default_rng(seed)
    targets = np.column_stack((rng.uniform(0.1, 0.9, ntargets),
                               rng.uniform(0.1, 0.5, ntargets)))
    B = hm.assemblePointwiseObservation(Vu, targets)

    hm.parRandom.set_seed(seed)
    if truth is None:
        noise = prior.noise_vector()
        prior.sample_noise(1.0, noise)
        mtrue = Vm.vector()
        prior.sample(noise, mtrue)
    else:
        mtrue = Vm.project(truth)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    noise_std = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, noise_std)
    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    return hm.Model(pde, prior, misfit), [Vu, Vm, Vu], mtrue, B


def test_newton_cg():
    if RANK == 0:
        print("ReducedSpaceNewtonCG (MAP point)")
    model, Vh, mtrue, B = build_problem(n=16, order=2, ntargets=50)
    Vm = Vh[PARAMETER]
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["abs_tolerance"] = 1e-12
    params["max_iter"] = 30
    params["globalization"] = "LS"
    params["GN_iter"] = 5
    params["print_level"] = 0 if RANK == 0 else -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, model.prior.mean.copy(), None])

    check("Newton-CG converged", solver.converged,
          "(%s, %d its, %d CG)" % (solver.termination_reasons[solver.reason],
                                   solver.it, solver.total_cg_iter))
    # the gradient at the MAP point must be small
    model.solveAdj(x[ADJOINT], x)
    g = Vm.vector()
    gn = model.evalGradientParameter(x, g)
    check("gradient small at the MAP point", gn < 1e-6 * max(solver.final_cost, 1.0),
          "(||g||_{R^-1} = %.3e)" % gn)

    # What optimization guarantees: the cost fell a long way, and the data is fit
    # to roughly the noise level (chi-square ~ ntargets/2).
    x0 = [model.generate_vector(STATE), model.prior.mean.copy(), None]
    model.solveFwd(x0[STATE], x0)
    c0 = model.cost(x0)
    cmap = model.cost(x)
    check("cost reduced from the prior mean", cmap[0] < 0.02 * c0[0],
          "(%.4e -> %.4e)" % (c0[0], cmap[0]))
    ntargets = model.misfit.B.ntargets
    check("data fitted to about the noise level",
          0.2 * ntargets < cmap[2] < 2.0 * ntargets,
          "(misfit %.2f, ntargets/2 = %.1f)" % (cmap[2], 0.5 * ntargets))
    if RANK == 0:
        print("      cost: prior mean %.4e -> MAP %.4e  (misfit %.3e, reg %.3e)"
              % (c0[0], cmap[0], cmap[2], cmap[1]))
    return model, Vh, mtrue, x, solver


def test_map_recovers_smooth_truth():
    """With a recoverable (smooth) truth, the MAP point must beat the prior mean."""
    if RANK == 0:
        print("MAP point with a smooth analytic truth")
    model, Vh, mtrue, B = build_problem(
        n=16, order=2, ntargets=80, seed=5, gamma=0.2, delta=1.0,
        truth=lambda xx: 1.2 * np.sin(np.pi * xx[0]) * np.sin(np.pi * xx[1]))
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-8
    params["max_iter"] = 30
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, model.prior.mean.copy(), None])
    check("Newton-CG converged (smooth truth)", solver.converged,
          "(%d its)" % solver.it)
    err_map = mtrue.copy().axpy(-1.0, x[PARAMETER]).norm("l2") / mtrue.norm("l2")
    err_pri = mtrue.copy().axpy(-1.0, model.prior.mean).norm("l2") / mtrue.norm("l2")
    check("MAP point beats the prior mean on a recoverable truth",
          err_map < 0.5 * err_pri, "(%.4f vs %.4f)" % (err_map, err_pri))
    if RANK == 0:
        print("      relative L2 error: MAP %.4f, prior mean %.4f"
              % (err_map, err_pri))


def test_trust_region():
    if RANK == 0:
        print("ReducedSpaceNewtonCG (trust region)")
    model, Vh, mtrue, B = build_problem(n=10, order=1, ntargets=30, seed=2)
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["globalization"] = "TR"
    params["max_iter"] = 40
    params["rel_tolerance"] = 1e-7
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, model.prior.mean.copy(), None])
    check("trust-region Newton-CG converged", solver.converged,
          "(%s, %d its)" % (solver.termination_reasons[solver.reason], solver.it))


def test_bfgs():
    """BFGS must find the same minimizer as Newton-CG.

    That is the useful check: two optimizers with nothing in common beyond the
    cost and gradient agreeing on the minimizer is strong evidence both the
    gradient and the optimizers are right.  How many iterations each takes is a
    property of the preconditioner, not a correctness claim, so it is reported
    rather than asserted.
    """
    if RANK == 0:
        print("BFGS")
    kw = dict(n=10, order=1, ntargets=30, seed=3)
    # The iteration budget has real margin: an effectively unpreconditioned BFGS
    # run on this problem takes 750-850 iterations, and the count moves by ~10%
    # under round-off-level changes (such as whether assembly keeps structural
    # zeros).  The test asserts convergence and agreement with Newton-CG, not the
    # count.
    bfgs_max_iter, bfgs_tol, agree_tol = 2000, 1e-8, 2e-3

    mN, _, _, _ = build_problem(**kw)
    pN = hm.ReducedSpaceNewtonCG_ParameterList()
    pN["rel_tolerance"] = 1e-10
    pN["max_iter"] = 40
    pN["print_level"] = -1
    sN = hm.ReducedSpaceNewtonCG(mN, pN)
    xN = sN.solve([None, mN.prior.mean.copy(), None])
    check("reference Newton-CG converged", sN.converged, "(%d its)" % sN.it)

    params = hm.BFGS_ParameterList()
    params["max_iter"] = bfgs_max_iter
    params["rel_tolerance"] = bfgs_tol
    params["print_level"] = -1
    params["BFGS_op"]["memory_limit"] = 25

    results = {}
    for tag in ("identity", "prior"):
        m, Vh, _, _ = build_problem(**kw)
        s = hm.BFGS(m, params)
        H0inv = (hm.RescaledIdentity(m.prior.init_vector) if tag == "identity"
                 else m.prior.Rsolver)
        x = s.solve([None, m.prior.mean.copy(), None], H0inv)
        check("BFGS (%s H0inv) converged" % tag, s.converged,
              "(%s, %d its)" % (s.termination_reasons[s.reason], s.it))
        rel = (x[PARAMETER].copy().axpy(-1.0, xN[PARAMETER]).norm("l2")
               / max(xN[PARAMETER].norm("l2"), 1e-300))
        check("BFGS (%s H0inv) agrees with Newton-CG" % tag, rel < agree_tol,
              "(rel %.3e)" % rel)
        results[tag] = s.it
    if RANK == 0:
        print("      iterations: scaled identity %d, prior %d"
              % (results["identity"], results["prior"]))


def test_laplace_approximation():
    if RANK == 0:
        print("Laplace approximation")
    model, Vh, mtrue, x, solver = test_newton_cg()
    Vm = Vh[PARAMETER]
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k, p = 40, 20
    Omega = hm.MultiVector(x[PARAMETER], k + p)
    hm.parRandom.set_seed(99)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, model.prior.R, model.prior.Rsolver, Omega, k, s=1)

    check("eigenvalues are real and sorted",
          np.all(np.diff(d) <= 1e-10 * max(abs(d[0]), 1.0)),
          "(d[0] = %.4e, d[-1] = %.4e)" % (d[0], d[-1]))
    check("spectrum decays", d[0] / max(abs(d[-1]), 1e-300) > 10.0,
          "(ratio %.3e)" % (d[0] / max(abs(d[-1]), 1e-300)))
    BU = hm.MultiVector(U[0], U.nvec())
    hm.MatMvMult(model.prior.R, U, BU)
    e = np.abs(U.dot_mv(BU) - np.eye(U.nvec())).max()
    check("U is R-orthonormal", e < 1e-8, "(%.2e)" % e)

    post = hm.GaussianLRPosterior(model.prior, d, U, mean=x[PARAMETER])

    # the low-rank precision and its inverse must be mutually inverse
    hm.parRandom.normal(1.0, Vm.vector())
    v = Vm.vector()
    hm.parRandom.normal(1.0, v)
    Hv = Vm.vector()
    post.Hlr.mult(v, Hv)
    back = Vm.vector()
    post.Hlr.solve(back, Hv)
    rel = back.copy().axpy(-1.0, v).norm("l2") / max(v.norm("l2"), 1e-300)
    check("low-rank precision and covariance are inverse", rel < 1e-6,
          "(%.2e)" % rel)

    # posterior variance must be below the prior variance everywhere
    pv, prv, corr = post.pointwise_variance(method="Exact")
    # the Monte Carlo prior variance is unbiased; the randomized one (hIPPYlib's
    # method) truncates the spectrum and so sits below the exact diagonal
    hm.parRandom.set_seed(11)
    mc = model.prior.pointwise_variance(method="MonteCarlo", n=1500)
    rel_mc = COMM.allreduce(float(np.abs(mc.array - prv.array).max() / max(np.abs(prv.array).max(), 1e-300)) if prv.local_size else 0.0, op=MPI.MAX)
    ratio_mc = mc.sum() / max(prv.sum(), 1e-300)
    check("Monte Carlo prior variance matches the exact diagonal (1500 samples)",
          rel_mc < 0.15 and abs(ratio_mc - 1.0) < 0.03, "(max rel dev %.3f, mean ratio %.3f)" % (rel_mc, ratio_mc))
    rnd = model.prior.pointwise_variance(method="Randomized", r=32)
    ratio_rnd = rnd.sum() / max(prv.sum(), 1e-300)
    check("randomized prior variance is a low-rank truncation (below exact)",
          ratio_rnd < 1.0 + 1e-6, "(mean ratio %.3f at r=32)" % ratio_rnd)
    check("posterior variance <= prior variance",
          COMM.allreduce(int(np.all(pv.array <= prv.array + 1e-12)), op=MPI.MIN) == 1)
    check("posterior variance positive",
          COMM.allreduce(int(np.all(pv.array > 0)), op=MPI.MIN) == 1,
          "(min %.3e)" % pv.min())
    # sum() is a collective: compute on every rank, print on one
    mean_prv = prv.sum() / prv.global_size
    mean_pv = pv.sum() / pv.global_size
    if RANK == 0:
        print("      mean prior var %.4e, mean posterior var %.4e"
              % (mean_prv, mean_pv))

    # traces must be consistent with the variance fields
    tr_post, tr_pr, tr_corr = post.trace(method="Exact")
    check("traces are consistent", tr_post > 0 and tr_corr > 0 and
          abs((tr_pr - tr_corr) - tr_post) < 1e-8 * max(abs(tr_pr), 1.0),
          "(post %.4e, prior %.4e, corr %.4e)" % (tr_post, tr_pr, tr_corr))

    # KL divergence from the prior must be positive
    kld, c_logdet, c_trace, c_shift = post.klDistanceFromPrior(sub_comp=True)
    check("KL divergence positive", kld > 0,
          "(%.4e = logdet %.3e + trace %.3e + shift %.3e)"
          % (kld, c_logdet, c_trace, c_shift))

    # posterior sample covariance must match the low-rank formula
    nsamp = 600
    hm.parRandom.set_seed(7)
    noise = model.prior.noise_vector()
    s_pr, s_po = Vm.vector(), Vm.vector()
    acc = np.zeros(Vm.GetTrueVSize())
    for _ in range(nsamp):
        model.prior.sample_noise(1.0, noise)
        post.sample(noise, s_pr, s_po, add_mean=False)
        acc += s_po.array ** 2
    acc /= nsamp
    num = (np.abs(acc - pv.array).max() / max(np.abs(pv.array).max(), 1e-300)
           if pv.local_size else 0.0)
    num = COMM.allreduce(num, op=MPI.MAX)
    check("posterior sample variance ~ pointwise variance (%d samples)" % nsamp,
          num < 0.35, "(rel %.3f)" % num)


def test_cg_sampler():
    if RANK == 0:
        print("CGSampler")
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(6, 6, mfem.Element.TRIANGLE))
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    # The LaplacianPrior's R is an assembled matrix, so CGSampler does pure
    # matrix-vector products.  The bi-Laplacian's R is A M^-1 A, which in parallel
    # would put an iterative mass solve inside every CG iteration of every sample,
    # a nested iteration that says nothing extra about the sampler.
    prior = hm.LaplacianPrior(Vm, 1.0, 4.0,
                              solver_type="lu" if NP == 1 else "krylov")
    sampler = hm.CGSampler()
    sampler.set_operator(prior.R)
    hm.parRandom.set_seed(77)
    nsamp = 3000 if NP == 1 else 1500
    s = Vm.vector()
    acc = np.zeros(Vm.GetTrueVSize())
    for _ in range(nsamp):
        sampler.sample(None, s)
        acc += s.array ** 2
    acc /= nsamp
    pv = prior.pointwise_variance("Exact")
    rel = (np.abs(acc - pv.array).max() / max(np.abs(pv.array).max(), 1e-300)
           if pv.local_size else 0.0)
    rel = COMM.allreduce(rel, op=MPI.MAX)
    check("CGSampler variance ~ diag(R^{-1})", rel < 0.35, "(rel %.3f)" % rel)

    # A ParVector of variates is read in global order on every rank, so it gives the
    # same sample as the same numbers passed as an array; reading each rank's own slice
    # would use different coefficients per rank and hang once one slice runs out.
    z = np.random.default_rng(5).standard_normal(4 * Vm.GlobalTrueVSize())
    zv = Vm.vector()
    off = COMM.scan(zv.local_size) - zv.local_size
    zv.array[:] = z[off:off + zv.local_size]
    sa, sv = Vm.vector(), Vm.vector()
    hm.parRandom.set_seed(3)
    sampler.sample(z, sa)
    hm.parRandom.set_seed(3)
    sampler.sample(zv, sv)
    d = sv.copy().axpy(-1.0, sa).norm("l2") / max(sa.norm("l2"), 1e-300)
    check("CGSampler reads a ParVector of variates in global order", d == 0.0,
          "(rel diff %.1e)" % d)


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 74)
        print("hIPPyMFEM optimization and posterior tests on %d rank(s)" % NP)
        print("=" * 74)
    test_cg_steihaug()
    test_randomized_eig()
    test_low_rank_operator()
    test_randomized_svd()
    test_trace_estimator()
    test_trust_region()
    test_bfgs_operator()
    test_bfgs()
    test_map_recovers_smooth_truth()
    test_laplace_approximation()
    test_cg_sampler()
    if RANK == 0:
        print("-" * 74)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

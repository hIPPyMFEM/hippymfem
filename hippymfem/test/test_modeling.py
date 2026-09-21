# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Priors, observation operators, misfits, the model and the reduced Hessian.

Run with ``mpirun -n N python -m hippymfem.test.test_modeling``.
"""

import math

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hm
from hippymfem.common.linalg import operator_to_dense
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


def mesh2d(n=12):
    return mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE))


# --------------------------------------------------------------------- priors
def test_sqrtM():
    """sqrtM sqrtM^T must equal the mass matrix of the same quadrature rule."""
    if RANK == 0:
        print("prior: square root of the mass matrix")
    pmesh = mesh2d(8)
    for order in (1, 2):
        V = hm.FunctionSpace.H1(pmesh, order)
        qdeg = 2 * order
        S = hm.QuadratureSqrtPrecision(V, qdeg, value_coeff=1.0, grad_coeff=None)
        ir = S.batches.groups[0].ir
        mi = mfem.MassIntegrator()
        mi.SetIntRule(ir)
        M = hm.assemble_native_matrix(V, [mi])

        hm.parRandom.set_seed(5)
        x = V.vector()
        hm.parRandom.normal(1.0, x)
        # <S S^T x, x> == <M x, x>
        noise = S.noise_vector()
        S.multTranspose(x, noise)
        y = V.vector()
        S.mult(noise, y)
        Mx = V.vector()
        M.Mult(x.hypre, Mx.hypre)
        e = y.copy().axpy(-1.0, Mx).norm("l2") / max(Mx.norm("l2"), 1e-300)
        check("P%d: sqrtM sqrtM^T == M" % order, e < 1e-12, "(%.2e)" % e)


def test_sqrtR():
    """sqrtR sqrtR^T must equal gamma*L + delta*M."""
    if RANK == 0:
        print("prior: square root of gamma L + delta M")
    pmesh = mesh2d(8)
    V = hm.FunctionSpace.H1(pmesh, 1)
    gamma, delta = 0.7, 2.3
    S = hm.QuadratureSqrtPrecision(V, 2, value_coeff=math.sqrt(delta),
                                   grad_coeff=math.sqrt(gamma))
    ir = S.batches.groups[0].ir
    di = mfem.DiffusionIntegrator(mfem.ConstantCoefficient(gamma))
    di.SetIntRule(ir)
    mi = mfem.MassIntegrator(mfem.ConstantCoefficient(delta))
    mi.SetIntRule(ir)
    R = hm.assemble_native_matrix(V, [di, mi])

    hm.parRandom.set_seed(6)
    x = V.vector()
    hm.parRandom.normal(1.0, x)
    noise = S.noise_vector()
    S.multTranspose(x, noise)
    y = V.vector()
    S.mult(noise, y)
    Rx = V.vector()
    R.Mult(x.hypre, Rx.hypre)
    e = y.copy().axpy(-1.0, Rx).norm("l2") / max(Rx.norm("l2"), 1e-300)
    check("sqrtR sqrtR^T == gamma L + delta M", e < 1e-12, "(%.2e)" % e)

    # S and S^T must be adjoint
    n2 = S.noise_vector()
    hm.parRandom.normal(1.0, n2)
    Sn = V.vector()
    S.mult(n2, Sn)
    STx = S.noise_vector()
    S.multTranspose(x, STx)
    lhs, rhs = Sn.inner(x), STx.inner(n2)
    check("<S n, x> == <n, S^T x>", abs(lhs - rhs) <= 1e-11 * max(1.0, abs(lhs)),
          "(%.6e vs %.6e)" % (lhs, rhs))


def test_prior_sampling():
    """Sample covariance of the prior must converge to R^{-1}."""
    if RANK == 0:
        print("prior: sample covariance vs R^{-1}")
    pmesh = mesh2d(6)
    V = hm.FunctionSpace.H1(pmesh, 1)
    prior = hm.BiLaplacianPrior(V, gamma=1.0, delta=4.0, robin_bc=False,
                                solver_type="lu" if NP == 1 else "krylov")
    n = V.GlobalTrueVSize()

    # exact R^{-1} as a dense matrix (small mesh)
    from hippymfem.common.operators import Solver2Operator

    Rinv = operator_to_dense(
        Solver2Operator(prior.Rsolver, init_vector=prior.init_vector), n, COMM)

    nsamp = 4000
    hm.parRandom.set_seed(2024)
    acc = np.zeros((n, n))
    s = V.vector()
    noise = prior.noise_vector()
    for k in range(nsamp):
        prior.sample_noise(1.0, noise)
        prior.sample(noise, s, add_mean=False)
        full = np.concatenate(COMM.allgather(s.array))
        acc += np.outer(full, full)
    acc /= nsamp
    num = np.linalg.norm(acc - Rinv) / np.linalg.norm(Rinv)
    # Monte Carlo error scales like sqrt(n^2/nsamp); be generous but meaningful
    check("sample covariance ~ R^{-1} (%d samples, n=%d)" % (nsamp, n),
          num < 0.25, "(rel %.3f)" % num)

    # the diagonal is the more accurate statistic
    dnum = np.abs(np.diag(acc) - np.diag(Rinv)).max() / np.abs(np.diag(Rinv)).max()
    check("sample variance ~ pointwise variance", dnum < 0.12, "(rel %.3f)" % dnum)

    pw = prior.pointwise_variance("Exact")
    pwfull = np.concatenate(COMM.allgather(pw.array))
    e = np.abs(pwfull - np.diag(Rinv)).max() / np.abs(np.diag(Rinv)).max()
    check("pointwise_variance('Exact') == diag(R^{-1})", e < 1e-8, "(%.2e)" % e)

    # R and Rsolver must be inverses
    hm.parRandom.normal(1.0, s)
    Rs = V.vector()
    prior.R.mult(s, Rs)
    back = V.vector()
    prior.Rsolver.solve(back, Rs)
    e = back.copy().axpy(-1.0, s).norm("l2") / max(s.norm("l2"), 1e-300)
    check("Rsolver inverts R", e < 1e-8, "(%.2e)" % e)

    # cost and grad consistency: grad = R (m - mean), cost = 0.5 <grad, m-mean>
    hm.parRandom.normal(1.0, s)
    g = V.vector()
    prior.grad(s, g)
    c = prior.cost(s)
    d = s.copy().axpy(-1.0, prior.mean)
    check("prior cost == 0.5 <grad, m-mean>",
          abs(c - 0.5 * g.inner(d)) <= 1e-11 * max(1.0, abs(c)),
          "(%.8e vs %.8e)" % (c, 0.5 * g.inner(d)))


def test_laplacian_prior():
    if RANK == 0:
        print("prior: LaplacianPrior")
    pmesh = mesh2d(6)
    V = hm.FunctionSpace.H1(pmesh, 1)
    prior = hm.LaplacianPrior(V, gamma=1.0, delta=2.0,
                              solver_type="lu" if NP == 1 else "krylov")
    s = V.vector()
    noise = prior.noise_vector()
    hm.parRandom.set_seed(9)
    prior.sample_noise(1.0, noise)
    prior.sample(noise, s)
    check("LaplacianPrior sample finite", np.all(np.isfinite(s.array)))
    g = V.vector()
    prior.grad(s, g)
    c = prior.cost(s)
    check("LaplacianPrior cost == 0.5 <grad, m>",
          abs(c - 0.5 * g.inner(s)) <= 1e-10 * max(1.0, abs(c)))


def test_gaussian_real_prior():
    if RANK == 0:
        print("prior: GaussianRealPrior")
    cov = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, -0.2], [0.0, -0.2, 0.5]])
    prior = hm.GaussianRealPrior(None, cov, comm=COMM)
    nsamp = 20000
    hm.parRandom.set_seed(123)
    acc = np.zeros((3, 3))
    s = prior.noise_vector()
    noise = prior.noise_vector()
    for _ in range(nsamp):
        prior.sample_noise(1.0, noise)
        prior.sample(noise, s, add_mean=False)
        full = np.concatenate(COMM.allgather(s.array))
        acc += np.outer(full, full)
    acc /= nsamp
    e = np.linalg.norm(acc - cov) / np.linalg.norm(cov)
    check("GaussianRealPrior sample covariance", e < 0.06, "(rel %.4f)" % e)


def test_prior_rank_invariance():
    """A prior sample must not depend on how the mesh is partitioned."""
    if RANK == 0:
        print("prior: partition independence of samples")
    pmesh = mesh2d(8)
    V = hm.FunctionSpace.H1(pmesh, 1)
    prior = hm.BiLaplacianPrior(V, 1.0, 4.0, solver_type="krylov")
    prior.Asolver.parameters["rel_tolerance"] = 1e-14
    hm.parRandom.set_seed(4242)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    s = V.vector()
    prior.sample(noise, s, add_mean=False)
    full = np.concatenate(COMM.allgather(s.array))
    # compare against a dof-order-independent functional: the mass-weighted norm
    M = hm.assemble_native_matrix(V, [mfem.MassIntegrator()])
    Ms = V.vector()
    M.Mult(s.hypre, Ms.hypre)
    # Compute the invariants on EVERY rank: inner() and sum() are collectives,
    # and calling them inside a rank-0-only block deadlocks the job.
    sMs = Ms.inner(s)
    ssum = s.sum()
    if RANK == 0:
        print("      <s, M s> = %.14f   sum(s) = %.14f" % (sMs, ssum))
    check("sample computed (see printed invariants for cross-rank comparison)",
          np.all(np.isfinite(full)))


# ------------------------------------------------------ observation operators
def test_pointwise_observation():
    if RANK == 0:
        print("observation operator")
    pmesh = mesh2d(10)
    V = hm.FunctionSpace.H1(pmesh, 2)
    rng = np.random.default_rng(7)
    targets = rng.uniform(0.05, 0.95, size=(37, 2))
    B = hm.assemblePointwiseObservation(V, targets)
    check("every target owned exactly once",
          COMM.allreduce(B.n_owned) == B.ntargets,
          "(%d of %d)" % (COMM.allreduce(B.n_owned), B.ntargets))

    # exactness on a quadratic field
    f = lambda x: 1.0 + 2.0 * x[0] - 0.5 * x[1] + x[0] * x[1] + 0.3 * x[0] ** 2
    u = V.project(f)
    obs = B.createVecLeft()
    B.mult(u, obs)
    got = B.gather(obs)
    want = np.array([f(t) for t in targets])
    e = np.abs(got - want).max()
    check("B reproduces a quadratic field exactly", e < 1e-11, "(%.2e)" % e)

    # adjoint identity
    hm.parRandom.set_seed(8)
    y = B.createVecLeft()
    hm.parRandom.normal(1.0, y)
    Bty = V.vector()
    B.multTranspose(y, Bty)
    lhs, rhs = obs.inner(y), Bty.inner(u)
    check("<Bu, y> == <u, B^T y>", abs(lhs - rhs) <= 1e-11 * max(1.0, abs(lhs)),
          "(%.8e vs %.8e)" % (lhs, rhs))

    # scatter/gather round-trip
    o2 = B.scatter(want)
    e = np.abs(B.gather(o2) - want).max()
    check("scatter/gather round-trip", e < 1e-14)

    # a target outside the mesh must be reported
    raised = False
    try:
        hm.assemblePointwiseObservation(V, np.array([[1.7, 0.5]]))
    except ValueError as exc:
        raised = "outside the mesh" in str(exc)
    check("off-mesh target is rejected", raised)


# ------------------------------------------------------------ model + Hessian
def build_inverse_problem(n=12, order=2, ntargets=40, seed=1):
    """The hIPPYlib subsurface-flow benchmark: -div(exp(m) grad u) = 0."""
    pmesh = mesh2d(n)
    Vu = hm.FunctionSpace.H1(pmesh, order)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad)

    bc = hm.DirichletBC(Vu, lambda x: x[1], bdr_attributes=[1, 3])
    pde = hm.PDEVariationalProblem(Vh, pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    kind = "lu" if NP == 1 else "krylov"
    if NP == 1:
        pde.solver = hm.LUSolver(COMM)
        pde.solver_fwd_inc = hm.LUSolver(COMM)
        pde.solver_adj_inc = hm.LUSolver(COMM)
    else:
        for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
            s = hm.KrylovSolver(COMM, "cg", "amg")
            s.parameters["rel_tolerance"] = 1e-14
            setattr(pde, a, s)

    theta0, theta1, alpha = 2.0, 0.5, math.pi / 4
    sa, ca = math.sin(alpha), math.cos(alpha)
    Theta = np.array([[theta0 * sa * sa + theta1 * ca * ca,
                       (theta0 - theta1) * sa * ca],
                      [(theta0 - theta1) * sa * ca,
                       theta0 * ca * ca + theta1 * sa * sa]])
    prior = hm.BiLaplacianPrior(Vm, gamma=0.1, delta=0.5, Theta=Theta,
                                robin_bc=True, solver_type=kind)

    rng = np.random.default_rng(seed)
    targets = np.column_stack((rng.uniform(0.1, 0.9, ntargets),
                               rng.uniform(0.1, 0.5, ntargets)))
    B = hm.assemblePointwiseObservation(Vu, targets)

    # synthetic truth and data
    hm.parRandom.set_seed(seed)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    rel_noise = 0.01
    noise_std = rel_noise * max(data.norm("linf"), 1e-30)
    B.perturb(data, noise_std)
    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    model = hm.Model(pde, prior, misfit)
    return model, Vh, mtrue, utrue, B


def test_model_verify():
    if RANK == 0:
        print("model: finite-difference gradient and Hessian")
    model, Vh, mtrue, utrue, B = build_inverse_problem(n=10, order=1, ntargets=25)
    m0 = Vh[PARAMETER].project(lambda x: np.sin(x[0]))
    hm.parRandom.set_seed(3)
    eps = np.power(2.0, -np.arange(2, 20))
    e, eg, eH = hm.modelVerify(model, m0, misfit_only=False, verbose=(RANK == 0),
                               eps=eps)
    sg = hm.best_slope(e, eg)
    sH = hm.best_slope(e, eH)
    check("gradient FD error is first order", 0.8 < sg < 1.3, "(slope %.3f)" % sg)
    check("Hessian FD error is first order", 0.8 < sH < 1.3, "(slope %.3f)" % sH)
    return model, Vh, m0


def test_model_verify_nonlinear_inhomogeneous_bc():
    """Finite differences with a state-nonlinear residual and nowhere-zero Dirichlet data.

    ``test_model_verify`` differences a residual that is linear in the state, so its
    ``W_uu`` block is identically zero and never enters the Hessian, and its Dirichlet
    value ``x[1]`` vanishes on one of the two constrained edges.  This one closes both
    gaps at once: the residual is cubic in the state, so ``W_uu`` is a real operator,
    and the boundary value is bounded away from zero everywhere it is imposed, so a
    mistake in lifting inhomogeneous data cannot hide on a part of the boundary where
    the data happens to be zero.  The essential rows still have to drop out of the
    gradient and the Hessian, which is what the slopes check.
    """
    if RANK == 0:
        print("model: FD with a nonlinear residual and inhomogeneous Dirichlet data")
    pmesh = mesh2d(8)
    Vu = hm.FunctionSpace.H1(pmesh, 2)
    Vm = hm.FunctionSpace.H1(pmesh, 1)

    def pde_varf(u, m, p, x):
        return (jnp.exp(m.val) * hm.inner(u.grad, p.grad)
                + u.val ** 3 * p.val)

    # nowhere zero on the constrained boundary, and not constant either
    bc = hm.DirichletBC(Vu, lambda x: 2.0 + x[1], bdr_attributes=[1, 3])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=False)
    kind = "lu" if NP == 1 else "krylov"
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if NP == 1:
            setattr(pde, a, hm.LUSolver(COMM))
        else:
            sol = hm.KrylovSolver(COMM, "gmres", "amg")
            sol.parameters["rel_tolerance"] = 1e-14
            setattr(pde, a, sol)

    prior = hm.BiLaplacianPrior(Vm, gamma=0.3, delta=1.0, robin_bc=True,
                                solver_type=kind)
    rng = np.random.default_rng(5)
    targets = np.column_stack((rng.uniform(0.15, 0.85, 20),
                               rng.uniform(0.15, 0.85, 20)))
    B = hm.assemblePointwiseObservation(Vu, targets)

    hm.parRandom.set_seed(5)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    # the state must actually feel the boundary data
    check("the state is nonzero under inhomogeneous data",
          utrue.norm("linf") > 1.0, "(|u|_inf = %.3f)" % utrue.norm("linf"))
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    model = hm.Model(pde, prior, hm.DiscreteStateObservation(B, data, nstd ** 2))

    m0 = Vm.vector()
    hm.parRandom.normal(0.3, m0)
    eps = np.power(2.0, -np.arange(2, 18))
    e, eg, eH = hm.modelVerify(model, m0, misfit_only=False, verbose=False, eps=eps)
    sg = hm.best_slope(e, eg)
    sH = hm.best_slope(e, eH)
    check("gradient FD error is first order", 0.8 < sg < 1.3, "(slope %.3f)" % sg)
    check("Hessian FD error is first order", 0.8 < sH < 1.3, "(slope %.3f)" % sH)

    H = hm.ReducedHessian(model)
    xx, yy = Vm.vector(), Vm.vector()
    hm.parRandom.normal(1.0, xx)
    hm.parRandom.normal(1.0, yy)
    a, b = H.inner(yy, xx), H.inner(xx, yy)
    rel = 2 * abs(a - b) / abs(a + b) if (a + b) != 0.0 else abs(a - b)
    check("reduced Hessian is symmetric", rel < 1e-10, "(%.2e)" % rel)


def test_hessian_properties():
    if RANK == 0:
        print("reduced Hessian: symmetry, positivity, FD agreement")
    model, Vh, mtrue, utrue, B = build_inverse_problem(n=8, order=1, ntargets=20)
    Vm = Vh[PARAMETER]
    x = model.generate_vector()
    x[PARAMETER] = Vm.project(lambda xx: 0.3 * np.cos(2 * xx[0]))
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)

    for gn in (True, False):
        model.setPointForHessianEvaluations(x, gauss_newton_approx=gn)
        H = hm.ReducedHessian(model, misfit_only=False)
        hm.parRandom.set_seed(11)
        a, b = Vm.vector(), Vm.vector()
        hm.parRandom.normal(1.0, a)
        hm.parRandom.normal(1.0, b)
        Ha, Hb = Vm.vector(), Vm.vector()
        H.mult(a, Ha)
        H.mult(b, Hb)
        s = abs(Ha.inner(b) - Hb.inner(a)) / max(abs(Ha.inner(b)), 1e-300)
        tag = "GN" if gn else "full"
        check("%s Hessian symmetric" % tag, s < 1e-9, "(%.2e)" % s)
        if gn:
            # The Gauss-Newton Hessian is C^T A^-T W_uu A^-1 C + R with W_uu
            # positive semi-definite and R positive definite, so it is positive
            # definite everywhere.  The full Hessian is not: away from a minimum it
            # can be indefinite, which is why Newton-CG starts in Gauss-Newton mode
            # and CGSolverSteihaug has a "negative direction" termination code.
            # Its positivity here holds only for some random directions, so
            # asserting it would make the test depend on the rank count, which
            # changes the random vector.
            check("GN Hessian positive definite at an arbitrary point",
                  Ha.inner(a) > 0, "(%.6e)" % Ha.inner(a))

    # At a converged minimum the full Hessian must be positive definite; that is
    # the meaningful statement, and the place to test it.
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-9
    params["max_iter"] = 30
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    xmap = solver.solve([None, model.prior.mean.copy(), None])
    check("reference Newton-CG converged", solver.converged,
          "(%s)" % solver.termination_reasons[solver.reason])
    model.setPointForHessianEvaluations(xmap, gauss_newton_approx=False)
    Hm = hm.ReducedHessian(model, misfit_only=False)
    hm.parRandom.set_seed(23)
    worst = None
    for _ in range(8):
        v = Vm.vector()
        hm.parRandom.normal(1.0, v)
        Hv = Vm.vector()
        Hm.mult(v, Hv)
        q = Hv.inner(v)
        worst = q if worst is None else min(worst, q)
    check("full Hessian positive definite at the MAP point", worst > 0,
          "(smallest of 8 random quadratic forms: %.6e)" % worst)

    # full Hessian vs finite differences of the gradient
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    H = hm.ReducedHessian(model, misfit_only=False)
    Hfd = hm.FDHessian(model, x[PARAMETER], 1e-5, misfit_only=False)
    hm.parRandom.set_seed(12)
    d = Vm.vector()
    hm.parRandom.normal(1.0, d)
    h1, h2 = Vm.vector(), Vm.vector()
    H.mult(d, h1)
    Hfd.mult(d, h2)
    e = h2.copy().axpy(-1.0, h1).norm("l2") / max(h1.norm("l2"), 1e-300)
    check("ReducedHessian == FDHessian", e < 1e-5, "(%.2e)" % e)


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 74)
        print("hIPPyMFEM modeling tests (priors, misfits, the model, the Hessian) on %d rank(s)" % NP)
        print("=" * 74)
    test_sqrtM()
    test_sqrtR()
    test_prior_sampling()
    test_laplacian_prior()
    test_gaussian_real_prior()
    test_prior_rank_invariance()
    test_pointwise_observation()
    test_model_verify()
    test_model_verify_nonlinear_inhomogeneous_bc()
    test_hessian_properties()
    if RANK == 0:
        print("-" * 74)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""MCMC and forward uncertainty propagation.

The MCMC checks use a *linear* forward problem, where the posterior is Gaussian
and known in closed form, so the chain can be compared against the exact answer
rather than against itself.  Run with
``mpirun -n N python -m hippymfem.test.test_uq``.
"""

import math

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hm
from hippymfem.common.linalg import operator_to_dense
from hippymfem.modeling.variables import PARAMETER

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
FAILS = []


def check(name, ok, detail=""):
    if RANK == 0:
        print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


def build_linear_problem(nx=8, ntargets=12, noise_std=0.05, seed=1):
    """A *linear* inverse problem, so the exact posterior is available.

    The forward map is ``-lap u = m`` with ``u = 0`` on the boundary, so
    ``u = A^{-1} m`` is linear in the parameter and the posterior is exactly
    Gaussian with precision ``H = B^T B / sigma^2 + R``.  Everything MCMC
    produces can then be checked against that.
    """
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(nx, nx,
                                                         mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, 1)
    Vm = hm.FunctionSpace.H1(pmesh, 1)

    def pde_varf(u, m, p, x):
        return hm.inner(u.grad, p.grad) - m.val * p.val

    bc = hm.DirichletBC(Vu, 0.0, bdr_attributes="all")
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if NP == 1:
            setattr(pde, a, hm.LUSolver(COMM))
        else:
            s = hm.KrylovSolver(COMM, "cg", "amg")
            s.parameters["rel_tolerance"] = 1e-14
            setattr(pde, a, s)

    prior = hm.BiLaplacianPrior(Vm, 1.0, 4.0,
                                solver_type="lu" if NP == 1 else "krylov")
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.15, 0.85, size=(ntargets, 2))
    B = hm.assemblePointwiseObservation(Vu, targets)

    hm.parRandom.set_seed(seed)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    B.perturb(data, noise_std)
    misfit = hm.DiscreteStateObservation(B, data, noise_std ** 2)
    return hm.Model(pde, prior, misfit), [Vu, Vm, Vu], mtrue


def exact_gaussian_posterior(model, Vm):
    """Dense mean and covariance of the posterior of a linear problem."""
    import scipy.linalg as sla

    n = Vm.GlobalTrueVSize()
    # MAP point, the exact minimizer of a quadratic cost.  The tolerance is 1e-8, not
    # 1e-12: the cost is flat at the minimum, so below about 1e-8 no trial step
    # registers a decrease and Newton-CG correctly stops.  The checks below compare
    # against the exact posterior to a few percent, so 1e-8 is ample.
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-8
    params["max_iter"] = 30
    params["GN_iter"] = 30          # the problem is linear: GN is exact
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, model.prior.mean.copy(), None])
    model.setPointForHessianEvaluations(x, gauss_newton_approx=True)
    H = hm.ReducedHessian(model, misfit_only=False)
    Hd = operator_to_dense(H, n, COMM)
    Hd = 0.5 * (Hd + Hd.T)
    cov = sla.inv(Hd)
    mean = np.concatenate(COMM.allgather(x[PARAMETER].array))
    return mean, cov, x, solver


def test_mcmc_pcn():
    if RANK == 0:
        print("MCMC: pCN against the exact Gaussian posterior")
    model, Vh, mtrue = build_linear_problem(nx=6, ntargets=8, noise_std=0.05)
    Vm = Vh[PARAMETER]
    mean_ex, cov_ex, xmap, ref = exact_gaussian_posterior(model, Vm)
    # A reference MAP point needs a small gradient, not a particular termination
    # reason.  With LU forward solves the gradient falls by 1e-13; with iterative
    # solves in parallel the linear solver sets a floor near 1e-8, where the line
    # search stops.  Either is ample for checks at a few percent, so the reduction
    # is what is asserted.
    reduction = ref.final_grad_norm / max(ref.initial_grad_norm, 1e-300)
    check("reference MAP point is a minimizer", reduction < 1e-6,
          "(||g||/||g_0|| = %.2e; %s after %d iterations)"
          % (reduction, ref.termination_reasons[ref.reason], ref.it))

    kernel = hm.pCNKernel(model)
    kernel.parameters["s"] = 0.3
    chain = hm.MCMC(kernel)
    chain.parameters["number_of_samples"] = 4000
    chain.parameters["burn_in"] = 1500
    chain.parameters["print_level"] = 0
    tracer = _MeanTracer(Vm)
    hm.parRandom.set_seed(1234)
    naccept = chain.run(xmap[PARAMETER].copy(), qoi=_FirstDofQoi(Vm), tracer=tracer)
    rate = naccept / chain.parameters["number_of_samples"]
    check("pCN acceptance rate is reasonable", 0.05 < rate < 0.95,
          "(%.1f%%)" % (100 * rate))

    mean_mc = tracer.mean()
    scale = max(np.abs(mean_ex).max(), 1e-300)
    e = np.abs(mean_mc - mean_ex).max() / scale
    check("pCN chain mean ~ posterior mean", e < 0.25, "(rel %.3f)" % e)

    var_mc = tracer.variance()
    var_ex = np.diag(cov_ex)
    ev = np.abs(var_mc - var_ex).max() / max(np.abs(var_ex).max(), 1e-300)
    check("pCN chain variance ~ posterior variance", ev < 0.5, "(rel %.3f)" % ev)
    # plain numpy on replicated arrays, but hoisted so the collective scanner
    # (tools/check_collectives.py) stays clean and usable as a gate
    a_mean, a_chain = float(np.abs(mean_ex).max()), float(np.abs(mean_mc).max())
    a_var, a_cvar = float(var_ex.max()), float(var_mc.max())
    if RANK == 0:
        print("      mean |exact| %.4e, |chain| %.4e; var |exact| %.4e, |chain| %.4e"
              % (a_mean, a_chain, a_var, a_cvar))
    return model, Vh, xmap


def test_mcmc_gpcn():
    if RANK == 0:
        print("MCMC: gpCN with the Laplace approximation as proposal")
    model, Vh, mtrue = build_linear_problem(nx=6, ntargets=8, noise_std=0.05)
    Vm = Vh[PARAMETER]
    mean_ex, cov_ex, xmap, _ref = exact_gaussian_posterior(model, Vm)

    # Laplace approximation at the MAP point
    model.setPointForHessianEvaluations(xmap, gauss_newton_approx=True)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)
    k = min(30, Vm.GlobalTrueVSize() - 1)
    Omega = hm.MultiVector(xmap[PARAMETER], k + 10)
    hm.parRandom.set_seed(77)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, model.prior.R, model.prior.Rsolver, Omega, k, s=2)
    nu = hm.GaussianLRPosterior(model.prior, d, U, mean=xmap[PARAMETER])

    kernel = hm.gpCNKernel(model, nu)
    kernel.parameters["s"] = 0.7
    chain = hm.MCMC(kernel)
    chain.parameters["number_of_samples"] = 3000
    chain.parameters["burn_in"] = 800
    chain.parameters["print_level"] = 0
    tracer = _MeanTracer(Vm)
    hm.parRandom.set_seed(999)
    naccept = chain.run(xmap[PARAMETER].copy(), qoi=_FirstDofQoi(Vm), tracer=tracer)
    rate = naccept / chain.parameters["number_of_samples"]
    # for a linear problem the Laplace approximation IS the posterior, so gpCN
    # proposals are exact and essentially everything is accepted
    check("gpCN acceptance rate is high on a linear problem", rate > 0.85,
          "(%.1f%%)" % (100 * rate))
    e = np.abs(tracer.mean() - mean_ex).max() / max(np.abs(mean_ex).max(), 1e-300)
    check("gpCN chain mean ~ posterior mean", e < 0.15, "(rel %.3f)" % e)


def test_mcmc_mala():
    if RANK == 0:
        print("MCMC: MALA")
    model, Vh, mtrue = build_linear_problem(nx=6, ntargets=8, noise_std=0.1)
    Vm = Vh[PARAMETER]
    mean_ex, cov_ex, xmap, _ref = exact_gaussian_posterior(model, Vm)
    kernel = hm.MALAKernel(model)
    kernel.parameters["delta_t"] = 0.3
    chain = hm.MCMC(kernel)
    chain.parameters["number_of_samples"] = 2000
    chain.parameters["burn_in"] = 600
    chain.parameters["print_level"] = 0
    tracer = _MeanTracer(Vm)
    hm.parRandom.set_seed(55)
    naccept = chain.run(xmap[PARAMETER].copy(), qoi=_FirstDofQoi(Vm), tracer=tracer)
    rate = naccept / chain.parameters["number_of_samples"]
    check("MALA acceptance rate is reasonable", 0.05 < rate <= 1.0,
          "(%.1f%%)" % (100 * rate))
    e = np.abs(tracer.mean() - mean_ex).max() / max(np.abs(mean_ex).max(), 1e-300)
    check("MALA chain mean ~ posterior mean", e < 0.35, "(rel %.3f)" % e)


def test_diagnostics():
    if RANK == 0:
        print("chain diagnostics")
    rng = np.random.default_rng(3)
    n = 20000
    # AR(1) with known IACT = (1+rho)/(1-rho)
    rho = 0.8
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.standard_normal() * math.sqrt(1 - rho ** 2)
    iact, lags, ac = hm.integratedAutocorrelationTime(x, max_lag=200)
    exact = (1 + rho) / (1 - rho)
    e = abs(iact - exact) / exact
    check("IACT of an AR(1) chain", e < 0.25,
          "(%.3f vs exact %.3f)" % (iact, exact))
    ess = hm.effective_sample_size(x, max_lag=200)
    check("effective sample size < n", 0 < ess < n, "(%.0f of %d)" % (ess, n))
    summ = hm.chain_summary(x, max_lag=200)
    check("chain_summary keys",
          set(summ) == {"mean", "std", "iact", "ess", "standard_error"})

    # an i.i.d. chain should have IACT ~ 1
    y = rng.standard_normal(n)
    iact_iid, _, _ = hm.integratedAutocorrelationTime(y, max_lag=200)
    check("IACT of an i.i.d. chain ~ 1", abs(iact_iid - 1.0) < 0.3,
          "(%.3f)" % iact_iid)
    check("QoiTracer records", _tracer_roundtrip())


def _tracer_roundtrip():
    t = hm.QoiTracer(5)
    for k in range(5):
        t.append(None, float(k))
    return np.array_equal(t.trim(), np.arange(5.0))


class _FirstDofQoi:
    """The parameter's value at a fixed interior point: a scalar to trace."""

    def __init__(self, Vm):
        self.Vm = Vm
        self.B = hm.assemblePointwiseObservation(Vm, np.array([[0.5, 0.5]]))
        self.out = self.B.createVecLeft()

    def eval(self, x):
        self.B.mult(x[PARAMETER], self.out)
        return float(self.B.gather(self.out)[0])


class _MeanTracer:
    """Accumulates the running mean and variance of the whole parameter field."""

    def __init__(self, Vm):
        self.Vm = Vm
        self.n = 0
        self.s1 = None
        self.s2 = None

    def append(self, current, q):
        full = np.concatenate(COMM.allgather(current.m.array))
        if self.s1 is None:
            self.s1 = np.zeros_like(full)
            self.s2 = np.zeros_like(full)
        self.s1 += full
        self.s2 += full * full
        self.n += 1

    def mean(self):
        return self.s1 / max(self.n, 1)

    def variance(self):
        m = self.mean()
        return np.maximum(self.s2 / max(self.n, 1) - m * m, 0.0)


# ------------------------------------------------------------------ forward UQ
def test_forward_uq():
    if RANK == 0:
        print("forward UQ: QoI map, Taylor approximation, variance reduction")
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(10, 10,
                                                         mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, 2)
    Vm = hm.FunctionSpace.H1(pmesh, 1)

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad) - 1.0 * p.val

    bc = hm.DirichletBC(Vu, 0.0, bdr_attributes="all")
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if NP == 1:
            setattr(pde, a, hm.LUSolver(COMM))
        else:
            s = hm.KrylovSolver(COMM, "cg", "amg")
            s.parameters["rel_tolerance"] = 1e-14
            setattr(pde, a, s)
    prior = hm.BiLaplacianPrior(Vm, 0.3, 3.0,
                                solver_type="lu" if NP == 1 else "krylov")

    # QoI: the spatial mean of the state, a linear functional
    qoi = hm.mean_state_qoi(Vu)
    p2q = hm.Parameter2QoiMap(pde, qoi)

    hm.parRandom.set_seed(5)
    m0 = prior.mean.copy()
    q0, g0 = p2q.reduced_gradient(m0)
    check("QoI evaluated", np.isfinite(q0) and q0 != 0.0, "(q = %.6e)" % q0)

    eps = np.power(0.5, np.arange(2, 18))
    e, eg, eH = hm.qoiVerify(p2q, m0, eps=eps, verbose=False)
    sg = hm.best_slope(e, eg)
    check("reduced QoI gradient is first order", 0.8 < sg < 1.3,
          "(slope %.3f)" % sg)
    sH = hm.best_slope(e, eH)
    check("reduced QoI Hessian is first order", 0.8 < sH < 1.3,
          "(slope %.3f)" % sH)

    # Hessian symmetry
    H = p2q.hessian(m0)
    a, b = Vm.vector(), Vm.vector()
    hm.parRandom.normal(1.0, a)
    hm.parRandom.normal(1.0, b)
    Ha, Hb = Vm.vector(), Vm.vector()
    H.mult(a, Ha)
    H.mult(b, Hb)
    s = abs(Ha.inner(b) - Hb.inner(a)) / max(abs(Ha.inner(b)), 1e-300)
    check("QoI Hessian symmetric", s < 1e-9, "(%.2e)" % s)

    # Taylor approximation and its analytic moments
    k = 25
    Omega = hm.MultiVector(m0, k)
    hm.parRandom.normal_multivector(1.0, Omega)
    tay = hm.TaylorApproximationQoi(p2q, prior)
    d, U = tay.computeLowRankFactorization(Omega, k=k, s=1)
    e1, e2 = tay.expectedValue(1), tay.expectedValue(2)
    v1, v2 = tay.variance(1), tay.variance(2)
    check("Taylor moments finite",
          all(np.isfinite([e1, e2, v1, v2])) and v1 > 0 and v2 > 0,
          "(E1 %.4e, E2 %.4e, V1 %.4e, V2 %.4e)" % (e1, e2, v1, v2))
    check("second-order variance >= first-order", v2 >= v1 - 1e-12)

    # the Taylor model must reproduce the map at the expansion point
    check("Taylor model exact at the mean",
          abs(tay.eval(m0, order=2) - tay.q_bar) < 1e-10 * max(abs(tay.q_bar), 1.0))

    # Monte Carlo with the Taylor control variate
    res = hm.varianceReductionMC(prior, p2q, tay, 150, order=2)
    check("variance reduction achieved", res["variance_reduction"] > 1.0,
          "(factor %.1f; sd_Q %.3e -> sd_diff %.3e)"
          % (res["variance_reduction"], res["sd_q"], res["sd_diff"]))
    dm = abs(res["reduced_mean"] - res["mc_mean"])
    tol = 4.0 * max(res["mc_stderr"], 1e-300)
    check("the two mean estimates agree within Monte Carlo error", dm < tol,
          "(|diff| %.3e vs 4*stderr %.3e)" % (dm, tol))
    if RANK == 0:
        print("      MC mean %.6e +- %.1e ; reduced %.6e +- %.1e ; Taylor %.6e"
              % (res["mc_mean"], res["mc_stderr"], res["reduced_mean"],
                 res["reduced_stderr"], res["taylor_mean"]))


def test_taylor_under_posterior():
    """The Taylor model of a QoI, expanded under a Laplace posterior.

    The posterior plays the prior's role in
    :class:`~hippymfem.forward_uq.taylorApproximationQoi.TaylorApproximationQoi`, but its
    precision operator carries both the precision (``mult``) and the covariance (``solve``),
    and the eigensolver takes anything with ``mult`` at face value.  Handing it the precision
    where the covariance is meant leaves the eigenvalues near zero and the second-order
    moments silently first-order, which is what this checks: the analytic moments of the
    quadratic model must match the moments of that same model over posterior samples, up to
    Monte Carlo error.
    """
    if RANK == 0:
        print("forward UQ: Taylor moments under a Laplace posterior")
    from hippymfem.algorithms.randomizedEigensolver import check_g
    from hippymfem.modeling.variables import ADJOINT, STATE

    model, Vh, _mtrue = build_linear_problem(nx=8)
    prior, Vm = model.prior, Vh[PARAMETER]

    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-10
    params["print_level"] = -1
    x = hm.ReducedSpaceNewtonCG(model, params).solve([None, prior.mean.copy(), None])
    model.solveAdj(x[ADJOINT], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = hm.ReducedHessian(model, misfit_only=True)

    # k below the parameter dimension: at k == n the last sketch directions are numerically
    # null and their B-norm is zero, which is a property of the sketch and not of the pair
    k = 55
    hm.parRandom.set_seed(11)
    Omega = hm.MultiVector(x[PARAMETER], k)
    hm.parRandom.normal_multivector(1.0, Omega)
    d, U = hm.doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)
    post = hm.GaussianLRPosterior(prior, d, U, mean=x[PARAMETER])

    qoi = hm.VariationalQoi(Vh, lambda u, m, xq: u.val * u.val)
    p2q = hm.Parameter2QoiMap(model.problem, qoi)
    Omega2 = hm.MultiVector(x[PARAMETER], k)
    hm.parRandom.normal_multivector(1.0, Omega2)
    tay = hm.TaylorApproximationQoi(p2q, post)
    dq, Uq = tay.computeLowRankFactorization(Omega2, k=k)

    # The eigen-residual is the check; Hlr-orthogonality is held to a looser tolerance
    # because on more than one rank the prior solves inside Hlr are Krylov with a stopping
    # tolerance, and that inexactness lands in the B inner products (1e-14 on one rank,
    # 1e-4 on two).  Handing the eigensolver the precision in the covariance's place gives
    # a residual of order 1 and an orthogonality error of 2, so both still separate it.
    resid, bortho, diag = check_g(tay.H, post.Hlr, Uq, dq)
    scale = max(float(np.abs(dq).max()), 1e-300)
    check("posterior-preconditioned eigenpairs solve H u = d Hlr u",
          resid.max() / scale < 1e-6 and bortho < 1e-2 and diag < 1e-8,
          "(residual %.2e, Hlr-orthogonality %.2e, diagonalization %.2e)"
          % (resid.max() / scale, bortho, diag))

    # the analytic moments are moments of *this* quadratic model, so sampling it checks them
    nsamples = 800
    q2 = np.zeros(nsamples)
    for i in range(nsamples):
        q2[i] = tay.eval(post.sample(), order=2)
    mean_s, var_s = float(q2.mean()), float(q2.var(ddof=1))
    stderr = float(q2.std(ddof=1)) / math.sqrt(nsamples)
    e1, e2, v2 = tay.expectedValue(1), tay.expectedValue(2), tay.variance(2)
    check("analytic mean of the quadratic model matches its samples",
          abs(e2 - mean_s) < 4.0 * stderr + 1e-12 * abs(e2),
          "(analytic %.6e, sampled %.6e +- %.1e)" % (e2, mean_s, stderr))
    check("analytic variance of the quadratic model matches its samples",
          abs(v2 - var_s) < 0.25 * max(var_s, 1e-300),
          "(analytic %.6e, sampled %.6e)" % (v2, var_s))
    # the second-order term is what the wrong operator would have thrown away
    check("the second-order correction is present",
          abs(e2 - e1) > 1e-3 * math.sqrt(max(v2, 0.0)),
          "(order 1 %.6e, order 2 %.6e)" % (e1, e2))


def test_mass_functional():
    """``mass_functional`` (a linear form) is the mass matrix's action, P1 and P2,
    for a constant and for a nodal indicator; ``weighted_mean_qoi`` normalizes it."""
    if RANK == 0:
        print("mass functional without the mass matrix")
    from hippymfem.fem.assemble import assemble_native_matrix
    from hippymfem.fem.assemble import mass_functional
    from hippymfem.fem.spaces import _ScalarPy
    from hippymfem.forward_uq.qoi import weighted_mean_qoi

    # straight and curved: MassIntegrator's rule is 2p + Trans.OrderW(), and a linear
    # form left at its own default integrates a curved element differently
    meshes = []
    for curvature in (0, 2):
        m = mfem.Mesh.MakeCartesian3D(3, 3, 3, mfem.Element.HEXAHEDRON)
        if curvature:
            m.SetCurvature(curvature)
            arr = m.GetNodes().GetDataArray()
            arr[:] = arr + 0.05 * np.random.default_rng(4).standard_normal(arr.size)
        meshes.append(mfem.ParMesh(COMM, m))
    worst = 0.0
    for pm, order in [(m, o) for m in meshes for o in (1, 2)]:
        V = hm.FunctionSpace.H1(pm, order)
        M = assemble_native_matrix(V, [mfem.MassIntegrator()])
        w = V.project(lambda z: ((np.abs(z[..., 0] - 0.5) < 0.3) & (z[..., 2] > 0.4)).astype(np.float64))
        one = V.vector()
        one.set(1.0)
        for field in (None, w):
            ref = V.vector()
            M.Mult((one if field is None else field).hypre, ref.hypre)
            ell = mass_functional(V, field)
            worst = max(worst, ell.copy().axpy(-1.0, ref).norm("linf") / ref.norm("linf"))
        # with a coefficient, as the mollified prior's right-hand side uses it: MFEM
        # adds nothing to the mass rule for a coefficient, so the two still agree
        cf = _ScalarPy(lambda x: 0.5 + x[0] * x[1])
        Mc = assemble_native_matrix(V, [mfem.MassIntegrator(cf)])
        ref = V.vector()
        Mc.Mult(w.hypre, ref.hypre)
        ell = mass_functional(V, w, coeff=cf)
        worst = max(worst, ell.copy().axpy(-1.0, ref).norm("linf") / ref.norm("linf"))
        q = weighted_mean_qoi(V, w)
        u = V.project(lambda z: 2.0 + z[..., 0])           # its w-weighted mean, exactly
        ref = V.vector()
        M.Mult(w.hypre, ref.hypre)
        expect = ref.inner(u) / ref.inner(one)
        worst = max(worst, abs(q.eval([u, None, None]) - expect) / abs(expect))
    check("mass functional and weighted mean match the matrix route", worst < 1e-13,
          "(worst rel %.1e, straight and curved hexahedra, P1 and P2)" % worst)


def test_variational_qoi():
    if RANK == 0:
        print("forward UQ: VariationalQoi")
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(8, 8,
                                                         mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, 2)
    Vm = hm.FunctionSpace.H1(pmesh, 1)

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hm.inner(u.grad, p.grad) - 1.0 * p.val

    def qoi_varf(u, m, x):
        """Dissipated power: integral of exp(m) |grad u|^2."""
        return jnp.exp(m.val) * hm.inner(u.grad, u.grad)

    bc = hm.DirichletBC(Vu, 0.0, bdr_attributes="all")
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True)
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        setattr(pde, a, hm.LUSolver(COMM) if NP == 1
                else hm.KrylovSolver(COMM, "cg", "amg"))
    qoi = hm.VariationalQoi([Vu, Vm, Vu], qoi_varf)
    p2q = hm.Parameter2QoiMap(pde, qoi)

    m0 = Vm.project(lambda x: 0.2 * np.sin(3 * x[0]))
    q, g = p2q.reduced_gradient(m0)
    check("variational QoI positive", q > 0, "(q = %.6e)" % q)
    eps = np.power(0.5, np.arange(2, 16))
    e, eg, eH = hm.qoiVerify(p2q, m0, eps=eps, verbose=False)
    check("variational QoI gradient is first order",
          0.8 < hm.best_slope(e, eg) < 1.3,
          "(slope %.3f)" % hm.best_slope(e, eg))
    check("variational QoI Hessian is first order",
          0.8 < hm.best_slope(e, eH) < 1.3,
          "(slope %.3f)" % hm.best_slope(e, eH))


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 74)
        print("hIPPyMFEM MCMC and forward-UQ tests on %d rank(s)" % NP)
        print("=" * 74)
    test_diagnostics()
    test_mcmc_pcn()
    test_mcmc_gpcn()
    test_mcmc_mala()
    test_forward_uq()
    test_taylor_under_posterior()
    test_mass_functional()
    test_variational_qoi()
    if RANK == 0:
        print("-" * 74)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

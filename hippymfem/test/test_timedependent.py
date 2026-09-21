# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Time-dependent problems.

The benchmark is initial-condition inversion for advection-diffusion, the
hIPPYlib ``ad_diff`` problem: the parameter is the initial state, observed only
at later times.  Run with
``mpirun -n N python -m hippymfem.test.test_timedependent``.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hm
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


def test_time_dependent_vector():
    if RANK == 0:
        print("TimeDependentVector")
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(6, 6, mfem.Element.TRIANGLE))
    V = hm.FunctionSpace.H1(pmesh, 1)
    times = np.linspace(0.0, 1.0, 6)
    u = hm.TimeDependentVector(times, comm=COMM).initialize(V)
    check("levels allocated", len(u) == 6 and u[0] is not None)

    hm.parRandom.set_seed(2)
    for k in range(6):
        hm.parRandom.normal(1.0, u[k])
    w = u.copy()
    check("copy is independent",
          abs(w.inner(u) - u.inner(u)) < 1e-12 * abs(u.inner(u)))
    w.axpy(-1.0, u)
    check("axpy", w.norm("linf", "l2") < 1e-14)

    v = V.vector()
    u.retrieve(v, times[3])
    check("retrieve by time", v.copy().axpy(-1.0, u[3]).norm("linf") < 1e-14)
    v.set(7.0)
    u.store(v, times[3])
    check("store by time", abs(u[3].max() - 7.0) < 1e-14)
    raised = False
    try:
        u.view(0.123456)
    except KeyError:
        raised = True
    check("unknown time is rejected", raised)

    ew = u.element_wise_inner(u)
    check("element_wise_inner", ew.size == 6 and np.all(ew >= 0))
    check("norm(l2, l2) consistent",
          abs(u.norm("l2", "l2") ** 2 - u.inner(u)) < 1e-10 * abs(u.inner(u)))


def build_ad_diff(nx=16, nt=8, kappa=0.02, order=1):
    """Initial-condition inversion for advection-diffusion.

    State: ``u(t)``; parameter: the initial condition ``m = u(0)``.  Observations
    are taken at the later time levels only, so the data must be propagated back
    through the PDE, the situation where the adjoint's reverse-in-time coupling
    matters.
    """
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(nx, nx,
                                                         mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, order)
    Vm = Vu                      # the parameter IS the initial state
    Vh = [Vu, Vm, Vu]
    t_init, t_final = 0.0, 0.4
    dt = (t_final - t_init) / nt
    vel = jnp.array([1.0, 0.4])

    def varf(u, u_old, m, p, x, t, dt_):
        """Implicit Euler for u_t + v.grad u - kappa lap u = 0."""
        return ((u.val - u_old.val) / dt_ * p.val
                + kappa * hm.inner(u.grad, p.grad)
                + jnp.dot(vel, u.grad) * p.val)

    bc = hm.DirichletBC(Vu, 0.0, bdr_attributes="all")
    u0 = Vu.vector()             # overwritten per parameter, see below
    pde = hm.TimeDependentPDEVariationalProblem(
        Vh, varf, bc, bc.homogeneous(), u0, t_init, t_final, dt,
        is_fwd_linear=True, quadrature_degree=2 * order + 2)
    if NP == 1:
        for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
            setattr(pde, a, hm.LUSolver(COMM))
    else:
        for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
            s = hm.KrylovSolver(COMM, "gmres", "amg")
            s.parameters["rel_tolerance"] = 1e-13
            setattr(pde, a, s)
    return pmesh, Vh, pde


class _ICProblem(hm.PDEProblem):
    """Wrap the time-dependent problem so the parameter sets the initial state.

    The only coupling is ``u0 = m``; everything else delegates.  Written here
    rather than in the library because how the parameter enters the initial
    condition is a modelling choice.
    """

    def __init__(self, pde):
        self.pde = pde
        self.Vh = pde.Vh
        self.times = pde.times

    def generate_state(self):
        return self.pde.generate_state()

    def generate_parameter(self):
        return self.pde.generate_parameter()

    def generate_adjoint(self):
        return self.pde.generate_adjoint()

    def init_parameter(self, m):
        return m

    def solveFwd(self, out, x):
        self.pde.u0 = x[PARAMETER].copy()
        self.pde.bc.zero(self.pde.u0)          # honour the essential conditions
        return self.pde.solveFwd(out, x)

    def solveAdj(self, out, x, adj_rhs):
        return self.pde.solveAdj(out, x, adj_rhs)

    def evalGradientParameter(self, x, out):
        r"""``dJ/dm`` = the residual gradient in ``m`` plus the initial-condition term.

        The residual does not depend on ``m`` here (the parameter enters only
        through ``u_0``), so the whole gradient is the adjoint at the first step
        acting through ``B_1``: :math:`B_1^{\!\top}p_1`.
        """
        out.zero()
        blk = self.pde._step_block(
            3, 1, x[STATE].view(self.times[1]), x[STATE].view(self.times[0]),
            x[PARAMETER], x[ADJOINT].view(self.times[1]), self.times[1],
            test_ess=self.pde.bc0.ess_tdof)
        blk.MultTranspose(x[ADJOINT].view(self.times[1]).hypre, out.hypre)
        self.pde.bc0.zero(out)
        self._keep = blk
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        return self.pde.setLinearizationPoint(x, gauss_newton_approx)

    def solveIncremental(self, out, rhs, is_adj):
        return self.pde.solveIncremental(out, rhs, is_adj)

    def apply_ij(self, i, j, dir, out):
        """``C`` carries the initial-condition coupling; the rest delegates."""
        if i == ADJOINT and j == PARAMETER:
            out.zero()
            blk = self.pde._blocks(1)["B"]
            blk.Mult(dir.hypre, out.view(self.times[1]).hypre)
            return out
        if i == PARAMETER and j == ADJOINT:
            out.zero()
            blk = self.pde._blocks(1)["B"]
            blk.MultTranspose(dir.view(self.times[1]).hypre, out.hypre)
            self.pde.bc0.zero(out)
            return out
        return self.pde.apply_ij(i, j, dir, out)


def test_forward_and_adjoint():
    if RANK == 0:
        print("time-dependent forward and adjoint")
    pmesh, Vh, pde = build_ad_diff()
    Vu = Vh[STATE]
    prob = _ICProblem(pde)

    m = Vu.project(lambda x: np.exp(-60.0 * ((x[0] - 0.25) ** 2
                                             + (x[1] - 0.35) ** 2)))
    pde.bc.zero(m)
    u = prob.generate_state()
    prob.solveFwd(u, [u, m, None])
    check("initial level equals the parameter",
          u.view(pde.times[0]).copy().axpy(-1.0, m).norm("linf") < 1e-14)
    peaks = [u[k].max() for k in range(len(pde.times))]
    check("solution decays in time", all(peaks[i] >= peaks[i + 1] - 1e-12
                                        for i in range(len(peaks) - 1)),
          "(peaks %s)" % np.array2string(np.array(peaks), precision=4))
    check("mass is bounded", 0 < u[-1].max() < u[0].max())

    # an adjoint solve over the whole trajectory, from a random right-hand side
    hm.parRandom.set_seed(3)
    rhs = prob.generate_adjoint()
    for k in range(1, len(pde.times)):
        hm.parRandom.normal(1.0, rhs[k])
        pde.bc0.zero(rhs[k])
    p = prob.generate_adjoint()
    prob.solveAdj(p, [u, m, None], rhs)
    check("adjoint solved", p.norm("linf", "l2") > 0)
    return pmesh, Vh, pde, prob, m


def test_td_gradient_and_hessian():
    if RANK == 0:
        print("time-dependent gradient and Hessian (finite differences)")
    pmesh, Vh, pde = build_ad_diff(nx=12, nt=6)
    Vu = Vh[STATE]
    prob = _ICProblem(pde)

    # Observe every level after the first, at many points, to keep the
    # initial-condition problem recoverable: observing only the last few levels
    # of an advection-diffusion run is so ill-posed that the MAP point is
    # dominated by the prior and an L2 error reduction is not a meaningful test.
    rng = np.random.default_rng(5)
    targets = rng.uniform(0.1, 0.9, size=(120, 2))
    B = hm.assemblePointwiseObservation(Vu, targets)
    misfits = [None] * pde.times.size
    for k in range(1, pde.times.size):
        # noise_variance is filled in below, once the data scale is known
        misfits[k] = hm.DiscreteStateObservation(B, B.createVecLeft(), None)
    misfit = hm.MisfitTD(misfits, pde.times)

    # Specify the prior by marginal variance and correlation length, matched to
    # the amplitude and width of the truth.  Raw (gamma, delta) picked by hand can
    # easily give a prior far too weak for CG to solve the resulting Newton system
    # (see applications/ad_diff).
    gamma, delta = hm.BiLaplacianComputeCoefficients(0.25, 0.15, 2)
    prior = hm.BiLaplacianPrior(Vu, gamma, delta,
                                solver_type="lu" if NP == 1 else "krylov")
    model = hm.Model(prob, prior, misfit)

    # synthetic data
    mtrue = Vu.project(lambda x: np.exp(-60.0 * ((x[0] - 0.3) ** 2
                                                 + (x[1] - 0.4) ** 2)))
    pde.bc.zero(mtrue)
    utrue = prob.generate_state()
    prob.solveFwd(utrue, [utrue, mtrue, None])
    clean_max = 0.0
    for k, t in enumerate(pde.times):
        if misfits[k] is not None:
            B.mult(utrue.view(t), misfits[k].d)
            clean_max = max(clean_max, misfits[k].d.norm("linf"))
    noise_std = 0.01 * max(clean_max, 1e-30)
    for k in range(pde.times.size):
        if misfits[k] is not None:
            B.perturb(misfits[k].d, noise_std)
            misfits[k].noise_variance = noise_std ** 2

    m0 = Vu.project(lambda x: 0.3 * np.exp(-40.0 * ((x[0] - 0.4) ** 2
                                                    + (x[1] - 0.4) ** 2)))
    pde.bc.zero(m0)
    eps = np.power(2.0, -np.arange(2, 16))
    e, eg, eH = hm.modelVerify(model, m0, is_quadratic=True, misfit_only=False,
                               verbose=(RANK == 0), eps=eps)
    sg = hm.best_slope(e, eg)
    check("time-dependent gradient is first order", 0.8 < sg < 1.3,
          "(slope %.3f)" % sg)
    # The forward map is linear in the initial condition and the misfit is
    # quadratic, so the cost is exactly quadratic in m and err_H is pure round-off.
    # A slope of 1 would be meaningless here, so the assertion is that the error
    # is negligible.
    scaleH = max(abs(eg[0]) / max(e[0], 1e-300), 1.0)
    check("time-dependent Hessian is exact (quadratic cost)",
          eH.max() < 1e-6 * scaleH,
          "(max err %.2e vs scale %.2e)" % (eH.max(), scaleH))

    # Hessian symmetry
    x = model.generate_vector()
    x[PARAMETER] = m0
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    H = hm.ReducedHessian(model, misfit_only=False)
    a, b = Vu.vector(), Vu.vector()
    hm.parRandom.normal(1.0, a)
    hm.parRandom.normal(1.0, b)
    pde.bc0.zero(a)
    pde.bc0.zero(b)
    Ha, Hb = Vu.vector(), Vu.vector()
    H.mult(a, Ha)
    H.mult(b, Hb)
    s = abs(Ha.inner(b) - Hb.inner(a)) / max(abs(Ha.inner(b)), 1e-300)
    check("time-dependent Hessian symmetric", s < 1e-8, "(%.2e)" % s)

    # MAP point
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-8
    params["max_iter"] = 25
    params["GN_iter"] = 25          # the cost is quadratic: Gauss-Newton is exact
    params["cg_max_iter"] = 400
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    xmap = solver.solve([None, prior.mean.copy(), None])
    check("Newton-CG converged on the time-dependent problem", solver.converged,
          "(%s, %d its)" % (solver.termination_reasons[solver.reason], solver.it))
    err = (mtrue.copy().axpy(-1.0, xmap[PARAMETER]).norm("l2")
           / max(mtrue.norm("l2"), 1e-300))
    err0 = (mtrue.copy().axpy(-1.0, prior.mean).norm("l2")
            / max(mtrue.norm("l2"), 1e-300))
    check("recovered initial condition beats the prior mean", err < 0.7 * err0,
          "(rel err %.4f vs %.4f)" % (err, err0))
    cmap = model.cost(xmap)          # collective: every rank
    if RANK == 0:
        print("      MAP: cost %.4e, misfit %.4e, reg %.4e"
              % (cmap[0], cmap[2], cmap[1]), flush=True)


# --------------------------------------------------- nonlinear, with a
# --------------------------------------------------- distributed parameter
def build_quasilinear(nx=10, nt=5, order=1):
    r"""Crank-Nicolson for a quasilinear parabolic problem with a parameter field.

    The advection-diffusion case above is **linear** in the state and its residual
    does not involve the parameter, so *every* second derivative of that residual
    vanishes: the cross-time Hessian blocks (:meth:`applyWuu`, :meth:`applyWum`,
    :meth:`applyWmu`, :meth:`applyWmm`) are identically zero, and the Hessian test
    there passes on the misfit's ``W_uu`` alone without exercising the time-coupling
    bookkeeping.

    This problem is built so that none of them vanish.  With
    :math:`u_{1/2} = (u_n + u_{n-1})/2`,

    .. math:: r_n = \frac{u_n-u_{n-1}}{\Delta t}p
              + (1 + u_{1/2}^2)\,\nabla u_{1/2}\cdot\nabla p
              + e^{m} u_{1/2} p - f p,

    every block is nonzero, and the cross-time ones
    (:math:`\partial^2 r/\partial u_n \partial u_{n-1}`) are nonzero precisely
    because both levels enter through :math:`u_{1/2}`.
    """
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(nx, nx,
                                                         mfem.Element.TRIANGLE))
    Vu = hm.FunctionSpace.H1(pmesh, order)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    t_final = 0.2
    dt = t_final / nt

    def varf(u, u_old, m, p, x, t, dt_):
        u_mid = 0.5 * (u.val + u_old.val)
        g_mid = 0.5 * (u.grad + u_old.grad)
        return ((u.val - u_old.val) / dt_ * p.val
                + (1.0 + u_mid ** 2) * hm.inner(g_mid, p.grad)
                + jnp.exp(m.val) * u_mid * p.val
                - 5.0 * p.val)

    bc = hm.DirichletBC(Vu, 0.0, bdr_attributes="all")
    u0 = Vu.vector()                       # start from rest
    pde = hm.TimeDependentPDEVariationalProblem(
        [Vu, Vm, Vu], varf, bc, bc.homogeneous(), u0, 0.0, t_final, dt,
        is_fwd_linear=False, quadrature_degree=2 * order + 3)
    pde.newton_parameters["print_level"] = -1
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if NP == 1:
            setattr(pde, a, hm.LUSolver(COMM))
        else:
            sv = hm.KrylovSolver(COMM, "gmres", "amg")
            sv.parameters["rel_tolerance"] = 1e-13
            setattr(pde, a, sv)
    return pmesh, Vu, Vm, pde


def test_nonlinear_time_dependent_hessian():
    """Exercise the cross-time Hessian blocks, which the linear case cannot."""
    if RANK == 0:
        print("time-dependent: nonlinear problem with a distributed parameter")
    pmesh, Vu, Vm, pde = build_quasilinear()

    # observe the state everywhere after the first level
    rng = np.random.default_rng(9)
    targets = rng.uniform(0.15, 0.85, size=(40, 2))
    B = hm.assemblePointwiseObservation(Vu, targets)
    misfits = [None] * pde.nt
    for k in range(1, pde.nt):
        misfits[k] = hm.DiscreteStateObservation(B, B.createVecLeft(), None)
    misfit = hm.MisfitTD(misfits, pde.times)
    gamma, delta = hm.BiLaplacianComputeCoefficients(0.25, 0.3, 2)
    prior = hm.BiLaplacianPrior(Vm, gamma, delta,
                                solver_type="lu" if NP == 1 else "krylov")
    model = hm.Model(pde, prior, misfit)

    mtrue = Vm.project(lambda x: 0.5 * np.sin(2 * np.pi * x[0]) * np.cos(np.pi * x[1]))
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    check("nonlinear forward solve produced a nonzero trajectory",
          utrue.norm("linf", "linf") > 0.1,
          "(peak %.4f)" % utrue.norm("linf", "linf"))

    hm.parRandom.set_seed(4)
    clean_max = 0.0
    for k in range(1, pde.nt):
        B.mult(utrue.view(pde.times[k]), misfits[k].d)
        clean_max = max(clean_max, misfits[k].d.norm("linf"))
    noise_std = 0.01 * max(clean_max, 1e-30)
    for k in range(1, pde.nt):
        B.perturb(misfits[k].d, noise_std)
        misfits[k].noise_variance = noise_std ** 2

    m0 = Vm.project(lambda x: 0.2 * np.cos(np.pi * x[0]))
    x = model.generate_vector()
    x[PARAMETER] = m0
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)

    # The blocks must actually be nonzero, or this test would silently degenerate
    # into the linear case it exists to escape.
    du = pde.generate_state()
    dm = Vm.vector()
    hm.parRandom.normal(1.0, dm)
    for k in range(1, pde.nt):
        hm.parRandom.normal(1.0, du.view(pde.times[k]))
        pde.bc0.zero(du.view(pde.times[k]))
    outu = pde.generate_state()
    outm = Vm.vector()
    pde.applyWuu(du, outu)
    check("cross-time W_uu is nonzero", outu.norm("linf", "linf") > 0,
          "(%.3e)" % outu.norm("linf", "linf"))
    pde.applyWum(dm, outu)
    check("cross-time W_um is nonzero", outu.norm("linf", "linf") > 0,
          "(%.3e)" % outu.norm("linf", "linf"))
    pde.applyWmu(du, outm)
    check("W_mu is nonzero", outm.norm("linf") > 0, "(%.3e)" % outm.norm("linf"))
    pde.applyWmm(dm, outm)
    check("W_mm is nonzero", outm.norm("linf") > 0, "(%.3e)" % outm.norm("linf"))

    # W_um and W_mu must be transposes: this is what tests the row-n / row-(n-1)
    # accumulation, since each step writes into two different time levels
    pde.applyWum(dm, outu)
    lhs = outu.inner(du)
    pde.applyWmu(du, outm)
    rhs = outm.inner(dm)
    check("time-dependent W_um and W_mu are transposes",
          abs(lhs - rhs) <= 1e-9 * max(1.0, abs(lhs)),
          "(%.8e vs %.8e)" % (lhs, rhs))

    # the reduced Hessian must be symmetric with all blocks live
    H = hm.ReducedHessian(model, misfit_only=False)
    a, b = Vm.vector(), Vm.vector()
    hm.parRandom.normal(1.0, a)
    hm.parRandom.normal(1.0, b)
    Ha, Hb = Vm.vector(), Vm.vector()
    H.mult(a, Ha)
    H.mult(b, Hb)
    sym = abs(Ha.inner(b) - Hb.inner(a)) / max(abs(Ha.inner(b)), 1e-300)
    check("nonlinear time-dependent Hessian symmetric", sym < 1e-8,
          "(%.2e)" % sym)

    # and the full finite-difference check: here the Hessian error must fall like
    # h rather than being zero
    eps = np.power(2.0, -np.arange(3, 14))
    e, eg, eH = hm.modelVerify(model, m0, misfit_only=False,
                               verbose=(RANK == 0), eps=eps)
    sg, sH = hm.best_slope(e, eg), hm.best_slope(e, eH)
    check("nonlinear time-dependent gradient is first order", 0.7 < sg < 1.4,
          "(slope %.3f)" % sg)
    check("nonlinear time-dependent Hessian is first order", 0.7 < sH < 1.4,
          "(slope %.3f)" % sH)

    # and Newton-CG on it
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-7
    params["max_iter"] = 25
    params["GN_iter"] = 3
    params["cg_max_iter"] = 200
    params["print_level"] = -1
    solver = hm.ReducedSpaceNewtonCG(model, params)
    xm = solver.solve([None, prior.mean.copy(), None])
    check("Newton-CG converged on the nonlinear time-dependent problem",
          solver.converged,
          "(%s, %d its)" % (solver.termination_reasons[solver.reason], solver.it))
    err = (mtrue.copy().axpy(-1.0, xm[PARAMETER]).norm("l2")
           / max(mtrue.norm("l2"), 1e-300))
    if RANK == 0:
        print("      relative error in the parameter: %.4f" % err)


def test_scalar_parameters_integrate():
    """``t`` and ``dt`` reach the density as true scalars, so it integrates exactly.

    A shape-(1,) parameter makes the whole density shape (nq, 1) by broadcasting, and
    the quadrature sum becomes an unweighted one times the element volume: every
    derivative stays consistent with the residual, so only an absolute integral with a
    non-constant integrand catches it.
    """
    if RANK == 0:
        print("scalar parameters integrate exactly")
    from hippymfem.fem.kernel import _params
    from hippymfem.modeling.TimeDependentPDEVariationalProblem import ADJ
    check("scalar parameters are 0-d", all(np.shape(v) == () for v in _params((0.5, 0.25))))
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.HEXAHEDRON))
    V = hm.FunctionSpace.H1(pmesh, 1)
    t, dt = 0.5, 0.25
    cases = [("t p", lambda u, uo, m, p, x, t_, dt_: t_ * p.val, t),
             ("x0^4 p + 0 dt", lambda u, uo, m, p, x, t_, dt_: x[0] ** 4 * p.val + 0.0 * dt_, 0.2),
             ("x0^4 p / dt", lambda u, uo, m, p, x, t_, dt_: x[0] ** 4 * p.val / dt_, 0.2 / dt)]
    for name, varf, exact in cases:
        pde = hm.TimeDependentPDEVariationalProblem([V, V, V], varf, None, None, V.vector(),
                                                    0.0, t, dt)
        r = pde._step_residual(ADJ, V.vector(), V.vector(), V.vector(), V.vector(), t)
        total = COMM.allreduce(float(r.array.sum()), op=MPI.SUM)
        check("integral of %s" % name, abs(total - exact) < 1e-12 * max(1.0, abs(exact)),
              "(%.15g, exact %.15g)" % (total, exact))


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 74)
        print("hIPPyMFEM time-dependent tests on %d rank(s)" % NP)
        print("=" * 74)
    test_time_dependent_vector()
    test_forward_and_adjoint()
    test_td_gradient_and_hessian()
    test_nonlinear_time_dependent_hessian()
    test_scalar_parameters_integrate()
    if RANK == 0:
        print("-" * 74)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

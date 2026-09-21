# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Forward and adjoint solves, and the incremental systems.

Run with ``mpirun -n N python -m hippymfem.test.test_solves``.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hp
from hippymfem.algorithms.linSolvers import KrylovSolver, LUSolver
from hippymfem.fem.bcs import BCSet, DirichletBC
from hippymfem.fem.jaxops import inner
from hippymfem.fem.spaces import FunctionSpace
from hippymfem.modeling.PDEVariationalProblem import PDEVariationalProblem
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


def poisson_setup(n=16, order=2):
    """-div(exp(m) grad u) = f, u = g on top and bottom."""
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE))
    Vu = FunctionSpace.H1(pmesh, order)
    Vm = FunctionSpace.H1(pmesh, 1)
    Vh = [Vu, Vm, Vu]

    def varf(u, m, p, x):
        return jnp.exp(m.val) * inner(u.grad, p.grad) - 0.0 * p.val

    # MakeCartesian2D attributes: 1 bottom, 2 right, 3 top, 4 left
    bc = BCSet([DirichletBC(Vu, lambda x: x[1], bdr_attributes=[1, 3])])
    bc0 = bc.homogeneous()
    pde = PDEVariationalProblem(Vh, varf, bc, bc0, is_fwd_linear=True)
    pde.solver = KrylovSolver(COMM, "cg", "amg")
    pde.solver.parameters["rel_tolerance"] = 1e-14
    pde.solver_fwd_inc = KrylovSolver(COMM, "cg", "amg")
    pde.solver_adj_inc = KrylovSolver(COMM, "cg", "amg")
    for s in (pde.solver_fwd_inc, pde.solver_adj_inc):
        s.parameters["rel_tolerance"] = 1e-14
    return pmesh, Vh, pde


def test_linear_forward():
    if RANK == 0:
        print("linear forward solve")
    pmesh, Vh, pde = poisson_setup()
    Vu, Vm = Vh[STATE], Vh[PARAMETER]

    # m == 0 so exp(m) == 1: the exact solution is u = y
    m = Vm.vector()
    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])
    exact = Vu.project(lambda x: x[1])
    e = u.copy().axpy(-1.0, exact).norm("linf")
    check("u == y for m = 0", e < 1e-9, "(%.2e)" % e)

    # nonconstant m: check the residual is zero and the BCs hold
    m = Vm.project(lambda x: 0.5 * np.sin(4 * x[0]) * np.cos(3 * x[1]))
    u.zero()
    pde.solveFwd(u, [u, m, None])
    r = pde._residual([u, m, None], ADJOINT, ess=pde.bc0.ess)
    check("residual zero after solve", r.norm("l2") < 1e-8, "(%.2e)" % r.norm("l2"))
    ub = u.copy()
    g = Vu.project(lambda x: x[1])
    if len(pde.bc.ess):
        bad = np.abs(ub.array[pde.bc.ess] - g.array[pde.bc.ess]).max()
    else:
        bad = 0.0
    check("boundary data satisfied", COMM.allreduce(bad, op=MPI.MAX) < 1e-12)
    return pmesh, Vh, pde, u, m


def test_adjoint_and_incremental():
    if RANK == 0:
        print("adjoint and incremental solves")
    pmesh, Vh, pde = poisson_setup(n=12)
    Vu, Vm = Vh[STATE], Vh[PARAMETER]
    m = Vm.project(lambda x: 0.3 * np.sin(3 * x[0]))
    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])

    # adjoint with an arbitrary rhs
    hp.parRandom.set_seed(31)
    rhs = Vu.vector()
    hp.parRandom.normal(1.0, rhs)
    pde.bc0.zero(rhs)
    p = pde.generate_adjoint()
    pde.solveAdj(p, [u, m, None], rhs)
    pde.setLinearizationPoint([u, m, p], gauss_newton_approx=False)
    # A^T p == rhs
    chk = Vu.vector()
    pde.apply_ij(STATE, ADJOINT, p, chk)
    e = chk.copy().axpy(-1.0, rhs).norm("l2") / max(rhs.norm("l2"), 1e-300)
    check("A^T p == adj_rhs", e < 1e-9, "(%.2e)" % e)

    # incremental forward: A du == rhs
    du = Vu.vector()
    pde.solveIncremental(du, rhs, False)
    pde.apply_ij(ADJOINT, STATE, du, chk)
    e = chk.copy().axpy(-1.0, rhs).norm("l2") / max(rhs.norm("l2"), 1e-300)
    check("incremental forward consistent with A", e < 1e-9, "(%.2e)" % e)

    # incremental adjoint: A^T dp == rhs
    dp = Vu.vector()
    pde.solveIncremental(dp, rhs, True)
    pde.apply_ij(STATE, ADJOINT, dp, chk)
    e = chk.copy().axpy(-1.0, rhs).norm("l2") / max(rhs.norm("l2"), 1e-300)
    check("incremental adjoint consistent with A^T", e < 1e-9, "(%.2e)" % e)

    # adjoint identity across the C block
    dm = Vm.vector()
    hp.parRandom.normal(1.0, dm)
    Cdm = Vu.vector()
    pde.apply_ij(ADJOINT, PARAMETER, dm, Cdm)
    dpv = Vu.vector()
    hp.parRandom.normal(1.0, dpv)
    pde.bc0.zero(dpv)
    Ctdp = Vm.vector()
    pde.apply_ij(PARAMETER, ADJOINT, dpv, Ctdp)
    lhs, rhs2 = Cdm.inner(dpv), Ctdp.inner(dm)
    check("<C dm, dp> == <dm, C^T dp>",
          abs(lhs - rhs2) <= 1e-11 * max(1.0, abs(lhs)),
          "(%.6e vs %.6e)" % (lhs, rhs2))


def test_gradient_by_fd():
    """The adjoint gradient of a simple functional vs finite differences.

    Uses J(m) = 1/2 ||u(m)||_M^2 so the whole adjoint machinery is exercised
    without the misfit and prior layers.
    """
    if RANK == 0:
        print("adjoint gradient vs finite differences")
    pmesh, Vh, pde = poisson_setup(n=10, order=1)
    Vu, Vm = Vh[STATE], Vh[PARAMETER]
    a = mfem.ParBilinearForm(Vu.fes)
    a.AddDomainIntegrator(mfem.MassIntegrator())
    a.Assemble()
    a.Finalize()
    M = a.ParallelAssemble()

    def cost_and_grad(m):
        u = pde.generate_state()
        pde.solveFwd(u, [u, m, None])
        Mu = Vu.vector()
        M.Mult(u.hypre, Mu.hypre)
        J = 0.5 * Mu.inner(u)
        # adjoint rhs = -dJ/du = -M u
        rhs = Mu.copy().scale(-1.0)
        p = pde.generate_adjoint()
        pde.solveAdj(p, [u, m, None], rhs)
        g = Vm.vector()
        pde.evalGradientParameter([u, m, p], g)
        return J, g

    hp.parRandom.set_seed(17)
    m0 = Vm.project(lambda x: 0.2 * np.cos(2 * x[0] + x[1]))
    dm = Vm.vector()
    hp.parRandom.normal(1.0, dm)
    J0, g0 = cost_and_grad(m0)
    gdm = g0.inner(dm)

    if RANK == 0:
        print("      J0 = %.12e   g.dm = %.12e" % (J0, gdm))
        print("      %-10s %-16s %-10s" % ("h", "FD", "rel err"))
    errs = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        Jp, _ = cost_and_grad(m0.copy().axpy(h, dm))
        Jm, _ = cost_and_grad(m0.copy().axpy(-h, dm))
        fd = (Jp - Jm) / (2 * h)
        rel = abs(fd - gdm) / max(abs(gdm), 1e-300)
        errs.append(rel)
        if RANK == 0:
            print("      %-10.0e %-16.9e %.3e" % (h, fd, rel))
    check("adjoint gradient matches FD", min(errs) < 1e-7, "(best %.2e)" % min(errs))


def test_nonlinear_forward():
    if RANK == 0:
        print("nonlinear forward solve (Newton)")
    n = 12
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE))
    Vu = FunctionSpace.H1(pmesh, 1)
    Vm = FunctionSpace.H1(pmesh, 1)

    # -div((1 + u^2) grad u) = 10 exp(m), u = 0 on the whole boundary
    def varf(u, m, p, x):
        return (1.0 + u.val ** 2) * inner(u.grad, p.grad) - 10.0 * jnp.exp(m.val) * p.val

    bc = BCSet([DirichletBC(Vu, 0.0, bdr_attributes="all")])
    pde = PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(),
                                is_fwd_linear=False)
    pde.newton_parameters["print_level"] = -1
    m = Vm.project(lambda x: 0.2 * x[0])
    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])
    r = pde._residual([u, m, None], ADJOINT, ess=pde.bc0.ess)
    check("Newton converged", r.norm("l2") < 1e-9,
          "(||r||=%.2e in %d iters)" % (r.norm("l2"), pde.fwd_iterations))
    check("solution is positive and bounded", 0.0 < u.max() < 10.0, "(max %.4f)" % u.max())

    # a mislabeled linear problem must be reported, not silently mis-solved
    pde2 = PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(),
                                 is_fwd_linear=True)
    u2 = pde2.generate_state()
    raised = False
    try:
        pde2.solveFwd(u2, [u2, m, None])
    except RuntimeError as exc:
        raised = "not linear in the state" in str(exc)
    check("is_fwd_linear mislabel is detected", raised)


def test_lu_solver():
    if NP > 1:
        if RANK == 0:
            print("LU solver (skipped: serial only)")
        return
    print("LU solver")
    pmesh, Vh, pde = poisson_setup(n=8, order=1)
    pde.solver = LUSolver(COMM)
    pde.solver_fwd_inc = LUSolver(COMM)
    pde.solver_adj_inc = LUSolver(COMM)
    m = Vh[PARAMETER].vector()
    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])
    exact = Vh[STATE].project(lambda x: x[1])
    e = u.copy().axpy(-1.0, exact).norm("linf")
    check("LU forward solve exact", e < 1e-12, "(%.2e)" % e)


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 74)
        print("hIPPyMFEM solve tests on %d rank(s)" % NP)
        print("=" * 74)
    test_linear_forward()
    test_adjoint_and_incremental()
    test_gradient_by_fd()
    test_nonlinear_forward()
    test_lu_solver()
    if RANK == 0:
        print("-" * 74)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

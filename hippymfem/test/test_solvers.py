# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Exact linear solves in parallel, and the PETSc bridge when it is available.

The standard parallel PyMFEM build links no direct solver, so an exact solve (which
hIPPYlib gets from ``PETScLUSolver`` and relies on whenever a Hessian action or a
spectrum should measure the discretization rather than a Krylov tolerance) has to
come from elsewhere.  What is checked here:

* ``LUSolver`` / ``ReplicatedLUSolver`` solves to machine precision on any rank
  count, and gives the **same answer** on 1, 2 and 4 ranks (the answers are
  written to a file the suite compares across runs);
* the gathered matrix really is the distributed one (compared against the dense
  form);
* transpose solves, the size guard, and singular-matrix behaviour;
* the whole inverse problem run with exact solves in parallel reproduces the MAP
  point the iterative solvers find;
* PETSc, if and only if petsc4py could be imported before PyMFEM, skipped with
  the reason printed otherwise, never silently.

Run with ``mpirun -n N python -m hippymfem.test.test_solvers``.
"""

import json
import os

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hp
from hippymfem.algorithms.directSolvers import (
    ReplicatedLUSolver,
    gather_matrix,
    petsc_available,
)
from hippymfem.common.linalg import to_dense
from hippymfem.common.parvector import ParVector
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
FAILS = []
REF = "/tmp/_hippymfem_solver_ref.json"


def check(name, ok, detail=""):
    if RANK == 0:
        print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


def make_operator(n=10, order=2):
    """An SPD ``HypreParMatrix`` (diffusion + reaction) and a right-hand side."""
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE))
    V = hp.FunctionSpace.H1(pm, order)
    form = mfem.ParBilinearForm(V.fes)
    form.AddDomainIntegrator(mfem.DiffusionIntegrator())
    form.AddDomainIntegrator(mfem.MassIntegrator())
    form.Assemble()
    form.Finalize()
    A = mfem.HypreParMatrix()
    bc = hp.DirichletBC(V, None, "all")
    form.FormSystemMatrix(bc.ess_tdof, A)
    A._keep = (form, V, pm)
    # The right-hand side is the interpolant of a fixed analytic function, not a
    # random dof vector: hypre numbers true dofs rank by rank, so a vector keyed
    # on the global dof index is a *different function* at a different rank count
    # and nothing about it would be comparable across runs.
    b = V.project(lambda x: np.sin(3.0 * x[0]) * np.cos(2.0 * x[1]) + x[0])
    bc.zero(b)
    return A, b, V


def residual(A, x, b):
    r = ParVector(COMM, A.Height())
    A.Mult(x.hypre, r.hypre)
    r.axpy(-1.0, b)
    return r.norm("l2") / max(b.norm("l2"), 1e-300)


def test_gather():
    """The replicated matrix must be the distributed one, not a rank's piece."""
    if RANK == 0:
        print("matrix replication")
    A, _, _ = make_operator(6, 1)
    D = to_dense(A, COMM)
    G = gather_matrix(A, COMM, format="csr")
    err = float(np.abs(G.toarray() - D).max()) / max(float(np.abs(D).max()), 1e-300)
    # every rank must hold the same copy
    h = float(np.abs(G.toarray()).sum())
    same = COMM.allreduce(abs(h - COMM.bcast(h, root=0)), op=MPI.MAX)
    check("gathered matrix equals the dense distributed matrix", err == 0.0,
          "(%.3e, shape %s)" % (err, G.shape))
    check("every rank gathers the same matrix", same == 0.0)


def test_exact_solve():
    """Exact to machine precision, and the same on every rank count."""
    if RANK == 0:
        print("exact solves on %d rank(s)" % NP)
    A, b, V = make_operator(10, 2)
    out = {}
    for name, solver in (("LUSolver", hp.LUSolver(COMM)),
                         ("ReplicatedLUSolver", ReplicatedLUSolver(COMM))):
        solver.set_operator(A)
        x = V.vector()
        solver.solve(x, b)
        rel = residual(A, x, b)
        check("%s residual is at round-off" % name, rel < 1e-12, "(%.3e)" % rel)
        out[name] = float(x.inner(b))

        # transpose solve: A is symmetric here, so it must give the same vector
        xt = V.vector()
        solver.solveTranspose(xt, b)
        xt.axpy(-1.0, x)
        d = xt.norm("l2") / max(x.norm("l2"), 1e-300)
        check("%s transpose solve matches (symmetric A)" % name, d < 1e-12,
              "(%.3e)" % d)

    agree = abs(out["LUSolver"] - out["ReplicatedLUSolver"])
    check("LUSolver and ReplicatedLUSolver agree", agree < 1e-9 * max(
        abs(out["LUSolver"]), 1e-300), "(%.3e)" % agree)

    # rank-count independence: store on the first run, compare on later ones
    key = "exact_solve_bTx_n10_p2"
    val = out["ReplicatedLUSolver"]
    if RANK == 0:
        ref = {}
        if os.path.exists(REF):
            try:
                ref = json.load(open(REF))
            except Exception:
                ref = {}
        if key in ref:
            rel = abs(val - ref[key]) / max(abs(ref[key]), 1e-300)
            msg = "(%d ranks vs stored %s: rel %.3e)" % (
                NP, ref.get(key + "_np", "?"), rel)
            ok = rel < 1e-11
        else:
            ref[key] = val
            ref[key + "_np"] = NP
            json.dump(ref, open(REF, "w"))
            ok, msg = True, "(stored reference from %d rank(s))" % NP
    else:
        ok, msg = True, ""
    ok = bool(COMM.bcast(ok, root=0))
    check("exact solve is rank-count independent", ok, COMM.bcast(msg, root=0))
    # b^T A^{-1} b is a sum over dofs of two partition-independent functions, so
    # it is invariant under the renumbering a different partition brings.


def test_guards():
    """A refused solve must say why; a singular matrix must not return garbage."""
    if RANK == 0:
        print("guards")
    A, b, V = make_operator(6, 1)
    s = ReplicatedLUSolver(COMM, max_global_size=4)
    raised = False
    try:
        s.set_operator(A)
    except RuntimeError as exc:
        raised = "max_global_size" in str(exc)
    check("oversized matrix is refused with an explanation", raised)

    # a zero matrix is singular: scipy raises, which must not be swallowed
    Z = mfem.HypreParMatrix()
    form = mfem.ParBilinearForm(V.fes)
    form.Assemble()
    form.Finalize()
    form.FormSystemMatrix(mfem.intArray(), Z)
    Z._keep = form
    s2 = ReplicatedLUSolver(COMM)
    failed = False
    try:
        s2.set_operator(Z)
        x = V.vector()
        s2.solve(x, b)
    except Exception:
        failed = True
    check("a singular matrix raises rather than returning garbage", failed)


def test_inverse_problem():
    """An inverse problem solved with exact solves, in parallel.

    ``LUSolver`` is exact on any rank count; the MAP point must agree with the
    one the iterative solvers reach.
    """
    if RANK == 0:
        print("inverse problem with exact solves")
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(12, 12, mfem.Element.TRIANGLE))
    Vu = hp.FunctionSpace.H1(pm, 2)
    Vm = hp.FunctionSpace.H1(pm, 1)

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * hp.inner(u.grad, p.grad)

    bc = hp.DirichletBC(Vu, lambda x: x[1], bdr_attributes=[1, 3])

    def build(kind):
        pde = hp.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc,
                                       bc.homogeneous(), is_fwd_linear=True)
        for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
            if kind == "lu":
                setattr(pde, a, hp.LUSolver(COMM))
            else:
                s = hp.KrylovSolver(COMM, "cg", "amg")
                s.parameters["rel_tolerance"] = 1e-13
                s.parameters["max_iter"] = 2000
                setattr(pde, a, s)
        prior = hp.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True,
                                    solver_type="lu" if kind == "lu" else "krylov")
        rng = np.random.default_rng(5)
        targets = np.column_stack((rng.uniform(0.1, 0.9, 40),
                                   rng.uniform(0.1, 0.5, 40)))
        B = hp.assemblePointwiseObservation(Vu, targets)
        hp.parRandom.set_seed(5)
        mtrue = Vm.project(lambda x: 1.0 + 0.5 * np.sin(3.0 * x[0]) * np.cos(2.0 * x[1]))
        utrue = pde.generate_state()
        pde.solveFwd(utrue, [utrue, mtrue, None])
        data = B.createVecLeft()
        B.mult(utrue, data)
        std = 0.01 * max(data.norm("linf"), 1e-30)
        B.perturb(data, std)
        misfit = hp.DiscreteStateObservation(B, data, std ** 2)
        return hp.Model(pde, prior, misfit)

    maps = {}
    for kind in ("lu", "krylov"):
        m = build(kind)
        p = hp.ReducedSpaceNewtonCG_ParameterList()
        p["rel_tolerance"] = 1e-9
        p["max_iter"] = 30
        p["print_level"] = -1
        # The last Newton steps predict cost decreases |(g, dm)| of 1e-13 and below on a
        # cost of about 11, under what double precision resolves, so the Armijo test
        # cannot confirm them, and whether an iterate reaches 1e-9 first is decided by
        # round-off, which differs between machines.  hIPPYlib's gdm_tolerance is the
        # criterion for this case; its 1e-18 default ignores the scale of the cost,
        # and 1e-12 is still far below any step that changes it.
        p["gdm_tolerance"] = 1e-12
        s = hp.ReducedSpaceNewtonCG(m, p)
        x = s.solve([None, m.prior.mean.copy(), None])
        check("Newton-CG converged with %s solves" % kind, s.converged,
              "(%d its, ||g||/||g0|| = %.2e)"
              % (s.it, s.final_grad_norm / max(s.initial_grad_norm, 1e-300)))
        maps[kind] = x[PARAMETER].copy()

    d = maps["lu"].copy()
    d.axpy(-1.0, maps["krylov"])
    rel = d.norm("l2") / max(maps["krylov"].norm("l2"), 1e-300)
    check("exact and iterative solves reach the same MAP point", rel < 1e-5,
          "(rel %.3e)" % rel)


def test_petsc():
    """PETSc, or a printed reason why not."""
    if RANK == 0:
        print("PETSc bridge")
    ok, reason = petsc_available()
    if not ok:
        if RANK == 0:
            print("      skipped: %s" % reason, flush=True)
        return
    from hippymfem.algorithms.directSolvers import (
        PETScKrylovSolver, PETScLUSolver, to_petsc_matrix)

    A, b, V = make_operator(10, 2)
    M = to_petsc_matrix(A, COMM)
    x = V.vector()
    exact = ReplicatedLUSolver(COMM)
    exact.set_operator(A)
    exact.solve(x, b)

    lu = PETScLUSolver(COMM)
    lu.set_operator(A)
    xl = V.vector()
    lu.solve(xl, b)
    rel = residual(A, xl, b)
    check("PETScLUSolver residual is at round-off", rel < 1e-11,
          "(%.3e, package %s)" % (rel, lu.package_used))
    d = xl.copy()
    d.axpy(-1.0, x)
    agree = d.norm("l2") / max(x.norm("l2"), 1e-300)
    check("PETScLUSolver agrees with the replicated LU", agree < 1e-9,
          "(%.3e)" % agree)

    kry = PETScKrylovSolver(COMM, "cg", "gamg")
    kry.parameters["rel_tolerance"] = 1e-12
    kry.set_operator(A)
    xk = V.vector()
    its = kry.solve(xk, b)
    rel = residual(A, xk, b)
    check("PETScKrylovSolver converges", rel < 1e-9, "(%.3e, %d its)" % (rel, its))


def main():
    test_gather()
    test_exact_solve()
    test_guards()
    test_inverse_problem()
    test_petsc()
    COMM.Barrier()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

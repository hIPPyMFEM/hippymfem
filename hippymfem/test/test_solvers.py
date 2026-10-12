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
import math
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
                         ("ReplicatedLUSolver", ReplicatedLUSolver(COMM)),
                         ("ReplicatedLUSolver(replicate=True)",
                          ReplicatedLUSolver(COMM, replicate=True))):
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
    agree = abs(out["ReplicatedLUSolver"] - out["ReplicatedLUSolver(replicate=True)"])
    check("the rank-0 and the replicated factorization agree", agree < 1e-12 * max(
        abs(out["ReplicatedLUSolver"]), 1e-300), "(%.3e)" % agree)
    # by default only rank 0 holds a factorization; every rank holds one when asked
    s0 = ReplicatedLUSolver(COMM).set_operator(A)
    s1 = ReplicatedLUSolver(COMM, replicate=True).set_operator(A)
    held = COMM.allreduce(int(s0._lu is not None), op=MPI.SUM)
    held_all = COMM.allreduce(int(s1._lu is not None), op=MPI.SUM)
    check("the factorization lives on rank 0 only by default, on every rank replicated",
          held == 1 and held_all == NP, "(%d and %d of %d ranks)" % (held, held_all, NP))

    # release gives the factorization back on every rank; the next set_operator rebuilds it
    for name, solver in (("LUSolver", hp.LUSolver(COMM)),
                         ("ReplicatedLUSolver", ReplicatedLUSolver(COMM))):
        solver.set_operator(A)
        solver.release()
        mine = (getattr(solver, "_lu", None) is None
                and getattr(solver, "_replicated", None) is None)
        gone = COMM.allreduce(int(mine), op=MPI.SUM) == NP
        refused = False
        try:
            solver.solve(V.vector(), b)
        except RuntimeError:
            refused = True
        solver.set_operator(A)
        x = V.vector()
        solver.solve(x, b)
        rel = residual(A, x, b)
        check("%s gives its factorization back on release and rebuilds it" % name,
              gone and refused and rel < 1e-12, "(%.3e)" % rel)

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


def test_hypre_pool_logic():
    """The recycling pool for hypre's device memory, against a stand-in for the driver:
    exact new blocks, recycling within the size bound, the two limits, and trim.  The
    pool itself needs a GPU (``test_device``); its bookkeeping does not."""
    import ctypes
    import types

    from hippymfem.common.mfemconfig import HyprePool

    live, nxt = {}, [0x10000]

    class Fn:                                  # takes the place of a ctypes function
        def __init__(self, f):
            self.f = f

        def __call__(self, *a):
            return self.f(*a)

    def dev_malloc(pp, n):
        p = nxt[0]
        nxt[0] += 0x100000
        live[p] = n
        ctypes.cast(pp, ctypes.POINTER(ctypes.c_void_p))[0] = p
        return 0

    def dev_free(p):
        live.pop(p if isinstance(p, int) else p.value)
        return 0

    runtime = types.SimpleNamespace(cudaMalloc=Fn(dev_malloc), cudaFree=Fn(dev_free))
    hypre = types.SimpleNamespace(hypre_SetUserDeviceMalloc=lambda f: None,
                                  hypre_SetUserDeviceMfree=lambda f: None)
    pool = HyprePool(hypre, runtime, max_cached=3000, max_block=2000)
    out = (ctypes.c_void_p * 1)()

    def alloc(n):
        pool._malloc(out, n)
        return out[0]

    p1, p2 = alloc(1000), alloc(1000)
    pool._free(p1)
    ok = pool.cached == 1000 and pool.in_use == 1000 and live[p1] == 1000
    p3 = alloc(900)                             # within 1.19 of a kept block: recycled
    ok = ok and p3 == p1 and pool.from_pool == 1 and pool.cached == 0
    pool._free(p3)
    p4 = alloc(700)                             # too small a request for that block
    ok = ok and p4 != p1 and live[p4] == 700
    check("pool: new blocks are exact, kept ones are recycled within the size bound", ok)
    pool._free(p2)
    pool._free(p4)
    p5 = alloc(2500)
    pool._free(p5)                              # larger than max_block: to the driver
    ok = p5 not in live and pool.cached == 2700
    p6 = alloc(1100)
    pool._free(p6)                              # full, and nothing larger to give up
    ok = ok and p6 not in live and pool.cached == 2700 and pool.peak_cached <= 3000
    check("pool: the limits on a block and on the total hold", ok)
    p7 = alloc(500)
    pool._free(p7)                              # full: one of the larger blocks goes
    ok = (p7 in live and pool.cached == 2200 and len(live) == 3
          and sorted(live.values()) == [500, 700, 1000] and pool.peak_cached <= 3000)
    check("pool: a full pool gives up a larger block for a smaller one", ok,
          "(%s)" % (sorted(live.values()),))
    live[0x7f0000000000] = 1                    # a block the pool did not hand out
    pool._free(0x7f0000000000)
    ok = 0x7f0000000000 not in live and pool.cached == 2200
    freed = pool.trim()
    check("pool: a foreign block goes to the driver, and trim returns the rest",
          ok and freed == 2200 and pool.cached == 0 and not live, "(%s)" % (pool.stats(),))
    # the pool that keeps blocks during a setup only (the default on a GPU)
    scoped = HyprePool(hypre, runtime, max_cached=3000, max_block=2000, scoped=True)

    def take(n):
        scoped._malloc(out, n)
        return out[0]

    q1 = take(1000)
    scoped._free(q1)                            # closed: straight back to the driver
    ok = q1 not in live and scoped.cached == 0
    scoped.open()
    q2 = take(1000)
    scoped._free(q2)                            # open: kept, and handed out again
    q3 = take(950)
    ok = ok and q3 == q2 and scoped.from_pool == 1
    scoped._free(q3)
    held = scoped.cached
    scoped.close()                              # closed again: nothing held, nothing kept
    q4 = take(1000)
    scoped._free(q4)
    check("pool: a scoped pool recycles between open and close and holds nothing outside",
          ok and held == 1000 and scoped.cached == 0 and not live, "(%s)" % (scoped.stats(),))
    # the same with something kept between setups (the default on a GPU): the smallest
    # blocks stay when the setup is over
    keeper = HyprePool(hypre, runtime, max_cached=3000, max_block=2000, scoped=True, keep=800)
    keeper.KEEP_SHARE = 1.0                     # the share of the peak is tested below

    def get(n):
        keeper._malloc(out, n)
        return out[0]

    keeper.open()
    keeper.open()                               # a second solver given its operator
    r = [get(1000), get(500), get(200), get(100)]
    for q in r:
        keeper._free(q)
    ok = keeper.cached == 1800 and len(live) == 4 and keeper.max_cached == 3000
    keeper.close()                              # the first solve of either ends the scope
    ok = ok and keeper.cached == 800 and sorted(live.values()) == [100, 200, 500]
    ok = ok and keeper.max_cached == 800
    r5 = get(450)                               # served from what was kept
    ok = ok and r5 == r[1] and keeper.from_pool == 1
    keeper._free(r5)
    r6 = get(1000)
    keeper._free(r6)                            # no setup pending and no room: to the driver
    ok = ok and r6 not in live and keeper.cached == 800
    keeper.trim()
    check("pool: between setups a scoped pool keeps its smallest blocks, up to its limit",
          ok and keeper.cached == 0 and not live, "(%s)" % (keeper.stats(),))
    # a small problem keeps little: no more than a quarter of the most it had in use
    small = HyprePool(hypre, runtime, max_cached=3000, max_block=2000, scoped=True, keep=800)

    def req(n):
        small._malloc(out, n)
        return out[0]

    small.open()
    s = [req(400), req(200), req(100), req(100)]      # 800 in use at the peak
    for q in s:
        small._free(q)
    small.close()
    ok = small.max_cached == 200 and small.cached == 200 and sorted(live.values()) == [100, 100]
    small.trim()
    check("pool: what is kept between setups is at most a quarter of the peak in use",
          ok and not live, "(%s)" % (small.stats(),))
    # a driver that is out of memory: the pool is emptied and the request tried again,
    # and the refusal the runtime remembers is read, or MFEM takes it for an error of its
    # next kernel
    full, read = [0], [0]                       # refusals to come, refusals read

    def stingy(pp, n):
        if full[0]:
            full[0] -= 1
            return 2                            # cudaErrorMemoryAllocation
        return dev_malloc(pp, n)

    def last_error():
        read[0] += 1
        return 2

    tight = types.SimpleNamespace(cudaMalloc=Fn(stingy), cudaFree=Fn(dev_free),
                                  cudaGetLastError=Fn(last_error))
    short = HyprePool(hypre, tight, max_cached=3000, max_block=2000)

    def ask(n):
        short._malloc(out, n)
        return out[0]

    kept = ask(600)
    short._free(kept)                           # held by the pool
    full[0] = 1
    got = ask(1500)                             # refused once: the pool gives its block back
    ok = bool(got) and kept not in live and live[got] == 1500 and read[0] == 1 and short.refused == 0
    full[0] = 2
    import contextlib
    import io

    said = io.StringIO()
    with contextlib.redirect_stderr(said):
        none = ask(1500)                        # refused twice: hypre is told, and so is the user
    check("pool: a refused request empties the pool and is tried again; the refusal is read",
          ok and not none and short.refused == 1 and read[0] == 2 and "out of memory" in said.getvalue(),
          "(%s)" % (short.stats(),))
    short._free(got)
    short.trim()


def test_boomeramg_options():
    """BoomerAMG settings beyond those of MFEM's class, through hypre's own handle of the
    preconditioner (``mfemconfig.set_boomeramg_options``, a solver's ``amg_options``)."""
    import ctypes

    from hippymfem.common import mfemconfig as mc

    if RANK == 0:
        print("BoomerAMG options through hypre's handle")
    opts = mc.parse_boomeramg_options(" relax_wt=1.8, pmax=6 ; theta=0.5,")
    check("a list of settings is read by name, each with its type",
          opts == {"relax_wt": 1.8, "pmax": 6, "theta": 0.5} and type(opts["pmax"]) is int
          and mc.parse_boomeramg_options("") == {} and mc.parse_boomeramg_options(None) == {})
    said = ""
    try:
        mc.parse_boomeramg_options("pmax=6,pmaxx=6")
    except ValueError as e:
        said = str(e)
    check("an unknown name is refused, with the names that exist",
          "'pmaxx'" in said and "relax_wt" in said)

    A, b, V = make_operator(n=24)
    pc = mfem.HypreBoomerAMG(A)
    pc.SetPrintLevel(0)
    H = mc._hypre_library()
    if H is None or mc.boomeramg_handle(pc) is None:
        raised = False
        try:
            mc.set_boomeramg_options(pc, {"pmax": 6})
        except RuntimeError:
            raised = True
        check("without hypre's handle the options are refused with an explanation (skipped: "
              "this MFEM library does not export it)", raised)
        return

    def read(pc, name, kind, *more):
        out, fn = kind(), getattr(H, name)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(kind)] + [ctypes.c_int] * len(more)
        fn(ctypes.c_void_p(mc.boomeramg_handle(pc)), ctypes.byref(out), *more)
        return out.value

    mc.set_boomeramg_options(pc, {"levels": 7, "theta": 0.5, "pmax": 6, "trunc": 0.125,
                                  "coarse_size": 40, "coarse_relax": 7})
    got = (read(pc, "HYPRE_BoomerAMGGetMaxLevels", ctypes.c_int),
           read(pc, "HYPRE_BoomerAMGGetStrongThreshold", ctypes.c_double),
           read(pc, "HYPRE_BoomerAMGGetPMaxElmts", ctypes.c_int),
           read(pc, "HYPRE_BoomerAMGGetTruncFactor", ctypes.c_double),
           read(pc, "HYPRE_BoomerAMGGetMaxCoarseSize", ctypes.c_int),
           read(pc, "HYPRE_BoomerAMGGetCycleRelaxType", ctypes.c_int, 3))
    check("the values arrive in hypre, integers and reals alike",
          got == (7, 0.5, 6, 0.125, 40, 7), "(%s)" % ", ".join("%g" % v for v in got))
    said = ""
    try:
        mc.set_boomeramg_options(pc, {"levels": 0})
    except ValueError as e:
        said = str(e)
    check("a value hypre refuses raises by the name of the setting, and leaves no error behind",
          "levels=0" in said and H.HYPRE_GetError() == 0
          and read(pc, "HYPRE_BoomerAMGGetMaxLevels", ctypes.c_int) == 7)
    del pc

    def solver(extra):
        ks = hp.KrylovSolver(COMM, "cg", "amg")
        ks.parameters["rel_tolerance"] = 1e-12
        ks.parameters["amg_options"] = extra
        ks.set_operator(A)
        return ks

    x0, x1 = V.vector(), V.vector()
    its0 = solver("").solve(x0, b)
    ks = solver("pmax=6,theta=0.3")
    arrived = (read(ks._pc, "HYPRE_BoomerAMGGetPMaxElmts", ctypes.c_int),
               read(ks._pc, "HYPRE_BoomerAMGGetStrongThreshold", ctypes.c_double))
    its1 = ks.solve(x1, b)
    res = residual(A, x1, b)
    e = x1.axpy(-1.0, x0).norm("l2") / x0.norm("l2")
    check("a solver's amg_options reach its preconditioner, and its solve agrees with the "
          "default one", arrived == (6, 0.3) and res < 1e-10 and e < 1e-9,
          "(residual %.1e, difference %.1e, %d and %d iterations)" % (res, e, its1, its0))
    raised = False
    try:
        solver("pmaxx=6")
    except ValueError:
        raised = True
    check("and a mistyped one stops the solver's setup", raised)

    # A hierarchy ended at a level of a few rows, which one process finishes, gathered
    # (seq_threshold).  hypre's own code does it on the host; with hypre on a device it
    # needs tools/hypre-3.2.0-seqdevice.patch, and without it the setting is refused
    # with the reason, since hypre would fault in the set-up.  (On one rank hypre drops
    # the setting, and the solve is the default one.)
    device = bool(mc._DEVICE)
    if device and not mc.seq_on_device(H):
        raised = False
        try:
            solver("seq_threshold=200")
        except RuntimeError:
            raised = True
        check("seq_threshold is refused by a device hypre without the patch (this one)", raised)
        return
    for extra, who in (("seq_threshold=200", "the first rank"),
                       ("seq_threshold=200,redundant=1", "every rank")):
        ks = solver(extra)
        arrived = read(ks._pc, "HYPRE_BoomerAMGGetSeqThreshold", ctypes.c_int)
        x2 = V.vector()
        its2 = ks.solve(x2, b)
        res = residual(A, x2, b)
        e = x2.axpy(-1.0, x0).norm("l2") / x0.norm("l2")
        check("seq_threshold ends the hierarchy at a level of that many rows and %s "
              "finishes it: the solve agrees with the default one" % who,
              arrived == 200 and res < 1e-10 and e < 1e-9 and its2 <= its0 + 3,
              "(residual %.1e, difference %.1e, %d iterations against %d%s)"
              % (res, e, its2, its0, ", on a device" if device else ""))
    # the whole matrix under the threshold (the gathered level is the matrix itself), and
    # a second operator through the same solver, which gathers again
    ks = solver("seq_threshold=%d" % (A.GetGlobalNumRows() + 1))
    x3 = V.vector()
    its3 = ks.solve(x3, b)
    e3 = x3.axpy(-1.0, x0).norm("l2") / x0.norm("l2")
    A2, b2, V2 = make_operator(n=16)
    ks.set_operator(A2)
    x4 = V2.vector()
    its4 = ks.solve(x4, b2)
    res4 = residual(A2, x4, b2)
    check("with a threshold above the matrix one process solves all of it, and a second "
          "operator of the same solver is gathered in its turn",
          e3 < 1e-9 and res4 < 1e-10, "(difference %.1e in %d iterations; residual %.1e in %d)"
          % (e3, its3, res4, its4))


def test_chebyshev_solver():
    """A fixed polynomial of multigrid cycles in place of a Krylov solve
    (``ChebyshevSolver``): its error against the exact solve is within its bound, and it
    is one linear map, symmetric and positive definite, which a Krylov solve stopped at
    a tolerance is not.  The prior hands out its precision solver made of two of them as
    the preconditioner of the Hessian."""
    if RANK == 0:
        print("a fixed polynomial of multigrid cycles in place of a Krylov solve")
    A, b, V = make_operator(n=24)
    ks = hp.KrylovSolver(COMM, "cg", "amg")
    ks.parameters["rel_tolerance"] = 1e-13
    ks.set_operator(A)
    x0 = V.vector()
    its0 = ks.solve(x0, b)

    def energy(v):
        w = ParVector(COMM, A.Height())
        A.Mult(v.hypre, w.hypre)
        return math.sqrt(max(w.inner(v), 0.0))

    said, worst = [], 0.0
    for degree in (1, 2, 4, 6):
        ch = hp.ChebyshevSolver(ks, degree)
        x = V.vector()
        n = ch.solve(x, b)
        err = energy(x.copy().axpy(-1.0, x0)) / energy(x0)
        bound = ch.error_bound
        said.append("%d: %.1e (bound %.1e)" % (degree, err, bound))
        worst = max(worst, err / bound)
        ok = n == degree and err <= 1.05 * bound + 1e-12
        if not ok:
            worst = float("inf")
    lo, hi = ch.interval
    check("the error of the polynomial is within its bound at every degree",
          worst <= 1.05 and 0.0 < lo < hi,
          "(%s; the cycle's eigenvalues in [%.3f, %.3f], Ritz values %.3f and %.3f; CG to 1e-13 "
          "takes %d iterations)" % ("; ".join(said), lo, hi, ch.ritz[0], ch.ritz[1], its0))
    # one linear map, symmetric and positive definite
    hp.parRandom.set_seed(77)
    u, v = V.vector(), V.vector()
    hp.parRandom.normal(1.0, u)
    hp.parRandom.normal(1.0, v)
    ch = hp.ChebyshevSolver(ks, 4)
    su, sv, sw = V.vector(), V.vector(), V.vector()
    ch.solve(su, u)
    ch.solve(sv, v)
    w = u.copy().scale(0.3).axpy(-1.7, v)
    ch.solve(sw, w)
    lin = sw.copy().axpy(-0.3, su).axpy(1.7, sv).norm("l2") / sw.norm("l2")
    a, c = v.inner(su), u.inner(sv)
    sym = abs(a - c) / max(abs(a), abs(c))
    check("it is one linear map, symmetric and positive definite",
          lin < 1e-10 and sym < 1e-10 and u.inner(su) > 0.0 and v.inner(sv) > 0.0,
          "(S(0.3 u - 1.7 v) off by %.1e, <v, S u> and <u, S v> by %.1e)" % (lin, sym))
    # the cycles applied from the iterate (BoomerAMG) and the recurrence with a residual
    # updated by a product (any other preconditioner) are one polynomial
    xa, xb = V.vector(), V.vector()
    ch.solve(xa, b)
    kept, ks.precond_type = ks.precond_type, "another"
    try:
        ch.solve(xb, b)
    finally:
        ks.precond_type = kept
    two = xb.axpy(-1.0, xa).norm("l2") / xa.norm("l2")
    check("cycles from the iterate and a residual updated by products give the same solve",
          two < 1e-11, "(difference %.1e)" % two)
    # refused where there is nothing to make it of
    raised = False
    try:
        hp.ChebyshevSolver(hp.KrylovSolver(COMM, "cg", "amg"), 4)
    except ValueError:
        raised = True
    check("and it is refused for a solver without an operator", raised)
    # the prior's precision solver of two such polynomials, as the Hessian's preconditioner
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(24, 24, mfem.Element.QUADRILATERAL))
    Vm = hp.FunctionSpace.H1(pm, 1)
    prior = hp.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True)
    g, r_exact, r_cheb = Vm.vector(), Vm.vector(), Vm.vector()
    hp.parRandom.set_seed(78)
    hp.parRandom.normal(1.0, g)
    prior.Rsolver.solve(r_exact, g)
    pc = prior.getHessianPreconditioner(cycles=5)
    pc.solve(r_cheb, g)
    krylov = isinstance(prior.Asolver, hp.KrylovSolver)
    if krylov:
        bound = pc.Asolver.error_bound
        diff = r_cheb.copy().axpy(-1.0, r_exact).norm("l2") / r_exact.norm("l2")
        # (the two bounds are in the energy norm of each solve; an order of magnitude is
        # asked of the Euclidean norm of their composition)
        check("the prior's precision solver with five cycles a solve is the exact one to the bound",
              pc is not prior.Rsolver and pc is prior.getHessianPreconditioner(cycles=5)
              and diff < 20.0 * bound and g.inner(r_cheb) > 0.0,
              "(difference %.1e, bound of one solve %.1e)" % (diff, bound))
    else:
        check("a prior solved by a factorization hands out its own solver", pc is prior.Rsolver)


def test_index_widths():
    """hypre's two integer types (``parvector._HYPRE_INT`` and ``_HYPRE_BIGINT``): arrays of
    global indices reach MFEM in the wider one, and a matrix goes from scipy to hypre and
    back with either pair of widths (one width, or the two of a hypre built
    ``--enable-mixedint``)."""
    import scipy.sparse as sp

    from hippymfem.common import parvector as pv
    from hippymfem.common.linalg import hypre_to_scipy, scipy_to_hypre

    if RANK == 0:
        print("hypre's local and global index types")
    narrow, wide = 8 * np.dtype(pv._HYPRE_INT).itemsize, 8 * np.dtype(pv._HYPRE_BIGINT).itemsize
    check("the two types are the build's", narrow == 8 * mfem.sizeof_HYPRE_Int()
          and wide == 8 * mfem.sizeof_HYPRE_BigInt(),
          "(HYPRE_Int %d bits, HYPRE_BigInt %d bits)" % (narrow, wide))
    vals = [3, 5, 2 ** 31 - 1] + ([2 ** 40 + 7] if wide == 64 else [])
    arr = pv.bigint_array(vals)
    back = [int(v) for v in pv.bigint_view(arr.GetData(), len(vals))]
    check("global indices go to an MFEM array and back, up to the type's range", back == vals,
          "(%s)" % back[-1])

    # this rank's rows with global columns: a square matrix, and a rectangular one with
    # its own column partition
    nloc = 3 + RANK
    counts = COMM.allgather(nloc)
    off, N = int(sum(counts[:RANK])), int(sum(counts))
    rows = np.repeat(np.arange(nloc), 3)
    cols = np.concatenate([[(off + i) % N, (off + i + 1) % N, (off + i + N // 2) % N]
                           for i in range(nloc)])
    data = 1.0 + np.arange(rows.size) + 10.0 * RANK
    S = sp.csr_matrix((data, (rows, cols)), shape=(nloc, N))
    S.sum_duplicates()
    A = scipy_to_hypre(S)
    T = hypre_to_scipy(A)
    err = abs(T - S).max() if T.shape == S.shape else np.inf
    check("a square matrix goes from scipy to hypre and back", err == 0.0
          and int(A.GetGlobalNumRows()) == N and int(A.GetGlobalNumCols()) == N, "(%g)" % err)
    mloc = 2
    M = mloc * NP
    R = sp.csr_matrix((data, (rows, cols % M)), shape=(nloc, M))
    R.sum_duplicates()
    B = scipy_to_hypre(R, col_starts=[mloc * RANK, mloc * (RANK + 1)])
    T = hypre_to_scipy(B)
    err = abs(T - R).max() if T.shape == R.shape else np.inf
    check("and a rectangular one with its column partition", err == 0.0
          and int(B.GetGlobalNumCols()) == M and B.Width() == mloc, "(%g)" % err)


def test_lumped_mass():
    """The lumped mass solver: its diagonal against MFEM's linear form of the constant one."""
    if RANK == 0:
        print("lumped mass solver")
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(9, 7, mfem.Element.TRIANGLE))
    V = hp.FunctionSpace.H1(pm, 1)
    M = hp.assemble_native_matrix(V, [mfem.MassIntegrator()])
    S = hp.LumpedMassSolver(M, COMM)

    one = mfem.ConstantCoefficient(1.0)
    lf = mfem.ParLinearForm(V.fes)
    lf.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    lf.Assemble()
    ref = V.vector()
    lf.ParallelAssemble(ref.hypre)

    ones, d = V.vector(), V.vector()
    ones.set(1.0)
    S.mult(ones, d)
    e = d.copy().axpy(-1.0, ref).norm("linf") / ref.norm("linf")
    check("the lumped diagonal is the integral of each basis function", e < 1e-13, "(%.2e)" % e)
    check("and sums to the area of the mesh", abs(d.sum() - 1.0) < 1e-13, "(%.15f)" % d.sum())
    x = V.vector()
    S.solve(x, ref)
    e = x.axpy(-1.0, ones).norm("linf")
    check("solve inverts it", e < 1e-13, "(%.2e)" % e)


def test_transpose_solver():
    """``A^T x = b`` for a nonsymmetric matrix, held against MFEM's ``MultTranspose``."""
    if RANK == 0:
        print("transpose solver")
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(10, 10, mfem.Element.TRIANGLE))
    V = hp.FunctionSpace.H1(pm, 1)
    wind = mfem.Vector(2)
    wind[0], wind[1] = 1.0, 0.5
    vel = mfem.VectorConstantCoefficient(wind)
    A = hp.assemble_native_matrix(V, [mfem.DiffusionIntegrator(), mfem.MassIntegrator(),
                                      mfem.ConvectionIntegrator(vel, 3.0)])
    b = V.project(lambda x: np.sin(3.0 * x[0]) * np.cos(2.0 * x[1]) + x[0])

    def factory():
        if NP == 1:
            return hp.LUSolver(COMM)
        ks = hp.KrylovSolver(COMM, "gmres", "amg")
        ks.parameters["rel_tolerance"] = 1e-13
        return ks

    S = hp.TransposeSolver(factory)
    S.set_operator(A)
    x = V.vector()
    S.solve(x, b)
    r = V.vector()
    A.MultTranspose(x.hypre, r.hypre)
    e = r.axpy(-1.0, b).norm("l2") / b.norm("l2")
    check("A^T x = b", e < 1e-10, "(%.2e)" % e)
    check("which is not A x = b", residual(A, x, b) > 1e-3, "(%.2e)" % residual(A, x, b))
    S.release()
    check("release drops the transpose and the inner solver", S.inner is None and S._At is None)


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
    test_hypre_pool_logic()
    test_boomeramg_options()
    test_chebyshev_solver()
    test_index_widths()
    test_single_precision_solves()
    test_lumped_mass()
    test_transpose_solver()
    test_petsc()
    COMM.Barrier()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""), flush=True)
    return 1 if FAILS else 0


def test_single_precision_solves():
    """The solves in a single-precision hypre (``HIPPYMFEM_HYPRE_SINGLE``) against the
    double-precision ones; skipped when no library is named."""
    from hippymfem.test import single_case

    single_case.run(check, COMM)


if __name__ == "__main__":
    raise SystemExit(main())

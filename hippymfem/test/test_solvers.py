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
    # A block that hypre frees while the pool is being emptied.  Python's collector can
    # start inside trim's loop over the bins, at a call of the driver, and where it
    # finalizes a dropped matrix, hypre frees that matrix's arrays through the pool there.
    # The driver here does at its first call what such a finalizer does.
    late, stray = [], []                        # blocks to come back; frees of blocks not held

    def free_and_finalize(p):
        if live.pop(p if isinstance(p, int) else p.value, None) is None:
            stray.append(p)
        while late:
            busy._free(late.pop())
        return 0

    reentered = types.SimpleNamespace(cudaMalloc=Fn(dev_malloc), cudaFree=Fn(free_and_finalize))
    busy = HyprePool(hypre, reentered, max_cached=2000, max_block=2000)

    def grab(n):
        busy._malloc(out, n)
        return out[0]

    t = [grab(1000), grab(700), grab(300), grab(40)]
    for q in t[:3]:
        busy._free(q)                           # the pool is full
    late.append(t[3])                           # and this one is of a size it does not hold
    try:
        freed = busy.trim()
    except RuntimeError as e:
        freed = str(e)
    kept = [q for blocks in busy._bins.values() for q, _n in blocks]
    ok = freed == 2000 and kept == list(live) == [t[3]] and busy.cached == 40 and busy.in_use == 0
    busy.trim()
    check("pool: a block freed while the pool is emptied goes to the driver once, like the rest",
          ok and not stray and not live,
          "(trim returned %r; %d freed twice; %d left)" % (freed, len(stray), len(live)))
    live.clear()                                # (what a failure left, for the check below)
    # The single-precision hypre allocates through the same pool where it is loaded
    # (singlesolve._Library gives it the pool's two functions).  Removing the pool gives
    # both libraries their own allocator back: one that kept the two functions went on
    # filling a pool that nothing opened, closed or trimmed any more.
    from hippymfem.common import mfemconfig as mc

    class Library:                              # takes the place of a libHYPRE
        malloc = free = None

        def hypre_SetUserDeviceMalloc(self, f):
            self.malloc = f

        def hypre_SetUserDeviceMfree(self, f):
            self.free = f

    first, second = Library(), Library()
    shared = HyprePool(first, runtime, max_cached=3000, max_block=2000)
    second.hypre_SetUserDeviceMalloc(shared._cb_malloc)
    second.hypre_SetUserDeviceMfree(shared._cb_free)
    ok = (first.malloc is second.malloc is shared._cb_malloc
          and first.free is second.free is shared._cb_free)
    was = mc.HYPRE_POOL, mc._single_library
    mc.HYPRE_POOL, mc._single_library = shared, lambda: second
    try:
        shared._malloc(out, 500)
        shared._free(out[0])                    # something for the removal to return
        shared.uninstall()
    finally:
        mc.HYPRE_POOL, mc._single_library = was
    check("pool: removing it gives every library that allocates through it its own allocator back",
          ok and not shared.installed and first.malloc is first.free is None
          and second.malloc is second.free is None and shared.cached == 0 and not live)


def test_gpu_aware_mpi_check():
    """The check of the MPI library before hypre hands it buffers on the GPU
    (``mfemconfig._check_gpu_aware_mpi``), with stand-ins for hypre's switch, for the MPI
    library and for a communicator of two ranks.  A refusal leaves hypre staging its
    buffers: ``configure_device`` has the device by then, so a second call returns it
    without the check, and a caller that caught the error went on with the exchange that
    was refused."""
    import types

    from hippymfem.common import mfemconfig as mc

    if RANK == 0:
        print("the check of the MPI library for hypre's buffers on the GPU")
    flag = [0]

    def switch(on):
        flag[0] = int(on.value)
        return 0

    hypre = types.SimpleNamespace(hypre_SetGpuAwareMPI=switch, hypre_GetGpuAwareMPI=lambda: flag[0])
    two = types.SimpleNamespace(Get_size=lambda: 2, Get_rank=lambda: 0)
    was = mc._HYPRE_HANDLE, mc.mpi_gpu_support
    seen = {}
    try:
        mc._HYPRE_HANDLE = hypre
        for can in (True, False):
            mc.mpi_gpu_support = lambda can=can: can
            mc.set_hypre_gpu_aware_mpi(True)
            try:
                passed = mc._check_gpu_aware_mpi(two, quiet=True)
            except RuntimeError as e:
                passed = "refused" if "HIPPYMFEM_GPU_AWARE_MPI=0" in str(e) else str(e)
            seen[can] = (passed, mc.hypre_gpu_aware_mpi())
    finally:
        mc._HYPRE_HANDLE, mc.mpi_gpu_support = was
    check("an MPI library that reads device memory passes, and hypre keeps handing it its buffers",
          seen[True] == (True, True), "(%r)" % (seen[True],))
    check("one that cannot is refused with the way out, and hypre is left staging its buffers",
          seen[False] == ("refused", False), "(%r)" % (seen[False],))


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
    # hypre's error flag is the process's, and every setter returns it: what an earlier
    # call left up (here the same refusal from hypre itself, which nothing clears) is not
    # a refusal of the settings that follow
    fn = H.HYPRE_BoomerAMGSetMaxLevels
    fn.restype, fn.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]
    up, said = fn(ctypes.c_void_p(mc.boomeramg_handle(pc)), 0), ""
    try:
        mc.set_boomeramg_options(pc, {"pmax": 5})
    except ValueError as e:
        said = str(e)
    check("an error of hypre's that an earlier call left up is not taken for a refusal",
          up != 0 and not said and H.HYPRE_GetError() == 0
          and read(pc, "HYPRE_BoomerAMGGetPMaxElmts", ctypes.c_int) == 5,
          "(the flag was %d%s)" % (up, said and "; " + said))
    # hypre's setters of the relaxation and of its sweeps reset those of the coarsest
    # level: the coarsest level's own hold wherever they are written
    mc.set_boomeramg_options(pc, {"coarse_relax": 7, "coarse_sweeps": 4, "relax": 18, "sweeps": 2})
    got = (read(pc, "HYPRE_BoomerAMGGetCycleRelaxType", ctypes.c_int, 1),
           read(pc, "HYPRE_BoomerAMGGetCycleRelaxType", ctypes.c_int, 3),
           read(pc, "HYPRE_BoomerAMGGetCycleNumSweeps", ctypes.c_int, 1),
           read(pc, "HYPRE_BoomerAMGGetCycleNumSweeps", ctypes.c_int, 3))
    check("the coarsest level's relaxation and sweeps hold when they are written before the others'",
          got == (18, 7, 2, 4), "(%s)" % ", ".join("%d" % v for v in got))
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
    from hippymfem.common.parvector import device_active

    device = device_active()
    if not device:
        # MFEM's "cpu" device, which the drivers configure on a host run, is not a card:
        # with it a hypre without the patch (this one answers as one here) takes the
        # setting in host memory as it does without
        saved_patch, saved_device, said = mc.seq_on_device, list(mc._DEVICE), ""
        try:
            mc.seq_on_device = lambda library: False
            mc._DEVICE.append("cpu")
            solver("seq_threshold=200")
        except RuntimeError as e:
            said = str(e)
        finally:
            mc.seq_on_device = saved_patch
            mc._DEVICE[:] = saved_device
        check("in host memory seq_threshold needs no patch, whatever device MFEM was configured with",
              not said, said and "(%s ...)" % said[:70])
    if device and not mc.seq_on_device(H):
        raised = False
        try:
            solver("seq_threshold=200")
        except RuntimeError:
            raised = True
        check("seq_threshold is refused by a device hypre without the patch (this one)", raised)
        return
    if NP == 1:
        # hypre drops the setting on one rank (the set-up's "if (num_procs == 1)
        # seq_threshold = 0"), so every hierarchy below would be the default one, compared
        # with itself under names that speak of a gathered level
        ks = solver("seq_threshold=200,redundant=1")
        arrived = read(ks._pc, "HYPRE_BoomerAMGGetSeqThreshold", ctypes.c_int)
        x2 = V.vector()
        its2 = ks.solve(x2, b)
        res = residual(A, x2, b)
        e = x2.axpy(-1.0, x0).norm("l2") / x0.norm("l2")
        who = mc.seq_solvers(ks._pc, COMM)
        check("on one rank hypre takes seq_threshold and drops it: nothing is gathered and the solve "
              "is the default one (the checks of a gathered level take two ranks: skipped)",
              arrived == 200 and who is None and res < 1e-10 and e < 1e-9 and its2 <= its0 + 3,
              "(solved by: %s; residual %.1e, difference %.1e, %d iterations against %d%s)"
              % (who, res, e, its2, its0, ", on a device" if device else ""))
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
    # a threshold above the whole matrix (hypre coarsens once before it compares a size
    # with the threshold, so the gathered level is the first coarse one, not the matrix
    # itself), and a second operator through the same solver, which gathers again
    ks = solver("seq_threshold=%d" % (A.GetGlobalNumRows() + 1))
    x3 = V.vector()
    its3 = ks.solve(x3, b)
    e3 = x3.axpy(-1.0, x0).norm("l2") / x0.norm("l2")
    A2, b2, V2 = make_operator(n=16)
    ks.set_operator(A2)
    x4 = V2.vector()
    its4 = ks.solve(x4, b2)
    res4 = residual(A2, x4, b2)
    check("with a threshold above the matrix its first coarse level is the gathered one, and a "
          "second operator of the same solver is gathered in its turn",
          e3 < 1e-9 and res4 < 1e-10, "(difference %.1e in %d iterations; residual %.1e in %d)"
          % (e3, its3, res4, its4))


def test_refused_settings():
    """Settings that cannot be taken are refused, or said, and nothing is left half done.

    A mistyped ``HIPPYMFEM_HYPRE_SPMV`` stops ``configure_device`` before there is a
    device: stopped after the device was registered, a second call returned that device
    without the check of the MPI library and without a pool.  A hypre before 2.33 has the
    switch of its exchange and an exchange that is compiled one way
    (``mfemconfig._gpu_aware_fixed``): the switch is left alone there, the check asks the
    MPI library whatever ``HIPPYMFEM_GPU_AWARE_MPI`` says, and a setting that changes
    nothing is said.  And a solver parameter that goes through one of MFEM's own setters
    and that hypre refuses stops the solver's set-up, where MFEM dropped it without a
    word.  MFEM's device, hypre's switch, the MPI library and a communicator of two ranks
    are stand-ins here, on any build and without a GPU; the solver is a real one."""
    import contextlib
    import ctypes
    import io
    import types
    import warnings

    from hippymfem.common import mfemconfig as mc

    if RANK == 0:
        print("settings that are refused, or that a library cannot follow")
    made = []

    class Device:                               # takes the place of mfem.Device
        def __init__(self, kind, index=0):
            made.append(kind)

    def hypre(version, built):
        """hypre as far as the switch of its exchange goes: its version, and the field of
        its handle that the build presets, with the field's switch."""
        lib = types.SimpleNamespace(field=int(built), written=0)

        def number(major, minor, patch, single):
            ctypes.cast(single, ctypes.POINTER(ctypes.c_longlong))[0] = version
            return 0

        def switch(on):
            lib.field, lib.written = int(on.value), lib.written + 1
            return 0

        lib.HYPRE_VersionNumber, lib.hypre_SetGpuAwareMPI = number, switch
        lib.hypre_GetGpuAwareMPI = lambda: lib.field
        return lib

    two = types.SimpleNamespace(Get_size=lambda: 2, Get_rank=lambda: 0)
    names = ("HIPPYMFEM_HYPRE_SPMV", "HIPPYMFEM_GPU_AWARE_MPI", "HIPPYMFEM_HYPRE_POOL",
             "HIPPYMFEM_HYPRE_POOL_KEEP")

    def configure(library, can, **variables):
        """``configure_device`` with that hypre under an MPI library that says ``can``:
        how it ended, how often the MPI library was asked, whether the device line says
        that hypre hands MPI its buffers, and the warnings about the variable."""
        asked = []
        mc._HYPRE_HANDLE, mc.mpi_gpu_support = library, (lambda: asked.append(1) or can)
        mc.GPU_AWARE_SOLVES = False
        for k in names:
            os.environ.pop(k, None)
        os.environ.update({"HIPPYMFEM_HYPRE_POOL": "0", **variables})
        line = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(line):
            warnings.simplefilter("always")
            try:
                mc.configure_device("gpu", two, quiet=False)
                end = "configured"
            except (ValueError, RuntimeError) as e:
                end = "%s: %s" % (type(e).__name__, " ".join(str(e).split()))
        said = [str(w.message) for w in caught if "HIPPYMFEM_GPU_AWARE_MPI" in str(w.message)]
        return end, len(asked), "hands MPI its buffers" in line.getvalue(), said

    env = {k: os.environ.get(k) for k in names}
    was = (mfem.Device, mc.mfem_gpu_backend, mc.hypre_on_device, mc._device_count, mc._node_rank,
           mc.mpi_gpu_support, mc._HYPRE_HANDLE, list(mc._DEVICE), mc.DEVICE_INDEX,
           mc.GPU_AWARE_SOLVES, mc.HYPRE_SPMV)
    backend = mc.mfem_gpu_backend() or "cuda"
    try:
        mfem.Device, mc.mfem_gpu_backend, mc.hypre_on_device = Device, (lambda: backend), (lambda: True)
        mc._device_count, mc._node_rank = (lambda: 1), (lambda comm=None: 0)
        # a mistyped kernel of the matrix-vector products, twice: the second call found the
        # device that the first had registered before it was stopped, and returned it
        del mc._DEVICE[:]
        ends = [configure(hypre(30200, False), True, HIPPYMFEM_HYPRE_SPMV="vendr")[0] for _ in range(2)]
        check("a mistyped HIPPYMFEM_HYPRE_SPMV stops configure_device before there is a device, at a "
              "second call as at the first",
              all(e.startswith("ValueError") and "HIPPYMFEM_HYPRE_SPMV" in e for e in ends)
              and not made and not mc._DEVICE,
              "(%s; %d device(s) made, %d registered)"
              % (" then ".join(e.split(":")[0] for e in ends), len(made), len(mc._DEVICE)))
        # the pool's two settings: one that is neither "auto" nor megabytes was taken for 0, no
        # pool at all, and a number of megabytes to keep that could not be read for the default
        del mc._DEVICE[:]
        ends = [configure(hypre(30200, False), True, **v)[0]
                for v in ({"HIPPYMFEM_HYPRE_POOL": "on"},
                          {"HIPPYMFEM_HYPRE_POOL": "auto", "HIPPYMFEM_HYPRE_POOL_KEEP": "some"})]
        os.environ.update(HIPPYMFEM_HYPRE_POOL="auto", HIPPYMFEM_HYPRE_POOL_KEEP="")
        read = [mc._pool_settings()]
        os.environ.update(HIPPYMFEM_HYPRE_POOL="512", HIPPYMFEM_HYPRE_POOL_KEEP="some")
        read.append(mc._pool_settings())
        check("a HIPPYMFEM_HYPRE_POOL or HIPPYMFEM_HYPRE_POOL_KEEP that cannot be read stops "
              "configure_device before there is a device, by the variable's name",
              ends[0].startswith("ValueError: HIPPYMFEM_HYPRE_POOL must")
              and ends[1].startswith("ValueError: HIPPYMFEM_HYPRE_POOL_KEEP must")
              and not made and not mc._DEVICE
              and read == [("auto", mc.HYPRE_POOL_KEEP_MB), (512.0, None)],
              "(%s; then %s; %d device(s) made; read where they can be: %s)"
              % (ends[0][:44], ends[1][:49], len(made), read))
        # hypre 2.32 compiled to hand MPI its buffers, under an MPI library that cannot read
        # device memory, with the variable that the refusal used to advise: the exchange
        # does not read what the variable sets, so the MPI library is asked all the same
        old = hypre(23200, True)
        del mc._DEVICE[:]
        end, asked, _, _ = configure(old, False, HIPPYMFEM_GPU_AWARE_MPI="0")
        check("a hypre before 2.33 that is compiled to hand MPI its buffers is checked against the MPI "
              "library whatever HIPPYMFEM_GPU_AWARE_MPI says, and is not told that the variable is a way out",
              end.startswith("RuntimeError") and "HIPPYMFEM_GPU_AWARE_MPI=0" not in end
              and "build scripts" in end and asked == 1 and (old.field, old.written) == (1, 0),
              "(%s; the MPI library asked %d time(s); the handle's field %d, written %d time(s))"
              % (end[:40], asked, old.field, old.written))
        # what such a library cannot follow is said, the device line says what the library
        # does, and no solve is taken for one around which the exchange is switched
        seen = {}
        for label, built, can, want in (("compiled to hand over, =solve", True, True, "solve"),
                                        ("compiled to stage, =1", False, False, "1")):
            lib = hypre(23200, built)
            del mc._DEVICE[:]
            end, asked, line, said = configure(lib, can, HIPPYMFEM_GPU_AWARE_MPI=want)
            seen[label] = (end[:20], line == built, len(said), mc.GPU_AWARE_SOLVES, lib.written)
        check("a setting that such a library cannot follow is said, the device line says what the "
              "library does, and no solve is taken for one that switches the exchange",
              all(v == ("configured", True, 1, False, 0) for v in seen.values()), "(%s)" % (seen,))
        # from 2.33 on the exchange reads the field: staged when told to, whatever the
        # library was compiled for, and switched around a solve
        new = hypre(23300, True)
        del mc._DEVICE[:]
        staged = configure(new, False, HIPPYMFEM_GPU_AWARE_MPI="0") + (new.field,)
        new = hypre(30200, False)
        del mc._DEVICE[:]
        solves = configure(new, True, HIPPYMFEM_GPU_AWARE_MPI="solve") + (new.field, mc.GPU_AWARE_SOLVES)
        check("from hypre 2.33 on the switch is followed as before",
              staged == ("configured", 0, False, [], 0) and solves == ("configured", 1, False, [], 0, True),
              "(=0: %s; =solve: %s)" % (staged, solves))
    finally:
        (mfem.Device, mc.mfem_gpu_backend, mc.hypre_on_device, mc._device_count, mc._node_rank,
         mc.mpi_gpu_support, mc._HYPRE_HANDLE, mc._DEVICE[:], mc.DEVICE_INDEX,
         mc.GPU_AWARE_SOLVES, mc.HYPRE_SPMV) = was
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # A value that hypre refuses through one of MFEM's setters, a strength threshold above
    # one.  MFEM does not pass on what hypre returns: the solver was set up with hypre's
    # own threshold, and hypre's error flag stayed up for whichever call returned it next.
    A, b, V = make_operator(n=10)
    H = mc._hypre_library()
    ks = hp.KrylovSolver(COMM, "cg", "amg")
    ks.parameters["amg_strength_threshold"] = 1.5
    said = ""
    try:
        ks.set_operator(A)
    except ValueError as e:
        said = str(e)
    left = int(H.HYPRE_GetError()) if H is not None else 0
    if left:
        H.HYPRE_ClearAllErrors()                # (not for the solvers that follow)
    ks = hp.KrylovSolver(COMM, "cg", "amg")
    ks.parameters["amg_strength_threshold"] = 0.5
    ks.parameters["rel_tolerance"] = 1e-10
    x = V.vector()
    its = ks.set_operator(A).solve(x, b)
    res = residual(A, x, b)
    if H is None:
        check("a solver parameter that hypre refuses stops the solver's set-up (skipped: hypre's "
              "library is not found in this process, and MFEM does not say what hypre answered)",
              res < 1e-8)
    else:
        check("a solver parameter that hypre refuses stops the solver's set-up by its name and leaves "
              "no error of hypre's behind, and one that hypre takes solves",
              "amg_strength_threshold=1.5" in said and left == 0 and res < 1e-8,
              "(%s; flag %d; with 0.5 %d iterations to a residual of %.1e)"
              % (said or "no error", left, its, res))


def test_hypre_pcg_verdict():
    """What ``KrylovSolver`` says of a CG solve that runs in hypre's own PCG
    (``linSolvers.HYPRE_PCG``): hypre's verdict, not the count of its iterations, which
    says nothing of an iteration that hypre abandons or does not begin; and hypre's error
    flag, which MFEM's ``HyprePCG`` does not look at, does not outlive the solve."""
    from hippymfem.common import mfemconfig as mc

    if RANK == 0:
        print("the verdict of a solve in hypre's PCG")
    A, b, V = make_operator(n=24)
    H = mc._hypre_library()

    def solver(A, extra="", most=200):
        ks = hp.KrylovSolver(COMM, "cg", "amg")
        ks.parameters["rel_tolerance"] = 1e-10
        ks.parameters["max_iter"] = most
        ks.parameters["amg_options"] = extra
        return ks.set_operator(A)

    def attempt(ks, x, rhs):
        try:
            ks.solve(x, rhs)
        except RuntimeError as exc:
            return str(exc)
        return ""

    def flag():
        return H.HYPRE_GetError() if H is not None else 0

    ks = solver(A)
    x0, x, y = V.vector(), V.vector(), V.vector()
    n = ks.solve(x0, b)
    options = H is not None and mc.boomeramg_handle(ks._pc) is not None
    # the tolerance met at the last iteration allowed: the same solve, its count the limit
    ks.parameters["max_iter"] = n
    said = attempt(ks, x, b)
    e = x.axpy(-1.0, x0).norm("l2") / x0.norm("l2")
    check("a solve that meets its tolerance at the last iteration allowed is converged",
          not said and ks.converged and ks.iterations == n and e < 1e-12,
          "(%d iterations of %d, difference %.1e%s)" % (ks.iterations, n, e, said and "; " + said))
    # a zero right-hand side: hypre does not begin, and keeps the count and the residual
    # of the solve before, here one that met its tolerance and below one that did not
    y.set(1.0)
    none = [attempt(ks, y, V.vector()), ks.converged, ks.iterations, y.norm("l2")]
    # at its limit without the tolerance: an error, and no flag for the next solver's
    # options to be judged by (every setter of hypre's returns the flag)
    ks = solver(A, most=2)
    said, left, took = attempt(ks, V.vector(), b), flag(), ""
    try:
        solver(A, "relax_wt=1.0,pmax=6" if options else "")
    except ValueError as exc:
        took = str(exc)
    check("one that ends at its limit raises and leaves no error of hypre's behind: the next "
          "solver's options are taken", "in 2 iterations" in said and not ks.converged
          and left == 0 and not took, "(flag %d%s)" % (left, took and "; " + took))
    y.set(1.0)
    none += [attempt(ks, y, V.vector()), ks.converged, ks.iterations, y.norm("l2")]
    check("a zero right-hand side gives zero without an iteration, whatever the solve before",
          none == ["", True, 0, 0.0] * 2, "(%s)" % (none,))
    # A preconditioner that is not positive definite (Jacobi relaxation with a weight far
    # above what the matrix allows): <C r, r> turns negative, which hypre's test takes for
    # converged after an iteration or two, the true residual of order one.
    if options:
        ks = solver(A, "relax=7,relax_wt=1.8")
        x = V.vector()
        said = attempt(ks, x, b)
        res = residual(A, x, b)
        check("a solve that hypre's PCG leaves because the preconditioner is not positive definite "
              "is not taken for converged", "failed to converge" in said and not ks.converged
              and not res < 1e-3 and flag() == 0, "(residual %.1e; %s)" % (res, said or "no error"))
    # A matrix that is not positive definite, here the negative of one that is, with a
    # hierarchy built on it: <C b, b> is negative, hypre's PCG takes the right-hand side
    # for zero, returns it as the solution and gives no verdict.
    minus = mfem.ConstantCoefficient(-1.0)
    N = hp.assemble_native_matrix(V, [mfem.DiffusionIntegrator(minus), mfem.MassIntegrator(minus)])
    ks = solver(N)
    x = V.vector()
    said = attempt(ks, x, b)
    res = residual(N, x, b)
    check("nor is one that hypre's PCG does not begin because the matrix is not positive definite",
          "failed to converge" in said and not ks.converged and not res < 1e-3 and flag() == 0,
          "(residual %.1e; %s)" % (res, said or "no error"))
    if getattr(ks, "_native", False):
        # (hypre's PCG only: MFEM's CG stops the process at a sum that is not a number)
        bad = b.copy()
        if bad.local_size:
            bad.array[0] = float("nan")
        ks = solver(A)
        x = V.vector()
        said = attempt(ks, x, bad)
        check("nor one whose right-hand side is not a number, which hypre's PCG answers with zero",
              "failed to converge" in said and not ks.converged and flag() == 0,
              "(%s)" % (said or "no error"))
        # A number of iterations and no test, both tolerances zero: at that limit hypre
        # raises nothing, and it is still a solve that hypre made, with its count.  The
        # right-hand side that is not a number gives none after it, though hypre keeps
        # that count.
        ks = solver(A, most=5)
        ks.parameters["rel_tolerance"] = ks.parameters["abs_tolerance"] = 0.0
        ks.parameters["error_on_nonconvergence"] = False
        made = [ks.solve(V.vector(), b), ks.converged, ks.solve(V.vector(), bad), ks.converged, flag()]
        check("one that makes the iterations it is allowed, both tolerances zero, has their count, "
              "and is not taken for one that hypre does not begin",
              made == [5, False, 0, False, 0], "(%s)" % (made,))


def test_solve_mode_setups():
    """``HIPPYMFEM_GPU_AWARE_MPI=solve`` hands MPI device buffers during the solves on a
    hierarchy that is set up, and that is: set up for the solver that applies it.  MFEM
    sets a preconditioner up at its own first application, whatever hypre's PCG did with
    it before, so the first solve by one of MFEM's solvers is a set-up as well; and
    hypre's PCG sets it up at its own first solve, whatever MFEM did with it before.  The
    switch of the mode is recorded by a stand-in here, on any build and without a GPU
    (the mode itself runs on a device in ``benchmarks/bench_scaling.py`` alone, when asked)."""
    from hippymfem.common import mfemconfig as mc

    if RANK == 0:
        print("the solves that HIPPYMFEM_GPU_AWARE_MPI=solve leaves staged")
    A, b, V = make_operator(n=24)
    A2, b2, V2 = make_operator(n=24)
    calls = []

    def new(reuse=0):
        ks = hp.KrylovSolver(COMM, "cg", "amg")
        ks.parameters["rel_tolerance"] = 1e-10
        ks.parameters["pc_reuse"] = reuse
        return ks

    def aware(ks, x, rhs):
        del calls[:]
        ks.solve(x, rhs)
        return True in calls

    saved = mc.GPU_AWARE_SOLVES, mc.set_hypre_gpu_aware_mpi
    mc.GPU_AWARE_SOLVES, mc.set_hypre_gpu_aware_mpi = True, lambda on=True: calls.append(bool(on))
    try:
        # a solve from the vector it is given after solves from zero: MFEM's CG on the
        # preconditioner that hypre's PCG has set up, where the solver runs hypre's PCG
        ks = new().set_operator(A)
        got = [aware(ks, V.vector(), b), aware(ks, V.vector(), b)]
        ks.parameters["nonzero_initial_guess"] = True
        got += [aware(ks, V.vector(), b), aware(ks, V.vector(), b)]
        first = getattr(ks, "_native", False)
        check("the first solve by MFEM's CG on a hierarchy of hypre's PCG is a set-up, and staged",
              got == [False, True, not first, True], "(GPU-aware: %s%s)"
              % (got, "" if first else "; no PCG of hypre's here"))
        # a hierarchy kept for the next operator, which MFEM's CG applies
        ks = new(reuse=1).set_operator(A)
        got = [aware(ks, V.vector(), b)]
        first = getattr(ks, "_native", False)
        ks.set_operator(A2)
        got += [ks._pc_age == 1, aware(ks, V2.vector(), b2), aware(ks, V2.vector(), b2)]
        check("so is the first solve with a hierarchy kept from the operator before",
              got == [False, True, not first, True], "(GPU-aware: %s)" % (got,))
        # a hierarchy lent without the PCG of its owner
        owner = new().set_operator(A)
        got = [aware(owner, V.vector(), b)]
        first = getattr(owner, "_native", False)
        other = new().set_operator(A, pc=owner._pc)
        got += [aware(other, V.vector(), b), aware(other, V.vector(), b), aware(owner, V.vector(), b)]
        check("and the first solve of a solver that was lent one without its owner's PCG",
              got == [False, not first, True, True], "(GPU-aware: %s)" % (got,))
        # the other way round, a solve from zero after solves from the vector it is given:
        # hypre's PCG sets up again, at its own first solve, what MFEM's CG has set up
        ks = new().set_operator(A)
        ks.parameters["nonzero_initial_guess"] = True
        got = [aware(ks, V.vector(), b), aware(ks, V.vector(), b)]
        ks.parameters["nonzero_initial_guess"] = False
        got += [aware(ks, V.vector(), b), aware(ks, V.vector(), b)]
        first = getattr(ks, "_native", False)
        check("the first solve by hypre's PCG on a hierarchy of MFEM's CG is a set-up as well, and staged",
              got == [False, True, not first, True], "(GPU-aware: %s)" % (got,))
        # and so in a solver that is lent hierarchy and PCG by an owner with such solves only
        owner = new().set_operator(A)
        owner.parameters["nonzero_initial_guess"] = True
        got = [aware(owner, V.vector(), b)]
        other = new().set_operator(A, pc=owner._pc, donor=owner)
        first = getattr(other, "_native", False)
        got += [aware(other, V.vector(), b), aware(other, V.vector(), b)]
        check("so is the first solve of a solver that was lent a PCG which has not solved yet",
              got == [False, not first, True], "(GPU-aware: %s)" % (got,))
    finally:
        mc.GPU_AWARE_SOLVES, mc.set_hypre_gpu_aware_mpi = saved


def test_systems_ordering():
    """BoomerAMG for a vector space is told how its components are ordered
    (``KrylovSolver(systems_dim=..., systems_bynodes=...)``): one after another, which is
    the ordering of this library's spaces unless another is asked for, or interleaved.
    With the number of components alone hypre takes the rows for components in turn, and
    on a space ordered by nodes it then pairs rows of different components: the solve
    takes several times the iterations (21 and 19 where it takes 6 and 7, on these
    matrices on one rank in host memory)."""
    if RANK == 0:
        print("the ordering of a vector space's components, for BoomerAMG")
    one = mfem.ConstantCoefficient(1.0)
    n = 12 if NP <= 2 else 16
    took = {}
    for name, ordering in (("by nodes", mfem.Ordering.byNODES), ("interleaved", mfem.Ordering.byVDIM)):
        pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON))
        V = hp.FunctionSpace.H1(pm, 1, vdim=3, ordering=ordering)
        A = hp.assemble_native_matrix(V, [mfem.VectorDiffusionIntegrator(one), mfem.VectorMassIntegrator(one)])
        b = V.vector()
        hp.parRandom.set_seed(5)
        hp.parRandom.normal(1.0, b)
        for told in (True, False):
            ks = hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3, systems_bynodes=told)
            ks.parameters["rel_tolerance"] = 1e-10
            ks.parameters["max_iter"] = 500
            ks.set_operator(A)
            x = V.vector()
            its = ks.solve(x, b)
            took[(name, told)] = (its, residual(A, x, b))
        # a solver that is handed the space reads the ordering from it
        took[(name, None)] = hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3, fes=V.fes).systems_bynodes
    right = (took[("by nodes", True)][0], took[("interleaved", False)][0])
    wrong = (took[("by nodes", False)][0], took[("interleaved", True)][0])
    # (numpy's maximum is not a number where a residual is not: Python's keeps one in the
    # first place only)
    worst = float(np.max([r for _, r in (took[k] for k in took if k[1] is not None)]))
    # (half as many again is asked of the least of them, not twice: on two GPUs that was
    # 31 against 15, and a device's coarsening draws from a random stream that every
    # set-up before it has moved.  Were the default the wrong one, the two would change
    # places.)
    check("BoomerAMG told the ordering of a vector space's components solves in the same few iterations "
          "for both orderings, and in half as many again or more when it is told the other one",
          abs(right[0] - right[1]) <= 3 and min(wrong) >= 1.5 * max(right) and worst < 1e-8
          and took[("by nodes", None)] is True and took[("interleaved", None)] is False,
          "(%d and %d iterations told right, %d and %d told wrong; worst residual %.1e)"
          % (right + wrong + (worst,)))
    # the default is the ordering of the library's spaces, and a prior's solver is told its space's
    Vd = hp.FunctionSpace.H1(mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON)),
                             1, vdim=3)
    prior = hp.VectorBiLaplacianPrior(Vd, [0.1] * 3, [0.5] * 3, robin_bc=True)
    check("the default is by nodes, as the library's spaces are, and a vector prior's solver is told its space's",
          hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3).systems_bynodes
          and prior.Asolver.systems_dim == 3 and prior.Asolver.systems_bynodes
          and Vd.fes.GetOrdering() == mfem.Ordering.byNODES)
    # a FunctionSpace for fes, as the library's spaces are handed around, is read as its MFEM
    # space is (it raised AttributeError: a FunctionSpace has no GetOrdering)
    via_space = []
    for ordering in (mfem.Ordering.byNODES, mfem.Ordering.byVDIM):
        Vo = hp.FunctionSpace.H1(mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.HEXAHEDRON)),
                                 1, vdim=3, ordering=ordering)
        try:
            via_space.append(hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3, fes=Vo).systems_bynodes)
        except Exception as e:                                       # noqa: BLE001
            via_space.append(repr(e)[:70])
    check("a FunctionSpace as fes gives the ordering of its MFEM space", via_space == [True, False],
          "(%s)" % (via_space,))
    # the default taken without a word of the space or the ordering is said, once
    import warnings

    hp.KrylovSolver._warned_ordering = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3)
        hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3)
        hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3, systems_bynodes=True)
        hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3, fes=Vd)
    said = [str(w.message) for w in caught if "systems_dim" in str(w.message)]
    check("systems_dim without fes or systems_bynodes says once which ordering it takes",
          len(said) == 1 and "byVDIM" in said[0], "(%d warnings)" % len(said))
    # the solver's own settings hold for a systems hierarchy: with the components MFEM writes a strength
    # threshold of its own (0.5), which used to come after them and take their place without a word
    import ctypes
    from hippymfem.common import mfemconfig as mc

    H = mc._hypre_library()

    def threshold(**parameters):
        ks = hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3, systems_bynodes=False)
        for k, v in parameters.items():
            ks.parameters[k] = v
        ks.set_operator(A)
        handle, out = mc.boomeramg_handle(ks._pc), ctypes.c_double(-1.0)
        if H is None or handle is None:
            return None
        get = H.HYPRE_BoomerAMGGetStrongThreshold
        get.restype, get.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        get(ctypes.c_void_p(handle), ctypes.byref(out))
        return out.value

    held = (threshold(), threshold(amg_strength_threshold=0.7), threshold(amg_options="theta=0.6"))
    check("a systems hierarchy keeps the solver's own strength threshold, by the parameter and by amg_options, "
          "where MFEM's for a vector problem used to take its place",
          held[0] is None or (abs(held[0] - 0.5) < 1e-12 and abs(held[1] - 0.7) < 1e-12 and abs(held[2] - 0.6) < 1e-12),
          "(skipped: no handle of BoomerAMG)" if held[0] is None
          else "(hypre holds %g where nothing is asked, %g and %g where it is)" % held)
    # rows that are no multiple of the components: an error of every rank, where MFEM stops the process
    Vs = hp.FunctionSpace.H1(mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.HEXAHEDRON)), 1)
    As = hp.assemble_native_matrix(Vs, [mfem.DiffusionIntegrator(one), mfem.MassIntegrator(one)])
    odd = 2 if COMM.allreduce(int(As.Height() % 2 != 0), op=MPI.MAX) else 3 if COMM.allreduce(
        int(As.Height() % 3 != 0), op=MPI.MAX) else 0
    said = ""
    if odd:
        try:
            hp.KrylovSolver(COMM, "cg", "amg", systems_dim=odd).set_operator(As)
        except ValueError as e:
            said = str(e)
    check("rows that are no multiple of the components are an error, not MFEM's stop of the process",
          (not odd) or "systems_bynodes" in said, "(%s)" % (said[:70] if odd else "skipped: every rank's rows divide"))


def test_elasticity_options():
    """BoomerAMG's elasticity options (``KrylovSolver(elasticity=True)``: nodal coarsening
    and the rigid-body modes of the space in the interpolation) where hypre runs on the
    host.  With hypre on a GPU MFEM has none and stopped the process when asked for them;
    the solver keeps the systems options there and says so once."""
    if RANK == 0:
        print("BoomerAMG's elasticity options, and the systems options alone with hypre on a GPU")
    import warnings
    from hippymfem.algorithms import linSolvers as ls

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(8, 8, 8, mfem.Element.HEXAHEDRON))
    V = hp.FunctionSpace.H1(pm, 1, vdim=3)
    lam, mu, small = (mfem.ConstantCoefficient(c) for c in (1.0, 1.0, 1e-2))
    A = hp.assemble_native_matrix(V, [mfem.ElasticityIntegrator(lam, mu),
                                      mfem.VectorMassIntegrator(small)])
    b = V.vector()
    hp.parRandom.set_seed(9)
    hp.parRandom.normal(1.0, b)
    gpu = ls._hypre_on_gpu()
    took = []
    hp.KrylovSolver._warned_elasticity = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for elastic in (False, True, True):
            ks = hp.KrylovSolver(COMM, "cg", "amg", systems_dim=3, elasticity=elastic, fes=V)
            ks.parameters["rel_tolerance"] = 1e-10
            ks.parameters["max_iter"] = 500
            ks.set_operator(A)
            x = V.vector()
            its = ks.solve(x, b)
            took.append((its, residual(A, x, b)))
    said = [str(w.message) for w in caught if "elasticity" in str(w.message)]
    worst = float(np.max([r for _, r in took]))
    check("elasticity=True solves with BoomerAMG's elasticity options on the host, and with the systems "
          "options alone and one warning where hypre runs on a GPU",
          worst < 1e-8 and len(said) == (1 if gpu else 0),
          "(hypre on %s; %d, %d and %d iterations without and with them; worst residual %.1e; "
          "%d warning(s))" % ("a GPU" if gpu else "the host", took[0][0], took[1][0], took[2][0],
                              worst, len(said)))


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
    ``--enable-mixedint``).  With 64-bit global indices and two or more ranks, a matrix
    with global columns beyond 32 bits keeps them, in MFEM's library and in the copy in
    the single-precision one."""
    import ctypes

    import scipy.sparse as sp

    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.common import parvector as pv
    from hippymfem.common.linalg import _offd_block, hypre_to_scipy, scipy_to_hypre

    if RANK == 0:
        print("hypre's local and global index types")
    # (the two types are made from the widths MFEM reports, so their widths are those by
    # construction: what can go wrong is an index on its way, which the checks below see)
    narrow, wide = 8 * np.dtype(pv._HYPRE_INT).itemsize, 8 * np.dtype(pv._HYPRE_BIGINT).itemsize
    vals = [3, 5, 2 ** 31 - 1] + ([2 ** 40 + 7] if wide == 64 else [])
    arr = pv.bigint_array(vals)
    back = [int(v) for v in pv.bigint_view(arr.GetData(), len(vals))]
    check("global indices go to an MFEM array and back, up to the type's range", back == vals,
          "(%s; HYPRE_Int %d bits, HYPRE_BigInt %d bits)" % (back[-1], narrow, wide))

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

    # Global columns beyond 32 bits.  hypre counts the columns a rank owns in HYPRE_Int, so
    # only two or more ranks have a matrix with such a column, and no vector of its columns
    # fits a test: a few entries a rank, in its own last columns and in the middle of the
    # next rank's, go through scipy_to_hypre, the column map of the off-diagonal block, the
    # refusal of hypre_to_scipy and the copy in the single-precision library.  No product.
    if wide != 64 or NP == 1:
        check("global columns beyond 32 bits (skipped: %s)"
              % ("HYPRE_BigInt has 32 bits" if wide != 64 else "one rank owns every column, fewer than 2^31"),
              True)
        return
    mloc = 3 * 2 ** 29                          # the columns of a rank: 3 * 2^30 on two ranks
    M, c0 = mloc * NP, mloc * RANK
    far = mloc * ((RANK + 1) % NP) + mloc // 2
    cols = np.concatenate([[c0 + mloc - 1 - i, far + 7 * i, far + 7 * i + 3] for i in range(nloc)])
    W = sp.csr_matrix((data, (rows, cols)), shape=(nloc, M))
    W.sum_duplicates()
    B = scipy_to_hypre(W, col_starts=[c0, c0 + mloc])
    want = np.unique(cols[(cols < c0) | (cols >= c0 + mloc)])
    top = COMM.allreduce(int(want.max()), op=MPI.MAX)
    try:
        cmap = _offd_block(B)[1]
    except RuntimeError as e:                   # (a static MFEM: the map cannot be read)
        check("global columns beyond 32 bits (skipped: %s)" % e, True)
        return
    refused = False
    try:
        hypre_to_scipy(B)
    except ValueError:
        refused = True
    ok = (int(B.GetGlobalNumCols()) == M and B.Width() == mloc
          and [int(v) for v in B.GetColPartArray()] == [c0, c0 + mloc]
          and np.array_equal(cmap, want) and refused)
    check("a matrix with global columns beyond 32 bits keeps them in its partition and in the column "
          "map of its off-diagonal block, and hypre_to_scipy refuses it",
          COMM.allreduce(int(ok), op=MPI.SUM) == NP and top >= 2 ** 31,
          "(%d columns; the largest in a column map %d)" % (M, top))
    # its rows as the copy in the single-precision library takes them, and the copy
    tb = ss.matrix_blocks(B, COMM)
    nd = tb.nnz_diag
    T = (sp.csr_matrix((tb.values[:nd], tb.J_diag.astype(np.int64) + tb.c0, tb.I_diag), shape=W.shape)
         + sp.csr_matrix((tb.values[nd:], tb.cmap[tb.J_offd], tb.I_offd), shape=W.shape))
    err = abs(T - W).max()
    check("its rows read from the two blocks, with their global columns, are the rows put in",
          COMM.allreduce(int(err == 0.0 and tb.gnc == M), op=MPI.SUM) == NP, "(%g)" % err)
    lib = ss.library()
    if lib is None:
        check("the copy of such a matrix in the single-precision library (skipped: no library)", True)
        return
    if lib.bigint.itemsize < 8:
        raised = False
        try:
            ss.SingleParMatrix.from_pattern(tb, tb.values, lib)
        except ValueError:
            raised = True
        check("a single-precision library with 32-bit global indices refuses such a matrix", raised)
        return
    C = ss.SingleParMatrix.from_pattern(tb, tb.values, lib)
    held = np.array(np.ctypeslib.as_array((ctypes.c_longlong * tb.n_offd).from_address(
        lib.pointer(C.par.value, "par_col_map_offd"))))
    ok = np.array_equal(held, want) and C.global_cols == M and C.Width() == mloc
    C.destroy()
    check("the copy of such a matrix in the single-precision library has its column map",
          COMM.allreduce(int(ok), op=MPI.SUM) == NP)


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


def test_benchmarks():
    """The two benchmarks of the scaling runs on a small problem, for what they report about
    themselves (``benchmarks/bench_scaling.py`` and ``bench_newton_device.py``; skipped where
    the package is installed without them).  The clock on the ten V-cycles of
    ``--profile-setup`` holds no set-up.  The profiles of BoomerAMG and the second
    matrix-vector kernel run to the record, and a Jacobian in the single-precision hypre is
    left without the profiles.  The Newton benchmark reads the card's memory before the
    kernels of ``--rank-table`` take theirs, and counts the three functions of a Jacobian
    that is assembled with the products of a p-multigrid."""
    import ctypes
    import importlib.util
    import sys
    import tempfile

    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.common import devicebridge
    from hippymfem.common import mfemconfig as mc

    if RANK == 0:
        print("the benchmarks of the scaling runs on a small problem")
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "benchmarks")
    if not os.path.isfile(os.path.join(here, "bench_scaling.py")):
        check("the benchmarks report what they measure", True,
              "(skipped: no benchmarks directory beside the package)")
        return

    def load(name):
        # (a benchmark puts the root of the tree in front of the path: as it was afterwards)
        path = list(sys.path)
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, name + ".py"))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = path
        # their own reports stay out of the suite's, and no rank asks the vendor's tool
        module.say, module.gpu_name = (lambda *a: None), (lambda: "none")
        return module

    scaling, newton = load("bench_scaling"), load("bench_newton_device")
    single = bool(ss.HYPRE_SINGLE) and ss.library() is not None
    out = COMM.bcast(os.path.join(tempfile.gettempdir(), "_hippymfem_bench_%d.json" % os.getpid()),
                     root=0)

    def record(bench, *args):
        """What ``main`` of a benchmark writes for these arguments (the record, and with
        ``--rank-table`` every rank's row), and what stopped it, if anything did."""
        argv, device = sys.argv, hp.configure_device
        sys.argv = [bench.__name__, "--out", out] + list(args)
        hp.configure_device = lambda *a, **k: None          # (this process has its device)
        raised = None
        try:
            code = bench.main()
            raised = "main returned %r" % (code,) if code else None
        except (Exception, SystemExit) as e:                         # noqa: BLE001
            raised = "%s: %s" % (type(e).__name__, " ".join(str(e).split())[:160])
        finally:
            sys.argv, hp.configure_device = argv, device
        written = [None, None]
        if RANK == 0:
            for i, name in enumerate((out, os.path.splitext(out)[0] + ".ranks.json")):
                if os.path.exists(name):
                    with open(name) as f:
                        written[i] = json.load(f)
                    os.remove(name)
        rec, table = COMM.bcast(written, root=0)
        return rec or {}, table or [], raised

    # ---- the set-ups of --profile-setup.  hypre counts the nonzeros of a hierarchy again at
    # every set-up where a count was asked for: a count of one, put there before a clock
    # starts, is gone after it exactly when a set-up ran inside.
    A, b, V = make_operator(n=24)
    H = mc._hypre_library()
    made, clocks, plain = [], [], scaling.timed

    def make_pc(M):
        made.append(mfem.HypreBoomerAMG(M))
        made[-1].SetPrintLevel(0)
        return made[-1]

    def timed(fn, sync=None):
        h = ctypes.c_void_p(mc.boomeramg_handle(made[-1]))
        H.HYPRE_BoomerAMGSetCumNnzAP(h, 1.0)
        t, result = plain(fn, sync)
        nnz = ctypes.c_double(0.0)
        H.HYPRE_BoomerAMGGetCumNnzAP(h, ctypes.byref(nnz))
        clocks.append((len(made), t, nnz.value != 1.0))
        return t, result

    handle = H is not None and mc.boomeramg_handle(make_pc(A)) is not None
    del made[:]
    if not handle:
        check("the clock on the V-cycles of --profile-setup holds no set-up", True,
              "(skipped: this MFEM library does not give hypre's handle of a preconditioner)")
    else:
        H.HYPRE_BoomerAMGSetCumNnzAP.argtypes = [ctypes.c_void_p, ctypes.c_double]
        H.HYPRE_BoomerAMGGetCumNnzAP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        scaling.timed = timed
        try:
            prof = scaling.profile_setup("ref:", A, make_pc, b, V.vector())
        finally:
            scaling.timed = plain
        runs = (prof.get("ref") or {}).get("runs", [])
        # of each run: the set-ups inside its clocks, in their order; the first clock is on the
        # first solve, and the one whose tenth is reported as the V-cycle is looked up
        inside = [[(t, s) for n, t, s in clocks if n == i + 1] for i in range(len(runs))]
        first = [[s for _, s in c][:1] for c in inside]
        cycle = [[s for t, s in c if t / 10.0 == r["cycle"]] for c, r in zip(inside, runs)]
        check("--profile-setup has the set-up of a hierarchy in the first solve of hypre's PCG and "
              "none in the clock on its ten V-cycles (MFEM's own, at the first direct application, "
              "is outside it)",
              len(runs) == 2 and first == [[True]] * 2 and cycle == [[False]] * 2,
              "(a set-up inside the clocks of a run, in their order: %s)"
              % "; ".join(str([s for _, s in c]) for c in inside))
        del made[:]

    # ---- bench_scaling.py with its two profiles (they need that handle) and the other
    # matrix-vector kernel
    spmv = (mc.HYPRE_SPMV, mc.set_hypre_spmv)
    if mc.HYPRE_SPMV is None:
        # (in host memory hypre has one kernel and no switch: one is stood in for, so that
        # --both-kernels takes its second action)
        mc.HYPRE_SPMV, mc.set_hypre_spmv = "vendor", (lambda kernel="auto", comm=None: kernel)
    try:
        rec, _, raised = record(scaling, "--n", "6", "--ntargets", "20", "--repeats", "1",
                                "--both-kernels",
                                *(["--profile-vcycle", "--profile-setup", "ref:"] if handle else []))
    finally:
        mc.HYPRE_SPMV, mc.set_hypre_spmv = spmv
    levels = len((rec.get("vcycle") or {}).get("levels", []))
    sets = len(((rec.get("setup_profile") or {}).get("ref") or {}).get("runs", []))
    other = (rec.get("other_kernel") or {}).get("t_action", 0.0)
    none = (rec.get("vcycle"), rec.get("setup_profile")) == (None, None)
    check("bench_scaling.py runs to its record with the two profiles of BoomerAMG and the second "
          "kernel: the profiles of a Jacobian of MFEM's, and none of one in the single-precision "
          "hypre",
          raised is None and other > 0.0
          and (none if (single or not handle) else (levels > 0 and sets == 2)),
          "(%s)" % (raised or "V-cycles of %d hierarchies, %d set-ups, an action with the other "
                    "kernel in %.4f s%s"
                    % (levels, sets, other,
                       "; the Jacobian in the single-precision hypre" if single else "")))
    # A Jacobian in the single-precision hypre is multiplied from the benchmark through the
    # conversion of its two vectors, in a library that stages its exchange always and that
    # keeps the kernel it was loaded with: its product is one number under that name, and
    # the action with the second kernel says which kernel its solves kept.
    product = sorted(rec.get("t_product") or {})
    kept = (rec.get("other_kernel") or {}).get("single_hypre_spmv", "no word")
    check("and names a product with the Jacobian and the action with the second kernel for what "
          "was measured, of a Jacobian of MFEM's and of one in the single-precision hypre",
          bool(rec.get("hypre_single")) == single
          and ((len(product) == 1 and product[0].startswith("single") and kept == (spmv[0] or "vendor"))
               if single else ("staged" in product and kept is None)),
          "(products: %s; the kernel that the single-precision solves kept: %s)"
          % (", ".join(product) or "none", kept))

    # ---- bench_newton_device.py --rank-table: the card's memory against the two kernels of
    # the table (stood in for: the kernels are counted, and the memory read is their count),
    # and the stages of a Jacobian with a p-multigrid's products where a library makes one
    taken = []
    kept = (newton.card_probe, devicebridge.device_memory, os.environ.get("HIPPYMFEM_PMG"))
    newton.card_probe = lambda reps=7: taken.append(1) or {"triad": 1.0, "exp": 1.0}
    devicebridge.device_memory = lambda: (len(taken), 1)
    # The benchmark puts its clocks on functions of the problem's module, of three of the
    # library's classes and of gc, and as a script it leaves them there: what those names
    # hold now is put back after it, for the tests that run later in this process.
    import gc

    pvp = sys.modules[hp.PDEVariationalProblem.__module__]
    owners = [o for o in (pvp, getattr(pvp, "_FiniteFlag", None), hp.KrylovSolver, hp.ReducedHessian, gc)
              if o is not None]
    unclocked = [(o, k, v) for o in owners for k, v in list(vars(o).items()) if callable(v)]
    if single:
        os.environ["HIPPYMFEM_PMG"] = "1"
    try:
        rec, table, raised = record(newton, "--n", "6", "--steps", "1", "--ntargets", "20",
                                    "--newton-only", "--rank-table")
    finally:
        for o, k, v in unclocked:
            if vars(o).get(k) is not v:
                setattr(o, k, v)
        newton.card_probe, devicebridge.device_memory = kept[:2]
        if kept[2] is None:
            os.environ.pop("HIPPYMFEM_PMG", None)
        else:
            os.environ["HIPPYMFEM_PMG"] = kept[2]
    card = rec.get("card_used_total") or ["?"]
    cards = [(row.get("memory") or {}).get("card") or ["?"] for row in table]
    check("bench_newton_device.py --rank-table reads the card's memory, for its record and for "
          "every rank's row, before the kernels of the table take theirs",
          raised is None and len(taken) == 1 and card == [0, 1] and cards == [[0, 1]] * NP,
          "(%s)" % (raised or "kernels run before the record's read: %s; before the ranks' reads: %s"
                    % (card[0], sorted(set(str(c[0]) for c in cards)))))
    stages = rec.get("stages_in_steps") or {}
    calls = lambda k: int((stages.get(k) or {}).get("calls", 0))             # noqa: E731
    n, each = calls("+jacobian_with_products"), 2 + int(ss.PMG_KEEP_AP)
    plans, scatters, finished = (calls("+jacobian " + k)
                                 for k in ("plan_block", "scatter_derived", "finish_block"))
    if single and rec.get("single_engine") == "PMGEngine" and rec.get("pmg_element_products"):
        check("and counts the three functions of a Jacobian that is assembled with the products of "
              "a p-multigrid: its plans, its scatter and its matrices",
              n > 0 and scatters == n and plans == finished == each * n,
              "(%d such Jacobians in the steps: %d plans, %d scatters, %d matrices)"
              % (n, plans, scatters, finished))
    else:
        check("and counts the three functions of a Jacobian that is assembled with the products of "
              "a p-multigrid", raised is None and n == 0,
              "(skipped: %s)" % ("no single-precision hypre is named" if not single
                                 else "the Jacobian came without the products"))


def test_hypre_pcg_breakdown():
    """hypre's PCG stops early when the matrix is not positive definite, with a count
    below the limit: that is a failure, not a convergence (it was reported as one, and
    an incremental or prior solve went on with a wrong vector)."""
    from hippymfem.algorithms import linSolvers

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(16, 16, mfem.Element.TRIANGLE))
    V = hp.FunctionSpace.H1(pm, 1)
    A = hp.assemble_native_matrix(V, [mfem.DiffusionIntegrator(),
                                      mfem.MassIntegrator(mfem.ConstantCoefficient(-200.0))])
    hp.parRandom.set_seed(7)
    b = V.vector()
    hp.parRandom.normal(1.0, b)
    old = linSolvers.set_hypre_pcg(True)
    try:
        s = hp.KrylovSolver(COMM, "cg", "amg")
        s.parameters["max_iter"] = 200
        s.parameters["error_on_nonconvergence"] = False
        s.set_operator(A)
        x = V.vector()
        s.solve(x, b)
        native = bool(getattr(s, "_native", False))
        bad = s.converged
        # and a positive definite system still converges on the same path
        Apd = hp.assemble_native_matrix(V, [mfem.DiffusionIntegrator(), mfem.MassIntegrator()])
        s2 = hp.KrylovSolver(COMM, "cg", "amg")
        s2.set_operator(Apd)
        s2.solve(x, b)
        good = s2.converged
    finally:
        linSolvers.set_hypre_pcg(old)
    check("hypre's PCG breaking off on an indefinite matrix is a failure",
          native and not bad and good, "(native %s, indefinite %s, definite %s)"
          % (native, bad, good))


def test_systems_amg_ordering():
    """BoomerAMG's systems mode is told the dof ordering of the space: byNODES, the
    default here, was taken for interleaved, and the iterations ran into the hundreds."""
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(32, 32, mfem.Element.TRIANGLE))
    V = hp.FunctionSpace.H1(pm, 2, vdim=2)
    A = hp.assemble_native_matrix(V, [mfem.VectorDiffusionIntegrator(),
                                      mfem.VectorMassIntegrator()])
    hp.parRandom.set_seed(5)
    b = V.vector()
    hp.parRandom.normal(1.0, b)
    s = hp.KrylovSolver(COMM, "cg", "amg", systems_dim=2)
    s.parameters["rel_tolerance"] = 1e-10
    s.set_operator(A)
    x = V.vector()
    its = s.solve(x, b)
    prior = hp.VectorBiLaplacianPrior(V, [0.1, 0.1], [1.0, 1.0])
    prior.Asolver.solve(x, b)
    its_prior = prior.Asolver.iterations
    check("systems AMG on a vector space ordered by nodes", its < 30 and its_prior < 30,
          "(%d and %d iterations)" % (its, its_prior))


def main():
    test_gather()
    test_exact_solve()
    test_guards()
    test_inverse_problem()
    test_hypre_pool_logic()
    test_gpu_aware_mpi_check()
    test_boomeramg_options()
    test_refused_settings()
    test_hypre_pcg_verdict()
    test_solve_mode_setups()
    test_systems_ordering()
    test_elasticity_options()
    test_chebyshev_solver()
    test_index_widths()
    test_single_precision_solves()
    test_lumped_mass()
    test_transpose_solver()
    test_hypre_pcg_breakdown()
    test_systems_amg_ordering()
    test_petsc()
    test_benchmarks()
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

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The solves in a single-precision hypre against the double-precision ones, for the
suites that run with hypre on the host (``test_solvers``) and on a device
(``test_device``).  Skipped when no library is named (``HIPPYMFEM_HYPRE_SINGLE``)."""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

import hippymfem as hp
from hippymfem.modeling.variables import ADJOINT, PARAMETER


def largest(values):
    """The largest of ``values``, and NaN where one of them is NaN.  Python's ``max`` keeps
    the argument it has unless the next one compares greater, which a NaN never does:
    ``max(0.0, nan)`` is 0.0, and a check ``max(...) < tol`` passes on a result that is
    not a number anywhere but in the first place."""
    values = [float(v) for v in values]
    return float("nan") if any(v != v for v in values) else max(values)


def run(check, COMM=MPI.COMM_WORLD):
    """The Jacobian is assembled into the single-precision library and exists there
    alone, the forward and the adjoint solve are corrected against double-precision
    residuals and agree with the double-precision solves, an incremental solve agrees
    to the accuracy of single precision, and Newton-CG takes the same steps."""
    RANK = COMM.rank
    from hippymfem.algorithms import singlesolve
    from hippymfem.modeling.variables import PARAMETER

    if RANK == 0:
        print("solves in a single-precision hypre")
    if not singlesolve.HYPRE_SINGLE:
        if RANK == 0:
            print("  [ok  ] skipped: HIPPYMFEM_HYPRE_SINGLE is not set")
        return
    why = singlesolve.unsupported()
    if why:
        # (a device that is neither CUDA nor HIP: the library is refused before it is
        # loaded, with a warning)
        check("the single-precision library is refused on this build, and the solves stay "
              "in double precision", singlesolve.library() is None and why in singlesolve.why_not()
              and singlesolve.action_tolerance(1e-9) == 1e-9, "(%s)" % why)
        return
    check("the single-precision library loads", singlesolve.library() is not None,
          "(%s)" % (singlesolve.why_not() or singlesolve.HYPRE_SINGLE))
    if singlesolve.library() is None:
        return
    import jax.numpy as jnp

    # hypre keeps its error flag per process, and an exception belongs to one rank:
    # what one rank finds alone has to stop them all, or the others wait for it in
    # their next exchange (a line search that backtracked on one rank only)
    lib, last = singlesolve.library(), COMM.size - 1

    def raised(call):
        try:
            call()
        except RuntimeError:
            return 1
        return 0

    stopped = [COMM.allreduce(raised(lambda: lib.check(7 if RANK == last else 0, "a probe", COMM)),
                              op=MPI.MIN),
               COMM.allreduce(raised(lambda: lib.agree(
                   COMM, ValueError("a probe") if RANK == last else None)), op=MPI.MIN),
               COMM.allreduce(raised(lambda: lib.check(0, "a probe", COMM)), op=MPI.MAX),
               COMM.allreduce(raised(lambda: lib.agree(COMM, None)), op=MPI.MAX)]
    check("a failure on one rank raises on every rank, and none without one",
          stopped == [1, 1, 0, 0], "(%s, on %d rank(s))" % (stopped, COMM.size))

    # A rank that owns no row of a matrix holds vectors of no entries, and MFEM has no
    # pointer for such a vector where it has allocated nothing for it (None: with hypre
    # on a device for every one of them, since no device memory is made for no entries).
    # A vector of the library without entries takes and gives nothing, whatever it is
    # handed.
    none, said = singlesolve.SingleVector(lib, COMM, 0, [0, 0]), ""
    try:
        none.set(mfem.Vector(0))
        none.get(mfem.Vector(0))
    except TypeError as e:
        said = "TypeError: %s" % e
    none.destroy()
    check("a vector of the library without entries (a rank without rows) takes and gives nothing, also "
          "where MFEM has no pointer for its own", not said, "(%s)" % (said or "set and get return"))

    # The options of a hierarchy in this library (the solver's, then HIPPYMFEM_SINGLE_AMG):
    # hypre's setters of the relaxation and of its sweeps reset those of the coarsest level,
    # so the coarsest level's own hold wherever they were written, and an error that an
    # earlier call of the library left up is not taken for a refusal of an option.
    import ctypes

    prm = hp.KrylovSolver(COMM, "cg", "amg").parameters
    prm["amg_options"] = "coarse_relax=7,coarse_sweeps=4,relax=18,sweeps=2"
    lib.H.HYPRE_BoomerAMGSetMaxLevels(ctypes.c_void_p(), 0)       # (no solver: hypre raises its flag)
    up, said, got = int(lib.H.HYPRE_GetError()), "", ()
    try:
        amg = singlesolve._boomeramg(lib, prm, gather=False)
        value = ctypes.c_int()
        for name, k in (("HYPRE_BoomerAMGGetCycleRelaxType", 1), ("HYPRE_BoomerAMGGetCycleRelaxType", 3),
                        ("HYPRE_BoomerAMGGetCycleNumSweeps", 1), ("HYPRE_BoomerAMGGetCycleNumSweeps", 3)):
            fn = getattr(lib.H, name)
            fn.restype, fn.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            fn(amg, ctypes.byref(value), k)
            got += (value.value,)
        lib.H.HYPRE_BoomerAMGDestroy(amg)
    except ValueError as e:
        said = str(e)
    lib.H.HYPRE_ClearAllErrors()
    check("the coarsest level's relaxation and sweeps hold when they are written before the others', and an "
          "error left up by an earlier call is not taken for a refusal",
          up != 0 and not said and got == (18, 7, 2, 4),
          "(%s; the flag was %d%s)" % (", ".join("%d" % v for v in got), up, said and "; " + said))

    n = 10
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON))
    Vu, Vm = hp.FunctionSpace.H1(pmesh, 2), hp.FunctionSpace.H1(pmesh, 1)
    bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hp.PDEVariationalProblem([Vu, Vm, Vu],
                                   lambda u, m, p, x: jnp.exp(m.val) * hp.inner(u.grad, p.grad),
                                   bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hp.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12, max_iter=500)
    prior = hp.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    hp.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    B = hp.assemblePointwiseObservation(Vu, np.column_stack([rng.uniform(0.1, 0.9, 40) for _ in range(3)]))
    pde.single_solves = False
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    model = hp.Model(pde, prior, hp.DiscreteStateObservation(B, data, nstd ** 2))
    m = prior.mean.copy().axpy(0.5, mtrue.copy().axpy(-1.0, prior.mean))
    rhs = Vu.vector()
    hp.parRandom.normal(1.0, rhs)
    pde.bc0.zero(rhs)
    dm = Vm.vector()
    hp.parRandom.normal(1.0, dm)
    got = {}
    for single in (False, True):
        pde.single_solves = single
        pde.invalidate_jacobian()
        pde.release_linearization_point()
        pde._release_operators("solver", "solver_adj")
        u, p = pde.generate_state(), pde.generate_adjoint()
        pde.solveFwd(u, [u, m, None])
        pde.solveAdj(p, [u, m, None], rhs)
        A, _ = pde._jacobian([u, m, None])
        pde.setLinearizationPoint([u, m, p], gauss_newton_approx=False)
        Cdm, uh = Vu.vector(), Vu.vector()
        pde.apply_ij(ADJOINT, PARAMETER, dm, Cdm)
        pde.solveIncremental(uh, Cdm, False)
        params = hp.ReducedSpaceNewtonCG_ParameterList()
        params["rel_tolerance"] = 1e-7
        params["max_iter"] = 40
        params["print_level"] = -1
        solver = hp.ReducedSpaceNewtonCG(model, params)
        x = solver.solve([None, prior.mean.copy(), None])
        got[single] = dict(kind=type(A).__name__, u=u.copy(), p=p.copy(), uh=uh.copy(),
                           m=x[PARAMETER].copy(), newton=solver.it, cg=solver.total_cg_iter,
                           converged=solver.converged, passes=pde.fwd_iterations)
        del A
    # back to double precision at a point reached with single-precision solves, as for
    # the stages of a Laplace approximation after the MAP point
    was = pde.set_single_solves(False)
    u, p = got[True]["u"].copy(), got[True]["p"].copy()
    pde.setLinearizationPoint([u, m, p], gauss_newton_approx=False)
    A, _ = pde._jacobian([u, m, None])
    Cdm, uh = Vu.vector(), Vu.vector()
    pde.apply_ij(ADJOINT, PARAMETER, dm, Cdm)
    pde.solveIncremental(uh, Cdm, False)
    back = uh.copy().axpy(-1.0, got[False]["uh"]).norm("l2") / got[False]["uh"].norm("l2")
    check("with the single-precision solves switched off the Jacobian is in double precision again",
          type(A).__name__ == "HypreParMatrix" and back < 1e-8,
          "(%s; the incremental solve against the first %.1e)" % (type(A).__name__, back))
    del A
    again = pde.set_single_solves(True)
    A, _ = pde._jacobian([u, m, None])
    check("set_single_solves reports the old choice, and switched on again the Jacobian is in the "
          "single-precision library",
          was is True and again is False and type(A).__name__ == "SingleParMatrix",
          "(%s, %s; %s)" % (was, again, type(A).__name__))
    del A

    def rel(key):
        d, s = got[False][key], got[True][key]
        return s.copy().axpy(-1.0, d).norm("l2") / d.norm("l2")

    check("the Jacobian is a matrix of the single-precision library",
          got[True]["kind"] == "SingleParMatrix" and got[False]["kind"] == "HypreParMatrix",
          "(%s; %s without it)" % (got[True]["kind"], got[False]["kind"]))
    eu, ep, eh, em = rel("u"), rel("p"), rel("uh"), rel("m")
    check("the refined forward and adjoint solves agree with double precision",
          eu < 1e-9 and ep < 1e-9, "(state %.1e, adjoint %.1e)" % (eu, ep))
    check("an incremental solve agrees to the accuracy of single precision",
          1e-12 < eh < 1e-3, "(%.1e)" % eh)
    check("Newton-CG reaches the same MAP point in the same steps",
          got[True]["converged"] and got[False]["converged"] and em < 1e-4
          and got[True]["newton"] == got[False]["newton"]
          and abs(got[True]["cg"] - got[False]["cg"]) <= max(2, got[False]["cg"] // 20),
          "(%d Newton and %d CG iterations, %d and %d in double; MAP differs by %.1e)"
          % (got[True]["newton"], got[True]["cg"], got[False]["newton"], got[False]["cg"], em))

    # Newton-CG may stop the refinement of the forward and the adjoint solve early while
    # it runs (single_refine_goal, off by default; never above 1e3 times the square of
    # its tolerance): fewer passes, the same MAP point, and the problem's own goal back
    # when it returns.
    params = hp.ReducedSpaceNewtonCG_ParameterList()
    params["rel_tolerance"] = 1e-6
    params["max_iter"] = 40
    params["print_level"] = -1
    params["single_refine_goal"] = 1e-9
    solver = hp.ReducedSpaceNewtonCG(model, params)
    x = solver.solve([None, prior.mean.copy(), None])
    eg = x[PARAMETER].copy().axpy(-1.0, got[False]["m"]).norm("l2") / got[False]["m"].norm("l2")
    check("with single_refine_goal Newton-CG refines in fewer passes to the same MAP point",
          solver.converged and pde.fwd_iterations < got[True]["passes"] and eg < 1e-3
          and "SINGLE_REFINE_GOAL" not in pde.__dict__ and pde.SINGLE_REFINE_GOAL == 0.0,
          "(%d passes instead of %d; MAP differs by %.1e; the problem's goal afterwards %g)"
          % (pde.fwd_iterations, got[True]["passes"], eg, pde.SINGLE_REFINE_GOAL))

    # Every single-precision matrix of a pattern lends the pattern's one copy of the
    # column indices (singlesolve.SHARE_COLUMNS), also two that live at once, and a
    # matrix made after the others are gone finds it still.
    from hippymfem.algorithms import singlesolve as ss

    if ss.SHARE_COLUMNS:
        pde.invalidate_jacobian()
        A1, _ = pde._jacobian([got[True]["u"], m, None])
        pde._jac_cache = None              # (A1 stays alive, held here)
        A2, _ = pde._jacobian([got[True]["u"], m, None])
        ys = []
        for A in (A1, A2):
            y = Vu.vector()
            A.Mult(rhs.hypre, y.hypre)
            ys.append(y)
        shared = A1._columns is not None and A1._columns is A2._columns
        del A1, A2
        pde.invalidate_jacobian()
        A3, _ = pde._jacobian([got[True]["u"], m, None])
        y3 = Vu.vector()
        A3.Mult(rhs.hypre, y3.hypre)
        again = A3._columns is not None
        del A3
        pde.invalidate_jacobian()
        d = largest([ys[0].copy().axpy(-1.0, ys[1]).norm("l2"), ys[0].copy().axpy(-1.0, y3).norm("l2")])
        check("the single-precision matrices of a pattern share its column indices",
              shared and again and d == 0.0,
              "(two live ones share: %s; a later one finds them: %s; products differ by %.1e)"
              % (shared, again, d))

    # On a device the Jacobian is accumulated in single precision where it is assembled
    # a chunk of elements at a time (the route of a mesh that does not fit whole; forced
    # here by a small chunk).  Accumulated in double precision and rounded once it is
    # the same matrix up to rounding.
    from hippymfem.fem import kernel as km
    from hippymfem.fem import pattern as pat
    from hippymfem.fem import tdofassemble as td

    if td.device_finish():
        seen, begin = [], pat.ScatterPattern.fused_begin

        def fused_begin(self, dtype=None):
            seen.append(np.dtype(np.float64 if dtype is None else dtype).name)
            return begin(self, dtype)

        products = []
        # (smaller than the elements of any rank: on eight ranks a rank has 125 of them
        # here, and a batch that is not split is not accumulated chunk by chunk)
        old_chunk, km.ELEMENT_CHUNK = km.ELEMENT_CHUNK, min(
            128, max(2, COMM.allreduce(pmesh.GetNE(), op=MPI.MIN) // 2))
        # (a chunk size a kernel has planned already goes before the one set here)
        kernels = list(getattr(pde.kernel, "group_kernels", []))
        planned = [dict(gk._chunk) for gk in kernels]
        for gk in kernels:
            gk._chunk.clear()
        pat.ScatterPattern.fused_begin = fused_begin
        try:
            for accumulate in (True, False):
                old, td.SINGLE_ACCUMULATE = td.SINGLE_ACCUMULATE, accumulate
                try:
                    pde.invalidate_jacobian()
                    A, _ = pde._jacobian([got[True]["u"], m, None])
                finally:
                    td.SINGLE_ACCUMULATE = old
                y = Vu.vector()
                A.Mult(rhs.hypre, y.hypre)
                products.append((type(A).__name__, y))
                del A
        finally:
            km.ELEMENT_CHUNK = old_chunk
            pat.ScatterPattern.fused_begin = begin
            for gk, old in zip(kernels, planned):
                gk._chunk.clear()
                gk._chunk.update(old)
        pde.invalidate_jacobian()
        d = (products[0][1].copy().axpy(-1.0, products[1][1]).norm("l2")
             / products[1][1].norm("l2"))
        check("accumulated in single precision the Jacobian is the one rounded from double precision",
              products[0][0] == "SingleParMatrix" and seen == ["float32", "float64"] and d < 1e-5,
              "(accumulators %s; a product differs by %.1e)" % (seen, d))

    # The verdict of a solve.  hypre's PCG works with the square of the right-hand side's
    # size and of the residual's, which leave the range of single precision long before
    # the vectors do: it breaks off after a few iterations on a right-hand side of size
    # 1e-16 and takes one of size 1e-30 for zero.  Such a system is solved in units in
    # which the size is near one, and a zero right-hand side gives zero without an
    # iteration.
    S, _ = pde._jacobian([got[True]["u"], m, None])

    def krylov(S):
        ks = hp.KrylovSolver(COMM, "cg", "amg")
        ks.parameters["rel_tolerance"] = 1e-5
        ks.parameters["abs_tolerance"] = 0.0
        ks.parameters["max_iter"] = 200
        return ks.set_operator(S)

    ks = krylov(S)
    one, y = Vu.vector(), Vu.vector()
    its, off, powers = [ks.solve(one, rhs)], [], (-100, -60, 60, 100)
    for power in powers:
        its.append(ks.solve(y, rhs.copy().scale(2.0 ** power)))
        off.append(y.scale(2.0 ** -power).axpy(-1.0, one).norm("l2") / one.norm("l2"))
    none = ks.solve(y, Vu.vector())
    check("right-hand sides of size 1e-30 to 1e30 are solved as one of size one, and zero gives zero",
          max(abs(k - its[0]) for k in its[1:]) <= 1 and largest(off) < 1e-3 and none == 0
          and ks.converged and y.norm("l2") == 0.0,
          "(%d iterations, and %s for the sizes 2^%s; the solutions differ by at most %.1e; "
          "zero: %d iterations)" % (its[0], its[1:], list(powers), largest(off), none))
    del ks, S

    # With a preconditioner that is not positive definite hypre's PCG stops after a few
    # iterations and reports convergence, the true residual of order one.  Plain Jacobi
    # relaxation (relax=7) makes one where the largest eigenvalue of D^-1 A is above 2:
    # here with a parameter three times the sample (6.9).  The solver says so, and the
    # problem's forward solve goes through double precision instead.
    import warnings

    big = mtrue.copy().scale(3.0)
    pde.set_single_solves(False)
    pde.invalidate_jacobian()
    pde._release_operators("solver", "solver_adj")
    ud = pde.generate_state()
    pde.solveFwd(ud, [ud, big, None])
    pde.set_single_solves(True)
    said, eu = "", float("nan")
    options, singlesolve.AMG_OPTIONS = (singlesolve.AMG_OPTIONS,
                                        singlesolve.parse_amg_options("relax=7,pmax=6"))
    try:
        pde.invalidate_jacobian()
        pde._release_operators("solver", "solver_adj")
        S, _ = pde._jacobian([ud, big, None])
        ks = krylov(S)
        try:
            ks.solve(y, rhs)
        except RuntimeError as e:
            said = str(e)
        accepted = ks.converged
        del ks, S
        us = pde.generate_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # (said once per process)
            pde.solveFwd(us, [us, big, None])
        eu = us.copy().axpy(-1.0, ud).norm("l2") / ud.norm("l2")
        # the other way out (SINGLE_GIVE_UP = "raise"): the forward solve raises, which a
        # line search answers by backtracking, and no Jacobian is assembled in double
        raised = ""
        pde.SINGLE_GIVE_UP = "raise"
        try:
            pde.invalidate_jacobian()
            pde._release_operators("solver", "solver_adj")
            ur = pde.generate_state()
            pde.solveFwd(ur, [ur, big, None])
        except RuntimeError as e:
            raised = str(e)
        finally:
            del pde.SINGLE_GIVE_UP             # (the class's default again)
        # "auto", the default, raises as well where a double-precision Jacobian would not
        # fit beside what the device holds: here made so by the hook of the free memory
        auto_raised = ""
        pde._device_free_bytes = lambda: 0
        try:
            pde.invalidate_jacobian()
            pde._release_operators("solver", "solver_adj")
            ua = pde.generate_state()
            pde.solveFwd(ua, [ua, big, None])
        except RuntimeError as e:
            auto_raised = str(e)
        finally:
            del pde._device_free_bytes
    finally:
        singlesolve.AMG_OPTIONS = options
        pde.invalidate_jacobian()
        pde._release_operators("solver", "solver_adj")
    check("a solve that hypre's PCG abandons is not taken for converged, and the forward solve "
          "falls back on double precision",
          not accepted and "not positive definite" in said and eu < 1e-9,
          "(the state against double precision %.1e; %s)" % (eu, said[said.find("failed"):] or "no error"))
    check("with SINGLE_GIVE_UP = \"raise\" that forward solve raises instead, on every rank",
          COMM.allreduce("cannot carry" in raised and "not positive definite" in raised, op=MPI.LAND)
          and pde.SINGLE_GIVE_UP == "auto", "(%s)" % (raised[:90] or "nothing raised"))
    check("with SINGLE_GIVE_UP = \"auto\" (the default) it raises where a double-precision "
          "Jacobian does not fit, on every rank, and falls back on double precision where it "
          "does (above)",
          COMM.allreduce("does not fit" in auto_raised, op=MPI.LAND),
          "(%s)" % (auto_raised[:120] or "nothing raised"))

    _row_sums(check, COMM, pde, Vu, m, rhs, got)
    _pmg(check, COMM, pde, model, prior, pmesh, Vu, Vm, m, rhs, got)
    _pmg_meshes(check, COMM)
    _prior_cycle(check, COMM)
    _rank_without_coarse_rows(check, COMM)
    _single_blocks(check, COMM)


def _row_sums(check, COMM, pde, Vu, m, rhs, got):
    """The defect of the Jacobian's row sums (``tdofassemble.SINGLE_ROW_SUMS``): what the
    rounding of its values to single precision takes from the sum of each row, kept as a
    vector of the library and added by the product of hypre's PCG
    (``singlesolve.RowDefect``)."""
    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.common import devicebridge as bridge
    from hippymfem.common.parvector import allreduce_extreme
    from hippymfem.fem import tdofassemble as td

    lib = ss.library()
    u = got[True]["u"]

    def jacobian(rows, single=True):
        old, td.SINGLE_ROW_SUMS = td.SINGLE_ROW_SUMS, rows
        was = pde.set_single_solves(single)
        try:
            pde.invalidate_jacobian()
            pde._release_operators("solver", "solver_adj")
            A, _ = pde._jacobian([u, m, None])
        finally:
            td.SINGLE_ROW_SUMS = old
            pde.set_single_solves(was)
        pde._jac_cache = None                  # (the matrix stays alive, held by the caller)
        return A

    def memory(address, n, dtype):
        """``n`` entries of the library's memory as a numpy array."""
        if not n:
            return np.zeros(0, dtype=dtype)
        if lib.device:
            bridge.synchronize()
            return np.asarray(bridge.copy_to_jax(address, n, dtype))
        return np.array(np.ctypeslib.as_array(
            (np.ctypeslib.as_ctypes_type(np.dtype(dtype)) * n).from_address(int(address))))

    def stored_row_sums(S):
        """The sums of the rows of the values the library holds, in double precision."""
        n = S.Height()
        out = np.zeros(n)
        for name in ("par_diag", "par_offd"):
            csr = lib.pointer(S.par.value, name)
            nnz = lib.integer(csr, "csr_num_nonzeros")
            if nnz:
                I = memory(lib.pointer(csr, "csr_i"), n + 1, np.int32).astype(np.int64)
                run = np.concatenate(([0.0], np.cumsum(
                    memory(lib.pointer(csr, "csr_data"), nnz, np.float32).astype(np.float64))))
                out += run[I[1:]] - run[I[:-1]]
        return out

    def norm(a):
        return float(np.sqrt(COMM.allreduce(float(np.dot(a, a)))))

    S, plain = jacobian(True), jacobian(False)
    D = jacobian(False, single=False)          # the same values in double precision
    rd = S.row_defect
    ones, y, x = Vu.vector(), Vu.vector(), Vu.vector()
    ones.set(1.0)
    D.Mult(ones.hypre, y.hypre)
    exact = y.array.copy()
    c = rd.values().astype(np.float64) if rd is not None else np.zeros(0)
    want = exact - stored_row_sums(S) if rd is not None else np.zeros(0)
    without = exact - stored_row_sums(plain)
    # (a rank's NaN has to reach every rank: MPI's MAX of one may drop it)
    size = allreduce_extreme(COMM, float(np.max(np.abs(want))) if want.size else 0.0, MPI.MAX)
    off = allreduce_extreme(COMM, float(np.max(np.abs(c - want))) if want.size else 0.0, MPI.MAX)
    from hippymfem.common.linalg import get_diagonal

    d = get_diagonal(D, Vu.vector()).array.copy()
    scale = allreduce_extreme(COMM, float(np.max(np.abs(d))) if d.size else 0.0, MPI.MAX)
    check("a Jacobian of the single-precision library keeps what the rounding of its values took "
          "from the sum of each row, where it is asked to, and no vector otherwise",
          rd is not None and plain.row_defect is None and 0.0 < size < 1e-5 * scale
          and off <= 1e-4 * size and norm(without) > 0.0,
          "(the defect at most %.1e of the largest diagonal entry; the vector against the sums of "
          "the stored rows %.1e of that; without it the rows are off by %.1e of it)"
          % (size / scale, off / max(size, 1e-300), norm(without) / max(norm(want), 1e-300)))

    # The product of hypre's PCG adds the vector.  With one put there by hand, half the
    # diagonal, the solver solves (A + diag(c)) x = b, as a residual in double precision
    # says; the solver of the matrix without a defect solves A x = b, far from that.
    lib.put(rd.c.data, np.ascontiguousarray(0.5 * d, dtype=np.float32))
    rd.finish()
    left = []
    for M in (S, plain):
        eng = ss.SingleEngine(M, pde.solver.parameters)
        try:
            its, _ = eng.solve(x.hypre, rhs.hypre, 1e-5, 0.0, 200)
            own = eng._product is not None
        finally:
            eng.destroy()
        D.Mult(x.hypre, y.hypre)
        left.append((norm(rhs.array - y.array - 0.5 * d * x.array) / norm(rhs.array), its, own))
    check("the product of hypre's PCG adds that vector: with half the diagonal in its place the "
          "system with the diagonal half as large again is the one solved",
          left[0][2] and not left[1][2] and left[0][0] < 1e-3 < 0.05 < left[1][0],
          "(its residual %.1e after %d iterations; %.1e with the matrix alone, %d iterations)"
          % (left[0][0], left[0][1], left[1][0], left[1][1]))

    # The ranks of one assembly need not hand values of one kind: a rank that finishes on
    # the host stands beside ranks on their cards, and values in single precision, which
    # keep no defect, beside values in double.  What the matrix makes of them is one for
    # all ranks (SingleParMatrix._set_row_defect): the product that finds a card's defect
    # is an exchange with a reduced check, and hypre's PCG with the defect's product on
    # some ranks only would leave out a residual that the others exchange for.  Here the
    # last rank hands another kind than the others, of values a 2^-27th above numbers of
    # single precision: the rounding takes exactly that, so the defect of a row is its sum
    # over 2^27, and zero on a rank whose values came in single precision.
    from hippymfem.fem.kernel import _put

    tb = ss.matrix_blocks(D, COMM)
    S0 = ss.SingleParMatrix.from_pattern(tb, tb.values, lib)
    S0.Mult(ones.hypre, y.hypre)
    sums = y.array.copy()
    S0.Mult(rhs.hypre, y.hypre)
    ref = y.copy()
    lifted = tb.values.astype(np.float64) * (1.0 + 2.0 ** -27)
    card = _put if lib.device else (lambda a: a)
    # (as above, the reductions that carry a rank's NaN to every rank)
    size = 2.0 ** -27 * allreduce_extreme(COMM, float(np.max(np.abs(sums))) if sums.size else 0.0, MPI.MAX)
    odd, mixed = COMM.size > 1 and COMM.rank == COMM.size - 1, []
    for name, other in (("in host memory", lambda: lifted), ("in single precision", lambda: card(tb.values))):
        M = ss.SingleParMatrix.from_pattern(tb, other() if odd else card(lifted), lib, row_sums=True)
        kept = COMM.allgather(M.row_defect is not None)
        want = 0.0 if odd and name == "in single precision" else 2.0 ** -27 * sums
        off = allreduce_extreme(COMM, float(np.max(np.abs(M.row_defect.values() - want)))
                                if kept[COMM.rank] and sums.size else 0.0 if kept[COMM.rank] else float("inf"),
                                MPI.MAX)
        M.Mult(rhs.hypre, y.hypre)
        dv = y.axpy(-1.0, ref).norm("l2") / ref.norm("l2")
        its, own, done = -1, [None], False
        if len(set(kept)) == 1:
            # (the same PCG on every rank: one with the vector's product on some ranks
            # only does not return)
            eng = ss.SingleEngine(M, pde.solver.parameters)
            try:
                its, _ = eng.solve(x.hypre, rhs.hypre, 1e-5, 0.0, 200)
                own, done = COMM.allgather(eng._product is not None), bool(eng.converged)
            finally:
                eng.destroy()
        mixed.append((name, sum(kept), len(kept), off / max(size, 1e-300), dv, its,
                      all(kept) and off <= 1e-3 * size and dv <= 1e-6 and done and len(set(own)) == 1
                      and abs(its - left[1][1]) <= 2))
        M.destroy()
    S0.destroy()
    check("with values of another kind on the last rank every rank keeps the defect of its rows, zero where "
          "the values came in single precision, the matrix has its values, and hypre's PCG is the same on "
          "all ranks",
          all(g[6] for g in mixed),
          "(%s; %d iterations with the matrix alone)"
          % ("; ".join("%s: a vector on %d of %d ranks, off by %.1e of the largest defect, a product with the "
                       "values by %.1e, %d iterations" % g[:6] for g in mixed), left[1][1]))
    del S, plain, D, rd
    pde.invalidate_jacobian()
    pde._release_operators("solver", "solver_adj")

    # The element matrices of a batch that is not split, in single precision and where a
    # card's kernels leave them (HIPPYMFEM_PRECISION=mixed on a device; in host memory
    # arrays of JAX's, which take the device's route through the scatter).  Summed there in
    # their own precision, the sums had their rounding in them and the matrix kept no
    # defect, without a word: they are summed in double precision.
    from hippymfem.fem import csrassemble as csr
    from hippymfem.modeling.variables import STATE

    loc = pde._locals([u, m, None]) + pde._aux_locals()
    mats = [np.asarray(E, dtype=np.float32) for E in pde.kernel.element_matrices(ADJOINT, STATE, loc)]
    kw = dict(test_ess=pde.bc0.ess_tdof, diag_policy="one")
    old, td.SINGLE_ROW_SUMS = td.SINGLE_ROW_SUMS, True
    try:
        with td.single_target():
            M = csr.assemble_matrix_csr(Vu, Vu, pde.batches.groups, [_put(E) for E in mats], pde.nelem, **kw)
    finally:
        td.SINGLE_ROW_SUMS = old
    # (the same entries summed in double precision, as a matrix of MFEM's)
    D = csr.assemble_matrix_csr(Vu, Vu, pde.batches.groups, mats, pde.nelem, **kw)
    D.Mult(ones.hypre, y.hypre)
    kept = COMM.allgather(M.row_defect is not None)
    want = y.array - stored_row_sums(M)
    c = M.row_defect.values().astype(np.float64) if kept[COMM.rank] else np.full(want.size, np.inf)
    size = allreduce_extreme(COMM, float(np.max(np.abs(want))) if want.size else 0.0, MPI.MAX)
    off = allreduce_extreme(COMM, float(np.max(np.abs(c - want))) if want.size else 0.0, MPI.MAX)
    check("and keeps it where the element matrices of a whole batch come in single precision from a device",
          all(kept) and size > 0.0 and off <= 1e-4 * size,
          "(a vector on %d of %d ranks; against the sums of the stored rows %.1e of the largest defect)"
          % (sum(kept), len(kept), off / max(size, 1e-300)))
    M.destroy()
    del M, D

    # the refined solves with it: the state of double precision, in as many passes
    old, td.SINGLE_ROW_SUMS = td.SINGLE_ROW_SUMS, True
    try:
        us = pde.generate_state()
        pde.solveFwd(us, [us, m, None])
        passes = pde.fwd_iterations
    finally:
        td.SINGLE_ROW_SUMS = old
        pde.invalidate_jacobian()
        pde._release_operators("solver", "solver_adj")
    eu = us.copy().axpy(-1.0, got[False]["u"]).norm("l2") / got[False]["u"].norm("l2")
    check("with the defect of its row sums the Jacobian's refined solve gives the state of double precision",
          eu < 1e-9 and passes <= got[True]["passes"],
          "(%.1e; %d passes, %d without)" % (eu, passes, got[True]["passes"]))


def _pmg(check, COMM, pde, model, prior, pmesh, Vu, Vm, m, rhs, got):
    """A p-multigrid in place of BoomerAMG on the Jacobian (the solvers' parameter ``pmg``):
    its interpolation and its coarse matrix are what they should be, its solves take about
    the iterations of BoomerAMG, and the forward solve and Newton-CG give the state and the
    MAP point of double precision."""
    import warnings

    import jax.numpy as jnp

    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.fem import ptransfer
    from hippymfem.fem.elementbatch import get_batches

    if ss.library() is None or ss.library().no_pmg:
        # (a library without one of the p-multigrid's calls: the solves keep BoomerAMG, and
        # an engine made by name, as below, raises)
        check("a p-multigrid in place of BoomerAMG on the Jacobian", True,
              "(skipped: %s)" % (ss.library().no_pmg if ss.library() is not None else ss.why_not()))
        return
    H = ss.library().H
    solvers = [getattr(pde, a) for a in ("solver", "solver_fwd_inc", "solver_adj_inc")]
    varf = lambda u, m_, p, x: jnp.exp(m_.val) * hp.inner(u.grad, p.grad)      # noqa: E731

    def reset():
        pde.invalidate_jacobian()
        pde.release_linearization_point()
        pde._release_operators("solver", "solver_adj")

    reset()
    S, _ = pde._jacobian([got[True]["u"], m, None])
    amg = ss.SingleEngine(S, pde.solver.parameters)
    xa = Vu.vector()
    its_amg, _ = amg.solve(xa.hypre, rhs.hypre, 1e-5, 0.0, 200)
    amg.destroy()
    del amg, S
    for s in solvers:
        s.parameters["pmg"] = 1
    try:
        reset()
        S, _ = pde._jacobian([got[True]["u"], m, None])
        T = getattr(S, "transfer", None)
        eng = ss.PMGEngine(S, T, pde.solver.parameters)
        xp, y = Vu.vector(), Vu.vector()
        its_pmg, _ = eng.solve(xp.hypre, rhs.hypre, 1e-5, 0.0, 200)
        dx = xp.copy().axpy(-1.0, xa).norm("l2") / xa.norm("l2")
        S.Mult(xp.hypre, y.hypre)
        res = y.axpy(-1.0, rhs).norm("l2") / rhs.norm("l2")
        check("a p-multigrid solves the Jacobian's system in about the iterations of BoomerAMG",
              T is not None and T is pde._ptransfer and eng.converged and its_pmg <= its_amg + 4
              and dx < 1e-3 and res < 1e-3,
              "(%d iterations, BoomerAMG %d; the solutions differ by %.1e, the residual left %.1e)"
              % (its_pmg, its_amg, dx, res))
        # the residual after the coarse correction from the kept product A P, or computed
        # again with the matrix: the same iteration
        kept, ss.PMG_KEEP_AP = ss.PMG_KEEP_AP, not ss.PMG_KEEP_AP
        try:
            other = ss.PMGEngine(S, T, pde.solver.parameters)
            xo = Vu.vector()
            its_other, _ = other.solve(xo.hypre, rhs.hypre, 1e-5, 0.0, 200)
            both = (eng.ap is not None) != (other.ap is not None)
            other.destroy()
            del other
        finally:
            ss.PMG_KEEP_AP = kept
        do = xo.axpy(-1.0, xp).norm("l2") / xp.norm("l2")
        check("with the product A P kept or the residual computed again a p-multigrid makes the same iteration",
              both and abs(its_other - its_pmg) <= 1 and do < 1e-4,
              "(%d and %d iterations; the solutions differ by %.1e)" % (its_pmg, its_other, do))
        # the last levels of the linear elements' hierarchy gathered and finished by one
        # process (seq_threshold): with hypre on a device where the library can
        # (tools/hypre-3.2.0-seqdevice.patch), on the host by hypre's own code
        from hippymfem.common import mfemconfig as _mc

        if not ss.library().device or _mc.seq_on_device(H):
            pars = pde.solver.parameters
            was = pars["amg_options"]
            cases = (("seq_threshold=60", "the first rank"),
                     ("seq_threshold=60,redundant=1", "every rank"))
            if COMM.size == 1:
                # hypre drops the setting on one rank (the set-up's "if (num_procs == 1)
                # seq_threshold = 0"), so both hierarchies would be the default one, compared
                # with itself under names that speak of a gathered level
                cases = (("seq_threshold=60,redundant=1", None),)
            for extra, who in cases:
                pars["amg_options"] = (str(was) + "," if was else "") + extra
                try:
                    other = ss.PMGEngine(S, T, pars)
                    xg = Vu.vector()
                    its_seq, _ = other.solve(xg.hypre, rhs.hypre, 1e-5, 0.0, 200)
                    held = other.converged
                    other.destroy()
                    del other
                finally:
                    pars["amg_options"] = was
                dseq = xg.axpy(-1.0, xp).norm("l2") / xp.norm("l2")
                check(("with the last levels of its hierarchy gathered and solved by %s a "
                       "p-multigrid takes about the same iterations" % who) if who else
                      ("on one rank, where hypre drops seq_threshold, a p-multigrid with the setting takes "
                       "about the default one's iterations (the checks of a gathered level take two "
                       "ranks: skipped)"),
                      held and its_seq <= its_pmg + 2 and dseq < 1e-3,
                      "(%d iterations against %d; the solutions differ by %.1e)"
                      % (its_seq, its_pmg, dseq))
        # the two products from the element matrices of the Jacobian's pass (the matrix
        # came with them) against hypre's products of the assembled matrices
        given = getattr(S, "products", None)
        its_hypre, dh, dq = -1, float("nan"), float("nan")
        if given is not None:
            S.products = None
            try:
                other = ss.PMGEngine(S, T, pde.solver.parameters)
                xo = Vu.vector()
                its_hypre, _ = other.solve(xo.hypre, rhs.hypre, 1e-5, 0.0, 200)
                dh = xo.axpy(-1.0, xp).norm("l2") / xp.norm("l2")
                other.destroy()
                del other
            finally:
                S.products = given
            if given.get("Q") is not None:
                # A P applied to a vector of the linear elements, two ways (one that is
                # zero on the essential dofs, as a coarse correction is: the columns of
                # those are the identity's in hypre's product and empty in the other)
                from hippymfem.fem.bcs import zero_essential

                zc, pz, y1, y2 = Vm.vector(), Vu.vector(), Vu.vector(), Vu.vector()
                hp.parRandom.normal(1.0, zc)
                zero_essential(zc, np.array(T.coarse_ess.ToList(), dtype=np.int64))
                T.P.Mult(zc.hypre, pz.hypre)
                S.Mult(pz.hypre, y1.hypre)
                given["Q"].Mult(zc.hypre, y2.hypre)
                dq = y2.axpy(-1.0, y1).norm("l2") / y1.norm("l2")
        check("the products of a p-multigrid from the element matrices are those of the assembled ones",
              given is not None and eng._given is given and abs(its_hypre - its_pmg) <= 1 and dh < 1e-4
              and (given.get("Q") is None or dq < 1e-5),
              "(%s; %d iterations and %d with hypre's products, the solutions %.1e apart; A P on a vector %.1e)"
              % ("with the matrix" if given is not None else "none came with the matrix", its_pmg, its_hypre, dh, dq))
        # the Galerkin product against the Jacobian of the same form on the linear
        # elements, assembled by the element kernels with the same quadrature
        bc1 = hp.DirichletBC(Vm, lambda x: x[2], bdr_attributes=[1, 6])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = hp.PDEVariationalProblem([Vm, Vm, Vm], varf, bc1, bc1.homogeneous(), is_fwd_linear=True,
                                           symmetric_jacobian=True, quadrature_degree=pde.quadrature_degree)
            low.set_solvers(hp.auto_solver, Vm, COMM, max_direct=0, rel_tolerance=1e-12, max_iter=500)
            A1, _ = low._jacobian([low.generate_state(), m, None])
        z, y1, y2 = Vm.vector(), Vm.vector(), Vm.vector()
        hp.parRandom.normal(1.0, z)
        gal = two = float("nan")
        if type(A1).__name__ == "SingleParMatrix":
            a, b, c = A1.vector(columns=True), A1.vector(), A1.vector()
            a.set(z.hypre)
            H.hypre_ParCSRMatrixMatvec(1.0, A1.par, a.par, 0.0, b.par)
            H.hypre_ParCSRMatrixMatvec(1.0, eng.coarse, a.par, 0.0, c.par)
            b.get(y1.hypre)
            c.get(y2.hypre)
            gal = y2.axpy(-1.0, y1).norm("l2") / y1.norm("l2")
            # (and by the two products that an older hypre forms it with)
            lib = ss.library()
            other = lib.galerkin(S.par, T.P.par, products=True)
            two = float("nan")
            if COMM.allreduce(1 if other else 0, op=MPI.MIN):
                H.hypre_ParCSRMatrixMatvec(1.0, other, a.par, 0.0, c.par)
                c.get(y2.hypre)
                two = y2.axpy(-1.0, y1).norm("l2") / y1.norm("l2")
            if other:
                H.hypre_ParCSRMatrixDestroy(other)
            for v in (a, b, c):
                v.destroy()
        check("its interpolation's rows sum to one, and its coarse matrix is the Jacobian of the "
              "form on the linear elements",
              T.row_sum_error < 1e-12 and gal < 1e-5 and two < 1e-5,
              "(rows off by %.1e; a product with the two matrices differs by %.1e, and by %.1e with the "
              "coarse matrix from two products)" % (T.row_sum_error, gal, two))
        eng.destroy()
        del eng, A1, low, S
        # the problem's own solves
        reset()
        u = pde.generate_state()
        pde.solveFwd(u, [u, m, None])
        kind = type(pde.solver._pc).__name__
        eu = u.copy().axpy(-1.0, got[False]["u"]).norm("l2") / got[False]["u"].norm("l2")
        params = hp.ReducedSpaceNewtonCG_ParameterList()
        params["rel_tolerance"] = 1e-7
        params["max_iter"] = 40
        params["print_level"] = -1
        solver = hp.ReducedSpaceNewtonCG(model, params)
        x = solver.solve([None, prior.mean.copy(), None])
        em = x[PARAMETER].copy().axpy(-1.0, got[False]["m"]).norm("l2") / got[False]["m"].norm("l2")
        check("with pmg the forward solve and Newton-CG give the state and the MAP point of double precision",
              kind == "PMGEngine" and eu < 1e-9 and solver.converged and em < 1e-4
              and solver.it == got[False]["newton"]
              and abs(solver.total_cg_iter - got[False]["cg"]) <= max(3, got[False]["cg"] // 10),
              "(%s; state %.1e; %d Newton and %d CG iterations, %d and %d in double; MAP differs by %.1e)"
              % (kind, eu, solver.it, solver.total_cg_iter, got[False]["newton"], got[False]["cg"], em))
        del solver, x
        # a state space of linear elements has no interpolation: BoomerAMG stays, and says so
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            low = hp.PDEVariationalProblem([Vm, Vm, Vm], varf, bc1, bc1.homogeneous(), is_fwd_linear=True,
                                           symmetric_jacobian=True)
            low.set_solvers(hp.auto_solver, Vm, COMM, max_direct=0, rel_tolerance=1e-12, max_iter=500, pmg=1)
            u1 = low.generate_state()
            low.solveFwd(u1, [u1, m, None])
        check("a state space of linear elements keeps BoomerAMG, with a warning",
              bool(ptransfer.why_not(Vm)) and type(low.solver._pc).__name__ == "SingleEngine"
              and any("pmg" in str(w.message) for w in said),
              "(%s; %s)" % (type(low.solver._pc).__name__, ptransfer.why_not(Vm)))
        del low, u1
        # other elements and orders: a linear function of the coarse space is interpolated
        # to the same function of the fine space
        worst = {}
        meshes = [("hexahedra", pmesh, 3),
                  ("tetrahedra", mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.TETRAHEDRON)), 3),
                  ("quadrilaterals", mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(8, 8, mfem.Element.QUADRILATERAL)), 2),
                  ("triangles", mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(8, 8, mfem.Element.TRIANGLE)), 2)]
        for name, pm, dim in meshes:
            f = (lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2]) if dim == 3 else (
                lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1])
            Vc = Vm if pm is pmesh else hp.FunctionSpace.H1(pm, 1)
            for order in (2, 3):
                Vf = hp.FunctionSpace.H1(pm, order)
                T2 = ss.PTransfer(ptransfer.interpolation(Vf, Vc, get_batches(pm, 2 * order).groups))
                vc, vf, y = Vc.project(f), Vf.project(f), Vf.vector()
                T2.P.Mult(vc.hypre, y.hypre)
                worst["%s %d" % (name, order)] = largest([y.axpy(-1.0, vf).norm("linf"), T2.row_sum_error])
                T2.destroy()
                del T2, Vf
        check("the interpolation carries a linear function to the same function, on four kinds of "
              "elements and for orders two and three",
              largest(worst.values()) < 1e-5, "(%s)" % ", ".join("%s: %.1e" % kv for kv in worst.items()))
    finally:
        for s in solvers:
            s.parameters["pmg"] = 0
        reset()


def _pmg_meshes(check, COMM):
    """The geometric levels of a p-multigrid (the solvers' parameter ``pmg_h``): on a mesh
    refined from coarser ones, the correction from the linear elements by a V-cycle over
    those meshes solves the Jacobian's system in about the iterations of BoomerAMG on the
    linear elements, to the same solution; the interpolation between the linear elements
    of two meshes carries a linear function to the same function; without coarser meshes
    the linear elements keep BoomerAMG, with a warning."""
    import warnings

    import jax.numpy as jnp

    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.fem import ptransfer

    if ss.library() is None or ss.library().no_pmg:
        return
    closed = mfem.BasisType.ClosedUniform
    hexa = mfem.Element.HEXAHEDRON
    # a 24^3 mesh from 4^3, cut in three a side and then in two: two coarser meshes
    coarse = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, hexa))
    mid = mfem.ParMesh.MakeRefined(coarse, 3, closed)
    pmesh = mfem.ParMesh.MakeRefined(mid, 2, closed)
    ptransfer.set_coarser_meshes(pmesh, [mid, coarse])
    Vu, Vm = hp.FunctionSpace.H1(pmesh, 2), hp.FunctionSpace.H1(pmesh, 1)
    bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hp.PDEVariationalProblem([Vu, Vm, Vu],
                                   lambda u, m, p, x: jnp.exp(m.val) * hp.inner(u.grad, p.grad),
                                   bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hp.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12, max_iter=500)
    solvers = [getattr(pde, a) for a in ("solver", "solver_fwd_inc", "solver_adj_inc")]
    # a parameter that the coarse meshes do not resolve: the coefficient varies by e^4
    m = Vm.project(lambda x: 2.0 * jnp.sin(7.0 * x[0]) * jnp.cos(9.0 * x[1]) + jnp.sin(11.0 * x[2]))
    rhs = Vu.vector()
    hp.parRandom.set_seed(3)
    hp.parRandom.normal(1.0, rhs)
    pde.bc0.zero(rhs)
    u = pde.generate_state()
    for s in solvers:
        s.parameters["pmg"], s.parameters["pmg_h"] = 1, 2
    try:
        S, _ = pde._jacobian([u, m, None])
        T = getattr(S, "transfer", None)
        chain = list(getattr(T, "h", None) or ())
        rows = [int(t.P.global_cols) for t in chain]
        want = [int(hp.FunctionSpace.H1(q, 1).fes.GlobalTrueVSize()) for q in (mid, coarse)]
        check("a p-multigrid asked for geometric levels has the interpolations from the meshes that its "
              "mesh was refined from",
              T is not None and rows == want and largest(t.row_sum_error for t in chain) < 1e-12,
              "(%s rows of linear elements; the meshes have %s)" % (rows, want))
        if chain:
            eng = ss.PMGEngine(S, T, pde.solver.parameters)
            xg, y = Vu.vector(), Vu.vector()
            its_h, _ = eng.solve(xg.hypre, rhs.hypre, 1e-5, 0.0, 200)
            held = eng.converged and len(eng._h) == 2
            S.Mult(xg.hypre, y.hypre)
            res = y.axpy(-1.0, rhs).norm("l2") / rhs.norm("l2")
            eng.destroy()
            del eng
            # the same matrix with BoomerAMG on the linear elements
            T.h = []
            try:
                eng = ss.PMGEngine(S, T, pde.solver.parameters)
                xa = Vu.vector()
                its_amg, _ = eng.solve(xa.hypre, rhs.hypre, 1e-5, 0.0, 200)
                eng.destroy()
                del eng
            finally:
                T.h = chain
            dx = xg.copy().axpy(-1.0, xa).norm("l2") / xa.norm("l2")
            ess = xg.copy()
            pde.bc0.zero(ess)
            moved = xg.copy().axpy(-1.0, ess).norm("linf")
            check("over the coarser meshes a p-multigrid solves the Jacobian's system in about the iterations "
                  "that it takes with BoomerAMG on the linear elements, and leaves the essential dofs alone",
                  held and its_h <= its_amg + 3 and dx < 1e-3 and res < 1e-3 and moved == 0.0,
                  "(%d iterations, with BoomerAMG %d; the solutions differ by %.1e, the residual left %.1e, "
                  "the essential dofs moved by %.1e)" % (its_h, its_amg, dx, res, moved))
            # the same levels in host memory beside a device (pmg_h_host): from the mesh in
            # between on, and from the linear elements of the mesh itself on; a setting that
            # only the last matrix meets leaves it on the device (it has no exchange between
            # ranks to save).  Without a device the setting changes nothing.
            lib, size = ss.library(), COMM.Get_size()
            beside = bool(lib.device and lib.host_levels)
            got, said = [], []
            for name, rows_a_rank, first in (("from the mesh in between", want[0] // size + 1, 1),
                                             ("from the linear elements", 10 ** 9, 0),
                                             ("the last matrix alone", want[1] // size + 1, None)):
                pde.solver.parameters["pmg_h_host"] = rows_a_rank
                if not beside:
                    # (what an engine before this one leaves in the record beside a device)
                    ss.HOST_LEVELS["PMGEngine"] = {"level": -1, "rows": 0}
                try:
                    eng = ss.PMGEngine(S, T, pde.solver.parameters)
                    said.append((name, dict(ss.HOST_LEVELS.get("PMGEngine") or {}),
                                 {"level": first, "rows": ([int(T.P.global_cols)] + rows)[first]}
                                 if beside and first is not None else {}))
                    xh = Vu.vector()
                    its, _ = eng.solve(xh.hypre, rhs.hypre, 1e-5, 0.0, 200)
                    where, done = eng._host_at, eng.converged
                    eng.destroy()
                    del eng
                finally:
                    pde.solver.parameters["pmg_h_host"] = 0
                d = xh.copy().axpy(-1.0, xg).norm("l2") / xg.norm("l2")
                got.append((name, its, d, where,
                            bool(done) and where == (first if beside else None) and abs(its - its_h) <= 1 and d < 1e-3))
            check("with levels of the coarser meshes in host memory beside the device (pmg_h_host) a "
                  "p-multigrid takes the iterations and gives the solution that it does with all of them "
                  "on the device",
                  all(g[4] for g in got),
                  "(%s; all on the device: %d iterations%s)"
                  % ("; ".join("%s: %d iterations, the solution differs by %.1e, host from level %s"
                               % g[:4] for g in got), its_h, "" if beside else "; no device here"))
            check("the record of what a p-multigrid put in host memory (HOST_LEVELS) is that engine's: its first "
                  "level there with the rows of that matrix, and nothing for one that put nothing there, "
                  "whatever the engine before it left",
                  all(record == wanted for _, record, wanted in said),
                  "(%s%s)" % ("; ".join("%s: %s" % (name, record or "nothing") for name, record, _ in said),
                              "" if beside else "; no device here, an older record put there by hand"))
            # the last matrix in host memory and gathered (seq_threshold): by the engine itself,
            # one system on every rank, beside a device and without one
            options = pde.solver.parameters["amg_options"]
            pde.solver.parameters["amg_options"] = ",".join(
                [t for t in (str(options or ""), "seq_threshold=%d" % (want[1] + 1), "redundant=1") if t])
            pde.solver.parameters["pmg_h_host"] = want[0] // size + 1
            apart, odd, warned = None, None, 0
            try:
                eng = ss.PMGEngine(S, T, pde.solver.parameters)
                xq = Vu.vector()
                its_q, _ = eng.solve(xq.hypre, rhs.hypre, 1e-5, 0.0, 200)
                whole, done = eng._gathered is not None and eng.amg is None, eng.converged
                if whole:
                    apart = (eng._gathered.spread, dict(ss.GATHERED.get("PMGEngine") or {}))
                eng.destroy()
                del eng
                if whole and size > 1:
                    # the last rank's solver of the gathered matrix made another one (its
                    # strength threshold): the engine says so when it is set up
                    theta = pde.solver.parameters["amg_strength_threshold"]
                    if COMM.rank == size - 1:
                        pde.solver.parameters["amg_strength_threshold"] = 0.9
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            eng = ss.PMGEngine(S, T, pde.solver.parameters)
                        odd = eng._gathered.spread
                        warned = COMM.allreduce(sum("differ" in str(w.message) for w in caught), op=MPI.SUM)
                        eng.destroy()
                        del eng
                    finally:
                        pde.solver.parameters["amg_strength_threshold"] = theta
            finally:
                pde.solver.parameters["amg_options"] = options
                pde.solver.parameters["pmg_h_host"] = 0
            dq = xq.copy().axpy(-1.0, xg).norm("l2") / xg.norm("l2")
            check("a last matrix in host memory that the solver's seq_threshold gathers is gathered by the "
                  "p-multigrid itself and solved by every rank, to the same solution",
                  whole and bool(done) and abs(its_q - its_h) <= 2 and dq < 1e-3,
                  "(%d iterations, %d without the gathering; the solutions differ by %.1e; gathered by the "
                  "engine: %s)" % (its_q, its_h, dq, whole))
            check("the ranks' solvers of a gathered matrix are compared when they are set up: the same on "
                  "every rank, and a rank whose solver is another is said",
                  apart is not None and apart[0] == 0.0 and apart[1] == {"rows": want[1], "spread": 0.0}
                  and (size == 1 or (odd is not None and odd > ss.GATHERED_AGREE and warned == 1)),
                  "(solutions of one right-hand side apart by %s of their size; with another solver on the "
                  "last rank %s, %d warning(s))"
                  % ("%.1e" % apart[0] if apart else "?", "%.1e" % odd if odd is not None else "not tried: one rank",
                     warned))
            # the refined forward solve through the solver itself, against double precision
            pde.single_solves = True
            us = pde.generate_state()
            pde.solveFwd(us, [us, m, None])
            kind = type(getattr(pde.solver, "_pc", None)).__name__
            levels = len(getattr(getattr(pde.solver, "_pc", None), "_h", ()) or ())
            pde.set_single_solves(False)
            ud = pde.generate_state()
            pde.solveFwd(ud, [ud, m, None])
            pde.set_single_solves(True)
            du = us.copy().axpy(-1.0, ud).norm("l2") / ud.norm("l2")
            check("with pmg_h the forward solve gives the state of double precision",
                  kind == "PMGEngine" and levels == 2 and du < 1e-8,
                  "(%s with %d coarser meshes; the states differ by %.1e)" % (kind, levels, du))
            # and with the levels under the linear elements in host memory
            for s in solvers:
                s.parameters["pmg_h_host"] = want[0] // size + 1
            pde.invalidate_jacobian()
            pde._release_operators("solver", "solver_adj")
            uh = pde.generate_state()
            pde.solveFwd(uh, [uh, m, None])
            at = getattr(getattr(pde.solver, "_pc", None), "_host_at", None)
            dh = uh.copy().axpy(-1.0, ud).norm("l2") / ud.norm("l2")
            check("with pmg_h_host the forward solve gives the state of double precision",
                  at == (1 if beside else None) and dh < 1e-8,
                  "(host memory from level %s of the coarser meshes; the states differ by %.1e)" % (at, dh))
        del S
    finally:
        for s in solvers:
            s.parameters["pmg"], s.parameters["pmg_h"], s.parameters["pmg_h_host"] = 0, 0, 0
        pde.invalidate_jacobian()
        pde.release_linearization_point()
        pde._release_operators("solver", "solver_adj")
    # the interpolation alone: a linear function of the coarse mesh is the same function on
    # the fine one, for cuts in two, three and five a side, on hexahedra and quadrilaterals
    # (MFEM 4.9 aborts in ParMesh.MakeRefined of a mesh of tetrahedra on more than one rank)
    worst = {}
    f3 = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2]            # noqa: E731
    f2 = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1]                         # noqa: E731
    cases = [("hexahedra", mfem.Mesh.MakeCartesian3D(4, 4, 4, hexa), f3),
             ("quadrilaterals", mfem.Mesh.MakeCartesian2D(6, 6, mfem.Element.QUADRILATERAL), f2)]
    for name, serial, f in cases:
        below = mfem.ParMesh(COMM, serial)
        for cut in (2, 3, 5):
            above = mfem.ParMesh.MakeRefined(below, cut, closed)
            Vc, Vf = hp.FunctionSpace.H1(below, 1), hp.FunctionSpace.H1(above, 1)
            T2 = ss.PTransfer(ptransfer.refinement(Vf, Vc))
            vc, vf, y = Vc.project(f), Vf.project(f), Vf.vector()
            T2.P.Mult(vc.hypre, y.hypre)
            worst["%s by %d" % (name, cut)] = largest([y.axpy(-1.0, vf).norm("linf"), T2.row_sum_error])
            T2.destroy()
            del T2, Vc, Vf, above
    check("the interpolation between the linear elements of a mesh and of the mesh cut from it carries a "
          "linear function to the same function, for cuts in two, three and five a side",
          largest(worst.values()) < 1e-5, "(%s)" % ", ".join("%s: %.1e" % kv for kv in worst.items()))
    # a mesh that knows no coarser one: BoomerAMG on the linear elements, and a warning
    lone = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(6, 6, 6, hexa))
    Wu, Wm = hp.FunctionSpace.H1(lone, 2), hp.FunctionSpace.H1(lone, 1)
    bcl = hp.DirichletBC(Wu, lambda x: x[2], bdr_attributes=[1, 6])
    low = hp.PDEVariationalProblem([Wu, Wm, Wu],
                                   lambda u, m, p, x: jnp.exp(m.val) * hp.inner(u.grad, p.grad),
                                   bcl, bcl.homogeneous(), is_fwd_linear=True)
    low.set_solvers(hp.auto_solver, Wu, COMM, max_direct=0, rel_tolerance=1e-12, max_iter=500)
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        getattr(low, a).parameters["pmg"], getattr(low, a).parameters["pmg_h"] = 1, 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ul = low.generate_state()
        low.solveFwd(ul, [ul, Wm.vector(), None])
    said = COMM.allreduce(int(any("pmg_h" in str(w.message) for w in caught)), op=MPI.MAX)
    pc = getattr(low.solver, "_pc", None)
    check("on a mesh that knows no coarser mesh pmg_h keeps BoomerAMG on the linear elements, with a warning",
          type(pc).__name__ == "PMGEngine" and not getattr(pc, "_h", None) and said == 1,
          "(%s, %d coarser meshes, warned: %s)" % (type(pc).__name__, len(getattr(pc, "_h", ()) or ()), bool(said)))


def _prior_cycle(check, COMM):
    """The multigrid cycle of a prior's elliptic operator over the meshes that its mesh was
    refined from, in the single-precision library (``HCycle``, a prior's ``h_levels``): the
    copy of the operator is the operator, the levels are the meshes', a polynomial of five
    cycles is within its bound of the exact solve and symmetric to the rounding of single
    precision, the prior takes it for its Hessian preconditioner, with the levels in host
    memory as well, and keeps BoomerAMG, with a warning, on a mesh that knows no coarser one
    and on one that was last cut by three a side."""
    import warnings

    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.fem import ptransfer

    lib = ss.library()
    if lib is None or lib.no_pmg:
        return
    closed, hexa = mfem.BasisType.ClosedUniform, mfem.Element.HEXAHEDRON
    # a 24^3 mesh from 4^3, cut in three a side and then in two
    coarse = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, hexa))
    mid = mfem.ParMesh.MakeRefined(coarse, 3, closed)
    pmesh = mfem.ParMesh.MakeRefined(mid, 2, closed)
    ptransfer.set_coarser_meshes(pmesh, [mid, coarse])
    Vm = hp.FunctionSpace.H1(pmesh, 1)
    prior = hp.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    inner = prior.Rsolver.Asolver
    A = inner.A
    want = [int(hp.FunctionSpace.H1(q, 1).fes.GlobalTrueVSize()) for q in (mid, coarse)]
    u, v, x0, su, sv, w, w1 = (hp.ParVector(COMM, A.Height()) for _ in range(7))
    hp.parRandom.set_seed(43)
    hp.parRandom.normal(1.0, u)
    hp.parRandom.normal(1.0, v)
    # the copy of the operator in the single-precision library
    blocks = ss.matrix_blocks(A, COMM)
    S = ss.SingleParMatrix.from_pattern(blocks, blocks.values, lib)
    S.Mult(u.hypre, w.hypre)
    A.Mult(u.hypre, w1.hypre)
    copy = w.copy().axpy(-1.0, w1).norm("l2") / w1.norm("l2")
    check("the copy of a prior's operator in the single-precision library is the operator, to single precision",
          copy < 1e-5 and int(S.GetGlobalNumRows()) == int(A.GetGlobalNumRows()),
          "(a product with it differs by %.1e; %d rows)" % (copy, int(S.GetGlobalNumRows())))
    S.destroy()
    del S, blocks
    was = inner.parameters["rel_tolerance"]
    inner.parameters["rel_tolerance"] = 1e-13
    try:
        inner.solve(x0, u)
    finally:
        inner.parameters["rel_tolerance"] = was
    size = COMM.Get_size()
    beside = bool(lib.device and lib.host_levels)
    got, records = [], []
    # (the record of a Jacobian's p-multigrid, which a prior's cycle must leave as it is)
    kept = ss.HOST_LEVELS.get("PMGEngine")
    ss.HOST_LEVELS["PMGEngine"] = other = {"level": 1, "rows": 1}
    for name, host_rows in (("on the device", 0), ("its small levels in host memory", want[0] // size + 1)):
        prior.h_levels = 2
        inner.parameters["pmg_h_host"] = host_rows
        prior.__dict__.pop("_hessian_cheb", None)
        try:
            pc = prior.getHessianPreconditioner(cycles=5)
        finally:
            inner.parameters["pmg_h_host"] = 0
        geo, ch = getattr(pc, "geometric", None), pc.Asolver
        if geo is None:
            got.append((name, None, 1.0, 1.0, 1.0, 0.0, False))
            continue
        records.append((name, dict(ss.HOST_LEVELS.get("HCycle") or {}), ss.HOST_LEVELS.get("PMGEngine"),
                        {"level": geo._host_at, "rows": ([int(A.GetGlobalNumRows())] + geo.rows)[geo._host_at]}
                        if geo._host_at is not None else {}))
        ch.solve(su, u)
        ch.solve(sv, v)
        d = su.copy().axpy(-1.0, x0)
        A.Mult(d.hypre, w.hypre)
        num = max(w.inner(d), 0.0) ** 0.5
        A.Mult(x0.hypre, w.hypre)
        err = num / max(w.inner(x0), 1e-300) ** 0.5
        a, c = v.inner(su), u.inner(sv)
        sym = abs(a - c) / max(abs(a), abs(c), 1e-300)
        where = geo._host_at
        fine = (geo.rows == want and err <= 1.05 * ch.error_bound + 1e-5 and sym < 1e-4 and u.inner(su) > 0.0
                and where == ((1 if host_rows else None) if beside else None))
        y1, y2 = hp.ParVector(COMM, A.Height()), hp.ParVector(COMM, A.Height())
        pc.solve(y1, u)
        prior.Rsolver.solve(y2, u)
        off = y1.copy().axpy(-1.0, y2).norm("l2") / y2.norm("l2")
        again = prior.getHessianPreconditioner(cycles=5)
        got.append((name, geo.rows, err, ch.error_bound, sym, off, fine and again is pc and off < 0.2 and u.inner(y1) > 0.0))
    check("with h_levels a prior's Hessian preconditioner is a polynomial of cycles over the meshes that its "
          "mesh was refined from, in the single-precision library: within its bound of the exact solve, "
          "symmetric to single precision, and the same with the small levels in host memory",
          all(g[6] for g in got),
          "(%s)" % "; ".join("%s: %s rows, error %.1e in the energy norm (bound %.1e), <v, S u> and <u, S v> differ "
                             "by %.1e, against the precision's solver %.1e" % g[:6] for g in got))
    check("a prior's cycle records what it put in host memory under its own kind (HOST_LEVELS), and leaves the "
          "record of a Jacobian's p-multigrid as it was",
          len(records) == 2 and all(record == wanted and pmg is other for _, record, pmg, wanted in records),
          "(%s)" % "; ".join("%s: %s" % (name, record or "nothing") for name, record, _, _ in records))
    if kept is None:
        ss.HOST_LEVELS.pop("PMGEngine", None)
    else:
        ss.HOST_LEVELS["PMGEngine"] = kept
    prior.__dict__.pop("_hessian_cheb", None)
    # a mesh that knows no coarser one
    lone = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(6, 6, 6, hexa))
    Wm = hp.FunctionSpace.H1(lone, 1)
    other = hp.BiLaplacianPrior(Wm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    other.h_levels = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pl = other.getHessianPreconditioner(cycles=5)
    said = COMM.allreduce(int(any("PRIOR_H" in str(m.message) for m in caught)), op=MPI.MAX)
    check("on a mesh that knows no coarser mesh the prior's cycle stays BoomerAMG, with a warning",
          pl is not None and getattr(pl, "geometric", 1) is None and said == 1,
          "(a cycle over meshes: %s, warned: %s)" % (getattr(pl, "geometric", "?"), bool(said)))
    # and a mesh whose last cut was by three a side: the cycle over the meshes would relax on
    # the finest matrix by a polynomial of three steps, which costs more than BoomerAMG's
    base = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, hexa))
    third = mfem.ParMesh.MakeRefined(base, 3, closed)
    ptransfer.set_coarser_meshes(third, [base])
    by3 = hp.BiLaplacianPrior(hp.FunctionSpace.H1(third, 1), 0.1, 0.5, robin_bc=True, solver_type="krylov")
    by3.h_levels = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p3 = by3.getHessianPreconditioner(cycles=5)
    said = COMM.allreduce(int(any("last cut by 3" in str(m.message) for m in caught)), op=MPI.MAX)
    check("on a mesh that was last cut by three a side the prior's cycle stays BoomerAMG, with a warning",
          p3 is not None and getattr(p3, "geometric", 1) is None and said == 1,
          "(a cycle over meshes: %s, warned: %s)" % (getattr(p3, "geometric", "?"), bool(said)))


def _rank_without_coarse_rows(check, COMM):
    """A rank that owns no row of the last matrix of a p-multigrid's coarser meshes: the
    case that the levels in host memory and the gathering by the engine
    (``_HostGathered``) were written for, and one that MFEM's own partition of a small
    cube gives no rank on 2 or on 8 ranks.  The last rank is handed the one element in the
    middle of a 3^3 mesh, whose vertices all belong to lower ranks; on the meshes cut from
    it by three and then by two a side it owns the vertices inside that element.  Over
    these meshes a p-multigrid solves the Jacobian's system as it does with BoomerAMG on
    the linear elements, with the last matrix on the device, in host memory between the
    ranks, and gathered by the engine; the interpolation from the coarsest mesh, of whose
    columns that rank has none, and its transpose are what they should be; and a prior's
    cycle over the meshes is within its bound, with the small levels in host memory too."""
    import jax.numpy as jnp

    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.fem import ptransfer

    lib, size = ss.library(), COMM.Get_size()
    if lib is None or lib.no_pmg:
        return
    if not 2 <= size <= 27:
        check("a rank without rows of the last matrix of a p-multigrid's coarser meshes", True,
              "(skipped: for 2 to 27 ranks, not %d)" % size)
        return
    closed, hexa = mfem.BasisType.ClosedUniform, mfem.Element.HEXAHEDRON
    serial = mfem.Mesh.MakeCartesian3D(3, 3, 3, hexa)
    # MFEM gives a vertex that ranks share to the lowest of them: the element whose vertices
    # are all inside the cube goes to the last rank, and the 26 around it to the lower
    # ranks in the order of their numbers
    ne = serial.GetNE()
    inside = [all(0.0 < c < 1.0 for v in serial.GetElementVertices(e) for c in serial.GetVertexArray(v))
              for e in range(ne)]
    around = [e for e in range(ne) if not inside[e]]
    part = [size - 1] * ne
    for k, e in enumerate(around):
        part[e] = k * (size - 1) // len(around)
    part = mfem.intArray(part)
    coarse = mfem.ParMesh(COMM, serial, part.GetData())
    mid = mfem.ParMesh.MakeRefined(coarse, 3, closed)
    pmesh = mfem.ParMesh.MakeRefined(mid, 2, closed)
    ptransfer.set_coarser_meshes(pmesh, [mid, coarse])
    Vu, Vm = hp.FunctionSpace.H1(pmesh, 2), hp.FunctionSpace.H1(pmesh, 1)
    Vmid, Vc = hp.FunctionSpace.H1(mid, 1), hp.FunctionSpace.H1(coarse, 1)
    owned = COMM.allgather(int(Vc.fes.GetTrueVSize()))
    want = [int(Vmid.fes.GlobalTrueVSize()), int(Vc.fes.GlobalTrueVSize())]
    bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hp.PDEVariationalProblem([Vu, Vm, Vu],
                                   lambda u, m, p, x: jnp.exp(m.val) * hp.inner(u.grad, p.grad),
                                   bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hp.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12, max_iter=500)
    solvers = [getattr(pde, a) for a in ("solver", "solver_fwd_inc", "solver_adj_inc")]
    # a parameter that the coarse meshes do not resolve: the coefficient varies by e^4
    m = Vm.project(lambda x: 2.0 * np.sin(7.0 * x[0]) * np.cos(9.0 * x[1]) + np.sin(11.0 * x[2]))
    rhs = Vu.vector()
    hp.parRandom.set_seed(47)
    hp.parRandom.normal(1.0, rhs)
    pde.bc0.zero(rhs)
    pars = pde.solver.parameters
    # (pmg_h_host: with this many rows a rank the mesh in between and the last matrix under
    # it live in host memory beside a device; without a device the setting changes nothing)
    small, beside = max(want[0] // size + 1, want[1]), bool(lib.device and lib.host_levels)

    def solve(host_rows=0, extra=""):
        """An engine of the Jacobian with these settings and its solve: the iterations, the
        solution, whether it converged, the level from which host memory holds it, its
        coarser meshes, and the rows a rank has of a last matrix that the engine gathered."""
        options = pars["amg_options"]
        pars["amg_options"] = ",".join(t for t in (str(options or ""), extra) if t)
        pars["pmg_h_host"] = host_rows
        try:
            eng = ss.PMGEngine(S, T, pars)
            x = Vu.vector()
            its, _ = eng.solve(x.hypre, rhs.hypre, 1e-5, 0.0, 200)
            held = None if eng._gathered is None else COMM.allgather(int(eng._gathered.n))
            out = (its, x, bool(eng.converged), eng._host_at, len(eng._h), held)
            eng.destroy()
        finally:
            pars["amg_options"], pars["pmg_h_host"] = options, 0
        return out

    for s in solvers:
        s.parameters["pmg"], s.parameters["pmg_h"] = 1, 2
    try:
        S, _ = pde._jacobian([pde.generate_state(), m, None])
        T = getattr(S, "transfer", None)
        chain = list(getattr(T, "h", None) or ())
        rows = [int(t.P.global_cols) for t in chain]
        check("the last rank owns no vertex of the coarsest mesh, and a p-multigrid has the interpolations "
              "from it and from the mesh in between",
              owned[-1] == 0 and rows == want,
              "(%s vertices a rank; %s rows of linear elements, the meshes have %s)" % (owned, rows, want))
        if rows == want:
            its_h, xg, done, _at, levels, _held = solve()
            # the same matrix with BoomerAMG on the linear elements
            T.h = []
            try:
                its_amg, xa = solve()[:2]
            finally:
                T.h = chain
            y = Vu.vector()
            S.Mult(xg.hypre, y.hypre)
            res = y.axpy(-1.0, rhs).norm("l2") / rhs.norm("l2")
            dx = xg.copy().axpy(-1.0, xa).norm("l2") / xa.norm("l2")
            check("where a rank owns no row of the last matrix of the coarser meshes a p-multigrid solves the "
                  "Jacobian's system in about the iterations that it takes with BoomerAMG on the linear elements",
                  done and levels == 2 and its_h <= its_amg + 3 and dx < 1e-3 and res < 1e-3,
                  "(%d iterations, with BoomerAMG %d; the solutions differ by %.1e, the residual left %.1e)"
                  % (its_h, its_amg, dx, res))
            # that matrix in host memory beside a device, with BoomerAMG between the ranks
            # and gathered by the engine (seq_threshold), where the last rank gathers nothing
            got = []
            for name, extra in (("between the ranks", ""),
                                ("gathered", "seq_threshold=%d,redundant=1" % (want[1] + 1))):
                its, x, done, where, _levels, held = solve(small, extra)
                d = x.axpy(-1.0, xg).norm("l2") / xg.norm("l2")
                whole = held is not None and held[-1] == 0 and sum(held) == want[1]
                got.append((name, its, d, where, held,
                            done and where == (1 if beside else None) and abs(its - its_h) <= 2 and d < 1e-3
                            and (whole if extra else held is None)))
            check("with that matrix in host memory, between the ranks and gathered by the engine, the rank "
                  "without rows of it changes neither the iterations nor the solution",
                  all(g[5] for g in got),
                  "(%s; all on the device: %d iterations%s)"
                  % ("; ".join("%s: %d iterations, the solution differs by %.1e, host from level %s, rows a rank "
                               "of a gathered matrix %s" % g[:5] for g in got), its_h,
                     "" if beside else "; no device here"))
            # the interpolation from the coarsest mesh and its transpose, on vectors of
            # which the last rank has no entry
            f = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2]            # noqa: E731
            T2 = ss.PTransfer(ptransfer.refinement(Vmid, Vc))
            vc, vf, y, w, z = Vc.project(f), Vmid.project(f), Vmid.vector(), Vmid.vector(), Vc.vector()
            hp.parRandom.normal(1.0, w)
            T2.P.Mult(vc.hypre, y.hypre)
            T2.P.MultTranspose(w.hypre, z.hypre)
            adj = abs(y.inner(w) - vc.inner(z)) / max(y.norm("l2") * w.norm("l2"), 1e-300)
            off = largest([y.axpy(-1.0, vf).norm("linf"), T2.row_sum_error])
            T2.destroy()
            del T2
            check("the interpolation from a mesh of which a rank owns no vertex carries a linear function to "
                  "the same function, and its transposed product is the transpose of the plain one",
                  off < 1e-5 and adj < 1e-5,
                  "(off by %.1e; <P c, w> and <c, P^T w> differ by %.1e of their sizes)" % (off, adj))
            # a prior's cycle over the same meshes (HCycle)
            prior = hp.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
            inner = prior.Rsolver.Asolver
            A = inner.A
            u, x0, su, w1 = (hp.ParVector(COMM, A.Height()) for _ in range(4))
            hp.parRandom.normal(1.0, u)
            was = inner.parameters["rel_tolerance"]
            inner.parameters["rel_tolerance"] = 1e-13
            try:
                inner.solve(x0, u)
            finally:
                inner.parameters["rel_tolerance"] = was
            A.Mult(x0.hypre, w1.hypre)
            energy = max(w1.inner(x0), 1e-300) ** 0.5
            got = []
            for name, host_rows in (("on the device", 0), ("its small levels in host memory", small)):
                prior.h_levels = 2
                inner.parameters["pmg_h_host"] = host_rows
                prior.__dict__.pop("_hessian_cheb", None)
                try:
                    pc = prior.getHessianPreconditioner(cycles=5)
                finally:
                    inner.parameters["pmg_h_host"] = 0
                geo, ch = getattr(pc, "geometric", None), pc.Asolver
                if geo is None:
                    got.append((name, None, 1.0, 1.0, False))
                    continue
                ch.solve(su, u)
                d = su.copy().axpy(-1.0, x0)
                A.Mult(d.hypre, w1.hypre)
                err = max(w1.inner(d), 0.0) ** 0.5 / energy
                got.append((name, geo.rows, err, ch.error_bound,
                            geo.rows == want and err <= 1.05 * ch.error_bound + 1e-5
                            and geo._host_at == ((1 if host_rows else None) if beside else None)))
            prior.__dict__.pop("_hessian_cheb", None)
            check("a prior's cycle over these meshes is within its bound of the exact solve, with the small "
                  "levels in host memory as well",
                  all(g[4] for g in got),
                  "(%s)" % "; ".join("%s: %s rows, error %.1e in the energy norm (bound %.1e)" % g[:4] for g in got))
        del S
    finally:
        for s in solvers:
            s.parameters["pmg"], s.parameters["pmg_h"], s.parameters["pmg_h_host"] = 0, 0, 0
        pde.invalidate_jacobian()
        pde.release_linearization_point()
        pde._release_operators("solver", "solver_adj")


def _single_blocks(check, COMM):
    """The blocks of a linearization point as matrices of the single-precision library
    (``tdofassemble.SINGLE_BLOCKS``): the blocks whose elimination folds into the scatter
    are made there and the others stay MFEM's, every product and every transposed product
    is the double-precision one to single precision, a transposed product is the
    transpose of the plain one, and a second point's blocks are that point's (with a
    device, the two blocks that are applied transposed keep their transposes, which must
    not outlive their matrices)."""
    import jax.numpy as jnp

    from hippymfem.algorithms import singlesolve as ss
    from hippymfem.fem import tdofassemble as td
    from hippymfem.modeling.variables import STATE

    if ss.library() is None:
        return
    pmesh = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(6, 6, 6, mfem.Element.HEXAHEDRON))
    Vu, Vm = hp.FunctionSpace.H1(pmesh, 2), hp.FunctionSpace.H1(pmesh, 1)
    bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    # (not linear in the state, so that the state-state block exists)
    pde = hp.PDEVariationalProblem(
        [Vu, Vm, Vu], lambda u, m, p, x: jnp.exp(m.val) * (1.0 + u.val * u.val) * hp.inner(u.grad, p.grad),
        bc, bc.homogeneous(), is_fwd_linear=False)
    pde.set_solvers(hp.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-10, max_iter=500)
    du, dm = Vu.vector(), Vm.vector()
    hp.parRandom.set_seed(53)
    hp.parRandom.normal(1.0, du)
    pde.bc0.zero(du)
    hp.parRandom.normal(1.0, dm)
    pairs = [(ADJOINT, PARAMETER), (PARAMETER, ADJOINT), (STATE, STATE), (STATE, PARAMETER),
             (PARAMETER, STATE), (PARAMETER, PARAMETER)]

    def point(seed):
        x = [Vu.vector(), Vm.vector(), Vu.vector()]
        hp.parRandom.set_seed(seed)
        hp.parRandom.normal(0.3, x[STATE])
        hp.parRandom.normal(0.3, x[PARAMETER])
        hp.parRandom.normal(1.0, x[ADJOINT])
        pde.bc0.zero(x[ADJOINT])
        return x

    def products(x, single):
        was, td.SINGLE_BLOCKS = td.SINGLE_BLOCKS, bool(single)
        try:
            pde.setLinearizationPoint(x, gauss_newton_approx=False)
        finally:
            td.SINGLE_BLOCKS = was
        kinds = [type(b).__name__ for b in (pde.C, pde.Wuu, pde.Wum, pde.Wmm)]
        out = {}
        for _ in range(2):                      # (the second transposed product finds the transpose kept)
            for i, j in pairs:
                y = (Vm if i == PARAMETER else Vu).vector()
                pde.apply_ij(i, j, du if j in (STATE, ADJOINT) else dm, y)
                out[(i, j)] = y
        return kinds, out

    worst, moved, kinds_single, adjoint = 0.0, [], None, 0.0
    first = None
    for seed in (57, 59):
        x = point(seed)
        kinds_single, s = products(x, True)
        kinds_double, d = products(x, False)
        worst = largest([worst] + [s[k].copy().axpy(-1.0, d[k]).norm("l2") / max(d[k].norm("l2"), 1e-300)
                                   for k in pairs])
        # <dm, C^T du> against <C dm, du>, both from the single-precision block
        a, b = dm.inner(s[(PARAMETER, ADJOINT)]), s[(ADJOINT, PARAMETER)].inner(du)
        adjoint = largest([adjoint, abs(a - b) / max(abs(a), abs(b), 1e-300)])
        if first is None:
            first = s
        else:
            moved = [first[k].copy().axpy(-1.0, s[k]).norm("l2") / max(s[k].norm("l2"), 1e-300)
                     for k in ((PARAMETER, ADJOINT), (PARAMETER, STATE))]
    pde.release_linearization_point()
    check("with SINGLE_BLOCKS the blocks of a linearization point that fold their elimination are "
          "matrices of the single-precision library, and their products and transposed products are "
          "the double-precision ones to single precision, at a first point and at a second",
          kinds_single[:3] == ["SingleParMatrix"] * 3 and "SingleParMatrix" not in kinds_double
          and worst < 5e-6 and adjoint < 5e-6 and all(v > 1e-3 for v in moved),
          "(C, W_uu, W_um, W_mm: %s; worst relative difference %.1e; <dm, C^T du> and <C dm, du> "
          "differ by %.1e; the transposed products of the two points differ by %.1e and %.1e)"
          % (", ".join(kinds_single), worst, adjoint, moved[0], moved[1]))

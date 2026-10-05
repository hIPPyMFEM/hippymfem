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
        # (a HIP build: the library is refused before it is loaded, with a warning)
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
        d = max(ys[0].copy().axpy(-1.0, ys[1]).norm("l2"), ys[0].copy().axpy(-1.0, y3).norm("l2"))
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
        old_chunk, km.ELEMENT_CHUNK = km.ELEMENT_CHUNK, 128
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
          max(abs(k - its[0]) for k in its[1:]) <= 1 and max(off) < 1e-3 and none == 0
          and ks.converged and y.norm("l2") == 0.0,
          "(%d iterations, and %s for the sizes 2^%s; the solutions differ by at most %.1e; "
          "zero: %d iterations)" % (its[0], its[1:], list(powers), max(off), none))
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
    finally:
        singlesolve.AMG_OPTIONS = options
        pde.invalidate_jacobian()
        pde._release_operators("solver", "solver_adj")
    check("a solve that hypre's PCG abandons is not taken for converged, and the forward solve "
          "falls back on double precision",
          not accepted and "not positive definite" in said and eu < 1e-9,
          "(the state against double precision %.1e; %s)" % (eu, said[said.find("failed"):] or "no error"))

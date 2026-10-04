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
    check("the single-precision library loads", singlesolve.library() is not None,
          "(%s)" % (singlesolve.why_not() or singlesolve.HYPRE_SINGLE))
    if singlesolve.library() is None:
        return
    import jax.numpy as jnp

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
    pde.single_solves = True

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

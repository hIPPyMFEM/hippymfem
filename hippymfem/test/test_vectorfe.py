# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""H(curl), H(div) and vector H1 spaces in the AD route.

Nedelec and Raviart-Thomas elements need two things scalar ones do not: their basis
functions are vector valued and Piola mapped, so they cannot be tabulated once for a
whole element group, and MFEM encodes their orientation partly as a dense
``DofTransformation`` rather than a sign.  Both are handled in
:mod:`hippymfem.fem.vectorfe` and checked by exact comparison against MFEM's own
integrators:

================================  =====================================
``dot(u.val, p.val)``             ``VectorFEMassIntegrator``
``u.curl * p.curl`` (2D)          ``CurlCurlIntegrator``
``dot(u.curl, p.curl)`` (3D)      ``CurlCurlIntegrator``
``u.div * p.div``                 ``DivDivIntegrator``
================================  =====================================

on triangles, quadrilaterals, tetrahedra and hexahedra, at orders that do and do
not require a ``DofTransformation`` (tetrahedral H(curl) of order 2 requires one;
everything else here does not, and the test reports which case is which so the
covered case cannot quietly disappear).

A full inverse problem is then solved in H(curl), a curl-curl operator with an
unknown coefficient, so that the gradient and Hessian built from these blocks are
checked by finite differences and not just the individual matrices.

Run with ``mpirun -n N python -m hippymfem.test.test_vectorfe``.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hp
from hippymfem.common.linalg import to_dense
from hippymfem.fem.assemble import assemble_matrix
from hippymfem.fem.elementbatch import MeshBatches
from hippymfem.fem.kernel import QuadratureKernel
from hippymfem.fem.spaces import FunctionSpace
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


def mk(kind, n):
    if kind == "tri":
        return mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE)
    if kind == "quad":
        return mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.QUADRILATERAL)
    if kind == "tet":
        return mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.TETRAHEDRON)
    return mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON)


def compare(kind, family, order, n=3):
    """One (geometry, family, order): mass and curl/div against MFEM."""
    pm = mfem.ParMesh(COMM, mk(kind, n))
    dim = pm.Dimension()
    V = (FunctionSpace.ND(pm, order) if family == "ND"
         else FunctionSpace.RT(pm, order))
    Vm = FunctionSpace.H1(pm, 1)
    b = MeshBatches(pm, 2 * order + 4, COMM)
    tab = b.tables(V)[0]
    loc = [V.local_values(V.vector()), Vm.local_values(Vm.vector()),
           V.local_values(V.vector())]

    def against(density, integrator, label):
        K = QuadratureKernel(density, [V, Vm, V], b)
        A = assemble_matrix(V, V, b.groups,
                            K.element_matrices(ADJOINT, STATE, loc), pm.GetNE())
        form = mfem.ParBilinearForm(V.fes)
        # the same integration rule on both sides, so the comparison measures the
        # element kernel and not a difference between two quadrature choices
        integrator.SetIntRule(b.groups[0].ir)
        form.AddDomainIntegrator(integrator)
        form.Assemble()
        form.Finalize()
        R = mfem.HypreParMatrix()
        form.FormSystemMatrix(mfem.intArray(), R)
        D1, D2 = to_dense(A, COMM), to_dense(R, COMM)
        return label, float(np.abs(D1 - D2).max()) / max(
            float(np.abs(D2).max()), 1e-300)

    errs = [against(lambda u, m, p, x: jnp.dot(u.val, p.val),
                    mfem.VectorFEMassIntegrator(), "mass")]
    if family == "ND":
        dens = ((lambda u, m, p, x: u.curl * p.curl) if dim == 2
                else (lambda u, m, p, x: jnp.dot(u.curl, p.curl)))
        errs.append(against(dens, mfem.CurlCurlIntegrator(), "curl-curl"))
    else:
        errs.append(against(lambda u, m, p, x: u.div * p.div,
                            mfem.DivDivIntegrator(), "div-div"))
    worst = max(e for _, e in errs)
    check("%-4s %s order %d (nd=%d, DofTransformation=%s)"
          % (kind, family, order, tab.nd, "yes" if tab.T is not None else "no"),
          worst < 1e-13,
          "(" + ", ".join("%s %.2e" % (k, v) for k, v in errs) + ")")
    if tab.T is not None:
        # MFEM folds the face DofTransformation of a shared face into P on the rank
        # that does not own it, so P is not boolean there and both the true-dof route
        # and the folded elimination must stand down.  (On one rank P is the identity.)
        from hippymfem.fem import csrassemble as csr

        check("a DofTransformation space is not taken for a boolean P",
              COMM.size == 1 or not csr._boolean_prolongation(V))
    return tab.T is not None


def test_vs_mfem():
    if RANK == 0:
        print("H(curl) / H(div) element matrices vs MFEM")
    saw_doftrans = False
    for kind in ("tri", "quad"):
        for order in (1, 2):
            saw_doftrans |= compare(kind, "ND", order)
        for order in (0, 1):
            saw_doftrans |= compare(kind, "RT", order)
    for kind in ("tet", "hex"):
        for order in (1, 2):
            saw_doftrans |= compare(kind, "ND", order)
        for order in (0, 1):
            saw_doftrans |= compare(kind, "RT", order)
    check("at least one case exercised a non-identity DofTransformation",
          saw_doftrans,
          "(tetrahedral H(curl) of order 2; without it that path is untested)")


def test_coefficient_and_derivatives():
    """A varying coefficient, and the derivative blocks it generates.

    ``C = d/dm (dR/dp)`` couples an H(curl) test space to an H1 parameter space, so
    this exercises a rectangular block between two different element families.
    """
    if RANK == 0:
        print("varying coefficient and mixed-family blocks")
    pm = mfem.ParMesh(COMM, mk("tri", 4))
    V = FunctionSpace.ND(pm, 1)
    Vm = FunctionSpace.H1(pm, 1)
    b = MeshBatches(pm, 6, COMM)

    # Two coefficients: one of position, which MFEM can reproduce exactly, and one of
    # the parameter, whose derivative is checked by finite differences below.  PyMFEM
    # cannot hand a Python exp() to TransformedCoefficient, so only the positional one
    # is compared against MFEM.
    def varf_x(u, m, p, x):
        return (1.0 + x[0] * x[1]) * u.curl * p.curl + jnp.dot(u.val, p.val)

    def varf(u, m, p, x):
        return jnp.exp(m.val) * u.curl * p.curl + jnp.dot(u.val, p.val)

    Kx = QuadratureKernel(varf_x, [V, Vm, V], b)
    K = QuadratureKernel(varf, [V, Vm, V], b)
    hp.parRandom.set_seed(3)
    mv = Vm.vector()
    hp.parRandom.normal(0.5, mv)
    uv = V.vector()
    hp.parRandom.normal(1.0, uv)
    pv = V.vector()
    hp.parRandom.normal(1.0, pv)
    loc = [V.local_values(uv), Vm.local_values(mv), V.local_values(pv)]

    # a position-dependent coefficient against CurlCurlIntegrator
    class _C(mfem.PyCoefficient):
        def EvalValue(self, x):
            return 1.0 + float(x[0]) * float(x[1])

    coeff = _C()
    A = assemble_matrix(V, V, b.groups, Kx.element_matrices(ADJOINT, STATE, loc),
                        pm.GetNE())
    form = mfem.ParBilinearForm(V.fes)
    cc = mfem.CurlCurlIntegrator(coeff)
    vm = mfem.VectorFEMassIntegrator()
    for it in (cc, vm):
        it.SetIntRule(b.groups[0].ir)
    form.AddDomainIntegrator(cc)
    form.AddDomainIntegrator(vm)
    form.Assemble()
    form.Finalize()
    R = mfem.HypreParMatrix()
    form.FormSystemMatrix(mfem.intArray(), R)
    DR = to_dense(R, COMM)
    e = float(np.abs(to_dense(A, COMM) - DR).max()) / max(
        float(np.abs(DR).max()), 1e-300)
    check("varying-coefficient curl-curl matches CurlCurlIntegrator", e < 1e-12,
          "(rel %.3e)" % e)

    # C is rectangular, H(curl) rows by H1 columns, and must match a finite
    # difference of the residual in m
    C = assemble_matrix(V, Vm, b.groups,
                        K.element_matrices(ADJOINT, PARAMETER, loc), pm.GetNE())
    check("C has H(curl) rows and H1 columns",
          C.GetGlobalNumRows() == V.GlobalTrueVSize()
          and C.GetGlobalNumCols() == Vm.GlobalTrueVSize(),
          "(%d x %d)" % (C.GetGlobalNumRows(), C.GetGlobalNumCols()))

    dm = Vm.vector()
    hp.parRandom.normal(1.0, dm)
    Cdm = V.vector()
    C.Mult(dm.hypre, Cdm.hypre)
    eps = 1e-7
    from hippymfem.fem.assemble import assemble_vector

    def resid(mvec):
        l = [V.local_values(uv), Vm.local_values(mvec), V.local_values(pv)]
        return assemble_vector(V, b.groups, K.element_vectors(ADJOINT, l),
                               pm.GetNE())

    rp = resid(mv.copy().axpy(eps, dm))
    rm = resid(mv.copy().axpy(-eps, dm))
    fd = rp.copy().axpy(-1.0, rm).scale(0.5 / eps)
    fd.axpy(-1.0, Cdm)
    rel = fd.norm("l2") / max(Cdm.norm("l2"), 1e-300)
    check("C matches a central difference of the residual in m", rel < 1e-6,
          "(rel %.3e)" % rel)


def test_hcurl_inverse_problem():
    """A full inverse problem with an H(curl) state.

    ``curl(exp(m) curl u) + u = f`` with a distributed L2 misfit.  The finite
    difference slopes are the test that the gradient and the Hessian built from the
    vector-element blocks are consistent.
    """
    if RANK == 0:
        print("inverse problem with an H(curl) state")
    pm = mfem.ParMesh(COMM, mk("tri", 8))
    V = FunctionSpace.ND(pm, 1)
    Vm = FunctionSpace.H1(pm, 1)

    def varf(u, m, p, x):
        src = jnp.array([jnp.sin(3.0 * x[1]), jnp.cos(2.0 * x[0])])
        return (jnp.exp(m.val) * u.curl * p.curl + jnp.dot(u.val, p.val)
                - jnp.dot(src, p.val))

    pde = hp.PDEVariationalProblem([V, Vm, V], varf, None, None,
                                   is_fwd_linear=True)
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if NP == 1:
            setattr(pde, a, hp.LUSolver(COMM))
        else:
            s = hp.KrylovSolver(COMM, "gmres", "amg")
            s.parameters["rel_tolerance"] = 1e-13
            s.parameters["max_iter"] = 3000
            setattr(pde, a, s)

    prior = hp.BiLaplacianPrior(Vm, 0.3, 1.0, robin_bc=True,
                                solver_type="lu" if NP == 1 else "krylov")
    mtrue = Vm.project(lambda z: 0.5 * np.sin(2.0 * z[0]) * np.cos(2.0 * z[1]))
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    check("H(curl) forward solve produced a nonzero state",
          utrue.norm("l2") > 1e-8 and np.isfinite(utrue.norm("l2")),
          "(||u|| = %.6f, %d tdofs)" % (utrue.norm("l2"), V.GlobalTrueVSize()))

    misfit = hp.ContinuousStateObservation(V, noise_variance=1.0)
    d = V.vector()
    d.assign(utrue)
    hp.parRandom.set_seed(2)
    noise = V.vector()
    hp.parRandom.normal(0.01 * max(utrue.norm("linf"), 1e-30), noise)
    d.axpy(1.0, noise)
    misfit.d = d
    model = hp.Model(pde, prior, misfit)

    m0 = prior.mean.copy()
    hp.parRandom.normal(0.2, m0)
    eps, err_grad, err_H = hp.modelVerify(
        model, m0, is_quadratic=False, verbose=False, eps=np.logspace(0, -6, 13))
    # The Hessian error reaches its round-off floor (about 4e-10) by eps = 3e-4, much
    # earlier than the gradient's: the misfit is nearly quadratic, so the first-order
    # term being measured is tiny.  The default window straddles that floor and gives
    # a negative slope, so each window is chosen from its measured error curve.
    gs = hp.best_slope(eps, err_grad, lo=1e-6, hi=1.0)
    hs = hp.best_slope(eps, err_H, lo=1e-3, hi=1.0)
    check("H(curl) gradient has first-order FD slope", abs(gs - 1.0) < 0.1,
          "(slope %.4f over eps in [1e-6, 1])" % gs)
    check("H(curl) Hessian has first-order FD slope", abs(hs - 1.0) < 0.1,
          "(slope %.4f over eps in [1e-3, 1]; floor %.1e at eps=%.0e)"
          % (hs, err_H.min(), eps[int(np.argmin(err_H))]))

    # and it actually inverts
    p = hp.ReducedSpaceNewtonCG_ParameterList()
    p["rel_tolerance"] = 1e-8
    p["max_iter"] = 25
    p["print_level"] = -1
    s = hp.ReducedSpaceNewtonCG(model, p)
    x = s.solve([None, prior.mean.copy(), None])
    check("Newton-CG converges on the H(curl) problem", s.converged,
          "(%d its, ||g||/||g0|| = %.2e)"
          % (s.it, s.final_grad_norm / max(s.initial_grad_norm, 1e-300)))


def test_vector_h1():
    """A vdim > 1 Lagrange space: linear elasticity against MFEM's integrator.

    Checks the vector field layout (``Field.val`` is ``(vdim,)`` and ``Field.grad``
    is ``(vdim, sdim)``) against ``ElasticityIntegrator``.
    """
    if RANK == 0:
        print("vector H1 (elasticity)")
    pm = mfem.ParMesh(COMM, mk("quad", 4))
    V = FunctionSpace.H1(pm, 2, vdim=2)
    Vm = FunctionSpace.H1(pm, 1)
    b = MeshBatches(pm, 6, COMM)
    lam, mu = 1.3, 0.7

    def varf(u, m, p, x):
        eu = 0.5 * (u.grad + u.grad.T)
        ep = 0.5 * (p.grad + p.grad.T)
        return (2.0 * mu * jnp.sum(eu * ep)
                + lam * jnp.trace(u.grad) * jnp.trace(p.grad))

    K = QuadratureKernel(varf, [V, Vm, V], b)
    loc = [V.local_values(V.vector()), Vm.local_values(Vm.vector()),
           V.local_values(V.vector())]
    A = assemble_matrix(V, V, b.groups, K.element_matrices(ADJOINT, STATE, loc),
                        pm.GetNE())
    form = mfem.ParBilinearForm(V.fes)
    lc = mfem.ConstantCoefficient(lam)
    mc = mfem.ConstantCoefficient(mu)
    form.AddDomainIntegrator(mfem.ElasticityIntegrator(lc, mc))
    form.Assemble()
    form.Finalize()
    R = mfem.HypreParMatrix()
    form.FormSystemMatrix(mfem.intArray(), R)
    D2 = to_dense(R, COMM)
    e = float(np.abs(to_dense(A, COMM) - D2).max()) / max(
        float(np.abs(D2).max()), 1e-300)
    check("elasticity operator matches ElasticityIntegrator", e < 1e-13,
          "(rel %.3e, %d tdofs)" % (e, V.GlobalTrueVSize()))


def test_mixed_formulation_blocks():
    r"""A mixed formulation whose two fields live in different families.

    Darcy, or mixed Poisson: find :math:`(\sigma, u)` in :math:`RT \times L^2` with
    ``R = int sigma.tau - u div tau + v div sigma``.  The four Jacobian blocks are second
    derivatives of that one density taken between slots of *different* families, each
    assembled by the ordinary block assembly, and they glue into the monolithic operator
    with ``mfem.HypreParMatrixFromBlocks``.  Two of the blocks have an MFEM integrator to
    compare against; the other two are each other's negative transpose, which the
    formulation requires.  ``PDEVariationalProblem`` gives each variable a single space, so such a
    system is assembled here directly.
    """
    if RANK == 0:
        print("mixed formulation across two element families (RT x L2)")
    SIG, U, TAU, V = 0, 1, 2, 3
    pm = mfem.ParMesh(COMM, mk("quad", 6))
    Vs = FunctionSpace.RT(pm, 1)
    Vu = FunctionSpace.L2(pm, 1)
    batches = MeshBatches(pm, 6, COMM)

    def density(sigma, u, tau, v, x):
        return jnp.dot(sigma.val, tau.val) - u.val * tau.div + v.val * sigma.div

    spaces = [Vs, Vu, Vs, Vu]
    K = QuadratureKernel(density, spaces, batches)
    zero = [sp.local_values(sp.vector()) for sp in spaces]      # the density is bilinear

    def block(i, j):
        return assemble_matrix(spaces[i], spaces[j], batches.groups,
                               K.element_matrices(i, j, zero), pm.GetNE())

    A11, A12, A21, A22 = block(TAU, SIG), block(TAU, U), block(V, SIG), block(V, U)
    form = mfem.ParMixedBilinearForm(Vs.fes, Vu.fes)
    form.AddDomainIntegrator(mfem.VectorFEDivergenceIntegrator())
    form.Assemble()
    form.Finalize()
    D_ref = form.ParallelAssemble()
    M_ref = hp.assemble_native_matrix(Vs, [mfem.VectorFEMassIntegrator()])
    worst = 0.0
    for ours, ref in ((A11, M_ref), (A21, D_ref)):
        a, b = to_dense(ours), to_dense(ref)
        worst = max(worst, np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    check("mixed-family blocks match MFEM's own integrators", worst < 1e-12,
          "(RT x RT mass and L2 x RT divergence, worst rel %.1e)" % worst)

    a12, a21 = to_dense(A12), to_dense(A21)
    e_t = np.abs(a12 + a21.T).max() / max(np.abs(a21).max(), 1e-300)
    check("the off-diagonal blocks are each other's negative transpose", e_t < 1e-12,
          "(|A12 + A21^T| rel %.1e)" % e_t)

    blocks = mfem.HypreParMatrixArray2D(2, 2)
    blocks[0, 0] = A11
    blocks[0, 1] = A12
    blocks[1, 0] = A21
    blocks[1, 1] = A22
    mono = mfem.HypreParMatrixFromBlocks(blocks)
    n = Vs.vector().global_size + Vu.vector().global_size
    check("the blocks glue into one parallel matrix", mono.GetGlobalNumRows() == n,
          "(%d x %d against %d unknowns)" % (mono.GetGlobalNumRows(),
                                             mono.GetGlobalNumCols(), n))


def test_poroelasticity_blocks():
    r"""Poroelasticity: a coupled multiphysics system from one residual density.

    One implicit step of Biot consolidation, displacement :math:`u` in a vector
    :math:`H^1` space and pressure :math:`p` in a scalar one::

        R = 2 mu eps(u):eps(v) + lam div u div v - alpha p div v
          + alpha div u q + (1/M) p q + kappa grad p . grad q

    The four Jacobian blocks are second derivatives of that density between slots in two
    different spaces.  The two diagonal blocks have MFEM integrators to compare against
    (elasticity, and mass plus diffusion); the off-diagonal pair must be each other's
    negative transpose, which is the discrete statement that the same :math:`\alpha`
    couples both equations; and the monolithic operator glued from the four must act as the four do.
    """
    if RANK == 0:
        print("poroelasticity: a coupled system from one density")
    U, P, V, Q = 0, 1, 2, 3
    mu, lam, alpha, Minv, kappa = 1.5, 2.0, 0.8, 0.25, 0.3
    pm = mfem.ParMesh(COMM, mk("quad", 6))
    dim = pm.Dimension()
    Vu = FunctionSpace.H1(pm, 2, vdim=dim)
    Vp = FunctionSpace.H1(pm, 1)
    batches = MeshBatches(pm, 8, COMM)

    def density(u, p, v, q, x):
        eu, ev = 0.5 * (u.grad + u.grad.T), 0.5 * (v.grad + v.grad.T)
        du, dv = jnp.trace(u.grad), jnp.trace(v.grad)
        return (2.0 * mu * jnp.sum(eu * ev) + lam * du * dv - alpha * p.val * dv
                + alpha * du * q.val + Minv * p.val * q.val
                + kappa * jnp.dot(p.grad, q.grad))

    spaces = [Vu, Vp, Vu, Vp]
    K = QuadratureKernel(density, spaces, batches)
    zero = [sp.local_values(sp.vector()) for sp in spaces]      # the density is bilinear

    def block(i, j):
        return assemble_matrix(spaces[i], spaces[j], batches.groups,
                               K.element_matrices(i, j, zero), pm.GetNE())

    Kuu, Kup, Kpu, Kpp = block(V, U), block(V, P), block(Q, U), block(Q, P)

    el = hp.assemble_native_matrix(Vu, [mfem.ElasticityIntegrator(
        mfem.ConstantCoefficient(lam), mfem.ConstantCoefficient(mu))])
    pr = hp.assemble_native_matrix(Vp, [mfem.MassIntegrator(mfem.ConstantCoefficient(Minv)),
                                        mfem.DiffusionIntegrator(mfem.ConstantCoefficient(kappa))])
    worst = 0.0
    for ours, ref in ((Kuu, el), (Kpp, pr)):
        a, b = to_dense(ours), to_dense(ref)
        worst = max(worst, np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    check("poroelastic diagonal blocks match MFEM's integrators", worst < 1e-12,
          "(elasticity, and mass + diffusion; worst rel %.1e)" % worst)

    up, pu = to_dense(Kup), to_dense(Kpu)
    e_t = np.abs(up + pu.T).max() / max(np.abs(pu).max(), 1e-300)
    check("the two coupling blocks are each other's negative transpose", e_t < 1e-12,
          "(|K_up + K_pu^T| rel %.1e)" % e_t)

    blocks = mfem.HypreParMatrixArray2D(2, 2)
    blocks[0, 0] = Kuu
    blocks[0, 1] = Kup
    blocks[1, 0] = Kpu
    blocks[1, 1] = Kpp
    mono = mfem.HypreParMatrixFromBlocks(blocks)

    xu, xp = Vu.vector(), Vp.vector()
    hp.parRandom.set_seed(11)
    hp.parRandom.normal(1.0, xu)
    hp.parRandom.normal(1.0, xp)
    yu, yp = Vu.vector(), Vp.vector()          # what the blocks give
    tmp = Vu.vector()
    Kuu.Mult(xu.hypre, yu.hypre)
    Kup.Mult(xp.hypre, tmp.hypre)
    yu.axpy(1.0, tmp)
    tmp2 = Vp.vector()
    Kpu.Mult(xu.hypre, yp.hypre)
    Kpp.Mult(xp.hypre, tmp2.hypre)
    yp.axpy(1.0, tmp2)

    x = hp.ParVector.from_array(COMM, np.concatenate([xu.array, xp.array]))
    y = hp.ParVector(COMM, x.local_size)
    mono.Mult(x.hypre, y.hypre)
    ref_all = np.concatenate([yu.array, yp.array])
    e = (np.abs(y.array - ref_all).max()
         / max(COMM.allreduce(np.abs(ref_all).max(), op=MPI.MAX), 1e-300))
    e = COMM.allreduce(e, op=MPI.MAX)
    check("the glued operator acts as the four blocks do", e < 1e-12,
          "(%d unknowns, max rel %.1e)" % (mono.GetGlobalNumRows(), e))


def test_vector_bc_values():
    """Essential data on a vector space: one number, a vector, and one component.

    A scalar on a ``vdim > 1`` space means the same value in every component.  MFEM's
    ``VectorConstantCoefficient`` takes a length-``vdim`` vector, so the scalar has to be
    broadcast before it is handed over; without that the constructor raises on a 0-d array
    and a clamp written as ``DirichletBC(Vvec, 0.0, ...)`` fails at the first solve.
    """
    if RANK == 0:
        print("essential data on a vector space")
    pm = mfem.ParMesh(COMM, mk("quad", 4))
    V = FunctionSpace.H1(pm, 2, vdim=2)
    left = 4

    bc = hp.DirichletBC(V, 0.5, bdr_attributes=[left])
    v = V.vector()
    v.array[:] = 7.0
    bc.apply(v)
    got = v.array[bc.ess] if bc.ess.size else np.zeros(0)
    worst = COMM.allreduce(float(np.abs(got - 0.5).max()) if got.size else 0.0, op=MPI.MAX)
    ndofs = COMM.allreduce(int(bc.ess.size), op=MPI.SUM)
    check("a scalar value applies to every component", worst < 1e-14 and ndofs > 0,
          "(%d essential dofs, worst |u - 0.5| = %.1e)" % (ndofs, worst))

    bc2 = hp.DirichletBC(V, [1.0, -2.0], bdr_attributes=[left])
    v2 = V.vector()
    v2.array[:] = 7.0
    bc2.apply(v2)
    vals = np.unique(np.round(v2.array[bc2.ess], 12)) if bc2.ess.size else np.zeros(0)
    seen = set(np.concatenate(COMM.allgather(vals)).round(12).tolist())
    check("a length-vdim value sets the components separately",
          seen <= {1.0, -2.0} and len(seen) > 0, "(values on the essential dofs: %s)"
          % sorted(seen))

    bc3 = hp.DirichletBC(V, 0.0, bdr_attributes=[left], component=1)
    n_all = COMM.allreduce(int(bc.ess.size), op=MPI.SUM)
    n_one = COMM.allreduce(int(bc3.ess.size), op=MPI.SUM)
    check("component= restricts the condition to one component", 2 * n_one == n_all,
          "(%d dofs of %d)" % (n_one, n_all))

    try:
        hp.DirichletBC(V, [1.0, 2.0, 3.0], bdr_attributes=[left]).apply(V.vector())
        raised = False
    except ValueError:
        raised = True
    check("a wrong number of components is refused", raised)


def main():
    test_vs_mfem()
    test_coefficient_and_derivatives()
    test_vector_h1()
    test_vector_bc_values()
    test_hcurl_inverse_problem()
    test_mixed_formulation_blocks()
    test_poroelasticity_blocks()
    COMM.Barrier()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

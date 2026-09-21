# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Interior facet terms: the ``dS`` of a discontinuous Galerkin form.

A facet density is written in the two traces a face has, and is differentiated like
any other density, so a DG residual generates its own blocks.  What is checked:

* an interior-penalty (SIPG) matrix against MFEM's own ``DGDiffusionIntegrator``,
  and the Nitsche boundary term against the same integrator on boundary faces,
  which is an exact comparison and the one that fixes every convention: the sign of
  the normal, the factor of a half in an average, and the face measure a penalty is
  scaled by;
* the facet residual against the matrix it differentiates to, which is the test of
  the scatter, since the two take different routes through the shared faces;
* a rectangular facet block (a DG state against a continuous parameter) against a
  finite difference of the residual, which no MFEM form assembles in parallel;
* a facet term whose test space is continuous, where the block is folded onto true
  dofs by the prolongation rather than being one already;
* a DG Poisson problem solved end to end against MFEM's assembly of the same
  problem, and against the solution it approximates;
* a DG inverse problem, where ``modelVerify``'s finite differences are the test
  that the facet term reaches ``C``, ``W_uu``, ``W_um`` and ``W_mm`` correctly.

Every check runs on any number of ranks: with more than one, the faces shared
between ranks are the interesting ones, since their second element lives on a
neighbour.

Run with ``mpirun -n N python -m hippymfem.test.test_facets``.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hp
from hippymfem.common.linalg import to_dense
from hippymfem.fem.facets import (assemble_facet_matrix, assemble_facet_vector,
                                  avg, avg_grad, facet_values, get_facet_batches,
                                  jump, jump_grad)
from hippymfem.fem.kernel import QuadratureKernel
from hippymfem.fem.spaces import FunctionSpace
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
FAILS = []

SIGMA, KAPPA = -1.0, 8.0


def check(name, ok, detail=""):
    if RANK == 0:
        print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


def mesh(n=4):
    return mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(n, n,
                                                        mfem.Element.QUADRILATERAL))


def sipg(u, p, x, n, h):
    """The interior-penalty form MFEM's ``DGDiffusionIntegrator`` assembles."""
    return (-jnp.dot(avg_grad(u), n) * jump(p)
            + SIGMA * jump(u) * jnp.dot(avg_grad(p), n)
            + KAPPA * 0.5 * (1.0 / h[0] + 1.0 / h[1]) * jump(u) * jump(p))


def nitsche(u, p, x, n, h):
    """Its boundary face counterpart: one side, so no average and no half."""
    return (-jnp.dot(u.grad, n) * p.val + SIGMA * u.val * jnp.dot(p.grad, n)
            + KAPPA * (1.0 / h) * u.val * p.val)


# --------------------------------------------------------------- against MFEM
def test_sipg_against_mfem():
    """The SIPG matrix, entry by entry, against the integrator it reproduces."""
    if RANK == 0:
        print("interior penalty against MFEM's DGDiffusionIntegrator")
    order = 2
    pm = mesh(4)
    V = FunctionSpace.L2(pm, order)
    batches = get_facet_batches(pm, 2 * order + 2)
    g = batches.groups[0]
    K = QuadratureKernel(sipg, [V, V], batches)
    zero = [facet_values(V, V.vector())] * 2
    A = assemble_facet_matrix(V, batches.groups, K.element_matrices(1, 0, zero))

    form = mfem.ParBilinearForm(V.fes)
    form.AddInteriorFaceIntegrator(
        mfem.DGDiffusionIntegrator(mfem.ConstantCoefficient(1.0), SIGMA, KAPPA))
    form.Assemble()
    form.Finalize()
    Aref = form.ParallelAssemble()
    a, b = to_dense(A, COMM), to_dense(Aref, COMM)
    e = np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
    check("SIPG matrix matches DGDiffusionIntegrator", e < 1e-13,
          "(max rel %.2e, %d faces, %d shared)"
          % (e, g.ne, int(COMM.allreduce(int(g.shared.sum()), op=MPI.SUM))))


def test_nitsche_against_mfem():
    """The boundary face term, where the face measure reaches a boundary density."""
    if RANK == 0:
        print("Nitsche boundary term against the same integrator")
    order = 2
    pm = mesh(4)
    V = FunctionSpace.L2(pm, order)
    from hippymfem.fem.boundary import (BoundaryKernel, assemble_boundary_matrix,
                                        get_boundary_batches)
    batches = get_boundary_batches(pm, 2 * order + 2, "all", COMM, space=V)
    K = BoundaryKernel(nitsche, [V, V], batches)
    zero = [V.local_values(V.vector())] * 2
    A = assemble_boundary_matrix(V, V, batches.groups,
                                 K.element_matrices(1, 0, zero))
    form = mfem.ParBilinearForm(V.fes)
    form.AddBdrFaceIntegrator(
        mfem.DGDiffusionIntegrator(mfem.ConstantCoefficient(1.0), SIGMA, KAPPA))
    form.Assemble()
    form.Finalize()
    Aref = form.ParallelAssemble()
    a, b = to_dense(A, COMM), to_dense(Aref, COMM)
    e = np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
    check("Nitsche matrix matches DGDiffusionIntegrator on boundary faces",
          e < 1e-13, "(max rel %.2e)" % e)


# ------------------------------------------------------------- residual, blocks
def test_residual_matches_matrix():
    """``R(u)`` from the one-sided scatter against ``A u`` from the matrix."""
    if RANK == 0:
        print("facet residual against the matrix it differentiates to")
    order = 2
    pm = mesh(4)
    V = FunctionSpace.L2(pm, order)
    batches = get_facet_batches(pm, 2 * order + 2)
    K = QuadratureKernel(sipg, [V, V], batches)
    hp.parRandom.set_seed(5)
    u = V.vector()
    hp.parRandom.normal(1.0, u)
    loc = [facet_values(V, u), facet_values(V, V.vector())]
    A = assemble_facet_matrix(V, batches.groups,
                              K.element_matrices(1, 0, loc))
    r = assemble_facet_vector(V, batches.groups, K.element_vectors(1, loc),
                              tables=batches.tables(V))
    Au = V.vector()
    A.Mult(u.hypre, Au.hypre)
    e = r.copy().axpy(-1.0, Au).norm("l2") / max(Au.norm("l2"), 1e-300)
    check("facet residual matches the matrix action", e < 1e-12,
          "(rel %.2e)" % e)


def test_rectangular_block():
    """A DG state against a continuous parameter: no MFEM form assembles it."""
    if RANK == 0:
        print("rectangular facet block against a finite difference")
    pm = mesh(4)
    Vu = FunctionSpace.L2(pm, 2)
    Vm = FunctionSpace.H1(pm, 1)
    batches = get_facet_batches(pm, 8)

    def facet(u, m, p, x, n, h):
        k = jnp.exp(avg(m))
        return (-k * jnp.dot(avg_grad(u), n) * jump(p)
                + SIGMA * k * jump(u) * jnp.dot(avg_grad(p), n)
                + KAPPA * k * 0.5 * (1 / h[0] + 1 / h[1]) * jump(u) * jump(p))

    K = QuadratureKernel(facet, [Vu, Vm, Vu], batches)
    hp.parRandom.set_seed(3)
    u, m, dm = Vu.vector(), Vm.vector(), Vm.vector()
    hp.parRandom.normal(1.0, u)
    hp.parRandom.normal(0.3, m)
    hp.parRandom.normal(1.0, dm)

    def loc(mv):
        return [facet_values(Vu, u), facet_values(Vm, mv),
                facet_values(Vu, Vu.vector())]

    tabs = batches.tables(Vu)
    C = assemble_facet_matrix(Vu, batches.groups,
                              K.element_matrices(ADJOINT, PARAMETER, loc(m)),
                              trial_space=Vm)
    Cdm = Vu.vector()
    C.Mult(dm.hypre, Cdm.hypre)
    eps = 1e-6
    rp = assemble_facet_vector(Vu, batches.groups,
                               K.element_vectors(ADJOINT, loc(m.copy().axpy(eps, dm))),
                               tables=tabs)
    rm = assemble_facet_vector(Vu, batches.groups,
                               K.element_vectors(ADJOINT, loc(m.copy().axpy(-eps, dm))),
                               tables=tabs)
    fd = rp.axpy(-1.0, rm).scale(1.0 / (2 * eps))
    e = fd.copy().axpy(-1.0, Cdm).norm("l2") / max(fd.norm("l2"), 1e-300)
    check("rectangular facet block matches a finite difference", e < 1e-7,
          "(rel %.2e, %d x %d)" % (e, C.GetGlobalNumRows(), C.GetGlobalNumCols()))


def test_continuous_test_space():
    """A facet term on a continuous space, where the rows are folded onto true dofs.

    Nothing in a DG form needs this, but a facet block of an inverse problem does:
    ``W_mm`` has the parameter space on both sides.  Two routes reach the true dofs,
    the prolongation's triple product for the matrix and the dual assembly for the
    vector, and they must agree.
    """
    if RANK == 0:
        print("facet term with a continuous test space")
    pm = mesh(4)
    V = FunctionSpace.H1(pm, 2)
    batches = get_facet_batches(pm, 6)

    def facet(u, p, x, n, h):
        # a face mass term: the jump of a continuous field is zero, the average is not
        return avg(u) * avg(p) * (1.0 + x[0])

    K = QuadratureKernel(facet, [V, V], batches)
    hp.parRandom.set_seed(11)
    u = V.vector()
    hp.parRandom.normal(1.0, u)
    loc = [facet_values(V, u), facet_values(V, V.vector())]
    A = assemble_facet_matrix(V, batches.groups, K.element_matrices(1, 0, loc))
    r = assemble_facet_vector(V, batches.groups, K.element_vectors(1, loc),
                              tables=batches.tables(V))
    Au = V.vector()
    A.Mult(u.hypre, Au.hypre)
    e = r.copy().axpy(-1.0, Au).norm("l2") / max(Au.norm("l2"), 1e-300)
    check("continuous-space facet block matches its residual", e < 1e-12,
          "(rel %.2e)" % e)


def test_vector_state():
    """A vector-valued DG space: the dof layout of a face is the volume one, twice."""
    if RANK == 0:
        print("a vector-valued DG state")
    pm = mesh(4)
    V = FunctionSpace.L2(pm, 1, vdim=2)
    batches = get_facet_batches(pm, 6)

    def facet(u, p, x, n, h):
        return (KAPPA * 0.5 * (1 / h[0] + 1 / h[1]) * jnp.dot(jump(u), jump(p))
                - jnp.dot(jnp.dot(avg_grad(u), n), jump(p)))

    K = QuadratureKernel(facet, [V, V], batches)
    hp.parRandom.set_seed(13)
    u = V.vector()
    hp.parRandom.normal(1.0, u)
    loc = [facet_values(V, u), facet_values(V, V.vector())]
    A = assemble_facet_matrix(V, batches.groups, K.element_matrices(1, 0, loc))
    r = assemble_facet_vector(V, batches.groups, K.element_vectors(1, loc),
                              tables=batches.tables(V))
    Au = V.vector()
    A.Mult(u.hypre, Au.hypre)
    e = r.copy().axpy(-1.0, Au).norm("l2") / max(Au.norm("l2"), 1e-300)
    check("vector-valued facet block matches its residual", e < 1e-12,
          "(rel %.2e, %d dofs)" % (e, A.GetGlobalNumRows()))


def test_mesh_without_interior_faces():
    """One element and no interior face at all: an empty block, not a crash."""
    if RANK == 0:
        print("a mesh with no interior face")
    if NP > 1:
        check("a mesh with no interior face gives an empty block", True,
              "(skipped: one rank only)")
        return
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(1, 1,
                                                      mfem.Element.QUADRILATERAL))
    V = FunctionSpace.L2(pm, 1)
    batches = get_facet_batches(pm, 4)
    K = QuadratureKernel(sipg, [V, V], batches)
    loc = [facet_values(V, V.vector())] * 2
    A = assemble_facet_matrix(V, batches.groups, K.element_matrices(1, 0, loc))
    r = assemble_facet_vector(V, batches.groups, K.element_vectors(1, loc),
                              tables=batches.tables(V))
    ok = (batches.groups[0].ne == 0 and A.GetGlobalNumRows() == V.fes.GlobalVSize()
          and np.abs(to_dense(A, COMM)).max() == 0.0 and r.norm("l2") == 0.0)
    check("a mesh with no interior face gives an empty block", ok,
          "(%d faces, %d x %d)" % (batches.groups[0].ne, A.GetGlobalNumRows(),
                                   A.GetGlobalNumCols()))


def test_nonconforming_mesh_refused():
    """A hanging node is refused by the facet kernels, with a message that says so.

    Domain and boundary densities work on a non-conforming mesh; interior facet terms do
    not, because a face with a hanging node is a master with slaves and the pairing the
    facet groups build (one element per side, one rule for both) does not describe it.
    Assembled anyway it is quietly inconsistent with the residual, and what the user sees
    is a Newton step that fails to solve a linear problem.
    """
    if RANK == 0:
        print("a non-conforming mesh is refused by the facet kernels")
    m = mfem.Mesh.MakeCartesian2D(8, 8, mfem.Element.QUADRILATERAL)
    m.EnsureNCMesh(True)
    marks = mfem.intArray()
    for e in range(m.GetNE()):
        verts = m.GetElementVertices(e)
        c = np.mean([m.GetVertexArray(int(v)) for v in verts], axis=0)
        if c[0] < 0.5 and c[1] < 0.5:
            marks.Append(e)
    m.GeneralRefinement(marks)
    pm = mfem.ParMesh(COMM, m)
    check("the refined mesh is non-conforming", bool(pm.Nonconforming()),
          "(%d elements)" % pm.GetNE())

    try:
        get_facet_batches(pm, 4)
        raised = ""
    except ValueError as e:
        raised = str(e)
    check("interior facet batches are refused on it",
          "non-conforming" in raised and "facet" in raised,
          "(%s)" % (raised.split(":")[0] if raised else "no error raised"))

    # ... while a domain integral on the same mesh is untouched
    V = FunctionSpace.H1(pm, 2)
    ok = V.GlobalTrueVSize() > 0
    try:
        from hippymfem.fem.elementbatch import get_batches

        b = get_batches(pm, 4)
        ok = ok and b.groups[0].ne > 0
    except Exception as e:                                       # noqa: BLE001
        ok = False
        raised = str(e)
    check("domain batches on the same mesh still build", ok,
          "(%d state dofs)" % V.GlobalTrueVSize())


# ------------------------------------------------------------------ end to end
def dg_poisson(n=8, order=2, kappa_of_m=False):
    """A SIPG Poisson problem: domain term, interior faces, Nitsche boundary."""
    pm = mesh(n)
    Vu = FunctionSpace.L2(pm, order)
    Vm = FunctionSpace.H1(pm, 1)
    two_pi2 = 2.0 * np.pi ** 2

    def source(x):
        return two_pi2 * jnp.sin(np.pi * x[0]) * jnp.sin(np.pi * x[1])

    if kappa_of_m:
        def kd(m):
            return jnp.exp(m.val)

        def kf(m):
            return jnp.exp(avg(m))

        def kb(m):
            return jnp.exp(m.val)
    else:
        kd = kf = kb = lambda m: 1.0

    def varf(u, m, p, x):
        return kd(m) * jnp.dot(u.grad, p.grad) - source(x) * p.val

    def facet(u, m, p, x, n, h):
        k = kf(m)
        return (-k * jnp.dot(avg_grad(u), n) * jump(p)
                + SIGMA * k * jump(u) * jnp.dot(avg_grad(p), n)
                + KAPPA * k * 0.5 * (1 / h[0] + 1 / h[1]) * jump(u) * jump(p))

    def bdr(u, m, p, x, n, h):
        k = kb(m)                       # homogeneous Dirichlet data, imposed weakly
        return (-k * jnp.dot(u.grad, n) * p.val + SIGMA * k * u.val * jnp.dot(p.grad, n)
                + KAPPA * k * (1.0 / h) * u.val * p.val)

    pde = hp.PDEVariationalProblem(
        [Vu, Vm, Vu], varf, None, None, is_fwd_linear=True,
        quadrature_degree=2 * order + 2, bdr_varf=bdr, facet_varf=facet,
        spd_jacobian=True)
    return pm, Vu, Vm, pde


class _Exact(mfem.PyCoefficient):
    """The solution the manufactured source produces."""

    def EvalValue(self, z):
        return float(np.sin(np.pi * z[0]) * np.sin(np.pi * z[1]))


def test_dg_solve():
    """The assembled system against MFEM's, and the solution against the exact one."""
    if RANK == 0:
        print("a DG Poisson problem, assembled and solved")
    order = 2
    pm, Vu, Vm, pde = dg_poisson(n=8, order=order)
    m = Vm.vector()
    u = pde.generate_state()
    x = [u, m, pde.generate_adjoint()]
    A = pde._block(ADJOINT, STATE, x)

    k = mfem.ConstantCoefficient(1.0)
    form = mfem.ParBilinearForm(Vu.fes)
    form.AddDomainIntegrator(mfem.DiffusionIntegrator(k))
    form.AddInteriorFaceIntegrator(mfem.DGDiffusionIntegrator(k, SIGMA, KAPPA))
    form.AddBdrFaceIntegrator(mfem.DGDiffusionIntegrator(k, SIGMA, KAPPA))
    form.Assemble()
    form.Finalize()
    Aref = form.ParallelAssemble()
    a, b = to_dense(A, COMM), to_dense(Aref, COMM)
    e = np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
    check("the DG Jacobian matches MFEM's assembly of the same problem", e < 1e-12,
          "(max rel %.2e, %d dofs)" % (e, A.GetGlobalNumRows()))

    pde.solveFwd(u, x)
    gf = Vu.to_gridfunction(u)
    err = gf.ComputeL2Error(_Exact())
    check("the DG solution approximates the exact one", err < 2e-3,
          "(L2 error %.3e at order %d, 8 x 8)" % (err, order))

    pm2, Vu2, Vm2, pde2 = dg_poisson(n=16, order=order)
    u2 = pde2.generate_state()
    pde2.solveFwd(u2, [u2, Vm2.vector(), None])
    err2 = Vu2.to_gridfunction(u2).ComputeL2Error(_Exact())
    rate = np.log2(max(err, 1e-300) / max(err2, 1e-300))
    check("the DG solution converges at the expected order",
          rate > order + 0.7, "(rate %.2f, expected %d)" % (rate, order + 1))


def test_dg_inverse_problem():
    """A DG forward problem inside an inverse problem: every block has a facet part."""
    if RANK == 0:
        print("a DG inverse problem, by finite differences of cost and gradient")
    pm, Vu, Vm, pde = dg_poisson(n=6, order=1, kappa_of_m=True)
    prior = hp.BiLaplacianPrior(Vm, 0.2, 0.8, robin_bc=True,
                                solver_type="lu" if NP == 1 else "krylov")
    hp.parRandom.set_seed(7)
    mtrue = Vm.project(lambda z: 0.4 * np.sin(2.0 * z[0]) + 0.2 * z[1])
    ut = pde.generate_state()
    pde.solveFwd(ut, [ut, mtrue, None])
    d = ut.copy()
    std = 0.01 * max(d.norm("linf"), 1e-30)
    hp.parRandom.normal_perturb(std, d)
    misfit = hp.ContinuousStateObservation(Vu, data=d, noise_variance=std ** 2)
    model = hp.Model(pde, prior, misfit)
    m0 = prior.mean.copy()
    hp.parRandom.normal(0.2, m0)
    eps, err_grad, err_H = hp.modelVerify(model, m0, is_quadratic=False,
                                          verbose=False,
                                          eps=np.logspace(-2, -7, 10))
    gslope = hp.best_slope(eps, err_grad)
    hslope = hp.best_slope(eps, err_H)
    check("the DG gradient has first-order FD slope", abs(gslope - 1.0) < 0.1,
          "(slope %.4f)" % gslope)
    check("the DG Hessian has first-order FD slope", abs(hslope - 1.0) < 0.1,
          "(slope %.4f)" % hslope)


def test_facets_with_essential_bcs():
    """A stabilised continuous problem: facet terms beside essential conditions.

    The jump of a continuous field is zero but the jump of its gradient is not, which
    is the interior-penalty stabilisation of a convection-dominated problem.  The
    facet block is summed into the domain block before the essential rows go, so those
    rows must come out as the identity and not twice it.
    """
    if RANK == 0:
        print("facet terms with essential boundary conditions")
    pm = mesh(6)
    Vu = FunctionSpace.H1(pm, 1)
    Vm = FunctionSpace.H1(pm, 1)
    bc = hp.DirichletBC(Vu, 0.0)

    def varf(u, m, p, x):
        return (jnp.exp(m.val) * jnp.dot(u.grad, p.grad)
                + jnp.dot(jnp.array([1.0, 0.5]), u.grad) * p.val - p.val)

    def facet(u, m, p, x, n, h):
        return (0.1 * h[0] ** 2 * jnp.exp(avg(m))
                * jnp.dot(jump_grad(u), n) * jnp.dot(jump_grad(p), n))

    pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(),
                                   is_fwd_linear=True, facet_varf=facet)
    plain = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(),
                                     is_fwd_linear=True)
    hp.parRandom.set_seed(17)
    m = Vm.vector()
    hp.parRandom.normal(0.3, m)
    u = pde.generate_state()
    pde.solveFwd(u, [u, m, None])
    x = [u, m, pde.generate_adjoint()]
    A = pde._block(ADJOINT, STATE, x, test_ess=pde.bc0.ess_tdof)
    B = plain._block(ADJOINT, STATE, x, test_ess=plain.bc0.ess_tdof)
    diff = float(np.abs(to_dense(A, COMM) - to_dense(B, COMM)).max())
    check("the facet term changes the Jacobian", diff > 1e-6,
          "(max diff %.2e)" % diff)

    ess = np.asarray(pde.bc0.ess, dtype=np.int64)
    v, Av = Vu.vector(), Vu.vector()
    v.array[:] = 0.0
    v.array[ess] = 1.0
    A.Mult(v.hypre, Av.hypre)
    got = Av.array.copy()
    worst = np.abs(got[ess] - 1.0).max() if ess.size else 0.0
    keep = np.ones(got.size, dtype=bool)
    keep[ess] = False
    worst = max(worst, np.abs(got[keep]).max() if keep.any() else 0.0)
    worst = COMM.allreduce(float(worst), op=MPI.MAX)
    check("the essential rows come out as the identity", worst < 1e-14,
          "(worst %.2e over %d essential dofs)"
          % (worst, COMM.allreduce(int(ess.size), op=MPI.SUM)))

    prior = hp.BiLaplacianPrior(Vm, 0.2, 0.8, robin_bc=True,
                                solver_type="lu" if NP == 1 else "krylov")
    d = u.copy()
    std = 0.01 * max(d.norm("linf"), 1e-30)
    hp.parRandom.normal_perturb(std, d)
    misfit = hp.ContinuousStateObservation(Vu, data=d, noise_variance=std ** 2)
    model = hp.Model(pde, prior, misfit)
    m0 = prior.mean.copy()
    hp.parRandom.normal(0.2, m0)
    eps, err_grad, err_H = hp.modelVerify(model, m0, is_quadratic=False,
                                          verbose=False,
                                          eps=np.logspace(-2, -7, 10))
    gslope = hp.best_slope(eps, err_grad)
    hslope = hp.best_slope(eps, err_H)
    check("the stabilised gradient has first-order FD slope",
          abs(gslope - 1.0) < 0.1, "(slope %.4f)" % gslope)
    check("the stabilised Hessian has first-order FD slope",
          abs(hslope - 1.0) < 0.1, "(slope %.4f)" % hslope)


def main():
    test_sipg_against_mfem()
    test_nitsche_against_mfem()
    test_residual_matches_matrix()
    test_rectangular_block()
    test_continuous_test_space()
    test_vector_state()
    test_mesh_without_interior_faces()
    test_nonconforming_mesh_refused()
    test_dg_solve()
    test_dg_inverse_problem()
    test_facets_with_essential_bcs()
    COMM.Barrier()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Boundary integrals in the AD route: ``ds`` terms, and the blocks they generate.

Boundary terms carry Neumann and Robin conditions, boundary sources and boundary
misfits: everything UFL writes with ``ds``.  What is checked:

* a boundary mass matrix and a boundary normal-flux functional against MFEM's own
  ``BoundaryIntegrator`` and ``BoundaryNormalLFIntegrator``, which is an exact
  comparison and the one that validates the face-to-volume mapping (the boundary
  element's own finite element cannot produce ``u.grad``, so each face quadrature
  point is mapped back into the adjacent volume element);
* a Robin problem solved end to end against the same problem assembled from MFEM
  integrators;
* the derivative blocks: with a boundary coefficient that depends on the
  parameter, the boundary term enters ``C``, ``W_um`` and ``W_mm``, and
  ``modelVerify``'s finite differences are the test that it enters them
  correctly;
* attribute-restricted boundary terms, and a boundary term coexisting with
  essential conditions on the rest of the boundary.

Run with ``mpirun -n N python -m hippymfem.test.test_boundary``.
"""

import gc

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hp
from hippymfem.common.linalg import to_dense
from hippymfem.fem.boundary import (
    BoundaryBatches,
    BoundaryKernel,
    assemble_boundary_matrix,
    assemble_boundary_vector,
    get_boundary_batches,
)
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


def mesh_of(kind, n):
    m = (mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE) if kind == "tri"
         else mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.QUADRILATERAL))
    return mfem.ParMesh(COMM, m)


class _VecPy(mfem.VectorPyCoefficient):
    def __init__(self, fn, dim=2):
        super(_VecPy, self).__init__(dim)
        self.fn = fn

    def EvalValue(self, x):
        return np.asarray(self.fn(np.asarray(x)), dtype=np.float64)


class _ScalPy(mfem.PyCoefficient):
    def __init__(self, fn):
        super(_ScalPy, self).__init__()
        self.fn = fn

    def EvalValue(self, x):
        return float(self.fn(np.asarray(x)))


def test_against_mfem():
    """Boundary mass and normal flux, against MFEM's own boundary integrators."""
    if RANK == 0:
        print("boundary integrals vs MFEM")
    for kind in ("tri", "quad"):
        for order in (1, 2):
            pm = mesh_of(kind, 5)
            Vu = FunctionSpace.H1(pm, order)
            Vm = FunctionSpace.H1(pm, 1)
            bb = BoundaryBatches(pm, 2 * order + 2, "all", COMM, space=Vu)
            zero = [Vu.local_values(Vu.vector()), Vm.local_values(Vm.vector()),
                    Vu.local_values(Vu.vector())]

            K = BoundaryKernel(lambda u, m, p, x, n: u.val * p.val,
                               [Vu, Vm, Vu], bb)
            Mb = assemble_boundary_matrix(
                Vu, Vu, bb.groups, K.element_matrices(ADJOINT, STATE, zero))
            form = mfem.ParBilinearForm(Vu.fes)
            one = mfem.ConstantCoefficient(1.0)
            form.AddBoundaryIntegrator(mfem.MassIntegrator(one))
            form.Assemble()
            form.Finalize()
            Ref = mfem.HypreParMatrix()
            form.FormSystemMatrix(mfem.intArray(), Ref)
            D1, D2 = to_dense(Mb, COMM), to_dense(Ref, COMM)
            e = float(np.abs(D1 - D2).max()) / max(float(np.abs(D2).max()), 1e-300)
            check("%s P%d boundary mass matrix" % (kind, order), e < 1e-13,
                  "(rel %.3e)" % e)

            # A field whose gradient is exact in this space, so the comparison
            # tests the gradient evaluation and not the interpolation error.
            if kind == "quad":
                fn = lambda x: x[0] * x[1]
                grad = lambda x: np.array([x[1], x[0]])
            else:
                fn = lambda x: 2.0 * x[0] - 3.0 * x[1]
                grad = lambda x: np.array([2.0, -3.0])
            uex = Vu.project(fn)
            locu = [Vu.local_values(uex), Vm.local_values(Vm.vector()),
                    Vu.local_values(Vu.vector())]
            Kf = BoundaryKernel(lambda u, m, p, x, n: jnp.dot(u.grad, n) * p.val,
                                [Vu, Vm, Vu], bb)
            got = assemble_boundary_vector(
                Vu, bb.groups, Kf.element_vectors(ADJOINT, locu))
            lf = mfem.ParLinearForm(Vu.fes)
            c = _VecPy(grad)
            lf.AddBoundaryIntegrator(mfem.BoundaryNormalLFIntegrator(c))
            lf.Assemble()
            ref = Vu.vector()
            ref.array[:] = hp.to_numpy(lf.ParallelAssemble(), copy=True)
            d = got.copy()
            d.axpy(-1.0, ref)
            e2 = d.norm("l2") / max(ref.norm("l2"), 1e-300)
            check("%s P%d boundary normal flux (needs u.grad)" % (kind, order),
                  e2 < 1e-12, "(rel %.3e)" % e2)


def test_attribute_restriction():
    """A term on one attribute must touch only that part of the boundary."""
    if RANK == 0:
        print("attribute restriction")
    pm = mesh_of("quad", 6)
    V = FunctionSpace.H1(pm, 1)
    Vm = FunctionSpace.H1(pm, 1)
    zero = [V.local_values(V.vector()), Vm.local_values(Vm.vector()),
            V.local_values(V.vector())]
    total = None
    pieces = []
    empty_somewhere = False
    for attrs in ("all", [1], [2], [3], [4]):
        bb = BoundaryBatches(pm, 4, attrs, COMM, space=V)
        # A rank that owns no boundary element of this attribute has no element
        # groups.  Everything downstream is collective, so it must still build a
        # correctly shaped empty contribution: a zero-row one hangs rather than
        # raising, and never shows on one rank.
        ngroups = COMM.allgather(len(bb.groups))
        empty_somewhere = empty_somewhere or (0 in ngroups)
        K = BoundaryKernel(lambda u, m, p, x, n: p.val, [V, Vm, V], bb)
        v = assemble_boundary_vector(V, bb.groups,
                                     K.element_vectors(ADJOINT, zero))
        s = v.sum()
        if attrs == "all":
            total = s
        else:
            pieces.append(s)
    check("some rank owns no boundary element of some attribute",
          NP == 1 or empty_somewhere,
          "(the case that must not deadlock; trivially absent on one rank)")
    check("the four sides sum to the whole boundary",
          abs(sum(pieces) - total) < 1e-13 * max(abs(total), 1e-300),
          "(perimeter %.12f vs %.12f)" % (sum(pieces), total))
    check("the whole boundary measure is 4", abs(total - 4.0) < 1e-13,
          "(%.12f)" % total)


def build_robin(m_field=None, kappa_of_m=False, dirichlet=False, n=8, order=2):
    """``-div(exp(m) grad u) = f`` with ``exp(m) du/dn + kappa u = g``."""
    pm = mesh_of("tri", n)
    Vu = FunctionSpace.H1(pm, order)
    Vm = FunctionSpace.H1(pm, 1)

    def f(x):
        return 1.0 + x[0]

    def g(x):
        return 0.5 * x[1]

    def varf(u, m, p, x):
        return jnp.exp(m.val) * jnp.dot(u.grad, p.grad) - (1.0 + x[0]) * p.val

    if kappa_of_m:
        def bdr(u, m, p, x, n):
            return (jnp.exp(m.val) * u.val - 0.5 * x[1]) * p.val
    else:
        def bdr(u, m, p, x, n):
            return (2.0 * u.val - 0.5 * x[1]) * p.val

    bc = None
    if dirichlet:
        bc = hp.DirichletBC(Vu, lambda x: x[1], bdr_attributes=[1])
    # Robin on the sides that carry no essential condition
    attrs = [2, 3, 4] if dirichlet else "all"
    pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc,
                                   bc.homogeneous() if bc else None,
                                   is_fwd_linear=True, bdr_varf=bdr,
                                   bdr_attributes=attrs)
    for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        if NP == 1:
            setattr(pde, a, hp.LUSolver(COMM))
        else:
            s = hp.KrylovSolver(COMM, "gmres", "amg")
            s.parameters["rel_tolerance"] = 1e-13
            s.parameters["max_iter"] = 2000
            setattr(pde, a, s)
    return pm, Vu, Vm, pde, f, g


def test_robin_solve():
    """A Robin problem against the same problem assembled from MFEM integrators.

    ``m = 0`` makes ``exp(m) = 1``, so MFEM's constant-coefficient integrators
    give an exact reference.  With no essential condition anywhere, the boundary
    term is what makes the problem solvable at all: get it wrong and the matrix is
    singular.
    """
    if RANK == 0:
        print("Robin problem")
    pm, Vu, Vm, pde, f, g = build_robin()
    u = pde.generate_state()
    m = Vm.vector()                       # m = 0 -> exp(m) = 1
    pde.solveFwd(u, [u, m, None])

    form = mfem.ParBilinearForm(Vu.fes)
    one = mfem.ConstantCoefficient(1.0)
    two = mfem.ConstantCoefficient(2.0)
    form.AddDomainIntegrator(mfem.DiffusionIntegrator(one))
    form.AddBoundaryIntegrator(mfem.MassIntegrator(two))
    form.Assemble()
    form.Finalize()
    A = mfem.HypreParMatrix()
    form.FormSystemMatrix(mfem.intArray(), A)
    lf = mfem.ParLinearForm(Vu.fes)
    fc, gc = _ScalPy(f), _ScalPy(g)
    lf.AddDomainIntegrator(mfem.DomainLFIntegrator(fc))
    lf.AddBoundaryIntegrator(mfem.BoundaryLFIntegrator(gc))
    lf.Assemble()
    b = Vu.vector()
    b.array[:] = hp.to_numpy(lf.ParallelAssemble(), copy=True)

    uref = Vu.vector()
    if NP == 1:
        solver = hp.LUSolver(COMM)
    else:
        solver = hp.KrylovSolver(COMM, "cg", "amg")
        solver.parameters["rel_tolerance"] = 1e-14
        solver.parameters["max_iter"] = 3000
    solver.set_operator(A)
    solver.solve(uref, b)

    d = u.copy()
    d.axpy(-1.0, uref)
    rel = d.norm("l2") / max(uref.norm("l2"), 1e-300)
    check("Robin solve matches the MFEM-integrator reference", rel < 1e-9,
          "(rel %.3e, ||u|| = %.6f)" % (rel, u.norm("l2")))

    # the sign convention is settled by the comparison above; this rules out both
    # solutions being trivially zero and agreeing for that reason
    check("Robin solution is nonzero", u.norm("linf") > 1e-6,
          "(max |u| = %.4f)" % u.norm("linf"))


def test_blocks_with_boundary():
    """Derivative blocks must include the boundary term.

    The boundary coefficient is ``exp(m)``, so the boundary contributes to ``C``,
    ``W_um`` and ``W_mm``.  ``modelVerify`` differences the cost and the gradient,
    which is the test that the contribution is right and not merely present.
    """
    if RANK == 0:
        print("derivative blocks with a parameter-dependent boundary term")
    pm, Vu, Vm, pde, _, _ = build_robin(kappa_of_m=True, n=6, order=1)

    # the boundary term must actually change the blocks
    pde_no_bdr = hp.PDEVariationalProblem(
        [Vu, Vm, Vu], lambda u, m, p, x: (jnp.exp(m.val) * jnp.dot(u.grad, p.grad)
                                          - (1.0 + x[0]) * p.val),
        None, None, is_fwd_linear=True)
    hp.parRandom.set_seed(4)
    mrand = Vm.vector()
    hp.parRandom.normal(0.3, mrand)
    u = pde.generate_state()
    pde.solveFwd(u, [u, mrand, None])
    x = [u, mrand, pde.generate_adjoint()]
    Cb = pde._block(ADJOINT, PARAMETER, x)
    Cn = pde_no_bdr._block(ADJOINT, PARAMETER, x)
    diff = float(np.abs(to_dense(Cb, COMM) - to_dense(Cn, COMM)).max())
    check("the boundary term changes C", diff > 1e-6, "(max diff %.3e)" % diff)

    prior = hp.BiLaplacianPrior(Vm, 0.2, 0.8, robin_bc=True,
                                solver_type="lu" if NP == 1 else "krylov")
    rng = np.random.default_rng(7)
    targets = np.column_stack((rng.uniform(0.15, 0.85, 25),
                               rng.uniform(0.15, 0.85, 25)))
    B = hp.assemblePointwiseObservation(Vu, targets)
    hp.parRandom.set_seed(7)
    mtrue = Vm.project(lambda z: 0.4 * np.sin(2.0 * z[0]) + 0.2 * z[1])
    ut = pde.generate_state()
    pde.solveFwd(ut, [ut, mtrue, None])
    data = B.createVecLeft()
    B.mult(ut, data)
    std = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, std)
    misfit = hp.DiscreteStateObservation(B, data, std ** 2)
    model = hp.Model(pde, prior, misfit)

    m0 = prior.mean.copy()
    hp.parRandom.normal(0.2, m0)
    eps, err_grad, err_H = hp.modelVerify(
        model, m0, is_quadratic=False, verbose=False,
        eps=np.logspace(-2, -7, 10))
    gslope = hp.best_slope(eps, err_grad)
    hslope = hp.best_slope(eps, err_H)
    check("boundary-aware gradient has first-order FD slope",
          abs(gslope - 1.0) < 0.1, "(slope %.4f)" % gslope)
    check("boundary-aware Hessian has first-order FD slope",
          abs(hslope - 1.0) < 0.1, "(slope %.4f)" % hslope)

    # symmetry of the reduced Hessian, which the boundary blocks must preserve
    xh = model.generate_vector()
    xh[PARAMETER] = m0.copy()
    model.solveFwd(xh[STATE], xh)
    model.solveAdj(xh[ADJOINT], xh)
    model.setPointForHessianEvaluations(xh)
    H = hp.ReducedHessian(model, misfit_only=True)
    a = model.generate_vector(PARAMETER)
    b = model.generate_vector(PARAMETER)
    hp.parRandom.normal(1.0, a)
    hp.parRandom.normal(1.0, b)
    Ha = model.generate_vector(PARAMETER)
    Hb = model.generate_vector(PARAMETER)
    H.mult(a, Ha)
    H.mult(b, Hb)
    num = abs(b.inner(Ha) - a.inner(Hb))
    den = max(abs(b.inner(Ha)), 1e-300)
    check("Hessian is symmetric with the boundary term", num / den < 1e-9,
          "(rel %.3e)" % (num / den))


def test_dirichlet_plus_robin():
    """A Robin term on part of the boundary next to an essential condition.

    The domain and boundary parts are summed before elimination; adding them after
    would put 2.0 on the essential diagonal, so this checks the diagonal.
    """
    if RANK == 0:
        print("Dirichlet and Robin together")
    pm, Vu, Vm, pde, _, _ = build_robin(dirichlet=True, n=6, order=1)
    u = pde.generate_state()
    m = Vm.vector()
    pde.solveFwd(u, [u, m, None])
    x = [u, m, pde.generate_adjoint()]
    A, _ = pde._jacobian(x)
    ess = pde.bc0.ess
    D = to_dense(A, COMM)
    lo = Vu.vector().owner_range[0]
    gess = np.unique(np.concatenate(COMM.allgather(ess + lo)).astype(np.int64)) \
        if NP > 1 or ess.size else ess
    if RANK == 0 and gess.size:
        diag = D[gess, gess]
        rows = D[gess, :].copy()
        rows[np.arange(gess.size), gess] = 0.0
        ok = (np.allclose(diag, 1.0, rtol=0, atol=1e-14)
              and float(np.abs(rows).max()) == 0.0)
        detail = "(%d essential dofs, diag in [%.3f, %.3f], off-diag %.1e)" % (
            gess.size, diag.min(), diag.max(), float(np.abs(rows).max()))
    else:
        ok, detail = True, ""
    check("essential diagonal is exactly 1 with a boundary term present",
          bool(COMM.bcast(ok, root=0)), COMM.bcast(detail, root=0))
    check("the mixed problem solves", np.isfinite(u.norm("l2")) and u.norm("l2") > 0,
          "(||u|| = %.6f)" % u.norm("l2"))


def test_rank_without_boundary_elements():
    """A rank that owns no element of the boundary pattern must neither hang nor
    corrupt the matrix.

    Two traps: element kernels that resolve their device with a collective on first
    use, which ranks without elements never reach (a deadlock), and a block
    constructor that hands hypre two empty views of one accumulator, which MFEM's
    memory manager registers twice on a device.  The mesh is tall so that a
    partition leaves the bottom face to a subset of the ranks; the check is the
    bottom face's area, ``1^T M 1 = 1`` for the boundary mass matrix, and a second
    matrix that survives the first one's release.
    """
    if RANK == 0:
        print("a rank without boundary elements")
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(2, 2, 8, mfem.Element.HEXAHEDRON))
    V = FunctionSpace.H1(pm, 1)
    bb = get_boundary_batches(pm, 4, [1])
    ne = COMM.allgather(sum(g.ne for g in bb.groups))
    K = BoundaryKernel(lambda u, m, p, x, n: u.val * p.val, [V, V, V], bb)
    loc = V.local_values(V.vector())
    M1 = assemble_boundary_matrix(V, V, bb.groups, K.element_matrices(2, 0, [loc, loc, loc]))
    M2 = assemble_boundary_matrix(V, V, bb.groups, K.element_matrices(2, 0, [loc, loc, loc]))
    ones = V.vector(); ones.array[:] = 1.0
    y = V.vector()
    M1.Mult(ones.hypre, y.hypre)
    area1 = y.inner(ones)
    del M1
    gc.collect()
    M2.Mult(ones.hypre, y.hypre)
    area2 = y.inner(ones)
    check("bottom-face mass matrix sums to the face area on every partition",
          abs(area1 - 1.0) < 1e-12 and abs(area2 - 1.0) < 1e-12,
          "(1^T M 1 = %.15f, %.15f; boundary elements per rank %s)" % (area1, area2, ne))


def test_streamed_pass_with_boundary():
    """A linearization point assembled by the streamed shared pass carries the
    boundary residual's blocks too.

    With a boundary residual present, the streamed pass scatters the residual's
    blocks into the domain blocks' own slots
    (``csrassemble.add_boundary_entries``).  The pass is forced here by pinning
    the element chunk, so it runs on the host; the blocks must match the unchunked
    point up to the round-off chunking introduces (about 1e-15 relative).
    """
    if RANK == 0:
        print("streamed linearization point with a boundary residual")
    from hippymfem.fem import kernel as K

    pm, Vu, Vm, pde, f, g = build_robin(kappa_of_m=True, dirichlet=True, n=6, order=2)
    m = Vm.vector()
    hp.parRandom.set_seed(21)
    hp.parRandom.normal(0.3, m)
    x = [Vu.vector(), m, Vu.vector()]
    pde.solveFwd(x[STATE], x)
    hp.parRandom.normal(1.0, x[ADJOINT])
    pde.bc0.zero(x[ADJOINT])
    du, dm = Vu.vector(), Vm.vector()
    hp.parRandom.normal(1.0, du)
    hp.parRandom.normal(1.0, dm)
    pairs = [(ADJOINT, PARAMETER), (STATE, STATE), (STATE, PARAMETER), (PARAMETER, PARAMETER)]

    def blocks():
        pde.setLinearizationPoint(x, gauss_newton_approx=False)
        out = {}
        for (i, j) in pairs:
            d = du if j == STATE else dm
            y = (Vm if i == PARAMETER else Vu).vector()    # rows of block (i, j)
            pde.apply_ij(i, j, d, y)
            out[(i, j)] = y
        return out

    plain = blocks()
    old_chunk = K.ELEMENT_CHUNK
    K.ELEMENT_CHUNK = 3
    try:
        streamed_used = pde._stream_shared_pass()
        chunked = blocks()
    finally:
        K.ELEMENT_CHUNK = old_chunk
    worst = max(plain[k].copy().axpy(-1.0, chunked[k]).norm("l2") / max(plain[k].norm("l2"), 1e-300)
                for k in pairs)
    check("the streamed pass ran with a boundary residual", bool(streamed_used))
    check("streamed and plain linearization points agree with a boundary residual",
          worst < 1e-12, "(worst rel %.1e)" % worst)


def main():
    test_against_mfem()
    test_attribute_restriction()
    test_robin_solve()
    test_blocks_with_boundary()
    test_dirichlet_plus_robin()
    test_rank_without_boundary_elements()
    test_streamed_pass_with_boundary()
    COMM.Barrier()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

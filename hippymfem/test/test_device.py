# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""hypre on a device: what only a GPU-built PyMFEM (CUDA or HIP) can exercise.

The rest of the suite runs hypre on the host, and :mod:`test_gpu` moves the element
kernels to a GPU while hypre stays there.  Neither sees the parallel matrices
themselves in device memory, which is where the main lifetime rule checked here
applies:

*Two live matrices must not be built from the same host arrays.*  With
``HYPRE_USING_GPU``, MFEM's ``CopyCSR`` aliases a ``SparseMatrix``'s arrays rather
than copying them, and the ``MakeAlias`` inside it registers the base host pointer
with MFEM's memory manager, which owns the device mirror and is keyed on that
pointer.  Matrices built from a pattern's shared ``I``/``J`` would share one mirror,
and destroying either would free the other's: an illegal address in the next
BoomerAMG setup.  A solver that keeps every operator it was given hides this and
leaks device memory, so operator release is checked too.

Needs a PyMFEM built for a GPU (``tools/build_pymfem_cuda.sh`` for NVIDIA,
``tools/build_pymfem_hip.sh`` for AMD) and ``HIPPYMFEM_HYPRE_DEVICE=1`` in the
environment before import; otherwise it says so and exits 0.  Run with::

    HIPPYMFEM_HYPRE_DEVICE=1 PYTHONPATH=<gpu pymfem site-packages> \\
        mpirun -n N tools/mpirun_pinned.sh python -m hippymfem.test.test_device
"""

import gc
import os
import subprocess
import sys

import hippymfem as hp                       # first, so it can pin this rank's card
from hippymfem.common.mfemconfig import mfem_gpu_backend
from hippymfem._jaxconfig import hypre_on_device
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.rank

if mfem_gpu_backend() is None or not hypre_on_device():
    if RANK == 0:
        print("test_device: needs a CUDA- or HIP-built PyMFEM and HIPPYMFEM_HYPRE_DEVICE=1 "
              "set before import; nothing to test with this build")
    sys.exit(0)

import numpy as np
import mfem.par as mfem

from hippymfem.fem import assemble as asm
from hippymfem.fem import csrassemble as csr
from hippymfem.modeling.variables import ADJOINT, STATE
from hippymfem.test.test_assembly import build, locals_at

FAILS = []


def check(name, ok, detail=""):
    ok = bool(COMM.allreduce(bool(ok), op=MPI.LAND))
    if RANK == 0:
        print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


hp.configure_device("gpu", COMM, quiet=(RANK != 0))


def card_mib():
    """Memory in use on this rank's card, from nvidia-smi or rocm-smi; None if unavailable."""
    from hippymfem._jaxconfig import _visible_var
    vis = (_visible_var() or "0").split(",")[0]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits", "-i", vis],
                             capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                             capture_output=True, text=True, timeout=20)
        rows = [r.split(",") for r in out.stdout.strip().splitlines() if r.startswith("card")]
        used = [r for r in rows if len(r) > 2]
        # rocm-smi lists only the cards this process may see; ROCR_VISIBLE_DEVICES
        # renumbers them from zero, so the first row is this rank's card
        return float(used[0][2]) / 2 ** 20
    except Exception:
        return None


def problem(n=6):
    pm, Vh, b, K = build("hex", n, 2)
    bc = hp.DirichletBC(Vh[0], None, "all")
    return pm, Vh, b, K, bc


def matrix(pm, Vh, b, K, bc, seed):
    loc = locals_at(Vh, seed=seed)
    return asm.assemble_matrix(Vh[0], Vh[0], b.groups,
                               K.element_matrices(ADJOINT, STATE, loc), pm.GetNE(),
                               test_ess=bc.ess_tdof)


def amg_cg(A, iters=8):
    amg = mfem.HypreBoomerAMG(A)
    amg.SetPrintLevel(0)
    cg = mfem.CGSolver(COMM)
    cg.SetOperator(A)
    cg.SetPreconditioner(amg)
    cg.SetMaxIter(iters)
    cg.SetPrintLevel(-1)
    x = hp.ParVector(COMM, A.Height())
    rhs = hp.ParVector(COMM, A.Height())
    rhs.set(1.0)
    cg.Mult(rhs.hypre, x.hypre)
    return float(x.hypre.Norml2())


def test_sibling_release():
    """Destroying one device matrix must leave its sibling usable, on both routes."""
    if RANK == 0:
        print("a matrix survives the destruction of one built from the same pattern")
    pm, Vh, b, K, bc = problem()
    for mode in ("auto", "mfem"):
        csr.set_parmat_mode(mode)
        csr.clear_pattern_cache()
        mats = [matrix(pm, Vh, b, K, bc, seed) for seed in (1, 2)]
        A2 = mats[1]
        del mats
        gc.collect()
        nrm = amg_cg(A2)                   # a crash here is the failure this guards
        check("%s route: AMG setup and solve on the survivor" % mode, np.isfinite(nrm),
              "(|x| = %.4e)" % nrm)
    csr.set_parmat_mode("auto")


def test_operator_reset():
    """A solver given a new operator releases the old one and still solves."""
    if RANK == 0:
        print("operator resets on a device")
    pm, Vh, b, K, bc = problem()
    s = hp.KrylovSolver(COMM, "cg", "amg")
    s.parameters["rel_tolerance"] = 1e-10
    norms = []
    for seed in range(4):
        A = matrix(pm, Vh, b, K, bc, seed)
        s.set_operator(A)
        del A
        gc.collect()
        x = hp.ParVector(COMM, s.A.Height())
        rhs = hp.ParVector(COMM, s.A.Height())
        rhs.set(1.0)
        s.solve(x, rhs)
        norms.append(float(x.hypre.Norml2()))
    check("four operators in turn, each solved after the previous was released",
          all(np.isfinite(norms)) and len(set(np.round(norms, 6))) == 4,
          "(|x| = %s)" % ", ".join("%.4e" % v for v in norms))


def test_routes_agree_on_device():
    """The true-dof route and MFEM's triple product give the same device matrix."""
    if RANK == 0:
        print("routes agree with hypre on a device")
    pm, Vh, b, K, bc = problem()
    v = Vh[0].vector()
    hp.parRandom.set_seed(5)
    hp.parRandom.normal(1.0, v)
    ys = {}
    for mode in ("auto", "mfem"):
        csr.set_parmat_mode(mode)
        csr.clear_pattern_cache()
        A = matrix(pm, Vh, b, K, bc, 3)
        y = hp.ParVector(COMM, A.Height())
        A.Mult(v.hypre, y.hypre)
        ys[mode] = y
        del A
    csr.set_parmat_mode("auto")
    ref = float(ys["mfem"].hypre.Norml2())
    ys["auto"].hypre.Add(-1.0, ys["mfem"].hypre)
    err = float(ys["auto"].hypre.Norml2()) / max(ref, 1e-300)
    check("A v agrees between routes", err < 1e-13, "(rel %.2e)" % err)


def test_duplicate_adjoint_space():
    """State and adjoint given as two equal-but-distinct space objects.

    hIPPYlib's examples build the two separately, and on the host that is harmless.  With
    MFEM and hypre on a device it is not: the two spaces' matrices share a device mirror,
    BoomerAMG's setup fails with "Error code: 12" and the solve returns NaN, with nothing
    in the message to say why.  ``PDEVariationalProblem`` therefore aliases equivalent
    spaces, and this checks both that it did and that the answer is the one space gives.
    """
    if RANK == 0:
        print("state and adjoint as two equal spaces")
    import jax.numpy as jnp

    from hippymfem.fem.spaces import FunctionSpace

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.HEXAHEDRON))
    Vu, Vm = FunctionSpace.H1(pm, 2), FunctionSpace.H1(pm, 1)

    def varf(u, m, p, x):
        return jnp.exp(m.val) * hp.inner(u.grad, p.grad)

    m = Vm.project(lambda x: 0.25 * np.ones_like(x[0]))
    out = []
    for spaces in ([Vu, Vm, Vu], [Vu, Vm, FunctionSpace.H1(pm, 2)]):
        bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
        pde = hp.PDEVariationalProblem(spaces, varf, bc, bc.homogeneous(), is_fwd_linear=True)
        u = pde.generate_state()
        pde.solveFwd(u, [u, m, None])
        out.append((pde.Vh[ADJOINT] is pde.Vh[STATE], float(u.norm("l2"))))
        del pde
    aliased = out[1][0]
    rel = abs(out[0][1] - out[1][1]) / max(out[0][1], 1e-300)
    check("a duplicate adjoint space is aliased and solves the same", aliased and rel < 1e-12,
          "(aliased %s, relative difference %.1e)" % (aliased, rel))


def test_kept_accumulator():
    """A kept accumulator assembles exactly what a fresh one does.

    The ``nnz``-sized accumulator of a fused assembly is its largest allocation, and at
    a few million elements a rank it is most of JAX's arena, which is why it is borrowed
    back and zeroed in place instead of allocated per assembly.  Reuse must leave the
    assembled values where they were: the device scatter is atomic, so repeated
    assemblies agree to round-off rather than bit for bit, and that is the tolerance
    checked here.  The chunk is pinned small so that the *fused* route runs at all (an
    unsplit batch takes the glued one, which has no accumulator).
    """
    if RANK == 0:
        print("the accumulator a fused assembly reuses")
    from hippymfem.fem import kernel as kern
    from hippymfem.fem import pattern as pat

    pm, Vh, b, K, bc = problem()
    v = Vh[0].vector()
    hp.parRandom.set_seed(11)
    hp.parRandom.normal(1.0, v)

    def acted(seed=3):
        """``A x`` for the assembled A, the values themselves up to one matvec.

        Assembled through the thunk, which is the route that scatters chunk by chunk
        into an accumulator; the eager one glues the element matrices instead.
        """
        loc = locals_at(Vh, seed=seed)
        A = asm.assemble_matrix(
            Vh[0], Vh[0], b.groups,
            K.element_matrices_or_chunks(ADJOINT, STATE, loc), pm.GetNE(),
            test_ess=bc.ess_tdof)
        y = hp.ParVector(COMM, A.Height())
        A.Mult(v.hypre, y.hypre)
        out = np.array(y.array, copy=True)
        del A
        return out

    keep, share, chunk = pat.FUSED_KEEP, pat.FUSED_KEEP_SHARE, kern.ELEMENT_CHUNK
    pat.clear_accumulators()
    try:
        # so that every rank's batch splits (216 elements on 4 ranks leave 54 a rank)
        kern.ELEMENT_CHUNK = max(8, pm.GetNE() // 3)
        for gk in getattr(K, "group_kernels", ()):
            gk._chunk.clear()          # a remembered plan wins over ELEMENT_CHUNK
        pat.FUSED_KEEP = False
        ref = acted()
        pat.FUSED_KEEP, pat.FUSED_KEEP_SHARE = True, 0.0     # keep every size
        got = [acted() for _ in range(3)]
        reused = any(bufs for bufs in pat._ACC_FREE.values())
    finally:
        pat.FUSED_KEEP, pat.FUSED_KEEP_SHARE = keep, share
        kern.ELEMENT_CHUNK = chunk
        pat.clear_accumulators()
    scale = max(float(np.abs(ref).max()), 1e-300)
    worst = max(float(np.abs(ref - g).max()) / scale for g in got)
    worst = COMM.allreduce(worst, op=MPI.MAX)
    check("assemblies on a kept accumulator agree with a fresh one",
          reused and worst < 1e-12,
          "(%d values, reuse %s, worst rel %.1e)"
          % (ref.size, "on" if reused else "off", worst))


def test_reductions_after_matvec():
    """``inner``/``norm`` right after a hypre matvec must see the device result.

    ``prior.cost`` is ``R.mult`` then ``inner``.  A reduction that reads the host copy
    the vector was created with returns exactly 0.0, so J silently becomes the misfit
    alone while the iterates still agree with the host.  MFEM's own inner product is
    the device-aware reference.
    """
    if RANK == 0:
        print("reductions after a device matvec")
    pm, Vh, b, K, bc = problem()
    prior = hp.BiLaplacianPrior(Vh[1], 0.1, 0.5, robin_bc=True, solver_type="krylov")
    m = prior.mean.copy()
    m.set(1.0)
    d = m.copy().axpy(-1.0, prior.mean)
    Rd = d.duplicate()
    prior.R.mult(d, Rd)
    ref = 0.5 * float(mfem.InnerProduct(Rd.hypre, d.hypre))
    got = prior.cost(m)
    check("prior cost equals MFEM's inner product after the matvec",
          ref > 0 and abs(got - ref) <= 1e-12 * ref, "(%.10e vs %.10e)" % (got, ref))
    # ``Norml2`` on a HypreParVector is MFEM's *local* norm; the global one is the
    # inner product, which is collective.
    ref = float(mfem.InnerProduct(Rd.hypre, Rd.hypre)) ** 0.5
    check("norm after a matvec is the device result",
          abs(Rd.norm("l2") - ref) <= 1e-12 * ref, "(%.6e vs %.6e)" % (Rd.norm("l2"), ref))


def test_multivector_sees_device_writes():
    """``MultiVector.dot`` must see columns hypre wrote on the device.

    The columns alias rows of one host array.  If ``dot`` reads that array without a
    sync, ``U^T BU`` after ``MatMvMult(R, U, BU)`` comes out as zeros: the
    eigensolver's R-orthonormality check then reads 1.0 and the Laplace posterior's
    trace correction exactly 0.0, while the eigenpairs themselves are right.  The
    reference is the per-vector inner product, which syncs.
    """
    if RANK == 0:
        print("MultiVector reductions after device matvecs")
    pm, Vh, b, K, bc = problem()
    prior = hp.BiLaplacianPrior(Vh[1], 0.1, 0.5, robin_bc=True, solver_type="krylov")
    U = hp.MultiVector(prior.mean, 6)
    hp.parRandom.set_seed(4)
    hp.parRandom.normal_multivector(1.0, U)
    BU = hp.MultiVector(U[0], U.nvec())
    hp.MatMvMult(prior.R, U, BU)
    G = U.dot_mv(BU)
    ref = np.array([[U[i].inner(BU[j]) for j in range(U.nvec())] for i in range(U.nvec())])
    err = float(np.abs(G - ref).max() / max(np.abs(ref).max(), 1e-300))
    check("U^T (R U) from dot equals the per-vector inner products", err < 1e-13 and np.abs(ref).max() > 0,
          "(rel %.1e, |ref| %.3e)" % (err, np.abs(ref).max()))
    nrm = BU.norm("l2")
    ref_n = np.array([BU[j].norm("l2") for j in range(BU.nvec())])
    check("column norms see the device result", np.allclose(nrm, ref_n, rtol=1e-13, atol=0) and ref_n.max() > 0,
          "(max rel %.1e)" % (np.abs(nrm - ref_n).max() / max(ref_n.max(), 1e-300)))
    # a copy reads the backing array too: it copied zeros for columns written on
    # the device (a Taylor sketch copied that way made every eigenvalue zero)
    BU2 = hp.MultiVector(prior.mean, 6)
    hp.MatMvMult(prior.R, U, BU2)
    C = hp.MultiVector(BU2)
    ref_c = np.array([BU2[j].norm("l2") for j in range(BU2.nvec())])
    got_c = np.array([C[j].norm("l2") for j in range(C.nvec())])
    check("a copy of device-written columns has their values",
          np.allclose(got_c, ref_c, rtol=1e-13, atol=0) and ref_c.max() > 0,
          "(copy norms %.3e .. %.3e, source %.3e .. %.3e)"
          % (got_c.min(), got_c.max(), ref_c.min(), ref_c.max()))


def test_device_memory_flat():
    """Card memory must not grow across operator resets (the leak described above)."""
    if RANK == 0:
        print("device memory across operator resets")
    pm, Vh, b, K, bc = problem(10)
    s = hp.KrylovSolver(COMM, "cg", "amg")
    x = None
    for seed in range(3):                  # warm up hypre's pool and JAX's allocator
        s.set_operator(matrix(pm, Vh, b, K, bc, seed))
        x = hp.ParVector(COMM, s.A.Height())
        rhs = hp.ParVector(COMM, s.A.Height())
        rhs.set(1.0)
        s.solve(x, rhs)
    gc.collect()
    m0 = card_mib()
    for seed in range(3, 15):
        s.set_operator(matrix(pm, Vh, b, K, bc, seed))
        s.solve(x, rhs)
    gc.collect()
    m1 = card_mib()
    if m0 is None or m1 is None:
        check("card memory readable", False, "(nvidia-smi unavailable)")
        return
    check("twelve operator resets do not grow card memory", m1 - m0 < 64.0,
          "(%+.0f MiB, from %.0f MiB)" % (m1 - m0, m0))


def test_parmat_block_route():
    """The two device constructors must give the same matrix.

    With hypre on a device the true-dof route hands MFEM one row-major CSR with
    global columns and lets it re-derive hypre's diagonal and off-diagonal blocks,
    although the pattern has both already: that re-derivation is a visible share of
    every assembly on a large mesh.  ``HIPPYMFEM_PARMAT_DEVICE=block`` builds the two blocks
    directly instead.  It *aliases* the arrays it is given, where the copying
    constructor shares nothing, so what is checked is not only that the values agree
    but that matrices of one pattern stay independent: several are built, kept, used
    and released in an order that would expose a shared device mirror.
    """
    if RANK == 0:
        print("the block constructor agrees with the copying one on a device")
    from hippymfem.fem import tdofassemble as td

    pm, Vh, b, K, bc = problem()
    csr.set_parmat_mode("tdof")
    SEEDS = (3, 11, 29)

    def norms(A):
        out = []
        for p in range(2):
            v = hp.ParVector(COMM, A.Width())
            hp.parRandom.set_seed(101 + p)
            hp.parRandom.normal(1.0, v)
            y = hp.ParVector(COMM, A.Height())
            A.Mult(v.hypre, y.hypre)
            out.append(float(y.hypre.Norml2()))
        return out

    def run(mode):
        was = td.set_device_parmat(mode)
        csr.clear_pattern_cache()
        try:
            live = [matrix(pm, Vh, b, K, bc, seed) for seed in SEEDS]
            out = [norms(A) for A in live]
            # Release one while its siblings are in use, then use them: a shared
            # registry entry would have taken their device mirrors with it.
            del live[1]
            out.append(norms(live[0]))
            out.append([amg_cg(live[-1])])
            del live
            return out
        finally:
            td.set_device_parmat(was)
            csr.clear_pattern_cache()
            csr.set_parmat_mode("auto")

    ref, got = run("copy"), run("block")
    err = max(abs(a - c) / max(abs(a), 1e-300)
              for ra, rc in zip(ref[:-1], got[:-1]) for a, c in zip(ra, rc))
    check("the two constructors give the same matrices", err < 1e-13, "(rel %.2e)" % err)
    check("the assemblies differ from each other",
          len({tuple(r) for r in ref[:len(SEEDS)]}) == len(SEEDS))
    # A solve is compared loosely: BoomerAMG picks its coarse grids with a random
    # number generator on the device, so two hierarchies of one matrix differ.
    cg_err = abs(ref[-1][0] - got[-1][0]) / max(abs(ref[-1][0]), 1e-300)
    check("a solve agrees between the two constructors", cg_err < 1e-4,
          "(rel %.2e)" % cg_err)


def test_empty_boundary_rank():
    """A boundary pattern some ranks have no element of, built with the block
    constructor on the device: no deadlock resolving the device, no shared
    registration of two empty accumulator views (MFEM's "Unknown pointer!"), and a
    matrix that survives its sibling's release."""
    if RANK == 0:
        print("a rank without boundary elements, on the device")
    from hippymfem.fem.boundary import (BoundaryKernel, assemble_boundary_matrix,
                                        get_boundary_batches)

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(2, 2, 8, mfem.Element.HEXAHEDRON))
    V = hp.FunctionSpace.H1(pm, 1)
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
    del M2
    gc.collect()
    check("empty-rank boundary pattern: face area on the device, sibling released",
          abs(area1 - 1.0) < 1e-12 and abs(area2 - 1.0) < 1e-12,
          "(1^T M 1 = %.15f, %.15f; boundary elements per rank %s)" % (area1, area2, ne))


def test_boundary_block_on_device():
    """A block with a boundary residual and essential dofs, assembled on the device
    by the unchunked route.

    ``ScatterPattern.data`` brings the accumulator to the host as JAX's read-only
    copy, and ``add_boundary_entries`` then writes into it: the boundary entries
    (``np.add.at`` does not check the flag) and the eliminated slots (an indexed
    assignment, which does).  The copy must come back writeable, the forward solve
    must go through, and the point assembled block by block must agree with the
    streamed pass."""
    if RANK == 0:
        print("boundary residual into the domain block, on the device")
    import numpy as np
    from hippymfem.fem import kernel as K
    from hippymfem.fem.csrassemble import plan_block
    from hippymfem.modeling.variables import STATE, PARAMETER, ADJOINT
    from .test_boundary import build_robin

    pm, Vu, Vm, pde, f, g = build_robin(kappa_of_m=True, dirichlet=True, n=4, order=2)
    m = Vm.vector()
    hp.parRandom.set_seed(21)
    hp.parRandom.normal(0.3, m)
    x = [Vu.vector(), m, Vu.vector()]
    loc = pde._locals(x) + pde._aux_locals()
    mats = pde.kernel.element_matrices(STATE, STATE, loc)
    on_device = not all(isinstance(E, np.ndarray) for E in mats)
    acc = plan_block(Vu, Vu, pde.batches.groups).target.data(mats)
    check("the device accumulator comes to the host writeable",
          on_device and isinstance(acc, np.ndarray) and acc.flags.writeable,
          "(element matrices on device: %s, writeable: %s)" % (on_device, acc.flags.writeable))
    pde.solveFwd(x[STATE], x)                 # _block(STATE, STATE) with the boundary
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
            y = (Vm if i == PARAMETER else Vu).vector()
            pde.apply_ij(i, j, d, y)
            out[(i, j)] = y
        return out

    plain = blocks()
    old_chunk = K.ELEMENT_CHUNK
    K.ELEMENT_CHUNK = 3

    def replan():
        # on a device the planned size is remembered per group (GroupKernel._plan),
        # ahead of the pin; forget it so the pin is what decides
        for gk in pde.kernel.group_kernels:
            gk._chunk.clear()
            gk._peak.clear()

    replan()
    try:
        streamed_used = pde._stream_shared_pass()
        chunked = blocks()
    finally:
        K.ELEMENT_CHUNK = old_chunk
        replan()
    worst = max(plain[k].copy().axpy(-1.0, chunked[k]).norm("l2") / max(plain[k].norm("l2"), 1e-300)
                for k in pairs)
    check("streamed and plain linearization points agree with a boundary residual on the device",
          bool(streamed_used) and worst < K.matrix_tolerance(1e-12),
          "(streamed %s, worst rel %.1e)" % (streamed_used, worst))


def test_coordinates_on_device():
    """``coordinates()`` and the nodal ``project`` agree with MFEM's callback route
    with hypre's vectors on the device.

    Both write node values through ``GetDataArray()``.  After the first
    ``GetTrueDofs`` the memory manager's device copy is the valid one, and a bare
    write to the host copy is ignored: at two or more ranks y and z would come back
    as x (``host_readwrite`` before the write, as elsewhere in the library)."""
    if RANK == 0:
        print("dof coordinates and nodal projection with hypre on the device")
    import numpy as np
    from hippymfem.fem.spaces import _ScalarPy

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.HEXAHEDRON))
    worst = 0.0
    for order in (1, 2):
        V = hp.FunctionSpace.H1(pm, order)
        X = V.coordinates()
        ref = np.zeros_like(X)
        gf = mfem.ParGridFunction(V.fes)
        tmp = V.vector()
        for d in range(3):
            gf.ProjectCoefficient(_ScalarPy(lambda x, d=d: x[d]))
            gf.GetTrueDofs(tmp.hypre)
            ref[:, d] = tmp.array
        worst = max(worst, float(np.abs(X - ref).max()))
        u = V.project(lambda z: 1.0 + 2.0 * z[..., 0] - 3.0 * z[..., 1] + 0.5 * z[..., 2])
        expect = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 2]
        worst = max(worst, float(np.abs(u.array - expect).max()))
    worst = COMM.allreduce(worst, op=MPI.MAX)
    check("dof coordinates and nodal projection on the device", worst < 1e-12,
          "(max abs diff %.1e)" % worst)


def test_hypre_spmv_kernel():
    """hypre's matrix-vector kernel: the vendor's on one rank, hypre's own on several
    (CUDA builds), and the same product and the same solve with either."""
    from hippymfem.common import mfemconfig as cfg

    chosen = cfg.set_hypre_spmv("auto", COMM)
    want = "hypre" if (COMM.size > 1 and mfem_gpu_backend() == "cuda") else "vendor"
    check("kernel chosen for %d rank(s)" % COMM.size, chosen == want and cfg.HYPRE_SPMV == want,
          "(%s)" % chosen)
    pm, Vh, b, K, bc = problem(6)
    A = matrix(pm, Vh, b, K, bc, seed=21)
    x = hp.ParVector(COMM, A.Width())
    hp.parRandom.set_seed(21)
    hp.parRandom.normal(1.0, x)
    out, sol = {}, {}
    for kernel in ("vendor", "hypre"):
        cfg.set_hypre_spmv(kernel, COMM)
        y = hp.ParVector(COMM, A.Height())
        A.Mult(x.hypre, y.hypre)
        out[kernel] = y
        sol[kernel] = amg_cg(A, iters=30)
    cfg.set_hypre_spmv("auto", COMM)
    d = out["vendor"].copy()
    d.axpy(-1.0, out["hypre"])
    rel = d.norm("l2") / max(out["vendor"].norm("l2"), 1e-300)
    check("the two kernels give the same product", rel < 1e-13, "(relative difference %.1e)" % rel)
    rel = abs(sol["vendor"] - sol["hypre"]) / max(abs(sol["vendor"]), 1e-300)
    check("the two kernels give the same AMG-CG solve", rel < 1e-9, "(relative difference %.1e)" % rel)
    try:
        cfg.set_hypre_spmv("fastest", COMM)
        bad = False
    except ValueError:
        bad = True
    check("an unknown kernel name is refused", bad)


def test_hypre_pool():
    """hypre's device memory through the recycling pool: the same solve, blocks served
    from the pool, never more held than allowed, nothing held after a trim."""
    from hippymfem.common import mfemconfig as cfg

    if mfem_gpu_backend() != "cuda":
        check("hypre device pool (NVIDIA builds only)", True, "(skipped)")
        return
    pm, Vh, b, K, bc = problem(6)
    A = matrix(pm, Vh, b, K, bc, seed=22)
    ref = amg_cg(A, iters=30)
    had = cfg.HYPRE_POOL is not None and cfg.HYPRE_POOL.installed
    caps = ((cfg.HYPRE_POOL.scoped, cfg.HYPRE_POOL.limit, cfg.HYPRE_POOL.keep,
             cfg.HYPRE_POOL.max_cached, cfg.HYPRE_POOL.max_block) if had else None)
    pool = cfg.set_hypre_pool(32.0)
    check("pool installed", pool is not None and pool.installed)
    if pool is None:
        return
    first = amg_cg(A, iters=30)             # a setup, its hierarchy freed on return
    served = pool.from_pool
    second = amg_cg(A, iters=30)            # the same setup again, from the pool
    ok = (abs(first - ref) <= 1e-12 * abs(ref)) and (abs(second - ref) <= 1e-12 * abs(ref))
    check("the same solve through the pool", ok, "(%.15e, %.15e, %.15e)" % (ref, first, second))
    check("the second setup is served from the pool", pool.from_pool > served,
          "(%d of %d requests, %d driver allocations)"
          % (pool.from_pool, pool.requests, pool.driver_allocs))
    check("the pool holds no more than it may", pool.peak_cached <= 32 * 2 ** 20,
          "(peak %.1f MiB)" % (pool.peak_cached / 2 ** 20))
    gc.collect()
    pool.trim()
    check("nothing held after a trim", pool.cached == 0)
    third = amg_cg(A, iters=30)
    check("the same solve after a trim", abs(third - ref) <= 1e-12 * abs(ref))
    if had:
        pool.trim()
        pool.scoped, pool.limit, pool.keep, pool.max_cached, pool.max_block = caps
    else:
        pool.uninstall()
        fourth = amg_cg(A, iters=30)
        check("the same solve with hypre's own allocator back",
              abs(fourth - ref) <= 1e-12 * abs(ref) and not pool.installed)


def test_hypre_pool_setup_scope():
    """The default pool: it recycles hypre's arrays while a solver sets its
    preconditioner up, and once the first solve is over it holds no more than it may
    keep between setups."""
    from hippymfem.common import mfemconfig as cfg

    if mfem_gpu_backend() != "cuda":
        check("hypre device pool (NVIDIA builds only)", True, "(skipped)")
        return
    pool = cfg.HYPRE_POOL
    if pool is None or not pool.installed or not pool.scoped:
        check("the default pool is the one with a limit of its own during a setup", True,
              "(skipped: HIPPYMFEM_HYPRE_POOL=%s)" % os.environ.get("HIPPYMFEM_HYPRE_POOL", "auto"))
        return
    pm, Vh, b, K, bc = problem(6)
    A = matrix(pm, Vh, b, K, bc, seed=24)
    ref = amg_cg(A, iters=200)                 # hypre's own allocations: the pool is closed
    served, requests = pool.from_pool, pool.requests
    s = hp.KrylovSolver(COMM, "cg", "amg")
    s.parameters["rel_tolerance"] = 1e-12
    s.parameters["max_iter"] = 200
    s.set_operator(A)
    open_now = pool.max_cached == pool.limit
    x, rhs = hp.ParVector(COMM, A.Height()), hp.ParVector(COMM, A.Height())
    rhs.set(1.0)
    s.solve(x, rhs)
    got = float(x.hypre.Norml2())
    check("the pool is open between set_operator and the first solve", open_now)
    check("the setup was served from the pool", pool.from_pool > served,
          "(%d of %d requests)" % (pool.from_pool - served, pool.requests - requests))
    check("after the first solve the pool holds no more than it may keep",
          pool.cached <= pool.max_cached <= pool.keep
          and pool.max_cached <= pool.KEEP_SHARE * pool.peak_in_use,
          "(%d of %d bytes; the most in use %d)" % (pool.cached, pool.keep, pool.peak_in_use))
    check("the same solve with the pool", abs(got - ref) <= 1e-8 * abs(ref),
          "(%.12e, %.12e)" % (ref, got))
    # a second solver on a new matrix of the same size: the hierarchy the first one
    # gives up and the blocks kept serve its setup
    del s
    gc.collect()
    driver = pool.driver_allocs
    served = pool.from_pool
    s2 = hp.KrylovSolver(COMM, "cg", "amg")
    s2.parameters["rel_tolerance"] = 1e-12
    s2.parameters["max_iter"] = 200
    s2.set_operator(A)
    x.zero()
    s2.solve(x, rhs)
    got2 = float(x.hypre.Norml2())
    check("a second setup is served from the pool and gives the same solve",
          pool.from_pool > served and abs(got2 - ref) <= 1e-8 * abs(ref),
          "(%d served, %d from the driver)" % (pool.from_pool - served, pool.driver_allocs - driver))
    freed = cfg.hypre_pool_trim()
    check("a trim returns what was kept", pool.cached == 0, "(%d bytes)" % freed)


def test_identity_route_on_device():
    """A block with identity prolongations (one rank) through the true-dof route and
    through the copying constructor: the same matrix.  Entry for entry up to round-off
    only, because the scatter on a device adds an entry's contributions in an order
    that is not fixed (on the host the two are identical)."""
    from hippymfem.common.linalg import hypre_to_scipy

    old = csr.TDOF_IDENTITY
    mats, norms = {}, {}
    try:
        for mode in ("0", "1"):
            csr.TDOF_IDENTITY = mode
            pm, Vh, b, K, bc = problem(5)          # new spaces: nothing cached is shared
            A = matrix(pm, Vh, b, K, bc, seed=23)
            mats[mode] = hypre_to_scipy(A).tocsr()
            norms[mode] = amg_cg(A, iters=30)
            del A
    finally:
        csr.TDOF_IDENTITY = old
    same_shape = mats["0"].shape == mats["1"].shape and mats["0"].nnz == mats["1"].nnz
    worst = float(abs(mats["0"] - mats["1"]).max()) if same_shape and mats["0"].nnz else 0.0
    scale = float(abs(mats["0"]).max()) if mats["0"].nnz else 1.0
    check("identity prolongation: true-dof route equals the copying constructor",
          same_shape and worst <= 1e-13 * scale,
          "(max abs difference %.1e, largest entry %.1e)" % (worst, scale))
    rel = abs(norms["0"] - norms["1"]) / max(abs(norms["0"]), 1e-300)
    check("identity prolongation: the same AMG-CG solve", rel < 1e-9,
          "(relative difference %.1e)" % rel)


def test_matrix_stays_on_device():
    """A matrix finished on the device, its values copied from JAX's memory into
    hypre's two blocks there, against the one whose values pass through the host:
    the same matrix to round-off (the scatter's order is not fixed on a device), read
    back to the host from the device-built one, and the same solve."""
    from hippymfem.common import devicebridge as bridge
    from hippymfem.common.linalg import hypre_to_scipy

    old = bridge.set_device_bridge("auto")
    mats, norms, used = {}, {}, {}
    try:
        usable, why = bridge.available(), bridge.why_not()
        if not usable:
            check("device bridge (kernels and hypre on one GPU)", True, "(skipped: %s)" % why)
            return
        for mode in ("0", "auto"):
            bridge.set_device_bridge(mode)
            pm, Vh, b, K, bc = problem(5)          # new spaces: nothing cached is shared
            before = bridge.stats()[0]
            A = matrix(pm, Vh, b, K, bc, seed=31)
            used[mode] = bridge.stats()[0] - before
            norms[mode] = amg_cg(A, iters=30)
            mats[mode] = hypre_to_scipy(A).tocsr()
            del A
    finally:
        bridge.set_device_bridge(old)
    check("the values are copied on the device with the bridge and not without",
          used["auto"] >= 2 and used["0"] == 0, "(%d and %d copies)" % (used["auto"], used["0"]))
    same = mats["0"].shape == mats["auto"].shape and mats["0"].nnz == mats["auto"].nnz
    worst = float(abs(mats["0"] - mats["auto"]).max()) if same and mats["0"].nnz else 0.0
    scale = float(abs(mats["0"]).max()) if mats["0"].nnz else 1.0
    check("a matrix finished on the device equals the one finished on the host",
          same and worst <= 1e-13 * scale,
          "(max abs difference %.1e, largest entry %.1e)" % (worst, scale))
    rel = abs(norms["0"] - norms["auto"]) / max(abs(norms["0"]), 1e-300)
    check("device-finished matrix: the same AMG-CG solve", rel < 1e-9,
          "(relative difference %.1e)" % rel)


def test_vectors_on_device():
    """ParVector arithmetic where the vector is: every update, copy and reduction of a
    vector hypre left on the device, done by MFEM there, against numpy on the host."""
    from hippymfem.common import parvector as pv

    pm, Vh, b, K, bc = problem(5)
    A = matrix(pm, Vh, b, K, bc, seed=37)
    bcz = hp.DirichletBC(Vh[0], lambda x: 1.0 + x[0], "all")
    old = pv.set_device_vectors("auto")
    res = {}
    try:
        for mode in ("0", "auto"):
            pv.set_device_vectors(mode)
            vs = []
            for k in range(3):
                v = Vh[0].vector()
                v.array[:] = np.random.default_rng(1000 * RANK + k).normal(size=v.local_size)
                vs.append(v)
            x, y, z = vs
            w = Vh[0].vector()
            A.Mult(x.hypre, w.hypre)                 # w is with hypre now
            flagged = w._dev
            w.axpy(0.3, y)
            w.scale(-1.7)
            w.aypx(0.25, x)
            w.axpby(0.5, 2.0, z)
            c = w.copy()
            c.assign(w)
            bcz.zero(c)
            bcz.apply(w)
            d = c.copy()
            d.zero()
            res[mode] = (flagged, w.inner(x), w.norm("l2"), c.norm("l2"), w.norm("linf"),
                         d.norm("l2"), float(np.abs(w.array).sum()))
    finally:
        pv.set_device_vectors(old)
    check("a vector handed to hypre is flagged for the device only when that is on",
          res["auto"][0] and not res["0"][0])
    err = max(abs(a - b) / max(abs(a), 1e-300) for a, b in zip(res["0"][1:], res["auto"][1:]))
    check("vector arithmetic on the device agrees with numpy on the host", err < 1e-12,
          "(largest relative difference %.1e)" % err)


def test_observation_on_device():
    """The pointwise observation operator as a hypre matrix on the device against the
    numpy route, for targets inside elements and on the faces ranks share."""
    from hippymfem.common import parvector as pv

    pm, Vh, b, K, bc = problem(6)
    rng = np.random.default_rng(5)
    targets = np.column_stack([rng.uniform(0.05, 0.95, 40) for _ in range(3)])
    targets[:4] = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.25, 0.0]]
    old = pv.set_device_vectors("auto")
    res, used = {}, {}
    try:
        for mode in ("0", "auto"):
            pv.set_device_vectors(mode)
            B = hp.assemblePointwiseObservation(Vh[0], targets)
            u = Vh[0].vector()
            u.array[:] = np.random.default_rng(77 + RANK).normal(size=u.local_size)
            d = B.createVecLeft()
            B.mult(u, d)
            back = Vh[0].vector()
            B.multTranspose(d, back)
            res[mode] = (B.gather(d), back.norm("l2"), back.inner(u))
            used[mode] = bool(B._Bh)
    finally:
        pv.set_device_vectors(old)
    check("the observation operator is a hypre matrix only with the vectors on the device",
          used["auto"] and not used["0"])
    e1 = float(np.abs(res["0"][0] - res["auto"][0]).max() / max(np.abs(res["0"][0]).max(), 1e-300))
    e2 = max(abs(res["0"][k] - res["auto"][k]) / max(abs(res["0"][k]), 1e-300) for k in (1, 2))
    check("B u and B^T d agree between the device and the host route", max(e1, e2) < 1e-12,
          "(relative differences %.1e and %.1e)" % (e1, e2))


def test_residual_on_device():
    """An assembled vector that never visits the host (the summed element vectors go
    from JAX's memory to MFEM's on the device, and the kernels read the dof values
    from MFEM's device memory) against the one assembled through numpy."""
    import jax.numpy as jnp

    from hippymfem.common import devicebridge as bridge

    old = bridge.set_device_bridge("auto")
    res = {}
    try:
        if not bridge.available():
            check("vectors over the device bridge", True, "(skipped: %s)" % bridge.why_not())
            return
        for mode in ("0", "auto"):
            bridge.set_device_bridge(mode)
            pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(5, 5, 5, mfem.Element.HEXAHEDRON))
            Vu, Vm = hp.FunctionSpace.H1(pm, 2), hp.FunctionSpace.H1(pm, 1)
            bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
            pde = hp.PDEVariationalProblem(
                [Vu, Vm, Vu], lambda u, m, p, x: jnp.exp(m.val) * hp.inner(u.grad, p.grad),
                bc, bc.homogeneous(), is_fwd_linear=True)
            x = [pde.generate_state(), Vm.vector(), pde.generate_state()]
            for k, v in enumerate(x):
                v.array[:] = np.random.default_rng(50 + 7 * k + RANK).normal(size=v.local_size)
            for v in (x[0], x[2]):
                A0 = v.copy()                              # hand them to hypre once
                A0.hypre
            r = pde._residual(x, ADJOINT, ess=pde.bc0.ess)
            g = pde._residual(x, 1)
            pde.solveFwd(x[0], x)
            res[mode] = (r.norm("l2"), g.norm("l2"), r.inner(x[2]), x[0].norm("l2"))
    finally:
        bridge.set_device_bridge(old)
    err = max(abs(a - b) / max(abs(a), 1e-300) for a, b in zip(res["0"], res["auto"]))
    check("residual, gradient and forward solve agree with and without the bridge",
          err < 1e-10, "(largest relative difference %.1e)" % err)


def test_streamed_geometry_pinned():
    """A geometry the chunk loops stream, copied by the runtime from registered host
    memory (``kernel.PINNED_STREAM``), gives the element arrays of the geometry kept
    on the device and of the one JAX moves, bit for bit, and only the arrays the
    kernel reads are copied (this density does not read ``x``)."""
    from hippymfem.common import devicebridge as bridge
    from hippymfem.fem import kernel as km

    if not bridge.available():
        check("streamed geometry over the bridge", True, "(skipped: %s)" % bridge.why_not())
        return
    pm, Vh, b, K, bc = problem(6)
    loc = locals_at(Vh, seed=11)
    old = km.ELEMENT_CHUNK, km.GEOMETRY_STREAM_FRACTION, km.PINNED_STREAM

    def reset(chunk, frac, pinned):
        km.ELEMENT_CHUNK, km.GEOMETRY_STREAM_FRACTION, km.PINNED_STREAM = chunk, frac, pinned
        for gk in K.group_kernels:
            gk._cache.clear()
            gk._chunk.clear()
            gk._dev.clear()

    out, moved = {}, {}
    try:
        for label, frac, pinned in (("cached", 0.0, True), ("by JAX", 1e-12, False),
                                    ("pinned", 1e-12, True)):
            reset(37, frac, pinned)
            before = bridge.stats()[1]
            out[label] = ([np.asarray(a) for a in K.element_matrices(ADJOINT, STATE, loc)]
                          + [np.asarray(a) for a in K.element_vectors(ADJOINT, loc)])
            moved[label] = bridge.stats()[1] - before
    finally:
        reset(*old)
    diff = max(max(float(np.abs(x - y).max()) for x, y in zip(out["cached"], out[k]))
               for k in ("by JAX", "pinned"))
    groups = [gk.group for gk in K.group_kernels]
    pinned = all(getattr(g, "_pinned_geometry", None) for g in groups)
    read = sum(g.Jinv.nbytes + g.wdet.nbytes for g in groups)
    passes = moved["pinned"] / max(read, 1)
    check("a geometry streamed from registered memory gives the same element arrays",
          diff == 0.0 and pinned and moved["by JAX"] == 0 and moved["cached"] == 0,
          "(largest difference %.1e; registered: %s)" % (diff, pinned))
    check("and only what the kernel reads is copied",
          passes >= 1 and passes == int(passes),
          "(%.2f passes over Jinv and wdet; x is not read)" % passes)


def test_single_precision_solves():
    """The solves in a single-precision hypre (``HIPPYMFEM_HYPRE_SINGLE``) against the
    double-precision ones; skipped when no library is named."""
    from hippymfem.test import single_case

    single_case.run(check, COMM)


if __name__ == "__main__":
    test_sibling_release()
    test_operator_reset()
    test_routes_agree_on_device()
    test_parmat_block_route()
    test_empty_boundary_rank()
    test_boundary_block_on_device()
    test_coordinates_on_device()
    test_duplicate_adjoint_space()
    test_kept_accumulator()
    test_reductions_after_matvec()
    test_multivector_sees_device_writes()
    test_device_memory_flat()
    test_hypre_spmv_kernel()
    test_identity_route_on_device()
    test_matrix_stays_on_device()
    test_vectors_on_device()
    test_observation_on_device()
    test_residual_on_device()
    test_streamed_geometry_pinned()
    test_hypre_pool_setup_scope()
    test_hypre_pool()
    test_single_precision_solves()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    sys.exit(1 if FAILS else 0)

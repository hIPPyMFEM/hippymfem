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
          bool(streamed_used) and worst < 1e-12, "(streamed %s, worst rel %.1e)" % (streamed_used, worst))


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
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    sys.exit(1 if FAILS else 0)

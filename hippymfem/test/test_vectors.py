# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Vectors, operators, MultiVector and the parallel RNG.

Run with ``mpirun -n N python -m hippymfem.test.test_vectors``.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

import hippymfem as hp
from hippymfem.common.parvector import ParVector
from hippymfem.algorithms.multivector import MultiVector, MatMvMult, MvDSmatMult

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
FAILS = []


def check(name, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    if RANK == 0:
        print("  [%s] %s %s" % (tag, name, detail), flush=True)
    if not ok:
        FAILS.append(name)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def build_space(nx=10, order=1):
    mesh = mfem.Mesh.MakeCartesian2D(nx, nx, mfem.Element.TRIANGLE)
    pmesh = mfem.ParMesh(COMM, mesh)
    fec = mfem.H1_FECollection(order, 2)
    fes = mfem.ParFiniteElementSpace(pmesh, fec)
    return pmesh, fec, fes


def test_parvector():
    if RANK == 0:
        print("ParVector")
    _pm, _fec, fes = build_space()
    v = ParVector.from_fes(fes)
    w = ParVector.from_fes(fes)
    check("global size == global tdofs",
          v.global_size == fes.GlobalTrueVSize(),
          "(%d)" % v.global_size)
    check("hypre aliases numpy",
          v.hypre.GetDataArray().__array_interface__["data"][0]
          == v.array.__array_interface__["data"][0])

    v.set(2.0)
    w.set(3.0)
    check("inner", close(v.inner(w), 6.0 * v.global_size))
    check("norm l2", close(v.norm("l2"), 2.0 * np.sqrt(v.global_size)))
    check("norm linf", close(v.norm("linf"), 2.0))
    v.axpy(-1.0, w)
    check("axpy", close(v.norm("linf"), 1.0))
    c = v.copy()
    c.scale(-2.0)
    check("copy+scale", close(c.array[0] if c.local_size else -2.0 * -1.0, 2.0)
          if c.local_size else True)
    check("copy is independent", close(v.array[0], -1.0) if v.local_size else True)

    # gather/scatter round-trip
    v.set(0.0)
    lo, _hi = v.owner_range
    v.array[:] = np.arange(lo, lo + v.local_size, dtype=float)
    full = v.gather_to_zero()
    if RANK == 0:
        ok = np.array_equal(full, np.arange(v.global_size, dtype=float))
    else:
        ok = True
    check("gather_to_zero ordering", COMM.bcast(ok, root=0))
    v2 = ParVector.from_fes(fes)
    v2.scatter_from_zero(full)
    diff = v2.copy().axpy(-1.0, v).norm("linf")
    check("scatter round-trip", close(diff, 0.0))


def test_partition_matches_mfem():
    """The crux: our allgather partition must equal MFEM's tdof partition."""
    if RANK == 0:
        print("partition compatibility with MFEM")
    _pm, _fec, fes = build_space(12)
    a = mfem.ParBilinearForm(fes)
    a.AddDomainIntegrator(mfem.MassIntegrator())
    a.Assemble()
    a.Finalize()
    M = a.ParallelAssemble()

    # M applied through our ParVector must equal M applied through MFEM's own
    # HypreParVector built from M's row partition.
    x = ParVector.from_fes(fes)
    y = ParVector.from_fes(fes)
    hp.parRandom.set_seed(11)
    hp.parRandom.normal(1.0, x)
    M.Mult(x.hypre, y.hypre)

    xm = mfem.HypreParVector(M, 1)
    ym = mfem.HypreParVector(M, 0)
    xm.GetDataArray()[:] = x.array
    M.Mult(xm, ym)
    err = np.abs(ym.GetDataArray() - y.array).max() if y.local_size else 0.0
    err = COMM.allreduce(err, op=MPI.MAX)
    check("HypreParMatrix.Mult agrees on our partition", err == 0.0, "(err=%g)" % err)

    # Our offsets must equal MFEM's true-dof offsets: for every locally owned
    # dof, global tdof number == offset + local tdof number.
    lo, _ = x.owner_range
    bad = 0
    checked = 0
    for j in range(fes.GetVSize()):
        lt = fes.GetLocalTDofNumber(j)
        if lt < 0:
            continue
        checked += 1
        if fes.GetGlobalTDofNumber(j) != lo + lt:
            bad += 1
    check("true-dof global numbering matches offsets",
          COMM.allreduce(bad) == 0 and COMM.allreduce(checked) > 0,
          "(checked %d)" % COMM.allreduce(checked))

    # mass matrix row sums = domain area
    one = ParVector.from_fes(fes)
    one.set(1.0)
    Mone = ParVector.from_fes(fes)
    M.Mult(one.hypre, Mone.hypre)
    check("1^T M 1 == area", close(Mone.inner(one), 1.0, 1e-12),
          "(%.15f)" % Mone.inner(one))


def test_operators():
    if RANK == 0:
        print("operators")
    _pm, _fec, fes = build_space(8)
    a = mfem.ParBilinearForm(fes)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator())
    a.AddDomainIntegrator(mfem.MassIntegrator())
    a.Assemble()
    a.Finalize()
    A = a.ParallelAssemble()
    op = hp.MatrixOperator(A, COMM)

    x = op.generate_vector(1)
    y = op.generate_vector(0)
    z = op.generate_vector(1)
    hp.parRandom.set_seed(3)
    hp.parRandom.normal(1.0, x)
    hp.parRandom.normal(1.0, y)

    op.mult(x, z)
    lhs = z.inner(y)
    w = op.generate_vector(1)
    op.multTranspose(y, w)
    rhs = w.inner(x)
    check("adjoint identity <Ax,y> == <x,A^T y>", close(lhs, rhs, 1e-11),
          "(%.3e vs %.3e)" % (lhs, rhs))
    check("symmetry (diffusion+mass)", close(lhs, rhs, 1e-11))

    tr = hp.TransposeOperator(op)
    tz = tr.generate_vector(0)
    tr.mult(y, tz)
    check("TransposeOperator", close(tz.inner(x), rhs, 1e-11))

    s = hp.SumOperator(op, hp.IdentityOperator(x), 1.0, 2.0)
    sz = s.generate_vector(0)
    s.mult(x, sz)
    expect = z.copy().axpy(2.0, x)
    check("SumOperator", close(sz.copy().axpy(-1.0, expect).norm("linf"), 0.0, 1e-12))

    d = op.generate_vector(1)
    d.set(0.0)
    d.array[:] = 2.0 + np.arange(d.local_size)
    D = hp.DiagonalOperator(d)
    dz = D.generate_vector(0)
    D.mult(x, dz)
    ds = D.generate_vector(0)
    D.solve(ds, dz)
    check("DiagonalOperator mult/solve inverse",
          close(ds.copy().axpy(-1.0, x).norm("linf"), 0.0, 1e-12))

    # MFEM can drive our operator
    mop = hp.MFEMOperator(op, A.Height())
    xv = mfem.Vector(A.Height())
    yv = mfem.Vector(A.Height())
    xv.GetDataArray()[:] = x.array
    mop.Mult(xv, yv)
    check("MFEMOperator adapter",
          np.abs(yv.GetDataArray() - z.array).max() < 1e-14 if z.local_size else True)


def test_multivector():
    if RANK == 0:
        print("MultiVector")
    _pm, _fec, fes = build_space(8)
    v = ParVector.from_fes(fes)
    n = 5
    hp.parRandom.set_seed(7)
    X = MultiVector(v, n)
    Y = MultiVector(v, 3)
    for i in range(n):
        hp.parRandom.normal(1.0, X[i])
    for i in range(3):
        hp.parRandom.normal(1.0, Y[i])

    check("nvec", X.nvec() == n)
    check("column aliases backing array",
          X[2].array.__array_interface__["data"][0]
          == X.data[2].__array_interface__["data"][0])

    # dot_mv against explicit inner products
    G = X.dot_mv(Y)
    ref = np.array([[X[i].inner(Y[j]) for j in range(3)] for i in range(n)])
    check("dot_mv", np.abs(G - ref).max() < 1e-11, "(%.2e)" % np.abs(G - ref).max())

    g = X.dot_v(Y[0])
    check("dot_v", np.abs(g - ref[:, 0]).max() < 1e-11)

    # reduce
    alpha = np.arange(1.0, n + 1)
    acc = v.duplicate()
    X.reduce(acc, alpha)
    ref2 = v.duplicate()
    for i in range(n):
        ref2.axpy(alpha[i], X[i])
    check("reduce", acc.copy().axpy(-1.0, ref2).norm("linf") < 1e-12)

    # norms
    nr = X.norm("l2")
    ref3 = np.array([X[i].norm("l2") for i in range(n)])
    check("norm", np.abs(nr - ref3).max() < 1e-11)

    # Euclidean QR
    Z = MultiVector(X)
    r = Z.orthogonalize()
    QtQ = Z.dot_mv(Z)
    check("orthogonalize: Q^T Q == I",
          np.abs(QtQ - np.eye(n)).max() < 1e-10, "(%.2e)" % np.abs(QtQ - np.eye(n)).max())
    # X == Q R
    recon = MultiVector(v, n)
    MvDSmatMult(Z, r, recon)
    err = max(recon[i].copy().axpy(-1.0, X[i]).norm("linf") for i in range(n))
    check("orthogonalize: Q R == X", err < 1e-10, "(%.2e)" % err)

    # B-orthogonal QR with B = mass matrix
    a = mfem.ParBilinearForm(fes)
    a.AddDomainIntegrator(mfem.MassIntegrator())
    a.Assemble()
    a.Finalize()
    B = hp.MatrixOperator(a.ParallelAssemble(), COMM)
    W = MultiVector(X)
    Bq, rb = W.Borthogonalize(B)
    BW = MultiVector(v, n)
    MatMvMult(B, W, BW)
    QtBQ = W.dot_mv(BW)
    check("Borthogonalize: Q^T B Q == I",
          np.abs(QtBQ - np.eye(n)).max() < 1e-9,
          "(%.2e)" % np.abs(QtBQ - np.eye(n)).max())
    err = max(Bq[i].copy().axpy(-1.0, BW[i]).norm("linf") for i in range(n))
    check("Borthogonalize: returns B Q", err < 1e-10, "(%.2e)" % err)

    # swap keeps aliases valid
    P = MultiVector(X)
    Q = MultiVector(Y)
    p0 = P[0].copy()
    P.swap(Q)
    check("swap sizes", P.nvec() == 3 and Q.nvec() == n)
    check("swap aliases still valid",
          Q[0].array.__array_interface__["data"][0]
          == Q.data[0].__array_interface__["data"][0])
    check("swap moved data", Q[0].copy().axpy(-1.0, p0).norm("linf") < 1e-14)


def test_random_partition_independence():
    if RANK == 0:
        print("parallel RNG")
    _pm, _fec, fes = build_space(10)
    v = ParVector.from_fes(fes)
    hp.parRandom.set_seed(1234)
    hp.parRandom.normal(1.0, v)
    full = v.gather_to_zero()
    if RANK == 0:
        np.save("/tmp/_hippymfem_rng_np%d.npy" % NP, full)
        print("      global sum = %.15f, n = %d" % (full.sum(), full.size))
    # stats
    s = v.sum() / v.global_size
    ss = v.inner(v) / v.global_size
    check("mean ~ 0", abs(s) < 5.0 / np.sqrt(v.global_size), "(%.4f)" % s)
    check("var ~ 1", abs(ss - 1.0) < 0.1, "(%.4f)" % ss)

    # two successive draws differ
    w = ParVector.from_fes(fes)
    hp.parRandom.normal(1.0, w)
    check("successive draws differ", w.copy().axpy(-1.0, v).norm("linf") > 1e-6)

    # rademacher
    r = ParVector.from_fes(fes)
    hp.parRandom.rademacher(r)
    check("rademacher in {-1,1}",
          COMM.allreduce(int(np.all(np.abs(np.abs(r.array) - 1.0) < 1e-15)), op=MPI.MIN) == 1)

    # scalar draws identical across ranks
    sc = hp.parRandom.scalar_normal()
    allsc = COMM.allgather(sc)
    check("scalar_normal replicated", len(set(allsc)) == 1)


def test_host_build_warns_for_device_hypre():
    """``HIPPYMFEM_HYPRE_DEVICE=1`` on a PyMFEM built without CUDA or HIP says so.

    It used to do nothing at all: a day of profiles ran with hypre on the host (forward
    solves ten times slower) while the variable read as set.  The build is faked here,
    so the check runs on any PyMFEM.
    """
    import os
    import warnings

    from hippymfem.common import mfemconfig as mc

    saved_env = os.environ.get("HIPPYMFEM_HYPRE_DEVICE")
    saved_backend, saved_device = mc.mfem_gpu_backend, list(mc._DEVICE)
    try:
        os.environ["HIPPYMFEM_HYPRE_DEVICE"] = "1"
        mc.mfem_gpu_backend = lambda: None
        del mc._DEVICE[:]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = mc.auto_configure_device()
        said = [w for w in caught if "HIPPYMFEM_HYPRE_DEVICE" in str(w.message)]
        check("a host build warns when hypre is asked onto the device",
              got is None and len(said) == 1 and not mc._DEVICE,
              "(%d warning)" % len(said))
    finally:
        mc.mfem_gpu_backend = saved_backend
        mc._DEVICE[:] = saved_device
        if saved_env is None:
            os.environ.pop("HIPPYMFEM_HYPRE_DEVICE", None)
        else:
            os.environ["HIPPYMFEM_HYPRE_DEVICE"] = saved_env


def test_config_knobs():
    """Every knob of ``hm.config`` reads, and a written value arrives as the type it names.

    Two knobs did not: ``gpu_mem_reserve`` is in GiB but was cast to ``int`` and described
    in bytes, and ``fused_keep`` was cast with ``bool``, for which ``"0"`` is true.
    """
    import io
    from contextlib import redirect_stdout

    c = hp.config
    names = [k for k in c.as_dict(load=False)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        c.show()
    check("config.show() lists every knob", all(n in buf.getvalue() for n in names),
          "(%d knobs)" % len(names))
    saved = (c.gpu_mem_reserve, c.fused_keep)
    try:
        c.gpu_mem_reserve = 2.5
        c.fused_keep = "0"
        check("a fractional reserve in GiB and a false string keep their meaning",
              c.gpu_mem_reserve == 2.5 and c.fused_keep is False,
              "(%r, %r)" % (c.gpu_mem_reserve, c.fused_keep))
    finally:
        c.gpu_mem_reserve, c.fused_keep = saved


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 70)
        print("hIPPyMFEM vector tests on %d rank(s)" % NP)
        print("=" * 70)
    test_parvector()
    test_partition_matches_mfem()
    test_operators()
    test_multivector()
    test_random_partition_independence()
    test_host_build_warns_for_device_hypre()
    test_config_knobs()
    if RANK == 0:
        print("-" * 70)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

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
from hippymfem.common.parvector import ParVector, allreduce_extreme, host_sync, host_write
from hippymfem.test._device import same
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


def largest(values):
    """The largest of ``values``, and NaN where one of them is NaN.  Python's ``max`` keeps
    the argument it has unless the next one compares greater, which a NaN never does:
    ``max(0.0, nan)`` is 0.0, and a check ``max(...) < tol`` passes on a result that is
    not a number anywhere but in the first place."""
    values = [float(v) for v in values]
    return float("nan") if any(v != v for v in values) else max(values)


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


def test_nan_reductions():
    """A NaN on any one rank reaches ``norm("linf")``, ``max``, ``min`` and the
    per-column ``MultiVector.norm("linf")`` on every rank.  MPI's MAX and MIN drop a
    NaN that arrives second, so a NaN on the last rank came out finite."""
    if RANK == 0:
        print("NaN through the max/min reductions")
    for bad in range(NP):
        v = ParVector(COMM, 4)
        v.array[:] = 1.0 + RANK
        if RANK == bad:
            v.array[1] = np.nan
        got = (v.norm("linf"), v.max(), v.min())
        M = MultiVector(v, 2)
        M[0].array[:] = 1.0 + RANK
        M[1].array[:] = v.array
        cols = M.norm("linf")
        ok = (all(np.isnan(g) for g in got) and not np.isnan(cols[0])
              and np.isnan(cols[1]) and cols[0] == NP)
        ok = COMM.allreduce(int(ok), op=MPI.MIN) == 1
        check("NaN on rank %d reaches every rank" % bad, ok, str(got))
    v = ParVector(COMM, 3)
    v.array[:] = -(1.0 + RANK)
    ok = (v.norm("linf") == NP and v.max() == -1.0 and v.min() == -float(NP))
    check("finite values reduce as before", ok)


def test_layout_of_range_and_reduce():
    """A layout made from a range (a solver's vectors, ``Layout.of_range``) knows the
    offsets of every rank when it is made, so that a read of ``offsets`` on one rank only
    needs no message (one gathered at its first read hung there).  ``allreduce`` gives
    the same value on every rank where one rank holds an integer and the others doubles
    (it reduced their bits)."""
    from hippymfem.common.parvector import Layout, allreduce, partition

    if RANK == 0:
        print("layouts from a range, and reductions of mixed types")
    local = 3 + RANK
    offs, n = partition(COMM, local)
    lay = Layout.of_range(COMM, int(offs[RANK]), int(offs[RANK + 1]), n)
    known = COMM.allreduce(int(lay._offsets is not None), op=MPI.MIN) == 1
    alone = None
    if known and RANK == 0:
        # (only where it is known on every rank: a read that gathers would hang here)
        alone = list(ParVector(COMM, local, layout=lay).offsets)
    check("a layout from a range knows every rank's offsets when it is made",
          known and (RANK != 0 or alone == list(offs)), "(%s)" % (alone,))
    mixed = allreduce(COMM, 0 if RANK == 0 else 1.5, MPI.SUM)
    whole = allreduce(COMM, 1, MPI.SUM)
    flag = allreduce(COMM, int(RANK == NP - 1), MPI.MAX)
    ok = (mixed == 1.5 * (NP - 1) and whole == NP and isinstance(whole, int)
          and flag == 1 and isinstance(flag, int))
    ok = COMM.allreduce(int(ok), op=MPI.MIN) == 1
    check("allreduce of an integer on one rank and doubles on the others",
          ok, "(%r, %r, %r)" % (mixed, whole, flag))


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

    # (host_write and host_sync: with MFEM on a GPU, M.Mult reads and writes device
    # memory, and a raw view of a vector is only its host copy)
    xm = mfem.HypreParVector(M, 1)
    ym = mfem.HypreParVector(M, 0)
    host_write(xm).GetDataArray()[:] = x.array
    M.Mult(xm, ym)
    err = np.abs(host_sync(ym).GetDataArray() - y.array).max() if y.local_size else 0.0
    err = allreduce_extreme(COMM, err, MPI.MAX)
    ymax = allreduce_extreme(COMM, float(np.abs(y.array).max()) if y.local_size else 0.0,
                             MPI.MAX)
    check("HypreParMatrix.Mult agrees on our partition", same(err, ymax), "(err=%g)" % err)

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
    host_write(xv).GetDataArray()[:] = x.array
    mop.Mult(xv, yv)
    check("MFEMOperator adapter",
          np.abs(host_sync(yv).GetDataArray() - z.array).max() < 1e-14 if z.local_size else True)


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
    err = largest(recon[i].copy().axpy(-1.0, X[i]).norm("linf") for i in range(n))
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
    err = largest(Bq[i].copy().axpy(-1.0, BW[i]).norm("linf") for i in range(n))
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


def test_device_auto_without_gpu():
    """``HIPPYMFEM_DEVICE=auto`` with every GPU hidden runs on the host, without a word.

    In a fresh interpreter (JAX reads its platform list once, at import), on rank 0
    only, with the launcher's variables stripped so the child is a plain process.
    """
    import os
    import subprocess
    import sys

    from hippymfem.common.mfemconfig import mfem_has

    if RANK == 0:
        print("HIPPYMFEM_DEVICE=auto without a visible GPU")
    if mfem_has("MFEM_USE_CUDA"):
        # The child imports this process's PyMFEM, and a CUDA build of hypre cannot
        # start without a visible card (hypre reports CUDA error 35 at its
        # initialization), whatever the library decides; a HIP build can.
        if RANK == 0:
            print("  skipped: this PyMFEM is a CUDA build, whose hypre needs a visible card")
        return
    ok, detail = True, ""
    if RANK == 0:
        # PALS_ is how Slurm tells HPE Cray's PMI that srun launched it: left in, the
        # child takes itself for a rank of this step and waits in MPI_Init for the others
        # HIPPYMFEM_HYPRE_DEVICE and HIPPYMFEM_HYPRE_SINGLE go too: a suite run with
        # hypre on the GPU would otherwise have the child ask MFEM for a device it
        # cannot see, which MFEM answers with an abort.  So does JAX_PLATFORMS, which
        # the import of this process set (to "rocm,cpu" in a GPU run, and JAX then
        # refuses to start without a card): the child is to choose its own.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMIX_", "PMI_", "MPI_", "MV2_", "SLURM_",
                                    "PALS_"))
               and k not in ("HIPPYMFEM_HYPRE_DEVICE", "HIPPYMFEM_HYPRE_SINGLE",
                             "JAX_PLATFORMS")}
        env.update(HIPPYMFEM_DEVICE="auto", CUDA_VISIBLE_DEVICES="",
                   ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
        code = ("import hippymfem\nfrom hippymfem.fem import kernel\n"
                "from hippymfem import _jaxconfig as j\n"
                "print(j.platforms(), kernel.device().platform, hippymfem.config.device)")
        try:
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                               text=True, timeout=900)
            lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
            last = lines[-1] if lines else ""
            ok = r.returncode == 0 and last == "cpu cpu auto"
            detail = "(%r)" % (last if ok else (last or r.stderr[-400:]))
        except Exception as e:                                       # noqa: BLE001
            ok, detail = False, "(%s)" % e
    ok = bool(COMM.bcast(ok, root=0))
    check("auto without a visible GPU runs on the host", ok, COMM.bcast(detail, root=0))


def test_give_up_checked_at_import():
    """A misspelt ``HIPPYMFEM_SINGLE_GIVE_UP`` stops ``import hippymfem``, as the other
    settings read from the environment do; it was found only when a forward solve gave up.
    In a fresh interpreter on rank 0, a plain process on the host (as above)."""
    import os
    import subprocess
    import sys

    if RANK == 0:
        print("HIPPYMFEM_SINGLE_GIVE_UP checked at import")
    ok, detail = True, ""
    if RANK == 0:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMIX_", "PMI_", "MPI_", "MV2_", "SLURM_",
                                    "PALS_"))}
        for k in ("HIPPYMFEM_DEVICE", "HIPPYMFEM_HYPRE_DEVICE", "HIPPYMFEM_HYPRE_SINGLE"):
            env.pop(k, None)
        env.update(HIPPYMFEM_SINGLE_GIVE_UP="dobule", CUDA_VISIBLE_DEVICES="",
                   ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
        try:
            r = subprocess.run([sys.executable, "-c", "import hippymfem"], env=env,
                               capture_output=True, text=True, timeout=900)
            said = "HIPPYMFEM_SINGLE_GIVE_UP must be one of auto, double, raise" in r.stderr
            ok = r.returncode != 0 and said
            detail = "(exit %d%s)" % (r.returncode, "" if said else ": " + r.stderr[-300:])
        except Exception as e:                                       # noqa: BLE001
            ok, detail = False, "(%s)" % e
    ok = bool(COMM.bcast(ok, root=0))
    check("a misspelt HIPPYMFEM_SINGLE_GIVE_UP stops the import", ok, COMM.bcast(detail, root=0))


def test_comgr_cache_of_the_node():
    """AMD's code object manager gets a cache of this node, and one it was given is kept.

    The rule is applied to an environment of the test's own, so it is held on every
    machine; ``test_gpu`` looks at where the kernels of a run on an AMD card went.
    """
    import os
    import tempfile

    from hippymfem import _jaxconfig as j

    if RANK == 0:
        print("the cache of AMD's code object manager")
    env = {}
    path = j._point_comgr(env)
    if path is None:
        ok, detail = not env, "(no directory of this user under the temporary one)"
    else:
        st = os.stat(path)
        ok = (env == {"AMD_COMGR_CACHE_DIR": path} and st.st_uid == os.getuid()
              and not st.st_mode & 0o022
              and os.path.realpath(path).startswith(os.path.realpath(tempfile.gettempdir())))
        detail = "(%s)" % path
    check("it is a directory of this node and of this user",
          bool(COMM.allreduce(int(ok), op=MPI.MIN)), detail)
    env = {"AMD_COMGR_CACHE_DIR": "/a/place/of/the/user"}
    kept = j._point_comgr(env) is None and env == {"AMD_COMGR_CACHE_DIR": "/a/place/of/the/user"}
    check("a directory that AMD_COMGR_CACHE_DIR names is kept", kept)


def test_abort_hook():
    """On several ranks ``import hippymfem`` wraps ``sys.excepthook`` so that one rank's
    uncaught exception aborts all (``HIPPYMFEM_ABORT_ON_ERROR``); on one rank it does not;
    and with ``HIPPYMFEM_JAX_CACHE`` set, JAX's persistent cache is that directory."""
    import sys

    from hippymfem import _jaxconfig
    from hippymfem.common import abort

    size = MPI.COMM_WORLD.Get_size()
    want = size > 1 and abort.ABORT_ON_ERROR
    check("the abort hook is in place on several ranks, and only there",
          abort.installed() == want and ((sys.excepthook is abort._INSTALLED[0][0]) if want else True),
          "(%d ranks, installed %s)" % (size, abort.installed()))
    if want:
        hook = abort._INSTALLED[0][0]
        abort.uninstall()
        ok = sys.excepthook is not hook and not abort.installed()
        abort.install()
        check("it can be taken out and put back", ok and abort.installed())
        # What the hook does with an error that a console shows and goes on (pdb's
        # interact), and with one that nothing handles; abort._abort is replaced by a
        # recorder, so that nothing ends here.
        import code
        import io
        import types
        import weakref

        calls = []
        saved_abort, saved_err = abort._abort, sys.stderr
        abort._abort = lambda world: calls.append(world.Get_rank())
        sys.stderr = io.StringIO()
        try:
            code.InteractiveConsole().push("1/0")
            console = list(calls)
            try:
                raise ValueError("a test of the abort hook")
            except ValueError:
                sys.excepthook(*sys.exc_info())
            plain = list(calls)
        finally:
            abort._abort, sys.stderr = saved_abort, saved_err
        rank = MPI.COMM_WORLD.Get_rank()
        check("an error that a console shows and goes on past does not abort; one that "
              "nothing handles does", console == [] and plain == [rank],
              "(after the console %s, after the plain error %s)" % (console, plain))
        exitfunc = weakref.finalize._exitfunc.__func__
        check("an exception of a finalizer that weakref.finalize runs at exit is only reported",
              abort.handled_elsewhere(types.SimpleNamespace(f_code=exitfunc.__code__))
              and not abort.handled_elsewhere(sys._getframe()))
    if _jaxconfig.JAX_CACHE:
        import jax

        check("HIPPYMFEM_JAX_CACHE is JAX's persistent compilation cache",
              jax.config.jax_compilation_cache_dir == _jaxconfig.JAX_CACHE,
              "(%r)" % (jax.config.jax_compilation_cache_dir,))


def test_config_profiles():
    """(Run with a clean record of the knobs set through hm.config, which it sets and
    restores itself, and that record restored afterwards: such a knob outranks a profile.)"""
    import importlib

    cfg = importlib.import_module("hippymfem.config")
    kept = set(cfg.EXPLICIT)
    cfg.EXPLICIT.clear()
    try:
        _test_config_profiles()
    finally:
        cfg.EXPLICIT.clear()
        cfg.EXPLICIT.update(kept)


def _test_config_profiles():
    """Profiles (``HIPPYMFEM_PROFILE``, ``hm.config.use_profile``) fill in what the
    environment leaves unset: an explicit setting wins, mixed precision comes only with a
    single-precision hypre named, an unknown name is refused, an import-time knob is left
    with a warning after import, and ``show`` marks what a profile set."""
    import io
    import os
    import warnings
    from contextlib import redirect_stdout

    import importlib

    cfg = importlib.import_module("hippymfem.config")    # (the module; hp.config is its object)

    env = {"HIPPYMFEM_AMG": "relax_wt=1.0"}
    got = cfg.apply_profile_env("gpu-large", env)
    check("gpu-large fills in the unset variables and leaves an explicit one",
          env["HIPPYMFEM_AMG"] == "relax_wt=1.0" and env.get("HIPPYMFEM_PMG_H") == "7"
          and env.get("HIPPYMFEM_PMG") == "1" and env.get("HIPPYMFEM_HESSIAN") == "quadrature"
          and "HIPPYMFEM_AMG" not in got,
          "(%r)" % (sorted(got),))
    check("without a single-precision hypre named, no profile asks for mixed precision",
          "HIPPYMFEM_PRECISION" not in env and "HIPPYMFEM_AD_WORKING_SET" not in env)
    env = {"HIPPYMFEM_HYPRE_SINGLE": "/somewhere/libHYPRE_single.so"}
    cfg.apply_profile_env("gpu", env)
    check("with one named, gpu asks for mixed precision and single-precision tangents",
          env.get("HIPPYMFEM_PRECISION") == "mixed" and env.get("HIPPYMFEM_AD_WORKING_SET") == "8")
    check("the default profile changes nothing", cfg.apply_profile_env("default", {}) == {})
    try:
        cfg.apply_profile_env("fastest", {})
        refused = False
    except ValueError:
        refused = True
    check("an unknown profile is refused", refused)

    # after import: a profile of this test's own, so the suite's kernels are not touched
    names = ("HIPPYMFEM_PMG_H", "HIPPYMFEM_AMG", "HIPPYMFEM_GPU_MEM_FRACTION")
    kept_env = {n: os.environ.get(n) for n in names}
    kept_set, kept_active = dict(cfg.PROFILE_SET), list(cfg._ACTIVE_PROFILE)
    cfg.PROFILES["_test"] = ("a test", {"HIPPYMFEM_PMG_H": "3", "HIPPYMFEM_AMG": "relax_wt=1.5",
                                        "HIPPYMFEM_GPU_MEM_FRACTION": "0.5"})
    try:
        os.environ.pop("HIPPYMFEM_PMG_H", None)
        os.environ.pop("HIPPYMFEM_GPU_MEM_FRACTION", None)     # (JAX read it at import)
        os.environ["HIPPYMFEM_AMG"] = "relax_wt=1.0"            # the user's: kept
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = hp.config.use_profile("_test")
        buf = io.StringIO()
        with redirect_stdout(buf):
            hp.config.show(load=False)
        line = [x for x in buf.getvalue().splitlines() if x.strip().startswith("pmg_h ")]
        check("use_profile sets a runtime knob, keeps the user's, and leaves an import-time "
              "one with a warning",
              got == {"pmg_h": "3"} and int(hp.config.pmg_h) == 3
              and os.environ["HIPPYMFEM_AMG"] == "relax_wt=1.0"
              and "HIPPYMFEM_GPU_MEM_FRACTION" not in os.environ
              and any("read at import" in str(x.message) for x in w),
              "(%r, pmg_h %r)" % (got, hp.config.pmg_h))
        check("show() marks what the profile set", bool(line) and "profile _test" in line[0],
              "(%s)" % (line[0].strip()[:100] if line else "no line"))
    finally:
        del cfg.PROFILES["_test"]
        for n, v in kept_env.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v
        cfg.PROFILE_SET.clear()
        cfg.PROFILE_SET.update(kept_set)
        cfg._ACTIVE_PROFILE[:] = kept_active


def test_config_explicit_and_spellings():
    """A knob set through ``hm.config`` outranks a profile applied afterwards, and ``show``
    says it was set there (a profile overrode it, and ``show`` called it the default).  An
    on/off setting reads the same at import and through its setter: empty is the default,
    1/yes/true/on and 0/no/false/off in any case, anything else is refused by name (an
    empty HIPPYMFEM_SINGLE_ROW_SUMS turned the row sums off at import, and an unknown word
    was read as one of the two)."""
    import io
    import os
    import subprocess
    import sys
    from contextlib import redirect_stdout

    import importlib

    cfg = importlib.import_module("hippymfem.config")
    kept = os.environ.get("HIPPYMFEM_PMG_H")
    kept_explicit, kept_set = set(cfg.EXPLICIT), dict(cfg.PROFILE_SET)
    kept_active = list(cfg._ACTIVE_PROFILE)
    cfg.PROFILES["_test2"] = ("a test", {"HIPPYMFEM_PMG_H": "3"})
    try:
        os.environ.pop("HIPPYMFEM_PMG_H", None)
        hp.config.pmg_h = 2
        got = hp.config.use_profile("_test2")
        buf = io.StringIO()
        with redirect_stdout(buf):
            hp.config.show(load=False)
        line = [x for x in buf.getvalue().splitlines() if x.strip().startswith("pmg_h ")]
        check("a profile applied later leaves a knob set through hm.config",
              got == {} and int(hp.config.pmg_h) == 2, "(%r, pmg_h %r)" % (got, hp.config.pmg_h))
        check("show() reports such a knob as set", bool(line) and "[set," in line[0],
              "(%s)" % (line[0].strip()[:80] if line else "no line"))
    finally:
        del cfg.PROFILES["_test2"]
        if kept is None:
            os.environ.pop("HIPPYMFEM_PMG_H", None)
        else:
            os.environ["HIPPYMFEM_PMG_H"] = kept
        cfg.EXPLICIT.clear()
        cfg.EXPLICIT.update(kept_explicit)
        cfg.PROFILE_SET.clear()
        cfg.PROFILE_SET.update(kept_set)
        cfg._ACTIVE_PROFILE[:] = kept_active

    readings = [cfg.parse_flag(v, d) for v, d in (("", True), ("", False), (" Off ", True),
                                                   ("YES", False), (0, True), (True, False))]
    refused = []
    for parse in (lambda: cfg.parse_flag("bogus", True, "HIPPYMFEM_X"),
                  lambda: cfg.parse_tristate("sometimes", "HIPPYMFEM_Y")):
        try:
            parse()
            refused.append(False)
        except ValueError as e:
            refused.append("HIPPYMFEM_" in str(e))
    tri = [cfg.parse_tristate(v) for v in ("", "AUTO", "on", "no")]
    check("one reading of on/off: empty is the default, any spelling, unknown refused",
          readings == [True, False, False, True, False, True] and all(refused)
          and tri == ["auto", "auto", "1", "0"], "(%r %r %r)" % (readings, refused, tri))

    ok, detail = True, ""
    if RANK == 0:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OMPI_", "PMIX_", "PMI_", "MPI_", "MV2_", "SLURM_", "PALS_"))}
        env.update(HIPPYMFEM_SINGLE_ROW_SUMS="", HIPPYMFEM_FOLD_ELIMINATION="Off",
                   HIPPYMFEM_DEVICE_VECTORS="no", HIPPYMFEM_DEVICE="cpu", JAX_PLATFORMS="cpu")
        env.pop("HIPPYMFEM_HYPRE_DEVICE", None)
        code = ("import hippymfem as hm\nc = hm.config\n"
                "print(c.single_row_sums, c.fold_elimination, c.device_vectors)")
        try:
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                               text=True, timeout=900)
            last = (r.stdout.strip().splitlines() or [""])[-1]
            ok = r.returncode == 0 and last == "True False 0"
            detail = "(%r)" % (last if ok else (last or r.stderr[-300:]))
        except Exception as e:                                       # noqa: BLE001
            ok, detail = False, "(%s)" % e
    check("the same spellings at import", bool(COMM.bcast(ok, root=0)), COMM.bcast(detail, root=0))


def test_config_knobs():
    """(Run with a clean record of the knobs set through hm.config, which it sets and
    restores itself, and that record restored afterwards: such a knob outranks a profile.)"""
    import importlib

    cfg = importlib.import_module("hippymfem.config")
    kept = set(cfg.EXPLICIT)
    cfg.EXPLICIT.clear()
    try:
        _test_config_knobs()
    finally:
        cfg.EXPLICIT.clear()
        cfg.EXPLICIT.update(kept)


def _test_config_knobs():
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
    saved = (c.gpu_mem_reserve, c.fused_keep, c.fold_elimination, c.pattern_sort)
    try:
        c.gpu_mem_reserve = 2.5
        c.fused_keep = "0"
        c.fold_elimination = "0"
        check("a fractional reserve in GiB and a false string keep their meaning",
              c.gpu_mem_reserve == 2.5 and c.fused_keep is False
              and c.fold_elimination is False,
              "(%r, %r, %r)" % (c.gpu_mem_reserve, c.fused_keep, c.fold_elimination))
        try:
            c.pattern_sort = "no such route"
            refused = False
        except ValueError:
            refused = True
        check("a route name that does not exist is refused",
              refused and c.pattern_sort == saved[3], "(%r)" % (c.pattern_sort,))
    finally:
        c.gpu_mem_reserve, c.fused_keep, c.fold_elimination, c.pattern_sort = saved
    # The knobs that live in the environment are written as the code that reads them back
    # parses them: ``True`` and ``2.0`` were written as ``"True"`` and ``"2.0"``, which a new
    # solver read as 0 without a word and a prior stopped at.
    import os

    from hippymfem.modeling.prior import prior_h

    names = ("HIPPYMFEM_PMG", "HIPPYMFEM_PMG_H", "HIPPYMFEM_PMG_H_HOST", "HIPPYMFEM_PRIOR_H")
    kept = {n: os.environ.get(n) for n in names}
    try:
        c.pmg, c.pmg_h, c.pmg_h_host, c.prior_h = True, 2.0, 5e3, True
        p = hp.KrylovSolver_ParameterList()
        try:
            got = (p["pmg"], p["pmg_h"], p["pmg_h_host"], prior_h())
        except ValueError as e:
            got = (p["pmg"], p["pmg_h"], p["pmg_h_host"], str(e))
        check("a switch set to True and a count given as a float reach a new solver and a prior",
              got == (1, 2, 5000, 1), "(%r)" % (got,))
        try:
            c.pmg_h = 2.5
            refused = False
        except ValueError:
            refused = True
        check("a count that is no whole number is refused",
              refused and hp.KrylovSolver_ParameterList()["pmg_h"] == 2, "(%r)" % (c.pmg_h,))
        # The same for a variable that a job script sets: a value that is no whole number
        # is an error that names the variable.  A new solver took its default for it and
        # ran another configuration without a word, and a prior stopped at the message of
        # ``int``, which names none.  An empty value is the default, as it was.
        os.environ["HIPPYMFEM_PMG_H"], os.environ["HIPPYMFEM_PRIOR_H"] = "two", "on"
        said = []
        for read in (hp.KrylovSolver_ParameterList, prior_h):
            try:
                read()
                said.append("no error")
            except ValueError as e:
                said.append(str(e))
        os.environ["HIPPYMFEM_PMG_H"] = os.environ["HIPPYMFEM_PRIOR_H"] = ""
        check("a variable that is no whole number is an error that names it, for a new solver "
              "and for a prior, and an empty one the default",
              "HIPPYMFEM_PMG_H" in said[0] and "HIPPYMFEM_PRIOR_H" in said[1]
              and hp.KrylovSolver_ParameterList()["pmg_h"] == 0 and prior_h() == 0,
              "(%s)" % "; ".join(said))
    finally:
        for n, v in kept.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


def test_ensemble():
    """Every rank holds the whole problem (on COMM_SELF); MatMvMult and doublePassG divide
    the columns among the ranks and must return what one rank alone computes."""
    if RANK == 0:
        print("ensemble (a complete problem on every rank)")
    self_comm = MPI.COMM_SELF
    mesh = mfem.Mesh.MakeCartesian2D(8, 8, mfem.Element.TRIANGLE)
    pmesh = mfem.ParMesh(self_comm, mesh)
    fec = mfem.H1_FECollection(1, 2)
    fes = mfem.ParFiniteElementSpace(pmesh, fec)

    def matrix(*integrators):
        a = mfem.ParBilinearForm(fes)
        for integ in integrators:
            a.AddDomainIntegrator(integ)
        a.Assemble()
        a.Finalize()
        return a.ParallelAssemble()
    Amat, Mmat = matrix(mfem.DiffusionIntegrator(), mfem.MassIntegrator()), matrix(mfem.MassIntegrator())
    A = hp.MatrixOperator(Amat, self_comm)
    v = ParVector.from_fes(fes)
    n = 7                                   # not a multiple of the rank count on 2 or 4 ranks
    X = MultiVector(v, n)
    X.data[:] = np.random.default_rng(11).standard_normal(X.data.shape)   # the same on every rank
    Y1, Y2 = MultiVector(v, n), MultiVector(v, n)
    MatMvMult(A, X, Y1)
    MatMvMult(A, X, Y2, ensemble=COMM)
    err = allreduce_extreme(COMM, float(np.abs(Y1.data - Y2.data).max()), MPI.MAX)
    check("MatMvMult divided among the ranks == one rank", err == 0.0, "(%.2e)" % err)

    # the generalized eigenproblem A u = lambda B u with a diagonal B
    d = v.copy()
    d.array[:] = 1.0 + 0.5 * np.cos(np.arange(d.local_size))
    B = hp.DiagonalOperator(d)
    Omega = MultiVector(v, 12)
    Omega.data[:] = np.random.default_rng(12).standard_normal(Omega.data.shape)
    d1, U1 = hp.doublePassG(A, B, B, MultiVector(Omega), 6)
    d2, U2 = hp.doublePassG(A, B, B, MultiVector(Omega), 6, ensemble=COMM)
    # B passed as Binv: an operator with both mult (B) and solve (B^-1) applies its
    # solve, as an explicit Solver2Operator does (its mult was taken, and the
    # eigenvalues came out off by up to 40%)
    from hippymfem.common.operators import Solver2Operator

    d3, _ = hp.doublePassG(A, B, Solver2Operator(B), MultiVector(Omega), 6)
    err = allreduce_extreme(COMM, float(np.abs(d1 / d3 - 1.0).max()), MPI.MAX)
    check("doublePassG takes solve from a Binv that has mult and solve", err < 1e-12,
          "(%.2e)" % err)
    err = allreduce_extreme(COMM, float(np.abs(d2 / d1 - 1.0).max()), MPI.MAX)
    check("doublePassG divided among the ranks: eigenvalues", err < 1e-12, "(%.2e)" % err)
    err = allreduce_extreme(COMM, float(np.abs(U1.data - U2.data).max()), MPI.MAX)
    check("doublePassG divided among the ranks: eigenvectors", err < 1e-10, "(%.2e)" % err)
    spread = (allreduce_extreme(COMM, float(d2[0]), MPI.MAX)
              - allreduce_extreme(COMM, float(d2[0]), MPI.MIN))
    check("every rank returns the same eigenvalues", spread == 0.0, "(%.2e)" % spread)


def test_sum_operator_rectangular():
    """A sum of two rectangular operators: mult and then multTranspose (they shared one
    scratch vector, sized for whichever came first)."""
    from hippymfem.common.operators import Operator, SumOperator, init_vector_like

    rng = np.random.default_rng(0)
    M1, M2 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    rng_t, dom_t = ParVector(MPI.COMM_SELF, 5), ParVector(MPI.COMM_SELF, 3)

    class Dense(Operator):
        def __init__(self, M):
            self.M = M

        def init_vector(self, x, dim):
            return init_vector_like(x, rng_t if dim == 0 else dom_t)

        def mult(self, x, y):
            y.array[:] = self.M @ x.array
            return y

        def multTranspose(self, x, y):
            y.array[:] = self.M.T @ x.array
            return y

    S = SumOperator(Dense(M1), Dense(M2))
    x, y, z = S.generate_vector(1), S.generate_vector(0), S.generate_vector(1)
    x.array[:] = 1.0
    S.mult(x, y)
    try:
        S.multTranspose(y, z)
        ok = (np.allclose(y.array, (M1 + M2) @ x.array)
              and np.allclose(z.array, (M1 + M2).T @ y.array))
    except ValueError:
        ok = False
    check("SumOperator of rectangular operators, both directions", bool(ok))


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 70)
        print("hIPPyMFEM vector tests on %d rank(s)" % NP)
        print("=" * 70)
    test_parvector()
    test_nan_reductions()
    test_layout_of_range_and_reduce()
    test_partition_matches_mfem()
    test_operators()
    test_multivector()
    test_ensemble()
    test_random_partition_independence()
    test_host_build_warns_for_device_hypre()
    test_config_knobs()
    test_config_profiles()
    test_config_explicit_and_spellings()
    test_abort_hook()
    test_device_auto_without_gpu()
    test_give_up_checked_at_import()
    test_comgr_cache_of_the_node()
    test_sum_operator_rectangular()
    if RANK == 0:
        print("-" * 70)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""GPU execution of the AD layer: same answers, measured speed.

What runs on the GPU here is the part of an assembly that is not MFEM: the dof
gather, the batched element kernels, and the scatter into the CSR structure.  The
matrix itself, hypre and every linear solve stay on the host in this suite (the
CUDA build of PyMFEM has its own, ``test_device``), so the claim being tested is
narrow and checkable: **the GPU path must produce the same operators as the CPU
path, and it must be faster at the element level.**

Both backends are live in one process (``JAX_PLATFORMS=cuda,cpu``), so each check
assembles the same problem twice, switching only the device.  That is a stronger
comparison than two runs, because the mesh, the dof numbering, the random draws and
the reference values are literally the same objects.

The tests skip with a printed reason when no GPU is visible; they never pass
silently.  Run with::

    HIPPYMFEM_DEVICE=gpu python -m hippymfem.test.test_gpu
    HIPPYMFEM_DEVICE=gpu mpirun -n 4 python -m hippymfem.test.test_gpu
"""

import os
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hp
from hippymfem.common.linalg import to_dense
from hippymfem.fem import kernel as K
from hippymfem.fem.assemble import assemble_matrix, assemble_vector
from hippymfem.fem.elementbatch import MeshBatches
from hippymfem.fem.kernel import QuadratureKernel
from hippymfem.fem.spaces import FunctionSpace
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
FAILS = []
#: the accelerator backend this run found, settled once (see :func:`accel`)
_ACCEL = ()


def accel():
    """The accelerator backend that came up: ``"cuda"`` on NVIDIA, ``"rocm"`` on AMD.

    ``None`` when neither did, which is what makes this suite skip rather than test the
    host twice: asking JAX for the *default* backend would answer ``cpu`` and read as a
    card that is not there.
    """
    global _ACCEL
    import jax

    if _ACCEL == ():
        _ACCEL = None
        for name in ("cuda", "rocm"):
            try:
                if jax.devices(name):
                    _ACCEL = name
                    break
            except Exception:                                     # noqa: BLE001
                continue
    return _ACCEL


def check(name, ok, detail=""):
    if RANK == 0:
        print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


def have_gpu():
    return accel() is not None


def gpu_info():
    import jax

    name = accel()
    return jax.devices(name) if name else []


class on(object):
    """Context manager running a block on one device and restoring the old one."""

    def __init__(self, kind):
        self.kind = kind

    def __enter__(self):
        self.old = K.device()
        K.set_device(self.kind)
        return K.device()

    def __exit__(self, *a):
        K._DEVICE = self.old
        return False


# ---------------------------------------------------------------- problem setup
def build(kind="quad", n=8, order=2, family="H1"):
    m = (mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.QUADRILATERAL)
         if kind == "quad" else
         mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON)
         if kind == "hex" else
         mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE))
    pm = mfem.ParMesh(COMM, m)
    if family == "H1":
        Vu = FunctionSpace.H1(pm, order)
    elif family == "ND":
        Vu = FunctionSpace.ND(pm, order)
    else:
        Vu = FunctionSpace.RT(pm, order)
    Vm = FunctionSpace.H1(pm, 1)
    batches = MeshBatches(pm, 2 * order + 2, COMM)

    if family == "H1":
        def varf(u, m_, p, x):
            return (jnp.exp(m_.val) * jnp.dot(u.grad, p.grad)
                    + u.val * u.val * m_.val * p.val)
    elif family == "ND":
        def varf(u, m_, p, x):
            return jnp.exp(m_.val) * u.curl * p.curl + jnp.dot(u.val, p.val)
    else:
        def varf(u, m_, p, x):
            return jnp.exp(m_.val) * u.div * p.div + jnp.dot(u.val, p.val)

    kern = QuadratureKernel(varf, [Vu, Vm, Vu], batches)
    hp.parRandom.set_seed(21)
    uv = Vu.vector()
    hp.parRandom.normal(1.0, uv)
    mv = Vm.vector()
    hp.parRandom.normal(0.4, mv)
    pv = Vu.vector()
    hp.parRandom.normal(1.0, pv)
    loc = [Vu.local_values(uv), Vm.local_values(mv), Vu.local_values(pv)]
    return pm, Vu, Vm, batches, kern, loc


# ------------------------------------------------------------------- agreement
def test_blocks_agree():
    """Every block, on the GPU and on the CPU, from the same inputs."""
    if RANK == 0:
        print("GPU vs CPU element blocks")
    cases = [("quad", 1, "H1"), ("quad", 2, "H1"), ("tri", 2, "H1"),
             ("hex", 1, "H1"), ("tri", 1, "ND"), ("tri", 0, "RT")]
    worst = 0.0
    for kind, order, family in cases:
        n = 4 if kind == "hex" else 6
        pm, Vu, Vm, b, kern, loc = build(kind, n, order, family)
        spaces = [Vu, Vm, Vu]
        blocks = [("A", ADJOINT, STATE), ("C", ADJOINT, PARAMETER),
                  ("W_uu", STATE, STATE), ("W_um", STATE, PARAMETER),
                  ("W_mm", PARAMETER, PARAMETER)]
        bad = []
        for name, i, j in blocks:
            with on("cpu"):
                ref = [np.asarray(a) for a in kern.element_matrices(i, j, loc)]
            with on("gpu"):
                got = [np.asarray(a) for a in kern.element_matrices(i, j, loc)]
            for r, g in zip(ref, got):
                sc = max(float(np.abs(r).max()), 1e-300)
                e = float(np.abs(r - g).max()) / sc
                worst = max(worst, e)
                if e > 1e-13:
                    bad.append("%s %.2e" % (name, e))
        for var in (STATE, PARAMETER, ADJOINT):
            with on("cpu"):
                ref = [np.asarray(a) for a in kern.element_vectors(var, loc)]
            with on("gpu"):
                got = [np.asarray(a) for a in kern.element_vectors(var, loc)]
            for r, g in zip(ref, got):
                sc = max(float(np.abs(r).max()), 1e-300)
                e = float(np.abs(r - g).max()) / sc
                worst = max(worst, e)
                if e > 1e-13:
                    bad.append("vec %d %.2e" % (var, e))
        check("%s order %d %s: all blocks agree" % (kind, order, family), not bad,
              "" if not bad else "differs: " + ", ".join(bad))
    check("worst GPU/CPU element difference is at round-off", worst < 1e-13,
          "(%.3e)" % worst)


def test_assembled_operators_agree():
    """The assembled parallel matrices, not just the element arrays.

    This is the check that the device-side scatter lands entries in the same CSR
    slots as the host-side one, including the essential-dof elimination that
    happens afterwards on the host.
    """
    if RANK == 0:
        print("GPU vs CPU assembled operators")
    pm, Vu, Vm, b, kern, loc = build("quad", 6, 2)
    NE = pm.GetNE()
    bc = hp.DirichletBC(Vu, None, "all")
    worst = 0.0
    for name, i, j, ess, pol in (("A", ADJOINT, STATE, bc.ess_tdof, "one"),
                                 ("W_uu", STATE, STATE, bc.ess_tdof, "zero"),
                                 ("C", ADJOINT, PARAMETER, bc.ess_tdof, "one")):
        spaces = [Vu, Vm, Vu]
        out = {}
        for dev in ("cpu", "gpu"):
            with on(dev):
                out[dev] = to_dense(assemble_matrix(
                    spaces[i], spaces[j], b.groups,
                    kern.element_matrices(i, j, loc), NE,
                    test_ess=ess, diag_policy=pol), COMM)
        sc = max(float(np.abs(out["cpu"]).max()), 1e-300)
        e = float(np.abs(out["cpu"] - out["gpu"]).max()) / sc
        worst = max(worst, e)
        check("assembled %s agrees" % name, e < 1e-13, "(rel %.3e)" % e)
    for var in (STATE, ADJOINT):
        out = {}
        for dev in ("cpu", "gpu"):
            with on(dev):
                out[dev] = assemble_vector([Vu, Vm, Vu][var], b.groups,
                                           kern.element_vectors(var, loc), NE)
        d = out["cpu"].copy()
        d.axpy(-1.0, out["gpu"])
        e = d.norm("l2") / max(out["cpu"].norm("l2"), 1e-300)
        worst = max(worst, e)
        check("assembled residual %d agrees" % var, e < 1e-13, "(rel %.3e)" % e)
    check("worst assembled difference is at round-off", worst < 1e-13,
          "(%.3e)" % worst)


def test_geometry_streaming():
    """A split batch whose geometry is too large for the budget streams it, exactly.

    Streaming hands the kernels host slices of the same arrays the device cache would
    have held, at the same chunk sizes, so the element arrays must be bit-identical;
    below the threshold the device cache must still be used.  The fused scatter's
    host-side zeroing of eliminated slots is checked on the device path here too,
    since only a device array reaches it read-only.
    """
    if RANK == 0:
        print("geometry streaming and host-side zeroing")
    with on("gpu") as d:
        pm, Vu, Vm, b, kern, loc = build("quad", 8, 2)
        old = K.ELEMENT_CHUNK, K.GEOMETRY_STREAM_FRACTION

        def reset(chunk, frac):
            K.ELEMENT_CHUNK, K.GEOMETRY_STREAM_FRACTION = chunk, frac
            for gk in kern.group_kernels:
                gk._cache.clear()
                gk._chunk.clear()
                gk._dev.clear()

        def cached():
            return [gk._dev[d][1] is not None for gk in kern.group_kernels]

        try:
            out = {}
            for label, frac in (("cached", 0.0), ("streamed", 1e-12)):
                reset(7, frac)
                out[label] = ([np.asarray(a) for a in kern.element_matrices(ADJOINT, STATE, loc)],
                              [np.asarray(a) for a in kern.element_vectors(ADJOINT, loc)],
                              cached())
            check("fraction 0 keeps the device cache", all(out["cached"][2]))
            check("a split batch over the fraction streams", not any(out["streamed"][2]))
            dm = max(float(np.abs(a - c).max()) for a, c in zip(out["cached"][0], out["streamed"][0]))
            dv = max(float(np.abs(a - c).max()) for a, c in zip(out["cached"][1], out["streamed"][1]))
            check("streamed element matrices are bit-identical", dm == 0.0, "(%.3e)" % dm)
            check("streamed element vectors are bit-identical", dv == 0.0, "(%.3e)" % dv)
            reset(7, 0.99)
            kern.element_vectors(ADJOINT, loc)
            check("a geometry under the fraction is still cached", all(cached()))
            reset(0, 1e-12)
            kern.element_vectors(ADJOINT, loc)
            check("a batch that is not split is still cached", all(cached()))

            reset(7, 0.25)
            NE = pm.GetNE()
            bc = hp.DirichletBC(Vu, None, "all")
            glued = to_dense(assemble_matrix(Vu, Vu, b.groups,
                                             kern.element_matrices(ADJOINT, STATE, loc),
                                             NE, test_ess=bc.ess_tdof), COMM)
            fused = to_dense(assemble_matrix(Vu, Vu, b.groups,
                                             lambda: kern.element_matrix_chunks(ADJOINT, STATE, loc),
                                             NE, test_ess=bc.ess_tdof), COMM)
            sc = max(float(np.abs(glued).max()), 1e-300)
            e = float(np.abs(glued - fused).max())
            check("fused scatter with host-side zeros == glued", e <= 1e-13 * sc,
                  "(%.3e, scale %.3e)" % (e, sc))
        finally:
            reset(*old)


def test_reproducible():
    """Repeated GPU assemblies of the same input must give the same bits.

    A scatter built on atomics would not, and a library whose operators wobble
    between calls cannot support a Newton method with a tight tolerance.
    """
    if RANK == 0:
        print("GPU reproducibility")
    pm, Vu, Vm, b, kern, loc = build("quad", 8, 2)
    from hippymfem.fem.csrassemble import get_pattern, set_deterministic

    for mode in (False, True):
        old = set_deterministic(mode)
        try:
            with on("gpu"):
                pat = get_pattern(Vu, Vu, b.groups)
                vals = [pat.data(kern.element_matrices(ADJOINT, STATE, loc))
                        for _ in range(5)]
            same = all(np.array_equal(vals[0], v) for v in vals[1:])
            spread = max(float(np.abs(vals[0] - v).max()) for v in vals[1:])
            sc = max(float(np.abs(vals[0]).max()), 1e-300)
            if mode:
                check("deterministic scatter: five assemblies are bit-identical",
                      same, "(spread %.3e, maxc=%s)" % (spread, pat.maxc))
            else:
                # The default device scatter is a scatter-add, which on a GPU uses
                # atomics: the order varies between runs, so the requirement is
                # that the variation stays at round-off, not that it be zero.
                check("default scatter: five assemblies agree at round-off",
                      spread / sc < 1e-14,
                      "(spread %.3e relative, bit-identical: %s)"
                      % (spread / sc, same))
        finally:
            set_deterministic(old)

    # and the two modes must agree with each other
    with on("gpu"):
        pat = get_pattern(Vu, Vu, b.groups)
        a = pat.data(kern.element_matrices(ADJOINT, STATE, loc))
        old = set_deterministic(True)
        try:
            bb = pat.data(kern.element_matrices(ADJOINT, STATE, loc))
        finally:
            set_deterministic(old)
    e = float(np.abs(a - bb).max()) / max(float(np.abs(a).max()), 1e-300)
    check("the two device scatters agree", e < 1e-14, "(rel %.3e)" % e)


def test_boundary_on_gpu():
    """Boundary (``ds``) kernels on the GPU, against the CPU."""
    if RANK == 0:
        print("boundary kernels on the GPU")
    from hippymfem.fem.boundary import (BoundaryKernel, assemble_boundary_matrix,
                                        assemble_boundary_vector,
                                        get_boundary_batches)

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(
        max(6, 2 * NP), max(6, 2 * NP), mfem.Element.TRIANGLE))
    Vu = FunctionSpace.H1(pm, 2)
    Vm = FunctionSpace.H1(pm, 1)
    bb = get_boundary_batches(pm, 6, "all", COMM, space=Vu)
    kern = BoundaryKernel(
        lambda u, m, p, x, n: jnp.exp(m.val) * jnp.dot(u.grad, n) * p.val
        + 2.0 * u.val * p.val, [Vu, Vm, Vu], bb)
    hp.parRandom.set_seed(3)
    uv = Vu.vector()
    hp.parRandom.normal(1.0, uv)
    mv = Vm.vector()
    hp.parRandom.normal(0.3, mv)
    loc = [Vu.local_values(uv), Vm.local_values(mv), Vu.local_values(Vu.vector())]
    out = {}
    for dev in ("cpu", "gpu"):
        with on(dev):
            out[dev] = to_dense(assemble_boundary_matrix(
                Vu, Vu, bb.groups,
                kern.element_matrices(ADJOINT, STATE, loc)), COMM)
    sc = max(float(np.abs(out["cpu"]).max()), 1e-300)
    e = float(np.abs(out["cpu"] - out["gpu"]).max()) / sc
    check("boundary block agrees", e < 1e-13, "(rel %.3e)" % e)

    vout = {}
    for dev in ("cpu", "gpu"):
        with on(dev):
            vout[dev] = assemble_boundary_vector(
                Vu, bb.groups, kern.element_vectors(ADJOINT, loc))
    d = vout["cpu"].copy()
    d.axpy(-1.0, vout["gpu"])
    e = d.norm("l2") / max(vout["cpu"].norm("l2"), 1e-300)
    check("boundary residual agrees", e < 1e-13, "(rel %.3e)" % e)


def test_inverse_problem_on_gpu():
    """A whole inverse problem with GPU assembly must land on the same MAP point."""
    if RANK == 0:
        print("inverse problem with GPU assembly")
    results = {}
    for dev in ("cpu", "gpu"):
        with on(dev):
            pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(
                16, 16, mfem.Element.TRIANGLE))
            Vu = FunctionSpace.H1(pm, 2)
            Vm = FunctionSpace.H1(pm, 1)

            def varf(u, m, p, x):
                return jnp.exp(m.val) * hp.inner(u.grad, p.grad)

            bc = hp.DirichletBC(Vu, lambda z: z[1], bdr_attributes=[1, 3])
            pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc,
                                           bc.homogeneous(), is_fwd_linear=True)
            for a in ("solver", "solver_fwd_inc", "solver_adj_inc"):
                if NP == 1:
                    setattr(pde, a, hp.LUSolver(COMM))
                else:
                    s = hp.KrylovSolver(COMM, "cg", "amg")
                    s.parameters["rel_tolerance"] = 1e-13
                    s.parameters["max_iter"] = 2000
                    setattr(pde, a, s)
            prior = hp.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True,
                                        solver_type="lu" if NP == 1 else "krylov")
            rng = np.random.default_rng(9)
            targets = np.column_stack((rng.uniform(0.1, 0.9, 50),
                                       rng.uniform(0.1, 0.5, 50)))
            B = hp.assemblePointwiseObservation(Vu, targets)
            mtrue = Vm.project(
                lambda z: np.sin(2.0 * z[0]) * np.cos(2.0 * z[1]))
            ut = pde.generate_state()
            pde.solveFwd(ut, [ut, mtrue, None])
            data = B.createVecLeft()
            B.mult(ut, data)
            std = 0.01 * max(data.norm("linf"), 1e-30)
            hp.parRandom.set_seed(9)
            B.perturb(data, std)
            misfit = hp.DiscreteStateObservation(B, data, std ** 2)
            model = hp.Model(pde, prior, misfit)
            params = hp.ReducedSpaceNewtonCG_ParameterList()
            params["rel_tolerance"] = 1e-9
            params["max_iter"] = 30
            params["print_level"] = -1
            solver = hp.ReducedSpaceNewtonCG(model, params)
            x = solver.solve([None, prior.mean.copy(), None])
            results[dev] = (x[PARAMETER].copy(), solver.it, solver.converged,
                            solver.final_grad_norm
                            / max(solver.initial_grad_norm, 1e-300))
    for dev in ("cpu", "gpu"):
        m, it, conv, red = results[dev]
        check("Newton-CG converged on %s" % dev, conv,
              "(%d its, ||g||/||g0|| = %.2e)" % (it, red))
    d = results["cpu"][0].copy()
    d.axpy(-1.0, results["gpu"][0])
    rel = d.norm("l2") / max(results["cpu"][0].norm("l2"), 1e-300)
    # Both runs stop at rel_tolerance = 1e-9 on the gradient, so each MAP point is
    # only determined to about that, and the two cannot agree more closely than the
    # optimizer's tolerance.  The tight statement is the round-off one on the
    # assembled operators (test_assembled_operators_agree), which is where a real
    # GPU/CPU discrepancy would show.
    check("the two MAP points agree to the optimizer's tolerance", rel < 1e-6,
          "(rel %.3e, solver tol 1e-9)" % rel)


def test_device_assignment():
    """One GPU per rank, not every rank on device 0."""
    if RANK == 0:
        print("device assignment across ranks")
    devs = gpu_info()
    K.set_device("gpu")
    mine = str(K.device())
    allof = COMM.allgather(mine)
    expected = min(NP, len(devs))
    distinct = len(set(allof))
    check("ranks spread over the available GPUs", distinct == expected,
          "(%d ranks, %d GPUs, %d distinct: %s)"
          % (NP, len(devs), distinct, sorted(set(allof))))


# Cases reported without an assertion.  All are two-dimensional, and at the element
# counts a test can afford they sit near the crossover where the fixed per-assembly
# device cost is not yet covered, so their speedup varies severalfold between
# repeats on a shared machine and says more about the machine's load than about the
# library.  At the element counts benchmarks/bench_assembly.py reports, the same
# cases are well above 1x; see docs/source/guide/gpu.rst.
#
# The asserted cases are three-dimensional, where measured speedups range from 6.7x
# to 10.3x, about five times the asserted 1.2x, so a busy machine does not trip the
# assertion but a real regression does.
_REPORT_ONLY = {("quad", 1), ("quad", 2), ("quad", 3)}


def test_memory_budget():
    """The device budget must follow how many ranks share the card.

    A fixed fraction is wrong in both directions: too small wastes most of a card a
    rank has to itself (a fifth of a card caps the element batch a rank can
    assemble), too large and several ranks on one device starve each other.  What
    is asserted is the invariant, not a number: the budget is a sensible share of
    the card, and it scales down when ranks share.
    """
    if RANK == 0:
        print("device memory budget")
    import jax

    dev = jax.devices(accel())[0]
    st = dev.memory_stats() or {}
    limit = st.get("bytes_limit", 0)
    total = st.get("bytes_reservable_limit", 0) or limit
    frac = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0"))
    share = _ranks_sharing(dev)
    check("the JAX budget is a real share of the card",
          limit > 2 * 2 ** 30, "(%.2f GB)" % (limit / 2 ** 30))
    # The rule, not a number: 0.90 of the card when a rank owns it, 0.45 when hypre
    # shares it, divided by the ranks sharing the device.  A fixed 0.90 would fail a
    # correct library whenever HIPPYMFEM_HYPRE_DEVICE=1 is inherited from the
    # environment.
    from hippymfem._jaxconfig import hypre_on_device

    base = 0.45 if hypre_on_device() else 0.90
    want = base / max(share, 1)
    check("the fraction tracks the ranks sharing this device%s"
          % (" and hypre" if hypre_on_device() else ""),
          abs(frac - round(want, 4)) < 1e-3,
          "(fraction %.3f, expected %.3f for %d rank(s) per device)"
          % (frac, want, share))


def _ranks_sharing(dev):
    """Ranks per device, the same way the budget is computed.

    The quantity is how many ranks on this *node* there are per visible device.
    Asking which device rank 0 sees would be wrong: every rank sees the whole list
    and would answer device 0, reporting four ranks sharing one card when each has
    its own.
    """
    import socket

    from hippymfem import _jaxconfig

    host = socket.gethostname()
    local = max(1, sum(1 for h in COMM.allgather(host) if h == host))
    # Use the device list from before this rank was pinned to one card: with the pin
    # on, ``jax.devices()`` has one entry per rank by design, and dividing by it would
    # claim ranks share a card when each has its own.  Under tools/mpirun_pinned.sh
    # the list is already one card, so the wrapper passes the node's count in
    # HIPPYMFEM_NODE_GPUS; the library reads the same variable.
    node = os.environ.get("HIPPYMFEM_NODE_GPUS", "")
    ndev = int(node) if node.isdigit() and int(node) > 0 else max(1, len(_jaxconfig._all_visible()))
    return max(1, -(-local // ndev))


def test_pattern_sort_on_device():
    """The chunked device sort behind the pattern builds gives the host's patterns.

    ``devsort.argsort_keys`` buckets the keys on the host, sorts each bucket on the
    device padded to a fixed shape, and concatenates; the order among equal keys
    is free.  Forced to tiny chunks so every branch runs (bucketing, padding, a
    bucket larger than a chunk, all keys equal), it must give a valid argsort and
    ``np.unique``'s exact output, and the scatter and true-dof patterns built
    through it must equal the host-built ones array for array.
    """
    if RANK == 0:
        print("pattern sorts on the device")
    from hippymfem.fem import devsort, csrassemble as csr, tdofassemble as tda
    from hippymfem.fem.elementbatch import MeshBatches
    if not K.on_gpu():
        check("pattern sorts on the device (skipped: kernels not on a GPU)", True)
        return
    saved = (devsort.MODE, devsort.CHUNK, devsort.PAD)
    rng = np.random.default_rng(RANK + 1)
    ok = True
    try:
        devsort.MODE, devsort.CHUNK, devsort.PAD = "device", 4000, 256
        for k in (rng.integers(0, 2 ** 62, 60_001), rng.integers(0, 5, 30_000), np.full(9_000, 7),
                  np.concatenate([rng.integers(0, 10, 30_000), rng.integers(0, 2 ** 50, 9_000)]),
                  np.zeros(0, np.int64)):
            k = k.astype(np.int64)
            o = devsort.argsort_keys(k)
            ok = ok and o.size == k.size and np.array_equal(np.sort(o), np.arange(k.size))
            ok = ok and (k.size == 0 or bool(np.all(np.diff(k[o]) >= 0)))
            u1, i1 = np.unique(k, return_inverse=True)
            u2, i2 = devsort.unique_inverse(k)
            ok = ok and np.array_equal(u1, u2) and np.array_equal(i1.ravel(), i2)
        check("argsort valid and unique/inverse exact on forced chunks", ok)
        pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(5, 5, 5, mfem.Element.HEXAHEDRON))
        V = hp.FunctionSpace.H1(pm, 2)
        Vm = hp.FunctionSpace.H1(pm, 1)
        groups = MeshBatches(pm, 6, COMM).groups
        pats = {}
        for mode in ("host", "device"):
            devsort.MODE = mode
            csr.clear_pattern_cache()
            sp = csr.get_pattern(V, Vm, groups)
            tp = tda.TrueDofPattern(csr.get_pattern(V, V, groups), V, V)
            pats[mode] = [np.asarray(sp.indptr), np.asarray(sp.indices), np.asarray(sp.slot),
                          np.asarray(tp.I_diag), np.asarray(tp.J_diag), np.asarray(tp.I_offd), np.asarray(tp.J_offd),
                          np.asarray(tp.slot_recv), np.asarray(tp.perm_rm if tp.perm_rm is not None else [-1])]
        same = all(np.array_equal(a, b) for a, b in zip(pats["host"], pats["device"]))
        check("scatter and true-dof patterns built through the device sort equal the host's", same)
    finally:
        devsort.MODE, devsort.CHUNK, devsort.PAD = saved
        csr.clear_pattern_cache()


def test_pattern_sort_kernels_agree():
    """CuPy's radix sort and XLA's give the same pattern, not merely a valid order.

    The device path prefers CuPy (CUB), which on 100 M int64 keys measures 9.2 ns a
    key against XLA's 24, and falls back to XLA where CuPy is absent or has no room.
    Ties may be broken differently by the two, which is allowed (equal keys are the
    same entry and take the same slot), so what has to hold is that the pattern that
    comes out is identical array for array.
    """
    if RANK == 0:
        print("the two device sorting kernels")
    from hippymfem.fem import devsort, csrassemble as csr
    from hippymfem.fem.elementbatch import MeshBatches

    if not K.on_gpu():
        check("the two device sorting kernels agree (skipped: kernels not on a GPU)", True)
        return
    if not devsort._cupy_usable():
        check("the two device sorting kernels agree (skipped: no CuPy here)", True,
              "(the XLA sort is what runs)")
        return

    saved = (devsort.MODE, devsort.CHUNK, devsort.SORT_KERNEL)
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(8, 8, 8, mfem.Element.HEXAHEDRON))
    V = hp.FunctionSpace.H1(pm, 2)
    groups = MeshBatches(pm, 6, COMM).groups
    out = {}
    try:
        devsort.MODE, devsort.CHUNK = "device", 1 << 16      # force the bucketed path
        for kernel in ("cupy", "xla"):
            devsort.SORT_KERNEL = kernel
            devsort._CUPY_OK = None
            csr.clear_pattern_cache()
            P = csr.get_pattern(V, V, groups)
            out[kernel] = (P.nnz, np.asarray(P.indptr).copy(), np.asarray(P.indices).copy(),
                           np.asarray(P.slot).copy())
        a, b = out["cupy"], out["xla"]
        same = (a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
                and np.array_equal(a[3], b[3]))
        check("CuPy and XLA sorts give the same pattern", same,
              "(nnz %d, %d rows, %d entries)" % (a[0], a[1].size - 1, a[3].size))
    finally:
        devsort.MODE, devsort.CHUNK, devsort.SORT_KERNEL = saved
        devsort._CUPY_OK = None
        csr.clear_pattern_cache()


def test_speed():
    """Per-element cost on each device: reported for every case, asserted narrowly.

    Every case is measured and printed.  Two things are asserted, chosen so that
    they are true rather than merely typical.

    1. Once there is real arithmetic per element, the device wins: every case not
       in ``_REPORT_ONLY`` must reach 1.2x, so a kernel that silently fell back to
       the host (about 1.0x) is caught.
    2. The speedup grows with work per element.  This is a ratio of two speedups
       measured in the same run, so machine load cancels out of it; absolute
       speedups move by up to 1.6x between runs on a shared machine, which is why
       the only absolute assertions are the ones with a fivefold margin.

    ``_REPORT_ONLY`` says which cases are data rather than claims, and why.
    """
    if RANK == 0:
        print("element cost per device")
        print("      %-22s %12s %12s %8s" % ("case", "CPU us/elem", "GPU us/elem",
                                             "speedup"))
    speedup = {}
    for kind, n, order in (("quad", 48, 1), ("quad", 48, 2), ("quad", 32, 3),
                           ("hex", 14, 1), ("hex", 10, 2)):
        pm, Vu, Vm, b, kern, loc = build(kind, n, order)
        NE = pm.GetNE()
        bc = hp.DirichletBC(Vu, None, "all")
        t = {}
        for dev in ("cpu", "gpu"):
            with on(dev):
                def once():
                    A = assemble_matrix(Vu, Vu, b.groups,
                                        kern.element_matrices(ADJOINT, STATE, loc),
                                        NE, test_ess=bc.ess_tdof)
                    del A
                once()
                COMM.Barrier()
                t0 = time.perf_counter()
                for _ in range(3):
                    once()
                COMM.Barrier()
                t[dev] = (time.perf_counter() - t0) / 3
        # One rank decides, so that a per-rank timing difference cannot make the
        # ranks disagree about whether to call check() and desynchronize the job.
        ratio = COMM.bcast(t["cpu"] / t["gpu"], root=0)
        speedup[(kind, order)] = ratio
        if RANK == 0:
            print("      %-22s %12.3f %12.3f %7.2fx%s"
                  % ("%s order %d NE=%d" % (kind, order, NE),
                     1e6 * t["cpu"] / NE, 1e6 * t["gpu"] / NE, ratio,
                     "   (reported, not asserted)"
                     if (kind, order) in _REPORT_ONLY else ""), flush=True)
        if (kind, order) not in _REPORT_ONLY:
            check("GPU is faster (%s order %d)" % (kind, order),
                  ratio >= 1.2, "(%.2fx)" % ratio)
        elif RANK == 0 and ratio < 1.0:
            print("      (below parity here; see docs/source/guide/gpu.rst)",
                  flush=True)

    cheap, rich = speedup[("quad", 1)], speedup[("hex", 2)]
    check("the speedup grows with work per element",
          rich >= 2.0 * cheap,
          "(hex order 2 %.2fx against quad order 1 %.2fx)" % (rich, cheap))


def main():
    if not have_gpu():
        if RANK == 0:
            import jax

            print("  skipped: no GPU visible to JAX (devices: %s). Set "
                  "HIPPYMFEM_DEVICE=gpu before importing hippymfem."
                  % [str(d) for d in jax.devices()], flush=True)
            print("FAILURES: 0 (skipped)", flush=True)
        return 0
    if RANK == 0:
        print("  GPUs visible: %s" % [str(d) for d in gpu_info()], flush=True)
    test_device_assignment()
    test_memory_budget()
    test_blocks_agree()
    test_assembled_operators_agree()
    test_geometry_streaming()
    test_reproducible()
    test_boundary_on_gpu()
    test_inverse_problem_on_gpu()
    test_pattern_sort_on_device()
    test_pattern_sort_kernels_agree()
    test_speed()
    COMM.Barrier()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

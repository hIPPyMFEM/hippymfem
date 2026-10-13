# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Direct-CSR assembly against the reference callback route, and matrix ownership.

The direct-CSR path (:mod:`hippymfem.fem.csrassemble`) replaces MFEM's
per-element Python callback with one vectorized scatter plus a hypre ``P^T A P``.
It is the default, so it is held against the callback route: every block of
every shape, on every geometry, under every diagonal policy, must come out
*identical*, not close.

Two things here are regression tests for decisions rather than for code:

* the pattern is built once and reused, so a slot that is exactly zero at the
  first assembly must still be available at the second.  ``test_structural_zeros``
  builds that situation on purpose.
* PyMFEM leaves ``%newobject`` off ``RAP``, ``ParAdd`` and the ``Eliminate*``
  methods, so their results leak unless Python is given ownership.  The assembly
  path calls all three on every iteration, so ``test_no_leak`` watches RSS.

Run with ``mpirun -n N python -m hippymfem.test.test_assembly``.
"""

import gc
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

import hippymfem as hp
from hippymfem.fem import assemble as asm
from hippymfem.test import reference_assembly as ref
from hippymfem.fem import csrassemble as csr
from hippymfem.fem import parmat
from hippymfem.fem.csrassemble import clear_pattern_cache, get_pattern
from hippymfem.fem.elementbatch import MeshBatches
from hippymfem.fem import kernel as kernel_mod
from hippymfem.fem.kernel import QuadratureKernel
from hippymfem.fem.spaces import FunctionSpace
from hippymfem.common.linalg import _local_diag, hypre_to_scipy, to_dense
from hippymfem.common.parvector import allreduce_extreme
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


def largest(values):
    """The largest of ``values``, and NaN where one of them is NaN.  Python's ``max`` keeps
    the argument it has unless the next one compares greater, which a NaN never does:
    ``max(0.0, nan)`` is 0.0, and a check ``max(...) < tol`` passes on a result that is
    not a number anywhere but in the first place."""
    values = [float(v) for v in values]
    return float("nan") if any(v != v for v in values) else max(values)


def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1])
    return 0


def make_mesh(kind, n):
    if kind == "tri":
        m = mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE)
    elif kind == "quad":
        m = mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.QUADRILATERAL)
    elif kind == "tet":
        m = mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.TETRAHEDRON)
    elif kind == "hex":
        m = mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON)
    elif kind == "mixed":
        from .mixed_mesh import write_mixed_mesh

        # sized so that every rank gets elements: MFEM's partitioner hangs on a
        # mesh with fewer elements than ranks
        nn = max(8, 4 * NP)
        path = "/tmp/_hippymfem_mixed_n%d.mesh" % nn
        if RANK == 0:
            write_mixed_mesh(path, nn)
        COMM.Barrier()
        m = mfem.Mesh(path, 1, 1)
    else:
        raise ValueError(kind)
    return mfem.ParMesh(COMM, m)


def build(kind="quad", n=4, order=1, morder=1):
    """A problem whose residual exercises every derivative block."""
    pm = make_mesh(kind, n)
    Vu = FunctionSpace.H1(pm, order)
    Vm = FunctionSpace.H1(pm, morder)
    batches = MeshBatches(pm, 2 * order + 2, COMM)

    def res(u, m, p, x):
        return (jnp.exp(m.val) * jnp.dot(u.grad, p.grad)
                + u.val * u.val * m.val * p.val)

    K = QuadratureKernel(res, [Vu, Vm, Vu], batches)
    return pm, [Vu, Vm, Vu], batches, K


def locals_at(spaces, seed=7):
    hp.parRandom.set_seed(seed)
    out = []
    for sp in spaces:
        v = sp.vector()
        hp.parRandom.normal(1.0, v)
        out.append(sp.local_values(v))
    return out


def both_backends(fn):
    """``fn(module)`` with MFEM's callback assembly (the reference) and with the
    library's direct scatter: ``(reference, direct)``."""
    return fn(ref), fn(asm)


# ------------------------------------------------------------------ equality
def test_sortfree_pattern():
    """The sort-free pattern build gives the sort route's arrays, exactly.

    ``patternbuild`` groups the entries of a pattern by row and orders each short row on
    its own instead of sorting one key per entry, several times faster on a large mesh.
    The output must not differ in any respect: nnz, indptr, indices (diagonal first on a
    square pattern), the slot map and the dtypes of all three.  The true-dof pattern
    built on top merges its own and its received entries the same way and lays out
    hypre's blocks with a compiled pass, so every array it keeps is compared too.  Forced
    on here so that it runs on small meshes, which by default take the sort route.  Also
    checks that its thread count is this rank's share of the node, never numba's default
    of every core.
    """
    if RANK == 0:
        print("the sort-free pattern build")
    from hippymfem.fem import pattern as pat, patternbuild as pb
    from hippymfem.fem.elementbatch import MeshBatches
    from hippymfem.fem.tdofassemble import TrueDofPattern

    # every array the true-dof pattern keeps for its assemblies
    tdof_arrays = ("I_diag", "J_diag", "I_offd", "J_offd", "cmap", "tslot", "slot_recv")
    tdof_counts = ("nnz_t", "nsend", "nrecv", "nnz_diag", "n_offd")

    def same_arrays(xs, ys):
        return all((x is None and y is None) or (
            x is not None and y is not None and x.dtype == y.dtype and np.array_equal(x, y))
            for x, y in zip(xs, ys))

    if not pb.usable():
        check("sort-free pattern (skipped: numba is not importable)", True)
        return
    saved_mode = pb.MODE
    saved_lean = (pat.LEAN, pat.LEAN_MIN_ENTRIES, pat._COMPACT_SLAB)
    worst = 0
    worst_t = 0
    worst_l = 0
    ncase = 0
    released = 0
    try:
        # "mixed" has two element groups, so it runs the reordering into mesh element
        # order that precedes the build, and runs of two lengths
        for kind, n, pu, pmo in (("tri", 6, 2, 1), ("quad", 5, 3, 2), ("hex", 3, 2, 1),
                                 ("tet", 3, 2, 2), ("mixed", 0, 2, 1)):
            pm = make_mesh(kind, n)
            Vu, Vm = FunctionSpace.H1(pm, pu), FunctionSpace.H1(pm, pmo)
            groups = MeshBatches(pm, 2 * pu + 2, COMM).groups
            for tst, trl in ((Vu, Vu), (Vu, Vm), (Vm, Vm)):
                got, tdof, lean = {}, {}, {}
                # "lean" is the build from the dof tables and row by row, with the maps
                # given up and made again (pattern.LEAN), forced on these small patterns
                # and with the compact map made in slabs of a few elements
                for mode in ("sort", "numba", "lean"):
                    pb.MODE, pb._USABLE = ("numba" if mode == "lean" else mode), None
                    pat.LEAN = mode == "lean"
                    pat.LEAN_MIN_ENTRIES, pat._COMPACT_SLAB = (
                        (0, 1 << 9) if mode == "lean" else saved_lean[1:])
                    pat.clear_pattern_cache()
                    P = pat.get_pattern(tst, trl, groups)
                    got[mode] = (P.nnz, np.asarray(P.indptr).copy(),
                                 np.asarray(P.indices).copy(), np.asarray(P.slot).copy())
                    T = TrueDofPattern(P, tst, trl)
                    rm = getattr(T, "_rm", None) or ()
                    tdof[mode] = ([getattr(T, c) for c in tdof_counts],
                                  [np.asarray(getattr(T, c)).copy() for c in tdof_arrays]
                                  + [None if x is None else np.asarray(x).copy() for x in rm])
                    if mode == "lean" and P._slot is None:
                        released += 1
                    # the maps after the scatter has what it needs of them: the compact
                    # map, the target's slots (given up by then, when lean) and the ldof
                    # pattern's, made again
                    cm = pat._compact_slots(T.target.slot, T.target.shapes, T.target.sizes)
                    if mode == "lean" and T.target.nnz:
                        T.target.release_slot()
                    rmask = np.zeros(P.nrow, dtype=bool)
                    rmask[::3] = True
                    cmask = np.zeros(P.ncol, dtype=bool)
                    cmask[1::4] = True
                    lean[mode] = ([np.asarray(T.target.slot).copy(), np.asarray(P.slot).copy(),
                                   P.masked_slots(rmask, cmask).copy(),
                                   P.masked_slots(None, cmask).copy()]
                                  + ([None, None] if cm is None else
                                     [cm[0].copy(), cm[1].copy(), np.array([cm[2]])]))
                a, b = got["sort"], got["numba"]
                same = (a[0] == b[0] and all(x.dtype == y.dtype and np.array_equal(x, y)
                                             for x, y in zip(a[1:], b[1:])))
                worst = max(worst, 0 if same else 1)
                ta, tb = tdof["sort"], tdof["numba"]
                same_t = (ta[0] == tb[0] and len(ta[1]) == len(tb[1])
                          and same_arrays(ta[1], tb[1]))
                worst_t = max(worst_t, 0 if same_t else 1)
                # lean against the route it replaces: every array and its type, but for
                # ``tslot``, which lean keeps as wide as the slots need and no wider
                c, tc = got["lean"], tdof["lean"]
                k = tdof_arrays.index("tslot")
                same_l = (b[0] == c[0]
                          and all(x.dtype == y.dtype and np.array_equal(x, y)
                                  for x, y in zip(b[1:], c[1:]))
                          and tb[0] == tc[0] and len(tb[1]) == len(tc[1])
                          and same_arrays(tb[1][:k] + tb[1][k + 1:], tc[1][:k] + tc[1][k + 1:])
                          and np.array_equal(tb[1][k], tc[1][k])
                          and tc[1][k].dtype == (np.int32 if tc[0][0] + tc[0][1] < 2 ** 31
                                                 else np.int64)
                          and len(lean["numba"]) == len(lean["lean"])
                          and same_arrays(lean["numba"], lean["lean"]))
                worst_l = max(worst_l, 0 if same_l else 1)
                ncase += 1
    finally:
        pb.MODE, pb._USABLE = saved_mode, None
        pat.LEAN, pat.LEAN_MIN_ENTRIES, pat._COMPACT_SLAB = saved_lean
        pat.clear_pattern_cache()
    worst = COMM.allreduce(worst, op=MPI.MAX)
    check("sort-free pattern identical to the sorted one", worst == 0,
          "(%d space pairs, triangles to tetrahedra, square and rectangular)" % ncase)
    worst_t = COMM.allreduce(worst_t, op=MPI.MAX)
    check("its true-dof pattern identical too", worst_t == 0,
          "(%d space pairs on %d ranks)" % (ncase, COMM.size))
    worst_l = COMM.allreduce(worst_l, op=MPI.MAX)
    released = COMM.allreduce(released, op=MPI.SUM)
    check("the same arrays from the dof tables and row by row, maps given up and made again",
          worst_l == 0 and released > 0,
          "(%d space pairs on %d ranks; the ldof map given up %d times)"
          % (ncase, COMM.size, released))

    import os
    old = os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")
    try:
        os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"] = str(max((os.cpu_count() or 1), 1))
        t = pb.threads()
    finally:
        if old is None:
            os.environ.pop("OMPI_COMM_WORLD_LOCAL_SIZE", None)
        else:
            os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"] = old
    check("a full node of ranks gets one pattern thread each", t == 1 or pb.THREADS > 0,
          "(%d thread with %d ranks on %d cores)" % (t, os.cpu_count() or 1, os.cpu_count() or 1))
    # numba's cache of the compiled passes: next to the sources by default, where the ranks
    # of a run on a shared file system rewrite it under one another
    where = os.path.realpath(pb._passes().runs.stats.cache_path)
    want = os.environ.get("NUMBA_CACHE_DIR") or pb.cache_dir()
    here = os.path.dirname(os.path.realpath(pb.__file__))
    check("the compiled passes are cached on this node, or where NUMBA_CACHE_DIR says",
          where.startswith(os.path.realpath(want)) if want else where.startswith(here),
          "(%s%s)" % (where, "" if want else "; no directory of this user under the temporary one"))


def test_blocks_identical():
    """Every block, every geometry, every policy: bit-identical to the callback route."""
    if RANK == 0:
        print("direct-CSR assembly vs the callback route")
    cases = [("quad", 1, 1), ("tri", 1, 1), ("quad", 2, 1), ("tri", 2, 2),
             ("tet", 1, 1), ("hex", 1, 1), ("mixed", 1, 1)]
    worst = 0.0
    for kind, order, morder in cases:
        n = 3 if kind in ("tet", "hex") else 4
        pm, spaces, batches, K = build(kind, n, order, morder)
        NE = pm.GetNE()
        loc = locals_at(spaces)
        bc = hp.DirichletBC(spaces[STATE], None, "all")
        bcm = hp.DirichletBC(spaces[PARAMETER], None, "all")
        ess, essm = bc.ess_tdof, bcm.ess_tdof

        blocks = [
            ("A", ADJOINT, STATE, ess, None, "one"),
            ("W_uu zero-diag", STATE, STATE, ess, None, "zero"),
            ("W_uu free", STATE, STATE, None, None, "one"),
            ("C", ADJOINT, PARAMETER, ess, None, "one"),
            ("W_um", STATE, PARAMETER, ess, essm, "one"),
            ("W_mu", PARAMETER, STATE, essm, ess, "one"),
            ("W_mm", PARAMETER, PARAMETER, None, None, "one"),
        ]
        bad = []
        for name, i, j, te, tr, pol in blocks:
            mats = K.element_matrices(i, j, loc)
            A1, A2 = both_backends(lambda a: a.assemble_matrix(
                spaces[i], spaces[j], batches.groups, mats, NE,
                test_ess=te, trial_ess=tr, diag_policy=pol))
            D1, D2 = to_dense(A1, COMM), to_dense(A2, COMM)
            scale = max(float(np.abs(D1).max()), 1e-300)
            err = float(np.abs(D1 - D2).max()) / scale
            worst = max(worst, err)
            if err != 0.0:
                bad.append("%s %.2e" % (name, err))
        for var in (STATE, PARAMETER, ADJOINT):
            vecs = K.element_vectors(var, loc)
            v1, v2 = both_backends(lambda a: a.assemble_vector(
                spaces[var], batches.groups, vecs, NE))
            d = np.concatenate(COMM.allgather(v1.array - v2.array))
            if np.abs(d).max() != 0.0:
                bad.append("vector %d %.2e" % (var, np.abs(d).max()))
        check("%s order %d/%d: all blocks identical" % (kind, order, morder),
              not bad, "" if not bad else "differs: " + ", ".join(bad))
    check("worst difference over all blocks is exactly zero", worst == 0.0,
          "(%.3e)" % worst)


def test_structural_zeros():
    """A slot that is zero at the first assembly must survive in the pattern.

    The pattern is built once and reused, so pruning exact zeros, which is what
    MFEM's ``skip_zeros=1`` insertion does every time it assembles, would be a
    silent wrong answer later.  Here the coefficient is chosen so that the
    off-diagonal coupling is identically zero for the first parameter and nonzero
    for the second, which is the failure mode.
    """
    if RANK == 0:
        print("pattern reuse")
    pm = make_mesh("quad", 4)
    Vu = FunctionSpace.H1(pm, 1)
    Vm = FunctionSpace.H1(pm, 1)
    batches = MeshBatches(pm, 4, COMM)

    def res(u, m, p, x):
        # m.val scales the whole operator: m = 0 makes every element matrix
        # exactly zero, so nothing at all would enter a pruned pattern.
        return m.val * jnp.dot(u.grad, p.grad)

    K = QuadratureKernel(res, [Vu, Vm, Vu], batches)
    NE = pm.GetNE()
    zero = Vm.vector()
    one = Vm.vector()
    one.set(1.0)
    loc0 = [Vu.local_values(Vu.vector()), Vm.local_values(zero),
            Vu.local_values(Vu.vector())]
    loc1 = [Vu.local_values(Vu.vector()), Vm.local_values(one),
            Vu.local_values(Vu.vector())]

    clear_pattern_cache()
    A0 = asm.assemble_matrix(Vu, Vu, batches.groups,
                             K.element_matrices(ADJOINT, STATE, loc0), NE)
    n0 = float(np.abs(to_dense(A0, COMM)).max())
    # the pattern is now cached, built from an all-zero assembly
    A1 = asm.assemble_matrix(Vu, Vu, batches.groups,
                             K.element_matrices(ADJOINT, STATE, loc1), NE)
    D1 = to_dense(A1, COMM)

    R1 = to_dense(ref.assemble_matrix(Vu, Vu, batches.groups,
                                      K.element_matrices(ADJOINT, STATE, loc1),
                                      NE), COMM)
    err = float(np.abs(D1 - R1).max()) / max(float(np.abs(R1).max()), 1e-300)
    check("zero first assembly does not prune the pattern",
          n0 == 0.0 and err == 0.0,
          "(|A(m=0)|=%.1e, then |A(m=1)-ref|/|ref|=%.1e)" % (n0, err))

    # and a Laplacian on right triangles, whose element matrices have exact
    # zeros from orthogonal gradients: those slots must be usable when an
    # anisotropic coefficient fills them in
    pmt = make_mesh("tri", 4)
    Vt = FunctionSpace.H1(pmt, 1)
    Vp = FunctionSpace.H1(pmt, 1)
    bt = MeshBatches(pmt, 4, COMM)
    TH = jnp.array([[2.0, 0.9], [0.9, 0.5]])

    def aniso(u, m, p, x):
        return jnp.dot(u.grad, TH @ p.grad) * m.val

    Kt = QuadratureKernel(aniso, [Vt, Vp, Vt], bt)
    onet = Vp.vector()
    onet.set(1.0)
    loct = [Vt.local_values(Vt.vector()), Vp.local_values(onet),
            Vt.local_values(Vt.vector())]
    iso = QuadratureKernel(lambda u, m, p, x: jnp.dot(u.grad, p.grad) * m.val,
                           [Vt, Vp, Vt], bt)
    clear_pattern_cache()
    asm.assemble_matrix(Vt, Vt, bt.groups,
                        iso.element_matrices(ADJOINT, STATE, loct), pmt.GetNE())
    Da = to_dense(asm.assemble_matrix(
        Vt, Vt, bt.groups, Kt.element_matrices(ADJOINT, STATE, loct),
        pmt.GetNE()), COMM)
    Ra = to_dense(ref.assemble_matrix(
        Vt, Vt, bt.groups, Kt.element_matrices(ADJOINT, STATE, loct),
        pmt.GetNE()), COMM)
    e2 = float(np.abs(Da - Ra).max()) / max(float(np.abs(Ra).max()), 1e-300)
    check("isotropic pattern carries an anisotropic refill", e2 == 0.0,
          "(%.3e)" % e2)


def test_keep_policy():
    """``diag_policy="keep"`` must keep the diagonal, in parallel too.

    The callback route cannot: MFEM's ``DiagonalPolicy`` is honored only on the
    serial ``SparseMatrix``, and hypre's ``EliminateRowsCols`` always writes 1.0.
    The direct path saves and restores the entries, so this is the one place the
    two routes are *meant* to differ.
    """
    if RANK == 0:
        print("diagonal policies")
    pm = make_mesh("quad", 4)
    V = FunctionSpace.H1(pm, 1)
    Vm = FunctionSpace.H1(pm, 1)
    b = MeshBatches(pm, 4, COMM)
    K = QuadratureKernel(lambda u, m, p, x: jnp.dot(u.grad, p.grad) + u.val * p.val,
                         [V, Vm, V], b)
    loc = locals_at([V, Vm, V])
    mats = K.element_matrices(ADJOINT, STATE, loc)
    bc = hp.DirichletBC(V, None, "all")
    ess = np.asarray(bc.ess_tdof.ToList(), dtype=np.int64)

    free = to_dense(asm.assemble_matrix(V, V, b.groups, mats, pm.GetNE()), COMM)
    got = {}
    for pol in ("one", "zero", "keep"):
        A = asm.assemble_matrix(V, V, b.groups, mats, pm.GetNE(),
                                test_ess=bc.ess_tdof, diag_policy=pol)
        got[pol] = to_dense(A, COMM)

    # gather the global indices of this rank's essential dofs
    lo = V.vector().owner_range[0]
    gess = np.concatenate(COMM.allgather(ess + lo)) if ess.size or NP > 1 else ess
    gess = np.unique(gess.astype(np.int64))
    if RANK == 0 and gess.size:
        d_one = got["one"][gess, gess]
        d_zero = got["zero"][gess, gess]
        d_keep = got["keep"][gess, gess]
        d_free = free[gess, gess]
        ok = (np.all(d_one == 1.0) and np.all(d_zero == 0.0)
              and np.allclose(d_keep, d_free, rtol=0, atol=0))
        detail = "(one=%.1f zero=%.1f keep matches free: %s)" % (
            d_one.max(), np.abs(d_zero).max(), np.array_equal(d_keep, d_free))
    else:
        ok, detail = True, ""
    ok = bool(COMM.bcast(ok, root=0))
    detail = COMM.bcast(detail, root=0)
    check("diag policies one/zero/keep", ok, detail)

    # the off-diagonal part of the essential rows must be cleared by all three
    if RANK == 0 and gess.size:
        rows = got["keep"][gess, :].copy()
        rows[np.arange(gess.size), gess] = 0.0
        cleared = float(np.abs(rows).max()) == 0.0
    else:
        cleared = True
    check("essential rows cleared under 'keep'", bool(COMM.bcast(cleared, root=0)))


def test_partial_boundary_elimination():
    """Essential conditions on part of the boundary, where some rank owns none.

    Guards against two failures that stay invisible below about eight ranks:

    * skipping the elimination on a rank whose local essential list is empty makes
      a **collective** conditional on rank-local state and desynchronizes the job;
    * ``hypre_ParCSRMatrixEliminateAAe`` builds the eliminated part ``Ae`` sharing
      state with ``A``, so freeing ``Ae`` damages ``A``, silently, until something
      walks its parallel structure, such as a transpose.

    The test therefore asserts that some rank owns no essential dofs (otherwise it
    is not exercising the case), then transposes and multiplies, and compares
    against the callback route.
    """
    if RANK == 0:
        print("essential conditions on part of the boundary")
    pm = make_mesh("quad", max(8, 4 * NP))
    Vu = FunctionSpace.H1(pm, 2)
    Vm = FunctionSpace.H1(pm, 1)
    b = MeshBatches(pm, 6, COMM)
    K = QuadratureKernel(lambda u, m, p, x: jnp.exp(m.val) * jnp.dot(u.grad, p.grad),
                         [Vu, Vm, Vu], b)
    loc = locals_at([Vu, Vm, Vu])
    mats = K.element_matrices(ADJOINT, STATE, loc)
    NE = pm.GetNE()
    # one edge only, so the ranks away from it own no essential dofs
    bc = hp.DirichletBC(Vu, None, bdr_attributes=[1])
    counts = COMM.allgather(int(bc.ess.size))
    empty = sum(1 for c in counts if c == 0)
    check("some rank owns no essential dofs (the case being tested)",
          NP == 1 or empty > 0,
          "(local counts %s)" % (counts if NP <= 8 else
                                 "%d of %d ranks empty" % (empty, NP)))

    A1, A2 = both_backends(lambda a: a.assemble_matrix(
        Vu, Vu, b.groups, mats, NE, test_ess=bc.ess_tdof, diag_policy="one"))
    D1, D2 = to_dense(A1, COMM), to_dense(A2, COMM)
    e = float(np.abs(D1 - D2).max()) / max(float(np.abs(D1).max()), 1e-300)
    check("partial-boundary elimination matches the callback route", e == 0.0,
          "(%.3e)" % e)

    # a transpose walks the parallel structure the damage would corrupt
    A = asm.assemble_matrix(Vu, Vu, b.groups, mats, NE, test_ess=bc.ess_tdof,
                            diag_policy="one")
    At = A.Transpose()
    x = hp.ParVector(COMM, At.Width())
    x.set(1.0)
    y = hp.ParVector(COMM, At.Height())
    At.Mult(x.hypre, y.hypre)
    DT = to_dense(At, COMM)
    sym = float(np.abs(DT - D2.T).max()) / max(float(np.abs(D2).max()), 1e-300)
    check("transposing the eliminated matrix works and is exact",
          np.isfinite(y.norm("l2")) and sym == 0.0,
          "(||A^T 1|| = %.6f, transpose error %.3e)" % (y.norm("l2"), sym))


def test_folded_elimination():
    """Folding the elimination into the scatter must change nothing at all.

    Essential-dof elimination is applied by masking the local CSR slots before
    the parallel triple product rather than by calling ``EliminateRowsCols``
    after it, which saves a large share of a GPU assembly.  The two are equal only
    because the prolongation is boolean, so the local mask is the pullback of the
    true-dof one; this asserts the equality directly, against MFEM's own calls, on
    every block and diagonal policy.
    """
    if RANK == 0:
        print("elimination folded into the scatter")
    for kind, order in (("quad", 1), ("quad", 2), ("tri", 2), ("hex", 2)):
        pm, Vh, b, K = build(kind, 4 if kind != "hex" else 3, order)
        loc = locals_at(Vh)
        bc = hp.DirichletBC(Vh[0], None, "all")
        cases = (("A", ADJOINT, STATE, 0, 0, bc.ess_tdof, bc.ess_tdof, "one"),
                 ("W_uu", STATE, STATE, 0, 0, bc.ess_tdof, bc.ess_tdof, "zero"),
                 ("C", ADJOINT, PARAMETER, 0, 1, bc.ess_tdof, None, "one"),
                 ("W_um", STATE, PARAMETER, 0, 1, bc.ess_tdof, None, "one"))
        for name, i, j, ti, ri, te, tr, pol in cases:
            got = {}
            for fold in (True, False):
                old = csr.set_fold_elimination(fold)
                try:
                    A = asm.assemble_matrix(Vh[ti], Vh[ri], b.groups,
                                            K.element_matrices(i, j, loc),
                                            pm.GetNE(), test_ess=te,
                                            trial_ess=tr, diag_policy=pol)
                    got[fold] = to_dense(A, COMM)
                    del A
                finally:
                    csr.set_fold_elimination(old)
            d = float(np.abs(got[True] - got[False]).max())
            check("folded == eliminated (%s %s order %d)" % (kind, name, order),
                  d == 0.0, "(%.3e)" % d)


def test_shared_hessian_pass():
    """One differentiation pass must give exactly what five separate ones give.

    ``hess_block(i, j)`` differentiates ``grad(R)[i]`` forward and keeps column
    block ``j``; one ``jacfwd(grad(R))`` gives every block at once, for about half
    the work of the five a Newton linearization point needs.  The substitution is
    only legitimate if the numbers are bit-identical, because the rest of this
    suite asserts exact equality against MFEM downstream of it.
    """
    if RANK == 0:
        print("shared Hessian pass")
    pairs = [(ADJOINT, STATE), (ADJOINT, PARAMETER), (STATE, STATE),
             (STATE, PARAMETER), (PARAMETER, PARAMETER)]
    for kind, order in (("quad", 2), ("tri", 1), ("hex", 2)):
        pm, Vh, b, K = build(kind, 4 if kind != "hex" else 3, order)
        loc = locals_at(Vh)
        many = K.element_matrices_many(pairs, loc)
        worst = 0.0
        for (i, j) in pairs:
            one = K.element_matrices(i, j, loc)
            for a, c in zip(one, many[(i, j)]):
                worst = largest([worst, float(np.abs(np.asarray(a)
                                                     - np.asarray(c)).max())])
        check("shared pass == per-block (%s order %d)" % (kind, order),
              worst == 0.0, "(%.3e)" % worst)


def test_column_restricted_pass():
    """The pass over the columns a linearization point needs is bit-identical to
    the full one, and a residual linear in the state gets no ``W_uu`` at all.

    ``hess_cols`` pushes forward tangents for the chosen column slots only (the
    state and parameter columns for a Newton point, the parameter columns alone
    when the residual is linear in the state); per tangent the arithmetic is that
    of ``hess_all``, so the blocks must agree to the bit.  With ``is_fwd_linear``
    the state-state block is not assembled (``W_uu = p . d2R/du2 = 0``), and the
    true Hessian applied through the KKT blocks must equal the one assembled
    with the block present, which is checked here against a problem declared
    nonlinear with the same, linear, residual.
    """
    if RANK == 0:
        print("column-restricted Hessian pass")
    full = [(ADJOINT, STATE), (ADJOINT, PARAMETER), (STATE, STATE),
            (STATE, PARAMETER), (PARAMETER, PARAMETER)]
    for kind, order in (("quad", 2), ("hex", 2)):
        pm, Vh, b, K = build(kind, 4 if kind != "hex" else 3, order)
        loc = locals_at(Vh)
        ref = K.element_matrices_many(full, loc)
        for pairs in ([(ADJOINT, PARAMETER), (STATE, STATE), (STATE, PARAMETER),
                       (PARAMETER, PARAMETER)],
                      [(ADJOINT, PARAMETER), (STATE, PARAMETER), (PARAMETER, PARAMETER)]):
            sub = K.element_matrices_many(pairs, loc)
            worst = 0.0
            for ij in pairs:
                for a, c in zip(sub[ij], ref[ij]):
                    worst = largest([worst, float(np.abs(np.asarray(a) - np.asarray(c)).max())])
            cols = sorted({j for _, j in pairs})
            check("columns %s == full pass (%s order %d)" % (cols, kind, order),
                  worst == 0.0, "(%.3e)" % worst)
    # a linear residual, declared linear and not: same true Hessian, no W_uu
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(3, 3, 3, mfem.Element.HEXAHEDRON))
    Vu = hp.FunctionSpace.H1(pm, 2)
    Vm = hp.FunctionSpace.H1(pm, 1)
    varf = lambda u, m, p, x: jnp.exp(m.val) * hp.inner(u.grad, p.grad) - p.val  # noqa: E731
    bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    outs = []
    for linear in (True, False):
        pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(),
                                       is_fwd_linear=linear)
        x = [Vu.vector(), Vm.vector(), Vu.vector()]
        hp.parRandom.set_seed(5)
        for v in x:
            hp.parRandom.normal(0.3, v)
        pde.setLinearizationPoint(x, gauss_newton_approx=False)
        check("W_uu %s when is_fwd_linear=%s" % ("skipped" if linear else "assembled", linear),
              (pde.Wuu is None) == linear)
        d = Vu.vector()
        hp.parRandom.set_seed(6)
        hp.parRandom.normal(1.0, d)
        out = Vu.vector()
        pde.apply_ij(STATE, STATE, d, out)
        outs.append(out.norm("l2"))
    check("W_uu action is zero for the linear residual (assembled: %.2e, skipped: %.2e)"
          % (outs[1], outs[0]), outs[0] == 0.0 and outs[1] < 1e-14)


def test_slot_order_is_the_lexsort():
    """The linear slot construction of the true-dof pattern equals the four-key lexsort.

    ``_slot_order`` produces the order ``np.lexsort((ucol, ~lead, urow, ~in_diag))``
    would, in linear passes instead of a four-key sort.  Random row-sorted patterns,
    with rows that have no diagonal entry, empty rows, no off-diagonal entries and
    no diagonal block at all, must give the identical permutation.
    """
    if RANK == 0:
        print("true-dof slot order without a sort")
    from hippymfem.fem.tdofassemble import _slot_order
    rng = np.random.default_rng(11)
    worst = 0
    for case in range(40):
        nrow = int(rng.integers(1, 40))
        gnc = int(rng.integers(nrow, 4 * nrow + 1))
        c0 = int(rng.integers(0, gnc - nrow + 1)) if case % 5 else 0
        c1 = c0 + nrow if case % 7 else min(gnc, c0 + int(rng.integers(1, nrow + 1)))
        n = int(rng.integers(0, 12 * nrow + 1))
        keys = np.unique(rng.integers(0, nrow, n) * gnc + rng.integers(0, gnc, n))
        if case % 9 == 0:
            keys = keys[(keys % gnc >= c0) & (keys % gnc < c1)]      # diagonal block only
        if case % 11 == 0:
            keys = keys[(keys % gnc < c0) | (keys % gnc >= c1)]      # off-diagonal only
        urow, ucol = keys // gnc, keys % gnc
        in_diag = (ucol >= c0) & (ucol < c1)
        lead = in_diag & (ucol - c0 == urow) if c1 - c0 == nrow else np.zeros(keys.size, bool)
        order2 = np.lexsort((ucol, ~lead, urow, ~in_diag))
        ref = np.empty(keys.size, dtype=np.int64)
        ref[order2] = np.arange(keys.size)
        got = _slot_order(urow, ucol, in_diag, lead, nrow)
        worst = max(worst, int(np.abs(got - ref).max()) if keys.size else 0)
    check("slot order == lexsort on 40 random patterns", worst == 0)


def test_transpose_free_adjoint():
    """The adjoint through ``A.MultTranspose`` with ``A``'s hierarchy equals the explicit one.

    For a residual nonlinear in the state (``k(u) = exp(m)(1 + u^2)``) the Jacobian is
    not symmetric.  ``transpose_free_adjoint=True`` solves ``A^T p = b`` with a
    transpose wrapper preconditioned by the forward AMG; the adjoint, the gradient
    and a reduced-Hessian action must agree with the explicit transpose to the solver
    tolerance, and the wrapper must hold no hierarchy of its own.
    """
    if RANK == 0:
        print("transpose-free adjoint")
    from hippymfem.modeling.reducedHessian import ReducedHessian
    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.HEXAHEDRON))
    Vu = hp.FunctionSpace.H1(pm, 2)
    Vm = hp.FunctionSpace.H1(pm, 1)
    varf = lambda u, m, p, x: jnp.exp(m.val) * (1.0 + u.val ** 2) * hp.inner(u.grad, p.grad) - p.val  # noqa: E731
    bc = hp.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    outs = {}
    for flag in (False, True):
        pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous(), is_fwd_linear=False,
                                       transpose_free_adjoint=flag)
        for attr in ("solver", "solver_fwd_inc", "solver_adj_inc"):
            sol = hp.auto_solver(Vu, COMM, max_direct=0, method="gmres")
            sol.parameters["rel_tolerance"] = 1e-13
            sol.parameters["max_iter"] = 500
            setattr(pde, attr, sol)
        prior = hp.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
        hp.parRandom.set_seed(4)
        m = Vm.vector()
        hp.parRandom.normal(0.3, m)
        rng = np.random.default_rng(2)
        targets = np.column_stack([rng.uniform(0.2, 0.8, 30) for _ in range(3)])
        B = hp.assemblePointwiseObservation(Vu, targets)
        u = pde.generate_state()
        pde.solveFwd(u, [u, m, None])
        d = B.createVecLeft()
        B.mult(u, d)
        model = hp.Model(pde, prior, hp.DiscreteStateObservation(B, d.copy().scale(1.05), 1e-4))
        x = [u, m, model.generate_vector(ADJOINT)]
        model.solveAdj(x[ADJOINT], x)
        g = model.generate_vector(PARAMETER)
        model.evalGradientParameter(x, g)
        model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        H = ReducedHessian(model)
        v = model.generate_vector(PARAMETER)
        hp.parRandom.set_seed(9)
        hp.parRandom.normal(1.0, v)
        Hv = model.generate_vector(PARAMETER)
        H.mult(v, Hv)
        outs[flag] = (x[ADJOINT].copy(), g.copy(), Hv.copy(), pde.At, pde.solver_adj)
    def rel(a, b):
        return a.copy().axpy(-1.0, b).norm("l2") / max(b.norm("l2"), 1e-300)
    ea, eg, eh = (rel(outs[True][i], outs[False][i]) for i in range(3))
    check("adjoint, gradient and Hessian action agree with the explicit transpose "
          "(%.1e, %.1e, %.1e)" % (ea, eg, eh), ea < 1e-9 and eg < 1e-9 and eh < 1e-8)
    check("the transpose is a wrapper of A and the adjoint solver shares the forward hierarchy",
          getattr(outs[True][3], "transposed_of", None) is not None and outs[True][4] is not None
          and outs[False][4] is None)


def test_element_chunking():
    """Splitting the element batch agrees to round-off, and repeats exactly.

    The batch is split when a device cannot hold the whole of it, which the full
    element Hessian makes likelier.  The pieces are mathematically independent,
    but XLA blocks each element's quadrature sum differently at different batch
    shapes, so the concatenation is *not* bit-identical to the whole (about 1e-15
    relative), and round-off agreement is what is asserted.  What must be exact is
    repeatability at a *fixed* chunk size, since that is what a user can control.
    The automatic split only fires under memory pressure a test cannot reliably
    provoke, so the size is forced.
    """
    if RANK == 0:
        print("element-batch chunking")
    pm, Vh, b, K = build("quad", 6, 2)
    loc = locals_at(Vh)
    whole = K.element_matrices(ADJOINT, STATE, loc)
    vec_whole = K.element_vectors(ADJOINT, loc)
    old = kernel_mod.ELEMENT_CHUNK
    try:
        for chunk in (7, 1):
            kernel_mod.ELEMENT_CHUNK = chunk
            for gk in K.group_kernels:          # drop the wrappers built at old size
                gk._cache.clear()
                gk._chunk.clear()
            got = K.element_matrices(ADJOINT, STATE, loc)
            gotv = K.element_vectors(ADJOINT, loc)
            scale = max(float(np.abs(np.asarray(a)).max()) for a in whole)
            vscale = max(float(np.abs(np.asarray(a)).max()) for a in vec_whole)
            dm = max(float(np.abs(np.asarray(a) - np.asarray(c)).max())
                     for a, c in zip(whole, got))
            dv = max(float(np.abs(np.asarray(a) - np.asarray(c)).max())
                     for a, c in zip(vec_whole, gotv))
            check("chunk of %d matches the whole batch to round-off" % chunk,
                  dm <= 1e-13 * scale, "(%.3e, scale %.3e)" % (dm, scale))
            check("chunk of %d matches on vectors too" % chunk,
                  dv <= 1e-13 * vscale, "(%.3e, scale %.3e)" % (dv, vscale))
            again = K.element_matrices(ADJOINT, STATE, loc)
            rep = max(float(np.abs(np.asarray(a) - np.asarray(c)).max())
                      for a, c in zip(got, again))
            check("chunk of %d repeats exactly" % chunk, rep == 0.0,
                  "(%.3e)" % rep)
    finally:
        kernel_mod.ELEMENT_CHUNK = old
        for gk in K.group_kernels:
            gk._cache.clear()
            gk._chunk.clear()


def test_fused_scatter():
    """Scattering chunk by chunk must agree with scattering the glued array.

    The element matrices are read only by the scatter, so once the batch is split
    there is no reason to glue the chunks back together first; that glued array is
    the largest resident allocation an assembly makes.  The fused route accumulates
    in a different order, so it agrees to round-off rather than exactly.  It is
    used *only* where the batch is split, which has already given up exact
    agreement with an unsplit run, so it costs no further exactness.

    "mixed" has two element groups whose elements interleave in the mesh, so its
    pattern keeps the entries in mesh-element order and a chunk, some elements of
    one group, is not a slice of them: scattered through such a slice, every chunk
    lands in the slots of other elements, in a matrix with the right graph.
    """
    if RANK == 0:
        print("fused chunk scatter")
    for kind, order in (("quad", 2), ("tri", 1), ("hex", 2), ("mixed", 2)):
        pm, Vh, b, K = build(kind, 5 if kind != "hex" else 3, order)
        loc = locals_at(Vh)
        bc = hp.DirichletBC(Vh[0], None, "all")
        for name, ti, ri, i, j, tr in (("A", 0, 0, ADJOINT, STATE, True),
                                       ("C", 0, 1, ADJOINT, PARAMETER, False)):
            args = dict(test_ess=bc.ess_tdof,
                        trial_ess=bc.ess_tdof if tr else None)
            glued = to_dense(asm.assemble_matrix(
                Vh[ti], Vh[ri], b.groups, K.element_matrices(i, j, loc),
                pm.GetNE(), **args), COMM)
            fused = to_dense(asm.assemble_matrix(
                Vh[ti], Vh[ri], b.groups,
                (lambda i=i, j=j: K.element_matrix_chunks(i, j, loc)),
                pm.GetNE(), **args), COMM)
            scale = max(float(np.abs(glued).max()), 1e-300)
            d = float(np.abs(glued - fused).max())
            check("fused == glued to round-off (%s %s order %d)"
                  % (kind, name, order), d <= 1e-13 * scale,
                  "(%.3e, scale %.3e)" % (d, scale))
        # and the chunking is what selects it: a batch that fits is untouched
        check("a batch that fits is not chunked (%s order %d)" % (kind, order),
              not K.will_chunk(), "(will_chunk=%s)" % K.will_chunk())


def test_split_and_whole_ranks():
    """A block on ranks whose element batch is split beside ranks whose batch runs whole.

    Whether a batch is split is each rank's own answer: its element count against a
    chunk that is pinned, planned from that rank's free memory or remembered from a retry
    on that rank alone.  Ranks of the two kinds must therefore make the same collectives
    in the same order.  They did not: the finite check of a whole batch was reduced
    before the assembly and that of a split one after it, and each kind waited for the
    other, in that reduction and in the collectives of the assembly.

    The order is recorded with every rank on one side and then with every rank on the
    other, where nothing can wait.  Only where it is one order are the ranks put on
    different sides: the chunk is the element count of the smallest rank, on a mesh whose
    element count is no multiple of the rank count, so that no partition gives every rank
    that many.  The blocks must then be those of whole batches to round-off, and a block
    that is not finite on one rank must raise on all of them.  A single rank is on one
    side at a time, and the order is what it checks.  Where a single-precision hypre is
    named the Jacobian is assembled into it as well, to that precision.
    """
    import contextlib
    import sys

    from hippymfem.algorithms import singlesolve
    from hippymfem.fem.tdofassemble import single_target

    if RANK == 0:
        print("ranks with a split element batch beside ranks with a whole one")
    n = max(9, 2 * NP)
    while NP > 1 and (n * n) % NP == 0:
        n += 1
    pm = make_mesh("quad", n)
    counts = COMM.allgather(pm.GetNE())
    lo, hi = min(counts), max(counts)
    Vu, Vm = FunctionSpace.H1(pm, 2), FunctionSpace.H1(pm, 1)
    bc = hp.DirichletBC(Vu, None, "all")

    def varf(u, m, p, x):
        return jnp.exp(m.val) * jnp.dot(u.grad, p.grad) + u.val * u.val * m.val * p.val

    pde = hp.PDEVariationalProblem([Vu, Vm, Vu], varf, bc, bc.homogeneous())
    ess = pde.bc0.ess_tdof
    hp.parRandom.set_seed(7)
    x = [Vu.vector(), Vm.vector(), Vu.vector()]
    probes = [Vu.vector(), Vm.vector()]
    for v in x + probes:
        hp.parRandom.normal(1.0, v)
    # A and C applied to a vector each, and A of the single-precision hypre where one is
    # named: the trial variable, the vector, where the matrix is made, what it may differ by
    tol = kernel_mod.matrix_tolerance(1e-13)
    cases = [(STATE, probes[0], contextlib.nullcontext, tol),
             (PARAMETER, probes[1], contextlib.nullcontext, tol)]
    if singlesolve.library() is not None:
        cases.append((STATE, probes[0], single_target, max(tol, 1e-5)))
    # a parameter whose exponential overflows in the one element at the origin: the
    # block is not finite on the rank that holds it, and finite on every other
    X = Vm.coordinates()
    corner = Vm.vector()
    corner.array[(X[:, 0] < 0.5 / n) & (X[:, 1] < 0.5 / n)] = 1.0e4

    # the module's reductions and its assemblies, in the order this rank makes them
    pvp = sys.modules[type(pde).__module__]
    order, real = [], (pvp._reduce, pvp.assemble_matrix)

    def reduction(*args, **kwargs):
        order.append("reduction")
        return real[0](*args, **kwargs)

    def assembly(*args, **kwargs):
        order.append("assembly")
        return real[1](*args, **kwargs)

    def pin(chunk):
        kernel_mod.ELEMENT_CHUNK = chunk
        for gk in pde.kernel.group_kernels:
            gk._chunk.clear()          # a remembered plan wins over ELEMENT_CHUNK
        return bool(pde.kernel.will_chunk())

    def blocks(chunk):
        """The products of the cases with the chunk pinned at ``chunk`` elements, this
        rank's order of reductions and assemblies, and whether its batch was split."""
        split = pin(chunk)
        del order[:]
        out = []
        for j, v, target, _tol in cases:
            with target():
                A = pde._block(ADJOINT, j, x, test_ess=ess)
            y = Vu.vector()
            A.Mult(v.hypre, y.hypre)
            out.append(y)
            del A
        return out, list(order), split

    def refused(chunk):
        """Whether the block that is not finite raised, rank by rank."""
        pin(chunk)
        said = ""
        try:
            pde._block(ADJOINT, STATE, [x[0], corner, x[2]], test_ess=ess)
        except RuntimeError as e:
            said = str(e)
        return COMM.allgather("non-finite" in said)

    def apart(got, ref):
        # the products' differences over what each may be (a sum, in which a
        # difference that is not a number stays one)
        return sum(a.copy().axpy(-1.0, b).norm("l2") / max(b.norm("l2"), 1e-300) / case[3]
                   for a, b, case in zip(got, ref, cases))

    half = max(1, lo // 2)
    old = kernel_mod.ELEMENT_CHUNK
    pvp._reduce, pvp.assemble_matrix = reduction, assembly
    try:
        whole, order_whole, s0 = blocks(hi)         # no rank holds more than the chunk
        split, order_split, s1 = blocks(half)       # every rank holds more
        sides = COMM.allgather((s0, s1))
        same = (all(COMM.allgather(order_whole == order_split))
                and "reduction" in order_whole and "assembly" in order_whole
                and all(s == (False, True) for s in sides))
        check("a whole batch and a split one make their reductions and assemblies in one order",
              same, "(whole: %s; split: %s%s)"
              % (", ".join(order_whole), ", ".join(order_split),
                 "" if same or NP == 1 else "; ranks on different sides are not tried, "
                 "they would wait for each other"))
        d = apart(split, whole)
        check("every batch split gives the blocks of whole batches, to round-off",
              d <= 1.0, "(%.1e of the tolerance; %d products)" % (d, len(cases)))
        raised = refused(hi) + refused(half)
        check("a block that is not finite on one rank raises on every rank, whole or split",
              all(raised), "(%d of %d)" % (sum(raised), len(raised)))
        if NP == 1:
            check("one rank is on one side at a time (ranks on different sides: skipped)", True)
        elif same:
            mixed, _order, s2 = blocks(lo)          # the smallest ranks hold no more
            sides = COMM.allgather(s2)
            d = apart(mixed, whole)
            check("ranks on different sides give the blocks of whole batches, to round-off",
                  any(sides) and not all(sides) and d <= 1.0,
                  "(%d of %d ranks split, with %d to %d elements a rank; %.1e of the "
                  "tolerance)" % (sum(sides), NP, lo, hi, d))
            raised = refused(lo)
            check("and a block that is not finite on one rank raises on every rank",
                  all(raised), "(%d of %d)" % (sum(raised), len(raised)))
    finally:
        kernel_mod.ELEMENT_CHUNK = old
        pvp._reduce, pvp.assemble_matrix = real
        for gk in pde.kernel.group_kernels:
            gk._chunk.clear()


def test_auto_route_from_a_run():
    """``HIPPYMFEM_HESSIAN=auto`` keeps the route that ran faster, not one that was traced.

    A chunk loop that streams geometry reads a kernel's program before the kernel's
    first call, to copy only the arrays that the kernel reads (``kernel._used_mapped``).
    That is a trace: nothing runs in it, so nothing can be timed, and it has to leave the
    choice to the first call on arrays.  It timed the two routes on tracers instead and
    kept the winner for the run.
    """
    if RANK == 0:
        print("the route of auto is timed on arrays, not in a trace")
    pm, Vh, b, K = build("hex", 3, 2)
    loc = locals_at(Vh)
    old = kernel_mod.HESSIAN_MODE
    keys, left, taken = [], True, True
    try:
        kernel_mod.set_hessian_mode("auto")
        for gk in K.group_kernels:
            ne = int(gk.group.ne)
            fn = gk._block_fn(ADJOINT, STATE)
            dofs, pr = gk.gather(loc), kernel_mod._params(())
            args = (dofs,) + tuple(gk.mapped(streaming=True)) + (pr,)
            key = gk._route_key(STATE) + (kernel_mod.device(),)
            keys.append(key)
            kernel_mod._ROUTES.pop(key, None)
            # what a chunk loop does ahead of its first chunk (kernel._chunk_slices)
            kernel_mod._used_mapped(fn._raw, args, gk._axes(), max(1, ne // 2))
            left = left and key not in kernel_mod._ROUTES and not gk.hessian_route
            for _chunk in gk.in_chunks(fn, dofs, gk.mapped(streaming=True), pr):
                pass
            taken = taken and (not ne or kernel_mod._ROUTES.get(key) in (0, 1))
    finally:
        kernel_mod.set_hessian_mode(old)
        for key in keys:
            kernel_mod._ROUTES.pop(key, None)
    left = COMM.allreduce(bool(left), op=MPI.LAND)
    taken = COMM.allreduce(bool(taken), op=MPI.LAND)
    check("a trace of the kernel leaves the choice of the route open", left)
    check("and the first pass over arrays makes it", taken)


def test_chunk_of_the_slot_pass():
    """Whether a block's batch is handed over whole or in chunks is asked at the weight
    of the pass that computes it, and an out-of-memory retry is counted.

    A slot pass plans its chunk by its share of the tangents: 1.3 times the working set
    of weight 1 for the columns of a quadratic state beside a linear parameter.  The
    question was asked at weight 1, so a batch that fits at 1 and not at 1.3 was handed
    over whole, and its pass then split it and joined the chunks: the array that the
    chunk thunk is there to avoid.  The budget here puts the batch between the two.
    """
    import warnings

    import jax

    if RANK == 0:
        print("the chunk of a slot pass")
    n = 8
    while n ** 3 < 512 * NP:
        n += 1                              # (a plan is never under 256 elements a rank)
    pm, Vh, b, K = build("hex", n, 2)
    loc = locals_at(Vh)
    gk, d = K.group_kernels[0], kernel_mod.device()
    ne = int(gk.group.ne)
    weight = gk._slot_weight(STATE)
    saved, retries = kernel_mod._free_budget, getattr(kernel_mod, "RETRIES", None)

    def forget():
        for g in K.group_kernels:
            g._chunk.clear()

    def route(j):
        """What a block with the columns of slot ``j`` is handed over as, and the chunks
        that its pass planned for the glued route and for the streamed one."""
        forget()
        mats = K.element_matrices_or_chunks(ADJOINT, j, loc)
        if callable(mats):
            for _chunk in mats():
                pass
        plans = {k[3]: v for k, v in gk._chunk.items() if len(k) == 4 and k[1] == "plan"}
        return callable(mats), plans.get(True), plans.get(False)

    try:
        kernel_mod._free_budget = lambda dev: 2 ** 50
        gk._plan(ne, d, 1, announce=False, remember=False)
        per = gk._last_estimate             # (the planner's bytes an element at weight 1)
        # a plan of 1.15 times the batch at weight 1 is 0.88 times the batch at 1.3
        kernel_mod._free_budget = lambda dev: int(1.15 * ne * per / 0.33)
        between = (gk._plan(ne, d, 1, announce=False, remember=False) == ne
                   > gk._plan(ne, d, weight, announce=False, remember=False))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # (a split batch says so)
            state, param = route(STATE), route(PARAMETER)
        ok = (ne == 0 or (weight > 1.2 and between and state[0] and state[1] is None
                          and state[2] is not None and state[2] < ne and param == (False, ne, None)))

        def words(handed, glued, streamed):
            ran = ("streamed in chunks of %d" % streamed if streamed is not None
                   else "joined from chunks of %d" % glued if glued is not None and glued < ne
                   else "in one piece")
            return "%s, the pass %s" % ("a chunk thunk" if handed else "arrays", ran)

        check("a batch that fits at weight 1 and not at the weight of its slot pass is handed "
              "over in chunks, and one whose pass has weight 1 whole",
              bool(COMM.allreduce(bool(ok), op=MPI.LAND)),
              "(%d elements; the state's columns, weight %.2f: %s; the parameter's, weight 1: %s)"
              % (ne, weight, words(*state), words(*param)))

        # a streamed chunk over 40 elements does not fit: every retry is counted, for a
        # benchmark's record, and its chunk stays with the pass
        real = gk.grad_block(ADJOINT)._raw

        def small_only(*a):
            lead = jax.tree_util.tree_leaves(a)[0]
            if not isinstance(lead, jax.core.Tracer) and lead.shape[0] > 40:
                raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to "
                                   "allocate 4096 bytes.")
            return real(*a)

        def streamed():
            pass
        streamed._raw, streamed._weight = small_only, 1
        forget()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spans = [(lo, hi) for lo, hi, _ in gk.in_chunks(
                streamed, gk.gather(loc), gk.mapped(streaming=True), kernel_mod._params(()))]
        tried = getattr(kernel_mod, "RETRIES", None)
        kept = gk._chunk.get((d, 1))
        ok = (ne == 0 or (retries is not None and tried > retries and kept is not None
                          and kept <= 40 and max(hi - lo for lo, hi in spans) == kept))
        check("an out-of-memory retry of an element pass is counted, and the pass keeps its chunk",
              bool(COMM.allreduce(bool(ok), op=MPI.LAND)),
              "(%s retries on rank 0, down to chunks of %s)"
              % ("no count of the" if retries is None else tried - retries, kept))
    finally:
        kernel_mod._free_budget = saved
        forget()


def test_interpolation_not_a_number():
    """A weight that is not a number in the interpolation between two meshes stays in it,
    and the row sums say so on every rank.

    MFEM gives that interpolation with an element's whole row, most of it zeros, which
    are left out.  A weight that is not a number went with them: its row kept a hole, the
    row sums were off by the size of the weight and not by something that is not a
    number, and the solver was given the interpolation.  And the largest of the ranks'
    figures was taken with MPI's MAX, which keeps a rank's NaN or loses it by the order
    the values meet in: the weight is spoiled on the first rank and then on the last.
    """
    import math

    from hippymfem.fem import ptransfer

    if RANK == 0:
        print("a weight that is not a number in the interpolation between two meshes")
    below = make_mesh("quad", max(6, 2 * NP))
    above = mfem.ParMesh.MakeRefined(below, 2, mfem.BasisType.ClosedUniform)
    Vc, Vf = FunctionSpace.H1(below, 1), FunctionSpace.H1(above, 1)
    clean = ptransfer.refinement(Vf, Vc)
    real, where = mfem.Opr2SparseMat, [None]

    def spoiled(op):
        sp = real(op)
        if RANK == where[0]:
            data = sp.GetDataArray()                # (a view of MFEM's values)
            data[np.flatnonzero(np.abs(data - 0.5) < 1e-12)[0]] = np.nan
        return sp

    seen = []
    mfem.Opr2SparseMat = spoiled
    try:
        for rank in sorted({0, NP - 1}):
            where[0] = rank
            T = ptransfer.refinement(Vf, Vc)
            kept = COMM.allreduce(int(np.isnan(T.values).sum()), op=MPI.SUM)
            said = COMM.allgather(math.isnan(T.row_sum_error))
            seen.append((rank, kept, T.global_nnz, sum(said), T.row_sum_error))
    finally:
        mfem.Opr2SparseMat = real
    # (the weight is in the matrix where its row is the spoiled rank's own, and in the row
    # sums of that rank either way)
    check("a weight that is not a number is not left out of the interpolation, and the row "
          "sums are not a number on every rank",
          clean.row_sum_error < 1e-12
          and all(nnz == clean.global_nnz and said == NP for _rank, _kept, nnz, said, _off in seen)
          and (NP > 1 or seen[0][1] == 1),
          "(%s; %d entries as MFEM gives it)"
          % ("; ".join("spoiled on rank %d: %d entries, %d of them not a number, row sums off by "
                       "%r on %d of %d ranks" % (rank, nnz, kept, off, said, NP)
                       for rank, kept, nnz, said, off in seen), clean.global_nnz))


def test_compact_scatter_map():
    """The compact scatter map gives the fused route the slots of the full one.

    One byte an entry and two bases an element row (:data:`~hippymfem.fem.pattern.
    COMPACT_PATTERN`) stand for the four-byte slot of every entry; the slots rebuilt
    from them are the same, and the scatter adds the same values in the same order, so
    the matrices are the same to the last bit.
    """
    from hippymfem.fem import pattern as pat

    if RANK == 0:
        print("compact scatter map")
    old, orig = pat.COMPACT_PATTERN, pat._fused_add_coded
    used = []

    def counting(*a):
        used.append(1)
        return orig(*a)

    pat._fused_add_coded = counting
    try:
        # ("mixed": both maps are made from the slots in the order of the element arrays)
        for kind, order in (("quad", 2), ("tri", 1), ("hex", 2), ("mixed", 2)):
            pm, Vh, b, K = build(kind, 5 if kind != "hex" else 3, order)
            loc = locals_at(Vh)
            bc = hp.DirichletBC(Vh[0], None, "all")
            for name, ti, ri, i, j, tr in (("A", 0, 0, ADJOINT, STATE, True),
                                           ("C", 0, 1, ADJOINT, PARAMETER, False)):
                args = dict(test_ess=bc.ess_tdof,
                            trial_ess=bc.ess_tdof if tr else None)
                got = []
                for compact in (False, True):
                    pat.COMPACT_PATTERN = compact
                    del used[:]
                    got.append(to_dense(asm.assemble_matrix(
                        Vh[ti], Vh[ri], b.groups,
                        (lambda i=i, j=j: K.element_matrix_chunks(i, j, loc)),
                        pm.GetNE(), **args), COMM))
                    coded = COMM.allreduce(len(used), op=MPI.SUM)
                d = float(np.abs(got[0] - got[1]).max())
                # (on a GPU the scatter's atomics add in an order of their own, so
                # there two assemblies agree to round-off only, whatever the map)
                exact = kernel_mod.device().platform == "cpu"
                scale = max(float(np.abs(got[0]).max()), 1e-300)
                check("the compact scatter map gives the same matrix (%s %s order %d)"
                      % (kind, name, order),
                      (d == 0.0 if exact else d <= 1e-13 * scale) and coded > 0,
                      "(largest difference %.1e%s; %d chunks through it)"
                      % (d, "" if exact else " of %.1e, on a device" % scale, coded))
    finally:
        pat.COMPACT_PATTERN, pat._fused_add_coded = old, orig


def test_geometric_factors_freed():
    """MFEM's cached geometric factors are dropped once the geometry is copied out.

    The element group copies ``J``, ``detJ`` and ``X`` out of MFEM's cache and never
    reads the cache again, so holding it would keep one more copy of the geometry
    for the life of the problem.  Freeing it must not change the geometry, and a
    second group on the same mesh, which recomputes the factors rather than finding
    them cached, must get exactly the same arrays.
    """
    from hippymfem.fem.elementbatch import ElementGroup

    if RANK == 0:
        print("geometric factors freed after the copy")
    pm = make_mesh("quad", 4 * NP)
    elems = np.arange(pm.GetNE())
    g1 = ElementGroup(pm, mfem.Geometry.SQUARE, elems, 4, batched=True)
    g2 = ElementGroup(pm, mfem.Geometry.SQUARE, elems, 4, batched=True)
    held = sum(isinstance(o, mfem.GeometricFactors) for o in g1._keep + g2._keep)
    check("no group keeps MFEM's factors", held == 0, "(%d kept)" % held)
    same = all(np.array_equal(getattr(g1, a), getattr(g2, a))
               for a in ("Jinv", "wdet", "X", "detJ"))
    check("a recomputed group matches the first exactly", same)
    check("and the arrays are finite", all(np.isfinite(getattr(g1, a)).all()
                                           for a in ("Jinv", "wdet", "X")))


def test_release_linearization_on_move():
    """The opt-in release drops the point at a moved forward solve, and only then.

    It must leave nothing behind that changes the answer: the blocks built at the
    next point are compared bit for bit against a problem that never released.
    """
    if RANK == 0:
        print("release the linearization point on a moved forward solve")
    pm = make_mesh("quad", 6)
    Vh = [FunctionSpace.H1(pm, 2), FunctionSpace.H1(pm, 1), None]
    Vh[2] = Vh[0]
    bc = hp.DirichletBC(Vh[0], None, "all")

    def varf(u, m, p, x):
        return jnp.exp(m.val) * jnp.dot(u.grad, p.grad) - p.val

    probs = {flag: hp.PDEVariationalProblem(Vh, varf, bc, bc.homogeneous(),
                                           is_fwd_linear=True,
                                           release_linearization_on_move=flag)
             for flag in (False, True)}
    hp.parRandom.set_seed(5)
    m0 = Vh[1].vector()
    hp.parRandom.normal(0.3, m0)
    p0 = Vh[2].vector()
    hp.parRandom.normal(1.0, p0)
    m1 = m0.copy()
    hp.parRandom.normal(0.05, m1)
    for pde in probs.values():
        u = Vh[0].vector()
        pde.solveFwd(u, [u, m0, p0])
        pde.setLinearizationPoint([u, m0, p0], gauss_newton_approx=False)
        pde.solveFwd(u, [u, m0, p0])            # same parameter: nothing released
    # the residual is linear in the state, so W_uu is never assembled (it is zero);
    # C and W_um are the blocks that show whether the point survived
    check("a forward solve at the linearization point keeps its blocks",
          all(p.C is not None and p.Wum is not None and p.Wuu is None
              for p in probs.values()))
    xs = {}
    for flag, pde in probs.items():
        u = Vh[0].vector()
        pde.solveFwd(u, [u, m1, p0])
        xs[flag] = u
    rel, keep = probs[True], probs[False]
    check("a moved forward solve releases the point when asked",
          rel.C is None and rel.Wum is None and rel.A is None
          and rel._lin_point is None)
    check("and keeps it otherwise", keep.C is not None and keep.A is not None)
    for flag, pde in probs.items():
        pde.setLinearizationPoint([xs[flag], m1, p0], gauss_newton_approx=False)
    worst = 0.0
    for name in ("A", "C", "Wum", "Wmm"):
        a = to_dense(getattr(keep, name), COMM)
        b = to_dense(getattr(rel, name), COMM)
        worst = largest([worst, float(np.abs(a - b).max())])
    check("the next point's blocks are identical", worst == 0.0, "(%.3e)" % worst)


def test_triple_product():
    """``P^T A P`` as the library forms it against MFEM's fused ``RAP``, bit for bit.

    The library takes two sparse products with the transpose kept.  That is only
    legitimate if it is the matrix ``RAP`` gives, values *and* structure: the folded
    elimination writes into entries that are structurally present but numerically
    zero, so a product that pruned them would break it.
    """
    if RANK == 0:
        print("triple product against MFEM's RAP")
    from hippymfem.common.linalg import take_ownership
    from hippymfem.fem.prolongation import _prolongation

    pm, Vh, b, K = build("quad", 5, 2)
    loc = locals_at(Vh)
    bc = hp.DirichletBC(Vh[0], None, "all")
    worst, same_nnz = 0.0, True
    for i, j in ((ADJOINT, STATE), (ADJOINT, PARAMETER)):
        test, trial = Vh[i], Vh[j]
        same = test.fes is trial.fes
        pat = csr.get_pattern(test, trial, b.groups)
        # the eliminated rows (and columns of the square block) as explicit zeros
        rows = csr._ess_ldof_mask(test, bc.ess_tdof)
        kill = pat.masked_slots(rows, rows if same else None)
        Aloc = csr.local_par_matrix(
            pat, pat.data(K.element_matrices(i, j, loc), zero_slots=kill), test, trial)
        P, ident = _prolongation(test)
        if ident:                       # one rank: the ldof matrix is the result
            continue
        Pr = _prolongation(trial)[0]
        if same:
            ours = csr._triple(Aloc, None, P, (test.fes,))
            ref = take_ownership(mfem.RAP(Aloc, P))
        else:
            ours = csr._triple(Aloc, P, Pr, (test.fes, trial.fes))
            ref = take_ownership(mfem.RAP(P, Aloc, Pr))
        worst = largest([worst, float(np.abs(to_dense(ours, COMM) - to_dense(ref, COMM)).max())])
        same_nnz = same_nnz and (COMM.allreduce(int(hypre_to_scipy(ours).nnz))
                                 == COMM.allreduce(int(hypre_to_scipy(ref).nnz)))
        del ours, ref, Aloc
    check("two products and MFEM's fused RAP agree", worst == 0.0, "(%.3e)" % worst)
    check("and keep the same structure", same_nnz)


def both_parmat(fn):
    """Run ``fn()`` on the triple-product route with each constructor of the local
    matrix, the copying one (which a device run takes) first, from a cold cache."""
    old = csr.set_parmat_mode("mfem")
    try:
        parmat.COPYING_CONSTRUCTOR = True
        csr.clear_pattern_cache()
        a = fn()
        parmat.COPYING_CONSTRUCTOR = False
        csr.clear_pattern_cache()
        b = fn()
    finally:
        parmat.COPYING_CONSTRUCTOR = False
        csr.set_parmat_mode(old)
        csr.clear_pattern_cache()
    return a, b


def test_parmat_routes():
    """Both constructors of the local matrix must give the same parallel matrix.

    One hands hypre the numpy arrays, which it copies (what a run with hypre on a
    device does); the other wraps them as an ``mfem.SparseMatrix`` that MFEM's
    constructor aliases (the host's).  They must be the same matrix, so this
    compares them densely over the block shapes and diagonal policies the library
    assembles.

    The repeated-assembly check guards a trap: the square constructor reorders
    each row to put the diagonal first and syncs that permutation back into the
    ``SparseMatrix`` it was given, so handing it the pattern's own ``indices``
    permutes the cache.  The first matrix then comes out right and every later one
    pairs permuted columns with values in the original slot order.  Assembling
    repeatedly and comparing against a cold cache catches that.
    """
    if RANK == 0:
        print("parallel-matrix construction: the copying against the aliasing constructor")
    for kind, n, order in (("quad", 5, 2), ("hex", 3, 2)):
        pm, Vh, b, K = build(kind, n, order)
        bc = hp.DirichletBC(Vh[0], None, "all")
        loc = locals_at(Vh)
        CASES = (("square, ess rows, diag=one", ADJOINT, STATE, 0, 0,
                  bc.ess_tdof, "one"),
                 ("square, ess rows, diag=zero", STATE, STATE, 0, 0,
                  bc.ess_tdof, "zero"),
                 ("square, no ess", ADJOINT, STATE, 0, 0, None, "one"),
                 ("rectangular, ess rows", ADJOINT, PARAMETER, 0, 1,
                  bc.ess_tdof, "one"))
        for tag, i, j, ti, ri, ess, pol in CASES:
            def one():
                A = asm.assemble_matrix(
                    Vh[ti], Vh[ri], b.groups, K.element_matrices(i, j, loc),
                    pm.GetNE(), test_ess=ess, diag_policy=pol)
                out = to_dense(A, COMM)
                del A
                return out
            a, c = both_parmat(one)
            d = float(np.abs(a - c).max())
            check("%s %s: %s identical both ways" % (kind, order, tag), d == 0.0,
                  "(%.3e)" % d)

        # repeated assemblies, and the pattern they share, must survive the route
        def repeat():
            pat = csr.get_pattern(Vh[0], Vh[0], b.groups)
            j0 = np.asarray(pat.indices).copy()
            i0 = np.asarray(pat.indptr).copy()
            out = []
            for _ in range(3):
                A = asm.assemble_matrix(
                    Vh[0], Vh[0], b.groups,
                    K.element_matrices(ADJOINT, STATE, loc), pm.GetNE(),
                    test_ess=bc.ess_tdof)
                out.append(to_dense(A, COMM))
                del A
            intact = (np.array_equal(j0, np.asarray(pat.indices))
                      and np.array_equal(i0, np.asarray(pat.indptr)))
            return out, intact
        (da, ia), (dc, ic) = both_parmat(repeat)
        check("%s %s: pattern indices survive the copying constructor" % (kind, order), ia)
        check("%s %s: pattern indices survive the aliasing constructor" % (kind, order), ic)
        worst = largest(float(np.abs(x - da[0]).max()) for x in da + dc)
        check("%s %s: three assemblies agree across both constructors" % (kind, order),
              worst == 0.0, "(%.3e)" % worst)


def test_tdof_route():
    """Assembling straight into true-dof rows must give the triple product's matrix.

    For a boolean prolongation ``P^T A P`` is a permutation with a sum over shared
    dofs, and the true-dof route does that sum itself: own rows are a fixed
    permutation of the ldof scatter, ghost rows go to their owner once symbolically
    and as values every assembly.  This is faster than the general triple product
    (measurements in ``benchmarks/DESIGN_NOTES.md``).

    The same terms are summed, own contributions first and received ones after, so
    on more than one rank the agreement with hypre's summation order is to round-off
    rather than exact; on one rank the route is not taken (``P`` is the identity) and
    the difference must be exactly zero.  The pattern's own arrays must survive, and
    repeated assemblies must agree with a cold rebuild, as for every other route.
    """
    if RANK == 0:
        print("true-dof route vs triple product")
    tol = 0.0 if COMM.size == 1 else 1e-13

    def under(mode, fn):
        old = csr.set_parmat_mode(mode)
        try:
            csr.clear_pattern_cache()
            return fn()
        finally:
            csr.set_parmat_mode(old)
            csr.clear_pattern_cache()

    for kind, n, order in (("quad", 5, 2), ("hex", 3, 2)):
        pm, Vh, b, K = build(kind, n, order)
        bc = hp.DirichletBC(Vh[0], None, "all")
        loc = locals_at(Vh)
        CASES = (("square, ess rows, diag=one", ADJOINT, STATE, 0, 0, bc.ess_tdof, "one"),
                 ("square, ess rows, diag=zero", STATE, STATE, 0, 0, bc.ess_tdof, "zero"),
                 ("square, no ess", ADJOINT, STATE, 0, 0, None, "one"),
                 ("rectangular, ess rows", ADJOINT, PARAMETER, 0, 1, bc.ess_tdof, "one"))
        for tag, i, j, ti, ri, ess, pol in CASES:
            def one():
                A = asm.assemble_matrix(
                    Vh[ti], Vh[ri], b.groups, K.element_matrices(i, j, loc),
                    pm.GetNE(), test_ess=ess, diag_policy=pol)
                out = (to_dense(A, COMM), int(A.NNZ()))
                del A
                return out
            (a, na), (c, nc) = under("mfem", one), under("tdof", one)
            scale = max(float(np.abs(a).max()), 1e-300)
            d = float(np.abs(a - c).max()) / scale
            check("%s %s: %s (tdof vs triple product)" % (kind, order, tag),
                  d <= tol and na == nc, "(rel %.2e, nnz %d vs %d)" % (d, na, nc))

        def repeat():
            pat = csr.get_pattern(Vh[0], Vh[0], b.groups)
            j0, i0 = np.asarray(pat.indices).copy(), np.asarray(pat.indptr).copy()
            out = []
            for _ in range(3):
                A = asm.assemble_matrix(
                    Vh[0], Vh[0], b.groups,
                    K.element_matrices(ADJOINT, STATE, loc), pm.GetNE(),
                    test_ess=bc.ess_tdof)
                out.append(to_dense(A, COMM))
                del A
            intact = (np.array_equal(j0, np.asarray(pat.indices))
                      and np.array_equal(i0, np.asarray(pat.indptr)))
            return out, intact
        outs, intact = under("tdof", repeat)
        check("%s %s: pattern indices survive the true-dof route" % (kind, order), intact)
        worst = largest(float(np.abs(x - outs[0]).max()) for x in outs)
        check("%s %s: repeated true-dof assemblies identical" % (kind, order),
              worst == 0.0, "(%.3e)" % worst)


def test_rank_without_elements_of_a_block():
    """A rank that holds no element of a block still owns rows of it and builds its
    true-dof pattern with the others.

    A boundary term's element matrices carry the dofs of the adjacent volume element,
    and a shared dof belongs to the lowest rank that holds it.  The strip here has one
    element a rank and the term on its right end: the last rank holds the one boundary
    element, and the rank before it, which holds none, owns the rows of the two vertices
    they share.  Two things went wrong on such a rank.  Its true-dof target has slots
    and no element arrays; taken for a rank that owns nothing of the block, it was given
    no accumulator and stopped, the others waiting for it in the exchange.  And it has
    no group to key its pattern on; keyed on the two spaces alone, the second term on
    them found the first one's pattern and true-dof pattern there, and the rank went on
    while the others were building theirs, which is collective.  The first two checks
    ask for the same on any number of ranks, one included.

    The block is ``u dp/dn``, whose rows of those two vertices are not zero, at two
    quadrature degrees (two sets of groups), and the check is ``x^T M 1``, the length of
    the end: with the two rows lost it is the number of elements instead.
    """
    from hippymfem.fem.boundary import BoundaryBatches, BoundaryKernel, assemble_boundary_matrix
    from hippymfem.fem.pattern import ScatterPattern

    if RANK == 0:
        print("a rank without elements of a block")
    n = max(NP, 2)
    m = mfem.Mesh.MakeCartesian2D(n, 1, mfem.Element.QUADRILATERAL)
    c, part = mfem.Vector(2), []
    for e in range(n):
        m.GetElementCenter(e, c)
        part.append(min(int(c[0] * n), NP - 1))          # one a rank, from left to right
    part = mfem.intArray(part)
    pm = mfem.ParMesh(COMM, m, part.GetData())
    V = FunctionSpace.H1(pm, 1)

    # what that rank's target is: no groups, and slots for the rows it is sent
    T = ScatterPattern([], [], test_space=V, trial_space=V)
    T.nnz = 5
    try:
        acc = T.data([])
    except ValueError:
        acc = None
    check("a target with slots and no elements gives a zero accumulator of its length",
          acc is not None and acc.shape == (5,) and not acc.any(),
          "(%s)" % ("none" if acc is None else "%d entries" % acc.size))
    # and its patterns: one for each set of groups, which its (empty) list stands for,
    # also once that list is gone
    none1, none2 = [], []
    P1, P2 = get_pattern(V, V, none1), get_pattern(V, V, none2)
    gone = [get_pattern(V, V, []) for _ in range(3)]
    apart = (P1 is not P2 and get_pattern(V, V, none1) is P1 and get_pattern(V, V, none2) is P2
             and len({id(P) for P in gone + [P1, P2]}) == 5)
    csr.clear_pattern_cache()
    check("two sets of groups without an element here have a pattern each", apart)
    if acc is None or not apart:
        return                 # (the assemblies below would stop or hang on several ranks)

    lengths, held, sent = [], None, None
    loc = V.local_values(V.vector())
    ones, xc, y = V.vector(), V.vector(), V.vector()
    ones.array[:] = 1.0
    xc.array[:] = V.coordinates()[:, 0]
    for degree in (4, 6):
        bb = BoundaryBatches(pm, degree, [2], COMM)      # the right end
        K = BoundaryKernel(lambda u, m_, p, x, nor: u.val * jnp.dot(p.grad, nor),
                           [V, V, V], bb)
        M = assemble_boundary_matrix(V, V, bb.groups,
                                     K.element_matrices(ADJOINT, STATE, [loc, loc, loc]))
        held = COMM.allgather(sum(g.ne for g in bb.groups))
        plan = csr.plan_block(V, V, bb.groups)           # (that assembly's, from the caches)
        sent = COMM.allgather(0 if plan.tpat is None else int(plan.tpat.nrecv))
        M.Mult(ones.hypre, y.hypre)
        lengths.append(y.inner(xc))
        del M
    csr.clear_pattern_cache()
    # (on the true-dof route of several ranks the rank before the last is that rank)
    there = NP == 1 or plan.tpat is None or any(h == 0 and s > 0 for h, s in zip(held, sent))
    check("a rank without the boundary element assembles the rows it owns, of two blocks",
          all(abs(length - 1.0) < kernel_mod.matrix_tolerance(1e-12) for length in lengths) and there,
          "(x^T M 1 = %.15f and %.15f; boundary elements per rank %s, entries received %s)"
          % (lengths[0], lengths[1], held, sent))


def test_eliminated_diagonal():
    """The diagonal the policy asks for has to be in the assembled matrix.

    ``EliminateRowsCols`` leaves 1.0 on an eliminated row whatever the policy says,
    so the value is written afterwards, through the wrapper ``GetDiag`` returns.  That
    wrapper is over MFEM's *host* copy of hypre's arrays: with hypre's memory on a
    device the write lands there and nowhere else unless the matrix itself is marked
    host-read-write first, and then put back.  A host build passes either way; on a
    device the failure is silent and shows up downstream as a singular or wrongly
    scaled essential block.
    """
    if RANK == 0:
        print("eliminated diagonal")
    pm, Vh, b, K = build("quad", 5, 2)
    bc = hp.DirichletBC(Vh[0], None, "all")
    loc = locals_at(Vh)
    idx = np.asarray(bc.ess_tdof.ToList(), dtype=np.int64)
    for pol, want in (("one", 1.0), ("zero", 0.0)):
        for copying in (True, False):
            old = csr.set_parmat_mode("mfem")
            parmat.COPYING_CONSTRUCTOR = copying
            try:
                csr.clear_pattern_cache()
                A = asm.assemble_matrix(
                    Vh[0], Vh[0], b.groups,
                    K.element_matrices(ADJOINT, STATE, loc), pm.GetNE(),
                    test_ess=bc.ess_tdof, diag_policy=pol)
                d = _local_diag(A)
                worst = (0.0 if idx.size == 0
                         else float(np.abs(d[idx] - want).max()))
                del A
            finally:
                parmat.COPYING_CONSTRUCTOR = False
                csr.set_parmat_mode(old)
                csr.clear_pattern_cache()
            worst = allreduce_extreme(COMM, worst, MPI.MAX)
            check("diag=%s leaves %.1f on every eliminated row (%s constructor)"
                  % (pol, want, "copying" if copying else "aliasing"),
                  worst == 0.0, "(%.3e)" % worst)


def test_boolean_prolongation():
    """``Conforming()`` has to be a safe stand-in for "P is a single 1.0 per row".

    The folded elimination needs a boolean ``P``.  Reading ``P``'s entries to find out
    goes through ``MergeDiagAndOffd``, which has hypre allocate the merged matrix in
    its own memory location and then deep-copies it from the host, so it cannot be
    used with hypre on a device.  ``Conforming()`` answers without touching the
    matrix, and is *sufficient* but not necessary: a mesh carrying an ``NCMesh`` with
    no hanging nodes yet says False while its ``P`` is still boolean.  That direction
    is the safe one (the fold is skipped and :func:`_eliminate` does the work), and
    this test pins it down so the implication is not inverted later.
    """
    if RANK == 0:
        print("boolean prolongation test")
    cases = []
    m = mfem.Mesh.MakeCartesian2D(4, 4, mfem.Element.QUADRILATERAL)
    cases.append(("conforming quad", mfem.ParMesh(COMM, m)))
    m2 = mfem.Mesh.MakeCartesian2D(4, 4, mfem.Element.QUADRILATERAL)
    m2.EnsureNCMesh(True)
    pm2 = mfem.ParMesh(COMM, m2)
    mk = mfem.intArray([i for i in range(pm2.GetNE()) if i % 3 == 0])
    pm2.GeneralRefinement(mk, 1)
    cases.append(("NC quad with hanging nodes", pm2))
    for label, pm in cases:
        for order in (1, 2):
            V = FunctionSpace.H1(pm, order)
            P = V.fes.Dof_TrueDof_Matrix()
            conf = bool(COMM.allreduce(bool(V.fes.Conforming()), op=MPI.LAND))
            vals = bool(COMM.allreduce(bool(csr._is_boolean(P)), op=MPI.LAND))
            # sufficient, not necessary: True must imply True, False may differ
            check("%s order %d: Conforming() implies a boolean P"
                  % (label, order), (not conf) or vals,
                  "(Conforming=%s, entries=%s)" % (conf, vals))
            check("%s order %d: _boolean_local agrees with the entries on the host"
                  % (label, order),
                  bool(COMM.allreduce(bool(csr._boolean_local(V, P)),
                                      op=MPI.LAND)) == vals)
    csr.clear_pattern_cache()


def test_no_leak():
    """``take_ownership`` must keep RSS flat across repeated assembly.

    Without it, ``RAP`` plus ``EliminateRowsCols`` leak one parallel matrix per
    assembly, which an optimizer turns into gigabytes.
    """
    if RANK == 0:
        print("matrix ownership")
    pm = make_mesh("quad", 24)
    V = FunctionSpace.H1(pm, 1)
    Vm = FunctionSpace.H1(pm, 1)
    b = MeshBatches(pm, 4, COMM)
    K = QuadratureKernel(lambda u, m, p, x: jnp.exp(m.val) * jnp.dot(u.grad, p.grad),
                         [V, Vm, V], b)
    loc = locals_at([V, Vm, V])
    mats = K.element_matrices(ADJOINT, STATE, loc)
    bc = hp.DirichletBC(V, None, "all")
    NE = pm.GetNE()

    for backend, mod in (("csr", asm), ("integrator", ref)):
        for _ in range(5):                     # warm up caches and allocators
            A = mod.assemble_matrix(V, V, b.groups, mats, NE,
                                    test_ess=bc.ess_tdof)
            del A
        gc.collect()
        r0 = rss_kb()
        for _ in range(60):
            A = mod.assemble_matrix(V, V, b.groups, mats, NE,
                                    test_ess=bc.ess_tdof)
            del A
        gc.collect()
        grew = rss_kb() - r0
        worst = COMM.allreduce(grew, op=MPI.MAX)
        check("%s: 60 assemblies do not grow RSS" % backend, worst < 20000,
              "(+%d kB)" % worst)

    from hippymfem.common.linalg import ParAdd, MatPtAP

    A = asm.assemble_matrix(V, V, b.groups, mats, NE)
    gc.collect()
    r0 = rss_kb()
    for _ in range(60):
        S = ParAdd(A, A)
        del S
    gc.collect()
    worst = COMM.allreduce(rss_kb() - r0, op=MPI.MAX)
    check("ParAdd does not leak", worst < 20000, "(+%d kB)" % worst)


def test_operator_reset_frees():
    """Re-setting a solver's operator must release the previous one and its AMG.

    A solver that kept every operator, preconditioner and MFEM solver it was
    ever given would leak one AMG hierarchy per line-search step of an optimizer
    that re-sets the forward solver at every trial point, and a linearization
    point that kept every block would do the same; on a GPU that exhausts memory
    within a few Newton steps.  Leaks are counted in live wrapper objects, which,
    unlike RSS, do not depend on the allocator.
    """
    if RANK == 0:
        print("operator resets free the previous operator")

    def live(name):
        return sum(1 for o in gc.get_objects() if type(o).__name__ == name)

    pm, Vh, b, K = build("quad", 6, 2)
    bc = hp.DirichletBC(Vh[0], None, "all")
    solver = hp.KrylovSolver(COMM, "cg", "amg")
    counts = []
    for k in range(6):
        loc = locals_at(Vh, seed=k)
        A = asm.assemble_matrix(Vh[0], Vh[0], b.groups,
                                K.element_matrices(ADJOINT, STATE, loc), pm.GetNE(),
                                test_ess=bc.ess_tdof)
        solver.set_operator(A)
        del A
        gc.collect()
        counts.append((live("HypreParMatrix"), live("HypreBoomerAMG")))
    check("re-setting the operator 6 times keeps one AMG live, not six",
          counts[-1] == counts[1], "(matrices, AMGs after each set: %s)" % counts)

    def varf(u, m, p, x):
        return jnp.exp(m.val) * jnp.dot(u.grad, p.grad) + u.val * u.val * m.val * p.val
    pde = hp.PDEVariationalProblem(Vh, varf, bc, bc.homogeneous())
    for attr in ("solver", "solver_fwd_inc", "solver_adj_inc"):
        setattr(pde, attr, hp.KrylovSolver(COMM, "cg", "amg"))
    x = [Vh[0].vector(), Vh[1].vector(), Vh[2].vector()]
    hp.parRandom.set_seed(3)
    for v in x:
        hp.parRandom.normal(1.0, v)
    kept, mats = [], []
    for k in range(5):
        hp.parRandom.normal(0.05, x[PARAMETER])
        pde.setLinearizationPoint(x, gauss_newton_approx=False)
        gc.collect()
        kept.append(len(pde._keep))
        mats.append(live("HypreParMatrix"))
    check("linearization points do not accumulate kept blocks",
          kept[-1] == kept[1] and mats[-1] == mats[1],
          "(kept: %s, live matrices: %s)" % (kept, mats))

    bcn = hp.DirichletBC(Vh[0], lambda z: z[0], "all")
    v = Vh[0].vector()
    bcn.apply(v)
    k0 = len(bcn._keep)
    for _ in range(5):
        bcn.apply(v)
    check("applying a Dirichlet value five times keeps nothing new",
          len(bcn._keep) == k0, "(%d -> %d kept)" % (k0, len(bcn._keep)))
    k0 = len(Vh[0]._keep)
    for _ in range(5):
        Vh[0].gridfunction()
        Vh[0].to_gridfunction(v)
        Vh[0].project(lambda z: z[0])
    check("grid functions made from a space are not kept by it",
          len(Vh[0]._keep) == k0, "(%d -> %d kept)" % (k0, len(Vh[0]._keep)))


def test_no_gpu_context():
    """A host run must hold no CUDA context on any card.

    When jax is imported before hippymfem it has already read its platform list, and
    its first computation would open a CUDA context on every card it can see.
    hippymfem therefore sets the platform list through
    ``jax.config`` when jax is already imported.  This suite imports jax first, so
    this process must be absent from ``nvidia-smi``'s process list.  Skipped where
    there is no nvidia-smi or when a GPU was requested.
    """
    import os
    import shutil
    import subprocess

    if RANK == 0:
        print("no GPU context for a host run")
    if os.environ.get("HIPPYMFEM_DEVICE", "").lower() in ("gpu", "cuda", "rocm", "gpu:0") \
            or not shutil.which("nvidia-smi"):
        if RANK == 0:
            print("  skipped: a GPU was requested or nvidia-smi is unavailable")
        return
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception as e:                                   # noqa: BLE001
        if RANK == 0:
            print("  skipped: nvidia-smi failed (%s)" % e)
        return
    held = str(os.getpid()) in [l.strip() for l in out.splitlines()]
    check("a host run holds no CUDA context", not held,
          "(CUDA_VISIBLE_DEVICES=%r)" % os.environ.get("CUDA_VISIBLE_DEVICES"))


def test_speed():
    """Report what the direct path buys per element; assert only that it is faster."""
    if RANK == 0:
        print("assembly cost per element")
    for kind, n, order in (("quad", 60, 1), ("quad", 40, 2)):
        pm, spaces, batches, K = build(kind, n, order, 1)
        NE = pm.GetNE()
        loc = locals_at(spaces)
        bc = hp.DirichletBC(spaces[STATE], None, "all")
        mats = K.element_matrices(ADJOINT, STATE, loc)
        t = {}
        for backend, mod in (("integrator", ref), ("csr", asm)):
            A = mod.assemble_matrix(spaces[0], spaces[0], batches.groups, mats,
                                    NE, test_ess=bc.ess_tdof)
            del A
            COMM.Barrier()
            t0 = time.perf_counter()
            for _ in range(5):
                A = mod.assemble_matrix(spaces[0], spaces[0], batches.groups,
                                        mats, NE, test_ess=bc.ess_tdof)
                del A
            COMM.Barrier()
            t[backend] = (time.perf_counter() - t0) / 5
        pat = get_pattern(spaces[0], spaces[0], batches.groups)
        if RANK == 0:
            print("      %s order %d, NE/rank=%d: callback %.2f us/elem, "
                  "direct %.2f us/elem, %.1fx  (nnz=%d)"
                  % (kind, order, NE, 1e6 * t["integrator"] / NE,
                     1e6 * t["csr"] / NE, t["integrator"] / t["csr"], pat.nnz),
                  flush=True)
        check("direct scatter is faster (%s order %d)" % (kind, order),
              t["csr"] < t["integrator"],
              "(%.2fx)" % (t["integrator"] / t["csr"]))


def test_coordinates_and_project():
    """``coordinates()`` and the nodal ``project``, vectorized and point by
    point, reproduce MFEM's ``ProjectCoefficient`` route at P1 and P2."""
    if RANK == 0:
        print("dof coordinates and nodal projection against the callback route")
    from hippymfem.fem.spaces import _ScalarPy

    pm = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian3D(4, 4, 4, mfem.Element.HEXAHEDRON))

    def f(x):                                   # point form: x is one point
        return 1.0 + 2.0 * x[0] - 3.0 * x[1] * x[2] + x[2] ** 2

    def fv(X):                                  # array form: X is (n, 3)
        return 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] * X[:, 2] + X[:, 2] ** 2

    worst = 0.0
    for order in (1, 2):
        V = FunctionSpace.H1(pm, order)
        X = V.coordinates()
        ref = np.zeros_like(X)
        gf = mfem.ParGridFunction(V.fes)
        tmp = V.vector()
        for d in range(3):
            gf.ProjectCoefficient(_ScalarPy(lambda x, d=d: x[d]))
            gf.GetTrueDofs(tmp.hypre)
            ref[:, d] = tmp.array
        gf.ProjectCoefficient(_ScalarPy(f))
        gf.GetTrueDofs(tmp.hypre)
        uref = tmp.array.copy()
        worst = largest([worst, float(np.abs(X - ref).max()),
                         float(np.abs(V.project(f).array - uref).max()),
                         float(np.abs(V.project(fv).array - uref).max())])
    worst = allreduce_extreme(COMM, worst, MPI.MAX)
    check("coordinates and nodal projection match the callback route", worst < 1e-12,
          "(max abs diff %.1e)" % worst)


def main():
    test_blocks_identical()
    test_sortfree_pattern()
    test_structural_zeros()
    test_keep_policy()
    test_partial_boundary_elimination()
    test_folded_elimination()
    test_shared_hessian_pass()
    test_column_restricted_pass()
    test_slot_order_is_the_lexsort()
    test_transpose_free_adjoint()
    test_element_chunking()
    test_triple_product()
    test_coordinates_and_project()
    test_fused_scatter()
    test_split_and_whole_ranks()
    test_auto_route_from_a_run()
    test_chunk_of_the_slot_pass()
    test_interpolation_not_a_number()
    test_compact_scatter_map()
    test_geometric_factors_freed()
    test_release_linearization_on_move()
    test_parmat_routes()
    test_tdof_route()
    test_rank_without_elements_of_a_block()
    test_eliminated_diagonal()
    test_boolean_prolongation()
    test_no_leak()
    test_operator_reset_frees()
    test_no_gpu_context()
    test_speed()
    COMM.Barrier()
    if RANK == 0:
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

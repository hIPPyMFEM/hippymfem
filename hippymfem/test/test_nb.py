# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Plotting helpers: sampled fields against the functions they represent.

Run with ``mpirun -n N python -m hippymfem.test.test_nb``.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np                     # noqa: E402
from mpi4py import MPI                 # noqa: E402

import mfem.par as mfem                # noqa: E402
import hippymfem as hm                 # noqa: E402
from hippymfem import nb               # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size
FAILS = []


def check(name, ok, detail=""):
    ok = COMM.bcast(bool(ok), root=0)       # decided on rank 0, where the data is
    if RANK == 0:
        print("  [%s] %s %s" % ("ok  " if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILS.append(name)


def mesh2d(kind, nx=6, ny=5):
    et = mfem.Element.TRIANGLE if kind == "tri" else mfem.Element.QUADRILATERAL
    return mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian2D(nx, ny, et, False, 2.0, 1.0))


def test_exact_samples():
    """A field in the space is reproduced exactly at every sample point."""
    if RANK == 0:
        print("sampled values against the projected function")
    cases = (("H1", 1, lambda x: 1.0 + 2.0 * x[0] - x[1]),
             ("H1", 2, lambda x: x[0] * x[1] + x[0] ** 2),
             ("L2", 1, lambda x: 3.0 * x[0] - x[1]))
    for kind in ("tri", "quad"):
        pmesh = mesh2d(kind)
        ne = COMM.allreduce(pmesh.GetNE())
        for fam, order, fn in cases:
            V = getattr(hm.FunctionSpace, fam)(pmesh, order)
            data = nb.sample_field(V, V.project(fn))
            ok, detail = True, ""
            if RANK == 0:
                pts, cells, vals = data
                err = float(np.abs(vals - fn(pts.T)).max())
                per_elem = order * order * (1 if kind == "tri" else 2)
                ok = (err < 1e-12 and cells.shape == (ne * per_elem, 3)
                      and cells.max() < pts.shape[0])
                detail = "(max error %.1e, %d cells)" % (err, cells.shape[0])
            check("%s %s%d" % (kind, fam, order), ok, detail)


def test_vector_and_1d():
    """Vector fields sample their magnitude or a component; 1D gives segments."""
    if RANK == 0:
        print("vector-valued and 1D fields")
    pmesh = mesh2d("tri")
    V = hm.FunctionSpace.H1(pmesh, 1, vdim=2)
    v = V.project(lambda x: np.array([x[0], -2.0 * x[1]]))
    mag = nb.sample_field(V, v)
    comp = nb.sample_field(V, v, component=1)
    ok, detail = True, ""
    if RANK == 0:
        pts = mag[0]
        e1 = np.abs(mag[2] - np.hypot(pts[:, 0], 2.0 * pts[:, 1])).max()
        e2 = np.abs(comp[2] + 2.0 * comp[0][:, 1]).max()
        ok = e1 < 1e-12 and e2 < 1e-12
        detail = "(magnitude %.1e, component %.1e)" % (e1, e2)
    check("vector H1: magnitude and component", ok, detail)

    mesh1 = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian1D(12, 1.0))
    V1 = hm.FunctionSpace.H1(mesh1, 2)
    data = nb.sample_field(V1, V1.project(lambda x: x[0] ** 2))
    ok, detail = True, ""
    if RANK == 0:
        pts, cells, vals = data
        err = float(np.abs(vals - pts[:, 0] ** 2).max())
        ok = err < 1e-12 and cells.shape == (24, 2)
        detail = "(max error %.1e, %d segments)" % (err, cells.shape[0])
    check("1D P2 on segments", ok, detail)


def test_drawing():
    """Every drawing function runs, on every rank, and returns on rank 0 only."""
    if RANK == 0:
        print("drawing (Agg backend)")
    import matplotlib.pyplot as plt

    pmesh = mesh2d("quad")
    V = hm.FunctionSpace.H1(pmesh, 2)
    u = V.project(lambda x: np.sin(x[0]) * x[1])
    w = V.project(lambda x: 1.0 + x[0])
    U = hm.MultiVector(u, 4)
    for i in range(4):
        U[i].assign(V.project(lambda x, i=i: np.cos(i * x[0]) + x[1]))
    results = {
        "plot field": nb.plot(V, u, mytitle="u"),
        "plot log scale": nb.plot(V, w, logscale=True),
        "plot mesh": nb.plot(pmesh),
        "multi1_plot": nb.multi1_plot([(V, u), (V, w)], ["u", "w"]),
        "plot_eigenvectors": nb.plot_eigenvectors(V, U, which=(0, 1, 3)),
        "plot_eigenvalues": nb.plot_eigenvalues([10.0, 3.0, 1.0, 0.2, -1e-14]),
    }
    mesh1 = mfem.ParMesh(COMM, mfem.Mesh.MakeCartesian1D(10, 1.0))
    V1 = hm.FunctionSpace.H1(mesh1, 1)
    results["plot 1D"] = nb.plot(V1, V1.project(lambda x: x[0]))
    results["plot 1D mesh"] = nb.plot(mesh1)
    if RANK == 0:
        results["plot_pts"] = nb.plot_pts(np.random.default_rng(0).uniform(size=(9, 2)),
                                          np.arange(9.0))
    for name, art in results.items():
        check(name, (art is not None) if RANK == 0 else True)
    ok = COMM.allreduce(int(all(a is None for k, a in results.items()
                                if k != "plot_pts")) if RANK else 1, op=MPI.MIN)
    check("other ranks return None", ok)
    plt.close("all")


if __name__ == "__main__":
    mfem.Hypre.Init()
    if RANK == 0:
        print("=" * 74)
        print("hIPPyMFEM plotting tests on %d rank(s)" % NP)
        print("=" * 74)
    test_exact_samples()
    test_vector_and_1d()
    test_drawing()
    if RANK == 0:
        print("-" * 74)
        print("FAILURES: %d %s" % (len(FAILS), FAILS if FAILS else ""))
    COMM.Barrier()
    raise SystemExit(1 if FAILS else 0)

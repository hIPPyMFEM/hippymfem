# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Quick plots of fields on 1D and 2D meshes, for notebooks and scripts.

The counterpart of ``hippylib.utils.nb``, drawn with matplotlib.  A field is not
reduced to its vertex values: every element is subdivided with MFEM's geometry
refiner and the field is evaluated at the sub-element vertices, so a P2 state or a
discontinuous L2 field is drawn as what it is.  3D fields belong in ParaView
(:func:`hippymfem.write_paraview`).

Every function that takes a field is collective: in parallel each rank evaluates
its own elements, rank 0 gathers and draws, and the other ranks return ``None``.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from ..fem.spaces import as_space

__all__ = ["sample_field", "plot", "plot_mesh", "plot_pts", "multi1_plot",
           "plot_eigenvalues", "plot_eigenvectors", "show_solution"]

_NVERT = {mfem.Geometry.SEGMENT: 2, mfem.Geometry.TRIANGLE: 3,
          mfem.Geometry.SQUARE: 4}


def _pyplot():
    import matplotlib.pyplot as plt     # here, so the library does not need it

    return plt


def _mesh_comm(mesh):
    comm = mesh.GetComm() if hasattr(mesh, "GetComm") else None
    return comm if isinstance(comm, MPI.Comm) else MPI.COMM_WORLD


def _is_vector_valued(space):
    return space.vdim > 1 or space.fec.Name().startswith(("ND", "RT"))


def _gather(comm, local):
    """Concatenate ``(points, cells, values)`` from every rank on rank 0."""
    if comm.size == 1:
        return local
    parts = comm.gather(local, root=0)
    if comm.rank != 0:
        return None
    offset, P, C, V = 0, [], [], []
    for p, c, v in parts:
        P.append(p)
        C.append(c + offset)
        V.append(v)
        offset += p.shape[0]
    return np.concatenate(P), np.concatenate(C), np.concatenate(V)


def sample_field(space, vector, refine=None, component=None):
    """Evaluate a field on a refinement of every element, gathered on rank 0.

    Parameters
    ----------
    space : FunctionSpace
    vector : ParVector
        True-dof values of the field.
    refine : int, optional
        Subdivisions per element edge; the polynomial degree by default.
    component : int, optional
        For a vector-valued field (``vdim > 1``, H(curl), H(div)), the component
        to return; the pointwise magnitude by default.

    Returns
    -------
    tuple or None
        On rank 0, ``(points, cells, values)``: sample points ``(npts, sdim)``,
        cells ``(ncells, 2)`` in 1D or ``(ncells, 3)`` in 2D with quadrilaterals
        split in two, and values ``(npts,)``.  ``None`` on the other ranks.
    """
    space = as_space(space)
    mesh = space.mesh
    dim, sdim = mesh.Dimension(), mesh.SpaceDimension()
    if dim not in (1, 2):
        raise NotImplementedError(
            "plotting is for 1D and 2D meshes; write 3D fields for ParaView with "
            "hippymfem.write_paraview")
    if refine is None:
        refine = max(1, space.order)
    gf = space.to_gridfunction(vector)
    vector_valued = _is_vector_valued(space)
    rules = {}
    scalar, coords, vec = mfem.Vector(), mfem.DenseMatrix(), mfem.DenseMatrix()
    pts, cells, vals = [], [], []
    npts = 0
    for e in range(mesh.GetNE()):
        geom = mesh.GetElementBaseGeometry(e)
        if geom not in rules:
            if geom not in _NVERT:
                raise NotImplementedError("cannot plot geometry type %d" % geom)
            rg = mfem.GlobGeometryRefiner.Refine(geom, int(refine))
            nv = _NVERT[geom]
            conn = np.asarray(rg.RefGeoms.ToList(), dtype=np.int64).reshape(-1, nv)
            if nv == 4:
                conn = np.vstack((conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]))
            rules[geom] = (rg, conn)
        rg, conn = rules[geom]
        ir = rg.RefPts
        nq = ir.GetNPoints()
        T = mesh.GetElementTransformation(e)
        T.Transform(ir, coords)
        x = np.array(coords.GetDataArray(), copy=True).reshape(sdim, nq).T
        if vector_valued:
            gf.GetVectorValues(T, ir, vec, coords)
            v = np.array(vec.GetDataArray(), copy=True).reshape(-1, nq)
            v = np.linalg.norm(v, axis=0) if component is None else v[component]
        else:
            gf.GetValues(e, ir, scalar)
            v = np.array(scalar.GetDataArray(), copy=True)
        pts.append(x)
        cells.append(conn + npts)
        vals.append(v)
        npts += nq
    ncol = 2 if dim == 1 else 3
    local = (np.concatenate(pts) if pts else np.zeros((0, sdim)),
             np.concatenate(cells) if cells else np.zeros((0, ncol), dtype=np.int64),
             np.concatenate(vals) if vals else np.zeros(0))
    return _gather(space.comm, local)


def _draw(data, ax, mytitle, show_axis, vmin, vmax, colorbar, cmap, logscale):
    plt = _pyplot()
    points, cells, values = data
    if cells.shape[1] == 2:                       # 1D: a line over x
        order = np.argsort(points[:, 0], kind="stable")
        art = ax.plot(points[order, 0], values[order], "-")[0]
        if logscale:
            ax.set_yscale("log")
        if vmin is not None or vmax is not None:
            ax.set_ylim(vmin, vmax)
    else:
        import matplotlib.tri as mtri
        from matplotlib.colors import LogNorm

        tri = mtri.Triangulation(points[:, 0], points[:, 1], cells)
        if logscale:
            art = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap,
                               norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            art = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap,
                               vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.axis(show_axis)
        if colorbar:
            plt.colorbar(art, ax=ax, fraction=0.046, pad=0.04)
    if mytitle is not None:
        ax.set_title(mytitle)
    return art


def _axes(ax, subplot_loc):
    if ax is not None:
        return ax
    plt = _pyplot()
    if subplot_loc is not None:
        return plt.subplot(subplot_loc)
    return plt.gca()


def plot(obj, vector=None, subplot_loc=None, mytitle=None, show_axis="off",
         vmin=None, vmax=None, colorbar=True, cmap=None, logscale=False,
         refine=None, component=None, ax=None):
    """Plot a field, or a mesh.

    ``plot(space, vector)`` draws a field and ``plot(mesh)`` draws a mesh.
    Returns the matplotlib artist on rank 0 and ``None`` on the other ranks.
    """
    if vector is None:
        return plot_mesh(obj, subplot_loc=subplot_loc, mytitle=mytitle,
                         show_axis=show_axis, ax=ax)
    data = sample_field(obj, vector, refine=refine, component=component)
    if data is None:
        return None
    return _draw(data, _axes(ax, subplot_loc), mytitle, show_axis, vmin, vmax,
                 colorbar, cmap, logscale)


def plot_mesh(mesh, subplot_loc=None, mytitle=None, show_axis="off", ax=None,
              color="k", linewidth=0.4):
    """Draw the element edges of a 2D mesh, or the vertices of a 1D one."""
    if hasattr(mesh, "fes"):                      # a FunctionSpace
        mesh = mesh.mesh
    comm = _mesh_comm(mesh)
    dim = mesh.Dimension()
    verts = np.asarray(mesh.GetVertexArray(), dtype=float).reshape(
        mesh.GetNV(), mesh.SpaceDimension())
    segs = []
    for e in range(mesh.GetNE()):
        ids = list(mesh.GetElementVertices(e))
        loop = verts[ids + ids[:1]] if dim == 2 else verts[ids]
        segs.append(np.stack((loop[:-1], loop[1:]), axis=1) if dim == 2 else loop)
    local = np.concatenate(segs) if segs else np.zeros((0, 2, 2))
    parts = comm.gather(local, root=0) if comm.size > 1 else [local]
    if comm.rank != 0:
        return None
    ax = _axes(ax, subplot_loc)
    allsegs = np.concatenate(parts)
    if dim == 2:
        from matplotlib.collections import LineCollection

        art = ax.add_collection(LineCollection(allsegs[:, :, :2], colors=color,
                                               linewidths=linewidth))
        ax.autoscale()
        ax.set_aspect("equal")
    else:
        xs = np.unique(allsegs.reshape(-1, allsegs.shape[-1])[:, 0])
        art = ax.plot(xs, np.zeros_like(xs), "|-", color=color)[0]
    ax.axis(show_axis)
    if mytitle is not None:
        ax.set_title(mytitle)
    return art


def plot_pts(points, values=None, colorbar=True, subplot_loc=None, mytitle=None,
             show_axis="on", vmin=None, vmax=None, xlim=(0, 1), ylim=(0, 1),
             cmap=None, ax=None, marker_size=25):
    """Scatter points in 2D, colored by ``values`` when given.

    Not collective: ``points`` and ``values`` are plain arrays, so call it where
    they live (for observations, gather them first).
    """
    plt = _pyplot()
    ax = _axes(ax, subplot_loc)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if values is None:
        art = ax.scatter(points[:, 0], points[:, 1], s=marker_size, c="k")
    else:
        art = ax.scatter(points[:, 0], points[:, 1], s=marker_size,
                         c=np.asarray(values, dtype=float), cmap=cmap,
                         vmin=vmin, vmax=vmax, edgecolors="none")
        if colorbar:
            plt.colorbar(art, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis(show_axis)
    if mytitle is not None:
        ax.set_title(mytitle)
    return art


def multi1_plot(objs, titles, same_colorbar=True, show_axis="off", logscale=False,
                vmin=None, vmax=None, cmap=None, refine=None):
    """Plot fields side by side on one row.

    ``objs`` is a sequence of ``(space, vector)`` pairs; with ``same_colorbar``
    every panel shares the color range.
    """
    data = [sample_field(s, v, refine=refine) for s, v in objs]
    if data[0] is None:
        return None
    plt = _pyplot()
    n = len(data)
    lo, hi = vmin, vmax
    if same_colorbar:
        lo = min(d[2].min() for d in data) if vmin is None else vmin
        hi = max(d[2].max() for d in data) if vmax is None else vmax
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.2), squeeze=False)
    arts = [_draw(d, ax, t, show_axis, lo, hi, not same_colorbar, cmap, logscale)
            for d, ax, t in zip(data, axes[0], titles)]
    if same_colorbar and data[0][1].shape[1] == 3:
        fig.colorbar(arts[-1], ax=list(axes[0]), fraction=0.02, pad=0.02)
    return fig


def plot_eigenvalues(d, mytitle=None, subplot_loc=None, ax=None):
    """Generalized eigenvalues on a log scale, with the line :math:`\\lambda = 1`.

    Eigenvalues above the line are the directions in which the data inform the
    parameter more than the prior does.  Nonpositive values (round-off in the tail
    of a randomized eigensolver) cannot be drawn on a log scale and are left out.
    ``d`` is the same array on every rank; only rank 0 draws.
    """
    if MPI.COMM_WORLD.rank != 0:
        return None
    ax = _axes(ax, subplot_loc)
    d = np.asarray(d, dtype=float)
    idx = np.arange(d.size)
    keep = d > 0
    art = ax.semilogy(idx[keep], d[keep], "ob", markersize=4)[0]
    ax.semilogy([0, max(d.size - 1, 1)], [1.0, 1.0], "-r")
    ax.set_xlabel("number")
    ax.set_ylabel("eigenvalue")
    if mytitle is not None:
        ax.set_title(mytitle)
    return art


def plot_eigenvectors(space, U, mytitle="Eigenvector", which=(0, 1, 2, 5, 10, 15),
                      cmap=None, refine=None):
    """Plot selected columns of a MultiVector on a grid of up to 2x3 panels."""
    which = [i for i in which if i < len(U)][:6]
    data = [sample_field(space, U[i], refine=refine) for i in which]
    if not data or data[0] is None:
        return None
    plt = _pyplot()
    ncols = min(3, len(data))
    nrows = (len(data) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                             squeeze=False)
    for k, (i, d) in enumerate(zip(which, data)):
        _draw(d, axes.flat[k], "%s %d" % (mytitle, i), "off", None, None, True,
              cmap, False)
    for ax in list(axes.flat)[len(data):]:
        ax.axis("off")
    return fig


def show_solution(space, ic, state, same_colorbar=True, colorbar=True,
                  mytitle=None, show_axis="off", logscale=False,
                  times=(0.0, 0.4, 1.0, 2.0, 3.0, 4.0), cmap=None, refine=None):
    """Plot an initial condition and a trajectory at up to five later times.

    ``state`` is a :class:`~hippymfem.TimeDependentVector`; the first entry of
    ``times`` is drawn from ``ic`` and the others from the stored time closest to
    each.
    """
    space = as_space(space)
    u = space.vector()
    data = []
    for k, t in enumerate(times[:6]):
        if k == 0:
            data.append(sample_field(space, ic, refine=refine))
        else:
            state.retrieve(u, t)
            data.append(sample_field(space, u, refine=refine))
    if data[0] is None:
        return None
    plt = _pyplot()
    lo = min(d[2].min() for d in data) if same_colorbar else None
    hi = max(d[2].max() for d in data) if same_colorbar else None
    ncols = min(3, len(data))
    nrows = (len(data) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                             squeeze=False)
    prefix = "%s: " % mytitle if mytitle else ""
    for k, (t, d) in enumerate(zip(times, data)):
        _draw(d, axes.flat[k], "%st = %g" % (prefix, t), show_axis, lo, hi,
              colorbar, cmap, logscale)
    for ax in list(axes.flat)[len(data):]:
        ax.axis("off")
    return fig

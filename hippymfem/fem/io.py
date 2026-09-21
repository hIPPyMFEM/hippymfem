# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Writing fields out for visualization, and reading them back.

ParaView is the default because MFEM's ``ParaViewDataCollection`` handles
high-order fields natively, so a P2 state is written as P2 rather than silently
sampled onto vertices.
"""

import os

import numpy as np

import mfem.par as mfem
from .spaces import as_space

_FORMATS = {
    "binary": getattr(mfem, "VTKFormat_BINARY", None),
    "ascii": getattr(mfem, "VTKFormat_ASCII", None),
    "binary32": getattr(mfem, "VTKFormat_BINARY32", None),
}


def write_paraview(basename, mesh, fields, cycle=0, time=0.0, order=None,
                   data_format="binary", high_order=True):
    """Write named fields to a ParaView collection.

    Parameters
    ----------
    basename : str
        Output path; the directory is created and the collection named after the
        final path component.
    mesh : mfem.ParMesh
    fields : dict
        ``{name: (space, ParVector)}`` or ``{name: ParGridFunction}``.
    order : int, optional
        Levels of detail; defaults to the highest polynomial degree present, so
        high-order fields are not under-resolved in the output.
    """
    directory = os.path.dirname(os.path.abspath(basename)) or "."
    name = os.path.basename(basename)
    os.makedirs(directory, exist_ok=True)

    gfs = {}
    maxorder = 1
    for key, val in fields.items():
        if isinstance(val, tuple):
            space, vec = val
            space = as_space(space)
            gf = mfem.ParGridFunction(space.fes)
            gf.SetFromTrueDofs(vec.hypre)
            maxorder = max(maxorder, space.order)
        else:
            gf = val
            maxorder = max(maxorder, gf.ParFESpace().GetOrder(0))
        gfs[key] = gf            # must outlive Save()

    pvd = mfem.ParaViewDataCollection(name, mesh)
    pvd.SetPrefixPath(directory)
    pvd.SetLevelsOfDetail(int(order if order is not None else maxorder))
    fmt = _FORMATS.get(data_format)
    if fmt is not None:
        pvd.SetDataFormat(fmt)
    if high_order:
        try:
            pvd.SetHighOrderOutput(True)
        except Exception:
            pass
    for key, gf in gfs.items():
        pvd.RegisterField(key, gf)
    pvd.SetCycle(int(cycle))
    pvd.SetTime(float(time))
    pvd.Save()
    return os.path.join(directory, name)


class ParaViewWriter:
    """Incremental ParaView output, for time-dependent or iteration histories.

    Register the fields once, then call :meth:`save` per time level; the grid
    functions are reused so nothing is reallocated per step.
    """

    def __init__(self, basename, mesh, spaces, order=None,
                 data_format="binary", high_order=True):
        directory = os.path.dirname(os.path.abspath(basename)) or "."
        os.makedirs(directory, exist_ok=True)
        self.mesh = mesh
        self.spaces = {k: as_space(v) for k, v in spaces.items()}
        self.gfs = {k: mfem.ParGridFunction(v.fes)
                    for k, v in self.spaces.items()}
        maxorder = max([v.order for v in self.spaces.values()] or [1])
        self.pvd = mfem.ParaViewDataCollection(os.path.basename(basename), mesh)
        self.pvd.SetPrefixPath(directory)
        self.pvd.SetLevelsOfDetail(int(order if order is not None else maxorder))
        fmt = _FORMATS.get(data_format)
        if fmt is not None:
            self.pvd.SetDataFormat(fmt)
        if high_order:
            try:
                self.pvd.SetHighOrderOutput(True)
            except Exception:
                pass
        for k, gf in self.gfs.items():
            gf.Assign(0.0)
            self.pvd.RegisterField(k, gf)
        self._cycle = 0

    def save(self, values, time=None, cycle=None):
        """Write one time level; ``values`` maps field name to a ParVector."""
        for k, v in values.items():
            self.gfs[k].SetFromTrueDofs(v.hypre)
        self.pvd.SetCycle(int(self._cycle if cycle is None else cycle))
        self.pvd.SetTime(float(self._cycle if time is None else time))
        self.pvd.Save()
        self._cycle += 1
        return self


def write_glvis(basename, mesh, fields):
    """Write MFEM's native mesh/solution files, readable by GLVis."""
    directory = os.path.dirname(os.path.abspath(basename)) or "."
    os.makedirs(directory, exist_ok=True)
    rank = mesh.GetComm().rank if hasattr(mesh.GetComm(), "rank") else 0
    mesh_name = "%s_mesh.%06d" % (basename, rank)
    mesh.Print(mesh_name, 8)
    out = [mesh_name]
    for key, val in fields.items():
        space, vec = val
        space = as_space(space)
        gf = mfem.ParGridFunction(space.fes)
        gf.SetFromTrueDofs(vec.hypre)
        fn = "%s_%s.%06d" % (basename, key, rank)
        gf.Save(fn, 8)
        out.append(fn)
    return out


def save_vector(path, v, comm=None):
    """Save a distributed vector as a single ``.npy`` on rank 0."""
    comm = comm if comm is not None else v.comm
    full = v.gather_to_zero()
    if comm.rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.save(path, full)
    comm.Barrier()
    return path


def load_vector(path, v, comm=None):
    """Load a vector saved by :func:`save_vector` back into ``v``."""
    comm = comm if comm is not None else v.comm
    full = np.load(path) if comm.rank == 0 else None
    return v.scatter_from_zero(full)


def write_point_csv(path, points, values, comm, header=None):
    """Write point coordinates and values as CSV on rank 0."""
    if comm.rank != 0:
        return None
    points = np.atleast_2d(np.asarray(points, dtype=float))
    values = np.asarray(values, dtype=float).reshape(points.shape[0], -1)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arr = np.column_stack((points, values))
    if header is None:
        header = (",".join(["x", "y", "z"][: points.shape[1]]) + ","
                  + ",".join("v%d" % i for i in range(values.shape[1])))
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    return path

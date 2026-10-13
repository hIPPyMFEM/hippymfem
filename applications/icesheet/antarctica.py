# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The Antarctic ice sheet: geometry, climate and surface velocity on a model grid.

Sources (fetched by ``python antarctica.py fetch`` into ``data/``, not distributed here):

* ALBMAP v1 (Le Brocq, Payne and Vieli, Earth Syst. Sci. Data 2, 2010; PANGAEA,
  CC-BY 3.0): bed and surface elevations, the grounded / floating mask, surface
  temperature, accumulation and geothermal heat flux on a 5 km polar stereographic grid
  (EPSG:3031).  It is the dataset of the paper (Isaac, Petra, Stadler and Ghattas, J.
  Comput. Phys. 296, 2015).
* ITS_LIVE v1 (Gardner et al., NASA MEaSUREs; public on AWS): the surface velocity
  mosaic of Antarctica, its 1.2 km overview (the 10 % overview of the 120 m product).
  It is made from Landsat images, which do not reach south of 82.7 S: it has no
  velocity within about 700 km of the pole.
* MEaSUREs InSAR-Based Antarctica Ice Velocity Map, version 2 (Rignot, Mouginot and
  Scheuchl, 2017; the paper's velocities are its first version), as the Quantarctica
  package of the Norwegian Polar Institute redistributes it: the speed at 450 m and
  the velocity vectors every 20 km.  It covers the pole, and fills what ITS_LIVE lacks
  (:meth:`ModelGrid.attach_velocity`); where both exist they agree to about a tenth of
  the speed for ice faster than 5 m/a.

Units of the model: kilometres, years, megapascals.  Velocities are in km/a, the basal
friction coefficient exp(beta) in MPa a / km (the paper's MPa / (km/a)), the rate
factor A in MPa^-3 / a.

A model grid coarsens the 5 km cells by an integer factor ``r``: a coarse cell is ice
when most of its 5 km cells are grounded ice at least ``H_MIN`` thick.  The ice sheet is
the set of those cells, one column of hexahedra each (:class:`Columns`).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ALBMAP_URL = "https://store.pangaea.de/Publications/LeBrocq_et_al_2010/ALBMAPv1.nc.zip"
ITSLIVE_URL = ("https://its-live-data.s3.amazonaws.com/velocity_mosaic/landsat/v00.0/static/cog/"
               "ANT_G0120_0000_%s_10_percent_overview.tif")
MEASURES_URL = ("https://media.npolar.no/download/quantarctica/Quantarctica3/Glaciology/"
                "MEaSUREs%20Ice%20Flow%20Velocity/")
MEASURES_FILES = ("MEaSUREs_IceFlowSpeed_450m.tif", "MEaSUREs_FlowVectors.shp", "MEaSUREs_FlowVectors.shx",
                  "MEaSUREs_FlowVectors.dbf")

RHO_I = 910.0                       # kg/m^3, the paper's
RHO_W = 1028.0                      # sea water
G = 9.81
RHO_G = RHO_I * G * 1e-3            # MPa / km: 8.927
RHO_W_G = RHO_W * G * 1e-3          # MPa / km
SECONDS_PER_YEAR = 3.15576e7
H_MIN = 50.0                        # m: thinner grounded ice is left out
GLEN_N = 3.0
K_ICE = 2.1                         # W/(m K)
KAPPA_ICE = 1.09e-6 * SECONDS_PER_YEAR   # m^2 / a
BETA_CC = 8.7e-4                    # K / m, pressure melting
T_RANGE = (-50.0, -10.0)            # deg C: the paper's range for the rate factor


# ------------------------------------------------------------------ the data
def fetch(folder=DATA):
    """Download ALBMAP, the ITS_LIVE overviews and the MEaSUREs speed and vectors (about
    570 MB)."""
    import urllib.request
    import zipfile

    os.makedirs(folder, exist_ok=True)
    nc = os.path.join(folder, "ALBMAPv1.nc")
    if not os.path.exists(nc):
        z = nc + ".zip"
        urllib.request.urlretrieve(ALBMAP_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(folder)
    for v in ("vx", "vy", "vx_err", "vy_err"):
        out = os.path.join(folder, "ANT_G0120_0000_%s_10_percent_overview.tif" % v)
        if not os.path.exists(out):
            urllib.request.urlretrieve(ITSLIVE_URL % v, out)
    for name in MEASURES_FILES:
        out = os.path.join(folder, name)
        if not os.path.exists(out):
            urllib.request.urlretrieve(MEASURES_URL + name, out)


def load_albmap(folder=DATA):
    """ALBMAP's fields as ``(ny, nx)`` arrays (row 0 the southernmost), and the cell
    centres ``x, y`` in metres."""
    import netCDF4

    f = netCDF4.Dataset(os.path.join(folder, "ALBMAPv1.nc"))
    out = {"x": np.asarray(f.variables["x1"][:], float), "y": np.asarray(f.variables["y1"][:], float)}
    for k in ("topg", "lsrf", "usrf", "mask", "mask_plus", "temp", "acca", "accr", "ghfsr"):
        a = np.ma.filled(f.variables[k][:], np.nan).astype(float)
        a[a <= -9998] = np.nan
        out[k] = a
    f.close()
    return out


def load_velocity(folder=DATA):
    """ITS_LIVE vx, vy and their errors (m/a, NaN where missing) and the pixel centres."""
    import rasterio

    out = {}
    for v in ("vx", "vy", "vx_err", "vy_err"):
        with rasterio.open(os.path.join(folder, "ANT_G0120_0000_%s_10_percent_overview.tif" % v)) as r:
            a = r.read(1).astype(float)
            a[a == r.nodata] = np.nan
            if v.endswith("_err"):
                a[a <= 0] = np.nan
            out[v] = a
            T = r.transform
    ny, nx = out["vx"].shape
    out["x"] = T.c + T.a * (np.arange(nx) + 0.5)
    out["y"] = T.f + T.e * (np.arange(ny) + 0.5)          # descending
    return out


def load_measures(folder=DATA):
    """The MEaSUREs speed (m/a, NaN where missing) with its pixel centres, and the
    velocity vectors given every 20 km: ``points`` (m) and ``vectors`` (m/a)."""
    import rasterio
    import shapefile

    with rasterio.open(os.path.join(folder, "MEaSUREs_IceFlowSpeed_450m.tif")) as r:
        speed = r.read(1)
        speed[speed == r.nodata] = np.nan
        T = r.transform
    sf = shapefile.Reader(os.path.join(folder, "MEaSUREs_FlowVectors"))
    return {"speed": speed, "x": T.c + T.a * (np.arange(speed.shape[1]) + 0.5),
            "y": T.f + T.e * (np.arange(speed.shape[0]) + 0.5),            # descending
            "points": np.array([sh.points[0] for sh in sf.shapes()], float),
            "vectors": np.array([[rec[0], rec[1]] for rec in sf.records()], float)}


# ------------------------------------------------------- the thermal model
def column_temperature(Ts, acc, ghf, H, zeta):
    """Steady temperature (deg C) of a column of thickness ``H`` (m) at the heights
    ``zeta * H`` above the bed: Robin's (1955) solution of vertical advection and
    diffusion, with the accumulation ``acc`` (m ice / a) as the surface's downward
    velocity falling linearly to zero at the bed, the surface temperature ``Ts`` on top
    and the geothermal flux ``ghf`` (mW/m^2) at the base, capped at the pressure melting
    point.  ``Ts, acc, ghf, H`` are arrays over the columns, ``zeta`` over the levels."""
    from scipy.special import erf

    Ts, acc, ghf, H = (np.asarray(a, float)[:, None] for a in (Ts, acc, ghf, H))
    z = np.asarray(zeta, float)[None, :] * H
    G = ghf * 1e-3
    a = np.maximum(acc, 1e-3)
    ell = np.sqrt(2.0 * KAPPA_ICE * H / a)
    T = Ts + np.sqrt(np.pi) * ell * G / (2.0 * K_ICE) * (erf(H / ell) - erf(z / ell))
    Tpm = -BETA_CC * (H - z)
    return np.minimum(T, Tpm)


def rate_factor(T):
    """Paterson and Budd (1982): A(T) in MPa^-3 / a, ``T`` in deg C, clamped to the
    paper's -50 .. -10 deg C (the range of their colder branch)."""
    Tk = np.clip(np.asarray(T, float), *T_RANGE) + 273.15
    return 3.985e-13 * np.exp(-60e3 / (8.314 * Tk)) * 1e18 * SECONDS_PER_YEAR


# ------------------------------------------------------------- model grids
class ModelGrid:
    """ALBMAP coarsened by ``r`` (cells of 5 r km), the grounded ice on it, and the
    fields the model needs at its nodes.

    Cells ``(J, I)`` (row J northward, column I eastward); nodes ``(J, I)`` at their
    south-west corners, ``(ny + 1, nx + 1)``.  Attributes, in model units (km, km/a):

    ``ice``      cells of the ice sheet
    ``s, b``     surface and bed elevation at the nodes, averaged over the ice cells
                 around each node
    ``uobs``     observed surface velocity ``(ny+1, nx+1, 2)`` (NaN where missing),
                 the mean of the 1.2 km pixels within half a cell of the node
    ``Ts, acc, ghf``  surface temperature (deg C), accumulation (m ice / a) and
                 geothermal flux (mW / m^2) at the nodes
    ``marine``   per ice cell and side (S, E, N, W): the neighbour across that side is
                 not ice and the margin there faces the ocean (floating ice, ocean, or a
                 bed below sea level); otherwise it ends on land
    """

    def __init__(self, r, albmap=None, vel=None, min_component=4, fill=None):
        A = albmap if albmap is not None else load_albmap()
        self.r = int(r)
        r = self.r
        ny5, nx5 = A["usrf"].shape
        ny, nx = ny5 // r, nx5 // r
        self.ny, self.nx = ny, nx
        self.h = 5.0 * r                                       # km
        thk = A["usrf"] - A["lsrf"]
        grounded = (A["mask"] == 1) & (thk >= H_MIN) & (A["mask_plus"] != 3)

        def blocks(a):
            return a[:ny * r, :nx * r].reshape(ny, r, nx, r).transpose(0, 2, 1, 3).reshape(ny, nx, r * r)

        g = blocks(grounded.astype(float))
        ice = g.mean(axis=2) > 0.5
        # the main ice sheet and the larger ice rises; specks of a few cells go
        ice = _keep_components(ice, min_component)
        self.ice = ice
        w = np.where(blocks(grounded), 1.0, 0.0)
        cnt = np.maximum(w.sum(axis=2), 1.0)
        Hc = (blocks(np.nan_to_num(thk)) * w).sum(axis=2) / cnt
        sc = (blocks(np.nan_to_num(A["usrf"])) * w).sum(axis=2) / cnt
        bc = sc - Hc
        floating = blocks((A["mask"] != 1).astype(float)).mean(axis=2) > 0.5
        self.cell_H, self.cell_s, self.cell_b = Hc / 1e3, sc / 1e3, bc / 1e3
        # node values: the mean over the ice cells around each node
        self.s = _node_mean(np.where(ice, sc, 0.0), ice) / 1e3
        self.b = _node_mean(np.where(ice, bc, 0.0), ice) / 1e3
        self.node_used = _node_mean(ice.astype(float), np.ones_like(ice)) > 0
        # coordinates (km): ALBMAP cell centres are 5 km apart; node (J, I) is the
        # south-west corner of coarse cell (J, I)
        x0 = A["x"][0] - 2500.0
        y0 = A["y"][0] - 2500.0
        self.xn = (x0 + 5000.0 * r * np.arange(nx + 1)) / 1e3
        self.yn = (y0 + 5000.0 * r * np.arange(ny + 1)) / 1e3
        # climate at the nodes: averaged as the geometry is
        for name, key in (("Ts", "temp"), ("acc", "acca"), ("ghf", "ghfsr")):
            v = A[key]
            vfill = np.where(np.isfinite(v), v, np.nan)
            m = blocks(np.isfinite(vfill) & grounded)
            c = np.where(m.sum(axis=2) > 0, np.nansum(np.where(m, blocks(np.nan_to_num(vfill)), 0.0), axis=2)
                         / np.maximum(m.sum(axis=2), 1), np.nan)
            c = _fill_nan(c)
            setattr(self, name, _node_mean(np.where(ice, c, 0.0), ice))
        self.acc = np.maximum(self.acc, 0.01)
        # margins: what lies across each side of an ice cell
        pad = np.pad(ice, 1, constant_values=False)
        fpad = np.pad(floating, 1, constant_values=True)
        bpad = np.pad(bc, 1, mode="edge")
        nb = {"S": (0, 1), "E": (1, 2), "N": (2, 1), "W": (1, 0)}
        self.marine = {}
        for side, (dj, di) in nb.items():
            out_ice = ~pad[dj:dj + ny, di:di + nx]
            fl = fpad[dj:dj + ny, di:di + nx]
            below = 0.5 * (bc + bpad[dj:dj + ny, di:di + nx]) < 0.0
            self.marine[side] = ice & out_ice & (fl | below)
            setattr(self, "edge_" + side, ice & out_ice)
        self.uobs = None
        if vel is not None:
            self.attach_velocity(vel, fill)

    def attach_velocity(self, V, M=None):
        """Mean of the 1.2 km pixels of ITS_LIVE within half a cell of each node (km/a),
        where at least half of them are present.  With ``M`` (:func:`load_measures`), a
        node left empty takes the MEaSUREs velocity: the mean of its speed over the same
        square, along the mean of its vectors within 30 km (or half a cell, if that is
        more); ``ufill`` marks those nodes."""
        from scipy.spatial import cKDTree

        hx = 1e3 * self.h / 2.0
        X, Y = V["x"], V["y"]
        out = np.full(self.s.shape + (2,), np.nan)
        err = np.full(self.s.shape + (2,), np.nan)
        filled = np.zeros(self.s.shape, dtype=bool)
        if M is not None:
            MX, MY, tree = M["x"], M["y"], cKDTree(M["points"])
            mdx, mdy = MX[1] - MX[0], MY[1] - MY[0]
        dx = X[1] - X[0]
        dy = Y[1] - Y[0]                                     # negative
        for J in range(self.ny + 1):
            yc = 1e3 * self.yn[J]
            r0 = int(np.floor((Y[0] - (yc + hx)) / -dy))
            r1 = int(np.ceil((Y[0] - (yc - hx)) / -dy))
            r0, r1 = max(r0, 0), min(r1, Y.size)
            if r1 <= r0:
                continue
            for I in np.nonzero(self.node_used[J])[0]:
                xc = 1e3 * self.xn[I]
                c0 = int(np.floor((xc - hx - X[0]) / dx))
                c1 = int(np.ceil((xc + hx - X[0]) / dx))
                c0, c1 = max(c0, 0), min(c1, X.size)
                if c1 <= c0:
                    continue
                for k, key in enumerate(("vx", "vy")):
                    blk = V[key][r0:r1, c0:c1]
                    ok = np.isfinite(blk)
                    if ok.mean() >= 0.5:
                        out[J, I, k] = blk[ok].mean() / 1e3
                        e = V[key + "_err"][r0:r1, c0:c1]
                        e = e[np.isfinite(e)]
                        err[J, I, k] = (np.sqrt(np.mean(e ** 2) / max(e.size, 1)) / 1e3 if e.size else np.nan)
                if M is None or np.isfinite(out[J, I]).all():
                    continue
                out[J, I] = np.nan                           # both components or neither
                a0, a1 = int(np.floor((xc - hx - MX[0]) / mdx)), int(np.ceil((xc + hx - MX[0]) / mdx))
                b0, b1 = int(np.floor((MY[0] - (yc + hx)) / -mdy)), int(np.ceil((MY[0] - (yc - hx)) / -mdy))
                blk = M["speed"][max(b0, 0):min(b1, MY.size), max(a0, 0):min(a1, MX.size)]
                ok = np.isfinite(blk)
                near = tree.query_ball_point([xc, yc], 1.01 * max(hx, 30e3))
                if not blk.size or ok.mean() < 0.5 or not near:
                    continue
                d = M["vectors"][near].mean(axis=0)
                if np.hypot(*d) > 0.0:
                    out[J, I] = blk[ok].mean() / 1e3 * d / np.hypot(*d)
                    filled[J, I] = True
        self.uobs, self.uobs_err, self.ufill = out, err, filled
        return self

    # ------------------------------------------------------------------ cache
    _FIELDS = ("ice", "cell_H", "cell_s", "cell_b", "s", "b", "node_used", "xn", "yn", "Ts", "acc", "ghf",
               "edge_S", "edge_E", "edge_N", "edge_W", "uobs", "uobs_err", "ufill")

    @classmethod
    def load(cls, r, folder=DATA):
        """The grid of factor ``r`` from ``grid_r<r>.npz`` in ``folder`` (made from the raw
        data the first time, where netCDF4 and rasterio are installed)."""
        path = os.path.join(folder, "grid_r%d.npz" % r)
        if not os.path.exists(path) or "ufill" not in np.load(path).files:
            g = cls(r, vel=load_velocity(), fill=load_measures())
            np.savez(path, r=r, ny=g.ny, nx=g.nx, h=g.h,
                     **{k: getattr(g, k) for k in cls._FIELDS},
                     **{"marine_" + k: v for k, v in g.marine.items()})
            return g
        z = np.load(path)
        g = cls.__new__(cls)
        g.r, g.ny, g.nx, g.h = int(z["r"]), int(z["ny"]), int(z["nx"]), float(z["h"])
        for k in cls._FIELDS:
            setattr(g, k, z[k])
        g.marine = {k: z["marine_" + k] for k in "SENW"}
        return g

    # ------------------------------------------------------------------ fields
    def temperature(self, zeta):
        """Column temperature (deg C) at the nodes, ``(ny+1, nx+1, len(zeta))``."""
        H = np.maximum(1e3 * (self.s - self.b), H_MIN)
        T = column_temperature(self.Ts.ravel(), self.acc.ravel(), self.ghf.ravel(), H.ravel(), zeta)
        return T.reshape(self.s.shape + (len(zeta),))

    def driving_stress_friction(self, smooth=2):
        """A basal friction coefficient exp(beta) (MPa a / km) from the driving stress
        and the observed speed, ``tau_d / |u|`` (the paper's Figure 1 builds its
        synthetic field the same way), capped to 1 .. 1e4 and smoothed over ``smooth``
        cells.  At the nodes."""
        from scipy.ndimage import uniform_filter

        sx = np.gradient(self.s, self.h, axis=1)
        sy = np.gradient(self.s, self.h, axis=0)
        tau = RHO_G * np.maximum(self.s - self.b, H_MIN / 1e3) * np.hypot(sx, sy)
        speed = np.hypot(self.uobs[..., 0], self.uobs[..., 1]) if self.uobs is not None else np.full_like(tau, np.nan)
        speed = _fill_nan(np.where(np.isfinite(speed), speed, np.nan))
        fr = np.clip(tau / np.maximum(speed, 1e-4), 1.0, 1e4)
        lf = uniform_filter(np.log(fr), size=2 * smooth + 1, mode="nearest")
        return np.exp(lf)

    def summary(self):
        n_obs = int(np.isfinite(self.uobs[..., 0])[self.node_used].sum()) if self.uobs is not None else 0
        return {"r": self.r, "h_km": self.h, "cells": int(self.ice.sum()), "nodes": int(self.node_used.sum()),
                "area_km2": float(self.ice.sum() * self.h ** 2),
                "marine_sides": int(sum(m.sum() for m in self.marine.values())),
                "land_sides": int(sum((getattr(self, "edge_" + s) & ~self.marine[s]).sum() for s in "SENW")),
                "thickness_km": [float((self.s - self.b)[self.node_used].min()),
                                 float((self.s - self.b)[self.node_used].max())],
                "velocity_nodes": n_obs,
                "velocity_nodes_filled": int(self.ufill[self.node_used].sum()) if self.uobs is not None else 0}


def _keep_components(ice, min_cells):
    from scipy.ndimage import label

    lab, n = label(ice)
    if n == 0:
        return ice
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_cells
    keep[0] = False
    return keep[lab]


def _node_mean(cellvals, weight):
    """Mean over the (up to four) cells around each node, weighted by ``weight``."""
    w = np.pad(weight.astype(float), 1)
    v = np.pad(cellvals * weight, 1)
    S = v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:]
    W = w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:]
    return np.where(W > 0, S / np.maximum(W, 1e-300), 0.0)


def _fill_nan(a):
    """NaNs replaced by the nearest finite value."""
    from scipy.ndimage import distance_transform_edt

    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


# ------------------------------------------------------------------- meshes
class Columns:
    """The hexahedral mesh of a model grid: one column of ``nz`` elements per ice cell,
    following the bed and the surface (the levels at ``zeta``, 0 at the bed).

    Numbering: the 2D nodes in use get ids ``0 .. n2 - 1`` (row by row); the 3D vertex of
    node ``i`` at level ``k`` is ``k * n2 + i``.  Boundary attributes: 1 the bed, 2 the
    surface, 3 a margin facing the ocean, 4 a margin on land.
    """

    BED, SURFACE, MARINE, LAND = 1, 2, 3, 4

    def __init__(self, grid, nz, zeta=None, min_thickness=H_MIN / 1e3):
        self.grid = grid
        self.nz = int(nz)
        self.zeta = np.linspace(0.0, 1.0, self.nz + 1) if zeta is None else np.asarray(zeta, float)
        used = grid.node_used
        self.node_id = -np.ones(used.shape, np.int64)
        self.node_id[used] = np.arange(int(used.sum()))
        self.n2 = int(used.sum())
        J, I = np.nonzero(used)
        self.node_JI = np.column_stack((J, I))
        self.xy = np.column_stack((grid.xn[I], grid.yn[J]))
        H = np.maximum(grid.s[used] - grid.b[used], min_thickness)
        self.bed = grid.b[used]
        self.H = H
        cJ, cI = np.nonzero(grid.ice)
        self.cell_JI = np.column_stack((cJ, cI))
        nid = self.node_id
        self.quads = np.column_stack((nid[cJ, cI], nid[cJ, cI + 1], nid[cJ + 1, cI + 1], nid[cJ + 1, cI]))
        if (self.quads < 0).any():
            raise RuntimeError("an ice cell has a corner node that is not in use")

    def vertices(self):
        """``(n2 * (nz + 1), 3)`` vertex coordinates (km)."""
        z = self.bed[None, :] + self.zeta[:, None] * self.H[None, :]
        xy = np.tile(self.xy, (self.nz + 1, 1))
        return np.column_stack((xy, z.ravel()))

    def hexes(self):
        q, n2 = self.quads, self.n2
        out = [np.column_stack((q + k * n2, q + (k + 1) * n2)) for k in range(self.nz)]
        return np.vstack(out)

    def boundary(self):
        """``(faces (nf, 4), attributes (nf,))``, each face ordered with its normal out."""
        q, n2, nz = self.quads, self.n2, self.nz
        faces = [q[:, ::-1], q + nz * n2]                     # bed (normal down), surface (up)
        attrs = [np.full(len(q), self.BED), np.full(len(q), self.SURFACE)]
        g = self.grid
        cJ, cI = self.cell_JI[:, 0], self.cell_JI[:, 1]
        # side faces: the cell's corners SW, SE, NE, NW are q[:, 0..3]
        pairs = {"S": (0, 1), "E": (1, 2), "N": (2, 3), "W": (3, 0)}
        for side, (a, bb) in pairs.items():
            edge = getattr(g, "edge_" + side)[cJ, cI]
            mar = g.marine[side][cJ, cI]
            for k in range(nz):
                lo, hi = k * n2, (k + 1) * n2
                f = np.column_stack((q[edge, a] + lo, q[edge, bb] + lo, q[edge, bb] + hi, q[edge, a] + hi))
                faces.append(f)
                attrs.append(np.where(mar[edge], self.MARINE, self.LAND))
        return np.vstack(faces), np.concatenate(attrs)

    def mesh3d(self):
        """The serial MFEM mesh (hexahedra)."""
        import mfem.par as mfem

        V, E = self.vertices(), self.hexes()
        F, A = self.boundary()
        m = mfem.Mesh(3, V.shape[0], E.shape[0], F.shape[0], 3)
        for v in V:
            m.AddVertex(*v)
        for e in E:
            m.AddHex(*[int(t) for t in e], 1)
        for f, a in zip(F, A):
            m.AddBdrQuad(*[int(t) for t in f], int(a))
        m.FinalizeHexMesh(1, 1, True)
        return m

    def mesh2d(self):
        """The serial MFEM mesh of the footprint (quadrilaterals, the parameter's mesh),
        with the margin edges as its boundary (attribute 1)."""
        import mfem.par as mfem

        q = self.quads
        edges = {}
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            for u, v in zip(q[:, a], q[:, b]):
                key = (min(u, v), max(u, v))
                edges[key] = edges.get(key, 0) + 1
        bnd = [k for k, c in edges.items() if c == 1]
        m = mfem.Mesh(2, self.n2, len(q), len(bnd), 2)
        for v in self.xy:
            m.AddVertex(float(v[0]), float(v[1]))
        for e in q:
            m.AddQuad(*[int(t) for t in e], 1)
        for u, v in bnd:
            m.AddBdrSegment(int(u), int(v), 1)
        m.FinalizeQuadMesh(1, 1, True)
        return m


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    elif len(sys.argv) > 1 and sys.argv[1] == "cache":
        for r in (32, 16, 8, 4, 2, 1):
            print(ModelGrid.load(r).summary(), flush=True)
    else:
        A, V = load_albmap(), load_velocity()
        for r in (8, 4, 2, 1):
            g = ModelGrid(r, A, V)
            print(g.summary())

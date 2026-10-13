# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Figures of a run of ``run.py``, from what it saved (no MFEM needed).

    python figures.py out/n32
    python figures.py --spectra out/n32 out/n48 -o spectra.png
    python figures.py --scaling weak_r*.json -o scaling.png

The parameter fields are saved in the order of their vertex keys, which name each
vertex by its layer and its place on the cubed sphere, so a spherical shell or a
plane through the centre is a structured grid read off the keys: nothing is
interpolated except between two layers, for a shell at a depth that is not one.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import earth  # noqa: E402


class Fields:
    """The saved fields of a run and the structure of the vertices they live on."""

    def __init__(self, out):
        self.out = out
        self.setup = json.load(open(os.path.join(out, "setup.json")))
        a = self.setup["args"]
        self.bm = earth.BallMesh(a["n"], cube=a["cube"])
        self.refine = a["refine"]
        v = np.load(os.path.join(out, "vertices.npz"))
        self.keys, self.xyz = v["key"], v["xyz"]
        self.nf = a["n"] * 2 ** self.refine                       # cells along an edge
        self.m = self.nf + 1
        self.L = self.bm.nlev * 2 ** self.refine                  # layers outside the cube
        self.lev0 = self.bm.n_core * 2 ** self.refine             # the core-mantle boundary's level
        self.radii = np.array([self.bm.level_radius(l, self.refine) for l in range(self.L + 1)])

    def has(self, name):
        return os.path.exists(os.path.join(self.out, name))

    def load(self, name):
        return np.load(os.path.join(self.out, name))

    def index(self, lev, I, J, K):
        m = self.m
        key = ((np.asarray(lev, dtype=np.int64) * m + I) * m + J) * m + K
        idx = np.searchsorted(self.keys, key)
        if not np.array_equal(self.keys[np.minimum(idx, self.keys.size - 1)], key):
            raise ValueError("a requested vertex is not in the mesh")
        return idx

    # ----------------------------------------------------------------- a shell
    def shell(self, lev):
        """Vertex indices and triangles of the sphere of layer ``lev``."""
        n = self.nf
        a = np.arange(n + 1)
        P, Q = np.meshgrid(a, a, indexing="ij")
        Z, E = np.zeros_like(P), np.full_like(P, n)
        faces = [(Z, P, Q), (E, P, Q), (P, Z, Q), (P, E, Q), (P, Q, Z), (P, Q, E)]
        idx, tri, off = [], [], 0
        loc = (P * (n + 1) + Q)
        quads = np.stack([loc[:-1, :-1], loc[1:, :-1], loc[1:, 1:], loc[:-1, 1:]], axis=-1).reshape(-1, 4)
        for (I, J, K) in faces:
            idx.append(self.index(lev, I, J, K).ravel())
            tri.append(off + np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))
            off += (n + 1) ** 2
        return np.concatenate(idx), np.concatenate(tri)

    def at_depth(self, field, depth):
        """``(xyz, values, triangles)`` of ``field`` on the sphere ``depth`` km down,
        interpolated between the two layers around it."""
        r = earth.R_EARTH - depth
        rr = self.radii[self.lev0:]
        k = int(np.clip(np.searchsorted(rr, r) - 1, 0, rr.size - 2))
        l0, l1 = self.lev0 + k, self.lev0 + k + 1
        w = (r - rr[k]) / (rr[k + 1] - rr[k])
        i0, tri = self.shell(l0)
        i1, _ = self.shell(l1)
        xyz = (1 - w) * self.xyz[i0] + w * self.xyz[i1]
        return xyz, (1 - w) * field[..., i0] + w * field[..., i1], tri

    # ----------------------------------------------------------------- a plane
    def plane(self, axis):
        """Vertex indices and triangles of the plane through the centre normal to
        ``axis`` (0, 1 or 2); the number of cells along an edge must be even."""
        n = self.nf
        if n % 2:
            raise ValueError("a plane through the centre needs an even number of cells")
        h = n // 2
        a = np.arange(n + 1)
        U, V = np.meshgrid(a, a, indexing="ij")

        def ijk(u, v):
            c = [None, None, None]
            o = [d for d in range(3) if d != axis]
            c[axis] = np.full_like(u, h)
            c[o[0]], c[o[1]] = u, v
            return c

        idx = [self.index(0, *ijk(U, V)).ravel()]
        loc = U * (n + 1) + V
        quads = [np.stack([loc[:-1, :-1], loc[1:, :-1], loc[1:, 1:], loc[:-1, 1:]], axis=-1).reshape(-1, 4)]
        # the square's perimeter, counterclockwise, and the loops of the layers outside
        pu = np.concatenate([a[:-1], np.full(n, n), a[:0:-1], np.zeros(n, dtype=int)])
        pv = np.concatenate([np.zeros(n, dtype=int), a[:-1], np.full(n, n), a[:0:-1]])
        prev = loc[pu, pv]
        off = (n + 1) ** 2
        for lev in range(1, self.L + 1):
            idx.append(self.index(lev, *ijk(pu, pv)))
            cur = off + np.arange(4 * n)
            nxt = np.roll(np.arange(4 * n), -1)
            quads.append(np.stack([prev, prev[nxt], cur[nxt], cur], axis=-1))
            prev, off = cur, off + 4 * n
        quads = np.concatenate(quads)
        tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        return np.concatenate(idx), tri


# ---------------------------------------------------------------------- drawing
def _frame(lon0, lat0):
    lo, la = np.radians(lon0), np.radians(lat0)
    v = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    e1 = np.array([-np.sin(lo), np.cos(lo), 0.0])
    e2 = np.cross(v, e1)
    return v, e1, e2


def globe(ax, xyz, val, tri, lon0, lat0, vmin, vmax, cmap, points=()):
    """An orthographic view of a sphere's field from above ``(lon0, lat0)``."""
    import matplotlib.tri as mtri

    v, e1, e2 = _frame(lon0, lat0)
    R = np.linalg.norm(xyz, axis=1).mean()
    depth = xyz @ v
    keep = (depth[tri] > -0.02 * R).all(axis=1)
    T = mtri.Triangulation(xyz @ e1, xyz @ e2, tri[keep])
    im = ax.tripcolor(T, val, shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(R * np.cos(th), R * np.sin(th), "k", lw=0.6)
    for pts, style in points:
        pts = np.asarray(pts) * (R / np.linalg.norm(pts, axis=1))[:, None]
        vis = pts @ v > 0
        ax.plot(pts[vis] @ e1, pts[vis] @ e2, **style)
    ax.set_aspect("equal")
    ax.set_xlim(-1.04 * R, 1.04 * R)
    ax.set_ylim(-1.04 * R, 1.04 * R)
    ax.axis("off")
    return im


def disc(ax, xy, val, tri, vmin, vmax, cmap):
    import matplotlib.tri as mtri

    T = mtri.Triangulation(xy[:, 0], xy[:, 1], tri)
    im = ax.tripcolor(T, val, shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    th = np.linspace(0, 2 * np.pi, 361)
    for r, lw in ((earth.R_EARTH, 0.6), (earth.R_CMB, 0.4), (earth.R_ICB, 0.3)):
        ax.plot(r * np.cos(th), r * np.sin(th), "k", lw=lw)
    ax.set_aspect("equal")
    ax.axis("off")
    return im


REC = dict(marker="o", ls="", ms=2.2, mfc="w", mec="k", mew=0.4)
SRC = dict(marker="*", ls="", ms=9, mfc="k", mec="w", mew=0.5)
VIEWS = (("receiver side", 90.0, 35.0), ("far side", -90.0, -35.0))


def fig_model(F, z, depths=(67.0, 222.0), path=None):
    """The truth and the MAP point at two depths, seen from the side with the
    receivers and from the opposite one."""
    import matplotlib.pyplot as plt

    truth, m_map = F.load("truth.npy"), F.load("map.npy")
    pts = ((z["rec"], REC), (z["src"], SRC))
    fig, axs = plt.subplots(len(depths), 4, figsize=(13.0, 3.3 * len(depths)))
    axs = np.atleast_2d(axs)
    for i, dep in enumerate(depths):
        xyz, tv, tri = F.at_depth(truth, dep)
        _, mv, _ = F.at_depth(m_map, dep)
        lim = np.abs(tv).max()
        for j, (name, lon0, lat0) in enumerate(VIEWS):
            for kk, (val, lab) in enumerate(((tv, "truth"), (mv, "MAP"))):
                ax = axs[i, 2 * j + kk]
                im = globe(ax, xyz, val, tri, lon0, lat0, -lim, lim, "RdBu", pts)
                ax.set_title("%s, %s, %g km" % (lab, name, dep), fontsize=9)
        cb = fig.colorbar(im, ax=axs[i, :], shrink=0.75, pad=0.01)
        cb.set_label("wave speed anomaly, km/s", fontsize=8)
    fig.savefig(path or os.path.join(F.out, "fig_model.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_slices(F, path=None):
    """The truth, the MAP point and the variance reduction on the equatorial plane and
    on the meridional plane through 90 E."""
    import matplotlib.pyplot as plt

    truth, m_map = F.load("truth.npy"), F.load("map.npy")
    cols = [("truth", truth, "RdBu", None), ("MAP", m_map, "RdBu", None)]
    if F.has("var_reduction.npy"):
        red = F.load("var_reduction.npy") / np.maximum(F.load("var_prior.npy").mean(), 1e-300)
        cols.append(("variance reduction / prior variance", red, "magma_r", (0.0, None)))
    fig, axs = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 8.0))
    for i, (axis, pl, name) in enumerate(((2, (0, 1), "equatorial plane (x to 0 E, y to 90 E)"),
                                          (0, (1, 2), "meridional plane through 90 E (y to 90 E, z north)"))):
        idx, tri = F.plane(axis)
        xy = F.xyz[idx][:, pl]
        for j, (lab, f, cmap, lim) in enumerate(cols):
            v = f[idx]
            if lim is None:
                top = np.abs(truth[idx]).max()
                vmin, vmax = -top, top
            else:
                vmin, vmax = lim[0], max(v.max(), 1e-12)
            im = disc(axs[i, j], xy, v, tri, vmin, vmax, cmap)
            axs[i, j].set_title("%s\n%s" % (lab, name), fontsize=8)
            fig.colorbar(im, ax=axs[i, j], shrink=0.7, pad=0.02)
    fig.savefig(path or os.path.join(F.out, "fig_slices.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_spectrum(F, path=None):
    import matplotlib.pyplot as plt

    d = F.load("eigenvalues.npy")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogy(np.arange(1, d.size + 1), np.maximum(d, 1e-16), "k.", ms=3)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xlabel("number")
    ax.set_ylabel("eigenvalue")
    ax.set_title("prior-preconditioned misfit Hessian, %d parameters" % F.setup["sizes"]["parameters"], fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(path or os.path.join(F.out, "fig_spectrum.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_variance(F, z, depth=0.0, path=None):
    """Prior variance, posterior variance (= prior - reduction) and the share of the
    prior variance the data remove, on the surface and on the equatorial plane."""
    import matplotlib.pyplot as plt

    pr, red = F.load("var_prior.npy"), F.load("var_reduction.npy")
    pts = ((z["rec"], REC), (z["src"], SRC))
    xyz, vpr, tri = F.at_depth(pr, depth)
    _, vred, _ = F.at_depth(red, depth)
    top = float(np.percentile(vpr, 99.5))
    fig, axs = plt.subplots(3, 3, figsize=(10.0, 10.2))
    for i, (name, lon0, lat0) in enumerate(VIEWS):
        globe(axs[i, 0], xyz, vpr, tri, lon0, lat0, 0.0, top, "viridis", pts)
        axs[i, 0].set_title("prior variance, %s" % name, fontsize=9)
        im1 = globe(axs[i, 1], xyz, np.maximum(vpr - vred, 0.0), tri, lon0, lat0, 0.0, top, "viridis", pts)
        axs[i, 1].set_title("posterior variance, %s" % name, fontsize=9)
        im2 = globe(axs[i, 2], xyz, vred / np.maximum(vpr, 1e-300), tri, lon0, lat0, 0.0, 1.0, "magma_r", pts)
        axs[i, 2].set_title("share of the variance removed, %s" % name, fontsize=9)
    if F.nf % 2 == 0:
        idx, tri2 = F.plane(2)
        xy = F.xyz[idx][:, (0, 1)]
        disc(axs[2, 0], xy, pr[idx], tri2, 0.0, top, "viridis")
        disc(axs[2, 1], xy, np.maximum(pr[idx] - red[idx], 0.0), tri2, 0.0, top, "viridis")
        disc(axs[2, 2], xy, red[idx] / np.maximum(pr[idx], 1e-300), tri2, 0.0, 1.0, "magma_r")
        for j in range(3):
            axs[2, j].set_title("equatorial plane (0 E to the right, 90 E up)", fontsize=8)
    cb = fig.colorbar(im1, ax=axs[:, :2], shrink=0.5, pad=0.02, location="bottom")
    cb.set_label("variance at the surface and on the plane, (km/s)$^2$", fontsize=8)
    cb = fig.colorbar(im2, ax=axs[:, 2], shrink=0.9, pad=0.02, location="bottom")
    cb.set_label("reduction / prior variance", fontsize=8)
    fig.savefig(path or os.path.join(F.out, "fig_variance.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_samples(F, z, depth=67.0, path=None):
    import matplotlib.pyplot as plt

    A, B = F.load("samples_prior.npy"), F.load("samples_posterior.npy")
    ns = min(A.shape[0], 4)
    pts = ((z["rec"], REC),)
    xyz, a, tri = F.at_depth(A[:ns], depth)
    _, b, _ = F.at_depth(B[:ns], depth)
    lim = np.abs(a).max()
    fig, axs = plt.subplots(2, ns, figsize=(3.1 * ns, 6.4))
    axs = axs.reshape(2, ns)
    for j in range(ns):
        globe(axs[0, j], xyz, a[j], tri, 90.0, 35.0, -lim, lim, "RdBu", pts)
        axs[0, j].set_title("prior sample %d" % (j + 1), fontsize=9)
        im = globe(axs[1, j], xyz, b[j], tri, 90.0, 35.0, -lim, lim, "RdBu", pts)
        axs[1, j].set_title("posterior sample %d" % (j + 1), fontsize=9)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.01)
    cb.set_label("wave speed anomaly at %g km, km/s" % depth, fontsize=8)
    fig.savefig(path or os.path.join(F.out, "fig_samples.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_data(F, z, receivers=(5, 45, 95), path=None):
    """Seismograms of the truth at a few receivers, how far PREM's and the MAP point's
    are from them, and the MAP solve's history."""
    import matplotlib.pyplot as plt

    t = z["t"]
    dt = t[1] - t[0]

    def velocity(y):
        # the recorded series is the ground's vertical acceleration; trapezoid rule
        v = np.zeros_like(y)
        v[1:] = dt * np.cumsum(0.5 * (y[1:] + y[:-1]), axis=0)
        return v

    ym = F.load("map_data.npz")["y"] if F.has("map_data.npz") else None
    receivers = [r for r in receivers if r < z["rec"].shape[0]]
    nr = len(receivers)
    fig = plt.figure(figsize=(10.0, 2.0 * nr + 2.6))
    gs = fig.add_gridspec(nr + 1, 2, height_ratios=[1.0] * nr + [1.3])
    vt, vp = velocity(z["y_true"]), velocity(z["y_prem"])
    vm = velocity(ym) if ym is not None else None
    for i, r in enumerate(receivers):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, vt[:, r], "k", lw=1.0)
        ax.set_ylabel("velocity", fontsize=8)
        ax.set_title("truth, receiver at %.1f N, %.1f E" % (z["rec_lat"][r], z["rec_lon"][r]), fontsize=8)
        ax.tick_params(labelsize=7)
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, vp[:, r] - vt[:, r], color="tab:blue", lw=0.9, label="PREM (start) - truth")
        if vm is not None:
            ax.plot(t, vm[:, r] - vt[:, r], color="tab:red", lw=0.9, label="MAP - truth")
        ax.set_title("difference from the truth", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper left")
    for j in range(2):
        fig.axes[2 * (nr - 1) + j].set_xlabel("time, s", fontsize=8)
    ax = fig.add_subplot(gs[nr, :])
    if F.has("map.json"):
        h = json.load(open(os.path.join(F.out, "map.json")))
        sigma = float(z["noise_std"])
        m0 = 0.5 * float(np.sum(((z["d_prem"] - z["d_obs"]) / sigma) ** 2))
        it = [0] + [e["it"] for e in h["iterations"]]
        cg = [0] + [e["cg"] for e in h["iterations"]]
        ax.semilogy(cg, [m0] + [e["misfit"] for e in h["iterations"]], "ko-", ms=3.5, label="misfit")
        ax.semilogy(cg[1:], [max(e["reg"], 1e-30) for e in h["iterations"]], "s--", color="tab:red", ms=3,
                    label="prior term")
        ax.axhline(0.5 * z["d_obs"].size, color="0.6", lw=0.8)
        ax.text(cg[-1], 0.5 * z["d_obs"].size, " half the number of data", fontsize=7, va="bottom", ha="right",
                color="0.4")
        for k, c in zip(it, cg):
            ax.annotate(str(k), (c, ([m0] + [e["misfit"] for e in h["iterations"]])[k]), fontsize=6,
                        textcoords="offset points", xytext=(3, 4))
        ax.set_xlabel("CG iterations (Hessian actions) so far; the points are the Gauss-Newton iterations",
                      fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path or os.path.join(F.out, "fig_data.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def stats(F, z):
    """Numbers of a run that the figures do not give: how close the MAP point is to the
    truth where there are receivers and where there are none, how many eigenvalues are
    above one, and how often the truth lies within two posterior standard deviations."""
    truth = F.load("truth.npy")
    out = dict(parameters=int(truth.size))
    x = F.xyz
    r = np.linalg.norm(x, axis=1)
    lat = np.degrees(np.arcsin(np.clip(x[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    lon = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
    upper = r >= earth.R_EARTH - 670.0
    regions = {
        "under the receivers, top 670 km": upper & (lat >= 0) & (lon >= 0) & (lon <= 180),
        "opposite quarter, top 670 km": upper & (lat <= 0) & (lon <= 0),
        "whole mantle": r >= earth.R_CMB,
    }
    if F.has("map.npy"):
        m = F.load("map.npy")
        out["map"] = {}
        for name, sel in regions.items():
            a, b = m[sel], truth[sel]
            out["map"][name] = dict(
                vertices=int(sel.sum()),
                relative_error=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                correlation=float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else 0.0,
                amplitude_ratio=float(np.linalg.norm(a) / np.linalg.norm(b)))
    if F.has("eigenvalues.npy"):
        d = F.load("eigenvalues.npy")
        out["eigenvalues"] = dict(kept=int(d.size), largest=float(d[0]), smallest=float(d[-1]),
                                  above_one=int((d > 1).sum()))
    if F.has("var_prior.npy") and F.has("map.npy"):
        pr, red = F.load("var_prior.npy"), F.load("var_reduction.npy")
        post = np.maximum(pr - red, 1e-30)
        out["variance"] = {}
        for name, sel in regions.items():
            inside = np.abs(m[sel] - truth[sel]) <= 2.0 * np.sqrt(post[sel])
            out["variance"][name] = dict(
                prior_std=float(np.sqrt(pr[sel].mean())), posterior_std=float(np.sqrt(post[sel].mean())),
                mean_reduction=float(red[sel].sum() / pr[sel].sum()),
                largest_reduction=float((red[sel] / pr[sel]).max()),
                truth_within_two_std=float(inside.mean()))
    if F.has("map.json"):
        h = json.load(open(os.path.join(F.out, "map.json")))
        if h["iterations"]:
            out["newton"] = dict(iterations=len(h["iterations"]), cg=h["iterations"][-1]["cg"],
                                 reduction=h.get("reduction"), seconds=h.get("seconds"))
    with open(os.path.join(F.out, "stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def fig_spectra(outs, path, labels=None):
    """The spectra of several runs on one plot: the same experiment on different meshes
    should give the same eigenvalues."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    marks = ("k.", "r+", "bx", "g.", "m+", "cx")
    for i, out in enumerate(outs):
        d = np.load(os.path.join(out, "eigenvalues.npy"))
        s = json.load(open(os.path.join(out, "setup.json")))["sizes"]
        e = json.load(open(os.path.join(out, "eig.json")))
        lab = labels[i] if labels else "%d parameters, %.4f Hz, %d products%s" % (
            s["parameters"], s["f0"], e["products"], " (double pass)" if e["products"] > d.size + 100 else "")
        ax.semilogy(np.arange(1, d.size + 1), np.maximum(d, 1e-16), marks[i % len(marks)], ms=3.5, label=lab)
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set_xlabel("number")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_scaling(files, path):
    """Weak scaling from the records ``scale.py`` writes: time per step of the three
    sweeps against the number of ranks, each rank holding about as many unknowns."""
    import matplotlib.pyplot as plt

    recs = sorted((json.load(open(f)) for f in files), key=lambda r: r["ranks"])
    r = np.array([x["ranks"] for x in recs])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key, lab, st in (("forward", "forward, 1 field", "ko-"), ("incremental", "forward with its sensitivity, 2 fields", "s--"),
                         ("adjoint", "backward with the adjoint, 2 fields", "^:")):
        y = [x["sweeps"][key]["ms_per_step"] for x in recs if key in x["sweeps"]]
        if len(y) == r.size:
            axs[0].plot(r, y, st, ms=4, label=lab)
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(r)
    axs[0].set_xticklabels([str(v) for v in r])
    axs[0].set_ylim(bottom=0)
    axs[0].set_xlabel("ranks (one card or slice each)")
    axs[0].set_ylabel("ms per time step")
    for x in recs:
        axs[0].annotate("%.2g G" % (x["wave_dofs"] / 1e9) if x["wave_dofs"] >= 1e9 else "%d M" % round(x["wave_dofs"] / 1e6),
                        (x["ranks"], x["sweeps"]["forward"]["ms_per_step"]), fontsize=7,
                        textcoords="offset points", xytext=(0, -12), ha="center")
    axs[0].legend(fontsize=7)
    axs[0].grid(True, alpha=0.25)
    axs[0].set_title("unknowns under each point", fontsize=8)
    axs[1].loglog(r, [x["sweeps"]["forward"]["dof_updates_per_s"] / 1e9 for x in recs], "ko-", ms=4, label="measured")
    y0 = recs[0]["sweeps"]["forward"]["dof_updates_per_s"] / 1e9 / recs[0]["ranks"]
    axs[1].loglog(r, y0 * r, "-", color="0.6", lw=0.8, label="ideal")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks(r)
    axs[1].set_xticklabels([str(v) for v in r])
    axs[1].set_xlabel("ranks")
    axs[1].set_ylabel("billion unknown-updates per second")
    axs[1].legend(fontsize=7)
    axs[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 1
    if argv[0] in ("--spectra", "--scaling"):
        files = [a for a in argv[1:] if a != "-o"]
        out = files.pop(argv[1:].index("-o")) if "-o" in argv else "figure.png"
        (fig_spectra if argv[0] == "--spectra" else fig_scaling)(files, out)
        print("wrote", out)
        return 0
    F = Fields(argv[0])
    z = F.load("data.npz")
    made = []
    if F.has("map.npy"):
        fig_model(F, z)
        fig_data(F, z)
        made += ["fig_model", "fig_data"]
        if F.nf % 2 == 0:
            fig_slices(F)
            made.append("fig_slices")
        else:
            print("no fig_slices: a plane through the centre needs an even number of cells along an edge")
    if F.has("eigenvalues.npy"):
        fig_spectrum(F)
        made.append("fig_spectrum")
    if F.has("var_prior.npy"):
        fig_variance(F, z)
        made.append("fig_variance")
    if F.has("samples_prior.npy"):
        fig_samples(F, z)
        made.append("fig_samples")
    st = stats(F, z)
    print("wrote %s and stats.json in %s" % (", ".join(made), F.out))
    for name, v in st.get("map", {}).items():
        print("  MAP %-34s relative error %.3f, correlation %.3f, amplitude ratio %.2f"
              % (name + ":", v["relative_error"], v["correlation"], v["amplitude_ratio"]))
    for name, v in st.get("variance", {}).items():
        print("  variance %-29s prior std %.3f, posterior std %.3f, mean reduction %.1f %%, truth within 2 std at %.1f %%"
              % (name + ":", v["prior_std"], v["posterior_std"], 100 * v["mean_reduction"], 100 * v["truth_within_two_std"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

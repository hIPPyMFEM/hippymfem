#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Figures from a ``run.py --dump`` file and its JSON record.

    python -m applications.geothermal.figures results/geothermal_n64_r4.npz \\
        --json results/geothermal_n64_r4.json --out results/figures/geothermal_n64

Writes ``<out>_slices.png`` (truth, MAP, posterior std on a horizontal slice through the
anomaly and a vertical section through it, with the boreholes), ``<out>_spectrum.png``
(the generalized eigenvalues), ``<out>_qoi.png`` (the sampled QoI against the linearized
Gaussian), ``<out>_profile.png`` (prior and posterior std and the MAP error along the
vertical line through the anomaly) and ``<out>_block.png`` (truth, MAP and posterior std
on the block of rock with a quarter cut away through the anomaly, and the boreholes: the
picture of the application's README).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from applications.geothermal.model import ANOMALY, H_DEPTH, L_HORIZ   # noqa: E402


def on_grid(xyz, vals, n):
    """A P1 field on the Cartesian ``n^3`` mesh as an ``(n+1, n+1, n+1)`` array [i, j, k]."""
    idx = np.rint(xyz * n).astype(int)
    out = np.full((n + 1,) * 3, np.nan)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
    return out


def block(ax, F, n, norm, cmap, ic, jc, km, kmz, zexag):
    """Draw the field ``F[i, j, k]`` (``k = 0`` at the base) on the block with the front
    right quarter cut away down to the base, the cut faces passing through ``(ic, jc)``.
    Coordinates in km, z up.  A mesh finer than 128^3 is drawn at every other vertex or so."""
    step = max(1, int(np.ceil(n / 128.0)))
    span = lambda a, b: np.unique(np.append(np.arange(a, b + 1, step), b))
    xs = np.linspace(0.0, km, n + 1)
    ys = np.linspace(0.0, km, n + 1)
    zs = np.linspace(-kmz, 0.0, n + 1) * zexag

    def face(X, Y, Z, V):
        ax.plot_surface(X, Y, Z, facecolors=cmap(norm(V)), rstride=1, cstride=1, shade=False,
                        linewidth=0, antialiased=False)

    # outer faces seen from the viewer's side (+x and -y), then the two cut faces, then the top
    I, K = np.meshgrid(span(0, ic), span(0, n), indexing="ij")
    face(xs[I], np.zeros_like(xs[I]), zs[K], F[I, 0, K])                       # front,   y = 0,  x < xc
    J, K = np.meshgrid(span(jc, n), span(0, n), indexing="ij")
    face(np.full(J.shape, km), ys[J], zs[K], F[n, J, K])                       # right,   x = L,  y > yc
    J, K = np.meshgrid(span(0, jc), span(0, n), indexing="ij")
    face(np.full(J.shape, xs[ic]), ys[J], zs[K], F[ic, J, K])                  # cut,     x = xc, y < yc
    I, K = np.meshgrid(span(ic, n), span(0, n), indexing="ij")
    face(xs[I], np.full(I.shape, ys[jc]), zs[K], F[I, jc, K])                  # cut,     y = yc, x > xc
    I, J = np.meshgrid(span(0, ic), span(0, n), indexing="ij")
    face(xs[I], ys[J], np.zeros(I.shape), F[I, J, n])                          # top,     x < xc
    I, J = np.meshgrid(span(ic, n), span(jc, n), indexing="ij")
    face(xs[I], ys[J], np.zeros(I.shape), F[I, J, n])                          # top,     x > xc, y > yc
    # the block's visible edges
    xc, yc, zb = xs[ic], ys[jc], -kmz * zexag
    edges = [((0, 0, 0), (xc, 0, 0)), ((xc, 0, 0), (xc, yc, 0)), ((xc, yc, 0), (km, yc, 0)), ((km, yc, 0), (km, km, 0)),
             ((0, 0, 0), (0, km, 0)), ((0, km, 0), (km, km, 0)),
             ((0, 0, 0), (0, 0, zb)), ((0, 0, zb), (xc, 0, zb)), ((xc, 0, 0), (xc, 0, zb)), ((xc, 0, zb), (xc, yc, zb)),
             ((xc, yc, 0), (xc, yc, zb)), ((xc, yc, zb), (km, yc, zb)), ((km, yc, 0), (km, yc, zb)),
             ((km, yc, zb), (km, km, zb)), ((km, km, 0), (km, km, zb))]
    for a, b in edges:
        ax.plot(*zip(a, b), color="#333333", lw=1.2, zorder=10)
    return xc, yc


def block_figure(fields, targets, n, out, vmax, std_max, zexag=1.3):
    """Truth, MAP and posterior std on the cut block, with the boreholes."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import proj3d

    c = ANOMALY["centre"]
    ic, jc = int(round(c[0] * n)), int(round(c[1] * n))
    km, kmz = L_HORIZ / 1000.0, H_DEPTH / 1000.0
    wells = np.unique(np.round(targets[:, :2], 9), axis=0) * km
    depth = (1.0 - targets[:, 2].min()) * kmz
    logk = colors.Normalize(-vmax, vmax)             # truth and MAP on one scale, so they compare
    panels = (("mtrue", "RdBu_r", logk, "true parameter"), ("mmap", "RdBu_r", logk, "MAP estimate"),
              ("std_post", "viridis", colors.Normalize(0.0, std_max), "posterior std"))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Lato", "DejaVu Sans"]}):
        fig = plt.figure(figsize=(15.0, 5.6))
        axes = []
        for p, (key, cmap_name, norm, title) in enumerate(panels):
            ax = fig.add_axes([-0.075 + 0.333 * p, 0.145, 0.48, 0.80], projection="3d")
            axes.append(ax)
            ax.set_proj_type("ortho")
            ax.computed_zorder = False
            xc, yc = block(ax, fields[key], n, norm, plt.get_cmap(cmap_name), ic, jc, km, kmz, zexag)
            # the wellheads in every panel; under the posterior std, which they explain, every
            # borehole straight down to its deepest log, drawn through the rock
            if key == "std_post":
                for wx, wy in wells:
                    ax.plot([wx, wx], [wy, wy], [0.0, -depth * zexag], color="#111111", lw=1.0, alpha=0.55, zorder=13)
            ax.scatter(wells[:, 0], wells[:, 1], np.zeros(len(wells)), s=13, color="#111111", depthshade=False, zorder=14)
            ax.view_init(elev=24, azim=-52)
            ax.set_box_aspect((km, km, kmz * zexag), zoom=1.22)
            ax.set_axis_off()
            fig.text(0.165 + 0.333 * p, 0.985, title, fontsize=23, color="#0d294d", fontweight="bold", ha="center", va="top")
        # the colour bars just under the lowest corner of the blocks, wherever the view puts it
        fig.canvas.draw()
        zb = -kmz * zexag
        base = np.array([(0.0, 0.0, zb), (xc, 0.0, zb), (xc, yc, zb), (km, yc, zb), (km, km, zb)])
        low = 1.0
        for ax in axes:
            X, Y, _ = proj3d.proj_transform(base[:, 0], base[:, 1], base[:, 2], ax.M)
            low = min(low, fig.transFigure.inverted().transform(ax.transData.transform(np.column_stack([X, Y])))[:, 1].min())
        for box, norm, cmap_name, label in (([0.07, low - 0.095, 0.52, 0.04], logk, "RdBu_r", "log conductivity"),
                                            ([0.715, low - 0.095, 0.25, 0.04], panels[2][2], "viridis",
                                             "posterior std (%d boreholes to %.1f km)" % (len(wells), depth))):
            cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name)), cax=fig.add_axes(box),
                              orientation="horizontal")
            cb.set_label(label, fontsize=17)
            cb.ax.tick_params(labelsize=15)
        fig.savefig(out + "_block.png", dpi=200, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vmax", type=float, default=1.8, help="the block's colours span -vmax to vmax in log conductivity")
    ap.add_argument("--std-max", type=float, default=0.7, help="... and 0 to this in its posterior std")
    ap.add_argument("--fields", default=None,
                    help="the file of movie_data.py fields: the block's posterior std from it, which has the noise "
                         "of 800 Monte Carlo samples where the dump's has that of 64")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.dump)
    n = int(z["n"])
    rec = json.load(open(args.json)) if args.json else {}
    out = args.out or os.path.splitext(args.dump)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fields = {k: on_grid(z["xyz"], z[k], n) for k in ("mtrue", "mmap", "std_post", "std_prior")}
    targets = z["targets"]
    c = ANOMALY["centre"]
    ic = [int(round(v * n)) for v in c]
    km = L_HORIZ / 1000.0
    kmz = H_DEPTH / 1000.0

    # ---- slices
    vmax = float(np.nanmax(np.abs(fields["mtrue"])))
    smax = float(np.nanmax(fields["std_prior"]))
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.4), constrained_layout=True)
    titles = ["log k / k_ref, truth", "log k / k_ref, MAP", "posterior std of log k"]
    for j, (key, vm, cmap) in enumerate((("mtrue", vmax, "RdBu_r"), ("mmap", vmax, "RdBu_r"), ("std_post", smax, "viridis"))):
        F = fields[key]
        # horizontal slice at the anomaly's depth: F[:, :, k] -> x horizontal, y vertical
        im = ax[0, j].imshow(F[:, :, ic[2]].T, origin="lower", extent=(0, km, 0, km), cmap=cmap,
                             vmin=(-vm if key != "std_post" else 0.0), vmax=vm, aspect="equal")
        ax[0, j].plot(targets[:, 0] * km, targets[:, 1] * km, "k.", ms=2.5)
        ax[0, j].add_patch(plt.Circle((c[0] * km, c[1] * km), ANOMALY["radius"] * km, fill=False, ec="k", lw=0.8, ls="--"))
        ax[0, j].set_title(titles[j] + ", z = %.1f km" % ((1 - c[2]) * kmz), fontsize=10)
        ax[0, j].set_xlabel("x [km]")
        ax[0, j].set_ylabel("y [km]")
        fig.colorbar(im, ax=ax[0, j], shrink=0.85)
        # vertical section at y = y_c: F[:, j, :] -> x horizontal, depth vertical
        im = ax[1, j].imshow(F[:, ic[1], :].T, origin="lower", extent=(0, km, -kmz, 0), cmap=cmap,
                             vmin=(-vm if key != "std_post" else 0.0), vmax=vm, aspect="auto")
        near = np.abs(targets[:, 1] - c[1]) < 0.06
        ax[1, j].plot(targets[near, 0] * km, -(1 - targets[near, 2]) * kmz, "k.", ms=2.0)
        ax[1, j].set_title(titles[j] + ", y = %.1f km" % (c[1] * km), fontsize=10)
        ax[1, j].set_xlabel("x [km]")
        ax[1, j].set_ylabel("depth [km]")
        fig.colorbar(im, ax=ax[1, j], shrink=0.85)
    fig.suptitle("geothermal %d^3: %s" % (n, "MAP rel. error %.2f, coverage %.0f%%" % (rec.get("err_m", np.nan), 100 * rec.get("coverage_2std", np.nan))
                                          if rec else ""), fontsize=11)
    fig.savefig(out + "_slices.png", dpi=150)
    plt.close(fig)

    # ---- spectrum
    d = np.asarray(z["d"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    ax.semilogy(np.arange(1, d.size + 1), d, "o-", ms=3)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("index")
    ax.set_ylabel("generalized eigenvalue of the misfit Hessian")
    ax.set_title("%d^3: %d of %d above 1" % (n, int((d > 1).sum()), d.size), fontsize=10)
    fig.savefig(out + "_spectrum.png", dpi=150)
    plt.close(fig)

    # ---- QoI
    q = np.asarray(z["qoi"])
    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    if q.size > 1:
        ax.hist(q, bins=max(8, min(30, q.size // 4)), density=True, alpha=0.6, label="%d posterior samples through the forward solve" % q.size)
    qm, qs, qt = float(z["qoi_map_K"]), float(z["qoi_std_lin_K"]), float(z["qoi_true_K"])
    if qs > 0:
        xs = np.linspace(qm - 4 * qs, qm + 4 * qs, 200)
        ax.plot(xs, np.exp(-0.5 * ((xs - qm) / qs) ** 2) / (qs * np.sqrt(2 * np.pi)), "k-", label="linearized: %.2f +- %.2f K" % (qm, qs))
    if "qoi_taylor2_mean_K" in z.files and np.isfinite(float(z["qoi_taylor2_mean_K"])):
        q2, s2 = float(z["qoi_taylor2_mean_K"]), float(z["qoi_taylor2_std_K"])
        xs = np.linspace(q2 - 4 * s2, q2 + 4 * s2, 200)
        ax.plot(xs, np.exp(-0.5 * ((xs - q2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi)), "k--", label="second-order mean %.2f K, linearized width" % q2)
    ax.axvline(qt, color="r", ls="--", label="truth %.2f K" % qt)
    ax.set_xlabel("mean temperature above the surface in the target volume [K]")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.savefig(out + "_qoi.png", dpi=150)
    plt.close(fig)

    # ---- vertical profile through the anomaly
    depth = -(1 - np.arange(n + 1) / n) * kmz
    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    ax.plot(fields["std_prior"][ic[0], ic[1], :], depth, label="prior std")
    ax.plot(fields["std_post"][ic[0], ic[1], :], depth, label="posterior std")
    ax.plot(np.abs(fields["mtrue"][ic[0], ic[1], :] - fields["mmap"][ic[0], ic[1], :]), depth, label="|truth - MAP|")
    zb = -(1 - targets[:, 2].min()) * kmz
    ax.axhline(zb, color="k", lw=0.8, ls=":", label="bottom of the boreholes")
    ax.axhline(-(1 - c[2]) * kmz, color="r", lw=0.8, ls="--", label="anomaly centre")
    ax.set_xlabel("log k")
    ax.set_ylabel("depth [km]")
    ax.legend(fontsize=7)
    ax.set_title("vertical line through the anomaly, %d^3" % n, fontsize=10)
    fig.savefig(out + "_profile.png", dpi=150)
    plt.close(fig)

    if args.fields:
        fields["std_post"] = np.load(args.fields)["std_post"].reshape((n + 1,) * 3).T       # [i, j, k] from x fastest
    block_figure(fields, targets, n, out, args.vmax, args.std_max)
    print("wrote %s_{slices,spectrum,qoi,profile,block}.png" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

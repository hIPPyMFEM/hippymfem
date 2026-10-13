#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Figures of a ``run.py`` output directory (no MFEM needed): the friction (true, MAP),
the prior and posterior standard deviations, prior and posterior samples, the observed
and modelled surface speed, the spectrum, and the predictions as the paper's Figure 10
(the gradients of the three mass fluxes and their inference-for-prediction directions).

    python figures.py out/syn40
    python figures.py out/obs80 out/obs40 out/obs20 --spectra spectra.png
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def on_grid(grid, arr):
    """A saved footprint field (columns x, y, value) as an image on the grid nodes (of a
    quadratic friction, which has dofs between the nodes too, its values at the nodes)."""
    img = np.full(grid.s.shape, np.nan)
    fj, fi = (arr[:, 1] - grid.yn[0]) / grid.h, (arr[:, 0] - grid.xn[0]) / grid.h
    J, I = np.rint(fj).astype(int), np.rint(fi).astype(int)
    at = (np.abs(fj - J) < 1e-6) & (np.abs(fi - I) < 1e-6)
    img[J[at], I[at]] = arr[at, 2]
    return img


def points_on_grid(grid, targets, vals):
    """Values at data points as an image on the grid nodes (the points of an integral
    misfit between the nodes, the Q2 mid nodes, are left out)."""
    img = np.full(grid.s.shape, np.nan)
    fj, fi = (targets[:, 1] - grid.yn[0]) / grid.h, (targets[:, 0] - grid.xn[0]) / grid.h
    J, I = np.rint(fj).astype(int), np.rint(fi).astype(int)
    at = (np.abs(fj - J) < 0.26) & (np.abs(fi - I) < 0.26)         # a point by a node (nudged inside)
    img[J[at], I[at]] = np.asarray(vals)[at]
    return img


def spectra(outs, target):
    """The eigenvalues of several runs on one plot, with those of the paper's Figure 5
    (``paper/isaac2015_fig5.csv``, read off the figure's vector graphics)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c, paper_c = "#52514e", "#898781", "#e1e0d9", "#eb6834"
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]            # coarse to fine
    runs = []
    for o in outs:
        e = json.load(open(os.path.join(o, "eig.json")))
        sz = json.load(open(os.path.join(o, "sizes.json")))
        d = np.asarray(e["d"])
        runs.append((sz["grid"]["h_km"], sz["parameters"], d, int((d > 1).sum())))
    runs.sort(key=lambda r: -r[0])
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", color=grid_c, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=muted, which="both", length=3)
    ax.axhline(1.0, color=muted, lw=0.8)
    for (h, n, d, above), c in zip(runs, ramp[-len(runs):]):
        keep = d > 1e-2
        ax.plot(np.arange(1, len(d) + 1)[keep], d[keep], color=c, lw=2.0,
                label="%g km mesh, %s parameters: %s" % (h, format(n, ","), format(above, ",")))
    ref = os.path.join(HERE, "paper", "isaac2015_fig5.csv")
    if os.path.exists(ref):
        r = np.loadtxt(ref, delimiter=",")
        ax.plot(r[:, 0], r[:, 1], color=paper_c, lw=2.0,
                label="Isaac et al. (2015), Fig. 5,\n409,545 parameters: about 3,800")
    ax.set_ylim(bottom=1e-2)
    ax.set_xlabel("index of the eigenvalue", color=ink)
    ax.set_ylabel("eigenvalue of the prior-preconditioned misfit Hessian", color=ink)
    # the corner above the curves; each entry ends with its count of eigenvalues above 1
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, title="eigenvalues above 1", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(target, dpi=140)
    plt.close(fig)
    print("wrote", target)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import antarctica as an

    p = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))
    p.add_argument("out", nargs="+")
    p.add_argument("--spectra", help="plot the spectra of all the runs given into this file instead")
    a = p.parse_args(argv)
    if a.spectra:
        return spectra(a.out, a.spectra)
    if len(a.out) != 1:
        p.error("one run at a time (or several with --spectra)")
    a.out = a.out[0]
    g = an.ModelGrid.load(json.load(open(os.path.join(a.out, "sizes.json")))["grid"]["r"])
    path = lambda f: os.path.join(a.out, f)
    ext = [g.xn[0], g.xn[-1], g.yn[0], g.yn[-1]]

    def show(ax, img, title, **kw):
        im = ax.imshow(img, origin="lower", extent=ext, interpolation="nearest", **kw)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        return im

    z = np.load(path("data.npz"))
    mmap = on_grid(g, np.load(path("map.npy")))
    figs = []
    # friction
    have_truth = "mtrue" in z.files
    n = 2 + have_truth
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 5))
    k = 0
    if have_truth:
        truth = on_grid(g, np.column_stack((z["mtrue_xy"], z["mtrue"])))
        im = show(axs[k], truth / np.log(10), "true log10 friction", cmap="RdBu", vmin=0, vmax=3); k += 1
    im = show(axs[k], mmap / np.log(10), "MAP log10 friction", cmap="RdBu", vmin=0, vmax=3); k += 1
    fig.colorbar(im, ax=axs[:k], shrink=0.75, label="log10 exp(beta), MPa a / km")
    if os.path.exists(path("var_post.npy")):
        sd = on_grid(g, np.load(path("var_post.npy")))
        if have_truth:
            ok = np.isfinite(truth) & np.isfinite(mmap) & np.isfinite(sd)
            print("the MAP point against the truth: correlation %.3f; the truth within two posterior standard "
                  "deviations at %.1f %% of the nodes"
                  % (np.corrcoef(truth[ok], mmap[ok])[0, 1],
                     100.0 * np.mean(np.abs(mmap[ok] - truth[ok]) <= 2.0 * np.sqrt(np.maximum(sd[ok], 0)))))
        im = show(axs[k], np.sqrt(np.maximum(sd, 0)), "posterior std of beta", cmap="viridis", vmin=0,
                  vmax=float(np.nanpercentile(np.sqrt(np.maximum(on_grid(g, np.load(path("var_prior.npy"))), 0)), 95)))
        fig.colorbar(im, ax=axs[k], shrink=0.75)
    fig.savefig(path("fig_friction.png"), dpi=110, bbox_inches="tight")
    figs.append("fig_friction.png")
    plt.close(fig)
    # observed and modelled speed
    if os.path.exists(path("map_velocity.npy")):
        d = z["d"]
        pr = np.load(path("map_velocity.npy"))
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        norm = LogNorm(vmin=1.0, vmax=3000.0)
        im = show(axs[0], points_on_grid(g, z["targets"], 1e3 * np.hypot(d[:, 0], d[:, 1])), "observed speed (m/a)",
                  cmap="magma", norm=norm)
        show(axs[1], points_on_grid(g, z["targets"], 1e3 * np.hypot(pr[:, 0], pr[:, 1])), "modelled at the MAP point (m/a)",
             cmap="magma", norm=norm)
        fig.colorbar(im, ax=axs, shrink=0.75)
        fig.savefig(path("fig_velocity.png"), dpi=110, bbox_inches="tight")
        figs.append("fig_velocity.png")
        plt.close(fig)
    # spectrum
    if os.path.exists(path("eig.json")):
        e = json.load(open(path("eig.json")))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(np.arange(1, len(e["d"]) + 1), np.maximum(e["d"], 1e-6))
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("index"); ax.set_ylabel("eigenvalue")
        ax.set_title("prior-preconditioned misfit Hessian at the MAP point")
        fig.savefig(path("fig_spectrum.png"), dpi=110, bbox_inches="tight")
        figs.append("fig_spectrum.png")
        plt.close(fig)
    # samples
    sp = sorted(f for f in os.listdir(a.out) if f.startswith("sample_prior_"))
    if sp:
        m = min(3, len(sp))
        fig, axs = plt.subplots(2, m, figsize=(4.2 * m, 8))
        axs = np.atleast_2d(axs)
        for j in range(m):
            show(axs[0, j], on_grid(g, np.load(path("sample_prior_%d.npy" % j))) / np.log(10), "prior sample", cmap="RdBu",
                 vmin=0, vmax=4)
            show(axs[1, j], on_grid(g, np.load(path("sample_post_%d.npy" % j))) / np.log(10), "posterior sample", cmap="RdBu",
                 vmin=0, vmax=4)
        fig.savefig(path("fig_samples.png"), dpi=100, bbox_inches="tight")
        figs.append("fig_samples.png")
        plt.close(fig)
    # the predictions, as the paper's Figure 10: the gradient of each flux in beta (top,
    # log10 of its magnitude) and its inference-for-prediction direction (bottom)
    regions = [r for r in ("all", "east", "totten") if os.path.exists(path("qoi_gradient_%s.npy" % r))]
    if regions:
        titles = dict(all="all of Antarctica", east="East Antarctica", totten="Totten Glacier")
        q = json.load(open(path("summary.json"))).get("qoi", {}).get("regions", {}) if os.path.exists(path("summary.json")) else {}
        fig, axs = plt.subplots(2, len(regions), figsize=(4.6 * len(regions), 8.4), squeeze=False)
        for j, r in enumerate(regions):
            gq = on_grid(g, np.load(path("qoi_gradient_%s.npy" % r)))
            top = np.log10(np.maximum(np.abs(gq), 1e-300))
            vmax = float(np.nanmax(top))
            t = titles[r]
            if r in q:
                t += "\n%.1f +- %.2f Gt/a" % (q[r]["flux_map"], q[r]["flux_std_post"])
            im = show(axs[0, j], top, t, cmap="YlOrBr", vmin=vmax - 6, vmax=vmax)
            fig.colorbar(im, ax=axs[0, j], shrink=0.7, label="log10 |dQ/dbeta|")
            w = on_grid(g, np.load(path("qoi_ifp_%s.npy" % r)))
            v = float(np.nanpercentile(np.abs(w), 99.5))
            im = show(axs[1, j], w, "inference-for-prediction direction", cmap="PuOr", vmin=-v, vmax=v)
            fig.colorbar(im, ax=axs[1, j], shrink=0.7)
        fig.savefig(path("fig_prediction.png"), dpi=100, bbox_inches="tight")
        figs.append("fig_prediction.png")
        plt.close(fig)
    # one row for the README: the data, the truth (when there is one), the MAP point and
    # the posterior std
    if os.path.exists(path("var_post.npy")):
        from matplotlib.colors import LogNorm as _LogNorm
        cols = 3 + have_truth
        fig, axs = plt.subplots(1, cols, figsize=(4.3 * cols, 4.6), gridspec_kw=dict(wspace=0.06))
        d = z["d"]
        im = show(axs[0], points_on_grid(g, z["targets"], 1e3 * np.hypot(d[:, 0], d[:, 1])),
                  "surface speed, the data", cmap="magma", norm=_LogNorm(vmin=1.0, vmax=3000.0))
        fig.colorbar(im, ax=axs[0], shrink=0.8, pad=0.01, label="m/a")
        ref = truth if have_truth else mmap
        lo, hi = np.nanpercentile(ref / np.log(10), [1, 99])
        k = 1
        if have_truth:
            show(axs[k], truth / np.log(10), "true friction", cmap="RdBu", vmin=lo, vmax=hi); k += 1
        im = show(axs[k], mmap / np.log(10), "MAP estimate", cmap="RdBu", vmin=lo, vmax=hi)
        fig.colorbar(im, ax=axs[1:k + 1], shrink=0.8, pad=0.01, label="log10 friction (MPa a/km)")
        sd = np.sqrt(np.maximum(on_grid(g, np.load(path("var_post.npy"))), 0))
        sp = np.sqrt(np.maximum(on_grid(g, np.load(path("var_prior.npy"))), 0))
        im = show(axs[k + 1], sd, "posterior std (prior %.1f)" % np.nanmean(sp), cmap="viridis", vmin=0,
                  vmax=float(np.nanpercentile(sp, 95)))
        fig.colorbar(im, ax=axs[k + 1], shrink=0.8, pad=0.01, label="std of log friction")
        fig.savefig(path("fig_overview.png"), dpi=100, bbox_inches="tight")
        figs.append("fig_overview.png")
        plt.close(fig)
    print("wrote", ", ".join(figs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

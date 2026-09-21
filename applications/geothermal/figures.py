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
Gaussian) and ``<out>_profile.png`` (prior and posterior std and the MAP error along the
vertical line through the anomaly).
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
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
    print("wrote %s_{slices,spectrum,qoi,profile}.png" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

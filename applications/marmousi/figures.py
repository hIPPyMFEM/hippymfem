# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Figures of a run of ``run.py`` and of the records of ``scale.py``, from what they saved.

    python figures.py out/small                      every figure the folder allows, into it
    python figures.py out/small -o pictures          into another folder
    python figures.py out/small --tag lbfgs-prior    the MAP point, samples and chains of that solve
    python figures.py --scaling records/ w64.json -o scaling.png

NumPy and Matplotlib are all it needs (no MFEM, no JAX, none of the application's other
modules): a folder copied to a laptop is enough, without its ``data``.  What a folder does
not hold is left out, with a line that says why.

``fig_overview.png``
    The true speed, the initial model and the MAP point, and how far the last two are from
    the first, on the section in the middle of the box (the paper's, ``y = 0``) and on a map
    at one depth; the window's half level; the sources and the receivers.
``fig_psf.png`` (``fig_psf_<key>.png`` for the other sample lattices of the folder)
    What the Hessian approximation is made of: the Hessian's response to the comb of unit
    offsets at the sample points; that response around a few of them, the impulse responses
    cut out of it, and the operator's own response to the same comb; and the operator
    against the Hessian on a check vector.
``fig_convergence.png``
    The L-BFGS solves of the folder side by side: gradient, cost and model error against
    forward solves and against wall time.
``fig_posterior.png``, ``fig_chain.png``
    The pointwise standard deviations of prior and posterior, and the traces of the chains.
``--scaling``
    Weak scaling from the records of ``scale.py`` (and of the two scripts that the
    propagator and the operator were first measured with): a time step of the sweeps, a
    product with the operator and an iteration of the solve with it against the number of
    GPUs, and each as a fraction of the fewest GPUs' time.

A lattice array is ``[k, j, i]``: depth, across the section, along it; ``survey.npz`` has the
nodes' coordinates in m.  The speed of an iterate ``m`` is ``initial + window * m``.  A speed,
an offset or a standard deviation is a function on the lattice and is drawn as its bilinear
interpolant; a product with the Hessian is a dual vector, a number a node, and is drawn as
a cell a node.  A magnitude takes one hue from light to dark, a signed quantity two hues
about a neutral grey whose arms darken alike: a grey print keeps the order of the one and
the size, not the sign, of the other.  Curves differ by their marks as well as their hues.

The operator of ``fig_psf.png`` is applied here, in NumPy, from the kernels on file:
``H~ = W S S^T W`` with ``S u = sum_k w_k ((g_k + eta G) * u)`` by transforms of the whole
lattice, the formula of ``psf.GlobalPC``.  The folder has the Hessian's products with the
check vectors but not the vectors: the white ones are drawn again from the library's
random stream (Philox, a word a node at its global number, seed ``--seed + 100``), and the
comparison is drawn only if the error of ``H~`` on the vector comes out as the stage
recorded it.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import textwrap
import types

import numpy as np

KM = 1e-3
DPI = 300
#: surface, inks, axis and grid: the tones of ``marmousi.figures``
SURFACE, INK, INK2, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
#: the hues of lines and points, in a fixed order: the first three stay apart for readers who
#: do not tell red from green
HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
#: a magnitude: one hue whose lightness falls in equal steps
RAMP = ("#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#06203f")
#: a signed quantity: blue below zero, a neutral grey at zero, red above; a step of the red arm
#: has the lightness of the same step of the blue one
POLES = ("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#f0efec",
         "#ffd4d0", "#f8aaa3", "#ef7e76", "#e34948", "#bc2f32", "#921d21", "#681013")
#: the marks of jobs and the dashes of series, with the dashes' names
MARKS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*")
DASHES = ("-", (0, (5, 2)), (0, (1.2, 1.6)), (0, (6, 2, 1.2, 2)), (0, (10, 3)),
          (0, (6, 2, 1.2, 2, 1.2, 2)), (0, (3, 1.2)), (0, (10, 2, 3, 2)))
DASH_NAMES = ("solid", "dashed", "dotted", "dash-dotted", "long-dashed", "dash-dot-dotted", "short-dashed",
              "long-and-short-dashed")
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "legend.fontsize": 7.5, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "mathtext.default": "regular",
}
#: the operator's name in a figure (no blank inside: headings are broken at blanks)
H_TILDE = r"$\tilde{H}$"
#: what a history's ``precondition`` is called, its hue and its mark, in the order of the
#: paper's Figure 14
PRECONDITIONERS = {"none": ("no preconditioner", HUES[0], "o"),
                   "prior": ("the prior's covariance", HUES[1], "s"),
                   "psf": (r"$(\tilde{H}+R)^{-1}$ of the point spread functions", HUES[2], "^")}
#: the points that the operator's products of ``fig_psf.png`` may transform in all: beyond it
#: they are left out (some two minutes of one core)
WORK = 4e9


# ------------------------------------------------------------------------------- the folder
class Folder:
    """The files of a run of ``run.py``: the set-up, the lattice with its three fields, the
    survey, and the names of what the stages left."""

    def __init__(self, out):
        self.out = os.path.abspath(out)
        for name in ("setup.json", "survey.npz", "truth.npy", "initial.npy", "window.npy"):
            if not self.has(name):
                raise SystemExit("figures: %s holds no %s: not the folder of a run whose stage data has "
                                 "ended" % (self.out, name))
        setup = self.json("setup.json")
        self.args, self.sizes = setup["args"], setup["sizes"]
        self.data = self.json("data.json") if self.has("data.json") else {}
        s = self.load("survey.npz")
        self.x, self.y, self.z = (np.asarray(s[c], dtype=float) for c in "xyz")
        self.sources, self.receivers = np.asarray(s["sources"]), np.asarray(s["receivers"])
        self.truth, self.initial, self.window = (self.load(n) for n in ("truth.npy", "initial.npy",
                                                                       "window.npy"))
        self.shape = self.truth.shape
        if self.shape != (self.z.size, self.y.size, self.x.size):
            raise SystemExit("figures: the fields of %s are %s and its nodes %d x %d x %d along z, y, x"
                             % (self.out, self.shape, self.z.size, self.y.size, self.x.size))
        #: whether the data are those of the true speed inside the window alone (the driver's
        #: default; ``data.json`` says which), and the speed they are of
        self.windowed = self.data.get("truth", "inside the window") != "everywhere"
        self.medium = self.truth
        if self.windowed:
            self.medium = self.initial + self.window * (self.truth - self.initial)
        #: the section in the middle of the box: the paper's, ``y = 0``
        self.jmid = int(np.argmin(np.abs(self.y - 0.5 * (self.y[0] + self.y[-1]))))
        self.h = float(self.x[1] - self.x[0])

    def path(self, name):
        return os.path.join(self.out, name)

    def has(self, name):
        return os.path.exists(self.path(name))

    def load(self, name):
        return np.load(self.path(name), allow_pickle=False)

    def json(self, name):
        with open(self.path(name)) as f:
            return json.load(f)

    # ---- what the stages left
    def solves(self):
        """``{tag: history}`` of the MAP solves whose iterate is on file."""
        out = {}
        for f in sorted(glob.glob(self.path("map_*.json"))):
            tag = os.path.basename(f)[4:-5]
            if self.has("map_%s.npy" % tag):
                hist = self.json("map_%s.json" % tag)
                if hist.get("iterations"):
                    out[tag] = hist
        return out

    def solve(self, tag=None):
        """``(tag, history)`` of the solve asked for, by its tag or by its preconditioner; with
        none asked the one with the point spread functions, then the prior's, then any, and
        of several the L-BFGS one.  None if the folder has no such solve."""
        solves = self.solves()
        if tag in solves:
            return tag, solves[tag]
        for pre in ("psf", "prior", "none") if tag is None else (tag,):
            found = sorted((h.get("solver") != "lbfgs", t) for t, h in solves.items()
                           if h.get("precondition") == pre)
            if found:
                return found[0][1], solves[found[0][1]]
        if tag is None and solves:
            return sorted(solves.items())[0]
        return None

    def keys(self):
        """The sample lattices whose comb responses at the initial model are on file, as the
        driver names them (``s16r8t1``: step, radius, teeth)."""
        head, tail = "psf_start_", "_responses.npy"
        files = glob.glob(self.path(head + "*" + tail))
        return sorted(os.path.basename(f)[len(head):-len(tail)] for f in files)

    def depth_node(self, depth=None):
        """The lattice plane of a map: the one nearest ``depth`` (m), or by default the middle
        one of those on which the window is one under the middle of the box."""
        if depth is not None:
            return int(np.argmin(np.abs(self.z - depth)))
        one = np.nonzero(self.window[:, self.jmid, self.shape[2] // 2] >= 1.0 - 1e-12)[0]
        return int(one[one.size // 2]) if one.size else self.shape[0] // 2

    def describe(self):
        """The box and the experiment in two lines, from the set-up's numbers."""
        a, s = self.args, self.sizes
        size = [float(v) * KM for v in str(a["size"]).split(",")]
        one = ("Marmousi in three dimensions, %s km in elements of %g m and degree %d: %s parameters, %s "
               "wave unknowns" % (" x ".join("%g" % v for v in size), s["h"], s["order"],
                                  _count(s["parameters"]), _count(s["wave_unknowns"])))
        two = ("%d shots, %d receivers %g m down, %.1f s of a wavelet about %g Hz; noise %g of the records' "
               "root mean square; prior %g m/s over %g m"
               % (s["shots"], s["receivers"], a["depth"], s["record"], s.get("wavelet", {}).get("fc", np.nan),
                  a["noise"], a["prior_sigma"], a["prior_rho"]))
        return one, two


def _count(n):
    """A count as it is read: 30 625, 11.6 million, 783 billion."""
    n = float(n)
    for size, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if n >= size:
            return "%.3g %s" % (n / size, name)
    return format(int(round(n)), ",").replace(",", " ")


def _short(n):
    """A count on an axis: 461 k, 12 M, 1.5 G, 783 G."""
    n = float(n)
    for size, name in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= size:
            return ("%.0f %s" if n / size >= 9.95 else "%.1f %s") % (n / size, name)
    return "%d" % n


# ---------------------------------------------------------------------------------- drawing
def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _maps():
    """The two colour maps: of a magnitude, of a signed quantity."""
    from matplotlib.colors import LinearSegmentedColormap

    return (LinearSegmentedColormap.from_list("magnitude", RAMP),
            LinearSegmentedColormap.from_list("signed", POLES))


def _canvas(width, height):
    """A page and the function that puts axes on it by inches from its left and its top: boxes
    cut to the proportions of what they show line up (as in ``marmousi.figures``)."""
    fig = _pyplot().figure(figsize=(width, height))

    def place(x0, top, w, h):
        return fig.add_axes([x0 / width, 1.0 - (top + h) / height, w / width, h / height])

    return fig, place


def _wrapped(lines, width, left):
    """The lines of a heading, the long ones broken at blanks to fit a page ``width`` inches
    wide (12.6 and 15.6 letters an inch: what the two sizes of the type come to)."""
    out = []
    for n, line in enumerate(lines):
        room = int((width - left - 0.25) * (12.6 if n == 0 else 15.6))
        out += textwrap.wrap(line, room, break_long_words=False, break_on_hyphens=False) or [""]
    return out


def _head_height(lines, width, left=0.72):
    """The inches that a heading takes at the top of a page."""
    return 0.22 + 0.25 + 0.19 * (len(_wrapped(lines, width, left)) - 1) + 0.08


def _head(fig, lines, left=0.72):
    """A figure's heading: its first line larger, the others in the second ink."""
    w, h = fig.get_size_inches()
    top = 0.22
    for n, line in enumerate(_wrapped(lines, w, left)):
        fig.text(left / w, 1.0 - top / h, line, ha="left", va="top", fontsize=10.5 if n == 0 else 8.5,
                 color=INK if n == 0 else INK2)
        top += 0.25 if n == 0 else 0.19


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=DPI)
    _pyplot().close(fig)
    return path


def _plane(ax, F, a, plane, smooth, **kw):
    """A field of the lattice on a plane, axes in km: ``a[k, i]`` on a section along the line
    (``plane`` "xz"), depth down, or ``a[j, i]`` on a map ("xy").  A node is the middle of
    its cell; the half cells beyond the box are cut off.  ``smooth``: the bilinear
    interpolant of a function on the lattice, else a cell a node."""
    x = F.x * KM
    v = (F.z if plane == "xz" else F.y) * KM
    hx, hv = 0.5 * (x[1] - x[0]), 0.5 * (v[1] - v[0])
    if plane == "xz":
        extent, origin, lim = [x[0] - hx, x[-1] + hx, v[-1] + hv, v[0] - hv], "upper", (v[-1], v[0])
    else:
        extent, origin, lim = [x[0] - hx, x[-1] + hx, v[0] - hv, v[-1] + hv], "lower", (v[0], v[-1])
    im = ax.imshow(a, extent=extent, origin=origin, aspect="auto",
                   interpolation="bilinear" if smooth else "nearest", **kw)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*lim)
    ax.tick_params(length=2.5)
    return im


def _window_line(ax, F, plane, at):
    """The half level of the window on a plane: inside it the unknown moves the speed."""
    w = F.window[:, at, :] if plane == "xz" else F.window[at, :, :]
    if w.min() < 0.5 < w.max():
        v = F.z if plane == "xz" else F.y
        for color, width in ((SURFACE, 1.5), (INK, 0.6)):
            ax.contour(F.x * KM, v * KM, w, levels=[0.5], colors=[color], linewidths=width)


def _cross(ax, at):
    """A thin line across a panel, light under dark: where the plane of the panel beside it
    cuts this one."""
    for color, width in ((SURFACE, 1.4), (INK2, 0.5)):
        ax.axhline(at, color=color, lw=width)


def _survey(ax, F, plane):
    """Sources and receivers on a panel: in plan, or on a section at their depth whatever
    their distance from it.  The marks shrink with the distance of their neighbours."""
    v = 2 if plane == "xz" else 1
    wide = ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0          # the panel, points
    for points, mark, most, look in ((F.receivers, "o", 1.5, dict(mfc=INK, mec=SURFACE, mew=0.2, zorder=4)),
                                     (F.sources, "*", 5.0, dict(mfc=INK, mec=SURFACE, mew=0.4, zorder=5))):
        apart = wide / max(np.unique(np.round(points[:, 0], 3)).size, 1)
        ax.plot(points[:, 0] * KM, points[:, v] * KM, ls="", marker=mark, ms=min(most, 0.3 * apart),
                clip_on=False, **look)


def _bar(fig, im, cax, label, horizontal=False, extend="neither"):
    cb = fig.colorbar(im, cax=cax, orientation="horizontal" if horizontal else "vertical", extend=extend)
    cb.set_label(label)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(length=2.5)
    return cb


def _limit(fields, share=99.8):
    """A limit for the colours of signed fields: that share (per cent) of their sizes lies
    within it, a round number; and whether any value lies beyond."""
    v = np.abs(np.concatenate([np.ravel(f) for f in fields]))
    top = float(v.max())
    if not top > 0.0:
        return 1.0, False
    want = float(np.percentile(v, share)) or top           # (a field that is zero nearly everywhere)
    unit = 10.0 ** np.floor(np.log10(want))
    lim = next(m * unit for m in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
               if m * unit >= want * (1.0 - 1e-12))
    return lim, top > lim


# ---------------------------------------------------------------------------- the overview
def fig_overview(F, tag=None, depth=None, path=None):
    """``fig_overview.png``: the true speed, the initial model and the MAP point of ``tag`` (a
    solve's tag or its preconditioner; by default the point spread functions' if the folder
    has it), with the differences of the last two from the first, on the section in the
    middle of the box and on the map at ``depth`` m (by default the middle of the depths where
    the window is one).  One scale for the speeds, one for the differences.

    The truth is the speed the data were made of: ``truth.npy`` everywhere for a run with
    ``--full-truth``, and otherwise inside the window alone, ``initial + window * (truth -
    initial)``: the speed that the unknown can reach, and the one that the model error of
    its history is measured against.

    Reads ``truth.npy``, ``initial.npy``, ``window.npy``, ``survey.npz``, ``setup.json``,
    ``data.json`` and, if they are there, ``map_<tag>.npy`` and ``map_<tag>.json``.  A box
    three times as long as it is deep or wide gets a row a field; a shorter one a column."""
    plt = _pyplot()
    ramp, signed = _maps()
    j, k = F.jmid, F.depth_node(depth)
    found = F.solve(tag)
    if tag is not None and found is None:
        print("fig_overview: %s has no MAP solve %r (it has: %s); drawn without one"
              % (F.out, tag, ", ".join(F.solves()) or "none"))
    lines = list(F.describe())
    lines.append("On the section across the middle of the box, y = %g km, and on the map %.2g km down.  Thin "
                 "lines: where each cuts the other, and the half level of the window.  Stars and dots: "
                 "sources and receivers.%s"
                 % (F.y[j] * KM, F.z[k] * KM, "  The truth is the data's: the Marmousi speed where the "
                    "window is one, the initial model where it is zero" if F.windowed else ""))
    items = [("true wave speed", F.medium, False), ("initial model", F.initial, False)]
    if found is not None:
        name, hist = found
        m = F.load("map_%s.npy" % name)
        last = hist["iterations"][-1]
        speed = F.initial + F.window * m
        # the driver's model error, from the arrays: the reading of the files held against its record
        q = F.window * (F.truth - F.initial)
        mine = np.linalg.norm(F.window * m - q) / np.linalg.norm(q)
        if abs(mine - last["model_error"]) > 1e-6 * max(mine, 1.0):
            print("fig_overview: the model error of map_%s.npy is %.6f and the last row of its history has "
                  "%.6f (a solve that is still running writes the two one after the other)"
                  % (name, mine, last["model_error"]))
        how = "L-BFGS" if hist.get("solver") == "lbfgs" else "Gauss-Newton-CG"
        pre = PRECONDITIONERS.get(hist.get("precondition"), (hist.get("precondition"),))[0]
        lines.append("MAP point (%s): %s with %s, %d iterations, %d forward solves, |g| / |g$_0$| = %.1e%s; "
                     "inside the window %.3f of the initial model's distance from the truth"
                     % (name, how, pre, last["it"], last["forward_solves"], last["gradient"] / hist["g0"],
                        "" if hist.get("done") else " (not ended)", mine))
        items.append(("MAP point", speed, False))
    items.append(("initial model - truth", F.initial - F.medium, True))
    if found is not None:
        items.append(("MAP point - truth", speed - F.medium, True))
    speeds = [f for _, f, difference in items if not difference]
    vmin, vmax = min(f.min() for f in speeds), max(f.max() for f in speeds)
    lim, beyond = _limit([plane for _, f, difference in items if difference for plane in (f[:, j, :], f[k])])
    Lx, Ly, Lz = (c[-1] - c[0] for c in (F.x, F.y, F.z))
    n, ns = len(items), len(speeds)
    stack = Lx >= 3.0 * max(Ly, Lz)
    left, right, boxes = 0.72, 0.3, {}
    with plt.rc_context(STYLE):
        if stack:
            # a row a field: the section and the map side by side, the two scales at the right
            width, gap = 12.0, 0.62
            a = (width - left - right - gap - 1.0) / 2.0
            va, pa = a * Lz / Lx, a * Ly / Lx
            row = max(va, pa) + 0.42
            top = _head_height(lines, width, left) + 0.3
            height = top + n * row + 0.2
            for i in range(n):
                boxes[i, 0] = (left, top + i * row, a, va)
                boxes[i, 1] = (left + a + gap, top + i * row, a, pa)
        else:
            # a column a field: the section above the map, the two scales underneath
            gap = 0.2
            a = min(2.5, (12.0 - left - right - (n - 1) * gap) / n)
            va, pa = a * Lz / Lx, a * Ly / Lx
            width = left + n * a + (n - 1) * gap + right
            top = _head_height(lines, width, left) + 0.3
            height = top + va + 0.2 + pa + 0.45 + 0.75
            for i in range(n):
                boxes[i, 0] = (left + i * (a + gap), top, a, va)
                boxes[i, 1] = (left + i * (a + gap), top + va + 0.2, a, pa)
        fig, place = _canvas(width, height)
        _head(fig, lines, left)
        shown = {}
        for i, (title, field, difference) in enumerate(items):
            kw = dict(cmap=ramp, vmin=vmin, vmax=vmax)
            if difference:
                kw = dict(cmap=signed, vmin=-lim, vmax=lim)
            for p, plane in enumerate(("xz", "xy")):
                ax = place(*boxes[i, p])
                shown[difference] = _plane(ax, F, field[:, j, :] if p == 0 else field[k], plane, True, **kw)
                _window_line(ax, F, plane, j if p == 0 else k)
                if i == 0:
                    _cross(ax, (F.z[k] if p == 0 else F.y[j]) * KM)
                    _survey(ax, F, plane)
                labelled = stack or i == 0                       # the panels that carry the vertical axis
                last_row = i == n - 1 if stack else p == 1       # ... and the horizontal one
                ax.set_ylabel(("depth (km)" if p == 0 else "y (km)") if labelled else None)
                ax.set_xlabel("x (km)" if last_row else None)
                ax.tick_params(labelbottom=last_row, labelleft=labelled)
                if p == 0:
                    ax.set_title(title)
                elif stack:
                    ax.set_title("%s, %.2g km down" % (title, F.z[k] * KM))
        extend = "both" if beyond else "neither"
        if stack:
            x0 = left + 2 * a + gap + 0.18
            _bar(fig, shown[False], place(x0, top, 0.13, ns * row - 0.42), "wave speed (m/s)")
            _bar(fig, shown[True], place(x0, top + ns * row, 0.13, (n - ns) * row - 0.42), "difference (m/s)",
                 extend=extend)
        else:
            y0 = top + va + 0.2 + pa + 0.5
            _bar(fig, shown[False], place(left, y0, ns * a + (ns - 1) * gap, 0.12), "wave speed (m/s)", True)
            _bar(fig, shown[True], place(left + ns * (a + gap), y0, (n - ns) * a + (n - ns - 1) * gap, 0.12),
                 "difference of the speeds (m/s)", True, extend)
        return _save(fig, path or F.path("fig_overview.png"))


# ------------------------------------------------- the point spread functions, in NumPy
def _fast_length(n):
    """The first length from ``n`` on with no factor but 2, 3 and 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def white(seed, index, shape):
    """The ``index``-th field of independent standard normals that the library's random
    stream of ``seed`` gives a lattice of ``shape``: ``prior.LatticePrior.noise`` draws a
    node's number at its global number ``i + nx (j + ny k)`` and moves on by the number of
    nodes, a word of Philox a node, and a normal is the inverse distribution function of
    the word's 53 leading bits (``hippymfem.common.random``)."""
    n = int(np.prod(shape))
    block, rem = divmod(index * n, 4)                 # the generator moves by blocks of four words
    words = np.random.Philox(key=int(seed))
    if block:
        words.advance(block)
    u = np.clip(np.random.Generator(words).random(rem + n)[rem:], 1e-300, 1.0 - 1e-16)
    return np.fromiter(map(statistics.NormalDist().inv_cdf, u), dtype=np.float64, count=n).reshape(shape)


def _weights(n, first, step, K, kind):
    """The interpolation weights of ``K`` sample points ``first + step i`` at the ``n`` nodes
    of one direction, ``(K, n)``: hats (or quadratic B-splines), a node beyond the outermost
    points counted to the nearest of them (``psf.weight_fields``)."""
    t = (np.arange(n) - first) / float(step)
    w = np.zeros((K, n))
    for i in range(int(np.floor(t.min())) - 2, int(np.ceil(t.max())) + 3):
        y = np.abs(t - i)
        if kind == "hat":
            b = np.maximum(0.0, 1.0 - y)
        else:
            b = np.where(y <= 0.5, 0.75 - y ** 2, 0.5 * np.maximum(0.0, 1.5 - y) ** 2)
        w[min(max(i, 0), K - 1)] += b
    return w


def _gaussian(sigma, r):
    """The kernel of the operator's low-frequency term, ``psf.gaussian``: cut at four widths
    and at the radius ``r``, of sum one."""
    y = np.arange(-r, r + 1)
    g1 = np.where(np.abs(y) <= int(4.0 * sigma + 0.5), np.exp(-0.5 * (y / float(sigma)) ** 2), 0.0)
    g1 = g1 / g1.sum()
    return g1[:, None, None] * g1[None, :, None] * g1[None, None, :]


class Operator:
    """``H~ = W S S^T W`` on a whole lattice from the kernels ``g`` of all its sample points,
    ``(K_z, K_y, K_x, 2 r + 1, 2 r + 1, 2 r + 1)``: ``S u = sum_k w_k (g_k * u)`` and ``S^T v =
    sum_k g_k * (w_k v)`` by transforms of the zero-padded lattice, a pair a sample point.
    What ``psf.PSFOperator`` computes box by box, here for a picture."""

    def __init__(self, g, first, K, step, window, weights="hat"):
        self.g, self.K, self.window = g, tuple(int(k) for k in K), window
        self.shape = window.shape
        r = self.r = (g.shape[-1] - 1) // 2
        self.big = tuple(_fast_length(n + r) for n in self.shape)
        self.w = [_weights(self.shape[d], int(first[d]), int(step), self.K[d], weights) for d in range(3)]
        self.crop = tuple(slice(0, n) for n in self.shape)
        count = int(np.prod(self.K))
        # the kernels' symbols are kept where they take under half a gigabyte
        self._kept = {} if count * self.big[0] * self.big[1] * (self.big[2] // 2 + 1) * 8 <= 2 ** 29 else None
        #: the points that one product with ``H~`` transforms
        self.work = (2 * count * (1 if self._kept is not None else 2) + 2) * int(np.prod(self.big))

    def _weight(self, at):
        wz, wy, wx = (self.w[d][at[d]] for d in range(3))
        return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]

    def _symbol(self, at):
        if self._kept is not None and at in self._kept:
            return self._kept[at]
        r = self.r
        box = np.zeros(self.big)
        box[:2 * r + 1, :2 * r + 1, :2 * r + 1] = self.g[at]
        s = np.fft.rfftn(np.roll(box, (-r, -r, -r), axis=(0, 1, 2)), axes=(0, 1, 2)).real
        if self._kept is not None:
            self._kept[at] = s
        return s

    def S(self, u):
        U = np.fft.rfftn(u, s=self.big, axes=(0, 1, 2))
        out = np.zeros(self.shape)
        for at in np.ndindex(*self.K):
            part = np.fft.irfftn(U * self._symbol(at), s=self.big, axes=(0, 1, 2))
            out += self._weight(at) * part[self.crop]
        return out

    def St(self, v):
        acc = 0.0
        for at in np.ndindex(*self.K):
            acc = acc + np.fft.rfftn(self._weight(at) * v, s=self.big, axes=(0, 1, 2)) * self._symbol(at)
        return np.fft.irfftn(acc, s=self.big, axes=(0, 1, 2))[self.crop]

    def H(self, u):
        return self.window * self.S(self.St(self.window * u))


def _signed_root(lim):
    """Colours that run with the square root of a signed value's size, between ``-lim`` and
    ``lim``: responses twenty times smaller than the largest still show."""
    from matplotlib.colors import FuncNorm

    def forward(v):
        return np.sign(v) * np.sqrt(np.abs(v) / lim)

    def inverse(t):
        return np.sign(t) * lim * np.asarray(t) ** 2

    return FuncNorm((forward, inverse), vmin=-lim, vmax=lim)


def _psf_material(F, key):
    """What ``fig_psf.png`` is drawn of, as far as the folder has it: the combs' responses;
    the sample points and those to look at closely; the operator and its response to their
    combs; the first check vector with the Hessian's product and the operator's."""
    name = "psf_start_" + key
    P = types.SimpleNamespace(name=name, resp=F.load(name + "_responses.npy"), first=None, K=None, step=None,
                              nodes=None, j=F.jmid, k=F.depth_node(), teeth=1, rho=0, shown=[], ours={},
                              check=None, psf=None)
    P.rec = F.json(name + ".json") if F.has(name + ".json") else None
    P.settings = P.rec["settings"] if P.rec is not None else {}
    P.filtered = bool(P.settings.get("filter"))
    kern = F.load(name + "_kernels.npz") if F.has(name + "_kernels.npz") else None
    if kern is not None:
        P.first, P.K = [int(v) for v in kern["sample_first"]], [int(v) for v in kern["sample_K"]]
        P.step = int(kern["step"])
    elif P.rec is not None:
        P.first, P.K = list(P.settings["sample_first"]), list(P.settings["sample_points"])
        P.step = int(P.settings["step"])
    else:
        print("fig_psf: %s has neither %s_kernels.npz nor %s.json: the sample points are not marked"
              % (F.out, name, name))
        return P
    first, K, step = P.first, P.K, P.step
    # (as many combs as teeth^3: sample points from one tooth of a comb to the next, a direction)
    P.teeth = teeth = int(P.settings.get("teeth") or round(len(P.resp) ** (1.0 / 3.0)))
    P.nodes = [first[d] + step * np.arange(K[d]) for d in range(3)]
    # the sample points whose kernels the file has: all of them, or those of the first rank's block
    have = [range(K[d]) for d in range(3)]
    if kern is not None:
        have = [range(int(a), int(a) + n) for a, n in zip(kern["first"], kern["g"].shape[:3])]
    whole = kern is not None and all(list(have[d]) == list(range(K[d])) for d in range(3))
    # the section: through the sample points nearest the middle of the box; the map: through the shallowest
    iy = min(have[1], key=lambda i: (abs(P.nodes[1][i] - F.jmid), i))
    P.j, P.k = int(P.nodes[1][iy]), int(P.nodes[0][0])

    def color(at):
        """The comb of a sample point, as ``psf.SamplePoints.color`` numbers them."""
        return int(np.ravel_multi_index(tuple(i % teeth for i in at), (teeth,) * 3))

    def comb(c):
        """The comb of a colour: offsets of one at its sample points."""
        v = np.zeros(F.shape)
        for at in np.ndindex(*K):
            if color(at) == c:
                v[tuple(first[d] + step * at[d] for d in range(3))] = 1.0
        return v

    P.color = color
    if kern is not None:
        # up to six sample points of the section, of the first and of the last depth on file
        P.rho = (kern["psf"].shape[-1] - 1) // 2
        rows = sorted(set((have[0][0], have[0][-1])))
        each = min(len(have[2]), 6 // len(rows))
        spread = np.unique(np.rint((np.arange(each) + 0.5) * len(have[2]) / each - 0.5).astype(int))
        P.shown = [(iz, iy, have[2][int(ix)]) for iz in rows for ix in spread]
        P.psf = {at: np.asarray(kern["psf"][tuple(at[d] - have[d][0] for d in range(3))][:, P.rho, :])
                 for at in P.shown}

    # ---- the operator, where the folder has all it is made of and the lattice is not too large
    op, why = None, None
    if kern is None or P.rec is None:
        why = "the folder has no %s" % (name + ("_kernels.npz" if kern is None else ".json"))
    elif not whole:
        why = ("%s_kernels.npz has the kernels of %s of the %s sample points (those of the first rank's "
               "block: a run of the decomposed layout)" % (name, " x ".join(str(len(h)) for h in have),
                                                           " x ".join(map(str, K))))
    else:
        g = np.asarray(kern["g"], dtype=np.float64)
        rg = (g.shape[-1] - 1) // 2
        if P.settings.get("eta"):
            # (the width of the low-frequency term is not in the record: the folder's last run's, or
            # the driver's default; the check below would tell another)
            g = g + P.settings["eta"] * P.rec["smax"] * _gaussian(F.args.get("psf_blur") or 0.25 * rg, rg)
        op = Operator(g, first, K, step, F.window, P.settings.get("weights", "hat"))
        if op.work > WORK:
            op, why = None, ("one product would transform %.1e points here (%d sample points on %s nodes)"
                             % (op.work, int(np.prod(K)), _count(F.window.size)))
    if why:
        print("fig_psf: the operator is not applied: %s" % why)
        return P
    budget = WORK - op.work                                 # (the check vector's product has the first claim)
    colors = sorted(set(color(at) for at in P.shown))
    if P.filtered:
        print("fig_psf: the combs of %s were high-pass filtered (%g nodes): the operator's response to them "
              "is not drawn" % (key, P.settings["filter"]))
    elif len(colors) * op.work > budget:
        print("fig_psf: the operator's response to the %d comb%s of the sample points looked at would "
              "transform %.1e points: not drawn"
              % (len(colors), "" if len(colors) == 1 else "s", len(colors) * op.work))
    else:
        P.ours = {c: op.H(comb(c)) for c in colors}

    # ---- the check vector: the first white one, drawn again and held against the stage's record
    label = "eta %g" % P.settings["eta"] if P.settings.get("eta") else "eta 0"
    errors = P.rec.get("errors", {})
    told = errors.get(label, {}).get("white", [])
    if not F.has(name + "_check.npy"):
        print("fig_psf: %s has no %s_check.npy: no check vector" % (F.out, name))
    elif not told:
        print("fig_psf: the stage psf of %s had no white check vector (--psf-check 0)" % key)
    else:
        Hv = F.load(name + "_check.npy")[0]
        ours = op.H(white(F.args["seed"] + 100, 0, F.shape))
        mine = float(np.linalg.norm(ours - Hv) / np.linalg.norm(Hv))
        if abs(mine - told[0]) > 1e-6 * told[0]:
            print("fig_psf: the first check vector could not be drawn again: the operator's error on it is "
                  "%.6f here and %.6f in %s.json; no check vector" % (mine, told[0], name))
        else:
            bare = errors.get("eta 0", {}).get("white", [None])[0] if label != "eta 0" else None
            P.check = types.SimpleNamespace(Hv=Hv, ours=ours, error=told[0], bare=bare,
                                            count=int(P.rec.get("checks", 0)))
    return P


def fig_psf(F, key=None, path=None):
    """``fig_psf.png``: the point spread functions of the sample lattice ``key`` (as the
    driver names it, ``s8r4t1``; by default the folder's first).

    The first row is the Hessian's response to offsets of one at all the sample points (the
    sum of the combs' responses), on the section through the sample points nearest the
    middle of the box and on the map through the shallowest of them.  The second, around a
    few sample points: that response, the impulse response that the mask cuts out of it,
    and the operator's response to the same comb.  The third: the Hessian's product with
    the first check vector, the operator's, and their difference, with the error that the
    stage recorded.

    Reads ``psf_start_<key>_responses.npy``; for the sample points and the second row
    ``psf_start_<key>_kernels.npz``; for the operator also ``psf_start_<key>.json``; for the
    third row ``psf_start_<key>_check.npy``."""
    plt = _pyplot()
    _, signed = _maps()
    keys = F.keys()
    if key is None and keys:
        key = keys[0]
    if key not in keys:
        print("no fig_psf: %s holds no comb responses%s (the stage psf writes psf_start_<key>_responses.npy; "
              "it has: %s)" % (F.out, "" if key is None else " of " + key, ", ".join(keys) or "none"))
        return None
    P = _psf_material(F, key)
    first, K, step, rho, j, k, check = P.first, P.K, P.step, P.rho, P.j, P.k, P.check
    total = P.resp.sum(axis=0)

    def around(field, at):
        """A field on the nodes within the mask's radius of a sample point, in its section."""
        c = [first[d] + step * at[d] for d in range(3)]
        padded = np.pad(field[:, c[1], :], rho)
        return padded[c[0]:c[0] + 2 * rho + 1, c[2]:c[2] + 2 * rho + 1]

    # ---- the page: its three rows by inches
    Lx, Ly, Lz = (c[-1] - c[0] for c in (F.x, F.y, F.z))
    left, right, width = 0.72, 0.25, 12.0
    a1 = min((width - left - right - 1.15 - 0.7) / 2.0, 2.2 * Lx / max(Ly, Lz))
    v1, p1 = a1 * Lz / Lx, a1 * Ly / Lx
    h1 = max(v1, p1)
    cols = len(P.shown)
    side = min(1.3, (width - left - right - 2.75 - 0.14 * max(cols - 1, 0)) / max(cols, 1))
    nrow2 = 2 + bool(P.ours)
    h2 = nrow2 * side + (nrow2 - 1) * 0.1
    a3 = min((width - left - right - 1.15 - 2 * 0.3) / 3.0, 1.9 * Lx / Lz)
    v3 = a3 * Lz / Lx
    lines = ["The point spread functions of the misfit Hessian at the initial model, sample lattice %s" % key,
             F.describe()[0]]
    if P.rec is not None:
        s = P.settings
        lines.append("%d sample points (%s along z, y, x) %d nodes apart; mask of radius %d nodes; %d comb%s "
                     "and %d check vectors in one batch of products; kernels of radius %d, weights %s, "
                     "low-frequency term %g"
                     % (int(np.prod(K)), " x ".join(map(str, K)), s["step"], s["rho"], P.rec["combs"],
                        "" if P.rec["combs"] == 1 else "s", P.rec["checks"], s["rho_g"], s["weights"],
                        s["eta"]))
    said = ["The Hessian's response to the comb%s: its product with offsets of 1 m/s at all the sample "
            "points%s, before the window%s.  The colour runs with the square root of the size"
            % ("s (the sum of the %d)" % len(P.resp) if len(P.resp) > 1 else "",
               " (circles)" if P.nodes is not None else "",
               "; the combs were high-pass filtered" if P.filtered else ""),
            "Around the numbered sample points, in the plane of the section, each column over the largest "
            "size of the Hessian's response at its point.%s"
            % ("  Under a column: that size, and %s's response against the Hessian's: at the point, and the "
               "difference of the two over the Hessian's within the mask's radius" % H_TILDE
               if P.ours else "")]
    if check:
        said.append("The first of the %d check vectors (white noise on the nodes): the Hessian's product "
                    "with it, the operator's, and their difference, on the section y = %g km"
                    % (check.count, F.y[j] * KM))
    said = [textwrap.wrap(t, int((width - left - 0.25) * 15.6)) for t in said]
    head = _head_height(lines, width, left)
    t1 = head + 0.08 + 0.19 * len(said[0]) + 0.3
    bottom = t1 + h1 + 0.45
    if cols:
        t2 = bottom + 0.15 + 0.19 * len(said[1]) + 0.27
        bottom = t2 + h2 + 0.8
    if check:
        over = 0.27 if check.bare is None else 0.42                      # one line of title, or two
        t3 = bottom + 0.15 + 0.19 * len(said[2]) + over
        bottom = t3 + v3 + 0.45
    with plt.rc_context(STYLE):
        fig, place = _canvas(width, bottom + 0.1)

        def say(top, rows):
            fig.text(left / width, 1.0 - top / (bottom + 0.1), "\n".join(rows), ha="left", va="top",
                     fontsize=8.5, color=INK, linespacing=1.3)

        _head(fig, lines, left)
        # ---- the response to the comb
        say(head + 0.08, said[0])
        lim = float(np.abs(total).max()) or 1.0
        for p, (plane, field, title) in enumerate((
                ("xz", total[:, j, :], "on the section y = %g km" % (F.y[j] * KM)),
                ("xy", total[k], "on the map %.2g km down" % (F.z[k] * KM)))):
            ax = place(left + p * (a1 + 0.7), t1, a1, v1 if p == 0 else p1)
            im = _plane(ax, F, field, plane, False, cmap=signed, norm=_signed_root(lim))
            _window_line(ax, F, plane, j if p == 0 else k)
            if P.nodes is not None:
                across = F.z[P.nodes[0]] if p == 0 else F.y[P.nodes[1]]
                X, V = np.meshgrid(F.x[P.nodes[2]] * KM, across * KM)
                ax.plot(X.ravel(), V.ravel(), ls="", marker="o", ms=3.2, mfc="none", mec=INK, mew=0.5)
            if p == 0:
                for n, at in enumerate(P.shown):
                    point = (F.x[first[2] + step * at[2]] * KM, F.z[first[0] + step * at[0]] * KM)
                    ax.annotate(str(n + 1), point, textcoords="offset points", xytext=(3.5, 3.5), fontsize=7,
                                color=INK)
            ax.set_title(title)
            ax.set_xlabel("x (km)")
            ax.set_ylabel("depth (km)" if p == 0 else "y (km)")
        cb = _bar(fig, im, place(left + 2 * a1 + 0.7 + 0.18, t1, 0.13, h1), "response (s/m)")
        ticks = [t * lim for t in (-1.0, -0.25, 0.0, 0.25, 1.0)]
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels(["%.2g" % t for t in ticks])
        # ---- around the sample points
        if cols:
            say(t1 + h1 + 0.45 + 0.15, said[1])
            names = ["the Hessian's response\nto the comb",
                     "cut by the mask: the\nimpulse response that\nthe kernel is made of"]
            if P.ours:
                names.append("%s's response\nto the same comb" % H_TILDE)
            e, half = rho * F.h * KM, 0.5 * F.h * KM
            tick = e if rho <= 2 else 0.5 * e
            clipped = False
            for c, at in enumerate(P.shown):
                fields = [around(P.resp[P.color(at)], at), P.psf[at]]
                if P.ours:
                    fields.append(around(P.ours[P.color(at)], at))
                peak = float(np.abs(fields[0]).max()) or 1.0
                for r, f in enumerate(fields):
                    ax = place(left + c * (side + 0.14), t2 + r * (side + 0.1), side, side)
                    im2 = ax.imshow(f / peak, extent=[-e - half, e + half, e + half, -e - half], cmap=signed,
                                    vmin=-1.0, vmax=1.0, interpolation="nearest", aspect="auto")
                    clipped = clipped or np.abs(f).max() > peak * (1.0 + 1e-9)
                    for axis in (ax.xaxis, ax.yaxis):
                        axis.set_ticks([-tick, 0.0, tick])
                        axis.set_ticklabels(["%g" % -tick, "0", "%g" % tick])
                    ax.tick_params(length=2.5, labelbottom=r == nrow2 - 1, labelleft=c == 0)
                    if c == 0:
                        ax.set_ylabel("depth - z$_k$ (km)", fontsize=7.5)
                    if r == 0:
                        ax.set_title("%d:  x %.2g, depth %.2g km" % (c + 1, F.x[first[2] + step * at[2]] * KM,
                                                                     F.z[first[0] + step * at[0]] * KM),
                                     fontsize=7.5)
                    if r == nrow2 - 1:
                        note = "%.2e s/m" % peak
                        if P.ours:
                            note += "\n%s: %.2f times, %.2f off" % (
                                H_TILDE, fields[2][rho, rho] / (fields[0][rho, rho] or 1.0),
                                np.linalg.norm(fields[2] - fields[0]) / (np.linalg.norm(fields[0]) or 1.0))
                        ax.set_xlabel("x - x$_k$ (km)\n%s" % note, fontsize=7.5, linespacing=1.3)
            x0 = left + cols * (side + 0.14)
            for r, text in enumerate(names):
                middle = t2 + r * (side + 0.1) + 0.5 * side
                fig.text(x0 / width, 1.0 - middle / (bottom + 0.1), text, ha="left", va="center",
                         fontsize=7.5, color=INK2, linespacing=1.3)
            _bar(fig, im2, place(x0 + 1.55, t2, 0.13, h2), "over the Hessian's largest at the point",
                 extend="both" if clipped else "neither")
        # ---- the check vector
        if check:
            say(t3 - over - 0.19 * len(said[2]), said[2])
            lim3, beyond = _limit([check.Hv[:, j, :], check.ours[:, j, :]], 99.9)
            titles = ("H v, the Hessian's", "%s v, the operator's" % H_TILDE,
                      "%s v - H v:  %.3f of |H v| over the lattice%s"
                      % (H_TILDE, check.error, "" if check.bare is None else
                         "\n(%.3f without the low-frequency term)" % check.bare))
            for p, f in enumerate((check.Hv, check.ours, check.ours - check.Hv)):
                ax = place(left + p * (a3 + 0.3), t3, a3, v3)
                im3 = _plane(ax, F, f[:, j, :], "xz", False, cmap=signed, vmin=-lim3, vmax=lim3)
                _window_line(ax, F, "xz", j)
                ax.set_title(titles[p], fontsize=8)
                ax.set_xlabel("x (km)")
                ax.set_ylabel("depth (km)" if p == 0 else None)
                ax.tick_params(labelleft=p == 0)
            _bar(fig, im3, place(left + 3 * a3 + 2 * 0.3 + 0.18, t3, 0.13, v3), "product (s/m)",
                 extend="both" if beyond else "neither")
        return _save(fig, path or F.path("fig_psf.png"))


# --------------------------------------------------------------------------- the MAP solves
def fig_convergence(F, path=None):
    """``fig_convergence.png``: the L-BFGS solves of the folder, a curve each: the gradient's
    norm over the first one's, the cost less the least cost any of them reached, and the
    model error, against the forward solves (top) and against the solve's wall time
    (bottom).  A forward solve is what the driver's history counts: a point at which every
    shot was propagated (with its gradient, unless the run was lazy).  The wall time is the
    solve's own: the products with the combs and the construction of the operator come
    before it.

    Reads ``map_<tag>.json`` of every tag, and for a note on what the operator cost the
    ``psf_start_<key>.json`` that a history names."""
    plt = _pyplot()
    solves = F.solves()
    for tag, h in solves.items():
        if h.get("solver") != "lbfgs":
            print("fig_convergence: map_%s.json is a solve by %s: its Hessian actions are no forward solves, "
                  "and it is not drawn" % (tag, h.get("solver")))
    solves = {t: h for t, h in solves.items() if h.get("solver") == "lbfgs"}
    if not solves:
        print("no fig_convergence: %s holds no L-BFGS history (the stage map writes map_<tag>.json)" % F.out)
        return None
    order = {p: n for n, p in enumerate(PRECONDITIONERS)}
    tags = sorted(solves, key=lambda t: (order.get(solves[t].get("precondition"), 9), t))
    least = min(r["cost"] for h in solves.values() for r in h["iterations"])
    shots = F.sizes["shots"]
    each = set()
    for h in solves.values():
        last = h["iterations"][-1]
        each.add(round(last["sweeps"] / float(max(last["forward_solves"], 1) * shots), 3))
    one, two = F.describe()
    sz = F.sizes
    lines = ["L-BFGS from the initial model: what its initial inverse Hessian is worth", one,
             "%s; the folder's last run was on %d rank%s (%s, %s)"
             % (two, sz.get("ranks", 0), "" if sz.get("ranks") == 1 else "s", sz.get("layout", "?"),
                sz.get("device", "?"))]
    if len(each) == 1:
        lines.append("A forward solve, as the histories count it: a point at which every shot was "
                     "propagated, %g sweeps a shot%s"
                     % (each.pop(), "" if F.args.get("lazy_gradient") else
                        " (forward for the misfit, backward for its gradient)"))
    note = _comb_cost(F, solves)
    width, left = 12.0, 0.78
    head = _head_height(lines, width, left)
    band = 0.2 + 0.2 * len(tags) + (0.42 if note else 0.0)       # the legend and the note underneath
    height = head + 0.25 + 5.6 + band
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 3, figsize=(width, height))
        fig.subplots_adjust(left=left / width, right=0.985, top=1.0 - (head + 0.25) / height,
                            bottom=(band + 0.48) / height, wspace=0.27, hspace=0.36)
        _head(fig, lines, left)
        seen, handles = {}, []
        for tag in tags:
            h = solves[tag]
            pre = h.get("precondition")
            label, hue, mark = PRECONDITIONERS.get(pre, (str(pre), HUES[3], "D"))
            dash = DASHES[seen.get(pre, 0) % len(DASHES)]          # a second solve with as much: dashes
            seen[pre] = seen.get(pre, 0) + 1
            rows = h["iterations"]
            last = rows[-1]
            g = np.array([r["gradient"] for r in rows]) / h["g0"]
            cost = np.array([r["cost"] for r in rows]) - least
            cost[cost <= 0.0] = np.nan                    # (the least one itself: off a logarithmic axis)
            err = np.array([r["model_error"] for r in rows])
            some = max(1, len(rows) // 40)                # a mark an iteration; of a long solve fewer
            for i, x in enumerate((np.array([r["forward_solves"] for r in rows]),
                                   np.array([r["seconds"] for r in rows]))):
                for ax, v in zip(axs[i], (g, cost, err)):
                    line, = ax.plot(x, v, color=hue, ls=dash, marker=mark, ms=3.6, mec=SURFACE, mew=0.5,
                                    markevery=some, lw=2.0 if pre == "psf" else 1.5)
                    ax.plot(x[-1:], v[-1:], color=hue, ls="", marker=mark, ms=6.5, mec=SURFACE, mew=1.0)
            how = "|g| / |g$_0$| = %.1e" % (last["gradient"] / h["g0"])
            if not h.get("done"):
                how += ", not ended"
            elif not h.get("converged"):
                how += ", %s" % h.get("reason")
            if last.get("inner_iterations") and last["it"]:
                how += "; %.0f iterations of the solve with %s + R an iteration" % (
                    last["inner_iterations"] / float(last["it"]), H_TILDE)
            if last.get("inner_unfinished"):
                how += ", %d of the solves cut short" % last["inner_unfinished"]
            if tag != "lbfgs-%s" % pre:
                label += " (%s%s)" % (tag, ", " + h["psf"] if h.get("psf") else "")
            line.set_label("%s: %d iterations, %d forward solves, %.0f s; %s"
                           % (label, last["it"], last["forward_solves"], last["seconds"], how))
            handles.append(line)
        rtol = F.args.get("rtol")
        for i in range(2):
            axs[i, 0].set_yscale("log")
            axs[i, 1].set_yscale("log")
            if rtol:
                axs[i, 0].axhline(rtol, color=MUTED, lw=0.8, zorder=1)
                axs[i, 0].annotate("--rtol %g" % rtol, (1.0, rtol), xycoords=("axes fraction", "data"),
                                   textcoords="offset points", xytext=(-2, 3), ha="right", fontsize=7,
                                   color=INK2)
            axs[i, 0].set_ylabel("gradient norm over the first, |g| / |g$_0$|")
            axs[i, 1].set_ylabel("cost - %.9e\n(the least cost of these solves)" % least)
            axs[i, 2].set_ylabel("model error, |W m - W q| / |W q|\n(q: the true offset from the initial "
                                 "model)")
            for ax in axs[i]:
                ax.grid(True, which="major")
                ax.set_xlim(left=0)
                ax.set_xlabel("forward solves (points at which all %d shots were propagated)" % shots
                              if i == 0 else "wall time of the solve (s)", fontsize=8)
        fig.legend(handles=handles, loc="upper left", ncol=1, fontsize=7.8, handlelength=3.2,
                   borderaxespad=0.0, bbox_to_anchor=((left - 0.04) / width, (band - 0.05) / height))
        if note:
            fig.text(left / width, 0.1 / height, note, ha="left", va="bottom", fontsize=7.8, color=INK2,
                     linespacing=1.35)
        return _save(fig, path or F.path("fig_convergence.png"))


def _comb_cost(F, solves):
    """What the curves of the point spread functions leave out: the products with the combs,
    in forward solves of the same folder (by the field-steps of the two records: a product
    on ``D`` directions carries ``1 + D`` fields through its two sweeps of a shot)."""
    for h in solves.values():
        name = "psf_start_%s.json" % h.get("psf")
        if h.get("precondition") != "psf" or not F.has(name):
            continue
        rec, last = F.json(name), h["iterations"][-1]
        combs, checks = rec.get("combs", 0), rec.get("checks", 0)
        if not (combs and rec.get("field_steps") and last.get("field_steps") and last.get("forward_solves")):
            continue
        solve = last["field_steps"] / float(last["forward_solves"])
        share = (1.0 + combs) / (1.0 + combs + checks)
        return ("Not in the curves: the products of the Hessian with the %d comb%s that %s (%s) is made of,\n"
                "%.1f forward solves' field-steps (%.1f with the %d check vectors of the same sweeps)"
                % (combs, "" if combs == 1 else "s", H_TILDE, h["psf"], share * rec["field_steps"] / solve,
                   rec["field_steps"] / solve, checks))
    return None


# ----------------------------------------------------------------- the Laplace approximation
def fig_posterior(F, tag=None, path=None):
    """``fig_posterior.png``: the pointwise standard deviation of the prior and of the
    Laplace approximation at the MAP point of ``tag``, the root mean squares of the stage
    post's draws, and their ratio, on the section in the middle of the box.

    Reads ``post_<tag>_samples.npz`` (``std``: prior, posterior) and ``post_<tag>.json``."""
    plt = _pyplot()
    ramp, signed = _maps()
    found = F.solve(tag)
    if found is None:
        print("no fig_posterior: %s holds no MAP solve%s (the stages map and post come first)"
              % (F.out, "" if tag is None else " " + tag))
        return None
    name = "post_%s" % found[0]
    if not F.has(name + "_samples.npz"):
        have = [os.path.basename(f)[5:-12] for f in glob.glob(F.path("post_*_samples.npz"))]
        print("no fig_posterior: %s holds no %s_samples.npz (the stage post writes it%s)"
              % (F.out, name, "; it has those of: " + ", ".join(have) if have else ""))
        return None
    z = F.load(name + "_samples.npz")
    std = np.asarray(z["std"])
    rec = F.json(name + ".json") if F.has(name + ".json") else {}
    asked = json.loads(str(z["of"])) if "of" in z.files else {}
    draws = int(rec.get("variance_samples") or asked.get("variance_samples") or z["samples"].shape[1])
    j = F.jmid
    Lx, Lz = F.x[-1] - F.x[0], F.z[-1] - F.z[0]
    left, right, width = 0.72, 0.25, 12.0
    a = min((width - left - right - 0.3 - 2 * 1.2) / 3.0, 2.0 * Lx / Lz)
    v = a * Lz / Lx
    at, last = rec.get("iteration", asked.get("iteration")), found[1]["iterations"][-1]["it"]
    where = "the MAP point of %s" % found[0]
    if at is not None and at != last:
        # (the stage post takes the iterate on file; the solve may have been continued since)
        where = "iterate %d of %s (the solve has gone on to iteration %d since)" % (at, found[0], last)
    lines = ["Pointwise standard deviation of the prior and of the Laplace approximation at %s" % where,
             F.describe()[0]]
    tail = "The root mean squares of %d draws, each within about %.0f %% of its value" % (
        draws, 100.0 / np.sqrt(2.0 * draws))
    if "prior_variance" in rec:
        tail += "; the prior's own, away from the faces: %.1f m/s" % np.sqrt(rec["prior_variance"])
    lines.append(tail)
    if "std_prior" in rec:
        lines.append("Where the window is one: prior %.2f m/s, posterior %.2f m/s%s; %.0f iterations a "
                     "draw's solve" % (rec["std_prior"], rec["std_post"],
                                       ", least ratio at a node %.2f" % rec["std_ratio_least"]
                                       if "std_ratio_least" in rec else "", rec.get("solve_iterations", 0)))
    with plt.rc_context(STYLE):
        width = left + 3 * a + 0.3 + 2 * 1.2 + right
        top = _head_height(lines, width, left) + 0.3
        fig, place = _canvas(width, top + v + 0.55)
        _head(fig, lines, left)
        most = float(max(std[0][:, j, :].max(), std[1][:, j, :].max()))
        ratio = std[1] / np.where(std[0] > 0.0, std[0], 1.0)
        both, about_one = dict(cmap=ramp, vmin=0.0, vmax=most), dict(cmap=signed, vmin=0.0, vmax=2.0)
        for p, (f, title, kw, label) in enumerate((
                (std[0], "prior", both, None), (std[1], "posterior", both, "standard deviation (m/s)"),
                (ratio, "posterior over prior", about_one, "ratio (1: nothing learnt)"))):
            x0 = left + p * a + (0.0, 0.3, 0.3 + 1.2)[p]
            ax = place(x0, top, a, v)
            im = _plane(ax, F, f[:, j, :], "xz", True, **kw)
            _window_line(ax, F, "xz", j)
            ax.set_title("%s, on the section y = %g km" % (title, F.y[j] * KM))
            ax.set_xlabel("x (km)")
            ax.set_ylabel("depth (km)" if p == 0 else None)
            ax.tick_params(labelleft=p == 0)
            if label:
                _bar(fig, im, place(x0 + a + 0.15, top, 0.13, v), label,
                     extend="max" if p == 2 and ratio[:, j, :].max() > 2.0 else "neither")
        return _save(fig, path or F.path("fig_posterior.png"))


def fig_chain(F, tag=None, path=None):
    """``fig_chain.png``: the traces of the chains from the MAP point of ``tag``: the misfit
    less the starting point's, the prior's term of the cost, and the distance from the
    starting point, a step at a time; a chain a curve.

    Reads ``chain_<tag>_<kernel>.json`` of every kernel (``gpcn``, ``pcn``), and the prior's
    term at the start from the row of ``map_<tag>.json`` that has the chains' first misfit
    (the last one, unless the solve was continued after the chains were run)."""
    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator

    found = F.solve(tag)
    if found is None:
        print("no fig_chain: %s holds no MAP solve%s (the stages map, post and chain come first)"
              % (F.out, "" if tag is None else " " + tag))
        return None
    name = "chain_%s_" % found[0]
    chains = {}
    for f in sorted(glob.glob(F.path(name + "*.json"))):
        with open(f) as fh:
            state = json.load(fh)
        if state.get("trace"):
            chains[os.path.basename(f)[len(name):-5]] = state
    if not chains:
        print("no fig_chain: %s holds no %s<kernel>.json with a trace (the stage chain writes it)"
              % (F.out, name))
        return None
    looks = {"gpcn": ("gpCN about the Laplace approximation", HUES[0], "o"),
             "pcn": ("pCN about the prior", HUES[1], "s")}
    panels = (("misfit", "misfit less the starting point's"), ("reg", "the prior's term of the cost"),
              ("from_map", "distance from the starting point,\n|W m - W m$_0$| / |W m$_0$|"))
    # the iterate the chains started from: the row of the solve's history that has their first misfit
    # (the solve may have been continued since the chains were run)
    start = next(iter(chains.values())).get("start_misfit")
    rows = found[1]["iterations"]
    row = next((r for r in rows if start is not None and abs(r["misfit"] - start) <= 1e-9 * abs(start)), None)
    where = "the MAP point of %s" % found[0]
    if row is not None and row["it"] != rows[-1]["it"]:
        where = "iterate %d of %s (the solve has gone on to iteration %d since)" % (row["it"], found[0],
                                                                                 rows[-1]["it"])
    elif row is None and start is not None:
        where = "a point of %s that its history on file does not have" % found[0]
    lines = ["Chains from %s" % where, F.describe()[0]]
    if start is not None:
        lines.append("The misfit at that point: %.6e" % start)
    width, left = 12.0, 0.78
    head = _head_height(lines, width, left)
    band = 0.2 + 0.2 * len(chains)
    height = head + 0.25 + 2.6 + 0.48 + band
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(1, 3, figsize=(width, height))
        fig.subplots_adjust(left=left / width, right=0.985, top=1.0 - (head + 0.25) / height,
                            bottom=(band + 0.48) / height, wspace=0.3)
        _head(fig, lines, left)
        handles = []
        for n, (kind, state) in enumerate(sorted(chains.items())):
            label, hue, mark = looks.get(kind, (kind, HUES[2 + n % 6], MARKS[2 + n % 10]))
            trace = state["trace"]
            steps = np.arange(len(trace) + 1)
            # step 0 is the point the chain started from: its misfit as the chain found it, its
            # prior's term from the solve's history
            at_start = dict(misfit=state.get("start_misfit", np.nan), from_map=0.0,
                            reg=row["reg"] if row is not None else np.nan)
            for ax, (what, _) in zip(axs, panels):
                v = np.array([at_start[what]] + [t[what] for t in trace], dtype=float)
                if what == "misfit":
                    v = v - (start if start is not None else trace[0]["misfit"])
                line, = ax.plot(steps, v, color=hue, marker=mark, ms=3.6 if steps.size > 40 else 5.0,
                                mec=SURFACE, mew=0.6)
            summary = state.get("summary") or {}
            line.set_label("%s, s = %g: %d of %d steps accepted, %d forward solves, %.0f s%s"
                           % (label, state.get("s", np.nan), state.get("accepted", 0), state.get("steps", 0),
                              state.get("forward_solves", 0), state.get("seconds", 0.0),
                              "; effective sample size of the misfit %.1f" % summary["ess"]
                              if "ess" in summary else ""))
            handles.append(line)
        for ax, (what, label) in zip(axs, panels):
            ax.set_xlabel("step of the chain (0: where it starts)")
            ax.set_ylabel(label)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True)
            if what == "misfit":
                ax.axhline(0.0, color=MUTED, lw=0.8, zorder=1)
        fig.legend(handles=handles, loc="upper left", ncol=1, fontsize=7.8, handlelength=3.2,
                   borderaxespad=0.0, bbox_to_anchor=((left - 0.04) / width, (band - 0.05) / height))
        return _save(fig, path or F.path("fig_chain.png"))


# -------------------------------------------------------------------------------- scaling
def read_record(path):
    """A scaling record in one form, or the reason it is not one: ``(record, None)`` or
    ``(None, reason)``.  Three drivers' records are read:

    ``scale.py`` (it has ``problem`` and ``cells``)
        ``sweeps.{forward, incremental, adjoint or backward}.ms_per_step`` and ``.fields``;
        ``products.H.ms`` and ``solves.<preconditioner>.ms``, each least, mean and greatest
        over the ranks; ``wave_dofs``, ``parameters``; ``ranks``, ``grid``, ``cells``;
        ``exchange``; of ``problem`` what a series must share (``float32, steps, directions,
        psf_step, psf_rho, psf_kernels``), and ``lattice.prior.solver``; ``ended``.
    ``box_ranks.py``, the propagator's own (``sweeps`` and ``args.n``)
        ``sweeps.{forward, incremental, backward}.ms_per_step``; ``wave_dofs``, ``parameters``;
        ``ranks``, ``grid``, ``args.n`` (the whole box: a rank's block is ``n / grid``);
        ``device_mpi``; ``args.float32, steps, directions, absorb, history``.
    ``psf_ranks.py``, the operator's own (``times``, ``step`` and ``rho``)
        ``times.H.ms`` and ``solves.<preconditioner>.iteration_ms`` as ``min, mean, max`` over
        the ranks; ``lattice`` (its nodes), ``nodes_a_rank.max``; ``ranks``, ``grid``;
        ``step, rho, weights, eta, dtype, halo``.

    The job is the number at the end of the folder that the record was written into
    (``args.out``), or of the folder it is read from."""
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, ValueError) as e:
        return None, "not read (%s)" % e
    if not isinstance(r, dict):
        return None, "no record"
    args = r.get("args") if isinstance(r.get("args"), dict) else {}
    where = args.get("out") if os.path.isabs(str(args.get("out") or "")) else path
    folder = os.path.basename(os.path.dirname(os.path.abspath(where)))
    digits = re.search(r"(\d{5,})$", folder)
    out = dict(file=path, name=os.path.basename(path), job=digits.group(1) if digits else folder,
               sweeps={}, lattice={})
    try:
        if "problem" in r and "cells" in r:
            p, lat = r["problem"], r.get("lattice") or {}
            out.update(driver="scale.py", ranks=int(r["ranks"]), grid=tuple(r["grid"]),
                       load=tuple(r["cells"]), exchange=r.get("exchange") or lat.get("exchange"),
                       what=(bool(p.get("float32")), p.get("steps"), p.get("directions"), p.get("psf_step"),
                             p.get("psf_rho"), p.get("psf_kernels"), (lat.get("prior") or {}).get("solver")),
                       unknowns=r.get("wave_dofs"), nodes=r.get("parameters") or lat.get("nodes"),
                       text="%s of " + " x ".join(map(str, r["cells"])) + " elements",
                       ended=r.get("ended", "whole"))
            for name, v in (r.get("sweeps") or {}).items():
                # (the backward sweep is "adjoint" in the driver's records since its first version)
                name = "backward" if name == "adjoint" else name
                out["sweeps"][name] = (v["ms_per_step"], v.get("fields", 1))
            if "H" in (r.get("products") or {}):
                out["lattice"]["H"] = tuple(r["products"]["H"]["ms"])
            for name, v in (r.get("solves") or {}).items():
                out["lattice"][name] = tuple(v["ms"])
        elif "sweeps" in r and "n" in args:
            grid = tuple(r["grid"])
            load = tuple(int(n) // g for n, g in zip(args["n"], grid))
            out.update(driver="box_ranks.py", ranks=int(r["ranks"]), grid=grid, load=load,
                       exchange="card to card" if r.get("device_mpi") else "host arrays",
                       what=(bool(args.get("float32")), args.get("steps"), args.get("directions"),
                             args.get("absorb"), args.get("history")),
                       unknowns=r.get("wave_dofs"), nodes=r.get("parameters"),
                       text="%s of " + " x ".join(map(str, load)) + " elements")
            for name, v in r["sweeps"].items():
                out["sweeps"][name] = (v["ms_per_step"], v.get("fields", 1))
        elif "times" in r and "step" in r and "rho" in r:
            most = r["nodes_a_rank"]["max"] if isinstance(r["nodes_a_rank"], dict) else r["nodes_a_rank"]
            what = (r.get("dtype"), r["step"], r["rho"], r.get("weights"), r.get("eta"), r.get("halo"))
            text = "sample points %d nodes apart, kernels of radius %d%s" % (
                r["step"], r["rho"], "" if r.get("dtype") == "float64" else ", %s" % r.get("dtype"))
            out.update(driver="psf_ranks.py", ranks=int(r["ranks"]), grid=tuple(r["grid"]), load=(int(most),),
                       exchange=None, unknowns=None, nodes=int(np.prod(r["lattice"])), what=what,
                       text="%s, " + text)
            t = r["times"]["H"]["ms"]
            out["lattice"]["H"] = (t["min"], t["mean"], t["max"])
            for name, v in (r.get("solves") or {}).items():
                t = v["iteration_ms"]
                out["lattice"][name] = (t["min"], t["mean"], t["max"])
        elif "times" in r and "checks" in r:
            return None, "a record of lattice_ranks.py (the halo's and the prior's times): not in this figure"
        else:
            return None, "no scaling record (its keys: %s)" % ", ".join(list(r)[:6])
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return None, "a record of another layout (%s: %s)" % (type(e).__name__, e)
    if not (out["sweeps"] or out["lattice"]):
        return None, "a record without times (a run that was stopped early)"
    return out, None


def _records(paths):
    """The records among ``paths``: files, and the ``.json`` files of folders and of the
    folders in them."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "*.json")) + glob.glob(os.path.join(p, "*", "*.json")))
            files += [f for f in found if os.path.isfile(f)]
        else:
            files.append(p)
    records = []
    for f in files:
        r, why = read_record(f)
        if r is None:
            print("fig_scaling: %s: %s" % (f, why))
            continue
        if r.get("ended", "whole") != "whole":
            # (the driver writes its record as it goes: a run that was stopped leaves what it had)
            print("fig_scaling: %s is the record of a run that did not reach its end (%s): drawn with what "
                  "it has" % (f, r["ended"]))
        records.append(r)
    return records


def _series(records):
    """The weak-scaling series among records: those of one driver with the same block of a
    rank and the same settings, cut alike (columns: a grid ``px, py, 1``; blocks: cut in
    depth too) and exchanging the same way; a record of one rank belongs to every series
    of its driver, block and settings.  Returns ``[(text, records by ranks)]``, the series
    of the most GPUs first, and prints what belongs to none."""
    groups = {}
    for r in records:
        if r["ranks"] > 1:
            kind = "columns" if r["grid"][2] == 1 else "blocks"
            groups.setdefault((r["driver"], r["load"], r["what"], kind, r["exchange"]), []).append(r)
    used, out = set(), []
    for key, rs in groups.items():
        ones = [r for r in records if r["ranks"] == 1 and (r["driver"], r["load"], r["what"]) == key[:3]]
        rs = sorted(ones + rs, key=lambda r: (r["ranks"], r["name"]))
        if len(set(r["ranks"] for r in rs)) < 2:
            continue
        used.update(id(r) for r in rs)
        out.append(("%s: %s%s" % (key[0], rs[-1]["text"] % key[3],
                                  ", exchange through host arrays" if key[4] == "host arrays" else ""), rs))
    for r in records:
        if id(r) not in used:
            print("fig_scaling: %s (job %s, %d rank%s): no record on another number of ranks with its block "
                  "of a rank and its settings; not drawn" % (r["file"], r["job"], r["ranks"],
                                                             "" if r["ranks"] == 1 else "s"))
    out.sort(key=lambda s: (-s[1][-1]["ranks"], -len(s[1]), s[0]))
    return out


def _loads(rs):
    """A series' load of a rank, from its record on the most ranks."""
    r = rs[-1]
    parts = []
    if r.get("unknowns"):
        parts.append("%s wave unknowns" % _short(r["unknowns"] / r["ranks"]))
    if r.get("nodes") and r["lattice"]:
        parts.append("%s lattice nodes" % _short(r["nodes"] / r["ranks"]))
    return ", ".join(parts) + " a GPU"


def fig_scaling(paths, path="fig_scaling.png"):
    """Weak scaling, from records of ``scale.py`` and of the two scripts before it (files, or
    folders of them; :func:`read_record` says which fields of which).  Left: a time step
    of the forward, the incremental and the backward sweep.  Right: a product with ``H~``
    (the slowest rank's time, with a thin bar down to the fastest's: the ranks wait for
    each other) and an iteration of the solve with ``H~ + R``.  Top: milliseconds against
    the number of GPUs.  Bottom: the time on the fewest GPUs of a series over the time, the
    weak-scaling efficiency.

    A series (:func:`_series`) is a line of its own dashes; a job a mark of its own shape.
    Where a series has several records on as many GPUs each is a mark and the line takes
    the fastest.  The axis on top has the unknowns of the series with the most GPUs."""
    plt = _pyplot()
    from matplotlib.ticker import FuncFormatter, LogLocator

    series = _series(_records(paths))
    if not series:
        print("no fig_scaling: no two records of one block of a rank on different numbers of ranks")
        return None
    if len(series) > len(DASHES):
        print("fig_scaling: %d series and %d kinds of line: some are drawn alike; give fewer records for a "
              "figure that reads" % (len(series), len(DASHES)))
    jobs = sorted(set(r["job"] for _, rs in series for r in rs))
    mark = {job: MARKS[n % len(MARKS)] for n, job in enumerate(jobs)}
    sides = (("sweeps", (("forward", "forward sweep"), ("incremental", "incremental sweep"),
                         ("backward", "backward sweep")),
              "a time step (ms)", "unknowns", "wave unknowns"),
             ("lattice", (("H", "a product with %s" % H_TILDE),
                          ("prior", "an iteration of the solve with %s + R, the prior preconditioning"
                           % H_TILDE),
                          ("symbol", "an iteration, the symbol-space inverse preconditioning"),
                          ("split", "an iteration, the split form preconditioning"),
                          ("none", "an iteration, no preconditioner")),
              "a product, an iteration (ms):\nthe slowest rank's", "nodes", "nodes of the lattice"))
    most = max(r["ranks"] for _, rs in series for r in rs)
    largest = [r for _, rs in series for r in rs if r["ranks"] == most]
    sizes = [text % _count(max(found)) for text, found in (
        ("%s wave unknowns", [r["unknowns"] for r in largest if r["sweeps"] and r.get("unknowns")]),
        ("%s nodes of the lattice", [r["nodes"] for r in largest if r["lattice"] and r.get("nodes")]))
        if found]
    count = sum(len(rs) for _, rs in series)
    lines = ["Weak scaling of the Marmousi application's kernels: a rank's load stays, the box grows with "
             "the GPUs", "%d record%s of %d job%s; on %d GPUs up to %s"
             % (count, "" if count == 1 else "s", len(jobs), "" if len(jobs) == 1 else "s", most,
                " and ".join(sizes))]
    width, left = 12.0, 0.8
    head = _head_height(lines, width, left)
    # the three legends underneath: what a hue is (a row for the sweeps, a row a quantity of the lattice),
    # what a kind of line is, what a mark is
    drawn = set(name for _, rs in series for r in rs for name in r["lattice"])
    rows = (max(1, len(drawn)), len(series), -(-len(jobs) // 8))
    band = 0.25 + 0.2 * sum(rows) + 0.24
    height = head + 0.65 + 5.3 + 0.5 + band
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 2, figsize=(width, height), sharex="col")
        fig.subplots_adjust(left=left / width, right=0.985, top=1.0 - (head + 0.65) / height,
                            bottom=(band + 0.5) / height, wspace=0.2, hspace=0.1)
        _head(fig, lines, left)
        every = set()
        for c, (group, quantities, ylabel, size, size_name) in enumerate(sides):
            top, low = axs[0, c], axs[1, c]
            mine = [(n, [r for r in rs if r[group]]) for n, (_, rs) in enumerate(series)
                    if any(r[group] for r in rs)]
            shown, lead = {}, None
            for n, rs in mine:
                dash = DASHES[n % len(DASHES)]
                if lead is None and all(r.get(size) for r in rs):
                    lead = (n, rs)
                for q, (name, label) in enumerate(quantities):
                    pts = [(r["ranks"], r[group][name], r) for r in rs if name in r[group]]
                    if len(set(pt[0] for pt in pts)) < 2:
                        continue
                    hue = HUES[q]
                    # a sweep's time is one number; a product's the slowest rank's of least, mean, greatest
                    value = (lambda v: v[0]) if group == "sweeps" else (lambda v: v[2])
                    best = {}
                    for ranks, v, r in pts:
                        best[ranks] = min(best.get(ranks, np.inf), value(v))
                    xs = sorted(best)
                    ys = np.array([best[x] for x in xs])
                    top.plot(xs, ys, color=hue, ls=dash, lw=1.6, zorder=2)
                    low.plot(xs, ys[0] / ys, color=hue, ls=dash, lw=1.6, zorder=2)
                    for ranks, v, r in pts:
                        every.add(ranks)
                        y = value(v)
                        if group == "lattice" and v[0] < v[2]:
                            top.plot([ranks, ranks], [v[0], v[2]], color=hue, lw=0.7, zorder=1)
                        look = dict(color=hue, ls="", marker=mark[r["job"]], ms=6.0, mec=SURFACE, mew=0.9,
                                    zorder=3)
                        top.plot([ranks], [y], **look)
                        low.plot([ranks], [ys[0] / y], **look)
                        if group == "lattice" and v[2] > 1.5 * v[1]:
                            # a rank far behind the others: said, with what the ranks took on average
                            top.annotate("one rank %.0f ms,\nthe ranks' mean %.0f" % (v[2], v[1]), (ranks, y),
                                         textcoords="offset points", xytext=(-8, 0), ha="right", va="center",
                                         fontsize=7, color=INK2, linespacing=1.2)
                    if name not in shown:
                        fields = pts[0][1][1] if group == "sweeps" else 0
                        shown[name] = plt.Line2D([], [], color=hue, lw=1.6, label=label + (
                            ", %d field%s" % (fields, "" if fields == 1 else "s") if fields else ""))
            top.set_yscale("log")
            top.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
            top.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            top.margins(y=0.12)
            top.set_ylabel(ylabel)
            low.set_ylabel("weak-scaling efficiency:\ntime on the fewest GPUs of a series / time")
            low.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
            low.set_ylim(0.0, 1.12)
            low.set_xlabel("GPUs (a rank each)")
            for ax in (top, low):
                ax.set_xscale("log", base=2)
                ax.grid(True, which="major")
            if shown:
                x0 = (top.get_position().x0 * width - 0.04) / width
                fig.legend(handles=list(shown.values()), loc="upper left", fontsize=7.5, borderaxespad=0.0,
                           bbox_to_anchor=(x0, (band - 0.02) / height), ncol=3 if group == "sweeps" else 1,
                           columnspacing=1.4, handlelength=2.2)
            if lead is not None:
                by = {}
                for r in lead[1]:
                    by.setdefault(r["ranks"], r[size])
                sec = top.secondary_xaxis("top")
                sec.set_xscale("log", base=2)
                sec.set_xticks(sorted(by))
                sec.set_xticklabels([_short(by[x]) for x in sorted(by)])
                sec.minorticks_off()
                sec.tick_params(length=2.5, colors=MUTED, labelcolor=INK2)
                sec.spines["top"].set_visible(True)
                sec.spines["top"].set_color(AXIS)
                if len(mine) > 1:
                    size_name += " (of the series drawn with the %s line)" % DASH_NAMES[lead[0] % len(DASHES)]
                sec.set_xlabel(size_name)
        ticks = sorted(every)
        for ax in axs.ravel():
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
        kinds = [plt.Line2D([], [], color=INK2, lw=1.6, ls=DASHES[n % len(DASHES)],
                            label="%s; %s" % (text, _loads(rs))) for n, (text, rs) in enumerate(series)]
        marks = [plt.Line2D([], [], color=INK2, ls="", marker=mark[job], ms=6.0, mec=SURFACE, mew=0.9,
                            label="job %s" % job) for job in jobs]
        y = band - 0.02 - 0.2 * rows[0] - 0.12
        x0 = (left - 0.04) / width
        fig.legend(handles=kinds, loc="upper left", bbox_to_anchor=(x0, y / height), ncol=1, fontsize=7.5,
                   handlelength=4.6, borderaxespad=0.0)
        y -= 0.2 * rows[1] + 0.12
        fig.legend(handles=marks, loc="upper left", bbox_to_anchor=(x0, y / height), ncol=8, fontsize=7.5,
                   borderaxespad=0.0, columnspacing=1.6, handletextpad=0.3)
        return _save(fig, path)


# ----------------------------------------------------------------------------------- main
FIGURES = ("overview", "psf", "convergence", "posterior", "chain")


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", metavar="PATH",
                   help="the folder of a run of run.py; with --scaling, records and folders of records")
    p.add_argument("--scaling", action="store_true",
                   help="draw the weak scaling of the records given, and nothing else")
    p.add_argument("-o", "--output", default=None,
                   help="where the figures go: a folder (default: the run's own); with --scaling a file "
                        "(default: fig_scaling.png)")
    p.add_argument("--tag", default=None,
                   help="the MAP solve whose point, samples and chains are drawn: its tag, or its "
                        "preconditioner (default: psf, then prior, then none)")
    p.add_argument("--key", default=None,
                   help="the sample lattice of fig_psf.png, as the driver names it (s16r8t1); by default "
                        "the one of the solve drawn, and the others on file as fig_psf_<key>.png")
    p.add_argument("--depth", type=float, default=None,
                   help="depth of the overview's map, m (default: the middle of where the window is one)")
    p.add_argument("--only", default=None, help="of " + ", ".join(FIGURES) + ", with commas (default: all)")
    a = p.parse_args(argv)
    if a.scaling:
        made = fig_scaling(a.paths, a.output or "fig_scaling.png")
        if made:
            print("wrote", made)
        return 0 if made else 1
    if len(a.paths) != 1:
        p.error("one run at a time (or records with --scaling)")
    only = FIGURES if a.only is None else tuple(s for s in a.only.split(",") if s)
    if set(only) - set(FIGURES):
        p.error("--only is a list of " + ", ".join(FIGURES))
    F = Folder(a.paths[0])
    out = a.output or F.out
    made = []
    if "overview" in only:
        made.append(fig_overview(F, a.tag, a.depth, os.path.join(out, "fig_overview.png")))
    if "psf" in only:
        keys, found = F.keys(), F.solve(a.tag)
        first = a.key or (found[1].get("psf") if found and found[1].get("psf") in keys else None)
        if first is None and keys:
            first = keys[0]
        made.append(fig_psf(F, first, os.path.join(out, "fig_psf.png")))
        if a.key is None:
            made += [fig_psf(F, key, os.path.join(out, "fig_psf_%s.png" % key))
                     for key in keys if key != first]
    if "convergence" in only:
        made.append(fig_convergence(F, os.path.join(out, "fig_convergence.png")))
    if "posterior" in only:
        made.append(fig_posterior(F, a.tag, os.path.join(out, "fig_posterior.png")))
    if "chain" in only:
        made.append(fig_chain(F, a.tag, os.path.join(out, "fig_chain.png")))
    print("wrote %s in %s" % (", ".join(os.path.basename(m) for m in made if m) or "nothing", out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

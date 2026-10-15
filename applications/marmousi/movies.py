"""Pictures of the example in three dimensions: the model on three planes of the box, a turn
around it, and the wave field of one shot as it crosses the box.

Two steps, as for ``figures.py``:

    python movies.py record OUT [run.py's options of the problem] [--shot K] [--every N] [--spacing M]
    python movies.py draw OUT [--folder RUN_FOLDER --tag TAG]

``record`` needs the library and a GPU (or the host): it builds the problem of ``run.py``'s
options (the defaults are the paper's box), sets the speed of the data (``c0 + W q_true``, or
``c_true`` with ``--full-truth``), fires shot ``--shot`` (the one nearest the middle of the box
by default), and keeps the pressure every ``--every`` steps at points ``--spacing`` metres apart
on three planes through the source: the section along the line (``y`` of the source), the
section across it (``x`` of the source) and a map at ``--map-depth``.  It also keeps the true
and the initial speed on the same planes and on the lattice, in ``OUT/movie.npz``.  One GPU.

``draw`` needs NumPy, Matplotlib and Pillow only: ``OUT/model3d.png`` (the true and the initial
speed on the three planes in perspective; with ``--folder`` and ``--tag`` also the MAP point of
that run folder of ``run.py``), ``OUT/turn.webp`` (a turn around the true model) and
``OUT/wavefield.webp`` (the shot), animated WebP (Pillow; any browser plays them; ``--gif`` for
GIF as well).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def planes(box, at, spacing, depth):
    """Points of the three planes through ``at`` (x, y), ``spacing`` apart, inside the box:
    ``{name: (points (n, 3), shape)}``."""
    o, s = np.asarray(box.origin, dtype=np.float64), np.asarray(box.size, dtype=np.float64)
    eps = 1e-6 * spacing
    axis = [np.arange(o[k], o[k] + s[k] + eps, spacing) for k in range(3)]
    axis = [np.clip(a, o[k] + eps, o[k] + s[k] - eps) for k, a in enumerate(axis)]
    x, y, z = axis
    out = {}
    X, Z = np.meshgrid(x, z, indexing="ij")
    out["along"] = (np.stack([X, np.full_like(X, at[1]), Z], -1).reshape(-1, 3), X.shape)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    out["across"] = (np.stack([np.full_like(Y, at[0]), Y, Z], -1).reshape(-1, 3), Y.shape)
    X, Y = np.meshgrid(x, y, indexing="ij")
    out["map"] = (np.stack([X, Y, np.full_like(X, depth)], -1).reshape(-1, 3), X.shape)
    return out, axis


def record(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    ap.add_argument("out")
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--every", type=int, default=20, help="steps between the snapshots")
    ap.add_argument("--spacing", type=float, default=25.0, help="metres between the points of a plane")
    ap.add_argument("--map-depth", type=float, default=800.0, help="depth of the map, metres")
    mine, rest = ap.parse_known_args(argv)
    sys.path.insert(0, HERE)
    import hippymfem as hm                               # noqa: F401  (first: it sets the device up)
    from mpi4py import MPI
    import jax
    import jax.numpy as jnp
    import run as runmod
    from model import Problem, LocalPool

    t_start = time.time()
    args = runmod.parse(rest + ["--out", mine.out])
    P = Problem(args)
    comm = MPI.COMM_SELF
    pool = LocalPool(comm, P)
    lat = pool.lat
    F = P.fields(lat, 0)
    q = F.q_true if args.full_truth else F.window * F.q_true
    pool._point(q)
    eng, col = pool.eng, pool.col
    src = np.asarray(P.sources)
    mid = np.asarray(P.box.origin) + 0.5 * np.asarray(P.box.size)
    shot = int(np.argmin(np.hypot(src[:, 0] - mid[0], src[:, 1] - mid[1]))) if mine.shot is None else mine.shot
    at = src[P.shots[shot][0]]
    eng.set_wavelets(P.wavelets(shot))
    pts, axis = planes(P.box, at, mine.spacing, mine.map_depth)
    ops = {}
    for name, (p, shape) in pts.items():
        op = P.box.value_operator(col, p, comm, P.order)
        if op.index.size != p.shape[0]:
            raise SystemExit("movies: %d of the %d points of the plane %s were not found" % (p.shape[0] - op.index.size,
                                                                                         p.shape[0], name))
        order = np.argsort(op.index)
        ops[name] = (jax.device_put(jnp.asarray(op.w[order], dtype=eng.dtype), eng.device),
                     jax.device_put(jnp.asarray(op.dof[order], dtype=jnp.int32), eng.device), shape)
    print("movies: %s, shot %d at (%.0f, %.0f, %.0f) m; %s points on three planes; set-up %.1f s"
          % ("x".join(str(int(v)) for v in P.box.n), shot, *at, sum(p.shape[0] for p, _ in pts.values()),
             time.time() - t_start), flush=True)
    value = jax.jit(lambda U, w, dof: jnp.sum(w * U[dof, 0], axis=1))
    zero = lambda: jax.device_put(jnp.zeros((eng.nloc, 1), dtype=eng.dtype), eng.device)  # noqa: E731
    U, Um = zero(), zero()
    frames = {name: [] for name in ops}
    times, n = [], 0
    t0 = time.time()
    while n < P.nsteps:
        m = min(mine.every, P.nsteps - n)
        U, Um, _, _ = eng._run(U, Um, range(n, n + m))
        n += m
        for name, (w, dof, shape) in ops.items():
            frames[name].append(np.asarray(value(U, w, dof)).reshape(shape))
        times.append(n * P.dt)
    print("movies: %d steps of %.3f ms, %d snapshots in %.1f s" % (P.nsteps, 1e3 * P.dt, len(times), time.time() - t0),
          flush=True)
    # the speeds on the planes: the lattice's trilinear interpolant at the points
    from scipy.interpolate import RegularGridInterpolator
    x, y, z = F.x, F.y, F.z
    speed = {}
    for label, c in (("true", F.c0 + q), ("initial", F.c0)):
        f = RegularGridInterpolator((z, y, x), c)
        speed[label] = {name: f(p[:, ::-1]).reshape(shape) for name, (p, shape) in pts.items()}
    os.makedirs(mine.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(mine.out, "movie.npz"), t=np.array(times), source=at, x=axis[0], y=axis[1], z=axis[2],
        map_depth=mine.map_depth, box_origin=np.asarray(P.box.origin), box_size=np.asarray(P.box.size),
        lattice_x=x, lattice_y=y, lattice_z=z, lattice_true=F.c0 + q, lattice_initial=F.c0, window=F.window,
        **{"field_" + k: np.stack(v).astype(np.float32) for k, v in frames.items()},
        **{"%s_%s" % (lab, k): v for lab, d in speed.items() for k, v in d.items()})
    print("movies: wrote %s in %.1f s" % (os.path.join(mine.out, "movie.npz"), time.time() - t_start))


# ------------------------------------------------------------------------- drawing
def _pieces(a, b, ia, ib):
    """The four parts of a plane on the grid ``(a, b)`` cut at the indices ``ia`` and ``ib`` (both
    parts keep the cut line): ``[(slice_a, slice_b)]``."""
    return [(sa, sb) for sa in (slice(0, ia + 1), slice(ia, len(a))) for sb in (slice(0, ib + 1), slice(ib, len(b)))
            if sa.stop - sa.start > 1 and sb.stop - sb.start > 1]


def _fence(ax, d, values, cmap, norm, alpha=1.0):
    """The three planes of ``d`` (``movie.npz``) coloured by ``values`` ({plane: array}), in
    kilometres, depth down.  Each plane is cut in four along its lines with the other two, so
    that Matplotlib, which orders whole surfaces by their mean depth, draws them back to
    front."""
    x, y, z = d["x"] / 1e3, d["y"] / 1e3, d["z"] / 1e3
    sx, sy = d["source"][0] / 1e3, d["source"][1] / 1e3
    md = float(d["map_depth"]) / 1e3
    ix, iy, iz = (int(np.argmin(np.abs(v - w))) for v, w in ((x, sx), (y, sy), (z, md)))
    kw = dict(rstride=1, cstride=1, shade=False, antialiased=False, linewidth=0, alpha=alpha)
    V = values["along"]
    for sa, sb in _pieces(x, z, ix, iz):
        X, Z = np.meshgrid(x[sa], z[sb], indexing="ij")
        ax.plot_surface(X, np.full_like(X, y[iy]), Z, facecolors=cmap(norm(V[sa, sb])), **kw)
    V = values["across"]
    for sa, sb in _pieces(y, z, iy, iz):
        Y, Z = np.meshgrid(y[sa], z[sb], indexing="ij")
        ax.plot_surface(np.full_like(Y, x[ix]), Y, Z, facecolors=cmap(norm(V[sa, sb])), **kw)
    V = values["map"]
    for sa, sb in _pieces(x, y, ix, iy):
        X, Y = np.meshgrid(x[sa], y[sb], indexing="ij")
        ax.plot_surface(X, Y, np.full_like(X, z[iz]), facecolors=cmap(norm(V[sa, sb])), **kw)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_zlim(z[-1], z[0])
    ax.set_box_aspect((x[-1] - x[0], y[-1] - y[0], 1.5 * (z[-1] - z[0])), zoom=1.15)
    ax.set_yticks(np.arange(np.ceil(y[0]), y[-1] + 1e-9, 1.0))
    ax.set_zticks(np.arange(0.0, z[-1] + 1e-9, 1.0))
    ax.set_xlabel("x (km)", labelpad=6)
    ax.set_ylabel("y (km)", labelpad=2)
    ax.set_zlabel("depth (km)", labelpad=2)


def _coarse(values, step):
    return {k: v[::step, ::step] for k, v in values.items()}


def _save_animation(frames, path, ms, gif=False):
    from PIL import Image
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, lossless=False, quality=85)
    if gif:
        frames[0].save(path.rsplit(".", 1)[0] + ".gif", save_all=True, append_images=frames[1:], duration=ms, loop=0)


def _canvas_image(fig):
    from PIL import Image
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def draw(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    ap.add_argument("out")
    ap.add_argument("--folder", default=None, help="a run folder of run.py on the same box, for its MAP point")
    ap.add_argument("--tag", default=None, help="the MAP solve's tag in that folder")
    ap.add_argument("--step", type=int, default=2, help="every step-th point of a plane is drawn")
    ap.add_argument("--frames", type=int, default=72, help="frames of the turn")
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    d = dict(np.load(os.path.join(a.out, "movie.npz")))
    sp = d["box_size"] / 1e3
    st = a.step
    cm = plt.get_cmap("viridis")
    true = _coarse({k: d["true_" + k] for k in ("along", "across", "map")}, st)
    init = _coarse({k: d["initial_" + k] for k in ("along", "across", "map")}, st)
    norm = colors.Normalize(1500.0, 4000.0)
    dd = dict(d)
    for k in ("x", "y", "z"):
        dd[k] = d[k][::st]
    title = "Marmousi in three dimensions, %.1f x %.1f x %.1f km" % tuple(sp)
    # ---- the model, in perspective
    panels = [("true speed (of the data)", true), ("initial speed", init)]
    if a.folder and a.tag:
        from scipy.interpolate import RegularGridInterpolator
        m = np.load(os.path.join(a.folder, "map_%s.npy" % a.tag))
        W = np.load(os.path.join(a.folder, "window.npy"))
        c0 = np.load(os.path.join(a.folder, "initial.npy"))
        est = c0 + W * m
        lz, ly, lx = (np.linspace(o, o + s, n) for o, s, n in zip(d["box_origin"][::-1], d["box_size"][::-1], est.shape))
        if est.shape != d["lattice_true"].shape:
            raise SystemExit("movies: the folder's lattice %s is not the movie's %s" % (est.shape, d["lattice_true"].shape))
        f = RegularGridInterpolator((lz, ly, lx), est)
        pts = {"along": np.stack(np.meshgrid(d["x"], [d["source"][1]], d["z"], indexing="ij"), -1)[:, 0],
               "across": np.stack(np.meshgrid([d["source"][0]], d["y"], d["z"], indexing="ij"), -1)[0],
               "map": np.stack(np.meshgrid(d["x"], d["y"], [d["map_depth"]], indexing="ij"), -1)[:, :, 0]}
        panels.append(("MAP point (%s)" % a.tag, _coarse({k: f(v[..., ::-1]) for k, v in pts.items()}, st)))
    fig = plt.figure(figsize=(6.0 * len(panels), 5.2))
    for i, (label, vals) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        _fence(ax, dd, vals, cm, norm)
        ax.view_init(elev=22, azim=-58)
        ax.set_title(label)
    fig.subplots_adjust(left=0.0, right=0.9, top=0.9, bottom=0.08, wspace=0.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    fig.colorbar(sm, cax=fig.add_axes([0.92, 0.25, 0.012, 0.5]), label="wave speed (m/s)")
    fig.suptitle(title + ": the section along the line through the source, the section across it, and a map %.1f km "
                 "down" % (float(d["map_depth"]) / 1e3), fontsize=10)
    fig.savefig(os.path.join(a.out, "model3d.png"), dpi=150)
    plt.close(fig)
    print("movies: wrote model3d.png", flush=True)
    # ---- a turn around the true model
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection="3d")
    imgs = []
    for k in range(a.frames):
        ax.clear()
        _fence(ax, dd, true, cm, norm)
        ax.view_init(elev=22, azim=-58 + 360.0 * k / a.frames)
        ax.set_title(title + ", true speed (1500 to 4000 m/s)", fontsize=10)
        imgs.append(_canvas_image(fig))
    plt.close(fig)
    _save_animation(imgs, os.path.join(a.out, "turn.webp"), 80, a.gif)
    print("movies: wrote turn.webp (%d frames)" % len(imgs), flush=True)
    # ---- the shot
    fields = {k: d["field_" + k] for k in ("along", "across", "map")}
    seis = plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection="3d")
    imgs = []
    for i, t in enumerate(d["t"]):
        ax.clear()
        top = max(float(np.percentile(np.abs(v[i]), 99.5)) for v in fields.values()) or 1.0
        wn = colors.Normalize(-0.5 * top, 0.5 * top)
        _fence(ax, dd, _coarse({k: v[i] for k, v in fields.items()}, st), seis, wn)
        ax.scatter(*(d["source"][:2] / 1e3), d["source"][2] / 1e3, color="k", s=12)
        ax.view_init(elev=22, azim=-58)
        ax.set_title("one shot through the true model, t = %.2f s\n(pressure; each frame scaled to its own range)" % t,
                     fontsize=9)
        imgs.append(_canvas_image(fig))
    plt.close(fig)
    _save_animation(imgs, os.path.join(a.out, "wavefield.webp"), 60, a.gif)
    print("movies: wrote wavefield.webp (%d frames)" % len(imgs), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "draw"):
        sys.exit(__doc__)
    (record if sys.argv[1] == "record" else draw)(sys.argv[2:])

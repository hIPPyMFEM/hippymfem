#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The geothermal inversion as an animation.

    python -m applications.geothermal.movie results/fields_n128.npz --out results/animations

The block of rock with a quarter cut away, four times:

* the true log conductivity, with the heat flowing through it: streamlines of the heat
  flux ``-k grad T`` from the basal flux to the surface, drawn as trails moving along
  them, bending into the buried conductive body, and the isotherms on the faces;
* the MAP point from the borehole logs;
* the posterior in motion: a closed path through exact samples of the Laplace
  posterior, ``m_MAP + sum_j (cos(2 pi f_j t) xi_j + sin(2 pi f_j t) xi_j') / sqrt(J)``
  with independent zero-mean posterior samples ``xi``, so that every frame is a sample;
  where the logs constrain the rock it hardly moves, elsewhere it changes as the prior;
* the posterior standard deviation.

The fields are those of ``movie_data.py fields``, which makes them from the dump of a
run.  A ``run.py --dump`` file can be given as it is: it has neither a temperature nor
samples, so the rock is then shown without the heat, and the truth minus the MAP point
in the place of the samples.  ``--temperature`` takes the true temperature from a
forward solve on a finer mesh (``movie_data.py forward``).  Written are an MP4, an
animated WebP for a web page and one frame as a PNG, for a light and for a dark page.
The film of the repository's README is

    python -m applications.geothermal.movie fields.npz --temperature forward.npz --name geothermal_turn \\
        --reveal 0.7 1.3 --turn 2 --hardware "<the forward solve's GPUs and time>" "<the inversion's>" \\
        --say "<a sentence>" "<another>" ...

This script needs neither MFEM nor JAX: NumPy, Matplotlib, Pillow and PyVista, and ffmpeg
for the MP4 (the one on the path, or the package ``imageio-ffmpeg``).  PyVista draws off
screen, on a GPU or in software; on a node with neither a display nor a GPU that VTK can
draw on, the wheel ``vtk-osmesa`` in the place of ``vtk`` draws in software:
``pip install --extra-index-url https://wheels.vtk.org vtk-osmesa``.
"""
import argparse
import inspect
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from movie_common import THEMES, encode, framing, hero_row, schedule, view_axes, with_sentence  # noqa: E402

L, H = 8.0, 4.0                                  # km


def block_faces(n, cut=0.5):
    """The visible faces of the unit cube with the quarter ``x > cut, y < cut`` removed,
    as ``(name, index arrays (i, j, k) of shape (a, b))`` on the vertex grid ``(n+1)^3``."""
    h = int(round(cut * n))
    I, J, K = np.arange(n + 1), np.arange(n + 1), np.arange(n + 1)
    out = []

    def grid(a, b):
        return np.meshgrid(a, b, indexing="ij")
    ii, jj = grid(I[:h + 1], J)
    out.append(("top", ii, jj, np.full_like(ii, n)))
    ii, jj = grid(I[h:], J[h:])
    out.append(("top", ii, jj, np.full_like(ii, n)))
    ii, kk = grid(I[:h + 1], K)
    out.append(("front", ii, np.zeros_like(ii), kk))
    jj, kk = grid(J[h:], K)
    out.append(("right", np.full_like(jj, n), jj, kk))
    jj, kk = grid(J[:h + 1], K)
    out.append(("inner_x", np.full_like(jj, h), jj, kk))
    ii, kk = grid(I[h:], K)
    out.append(("inner_y", ii, np.full_like(ii, h), kk))
    # the two far sides, which a camera that goes around the block comes to see
    jj, kk = grid(J, K)
    out.append(("left", np.zeros_like(jj), jj, kk))
    ii, kk = grid(I, K)
    out.append(("back", ii, np.full_like(ii, n), kk))
    return out


def heat_flux_lines(u, k, n, nseed=4, cut=0.5):
    """Streamlines of the heat flux ``-k grad T`` through the true rock, from seeds near
    the base of the removed quarter, with their arc length.  ``u`` and ``k`` are the
    temperature and the conductivity on the vertex grid."""
    import pyvista as pv

    ut = u.reshape(n + 1, n + 1, n + 1)                                   # (k, j, i)
    dTz, dTy, dTx = np.gradient(ut, 1.0 / n)
    # flux in kilometres: the unit cube's x and y are L km, z is H km
    q = -k.reshape(ut.shape)[..., None] * np.stack((dTx / L, dTy / L, dTz / H), axis=-1)
    img = pv.ImageData(dimensions=(n + 1, n + 1, n + 1), spacing=(L / n, L / n, H / n))
    img.point_data["q"] = q.reshape(-1, 3)
    s = np.linspace(cut + 0.06, 0.94, nseed)
    t = np.linspace(0.06, cut - 0.06, nseed)
    SX, SY = np.meshgrid(L * s, L * t)
    seeds = pv.PolyData(np.column_stack((SX.ravel(), SY.ravel(), np.full(SX.size, 0.03 * H))))
    # a line ends where it leaves the block, not at a length: the argument for that was renamed in PyVista 0.46
    far = "max_length" if "max_length" in inspect.signature(img.streamlines_from_source).parameters else "max_time"
    lines = img.streamlines_from_source(seeds, vectors="q", integration_direction="forward",
                                        initial_step_length=0.2, max_step_length=0.5,
                                        terminal_speed=1e-12, max_steps=4000, **{far: 1e6})
    return lines.compute_arc_length()


class Picture:
    """One view of the cut block; the face colours, the trails and the decorations are
    set per frame."""

    def __init__(self, n, theme, size, scale, boreholes=None, well_bottom=0.3, lines=None, cut=0.5):
        import pyvista as pv

        self.n, self.th, self.size, self.scale = n, THEMES[theme], size, scale
        self.faces = []
        pl = pv.Plotter(off_screen=True, window_size=(size * scale, size * scale), lighting="none")
        pl.set_background(self.th["bg"])
        for name, ii, jj, kk in block_faces(n, cut):
            if min(ii.shape) < 2:
                continue                                     # no quarter removed (cut = 1): a face of no width
            mesh = pv.StructuredGrid(L * ii / n, L * jj / n, H * kk / n)
            mesh.point_data["rgb"] = np.zeros((mesh.n_points, 3), dtype=np.uint8)
            idx = ((kk * (n + 1) + jj) * (n + 1) + ii).ravel(order="F")
            pl.add_mesh(mesh, scalars="rgb", rgb=True, ambient=0.78, diffuse=0.30, specular=0.0)
            self.faces.append((name, mesh, idx))
        c = cut
        # the block's edges
        pl.add_mesh(self._segments([((0, 0, 1), (c, 0, 1)), ((c, 0, 1), (c, c, 1)), ((c, c, 1), (1, c, 1)),
                                    ((1, c, 1), (1, 1, 1)), ((1, 1, 1), (0, 1, 1)), ((0, 1, 1), (0, 0, 1)),
                                    ((0, 0, 0), (c, 0, 0)), ((c, 0, 0), (c, c, 0)), ((c, c, 0), (1, c, 0)),
                                    ((1, c, 0), (1, 1, 0)), ((0, 0, 0), (0, 0, 1)), ((c, 0, 0), (c, 0, 1)),
                                    ((c, c, 0), (c, c, 1)), ((1, c, 0), (1, c, 1)), ((1, 1, 0), (1, 1, 1))]),
                    color=self.th["line"], line_width=1.6 * scale, lighting=False)
        # the removed quarter, faintly, so that the space the trails rise in reads as such
        pl.add_mesh(self._segments([((c, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, c, 0)), ((c, 0, 1), (1, 0, 1)),
                                    ((1, 0, 1), (1, c, 1)), ((1, 0, 0), (1, 0, 1))]),
                    color=self.th["line"], opacity=0.35, line_width=1.0 * scale, lighting=False)
        # the wellheads on the surface (over the removed quarter too, where the surface
        # was); the wells themselves are drawn over the picture, see ``shot``
        self.wells, self.well_bottom = boreholes, well_bottom
        if boreholes is not None:
            top = np.column_stack((L * boreholes[:, 0], L * boreholes[:, 1], np.full(len(boreholes), H * 1.004)))
            pl.add_mesh(pv.PolyData(top).glyph(geom=pv.Sphere(radius=0.075, theta_resolution=12, phi_resolution=12),
                                               scale=False, orient=False),
                        color=self.th["receiver"], ambient=0.8, diffuse=0.3)
        self.lines = lines
        self.trail_actor = self.iso_actor = None
        if lines is not None:
            pl.add_mesh(lines, color=self.th["line"], opacity=0.22, line_width=1.0 * scale, lighting=False)
        pl.add_light(pv.Light(light_type="camera light", position=(-0.6, 0.9, 1.0), intensity=1.0))
        pl.add_light(pv.Light(light_type="headlight", intensity=0.25))
        self.pl = pl

    @staticmethod
    def _segments(segs):
        import pyvista as pv

        pts, cells = [], []
        for a, b in segs:
            cells.append([2, len(pts), len(pts) + 1])
            pts += [(L * a[0], L * a[1], H * a[2]), (L * b[0], L * b[1], H * b[2])]
        return pv.PolyData(np.asarray(pts, float), lines=np.asarray(cells).ravel())

    def frame(self, azimuths, elevation, box):
        """Fix the focal point and the distance of the camera: the block, with the quarter
        that was removed, fills ``box`` of the picture and stays in it as the camera swings."""
        corners = np.array([[x, y, zz] for x in (0.0, L) for y in (0.0, L) for zz in (0.0, H)])
        self.focus, self.distance, self.elevation = *framing(corners, azimuths, elevation, box), elevation

    def camera(self, azimuth):
        d = view_axes(azimuth, self.elevation)[0]
        cam = self.pl.camera
        cam.position, cam.focal_point, cam.up, cam.view_angle = tuple(self.focus + self.distance * d), tuple(self.focus), (0, 0, 1), 20.0
        self.pl.reset_camera_clipping_range()

    def colours(self, rgb_of_field, field):
        for name, mesh, idx in self.faces:
            mesh.point_data["rgb"] = (255.0 * np.clip(rgb_of_field(field[idx]), 0.0, 1.0)).astype(np.uint8)

    def isotherms(self, u, levels):
        """Contour lines of the temperature on the faces."""
        parts = []
        for name, mesh, idx in self.faces:
            m = mesh.copy()
            m.point_data["u"] = u[idx]
            c = m.contour(isosurfaces=list(levels), scalars="u")
            if c.n_points:
                parts.append(c)
        if parts:
            iso = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
            self.iso_actor = self.pl.add_mesh(iso, color=self.th["line"], line_width=1.0 * self.scale,
                                              opacity=0.7, lighting=False)

    def trail_frame(self, t, wavelength=2.6, width=0.38, speed=2):
        """The bright part of every streamline at time ``t`` in [0, 1): pieces where
        ``frac(s / wavelength - speed t) < width``, as tubes."""
        import pyvista as pv

        if self.lines is None:
            return
        if self.trail_actor is not None:
            self.pl.remove_actor(self.trail_actor)
            self.trail_actor = None
        s = self.lines.point_data["arc_length"]
        keep = np.mod(s / wavelength - speed * t, 1.0) < width
        lines = self.lines.lines.reshape(-1)
        pts, cells, a, pos = [], [], 0, 0
        while pos < lines.size:
            m = lines[pos]
            ids = lines[pos + 1:pos + 1 + m]
            pos += m + 1
            # runs of kept points along this line
            edges = np.flatnonzero(np.diff(np.concatenate(([0], keep[ids].astype(int), [0]))))
            for b0, b1 in zip(edges[0::2], edges[1::2]):
                if b1 - b0 < 2:
                    continue
                run = ids[b0:b1]
                cells.append(np.concatenate(([run.size], np.arange(a, a + run.size))))
                pts.append(self.lines.points[run])
                a += run.size
        if not pts:
            return
        tube = pv.PolyData(np.concatenate(pts), lines=np.concatenate(cells)).tube(radius=0.035, n_sides=10)
        self.trail_actor = self.pl.add_mesh(tube, color=self.th["source"], ambient=0.65, diffuse=0.5, specular=0.3)

    def _pixels(self, pts):
        """Pixel positions (origin at the top left) of points in space under the present camera."""
        ren, h = self.pl.renderer, self.size * self.scale
        out = np.empty((len(pts), 2))
        for a, (x, y, zz) in enumerate(pts):
            ren.SetWorldPoint(float(x), float(y), float(zz), 1.0)
            ren.WorldToDisplay()
            px, py, _ = ren.GetDisplayPoint()
            out[a] = px, h - 1 - py
        return out

    def shot(self, alpha=0.55):
        """The picture.  Every well is drawn over it, straight down from its wellhead to
        its deepest log, through the block."""
        from PIL import Image, ImageDraw

        self.pl.render()
        img = Image.fromarray(self.pl.screenshot(None, return_img=True))
        if self.wells is not None:
            xy = self.wells
            top = self._pixels(np.column_stack((L * xy[:, 0], L * xy[:, 1], np.full(len(xy), H))))
            bot = self._pixels(np.column_stack((L * xy[:, 0], L * xy[:, 1], np.full(len(xy), self.well_bottom * H))))
            over = Image.new("RGBA", img.size, (0, 0, 0, 0))
            pen = ImageDraw.Draw(over)
            ink = tuple(int(round(255 * c)) for c in self.th["receiver"]) + (int(round(255 * alpha)),)
            for a, b in zip(top, bot):
                pen.line([tuple(a), tuple(b)], fill=ink, width=max(1, int(round(1.1 * self.scale))))
            img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
        return img.resize((self.size, self.size), Image.LANCZOS) if self.scale != 1 else img


def words(k):
    """A number of unknowns, as it is written under the pictures."""
    return ("%.2f billion" % (k / 1e9) if k >= 1e9 else "%d million" % round(k / 1e6) if k >= 1e7
            else "%.2f million" % (k / 1e6) if k >= 1e6 else "{:,}".format(int(k)))


def load(args):
    """The fields that are drawn, on the vertex grid (x fastest, then y, then z): those of
    ``movie_data.py fields``, or of a ``run.py --dump`` file; with ``--temperature`` the
    true temperature of a finer mesh in the place of the inversion's own."""
    z = dict(np.load(args.data))
    n = int(z["n"])
    if "xyz" in z:                                       # a dump: its vertices in the order of the grid
        ijk = np.rint(z["xyz"] * n).astype(np.int64)
        idx = (ijk[:, 2] * (n + 1) + ijk[:, 1]) * (n + 1) + ijk[:, 0]
        for k in ("mtrue", "mmap", "std_post"):
            grid = np.empty((n + 1) ** 3)
            grid[idx] = z[k]
            z[k] = grid
        z.update(forward_n=n, forward_dofs=(2 * n + 1) ** 3, T_SCALE=100.0)
    if args.temperature:
        f = np.load(args.temperature)
        if int(f["n"]) != n:
            raise SystemExit("%s is on the %d^3 grid, the inversion on %d^3: movie_data.py forward --onto %d"
                             % (args.temperature, int(f["n"]), n, n))
        if np.abs(f["mtrue"] - z["mtrue"]).max() > 1e-4:
            raise SystemExit("%s is the temperature of another rock than the inversion's" % args.temperature)
        z.update(u_true=f["u_true"], k_true=f["k_true"], forward_n=f["forward_n"], forward_dofs=f["forward_dofs"])
    return z


def render(args):
    import matplotlib
    from PIL import Image

    z = load(args)
    n = int(z["n"])
    mt, mm, sd = (np.asarray(z[k], float) for k in ("mtrue", "mmap", "std_post"))
    targets = np.asarray(z["targets"], float)
    bh = np.unique(np.round(targets[:, :2], 9), axis=0)
    well_bottom = float(targets[:, 2].min())
    d = np.asarray(z["d"], float)
    # the heat needs the true temperature and the motion of the posterior its samples: a dump has neither
    flow, moving = "u_true" in z, "s_post" in z
    ut = np.asarray(z["u_true"], float) if flow else None
    lines = heat_flux_lines(ut, np.asarray(z["k_true"], float), n, nseed=args.seeds) if flow else None
    xi = np.asarray(z["s_post"], float) if moving else np.zeros((0, mm.size))
    nfr = args.frames
    J = min(xi.shape[0] // 2, args.pairs)
    freqs = [1 + (j // 2) for j in range(J)]
    phases = 2 * np.pi * np.random.default_rng(3).random(J)
    vmax, smax = args.vmax, args.std_max
    levels = [v / float(z["T_SCALE"]) for v in args.isotherms]
    print("geothermal movie: %d^3, %d of %d eigenvalues above 1; %s; %s"
          % (n, int((d > 1).sum()), d.size,
             "%d streamlines of the heat flux" % lines.n_lines if flow else "no temperature in the file: the rock without the heat",
             "%d pairs of posterior samples (frequencies %s)" % (J, freqs) if moving
             else "no samples in the file: the truth minus the MAP in their place"), flush=True)
    os.makedirs(args.out, exist_ok=True)
    # the quarter that is removed, shown at first and faded away (--reveal): the same pictures of the
    # whole block laid over the others in an opening that is played once in front of the loop.  The
    # opening is the end of the loop again, so that the motion runs on into it.
    n_hold, n_fade = (int(round(s * args.fps)) for s in args.reveal)
    n_intro = n_hold + n_fade
    # every frame of the film: its place in the loop, its number in the opening (or None), and its
    # place in the film counted from the loop's first frame (the camera's, when it goes around)
    turns = max(args.turn, 1)
    film = [(nfr - n_intro + j, j, j - n_intro) for j in range(n_intro)] + \
           [(i, None, a * nfr + i) for a in range(turns) for i in range(nfr)]
    frames = list(enumerate(film)) if args.only is None else [(k, film[k]) for k in args.only]
    around = lambda place: args.azimuth + 360.0 * place / (turns * nfr)       # once around in --turn loops
    cues = schedule(args.say, len(film) / args.fps)
    # the sizes under the pictures: on the left the unknowns of the state in the forward solve,
    # on the right the parameters that are inferred, and the rest in a line under them
    forward = (args.forward[0] or "%s state unknowns" % words(int(z["forward_dofs"])),
               args.forward[1] or "forward solve on a %d³ mesh" % int(z["forward_n"]), args.hardware[0])
    inversion = (args.inversion[0] or "%s parameters · %d eigenpairs" % (words(mm.size), d.size),
                 args.inversion[1] or "inversion on a %d³ mesh · %s state unknowns" % (n, words((2 * n + 1) ** 3)),
                 args.hardware[1])
    about = args.about or "heat conduction · conductivity of the rock from temperatures logged in %d boreholes" % len(bh)
    for theme in args.themes:
        th = THEMES[theme]
        div = lambda x: matplotlib.colormaps[th["diverging"]](np.clip(x, 0, 1))[:, :3]
        seq = lambda x: matplotlib.colormaps[th["sequential_std"]](np.clip(x, 0, 1))[:, :3]
        field_rgb = lambda v: div(0.5 + 0.5 * v / vmax)
        std_rgb = lambda v: seq(v / smax)
        big, small = 560, 336
        swing = tuple(np.linspace(args.azimuth - args.swing, args.azimuth + args.swing, 9))
        if args.turn:
            swing = tuple(np.linspace(around(-n_intro), around(turns * nfr), 145))   # every direction the camera looks from
        wells = dict(boreholes=bh, well_bottom=well_bottom)
        P0 = Picture(n, theme, big, args.supersample, lines=lines)
        P1, P2, P3 = (Picture(n, theme, small, args.supersample, **wells) for _ in range(3))
        whole = []
        if n_intro:
            whole = [Picture(n, theme, big, args.supersample, cut=1.0)] + \
                    [Picture(n, theme, small, args.supersample, cut=1.0, **wells) for _ in range(3)]
        for a, P in enumerate([P0, P1, P2, P3] + whole):
            k = a % 4                    # the truth, the MAP, the samples (the error where there are none), the std
            P.colours(std_rgb if k == 3 else field_rgb, (mt, mm, mt - mm, sd)[k])
            if k == 0 and flow:
                P.isotherms(ut, levels)
            P.frame(swing, args.elevation, (-0.93, 0.93, -0.93, 0.93))
        tmp = os.path.join(args.out, "frames_%s_%s" % (args.name, theme))
        os.makedirs(tmp, exist_ok=True)
        dticks = [((v + vmax) / (2 * vmax), ("%+.1f" % v).replace("-", "−") if abs(v) > 1e-12 else "0")
                  for v in np.linspace(-vmax, vmax, 5)]
        sticks = [(v / smax, "%.1f" % v) for v in np.arange(0.0, smax + 1e-9, 0.2)]
        t0 = time.time()
        for f, (i, intro, place) in frames:
            t = i / nfr
            az = around(place) if args.turn else args.azimuth + args.swing * np.sin(2 * np.pi * t)
            if moving:
                sample = mm.copy()
                for j in range(J):
                    a = 2 * np.pi * freqs[j] * t + phases[j]
                    sample += (np.cos(a) * xi[2 * j] + np.sin(a) * xi[2 * j + 1]) / np.sqrt(J)
                for P in [P2] + (whole[2:3] if intro is not None else []):
                    P.colours(field_rgb, sample)
            P0.trail_frame(t)
            pics = []
            for P in (P0, P1, P2, P3):
                P.camera(az)
                pics.append(P.shot())
            if intro is not None:
                # how much of the quarter is still there: all of it for the hold, then an eased fall
                s = min(max((intro - n_hold + 1) / (n_fade + 1.0), 0.0), 1.0)
                there = 1.0 - s * s * (3.0 - 2.0 * s)
                for k, P in enumerate(whole):
                    P.camera(az)
                    pics[k] = Image.blend(pics[k], P.shot(), there)
            page = hero_row(theme, "heat flowing through the true rock" if flow else "the true rock", pics[0],
                            "Geothermal reservoir", about,
                            pics[1:], ("MAP estimate", "posterior samples" if moving else "truth − MAP", "posterior std"),
                            ((div, dticks, "log conductivity"), (seq, sticks, "std of log conductivity")),
                            forward, inversion).img
            if cues:
                page = with_sentence(page, theme, cues, f / args.fps)
            page.save(os.path.join(tmp, "f%04d.png" % f))
        name = os.path.join(args.out, "%s_%s" % (args.name, theme))
        print("%s: %d frames in %.1f s" % (name, len(frames), time.time() - t0), flush=True)
        if args.only is None:
            shutil.copy(os.path.join(tmp, "f%04d.png" % (n_intro + nfr * 3 // 10)), name + ".png")
            written = encode(tmp, name, args.fps, args.web_width, args.web_step, gif=args.gif)
            if not args.keep_frames:
                shutil.rmtree(tmp)
            print("  wrote %s and %s.png"
                  % (", ".join("%s (%.1f MB)" % (w, os.path.getsize(w) / 2 ** 20) for w in written), name), flush=True)
        else:
            print("  the frames are in %s" % tmp, flush=True)
    return 0


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="the file of movie_data.py fields, or a run.py --dump file")
    p.add_argument("--temperature", default=None,
                   help="the file of movie_data.py forward: the true temperature of a finer mesh in the large picture")
    p.add_argument("--out", default="results/animations")
    p.add_argument("--name", default="geothermal_hero")
    p.add_argument("--themes", nargs="+", default=["light", "dark"], choices=sorted(THEMES))
    p.add_argument("--frames", type=int, default=250, help="frames of the loop")
    p.add_argument("--only", type=int, nargs="+", default=None, help="draw these frames of the film only (PNG)")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--web-width", type=int, default=900, help="pixels across the WebP (and the GIF)")
    p.add_argument("--web-step", type=int, default=2, help="every this many frames in the WebP (and the GIF)")
    p.add_argument("--gif", action="store_true", help="a GIF as well")
    p.add_argument("--supersample", type=int, default=3)
    p.add_argument("--azimuth", type=float, default=-45.0)
    p.add_argument("--elevation", type=float, default=27.0)
    p.add_argument("--swing", type=float, default=35.0, help="the camera swings this many degrees to either side")
    p.add_argument("--vmax", type=float, default=1.8)
    p.add_argument("--std-max", type=float, default=0.7)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--seeds", type=int, default=4, help="streamlines of the heat flux a side")
    p.add_argument("--isotherms", type=float, nargs="+", default=[25, 50, 75, 100, 125, 150], help="kelvin above the surface")
    p.add_argument("--about", default="", help="the line under the application's name (default: from the data)")
    p.add_argument("--forward", nargs=2, default=["", ""], metavar=("UNKNOWNS", "REST"),
                   help="the two lines under the large picture (default: from the data)")
    p.add_argument("--inversion", nargs=2, default=["", ""], metavar=("UNKNOWNS", "REST"),
                   help="the two lines under the three small pictures (default: from the data)")
    p.add_argument("--hardware", nargs=2, default=["", ""], metavar=("FORWARD", "INVERSION"),
                   help="a third line under the large picture and under the small ones: what the forward solve and "
                        "the inversion ran on, and how long they took (under about 75 characters each)")
    p.add_argument("--turn", type=int, default=0,
                   help="the camera goes once around in this many loops, in the place of its swing; the film is that "
                        "many loops long (0: the swing)")
    p.add_argument("--reveal", type=float, nargs=2, default=[0.0, 0.0], metavar=("HOLD", "FADE"),
                   help="an opening in front of the loop, for a film that is played once: seconds for which the "
                        "quarter that is removed is still there, and seconds over which it then fades away (the "
                        "frames are those of the loop's end again, so the film is that much longer than the loop)")
    p.add_argument("--say", nargs="+", default=[], metavar="SENTENCE",
                   help="sentences written under the pictures one after another, each for the time it takes to say it")
    p.add_argument("--keep-frames", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(render(parse()))

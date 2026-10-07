# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The page, the camera and the encoding of the animation that ``movie.py`` draws.

Nothing here needs MFEM, JAX or MPI: NumPy, Pillow and Matplotlib (for its colour maps and
its fonts), and ffmpeg for the MP4.
"""
import glob
import os
import shutil
import subprocess

import numpy as np

THEMES = {
    "light": dict(bg=(1.0, 1.0, 1.0), ink=(0.05, 0.16, 0.30), line=(0.15, 0.17, 0.20),
                  source=(1.0, 0.72, 0.05), receiver=(0.10, 0.10, 0.12),
                  diverging="RdBu_r", sequential_std="viridis"),
    "dark": dict(bg=(0.051, 0.067, 0.090), ink=(0.90, 0.93, 0.96), line=(0.72, 0.76, 0.80),
                 source=(1.0, 0.85, 0.20), receiver=(0.95, 0.95, 0.95),
                 diverging="berlin", sequential_std="viridis"),
}


def font_files():
    """The files of the regular and the bold face: Lato where it is installed (the face of
    the repository's README animation; ``HIPPYMFEM_FONT_DIR`` names a folder that holds
    ``Lato-Regular.ttf`` and ``Lato-Bold.ttf``), DejaVu Sans, which Matplotlib carries,
    anywhere else."""
    folders = [os.environ.get("HIPPYMFEM_FONT_DIR", ""), "/usr/share/fonts/truetype/lato", "/usr/share/fonts/lato",
               "/usr/share/fonts/truetype", os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts")]
    for d in folders:
        files = {w: os.path.join(d, "Lato-%s.ttf" % w.capitalize()) for w in ("regular", "bold")}
        if d and all(os.path.exists(f) for f in files.values()):
            return files
    from matplotlib import font_manager

    return {"regular": font_manager.findfont("DejaVu Sans"), "bold": font_manager.findfont("DejaVu Sans:bold")}


FONTS = None


class Canvas:
    """A page the pictures are pasted on, with titles and colour bars."""

    def __init__(self, width, height, theme):
        from PIL import Image, ImageDraw

        self.th = THEMES[theme]
        self.rgb = lambda c: tuple(int(round(255 * x)) for x in c)
        self.img = Image.new("RGB", (width, height), self.rgb(self.th["bg"]))
        self.draw = ImageDraw.Draw(self.img)
        self._fonts = {}

    def font(self, size, weight="regular"):
        from PIL import ImageFont

        global FONTS
        if FONTS is None:
            FONTS = font_files()
        key = (size, weight)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(FONTS[weight], size)
        return self._fonts[key]

    def text(self, xy, string, size, weight="regular", anchor="mm", color=None):
        self.draw.text(xy, string, font=self.font(size, weight), fill=self.rgb(color or self.th["ink"]), anchor=anchor)

    def paste(self, image, xy):
        self.img.paste(image, xy)

    def colorbar(self, x, y, w, h, colors, ticks, label, size=19):
        from PIL import Image

        bar = (255 * colors(np.linspace(0.0, 1.0, w))).astype(np.uint8)
        self.img.paste(Image.fromarray(np.broadcast_to(bar[None, :, :], (h, w, 3)).copy()), (x, y))
        ink = self.rgb(self.th["line"])
        self.draw.rectangle([x - 1, y - 1, x + w, y + h], outline=ink, width=1)
        for pos, string in ticks:
            px = x + int(round(pos * (w - 1)))
            self.draw.line([px, y + h, px, y + h + 5], fill=ink, width=1)
            self.text((px, y + h + size), string, size - 1)
        self.text((x + w // 2, y + h + int(2.45 * size)), label, size)

    def width(self, string, size, weight="regular"):
        return self.draw.textlength(string, font=self.font(size, weight))


#: the page: the large picture on the left, three small ones on the right; ``more`` is
#: what a third line under the pictures adds to its height, ``band`` what a sentence does
ROW = dict(width=1600, height=716, big=560, small=336, x0=606, y0=136, dx=331, more=36, band=96)


def hero_row(theme, title, big, app, about, pics, titles, bars, forward, inversion):
    """The page of one frame.  Left: ``title`` over the large picture ``big``, and under
    it the size of the forward solve, ``forward = (the unknowns, in bold; the rest)``.
    Right: the application (``app`` in bold, ``about`` under it), the three small
    pictures with their titles, two colour bars ``(colors, ticks, label)`` (the first
    under the first two pictures), and the size of the inversion as on the left.
    A third string in ``forward`` or ``inversion`` is a line more under them (what the
    solve ran on and how long it took); the page is then that line higher."""
    R = ROW
    third = any(len(lines) > 2 and lines[2] for lines in (forward, inversion))
    height = R["height"] + (R["more"] if third else 0)
    cv = Canvas(R["width"], height, theme)
    small, x0, y0, dx = R["small"], R["x0"], R["y0"], R["dx"]
    cx = x0 + dx + small // 2
    cv.text((300, 30), title, 34, "bold")
    cv.paste(big, (20, 56))
    cv.text((cx, 26), app, 32, "bold")
    cv.text((cx, 64), about, 24)
    for j, (pic, t) in enumerate(zip(pics, titles)):
        cv.text((x0 + j * dx + small // 2, y0 - 25), t, 29, "bold")
        cv.paste(pic, (x0 + j * dx, y0))
    cv.colorbar(x0 + 40, y0 + small + 20, 2 * dx - 80 + (small - dx), 16, *bars[0], size=21)
    cv.colorbar(x0 + 2 * dx + 34, y0 + small + 20, small - 74, 16, *bars[1], size=21)
    for x, lines in ((300, forward), (cx, inversion)):
        rows = [(R["height"] - 58, lines[0], 38, "bold"), (R["height"] - 20, lines[1], 25, "regular")]
        if third:
            rows.append((R["height"] + R["more"] - 21, lines[2] if len(lines) > 2 else "", 24, "regular"))
        for y, string, size, weight in rows:
            if string:
                cv.text((x, y), string, size, weight)
    return cv


def wrap(cv, text, size, width):
    """``text`` in as few lines as fit ``width`` pixels, the lines of about one length."""
    words = text.split()
    for cuts in ([()], [(a,) for a in range(1, len(words))]):
        best = None
        for cut in cuts:
            lines = [" ".join(words[a:b]) for a, b in zip((0,) + cut, cut + (len(words),))]
            w = [cv.width(line, size) for line in lines]
            if max(w) <= width and (best is None or max(w) - min(w) < best[0]):
                best = (max(w) - min(w), lines)
        if best:
            return best[1]
    return [text]


def schedule(sentences, length, lead=0.6, pause=0.35, pace=0.063, shortest=2.5):
    """``[(start, end, sentence)]``, in seconds, for sentences that are shown one after
    another in a film of ``length`` seconds: each for the time it takes to say it
    (``pace`` seconds a character), sooner where the film is too short for that."""
    spans = [max(shortest, pace * len(s)) for s in sentences]
    room = length - lead - pause * len(sentences)
    scale = min(1.0, room / sum(spans)) if spans else 1.0
    out, t = [], lead
    for s, span in zip(sentences, spans):
        out.append((t, t + scale * span, s))
        t += scale * span + pause
    return out


def with_sentence(page, theme, cues, t):
    """``page`` with a band under it that holds the sentence shown at ``t`` seconds."""
    W, H = page.width, page.height + ROW["band"]
    H += H % 2
    cv = Canvas(W, H, theme)
    cv.paste(page, (0, 0))
    rule = tuple(int(round(255 * (0.82 * c + 0.18 * (1.0 - c)))) for c in cv.th["bg"])     # a little off the background
    cv.draw.rectangle([40, page.height + 5, W - 41, page.height + 6], fill=rule)
    say = next((s for t0, t1, s in cues if t0 - 0.05 <= t <= t1 + 0.25), None)
    if say:
        lines = wrap(cv, say, 30, W - 120)
        y = H - ROW["band"] // 2 - 19 * (len(lines) - 1) - 4
        for line in lines:
            cv.text((W // 2, y), line, 30)
            y += 38
    return cv.img


def view_axes(azimuth, elevation):
    """Unit vectors from the focus to the camera, to the right of the picture and up it."""
    az, el = np.radians(azimuth), np.radians(elevation)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    r = np.array([-np.sin(az), np.cos(az), 0.0])
    return d, r, np.cross(d, r)


def framing(points, azimuths, elevation, box, view_angle=20.0):
    """A focal point and a distance of the camera from which ``points`` stay inside ``box``
    (left, right, bottom, top, the picture being [-1, 1] squared; left = -right) at every
    one of ``azimuths``, as large as that allows.  The camera turns about the vertical
    through the middle of the points' footprint (the centre of the smallest circle around
    it), so that they stay in place as it swings; the height of the focal point puts
    them in the middle of the box."""
    th = np.tan(np.radians(0.5 * view_angle))
    xy = points[:, :2]
    c = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    for k in range(1, 300):                                   # Badoiu and Clarkson: towards the farthest point
        far = xy[np.argmax(((xy - c) ** 2).sum(axis=1))]
        c = c + (far - c) / (k + 1.0)
    f = np.array([c[0], c[1], 0.5 * (points[:, 2].min() + points[:, 2].max())])
    D = 4.0 * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    cy, hx, hy = 0.5 * (box[2] + box[3]), 0.5 * (box[1] - box[0]), 0.5 * (box[3] - box[2])
    for _ in range(30):
        lo, hi, wide = np.inf, -np.inf, 0.0
        for az in azimuths:
            d, r, u = view_axes(az, elevation)
            q = points - f
            depth = (D - q @ d) * th
            xs, ys = (q @ r) / depth, (q @ u) / depth
            lo, hi, wide = min(lo, ys.min()), max(hi, ys.max()), max(wide, np.abs(xs).max())
        f[2] += D * th * (0.5 * (lo + hi) - cy) / np.cos(np.radians(elevation))
        D *= max(wide / hx, 0.5 * (hi - lo) / hy)
    return f, D


def ffmpeg():
    """The ffmpeg to run: the one on the path, or the one the package ``imageio-ffmpeg``
    brings (``pip install imageio-ffmpeg``); None where there is neither."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:                                              # noqa: BLE001
            exe = None
    return exe


def encode(folder, out, fps, web_width, web_step, gif=False, crf=19):
    """The PNG frames ``f0000.png, ...`` of ``folder`` as ``out.mp4`` and as an animated
    WebP for a web page (``web_width`` pixels wide, every ``web_step``-th frame), and
    with ``gif`` as a GIF of that size.  Returns the files written."""
    from PIL import Image

    pattern, written = os.path.join(folder, "f%04d.png"), []
    exe = ffmpeg()
    run = lambda cmd: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if exe is None:
        print("  no ffmpeg on the path and no imageio-ffmpeg package: no MP4 is written (the WebP needs neither)", flush=True)
    else:
        run([exe, "-y", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
             "-preset", "slow", "-movflags", "+faststart", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out + ".mp4"])
        written.append(out + ".mp4")
        if gif:
            vf = "select='not(mod(n\\,%d))',setpts=N/(%g*TB),scale=%d:-1:flags=lanczos" % (web_step, fps / web_step, web_width)
            run([exe, "-y", "-framerate", str(fps), "-i", pattern, "-vf", vf + ",palettegen=max_colors=128:stats_mode=diff",
                 out + "_palette.png"])
            run([exe, "-y", "-framerate", str(fps), "-i", pattern, "-i", out + "_palette.png", "-lavfi",
                 vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-r", "%g" % (fps / web_step),
                 out + ".gif"])
            os.remove(out + "_palette.png")
            written.append(out + ".gif")
    ims = []
    for f in sorted(glob.glob(os.path.join(folder, "f*.png")))[::web_step]:
        im = Image.open(f).convert("RGB")
        ims.append(im.resize((web_width, round(im.height * web_width / im.width)), Image.LANCZOS))
    ims[0].save(out + ".webp", save_all=True, append_images=ims[1:], duration=int(round(1000.0 * web_step / fps)), loop=0,
                quality=65, method=6, minimize_size=True)
    written.append(out + ".webp")
    return written

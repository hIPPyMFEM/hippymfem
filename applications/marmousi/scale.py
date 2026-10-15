# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Throughput of the Marmousi application's kernels on a box cut into blocks, at any size.

What an inversion with the point spread function Hessian spends its time in, on one box
cut into the blocks of a grid of ranks (a rank a GPU), over a number of time steps rather
than a whole record so that the largest boxes fit a short job:

* the set-up, piece by piece: the mesh by cuts, the names of the wave field's nodes with
  the search for those that ranks share, the propagator, its coefficients, the points, the
  map between the library's vectors and the lattice;
* a step of the forward sweep (one field), of the incremental sweep and of the backward
  sweep (``1 + D`` fields for ``--directions D``; the backward one takes the forward field
  at the absorbing faces from the values that the incremental sweep kept), one source in
  every tile of the survey firing together;
* on the same blocks, as lattice arrays on the device: the construction of the point
  spread function operator, its products ``S``, ``S^T`` and ``H~ = W S S^T W``, the prior's
  product and solve, and ``--inner-its`` iterations of the solve with ``H~ + R`` for each
  preconditioner of ``--precondition``.

    mpirun -n 8 python scale.py --cells 96,96,48 --grid 2,2,2 --check --out blocks8.json

``--cells`` are the elements of ONE rank's block and ``--grid`` the blocks, so the load of
a rank stays and the box grows with the ranks.  The two weak-scaling series are one
command each:

    --cells 96,96,48 --grid PX,PY,1     columns: a rank holds a tile of the survey, top to bottom
    --cells 96,96,48 --grid P,P,P       blocks: cut in depth as well

At order 3 a block of 96 x 96 x 48 elements of 50 m is 12.1 million wave unknowns and 461
thousand parameters; ``--cells 384,384,48`` (columns) and ``192,192,192`` (blocks) are 192
million and 7.2 million, two fifths of a card of 64 GiB in double precision.

The model is the application's (:mod:`marmousi`): the three-dimensional Marmousi speed, a
formula of the point, so the box may be of any size (``--eps`` makes its tiles differ); the
initial model as the propagator's reference speed; the paper's window.  The parameter is
set halfway from the initial model to the truth.  The source's time function is the
application's, compressed in time when the steps taken are fewer than it lasts, so that
the source is active during them whatever their number.

The kernels of the operator are made by the application's own construction
(:func:`psf.build`) from the response to ONE comb of unit vectors, ``--psf-kernels``:

* ``synthetic``: a formula of the node and of the sample point, the same on any number of
  ranks: around every sample point an oscillating blob whose wavelength grows and whose
  amplitude falls with depth, as large against the prior as a Hessian that is well
  informed at the top;
* ``comb``: one true product of the Gauss-Newton Hessian with the comb over the steps
  taken, an incremental and a backward sweep, through the ice sheet application's comb
  loop.  The sample points lie under the sea bed, and a wave needs some 800 steps of
  1.34 ms to go down to the first of them and back: with fewer steps the response is zero
  around every one (the driver says so, and times the operator all the same).

Times.  Of a sweep, per step, as ``applications/seismic/scale.py`` has them (``sweeps`` in
the record): ``ms_per_step`` by rank 0's clock (the ranks advance together) in the fastest
of ``--reps`` runs after three steps that compile it, and a rank's time outside the
exchange, inside it, waiting for the element work and bringing values to the host, each
as least, mean and greatest over the ranks.  The first run of a sweep also fills the
device's memory pool and takes up to twice the later ones: ``all_ms_per_step`` has every
run.  Of the lattice's products (``products``, ``solves``): the mean of ``--product-reps``
calls after two that compile and warm up (and the fastest of them, ``least_ms``), a rank's
own milliseconds as least, mean and greatest, and of them the time in the transforms, in
the two halo exchanges (packing and unpacking included) and inside MPI (with the copies
to and from the host where the exchange goes through it).  ``HIPPYMFEM_DEVICE_MPI``
(``sem.py``, ``lattice.py``) says whether the exchanges go from card to card.
``setup_seconds`` and ``memory_gib`` are a rank's own as well, the three numbers each.

What must agree between runs of the same box on other grids of ranks, with the same
``--tile``, is ``checksums`` in the record (sums and inner products, each held against the
size of its terms, ``checksum_scales``).  ``--compare A.json B.json ...`` holds records
against the first and exits with 1 where they differ: the sweeps' and the products' sums
to round-off (a shared unknown is summed over its ranks in rank order, and the cuts of
the mesh move its vertices in their last digits), those that pass through the prior's
solve to its tolerance (one rank solves exactly, several by their own transforms to
``--prior-tol``); with ``--psf-kernels comb`` the lattice's sums to 1e-5 (the comb loop
keeps the response in single precision).  It also exits with 1, and says why, where a sum
is not a number, where one of two records lacks sums the other has, and where a record is
of a run that did not reach its end: the record is written at the start, after the
sweeps, after the products and at the end, ``ended`` in it says which it is, and a run
that was stopped holds the sums of what it had finished.

``--check`` prints, and fails on, the identities that hold to round-off whatever the size
and the ranks: ``<J dm, q> = <dm, J^T q>`` of the sweeps on the steps taken, ``<S u, c> =
<u, S^T c>``, the symmetry of ``H~`` and of ``R``, and ``R^-1 R u = u`` (to the solve's
tolerance on several ranks).
"""

import argparse
import contextlib
import functools
import gc
import itertools
import json
import os
import resource
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

#: how many times the prior's symbol the made-up Hessian's is, at the top and at the
#: wavenumber of its own kernel there (``--psf-kernels synthetic``)
STRENGTH = 1e3
#: the wavelength of the made-up kernels in nodes, at the top and what it gains down to
#: the bottom: about the half wavelengths of 7 Hz at 1500 and at 4000 m/s on nodes of 50 m
WAVELENGTH = (2.5, 3.5)
#: what an identity of ``--check`` may be, in double and in single precision: round-off of
#: sums of millions of terms
ROUNDOFF = {False: 1e-11, True: 1e-4}
#: what ``R^-1 R u = u`` may be where the solve is the exact one (one rank's cosine
#: transform): round-off times the condition of ``R``
INVERSE = {False: 1e-9, True: 1e-3}
#: the relative difference of a checksum that ``--compare`` lets pass
AGREE = {False: 1e-9, True: 1e-2}
#: the same for the lattice's checksums when the kernels are cut from a true comb product: the
#: comb loop keeps the response in single precision, and a value that rounds the other way on
#: another grid of ranks (its 14th digit differs) moves a kernel in its 7th
COMB = 1e-5


def _ints(count, least=1):
    """An argument of ``count`` integers, ``a,b,c`` (or ``axbxc``)."""
    def read(text):
        try:
            v = tuple(int(x) for x in text.replace("x", ",").split(","))
        except ValueError:
            v = ()
        if len(v) != count or min(v) < least:
            raise argparse.ArgumentTypeError("%d integers of at least %d, a,b,..: not %r"
                                             % (count, least, text))
        return v
    return read


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=_ints(3), default=(96, 96, 48), metavar="EX,EY,EZ",
                   help="elements of ONE rank's block along x, y and z")
    p.add_argument("--grid", type=_ints(3), default=(1, 1, 1), metavar="PX,PY,PZ",
                   help="blocks along x, y and z, as many as there are ranks (PX,PY,1: columns); the box "
                        "is --cells times --grid elements")
    p.add_argument("--h", type=float, default=50.0, help="side of an element in m")
    p.add_argument("--order", type=int, default=3, help="degree of the wave field's elements")
    p.add_argument("--float32", action="store_true",
                   help="single precision for the wave fields and for the lattice arrays")
    p.add_argument("--absorb", default="x0,x1,y0,y1,z1",
                   help="faces of the WHOLE box that absorb (a rank's block has those of its faces that "
                        "are faces of the box), or none; z0, the top, is the free surface")
    p.add_argument("--history", default="device", choices=("device", "host"),
                   help="where a sweep keeps the field's values at the absorbing nodes")
    p.add_argument("--steps", type=int, default=100, help="time steps each sweep is timed over")
    p.add_argument("--directions", type=int, default=1,
                   help="parameter directions of the incremental and of the backward sweep (0: the "
                        "forward sweep alone)")
    p.add_argument("--reps", type=int, default=2, help="runs of each sweep; the fastest is recorded")
    p.add_argument("--overlap-both", action="store_true",
                   help="time the three sweeps a second time with the exchange of a step the other way "
                        "(between the two halves of the step where sem.py's rule put it behind the "
                        "update, and the reverse; HIPPYMFEM_SEM_OVERLAP): `sweeps_other` in the record. "
                        "Nothing on one rank or without --directions")
    p.add_argument("--check", action="store_true",
                   help="verify the identities of the sweeps and of the lattice's operators, and exit "
                        "with 1 where one is not round-off")
    p.add_argument("--tile", type=_ints(2), default=None, metavar="TX,TY",
                   help="tiles of the survey along x and y, a source in each (default: the ranks' blocks "
                        "in plan; give it to compare runs of one box on other grids)")
    p.add_argument("--receivers", type=int, default=2,
                   help="receivers a side of the patch around every source: the records of all of them "
                        "are gathered on every rank, so few")
    p.add_argument("--depth", type=float, default=25.0, help="depth of sources and receivers in m")
    p.add_argument("--eps", type=float, default=0.5,
                   help="the model's modulation along x, with which no two tiles hold the same medium")
    p.add_argument("--margins", type=_ints(3, 0), default=(600, 600, 400), metavar="MX,MY,MZ",
                   help="m over which the window falls to zero at the sides (x, y) and at the bottom")
    p.add_argument("--cuts", default="auto",
                   help="the mesh's cuts: auto (the factors of 2 to 8 that divide a block's three sides, "
                        "so that the coarse mesh every rank builds whole is the smallest), none (the "
                        "whole mesh on every rank) or the factors, e.g. 6,8")
    p.add_argument("--parts", default="wave,lattice",
                   help="what to run: wave (the sweeps), lattice (the operator, the prior, the solve; "
                        "without wave no mesh is built at all) or both")
    p.add_argument("--psf-step", type=int, default=16, help="nodes between the sample points")
    p.add_argument("--psf-rho", type=int, default=8,
                   help="radius of the impulse responses and of the kernels, in nodes")
    p.add_argument("--psf-anchor", type=int, default=0,
                   help="a node of the sample lattice (0 puts its planes on the blocks' faces where the "
                        "step divides a block's elements: the fewest boxes a rank)")
    p.add_argument("--psf-kernels", default="synthetic", choices=("synthetic", "comb"),
                   help="where the comb's response comes from: a formula, or one Hessian product on the "
                        "steps taken (see above)")
    p.add_argument("--psf-eta", type=float, default=0.1,
                   help="weight of the operator's low-frequency term, a fraction of its largest symbol")
    p.add_argument("--precondition", default="prior,symbol",
                   help="preconditioners of the solve with H~ + R, each timed: of prior, symbol, split, "
                        "none; or no for no solve")
    p.add_argument("--inner-its", type=int, default=10,
                   help="iterations of that solve, whatever its residual")
    p.add_argument("--product-reps", type=int, default=5,
                   help="calls a product of the lattice part is timed over")
    p.add_argument("--prior-sigma", type=float, default=25.0, help="the prior's pointwise deviation in m/s")
    p.add_argument("--prior-rho", type=float, default=200.0, help="the prior's correlation length in m")
    p.add_argument("--prior-tol", type=float, default=None,
                   help="accuracy of the prior's solve on several ranks, which sets the halo of a rank's "
                        "own transform (default 1e-8; in single precision 1e-5, which it can meet)")
    p.add_argument("--prior-solver", default="auto", choices=("auto", "cosine", "fft", "chebyshev"),
                   help="the prior's solve (prior.LatticePrior); auto is the exact cosine transform on "
                        "one rank and a rank's own transform with a halo on several: fft on one rank is "
                        "the same work as on many")
    p.add_argument("--time-limit", type=float, default=0.0,
                   help="seconds after which no further part is started and the record is written "
                        "(0: none)")
    p.add_argument("--out", default="", help="the JSON record")
    p.add_argument("--compare", nargs="+", default=None, metavar="JSON",
                   help="hold records of the same box on other grids against the first, and run nothing")
    p.add_argument("--tol", type=float, default=None,
                   help="relative difference that --compare lets pass (default 1e-9; in single precision "
                        "1e-2)")
    args = p.parse_args(argv)
    args.parts = tuple(args.parts.split(","))
    if not args.parts or set(args.parts) - {"wave", "lattice"}:
        p.error("--parts is wave, lattice or wave,lattice")
    args.precondition = () if args.precondition == "no" else tuple(args.precondition.split(","))
    if set(args.precondition) - {"prior", "symbol", "split", "none"}:
        p.error("--precondition is a list of prior, symbol, split, none; or no")
    faces = () if args.absorb == "none" else tuple(args.absorb.split(","))
    if set(faces) - {"x0", "x1", "y0", "y1", "z1"}:
        p.error("--absorb is a list of x0, x1, y0, y1, z1; or none")
    args.absorb = faces
    if args.prior_tol is None:
        args.prior_tol = 1e-5 if args.float32 else 1e-8
    if args.psf_kernels == "comb" and set(args.parts) != {"wave", "lattice"}:
        p.error("--psf-kernels comb is a product of the wave part for the lattice part: it needs both")
    if 2 * args.psf_rho > args.psf_step:
        p.error("--psf-rho is at most half of --psf-step: one comb gives every sample point's response")
    if min(args.steps, args.reps, args.product_reps, args.receivers) < 1 or args.directions < 0:
        p.error("--steps, --reps, --product-reps and --receivers are at least 1")
    return args


# ------------------------------------------------------------------ records side by side
def compare(files, tol=None):
    """Hold the checksums of the records ``files`` against the first one's, a difference
    against the size of the sum's terms; 1 if one is above ``tol``, a record is of another
    problem, an identity is above its limit, or something could not be held against
    anything: the record of a run that did not reach its end, sums that one of two records
    lacks, a sum that is not a number."""
    recs = []
    for f in files:
        with open(f) as fh:
            recs.append(json.load(fh))
    names = [os.path.basename(f) for f in files]
    ref = recs[0]
    tol = AGREE[bool(ref["problem"]["float32"])] if tol is None else tol
    limits = dict(sweeps=tol, products=tol, solves=tol)
    if ref["problem"].get("psf_kernels") == "comb":
        limits.update(products=max(tol, COMB), solves=max(tol, COMB))
    # what passed through the prior's solve agrees to that solve's tolerance: one rank solves
    # exactly, several with a kernel of the inverse that is cut where it has fallen to it
    limits["solves"] = max(limits["solves"], 300.0 * ref["problem"].get("prior_tol", 0.0))
    # ``ok``: nothing compared differs; ``whole``: nothing was left without its counterpart
    ok, whole = True, True
    for name, r in zip(names, recs):
        single = bool(r["problem"]["float32"])
        # (a record without its limits: round-off, or the tolerance of a solve that is not exact)
        most = {k: 300.0 * r["problem"].get("prior_tol", 0.0) if k == "R solve" else ROUNDOFF[single]
                for k in r["identities"]}
        most.update(r.get("identity_limits", {}))
        bad = {k: v for k, v in r["identities"].items() if not v <= most[k]}
        if bad:
            ok = False
            print("%s: not round-off: %s" % (name, ", ".join("%s %.1e" % kv for kv in bad.items())))
        # a run that was stopped or killed leaves the record of what it had done (of nothing, if
        # it fell in its set-up): what there is is held against the others', and it is said
        if r.get("ended", "whole") != "whole":
            whole = False
            print("%s: not a whole run (%s)" % (name, r["ended"]))
    for name, r in zip(names[1:], recs[1:]):
        a, b = ref["problem"], r["problem"]
        differs = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        if differs:
            ok = False
            print("%s: another problem than %s: %s" % (name, names[0], ", ".join(
                "%s %s and %s" % (k, b.get(k), a.get(k)) for k in differs)))
            continue
        lines = []
        mine, other = ref["checksums"], r["checksums"]
        for group in sorted(set(mine) | set(other)):
            if group not in mine or group not in other:
                whole = False
                lines.append("%s in %s only" % (group, names[0] if group in mine else name))
                continue
            limit = limits.get(group, tol)
            worst, which = 0.0, ""
            sizes = ref.get("checksum_scales", {}).get(group, {})
            for k in sorted(set(mine[group]) & set(other[group])):
                x = np.atleast_1d(np.asarray(mine[group][k], dtype=float))
                y = np.atleast_1d(np.asarray(other[group][k], dtype=float))
                # against the size of the sum's terms where the record has it: a sum that cancels
                # is not small against itself
                size = np.abs(np.atleast_1d(np.asarray(sizes.get(k, x), dtype=float)))
                e = np.inf
                if x.shape == y.shape == size.shape:
                    e = float(np.max(np.abs(x - y) / np.maximum(size, 1e-300)))
                # (a sum that is not a number is above every limit: NaN compares false both ways,
                # and was the smallest difference of all)
                if np.isnan(e) or e >= worst:
                    worst, which = (np.inf if np.isnan(e) else e), k
            # sums that one record alone has: the iterates of a solve ("x.x prior") are there for
            # the preconditioners a run was asked for, every other sum in every whole run
            lone = sorted(set(mine[group]) ^ set(other[group]))
            if not which or any(not k.startswith("x.") for k in lone):
                whole = False
            ok = ok and worst <= limit
            lines.append("%s %.1e (%s; %s %.0e)%s"
                         % (group, worst, which, "within" if worst <= limit else "ABOVE", limit,
                            ", in one only: " + ", ".join(lone) if lone else ""))
        whole = whole and bool(lines)
        print("%s (%d ranks, grid %s) against %s (%d ranks, grid %s): %s"
              % (name, r["ranks"], "x".join(map(str, r["grid"])), names[0], ref["ranks"],
                 "x".join(map(str, ref["grid"])), "; ".join(lines) or "nothing in common"))
    print("the records %s" % ("DIFFER" if not ok else "agree" if whole else
                              "agree in what could be compared, which is NOT ALL of them (above)"))
    return 0 if ok and whole else 1


# ------------------------------------------------------------------ what a run is made of
def field(own, shape, salt):
    """A field on the nodes ``own`` (a box of global nodes) of a lattice of ``shape``, between
    -1/2 and 1/2: a function of the node's number, in integers, so that the bits are the
    same whichever rank holds the node.

    Not a white field: the fractional part of a multiple of the node's number is a sawtooth
    along one direction of the lattice, a plane wave of 2.3 nodes along x with its
    harmonics (on 49 x 49 x 25 nodes its 32 largest Fourier coefficients hold three
    quarters of its energy), and two salts give the same wave at two phases (salts 1 and 2
    are correlated by -0.18).  That serves what it is used for: sums that differ if a value
    is wrong anywhere, and identities held to round-off.  It is no measure of an operator
    over all scales (``H_against_R`` in the record is its size at that one wave)."""
    k, j, i = np.meshgrid(*[np.arange(lo, hi, dtype=np.int64) for lo, hi in own], indexing="ij",
                          sparse=True)
    n = ((k * shape[1] + j) * shape[2] + i + 7919 * salt) % 1000003
    return ((n * 2654435761) % 1000003).astype(np.float64) / 1000003.0 - 0.5


def survey(box, tile, depth, side):
    """``(sources, receivers)`` of a survey of ``tile = (tx, ty)`` tiles: one source a tile, a
    little off its middle (the middle is a node, and on a cut between ranks where the tiles
    are the blocks), and ``side x side`` receivers 1.7 elements apart around the middle,
    within reach of the source in the few steps a sweep is timed over, all at ``depth``."""
    tx, ty = tile
    h, lo = box.h, box.origin
    L = box.size[:2] / (tx, ty)
    J, I = np.meshgrid(np.arange(ty), np.arange(tx), indexing="ij")              # tile I + tx J
    mid = lo[:2] + (np.stack([I.ravel(), J.ravel()], axis=1) + 0.5) * L
    if ((0.85 * (side - 1) + 1.0) * h[:2] >= 0.5 * L).any():
        raise ValueError("%d x %d receivers 1.7 elements apart do not fit a tile of %g x %g m"
                         % (side, side, L[0], L[1]))
    z = lo[2] + depth
    src = np.column_stack([mid + (0.3 * h[0], -0.2 * h[1]), np.full(mid.shape[0], z)])
    a = 1.7 * (np.arange(side) - 0.5 * (side - 1))
    off = np.stack(np.meshgrid(a * h[0], a * h[1], indexing="ij"), axis=-1).reshape(-1, 2)
    rec = (mid[:, None, :] + off[None, :, :]).reshape(-1, 2)
    return src, np.column_stack([rec, np.full(rec.shape[0], z)])


def source_time(t, h):
    """The application's source time function for elements of ``h`` m (the paper's band at
    50 m, scaled with the mesh) at the times ``t``, compressed in time where they end
    before it does: active during the steps taken, whatever their number."""
    import marmousi

    scale = h / 50.0
    lasts = 0.88 * scale                              # below 1e-3 of its peak from here on
    s = min(1.0, float(t[-1]) / lasts) * scale
    return marmousi.wavelet(t, fc=7.0 / s, bw=1.5 / s, t0=0.5 * s)


def sample_constraint(box, margins, rho):
    """Where a sample point may be, ``[(lo, hi)]`` in nodes along z, y, x, both included:
    where the window is one (below the water, and the margins away from the sides and the
    bottom) and ``rho`` nodes from where it is not, so that no impulse response is taken
    across the edge of the window.  The window is a product of one factor a direction, so
    it is read along the three lines through the middle of the box, as ``run.py`` reads
    it: a few thousand nodes of the largest box."""
    import marmousi

    x, y, z = marmousi.nodes(box, box.h)
    mid = [0.5 * (c[0] + c[-1]) for c in (x, y, z)]
    lines = (marmousi.window(mid[0], mid[1], z, box, margins),
             marmousi.window(mid[0], y, mid[2], box, margins),
             marmousi.window(x, mid[1], mid[2], box, margins))
    out = []
    for w in lines:
        one = np.nonzero(w >= 1.0 - 1e-12)[0]
        # (where it is one nowhere: a range that holds no node, which psf.combs refuses)
        out.append((int(one[0]) + rho, int(one[-1]) - rho) if one.size else (1, 0))
    return out


def synthetic_response(sample, own, rho, amplitude, put):
    """The response of a made-up Hessian to the comb of unit vectors at the sample points,
    on the nodes ``own``, as a device array: around the sample point ``x_s`` the blob

        ``A(z_s) exp(-|y|^2 / 2 s^2) cos(2 pi e_s . y / lambda(z_s))``,   ``y = x - x_s``,

    for ``|y| <= rho`` in every direction, with ``s = 0.3 rho``, a wavelength that grows and
    an amplitude that falls with the depth of the sample point, and a direction ``e_s`` that
    leans from the vertical by an angle that changes slowly from sample point to sample
    point.  Its symbol is positive (two Gaussians at ``+- e_s / lambda``), as a Hessian's.
    A formula of global numbers: the same field on any blocks.  ``put`` takes a host array
    to the device in double precision; the three-dimensional work is done there."""
    import jax.numpy as jnp

    nodes = [np.arange(lo, hi) for lo, hi in own]
    depth = max(sample.shape[0] - 1, 1)
    s2 = 2.0 * (0.3 * rho) ** 2
    out = 0.0
    # a node is within rho of the sample point before it, of the one after it, or of both
    for side in itertools.product((0, 1), repeat=3):
        index, y, factor = [], [], []
        for k in range(3):
            a = (nodes[k] - sample.first[k]) // sample.step + side[k]
            d = (nodes[k] - (sample.first[k] + sample.step * a)).astype(np.float64)
            near = (a >= 0) & (a < sample.K[k]) & (np.abs(d) <= rho)
            index.append(np.clip(a, 0, sample.K[k] - 1))
            y.append(put(d.reshape([-1 if j == k else 1 for j in range(3)])))
            factor.append(np.where(near, np.exp(-d * d / s2), 0.0))
        t = (sample.first[0] + sample.step * index[0]) / depth
        factor[0] = factor[0] * amplitude / (1.0 + 3.0 * t) ** 2
        wave = put((2.0 * np.pi / (WAVELENGTH[0] + WAVELENGTH[1] * t))[:, None, None])
        lean = 0.5 * np.sin(0.3 * index[1][:, None] + 0.2 * index[2][None, :])
        blob = jnp.cos(wave * (put(np.cos(lean)[None]) * y[0]
                               + put(np.sin(lean)[None]) * (0.6 * y[1] + 0.8 * y[2])))
        for k in range(3):
            blob = blob * put(factor[k].reshape([-1 if j == k else 1 for j in range(3)]))
        out = out + blob
    return out


def _timed(fn, seconds, name):
    @functools.wraps(fn)
    def call(*args, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            seconds[name] = seconds.get(name, 0.0) + time.perf_counter() - t0
    return call


@contextlib.contextmanager
def clocked(seconds, targets):
    """While the block runs, the seconds of every call of the functions ``targets``
    (``{name: (module or class, attribute)}``) are added to ``seconds[name]``.

    The propagator's set-up is two calls of other modules (:func:`box.engine`, and in it
    ``sem.Collocation.from_keys``) whose pieces grow differently with the ranks: this times
    the pieces from outside and leaves the modules as they are."""
    kept = []
    for name, (owner, attribute) in targets.items():
        raw = vars(owner)[attribute]
        kept.append((owner, attribute, raw))
        setattr(owner, attribute, classmethod(_timed(raw.__func__, seconds, name))
                if isinstance(raw, classmethod) else _timed(raw, seconds, name))
    try:
        yield
    finally:
        for owner, attribute, raw in kept:
            setattr(owner, attribute, raw)


class Run:
    """What the parts of a run share: the communicator, the box, its lattice and this rank's
    block of it, the record, and this rank's own numbers on their way into it.

    A rank's numbers (seconds of a piece, memory, sizes) are noted under a key, the same
    keys in the same order on every rank, and :meth:`collect` turns them into ``[least,
    mean, greatest]`` over the ranks in the record, three reductions for all of them."""

    def __init__(self, args, comm):
        import box
        import lattice
        from hippymfem.fem import kernel

        self.args, self.comm, self.rank0 = args, comm, comm.rank == 0
        self.t_start = time.time()
        grid = tuple(args.grid)
        if int(np.prod(grid)) != comm.size:
            raise ValueError("--grid %s is %d blocks, and there are %d ranks"
                             % (grid, int(np.prod(grid)), comm.size))
        n = np.asarray(args.cells) * np.asarray(grid)
        size = args.h * n
        #: the box, with the paper's section in its middle plane ``y = 0``
        self.box = box.Box(n, size, (0.0, -0.5 * size[1], 0.0))
        self.grid, self.tile = grid, tuple(args.tile or grid[:2])
        self.lat = lattice.Lattice(n, grid)
        block = self.lat.block(comm.rank)
        self.closed, self.own = self.lat.closed(block), self.lat.owned(block)
        self.own_shape = tuple(hi - lo for lo, hi in self.own)
        self.single = bool(args.float32)
        self.dtype = np.dtype(np.float32 if self.single else np.float64)
        self.device = kernel.device()
        self.mine = {}
        self._lap = time.perf_counter()
        #: whether the time limit has stopped a part
        self.stopped = False
        # what two runs must share for their checksums to agree
        problem = dict(elements=[int(v) for v in n], h=args.h, float32=self.single, eps=args.eps,
                       margins=list(args.margins))
        if "wave" in args.parts:
            problem.update(order=args.order, absorb=list(args.absorb), steps=args.steps, depth=args.depth,
                           directions=args.directions, tile=list(self.tile), receivers=args.receivers)
        if "lattice" in args.parts:
            problem.update(psf_step=args.psf_step, psf_rho=args.psf_rho, psf_anchor=args.psf_anchor,
                           psf_kernels=args.psf_kernels, psf_eta=args.psf_eta, inner_its=args.inner_its,
                           prior_sigma=args.prior_sigma, prior_rho=args.prior_rho,
                           prior_tol=args.prior_tol)
        self.record = dict(args=vars(args), problem=problem, ranks=comm.size, grid=list(grid),
                           cells=list(args.cells), device=str(self.device), ended="early",
                           checksums={}, checksum_scales={}, identities={}, identity_limits={})

    def say(self, *a):
        if self.rank0:
            print(*a, flush=True)

    def put(self, x, dtype=None):
        """A host array on the device, in the precision of the run (or in ``dtype``)."""
        import jax

        return jax.device_put(np.ascontiguousarray(x, dtype=dtype or self.dtype), self.device)

    def note(self, *key_and_value):
        """Keep this rank's number ``value`` under ``key`` (the names of nested dictionaries of
        the record) for :meth:`collect`."""
        self.mine[tuple(key_and_value[:-1])] = float(key_and_value[-1])

    def lap(self, name, less=0.0):
        """This rank's seconds since the last lap, less ``less``, as the set-up's piece ``name``
        (None: the clock is only started again); the ranks then wait for each other, so that
        the next piece starts on all at once."""
        if name is not None:
            self.note("setup_seconds", name, time.perf_counter() - self._lap - less)
        self.comm.Barrier()
        self._lap = time.perf_counter()

    def over_ranks(self, x):
        """Least, mean and greatest over the ranks of each of the numbers ``x`` a rank holds,
        a row each (MPI's own reductions, on buffers)."""
        from mpi4py import MPI

        x = np.array(x, dtype=np.float64)
        lo, hi, tot = np.empty_like(x), np.empty_like(x), np.empty_like(x)
        self.comm.Allreduce(x, lo, op=MPI.MIN)
        self.comm.Allreduce(x, hi, op=MPI.MAX)
        self.comm.Allreduce(x, tot, op=MPI.SUM)
        return np.stack([lo, tot / self.comm.size, hi], axis=1)

    def collect(self):
        """Every number noted so far into the record, as ``[least, mean, greatest]``."""
        if not self.mine:
            return
        rows = self.over_ranks(list(self.mine.values()))
        for key, row in zip(self.mine, rows):
            at = self.record
            for name in key[:-1]:
                at = at.setdefault(name, {})
            at[key[-1]] = row.tolist()
        self.mine = {}

    def identity(self, name, value, limit=None):
        """Record what an identity holds to, and what it may be (round-off by default)."""
        self.record["identities"][name] = float(value)
        self.record["identity_limits"][name] = ROUNDOFF[self.single] if limit is None else float(limit)

    def memory(self, stage):
        """Note what the card holds now and held at most, and the host's peak."""
        stats = (self.device.memory_stats() or {}) if hasattr(self.device, "memory_stats") else {}
        self.note("memory_gib", stage, "card", stats.get("bytes_in_use", 0) / 2.0 ** 30)
        self.note("memory_gib", stage, "card_peak", stats.get("peak_bytes_in_use", 0) / 2.0 ** 30)
        self.note("memory_gib", stage, "host_peak",
                  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2.0 ** 20)
        self.record["card_limit_gib"] = stats.get("bytes_limit", 0) / 2.0 ** 30

    def late(self):
        """Whether the time limit has passed, by one answer for every rank: a part that some
        ranks started and others skipped would leave the first waiting."""
        from mpi4py import MPI
        from hippymfem.common.parvector import allreduce

        limit = self.args.time_limit
        if limit > 0.0 and allreduce(self.comm, int(time.time() - self.t_start > limit), MPI.MAX):
            self.stopped = True
        return self.stopped

    def write(self):
        """The record as it stands, to ``--out``: a run that is stopped leaves what it had."""
        self.collect()
        self.record["seconds"] = time.time() - self.t_start
        if self.args.out and self.rank0:
            os.makedirs(os.path.dirname(os.path.abspath(self.args.out)), exist_ok=True)
            with open(self.args.out + ".part", "w") as f:
                json.dump(self.record, f, indent=1)
            os.replace(self.args.out + ".part", self.args.out)


def lattice_fields(run):
    """The application's model on this rank's closed block of the lattice (the nodes of its
    elements), and of it on the nodes the rank owns: the window, and the point at which the
    sweeps linearize, halfway from the initial model to the truth inside the window."""
    import marmousi

    F = marmousi.on_lattice(run.box, run.box.h, block=run.closed, comm=run.comm, eps=run.args.eps,
                            margins=tuple(float(m) for m in run.args.margins))
    inside = tuple(slice(o[0] - c[0], o[1] - c[0]) for o, c in zip(run.own, run.closed))
    window = np.ascontiguousarray(F.window[inside])
    run.lap("model on the lattice")
    return F, window, 0.5 * window * F.q_true[inside]


def wave_part(run, F, point):
    """The decomposed box's propagator and its three sweeps.  Returns ``[engine, map,
    apply]``: the propagator, the map between the parameter's vectors and the lattice, and
    the Gauss-Newton Hessian of the steps taken as a function of blocks of lattice columns
    (what the comb loop asks for)."""
    import hippymfem as hm
    from hippymfem.common.parvector import allreduce
    import jax.numpy as jnp
    import box
    import lattice
    import marmousi

    args, comm, B, say, record = run.args, run.comm, run.box, run.say, run.record
    sem, order, N, D = box.sem, args.order, args.steps, args.directions

    def density(u, a, m, p, x, c0):
        # the application's wave equation: the parameter is the offset of the speed from the reference
        return a.val * p.val / (c0 + m.val) ** 2 + hm.inner(u.grad, p.grad)

    # ---------------------------------------------------------------- the set-up, piece by piece
    cuts = {"auto": B.cuts(run.grid, few=0), "none": ()}.get(args.cuts)
    cuts = tuple(int(c) for c in args.cuts.split(",")) if cuts is None else cuts
    coarse = int(np.prod(B.n // (int(np.prod(cuts)) if cuts else 1)))
    if comm.size > 1 and coarse > box.SERIAL["most"]:
        raise ValueError("every rank would build a coarse mesh of %d elements whole: give --cells three "
                         "numbers that share a factor made of 2 to 8 (cuts %s)" % (coarse, cuts))
    pmesh = B.mesh(comm, run.grid, cuts)
    run.lap("mesh")
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    run.lap("parameter space")
    inner = {}
    pieces = {"keys": (sem.Collocation, "from_keys"), "keys, of which their sort": (sem, "unique_inverse"),
              "keys, of which shared vertices": (sem.Interface, "__init__"),
              "keys, of which shared nodes": (sem.Interface, "from_keys"), "block": (box.Box, "block"),
              "absorbing nodes": (box.Box, "absorbing"), "propagator": (sem.SpectralWave, "__init__")}
    with clocked(inner, pieces):
        pmesh, Vm, col, eng = box.engine(
            comm, B, order, density, aux=marmousi.lattice_aux(F.c0, B, B.h, block=run.closed),
            dtype=jnp.float32 if run.single else jnp.float64, absorb=args.absorb, history=args.history,
            grid=run.grid, space=Vm)
    for name in pieces:
        run.note("setup_seconds", name, inner.get(name, 0.0))
    # (what is left of the call when its timed pieces are taken out)
    run.lap("engine, the rest", less=sum(inner.get(k, 0.0) for k in (
        "keys", "block", "absorbing nodes", "propagator")))
    lmap = lattice.LatticeMap(Vm, run.lat, B.size, B.origin)
    run.lap("lattice map")
    q = lmap.from_lattice(point)
    eng.set_parameter(q)
    run.lap("coefficients")
    eng.dt, eng.nsteps = B.dt(order, marmousi.CMAX), N
    sources, receivers = survey(B, run.tile, args.depth, args.receivers)
    wavelets = np.tile(source_time(eng.dt * np.arange(N + 1), args.h)[:, None], (1, sources.shape[0]))
    eng.set_sources(B.value_operator(col, sources, comm, order), wavelets)
    eng.set_receivers(B.value_operator(col, receivers, comm, order))
    run.lap("points")
    ne, nd, nm = int(pmesh.GetGlobalNE()), int(eng.global_size), int(Vm.GlobalTrueVSize())
    nab = allreduce(comm, int(eng.ab_dofs.size)) if args.absorb else 0
    shared = allreduce(comm, int(eng.iface.iface.size))
    # (sem.py's rule for the way of its exchange)
    card_to_card = sem.DEVICE_MPI in ("1", "yes", "true", "on") or (
        sem.DEVICE_MPI == "auto" and os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0") == "1"
        and run.device.platform != "cpu")
    run.note("rank", "elements", col.ne)
    run.note("rank", "wave_dofs", eng.nloc)
    run.note("rank", "absorbing_nodes", eng.ab_dofs.size if args.absorb else 0)
    run.note("rank", "neighbours", len(eng.iface.neighbors))
    run.note("rank", "sent_values", sum(p.size for p in eng.iface.pos.values()))
    run.memory("after the set-up")
    run.collect()
    record.update(elements=ne, wave_dofs=nd, parameters=nm, cuts=list(cuts), absorbing_nodes=nab,
                  shared_dofs=shared, dt=eng.dt, sources=int(sources.shape[0]),
                  receivers=int(receivers.shape[0]),
                  exchange="card to card" if card_to_card else "host arrays")
    T, R = record["setup_seconds"], record["rank"]
    say("marmousi scale: %s elements of %g m and order %d in %s blocks of %s (cuts %s): %d elements, %d "
        "wave unknowns, %d parameters; %d ranks, %s, %s; exchange through %s"
        % ("x".join(str(int(v)) for v in B.n), args.h, order, "x".join(map(str, run.grid)),
           "x".join(map(str, args.cells)), cuts, ne, nd, nm, comm.size, run.device,
           "fp32" if run.single else "fp64", record["exchange"]))
    say("  setup, s, the slowest rank: " + ", ".join("%s %.2f" % (k, v[2]) for k, v in T.items()))
    say("  steps: %d of %.4f ms; %d sources, one a tile of %d x %d, and %d receivers; absorbing nodes %d "
        "and shared unknowns %d, counted on every rank that holds them (%.2f %% of all); boundary values "
        "on the %s" % (N, 1e3 * eng.dt, sources.shape[0], *run.tile, receivers.shape[0], nab, shared,
                       100.0 * shared / max(nd, 1), args.history))
    say("  neighbours of a rank: %d to %d; values a rank sends a field a step: %d at most (%.1f MB), %d "
        "on average" % (R["neighbours"][0], R["neighbours"][2], R["sent_values"][2],
                        1e-6 * R["sent_values"][2] * run.dtype.itemsize, R["sent_values"][1]))

    def overlapped(fields):
        """Whether a sweep of that many fields makes its exchange behind the update of a step
        (sem.py's OVERLAP and OVERLAP_MIN; said of the ranks that share something)."""
        return allreduce(comm, int(eng._overlapped(fields)) if hasattr(eng, "_overlapped") else 0) > 0

    def way():
        return dict(setting=getattr(sem, "OVERLAP", None), least=getattr(sem, "OVERLAP_MIN", None),
                    one_field=overlapped(1), fields=overlapped(1 + D))

    record["overlap"] = way()
    if comm.size > 1:
        say("  exchange of a step behind its update: %s with one field, %s with %d (HIPPYMFEM_SEM_OVERLAP %s)"
            % ("yes" if record["overlap"]["one_field"] else "no", "yes" if record["overlap"]["fields"] else "no",
               1 + D, record["overlap"]["setting"]))

    # ---------------------------------------------------------------- the three sweeps
    def clocks():
        return np.array([eng.t_wave, eng.t_exchange, eng.t_element, eng.t_copy])

    def timed(fn):
        comm.Barrier()
        c0 = clocks()
        t0 = time.perf_counter()
        out = fn()
        comm.Barrier()
        seconds = time.perf_counter() - t0
        w, x, e, c = clocks() - c0
        # over the ranks: a rank's time outside MPI, inside it, waiting for the element work,
        # bringing the values it sends to the host
        return out, seconds, w, run.over_ranks([w - x, x, e, c])

    sweeps = record.setdefault("sweeps", {})
    text = {"adjoint": "backward"}

    def sweep(name, fields, warm, go, into=None):
        """Compile the sweep on 3 steps, run it ``--reps`` times, record and print the fastest.
        (The backward sweep is ``adjoint`` in the record, as in the seismic driver's.)"""
        sweeps = record["sweeps"] if into is None else into
        warmup = timed(warm)[1]
        every, best, out = [], None, None
        for _ in range(args.reps):
            if every and run.late():
                break
            out = None                # the last run's states and boundary values, before the next
            out, seconds, w, tr = timed(go)
            every.append(1e3 * w / N)
            if best is None or w < best[1]:
                best = (seconds, w, tr)
        seconds, w, tr = best
        ms = 1e3 * tr / N
        sweeps[name] = dict(
            fields=fields, seconds=seconds, ms_per_step=1e3 * w / N, all_ms_per_step=every,
            exchange_ms_per_step=ms[1, 2], warmup_seconds=warmup, rank_own_ms_per_step=ms[0].tolist(),
            rank_exchange_ms_per_step=ms[1].tolist(), rank_element_ms_per_step=ms[2].tolist(),
            rank_copy_ms_per_step=ms[3].tolist(), dof_updates_per_s=fields * nd * N / w,
            ns_per_dof=1e9 * w * comm.size / (N * nd))
        say("  %-24s %8.3f ms/step, %.4f ns an unknown of a rank, %.2f G unknown-updates/s; runs %s; "
            "compiled on 3 steps in %.1f s"
            % ("%s (%d field%s):" % (text.get(name, name), fields, "" if fields == 1 else "s"), 1e3 * w / N,
               sweeps[name]["ns_per_dof"], 1e-9 * fields * nd * N / w, " ".join("%.3f" % v for v in every),
               warmup))
        say("      a rank, least / mean / most over the ranks, ms/step: outside MPI %.3f / %.3f / %.3f "
            "(waiting for the element work %.3f / %.3f / %.3f, values to the host %.3f / %.3f / %.3f), in "
            "MPI %.3f / %.3f / %.3f" % (*ms[0], *ms[2], *ms[3], *ms[1]))
        return out

    y = sweep("forward", 1, lambda: eng.forward(limit=3), eng.forward)
    # as before every shot of an inversion; it also gives up the forward sweep's last states and its
    # boundary values, which the two sweeps below do not start from
    eng.set_wavelets(wavelets)
    # the records' root mean square stands for the data's in the noise: one number of the whole box
    noise = 0.01 * float(np.sqrt(np.mean(y ** 2))) or 1.0

    def apply_block(M):
        """The products of the Gauss-Newton Hessian of the steps taken (the noise 1 % of the
        records) with the columns of ``M``, lattice fields on the owned nodes."""
        dirs = [lmap.from_lattice(np.ascontiguousarray(M[:, j]).reshape(run.own_shape))
                for j in range(M.shape[1])]
        dy, last = eng.incremental(dirs, final=True)
        out = eng.adjoint(dy / noise ** 2, final=last)
        return np.stack([lmap.to_lattice(g).reshape(-1) for g in out], axis=1)

    run.memory("after the forward sweep")
    sums = record["checksums"]["sweeps"] = dict(y_norm=float(np.linalg.norm(y)))
    # the size of a sum's terms, which a difference between two runs is held against: a sum
    # that cancels (a gradient against a field that is zero where the gradient is not) is
    # round-off itself, and no two runs agree on it
    sizes = record["checksum_scales"]["sweeps"] = dict(y_norm=sums["y_norm"])
    if D > 0 and not run.late():
        dms = [lmap.from_lattice(30.0 * field(run.own, run.lat.shape, 11 + k)) for k in range(D)]
        dy, last = sweep("incremental", 1 + D, lambda: eng.incremental(dms, limit=3),
                         lambda: eng.incremental(dms, final=True))
        # the same series on every rank: all of them hold the records of all the receivers
        series = np.random.default_rng(5).standard_normal((D, N + 1, receivers.shape[0]))
        series[:, 0] = 0.0
        g = sweep("adjoint", 1 + D, lambda: eng.adjoint(series, final=last, limit=3),
                  lambda: eng.adjoint(series, final=last))
        del last
        lhs = [float(np.sum(dy[k] * series[k])) for k in range(D)]
        rhs = [float(g[k].inner(dms[k])) for k in range(D)]
        # sums that see every entry with a weight of its own: a permutation of the receivers would
        # show; the gradients against their directions, against the point (zero in the water, where
        # the gradient of a few steps is all there is) and against one more field
        weight = np.cos(0.37 * np.arange(y.size) + 0.1).reshape(y.shape)
        other = lmap.from_lattice(field(run.own, run.lat.shape, 7))
        wn, on, qn = (float(np.linalg.norm(weight)), float(np.sqrt(other.inner(other))),
                      float(np.sqrt(q.inner(q))))
        dyn = [float(np.linalg.norm(dy[k])) for k in range(D)]
        gn = [float(np.sqrt(g[k].inner(g[k]))) for k in range(D)]
        dmn = [float(np.sqrt(dms[k].inner(dms[k]))) for k in range(D)]
        sums.update(y_weighted=float(np.sum(weight * y)),
                    dy_weighted=[float(np.sum(weight * dy[k])) for k in range(D)], dy_norm=dyn,
                    g_norm=gn, g_direction=rhs, g_point=[float(g[k].inner(q)) for k in range(D)],
                    g_other=[float(g[k].inner(other)) for k in range(D)])
        sizes.update(y_weighted=wn * sums["y_norm"], dy_weighted=[wn * v for v in dyn], dy_norm=dyn,
                     g_norm=gn, g_direction=[a * b for a, b in zip(gn, dmn)],
                     g_point=[a * qn for a in gn], g_other=[a * on for a in gn])
        if not np.abs(dy).max() > 0.0:
            say("  <J dm, q> = <dm, J^T q>: not tested, nothing has reached a receiver in %d steps" % N)
            if args.check:
                # an identity that could not be tested is not one that holds.  Here, unlike on the
                # globe, a receiver of every tile stands in its source's element or in one beside
                # it, and J dm is there within three steps: zero is a sweep that did not do its work
                run.identity("sweeps", np.inf)
        elif args.check:
            size = [np.linalg.norm(dy[k]) * np.linalg.norm(series[k]) for k in range(D)]
            # (numpy's: a direction whose number is not one makes the whole not one; max() of
            # Python lets a NaN that is not the first pass)
            worst = float(np.max([abs(lhs[k] - rhs[k]) / size[k] for k in range(D)]))
            run.identity("sweeps", worst)
            say("  <J dm, q> = <dm, J^T q> to %.2e of |J dm| |q| (%.2e of <J dm, q>); max |y| %.3e, max "
                "|J dm| %.3e" % (worst, np.max([abs(lhs[k] - rhs[k]) / max(abs(lhs[k]), 1e-300)
                                                for k in range(D)]), np.abs(y).max(), np.abs(dy).max()))
        per_step = 1e-3 * (sweeps["incremental"]["ms_per_step"] + sweeps["adjoint"]["ms_per_step"])
        record["hessian_product_seconds_per_step"] = per_step
        say("  one shot: a Hessian product on %d direction%s %.3f s a hundred steps, a gradient %.3f s"
            % (D, "" if D == 1 else "s", 100.0 * per_step,
               0.1 * (sweeps["forward"]["ms_per_step"] + sweeps["adjoint"]["ms_per_step"])))
        if args.overlap_both and comm.size > 1 and hasattr(sem, "OVERLAP") and not run.late():
            # the same three sweeps with the exchange of a step the other way: what the choice is worth
            # at this size and on these ranks, from one run (the set-up is most of a run's time)
            was = sem.OVERLAP
            sem.OVERLAP = "0" if record["overlap"]["one_field"] or record["overlap"]["fields"] else "1"
            try:
                other = record["sweeps_other"] = {}
                record["overlap_other"] = way()
                say("  the same sweeps with HIPPYMFEM_SEM_OVERLAP=%s (the exchange %s):"
                    % (sem.OVERLAP, "behind the update" if sem.OVERLAP == "1"
                       else "between the two halves of a step"))
                y2 = sweep("forward", 1, lambda: eng.forward(limit=3), eng.forward, into=other)
                eng.set_wavelets(wavelets)
                dy2, last = sweep("incremental", 1 + D, lambda: eng.incremental(dms, limit=3),
                                  lambda: eng.incremental(dms, final=True), into=other)
                g2 = sweep("adjoint", 1 + D, lambda: eng.adjoint(series, final=last, limit=3),
                           lambda: eng.adjoint(series, final=last), into=other)
                del last
                apart = [float(np.max(np.abs(y2 - y)) / max(float(np.max(np.abs(y))), 1e-300)),
                         float(np.max(np.abs(dy2 - dy)) / max(float(np.max(np.abs(dy))), 1e-300)),
                         float(np.max([abs(float(g2[k].inner(dms[k])) - rhs[k]) / max(gn[k] * dmn[k], 1e-300)
                                       for k in range(D)]))]
                record["overlap_other"]["apart"] = apart
                say("      the two ways apart: the records %.1e, J dm %.1e, the gradient against its "
                    "direction %.1e" % tuple(apart))
                if args.check:
                    run.identity("overlap", max(apart))
            finally:
                sem.OVERLAP = was
    run.memory("after the sweeps")
    return [eng, lmap, apply_block]


def lattice_part(run, window, wave=None):
    """The point spread function operator, the prior and the solve with their sum on this
    rank's block of the lattice.  ``wave`` is what the wave part returned, in a list that
    is emptied here as soon as the comb's response is there: the propagator's arrays leave
    the card before this part's come, and what the card then holds is this part's own."""
    from mpi4py import MPI
    from hippymfem.common.parvector import allreduce
    import jax
    import lattice
    import prior
    import psf

    args, comm, lat, say, record = run.args, run.comm, run.lat, run.say, run.record
    h3, rho, step = tuple(float(v) for v in run.box.h), args.psf_rho, args.psf_step
    ready = jax.block_until_ready

    def new_halo(width):
        return lattice.DeviceHalo(lattice.Halo(lat, comm.rank, width), comm, run.dtype)

    # ---------------------------------------------------------------- the set-up, piece by piece
    run.lap(None)
    margins = tuple(float(m) for m in args.margins)
    sample = psf.combs(lat.shape, step, args.psf_anchor, 1, sample_constraint(run.box, margins, rho))
    # the halo of the construction: its width from one block of every index a direction, which
    # between them have every range of nodes that the blocks of all the ranks have (px + py + pz
    # boxes, not px py pz, whose number grows with the ranks)
    lines = [lat.owned(tuple(i if d == a else 0 for d in range(3)))
             for a in range(3) for i in range(lat.P[a])]
    halo_b = new_halo(np.maximum(psf.build_width(sample, lines, rho), rho))
    halo = new_halo(rho)
    run.lap("sample points and halos")
    gamma, delta = prior.coefficients(args.prior_sigma, args.prior_rho, h3)
    pr = prior.LatticePrior(lat, comm.rank, comm, h3, gamma, delta, dtype=run.dtype, tol=args.prior_tol,
                            solver=args.prior_solver)
    run.lap("prior")
    if args.psf_kernels == "comb":
        responses, _ = psf.comb_products(wave[2], sample, halo_b, dtype=run.dtype)
    else:
        # STRENGTH times the prior's symbol at the wavenumber of the kernel at the top: the sum of
        # the blob's values is its symbol there, half of its Gaussian's
        m = 64
        r_top = float(pr.symbols(m)[1][int(round(m / WAVELENGTH[0])), 0, 0])
        amplitude = 2.0 * STRENGTH * r_top / (0.3 * rho * np.sqrt(2.0 * np.pi)) ** 3
        responses = synthetic_response(sample, run.own, rho, amplitude,
                                       lambda x: run.put(x, np.float64))[None]
    if wave:
        wave.clear()
        gc.collect()
    run.lap("comb response")
    kern = psf.build(responses, sample, halo_b, rho, window=window, comm=comm, dtype=run.dtype)
    del responses
    run.lap("kernels")
    # (the largest symbol is that of all the ranks: one answer for every rank)
    empty = not kern.smax > 0.0
    if empty:
        say("  kernels: ALL ZERO.  In %d steps no wave has gone from a source down to a sample point "
            "and back to a receiver; the operator is timed all the same (its work does not depend on "
            "its numbers)" % args.steps)
    op = psf.PSFOperator(halo, kern, window=window, eta=0.0 if empty else args.psf_eta,
                         blur_sigma=0.25 * rho, dtype=run.dtype)
    run.lap("operator")
    mem = op.memory()
    # (every number of it: a counter that psf.py gains is in the record without a change here)
    for name, value in mem.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            run.note("rank", "psf_" + name, value)
    run.note("rank", "lattice_nodes", int(np.prod(run.own_shape)))
    run.note("rank", "halo_neighbours", len(halo.plan.neighbors))
    run.note("rank", "halo_sent_values", halo.plan.nsend)
    run.memory("after the operator's set-up")
    run.collect()
    exact = pr.solver == "cosine"
    record.update(lattice=dict(
        nodes=int(np.prod(lat.shape)), shape=list(lat.shape), sample_points=list(sample.K),
        sample_first=list(sample.first), box=mem["box"], cell=mem["cell"], radius=mem["radius"],
        fft_points_per_node=mem["fft_points_per_node"], smax=kern.smax,
        prior=dict(gamma=gamma, delta=delta, solver=pr.solver, width=pr.width, degree=pr.degree,
                   bound=pr.bound),
        exchange="card to card" if halo.on_device else "host arrays"))
    T, R = record["setup_seconds"], record["rank"]
    # (an operator that leaves out, in H~, the tiles on which the window is zero counts those it keeps)
    kept = R.get("psf_boxes_tapered")
    say("  lattice: %d nodes, %d a rank at most; sample points %s along z, y, x from node %s, %d nodes "
        "apart; a rank: %d to %d boxes of %d^3%s in %d batches at most, symbols %.0f MB at most, %.1f "
        "transformed points a node"
        % (record["lattice"]["nodes"], R["lattice_nodes"][2], "x".join(map(str, sample.K)),
           list(sample.first), step, R["psf_boxes"][0], R["psf_boxes"][2], mem["box"],
           " (%d to %d in H~)" % (kept[0], kept[2]) if kept else "", R["psf_batches"][2],
           R["psf_symbols_bytes"][2] / 2.0 ** 20, mem["fft_points_per_node"]))
    if exact:
        solve = "the cosine transform, exact"
    elif pr.solver == "fft":
        solve = "a rank's own transform, halo %s, to %.1e" % (pr.width, pr.bound)
    else:
        solve = "a polynomial of degree %d, to %.1e" % (pr.degree, pr.bound)
    say("  halo: %d nodes, %d to %d neighbours, %.2f MB sent a rank at most, through %s; the prior's "
        "solve: %s" % (rho, R["halo_neighbours"][0], R["halo_neighbours"][2],
                       1e-6 * R["halo_sent_values"][2] * run.dtype.itemsize, record["lattice"]["exchange"],
                       solve))
    say("  setup of the lattice, s, the slowest rank: " + ", ".join("%s %.2f" % (k, T[k][2]) for k in (
        "sample points and halos", "prior", "comb response", "kernels", "operator")))

    # ---------------------------------------------------------------- products
    def in_mpi():
        """The seconds so far inside the exchanges of every halo of this part.  (The halos of
        the prior's polynomial are made as it runs and kept in a table of its own.)"""
        halos = [halo_b, halo, pr.halo1, pr.halo2] + ([pr.haloW] if pr.solver == "fft" else [])
        return sum(x.seconds for x in halos + [c[0] for c in getattr(pr, "_chunks", {}).values()])

    def product(name, fn, x, operator=True):
        """This rank's ms a call of ``fn(x)`` after two calls that compile and warm up (the mean
        of the calls, and the fastest of them), and of them inside the exchanges' MPI; with
        the operator's own clocks for its products."""
        out = ready(fn(x))
        out = ready(fn(x))
        comm.Barrier()
        for k in op.timers:
            op.timers[k] = 0.0
        before, t0, least = in_mpi(), time.perf_counter(), np.inf
        for _ in range(args.product_reps):
            t1 = time.perf_counter()
            out = ready(fn(x))
            least = min(least, time.perf_counter() - t1)
        scale = 1e3 / args.product_reps
        run.note("products", name, "ms", scale * (time.perf_counter() - t0))
        run.note("products", name, "least_ms", 1e3 * least)
        if operator:
            run.note("products", name, "transforms_ms", scale * (op.timers["S"] + op.timers["St"]))
            run.note("products", name, "exchange_ms", scale * op.timers["exchange"])
        run.note("products", name, "mpi_ms", scale * (in_mpi() - before))
        return out

    dot = halo.dot

    def norm(a):
        return np.sqrt(dot(a, a))

    def apart(a, b, x, y):
        """``|a - b|`` over ``|x| |y|``, and zero where both are (an operator that is zero)."""
        return abs(a - b) / (norm(x) * norm(y) or 1.0)

    u, c = run.put(field(run.own, lat.shape, 1)), run.put(field(run.own, lat.shape, 2))
    Su, Stu, Hu = product("S", op.S, u), product("St", op.St, u), product("H", op.H, u)
    Ru, Ku = product("R", pr.apply_R, u, False), product("R solve", pr.solve_R, u, False)
    run.collect()
    fields = dict(u=u, c=c, Su=Su, Stu=Stu, Hu=Hu, Ru=Ru, Ku=Ku)
    size = {name: norm(a) for name, a in fields.items()}

    def inner_products(pairs):
        """The inner products of the pairs of fields named, and the sizes of their terms."""
        return ({"%s.%s" % (a, b): dot(fields[a], fields[b]) for a, b in pairs},
                {"%s.%s" % (a, b): size[a] * size[b] for a, b in pairs})

    sums, sizes = inner_products([("u", "u"), ("Su", "Su"), ("Stu", "Stu"), ("Hu", "Hu"), ("Hu", "u"),
                                  ("Hu", "c"), ("Su", "c"), ("Stu", "c"), ("Ru", "Ru"), ("Ru", "u"),
                                  ("Ru", "c")])
    solved, solved_sizes = inner_products([("Ku", "Ku"), ("Ku", "c")])
    record["checksums"].update(products=sums, solves=solved)
    record["checksum_scales"].update(products=sizes, solves=solved_sizes)
    record["lattice"]["H_against_R"] = sums["Hu.u"] / sums["Ru.u"]
    n_own = allreduce(comm, int(np.prod(run.own_shape)), MPI.MAX)
    P = record["products"]
    for name, text in (("S", "S"), ("St", "S^T"), ("H", "H~ = W S S^T W")):
        v = P[name]
        say("  psf product: %-15s %8.2f ms (a rank %.2f / %.2f / %.2f), %.2f ns a node; transforms %.2f, "
            "the two exchanges %.2f / %.2f / %.2f, of them in MPI %.2f / %.2f / %.2f"
            % (text, v["ms"][2], *v["ms"], 1e6 * v["ms"][2] / n_own, v["transforms_ms"][1],
               *v["exchange_ms"], *v["mpi_ms"]))
    for name, text in (("R", "prior product: R"), ("R solve", "prior solve: R^-1")):
        v = P[name]
        say("  %-28s %8.2f ms (a rank %.2f / %.2f / %.2f), %.2f ns a node; in MPI %.2f / %.2f / %.2f"
            % (text, v["ms"][2], *v["ms"], 1e6 * v["ms"][2] / n_own, *v["mpi_ms"]))
    if args.check:
        ids = {"S adjoint": apart(sums["Su.c"], dot(u, op.St(c)), Su, c),
               "H symmetric": apart(sums["Hu.c"], dot(u, op.H(c)), Hu, c),
               "R symmetric": apart(sums["Ru.c"], dot(u, pr.apply_R(c)), Ru, c)}
        for name, value in ids.items():
            run.identity(name, value)
        back = norm(pr.solve_R(Ru) - u) / norm(u)
        # one rank's cosine transform is the inverse; a rank's own transform is it to the tolerance
        run.identity("R solve", back, INVERSE[run.single] if exact else 300.0 * args.prior_tol)
        say("  identities: <S u, c> = <u, S^T c> to %.1e, H~ symmetric to %.1e, R symmetric to %.1e, "
            "R^-1 R u = u to %.1e%s; <u, H~ u> / <u, R u> = %.3g"
            % (ids["S adjoint"], ids["H symmetric"], ids["R symmetric"], back,
               "" if exact else " (the solve's tolerance: %.0e)" % args.prior_tol,
               record["lattice"]["H_against_R"]))
    del Su, Stu, Hu, Ru, Ku, fields
    run.memory("after the products")
    run.write()

    # ---------------------------------------------------------------- the solve with H~ + R
    b = run.put(field(run.own, lat.shape, 3))
    for pre in args.precondition:
        if run.late():
            break
        t0 = time.perf_counter()
        A = psf.ApproximateHessian(op, pr, precondition=pre, rtol=1e-30)
        run.note("solves", pre, "setup_seconds", time.perf_counter() - t0)
        # compiles: two iterations, because the new direction (psf._direction) is first made after
        # the first one, and with one it was compiled inside the solve that is timed below, of the
        # first preconditioner only (18 ms on an MI210: 1.8 ms an iteration of ten)
        A.solve_lattice(b, maxit=2)
        for k in A.timers:
            A.timers[k] = 0.0
        comm.Barrier()
        before = in_mpi()
        x, its = A.solve_lattice(b, maxit=args.inner_its)
        scale = 1e3 / max(its, 1)
        for key, name in (("solve", "ms"), ("H", "H_ms"), ("R", "R_ms"), ("precondition", "precondition_ms")):
            run.note("solves", pre, name, scale * A.timers[key])
        run.note("solves", pre, "mpi_ms", scale * (in_mpi() - before))
        run.note("solves", pre, "iterations", its)
        solved["x.x " + pre], solved["x.c " + pre] = dot(x, x), dot(x, c)
        solved_sizes["x.x " + pre], solved_sizes["x.c " + pre] = solved["x.x " + pre], norm(x) * size["c"]
        run.collect()
        v = record["solves"][pre]
        say("  solve with %-7s %8.2f ms an iteration of H~ + R (a rank %.2f / %.2f / %.2f) over %d: H~ "
            "%.2f, R %.2f, the preconditioner %.2f, in MPI %.2f / %.2f / %.2f; its set-up %.2f s"
            % (pre + ":", v["ms"][2], *v["ms"], int(v["iterations"][0]), v["H_ms"][1], v["R_ms"][1],
               v["precondition_ms"][1], *v["mpi_ms"], v["setup_seconds"][2]))
        del A, x
    run.memory("after the solves")


def main(argv=None):
    args = parse(argv)
    if args.compare:
        return compare(args.compare, args.tol)
    import hippymfem as hm                           # noqa: F401  (first: it sets the device up)
    from mpi4py import MPI
    import mfem.par as mfem

    comm = MPI.COMM_WORLD
    mfem.Hypre.Init()
    run = Run(args, comm)
    record, say = run.record, run.say
    # the record of the start ("ended": "early"): a run that falls in its set-up leaves this, and
    # not the record of an earlier run under the same name, for whoever reads the file afterwards
    run.write()
    F, window, point = lattice_fields(run)
    wave = None
    if "wave" in args.parts:
        wave = wave_part(run, F, point)
        run.write()
    del F, point
    if "lattice" in args.parts and not run.late():
        if wave is None:
            say("marmousi scale: the lattice of %s elements of %g m in %s blocks of %s, no mesh; %d ranks, "
                "%s, %s" % ("x".join(str(int(v)) for v in run.box.n), args.h, "x".join(map(str, run.grid)),
                            "x".join(map(str, args.cells)), comm.size, run.device,
                            "fp32" if run.single else "fp64"))
        elif args.psf_kernels != "comb":
            wave.clear()
            gc.collect()
        lattice_part(run, window, wave)
    wave = None
    run.memory("at the end")
    run.collect()
    M = record["memory_gib"]
    say("  device memory at peak: %.2f GiB a rank at most (of %.2f it may take), %.2f on average; host "
        "%.1f GiB a rank at most" % (M["at the end"]["card_peak"][2], record["card_limit_gib"],
                                     M["at the end"]["card_peak"][1], M["at the end"]["host_peak"][2]))
    say("      a rank at most, GiB on the card then / on the card at most until then / on the host at "
        "most: " + "; ".join("%s %.2f / %.2f / %.1f" % (k, v["card"][2], v["card_peak"][2], v["host_peak"][2])
                             for k, v in M.items()))
    record["ended"] = "stopped by the time limit" if run.stopped else "whole"
    # (the seismic driver's names for the two peaks)
    record.update(device_peak_gib=M["at the end"]["card_peak"], host_rss_gib=M["at the end"]["host_peak"])
    run.write()
    failed = {k: v for k, v in record["identities"].items() if not v <= record["identity_limits"][k]}
    if failed:
        say("marmousi scale: NOT ROUND-OFF: " + ", ".join("%s %.1e" % kv for kv in failed.items()))
    say("marmousi scale: %s in %.1f s" % ("stopped by --time-limit" if run.stopped else "done",
                                          time.time() - run.t_start))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # an exception on one rank would leave the others waiting in their next exchange until the
        # job's time limit: print it and stop them all (what sem.abort_all() does, here without the
        # need for the propagator's module to have been found)
        from mpi4py import MPI

        if MPI.COMM_WORLD.size > 1:
            traceback.print_exc()
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)
        raise

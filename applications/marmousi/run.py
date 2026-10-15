# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Bayesian inversion of the wave speed of a box of the Marmousi model from shot records,
with the point spread function approximation of the Hessian, end to end.

The stages, each resuming what an earlier run saved in ``--out``:

``data``    the truth, the records of every shot with noise (a file a shot), the true and
            the initial model and the window on the lattice;
``verify``  the gradient against differences of the cost, the Hessian's symmetry, the
            Gauss-Newton Hessian against differences of the gradient (on request);
``psf``     products of the misfit Hessian at the starting point with combs of unit
            vectors and with check vectors in the same sweeps; the point spread function
            operator of them, its errors on the check vectors, its symbols against the
            prior's, the iterations of a solve with it plus the prior;
``map``     the MAP point by L-BFGS whose initial inverse Hessian is nothing, the prior's
            covariance or the solve with that operator plus the prior (``--precondition``),
            or by the library's Gauss-Newton-CG with the same three as preconditioners
            (``--map-solver newton``); ``--tag`` names the run, so that several sit side by
            side in one folder;
``post``    the operator again at the MAP point and the Laplace approximation with it:
            samples, pointwise standard deviations, and a test of the samples against
            products with the Hessian itself;
``chain``   gpCN with that Laplace approximation as its reference, and pCN beside it (on
            request).

Two layouts (``--layout``).  ``farm``: every rank holds the whole box on its own card and
a share of the shots, rank 0 runs the stages and the others do the sweeps they are told.
``decomposed``: one box cut into the ranks' blocks, every rank in every shot and in every
operation on the lattice.  A run can be continued in either, on any number of ranks: what
is saved is in the lattice's order.

``python run.py --help`` lists the options.  The small case (4.8 x 2.4 x 2.4 km in elements
of 100 m, 2 to 5 Hz, 32 shots; a gradient takes a minute on one card) and its three
histories:

    S="--size 4800,2400,2400 --h 100 --source-step 600 --receiver-step 200 --depth 50
       --margins 400,400,300 --noise 0.5 --prior-sigma 25 --prior-rho 400 --psf-step 8
       --psf-rho 4 --out out/small"
    mpirun -n 8 python run.py $S --stages data,psf
    mpirun -n 8 python run.py $S --stages map --precondition none
    mpirun -n 8 python run.py $S --stages map --precondition prior
    mpirun -n 8 python run.py $S --stages map,post --precondition psf
"""

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STAGES = ("data", "verify", "psf", "map", "post", "chain")
#: iterations at most of the solves by which the stage psf compares the inner preconditioners
PSF_TEST_ITERATIONS = 400


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("box and discretization")
    g.add_argument("--size", default="9200,2400,2400",
                   help="the box's lengths along the section, across it and down, m")
    g.add_argument("--x0", type=float, default=None,
                   help="where the box begins along the section, m (default: centred on the middle of "
                        "the section's 9200 m)")
    g.add_argument("--h", type=float, default=50.0,
                   help="side of an element and step of the parameter's lattice, m")
    g.add_argument("--order", type=int, default=3, help="polynomial degree of the wave field")
    g.add_argument("--cfl", type=float, default=0.8,
                   help="fraction of the time step that is stable for 4000 m/s")
    g.add_argument("--record", type=float, default=4.0, help="length of a shot's record, s")
    g.add_argument("--float32", action="store_true", help="single precision wave fields")
    g.add_argument("--absorb", default="x0,x1,y0,y1,z1",
                   help="the faces that let the waves out (none: all natural); the top is a free surface")
    g.add_argument("--history", choices=("device", "host"), default="device",
                   help="where a sweep keeps the forward field's values at the absorbing faces")
    g.add_argument("--batch", type=int, default=16,
                   help="directions of a Hessian product in one pair of sweeps of a shot, at most")
    g = p.add_argument_group("experiment")
    g.add_argument("--source-step", type=float, default=400.0, help="distance of the sources in plan, m")
    g.add_argument("--receiver-step", type=float, default=100.0, help="distance of the receivers, m")
    g.add_argument("--depth", type=float, default=25.0, help="depth of sources and receivers, m")
    g.add_argument("--tile", default=None,
                   help="tx,ty: the survey repeated over that many tiles, a shot firing one source in "
                        "every tile")
    g.add_argument("--wavelet", choices=("compact", "paper"), default="compact",
                   help="compact: a cosine under a Gaussian with the paper's band scaled to the mesh "
                        "(10 Hz x 50 / h at its upper end); paper: the paper's own, from its file")
    g.add_argument("--fc", type=float, default=None, help="compact: centre frequency, Hz (7 x 50 / h)")
    g.add_argument("--bw", type=float, default=None,
                   help="compact: standard deviation of the spectrum, Hz (1.5 x 50 / h)")
    g.add_argument("--t0", type=float, default=None, help="compact: time of the peak, s (0.5 x h / 50)")
    g.add_argument("--margins", default="600,600,400",
                   help="widths of the window's taper at the two pairs of sides and at the bottom, m")
    g.add_argument("--initial-smooth", type=float, default=120.0,
                   help="width of the Gaussian that makes the initial model of the true one, m")
    g.add_argument("--eps", type=float, default=0.0,
                   help="modulation of the third dimension along the section (1/2 for surveys of tiles: "
                        "no two tiles then hold the same medium)")
    g.add_argument("--full-truth", action="store_true",
                   help="the data are those of the true speed everywhere, also on the faces and in the "
                        "margins where the window keeps the unknown from following it (the paper's "
                        "situation); by default those of c0 + W (c_true - c0), which the unknown reaches")
    g.add_argument("--noise", type=float, default=0.01,
                   help="noise standard deviation as a fraction of the root mean square of the true data")
    g.add_argument("--seed", type=int, default=1)
    g = p.add_argument_group("prior")
    g.add_argument("--prior-sigma", type=float, default=25.0,
                   help="pointwise standard deviation of the offset on the lattice, m/s")
    g.add_argument("--prior-rho", type=float, default=200.0, help="correlation length, m")
    g = p.add_argument_group("point spread functions (lengths in nodes of the lattice)")
    g.add_argument("--psf-step", type=int, default=16, help="distance of the sample points")
    g.add_argument("--psf-rho", type=int, default=8,
                   help="radius of the mask that cuts an impulse response out of a comb's response (at "
                        "most half the distance of a comb's teeth, step x teeth)")
    g.add_argument("--psf-rho-g", type=int, default=None, help="radius of the kernels (default: --psf-rho)")
    g.add_argument("--psf-teeth", type=int, default=1,
                   help="sample points from one tooth of a comb to the next, a direction: teeth^3 "
                        "products with the Hessian")
    g.add_argument("--psf-filter", type=float, default=0.0,
                   help="width of the paper's high-pass filter on both sides of the Hessian (0: none)")
    g.add_argument("--psf-eta", type=float, default=0.1,
                   help="weight of the low-frequency term in the operator's square root, as a fraction "
                        "of its largest symbol (0: none)")
    g.add_argument("--psf-blur", type=float, default=None,
                   help="width of that term's Gaussian (default: a quarter of the kernels' radius, "
                        "which cuts it at four widths)")
    g.add_argument("--psf-weights", choices=("hat", "bspline"), default="hat",
                   help="how the sample points' kernels are interpolated")
    g.add_argument("--psf-check", type=int, default=4,
                   help="vectors the operator is checked on, in the sweeps of the combs: half white, "
                        "half samples of the prior")
    g.add_argument("--psf-amplitude", type=float, default=0.0,
                   help="the operator as W a S S^T a W: a the local size of the Hessian (the square root "
                        "of its diagonal, from the products with the white check vectors, smoothed by a "
                        "Gaussian of this width in nodes across and a third of it in depth), the kernels "
                        "of S from the responses divided by a at both ends (0: no amplitude; needs "
                        "--psf-check 1 at least)")
    g.add_argument("--psf-weight", type=float, default=1.0,
                   help="the weight of H~ against the prior in the operator H~ + R that preconditions L-BFGS "
                        "(the objective is not changed): below one a damped Gauss-Newton inverse, which holds "
                        "back the steps where the data say little and the misfit is far from its quadratic "
                        "model")
    g.add_argument("--inner-precondition", choices=("prior", "symbol", "split", "levels", "none"), default="prior",
                   help="preconditioner of the conjugate gradients on the operator plus the prior")
    g.add_argument("--inner-rtol", type=float, default=1e-6, help="their relative residual")
    g.add_argument("--inner-maxit", type=int, default=2000, help="their iterations at most")
    g = p.add_argument_group("MAP point")
    g.add_argument("--map-solver", choices=("lbfgs", "newton"), default="lbfgs")
    g.add_argument("--precondition", choices=("none", "prior", "psf"), default="psf",
                   help="the initial inverse Hessian of L-BFGS, or the preconditioner of Newton's "
                        "conjugate gradients: nothing, the prior's covariance, or the solve with the "
                        "point spread function operator at the starting point plus the prior")
    g.add_argument("--tag", default=None,
                   help="name of the MAP solve in --out (default: solver-precondition)")
    g.add_argument("--max-iter", type=int, default=100, help="iterations in all, earlier runs' included")
    g.add_argument("--rtol", type=float, default=1e-6, help="reduction of the gradient's norm that ends it")
    g.add_argument("--memory", type=int, default=100, help="pairs L-BFGS keeps")
    g.add_argument("--first-step", type=float, default=100.0,
                   help="L-BFGS: the most its first trial step may change the speed, m/s (there is no "
                        "pair yet to take a scale from)")
    g.add_argument("--start-from", default=None, metavar="TAG",
                   help="a new MAP solve starts from the last point of the solve TAG of this folder (from the "
                        "initial model without it); with --precondition psf its operator is built at that point, "
                        "from products of the Hessian there: plain L-BFGS through the strongly nonlinear first "
                        "iterations, the point spread functions where the misfit is nearer its quadratic model")
    g.add_argument("--max-step", type=float, default=0.0,
                   help="L-BFGS: the most a trial step may change the speed, m/s, at every iteration (0: only "
                        "the first is bounded, by --first-step).  A preconditioned direction far from the data "
                        "can ask for steps that leave the region where the misfit is near its quadratic model")
    g.add_argument("--max-backtrack", type=int, default=10, help="halvings of a step in a line search")
    g.add_argument("--cg-max", type=int, default=50, help="newton: conjugate gradient iterations a step")
    g.add_argument("--lazy-gradient", action="store_true",
                   help="the library's order: a trial point's misfit alone, and the gradient of the point "
                        "that is kept in sweeps of its own (default: both in one pass over the shots)")
    g = p.add_argument_group("Laplace approximation and chains")
    g.add_argument("--samples", type=int, default=4, help="samples of prior and posterior that are saved")
    g.add_argument("--var-samples", type=int, default=32,
                   help="samples the pointwise standard deviations are of")
    g.add_argument("--post-check", type=int, default=2,
                   help="posterior samples tested against products with the Hessian itself")
    g.add_argument("--chain-kernels", default="gpcn,pcn", help="the chains of the stage chain")
    g.add_argument("--chain-steps", type=int, default=200, help="steps of a chain, earlier runs' included")
    g.add_argument("--chain-s", type=float, default=0.05, help="step of gpCN")
    g.add_argument("--pcn-s", type=float, default=0.004, help="step of pCN")
    g = p.add_argument_group("run")
    g.add_argument("--out", default="out/run", help="the folder of the run")
    g.add_argument("--stages", default="data,psf,map,post", help="of " + ", ".join(STAGES) + ", with commas")
    g.add_argument("--layout", choices=("farm", "decomposed"), default="farm",
                   help="farm: the whole box on every rank, the shots divided; decomposed: the box divided")
    g.add_argument("--time-limit", type=float, default=0.0,
                   help="seconds after which a stage stops at its next checkpoint (0: none)")
    g.add_argument("--verify-shots", type=int, default=2, help="shots the checks of verify are made on")
    args = p.parse_args(argv)
    unknown = [s for s in args.stages.split(",") if s and s not in STAGES]
    if unknown:
        p.error("no stage %s: %s" % (", ".join(unknown), ", ".join(STAGES)))
    if args.tag is None:
        args.tag = "%s-%s" % (args.map_solver, args.precondition)
    return args


#: options that say how a run proceeds, not what problem it solves: the others must be the
#: same when a run is continued
RUN_ONLY = ("out", "stages", "layout", "time_limit", "verify_shots", "batch", "history", "psf_step",
            "psf_rho", "psf_rho_g", "psf_teeth", "psf_filter", "psf_eta", "psf_blur", "psf_weights",
            "psf_check", "psf_amplitude", "psf_weight", "inner_precondition", "inner_rtol", "inner_maxit", "map_solver", "precondition",
            "tag", "max_iter", "rtol", "memory", "first_step", "max_step", "start_from", "max_backtrack", "cg_max", "lazy_gradient",
            "samples", "var_samples", "post_check", "chain_kernels", "chain_steps", "chain_s", "pcn_s")


class Scaled:
    """``gamma`` times a solver (times the identity without one): the initial inverse Hessian
    of an L-BFGS iteration, or with ``gamma`` left at one a preconditioner that is watched.
    ``unfinished`` counts the solves that used all of ``limit`` iterations: conjugate
    gradients that end on their limit return what they have, without a word."""

    def __init__(self, inner=None, limit=None):
        self.inner, self.gamma, self.limit, self.unfinished = inner, 1.0, limit, 0

    def solve(self, x, b):
        if self.inner is None:
            x.zero()
            x.axpy(self.gamma, b)
            return x
        its = self.inner.solve(x, b)
        if self.limit is not None and its is not None and its >= self.limit:
            self.unfinished += 1
        if self.gamma != 1.0:
            x.scale(self.gamma)
        return x


class Run:
    """What the stages share: the problem, the pool of shots, the lattice with its fields,
    the prior and the model, and the bookkeeping of the output folder.

    ``comm`` is the communicator of the parameter space: ``MPI.COMM_SELF`` on rank 0 of a
    farm, which holds the whole lattice, and the world in the decomposed layout, where
    every rank runs the stages on its block.  ``root`` reads, writes and prints.
    """

    def __init__(self, args, world, problem, pool):
        import hippymfem as hm
        import jax
        import model
        from mpi4py import MPI

        self.args, self.world, self.problem, self.pool = args, world, problem, pool
        self.comm = MPI.COMM_SELF if args.layout == "farm" else world
        self.root = world.rank == 0
        self.t_start = time.time()
        self.lat, self.lmap, self.Vm, self.rank = pool.lat, pool.lmap, pool.Vm, self.comm.rank
        self._device, self._put = hm.fem.kernel.device(), jax.device_put
        F = problem.fields(self.lat, self.rank)
        self.c_true, self.c0, self.W = F.c_true, F.c0, F.window
        #: the truth the unknown is held against: its offset inside the window
        self.q_ref = F.window * F.q_true
        #: what the data are made of: that, or the true offset everywhere
        self.q_data = F.q_true if args.full_truth else self.q_ref
        self.prior = problem.prior(self.lat, self.comm, self.lmap)
        self.model = model.MarmousiModel(pool, self.prior, self.W, lazy=args.lazy_gradient)
        # the offsets a trial point of a line search may have: those whose speed the time step is
        # stable for, and any where the window keeps the offset from the speed
        inside = self.W > 0.0
        safe = np.where(inside, self.W, 1.0)
        self.bounds = tuple(self.lmap.from_lattice(np.where(inside, (c - self.c0) / safe, sign * np.inf))
                            for c, sign in zip(problem.speed_range, (-1.0, 1.0)))
        #: whether a stage stopped at the time limit: the later ones are then left for the next run
        self.stopped = False
        if self.root:
            os.makedirs(args.out, exist_ok=True)
        self.comm.Barrier()

    # ------------------------------------------------------------------ the lattice
    def device(self, a):
        """An owned block on the device, in double precision: a copy (on the host JAX would
        take a NumPy array's own memory)."""
        return self._put(np.array(a, dtype=np.float64), self._device)

    def halo(self, width):
        """This rank's halo of ``width`` nodes, on the device."""
        from lattice import DeviceHalo, Halo

        return DeviceHalo(Halo(self.lat, self.rank, width), self.comm)

    def total(self, value):
        """A rank's number summed over the ranks of the lattice."""
        return float(value) if self.lat.size == 1 else self.comm.allreduce(float(value))

    def greatest(self, value):
        from mpi4py import MPI

        return float(value) if self.lat.size == 1 else self.comm.allreduce(float(value), op=MPI.MAX)

    def dot(self, a, b):
        return self.total(np.sum(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)))

    def norm(self, a):
        return np.sqrt(self.dot(a, a))

    def model_error(self, m):
        """``|W m - W q_true| / |W q_true|`` over the nodes: the distance of the speed from
        the true one inside the window, against that of the initial model."""
        return self.norm(self.W * self.lmap.to_lattice(m) - self.q_ref) / self.norm(self.q_ref)

    def _where(self, rank):
        return (Ellipsis,) + tuple(slice(lo, hi) for lo, hi in self.lat.owned(self.lat.block(rank)))

    def gather(self, a):
        """Owned blocks ``(..., *own_shape)`` as the whole lattice ``(..., *shape)`` on
        ``root`` (None elsewhere).  A rank's blocks travel as one object: under 2^31 bytes,
        which a stack of a hundred fields of ten million nodes a rank would not be."""
        a = np.asarray(a)
        if self.lat.size == 1:
            return a
        parts = self.comm.gather(a)
        if parts is None:
            return None
        out = np.empty(a.shape[:-3] + self.lat.shape, dtype=a.dtype)
        for r, part in enumerate(parts):
            out[self._where(r)] = part
        return out

    def scatter(self, whole):
        """The whole lattice ``(..., *shape)`` on ``root`` to this rank's owned block."""
        if self.lat.size == 1:
            return np.asarray(whole)
        parts = None
        if self.root:
            parts = [np.ascontiguousarray(whole[self._where(r)]) for r in range(self.lat.size)]
        return self.comm.scatter(parts)

    # ------------------------------------------------------------------ the folder
    def say(self, *a):
        if self.root:
            print(*a, flush=True)

    def path(self, name):
        return os.path.join(self.args.out, name)

    def have(self, name):
        return self.comm.bcast(os.path.exists(self.path(name)) if self.root else None)

    def _tmp(self, name, ending):
        """A name to write ``name`` under before it takes its place: this process's own, since
        solves with other tags may run side by side in the folder and write the same file."""
        return "%s.%s.%d.tmp%s" % (self.path(name), socket.gethostname(), os.getpid(), ending)

    def write(self, name, arr=None, **arrays):
        """An array into ``name`` (``.npy``), or several (``.npz``), in one step: a run killed
        while it writes leaves the file as it was."""
        if self.root:
            tmp = self._tmp(name, "." + name.rsplit(".", 1)[1])
            np.save(tmp, arr) if arr is not None else np.savez(tmp, **arrays)
            os.replace(tmp, self.path(name))

    def save(self, name, a):
        """A field of owned blocks, in the lattice's order whatever the ranks."""
        self.write(name, self.gather(a))

    def load(self, name):
        return self.scatter(np.load(self.path(name)) if self.root else None)

    def remove(self, name):
        if self.root and os.path.exists(self.path(name)):
            os.remove(self.path(name))

    def dump(self, name, obj):
        if self.root:
            tmp = self._tmp(name, "")
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=1)
            os.replace(tmp, self.path(name))

    def read(self, name):
        def get():
            with open(self.path(name)) as f:
                return json.load(f)

        return self.comm.bcast(get() if self.root else None)

    def out_of_time(self):
        """Whether the time limit has passed, by ``root``'s clock: one answer for every rank."""
        lim = self.args.time_limit
        late = (bool(lim) and (time.time() - self.t_start) > lim) if self.root else None
        return self.comm.bcast(late)

    # ------------------------------------------------------- point spread functions
    def psf_key(self):
        """What the products with the combs depend on, as a part of their files' names."""
        a = self.args
        key = "s%dr%dt%d" % (a.psf_step, a.psf_rho, a.psf_teeth)
        key += "f%g" % a.psf_filter if a.psf_filter else ""
        # (the comb products do not depend on the amplitude, but the operator's files and the
        # histories that name them do: a folder keeps both side by side)
        return key + ("a%g" % a.psf_amplitude if a.psf_amplitude else "")

    def sample_points(self):
        """The sample points, in global nodes ``[k, j, i]``: where the window is one and at
        least the mask's radius from where it is not (an impulse response is that of the
        Hessian without the window, which stands on both sides of the operator), as many a
        direction as fit, centred."""
        import marmousi
        import psf

        a, P = self.args, self.problem
        x, y, z = marmousi.nodes(P.box, P.h)
        mid = [0.5 * (c[0] + c[-1]) for c in (x, y, z)]
        # the window is a product of a factor a direction: along the three lines through the middle
        lines = (marmousi.window(mid[0], mid[1], z, P.box, P.margins),
                 marmousi.window(mid[0], y, mid[2], P.box, P.margins),
                 marmousi.window(x, mid[1], mid[2], P.box, P.margins))
        constraint, first = [], []
        for w, axis in zip(lines, "zyx"):
            one = np.nonzero(w >= 1.0 - 1e-12)[0]
            lo, hi = (int(one[0]) + a.psf_rho, int(one[-1]) - a.psf_rho) if one.size else (1, 0)
            if lo > hi:
                where = "the nodes %d to %d" % (one[0], one[-1]) if one.size else "no node"
                raise SystemExit("marmousi: no sample point along %s: the window is one on %s and the mask's "
                                 "radius is %d nodes (--psf-rho, --margins)" % (axis, where, a.psf_rho))
            constraint.append((lo, hi))
            first.append(lo + ((hi - lo) % a.psf_step) // 2)
        return psf.combs(self.lat.shape, a.psf_step, first, a.psf_teeth, constraint)

    def psf_halos(self, sample):
        """``(build, apply)``: the halo the kernels are built through, wide enough for every
        rank to compute those of its own cells' corners (and for the filter), and the
        halo of a product, the kernels' radius."""
        import psf

        a, lat = self.args, self.lat
        blocks = [lat.owned(lat.block(r)) for r in range(lat.size)]
        wide = np.maximum(psf.build_width(sample, blocks, a.psf_rho, a.psf_weights), a.psf_rho)
        if a.psf_filter:
            wide = np.maximum(wide, int(4.0 * a.psf_filter + 0.5))
        if a.psf_amplitude:
            wide = np.maximum(wide, psf.amplitude_width(3, a.psf_amplitude)[1])
        rho_g = a.psf_rho if a.psf_rho_g is None else a.psf_rho_g
        return self.halo(tuple(int(w) for w in wide)), self.halo(rho_g)

    def amplitude(self, field=None):
        """The local size ``a`` of the Hessian on this rank's block (``--psf-amplitude``; None
        without it): ``field``, which the stage psf made at the starting point and which is
        saved here, or what that stage left in the folder.  The operator at any other point
        (the stage post) takes the same one: it is a positive field on both sides of ``S
        S^T``, and what the size has changed by since stays in the kernels."""
        if not self.args.psf_amplitude:
            return None
        name = "psf_start_" + self.psf_key() + "_amplitude.npy"
        if field is not None:
            self._amplitude = np.asarray(field, dtype=np.float64)
            self.save(name, self._amplitude)
        elif getattr(self, "_amplitude", None) is None:
            if not self.have(name):
                raise SystemExit("marmousi: --psf-amplitude needs the stage psf of these settings (%s) in "
                                 "%s: it makes the amplitude" % (self.psf_key(), self.args.out))
            self._amplitude = self.load(name)
        return self._amplitude

    def kernels(self, responses, sample=None, halos=None):
        """``(kernels, halo)`` from the products of the Hessian with the combs."""
        import psf

        a = self.args
        sample = self.sample_points() if sample is None else sample
        build, apply = self.psf_halos(sample) if halos is None else halos
        amp = self.amplitude()
        more = {} if amp is None else dict(amplitude=amp)
        kern = psf.build(responses, sample, build, a.psf_rho, a.psf_rho_g, a.psf_weights,
                         filter_sigma=a.psf_filter, window=self.W if amp is None else self.W * amp,
                         comm=self.comm if self.lat.size > 1 else None, **more)
        return kern, apply

    def operator(self, kern, halo, eta=None):
        """``H~ = W S S^T W`` of the kernels, with the low-frequency term of ``--psf-eta``; with
        ``--psf-amplitude`` ``W a S S^T a W`` (the kernels are then those of unit size)."""
        import psf

        a = self.args
        blur = 0.25 * kern.rho_g if a.psf_blur is None else a.psf_blur
        amp = self.amplitude()
        # --psf-weight w: H~ = W a S S^T a W times w, as the taper times the square root of w
        win = (self.W if amp is None else self.W * amp) * float(np.sqrt(a.psf_weight))
        return psf.PSFOperator(halo, kern, window=win, eta=a.psf_eta if eta is None else eta, blur_sigma=blur)

    def approximate_hessian(self, op, precondition=None):
        """``H~ + R`` with its solve, on the lattice and as an operator of the library."""
        import psf

        a = self.args
        return psf.ApproximateHessian(op, self.prior, precondition or a.inner_precondition,
                                      rtol=a.inner_rtol, maxit=a.inner_maxit, lmap=self.lmap)

    def by_depth(self, a):
        """The sums of an owned block over the lattice planes of equal depth, one a plane of the
        whole lattice, over all ranks."""
        (lo, hi), _, _ = self.lat.owned(self.lat.block(self.rank))
        out = np.zeros(self.lat.shape[0])
        out[lo:hi] = np.asarray(a, dtype=np.float64).sum(axis=(1, 2))
        if self.lat.size > 1:
            mine, out = out, np.empty_like(out)
            self.comm.Allreduce(mine, out)
        return out

    def mark(self, m):
        """Two numbers that tell one point from another: sums over the nodes of its offset."""
        q = self.model.offset(m)
        return [self.total(np.sum(q)), self.total(np.sum(q * q))]

    def products(self, name, m, dirs, what):
        """``sum over the shots of H0 d`` for the directions ``dirs`` (owned blocks) at the
        point ``m``, ``(len(dirs), *own_shape)``, or None where the time limit stopped it.

        The shots are taken as many at a time as the pool works on at once, and the sum
        so far is saved after each such piece (``name_partial.npz``) with a mark of the
        directions and the point it belongs to: a run that is stopped goes on from there."""
        pool, nshots = self.pool, len(self.problem.shots)
        dirs = [np.asarray(d, dtype=np.float64) for d in dirs]
        q = self.model.offset(m)
        # (sums over the nodes: the same numbers on any number of ranks, to round-off)
        mark = [self.total(f(v)) for v in dirs + [q] for f in (np.sum, lambda v: np.sum(v * v))]
        done, acc = 0, np.zeros((len(dirs),) + self.lmap.own_shape)
        partial = name + "_partial.npz"
        if self.have(partial):
            z = np.load(self.path(partial)) if self.root else None
            was = self.comm.bcast((int(z["done"]), z["mark"].tolist(), int(z["shots"])) if self.root
                                  else None)
            if was[2] == nshots and len(was[1]) == len(mark) and np.allclose(was[1], mark, 1e-9, 0.0):
                done, acc = was[0], self.scatter(z["sum"] if self.root else None)
                self.say("%s: going on after %d of %d shots" % (what, done, nshots))
            else:
                self.say("%s: the unfinished products on file are of other directions or of another "
                         "point; taken again" % what)
        t0 = time.time()
        while done < nshots:
            piece = range(done, min(done + pool.width, nshots))
            acc += pool.hessian(q, dirs, piece)
            done = piece.stop
            self.write(partial, done=done, mark=np.array(mark), shots=nshots, sum=self.gather(acc))
            self.say("  %s: %d of %d shots, %d directions, %.1f s"
                     % (what, done, nshots, len(dirs), time.time() - t0))
            # (after a piece, not before it: a run gets somewhere whatever its limit)
            if done < nshots and self.out_of_time():
                self.stopped = True
                self.say("%s: stopped at the time limit after %d of %d shots" % (what, done, nshots))
                return None
        return acc


def _delta(after, before):
    return {k: after[k] - before[k] for k in after}


def _onto(base, c):
    """The work of this run on top of what the runs before it had done."""
    return {k: base.get(k, 0) + v for k, v in c.items()}


# ----------------------------------------------------------------------- stages
def stage_data(R):
    """The truth's records with noise, a file a shot, and the models on the lattice."""
    import marmousi

    a, P, pool = R.args, R.problem, R.pool
    folder = R.path("data")
    if R.have("data.json"):
        rec = R.read("data.json")
        pool.load_data(folder, rec["noise_std"])
        R.say("data: read %d shots from %s (noise %.3e)" % (rec["shots"], folder, rec["noise_std"]))
        return rec
    if "data" not in a.stages.split(","):
        raise SystemExit("marmousi: %s holds no data: run the stage data first" % a.out)
    t0, c0 = time.time(), R.model.counters()
    if R.root:
        os.makedirs(folder, exist_ok=True)
    total, count = pool.simulate(R.q_data)
    rms = np.sqrt(total / count)
    at_truth = pool.observe(a.noise * rms, a.seed, folder)
    R.save("truth.npy", R.c_true)
    R.save("initial.npy", R.c0)
    R.save("window.npy", R.W)
    x, y, z = marmousi.nodes(P.box, P.h)
    R.write("survey.npz", x=x, y=y, z=z, sources=P.sources, receivers=P.receivers, t=P.t, wavelet=P.wavelet,
            shots=np.array([s[0] for s in P.shots]))
    rec = dict(shots=len(P.shots), samples=int(count), rms=float(rms), noise=a.noise,
               noise_std=float(a.noise * rms), seed=a.seed, misfit_at_truth=at_truth,
               truth="everywhere" if a.full_truth else "inside the window",
               offset_rms=R.norm(R.q_ref) / np.sqrt(R.total(np.count_nonzero(R.W))),
               seconds=time.time() - t0, **_delta(R.model.counters(), c0))
    R.dump("data.json", rec)                      # last: a later run looks for this one
    R.say("data: %d shots x %d receivers x %d steps, root mean square %.3e, noise %.3e (%g of it), misfit of "
          "the truth %.6e; the true offset inside the window has the root mean square %.1f m/s; %.1f s "
          "(%.1f s in %d sweeps)"
          % (rec["shots"], P.receivers.shape[0], P.nsteps, rms, rec["noise_std"], a.noise, at_truth,
             rec["offset_rms"], rec["seconds"], rec["wave_seconds"], rec["sweeps"]))
    return rec


def stage_verify(R, data):
    """Differences against derivatives, on ``--verify-shots`` shots: the gradient at a
    generic point with the noisy data, the symmetry of the Hessian without the window, the
    pieces of a Hessian action against the batched product, and, with data made at the
    point itself so that the Gauss-Newton Hessian is the Hessian, differences of the
    gradient and the library's ``modelVerify``."""
    import hippymfem as hm
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    a, model, prior, pool = R.args, R.model, R.prior, R.pool
    t0, c0 = time.time(), model.counters()
    nshots = len(R.problem.shots)
    shots = sorted(set(int(s) for s in np.rint(np.linspace(0, nshots - 1, max(a.verify_shots, 1)))))
    lazy, model.shots = model.lazy, shots
    hm.parRandom.set_seed(a.seed + 7)
    m0, h, k = (prior.sample(add_mean=False) for _ in range(3))
    out = dict(shots=shots)

    def point(m, with_gradient):
        x = model.generate_vector()
        x[PARAMETER] = m
        model.lazy = not with_gradient
        model.solveFwd(x[STATE], x)
        if with_gradient:
            model.solveAdj(x[ADJOINT], x)
        return x

    x = point(m0, True)
    J = model.cost(x)
    g = R.Vm.vector()
    model.evalGradientParameter(x, g)
    R.say("verify: shots %s; at a sample of the prior J = %.6e (misfit %.6e)" % (shots, J[0], J[2]))
    rows = []
    for name, d in (("h", h), ("k", k)):
        gd = g.inner(d)
        for eps in (1.0, 0.1, 0.01):
            c = [model.cost(point(m0.copy().axpy(s * eps, d), False))[0] for s in (1.0, -1.0)]
            fd = (c[0] - c[1]) / (2.0 * eps)
            rows.append(dict(direction=name, eps=eps, difference=fd, derivative=gd,
                             error=abs(fd - gd) / abs(gd)))
            R.say("   (g, %s) = %.10e; eps %.0e: central difference %.10e, relative error %.2e"
                  % (name, gd, eps, fd, rows[-1]["error"]))
    out["gradient"] = rows
    # the Hessian of the misfit without the window, on two white fields
    u, v = (np.asarray(prior.noise()) for _ in range(2))
    Hu, Hv = pool.hessian(model.offset(m0), [u, v], shots)
    uv, vu = R.dot(Hu, v), R.dot(Hv, u)
    out["symmetry"] = abs(uv - vu) / abs(uv)
    # the five pieces the library's reduced Hessian calls, against the batched product
    model.setPointForHessianEvaluations(x)
    both = model.hessian_misfit([h, k])
    y = R.Vm.vector()
    hm.ReducedHessian(model, misfit_only=True).mult(h, y)
    out["pieces"] = y.copy().axpy(-1.0, both[0]).norm("l2") / both[0].norm("l2")
    R.say("verify: (H0 u, v) = %.10e, (H0 v, u) = %.10e, relative difference %.1e; the library's reduced "
          "Hessian against the batched product %.1e" % (uv, vu, out["symmetry"], out["pieces"]))
    # data made at the point itself: the residual vanishes there and Gauss-Newton is the Hessian
    pool.simulate(model.offset(m0), shots)
    pool.observe(data["noise_std"], None, None, shots)
    Hh = model.hessian_misfit([h])[0]
    rows = []
    for eps in (1.0, 0.1, 0.01):
        gs = []
        for s in (1.0, -1.0):
            xp = point(m0.copy().axpy(s * eps, h), True)
            gs.append(R.Vm.vector())
            model.evalGradientParameter(xp, gs[-1], misfit_only=True)
        fd = gs[0].axpy(-1.0, gs[1]).scale(0.5 / eps)
        rows.append(dict(eps=eps, error=fd.axpy(-1.0, Hh).norm("l2") / Hh.norm("l2")))
        R.say("   eps %.0e: |difference of the gradient - H h| / |H h| = %.2e" % (eps, rows[-1]["error"]))
    out["hessian"] = rows
    model.lazy = False
    eps, err_g, err_H = hm.modelVerify(model, m0, misfit_only=True, verbose=R.root, eps=0.5 ** np.arange(4))
    out["modelVerify"] = dict(eps=eps.tolist(), gradient=err_g.tolist(), hessian=err_H.tolist())
    # the data again, and the model as it was
    pool.load_data(R.path("data"), data["noise_std"], shots)
    model.lazy, model.shots = lazy, None
    out.update(seconds=time.time() - t0, **_delta(model.counters(), c0))
    R.dump("verify.json", out)
    R.say("verify: %.1f s (%.1f s in %d sweeps)" % (out["seconds"], out["wave_seconds"], out["sweeps"]))
    return out


def psf_products(R, m, name, checks):
    """The products of ``H0`` at the point ``m`` with the combs of the sample points and, in
    the same sweeps, with ``checks`` vectors the operator is not made of (white ones, then
    samples of the prior, through the window: their products are those of ``W H0 W``).
    Returns ``(sample, halos, responses, vectors, products)`` or None at the time limit;
    read from ``name``'s files where an earlier run left them."""
    import hippymfem as hm
    import psf

    a, prior = R.args, R.prior
    sample = R.sample_points()
    halos = R.psf_halos(sample)
    hm.parRandom.set_seed(a.seed + 100)
    white = [prior.noise() for _ in range((checks + 1) // 2)]
    vectors = [np.asarray(v) for v in white + [prior.sample(prior.noise()) for _ in range(checks // 2)]]
    mine = dict(point=R.mark(m), combs=sample.ncomb, checks=checks)
    if R.have(name + "_products.json"):
        if _same_products(R.read(name + "_products.json"), mine):
            R.say("psf: read the responses of %d combs and the products of %d check vectors from %s"
                  % (sample.ncomb, checks, R.path(name + "_responses.npy")))
            return sample, halos, R.load(name + "_responses.npy"), vectors, R.load(name + "_check.npy")
        R.say("psf: the products on file are of another point, of other combs or check vectors; taken again")
    combs = psf.comb_vectors(sample, halos[0], a.psf_filter)
    dirs = [np.asarray(c, dtype=np.float64) for c in combs] + [R.W * v for v in vectors]
    acc = R.products(name, m, dirs, "psf")
    if acc is None:
        return None
    responses, products = acc[:sample.ncomb], R.W * acc[sample.ncomb:]
    R.save(name + "_check.npy", products)
    R.save(name + "_responses.npy", responses)
    R.dump(name + "_products.json", mine)                 # last: a later run looks for this one
    R.remove(name + "_partial.npz")
    return sample, halos, responses, vectors, products


def _same_products(was, now):
    """Whether the comb products on file are those a stage is about to take: of the same point
    (two sums over its nodes, which agree to round-off on any number of ranks) and as many."""
    return (was["combs"] == now["combs"] and was["checks"] == now["checks"]
            and np.allclose(was["point"], now["point"], 1e-9, 0.0))


def psf_symbols(R, kern):
    """The largest symbol of ``H~`` at the sample points of each depth (the middle one of
    the plane) against the prior's: where the data hold the offset and where the prior
    does.  ``H~`` is ``S S^T`` at a sample point (the window is one there), so its symbol
    is the square of the kernel's.  Of the rank that writes."""
    import psf

    P, r = R.problem, R.prior.symbols(kern.box)[1]
    n = kern.g.shape[:3]

    def wavelength(at):
        """Of the wave that an entry of a symbol on the box multiplies, m (None: the mean)."""
        k = np.sqrt(sum((min(i, kern.box - i) / (kern.box * P.h)) ** 2 for i in at))
        return float(1.0 / k) if k > 0.0 else None

    out = []
    for iz in range(n[0]):
        s = np.asarray(psf.symbol(R.device(kern.g[iz, n[1] // 2, n[2] // 2]), kern.box, 3)) ** 2
        if getattr(kern, "amplitude", None) is not None:
            s = s * kern.amplitude[iz, n[1] // 2, n[2] // 2] ** 2        # (H~ is a S S^T a there)
        at, most = (np.unravel_index(np.argmax(v), s.shape) for v in (s, s / r))
        node = kern.sample.first[0] + kern.sample.step * (kern.first[0] + iz)
        out.append(dict(depth=float(P.box.origin[2] + P.h * node), symbol=float(s[at]), prior=float(r[at]),
                        wavelength=wavelength(at), ratio=float((s / r)[most]),
                        ratio_wavelength=wavelength(most)))
    return out


def _say_symbols(R, stage, rows):
    def length(v):
        return "the mean" if v is None else "the wavelength %.0f m" % v

    for row in rows:
        R.say("%s: %.0f m down the largest symbol of H~ is %.3e (at %s) where the prior's is %.3e; the "
              "largest ratio of the two is %.1f (at %s)"
              % (stage, row["depth"], row["symbol"], length(row["wavelength"]), row["prior"], row["ratio"],
                 length(row["ratio_wavelength"])))


def psf_rayleigh(R, op, vectors, products, nw):
    """What an optimizer sees of ``H~`` on vectors it was not made from: ``(v, H v) / (v, H~ v)``,
    1 for a perfect operator and above 1 where ``H~`` is too small (its inverse then steps too
    far); the same with the prior added to both, the ratio of the whole Hessians; and the first
    by depth, the sums over a lattice plane of the white vectors (``nw`` of them, the first),
    for the planes on which the window is one somewhere.  Too small under the sea bed and too
    large at the bottom is the size that the sample points do not follow
    (``--psf-amplitude``)."""
    prior = R.prior
    out = dict(misfit=[], whole=[])
    top, below = np.zeros(R.lat.shape[0]), np.zeros(R.lat.shape[0])
    for i, (v, Hv) in enumerate(zip(vectors, products)):
        ours, Rv = np.asarray(op.H(v)), np.asarray(prior.apply_R(v))
        v, Hv = np.asarray(v), np.asarray(Hv)
        a, b, c = R.dot(v, Hv), R.dot(v, ours), R.dot(v, Rv)
        out["misfit"].append(a / b)
        out["whole"].append((a + c) / (b + c))
        if i < nw:
            top, below = top + R.by_depth(v * Hv), below + R.by_depth(v * ours)
    one = R.by_depth(R.W >= 1.0 - 1e-12) > 0.0
    z = R.problem.box.origin[2] + R.problem.h * np.arange(R.lat.shape[0])
    if nw and one.any() and np.all(below[one] > 0.0):
        out["depth"], out["by_depth"] = z[one].tolist(), (top[one] / below[one]).tolist()
    return out


def stage_psf(R):
    """The point spread function operator at the starting point, and what it is worth."""
    import psf

    a, prior, model = R.args, R.prior, R.model
    name = "psf_start_" + R.psf_key()
    t0, c0 = time.time(), model.counters()
    got = psf_products(R, prior.mean, name, a.psf_check)
    if got is None:
        return None
    sample, halos, responses, vectors, products = got
    if a.psf_amplitude:
        # the Hessian's size from the white check vectors' products, which are W H0 W v
        white = (len(vectors) + 1) // 2
        if not white:
            raise SystemExit("marmousi: --psf-amplitude takes the Hessian's size from the white check "
                             "vectors: --psf-check 1 at least")
        amp = R.amplitude(psf.amplitude(halos[0], vectors[:white], products[:white], R.W, a.psf_amplitude,
                                        comm=R.comm if R.lat.size > 1 else None))
        inside = R.W >= 1.0 - 1e-12
        R.say("psf: the size a(x) of the Hessian from %d white vector%s, smoothed over %g nodes across and "
              "%.2g in depth: from %.3e to %.3e where the window is one"
              % (white, "" if white == 1 else "s", a.psf_amplitude, psf.amplitude_width(3, a.psf_amplitude)[0][0],
                 -R.greatest(-np.where(inside, amp, np.inf).min()), R.greatest(np.where(inside, amp, 0.0).max())))
    t1 = time.time()
    kern, halo = R.kernels(responses, sample, halos)
    t_build = time.time() - t1
    # (the kernels of the cells of the rank that writes: all of them where it holds the lattice)
    R.write(name + "_kernels.npz", g=kern.g, psf=kern.psf, first=np.array(kern.first),
            sample_first=np.array(sample.first), sample_K=np.array(sample.K), step=sample.step, box=kern.box)
    rec = dict(settings=dict(step=a.psf_step, rho=a.psf_rho, rho_g=kern.rho_g, teeth=a.psf_teeth,
                             filter=a.psf_filter, eta=a.psf_eta, weights=a.psf_weights,
                             sample_first=list(sample.first), sample_points=list(sample.K), box=kern.box),
               combs=sample.ncomb, checks=len(vectors), smax=kern.smax)
    if a.psf_amplitude:
        rec["settings"]["amplitude"] = a.psf_amplitude
    # the errors on vectors the operator was not made from, without and with the low-frequency term
    nw = (len(vectors) + 1) // 2
    vs, Hs = [R.device(v) for v in vectors], [R.device(p) for p in products]
    ops = {"eta 0": R.operator(kern, halo, 0.0)}
    if a.psf_eta:
        ops["eta %g" % a.psf_eta] = R.operator(kern, halo)
    rec["errors"], rec["rayleigh"] = {}, {}
    for label, op in ops.items():
        e = psf.check_action(op.H, vs, Hs, halo.dot)
        rec["errors"][label] = dict(white=e[:nw], prior=e[nw:])
        R.say("psf: |H~ v - H v| / |H v| with %s: %s on white vectors%s, %s on samples of the prior"
              % (label, ", ".join("%.3f" % v for v in e[:nw]) or "-",
                 " (which the amplitude is made of)" if a.psf_amplitude and nw else "",
                 ", ".join("%.3f" % v for v in e[nw:]) or "-"))
        ray = rec["rayleigh"][label] = psf_rayleigh(R, op, vs, Hs, nw)
        depth = ""
        if "by_depth" in ray:
            lo, hi = int(np.argmin(ray["by_depth"])), int(np.argmax(ray["by_depth"]))
            depth = ("; by depth on the white ones from %.2f (%.0f m down) to %.2f (%.0f m)"
                     % (ray["by_depth"][lo], ray["depth"][lo], ray["by_depth"][hi], ray["depth"][hi]))
        R.say("psf: (v, H v) / (v, H~ v) with %s: %s on white vectors, %s on samples of the prior (with the "
              "prior added to both: %s)%s"
              % (label, ", ".join("%.2f" % v for v in ray["misfit"][:nw]) or "-",
                 ", ".join("%.2f" % v for v in ray["misfit"][nw:]) or "-",
                 ", ".join("%.2f" % v for v in ray["whole"]), depth))
    op = ops["eta %g" % a.psf_eta if a.psf_eta else "eta 0"]
    u = R.device(R.W)
    op.H(u)                                        # (compiled by now, whatever was checked)
    t1 = time.time()
    for _ in range(3):
        op.H(u)
    t_apply = (time.time() - t1) / 3.0
    rec["memory"] = op.memory()
    rec["symbols"] = psf_symbols(R, kern) if R.root else []
    _say_symbols(R, "psf", rec["symbols"])
    # A solve with H~ + R, by each preconditioner of its conjugate gradients.  The right-hand side
    # is the Hessian's own product with a sample of the prior (the gradient of a model
    # perturbation), so that the first solve also says what (H~ + R)^-1 is worth as an inverse of
    # H + R: it should give the sample back.  PSF_TEST_ITERATIONS bound each: a count for the
    # record, not a solve that anything waits for.
    rec["inner"] = {}
    if len(vectors) > nw:
        v = vs[nw]
        b = Hs[nw] + prior.apply_R(v)
        for kind in ("prior", "symbol", "split", "none"):
            hess = R.approximate_hessian(op, kind)
            t1 = time.time()
            x, its = hess.solve_lattice(b, maxit=min(a.inner_maxit, PSF_TEST_ITERATIONS))
            res = hess.apply(x) - b
            rec["inner"][kind] = dict(iterations=int(its), seconds=time.time() - t1,
                                      residual=float(np.sqrt(halo.dot(res, res) / halo.dot(b, b))))
            if kind == "prior":
                rec["inverse_error"] = float(np.sqrt(halo.dot(x - v, x - v) / halo.dot(v, v)))
        R.say("psf: (H~ + R) x = b to %.0e: %s" % (a.inner_rtol, ", ".join(
            "%s %d iterations%s (%.1f s)" % (k, t["iterations"], "" if t["residual"] < 2.0 * a.inner_rtol
                                             else ", not there: %.1e" % t["residual"], t["seconds"])
            for k, t in rec["inner"].items())))
        R.say("psf: |(H~ + R)^-1 (H + R) v - v| / |v| = %.3f on a sample of the prior" % rec["inverse_error"])
    rec.update(seconds=dict(all=time.time() - t0, build=t_build, product=t_apply),
               **_delta(model.counters(), c0))
    R.dump(name + ".json", rec)
    R.say("psf: %d sample points (%s along z, y, x), %d combs and %d check vectors; kernels in %.1f s, a "
          "product with H~ %.1f ms; %.1f s (%.1f s in %d sweeps)"
          % (int(np.prod(sample.K)), " x ".join(str(n) for n in sample.K), sample.ncomb, len(vectors),
             t_build, 1e3 * t_apply, rec["seconds"]["all"], rec["wave_seconds"], rec["sweeps"]))
    return rec


def start_hessian(R):
    """``H~ + R`` at the starting point, from the responses the stage psf saved."""
    name = "psf_start_" + R.psf_key()
    if not R.have(name + "_products.json"):
        raise SystemExit("marmousi: --precondition psf needs the stage psf of these settings (%s) in %s"
                         % (R.psf_key(), R.args.out))
    kern, halo = R.kernels(R.load(name + "_responses.npy"))
    return R.approximate_hessian(R.operator(kern, halo))


def point_hessian(R, m, tag):
    """``H~ + R`` at the last point ``m`` of the solve ``tag``, from products of the Hessian
    there with the combs (taken once and kept in the folder, as the stage post does)."""
    got = psf_products(R, m, "psf_%s_%s" % (tag, R.psf_key()), 0)
    if got is None:
        raise SystemExit("marmousi: the time limit came during the comb products at the point of %s" % tag)
    sample, halos, responses, _, _ = got
    kern, halo = R.kernels(responses, sample, halos)
    return R.approximate_hessian(R.operator(kern, halo))


def stage_map(R, data):
    """The MAP point, from the prior's mean (the initial model), or from the last point of
    the solve ``--start-from``."""
    a = R.args
    name = "map_" + a.tag
    if R.have(name + ".json"):
        hist = R.read(name + ".json")
        if hist["solver"] != a.map_solver or hist["precondition"] != a.precondition:
            raise SystemExit("marmousi: %s in %s is a solve by %s with %s; give another --tag"
                             % (name, a.out, hist["solver"], hist["precondition"]))
    start = None
    if a.start_from:
        if a.map_solver != "lbfgs":
            raise SystemExit("marmousi: --start-from is for --map-solver lbfgs")
        if not R.have("map_%s.npy" % a.start_from):
            raise SystemExit("marmousi: --start-from %s: no such solve in %s" % (a.start_from, a.out))
        start = R.lmap.from_lattice(R.load("map_%s.npy" % a.start_from))
    if a.precondition != "psf":
        hess = None
    else:
        hess = start_hessian(R) if start is None else point_hessian(R, start, a.start_from)
    if a.map_solver == "lbfgs":
        return map_lbfgs(R, name, hess, data, start)
    return map_newton(R, name, hess, data)


def _ended(hist, max_iter):
    """Whether a solve on file is one to read: it ended, and not for want of iterations that
    this run would give it."""
    more = hist.get("reason", "").startswith(("the iterations", "Maximum number of Iteration"))
    return hist["done"] and not (more and hist["iterations"][-1]["it"] < max_iter)


def _map_row(R, tag, row, g0):
    """A row of a history as a line: an L-BFGS iteration with its step, a Newton step with the
    Hessian actions so far."""
    how = "step %-8.3g" % row["step"] if "step" in row else "%4d Hessian actions" % row["cg"]
    R.say("map %s: it %3d  cost %.6e  misfit %.6e  reg %.3e  |g| %.3e (%.1e of the first)  %s  model error "
          "%.4f  %d forward solves, %.0f s"
          % (tag, row["it"], row["cost"], row["misfit"], row["reg"], row["gradient"], row["gradient"] / g0,
             how, row["model_error"], row["forward_solves"], row["seconds"]))


def _map_report(R, name, hist, x, data, t0):
    """What both solvers leave and say at their end."""
    from hippymfem.modeling.variables import PARAMETER

    first, last = hist["iterations"][0], hist["iterations"][-1]
    R.save(name + ".npy", R.lmap.to_lattice(x[PARAMETER]))
    R.dump(name + ".json", hist)
    if last.get("inner_unfinished"):
        R.say("map %s: %d of the solves with H~ + R used all %d iterations of --inner-maxit: what they "
              "returned was not the solve" % (R.args.tag, last["inner_unfinished"], R.args.inner_maxit))
    R.say("map %s: %s; %d iterations, |g| / |g0| = %.2e, misfit %.6e (%.6e at the start, %.6e at the truth), "
          "model error %.4f; %d forward solves, %d sweeps, %.1f s in this run"
          % (R.args.tag, hist["reason"], last["it"], last["gradient"] / hist["g0"], last["misfit"],
             first["misfit"], data["misfit_at_truth"], last["model_error"], last["forward_solves"],
             last["sweeps"], time.time() - t0))


def map_lbfgs(R, name, hess, data, start=None):
    """L-BFGS from the library's pieces (its two-loop recursion with Powell's damping, its
    backtracking line search), the loop here: for a record and a checkpoint of every
    iteration, a time limit, and the reference code's rescaling of the initial inverse
    Hessian, ``gamma = y.(B0 s) / y.y`` of the newest pair with ``B0`` the prior's precision
    or ``H~ + R`` (``s.y / y.y`` without one), kept as it was where it is not positive."""
    import hippymfem as hm
    from hippymfem.algorithms.linesearch import armijo_backtrack
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    a, model, prior, Vm, lmap = R.args, R.model, R.prior, R.Vm, R.lmap
    B0 = {"none": None, "prior": prior.R, "psf": hess}[a.precondition]
    H0 = Scaled({"none": None, "prior": prior.Rsolver, "psf": hess}[a.precondition],
                None if hess is None else hess.maxit)
    op = hm.BFGS_operator()
    op.parameters["memory_limit"] = a.memory
    op.set_H0inv(H0)
    hist = dict(solver="lbfgs", precondition=a.precondition, iterations=[], g0=None, done=False, pairs=[],
                gamma=1.0, gamma_kept=0, pairs_skipped=0, psf=R.psf_key() if hess is not None else None)
    x = model.generate_vector()
    pairs = name + "_pairs"
    t0, c0 = time.time(), model.counters()
    if R.have(name + ".json"):
        hist = R.read(name + ".json")
        it = hist["iterations"][-1]["it"]
        z = R.load("%s_it%05d.npy" % (name, it))
        x[PARAMETER] = lmap.from_lattice(z[0])
        x[STATE].m, x[STATE].g, x[STATE].misfit = x[PARAMETER].copy(), lmap.from_lattice(z[1]), hist["misfit"]
        for i in hist["pairs"]:
            s, y = (lmap.from_lattice(R.load(os.path.join(pairs, "%s_%05d.npy" % (c, i)))) for c in "sy")
            op.S.append(s)
            op.Y.append(y)
            op.R.append(1.0 / s.inner(y))
        H0.gamma = hist["gamma"]
        if _ended(hist, a.max_iter):
            R.say("map %s: read the solve of %d iterations (%s)" % (a.tag, it, hist["reason"]))
            return hist
        R.say("map %s: going on after %d iterations with %d pairs" % (a.tag, it, len(op.S)))
        fresh = False
    else:
        if R.root:
            os.makedirs(R.path(pairs), exist_ok=True)
        x[PARAMETER] = prior.mean.copy() if start is None else start.copy()
        model.solveFwd(x[STATE], x)
        it, fresh = 0, True
    cost = model.cost(x)
    if not np.isfinite(cost[0]):
        raise RuntimeError("the wave field is not finite at the point the solve starts from")
    base = hist["iterations"][-1] if hist["iterations"] else {}
    mg, mg_old, mhat, m_old = (Vm.vector() for _ in range(4))
    x_star = [model.generate_vector(STATE), model.generate_vector(PARAMETER), None]
    alpha, backtracks, hist["done"] = base.get("step", 0.0), base.get("backtracks", 0), False

    def inner():
        return hess.counts["iterations"] if hess is not None else 0

    def record(gn):
        row = dict(it=it, cost=cost[0], misfit=cost[2], reg=cost[1], gradient=gn, step=alpha,
                   backtracks=backtracks, gamma=H0.gamma, pairs=len(op.S),
                   model_error=R.model_error(x[PARAMETER]),
                   inner_iterations=base.get("inner_iterations", 0) + inner(),
                   inner_unfinished=base.get("inner_unfinished", 0) + H0.unfinished,
                   seconds=base.get("seconds", 0.0) + time.time() - t0,
                   **_onto(base, _delta(model.counters(), c0)))
        hist["iterations"].append(row)
        hist.update(misfit=cost[2], gamma=H0.gamma)
        # the iterate and its gradient under the iteration's own name, then the history that names
        # it: a run killed between the two goes on from the iteration before
        R.save("%s_it%05d.npy" % (name, it),
               np.stack([lmap.to_lattice(x[PARAMETER]), lmap.to_lattice(x[STATE].g)]))
        R.dump(name + ".json", hist)
        R.remove("%s_it%05d.npy" % (name, it - 1))
        R.save(name + ".npy", lmap.to_lattice(x[PARAMETER]))
        _map_row(R, a.tag, row, hist["g0"])

    reason = None
    while reason is None:
        model.solveAdj(x[ADJOINT], x)
        gn = model.evalGradientParameter(x, mg)
        # (an iteration that was read from the folder is not written again, and the time limit is
        # not asked before this run has made one of its own: it gets somewhere whatever its limit)
        made = fresh
        if fresh:
            if it == 0:
                hist["g0"] = gn
            else:
                # the step that was taken (a trial point may have been clipped to the speeds allowed)
                s = x[PARAMETER].copy().axpy(-1.0, m_old)
                y = mg.copy().axpy(-1.0, mg_old)
                if B0 is None:
                    gamma = s.inner(y) / y.inner(y)
                else:
                    Bs = Vm.vector()
                    B0.mult(s, Bs)
                    gamma = y.inner(Bs) / y.inner(y)
                if gamma > 0.0 and np.isfinite(gamma):
                    H0.gamma = gamma
                else:
                    hist["gamma_kept"] += 1
                try:
                    op.update(s, y)
                    for c, v in (("s", op.S[-1]), ("y", op.Y[-1])):
                        R.save(os.path.join(pairs, "%s_%05d.npy" % (c, it)), lmap.to_lattice(v))
                    hist["pairs"] = (hist["pairs"] + [it])[-len(op.S):]
                except FloatingPointError:
                    hist["pairs_skipped"] += 1         # no curvature along the step: the pair is left out
            record(gn)
            # (the files of the pairs that the memory has let go of, now that the history says so)
            for old in [i for i in range(max(it - a.memory - 1, 0), it) if i not in hist["pairs"]]:
                for c in "sy":
                    R.remove(os.path.join(pairs, "%s_%05d.npy" % (c, old)))
        fresh = True
        if it > 0 and gn <= a.rtol * hist["g0"]:
            reason = "the gradient fell by --rtol"
        elif it >= a.max_iter:
            reason = "the iterations of --max-iter ran out"
        elif made and R.out_of_time():
            R.stopped = True
            R.say("map %s: stopped at the time limit after %d iterations" % (a.tag, it))
            return hist
        if reason is not None:
            break
        op.solve(mhat, mg)
        mhat.scale(-1.0)
        alpha = 1.0
        if not op.S:
            # no pair yet to take a scale from: the first trial step moves the speed by --first-step
            # at most, and where there is no preconditioner either that is the identity's scale
            scale = a.first_step / max(R.greatest(np.abs(model.offset(mhat)).max()), 1e-300)
            if B0 is None:
                H0.gamma *= scale
                mhat.scale(scale)
            else:
                alpha = min(1.0, scale)
        if a.max_step > 0.0:
            # every trial step moves the speed by --max-step at most
            big = R.greatest(np.abs(model.offset(mhat)).max())
            if alpha * big > a.max_step:
                alpha = a.max_step / big
        mg_old.assign(mg)
        m_old.assign(x[PARAMETER])
        gdm = mg.inner(mhat)
        if not gdm < 0.0:
            reason = "the direction does not descend"
            break
        ok, alpha, backtracks, trial, _ = armijo_backtrack(
            model, x, x_star, mhat, gdm, cost[0], alpha=alpha, max_backtracking=a.max_backtrack,
            bounds=R.bounds)
        if not ok:
            reason = "the line search found no decrease"
            break
        cost, it = trial, it + 1
    hist.update(done=True, reason=reason, converged=reason.startswith("the gradient"),
                final_gradient=gn, reduction=gn / hist["g0"])
    _map_report(R, name, hist, x, data, t0)
    return hist


def map_newton(R, name, hess, data):
    """The library's inexact Gauss-Newton-CG, its conjugate gradients preconditioned by what
    the model's ``Rsolver`` returns: the prior's covariance, the solve with ``H~ + R``, or
    nothing.  A row of the history is a Newton step, ``cg`` the Hessian actions so far."""
    import hippymfem as hm
    import model as model_module
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    a, model, lmap = R.args, R.model, R.lmap
    watched = None if hess is None else Scaled(hess, hess.maxit)
    model.preconditioner = {"none": model_module.Identity(), "prior": None, "psf": watched}[a.precondition]
    hist = dict(solver="newton", precondition=a.precondition, iterations=[], g0=None, done=False,
                psf=R.psf_key() if hess is not None else None)
    x = model.generate_vector()
    if R.have(name + ".json"):
        hist = R.read(name + ".json")
        x[PARAMETER] = lmap.from_lattice(R.load(name + ".npy"))
        if _ended(hist, a.max_iter):
            R.say("map %s: read the solve of %d iterations (%s)"
                  % (a.tag, hist["iterations"][-1]["it"], hist["reason"]))
            return hist
        R.say("map %s: going on after %d iterations" % (a.tag, hist["iterations"][-1]["it"]))
    base = hist["iterations"][-1] if hist["iterations"] else {}
    t0, c0 = time.time(), model.counters()
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["max_iter"] = max(a.max_iter - base.get("it", 0), 1)
    params["GN_iter"] = 10 ** 6
    params["cg_max_iter"] = a.cg_max
    params["print_level"] = -1
    params["rel_tolerance"], params["abs_tolerance"] = a.rtol, 0.0
    params["LS"]["max_backtracking_iter"] = a.max_backtrack
    if hist["g0"] is not None:
        # a solve that is continued measures its progress from the first gradient
        params["rel_tolerance"], params["abs_tolerance"] = 0.0, a.rtol * hist["g0"]
    solver = hm.ReducedSpaceNewtonCG(model, params)
    hist["done"] = False

    def record(it, x):
        # the gradient of the point the step led to (the solver's own norm is of the point before)
        model.solveAdj(x[ADJOINT], x)
        gn = model.evalGradientParameter(x, R.Vm.vector())
        if hist["g0"] is None:
            hist["g0"] = gn
        cost = model.cost(x)
        row = dict(it=it, cg=base.get("cg", 0) + int(solver.total_cg_iter), cost=cost[0], misfit=cost[2],
                   reg=cost[1], gradient=gn, model_error=R.model_error(x[PARAMETER]),
                   inner_iterations=(base.get("inner_iterations", 0)
                                     + (hess.counts["iterations"] if hess is not None else 0)),
                   inner_unfinished=(base.get("inner_unfinished", 0)
                                     + (watched.unfinished if watched is not None else 0)),
                   seconds=base.get("seconds", 0.0) + time.time() - t0,
                   **_onto(base, _delta(model.counters(), c0)))
        hist["iterations"].append(row)
        R.save(name + ".npy", lmap.to_lattice(x[PARAMETER]))
        R.dump(name + ".json", hist)
        _map_row(R, a.tag, row, hist["g0"])

    class Stop(Exception):
        pass

    def callback(it, x):
        record(base.get("it", 0) + it, x)
        if R.out_of_time():
            raise Stop()

    if hess is not None:
        # the reorthogonalized conjugate gradients take their preconditioner for one linear map:
        # the library's rule for a solve that stands there is a thousandth of their tolerance
        hess.rtol = min(hess.rtol, 1e-3 * np.sqrt(a.rtol))
    if not hist["iterations"]:
        # the row of the starting point, which the solver does not report (it finds the state
        # solved and does not propagate the shots again)
        model.solveFwd(x[STATE], x)
        record(0, x)
    solver.callback = callback
    try:
        x = solver.solve(x)
    except Stop:
        R.stopped = True
        R.say("map %s: stopped at the time limit after %d iterations"
              % (a.tag, hist["iterations"][-1]["it"]))
        return hist
    # (the gradient of the last row, the point the solve ended at: the solver's own last norm is of
    # the point one step before it)
    last = hist["iterations"][-1]["gradient"]
    hist.update(done=True, converged=bool(solver.converged), reason=solver.termination_reasons[solver.reason],
                final_gradient=last, reduction=last / hist["g0"])
    _map_report(R, name, hist, x, data, t0)
    return hist


def stage_post(R):
    """The Laplace approximation at the MAP point of ``--tag`` with the point spread
    function Hessian there: samples of prior and posterior from the same noise, their
    pointwise standard deviations, and the ice sheet application's test of the samples,
    ``x.(H + R) x / n`` with products of the Hessian itself, which is 1 on average for
    samples of the Gaussian whose precision is ``H + R``."""
    import hippymfem as hm
    import model as model_module

    a, model, prior, lmap = R.args, R.model, R.prior, R.lmap
    tag, name = a.tag, "post_" + a.tag
    if not R.have("map_%s.npy" % tag):
        raise SystemExit("marmousi: the stage post needs the MAP point of --tag %s in %s" % (tag, a.out))
    n, keep = max(a.samples, a.var_samples, a.post_check), max(a.samples, a.post_check)
    asked = dict(tag=tag, psf=R.psf_key(), eta=a.psf_eta, samples=keep, variance_samples=n,
                 checked=a.post_check, iteration=R.read("map_%s.json" % tag)["iterations"][-1]["it"])
    if R.have(name + ".json"):
        rec = R.read(name + ".json")
        if all(rec.get(k) == v for k, v in asked.items()):
            R.say("post: read the record of %d samples at iteration %d of %s" % (n, asked["iteration"], tag))
            return rec
    t0, c0 = time.time(), model.counters()
    m = lmap.from_lattice(R.load("map_%s.npy" % tag))
    got = psf_products(R, m, "psf_%s_%s" % (tag, R.psf_key()), 0)
    if got is None:
        return None
    sample, halos, responses, _, _ = got
    kern, halo = R.kernels(responses, sample, halos)
    hess = R.approximate_hessian(R.operator(kern, halo))
    post = model_module.PSFPosterior(prior, hess, m)
    draws = None
    if R.have(name + "_samples.npz"):
        # (the draws on file are of the operator at one iterate, with one low-frequency term)
        z = np.load(R.path(name + "_samples.npz")) if R.root else None
        its, was = R.comm.bcast((float(z["iterations"]), str(z["of"])) if R.root else None)
        if was == json.dumps(asked, sort_keys=True):
            draws, std = R.scatter(z["samples"] if R.root else None), R.scatter(z["std"] if R.root else None)
            R.say("post: read %d samples of prior and posterior" % keep)
    if draws is None:
        # prior and posterior from the same noise, the standard deviations as the root mean squares
        # of the draws (neither has a mean in them)
        hm.parRandom.set_seed(a.seed + 300)
        sums, draws = np.zeros((2,) + lmap.own_shape), np.zeros((2, keep) + lmap.own_shape)
        for i in range(n):
            pair = [np.asarray(v) for v in post.draw()]
            for j in range(2):
                sums[j] += pair[j] ** 2
                if i < keep:
                    draws[j, i] = pair[j]
        std, its = np.sqrt(sums / max(n, 1)), post.iterations / max(post.draws, 1)
        if post.unfinished:
            R.say("post: %d of the %d draws' solves used all %d iterations of --inner-maxit: those are not "
                  "samples of the Gaussian" % (post.unfinished, n, a.inner_maxit))
        R.write(name + "_samples.npz", samples=R.gather(draws), std=R.gather(std), iterations=its,
                of=json.dumps(asked, sort_keys=True))
    rec = dict(asked, solve_iterations=its, prior_variance=prior.variance(),
               symbols=psf_symbols(R, kern) if R.root else [])
    _say_symbols(R, "post", rec["symbols"])
    inside = R.W >= 1.0 - 1e-12
    count = max(R.total(np.count_nonzero(inside)), 1.0)
    rec["std_prior"] = float(np.sqrt(R.total(np.sum(std[0][inside] ** 2)) / count))
    rec["std_post"] = float(np.sqrt(R.total(np.sum(std[1][inside] ** 2)) / count))
    rec["std_ratio_least"] = float(-R.greatest(-(std[1][inside] / std[0][inside]).min()))
    R.say("post: %d draws, %.0f iterations a solve; root mean square of the standard deviation where the "
          "window is one: prior %.2f m/s, posterior %.2f m/s; least ratio of the two at a node %.2f"
          % (n, its, rec["std_prior"], rec["std_post"], rec["std_ratio_least"]))
    if a.post_check:
        xs = [draws[1, i] for i in range(a.post_check)]
        acc = R.products(name + "_check", m, [R.W * v for v in xs], "post")
        if acc is None:
            return None
        nodes = float(np.prod(R.lat.shape))
        true, ours, priors = [], [], []
        for i, (v, Hv) in enumerate(zip(xs, R.W * acc)):
            d, p = R.device(v), R.device(draws[0, i])
            true.append((R.dot(v, Hv) + R.dot(v, np.asarray(prior.apply_R(d)))) / nodes)
            ours.append(R.dot(v, np.asarray(hess.apply(d))) / nodes)
            priors.append(R.dot(draws[0, i], np.asarray(prior.apply_R(p))) / nodes)
        rec["quadratic_form"] = dict(hessian=true, approximation=ours, prior=priors, nodes=nodes,
                                     exact_std=float(np.sqrt(2.0 / nodes)))
        R.say("post: x.(H + R) x / n on %d samples of the posterior: %s with the Hessian's own products, %s "
              "with H~; s.R s / n on those of the prior: %s (1 +- %.4f for samples of the Gaussian that "
              "each precision defines)"
              % (len(xs), ", ".join("%.4f" % v for v in true), ", ".join("%.4f" % v for v in ours),
                 ", ".join("%.4f" % v for v in priors), rec["quadratic_form"]["exact_std"]))
        R.remove(name + "_check_partial.npz")
    rec.update(seconds=time.time() - t0, **_delta(model.counters(), c0))
    R.dump(name + ".json", rec)
    R.say("post: %.1f s (%.1f s in %d sweeps)" % (rec["seconds"], rec["wave_seconds"], rec["sweeps"]))
    return rec


def stage_chain(R):
    """Chains from the MAP point of ``--tag``: gpCN, whose reference is the Laplace
    approximation of the stage post, and pCN, whose reference is the prior.  The loop over
    the library's kernels is here, for a trace and a checkpoint: the parameter, the
    misfit and the position of the random stream, from which a chain goes on as if it had
    not stopped."""
    import hippymfem as hm
    import model as model_module

    a, model, prior, lmap = R.args, R.model, R.prior, R.lmap
    tag = a.tag
    name = "psf_%s_%s" % (tag, R.psf_key())
    if not (R.have("map_%s.npy" % tag) and R.have(name + "_products.json")):
        raise SystemExit("marmousi: the stage chain needs the stage post of --tag %s in %s" % (tag, a.out))
    m_map = lmap.from_lattice(R.load("map_%s.npy" % tag))
    if not np.allclose(R.read(name + "_products.json")["point"], R.mark(m_map), 1e-9, 0.0):
        raise SystemExit("marmousi: the MAP point of --tag %s has moved since the stage post took its comb "
                         "products: run the stage post again" % tag)
    kern, halo = R.kernels(R.load(name + "_responses.npy"))
    post = model_module.PSFPosterior(prior, R.approximate_hessian(R.operator(kern, halo)), m_map)
    lazy, model.lazy = model.lazy, True               # a chain asks for the misfit alone
    q_map, reference = model.offset(m_map), R.mark(m_map)
    out = {}
    for number, kind in enumerate(k for k in a.chain_kernels.split(",") if k):
        if kind not in ("gpcn", "pcn"):
            raise SystemExit("marmousi: no chain %r: gpcn, pcn" % kind)
        name = "chain_%s_%s" % (tag, kind)
        kernel = hm.gpCNKernel(model, post) if kind == "gpcn" else hm.pCNKernel(model)
        kernel.parameters["s"] = a.chain_s if kind == "gpcn" else a.pcn_s
        cur, prop = hm.SampleStruct(kernel), hm.SampleStruct(kernel)
        hm.parRandom.set_seed(a.seed + 400 + number)
        t0, c0 = time.time(), model.counters()
        state = dict(kernel=kind, s=kernel.parameters["s"], steps=0, accepted=0, trace=[], seconds=0.0,
                     reference=reference)
        if R.have(name + ".json"):
            state = R.read(name + ".json")
            if not np.allclose(state["reference"], reference, 1e-9, 0.0):
                raise SystemExit("marmousi: %s in %s was run about another MAP point than --tag %s has now; "
                                 "remove its files to start it again" % (name, a.out, tag))
            cur.m.assign(lmap.from_lattice(R.load("%s_step%07d.npy" % (name, state["steps"]))))
            cur.cost, cur.u.misfit, cur.u.m = state["misfit"], state["misfit"], cur.m.copy()
            hm.parRandom.seek(state["position"])
            R.say("chain %s: going on after %d steps" % (kind, state["steps"]))
        else:
            cur.m.assign(m_map)
            kernel.init_sample(cur)
            state["start_misfit"] = cur.cost
        base, first = dict(state), state["steps"]
        while state["steps"] < a.chain_steps and not R.stopped:
            accepted = kernel.sample(cur, prop)
            state["steps"] += 1
            state["accepted"] += accepted
            away = R.norm(model.offset(cur.m) - q_map) / R.norm(q_map)
            state["trace"].append(dict(misfit=cur.cost, reg=prior.cost(cur.m), accepted=int(accepted),
                                       from_map=away))
            R.stopped = R.out_of_time()
            if state["steps"] % 10 == 0 or state["steps"] == a.chain_steps or R.stopped:
                # the parameter under the step's own name, then the record that names it
                state.update(misfit=cur.cost, position=hm.parRandom.tell(),
                             seconds=base["seconds"] + time.time() - t0,
                             **_onto(base, _delta(model.counters(), c0)))
                R.save("%s_step%07d.npy" % (name, state["steps"]), lmap.to_lattice(cur.m))
                R.dump(name + ".json", state)
                R.remove("%s_step%07d.npy" % (name, first))
                first = state["steps"]
        misfits = [t["misfit"] for t in state["trace"]]
        state["summary"] = hm.chain_summary(misfits) if len(misfits) > 1 else {}
        R.dump(name + ".json", state)
        R.say("chain %s: %d steps of s = %g, %d accepted (%.0f %%); misfit %.6e at the MAP point, mean %.6e "
              "over the chain; %d forward solves, %.1f s%s"
              % (kind, state["steps"], state["s"], state["accepted"],
                 100.0 * state["accepted"] / max(state["steps"], 1), state["start_misfit"],
                 float(np.mean(misfits)) if misfits else 0.0, state.get("forward_solves", 0),
                 state["seconds"], "; stopped at the time limit" if R.stopped else ""))
        out[kind] = {k: v for k, v in state.items() if k != "trace"}
        if R.stopped:
            break
    model.lazy = lazy
    return out


def other_settings(args):
    """Where the options that define the problem differ from those of the run in ``--out``."""
    f = os.path.join(args.out, "setup.json")
    if not os.path.exists(f):
        return []
    with open(f) as fh:
        was = json.load(fh).get("args", {})
    return ["%s %r, now %r" % (k, was[k], v) for k, v in sorted(vars(args).items())
            if k not in RUN_ONLY and k in was and was[k] != v]


def run_stages(args, world, problem, pool):
    """The stages asked for, in their order, on what the folder holds."""
    R = Run(args, world, problem, pool)
    stages = [s for s in args.stages.split(",") if s]
    P, s = problem, problem.sizes()
    s.update(ranks=world.size, layout=args.layout, device=str(R._device),
             precision="fp32" if args.float32 else "fp64")
    R.say("marmousi: %s m in elements of %g m, order %d: %d elements, %d wave unknowns, %d parameters; %d "
          "ranks (%s), %s, %s"
          % (" x ".join("%g" % v for v in P.box.size), P.h, P.order, s["elements"], s["wave_unknowns"],
             s["parameters"], world.size, args.layout, s["device"], s["precision"]))
    R.say("  %d shots of %d sources, %d receivers; %.2f s in %d steps of %.3f ms; wavelet %s about %.2f Hz; "
          "prior %g m/s over %g m (%.2f nodes)"
          % (s["shots"], s["sources"], s["receivers"], s["record"], s["nsteps"], 1e3 * s["dt"], args.wavelet,
             P.fc, args.prior_sigma, args.prior_rho, args.prior_rho / P.h))
    R.dump("setup.json", dict(args=vars(args), sizes=s))
    mine, tagged = dict(args=vars(args), sizes=s), {}
    data = mine["data"] = stage_data(R)
    if "verify" in stages:
        mine["verify"] = stage_verify(R, data)
    if "psf" in stages and not R.stopped:
        mine["psf"] = stage_psf(R)
    for stage, run in (("map", lambda: stage_map(R, data)), ("post", lambda: stage_post(R)),
                       ("chain", lambda: stage_chain(R))):
        if stage in stages and not R.stopped:
            tagged[stage] = run()
    c = mine["counters"] = R.model.counters()
    mine["seconds"] = time.time() - R.t_start
    # The folder's summary: this run's records over what the folder holds NOW, the solves by
    # their tags.  Read here and not at the start: solves that ran side by side in the folder
    # have written theirs in the meantime, and a copy read at the start would put them back.
    summary = R.read("summary.json") if R.have("summary.json") else {}
    summary.update(mine)
    for stage, record in tagged.items():
        summary.setdefault(stage, {})[args.tag] = record
    R.dump("summary.json", summary)
    R.say("marmousi: %s in %.1f s; %d field-steps in %d sweeps, %.1f s in the sweeps"
          % ("stopped at the time limit (the same command goes on)" if R.stopped else "done",
             mine["seconds"], c["field_steps"], c["sweeps"], c["wave_seconds"]))
    return 0


def main(argv=None):
    args = parse(argv)
    import hippymfem as hm                                    # noqa: F401  (sets the device up)
    from mpi4py import MPI
    import mfem.par as mfem
    import model

    world = MPI.COMM_WORLD
    mfem.Hypre.Init()
    # one answer for every rank, before any of them builds or waits for anything: a folder holds
    # one problem
    diff = world.bcast(other_settings(args) if world.rank == 0 else None)
    if diff:
        if world.rank == 0:
            print("marmousi: %s holds a run with other settings (%s); give the same ones or another --out"
                  % (args.out, ", ".join(diff)), flush=True)
        return 1
    problem = model.Problem(args)
    if args.layout == "decomposed":
        return run_stages(args, world, problem, model.LocalPool(world, problem))
    pool = model.FarmPool(world, problem)
    if world.rank > 0:
        pool.serve()
        return 0
    try:
        code = run_stages(args, world, problem, pool)
    except SystemExit:
        pool.close()                  # the others wait for a command: send them home first
        raise
    pool.close()
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import sem

        sem.abort_all()
        raise

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""The inverse problem for the wave speed of a box from the records of many shots, in the
shape hIPPyMFEM's solvers expect.

The speed is :math:`c = c_0 + W m`: :math:`c_0` the initial model, :math:`W` the paper's
window (one inside, falling to zero at the sides and the bottom, zero in the water) and
:math:`m` the unknown, an offset in m/s on the lattice of the elements' vertices.  The
propagator knows nothing of the window.  Its parameter is the whole offset :math:`q = W m`,
and what it returns for :math:`q` is multiplied by :math:`W` here, so that

.. math:: J(m) = \sum_{s} \frac{1}{2\sigma^2}\,\|y_s(W m) - d_s\|^2
                 + \frac12\,\|m - \bar m\|_R^2 , \qquad H = W H_0 W ,

with :math:`y_s` the receivers' series of shot :math:`s`, :math:`d_s` its data, and
:math:`H_0` the Gauss-Newton Hessian of the misfit with respect to :math:`q`, which is what
the point spread functions are read off (:mod:`psf`).

Four things, each on top of the one before:

:class:`Problem`
    what a run is made of, from its arguments: the box, the time step, the survey, the
    wavelet, the shots, the prior's coefficients.  Numbers only, the same on every rank;
    :meth:`Problem.engine` makes a rank's propagator of them.
:class:`LocalPool`
    one propagator and a list of shots.  A point :math:`q` becomes a misfit, a gradient or
    a block of products with :math:`H_0`, shot by shot: a shot is another wavelet matrix
    over the same sources (no compilation), its forward field is rebuilt backward from
    its last two states and its values at the absorbing faces, and nothing of it is kept
    for the next shot.  On ``MPI.COMM_WORLD`` every rank takes part in every shot of a box
    cut into blocks.
:class:`FarmPool`
    the other layout, for a box that one card holds: every rank has the whole box and a
    share of the shots, rank 0 gives the commands and adds up what comes back.  No
    exchange inside a sweep, so 32 cards do 32 shots in the time of one.
:class:`MarmousiModel`
    the object the library's algorithms talk to (``hm.Model``): ``solveFwd``, ``cost``,
    the gradient, the pieces of a Gauss-Newton Hessian action, and :meth:`Rsolver`, which
    hands the Newton-CG a preconditioner of the application's own.

:class:`PSFPosterior` is the Laplace approximation with the point spread function Hessian
in the place of a low rank, with the four members the library's gpCN kernel asks for.

Nothing here is run by itself: ``run.py`` drives it.

    python run.py --help
"""

import os
import traceback

import numpy as np

import hippymfem as hm
from hippymfem.common.parvector import ParVector
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE
from mpi4py import MPI

import jax.numpy as jnp

import box as boxmod
import marmousi
import psf
from lattice import Lattice, LatticeMap
from prior import LatticePrior


def density(u, a, m, p, x, c0):
    """The acoustic wave equation of constant density at a point: ``u`` the pressure, ``a``
    its second time derivative, ``m`` the offset of the wave speed from the initial model
    ``c0`` (the whole offset: the window is the model's business), ``p`` the test
    function.  The stiffness has no coefficient, which is what lets the propagator apply
    it to a rank's block of unknowns as an array."""
    return a.val * p.val / (c0 + m.val) ** 2 + hm.inner(u.grad, p.grad)


def _numbers(text, n, what, kind=float):
    """``n`` numbers of a comma list."""
    try:
        out = tuple(kind(v) for v in str(text).split(","))
    except ValueError:
        out = ()
    if len(out) != n:
        raise SystemExit("marmousi: %s takes %d numbers with commas between them, not %r" % (what, n, text))
    return out


# ------------------------------------------------------------------------------ the problem
class Problem:
    """What a run is made of, from the parsed arguments of ``run.py`` (its groups "box and
    discretization", "experiment" and "prior").

    Attributes
    ----------
    box : box.Box
        ``--size`` in elements of ``--h``, across the section centred on ``y = 0`` and
        along it on the section's middle (``--x0`` for another place).
    h, order, dt, nsteps, t
        The lattice step, the degree of the wave field, the time step
        (``box.dt(order, CMAX, cfl)``, in closed form), the steps of ``--record`` seconds
        and their times.
    sources, receivers : ndarray, ``(n, 3)``
    shots : list of ndarray
        The source rows that fire together: one a shot, or with ``--tile`` the sources
        at the same place of every tile.
    wavelet : ndarray, ``(nsteps + 1,)``
    gamma, delta : float
        The prior's coefficients, for the pointwise standard deviation ``--prior-sigma``
        and the correlation length ``--prior-rho`` ON THE LATTICE
        (``LatticePrior.coefficients``).
    """

    def __init__(self, args):
        a = self.args = args
        size = np.array(_numbers(a.size, 3, "--size"))
        self.h, self.order = float(a.h), int(a.order)
        n = np.rint(size / self.h).astype(np.int64)
        if np.abs(n * self.h - size).max() > 1e-6 * self.h or n.min() < 1:
            raise SystemExit("marmousi: the box %s m is not a whole number of elements of %g m"
                             % (size.tolist(), self.h))
        x0 = 0.5 * (marmousi.LX - size[0]) if a.x0 is None else float(a.x0)
        self.box = boxmod.Box(n, size, (x0, -0.5 * size[1], 0.0))
        self.absorb = () if a.absorb in ("", "none") else tuple(a.absorb.split(","))
        self.margins = _numbers(a.margins, 3, "--margins")
        # the step is a bound in closed form (no power iteration, the same on every partition),
        # and the record a whole number of them
        self.dt = self.box.dt(self.order, marmousi.CMAX, a.cfl)
        self.nsteps = int(np.ceil(a.record / self.dt - 1e-9))
        self.t = self.dt * np.arange(self.nsteps + 1)
        # the band follows the mesh: an element carries the same part of a wavelength at any h
        scale = 50.0 / self.h
        self.fc = 7.0 * scale if a.fc is None else float(a.fc)
        self.bw = 1.5 * scale if a.bw is None else float(a.bw)
        self.t0 = 0.5 / scale if a.t0 is None else float(a.t0)
        self.wavelet = marmousi.wavelet(self.t, a.wavelet, self.fc, self.bw, self.t0)
        if a.tile is None:
            self.sources, self.receivers = marmousi.survey(self.box, a.source_step, a.receiver_step, a.depth)
            self.shots = [np.array([k]) for k in range(self.sources.shape[0])]
        else:
            self.sources, self.receivers, _, place = marmousi.survey(
                self.box, a.source_step, a.receiver_step, a.depth, tile=_numbers(a.tile, 2, "--tile", int))
            self.shots = [np.nonzero(place == k)[0] for k in range(int(place.max()) + 1)]
        self.gamma, self.delta = LatticePrior.coefficients(a.prior_sigma, a.prior_rho, tuple(self.box.h))
        #: the speeds a trial point may have: below what the time step is stable for (it is made for
        #: 4000 m/s at ``cfl`` of the limit, so 4 % are left), and far above zero
        self.speed_range = (0.5 * marmousi.CMIN, 0.96 * marmousi.CMAX / a.cfl)

    # ------------------------------------------------------------------ a rank's part
    def lattice(self, comm):
        """The parameter's lattice in the blocks of ``comm``'s ranks: one block on
        ``MPI.COMM_SELF``."""
        return Lattice(self.box.n, self.box.grid(comm.size))

    def fields(self, lat, rank, core="owned"):
        """``marmousi.on_lattice`` on a rank's owned (or ``closed``) block: the true and the
        initial speed, the window and their difference, arrays ``[k, j, i]``."""
        a = self.args
        B = lat.block(rank)
        return marmousi.on_lattice(self.box, self.h, block=lat.owned(B) if core == "owned" else lat.closed(B),
                                   eps=a.eps, sigma=a.initial_smooth, margins=self.margins)

    def engine(self, comm):
        """``(pmesh, Vm, col, eng, lmap)``: this rank's mesh of the box on ``comm``, the
        parameter's space, the wave field's tables, the propagator with ALL sources as the
        rows of one operator (a shot is a matrix of wavelets over them, :meth:`wavelets`)
        and the receivers, and the map between ``Vm`` and the lattice."""
        a = self.args
        lat = self.lattice(comm)
        closed = lat.closed(lat.block(comm.rank))
        # the initial speed as the trilinear interpolant of the lattice's, so that c0 + q is
        # within the lattice's speeds at every node of every element (marmousi.lattice_aux)
        aux = marmousi.lattice_aux(self.fields(lat, comm.rank, "closed").c0, self.box, self.h, block=closed)
        pmesh, Vm, col, eng = boxmod.engine(
            comm, self.box, self.order, density, aux=aux, dtype=jnp.float32 if a.float32 else jnp.float64,
            absorb=self.absorb, history=a.history, grid=lat.P, dt=self.dt, nsteps=self.nsteps)
        # before anything gives the mesh nodes: the map reads the lattice off the vertices
        lmap = LatticeMap(Vm, lat, self.box.size, self.box.origin)
        eng.set_sources(self.box.value_operator(col, self.sources, comm, self.order), self.wavelets(0))
        eng.set_receivers(self.box.value_operator(col, self.receivers, comm, self.order))
        return pmesh, Vm, col, eng, lmap

    def wavelets(self, shot):
        """The amplitudes in time of every source for shot ``shot``, ``(nsteps + 1,
        nsources)``: the wavelet in the columns of the shot's sources."""
        w = np.zeros((self.nsteps + 1, self.sources.shape[0]))
        w[:, self.shots[shot]] = self.wavelet[:, None]
        return w

    def prior(self, lat, comm, lmap):
        """The bi-Laplacian prior on the lattice, with the library's face on ``lmap``'s space."""
        return LatticePrior(lat, comm.rank, comm, tuple(self.box.h), self.gamma, self.delta, lmap=lmap)

    def sizes(self):
        """What the problem is, in numbers, for a record."""
        b = self.box
        nodes = self.order * b.n + 1
        return dict(elements=int(np.prod(b.n)), wave_unknowns=int(np.prod(nodes)),
                    parameters=int(np.prod(b.n + 1)), box_elements=[int(v) for v in b.n],
                    origin=b.origin.tolist(), h=self.h, order=self.order, dt=self.dt, nsteps=self.nsteps,
                    record=self.dt * self.nsteps, sources=int(self.sources.shape[0]),
                    receivers=int(self.receivers.shape[0]), shots=len(self.shots),
                    wavelet=dict(kind=self.args.wavelet, fc=self.fc, bw=self.bw, t0=self.t0),
                    prior=dict(sigma=self.args.prior_sigma, rho=self.args.prior_rho, gamma=self.gamma,
                               delta=self.delta, nodes_per_rho=self.args.prior_rho / self.h),
                    speed_range=list(self.speed_range))


# -------------------------------------------------------------------------------- the pools
def _shot_file(folder, shot):
    return os.path.join(folder, "shot_%05d.npy" % shot)


class LocalPool:
    """One propagator on ``comm`` and the shots ``shots`` (numbers into ``problem.shots``;
    all of them by default).

    A point ``q``, the whole offset of the speed, and the directions of a Hessian product
    are owned blocks of the lattice (NumPy or device arrays ``[k, j, i]``) or vectors of
    ``Vm``; what comes back is owned blocks, NumPy arrays: dual vectors, the sums over the
    shots.  ``shots`` in a call names some of the pool's shots (the others' turn comes in
    another call: a stage that has to stop at a time limit works through them in pieces).

    ``counts`` are the sweeps, the field-steps and the seconds in the sweeps so far.
    ``width`` is how many shots are worked on at once: one.
    """

    width = 1

    def __init__(self, comm, problem, shots=None):
        self.comm, self.problem = comm, problem
        self.shots = list(range(len(problem.shots))) if shots is None else [int(s) for s in shots]
        self.pmesh, self.Vm, self.col, self.eng, self.lmap = problem.engine(comm)
        self.lat = self.lmap.lat
        self.noise_std = None
        self._clean, self._data = {}, {}

    @property
    def counts(self):
        e = self.eng
        return dict(sweeps=int(e.n_sweeps), field_steps=int(e.n_field_steps), seconds=float(e.t_wave))

    # ---- what a call is given
    def vector(self, a):
        """A point or a direction as a vector of ``Vm``."""
        return a if isinstance(a, ParVector) else self.lmap.from_lattice(a)

    def _mine(self, shots):
        if shots is None:
            return self.shots
        asked = set(int(s) for s in shots)
        return [s for s in self.shots if s in asked]

    def _point(self, q):
        self.eng.set_parameter(self.vector(q))

    def _records(self, shot):
        """The receivers' series of a shot at the point that is set, ``(nsteps + 1, nrec)``,
        or None where the field did not stay finite: a trial point of a line search may
        have speeds the time step is not stable for, which is for the line search to hear
        of (as a misfit of ``inf``), not a failure of the run.  The propagator raises there,
        on every rank of its communicator alike."""
        self.eng.set_wavelets(self.problem.wavelets(shot))
        try:
            return self.eng.forward()
        except RuntimeError as err:
            if "not finite" not in str(err):
                raise
            return None

    def _residual(self, shot, y):
        if shot not in self._data:
            raise RuntimeError("shot %d has no data: the stage data makes them" % shot)
        return y - self._data[shot]

    # ---- data
    def simulate(self, q, shots=None):
        """The records of the pool's shots at ``q``, kept for :meth:`observe`; returns
        their sum of squares and their number (the row of time zero, where nothing has
        moved, is not counted)."""
        self._point(q)
        total, count = 0.0, 0
        for s in self._mine(shots):
            y = self._records(s)
            if y is None:
                raise RuntimeError("the wave field of shot %d is not finite: the time step exceeds the "
                                   "stability limit of this speed" % s)
            self._clean[s] = y
            total += float(np.sum(y * y))
            count += y[1:].size
        return total, count

    def observe(self, noise_std, seed=None, folder=None, shots=None):
        """The records kept by :meth:`simulate` become the data, with independent noise of
        standard deviation ``noise_std`` on every sample (none if ``seed`` is None), and
        are saved a shot a file in ``folder``.  The noise of a shot comes from a generator
        seeded by ``seed`` and the SHOT's number: the data do not depend on which rank
        made them.  Returns the misfit of the point the records were made at, half the
        sum of the squares of the noise over its variance: what a solve can come down to."""
        self.noise_std = float(noise_std)
        misfit = 0.0
        for s in self._mine(shots):
            d = self._clean.pop(s)
            if seed is not None:
                z = np.random.default_rng([int(seed), int(s)]).standard_normal(d[1:].shape)
                d[1:] += self.noise_std * z
                misfit += 0.5 * float(np.sum(z * z))
            self._data[s] = d
            if folder is not None and self.comm.rank == 0:
                # whole or not at all: a run killed while it writes leaves no cut file behind
                tmp = _shot_file(folder, s) + ".tmp.npy"
                np.save(tmp, d)
                os.replace(tmp, _shot_file(folder, s))
        return misfit

    def load_data(self, folder, noise_std, shots=None):
        """The data of the pool's shots from the files of :meth:`observe`."""
        self.noise_std = float(noise_std)
        for s in self._mine(shots):
            # (the first rank reads for all: on COMM_SELF that is the rank itself)
            first = self.comm.rank == 0
            self._data[s] = self.comm.bcast(np.load(_shot_file(folder, s)) if first else None)

    def data(self, q, noise, seed, folder=None):
        """The data of every shot at the true point ``q``: ``noise`` is the standard
        deviation as a fraction of the root mean square of the records.  Returns the
        standard deviation."""
        total, count = self.simulate(q)
        sigma = float(noise) * np.sqrt(total / max(count, 1))
        self.observe(sigma, seed, folder)
        return sigma

    # ---- a point to numbers and dual vectors
    def forward(self, q, shots=None):
        """The misfit of the shots at ``q``: ``1/2 sum (y - d)^2 / sigma^2`` over steps,
        receivers and shots (``inf`` where a field did not stay finite)."""
        self._point(q)
        total = 0.0
        for s in self._mine(shots):
            y = self._records(s)
            if y is None:
                return np.inf
            r = self._residual(s, y)
            total += 0.5 * float(np.sum(r * r)) / self.noise_std ** 2
        return total

    def cost_gradient(self, q, shots=None):
        """``(misfit, g)``, ``g`` the misfit's gradient with respect to ``q``: a shot's
        forward sweep, then its residual carried backward beside the forward field
        rebuilt from the last two states.  One call a point, where the library's order
        (the cost, then the gradient if the point is kept) would propagate every accepted
        point's forward field twice."""
        self._point(q)
        total, g = 0.0, self.Vm.vector()
        for s in self._mine(shots):
            y = self._records(s)
            if y is None:
                return np.inf, self.lmap.to_lattice(g)
            r = self._residual(s, y) / self.noise_std ** 2
            total += 0.5 * float(np.sum(r * r)) * self.noise_std ** 2
            g.axpy(1.0, self.eng.adjoint(r[None], final=self.eng.final_state())[0])
        return total, self.lmap.to_lattice(g)

    def hessian(self, q, dirs, shots=None):
        """``[H0 d for d in dirs]`` at ``q``, summed over the shots: for each shot the
        forward field with the increments of ``--batch`` directions at a time, then their
        records divided by the noise variance carried backward beside the forward field
        rebuilt from that same sweep's end.  Returns ``(len(dirs), *own_shape)``."""
        self._point(q)
        dirs = [self.vector(d) for d in dirs]
        out = [self.Vm.vector() for _ in dirs]
        each = max(1, int(self.problem.args.batch))
        for s in self._mine(shots):
            self.eng.set_wavelets(self.problem.wavelets(s))
            for a in range(0, len(dirs), each):
                dy, final = self.eng.incremental(dirs[a:a + each], final=True)
                for o, g in zip(out[a:a + each], self.eng.adjoint(dy / self.noise_std ** 2, final=final)):
                    o.axpy(1.0, g)
                del final                 # (its boundary values, before the next sweep keeps its own)
        return np.array([self.lmap.to_lattice(o) for o in out]).reshape((len(out),) + self.lmap.own_shape)


class FarmError(Exception):
    """A rank of the farm failed in a command.  Not a ``RuntimeError``: a line search takes
    that for a trial point without a solution and goes on, with ranks that have left."""


class FarmPool:
    """Every rank of ``world`` holds the whole box on its own card, a :class:`LocalPool` on
    ``MPI.COMM_SELF`` with the shots ``rank, rank + size, ...``.  Rank 0 runs the driver
    and calls the pool as it would a local one; the others call :meth:`serve` and do what
    they are told until :meth:`close`.

    A command is a small header, sent as an object (``world.bcast``), the lattice arrays
    it names (the point, the directions: ``world.Bcast``, one message an array), every
    rank's own work, and the sums of what the ranks return (``world.Reduce``): a few
    numbers, and lattice arrays, whose count the header fixes, so that no rank decides a
    size for itself.  Between the work and the sums one number is reduced with ``MAX``:
    whether a rank raised.  If one did, every rank knows before any of them waits for a
    result that will not come, rank 0 raises :class:`FarmError` with the ranks' reports,
    and the others raise too; a driver ends all ranks then (``sem.abort_all``).

    ``counts`` are the sums over the ranks; ``width`` the shots worked on at once, one a
    rank.
    """

    def __init__(self, world, problem):
        self.world, self.problem = world, problem
        self.root = world.rank == 0
        nshots = len(problem.shots)
        self.local = LocalPool(MPI.COMM_SELF, problem, range(world.rank, nshots, world.size))
        self.Vm, self.lmap, self.lat = self.local.Vm, self.local.lmap, self.local.lat
        self.shots = list(range(nshots))
        self.width = world.size
        self.noise_std = None
        self._total = np.zeros(3)
        self._closed = False

    @property
    def counts(self):
        sweeps, field_steps, seconds = self._total
        return dict(sweeps=int(sweeps), field_steps=int(field_steps), seconds=float(seconds))

    def lattice(self, a):
        """A point or a direction as a whole lattice array on the host."""
        if isinstance(a, ParVector):
            return self.lmap.to_lattice(a)
        return np.ascontiguousarray(np.asarray(a, dtype=np.float64))

    # ---- the two ends of a command
    def _call(self, command, arrays=(), **more):
        if not self.root:
            raise RuntimeError("the farm's commands are rank 0's to give")
        if self._closed:
            raise RuntimeError("the farm is closed")
        arrays = [self.lattice(a) for a in arrays]
        if any(a.shape != self.lat.shape for a in arrays):
            # before anything is sent: the others expect whole lattice arrays
            raise ValueError("the farm takes arrays of the whole lattice %s, not %s"
                             % (self.lat.shape, [a.shape for a in arrays]))
        head = dict(more, command=command, arrays=len(arrays))
        self.world.bcast(head, root=0)
        return self._run(head, arrays)

    def serve(self):
        """What a rank other than the first does: commands, until the one that closes."""
        while True:
            head = self.world.bcast(None, root=0)
            if head["command"] == "close":
                return
            self._run(head)

    def close(self):
        """Send the others home.  A driver calls it when its stages have ended, also by a
        ``SystemExit``; after an exception it ends all ranks instead."""
        if self.root and not self._closed:
            self._closed = True
            self.world.bcast(dict(command="close"), root=0)

    def _run(self, head, arrays=None):
        """A command on this rank, between the broadcast of its header and the sums."""
        world, shape = self.world, self.lat.shape
        arrays = arrays if self.root else [np.empty(shape) for _ in range(head["arrays"])]
        for a in arrays:
            world.Bcast(a, root=0)
        before = self.local.counts
        failed, report, values, fields = 0, None, (), ()
        try:
            values, fields = getattr(self, "_do_" + head["command"])(head, arrays)
        except Exception:                                    # noqa: BLE001  (reported, and raised below)
            failed, report = 1, "rank %d:\n%s" % (world.rank, traceback.format_exc())
        if world.allreduce(failed, op=MPI.MAX):
            reports = world.gather(report, root=0)
            if self.root:
                raise FarmError("the command %r failed on %d of %d ranks\n%s"
                                % (head["command"], sum(r is not None for r in reports), world.size,
                                   "\n".join(r for r in reports if r is not None)))
            raise FarmError("the command %r failed on another rank" % head["command"] if report is None
                            else report)
        after = self.local.counts
        mine = np.array([after[k] - before[k] for k in ("sweeps", "field_steps", "seconds")] + list(values),
                        dtype=np.float64)
        total = np.zeros_like(mine)
        world.Reduce(mine, total, op=MPI.SUM, root=0)
        out = []
        for f in fields:
            whole = np.zeros(shape) if self.root else None
            world.Reduce(np.ascontiguousarray(f, dtype=np.float64), whole, op=MPI.SUM, root=0)
            out.append(whole)
        if self.root:
            self._total += total[:3]
        return total[3:], out

    # ---- a rank's work on a command: (numbers, lattice arrays), their counts fixed by the header
    def _do_simulate(self, head, arrays):
        return self.local.simulate(arrays[0], head["shots"]), ()

    def _do_observe(self, head, arrays):
        return (self.local.observe(head["noise_std"], head["seed"], head["folder"], head["shots"]),), ()

    def _do_load_data(self, head, arrays):
        self.local.load_data(head["folder"], head["noise_std"], head["shots"])
        return (), ()

    def _do_forward(self, head, arrays):
        # (inf where a field did not stay finite, which the sum over the ranks keeps)
        return (self.local.forward(arrays[0], head["shots"]),), ()

    def _do_cost_gradient(self, head, arrays):
        misfit, g = self.local.cost_gradient(arrays[0], head["shots"])
        return (misfit,), (g,)

    def _do_hessian(self, head, arrays):
        return (), tuple(self.local.hessian(arrays[0], arrays[1:], head["shots"]))

    # ---- the pool, as rank 0 sees it
    @staticmethod
    def _shots(shots):
        return None if shots is None else [int(s) for s in shots]

    def simulate(self, q, shots=None):
        (total, count), _ = self._call("simulate", [q], shots=self._shots(shots))
        return float(total), int(count)

    def observe(self, noise_std, seed=None, folder=None, shots=None):
        self.noise_std = float(noise_std)
        (misfit,), _ = self._call("observe", noise_std=float(noise_std), seed=seed, folder=folder,
                                  shots=self._shots(shots))
        return float(misfit)

    def load_data(self, folder, noise_std, shots=None):
        self.noise_std = float(noise_std)
        self._call("load_data", folder=folder, noise_std=float(noise_std), shots=self._shots(shots))

    def data(self, q, noise, seed, folder=None):
        total, count = self.simulate(q)
        sigma = float(noise) * np.sqrt(total / max(count, 1))
        self.observe(sigma, seed, folder)
        return sigma

    def forward(self, q, shots=None):
        (misfit,), _ = self._call("forward", [q], shots=self._shots(shots))
        return float(misfit)

    def cost_gradient(self, q, shots=None):
        (misfit,), (g,) = self._call("cost_gradient", [q], shots=self._shots(shots))
        return float(misfit), g

    def hessian(self, q, dirs, shots=None):
        _, out = self._call("hessian", [q] + list(dirs), shots=self._shots(shots))
        return np.array(out).reshape((len(out),) + self.lat.shape)


# -------------------------------------------------------------------------------- the model
class Point:
    """What stands where the library's solvers keep a state or an adjoint vector.  The
    state of this problem is the wave fields of every shot, of which nothing is kept:
    ``misfit``, ``g`` (the misfit's gradient with respect to ``m``, where it was computed
    with the misfit), ``m`` and ``shots`` (which shots, None for all) after a forward
    solve; ``g`` after an adjoint solve; ``dm``, a direction on its way through the pieces
    of a Hessian action."""

    __slots__ = ("misfit", "g", "m", "shots", "dm")

    def __init__(self):
        for k in self.__slots__:
            setattr(self, k, None)

    def assign(self, other):
        for k in self.__slots__:
            setattr(self, k, getattr(other, k))
        return self

    def zero(self):
        for k in self.__slots__:
            setattr(self, k, None)
        return self

    def copy(self):
        return Point().assign(self)

    duplicate = copy


class MarmousiModel(hm.Model):
    r"""Cost, gradient and Gauss-Newton Hessian of the wave speed's offset ``m`` from shot
    records, over a pool of shots.

    Parameters
    ----------
    pool : LocalPool or FarmPool
        With the data of its shots (``pool.data`` or ``pool.load_data``).
    prior : a prior on ``pool.Vm`` (``prior.LatticePrior`` with the library's face)
    window : ndarray
        ``W`` on this rank's owned block of the lattice.  ``q = W m`` goes to the pool and
        ``W`` multiplies what comes back: ``H = W H0 W``.
    lazy : bool
        False (the default): a forward solve evaluates the misfit AND its gradient, a
        forward sweep and a backward sweep of two fields a shot, and ``solveAdj`` hands the
        gradient on.  True is the library's order: the misfit alone, and the gradient when
        asked, for which every shot is propagated forward again.  The first is cheaper
        when most trial points of a line search are accepted (one forward sweep a point
        less); a chain, which asks for no gradient, takes the second.

    ``shots`` (None: all) restricts every evaluation to some shots, for the checks.
    ``preconditioner``, if set, is what :meth:`Rsolver` returns: an object with ``solve(x,
    b)`` in the place of the prior's covariance as the preconditioner of Newton-CG.
    """

    def __init__(self, pool, prior, window, lazy=False, gradient_norm="M"):
        hm.Model.__init__(self, None, prior, None, gradient_norm=gradient_norm)
        self.pool, self.lmap, self.Vm = pool, pool.lmap, pool.Vm
        self.window = np.asarray(window, dtype=np.float64)
        self.lazy = bool(lazy)
        self.shots = None
        self.preconditioner = None
        self.gauss_newton_approx = True
        self._lin = None
        self.n_hess = 0

    # ------------------------------------------------------------------ vectors
    def generate_vector(self, component="ALL"):
        if component == "ALL":
            return [Point(), self.Vm.vector(), Point()]
        if component == PARAMETER:
            return self.Vm.vector()
        return Point()

    def init_parameter(self, m):
        m.zero()
        return m

    def offset(self, m):
        """``q = W m`` on the lattice: the offset of the speed that the propagator sees."""
        return self.window * self.lmap.to_lattice(m)

    def _dual(self, g):
        """``W g`` as a vector: a dual vector with respect to ``q`` to one with respect to ``m``."""
        return self.lmap.from_lattice(self.window * g)

    # ------------------------------------------------------------------- solves
    def solveFwd(self, out, x):
        """The misfit at ``x[PARAMETER]``, and its gradient unless ``lazy``.  The misfit is
        ``inf`` where a wave field did not stay finite (a speed the time step is not
        stable for): no number is less, so a line search halves its step there and a
        chain rejects the proposal, with no exception between the ranks of a farm."""
        m = x[PARAMETER]
        if self._solved(out, m):
            return out
        self.n_fwd_solve += 1
        if self.lazy:
            misfit, g = self.pool.forward(self.offset(m), self.shots), None
        else:
            misfit, g = self.pool.cost_gradient(self.offset(m), self.shots)
        out.misfit, out.m, out.shots = float(misfit), m.copy(), self.shots
        out.g = None if g is None else self._dual(g)
        return out

    def _solved(self, st, m):
        """Whether ``st`` is already the forward solve of ``m``: a solver that is handed a
        point with its state (a driver's row of the starting point) propagates every
        shot once, not twice.  One answer for every rank."""
        if st.m is None or st.misfit is None or st.shots != self.shots or (st.g is None and not self.lazy):
            return False
        same = bool(np.array_equal(st.m.array, m.array))
        comm = self.Vm.comm
        return same if comm.size == 1 else comm.allreduce(same, op=MPI.LAND)

    def solveAdj(self, out, x):
        """The misfit's gradient at ``x``: the one the forward solve made, or the sweeps
        for it now."""
        self.n_adj_solve += 1
        st = x[STATE]
        if st.g is None:
            _, g = self.pool.cost_gradient(self.offset(st.m), self.shots)
            st.g = self._dual(g)
        out.g = st.g
        return out

    # --------------------------------------------------------------------- cost
    def cost(self, x):
        """``[total, regularization, misfit]`` at ``x = [u, m, p]``."""
        reg = self.prior.cost(x[PARAMETER])
        return [x[STATE].misfit + reg, reg, x[STATE].misfit]

    def evalGradientParameter(self, x, mg, misfit_only=False):
        mg.zero()
        mg.axpy(1.0, x[ADJOINT].g)
        if not misfit_only:
            tmp = self.Vm.vector()
            self.prior.grad(x[PARAMETER], tmp)
            mg.axpy(1.0, tmp)
        g = self.Vm.vector()
        solver = self.prior.Msolver if self.gradient_norm == "M" else self.prior.Rsolver
        solver.solve(g, mg)
        return max(g.inner(mg), 0.0) ** 0.5

    # ------------------------------------------------------------------ Hessian
    def setLinearizationPoint(self, x, gauss_newton_approx=True, matrix_free=False):
        self.gauss_newton_approx = True
        self._lin = x[PARAMETER].copy()
        return self

    def setPointForHessianEvaluations(self, x, gauss_newton_approx=True, matrix_free=False):
        return self.setLinearizationPoint(x)

    def hessian_misfit(self, directions):
        """The Gauss-Newton Hessian of the misfit, ``W H0 W``, on several directions at
        once: one pair of sweeps a shot carries the forward field once for all of them.
        Returns one dual vector a direction."""
        if self._lin is None:
            raise RuntimeError("setPointForHessianEvaluations has not been called")
        dirs = [self.offset(d) for d in directions]
        self.n_hess += len(dirs)
        return [self._dual(g) for g in self.pool.hessian(self.offset(self._lin), dirs, self.shots)]

    # The five pieces ReducedHessian.GNHessian calls.  A shot's incremental sweep and the
    # backward sweep that follows it share the forward field and nothing outlives the shot,
    # so the pieces only pass the direction along and the last but one does the work.
    def applyC(self, dm, out):
        out.dm = dm.copy()
        return out

    def solveFwdIncremental(self, sol, rhs):
        self.n_inc_solve += 1
        sol.dm = rhs.dm
        return sol

    def applyWuu(self, du, out):
        out.dm = du.dm
        return out

    def solveAdjIncremental(self, sol, rhs):
        self.n_inc_solve += 1
        sol.g = self.hessian_misfit([rhs.dm])[0]
        return sol

    def applyCt(self, dp, out):
        out.zero()
        out.axpy(1.0, dp.g)
        return out

    def _second_order(self, i, j, d, out):
        raise NotImplementedError("the Marmousi model applies the Gauss-Newton Hessian only")

    def Rsolver(self, tolerance=None, cycles=None):
        """What preconditions the conjugate gradients of a Newton step: ``preconditioner``
        if one is set (the solve with the point spread function Hessian plus the prior),
        the prior's covariance otherwise."""
        if self.preconditioner is not None:
            return self.preconditioner
        return hm.Model.Rsolver(self, tolerance, cycles)

    # ----------------------------------------------------------------- counters
    def counters(self):
        """The work so far: points at which the shots were propagated, gradients, Hessian
        products, and the pool's sweeps, field-steps and seconds in them."""
        c = self.pool.counts
        return dict(forward_solves=int(self.n_fwd_solve), gradients=int(self.n_adj_solve),
                    hessian_products=int(self.n_hess), sweeps=c["sweeps"], field_steps=c["field_steps"],
                    wave_seconds=c["seconds"])


class Identity:
    """No preconditioner, as a solver: ``x = b``."""

    def solve(self, x, b):
        x.zero()
        x.axpy(1.0, b)
        return 0


# ---------------------------------------------------------------------------- the posterior
class PSFPosterior:
    r"""The Laplace approximation :math:`N(\bar m, (\tilde H + R)^{-1})` with the point spread
    function Hessian (``psf.ApproximateHessian``), with the members the library's gpCN
    kernel and a driver ask of a posterior: ``mean``, ``init_vector``, ``sample`` and
    ``Hlr`` (``inner``, ``mult``, ``solve``).

    A sample is one solve: :math:`(\tilde H + R)^{-1} (R s + W S \xi)` with :math:`s` a
    sample of the prior and :math:`\xi` white on the nodes, whose right-hand side has the
    covariance :math:`R + \tilde H` exactly because :math:`\tilde H = W S S^T W` is given as
    a square (``psf.draw``).  So the samples have the covariance of the Gaussian whose
    density ``Hlr.inner`` evaluates, to the tolerance of the solve: what the acceptance
    ratio of a gpCN chain assumes.  The second white field is drawn from the library's
    stream after the prior's, so a chain has one stream and its position resumes it.
    """

    def __init__(self, prior, hessian, mean):
        self.prior, self.Hlr, self.mean = prior, hessian, mean
        #: draws so far, the iterations of their solves, and the solves that used all they may
        self.draws, self.iterations, self.unfinished = 0, 0, 0

    def init_vector(self, x, dim):
        return self.prior.init_vector(x, dim)

    def draw(self, noise=None, rng=None):
        """``(s_prior, s_post)``: a sample of the prior and one of the posterior from the
        same noise, both without their means, owned lattice arrays on the device."""
        prior = self.prior
        s = prior.sample(prior.noise(rng=rng) if noise is None else noise)
        x, its = psf.draw(self.Hlr, s, prior.noise(rng=rng))
        self.draws += 1
        self.iterations += its
        self.unfinished += its >= self.Hlr.maxit
        return s, x

    def sample(self, noise, s_prior, s_post, add_mean=True):
        """As ``hm.GaussianLRPosterior.sample(noise, s_prior, s_post)``: ``noise`` the
        prior's white noise, a vector; fills the two samples and returns the second."""
        s, x = self.draw(self.prior.lattice(noise))
        self.prior.vector(s, s_prior)
        self.prior.vector(x, s_post)
        if add_mean:
            s_prior.axpy(1.0, self.prior.mean)
            s_post.axpy(1.0, self.mean)
        return s_post

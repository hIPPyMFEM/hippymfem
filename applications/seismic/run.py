# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Bayesian inversion of the Earth's wave speed from seismograms, end to end.

The stages, each resumable from what the previous ones saved in ``--out``:

``data``    the truth (PREM + S20RTS), its seismograms, the noisy observations;
``verify``  finite-difference checks of the gradient and of the Hessian;
``map``     the MAP point by inexact Gauss-Newton-CG from the prior mean (PREM);
``eig``     the low-rank approximation of the prior-preconditioned misfit Hessian;
``post``    pointwise variances, and samples of the prior and of the posterior.

Run ``python run.py --help`` for the options; the README has the commands of a small
run and of the paper's size.

    mpirun -n 2 python run.py --n 16 --out out/n16
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("mesh and discretization")
    g.add_argument("--n", type=int, default=16, help="elements along an edge of each of the 13 blocks' faces")
    g.add_argument("--refine", type=int, default=0, help="uniform refinements in parallel after partitioning")
    g.add_argument("--order", type=int, default=3, help="polynomial degree of the wave field")
    g.add_argument("--cube", type=float, default=1700.0, help="half-width of the central cube, km")
    g.add_argument("--cfl", type=float, default=0.8, help="fraction of the stability limit")
    g.add_argument("--float32", action="store_true", help="single precision wave fields")
    g.add_argument("--geometry-float32", action="store_true", help="geometric factors stored in single precision")
    g.add_argument("--element-chunk", type=int, default=10000,
                   help="element-fields per pass of the stiffness kernel (0: all the elements at once)")
    g.add_argument("--element-block", type=int, default=0,
                   help="elements per block of the per-element device arrays (0: one array; "
                        "524288 where a kernel cannot index one array of them, see sem.py)")
    g = p.add_argument_group("experiment")
    g.add_argument("--T", type=float, default=1000.0, help="observation window, s")
    g.add_argument("--f0", type=float, default=None,
                   help="peak frequency of the Ricker source, Hz (default: what the mesh resolves, at most 0.028)")
    g.add_argument("--ppw", type=float, default=5.0,
                   help="nodes per shortest wavelength at 2.5 f0 that the default f0 asks of the surface elements")
    g.add_argument("--nfreq", type=int, default=61, help="Fourier coefficients per receiver")
    g.add_argument("--nlat", type=int, default=10)
    g.add_argument("--nlon", type=int, default=10)
    g.add_argument("--noise", type=float, default=9.34e-3, help="noise std, the largest datum being 1")
    g.add_argument("--truth-lmax", type=int, default=20, help="spherical harmonic degree the truth is cut at")
    g.add_argument("--truth-scale", type=float, default=1.0,
                   help="dln vp / dln vs of the truth (1: the shear perturbations of S20RTS taken as they are, the "
                        "amplitude of the paper's figures; 0.5: the usual conversion to a compressional model)")
    g.add_argument("--seed", type=int, default=1)
    g = p.add_argument_group("prior")
    g.add_argument("--prior", choices=("paper", "matern"), default="paper",
                   help="matern: the three options below, with a Robin condition.  paper: the operator of the paper's "
                        "companion (Bui-Thanh, Ghattas, Martin, Stadler, SISC 2013, section 6.2) as its figures "
                        "have it, A = -div(Theta grad) + alpha with alpha = 1.5e-2, Theta = beta (I - theta(x) x x^T), "
                        "beta = 7.5e-3 R^2, lengths in 100 km and a natural condition: correlations decay over "
                        "4500 km along the surface and a fifth of that across it, the standard deviation is 0.15 km/s "
                        "67 km down and 0.095 at the centre")
    g.add_argument("--prior-std", type=float, default=0.1, help="matern: pointwise standard deviation, km/s")
    g.add_argument("--prior-corr", type=float, default=1500.0, help="matern: lateral correlation length, km")
    g.add_argument("--prior-aniso", type=float, default=0.25,
                   help="matern: radial over lateral diffusion (the radial correlation length is its square root "
                        "times the lateral)")
    g = p.add_argument_group("solvers")
    g.add_argument("--rtol", type=float, default=1e-3, help="gradient reduction that ends the MAP solve")
    g.add_argument("--max-newton", type=int, default=25)
    g.add_argument("--cg-max", type=int, default=100, help="CG iterations per Newton step at most")
    g.add_argument("--rank", type=int, default=64, help="Hessian products of the low-rank approximation")
    g.add_argument("--oversample", type=int, default=10)
    g.add_argument("--batch", type=int, default=8, help="directions per pair of sweeps")
    g.add_argument("--double-pass", action="store_true")
    g.add_argument("--nsamples", type=int, default=4)
    g.add_argument("--var-samples", type=int, default=300, help="prior samples for its pointwise variance")
    g = p.add_argument_group("run")
    g.add_argument("--out", default="out/run")
    g.add_argument("--stages", default="data,map,eig,post")
    g.add_argument("--resume", action="store_true", help="continue from what --out holds")
    g.add_argument("--time-limit", type=float, default=0.0,
                   help="seconds after which a stage stops cleanly at its next checkpoint (0: none)")
    g.add_argument("--ensemble", action="store_true",
                   help="after the MAP point (read from --out), every rank holds the whole problem and the "
                        "Hessian products, the prior's solves and the variance's samples are divided among the "
                        "ranks; needs a library whose MatMvMult and pointwise_variance take ensemble=")
    args = p.parse_args(argv)
    if args.rank <= args.oversample:
        p.error("--rank (%d) must be larger than --oversample (%d)" % (args.rank, args.oversample))
    return args


#: options that say how a run proceeds, not what problem it solves: the others must be the
#: same when a run is resumed
RUN_ONLY = ("out", "stages", "resume", "time_limit", "ensemble", "rank", "oversample", "batch", "double_pass",
            "nsamples", "var_samples", "rtol", "max_newton", "cg_max", "element_chunk", "element_block")


def density(u, a, m, p, x, rho, c0):
    """The acoustic wave equation in its pressure form, at a point: ``u`` the
    pressure, ``a`` its second time derivative, ``m`` the wave speed's departure from
    the reference ``c0``, ``p`` the test function."""
    import hippymfem as hm

    return a.val * p.val / (rho * (c0 + m.val) ** 2) + hm.inner(u.grad, p.grad) / rho


class Canonical:
    """Parameter fields in an order that does not depend on the partition: sorted by
    the key of their vertex.  Fields are saved and read back in that order.

    Stacks of fields travel between the ranks a few rows at a time: one message of
    more than 2^31 bytes is more than MPI takes.
    """

    def __init__(self, comm, key, message_bytes=1 << 28):
        self.comm = comm
        allk = np.concatenate(comm.allgather(np.asarray(key, dtype=np.int64)))
        self.keys = np.sort(allk)
        if np.any(self.keys[1:] == self.keys[:-1]):
            raise RuntimeError("two parameter dofs share a vertex key")
        self.pos = np.searchsorted(self.keys, key)
        self.nglobal = allk.size
        self._allpos = comm.gather(self.pos)                 # rank 0: where every rank's values go
        self._rows = max(1, int(message_bytes // max(8 * self.nglobal, 1)))

    def gather(self, arr):
        """Local values ``(..., nlocal)`` to the canonical array on rank 0 (None elsewhere)."""
        arr = np.asarray(arr)
        rank0 = self.comm.rank == 0
        out = np.empty(arr.shape[:-1] + (self.nglobal,)) if rank0 else None
        if arr.ndim == 1:
            parts = self.comm.gather(arr)
            if rank0:
                for pos, a in zip(self._allpos, parts):
                    out[pos] = a
            return out
        for a in range(0, arr.shape[0], self._rows):
            b = min(a + self._rows, arr.shape[0])
            parts = [arr[a:b]] if self.comm.size == 1 else self.comm.gather(arr[a:b])
            if rank0:
                for pos, part in zip(self._allpos, parts):
                    out[a:b][..., pos] = part
        return out

    def scatter(self, arr):
        """The canonical array (on rank 0) to this rank's local values."""
        rank0 = self.comm.rank == 0
        shape = self.comm.bcast(arr.shape if rank0 else None)
        if len(shape) == 1:
            arr = self.comm.bcast(arr if rank0 else None)
            return np.ascontiguousarray(arr[self.pos])
        out = np.empty(shape[:-1] + (self.pos.size,))
        for a in range(0, shape[0], self._rows):
            b = min(a + self._rows, shape[0])
            piece = arr[a:b] if self.comm.size == 1 else self.comm.bcast(np.ascontiguousarray(arr[a:b]) if rank0 else None)
            out[a:b] = piece[..., self.pos]
        return out


class Problem:
    """Everything the stages share: mesh, spaces, propagator, sources and receivers,
    prior, and the bookkeeping of the output directory."""

    def __init__(self, args, comm, writer=None, ensemble=None):
        import jax.numpy as jnp
        import hippymfem as hm
        import earth
        import sem

        self.args, self.comm = args, comm
        self.rank0 = comm.rank == 0                  # reads the files and passes them to its communicator
        self.writer = self.rank0 if writer is None else bool(writer)      # writes the files and prints
        self.ensemble = ensemble
        self.t_start = time.time()
        t0 = time.time()
        bm = earth.BallMesh(args.n, cube=args.cube)
        self.bm = bm
        pmesh = bm.cube_mesh(comm, args.refine)
        Vm = hm.FunctionSpace.H1(pmesh, 1)
        q = Vm.coordinates()
        self.canon = Canonical(comm, bm.vertex_key(q, args.refine))
        self.pmesh, self.Vm, self.qm = pmesh, Vm, q
        qloc = sem.vertex_coordinates(pmesh)         # the vertices in the cube configuration
        # In MFEM the mesh takes the trilinear geometry of the warped vertices, which is
        # what the prior is assembled on (the parameter is trilinear, and the curved
        # boundary is 2 km from it at n = 32).  The wave field's geometry is the
        # analytic map itself at the nodes of each element, evaluated on the device.
        t1 = time.time()
        bm.warp_mesh(pmesh, 1)
        self.xm = Vm.coordinates()
        self.prior = self._prior()
        t_prior = time.time() - t1
        # what one application of the prior costs (it preconditions every CG iteration)
        v, w = Vm.vector(), Vm.vector()
        self.prior.sample(self.prior.sample_noise(1.0), v, add_mean=False)
        t1 = time.time()
        self.prior.Rsolver.solve(w, v)
        t_rsolve = time.time() - t1
        t1 = time.time()
        self.prior.R.mult(v, w)
        t_rmult = time.time() - t1
        Vu = hm.FunctionSpace.H1(pmesh, args.order)
        self.Vu = Vu
        t_mesh = time.time() - t0 - t_prior
        t0 = time.time()
        col = sem.Collocation(Vu, mapped=(Vm, qloc, bm.warp_jax()))
        eng = sem.SpectralWave(
            Vu, Vm, density, aux=earth.prem_aux(), ess_bdr="all", col=col, cfl=args.cfl,
            dtype=jnp.float32 if args.float32 else jnp.float64,
            geometry_dtype=jnp.float32 if (args.geometry_float32 or args.float32) else None,
            element_chunk=args.element_chunk, element_block=args.element_block)
        self.col, self.eng = col, eng
        eng.set_parameter(Vm.vector())
        dt_lim = eng.stable_dt()
        # a whole number of steps, a multiple of ten so that the round-off of the
        # stability estimate cannot change it from one partition to another
        nsteps = int(np.ceil(args.T / (args.cfl * dt_lim) / 10.0) * 10)
        eng.dt, eng.nsteps = args.T / nsteps, nsteps
        self.dt_limit = dt_lim
        t_engine = time.time() - t0
        # the source's frequency: what the surface elements resolve
        h_node = bm.h_surface / (2 ** args.refine) / args.order
        c_min = float(earth.prem(earth.R_EARTH)[1])
        f0_mesh = c_min / (args.ppw * h_node) / 2.5
        self.f0 = float(args.f0) if args.f0 is not None else min(0.028, f0_mesh)
        self.f0_mesh = f0_mesh
        t = eng.dt * np.arange(nsteps + 1)
        self.t = t
        self.src = earth.paper_sources()
        self.rec, self.rec_lat, self.rec_lon = earth.paper_receivers(args.nlat, args.nlon)
        self.wavelet = earth.ricker(t, self.f0)
        self.src_op = sem.value_operator(col, self.src, comm)
        self.amp = 1.0
        self._set_sources()
        rhat = self.rec / np.linalg.norm(self.rec, axis=1)[:, None]
        rho_s = float(earth.prem(earth.R_EARTH)[0])
        eng.set_receivers(sem.derivative_operator(col, self.rec, rhat, 1.0 / rho_s, comm))
        from model import FourierData

        self.data = FourierData(nsteps, eng.dt, args.nfreq)
        self.sizes = dict(
            volume_error=abs(eng.volume / (4.0 / 3.0 * np.pi * earth.R_EARTH ** 3) - 1.0),
            elements=int(pmesh.GetGlobalNE()), wave_dofs=int(Vu.GlobalTrueVSize()),
            parameters=int(Vm.GlobalTrueVSize()), ranks=comm.size,
            n=args.n, refine=args.refine, order=args.order, n_core=bm.n_core, n_mantle=bm.n_mantle,
            h_surface_km=bm.h_surface / 2 ** args.refine,
            dt=eng.dt, nsteps=nsteps, dt_limit=dt_lim, T=args.T, f0=self.f0, f0_resolved=f0_mesh,
            nodes_per_wavelength_at_f0=c_min / self.f0 / h_node,
            data=int(self.data.ndata * self.rec.shape[0]),
            device=str(hm.fem.kernel.device()), precision="fp32" if args.float32 else "fp64",
            prior_seconds=dict(solve=t_rsolve, mult=t_rmult),
            setup_seconds=dict(mesh=t_mesh, engine=t_engine, prior=t_prior))
        os.makedirs(args.out, exist_ok=True) if self.writer else None
        comm.Barrier()

    # ------------------------------------------------------------------ pieces
    def _set_sources(self):
        w = self.amp * np.tile(self.wavelet[:, None], (1, self.src.shape[0]))
        self.eng.set_sources(self.src_op, w)

    def _prior(self):
        import mfem.par as mfem
        import hippymfem as hm

        a = self.args
        if a.prior == "paper":
            return self._paper_prior()
        gamma, delta = hm.BiLaplacianComputeCoefficients(a.prior_std ** 2, a.prior_corr, 3)
        Theta, keep = None, []
        if a.prior_aniso != 1.0:
            # Theta = I + (aniso - 1) r r^T with r the radial direction, interpolated
            # from the vertices; the scaling keeps the pointwise variance
            s = a.prior_aniso ** -0.25
            gamma, delta = gamma * s, delta * s
            from hippymfem.common.parvector import host_readwrite

            Vr = hm.FunctionSpace.H1(self.pmesh, 1, vdim=3)
            gf = mfem.ParGridFunction(Vr.fes)
            X = Vr._node_coordinates()                       # the vertices, local numbering
            r = np.linalg.norm(X, axis=1)
            rh = np.where(r[:, None] > 1e-9, X / np.maximum(r, 1e-9)[:, None], 0.0)
            host_readwrite(gf)
            gf.GetDataArray()[:] = rh.T.reshape(-1)          # by nodes: all x, all y, all z
            rc = mfem.VectorGridFunctionCoefficient(gf)
            outer = mfem.OuterProductCoefficient(rc, rc)
            ident = mfem.IdentityMatrixCoefficient(3)
            Theta = mfem.MatrixSumCoefficient(ident, outer, 1.0, a.prior_aniso - 1.0)
            keep = [Vr, gf, rc, outer, ident]
        prior = hm.BiLaplacianPrior(self.Vm, gamma, delta, Theta=Theta, robin_bc=True)
        prior.keep(*keep)
        return prior

    def _paper_prior(self, alpha=1.5e-2, beta=7.5e-3, theta=4e-2, unit=100.0):
        """The prior of the paper's companion (SISC 2013, section 6.2), covariance ``A^-2`` with

            A = -div(Theta grad) + alpha,   Theta = beta R^2 (I - c(|x| / R) r r^T),
            c(s) = (1 - theta) (2 s - s^2),

        ``r`` the radial direction, lengths in units of ``unit`` km and a natural boundary
        condition.  Along the surface correlations decay over ``sqrt(beta / alpha) R``, 4500
        km; across it over ``sqrt(theta)`` of that at the surface and the same at the
        centre, where the operator is isotropic.  The companion's (2.7a) has ``alpha`` on
        the diffusion term too; it is this form, with lengths in 100 km, that gives its
        figures: a standard deviation of 0.151 km/s 67 km down and 0.095 at the centre
        (0.146 in the SC12 paper's figure 7), and samples with features the size of
        continents.
        """
        import mfem.par as mfem
        import hippymfem as hm
        import earth
        from hippymfem.common.parvector import host_readwrite

        Vr = hm.FunctionSpace.H1(self.pmesh, 1, vdim=3)
        gf = mfem.ParGridFunction(Vr.fes)
        X = Vr._node_coordinates()                           # the vertices, local numbering
        r = np.linalg.norm(X, axis=1)
        s = r / earth.R_EARTH
        c = np.clip((1.0 - theta) * (2.0 * s - s * s), 0.0, 1.0)
        v = np.where(r[:, None] > 1e-9, np.sqrt(c)[:, None] * X / np.maximum(r, 1e-9)[:, None], 0.0)
        host_readwrite(gf)
        gf.GetDataArray()[:] = v.T.reshape(-1)               # by nodes: all x, all y, all z
        vc = mfem.VectorGridFunctionCoefficient(gf)
        outer = mfem.OuterProductCoefficient(vc, vc)
        ident = mfem.IdentityMatrixCoefficient(3)
        Theta = mfem.MatrixSumCoefficient(ident, outer, 1.0, -1.0)
        # the mesh is in km: A^-2 with lengths in `unit` km is unit^3 times that with lengths in km
        scale = unit ** -1.5
        prior = hm.BiLaplacianPrior(self.Vm, beta * earth.R_EARTH ** 2 * scale, alpha * scale, Theta=Theta, robin_bc=False)
        prior.keep(Vr, gf, vc, outer, ident)
        return prior

    def say(self, *a):
        if self.writer:
            print(*a, flush=True)

    def path(self, name):
        return os.path.join(self.args.out, name)

    def have(self, name):
        return self.comm.bcast(os.path.exists(self.path(name)) if self.rank0 else None)

    def write(self, name, arr):
        """``arr`` into ``name`` in one step, so that a run killed while it writes leaves
        the file as it was."""
        if self.writer:
            tmp = self.path(name) + ".tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, self.path(name))

    def save(self, name, vec):
        self.write(name, self.canon.gather(vec.array if hasattr(vec, "array") else vec))

    def load(self, name, vec=None):
        arr = np.load(self.path(name)) if self.rank0 else None
        loc = self.canon.scatter(arr)
        if vec is None:
            vec = self.Vm.vector()
        vec.array[:] = loc
        return vec

    def dump(self, name, obj):
        if self.writer:
            with open(self.path(name) + ".tmp", "w") as f:
                json.dump(obj, f, indent=1)
            os.replace(self.path(name) + ".tmp", self.path(name))

    def out_of_time(self):
        """Whether the time limit has passed, by rank 0's clock: one answer for every rank."""
        lim = self.args.time_limit
        late = (bool(lim) and (time.time() - self.t_start) > lim) if self.rank0 else None
        return self.comm.bcast(late, root=0)

    def counters(self):
        e = self.eng
        return dict(field_steps=int(e.n_field_steps), sweeps=int(e.n_sweeps), wave_seconds=float(e.t_wave))


# ----------------------------------------------------------------------- stages
def stage_data(P):
    """The truth, its seismograms and the noisy data."""
    import earth

    a, comm, eng = P.args, P.comm, P.eng
    if a.resume and P.have("data.npz"):
        z = np.load(P.path("data.npz")) if P.rank0 else None
        z = comm.bcast({k: z[k] for k in z.files} if P.rank0 else None)
        P.amp = float(z["amp"])
        P._set_sources()
        P.m_true = P.load("truth.npy")
        P.say("data: read from %s" % P.path("data.npz"))
        return z
    t0 = time.time()
    m_true = P.Vm.vector()
    m_true.array[:] = earth.s20rts_perturbation(P.xm, lmax=a.truth_lmax, scale=a.truth_scale)
    P.m_true = m_true
    eng.set_parameter(m_true)
    y = eng.forward()
    d = P.data.apply(y)
    # the source's amplitude is a free constant: choose it so the largest datum is 1,
    # which makes the noise level the paper's number
    P.amp = 1.0 / np.abs(d).max()
    P._set_sources()
    y, d = y * P.amp, d * P.amp
    rng = np.random.default_rng(a.seed)
    noise = a.noise * rng.standard_normal(d.shape)
    d_obs = d + noise
    eng.set_parameter(P.Vm.vector())
    y0 = eng.forward()
    d0 = P.data.apply(y0)
    z = dict(y_true=y, d_true=d, d_obs=d_obs, y_prem=y0, d_prem=d0, amp=P.amp, noise_std=a.noise,
             t=P.t, rec=P.rec, rec_lat=P.rec_lat, rec_lon=P.rec_lon, src=P.src,
             wavelet=P.amp * P.wavelet, frequencies=P.data.frequencies)
    if P.writer:
        np.savez(P.path("data.npz"), **z)
    P.save("truth.npy", m_true)
    xyz = P.canon.gather(P.xm.T)
    qq = P.canon.gather(P.qm.T)
    if P.writer:
        np.savez(P.path("vertices.npz"), key=P.canon.keys, xyz=xyz.T, q=qq.T)
    snr = np.linalg.norm(d - d0) / (a.noise * np.sqrt(d.size))
    P.say("data: %d receivers x %d coefficients, truth |m| max %.3f km/s; |d_true - d_prem| / (sigma sqrt(n)) = %.1f; %.1f s"
          % (P.rec.shape[0], P.data.ndata, comm.allreduce(float(np.abs(m_true.array).max()), op=_MAX()), snr, time.time() - t0))
    return z


def _MAX():
    from mpi4py import MPI

    return MPI.MAX


def make_model(P, z):
    from model import SeismicModel

    return SeismicModel(P.eng, P.data, P.prior, z["d_obs"], float(z["noise_std"]))


def stage_verify(P, model):
    """Finite-difference checks: the gradient at a generic point, and the
    Gauss-Newton Hessian where it is the Hessian (data fitted exactly)."""
    import hippymfem as hm
    from model import SeismicModel
    from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

    hm.parRandom.set_seed(7)
    m0 = P.Vm.vector()
    noise = P.prior.sample_noise(1.0)
    P.prior.sample(noise, m0, add_mean=False)
    h = P.Vm.vector()
    noise = P.prior.sample_noise(1.0)
    P.prior.sample(noise, h, add_mean=False)
    out = {}
    for label, mdl, mis in (("cost and gradient (noisy data, with the prior)", model, False),):
        x = mdl.generate_vector()
        x[PARAMETER] = m0
        mdl.solveFwd(x[STATE], x)
        mdl.solveAdj(x[ADJOINT], x)
        c0 = mdl.cost(x)
        g = P.Vm.vector()
        mdl.evalGradientParameter(x, g, misfit_only=mis)
        gh = g.inner(h)
        P.say("verify: %s; J = %.6e, (g, h) = %.6e" % (label, c0[0], gh))
        rows = []
        for eps in (1e-1, 1e-2, 1e-3, 1e-4):
            cs = []
            for sgn in (1.0, -1.0):
                xp = mdl.generate_vector()
                xp[PARAMETER] = m0.copy().axpy(sgn * eps, h)
                mdl.solveFwd(xp[STATE], xp)
                cs.append(mdl.cost(xp)[0])
            fd = (cs[0] - cs[1]) / (2 * eps)
            rows.append((eps, abs(fd - gh) / abs(gh)))
            P.say("   eps %.0e: central difference %.10e, relative error %.3e" % (eps, fd, rows[-1][1]))
        out["gradient"] = rows
    # Hessian: data generated at m0 itself, so the residual vanishes there
    x = model.generate_vector()
    x[PARAMETER] = m0
    model.solveFwd(x[STATE], x)
    exact = SeismicModel(P.eng, P.data, P.prior, P.data.apply(x[STATE].y), model.noise_std)
    exact.setPointForHessianEvaluations(x)
    Hh = exact.hessian_misfit([h])[0]
    k = P.Vm.vector()
    P.prior.sample(P.prior.sample_noise(1.0), k, add_mean=False)
    Hk = exact.hessian_misfit([k])[0]
    sym = abs(Hh.inner(k) - Hk.inner(h)) / abs(Hh.inner(k))
    both = exact.hessian_misfit([h, k])
    bat = max(both[0].copy().axpy(-1.0, Hh).norm("l2") / Hh.norm("l2"),
              both[1].copy().axpy(-1.0, Hk).norm("l2") / Hk.norm("l2"))
    P.say("verify: Gauss-Newton Hessian; (H h, k) = %.10e, (H k, h) = %.10e, relative difference %.2e; "
          "batched against single products %.2e" % (Hh.inner(k), Hk.inner(h), sym, bat))
    rows = []
    for eps in (1e-1, 1e-2, 1e-3):
        gs = []
        for sgn in (1.0, -1.0):
            xp = exact.generate_vector()
            xp[PARAMETER] = m0.copy().axpy(sgn * eps, h)
            exact.solveFwd(xp[STATE], xp)
            exact.solveAdj(xp[ADJOINT], xp)
            g = P.Vm.vector()
            exact.evalGradientParameter(xp, g, misfit_only=True)
            gs.append(g)
        fd = gs[0].copy().axpy(-1.0, gs[1]).scale(0.5 / eps)
        rows.append((eps, fd.copy().axpy(-1.0, Hh).norm("l2") / Hh.norm("l2")))
        P.say("   eps %.0e: |FD of the gradient - H h| / |H h| = %.3e" % rows[-1])
    out.update(hessian=rows, symmetry=sym, batch=bat)
    P.dump("verify.json", out)
    return out


def stage_map(P, model, z):
    """The MAP point by inexact Gauss-Newton-CG, started at the prior mean."""
    import hippymfem as hm
    from hippymfem.modeling.variables import PARAMETER, STATE

    a, comm = P.args, P.comm
    hist = {"iterations": [], "g0": None, "done": False}
    m = P.Vm.vector()
    if a.resume and P.have("map.json"):
        hist = comm.bcast(json.load(open(P.path("map.json"))) if P.rank0 else None)
        P.load("map.npy", m)
        P.say("map: resuming after %d Newton iterations" % len(hist["iterations"]))
    if hist["done"] and not hist.get("converged", True) and len(hist["iterations"]) < a.max_newton:
        P.say("map: the solve on file ended without converging (%s); continuing it" % hist.get("reason"))
        hist["done"] = False
    if P.ensemble is not None and not hist["done"]:
        raise SystemExit("--ensemble needs the MAP point in --out: compute it first without --ensemble")
    if hist["done"]:
        x = model.generate_vector()
        x[PARAMETER] = m
        model.solveFwd(x[STATE], x)
        return x, hist
    t0 = time.time()
    c0 = P.counters()
    params = hm.ReducedSpaceNewtonCG_ParameterList()
    params["max_iter"] = max(a.max_newton - len(hist["iterations"]), 1)     # --max-newton counts them all
    params["GN_iter"] = 10 ** 6
    params["cg_max_iter"] = a.cg_max
    params["print_level"] = 0 if P.rank0 else -1
    params["rel_tolerance"] = a.rtol
    params["abs_tolerance"] = 0.0
    if hist["g0"] is not None:
        # a resumed solve measures its progress from the original starting point
        params["rel_tolerance"] = 0.0
        params["abs_tolerance"] = a.rtol * hist["g0"]
    solver = hm.ReducedSpaceNewtonCG(model, params)

    class Stop(Exception):
        pass

    cg_base = hist["iterations"][-1]["cg"] if hist["iterations"] else 0
    t_base = hist["iterations"][-1]["seconds"] if hist["iterations"] else 0.0

    def callback(it, x):
        cost = model.cost(x)
        hist["iterations"].append(dict(
            it=len(hist["iterations"]) + 1, cg=cg_base + int(solver.total_cg_iter), cost=cost[0],
            reg=cost[1], misfit=cost[2], seconds=t_base + time.time() - t0, **P.counters()))
        if hist["g0"] is None:
            hist["g0"] = float(solver.initial_grad_norm)
        P.save("map.npy", x[PARAMETER])
        P.dump("map.json", hist)
        if P.out_of_time():
            raise Stop()

    solver.callback = callback
    x = [None, m, None]
    stopped = False
    try:
        x = solver.solve(x)
    except Stop:
        stopped = True
        P.say("map: stopped at the time limit after %d Newton iterations" % len(hist["iterations"]))
    if not stopped:
        g0 = hist["g0"] if hist["g0"] is not None else float(solver.initial_grad_norm)
        hist.update(done=True, converged=bool(solver.converged), reason=solver.termination_reasons[solver.reason],
                    final_grad=float(solver.final_grad_norm), g0=g0,
                    reduction=float(solver.final_grad_norm) / g0 if g0 else 0.0,
                    cg_total=int(hist["iterations"][-1]["cg"] if hist["iterations"] else 0),
                    seconds=t_base + time.time() - t0)
        P.save("map.npy", x[PARAMETER])
        if P.writer:
            np.savez(P.path("map_data.npz"), y=x[STATE].y, d=P.data.apply(x[STATE].y))
        P.dump("map.json", hist)
        c1 = P.counters()
        P.say("map: %s; %d Newton iterations, |g|/|g0| = %.2e, %.1f s (%.1f s in the wave sweeps, %d field-steps)"
              % (hist["reason"], len(hist["iterations"]), hist["reduction"], time.time() - t0,
                 c1["wave_seconds"] - c0["wave_seconds"], c1["field_steps"] - c0["field_steps"]))
        err = x[PARAMETER].copy().axpy(-1.0, P.m_true)
        P.say("map: |m_map - m_true| / |m_true| = %.3f (l2 over the vertices), misfit %.4e against %.4e at PREM"
              % (err.norm("l2") / P.m_true.norm("l2"), model.cost(x)[2],
                 0.5 * float(np.sum(((z["d_prem"] - z["d_obs"]) / float(z["noise_std"])) ** 2))))
    return (None if stopped else x), hist


def stage_eig(P, model, x):
    """Dominant eigenpairs of the prior-preconditioned misfit Hessian at the MAP."""
    import hippymfem as hm
    from model import low_rank_hessian

    a, comm = P.args, P.comm
    r, k = a.rank, a.rank - a.oversample
    if a.resume and P.have("eigenvalues.npy"):
        d = comm.bcast(np.load(P.path("eigenvalues.npy")) if P.rank0 else None)
        if d.size == k:
            U = hm.MultiVector(P.Vm.vector(), d.size)
            U.data[:] = P.canon.scatter(np.load(P.path("eigenvectors.npy")) if P.rank0 else None)
            P.say("eig: read %d eigenpairs" % d.size)
            return d, U
        P.say("eig: %d eigenpairs on file, %d asked for: taking more products" % (d.size, k))
    t0 = time.time()
    c0 = P.counters()
    model.setPointForHessianEvaluations(x)
    hm.parRandom.set_seed(100 + a.seed)
    Omega = hm.MultiVector(P.Vm.vector(), r)
    hm.parRandom.normal_multivector(1.0, Omega)
    # the first direction as a function on the mesh, in one number: the directions change
    # with the number of ranks, and a single pass pairs every product with its direction
    first = P.canon.gather(np.asarray(Omega.data[:1]))
    mark = comm.bcast(float(np.dot(np.ravel(first), np.cos(np.arange(np.size(first))))) if P.rank0 else None)
    del first
    Y, ndone = None, 0
    if a.resume and P.have("eig_products.npy"):
        was = comm.bcast(json.load(open(P.path("eig_products.json")))
                         if P.rank0 and os.path.exists(P.path("eig_products.json")) else None)
        same = was is not None and abs(was["directions"] - mark) <= 1e-9 * max(abs(mark), 1.0)
        if same or a.double_pass:
            ndone = min(r, comm.bcast(int(np.load(P.path("eig_done.npy"))) if P.rank0 else None))
            Y = hm.MultiVector(P.Vm.vector(), r)
            Y.data[:ndone] = P.canon.scatter(np.load(P.path("eig_products.npy"))[:ndone] if P.rank0 else None)
            P.say("eig: resuming with %d of %d products" % (ndone, r))
        else:
            P.say("eig: the products on file were taken along other random directions (another number of "
                  "ranks or seed); a single pass needs its own, so they are taken again")

    class Stop(Exception):
        pass

    def on_batch(Yc, n):
        if (n == r) or P.out_of_time() or (n // a.batch) % 8 == 0:
            P.write("eig_products.npy", P.canon.gather(Yc.data))
            P.write("eig_done.npy", np.array(n))
            P.dump("eig_products.json", dict(done=int(n), directions=mark))
        if P.out_of_time() and n < r:
            raise Stop()

    try:
        d, U, Y = low_rank_hessian(model, Omega, k, batch=a.batch, double_pass=a.double_pass,
                                   products=Y, ndone=ndone, on_batch=on_batch if P.ensemble is None else None,
                                   log=P.say, ensemble=P.ensemble)
    except Stop:
        P.say("eig: stopped at the time limit")
        return None, None
    P.write("eigenvectors.npy", P.canon.gather(U.data))
    P.write("eigenvalues.npy", d)                 # last: a resumed run looks for this one
    c1 = P.counters()
    P.say("eig: %d products (%s pass), %d eigenvalues from %.3e to %.3e, %d above 1; %.1f s (%.1f s in the wave sweeps)"
          % (r * (2 if a.double_pass else 1), "double" if a.double_pass else "single", k, d[0], d[-1],
             int((d > 1).sum()), time.time() - t0, c1["wave_seconds"] - c0["wave_seconds"]))
    P.dump("eig.json", dict(products=r * (2 if a.double_pass else 1), k=k, seconds=time.time() - t0,
                            wave_seconds=c1["wave_seconds"] - c0["wave_seconds"],
                            field_steps=c1["field_steps"] - c0["field_steps"]))
    return d, U


def stage_post(P, model, x, d, U):
    """Pointwise variances and samples of the Laplace approximation."""
    import hippymfem as hm
    from hippymfem.modeling.variables import PARAMETER

    a = P.args
    t0 = time.time()
    keep = d > 0.0
    if not keep.all():
        P.say("post: dropping %d non-positive eigenvalues" % int((~keep).sum()))
        Uk = hm.MultiVector(P.Vm.vector(), int(keep.sum()))
        Uk.data[:] = U.data[keep]
        d, U = d[keep], Uk
    post = hm.GaussianLRPosterior(P.prior, d, U, mean=x[PARAMETER])
    hm.parRandom.set_seed(200 + a.seed)
    kw = {} if P.ensemble is None else dict(ensemble=P.ensemble)
    pv_post, pv_prior, pv_corr = post.pointwise_variance(method="MonteCarlo", n=a.var_samples, **kw)
    P.save("var_prior.npy", pv_prior)
    P.save("var_reduction.npy", pv_corr)
    hm.parRandom.set_seed(300 + a.seed)
    pri, pos = [], []
    for _ in range(a.nsamples):
        s_prior, s_post = P.Vm.vector(), P.Vm.vector()
        noise = P.prior.sample_noise(1.0)
        post.sample(noise, s_prior, s_post, add_mean=True)
        pri.append(s_prior.array.copy())
        pos.append(s_post.array.copy())
    if a.nsamples:
        A, B = P.canon.gather(np.array(pri)), P.canon.gather(np.array(pos))
        if P.writer:
            np.save(P.path("samples_prior.npy"), A)
            np.save(P.path("samples_posterior.npy"), B)
    red = pv_corr.sum() / max(pv_prior.sum(), 1e-300)
    P.say("post: pointwise variance (prior from %d samples), mean reduction %.1f %% of the prior variance, "
          "largest %.1f %%; %d samples of each; %.1f s"
          % (a.var_samples, 100 * red,
             100 * P.comm.allreduce(float((pv_corr.array / np.maximum(pv_prior.array, 1e-300)).max()), op=_MAX()),
             a.nsamples, time.time() - t0))
    return dict(seconds=time.time() - t0, mean_reduction=float(red))


def main(argv=None):
    args = parse(argv)
    import hippymfem as hm                                    # noqa: F401  (sets the device up)
    from mpi4py import MPI
    import mfem.par as mfem

    comm = MPI.COMM_WORLD
    mfem.Hypre.Init()
    stages = [s for s in args.stages.split(",") if s]
    asked = dict(rank=args.rank, oversample=args.oversample, double_pass=args.double_pass)

    def on_file(name):
        f = os.path.join(args.out, name)
        return json.load(open(f)) if os.path.exists(f) else None

    def other_settings():
        # a folder holds one problem: the options that define it, where they differ
        was = (on_file("setup.json") or {}).get("args", {})
        return ["%s %r, now %r" % (k, was[k], v) for k, v in sorted(vars(args).items())
                if k not in RUN_ONLY and k in was and was[k] != v]

    def finished_before():
        # the same stages with the same low rank: a larger --rank continues a finished run,
        # and more Newton iterations a MAP solve that ended without converging
        done, hist = on_file("done.json"), on_file("map.json") or {}
        if done is None:
            return False
        if any(st in stages for st in ("map", "eig", "post")):
            if not hist.get("done"):
                return False                       # a later run took the MAP solve up again
            if not hist.get("converged", True) and len(hist["iterations"]) < args.max_newton:
                return False
        return set(stages) <= set(done.get("stages", stages)) and all(done.get(k, v) == v for k, v in asked.items())

    if "data" not in stages:
        args.resume = True                # a later stage alone reads what the earlier ones saved
    if args.resume:
        diff = comm.bcast(other_settings() if comm.rank == 0 else None)
        if diff:
            raise SystemExit("seismic: %s holds a run with other settings (%s); give the same ones or another --out"
                             % (args.out, ", ".join(diff)))
        if comm.bcast(finished_before() if comm.rank == 0 else None):
            if comm.rank == 0:
                print("seismic: %s holds a finished run; nothing to do" % args.out, flush=True)
            return 0
    ensemble = None
    if args.ensemble:
        # every rank sets the whole problem up on its own GPU; the world divides the solves
        ensemble, comm, args.resume = comm, MPI.COMM_SELF, True
    P = Problem(args, comm, writer=MPI.COMM_WORLD.rank == 0, ensemble=ensemble)
    s = P.sizes
    P.say("seismic: n = %d (refine %d), order %d: %d elements, %d wave unknowns, %d parameters, %d ranks, %s, %s"
          % (args.n, args.refine, args.order, s["elements"], s["wave_dofs"], s["parameters"], s["ranks"],
             s["device"], s["precision"]))
    P.say("  T = %.0f s in %d steps of %.4f s (limit %.4f s); Ricker f0 = %.4f Hz (%.1f nodes per wavelength at "
          "the surface; the mesh resolves %.4f Hz); %d data; volume error %.1e; setup %s"
          % (args.T, s["nsteps"], s["dt"], s["dt_limit"], s["f0"], s["nodes_per_wavelength_at_f0"],
             s["f0_resolved"], s["data"], s["volume_error"],
             ", ".join("%s %.1f s" % kv for kv in s["setup_seconds"].items())))
    P.say("  one prior solve %.2f s, one prior product %.2f s" % (s["prior_seconds"]["solve"], s["prior_seconds"]["mult"]))
    summary = dict(args=vars(args), sizes=s)
    P.dump("setup.json", summary)
    z = stage_data(P)
    model = make_model(P, z)
    if "verify" in stages:
        summary["verify"] = stage_verify(P, model)
    x = None
    if "map" in stages or "eig" in stages or "post" in stages:
        x, hist = stage_map(P, model, z)
        summary["map"] = hist
    d = U = None
    if x is not None and ("eig" in stages or "post" in stages):
        d, U = stage_eig(P, model, x)
    if d is not None and "post" in stages:
        summary["post"] = stage_post(P, model, x, d, U)
    summary["counters"] = P.counters()
    summary["seconds"] = time.time() - P.t_start
    P.dump("summary.json", summary)
    finished = ("post" not in stages or "post" in summary) and ("eig" not in stages or d is not None) \
        and ("map" not in stages or (x is not None))
    if finished:
        P.dump("done.json", dict(stages=stages, seconds=summary["seconds"], **asked))
    P.say("seismic: done in %.1f s; %d field-steps in %d sweeps, %.1f s in the wave sweeps"
          % (summary["seconds"], P.eng.n_field_steps, P.eng.n_sweeps, P.eng.t_wave))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import sem

        sem.abort_all()
        raise

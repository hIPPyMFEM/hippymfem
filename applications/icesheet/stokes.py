# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""Nonlinear Stokes flow of an ice sheet, and the inverse problem for its basal friction.

The forward problem of Isaac, Petra, Stadler and Ghattas (J. Comput. Phys. 296, 2015):

.. math::

    -\nabla\cdot[\eta(u)(\nabla u + \nabla u^T) - I p] = \rho g,\qquad \nabla\cdot u = 0,
    \qquad \eta(u) = \tfrac12 A^{-1/n}\,\dot\varepsilon_{II}^{(1-n)/(2n)},

traction free on the surface, :math:`u\cdot n = 0` and :math:`T\sigma n + e^{\beta} T u = 0`
on the bed, sea-water pressure below sea level on margins facing the ocean, no slip on
margins on land.  The unknown of the inverse problem is :math:`\beta`, the log of the
basal friction coefficient, a field on the bed.

Written for hIPPyMFEM as one residual density over the slots
``(u, p, m, B, v, q)`` (velocity, pressure, log friction on the 3D mesh, the rate factor
``B = A^{-1/n}``, and the two test functions) plus boundary densities.  The library's
kernels differentiate them into every block: the Newton Jacobian of Glen's law with its
anisotropic fourth-order viscosity tensor, the divergence blocks, the friction's
derivative in the parameter, and the second derivatives of the Hessian.  What
``PDEVariationalProblem`` cannot do, a velocity-pressure pair of two spaces (Taylor-Hood
Q2-Q1), is done here: the blocks are assembled one by one and glued into one parallel
matrix, and this class implements the library's ``PDEProblem`` interface, so
``hm.Model``, ``ReducedSpaceNewtonCG``, the randomized eigensolvers and
``GaussianLRPosterior`` run on it unchanged.

Two departures from the paper, both stated in the README: the velocity-pressure pair is
Taylor-Hood Q2-Q1 on a conforming mesh of columns (the paper: Q_k x P_{k-2}^disc on
adapted octree meshes), and ``u . n = 0`` on the bed is imposed by a penalty (no slip on
land margins likewise), which keeps the Jacobian symmetric: the adjoint and incremental
solves reuse the forward factorization.

The parameter lives on the 2D footprint mesh: bilinear on its cells, or with
``friction_order=2`` biquadratic, which is the trace of the velocity's space on the bed and
the paper's choice (four times the parameters on the same mesh).  A parallel matrix ``P``
copies it onto the bed nodes of the 3D mesh, and its transpose takes gradients back.
"""
import json
import os
import time

import numpy as np
import jax.numpy as jnp
import mfem.par as mfem
from mpi4py import MPI

import hippymfem as hm
from hippymfem.common.linalg import hypre_to_scipy, scipy_to_hypre, take_ownership
from hippymfem.common.parvector import ParVector
from hippymfem.fem.assemble import assemble_matrix, assemble_vector
from hippymfem.fem.csrassemble import assemble_matrix_csr
from hippymfem.fem.boundary import BoundaryKernel, assemble_boundary_vector, get_boundary_batches
from hippymfem.fem.elementbatch import get_batches
from hippymfem.fem.kernel import QuadratureKernel
from hippymfem.modeling.PDEProblem import PDEProblem
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE

import antarctica as an

U, P, M, B, V, Q = range(6)
GLEN = an.GLEN_N
RHO_ICE = 0.91                        # Gt per km^3: a flux in km^3/a times this is in Gt/a


# ------------------------------------------------------------------ densities
def sym(G):
    return 0.5 * (G + G.T)


def volume_density(u, p, m, b, v, q, x, eps2):
    """Momentum and mass, Glen's law; ``eps2`` regularizes the strain-rate invariant."""
    e = sym(u.grad)
    eII = 0.5 * jnp.sum(e * e) + eps2
    eta = 0.5 * b.val * eII ** ((1.0 - GLEN) / (2.0 * GLEN))
    return (2.0 * eta * jnp.sum(e * sym(v.grad)) - p.val * jnp.trace(v.grad)
            - q.val * jnp.trace(u.grad) + an.RHO_G * v.val[2])


def friction(beta, cap=None):
    """The friction coefficient of the sliding law: the paper's ``exp(beta)``, or with a
    ``cap`` the smooth ``1 / (exp(-beta) + 1 / cap)``, which is ``exp(beta)`` far below the
    cap and the cap far above it.  Above the penalty that holds the ice on the bed a
    friction has no meaning (the bed is frozen either way), but an inversion with a weak
    prior walks ``beta`` there (to 43 in 13 Gauss-Newton iterations against the observed
    velocities at 80 km), and coefficients of e^43 beside the others of the Jacobian leave
    the forward solve stalled at a relative residual of 1e-6."""
    if cap is None:
        return jnp.exp(beta)
    return 1.0 / (jnp.exp(-beta) + 1.0 / cap)


def make_base_density(gamma, cap=None):
    """The bed (friction and the penalty on ``u . n``; its normal points down) and the
    land margins (the no-slip penalty; vertical faces) as one density, so that both go
    into the velocity block in one pass."""
    def base(u, p, m, b, v, q, x, n):
        un, vn = jnp.dot(u.val, n), jnp.dot(v.val, n)
        bed = friction(m.val, cap) * jnp.dot(u.val - un * n, v.val - vn * n) + gamma * un * vn
        land = gamma * jnp.dot(u.val, v.val)
        return jnp.where(n[2] < -0.5, bed, land)
    return base


def marine_density(u, p, m, b, v, q, x, n):
    """Sea water pressure below sea level (x[2] is the elevation, km)."""
    return an.RHO_W_G * jnp.maximum(-x[2], 0.0) * jnp.dot(n, v.val)


#: the margins of the paper's three predictions (its Figure 10): all of them, East
#: Antarctica (here the margins from 30 W to 165 E of longitude, a cut by longitude and not
#: by drainage basin) and the Totten Glacier (the margins within 120 km of its grounding
#: line at 116 E, 67.3 S, which is 2245, -1095 km on the polar stereographic grid)
REGIONS = ("all", "east", "totten")


def region_weight(region, x):
    """1 on the margins of ``region`` and 0 elsewhere, at a point ``x`` (km)."""
    if region == "all":
        return 1.0
    if region == "east":
        lon = jnp.degrees(jnp.arctan2(x[0], x[1]))
        return jnp.where((lon > -30.0) & (lon < 165.0), 1.0, 0.0)
    if region == "totten":
        return jnp.where(jnp.hypot(x[0] - 2245.1, x[1] + 1095.0) < 120.0, 1.0, 0.0)
    raise ValueError("unknown region %r (one of %s)" % (region, ", ".join(REGIONS)))


def pressure_mass_density(u, p, m, b, v, q, x, eps2):
    """``p q / eta(u)``: the viscosity-weighted pressure mass of the Schur complement."""
    e = sym(u.grad)
    eII = 0.5 * jnp.sum(e * e) + eps2
    eta = 0.5 * b.val * eII ** ((1.0 - GLEN) / (2.0 * GLEN))
    return p.val * q.val / eta


def column_partition(columns, nparts):
    """A partition of the footprint's cells into ``nparts`` blocks of nearly equal size, by
    recursive coordinate bisection of the cell centres (the longer extent at the median)."""
    xy = columns.xy[columns.quads].mean(axis=1)
    part = np.zeros(len(xy), dtype=np.int64)

    def split(idx, n, base):
        if n == 1:
            part[idx] = base
            return
        n1 = n // 2
        c = xy[idx]
        ax = int(np.argmax(c.max(axis=0) - c.min(axis=0)))
        order = idx[np.argsort(c[:, ax], kind="stable")]
        cut = int(round(len(order) * n1 / n))
        split(order[:cut], n1, base)
        split(order[cut:], n - n1, base + n1)
    split(np.arange(len(xy)), nparts, 0)
    return part


# ------------------------------------------------------------- the problem
class ForwardStopped(Exception):
    """The forward solve stopped at its deadline with its iterate saved (see
    ``IceSheet.set_checkpoint``); the same call in a later job continues it."""


class IceSheet(PDEProblem):
    """The forward Stokes problem on a :class:`antarctica.Columns` mesh, with the
    library's ``PDEProblem`` interface for the inverse problem in ``m = beta``.

    Parameters
    ----------
    columns : antarctica.Columns
    comm : MPI communicator
    refine : int
        Uniform refinements of both meshes (the geometry is that of the columns).
    order : int
        Velocity order (pressure ``order - 1``).
    friction_order : int
        Order of the friction field on the footprint (1 or 2).
    gamma : float
        Penalty (MPa a / km) for ``u . n = 0`` on the bed and ``u = 0`` on land margins.
    eps_strain : float
        Strain-rate floor (1/a) of Glen's law.
    solver : object with ``factorize(J)`` and ``solve(x, b)``
        The linear solver of the Newton, adjoint and incremental systems.
    on_mesh : callable
        Called with the mesh of the columns and 0 before it is refined, then with the mesh
        and ``k`` after the ``k``-th refinement (a grid function put on it at 0 and updated
        at every ``k`` is carried through the refinements by MFEM).
    """

    def __init__(self, columns, comm, refine=0, order=2, gamma=1e6, eps_strain=1e-5,
                 solver=None, verbose=True, partition="metis", friction_cap=None, on_mesh=None, friction_order=1):
        t0 = time.time()
        self.comm = comm
        self.columns = columns
        self.verbose = verbose and comm.rank == 0
        if partition == "columns" and comm.size > 1:
            # whole columns on a rank: the strong vertical coupling of a thin ice sheet
            # then stays inside a subdomain (see BlockJacobiCuDSS)
            part = mfem.intArray(np.tile(column_partition(columns, comm.size), columns.nz).tolist())
            self.mesh = mfem.ParMesh(comm, columns.mesh3d(), part.GetData())    # PyMFEM wants the pointer
        else:
            self.mesh = mfem.ParMesh(comm, columns.mesh3d())
        self.mesh2 = mfem.ParMesh(comm, columns.mesh2d())
        if on_mesh is not None:
            on_mesh(self.mesh, 0)             # the mesh of the columns, before it is refined
        for k in range(refine):
            self.mesh.UniformRefinement()
            self.mesh2.UniformRefinement()
            if on_mesh is not None:
                on_mesh(self.mesh, k + 1)
        self.refine = refine
        self.order = order
        self.Vv = hm.FunctionSpace.H1(self.mesh, order, vdim=3)
        # one P1 space for the pressure, the friction on the 3D mesh and the rate
        # factor: equivalent spaces built twice collide on a GPU (see the library)
        self.Vs = hm.FunctionSpace.H1(self.mesh, 1)
        # the friction: on the footprint, and on the 3D mesh the space whose trace on the bed
        # it is (the pressure's where it is bilinear, a scalar one of its order otherwise)
        self.friction_order = int(friction_order)
        self.Vm = hm.FunctionSpace.H1(self.mesh2, self.friction_order)
        self.Vf = self.Vs if self.friction_order == 1 else hm.FunctionSpace.H1(self.mesh, self.friction_order)
        self.nv = self.Vv.fes.GetTrueVSize()
        self.np = self.Vs.fes.GetTrueVSize()
        self.ne = self.mesh.GetNE()
        spaces = self.spaces = [self.Vv, self.Vs, self.Vf, self.Vs, self.Vv, self.Vs]
        self.qdeg = 2 * order + 1
        self.batches = get_batches(self.mesh, self.qdeg)
        self.kernel = QuadratureKernel(volume_density, spaces, self.batches, nparams=1)
        self.mass_kernel = QuadratureKernel(pressure_mass_density, spaces, self.batches, nparams=1)
        self.gamma = float(gamma)
        self.eps2 = float(eps_strain) ** 2
        attrs = set(int(a) for a in self.mesh.bdr_attributes.ToList())
        attrs = set().union(*comm.allgather(attrs))
        base = [a for a in (an.Columns.BED, an.Columns.LAND) if a in attrs]
        # the bed and the land margins depend on the state; the ocean's pressure does not
        self.base_batches = get_boundary_batches(self.mesh, self.qdeg, base, comm, space=self.Vv)
        #: the largest friction coefficient (MPa a/km), approached smoothly; None: exp(beta)
        self.friction_cap = None if friction_cap is None else float(friction_cap)
        self.base_kernel = BoundaryKernel(make_base_density(self.gamma, self.friction_cap), spaces, self.base_batches)
        self.bdr = [(None, self.base_batches, self.base_kernel)]
        if an.Columns.MARINE in attrs:
            bb = get_boundary_batches(self.mesh, self.qdeg, [an.Columns.MARINE], comm, space=self.Vv)
            self.bdr.append((an.Columns.MARINE, bb, BoundaryKernel(marine_density, spaces, bb)))
        self.bed_batches, self.bed_kernel = self.base_batches, self.base_kernel
        # the rate factor B = A^{-1/n} at the P1 nodes, from the column temperature
        self.Bfield = self.Vs.vector()
        self.Bfield.array[:] = self.rate_coefficient(self.Vs.coordinates())
        self.P2to3 = self._footprint_map()
        self.solver = solver
        self._state0 = None                   # last converged state (warm start)
        self._ck = None                       # checkpoints of the forward solve (set_checkpoint)
        self._lin = None                      # linearization point and its blocks
        self.counters = dict(forward=0, newton=0, factorizations=0, solves=0, assemble_s=0.0,
                             factor_s=0.0, solve_s=0.0, adjoint=0, incremental=0)
        self.t_setup = time.time() - t0

    # -------------------------------------------------------------- geometry
    def column_fields(self, xy):
        """Bilinear interpolation of the grid's node fields at points ``(n, 2)`` (km)."""
        g = self.columns.grid
        fx = (xy[:, 0] - g.xn[0]) / g.h
        fy = (xy[:, 1] - g.yn[0]) / g.h
        i0 = np.clip(np.floor(fx).astype(int), 0, g.nx - 1)
        j0 = np.clip(np.floor(fy).astype(int), 0, g.ny - 1)
        tx, ty = fx - i0, fy - j0
        used = g.node_used.astype(float)

        def interp(a):
            w = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
            c = [(j0, i0), (j0, i0 + 1), (j0 + 1, i0), (j0 + 1, i0 + 1)]
            num = sum(wi * a[ji] * used[ji] for wi, ji in zip(w, c))
            den = sum(wi * used[ji] for wi, ji in zip(w, c))
            return num / np.maximum(den, 1e-12)
        return {k: interp(getattr(g, k)) for k in ("s", "b", "Ts", "acc", "ghf")}

    def rate_coefficient(self, xyz):
        """``B = A(T)^{-1/n}`` (MPa a^{1/n}) at points, ``T`` from the column model."""
        f = self.column_fields(xyz[:, :2])
        H = np.maximum(f["s"] - f["b"], an.H_MIN / 1e3)
        zeta = np.clip((xyz[:, 2] - f["b"]) / H, 0.0, 1.0)
        from scipy.special import erf

        Hm = 1e3 * H
        a = np.maximum(f["acc"], 1e-3)
        ell = np.sqrt(2.0 * an.KAPPA_ICE * Hm / a)
        T = f["Ts"] + np.sqrt(np.pi) * ell * f["ghf"] * 1e-3 / (2.0 * an.K_ICE) * (erf(Hm / ell) - erf(zeta * Hm / ell))
        T = np.minimum(T, -an.BETA_CC * Hm * (1.0 - zeta))
        return an.rate_factor(T) ** (-1.0 / GLEN)

    def _footprint_map(self):
        """The parallel matrix copying a footprint field onto the bed nodes of its 3D space."""
        comm = self.comm
        g = self.columns.grid
        step = g.h / 2 ** self.refine / self.friction_order

        def keys(xy):
            k = np.rint((xy - np.array([g.xn[0], g.yn[0]])) / step).astype(np.int64)
            return k[:, 0] * 10 ** 7 + k[:, 1]
        x2 = self.Vm.coordinates()
        off2 = self.Vm.fes.GetMyTDofOffset()
        k2 = keys(x2)
        allk = np.concatenate(comm.allgather(k2))
        allg = np.concatenate(comm.allgather(off2 + np.arange(k2.size, dtype=np.int64)))
        order = np.argsort(allk)
        allk, allg = allk[order], allg[order]
        marker = mfem.intArray([1 if a == an.Columns.BED else 0
                                for a in range(1, int(self.mesh.bdr_attributes.Max()) + 1)])
        ess = mfem.intArray()
        self.Vf.fes.GetEssentialTrueDofs(marker, ess)
        bed = np.asarray(ess.ToList(), dtype=np.int64)
        x3 = self.Vf.coordinates()
        kb = keys(x3[bed, :2])
        pos = np.searchsorted(allk, kb)
        if bed.size and (np.any(pos >= allk.size) or np.any(allk[np.minimum(pos, allk.size - 1)] != kb)):
            raise RuntimeError("a bed node has no footprint node under it")
        import scipy.sparse as sp

        n3 = self.Vf.fes.GetTrueVSize()
        csr = sp.csr_matrix((np.ones(bed.size), (bed, allg[pos])), shape=(n3, int(self.Vm.fes.GlobalTrueVSize())))
        self.bed_tdofs = bed
        return scipy_to_hypre(csr, col_starts=np.array([off2, off2 + k2.size]))

    def to_bed(self, m2, out=None):
        """The footprint field ``m2`` on its 3D space (the bed's nodes; zero elsewhere)."""
        out = out if out is not None else self.Vf.vector()
        self.P2to3.Mult(m2.hypre, out.hypre)
        return out

    def from_bed(self, g3, out=None):
        out = out if out is not None else self.Vm.vector()
        self.P2to3.MultTranspose(g3.hypre, out.hypre)
        return out

    # --------------------------------------------------------------- vectors
    def generate_state(self):
        return ParVector(self.comm, self.nv + self.np)

    generate_adjoint = generate_state

    def generate_parameter(self):
        return self.Vm.vector()

    def init_parameter(self, m):
        return m

    def split(self, s):
        u, p = self.Vv.vector(), self.Vs.vector()
        a = s.array
        u.array[:] = a[:self.nv]
        p.array[:] = a[self.nv:]
        return u, p

    def join(self, u, p, out):
        out.array[:self.nv] = u.array
        out.array[self.nv:] = p.array
        return out

    def velocity(self, s):
        u = self.Vv.vector()
        u.array[:] = s.array[:self.nv]
        return u

    # -------------------------------------------------------------- assembly
    def _loc(self, u, p, m3, lam_u=None, lam_p=None):
        Vv, Vs = self.Vv, self.Vs
        return [Vv.local_values(u, device=True), Vs.local_values(p, device=True),
                self.Vf.local_values(m3, device=True), Vs.local_values(self.Bfield, device=True),
                Vv.local_values(lam_u if lam_u is not None else Vv.vector(), device=True),
                Vs.local_values(lam_p if lam_p is not None else Vs.vector(), device=True)]

    def residual(self, u, p, m3, eps2=None):
        """``(r_u, r_p)``, the residual's velocity and pressure parts."""
        eps2 = self.eps2 if eps2 is None else eps2
        loc = self._loc(u, p, m3)
        groups = self.batches.groups
        ru = assemble_vector(self.Vv, groups, self.kernel.element_vectors(V, loc, params=(eps2,)), self.ne)
        rp = assemble_vector(self.Vs, groups, self.kernel.element_vectors(Q, loc, params=(eps2,)), self.ne)
        for attr, bb, k in self.bdr:
            ru.axpy(1.0, assemble_boundary_vector(self.Vv, bb.groups, k.element_vectors(V, loc)))
        return ru, rp

    def jacobian(self, u, p, m3, eps2=None):
        """The monolithic Jacobian ``[[A, B^T], [B, 0]]`` at ``(u, p)``."""
        t0 = time.time()
        eps2 = self.eps2 if eps2 is None else eps2
        loc = self._loc(u, p, m3)
        groups = self.batches.groups
        k = self.kernel
        # a thunk of chunks when the element batch does not fit the device whole; the bed
        # and land entries go into the same matrix (a rank may own no such face, which
        # assembling them as a matrix of their own does not allow)
        bb, bk = self.base_batches, self.base_kernel
        A = assemble_matrix_csr(self.Vv, self.Vv, groups, k.element_matrices_or_chunks(V, U, loc, params=(eps2,)),
                                self.ne, boundary=(bb.tables(self.Vv), bb.tables(self.Vv), bk.element_matrices(V, U, loc)))
        Bt = assemble_matrix(self.Vv, self.Vs, groups, k.element_matrices_or_chunks(V, P, loc, params=(eps2,)), self.ne)
        Bd = assemble_matrix(self.Vs, self.Vv, groups, k.element_matrices_or_chunks(Q, U, loc, params=(eps2,)), self.ne)
        blocks = mfem.HypreParMatrixArray2D(2, 2)
        blocks[0, 0], blocks[0, 1] = A, Bt
        blocks[1, 0], blocks[1, 1] = Bd, None
        # a new C++ matrix that PyMFEM hands over without ownership: without this every
        # Newton step leaked a Jacobian (400 GB of host memory at 20 km)
        J = take_ownership(mfem.HypreParMatrixFromBlocks(blocks))
        self._keep_blocks = (A, Bt, Bd)
        self._block_state = (u, p, m3, eps2)          # for the preconditioner's Schur part
        self.counters["assemble_s"] += time.time() - t0
        return J

    def linearization(self, u, p, m3, eps2=None):
        """What the solver factorizes at ``(u, p)``: the Jacobian, or for a solver that works
        without the matrix (``solver.matrix_free``) the state itself."""
        if getattr(self.solver, "matrix_free", False):
            return (u, p, m3, self.eps2 if eps2 is None else eps2)
        return self.jacobian(u, p, m3, eps2)

    def pressure_mass(self, u, p, m3, eps2=None):
        eps2 = self.eps2 if eps2 is None else eps2
        loc = self._loc(u, p, m3)
        return assemble_matrix(self.Vs, self.Vs, self.batches.groups,
                               self.mass_kernel.element_matrices(Q, P, loc, params=(eps2,)), self.ne)

    def _factorize(self, J):
        t0 = time.time()
        self.solver.factorize(J)
        self._factored = J                    # what the solver holds now
        self.counters["factorizations"] += 1
        self.counters["factor_s"] += time.time() - t0

    def _solve(self, x, b, incremental=False):
        t0 = time.time()
        if incremental and hasattr(self.solver, "solve_incremental"):
            self.solver.solve_incremental(x, b)            # a looser refinement (the direct solver)
        else:
            self.solver.solve(x, b)
        self.counters["solves"] += 1
        self.counters["solve_s"] += time.time() - t0

    # ------------------------------------------------------------- forward
    def set_checkpoint(self, folder, deadline=None, warm=False, budget_end=None):
        """Save the iterate of a forward solve from rest in ``folder`` after every Newton
        step and continue from the one that is there, for a solve longer than a job;
        ``deadline`` (a ``time.time()`` value): raise ``ForwardStopped`` after the first
        step that ends later.  The checkpoint is one array per rank, so the next job has
        to run on as many ranks (the partition by columns is the same then).  ``warm``:
        for a solve started from a given state (:meth:`start_from`) as well, which every
        job of the run has to be given again.  ``budget_end`` (a ``time.time()`` value):
        a Newton step is not started if, judged by the last one, it would end after it."""
        self._ck = dict(folder=folder, deadline=deadline, slot=1, warm=warm, budget_end=budget_end)
        if self.comm.rank == 0:
            os.makedirs(folder, exist_ok=True)
        self.comm.Barrier()

    def _ck_file(self, slot):
        return os.path.join(self._ck["folder"], "state%d_r%04d_of%04d.npy" % (slot, self.comm.rank, self.comm.size))

    def _ck_write(self, u, p, **info):
        # two slots in turn, the record last: a job killed while writing leaves the
        # checkpoint before it whole
        slot = 1 - self._ck["slot"]
        s = self.generate_state()
        self.join(u, p, s)
        np.save(self._ck_file(slot), np.asarray(s.array))
        self.comm.Barrier()
        if self.comm.rank == 0:
            f = os.path.join(self._ck["folder"], "forward.json")
            krylov = self._ck.get("krylov", []) + [int(i) for i in getattr(self.solver, "iterations", [])]
            with open(f + ".tmp", "w") as fh:
                json.dump(dict(info, slot=slot, ranks=self.comm.size, counters=self.counters, krylov=krylov), fh, indent=1)
            os.replace(f + ".tmp", f)
        self._ck["slot"] = slot

    def _ck_read(self):
        f = os.path.join(self._ck["folder"], "forward.json")
        info = json.load(open(f)) if self.comm.rank == 0 and os.path.exists(f) else None
        info = self.comm.bcast(info)
        if info is None:
            return None, None
        if info["ranks"] != self.comm.size:
            raise RuntimeError("the checkpoint in %s was written on %d ranks" % (self._ck["folder"], info["ranks"]))
        s = self.generate_state()
        s.array[:] = np.load(self._ck_file(info["slot"]))
        self._ck["slot"] = info["slot"]
        return info, s

    def start_from(self, x):
        """Start the next forward solve from the state of ``x``, a solution at its parameter."""
        self._state0, self._m0 = x[STATE].copy(), x[PARAMETER].copy()

    def forget(self):
        """Start the next forward solve from rest."""
        self._state0 = None

    def is_solution(self, x, tol=1e-6):
        """Whether the state of ``x`` solves the forward problem at its parameter (the
        residual below ``tol`` of the one at rest); if so the next solve starts from it."""
        m3 = self.to_bed(x[PARAMETER])
        ru, rp = self.residual(*self.split(x[STATE]), m3)
        ok = bool(np.sqrt(ru.inner(ru) + rp.inner(rp)) <= tol * self.rest_residual(m3))
        if ok:
            self.start_from(x)
        return ok

    #: the residual's fall that ends a forward solve (``solveFwd``), where the caller gives none
    forward_rtol = 1e-8

    def solveFwd(self, state, x, rtol=None, max_iter=40, continuation=True):
        rtol = self.forward_rtol if rtol is None else rtol
        if self._state0 is None:
            return self._newton(state, x, rtol, max_iter, continuation)
        # from the last converged state; if Newton makes no headway from there (a long step
        # of a line search), walk the parameter from the one of that state to this one in
        # pieces, each started from the last, and only then solve from rest
        try:
            return self._newton(state, x, rtol, max_iter, continuation, patience=self.warm_steps)
        except RuntimeError:
            self.counters["walks"] = self.counters.get("walks", 0) + 1
        m_end, m_start = x[PARAMETER], getattr(self, "_m0", None)
        t, dt = 0.0, 0.5
        while m_start is not None and t < 1.0 and dt > 1.0 / 64:
            t1 = min(1.0, t + dt)
            m = m_start.copy()
            m.scale(1.0 - t1)
            m.axpy(t1, m_end)
            try:
                self._newton(state, [x[STATE], m, None], rtol if t1 == 1.0 else 1e-4, max_iter, continuation,
                             remember=False, patience=self.warm_steps)
                t, dt = t1, 1.5 * dt
            except RuntimeError:
                dt *= 0.5
        if t == 1.0:
            self._m0 = m_end.copy()
            return state
        self.counters["from_rest"] = self.counters.get("from_rest", 0) + 1
        self._state0 = None
        return self._newton(state, x, rtol, max_iter, continuation)

    #: a solve started from the last converged state gives up if this many Newton steps
    #: have not taken a tenth off its residual
    warm_steps = 12
    #: a whole Newton step is taken without trying shorter ones if it leaves this much of the
    #: residual or less (1: the first step that lowers the residual at all is taken)
    good_step = 1.0
    #: the residual below which a state counts as converged whatever it started from, as a
    #: fraction of the residual at rest (the weight of the ice).  A solve started from a
    #: state that is already a solution to working accuracy (a short step of a line
    #: search) cannot take another eight orders off its own first residual, and must not
    #: be asked to: it would fail, be solved again from rest, and return a cost whose last
    #: digits differ from its neighbours' by more than the step changes it.
    floor_rtol = 1e-12

    def rest_residual(self, m3):
        """The norm of the residual at rest: the loads (gravity and the sea), the same for
        every friction."""
        if not hasattr(self, "_r_rest"):
            ru, rp = self.residual(self.Vv.vector(), self.Vs.vector(), m3, self.eps2)
            self._r_rest = float(np.sqrt(ru.inner(ru) + rp.inner(rp)))
        return self._r_rest

    def _newton(self, state, x, rtol, max_iter, continuation, remember=True, patience=None):
        """Newton's method with a backtracking line search on the residual norm.

        Started from the last converged state when there is one (the parameter moved
        little); from rest otherwise, through a continuation in the strain-rate floor
        (1e-1/a, 1e-2/a, ... down to the model's), each stage's Newton solve started
        from the last.
        """
        t0 = time.time()
        self.counters["forward"] += 1
        m3 = self.to_bed(x[PARAMETER])
        if self._state0 is not None:
            u, p = self.split(self._state0)
            stages = [self.eps2]
        else:
            u, p = self.Vv.vector(), self.Vs.vector()
            floors = [1e-1, 1e-2, 1e-3, 1e-4]
            stages = ([f * f for f in floors if f * f > self.eps2] if continuation else []) + [self.eps2]
        its, first, r0_first, t_before, rn_resume, last_step = 0, 0, None, 0.0, None, None
        floor = self.floor_rtol * self.rest_residual(m3)
        ck = self._ck if self._state0 is None or (self._ck or {}).get("warm") else None
        if ck is not None:
            info, saved = self._ck_read()
            if info is not None:
                if info["stages"] != stages:
                    raise RuntimeError("the checkpoint in %s is of another continuation" % ck["folder"])
                self.counters.update(info["counters"])           # totals over the jobs
                ck["krylov"] = info.get("krylov", [])
                if info["done"]:
                    state.array[:] = saved.array
                    self._state0 = state.copy()
                    self.last_forward = dict(seconds=info["seconds"], newton=info["newton"], stages=len(stages),
                                             residual=info["residual"], jobs=info["jobs"], krylov=ck["krylov"],
                                             residual_norm=info["residual"] * info["r0"], loads_norm=float(self.rest_residual(m3)))
                    return state
                u, p = self.split(saved)
                first, r0_first, its, t_before = info["stage"], info["r0"], info["newton"], info["seconds"]
                rn_resume = info.get("rn_last")
            jobs = (info["jobs"] if info is not None else 0) + 1
        for si in range(first, len(stages)):
            eps2 = stages[si]
            final = si == len(stages) - 1
            r0 = r0_first if si == first else None
            rn_last = rn_resume if si == first else None
            for k in range(max_iter):
                t_step = time.time()
                ru, rp = self.residual(u, p, m3, eps2)
                rn = np.sqrt(ru.inner(ru) + rp.inner(rp))
                if getattr(self.solver, "forcing", False):
                    # the linear solve no more accurate than Newton's method can use (Eisenstat
                    # and Walker): loose while the residual falls slowly, tight once it falls fast
                    self.solver.rtol = (self.solver.rtol_loose if rn_last is None else
                                        float(np.clip(0.9 * (rn / rn_last) ** 2, self.solver.rtol_tight, self.solver.rtol_loose)))
                    rn_last = rn
                if r0 is None:
                    r0 = max(rn, 1e-300)
                if self.verbose and final:
                    print("    newton %2d: |r| %.3e (%.1e)" % (k, rn, rn / r0), flush=True)
                if rn <= max((rtol if final else 1e-3) * r0, floor):
                    break
                if patience is not None and k >= patience and rn > 0.1 * r0:
                    raise RuntimeError("Newton made no headway in %d iterations" % k)
                if ck is not None and ck.get("budget_end") is not None and last_step is not None:
                    # the step ahead judged by the last one, its solve longer by the digits a
                    # tighter tolerance asks for; rank 0's clock, every rank the same answer
                    more = 1.0
                    if getattr(self.solver, "forcing", False):
                        more = max(1.0, np.log(self.solver.rtol) / np.log(last_step["rtol"]))
                    need = last_step["setup"] + 1.25 * more * last_step["solve"] + last_step["rest"]
                    if self.comm.bcast(time.time() + need > ck["budget_end"] if self.comm.rank == 0 else None, root=0):
                        raise ForwardStopped("stopped after %d Newton steps, in stage %d of %d (the next would not end "
                                             "within the job)" % (its, si + 1, len(stages)))
                f0, s0 = self.counters["factor_s"], self.counters["solve_s"]
                if ck is not None and ck.get("budget_end") is not None and hasattr(self.solver, "deadline"):
                    # the linear solve gives up what it has when the job is nearly over: the
                    # step is then an inexact one, and it is kept
                    self.solver.deadline = ck["budget_end"] - (60.0 if last_step is None else 1.5 * last_step["rest"] + 20.0)
                J = self.linearization(u, p, m3, eps2)
                self._factorize(J)
                rhs = self.generate_state()
                self.join(ru, rp, rhs)
                rhs.scale(-1.0)
                d = self.generate_state()
                self._solve(d, rhs)
                du, dp = self.split(d)
                # the step: Newton's whole one if it takes the residual down by ``self.good_step``;
                # if not, shorter ones are tried as long as they do better, and the best is kept
                # (far from the solution a whole step often lowers the residual by a few percent
                # where half of it lowers it by half)
                alpha, best = 1.0, None
                for _ in range(12):
                    ut, pt = u.copy().axpy(alpha, du), p.copy().axpy(alpha, dp)
                    rut, rpt = self.residual(ut, pt, m3, eps2)
                    rt = np.sqrt(rut.inner(rut) + rpt.inner(rpt))
                    if best is not None and rt >= best[0] and best[0] < (1.0 - 1e-4 * best[1]) * rn:
                        break                                    # shorter is worse: the best so far
                    if best is None or rt < best[0]:
                        best = (rt, alpha, ut, pt)
                    if rt < min(self.good_step, 1.0 - 1e-4 * alpha) * rn:
                        break
                    alpha *= 0.5
                alpha, ut, pt = best[1:]
                if self.verbose and final and alpha < 1.0:
                    print("      step shortened to %.3g of Newton's" % alpha, flush=True)
                u, p = ut, pt
                its += 1
                self.counters["newton"] += 1
                if ck is not None:
                    self._ck_write(u, p, done=False, stages=stages, stage=si, r0=float(r0), newton=its,
                                   seconds=t_before + time.time() - t0, jobs=jobs, rn_last=float(rn))
                    # rank 0's clock: every rank stops at the same step
                    if getattr(self.solver, "timed_out", False) or self.comm.bcast(
                            ck["deadline"] is not None and time.time() > ck["deadline"] if self.comm.rank == 0 else None, root=0):
                        raise ForwardStopped("stopped after %d Newton steps, in stage %d of %d"
                                             % (its, si + 1, len(stages)))
                last_step = dict(setup=self.counters["factor_s"] - f0, solve=self.counters["solve_s"] - s0,
                                 rtol=float(getattr(self.solver, "rtol", 1e-6)))
                last_step["rest"] = time.time() - t_step - last_step["setup"] - last_step["solve"]
            else:
                if final:
                    raise RuntimeError("Newton did not converge in %d iterations" % max_iter)
        self.join(u, p, state)
        self._state0 = state.copy()
        if remember:
            self._m0 = x[PARAMETER].copy()               # the parameter of that state
        self.last_forward = dict(seconds=t_before + time.time() - t0, newton=its, stages=len(stages),
                                 residual=float(rn / r0), residual_norm=float(rn), loads_norm=float(self.rest_residual(m3)))
        if ck is not None:
            self.last_forward.update(jobs=jobs, krylov=ck.get("krylov", [])
                                     + [int(i) for i in getattr(self.solver, "iterations", [])])
            self._ck_write(u, p, done=True, stages=stages, stage=len(stages) - 1, r0=float(r0), newton=its,
                           seconds=self.last_forward["seconds"], residual=self.last_forward["residual"], jobs=jobs)
        return state

    # ------------------------------------------------- the inverse problem
    def _at(self, x):
        """Make the Jacobian at ``x``'s state and parameter current (factorized)."""
        key = (float(x[STATE].inner(x[STATE])), float(x[PARAMETER].inner(x[PARAMETER])))
        if self._lin is None or self._lin["key"] != key:
            u, p = self.split(x[STATE])
            m3 = self.to_bed(x[PARAMETER])
            J = self.linearization(u, p, m3)
            self._factorize(J)
            if getattr(self.solver, "forcing", False):
                self.solver.rtol = self.solver.rtol_tight       # the adjoint and incremental solves
            self._lin = {"key": key, "J": J, "u": u, "p": p, "m3": m3}
        elif getattr(self, "_factored", None) is not self._lin["J"]:
            # a forward solve in between (a line search) factorized another matrix
            self._factorize(self._lin["J"])
        return self._lin

    def solveAdj(self, adj, x, adj_rhs):
        self._at(x)
        self.counters["adjoint"] += 1
        self._solve(adj, adj_rhs)                # J is symmetric
        return adj

    def evalGradientParameter(self, x, out):
        """``C^T lambda``: the friction's derivative in the parameter, tested with the
        adjoint velocity, on the bed; then onto the footprint."""
        u, p = self.split(x[STATE])
        lu, lp = self.split(x[ADJOINT])
        m3 = self.to_bed(x[PARAMETER])
        loc = self._loc(u, p, m3, lu, lp)
        g3 = assemble_boundary_vector(self.Vf, self.bed_batches.groups,
                                      self.bed_kernel.element_vectors(M, loc))
        self.from_bed(g3, out)
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        lin = self._at(x)
        if "C" not in lin:
            loc = self._loc(lin["u"], lin["p"], lin["m3"])
            # the domain part is zero (only the bed sees the friction); the boundary entries
            # are folded in as in the Jacobian, which also serves ranks without a bed face
            bb, bk = self.base_batches, self.base_kernel
            C3 = assemble_matrix_csr(self.Vv, self.Vf, self.batches.groups,
                                     self.kernel.element_matrices_or_chunks(V, M, loc, params=(self.eps2,)), self.ne,
                                     boundary=(bb.tables(self.Vv), bb.tables(self.Vf), bk.element_matrices(V, M, loc)))
            lin["C"] = mfem.ParMult(C3, self.P2to3)
            lin["C3"] = C3
        self.gauss_newton_approx = bool(gauss_newton_approx)
        if not self.gauss_newton_approx:
            raise NotImplementedError("the full Hessian blocks: use gauss_newton_approx=True")

    def solveIncremental(self, out, rhs, is_adj):
        self._at_check()                           # the solver holds this point's Jacobian
        self.counters["incremental"] += 1
        self._solve(out, rhs, incremental=True)    # J = J^T
        return out

    def apply_ij(self, i, j, dir, out):
        C = self._lin["C"]
        if (i, j) == (ADJOINT, PARAMETER):
            cu = self.Vv.vector()
            C.Mult(dir.hypre, cu.hypre)
            out.zero()
            out.array[:self.nv] = cu.array
            return out
        if (i, j) == (PARAMETER, ADJOINT):
            C.MultTranspose(self.velocity(dir).hypre, out.hypre)
            return out
        raise NotImplementedError("block (%d, %d) under Gauss-Newton is zero" % (i, j))

    def block_is_zero(self, i, j):
        return self.gauss_newton_approx and (i, j) not in ((ADJOINT, PARAMETER), (PARAMETER, ADJOINT))

    def misfit_hessian_block(self, misfit, M, batch=None):
        """The Gauss-Newton misfit Hessian applied to the columns of ``M`` (``(n_param, k)``,
        numpy): ``C^T J^{-1} B^T W B J^{-1} C M`` with every solve a block solve of the
        factorization at the linearization point.  One rank, the direct solver.  ``batch``
        columns are solved at a time: 64, or fewer where the state is large (a block solve
        holds about six arrays of ``batch`` states on the card beside the factors)."""
        import cupy as cp

        if batch is None:
            batch = max(4, min(64, int(2.4e8 // (self.nv + self.np))))

        lin = self._lin
        if "Cs" not in lin:
            lin["Cs"] = hypre_to_scipy(lin["C"]).tocsr()
            lin["Bs"], lin["w"] = misfit.scipy_BW()
        Cs, Bs, w = lin["Cs"], lin["Bs"], lin["w"]
        self._at_check()
        out = np.zeros_like(M)
        n = self.nv + self.np
        for a in range(0, M.shape[1], batch):
            cols = slice(a, min(a + batch, M.shape[1]))
            Cm = Cs @ M[:, cols]
            R = np.zeros((n, Cm.shape[1]))
            R[:self.nv] = -Cm
            X = self.solver.solve_many(cp.asarray(R, order="F"))           # u_hat = -J^{-1} C m
            Xu = cp.asnumpy(X[:self.nv])
            W = Bs.T @ (w[:, None] * (Bs @ Xu))
            R[:] = 0.0
            R[:self.nv] = -W
            L = self.solver.solve_many(cp.asarray(R, order="F"))           # p_hat = -J^{-T} W u_hat
            out[:, cols] = Cs.T @ cp.asnumpy(L[:self.nv])
            self.counters["incremental"] += 2 * Cm.shape[1]
        return out

    def misfit_hessian_sqrt_block(self, misfit, Z, batch=None):
        """``C^T J^{-1} B^T W^{1/2} Z`` for white columns ``Z`` (``(n_data, k)``, numpy): vectors whose
        covariance is the Gauss-Newton misfit Hessian, by one block solve each (the right-hand
        sides of posterior samples that do not need a factor of the Hessian)."""
        import cupy as cp

        if batch is None:
            batch = max(4, min(64, int(2.4e8 // (self.nv + self.np))))
        lin = self._lin
        if "Cs" not in lin:
            lin["Cs"] = hypre_to_scipy(lin["C"]).tocsr()
            lin["Bs"], lin["w"] = misfit.scipy_BW()
        Cs, Bs, w = lin["Cs"], lin["Bs"], lin["w"]
        self._at_check()
        out = np.zeros((Cs.shape[1], Z.shape[1]))
        n = self.nv + self.np
        if not hasattr(self.solver, "solve_many"):
            # a solver without block solves (the matrix-free one): a solve a column
            rhs, sol = self.generate_state(), self.generate_state()
            for k in range(Z.shape[1]):
                rhs.array[:] = 0.0
                rhs.array[:self.nv] = Bs.T @ (np.sqrt(w) * Z[:, k])
                self._solve(sol, rhs, incremental=True)
                out[:, k] = Cs.T @ np.asarray(sol.array[:self.nv])
                self.counters["incremental"] += 1
            return out
        for a in range(0, Z.shape[1], batch):
            cols = slice(a, min(a + batch, Z.shape[1]))
            R = np.zeros((n, Z[:, cols].shape[1]))
            R[:self.nv] = Bs.T @ (np.sqrt(w)[:, None] * Z[:, cols])
            L = self.solver.solve_many(cp.asarray(R, order="F"))
            out[:, cols] = Cs.T @ cp.asnumpy(L[:self.nv])
            self.counters["incremental"] += Z[:, cols].shape[1]
        return out

    def _at_check(self):
        if getattr(self, "_factored", None) is not self._lin["J"]:
            self._factorize(self._lin["J"])

    # ----------------------------------------------------------- quantities
    def surface_nodes(self):
        """The nodes of the Q2 velocity on the upper surface that this rank owns: ``idx``
        (their indices in a velocity component's block of true dofs), ``xyz`` and ``area``,
        the row sums ``int phi_i ds`` of the surface mass matrix (km^2; for Q2 these are
        Simpson's weights, all positive)."""
        if not hasattr(self, "_surface"):
            bb = get_boundary_batches(self.mesh, self.qdeg, [an.Columns.SURFACE], self.comm, space=self.Vv)
            k = BoundaryKernel(lambda u, p, m, b, v, q, x, n: v.val[0], self.spaces, bb)
            loc = self._loc(self.Vv.vector(), self.Vs.vector(), self.Vf.vector())
            a = assemble_boundary_vector(self.Vv, bb.groups, k.element_vectors(V, loc))
            ns = self.nv // 3
            area = np.array(a.array[:ns])
            idx = np.nonzero(area != 0.0)[0]
            if np.any(area[idx] <= 0.0):
                raise RuntimeError("a surface node with a negative area: a surface face too distorted for the row-sum rule")
            X = hm.FunctionSpace.H1(self.mesh, self.order).coordinates()
            self._surface = dict(idx=idx, xyz=X[idx], area=area[idx])
        return self._surface

    def flux_vector(self, region="all"):
        """``f`` with ``RHO_ICE f . u`` the ice mass flux (Gt/a) out through the margins facing
        the ocean in ``region`` (one of ``REGIONS``): rho int u . n ds."""
        if not hasattr(self, "_flux_vecs"):
            self._flux_vecs = {}
        if region not in self._flux_vecs:
            bb = get_boundary_batches(self.mesh, self.qdeg, [an.Columns.MARINE], self.comm, space=self.Vv)
            k = BoundaryKernel(lambda u, p, m, b, v, q, x, n: jnp.dot(v.val, n) * region_weight(region, x),
                               self.spaces, bb)
            loc = self._loc(self.Vv.vector(), self.Vs.vector(), self.Vf.vector())
            self._flux_vecs[region] = assemble_boundary_vector(self.Vv, bb.groups, k.element_vectors(V, loc))
        return self._flux_vecs[region]

    def mass_flux(self, s, region="all"):
        """Ice mass flux (Gt/a) out through the margins facing the ocean in ``region``."""
        return RHO_ICE * self.flux_vector(region).inner(self.velocity(s))

    def summary(self):
        return dict(elements=int(self.mesh.GetGlobalNE()), velocity_dofs=int(self.Vv.fes.GlobalTrueVSize()),
                    pressure_dofs=int(self.Vs.fes.GlobalTrueVSize()), parameters=int(self.Vm.fes.GlobalTrueVSize()),
                    refine=self.refine, order=self.order, friction_order=self.friction_order, ranks=self.comm.size,
                    setup_s=self.t_setup)


# --------------------------------------------------------- linear solvers
class BlockFGMRES:
    """FGMRES on the monolithic Jacobian, right-preconditioned by the block lower
    triangular ``[[A, 0], [B, -S]]^{-1}``: for the velocity block, every rank's diagonal
    block factorized by cuDSS on its GPU (``BlockJacobiCuDSS``), and for the Schur
    complement an AMG V-cycle on the viscosity-weighted pressure mass ``M_{1/eta}``, the
    standard choice for variable-viscosity Stokes.  For meshes too large for one
    factorization; use it with ``IceSheet(partition="columns")``."""

    def __init__(self, problem, rtol=1e-6, maxit=1000, kdim=200, block_precision="single"):
        self.pb = problem
        self.block_precision = block_precision
        self.comm = problem.comm
        self.rtol, self.maxit, self.kdim = rtol, maxit, kdim
        self.iterations = []
        self.times = {"setup": 0.0, "solve": 0.0}
        self._block = None

    def factorize(self, J):
        t0 = time.time()
        pb = self.pb
        A, Bt, Bd = pb._keep_blocks
        u, p, m3, eps2 = pb._block_state
        S = pb.pressure_mass(u, p, m3, eps2)
        # one block solver for the run: the pattern does not change, so a new Jacobian is
        # a refactorization in place
        if self._block is None:
            self._block = BlockJacobiCuDSS(A, self.block_precision)
        else:
            self._block.update(A)
        amgS = mfem.HypreBoomerAMG(S)
        amgS.SetPrintLevel(0)
        negS = mfem.ScaledOperator(amgS, -1.0)
        self.offsets = mfem.intArray([0, pb.nv, pb.nv + pb.np])
        P = mfem.BlockLowerTriangularPreconditioner(self.offsets)
        P.SetDiagonalBlock(0, self._block)
        P.SetBlock(1, 0, Bd)
        P.SetDiagonalBlock(1, negS)
        solver = mfem.FGMRESSolver(self.comm)
        solver.SetOperator(J)
        solver.SetPreconditioner(P)
        solver.SetRelTol(self.rtol)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(self.maxit)
        solver.SetKDim(self.kdim)
        solver.SetPrintLevel(0)
        self._keep = (J, A, Bt, Bd, S, amgS, negS, P)       # MFEM holds raw pointers to these
        self.solver = solver
        self.times["setup"] += time.time() - t0
        return self

    def solve(self, x, b):
        t0 = time.time()
        x.zero()
        self.solver.Mult(b.hypre, x.hypre)
        self.iterations.append(int(self.solver.GetNumIterations()))
        if not self.solver.GetConverged() and self.comm.rank == 0:
            print("    FGMRES did not converge: %d iterations, final norm %.2e"
                  % (self.solver.GetNumIterations(), self.solver.GetFinalNorm()), flush=True)
        self.times["solve"] += time.time() - t0
        return x

    def describe(self):
        it = np.array(self.iterations) if self.iterations else np.zeros(1)
        return {"solver": "fgmres, subdomain factorizations", "rtol": self.rtol, "solves": len(self.iterations),
                "iterations_mean": float(it.mean()), "iterations_max": int(it.max()),
                "iterations": [int(i) for i in self.iterations], "times": dict(self.times)}


class BlockJacobiCuDSS(mfem.PyOperator):
    """Block Jacobi for the velocity block: the rank's diagonal block, factorized by cuDSS
    on the rank's GPU, applied to the rank's part of a vector.  With a partition by whole
    columns (``IceSheet(partition="columns")``) the strong vertical coupling of the thin
    ice is inside the blocks and FGMRES only has the horizontal coupling between
    subdomains to resolve."""

    def __init__(self, A, precision="single"):
        import gpu_direct

        n = A.Height()
        mfem.PyOperator.__init__(self, n)
        # a preconditioner: single-precision factors (half the memory) are as good here;
        # the velocity block is symmetric positive definite, so Cholesky
        self.solver = gpu_direct.CuDSSDirect(MPI.COMM_SELF, precision=precision, ir_max=0 if precision == "single" else 6,
                                             spd=True)
        self.solver.factorize(A)
        self.x, self.y = _Host(n), _Host(n)
        self._A = A

    def update(self, A):
        """Refactorize for a new matrix of the same pattern."""
        self.solver.factorize(A)
        self._A = A

    def Mult(self, xin, yout):
        self.x.array = np.asarray(xin.GetDataArray()).copy()
        self.solver.solve(self.y, self.x)
        yout.Assign(self.y.array)


class _Host:
    def __init__(self, n):
        self.array = np.zeros(n)


class ScipyDirect:
    """The monolithic system gathered to rank 0 and factorized by SuperLU: a host solve
    for small meshes, without a GPU."""

    def __init__(self, comm):
        self.comm = comm

    def factorize(self, J):
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu

        loc = hypre_to_scipy(J)
        parts = self.comm.gather(loc, root=0)
        self.sizes = self.comm.allgather(loc.shape[0])
        if self.comm.rank == 0:
            self.lu = splu(sp.vstack(parts).tocsc())
        return self

    def solve(self, x, b):
        parts = self.comm.gather(np.asarray(b.array), root=0)
        sol = None
        if self.comm.rank == 0:
            sol = self.lu.solve(np.concatenate(parts))
            sol = np.split(sol, np.cumsum(self.sizes)[:-1])
        x.array[:] = self.comm.scatter(sol, root=0)
        return x


class SurfaceVelocityMisfit:
    """Horizontal surface velocity at points, independent Gaussian noise per datum:
    ``1/2 sum ((B u - d) / sigma)^2``."""

    def __init__(self, problem, targets, data_xy, sigma, missing="warn"):
        self.problem = problem
        T = np.repeat(np.asarray(targets, float), 2, axis=0)
        comps = np.zeros((T.shape[0], 3))
        comps[0::2, 0] = 1.0
        comps[1::2, 1] = 1.0
        self.B = hm.assemblePointwiseObservation(problem.Vv, T, components=comps, missing=missing)
        self.d = self.B.scatter(np.asarray(data_xy, float).reshape(-1))
        w = np.repeat(1.0 / np.asarray(sigma, float) ** 2, 2)
        self.w = self.B.scatter(w)

    def located(self):
        """Which of the targets the mesh's point location found (it misses some in thin,
        steep elements; their rows of ``B`` are empty)."""
        return (np.asarray(self.B.owner) != np.iinfo(np.int32).max)[0::2]

    def _Bu(self, s):
        out = self.B.createVecLeft()
        self.B.mult(self.problem.velocity(s), out)
        return out

    def prediction(self, s):
        """The model's horizontal velocity at the data points, ``(n, 2)`` on rank 0."""
        v = self.B.gather(self._Bu(s))
        return None if v is None else np.asarray(v).reshape(-1, 2)

    def scipy_BW(self):
        """The observation matrix (scipy, velocity dofs to data) and the weights (one rank)."""
        import scipy.sparse as sp
        B = self.B
        Bs = sp.coo_matrix((B._vals, (B._rows, B._cols)), shape=(B.n_owned, self.problem.nv)).tocsr()
        return Bs, np.asarray(self.w.array)

    def cost(self, x):
        r = self._Bu(x[STATE]).axpy(-1.0, self.d)
        return 0.5 * self.problem.comm.allreduce(float(np.sum(self.w.array * r.array ** 2)), op=MPI.SUM)

    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            r = self._Bu(x[STATE]).axpy(-1.0, self.d)
            r.array[:] *= self.w.array
            gu = self.problem.Vv.vector()
            self.B.multTranspose(r, gu)
            out.array[:self.problem.nv] = gu.array
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass

    def apply_ij(self, i, j, dir, out):
        out.zero()
        if (i, j) == (STATE, STATE):
            r = self._Bu(dir)
            r.array[:] *= self.w.array
            gu = self.problem.Vv.vector()
            self.B.multTranspose(r, gu)
            out.array[:self.problem.nv] = gu.array
        return out

    def block_is_zero(self, i, j):
        return (i, j) != (STATE, STATE)


class SurfaceIntegralMisfit:
    """The paper's misfit: the integral over the upper surface of the squared mismatch of
    the horizontal velocity over its noise variance, ``1/2 int |B u - d|^2 / sigma^2 ds``,
    by the row-sum (Simpson) rule on the surface nodes of the Q2 velocity.  In matrix
    terms a diagonal noise covariance with entries ``sigma_i^2 / a_i``, ``a_i = int phi_i
    ds`` in km^2: the observations are a field, refined with the mesh, and what they say
    does not grow with the number of nodes (that of a plain sum over the nodes does).

    ``d`` (n, 2) and ``sigma`` (n,) at this rank's surface nodes
    (``problem.surface_nodes()``), ``have`` the nodes that have a datum, ``rows`` their
    places in the run's global data arrays."""

    def __init__(self, problem, d, sigma, have=None, rows=None, n_global=None):
        self.problem = problem
        sn = problem.surface_nodes()
        ns = problem.nv // 3
        have = np.ones(len(sn["idx"]), bool) if have is None else np.asarray(have, bool)
        self.idx = sn["idx"][have]
        self.n = len(self.idx)
        self.cols = np.concatenate([self.idx, ns + self.idx])          # the x block, then the y block
        d = np.asarray(d, float)[have]
        self.dvals = np.concatenate([d[:, 0], d[:, 1]])
        w = sn["area"][have] / np.asarray(sigma, float)[have] ** 2
        self.wvals = np.concatenate([w, w])
        self.rows = np.arange(self.n) if rows is None else np.asarray(rows)[have]
        self.n_global = problem.comm.allreduce(self.n, op=MPI.SUM) if n_global is None else int(n_global)

    def _Bu(self, s):
        return np.asarray(s.array)[self.cols]

    def prediction(self, s):
        """The model's horizontal surface velocity at the data, ``(n_global, 2)`` on rank 0."""
        comm = self.problem.comm
        v = self._Bu(s)
        parts = comm.gather((self.rows, v[:self.n], v[self.n:]), root=0)
        if comm.rank != 0:
            return None
        out = np.full((self.n_global, 2), np.nan)
        for rows, vx, vy in parts:
            out[rows, 0], out[rows, 1] = vx, vy
        return out

    def cost(self, x):
        r = self._Bu(x[STATE]) - self.dvals
        return 0.5 * self.problem.comm.allreduce(float(np.sum(self.wvals * r * r)), op=MPI.SUM)

    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            out.array[self.cols] = self.wvals * (self._Bu(x[STATE]) - self.dvals)
        return out

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass

    def apply_ij(self, i, j, dir, out):
        out.zero()
        if (i, j) == (STATE, STATE):
            out.array[self.cols] = self.wvals * self._Bu(dir)
        return out

    def block_is_zero(self, i, j):
        return (i, j) != (STATE, STATE)

    def scipy_BW(self):
        import scipy.sparse as sp
        n = len(self.cols)
        Bs = sp.coo_matrix((np.ones(n), (np.arange(n), self.cols)), shape=(n, self.problem.nv)).tocsr()
        return Bs, self.wvals

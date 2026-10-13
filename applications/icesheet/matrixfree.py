# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""The Newton systems of the ice sheet solved without their matrices, for meshes of a
billion unknowns.

The Jacobian ``K = [[A, B^T], [B, 0]]`` is never assembled: a product with it is one pass of
the library's element kernels over the mesh (a forward tangent through the residual), on the
rank's GPU.  FGMRES is preconditioned by the block lower triangular
``[[\hat A, 0], [B, -\hat S]]^{-1}``, both parts made for ice that is thin:

* the nodes of a mesh of columns stand in vertical lines, and the strong coupling of thin
  ice is along them.  ``\hat A^{-1}`` is a two-level method: an exact solve on every
  *column of nodes* (the blocks ``D`` of ``A`` that couple the unknowns of one line: block
  tridiagonal, factorized for all the columns of a rank at once), then the problem
  *collapsed over the depth* (velocities that do not change along a column: a 2D problem
  with three unknowns a column, solved by preconditioned conjugate gradients), then the
  columns again;
* ``\hat S`` is the Schur complement of the columns, ``B D^{-1} B^T``, which is what goes
  with that ``\hat A``: the pressure mass over the viscosity, the usual choice, holds modes
  that live in a column near the bed and its neighbours up to a few hundred times too
  stiff, and the count of Krylov iterations then grows with the mesh.  ``B D^{-1} B^T``
  couples columns of pressure nodes up to two cells apart; its blocks are found by probing
  (products with vectors that are one on a level of every fifth column each way), and it is
  inverted on every patch of 3 x 3 columns, the centre column kept (restricted additive
  Schwarz).

Everything a rank holds is on its GPU in two layouts, ``(columns, levels, 3)`` for the
velocity and ``(columns, levels)`` for the pressure, the columns it owns first; the columns
of other ranks that its elements touch are exchanged through the host (they are the rims of
the rank's patch of the footprint, a megabyte).

Needs ``cupy`` beside the library.  The mesh has to be cut into whole columns
(``IceSheet(partition="columns")``).
"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import mfem.par as mfem
from mpi4py import MPI

import hippymfem as hm
from hippymfem.common.linalg import scipy_to_hypre
from hippymfem.common.parvector import ParVector
from hippymfem.fem.kernel import _chunk_slices

U, P, M, B, V, Q = range(6)


def _private_kernel_cache(comm):
    """A folder of this rank's own for the kernels CuPy compiles, seeded from the shared one.

    CuPy compiles a kernel at its first use and keeps it in a folder, by default one in the
    home directory.  Sixty ranks that write the same new kernel there at once fail on a
    network file system (a stale file handle).  Returns the two folders, or None on one rank."""
    import shutil
    import tempfile

    if comm.size == 1:
        return None
    shared = os.environ.get("CUPY_CACHE_DIR", os.path.expanduser("~/.cupy/kernel_cache"))
    mine = os.path.join(tempfile.gettempdir(), "cupy_kernels_%s_%s_%04d"
                        % (os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", str(os.getppid())), comm.rank))
    if not os.path.isdir(mine):
        if os.path.isdir(shared):
            shutil.copytree(shared, mine, dirs_exist_ok=True)
        os.makedirs(mine, exist_ok=True)
    os.environ["CUPY_CACHE_DIR"] = mine
    return shared, mine


def _return_kernels(comm, folders):
    """What the first and the last rank compiled goes to the shared folder (one writer at a
    time), so that the next job starts with it: the first rank has no columns of other
    ranks, the last one has."""
    import shutil

    if folders is None:
        return
    shared, mine = folders
    for writer in (0, comm.size - 1):
        if comm.rank == writer:
            os.makedirs(shared, exist_ok=True)
            for f in os.listdir(mine):
                dst = os.path.join(shared, f)
                if not os.path.exists(dst) and os.path.isfile(os.path.join(mine, f)):
                    shutil.copyfile(os.path.join(mine, f), dst + ".part%d" % os.getpid())
                    os.replace(dst + ".part%d" % os.getpid(), dst)
        comm.Barrier()


def _jax(a):
    """A CuPy array as a JAX array on the same memory."""
    return jnp.from_dlpack(a)


@jax.jit
def _scatter(y, idx, vals):
    return y.at[idx.reshape(-1)].add(vals.reshape(-1))


class NodeColumns:
    """The nodes of a scalar Lagrange space on a mesh of whole columns, column by column.

    A rank's columns are numbered with the ones it owns first; a node's place in the layout
    ``(columns, levels)`` is ``column * L + level``, the levels from the bed up.

    Attributes
    ----------
    nco, ncl, L : the columns owned, the columns held (with those of other ranks), the levels
    place : ``(nodes,)``, the place of every local node of the space
    p2t, t2p : the true dof at every owned place, and the place of every true dof
    ix, iy : ``(ncl,)``, the columns on the lattice of the footprint (in units of ``step``)
    gcol : ``(ncl,)``, the global number of every column held
    """

    def __init__(self, space, comm, origin, step, cp):
        self.comm, self.cp = comm, cp
        fes = space.fes
        X = space._node_coordinates()
        nl = X.shape[0]
        nt, off = int(fes.GetTrueVSize()), int(fes.GetMyTDofOffset())
        v = space.vector()
        v.array[:] = off + np.arange(nt, dtype=np.float64)
        gid = np.rint(space.local_values(v)).astype(np.int64)
        owned = (gid >= off) & (gid < off + nt)
        if int(owned.sum()) != nt:
            raise RuntimeError("the true dofs of the space are not one local node each")
        ix = np.rint((X[:, 0] - origin[0]) / step).astype(np.int64)
        iy = np.rint((X[:, 1] - origin[1]) / step).astype(np.int64)
        if nl and max(np.abs(X[:, 0] - origin[0] - ix * step).max(), np.abs(X[:, 1] - origin[1] - iy * step).max()) > 1e-6 * step:
            raise RuntimeError("the nodes are not on the lattice of the footprint")
        key = ix * (1 << 31) + iy
        uk, first, col = np.unique(key, return_index=True, return_inverse=True)
        col = col.ravel()
        ncl = uk.size
        L = nl // max(ncl, 1)
        if nl and (nl != ncl * L or np.any(np.bincount(col, minlength=ncl) != L)):
            raise RuntimeError("the nodes of this rank do not form whole columns (cut the mesh with partition='columns')")
        order = np.lexsort((X[:, 2], col))                      # by column, from the bed up
        lev = np.empty(nl, dtype=np.int64)
        lev[order] = np.tile(np.arange(L), ncl)
        own_c = owned[order].reshape(ncl, L)
        if np.any(own_c.any(axis=1) != own_c.all(axis=1)):
            raise RuntimeError("a column of nodes is owned by two ranks")
        own_c = own_c[:, 0] if L else np.zeros(0, bool)
        nco = int(own_c.sum())
        new = np.empty(ncl, dtype=np.int64)                     # the owned columns first
        new[np.nonzero(own_c)[0]] = np.arange(nco)
        new[np.nonzero(~own_c)[0]] = nco + np.arange(ncl - nco)
        col = new[col]
        self.nl, self.nt, self.nco, self.ncl, self.L = nl, nt, nco, ncl, L
        self.place = col * L + lev
        self.ix, self.iy = np.empty(ncl, np.int64), np.empty(ncl, np.int64)
        self.ix[new], self.iy[new] = ix[first], iy[first]
        self.t2p = np.empty(nt, dtype=np.int64)
        self.t2p[gid[owned] - off] = self.place[owned]
        self.p2t = np.argsort(self.t2p)
        # the global numbers of the columns, and who sends which to whom
        c0 = comm.scan(nco) - nco
        v.array[gid[owned] - off] = c0 + col[owned]
        gl = np.rint(space.local_values(v)).astype(np.int64)
        self.gcol = np.empty(ncl, dtype=np.int64)
        self.gcol[col] = gl
        self.c0 = c0
        self.starts = np.array(comm.allgather(c0) + [comm.allreduce(nco)], dtype=np.int64)
        ghost = self.gcol[nco:]
        owner = np.searchsorted(self.starts, ghost, side="right") - 1
        want, self.recv = [None] * comm.size, {}
        for o in np.unique(owner):
            idx = np.nonzero(owner == o)[0]
            self.recv[int(o)] = cp.asarray(nco + idx)
            want[int(o)] = ghost[idx]
        got = comm.alltoall(want)
        self.send = {r: cp.asarray(g - c0) for r, g in enumerate(got) if g is not None and len(g)}
        self.t2p_g, self.p2t_g = cp.asarray(self.t2p), cp.asarray(self.p2t)

    def element_places(self, edofs, vdim=1):
        """The places of the dofs of elements (MFEM's local numbering, ordered by component)
        in the layout ``(columns, levels, vdim)``."""
        if vdim == 1:
            return self.place[edofs]
        return self.place[edofs % self.nl] * vdim + edofs // self.nl

    # ------------------------------------------------------------- exchange
    def _swap(self, a, src, dst, tag, add):
        cp, comm = self.cp, self.comm
        tail = a.shape[1:]
        bufs = {r: np.empty((int(idx.size),) + tail, dtype=a.dtype) for r, idx in dst.items()}
        reqs = [comm.Irecv(b, source=r, tag=tag) for r, b in bufs.items()]
        out = [(r, cp.asnumpy(a[idx])) for r, idx in src.items()]
        reqs += [comm.Isend(b, dest=r, tag=tag) for r, b in out]
        MPI.Request.Waitall(reqs)
        for r, b in bufs.items():
            if add:
                a[dst[r]] += cp.asarray(b)
            else:
                a[dst[r]] = cp.asarray(b)
        return a

    def fill(self, a):
        """The columns of other ranks in ``a`` (first axis: the columns held) from their owners."""
        return self._swap(a, self.send, self.recv, 71, False)

    def reduce(self, a):
        """What this rank's elements gave to the columns of other ranks, added at their owners."""
        return self._swap(a, self.recv, self.send, 72, True)


class ColumnBlocks:
    """The blocks of a symmetric matrix on columns of nodes, block tridiagonal in pairs of
    levels (a level of second-order nodes couples with the two on either side), factorized
    and solved for all the columns at once.

    ``band[c, i, d]`` is the ``3 x 3`` block between levels ``i`` and ``i + d - 2`` of column ``c``.
    """

    def __init__(self, band, cp):
        self.cp = cp
        nc, L = band.shape[:2]
        nb = (L + 1) // 2
        self.nc, self.L, self.nb = nc, L, nb
        diag = cp.zeros((nc, nb, 6, 6))
        low = cp.zeros((nc, max(nb - 1, 0), 6, 6))
        if 2 * nb > L:
            diag[:, -1, 3:, 3:] = cp.eye(3)                     # the level that is not there
        for s in range(2):
            for t in range(2):
                blk = band[:, s::2, t - s + 2]
                diag[:, :blk.shape[1], 3 * s:3 * s + 3, 3 * t:3 * t + 3] = blk
                if t >= s:
                    blk = band[:, 2 + s::2, t - s]
                    low[:, :blk.shape[1], 3 * s:3 * s + 3, 3 * t:3 * t + 3] = blk
        self.low = low
        self.sinv = cp.empty_like(diag)
        self.g = cp.empty_like(low)
        self.sinv[:, 0] = cp.linalg.inv(diag[:, 0])
        for k in range(1, nb):
            self.g[:, k - 1] = low[:, k - 1] @ self.sinv[:, k - 1]
            self.sinv[:, k] = cp.linalg.inv(diag[:, k] - self.g[:, k - 1] @ low[:, k - 1].transpose(0, 2, 1))

    def solve(self, r):
        """``D^{-1} r`` for ``r`` of shape ``(columns, L, 3)``."""
        cp, nb, L = self.cp, self.nb, self.L
        y = cp.zeros((self.nc, 2 * nb, 3))
        y[:, :L] = r
        y = y.reshape(self.nc, nb, 6, 1)
        for k in range(1, nb):
            y[:, k] -= self.g[:, k - 1] @ y[:, k - 1]
        y[:, nb - 1] = self.sinv[:, nb - 1] @ y[:, nb - 1]
        for k in range(nb - 2, -1, -1):
            y[:, k] = self.sinv[:, k] @ (y[:, k] - self.low[:, k].transpose(0, 2, 1) @ y[:, k + 1])
        return y.reshape(self.nc, 2 * nb, 3)[:, :L]


def _kernels(gk, volume):
    """The passes over one group of elements, compiled: the Jacobian's blocks applied to
    directions (forward tangents through the residual, which is the gradient of the density
    in the test functions)."""
    R = gk._functional()

    def grad_v(d, tabs, geo, params):
        return jax.grad(lambda v: R((d[U], d[P], d[M], d[B], v, d[Q]), tabs, geo, params))(d[V])

    def grad_q(d, tabs, geo, params):
        return jax.grad(lambda q: R((d[U], d[P], d[M], d[B], d[V], q), tabs, geo, params))(d[Q])

    def swap(d, u=None, p=None):
        return (d[U] if u is None else u, d[P] if p is None else p) + tuple(d[2:])

    def stokes(d, du, dp, tabs, geo, params):
        return jax.jvp(lambda u, p: (grad_v(swap(d, u, p), tabs, geo, params), grad_q(swap(d, u, p), tabs, geo, params)),
                       (d[U], d[P]), (du, dp))[1]

    def velocity(d, du, tabs, geo, params):
        return jax.jvp(lambda u: grad_v(swap(d, u), tabs, geo, params), (d[U],), (du,))[1]

    def div(d, du, tabs, geo, params):
        return jax.jvp(lambda u: grad_q(swap(d, u), tabs, geo, params), (d[U],), (du,))[1]

    def grad(d, dp, tabs, geo, params):
        return jax.jvp(lambda p: grad_v(swap(d, p=p), tabs, geo, params), (d[P],), (dp,))[1]

    def make(f, nlead):
        return jax.jit(jax.vmap(f, in_axes=gk._axes(nlead))), gk._axes(nlead)

    out = {"velocity": make(velocity, 2)}
    if volume:
        out.update(stokes=make(stokes, 3), div=make(div, 2), grad=make(grad, 2))
    return out


class ColumnStokes:
    """FGMRES on the Newton system of an :class:`stokes.IceSheet`, matrix-free (see the
    module).  The problem's ``solver``: ``factorize`` takes the state the system is
    linearized at, not a matrix.

    Parameters
    ----------
    rtol, maxit, restart : of FGMRES
    damping : of the solves on the columns in the velocity's two-level method (one sweep)
    sweeps, smooth_range : Chebyshev steps of the solves on the columns before and after the
        collapsed problem, and the part of the spectrum they take down (see ``smooth``)
    patch_bytes : device memory for the patches of the Schur complement inverted at a time
    schur_every : the patches are made anew at every ``schur_every``-th linearization point
        (they change with the viscosity, slowly once Newton's method is converging)
    coarse_rtol, coarse_maxit : of the conjugate gradients on the collapsed problem
    coarse, coarse_fill : their preconditioner, "ilu" (every rank's part factorized
        incompletely, with this level of fill) or "amg"
    chunk : elements in one launch of a pass
    forcing, rtol_loose : in a Newton solve, the tolerance of the linear solves chosen step by
        step between ``rtol_loose`` and ``rtol`` (the adjoint and incremental solves use ``rtol``)
    """

    matrix_free = True

    def __init__(self, problem, rtol=1e-6, maxit=300, restart=40, damping=0.6, sweeps=1, smooth_range=30.0, patch_bytes=2e9, schur_every=1,
                 coarse_rtol=1e-2, coarse_maxit=20, coarse="ilu", coarse_fill=1, chunk=65536, verbose=None,
                 forcing=False, rtol_loose=1e-2):
        self._kernel_folders = _private_kernel_cache(problem.comm)
        import cupy as cp

        t0 = time.time()
        self.cp = cp
        pb = self.pb = problem
        comm = self.comm = pb.comm
        self.rtol, self.maxit, self.restart = rtol, maxit, restart
        #: in a Newton solve: the tolerance of every linear solve between ``rtol_loose`` and
        #: ``rtol`` as the residual's fall allows (set by ``IceSheet._newton``)
        self.forcing, self.rtol_tight, self.rtol_loose = bool(forcing), rtol, max(rtol_loose, rtol)
        #: a ``time.time()`` value after which a solve returns what it has (a job's limit)
        self.deadline, self.timed_out = None, False
        self.damping, self.sweeps, self.smooth_range = damping, int(sweeps), float(smooth_range)
        self._power, self.lmax = None, None
        self.coarse_rtol, self.coarse_maxit = coarse_rtol, coarse_maxit
        self.coarse_solver, self.coarse_fill = coarse, int(coarse_fill)
        self.chunk = int(chunk)
        self.verbose = pb.verbose if verbose is None else (verbose and comm.rank == 0)
        self.iterations, self.coarse_its, self.probes = [], [], 0
        self.block_dtype, self.patch_bytes = cp.float32, float(patch_bytes)
        self.Be, self.pmid, self.factorizations, self.schur_every = None, None, 0, max(1, int(schur_every))
        self.times = {"setup": 0.0, "solve": 0.0, "blocks": 0.0, "collapsed": 0.0, "probe": 0.0,
                      "passes": 0.0, "coarse": 0.0, "schur": 0.0, "columns": 0.0}
        if pb.Vv.fes.GetOrdering() != mfem.Ordering.byNODES:
            raise NotImplementedError("the velocity's dofs have to be ordered by component")
        g = pb.columns.grid
        step = g.h / 2 ** pb.refine
        origin = (float(g.xn[0]), float(g.yn[0]))
        self.Vq = hm.FunctionSpace.H1(pb.mesh, pb.order)
        self.cu = NodeColumns(self.Vq, comm, origin, step / pb.order, cp)
        self.cq = NodeColumns(pb.Vs, comm, origin, step, cp)
        cu, cq = self.cu, self.cq
        self.nu, self.npr = cu.nco * cu.L * 3, cq.nco * cq.L
        if self.nu != pb.nv or self.npr != pb.np:
            raise RuntimeError("the columns do not hold the problem's unknowns")
        if len(pb.kernel.group_kernels) != 1:
            raise NotImplementedError("one kind of element")
        self.gk = pb.kernel.group_kernels[0]
        self.ne = int(self.gk.group.ne)
        self.k = _kernels(self.gk, True)
        tu, tp = self.gk.tables[U], self.gk.tables[P]
        self.nd = tu.nd
        self.eu = jnp.asarray(cu.element_places(tu.edofs, 3).astype(np.int32))
        self.ep = jnp.asarray(cq.element_places(tp.edofs).astype(np.int32))
        # the bed and the land margins: faces, with the dofs of the element behind each
        self.bgk = [gk for gk in pb.base_kernel.group_kernels if int(gk.group.ne)]
        self.bslot = {g_: i for i, g_ in enumerate(g_ for g_, gk in enumerate(pb.base_kernel.group_kernels) if int(gk.group.ne))}
        self.bk = [_kernels(gk, False) for gk in self.bgk]
        self.beu = [jnp.asarray(cu.element_places(gk.tables[U].edofs, 3).astype(np.int32)) for gk in self.bgk]
        self._structure(tu.edofs, [gk.tables[U].edofs for gk in self.bgk])
        self._check_layout()
        self.t_init = time.time() - t0
        self.times["setup"] += self.t_init

    # ------------------------------------------------------------ structure
    def _structure(self, edofs, bedofs):
        """Where the entries of an element's matrix go: into the blocks of the columns, and
        into the collapsed problem."""
        cp, cu, nd = self.cp, self.cu, self.nd
        L = cu.L
        node = edofs[:, :nd]                                    # the element's nodes (first component)
        pl = cu.place[node]
        ecol, elev = pl // L, pl % L
        if self.ne == 0:
            raise NotImplementedError("a rank without elements")
        same = ecol[0][:, None] == ecol[0][None, :]
        pa, pb_ = np.nonzero(same)
        # every element has the pattern of the first: the same nodes stand above each other
        ok = bool(np.all(ecol[:, pa] == ecol[:, pb_]))
        f_of = np.unique(ecol[0], return_inverse=True)[1].ravel()          # the node's line among the element's
        nf = int(f_of.max()) + 1
        rep = np.array([int(np.nonzero(f_of == f)[0][0]) for f in range(nf)])
        per = nd // nf
        lines = np.sort(ecol[:, rep], axis=1)
        ok = ok and nf * per == nd and int(same.sum()) == nf * per * per and bool(np.all(lines[:, 1:] != lines[:, :-1]))
        del lines
        if not self.comm.allreduce(ok, op=MPI.LAND):
            raise NotImplementedError("the elements are not all numbered alike along their columns")
        self.nf = nf
        c1, c2 = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        # (node pair in one line) x (component pair): the entry of the element matrix, and its 3 x 3 place
        self.src_r = cp.asarray((c1.ravel()[None, :] * nd + pa[:, None]).ravel())
        self.src_c = cp.asarray((c2.ravel()[None, :] * nd + pb_[:, None]).ravel())
        self._pa, self._pb = cp.asarray(np.repeat(pa, 9)), cp.asarray(np.repeat(pb_, 9))
        self._cc = cp.asarray(np.tile(c1.ravel() * 3 + c2.ravel(), pa.size))
        # the lines of an element in a fixed order, and the nodes line by line
        self.by_line = cp.asarray(np.argsort(f_of, kind="stable"))
        self.per_line = per
        ccols = ecol[:, rep]                                    # (ne, nf): the element's lines
        centre = self._centre(ccols[0])
        cells, firsts, ecell = np.unique(ccols[:, centre], return_index=True, return_inverse=True)
        self.ncell = cells.size
        self.cell_cols = ccols[firsts]                          # (ncell, nf)
        self.cell_of_centre = np.full(cu.ncl, -1, dtype=np.int64)
        self.cell_of_centre[cells] = np.arange(self.ncell)
        self._centre_line, self._rep = centre, rep
        self.geom = [(cp.asarray(ecol.astype(np.int32)), cp.asarray(elev.astype(np.int32)), cp.asarray(ecell.ravel().astype(np.int32)))]
        for eb in bedofs:
            plb = cu.place[eb[:, :nd]]
            cb, lb = plb // L, plb % L
            if not bool(np.all(cb[:, pa] == cb[:, pb_])):
                raise NotImplementedError("the elements behind the faces are not numbered as the others")
            cellb = self.cell_of_centre[cb[:, rep][:, centre]]
            if np.any(cellb < 0):
                raise RuntimeError("a face of an element that is not this rank's")
            self.geom.append((cp.asarray(cb.astype(np.int32)), cp.asarray(lb.astype(np.int32)), cp.asarray(cellb.astype(np.int32))))

    def _centre(self, cols):
        """Which of an element's lines is the one in the middle of its footprint."""
        cu = self.cu
        x, y = cu.ix[cols].astype(float), cu.iy[cols].astype(float)
        d = (x - x.mean()) ** 2 + (y - y.mean()) ** 2
        return int(np.argmin(d))

    def _check_layout(self):
        """The layouts against the library's own gather: a random vector either way."""
        cp, pb = self.cp, self.pb
        rng = np.random.default_rng(1 + self.comm.rank)
        u, p = pb.Vv.vector(), pb.Vs.vector()
        u.array[:] = rng.standard_normal(u.array.size)
        p.array[:] = rng.standard_normal(p.array.size)
        ul = np.asarray(pb.Vv.local_values(u))
        pl = np.asarray(pb.Vs.local_values(p))
        mine_u = cp.asnumpy(self._local_u(self.to_layout_u(cp.asarray(np.asarray(u.array)))))
        mine_p = cp.asnumpy(self._local_p(self.to_layout_p(cp.asarray(np.asarray(p.array)))))
        nl = self.cu.nl
        idx = self.cu.place[:, None] * 3 + np.arange(3)[None, :]            # place of (node, component)
        got_u = mine_u[idx]                                     # (nodes, 3)
        ok = bool(np.allclose(got_u, ul.reshape(3, nl).T, rtol=0, atol=1e-12)) and \
            bool(np.allclose(mine_p[self.cq.place], pl, rtol=0, atol=1e-12))
        if not self.comm.allreduce(ok, op=MPI.LAND):
            raise RuntimeError("the column layout does not reproduce the space's local values")

    # -------------------------------------------------------------- layouts
    def to_layout_u(self, a):
        """True dofs of the velocity (cupy, ordered by component) in the layout, flat."""
        return self.cp.ascontiguousarray(a.reshape(3, -1)[:, self.cu.p2t_g].T).ravel()

    def from_layout_u(self, a):
        return self.cp.ascontiguousarray(a.reshape(-1, 3)[self.cu.t2p_g].T).ravel()

    def to_layout_p(self, a):
        return a[self.cq.p2t_g]

    def from_layout_p(self, a):
        return a[self.cq.t2p_g]

    def _local_u(self, xu):
        cp, c = self.cp, self.cu
        a = cp.empty((c.ncl, c.L, 3))
        a[:c.nco] = xu.reshape(c.nco, c.L, 3)
        return c.fill(a).ravel()

    def _local_p(self, xp):
        cp, c = self.cp, self.cq
        a = cp.empty((c.ncl, c.L))
        a[:c.nco] = xp.reshape(c.nco, c.L)
        return c.fill(a).ravel()

    def _owned_u(self, y):
        c = self.cu
        a = self.cp.from_dlpack(y).copy().reshape(c.ncl, c.L, 3)
        return c.reduce(a)[:c.nco].ravel()

    def _owned_p(self, y):
        c = self.cq
        a = self.cp.from_dlpack(y).copy().reshape(c.ncl, c.L)
        return c.reduce(a)[:c.nco].ravel()

    # --------------------------------------------------------------- passes
    def _run(self, kern, gk, dofs, leads, params, sinks):
        """One pass over a group in chunks: ``kern`` on the directions ``leads`` (gathered to
        the elements), every output added into its accumulator at ``idx``."""
        fn, axes = kern
        tabs, geo = gk.mapped()
        ne = int(gk.group.ne)
        args = (dofs,) + tuple(leads) + (tabs, geo, params)
        accs = [s[0] for s in sinks]
        for a in range(0, ne, self.chunk):
            b = min(a + self.chunk, ne)
            res = fn(*_chunk_slices(args, axes, a, b))
            res = res if isinstance(res, tuple) else (res,)
            for k, r in enumerate(res):
                accs[k] = _scatter(accs[k], sinks[k][1][a:b], r)
        return accs

    def _velocity_local(self, ju, yu):
        """``A`` times the local velocity ``ju``, added to the local ``yu`` (JAX arrays)."""
        yu, = self._run(self.k["velocity"], self.gk, self.dofs, (ju[self.eu],), self.params, [(yu, self.eu)])
        for kb, gk, d, e in zip(self.bk, self.bgk, self.bdofs, self.beu):
            yu, = self._run(kb["velocity"], gk, d, (ju[e],), (), [(yu, e)])
        return yu

    def apply(self, x):
        """``K x`` for the unknowns this rank owns (cupy, the velocity then the pressure)."""
        t0 = time.time()
        ju, jp = _jax(self._local_u(x[:self.nu])), _jax(self._local_p(x[self.nu:]))
        yu, yp = self._run(self.k["stokes"], self.gk, self.dofs, (ju[self.eu], jp[self.ep]), self.params,
                           [(jnp.zeros_like(ju), self.eu), (jnp.zeros_like(jp), self.ep)])
        for kb, gk, d, e in zip(self.bk, self.bgk, self.bdofs, self.beu):
            yu, = self._run(kb["velocity"], gk, d, (ju[e],), (), [(yu, e)])
        out = self.cp.concatenate([self._owned_u(yu), self._owned_p(yp)])
        self.times["passes"] += time.time() - t0
        return out

    def apply_A(self, xu):
        t0 = time.time()
        ju = _jax(self._local_u(xu))
        out = self._owned_u(self._velocity_local(ju, jnp.zeros_like(ju)))
        self.times["passes"] += time.time() - t0
        return out

    def apply_B(self, xu):
        t0 = time.time()
        ju = _jax(self._local_u(xu))
        yp, = self._run(self.k["div"], self.gk, self.dofs, (ju[self.eu],), self.params,
                        [(jnp.zeros(self.cq.ncl * self.cq.L), self.ep)])
        out = self._owned_p(yp)
        self.times["passes"] += time.time() - t0
        return out

    def apply_Bt(self, xp):
        t0 = time.time()
        jp = _jax(self._local_p(xp))
        yu, = self._run(self.k["grad"], self.gk, self.dofs, (jp[self.ep],), self.params,
                        [(jnp.zeros(self.cu.ncl * self.cu.L * 3), self.eu)])
        out = self._owned_u(yu)
        self.times["passes"] += time.time() - t0
        return out

    # ---------------------------------------------------------------- setup
    def factorize(self, state):
        """The preconditioner at ``state = (u, p, m3, eps2)``, the point the Newton system is
        linearized at: the blocks of the columns, the collapsed problem and its multigrid,
        the blocks of the columns' Schur complement."""
        t_all = time.time()
        import cupyx

        cp, pb, cu = self.cp, self.pb, self.cu
        u, p, m3, eps2 = state
        loc = pb._loc(u, p, m3)
        self.params = (jnp.asarray(float(eps2)),)
        self.dofs = self.gk.gather(loc)
        self.bdofs = [gk.gather(loc) for gk in self.bgk]
        L, nd, nf, per = cu.L, self.nd, self.nf, self.per_line
        # --- one pass over the element matrices of the velocity block
        t0 = time.time()
        band = cp.zeros(cu.ncl * L * 5 * 9)
        cells = cp.zeros((self.ncell, 3 * nf, 3 * nf))

        def take(Ae, geom, a, b):
            ecol, elev, ecell = (g_[a:b] for g_ in geom)
            vals = Ae[:, self.src_r, self.src_c]
            la, lb = elev[:, self._pa], elev[:, self._pb]
            tgt = ((ecol[:, self._pa].astype(cp.int64) * L + la) * 5 + (lb - la + 2)) * 9 + self._cc[None, :]
            cupyx.scatter_add(band, tgt.ravel(), vals.ravel())
            n = Ae.shape[0]
            C = Ae.reshape(n, 3, nd, 3, nd)[:, :, self.by_line][:, :, :, :, self.by_line]
            C = C.reshape(n, 3, nf, per, 3, nf, per).sum(axis=(3, 6))          # (n, 3, nf, 3, nf)
            cupyx.scatter_add(cells, ecell, C.transpose(0, 2, 1, 4, 3).reshape(n, 3 * nf, 3 * nf))

        for g_, a, b, Ae in pb.kernel.element_matrix_chunks(V, U, loc, params=(eps2,)):
            take(cp.from_dlpack(Ae), self.geom[0], a, b)
        for g_, a, b, Ae in pb.base_kernel.element_matrix_chunks(V, U, loc):
            take(cp.from_dlpack(Ae), self.geom[1 + self.bslot[g_]], a, b)
        band = cu.reduce(band.reshape(cu.ncl, L * 45))[:cu.nco].reshape(cu.nco, L, 5, 3, 3)
        self.columns = ColumnBlocks(band, cp)
        band = None                                           # the factors hold what is needed of it
        if self.sweeps > 1:
            self.lmax = self._largest()
        t_blocks = time.time() - t0
        self.times["blocks"] += t_blocks
        # --- the collapsed problem
        t0 = time.time()
        self._collapsed(cp.asnumpy(cells))
        cells = None
        t_coll = time.time() - t0
        self.times["collapsed"] += t_coll
        # --- the columns' Schur complement on patches of columns of pressure nodes
        t0 = time.time()
        if self.Be is None:
            self._divergence(loc, eps2)
        if self.pmid is None or self.factorizations % self.schur_every == 0:
            self._probe()
        self.factorizations += 1
        self.times["probe"] += time.time() - t0
        cp.get_default_memory_pool().free_all_blocks()
        self.times["setup"] += time.time() - t_all
        if self.verbose:
            print("      preconditioner: %.1f s (columns and collapsed matrix %.1f s, its factorization %.1f s, "
                  "Schur patches %.1f s)" % (time.time() - t_all, t_blocks, t_coll, time.time() - t0), flush=True)
        return self

    def _collapsed(self, cells):
        """The velocity block on the velocities that do not change along a column, as a
        parallel matrix, and its multigrid."""
        import scipy.sparse as sp

        comm, cu, nf = self.comm, self.cu, self.nf
        N = 3 * int(cu.starts[-1])
        rows = (cu.gcol[self.cell_cols][:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(self.ncell, 3 * nf)
        I = np.repeat(rows, 3 * nf, axis=1).ravel()
        J = np.tile(rows, (1, 3 * nf)).ravel()
        lo, hi = 3 * cu.c0, 3 * (cu.c0 + cu.nco)
        mine = (I >= lo) & (I < hi)
        A = sp.csr_matrix((cells.ravel()[mine], (I[mine] - lo, J[mine])), shape=(3 * cu.nco, N))
        # rows of columns that other ranks own go to them
        out = [None] * comm.size
        if not mine.all():
            Io, Jo, Vo = I[~mine], J[~mine], cells.ravel()[~mine]
            owner = np.searchsorted(3 * cu.starts, Io, side="right") - 1
            for o in np.unique(owner):
                k = owner == o
                out[int(o)] = (Io[k], Jo[k], Vo[k])
        for got in comm.alltoall(out):
            if got is not None:
                A = A + sp.csr_matrix((got[2], (got[0] - lo, got[1])), shape=A.shape)
        A.sum_duplicates()
        self._Ac_scipy_nnz = comm.allreduce(A.nnz)
        if self.coarse_solver == "direct":
            # one rank: the collapsed problem (three unknowns per column, a 2D matrix) factorized on the
            # card.  The incomplete factorization on the host takes most of an iteration's time once the
            # whole sheet is on one rank (2.3 s a solve at 10 km against 0.2 s for the element passes)
            if comm.size != 1:
                raise ValueError("the direct solve of the collapsed problem is for one rank")
            import gpu_direct

            if getattr(self, "_coarse_direct", None) is None:
                self._coarse_direct = gpu_direct.CuDSSDirect(comm, spd=True)
            A.sort_indices()
            self._coarse_direct.factorize_csr(A.shape[0], A.indptr, A.indices, A.data)
            self._coarse = None
            return
        Ac = scipy_to_hypre(A)
        if self.coarse_solver == "ilu":
            # the bed holds every column (friction, and the penalty on the normal velocity): the
            # problem is close to local, and an incomplete factorization of each rank's part
            # is a better preconditioner than multigrid, which finds nothing to coarsen
            amg = mfem.HypreILU()
            amg.SetLevelOfFill(self.coarse_fill)
        else:
            amg = mfem.HypreBoomerAMG(Ac)
            amg.SetPrintLevel(0)
            amg.SetSystemsOptions(3, False)
        pcg = mfem.HyprePCG(Ac)
        pcg.SetTol(self.coarse_rtol)
        pcg.SetMaxIter(self.coarse_maxit)
        pcg.SetPrintLevel(0)
        pcg.SetPreconditioner(amg)
        self._coarse = (Ac, amg, pcg)
        self._cb, self._cx = ParVector(comm, 3 * cu.nco), ParVector(comm, 3 * cu.nco)

    def coarse(self, r):
        """The collapsed problem solved for the residual ``r`` (owned velocity, flat), back
        on the columns."""
        t0 = time.time()
        cp, cu = self.cp, self.cu
        if self._coarse is None:
            b = cp.asfortranarray(r.reshape(cu.nco, cu.L, 3).sum(axis=1).reshape(-1, 1))
            z = self._coarse_direct.solve_many(b, refine=1).reshape(cu.nco, 1, 3)
            self.coarse_its.append(1)
            out = cp.broadcast_to(z, (cu.nco, cu.L, 3)).ravel()
            self.times["coarse"] += time.time() - t0
            return out
        self._cb.array[:] = cp.asnumpy(r.reshape(cu.nco, cu.L, 3).sum(axis=1)).ravel()
        self._cx.array[:] = 0.0
        self._coarse[2].Mult(self._cb.hypre, self._cx.hypre)
        its = mfem.intp()
        self._coarse[2].GetNumIterations(its)
        self.coarse_its.append(int(its.value()))
        z = cp.asarray(np.asarray(self._cx.array).reshape(cu.nco, 1, 3))
        out = cp.broadcast_to(z, (cu.nco, cu.L, 3)).ravel()
        self.times["coarse"] += time.time() - t0
        return out

    def columns_solve(self, r):
        t0 = time.time()
        cu = self.cu
        out = self.columns.solve(r.reshape(cu.nco, cu.L, 3)).ravel()
        self.times["columns"] += time.time() - t0
        return out

    def _largest(self):
        """The largest eigenvalue of (columns)^{-1} A, by a few steps of the power method (the
        vector is kept from one linearization point to the next)."""
        cp = self.cp
        steps = 3
        if self._power is None:
            self._power = cp.random.default_rng(7 + self.comm.rank).standard_normal(self.nu)
            steps = 10
        v, lam = self._power, 0.0
        for _ in range(steps):
            v = v / np.sqrt(self.dot(v, v))
            w = self.columns_solve(self.apply_A(v))
            lam = np.sqrt(self.dot(w, w))
            v = w
        self._power = v
        return lam

    def smooth(self, r, z=None):
        """The smoother: ``self.sweeps`` steps of the Chebyshev iteration on the velocity block
        with the solves on the columns in the place of a preconditioner, started from ``z``;
        one sweep is the damped solve on the columns.

        The columns leave a mode that varies along them and is smooth across them, which the
        collapsed problem does not hold either, damped by a factor near one where the cells
        are no wider than the ice is thick: a few Chebyshev steps on the upper part of the
        spectrum (down to ``1 / smooth_range`` of its top) take it down instead."""
        res = r if z is None else r - self.apply_A(z)
        if self.sweeps <= 1:
            d = self.damping * self.columns_solve(res)
            return d if z is None else z + d
        lmax = 1.1 * self.lmax
        lmin = lmax / self.smooth_range
        theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
        sigma = theta / delta
        rho = 1.0 / sigma
        d = self.columns_solve(res) / theta
        x = d if z is None else z + d
        for _ in range(self.sweeps - 1):
            res = res - self.apply_A(d)
            rho1 = 1.0 / (2.0 * sigma - rho)
            d = rho1 * rho * d + (2.0 * rho1 / delta) * self.columns_solve(res)
            rho = rho1
            x = x + d
        return x

    def velocity(self, r):
        """The two-level method of the velocity block: columns, collapsed, columns."""
        z = self.smooth(r)
        z = z + self.coarse(r - self.apply_A(z))
        return self.smooth(r, z)

    def _divergence(self, loc, eps2):
        """The element matrices of ``B`` (it does not depend on the state), in single
        precision: what the preconditioner uses, where a product with them is ten times
        faster than a pass of the kernels and the probing asks for hundreds."""
        cp = self.cp
        self.Be = cp.empty((self.ne, self.gk.tables[P].nd_total, self.gk.tables[U].nd_total), dtype=cp.float32)
        for g_, a, b, Be in self.pb.kernel.element_matrix_chunks(Q, U, loc, params=(eps2,)):
            self.Be[a:b] = cp.from_dlpack(Be)
        self.eu_c, self.ep_c = cp.from_dlpack(self.eu), cp.from_dlpack(self.ep)

    def fast_B(self, xu):
        """``B x`` with the stored element matrices."""
        import cupyx

        cp, cq = self.cp, self.cq
        xe = self._local_u(xu)[self.eu_c].astype(cp.float32)
        ye = cp.matmul(self.Be, xe[:, :, None])[:, :, 0].astype(cp.float64)
        a = cp.zeros(cq.ncl * cq.L)
        cupyx.scatter_add(a, self.ep_c, ye)
        return cq.reduce(a.reshape(cq.ncl, cq.L))[:cq.nco].ravel()

    def fast_Bt(self, xp):
        """``B^T x`` with the stored element matrices."""
        import cupyx

        cp, cu = self.cp, self.cu
        xe = self._local_p(xp)[self.ep_c].astype(cp.float32)
        ye = cp.matmul(self.Be.transpose(0, 2, 1), xe[:, :, None])[:, :, 0].astype(cp.float64)
        a = cp.zeros(cu.ncl * cu.L * 3)
        cupyx.scatter_add(a, self.eu_c, ye)
        return cu.reduce(a.reshape(cu.ncl, cu.L, 3))[:cu.nco].ravel()

    def schur(self, q):
        """``B D^{-1} B^T q``: the Schur complement of the columns."""
        return self.fast_B(self.columns_solve(self.fast_Bt(q)))

    def _patches(self):
        """Where the columns of pressure nodes stand to each other: for every owned column
        the eight around it (the patch), and for every colour of a 5 x 5 colouring of the
        lattice which block of the Schur complement a product with that colour gives."""
        cp, cq = self.cp, self.cq
        K = 1 << 31
        key = cq.ix * K + cq.iy
        order = np.argsort(key)
        skey = key[order]
        ix, iy = cq.ix[:cq.nco], cq.iy[:cq.nco]
        nbr = np.empty((cq.nco, 9), dtype=np.int64)
        for k, (a, b) in enumerate((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)):
            want = (ix + a) * K + (iy + b)
            pos = np.minimum(np.searchsorted(skey, want), max(skey.size - 1, 0))
            nbr[:, k] = np.where(skey[pos] == want, order[pos], cq.ncl) if skey.size else cq.ncl
        self.nbr = cp.asarray(nbr)                           # ncl: no such column on this rank
        cx, cy = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
        dx = (cx.ravel()[:, None] - ix[None, :] + 2) % 5 - 2
        dy = (cy.ravel()[:, None] - iy[None, :] + 2) % 5 - 2
        self.slot = cp.asarray(((dx + 2) * 5 + dy + 2).astype(np.int32))       # (25 colours, nco)
        self.colour = cp.asarray((ix % 5) * 5 + iy % 5)
        self.has_colour = [bool(self.comm.allreduce(int(np.sum((ix % 5) * 5 + iy % 5 == c)))) for c in range(25)]

    def _probe(self):
        """The Schur complement of the columns, ``B D^{-1} B^T``, on patches of 3 x 3 columns
        of pressure nodes, inverted.

        Its blocks between columns (two that share a column of velocity nodes: up to two
        cells apart each way) come from products with vectors that are one on one level of
        every fifth column each way; a patch's matrix is put together from them, inverted,
        and the rows of its centre column kept (restricted additive Schwarz).  The modes the
        pressure mass misses live on a column near the bed and its neighbours, and a patch
        holds them."""
        cp, cq = self.cp, self.cq
        L = cq.L
        if not hasattr(self, "nbr"):
            self._patches()
        blocks = cp.zeros((cq.ncl + 1, 25, L, L), dtype=self.block_dtype)      # the last: no column
        rows = cp.arange(cq.nco)
        for c in range(25):
            if not self.has_colour[c]:
                continue
            sel = self.colour == c
            for k in range(L):
                q = cp.zeros((cq.nco, L))
                q[sel, k] = 1.0
                y = self.schur(q.ravel()).reshape(cq.nco, L)
                blocks[rows, self.slot[c], :, k] = y.astype(blocks.dtype)
                self.probes += 1
        cq.fill(blocks[:cq.ncl])
        # the patches, a few thousand at a time
        P9 = 9 * L
        self.pmid = cp.empty((cq.nco, L, P9), dtype=self.block_dtype)
        offs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        step = max(1, int(self.patch_bytes // (3 * 8 * P9 * P9)))
        eye = cp.eye(L)
        for s in range(0, cq.nco, step):
            e = min(s + step, cq.nco)
            nb = self.nbr[s:e]
            Pm = cp.zeros((e - s, 9, L, 9, L))                  # double: the inverse is taken in it
            for i, (a, b) in enumerate(offs):
                there = nb[:, i] < cq.ncl
                for j, (a2, b2) in enumerate(offs):
                    Pm[:, i, :, j, :] = blocks[nb[:, i], (a2 - a + 2) * 5 + (b2 - b + 2)]
                    # a column that is not there: nothing to or from it, one on its diagonal
                    Pm[:, i, :, j, :] *= (there & (nb[:, j] < cq.ncl))[:, None, None]
                Pm[:, i, :, i, :] += (~there)[:, None, None] * eye[None]
            inv = cp.linalg.inv(Pm.reshape(e - s, P9, P9))
            self.pmid[s:e] = inv[:, 4 * L:5 * L, :]
            del Pm, inv
        del blocks

    def schur_solve(self, r):
        """The Schur complement's part of the preconditioner: on every patch of 3 x 3 columns
        of pressure nodes the columns' Schur complement solved, the centre column kept."""
        t0 = time.time()
        cp, cq = self.cp, self.cq
        a = cp.zeros((cq.ncl + 1, cq.L))
        a[:cq.nco] = r.reshape(cq.nco, cq.L)
        cq.fill(a[:cq.ncl])
        z = (self.pmid @ a[self.nbr].reshape(cq.nco, 9 * cq.L, 1).astype(self.pmid.dtype)).reshape(-1)
        self.times["schur"] += time.time() - t0
        return z.astype(cp.float64)

    def dot(self, a, b):
        return self.comm.allreduce(float(a @ b))

    def precondition(self, r):
        zu = self.velocity(r[:self.nu])
        zp = -self.schur_solve(r[self.nu:] - self.fast_B(zu))
        return self.cp.concatenate([zu, zp])

    # ---------------------------------------------------------------- solve
    def fgmres(self, b, x0=None):
        """Flexible GMRES with restarts, the orthogonalization twice (classical
        Gram-Schmidt, one reduction a pass); returns the solution and the iterations."""
        cp, comm, m = self.cp, self.comm, self.restart
        n = b.size
        x = cp.zeros(n) if x0 is None else x0.copy()
        bnorm = np.sqrt(self.dot(b, b))
        if bnorm == 0.0:
            return x, 0, 0.0
        Vb = cp.empty((m + 1, n))
        Zb = cp.empty((m, n))
        total = 0
        rel = 1.0
        late = False
        while total < self.maxit:
            r = b - self.apply(x) if (x0 is not None or total) else b.copy()
            beta = np.sqrt(self.dot(r, r))
            rel = beta / bnorm
            if rel <= self.rtol:
                break
            Vb[0] = r / beta
            H = np.zeros((m + 1, m))
            gvec = np.zeros(m + 1)
            gvec[0] = beta
            k = 0
            for k in range(m):
                z = self.precondition(Vb[k])
                Zb[k] = z
                w = self.apply(z)
                for _ in range(2):
                    h = comm.allreduce(cp.asnumpy(Vb[:k + 1] @ w))
                    w -= cp.asarray(h) @ Vb[:k + 1]
                    H[:k + 1, k] += h
                H[k + 1, k] = np.sqrt(self.dot(w, w))
                Vb[k + 1] = w / H[k + 1, k]
                total += 1
                y, *_ = np.linalg.lstsq(H[:k + 2, :k + 1], gvec[:k + 2], rcond=None)
                rel = float(np.linalg.norm(H[:k + 2, :k + 1] @ y - gvec[:k + 2])) / bnorm
                if self.verbose and (total % 10 == 0):
                    print("      fgmres %3d: %.2e" % (total, rel), flush=True)
                # out of time (rank 0's clock): what there is goes back, an inexact Newton step
                late = self.deadline is not None and comm.bcast(time.time() > self.deadline if comm.rank == 0 else None)
                if rel <= self.rtol or total >= self.maxit or late:
                    break
            x += cp.asarray(y) @ Zb[:k + 1]
            if rel <= self.rtol or late:
                break
        self.timed_out = bool(late)
        return x, total, rel

    def solve_incremental(self, x, b):
        self.rtol = self.rtol_tight
        return self.solve(x, b)

    def solve(self, x, b):
        """The system at the state of the last ``factorize``; ``x`` and ``b`` are the
        problem's vectors (true dofs)."""
        t0 = time.time()
        before, nc = dict(self.times), len(self.coarse_its)
        cp, nv = self.cp, self.pb.nv
        ba = cp.asarray(np.asarray(b.array))
        rhs = cp.concatenate([self.to_layout_u(ba[:nv]), self.to_layout_p(ba[nv:])])
        sol, its, rel = self.fgmres(rhs)
        x.array[:nv] = cp.asnumpy(self.from_layout_u(sol[:self.nu]))
        x.array[nv:] = cp.asnumpy(self.from_layout_p(sol[self.nu:]))
        self.iterations.append(int(its))
        if len(self.iterations) == 1:
            _return_kernels(self.comm, self._kernel_folders)
        if rel > self.rtol and self.comm.rank == 0:
            print("    FGMRES stopped at %d iterations, residual %.2e of the first%s"
                  % (its, rel, " (out of time)" if self.timed_out else ""), flush=True)
        self.times["solve"] += time.time() - t0
        if self.verbose:
            d = {k: self.times[k] - before[k] for k in ("passes", "coarse", "columns", "schur")}
            print("      fgmres: %d iterations to %.1e in %.1f s (passes %.1f s, collapsed solves %.1f s with %.1f iterations "
                  "each, columns %.1f s, patches %.1f s)"
                  % (its, self.rtol, time.time() - t0, d["passes"], d["coarse"],
                     float(np.mean(self.coarse_its[nc:])) if len(self.coarse_its) > nc else 0.0, d["columns"], d["schur"]), flush=True)
        return x

    def describe(self):
        it = np.array(self.iterations) if self.iterations else np.zeros(1)
        return {"solver": "fgmres, matrix-free, columns and collapsed", "rtol": self.rtol_tight,
                "rtol_loose": self.rtol_loose if self.forcing else None, "solves": len(self.iterations),
                "iterations_mean": float(it.mean()), "iterations_max": int(it.max()),
                "iterations": [int(i) for i in self.iterations],
                "coarse_iterations_mean": float(np.mean(self.coarse_its)) if self.coarse_its else 0.0,
                "probes": int(self.probes), "sweeps": self.sweeps, "largest_eigenvalue": self.lmax,
                "columns": int(self.cu.starts[-1]), "levels": int(self.cu.L),
                "collapsed_unknowns": 3 * int(self.cu.starts[-1]), "times": dict(self.times)}

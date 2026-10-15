# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""A box of equal bricks for the spectral-element wave propagator: its mesh, the names
of its nodes, its absorbing faces, its points, and the propagator on it.

The box is ``origin + [0, Lx] x [0, Ly] x [0, Lz]``, ``x`` along the section, ``y`` across
it and ``z`` the depth.  Its faces are named ``x0, x1, y0, y1, z0, z1`` (the low and the
high end of each axis): ``z0``, the top, is the free surface, where the pressure is zero,
and the five others let the waves out to first order, unless told otherwise.

The propagator is the seismic application's (:mod:`sem`), which asks three things of a
mesh that MFEM holds as straight hexahedra: an integer key for every node of the wave
field, the same whichever element or rank sees it; the nodes where the field is zero; and
the geometry.  For the globe the first two come from the cubed sphere's lattice and the
third from its map.  A box of ``nx x ny x nz`` equal bricks is the simplest such mesh,
and everything here follows from that by arithmetic:

* the nodes of order ``p`` are the points ``(I, J, K)`` of a lattice of ``p nx + 1`` by
  ``p ny + 1`` by ``p nz + 1``, and a node's key is its place there, ``(I my + J) mz + K``.
  A rank's unknowns numbered by their keys are then its block of the lattice as a 3D
  array (``z`` fastest), to which the stiffness is applied without a table of indices;
* every element has the same geometric factors, three numbers a node, in closed form;
* the absorbing boundary's diagonal mass at a node is the product of two assembled
  Gauss-Lobatto weights times an admittance: no integral over faces, no exchange;
* the stability limit of the time step has a bound in closed form (:meth:`Box.dt`), so
  no power iteration estimates it;
* the element that holds a point is ``floor((x - origin) / h)``: no search.

Two ways to the mesh (:meth:`Box.mesh`).  A small box is MFEM's Cartesian mesh cut into
the ranks' blocks.  That builds the whole mesh on every rank, which many ranks cannot
afford (1.9 microseconds and 1.0 kB an element of the WHOLE mesh, on each of them).  A
large box is a coarse box of a few elements a rank, cut into the same blocks, whose
elements every rank then cuts for itself (``ParMesh.MakeRefined``), as the geothermal
application does for its cube: 1.0 microsecond and 0.5 kB an element of the rank's own,
on one rank as on many.
"""

import math
import os
import sys

import numpy as np
from mpi4py import MPI

import hippymfem as hm

# the seismic application's folder, after everything else on the path: only its propagator
# is wanted from there
_SEISMIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seismic")
if _SEISMIC not in sys.path:
    sys.path.append(_SEISMIC)

import sem                                            # noqa: E402

import jax.numpy as jnp                               # noqa: E402

FACES = ("x0", "x1", "y0", "y1", "z0", "z1")
#: the free surface, and the faces that absorb, unless told otherwise
FREE = ("z0",)
ABSORB = ("x0", "x1", "y0", "y1", "z1")
#: elements of the mesh every rank builds whole: up to ``few`` no rank cuts anything (half a
#: second and a quarter of a GB a rank), and above ``most`` :meth:`Box.mesh` refuses (8 s and
#: 4 GB)
SERIAL = {"few": 1 << 18, "most": 1 << 22}


def line_bound(order):
    r""":math:`h^2 \lambda_{\max}(W^{-1} K)` of a line of spectral elements of side ``h`` and
    degree ``order``, ``K`` its stiffness and ``W`` its diagonal mass: 4, 24, 74.311 and
    183.35 for degrees 1 to 4.  It is one element's, which a line of any length attains
    (the element's vector, mirrored from element to element, is the line's)."""
    _, w, D = sem.gll(int(order))
    s = 1.0 / np.sqrt(w)
    return float(np.linalg.eigvalsh(s[:, None] * ((D.T * w) @ D) * s[None, :])[-1])


def _faces(faces, what):
    faces = tuple(faces or ())
    if any(f not in FACES for f in faces):
        raise ValueError("%s: faces are named %s, not %s" % (what, ", ".join(FACES), faces))
    return faces


class Box:
    """``n = (nx, ny, nz)`` equal bricks filling ``origin + [0, size]``.

    Attributes
    ----------
    n : ndarray of int64, ``(3,)``
        Elements along each axis.
    size, origin, h : ndarray, ``(3,)``
        The box's lengths, its low corner (``origin[2]`` is the depth of the top), and
        the sides of an element.
    """

    def __init__(self, n, size, origin=(0.0, 0.0, 0.0)):
        self.n = np.asarray(n, dtype=np.int64).reshape(3)
        self.size = np.asarray(size, dtype=np.float64).reshape(3)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        if (self.n < 1).any() or not (self.size > 0.0).all():
            raise ValueError("a box of %s elements and sides %s" % (self.n.tolist(), self.size.tolist()))
        self.h = self.size / self.n

    # ------------------------------------------------------------------ the time step
    def dt(self, order, c_max, cfl=0.8):
        r"""A time step that is stable for every wave speed up to ``c_max``: ``cfl`` times

        .. math:: \frac{2}{c_{\max}\sqrt{\lambda_p\,(h_x^{-2} + h_y^{-2} + h_z^{-2})}} ,

        :math:`\lambda_p` of :func:`line_bound`.  The leapfrog scheme is stable for
        :math:`\Delta t \le 2 / \sqrt{\lambda_{\max}(M^{-1}K)}`, with or without absorbing
        faces, and with the speed in the mass alone :math:`M^{-1}K` is the box's
        :math:`W^{-1}K` times :math:`c^2` at the nodes, whose largest eigenvalue is at most
        this.  A bound, so a little below the limit that the propagator's power iteration
        finds for a given model (3 to 5 % on boxes with speeds from 1500 to 4000 m/s), but
        the same on every partition and at no cost."""
        lam = line_bound(order) * float(np.sum(self.h ** -2.0))
        return float(cfl) * 2.0 / (float(c_max) * math.sqrt(lam))

    # ------------------------------------------------------------------ the mesh
    def grid(self, ranks, columns=False):
        """``ranks`` as a grid of blocks ``(px, py, pz)`` that divide the elements, a rank
        the block ``I + px (J + py K)``: the grid whose blocks have the least surface,
        which is what a rank exchanges at a step.  Of several such, the one with the most
        blocks along ``x``, then ``y`` (a block's unknowns are an array with ``z`` fastest,
        and its faces across ``x`` are whole pieces of it).  ``columns`` keeps ``pz = 1``:
        every rank then holds a piece of the surface, with its sources and receivers."""
        ranks = int(ranks)
        nx, ny, nz = (int(v) for v in self.n)
        best = None
        for px in range(1, ranks + 1):
            if ranks % px or nx % px:
                continue
            for py in range(1, ranks // px + 1):
                pz = ranks // (px * py)
                if (ranks // px) % py or ny % py or nz % pz or (columns and pz > 1):
                    continue
                b = (nx // px, ny // py, nz // pz)
                key = (b[0] * b[1] + b[1] * b[2] + b[0] * b[2], -px, -py)
                if best is None or key < best[0]:
                    best = (key, (px, py, pz))
        if best is None:
            raise ValueError("%d ranks make no grid of blocks%s that divides %d x %d x %d elements"
                             % (ranks, " in columns" if columns else "", nx, ny, nz))
        return best[1]

    def cuts(self, grid, few=None):
        """The cuts :meth:`mesh` makes when none are named: none for a box of ``few``
        elements or fewer (``SERIAL["few"]``), which every rank can build whole;
        otherwise factors of 2 to 8 (what one cut can be) whose product is the largest
        that divides a block's three sides.  The coarse box then has the fewest elements
        a rank it can have.  The smaller factors first: 120^3 elements take 1.7 s cut by
        3, 5, 8 and 2.0 s cut by 8, 5, 3 (3.4 s whole)."""
        if int(np.prod(self.n)) <= (SERIAL["few"] if few is None else few):
            return ()
        f = math.gcd(*(int(n) // int(g) for n, g in zip(self.n, grid)))
        out = []
        for c in range(8, 1, -1):
            while f % c == 0:
                out.append(c)
                f //= c
        return tuple(out[::-1])

    def mesh(self, comm, grid=None, cuts=None):
        """The box as a ``ParMesh`` of straight hexahedra on ``comm``, rank ``I + px (J +
        py K)`` holding block ``(I, J, K)`` of ``grid`` (:meth:`grid` of the ranks by
        default).

        ``cuts=()``: MFEM's Cartesian mesh of all the elements, built and cut into the
        blocks on every rank.  ``cuts=(f1, f2, ...)``, factors of 2 to 8: a coarse box of
        ``n / (f1 f2 ...)`` elements is built and cut into the blocks so, and every rank
        then cuts its own elements ``f1`` a side, then ``f2``, ...  The elements and the
        vertices are the same to round-off; their order on a rank is not (the propagator
        does not mind: its unknowns go by their keys).  ``cuts=None`` chooses
        (:meth:`cuts`), and on several ranks refuses a coarse box that is still large
        (choose blocks whose sides share a factor).
        """
        import mfem.par as mfem

        grid = self.grid(comm.size) if grid is None else tuple(int(g) for g in grid)
        n = [int(v) for v in self.n]
        if len(grid) != 3 or int(np.prod(grid)) != comm.size or any(v % g for v, g in zip(n, grid)):
            raise ValueError("a grid of %s blocks for %d ranks and %s elements" % (grid, comm.size, n))
        sides = [v // g for v, g in zip(n, grid)]                 # of a block, in elements
        chosen = cuts is None
        cuts = self.cuts(grid) if chosen else tuple(int(f) for f in cuts)
        f = int(np.prod(cuts)) if cuts else 1
        if any(c < 2 or c > 8 for c in cuts) or any(v % f for v in sides):
            raise ValueError("cuts %s: each 2 to 8, their product a divisor of a block's sides %s"
                             % (cuts, sides))
        n0 = [v // f for v in n]
        if chosen and comm.size > 1 and int(np.prod(n0)) > SERIAL["most"]:
            raise ValueError("every rank would build a coarse box of %d x %d x %d elements: choose blocks "
                             "whose sides share a factor made of 2 to 8 (they are %s)" % (*n0, sides))
        Lx, Ly, Lz = (float(v) for v in self.size)
        serial = mfem.Mesh.MakeCartesian3D(n0[0], n0[1], n0[2], mfem.Element.HEXAHEDRON, Lx, Ly, Lz, False)
        if self.origin.any():
            v = mfem.Vector()
            serial.GetVertices(v)                                 # all x, then all y, then all z
            X = np.array(v.GetDataArray()).reshape(3, -1) + self.origin[:, None]
            serial.SetVertices(mfem.Vector(np.ascontiguousarray(X.reshape(-1))))
        if comm.size > 1:
            nxyz = mfem.intArray(list(grid))
            pmesh = mfem.ParMesh(comm, serial, serial.CartesianPartitioning(nxyz.GetData()))
        else:
            pmesh = mfem.ParMesh(comm, serial)
        del serial
        for c in cuts:
            pmesh = mfem.ParMesh.MakeRefined(pmesh, c, mfem.BasisType.ClosedUniform)
        return pmesh

    # ------------------------------------------------------------------ the nodes' names
    def last(self, order):
        """The last index of the nodes' lattice along each axis: ``order`` times the
        elements."""
        return int(order) * self.n

    def node_keys(self, order, free=FREE):
        """What :meth:`sem.Collocation.from_keys` asks for: ``f(qv, W) -> (keys, boundary)``.

        A node's key is ``(I my + J) mz + K`` with ``(I, J, K)`` its place on the lattice
        of ``order`` times the elements and ``my``, ``mz`` the lattice's points along ``y``
        and ``z``.  The place is the trilinear interpolant of those of the element's
        vertices, which is exact (an element is a unit cube of the vertices' lattice)
        whatever way the element lies.  ``boundary`` marks the nodes of the faces
        ``free``, where the field is zero.
        """
        p, free = int(order), _faces(free, "free")
        N, h, origin = self.last(order), self.h, self.origin
        my, mz = int(N[1]) + 1, int(N[2]) + 1

        def keys(qv, W, block=1 << 20):
            ne, nq = qv.shape[0], W.shape[0]
            key = np.empty((ne, nq), dtype=np.int64)
            boundary = np.zeros((ne, nq), dtype=bool)
            Wt = np.ascontiguousarray(p * W.T)                               # (8, nq)
            for a in range(0, ne, block):
                b = min(a + block, ne)
                s = (qv[a:b] - origin) / h
                iv = np.rint(s)                                              # the vertices' lattice places
                if np.abs(s - iv).max() > 1e-6 or iv.min() < 0 or (iv > self.n).any():
                    raise ValueError("the mesh's vertices are not this box's lattice")
                t = [np.rint(iv[:, :, d] @ Wt).astype(np.int64) for d in range(3)]
                key[a:b] = (t[0] * my + t[1]) * mz + t[2]
                for f in free:
                    d = "xyz".index(f[0])
                    boundary[a:b] |= t[d] == (0 if f[1] == "0" else N[d])
            return key, boundary

        return keys

    def decode(self, keys, order):
        """``(I, J, K)`` of node keys, ``(n, 3)``."""
        N = self.last(order)
        my, mz = int(N[1]) + 1, int(N[2]) + 1
        keys = np.asarray(keys, dtype=np.int64)
        return np.stack([keys // (my * mz), (keys // mz) % my, keys % mz], axis=1)

    def _decoded(self, keys, order, chunk=1 << 22):
        """``(start, (I, J, K))`` of the keys a piece at a time: a rank's keys decoded at
        once are 24 bytes an unknown of the host's memory."""
        for a in range(0, keys.size, chunk):
            yield a, self.decode(keys[a:a + chunk], order)

    def line(self, order):
        """Along each axis, for every index of the lattice: the node's position, and its
        assembled Gauss-Lobatto weight (the element's side times the weight on ``[0, 1]``,
        twice that where two elements share the node)."""
        p = int(order)
        nodes, w1, _ = sem.gll(p)
        pos, wts = [], []
        for d in range(3):
            i = np.arange(p * int(self.n[d]) + 1)
            e = np.minimum(i // p, int(self.n[d]) - 1)
            r = i - p * e
            pos.append(self.origin[d] + self.h[d] * (e + nodes[r]))
            w = self.h[d] * w1[r]
            w[p:-1:p] *= 2.0
            wts.append(w)
        return pos, wts

    def node_positions(self, keys, order):
        """Where the nodes of these keys are, ``(n, 3)``."""
        ijk = self.decode(keys, order)
        pos, _ = self.line(order)
        return np.stack([pos[d][ijk[:, d]] for d in range(3)], axis=1)

    def block(self, col, order):
        """The elements along ``x``, ``y``, ``z`` of the block of the lattice that this rank's
        unknowns are (what ``SpectralWave(lattice=...)`` takes), or None where they are
        not a whole block."""
        if col.nloc == 0:
            return None
        lo, hi = None, None
        for _, ijk in self._decoded(col.keys, order):
            a, b = ijk.min(axis=0), ijk.max(axis=0)
            lo, hi = (a, b) if lo is None else (np.minimum(lo, a), np.maximum(hi, b))
        N = hi - lo + 1
        if int(np.prod(N)) != col.nloc or np.any((N - 1) % int(order)):
            return None
        return tuple(int(v) for v in (N - 1) // int(order))

    def along_axes(self, col, order, chunk=1 << 20):
        """Whether every element of ``col`` lies along the axes with its first node the lowest:
        the nodes after the first along its three reference axes are one step of the lattice
        in ``x``, in ``y`` and in ``z``.  One brick's geometric factors for every element
        (``brick=``) and :meth:`locate` take that for granted; MFEM's Cartesian meshes and
        their cuts are so, a mesh built elsewhere may not be."""
        n = int(order) + 1
        for a in range(0, col.ne, chunk):
            near = col.keys[col.idx[a:a + chunk][:, [0, 1, n, n * n]]]
            step = self.decode(near[:, 1:].reshape(-1), order).reshape(-1, 3, 3) \
                - self.decode(near[:, 0], order)[:, None, :]
            if (step != np.eye(3, dtype=np.int64)).any():
                return False
        return True

    # ------------------------------------------------------------------ the absorbing faces
    def absorbing(self, col, order, admittance=1.0, faces=ABSORB):
        """``(dofs, C)`` for ``SpectralWave(absorbing=...)``: the local unknowns of ``col``
        on ``faces`` and the boundary mass there,

            ``C_i = admittance(x_i) * sum over the faces through i of  W_s(i) W_t(i)``,

        ``s, t`` the two axes of a face and ``W`` the assembled weights of :meth:`line`:
        the integral of the admittance over the faces against the node's basis function,
        by the nodes' own quadrature.  It is computed from the node's place on the
        lattice alone, so every rank that holds a node has the same number for it, the
        whole one.

        ``admittance`` is ``1 / (rho c)`` of a REFERENCE medium, a number or a function of
        positions ``(n, 3)``: not of the model that is inverted for, so that the
        derivatives of the scheme have no term from the boundary.  The price is a
        reflection of ``(c - c_ref) / (c + c_ref)`` at normal incidence where the model
        has moved away from the reference at a face.
        """
        faces = _faces(faces, "absorbing")
        N = self.last(order)
        pos, wts = self.line(order)
        dofs, C, X = [np.zeros(0, dtype=np.int64)], [np.zeros(0)], [np.zeros((0, 3))]
        for a, ijk in self._decoded(col.keys, order):
            Ca = np.zeros(ijk.shape[0])
            for f in faces:
                d = "xyz".index(f[0])
                s, t = (k for k in range(3) if k != d)
                on = ijk[:, d] == (0 if f[1] == "0" else N[d])
                Ca += np.where(on, wts[s][ijk[:, s]] * wts[t][ijk[:, t]], 0.0)
            at = np.nonzero(Ca > 0.0)[0]
            dofs.append(a + at)
            C.append(Ca[at])
            X.append(np.stack([pos[d][ijk[at, d]] for d in range(3)], axis=1))
        dofs, C = np.concatenate(dofs), np.concatenate(C)
        if not callable(admittance):
            C *= float(admittance)
        elif C.size:
            C *= np.asarray(admittance(np.concatenate(X)), dtype=np.float64).reshape(-1)
        return dofs, C

    # ------------------------------------------------------------------ points
    def locate(self, col, points, comm, order):
        """:func:`sem.locate` for the box, without a search: ``(rows, elems, xi)`` of the
        points this rank holds, ``xi`` in ``[0, 1]^3``.

        A point belongs to the element ``floor((x - origin) / h)``, the last one for a
        point of the far faces, and one rank holds that element: no point is taken twice
        or left out, also on a cut between blocks.  The local elements are found by the
        keys of their first nodes; the three nodes next to it along the element's axes
        say that the element lies along ``x``, ``y``, ``z``, which ``xi`` takes for granted.
        """
        p, n = int(order), int(order) + 1
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        s = (points - self.origin) / self.h
        if (s < -1e-9).any() or (s > self.n + 1e-9).any():
            raise ValueError("point(s) %s lie outside the box"
                             % np.nonzero(((s < -1e-9) | (s > self.n + 1e-9)).any(axis=1))[0].tolist())
        e3 = np.clip(np.floor(s).astype(np.int64), 0, self.n - 1)
        xi = np.clip(s - e3, 0.0, 1.0)
        rows = elems = np.zeros(0, dtype=np.int64)
        if col.ne:
            # the local elements by their place in the box: a table over the block they span
            first = self.decode(col.keys[col.idx[:, 0]], order)
            if (first % p).any():
                raise RuntimeError("an element's first node is not a vertex of the box's lattice")
            first //= p
            lo, span = first.min(axis=0), first.max(axis=0) - first.min(axis=0) + 1
            table = np.full(int(np.prod(span)), -1, dtype=np.int64)
            table[((first[:, 0] - lo[0]) * span[1] + first[:, 1] - lo[1]) * span[2] + first[:, 2] - lo[2]] = \
                np.arange(col.ne)
            inside = ((e3 >= lo) & (e3 < lo + span)).all(axis=1)
            r = e3[inside] - lo
            hit = table[(r[:, 0] * span[1] + r[:, 1]) * span[2] + r[:, 2]]
            rows, elems = np.nonzero(inside)[0][hit >= 0], hit[hit >= 0]
            # the nodes after the first along the element's three axes: one step in x, y, z
            near = col.keys[col.idx[elems][:, [0, 1, n, n * n]]]
            step = self.decode(near[:, 1:].reshape(-1), order).reshape(-1, 3, 3) \
                - self.decode(near[:, 0], order)[:, None, :]
            if (step != np.eye(3, dtype=np.int64)).any():
                raise RuntimeError("an element does not lie along the axes as the box expects")
        found = comm.allreduce(int(rows.size))
        if found != points.shape[0]:
            raise RuntimeError("%d of %d points found" % (found, points.shape[0]))
        return rows, elems, xi[rows]

    def value_operator(self, col, points, comm, order):
        """:func:`sem.value_operator` with :meth:`locate`: point values at ``points``, and
        the load of unit point sources there."""
        rows, elems, xi = self.locate(col, points, comm, order)
        w = np.zeros((rows.size, col.n ** 3))
        for i in range(rows.size):
            la, _ = sem.lagrange(col.nodes, xi[i, 0])
            lb, _ = sem.lagrange(col.nodes, xi[i, 1])
            lc, _ = sem.lagrange(col.nodes, xi[i, 2])
            w[i] = np.einsum("c,b,a->cba", lc, lb, la).reshape(-1)
        return sem.PointOperator(w, col.idx[elems], rows, np.asarray(points).reshape(-1, 3).shape[0], comm)


def engine(comm, box, order, density, aux=(), dtype=jnp.float64, absorb=ABSORB, history="device",
           lattice=True, grid=None, cuts=None, *, free=FREE, admittance=None, space=None,
           dt=0.0, nsteps=None, **wave):
    """The propagator on the box: ``(pmesh, Vm, col, eng)``, the mesh, the parameter's
    trilinear space on it, the wave field's tables and the :class:`sem.SpectralWave`,
    ready for ``set_parameter``.  On ``MPI.COMM_SELF`` a rank holds the whole box; on a
    communicator of several ranks they hold the blocks of ``grid``.

    Parameters
    ----------
    density, aux, dtype
        As :class:`sem.SpectralWave` takes them.  This application's density is

        .. code-block:: python

            def density(u, a, m, p, x, c0):
                return a.val * p.val / (c0 + m.val) ** 2 + inner(u.grad, p.grad)

        with ``aux`` a JAX function of a block's node positions that returns ``(c0,)``.
    absorb : sequence of faces
        The faces that let the waves out; none (``()``) leaves them all natural.
    history : "device" or "host"
        Where a forward sweep keeps the field at the absorbing nodes.
    lattice : bool
        The stiffness without a table of indices, the unknowns numbered by their keys
        (it needs a stiffness coefficient that is one number, as the density above
        has).  False: the kernel with a table, on unknowns numbered as the elements
        meet them, with one brick's geometric factors.
    grid, cuts
        As :meth:`Box.mesh` takes them.
    free : sequence of faces
        Where the field is zero.
    admittance : float or callable, optional
        The reference medium's ``1 / (rho c)`` at the absorbing nodes
        (:meth:`Box.absorbing`).  By default ``1 / c0``, ``c0`` the first field of
        ``aux``, which then has to be a function (it is called once more, with the
        absorbing nodes' positions as one block, ``(1, n, 3)``).
    space : FunctionSpace, optional
        A trilinear space on a mesh of the box built elsewhere, in the place of
        :meth:`Box.mesh` on ``comm``.
    dt, nsteps
        The time step and the number of steps, which may be set later (``eng.dt``,
        ``eng.nsteps``) but before a sweep: :meth:`Box.dt` gives a step.  (``dt=None``
        has the propagator estimate the limit by a power iteration at the first
        parameter.)
    wave
        Anything else :class:`sem.SpectralWave` takes.
    """
    free, absorb = _faces(free, "free"), _faces(absorb, "absorb")
    if set(free) & set(absorb):
        raise ValueError("a face is free or absorbing, not both: %s" % sorted(set(free) & set(absorb)))
    Vm = hm.FunctionSpace.H1(box.mesh(comm, grid, cuts), 1) if space is None else space
    pmesh, comm = Vm.mesh, Vm.comm
    col = sem.Collocation.from_keys(Vm, order, sem.vertex_coordinates(pmesh), lambda q: q,
                                    box.node_keys(order, free), brick=box.h,
                                    numbering="keys" if lattice else "touch")
    # one answer for every rank: brick= is taken on the caller's word by the kernel with a table
    # (a brick of unequal sides turned about an axis would get another brick's factors)
    if not comm.allreduce(box.along_axes(col, order), op=MPI.LAND):
        raise ValueError("an element of the mesh does not lie along the axes of the box, its first node "
                         "the lowest")
    block = box.block(col, order) if lattice else None
    # one answer for every rank
    if lattice and not comm.allreduce(block is not None, op=MPI.LAND):
        raise ValueError("lattice=True: a rank's unknowns are not a whole block of the box's lattice")
    ab = None
    if absorb:
        if admittance is None:
            if not callable(aux):
                raise ValueError("the reference admittance of the absorbing faces: give admittance=, or aux "
                                 "as a function of positions whose first field is the reference speed")

            def admittance(X):
                return 1.0 / np.asarray(aux(jnp.asarray(X)[None])[0], dtype=np.float64)[0]

        ab = box.absorbing(col, order, admittance, absorb)
    eng = sem.SpectralWave(None, Vm, density, aux=aux, col=col, dt=dt, nsteps=nsteps, dtype=dtype,
                           absorbing=ab, history=history, lattice=block, **wave)
    return pmesh, Vm, col, eng

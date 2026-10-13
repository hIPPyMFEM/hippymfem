# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
r"""An explicit spectral-element wave propagator with its adjoint and its
parameter sensitivities, on the device.

The wave equation is given as hIPPyMFEM gives every PDE, by its weak residual at a
quadrature point,

.. code-block:: python

    def density(u, a, m, p, x, rho, c0):
        return a.val * p.val / (rho * (c0 + m.val) ** 2) + inner(u.grad, p.grad) / rho

with ``u`` the wave field, ``a`` its second time derivative, ``m`` the parameter and
``p`` the test function.  Two things distinguish a wave from the problems the rest of
the library solves, and both are choices of discretization rather than of interface:

* The quadrature points are the element's own Gauss-Lobatto-Legendre nodes (the
  spectral element method).  The basis is then the identity at the points, so the
  mass matrix :math:`\partial^2 r/\partial a\,\partial p` is *diagonal*, and a time
  step needs no linear solve.  Its entries, and their derivatives with respect to the
  parameter, are derivatives of the density at each point, taken by JAX.
* The stiffness action :math:`\partial^2 r/\partial u\,\partial p\;u` is applied
  matrix-free in tensor-product form (three one-dimensional derivative matrices per
  element), which is what an explicit scheme spends its time in.

Time is advanced by the leapfrog scheme

.. math:: M(m)\,(u_{n+1} - 2u_n + u_{n-1}) + \Delta t^2 K u_n = \Delta t^2 f_n ,

which is second order, conserves a discrete energy, and is **exactly reversible**:
the same recurrence run from :math:`(u_N, u_{N-1})` returns the whole history.  The
adjoint and the Hessian actions use that instead of storing or checkpointing the
forward field: a backward sweep carries the adjoint and the reconstructed forward
field together and correlates them as it goes.  All derivatives are those of this
discrete scheme, so the gradient is the exact gradient of the discrete misfit.

In parallel each rank holds the dofs of its own elements; a step computes the
element contributions on the device, sums them over the ranks that share a dof
(in rank order, so every copy of a shared dof holds the same bits), and updates.
"""

import sys
import time
import traceback

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from hippymfem.fem.elementbatch import _basis_tables, _element_dofs
from hippymfem.fem import kernel as kernel_mod
from hippymfem.fem.kernel import Field

import jax
import jax.numpy as jnp


def abort_all():
    """For the ``except`` block of a driver: with several ranks, print the exception
    and stop them all.  An exception on one rank would otherwise leave the others
    waiting in their next exchange until the job's time limit."""
    if MPI.COMM_WORLD.size > 1:
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)


# ------------------------------------------------------------------ one dimension
def gll(p):
    """Gauss-Lobatto-Legendre nodes and weights on ``[0, 1]`` and the matrix of the
    Lagrange basis' derivatives at the nodes, ``D[a, b] = l_b'(x_a)``."""
    from numpy.polynomial import legendre as L

    c = np.zeros(p + 1)
    c[p] = 1.0
    if p == 1:
        x = np.array([-1.0, 1.0])
    else:
        x = np.concatenate([[-1.0], np.sort(np.real(L.legroots(L.legder(c)))), [1.0]])
    Pp = L.legval(x, c)
    w = 2.0 / (p * (p + 1) * Pp ** 2)
    n = p + 1
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (Pp[i] / Pp[j]) / (x[i] - x[j])
    D[0, 0] = -0.25 * p * (p + 1)
    D[p, p] = 0.25 * p * (p + 1)
    return 0.5 * (x + 1.0), 0.5 * w, 2.0 * D


def lagrange(nodes, x):
    """Values and derivatives at ``x`` of the Lagrange basis on ``nodes``."""
    n = nodes.size
    val, der = np.ones(n), np.zeros(n)
    for j in range(n):
        den = 1.0
        for k in range(n):
            if k != j:
                den *= nodes[j] - nodes[k]
        s = 0.0
        for i in range(n):
            if i == j:
                continue
            t = 1.0
            for k in range(n):
                if k != j and k != i:
                    t *= x - nodes[k]
            s += t
        v = 1.0
        for k in range(n):
            if k != j:
                v *= x - nodes[k]
        val[j], der[j] = v / den, s / den
    return val, der


def _inverse3(J):
    """Inverses and determinants of a stack of 3 x 3 matrices, in closed form (a
    LAPACK call per matrix is several times slower at this size)."""
    a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]
    d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]
    g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]
    out = np.empty_like(J)
    out[..., 0, 0], out[..., 0, 1], out[..., 0, 2] = e * i - f * h, c * h - b * i, b * f - c * e
    out[..., 1, 0], out[..., 1, 1], out[..., 1, 2] = f * g - d * i, a * i - c * g, c * d - a * f
    out[..., 2, 0], out[..., 2, 1], out[..., 2, 2] = d * h - e * g, b * g - a * h, a * e - b * d
    det = a * out[..., 0, 0] + b * out[..., 1, 0] + c * out[..., 2, 0]
    out /= det[..., None, None]
    return out, det


# --------------------------------------------------------------- the space's tables
class Collocation:
    """What a hexahedral H1 space of degree ``p`` looks like as a spectral element
    discretization: the element dofs in tensor order, and the geometry at the nodes.

    The geometry comes from one of two places.  By default it is the mesh's own: the
    node positions and the Jacobians of MFEM's nodal geometry, evaluated on the host
    (``X``, ``G``, ``wdet`` below).  With ``mapped=(Vq, q, fn)`` the mesh is the
    image of a straight one under an analytic map: ``Vq`` is the trilinear space on
    the mesh, ``q`` the positions of its local dofs (the vertices) *before* the map,
    ``(nldof, 3)``, and ``fn`` the map as a JAX function of ``(..., 3)`` arrays.  The
    nodes of an element are then the map of the trilinear interpolant of its
    vertices, and the Jacobian is that of their degree-``p`` interpolant, which is
    what an isoparametric mesh of degree ``p`` with those nodes has.  Nothing of it
    is stored: a block of elements is computed on the device when asked for, so a
    mesh needs neither high-order nodes in MFEM nor geometry arrays on the host.

    Attributes
    ----------
    idx : ndarray of int32, ``(ne, n^3)``
        Local dof of each element node, lexicographic (``x`` fastest).
    X, G, wdet : ndarray (host geometry only)
        Node positions ``(ne, n^3, 3)``; ``w |J| J^{-1} J^{-T}`` at the nodes, its six
        distinct entries ``xx, xy, xz, yy, yz, zz`` (reference directions),
        ``(6, ne, n^3)``; and ``w |J|``, ``(ne, n^3)``.
    """

    def __init__(self, space, slice_ne=None, mapped=None):
        fes, mesh = space.fes, space.mesh
        if space.vdim != 1 or mesh.Dimension() != 3:
            raise NotImplementedError("the spectral element tables are for scalar H1 spaces on hexahedra")
        self.space = space
        ne = mesh.GetNE()
        self.ne = ne
        self.p = p = space.order
        self.n = n = p + 1
        self.nodes, self.w1, self.D = gll(p)
        nd = n ** 3
        fe = fes.GetFE(0) if ne else None
        if ne:
            if mesh.GetElementBaseGeometry(0) != mfem.Geometry.CUBE or fe.GetDof() != nd:
                raise NotImplementedError("hexahedral tensor-product elements only")
            ir = fe.GetNodes()
            ref = np.array([[ir.IntPoint(i).x, ir.IntPoint(i).y, ir.IntPoint(i).z]
                            for i in range(nd)])
            ijk = np.argmin(np.abs(ref[:, :, None] - self.nodes[None, None, :]), axis=2)
            if np.abs(ref - self.nodes[ijk]).max() > 1e-10:
                raise ValueError("the space's nodes are not Gauss-Lobatto points")
            lex = ijk[:, 0] + n * ijk[:, 1] + n * n * ijk[:, 2]
            self.nat_of_lex = np.argsort(lex)
            dofs, _ = _element_dofs(fes, np.arange(ne), 1, nd)
            self.idx = np.ascontiguousarray(dofs[:, self.nat_of_lex]).astype(np.int32)
        else:
            self.nat_of_lex = np.arange(nd)
            self.idx = np.zeros((0, nd), dtype=np.int32)
        self._reference()
        self.mapped = mapped is not None
        self.keyed = False
        self.comm = space.comm
        self._boxes = None
        if mapped is None:
            self._geometry(slice_ne)
        else:
            Vq, q, fn = mapped
            Nq, edofs = self.tabulate(Vq)
            self._Nq = Nq
            self._qv = np.ascontiguousarray(np.asarray(q, dtype=np.float64)[edofs])   # (ne, 8, 3)
            self._fn = fn
            self._geom = None

    @classmethod
    def from_keys(cls, Vq, order, q, fn, node_keys):
        """The tables of a degree ``order`` discretization of a structured mesh that
        MFEM holds only as a straight trilinear mesh.

        No MFEM space of degree ``order`` is built.  ``node_keys(qv, W) -> (keys,
        boundary)`` names the nodes: given the element vertices before the map,
        ``(ne, 8, 3)``, and the trilinear weights ``W`` (``(n^3, 8)``) of the vertices at
        the equispaced points of the reference cube, it returns an integer key for each
        of the ``n^3`` nodes of each element, the same for a node whichever element or
        rank sees it, and marks those on the boundary.  Local dofs are the distinct
        keys in increasing order, and ranks find what they share by the keys (see
        :meth:`Interface.from_keys`), so the number of unknowns is not bounded by
        MFEM's global numbering.  ``Vq``, ``q`` and ``fn`` are as in ``mapped``.
        """
        self = cls.__new__(cls)
        mesh = Vq.mesh
        self.space, self.comm = None, Vq.comm
        self.ne = ne = mesh.GetNE()
        self.p = p = int(order)
        self.n = n = p + 1
        self.nodes, self.w1, self.D = gll(p)
        self._reference()
        self.mapped, self.keyed, self._boxes, self._geom = True, True, None, None
        Nq, edofs = self.tabulate(Vq)
        self._Nq, self._fn = Nq, fn
        self._qv = np.ascontiguousarray(np.asarray(q, dtype=np.float64)[edofs])       # (ne, 8, 3)
        # the trilinear weights at the equispaced lattice of the reference cube
        a = np.arange(n ** 3)
        lattice = mfem.IntegrationRule(n ** 3)
        for k in range(n ** 3):
            lattice.IntPoint(k).Set3(float((a[k] % n) / p), float(((a[k] // n) % n) / p), float((a[k] // (n * n)) / p))
        W = (_basis_tables(Vq.fes.GetFE(0), lattice, 3, 8)[0] if ne else np.zeros((n ** 3, 8)))
        keys, boundary = node_keys(self._qv, W)
        uniq, inv = unique_inverse(keys.reshape(-1))
        self.keys = uniq
        self.nloc = int(uniq.size)
        self.idx = np.ascontiguousarray(inv.reshape(ne, n ** 3)).astype(np.int32)
        self.ess = np.zeros(self.nloc, dtype=bool)
        self.ess[self.idx[boundary]] = True
        # a node can be shared only if the vertices it hangs on are all shared
        vi = Interface(Vq)
        shared_v = np.zeros(Vq.fes.GetVSize(), dtype=bool)
        shared_v[vi.iface] = True
        hangs = W > 1e-12                                              # (n^3, 8)
        cand_e = ((~shared_v[edofs]).astype(np.int8) @ hangs.T.astype(np.int8)) == 0 if ne else np.zeros((0, n ** 3), bool)
        cand = np.zeros(self.nloc, dtype=bool)
        cand[self.idx[cand_e]] = True
        self.iface = Interface.from_keys(self.comm, uniq, cand)
        self.global_size = int(self.comm.allreduce(int(self.iface.owned.sum())))
        return self

    def _reference(self):
        """The nodes in lexicographic order, their weights, and the same as an MFEM rule
        (to tabulate other bases on)."""
        n = self.n
        nd = n ** 3
        a = np.arange(nd)
        self.ref = np.stack([self.nodes[a % n], self.nodes[(a // n) % n],
                             self.nodes[a // (n * n)]], axis=1)
        self.w = (self.w1[a % n] * self.w1[(a // n) % n] * self.w1[a // (n * n)])
        self.rule = mfem.IntegrationRule(nd)
        for q in range(nd):
            ip = self.rule.IntPoint(q)
            ip.Set3(float(self.ref[q, 0]), float(self.ref[q, 1]), float(self.ref[q, 2]))
            ip.weight = float(self.w[q])

    # ------------------------------------------------------- geometry of a block
    def _mapped_geometry(self):
        """The compiled geometry of a block of elements under the analytic map."""
        if self._geom is not None:
            return self._geom
        n, fn = self.n, self._fn
        dev = kernel_mod.device()
        Nq = jax.device_put(jnp.asarray(self._Nq), dev)
        D = jax.device_put(jnp.asarray(self.D), dev)
        w = jax.device_put(jnp.asarray(self.w), dev)

        def nodes(qv):
            return fn(jnp.einsum("qa,ead->eqd", Nq, qv))

        def geom(qv):
            X = nodes(qv)
            Xe = X.reshape(-1, n, n, n, 3)                                # [e, c, b, a, d]
            d0 = sum(Xe[:, :, :, i:i + 1] * D[:, i][:, None] for i in range(n))
            d1 = sum(Xe[:, :, i:i + 1] * D[:, i][:, None, None] for i in range(n))
            d2 = sum(Xe[:, i:i + 1] * D[:, i][:, None, None, None] for i in range(n))
            J = jnp.stack([d0, d1, d2], axis=-1).reshape(X.shape[0], n ** 3, 3, 3)   # dx_d / dxi_k
            a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]
            d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]
            g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]
            A = jnp.stack([jnp.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
                           jnp.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
                           jnp.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1)], -2)
            det = a * A[..., 0, 0] + b * A[..., 1, 0] + c * A[..., 2, 0]
            Ji = A / det[..., None, None]                                 # dxi_k / dx_d
            wd = w[None, :] * det
            G = jnp.stack([(Ji[..., k, 0] * Ji[..., l, 0] + Ji[..., k, 1] * Ji[..., l, 1]
                            + Ji[..., k, 2] * Ji[..., l, 2]) * wd
                           for (k, l) in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))])
            return X, G, wd, jnp.min(det)

        self._geom = (jax.jit(nodes), jax.jit(geom))
        return self._geom

    def block(self, a, b):
        """``(X, G, wdet)`` of elements ``a:b`` on the device: node positions
        ``(b - a, n^3, 3)``, the six geometric factors ``(6, b - a, n^3)`` and
        ``w |J|``, ``(b - a, n^3)``."""
        dev = kernel_mod.device()
        if not self.mapped:
            put = lambda x: jax.device_put(jnp.asarray(x), dev)
            return put(self.X[a:b]), put(self.G[:, a:b]), put(self.wdet[a:b])
        _, geom = self._mapped_geometry()
        X, G, wd, dmin = geom(jax.device_put(jnp.asarray(self._qv[a:b]), dev))
        if float(dmin) <= 0.0:
            raise ValueError("an element has a non-positive Jacobian at a node (min %.3e)" % float(dmin))
        return X, G, wd

    def block_nodes(self, a, b):
        """Node positions of elements ``a:b`` on the device, ``(b - a, n^3, 3)``."""
        dev = kernel_mod.device()
        if not self.mapped:
            return jax.device_put(jnp.asarray(self.X[a:b]), dev)
        nodes, _ = self._mapped_geometry()
        return nodes(jax.device_put(jnp.asarray(self._qv[a:b]), dev))

    def element_nodes(self, e):
        """Node positions of the single element ``e`` on the host, ``(n^3, 3)``."""
        if not self.mapped:
            return self.X[e]
        nodes, _ = self._mapped_geometry()
        return np.asarray(nodes(jnp.asarray(self._qv[e:e + 1])))[0]

    def boxes(self, block=65536):
        """Bounding boxes of the elements' nodes, ``(lo, hi)``, ``(ne, 3)`` each."""
        if self._boxes is None:
            if not self.mapped:
                self._boxes = (self.X.min(axis=1), self.X.max(axis=1))
            else:
                nodes, _ = self._mapped_geometry()
                lo, hi = np.empty((self.ne, 3)), np.empty((self.ne, 3))
                for a in range(0, self.ne, block):
                    b = min(a + block, self.ne)
                    X = nodes(jnp.asarray(self._qv[a:b]))
                    lo[a:b], hi[a:b] = np.asarray(X.min(axis=1)), np.asarray(X.max(axis=1))
                self._boxes = (lo, hi)
        return self._boxes

    def _geometry(self, slice_ne):
        mesh, ne, nq = self.space.mesh, self.ne, self.n ** 3
        nodes = mesh.GetNodes()
        if nodes is None:
            mesh.EnsureNodes()                                   # collective
            nodes = mesh.GetNodes()
        self.X = np.empty((ne, nq, 3))
        self.G = np.empty((6, ne, nq))
        self.wdet = np.empty((ne, nq))
        if ne == 0:
            return
        nfes = nodes.FESpace()
        gfe = nfes.GetFE(0)
        ng = gfe.GetDof()
        N, Gt = _basis_tables(gfe, self.rule, 3, ng)
        edofs, _ = _element_dofs(nfes, np.arange(ne), 3, ng)
        xall = np.array(nodes.GetDataArray())
        Gm = np.ascontiguousarray(np.transpose(Gt, (1, 0, 2)).reshape(ng, nq * 3))
        Nm = np.ascontiguousarray(N.T)
        m = int(slice_ne or max(256, 2 ** 26 // (nq * 9 * 8)))
        for a in range(0, ne, m):
            b = min(a + m, ne)
            x = xall[edofs[a:b]].reshape((b - a) * 3, ng)
            J = (x @ Gm).reshape(b - a, 3, nq, 3).transpose(0, 2, 1, 3)   # dx_d / dxi_k
            self.X[a:b] = (x @ Nm).reshape(b - a, 3, nq).transpose(0, 2, 1)
            Ji, det = _inverse3(J)                                        # dxi_k / dx_d
            if det.min() <= 0.0:
                raise ValueError("an element has a non-positive Jacobian at a node (min %.3e)" % det.min())
            wd = self.w[None, :] * det
            self.wdet[a:b] = wd
            for c, (k, l) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
                self.G[c, a:b] = (Ji[:, :, k, 0] * Ji[:, :, l, 0] + Ji[:, :, k, 1] * Ji[:, :, l, 1]
                                  + Ji[:, :, k, 2] * Ji[:, :, l, 2]) * wd

    def tabulate(self, other):
        """``(N, edofs)``: another H1 space's basis at this space's nodes (lexicographic)
        and its element dofs, for interpolating its functions to the nodes."""
        fes = other.fes
        if self.ne == 0:
            return np.zeros((self.n ** 3, 8)), np.zeros((0, 8), dtype=np.int32)
        fe = fes.GetFE(0)
        nd = fe.GetDof()
        N, _ = _basis_tables(fe, self.rule, 3, nd)
        edofs, _ = _element_dofs(fes, np.arange(self.ne), 1, nd)
        return N, edofs.astype(np.int32)


def unique_inverse(keys, chunk=1 << 27):
    """``np.unique(keys, return_inverse=True)`` for the keys of a rank's element nodes, the
    sort on the device when there is one.

    The host's indirect sort of the 64 keys of each of 15 million elements takes three to
    five minutes of a job; on the card the keys are sorted in two seconds, the distinct
    ones kept, and each key's place among them found by bisection, a chunk at a time
    (11 s in all, 21 GiB of the card at its peak, freed before the propagator takes it)."""
    keys = np.ascontiguousarray(keys)
    if keys.size < (1 << 24) or not kernel_mod.on_gpu():
        return np.unique(keys, return_inverse=True)
    dev = kernel_mod.device()
    s = np.asarray(jnp.sort(jax.device_put(keys, dev)))
    new = np.empty(s.size, dtype=bool)
    new[0] = True
    np.not_equal(s[1:], s[:-1], out=new[1:])
    uniq = s[new]
    del s, new
    ud = jax.device_put(uniq, dev)
    find = jax.jit(lambda u, k: jnp.searchsorted(u, k).astype(jnp.int32))
    inv = np.empty(keys.size, dtype=np.int32)
    for a in range(0, keys.size, chunk):
        inv[a:a + chunk] = np.asarray(find(ud, jax.device_put(keys[a:a + chunk], dev)))
    del ud
    return uniq, inv


def vertex_coordinates(mesh):
    """Positions of a straight-sided mesh's local vertices, ``(nv, 3)``: the local dofs
    of its trilinear space, in MFEM's numbering of both."""
    if mesh.GetNodes() is not None:
        raise ValueError("the mesh has nodes: its vertices are not where its geometry is")
    v = mfem.Vector()
    mesh.GetVertices(v)
    return np.array(v.GetDataArray()).reshape(mesh.SpaceDimension(), -1).T.copy()


# ------------------------------------------------------- dofs shared between ranks
class Interface:
    """The dofs a rank shares with others, and the exchange that sums over sharers.

    ``iface`` lists this rank's shared local dofs; for each neighbour, ``pos`` gives
    the positions in ``iface`` of the dofs shared with it, in the order of the global
    dof number (or key), which is the order the neighbour uses too.

    Built from an MFEM space, whose global true-dof numbers identify the dofs, or by
    :meth:`from_keys` from integer keys that name each dof whatever the rank.
    """

    def __init__(self, space=None):
        if space is None:
            return
        comm = space.comm
        self.comm = comm
        fes = space.fes
        nloc, ntrue = fes.GetVSize(), fes.GetTrueVSize()
        self.nloc = nloc
        sizes = np.array(comm.allgather(ntrue), dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(sizes)])
        me, size = comm.rank, comm.size
        if size == 1:
            self._alone(nloc)
            return
        label = space.vector()
        label.array[:] = starts[me] + np.arange(ntrue)
        gid = np.rint(space.local_values(label)).astype(np.int64)
        owner = np.searchsorted(starts, gid, side="right") - 1
        ghost = owner != me
        self.owned = ~ghost
        send = [np.zeros(0, dtype=np.int64) for _ in range(size)]
        gg, go = gid[ghost], owner[ghost]
        for o in np.unique(go):
            send[int(o)] = np.unique(gg[go == o])
        recv = comm.alltoall(send)                   # recv[r]: my dofs that rank r holds too
        sh_g = np.concatenate([recv[r] for r in range(size)] + [np.zeros(0, dtype=np.int64)])
        sh_r = np.concatenate([np.full(recv[r].size, r, dtype=np.int64) for r in range(size)]
                              + [np.zeros(0, dtype=np.int64)])
        ug = np.unique(sh_g)
        sh_g = np.concatenate([sh_g, ug])            # the owner shares each of them too
        sh_r = np.concatenate([sh_r, np.full(ug.size, me, dtype=np.int64)])
        got = comm.alltoall(self._pairs(sh_g, sh_r, size))
        lorder = np.argsort(gid, kind="stable")
        self._finish(got, gid[lorder], lorder)

    @classmethod
    def from_keys(cls, comm, keys, candidates):
        """The interface of dofs named by integer ``keys`` (sorted, one per local dof).

        ``candidates`` marks the local dofs that may be held by another rank (those
        that are certainly not need not be asked about).  Each candidate key is sent
        to a directory rank chosen by the key alone; the directory sees which ranks
        hold each key and tells every holder of a key held more than once who the
        others are.  A dof is owned by the lowest rank that holds it.
        """
        self = cls()
        self.comm = comm
        nloc = keys.size
        self.nloc = nloc
        me, size = comm.rank, comm.size
        if size == 1:
            self._alone(nloc)
            return self
        ck = np.ascontiguousarray(keys[candidates])
        dest = ((ck.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(size)
        order = np.argsort(dest, kind="stable")
        cut = np.searchsorted(dest[order], np.arange(size + 1))
        send = [ck[order[cut[r]:cut[r + 1]]] for r in range(size)]
        recv = comm.alltoall(send)
        sh_g = np.concatenate(recv)
        sh_r = np.concatenate([np.full(recv[r].size, r, dtype=np.int64) for r in range(size)])
        got = comm.alltoall(self._pairs(sh_g, sh_r, size))
        self._finish(got, keys, np.arange(nloc))
        self.owned = np.ones(nloc, dtype=bool)
        lower = np.concatenate([self._ldof[self._others == r] for r in self.neighbors if r < me]
                               + [np.zeros(0, dtype=np.int64)])
        self.owned[lower] = False
        return self

    def _alone(self, nloc):
        self.iface = np.zeros(0, dtype=np.int32)
        self.neighbors, self.pos = [], {}
        self.owned = np.ones(nloc, dtype=bool)

    @staticmethod
    def _pairs(sh_g, sh_r, size):
        """From the holders ``sh_r`` of the keys ``sh_g``, what to tell each rank: for
        every key held by several, the key and each other holder."""
        if sh_g.size == 0:
            e = np.zeros(0, dtype=np.int64)
            return [(e, e) for _ in range(size)]
        order = np.lexsort((sh_r, sh_g))
        sh_g, sh_r = sh_g[order], sh_r[order]
        first = np.concatenate([[True], sh_g[1:] != sh_g[:-1]])
        start = np.nonzero(first)[0]
        grp = np.cumsum(first) - 1
        count = np.diff(np.concatenate([start, [sh_g.size]]))
        posn = np.arange(sh_g.size) - start[grp]
        cnt = count[grp]
        pg, pa, pb = [], [], []                      # (key, rank a, other rank b)
        for shift in range(1, int(count.max())):
            sel = np.nonzero(cnt > shift)[0]
            oth = start[grp[sel]] + (posn[sel] + shift) % cnt[sel]
            pg.append(sh_g[sel]); pa.append(sh_r[sel]); pb.append(sh_r[oth])
        pg = np.concatenate(pg) if pg else np.zeros(0, np.int64)
        pa = np.concatenate(pa) if pa else np.zeros(0, np.int64)
        pb = np.concatenate(pb) if pb else np.zeros(0, np.int64)
        by = np.argsort(pa, kind="stable")
        cut = np.searchsorted(pa[by], np.arange(size + 1))
        return [(pg[by[cut[r]:cut[r + 1]]], pb[by[cut[r]:cut[r + 1]]]) for r in range(size)]

    def _finish(self, got, sorted_keys, local_of_sorted):
        """Set ``iface``, ``neighbors`` and ``pos`` from the directory's answers."""
        g_all = np.concatenate([g for g, _ in got])
        o_all = np.concatenate([o for _, o in got])
        where = np.searchsorted(sorted_keys, g_all) if g_all.size else np.zeros(0, np.int64)
        if g_all.size and not np.array_equal(sorted_keys[np.minimum(where, sorted_keys.size - 1)], g_all):
            raise RuntimeError("a shared dof was reported that this rank does not hold")
        ldof = local_of_sorted[where] if g_all.size else np.zeros(0, np.int64)
        self._ldof, self._others = ldof, o_all
        self.iface = np.unique(ldof).astype(np.int32)
        self.neighbors = [int(r) for r in np.unique(o_all)]
        self.pos = {}
        for r in self.neighbors:
            sel = o_all == r
            gs, ls = g_all[sel], ldof[sel]
            by = np.argsort(gs, kind="stable")
            self.pos[r] = np.searchsorted(self.iface, ls[by]).astype(np.int64)

    def layout(self):
        """What the device needs to do the gathering and the summing itself:
        ``(send, order)`` with ``send`` the positions in ``iface`` whose values go out,
        neighbour after neighbour (the same list a neighbour's values come back in),
        and ``order`` the sharers in ascending rank as ``(rank, start, stop)`` slices
        of that list, this rank itself as ``(rank, None, None)``."""
        me = self.comm.rank
        send = (np.concatenate([self.pos[r] for r in self.neighbors]).astype(np.int32)
                if self.neighbors else np.zeros(0, dtype=np.int32))
        starts = np.concatenate([[0], np.cumsum([self.pos[r].size for r in self.neighbors])]).astype(int)
        slices = {r: (int(starts[i]), int(starts[i + 1])) for i, r in enumerate(self.neighbors)}
        order = [(r,) + (slices[r] if r != me else (None, None)) for r in sorted(self.neighbors + [me])]
        self._slices = slices
        return send, order

    def swap(self, send):
        """Send the rows of ``send`` (``(nsend, nf)``, laid out as :meth:`layout` says) to
        the neighbours and return theirs in the same layout."""
        comm = self.comm
        send = np.ascontiguousarray(send, dtype=np.float64)
        recv = np.empty_like(send)
        reqs = []
        for r in self.neighbors:
            a, b = self._slices[r]
            reqs.append(comm.Irecv(recv[a:b], source=r, tag=71))
        for r in self.neighbors:
            a, b = self._slices[r]
            reqs.append(comm.Isend(send[a:b], dest=r, tag=71))
        MPI.Request.Waitall(reqs)
        return recv


# ------------------------------------------------------------ operators at points
class PointOperator:
    """Rows of weights on the nodes of single elements: point values, point values of a
    directional derivative, point sources.

    ``w`` is ``(nrow, n^3)``, ``dof`` the local dofs it multiplies, and ``index`` the
    global row number of each local row (a row lives on exactly one rank).
    """

    def __init__(self, w, dof, index, nglobal, comm):
        w = np.asarray(w, dtype=np.float64)
        self.w = np.ascontiguousarray(w).reshape(-1, w.shape[-1])
        self.dof = np.ascontiguousarray(dof, dtype=np.int32).reshape(self.w.shape)
        self.index = np.asarray(index, dtype=np.int64)
        self.nglobal = int(nglobal)
        self.comm = comm

    @property
    def nlocal(self):
        return self.index.size


def locate(col, points, comm, tol=2e-2, max_newton=30):
    """Which local element holds each point, and where in it.

    Returns ``(rows, elems, xi)`` for the points this rank is responsible for: every
    point found in an element of a rank goes to the lowest such rank, so each is
    handled once.  ``xi`` are reference coordinates in ``[0, 1]^3``, clipped: a point
    of the exact boundary lies a little outside the polynomial boundary of a coarse
    mesh, so a point up to ``tol`` outside an element (in reference units) is taken
    to be on its face, and among several candidates the one it is least outside of.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    npt = points.shape[0]
    n, nodes = col.n, col.nodes
    found = np.full(npt, -1, dtype=np.int64)
    xis = np.zeros((npt, 3))
    if col.ne:
        lo, hi = col.boxes()
        pad = 0.05 * (hi - lo).max(axis=1, keepdims=True) + 1e-12
        lo, hi = lo - pad, hi + pad
        for ip in range(npt):
            x = points[ip]
            cand = np.nonzero(np.all((x >= lo) & (x <= hi), axis=1))[0]
            best = None
            for e in cand:
                Xe = col.element_nodes(e).reshape(n, n, n, 3)    # [c, b, a]
                xi = np.full(3, 0.5)
                ok = False
                for _ in range(max_newton):
                    la, da = lagrange(nodes, xi[0])
                    lb, db = lagrange(nodes, xi[1])
                    lc, dc = lagrange(nodes, xi[2])
                    pos = np.einsum("c,b,a,cbad->d", lc, lb, la, Xe)
                    J = np.stack([np.einsum("c,b,a,cbad->d", lc, lb, da, Xe),
                                  np.einsum("c,b,a,cbad->d", lc, db, la, Xe),
                                  np.einsum("c,b,a,cbad->d", dc, lb, la, Xe)], axis=1)
                    dxi = np.linalg.solve(J, x - pos)
                    xi = np.clip(xi + dxi, -0.5, 1.5)
                    if np.abs(dxi).max() < 1e-13:
                        ok = True
                        break
                if ok and xi.min() >= -tol and xi.max() <= 1.0 + tol:
                    out = np.abs(np.clip(xi, 0.0, 1.0) - xi).max()
                    if best is None or out < best[0]:
                        best = (out, int(e), np.clip(xi, 0.0, 1.0))
            if best is not None:
                found[ip], xis[ip] = best[1], best[2]
    have = np.where(found >= 0, comm.rank, comm.size).astype(np.int64)
    owner = np.empty_like(have)
    comm.Allreduce(have, owner, op=MPI.MIN)
    if owner.max() >= comm.size:
        raise ValueError("point(s) %s lie in no element" % np.nonzero(owner >= comm.size)[0].tolist())
    mine = np.nonzero(owner == comm.rank)[0]
    return mine, found[mine], xis[mine]


def value_operator(col, points, comm):
    """Point values at ``points`` (also the load of unit point sources there)."""
    rows, elems, xi = locate(col, points, comm)
    n = col.n
    w = np.zeros((rows.size, n ** 3))
    for i in range(rows.size):
        la, _ = lagrange(col.nodes, xi[i, 0])
        lb, _ = lagrange(col.nodes, xi[i, 1])
        lc, _ = lagrange(col.nodes, xi[i, 2])
        w[i] = np.einsum("c,b,a->cba", lc, lb, la).reshape(-1)
    return PointOperator(w, col.idx[elems], rows, np.asarray(points).reshape(-1, 3).shape[0], comm)


def derivative_operator(col, points, directions, scale, comm):
    """``scale * direction . grad`` at ``points``: the rows a receiver of ground motion
    reads a pressure field with."""
    rows, elems, xi = locate(col, points, comm)
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    scale = np.broadcast_to(np.asarray(scale, dtype=float), (directions.shape[0],))
    n = col.n
    w = np.zeros((rows.size, n ** 3))
    for i in range(rows.size):
        e = elems[i]
        la, da = lagrange(col.nodes, xi[i, 0])
        lb, db = lagrange(col.nodes, xi[i, 1])
        lc, dc = lagrange(col.nodes, xi[i, 2])
        Xe = col.element_nodes(e).reshape(n, n, n, 3)
        J = np.stack([np.einsum("c,b,a,cbad->d", lc, lb, da, Xe),
                      np.einsum("c,b,a,cbad->d", lc, db, la, Xe),
                      np.einsum("c,b,a,cbad->d", dc, lb, la, Xe)], axis=1)   # dx_d/dxi_k
        Ji = np.linalg.inv(J)                      # dxi_k / dx_d
        nd = directions[rows[i]] @ Ji.T            # n_d dxi_k/dx_d, k = 0..2
        ref = (nd[0] * np.einsum("c,b,a->cba", lc, lb, da)
               + nd[1] * np.einsum("c,b,a->cba", lc, db, la)
               + nd[2] * np.einsum("c,b,a->cba", dc, lb, la))
        w[i] = scale[rows[i]] * ref.reshape(-1)
    return PointOperator(w, col.idx[elems], rows, directions.shape[0], comm)


# ------------------------------------------------------------------ the propagator
def _split_density(density):
    """The mass coefficient and the stiffness coefficient of a wave density, as
    functions of the parameter value, the position and the auxiliary values at a
    point: the second derivatives of the density with respect to ``(a, p)`` and to
    ``(grad u, grad p)``, which do not depend on the fields for a linear wave."""
    z3 = jnp.zeros(3)
    e0 = jnp.array([1.0, 0.0, 0.0])

    def kappa(m, x, *aux):
        def f(a, p):
            return density(Field(0.0, z3), Field(a, z3), Field(m, z3), Field(p, z3), x, *aux)
        return jax.grad(lambda a: jax.grad(lambda p: f(a, p))(0.0))(0.0)

    def mu(m, x, *aux):
        def f(s, t):
            return density(Field(0.0, s * e0), Field(0.0, z3), Field(m, z3), Field(0.0, t * e0), x, *aux)
        return jax.grad(lambda s: jax.grad(lambda t: f(s, t))(0.0))(0.0)

    def tensor(m, x, *aux):
        def f(gu, gp):
            return density(Field(0.0, gu), Field(0.0, z3), Field(m, z3), Field(0.0, gp), x, *aux)
        return jax.jacfwd(lambda gu: jax.grad(lambda gp: f(gu, gp))(z3))(z3)

    return kappa, mu, tensor


class SpectralWave:
    """Leapfrog spectral-element propagation of ``M(m) u_tt + K u = f`` with its
    discrete adjoint and its sensitivities to ``m``.

    Parameters
    ----------
    Vu, Vm : FunctionSpace
        The wave field's space (H1 on hexahedra, any degree) and the parameter's.
        ``Vu`` may be None when ``col`` numbers the unknowns itself
        (:meth:`Collocation.from_keys`).
    density : callable
        ``density(u, a, m, p, x, *aux) -> scalar`` at a point (see the module text).
    aux : sequence of ndarray, or callable, optional
        What the density takes after ``x``: values at the nodes of every element,
        ``(ne, n^3)`` each, or a JAX function of a block's node positions
        ``(nb, n^3, 3)`` returning them for the block, which is then evaluated on the
        device as the coefficients are.
    ess_bdr : sequence of int or "all" or None
        Boundary attributes where the field is zero.
    dt, nsteps : float, int
        Time step and number of steps; ``dt=None`` takes ``cfl`` of the stability
        limit at the parameter first set.
    dtype : jnp.float64 or jnp.float32
        Precision of the fields on the device.
    geometry_dtype : dtype, optional
        Precision the geometric factors are stored in on the device (``dtype`` by
        default).  They are the larger part of what a propagation keeps there, and in
        single precision they still define an exactly symmetric stiffness matrix, so
        every identity between the sweeps holds as before; the matrix differs from
        the double-precision one by about 1e-7 relative.
    element_chunk : int
        Element-fields per pass of the stiffness kernel: with ``nf`` fields a pass
        takes ``element_chunk / nf`` elements (512 at least).  0 takes all the
        elements in one pass.

    element_block : int
        Elements per block of the arrays kept per element on the device (geometric
        factors, dof numbers); 0 keeps each in one array.  Splitting them changes
        nothing but the size of the arrays a kernel indexes, which matters on one
        card: for the RTX PRO 6000 Blackwell, CUDA 12.9's assembler crashes on a
        kernel that indexes an array of more than about 2^30 entries (the geometric
        factors of 2.8 million elements), and half a million elements to a block
        keeps every array under it.
    sensitivities : bool
        False keeps nothing for the derivatives with respect to the parameter (the
        mass coefficient's derivative at the nodes, a sixth of what a propagation
        holds on the device): the forward sweep alone is then available.

    Several fields are propagated at once, stored ``(nloc, nfields)`` with the field
    index last: the geometric factors and the gather and scatter indices are then
    read once for all of them, so nine fields cost six to eight times one, not nine.
    """

    def __init__(self, Vu, Vm, density, aux=(), ess_bdr="all", dt=None, nsteps=None,
                 cfl=0.8, dtype=jnp.float64, element_chunk=10000, col=None,
                 geometry_dtype=None, param_chunk=65536, element_block=0, sensitivities=True):
        self.Vu, self.Vm = Vu, Vm
        self.comm = Vm.comm
        self.dtype = dtype
        self.sensitivities = bool(sensitivities)
        self.geometry_dtype = geometry_dtype if geometry_dtype is not None else dtype
        self.density = density
        self.col = col if col is not None else Collocation(Vu)
        col = self.col
        self.ne, self.n, self.nq = col.ne, col.n, col.n ** 3
        self.dt, self.nsteps, self.cfl = dt, nsteps, float(cfl)
        if col.keyed:                    # the wave field's unknowns are numbered by keys, not by MFEM
            self.nloc, self.iface = col.nloc, col.iface
            self.global_size = col.global_size
        else:
            self.nloc, self.iface = Vu.fes.GetVSize(), Interface(Vu)
            self.global_size = int(Vu.GlobalTrueVSize())
        dev = kernel_mod.device()
        self.device = dev
        put = lambda x, dt_=None: jax.device_put(jnp.asarray(x, dtype=dt_ if dt_ is not None else dtype), dev)
        self._put = put
        # parameter interpolation to the nodes
        Nm, edofs_m = col.tabulate(Vm)
        self.nloc_m = Vm.fes.GetVSize()
        self._Nm = put(Nm)
        # the arrays kept per element, in blocks of elements (one block by default)
        B = int(element_block) or max(self.ne, 1)
        self._eblocks = [(a, min(a + B, self.ne)) for a in range(0, self.ne, B)] or [(0, 0)]
        self._edofs_m = tuple(put(edofs_m[a:b], jnp.int32) for (a, b) in self._eblocks)
        self._idx = tuple(put(col.idx[a:b], jnp.int32) for (a, b) in self._eblocks)
        self._D = put(col.D)
        self._aux_fn = aux if callable(aux) else None
        self._aux_host = () if callable(aux) else tuple(np.asarray(a, dtype=np.float64) for a in aux)
        self._iface_idx = put(self.iface.iface, jnp.int32)
        send, self._iface_order = self.iface.layout()
        self._iface_send = put(send, jnp.int32)
        # (block, range in the block, range in the mesh) of the pieces the coefficients
        # and the geometry are computed in
        pc = int(param_chunk)
        self._pieces = [(j, la, min(la + pc, b - a), a + la, a + min(la + pc, b - a))
                        for j, (a, b) in enumerate(self._eblocks) for la in range(0, b - a, pc)]
        # essential dofs, as a mask on local dofs
        ess = np.zeros(self.nloc, dtype=bool)
        mesh = Vm.mesh
        if col.keyed:
            ess = col.ess if ess_bdr is not None else ess
        elif ess_bdr is not None and mesh.bdr_attributes.Size():
            nb = mesh.bdr_attributes.Max()
            marker = mfem.intArray([1] * nb if ess_bdr == "all" else
                                   [1 if (k + 1) in set(int(b) for b in ess_bdr) else 0 for k in range(nb)])
            vd = mfem.intArray()
            Vu.fes.GetEssentialVDofs(marker, vd)
            ess = np.array(vd.GetDataArray()) != 0
        self.ess = ess
        self._free = put(~ess, jnp.bool_)
        self._kappa, self._mu, self._tensor = _split_density(density)
        self.element_chunk = int(element_chunk)
        self._G = None
        self._Minv = None
        self._m_key = None
        self._final = None
        self._jit = {}
        self.sources = None
        self.receivers = None
        self.n_field_steps = 0           # field-steps taken: the unit of work
        self.n_sweeps = 0
        self.t_wave = 0.0
        self.t_exchange = 0.0            # of which, in the exchange between ranks

    # ------------------------------------------------------------------ pieces
    def _stiff_fn(self):
        """``K U`` for fields ``U`` (``(nloc, nf)``), element contributions only (not
        yet summed over the ranks sharing a dof).

        The three reference derivatives are written as sums over the ``n`` nodes of a
        line rather than as tensor contractions: XLA then fuses each into one
        elementwise kernel, which in double precision is eight times faster than the
        contraction it would otherwise lower to.

        The elements are taken a chunk at a time in a compiled loop that accumulates
        into one output, so the kernel's temporaries (five arrays of element-node
        values per field) are a chunk's, not the mesh's.  That is about half the
        memory on every card.  The speed depends on the card: with 10 000
        element-fields to a chunk the temporaries stay in an L40S's cache, and one
        field is 1.7 times and nine fields 1.45 times faster than in a single pass;
        on an H100 or H200 a single pass is 10 to 25 % faster for one and for nine
        fields (``element_chunk=0``).
        """
        n, nq = self.n, self.nq
        work = self.element_chunk

        def element(ue, g, D):
            # ue [e, c, b, a, f], g [6, e, c, b, a, 1] -> the elements' K u, same shape as ue
            Dt = D.T
            ux = sum(ue[:, :, :, i:i + 1] * D[:, i][:, None] for i in range(n))
            uy = sum(ue[:, :, i:i + 1] * D[:, i][:, None, None] for i in range(n))
            uz = sum(ue[:, i:i + 1] * D[:, i][:, None, None, None] for i in range(n))
            fx = g[0] * ux + g[1] * uy + g[2] * uz
            fy = g[1] * ux + g[3] * uy + g[4] * uz
            fz = g[2] * ux + g[4] * uy + g[5] * uz
            return (sum(fx[:, :, :, i:i + 1] * Dt[:, i][:, None] for i in range(n))
                    + sum(fy[:, :, i:i + 1] * Dt[:, i][:, None, None] for i in range(n))
                    + sum(fz[:, i:i + 1] * Dt[:, i][:, None, None, None] for i in range(n)))

        def add(out, U, ie, ge, D):
            nf = U.shape[1]
            k = ie.shape[0]
            r = element(U[ie].reshape(k, n, n, n, nf), ge.reshape(6, k, n, n, n)[..., None], D)
            return out.at[ie.reshape(-1)].add(r.reshape(-1, nf).astype(U.dtype))

        def block(out, U, idx, G, D):
            nb = idx.shape[0]
            m = nb if not work else min(nb, max(512, work // U.shape[1]))
            nfull = nb // m if m else 0
            rem = nb - nfull * m
            if nfull <= 1 and rem == 0:
                return add(out, U, idx, G, D)

            def body(out, k):
                ie = jax.lax.dynamic_slice(idx, (k * m, 0), (m, nq))
                ge = jax.lax.dynamic_slice(G, (0, k * m, 0), (6, m, nq))
                return add(out, U, ie, ge, D), None

            out, _ = jax.lax.scan(body, out, jnp.arange(nfull))
            if rem:
                out = add(out, U, idx[nfull * m:], G[:, nfull * m:], D)
            return out

        def stiff(U, idxs, Gs, D):
            out = jnp.zeros_like(U)
            for idx, G in zip(idxs, Gs):
                if idx.shape[0]:
                    out = block(out, U, idx, G, D)
            return out

        return stiff

    def _interp_m(self, m_loc, j, a=0, b=None):
        """Parameter values at the nodes of elements ``a:b`` of block ``j``,
        ``(b - a, n^3)``."""
        return jnp.einsum("qa,ea->eq", self._Nm, m_loc[self._edofs_m[j][a:b]])

    def _adder(self):
        """``M.at[i].add(v)``, written into ``M``'s own buffer."""
        return self._get("add", lambda: jax.jit(lambda M, i, v: M.at[i].add(v.astype(M.dtype)), donate_argnums=0))

    def _assemble_nodal(self, elem_vals, k):
        """Sum element nodal values (one ``(ne_j, n^3, k)`` array per block, taken from
        the iterable one at a time) into local dof vectors ``(nloc, k)`` and over the
        sharing ranks."""
        add = self._adder()
        loc = jax.device_put(jnp.zeros((self.nloc, k), dtype=self.dtype), self.device)
        for idx, v in zip(self._idx, elem_vals):
            loc = add(loc, idx.reshape(-1), v.reshape(-1, k))
        return self._sync(loc)

    def _merge_fn(self):
        """``(shared, merge)``: the partial values this rank sends for the shared dofs,
        and the residual with the shared dofs' totals in place of its partial values.

        Both the gather and the sum are done on the device; the host only passes the
        buffers between ranks.  The contributions are added in ascending rank order,
        this rank's among them, so every sharer of a dof computes the same bits."""
        order = self._iface_order

        def shared(R, iface, send):
            return R[iface][send]

        def merge(R, recv, iface, send):
            own = R[iface]
            tot = jnp.zeros_like(own)
            recv = recv.astype(R.dtype)
            for r, a, b in order:
                tot = tot + own if a is None else tot.at[send[a:b]].add(recv[a:b])
            return R.at[iface].set(tot)

        return shared, merge

    def _sync(self, R):
        """Replace the shared dofs' partial sums in ``R`` (``(nloc, nf)``) by their
        totals over the ranks."""
        if not self.iface.neighbors:
            return R
        # the totals are written into R's own buffer: R is the caller's no longer
        shared, merge = self._get("merge", lambda: (jax.jit(self._merge_fn()[0]),
                                                    jax.jit(self._merge_fn()[1], donate_argnums=0)))
        recv = self.iface.swap(np.asarray(shared(R, self._iface_idx, self._iface_send)))
        return merge(R, recv, self._iface_idx, self._iface_send)

    # --------------------------------------------------------------- parameter
    def _aux_of(self, a, b, X):
        if self._aux_fn is not None:
            return tuple(self._aux_fn(X))
        return tuple(self._put(x[a:b]) for x in self._aux_host)

    def set_parameter(self, m, force=False):
        """Fix the parameter: the diagonal mass and its derivative at the nodes.

        The coefficients are evaluated a block of elements at a time, with the block's
        geometry (computed or streamed, see :class:`Collocation`) on the device for
        that long only.  The first call also builds the geometric factors of the
        stiffness, which do not depend on the parameter.
        """
        key = np.array(m.array, copy=True)
        same = (not force) and self._m_key is not None and np.array_equal(key, self._m_key)
        # one answer for every rank: what follows communicates, and a rank whose part of
        # the parameter did not change (or is empty) must not leave the others waiting
        if self.comm.allreduce(bool(same), op=MPI.LAND):
            return self
        col, nq = self.col, self.nq
        m_loc = self._put(self.Vm.local_values(m))
        kap_fn = self._get("kappa", lambda: jax.jit(jax.vmap(jax.vmap(self._kappa))))
        dkap_fn = self._get("dkappa", lambda: jax.jit(jax.vmap(jax.vmap(jax.grad(self._kappa)))))
        first = self._G is None
        # the arrays kept per element are filled piece by piece in place: a
        # concatenation would hold each twice
        zeros = lambda shape, dt_: jax.device_put(jnp.zeros(shape, dtype=dt_), self.device)
        fill = self._get("fill", lambda: jax.jit(
            lambda G, c, a: jax.lax.dynamic_update_slice(G, c.astype(G.dtype), (0,) * (G.ndim - 2) + (a, 0)),
            donate_argnums=0))
        add = self._adder()
        if first:
            mu_fn = jax.jit(jax.vmap(jax.vmap(self._mu)))
            dmu_fn = jax.jit(jax.vmap(jax.vmap(jax.grad(self._mu))))
            ten_fn = jax.jit(jax.vmap(self._tensor))
            Gs = [zeros((6, b - a, nq), self.geometry_dtype) for (a, b) in self._eblocks]
            bad, off, vol = 0.0, 0.0, 0.0
        Mloc = zeros(self.nloc, self.dtype)
        keep = self.sensitivities
        self._wdkap = None                                    # the last parameter's, before the new one's
        wdk = [zeros(((b - a) if keep else 0, nq), self.dtype) for (a, b) in self._eblocks]
        for (j, la, lb, a, b) in self._pieces:
            X, G, wd = col.block(a, b)
            aux = self._aux_of(a, b, X)
            me = self._interp_m(m_loc, j, la, lb)
            Mloc = add(Mloc, self._idx[j][la:lb].reshape(-1), (wd * kap_fn(me, X, *aux)).reshape(-1))
            if keep:
                wdk[j] = fill(wdk[j], wd * dkap_fn(me, X, *aux), la)
            if first:
                mu = mu_fn(me, X, *aux)
                bad = max(bad, float(jnp.max(jnp.abs(dmu_fn(me, X, *aux)))))
                # the density's stiffness tensor must be isotropic: mu times the identity
                T = ten_fn(me[:, 0], X[:, 0], *[x[:, 0] for x in aux])
                off = max(off, float(jnp.max(jnp.abs(T - mu[:, 0, None, None] * jnp.eye(3))))
                          / max(float(jnp.max(jnp.abs(mu))), 1e-300))
                vol += float(jnp.sum(wd))
                Gs[j] = fill(Gs[j], G * mu[None], la)
            del X, G, wd, aux
        if first:
            if self.comm.allreduce(bad, op=MPI.MAX) > 0.0:
                raise NotImplementedError(
                    "the stiffness coefficient depends on the parameter; only the mass "
                    "coefficient may (write the wave equation with the parameter in the "
                    "second time derivative's coefficient)")
            if self.comm.allreduce(off, op=MPI.MAX) > 1e-12:
                raise NotImplementedError("an anisotropic stiffness tensor is not supported")
            self._G = tuple(Gs)
            #: the domain's volume, summed from the nodal weights
            self.volume = self.comm.allreduce(vol)
        self._M = self._Minv = None
        M = self._sync(Mloc[:, None])[:, 0]
        del Mloc
        if self.comm.allreduce(float(jnp.min(jnp.where(self._free, M, 1.0))) if self.nloc else 1.0,
                               op=MPI.MIN) <= 0.0:
            raise ValueError("the mass matrix has a non-positive entry")
        self._M = M
        self._Minv = jnp.where(self._free, 1.0 / M, 0.0)
        # d M_e / d m at the nodes, (ne_j, nq) per block
        self._wdkap = tuple(wdk)
        self._m_key = key
        self._final = None
        if self.dt is None:
            self.dt = self.cfl * self.stable_dt()
        return self

    def stable_dt(self, iters=60):
        """The leapfrog stability limit ``2 / sqrt(lambda_max(M^{-1} K))`` by power
        iteration (the largest eigenvalue from below, so apply a safety factor).

        The start vector is a function of the node positions, so the estimate does not
        depend on how the mesh is partitioned (beyond round-off).
        """
        stiff = self._get("stiff", lambda: jax.jit(self._stiff_fn()))
        # every step below writes into the buffer of the vector it replaces, so the
        # iteration holds two vectors next to the one the stiffness is applied to
        put = self._get("put", lambda: jax.jit(lambda v, i, h: v.at[i].set(h.astype(v.dtype)), donate_argnums=0))
        v = jax.device_put(jnp.zeros(self.nloc, dtype=self.dtype), self.device)
        k = self._put(np.array([12.9898, 78.233, 37.719]) * 1e-2, jnp.float64)
        for (j, la, lb, a, b) in self._pieces:
            h = jnp.sin(self.col.block_nodes(a, b) @ k) * 43758.5453
            v = put(v, self._idx[j][la:lb].reshape(-1), (h - jnp.floor(h) - 0.5).reshape(-1))
            del h
        v = jnp.where(self._free, v, 0.0)[:, None]
        owned = self._put(self.iface.owned, jnp.bool_)

        def rayleigh(w, v, owned, M, free, Minv):
            # (v, K v), (v, M v) and M^{-1} K v with its norm, over the dofs this rank owns
            w = jnp.where(free, w[:, 0], 0.0)
            num = jnp.sum(jnp.where(owned, w * v[:, 0], 0.0))
            den = jnp.sum(jnp.where(owned, M * v[:, 0] * v[:, 0], 0.0))
            w = w * Minv
            return w[:, None], num, den, jnp.sum(jnp.where(owned, w * w, 0.0))

        quotient = self._get("quotient", lambda: jax.jit(rayleigh, donate_argnums=0))
        scale = self._get("scale", lambda: jax.jit(lambda w, s: w / s, donate_argnums=0))
        lam = 0.0
        for _ in range(iters):
            w = self._sync(stiff(v, self._idx, self._G, self._D))
            w, num, den, nrm2 = quotient(w, v, owned, self._M, self._free, self._Minv)
            lam = self.comm.allreduce(float(num)) / self.comm.allreduce(float(den))   # Rayleigh quotient of (K, M)
            v = scale(w, np.sqrt(self.comm.allreduce(float(nrm2))))
        return 2.0 / np.sqrt(lam)

    def _get(self, name, build):
        if name not in self._jit:
            self._jit[name] = build()
        return self._jit[name]

    # ---------------------------------------------------------- sources, receivers
    def set_sources(self, op, wavelets):
        """Point sources: ``op`` a :class:`PointOperator` (one row per source) and
        ``wavelets`` their amplitudes in time, ``(nsteps + 1, nsources)``."""
        self.sources = op
        self._src_w = self._put(op.w)
        self._src_dof = self._put(op.dof, jnp.int32)
        self._src_t = self._put(np.asarray(wavelets)[:, op.index])    # (N+1, nloc_src)
        self._drop_steps()
        return self

    def set_receivers(self, op):
        self.receivers = op
        self._rec_w = self._put(op.w)
        self._rec_dof = self._put(op.dof, jnp.int32)
        self._drop_steps()
        return self

    def _drop_steps(self):
        for k in [k for k in self._jit if isinstance(k, tuple) and k[0] == "step"]:
            del self._jit[k]
        self._final = None

    # ------------------------------------------------------------------- steps
    def _build_step(self, has_q, has_dM, record, correlate):
        """The compiled halves of a step, ``(first, second, whole)``.

        ``first`` is the element work, up to where the shared dofs must be summed over
        the ranks, and returns the residual with this rank's partial values at them;
        ``second`` takes their totals and updates.  ``whole`` is the two in one call,
        for a single rank.

        Field 0 is the main field, ``R = K u - f_n``.  Fields 1.. are either adjoint
        fields, with the series ``q`` injected at the receivers, or incremental fields,
        forced through ``second`` by the mass perturbations times the main field's
        increment.
        """
        stiff = self._stiff_fn()
        has_src = self.sources is not None and self.sources.nlocal > 0
        has_rec = self.receivers is not None and self.receivers.nlocal > 0
        serial = not self.iface.neighbors

        def first(c, U, n, q):
            R = stiff(U, c["idx"], c["G"], c["D"])
            if has_src:
                R = R.at[c["src_dof"], 0].add(-(c["src_t"][n][:, None] * c["src_w"]))
            if has_q and has_rec:
                nb = U.shape[1] - 1
                add = jnp.einsum("rk,br->rkb", c["rec_w"], q[:, n]) / c["dt2"]
                R = R.at[c["rec_dof"][:, :, None], jnp.arange(1, nb + 1)[None, None, :]].add(add)
            return R

        def second(c, U, Um, R, S, MdM):
            A = (-c["dt2"]) * c["Minv"][:, None] * R
            if has_dM:
                # by concatenation: an indexed update of a slice lowers to a scatter
                A = jnp.concatenate([A[:, :1], A[:, 1:] - MdM * A[:, :1]], axis=1)
            Unew = 2.0 * U - Um + A
            if correlate:
                S = S + Unew[:, 1:] * A[:, :1]
            rec = (jnp.einsum("rk,rkf->fr", c["rec_w"], Unew[c["rec_dof"]])
                   if (record and has_rec) else None)
            return Unew, S, rec

        shared, merge = self._merge_fn()

        def first_par(c, U, n, q):
            R = first(c, U, n, q)
            return R, shared(R, c["iface"], c["send"])

        def second_par(c, U, Um, R, recv, S, MdM):
            # the merged residual is returned too: that lets it take R's buffer
            R = merge(R, recv, c["iface"], c["send"])
            return second(c, U, Um, R, S, MdM) + (R,)

        def whole(c, U, Um, n, q, S, MdM):
            return second(c, U, Um, first(c, U, n, q), S, MdM)

        # the previous state, the residual and the correlation are donated: the new
        # state, the merged residual and the new correlation are written into their buffers
        if serial:
            return None, None, jax.jit(whole, donate_argnums=(2, 5))
        return jax.jit(first_par), jax.jit(second_par, donate_argnums=(2, 3, 5)), None

    def _consts(self):
        z = jnp.zeros((0, self.nq), dtype=self.dtype)
        zi = jnp.zeros((0, self.nq), dtype=jnp.int32)
        c = {"idx": self._idx, "G": self._G, "D": self._D, "Minv": self._Minv,
             "iface": self._iface_idx, "send": self._iface_send,
             "dt2": jnp.asarray(self.dt ** 2, dtype=self.dtype)}
        if self.sources is not None:
            c.update(src_w=self._src_w, src_dof=self._src_dof, src_t=self._src_t)
        else:
            c.update(src_w=z, src_dof=zi, src_t=jnp.zeros((self.nsteps + 1, 0), dtype=self.dtype))
        if self.receivers is not None:
            c.update(rec_w=self._rec_w, rec_dof=self._rec_dof)
        else:
            c.update(rec_w=z, rec_dof=zi)
        return c

    def _run(self, U, Um, steps, q=None, dM=None, record=False, correlate=False):
        """March ``(U, Um)`` (``(nloc, nf)``) through ``steps`` (the step indices handed
        to the sources and to ``q``), optionally recording the receivers after each
        step or accumulating the correlation of fields 1.. with field 0's increment.

        Returns ``(U, Um, records, S)``; the records are ``(nsteps, nf, nlocal)``.
        ``Um`` and ``dM`` are consumed (their buffers take the next state and
        ``M^{-1} dM``), so ``Um`` must not be ``U``.
        """
        nf = U.shape[1]
        has_q, has_dM = q is not None, dM is not None
        key = ("step", nf, has_q, has_dM, record, correlate)
        first, second, whole = self._get(key, lambda: self._build_step(has_q, has_dM, record, correlate))
        c = self._consts()
        MdM = None
        if has_dM:
            # M^{-1} dM, in dM's own buffer
            scale = self._get("mdm", lambda: jax.jit(lambda Minv, d: Minv[:, None] * d, donate_argnums=1))
            MdM, dM = scale(self._Minv, dM), None
        S = jax.device_put(jnp.zeros((self.nloc, nf - 1), dtype=self.dtype), self.device) if correlate else None
        recs = []
        t0 = time.perf_counter()
        for n in steps:
            if whole is not None:
                Unew, S, rec = whole(c, U, Um, n, q, S, MdM)
            else:
                R, out = first(c, U, n, q)
                out = np.asarray(out)                       # waits for the element work
                tx = time.perf_counter()
                recv = self.iface.swap(out)
                self.t_exchange += time.perf_counter() - tx
                Unew, S, rec, R = second(c, U, Um, R, recv, S, MdM)
                del R
            Um, U = U, Unew
            if rec is not None:
                recs.append(rec)
        nrec = self.receivers.nlocal if self.receivers is not None else 0
        if record:
            out = (np.stack([np.asarray(r, dtype=np.float64) for r in recs]) if recs
                   else np.zeros((len(steps), nf, nrec)))
        else:
            out = None
            jax.block_until_ready(U)
        self.n_field_steps += nf * len(steps)
        self.n_sweeps += 1
        self.t_wave += time.perf_counter() - t0
        return U, Um, out, S

    def _gather_records(self, loc):
        """``(nsteps, nf, nlocal)`` records of this rank's receivers to ``(nf, nsteps + 1,
        nreceivers)`` on every rank, the row of step 0 being zero."""
        rec = self.receivers
        parts = self.comm.allgather((rec.index, loc))
        nf = loc.shape[1]
        y = np.zeros((nf, self.nsteps + 1, rec.nglobal))
        for index, part in parts:
            if index.size:
                y[:, 1:, index] = np.transpose(part, (1, 0, 2))
        if not np.isfinite(y).all():
            raise RuntimeError("the wave field is not finite: the time step exceeds the "
                               "stability limit of this parameter")
        return y

    # ------------------------------------------------------------------ sweeps
    def forward(self, limit=None):
        """Propagate from rest; returns the receivers' series ``(nsteps + 1, nrec)`` and
        keeps the last two states, from which every other sweep starts.  ``limit``
        stops after that many steps and keeps nothing (to compile the sweep)."""
        zero = lambda: jax.device_put(jnp.zeros((self.nloc, 1), dtype=self.dtype), self.device)
        if limit is not None:
            self._run(zero(), zero(), range(int(limit)), record=True)
            return None
        self._final = None
        U, Um, out, _ = self._run(zero(), zero(), range(self.nsteps), record=True)
        self._final = (U, Um)                       # the sweep's own arrays: no copy
        return self._gather_records(out)[0]

    def final_state(self):
        """The last two states ``(u_N, u_{N-1})`` of the forward sweep, on the device
        (columns, ``(nloc, 1)``)."""
        return self._final

    def _need_sensitivities(self):
        if not self.sensitivities:
            raise RuntimeError("this propagator was built with sensitivities=False: it has the forward sweep only")

    def mass_perturbations(self, directions):
        """``dM/dm [dm]`` as local dof vectors, ``(nloc, ndir)``, for parameter
        directions ``directions``."""
        self._need_sensitivities()
        mloc = [self._put(self.Vm.local_values(d)) for d in directions]
        vals = (jnp.stack([w * self._interp_m(m, j) for m in mloc], axis=-1)
                for j, w in enumerate(self._wdkap))               # a block at a time
        return self._assemble_nodal(vals, len(mloc))

    def incremental(self, directions, limit=None):
        """Sensitivity of the receivers' series to the parameter in each direction:
        ``(ndir, nsteps + 1, nrec)``.  The forward field is propagated again alongside.
        ``limit`` stops after that many steps (to compile the sweep before timing it)."""
        nb = len(directions)
        dM = self.mass_perturbations(directions)
        zero = lambda: jax.device_put(jnp.zeros((self.nloc, 1 + nb), dtype=self.dtype), self.device)
        if limit is not None:
            self._run(zero(), zero(), range(int(limit)), dM=dM, record=True)
            return None
        _, _, out, _ = self._run(zero(), zero(), range(self.nsteps), dM=dM, record=True)
        return self._gather_records(out)[1:]

    def adjoint(self, q, final=None, limit=None):
        """The action of the transposed sensitivity on receiver series ``q``
        (``(nq, nsteps + 1, nrec)``): one parameter-space dual vector per series.

        The forward field is rebuilt backward from its last two states while the adjoint
        fields are carried backward with it.  ``limit`` stops the sweep after that many
        steps (to compile it before timing it)."""
        self._need_sensitivities()
        final = final if final is not None else self._final
        if final is None:
            raise RuntimeError("no forward solve to reverse: call forward() first")
        q = np.asarray(q, dtype=np.float64)
        nb, N = q.shape[0], self.nsteps
        rec = self.receivers
        # field 0 runs the forward recurrence backward: at sweep step j (N .. 1) it applies
        # K to u_{j-1} with the source of step j-1; the adjoint fields take q[:, j]
        qloc = self._put(q[:, :, rec.index]) if rec.nlocal else jnp.zeros((nb, N + 1, 0), dtype=self.dtype)
        zb = jnp.zeros((self.nloc, nb), dtype=self.dtype)
        U = jnp.concatenate([final[1].reshape(self.nloc, 1), zb], axis=1)
        Um = jnp.concatenate([final[0].reshape(self.nloc, 1), zb], axis=1)
        del zb
        # the source index of the main field at sweep step j is j-1, the adjoint's is j:
        # shift q by one so that both read index j-1
        qshift = jnp.concatenate([qloc[:, 1:], jnp.zeros_like(qloc[:, :1])], axis=1)
        last = -1 if limit is None else max(N - 1 - int(limit), -1)
        _, _, _, S = self._run(U, Um, range(N - 1, last, -1), q=qshift, correlate=True)
        def to_parameters(S, wdkaps, idxs, Nm, edofss):
            loc = jnp.zeros((self.nloc_m, S.shape[1]), dtype=S.dtype)
            for wdkap, idx, edofs in zip(wdkaps, idxs, edofss):
                loc = loc.at[edofs.reshape(-1)].add(
                    jnp.einsum("qa,eqb->eab", Nm, wdkap[:, :, None] * S[idx]).reshape(-1, S.shape[1]))
            return loc

        fold = self._get("fold", lambda: jax.jit(to_parameters))
        loc = np.asarray(fold(S, self._wdkap, self._idx, self._Nm, self._edofs_m), dtype=np.float64)
        return [self.Vm.assemble_dual(np.ascontiguousarray(loc[:, b])) for b in range(nb)]

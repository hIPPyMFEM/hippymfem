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

The update needs the other ranks' contributions at the shared dofs alone, so where
they travel from card to card, and a rank is large enough for its update to last as
long as they take, the exchange of a step is made *behind* its update
(``HIPPYMFEM_SEM_OVERLAP``, :data:`OVERLAP`, :data:`OVERLAP_MIN`): the update of
every row is launched with this rank's own contributions, the neighbours' arrive
while the card computes it, and the rows it could not know (the shared dofs, and the
absorbing nodes with them: a block's faces, a hundredth of a rank of 192 million
unknowns) are then computed again from their old values and the summed residual, by
the same formulas and the same ordered sum, so that every copy of a shared dof still
holds the same bits.  What that is for, on Frontier's MI250X GCDs: a step of 192
million unknowns a rank took 18.67 ms on one GCD, 19.92 on eight and 20.18 on 32
with the exchange between the two halves of the step (51.6, 53.3 and 54.1 ms with
two fields), and all of the loss was the exchange (0.9 to 1.4 ms a step inside MPI,
with 7 to 17 neighbours) and the two compiled calls around it.  There an exchange of
8 MB with 7 neighbours takes 0.44 ms alone, an update of 150 million doubles 5.54
ms, the two one after the other 6.57 ms, and 5.60 ms with the update launched, the
exchange made and the update waited for (32 MB: 1.55, 5.55, 7.77 and 5.78 ms): an
exchange between cards runs while the card computes.

A boundary that lets the waves out (``absorbing=``) adds a diagonal boundary mass
:math:`C` at its nodes, centred in time,

.. math:: M(u_{n+1} - 2u_n + u_{n-1}) + \tfrac{\Delta t}{2} C\,(u_{n+1} - u_{n-1})
          + \Delta t^2 K u_n = \Delta t^2 f_n ,

which is stable under the same limit and loses energy only through :math:`C`.  It is
no longer reversible at those nodes: solved for :math:`u_{n-1}` it divides by
:math:`1 - \beta`, :math:`\beta = \Delta t\,C / (2M)`, which is 0.64 on a face at 0.8
of the limit and passes through 1 on an edge of two faces, and an error there grows
:math:`(1+\beta)/(1-\beta)` times a step.  So the forward sweep keeps the field's
values at the absorbing nodes, step by step (:class:`BoundaryHistory`), and the
backward sweep takes them from it; everywhere else the field is rebuilt by the
recurrence as before, now the interior's own with given boundary data.

Three things serve a mesh of equal bricks, each off unless asked for: one element's
geometric factors in the place of every node's (``brick=``), the unknowns numbered in
the order of their keys (``numbering="keys"``), and with both a stiffness that is
applied to a rank's block of unknowns as a 3D array, with no table of indices
(``lattice=``).
"""

import os
import sys
import time
import traceback
import weakref

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from hippymfem.fem.elementbatch import _basis_tables, _element_dofs
from hippymfem.fem import kernel as kernel_mod
from hippymfem.fem.kernel import Field

import jax
import jax.numpy as jnp

#: Whether the exchange between ranks goes from and into the cards' own memory, with no
#: copy through the host (:meth:`Interface.swap_device`): the MPI library is then handed
#: addresses of device memory and has to know them for what they are.  ``auto``: where
#: Cray MPICH runs with its GPU support (``MPICH_GPU_SUPPORT_ENABLED=1``, which it refuses
#: to start with unless its transport library for the cards is loaded; ``MPI_Init`` then
#: starts AMD's runtime, and a program that imports mpi4py before hippymfem has to have
#: ``ROCPROFILER_REGISTER_ENABLED=0`` in its environment, see ``hippymfem/_jaxconfig.py``:
#: a rank without neighbours waits for nothing in a sweep and would fill the launch
#: queue).  On Frontier the
#: way through the host costs a step of 211 million unknowns a rank some 11 ms of its
#: 171 (22 MB to the host, through MPI and back to the card, about 3.6 ms each); from
#: card to card it is about a millisecond.  ``HIPPYMFEM_DEVICE_MPI`` (``1``, ``0`` or
#: ``auto``).
DEVICE_MPI = os.environ.get("HIPPYMFEM_DEVICE_MPI", "auto").strip().lower() or "auto"

#: MiB of the pieces in which the values a rank sends are brought from the device when
#: they are more than two such pieces.  JAX makes a new host array for every array it
#: brings from a card, and glibc serves an array above 32 MiB by a new mapping of memory
#: each time: 20 to 35 ms for 34 to 57 MB on an EPYC 7763 where a kept array takes 2 to
#: 3, and two fields of a rank of 200 million unknowns are that large from 64 ranks on.
#: Pieces below that size, brought one after the other into an array that is kept, each
#: take the block the one before left.  ``auto``: 16 on a card, and one array on the
#: host, where JAX hands out the device array's own memory.  ``HIPPYMFEM_SEM_PIECE``
#: (MiB, ``0`` for one array always, or ``auto``).
HOST_PIECE = os.environ.get("HIPPYMFEM_SEM_PIECE", "auto").strip().lower() or "auto"

#: Whether what a step hands to its compiled calls from host memory (the step index, the
#: values received from the other ranks) is put on the device and waited for before the
#: call.  It is otherwise copied there while the call is launched, the pattern at which
#: ROCm 7.2's runtime was seen to fail in another program of this library; whether it
#: fails here is not known.  ``HIPPYMFEM_SEM_PUT`` (``1`` or ``0``).
PUT_FIRST = os.environ.get("HIPPYMFEM_SEM_PUT", "0").strip().lower() in ("1", "yes", "true", "on")

#: Whether a step on several ranks makes its exchange behind its update: the update of
#: every row is launched with the rank's own residual and not waited for, the exchange is
#: made, and the rows that needed the neighbours' values are computed again
#: (:meth:`SpectralWave._build_step`; the module text has the measurements it rests on).
#: ``auto``: where the exchange goes from card to card (:data:`DEVICE_MPI`), which runs
#: while the card computes, on ranks large enough for their update to hide it
#: (:data:`OVERLAP_MIN`).  ``1``: always, through host arrays as well, where less is to
#: be had: the values go to the host before the update is launched (on ROCm a copy to
#: the host waits for every kernel in the queue) and to the card after it is done, so
#: only the time inside MPI hides behind it; it is how the step is tested without
#: cards.  ``0``: never, which is the step as it was before, with the bits it had.  (On
#: the host the overlapped step computes those very bits, in every configuration of
#: ranks, faces, kernels and precisions that was tried; on a card it agrees with the
#: other to round-off, as two runs of one step do there.)  A rank that is alone, or
#: shares nothing, steps in one call either way.  The same on every rank of a run, as
#: the environment is.  ``HIPPYMFEM_SEM_OVERLAP`` (``1``, ``0`` or ``auto``).
OVERLAP = os.environ.get("HIPPYMFEM_SEM_OVERLAP", "auto").strip().lower() or "auto"

#: The unknowns a rank has to hold on average, times the fields of a sweep, for ``auto``
#: to overlap its steps (:data:`OVERLAP`); ``0`` for every size.  The overlapped step is
#: three compiled calls where the other is two, and what it hides is as long as the
#: shorter of the exchange and the update.  On an MI250X the update takes 0.037 ms a
#: million unknowns of a field (5.54 ms for 150 million), and an exchange with 7
#: neighbours 0.2 ms for ranks of 0.9 million unknowns, whose whole step is 0.8 ms:
#: there the update is over in 0.03 ms, nothing hides behind it, and the third call is
#: paid for in full.  What it costs was measured on an MI210 alone (one rank whose
#: neighbour is not there and sends nothing, so nothing is hidden): the host spends 0.13
#: to 0.22 ms a step launching the third call at 3 million unknowns, 0.2 to 0.4 in each
#: of the other two, and the overlapped step came out 0.1 to 0.5 ms a step slower than
#: the other in most of 15 sweeps at 0.9 to 24 million unknowns (from 0.15 faster to 1.5
#: slower: the node and its card are shared).  The default asks for an update of 0.6 ms.
#: It is an estimate, not a measurement on real ranks, which is where it has to be
#: settled.  ``HIPPYMFEM_SEM_OVERLAP_MIN`` (a number of unknowns).
OVERLAP_MIN = float(os.environ.get("HIPPYMFEM_SEM_OVERLAP_MIN", "").strip() or "16e6")


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
    keys : ndarray of int64, ``(nloc,)`` (:meth:`from_keys` only)
        The key of every local dof.
    brick : ndarray ``(3,)`` or None
        The sides of the one brick every element is (:meth:`from_keys`).
    numbering : str
        How the local dofs are numbered: ``"space"`` (MFEM's), ``"touch"`` or ``"keys"``.
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
        self.brick, self.numbering = None, "space"
        self.comm = space.comm
        self._boxes = None
        self._tab = None
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
    def from_keys(cls, Vq, order, q, fn, node_keys, brick=None, numbering="touch"):
        """The tables of a degree ``order`` discretization of a structured mesh that
        MFEM holds only as a straight trilinear mesh.

        No MFEM space of degree ``order`` is built.  ``node_keys(qv, W) -> (keys,
        boundary)`` names the nodes: given the element vertices before the map,
        ``(ne, 8, 3)``, and the trilinear weights ``W`` (``(n^3, 8)``) of the vertices at
        the equispaced points of the reference cube, it returns an integer key for each
        of the ``n^3`` nodes of each element, the same for a node whichever element or
        rank sees it, and marks those on the boundary.  Local dofs are the distinct
        keys, numbered in the order the elements first name them (:func:`first_touch`),
        and ranks find what they share by the keys (see :meth:`Interface.from_keys`),
        so the number of unknowns is not bounded by MFEM's global numbering.  ``Vq``,
        ``q`` and ``fn`` are as in ``mapped``.

        ``brick=(hx, hy, hz)`` says that every element is the same brick with those
        sides along the axes, its reference axes along ``x``, ``y`` and ``z`` (``fn`` the
        identity, or a shift).  The geometric factors are then one element's, known in
        closed form (:meth:`brick_factors`), and nothing of them is computed or kept
        per element: half of what a propagation holds on the device otherwise.

        ``numbering="keys"`` numbers the local dofs in the order of their keys instead.
        Where the keys count the nodes of a lattice and a rank holds a whole block of
        it, the rank's unknowns are then that block as a 3D array, which a stiffness
        without a table of indices needs (``SpectralWave(lattice=...)``); a stiffness
        with a table is slower for it (:func:`first_touch`).
        """
        if numbering not in ("touch", "keys"):
            raise ValueError("numbering is 'touch' or 'keys', not %r" % (numbering,))
        self = cls.__new__(cls)
        self.numbering = numbering
        self.brick = None if brick is None else np.asarray(brick, dtype=np.float64).reshape(3)
        mesh = Vq.mesh
        self.space, self.comm = None, Vq.comm
        self.ne = ne = mesh.GetNE()
        self.p = p = int(order)
        self.n = n = p + 1
        self.nodes, self.w1, self.D = gll(p)
        self._reference()
        self.mapped, self.keyed, self._boxes, self._geom, self._tab = True, True, None, None, None
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
        self.nloc = int(uniq.size)
        idx = np.ascontiguousarray(inv.reshape(ne, n ** 3)).astype(np.int32, copy=False)
        del inv
        # a node can be shared only if the vertices it hangs on are all shared, so only
        # the elements with a shared vertex are looked at (those at the rank's surface)
        vi = Interface(Vq)
        shared_v = np.zeros(Vq.fes.GetVSize(), dtype=bool)
        shared_v[vi.iface] = True
        hangs = W > 1e-12                                              # (n^3, 8)
        sv = shared_v[edofs]
        es = np.nonzero(sv.any(axis=1))[0]
        cand_e = ((~sv[es]).astype(np.int8) @ hangs.T.astype(np.int8)) == 0 if ne else np.zeros((0, n ** 3), bool)
        cand = np.zeros(self.nloc, dtype=bool)
        cand[idx[es][cand_e]] = True
        new = first_touch(idx, self.nloc) if numbering == "touch" else np.arange(self.nloc, dtype=np.int32)
        self.idx = new[idx]
        del idx
        self.keys = np.empty_like(uniq)                                # the key of every local dof
        self.keys[new] = uniq
        self.ess = np.zeros(self.nloc, dtype=bool)
        self.ess[self.idx[boundary]] = True
        self.iface = Interface.from_keys(self.comm, uniq, cand, new)
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

    def brick_factors(self):
        """One element's geometric factors on a mesh of equal bricks (``brick``), which
        are every element's: ``(g, wdet)`` on the host, ``g`` the three factors that are
        not zero, ``w |J| / h_k^2`` for ``xx, yy, zz``, ``(3, n^3)``, and ``wdet = w |J|``,
        ``(n^3,)``.

        In closed form, and not from the map: the Jacobian of a brick by differentiation
        of its nodes' coordinates has off-diagonal entries of the coordinates' round-off
        over the element's size, not zero, and the factors of two elements then differ
        in their last digits."""
        hx, hy, hz = self.brick
        det = hx * hy * hz
        return np.stack([self.w * (det / h ** 2) for h in (hx, hy, hz)]), self.w * det

    def block(self, a, b):
        """``(X, G, wdet)`` of elements ``a:b`` on the device: node positions
        ``(b - a, n^3, 3)``, the six geometric factors ``(6, b - a, n^3)`` and
        ``w |J|``, ``(b - a, n^3)``.  On a mesh of equal bricks ``G`` is None: the
        factors are :meth:`brick_factors`."""
        dev = kernel_mod.device()
        if self.brick is not None:
            wd = jax.device_put(jnp.asarray(self.brick_factors()[1]), dev)
            return self.block_nodes(a, b), None, jnp.broadcast_to(wd[None, :], (b - a, self.n ** 3))
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
        return np.asarray(nodes(jax.device_put(jnp.asarray(self._qv[e:e + 1]), kernel_mod.device())))[0]

    def boxes(self, block=65536):
        """Bounding boxes of the elements' nodes, ``(lo, hi)``, ``(ne, 3)`` each."""
        if self._boxes is None:
            if not self.mapped:
                self._boxes = (self.X.min(axis=1), self.X.max(axis=1))
            else:
                nodes, _ = self._mapped_geometry()
                dev = kernel_mod.device()
                lo, hi = np.empty((self.ne, 3)), np.empty((self.ne, 3))
                for a in range(0, self.ne, block):
                    b = min(a + block, self.ne)
                    X = nodes(jax.device_put(jnp.asarray(self._qv[a:b]), dev))
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
        # The tables of the space asked for last are kept: the propagator asks for the
        # parameter's, which a mapped or keyed geometry has read as its vertices' already
        # (reading MFEM's table again takes 1.2 to 1.5 s at eight million elements, 3 s at fifteen).
        if self._tab is not None and self._tab[0] is other:
            return self._tab[1:]
        fe = fes.GetFE(0)
        nd = fe.GetDof()
        N, _ = _basis_tables(fe, self.rule, 3, nd)
        edofs, _ = _element_dofs(fes, np.arange(self.ne), 1, nd)
        self._tab = (other, N, edofs.astype(np.int32))
        return self._tab[1:]


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
    # What goes to the card is waited for before a compiled call takes it.  Of 4096 ranks
    # two ended in a segmentation fault inside the compilation of ``find`` below, a
    # gigabyte of keys on its way to the card beside it (MI250X, ROCm 7.2; 15 million
    # elements a rank, where runs with half as many had passed); a copy still under way
    # beside a launch is where that runtime was seen to fail in this library before.
    put = lambda x: jax.block_until_ready(jax.device_put(x, dev))
    s = np.asarray(jnp.sort(put(keys)))
    new = np.empty(s.size, dtype=bool)
    new[0] = True
    np.not_equal(s[1:], s[:-1], out=new[1:])
    uniq = s[new]
    del s, new
    ud = put(uniq)
    find = jax.jit(lambda u, k: jnp.searchsorted(u, k).astype(jnp.int32))
    inv = np.empty(keys.size, dtype=np.int32)
    for a in range(0, keys.size, chunk):
        inv[a:a + chunk] = np.asarray(find(ud, put(keys[a:a + chunk])))
    del ud
    return uniq, inv


def first_touch(idx, n, chunk=1 << 27):
    """The ``n`` numbers that ``idx`` holds (each at least once) in the order in which
    ``idx``, read row by row, first names them: ``new``, with ``new[old]`` the new number.

    A mesh's elements come in an order that keeps neighbours near each other, and their
    unknowns numbered in the order the elements meet them lie near each other in memory
    as well, which is what the gather and the scatter of every step read and write.  On
    an MI210 a step of 27 million unknowns numbered by their keys takes 22.4 ms and
    numbered so 21.3 (two fields 26.3 and 24.6); numbered at random it takes 44.  The
    ranks of a run, whose pieces of the mesh lie differently across the keys, also differ
    less from each other.  On a device when there is one, as the sort of the keys is."""
    flat = np.ascontiguousarray(idx).reshape(-1)
    if flat.size < (1 << 24) or not kernel_mod.on_gpu():
        first = np.empty(n, dtype=np.int64)
        first[flat[::-1]] = np.arange(flat.size - 1, -1, -1)       # of several writes the last stays: the first place
        order = np.argsort(first, kind="stable")
    else:
        dev = kernel_mod.device()
        low = jax.jit(lambda f, i, a: f.at[i].min(a + jnp.arange(i.shape[0], dtype=jnp.int64)), donate_argnums=0)
        first = jax.device_put(jnp.full((n,), flat.size, dtype=jnp.int64), dev)
        for a in range(0, flat.size, chunk):
            # (waited for before the call takes it, as in unique_inverse)
            first = low(first, jax.block_until_ready(jax.device_put(flat[a:a + chunk], dev)), a)
        order = np.asarray(jnp.argsort(first))
        del first
    new = np.empty(n, dtype=np.int32)
    new[order] = np.arange(n, dtype=np.int32)
    return new


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
    def from_keys(cls, comm, keys, candidates, local=None):
        """The interface of dofs named by integer ``keys`` (sorted, one per local dof).
        ``local`` gives the local dof of each key where the dofs are not numbered in the
        keys' order.

        ``candidates`` marks the keys that may be held by another rank (those
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
        self._finish(got, keys, np.arange(nloc) if local is None else local)
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
        self._kept, self._turn = {}, {}
        return send, order

    def kept(self, use, shape, dtype):
        """The host array of ``shape`` and ``dtype`` that is kept for ``use``.  An exchange
        that made its arrays anew would, above 32 MiB, map new memory every time."""
        key = (use, tuple(shape), np.dtype(dtype).str)
        if key not in self._kept:
            self._kept[key] = np.empty(shape, dtype)
        return self._kept[key]

    def post(self, shape, dtype):
        """Post the receives of an exchange of ``shape`` (``(nsend, nf)``) values of
        ``dtype``, which can be done before this rank's own values exist: the
        neighbours' are then written as they arrive.  :meth:`swap` takes what this
        returns.

        The values arrive in one of two kept arrays of the shape, taken in turn: the
        device may still be reading the last exchange's when these receives are posted,
        but no longer those of the exchange before it."""
        key = (tuple(shape), np.dtype(dtype).str)
        turn = self._turn[key] = 1 - self._turn.get(key, 0)
        recv = self.kept(("receive", turn), shape, dtype)
        reqs = []
        for r in self.neighbors:
            a, b = self._slices[r]
            reqs.append(self.comm.Irecv(recv[a:b], source=r, tag=71))
        return recv, reqs

    def swap(self, send, posted=None):
        """Send the rows of ``send`` (``(nsend, nf)``, laid out as :meth:`layout` says) to
        the neighbours and return theirs in the same layout, in the precision of
        ``send``.  ``posted`` is what :meth:`post` returned if the receives were posted
        ahead.  The array returned is a kept one: it holds these values until the
        exchange after the next one of this shape."""
        send = np.ascontiguousarray(send)
        recv, reqs = posted if posted is not None else self.post(send.shape, send.dtype)
        for r in self.neighbors:
            a, b = self._slices[r]
            reqs.append(self.comm.Isend(send[a:b], dest=r, tag=71))
        MPI.Request.Waitall(reqs)
        return recv

    def swap_device(self, send, recv):
        """:meth:`swap` between the cards' own memory: ``send`` and ``recv`` are device
        arrays of the layout's shape and one precision, and MPI is handed their addresses
        (see :data:`DEVICE_MPI`).  The neighbours' rows are written into ``recv`` behind
        JAX's back, so it has to be an array of the caller's own that nothing computed is
        still to be read from; it is returned."""
        comm = self.comm
        jax.block_until_ready((send, recv))
        row = send.dtype.itemsize * int(np.prod(send.shape[1:], dtype=np.int64))
        sp, rp = send.unsafe_buffer_pointer(), recv.unsafe_buffer_pointer()
        typ = MPI._typedict[send.dtype.char]
        at = getattr(MPI, "buffer", None) or MPI.memory
        reqs = []
        for r in self.neighbors:
            a, b = self._slices[r]
            reqs.append(comm.Irecv([at.fromaddress(rp + a * row, (b - a) * row), typ], source=r, tag=71))
        for r in self.neighbors:
            a, b = self._slices[r]
            reqs.append(comm.Isend([at.fromaddress(sp + a * row, (b - a) * row), typ], dest=r, tag=71))
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

    Returns ``(rows, elems, xi)`` for the points this rank is responsible for.  ``xi``
    are reference coordinates in ``[0, 1]^3``, clipped: a point of the exact boundary
    lies a little outside the polynomial boundary of a coarse mesh, so a point up to
    ``tol`` outside an element (in reference units) is taken to be on its face, and
    among several candidates the one it is least outside of.  That holds across the
    ranks too: a point goes to the rank whose element it is least outside of, and to
    the lowest of several such, so each is handled once and by the element one rank
    alone would give it (the lowest rank that found it within ``tol`` took it before,
    also where another held the element it lies in: a receiver near a cut was then
    read up to ``tol`` of an element away from its place).
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    npt = points.shape[0]
    n, nodes = col.n, col.nodes
    found = np.full(npt, -1, dtype=np.int64)
    xis = np.zeros((npt, 3))
    far = np.full(npt, np.inf)                       # how far outside its element a point found here lies
    if col.ne:
        lo, hi = col.boxes()
        # The box of all the rank's elements comes first, with a pad no smaller than any
        # element's below.  Most ranks hold none of the points, and for a point outside
        # that box nothing more is done: no padded boxes, and no pass over the elements,
        # which is a quarter of a second a point at eight million of them.  (Column by
        # column: numpy reduces an (ne, 3) array along its long axis ten times more slowly.)
        glo = np.array([lo[:, d].min() for d in range(3)])
        ghi = np.array([hi[:, d].max() for d in range(3)])
        gpad = 0.05 * (ghi - glo).max() + 1e-12
        near = np.nonzero(np.all((points >= glo - gpad) & (points <= ghi + gpad), axis=1))[0]
        if near.size:
            # for a point, one coordinate against every element with the largest pad of
            # them all (the coordinate the rank is longest in, where the fewest pass),
            # then the box with its own pad for those
            ax = int(np.argmax(ghi - glo))
            wide = 0.05 * (hi - lo).max() + 1e-12
            loa, hia = lo[:, ax] - wide, hi[:, ax] + wide
        for ip in near:
            x = points[ip]
            cand = np.nonzero((x[ax] >= loa) & (x[ax] <= hia))[0]
            pad = 0.05 * (hi[cand] - lo[cand]).max(axis=1, keepdims=True) + 1e-12
            cand = cand[np.all((x >= lo[cand] - pad) & (x <= hi[cand] + pad), axis=1)]
            best = None
            for e in cand:
                Xe = col.element_nodes(e).reshape(n, n, n, 3)    # [c, b, a]
                xi = np.full(3, 0.5)
                ok, last = False, np.inf
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
                    step = np.abs(dxi).max()
                    # Converged: the increment is negligible, or it has stopped falling.  It
                    # stops at the round-off of the positions over the element's size, 1e-12
                    # km over 5 km on a fine mesh of the Earth, which no fixed bound is
                    # under at every size.  Newton squares an increment, so one below 1e-8
                    # that is not followed by a smaller one has reached that floor.
                    if step < 1e-13 or (last < 1e-8 and step >= last):
                        ok = True
                        break
                    last = step
                if ok and xi.min() >= -tol and xi.max() <= 1.0 + tol:
                    out = np.abs(np.clip(xi, 0.0, 1.0) - xi).max()
                    if best is None or out < best[0]:
                        best = (out, int(e), np.clip(xi, 0.0, 1.0))
            if best is not None:
                far[ip], found[ip], xis[ip] = best
    # (MPI's pair of a double and an int, with C's padding between two of them)
    have = np.empty(npt, dtype=np.dtype([("out", np.float64), ("rank", np.intc)], align=True))
    have["out"], have["rank"] = far, comm.rank
    owner = np.empty_like(have)
    comm.Allreduce([have, MPI.DOUBLE_INT], [owner, MPI.DOUBLE_INT], op=MPI.MINLOC)
    if not np.isfinite(owner["out"]).all():
        raise ValueError("point(s) %s lie in no element" % np.nonzero(~np.isfinite(owner["out"]))[0].tolist())
    mine = np.nonzero(owner["rank"] == comm.rank)[0]
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
class BoundaryHistory:
    """The main field at the absorbing nodes after every step of a forward sweep,
    ``u_1 .. u_N`` there, for the backward sweep that rebuilds the field.

    Only the main field is ever rebuilt, so this is one field's values whatever the
    number of fields in a sweep: ``nab * nsteps`` numbers of the fields' precision
    (3.4 MB a step for the 430 thousand nodes on five faces of 26 million unknowns).
    The rows of ``piece`` steps make one array ``(piece, nab)``, which stays on the
    device (``where="device"``) or is brought to the host when it is full (``"host"``)
    and put back when the backward sweep reaches it: that sweep asks for the rows in
    descending order, so one piece at a time is on the device.

    To the host a piece goes in parts of 16 MiB at most, into an array that an earlier
    history has given up where there is one (``pool``, a list the histories of one
    propagator share: a history that is no longer referred to leaves its arrays
    there).  A new host array is new memory, which the system maps as it is first
    written, and JAX makes a new one of whatever it brings from a card: 53 MB brought
    as one array take 56 ms on an EPYC 7763 beside an MI210, and 12 ms so.  The first
    sweep of a propagator still pays for the memory, the sweeps after it (the next
    shot's) do not: with 414 thousand absorbing nodes, 3.3 MB a step, a forward step
    of 24 million unknowns takes 2.3 ms with the values on the device, 6 ms in the
    first sweep with them on the host and 3.0 ms in the later ones.
    """

    def __init__(self, nab, dtype, device, where="device", piece=64, pool=None):
        if where not in ("device", "host"):
            raise ValueError("the boundary values are kept on the 'device' or the 'host', not %r" % (where,))
        self.nab, self.dtype, self.device = int(nab), dtype, device
        self.where, self.piece = where, max(1, int(piece))
        self.pieces, self.nrows = [], 0
        self._open = []                                  # the rows of the piece that is not full yet
        self._zero = jax.device_put(jnp.zeros((self.nab,), dtype=dtype), device)
        self._loaded = (-1, None)                        # the piece on the device for row(), and its number
        self._pool = pool if pool is not None else []
        self._host = []                                  # this history's host arrays, the pool's after it
        weakref.finalize(self, self._pool.extend, self._host)

    def append(self, row):
        """Take the values after the next step, a device array ``(nab,)``."""
        self._open.append(row)
        self.nrows += 1
        if len(self._open) == self.piece:
            self.close()

    def close(self):
        """Make a piece of the rows taken since the last one (the sweep has ended)."""
        if self._open and self.where == "device":
            self.pieces.append(jnp.stack(self._open))
        elif self._open:
            shape, dtype = (self.piece, self.nab), np.dtype(self.dtype)
            fits = [i for i, x in enumerate(self._pool) if x.shape == shape and x.dtype == dtype]
            out = self._pool.pop(fits[-1]) if fits else np.empty(shape, dtype)
            rows = jax.block_until_ready(jnp.stack(self._open))
            # in parts of 16 MiB at most (as SpectralWave._to_host brings what a rank sends): the
            # host array JAX makes of a part is given up before the next, which takes its block
            n = rows.shape[0]
            m = max(1, (1 << 24) // max(1, self.nab * dtype.itemsize))
            for a in range(0, n, m):
                out[a:min(a + m, n)] = np.asarray(rows[a:a + m])
            self._host.append(out)
            self.pieces.append(out[:n])
        self._open = []
        return self

    def row(self, k):
        """``u_k`` at the absorbing nodes, on the device; zero for ``k < 1``, the rest
        the forward sweep starts from."""
        if k < 1:
            return self._zero
        j, r = divmod(k - 1, self.piece)
        if self._loaded[0] != j:
            self._loaded = (-1, None)                    # the piece before is given up first
            # (from the host it is waited for before a compiled call takes a row of it, as in
            # unique_inverse: tens to hundreds of MB on their way to the card)
            self._loaded = (j, jax.block_until_ready(jax.device_put(self.pieces[j], self.device)))
        return self._loaded[1][r]

    @property
    def nbytes(self):
        return self.nrows * self.nab * np.dtype(self.dtype).itemsize


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
    absorbing : (dofs, C), optional
        The nodes of a boundary that lets the waves out: their local dof numbers and
        the diagonal boundary mass there, the integral of ``1 / (rho c)`` of a
        reference medium over the boundary against the node's basis function.  For a
        node several ranks hold, each gives the whole number, not its part (so nothing
        is exchanged for it), and every rank gives the argument, with no nodes where it
        has none.  ``C`` must not depend on the parameter: the derivatives of the
        scheme are taken with it fixed.  Nodes where the field is zero are left out.
    history : "device" or "host"
        Where a forward sweep keeps the field's values at the absorbing nodes
        (:class:`BoundaryHistory`), ``history_piece`` steps to an array.
    lattice : (ex, ey, ez), optional
        This rank's elements are a block of that many equal bricks along ``x``, ``y``
        and ``z``, with one stiffness coefficient, and its unknowns are the block's
        nodes in the order of a 3D array, ``z`` fastest (``Collocation.from_keys`` with
        ``brick=`` and ``numbering="keys"``, the keys counting the nodes so).  The
        stiffness is then applied through strided slices of that array
        (:meth:`_lattice_stiff_fn`), with no table of indices.  On an MI210 a step of 7
        million unknowns in double precision then takes 0.9 ms for one field, where the
        kernel with a table takes 2.7 with one brick's factors and 5.9 with every
        node's; 2.1 ms for two fields (4.5 and 7.3) and 12.6 for eight (16.3 with one
        brick's factors): what it saves, it saves once a field, and the fields of a
        batch cost it more each than one alone (their index is the array's last).

    Several fields are propagated at once, stored ``(nloc, nfields)`` with the field
    index last: the geometric factors and the gather and scatter indices are then
    read once for all of them, so nine fields cost six to eight times one, not nine.
    """

    def __init__(self, Vu, Vm, density, aux=(), ess_bdr="all", dt=None, nsteps=None,
                 cfl=0.8, dtype=jnp.float64, element_chunk=10000, col=None,
                 geometry_dtype=None, param_chunk=65536, element_block=0, sensitivities=True,
                 absorbing=None, history="device", history_piece=64, lattice=None):
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
        # (gigabytes on their way to the card: waited for before anything is compiled beside
        # them, as in unique_inverse)
        jax.block_until_ready((self._edofs_m, self._idx))
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
        self._lattice = None if lattice is None else self._block_of_bricks(lattice)
        # the absorbing nodes in ascending order, without those where the field is zero (the
        # rim of a free surface, say)
        if history not in ("device", "host"):
            raise ValueError("history is 'device' or 'host', not %r" % (history,))
        self.history, self.history_piece = history, int(history_piece)
        self._ab = absorbing is not None
        if self._ab:
            dofs = np.asarray(absorbing[0], dtype=np.int64).reshape(-1)
            C = np.asarray(absorbing[1], dtype=np.float64).reshape(-1)
            if dofs.size != C.size or (C < 0.0).any():
                raise ValueError("absorbing=(dofs, C): a boundary mass, not negative, for every node")
            keep = ~ess[dofs]
            order = np.argsort(dofs[keep], kind="stable")
            self.ab_dofs, self._ab_C = dofs[keep][order], C[keep][order]
            if np.any(self.ab_dofs[1:] == self.ab_dofs[:-1]):
                raise ValueError("an absorbing node is listed twice")
            self._ab_idx = put(self.ab_dofs, jnp.int32)
            self._ab_gain = None         # (dt, beta / (1 + beta), beta) on the device, see _consts
            self._history_pool = []      # host arrays that histories of this propagator have given up
        #: True makes :meth:`adjoint` keep in ``rebuilt_rest`` the two states its main field
        #: ends on, device arrays ``(nloc,)``.  After all the steps they are the rest the
        #: forward sweep started from: zero, as well as the field was rebuilt (a check).
        self.keep_rest = False
        self.rebuilt_rest = None
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
        self.t_exchange = 0.0            # of which, in the exchange between ranks (inside MPI)
        self.t_element = 0.0             # of which, waiting for the device at the element work
        # of which, bringing the values a rank sends to the host (the copy of those it receives
        # to the device runs beside the next call, and is counted only with PUT_FIRST)
        self.t_copy = 0.0
        self._recv = {}                  # what the neighbours send, on the device (DEVICE_MPI)
        self._rows = None                # the rows an overlapped step computes again (OVERLAP), see _rows_tables

    def _block_of_bricks(self, lattice):
        """``lattice`` as three integers, once this rank's elements and unknowns are seen to
        be what the stiffness without a table takes them for: with ``N_d = p e_d + 1`` nodes
        along axis ``d``, the unknown ``(I N_1 + J) N_2 + K`` is the node ``(I, J, K)`` of the
        block, and the elements are its ``e_0 e_1 e_2`` cubes of ``p`` steps, each with its
        reference axes along the block's.  Read from the table of the elements' unknowns,
        which the other kernels go by: the two then apply the same matrix."""
        col, p, n = self.col, self.col.p, self.n
        e3 = tuple(int(v) for v in lattice)
        if col.brick is None or col.numbering != "keys":
            raise ValueError("lattice= needs a mesh of equal bricks with the unknowns numbered by their "
                             "keys (Collocation.from_keys with brick= and numbering='keys')")
        N3 = tuple(p * e + 1 for e in e3)
        if len(e3) != 3 or min(e3) < 1 or int(np.prod(N3)) != self.nloc or int(np.prod(e3)) != self.ne:
            raise ValueError("this rank's %d elements and %d unknowns are not a block of %s elements"
                             % (self.ne, self.nloc, "x".join(str(e) for e in e3)))
        q = np.arange(n ** 3)
        # where an element's nodes are from its first
        offsets = ((q % n) * N3[1] + (q // n) % n) * N3[2] + q // (n * n)
        seen = np.zeros(e3, dtype=bool)
        chunk = 1 << 18
        for a in range(0, self.ne, chunk):
            idx = col.idx[a:a + chunk]
            first = idx[:, 0].astype(np.int64)
            at = np.stack([first // (N3[1] * N3[2]), (first // N3[2]) % N3[1], first % N3[2]], axis=1)
            if (idx - idx[:, :1] != offsets).any() or (at % p).any() or (at // p >= e3).any():
                raise ValueError("lattice=: an element's unknowns are not a cube of the block's nodes "
                                 "along its axes")
            seen[tuple((at // p).T)] = True
        if not seen.all():
            raise ValueError("lattice=: the elements do not fill the block")
        return e3

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

        The factors' own shape says what they are: ``(6, ne, n^3)``, those of every node;
        ``(3, ne, n^3)``, those of bricks along the axes, which have no mixed entries;
        ``(3, 1, n^3)``, one brick's for all the elements, which is then not cut per
        pass but broadcast (on an MI210 a step of one field takes 0.45 of the time
        with it, of two fields 0.6).
        """
        if self._lattice is not None:
            return self._lattice_stiff_fn()
        n, nq = self.n, self.nq
        work = self.element_chunk

        def element(ue, g, D):
            # ue [e, c, b, a, f], g [6 or 3, e or 1, c, b, a, 1] -> the elements' K u, same shape as ue
            Dt = D.T
            ux = sum(ue[:, :, :, i:i + 1] * D[:, i][:, None] for i in range(n))
            uy = sum(ue[:, :, i:i + 1] * D[:, i][:, None, None] for i in range(n))
            uz = sum(ue[:, i:i + 1] * D[:, i][:, None, None, None] for i in range(n))
            if g.shape[0] == 3:
                fx, fy, fz = g[0] * ux, g[1] * uy, g[2] * uz
            else:
                fx = g[0] * ux + g[1] * uy + g[2] * uz
                fy = g[1] * ux + g[3] * uy + g[4] * uz
                fz = g[2] * ux + g[4] * uy + g[5] * uz
            return (sum(fx[:, :, :, i:i + 1] * Dt[:, i][:, None] for i in range(n))
                    + sum(fy[:, :, i:i + 1] * Dt[:, i][:, None, None] for i in range(n))
                    + sum(fz[:, i:i + 1] * Dt[:, i][:, None, None, None] for i in range(n)))

        def add(out, U, ie, ge, D):
            nf = U.shape[1]
            k = ie.shape[0]
            r = element(U[ie].reshape(k, n, n, n, nf), ge.reshape(ge.shape[:2] + (n, n, n))[..., None], D)
            return out.at[ie.reshape(-1)].add(r.reshape(-1, nf).astype(U.dtype))

        def block(out, U, idx, G, D):
            nb = idx.shape[0]
            one = G.shape[1] == 1                    # one element's factors for all of them
            m = nb if not work else min(nb, max(512, work // U.shape[1]))
            nfull = nb // m if m else 0
            rem = nb - nfull * m
            if nfull <= 1 and rem == 0:
                return add(out, U, idx, G, D)

            def body(out, k):
                ie = jax.lax.dynamic_slice(idx, (k * m, 0), (m, nq))
                ge = G if one else jax.lax.dynamic_slice(G, (0, k * m, 0), (G.shape[0], m, nq))
                return add(out, U, ie, ge, D), None

            out, _ = jax.lax.scan(body, out, jnp.arange(nfull))
            if rem:
                out = add(out, U, idx[nfull * m:], G if one else G[:, nfull * m:], D)
            return out

        def stiff(U, idxs, Gs, D):
            out = jnp.zeros_like(U)
            for idx, G in zip(idxs, Gs):
                if idx.shape[0]:
                    out = block(out, U, idx, G, D)
            return out

        return stiff

    def _lattice_stiff_fn(self):
        """``K U`` on a rank's block of equal bricks with one stiffness coefficient, the
        unknowns a 3D array (``lattice=``): no gather, no scatter, no table.

        The sum over the elements is then a sum of three Kronecker products,

            ``K = K0 (x) W1 (x) W2 + W0 (x) K1 (x) W2 + W0 (x) W1 (x) K2``,

        with, along each axis, the stiffness matrix ``K_d`` of the block's line of
        elements and its diagonal mass ``W_d``.  A line's matrix is applied through
        strided slices: node ``i`` of the line's elements is every ``p``-th entry from
        ``i``; row ``b`` of the element's matrix times those ``n`` slices is what the
        elements give their node ``b``; and the last node of an element is the first of
        the next, where the two are added.  The lines end with the block, so at its
        faces the result holds this rank's elements' part alone, as the kernel with a
        table leaves it for the exchange.

        The same signature as that kernel: ``Gs[0]`` holds the six small arrays
        (:meth:`_lattice_tables`), ``idxs`` and ``D`` are not read.
        """
        p, n = self.col.p, self.n
        e3 = self._lattice
        N3 = tuple(p * e + 1 for e in e3)

        def along(U, axis, nel, Kd):
            pre = (slice(None),) * axis
            s = [U[pre + (slice(i, i + p * nel, p),)] for i in range(n)]
            r = [sum(Kd[b, i] * s[i] for i in range(n)) for b in range(n)]
            shape = list(r[0].shape)
            shape[axis] = 1
            # an element's first node takes the last one's of the element before
            first = r[0] + jnp.concatenate([jnp.zeros(shape, dtype=r[0].dtype),
                                            r[p][pre + (slice(0, nel - 1),)]], axis=axis)
            body = jnp.stack([first] + r[1:p], axis=axis + 1)
            shape = list(U.shape)
            shape[axis] = p * nel
            return jnp.concatenate([body.reshape(shape), r[p][pre + (slice(nel - 1, nel),)]], axis=axis)

        def stiff(U, idxs, Gs, D):
            K0, K1, K2, W0, W1, W2 = Gs[0]
            U3 = U.reshape(N3 + (U.shape[1],))
            out = ((W1[None, :, None, None] * W2[None, None, :, None]) * along(U3, 0, e3[0], K0)
                   + (W0[:, None, None, None] * W2[None, None, :, None]) * along(U3, 1, e3[1], K1)
                   + (W0[:, None, None, None] * W1[None, :, None, None]) * along(U3, 2, e3[2], K2))
            return out.reshape(U.shape).astype(U.dtype)

        return stiff

    def _lattice_tables(self, mu):
        """The six arrays of :meth:`_lattice_stiff_fn` for the stiffness coefficient ``mu``:
        along each axis the element's stiffness matrix ``mu (D^T w D) / h_d``, ``(n, n)``,
        and the weights ``h_d w`` of the block's line of elements, added where two
        elements share a node, ``(p e_d + 1,)``."""
        col, p = self.col, self.col.p
        Ke = (col.D.T * col.w1) @ col.D
        out = [mu * Ke / h for h in col.brick]
        for h, nel in zip(col.brick, self._lattice):
            wl = np.concatenate([np.tile(col.w1[:p], nel), [col.w1[p]]])
            wl[p:-1:p] += col.w1[p]
            out.append(h * wl)
        return tuple(jax.device_put(jnp.asarray(x, dtype=self.geometry_dtype), self.device) for x in out)

    def _interp_m(self, m_loc, j, a=0, b=None):
        """Parameter values at the nodes of elements ``a:b`` of block ``j``,
        ``(b - a, n^3)``."""
        return jnp.einsum("qa,ea->eq", self._Nm, m_loc[self._edofs_m[j][a:b]])

    def _adder(self):
        """``M.at[i].add(v)``, written into ``M``'s own buffer."""
        return self._get("add", lambda: jax.jit(lambda M, i, v: M.at[i].add(v.astype(M.dtype)), donate_argnums=0))

    def _assemble_nodal(self, elem_vals, k):
        """Sum element nodal values (one ``(ne_j, n^3, k)`` array per piece of
        ``self._pieces``, taken from the iterable one at a time) into local dof vectors
        ``(nloc, k)`` and over the sharing ranks."""
        add = self._adder()
        loc = jax.device_put(jnp.zeros((self.nloc, k), dtype=self.dtype), self.device)
        for (j, la, lb, a, b), v in zip(self._pieces, elem_vals):
            loc = add(loc, self._idx[j][la:lb].reshape(-1), v.reshape(-1, k))
        return self._sync(loc)

    def _merge_fn(self):
        """``(shared, merge)``: the partial values this rank sends for the shared dofs,
        and the residual with the shared dofs' totals in place of its partial values.

        Both the gather and the sum are done on the device; the host only passes the
        buffers between ranks.  The contributions are added in ascending rank order,
        this rank's among them, so every sharer of a dof computes the same bits.  The
        barrier holds the compiler to that order: it otherwise rewrites the scattered
        values plus this rank's own as a scatter onto the own values, which puts them
        first on every rank that is third or later among the sharers, and the copies
        of such a dof then drift apart with the steps (the field is never exchanged)."""
        order = self._iface_order

        def shared(R, iface, send):
            return R[iface][send]

        def merge(R, recv, iface, send):
            own = R[iface]
            tot = jnp.zeros_like(own)
            recv = recv.astype(R.dtype)
            for r, a, b in order:
                tot = jax.lax.optimization_barrier(tot) + own if a is None else tot.at[send[a:b]].add(recv[a:b])
            return R.at[iface].set(tot)

        return shared, merge

    def _to_host(self, x):
        """The values this rank sends, a device array ``(nsend, nf)``, on the host: the
        array JAX makes of it, or, when that would be a large one, its pieces brought
        one after the other into an array that is kept (:data:`HOST_PIECE`)."""
        piece = (0 if self.device.platform == "cpu" else 16) if HOST_PIECE == "auto" else float(HOST_PIECE)
        piece = int(piece * 2 ** 20)
        if not piece or x.nbytes <= 2 * piece:
            return np.asarray(x)
        out = self.iface.kept("send", x.shape, x.dtype)
        m = max(1, piece * x.shape[0] // x.nbytes)
        for a in range(0, x.shape[0], m):
            out[a:a + m] = np.asarray(x[a:a + m])
        return out

    def _to_device(self, x):
        """What a compiled call is handed from host memory: ``x`` itself, which JAX then
        copies while it launches the call, or ``x`` on the device, waited for
        (:data:`PUT_FIRST`)."""
        return jax.block_until_ready(jax.device_put(x, self.device)) if PUT_FIRST else x

    def _sync(self, R):
        """Replace the shared dofs' partial sums in ``R`` (``(nloc, nf)``) by their
        totals over the ranks."""
        if not self.iface.neighbors:
            return R
        # the totals are written into R's own buffer: R is the caller's no longer
        shared, merge = self._get("merge", lambda: (jax.jit(self._merge_fn()[0]),
                                                    jax.jit(self._merge_fn()[1], donate_argnums=0)))
        out = shared(R, self._iface_idx, self._iface_send)
        if self._device_mpi():
            R = merge(R, self._swap_device(out), self._iface_idx, self._iface_send)
            return jax.block_until_ready(R)     # the neighbours' values are read before the next arrive
        recv = self.iface.swap(self._to_host(out))
        return merge(R, self._to_device(recv), self._iface_idx, self._iface_send)

    def _device_mpi(self):
        """Whether the exchange is between the cards' own memory (:data:`DEVICE_MPI`)."""
        if DEVICE_MPI in ("1", "yes", "true", "on"):
            return True
        return (DEVICE_MPI == "auto" and os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0") == "1"
                and self.device.platform != "cpu")

    def _swap_device(self, out):
        """The neighbours' values for this rank's shared ones ``out`` (a device array, laid
        out as the interface says), from card to card, in the device array that is kept
        for the shape and the precision."""
        key = (tuple(out.shape), str(out.dtype))
        if key not in self._recv:
            self._recv[key] = jax.device_put(jnp.zeros(out.shape, dtype=out.dtype), self.device)
        return self.iface.swap_device(out, self._recv[key])

    def _overlapped(self, nf=1):
        """Whether this rank's steps of ``nf`` fields make their exchange behind their
        update (:data:`OVERLAP`, :data:`OVERLAP_MIN`); never on a rank that shares
        nothing.  Decided from what every rank knows, so all ranks decide alike."""
        if not self.iface.neighbors:
            return False
        if OVERLAP in ("1", "yes", "true", "on"):
            return True
        return (OVERLAP == "auto" and self._device_mpi()
                and nf * self.global_size >= OVERLAP_MIN * self.comm.size)

    def _rows_tables(self):
        """``(rows, shared, absorbing)``, device arrays of int32: the rows that an
        overlapped step computes again behind its update, which are the shared dofs and
        the absorbing nodes, each once, in ascending order (``rows_dofs`` on the host),
        and where among them the shared dofs (``iface``) and the absorbing nodes
        (``ab_dofs``) are.  A rank's own, and the same for every sweep."""
        if self._rows is None:
            ab = self.ab_dofs if self._ab else np.zeros(0, dtype=np.int64)
            #: the rows of an overlapped step, local dof numbers in ascending order
            self.rows_dofs = np.union1d(np.asarray(self.iface.iface, dtype=np.int64), ab)
            self._rows = tuple(self._put(x, jnp.int32) for x in (
                self.rows_dofs, np.searchsorted(self.rows_dofs, self.iface.iface),
                np.searchsorted(self.rows_dofs, ab)))
        return self._rows

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
        stiffness, which do not depend on the parameter: those of every node times the
        stiffness coefficient there, or, on a mesh of equal bricks, one brick's three
        times the coefficient where that is one number over the whole mesh (three
        arrays of every node, in a second pass, where it is not).
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
        brick = col.brick is not None
        if first:
            mu_fn = jax.jit(jax.vmap(jax.vmap(self._mu)))
            dmu_fn = jax.jit(jax.vmap(jax.vmap(jax.grad(self._mu))))
            ten_fn = jax.jit(jax.vmap(self._tensor))
            bad, off, vol = 0.0, 0.0, 0.0
            if brick:
                # equal bricks: nothing is kept per element if the stiffness coefficient is one
                # number, which is known when every rank has seen its range (below)
                mu_lo, mu_hi = np.inf, -np.inf
            else:
                Gs = [zeros((6, b - a, nq), self.geometry_dtype) for (a, b) in self._eblocks]
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
                if brick:
                    mu_lo, mu_hi = min(mu_lo, float(jnp.min(mu))), max(mu_hi, float(jnp.max(mu)))
                else:
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
            if brick:
                # one answer for every rank: the kernels of all are built for the same factors
                lo = self.comm.allreduce(mu_lo, op=MPI.MIN)
                hi = self.comm.allreduce(mu_hi, op=MPI.MAX)
                g3 = col.brick_factors()[0]                                      # (3, nq)
                if hi - lo <= 1e-13 * abs(hi):
                    Gs = ([self._lattice_tables(lo)] if self._lattice is not None else
                          [self._put(g3[:, None, :] * lo, self.geometry_dtype)] * len(self._eblocks))
                elif self._lattice is not None:
                    raise NotImplementedError("lattice= takes one stiffness coefficient, and this density's "
                                              "varies over the mesh (from %g to %g)" % (lo, hi))
                else:
                    # a coefficient that varies: a brick's three factors times it at every node,
                    # in a second pass over the elements
                    g3 = self._put(g3, jnp.float64)
                    Gs = [zeros((3, b - a, nq), self.geometry_dtype) for (a, b) in self._eblocks]
                    for (j, la, lb, a, b) in self._pieces:
                        X = col.block_nodes(a, b)
                        mu = mu_fn(self._interp_m(m_loc, j, la, lb), X, *self._aux_of(a, b, X))
                        Gs[j] = fill(Gs[j], g3[:, None, :] * mu[None], la)
                        del X, mu
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
        if self._ab:
            self._ab_gain = None                               # beta is of the mass that was
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
        # (the norm arrives as a double: the iterate must not become one with it, or every
        # kernel of the iteration is compiled again for doubles)
        scale = self._get("scale", lambda: jax.jit(lambda w, s: w / s.astype(w.dtype), donate_argnums=0))
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

    def set_wavelets(self, wavelets):
        """Other amplitudes in time for the sources that are set, ``(nsteps + 1,
        nsources)``: another shot of the same survey, say, with every column zero but
        its own.  The compiled steps stay, which :meth:`set_sources` drops (whether a
        rank holds a source is compiled into them): a shot then costs no compilation."""
        if self.sources is None:
            raise RuntimeError("no sources to give other wavelets: call set_sources() first")
        w = self._put(np.asarray(wavelets)[:, self.sources.index])
        if w.shape != self._src_t.shape:
            raise ValueError("wavelets of shape %s for sources set with %s: set the sources again"
                             % (w.shape, self._src_t.shape))
        self._src_t = w
        self._final = None
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
    def _build_step(self, has_q, has_dM, record, correlate, inject=False, overlap=False):
        """The compiled halves of a step, ``(first, second, whole)``.

        ``first`` is the element work, up to where the shared dofs must be summed over
        the ranks, and returns the residual with this rank's partial values at them;
        ``second`` takes their totals and updates.  ``whole`` is the two in one call,
        for a single rank.

        Field 0 is the main field, ``R = K u - f_n``.  Fields 1.. are either adjoint
        fields, with the series ``q`` injected at the receivers, or incremental fields,
        forced through ``second`` by the mass perturbations times the main field's
        increment.

        With absorbing nodes ``second`` is the same step and then the rows of those
        nodes again, by three rules (``A = - dt^2 M^{-1} R`` as everywhere, ``beta = dt C /
        (2 M)``, "new" the next state in the direction of the sweep):

        * every field takes ``(1 + beta) x_new = 2 x - (1 - beta) x_old + A`` there.  For
          the incremental fields that is the derivative of the scheme, ``C`` not
          depending on the parameter, and for the adjoint fields its transpose;
        * the main field of a backward sweep (``inject``) takes the forward sweep's own
          values there instead (see the module text);
        * the main field's second difference ``u_{n+1} - 2 u_n + u_{n-1}``, which forces
          the incremental fields and is correlated with the adjoint ones, is ``A``
          inside and ``A - beta (u_{n+1} - u_{n-1})`` there.  Taken so and not as the
          difference of the three states, which loses the digits they share.

        ``second`` and ``whole`` then take three more arguments, the forward sweep's
        values (``inject``) and the rows of the two states at the nodes, and return
        the new state's rows before the merged residual.

        ``overlap`` builds the step of several ranks whose exchange is made behind its
        update instead (:data:`OVERLAP`), ``(first, (bulk, rows), None)``.  Only the rows
        ``J`` of the update need more than this rank has when its element work is done:
        the shared dofs, which wait for the other ranks' part of the residual, and with
        them the absorbing nodes, whose rows are not the plain update
        (:meth:`_rows_tables`).  So:

        * ``first`` returns a third array, the rows ``J`` of the residual, with this
          rank's partial values still;
        * ``bulk`` is the plain update of every row with that residual, written into
          the buffers of the previous state and of the correlation as ``second`` writes
          them.  It is launched and not waited for: the exchange runs beside it.  At
          the rows ``J`` what it writes is wrong, and nowhere else;
        * ``rows`` writes the rows ``J`` again, in place.  The neighbours' values are
          summed into the residual's rows by the sum of the step above (``merge`` on
          the rows: the same contributions in the same order), and the rows are
          updated from the OLD rows of the two states and of the correlation by the
          update's own formulas, which are functions of rows and written once: ``second``
          and ``second_ab`` are handed all the rows of a rank in the step above and in
          ``bulk``, and here the rows ``J``, with the inverse mass at them and with the
          places of the absorbing nodes among them.  Then the receivers are read, from
          the finished state.  The rows are computed again and not corrected by the
          neighbours' part: each sharer of a dof would round its own correction, and
          their copies would differ; computed so, every sharer still does the same
          arithmetic on the same numbers.

        The old rows are not gathered by the step.  As the rows at the absorbing nodes
        are, those of the two states and of the correlation at ``J`` are carried from
        step to step by :meth:`_run`: ``rows`` returns what it writes into the arrays.
        Of its whole arrays a step so gathers the residual's rows ``J`` and no others,
        and scatters the rows ``J`` of the new state and of the correlation.
        """
        stiff = self._stiff_fn()
        ab = self._ab
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

        # The update is a function of rows: second and second_ab take rows of the two
        # states, of the residual, of the correlation and of M^{-1} dM, with the inverse
        # mass of those rows in c, and return the new rows.  They are handed all the rows
        # of a rank (whole, second_par, bulk) or the rows J of an overlapped step (rows),
        # so that the formulas are written once.
        def second(c, U, Um, R, S, MdM):
            A = (-c["dt2"]) * c["Minv"][:, None] * R
            if has_dM:
                # by concatenation: an indexed update of a slice lowers to a scatter
                A = jnp.concatenate([A[:, :1], A[:, 1:] - MdM * A[:, :1]], axis=1)
            Unew = 2.0 * U - Um + A
            if correlate:
                S = S + Unew[:, 1:] * A[:, :1]
            return Unew, S

        def second_ab(c, U, Um, R, S, MdM, hb, Ub, Umb):
            # Ub and Umb are the rows of U and Um at the absorbing nodes, (nab, nf), carried
            # from step to step: gathered here, they kept both states alive beside the
            # update, a tenth more memory at a sweep's peak and 3 to 5 % more time
            A = (-c["dt2"]) * c["Minv"][:, None] * R
            ib, g, beta = c["ab_idx"], c["ab_g"], c["ab_beta"]      # the nodes, beta / (1 + beta), beta
            Ab = A[ib]
            # the main field: V + g (u_old - V) is (V + beta u_old) / (1 + beta), V the step
            # without C; or the forward sweep's own values
            V0 = 2.0 * Ub[:, 0] - Umb[:, 0] + Ab[:, 0]
            new0 = hb if inject else V0 + g * (Umb[:, 0] - V0)
            # what its second difference has beside A, - beta (u_{n+1} - u_{n-1}): u_{n+1} is
            # the new state of a forward sweep and the old one of a backward sweep
            d0 = beta * (new0 - Umb[:, 0]) if inject else beta * (Umb[:, 0] - new0)
            if has_dM:
                Ar = Ab[:, 1:] - MdM[ib] * (Ab[:, :1] + d0[:, None])
                A = jnp.concatenate([A[:, :1], A[:, 1:] - MdM * A[:, :1]], axis=1)
            else:
                Ar = Ab[:, 1:]
            Vr = 2.0 * Ub[:, 1:] - Umb[:, 1:] + Ar
            newb = jnp.concatenate([new0[:, None], Vr + g[:, None] * (Umb[:, 1:] - Vr)], axis=1)
            Unew = (2.0 * U - Um + A).at[ib].set(newb, indices_are_sorted=True, unique_indices=True)
            if correlate:
                S = S + Unew[:, 1:] * A[:, :1]
                S = S.at[ib].add(newb[:, 1:] * d0[:, None], indices_are_sorted=True, unique_indices=True)
            return Unew, S, newb

        def read(c, Unew):
            # the receivers' series, from a new state (all its rows)
            return (jnp.einsum("rk,rkf->fr", c["rec_w"], Unew[c["rec_dof"]])
                    if (record and has_rec) else None)

        shared, merge = self._merge_fn()

        def first_par(c, U, n, q):
            R = first(c, U, n, q)
            return R, shared(R, c["iface"], c["send"])

        if ab:
            def second_par(c, U, Um, R, recv, S, MdM, hb, Ub, Umb):
                R = merge(R, recv, c["iface"], c["send"])
                Unew, S, newb = second_ab(c, U, Um, R, S, MdM, hb, Ub, Umb)
                return Unew, S, read(c, Unew), newb, R

            def whole(c, U, Um, n, q, S, MdM, hb, Ub, Umb):
                Unew, S, newb = second_ab(c, U, Um, first(c, U, n, q), S, MdM, hb, Ub, Umb)
                return Unew, S, read(c, Unew), newb
        else:
            def second_par(c, U, Um, R, recv, S, MdM):
                # the merged residual is returned too: that lets it take R's buffer
                R = merge(R, recv, c["iface"], c["send"])
                Unew, S = second(c, U, Um, R, S, MdM)
                return Unew, S, read(c, Unew), R

            def whole(c, U, Um, n, q, S, MdM):
                Unew, S = second(c, U, Um, first(c, U, n, q), S, MdM)
                return Unew, S, read(c, Unew)

        if overlap and not serial:
            def first_rows(c, U, n, q):
                R = first(c, U, n, q)
                return R, shared(R, c["iface"], c["send"]), R[c["rows"]]

            def bulk(c, U, Um, R, S, MdM):
                # (never second_ab: the absorbing nodes are among the rows written again)
                return second(c, U, Um, R, S, MdM)

            def rows(cj, Unew, S, UJ, UmJ, RJ, SJ, MdMJ, recv, *more):
                # cj: the constants of the rows J, which are the step's with the inverse mass
                # at J in the place of Minv, and the places among J of the shared dofs and of
                # the absorbing nodes in those of iface and ab_idx (see _run).  UJ, UmJ, SJ,
                # MdMJ: the rows J of the two states, of the correlation and of M^{-1} dM as
                # they were before bulk wrote; RJ: the residual's, this rank's part; more: what
                # second_ab takes after them, the rows at the absorbing nodes as in every step
                RJ = merge(RJ, recv, cj["iface"], cj["send"])
                newJ, SJ, *newb = (second_ab if ab else second)(cj, UJ, UmJ, RJ, SJ, MdMJ, *more)
                Unew = Unew.at[cj["rows"]].set(newJ, indices_are_sorted=True, unique_indices=True)
                if correlate:
                    S = S.at[cj["rows"]].set(SJ, indices_are_sorted=True, unique_indices=True)
                return (Unew, S, read(cj, Unew), newJ, SJ) + tuple(newb)

            # bulk writes the new state and the new correlation into the buffers of the
            # previous state and of the correlation, as second_par does, and rows writes its
            # rows into bulk's two arrays: a step holds what it held, and the rows.  (The
            # residual is not donated to bulk: nothing it returns could take the buffer.)
            return (jax.jit(first_rows),
                    (jax.jit(bulk, donate_argnums=(2, 4)), jax.jit(rows, donate_argnums=(1, 2))), None)
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
        if self._ab:
            if self._ab_gain is None or self._ab_gain[0] != self.dt:
                # beta = dt C / (2 M), in double from the mass the fields have.  M holds every
                # sharer's total and C is the whole number on each of them, so the copies of a
                # shared node get the same beta with nothing exchanged
                M = np.asarray(self._M[self._ab_idx], dtype=np.float64)
                #: ``beta`` at the absorbing nodes ``ab_dofs``, on the host
                self.ab_beta = self.dt * self._ab_C / (2.0 * M)
                beta = self.ab_beta
                self._ab_gain = (self.dt, self._put(beta / (1.0 + beta)), self._put(beta))
            c.update(ab_idx=self._ab_idx, ab_g=self._ab_gain[1], ab_beta=self._ab_gain[2])
        return c

    def _run(self, U, Um, steps, q=None, dM=None, record=False, correlate=False, keep=None, history=None):
        """March ``(U, Um)`` (``(nloc, nf)``) through ``steps`` (the step indices handed
        to the sources and to ``q``), optionally recording the receivers after each
        step or accumulating the correlation of fields 1.. with field 0's increment.

        Returns ``(U, Um, records, S)``; the records are ``(nsteps, nf, nlocal)``.
        ``Um`` and ``dM`` are consumed (their buffers take the next state and
        ``M^{-1} dM``), so ``Um`` must not be ``U``.

        With absorbing nodes, ``keep`` (a :class:`BoundaryHistory`) takes the main field's
        values there after every step, and ``history`` (one that a forward sweep filled)
        gives them to a backward sweep, which at step index ``n`` rebuilds ``u_{n-1}``.

        On several ranks a step's exchange is made between its two halves, or behind its
        update (:data:`OVERLAP`; the two steps are those of :meth:`_build_step`).
        """
        if self.dt is None or not self.dt > 0.0:
            # (a step of zero moves nothing: every sweep would return zeros, without a word)
            raise RuntimeError("the time step is not set (dt = %r): give dt= or set .dt before a sweep" % (self.dt,))
        nf = U.shape[1]
        has_q, has_dM = q is not None, dM is not None
        ab = self._ab
        inject = ab and history is not None
        key = ("step", nf, has_q, has_dM, record, correlate) + ((inject,) if ab else ())
        over = self._overlapped(nf)
        if over:
            key += ("rows",)
        first, second, whole = self._get(
            key, lambda: self._build_step(has_q, has_dM, record, correlate, inject, over))
        c = self._consts()
        MdM = None
        if has_dM:
            # M^{-1} dM, in dM's own buffer
            scale = self._get("mdm", lambda: jax.jit(lambda Minv, d: Minv[:, None] * d, donate_argnums=1))
            MdM, dM = scale(self._Minv, dM), None
        S = jax.device_put(jnp.zeros((self.nloc, nf - 1), dtype=self.dtype), self.device) if correlate else None
        recs = []
        t0 = time.perf_counter()
        clock, shape, on_device = time.perf_counter, (self._iface_send.shape[0], nf), self._device_mpi()
        more = ()
        if ab:
            # the rows of the two states at the absorbing nodes, carried beside them
            Ub, Umb = U[self._ab_idx], Um[self._ab_idx]
        if over:
            # The overlapped step (see _build_step).  The rows J it computes again, their
            # constants (the step's, with the inverse mass at J and with the places among J
            # of the shared dofs and of the absorbing nodes), and what of the rows J it
            # carries from step to step as the absorbing rows are carried above: the two
            # states and the correlation, which hold what the steps wrote there; M^{-1} dM
            # at them does not change.
            bulk, rows = second
            J, shared_in, ab_in = self._rows_tables()
            c["rows"] = J
            cj = dict(c, Minv=c["Minv"][J], iface=shared_in)
            if ab:
                cj["ab_idx"] = ab_in
            UJ, UmJ = U[J], Um[J]
            SJ = S[J] if correlate else None
            MdMJ = MdM[J] if has_dM else None
        for n in steps:
            if ab:
                more = (history.row(int(n) - 1) if inject else None, Ub, Umb)
            n = self._to_device(n)
            if whole is not None:
                Unew, S, rec, *new = whole(c, U, Um, n, q, S, MdM, *more)
            elif over:
                # The step below with its exchange behind its update: the update of every row
                # is launched when the element work is done and NOT waited for, the exchange is
                # made while the device computes it, and the rows J are then written again
                # from what they were, with the neighbours' values.  From card to card nothing
                # waits for the two calls but the next step's element work, so t_element still
                # is the device's work of a step, less what of the update the exchange hid.
                ta = clock()
                posted = None if on_device else self.iface.post(shape, U.dtype)
                tb = clock()
                R, out, RJ = first(c, U, n, q)
                jax.block_until_ready(out)                  # waits for the element work
                tc = clock()
                if on_device:
                    tx = tc
                    Unew, S = bulk(c, U, Um, R, S, MdM)     # launched
                    td = clock()
                    recv = self._swap_device(out)           # beside it, from card to card
                    te = ty = tf = clock()
                else:
                    # through host arrays the values this rank sends are brought from the card
                    # before the update is launched (behind it, the copy would wait for it), and
                    # those it receives go to the card when the update is done, as they go to
                    # an idle card in the step below: the time inside MPI is what overlaps
                    out = self._to_host(out)
                    tx = clock()
                    Unew, S = bulk(c, U, Um, R, S, MdM)
                    td = clock()
                    recv = self.iface.swap(out, posted)
                    te = clock()
                    jax.block_until_ready(Unew)
                    ty = clock()
                    recv = self._to_device(recv)
                    tf = clock()
                del R
                Unew, S, rec, UJn, SJ, *new = rows(cj, Unew, S, UJ, UmJ, RJ, SJ, MdMJ, recv, *more)
                UmJ, UJ = UJ, UJn
                self.t_exchange += (tb - ta) + (te - td)
                self.t_element += tc - tb
                self.t_copy += (tx - tc) + (tf - ty)
            else:
                # the receives are posted before the element work: what the neighbours
                # send while this rank still computes is then written as it arrives
                # (from card to card they are posted with the sends: the device may still
                # be reading the array they are written into until its element work is done)
                ta = clock()
                posted = None if on_device else self.iface.post(shape, U.dtype)
                tb = clock()
                R, out = first(c, U, n, q)
                jax.block_until_ready(out)                  # waits for the element work
                tc = clock()
                if on_device:
                    td = tc
                    recv = self._swap_device(out)
                    te = tf = clock()
                else:
                    out = self._to_host(out)
                    td = clock()
                    recv = self.iface.swap(out, posted)
                    te = clock()
                    recv = self._to_device(recv)
                    tf = clock()
                Unew, S, rec, *new, R = second(c, U, Um, R, recv, S, MdM, *more)
                del R
                self.t_exchange += (tb - ta) + (te - td)
                self.t_element += tc - tb
                self.t_copy += (td - tc) + (tf - te)
            Um, U = U, Unew
            if ab:
                Umb, Ub = Ub, new[0]
                if keep is not None:
                    keep.append(Ub[:, 0])
            if rec is not None:
                recs.append(rec)
        nrec = self.receivers.nlocal if self.receivers is not None else 0
        if record:
            out = (np.stack([np.asarray(r, dtype=np.float64) for r in recs]) if recs
                   else np.zeros((len(steps), nf, nrec)))
        else:
            out = None
        if not recs:
            # (no record of this rank waits for the last step: the sweep ends, and its clock
            # stops, when the device has done it, on this rank as on those with receivers)
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
        keeps the last two states, from which every other sweep starts (and with
        absorbing nodes the field's values there after every step).  ``limit`` stops
        after that many steps and keeps nothing (to compile the sweep)."""
        zero = lambda: jax.device_put(jnp.zeros((self.nloc, 1), dtype=self.dtype), self.device)
        if limit is not None:
            self._run(zero(), zero(), range(int(limit)), record=True)
            return None
        self._final = None
        keep = self._new_history()
        U, Um, out, _ = self._run(zero(), zero(), range(self.nsteps), record=True, keep=keep)
        # the sweep's own arrays: no copy
        self._final = (U, Um) if keep is None else (U, Um, keep.close())
        return self._gather_records(out)[0]

    def _new_history(self):
        """What a sweep keeps the main field's values at the absorbing nodes in; None
        without such nodes."""
        if not self._ab:
            return None
        return BoundaryHistory(self.ab_dofs.size, self.dtype, self.device, self.history, self.history_piece,
                               pool=self._history_pool)

    def final_state(self):
        """The last two states ``(u_N, u_{N-1})`` of the forward sweep, on the device
        (columns, ``(nloc, 1)``); with absorbing nodes a third entry, the
        :class:`BoundaryHistory` of the sweep."""
        return self._final

    def _need_sensitivities(self):
        if not self.sensitivities:
            raise RuntimeError("this propagator was built with sensitivities=False: it has the forward sweep only")

    def mass_perturbations(self, directions):
        """``dM/dm [dm]`` as local dof vectors, ``(nloc, ndir)``, for parameter
        directions ``directions``."""
        self._need_sensitivities()
        mloc = [self._put(self.Vm.local_values(d)) for d in directions]
        # a piece of elements at a time, as the coefficients are computed: a whole block
        # of these products, of their stack and of its reshaped copy is as large as the
        # fields of a sweep when the block is the rank
        vals = (jnp.stack([self._wdkap[j][la:lb] * self._interp_m(m, j, la, lb) for m in mloc], axis=-1)
                for (j, la, lb, a, b) in self._pieces)
        return self._assemble_nodal(vals, len(mloc))

    def incremental(self, directions, limit=None, final=False):
        """Sensitivity of the receivers' series to the parameter in each direction:
        ``(ndir, nsteps + 1, nrec)``.  The forward field is propagated again alongside.
        ``limit`` stops after that many steps (to compile the sweep before timing it).

        ``final=True`` returns ``(series, final)``, the second what :meth:`final_state`
        gives after a forward sweep, here of the field propagated alongside:
        :meth:`adjoint` can start from it.  A Hessian action on one shot of many is then
        this sweep and a backward one, and nothing of a forward sweep is kept between
        the shots (two states and the boundary values each)."""
        nb = len(directions)
        dM = self.mass_perturbations(directions)
        zero = lambda: jax.device_put(jnp.zeros((self.nloc, 1 + nb), dtype=self.dtype), self.device)
        if limit is not None:
            self._run(zero(), zero(), range(int(limit)), dM=dM, record=True)
            return None
        keep = self._new_history() if final else None
        U, Um, out, _ = self._run(zero(), zero(), range(self.nsteps), dM=dM, record=True, keep=keep)
        # (the main field's columns alone: the sweep's arrays go with this call)
        last = (U[:, :1], Um[:, :1]) + (() if keep is None else (keep.close(),)) if final else None
        del U, Um
        dy = self._gather_records(out)[1:]
        return (dy, last) if final else dy

    def adjoint(self, q, final=None, limit=None):
        """The action of the transposed sensitivity on receiver series ``q``
        (``(nq, nsteps + 1, nrec)``): one parameter-space dual vector per series.

        The forward field is rebuilt backward from its last two states while the adjoint
        fields are carried backward with it; at absorbing nodes it is taken from the
        values the forward sweep kept.  ``final`` is that sweep's :meth:`final_state`
        (the last forward sweep's by default), or what ``incremental(..., final=True)``
        returned.  ``limit`` stops the sweep after that many steps (to compile it before
        timing it)."""
        self._need_sensitivities()
        final = final if final is not None else self._final
        if final is None:
            raise RuntimeError("no forward solve to reverse: call forward() first")
        history = None
        if self._ab:
            if len(final) < 3 or final[2].nrows != self.nsteps:
                raise RuntimeError("the forward solve to reverse kept no values at the absorbing nodes "
                                   "for these %d steps" % self.nsteps)
            history = final[2]
        q = np.asarray(q, dtype=np.float64)
        nb, N = q.shape[0], self.nsteps
        rec = self.receivers
        # field 0 runs the forward recurrence backward: at sweep step j (N .. 1) it applies
        # K to u_{j-1} with the source of step j-1; the adjoint fields take q[:, j]
        # (waited for before anything is launched beside them, as in unique_inverse: the series of a
        # survey's receivers for a batch of directions are hundreds of MB)
        qloc = (jax.block_until_ready(self._put(q[:, :, rec.index])) if rec.nlocal
                else jnp.zeros((nb, N + 1, 0), dtype=self.dtype))
        zb = jnp.zeros((self.nloc, nb), dtype=self.dtype)
        U = jnp.concatenate([final[1].reshape(self.nloc, 1), zb], axis=1)
        Um = jnp.concatenate([final[0].reshape(self.nloc, 1), zb], axis=1)
        del zb
        # the source index of the main field at sweep step j is j-1, the adjoint's is j:
        # shift q by one so that both read index j-1
        qshift = jnp.concatenate([qloc[:, 1:], jnp.zeros_like(qloc[:, :1])], axis=1)
        last = -1 if limit is None else max(N - 1 - int(limit), -1)
        U, Um, _, S = self._run(U, Um, range(N - 1, last, -1), q=qshift, correlate=True, history=history)
        # what the main field ended on: (u_{-1}, u_0) after all the steps
        self.rebuilt_rest = (U[:, 0], Um[:, 0]) if self.keep_rest else None
        del U, Um

        def to_parameters(loc, S, wdkap, idx, Nm, edofs):
            return loc.at[edofs.reshape(-1)].add(
                jnp.einsum("qa,eqb->eab", Nm, wdkap[:, :, None] * S[idx]).reshape(-1, S.shape[1]))

        # a piece of elements at a time, into one array: the fields' values at the nodes
        # of a whole block are as large as the fields when the block is the rank
        fold = self._get("fold", lambda: jax.jit(to_parameters, donate_argnums=0))
        loc = jax.device_put(jnp.zeros((self.nloc_m, nb), dtype=self.dtype), self.device)
        for (j, la, lb, a, b) in self._pieces:
            loc = fold(loc, S, self._wdkap[j][la:lb], self._idx[j][la:lb], self._Nm, self._edofs_m[j][la:lb])
        loc = np.asarray(loc, dtype=np.float64)
        return [self.Vm.assemble_dual(np.ascontiguousarray(loc[:, b])) for b in range(nb)]

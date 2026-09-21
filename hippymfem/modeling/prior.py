# Derived from hIPPYlib / hIPPYlibx (https://hippylib.github.io):
# Copyright (c) 2016-2018, The University of Texas at Austin & University of
# California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin, University of
# California--Merced, Washington University in St. Louis.
# Copyright (c) 2025-, Georgia Institute of Technology.
# Modified in 2026 for MFEM, hypre and JAX by Peng Chen, Georgia Institute of
# Technology.  See the file COPYRIGHT for details.
#
# hIPPyMFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software
# Foundation) version 2.0 dated June 1991.  See the file LICENSE.
r"""Gaussian prior models.

Matern-class Gaussian priors whose precision is a differential operator.  Writing
:math:`\mathcal{A}` for the square-root precision operator and :math:`M` for the
mass matrix, the bi-Laplacian prior has precision

.. math:: R = \mathcal{A} M^{-1} \mathcal{A},

so a sample is :math:`s = \mathcal{A}^{-1} \sqrt{M}\,\xi` with
:math:`\xi \sim N(0, I)`, because then
:math:`\mathrm{Cov}(s) = \mathcal{A}^{-1} M \mathcal{A}^{-1} = R^{-1}`.

**The square root of the mass matrix, without a square root.**  hIPPYlib(x) gets
:math:`\sqrt{M}` from a quadrature element space: with a diagonal quadrature mass
matrix :math:`M_q` it forms :math:`\sqrt{M} = G M_q^{-1/2}`.  The same operator
has an explicit entry-wise form, which is what is used here:

.. math:: \sqrt{M}_{i,(e,q)} = N_i(x_q^e)\,\sqrt{w_q\,|\det J^e_q|},

and indeed :math:`\sqrt{M}\sqrt{M}^{\!\top} = \sum_{e,q} w_q |\det J| N_i N_j = M`
exactly for that quadrature rule.  The Laplacian prior needs the same thing for
:math:`R = \gamma L + \delta M` directly, with the gradient components carrying
:math:`\sqrt{\gamma}` and the value component :math:`\sqrt{\delta}`.

So no square root is ever computed, and the white-noise vector lives on
quadrature points, which are **element-local**: drawing a prior sample needs no
communication.  Keying the noise on each element's centroid, not on its
partition-dependent number, makes it bit-identical under repartitioning, so
results can be compared across rank counts.
"""

import math
import numbers

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from ..common.keepalive import KeepAlive
from ..common.naming import SnakeCamel, sync_spellings
from ..common.operators import Operator, Solver2Operator, init_vector_like
from ..common.parvector import ParVector
from ..common.random import parRandom
from ..fem.assemble import assemble_native_matrix, mass_functional
from ..fem.elementbatch import get_batches
from ..fem.spaces import as_space
from ..algorithms.multivector import MultiVector
from ..algorithms.linSolvers import KrylovSolver, LUSolver
from ..common.operators import MatrixOperator, Operator2Solver
from ..algorithms.randomizedEigensolver import doublePass, doublePassG
from ..common.linalg import estimate_diagonal_inv2
from ..algorithms.traceEstimator import TraceEstimator


def _allreduce_array(comm, arr, op):
    """Elementwise allreduce of a small float array."""
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    out = np.empty_like(arr)
    comm.Allreduce(arr, out, op=op)
    return out


# --------------------------------------------------------------- sqrt operator
class QuadratureSqrtPrecision(Operator):
    r"""Map element-local white noise at quadrature points to the FE dual space.

    Builds

    .. math:: S_{i,(e,q,c)} = c_0\,N_i(x_q)\,\sqrt{w_q|\det J|}
              \quad\text{and}\quad
              c_1\,\partial_d N_i(x_q)\,\sqrt{w_q|\det J|}

    so that :math:`S S^{\!\top} = c_0^2 M + c_1^2 L`.  With
    ``(value_coeff, grad_coeff) = (1, None)`` this is :math:`\sqrt{M}`; with
    ``(sqrt(delta), sqrt(gamma))`` it is the square root of
    :math:`\delta M + \gamma L`.

    Applied matrix-free.  The noise space is purely local (no ghost dofs), and
    its entries are keyed on element centroids so that a sample does not depend
    on the partition.
    """

    def __init__(self, space, quadrature_degree, value_coeff=1.0, grad_coeff=None):
        self.space = as_space(space)
        self.comm = self.space.comm
        self.batches = get_batches(self.space.mesh, quadrature_degree)
        self.tables = self.batches.tables(self.space)
        self.vdim = self.space.vdim
        self.sdim = self.space.sdim
        self.value_coeff = None if value_coeff is None else float(value_coeff)
        self.grad_coeff = None if grad_coeff is None else float(grad_coeff)
        self.ncomp = (0 if self.value_coeff is None else 1) + \
                     (0 if self.grad_coeff is None else self.sdim)
        if self.ncomp == 0:
            raise ValueError("at least one of value_coeff, grad_coeff is required")

        mesh = self.space.mesh
        self._glob = self._element_keys(mesh)
        self._layout = []
        off = 0
        for gi, g in enumerate(self.batches.groups):
            n = g.ne * g.nq * self.ncomp * self.vdim
            self._layout.append((off, n, gi))
            off += n
        self.noise_local_size = off
        self._stride_max = max(
            [g.nq * self.ncomp * self.vdim for g in self.batches.groups] or [1]
        )
        self._global_ne = (mesh.GetGlobalNE() if hasattr(mesh, "GetGlobalNE")
                           else mesh.GetNE())
        # centroid keys are not contiguous: each element has its own stream offset
        self._contiguous = False
        self._nldof = self.space.fes.GetVSize()
        self._sqrt_wdet = [np.sqrt(g.wdet) for g in self.batches.groups]
        self._range_template = self.space.vector()
        # one allgather here, so noise_vector() never communicates
        self._noise_layout = ParVector(self.comm, self.noise_local_size).layout

    #: bits per coordinate when hashing an element centroid to a stream offset
    _KEY_BITS = 20

    def _element_keys(self, mesh):
        """A partition-independent integer key per local element.

        MFEM's ``GetGlobalElementNum`` is unique but **partition dependent** (the
        partitioner reorders elements), so noise keyed on it would be valid but
        would change with the rank count.  The element's *centroid* does not move
        under repartitioning, so it is used instead: quantized to ``_KEY_BITS``
        bits per dimension within the global bounding box, digits concatenated.
        Two elements collide only if their centroids agree to one part in
        ``2**_KEY_BITS`` of the domain, which is checked for.
        """
        ne = mesh.GetNE()
        sdim = mesh.SpaceDimension()
        cent = np.zeros((ne, sdim))
        if ne:
            # vertex coordinates in bulk, then one SWIG call per element for its
            # vertex list (PyMFEM has no element-to-vertex table)
            V = np.asarray(mesh.GetVertexArray(), dtype=np.float64)[:, :sdim]
            verts = [np.asarray(mesh.GetElementVertices(e), dtype=np.int64)
                     for e in range(ne)]
            nv = np.array([v.size for v in verts])
            for n in np.unique(nv):
                sel = np.flatnonzero(nv == n)
                idx = np.stack([verts[e] for e in sel])            # (nsel, n)
                cent[sel, :] = V[idx].mean(axis=1)

        comm = self.comm
        lo = cent.min(axis=0) if ne else np.full(sdim, np.inf)
        hi = cent.max(axis=0) if ne else np.full(sdim, -np.inf)
        lo = _allreduce_array(comm, lo, MPI.MIN)
        hi = _allreduce_array(comm, hi, MPI.MAX)
        span = np.where(hi - lo > 0, hi - lo, 1.0)

        M = 1 << self._KEY_BITS
        q = np.floor((cent - lo) / span * (M - 1) + 0.5).astype(np.int64)
        q = np.clip(q, 0, M - 1)
        # concatenate the digits; sdim * _KEY_BITS bits must fit an int64, so
        # _KEY_BITS <= 21 in 3D
        packed = np.zeros(ne, dtype=np.int64)
        for d in range(sdim):
            packed = (packed << self._KEY_BITS) | q[:, d]
        keys = np.empty(ne, dtype=object)
        keys[:] = [int(k) for k in packed]
        if ne and len(set(keys.tolist())) != ne:
            raise RuntimeError(
                "element centroids collided when quantized to %d bits per "
                "dimension; the mesh has elements closer than one part in 2**%d "
                "of its bounding box" % (self._KEY_BITS, self._KEY_BITS)
            )
        return keys

    # ----------------------------------------------------------- noise vectors
    def noise_vector(self):
        """A zero white-noise vector (element-local layout); no communication."""
        return ParVector(self.comm, self.noise_local_size,
                         layout=self._noise_layout)

    def sample_noise(self, sigma=1.0, out=None, rng=None):
        """Fill a noise vector with N(0, sigma^2), partition-independently.

        Entries are keyed on a hash of the element centroid (see
        :meth:`_element_keys`), so the same sample comes out on any number of
        ranks.  Nothing here communicates.
        """
        rng = rng if rng is not None else parRandom
        if out is None:
            out = self.noise_vector()
        if self._glob is None:
            rng.normal(sigma, out)          # no element keys available
            return out

        stride = self._stride_max
        # one vectorized draw for every element: a generator per element would
        # dominate the cost of posterior sampling
        offs = getattr(self, "_word_offsets", None)
        if offs is None or offs[0] != stride:
            # key * stride can pass 64 bits, so split it once into (low, high)
            # uint64 words and keep every later draw in array arithmetic
            mask = (1 << 64) - 1
            vals = [int(g) * stride for g in self._glob]
            lo = np.array([v & mask for v in vals], dtype=np.uint64)
            hi = np.array([v >> 64 for v in vals], dtype=np.uint64)
            offs = self._word_offsets = (stride, (lo, hi))
        block = rng.normal_blocks(sigma, offs[1], stride)

        for off, n, gi in self._layout:
            g = self.batches.groups[gi]
            s_g = g.nq * self.ncomp * self.vdim
            out.array[off:off + n] = block[g.elems, :s_g].reshape(-1)
        # advance past the whole key space so a later draw cannot overlap
        rng.advance((1 << (self._KEY_BITS * self.sdim)) * stride)
        return out

    # --------------------------------------------------------------- interface
    def init_vector(self, x, dim):
        if dim == "noise" or dim == 1:
            return init_vector_like(x, self.noise_vector())
        return init_vector_like(x, self._range_template)

    def mult(self, noise, out):
        """``out = S noise`` (quadrature space -> FE dual space)."""
        local = np.zeros(self._nldof)
        for off, n, gi in self._layout:
            g = self.batches.groups[gi]
            t = self.tables[gi]
            sq = self._sqrt_wdet[gi]
            xi = noise.array[off:off + n].reshape(
                g.ne, g.nq, self.ncomp, self.vdim
            )
            contrib = np.zeros((g.ne, self.vdim, t.nd))
            c = 0
            if self.value_coeff is not None:
                contrib += self.value_coeff * np.einsum(
                    "qi,eq,eqv->evi", t.N, sq, xi[:, :, c, :], optimize=True
                )
                c += 1
            if self.grad_coeff is not None:
                contrib += self.grad_coeff * np.einsum(
                    "qik,eqkd,eq,eqdv->evi",
                    t.G, g.Jinv, sq, xi[:, :, c:c + self.sdim, :], optimize=True
                )
            flat = (contrib.reshape(g.ne, self.vdim * t.nd) * t.signs).reshape(-1)
            local += np.bincount(t.edofs.reshape(-1), weights=flat,
                                 minlength=self._nldof)
        return self.space.assemble_dual(local, out)

    def multTranspose(self, v, noise_out):
        """``noise_out = S^T v`` (FE space -> quadrature space)."""
        local = self.space.local_values(v)
        for off, n, gi in self._layout:
            g = self.batches.groups[gi]
            t = self.tables[gi]
            sq = self._sqrt_wdet[gi]
            ve = (local[t.edofs] * t.signs).reshape(g.ne, self.vdim, t.nd)
            xi = np.zeros((g.ne, g.nq, self.ncomp, self.vdim))
            c = 0
            if self.value_coeff is not None:
                xi[:, :, c, :] = self.value_coeff * np.einsum(
                    "qi,eq,evi->eqv", t.N, sq, ve, optimize=True
                )
                c += 1
            if self.grad_coeff is not None:
                xi[:, :, c:c + self.sdim, :] = self.grad_coeff * np.einsum(
                    "qik,eqkd,eq,evi->eqdv", t.G, g.Jinv, sq, ve, optimize=True
                )
            noise_out.array[off:off + n] = xi.reshape(-1)
        return noise_out


# ----------------------------------------------------------- derived operators
class _BilaplacianR(Operator):
    r""":math:`R = \mathcal{A} M^{-1} \mathcal{A}` applied matrix-free.

    Assumes :math:`\mathcal{A}` symmetric, which holds for the diffusion + mass
    (+ Robin) forms used by the Matern priors.
    """

    def __init__(self, A, Msolver, comm):
        self.A = A
        self.Msolver = Msolver
        self.comm = comm
        self.keep(A)
        self._t1 = ParVector(comm, A.Height())
        self._t2 = ParVector(comm, A.Height())

    def init_vector(self, x, dim):
        return init_vector_like(x, self._t1)

    def mult(self, x, y):
        self.A.Mult(x.hypre, self._t1.hypre)
        self._t2.zero()
        self.Msolver.solve(self._t2, self._t1)
        self.A.Mult(self._t2.hypre, y.hypre)
        return y

    multTranspose = mult

    def getSize(self):
        return self.A.GetGlobalNumRows()

    def getComm(self):
        return self.comm


class _BilaplacianRsolver(KeepAlive):
    r""":math:`R^{-1} = \mathcal{A}^{-1} M \mathcal{A}^{-1}`."""

    def __init__(self, Asolver, M, comm):
        self.Asolver = Asolver
        self.M = M
        self.comm = comm
        self.keep(M)
        self._t1 = ParVector(comm, M.Height())
        self._t2 = ParVector(comm, M.Height())

    def init_vector(self, x, dim):
        return init_vector_like(x, self._t1)

    def solve(self, x, b):
        self._t1.zero()
        n1 = self.Asolver.solve(self._t1, b)
        self.M.Mult(self._t1.hypre, self._t2.hypre)
        x.zero()
        n2 = self.Asolver.solve(x, self._t2)
        return (n1 or 0) + (n2 or 0)


class _RinvM(Operator):
    r""":math:`R^{-1} M`, whose diagonal is the pointwise prior variance scaled
    by the mass matrix (used by :meth:`_Prior.trace`)."""

    def __init__(self, Rsolver, M, comm):
        self.Rsolver = Rsolver
        self.M = M
        self.comm = comm
        self.keep(M)
        self._tmp = ParVector(comm, M.Height())

    def init_vector(self, x, dim):
        return init_vector_like(x, self._tmp)

    def mult(self, x, y):
        self.M.Mult(x.hypre, self._tmp.hypre)
        y.zero()
        self.Rsolver.solve(y, self._tmp)
        return y


# ------------------------------------------------------------------ base class
class _Prior(SnakeCamel, KeepAlive):
    """Shared prior behaviour: cost, gradient, trace, pointwise variance."""

    #: set by subclasses
    R = None
    Rsolver = None
    M = None
    Msolver = None
    mean = None
    comm = None

    def init_vector(self, x, dim):
        raise NotImplementedError

    def sample(self, noise=None, s=None, add_mean=True, rng=None):
        """A sample of the prior, ``s = mean + sqrt(C) noise``; returns ``s``.

        hIPPYlib's form ``prior.sample(noise, s)`` fills a given ``s`` from a given
        white-noise vector.  Both may be left out: ``prior.sample()`` draws the
        noise (partition-independently, from ``rng`` or ``parRandom``) and returns
        a new vector.
        """
        if noise is None:
            noise = self.sample_noise(1.0, rng=rng)
        if s is None:
            s = self.mean.duplicate()
        return self._sample(noise, s, add_mean)

    def _sample(self, noise, s, add_mean=True):
        raise NotImplementedError

    def noise_vector(self):
        """A white-noise vector of the right shape for :meth:`sample`."""
        raise NotImplementedError

    def sample_noise(self, sigma=1.0, out=None, rng=None):
        raise NotImplementedError

    def getHessianPreconditioner(self):
        return self.Rsolver

    def cost(self, m):
        r""":math:`\tfrac12 (m - \bar m)^{\!\top} R (m - \bar m)`."""
        d = m.copy().axpy(-1.0, self.mean)
        Rd = d.duplicate()
        self.R.mult(d, Rd)
        return 0.5 * Rd.inner(d)

    def grad(self, m, out):
        r""":math:`R (m - \bar m)`."""
        d = m.copy().axpy(-1.0, self.mean)
        self.R.mult(d, out)
        return out

    def trace(self, method="Exact", tol=1e-1, min_iter=20, max_iter=100, r=200):
        r"""Trace of :math:`R^{-1} M`, i.e. the integrated prior variance."""
        op = _RinvM(self.Rsolver, self.M, self.comm)
        if method == "Exact":
            mv = op.generate_vector(0)
            _diagonal_of_operator(op, mv)
            return mv.sum()
        if method == "Estimator":

            est = TraceEstimator(op, False, tol)
            tr, _ = est(min_iter, max_iter)
            return tr
        if method == "Randomized":

            dummy = op.generate_vector(0)
            Omega = MultiVector(dummy, r)
            parRandom.normal_multivector(1.0, Omega)
            d, _ = doublePassG(
                Solver2Operator(self.Rsolver, init_vector=self.init_vector),
                Solver2Operator(self.Msolver, init_vector=self.init_vector),
                _M_as_solver(self.M, self.comm),
                Omega, r, s=1, check=False,
            )
            return float(d.sum())
        raise ValueError("unknown trace method %r" % (method,))

    def pointwise_variance(self, method="Exact", k=1000000, r=200, n=None):
        r"""Diagonal of :math:`R^{-1}`, the pointwise prior variance.

        ``"Exact"`` applies :math:`R^{-1}` to every unit vector (small problems only).
        ``"Randomized"`` is hIPPYlib's: the diagonal of the rank-``r`` truncation of
        :math:`R^{-1}` from a randomized eigensolver.  It is **biased low**, even for
        large ``r``, because a prior covariance has a slowly decaying spectrum.
        ``"MonteCarlo"`` averages the squares of ``n`` prior samples (``n`` defaults
        to ``r``): unbiased at one solve per sample, so it is the method to use when
        samples are cheap, as they are with hypre on a device.
        """
        pw = ParVector(self.comm, self.M.Height())
        if method == "MonteCarlo":
            n = int(r if n is None else n)
            noise = self.noise_vector()
            s = ParVector(self.comm, self.M.Height())
            acc = np.zeros(pw.local_size)
            for _ in range(n):
                self.sample_noise(1.0, noise)
                self.sample(noise, s, add_mean=False)
                acc += s.array ** 2
            pw.array[:] = acc / max(n, 1)
            return pw
        if method == "Exact":
            _diagonal_of_operator(
                Solver2Operator(self.Rsolver, init_vector=self.init_vector), pw
            )
        elif method == "Estimator":

            estimate_diagonal_inv2(self.Rsolver, k, pw)
        elif method == "Randomized":

            Omega = MultiVector(pw, r)
            parRandom.normal_multivector(1.0, Omega)
            d, U = doublePass(
                Solver2Operator(self.Rsolver, init_vector=self.init_vector),
                Omega, r, s=1, check=False,
            )
            pw.zero()
            for i in range(U.nvec()):
                pw.array[:] += float(d[i]) * U[i].array ** 2
        else:
            raise ValueError("unknown pointwise_variance method %r" % (method,))
        return pw


def _diagonal_of_operator(op, d):
    """Exact diagonal of a matrix-free operator, by applying it to unit vectors.

    Global in cost (one application per global dof), so it is only for the small
    problems where hIPPYlib also does this.
    """
    x = op.generate_vector(1)
    y = op.generate_vector(0)
    lo, hi = d.owner_range
    n = d.global_size
    d.zero()
    for j in range(n):
        x.zero()
        if lo <= j < hi:
            x.array[j - lo] = 1.0
        op.mult(x, y)
        if lo <= j < hi:
            d.array[j - lo] = y.array[j - lo]
    return d


def _M_as_solver(M, comm):
    """Wrap a matrix as an object with ``solve`` meaning ``mult`` (for doublePassG)."""

    return Operator2Solver(MatrixOperator(M, comm))


# -------------------------------------------------------- sqrt-precision prior
class SqrtPrecisionPDE_Prior(_Prior):
    r"""Prior with precision :math:`R = \mathcal{A} M^{-1} \mathcal{A}`.

    Parameters
    ----------
    Vh : FunctionSpace
    domain_integrators : sequence of mfem.BilinearFormIntegrator
        The square-root precision form :math:`\mathcal{A}`.
    bdr_integrators : sequence, optional
        Boundary contributions to :math:`\mathcal{A}` (the Robin term).
    mean : ParVector, optional
    solver_type : {"krylov", "lu"}
        ``"lu"`` is serial only (see :mod:`hippymfem.algorithms.linSolvers`).
    """

    def __init__(self, Vh, domain_integrators, bdr_integrators=(), mean=None,
                 rel_tol=1e-12, max_iter=1000, solver_type="krylov",
                 quadrature_degree=None, systems_dim=None):
        self.Vh = as_space(Vh)
        self.comm = self.Vh.comm
        self.M = assemble_native_matrix(self.Vh, [mfem.MassIntegrator()])
        self.A = assemble_native_matrix(self.Vh, list(domain_integrators),
                                        list(bdr_integrators))
        self.keep(self.M, self.A)

        self.Msolver, self.Asolver = _make_prior_solvers(
            self.comm, self.M, self.A, solver_type, rel_tol, max_iter,
            systems_dim=systems_dim or (self.Vh.vdim if self.Vh.vdim > 1 else None),
        )
        if quadrature_degree is None:
            quadrature_degree = 2 * self.Vh.order
        self.sqrtM = QuadratureSqrtPrecision(self.Vh, quadrature_degree,
                                             value_coeff=1.0, grad_coeff=None)
        self.R = _BilaplacianR(self.A, self.Msolver, self.comm)
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M, self.comm)
        self.mean = mean if mean is not None else self.Vh.vector()
        self.keep(self.Msolver, self.Asolver, self.sqrtM)

    def init_vector(self, x, dim):
        if dim == "noise":
            return init_vector_like(x, self.sqrtM.noise_vector())
        return init_vector_like(x, self.Vh.vector())

    def noise_vector(self):
        return self.sqrtM.noise_vector()

    def sample_noise(self, sigma=1.0, out=None, rng=None):
        return self.sqrtM.sample_noise(sigma, out, rng)

    def _sample(self, noise, s, add_mean=True):
        r"""``s`` such that :math:`\mathcal{A} s = \sqrt{M}\,\xi`."""
        rhs = self.Vh.vector()
        self.sqrtM.mult(noise, rhs)
        s.zero()
        self.Asolver.solve(s, rhs)
        if add_mean:
            s.axpy(1.0, self.mean)
        return s


class LaplacianPrior(_Prior):
    r"""Prior with precision :math:`R = \gamma L + \delta M` assembled directly.

    Its square root is available in closed form on quadrature points, so samples
    need one :math:`R` solve rather than two :math:`\mathcal{A}` solves.
    """

    def __init__(self, Vh, gamma, delta, mean=None, rel_tol=1e-12, max_iter=1000,
                 solver_type="krylov", quadrature_degree=None):
        if float(delta) == 0.0:
            raise ValueError("intrinsic Gaussian priors (delta = 0) are not supported")
        self.Vh = as_space(Vh)
        self.comm = self.Vh.comm
        self.gamma = float(gamma)
        self.delta = float(delta)

        self.M = assemble_native_matrix(self.Vh, [mfem.MassIntegrator()])
        self.R = assemble_native_matrix(
            self.Vh,
            [mfem.DiffusionIntegrator(mfem.ConstantCoefficient(self.gamma)),
             mfem.MassIntegrator(mfem.ConstantCoefficient(self.delta))],
        )
        self.keep(self.M, self.R)
        self.Msolver, self.Rsolver = _make_prior_solvers(
            self.comm, self.M, self.R, solver_type, rel_tol, max_iter,
            systems_dim=self.Vh.vdim if self.Vh.vdim > 1 else None,
        )
        if quadrature_degree is None:
            quadrature_degree = 2 * self.Vh.order
        self.sqrtR = QuadratureSqrtPrecision(
            self.Vh, quadrature_degree,
            value_coeff=math.sqrt(self.delta), grad_coeff=math.sqrt(self.gamma),
        )
        self.mean = mean if mean is not None else self.Vh.vector()
        self.keep(self.Msolver, self.Rsolver, self.sqrtR)
        # R is a matrix here, not an operator; wrap it so cost/grad work uniformly

        self.Rmat = self.R
        self.R = MatrixOperator(self.Rmat, self.comm)

    def init_vector(self, x, dim):
        if dim == "noise":
            return init_vector_like(x, self.sqrtR.noise_vector())
        return init_vector_like(x, self.Vh.vector())

    def noise_vector(self):
        return self.sqrtR.noise_vector()

    def sample_noise(self, sigma=1.0, out=None, rng=None):
        return self.sqrtR.sample_noise(sigma, out, rng)

    def _sample(self, noise, s, add_mean=True):
        rhs = self.Vh.vector()
        self.sqrtR.mult(noise, rhs)
        s.zero()
        self.Rsolver.solve(s, rhs)
        if add_mean:
            s.axpy(1.0, self.mean)
        return s


def _make_prior_solvers(comm, M, A, solver_type, rel_tol, max_iter,
                        systems_dim=None):
    """Mass and precision solvers with hIPPYlib's choices of method."""

    if solver_type == "lu":
        Ms, As = LUSolver(comm), LUSolver(comm)
    elif solver_type == "krylov":
        Ms = KrylovSolver(comm, "cg", "jacobi")
        As = KrylovSolver(comm, "cg", "amg", systems_dim=systems_dim)
    else:
        raise ValueError("unknown solver_type %r" % (solver_type,))
    Ms.set_operator(M)
    As.set_operator(A)
    for s in (Ms, As):
        s.parameters["rel_tolerance"] = rel_tol
        s.parameters["max_iter"] = max_iter
        s.parameters["error_on_nonconvergence"] = True
        s.parameters["nonzero_initial_guess"] = False
    return Ms, As


# ---------------------------------------------------------------- constructors
def BiLaplacianPrior(Vh, gamma, delta, Theta=None, mean=None, rel_tol=1e-12,
                     max_iter=1000, robin_bc=False, solver_type="krylov",
                     quadrature_degree=None):
    r"""Matern-class prior with :math:`\mathcal{A} = \gamma\,\mathrm{div}(\Theta\nabla)
    + \delta`.

    ``Theta`` may be ``None`` (isotropic), a scalar, a numpy array (a constant
    tensor), or an ``mfem.MatrixCoefficient``.  ``robin_bc=True`` adds
    :math:`\sqrt{\gamma\delta}/1.42` on the boundary, which reduces the
    boundary-layer artefact in the pointwise variance.
    """
    Vh = as_space(Vh)
    gamma = float(gamma)
    delta = float(delta)
    keep = []
    if Theta is None:
        diff = mfem.DiffusionIntegrator(mfem.ConstantCoefficient(gamma))
    elif isinstance(Theta, numbers.Number):
        diff = mfem.DiffusionIntegrator(mfem.ConstantCoefficient(gamma * float(Theta)))
    elif isinstance(Theta, np.ndarray):
        dm = mfem.DenseMatrix(np.ascontiguousarray(gamma * np.asarray(Theta, float)))
        mc = mfem.MatrixConstantCoefficient(dm)
        keep += [dm, mc]
        diff = mfem.DiffusionIntegrator(mc)
    else:
        sc = mfem.ScalarMatrixProductCoefficient(gamma, Theta)
        keep += [Theta, sc]
        diff = mfem.DiffusionIntegrator(sc)
    mass = mfem.MassIntegrator(mfem.ConstantCoefficient(delta))
    bdr = []
    if robin_bc:
        beta = math.sqrt(gamma * delta) / 1.42
        bdr.append(mfem.MassIntegrator(mfem.ConstantCoefficient(beta)))
    prior = SqrtPrecisionPDE_Prior(
        Vh, [diff, mass], bdr, mean, rel_tol, max_iter, solver_type,
        quadrature_degree,
    )
    prior.keep(*keep)
    prior.gamma, prior.delta, prior.robin_bc = gamma, delta, robin_bc
    return prior


def VectorBiLaplacianPrior(Vh, gamma, delta, mean=None, rel_tol=1e-12,
                           max_iter=1000, robin_bc=False, solver_type="krylov",
                           quadrature_degree=None):
    r"""Componentwise Matern prior on a vector space.

    ``gamma`` and ``delta`` are per-component sequences whose entries must agree:
    MFEM's vector integrators apply one scalar coefficient to every component, so
    differing values are rejected rather than silently averaged.
    """
    Vh = as_space(Vh)
    gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    if Vh.vdim == 1:
        return BiLaplacianPrior(Vh, gamma[0], delta[0], None, mean, rel_tol,
                                max_iter, robin_bc, solver_type, quadrature_degree)
    if not (np.allclose(gamma, gamma[0]) and np.allclose(delta, delta[0])):
        raise NotImplementedError(
            "component-dependent gamma/delta are not supported: MFEM's vector "
            "integrators apply one scalar coefficient to every component. "
            "Build the blocks explicitly with SqrtPrecisionPDE_Prior instead."
        )
    g, d = float(gamma[0]), float(delta[0])
    integs = [
        mfem.VectorDiffusionIntegrator(mfem.ConstantCoefficient(g)),
        mfem.VectorMassIntegrator(mfem.ConstantCoefficient(d)),
    ]
    bdr = []
    if robin_bc:
        bdr.append(mfem.VectorMassIntegrator(
            mfem.ConstantCoefficient(math.sqrt(g * d) / 1.42)))
    prior = SqrtPrecisionPDE_Prior(Vh, integs, bdr, mean, rel_tol, max_iter,
                                   solver_type, quadrature_degree,
                                   systems_dim=Vh.vdim)
    prior.gamma, prior.delta, prior.robin_bc = g, d, robin_bc
    return prior


def MollifiedBiLaplacianPrior(Vh, gamma, delta, locations, m_true, Theta=None,
                              pen=1e1, order=2, rel_tol=1e-12, max_iter=1000,
                              solver_type="krylov", quadrature_degree=None):
    r"""Matern prior with a mollifier pinning the field near given locations.

    Adds :math:`\mathrm{pen}\,\delta\, \mathrm{mollifier}\, m\, \hat m` to the
    square-root precision form and sets the mean so that the prior mean matches
    ``m_true`` where the mollifier is active.
    """
    Vh = as_space(Vh)
    if Vh.vdim != 1:
        raise ValueError("MollifiedBiLaplacianPrior expects a scalar space")
    gamma, delta = float(gamma), float(delta)
    if delta == 0.0:
        raise ValueError("delta must be nonzero for the mollifier scaling")
    locations = np.atleast_2d(np.asarray(locations, dtype=float))

    h = math.sqrt(gamma / delta)          # correlation length scale

    def mollifier(x):
        s = 0.0
        for c in locations:
            d2 = np.sum(((x[: c.size] - c) / h) ** 2)
            s += math.exp(-0.5 * d2 ** (0.5 * order) if order != 2 else -0.5 * d2)
        return s

    mfun = Vh.project(mollifier)
    mgf = Vh.to_gridfunction(mfun)
    moll = _GridFunctionCoefficient(mgf)
    pen_coeff = mfem.ProductCoefficient(
        mfem.ConstantCoefficient(pen * delta), moll
    )

    keep = [mfun, mgf, moll, pen_coeff]
    if Theta is None:
        diff = mfem.DiffusionIntegrator(mfem.ConstantCoefficient(gamma))
    elif isinstance(Theta, np.ndarray):
        dm = mfem.DenseMatrix(np.ascontiguousarray(gamma * np.asarray(Theta, float)))
        mc = mfem.MatrixConstantCoefficient(dm)
        keep += [dm, mc]
        diff = mfem.DiffusionIntegrator(mc)
    else:
        sc = mfem.ScalarMatrixProductCoefficient(gamma, Theta)
        keep += [Theta, sc]
        diff = mfem.DiffusionIntegrator(sc)
    integs = [diff,
              mfem.MassIntegrator(mfem.ConstantCoefficient(delta)),
              mfem.MassIntegrator(pen_coeff)]

    prior = SqrtPrecisionPDE_Prior(Vh, integs, (), None, rel_tol, max_iter,
                                   solver_type, quadrature_degree)
    prior.keep(*keep)
    # mean solves A mean = pen*delta * moll * m_true  (in the weak sense).  The
    # right-hand side is assembled as a linear form, equal to round-off to the
    # weighted mass matrix times m_true, so no parameter-space matrix lives as
    # long as the prior.
    rhs = mass_functional(Vh, m_true, coeff=pen_coeff)
    mean = Vh.vector()
    prior.Asolver.solve(mean, rhs)
    prior.mean = mean
    prior.gamma, prior.delta = gamma, delta
    return prior


class _GridFunctionCoefficient(mfem.PyCoefficient):
    def __init__(self, gf):
        super(_GridFunctionCoefficient, self).__init__()
        self.gf = gf

    def Eval(self, T, ip):
        return float(self.gf.GetValue(T.ElementNo, ip))


def BiLaplacianComputeCoefficients(sigma2, rho, ndim):
    r"""``(gamma, delta)`` giving marginal variance ``sigma2`` and correlation
    length ``rho`` for the Matern-class bi-Laplacian prior."""
    nu = 2.0 - 0.5 * ndim
    kappa = np.sqrt(8.0 * nu) / rho
    s = (np.sqrt(sigma2) * np.power(kappa, nu)
         * np.sqrt(np.power(4.0 * np.pi, 0.5 * ndim) / math.gamma(nu)))
    return 1.0 / s, np.power(kappa, 2) / s


# --------------------------------------------------------------- finite-dim
class GaussianRealPrior(_Prior):
    """Gaussian prior on a finite-dimensional (``Real``-type) space.

    The covariance is a dense ``n x n`` matrix; used for hyper-parameters and
    for testing, where an exact reference is wanted.
    """

    def __init__(self, Vh=None, covariance=None, mean=None, comm=None):
        self.Vh = None if Vh is None or isinstance(Vh, int) else as_space(Vh)
        self.comm = comm if comm is not None else (
            self.Vh.comm if self.Vh is not None else MPI.COMM_WORLD
        )
        if covariance is None:
            raise ValueError("GaussianRealPrior needs a covariance matrix")
        cov = np.atleast_2d(np.asarray(covariance, dtype=float))
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("covariance must be square")
        self.dim = cov.shape[0]
        self.covariance = cov
        self.precision = np.linalg.inv(cov)
        self.chol = np.linalg.cholesky(cov)

        n_local = self.dim if self.comm.rank == 0 else 0
        self._template = ParVector(self.comm, n_local)
        if self._template.global_size != self.dim:
            raise ValueError("GaussianRealPrior must own all dofs on rank 0")

        self.R = _DenseOperator(self.precision, self._template)
        self.Rsolver = _DenseSolver(self.covariance, self._template)
        self.M = _DenseMatrixLike(np.eye(self.dim), self._template)
        self.Msolver = _DenseSolver(np.eye(self.dim), self._template)
        self.mean = mean if mean is not None else self._template.duplicate()

    def init_vector(self, x, dim):
        return init_vector_like(x, self._template)

    def noise_vector(self):
        return self._template.duplicate()

    def sample_noise(self, sigma=1.0, out=None, rng=None):
        rng = rng if rng is not None else parRandom
        if out is None:
            out = self.noise_vector()
        return rng.normal(sigma, out)

    def _sample(self, noise, s, add_mean=True):
        if self.comm.rank == 0:
            s.array[:] = self.chol @ noise.array
        if add_mean:
            s.axpy(1.0, self.mean)
        return s

    def trace(self, method="Exact", **kw):
        return float(np.trace(self.covariance))

    def pointwise_variance(self, method="Exact", **kw):
        out = self._template.duplicate()
        if self.comm.rank == 0:
            out.array[:] = np.diag(self.covariance)
        return out


class _DenseOperator(Operator):
    """Small dense operator replicated on rank 0."""

    def __init__(self, A, template):
        self.Adense = A
        self.template = template

    def init_vector(self, x, dim):
        return init_vector_like(x, self.template)

    def mult(self, x, y):
        if x.comm.rank == 0:
            y.array[:] = self.Adense @ x.array
        return y

    multTranspose = mult

    def getSize(self):
        return self.Adense.shape[0]

    def getComm(self):
        return self.template.comm


class _DenseMatrixLike(_DenseOperator):
    """A dense operator that also quacks like a HypreParMatrix for ``Mult``."""

    def Mult(self, x, y):
        arr = y.GetDataArray()
        if self.template.comm.rank == 0:
            arr[:] = self.Adense @ x.GetDataArray()
        return y

    def Height(self):
        return self.Adense.shape[0]

    def Width(self):
        return self.Adense.shape[1]


class _DenseSolver(KeepAlive):
    def __init__(self, Ainv_action, template):
        self.Adense = Ainv_action
        self.template = template
        self.parameters = {}

    def init_vector(self, x, dim):
        return init_vector_like(x, self.template)

    def solve(self, x, b):
        if b.comm.rank == 0:
            x.array[:] = self.Adense @ b.array
        return 1


sync_spellings(_Prior)

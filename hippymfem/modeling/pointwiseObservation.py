# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Pointwise observation operators.

``Mesh::FindPoints`` locates each target in the local partition; the shape
functions of the containing element, evaluated at the reference point it returns,
are the row of the observation matrix.

**Ownership.** A target on a shared face is found by more than one rank and a
target outside the domain by none.  The first case is resolved by giving the
target to the lowest-numbered rank that found it, so no observation is counted
twice; the second is by default an error, reported with the offending
coordinates rather than silently dropped (a missing observation changes the
inverse problem).

**Ordering.** Observation vectors are distributed by owning rank, so their local
order is not the order of the ``targets`` array.  That is invisible as long as
data comes from the operator itself; :meth:`PointwiseObservation.gather` and
:meth:`PointwiseObservation.scatter` convert to and from the original target
order for I/O.
"""

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from ..common.operators import Operator, init_vector_like
from ..common.parvector import ParVector
from ..fem.spaces import as_space


class PointwiseObservation(Operator):
    r"""Evaluation of a finite element field at a set of points.

    Parameters
    ----------
    Vh : FunctionSpace
    targets : (ntargets, sdim) array
    components : (ntargets, vdim) array, optional
        Weights across the solution components, giving a line-of-sight
        observation :math:`\sum_c w_c u_c(x_t)`.  Defaults to observing
        component 0 for scalar spaces, and is required for vector spaces unless
        ``component`` is given.
    component : int, optional
        Observe a single component of a vector space.
    """

    def __init__(self, Vh, targets, components=None, component=None,
                 warn_missing=None, missing="raise"):
        if warn_missing is not None:
            # legacy switch, overriding ``missing``: True raises, False warns
            missing = "raise" if warn_missing else "warn"
        if missing not in ("raise", "warn", "ignore"):
            raise ValueError("missing must be 'raise', 'warn' or 'ignore', got %r" % (missing,))
        self.space = as_space(Vh)
        self.comm = self.space.comm
        self.targets = np.atleast_2d(np.asarray(targets, dtype=float))
        self.ntargets = self.targets.shape[0]
        sdim = self.space.sdim
        if self.targets.shape[1] != sdim:
            raise ValueError(
                "targets have %d columns but the mesh has space dimension %d"
                % (self.targets.shape[1], sdim)
            )
        vdim = self.space.vdim
        if components is not None:
            self.components = np.atleast_2d(np.asarray(components, dtype=float))
            if self.components.shape != (self.ntargets, vdim):
                raise ValueError("components must have shape (ntargets, vdim)")
        elif component is not None:
            self.components = np.zeros((self.ntargets, vdim))
            self.components[:, int(component)] = 1.0
        elif vdim == 1:
            self.components = np.ones((self.ntargets, 1))
        else:
            raise ValueError(
                "a vector space needs `component` or `components` to say what "
                "is observed"
            )

        self._build(missing)
        self._range_template = ParVector(self.comm, self.n_owned)
        self._domain_template = self.space.vector()

    # -------------------------------------------------------------------- setup
    def _build(self, missing_policy):
        fes = self.space.fes
        mesh = self.space.mesh
        count, ids, ips = mesh.FindPoints(self.targets)
        ids = np.asarray(list(ids), dtype=np.int64)
        found = ids >= 0

        # lowest rank that found a target owns it
        rank = self.comm.rank
        claim = np.where(found, rank, np.iinfo(np.int32).max).astype(np.int32)
        owner = np.empty_like(claim)
        self.comm.Allreduce(claim, owner, op=MPI.MIN)
        missing = np.nonzero(owner == np.iinfo(np.int32).max)[0]
        if missing.size:
            msg = ("%d observation target(s) are outside the mesh, e.g. %s"
                   % (missing.size, self.targets[missing[:3]].tolist()))
            if missing_policy == "raise":
                raise ValueError(msg)
            elif missing_policy == "warn" and rank == 0:
                print("warning: " + msg, flush=True)
        mine = np.nonzero(owner == rank)[0]
        self.owned_targets = mine                     # global target indices
        self.n_owned = int(mine.size)
        self.owner = owner

        # rows of the local observation matrix, over local dofs
        nldof = fes.GetVSize()
        rows, cols, vals = [], [], []
        for k, t in enumerate(mine):
            e = int(ids[t])
            fe = fes.GetFE(e)
            nd = fe.GetDof()
            sh = mfem.Vector(nd)
            fe.CalcShape(ips[int(t)], sh)
            shape = np.array(sh.GetDataArray(), copy=True)
            vd = np.asarray(fes.GetElementVDofs(e), dtype=np.int64)
            neg = vd < 0
            idx = np.where(neg, -1 - vd, vd)
            sgn = np.where(neg, -1.0, 1.0)
            idx = idx.reshape(self.space.vdim, nd)
            sgn = sgn.reshape(self.space.vdim, nd)
            w = self.components[t]
            for c in range(self.space.vdim):
                if w[c] == 0.0:
                    continue
                rows.append(np.full(nd, k, dtype=np.int64))
                cols.append(idx[c])
                vals.append(w[c] * shape * sgn[c])
        if rows:
            self._rows = np.concatenate(rows)
            self._cols = np.concatenate(cols)
            self._vals = np.concatenate(vals)
        else:
            self._rows = np.zeros(0, np.int64)
            self._cols = np.zeros(0, np.int64)
            self._vals = np.zeros(0)
        self._nldof = nldof

    # ---------------------------------------------------------------- interface
    def init_vector(self, x, dim):
        tpl = self._range_template if dim == 0 else self._domain_template
        return init_vector_like(x, tpl)

    def mult(self, u, obs):
        """``obs = B u``: evaluate the field at the owned targets."""
        local = self.space.local_values(u)
        obs.zero()
        if self._rows.size:
            np.add.at(obs.array, self._rows, self._vals * local[self._cols])
        return obs

    def multTranspose(self, obs, u):
        """``u = B^T obs``: spread observation weights back to the dofs."""
        local = np.zeros(self._nldof)
        if self._rows.size:
            np.add.at(local, self._cols, self._vals * obs.array[self._rows])
        return self.space.assemble_dual(local, u)

    transpmult = multTranspose

    def getSize(self):
        return self.ntargets

    def getComm(self):
        return self.comm

    def createVecLeft(self):
        return self._range_template.duplicate()

    def createVecRight(self):
        return self._domain_template.duplicate()

    # -------------------------------------------------------------- reordering
    def gather(self, obs):
        """Values at every target, in the order of ``targets``, on every rank."""
        idx = np.concatenate(self.comm.allgather(self.owned_targets))
        val = np.concatenate(self.comm.allgather(obs.array))
        out = np.zeros(self.ntargets)
        out[idx] = val
        return out

    def scatter(self, values, obs=None):
        """Fill an observation vector from an array in ``targets`` order."""
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != self.ntargets:
            raise ValueError("expected %d values, got %d"
                             % (self.ntargets, values.size))
        if obs is None:
            obs = self.createVecLeft()
        obs.array[:] = values[self.owned_targets]
        return obs

    # ------------------------------------------------------- synthetic noise
    def noise(self, sigma=1.0, rng=None, out=None):
        """Observation noise keyed on the **global** target index.

        Drawing in the partition-dependent local order would make synthetic data,
        and with it the MAP point, depend on the rank count.  Targets are few, so
        every rank draws the whole global vector and keeps its own entries.
        """
        from ..common.random import parRandom

        rng = rng if rng is not None else parRandom
        full = rng.normal_block(sigma, 0, self.ntargets)
        rng.advance(self.ntargets)
        if out is None:
            out = self.createVecLeft()
        out.array[:] = full[self.owned_targets]
        return out

    def perturb(self, obs, sigma, rng=None):
        """Add partition-independent noise of standard deviation ``sigma``."""
        return obs.axpy(1.0, self.noise(sigma, rng))

    def observe(self, u, rel_noise=None, sigma=None, rng=None):
        """Synthetic data from a state: ``B u`` plus noise; returns ``(data, sigma)``.

        The noise standard deviation is ``sigma``, or ``rel_noise`` times the
        largest observed value; it is returned because the misfit needs it as the
        noise variance.  ``rel_noise=0`` or neither given returns clean data with
        ``sigma = 0``.
        """
        data = self.createVecLeft()
        self.mult(u, data)
        if sigma is None:
            sigma = (float(rel_noise) * max(data.norm("linf"), 1e-30)
                     if rel_noise else 0.0)
        if sigma > 0.0:
            self.perturb(data, sigma, rng)
        return data, sigma

    def __repr__(self):
        return "PointwiseObservation(%d targets, %d owned here)" % (
            self.ntargets, self.n_owned)


def assemblePointwiseObservation(Vh, targets, **kw):
    """Observation operator evaluating the state at ``targets``."""
    return PointwiseObservation(Vh, targets, **kw)


def assemblePointwiseLOSObservation(Vh, targets, directions):
    """Line-of-sight observation: the component of a vector field along a direction."""
    return PointwiseObservation(Vh, targets, components=directions)


def exportPointwiseObservation(B, data, filename, comm=None):
    """Write targets and observed values as CSV on rank 0."""
    comm = comm if comm is not None else B.comm
    vals = B.gather(data)
    if comm.rank != 0:
        return
    arr = np.column_stack((B.targets, vals.reshape(-1, 1)))
    header = ",".join(["x", "y", "z"][: B.targets.shape[1]]) + ",value"
    np.savetxt(filename, arr, delimiter=",", header=header, comments="")


#: snake_case spellings (see :mod:`hippymfem.common.naming`)
assemble_pointwise_observation = assemblePointwiseObservation
assemble_pointwise_los_observation = assemblePointwiseLOSObservation
export_pointwise_observation = exportPointwiseObservation

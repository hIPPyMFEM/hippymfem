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
r"""Time-dependent forward problems from a one-step residual density.

The user writes the residual of a single time step,

.. math:: r_n(u_n, u_{n-1}, m, p_n; x, t_n, \Delta t) = 0,

as a pointwise density.  Implicit Euler for a parabolic problem is

.. math:: r_n = \frac{u_n - u_{n-1}}{\Delta t}\,p_n + a(u_n, m; p_n) - f(t_n)\,p_n .

Because the previous level :math:`u_{n-1}` is a **differentiation slot** of the
kernel and not a frozen coefficient, every operator the space-time inverse
problem needs is an exact derivative of that one density:

================================================  =================================
block                                             what it is
================================================  =================================
:math:`A_n = \partial_{u_n}\partial_{p}r_n`        the step Jacobian
:math:`B_n = \partial_{u_{n-1}}\partial_{p}r_n`    coupling to the level before
:math:`C_n = \partial_{m}\partial_{p}r_n`          the control block
:math:`W^{(n)}_{ab}`                              second derivatives in
                                                  :math:`a,b \in \{u_n, u_{n-1}, m\}`
================================================  =================================

The forward problem marches :math:`A_n u_n = \ldots` forward; the adjoint marches
:math:`A_n^{\!\top} p_n = \text{rhs}_n - B_{n+1}^{\!\top} p_{n+1}` **backward**,
the :math:`B^{\!\top}` term being exactly where reverse-in-time coupling comes
from.  The second-order blocks are accumulated step by step, each step writing
into the two time levels it touches.

**Operators are built once per trajectory: one per step, or one for all steps
when they do not change in time.**  A step Jacobian that is linear in :math:`u_n`
with coefficients independent of :math:`t` (implicit Euler for a linear PDE, the
common case) is the same matrix at every step.  Nothing here assumes that: the
first two steps are assembled and compared through a random matrix-vector
product, and when they agree one :math:`A`, one :math:`A^{\!\top}`, one :math:`B`
and one solver with its multigrid hierarchy serve the whole march.  Otherwise
every step keeps its own, built once at the linearization point.  Either way no
operator is assembled or a hierarchy set up inside a solve loop, where a step-wise
implementation would set BoomerAMG up once per step inside every Newton iteration
of the forward solve and every CG iteration of the incremental solves.  The blocks
of a linearization point come from one differentiation pass per step rather than
one per block.

The trajectory lives in a :class:`~.timeDependentVector.TimeDependentVector`; the
time and step size reach the kernel as traced scalars, so the compiled kernels
are reused across all steps.
"""

import numpy as np

from ..common.keepalive import KeepAlive
from ..fem.assemble import assemble_matrix, assemble_vector
from ..fem.bcs import as_bcset
from ..fem.elementbatch import default_quadrature_degree, get_batches
from ..fem.spaces import as_space
from .PDEProblem import PDEProblem
from .PDEVariationalProblem import (_same, _set_operator_once, require_finite,
                                    require_finite_arrays)
from .timeDependentVector import TimeDependentVector
from .variables import ADJOINT, NVAR, PARAMETER, STATE
from ..common.random import Random
from ..fem.io import ParaViewWriter
from ..fem.assemble import assembly_backend
from ..fem.boundary import (assemble_boundary_matrix, assemble_boundary_vector,
                            get_boundary_batches)
from ..fem.csrassemble import _eliminate, assemble_matrix_csr

#: kernel field slots of the one-step residual
NEW = 0          #: u_n
OLD = 1          #: u_{n-1}
PARAM = 2        #: m
ADJ = 3          #: p_n

#: The blocks of one step: name -> (row slot, column slot, eliminate test rows,
#: diagonal policy).  ``A`` and the ``W`` blocks on the state get the essential rows
#: (and columns, being square) eliminated as the steady problem does; ``A`` keeps a
#: unit diagonal so the forward solve works on a right-hand side with zeroed
#: essential entries, the ``W`` blocks a zero one so they annihilate that subspace.
_BLOCKS = {
    "A": (ADJ, NEW, True, "one"),
    "B": (ADJ, OLD, True, "one"),
    "C": (ADJ, PARAM, True, "one"),
    "Wnn": (NEW, NEW, True, "zero"),
    "Wno": (NEW, OLD, True, "one"),
    "Woo": (OLD, OLD, True, "zero"),
    "Wnm": (NEW, PARAM, True, "one"),
    "Wom": (OLD, PARAM, True, "one"),
    "Wmm": (PARAM, PARAM, False, "one"),
}
_FORWARD_BLOCKS = ("A", "B")
_HESSIAN_BLOCKS = ("Wnn", "Wno", "Woo", "Wnm", "Wom", "Wmm")


class TimeDependentPDEVariationalProblem(PDEProblem, KeepAlive):
    """One-step time-dependent forward problem.

    Parameters
    ----------
    Vh : sequence of 3 spaces
        ``[STATE, PARAMETER, ADJOINT]``, the spatial spaces for one time level.
    varf_handler : callable
        ``varf(u, u_old, m, p, x, t, dt) -> scalar``, JAX-traceable, with
        :class:`~hippymfem.fem.kernel.Field` arguments for the four fields.
    bdr_varf : callable, optional
        ``bdr_varf(u, u_old, m, p, x, n, t, dt) -> scalar``, a boundary density of
        the same step residual with the outward unit normal ``n``, integrated over
        the boundary elements of ``bdr_attributes`` and added to the domain term:
        a Robin condition, a prescribed flux, a boundary source.  Its derivative
        blocks are added to the step blocks before the essential rows are
        eliminated, as in :class:`~.PDEVariationalProblem.PDEVariationalProblem`.
    bdr_attributes : "all" or sequence of int
        Which boundary attributes ``bdr_varf`` applies to (MFEM's 1-based
        attributes); all of them by default.
    bdr_quadrature_degree : int, optional
        Quadrature degree on the boundary; ``quadrature_degree`` by default.
    bc, bc0 : boundary conditions on the state, with data and homogeneous.
    u0 : ParVector
        Initial condition at ``t_init``.
    t_init, t_final, dt : float
    is_fwd_linear : bool
        The step residual is linear in ``u_n``; one Newton step is then exact.
    spd_jacobian : bool
        The step Jacobian is symmetric positive definite; the default solvers are
        then CG with BoomerAMG (see :attr:`~.PDEProblem.PDEProblem.spd_jacobian`).
    """

    #: relative tolerance of the "same operator" probe (see ``_same_operator``)
    INVARIANCE_PROBE_TOL = 1e-13

    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final, dt,
                 is_fwd_linear=False, quadrature_degree=None, spd_jacobian=False,
                 bdr_varf=None, bdr_attributes="all", bdr_quadrature_degree=None):
        if len(Vh) != NVAR:
            raise ValueError("Vh must have %d entries" % NVAR)
        #: see :attr:`~.PDEProblem.PDEProblem.spd_jacobian`
        self.spd_jacobian = bool(spd_jacobian)
        self.Vh = [as_space(v) for v in Vh]
        self.varf = varf_handler
        self.comm = self.Vh[STATE].comm
        self.mesh = self.Vh[STATE].mesh
        self.u0 = u0
        self.t_init = float(t_init)
        self.t_final = float(t_final)
        self.dt = float(dt)
        self.times = np.arange(self.t_init, self.t_final + 0.5 * self.dt, self.dt)
        self.nt = self.times.size
        self.is_fwd_linear = bool(is_fwd_linear)

        self.bc = as_bcset(bc, self.Vh[STATE])
        self.bc0 = as_bcset(bc0 if bc0 is not None else
                            (self.bc.homogeneous() if self.bc else None),
                            self.Vh[STATE])

        if quadrature_degree is None:
            quadrature_degree = default_quadrature_degree(self.Vh)
        self.quadrature_degree = int(quadrature_degree)
        self.batches = get_batches(self.mesh, self.quadrature_degree)
        self.nelem = self.mesh.GetNE()

        from ..fem.kernel import QuadratureKernel

        #: slot spaces: (u_n, u_{n-1}, m, p_n)
        self.slots = [self.Vh[STATE], self.Vh[STATE], self.Vh[PARAMETER],
                      self.Vh[ADJOINT]]
        self.kernel = QuadratureKernel(varf_handler, self.slots, self.batches,
                                       nparams=2)

        # ------------------------------------------------------ boundary term
        self.bdr_varf = bdr_varf
        self.bdr_kernel = None
        self.bdr_batches = None
        if bdr_varf is not None:
            from ..fem.kernel import BoundaryKernel

            self.bdr_quadrature_degree = int(
                bdr_quadrature_degree if bdr_quadrature_degree is not None
                else self.quadrature_degree)
            self.bdr_batches = get_boundary_batches(
                self.mesh, self.bdr_quadrature_degree, bdr_attributes, self.comm,
                space=self.Vh[STATE])
            self.bdr_kernel = BoundaryKernel(bdr_varf, self.slots, self.bdr_batches,
                                             nparams=2)

        # the four solvers (see PDEProblem) start unset and default on first use;
        # per-step solvers, for a step Jacobian that changes in time, live in the
        # trajectory cache and are cloned from these
        self.solver = self.solver_adj = None
        self.solver_fwd_inc = self.solver_adj_inc = None
        self.linearize_x = None
        self.gauss_newton_approx = False
        self.newton_parameters = {
            "rel_tolerance": 1e-10, "abs_tolerance": 1e-14,
            "max_iter": 25, "print_level": -1,
        }
        self.n_calls = {"forward": 0, "adjoint": 0,
                        "incremental_forward": 0, "incremental_adjoint": 0}
        #: the operators of the trajectory the adjoint and the Hessian work on
        self._traj = None
        #: the step Jacobian of the last *linear* forward solve, keyed on the parameter
        self._fwd_jac = None
        self._probe_rng = None

    # ------------------------------------------------------------------ shapes
    def generate_state(self):
        return TimeDependentVector(self.times, comm=self.comm).initialize(
            self.Vh[STATE])

    def generate_adjoint(self):
        return TimeDependentVector(self.times, comm=self.comm).initialize(
            self.Vh[ADJOINT])

    def generate_parameter(self):
        return self.Vh[PARAMETER].vector()

    def generate_static_state(self):
        return self.Vh[STATE].vector()

    def generate_static_adjoint(self):
        return self.Vh[ADJOINT].vector()

    def init_parameter(self, m):
        return m

    # ------------------------------------------------------------------ kernel
    def _slot_locals(self, u_new, u_old, m, p):
        return [self.Vh[STATE].local_values(u_new),
                self.Vh[STATE].local_values(u_old),
                self.Vh[PARAMETER].local_values(m),
                self.Vh[ADJOINT].local_values(p)]

    def _step_residual(self, slot, u_new, u_old, m, p, t, ess=None, out=None):
        """``d r_n / d(slot)`` as a true-dof vector."""
        loc = self._slot_locals(u_new, u_old, m, p)
        vecs = self.kernel.element_vectors(slot, loc, (t, self.dt))
        require_finite_arrays(vecs, self.comm, "the step residual")
        space = self.slots[slot]
        if self.bdr_kernel is None:
            v = assemble_vector(space, self.batches.groups, vecs, self.nelem, ess=ess)
        else:
            # the boundary part joins before the essential rows are zeroed
            v = assemble_vector(space, self.batches.groups, vecs, self.nelem)
            bvecs = self.bdr_kernel.element_vectors(slot, loc, (t, self.dt))
            require_finite_arrays(bvecs, self.comm, "the boundary step residual")
            v.axpy(1.0, assemble_boundary_vector(space, self.bdr_batches.groups,
                                                 bvecs))
            if ess is not None and len(ess):
                v.array[np.asarray(ess, dtype=np.int64)] = 0.0
        if out is not None:
            out.assign(v)
            return out
        return v

    def _step_block(self, i, j, u_new, u_old, m, p, t, test_ess=None,
                    diag_policy="one"):
        """``d^2 r_n / d(slot i) d(slot j)`` as a matrix (one block, one pass)."""
        loc = self._slot_locals(u_new, u_old, m, p)
        mats = self.kernel.element_matrices(i, j, loc, (t, self.dt))
        require_finite_arrays(mats, self.comm, "step block (%d, %d)" % (i, j))
        if self.bdr_kernel is None:
            return assemble_matrix(self.slots[i], self.slots[j], self.batches.groups,
                                   mats, self.nelem, test_ess=test_ess,
                                   diag_policy=diag_policy)
        bmats = self.bdr_kernel.element_matrices(i, j, loc, (t, self.dt))
        require_finite_arrays(bmats, self.comm,
                              "boundary step block (%d, %d)" % (i, j))
        return self._with_boundary(i, j, mats, bmats, test_ess, diag_policy)

    def _with_boundary(self, i, j, mats, bmats, test_ess, diag_policy):
        """Block ``(i, j)`` with its boundary part, the two summed before the
        essential rows are eliminated (eliminating each and adding would leave 2.0
        on the essential diagonal), as
        :meth:`.PDEVariationalProblem.PDEVariationalProblem._domain_block` does."""
        ti, tj = self.slots[i], self.slots[j]
        if assembly_backend() == "csr":
            return assemble_matrix_csr(
                ti, tj, self.batches.groups, mats, self.nelem, test_ess=test_ess,
                diag_policy=diag_policy,
                boundary=(self.bdr_batches.tables(ti), self.bdr_batches.tables(tj),
                          bmats))
        from ..common.linalg import ParAdd

        dom = assemble_matrix(ti, tj, self.batches.groups, mats, self.nelem)
        bdr = assemble_boundary_matrix(ti, tj, self.bdr_batches.groups, bmats)
        A = ParAdd(dom, bdr)
        del dom, bdr
        return _eliminate(A, ti, tj, test_ess, None, diag_policy, ti.fes is tj.fes)

    def _step_blocks(self, names, u_new, u_old, m, p, t):
        """Several blocks of one step from **one** differentiation pass.

        The blocks are slices of the same element Hessian, so differentiating once
        over the column slots they use and assembling each from its slice gives
        the same matrices as one pass per block, for a fraction of the kernel time.
        """
        names = tuple(names)
        pairs = [(_BLOCKS[n][0], _BLOCKS[n][1]) for n in names]
        loc = self._slot_locals(u_new, u_old, m, p)
        mats = self.kernel.element_matrices_many(pairs, loc, (t, self.dt))
        bmats = None
        if self.bdr_kernel is not None:
            bmats = self.bdr_kernel.element_matrices_many(pairs, loc, (t, self.dt))
        out = {}
        ess = self.bc0.ess_tdof
        for n, ij in zip(names, pairs):
            require_finite_arrays(mats[ij], self.comm, "step block %s" % n)
            _, _, elim, policy = _BLOCKS[n]
            if bmats is None:
                out[n] = assemble_matrix(self.slots[ij[0]], self.slots[ij[1]],
                                         self.batches.groups, mats[ij], self.nelem,
                                         test_ess=ess if elim else None,
                                         diag_policy=policy)
            else:
                require_finite_arrays(bmats[ij], self.comm,
                                      "boundary step block %s" % n)
                out[n] = self._with_boundary(ij[0], ij[1], mats[ij], bmats[ij],
                                             ess if elim else None, policy)
        return out

    # ------------------------------------------------------------ invariance
    def _same_operator(self, A, B):
        """Whether two step operators are the same matrix, to round-off.

        Compared through their action on a random vector from a private
        counter-based stream (every rank draws the same one, so the verdict is
        partition independent), against ``|A v| + |B v|``; the cost is two
        matrix-vector products, against the assembly and multigrid setup it
        saves at every later step.
        """
        if A.Height() != B.Height() or A.Width() != B.Width():
            return False

        if self._probe_rng is None:
            self._probe_rng = Random(seed=20260915)
        rng = self._probe_rng
        rng.set_seed(rng.seed)
        v = self.Vh[STATE].vector()
        rng.normal(1.0, v)
        Av, Bv = self.Vh[STATE].vector(), self.Vh[STATE].vector()
        A.Mult(v.hypre, Av.hypre)
        B.Mult(v.hypre, Bv.hypre)
        bound = Av.norm("l2") + Bv.norm("l2")
        if bound == 0.0:
            return True
        Av.axpy(-1.0, Bv)
        return Av.norm("l2") <= self.INVARIANCE_PROBE_TOL * bound

    # ------------------------------------------------------------------ forward
    def solveFwd(self, out, x):
        """March the state forward, storing every level in ``out``.

        For a residual linear in ``u_n`` the step Jacobian is assembled at the
        first step and again at the second; when the two agree it serves every
        later step and the forward solver keeps one hierarchy for the whole
        march.  That Jacobian is also kept across forward solves at the same
        parameter.
        """
        self.n_calls["forward"] += 1
        m = x[PARAMETER]
        require_finite(m, "the parameter passed to solveFwd")
        out.zero()
        out.store(self.u0, self.times[0])
        u_old = self.u0.copy()
        u = self.Vh[STATE].vector()
        p0 = self.Vh[ADJOINT].vector()
        du = self.Vh[STATE].vector()
        solver = self._get_solver("solver")
        prm = self.newton_parameters
        maxit = 1 if self.is_fwd_linear else int(prm["max_iter"])

        # A linear step: one Jacobian per parameter when it is time invariant.
        # ``jac`` is (matrix, verdict) with verdict None (one step seen), True
        # (the same at step two: reuse from here on) or False (varies: assemble).
        jac = None
        if self.is_fwd_linear and self._fwd_jac is not None \
                and _same(self._fwd_jac[0], m):
            jac = (self._fwd_jac[1], True)
        else:
            self._fwd_jac = None
            if self.is_fwd_linear:
                self._release_operators("solver")

        for k in range(1, self.nt):
            t = self.times[k]
            u.assign(u_old)
            self.bc.apply(u)
            r = self._step_residual(ADJ, u, u_old, m, p0, t, ess=self.bc0.ess)
            r0 = r.norm("l2")
            tol = max(prm["rel_tolerance"] * r0, prm["abs_tolerance"])
            converged = False
            for _ in range(maxit):
                if jac is not None and jac[1]:
                    J = jac[0]
                else:
                    J = self._step_block(ADJ, NEW, u, u_old, m, p0, t,
                                         test_ess=self.bc0.ess_tdof,
                                         diag_policy="one")
                    if self.is_fwd_linear:
                        if jac is None:
                            jac = (J, None)
                        elif jac[1] is None:
                            same = self._same_operator(jac[0], J)
                            if same:
                                J = jac[0]                 # keep the first, drop this one
                            jac = (jac[0], same)
                _set_operator_once(solver, J)
                r.scale(-1.0)
                solver.solve(du, r)
                u.axpy(1.0, du)
                r = self._step_residual(ADJ, u, u_old, m, p0, t, ess=self.bc0.ess)
                if r.norm("l2") < tol:
                    converged = True
                    break
            if not converged and not self.is_fwd_linear:
                raise RuntimeError(
                    "forward Newton did not converge at t = %g: ||r|| = %.3e "
                    "(started at %.3e)" % (t, r.norm("l2"), r0))
            if self.is_fwd_linear and r.norm("l2") > max(1e-6 * max(r0, 1.0), 1e-8):
                raise RuntimeError(
                    "is_fwd_linear=True but one Newton step left ||r|| = %.3e at "
                    "t = %g; the step residual is not linear in the new level"
                    % (r.norm("l2"), t))
            out.store(u, t)
            u_old.assign(u)
        if jac is not None and jac[1]:
            self._fwd_jac = (m.copy(), jac[0])
        return out

    # -------------------------------------------------------------- trajectory
    def _trajectory(self, x, extra=()):
        """The step operators along the trajectory ``x``, built once per point.

        Returns a dict holding, per block name, a list indexed by step: ``A``,
        ``At`` and ``B`` always, plus the blocks named in ``extra`` (``C`` and the
        ``W`` blocks for a linearization point).  When the step operators do not
        change in time the ``A``, ``At`` and ``B`` lists hold one shared matrix
        each.  The comparison is made at the second step; a match there is taken
        to hold for the whole march, which is what a time-invariant residual
        guarantees and a time-varying one fails at the first opportunity.

        The cache is reused while ``x`` is the point it was built for, so the
        adjoint solve and the linearization point that follows it share one
        assembly of ``A`` and ``B``; blocks it lacks are added in one pass per
        step.  The point is remembered as a copy of ``m`` and of the trajectory
        (``nt`` vectors, small next to the ``9 nt`` matrices of a full point).
        """
        extra = tuple(extra)
        tr = self._traj
        if tr is not None and self._point_matches(tr, x):
            missing = tuple(n for n in extra if n not in tr)
            if missing:
                for n in missing:
                    tr[n] = [None] * self.nt
                for k in range(1, self.nt):
                    for n, M in self._step_blocks(missing, *self._at(x, k)).items():
                        tr[n][k] = M
            return tr
        self._traj = None
        self._release_operators("solver_adj", "solver_fwd_inc", "solver_adj_inc")
        names = _FORWARD_BLOCKS + extra
        tr = {"m": x[PARAMETER].copy(), "u": x[STATE].copy(), "invariant": False,
              "fwd_inc": [None] * self.nt, "adj_inc": [None] * self.nt}
        tr.update({n: [None] * self.nt for n in names + ("At",)})
        invariant = None
        for k in range(1, self.nt):
            want = names if not invariant else extra
            blocks = self._step_blocks(want, *self._at(x, k)) if want else {}
            if invariant is None and k >= 2:
                invariant = (self._same_operator(tr["A"][1], blocks["A"])
                             and self._same_operator(tr["B"][1], blocks["B"]))
                if invariant:
                    for n in _FORWARD_BLOCKS:
                        blocks.pop(n)
            for n, M in blocks.items():
                tr[n][k] = M
        tr["invariant"] = True if self.nt <= 2 else bool(invariant)
        if tr["invariant"]:
            A, B = tr["A"][1], tr["B"][1]
            At = A.Transpose()
            for k in range(1, self.nt):
                tr["A"][k], tr["B"][k], tr["At"][k] = A, B, At
        else:
            for k in range(1, self.nt):
                tr["At"][k] = tr["A"][k].Transpose()
        self._traj = tr
        return tr

    def _at(self, x, k):
        """The arguments of step ``k``: ``(u_k, u_{k-1}, m, p_k, t_k)``."""
        t = self.times[k]
        p = x[ADJOINT].view(t) if (len(x) > ADJOINT and x[ADJOINT] is not None) \
            else self.Vh[ADJOINT].vector()
        return (x[STATE].view(t), x[STATE].view(self.times[k - 1]), x[PARAMETER], p, t)

    def _point_matches(self, tr, x):
        """Whether the cached trajectory operators were built at this point."""
        if not _same(tr["m"], x[PARAMETER]):
            return False
        return all(_same(v, w) for v, w in zip(tr["u"], x[STATE]))

    def _step_solver(self, tr, kind, k):
        """The solver holding step ``k``'s operator; one for all steps when invariant.

        ``kind`` is ``"fwd_inc"`` or ``"adj_inc"``.  The user's solver attribute is
        the template: it takes the operator itself when one serves every step, and
        is cloned per step otherwise, so that no ``set_operator`` (for BoomerAMG,
        the expensive part) ever happens inside a solve loop.
        """
        attr = "solver_" + kind
        template = self._get_solver(attr)
        op = tr["A" if kind == "fwd_inc" else "At"][k]
        if tr["invariant"]:
            _set_operator_once(template, op)
            return template
        s = tr[kind][k]
        if s is None:
            s = tr[kind][k] = self._clone_solver(template)
            s.set_operator(op)
        return s

    # ------------------------------------------------------------------ adjoint
    def solveAdj(self, out, x, adj_rhs):
        r"""March the adjoint backwards.

        :math:`A_n^{\!\top}p_n = \text{rhs}_n - B_{n+1}^{\!\top} p_{n+1}`, with the
        step operators of the trajectory ``x`` built once (see ``_trajectory``) and
        the adjoint solver holding one transposed Jacobian for the whole march
        when the steps agree.
        """
        self.n_calls["adjoint"] += 1
        tr = self._trajectory(x)
        if self.solver_adj is None and self.solver is not None:
            # a user who set one solver expects the adjoint solved the same way
            self.solver_adj = self._clone_solver(self.solver)
        solver = self._get_solver("solver_adj")
        out.zero()
        p = self.Vh[ADJOINT].vector()
        p_next = self.Vh[ADJOINT].vector()
        rhs = self.Vh[STATE].vector()
        coupling = self.Vh[STATE].vector()
        for k in range(self.nt - 1, 0, -1):
            t = self.times[k]
            rhs.assign(adj_rhs.view(t))
            if k < self.nt - 1:
                tr["B"][k + 1].MultTranspose(p_next.hypre, coupling.hypre)
                rhs.axpy(-1.0, coupling)
            self.bc0.zero(rhs)
            _set_operator_once(solver, tr["At"][k])
            p.zero()
            solver.solve(p, rhs)
            out.store(p, t)
            p_next.assign(p)
        return out

    def evalGradientParameter(self, x, out):
        r"""``out`` = :math:`\sum_n \partial_m r_n`."""
        out.zero()
        m = x[PARAMETER]
        for k in range(1, self.nt):
            t = self.times[k]
            g = self._step_residual(PARAM, x[STATE].view(t),
                                    x[STATE].view(self.times[k - 1]), m,
                                    x[ADJOINT].view(t), t)
            out.axpy(1.0, g)
        return out

    # ----------------------------------------------------- linearization point
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """Build (or complete) the step operators at ``x`` and point the incremental
        solvers at them, once.

        With Gauss-Newton the ``W`` blocks are not needed and not assembled; a
        trajectory that already has them keeps them.
        """
        self.linearize_x = [x[STATE], x[PARAMETER], x[ADJOINT]]
        self.gauss_newton_approx = bool(gauss_newton_approx)
        extra = ("C",) if self.gauss_newton_approx else ("C",) + _HESSIAN_BLOCKS
        tr = self._trajectory(x, extra)
        for k in range(1, self.nt):
            self._step_solver(tr, "fwd_inc", k)
            self._step_solver(tr, "adj_inc", k)
        return self

    def _blocks(self, k):
        """The blocks of step ``k`` as a dict (``A``, ``At``, ``B``, ``C`` and,
        after a full Newton linearization, the ``W`` blocks)."""
        tr = self._traj
        if tr is None or "C" not in tr:
            raise RuntimeError("setLinearizationPoint must be called first")
        return {n: tr[n][k] for n in ("A", "At", "B", "C") + _HESSIAN_BLOCKS
                if n in tr}

    # ----------------------------------------------------------- incremental
    def solveIncremental(self, out, rhs, is_adj):
        r"""Solve the space-time incremental system.

        Forward (``is_adj=False``): march :math:`A_n \hat u_n = \text{rhs}_n -
        B_n \hat u_{n-1}` from :math:`\hat u_0 = 0` (the initial condition does
        not depend on the parameter).

        Adjoint (``is_adj=True``): march
        :math:`A_n^{\!\top}\hat p_n = \text{rhs}_n - B_{n+1}^{\!\top}\hat p_{n+1}`
        backward.  The solvers were pointed at their operators by
        :meth:`setLinearizationPoint`; nothing is set up here.
        """
        tr = self._traj
        if tr is None or "C" not in tr:
            raise RuntimeError("setLinearizationPoint must be called first")
        out.zero()
        work = self.Vh[STATE].vector()
        r = self.Vh[STATE].vector()
        if not is_adj:
            self.n_calls["incremental_forward"] += 1
            prev = self.Vh[STATE].vector()          # u_hat_0 = 0
            cur = self.Vh[STATE].vector()
            for k in range(1, self.nt):
                r.assign(rhs.view(self.times[k]))
                tr["B"][k].Mult(prev.hypre, work.hypre)
                r.axpy(-1.0, work)
                self.bc0.zero(r)
                cur.zero()
                self._step_solver(tr, "fwd_inc", k).solve(cur, r)
                out.store(cur, self.times[k])
                prev.assign(cur)
        else:
            self.n_calls["incremental_adjoint"] += 1
            nxt = self.Vh[ADJOINT].vector()          # p_hat_{N+1} = 0
            cur = self.Vh[ADJOINT].vector()
            for k in range(self.nt - 1, 0, -1):
                r.assign(rhs.view(self.times[k]))
                if k < self.nt - 1:
                    tr["B"][k + 1].MultTranspose(nxt.hypre, work.hypre)
                    r.axpy(-1.0, work)
                self.bc0.zero(r)
                cur.zero()
                self._step_solver(tr, "adj_inc", k).solve(cur, r)
                out.store(cur, self.times[k])
                nxt.assign(cur)
        return out

    # --------------------------------------------------------------- blocks
    def applyC(self, dm, out):
        """``out_n = C_n dm`` at every level."""
        out.zero()
        for k in range(1, self.nt):
            self._blocks(k)["C"].Mult(dm.hypre, out.view(self.times[k]).hypre)
        return out

    def applyCt(self, dp, out):
        r"""``out`` = :math:`\sum_n C_n^{\!\top} dp_n`."""
        out.zero()
        tmp = self.Vh[PARAMETER].vector()
        for k in range(1, self.nt):
            self._blocks(k)["C"].MultTranspose(dp.view(self.times[k]).hypre,
                                               tmp.hypre)
            out.axpy(1.0, tmp)
        return out

    def applyWuu(self, du, out):
        r"""State-state block of the Lagrangian, including the time coupling."""
        out.zero()
        if self.gauss_newton_approx:
            return out
        work = self.Vh[STATE].vector()
        for k in range(1, self.nt):
            blk = self._blocks(k)
            t, t_prev = self.times[k], self.times[k - 1]
            dn, do = du.view(t), du.view(t_prev)
            # row n gets W[new,new] du_n + W[new,old] du_{n-1}
            blk["Wnn"].Mult(dn.hypre, work.hypre)
            out.view(t).axpy(1.0, work)
            blk["Wno"].Mult(do.hypre, work.hypre)
            out.view(t).axpy(1.0, work)
            # row n-1 gets W[old,new] du_n + W[old,old] du_{n-1}
            blk["Wno"].MultTranspose(dn.hypre, work.hypre)
            out.view(t_prev).axpy(1.0, work)
            blk["Woo"].Mult(do.hypre, work.hypre)
            out.view(t_prev).axpy(1.0, work)
        out.view(self.times[0]).zero()       # the initial condition is fixed
        return out

    def applyWum(self, dm, out):
        r"""State rows of the state-parameter block."""
        out.zero()
        if self.gauss_newton_approx:
            return out
        work = self.Vh[STATE].vector()
        for k in range(1, self.nt):
            blk = self._blocks(k)
            blk["Wnm"].Mult(dm.hypre, work.hypre)
            out.view(self.times[k]).axpy(1.0, work)
            blk["Wom"].Mult(dm.hypre, work.hypre)
            out.view(self.times[k - 1]).axpy(1.0, work)
        out.view(self.times[0]).zero()
        return out

    def applyWmu(self, du, out):
        r"""Parameter row of the parameter-state block (the transpose of Wum)."""
        out.zero()
        if self.gauss_newton_approx:
            return out
        tmp = self.Vh[PARAMETER].vector()
        for k in range(1, self.nt):
            blk = self._blocks(k)
            blk["Wnm"].MultTranspose(du.view(self.times[k]).hypre, tmp.hypre)
            out.axpy(1.0, tmp)
            blk["Wom"].MultTranspose(du.view(self.times[k - 1]).hypre, tmp.hypre)
            out.axpy(1.0, tmp)
        return out

    def applyWmm(self, dm, out):
        out.zero()
        if self.gauss_newton_approx:
            return out
        tmp = self.Vh[PARAMETER].vector()
        for k in range(1, self.nt):
            self._blocks(k)["Wmm"].Mult(dm.hypre, tmp.hypre)
            out.axpy(1.0, tmp)
        return out

    _APPLY = {
        (ADJOINT, PARAMETER): "applyC",
        (PARAMETER, ADJOINT): "applyCt",
        (STATE, STATE): "applyWuu",
        (STATE, PARAMETER): "applyWum",
        (PARAMETER, STATE): "applyWmu",
        (PARAMETER, PARAMETER): "applyWmm",
    }

    def apply_ij(self, i, j, dir, out):
        name = self._APPLY.get((i, j))
        if name is None:
            raise ValueError("no block (%d, %d)" % (i, j))
        return getattr(self, name)(dir, out)

    def apply_ijk(self, i, j, k, x, jdir, kdir, out):
        raise NotImplementedError(
            "third derivatives of the time-dependent residual are not "
            "implemented; Newton-CG does not use them"
        )

    def release_linearization_point(self):
        """Drop the trajectory operators and the solvers' hold on them."""
        self._traj = None
        self.linearize_x = None
        self._release_operators("solver_adj", "solver_fwd_inc", "solver_adj_inc")
        return self

    # ------------------------------------------------------------------ output
    def exportState(self, u, basename, field="u"):
        """Write a trajectory as a ParaView time series."""

        w = ParaViewWriter(basename, self.mesh, {field: self.Vh[STATE]})
        for k, t in enumerate(self.times):
            if u[k] is not None:
                w.save({field: u[k]}, time=float(t), cycle=k)
        return basename


class ImplicitEulerTimeDependentPDEVariationalProblem(
        TimeDependentPDEVariationalProblem):
    """Alias kept for source compatibility with hIPPYlib.

    The base class already covers any one-step scheme, implicit Euler included;
    which scheme it is lives in the residual density the user writes.
    """

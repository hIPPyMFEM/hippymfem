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
"""Abstract interface for the forward problem.

The signatures match hIPPYlib's ``PDEProblem`` so that the algorithms, the
model, and user driver scripts port across with no changes.
"""

from ..common.naming import SnakeCamel, sync_spellings
from ..algorithms.linSolvers import KrylovSolver, KrylovSolver_ParameterList, LUSolver
from .variables import ADJOINT, PARAMETER, STATE, Variables



class PDEProblem(SnakeCamel):
    """Forward PDE, its adjoint, and the derivative blocks the Hessian needs.

    Besides the interface, this carries what every concrete problem needs the same
    way: the four linear solvers as assignable attributes with a common default,
    vector generation from the three shapes, and the release of a solver's operator
    before it is rebuilt.
    """

    #: linear solvers; assignable, as in hIPPYlib.  ``solver`` is the forward solve,
    #: ``solver_adj`` the adjoint's own (it shares the forward preconditioner when the
    #: Jacobian is symmetric), ``solver_fwd_inc``/``solver_adj_inc`` the incremental
    #: systems'.  ``None`` means "build the default on first use".
    solver = None
    solver_adj = None
    solver_fwd_inc = None
    solver_adj_inc = None

    def generate_state(self):
        """A vector in the shape of the state."""
        raise NotImplementedError

    def generate_parameter(self):
        """A vector in the shape of the parameter."""
        raise NotImplementedError

    def generate_adjoint(self):
        """A vector in the shape of the adjoint."""
        raise NotImplementedError

    def generate_vector(self, component="ALL"):
        """A vector, or the :class:`~.variables.Variables` triple, of the problem."""

        if component == "ALL":
            return Variables(self.generate_state(), self.generate_parameter(),
                             self.generate_adjoint())
        if component == STATE:
            return self.generate_state()
        if component == PARAMETER:
            return self.generate_parameter()
        if component == ADJOINT:
            return self.generate_adjoint()
        raise ValueError("unknown component %r" % (component,))

    # ------------------------------------------------------------------ solvers
    #: Declare ``dR/du`` symmetric positive definite.  The default solvers are then
    #: CG with BoomerAMG instead of GMRES: the same iterations at a lower cost per
    #: solve, markedly so with hypre on a device.  Off by default because the symmetry
    #: probe cannot tell definite from indefinite, and CG on an indefinite Jacobian
    #: fails rather than converging slowly.
    spd_jacobian = False

    def _default_solver(self):
        """Krylov + BoomerAMG at a tight tolerance: GMRES, or CG when
        :attr:`spd_jacobian` says the Jacobian allows it."""

        s = KrylovSolver(self.comm, method="cg" if self.spd_jacobian else "gmres",
                         precond="amg")
        s.parameters["rel_tolerance"] = 1e-12
        s.parameters["max_iter"] = 2000
        return s

    #: the solver attributes a problem holds: forward, adjoint, and the two
    #: incremental ones
    SOLVER_ATTRIBUTES = ("solver", "solver_adj", "solver_fwd_inc", "solver_adj_inc")

    def set_solvers(self, solver, *args, attributes=None, **kwargs):
        """Install one kind of linear solver for every solve this problem makes.

        ``solver`` is either a factory, such as
        :func:`~hippymfem.algorithms.linSolvers.auto_solver` or
        :class:`~hippymfem.algorithms.linSolvers.KrylovSolver`, called once per
        attribute with ``*args`` and the keywords the factory takes (each solve then
        gets a solver and hierarchy of its own), or a solver instance, which serves
        the forward solve and is cloned for the others.  Keywords that name solver
        parameters (``rel_tolerance``, ``max_iter``, ``amg_relax_type``, ...) are set
        on every solver that has them instead, so ``rel_tolerance=1e-13`` on a direct
        solver is ignored.  ``attributes`` restricts the install to some of
        :attr:`SOLVER_ATTRIBUTES` (the default is all four).  For example::

            pde.set_solvers(hm.auto_solver, Vu, comm, max_direct=0,
                            rel_tolerance=1e-13, max_iter=3000)

        Returns ``self``.
        """

        param_keys = set(KrylovSolver_ParameterList().keys())
        parameters = {k: v for k, v in kwargs.items() if k in param_keys}
        factory_kwargs = {k: v for k, v in kwargs.items() if k not in param_keys}
        # an instance solves (it has ``solve``); a class or a function makes one
        template = (solver if hasattr(solver, "solve") and not isinstance(solver, type)
                    else None)
        if template is not None and factory_kwargs:
            raise TypeError("set_solvers: %s are factory keywords, but a solver "
                            "instance was given" % sorted(factory_kwargs))
        attrs = tuple(attributes) if attributes is not None else self.SOLVER_ATTRIBUTES
        for attr in attrs:
            if attr not in self.SOLVER_ATTRIBUTES:
                raise ValueError("unknown solver attribute %r; one of %s"
                                 % (attr, self.SOLVER_ATTRIBUTES))
            if template is None:
                s = solver(*args, **factory_kwargs)
            elif attr == attrs[0]:
                s = template
            else:
                s = self._clone_solver(template)
            for k, v in parameters.items():
                try:
                    if k in s.parameters:
                        s.parameters[k] = v
                except (AttributeError, TypeError):
                    pass
            setattr(self, attr, s)
        return self

    def _get_solver(self, attr):
        """The named solver, building the default the first time it is asked for."""
        s = getattr(self, attr)
        if s is None:
            s = self._default_solver()
            setattr(self, attr, s)
        return s

    def _clone_solver(self, template):
        """A fresh solver with ``template``'s kind and parameters, holding no operator.

        For problems that need one solver per operator (a time-dependent problem
        whose step Jacobian changes from step to step) where the user has set a
        single template.  Kinds this cannot rebuild fall back to the default.
        """

        if isinstance(template, KrylovSolver):
            s = KrylovSolver(template.comm, method=template.method,
                             precond=template.precond_type,
                             systems_dim=template.systems_dim,
                             elasticity=template.elasticity, fes=template.fes)
        elif isinstance(template, LUSolver):
            s = LUSolver(template.comm, method=template.method,
                         max_global_size=template.max_global_size)
        else:
            s = self._default_solver()
        try:
            for k in template.parameters.keys():
                s.parameters[k] = template.parameters[k]
        except Exception:                                   # noqa: BLE001
            pass
        return s

    def _release_operators(self, *attrs):
        """Let the named solvers drop their operators before a rebuild.

        Called before the matrix a solver holds is replaced, so the old matrix and
        its AMG hierarchy are freed before the new ones are allocated and a Newton
        step never holds two of each at its peak.
        """
        for a in attrs:
            rel = getattr(getattr(self, a, None), "release", None)
            if rel is not None:
                rel()

    def init_parameter(self, m):
        """Resize ``m`` to the shape of the parameter."""
        raise NotImplementedError

    def solveFwd(self, state, x):
        r"""Solve the forward problem: given :math:`m`, find :math:`u` with
        :math:`\delta_p F(u,m,p;\hat p) = 0` for all :math:`\hat p`."""
        raise NotImplementedError

    def solveAdj(self, adj, x, adj_rhs):
        r"""Solve the adjoint problem: given :math:`u, m`, find :math:`p` with
        :math:`\delta_u F(u,m,p;\hat u) = \mathrm{adj\_rhs}`."""
        raise NotImplementedError

    def evalGradientParameter(self, x, out):
        r"""``out`` = :math:`\delta_m F(u,m,p;\hat m)`."""
        raise NotImplementedError

    def setLinearizationPoint(self, x, gauss_newton_approx):
        """Fix the point at which the incremental solves and second derivatives
        are evaluated."""
        raise NotImplementedError

    def solveIncremental(self, out, rhs, is_adj):
        """Solve the incremental forward (``is_adj=False``) or adjoint system."""
        raise NotImplementedError

    def apply_ij(self, i, j, dir, out):
        r"""``out`` = :math:`\delta_{ij} F(u,m,p;\hat i,\tilde j)` with
        :math:`\tilde j` = ``dir``."""
        raise NotImplementedError

    def apply_ijk(self, i, j, k, x, jdir, kdir, out):
        r"""``out`` = the third derivative block contracted with ``jdir`` and
        ``kdir``."""
        raise NotImplementedError


sync_spellings(PDEProblem)

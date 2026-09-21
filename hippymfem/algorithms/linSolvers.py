# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Linear solvers.

hIPPYlib leans on PETSc here, and in particular on ``PETScLUSolver`` for the
forward, adjoint and incremental solves.  PyMFEM's standard parallel build links
no direct solver (no MUMPS, SuperLU_dist, STRUMPACK or PETSc), so the exact-solve
capability is supplied from the Python side in
:mod:`~hippymfem.algorithms.directSolvers`:

===========================  ================================================
hIPPYlib                     hIPPyMFEM
===========================  ================================================
``PETScKrylovSolver``        :class:`KrylovSolver` (MFEM Krylov + hypre PC), or
                             ``PETScKrylovSolver`` when petsc4py is importable
``PETScLUSolver``            :class:`LUSolver`, exact on **any** rank count,
                             by replicating the matrix and factorizing with
                             scipy's SuperLU; or ``PETScLUSolver``, which uses a
                             genuinely distributed factorization when the local
                             PETSc has one
``amg_method()``             hypre BoomerAMG
===========================  ================================================

In parallel the factorization is replicated on every rank, so :class:`LUSolver`
refuses a matrix larger than ``max_global_size`` rather than silently exhausting
memory.
"""

import math
import os

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from ..common.keepalive import KeepAlive
from ..common.linalg import as_matrix, hypre_to_scipy
from ..common.parameterList import ParameterList
from ..common.parvector import ParVector
from ..common.operators import init_vector_like


def _env_default(name, cast, fallback):
    """A default that an environment variable can override, for sweeping a setting
    without touching every script that builds a solver."""
    v = os.environ.get(name)
    if v is None or v == "":
        return fallback
    try:
        return cast(v)
    except (TypeError, ValueError):
        return fallback


def KrylovSolver_ParameterList():
    return ParameterList({
        "rel_tolerance": [1e-12, "relative residual tolerance"],
        "abs_tolerance": [1e-20, "absolute residual tolerance"],
        "max_iter": [1000, "maximum number of iterations"],
        "print_level": [-1, "MFEM print level; -1 silent"],
        "nonzero_initial_guess": [False, "use the incoming x as initial guess"],
        "error_on_nonconvergence": [True, "raise if the solver does not converge"],
        "kdim": [50, "restart dimension for GMRES/FGMRES"],
        "amg_relax_type": [_env_default("HIPPYMFEM_AMG_RELAX", int, -1),
                           "hypre BoomerAMG relaxation type; -1 keeps MFEM's default "
                               "(l1-Jacobi on a device).  16 is Chebyshev, for SPD "
                               "operators only: faster per solve, slower per stage, so "
                               "not the default (benchmarks/DESIGN_NOTES.md, section 4)"],
        "amg_max_levels": [_env_default("HIPPYMFEM_AMG_MAX_LEVELS", int, -1),
                           "hypre BoomerAMG maximum number of levels; -1 keeps MFEM's "
                               "default of 25.  On a GPU the coarse levels are too small to "
                               "fill the card, so a V-cycle is bound by kernel launches; "
                               "fewer levels mean fewer launches and a bigger (more "
                               "expensive) coarsest solve"],
        "amg_agg_levels": [-1, "hypre BoomerAMG levels of aggressive coarsening; -1 keeps "
                               "MFEM's default of 1.  More of them coarsen faster, so the "
                               "hierarchy is shorter and cheaper to apply and to set up, at "
                               "the cost of a weaker preconditioner (more CG iterations)"],
        "pc_reuse": [_env_default("HIPPYMFEM_PC_REUSE", int, 0),
                     "keep the preconditioner for this many further operators before "
                     "rebuilding it.  A preconditioner changes the iteration count, never "
                     "the answer, and on a GPU its setup is most of a forward solve, so "
                     "reusing an older hierarchy saves setups for the price of more "
                     "iterations and of keeping one extra matrix and hierarchy alive.  "
                     "0, the default, rebuilds every time: a hierarchy from an older "
                     "parameter usually costs more iterations than it saves.  Worth it "
                     "for a problem whose Jacobian barely moves"],
        "amg_strength_threshold": [-1.0, "hypre BoomerAMG strength threshold; -1 keeps "
                                         "MFEM's default of 0.25.  0.5 is the usual advice "
                                         "for 3D problems: sparser interpolation, cheaper "
                                         "V-cycle, typically more iterations"],
    })


_METHODS = {
    "cg": mfem.CGSolver,
    "gmres": mfem.GMRESSolver,
    "fgmres": mfem.FGMRESSolver,
    "bicgstab": mfem.BiCGSTABSolver,
    "minres": mfem.MINRESSolver,
}


class _SolverBase(KeepAlive):
    """Common bookkeeping: parameters, iteration counts, hIPPYlib method names.

    The base of every solver here and in :mod:`.directSolvers`.
    """

    def __init__(self, comm=None):
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.parameters = KrylovSolver_ParameterList()
        self.iterations = 0
        self.converged = True
        self.A = None
        #: the operator the PDE last pointed this solver at (``_set_operator_once``
        #: skips the rebuild when it is asked for the same one again)
        self.current_operator = None
        self._template = None
        self._pc = None
        self._pc_built_on = None
        self._pc_age = 0

    # hIPPYlib spells these two ways; accept both
    def set_operator(self, A):
        raise NotImplementedError

    SetOperator = set_operator

    def solve(self, x, b):
        raise NotImplementedError

    def __call__(self, x, b):
        return self.solve(x, b)

    def init_vector(self, v, dim):

        if self._template is None:
            raise RuntimeError("set_operator must be called before init_vector")
        return init_vector_like(v, self._template)


    def release(self):
        """Drop the operator, the preconditioner and anything built from them.

        Called by the PDE before it rebuilds the matrix this solver holds, so the old
        matrix and its AMG hierarchy are freed *before* the new ones are allocated
        rather than after; without it a Newton step briefly holds two of everything
        (``benchmarks/DESIGN_NOTES.md``, section 4).  The next ``set_operator``
        rebuilds as usual.
        """
        for name in ("A", "_pc", "_solver", "_held", "_lu", "_template",
                     "_pc_built_on"):
            if hasattr(self, name):
                setattr(self, name, None)
        self.current_operator = None
        self._pc_age = 0
        return self


class KrylovSolver(_SolverBase):
    """MFEM Krylov solver with a hypre preconditioner.

    Parameters
    ----------
    comm : mpi4py communicator
    method : {"cg", "gmres", "fgmres", "bicgstab", "minres"}
    precond : {"amg", "ilu", "jacobi", "l1jacobi", "gs", "parasails", "euclid", "none"}
    systems_dim : int, optional
        Pass the number of components for BoomerAMG's systems-of-PDEs mode; use
        this for vector-valued (elasticity-like) operators, where scalar AMG
        converges poorly.
    elasticity : bool
        Turn on BoomerAMG's elasticity interpolation (needs ``systems_dim``).
    """

    def __init__(self, comm, method="cg", precond="amg", systems_dim=None,
                 elasticity=False, fes=None):
        super(KrylovSolver, self).__init__(comm)
        if method not in _METHODS:
            raise ValueError("unknown Krylov method %r; choose from %s"
                             % (method, sorted(_METHODS)))
        self.method = method
        self.precond_type = precond
        self.systems_dim = systems_dim
        self.elasticity = elasticity
        self.fes = fes
        self._solver = None
        self._pc = None
        self._template = None

    #: How hypre setup failures are handled.  MFEM's default aborts the whole job
    #: via ``MFEM_VERIFY``, which is wrong inside an optimizer: a line search
    #: routinely proposes a parameter whose operator defeats AMG's coarsening
    #: (``exp(m)`` spanning hundreds of decades), and that step should be rejected,
    #: not the run killed.  Warning lets the solve fail to converge and surface as
    #: a ``RuntimeError``, which the line search already catches.
    ERROR_MODE = mfem.HypreSolver.WARN_HYPRE_ERRORS

    def _set_error_mode(self, pc):
        try:
            pc.SetErrorMode(self.ERROR_MODE)
        except Exception:
            pass
        return pc

    def _make_pc(self, A):
        t = self.precond_type
        if t in (None, "none"):
            return None
        if t == "amg":
            pc = mfem.HypreBoomerAMG(A)
            pc.SetPrintLevel(0)
            self._set_error_mode(pc)
            relax = int(self.parameters["amg_relax_type"])
            if relax >= 0:
                pc.SetRelaxType(relax)
            levels = int(self.parameters["amg_max_levels"])
            if levels > 0:
                pc.SetMaxLevels(levels)
            agg = int(self.parameters["amg_agg_levels"])
            if agg >= 0:
                pc.SetAggressiveCoarsening(agg)
            theta = float(self.parameters["amg_strength_threshold"])
            if theta >= 0.0:
                pc.SetStrengthThresh(theta)
            if self.systems_dim:
                pc.SetSystemsOptions(int(self.systems_dim))
                if self.elasticity:
                    pc.SetElasticityOptions(self.fes)
            return pc
        if t == "ilu":
            pc = mfem.HypreILU()
            pc.SetLevelOfFill(1)
            pc.SetPrintLevel(0)
            self._set_error_mode(pc)
            pc.SetOperator(A)
            return pc
        if t in ("jacobi", "l1jacobi", "gs", "l1gs"):
            kind = {
                "jacobi": mfem.HypreSmoother.Jacobi,
                "l1jacobi": mfem.HypreSmoother.l1Jacobi,
                "gs": mfem.HypreSmoother.GS,
                "l1gs": mfem.HypreSmoother.l1GS,
            }[t]
            return self._set_error_mode(mfem.HypreSmoother(A, kind))
        if t == "parasails":
            pc = mfem.HypreParaSails(A)
            pc.SetSymmetry(1)
            return self._set_error_mode(pc)
        if t == "euclid":
            return self._set_error_mode(mfem.HypreEuclid(A))
        raise ValueError("unknown preconditioner %r" % (t,))

    def set_operator(self, A, pc=None):
        """Point the solver at ``A``; ``pc`` shares an existing preconditioner of ``A``.

        The forward solver and the forward-incremental solver hold the same Jacobian;
        the PDE passes the forward solver's preconditioner here when the operators are
        the same object, so the hierarchy is built once and held once.
        """
        A = as_matrix(A)
        # Release the previous operator, preconditioner and MFEM solver *before*
        # building the new ones: an optimizer re-sets the forward solver at every
        # trial point, and keeping the old ones would leak an AMG hierarchy per
        # line-search step.  MFEM's solver holds raw pointers to ``A`` and the
        # preconditioner, so the *current* three are held, and only they.
        reuse = int(self.parameters["pc_reuse"])
        keep = None
        if pc is None and reuse > 0 and self._pc is not None \
                and getattr(self, "_pc_age", 0) < reuse \
                and self._pc_built_on is not None \
                and self._pc_built_on.Height() == A.Height() \
                and self._pc_built_on.Width() == A.Width():
            # An older hierarchy, still the right shape: keep it and the matrix it was
            # built on (hypre's setup holds that matrix), and count one more use.
            keep = (self._pc, self._pc_built_on, getattr(self, "_pc_age", 0) + 1)
        self._solver = None
        self._pc = None
        self._held = None
        self.A = A
        self._template = ParVector(self.comm, A.Height())
        if keep is not None:
            self._pc, self._pc_built_on, self._pc_age = keep
        else:
            self._pc = pc if pc is not None else self._make_pc(A)
            self._pc_built_on = None if pc is not None else A
            self._pc_age = 0
        s = _METHODS[self.method](self.comm)
        s.SetOperator(A)
        if self._pc is not None:
            s.SetPreconditioner(self._pc)
        self._solver = s
        self._held = (A, self._pc, s, getattr(self, "_pc_built_on", None))
        return self

    SetOperator = set_operator

    def _configure(self):
        p = self.parameters
        s = self._solver
        s.SetRelTol(float(p["rel_tolerance"]))
        s.SetAbsTol(float(p["abs_tolerance"]))
        s.SetMaxIter(int(p["max_iter"]))
        s.SetPrintLevel(int(p["print_level"]))
        if self.method in ("gmres", "fgmres"):
            s.SetKDim(int(p["kdim"]))
        s.iterative_mode = bool(p["nonzero_initial_guess"])

    def solve(self, x, b):
        """``x = A^{-1} b``; returns the iteration count."""
        if self._solver is None:
            raise RuntimeError("set_operator must be called before solve")
        self._configure()
        if not self.parameters["nonzero_initial_guess"]:
            x.zero()
        self._solver.Mult(b.hypre, x.hypre)
        self.iterations = self._solver.GetNumIterations()
        self.converged = bool(self._solver.GetConverged())
        if self.converged and not math.isfinite(x.hypre.Norml2()):
            # A preconditioner whose setup failed can leave the Krylov solver
            # reporting convergence on a vector full of NaN.  The norm is a device
            # reduction to one scalar (NaN or inf in any entry makes it NaN or inf),
            # not a copy of the whole vector to the host.
            self.converged = False
        if not self.converged and self.parameters["error_on_nonconvergence"]:
            raise RuntimeError(
                "%s(%s) failed to converge in %d iterations (final rel. norm %.3e)"
                % (self.method, self.precond_type, self.iterations,
                   self._solver.GetFinalRelNorm())
            )
        return self.iterations


class LUSolver(_SolverBase):
    """Sparse direct solve via scipy's SuperLU, on any number of ranks.

    Stands in for hIPPYlib's ``PETScLUSolver``.  On one rank this factorizes the
    matrix in place.  In parallel it gathers the matrix onto every rank and
    factorizes it there, which is exact and independent of the rank count; see
    :class:`~hippymfem.algorithms.directSolvers.ReplicatedLUSolver`, which this
    delegates to, for the memory trade and the size guard.

    Parameters
    ----------
    comm : mpi4py communicator
    method : str
        Kept for source compatibility; ``"superlu"`` is the only factorization.
    max_global_size : int, optional
        Passed through to the replicated solver.
    """

    def __init__(self, comm=None, method="superlu", max_global_size=None):
        comm = comm if comm is not None else MPI.COMM_WORLD
        super(LUSolver, self).__init__(comm)
        self.method = method
        self.max_global_size = max_global_size
        self._lu = None
        self._replicated = None
        self._template = None

    def set_operator(self, A):
        import scipy.sparse.linalg as sla

        A = as_matrix(A)
        self.A = A
        if self.comm.size > 1:
            from .directSolvers import ReplicatedLUSolver

            self._replicated = ReplicatedLUSolver(
                self.comm, max_global_size=self.max_global_size)
            self._replicated.set_operator(A)
            self._template = self._replicated._template
        else:
            self._lu = sla.splu(hypre_to_scipy(A).tocsc())
            self._template = ParVector(self.comm, A.Height())
        self.iterations = 1
        self.converged = True
        return self

    SetOperator = set_operator

    def solve(self, x, b):
        if self._replicated is not None:
            return self._replicated.solve(x, b)
        if self._lu is None:
            raise RuntimeError("set_operator must be called before solve")
        x.array[:] = self._lu.solve(b.array)
        return 1

    def solveTranspose(self, x, b):
        """``x = A^{-T} b``."""
        if self._replicated is not None:
            return self._replicated.solveTranspose(x, b)
        if self._lu is None:
            raise RuntimeError("set_operator must be called before solve")
        x.array[:] = self._lu.solve(b.array, trans="T")
        return 1

    def mult(self, x, y):
        """``y = A^{-1} x``, so the solver can be used as an operator."""
        self.solve(y, x)
        return y


class TransposeSolver(KeepAlive):
    """Solve with ``A^T`` by forming the transpose once and wrapping a solver."""

    def __init__(self, solver_factory):
        self.factory = solver_factory
        self.inner = None
        self._At = None

    def release(self):
        """As :meth:`KrylovSolver.release`: drop the transpose and the inner solver."""
        inner = getattr(self, "inner", None)
        if inner is not None and hasattr(inner, "release"):
            inner.release()
        self.inner = None
        self._At = None
        self._held = None
        self.current_operator = None
        return self

    def set_operator(self, A):
        A = as_matrix(A)
        # As for KrylovSolver: hold the current transpose and inner solver, not every
        # one there has ever been.
        self.inner = None
        self._At = None
        self._held = None
        self._At = A.Transpose()
        self.inner = self.factory()
        self.inner.set_operator(self._At)
        self._held = (A, self._At, self.inner)
        return self

    SetOperator = set_operator

    @property
    def parameters(self):
        return self.inner.parameters

    def solve(self, x, b):
        return self.inner.solve(x, b)

    def init_vector(self, v, dim):
        return self.inner.init_vector(v, dim)


class LumpedMassSolver(KeepAlive):
    """Inverse of the row-sum-lumped diagonal of a mass matrix.

    A cheap, robust stand-in for an exact mass solve when only a spectrally
    equivalent operator is needed.
    """

    def __init__(self, M, comm=None):
        M = as_matrix(M)
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.M = M
        ones = ParVector(self.comm, M.Width())
        ones.set(1.0)
        self.d = ParVector(self.comm, M.Height())
        M.Mult(ones.hypre, self.d.hypre)
        if np.any(np.abs(self.d.array) < 1e-300):
            raise ValueError("lumped mass has a zero entry")
        self.parameters = KrylovSolver_ParameterList()
        self.keep(M)
        self._template = self.d

    def set_operator(self, M):
        return self

    def solve(self, x, b):
        np.divide(b.array, self.d.array, out=x.array)
        return 1

    def mult(self, x, y):
        np.multiply(x.array, self.d.array, out=y.array)
        return y

    def init_vector(self, v, dim):

        return init_vector_like(v, self.d)


def auto_solver(space_or_size, comm=None, max_direct=None, method="cg",
                precond="amg", **kw):
    """A direct solve where one is affordable, Krylov where it is not.

    An exact solve makes a Hessian spectrum a property of the discretization
    rather than of a Krylov tolerance, and it is rank-count independent.  It is
    the wrong default at scale, because the replicated factorization (the only
    one available without petsc4py) costs ``O(nnz(L+U))`` on *every* rank.  The
    choice therefore follows the problem size, not the rank count: a one-rank
    run on a fine mesh is exactly where a direct solve fails.

    Parameters
    ----------
    space_or_size : FunctionSpace or int
        The space to be solved on, or its global number of unknowns.
    max_direct : int, optional
        Largest problem to factorize.  Defaults to the replicated solver's own
        refusal threshold, so this never selects a solver that would then decline.

    Where petsc4py is available its distributed factorization is preferred, since
    it is exact without replicating the fill.
    """
    from .._petsc import petsc_available
    from .directSolvers import DEFAULT_MAX_GLOBAL_SIZE

    n = (int(space_or_size) if isinstance(space_or_size, (int, np.integer))
         else int(space_or_size.fes.GlobalTrueVSize()))
    cap = int(max_direct if max_direct is not None else DEFAULT_MAX_GLOBAL_SIZE)
    if n <= cap:
        # petsc_available() returns (ok, reason); a truth test on the pair itself
        # would always pass.
        if petsc_available()[0]:
            return make_solver(comm, "petsc_lu", **kw)
        return LUSolver(comm, max_global_size=cap, **kw)
    return KrylovSolver(comm, method=method, precond=precond, **kw)


def make_solver(comm, kind="krylov", method="cg", precond="amg", **kw):
    """Factory used by the priors and PDE problems.

    ``kind`` is one of

    ``"krylov"`` / ``"iterative"``
        MFEM Krylov with a hypre preconditioner.
    ``"lu"`` / ``"direct"``
        Exact scipy SuperLU, replicated in parallel (:class:`LUSolver`).
    ``"petsc_lu"``
        PETSc's best available distributed factorization.
    ``"petsc"``
        PETSc Krylov.
    ``"auto"``
        :func:`auto_solver`; needs ``space_or_size``.

    The PETSc kinds need petsc4py to have been imported before PyMFEM; see
    :mod:`~hippymfem.algorithms.directSolvers`.
    """
    if kind == "auto":
        return auto_solver(kw.pop("space_or_size"), comm, method=method,
                           precond=precond, **kw)
    if kind in ("lu", "direct"):
        return LUSolver(comm, **kw)
    if kind in ("krylov", "iterative"):
        return KrylovSolver(comm, method=method, precond=precond, **kw)
    if kind == "petsc_lu":
        from .directSolvers import PETScLUSolver

        return PETScLUSolver(comm, **kw)
    if kind == "petsc":
        from .directSolvers import PETScKrylovSolver

        return PETScKrylovSolver(comm, method=method, precond=precond, **kw)
    raise ValueError("unknown solver kind %r" % (kind,))


def _all_finite(v):
    """True on every rank when ``v`` has no non-finite entry."""
    from mpi4py import MPI as _MPI

    ok = bool(np.isfinite(v.array).all()) if v.local_size else True
    return bool(v.comm.allreduce(int(ok), op=_MPI.MIN))

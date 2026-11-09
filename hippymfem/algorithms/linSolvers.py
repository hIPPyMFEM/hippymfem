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
from ..config import env_flag

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from ..common.keepalive import KeepAlive
from ..common.linalg import as_matrix, hypre_to_scipy
from ..common.parameterList import ParameterList
from ..common.parvector import Layout, ParVector, allreduce as _reduce
from ..common.operators import init_vector_like


def _env_default(name, cast, fallback):
    """A default that an environment variable can override, for sweeping a setting
    without touching every script that builds a solver.  A value that ``cast`` does not
    take raises, with the variable's name: the default in its place was another
    configuration than the job script asked for, run without a word
    (``HIPPYMFEM_PMG=on`` kept BoomerAMG).  An empty value is the default."""
    v = os.environ.get(name)
    if v is None or not v.strip():
        return fallback
    try:
        return cast(v)
    except (TypeError, ValueError):
        raise ValueError("%s must be %s, not %r"
                         % (name, "a whole number" if cast is int else "a number", v)) from None


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
                               "not the default (benchmarks/DESIGN_NOTES.md, section 4).  "
                               "7 is Jacobi: on the quadratic hexahedra of the model "
                               "problem a third fewer iterations at the same cost each "
                               "(guide, gpu.rst, Single precision), not yet the default"],
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
        "amg_options": [os.environ.get("HIPPYMFEM_AMG", ""),
                        "further BoomerAMG settings, applied after the ones above through "
                        "hypre's own interface, which has more of them than MFEM's class: a "
                        "list such as \"relax_wt=1.8,pmax=6\" of the names in "
                        "mfemconfig.BOOMERAMG_SETTERS (\"print=1\" has hypre print the "
                        "hierarchy it builds)"],
        "pc_reuse": [_env_default("HIPPYMFEM_PC_REUSE", int, 0),
                     "keep the preconditioner for this many further operators before "
                     "rebuilding it.  A preconditioner changes the iteration count, never "
                     "the answer, and on a GPU its setup is most of a forward solve, so "
                     "reusing an older hierarchy saves setups for the price of more "
                     "iterations and of keeping one extra matrix and hierarchy alive.  "
                     "0, the default, rebuilds every time: a hierarchy from an older "
                     "parameter usually costs more iterations than it saves.  Worth it "
                     "for a problem whose Jacobian barely moves"],
        "pmg": [_env_default("HIPPYMFEM_PMG", int, 0),
                "in the single-precision hypre, precondition the matrix of a space of order "
                "two or more by a p-multigrid instead of BoomerAMG on the matrix itself: a "
                "relaxation on the matrix around one BoomerAMG V-cycle on the Galerkin "
                "product for the linear elements of the same mesh "
                "(singlesolve.PMGEngine).  The same iterations for a set-up that is a "
                "third of BoomerAMG's on quadratic hexahedra (guide, gpu.rst, Single "
                "precision).  0, the default, keeps BoomerAMG; where no interpolation can "
                "be built (fem.ptransfer.why_not) it is kept as well"],
        "pmg_h": [_env_default("HIPPYMFEM_PMG_H", int, 0),
                  "with pmg, on a mesh that knows the meshes it was refined from "
                  "(fem.ptransfer.set_coarser_meshes): correct from the linear elements by a "
                  "V-cycle over up to this many of those meshes, with Galerkin products of the "
                  "interpolations between them and BoomerAMG on the last one only, in place of "
                  "BoomerAMG on the linear elements' matrix.  The products keep 27 entries a "
                  "row and the neighbours of the mesh at every level, where an algebraic "
                  "coarsening has 100 to 170 entries a row from its second level on and a "
                  "level more for every eight times the ranks.  0: BoomerAMG on the linear "
                  "elements"],
        "pmg_h_host": [_env_default("HIPPYMFEM_PMG_H_HOST", int, 0),
                       "with pmg_h and hypre on a device: the levels of the coarser meshes whose "
                       "matrix has at most this many rows a rank live in host memory, with "
                       "everything under them (singlesolve.PMG_H_HOST).  A level of a few thousand "
                       "rows a rank costs a device its kernel launches and two copies around each "
                       "exchange between ranks, whatever its size; on the host it costs "
                       "microseconds and its exchange is a message.  Not where the last matrix is "
                       "gathered (seq_threshold) and has more rows than this in all: one core a rank "
                       "then solves it slower than the device.  0, the default: every level on "
                       "the device"],
        "amg_strength_threshold": [-1.0, "hypre BoomerAMG strength threshold; -1 keeps "
                                         "MFEM's default of 0.25.  0.5 is the usual advice "
                                         "for 3D problems: sparser interpolation, cheaper "
                                         "V-cycle, typically more iterations"],
    })


#: Whether a CG solve with a hypre preconditioner runs in hypre's own PCG (MFEM's
#: ``HyprePCG``) instead of MFEM's ``CGSolver`` (``HIPPYMFEM_HYPRE_PCG``, on by default).
#: The two make the same iteration: the same iterates, counts and stopping test from a
#: zero initial guess.  hypre's PCG tells BoomerAMG that the initial guess of a V-cycle
#: is zero, which saves the first relaxation its matrix-vector product on the finest
#: level; MFEM zeroes the vector without saying so.  With l1-Jacobi relaxation on a GPU
#: that is one product of four per iteration: a solve of 2.1 million dofs to 1e-12 took
#: 0.103 s and 0.089 s on an H100, 0.274 s and 0.227 s on an L40S (24 iterations each).
HYPRE_PCG = env_flag("HIPPYMFEM_HYPRE_PCG", True)

#: preconditioners that are hypre solvers, which hypre's PCG can call itself
_HYPRE_SOLVER_PCS = ("amg", "ilu", "parasails", "euclid")


def set_hypre_pcg(on=True):
    """Choose hypre's own PCG (``True``) or MFEM's CG (``False``) for the CG solves with a
    hypre preconditioner that are set up from now on (see :data:`HYPRE_PCG`); returns the
    old choice."""
    global HYPRE_PCG
    old = HYPRE_PCG
    HYPRE_PCG = bool(on) if not isinstance(on, str) else on.strip().lower() not in ("0", "off", "false", "no")
    return old


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
        for name in ("A", "_pc", "_solver", "_held", "_lu", "_replicated", "_n",
                     "_ksp", "_mat", "_template", "_pc_built_on", "_fallback"):
            if hasattr(self, name):
                setattr(self, name, None)
        self.current_operator = None
        self._pc_age = 0
        self._single = False
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
    systems_bynodes : bool, optional
        How the unknowns of the components are ordered, which BoomerAMG has to be told
        to pair the rows of one component: component after component
        (``mfem.Ordering.byNODES``, the ordering of this library's spaces unless another
        is asked for, and the default here), or interleaved (``byVDIM``: pass ``False``).
        Taken from ``fes`` where that is given.  With ``systems_dim`` and neither of the
        two, the default is taken and a warning says so once: on an interleaved space
        BoomerAMG then pairs rows of different components, and a solve takes 15 to 50
        times the iterations.
    fes : FunctionSpace or mfem.ParFiniteElementSpace, optional
        The space of the unknowns, for their ordering and for the elasticity options.
    """

    #: whether the warning about an ordering left to the default has been given
    _warned_ordering = False

    def __init__(self, comm, method="cg", precond="amg", systems_dim=None,
                 elasticity=False, fes=None, systems_bynodes=None):
        super(KrylovSolver, self).__init__(comm)
        if method not in _METHODS:
            raise ValueError("unknown Krylov method %r; choose from %s"
                             % (method, sorted(_METHODS)))
        self.method = method
        self.precond_type = precond
        self.systems_dim = systems_dim
        self.elasticity = elasticity
        self.fes = fes
        space = self._mfem_space()
        if systems_bynodes is None:
            if space is not None:
                systems_bynodes = space.GetOrdering() == mfem.Ordering.byNODES
            else:
                systems_bynodes = True
                if systems_dim and not KrylovSolver._warned_ordering:
                    KrylovSolver._warned_ordering = True
                    import warnings

                    warnings.warn(
                        "KrylovSolver(systems_dim=%d) without fes or systems_bynodes takes the "
                        "components one after another (byNODES, the ordering of this library's "
                        "spaces).  For an interleaved (byVDIM) space pass fes= or "
                        "systems_bynodes=False: BoomerAMG would pair rows of different "
                        "components." % int(systems_dim), RuntimeWarning, stacklevel=2)
        self.systems_bynodes = bool(systems_bynodes)
        self._solver = None
        self._pc = None
        self._template = None

    def _mfem_space(self):
        """MFEM's space of :attr:`fes`: a ``FunctionSpace``'s, or the one given (``None``
        without one).  A ``FunctionSpace`` has no ``GetOrdering``, and passing one raised."""
        fes = self.fes
        if fes is None:
            return None
        inner = getattr(fes, "fes", None)
        return inner if inner is not None else fes

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
            from ..common import mfemconfig as _mc

            pc = mfem.HypreBoomerAMG(A)
            pc.SetPrintLevel(0)
            self._set_error_mode(pc)
            # MFEM's setters do not pass on what hypre says of a value.  hypre drops one that
            # it refuses (a strength threshold above one) and raises its error flag: the
            # solver went on with hypre's own value without a word, and the flag stayed up
            # for whichever call returned it next (MFEM's set-up of the preconditioner, as
            # an error of the set-up).  The flag is the process's: cleared here and read
            # after each setter, it is that setter's verdict, the same on every rank (as in
            # mfemconfig.set_boomeramg_options).
            H = _mc._hypre_library()
            if H is not None:
                H.HYPRE_ClearAllErrors()

            def taken(name, value):
                flag = int(H.HYPRE_GetError()) if H is not None else 0
                if flag:
                    H.HYPRE_ClearAllErrors()
                    raise ValueError("hypre refused the solver parameter %s=%r (its error flag "
                                     "read %d)" % (name, value, flag))

            if self.systems_dim:
                # hypre takes the rows for components in turn unless it is told which
                # component each is, and the rows of a space ordered by nodes are one
                # component after another: with the number alone it pairs rows of
                # different components, and a solve takes several times the iterations
                dim, bynodes = int(self.systems_dim), self.systems_bynodes
                if bynodes and _reduce(self.comm, int(A.Height() % dim != 0), MPI.MAX):
                    # (MFEM stops the process there; an error of every rank instead)
                    raise ValueError("systems_dim=%d with the components one after another "
                                     "(systems_bynodes) needs the rows of every rank to be "
                                     "that many times its nodes" % dim)
                pc.SetSystemsOptions(dim, bynodes)
                if self.elasticity:
                    pc.SetElasticityOptions(self._mfem_space())
                # (Before the solver's own settings, which follow.  With the components MFEM
                # writes settings of its own, no aggressive levels and a strength threshold
                # of 0.5 among them: written last, they took the place of amg_agg_levels,
                # amg_strength_threshold and the amg_options of a vector problem without a
                # word.)
                taken("systems_dim", dim)
            relax = int(self.parameters["amg_relax_type"])
            if relax >= 0:
                pc.SetRelaxType(relax)
                taken("amg_relax_type", relax)
            levels = int(self.parameters["amg_max_levels"])
            if levels > 0:
                pc.SetMaxLevels(levels)
                taken("amg_max_levels", levels)
            agg = int(self.parameters["amg_agg_levels"])
            if agg >= 0:
                pc.SetAggressiveCoarsening(agg)
                taken("amg_agg_levels", agg)
            theta = float(self.parameters["amg_strength_threshold"])
            if theta >= 0.0:
                pc.SetStrengthThresh(theta)
                taken("amg_strength_threshold", theta)
            extra = str(self.parameters["amg_options"] or "")
            if extra:
                _mc.set_boomeramg_options(pc, _mc.parse_boomeramg_options(extra))
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

    def set_operator(self, A, pc=None, donor=None):
        """Point the solver at ``A``; ``pc`` shares an existing preconditioner of ``A``.

        The forward solver and the forward-incremental solver hold the same Jacobian;
        the PDE passes the forward solver's preconditioner here when the operators are
        the same object, so the hierarchy is built once and held once.  ``donor`` is
        the solver ``pc`` comes from: when that one runs hypre's PCG on this very
        matrix (:data:`HYPRE_PCG`), its PCG object is shared too, since each such
        object sets its preconditioner up at its first solve.

        A :class:`~hippymfem.algorithms.singlesolve.SingleParMatrix` is solved in the
        single-precision hypre it lives in (CG with BoomerAMG only); ``pc`` is then the
        engine of another solver of the same matrix.
        """
        from .singlesolve import SingleParMatrix

        if isinstance(A, SingleParMatrix):
            return self._set_single(A, pc)
        self._single = False
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
        self._template = ParVector(self.comm, A.Height(), layout=_layout_of(self.comm, A))
        if keep is not None:
            self._pc, self._pc_built_on, self._pc_age = keep
        else:
            self._pc = pc if pc is not None else self._make_pc(A)
            self._pc_built_on = None if pc is not None else A
            self._pc_age = 0
        s = None
        self._fallback = None
        if (HYPRE_PCG and self.method == "cg" and self.precond_type in _HYPRE_SOLVER_PCS
                and keep is None and self._pc is not None and isinstance(A, mfem.HypreParMatrix)):
            # (a hierarchy kept from an older matrix stays with MFEM's CG: the setup
            # of hypre's PCG would rebuild it on the new one)
            if pc is None:
                s = mfem.HyprePCG(A)
                s.SetPreconditioner(self._pc)
            else:
                shared = getattr(donor, "_solver", None) if donor is not None else None
                if (isinstance(shared, mfem.HyprePCG) and getattr(donor, "A", None) is A
                        and getattr(donor, "_pc", None) is pc):
                    s = shared
        self._native = s is not None
        if s is None:
            s = _METHODS[self.method](self.comm)
            s.SetOperator(A)
            if self._pc is not None:
                s.SetPreconditioner(self._pc)
        self._solver = s
        # hypre sets a preconditioner up at its first application: until the first
        # solve is over, a scoped pool recycles the arrays the setup allocates and
        # frees (mfemconfig.set_hypre_pool); it holds nothing afterwards
        self._setup_pending = self._pc is not None
        #: a hierarchy kept from an older matrix (pc_reuse) is set up already, and so is
        #: one that another solver handed over after a solve of its own (read by a tally
        #: of the solves, bench_newton_device's; solve asks the object that sets it up for
        #: the solver it runs, the preconditioner for MFEM's and the PCG for hypre's)
        self._setup_due = (self._setup_pending and keep is None
                           and not getattr(self._pc, "_hm_set_up", False))
        if self._setup_pending:
            from ..common.mfemconfig import hypre_pool_open

            hypre_pool_open()
        self._held = (A, self._pc, s, getattr(self, "_pc_built_on", None))
        return self

    SetOperator = set_operator

    @property
    def tolerance_floor(self):
        """The relative residual below which a solve with the present operator does not
        go, whatever tolerance is asked: that of single precision for a matrix of the
        single-precision hypre, zero otherwise."""
        if getattr(self, "_single", False):
            from .singlesolve import SINGLE_TOL_FLOOR

            return SINGLE_TOL_FLOOR
        return 0.0

    def can_solve_single(self):
        """Whether this solver could run in the single-precision hypre: CG or GMRES with a
        plain BoomerAMG, the combinations set up there (GMRES for a Jacobian that is not
        symmetric, and for its transpose)."""
        return bool(self.method in ("cg", "gmres") and self.precond_type == "amg"
                    and not self.systems_dim and int(self.parameters["pc_reuse"]) == 0)

    def _set_single(self, S, engine=None):
        from .singlesolve import PMGEngine, SingleEngine, SingleGMRESEngine

        if not self.can_solve_single():
            raise ValueError("a single-precision matrix is solved by cg or gmres with a plain "
                             "amg preconditioner, not by %s with %s" % (self.method, self.precond_type))
        self._solver = None
        self._pc = None
        self._held = None
        self._fallback = None
        self._native = False
        self.A = S
        # (the matrix knows its rows on this rank: a layout without a message, where a
        # vector made from a local size gathers every rank's at each new operator)
        self._template = ParVector(self.comm, S.Height(), layout=Layout.of_range(
            self.comm, int(S.row_starts[0]), int(S.row_starts[1]), int(S.global_rows)))
        gmres = self.method == "gmres"
        if (isinstance(engine, SingleEngine) and engine.matrix is S
                and isinstance(engine, SingleGMRESEngine) == gmres):
            self._pc = engine
        elif gmres:
            self._pc = SingleGMRESEngine(S, self.parameters)
        else:
            # a p-multigrid where the solver asks for one and the matrix came with the
            # interpolation for it (PDEVariationalProblem._p_transfer)
            transfer = getattr(S, "transfer", None) if int(self.parameters["pmg"]) else None
            self._pc = (PMGEngine(S, transfer, self.parameters) if transfer is not None
                        else SingleEngine(S, self.parameters))
        self._pc_built_on = None
        self._pc_age = 0
        self._setup_pending = False
        self._single = True
        self._held = (S, self._pc)
        return self

    def _gmres_verdict(self, final, b, tol, atol):
        """hypre's GMRES verdict, checked against its own final relative residual.

        hypre's GMRES can report convergence where the residual it reports is far above
        the tolerance it was given: on a nearly singular operator (diffusion and
        convection with a mass term of 1e-6, a smooth right-hand side) it stopped after
        20 iterations as converged with a relative residual of 3.6 for a tolerance of
        1e-5.  The residual of right-preconditioned GMRES is the true one in the
        Euclidean norm, relative to ``b``: a verdict is kept where it is below the
        relative tolerance hypre was given or, times ``|b|``, below the absolute one.
        The norm of ``b`` is a collective, taken only where the first test fails, which
        it does on every rank alike (``final`` is hypre's global value)."""
        from .singlesolve import SINGLE_TOL_FLOOR

        slack = 1.01                                 # (hypre reports it in single precision)
        if final <= slack * max(float(tol), SINGLE_TOL_FLOOR):
            return True
        return atol > 0.0 and final * b.norm("l2") <= slack * atol

    def _solve_single(self, x, b):
        from .singlesolve import AMG_OPTIONS

        p = self.parameters
        if p["nonzero_initial_guess"]:
            raise ValueError("a single-precision solve starts from zero: its matrix cannot "
                             "form the residual of a double-precision guess")
        tol, atol, most = p["rel_tolerance"], float(p["abs_tolerance"]), int(p["max_iter"])
        gmres = self.method == "gmres"
        self.iterations, final = self._pc.solve(x.hypre, b.hypre, tol, atol, most)
        # Not the count.  With a preconditioner that is not positive definite (BoomerAMG
        # with plain Jacobi relaxation outside the meshes that suit it) the squared
        # residual in the preconditioner's norm turns negative: hypre's test takes that
        # for converged after a few iterations and returns a residual that is not a
        # number, with the true one of order one.  So: hypre's verdict, and a residual
        # that is a number; for GMRES also one below the tolerance (_gmres_verdict).
        self.converged = bool(self._pc.converged) and math.isfinite(final)
        stalled = False
        if self.converged and gmres and not self._gmres_verdict(final, b, tol, atol):
            self.converged, stalled = False, True
        size = 1.0
        if not self.converged and self.iterations < most:
            # hypre's PCG stopped by itself.  It works with squares, <C b, b> and
            # <C r, r>, and in single precision those leave the range of numbers long
            # before the vectors do: with a right-hand side of size 1e-16 the second is
            # subnormal at a relative residual of 1e-4, and hypre breaks off there; one
            # of size 1e-23 it takes for zero, and one of size 1e20 for a wrong input.
            # The system is linear: it is solved again for the right-hand side in units
            # in which its size is near one (a power of two, so that only exponents
            # change).  A preconditioner that is not positive definite fails again.
            size = b.norm("l2")
            if size < 1e-300 or size <= atol:
                # zero, nothing a double can scale, or below the absolute tolerance,
                # which the zero guess meets as it does for the double-precision solvers
                # (hypre's GMRES takes such a right-hand side for zero and does not
                # iterate, without a verdict)
                x.zero()
                self.iterations, final, self.converged, stalled = 0, 0.0, True, False
            elif math.isfinite(size):
                power = math.frexp(size)[1]
                unit = b.copy().scale(math.ldexp(1.0, -power))
                uatol = math.ldexp(atol, -power)
                self.iterations, final = self._pc.solve(x.hypre, unit.hypre, tol, uatol, most)
                x.scale(math.ldexp(1.0, power))
                self.converged = bool(self._pc.converged) and math.isfinite(final)
                if self.converged and gmres and not self._gmres_verdict(final, unit, tol, uatol):
                    self.converged, stalled = False, True
        # the norm over all ranks: one that judged its own part would raise alone
        if self.converged and not math.isfinite(x.norm("l2")):
            self.converged = False
        #: hypre's final relative residual of the last single-precision solve (in the
        #: preconditioner's norm for PCG, the Euclidean one for GMRES)
        self.final_rel_norm = final
        if not self.converged and p["error_on_nonconvergence"]:
            why = ""
            if not math.isfinite(size):
                why = ": the right-hand side is not a number"
            elif stalled:
                why = (": hypre's GMRES reported convergence with a residual above the "
                       "tolerance")
            elif math.isnan(final):
                why = (": hypre's GMRES broke down" if gmres else
                       ": the preconditioner is not positive definite on this matrix")
                if AMG_OPTIONS:
                    why += " (BoomerAMG with %s)" % ", ".join(
                        "%s=%s" % kv for kv in AMG_OPTIONS.items())
            elif self.iterations < most:
                why = ": hypre's %s broke off" % ("GMRES" if gmres else "PCG")
            raise RuntimeError(
                "%s(amg) in single precision failed to converge in %d iterations "
                "(final rel. norm %.3e)%s" % (self.method, self.iterations, final, why))
        return self.iterations

    def _mfem_cg(self):
        """MFEM's CG on the operator and preconditioner of a solver that runs hypre's
        PCG, for a solve from a nonzero initial guess: MFEM's stopping test is relative
        to the first residual, hypre's to the right-hand side."""
        if getattr(self, "_fallback", None) is None:
            s = mfem.CGSolver(self.comm)
            s.SetOperator(self.A)
            s.SetPreconditioner(self._pc)
            self._fallback = s
        return self._fallback

    def _configure(self, s=None):
        p = self.parameters
        if s is None:
            s = self._solver
            if getattr(self, "_native", False):
                s.SetTol(float(p["rel_tolerance"]))
                s.SetAbsTol(float(p["abs_tolerance"]))
                s.SetMaxIter(int(p["max_iter"]))
                s.SetPrintLevel(max(0, int(p["print_level"])))
                s.iterative_mode = False
                return
        s.SetRelTol(float(p["rel_tolerance"]))
        s.SetAbsTol(float(p["abs_tolerance"]))
        s.SetMaxIter(int(p["max_iter"]))
        s.SetPrintLevel(int(p["print_level"]))
        if self.method in ("gmres", "fgmres"):
            s.SetKDim(int(p["kdim"]))
        s.iterative_mode = bool(p["nonzero_initial_guess"])

    def solve(self, x, b):
        """``x = A^{-1} b``; returns the iteration count."""
        if getattr(self, "_single", False):
            return self._solve_single(x, b)
        if self._solver is None:
            raise RuntimeError("set_operator must be called before solve")
        native = getattr(self, "_native", False) and not self.parameters["nonzero_initial_guess"]
        solver = self._solver if (native or not getattr(self, "_native", False)) else self._mfem_cg()
        self._configure(None if solver is self._solver else solver)
        if not self.parameters["nonzero_initial_guess"]:
            x.zero()
        from ..common import mfemconfig as _mc

        # a solve on a hierarchy that is set up exchanges through the buffers its
        # matrices keep: those may go to a GPU-aware MPI (mfemconfig.GPU_AWARE_SOLVES).
        # Set up for the solver that applies it, that is: MFEM sets a preconditioner up
        # at its own first application, whatever hypre's PCG did with it before (a
        # hierarchy kept for the next operator, one lent without its owner's PCG, a solve
        # from a nonzero guess after one from zero).  And hypre's PCG sets it up at its
        # own first solve, whatever MFEM did with it before (a solve from zero after one
        # from a nonzero guess, a PCG lent by an owner whose solves were all of those).
        direct = self._pc is not None and not native
        if direct:
            setup = not getattr(self._pc, "_hm_direct", False)
        else:
            setup = native and not getattr(solver, "_hm_native", False)
        aware = _mc.GPU_AWARE_SOLVES and not setup
        if aware:
            _mc.set_hypre_gpu_aware_mpi(True)
        try:
            solver.Mult(b.hypre, x.hypre)
        finally:
            if aware:
                _mc.set_hypre_gpu_aware_mpi(False)
        if direct:
            try:
                self._pc._hm_direct = True          # MFEM has set it up, for every solver
            except AttributeError:
                pass
        elif native:
            try:
                solver._hm_native = True            # hypre's PCG has, for all that share it
            except AttributeError:
                pass
        if getattr(self, "_setup_pending", False):
            from ..common.mfemconfig import hypre_pool_close

            self._setup_pending = False
            hypre_pool_close()
            try:
                self._pc._hm_set_up = True          # for the solvers this one lends it to
            except AttributeError:
                pass
        if native:
            its, norm = mfem.intp(), mfem.doublep()
            solver.GetNumIterations(its)
            solver.GetFinalResidualNorm(norm)
            self.iterations = int(its.value())
            final = float(norm.value())
            # Not the count.  hypre's PCG leaves an iteration that breaks down before the
            # last one allowed (a matrix that is not positive definite); it takes one for
            # converged when <C r, r> turns negative (a preconditioner that is not), with
            # a residual that is not a number and a true one of order one; and where
            # <C b, b> is not positive, or not a number, it does not begin, and returns
            # the right-hand side or zero with the count and the residual of the solve
            # before.  So: hypre's verdict, and a residual that is a number.
            verdict, error = _pcg_verdict(solver)
            if verdict is None:
                # (no handle of hypre's solver from this MFEM library: the count, for a
                # solve stopped by the absolute tolerance, or by the relative one at the
                # last iteration allowed)
                verdict = (self.iterations < int(self.parameters["max_iter"])
                           or final <= float(self.parameters["rel_tolerance"]))
            elif not verdict:
                from .singlesolve import HYPRE_ERROR_CONV

                # An iteration that fails, by a breakdown or at its limit, raises the bit.
                # The verdict is one for all ranks, but the flag is a process's: what is
                # read from it is reduced over them, or one rank would go on alone.
                failed = _reduce(self.comm, 1 if error & HYPRE_ERROR_CONV else 0, MPI.MAX)
                if not failed:
                    # At its limit hypre raises the bit only where it has a test to miss,
                    # and with both tolerances zero, or of a square that is, it has none.
                    # What it returns then tells an iteration from none, which leaves zero
                    # or the right-hand side (the count does not: without an iteration it
                    # is the solve before's).
                    p = self.parameters
                    rel, least = float(p["rel_tolerance"]), float(p["abs_tolerance"])
                    failed = (rel * rel == 0.0 == least * least and x.norm("l2") > 0.0
                              and x.copy().axpy(-1.0, b).norm("l2") > 0.0)
                if b.norm("l2") < 1e-300:
                    x.zero()                       # zero (or nothing a double can square)
                    self.iterations, final, verdict = 0, 0.0, True
                elif not failed:
                    self.iterations, final = 0, float("nan")      # hypre did not begin
            self.converged = bool(verdict) and math.isfinite(final)
        else:
            self.iterations = solver.GetNumIterations()
            self.converged = bool(solver.GetConverged())
            final = None
        if self.converged and not math.isfinite(x.norm("l2")):
            # A preconditioner whose setup failed can leave the Krylov solver
            # reporting convergence on a vector full of NaN.  The norm is a device
            # reduction to one scalar (NaN or inf in any entry makes it NaN or inf),
            # not a copy of the whole vector to the host, and it is the norm over all
            # ranks: one that judged its own part would raise alone.
            self.converged = False
        if not self.converged and self.parameters["error_on_nonconvergence"]:
            why = ""
            if native and self.iterations < int(self.parameters["max_iter"]):
                why = (": hypre's PCG broke off (the matrix or the preconditioner is "
                       "not positive definite)")
            raise RuntimeError(
                "%s(%s) failed to converge in %d iterations (final rel. norm %.3e)%s"
                % (self.method, self.precond_type, self.iterations,
                   final if final is not None else solver.GetFinalRelNorm(), why)
            )
        return self.iterations


def _pcg_verdict(pcg):
    """What hypre says of the solve that an ``mfem.HyprePCG`` has just made, which MFEM's
    class does not pass on: its verdict (``HYPRE_PCGGetConverged``; ``None`` where the
    MFEM library does not give hypre's handle of the solver, as for
    :func:`~hippymfem.common.mfemconfig.boomeramg_handle`) and its error flag, which is
    read and cleared.

    The flag is the process's, and MFEM's ``HyprePCG`` does not look at it: left up after
    a solve that ended at its limit, or after a set-up that met a zero row on this rank,
    it is what the next call that returns the flag hands back as its own, a setter of a
    BoomerAMG option as a refusal and a set-up by MFEM as a failure."""
    import ctypes

    from ..common import mfemconfig as _mc

    H, lib = _mc._hypre_library(), _mc._mfem_library()
    if H is None:
        return None, 0
    error = int(H.HYPRE_GetError())
    if error:
        H.HYPRE_ClearAllErrors()
    name = "_ZNK4mfem8HyprePCGcvP19hypre_Solver_structEv"
    if lib is None or not hasattr(lib, name) or not hasattr(H, "HYPRE_PCGGetConverged"):
        return None, error
    try:
        handle, get = getattr(lib, name), H.HYPRE_PCGGetConverged
        handle.restype, handle.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
        get.restype, get.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]
        solver = handle(int(pcg.this))
        if not solver:
            return None, error
        said = ctypes.c_longlong(0)                 # (room for hypre's integer, 4 or 8 bytes)
        get(ctypes.c_void_p(solver), ctypes.byref(said))
    except Exception:                                            # noqa: BLE001
        return None, error
    return bool(said.value), error


def _layout_of(comm, A):
    """The layout of a vector on the rows of ``A`` without a message, where ``A`` is a
    parallel matrix that knows its rows on this rank (one of MFEM's); ``None`` otherwise,
    and the vector then gathers every rank's size for its own."""
    try:
        part, n = A.GetRowPartArray(), int(A.GetGlobalNumRows())
        first, last = int(part[0]), int(part[1])
    except Exception:                                            # noqa: BLE001
        return None
    if last - first != int(A.Height()):
        return None
    return Layout.of_range(comm, first, last, n)


class ChebyshevSolver(KeepAlive):
    r"""A fixed polynomial in place of a Krylov solve: :math:`x = q(CA)\,C\,b`.

    ``krylov`` is a :class:`KrylovSolver` whose operator :math:`A` is set, with its
    preconditioner :math:`C` (a multigrid cycle).  ``degree`` steps of the Chebyshev
    iteration on the interval of the eigenvalues of :math:`CA` apply :math:`C` that many
    times (a BoomerAMG cycle from the iterate before it, any other preconditioner to a
    residual updated with a product of :math:`A`), with coefficients that do not depend
    on the right-hand side.  The result is one linear map, symmetric and positive definite
    where :math:`C` is, and its error in the energy norm is at most

    .. math:: 1 / T_{\mathrm{degree}}\Bigl(\frac{\lambda_{\max} + \lambda_{\min}}
              {\lambda_{\max} - \lambda_{\min}}\Bigr)

    of the solution (:attr:`error_bound`): 1.6 % with four cycles of a multigrid with
    which CG gains a factor of 0.3 an iteration, where CG needs twelve iterations for
    six digits.

    That is what a preconditioner has to be.  A Krylov solve stopped at a tolerance is
    a different map for every right-hand side, so it has to be stopped far below the
    tolerance of the iteration it preconditions (``ReducedSpaceNewtonCG``,
    ``cg_preconditioner_tolerance``); a fixed polynomial needs no tolerance, and the
    iteration it preconditions sees an operator whose spectrum is that of the exact
    preconditioner moved by at most twice the bound.  It does not replace the solve
    where the solution itself is wanted (a sample of the prior, a variance).

    The interval is estimated once, at the first solve, from the Lanczos matrix of
    ``lanczos`` CG iterations on a vector of random signs; the Ritz values lie inside
    the spectrum, so the interval used is wider (``safety``).  Outside the interval the
    polynomial's error grows slowly and stays below one up to twice its centre, so an
    estimate that is somewhat off costs accuracy, not definiteness.
    """

    def __init__(self, krylov, degree=4, lanczos=12, safety=(0.85, 1.05)):
        if getattr(krylov, "_single", False) or getattr(krylov, "_pc", None) is None \
                or getattr(krylov, "A", None) is None:
            raise ValueError("ChebyshevSolver needs a KrylovSolver with an operator and a "
                             "preconditioner in double precision")
        if int(degree) < 1:
            raise ValueError("degree must be at least 1")
        self.krylov, self.degree = krylov, int(degree)
        self.comm = krylov.comm
        self.lanczos, self.safety = int(lanczos), (float(safety[0]), float(safety[1]))
        #: the interval of the polynomial, and the Ritz values it came from
        self.interval = self.ritz = None
        self.iterations, self.converged = 0, True
        self._work = None
        self.keep(krylov)

    @staticmethod
    def fits(krylov):
        """Whether ``krylov`` can be replaced: CG with a preconditioner, in double."""
        return (isinstance(krylov, KrylovSolver) and krylov.method == "cg"
                and not getattr(krylov, "_single", False)
                and getattr(krylov, "_pc", None) is not None
                and getattr(krylov, "A", None) is not None)

    def init_vector(self, v, dim):
        return self.krylov.init_vector(v, dim)

    @property
    def error_bound(self):
        """The bound on the relative error of a solve in the energy norm, for
        eigenvalues inside :attr:`interval` (``None`` before the first solve)."""
        if self.interval is None:
            return None
        lo, hi = self.interval
        if hi <= lo:
            return 0.0
        s = (hi + lo) / (hi - lo)
        return 1.0 / math.cosh(self.degree * math.acosh(s))

    def _vectors(self):
        A = self.krylov.A
        if self._work is None or self._work[0] is not A:
            layout = _layout_of(self.comm, A)
            first = ParVector(self.comm, A.Height(), layout=layout)
            self._work = (A, first) + tuple(first.duplicate() for _ in range(3))
            self.interval = None
        return self._work[1:]

    def setup(self):
        """Estimate the interval (and let MFEM set the preconditioner up for a direct
        application, which it does at the first one whatever hypre's PCG did before)."""
        r, z, d, t = self._vectors()
        if self.interval is not None:
            return self
        from ..common import mfemconfig as _mc

        A, C = self.krylov.A, self.krylov._pc
        # random signs, from a generator of its own: the stream of parRandom is the
        # caller's, and a benchmark's samples must not depend on whether this ran
        rng = np.random.RandomState(8191 + 7919 * self.comm.rank)
        r._host_write()[:] = rng.randint(0, 2, r.local_size) * 2.0 - 1.0
        p = d
        _mc.hypre_pool_open()
        try:
            C.Mult(r.hypre, z.hypre)
        finally:
            _mc.hypre_pool_close()
        p.assign(z)
        rz = r.inner(z)
        alphas, betas, first = [], [], rz
        for _ in range(max(2, self.lanczos)):
            if not (rz > 0.0) or not math.isfinite(rz):
                break
            A.Mult(p.hypre, t.hypre)
            pap = p.inner(t)
            if not (pap > 0.0) or not math.isfinite(pap):
                break
            alpha = rz / pap
            r.axpy(-alpha, t)
            C.Mult(r.hypre, z.hypre)
            new = r.inner(z)
            alphas.append(alpha)
            betas.append(new / rz)
            if new <= 1e-24 * first:
                break
            p.axpby(1.0, new / rz, z)
            rz = new
        m = len(alphas)
        if m < 2:
            raise RuntimeError("ChebyshevSolver: the preconditioned operator is not positive "
                               "definite on a random vector (no interval to work on)")
        T = np.zeros((m, m))
        for j in range(m):
            T[j, j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j else 0.0)
            if j:
                T[j, j - 1] = T[j - 1, j] = math.sqrt(betas[j - 1]) / alphas[j - 1]
        ritz = np.linalg.eigvalsh(T)
        self.ritz = (float(ritz[0]), float(ritz[-1]))
        self.interval = (self.safety[0] * self.ritz[0], self.safety[1] * self.ritz[1])
        return self

    def solve(self, x, b):
        """``x = q(C A) C b``; returns the number of cycles.

        A step of the iteration needs ``C r`` for the residual ``r`` of the current
        iterate ``y``.  With BoomerAMG a cycle applied from ``y`` (MFEM's
        ``iterative_mode``) returns ``y + C (b - A y)``, the residual formed inside it
        by the relaxation that a cycle begins with: the step is the difference, and
        no product of the operator is made here.  (Applied from zero, a cycle begins
        with that same product, of a zero vector, since MFEM does not mark the vector
        as zero for hypre: one product with its exchange more in every cycle.)  Any
        other preconditioner is applied to a residual that is updated with a product,
        since not every one of them starts from the vector it is given."""
        if self.interval is None or self._work is None or self._work[0] is not self.krylov.A:
            self.setup()
        r, z, d, t = self._vectors()
        A, C = self.krylov.A, self.krylov._pc
        whole = getattr(C, "polynomial", None)
        if whole is not None:
            # a cycle that runs the polynomial itself, in its own precision and memory
            # (singlesolve.HCycle): the same recurrence, two conversions a solve
            getattr(b, "hypre", None)
            whole(x.hypre, b.hypre, self.degree, self.interval)
            self.iterations = self.degree
            return self.degree
        lo, hi = self.interval
        theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
        # the right-hand side with MFEM, as a Krylov solve takes it: its arithmetic, and
        # the caller's with it, then runs on the device where MFEM is on one
        getattr(b, "hypre", None)
        r.assign(b)
        C.Mult(r.hypre, z.hypre)
        d.assign(z)
        d.scale(1.0 / theta)
        x.assign(d)
        if delta > 0.0 and self.degree > 1:
            sigma = theta / delta
            rho = 1.0 / sigma
            from_iterate = self.krylov.precond_type == "amg"
            was = C.iterative_mode
            try:
                for _ in range(1, self.degree):
                    if from_iterate:
                        C.iterative_mode = True
                        z.assign(x)
                        C.Mult(r.hypre, z.hypre)        # z = x + C (b - A x)
                        z.axpy(-1.0, x)
                    else:
                        A.Mult(d.hypre, t.hypre)
                        r.axpy(-1.0, t)
                        C.Mult(r.hypre, z.hypre)
                    new = 1.0 / (2.0 * sigma - rho)
                    d.axpby(2.0 * new / delta, new * rho, z)
                    x.axpy(1.0, d)
                    rho = new
            finally:
                C.iterative_mode = was
        self.iterations = self.degree
        return self.degree


class CycleOf:
    """A Krylov solver's operator with another preconditioner than its own, as
    :class:`ChebyshevSolver` reads a solver: ``cycle`` (``Mult(b, x)`` on MFEM's vectors,
    from zero) in place of the preconditioner of ``krylov``."""

    _single = False
    method = "cg"
    #: (not "amg": the polynomial then applies the cycle to a residual that it updates)
    precond_type = "cycle"

    def __init__(self, cycle, krylov):
        self._pc, self.base = cycle, krylov
        self.comm = krylov.comm

    @property
    def A(self):
        return self.base.A

    def init_vector(self, v, dim):
        return self.base.init_vector(v, dim)


class LUSolver(_SolverBase):
    """Sparse direct solve via scipy's SuperLU, on any number of ranks.

    Stands in for hIPPYlib's ``PETScLUSolver``.  On one rank this factorizes the
    matrix in place.  In parallel it gathers the matrix onto rank 0 and
    factorizes it there (on every rank with ``replicate=True``), which is exact
    and independent of the rank count; see
    :class:`~hippymfem.algorithms.directSolvers.ReplicatedLUSolver`, which this
    delegates to, for the memory trade and the size guard.

    Parameters
    ----------
    comm : mpi4py communicator
    method : str
        Kept for source compatibility; ``"superlu"`` is the only factorization.
    max_global_size : int, optional
        Passed through to the replicated solver.
    replicate : bool
        Passed through: the factorization on every rank rather than on rank 0.
    """

    def __init__(self, comm=None, method="superlu", max_global_size=None,
                 replicate=False):
        comm = comm if comm is not None else MPI.COMM_WORLD
        super(LUSolver, self).__init__(comm)
        self.method = method
        self.max_global_size = max_global_size
        self.replicate = bool(replicate)
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
                self.comm, max_global_size=self.max_global_size,
                replicate=self.replicate)
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
        d = self.d.array
        smallest = float(np.abs(d).min()) if d.size else np.inf
        if self.comm.allreduce(smallest, op=MPI.MIN) < 1e-300:     # every rank raises
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
    return bool(_reduce(v.comm, int(ok), _MPI.MIN))

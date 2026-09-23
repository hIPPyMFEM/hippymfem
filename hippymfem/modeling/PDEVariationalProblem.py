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
"""Stationary forward problem from a pointwise residual density.

The user writes the weak residual of the forward PDE as a density at one
quadrature point; every operator the inverse problem needs follows by automatic
differentiation (see :mod:`hippymfem.fem.kernel`).

The essential-dof treatment of each block (``A`` with a unit diagonal, ``W_uu``
with a zero one, the rectangular blocks on the test side, the transposes
carrying it across) is the table in :mod:`hippymfem.fem.bcs`; with it baked into
the matrices, ``apply_ij`` is a plain matrix product with no masking.
"""

import gc
import os

import numpy as np
from mpi4py import MPI

from ..common.keepalive import KeepAlive
from ..common.parvector import ParVector
from ..fem.assemble import assemble_matrix, assemble_scalar, assemble_vector
from ..fem.bcs import as_bcset
from ..fem.elementbatch import default_quadrature_degree, get_batches
from ..fem.spaces import as_space
from .PDEProblem import PDEProblem
from .variables import ADJOINT, NVAR, PARAMETER, STATE
import mfem.par as mfem
from ..fem.boundary import assemble_boundary_matrix, assemble_boundary_vector, get_boundary_batches
from ..fem.assemble import assembly_backend
from ..fem.csrassemble import (_eliminate, add_boundary_entries, assemble_matrix_csr,
                               finish_block, plan_block, scatter_many)
from ..common.random import Random


#: Take the linearization point's blocks from one differentiation pass instead of
#: one per block.  The blocks are bit-identical either way (for an element batch that
#: is not split) and the shared pass is faster; ``HIPPYMFEM_SHARE_HESSIAN=0`` turns it
#: off.
SHARE_HESSIAN_PASS = os.environ.get("HIPPYMFEM_SHARE_HESSIAN", "1").lower() not in (
    "0", "no", "false", "off")


def set_share_hessian_pass(flag=True):
    """Turn the shared Hessian pass on or off; returns the old value."""
    global SHARE_HESSIAN_PASS
    old, SHARE_HESSIAN_PASS = SHARE_HESSIAN_PASS, bool(flag)
    return old


def _equivalent_spaces(a, b):
    """Whether two spaces are the same space, built twice.

    Same mesh object, same element family and order, same vector dimension and
    ordering, same global size: then one may stand for the other.
    """
    fa, fb = a.fes, b.fes
    try:
        # ``fes.GetMesh()`` hands back a fresh wrapper every call, so the mesh is
        # compared through the space object, which keeps the one it was built on.
        return (a.mesh is b.mesh
                and fa.FEColl().Name() == fb.FEColl().Name()
                and fa.GetVDim() == fb.GetVDim()
                and fa.GetOrdering() == fb.GetOrdering()
                and fa.GlobalTrueVSize() == fb.GlobalTrueVSize())
    except Exception:                                        # noqa: BLE001
        return False


class PDEVariationalProblem(PDEProblem, KeepAlive):
    """Forward problem defined by a pointwise residual density.

    Parameters
    ----------
    Vh : sequence of 3 spaces
        ``[STATE, PARAMETER, ADJOINT]``; each a
        :class:`~hippymfem.fem.spaces.FunctionSpace` or a raw
        ``ParFiniteElementSpace``.
    varf_handler : callable
        ``varf_handler(u, m, p, x) -> scalar``, JAX-traceable, with
        :class:`~hippymfem.fem.kernel.Field` arguments.  This is the weak
        residual density: the forward problem is ``dR/dp = 0``.
    bc : DirichletBC, list, BCSet, or None
        Essential conditions **with data**, for the forward solve.
    bc0 : same
        Homogeneous counterpart, for the adjoint and incremental solves.
    is_fwd_linear : bool
        Declares the residual linear in ``u``.  One Newton step is then exact;
        the residual is still checked, so a mislabeled nonlinear problem is
        reported rather than silently mis-solved.
    quadrature_degree : int, optional
        Integration rule degree.  Defaults to ``2*max(order) + 2``.  Every
        derivative block uses this one rule, so they stay mutually consistent.
    aux_spaces : sequence of spaces, optional
        Extra fields appended to ``varf_handler``'s signature; set their values
        through :attr:`aux_values`.
    bdr_varf : callable, optional
        ``bdr_varf(u, m, p, x, n, *aux) -> scalar``, a **boundary** residual
        density, with ``n`` the outward unit normal.  This is the ``ds`` of the
        UFL form: Neumann and Robin conditions, boundary sources, penalty terms.
        It is differentiated exactly like the domain density, so every block
        picks up its boundary part; see :mod:`hippymfem.fem.boundary`.  The
        fields carry their full gradient there, so ``dot(u.grad, n)`` works.
    bdr_attributes : sequence of int or "all", optional
        Which boundary attributes ``bdr_varf`` applies to (MFEM's 1-based
        numbering).
    bdr_quadrature_degree : int, optional
        Rule degree on the boundary; defaults to ``quadrature_degree``.
    facet_varf : callable, optional
        ``facet_varf(u, m, p, x, n, h, *aux) -> scalar``, an **interior facet**
        residual density, the ``dS`` of a discontinuous Galerkin form.  Each field
        arrives as a :class:`~hippymfem.fem.facets.FacetField` carrying its trace from
        both sides, so the form is written in ``jump`` and ``avg``; ``n`` is the normal
        out of the first element and ``h`` the pair of face measures.  It is
        differentiated like the domain density, so the residual and every block pick up
        their facet part; see :mod:`hippymfem.fem.facets`.
    facet_quadrature_degree : int, optional
        Rule degree on interior faces; defaults to ``quadrature_degree``.
    symmetric_jacobian : bool or "auto", optional
        Whether ``dR/du`` is symmetric; ``"auto"`` (the default) probes the
        assembled matrix.  See :attr:`symmetric_jacobian`.
    spd_jacobian : bool, optional
        Declare ``dR/du`` symmetric positive definite, so the default solvers use
        CG with BoomerAMG instead of GMRES (see
        :attr:`~.PDEProblem.PDEProblem.spd_jacobian`).  Implies
        ``symmetric_jacobian=True``.
    release_linearization_on_move : bool
        Drop the linearization point at a forward solve for a different parameter;
        see :attr:`release_linearization_on_move`.
    transpose_free_adjoint : bool
        Solve the adjoint through ``A``'s ``MultTranspose`` with ``A``'s own AMG
        hierarchy, without forming ``A^T``; see :attr:`transpose_free_adjoint`.
    """

    def __init__(self, Vh, varf_handler, bc=None, bc0=None, is_fwd_linear=False,
                 quadrature_degree=None, aux_spaces=(), bdr_varf=None,
                 bdr_attributes="all", bdr_quadrature_degree=None,
                 symmetric_jacobian="auto", release_linearization_on_move=False,
                 transpose_free_adjoint=False, spd_jacobian=False,
                 facet_varf=None, facet_quadrature_degree=None):
        if len(Vh) != NVAR:
            raise ValueError("Vh must have %d entries" % NVAR)
        #: see :attr:`~.PDEProblem.PDEProblem.spd_jacobian`; SPD implies symmetric,
        #: so the probe is skipped
        self.spd_jacobian = bool(spd_jacobian)
        if self.spd_jacobian and symmetric_jacobian == "auto":
            symmetric_jacobian = True
        self.Vh = [as_space(v) for v in Vh]
        # The adjoint lives in the state space, and hIPPYlib's examples often build the
        # two separately.  On the host that is harmless; with MFEM and hypre on a device
        # it is not, because two ParFiniteElementSpaces over one mesh produce matrices
        # whose device mirrors collide: BoomerAMG's setup fails ("Error code: 12") and
        # the solve returns NaN, with nothing to say why.  Equivalent spaces are
        # therefore aliased here, which changes no mathematics and removes the trap.
        if self.Vh[STATE] is not self.Vh[ADJOINT] and _equivalent_spaces(
                self.Vh[STATE], self.Vh[ADJOINT]):
            self.Vh[ADJOINT] = self.Vh[STATE]
        self.varf_handler = varf_handler
        self.is_fwd_linear = bool(is_fwd_linear)
        #: Whether ``dR/du`` is symmetric.  If it is, the adjoint and
        #: adjoint-incremental solves reuse the Jacobian and its preconditioner,
        #: saving a transposed copy, a second AMG hierarchy and one BoomerAMG setup
        #: per Newton step.
        #:
        #: ``"auto"``, the default, decides from the assembled matrix rather than
        #: from a promise about the PDE: every newly assembled Jacobian is probed with
        #: :meth:`_probe_symmetry` (four matrix-vector products, far cheaper than the
        #: AMG setup they can save), so a residual that is self-adjoint at one
        #: parameter and not at another is handled correctly.  ``True`` and ``False``
        #: declare it outright and skip the probe.
        self.symmetric_jacobian = (
            "auto" if symmetric_jacobian == "auto" else bool(symmetric_jacobian))
        #: At a forward solve for a parameter other than the linearization point's,
        #: release that point (``C``, the ``W`` blocks, ``A``/``At`` and the
        #: incremental solvers' hierarchies) before the new Jacobian is assembled, so
        #: a line search never holds the old point's operators beside the trial
        #: point's.  Opt-in, because it is a claim about the caller: a line-search
        #: Newton-CG has its direction before it moves and rebuilds the point after,
        #: but a trust-region step, a finite-difference check or anything that applies
        #: the Hessian after a trial solve needs the old point to survive it.  See
        #: the GPU guide.
        self.release_linearization_on_move = bool(release_linearization_on_move)
        #: Solve with ``A^T`` through ``A``'s ``MultTranspose`` and ``A``'s own AMG
        #: hierarchy as the preconditioner, instead of forming the transpose and a
        #: second hierarchy.  For a residual that is not linear in the state a Newton
        #: step then holds ``A`` and one hierarchy instead of ``A``, ``A^T`` and two,
        #: which can decide whether a large problem fits in device memory.  GMRES
        #: accepts ``A``'s V-cycle as a preconditioner for ``A^T``; it is as good as
        #: the asymmetry is small (for example the ``k'(u)`` term of a
        #: temperature-dependent conductivity).  Krylov adjoint solvers only; a
        #: direct solver keeps the explicit transpose.
        self.transpose_free_adjoint = bool(transpose_free_adjoint)
        self.comm = self.Vh[STATE].comm
        self.mesh = self.Vh[STATE].mesh
        self.aux_spaces = [as_space(s) for s in aux_spaces]
        #: true-dof vectors for the auxiliary fields, in declaration order
        self.aux_values = [s.vector() for s in self.aux_spaces]

        self.bc = as_bcset(bc, self.Vh[STATE])
        self.bc0 = as_bcset(bc0 if bc0 is not None else
                            (self.bc.homogeneous() if self.bc else None),
                            self.Vh[STATE])

        if quadrature_degree is None:
            quadrature_degree = default_quadrature_degree(self.Vh + self.aux_spaces)
        self.quadrature_degree = int(quadrature_degree)
        self.batches = get_batches(self.mesh, self.quadrature_degree)

        from ..fem.kernel import QuadratureKernel

        # auxiliary fields occupy slots after ADJOINT; they are differentiable
        # like any other slot, which costs nothing and keeps one code path
        self.kernel = QuadratureKernel(
            varf_handler, self.Vh + self.aux_spaces, self.batches
        )
        self.nelem = self.mesh.GetNE()

        # ------------------------------------------------------ boundary term
        self.bdr_varf = bdr_varf
        self.bdr_kernel = None
        self.bdr_batches = None
        if bdr_varf is not None:
            from ..fem.boundary import BoundaryKernel   # resolved from the JAX kernel module: lazy

            self.bdr_quadrature_degree = int(
                bdr_quadrature_degree if bdr_quadrature_degree is not None
                else self.quadrature_degree)
            self.bdr_batches = get_boundary_batches(
                self.mesh, self.bdr_quadrature_degree, bdr_attributes,
                self.comm, space=self.Vh[STATE])
            self.bdr_kernel = BoundaryKernel(
                bdr_varf, self.Vh + self.aux_spaces, self.bdr_batches)

        # --------------------------------------------------------- facet term
        self.facet_varf = facet_varf
        self.facet_kernel = None
        self.facet_batches = None
        if facet_varf is not None:
            from ..fem.facets import get_facet_batches

            self.facet_quadrature_degree = int(
                facet_quadrature_degree if facet_quadrature_degree is not None
                else self.quadrature_degree)
            self.facet_batches = get_facet_batches(
                self.mesh, self.facet_quadrature_degree, space=self.Vh[STATE],
                comm=self.comm)
            # The facet slots are the same variables; a two-sided evaluator and the
            # face geometry are all that differ, so one kernel class serves both.
            self.facet_kernel = QuadratureKernel(
                facet_varf, self.Vh + self.aux_spaces, self.facet_batches)

        # assembled blocks, filled by setLinearizationPoint
        self.A = None
        self.At = None
        self.C = None
        self.Wuu = None
        self.Wum = None
        self.Wmm = None

        # the four solvers (see PDEProblem) start unset and default on first use
        self.solver = self.solver_adj = None
        self.solver_fwd_inc = self.solver_adj_inc = None

        #: Newton controls for a nonlinear forward solve
        self.newton_parameters = {
            "rel_tolerance": 1e-10,
            "abs_tolerance": 1e-14,
            "max_iter": 25,
            "line_search": True,
            "max_backtrack": 12,
            "print_level": -1,
        }
        self.n_calls = {"forward": 0, "adjoint": 0,
                        "incremental_forward": 0, "incremental_adjoint": 0}
        self._lin_point = None
        #: last assembled forward Jacobian and the point it was built at
        self._jac_cache = None
        #: private generator for the symmetry probe (see :meth:`_probe_symmetry`)
        self._symmetry_rng = None

    # ------------------------------------------------------------------ shapes
    def generate_state(self):
        return self.Vh[STATE].vector()

    def generate_parameter(self):
        return self.Vh[PARAMETER].vector()

    def generate_adjoint(self):
        return self.Vh[ADJOINT].vector()

    def init_parameter(self, m):
        if isinstance(m, ParVector):
            return m
        from ..common.operators import init_vector_like

        return init_vector_like(m, self.generate_parameter())

    # ------------------------------------------------------------ local gather
    def _locals(self, x):
        """Local (ghosted) dof arrays for ``x = [u, m, p]``, zeros where absent."""
        out = []
        for i in range(NVAR):
            v = x[i] if x[i] is not None else self.Vh[i].vector()
            out.append(self.Vh[i].local_values(v))
        return out

    def _aux_locals(self):
        return [sp.local_values(v)
                for sp, v in zip(self.aux_spaces, self.aux_values)]

    def _facet_locals(self, x):
        """Dof values for the facet kernel: this rank's, then its face neighbours'.

        The far side of a face shared with another rank is in no vector of this rank;
        :func:`~hippymfem.fem.facets.facet_values` appends the neighbour's values and
        the facet gathers index the two as one array.  Collective.
        """
        from ..fem.facets import facet_values

        out = [facet_values(self.Vh[i], x[i] if x[i] is not None
                            else self.Vh[i].vector()) for i in range(NVAR)]
        return out + [facet_values(sp, v)
                      for sp, v in zip(self.aux_spaces, self.aux_values)]

    # -------------------------------------------------------------- assembly
    def _residual(self, x, var=ADJOINT, ess=None):
        """Assemble ``dR/d(var)`` at ``x`` as a true-dof vector."""
        loc = self._locals(x) + self._aux_locals()
        vecs = self.kernel.element_vectors(var, loc)
        require_finite_arrays(vecs, self.comm, "the residual")
        out = assemble_vector(self.Vh[var], self.batches.groups, vecs,
                              self.nelem)
        if self.bdr_kernel is not None:

            bvecs = self.bdr_kernel.element_vectors(var, loc)
            require_finite_arrays(bvecs, self.comm, "the boundary residual")
            out.axpy(1.0, assemble_boundary_vector(
                self.Vh[var], self.bdr_batches.groups, bvecs))
        if self.facet_kernel is not None:
            from ..fem.facets import assemble_facet_vector

            floc = self._facet_locals(x)
            fvecs = self.facet_kernel.element_vectors(var, floc)
            require_finite_arrays(fvecs, self.comm, "the facet residual")
            out.axpy(1.0, assemble_facet_vector(
                self.Vh[var], self.facet_batches.groups, fvecs,
                tables=self.facet_batches.tables(self.Vh[var])))
        if ess is not None and len(ess):
            out.array[np.asarray(ess, dtype=np.int64)] = 0.0
        return out

    def _block(self, i, j, x, test_ess=None, diag_policy="one", mats=None,
               loc=None, floc=None):
        """Assemble block ``(i, j)`` at ``x``, with its facet part if there is one."""
        if self.facet_kernel is None:
            return self._domain_block(i, j, x, test_ess, diag_policy, mats, loc)
        from ..common.linalg import ParAdd
        from ..fem.facets import assemble_facet_matrix

        # Domain and facet parts are summed before the essential rows are eliminated:
        # eliminating each and adding afterwards would leave 2.0 on the essential
        # diagonal.
        A = self._domain_block(i, j, x, None, diag_policy, mats, loc)
        fmats = self.facet_kernel.element_matrices(
            i, j, self._facet_locals(x) if floc is None else floc)
        require_finite_arrays(fmats, self.comm, "facet block (%d, %d)" % (i, j))
        fac = assemble_facet_matrix(
            self.Vh[i], self.facet_batches.groups, fmats, trial_space=self.Vh[j],
            test_tables=self.facet_batches.tables(self.Vh[i]),
            trial_tables=self.facet_batches.tables(self.Vh[j]))
        A = ParAdd(A, fac)
        del fac
        return _eliminate(A, self.Vh[i], self.Vh[j], test_ess, None, diag_policy,
                          self.Vh[i].fes is self.Vh[j].fes)

    def _domain_block(self, i, j, x, test_ess=None, diag_policy="one", mats=None,
                      loc=None):
        """Assemble block ``(i, j)`` at ``x``.

        ``mats`` and ``loc`` let a caller that already differentiated at this
        point hand the element arrays in rather than paying for another pass, as
        :meth:`setLinearizationPoint` does for its blocks.
        """
        if loc is None:
            loc = self._locals(x) + self._aux_locals()
        if mats is None:
            # Arrays when the element batch fits on the device whole, a chunk thunk
            # when it does not: the scatter then consumes each chunk as it is
            # produced and the full (ne, nd, nd) array is never formed.
            mats = self.kernel.element_matrices_or_chunks(i, j, loc)
        finite = None
        if callable(mats):
            # The arrays never exist all at once, so the overflow check rides along
            # with the chunks: each reduces to a flag on its own device and the one
            # collective comes after the loop.  An allreduce per chunk would
            # desynchronize the ranks, which need not have the same number of chunks.
            finite = _FiniteFlag()
            mats = finite.wrap(mats)
        else:
            require_finite_arrays(mats, self.comm, "block (%d, %d)" % (i, j))
        if self.bdr_kernel is None:
            A = assemble_matrix(self.Vh[i], self.Vh[j], self.batches.groups,
                                mats, self.nelem, test_ess=test_ess,
                                diag_policy=diag_policy)
            if finite is not None:
                finite.check(self.comm, "block (%d, %d)" % (i, j))
            return A
        bmats = self.bdr_kernel.element_matrices(i, j, loc)
        require_finite_arrays(bmats, self.comm,
                              "boundary block (%d, %d)" % (i, j))
        if assembly_backend() == "csr":
            # The boundary entries go into the domain block's own slots, so one
            # matrix comes out with the elimination folded in (see
            # csrassemble.add_boundary_entries).
            A = assemble_matrix_csr(self.Vh[i], self.Vh[j], self.batches.groups,
                                    mats, self.nelem, test_ess=test_ess,
                                    diag_policy=diag_policy,
                                    boundary=self._boundary_arrays(i, j, bmats))
            if finite is not None:
                finite.check(self.comm, "block (%d, %d)" % (i, j))
            return A
        # The callback route: the two parts are summed *before* elimination, since
        # eliminating each and then adding would leave 2.0 on the essential diagonal.
        from ..common.linalg import ParAdd

        dom = assemble_matrix(self.Vh[i], self.Vh[j], self.batches.groups, mats,
                              self.nelem)
        bdr = assemble_boundary_matrix(self.Vh[i], self.Vh[j],
                                       self.bdr_batches.groups, bmats)
        A = ParAdd(dom, bdr)
        del dom, bdr
        return _eliminate(A, self.Vh[i], self.Vh[j], test_ess, None,
                          diag_policy, self.Vh[i].fes is self.Vh[j].fes)

    def _boundary_arrays(self, i, j, bmats):
        """``(test_tables, trial_tables, element_matrices)`` of block ``(i, j)``'s
        boundary part, for :func:`~hippymfem.fem.csrassemble.add_boundary_entries`."""
        return (self.bdr_batches.tables(self.Vh[i]), self.bdr_batches.tables(self.Vh[j]),
                bmats)

    def _jacobian(self, x, u=None):
        r"""The assembled forward Jacobian ``A`` and its transpose at ``x``.

        Reused while the point has not moved: assembling ``A`` and building its AMG
        preconditioner dominate the cost of a gradient-only optimizer such as BFGS,
        whose ``solveFwd`` and ``solveAdj`` at each iteration need the same matrix.

        When the residual is linear in the state, ``A`` depends on the parameter
        alone, so the parameter is the whole key; otherwise the state is part of
        it too and a moved state forces a rebuild.
        """
        u = x[STATE] if u is None else u
        m = x[PARAMETER]
        require_finite(m, "the parameter")
        if not self.is_fwd_linear:
            require_finite(u, "the state")
        if self._jac_cache is not None:
            cu, cm, A, At = self._jac_cache
            if _same(cm, m) and (self.is_fwd_linear or _same(cu, u)):
                return A, At
        # The point moved: drop the old Jacobian and the solvers built on it *before*
        # assembling the new one, so a Newton step never holds two of each at its
        # peak.  The incremental solvers and the linearization point's ``A``/``At``
        # stay: a forward solve at a perturbed point (a line search, a trust-region
        # trial, a finite-difference check) must leave the linearization point
        # usable, so only ``setLinearizationPoint`` and
        # ``release_linearization_point`` release them.
        self._jac_cache = None
        self._release_operators("solver", "solver_adj")
        xx = [u, m, x[ADJOINT] if len(x) > ADJOINT else None]
        A = self._block(ADJOINT, STATE, xx, test_ess=self.bc0.ess_tdof,
                        diag_policy="one")
        symmetric = (self._probe_symmetry(A) if self.symmetric_jacobian == "auto"
                     else self.symmetric_jacobian)
        if symmetric:
            At = A
        elif self.transpose_free_adjoint and isinstance(self._get_solver("solver"), _krylov_class()):
            At = _transpose_operator(A)
        else:
            At = A.Transpose()
        self._jac_cache = (None if self.is_fwd_linear else u.copy(), m.copy(),
                           A, At)
        return A, At

    #: relative tolerance of the symmetry probe, and how many vector pairs it uses
    SYMMETRY_PROBE_TOL = 1e-13
    SYMMETRY_PROBE_VECTORS = 2

    def _probe_symmetry(self, A):
        """Whether the assembled Jacobian is symmetric, to round-off.

        Compares the bilinear form both ways, ``w^T (A v)`` against ``v^T (A w)``, for
        a couple of random pairs, against the Cauchy-Schwarz bound
        ``max(|w| |A v|, |v| |A w|)``.  Only ``Mult`` is used: hypre's device
        ``MultTranspose`` forms the transposed matrix, a full copy of the Jacobian,
        on every call.  The vectors come from a counter-based generator, so every
        rank probes the same ones and the verdict is partition independent.

        For a symmetric matrix the two products differ by the rounding of two inner
        products, a few machine epsilons relative to the bound.  For a random pair the
        skew part ``S = (A - A^T)/2`` shows up at about ``|S|_F / (sqrt(n) |A|_F)``
        relative, so at the default tolerance a skew part above ~1e-10 of the matrix
        (for a million unknowns) is caught; a false positive, which would make the
        adjoint use ``A`` for ``A^T``, is confined to asymmetries below that.
        """

        # A private generator, never the library's ``parRandom``: drawing from the
        # shared stream here would shift every prior sample and noise vector drawn
        # afterwards.
        if self._symmetry_rng is None:
            self._symmetry_rng = Random(seed=20260914)
        rng = self._symmetry_rng
        rng.set_seed(rng.seed)                       # same vectors at every probe
        v, w = self.generate_state(), self.generate_state()
        Av, Aw = self.generate_state(), self.generate_state()
        for _ in range(self.SYMMETRY_PROBE_VECTORS):
            rng.normal(1.0, v)
            rng.normal(1.0, w)
            A.Mult(v.hypre, Av.hypre)
            A.Mult(w.hypre, Aw.hypre)
            bound = max(w.norm("l2") * Av.norm("l2"), v.norm("l2") * Aw.norm("l2"))
            if bound == 0.0:
                continue
            if abs(w.inner(Av) - v.inner(Aw)) > self.SYMMETRY_PROBE_TOL * bound:
                return False
        return True

    def _jacobian_is_current(self, x):
        """Whether ``_jacobian(x)`` would be a cache hit (no assembly, no release)."""
        if self._jac_cache is None:
            return False
        cu, cm, _A, _At = self._jac_cache
        return _same(cm, x[PARAMETER]) and (self.is_fwd_linear or _same(cu, x[STATE]))

    def invalidate_jacobian(self):
        """Drop the cached Jacobian; call this if a coefficient changed in place."""
        self._jac_cache = None
        return self

    def release_linearization_point(self):
        """Drop every operator of the linearization point; the next use must set one.

        One ``gc.collect`` at the end frees the dropped hypre matrices before the
        next point's blocks exist, which lowers the memory peak.
        """
        self.C = self.Wuu = self.Wum = self.Wmm = None
        self._release_operators("solver_fwd_inc", "solver_adj_inc")
        self.A = self.At = None
        self._lin_point = None
        gc.collect()
        return self

    # -------------------------------------------------------------- forward
    def solveFwd(self, state, x):
        """Newton solve of ``dR/dp = 0`` for the state, given ``x[PARAMETER]``.

        A linear residual converges in one step; the declaration
        ``is_fwd_linear`` only stops the loop early, it does not change the step.
        """
        self.n_calls["forward"] += 1
        m = x[PARAMETER]
        require_finite(m, "the parameter passed to solveFwd")
        lp = self._lin_point
        if (self.release_linearization_on_move and lp is not None
                and lp[PARAMETER] is not None and not _same(lp[PARAMETER], m)):
            self.release_linearization_point()
        p = self.Vh[ADJOINT].vector()          # the residual is linear in p

        # start from the boundary data (zero elsewhere), or the incoming guess
        u = state
        if not self.is_fwd_linear and state.norm("linf") > 0.0:
            pass                                # warm start from the caller
        else:
            u.zero()
        self.bc.apply(u)

        prm = self.newton_parameters
        solver = self._get_solver("solver")
        du = self.Vh[STATE].vector()
        r = self._residual([u, m, p], ADJOINT, ess=self.bc0.ess)
        require_finite(r, "the forward residual")
        r0 = r.norm("l2")
        tol = max(prm["rel_tolerance"] * r0, prm["abs_tolerance"])
        self.fwd_iterations = 0

        maxit = 1 if self.is_fwd_linear else int(prm["max_iter"])
        for it in range(maxit):
            J, _Jt = self._jacobian([u, m, p], u=u)
            _set_operator_once(solver, J)
            r.scale(-1.0)
            solver.solve(du, r)
            alpha = 1.0
            if prm["line_search"] and not self.is_fwd_linear:
                alpha, rnew, r = self._backtrack(u, du, m, p, r.norm("l2"))
            else:
                u.axpy(1.0, du)
                r = self._residual([u, m, p], ADJOINT, ess=self.bc0.ess)
                rnew = r.norm("l2")
            self.fwd_iterations = it + 1
            if prm["print_level"] >= 0 and self.comm.rank == 0:
                print("  fwd Newton %2d: ||r|| = %.6e  alpha = %.3g"
                      % (it + 1, rnew, alpha), flush=True)
            if rnew < tol:
                break
        else:
            if not self.is_fwd_linear:
                raise RuntimeError(
                    "forward Newton solve did not converge: ||r|| = %.3e, "
                    "tolerance %.3e after %d iterations" % (rnew, tol, maxit)
                )
        if self.is_fwd_linear:
            rn = r.norm("l2")
            # The floor is the element kernel's precision, not the solver's: in fp32
            # one Newton step on a linear residual lands at about 1e-5 relative (the
            # assembled operator's own error, not a nonlinearity), which the fp64
            # threshold would reject.
            from ..fem.kernel import PRECISION as _KPREC
            rtol, atol = ((1e-4, 1e-6) if _KPREC == "fp32" else (1e-6, 1e-8))
            if rn > max(rtol * max(r0, 1.0), atol):
                raise RuntimeError(
                    "is_fwd_linear=True but one Newton step left ||r|| = %.3e "
                    "(started at %.3e); the residual is not linear in the state."
                    % (rn, r0)
                )
        return state

    def _backtrack(self, u, du, m, p, rprev):
        """Backtracking line search on the residual norm."""
        alpha = 1.0
        u0 = u.copy()
        for _ in range(int(self.newton_parameters["max_backtrack"])):
            u.assign(u0).axpy(alpha, du)
            r = self._residual([u, m, p], ADJOINT, ess=self.bc0.ess)
            rn = r.norm("l2")
            if rn < rprev or alpha < 1e-8:
                return alpha, rn, r
            alpha *= 0.5
        return alpha, rn, r

    # -------------------------------------------------------------- adjoint
    def solveAdj(self, adj, x, adj_rhs):
        """Solve ``A^T p = adj_rhs`` with homogeneous essential conditions."""
        self.n_calls["adjoint"] += 1
        A, At = self._jacobian(x)
        if getattr(At, "transposed_of", None) is A:
            fwd = self._get_solver("solver")
            _set_operator_once(fwd, A)
            solver = self._get_solver("solver_adj")
            _set_operator_once(solver, At, share_from=fwd)
        else:
            solver = self._get_solver("solver")
            _set_operator_once(solver, At)
        rhs = adj_rhs.copy()
        self.bc0.zero(rhs)
        adj.zero()
        solver.solve(adj, rhs)
        return adj

    def evalGradientParameter(self, x, out):
        """``out = dR/dm`` at ``x``; equals ``C^T p`` but needs no matrix."""
        g = self._residual(x, PARAMETER)
        out.assign(g)
        return out

    # ------------------------------------------------------ linearization point
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """Assemble the blocks used by the incremental solves and the Hessian."""
        self._lin_point = [v.copy() if v is not None else None for v in x]
        # The previous point's blocks are replaced below; dropping them first keeps
        # the peak at one set of blocks rather than two (the Jacobian and the solvers
        # are handled the same way in ``_jacobian`` when the point has moved).
        self.C = self.Wuu = self.Wum = self.Wmm = None
        if not self._jacobian_is_current(x):
            # A new Jacobian is coming, so the incremental solvers' operators and AMG
            # hierarchies (built on the old one) go first.  At the same point they are
            # kept, and ``_set_operator_once`` below skips the rebuild.
            self._release_operators("solver_fwd_inc", "solver_adj_inc")
            self.A = self.At = None
        ess = self.bc0.ess_tdof

        self.gauss_newton_approx = bool(gauss_newton_approx)
        # Every block below is a slice of the same element Hessian, so they are
        # differentiated once rather than once each.  A Gauss-Newton point needs
        # only C; a full one also needs the W blocks.
        need = [(ADJOINT, PARAMETER)]
        if not gauss_newton_approx:
            # W_uu is p . d2R/du2, identically zero when the residual is linear in
            # the state, which is_fwd_linear declares (and solveFwd checks).  Not
            # assembling it saves the largest block (the state-state pattern, the
            # size of A) and the state columns of the differentiation pass.
            if not self.is_fwd_linear:
                need.append((STATE, STATE))
            need += [(STATE, PARAMETER), (PARAMETER, PARAMETER)]
        loc = self._locals(x) + self._aux_locals()
        streamed = None
        if len(need) > 1 and SHARE_HESSIAN_PASS and self._stream_shared_pass():
            # One pass, a chunk at a time, straight into every block's scatter:
            # nothing larger than a chunk is ever resident, where gluing the blocks
            # first (what element_matrices_many does) would make the glued arrays
            # the device peak whatever the chunk size.
            streamed = self._blocks_streamed(need, loc, ess)
            mats = None
        elif len(need) > 1 and SHARE_HESSIAN_PASS:
            mats = self.kernel.element_matrices_many(need, loc)
        else:
            mats = {ij: self.kernel.element_matrices(ij[0], ij[1], loc)
                    for ij in need}
        self.A, self.At = self._jacobian(x)
        if streamed is not None:
            self.C = streamed[(ADJOINT, PARAMETER)]
        else:
            self.C = self._block(ADJOINT, PARAMETER, x, test_ess=ess,
                                 mats=mats[(ADJOINT, PARAMETER)], loc=loc)
        # The attributes alone hold the blocks: anything else keeping them would
        # keep every linearization point's blocks alive for the life of the problem.
        if gauss_newton_approx:
            self.Wuu = None
            self.Wum = None
            self.Wmm = None
        elif streamed is not None:
            self.Wuu = streamed.get((STATE, STATE))
            self.Wum = streamed[(STATE, PARAMETER)]
            self.Wmm = streamed[(PARAMETER, PARAMETER)]
        else:
            self.Wuu = None if self.is_fwd_linear else self._block(
                STATE, STATE, x, test_ess=ess, diag_policy="zero",
                mats=mats[(STATE, STATE)], loc=loc)
            self.Wum = self._block(STATE, PARAMETER, x, test_ess=ess,
                                   mats=mats[(STATE, PARAMETER)], loc=loc)
            self.Wmm = self._block(PARAMETER, PARAMETER, x,
                                   mats=mats[(PARAMETER, PARAMETER)], loc=loc)
        del mats, streamed
        fwd = self._get_solver("solver_fwd_inc")
        adj = self._get_solver("solver_adj_inc")
        _set_operator_once(fwd, self.A, share_from=self.solver)
        _set_operator_once(adj, self.At, share_from=fwd if (
            self.At is self.A or getattr(self.At, "transposed_of", None) is self.A) else None)
        return self

    def _stream_shared_pass(self):
        """Whether the linearization point can be assembled a chunk at a time.

        It needs the direct-CSR route, no interior-facet term, and a batch the
        device would split anyway: the condition under which a single block takes
        the fused scatter, since splitting has already given up bit-identity with
        an unsplit run.  Whether a batch splits is a per-rank fact (its element
        count, its device), and the two branches make different collectives, so the
        per-rank answer is reduced first and every rank takes the same branch.
        """

        if assembly_backend() != "csr" or self.facet_kernel is not None:
            # A facet block is a second matrix that has to be added before the
            # essential rows go, and the streamed pass finishes each block with the
            # elimination already folded in.
            return False
        local = self.kernel.will_chunk(weight=self.kernel.nslots)
        return bool(self.comm.allreduce(int(local), op=MPI.MAX))

    def _blocks_streamed(self, need, loc, ess):
        """The blocks of a linearization point from one streamed pass."""

        spec = {(ADJOINT, PARAMETER): (ess, "one"), (STATE, STATE): (ess, "zero"),
                (STATE, PARAMETER): (ess, "one"), (PARAMETER, PARAMETER): (None, "one")}
        plans = {}
        for ij in need:
            e, pol = spec[ij]
            plans[ij] = plan_block(self.Vh[ij[0]], self.Vh[ij[1]],
                                   self.batches.groups, test_ess=e, diag_policy=pol)
        finite = _FiniteFlag()
        chunks = finite.wrap(lambda: self.kernel.element_matrix_chunks_many(need, loc))
        accs = scatter_many(plans, chunks())
        finite.check(self.comm, "linearization point")
        if self.bdr_kernel is not None:
            # the boundary residual's blocks, one pass over the boundary elements
            bmats = self.bdr_kernel.element_matrices_many(need, loc)
            for ij in need:
                require_finite_arrays(bmats[ij], self.comm, "boundary block %s" % (ij,))
                accs[ij] = add_boundary_entries(
                    plans[ij], accs[ij], self._boundary_arrays(ij[0], ij[1], bmats[ij]))
        return {ij: finish_block(plans[ij], accs[ij]) for ij in need}

    def solveIncremental(self, out, rhs, is_adj):
        """Solve the incremental forward or adjoint system."""
        r = rhs.copy()
        self.bc0.zero(r)
        out.zero()
        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            self._get_solver("solver_adj_inc").solve(out, r)
        else:
            self.n_calls["incremental_forward"] += 1
            self._get_solver("solver_fwd_inc").solve(out, r)
        return out

    # ------------------------------------------------------------- derivatives
    def apply_ij(self, i, j, dir, out):
        """Apply the ``(i, j)`` second-derivative block to ``dir``."""
        if self.A is None:
            raise RuntimeError("setLinearizationPoint must be called first")
        KKT = {
            (ADJOINT, STATE): (self.A, False),
            (STATE, ADJOINT): (self.A, True),
            (ADJOINT, PARAMETER): (self.C, False),
            (PARAMETER, ADJOINT): (self.C, True),
            (STATE, STATE): (self.Wuu, False),
            (STATE, PARAMETER): (self.Wum, False),
            (PARAMETER, STATE): (self.Wum, True),
            (PARAMETER, PARAMETER): (self.Wmm, False),
        }
        if (i, j) not in KKT:
            raise ValueError("no block (%d, %d)" % (i, j))
        mat, transpose = KKT[(i, j)]
        if mat is None:                         # Gauss-Newton: second order = 0
            out.zero()
            return out
        # hypre does not check sizes: a wrong output vector is written past its
        # buffer and the heap is corrupted, which surfaces much later and elsewhere
        nrow, ncol = (mat.Width(), mat.Height()) if transpose else (mat.Height(), mat.Width())
        if out.local_size != nrow or dir.local_size != ncol:
            raise ValueError(
                "apply_ij(%d, %d): block is %d x %d locally, got dir of size %d and "
                "out of size %d" % (i, j, nrow, ncol, dir.local_size, out.local_size))
        if transpose:
            mat.MultTranspose(dir.hypre, out.hypre)
        else:
            mat.Mult(dir.hypre, out.hypre)
        return out

    def apply_ijk(self, i, j, k, x, jdir, kdir, out):
        """Third-derivative block contracted with ``jdir`` and ``kdir``."""
        loc = self._locals(x) + self._aux_locals()
        jl = self.Vh[j].local_values(jdir)
        kl = self.Vh[k].local_values(kdir)
        vecs = self.kernel.element_third(i, j, k, loc, jl, kl)
        res = assemble_vector(self.Vh[i], self.batches.groups, vecs, self.nelem)
        if self.bdr_kernel is not None:

            res.axpy(1.0, assemble_boundary_vector(
                self.Vh[i], self.bdr_batches.groups,
                self.bdr_kernel.element_third(i, j, k, loc, jl, kl)))
        if self.facet_kernel is not None:
            from ..fem.facets import assemble_facet_vector, facet_values

            floc = self._facet_locals(x)
            res.axpy(1.0, assemble_facet_vector(
                self.Vh[i], self.facet_batches.groups,
                self.facet_kernel.element_third(
                    i, j, k, floc, facet_values(self.Vh[j], jdir),
                    facet_values(self.Vh[k], kdir)),
                tables=self.facet_batches.tables(self.Vh[i])))
        ess = self.bc0.ess if i in (STATE, ADJOINT) else None
        if ess is not None and len(ess):
            res.array[np.asarray(ess, dtype=np.int64)] = 0.0
        out.assign(res)
        return out

    def apply_third_dir(self, i, x, dirs, weights, out):
        r"""Weighted second directional derivatives of the slot-``i`` gradient,
        :math:`\sum_m w_m\, D^2(\partial_i R)[t_m, t_m]`.

        ``dirs`` holds one direction ``t_m`` per weight ``w_m``, each a sequence
        ``(u, m, p)`` of vectors (or longer, when :attr:`Vh` lists further
        variables), ``None`` where the direction has no component.  It
        equals the sum of :meth:`apply_ijk` over every ordered pair of nonzero
        components, :math:`\sum_{j,k} R_{ijk}[t_j, t_k]`, but the quadrature
        kernel differentiates the whole direction at once: one pass instead of up
        to nine, which is what a second-order adjoint over many modes spends its
        time in.
        """
        dirs = [tuple(d) for d in dirs]
        weights = [float(w) for w in weights]
        if len(dirs) != len(weights):
            raise ValueError("apply_third_dir: %d directions, %d weights"
                             % (len(dirs), len(weights)))
        if not dirs:
            out.zero()
            return out
        loc = self._locals(x) + self._aux_locals()
        dl = [tuple(None if d[s] is None else self.Vh[s].local_values(d[s])
                    for s in range(len(d))) + (None,) * (len(loc) - len(d))
              for d in dirs]
        vecs = self.kernel.element_third_dir(i, loc, dl, weights)
        res = assemble_vector(self.Vh[i], self.batches.groups, vecs, self.nelem)
        if self.bdr_kernel is not None:
            res.axpy(1.0, assemble_boundary_vector(
                self.Vh[i], self.bdr_batches.groups,
                self.bdr_kernel.element_third_dir(i, loc, dl, weights)))
        if self.facet_kernel is not None:
            from ..fem.facets import assemble_facet_vector, facet_values

            floc = self._facet_locals(x)
            for d, w in zip(dirs, weights):
                for j in range(len(d)):
                    for k in range(len(d)):
                        if d[j] is None or d[k] is None:
                            continue
                        vec = assemble_facet_vector(
                            self.Vh[i], self.facet_batches.groups,
                            self.facet_kernel.element_third(
                                i, j, k, floc, facet_values(self.Vh[j], d[j]),
                                facet_values(self.Vh[k], d[k])),
                            tables=self.facet_batches.tables(self.Vh[i]))
                        res.axpy(w, vec)
        ess = self.bc0.ess if i in (STATE, ADJOINT) else None
        if ess is not None and len(ess):
            res.array[np.asarray(ess, dtype=np.int64)] = 0.0
        out.assign(res)
        return out

    # --------------------------------------------------------------- utilities
    def functional(self, x):
        """The integral of the residual density itself, ``R(u, m, p)``."""
        loc = self._locals(x) + self._aux_locals()
        total = assemble_scalar(self.comm, self.kernel.element_values(loc))
        if self.bdr_kernel is not None:
            total += assemble_scalar(self.comm,
                                     self.bdr_kernel.element_values(loc))
        if self.facet_kernel is not None:
            # A face shared with another rank is seen from both sides and both compute
            # the same value for it; half from each counts it once.
            vals = self.facet_kernel.element_values(self._facet_locals(x))
            local = sum(float(np.dot(np.asarray(v, dtype=float).reshape(-1),
                                     np.where(g.shared, 0.5, 1.0)))
                        for g, v in zip(self.facet_batches.groups, vals))
            total += self.comm.allreduce(local, op=MPI.SUM)
        return total


class _FiniteFlag:
    """Collects "was every chunk finite" as the chunks stream past."""

    def __init__(self):
        self.ok = True

    def wrap(self, thunk):
        def gen():
            from ..fem.kernel import all_finite

            for g, a, b, arr in thunk():
                arrs = list(arr.values()) if isinstance(arr, dict) else [arr]
                self.ok = self.ok and bool(all_finite(arrs))
                yield g, a, b, arr

        return gen

    def check(self, comm, what):
        if not comm.allreduce(int(self.ok), op=MPI.MIN):
            raise RuntimeError(
                "%s produced non-finite element values; the residual density "
                "overflowed at this point (a line search should reject it)" % what)


def require_finite_arrays(arrays, comm, what):
    """Raise if any element array has a non-finite entry, on every rank.

    Checking the *inputs* is not enough: a parameter of 800 is finite while
    ``exp(m)`` is not, so the overflow happens inside the density and only the
    element arrays show it.  Unchecked, the matrix reaches hypre, whose AMG setup
    either hangs or aborts the whole job (``Error during setup! Error code: 12``),
    neither of which a line search can recover from.  The cost is one allreduce
    per assembly.
    """
    # Reduce on whichever device the arrays live on: np.isfinite on a device array
    # would copy every element matrix to the host at every assembly.
    from ..fem.kernel import all_finite

    ok = all_finite(arrays)
    if not comm.allreduce(int(ok), op=MPI.MIN):
        raise RuntimeError(
            "%s produced non-finite element values; the residual density "
            "overflowed at this point (a line search should reject it)" % what
        )
    return arrays


def require_finite(v, what):
    """Raise if ``v`` has any non-finite entry, on every rank.

    A matrix assembled from non-finite values has ``inf`` or ``nan`` entries, on
    which hypre's AMG setup **hangs** rather than fails.  Checking first turns
    that into a ``RuntimeError``, which the line searches handle by backtracking;
    the check costs one allreduce.
    """
    ok = bool(np.isfinite(v.array).all()) if v.local_size else True
    if not v.comm.allreduce(int(ok), op=MPI.MIN):
        raise RuntimeError(
            "%s contains non-finite values; the step that produced it is too "
            "large for this problem (a line search should reject it)" % what
        )
    return v


def _same(a, b):
    """True when two vectors are bitwise equal (one global reduction)."""
    if a is None or b is None:
        return a is b
    if a.local_size != b.local_size:
        return False
    return a.copy().axpy(-1.0, b).norm("linf") == 0.0


def _krylov_class():
    from ..algorithms.linSolvers import KrylovSolver
    return KrylovSolver


class TransposeOf(mfem.TransposeOperator):
    """``A^T`` as an MFEM operator over ``A.MultTranspose``, no transpose formed.

    ``transposed_of`` is ``A`` itself: MFEM holds a raw pointer to it, so the
    wrapper keeps it alive, and :func:`_set_operator_once` reads it to hand over
    ``A``'s preconditioner instead of building one on the wrapper.
    """

    def __init__(self, A):
        super(TransposeOf, self).__init__(A)
        self.transposed_of = A


def _transpose_operator(A):
    return TransposeOf(A)


def _set_operator_once(solver, A, share_from=None):
    """Point a solver at ``A``, skipping the work if it is already there.

    ``set_operator`` rebuilds the preconditioner, which for BoomerAMG is the
    expensive part and is pure waste when the matrix has not changed.  With
    ``share_from`` a solver that already holds this very operator, its
    preconditioner is handed over instead of built a second time (the forward and
    the forward-incremental solver hold the same Jacobian).
    """
    if getattr(solver, "current_operator", None) is A:
        return solver
    pc = None
    base = getattr(A, "transposed_of", None)            # a TransposeOf wrapper of base
    if share_from is not None and share_from is not solver:
        cur = getattr(share_from, "current_operator", None)
        if cur is A or (base is not None and cur is base):
            pc = getattr(share_from, "_pc", None)
    if pc is None and base is not None and hasattr(solver, "_make_pc"):
        pc = solver._make_pc(base)                      # a hierarchy of A, not of the wrapper
    try:
        solver.set_operator(A, pc=pc) if pc is not None else solver.set_operator(A)
    except TypeError:                                    # a solver without the option
        solver.set_operator(A)
    solver.current_operator = A
    return solver

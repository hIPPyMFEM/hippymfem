# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Linear algebra and optimization algorithms."""

from .multivector import MatMvMult, MatMvTranspmult, MultiVector, MvDSmatMult
from .linSolvers import (
    KrylovSolver,
    KrylovSolver_ParameterList,
    auto_solver,
    LUSolver,
    LumpedMassSolver,
    TransposeSolver,
    make_solver,
)
from .cgsolverSteihaug import CGSolverSteihaug, CGSolverSteihaug_ParameterList
from .NewtonCG import (
    LS_ParameterList,
    ModelConvergenceError,
    ReducedSpaceNewtonCG,
    ReducedSpaceNewtonCG_ParameterList,
    TR_ParameterList,
)
from .steepestDescent import SteepestDescent, SteepestDescent_ParameterList
from .bfgs import (
    BFGS,
    BFGS_ParameterList,
    BFGS_operator,
    BFGSoperator_ParameterList,
    RescaledIdentity,
)
from .randomizedEigensolver import (
    check_g,
    check_std,
    doublePass,
    doublePassG,
    singlePass,
    singlePassG,
)
from .randomizedSVD import accuracyEnhancedSVD, check_SVD, singlePassSVD
from .lowRankOperator import LowRankOperator
from .traceEstimator import TraceEstimator
from .cgsampler import CGSampler, CGSampler_ParameterList
from .directSolvers import (
    PETScKrylovSolver,
    PETScLUSolver,
    ReplicatedLUSolver,
    gather_matrix,
    petsc_available,
    preload_petsc,
)

__all__ = [
    "MultiVector", "MatMvMult", "MatMvTranspmult", "MvDSmatMult",
    "KrylovSolver", "LUSolver", "auto_solver", "TransposeSolver", "LumpedMassSolver",
    "KrylovSolver_ParameterList", "make_solver",
    "CGSolverSteihaug", "CGSolverSteihaug_ParameterList",
    "ReducedSpaceNewtonCG", "ReducedSpaceNewtonCG_ParameterList",
    "LS_ParameterList", "TR_ParameterList", "ModelConvergenceError",
    "SteepestDescent", "SteepestDescent_ParameterList",
    "BFGS", "BFGS_ParameterList", "BFGS_operator", "BFGSoperator_ParameterList",
    "RescaledIdentity",
    "singlePass", "doublePass", "singlePassG", "doublePassG",
    "check_std", "check_g",
    "accuracyEnhancedSVD", "singlePassSVD", "check_SVD",
    "LowRankOperator", "TraceEstimator",
    "CGSampler", "CGSampler_ParameterList",
    "ReplicatedLUSolver", "PETScLUSolver", "PETScKrylovSolver",
    "gather_matrix", "petsc_available", "preload_petsc",
]

# snake_case spellings of the module-level functions (hippymfem.common.naming)
from .randomizedEigensolver import single_pass, double_pass, single_pass_g, double_pass_g  # noqa: E402
from .randomizedSVD import accuracy_enhanced_svd, single_pass_svd  # noqa: E402
__all__ += ["single_pass", "double_pass", "single_pass_g", "double_pass_g", "accuracy_enhanced_svd", "single_pass_svd"]

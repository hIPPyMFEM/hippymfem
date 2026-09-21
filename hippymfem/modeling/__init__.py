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
"""Modeling layer: forward problems, priors, misfits, the model, the Hessian."""

from .variables import ADJOINT, NVAR, PARAMETER, STATE, Variables
from .PDEProblem import PDEProblem
from .PDEVariationalProblem import PDEVariationalProblem
from .timeDependentVector import TimeDependentVector
from .TimeDependentPDEVariationalProblem import (
    ImplicitEulerTimeDependentPDEVariationalProblem,
    TimeDependentPDEVariationalProblem,
)
from .pointwiseObservation import (
    PointwiseObservation,
    assemblePointwiseLOSObservation,
    assemblePointwiseObservation,
    exportPointwiseObservation,
)
from .misfit import (
    ContinuousStateObservation,
    DiscreteStateObservation,
    Misfit,
    MisfitTD,
    MultDiscreteStateObservation,
    MultiStateMisfit,
    MultPointwiseStateObservation,
    PointwiseStateObservation,
)
from .prior import (
    BiLaplacianComputeCoefficients,
    BiLaplacianPrior,
    GaussianRealPrior,
    LaplacianPrior,
    MollifiedBiLaplacianPrior,
    QuadratureSqrtPrecision,
    SqrtPrecisionPDE_Prior,
    VectorBiLaplacianPrior,
)
from .posterior import (
    GaussianLRPosterior,
    LowRankHessian,
    LowRankPosteriorSampler,
)
from .model import Model
from .reducedHessian import FDHessian, ReducedHessian
from .modelVerify import best_slope, fd_slopes, modelVerify

__all__ = [
    "STATE", "PARAMETER", "ADJOINT", "NVAR", "Variables",
    "PDEProblem", "PDEVariationalProblem", "TimeDependentVector",
    "TimeDependentPDEVariationalProblem",
    "ImplicitEulerTimeDependentPDEVariationalProblem",
    "PointwiseObservation", "assemblePointwiseObservation",
    "assemblePointwiseLOSObservation", "exportPointwiseObservation",
    "Misfit", "DiscreteStateObservation", "PointwiseStateObservation",
    "MultDiscreteStateObservation", "MultPointwiseStateObservation",
    "ContinuousStateObservation", "MultiStateMisfit", "MisfitTD",
    "SqrtPrecisionPDE_Prior", "LaplacianPrior", "BiLaplacianPrior",
    "VectorBiLaplacianPrior", "MollifiedBiLaplacianPrior", "GaussianRealPrior",
    "BiLaplacianComputeCoefficients", "QuadratureSqrtPrecision",
    "Model", "ReducedHessian", "FDHessian",
    "GaussianLRPosterior", "LowRankHessian", "LowRankPosteriorSampler",
    "modelVerify", "fd_slopes", "best_slope",
]

# snake_case spellings of the module-level functions (hippymfem.common.naming)
from .modelVerify import model_verify  # noqa: E402
from .pointwiseObservation import assemble_pointwise_observation, assemble_pointwise_los_observation, export_pointwise_observation  # noqa: E402
__all__ += ["model_verify", "assemble_pointwise_observation", "assemble_pointwise_los_observation", "export_pointwise_observation"]

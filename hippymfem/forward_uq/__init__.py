# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Forward uncertainty propagation: quantities of interest and their moments."""

from .qoi import (
    LinearStateQoi,
    NullQoi,
    Qoi,
    QuadraticStateQoi,
    l2_norm_qoi,
    mass_functional,
    mean_state_qoi,
    weighted_mean_qoi,
)
from .parameter2QoiMap import (
    Parameter2QoiHessian,
    Parameter2QoiMap,
    parameter2QoiMapVerify,
    qoiVerify,
)
from .taylorApproximationQoi import TaylorApproximationQoi
from .varianceReductionMC import varianceReductionMC
from .variationalQoi import VariationalQoi

__all__ = [
    "Qoi", "NullQoi", "LinearStateQoi", "QuadraticStateQoi",
    "l2_norm_qoi", "mean_state_qoi", "weighted_mean_qoi", "mass_functional", "VariationalQoi",
    "Parameter2QoiMap", "Parameter2QoiHessian",
    "parameter2QoiMapVerify", "qoiVerify",
    "TaylorApproximationQoi", "varianceReductionMC",
]

# snake_case spellings of the module-level functions (hippymfem.common.naming)
from .varianceReductionMC import variance_reduction_mc  # noqa: E402
__all__ += ["variance_reduction_mc"]

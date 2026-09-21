# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Shared infrastructure: vectors, operators, linear algebra, randomness."""

from .keepalive import KeepAlive
from .mfemconfig import (configure_device, mfem_config, mfem_has,
                          mfem_version)
from .parvector import (Layout, ParVector, as_parvector, host_sync,
                        partition, to_numpy)
from .parameterList import ParameterList
from .random import Random, parRandom
from .operators import (
    DiagonalOperator,
    IdentityOperator,
    MFEMOperator,
    MatrixOperator,
    Operator,
    Operator2Solver,
    ProductOperator,
    ScaledOperator,
    Solver2Operator,
    SumOperator,
    TransposeOperator,
    init_vector_like,
    make_vector,
)
from .linalg import (
    MatAtB,
    MatMatMatMult,
    MatMatMult,
    MatPtAP,
    ParAdd,
    Transpose,
    amg_method,
    estimate_diagonal_inv2,
    set_diagonal_entries,
    get_diagonal,
    hypre_to_scipy,
    scipy_to_hypre,
    trace,
)

__all__ = [
    "KeepAlive", "ParVector", "Layout", "as_parvector", "partition", "to_numpy", "ParameterList",
    "Random", "parRandom", "Operator", "MatrixOperator", "TransposeOperator",
    "IdentityOperator", "ScaledOperator", "SumOperator", "ProductOperator",
    "DiagonalOperator", "Solver2Operator", "Operator2Solver", "MFEMOperator",
    "init_vector_like", "make_vector", "MatMatMult", "MatAtB", "MatPtAP", "MatMatMatMult",
    "Transpose", "ParAdd", "get_diagonal", "trace", "estimate_diagonal_inv2",
    "hypre_to_scipy", "scipy_to_hypre", "set_diagonal_entries",
    "amg_method",
    "mfem_config", "mfem_has", "mfem_version", "configure_device",
    "host_sync",
]

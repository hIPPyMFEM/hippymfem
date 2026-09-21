# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Finite element layer: spaces, batched geometry, AD kernels, assembly, BCs."""

from .spaces import FunctionSpace, as_space
from .elementbatch import (
    ElementGroup,
    MeshBatches,
    SpaceTables,
    default_quadrature_degree,
    get_batches,
    group_tables,
)
from .bcs import BCSet, DirichletBC, as_bcset
from .vectorfe import (
    HCurlField,
    HDivField,
    VectorSpaceTables,
    is_vector_space,
    vector_kind,
)
from .boundary import (
    BoundaryBatches,
    BoundaryGroup,
    BoundarySpaceTables,
    assemble_boundary_matrix,
    assemble_boundary_vector,
    boundary_marker,
    get_boundary_batches,
)
from .facets import (
    FacetBatches,
    FacetField,
    FacetSpaceTables,
    InteriorFacetGroup,
    assemble_facet_matrix,
    assemble_facet_vector,
    avg,
    avg_grad,
    clear_facet_cache,
    facet_values,
    get_facet_batches,
    jump,
    jump_grad,
)
from .integrators import (
    CachedMatrixIntegrator,
    CachedVectorIntegrator,
    ElementLookup,
)
from .io import (
    ParaViewWriter,
    load_vector,
    save_vector,
    write_glvis,
    write_paraview,
    write_point_csv,
)
from .assemble import (
    own,
    assemble_matrix,
    assemble_native_matrix,
    assemble_scalar,
    assemble_vector,
    assembly_backend,
    set_assembly_backend,
)
from .csrassemble import (
    ScatterPattern,
    VectorPattern,
    get_vector_pattern,
    assemble_matrix_csr,
    assemble_vector_csr,
    clear_pattern_cache,
    get_pattern,
)

__all__ = [
    "FunctionSpace", "as_space", "ElementGroup", "MeshBatches", "SpaceTables",
    "default_quadrature_degree", "get_batches", "DirichletBC", "BCSet",
    "as_bcset", "CachedMatrixIntegrator", "CachedVectorIntegrator",
    "ElementLookup", "assemble_matrix", "assemble_vector", "assemble_scalar",
    "assemble_native_matrix", "own", "group_tables",
    "assembly_backend", "set_assembly_backend", "ScatterPattern", "get_pattern",
    "assemble_matrix_csr", "assemble_vector_csr", "clear_pattern_cache",
    "VectorPattern", "get_vector_pattern",
    "BoundaryBatches", "BoundaryGroup", "BoundarySpaceTables",
    "assemble_boundary_matrix", "assemble_boundary_vector", "boundary_marker",
    "FacetBatches", "FacetField", "FacetSpaceTables", "InteriorFacetGroup",
    "assemble_facet_matrix", "assemble_facet_vector", "avg", "avg_grad",
    "clear_facet_cache", "facet_values", "get_facet_batches", "jump", "jump_grad",
    "get_boundary_batches",
    "HCurlField", "HDivField", "VectorSpaceTables", "is_vector_space",
    "vector_kind",
    "write_paraview", "ParaViewWriter", "write_glvis", "save_vector",
    "load_vector", "write_point_csv",
]


def _lazy_kernel():
    """Import the JAX-dependent kernel layer only when it is first needed."""
    from . import kernel

    return kernel


def __getattr__(name):
    """The JAX-dependent modules, imported on first use (PEP 562).

    ``importlib.import_module`` and not ``from . import jaxops``, which asks this
    package for the attribute first and re-enters here without end.
    """
    import importlib

    if name in ("QuadratureKernel", "BoundaryKernel", "GroupKernel", "Field",
                "set_device"):
        return getattr(importlib.import_module("hippymfem.fem.kernel"), name)
    if name in ("jaxops", "kernel"):
        return importlib.import_module("hippymfem.fem." + name)
    raise AttributeError(name)

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""petsc4py loading, which has to happen before PyMFEM is imported.

petsc4py and PyMFEM link against MPI libraries that, in some installations, must
be resolved in that order: importing petsc4py *after* ``mfem`` fails with

    ImportError: libmpi_mpifh.so.40: undefined symbol: mpi_conversion_fn_null_

while importing it first works.  This module therefore imports no MFEM itself and
is loaded from ``hippymfem/__init__.py`` ahead of everything else.

It does nothing unless ``HIPPYMFEM_PETSC`` is set to a truthy value or petsc4py is
already imported.  A failure is recorded rather than raised;
:func:`petsc_available` reports it.
"""

import os
import sys

#: ``None`` until an attempt is made, then ``(PETSc_module_or_None, reason)``.
_STATE = None


def preload_petsc():
    """Import petsc4py and return the ``PETSc`` module, or ``None`` on failure."""
    global _STATE
    if _STATE is not None:
        return _STATE[0]
    try:
        import petsc4py

        try:
            petsc4py.init([sys.argv[0] if sys.argv else "hippymfem"])
        except Exception:
            pass                      # already initialized, which is fine
        from petsc4py import PETSc

        _STATE = (PETSc, None)
    except Exception as exc:
        reason = str(exc).split("\n")[0]
        if "undefined symbol" in reason or "mpi_conversion_fn_null_" in reason:
            reason += (" -- petsc4py must be imported before PyMFEM: set "
                       "HIPPYMFEM_PETSC=1 before importing hippymfem, or import "
                       "petsc4py at the top of your script")
        _STATE = (None, reason)
    return _STATE[0]


def petsc_available():
    """``(True, "")`` when the PETSc route can be used, else ``(False, reason)``."""
    preload_petsc()
    mod, reason = _STATE
    return (mod is not None), (reason or "")


def _wanted():
    if "petsc4py" in sys.modules:
        return True
    return os.environ.get("HIPPYMFEM_PETSC", "").lower() not in (
        "", "0", "no", "false", "off")


if _wanted():
    preload_petsc()

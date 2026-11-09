# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""One rank's uncaught exception ends every rank.

A run on several ranks in which one rank raises leaves the others waiting in the next
collective until the job's time limit: on Frontier that held whole nodes for ten minutes
and more, three times in one afternoon.  ``import hippymfem`` on more than one rank
therefore wraps ``sys.excepthook``: the traceback is printed as before, with the rank, and
``MPI.COMM_WORLD.Abort`` follows.  ``SystemExit`` does not reach the hook, so a script or
a test suite that exits with a status of its own is not affected.  Set
``HIPPYMFEM_ABORT_ON_ERROR=0`` before the import to keep the other ranks waiting.
"""

import os
import sys

#: Whether :func:`install` wraps the hook (read at import: ``HIPPYMFEM_ABORT_ON_ERROR``).
ABORT_ON_ERROR = os.environ.get("HIPPYMFEM_ABORT_ON_ERROR", "1").strip().lower() in (
    "1", "yes", "true", "on")
_INSTALLED = [None]


def installed():
    """Whether the hook is in place."""
    return _INSTALLED[0] is not None


def install(comm=None):
    """Wrap ``sys.excepthook`` so that an uncaught exception aborts every rank of ``comm``
    (default ``MPI.COMM_WORLD``), where MPI runs on more than one rank and
    :data:`ABORT_ON_ERROR` is set.  Returns whether it did (once per process)."""
    if _INSTALLED[0] is not None or not ABORT_ON_ERROR:
        return False
    try:
        from mpi4py import MPI
    except ImportError:
        return False
    if not MPI.Is_initialized() or MPI.Is_finalized():
        return False
    world = comm if comm is not None else MPI.COMM_WORLD
    if world.Get_size() < 2:
        return False
    previous = sys.excepthook

    def hook(kind, value, tb):
        try:
            sys.stderr.write("[hippymfem] rank %d of %d: uncaught %s; aborting every rank "
                             "(HIPPYMFEM_ABORT_ON_ERROR=0 keeps them waiting)\n"
                             % (world.Get_rank(), world.Get_size(), kind.__name__))
            previous(kind, value, tb)
            sys.stderr.flush()
            sys.stdout.flush()
        finally:
            world.Abort(1)

    sys.excepthook = hook
    _INSTALLED[0] = (hook, previous)
    return True


def uninstall():
    """Put the previous hook back (for tests)."""
    if _INSTALLED[0] is not None:
        hook, previous = _INSTALLED[0]
        if sys.excepthook is hook:
            sys.excepthook = previous
        _INSTALLED[0] = None

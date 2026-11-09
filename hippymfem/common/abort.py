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

The hook also receives errors that are handled where they are shown, and those it only
reports (:func:`handled_elsewhere`).  A console of the standard library (pdb's
``interact``, the REPL of Python 3.13) shows an error typed into it through the hook and
goes on.  ``weakref.finalize`` reports through it an exception of a finalizer that it runs
at exit, when the program's work is done.  An interactive session (``python -i``, the
REPL) is the user's to end.  After ``MPI.Finalize`` no rank can be reached.  In each of
these cases the traceback is printed as before, and nothing is aborted.
"""

import sys

from ..config import env_flag

#: Whether :func:`install` wraps the hook (read at import: ``HIPPYMFEM_ABORT_ON_ERROR``).
ABORT_ON_ERROR = env_flag("HIPPYMFEM_ABORT_ON_ERROR", True)
_INSTALLED = [None]


def installed():
    """Whether the hook is in place."""
    return _INSTALLED[0] is not None


def _reporting_codes():
    """The code objects of the functions that call ``sys.excepthook`` for an error they
    handle or only report: the traceback methods of ``code.InteractiveInterpreter`` (named
    differently from one Python to the next) and ``weakref.finalize._exitfunc``."""
    import code
    import weakref

    codes = set()
    for name, f in vars(code.InteractiveInterpreter).items():
        if ("showtraceback" in name or "showsyntaxerror" in name) and hasattr(f, "__code__"):
            codes.add(f.__code__)
    exitfunc = getattr(weakref.finalize, "_exitfunc", None)
    exitfunc = getattr(exitfunc, "__func__", exitfunc)
    if exitfunc is not None and hasattr(exitfunc, "__code__"):
        codes.add(exitfunc.__code__)
    return codes


def handled_elsewhere(caller):
    """Whether the hook, called from the frame ``caller``, reports an error that is
    handled there or that it is not to end the run for: shown by a console that goes on,
    raised by a finalizer that ``weakref.finalize`` runs at exit, or in an interactive
    session."""
    if hasattr(sys, "ps1") or sys.flags.interactive:
        return True
    return caller is not None and getattr(caller, "f_code", None) in _reporting_codes()


def _abort(world):
    """End every rank (replaced by the tests, which must not)."""
    world.Abort(1)


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
            caller = sys._getframe(1)
        except (AttributeError, ValueError):
            caller = None
        try:
            finalized = MPI.Is_finalized()
        except Exception:                                    # noqa: BLE001
            finalized = True
        if finalized or handled_elsewhere(caller):
            previous(kind, value, tb)
            return
        try:
            sys.stderr.write("[hippymfem] rank %d of %d: uncaught %s; aborting every rank "
                             "(HIPPYMFEM_ABORT_ON_ERROR=0 keeps them waiting)\n"
                             % (world.Get_rank(), world.Get_size(), kind.__name__))
            previous(kind, value, tb)
            sys.stderr.flush()
            sys.stdout.flush()
        finally:
            _abort(world)

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

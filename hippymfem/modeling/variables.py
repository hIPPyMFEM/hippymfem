# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Indices of the three variables of a PDE-constrained inverse problem."""

#: the state (forward solution)
STATE = 0
#: the parameter (inversion variable)
PARAMETER = 1
#: the adjoint variable
ADJOINT = 2
#: number of variables
NVAR = 3

_NAMES = {STATE: "STATE", PARAMETER: "PARAMETER", ADJOINT: "ADJOINT"}


def name(i):
    """Human-readable name of a variable index."""
    return _NAMES.get(i, "VAR%d" % i)


class Variables(list):
    """The ``[state, parameter, adjoint]`` triple of a PDE-constrained problem.

    A list, so everything written for hIPPYlib's convention keeps working:
    ``x[STATE]``, ``x[PARAMETER] = m``, ``len(x)``, extra trailing entries.  The
    three slots are also attributes, ``x.state``, ``x.parameter``, ``x.adjoint``,
    readable and assignable, so application code need not import the indices.
    ``generate_vector()`` on the problems, the model and the QoI map returns one.
    """

    __slots__ = ()

    def __init__(self, state=None, parameter=None, adjoint=None, *extra):
        super(Variables, self).__init__((state, parameter, adjoint) + tuple(extra))

    def _get(self, i):
        return self[i]

    def _set(self, i, v):
        self[i] = v

    state = property(lambda self: self[STATE], lambda self, v: self._set(STATE, v),
                     doc="the state, ``x[STATE]``")
    parameter = property(lambda self: self[PARAMETER],
                         lambda self, v: self._set(PARAMETER, v),
                         doc="the parameter, ``x[PARAMETER]``")
    adjoint = property(lambda self: self[ADJOINT], lambda self, v: self._set(ADJOINT, v),
                       doc="the adjoint, ``x[ADJOINT]``")

    def __repr__(self):
        return "Variables(%s)" % ", ".join(
            "%s=%r" % (name(i) .lower(), v) if i < NVAR else repr(v)
            for i, v in enumerate(self))


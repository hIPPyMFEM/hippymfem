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
"""A dictionary of ``(value, description)`` pairs that rejects unknown keys.

Ported from hIPPYlib so that parameter dictionaries in driver scripts are
interchangeable between the two libraries.
"""

from .naming import sync_spellings



class ParameterList(object):
    """Parameter store that raises on an unknown key instead of silently adding it."""

    def __init__(self, data):
        #: ``{key: [value, description]}``
        self.data = data

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key][0]
        raise ValueError(key)

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key][0] = value
        else:
            raise ValueError(key)

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def update(self, other):
        for k, v in dict(other).items():
            self[k] = v
        return self

    def showMe(self, indent=""):
        """Print a tree view of the parameter list."""
        for k in sorted(self.data.keys()):
            print(indent, "---")
            if isinstance(self.data[k][0], ParameterList):
                print(indent, k, "(ParameterList):", self.data[k][1])
                self.data[k][0].showMe(indent + "    ")
            else:
                print(indent, k, "({0}):".format(self.data[k][0]), self.data[k][1])
        print(indent, "---")


sync_spellings(ParameterList)

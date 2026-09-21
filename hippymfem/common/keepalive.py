# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Reference retention for PyMFEM objects.

PyMFEM hands raw C++ pointers to MFEM.  If the Python wrapper of, say, a
``FiniteElementCollection`` is garbage collected while a
``ParFiniteElementSpace`` still points at it, the next call into MFEM
segfaults.

Every class in hIPPyMFEM that hands an object to MFEM keeps a strong
reference to it through :class:`KeepAlive`.
"""


class KeepAlive:
    """Mixin providing a list of objects that must outlive ``self``."""

    @property
    def _keep(self):
        try:
            return self.__keep
        except AttributeError:
            self.__keep = []
            return self.__keep

    def keep(self, *objs):
        """Retain ``objs`` for the lifetime of ``self``; returns the first one."""
        k = self._keep
        for o in objs:
            if o is not None:
                k.append(o)
        return objs[0] if objs else None

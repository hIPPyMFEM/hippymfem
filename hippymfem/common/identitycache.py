# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""A cache keyed on the identity of live objects that cannot answer for a dead one."""

import weakref


class IdentityCache:
    """Values keyed on the identity of objects, plus hashable extras.

    The assembler's pattern caches key on ``id()`` (the tables of a space, the pattern
    of a space pair on a set of element groups, the transpose of a prolongation).  An
    address is reused once its object dies, so a stale entry could answer for the
    wrong object; holding every keyed object alive would prevent that but keep them
    all for the life of the process.  Instead an entry is dropped as soon as any
    object it is keyed on is finalized (``weakref.finalize``), so objects are freed
    when their owners release them.

    ``objs`` are the identity-keyed objects (``None`` is allowed and keyed as itself);
    ``extra`` is any hashable value that is part of the key by value, such as a
    quadrature degree or a device.
    """

    def __init__(self):
        self._data = {}

    @staticmethod
    def _key(objs, extra):
        return (tuple(id(o) for o in objs), extra)

    def get(self, objs, extra=(), default=None):
        """The value stored for ``objs`` (and ``extra``), or ``default``."""
        return self._data.get(self._key(objs, extra), default)

    def put(self, objs, value, extra=()):
        """Store ``value`` for ``objs`` (and ``extra``); returns ``value``."""
        key = self._key(objs, extra)
        for o in objs:
            if o is None:
                continue
            try:
                f = weakref.finalize(o, self._data.pop, key, None)
            except TypeError:
                raise TypeError("an IdentityCache cannot key on a %s: it does not "
                                "support weak references" % type(o).__name__)
            f.atexit = False
        self._data[key] = value
        return value

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)

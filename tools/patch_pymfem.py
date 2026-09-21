#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Apply the upstream PyMFEM build workarounds, for any build (CUDA or not).

Two of the four problems ``tools/build_pymfem_cuda.sh`` documents are not specific
to CUDA and bite a plain parallel build on a current Python just as hard:

1. ``pyproject.toml`` uses the PEP 639 ``license = "..."`` string form, which older
   setuptools rejects with a schema error rather than a readable message.
2. The SWIG wrappers need ``-fpermissive`` (SWIG 4.2 generates ``SWIG_init``
   returning ``PyObject*`` while numpy's ``import_array1`` returns ``int``), and
   their ``setup.py`` imports ``distutils``, which Python 3.12 removed.  Importing
   setuptools instead pulls in the standard library's ``socket``, which does
   ``import array`` and finds PyMFEM's own ``array.py``; the script's directory has
   to leave ``sys.path`` first, and ``''`` is on that path as well.

It also turns off MFEM's examples and miniapps, which the wrapper never uses and
which are most of the build time; pass ``--keep-examples`` to leave them on.

Each patch is a no-op when upstream has moved on, so this is safe to re-run.

Usage: patch_pymfem.py <path to the PyMFEM checkout> [--keep-examples]
"""

import io
import os
import re
import sys


def patch_pyproject(root):
    p = os.path.join(root, "pyproject.toml")
    if not os.path.exists(p):
        return "no pyproject.toml"
    s = io.open(p, encoding="utf-8").read()
    out = re.sub(r'^license\s*=\s*"([^"]+)"\s*$', r'license = {text = "\1"}',
                 s, flags=re.M)
    out = re.sub(r'^license-files\s*=.*$\n', '', out, flags=re.M)
    if out == s:
        return "already fine"
    io.open(p, "w", encoding="utf-8").write(out)
    return "patched"


def patch_swig_setup(path):
    s = io.open(path, encoding="utf-8").read()
    before = s
    old = "    extra_compile_args = [cxxstdflag, '-DSWIG_TYPE_TABLE=PyMFEM']"
    new = ("    # SWIG generates SWIG_init returning PyObject* while numpy's\n"
           "    # import_array1 returns int; g++ names -fpermissive as the fix,\n"
           "    # and the value is only used on the import-failure path.\n"
           "    extra_compile_args = [cxxstdflag, '-DSWIG_TYPE_TABLE=PyMFEM',\n"
           "                          '-fpermissive']")
    if old in s:
        s = s.replace(old, new)
    old2 = "sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))"
    new2 = ("# This directory holds array.py; importing setuptools pulls in the\n"
            "# standard library's socket, which does `import array` and would\n"
            "# find it.  Removing sys.argv[0]'s directory is not enough: '' is\n"
            "# on the path as well.\n"
            "_here = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))\n"
            "sys.path[:] = [_p for _p in sys.path\n"
            "               if os.path.abspath(_p if _p else os.getcwd()) != _here]")
    if old2 in s:
        s = s.replace(old2, new2)
    s = s.replace("from distutils.core import Extension, setup",
                  "from setuptools import Extension, setup   # distutils is gone in 3.12")
    if s == before:
        return "already fine"
    io.open(path, "w", encoding="utf-8").write(s)
    return "patched"


def patch_examples_off(root):
    """Stop building MFEM's examples and miniapps.

    They are not used by the wrapper, they are most of the MFEM build time, and
    under CUDA they fail to link outright (the parallel examples get no MPI on
    the CUDA link line).
    """
    p = os.path.join(root, "_build_system", "build_mfem.py")
    if not os.path.exists(p):
        return "no build_mfem.py"
    s = io.open(p, encoding="utf-8").read()
    old = """                  'DMFEM_ENABLE_EXAMPLES': '1',
                  'DMFEM_ENABLE_MINIAPPS': '1',"""
    new = """                  # not used by the wrapper, most of the build time, and
                  # they fail to link under CUDA
                  'DMFEM_ENABLE_EXAMPLES': '0',
                  'DMFEM_ENABLE_MINIAPPS': '0',"""
    if old not in s:
        return "already fine"
    io.open(p, "w", encoding="utf-8").write(s.replace(old, new))
    return "patched"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    print("  pyproject.toml: %s" % patch_pyproject(root))
    if "--keep-examples" not in sys.argv:
        print("  examples/miniapps: %s" % patch_examples_off(root))
    for sub in ("_par", "_ser"):
        p = os.path.join(root, "mfem", sub, "setup.py")
        if os.path.exists(p):
            print("  mfem/%s/setup.py: %s" % (sub, patch_swig_setup(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

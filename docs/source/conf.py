"""Sphinx configuration for the hIPPyMFEM documentation.

The API reference is generated from the docstrings, and the tutorials are the
notebooks in ``tutorial/``, rendered with the outputs they were last executed with,
so neither can drift from the code.  The build does not need MFEM: when PyMFEM, JAX
or mpi4py cannot be imported (as on Read the Docs), autodoc mocks them.

    pip install -r docs/requirements.txt
    sphinx-build -b html docs/source docs/build/html
"""

import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

project = "hIPPyMFEM"
copyright = "2026, Peng Chen and contributors"
author = "Peng Chen"

_version = {}
with open(os.path.join(ROOT, "hippymfem", "version.py")) as f:
    exec(f.read(), _version)
release = _version["__version__"]
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
]


def _importable(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


#: Mock only what is missing, so a machine with the full environment documents the
#: real objects (signature defaults included) and Read the Docs still builds.
autodoc_mock_imports = [name for name, probe in (("mfem", "mfem.par"), ("jax", "jax"),
                                                 ("mpi4py", "mpi4py.MPI"),
                                                 ("petsc4py", "petsc4py"))
                        if not _importable(probe)]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

# The tutorials are copied in from tutorial/ at every build (docs/source/tutorials is
# generated and ignored by git), and shown with their stored outputs: executing them
# needs PyMFEM, and CI executes them separately.
_TUTORIALS = os.path.join(HERE, "tutorials")
# rebuilt from scratch: a renamed notebook would otherwise leave its old copy behind, and
# Sphinx would warn about a page that is in no toctree (or render it under its old name)
shutil.rmtree(_TUTORIALS, ignore_errors=True)
os.makedirs(_TUTORIALS, exist_ok=True)
for _nb in sorted(glob.glob(os.path.join(ROOT, "tutorial", "*.ipynb"))):
    shutil.copy2(_nb, _TUTORIALS)
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["**/.ipynb_checkpoints"]
html_theme = "sphinx_rtd_theme"
# "Edit on GitHub" links on every page
html_context = {
    "display_github": True,
    "github_user": "hIPPyMFEM",
    "github_repo": "hippymfem",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
html_static_path = ["_static"]
html_title = "hIPPyMFEM %s" % release
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
nitpicky = False

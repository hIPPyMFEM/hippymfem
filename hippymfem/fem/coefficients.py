# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Coefficients built from mesh attributes."""

import numpy as np

import mfem.par as mfem


def attribute_indicator(mesh, attributes):
    """A piecewise-constant coefficient: 1 on the elements whose attribute is in
    ``attributes``, 0 elsewhere.

    Returns ``(coefficient, keep)``.  MFEM's coefficient holds a raw pointer to the
    marker vector, so the caller keeps everything in ``keep`` alive with it.
    """
    nattr = int(mesh.attributes.Max()) if mesh.attributes.Size() else 0
    marker = mfem.Vector(nattr)
    marker.Assign(0.0)
    for a in np.atleast_1d(attributes):
        marker[int(a) - 1] = 1.0
    return mfem.PWConstCoefficient(marker), [marker]

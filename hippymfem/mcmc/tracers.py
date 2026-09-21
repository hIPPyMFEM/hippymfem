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
"""Recording what a chain visits."""

import numpy as np


class NullTracer(object):
    """Records nothing."""

    def __init__(self):
        self.i = 0

    def append(self, current, q):
        self.i += 1


class QoiTracer(object):
    """Records the quantity of interest at every sample."""

    def __init__(self, n):
        self.data = np.zeros(int(n))
        self.i = 0

    def append(self, current, q):
        if self.i < self.data.size:
            self.data[self.i] = q
        self.i += 1

    def trim(self):
        return self.data[: self.i]


class FullTracer(object):
    """Records the quantity of interest and, optionally, the parameter fields.

    Writing every field of a long chain is expensive, so ``every`` subsamples.
    """

    def __init__(self, n, space=None, basename=None, every=1):
        self.data = np.zeros(int(n))
        self.i = 0
        self.every = max(int(every), 1)
        self.space = space
        self.basename = basename
        self.writer = None
        if space is not None and basename is not None:
            from ..fem.io import ParaViewWriter

            self.writer = ParaViewWriter(basename, space.mesh, {"m": space})

    def append(self, current, q):
        if self.i < self.data.size:
            self.data[self.i] = q
        if self.writer is not None and self.i % self.every == 0:
            self.writer.save({"m": current.m}, time=float(self.i), cycle=self.i)
        self.i += 1

    def trim(self):
        return self.data[: self.i]

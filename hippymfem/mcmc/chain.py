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
"""Running an MCMC chain."""

from ..common.parameterList import ParameterList
from ..modeling.timeDependentVector import TimeDependentVector
from ..modeling.variables import ADJOINT, PARAMETER, STATE


class NullQoi(object):
    """A quantity of interest that is always zero."""

    def eval(self, x):
        return 0.0


class SampleStruct:
    """One chain state: the parameter, the state, and whatever the kernel needs."""

    def __init__(self, kernel):
        self.derivative_info = kernel.derivativeInfo()
        self.u = kernel.model.generate_vector(STATE)
        self.m = kernel.model.generate_vector(PARAMETER)
        self.cost = 0.0
        self.weight = 1.0
        if self.derivative_info >= 1:
            self.p = kernel.model.generate_vector(ADJOINT)
            self.g = kernel.model.generate_vector(PARAMETER)
            self.Cg = kernel.model.generate_vector(PARAMETER)
        else:
            self.p = None
            self.g = None
            self.Cg = None

    def assign(self, other):
        if self.derivative_info != other.derivative_info:
            raise ValueError("sample structures carry different derivative info")
        self.cost = other.cost
        self.weight = other.weight
        self.m.assign(other.m)
        _assign(self.u, other.u)
        if self.derivative_info >= 1:
            _assign(self.p, other.p)
            self.g.assign(other.g)
            self.Cg.assign(other.Cg)
        return self


def _assign(a, b):
    """Assign for either a ParVector or a TimeDependentVector."""
    if a is None or b is None:
        return a
    if isinstance(b, TimeDependentVector):
        return a.copy(b)
    return a.assign(b)


class MCMC(object):
    """Drive a transition kernel, with burn-in and a tracer.

    Parameters are ``number_of_samples``, ``burn_in``, ``print_progress`` (how
    many progress lines to print) and ``print_level``.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.parameters = ParameterList({
            "number_of_samples": [2000, "samples kept after burn-in"],
            "burn_in": [1000, "samples discarded first"],
            "print_progress": [20, "progress lines per run"],
            "print_level": [1, "0 silent"],
        })
        self.sum_q = 0.0
        self.sum_q2 = 0.0

    def _log(self, msg):
        comm = getattr(self.kernel.model.prior, "comm", None)
        if self.parameters["print_level"] > 0 and (comm is None or comm.rank == 0):
            print(msg, flush=True)

    def run(self, m0, qoi=None, tracer=None):
        """Run the chain from ``m0``; returns the number of accepted proposals."""
        if qoi is None:
            qoi = NullQoi()
        if tracer is None:
            from .tracers import NullTracer

            tracer = NullTracer()
        nsamp = int(self.parameters["number_of_samples"])
        burn_in = int(self.parameters["burn_in"])

        current = SampleStruct(self.kernel)
        proposed = SampleStruct(self.kernel)
        current.m.assign(m0)
        self.kernel.init_sample(current)

        self._log("Burning %d samples (%s kernel)" % (burn_in, self.kernel.name()))
        self._phase(current, proposed, burn_in, None, None)

        self._log("Generating %d samples" % nsamp)
        self.sum_q = 0.0
        self.sum_q2 = 0.0
        return self._phase(current, proposed, nsamp, qoi, tracer)

    def _phase(self, current, proposed, nsteps, qoi, tracer):
        naccept = 0
        nprint = max(int(self.parameters["print_progress"]), 1)
        n_check = max(nsteps // nprint, 1)
        for k in range(nsteps):
            naccept += self.kernel.sample(current, proposed)
            if qoi is not None:
                q = qoi.eval([current.u, current.m])
                self.sum_q += q
                self.sum_q2 += q * q
                tracer.append(current, q)
            if (k + 1) % n_check == 0:
                self._log("  %5.1f%% complete, acceptance ratio %5.1f%%"
                          % (100.0 * (k + 1) / nsteps, 100.0 * naccept / (k + 1)))
        return naccept

    def consume_random(self):
        """Advance the random stream as a full run would, without running it."""
        total = (int(self.parameters["number_of_samples"])
                 + int(self.parameters["burn_in"]))
        for _ in range(total):
            self.kernel.consume_random()

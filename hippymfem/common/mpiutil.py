# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""MPI helpers shared by modules that must not import one another."""


def local_rank(comm=None):
    """This process's index among the ranks sharing its node.

    A collective (it splits the communicator), so it must be called by every rank
    of ``comm`` and never behind a rank-local condition.  Returns 0 when MPI is not
    available.  The import-time device pinning in :mod:`hippymfem._jaxconfig`
    cannot use it, MPI not being up yet, and reads the launcher's variables instead.
    """
    try:
        from mpi4py import MPI

        comm = comm if comm is not None else MPI.COMM_WORLD
        node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
        r = node.rank
        node.Free()
        return r
    except Exception:
        return 0

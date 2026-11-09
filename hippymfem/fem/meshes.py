# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Meshes that every rank builds its own part of.

``ParMesh(comm, Mesh.MakeCartesian3D(n, n, n))`` builds and partitions the whole mesh on
every rank: 635 bytes an element while METIS partitions it, which is 10 GB and 42 s a rank
at 256^3 and would be 79 GB at 512^3, where the eight ranks of a node have 512 GB between
them.  :func:`box_mesh` instead builds and partitions a coarse mesh on every rank and cuts
the elements of its part (``ParMesh.MakeRefined``): 0.3 GB and 0.7 s a rank at 256^3 on
32 ranks.  It can keep the meshes between the coarse one and the last, which the
p-multigrid's geometric levels and the prior's cycles over the meshes use
(:func:`hippymfem.fem.ptransfer.set_coarser_meshes`, the solver parameter ``pmg_h``).
"""

import numpy as np
import mfem.par as mfem

#: Coarse elements a rank that :func:`coarse_size` keeps, so that the partition has parts to
#: shape and to balance, and the largest number of cuts of a coarse element's side at once
#: (cutting builds one finite element collection of that order: 0.06 s at 8, 24 s at 16).
PER_RANK = 512
MAX_FACTOR = 8


def coarse_size(n, ranks):
    """Elements per side of the coarse mesh that :func:`box_mesh` builds on every rank: the
    smallest ``n / f``, ``f`` a divisor of ``n`` up to :data:`MAX_FACTOR`, that leaves
    :data:`PER_RANK` elements a rank (``n`` itself, the whole mesh, when no divisor does)."""
    for f in range(MAX_FACTOR, 1, -1):
        if n % f == 0 and (n // f) ** 3 >= PER_RANK * ranks:
            return n // f
    return n


def rank_grid(ranks, n0):
    """The ranks as a grid of equal boxes of the ``n0^3`` coarse mesh: the most cubic grid of
    ``ranks`` boxes, when its sides divide ``n0`` and differ by no more than a factor two;
    ``None`` when the ranks make no such grid."""
    grid = min(((a, b, ranks // (a * b)) for a in range(1, ranks + 1) if ranks % a == 0
                for b in range(a, ranks // a + 1) if (ranks // a) % b == 0 and b <= ranks // (a * b)),
               key=lambda g: g[2] / g[0])
    return grid if grid[2] <= 2 * grid[0] and all(n0 % g == 0 for g in grid) else None


def box_mesh(n, comm, coarse=None, levels=0):
    """The ``n^3`` hexahedral mesh of the unit cube, each rank building its own part.

    Every rank builds and partitions a mesh of ``coarse^3`` elements and cuts the elements
    of its part into ``(n / coarse)^3``.  The vertices are the lattice of
    ``Mesh.MakeCartesian3D(n, n, n)``, to the last bit where every cut is a power of two
    and within a few units in the last place where one is not (3e-16 with a cut in three
    or five), and the boundary attributes the same faces (1 the bottom, 6 the top).  The parts are unions of coarse elements; METIS leaves
    the largest of them 3 % above the mean, and the slowest card sets the pace, so the
    coarse mesh is cut into equal boxes where the ranks make a grid (:func:`rank_grid`:
    4 x 8 x 8 on 256 ranks) and by METIS where they do not.

    ``coarse`` defaults to :func:`coarse_size`; ``coarse = n`` is the whole mesh on every
    rank, partitioned by METIS.

    ``levels``: cut the elements in two a side that many times at the end instead of all
    at once, and keep the meshes in between for the library (the meshes of a p-multigrid's
    geometric levels).  The same vertices, elements and parts in another order; ``n /
    coarse`` must be a multiple of ``2 ** levels``.  Or the cuts themselves, from the first
    to the last: ``(5, 3, 2, 2, 2)`` cuts a coarse element in five a side, then in three,
    then three times in two, and keeps the five meshes before each cut; ``coarse`` then
    defaults to ``n`` over their product.

    Returns the mesh, ``coarse`` and the grid of the ranks (``None`` for METIS).
    """
    cuts = None
    if not isinstance(levels, (int, np.integer)):
        cuts = [int(f) for f in levels]
        if any(f < 2 or f > MAX_FACTOR for f in cuts):
            raise ValueError("the cuts of a mesh are 2 to %d a side each: not %s" % (MAX_FACTOR, cuts))
        levels = len(cuts)
    product = int(np.prod(cuts)) if cuts else 2 ** max(int(levels or 0), 0)
    if cuts and coarse is None:
        if n % product:
            raise ValueError("the cuts %s make %d of a coarse element's side: not a divisor of %d"
                             % (cuts, product, n))
        coarse = n // product
    if coarse is None:
        n0 = coarse_size(n, comm.size)
        # with levels, a default whose side the halvings do not divide (or leave more than
        # MAX_FACTOR of) gives way to the coarse mesh of the mesh before them
        if ((n // n0) % product or (n // n0) // product > MAX_FACTOR) and n % product == 0:
            n0 = coarse_size(n // product, comm.size)
    else:
        n0 = int(coarse)
    if n0 < 1 or n % n0:
        raise ValueError("the coarse mesh of a %d^3 mesh has a divisor of n elements a side: "
                         "not %d" % (n, n0))
    serial = mfem.Mesh.MakeCartesian3D(n0, n0, n0, mfem.Element.HEXAHEDRON)
    grid = rank_grid(comm.size, n0) if n0 < n else None
    if grid is None:
        pmesh = mfem.ParMesh(comm, serial)
    else:
        nxyz = mfem.intArray(list(grid))
        pmesh = mfem.ParMesh(comm, serial, serial.CartesianPartitioning(nxyz.GetData()))
    levels = int(levels or 0)
    if cuts is None:
        cuts = [2] * max(levels, 0)
        product = 2 ** max(levels, 0)
    if (n // n0) % product or (n // n0) // product > MAX_FACTOR:
        raise ValueError("a %d^3 mesh is %d times its %d^3 coarse mesh a side, which must be %d "
                         "(the cuts %s) times 1 to %d" % (n, n // n0, n0, product, cuts, MAX_FACTOR))
    first = (n // n0) // product
    if first > 1:
        pmesh = mfem.ParMesh.MakeRefined(pmesh, first, mfem.BasisType.ClosedUniform)
    coarser = []
    for f in cuts:
        coarser.insert(0, pmesh)
        pmesh = mfem.ParMesh.MakeRefined(pmesh, f, mfem.BasisType.ClosedUniform)
    if coarser:
        from .ptransfer import set_coarser_meshes

        set_coarser_meshes(pmesh, coarser)
    return pmesh, n0, grid

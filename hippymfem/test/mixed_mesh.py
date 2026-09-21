# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Generate a 2D mesh containing both triangles and quadrilaterals.

MFEM has no built-in mixed-geometry generator, and the element-batch machinery
needs one to be exercised: a mixed mesh is the case where a single integration
rule does not cover the whole mesh, so the elements have to be handled in groups.

``n`` must be large enough that every MPI rank receives elements.  Partitioning a
mesh with fewer elements than ranks **hangs** inside MFEM's partitioner rather
than raising, so callers should scale ``n`` with the rank count.
"""


def write_mixed_mesh(path, n=8):
    """Write an ``n x n`` mixed triangle/quadrilateral mesh of the unit square.

    The left half of the grid is split into triangles and the right half left as
    quadrilaterals.  Boundary attributes follow MFEM's Cartesian convention so
    that tests can reuse it: 1 bottom, 2 right, 3 top, 4 left.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    nv = (n + 1) * (n + 1)

    def vid(i, j):
        return j * (n + 1) + i

    elems = []
    half = n // 2
    for j in range(n):
        for i in range(n):
            v00, v10 = vid(i, j), vid(i + 1, j)
            v11, v01 = vid(i + 1, j + 1), vid(i, j + 1)
            if i < half:
                elems.append((2, (v00, v10, v11)))        # Geometry::TRIANGLE
                elems.append((2, (v00, v11, v01)))
            else:
                elems.append((3, (v00, v10, v11, v01)))   # Geometry::SQUARE

    bdr = []
    for i in range(n):
        bdr.append((1, (vid(i, 0), vid(i + 1, 0))))           # bottom
    for j in range(n):
        bdr.append((2, (vid(n, j), vid(n, j + 1))))           # right
    for i in range(n):
        bdr.append((3, (vid(i, n), vid(i + 1, n))))           # top
    for j in range(n):
        bdr.append((4, (vid(0, j), vid(0, j + 1))))           # left

    lines = ["MFEM mesh v1.0", "", "dimension", "2", "",
             "elements", str(len(elems))]
    for geom, vs in elems:
        lines.append("1 %d %s" % (geom, " ".join(str(v) for v in vs)))
    lines += ["", "boundary", str(len(bdr))]
    for attr, vs in bdr:
        lines.append("%d 1 %s" % (attr, " ".join(str(v) for v in vs)))
    lines += ["", "vertices", str(nv), "2"]
    h = 1.0 / n
    for j in range(n + 1):
        for i in range(n + 1):
            lines.append("%.17g %.17g" % (i * h, j * h))
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path

# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The benchmark shared by the hIPPYlibx and hIPPyMFEM validation drivers.

Nothing here imports either library: it only defines the *problem*, as plain
arrays and numbers, so that both drivers solve the same discrete problem rather
than two nearby ones.

What is shared, and why each matters:

mesh
    Vertices and cells, emitted here and read by both sides (dolfinx through
    ``create_mesh``, MFEM through a v1.0 mesh file).  A dolfinx unit square and
    an MFEM Cartesian mesh are both "n x n triangles" but are not the same mesh,
    and comparing across two meshes can only ever confirm agreement to
    discretization error.
observation targets
    Fixed coordinates, so the observation operators see identical points.
true parameter
    Analytic, so that projecting it gives the same discrete field on both sides
    without transferring dof vectors between incompatible orderings.
data
    Written by whichever driver runs first and read by the other, so the noise
    realization is bit-identical.
quadrature degree
    Pinned on both sides.  The residual density is not polynomial, so different
    rules give different (both valid) discrete operators, and that difference
    would otherwise be mistaken for an implementation discrepancy.
sample points
    Where fields are compared, since dof orderings differ.
"""

import json
import os

import numpy as np

#: mesh resolution
NX = 24
#: polynomial degrees of (state, parameter)
ORDER_STATE = 2
ORDER_PARAM = 1
#: integration rule degree, pinned on both sides
QUADRATURE_DEGREE = 6
#: prior coefficients and anisotropy
GAMMA = 0.1
DELTA = 0.5
THETA0, THETA1, ALPHA = 2.0, 0.5, np.pi / 4.0
ROBIN_BC = True
#: observations
NTARGETS = 80
TARGET_SEED = 1
REL_NOISE = 0.01
#: Laplace approximation
N_EIG = 40
N_OVERSAMPLE = 25
#: power iterations in the randomized eigensolver
N_POWER = 2
#: how many randomized eigenvalues are numerically determined well enough to
#: compare tightly.  Beyond this the spectral ratio exceeds 1e5 and the
#: B-orthogonalization loses most of its digits, so the two libraries differ by
#: round-off amplification rather than by anything about their implementations;
#: the deterministic dense spectrum below is the decisive comparison instead.
N_EIG_TIGHT = 14
#: leading eigenvalues compared from the exact (dense) generalized spectrum
N_DENSE_EIG = 40
#: mesh resolution for the dense spectrum (densifying costs one Hessian apply
#: per parameter dof, so this is deliberately coarser than the main case)
NX_DENSE = 12
#: seed for the deterministic analytic sketch
SKETCH_SEED = 20260911
#: Newton-CG
NEWTON_REL_TOL = 1e-9
NEWTON_MAX_ITER = 30
GN_ITER = 5
#: field comparison grid
NSAMPLE = 21


def theta_matrix():
    """The anisotropic diffusion tensor of the prior."""
    sa, ca = np.sin(ALPHA), np.cos(ALPHA)
    return np.array([
        [THETA0 * sa * sa + THETA1 * ca * ca, (THETA0 - THETA1) * sa * ca],
        [(THETA0 - THETA1) * sa * ca, THETA0 * ca * ca + THETA1 * sa * sa],
    ])


def m_true(x):
    """Smooth, recoverable true log-coefficient field.

    ``x`` may be ``(2,)`` or ``(2, npts)``; returns a scalar or ``(npts,)``.
    """
    x = np.asarray(x)
    return 1.0 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 0.4 * np.cos(2.0 * x[0])


def m_init(x):
    """Starting point for the gradient/Hessian checks (not the MAP iteration)."""
    x = np.asarray(x)
    return 0.3 * np.sin(x[0]) * np.cos(x[1])


def u_boundary(x):
    """Dirichlet data on the top and bottom edges."""
    x = np.asarray(x)
    return x[1]


def unit_square_triangles(n=NX):
    """``(vertices, cells, boundary)`` of an ``n x n`` triangulated unit square.

    ``boundary`` is a list of ``(attribute, (v0, v1))`` edges with MFEM's
    1-based attributes: 1 bottom, 2 right, 3 top, 4 left.
    """
    h = 1.0 / n
    verts = np.array([[i * h, j * h] for j in range(n + 1) for i in range(n + 1)],
                     dtype=np.float64)

    def vid(i, j):
        return j * (n + 1) + i

    cells = []
    for j in range(n):
        for i in range(n):
            cells.append([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)])
            cells.append([vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)])
    bdr = []
    for i in range(n):
        bdr.append((1, (vid(i, 0), vid(i + 1, 0))))
        bdr.append((3, (vid(i, n), vid(i + 1, n))))
    for j in range(n):
        bdr.append((2, (vid(n, j), vid(n, j + 1))))
        bdr.append((4, (vid(0, j), vid(0, j + 1))))
    return verts, np.array(cells, dtype=np.int64), bdr


def write_mfem_mesh(path, n=NX):
    """Write the shared mesh as an MFEM v1.0 mesh file."""
    verts, cells, bdr = unit_square_triangles(n)
    lines = ["MFEM mesh v1.0", "", "dimension", "2", "",
             "elements", str(len(cells))]
    for c in cells:
        lines.append("1 2 %d %d %d" % tuple(c))
    lines += ["", "boundary", str(len(bdr))]
    for attr, (a, b) in bdr:
        lines.append("%d 1 %d %d" % (attr, a, b))
    lines += ["", "vertices", str(len(verts)), "2"]
    for v in verts:
        lines.append("%.17g %.17g" % (v[0], v[1]))
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def targets(n=NTARGETS, seed=TARGET_SEED):
    """Observation coordinates, strictly inside the domain."""
    rng = np.random.default_rng(seed)
    return np.column_stack((rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.5, n)))


def sample_points(n=NSAMPLE):
    """Interior grid where fields are compared between the two libraries."""
    t = np.linspace(0.12, 0.88, n)
    X, Y = np.meshgrid(t, t, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel()))


def sketch_functions(nvec, seed=SKETCH_SEED):
    """Deterministic analytic functions spanning the randomized sketch.

    The randomized eigensolver resolves the leading eigenpairs to near machine
    precision but the trailing ones only to a few percent, so two libraries
    drawing *different* random sketches disagree on the tail by that much -- the
    algorithm's own sampling error, which would otherwise be mistaken for an
    implementation difference.  Projecting the same smooth analytic functions on
    both sides makes the sketch identical, and with it the whole eigensolve.

    Returns a list of ``nvec`` callables suitable for projection onto the
    parameter space.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(-6.0, 6.0, size=(nvec, 2))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=nvec)
    amps = rng.uniform(0.5, 1.5, size=nvec)

    def make(k):
        a, b = freqs[k]
        c, s_ = phases[k], amps[k]

        def f(x):
            x = np.asarray(x)
            return s_ * np.sin(a * x[0] + b * x[1] + c)

        return f

    return [make(k) for k in range(nvec)]


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(repr(o))


def read_json(path):
    with open(path) as fh:
        return json.load(fh)

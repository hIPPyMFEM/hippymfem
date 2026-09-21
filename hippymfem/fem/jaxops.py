# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""UFL-flavoured helpers for writing residual densities.

These are one-line wrappers over ``jax.numpy`` whose only purpose is to let a
residual density read like the UFL form it replaces::

    def pde_varf(u, m, p, x):
        return jnp.exp(m.val) * inner(u.grad, p.grad) - f(x) * p.val
"""

from .. import _jaxconfig        # noqa: F401  (must precede the jax import)

import jax.numpy as jnp


def inner(a, b):
    """Full contraction of two arrays of equal shape (scalars included)."""
    return jnp.sum(a * b)


def dot(a, b):
    """Single contraction over the last axis of ``a`` and the first of ``b``."""
    return jnp.tensordot(a, b, axes=1)


def div(grad_v):
    """Divergence of a vector field from its gradient ``(vdim, sdim)``."""
    return jnp.trace(grad_v)


def tr(A):
    """Trace."""
    return jnp.trace(A)


def sym(A):
    """Symmetric part."""
    return 0.5 * (A + A.T)


def skew(A):
    """Antisymmetric part."""
    return 0.5 * (A - A.T)


def Identity(n):
    """``n x n`` identity."""
    return jnp.eye(n)


def outer(a, b):
    """Outer product."""
    return jnp.outer(a, b)


def cross(a, b):
    """3-vector cross product."""
    return jnp.cross(a, b)


def nabla_grad(grad_v):
    """``grad_v`` transposed, i.e. the other gradient convention."""
    return grad_v.T


def sqrt(x):
    return jnp.sqrt(x)


def exp(x):
    return jnp.exp(x)


def ln(x):
    return jnp.log(x)


log = ln


def norm2(a):
    """Squared Euclidean norm."""
    return jnp.sum(a * a)


__all__ = [
    "inner", "dot", "div", "tr", "sym", "skew", "Identity", "outer", "cross",
    "nabla_grad", "sqrt", "exp", "ln", "log", "norm2",
]

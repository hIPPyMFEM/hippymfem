# Cross-validation against hIPPYlibx

This directory solves the **same discrete problem** with hIPPyMFEM and with
[hIPPYlibx](https://github.com/hIPPyMFEM/hippylibx) (hIPPYlib on FEniCSx), and diffs the results.

## Running

```bash
./validation/run_validation.sh [nx] [ranks]     # default: nx=12, 1 rank
```

The drivers run under different interpreters, because the two libraries usually live in
different environments; name them in the environment (both default to `python`):

| | variable | needs |
|---|---|---|
| hIPPyMFEM | `MFEM_PY` | PyMFEM 4.8 built with MPI, hIPPyMFEM |
| hIPPYlibx | `FENICSX_PY` | dolfinx 0.10 and hIPPYlibx: `pip install git+https://github.com/hIPPyMFEM/hippylibx.git`, or set `HIPPYLIBX_DIR` to a source checkout |

## What makes the comparison meaningful

Comparing two nearby discretizations can only ever confirm agreement to
discretization error, which is far too loose to catch a real bug.  So
`shared_case.py` pins everything that would otherwise differ:

| shared | why |
|---|---|
| mesh vertices and cells | a dolfinx unit square and an MFEM Cartesian mesh are both "n x n triangles" but are *not* the same mesh |
| quadrature degree | `exp(m)|grad u|^2` is not polynomial, so different rules give different (both valid) discrete operators |
| observation targets | identical observation operators |
| true parameter (analytic) | projecting the same function gives the same discrete field without transferring dof vectors between incompatible orderings |
| observation data | written by whichever driver runs first, read by the other, so the noise realization is bit-identical |
| randomized sketch | built from the same analytic functions, so the eigensolve is deterministic on both sides |
| field sample points | dof orderings differ, so fields are compared by evaluation, not by dof vector |

## Benchmark

A subsurface-flow problem: infer the log-coefficient `m` in
`-div(exp(m) grad u) = 0` on the unit square, `u = y` on the top and bottom
edges, from pointwise observations of `u`, with an anisotropic bi-Laplacian
(Matern) prior and Robin boundary conditions on the prior.

## Result (nx = 12, 1 rank), 27 quantities compared

Everything that is numerically determined agrees to near machine precision:

| quantity | relative difference |
|---|---|
| cost, regularization, misfit at a fixed `m₀` | 2e-14 |
| gradient norm, `g · m_true` | 2e-14 |
| `m_trueᵀ H m_true`, full and Gauss-Newton | 2e-14 |
| prior cost and trace | 1e-15 |
| **exact generalized spectrum of the misfit Hessian at a fixed point (40 values)** | **3.7e-13** |
| MAP total cost | 1.3e-14 |
| MAP state field at 441 points | 1.4e-09 |
| prior pointwise variance at 441 points | 9.4e-15 |
| observed data, true fields | 1e-14 |
| MAP parameter field at 441 points | 3.1e-08 |
| leading 14 randomized eigenvalues | 1e-07 |

The first group is the library under test: the discrete operators agree to
machine precision. The MAP *parameter field* agrees to 3e-08 rather than 1e-13
for a reason worth stating: the two optimizers stop at 10 and 11 Newton
iterations respectively, both satisfying a gradient tolerance of 1e-9. They
therefore agree on the cost to 1.3e-14 (the cost is flat at a minimum) while
landing about 1e-8 apart in the parameter. That is why the harness also evaluates
the Hessian spectrum at a **fixed analytic point**, where neither optimizer
enters, and gets 3.7e-13.

The full report is in `out/report_nx12_np1.txt`.

### The one loose comparison, and why

The **tail** of the randomized spectrum agrees only to ~1e-1, and the quantities
derived from it (posterior trace, posterior pointwise variance, KL divergence)
to ~1e-4 to 1e-3.  That is not an implementation difference: by eigenvalue 40
the spectral ratio `d[0]/d[39]` is about 5e7, the power iteration squares it,
and the `R`-orthogonalization loses most of its digits.  Both libraries are
round-off limited there, in different ways.

This is why the harness also computes the spectrum **exactly**, by densifying
the Hessian and the prior precision and calling `scipy.linalg.eigh`.  That
removes the randomized solver from the comparison entirely and tests the Hessian
operator itself, which is the thing actually under test.  It is reported twice:
at the MAP point (`dense_eigenvalues`, 4.8e-08, inheriting the MAP-point
difference) and at the fixed analytic point `m_init`
(`dense_eigenvalues_at_m0`, **3.7e-13**), which depends on neither optimizer and
is the decisive comparison.  The randomized eigensolver is separately checked
against dense `eigh` in `hippymfem/test/test_optimization.py`, where it matches to
1e-15.

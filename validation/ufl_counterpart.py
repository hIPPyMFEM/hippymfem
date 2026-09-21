#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""The UFL side of E1: which of the expressiveness cases a form language can write, at
what cost, and with what derivatives.

The cases are those of ``benchmarks/bench_vs_hippylibx.py`` (the same weights, tables
and constants), written in UFL where UFL can write them: the MLP as sixteen ``tanh``
terms, the inner Newton solve unrolled into an expression, the branch as
``ufl.conditional``, the table as a nested conditional (linear) or a sum of Gaussians
(smooth).  For each: FFCx compile time and generated code size for the six forms a
Bayesian inversion needs (residual, Jacobian, parameter derivative and the three
Hessian blocks), assembly time of the residual and the Jacobian, and finite-difference
slopes of the Jacobian in ``u`` and of the parameter derivative in ``m``.

What no case here can show is what UFL has no syntax for: a loop whose trip count depends
on the data (a Newton solve to a tolerance), a gather by a computed index (``jnp.take``,
``jnp.interp`` on a large table), and code that lives outside the form (a trained model
called at the quadrature points).  The nested-conditional table at 33 and 257 knots
measures how the expressible substitute for a gather scales.

    python validation/ufl_counterpart.py --out results/ufl_counterpart.json
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cases():
    import ufl

    out = []

    def reference(u, m, p):
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) - p
    out.append(("reference", "exp(m) grad u . grad p", reference, "yes"))

    rng = np.random.default_rng(0)
    width = 16
    W1 = rng.standard_normal((width, 2)) / np.sqrt(2.0)
    b1 = rng.standard_normal(width) * 0.1
    W2 = rng.standard_normal((1, width)) / np.sqrt(width)
    b2 = rng.standard_normal(1) * 0.1

    def mlp(u, m, p):
        acc = b2[0]
        for i in range(width):
            acc = acc + W2[0, i] * ufl.tanh(W1[i, 0] * u + W1[i, 1] * m + b1[i])
        return ufl.exp(acc) * ufl.inner(ufl.grad(u), ufl.grad(p)) - p
    out.append(("A", "neural-network closure, 2x16 tanh MLP written out", mlp, "yes, transcribed by hand"))

    def local_newton(u, m, p, steps=3):
        target = ufl.exp(m) * (1.0 + u ** 2)
        s = target / 2.0
        for _ in range(steps):
            s = s - (s ** 3 + s - target) / (3.0 * s ** 2 + 1.0)
        return s * ufl.inner(ufl.grad(u), ufl.grad(p)) - p
    out.append(("B", "3-step inner Newton solve, unrolled", local_newton, "yes, unrolled (a fixed count only)"))

    def branch(u, m, p, eps=1e-2):
        g2 = ufl.inner(ufl.grad(u), ufl.grad(u))
        nu = ufl.conditional(ufl.gt(g2, eps ** 2), ufl.exp(m) * (g2 + eps ** 2) ** (-0.25), ufl.exp(m) * eps ** (-0.5))
        return nu * ufl.inner(ufl.grad(u), ufl.grad(p)) - p
    out.append(("C", "regularized non-smooth law with a branch (ufl.conditional)", branch, "yes"))

    def table_linear_factory(n):
        rng = np.random.default_rng(1)
        grid = np.linspace(-3.0, 3.0, n)
        table = np.exp(0.5 * grid) + 0.1 * rng.standard_normal(n) ** 2

        def varf(u, m, p):
            kappa = ufl.conditional(ufl.lt(m, grid[0]), table[0], 0.0)
            for i in range(n - 1):
                x0, x1, y0, y1 = grid[i], grid[i + 1], table[i], table[i + 1]
                lin = y0 + (m - x0) * (y1 - y0) / (x1 - x0)
                kappa = kappa + ufl.conditional(ufl.And(ufl.ge(m, x0), ufl.lt(m, x1)), lin, 0.0)
            kappa = kappa + ufl.conditional(ufl.ge(m, grid[-1]), table[-1], 0.0)
            return kappa * ufl.inner(ufl.grad(u), ufl.grad(p)) - p
        return varf
    out.append(("D1", "tabulated coefficient, 33 knots, linear (32 nested conditionals)", table_linear_factory(33),
                "yes, as one conditional per interval"))
    out.append(("D1-257", "the same table at 257 knots (256 conditionals)", table_linear_factory(257),
                "yes, as one conditional per interval"))

    def table_smooth(u, m, p, n=33):
        rng = np.random.default_rng(1)
        nodes = np.linspace(-3.0, 3.0, n)
        w = 1.2 * (nodes[1] - nodes[0])
        vals = np.exp(0.5 * nodes) + 0.1 * rng.standard_normal(n) ** 2
        num, den = 0.0, 0.0
        for i in range(n):
            wt = ufl.exp(-0.5 * ((m - nodes[i]) / w) ** 2)
            num = num + wt * vals[i]
            den = den + wt
        return (num / den) * ufl.inner(ufl.grad(u), ufl.grad(p)) - p
    out.append(("D2", "the same table, Gaussian kernel (33 exponentials)", table_smooth, "yes, as a sum of exponentials"))
    return out


def fd_slope(f, x0, d, apply_lin, eps_list):
    """Median slope of ``|f(x0 + e d) - f(x0) - e L d|`` against ``e``."""
    f0 = f(x0)
    Ld = apply_lin(d)
    errs = []
    for e in eps_list:
        errs.append(np.linalg.norm(f(x0 + e * d) - f0 - e * Ld))
    errs = np.array(errs)
    good = (errs[:-1] > 0) & (errs[1:] > 0)
    sl = np.log(errs[:-1][good] / errs[1:][good]) / np.log(np.asarray(eps_list[:-1])[good] / np.asarray(eps_list[1:])[good])
    return float(np.median(sl)) if sl.size else float("nan"), errs


def run_case(key, desc, varf, n, order, cache_root, qdeg=None):
    import ufl
    from dolfinx import fem, mesh as dmesh
    from mpi4py import MPI

    # 256 nested conditionals overflow UFL's recursive DAG traverser at the default limit
    sys.setrecursionlimit(1000000)
    # Without a fixed quadrature degree UFL estimates one per form, and the derivative
    # forms can get a different rule from the residual's: the Jacobian is then not the
    # derivative of the assembled residual (FD slopes of 1 instead of 2 for the MLP), and
    # the unrolled Newton solve's estimated degree explodes (FFCx asked for a
    # 765625 x 765625 table).  ``dx`` here carries the degree when one is given.
    dx = ufl.dx if qdeg is None else ufl.dx(metadata={"quadrature_degree": int(qdeg)})
    comm = MPI.COMM_WORLD
    msh = dmesh.create_unit_square(comm, n, n, dmesh.CellType.quadrilateral)
    Vu = fem.functionspace(msh, ("Lagrange", order))
    Vm = fem.functionspace(msh, ("Lagrange", 1))
    u, m, pf = fem.Function(Vu), fem.Function(Vm), fem.Function(Vu)
    rng = np.random.default_rng(3)
    u.x.array[:] = 0.5 * rng.standard_normal(u.x.array.size)
    m.x.array[:] = 0.5 * rng.standard_normal(m.x.array.size)
    pf.x.array[:] = rng.standard_normal(pf.x.array.size)
    p = ufl.TestFunction(Vu)
    F = varf(u, m, p) * dx
    J = ufl.derivative(F, u, ufl.TrialFunction(Vu))
    G = ufl.derivative(F, m, ufl.TrialFunction(Vm))
    L = varf(u, m, pf) * dx
    Wuu = ufl.derivative(ufl.derivative(L, u, ufl.TestFunction(Vu)), u, ufl.TrialFunction(Vu))
    Wum = ufl.derivative(ufl.derivative(L, u, ufl.TestFunction(Vu)), m, ufl.TrialFunction(Vm))
    Wmm = ufl.derivative(ufl.derivative(L, m, ufl.TestFunction(Vm)), m, ufl.TrialFunction(Vm))
    cache = tempfile.mkdtemp(prefix="ffcx_%s_" % key, dir=cache_root)
    t0 = time.perf_counter()
    forms = [fem.form(f, jit_options={"cache_dir": cache}) for f in (F, J, G, Wuu, Wum, Wmm)]
    t_compile = time.perf_counter() - t0
    code_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(cache, "*.c")))
    fF, fJ, fG = forms[:3]
    # assembly cost, warm
    fem.assemble_vector(fF)
    t0 = time.perf_counter()
    b = fem.assemble_vector(fF)
    t_res = time.perf_counter() - t0
    A = fem.assemble_matrix(fJ)
    t0 = time.perf_counter()
    A = fem.assemble_matrix(fJ)
    t_jac = time.perf_counter() - t0
    ne = msh.topology.index_map(2).size_local
    # finite differences: the Jacobian in u and the parameter derivative in m
    eps = [2.0 ** -k for k in range(3, 20)]
    u0, m0 = u.x.array.copy(), m.x.array.copy()

    def res_u(x):
        u.x.array[:] = x
        return fem.assemble_vector(fF).array.copy()

    def res_m(x):
        m.x.array[:] = x
        return fem.assemble_vector(fF).array.copy()
    Au = fem.assemble_matrix(fJ).to_scipy()
    du = rng.standard_normal(u0.size)
    slope_u, err_u = fd_slope(res_u, u0, du, lambda d: Au @ d, eps)
    u.x.array[:] = u0
    Am = fem.assemble_matrix(fG).to_scipy()
    dm = rng.standard_normal(m0.size)
    slope_m, err_m = fd_slope(res_m, m0, dm, lambda d: Am @ d, eps)
    m.x.array[:] = m0
    shutil.rmtree(cache, ignore_errors=True)
    return {"case": key, "description": desc, "t_compile": t_compile, "code_kb": code_bytes / 1024.0,
            "t_residual_us_per_elem": 1e6 * t_res / ne, "t_jacobian_us_per_elem": 1e6 * t_jac / ne,
            "slope_u": slope_u, "slope_m": slope_m, "elements": int(ne), "err_u_first": float(err_u[0]), "err_m_first": float(err_m[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per case; FFCx may not finish")
    ap.add_argument("--quadrature-degree", type=int, default=6, help="fixed rule for every form; 0 = UFL's estimate per form")
    ap.add_argument("--case", default=None, help="(internal) run one case and print its JSON")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import dolfinx
    if args.case:
        # one case in this process, so that a compile that does not finish can be timed out
        cache_root = tempfile.mkdtemp(prefix="ufl_counterpart_")
        for key, desc, varf, expressible in cases():
            if key == args.case:
                r = run_case(key, desc, varf, args.n, args.order, cache_root,
                             qdeg=(args.quadrature_degree if args.quadrature_degree > 0 else None))
                r["expressible"] = expressible
                print("JSON " + json.dumps(r), flush=True)
        shutil.rmtree(cache_root, ignore_errors=True)
        return 0
    import subprocess
    rows = []
    print("%-8s %-62s %-34s %9s %9s %9s %9s %7s %7s" % ("case", "law", "expressible", "compile s", "code KB", "res us/el", "jac us/el", "slope u", "slope m"), flush=True)
    for key, desc, varf, expressible in cases():
        cmd = [sys.executable, os.path.abspath(__file__), "--case", key, "--n", str(args.n), "--order", str(args.order),
               "--quadrature-degree", str(args.quadrature_degree)]
        t0 = time.perf_counter()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            lines = [ln for ln in out.stdout.splitlines() if ln.startswith("JSON ")]
            r = json.loads(lines[-1][5:]) if lines else {"case": key, "description": desc, "expressible": expressible,
                                                          "error": (out.stderr or "")[-2000:]}
        except subprocess.TimeoutExpired:
            r = {"case": key, "description": desc, "expressible": expressible, "timed_out": True,
                 "t_compile": float("nan"), "timeout": args.timeout}
        r["t_wall"] = time.perf_counter() - t0
        rows.append(r)
        if "t_compile" in r and not r.get("timed_out"):
            print("%-8s %-62s %-34s %9.2f %9.0f %9.2f %9.2f %7.3f %7.3f"
                  % (key, desc[:62], expressible[:34], r["t_compile"], r["code_kb"], r["t_residual_us_per_elem"], r["t_jacobian_us_per_elem"], r["slope_u"], r["slope_m"]), flush=True)
        else:
            print("%-8s %-62s %-34s %s" % (key, desc[:62], expressible[:34],
                                            "compile did not finish in %.0f s" % args.timeout if r.get("timed_out") else "failed: " + r.get("error", "")[-300:].replace("\n", " ")), flush=True)
    rec = {"dolfinx": dolfinx.__version__, "n": args.n, "order": args.order, "quadrature_degree": args.quadrature_degree, "rows": rows,
           "not_expressible": ["a Newton solve iterated to a tolerance (data-dependent trip count)",
                               "a gather by a computed index (jnp.take, jnp.interp on a large table)",
                               "code outside the form: a trained model evaluated by its own library at the quadrature points"]}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

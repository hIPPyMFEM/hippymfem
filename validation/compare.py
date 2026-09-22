#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Compare the two validation runs and report the agreement.

Usage::

    python validation/compare.py validation/out/mfem.json validation/out/hippylibx.json
"""

import argparse
import json
import sys

import numpy as np

#: tolerances on the relative difference of each reported quantity
TOL = {
    "at_m0.cost_total": 1e-9,
    "at_m0.cost_reg": 1e-9,
    "at_m0.cost_misfit": 1e-9,
    "at_m0.grad_norm_Rinv": 1e-7,
    "at_m0.grad_dot_mtrue": 1e-8,
    "at_m0.mtrue_H_mtrue": 1e-7,
    "at_m0.mtrue_HGN_mtrue": 1e-7,
    "prior.cost_mtrue": 1e-10,
    "prior.trace_exact": 1e-9,
    "map.final_cost": 1e-8,
    "map.final_reg": 1e-6,
    "map.final_misfit": 1e-6,
    "kl_from_prior": 1e-3,
    "traces.posterior": 1e-3,
    "traces.prior": 1e-9,
    "traces.correction": 1e-3,
    "eigenvalues_leading": 1e-6,
    "eigenvalues_tail": 1.0,       # informational: the two solvers differ there, see NOTES
    "dense_eigenvalues_at_m0": 1e-9,
    "dense_eigenvalues": 1e-6,
    "clean_data": 1e-10,
    "fields.m_true": 1e-10,
    "fields.u_true": 1e-9,
    "fields.m_map": 1e-6,
    "fields.u_map": 1e-7,
    "fields.post_variance": 5e-3,
    "fields.prior_variance": 1e-8,
}


def get(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def relerr(a, b, elementwise=False):
    """Relative difference, scaled either by the array maximum or entrywise.

    Entrywise scaling matters for the eigenvalue list: a spectrum spanning six
    decades hides a several-percent error in its tail behind a 1e-10 error in its
    head when everything is scaled by the largest entry.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        return float("inf")
    if elementwise:
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
        return float((np.abs(a - b) / scale).max())
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    return float(np.abs(a - b).max() / scale)


#: quantities compared entrywise rather than scaled by the array maximum
ELEMENTWISE = {"eigenvalues_leading", "eigenvalues_tail",
               "dense_eigenvalues", "dense_eigenvalues_at_m0"}

#: how many leading randomized eigenvalues to hold to a tight tolerance
N_EIG_TIGHT = 14

#: notes printed for quantities whose loose tolerance is deliberate
NOTES = {
    "dense_eigenvalues": ("evaluated at the MAP point, so it inherits the "
                          "~1e-8 difference between the two MAP points; "
                          "dense_eigenvalues_at_m0 is the clean test of the "
                          "Hessian operator"),
    "fields.m_map": ("the two optimizers stop at the same cost to 1e-14 but at "
                     "points ~1e-8 apart, which is the gradient tolerance"),
    "eigenvalues_tail": ("beyond the leading %d the two randomized solvers differ: "
                         "hIPPyMFEM re-orthonormalizes between power iterations and "
                         "its tail is within 2e-2 of the dense spectrum; hIPPYlibx "
                         "orthonormalizes once, and at this spectral ratio its tail "
                         "is round-off limited (0.77 at the 40th).  Only the dense "
                         "spectrum is a comparison of the operators" % N_EIG_TIGHT),
    "traces.posterior": "inherits the randomized tail",
    "traces.correction": "inherits the randomized tail",
    "kl_from_prior": "inherits the randomized tail",
    "fields.post_variance": "inherits the randomized tail",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if any quantity exceeds its tolerance")
    args = ap.parse_args()

    A = json.load(open(args.a))
    B = json.load(open(args.b))
    for D in (A, B):
        ev = D.get("eigenvalues")
        if ev is not None:
            D["eigenvalues_leading"] = ev[:N_EIG_TIGHT]
            D["eigenvalues_tail"] = ev[N_EIG_TIGHT:]

    print("=" * 78)
    print("%-28s %-22s %-22s" % ("", A.get("library", args.a), B.get("library", args.b)))
    print("=" * 78)
    for key in ("nx", "nranks"):
        print("%-28s %-22s %-22s" % (key, A.get(key), B.get(key)))
    for key in ("state", "parameter"):
        print("%-28s %-22s %-22s" % ("ndofs." + key,
                                     get(A, "ndofs." + key), get(B, "ndofs." + key)))
    for key in ("newton_iterations", "total_cg_iterations", "converged"):
        print("%-28s %-22s %-22s" % ("map." + key,
                                     get(A, "map." + key), get(B, "map." + key)))
    print("-" * 78)
    print("%-28s %14s %14s %12s %6s" % ("quantity", "A", "B", "rel diff", "tol"))
    print("-" * 78)

    failures = []
    for key, tol in TOL.items():
        va, vb = get(A, key), get(B, key)
        if va is None or vb is None:
            print("%-28s %14s %14s %12s %6s" % (key, "-", "-", "missing", ""))
            continue
        e = relerr(va, vb, elementwise=key in ELEMENTWISE)
        scalar = np.ndim(va) == 0
        fa = "%14.7e" % va if scalar else "%14s" % ("[%d]" % np.size(va))
        fb = "%14.7e" % vb if scalar else "%14s" % ("[%d]" % np.size(vb))
        flag = "" if e <= tol else "  <== EXCEEDS"
        print("%-28s %s %s %12.2e %6.0e%s" % (key, fa, fb, e, tol, flag))
        if key in NOTES and e > 1e-6:
            print("%-28s %s" % ("", "note: " + NOTES[key]))
        if e > tol:
            failures.append((key, e, tol))

    print("-" * 78)
    if failures:
        print("%d quantit%s exceeded tolerance:" % (len(failures),
                                                    "y" if len(failures) == 1 else "ies"))
        for k, e, t in failures:
            print("   %-30s rel diff %.3e > %.0e" % (k, e, t))
    else:
        print("all %d compared quantities agree within tolerance" % len(TOL))
    print("=" * 78)
    return 1 if (failures and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

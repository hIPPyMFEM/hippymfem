#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Record every device allocation and free that hypre makes during forward solves at new
parameters, so that pool policies can be compared offline (``hypre_pool_replay.py``).

The library's pool is installed with a limit of zero (it then hands every request to the
driver and keeps nothing) and its two entry points are wrapped: each event is
``("a", id, bytes, phase, seconds)`` or ``("f", id, phase, seconds)``, where ``phase``
is the step of the solve that made the call and ``seconds`` the time the driver took,
or ``("open", phase)`` and ``("close", phase)`` where a solver would open and close the
pool around a BoomerAMG setup.  The problem is that of ``bench_forward_steps.py``; with
``--hessian`` each forward solve is followed by the adjoint solve, the Hessian blocks and
two actions::

    HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n 16 tools/mpirun_pinned.sh \\
        python benchmarks/hypre_pool_trace.py --n 161 --cart-part --hessian --out trace.pkl
    python benchmarks/hypre_pool_replay.py trace.pkl --rank 0
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

HM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HM)

import hippymfem as hm                                               # noqa: E402
from mpi4py import MPI                                               # noqa: E402

import mfem.par as mfem                                              # noqa: E402
import jax.numpy as jnp                                              # noqa: E402

from hippymfem.common import mfemconfig                              # noqa: E402
from hippymfem.modeling.variables import PARAMETER, STATE, ADJOINT   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=200)
    ap.add_argument("--solves", type=int, default=3)
    ap.add_argument("--hessian", action="store_true", help="also a linearization point and two actions")
    ap.add_argument("--cart-part", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hm.configure_device("cuda", COMM, quiet=(RANK != 0))
    pool = mfemconfig.set_hypre_pool(1.0, scoped=True)       # scoped and never opened: keeps nothing
    if pool is None:
        raise SystemExit("no pool: hypre is not on a CUDA device")
    events, phase, ids = [], ["setup of the problem"], {}
    # the solvers call these two around a BoomerAMG setup: record where, and keep nothing
    mfemconfig.hypre_pool_open = lambda: events.append(("open", phase[0]))
    mfemconfig.hypre_pool_close = lambda: events.append(("close", phase[0]))
    counter = [0]
    # hypre holds the pool's two callbacks; they look these two up on the instance
    take, dev_free = pool._take, pool._dev_free
    clock = time.perf_counter

    def _take(out, size):
        t0 = clock()
        take(out, size)
        dt = clock() - t0
        p = out[0]
        if p:
            counter[0] += 1
            ids[p] = counter[0]
            events.append(("a", counter[0], int(size), phase[0], dt))

    def _dev_free(p):
        t0 = clock()
        rc = dev_free(p)
        dt = clock() - t0
        if p and p in ids:
            events.append(("f", ids.pop(p), phase[0], dt))
        return rc

    pool._take, pool._dev_free = _take, _dev_free

    N, ORDER = args.n, args.order
    serial = mfem.Mesh.MakeCartesian3D(N, N, N, mfem.Element.HEXAHEDRON)
    if args.cart_part:
        p = COMM.size
        grid = min(((a, b, p // (a * b)) for a in range(1, p + 1) if p % a == 0
                    for b in range(a, p // a + 1) if (p // a) % b == 0 and b <= p // (a * b)),
                   key=lambda g: max(g) / min(g))
        nxyz = mfem.intArray(list(grid))
        pmesh = mfem.ParMesh(COMM, serial, serial.CartesianPartitioning(nxyz.GetData()))
    else:
        pmesh = mfem.ParMesh(COMM, serial)
    del serial
    Vu = hm.FunctionSpace.H1(pmesh, ORDER)
    Vm = hm.FunctionSpace.H1(pmesh, 1)
    ndof = int(Vu.GlobalTrueVSize())
    say("%d^3 hex order %d: %d state dofs, %d ranks, tree %s" % (N, ORDER, ndof, COMM.size, HM))
    pde_varf = lambda u, m, p, x: jnp.exp(m.val) * hm.inner(u.grad, p.grad)   # noqa: E731
    bc = hm.DirichletBC(Vu, lambda x: x[2], bdr_attributes=[1, 6])
    pde = hm.PDEVariationalProblem([Vu, Vm, Vu], pde_varf, bc, bc.homogeneous(), is_fwd_linear=True)
    pde.set_solvers(hm.auto_solver, Vu, COMM, max_direct=0, rel_tolerance=1e-12,
                    max_iter=2000, attributes=("solver", "solver_fwd_inc", "solver_adj_inc"))
    prior = hm.BiLaplacianPrior(Vm, 0.1, 0.5, robin_bc=True, solver_type="krylov")
    hm.parRandom.set_seed(1)
    noise = prior.noise_vector()
    prior.sample_noise(1.0, noise)
    mtrue = Vm.vector()
    prior.sample(noise, mtrue)
    rng = np.random.default_rng(1)
    targets = np.column_stack([rng.uniform(0.1, 0.9, args.ntargets) for _ in range(3)])
    B = hm.assemblePointwiseObservation(Vu, targets)
    utrue = pde.generate_state()
    pde.solveFwd(utrue, [utrue, mtrue, None])
    data = B.createVecLeft()
    B.mult(utrue, data)
    nstd = 0.01 * max(data.norm("linf"), 1e-30)
    B.perturb(data, nstd)
    model = hm.Model(pde, prior, hm.DiscreteStateObservation(B, data, nstd ** 2))
    x = [model.generate_vector(STATE), prior.mean.copy(), model.generate_vector(ADJOINT)]
    model.solveFwd(x[STATE], x)
    from hippymfem.modeling.PDEVariationalProblem import _set_operator_once

    for rep in range(args.solves):
        tag = "solve %d: " % rep
        x[PARAMETER].axpy(1e-3, mtrue)
        m = x[PARAMETER]
        phase[0] = tag + "release"
        if pde._lin_point is not None:
            pde.release_linearization_point()
        u = x[STATE]
        u.zero()
        pde.bc.apply(u)
        p = pde.Vh[ADJOINT].vector()
        du = pde.Vh[STATE].vector()
        solver = pde._get_solver("solver")
        phase[0] = tag + "residual"
        r = pde._residual([u, m, p], ADJOINT, ess=pde.bc0.ess)
        r.norm("l2")
        phase[0] = tag + "jacobian"
        J, _Jt = pde._jacobian([u, m, p], u=u)
        phase[0] = tag + "set operator"
        _set_operator_once(solver, J)
        r.scale(-1.0)
        phase[0] = tag + "first solve"
        solver.solve(du, r)
        phase[0] = tag + "after"
        u.axpy(1.0, du)
        r = pde._residual([u, m, p], ADJOINT, ess=pde.bc0.ess)
        r.norm("l2")
        if args.hessian:
            phase[0] = tag + "adjoint"
            model.solveAdj(x[ADJOINT], x)
            phase[0] = tag + "linearization point"
            model.setLinearizationPoint(x, gauss_newton_approx=False)
            H = hm.ReducedHessian(model)
            v, y = model.generate_vector(PARAMETER), model.generate_vector(PARAMETER)
            hm.parRandom.normal(1.0, v)
            for k in range(2):
                phase[0] = tag + "action %d" % k
                H.mult(v, y)
                y.norm("l2")
    phase[0] = "end"
    all_events = COMM.gather(events, root=0)
    if RANK == 0:
        with open(args.out, "wb") as f:
            pickle.dump({"n": N, "ranks": COMM.size, "tdofs": ndof, "events": all_events}, f)
        for rk, ev in enumerate(all_events[:2]):
            na = sum(1 for e in ev if e[0] == "a")
            say("rank %d: %d allocations, %d frees, %.1f MB allocated in total"
                % (rk, na, len(ev) - na, sum(e[2] for e in ev if e[0] == "a") / 2 ** 20))
    COMM.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

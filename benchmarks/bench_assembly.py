#!/usr/bin/env python
# Copyright (c) 2026, Peng Chen, Georgia Institute of Technology.
# This file is part of hIPPyMFEM, free software under the GNU General Public
# License version 2.0 dated June 1991 (GPL-2.0-only); see the files LICENSE and
# COPYRIGHT.
"""Assembly throughput: callback vs direct scatter, CPU vs GPU, across order and dimension.

Every assembly number in the documentation comes from here.  The measurement is deliberately narrow and repeatable:

* one MPI rank per measurement unless ``--ranks`` says otherwise, so per-element
  costs are per-rank costs and not confounded by partitioning;
* the timed region is **one full assembly** of the forward Jacobian -- dof gather,
  element kernel, scatter, ``P^T A P`` and essential-dof elimination -- because
  that is what an optimizer actually pays, not just the kernel;
* the first call is discarded (JAX compiles, the sparsity pattern is built), and
  the reported time is the mean of ``--reps`` later calls, which is the steady
  state a Newton or MCMC run spends its life in;
* both backends are compared **in one process**, so the mesh, the dof numbering
  and the input vectors are the same objects.

Results are written as JSON, so tables and figures are generated from the
measurement rather than retyped.

Usage::

    HIPPYMFEM_DEVICE=gpu python benchmarks/bench_assembly.py --out results/asm.json
    HIPPYMFEM_DEVICE=gpu mpirun -n 4 python benchmarks/bench_assembly.py --ranks 4
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
from mpi4py import MPI

import mfem.par as mfem
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hippymfem as hp                                              # noqa: E402
from hippymfem.fem import assemble as asm                           # noqa: E402
from hippymfem.fem import kernel as K                               # noqa: E402
from hippymfem.fem.csrassemble import get_pattern                   # noqa: E402
from hippymfem.fem.elementbatch import MeshBatches                  # noqa: E402
from hippymfem.fem.kernel import QuadratureKernel                   # noqa: E402
from hippymfem.fem.spaces import FunctionSpace                      # noqa: E402
from hippymfem.modeling.variables import ADJOINT, PARAMETER, STATE   # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank
NP = COMM.size


def say(*a):
    if RANK == 0:
        print(*a, flush=True)


def have_gpu():
    import jax

    try:
        return len(jax.devices("cuda")) > 0
    except RuntimeError:
        return any(d.platform == "gpu" for d in jax.devices())


def machine():
    """What the numbers were measured on, recorded with them."""
    info = {"host": platform.node(), "ranks": NP,
            "cpu_count": os.cpu_count(),
            "python": platform.python_version()}
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        info["gpus"] = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
    except Exception:
        info["gpus"] = []
    try:
        import mfem

        info["pymfem"] = mfem.__version__
    except Exception:
        pass
    info["mfem_cuda"] = mfem_has_cuda()
    return info


def mfem_has_cuda():
    """Whether this PyMFEM was built with CUDA, read from its config header.

    Probing by calling ``Device::Configure("cuda")`` would be simpler and is wrong:
    on a CPU-only build MFEM responds with ``MFEM_ABORT``, which takes the whole
    MPI job down rather than returning false.
    """
    return hp.mfem_config().get("MFEM_USE_CUDA", False)


# --------------------------------------------------------------------- problems
def make(kind, n, order, density="diffusion"):
    if kind == "quad":
        m = mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.QUADRILATERAL)
    elif kind == "tri":
        m = mfem.Mesh.MakeCartesian2D(n, n, mfem.Element.TRIANGLE)
    elif kind == "hex":
        m = mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.HEXAHEDRON)
    else:
        m = mfem.Mesh.MakeCartesian3D(n, n, n, mfem.Element.TETRAHEDRON)
    pm = mfem.ParMesh(COMM, m)
    Vu = FunctionSpace.H1(pm, order)
    Vm = FunctionSpace.H1(pm, 1)
    batches = MeshBatches(pm, 2 * order + 2, COMM)

    if density == "diffusion":
        def varf(u, m_, p, x):
            return jnp.exp(m_.val) * jnp.dot(u.grad, p.grad)
    elif density == "nonlinear":
        def varf(u, m_, p, x):
            return (jnp.exp(m_.val) * (1.0 + u.val ** 2) * jnp.dot(u.grad, p.grad)
                    + jnp.sin(u.val) * p.val)
    else:
        raise ValueError(density)

    kern = QuadratureKernel(varf, [Vu, Vm, Vu], batches)
    hp.parRandom.set_seed(4)
    uv = Vu.vector()
    hp.parRandom.normal(1.0, uv)
    mv = Vm.vector()
    hp.parRandom.normal(0.3, mv)
    pv = Vu.vector()
    hp.parRandom.normal(1.0, pv)
    loc = [Vu.local_values(uv), Vm.local_values(mv), Vu.local_values(pv)]
    bc = hp.DirichletBC(Vu, None, "all")
    return pm, Vu, Vm, batches, kern, loc, bc


def timed(fn, reps):
    fn()
    COMM.Barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    COMM.Barrier()
    return (time.perf_counter() - t0) / reps


def measure(kind, n, order, backend, device, reps, density="diffusion"):
    """One configuration: kernel time, assembly time, and the total."""
    pm, Vu, Vm, b, kern, loc, bc = make(kind, n, order, density)
    NE = pm.GetNE()
    old_backend = asm.set_assembly_backend(backend)
    old_dev = K.device()
    K.set_device(device)
    try:
        # kernel alone: force completion without moving the result to the host
        def kernel_only():
            mats = kern.element_matrices(ADJOINT, STATE, loc)
            float(jnp.sum(mats[0][0, 0])) if not isinstance(
                mats[0], np.ndarray) else float(mats[0][0, 0, 0])
        t_kernel = timed(kernel_only, reps)

        mats = kern.element_matrices(ADJOINT, STATE, loc)

        def scatter_only():
            A = asm.assemble_matrix(Vu, Vu, b.groups, mats, NE,
                                    test_ess=bc.ess_tdof)
            del A
        t_scatter = timed(scatter_only, reps)

        def full():
            A = asm.assemble_matrix(Vu, Vu, b.groups,
                                    kern.element_matrices(ADJOINT, STATE, loc),
                                    NE, test_ess=bc.ess_tdof)
            del A
        t_full = timed(full, reps)
        nnz = (get_pattern(Vu, Vu, b.groups).nnz if backend == "csr" else None)
    finally:
        asm.set_assembly_backend(old_backend)
        K._DEVICE = old_dev
    ne_glob = COMM.allreduce(NE)
    return {
        "kind": kind, "n": n, "order": order, "backend": backend,
        "device": device, "density": density,
        "NE_local": NE, "NE_global": ne_glob,
        "tdofs": Vu.GlobalTrueVSize(), "nnz_local": nnz,
        "t_kernel": t_kernel, "t_scatter": t_scatter, "t_full": t_full,
        "us_per_elem_kernel": 1e6 * t_kernel / NE,
        "us_per_elem_scatter": 1e6 * t_scatter / NE,
        "us_per_elem_full": 1e6 * t_full / NE,
        "elem_per_sec_node": ne_glob / t_full,
    }


# ------------------------------------------------------------------ experiments
def exp_backend(reps):
    """The callback route against the direct scatter, on the CPU."""
    say("\n=== 1. assembly route: MFEM per-element callback vs direct CSR scatter "
        "(CPU) ===")
    say("%-18s %10s %12s %12s %9s" % ("case", "NE/rank", "callback us/e",
                                      "direct us/e", "speedup"))
    rows = []
    for kind, n, order in (("quad", 180, 1), ("quad", 128, 2), ("quad", 90, 3),
                           ("tri", 180, 1), ("hex", 30, 1), ("hex", 20, 2)):
        a = measure(kind, n, order, "integrator", "cpu", reps)
        c = measure(kind, n, order, "csr", "cpu", reps)
        rows += [a, c]
        say("%-18s %10d %12.3f %12.3f %8.2fx"
            % ("%s P%d" % (kind, order), a["NE_local"],
               a["us_per_elem_scatter"], c["us_per_elem_scatter"],
               a["us_per_elem_scatter"] / c["us_per_elem_scatter"]))
    return rows


def exp_device(reps, gpu):
    """CPU against GPU for the AD layer, across order and dimension."""
    say("\n=== 2. AD layer on CPU vs GPU (direct CSR assembly) ===")
    if not gpu:
        say("  no GPU visible; skipped")
        return []
    say("%-18s %10s %11s %11s %9s %11s %11s %9s"
        % ("case", "NE/rank", "CPU kern", "GPU kern", "kern x",
           "CPU full", "GPU full", "full x"))
    rows = []
    cases = (("quad", 180, 1), ("quad", 128, 2), ("quad", 90, 3), ("quad", 64, 4),
             ("hex", 30, 1), ("hex", 20, 2), ("hex", 14, 3),
             ("tri", 180, 1), ("tet", 30, 1), ("tet", 20, 2))
    for kind, n, order in cases:
        c = measure(kind, n, order, "csr", "cpu", reps)
        g = measure(kind, n, order, "csr", "gpu", reps)
        rows += [c, g]
        say("%-18s %10d %11.3f %11.3f %8.2fx %11.3f %11.3f %8.2fx"
            % ("%s P%d" % (kind, order), c["NE_local"],
               c["us_per_elem_kernel"], g["us_per_elem_kernel"],
               c["us_per_elem_kernel"] / g["us_per_elem_kernel"],
               c["us_per_elem_full"], g["us_per_elem_full"],
               c["us_per_elem_full"] / g["us_per_elem_full"]))
    return rows


def exp_size(reps, gpu):
    """How the per-element cost depends on the batch size."""
    say("\n=== 3. cost per element vs elements per rank ===")
    say("%-10s %10s %11s %11s %9s" % ("case", "NE/rank", "CPU us/e", "GPU us/e",
                                      "speedup"))
    rows = []
    for n in (16, 32, 64, 128, 180, 256, 320):
        c = measure("quad", n, 2, "csr", "cpu", max(2, reps // 2))
        rows.append(c)
        if gpu:
            g = measure("quad", n, 2, "csr", "gpu", max(2, reps // 2))
            rows.append(g)
            say("%-10s %10d %11.3f %11.3f %8.2fx"
                % ("quad P2", c["NE_local"], c["us_per_elem_full"],
                   g["us_per_elem_full"],
                   c["us_per_elem_full"] / g["us_per_elem_full"]))
        else:
            say("%-10s %10d %11.3f %11s %9s"
                % ("quad P2", c["NE_local"], c["us_per_elem_full"], "-", "-"))
    return rows


def exp_nonlinear(reps, gpu):
    """A harder density, which is where the AD route and the GPU both matter most."""
    say("\n=== 4. a nonlinear density (exp(m)(1+u^2) grad u . grad p + sin(u) p) ===")
    say("%-18s %10s %11s %11s %9s" % ("case", "NE/rank", "CPU us/e", "GPU us/e",
                                      "speedup"))
    rows = []
    for kind, n, order in (("quad", 128, 2), ("hex", 20, 2)):
        c = measure(kind, n, order, "csr", "cpu", reps, density="nonlinear")
        rows.append(c)
        if gpu:
            g = measure(kind, n, order, "csr", "gpu", reps, density="nonlinear")
            rows.append(g)
            say("%-18s %10d %11.3f %11.3f %8.2fx"
                % ("%s P%d" % (kind, order), c["NE_local"],
                   c["us_per_elem_full"], g["us_per_elem_full"],
                   c["us_per_elem_full"] / g["us_per_elem_full"]))
        else:
            say("%-18s %10d %11.3f %11s %9s"
                % ("%s P%d" % (kind, order), c["NE_local"],
                   c["us_per_elem_full"], "-", "-"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", default="all",
                    help="comma-separated subset of 1,2,3,4")
    args = ap.parse_args()

    mfem.Hypre.Init()
    gpu = have_gpu()
    info = machine()
    say("hIPPyMFEM assembly benchmark")
    say("  host %s, %d ranks, %d cores, GPUs: %s"
        % (info["host"], NP, info["cpu_count"] or -1, info["gpus"] or "none"))
    say("  PyMFEM %s, MFEM CUDA: %s, JAX GPU: %s"
        % (info.get("pymfem", "?"), info["mfem_cuda"], gpu))

    want = ("1", "2", "3", "4") if args.only == "all" else tuple(
        args.only.split(","))
    rows = []
    if "1" in want:
        rows += exp_backend(args.reps)
    if "2" in want:
        rows += exp_device(args.reps, gpu)
    if "3" in want:
        rows += exp_size(args.reps, gpu)
    if "4" in want:
        rows += exp_nonlinear(args.reps, gpu)

    if args.out and RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"machine": info, "rows": rows}, open(args.out, "w"), indent=1)
        say("\nwrote %s (%d rows)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

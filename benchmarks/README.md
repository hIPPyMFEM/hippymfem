# Benchmarks

The scripts behind the numbers in the README and the user guide. Each one prints its table
and writes a JSON record with `--out`; `--help` lists the options. Run the GPU ones through
`tools/mpirun_pinned.sh`, which gives each rank one card.

| script | what it measures | where the numbers are |
|---|---|---|
| `bench_newton_device.py` | the stages and two Newton-CG steps of the 3D model problem, on GPUs or on the host | README ("How it scales"), GPU guide (both tables of "What to expect") |
| `bench_newton_hippylibx.py` | the same problem solved by [hIPPYlibx](https://github.com/hIPPyMFEM/hippylibx); needs a dolfinx environment | README ("How it compares"), GPU guide |
| `bench_laplace.py` | after the MAP point: the randomized eigensolver, posterior samples, pointwise variance and traces; `--ensemble` divides the vectors over the GPUs instead of the mesh | GPU guide ("The Laplace approximation on the cards") |
| `bench_precision.py` | the three precisions of the element kernels (`fp64`, `mixed`, `fp32`): kernel and assembly times, the stages of an iteration, how far the state, the gradient and a Hessian action are from the double-precision ones, and a Newton-CG solve | GPU guide ("Single precision") |
| `bench_laplace_precision.py` | the Laplace stages at one MAP point with single- against double-precision solves: eigenvalues, pointwise variance and traces against a reference configuration, and the time of each stage | GPU guide ("Single precision") |
| `bench_amg_single.py` | BoomerAMG settings for the single-precision incremental solves, on right-hand sides captured from Hessian actions: setup time, iterations, time per solve and the error against double precision | GPU guide ("Single precision") |
| `cg_recycling_offline.py` | whether the CG of one Newton step helps the next, offline: its Ritz pairs as a limited-memory preconditioner or as a deflated initial guess, with the CG count of each | none: a feasibility study |
| `bench_scaling.py` | one reduced-Hessian action split into its operations, with Krylov iteration counts, to compare rank counts | GPU guide ("Several GPUs") |
| `bench_solvers_gpu.py` | the solves alone: hypre's CG with BoomerAMG on the host against the GPU, over mesh sizes | GPU guide ("What to expect") |
| `bench_assembly.py` | assembly time per element: the direct scatter against MFEM's callback, CPU against GPU, across element types and batch sizes | performance guide |
| `bench_assembly_sweep.py` | the same measurement over a range of mesh sizes, optionally with hypre on the GPU | GPU guide ("Where the device loses") |
| `bench_pipeline.py` | one assembly as cumulative stages: kernel, scatter, parallel reduction, elimination | performance guide ("Where the time goes") |
| `bench_vs_hippylibx.py` | six residual densities that a form language cannot write (a neural-network closure, an inner Newton solve, a table lookup, ...), each with finite-difference checks of its gradient and Hessian | user guide ("Densities a form language cannot express") |

The model problem of the Newton and Laplace benchmarks: `n^3` hexahedra with a second-order
state and adjoint and a first-order parameter, the PDE `-div(exp(m) grad u) = 0` with
`u = z` on the bottom and top faces, a BiLaplacian prior (gamma 0.1, delta 0.5, Robin),
a prior sample as the true parameter, 200 pointwise observations at 1 % noise, CG with
BoomerAMG for every solve.

```bash
# two Newton-CG steps at 64^3 on four GPUs
HIPPYMFEM_DEVICE=gpu HIPPYMFEM_HYPRE_DEVICE=1 mpirun -n 4 tools/mpirun_pinned.sh \
    python benchmarks/bench_newton_device.py --n 64 --steps 2

# the same on four host ranks, and with hIPPYlibx
mpirun -n 4 python benchmarks/bench_newton_device.py --n 64 --steps 2 --device cpu
mpirun -n 4 python benchmarks/bench_newton_hippylibx.py --n 64 --steps 2
```

[`DESIGN_NOTES.md`](DESIGN_NOTES.md) explains, module by module, why the code is built the
way it is and what was measured to decide it.

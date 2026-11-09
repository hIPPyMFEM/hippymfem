# What hIPPyMFEM carries for others, ready to send upstream

hIPPyMFEM builds hypre with three patches (`tools/hypre-3.2.0-*.patch`) and works around
four faults of the ROCm stack on Frontier.  Each belongs upstream.  This folder has the
patches rebased on hypre's `master` (checked on 8 November 2026 against `a1badec8`; each
one is a `git format-patch` file, applied with `git am`) and drafts of the reports.

## hypre: three pull requests

The patches were written against hypre 3.2.0 for the weak-scaling runs on Frontier (MI250X,
HIP) and measured there; the numbers come from
`results/scaling-frontier-2026-10-09/README.md` of the scaling work (two Newton-CG steps of
the model problem, quadratic hexahedra, operations charged at each size's own costs).  Before
a pull request: replace the `getenv` switches with hypre options (`HYPRE_Set...`), drop the
mentions of hIPPyMFEM from the comments, and build and test with CUDA as well; none of them
has been built with CUDA yet.

| patch | applies to hypre `master` | what it does | measured on Frontier |
|---|---|---|---|
| `hypre-master-staging.patch` | rebased: the context changed in `par_csr_matvec_device.c` (hypre PR #1627 moved the stream synchronization before a GPU-aware exchange); the two act on different cases | Where the halo exchange of a product is staged through CPU memory (no GPU-aware MPI), the page-locked staging buffers are kept from one exchange to the next instead of being allocated and freed each time, and the staged exchange starts before the local product, so that the messages travel while the device computes (`HYPRE_STAGE_EARLY=0` restores the order of before) | 2 to 5 points more weak-scaling efficiency, e.g. 76 % instead of 72 % from one GCD to 1024 with four 64³ blocks a GCD |
| `hypre-master-spgemm.patch` | cleanly | The sparse products of a BoomerAMG set-up on a device skip the sorting and copying of operands that are already sorted | every size 9.5 to 14 % faster; 0.7 s less of each set-up on one GCD and 1.1 to 1.4 s on more |
| `hypre-master-seqdevice.patch` | cleanly | `seq_threshold` with hypre on a device: the last level of a hierarchy is gathered and solved on a device (by the first rank, or with `redundant=1` by every rank, with the ranks' coarsenings drawing the same random numbers); hypre's own code for it reads device memory through host pointers and faults | the last levels of a linear-element hierarchy were a third to a half of a cycle on 512 MI250X; part of the configuration that took two Newton-CG steps with 120 billion unknowns on 4096 GCDs to 19.0 s |

Suggested titles: "Keep the staging buffers of host-staged halo exchanges and start them
before the local product"; "Skip sorting already-sorted operands in device SpGEMM";
"Support seq_threshold (gathered coarse level) in GPU builds of BoomerAMG".

## Reports for ROCm, JAX and OLCF (drafts)

**1. Kernel launches crash once the AQL queue fills, when the JAX ROCm plugin is loaded
(ROCm, rocprofiler-register; copy to JAX).**  JAX's ROCm plugin links
`librocprofiler-sdk.so.1`; finding it in the process, the HIP runtime routes every kernel
launch through the profiler layer even when nothing is profiled.  When the launch queue
fills (`ROC_AQL_QUEUE_SIZE`, 16384 packets by default; JAX queues launches without waiting),
the result is a segmentation fault in `libhsa-runtime64`, "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
The AQL packet is malformed", or a hang.  With `ROCPROFILER_REGISTER_ENABLED=0` set before
the runtime starts, a reproducer that failed 3 of 3 times passed 3 of 3 (MI210, ROCm 7.2.0,
JAX 0.11.1).  hIPPyMFEM sets the variable on ROCm.  An occasional segmentation fault of a
one-rank test inside JAX's device gather and scatter (about one run in fifteen, October 2026)
was seen before this variable was set and may be the same fault.

**2. XLA's command buffers segfault in `libamdhip64` (JAX/XLA, ROCm).**  On MI250X with ROCm
7.2.0, a jitted element kernel crashes inside `libamdhip64` once its batch passes a size that
depends on the block, so no chunk size is a safe cap.  `--xla_gpu_enable_command_buffer=`
avoids it; hIPPyMFEM sets it on ROCm.

**3. `jax-rocm7-plugin` 0.11.2 crashes at load against ROCm 7.2.0 (JAX).**  The plugin links
`libamd_smi`, whose symbols clash with `librocm_smi64`, which RCCL pulls in; 0.11.1 loads.
Frontier users of JAX meet it with the system's ROCm 7.2.0.

**4. ROCm 7.2.0's clang fails at `-O2` on three of hypre's SpGEMM kernels (AMD).**
`tools/build_pymfem_hip.sh` builds those three at `-Os`.

**For OLCF** (help@olcf.ornl.gov): 1 and 3 affect every JAX user on Frontier with ROCm 7.2.0;
a note in the JAX section of the user guide would spare others the search
(`ROCPROFILER_REGISTER_ENABLED=0` before the runtime starts, which for MPI programs means in
the job script, and JAX 0.11.1 with the ROCm 7 plugin).

# Why the code is the way it is: the measurements behind the design decisions

The API docstrings state what each routine does. This note holds the history that used
to live beside those statements: what was tried, what it measured, and why the code
chose as it did. Each entry names the code it explains. The numbers are from the
machines named; `PARALLEL_EFFICIENCY.md` has the end-to-end profiles and `REVIEW.md`
the 2026-09 review and its execution log.

## 1. Element kernels (`hippymfem/fem/kernel.py`)

**One differentiation primitive behind every Hessian block** (`GroupKernel._hess_slot`).
XLA rounds a forward pass differently for a different number of tangents: a pass over 35
of the 62 dofs of a P2/P1 hexahedron differed from the 62-tangent pass at 4e-16. So
`hess_block`, `hess_cols` and `hess_all` are all slices of one per-column-slot pass, and
selecting rows happens outside the compiled function. Against the old single pass over
every dof, a linearization point pushes 35 tangents instead of 62 (8 when the residual is
linear in the state) and the Jacobian 27.

**Chunking is a fallback, not a default** (`GroupKernel._chunked`). The element arrays are
mathematically independent, but XLA blocks each element's quadrature sum differently at
different batch shapes, so a chunked result differs from an unchunked one by about 1e-15
relative. A given chunk size is reproducible; the choice of size is not, since it depends
on what else is on the device. The whole batch is tried first; on an out-of-memory error
the chunk is sized from what the allocator asked for (a request eight times the budget
would otherwise need eight halvings, each a failed compile). The full element Hessian
(`hess_all`) measured 3.1-3.5x the single-block working set at 64^3 hexahedral P2 (a chunk
planned at 6.2 GB reached 19 GB), so it plans with `weight = nslots` and keeps its own
remembered chunk size.

**Planning the chunk from the shape** (`GroupKernel._plan`). Trying the whole batch and
shrinking on failure is ruinous when it does not fit: at 884 736 hexahedral P2 elements the
first attempt asked for 252 GB and the wreckage of that request made the retries fail too,
on a problem that ran with the chunk pinned. A probe measurement cannot help, because JAX's
`peak_bytes_in_use` is an all-time high-water mark with no reset. Forward mode carries
`nd_total` tangents through `nq` points, and the peak measured 7 to 13 doubles per
tangent-quadrature pair across P1/P2 on quadrilaterals, tetrahedra and hexahedra
(`AD_DOUBLES_PER_TANGENT_QP`, with margin). A third of what is free, not a half: the budget
is what is free when the first chunk is planned and XLA's peak exceeds the largest buffer,
so a half overshot at 1.3e6 elements (11.3 GB allowed, one buffer asked 12.1). Many small
chunks once looked like a cliff: on P2 hexahedra the cost per element was flat at 66 us up
to 20 chunks and 378 us beyond (5.6x at 100 chunks), put down to launch overheads. It does
not reproduce with the code of 2026-09-21: at 1.9e6 elements on one L40S a warm Jacobian
assembly takes 9.3 s in 128 chunks and 9.1 s in 16 (4.9 us an element), and a launch costs
about 5 ms. The warning reports the count and no longer advises against it.

**Streaming the geometry** (`GroupKernel.mapped(streaming=True)`). At 1.3e6 hexahedral P2
elements the per-element geometry `(ne, nq, dim, sdim)` is 8.2 GB, against 39 MB for a
chunk's share; the transfer is a few milliseconds beside a chunk that takes seconds, so
above `GEOMETRY_STREAM_FRACTION` of the budget the geometry stays on the host and the chunk
loop slices it.

**Stepping through the batch on the host** (`_mapped_kernel`, `HOST_BATCH`). On a CPU the
cost of an element grew with the batch: a P1-tetrahedron Jacobian took 0.5 us an element
at 3 072 elements and 1.73-1.83 us from 196 608 up, a third-derivative contraction 0.2 us
and 0.7 us, once the per-element intermediates of a whole-batch `vmap` no longer fit in
cache. So SOUPyMFEM's gap to a FEniCSx code grew with the mesh in 3D (1.7x at 4 913 dofs,
3.0x at 274 625). The host program now walks the batch in steps of `HOST_BATCH` elements
inside the compiled function (a loop over windows of the arguments, the last window
shifted back to end at the last element): 2.2x on the
Jacobian kernel and 3x on the third derivative at 196 608 elements, and SOUPyMFEM's
`bench_saa.py --dim 3 --n 32` on one pinned core went from 6.35 to 3.49 s (cost and
gradient), 9.24 to 6.48 s (Hessian) and 30.2 to 21.2 s (quadratic Taylor gradient), with
the same gradient norm to the last digit. Steps of 1 024, 2 048 and 4 096 are within 5 % of
each other, 256 and 512 no better; 2 048 is the default and `HIPPYMFEM_HOST_BATCH=0` maps
the whole batch. Steps of 512 elements and more gave the whole-batch element arrays bit
for bit (P1 and P2 tetrahedra and P1 hexahedra; Jacobians, residual vectors and third
derivatives); steps of 100 and of 7 elements differ by up to 6e-16 relative, since XLA
vectorizes a short step differently, the same effect as chunking (`test_kernels.py` checks
1e-15). `jax.lax.map(..., batch_size=)` does the same stepping but compiles the body a
second time for the remainder and copies the arguments to split it off: a first
Jacobian call at 10 368 elements took 0.45 s with it, 0.29 s with the window loop and
0.22 s whole, the warm calls the same. A GPU keeps the whole-batch map, which is what it
is fast at.

**The dof gather is compiled** (`kernel.gather_rows`). The gather in front of every kernel,
local dof vector to `(ne, nd)` element arrays, was eager JAX indexing, which normalizes and
bounds-checks the index array with separate array operations at every call: at 1.57e6
tetrahedra (64^3) 0.10 s a slot against 0.0036 s for the same gather compiled, and the
five-slot gathers of a residual vector took 0.53 s beside a 0.24 s kernel. A gather is a
copy and the sign flip exact, so the values are unchanged. A residual-vector assembly at
64^3 went from 1.11 to 0.35 s, at 32^3 from 0.068 to 0.032 s.

**Second directional derivatives in one pass** (`GroupKernel.third_dir_block`,
`PDEVariationalProblem.apply_third_dir`). A second-order adjoint over `k` directions needs
sums like `R_ijj[a, a] + 2 R_ijk[a, b] + R_ikk[b, b]` for every direction, which
`apply_ijk` assembles one block at a time: in SOUPyMFEM's Taylor gradient, 14 calls per
direction on the residual and 6 on the QoI. They are `D^2(d_i R)[t, t]` for `t = (a, b)`,
one nested forward derivative of the slot-`i` gradient along a direction spanning every
slot, and `apply_third_dir` takes all directions and their weights in one kernel call and
one scatter. It equals the sum of `apply_ijk` pairs to 3e-16 (`test_modeling.py`); the
quadratic Taylor gradient above went from 21.2 to 10.4 s. A facet density still goes pair
by pair.

**fp32 is a tool, not the solve path** (`PRECISION`, `_jaxconfig`). Measured on an L40S at
13 824 hexahedral P2 elements per rank: the element kernel is 4.1x faster in fp32, a full
assembly only 1.8x at one rank and 1.0x at four, because the scatter and the triple product
do not get faster; the assembled operator moves by 1.7e-4; and the forward solve's own
linearity check rejects it, one Newton step leaving a residual of 2e-3 where a linear
residual leaves 1e-14. XLA's default for an fp32 dot on an NVIDIA card is TF32 (10-bit
mantissa, about 5e-4 relative): the three chained einsums of the table evaluator, not the
AD pass, moved the operator by 1.7e-4; with `default_matmul_precision("highest")` that is
1.1e-7 at 24^3 on a Blackwell card, for 7 % of the kernel speedup. Double precision on an
L40S is 1.25 TFLOP/s against 125 TFLOP/s single, a factor of 100; on an A100/H100 about
half the single rate. `test_gpu.py` reports the numbers for the card present.

## 2. Direct CSR assembly (`hippymfem/fem/csrassemble.py`)

**Why the route exists.** MFEM's per-element callback costs a Python call per element,
3.0 microseconds, which was 51 % of a full assembly at 1e5 elements per rank. The direct
scatter does what `ParBilinearForm::ParallelAssemble` does, one level up, with one
vectorized reduction. At 2.1e6 dofs on four L40S the local matrix plus triple product costs
0.20 s against 26.4 s for the callback.

**Boolean prolongations, decided collectively** (`_boolean_local`, `_boolean_prolongation`).
For a space with a `DofTransformation` the answer is no before anything is looked at: MFEM
folds the face transformation of a shared face into `P` on the rank that does not own it.
Measured on H(curl) order 2 on tetrahedra at two ranks: 36 of 720 rows had two entries, 38
entries were not 1, and 18 rows with a single 1 had it in a different column than
`GetGlobalTDofNumber` names (the two dofs of a face, swapped); the true-dof route through
such a `P` gave a 37 % error. `Conforming()` is sufficient and needs no traversal, but not
necessary: a mesh carrying an `NCMesh` with no hanging nodes answers False while its `P`
is still a single 1.0 per row (measured on hex and quad meshes at orders 1 and 2), so the
entries are inspected before the fold is given up. The answer selects between code paths
with different collectives, so it is reduced with a logical AND before use.

**Private index arrays per matrix on a device** (`_as_sparse`, `TrueDofPattern.finish`).
MFEM's `CopyCSR` takes its shallow path whenever a device is configured; the `MakeAlias`
inside registers the base host pointer with MFEM's memory manager, which owns the device
mirror. Two live matrices built from the same host arrays share one registry entry, and
destroying either frees the mirror of both: an illegal address in the next BoomerAMG setup
the moment a solver first released an operator, masked until then by the solver keeping
every operator it was ever given. So on a device `I` and `J` are copied per matrix (0.5 GB,
of the order of 0.1 s at 64^3, against a 12 s block assembly) and `data` is not, being a
fresh device-to-host transfer per assembly already. The 9-argument copying constructor
shares nothing and survives a sibling's destruction; the aliasing constructor does not.

**Two constructors for the local matrix** (`local_par_matrix`). The raw-pointer
constructor is fastest and runs whenever hypre is on the host; the `mfem.SparseMatrix`
route stages into hypre's memory with `CopyCSR`, slower by a copy, and is the only one that
works with hypre's memory on a device, where a host pointer segfaults. The two were measured
bit-identical.

**The triple product, two ways** (`_triple`). `RAP` fused against two `ParMult`s: identical
output including the explicitly stored zeros the folded elimination writes (difference
exactly zero, nonzero counts equal); which is faster depends on rank count and size, so it
is timed once per space pair, the timings reduced with MAX so every rank compares the same
numbers.

## 3. The true-dof route (`hippymfem/fem/tdofassemble.py`)

**Why.** For a boolean `P`, `P^T A P` is a permutation with a sum over shared dofs, and
hypre was traversing every nonzero to do it. Re-targeting the pattern at true-dof numbering
once and exchanging the foreign rows with one `Alltoallv` measured 2.1-2.8x less than
forming the product; the prototype gave identical nonzero counts and matvec agreement to
1.7e-17 (32^3 P2, two and four ranks). On more than one rank it agrees with `RAP` to
round-off, not bit for bit, because the summation order at shared dofs differs.

**The pattern build.** The true-dof graph is one `np.unique` over row-major keys; its
inverse goes through a stable merge sort, the right one for keys that arrive grouped by row
(1.9 s against 13.6 s for an introsort at 1.3e8 keys). The slot order (block, row, leading
diagonal, column) is produced in linear passes rather than a four-key `lexsort`, which took
98 s of a 179 s pattern build at 6.8e8 entries on the H100 node.

**Block-major slots and the aliasing constructor** (`finish`, `_finish_block`). hypre keeps
a parallel matrix as two CSR blocks, so the slots are laid out block-major and both blocks
are contiguous views of one accumulator, which MFEM's block constructor aliases without a
copy: 1 ms to build against 44-91 ms for the copying constructor at 32^3, and on four L40S
1.45x faster forward solves at 64^3 and 1.47x at 128^3 with 2.15 GiB per rank less host
memory. The row-major arrays the copying constructor takes (`J_t` alone 1.09 GiB per rank
at 128^3) are built only where that constructor is the route in force and rebuilt from the
blocks otherwise (E3 of the review).

**Refilling a matrix in place does not work here.** `EliminateRowsCols` rebuilds the
off-diagonal block and the column map, so values refilled into the original structure no
longer match it; the block constructor already took the win a refill was after.

**The eliminated diagonal is written before the matrix exists** (`diagonal_slots`). Writing
it into the built matrix cost, with hypre on a device, a host/device round trip of the whole
matrix (0.28 s per assembly at 128^3 on four L40S) to set a few thousand values. Written
into the accumulator, the forward solve went from 6.50 to 6.22 s at 128^3 and from 0.98 to
0.91 s at 64^3, same cost functional, same CG counts (E1 of the review).

**Collectives never sit behind a rank-local condition.** `finish` used to skip its
`Alltoallv` when a rank had nothing to send or receive; on a small boundary pattern at
four ranks one rank owned no boundary element, skipped, and the other three waited the full
40-minute timeout. The copying and the diag/offd constructors communicate differently too,
so the choice between them is an allreduce, not a rank's own `n_offd`.

## 4. Solvers (`hippymfem/algorithms/linSolvers.py`)

**Release before rebuild** (`_SolverBase.release`). Without it a Newton step briefly held
two of everything: at 128^3 on four 46 GB cards the pieces of a step fit in 34 GB and the
step itself ran out of device memory at 40 GB. `set_operator` used to `keep` every operator,
preconditioner and MFEM solver it was ever given (append-only), which leaked one BoomerAMG
hierarchy per line-search step: invisible in host RAM, out of memory on a 46 GB card at
2.1e6 dofs on the second Newton step.

**Sharing a hierarchy between the forward and the forward-incremental solver**
(`set_operator(A, pc=...)`). Both hold the same Jacobian and each built its own BoomerAMG
hierarchy: two setups per Newton step and about 6 GB twice at 128^3 on a 46 GB card.

**Chebyshev smoothing (`amg_relax_type=16`) is not the default.** 12 % faster per CG+AMG
solve on L40S cards at 2.1e6 dofs, one and four cards alike, with half the CG iterations,
for SPD operators only. Turned on for the prior's solvers it cost the Laplace approximation
at 64^3 on four L40S 2 % on the posterior samples (3.98 to 4.06 s) and 10 % on the randomized
pointwise variance (7.53 to 8.27 s), the eigensolver being flat: the setup it adds is paid
once per operator and the iterations it saves are already few.

**Reusing a hierarchy across operators (`pc_reuse`)** measured a loss on this problem class.

## 5. The PDE problem (`hippymfem/modeling/PDEVariationalProblem.py`)

**`symmetric_jacobian="auto"`** probes every newly assembled Jacobian with four
matrix-vector products comparing `w^T A v` with `v^T A w`, about a hundredth of the AMG
setup they can save; the first version compared `A v` with `A^T v`, but hypre's device
`MultTranspose` forms the transposed matrix on every call and at 128^3 on two L40S that copy
ran the cards out of memory.

**Releasing the linearization point before a trial forward solve** exists because the old
point's blocks were measured to be freed only after the next point's blocks existed; the
release is opt-in (`release_linearization_on_move`) since it is a claim about the caller.

**One shared differentiation pass at a linearization point** measured 1.96x less kernel
time than one pass per block.

## 6. Device configuration (`hippymfem/_jaxconfig.py`)

**The memory fraction follows the ranks per card and whether hypre shares it**
(`_mem_fraction`). A fixed 20 % cap gave JAX 8.9 GB of an L40S's 45, so a P2 hexahedral
assembly fell back to chunking at 32 768 elements per rank with four fifths of the device
idle. With hypre on the card, JAX taking 90 % is fatal: at 64^3 the allocator had reserved
39 GB of 45 while holding 1.9 GB live, and hypre's hierarchy needs 0.7 GB at that size and
grows with the problem. The two halves get half the card each; the knob is
`HIPPYMFEM_GPU_MEM_FRACTION`, and XLA fixes the fraction when JAX is imported, before the
mesh exists, which is why it cannot be size-aware.

**Pinning the visible device** (`_pin_visible_device`, `_node`). JAX opens a context on
every device it can see: a one-rank job on a four-card node held 439 MB on each idle card
against 30 GB of real use on the card in play, and at four ranks three stray contexts per
card. `CUDA_VISIBLE_DEVICES` with one entry by node-local rank fixes it (measured with hypre
on the device and CUDA-aware MPI at two ranks: the two assigned cards carried memory, the
other two stayed at 4 MB). The variable latches at the process's first CUDA call, which with
a CUDA-aware MPI is `MPI_Init`: 4 devices seen when it was set after MPI came up, 1 when set
before. So the node-local rank is read from the launcher's variables, never from MPI, and
the pin only lands if hippymfem is imported before mpi4py and MFEM; `tools/mpirun_pinned.sh`
sets it at the launcher and works whatever the order. `JAX_CUDA_VISIBLE_DEVICES` is wired
only into the legacy CUDA client and the `jax-cuda12-plugin` backend ignores it. A hang met
during this work was blamed on the pin and was not it: a collective behind a
rank-dependent `if` in `configure_device`.

**JAX is imported on first use.** `import hippymfem` used to import JAX eagerly through
the UFL-style helpers (`hm.inner`, ...): 1.6 s and an open backend for a script that never
writes a residual density; resolved lazily it is 1.1 s (B2 of the review).

## 7. The vector layer, and what a device-resident vector would buy (E16)

`ParVector` is a numpy buffer aliased by an `mfem.HypreParVector`; with MFEM on a device
the memory manager keeps a device mirror, every Python-level operation (`axpy`, `inner`,
`assign`, ...) pulls the values to the host (`HostReadWrite`) and the next hypre call
pushes them back. Krylov iterations inside MFEM never leave the card, which is why the
device numbers are good; what pays is the algebra *between* operator applies.

**Measured (64^3 P1, 274 625 dofs, one rank, one L40S, hypre on the card):** a hypre
matvec followed by one Python-level operation on its result costs 0.36 ms, of which the
operation itself is about 0.08 ms (the host figure) and the round trip the rest, about
0.25 ms for 2.2 MB each way; five operations after one matvec cost 0.66 ms, so the round
trip is paid once per excursion, not per operation. A Newton-CG iteration makes two such
excursions around two incremental solves of 0.18-0.24 s each: about 0.2 %. The
eigensolver's `MatMvMult` makes one per column against a Hessian action of the same
order. For the algorithms as written the gain is under 1 %; it would matter for a
Python-level Krylov loop with cheap operators, which nothing here is.

**The design, should it become worth it:** keep `ParVector`'s interface and route its
algebra through MFEM's device-capable `Vector` operations (`Add`, `Set`, `*=`,
`InnerProduct(comm, x, y)`, `Norml2`) instead of numpy, so a value that lives on the card
stays there until `.array` is asked for explicitly; `.array` remains the host escape
hatch and the synchronization point. The blocker the code documents (PyMFEM lacks the
allocating `HypreParVector` constructor for an arbitrary partition) concerns who owns the
buffer, not this: the aliasing constructor with the memory manager's mirror is what runs
today. The eigensolver's orthogonalization (E11) and the dof gather (E12) would then be
device-resident for free. Not planned while the measured share is what it is.

## 8. Phase-2 measurements of the review (2026-09-16)

- **E11, orthogonalization in `doublePassG` (k=50, p=20):** host, 4 ranks, 32^3 P1: at
  most 1.2 of 7.8 s (two `Borthogonalize` calls counted, one of them elsewhere); 64^3
  P1: at most 5.3 of 57 s. Device, four L40S, hypre on the card: 64^3 P1 at most 1.6 of
  14.4 s; 64^3 P2 (2.1e6 state dofs) at most 1.6 of 37.7 s. The stage is the operator
  applies (13.6 and 36.9 s). A blocked CGS2 would recover part of a few per cent.
- **E7, CG+AMG against GMRES+AMG on the SPD Poisson Jacobian, same iteration counts:**
  host 48^3-64^3 P1, warm solve 5-8 % less (0.162 against 0.175 s at 64^3, tolerance
  1e-12); device 64^3 P1 27 % less (0.178 against 0.244 s), 96^3 P1 30 % less (0.604
  against 0.859 s), 17-21 % less including the setup. Now `spd_jacobian=True` on the
  PDE problem; opt-in because the symmetry probe cannot tell definite from indefinite.
- **`mfem.SparseMatrix()` on the CUDA build:** the default constructor segfaults inside
  `SparseMatrix.__init__` in a process that has configured the device and done nothing
  else on it (one rank, a small `P`), and works once something has; the sized
  constructor `SparseMatrix(1)` works in both states. `linalg._scratch` uses it.

## 9. Phase 3-5 measurements of the review (2026-09-16)

- **E14, element dof maps from the table** (`elementbatch._element_dofs`): one call per
  element to `GetElementVDofs` against the space's element-to-dof table read in bulk
  (`mfem.intArray((table.GetI(), n))` wraps MFEM's arrays without owning them; vdofs by
  `DofToVDof`'s rule; signs decoded as before): 0.394 -> 0.014 s at 64 000 hexahedral P2
  elements, 0.078 -> 0.002 s for a vector H1 space, identical maps and signs for H1, ND and
  RT. The prior's element keys from the bulk vertex array with one vertex-list call per
  element: 0.39 -> 0.14 s at 32^3 hexahedra, 0.70 -> 0.27 s at 200^2 triangles, keys and
  samples identical.
- **E13, projection without callbacks** (`FunctionSpace.project`, `coordinates`): each
  element's reference nodes mapped in one transformation call, `fn` evaluated on the
  coordinate array: `coordinates` 1.58 -> 0.10 s and `project` 0.90 -> 0.09 s at 117 649 P2
  dofs on hexahedra; coordinates identical, the point-by-point path bit-identical, a
  vectorized `fn` at numpy's last bit (its transcendental kernels round differently from the
  scalar ones).
  Under a device-configured MFEM the node values are written through `GetDataArray()` after
  `host_readwrite`: a bare write fills the host copy while the device copy, valid once
  `GetTrueDofs` has run, is the one the next restriction reads. `coordinates()` reused one
  grid function for x, y and z and so returned y and z as x at two or more device ranks; the
  host and one rank were exact, and the geothermal records' anomaly-box statistics found it.
- **E15, the boundary residual in the domain block** (`ScatterPattern.slots_of`,
  `csrassemble.add_boundary_entries`): the slots of arbitrary (row, col) pairs come from one
  sort of the graph's keys per pattern and one `searchsorted` per call; the boundary entries
  land in the domain accumulator (through `tslot` on the true-dof route), the folded
  elimination's slots are zeroed again, and one matrix comes out. The streamed shared pass
  of a linearization point then runs with a boundary residual; forced by pinning the element
  chunk to 3, its blocks agree with the plain point to 1e-16 on the host and on a device
  (where the planned chunk size is remembered per group and has to be forgotten first). On a
  device the accumulator reaches the host as JAX's read-only copy: `pattern.host_writable`
  lifts the flag in one place for `data`, `fused_end`, `finish` and the boundary entries.
  numpy's `add.at` does not check the flag and an indexed assignment does, so the 128^3
  record died only on the ranks with essential dofs, an hour into a collective for the rest.
- **E4, the true-dof maps from `P`** (`tdofassemble._tdof_maps`): a boolean `P` has one entry
  per local row, whose column is the global true dof; read in bulk from `hypre_to_scipy(P)`
  the maps take 0.010 s against 0.174 s of `GetGlobalTDofNumber`/`GetLocalTDofNumber` calls
  at 545 971 dofs per rank (64^3 P2, four ranks), identical. Host only: the merged rows
  cannot be read with hypre's memory on a device, where the loop stays.
- **The QoI as a linear form** (`forward_uq.qoi.mass_functional`, `weighted_mean_qoi`): a
  weighted mean of the state needs `ell = M w`, one vector; forming `M` for it costs a
  state-space matrix and hypre's assembly temporaries, which at 128^3 (P2, 17 M dofs) was
  the allocation the geothermal QoI stage died on before and after the plan. The linear form
  of the grid-function coefficient `w_h` is the matrix action to round-off (both quadratures
  exact for a product of basis functions on affine elements). Card memory polled at 64^3 on
  four L40S: the QoI stage's peak 8037 -> 7611 MiB, results identical.
- **A wrong-sized vector into hypre** (`PDEVariationalProblem.apply_ij`): hypre's `Mult` does
  not check sizes; an output vector of the wrong space let it write past the buffer, and the
  damage surfaced as "corrupted size vs. prev_size" in JAX's teardown, minutes later and in
  another library. `apply_ij` checks its vectors against the block and raises.


# msbg-rs Roadmap

A dependency-ordered tree for reaching feature parity with the C++ MSBG library. Steps are ordered so that each one builds only on earlier steps.

---

## Step 1 — Block & BlockPool

**Status:** DONE

**Parity target:** `src/blockpool.h` / `src/blockpool.cpp` — `Block<T>` (aligned
data + header + SIMD-safe pad), `BlockPool` (monotonic `allocBlock`/`reset`,
segment `extend_`, empty/full dummy blocks).

**Scope:** a single aligned block type and a growable monotonic allocator that
hands out blocks across thread-safe segments.

**Acceptance:**
- Block layout/alignment unit tests (SIMD width, cache line).
- `blockpool_hot_path` / `blockpool_contention` / `blockpool_cold_alloc` benches
  vs the C++ `BlockPool` (scenarios A/B/C in `MSBG/benchmark.cpp`).
- Current result: hot path ~2× faster than C++ (relaxed atomic + zero header
  writes); contention leg uses the same `blocks_per_seg = 4096` as C++.
- Cross-platform: benches are sized by `Machine { Dell, Macbook }` ×
  `Size { Small, Big, XBig }` (`MSBG_BENCH_MACHINE` / `MSBG_BENCH_SCALE`);
  macOS runs are Rust-only (no C++ baseline) — see refactor.md §10.

---

## Step 2 — Single-level SparseGrid

**Status:** DONE

**Parity target:** `src/sbg.h` / `src/sbg.cpp` level-0 — `SparseGrid<T>` with
`blockmap`, `isValueBlock`/`isEmptyBlock`, empty/full value semantics,
`getBlockDataPtr(bid, doAlloc, doZero)`, lazy `prepareDataAccess`,
`getBlockIndex`/`getBlockCoords`.

**Scope:** a single-resolution sparse grid over blocks: coordinate ↔ block index
mapping, a block map of `Option<Block>`, empty/full dummy blocks, and lazily
allocated blocks whose unwritten voxels read back as `empty_value`.

**Acceptance:**
- Voxel get/set, lazy-alloc initialization, and empty/full block semantics tests
  (including the "unwritten voxel in a partial block returns `empty_value`" case).
- Block iteration over allocated blocks (the `BlockIterator`).

---

## Step 3 — Typed channels & density dequantization

**Status:** DONE (core set — typed table + dequant + the channels steps 4–6
need; remaining channels re-tagged to steps 7–9 in `channel.rs`)

**Parity target:** MSBG's channel enum (`src/msbg.h`, `CH_FLOAT_1`…`CH_HEAT_DIFF`)
and `renderDensToFloat_` / `renderDensFromFloat_` (density ↔ quantized storage).

**Scope:** replace magic `int` channel ids with typed channels: a per-channel
newtype carrying its element type, a `ChannelId` enum plus `ChannelRef`/
`ChannelRefMut` for type-erased iteration, a `ChannelTable` that owns one
`SparseGrid<T>` per channel with safe typed access, and concrete element types:
`Density` (quantized `u16`), `Density8` (`u8`), `Velocity` (`Vec3f32`),
`Pressure` (`f32`), `CellFlags` (`u16`). Density dequantization is an explicit
typed `Density::to_f32` / `from_f32` (+ sqrt/stochastic + SIMD batch).

**Acceptance:**
- Typed insert/retrieve tests on `ChannelTable` (wrong-type access fails to
  compile or is safely rejected).
- Dequantize round-trip and range tests.
- Benchmark: channel lookup overhead is negligible vs raw `SparseGrid` access
  (lookup happens once per block sweep, not per voxel). *(deferred — not
  measurable until channels are wired into a sweep in steps 4/6; only the
  dequant/quant batches are benched today in `density_bench`.)*

**Note:** the step-4 sampler is generic over `SparseGrid<T>` via `InterpElem`
(`Density`, `Density8`, `Pressure`, …), but it does not go through the
`ChannelTable`/`ChannelId` enums yet — that indirection lands with the multires
work (step 7). `laplacian`/`halo`/`BlockIterator` still operate on raw
`SparseGrid<f32>`/`Block<u16>`.

---

## Step 4 — Field sampling & interpolation

**Status:** DONE

**Parity target:** `src/msbg.cpp` `interpolateScalarFast2` / `interpolate`, and
`src/sbg.h` `interpolateFloatFast` / `interpolateLinearWithGradient` /
`interpolateUint16ToFloatFast` / `interpolateWithDerivs`
(`IP_BSPLINE_CUBIC`) / `interpolateWithSecondDerivs`.

**Scope:** a `Sample` / `SampleVec3` trait (+ stateful `Sampler` /
`SamplerVec3` wrappers) over `SparseGrid<T>`, generic over element type via
`InterpElem` (dequant). Trilinear and cubic B-spline value + analytic gradient,
cubic Hessian, and a Vec3 sampler. `BoundaryCondition { Clamp, Neumann,
Dirichlet }` and `GridAlignment { Corner, CellCentered }` replace C++'s
`OPT_IPBC_*`/`OPT_IPCORNER` flags; `const IP: Interpolation` dispatch.

**Acceptance:**
- Property tests against analytic fields (affine reproduced exactly by both
  methods; gradient/hessian cross-checked against finite differences) plus a
  boundary/awkward/weird test matrix (block-boundary straddle, partial last
  block, single-voxel grid, dummy blocks, Dirichlet/Neumann/Clamp, u16 dequant).
- Tolerance-based difftest vs the C++ `interpolate*` path
  (`tests/difftest_interp.rs` + `../MSBG/interptest.cpp`): value/grad/Hessian
  match within 1e-4 (not bit-exact — Rust uses f32 + FMA, C++ uses `double`).
- Benchmark (`benches/interp_bench.rs` + `../MSBG` scenario G): linear value at
  parity, cubic value+grad ~20% faster, cubic Hessian parity-to-faster, linear
  value+grad ~10% slower (see refactor.md §7 for rationale).

**Note:** fine-coarse (`CH_DIST_FINECOARSE`) level blending is deferred to step
7; `interpolate`'s multires alpha-blend builds on this single-level sampler.

---

## Step 5 — Halo / ghost gather

**Status:** DONE (single-level; fine-coarse multires deferred to step 7)

**Parity target:** `src/halo.cpp` `HaloBlockSet::fillHaloBlock_` (single-level;
the `sgLo` multires path lands in step 7).

**Scope:** a concrete `f32` halo buffer (`BSX + 2` cube), `fill::<FULL>` generic
over the source element type via `Dequant<f32>` (u16/u8 density dequantized on
gather), pre-resolving the 3×3×3 block neighborhood to 27 raw pointers, a
contiguous middle copy (`memcpy` for `f32`), and `BoundaryCondition` reuse
(Neumann/Clamp/Dirichlet via `math::boundary`).

**Acceptance:**
- Boundary/awkward test matrix (domain corner/edge, partial last block, single
  block, empty/full dummies, Neumann vs Dirichlet, full-vs-faces consistency,
  u16 dequant) plus the 7-pt Laplacian happy path.
- `halo_gather` bench vs C++ `fillHaloBlock_` (scenario D, both legs): Rust
  ~1.8× faster on full fill, ~1.7× on faces-only (~4.5–5.0 vs ~2.4–3.0
  Gvoxels/s).
- Note: fine-coarse (multires) neighbor handling deferred to step 7.

---

## Step 6 — SIMD stencil kernels

**Status:** DONE (Laplacian, mean-curvature, bi-Laplacian; downsample/upsample deferred to step 7)

**Parity target:** `src/halo.h` `procLineSegmentSIMD`, `src/msbg4.cpp`
`smoothBlockSIMD4f`, `src/kernels_ispc.h` (`ispc_meancurv_smooth_halo_block`).

**Scope:** the compute kernels that run over a halo buffer: 7-point Laplacian,
mean-curvature (19-tap Hessian), bi-Laplacian (25-tap). Kernels are written once,
generic over `const W: usize` lanes (`Simd<f32, W>`), and instantiated at the
native register width per machine (`math::simd::LANES`: 16/8/4 for
AVX-512/AVX2/NEON). Fluid masking is decoupled into a precomputed
`MaskBlock<W, CHUNKS>` (built once from `CellFlags`, reused across iterations).
`HaloBlock::fill::<HALO, FULL>` was generalized to a `HALO`-wide halo
(1 for Laplacian/mean-curvature, 2 for bi-Laplacian). Kernels are boundary-blind
(raw-pointer halo loads; `debug_assert` proves the layout) and mirror the C++
full updates (`laplTyp==1`: `c + dt·L·G`; `laplTyp==4`: `f0 + clamp(dt·H, ±0.1)`;
`laplTyp==2`: `f0 + clamp(-dt·H·G, ±0.1)`).

**Downsample/upsample moved to step 7** (multires-coupled: they read
`sgDataSrcHi` at `level-1` and use the refinement map — see `msbg4.cpp`
`downsampleFloatChannelNew`/`downsampleChannel`).

**Acceptance:**
- Unit tests: each kernel matches a scalar reference to within tolerance, plus
  analytic/boundary cases (sphere curvature `2/r`, planar/constant fields,
  `gradMagSq` guard, clamp saturation, `CELL_IS_FLUID_` mask semantics).
- `difftest_stencil.rs` + `../MSBG/meancurvtest.cpp`: all three kernels match the
  real C++ `applyChannelPdeFast` (one Jacobi iteration) within 1e-4.
- Benches: `laplacian_compute_only` (real halo, kernel only — no mock),
  `mean_curvature_compute_only`, `mean_curvature_shell_sweep`, and the existing
  `laplacian_smoothing_e2e`, vs C++ scenario E (`applyChannelPdeFast`, laplTyp 1
  and 4).
- Current result (Dell 5500U, small scale, 12 threads): mean-curvature e2e
  ~1.94–2.06 vs C++ 1.65 Gvox/s (~+20%); Laplacian e2e ~2.17–2.35 vs C++
  1.90–2.02 Gvox/s (~+10%); kernel-only (parallel) ~3.3–3.5 Gvox/s. The win is
  the **non-temporal output store** — C++ `renderDensFromFloat_storeSimd8` uses
  `vstream` (`_mm256_stream_ps`, no write-allocate RFO); the Rust `store_nt`
  matches it, halving output memory traffic and lifting the parallel kernel off
  the ~15 GB/s memory wall.

---

## Step 7 — Multires hierarchy

**Status:** DONE

**Parity target:** `src/msbg.cpp` `MultiresSparseGrid::create`,
`setRefinementMap` / `regularizeRefinementMap`, `BlockInfo`.

**Scope:** resolution levels, per-level typed channels + directional face
channels, per-block `BlockInfo` (level + flags), the refinement map and its
regularization, coarse-level ghost sampling (finishes the `fillHaloBlock_`
parity from Step 5), downsampling, and Morton block-list ordering.

**Channels added here:** per-level `LevelData` (density f32 / cell flags /
distFineCoarse / face_area×3 / face_coeff×3) behind a closed `Level` enum over
the supported block sizes.

**Acceptance:**
- Refinement-map correctness: legal transitions, `bi->level` semantics
  (`regularize_refinement_map` + `compute_block_topology` unit tests, incl. the
  in-place aliasing case and a >1 level jump).
- "Coarse sample == fine downsample" consistency (`downsample_channel_avg` +
  `sample_coarse` tests).
- Benchmark: multires halo fill with a coarse neighbor vs single-level
  (`benches/multires_bench.rs` vs `../MSBG/benchmark.cpp multires`). Rust
  `fill_multires` ~1.8–2.3× faster than C++ `fillHaloBlock_` +
  `OPT_BC_COARSE_LEVEL` (coarse pointers pre-resolved once per fill instead of
  C++'s per-voxel `getValuePtr`); full `set_refinement_map` ~1.1–1.3× faster
  than C++ `regularizeRefinementMap`+`setRefinementMap` *while also* doing the
  cell-flag init the C++ path skips (see refactor.md §11).

---

## Step 8 — PDE smoothing (happy path: 8-color in-place sweeps)

**Status:** HAVEN'T STARTED

**Parity target:** `src/msbg4.cpp` `applyChannelPdeFast` (laplTyp 1 Laplacian,
laplTyp 4 mean-curvature) — the *live* path; `src/msbg_demo.cpp:716` drives it
with `-(PDE_MEAN_CURVATURE + OPT_8_COLOR_SCHEME)`.

**Scope:** a generic **8-color in-place block sweep** (read halo + run a
per-block stencil + write back), then the mean-curvature (19-tap Hessian) and
Laplacian (7-pt) kernels on top of it. The sweep is use-case-independent — the
same primitive the paper (phase-field mean-curvature), the bunny demo, and a
future pressure matvec all share — so the core library stays solver-agnostic.

**Channels added here:** `CH_FLOAT_2`/`CH_FLOAT_3` (smoother scratch src/dst),
the stochastic-quantize batch, and `prepareDataAccess` / `resetChannel` /
`protectChannel` semantics.

**Acceptance:**
- `laplacian_smoothing_e2e` bench vs C++ `applyChannelPdeFast` (scenario E,
  laplTyp 1 and 4) — the bench already exists as a manual halo+stencil
  composition; this step wires it to the real solver.
- Convergence tests on known Laplacian/mean-curvature fields.

**Deferred (later extension, not parity): the multigrid pressure solver.**
`multiplyLaplacianMatrixOpt` / `relax` / dense coarse levels / `AXPBY*` /
`dotProdChannel` have **no call sites** in the C++ demo — they are the
pressure-projection (Poisson) machinery for *standard* FLIP, baked into the
library but unused by the phase-field demo. When ported, design them Rust-native
(see `ideas.md` §2), not as C++ parity. Their channels (`CH_FLOAT_4..8`,
`CH_FLOAT_TMP_3`, `CH_DIVERGENCE_ADJ`, `CH_VEC3_2/3/4`, `velocityAirDiff`,
`sootDiff`, `heatDiff`, `pressureOld`, `genUint16*`/`uint8*`) land then too.
Redistancing (eikonal/FIM) is likewise written new — referenced in MSBG but not
implemented there.

---

## Step 9 — Surface reconstruction pipeline

**Status:** HAVEN'T STARTED

**Parity target:** `src/msbg_demo.cpp` `msbg_test_sparse` (PLY load, 8-color
lock-free particle splatting, finalize, active-block determination).

**Scope:** read particles, splat them into a density channel, determine active
blocks, and run the Step 8 smoother — the "bunny" pipeline without rendering.

**Channels added here:** `FaceDensity` (Vec3u16) render channel, particle/uint8
tmp channels, and the SIMD batch for `Density8` (8-bit build path).

Deferred (no step owns them yet): `PSFloat`/`double` pressure (opt-in build),
and a shared byte-arena pool for lower peak DRAM (a separate `BlockPool`
redesign).

**Acceptance:**
- A `.ply` produces a density field with the expected block occupancy.
- Benchmark: splatting and smoothing throughput on the bunny at scale.

---

## Step 10 — Rendering & end-user test

**Status:** HAVEN'T STARTED

**Parity target:** `src/render.cpp` (raymarch), `src/bitmap*` / `readpng.c`
(PNG output), `src/visualizeSlices.cpp` (2D slices).

**Scope:** offline rendering (marching cubes or raymarch) + image output,
implemented with Rust crates and driven **only through the library's public
API** — the point is to surface end-user design flaws, not to port the C++
renderer (its UI/panel code is kinda bad).

**Acceptance:**
- Render the bunny-of-bunnies from a Step 9 density field.
- The render consumes only the `Sample`/iteration public API (no crate-internal
  access), confirming the API is usable.

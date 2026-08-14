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

**Status:** HAVEN'T STARTED

**Parity target:** MSBG's channel enum (`src/msbg.h`, `CH_FLOAT_1`…`CH_CELL_FLAGS`)
and `renderDensToFloat_` / `renderDensFromFloat_` (density ↔ quantized storage).

**Scope:** replace magic `int` channel ids with typed channels. A `Channel<T>`
identifier carrying its element type, a `ChannelTable` that owns one
`SparseGrid<T>` per channel with safe typed access, and concrete element types:
`Density` (quantized `u16`), `Velocity` (`Vec3f32`), `Pressure` (`f32`),
`CellFlags` (`u16` bitfield). Density dequantization is an explicit typed
`Density::to_f32` / `from_f32`.

**Acceptance:**
- Typed insert/retrieve tests on `ChannelTable` (wrong-type access fails to
  compile or is safely rejected).
- Dequantize round-trip and range tests.
- Benchmark: channel lookup overhead is negligible vs raw `SparseGrid` access
  (lookup happens once per block sweep, not per voxel).

---

## Step 4 — Field sampling & interpolation

**Status:** KINDA (trilinear only)

**Parity target:** `src/msbg.cpp` `interpolateScalarFast2` / `interpolate`, and
`src/sbg.h` `interpolateWithDerivs` with `IP_BSPLINE_CUBIC`.

**Scope:** a `Sample`/`Field3` trait exposing `sample(pos) -> f32` and
`gradient(pos) -> Vec3`, implemented over a channel. Trilinear interpolation,
then cubic B-spline interpolation with derivatives (analytic gradient).

**Acceptance:**
- Property tests against analytic fields (e.g. `f = x²+y²+z²` gives
  `∇f = 2·(x,y,z)`).
- Benchmark: interpolation throughput vs the C++ `interpolate*` path, using only
  a Step 2/3 grid (no halo, no solver).

---

## Step 5 — Halo / ghost gather

**Status:** ALMOST DONE (simplified; full parity pending)

**Parity target:** `src/halo.h` / `src/halo.cpp` `HaloBlockSet::fillHaloBlock_`.

**Scope:** the 18³ halo gather — copy a block plus its six face neighbors into a
padded halo buffer with boundary handling, backed by a per-thread halo pool.

**Acceptance:**
- `halo_gather` bench vs C++ `fillHaloBlock_` (scenario D in `MSBG/benchmark.cpp`).
- Correctness: boundary/Neumann handling tests on edge blocks.
- Note: the current Rust `fill` is a simplified copy (no multires/coarse
  handling); full `fillHaloBlock_` parity is finalized in Step 7.

---

## Step 6 — SIMD stencil kernels

**Status:** A LITTLE (7-pt Laplacian only)

**Parity target:** `src/halo.h` `procLineSegmentSIMD`, `src/msbg4.cpp`
`smoothBlockSIMD4f`, `src/kernels_ispc.h` (`ispc_meancurv_smooth_halo_block`).

**Scope:** the compute kernels that run over a halo buffer: 7-point Laplacian
(the fluid mask is currently applied inline in the kernel, not split out),
mean-curvature (19-tap Hessian), bi-Laplacian, and downsample/upsample.
Introduce the cross-platform SIMD dispatch (widest native width per machine)
here.

**Acceptance:**
- Unit tests: each kernel matches a scalar reference to within tolerance.
- `laplacian_compute_only` bench (already exists) for the 7-pt; add a
  mean-curvature compute-only leg.

---

## Step 7 — Multires hierarchy

**Status:** HAVEN'T STARTED

**Parity target:** `src/msbg.cpp` `MultiresSparseGrid::create`,
`setRefinementMap` / `regularizeRefinementMap`, `BlockInfo`.

**Scope:** resolution levels, a `ChannelTable` per level, per-block `BlockInfo`
(level + flags), the refinement map and its regularization, and coarse-level
ghost sampling (finishes the `fillHaloBlock_` parity from Step 5).

**Acceptance:**
- Refinement-map correctness: legal transitions, `bi->level` semantics.
- "Coarse sample == fine downsample" consistency checks.
- Benchmark: multires halo fill with a coarse neighbor vs single-level.

---

## Step 8 — PDE solvers

**Status:** HAVEN'T STARTED

**Parity target:** `src/msbg4.cpp` `applyLaplacianSmoothing` /
`applyChannelPdeFast` (laplTyp 1 and 4), `src/msbg3.cpp`
`multiplyLaplacianMatrixOpt` / `relax`; plus eikonal/FIM redistancing (written
new — referenced in MSBG but not implemented there).

**Scope:** mean-curvature/Laplacian smoothing, redistancing, and the multigrid
Laplacian (matrix-vector product + relaxation + CG) used by pressure projection.

**Acceptance:**
- Convergence tests on known problems (e.g. Laplace's equation, a known eikonal
  distance field).
- `laplacian_smoothing_e2e` bench vs C++ `applyChannelPdeFast` (scenario E) —
  the bench already exists as a manual halo+stencil composition; this step wires
  it to the real solver.

---

## Step 9 — Surface reconstruction pipeline

**Status:** HAVEN'T STARTED

**Parity target:** `src/msbg_demo.cpp` `msbg_test_sparse` (PLY load, 8-color
lock-free particle splatting, finalize, active-block determination).

**Scope:** read particles, splat them into a density channel, determine active
blocks, and run the Step 8 smoother — the "bunny" pipeline without rendering.

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

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

**Status:** DONE

**Parity target:** `src/msbg4.cpp` `applyChannelPdeFast` (laplTyp 1 Laplacian,
laplTyp 4 mean-curvature, laplTyp 2 bi-Laplacian) — the *live* 8-color
in-place path; `src/msbg_demo.cpp:716` drives it with
`-(PDE_MEAN_CURVATURE + OPT_8_COLOR_SCHEME)`.

**Scope:** a generic **8-color in-place block sweep** (`src/solver.rs`) — read
halo → run a per-block stencil → store back into the same block — with the
mean-curvature (19-tap), Laplacian (7-pt) and bi-Laplacian (25-tap) kernels on
top. Generic over storage element type via `StoreBack` (`f32` NT store, `u16`
quantize, `u8` sqrt+stochastic-rounding quantize). The sweep is
use-case-independent; the core stays solver-agnostic.

**Decisions (owned):** generic `f32`/`u16`/`u8` storage from day one; **no
fluid mask** (match C++ — smooth every voxel + `[0,1]` clamp); **no scratch
channels** (`CH_FLOAT_2`/`CH_FLOAT_3` dropped — only the dead Jacobi
double-buffer path used them); `Stencil` enum + `PdeParams` replace the
19-arg `int laplTyp` API; kernels refactored to a raw `*mut D` output;
pre-bucketed color lists + Morton sort; `sfence` per block (benchmarked vs
`mfence`/none — all within ~1%, `sfence` marginally best).

**Acceptance:**
- `tests/difftest_smoother.rs` vs the C++ `smoothertest.cpp` (8-color, 4 iters,
  full level-0 field): Laplacian within 1e-4, mean-curvature within 1e-3 (the
  looser bound is the deliberate `0.25*(a-b-c+d)` mixed-partial factoring +
  the `gradMagSq > 1e-7` guard cliff — see refactor.md §8).
- `laplacian_smoothing_e2e` bench (scenario E, both legs) wired to the real
  `Sweeper`; C++ `benchmark.cpp` scenario E switched to the 8-color flag so both
  sides run the live path. Small scale (Dell 5500U, 12 threads): Rust ~2.10 /
  1.73 / 1.72 Gvox/s (Laplacian 1k/5k/10k) vs C++ 1.79 / 1.48 / 1.52;
  mean-curvature Rust 1.85 / 1.48 / 1.52 vs C++ 1.40 / 1.34 / 1.35.
  **Rust ~+13–32%.** `perf`: 90% of cycles in the sweep hot loop.
- Unit tests: color-schedule == serial Gauss-Seidel reference (bit-level),
  affine fixed point, clamp, dummy/gap/empty active lists, odd iterations,
  `StoreBack` RNG range.

**Deferred (no call sites in the demo, designed Rust-native when ported):**
- the **Jacobi double-buffer path** (`chSrc != chDst`, `CH_FLOAT_2`/`CH_FLOAT_3`
  scratch) — `applyChannelPdeFast` with a *positive* `laplTyp`;
- `prepareDataAccess` / `resetChannel` / `protectChannel` semantics — Rust's
  lazy `Option<BlockPtr>` blockmap subsumes them for the live path;
- the **multigrid pressure solver** (`multiplyLaplacianMatrixOpt`, `relax`,
  `relaxBlockList`, dense coarse MG levels, `AXPBY*`, `dotProdChannel`) —
  the pressure-projection machinery for *standard* FLIP, unused by the
  phase-field demo. Its channels (`CH_FLOAT_4..8`, `CH_VEC3_*`, `pressureOld`,
  etc.) land with it.
- Redistancing (eikonal/FIM) is likewise written new — referenced in MSBG but
  not implemented there.

---

## Step 9 — Surface reconstruction pipeline

**Status:** DONE

**Parity target:** `src/msbg_demo.cpp` `msbg_test_sparse` (PLY load, 8-color
lock-free particle splatting, finalize, active-block determination).

**Scope:** read particles, splat them into a density channel, determine active
blocks, and run the Step 8 smoother — the "bunny" pipeline without rendering.
`src/io/ply.rs` (PLY load via `ply-rs`), `src/particles/{sort,splat,active,
finalize,mod}.rs`, and the `Quant` storage trait in `src/channel.rs`.

**Channels added here:** `FaceDensity` (Vec3u16) type + channel-table row +
`Dequant<Vec3>` (a multires phase-field channel; not used by the pipeline),
`quantize_density8`/`dequantize_density8` SIMD batches.

**Decisions (owned):**
- **Thread-local staging + SIMD commit instead of direct 8-color scatter.** The
  C++ path does a per-voxel `min`-RMW into the grid for every overlapping
  particle and becomes DRAM-latency bound at scale (measured ~17 min for 16.7M
  particles vs 0.2 s for the staged path). The Rust splat accumulates each
  block's particles into a thread-local `24³` (`MSX=24`) buffer in L1, then a
  SIMD `commit_chunk` writes each *touched* voxel to the grid exactly once per
  contributing block (8-coloring keeps it race-free, `rScan < BSX`).
- **SIMD8 staging** (`stage_chunk`): the splat's hot loop quantizes 8 voxels
  per lane group; MSX=24 keeps the buffer L1-resident (the MSX=32/SIMD16
  variant spilled to L2 and regressed).
- **No scratch channels** — the pipeline runs entirely on one `Density` (u16)
  channel; `applyChannelPdeFast` with `CH_NULL` scratch is in-place (matches
  the step-8 finding).
- **No multires** — `OPT_SINGLE_LEVEL` demo semantics map to a plain
  `SparseGrid<Density>` + the step-8 `Sweeper`; refinement maps are field-neutral.
- **Placement stays scalar** — see refactor.md §12 for the SIMD4 gap.

**Acceptance:**
- `tests/difftest_splat.rs` vs `../MSBG/splattest.cpp` (res2 bunny, 256³,
  single instance): block occupancy identical, full 16.7M-voxel field A (after
  finalize) and B (after 6 MC sweeps) within the budget (L-inf ≤ 2 ulps, ≤0.1%
  off by 1 ulp) — verified against the live C++ binary.
- Unit test matrix across `particles/*` (happy paths + boundary/awkward: block
  boundary-crossing splats, domain corners, overlap min, radius cull, quantize
  endpoints, degenerate clouds, dedup, asymmetric dims).
- Benchmark (`benches/surface_bench.rs` vs C++ `benchmark.cpp surface`, 512³ /
  64 instances / 523K particles, Dell 5500U): **splat 3.7× faster, finalize
  1.4×, parse 1.7×, e2e (load→place→splat→finalize→6 MC) 2.6× faster
  (156 vs 400 ms)**. Placement is 0.15× (scalar vs C++ SIMD4) — see refactor.md
  §12. The full bunny-of-bunnies (1024³, 1.29G particles) needs >20 GB RAM and
  is deferred (below).

**Deferred:**
- **A3 streaming generate-on-splat** (no materialized particle array; bucket
  `(instance, base-range)` per footprint block) and the memory-optimized
  in-place counting sort — fold into the "shared byte-arena pool" deferred item.
- **Placement SOA SIMD** (close the 6.9× scalar-vs-SIMD4 gap; see refactor.md
  §12).
- Full paper-scale bunny-of-bunnies (1024³, 1.29G particles) — needs >20 GB.
- `FaceDensity` downsampling / multires phase-field machinery (no demo call sites).

---

## Step 10 — Rendering & end-user test

**Status:** DONE

**Parity target:** `src/render.cpp` (`RaymarchRenderer`), `src/msbg.cpp`
`getSlices2D`, `src/bitmap*`/`readpng.c` (PNG output).

**Scope:** offline rendering (2D orthogonal slices + 3D isosurface raymarch)
+ PNG output, implemented as a **separate workspace crate `msbg-render`** that
depends only on `msbg-rs`'s `pub` surface (`SparseGrid`, `Sampler`, channel
types). Being outside the library crate, it *cannot* reach crate-internal items
— the step doubles as proof that the public API is usable end-to-end. The
C++ panel/UI glue (`visualizeSlices.cpp`, `panel.c`, `bitmap.c`) is not ported.

Layout: `msbg-render/{src/{camera,colormap,raymarch,render_elem,slice}.rs,
examples/render_bunny.rs, benches/render_bench.rs, tests/render_tests.rs}`;
C++ side gains `../MSBG/rendertest.cpp` + `build_rendertest.sh` (the real
`getSlices2D` + `RaymarchRenderer` on the real splatted field).

**Decisions (owned):**
- **O(N²) direct slicing.** The C++ `getSlices2D` scans every voxel (`sx·sy·sz`)
  and tests slice-plane membership; the Rust slicer iterates only the output
  pixels and samples once per pixel (Rayon over rows). `512³` → `512²` samples
  instead of `512³` scans.
- **ESS-DDA raymarcher.** C++ `RaymarchRenderer` marches every ray at a fixed
  step with a per-sample `isEmptyBlock` early-out that returns `0` *without
  advancing*. Rust instead walks the `16³` block lattice with an Amanatides–Woo
  DDA and skips an empty block in O(1) (jump to the ray's block-exit plane),
  fine-stepping only inside value blocks.
- **Unified linear sampling + linear gradient** (C++ splits trilinear iso
  detection vs cubic shading — a wart, not a feature). Samples are taken in
  *physical* density space via a `RenderElem` trait (`Density`/`f32` identity,
  `Density8` squared), so `Density8`'s sqrt compression is undone before the
  `iso = 0.5` comparison.
- **Native `Vec3` camera, no `glam`** — the C++ basis is a 3-line left-handed
  look-at (`right = cross(forward, up)`), reproduced exactly; `image` (png) is
  the only new dependency.
- **No scratch/float channels.** Rendering is read-only over the step-9
  `SparseGrid<Density>`; `Sampler` dequantizes on the fly.

**Acceptance:**
- `msbg-render/tests/render_tests.rs` — 10 tests: 2 happy paths (sphere slice /
  sphere raymarch) + boundary/awkward (slice out-of-bounds Dirichlet-vs-Clamp,
  grazing/coplanar ray, camera-inside-density, ESS≡micro-step equivalence,
  `Density8` sqrt round-trip, 512×16×16 ribbon, ray-misses-grid, all-empty
  grid).
- `examples/render_bunny.rs` reconstructs the bunny through the public API and
  writes PNG slices + a raymarch frame.
- Benchmarks (`benches/render_bench.rs` vs `../MSBG/rendertest.cpp`, real bunny,
  Dell 5500U):
  | workload | C++ | Rust | ratio |
  |---|---|---|---|
  | slice (256³ / 3 planes) | 34.9 Mpix/s | 231.8 Mpix/s | **~6.6×** |
  | slice (512³ / 3 planes) | 21.5 Mpix/s | 200.9 Mpix/s | **~9.3×** |
  | raymarch (256³, 640×480) | 2.77 Mray/s | 7.12 Mray/s | **~2.6×** |
  | raymarch (512³, 960×540) | 1.26 Mray/s | 9.52 Mray/s | **~7.6×** |
  The slice ratio grows with resolution (O(N³)→O(N²)); the raymarch ratio grows
  because a larger domain is proportionally emptier. ESS-on vs ESS-off (the
  fixed-step reference): **5.0× @256³, 18.5× @512³**. `perf`: 86.5% of raymarch
  cycles in `Sample::sample::<Linear>` — the remaining cost is the essential
  surface sampling, not traversal overhead (see refactor.md §13).

**Deferred:**
- SIMD ray packets (4–8 rays per `Simd` lane sharing one DDA traversal) — the
  natural next win now that ESS has removed the empty-space overhead.
- Marching-cubes mesh export (`.ply`/`.obj`) — a separate step.
- Volumetric emission/absorption rendering (the isosurface path is the parity
  target).

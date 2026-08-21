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

---

# Phase II — Paper replication (steps 11–22)

Two goals: **(1) the true bunny-of-bunnies** — testCase 2 (`bun_zipper.ply`,
35,947² = 1.29B particles) at 32,768³, ~100B active voxels, 256 GB — and
**(2) the Adaptive Phase-Field-FLIP 2-phase simulation**, ending in a
droplet-crown then a dam-break animation at paper scale.

Decisions locked: **sparse blockmap + block 16** (not the paper's block-32 dense
map); single-level FLIP first, then adaptive; single-phase first, then 2-phase;
**MAC staggered face-velocity** (the paper's representation — splat to faces, not
cell centers); **adaptive algebraic-aggregation multigrid + flexible PCG** (the
paper's *actual* pressure solver — NOT the geometric multigrid in the C++
library). Two facts re-orient steps 13–15 vs. the original draft:

- The C++ *pressure* machinery (`multiplyLaplacianMatrixOpt`, `relax`,
  `relaxBlockList`, `downsampleChannel`, `downsampleVelocity`,
  `downsampleFaceDensity`, `dotProdChannel`, `AXPBY*`, dense coarse MG levels)
  is **complete but dead** — zero call sites in `../MSBG` (see refactor.md §8).
  Its `_blocksRelax` list is never populated, `_mgSmType` is written but never
  read, and `_mgSmOmegaSched1 = 1.7319` is gated behind an undefined macro.
- The paper's real solver (its "3× fewer CG iterations" headline) is the
  adaptive scheme in §6, which shares only the *operators* (7-pt matvec/relax)
  with that dead code — the coarsening, smoothing, and outer loop are different.

So steps 13–15 verify their *operators* against the real-but-dead C++ library
(operator-level difftests), while the driver (step 18+) verifies against the
paper's *published* throughput (~10B unknowns/s) + physical laws, since the FLIP
driver loop and the scenarios have no C++ counterpart in the repo (their sim
driver is closed-source).

Paper source: PDF at `../MSBG/Adaptive Phase-Field-FLIP for Very Large Scale
Two-Phase Fluid Simulation - Adaptive_Phase_Field_FLIP_preprint.pdf`; its
extracted text (with the line numbers cited below) is at `/tmp/msbg_paper.txt`.
If that file is gone (it lives in `/tmp`), regenerate it with:

```bash
nix shell nixpkgs#poppler-utils --command pdftotext -layout \
  "../MSBG/Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation - Adaptive_Phase_Field_FLIP_preprint.pdf" \
  /tmp/msbg_paper.txt
```

---

## Step 11 — Sparse blockmap

**Status:** DONE

**Parity target:** C++ `_blockmap` (dense per-block `BlockInfo`, `msbg.cpp`) vs
our dense per-channel `Vec<Option<BlockPtr>>`.

**Scope:** replace the dense blockmap with a sparse, block-id-keyed `BlockMap`
(open-addressing) so map cost scales with *active* blocks, not *virtual* blocks.
At 32,768³/block-16 the dense map is 68.7 GB per grid — and ~1.85 TB across the
27 multires channel maps; the sparse map is ~0.4 GB. This is the "smarter than
them" architectural win that unblocks both goals. Coordinate indexing
(`get_block_id`/`get_voxel_id`) is unchanged; only the storage changes.

**Decisions (owned):** block 16 (L1-resident, our kernels stay valid) + sparse
map instead of block 32 + dense map; 64-bit block ids (`usize`; 2048³ = 8.59B >
`u32::MAX`); hand-rolled open-addressing over `hashbrown` (measured — see
below); the multires `LevelData` collapses its 9 maps to one shared sparse map
with a per-block SoA payload (see refactor.md §14).

**Acceptance:**
- All existing unit + difftests pass unchanged (map sits behind the existing API).
- `blockmap_lookup` micro-bench: sparse probe within ~2× of a dense-array index.
- **No-regression re-run** of every C++-backed bench (blockpool, halo, stencil,
  interp, splat, surface, multires, render) — step-1..10 ratios must hold.
- Memory: the multires scale-stress bench (`benches/multires_scale_bench.rs`)
  RSS drops by the map overhead factor at a given domain.

**Measured (Dell 5500U, small scale, 12 threads):**

- `blockmap_lookup` (`benches/blockmap_bench.rs`, real block pointers): dense
  `Vec` index **0.53 ns/probe** vs `BlockMap::get` **0.88 ns/probe** (1.66×,
  within the 2× budget); `hashbrown` 1.6–3.1× slower than hand-rolled → rejected.
  The first cut used a SplitMix64 finalizer (~3.5× over dense); a single
  odd-constant multiply (`key * 0x9E37…`, a bijection mod 2^k that spreads
  consecutive `bx` bids uniformly) is what closed the gap.
- **Interpolation no-regression:** `interp_linear_value` 4.87 ms vs the dense
  baseline 4.78 ms (+2%, inside the ±15% throttle band) — the gather fast path
  is one probe per sample and the probe is ~1 ns of a ~48 ns sample.
- **Multires set_refinement_map:** Rust 98.6 ms vs C++ 94.1 ms (35,937 blocks)
  — parity *while also* initializing the cell flags the C++ benchmark path
  (`doInitCellFlags=false`) skips. (A first SoA cut was 2.05× slower: the SoA
  block co-filled density+flags+dfc 32 KiB/block during flag-only init; see
  refactor.md §14.)
- **Multires halo gather:** Rust 3.72 Gvox/s vs C++ 2.10 Gvox/s (~1.77×).
- `surface` (step-9) and `render` (step-10) benches + difftests pass unchanged.

---

## Step 12 — Paper-scale reconstruction (true bunny-of-bunnies)

**Status:** DONE

**Parity target:** `msbg_demo.cpp` `msbg_test_sparse` testCase 2 (`bun_zipper.ply`,
35,947² = 1.29B particles) — the README's 32,768³ / ~100B-active-voxel run, with
the paper's "10 billion unknowns/s" mean-curvature headline.

**Scope:** dense-by-bid → dense-by-rank across the particle path (refactor.md
§15): `bucket_by_block` is a parallel sparse counting sort (`BlockMap<u32>`
histogram + dense-by-rank atomic scatter), the footprint active-set is a sparse
`BlockSet` union, and `Bucketed` uses compact `starts`. Removes the `counts`/
`block_start`/`cursor` dense arrays (206 GB @ 32,768³) and the footprint `Vec`
(~82 GB). Added `Machine::Aws` to the benches, a `bunny_of_bunnies` example that
reconstructs **and** renders (whole-bunny + close-up) in one process, and
parallel bulk block allocation (`SparseGrid::ensure_blocks_parallel` +
`fill_blocks_parallel`).

**Decisions (owned):** materialize the particle array (fits 256 GB; streaming
deferred, not needed). **Overturned: "keep block 16".** Block-32 is **1.64×
faster** than block-16 at the smoothing step (halo read-amplification `(B+2)³/B³`
is 1.42× vs 1.20×), matching the paper's choice for this demo; the correct block
size is kernel-dependent, not a constant. `-C target-cpu=native` must be pinned to
`znver3` on AWS `m6a` (its CPUID masks AVX-512, but LLVM's native detection
emitted `zmm` code → SIGILL).

**Acceptance:**
- `surface_bench` 1024³/testCase 1 vs C++ `benchmark.cpp surface` — real
  side-by-side (splat 3.4×, e2e 2.64× on Dell 512³/64).
- testCase 2 (1.29B) per-phase throughput + RSS, run on AWS.
- The 32,768³ run: 107.7B active voxels (block-32; 68.7B at block-16), ~223 GiB
  peak RSS, teaser frames via step 10 (`bunny_of_bunnies{,_close}.png`).
- **Beat:** `Sweeper` 12.9 G unknowns/s on the 32c/64t box (their headline is
  10 G/s on the same class) — 1.29×, *without* AVX-512; 16.5 G/s on 64-core
  Genoa with AVX-512.

**Measured:**

| machine | threads | AVX-512 | block | mean_curvature |
|---|---|---|---|---|
| paper (Threadripper) | 32c/64t | ? | 32 | 10.0 G unknowns/s |
| m6a (Zen 3) | 32c/64t | no | 16 | 7.85 G/s |
| m6a (Zen 3) | 32c/64t | no | 32 | **12.9 G/s** |
| Genoa (Zen 4) | 64c | yes | 16 | 16.5 G/s |

- **Bucket** (`m6a`, block-32): 375 s → 6.5 s (**58×**) after replacing the
  scheduler-dependent histogram (rayon `fold` thief-splitting made 2,685
  accumulators; see the post-mortem) with explicit one-map-per-thread + a
  pre-sized merge. Dell small-scale: place+bucket 70.4 → 20.0 ms (**3.5×**,
  also fixing the old counting sort's single-threaded scatter).
- **alloc_fill** (block-32, ~215 GB): 110.7 s → 16.3 s (**6.8×**) by
  parallelizing first-touch allocation + fill.
- Full pipeline (m6a, block-32, 32,768³): place 35.6 s, bucket 3.3 s, alloc_fill
  16.3 s, splat 22.6 s, finalize 3.7 s, mean_curvature 51.1 s, render 10.6 s.
- Every optimization held the live `difftest_splat` (full field vs the real C++
  binary) at ≤2 ulp / ≤0.1% — bit-equivalent throughout. Narrative in
  `docs/step_12_story.md`.

---

## Step 13 — MAC face-velocity transfer (P2G splat + G2P gather)

**Status:** DONE — `src/fluid/` (MacGrid, P2G splat, G2P sampler, divergence),
12 boundary/awkward tests, `benches/velocity_bench.rs` + `../MSBG/velocitytest.cpp`.
Benchmarks (5500U, res=128): P2G splat ~5.8 Mparts/s, staggered G2P ~3.6
Msamples/s, cell-centered Vec3 gather **10.2 Msamples/s vs C++
`interpolateVec3Float` 14.3** (1.4×, same result to FMA). The cell-centered
gather started 2.7× behind and was closed via SIMD trilinear + an unsafe
16-byte slop load (matching C++ `Vec4f::load`) + an unchecked `floor→i32`; the
remaining 1.4× is the generic gather's structure vs C++'s hand-specialized
interior-only path. Full narrative in `../stories/step_13_story.md`.

**Build order:** FIRST of the three pressure steps. The face mass accumulators
this step produces *are* the phase-field raw density (Eq. 7) *and* the Poisson
coefficient `β = 1/ρ` (Eq. 9) that steps 14/15 consume — so this step precedes
them.

**Paper:** §3.2–3.6 (`/tmp/msbg_paper.txt:373–526`). P2G splats mass `M_a` and
momentum `P_a` to **cell faces** (staggered MAC), not cell centers, with a
cubic-falloff kernel `w = max(1 − (‖x_c + e_a/2 − x_p‖/r_p)², 0)³` (Eq. 6,
line 414 — no sqrt, spherical support). Face velocity is `ũ* = P/M`. The face
mass *is* the raw phase-field density (Eq. 7, line 437) *and* the variable
coefficient `β = 1/ρ` (Eq. 9, line 472). G2P is `u_new = α_FLIP·u_old + I(Δu) +
(1−α_FLIP)·I(u)` (Eq. 12, line 524), `Δu = u − ũ*` (Eq. 11).

**Parity target:** none in the demo (it splats *density* only, as a min-SDF to
cell centers). This is a **new** primitive — a *sum*-reduction `Vec3` splat to
faces — so there is no C++ velocity-splat counterpart to diff against; verify by
physics + the density-component difftest.

**Scope:**
- An `Accum` trait (sum-reduction, `Vec3` payload) generalizing the step-9
  `Quant`-min pattern, plus a 3-component staged splat reusing the 8-color +
  thread-local staging + 3×3×3 commit decomposition from `particles/splat.rs`.
- The cubic-falloff kernel (Eq. 6) — replaces the step-9 min-SDF weight.
- Three staggered face-velocity channels + face-mass/face-density channels.
- A **staggered G2P sampler** (trilinear per face component at the `+e_a/2`
  offset, reusing the step-4 trilinear kernel) + the FLIP/PIC blend (Eq. 12).
- Divergence `∇·u*` (the RHS for steps 14/15).

**Decisions (owned):** MAC face-velocity now (matches the paper end-to-end; the
face mass feeds both the phase field and `β = 1/ρ`; `SampleVec3` is reused
per-face with a half-cell offset); velocity splat reuses the density staging
(3× bandwidth, still L1-resident); `α_FLIP` per particle type (liquid/air) per
§3.7 (Eq. 13, line 534).

**Acceptance:**
- `velocity_splat` bench: GB/s at 3 channels vs the density splat — bandwidth
  scaling is the regression check.
- Physical: rigid translation splats/gathers back exactly; momentum
  conservation; face vs cell consistency.
- Unit tests: 8-color race-freedom, cross-block splats, kernel `d > r` cull,
  FLIP/PIC `α ∈ {0,1}` endpoints.

---

## Step 14 — Matrix-free variable-coefficient Poisson operator (matvec + relaxation)

**Status:** HAVEN'T STARTED

**Build order:** SECOND. Consumes the step-13 face coefficients (`β = 1/ρ`) and
divergence; produces the operator that step 15's multigrid smooths.

**Paper:** §3.5 (`/tmp/msbg_paper.txt:534–556`), Eq. 8–10 (lines 469–481): the
variable-coefficient Poisson `Δt·∇·(β∇p) = ∇·u* + DIVCORR`, discretized on the
MAC grid as a symmetric 7-point stencil with coefficients sampled at faces (Eq.
10). §6.3 (`:847–875`): the smoother is a **two-stage red-black hybrid
GS-Jacobi** — block-outer red-black + in-block GS on whole SIMD words, `ω = 6/7`,
in-place (no destination buffer).

**Parity target:** C++ `IprocessBlockLaplacian`/`processBlockLaplacian` +
`multiplyLaplacianMatrixOpt`/`relaxBlockList` (`msbg3.cpp`) — **operator level
only**; these are dead in the repo (refactor.md §8) and the driver is
closed-source. The matvec/relax updates are identical to C++'s:
matvec `y = D·x − Σ_nb`, relax `y = ω(b + Σ_nb)·D + (1−ω)x`, with `D = 1/diag`.

**Scope:** the *implicit* operator over the step-5 halo + step-6 kernel infra:
- 7-point matvec + relaxed-GS update, with the **face-coefficient weighting**
  (the C++ `blockHasFaceCoeffs` branch, `S = Σ w_i·F_i`) from day one — `β = 1/ρ`
  from step 13.
- Inverse-diagonal precompute (`D = 1/Σ face_coeff`; the `Diagonal` channel type
  already exists).
- Mixed-resolution neighbors via the existing `fill_multires` coarse-ghost path
  + the paper's `k_CF = 1/(4Δx)`, `k_FC = 2·k_CF` resolution-transition scalars
  (§6.1 / Algorithm 1, lines 763–796).

**Decisions (owned):** 8-color in-place GS (convergence == GS, no scratch —
matches the paper's "halve the memory" claim, §6.3); `ω = 6/7` default, exposed
(skip `_mgSmOmegaSched1 = 1.7319` — dead even in C++, gated behind an undefined
macro); matrix-free (no assembled `L`).

**Acceptance:**
- `difftest_pressure.rs` + `../MSBG/pressuretest.cpp`: matvec + one relax sweep
  match C++ on the **constant-coefficient** branch (golden + live). The C++
  harness must call `relaxBlockList` directly — `relax()`'s `_blocksRelax` list
  is never populated. (The weighted branch is verified by the step-18/19
  physical tests + a scalar-reference oracle; wiring C++'s face-area channels
  for a live weighted diff is heavy and low-value.)
- `pressure_matvec` / `pressure_relax` benches vs the C++ harness.
- Unit tests: matvec on affine fields (Laplacian = 0), BC handling, GS vs
  Jacobi convergence rate, `ω` clamp endpoints, empty/full dummy neighbors.

---

## Step 15 — Adaptive algebraic-aggregation multigrid + flexible PCG (pressure solve)

**Status:** HAVEN'T STARTED

**Build order:** THIRD. Builds the paper's actual pressure solver on the step-14
operator + the step-11 `BlockMap` (for the lock-free active FIFO).

**Paper:** §6 (`/tmp/msbg_paper.txt:748–950`). This is **not** the geometric
multigrid in the C++ library. Three pieces:
1. **Galerkin (algebraic-aggregation) coarsening** of the *face coefficients* —
   Algorithm 1 (`:763–796`): per MG level / block / face-direction, coarsen the
   3 face coefficients from the fine level; copy-by-reference when no finer child
   exists; `k_CF`/`k_FC` resolution-transition scalars.
2. **Two-stage-red-black adaptive relaxation** — Algorithm 2 (`:885–936`):
   block-outer red-black (Jacobi half-passes) + in-block 8-color GS (our step-8
   sweeper), `ω = 6/7`; adaptive active-block FIFO reinserting blocks whose
   **contrast-equalizing** residual (`max Δ·diag`, §6.4 `:819–846`) exceeds `τ`;
   terminate after 10–20 steps; `τ` from a log-histogram (mean + 2σ).
3. **Flexible PCG** (Polak–Ribière) as the outer loop — adaptive relaxation makes
   the preconditioner asymmetric, so standard CG is invalid (`:922–950`). Reuses
   the existing `CgP`/`CgQ`/`Divergence` channel types.

**Parity target:** none in the C++ repo (its `_mgLevels`/`relax`/
`downsampleChannel` geometric-MG machinery is dead; the adaptive/FCG driver is
closed-source). The benchmark is the paper's *published* result: ~30 CG
iterations/step and pressure ≤ ~30% of sim time (Table 6, `:1156–1168`), 3×
fewer iterations than Galerkin MG + boundary relax (`:1091–1105`), and 4-point
stencil vs BoxMG's 14-point (Table 5, `:1136–1153`).

**Scope:** the three pieces above over the step-7 levels; `D = 1/diag` from step
14; the step-13 divergence as RHS.

**Decisions (owned):** go straight to the adaptive scheme (skipping a geometric
V-cycle deliverable); the verification anchor is operator-level difftests (step
14) + convergence/physical tests, not a C++ driver diff; measure
iterations-to-convergence against the paper's numbers rather than a live C++
side-by-side.

**Acceptance:**
- `multigrid_vcycle` bench: iterations-to-1e-6-residual + unknowns/s per cycle,
  compared against the paper's ~30 iters/step and ~10B-unknown-class throughput.
- Unit tests: constant-field solve, analytic sphere Poisson, MG == direct solve,
  the rising-bubble "difficult scenario" (§8.3) ~30-CG-iter target.
- Difftest the *operators* (matvec/relax, step 14) — the V-cycle itself has no
  C++ counterpart.

---

## Step 16 — Flat FLIP renderer (particles + clear-air water, no shading)

**Status:** HAVEN'T STARTED

**Parity target:** none in the C++ renderer (their `render.cpp` is a shaded
density isosurface only) — the water-surface intersection reuses the step-10
ESS-DDA raymarcher (already ~7.6× vs C++).

**Scope:** a FLIP visualization mode in `msbg-render`: the phase-field iso
raymarched (ESS-DDA) and filled flat opaque/translucent blue, the FLIP particles
rendered as depth-occluded point splats, and transparent air. No lighting. This
is the "can I see my sim" renderer — it makes the 1-phase-vs-2-phase difference
legible: 2-phase air turbulence kicks spray particles into the air, 1-phase has
none. Built and tested on synthetic particles + the step-9 density field,
independently of the sim steps.

**Decisions (owned):** flat surface color (bypass the step-10 `shade`);
particles projected via the step-10 camera and depth-occluded by the surface;
summed particle alpha as a cheap spray-density look; translucency is a flat
alpha blend over the background (no volumetric integration — "slightly
translucent" is a blend, not scattering).

**Acceptance:**
- Unit tests: a known particle set projects to known pixels; particles behind
  the surface are occluded; flat surface color is normal-independent; alpha
  blend math; determinism (same input → same frame).
- Render a step-9 bunny density + a synthetic spray cloud: splatter visible,
  surface flat blue, air transparent.
- `flip_flat_render` bench: Mpix/s + Mparticles/s — the surface pass is the
  step-10 ESS-DDA number (no regression), the particle pass is point-projection
  throughput.

---

## Step 17 — Shaded/translucent FLIP renderer (lighting + velocity viz)

**Status:** HAVEN'T STARTED

**Parity target:** the step-10 shaded isosurface (Blinn-Phong, ~7.6× vs C++
`render.cpp`) + velocity-field rendering (no C++ counterpart).

**Scope:** the "pretty" renderer: (1) Blinn-Phong water surface (step 10) with
translucent blending, (2) velocity-field visualization — a `|v|`/vorticity
colormap slice overlay and/or an additive 3D field raymarch via `SampleVec3` —
so the 2-phase air turbulence is visible, (3) lit particle drops. Tested on
synthetic velocity fields + the step-9 density, independently of the sim steps.

**Decisions (owned):** velocity viz = `SampleVec3` + turbo colormap (slices or
additive 3D raymarch); vorticity = curl of the sampled velocity; translucency
stays flat alpha (the deferred step-10 volumetric mode stays deferred).

**Acceptance:**
- Unit tests: colormap endpoints (zero velocity = background, max = hot);
  vorticity of a rigid rotation is constant; translucent blend == expected
  compositing.
- Render the step-9 density + a synthetic vortex field: turbulence visible,
  surface shaded + translucent.
- `flip_shaded_render` bench: surface pass holds the step-10 ~7.6× ratio;
  velocity-viz throughput reported separately.
- Side-by-side flat (step 16) vs shaded (step 17) on the same frame — the
  two-part deliverable.

---

## Step 18 — Single-phase FLIP driver

**Status:** HAVEN'T STARTED

**Parity target:** none in the repo (their FLIP loop is closed-source) — steps
13/14/15 compose into the driver.

**Paper:** §3 (`/tmp/msbg_paper.txt:351–557`) + §5.4 dual particle-grid
adaptivity algorithm (`:710–749`). The per-step breakdown target is Table 6
(`:1156–1168`): P2G 27% / pressure 30% / advect 13% / other 30%.

**Scope:** the timestep loop: P2G (step 13) → phase-field mean-curvature
smoothing (step 8) + phase-field compression Eq. 7 (`:422–495`) → pressure
projection (step 15) → G2P (step 13) → RK3 advection with CFL-limited adaptive
timestep (§3.7, `:526–557`). Single-level, single-phase (water in a box),
constant density — `β` collapses to a constant but the MAC grid +
variable-coefficient machinery still runs.

**Decisions (owned):** phase-field (no level-set reinit — `solveEikonalFIM`
skipped); explicit mean-curvature keeps the interface narrow.

**Acceptance:**
- Physical: a translating droplet doesn't deform; divergence-free to tolerance;
  volume/energy conservation over N steps.
- `flip_timestep` bench: full-timestep throughput (particles/s, voxels/s) with
  the operator breakdown vs Table 6 — internal (no C++ driver), but each
  operator already parity-benchmarked in 13–15.
- A dam-break preview frame via the step-16 flat renderer (visual sanity).

---

## Step 19 — Adaptive multires FLIP

**Status:** HAVEN'T STARTED

**Parity target:** C++ `downsampleVelocity` / `downsampleFaceDensity` (multires
velocity/face restriction, `msbg4.cpp`); the multires pressure solve (our step
15 over step-7 levels).

**Paper:** §5 adaptivity (`/tmp/msbg_paper.txt:559–749`), §6.2 leveraging MSBG
(`:799–815`). Refinement driven by the phase-field (interface = fine, bulk =
coarse).

**Scope:** refine from the phase-field (step-7 `RefinementMap`),
velocity/face-density restriction to coarse levels, multires pressure V-cycle
(step 15 over step-7 levels), and the fine-coarse transfer each timestep.

**Decisions (owned):** adaptivity driven by the phase-field (the "adaptive"
claim); reuse step-7 refinement + step-15 V-cycle; *measure* whether adaptive
beats single-level for the target scenario before committing to it (don't pay
transfer cost for a marginal win).

**Acceptance:**
- `downsample_velocity` / `downsample_face_density` difftest + bench vs C++
  `downsampleVelocity` / `downsampleFaceDensity`.
- Adaptive vs single-level timestep throughput on the same scenario (the "is
  adaptivity worth it" evidence).
- Unit tests: restriction/prolongation round-trip, refinement consistency
  across a moving interface.

---

## Step 20 — 2-phase cut-cell MAC

**Status:** HAVEN'T STARTED

**Parity target:** `computeCellFaceAreaFractionsGhost` / `getFaceAreaGen` /
`getFaceCoeffRightDomBorder` / `getGhostPressure_` — the sub-voxel air/water
face-area machinery, `msbg.cpp`.

**Note — two distinct face-coefficient concepts, don't conflate them:**
- The **fluid density coefficient** `β = 1/ρ` (paper Eq. 9, `/tmp/msbg_paper.txt:472`)
  lives in step 14's matvec; it comes from the step-13 face mass. That is the
  two-fluid density contrast.
- This step's `face_area`/`face_coeff` are the **solid cut-cell** fractions
  (sub-voxel obstacle geometry) — a *different* quantity, already scaffolded as
  `FaceBlock { face_area, face_coeff }` + `LevelData::ensure_faces` (step 7).
  Both multiply into the same 7-point stencil (the C++ `blockHasFaceCoeffs`
  branch uses the solid face areas; the paper's density coefficient is separate).

Also: the C++ parity targets here are themselves dead — `computeCellFaceAreaFractionsGhost`
is *declared but never defined*, `getFaceCoeffRightDomBorder` is an
`UT_ASSERT0(FALSE)` stub, and `getFaceAreaGen`/`getGhostPressure_` have zero call
sites. The only live "2-phase" code is `getGhostPressure_`'s *logic*, which the
paper describes but the repo never invokes. Port from the paper's §3.5/§6, treat
the C++ as a naming reference only.

**Scope:** air/water cell classification (`CELL_IS_LIQUID`/`CELL_AIR`/
`CELL_SOLID`/`CELL_VOID`), face-area fractions for the cut-cell MAC pressure
solve, free-surface BC, and surface tension via the phase-field curvature.

**Decisions (owned):** keep the phase-field for the interface (no explicit level
set); the face-area fractions are the C++ "2-phase" essence and are ported
directly (they are the sub-voxel accuracy the paper shows off).

**Acceptance:**
- `difftest_facearea.rs` + C++ harness: face-area fractions match
  `computeCellFaceAreaFractionsGhost` within tolerance.
- 2-phase pressure solve with a free surface: hydrostatic equilibrium + the
  classic 2-phase test cases.
- Bench vs C++ face-area throughput.

---

## Step 21 — Droplet crown (Phase C #1)

**Status:** HAVEN'T STARTED

**Scope:** the full 2-phase droplet-crown scenario (steps 18–20 composed) at
increasing resolution, ending at 32,768³ on AWS. The animation is the
deliverable.

**Acceptance:**
- Droplet-crown animation at scale, visually matching the expected 2-phase
  result.
- Throughput vs the paper's ~10B unknowns/s on the same-class 256 GB box.
- Regression: every operator holds its step-13..20 budget during the run.

---

## Step 22 — Dam-break + the showstopper (Phase C #2)

**Status:** HAVEN'T STARTED

**Scope:** the dam-break scenario at paper scale; the final side-by-side
"we beat them" numbers (100B-voxel reconstruction + the 2-phase animation),
plus the writeup (refactor.md sections) documenting where we're faster and why.

**Acceptance:**
- Dam-break animation at 32,768³.
- The headline table: our e2e vs the paper's (reconstruction, smoothing
  unknowns/s, pressure-solve throughput, total sim time) on the same 256 GB
  machine.

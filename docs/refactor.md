# MSBG Library Rust Rewrite: Architectural Blueprint (Revised)

## 1. Overview

This document defines the core Rust architecture for re-implementing the MSBG (Multiresolution Sparse Block Grids) library. The target is an **as-fast-or-faster**, **safe-by-construction** replacement that leverages Rust’s zero-cost abstractions, const generics, SIMD intrinsics (`std::simd`), and Rayon for parallelism.

Design pillars:

* **Const Generics for Zero-Cost Indexing:** Block dimensions are known at compile-time, turning integer math into bit-shifts and eliminating array bounds checks.
* **Cache-line / SIMD Aligned Memory:** Explicit 64-byte alignment with the data payload strictly at offset 0—no unsafe pointer subtraction required.
* **Branchless Inner-Loop Stencils:** Cell boundaries are handled via SIMD bit-masks rather than branches.


* **Mechanical Sympathy:** Explicit suppression of LLVM `noalias` miscompilations for in-place sweeps, and FTZ/DAZ hardware flags to avoid denormal float stalls.



---

## 2. Memory Management & BlockPool

### 2.1 Block Layout

Unlike the C++ implementation—which buried its data array after runtime padding and header metadata—we place the statically-sized data payload at offset 0. This guarantees strict 64-byte alignment for the data itself, keeping the layout safe for AVX-512 vectorization without unsafe pointer subtraction.

```rust
/// Represents a single 3D chunk of grid data.
/// Guaranteed 64-byte alignment for cache lines and SIMD loads.
/// 
/// Invariants:
/// - `N` MUST be equal to `BSX * BSX * BSX`.
/// - `BSX` MUST be a power of two.
#[repr(C, align(64))]
pub struct Block<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> {
    /// Contiguous voxel payload array placed at exactly offset 0.
    pub data: [D; N],

    /// Block status metadata flags.
    pub flags: u16,

    /// Explicit alignment padding to ensure 64-byte structural balance.
    _pad: [u8; 62],
}

impl<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> Block<D, BSX, N> {
    #[inline(always)]
    pub fn get_voxel(&self, vx: usize, vy: usize, vz: usize) -> D {
        debug_assert!(vx < BSX && vy < BSX && vz < BSX, "Voxel indices out of bounds");
        let bsx_log2 = BSX.trailing_zeros() as usize;
        let index = vx | (vy << bsx_log2) | (vz << (bsx_log2 * 2));
        
        // Safety: Bounds strictly checked by debug_assert and proven by bit-shifts.
        unsafe { *self.data.get_unchecked(index) }
    }
}

```

#### 2.1.1 Denormal Float Prevention

Loading bytes from padded or boundary areas into floating-point SIMD registers can occasionally produce subnormal (denormal) values. On x86-64 hardware, this triggers microcode assists that stall the pipeline.

To avoid this penalty, we configure the CPU's FTZ (Flush-to-Zero) and DAZ (Denormals-Are-Zero) flags at thread initialization:

```rust
#[cfg(target_feature = "sse2")]
unsafe fn enable_ftz_daz() {
    use std::arch::x86_64::*;
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

```

Rayon’s `ThreadPoolBuilder::start_handler` must call `enable_ftz_daz()` on each worker thread.

### 2.2 Strongly-Typed Monotonic Allocator

We use a lock-free monotonic allocator, but unlike raw C/C++ `void*` pools, we retain strict Rust type safety using `AtomicPtr<Block<D, BSX, N>>`.

```rust
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct BlockPool<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> {
    next_free: AtomicUsize,
    blocks_per_seg: usize,
    blocks_per_seg_log2: u32,
    blocks_per_seg_mask: usize,
    /// Lock-free atomic segment pointer table retaining full type information.
    segments: Vec<AtomicPtr<Block<D, BSX, N>>>,
    extend_lock: Mutex<()>,
}

impl<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> BlockPool<D, BSX, N> {
    #[inline]
    pub fn alloc_block(&self) -> NonNull<Block<D, BSX, N>> {
        let index = self.next_free.fetch_add(1, Ordering::Relaxed);
        let seg_idx = index >> self.blocks_per_seg_log2;
        let block_idx = index & self.blocks_per_seg_mask;

        let mut seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        if seg_ptr.is_null() {
            seg_ptr = self.extend_pool(seg_idx);
        }

        unsafe { NonNull::new_unchecked(seg_ptr.add(block_idx)) }
    }
    
    // Implementation of extend_pool uses self.extend_lock to safely Box::into_raw new segments.
}

```

---

## 3. SparseGrid & Borrowing Strategy

### 3.1 Grid Definition

```rust
pub struct SparseGrid<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> {
    pub sx: usize, pub sy: usize, pub sz: usize,
    pub nx: usize, pub ny: usize, pub nz: usize,
    
    /// Block-id -> optional aligned block pointer. 
    /// Option<NonNull> is 8 bytes, exact same as a C++ pointer.
    pub blockmap: Vec<Option<NonNull<Block<D, BSX, N>>>>,
    pub block_pool: std::sync::Arc<BlockPool<D, BSX, N>>,
    // ... precomputed SIMD strides and dummy blocks
}

```

### 3.2 Aliasing and In-Place Mutations

When executing read-modify-write kernels, we utilize two distinct borrowing paradigms:

1. **Double-Buffering (Safe Rust):** For algorithms like `filter` that read from a source grid and write to a destination grid, standard safe borrows (`&src` and `&mut dst`) work perfectly.


2. **In-Place Sweeps (Red-Black Gauss-Seidel):** Reading neighbor cells while mutating the center cell in the *same* grid violates Rust’s aliasing rules. If implemented naively, LLVM emits `noalias` metadata, which leads to miscompilations.


* **Solution:** In-place grids must use `UnsafeCell<f32>` for their payload. We expose raw `*const f32` and `*mut f32` pointers during these passes, which inherently suppresses the `noalias` attribute and guarantees correct code generation.





---

## 4. SIMD Strategy (`std::simd`)

We utilize `#![feature(portable_simd)]` to replace C++ vector classes with native `std::simd` types.

### 4.1 Masked Branchless Inner Loops

To maximize throughput and avoid unaligned cache-line straddling, we process rows in their entirety (`x = 0..BSX`) using unconditionally aligned SIMD loads.

Instead of writing `if/else` logic to exclude boundary/solid cells, we map `CellFlags` into a SIMD mask. The mathematical update is calculated for *all* cells, but the mask selects the original value for boundary cells before writing back.

```rust
use std::simd::f32x8;

/// Block-wise Laplacian – aligned loads + cell-flags mask.
unsafe fn relax_block_simd_masked<const BSX: usize>(
    p: *const f32x8, b: *const f32x8, diag: *const f32x8,
    flags: *const f32x8,                 // CellFlags reinterpreted as f32 lanes
    mask_interior: f32x8,                // set bits for interior fluid cells
    r: *mut f32x8,
) {
    for z in 0..BSX {
        for y in 0..BSX {
            let row_start = (z * BSX + y) * BSX / 8;
            for x in 0..(BSX / 8) {
                let idx = row_start + x;

                // Aligned loads – never straddle a cache line
                let p0 = p.add(idx).read();
                // (Padding guarantees out-of-bound reads at x=0 safely return 0.0)
                
                // ... calculate lapl ...
                let new_val = b.add(idx).read() - lapl * diag.add(idx).read();

                // Keep original value for non-interior (solid/boundary) cells
                let preserved = r.add(idx).read();
                let final_val = mask_interior.select(new_val, preserved);

                r.add(idx).write(final_val);
            }
        }
    }
}

```

---

## 5. Parallel Reduction Strategy

For grid-wide aggregations (e.g., maximum velocities, total kinetic energy), we must **never** use `std::thread_local!` inside Rayon closures, as OS-level TLS lookups carry a severe dynamic performance penalty.

Instead, every reduction must use Rayon’s native `.fold().reduce()` pipeline to guarantee zero-overhead thread-local aggregation:

```rust
fn max_abs_channel<const BSX: usize, const N: usize>(sg: &SparseGrid<f32, BSX, N>) -> f32 {
    use rayon::prelude::*;
    
    sg.blockmap.par_iter()
        .fold(
            || 0.0f32, // Thread-local accumulator
            |mut local_max, block_opt| {
                if let Some(block) = block_opt {
                    let block_ref = unsafe { block.as_ref() };
                    // SIMD max operations here...
                }
                local_max
            },
        )
        .reduce(
            || 0.0f32,
            |a, b| a.max(b), // Global merge
        )
}

```

---

## 6. Directory Layout (Proposed)

```text
src/
  blockpool.rs      // Monotonic lock-free allocator
  sparse_grid.rs    // Data structures & generic block logic
  multires.rs       // Topology and BlockInfo storage
  simd_kernels.rs   // Masked SIMD inner loops and math
  channels.rs
  relax.rs          // In-place solvers (Red-Black Gauss-Seidel)
  lib.rs
  benches/
    bench_blockpool.rs
  tests/
    difftest_cpp.rs   # Differential testing against C++ baseline

```

---

## 7. Field sampling & interpolation (step 4)

The C++ interpolation layer (`sbg.h`) is ~40 template functions with three
interleaved dispatch paths (domain-border / within-block / straddle) plus
academic code (WENO, cubic-mono, quadratic, ghost, dense-scalar, coherence
cache). The Rust port collapses this to four small modules:

```
src/math/
  sample.rs     # InterpElem/Sample/SampleVec3 traits, Sampler structs, reducers
  bspline.rs    # const cubic B-spline weights (value / 1st / 2nd derivative)
  gather.rs     # stencil-window gather (branchless fast path + 8-slot fallback)
  boundary.rs   # BoundaryCondition / GridAlignment / Interpolation + resolve_axis
```

Deliberate deviations from C++ (each a win or a wash, none a 1:1 port):

- **f32-only arithmetic.** C++ computes the trilinear value/gradient in `double`
  (`MtTrilinearInterpolationOld`). Rust stays in f32 (FMA via `mul_add`), which
  is faster and changes results by ≤1 ulp — hence the *tolerance-based* difftest.
- **Unified branchless gather.** One `gather` serves linear and cubic windows:
  a single mask selects a fast path (window inside one block + domain → one
  blockmap lookup, direct index math, no BC branch) vs. an 8-slot fallback
  (resolve the ≤8 surrounding blocks once; empty/full dummies map to their
  materialized dummy-block data so reads are unconditional). This replaces
  C++'s three-path dispatch.
- **Dequant-per-corner.** `Dequant::dequant` (the `InterpElem` /
  `InterpVec3Elem` marker traits) linearizes a stored sample before
  interpolation. For `Density` (u16) and `Density8` (u8) this is equivalent to
  C++'s "interpolate in storage units, scale once at the end" for linear/cubic
  (scaling commutes with the affine weights); for sqrt-compressed `Density8`
  the sampler returns the interpolated sqrt-space value (square it to recover a
  physical density), matching C++'s render path. `gather_map` applies it inline
  so there is no intermediate `[D; W]` scratch array.
- **Not ported (dead/academic):** `interpolateGhost`, `interpolateDenseScalarFast`,
  `IpolAccessCache` (never used in the live C++ scalar path),
  `interpolateWithSecondDerivs`' `Vec4d` double-gradient accumulation (Rust uses
  f32), `IP_WENO4` / `IP_CUBIC_MONO(_2)` / `IP_LINEAR_FADE` /
  `IP_LINEAR_DIVERGENCE_FREE`, `wenoParabola`, `Interpolate1D_MINMOD`.

Benchmark (scenario G, `../MSBG/benchmark.cpp interp` vs `benches/interp_bench.rs`,
96³ grid, 100k pseudo-random interior samples): at parity with C++ on linear
value, ~10% slower on linear value+grad, and **faster on cubic** (value+grad
~20% faster, Hessian parity-to-faster). Two findings drove this: (1) the first
cut was ~3× slower on linear until a within-block fast path was added (the
unified 8-slot gather paid 8 blockmap lookups per sample vs C++'s 1); (2) the
cubic y/z accumulation became FMAs via `mul_add` (`val += wy*wz*sx` → `val =
sx.mul_add(wy*wz, val)`), since LLVM won't reassociate FP — a ~10% win. `#[inline]`
(not `always`) on the trait methods is required for cross-crate inlining.

Two rejected experiments: a 4-lane SIMD trilinear gradient (4×8 dot products)
was ~3× the op count of the Maxima-factored `MtTrilinearInterpolationOld`
formula (which is op-minimal — ~32 flops), so it was reverted; and a
closure-based dequant (`Fn(D) -> O`) didn't inline reliably (linear value
regressed 33 → 24 Ms/s), so the gather uses a `Dequant<O>` trait method instead.

---

## 8. Solver architecture (step 8) & the "dead code" finding

The C++ multigrid pressure solver — `multiplyLaplacianMatrixOpt`, `relax`,
`relaxBlockList`, `dotProdChannel`, `AXPBYChannel`(`Combined`), and the dense
coarse MG levels (`blockSize = 1`, `isDenseLevel`) — has **zero call sites** in
the repo. The happy path is a single call:

```cpp
applyChannelPdeFast<RenderDensity>(..., -(PDE_MEAN_CURVATURE + OPT_8_COLOR_SCHEME), ...);
```

i.e. **8-color in-place mean-curvature smoothing** (`msbg_demo.cpp:716`), the
"PDE solver" the README's "~10B unknowns/sec" bullet refers to. The pressure
solver is real, useful machinery (it is what a *standard* FLIP pressure
projection needs), baked into the library but not exercised by the phase-field
demo — the paper's method replaces the pressure Poisson solve with mean-curvature
flow. `_mgSmType` (Jacobi/GS selector) is set at `msbg.cpp:818` but never read;
red-black (`RELAX_BLOCKS_RED_BLACK`) colors *blocks* (`msbg3.cpp:2212`), not
voxels, in the dead `relaxBlockList`; `MSBG_SORT_BLOCKLISTS` is never defined, so
the Morton sort at `msbg.cpp:2043` is compiled out.

Implications for the Rust port:

- **Core stays use-case-independent.** `blockpool` / `sparse_grid` / `channel` /
  `math` / `multires` provide data structures + the **8-color in-place sweep**
  as a generic primitive (read halo → per-block stencil → write back). Mean-
  curvature, Laplacian, and a future pressure matvec are all just stencils over
  that primitive — no solver leaks into the core.
- **Happy path first.** Port `applyChannelPdeFast` (laplTyp 1 + 4, 8-color) as
  the live, benchmarkable target. The MG pressure solver is a later extension,
  designed Rust-native (not as parity with the dead code).
- **Threading.** The happy path threads over a flat active-block list via
  `ThrRunParallel` (→ TBB, `thread.h:287`) with no locks; Rust's rayon
  equivalent is already the pattern in `halo.rs`.

### What step 8 shipped (`src/solver.rs`)

- `Stencil { Laplacian, MeanCurvature, BiLaplacian }` + `PdeParams { dt,
  iterations, do_constr_zero_one }` replace the 19-arg `int laplTyp` API (whose
  negative value + `OPT_8_COLOR_SCHEME` bit encodes "in-place, 8-color").
- A `Sweeper<D, BSX, N, HSX>` primitive: pre-bucket the active list into 8 color
  lists (color = `bx&1 | (by&1)<<1 | (bz&1)<<2`) once, Morton-sort each bucket,
  then for each iteration run 8 parallel passes. Blocks of one color are ≥2
  blocks apart on some axis, so their ≤2-voxel halos never overlap — the in-place
  write is race-free by construction. C++ instead re-scans the *full* active list
  8× per iteration with a per-block `getBlockCoordsById` + color branch (its
  `USE_RB_SEP_LISTS` pre-bucketing is `#ifdef`'d out).
- The step-6 kernels were refactored from `&mut Block<f32>` output to a raw
  `*mut D` sink behind `StoreBack<W>`: `f32` → `store_nt` (non-temporal) +
  `sfence`; `Density` → round-to-`u16`; `Density8` → sqrt + stochastic-rounding
  `u8` (a lane-parallel `SimdRng` mirroring C++ `FMA_FastRandSeed`/`FMA_FastRand`).
  The fluid `MaskBlock` was removed — C++ smooths every voxel, only clamping to
  `[0,1]` (`doConstrZeroOne`), and the demo relies on that.
- **Scratch channels dropped.** `CH_FLOAT_2`/`CH_FLOAT_3` are only touched by the
  non-colored Jacobi double-buffer path (`laplTyp >= 0`); the demo passes
  `CH_NULL`, so `chSrc == chDst` and the trailing `resetChannel(...)` calls are
  all `resetChannel(CH_NULL)` no-ops. `prepareDataAccess`/`resetChannel`/
  `protectChannel` are likewise unneeded (Rust's lazy `Option<BlockPtr>` blockmap
  covers allocation; there is nothing to protect/free on the live path).

### Measured (Dell 5500U, small scale, 12 threads)

Scenario E is now **8-color in-place on both sides** (the C++ `benchmark.cpp`
scenario E was switched from `laplTyp` to `-(laplTyp + OPT_8_COLOR_SCHEME)` so it
runs the demo's live path, not the Jacobi path). Gvox/s (per iteration):

| leg | C++ 1k / 5k / 10k | Rust 1k / 5k / 10k | ratio |
|---|---|---|---|
| Laplacian | 1.79 / 1.48 / 1.52 | 2.10 / 1.73 / 1.72 | +13–17% |
| mean-curvature | 1.40 / 1.34 / 1.35 | 1.85 / 1.48 / 1.52 | +11–32% |

`perf record` on the mean-curvature sweep: **90% of cycles** in the inlined
`sweep` hot loop (the kernel is fully inlined — no standalone `kernel_meancurv`
symbol survives). The win is the scheduling (pre-bucket vs 8× rescan), the
step-5 pre-resolved 27-pointer halo gather, and the step-6 NT store; the fence
choice is noise — `sfence` (15.26 ms), `mfence` (15.47 ms), none (15.50 ms) for a
5000-active mean-curvature pass.

The multi-iteration difftest tolerance is **1e-4 (Laplacian) / 1e-3
(mean-curvature)**: the mean-curvature mixed partials are factored
`0.25*(a-b-c+d)` (fewer FLOPs than C++'s two-stage `0.5` form), and near the
`gradMagSq > 1e-7` guard cliff that ~1-ulp `H` difference flips a voxel between
`H = hnum/grad` and `H = 0`; four Gauss-Seidel iterations propagate the
discontinuity. The single-iteration kernel difftest still matches at 1e-4.

Micro-optimization experiments (all rejected — no significant gain):

- `blockmap.get_unchecked` in the halo gather: ~2% within noise (memory-bound
  18³ copy dominates the 7 bounds checks). Reverted.
- Y-unrolling the halo center copy: noise. Reverted.
- SIMD-vectorizing the interpolation gather: unnecessary — a `Density` (u16)
  cubic gradient is ~45% *faster* than `f32` (109 vs 158 ns) because the gather
  reads half the bytes; LLVM already vectorizes the contiguous loads.
- `Block::_pad: [u8; 62]` is redundant (`#[repr(align(64))]` already rounds the
  size up) — a cleanup, not a perf change.

---

## 9. Halo / ghost gather (step 5)

The C++ `fillHaloBlock_` iterates the 18³ halo as three x-segments per row
(left halo / bulk middle / right halo) with a **per-row `getBlock`** (~324
blockmap lookups per fill) and per-voxel `getOutOfBlockValue` for the boundary.
The Rust port instead:

- pre-resolves the **3×3×3 neighborhood to 27 raw pointers once** (empty/full
  dummies map to their data, out-of-range blocks to the empty dummy) — 27
  lookups vs C++'s ~324;
- bulk-copies the middle via `Dequant::copy_row` (`memcpy` for `f32`, scalar
  dequant for `u16`/`u8`), and resolves the two x-halo columns per row from
  pre-resolved coords (no per-voxel `resolve_axis`);
- makes the halo buffer **concrete `f32`** (dequantized on gather, like C++),
  so the step-6 stencils never re-dequantize;
- reuses `math::boundary` (`BoundaryCondition` + `resolve_axis`) for
  Neumann/Clamp/Dirichlet — the first cross-module reuse of step 4.

`fill::<FULL>` (const-generic) selects full 18³ (mean-curvature) vs faces-only
(7-point Laplacian), mirroring C++'s `do1stOrderOnly` template param.

Benchmark (scenario D, both legs, single-level): Rust ~1.8× faster than C++ on
full fill and ~1.7× on faces-only — the gap comes from the 27-pointer
pre-resolution and the memcpy middle (vs C++'s per-row `getBlock` + SIMD
transfer). Note: the laptop (Ryzen 5500U) thermal-throttles under sustained
benchmarks, so ~10% deltas are noise; this gap is far outside that.

---

## 10. Cross-platform benchmarking (Dell 5500U vs M3 Pro)

The benches are sized by `Machine { Dell, Macbook }` × `Size { Small, Big,
XBig }` (`MSBG_BENCH_MACHINE`, `MSBG_BENCH_SCALE`). `small` is identical on both
machines for apples-to-apples comparison; `big` is the per-machine stress
(MacBook ~20 GB peak); `xbig` is the MacBook-only ~30 GB peak. On macOS there is
no C++ baseline (that is the point), so MacBook runs are compared against the
Dell *Rust* numbers only.

aarch64 caveats to keep in mind when reading M3 Pro numbers:

- **128-bit NEON vs 256-bit AVX2.** `f32x16` lowers to 4 NEON ops on aarch64 vs
  2 AVX2 ops on the 5500U. Per-core FMA width is comparable (4×128 ≈ 2×256 per
  cycle), so compute-bound kernels are ~per-core-neutral; the M3's edge is
  memory bandwidth (~150 vs ~50 GB/s), more real cores (12 vs 6c/12t SMT), and
  higher IPC.
- **`enable_ftz_daz` is a no-op on aarch64** (x86 MXCSR only). ARM handles
  denormals in hardware, so no microcode-assist hazard; only relevant if a
  kernel ever generates denormals.
- **P/E asymmetry.** The M3 Pro's 6 efficiency cores are much slower than its 6
  performance cores, and rayon has no P/E affinity — multithreaded benches may
  scale sub-linearly past ~6 threads (E-core stragglers). If observed, the fix
  is P-core pinning / thread-count tuning (follow-up, not yet done).

### Measured on the M3 Pro (12c, 36 GB, aarch64)

P-core L1d=128 KB / L1i=192 KB / L2=16 MB; E-core L1d=64 KB /
L2=4 MB; SLC ~36 MB. `rayon_threads=12`.

| workload | Dell (5500U, throttled floor) | M3 Pro | ratio |
|---|---|---|---|
| halo full, small (cache-resident) | ~4.5–5.0 Gvox/s | 24.8 Gvox/s | ~5× |
| halo full, 500k (DRAM-bound) | ~4.9 Gvox/s | 17.6 Gvox/s | ~3.6× |
| interp linear value (96³ grid) | ~30 ns/sample | 6.2 ns/sample | ~4.8× |
| interp cubic grad / hess | ~150 / ~200 ns | 59.3 / 75.5 ns | ~2.5–2.6× |
| density dequant / quantize | — | 17.5–19.1 / 16.9 G/s | — |
| blockpool hot, 1k blocks | — | 1.79 ns/block | — |

Everything is at or above the hypotheses above. The "above" cases (halo small,
interp) are cache artifacts: the 96³ interp grid (3.5 MB) and small halo sets fit
the M3's 16 MB L2 / 36 MB SLC, while the 5500U's 8 MB L3 cannot hold them. The
DRAM-bound halo (~3.6×) is the honest bandwidth ratio (~150 vs ~50 GB/s).

Reading the criterion output:

- **Ignore every `change: %` and "regressed/improved" line.** Criterion diffs
  each run against the *previous* run's stored estimate, and the small/big/xbig
  runs use different sizes — so `voxel_access +718%` is 64³→128³, and
  `blockpool 1M −97.9%` is a clean measurement beating a prior throttled one.
  Only the absolute `time:`/`thrpt:` lines are comparable.
- **Thermal throttling.** The 5500U throttles within ±15% noise; the M3 Pro
  throttles under the *big/xbig* legs: `laplacian_e2e` collapses 8.0→3.2→1.74
  Gvox/s from 250k→500k→750k active, and the first-touch of the 16 GB pool in
  `blockpool_hot/1M` produced one 881 ms outlier (the clean re-run is 17.7 ms).
  Trust the 250k/500k halo numbers (17.6–17.8 Gvox/s, stable pre-throttle) for
  DRAM characterization, and re-run cold (plugged in, idle) for the big legs.
- **`target-cpu=native` matters differently per arch.** On x86 it's essential
  (baseline SSE2 would scalarize the SIMD kernels). On aarch64 NEON is baseline,
  so the flag's win is **LSE atomics** (`ldadd` vs LL/SC) for
  `AtomicUsize::fetch_add` — visible in the blockpool benches.



---

## 11. Multires hierarchy (step 7)

The C++ multires layer has three structural costs the port removes:

- **AoS `BlockInfo { uint16 level, flags }` per (levelMg, block)** — `nLevels`
  copies, rewritten every `setRefinementMap`. Rust stores the level-0 refinement
  level as a single `u8` per block (`BlockInfoStore::level0`) and derives the
  per-levelMG effective level as `max(level0[bid], levelMg)` on demand — the hot
  refinement sweeps walk 1 byte/block instead of 4.
- **`getChannelAddr` void\*\* indirection** over ~40 named `SparseGrid*` fields.
  Rust uses a typed `LevelData<BSX, N>` (density f32, cell flags, distFineCoarse,
  3×face_area, 3×face_coeff) behind a closed `Level` enum — cross-level dispatch
  happens once per operation, not per voxel.
- **`setRefinementMap` unconditionally `resetChannel(CH_CELL_FLAGS)`** (free +
  reallocate the blockmap + `BlockPool` for every MG level) even when
  `doInitCellFlags=false`. The Rust topology computation touches nothing but the
  level/flags arrays.

Benchmarks (`benches/multires_bench.rs` vs `../MSBG/benchmark.cpp multires`,
small scale, spherical-shell refinement map, Dell 5500U):

| workload | C++ | Rust | ratio |
|---|---|---|---|
| multires halo gather (608 fine blocks) | 1.35 Gvox/s | 3.16 Gvox/s | ~2.3× |
| multires halo gather (5240 fine blocks) | 2.02 Gvox/s | 3.63 Gvox/s | ~1.8× |
| set_refinement_map (4096 blocks) | 12.1 ms | 11.0 ms | ~1.1× |
| set_refinement_map (35937 blocks) | 93.3 ms | 81.7 ms | ~1.14× |

The halo win is the step-5 pre-resolution carried to the coarse path: `fill_multires`
resolves each coarse neighbor's data pointer once (per 3×3×3 neighborhood) and
reads it with direct index math, where C++ `getOutOfBlockValue` +
`OPT_BC_COARSE_LEVEL` does a full `getValuePtr` (blockmap lookup + clip) per halo
voxel. The `set_refinement_map` win is understated: the Rust path *also*
initializes the cell flags the C++ benchmark path (`doInitCellFlags=false`)
skips. Note the 5500U thermal-throttles under sustained benchmarking, so the
absolute deltas wobble ±10–15%; the halo gap is far outside that.

`init_dist_fine_coarse` (the `CH_DIST_FINECOARSE` fill) is SIMD-vectorized across
x (`Simd<f32, LANES>`), hoisting the y/z distance out of the x loop and using
`to_int_unchecked` for the quantize; the C++ `distToBoxSq` is a 4-wide `Vec4f`
that wastes its 4th lane. Two rejected experiments: a scalar `f32::INFINITY`-init
`simd_min` accumulate (fine), and a `f32 as u16` store (the Rust saturating cast
added ~40% over `to_int_unchecked` + `cast`).

Morton ordering (`sort_block_list_morton`) is enabled (C++ `sortBlockListMorton`
is `#ifdef`'d out): the key is computed on the coarsest lattice so a coarse block
and its ≤8 fine descendants are contiguous in the sorted active-block list — the
index-space interleaving a cache-friendly sweep and the eventual temporal-blocking
scheme need, without touching physical allocation.

## 12. Surface reconstruction (step 9)

### The splat is DRAM-latency bound in C++, so we changed the architecture

The C++ `msbg_test_sparse` splat does a per-voxel `min`-RMW directly into the
grid for every overlapping particle, race-free via the 8-color scheme. At scale
the RMWs all miss cache (the active set is far larger than L3), so the splat is
DRAM-latency bound: a 16.7M-particle bunny-of-bunnies took **~17 minutes**
(≈ 16K particles/s) before we killed it.

The Rust port instead uses **thread-local staging + a one-time commit**:

1. **Stage** (`stage_chunk`, SIMD8): each block's particle chunk accumulates its
   `min` into a thread-local `(BSX+2·ceil(rScan))³ = 24³` buffer (`MSX=24`).
   The whole buffer lives in L1 (27 KB), so the ~140M particle-voxel writes are
   L1 hits. The inner loop quantizes 8 voxels per SIMD lane group.
2. **Commit** (`commit_chunk`, SIMD): write each *touched* staging voxel to its
   real block exactly once per contributing block — `min(grid, staging)` with a
   16-lane (interior) / 4-lane (margin) SIMD min that skips untouched chunks.
   The 8-color pass order still makes it race-free (`rScan < BSX`).

The RMW count drops from `#overlapping particles` to `#contributing blocks`
(≤ 8), cutting the grid traffic ~100× for dense clouds. Result: the splat went
from ~17 minutes to ~0.1 s for the same particle count (per-particle 390 → 106 ms
at 523K particles). Two experiments that lost: MSX=32/SIMD16 (the 64 KB buffer
spills L1 → 226 ms), and keeping the scalar stage (175 ms).

### Placement: C++ SIMD4 vs Rust scalar

The placement loop (`sort::place`) is the one phase where C++ wins (95.7 vs
14.5 Mparticles/s at scale). The C++ computes the 3-axis position + domain +
footprint with `Vec4f` SIMD4 (~8 ops) and writes to a preallocated array; the
Rust path is scalar (~30 ops) with Vec pushes. The position math is identical
bit-for-bit (the non-fused `origin + inst_scale·(bj−bbox)` form; a `mul_add`
contraction diverged from g++'s codegen by up to 14 density ulps after the
finalize sqrt near ratio 0). The footprint uses the C++-faithful division form
(`trunc(p/bsx ± rScan/bsx)`); folding it into the center-block offset
(`fp = p − 16·bx`) was within noise, so placement and active-block determination
share the single implementation. An SOA-SIMD placement is deferred (roadmap
step 9).

### Diff-testing the u16 field: ulp budgets and the finalize sqrt cliff

The pipeline is compared against `../MSBG/splattest.cpp` (the real demo path)
field-by-field. Field A (after finalize) and B (after 6 MC sweeps) match within
**max 2 ulps** with **≤0.1% of voxels off by 1** — the residual is the expected
f32/FMA divergence. Note the sharp edge: the finalize maps `sqrt(distSq)` near
the particle centers where `d(sqrt)/d(ratio) → ∞`, so a 1-ulp difference in the
splat's stored ratio can become a ~14-ulp difference in the density. The budget
was set empirically to cover that amplification without masking real bugs
(a missing footprint block shows up as thousands of ulps, not a few).

### Other step-9 findings

- **The demo is single-level and scratch-free.** `msbg_test_sparse` uses
  `OPT_SINGLE_LEVEL`; the refinement map is field-neutral. The splat, finalize,
  and smoother all operate on the one `CH_UINT16_1` channel.
- **The 1-voxel-halo staging plan was wrong for `rScan=4`.** A particle within
  `rScan` of its block boundary spills up to 4 voxels into the neighbor; the
  staging window must be `ceil(rScan)` deep.
- **`ply-rs` handles the ASCII bunny files** but its payload grammar rejects
  uppercase `E` scientific notation and `.5`/`5.` forms; the shipped files are
  plain decimals. A hand-rolled scanner is the documented fallback.

## 13. Rendering (step 10)

The renderer lives in a **separate workspace crate** (`msbg-render`) so it can
only see `msbg-rs`'s `pub` items — the step-10 acceptance criterion made
structural, not conventional.

### The C++ slice extractor is O(N³) *and* dead

`getSlices2D` (`msbg.cpp:5057`) has **zero call sites** in the repo; the demo's
`visualizeSlices` runs the same O(N³) triple loop through the panel UI. Both
scan every voxel and test `xIsSlice||yIsSlice||zIsSlice`. The Rust slicer
iterates only the output pixels and samples once each, so a `512³` slice costs
`512²` samples. Measured (Dell 5500U, real bunny):

| slices (3 planes) | C++ `getSlices2D` | Rust `render_slice` | ratio |
|---|---|---|---|
| 256³ | 34.9 Mpix/s | 231.8 Mpix/s | ~6.6× |
| 512³ | 21.5 Mpix/s | 200.9 Mpix/s | ~9.3× |

The ratio *grows* with resolution (the O(N³) scan vs O(N²) sample gap widens);
note the Rust side does *linear* interpolation per pixel while the C++ demo path
uses `IP_NEAREST`, so the Rust number is conservative.

### The raymarch win is empty-space skipping, not faster sampling

The C++ `RaymarchRenderer` marches a fixed `1/res` step with a per-sample
`isEmptyBlock(bid) → return 0` early-out — it does not *advance* past the empty
block, so empty regions are still micro-stepped one voxel at a time. The Rust
renderer walks the `16³` block lattice with an Amanatides–Woo DDA and jumps to a
block's exit plane in O(1) when it is empty, fine-stepping only inside value
blocks (where the surface is). `dir == 0` axes are handled by infinite `tMax`,
so grazing rays don't divide by zero.

Measured (Dell 5500U, real bunny):

| raymarch | C++ | Rust (ESS) | ratio |
|---|---|---|---|
| 256³, 640×480 | 2.77 Mray/s | 7.12 Mray/s | ~2.6× |
| 512³, 960×540 | 1.26 Mray/s | 9.52 Mray/s | ~7.6× |

ESS-on vs ESS-off (the fixed-step reference with the same sampler): **5.0×
@256³, 18.5× @512³**. The `perf` profile is clean: **86.5%** of raymarch cycles
in `Sample::sample::<Linear>` (the trilinear gather), 11.8% in the trace
closure — after ESS, traversal overhead is gone and the remaining cost is the
essential surface sampling. The natural next lever is SIMD ray packets (4–8
rays per `Simd` lane sharing one DDA traversal).

### DDA gotcha: a ray origin exactly on a block boundary

The demo camera sits at `x = 0.5·sx`, a multiple of `BSX`, so every ray with a
negative `x` direction starts with `t_max[x] == 0` — the block boundary is
"now". The first cut of the empty-block skip treated `tnext <= t` as "no
progress" and returned no-hit, killing every negative-x ray at `t = 0` and
blanking half the frame (the "bunny cut in half" bug). The fix: `advance()` to
the next block unconditionally (it moves the block index with no `t` change) and
only bail when the block exit is past the ray limit. This also made the
*benchmark* look better than it was — half the rays returned `None` for free —
so the table above is the corrected measurement. Regression test:
`raymarch_11_origin_on_block_boundary`.

### A real public-API finding: the sampler has no empty-block early-out

The ESS-off reference is ~2.4× *slower* than the C++ raymarcher, not at parity.
The reason is instructive: C++ `sampleDensity` fuses "is this block empty" +
"interpolate" into one function, so it never interpolates in empty space. The
Rust public `Sampler::sample::<Linear>` has no such hint — a naive consumer
pays the full 8-corner gather for every empty-space sample. The renderer's ESS
is exactly the workaround a user must write. This is the kind of design flaw
step 10 exists to surface; a future `Sampler` "empty block" fast path (or a
documented `is_empty_block` + gather pairing) would close the gap for naive
consumers.

### Other step-10 findings

- **The camera is left-handed and in normalized `[0,1]³`.** C++
  `right = cross(forward, worldUp)` with `worldUp = +Y` points *left*; the ray
  is `normalize(forward·focal + right·ndcX + up·ndcY)`, camera at `{0,0,-5}` by
  default (but the demo overrides it to `{0.5,0.8,0.7}`, inside the grid).
  `msbg-render` reproduces this basis in native `Vec3` (no `glam` needed — the
  whole thing is three lines).
- **`getSlices2D` needs the vec3 channel just for `sx/sy/sz`** (`_sparseGrids[0]
  .vec3_1[0]`) and asserts `levelMg==0`; `RaymarchRenderer` is in `libmsbg.a`
  (`render.o`), so the harness links against the existing static lib.
- **The C++ renders in `double` (`RealType`)** for ray/camera math; Rust uses
  `f32` throughout, matching the interpolation layer.

## 14. Sparse blockmap (step 11)

The C++ `SparseGrid<T>::_blockmap` is a dense `Block<T>**` (sbg.h:2387) indexed
by an `int bid` (`getBlockIndex` returns `int`, sbg.h:1744), and
`MultiresSparseGrid::_blockmap[MSBG_MAXRESLEVELS]` is a second dense AoS
`BlockInfo` array (msbg.h:958). Two hard limits at paper scale: the dense map is
`n_blocks × 8` = **68.7 GB/grid @ 32,768³/block-16**, and the `int` bid
overflows past 2³¹ blocks (8.59e9) — the C++ cannot even *address* its own
headline resolution with block-16. The paper dodges this with block-32, at the
cost of a non-L1-resident kernel; we keep block-16 and fix the map instead.

### The `BlockMap` primitive (`src/blockmap.rs`)

Open addressing, SoA (`keys: Vec<usize>` + `vals: Vec<MaybeUninit<V>>`), linear
probing, power-of-two capacity, tombstones on delete. Absent = empty (the empty
dummy is never stored); the full dummy is a real entry; `set_empty_block` is a
`remove`. The `MaybeUninit` vals are sound because `V: Copy` never needs a drop.

The first hash was a SplitMix64 finalizer (3 multiplies, a ~15-cycle serial
dependency chain) — 3.3–3.9× slower than a dense index. The fix is a single
odd-constant multiply `key * 0x9E37_79B9_7F4A_7C15`: multiplication by an odd
constant is a *bijection mod 2^k*, so the low `k` bits (which linear probing
consumes) are a perfect permutation of the key's low bits, spreading consecutive
`bx`-cluster bids uniformly across the table. Measured (`benches/blockmap_bench.rs`,
real block pointers, hits/misses): dense `Vec` index 0.53 ns, `BlockMap::get`
0.88 ns (1.66×, inside the 2× acceptance), `hashbrown` 1.6–3.1× slower — so the
map stays hand-rolled and hashbrown is only a dev-dependency of the comparison
bench.

The end-to-end no-regression is what matters: the interpolation fast path (one
probe per sample) went 4.78 → 4.87 ms (+2%, inside the ±15% thermal band) — the
probe is ~1 ns of a ~48 ns sample, and the dominant accesses (active-list
sweeps, halo neighbors, gather) are hit-heavy where the gap is 1.66×.

### SoA multires levels — and the co-fill regression it caused

`LevelData` was 9 `SparseGrid`s (density f32, cell flags, fine-coarse distance,
3× face_area, 3× face_coeff), each with its own dense map + pool. It is now one
shared `BlockMap<LevelBlockPtr>` per level plus a per-block payload, giving one
map lookup + offset math instead of the C++'s ~9 `_blockmap` arrays + 9
scattered allocations per block:

- **Metadata co-resident**: `LevelBlock` holds `cell_flags` + `dist_fine_coarse`
  (the solver's mask reads share cache lines).
- **Density lazy + contiguous**: the `density` field is `Option<NonNull<DensityBlock>>`,
  materialized only when a block is actually written (the halo/solver density
  stream stays contiguous — the first cut put density *in* the block and paid a
  16 KiB stride).
- **Faces lazy**: `face_area`/`face_coeff` are a `FaceBlock` behind
  `ensure_faces`, so the single-phase steps (7..19) don't pay their ~6× f32
  footprint; step 20 materializes them without a layout change.

The first cut (full SoA — density+flags+dfc in one 32 KiB block) was **2.05×
slower** on `set_refinement_map` (Rust 179.5 ms vs C++ 87.4 ms at 35,937
blocks). Root cause: `init_cell_flags` materializes *every* block's flags, and
the co-allocated `ensure_block` wrote 32 KiB/block (density+flags+dfc) where the
old code wrote 8 KiB (flags only) — 4× the writes. Making density a separate
lazy payload recovered parity: **98.6 ms vs C++ 94.1 ms** while still doing the
cell-flag init the C++ `benchmark.cpp` path skips (`doInitCellFlags=false`).

### 64-bit block ids

`bid` was `u32` in the solver/particles/topology signatures; it is `usize` now
(64-bit on every supported target). `morton3` was 10-bit-per-axis (`u32`,
overflow past 1024³); it is 21-bit-per-axis (`u64`) so the 2048³ paper domain's
11-bit block coords interleave bijectively. `BlockPool` was generalized to
`Pool<T>` (`BlockPool<D, BSX, N> = Pool<Block<D, BSX, N>>`) so the SoA
`LevelBlock`/`DensityBlock`/`FaceBlock` pools share one allocator.

### Remaining dense structures (not step 11)

`BlockInfoStore.level0` (1 B/block), `BlockInfoStore.flags` (2 B/level/block),
and `RefinementMap.levels` (1 B/block) are still dense — ~8.6 GB + ~17 GB at
8.59B blocks. They sit behind the step-12/14 multires solver and are the next
dense→sparse frontier.


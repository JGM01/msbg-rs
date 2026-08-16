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

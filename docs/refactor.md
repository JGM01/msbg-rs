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
  difftest_cpp.rs   // Differential testing against C++ baseline

```

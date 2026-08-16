//! Shared stencil infrastructure: fluid mask precomputation and the constants
//! shared by every PDE smoothing kernel.

use crate::blockpool::Block;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Mask, Simd};

/// Per-voxel cell flags that are *not* fluid. Matches the C++ `CELL_IS_FLUID_`
/// predicate (`msbg.h`): `!(cell & (CELL_SOLID | CELL_VOID))`, so `CELL_AIR`
/// counts as fluid.
pub const FLUID_MASK: u16 = 0x1 | 0x1000; // CELL_SOLID | CELL_VOID

/// Load `W` contiguous `f32` lanes from `base + off` without bounds checks.
///
/// # Safety
///
/// The caller must ensure `base[off..off + W]` is in-bounds of the halo
/// buffer. Every kernel proves this up front with `debug_assert_eq!` on the
/// halo size (`HSX*HSX*HSX`) and on the loop ranges (`BSX % W == 0`).
#[inline(always)]
pub unsafe fn load<const W: usize>(base: *const f32, off: usize) -> Simd<f32, W> {
    unsafe { Simd::from_slice(std::slice::from_raw_parts(base.add(off), W)) }
}

/// Store `W` contiguous `f32` lanes to `base + off` without bounds checks.
///
/// # Safety
///
/// See [`load`]; the output block is a fixed `[f32; N]` array so its bounds
/// are provable, and the store offset stays within `N`.
#[inline(always)]
pub unsafe fn store<const W: usize>(base: *mut f32, off: usize, v: Simd<f32, W>) {
    unsafe { v.copy_to_slice(std::slice::from_raw_parts_mut(base.add(off), W)) };
}

/// Non-temporal (streaming) store of `W` lanes to `base + off`.
///
/// The C++ smoothing kernels write their destination block with `vstream`
/// (`_mm256_stream_ps`, see `renderDensFromFloat_storeSimd8` in `sbg.h`), which
/// bypasses the cache and skips the write-allocate read-for-ownership. On this
/// machine (single-channel DDR4, ~15 GB/s effective) regular stores make the
/// parallel kernel bandwidth-bound: 16 KB written + 16 KB RFO read per block.
/// NT stores halve the traffic and roughly double parallel throughput.
///
/// # Safety
///
/// `base + off` must be 16/32/64-byte aligned for W = 4/8/16 respectively
/// (true: `Block` is 64-byte aligned and `off` is a multiple of `W`) and
/// in-bounds of the output block.
#[inline(always)]
pub unsafe fn store_nt<const W: usize>(base: *mut f32, off: usize, v: Simd<f32, W>) {
    unsafe {
        let p = base.add(off);
        #[cfg(target_arch = "x86_64")]
        {
            let a = v.to_array();
            #[cfg(target_feature = "avx512f")]
            if W == 16 {
                let m = std::arch::x86_64::_mm512_loadu_ps(a.as_ptr());
                std::arch::x86_64::_mm512_stream_ps(p, m);
                return;
            }
            if W == 8 {
                let m = std::arch::x86_64::_mm256_loadu_ps(a.as_ptr());
                std::arch::x86_64::_mm256_stream_ps(p, m);
                return;
            } else if W == 4 {
                let m = std::arch::x86_64::_mm_loadu_ps(a.as_ptr());
                std::arch::x86_64::_mm_stream_ps(p, m);
                return;
            }
        }
        v.copy_to_slice(std::slice::from_raw_parts_mut(p, W));
    }
}

/// Precomputed, SIMD-ready fluid mask for one block.
///
/// Decoupled from the math kernels: built once per block (per frame) from the
/// `CellFlags` channel and reused across iterations, so the hot loop stays
/// pure arithmetic with a single branchless select at the end.
///
/// `W` is the SIMD lane width and `CHUNKS == N / W` the number of chunks
/// (passed explicitly to avoid unstable `generic_const_exprs`).
pub struct MaskBlock<const W: usize, const CHUNKS: usize> {
    masks: [Mask<i32, W>; CHUNKS],
}

impl<const W: usize, const CHUNKS: usize> MaskBlock<W, CHUNKS> {
    /// Precompute the fluid mask from a `CellFlags` block.
    pub fn build<const BSX: usize, const N: usize>(flags: &Block<u16, BSX, N>) -> Self {
        debug_assert_eq!(CHUNKS * W, N, "CHUNKS * W must equal N");
        let mut masks = [Mask::<i32, W>::splat(false); CHUNKS];
        let mask_splat = Simd::<u16, W>::splat(FLUID_MASK);
        let zero = Simd::<u16, W>::splat(0);
        for (i, slot) in masks.iter_mut().enumerate() {
            let v = Simd::<u16, W>::from_slice(&flags.data[i * W..i * W + W]);
            let is_fluid = (v & mask_splat).simd_eq(zero);
            *slot = is_fluid.cast::<i32>();
        }
        Self { masks }
    }

    /// Mask for chunk `i` (in output/`flags` layout order).
    #[inline(always)]
    pub fn chunk(&self, i: usize) -> Mask<i32, W> {
        self.masks[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = crate::math::simd::LANES;
    const BSX: usize = 16;
    const N: usize = 4096;
    const CHUNKS: usize = N / W;

    #[test]
    fn test_msk_01_fluid_bits_match_cpp_predicate() {
        let mut flags = Block::<u16, BSX, N>::new();
        // solid, void, air, fluid
        flags.data[0] = 0x1; // CELL_SOLID -> not fluid
        flags.data[1] = 0x1000; // CELL_VOID -> not fluid
        flags.data[2] = 0x4; // CELL_AIR -> fluid (CELL_IS_FLUID_)
        flags.data[3] = 0x0; // fluid

        let m = MaskBlock::<W, CHUNKS>::build(&flags);

        // all three sample cells live in chunk 0
        let chunk = m.chunk(0);
        let bits = chunk.to_array();
        assert!(!bits[0]); // solid masked
        assert!(!bits[1]); // void masked
        assert!(bits[2]); // air is fluid
        assert!(bits[3]); // fluid
    }

    #[test]
    fn test_msk_02_all_fluid_when_zero() {
        let flags = Block::<u16, BSX, N>::new(); // all zeros = all fluid
        let m = MaskBlock::<W, CHUNKS>::build(&flags);
        for i in 0..CHUNKS {
            assert!(m.chunk(i).all());
        }
    }
}

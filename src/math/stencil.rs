//! Shared stencil infrastructure: raw SIMD loads and (non-)temporal stores, the
//! write-back `StoreBack` conversion, and the stochastic-rounding RNG used by
//! the `Density8` path.

use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{Select, Simd, StdFloat};

use crate::channel::{Density, Density8};

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

/// Store-only fence ordering non-temporal stores before later loads (x86).
///
/// After a block is written with streaming stores, a later pass's halo gather
/// re-reads it with regular cached loads. Streaming stores are weakly ordered,
/// so a fence is required before the data is observed by another core.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn store_fence() {
    unsafe { std::arch::x86_64::_mm_sfence() };
}

/// Full fence — the C++ baseline (`_mm_mfence()` per block).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn store_mfence() {
    unsafe { std::arch::x86_64::_mm_mfence() };
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn store_fence() {}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn store_mfence() {}

/// Lane-parallel PRNG for the `Density8` stochastic-rounding quantizer (a
/// murmur-hashed seed followed by an LCG, widened to `W` lanes).
pub struct SimdRng<const W: usize> {
    state: Simd<u32, W>,
}

impl<const W: usize> SimdRng<W> {
    /// Seed `W` lanes deterministically from a scalar seed (distinct lanes are
    /// decorrelated so a uniform scalar seed yields uniform lanes).
    pub fn seed(seed: u32) -> Self {
        let lane = Simd::<u32, W>::from_array(std::array::from_fn(|i| i as u32));
        let mut h = Simd::<u32, W>::splat(seed) ^ (lane * Simd::splat(0x9e37_79b9));
        h ^= h >> Simd::splat(16u32);
        h *= Simd::splat(0x85eb_ca6b);
        h ^= h >> Simd::splat(13u32);
        h *= Simd::splat(0xc2b2_ae35);
        h ^= h >> Simd::splat(16u32);
        let nonzero = h
            .simd_eq(Simd::splat(0u32))
            .select(Simd::splat(0xdead_beef), h);
        Self { state: nonzero }
    }

    /// Uniform `[0, 1)` in `W` lanes.
    #[inline(always)]
    pub fn next(&mut self) -> Simd<f32, W> {
        self.state *= Simd::splat(16807u32);
        let bits = (self.state >> Simd::splat(9u32)) | Simd::splat(0x3f80_0000);
        let f: Simd<f32, W> = Simd::from_bits(bits);
        f - Simd::splat(1.0f32)
    }
}

/// Converts a smoothed `Simd<f32, W>` chunk into a stored element and writes it
/// to `out + off` (element offset). The storage conversion is monomorphized
/// away: `f32` is a raw non-temporal store, `Density` is a round-to-`u16`
/// store, `Density8` is a sqrt-compress + stochastic-rounding `u8` store.
///
/// # Safety
///
/// `out` must point at a writable block payload of at least `off + W` elements.
pub trait StoreBack<const W: usize>: Copy {
    /// Whether the store is non-temporal (and therefore needs a store fence
    /// between color passes before another thread re-reads the block).
    const USES_NT: bool;

    /// Write one smoothed chunk. `rng` is only read by stochastic quantizers.
    unsafe fn store_chunk(out: *mut Self, off: usize, v: Simd<f32, W>, rng: &mut SimdRng<W>);
}

impl<const W: usize> StoreBack<W> for f32 {
    const USES_NT: bool = true;

    #[inline(always)]
    unsafe fn store_chunk(out: *mut f32, off: usize, v: Simd<f32, W>, _rng: &mut SimdRng<W>) {
        unsafe { store_nt::<W>(out, off, v) };
    }
}

impl<const W: usize> StoreBack<W> for Density {
    const USES_NT: bool = false;

    #[inline(always)]
    unsafe fn store_chunk(out: *mut Density, off: usize, v: Simd<f32, W>, _rng: &mut SimdRng<W>) {
        let scale = Simd::<f32, W>::splat(Density::MAX);
        let rounded = (v * scale).round();
        // SAFETY: `v` is in `[0, 1]`, so `rounded` is in `[0, 65535]`; the i32
        // conversion cannot overflow (`cvttps2dq`).
        let i: Simd<i32, W> = unsafe { rounded.to_int_unchecked() };
        let u: Simd<u16, W> = i.cast();
        let p = unsafe { (out as *mut u16).add(off) };
        unsafe { u.copy_to_slice(std::slice::from_raw_parts_mut(p, W)) };
    }
}

impl<const W: usize> StoreBack<W> for Density8 {
    const USES_NT: bool = false;

    #[inline(always)]
    unsafe fn store_chunk(out: *mut Density8, off: usize, v: Simd<f32, W>, rng: &mut SimdRng<W>) {
        let vf = v.sqrt() * Simd::splat(Density8::MAX);
        let floor = vf.floor();
        let frac = vf - floor;
        let u = rng.next();
        let rounded = u.simd_lt(frac).select(floor + Simd::splat(1.0), floor);
        // SAFETY: `rounded` is in `[0, 255]`.
        let i: Simd<i32, W> = unsafe { rounded.to_int_unchecked() };
        let u8s: Simd<u8, W> = i.cast();
        let p = unsafe { (out as *mut u8).add(off) };
        unsafe { u8s.copy_to_slice(std::slice::from_raw_parts_mut(p, W)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = crate::math::simd::LANES;

    #[test]
    fn test_rng_01_uniform_in_range() {
        let mut rng = SimdRng::<W>::seed(0x1234_5678);
        for _ in 0..100 {
            let v = rng.next();
            let arr = v.to_array();
            for &f in &arr {
                assert!((0.0..1.0).contains(&f), "rng out of range {f}");
            }
        }
    }

    #[test]
    fn test_rng_02_distinct_seeds_distinct_streams() {
        let mut a = SimdRng::<W>::seed(1);
        let mut b = SimdRng::<W>::seed(2);
        assert_ne!(a.next().to_array(), b.next().to_array());
    }
}

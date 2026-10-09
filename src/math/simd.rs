//! SIMD width selection by target feature.
//!
//! Kernels use a const lane count: 16 for AVX-512, 8 for AVX, and 4 for NEON/SSE2.
//! Matching the register width limits vector splitting and register pressure.

use std::simd::{Mask, Simd};

/// Number of `f32` lanes per native vector register.
///
/// * AVX-512: 16 (one ZMM)
/// * AVX2/AVX: 8 (one YMM)
/// * NEON/SSE2: 4 (one 128-bit register)
#[cfg(target_feature = "avx512f")]
pub const LANES: usize = 16;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub const LANES: usize = 8;

#[cfg(all(not(target_feature = "avx2"), not(target_feature = "avx512f")))]
pub const LANES: usize = 4;

/// Native-width `f32` vector for the current target.
pub type NativeSimd = Simd<f32, LANES>;

/// Native-width integer mask for `NativeSimd` selects.
pub type NativeMask = Mask<i32, LANES>;

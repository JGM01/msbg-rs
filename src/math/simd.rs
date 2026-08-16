//! Hardware-sympathetic SIMD lane-width selection.
//!
//! The stencil kernels are written once, generic over `const W: usize` lanes,
//! and instantiated at the width matching the target's *native* vector
//! register. Hardcoding `f32x16` everywhere forces LLVM to split the 16-lane
//! vectors (2x 256-bit on AVX2, 4x 128-bit on NEON) and inflates register
//! pressure in the 19/25-tap kernels, causing spills.

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

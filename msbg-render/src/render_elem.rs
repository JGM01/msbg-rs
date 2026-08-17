//! Storage-to-physical mapping for rendering.
//!
//! [`msbg_rs::math::Sampler`] returns the interpolated value in *storage
//! space* (linear for `Density`/`f32`). [`Density8`] is sqrt-compressed, so its
//! sampled value must be squared to recover a physical `[0, 1]` density —
//! exactly what the C++ render path does for 8-bit builds.

use msbg_rs::channel::{Density, Density8};
use msbg_rs::math::InterpElem;

/// Converts an interpolated storage-space sample into the physical `[0, 1]`
/// density the renderer shades and thresholds against the iso level.
pub trait RenderElem: InterpElem {
    fn physical(v: f32) -> f32;
}

impl RenderElem for f32 {
    #[inline(always)]
    fn physical(v: f32) -> f32 {
        v
    }
}

impl RenderElem for Density {
    #[inline(always)]
    fn physical(v: f32) -> f32 {
        v
    }
}

impl RenderElem for Density8 {
    #[inline(always)]
    fn physical(v: f32) -> f32 {
        v * v
    }
}

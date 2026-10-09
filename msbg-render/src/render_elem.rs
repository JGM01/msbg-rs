//! Conversion from sampled storage values to physical density.
//!
//! `Density` and `f32` samples are linear. `Density8` samples use sqrt space.
//! Square a `Density8` sample to recover physical density.

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

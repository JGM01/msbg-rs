//! Typed data channels over [`SparseGrid`](crate::sparse_grid::SparseGrid).
//!
//! A fluid simulation tracks many scalar and vector fields — density, pressure,
//! velocity, cell flags, and more. Rather than magic `int` channel ids, each
//! channel is its own type ([`Density`], [`Pressure`], [`Velocity`], ...), so a
//! field's element type is known at compile time and cross-channel mixing is a
//! compile error.
//!
//! [`ChannelTable`] owns one `SparseGrid` per channel with typed `get_*` /
//! `set_*` accessors; [`ChannelId`] and [`ChannelRef`] allow type-erased
//! iteration over the channels a table currently holds.
//!
//! ```compile_fail
//! use msbg_rs::channel::{ChannelTable, Density};
//! use msbg_rs::sparse_grid::SparseGrid;
//!
//! let t = ChannelTable::<16, 4096>::new();
//! // `get_pressure()` yields `SparseGrid<Pressure>`; ascribing it to
//! // `SparseGrid<Density>` is a type error.
//! let _: &SparseGrid<Density, 16, 4096> = t.get_pressure().unwrap();
//! ```

use crate::math::gather::Dequant;
use crate::sparse_grid::SparseGrid;
use std::mem::{align_of, size_of};
use std::ops::{Add, Mul, Sub};
use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt, SimdUint};
use std::simd::{f32x16, i32x16, u16x16, u8x16, Mask, Select, Simd, StdFloat};

/// Quantized scalar density, `u16` in `[0, 65535]`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Density(pub u16);

/// Per-voxel cell type flags (solid/air/void/fluid bitfield).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CellFlags(pub u16);

/// Scalar pressure field.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pressure(pub f32);

/// A 3-component float vector.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3(pub [f32; 3]);

/// Fluid velocity channel element.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Velocity(pub Vec3);

/// Quantized 8-bit density (`RSURF_8_BIT`), sqrt-compressed in storage.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Density8(pub u8);

/// Quantized per-face density (`FACE_DENSITY_16_BIT`): three `u16` components
/// in `[0, 65535]`, used by the multires phase-field solver's face areas.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FaceDensity(pub [u16; 3]);

// Solver/render channels: One newtype per channel to make the compiler
// reject cross-channel mixing (Pressure vs Divergence vs Curvature, etc).

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Divergence(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CgP(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CgQ(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Diagonal(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Curvature(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Heat(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DensityDiff(pub f32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DistFineCoarse(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VelocityAir(pub Vec3);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VelocityAvg(pub Vec3);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MassDensity(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CellFlagsTmp(pub u16);

impl Density {
    /// Full-scale value used by the quantizer (`u16::MAX`).
    pub const MAX: f32 = u16::MAX as f32;

    /// Decode to linear `[0, 1]` float.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        // Multiply by the reciprocal (not divide) to match C++
        // `renderDensToFloat_` and the SIMD batch.
        self.0 as f32 * (1.0 / Self::MAX)
    }

    /// Decode a sqrt-compressed value: `(v/max)^2`.
    #[inline(always)]
    pub fn to_f32_sqrt(self) -> f32 {
        let f = self.to_f32();
        f * f
    }

    /// Quantize a `[0, 1]` float (nearest, half away from zero).
    #[inline(always)]
    pub fn from_f32(f: f32) -> Self {
        debug_assert!(f.is_finite() && (0.0..=1.0).contains(&f));
        Density((f * Self::MAX).round() as u16)
    }

    /// Quantize with stochastic rounding: `rand` is a uniform `[0, 1)` draw.
    #[inline(always)]
    pub fn from_f32_sr(f: f32, rand: f32) -> Self {
        debug_assert!(f.is_finite() && (0.0..=1.0).contains(&f));
        debug_assert!((0.0..1.0).contains(&rand));
        let v = f * Self::MAX;
        let floor = v.floor();
        Density(if rand < v - floor { floor + 1.0 } else { floor } as u16)
    }
}

impl Density8 {
    /// Full-scale value used by the quantizer (`u8::MAX`).
    pub const MAX: f32 = u8::MAX as f32;

    /// Linear decode (u8 storage actually uses sqrt compression; see below).
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 * (1.0 / Self::MAX)
    }

    /// Sqrt-compression decode: `(v/255)^2`
    #[inline(always)]
    pub fn to_f32_sqrt(self) -> f32 {
        let f = self.to_f32();
        f * f
    }

    /// Sqrt-compression encode: `round(sqrt(f) * 255)`
    /// (C++ applies `doSqrtCompr` only for `sizeof(T)==1`).
    #[inline(always)]
    pub fn from_f32_sqrt(f: f32) -> Self {
        debug_assert!(f.is_finite() && (0.0..=1.0).contains(&f));
        Density8((f.sqrt() * Self::MAX).round() as u8)
    }

    /// Stochastic sqrt-compression encode.
    #[inline(always)]
    pub fn from_f32_sqrt_sr(f: f32, rand: f32) -> Self {
        debug_assert!(f.is_finite() && (0.0..=1.0).contains(&f));
        debug_assert!((0.0..1.0).contains(&rand));
        let v = f.sqrt() * Self::MAX;
        let floor = v.floor();
        Density8(if rand < v - floor { floor + 1.0 } else { floor } as u8)
    }
}

/// Storage-side quantization contract of the surface-reconstruction pipeline,
/// mirroring C++ `renderDensFromFloat_` / `renderDensToFloat_` with default
/// flags (`doSqrtCompr=false`, `doSR=false`): a plain round on the
/// splat/finalize path for every storage width. Sqrt compression only enters
/// through the smoother's [`StoreBack`](crate::math::stencil::StoreBack) for
/// the 8-bit build.
///
/// `dequant_chunk` / `quant_chunk` are the SIMD batch forms used by the
/// finalize pass; `from_unit` is the scalar write form used by the splat.
pub trait Quant: Copy + Default + PartialEq + Send + Sync + Dequant<f32> {
    /// Quantize a `[0, 1]` float to storage (plain round).
    fn from_unit(f: f32) -> Self;
    /// `renderDensFromFloat(1.0)` — the untouched-voxel sentinel.
    fn full() -> Self;
    /// `renderDensFromFloat(0.0)`.
    fn empty() -> Self;
    /// Element-wise min (the splat's reduction over overlapping particles).
    fn min_of(self, o: Self) -> Self;

    /// SIMD-dequantize `W` contiguous elements to `f32` lanes.
    ///
    /// # Safety
    ///
    /// `src + off..off+W` must be in-bounds.
    unsafe fn dequant_chunk<const W: usize>(src: *const Self, off: usize) -> Simd<f32, W>;

    /// SIMD-quantize `W` `f32` lanes (plain round) into `dst`.
    ///
    /// # Safety
    ///
    /// `dst + off..off+W` must be in-bounds and the lanes in `[0, 1]`.
    unsafe fn quant_chunk<const W: usize>(dst: *mut Self, off: usize, v: Simd<f32, W>);

    /// SIMD `min`-commit: `dst[i] = min(dst[i], src[i])` for `W` lanes, skipping
    /// entirely when every source lane is `full()` (untouched — the grid value
    /// cannot decrease). One cache-line miss amortized over `W` voxels.
    ///
    /// # Safety
    ///
    /// `src`/`dst` must point at `W` in-bounds elements.
    unsafe fn commit_chunk<const W: usize>(src: *const Self, dst: *mut Self);

    /// SIMD stage: quantize `dist_sq * inv` (plain round) to `Self` lanes and
    /// take the element-wise `min` into `dst + off` for lanes that are both
    /// `in_run` and within `dist_sq_max`. Lanes outside the mask keep `dst`.
    /// The splat's hot loop — one L1 hit + quantize per `W` voxels.
    ///
    /// # Safety
    ///
    /// `dst + off..off+W` must be in-bounds.
    unsafe fn stage_chunk<const W: usize>(
        dst: *mut Self,
        off: usize,
        dist_sq: Simd<f32, W>,
        dist_sq_max: f32,
        inv: f32,
        in_run: Mask<i32, W>,
    );
}

impl Quant for f32 {
    #[inline(always)]
    fn from_unit(f: f32) -> Self {
        f
    }
    #[inline(always)]
    fn full() -> Self {
        1.0
    }
    #[inline(always)]
    fn empty() -> Self {
        0.0
    }
    #[inline(always)]
    fn min_of(self, o: Self) -> Self {
        self.min(o)
    }
    #[inline(always)]
    unsafe fn dequant_chunk<const W: usize>(src: *const f32, off: usize) -> Simd<f32, W> {
        unsafe { Simd::from_slice(std::slice::from_raw_parts(src.add(off), W)) }
    }
    #[inline(always)]
    unsafe fn quant_chunk<const W: usize>(dst: *mut f32, off: usize, v: Simd<f32, W>) {
        unsafe { v.copy_to_slice(std::slice::from_raw_parts_mut(dst.add(off), W)) };
    }
    #[inline(always)]
    unsafe fn commit_chunk<const W: usize>(src: *const f32, dst: *mut f32) {
        let s = unsafe { Simd::<f32, W>::from_slice(std::slice::from_raw_parts(src, W)) };
        let full = Simd::splat(1.0);
        if s.simd_eq(full).all() {
            return;
        }
        let d = unsafe { Simd::<f32, W>::from_slice(std::slice::from_raw_parts(dst, W)) };
        unsafe { d.simd_min(s).copy_to_slice(std::slice::from_raw_parts_mut(dst, W)) };
    }
    #[inline(always)]
    unsafe fn stage_chunk<const W: usize>(
        dst: *mut f32,
        off: usize,
        dist_sq: Simd<f32, W>,
        dist_sq_max: f32,
        inv: f32,
        in_run: Mask<i32, W>,
    ) {
        let m = in_run & dist_sq.simd_le(Simd::splat(dist_sq_max));
        let v = dist_sq * Simd::splat(inv);
        let old = unsafe { Simd::<f32, W>::from_slice(std::slice::from_raw_parts(dst.add(off), W)) };
        let new = m.select(v.simd_min(old), old);
        unsafe { new.copy_to_slice(std::slice::from_raw_parts_mut(dst.add(off), W)) };
    }
}

impl Quant for Density {
    #[inline(always)]
    fn from_unit(f: f32) -> Self {
        Density::from_f32(f)
    }
    #[inline(always)]
    fn full() -> Self {
        Density(u16::MAX)
    }
    #[inline(always)]
    fn empty() -> Self {
        Density(0)
    }
    #[inline(always)]
    fn min_of(self, o: Self) -> Self {
        Density(self.0.min(o.0))
    }
    #[inline(always)]
    unsafe fn dequant_chunk<const W: usize>(src: *const Density, off: usize) -> Simd<f32, W> {
        let u = Simd::<u16, W>::from_slice(unsafe {
            std::slice::from_raw_parts((src as *const u16).add(off), W)
        });
        let f: Simd<f32, W> = u.cast();
        f * Simd::splat(1.0 / u16::MAX as f32)
    }
    #[inline(always)]
    unsafe fn quant_chunk<const W: usize>(dst: *mut Density, off: usize, v: Simd<f32, W>) {
        let rounded = (v * Simd::splat(u16::MAX as f32)).round();
        // SAFETY: lanes are in [0, 1], so `rounded` is in [0, 65535].
        let i: Simd<i32, W> = unsafe { rounded.to_int_unchecked() };
        let u: Simd<u16, W> = i.cast();
        unsafe { u.copy_to_slice(std::slice::from_raw_parts_mut((dst as *mut u16).add(off), W)) };
    }
    #[inline(always)]
    unsafe fn commit_chunk<const W: usize>(src: *const Density, dst: *mut Density) {
        let s = Simd::<u16, W>::from_slice(unsafe {
            std::slice::from_raw_parts(src as *const u16, W)
        });
        if s.simd_eq(Simd::splat(u16::MAX)).all() {
            return;
        }
        let d = Simd::<u16, W>::from_slice(unsafe {
            std::slice::from_raw_parts(dst as *const u16, W)
        });
        unsafe {
            d.simd_min(s)
                .copy_to_slice(std::slice::from_raw_parts_mut(dst as *mut u16, W))
        };
    }
    #[inline(always)]
    unsafe fn stage_chunk<const W: usize>(
        dst: *mut Density,
        off: usize,
        dist_sq: Simd<f32, W>,
        dist_sq_max: f32,
        inv: f32,
        in_run: Mask<i32, W>,
    ) {
        let m = in_run & dist_sq.simd_le(Simd::splat(dist_sq_max));
        let t = dist_sq * Simd::splat(inv);
        let v = (t * Simd::splat(u16::MAX as f32)).round();
        // SAFETY: in-run lanes are <= dist_sq_max; out-of-run lanes are bounded
        // by the staging window (~1728 max), far below i32::MAX.
        let i: Simd<i32, W> = unsafe { v.to_int_unchecked() };
        let val: Simd<u16, W> = i.cast();
        let old = Simd::<u16, W>::from_slice(unsafe {
            std::slice::from_raw_parts((dst as *const u16).add(off), W)
        });
        let mu: Mask<i16, W> = m.cast();
        let new = mu.select(val.simd_min(old), old);
        unsafe { new.copy_to_slice(std::slice::from_raw_parts_mut((dst as *mut u16).add(off), W)) };
    }
}

impl Quant for Density8 {
    #[inline(always)]
    fn from_unit(f: f32) -> Self {
        debug_assert!(f.is_finite() && (0.0..=1.0).contains(&f));
        Density8((f * Self::MAX).round() as u8)
    }
    #[inline(always)]
    fn full() -> Self {
        Density8(u8::MAX)
    }
    #[inline(always)]
    fn empty() -> Self {
        Density8(0)
    }
    #[inline(always)]
    fn min_of(self, o: Self) -> Self {
        Density8(self.0.min(o.0))
    }
    #[inline(always)]
    unsafe fn dequant_chunk<const W: usize>(src: *const Density8, off: usize) -> Simd<f32, W> {
        let u = Simd::<u8, W>::from_slice(unsafe {
            std::slice::from_raw_parts((src as *const u8).add(off), W)
        });
        let f: Simd<f32, W> = u.cast();
        f * Simd::splat(1.0 / u8::MAX as f32)
    }
    #[inline(always)]
    unsafe fn quant_chunk<const W: usize>(dst: *mut Density8, off: usize, v: Simd<f32, W>) {
        let rounded = (v * Simd::splat(u8::MAX as f32)).round();
        // SAFETY: lanes are in [0, 1], so `rounded` is in [0, 255].
        let i: Simd<i32, W> = unsafe { rounded.to_int_unchecked() };
        let u: Simd<u8, W> = i.cast();
        unsafe { u.copy_to_slice(std::slice::from_raw_parts_mut((dst as *mut u8).add(off), W)) };
    }
    #[inline(always)]
    unsafe fn commit_chunk<const W: usize>(src: *const Density8, dst: *mut Density8) {
        let s = Simd::<u8, W>::from_slice(unsafe {
            std::slice::from_raw_parts(src as *const u8, W)
        });
        if s.simd_eq(Simd::splat(u8::MAX)).all() {
            return;
        }
        let d = Simd::<u8, W>::from_slice(unsafe {
            std::slice::from_raw_parts(dst as *const u8, W)
        });
        unsafe {
            d.simd_min(s)
                .copy_to_slice(std::slice::from_raw_parts_mut(dst as *mut u8, W))
        };
    }
    #[inline(always)]
    unsafe fn stage_chunk<const W: usize>(
        dst: *mut Density8,
        off: usize,
        dist_sq: Simd<f32, W>,
        dist_sq_max: f32,
        inv: f32,
        in_run: Mask<i32, W>,
    ) {
        let m = in_run & dist_sq.simd_le(Simd::splat(dist_sq_max));
        let t = dist_sq * Simd::splat(inv);
        let v = (t * Simd::splat(u8::MAX as f32)).round();
        let i: Simd<i32, W> = unsafe { v.to_int_unchecked() };
        let val: Simd<u8, W> = i.cast();
        let old = Simd::<u8, W>::from_slice(unsafe {
            std::slice::from_raw_parts((dst as *const u8).add(off), W)
        });
        let mu: Mask<i8, W> = m.cast();
        let new = mu.select(val.simd_min(old), old);
        unsafe { new.copy_to_slice(std::slice::from_raw_parts_mut((dst as *mut u8).add(off), W)) };
    }
}

/// Batch dequantize: `Density` (u16) -> `f32` (16-wide SIMD + scalar tail).
pub fn dequantize_density(src: &[Density], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    debug_assert_eq!(size_of::<Density>(), size_of::<u16>());
    debug_assert_eq!(align_of::<Density>(), align_of::<u16>());
    // SAFETY: `Density` is `#[repr(transparent)]` over `u16`, so the slices
    // alias memory with identical layout.
    let src = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, src.len()) };
    let n = src.len();
    let scale = f32x16::splat(1.0 / Density::MAX);
    let mut i = 0;
    while i + 16 <= n {
        let u = u16x16::from_slice(&src[i..i + 16]);
        let f: f32x16 = u.cast();
        (f * scale).copy_to_slice(&mut dst[i..i + 16]);
        i += 16;
    }
    for j in i..n {
        dst[j] = src[j] as f32 * (1.0 / Density::MAX);
    }
}

/// Batch quantize: `f32` -> `Density` (u16) (16-wide SIMD + scalar tail).
pub fn quantize_density(src: &[f32], dst: &mut [Density]) {
    assert_eq!(src.len(), dst.len());
    debug_assert_eq!(size_of::<Density>(), size_of::<u16>());
    debug_assert_eq!(align_of::<Density>(), align_of::<u16>());
    // SAFETY: as above — `Density` and `u16` share layout.
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u16, dst.len()) };
    let n = src.len();
    let scale = f32x16::splat(Density::MAX);
    let mut i = 0;
    while i + 16 <= n {
        let f = f32x16::from_slice(&src[i..i + 16]);
        let rounded = (f * scale).round();
        // SAFETY: `rounded` is in `[0, 65535]`, so the i32 conversion cannot
        // overflow. `to_int_unchecked` lowers to `cvttps2dq`; the `as`-semantics
        // `cast` (saturating) scalarizes on the baseline target.
        let i32s: i32x16 = unsafe { rounded.to_int_unchecked::<i32>() };
        let u: u16x16 = i32s.cast();
        u.copy_to_slice(&mut dst[i..i + 16]);
        i += 16;
    }
    for j in i..n {
        dst[j] = Density::from_f32(src[j]).0;
    }
}

/// Batch dequantize: `Density8` (u8) -> `f32` (16-wide SIMD + scalar tail).
/// Linear decode only; sqrt decompression is the caller's job (matches
/// `renderDensToFloat_` with `doSqrtCompr=false`).
pub fn dequantize_density8(src: &[Density8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    debug_assert_eq!(size_of::<Density8>(), size_of::<u8>());
    debug_assert_eq!(align_of::<Density8>(), align_of::<u8>());
    // SAFETY: `Density8` is `#[repr(transparent)]` over `u8`.
    let src = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len()) };
    let n = src.len();
    let scale = f32x16::splat(1.0 / Density8::MAX);
    let mut i = 0;
    while i + 16 <= n {
        let u = u8x16::from_slice(&src[i..i + 16]);
        let f: f32x16 = u.cast();
        (f * scale).copy_to_slice(&mut dst[i..i + 16]);
        i += 16;
    }
    for j in i..n {
        dst[j] = src[j] as f32 * (1.0 / Density8::MAX);
    }
}

/// Batch quantize: `f32` -> `Density8` (u8) (16-wide SIMD + scalar tail).
/// Plain round, matching `renderDensFromFloat_` with `doSqrtCompr=false`.
pub fn quantize_density8(src: &[f32], dst: &mut [Density8]) {
    assert_eq!(src.len(), dst.len());
    debug_assert_eq!(size_of::<Density8>(), size_of::<u8>());
    debug_assert_eq!(align_of::<Density8>(), align_of::<u8>());
    // SAFETY: as above — `Density8` and `u8` share layout.
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len()) };
    let n = src.len();
    let scale = f32x16::splat(Density8::MAX);
    let mut i = 0;
    while i + 16 <= n {
        let f = f32x16::from_slice(&src[i..i + 16]);
        let rounded = (f * scale).round();
        // SAFETY: `rounded` is in `[0, 255]`; `to_int_unchecked` lowers to
        // `cvttps2dq` and the `as`-semantics `cast` would scalarize.
        let i32s: i32x16 = unsafe { rounded.to_int_unchecked::<i32>() };
        let u: u8x16 = i32s.cast();
        u.copy_to_slice(&mut dst[i..i + 16]);
        i += 16;
    }
    for j in i..n {
        dst[j] = Density8::from_unit(src[j]).0;
    }
}

impl Vec3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3([x, y, z])
    }

    #[inline(always)]
    pub fn x(self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn y(self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn z(self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn dot(self, o: Vec3) -> f32 {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }

    #[inline(always)]
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn add(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec3 {
        Vec3([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }
}

/// Declares the typed channel table. Each `field: Type` line adds:
/// a `ChannelId` variant, a table field, `get_/get_*_mut/set_*` accessors, and
/// a `ChannelRef`/`ChannelRefMut` variant. The element type must be a single
/// identifier (the newtypes above are).
#[rustfmt::skip]
macro_rules! channel_table {
    (
        $(
            $field:ident : $ty:ident,
        )+
    ) => {
        /// Runtime identifier of a channel (for iteration, reset, debug).
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum ChannelId {
            $(
                $ty,
            )+
        }

        impl ChannelId {
            pub fn name(self) -> &'static str {
                match self {
                    $(
                        Self::$ty => stringify!($ty),
                    )+
                }
            }
        }

        /// Borrowed view of one channel, type-erased to an enum for iteration.
        pub enum ChannelRef<'a, const BSX: usize, const N: usize> {
            $(
                $ty(&'a SparseGrid<$ty, BSX, N>),
            )+
        }

        /// Mutable borrowed view of one channel.
        pub enum ChannelRefMut<'a, const BSX: usize, const N: usize> {
            $(
                $ty(&'a mut SparseGrid<$ty, BSX, N>),
            )+
        }

        impl<'a, const BSX: usize, const N: usize> ChannelRef<'a, BSX, N> {
            pub fn id(&self) -> ChannelId {
                match self {
                    $(
                        Self::$ty(_) => ChannelId::$ty,
                    )+
                }
            }
        }

        impl<'a, const BSX: usize, const N: usize> ChannelRefMut<'a, BSX, N> {
            pub fn id(&self) -> ChannelId {
                match self {
                    $(
                        Self::$ty(_) => ChannelId::$ty,
                    )+
                }
            }
        }

        /// A typed, single-level channel table. Owns one `SparseGrid` per
        /// channel; grids are built externally and inserted with `set_*`.
        pub struct ChannelTable<const BSX: usize, const N: usize> {
            $(
                $field: Option<SparseGrid<$ty, BSX, N>>,
            )+
        }

        impl<const BSX: usize, const N: usize> ChannelTable<BSX, N> {
            pub fn new() -> Self {
                Self {
                    $(
                        $field: None,
                    )+
                }
            }

            $(
                paste::paste! {
                    pub fn [<get_ $field>](&self) -> Option<&SparseGrid<$ty, BSX, N>> {
                        self.$field.as_ref()
                    }

                    pub fn [<get_ $field _mut>](&mut self) -> Option<&mut SparseGrid<$ty, BSX, N>> {
                        self.$field.as_mut()
                    }

                    pub fn [<set_ $field>](&mut self, grid: SparseGrid<$ty, BSX, N>) {
                        self.$field = Some(grid);
                    }
                }
            )+

            pub fn get(&self, id: ChannelId) -> Option<ChannelRef<'_, BSX, N>> {
                match id {
                    $(
                        ChannelId::$ty => self.$field.as_ref().map(ChannelRef::$ty),
                    )+
                }
            }

            pub fn get_mut(&mut self, id: ChannelId) -> Option<ChannelRefMut<'_, BSX, N>> {
                match id {
                    $(
                        ChannelId::$ty => self.$field.as_mut().map(ChannelRefMut::$ty),
                    )+
                }
            }

            pub fn contains(&self, id: ChannelId) -> bool {
                match id {
                    $(
                        ChannelId::$ty => self.$field.is_some(),
                    )+
                }
            }

            pub fn len(&self) -> usize {
                0usize $( + usize::from(self.$field.is_some()) )+
            }

            pub fn is_empty(&self) -> bool {
                $( self.$field.is_none() )&&+
            }

            pub fn remove(&mut self, id: ChannelId) {
                match id {
                    $(
                        ChannelId::$ty => self.$field = None,
                    )+
                }
            }

            pub fn clear(&mut self) {
                $(
                    self.$field = None;
                )+
            }

            pub fn for_each<'a>(&'a self, mut f: impl FnMut(ChannelRef<'a, BSX, N>)) {
                $(
                    if let Some(g) = self.$field.as_ref() {
                        f(ChannelRef::$ty(g));
                    }
                )+
            }

            pub fn for_each_mut<'a>(&'a mut self, mut f: impl FnMut(ChannelRefMut<'a, BSX, N>)) {
                $(
                    if let Some(g) = self.$field.as_mut() {
                        f(ChannelRefMut::$ty(g));
                    }
                )+
            }

            pub fn iter<'a>(&'a self) -> impl Iterator<Item = ChannelRef<'a, BSX, N>> + 'a {
                let mut out = Vec::new();
                self.for_each(|c| out.push(c));
                out.into_iter()
            }

            pub fn iter_mut<'a>(
                &'a mut self,
            ) -> impl Iterator<Item = ChannelRefMut<'a, BSX, N>> + 'a {
                let mut out = Vec::new();
                self.for_each_mut(|c| out.push(c));
                out.into_iter()
            }
        }

        impl<const BSX: usize, const N: usize> Default for ChannelTable<BSX, N> {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

channel_table! {
    density:          Density,
    density8:         Density8,
    face_density:     FaceDensity,
    velocity:         Velocity,
    pressure:         Pressure,
    cell_flags:       CellFlags,
    divergence:       Divergence,
    cg_p:             CgP,
    cg_q:             CgQ,
    diagonal:         Diagonal,
    curvature:        Curvature,
    heat:             Heat,
    density_diff:     DensityDiff,
    dist_fine_coarse: DistFineCoarse,
    velocity_air:     VelocityAir,
    velocity_avg:     VelocityAvg,
    mass_density:     MassDensity,
    cell_flags_tmp:   CellFlagsTmp,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn grid<D: Copy + Default + Send + Sync>(
        name: &str,
        empty: D,
        full: D,
    ) -> SparseGrid<D, BSX, N> {
        let pool = Arc::new(BlockPool::<D, BSX, N>::new(16, 16));
        SparseGrid::new(name.to_string(), 32, 32, 32, empty, full, pool)
    }

    // dequant endpoints must map exactly.
    #[test]
    fn test_density_01_endpoints() {
        assert_eq!(Density(0).to_f32(), 0.0);
        assert_eq!(Density(u16::MAX).to_f32(), 1.0);
    }

    // Round-trip at awkward midpoints, within one quantization step.
    #[test]
    fn test_density_02_roundtrip() {
        assert_eq!(Density::from_f32(0.0), Density(0));
        assert_eq!(Density::from_f32(1.0), Density(u16::MAX));
        let v = Density::from_f32(0.5);
        assert!((v.to_f32() - 0.5).abs() < 1.0 / Density::MAX);
    }

    // sqrt-compression decode is the square of the linear decode.
    #[test]
    fn test_density_03_sqrt_decode() {
        let v = Density::from_f32(0.5).to_f32_sqrt();
        assert!((v - 0.25).abs() < 1e-3);
    }

    // Stochastic rounding: small `rand` rounds up, large `rand` rounds down.
    #[test]
    fn test_density_04_stochastic_round() {
        assert_eq!(Density::from_f32_sr(0.5, 0.0).0, 32768);
        assert_eq!(Density::from_f32_sr(0.5, 0.9).0, 32767);
    }

    // Out-of-range input must trip the debug assertion.
    #[test]
    #[should_panic]
    fn test_density_05_out_of_range_panics() {
        let _ = Density::from_f32(1.5);
    }

    // Awkward non-axis-aligned vector: len/dot/add/scale.
    #[test]
    fn test_vec3_01_non_axis_ops() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert_eq!(v.dot(v), 9.0);
        assert_eq!(v.len(), 3.0);
        assert_eq!(v + v, Vec3::new(2.0, 4.0, 4.0));
        assert_eq!(v * 2.0, Vec3::new(2.0, 4.0, 4.0));
    }

    // nothing present.
    #[test]
    fn test_table_01_empty() {
        let t = ChannelTable::<BSX, N>::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(!t.contains(ChannelId::Density));
        assert!(t.get_density().is_none());
    }

    // Set/remove one channel flips presence without touching others.
    #[test]
    fn test_table_02_set_remove() {
        let mut t = ChannelTable::<BSX, N>::new();
        t.set_density(grid("d", Density(0), Density(u16::MAX)));
        assert!(t.contains(ChannelId::Density));
        assert_eq!(t.len(), 1);
        assert!(t.get_pressure().is_none());
        t.remove(ChannelId::Density);
        assert!(t.is_empty());
    }

    // for_each visits exactly the set channels, with the right identity.
    #[test]
    fn test_table_03_for_each_visits_set() {
        let mut t = ChannelTable::<BSX, N>::new();
        t.set_density(grid("d", Density(0), Density(u16::MAX)));
        t.set_cell_flags(grid("cf", CellFlags(0), CellFlags(1)));

        let d_ptr = t.get_density().unwrap() as *const _;
        let mut seen = Vec::new();
        t.for_each(|c| {
            if c.id() == ChannelId::Density {
                assert_eq!(c.id(), ChannelId::Density);
                assert!(matches!(c, ChannelRef::Density(_)));
                // Same grid object we inserted.
                assert!(std::ptr::eq(
                    match c {
                        ChannelRef::Density(g) => g as *const _,
                        _ => unreachable!(),
                    },
                    d_ptr,
                ));
            }
            seen.push(c.id());
        });
        seen.sort_by_key(|id| id.name());
        assert_eq!(seen, vec![ChannelId::CellFlags, ChannelId::Density]);
    }

    // Mutation through get_*_mut is visible through get_*.
    #[test]
    fn test_table_04_get_mut_mutates() {
        let mut t = ChannelTable::<BSX, N>::new();
        t.set_pressure(grid("p", Pressure(0.0), Pressure(1.0)));
        t.get_pressure_mut().unwrap().set_voxel(3, 3, 3, Pressure(7.0));
        assert_eq!(t.get_pressure().unwrap().get_voxel(3, 3, 3), Pressure(7.0));
    }

    // Batch dequant matches the scalar path, incl. non-multiple-of-16 tails.
    #[test]
    fn test_batch_01_dequant_matches_scalar_odd_lengths() {
        for n in [0usize, 1, 15, 16, 17, 4095, 4096] {
            let src: Vec<Density> = (0..n).map(|i| Density((i * 26_543 + 1) as u16)).collect();
            let mut got = vec![0.0f32; n];
            dequantize_density(&src, &mut got);
            for (i, d) in src.iter().enumerate() {
                assert_eq!(got[i], d.to_f32(), "dequant mismatch at {i}/{n}");
            }
        }
    }

    // Batch quantize matches the scalar path, incl. non-multiple-of-16 tails.
    #[test]
    fn test_batch_02_quantize_matches_scalar_odd_lengths() {
        for n in [0usize, 1, 15, 16, 17, 4095, 4096] {
            let src: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 / 999.0).collect();
            let mut got = vec![Density(0); n];
            quantize_density(&src, &mut got);
            for (i, f) in src.iter().enumerate() {
                assert_eq!(got[i], Density::from_f32(*f), "quantize mismatch at {i}/{n}");
            }
        }
    }

    // Mismatched src/dst lengths must panic.
    #[test]
    #[should_panic]
    fn test_batch_03_length_mismatch_panics() {
        let src = [Density(0); 16];
        let mut dst = [0.0f32; 15];
        dequantize_density(&src, &mut dst);
    }

    // Density8 endpoints must map exactly.
    #[test]
    fn test_density8_01_endpoints() {
        assert_eq!(Density8(0).to_f32_sqrt(), 0.0);
        assert_eq!(Density8(u8::MAX).to_f32_sqrt(), 1.0);
    }

    // u8 sqrt encode/decode round-trips within one quantization step.
    #[test]
    fn test_density8_02_sqrt_roundtrip() {
        for f in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let back = Density8::from_f32_sqrt(f).to_f32_sqrt();
            assert!((back - f).abs() < 1.0 / Density8::MAX, "sqrt roundtrip {f} -> {back}");
        }
    }
}

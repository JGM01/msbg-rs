//! Field sampling over a [`SparseGrid`](crate::sparse_grid::SparseGrid):
//! trilinear and cubic B-spline interpolation with analytic gradient (and
//! Hessian), plus a vector-field sampler.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use msbg_rs::blockpool::BlockPool;
//! use msbg_rs::channel::{Density, Vec3};
//! use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sampler};
//! use msbg_rs::sparse_grid::SparseGrid;
//!
//! let pool = Arc::new(BlockPool::<Density, 16, 4096>::new(1, 16));
//! let grid = SparseGrid::new("density".into(), 32, 32, 32, Density(0), Density(u16::MAX), pool);
//! let sampler = Sampler::new(&grid, GridAlignment::Corner, BoundaryCondition::Clamp);
//!
//! let pos = Vec3::new(3.2, 7.7, 1.5);
//! let density = sampler.sample::<{ Interpolation::Linear }>(pos);
//! let normal = sampler.gradient::<{ Interpolation::CubicBSpline }>(pos);
//! ```
//!
//! # Dequantization
//!
//! Samples are linearized to `f32` before interpolation (`u16` -> `[0, 1]`,
//! `u8` -> `[0, 1]`). [`Density8`](crate::channel::Density8) is stored
//! sqrt-compressed, so sampling it returns the interpolated sqrt-space value;
//! square it to recover a physical density.

use std::simd::f32x4;
use std::simd::num::SimdFloat;
use std::simd::StdFloat;

use crate::channel::{Density, Density8, FaceDensity, Vec3, Velocity};
use crate::math::boundary::{BoundaryCondition, GridAlignment, Interpolation};
use crate::math::bspline::{cubic_deriv2_weights, cubic_deriv_weights, cubic_weights};
use crate::math::gather::{gather_map, Dequant};
use crate::sparse_grid::SparseGrid;

/// Element type of a scalar channel: dequantizes to `f32`.
pub trait InterpElem: Dequant<f32> {}

impl<T: Dequant<f32>> InterpElem for T {}

impl Dequant<f32> for f32 {
    #[inline(always)]
    fn dequant(self) -> f32 {
        self
    }

    #[inline(always)]
    fn copy_row(src: *const f32, dst: *mut f32, n: usize) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, n) };
    }
}

impl Dequant<f32> for Density {
    #[inline(always)]
    fn dequant(self) -> f32 {
        self.0 as f32 * (1.0 / u16::MAX as f32)
    }
}

impl Dequant<f32> for Density8 {
    #[inline(always)]
    fn dequant(self) -> f32 {
        // Linear only; sqrt decompression is the caller's job.
        self.0 as f32 * (1.0 / u8::MAX as f32)
    }
}

/// Element type of a vector channel (e.g. `Velocity`): dequantizes to `Vec3`.
pub trait InterpVec3Elem: Dequant<Vec3> {}

impl<T: Dequant<Vec3>> InterpVec3Elem for T {}

impl Dequant<Vec3> for Velocity {
    #[inline(always)]
    fn dequant(self) -> Vec3 {
        self.0
    }
}

impl Dequant<Vec3> for FaceDensity {
    #[inline(always)]
    fn dequant(self) -> Vec3 {
        let inv = 1.0 / u16::MAX as f32;
        Vec3([self.0[0] as f32 * inv, self.0[1] as f32 * inv, self.0[2] as f32 * inv])
    }
}

// SIMD gather targets: `dequant` packs the (x, y, z) triplet into an `f32x4`
// (4th lane zero) so `gather_map` fills a `[f32x4; N]` directly — no
// intermediate `[Vec3; N]` AoS array and no per-corner `pack3` step.

impl Dequant<f32x4> for Velocity {
    #[inline(always)]
    fn dequant(self) -> f32x4 {
        pack3(self.0)
    }

    /// `Velocity` is `#[repr(transparent)]` over `[f32; 3]` (12 bytes); issue a
    /// single 16-byte load whose 4th lane reads the *next* 4 bytes — for every
    /// element except the last that's the successor element's first float, and
    /// for the last it's the block's zeroed `flags`/`_pad` tail. Either way the
    /// 4th lane is discarded by the trilinear, so the value is irrelevant. This
    /// matches C++ `Vec4f::load` on the 12-byte `Vec3Float` and removes the
    /// 3× `movss`+insert of `pack3`.
    ///
    /// # Safety
    ///
    /// Unconditional: every `src` passed here is the `data` field of a
    /// `Block<Velocity, BSX, N>`, which is *immediately followed* by `flags:
    /// u16` + `_pad: [u8; 62]` — 64 bytes of initialized tail. A 16-byte load at
    /// `3*idx` reads bytes `[12*idx, 12*idx+16)`; for the last element
    /// (`idx = len-1`) the upper bound is `12*(len-1)+16 = 12*len + 4`, which is
    /// ≤ `12*len + 64` — in-bounds for every `idx < len`. No branch needed.
    ///
    /// The `debug_assert!(idx < len)` bounds the *index*; it does **not** verify
    /// the tail exists. The tail is guaranteed by construction of the sole
    /// caller, `gather_map`, which only receives `SparseGrid::block_data_ptr`
    /// output (a `Block::data` pointer for real, full, and empty blocks alike).
    /// A caller handing this an unpadded `[Velocity; N]` slice would be unsound
    /// here — see `Dequant::dequant_at`'s contract.
    #[inline(always)]
    unsafe fn dequant_at(src: *const Velocity, idx: usize, len: usize) -> f32x4 {
        debug_assert!(idx < len);
        // SAFETY: see above — `Velocity` is transparent over `[f32; 3]`, so
        // `data` is a contiguous `[f32; 3*len]` followed by 64 bytes of
        // initialized `flags`/`_pad`; the 4-float read `[3*idx, 3*idx+4)` stays
        // within the block allocation (and `f32` accepts any bit pattern).
        let p = unsafe { (src as *const f32).add(3 * idx) };
        unsafe { f32x4::from_slice(std::slice::from_raw_parts(p, 4)) }
    }
}

impl Dequant<f32x4> for FaceDensity {
    #[inline(always)]
    fn dequant(self) -> f32x4 {
        let inv = 1.0 / u16::MAX as f32;
        f32x4::from_array([
            self.0[0] as f32 * inv,
            self.0[1] as f32 * inv,
            self.0[2] as f32 * inv,
            0.0,
        ])
    }
}

/// Second-order derivatives of a scalar field, ordered `[fxx, fyy, fzz, fxy,
/// fxz, fyz]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Hessian {
    pub fxx: f32,
    pub fyy: f32,
    pub fzz: f32,
    pub fxy: f32,
    pub fxz: f32,
    pub fyz: f32,
}

/// Scalar field sampling over a [`SparseGrid<T>`].
pub trait Sample<const BSX: usize, const N: usize> {
    fn sample<const IP: Interpolation>(
        &self,
        pos: Vec3,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> f32;

    fn gradient<const IP: Interpolation>(
        &self,
        pos: Vec3,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> Vec3;

    /// Cubic B-spline value + gradient + Hessian. Cubic-only.
    fn hessian(&self, pos: Vec3, align: GridAlignment, bc: BoundaryCondition) -> Hessian;
}

/// Vector field sampling over a [`SparseGrid<T>`] (value only).
pub trait SampleVec3<const BSX: usize, const N: usize> {
    fn sample_vec3<const IP: Interpolation>(
        &self,
        pos: Vec3,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> Vec3;
}

// Reduction kernels (value / value+grad / value+grad+hess)
#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a).mul_add(t, a)
}

#[inline(always)]
fn pack3(v: Vec3) -> f32x4 {
    f32x4::from_array([v.0[0], v.0[1], v.0[2], 0.0])
}

fn trilinear_value(c: &[f32; 8], u: f32, v: f32, w: f32) -> f32 {
    let c00 = lerp(c[0], c[1], u);
    let c10 = lerp(c[2], c[3], u);
    let c01 = lerp(c[4], c[5], u);
    let c11 = lerp(c[6], c[7], u);
    let c0 = lerp(c00, c10, v);
    let c1 = lerp(c01, c11, v);
    lerp(c0, c1, w)
}

/// Trilinear value + analytic gradient. `c` holds the 8 corners in lex order
/// (`i + 2j + 4k`); the gradient uses the factored derivative coefficients.
fn trilinear_value_grad(c: &[f32; 8], u: f32, v: f32, w: f32) -> (f32, [f32; 3]) {
    let f0 = c[0];
    let f1 = c[1];
    let f2 = c[2];
    let f3 = c[3];
    let f4 = c[4];
    let f5 = c[5];
    let f6 = c[6];
    let f7 = c[7];

    let value = trilinear_value(c, u, v, w);

    let f01234567 = f7 - f6 - f5 + f4 - f3 + f2 + f1 - f0;
    let f01234567u = f01234567 * u;
    let f01234567uf6420 = f01234567u + f6 - f4 - f2 + f0;
    let f5410 = f5 - f4 - f1 + f0;
    let f3210 = f3 - f2 - f1 + f0;

    let gx = (f01234567 * v + f5410).mul_add(w, f3210 * v + f1 - f0);
    let gy = f01234567uf6420.mul_add(w, f3210 * u + f2 - f0);
    let gz = f01234567uf6420.mul_add(v, f5410 * u + f4 - f0);

    (value, [gx, gy, gz])
}

fn cubic_value(win: &[f32; 64], tx: f32, ty: f32, tz: f32) -> f32 {
    let wx = f32x4::from_array(cubic_weights(tx));
    let wy = cubic_weights(ty);
    let wz = cubic_weights(tz);

    let mut val = 0.0f32;
    for k in 0..4 {
        for j in 0..4 {
            let off = 16 * k + 4 * j;
            let f = f32x4::from_slice(&win[off..off + 4]);
            let sx = (f * wx).reduce_sum();
            val = sx.mul_add(wy[j] * wz[k], val);
        }
    }
    val
}

fn cubic_value_grad(win: &[f32; 64], tx: f32, ty: f32, tz: f32) -> (f32, [f32; 3]) {
    let wx = f32x4::from_array(cubic_weights(tx));
    let wpx = f32x4::from_array(cubic_deriv_weights(tx));
    let wy = cubic_weights(ty);
    let wpy = cubic_deriv_weights(ty);
    let wz = cubic_weights(tz);
    let wpz = cubic_deriv_weights(tz);

    let mut val = 0.0f32;
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    let mut gz = 0.0f32;

    for k in 0..4 {
        for j in 0..4 {
            let off = 16 * k + 4 * j;
            let f = f32x4::from_slice(&win[off..off + 4]);
            let sx = (f * wx).reduce_sum();
            let sxp = (f * wpx).reduce_sum();
            let wyz = wy[j] * wz[k];
            val = sx.mul_add(wyz, val);
            gx = sxp.mul_add(wyz, gx);
            gy = sx.mul_add(wpy[j] * wz[k], gy);
            gz = sx.mul_add(wy[j] * wpz[k], gz);
        }
    }

    (val, [gx, gy, gz])
}

fn cubic_value_grad_hess(
    win: &[f32; 64],
    tx: f32,
    ty: f32,
    tz: f32,
) -> (f32, [f32; 3], Hessian) {
    let wx = f32x4::from_array(cubic_weights(tx));
    let wpx = f32x4::from_array(cubic_deriv_weights(tx));
    let wppx = f32x4::from_array(cubic_deriv2_weights(tx));
    let wy = cubic_weights(ty);
    let wpy = cubic_deriv_weights(ty);
    let wppy = cubic_deriv2_weights(ty);
    let wz = cubic_weights(tz);
    let wpz = cubic_deriv_weights(tz);
    let wppz = cubic_deriv2_weights(tz);

    let mut val = 0.0f32;
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    let mut gz = 0.0f32;
    let mut fxx = 0.0f32;
    let mut fyy = 0.0f32;
    let mut fzz = 0.0f32;
    let mut fxy = 0.0f32;
    let mut fxz = 0.0f32;
    let mut fyz = 0.0f32;

    for k in 0..4 {
        for j in 0..4 {
            let off = 16 * k + 4 * j;
            let f = f32x4::from_slice(&win[off..off + 4]);
            let sx = (f * wx).reduce_sum();
            let sxp = (f * wpx).reduce_sum();
            let sxpp = (f * wppx).reduce_sum();
            let wyz = wy[j] * wz[k];
            let wpyz = wpy[j] * wz[k];
            let wypz = wy[j] * wpz[k];
            let wppyz = wppy[j] * wz[k];
            let wyppz = wy[j] * wppz[k];
            let wpyzp = wpy[j] * wpz[k];
            val = sx.mul_add(wyz, val);
            gx = sxp.mul_add(wyz, gx);
            gy = sx.mul_add(wpyz, gy);
            gz = sx.mul_add(wypz, gz);
            fxx = sxpp.mul_add(wyz, fxx);
            fyy = sx.mul_add(wppyz, fyy);
            fzz = sx.mul_add(wyppz, fzz);
            fxy = sxp.mul_add(wpyz, fxy);
            fxz = sxp.mul_add(wypz, fxz);
            fyz = sx.mul_add(wpyzp, fyz);
        }
    }

    (
        val,
        [gx, gy, gz],
        Hessian {
            fxx,
            fyy,
            fzz,
            fxy,
            fxz,
            fyz,
        },
    )
}

fn trilinear_value_vec3(c: &[f32x4; 8], u: f32, v: f32, w: f32) -> Vec3 {
    // The corners arrive already packed (x, y, z, 0) in f32x4; run the
    // trilinear as a 4-lane SIMD tree. Written as seven `mul_add` lerps
    // (`(b − a)·t + a`) rather than the `(1−u)`-factored form: fewer
    // instructions (7 FMA + 7 sub vs ~24 mul/add) and it emits hardware FMA
    // (`vfmadd`), matching the C++ `interpolateVec3Float` tree (which g++ fuses)
    // and the scalar `lerp` helper below.
    let c0 = c[0];
    let c1 = c[1];
    let c2 = c[2];
    let c3 = c[3];
    let c4 = c[4];
    let c5 = c[5];
    let c6 = c[6];
    let c7 = c[7];
    let u = f32x4::splat(u);
    let v = f32x4::splat(v);
    let w = f32x4::splat(w);
    let c00 = (c1 - c0).mul_add(u, c0);
    let c10 = (c3 - c2).mul_add(u, c2);
    let c01 = (c5 - c4).mul_add(u, c4);
    let c11 = (c7 - c6).mul_add(u, c6);
    let c0 = (c10 - c00).mul_add(v, c00);
    let c1 = (c11 - c01).mul_add(v, c01);
    let out = (c1 - c0).mul_add(w, c0);
    let [x, y, z, _] = out.to_array();
    Vec3([x, y, z])
}

fn cubic_value_vec3(win: &[f32x4; 64], tx: f32, ty: f32, tz: f32) -> Vec3 {
    let wx = cubic_weights(tx);
    let wy = cubic_weights(ty);
    let wz = cubic_weights(tz);

    let mut acc = f32x4::splat(0.0);
    for k in 0..4 {
        for j in 0..4 {
            for i in 0..4 {
                acc = win[16 * k + 4 * j + i].mul_add(f32x4::splat(wx[i] * wy[j] * wz[k]), acc);
            }
        }
    }
    let [x, y, z, _] = acc.to_array();
    Vec3([x, y, z])
}

// Gather + dispatch helpers
#[inline(always)]
fn gather_linear_f32<T: InterpElem, const BSX: usize, const N: usize>(
    grid: &SparseGrid<T, BSX, N>,
    ix: i32,
    iy: i32,
    iz: i32,
    bc: BoundaryCondition,
) -> [f32; 8] {
    let mut c = [0.0f32; 8];
    gather_map::<T, f32, BSX, N>(grid, ix, iy, iz, bc, 1, &mut c);
    c
}

#[inline(always)]
fn gather_cubic_f32<T: InterpElem, const BSX: usize, const N: usize>(
    grid: &SparseGrid<T, BSX, N>,
    ix: i32,
    iy: i32,
    iz: i32,
    bc: BoundaryCondition,
) -> [f32; 64] {
    let mut c = [0.0f32; 64];
    gather_map::<T, f32, BSX, N>(grid, ix, iy, iz, bc, 3, &mut c);
    c
}

#[inline(always)]
fn gather_linear_vec3<T: InterpVec3Elem + Dequant<f32x4>, const BSX: usize, const N: usize>(
    grid: &SparseGrid<T, BSX, N>,
    ix: i32,
    iy: i32,
    iz: i32,
    bc: BoundaryCondition,
) -> [f32x4; 8] {
    let mut c = [f32x4::splat(0.0); 8];
    gather_map::<T, f32x4, BSX, N>(grid, ix, iy, iz, bc, 1, &mut c);
    c
}

#[inline(always)]
fn gather_cubic_vec3<T: InterpVec3Elem + Dequant<f32x4>, const BSX: usize, const N: usize>(
    grid: &SparseGrid<T, BSX, N>,
    ix: i32,
    iy: i32,
    iz: i32,
    bc: BoundaryCondition,
) -> [f32x4; 64] {
    let mut c = [f32x4::splat(0.0); 64];
    gather_map::<T, f32x4, BSX, N>(grid, ix, iy, iz, bc, 3, &mut c);
    c
}

/// Position in the node-centered frame (`Corner`), or cell-centered frame.
#[inline(always)]
fn aligned(pos: Vec3, align: GridAlignment) -> (f32, f32, f32) {
    match align {
        GridAlignment::Corner => (pos.x(), pos.y(), pos.z()),
        GridAlignment::CellCentered => (pos.x() - 0.5, pos.y() - 0.5, pos.z() - 0.5),
    }
}

/// Cubic B-spline reconstruction is inherently cell-centered (the `-0.5` is
/// part of the basis alignment); the alignment flag is ignored here.
#[inline(always)]
fn cell_centered(pos: Vec3) -> (f32, f32, f32) {
    (pos.x() - 0.5, pos.y() - 0.5, pos.z() - 0.5)
}

/// Cubic window base: `t = frac(x)`, `ix = floor(x) - 1` (the `-dmax/2` offset).
#[inline(always)]
fn cubic_base(x: f32) -> (f32, i32) {
    let xf = x.floor();
    (x - xf, floor_to_i32(xf) - 1)
}

/// Floored grid coordinate to `i32`, unchecked. The saturating `as i32` cast
/// adds a NaN/overflow branch per axis (visible as `vucomiss`/`cmov` chains in
/// the hot loop); grid coordinates are finite and far below `i32` range, so the
/// unchecked conversion matches C++ `(int)floorf(x)`.
#[inline(always)]
fn floor_to_i32(f: f32) -> i32 {
    // SAFETY: `f` is a floored, finite grid coordinate (bounded by the grid
    // extent, ~2^21 per axis at paper scale); `float_to_int_unchecked` is UB
    // only for NaN or out-of-range values, neither of which occurs here.
    unsafe { std::intrinsics::float_to_int_unchecked(f) }
}

// Trait impls
impl<T, const BSX: usize, const N: usize> Sample<BSX, N> for SparseGrid<T, BSX, N>
where
    T: InterpElem,
{
    #[inline]
    fn sample<const IP: Interpolation>(
        &self,
        pos: Vec3,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> f32 {
        match IP {
            Interpolation::Linear => {
                let (x, y, z) = aligned(pos, align);
                let ix = x.floor();
                let iy = y.floor();
                let iz = z.floor();
                trilinear_value(
                    &gather_linear_f32::<T, BSX, N>(self, floor_to_i32(ix), floor_to_i32(iy), floor_to_i32(iz), bc),
                    x - ix,
                    y - iy,
                    z - iz,
                )
            }
            Interpolation::CubicBSpline => {
                let (x, y, z) = cell_centered(pos);
                let (tx, ix) = cubic_base(x);
                let (ty, iy) = cubic_base(y);
                let (tz, iz) = cubic_base(z);
                cubic_value(
                    &gather_cubic_f32::<T, BSX, N>(self, ix, iy, iz, bc),
                    tx,
                    ty,
                    tz,
                )
            }
        }
    }

    #[inline]
    fn gradient<const IP: Interpolation>(
        &self,
        pos: Vec3,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> Vec3 {
        match IP {
            Interpolation::Linear => {
                let (x, y, z) = aligned(pos, align);
                let ix = x.floor();
                let iy = y.floor();
                let iz = z.floor();
                let (_, g) = trilinear_value_grad(
                    &gather_linear_f32::<T, BSX, N>(self, floor_to_i32(ix), floor_to_i32(iy), floor_to_i32(iz), bc),
                    x - ix,
                    y - iy,
                    z - iz,
                );
                Vec3(g)
            }
            Interpolation::CubicBSpline => {
                let (x, y, z) = cell_centered(pos);
                let (tx, ix) = cubic_base(x);
                let (ty, iy) = cubic_base(y);
                let (tz, iz) = cubic_base(z);
                let (_, g) = cubic_value_grad(
                    &gather_cubic_f32::<T, BSX, N>(self, ix, iy, iz, bc),
                    tx,
                    ty,
                    tz,
                );
                Vec3(g)
            }
        }
    }

    #[inline]
    fn hessian(&self, pos: Vec3, align: GridAlignment, bc: BoundaryCondition) -> Hessian {
        let _ = align; // cubic is cell-centered regardless
        let (x, y, z) = cell_centered(pos);
        let (tx, ix) = cubic_base(x);
        let (ty, iy) = cubic_base(y);
        let (tz, iz) = cubic_base(z);
        cubic_value_grad_hess(
            &gather_cubic_f32::<T, BSX, N>(self, ix, iy, iz, bc),
            tx,
            ty,
            tz,
        )
        .2
    }
}

impl<T, const BSX: usize, const N: usize> SampleVec3<BSX, N> for SparseGrid<T, BSX, N>
where
    T: InterpVec3Elem + Dequant<f32x4>,
{
    #[inline]
    fn sample_vec3<const IP: Interpolation>(
        &self,
        pos: Vec3,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> Vec3 {
        match IP {
            Interpolation::Linear => {
                let (x, y, z) = aligned(pos, align);
                let ix = x.floor();
                let iy = y.floor();
                let iz = z.floor();
                trilinear_value_vec3(
                    &gather_linear_vec3::<T, BSX, N>(self, floor_to_i32(ix), floor_to_i32(iy), floor_to_i32(iz), bc),
                    x - ix,
                    y - iy,
                    z - iz,
                )
            }
            Interpolation::CubicBSpline => {
                let (x, y, z) = cell_centered(pos);
                let (tx, ix) = cubic_base(x);
                let (ty, iy) = cubic_base(y);
                let (tz, iz) = cubic_base(z);
                cubic_value_vec3(
                    &gather_cubic_vec3::<T, BSX, N>(self, ix, iy, iz, bc),
                    tx,
                    ty,
                    tz,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stateful samplers (cache hook reserved for step 10)
// ---------------------------------------------------------------------------

/// A stateful scalar sampler binding a grid to a fixed alignment + BC.
pub struct Sampler<'a, T: InterpElem, const BSX: usize, const N: usize> {
    grid: &'a SparseGrid<T, BSX, N>,
    align: GridAlignment,
    bc: BoundaryCondition,
}

impl<'a, T: InterpElem, const BSX: usize, const N: usize> Sampler<'a, T, BSX, N> {
    pub fn new(
        grid: &'a SparseGrid<T, BSX, N>,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> Self {
        Self { grid, align, bc }
    }

    #[inline]
    pub fn sample<const IP: Interpolation>(&self, pos: Vec3) -> f32 {
        <SparseGrid<T, BSX, N> as Sample<BSX, N>>::sample::<IP>(self.grid, pos, self.align, self.bc)
    }

    #[inline]
    pub fn gradient<const IP: Interpolation>(&self, pos: Vec3) -> Vec3 {
        <SparseGrid<T, BSX, N> as Sample<BSX, N>>::gradient::<IP>(
            self.grid, pos, self.align, self.bc,
        )
    }

    #[inline]
    pub fn hessian(&self, pos: Vec3) -> Hessian {
        <SparseGrid<T, BSX, N> as Sample<BSX, N>>::hessian(self.grid, pos, self.align, self.bc)
    }
}

/// A stateful vector-field sampler.
pub struct SamplerVec3<'a, T: InterpVec3Elem + Dequant<f32x4>, const BSX: usize, const N: usize> {
    grid: &'a SparseGrid<T, BSX, N>,
    align: GridAlignment,
    bc: BoundaryCondition,
}

impl<'a, T: InterpVec3Elem + Dequant<f32x4>, const BSX: usize, const N: usize>
    SamplerVec3<'a, T, BSX, N>
{
    pub fn new(
        grid: &'a SparseGrid<T, BSX, N>,
        align: GridAlignment,
        bc: BoundaryCondition,
    ) -> Self {
        Self { grid, align, bc }
    }

    #[inline]
    pub fn sample<const IP: Interpolation>(&self, pos: Vec3) -> Vec3 {
        <SparseGrid<T, BSX, N> as SampleVec3<BSX, N>>::sample_vec3::<IP>(
            self.grid, pos, self.align, self.bc,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;
    const CORNER: GridAlignment = GridAlignment::Corner;
    const CLAMP: BoundaryCondition = BoundaryCondition::Clamp;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn grid<D: Copy + Default + Send + Sync>(
        sx: usize,
        sy: usize,
        sz: usize,
        empty: D,
        full: D,
    ) -> SparseGrid<D, BSX, N> {
        let pool = Arc::new(BlockPool::new(8, 64));
        SparseGrid::new("t".into(), sx, sy, sz, empty, full, pool)
    }

    fn fill<D: Copy + Default + Send + Sync>(
        g: &mut SparseGrid<D, BSX, N>,
        f: impl Fn(usize, usize, usize) -> D,
    ) {
        for z in 0..g.sz {
            for y in 0..g.sy {
                for x in 0..g.sx {
                    g.set_voxel(x, y, z, f(x, y, z));
                }
            }
        }
    }

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn sample_at(g: &SparseGrid<f32, BSX, N>, ip: Interpolation, pos: Vec3) -> f32 {
        match ip {
            Interpolation::Linear => g.sample::<{ Interpolation::Linear }>(pos, CORNER, CLAMP),
            Interpolation::CubicBSpline => g.sample::<{ Interpolation::CubicBSpline }>(pos, CORNER, CLAMP),
        }
    }

    fn grad_at(g: &SparseGrid<f32, BSX, N>, ip: Interpolation, pos: Vec3) -> Vec3 {
        match ip {
            Interpolation::Linear => g.gradient::<{ Interpolation::Linear }>(pos, CORNER, CLAMP),
            Interpolation::CubicBSpline => g.gradient::<{ Interpolation::CubicBSpline }>(pos, CORNER, CLAMP),
        }
    }

    // an affine field is reproduced exactly by both methods.
    #[test]
    fn test_sample_affine_exact() {
        let mut g = grid(48, 48, 48, 0.0f32, 1.0);
        fill(&mut g, |x, y, z| 2.0 * x as f32 + 3.0 * y as f32 + 4.0 * z as f32 + 5.0);

        for &(px, py, pz) in &[(10.3, 7.7, 3.2), (20.5, 21.25, 40.75), (5.1, 30.9, 10.4)] {
            let pos = v(px, py, pz);
            // Cubic B-spline is cell-centered: it reconstructs at `pos - 0.5`.
            let expect_linear = 2.0 * px + 3.0 * py + 4.0 * pz + 5.0;
            let expect_cubic = 2.0 * (px - 0.5) + 3.0 * (py - 0.5) + 4.0 * (pz - 0.5) + 5.0;

            assert!(close(g.sample::<{ Interpolation::Linear }>(pos, CORNER, CLAMP), expect_linear, 1e-3));
            assert!(close(g.sample::<{ Interpolation::CubicBSpline }>(pos, CORNER, CLAMP), expect_cubic, 1e-3));

            let gl = g.gradient::<{ Interpolation::Linear }>(pos, CORNER, CLAMP);
            let gc = g.gradient::<{ Interpolation::CubicBSpline }>(pos, CORNER, CLAMP);
            assert!(close(gl.x(), 2.0, 1e-3) && close(gl.y(), 3.0, 1e-3) && close(gl.z(), 4.0, 1e-3));
            assert!(close(gc.x(), 2.0, 1e-3) && close(gc.y(), 3.0, 1e-3) && close(gc.z(), 4.0, 1e-3));
        }
    }

    // Clamp replicates the edge voxel past the last sample.
    #[test]
    fn test_boundary_clamp() {
        let mut g = grid(16, 16, 16, 0.0f32, 1.0);
        fill(&mut g, |x, _, _| x as f32);

        // x = 15.9 -> floor 15, u = 0.9, both stencil indices clamp to 15.
        assert!(close(g.sample::<{ Interpolation::Linear }>(v(15.9, 4.0, 4.0), CORNER, CLAMP), 15.0, 1e-6));
        // x = -0.1 -> floor -1, both indices clamp to 0.
        assert!(close(g.sample::<{ Interpolation::Linear }>(v(-0.1, 4.0, 4.0), CORNER, CLAMP), 0.0, 1e-6));
    }

    // Dirichlet reads the empty sentinel out-of-domain (not the edge).
    #[test]
    fn test_boundary_dirichlet() {
        let mut g = grid(16, 16, 16, 100.0f32, 1.0);
        fill(&mut g, |x, _, _| x as f32);

        let inside = g.sample::<{ Interpolation::Linear }>(v(5.0, 4.0, 4.0), CORNER, BoundaryCondition::Dirichlet);
        // x = -1.2 -> floor -2; both stencil indices are out-of-domain, so the
        // whole sample reads the empty sentinel.
        let outside = g.sample::<{ Interpolation::Linear }>(v(-1.2, 4.0, 4.0), CORNER, BoundaryCondition::Dirichlet);
        assert!(close(inside, 5.0, 1e-6));
        assert!(close(outside, 100.0, 1e-6));
    }

    // Neumann mirrors (reflect(-2)=1, reflect(-1)=0).
    #[test]
    fn test_boundary_neumann() {
        let mut g = grid(16, 16, 16, 0.0f32, 1.0);
        fill(&mut g, |x, _, _| x as f32);

        // x = -1.7 -> floor -2, u = 0.3, indices -2,-1 mirror to 1,0 -> 0.7.
        let got = g.sample::<{ Interpolation::Linear }>(v(-1.7, 4.0, 4.0), CORNER, BoundaryCondition::Neumann);
        assert!(close(got, 0.7, 1e-6), "got {got}");
    }

    // linear sample exactly on a block boundary averages two blocks.
    #[test]
    fn test_linear_cross_block() {
        let mut g = grid(32, 16, 16, 0.0f32, 1.0);
        fill(&mut g, |x, _, _| x as f32);

        let val = g.sample::<{ Interpolation::Linear }>(v(15.5, 4.0, 4.0), CORNER, CLAMP);
        assert!(close(val, 15.5, 1e-6));
        let gr = g.gradient::<{ Interpolation::Linear }>(v(15.5, 4.0, 4.0), CORNER, CLAMP);
        assert!(close(gr.x(), 1.0, 1e-5));
    }

    // cubic stencil straddling a block boundary still reproduces the
    // affine field exactly. Cubic is cell-centered, so pos=16.0 samples the
    // field at 15.5, whose 4-wide stencil (14,15 | 16,17) spans two blocks.
    #[test]
    fn test_cubic_cross_block() {
        let mut g = grid(32, 16, 16, 0.0f32, 1.0);
        fill(&mut g, |x, _, _| x as f32);

        let val = g.sample::<{ Interpolation::CubicBSpline }>(v(16.0, 4.0, 4.0), CORNER, CLAMP);
        assert!(close(val, 15.5, 1e-4), "got {val}");
        let gr = g.gradient::<{ Interpolation::CubicBSpline }>(v(16.0, 4.0, 4.0), CORNER, CLAMP);
        assert!(close(gr.x(), 1.0, 1e-3));
    }

    // a single-voxel grid clamps the whole cubic stencil to that voxel.
    #[test]
    fn test_cubic_single_voxel() {
        let mut g = grid(1, 1, 1, 0.0f32, 1.0);
        fill(&mut g, |_, _, _| 7.0f32);

        let val = g.sample::<{ Interpolation::CubicBSpline }>(v(0.3, 0.3, 0.3), CORNER, CLAMP);
        assert!(close(val, 7.0, 1e-6));
        let gr = g.gradient::<{ Interpolation::CubicBSpline }>(v(0.3, 0.3, 0.3), CORNER, CLAMP);
        assert!(close(gr.x(), 0.0, 1e-6));
    }

    // a partial last block (sx=17) clamps reads to the last valid voxel.
    #[test]
    fn test_linear_partial_block() {
        let mut g = grid(17, 16, 16, 0.0f32, 1.0);
        fill(&mut g, |x, _, _| x as f32);

        // x = 16.5 -> floor 16, u = 0.5, indices 16 and 17 (clamped to 16).
        let val = g.sample::<{ Interpolation::Linear }>(v(16.5, 4.0, 4.0), CORNER, CLAMP);
        assert!(close(val, 16.0, 1e-6));
    }

    // empty/full dummy blocks interpolate via their sentinel values.
    #[test]
    fn test_dummy_blocks() {
        let mut g = grid(32, 16, 16, 0.0f32, 1.0);
        // block 0 left empty (0.0), block 1 marked full (1.0).
        g.set_full_block(g.get_block_id(16, 0, 0));

        let val = g.sample::<{ Interpolation::Linear }>(v(15.5, 4.0, 4.0), CORNER, CLAMP);
        assert!(close(val, 0.5, 1e-6), "got {val}");
    }

    // u16 density dequantizes to [0,1] while sampling.
    #[test]
    fn test_density_dequant() {
        let mut g = grid(32, 16, 16, Density(0), Density(u16::MAX));
        fill(&mut g, |x, _, _| Density(((x as f32 / 31.0) * u16::MAX as f32).round() as u16));

        let val = g.sample::<{ Interpolation::Linear }>(v(15.5, 4.0, 4.0), CORNER, CLAMP);
        assert!(close(val, 0.5, 1e-3), "got {val}");
    }

    // gradient equals a finite difference of the sampled field
    // (self-consistency between value and gradient for both methods).
    #[test]
    fn test_gradient_matches_finite_difference() {
        let mut g = grid(48, 48, 48, 0.0f32, 1.0);
        fill(&mut g, |x, y, z| {
            0.01 * x as f32 * x as f32 - 0.02 * y as f32 + 0.03 * (z as f32).sin()
        });

        let h = 1e-3f32;
        for &(px, py, pz) in &[(11.3, 12.7, 13.2), (20.5, 21.25, 30.75), (8.1, 25.9, 40.4)] {
            for ip in [Interpolation::Linear, Interpolation::CubicBSpline] {
                let gr = grad_at(&g, ip, v(px, py, pz));
                let fdx = (sample_at(&g, ip, v(px + h, py, pz)) - sample_at(&g, ip, v(px - h, py, pz))) / (2.0 * h);
                let fdy = (sample_at(&g, ip, v(px, py + h, pz)) - sample_at(&g, ip, v(px, py - h, pz))) / (2.0 * h);
                let fdz = (sample_at(&g, ip, v(px, py, pz + h)) - sample_at(&g, ip, v(px, py, pz - h))) / (2.0 * h);
                assert!(close(gr.x(), fdx, 2e-3), "gx {ip:?}: {} vs {fdx}", gr.x());
                assert!(close(gr.y(), fdy, 2e-3), "gy {ip:?}");
                assert!(close(gr.z(), fdz, 2e-3), "gz {ip:?}");
            }
        }
    }

    // Hessian equals a finite difference of the gradient (quadratic field).
    #[test]
    fn test_hessian_matches_finite_difference() {
        let mut g = grid(48, 48, 48, 0.0f32, 1.0);
        fill(&mut g, |x, y, z| {
            let (x, y, z) = (x as f32, y as f32, z as f32);
            x * x + 2.0 * y * y + 3.0 * z * z
        });

        let h = 1e-2f32;
        for &(px, py, pz) in &[(11.3, 12.7, 13.2), (20.5, 21.25, 30.75)] {
            let hess = g.hessian(v(px, py, pz), CORNER, CLAMP);
            let gxp = grad_at(&g, Interpolation::CubicBSpline, v(px + h, py, pz));
            let gxm = grad_at(&g, Interpolation::CubicBSpline, v(px - h, py, pz));
            let gyp = grad_at(&g, Interpolation::CubicBSpline, v(px, py + h, pz));
            let gym = grad_at(&g, Interpolation::CubicBSpline, v(px, py - h, pz));
            let gzp = grad_at(&g, Interpolation::CubicBSpline, v(px, py, pz + h));
            let gzm = grad_at(&g, Interpolation::CubicBSpline, v(px, py, pz - h));

            assert!(close(hess.fxx, (gxp.x() - gxm.x()) / (2.0 * h), 2e-2), "fxx {}", hess.fxx);
            assert!(close(hess.fyy, (gyp.y() - gym.y()) / (2.0 * h), 2e-2), "fyy {}", hess.fyy);
            assert!(close(hess.fzz, (gzp.z() - gzm.z()) / (2.0 * h), 2e-2), "fzz {}", hess.fzz);
            assert!(close(hess.fxy, (gyp.x() - gym.x()) / (2.0 * h), 2e-2), "fxy {}", hess.fxy);
            assert!(close(hess.fxz, (gzp.x() - gzm.x()) / (2.0 * h), 2e-2), "fxz {}", hess.fxz);
            assert!(close(hess.fyz, (gzp.y() - gzm.y()) / (2.0 * h), 2e-2), "fyz {}", hess.fyz);
        }
    }

    // affine velocity field.
    #[test]
    fn test_vec3_affine_exact() {
        let mut g = grid(48, 48, 48, Velocity(Vec3::default()), Velocity(Vec3::default()));
        fill(&mut g, |x, y, z| Velocity(Vec3::new(x as f32, 2.0 * y as f32, 3.0 * z as f32)));

        let pos = v(10.5, 20.25, 30.75);
        let out = g.sample_vec3::<{ Interpolation::Linear }>(pos, CORNER, CLAMP);
        assert!(close(out.x(), 10.5, 1e-3) && close(out.y(), 40.5, 1e-3) && close(out.z(), 92.25, 1e-3));
        // Cubic is cell-centered: reconstructs at (10.0, 39.75, 90.25).
        let outc = g.sample_vec3::<{ Interpolation::CubicBSpline }>(pos, CORNER, CLAMP);
        assert!(close(outc.x(), 10.0, 1e-3) && close(outc.y(), 39.5, 1e-3) && close(outc.z(), 90.75, 1e-3));
    }

    // clamp past the edge.
    #[test]
    fn test_vec3_clamp() {
        let mut g = grid(16, 16, 16, Velocity(Vec3::default()), Velocity(Vec3::default()));
        fill(&mut g, |x, y, z| Velocity(Vec3::new(x as f32, y as f32, z as f32)));

        let out = g.sample_vec3::<{ Interpolation::Linear }>(v(15.9, 15.9, 15.9), CORNER, CLAMP);
        assert!(close(out.x(), 15.0, 1e-6) && close(out.y(), 15.0, 1e-6) && close(out.z(), 15.0, 1e-6));
    }
}

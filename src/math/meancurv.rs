//! 19-tap mean-curvature smoothing kernel (C++ `applyChannelPdeFast`, `laplTyp==4`).

use crate::blockpool::Block;
use crate::math::stencil::{self, MaskBlock};
use crate::multires::halo::HaloBlock;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::{Select, Simd, StdFloat};

/// Full-update mean-curvature smoothing over a pre-staged full halo.
///
/// Mirrors the C++ `laplTyp==4` update: `H = Hnum / gradMagSq` (guarded at
/// `gradMagSq > 1e-7`), `D = clamp(dt * H, ±0.1)`, `new = F0 + D`.
/// Non-fluid cells keep their previous value (`f0`). Mixed partials are factored
/// as `0.25*(a-b-c+d)` — fewer FLOPs than C++'s two-stage `0.5` form.
#[inline(always)]
pub fn kernel_meancurv<
    const W: usize,
    const CHUNKS: usize,
    const BSX: usize,
    const HSX: usize,
    const N: usize,
>(
    halo: &HaloBlock<BSX, HSX>,
    mask: &MaskBlock<W, CHUNKS>,
    dt: f32,
    output: &mut Block<f32, BSX, N>,
) {

    debug_assert_eq!(HSX, BSX + 2, "mean-curvature needs a 1-voxel halo");
    debug_assert_eq!(BSX % W, 0, "BSX must be a multiple of W");
    debug_assert_eq!(CHUNKS * W, N, "CHUNKS must equal N / W");

    let base = halo.data.as_ptr();
    let out = output.data.as_mut_ptr();
    let dy = HSX;
    let dz = HSX * HSX;

    let dtv = Simd::<f32, W>::splat(dt);
    let half = Simd::<f32, W>::splat(0.5);
    let quarter = Simd::<f32, W>::splat(0.25);
    let two = Simd::<f32, W>::splat(2.0);
    let eps = Simd::<f32, W>::splat(1e-7);
    let lo = Simd::<f32, W>::splat(-0.1);
    let hi = Simd::<f32, W>::splat(0.1);
    let zero = Simd::<f32, W>::splat(0.0);

    let mut out_idx = 0;
    for z in 1..=BSX {
        for y in 1..=BSX {
            let mut hidx = z * dz + y * dy + 1;
            for _ in (0..BSX).step_by(W) {
                unsafe {
                    let f0 = stencil::load::<W>(base, hidx);
                    let f1 = stencil::load::<W>(base, hidx - 1);
                    let f2 = stencil::load::<W>(base, hidx + 1);
                    let f3 = stencil::load::<W>(base, hidx - dy);
                    let f4 = stencil::load::<W>(base, hidx + dy);
                    let f5 = stencil::load::<W>(base, hidx - dz);
                    let f6 = stencil::load::<W>(base, hidx + dz);

                    let fx = (f2 - f1) * half;
                    let fy = (f4 - f3) * half;
                    let fz = (f6 - f5) * half;

                    let fxx = f1 + f2 - two * f0;
                    let fyy = f3 + f4 - two * f0;
                    let fzz = f5 + f6 - two * f0;

                    // 12 diagonal taps.
                    let d1 = stencil::load::<W>(base, hidx + 1 + dy); // F(1,1,0)
                    let d2 = stencil::load::<W>(base, hidx + 1 - dy); // F(1,-1,0)
                    let d3 = stencil::load::<W>(base, hidx - 1 + dy); // F(-1,1,0)
                    let d4 = stencil::load::<W>(base, hidx - 1 - dy); // F(-1,-1,0)
                    let d5 = stencil::load::<W>(base, hidx + 1 + dz); // F(1,0,1)
                    let d6 = stencil::load::<W>(base, hidx + 1 - dz); // F(1,0,-1)
                    let d7 = stencil::load::<W>(base, hidx - 1 + dz); // F(-1,0,1)
                    let d8 = stencil::load::<W>(base, hidx - 1 - dz); // F(-1,0,-1)
                    let d9 = stencil::load::<W>(base, hidx + dy + dz); // F(0,1,1)
                    let d10 = stencil::load::<W>(base, hidx + dy - dz); // F(0,1,-1)
                    let d11 = stencil::load::<W>(base, hidx - dy + dz); // F(0,-1,1)
                    let d12 = stencil::load::<W>(base, hidx - dy - dz); // F(0,-1,-1)

                    let fyx = (d1 - d2 - d3 + d4) * quarter;
                    let fzx = (d5 - d6 - d7 + d8) * quarter;
                    let fzy = (d9 - d10 - d11 + d12) * quarter;

                    let cross = fx * (fy * fyx) + fx * (fz * fzx) + fy * (fz * fzy);
                    let hnum = (fy * fy + fz * fz).mul_add(
                        fxx,
                        (fx * fx + fz * fz).mul_add(
                            fyy,
                            (fx * fx + fy * fy).mul_add(fzz, -two * cross),
                        ),
                    );

                    let grad = fx.mul_add(fx, fy.mul_add(fy, fz * fz));
                    let big = grad.simd_gt(eps);
                    let h = big.select(hnum / grad, zero);
                    let d = (dtv * h).simd_clamp(lo, hi);

                    let new = f0 + d;
                    let m = mask.chunk(out_idx / W);
                    stencil::store_nt::<W>(out, out_idx, m.select(new, f0));
                }
                hidx += W;
                out_idx += W;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::simd::LANES;

    const BSX: usize = 16;
    const N: usize = 4096;
    const HSX: usize = 18;
    const CHUNKS: usize = N / LANES;

    /// Curvature `H = Hnum/gradMagSq` (guarded) at a scalar halo index.
    fn scalar_curvature(halo: &HaloBlock<BSX, HSX>, h: usize) -> f32 {
        const DY: usize = HSX;
        const DZ: usize = HSX * HSX;
        let f0 = halo.data[h];
        let f1 = halo.data[h - 1];
        let f2 = halo.data[h + 1];
        let f3 = halo.data[h - DY];
        let f4 = halo.data[h + DY];
        let f5 = halo.data[h - DZ];
        let f6 = halo.data[h + DZ];
        let fx = 0.5 * (f2 - f1);
        let fy = 0.5 * (f4 - f3);
        let fz = 0.5 * (f6 - f5);
        let fxx = f1 + f2 - 2.0 * f0;
        let fyy = f3 + f4 - 2.0 * f0;
        let fzz = f5 + f6 - 2.0 * f0;
        let d1 = halo.data[h + 1 + DY];
        let d2 = halo.data[h + 1 - DY];
        let d3 = halo.data[h - 1 + DY];
        let d4 = halo.data[h - 1 - DY];
        let d5 = halo.data[h + 1 + DZ];
        let d6 = halo.data[h + 1 - DZ];
        let d7 = halo.data[h - 1 + DZ];
        let d8 = halo.data[h - 1 - DZ];
        let d9 = halo.data[h + DY + DZ];
        let d10 = halo.data[h + DY - DZ];
        let d11 = halo.data[h - DY + DZ];
        let d12 = halo.data[h - DY - DZ];
        let fyx = 0.25 * (d1 - d2 - d3 + d4);
        let fzx = 0.25 * (d5 - d6 - d7 + d8);
        let fzy = 0.25 * (d9 - d10 - d11 + d12);
        let hnum = (fy * fy + fz * fz) * fxx
            + (fx * fx + fz * fz) * fyy
            + (fx * fx + fy * fy) * fzz
            - 2.0 * (fx * fy * fyx + fx * fz * fzx + fy * fz * fzy);
        let grad = fx * fx + fy * fy + fz * fz;
        if grad > 1e-7 {
            hnum / grad
        } else {
            0.0
        }
    }

    /// Scalar full-update reference.
    fn scalar_meancurv(
        halo: &HaloBlock<BSX, HSX>,
        flags: &Block<u16, BSX, N>,
        dt: f32,
        out: &mut Block<f32, BSX, N>,
    ) {
        const DY: usize = HSX;
        const DZ: usize = HSX * HSX;
        let mut out_idx = 0;
        for z in 1..=16 {
            for y in 1..=16 {
                for x in 1..=16 {
                    let h = z * DZ + y * DY + x;
                    let f0 = halo.data[h];
                    let hc = scalar_curvature(halo, h);
                    let d = (dt * hc).clamp(-0.1, 0.1);
                    let new = f0 + d;
                    let is_fluid = (flags.data[out_idx] & crate::math::stencil::FLUID_MASK) == 0;
                    out.data[out_idx] = if is_fluid { new } else { f0 };
                    out_idx += 1;
                }
            }
        }
    }

    fn fill_halo_sphere(halo: &mut HaloBlock<BSX, HSX>, cx: f32, cy: f32, cz: f32) {
        for z in 0..HSX {
            for y in 0..HSX {
                for x in 0..HSX {
                    let gx = (x as i32 - 1) as f32 - cx;
                    let gy = (y as i32 - 1) as f32 - cy;
                    let gz = (z as i32 - 1) as f32 - cz;
                    halo.data[z * HSX * HSX + y * HSX + x] = (gx * gx + gy * gy + gz * gz).sqrt();
                }
            }
        }
    }

    #[test]
    fn test_mc_01_sphere_curvature_analytic() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        // Distance field from center (8,8,8) -> mean curvature 2/r.
        fill_halo_sphere(&mut halo, 8.0, 8.0, 8.0);
        // Sample at relative (11,8,8): r = 3 -> curvature ~= 2/3.
        let h = 9 * HSX * HSX + 9 * HSX + 12; // z=9,y=9,x=12 -> rel (11,8,8)
        let c = scalar_curvature(&halo, h);
        let expected = 2.0 / 3.0;
        assert!(
            (c - expected).abs() < 2e-2,
            "sphere curvature {c} != 2/r {expected}"
        );
    }

    #[test]
    fn test_mc_02_simd_matches_scalar_random() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        for (i, v) in halo.data.iter_mut().enumerate() {
            *v = ((i * 31) % 200) as f32 * 0.05;
        }
        let mut flags = Block::<u16, BSX, N>::new();
        for i in 0..N {
            flags.data[i] = match i % 6 {
                0 => 0x1,
                1 => 0x1000,
                _ => 0x0,
            };
        }
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);

        let mut out = Block::<f32, BSX, N>::new();
        let mut ref_out = Block::<f32, BSX, N>::new();
        let dt = 0.025;
        kernel_meancurv::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, dt, &mut out);
        scalar_meancurv(&halo, &flags, dt, &mut ref_out);

        for i in 0..N {
            let d = (out.data[i] - ref_out.data[i]).abs();
            assert!(d < 1e-4, "mismatch at {i}: simd={} scalar={}", out.data[i], ref_out.data[i]);
        }
    }

    #[test]
    fn test_mc_03_planar_and_constant_are_unchanged() {
        let flags = Block::<u16, BSX, N>::new();
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);

        // Planar field f = 2x + 3y - z: zero curvature, zero Laplacian.
        let mut halo = HaloBlock::<BSX, HSX>::new();
        for z in 0..HSX {
            for y in 0..HSX {
                for x in 0..HSX {
                    let gx = (x as i32 - 1) as f32;
                    let gy = (y as i32 - 1) as f32;
                    let gz = (z as i32 - 1) as f32;
                    halo.data[z * HSX * HSX + y * HSX + x] = 2.0 * gx + 3.0 * gy - gz;
                }
            }
        }
        let mut out = Block::<f32, BSX, N>::new();
        kernel_meancurv::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, 0.5, &mut out);
        let mut out_idx = 0;
        for z in 1..=16 {
            for y in 1..=16 {
                for x in 1..=16 {
                    let expected = 2.0 * (x as i32 - 1) as f32 + 3.0 * (y as i32 - 1) as f32
                        - (z as i32 - 1) as f32;
                    assert!((out.data[out_idx] - expected).abs() < 1e-5);
                    out_idx += 1;
                }
            }
        }

        // Constant field: grad guard kicks in, no NaN, field preserved.
        let mut halo = HaloBlock::<BSX, HSX>::new();
        for v in halo.data.iter_mut() {
            *v = 5.0;
        }
        let mut out = Block::<f32, BSX, N>::new();
        kernel_meancurv::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, 0.5, &mut out);
        for v in out.data.iter() {
            assert_eq!(*v, 5.0);
        }
    }

    #[test]
    fn test_mc_04_clamp_saturates() {
        // Distance field sampled near the center gives large curvature; a large
        // dt drives D into the ±0.1 clamp.
        let mut halo = HaloBlock::<BSX, HSX>::new();
        fill_halo_sphere(&mut halo, 8.0, 8.0, 8.0);
        let flags = Block::<u16, BSX, N>::new();
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);

        let mut out = Block::<f32, BSX, N>::new();
        kernel_meancurv::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, 100.0, &mut out);

        // Cell at relative (10,8,8): r=2, curvature ~=1, D = +100*1 -> clamped +0.1.
        // Its out index = x=11,y=9,z=9 (relative x=10,y=8,z=8).
        let idx = 10 + 8 * 16 + 8 * 256;
        let f0 = 2.0f32; // r=2
        assert!(
            (out.data[idx] - (f0 + 0.1)).abs() < 1e-4,
            "clamp failed: got {}",
            out.data[idx]
        );
    }
}

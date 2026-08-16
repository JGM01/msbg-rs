//! 25-tap bi-Laplacian smoothing kernel.

use crate::math::stencil::{self, SimdRng, StoreBack};
use crate::multires::halo::HaloBlock;
use std::simd::num::SimdFloat;
use std::simd::{Simd, StdFloat};

/// Full-update bi-Laplacian smoothing over a pre-staged full 2-voxel halo,
/// writing the smoothed chunk back through the storage's [`StoreBack`].
///
/// Update `H = 42*F0 - 12*sum6(±1) + 1*sum6(±2) + 2*sum12(diag ±1)`,
/// `D = clamp(-dt * H * G, ±0.1)`, `new = F0 + D`. No fluid mask; `clamp01` is
/// the `doConstrZeroOne` `[0, 1]` clamp.
#[inline(always)]
pub fn kernel_bilaplacian<const W: usize, const BSX: usize, const HSX: usize, D>(
    halo: &HaloBlock<BSX, HSX>,
    dt: f32,
    clamp01: bool,
    out: *mut D,
    rng: &mut SimdRng<W>,
) where
    D: StoreBack<W>,
{
    debug_assert_eq!(HSX, BSX + 4, "bi-Laplacian needs a 2-voxel halo");
    debug_assert_eq!(BSX % W, 0, "BSX must be a multiple of W");

    let base = halo.data.as_ptr();
    let dy = HSX;
    let dz = HSX * HSX;

    let dtv = Simd::<f32, W>::splat(dt);
    let half = Simd::<f32, W>::splat(0.5);
    let c42 = Simd::<f32, W>::splat(42.0);
    let c12 = Simd::<f32, W>::splat(12.0);
    let two = Simd::<f32, W>::splat(2.0);
    let lo = Simd::<f32, W>::splat(-0.1);
    let hi = Simd::<f32, W>::splat(0.1);
    let zero = Simd::<f32, W>::splat(0.0);
    let one = Simd::<f32, W>::splat(1.0);

    let mut out_idx = 0;
    for z in 2..=(BSX + 1) {
        for y in 2..=(BSX + 1) {
            let mut hidx = z * dz + y * dy + 2;
            for _ in (0..BSX).step_by(W) {
                unsafe {
                    let f0 = stencil::load::<W>(base, hidx);
                    let f1 = stencil::load::<W>(base, hidx - 1);
                    let f2 = stencil::load::<W>(base, hidx + 1);
                    let f3 = stencil::load::<W>(base, hidx - dy);
                    let f4 = stencil::load::<W>(base, hidx + dy);
                    let f5 = stencil::load::<W>(base, hidx - dz);
                    let f6 = stencil::load::<W>(base, hidx + dz);

                    let g1 = stencil::load::<W>(base, hidx - 2);
                    let g2 = stencil::load::<W>(base, hidx + 2);
                    let g3 = stencil::load::<W>(base, hidx - 2 * dy);
                    let g4 = stencil::load::<W>(base, hidx + 2 * dy);
                    let g5 = stencil::load::<W>(base, hidx - 2 * dz);
                    let g6 = stencil::load::<W>(base, hidx + 2 * dz);

                    let s1 = f1 + f2 + f3 + f4 + f5 + f6;
                    let s2 = g1 + g2 + g3 + g4 + g5 + g6;
                    let sd = stencil::load::<W>(base, hidx + 1 + dy)
                        + stencil::load::<W>(base, hidx + 1 - dy)
                        + stencil::load::<W>(base, hidx - 1 + dy)
                        + stencil::load::<W>(base, hidx - 1 - dy)
                        + stencil::load::<W>(base, hidx + 1 + dz)
                        + stencil::load::<W>(base, hidx + 1 - dz)
                        + stencil::load::<W>(base, hidx - 1 + dz)
                        + stencil::load::<W>(base, hidx - 1 - dz)
                        + stencil::load::<W>(base, hidx + dy + dz)
                        + stencil::load::<W>(base, hidx + dy - dz)
                        + stencil::load::<W>(base, hidx - dy + dz)
                        + stencil::load::<W>(base, hidx - dy - dz);

                    let h = c42 * f0 - c12 * s1 + s2 + two * sd;

                    let fx = (f2 - f1) * half;
                    let fy = (f4 - f3) * half;
                    let fz = (f6 - f5) * half;
                    let g = fx.mul_add(fx, fy.mul_add(fy, fz * fz)).sqrt();

                    let upd = (dtv * h) * g;
                    let d = (-upd).simd_clamp(lo, hi);

                    let new = f0 + d;
                    let new = if clamp01 { new.simd_clamp(zero, one) } else { new };

                    D::store_chunk(out, out_idx, new, rng);
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
    use crate::blockpool::Block;
    use crate::math::simd::LANES;

    const BSX: usize = 16;
    const HSX: usize = 20;
    const N: usize = 4096;

    fn scalar_bilaplacian(halo: &HaloBlock<BSX, HSX>, dt: f32, out: &mut [f32; N]) {
        const DY: usize = HSX;
        const DZ: usize = HSX * HSX;
        let mut out_idx = 0;
        for z in 2..=17 {
            for y in 2..=17 {
                for x in 2..=17 {
                    let h = z * DZ + y * DY + x;
                    let f0 = halo.data[h];
                    let f1 = halo.data[h - 1];
                    let f2 = halo.data[h + 1];
                    let f3 = halo.data[h - DY];
                    let f4 = halo.data[h + DY];
                    let f5 = halo.data[h - DZ];
                    let f6 = halo.data[h + DZ];
                    let s1 = f1 + f2 + f3 + f4 + f5 + f6;
                    let s2 = halo.data[h - 2]
                        + halo.data[h + 2]
                        + halo.data[h - 2 * DY]
                        + halo.data[h + 2 * DY]
                        + halo.data[h - 2 * DZ]
                        + halo.data[h + 2 * DZ];
                    let sd = halo.data[h + 1 + DY]
                        + halo.data[h + 1 - DY]
                        + halo.data[h - 1 + DY]
                        + halo.data[h - 1 - DY]
                        + halo.data[h + 1 + DZ]
                        + halo.data[h + 1 - DZ]
                        + halo.data[h - 1 + DZ]
                        + halo.data[h - 1 - DZ]
                        + halo.data[h + DY + DZ]
                        + halo.data[h + DY - DZ]
                        + halo.data[h - DY + DZ]
                        + halo.data[h - DY - DZ];
                    let hh = 42.0 * f0 - 12.0 * s1 + s2 + 2.0 * sd;
                    let fx = 0.5 * (f2 - f1);
                    let fy = 0.5 * (f4 - f3);
                    let fz = 0.5 * (f6 - f5);
                    let g = (fx * fx + fy * fy + fz * fz).sqrt();
                    let d = (-dt * hh * g).clamp(-0.1, 0.1);
                    out[out_idx] = f0 + d;
                    out_idx += 1;
                }
            }
        }
    }

    #[test]
    fn test_bl_01_simd_matches_scalar_random() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        for (i, v) in halo.data.iter_mut().enumerate() {
            *v = ((i * 13) % 250) as f32 * 0.01;
        }

        let mut out = Block::<f32, BSX, N>::new();
        let mut ref_out = [0.0f32; N];
        let dt = 0.01;
        let mut rng = SimdRng::<LANES>::seed(0);
        kernel_bilaplacian::<LANES, BSX, HSX, f32>(&halo, dt, false, out.data.as_mut_ptr(), &mut rng);
        scalar_bilaplacian(&halo, dt, &mut ref_out);

        for i in 0..N {
            let d = (out.data[i] - ref_out[i]).abs();
            assert!(d < 1e-4, "mismatch at {i}: simd={} scalar={}", out.data[i], ref_out[i]);
        }
    }

    #[test]
    #[should_panic]
    fn test_bl_02_wrong_hsx_panics() {
        // HSX=18 is wrong for bi-Laplacian (needs BSX+4).
        let halo = HaloBlock::<16, 18>::new();
        let mut out = Block::<f32, BSX, N>::new();
        let mut rng = SimdRng::<LANES>::seed(0);
        kernel_bilaplacian::<LANES, 16, 18, f32>(&halo, 0.01, false, out.data.as_mut_ptr(), &mut rng);
    }
}

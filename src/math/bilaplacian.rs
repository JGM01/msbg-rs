//! 25-tap bi-Laplacian smoothing kernel (C++ `applyChannelPdeFast`, `laplTyp==2`).

use crate::blockpool::Block;
use crate::math::stencil::{self, MaskBlock};
use crate::multires::halo::HaloBlock;
use std::simd::num::SimdFloat;
use std::simd::{Select, Simd, StdFloat};

/// Full-update bi-Laplacian smoothing over a pre-staged full 2-voxel halo.
///
/// Mirrors the C++ `laplTyp==2` update:
/// `H = 42*F0 - 12*sum6(±1) + 1*sum6(±2) + 2*sum12(diag ±1)`,
/// `D = clamp(-dt * H * G, ±0.1)`, `new = F0 + D`.
#[inline(always)]
pub fn kernel_bilaplacian<
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

    debug_assert_eq!(HSX, BSX + 4, "bi-Laplacian needs a 2-voxel halo");
    debug_assert_eq!(BSX % W, 0, "BSX must be a multiple of W");
    debug_assert_eq!(CHUNKS * W, N, "CHUNKS must equal N / W");

    let base = halo.data.as_ptr();
    let out = output.data.as_mut_ptr();
    let dy = HSX;
    let dz = HSX * HSX;

    let dtv = Simd::<f32, W>::splat(dt);
    let half = Simd::<f32, W>::splat(0.5);
    let c42 = Simd::<f32, W>::splat(42.0);
    let c12 = Simd::<f32, W>::splat(12.0);
    let two = Simd::<f32, W>::splat(2.0);
    let lo = Simd::<f32, W>::splat(-0.1);
    let hi = Simd::<f32, W>::splat(0.1);

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
                    let g = (fx.mul_add(fx, fy.mul_add(fy, fz * fz))).sqrt();

                    let upd = (dtv * h) * g;
                    let d = (-upd).simd_clamp(lo, hi);

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
    const HSX: usize = 20;
    const CHUNKS: usize = N / LANES;

    fn scalar_bilaplacian(
        halo: &HaloBlock<BSX, HSX>,
        flags: &Block<u16, BSX, N>,
        dt: f32,
        out: &mut Block<f32, BSX, N>,
    ) {
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
                    let new = f0 + d;
                    let is_fluid = (flags.data[out_idx] & crate::math::stencil::FLUID_MASK) == 0;
                    out.data[out_idx] = if is_fluid { new } else { f0 };
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
        let mut flags = Block::<u16, BSX, N>::new();
        for i in 0..N {
            flags.data[i] = if i % 7 == 0 { 0x1 } else { 0x0 };
        }
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);

        let mut out = Block::<f32, BSX, N>::new();
        let mut ref_out = Block::<f32, BSX, N>::new();
        let dt = 0.01;
        kernel_bilaplacian::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, dt, &mut out);
        scalar_bilaplacian(&halo, &flags, dt, &mut ref_out);

        for i in 0..N {
            let d = (out.data[i] - ref_out.data[i]).abs();
            assert!(d < 1e-4, "mismatch at {i}: simd={} scalar={}", out.data[i], ref_out.data[i]);
        }
    }

    #[test]
    #[should_panic]
    fn test_bl_02_wrong_hsx_panics() {
        // HSX=18 is wrong for bi-Laplacian (needs BSX+4).
        let halo = HaloBlock::<16, 18>::new();
        let flags = Block::<u16, BSX, N>::new();
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);
        let mut out = Block::<f32, BSX, N>::new();
        kernel_bilaplacian::<LANES, CHUNKS, 16, 18, N>(&halo, &mask, 0.01, &mut out);
    }
}

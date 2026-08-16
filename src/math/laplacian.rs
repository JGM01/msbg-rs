//! 7-point Laplacian smoothing kernel (C++ `applyChannelPdeFast`, `laplTyp==1`).

use crate::blockpool::Block;
use crate::math::stencil::{self, MaskBlock};
use crate::multires::halo::HaloBlock;
use std::simd::{Select, Simd, StdFloat};

/// Full-update 7-point Laplacian smoothing over a pre-staged faces-only halo.
///
/// Mirrors the C++ `laplTyp==1` update `phi += dt * L * G` where
/// `L = sum(6 neighbors) - 6*center` and `G` is the gradient magnitude of the
/// center stencil. Non-fluid cells keep their previous value (`f0`).
#[inline(always)]
pub fn kernel_laplacian<
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

    debug_assert_eq!(HSX, BSX + 2, "Laplacian needs a 1-voxel halo");
    debug_assert_eq!(BSX % W, 0, "BSX must be a multiple of W");
    debug_assert_eq!(CHUNKS * W, N, "CHUNKS must equal N / W");

    let base = halo.data.as_ptr();
    let out = output.data.as_mut_ptr();
    let dy = HSX;
    let dz = HSX * HSX;

    let dtv = Simd::<f32, W>::splat(dt);
    let six = Simd::<f32, W>::splat(6.0);
    let half = Simd::<f32, W>::splat(0.5);

    let mut out_idx = 0;
    for z in 1..=BSX {
        for y in 1..=BSX {
            let mut hidx = z * dz + y * dy + 1;
            for _ in (0..BSX).step_by(W) {
                unsafe {
                    let c = stencil::load::<W>(base, hidx);
                    let lx = stencil::load::<W>(base, hidx - 1);
                    let rx = stencil::load::<W>(base, hidx + 1);
                    let dyu = stencil::load::<W>(base, hidx - dy);
                    let dyd = stencil::load::<W>(base, hidx + dy);
                    let dzb = stencil::load::<W>(base, hidx - dz);
                    let dzf = stencil::load::<W>(base, hidx + dz);

                    let neighbors = lx + rx + dyu + dyd + dzb + dzf;
                    let lap = c.mul_add(-six, neighbors); // neighbors - 6*c

                    let fx = (rx - lx) * half;
                    let fy = (dyd - dyu) * half;
                    let fz = (dzf - dzb) * half;
                    let g = (fx.mul_add(fx, fy.mul_add(fy, fz * fz))).sqrt();

                    let new = (dtv * lap).mul_add(g, c); // c + dt*lap*g

                    let m = mask.chunk(out_idx / W);
                    stencil::store_nt::<W>(out, out_idx, m.select(new, c));
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

    fn fill_halo_quadratic(halo: &mut HaloBlock<BSX, HSX>) {
        // Halo coords run [-1 .. 16] relative to the interior block.
        for z in 0..HSX {
            for y in 0..HSX {
                for x in 0..HSX {
                    let gx = (x as i32 - 1) as f32;
                    let gy = (y as i32 - 1) as f32;
                    let gz = (z as i32 - 1) as f32;
                    halo.data[z * HSX * HSX + y * HSX + x] = gx * gx + gy * gy + gz * gz;
                }
            }
        }
    }

    /// Scalar full-update 7-point Laplacian (reference).
    fn scalar_laplacian_7pt(
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
                    let c = halo.data[h];
                    let lx = halo.data[h - 1];
                    let rx = halo.data[h + 1];
                    let dyu = halo.data[h - DY];
                    let dyd = halo.data[h + DY];
                    let dzb = halo.data[h - DZ];
                    let dzf = halo.data[h + DZ];

                    let lap = -6.0 * c + lx + rx + dyu + dyd + dzb + dzf;
                    let fx = 0.5 * (rx - lx);
                    let fy = 0.5 * (dyd - dyu);
                    let fz = 0.5 * (dzf - dzb);
                    let g = (fx * fx + fy * fy + fz * fz).sqrt();
                    let new = c + dt * lap * g;

                    let flag = flags.data[out_idx];
                    let is_fluid = (flag & crate::math::stencil::FLUID_MASK) == 0;
                    out.data[out_idx] = if is_fluid { new } else { c };
                    out_idx += 1;
                }
            }
        }
    }

    #[test]
    fn test_lap_01_quadratic_field_matches_scalar() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        fill_halo_quadratic(&mut halo);
        let flags = Block::<u16, BSX, N>::new();
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);

        let mut out = Block::<f32, BSX, N>::new();
        let mut ref_out = Block::<f32, BSX, N>::new();
        let dt = 0.025;

        kernel_laplacian::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, dt, &mut out);
        scalar_laplacian_7pt(&halo, &flags, dt, &mut ref_out);

        for i in 0..N {
            let d = (out.data[i] - ref_out.data[i]).abs();
            assert!(d < 1e-4, "mismatch at {i}: simd={} scalar={}", out.data[i], ref_out.data[i]);
        }
    }

    #[test]
    fn test_lap_02_simd_matches_scalar_random_field() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        for (i, v) in halo.data.iter_mut().enumerate() {
            *v = ((i * 17) % 100) as f32 * 0.1;
        }
        let mut flags = Block::<u16, BSX, N>::new();
        for i in 0..N {
            flags.data[i] = match i % 5 {
                0 => 0x1, // solid
                1 => 0x4, // air (fluid under CELL_IS_FLUID_)
                2 => 0x1000, // void
                _ => 0x0, // fluid
            };
        }
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);

        let mut out = Block::<f32, BSX, N>::new();
        let mut ref_out = Block::<f32, BSX, N>::new();
        let dt = 0.05;

        kernel_laplacian::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, dt, &mut out);
        scalar_laplacian_7pt(&halo, &flags, dt, &mut ref_out);

        for i in 0..N {
            let d = (out.data[i] - ref_out.data[i]).abs();
            assert!(d < 1e-4, "mismatch at {i}: simd={} scalar={}", out.data[i], ref_out.data[i]);
        }
    }

    #[test]
    fn test_lap_03_mask_semantics_air_is_fluid() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        fill_halo_quadratic(&mut halo);

        let mut flags = Block::<u16, BSX, N>::new();
        flags.data[0] = 0x1; // solid -> preserved (== f0)
        flags.data[1] = 0x1000; // void -> preserved (== f0)
        flags.data[2] = 0x4; // air -> fluid -> updated
        flags.data[3] = 0x0; // fluid -> updated

        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);
        let mut out = Block::<f32, BSX, N>::new();
        let dt = 0.025;
        kernel_laplacian::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, dt, &mut out);

        // Cell 0 and 1 live in the halo interior (global block coords 0..16);
        // their f0 values are the halo center = quadratic field at (x,y,z).
        // f0(0,0,0)=0; f0 for cell 1 = (1,0,0)->1; cell 2 = (2,0,0)->4; cell 3=(3,0,0)->9.
        assert_eq!(out.data[0], 0.0); // solid preserved == f0 == 0
        assert_eq!(out.data[1], 1.0); // void preserved == f0 == 1
        // air (fluid): new = c + dt*L*G ; for f=x^2+y^2+z^2 at (2,0,0): c=4, L=6, G=2r=4 -> 4 + 0.025*24
        let expected_air = 4.0 + 0.025 * 6.0 * (2.0 * 2.0);
        assert!((out.data[2] - expected_air).abs() < 1e-4);
        let expected_fluid = 9.0 + 0.025 * 6.0 * (2.0 * 3.0);
        assert!((out.data[3] - expected_fluid).abs() < 1e-4);
    }

    #[test]
    fn test_lap_04_dt_zero_is_identity() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        fill_halo_quadratic(&mut halo);
        let flags = Block::<u16, BSX, N>::new();
        let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);

        let mut out = Block::<f32, BSX, N>::new();
        for v in out.data.iter_mut() {
            *v = -1.0;
        }
        kernel_laplacian::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, 0.0, &mut out);

        // dt=0 -> new == c (halo center), i.e. the quadratic field value.
        let mut out_idx = 0;
        for z in 1..=16 {
            for y in 1..=16 {
                for x in 1..=16 {
                    let gx = (x as i32 - 1) as f32;
                    let gy = (y as i32 - 1) as f32;
                    let gz = (z as i32 - 1) as f32;
                    assert!((out.data[out_idx] - (gx * gx + gy * gy + gz * gz)).abs() < 1e-5);
                    out_idx += 1;
                }
            }
        }
    }
}

//! 7-point Laplacian smoothing kernel.

use crate::math::stencil::{self, SimdRng, StoreBack};
use crate::multires::halo::HaloBlock;
use std::simd::num::SimdFloat;
use std::simd::{Simd, StdFloat};

/// Full-update 7-point Laplacian smoothing over a pre-staged faces-only halo,
/// writing the smoothed chunk back through the storage's [`StoreBack`].
///
/// Update `phi += dt * L * G` where `L = sum(6 neighbors) - 6*center` and `G`
/// is the gradient magnitude of the center stencil. No fluid mask (every voxel
/// is smoothed); an optional `[0, 1]` clamp (`doConstrZeroOne`) is applied
/// before the store.
#[inline(always)]
pub fn kernel_laplacian<const W: usize, const BSX: usize, const HSX: usize, D>(
    halo: &HaloBlock<BSX, HSX>,
    dt: f32,
    clamp01: bool,
    out: *mut D,
    rng: &mut SimdRng<W>,
) where
    D: StoreBack<W>,
{
    debug_assert_eq!(HSX, BSX + 2, "Laplacian needs a 1-voxel halo");
    debug_assert_eq!(BSX % W, 0, "BSX must be a multiple of W");

    let base = halo.data.as_ptr();
    let dy = HSX;
    let dz = HSX * HSX;

    let dtv = Simd::<f32, W>::splat(dt);
    let six = Simd::<f32, W>::splat(6.0);
    let half = Simd::<f32, W>::splat(0.5);
    let zero = Simd::<f32, W>::splat(0.0);
    let one = Simd::<f32, W>::splat(1.0);

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
                    let g = fx.mul_add(fx, fy.mul_add(fy, fz * fz)).sqrt();

                    let new = (dtv * lap).mul_add(g, c); // c + dt*lap*g
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
    const HSX: usize = 18;
    const N: usize = 4096;

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
    fn scalar_laplacian_7pt(halo: &HaloBlock<BSX, HSX>, dt: f32, out: &mut [f32; 4096]) {
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

                    out[out_idx] = new;
                    out_idx += 1;
                }
            }
        }
    }

    #[test]
    fn test_lap_01_quadratic_field_matches_scalar() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        fill_halo_quadratic(&mut halo);

        let mut out = Block::<f32, BSX, N>::new();
        let mut ref_out = [0.0f32; N];
        let dt = 0.025;
        let mut rng = SimdRng::<LANES>::seed(0);

        kernel_laplacian::<LANES, BSX, HSX, f32>(&halo, dt, false, out.data.as_mut_ptr(), &mut rng);
        scalar_laplacian_7pt(&halo, dt, &mut ref_out);

        for i in 0..N {
            let d = (out.data[i] - ref_out[i]).abs();
            assert!(d < 1e-4, "mismatch at {i}: simd={} scalar={}", out.data[i], ref_out[i]);
        }
    }

    #[test]
    fn test_lap_02_simd_matches_scalar_random_field() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        for (i, v) in halo.data.iter_mut().enumerate() {
            *v = ((i * 17) % 100) as f32 * 0.1;
        }

        let mut out = Block::<f32, BSX, N>::new();
        let mut ref_out = [0.0f32; N];
        let dt = 0.05;
        let mut rng = SimdRng::<LANES>::seed(1);

        kernel_laplacian::<LANES, BSX, HSX, f32>(&halo, dt, false, out.data.as_mut_ptr(), &mut rng);
        scalar_laplacian_7pt(&halo, dt, &mut ref_out);

        for i in 0..N {
            let d = (out.data[i] - ref_out[i]).abs();
            assert!(d < 1e-4, "mismatch at {i}: simd={} scalar={}", out.data[i], ref_out[i]);
        }
    }

    #[test]
    fn test_lap_03_dt_zero_is_identity() {
        let mut halo = HaloBlock::<BSX, HSX>::new();
        fill_halo_quadratic(&mut halo);

        let mut out = Block::<f32, BSX, N>::new();
        let mut rng = SimdRng::<LANES>::seed(2);
        kernel_laplacian::<LANES, BSX, HSX, f32>(&halo, 0.0, false, out.data.as_mut_ptr(), &mut rng);

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

    #[test]
    fn test_lap_04_clamp_zero_one() {
        // A field with magnitude >> 1 and a large dt drives `new` far out of
        // range; with clamp01 the result must stay within [0, 1].
        let mut halo = HaloBlock::<BSX, HSX>::new();
        for (i, v) in halo.data.iter_mut().enumerate() {
            *v = 100.0 + (i % 10) as f32;
        }
        let mut out = Block::<f32, BSX, N>::new();
        let mut rng = SimdRng::<LANES>::seed(3);
        kernel_laplacian::<LANES, BSX, HSX, f32>(&halo, 1000.0, true, out.data.as_mut_ptr(), &mut rng);
        for &v in &out.data {
            assert!((0.0..=1.0).contains(&v), "clamp01 violated: {v}");
        }
    }
}

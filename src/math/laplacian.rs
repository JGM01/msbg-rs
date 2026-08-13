use crate::blockpool::Block;
use crate::multires::halo::HaloBlock;
use std::simd::{Select, StdFloat};
use std::simd::{cmp::SimdPartialEq, f32x16, u16x16};

// Stolen from MSBG C++ `msbgcell.h`: CELL_SOLID (1), CELL_AIR (4), CELL_VOID (4096)
const FLUID_MASK: u16 = 1 | 4 | 4096;

/// Executes a branchless 7-point Laplacian stencil using AVX-512/AVX2 over a pre-staged halo buffer.
/// Hardcoded for 16-wide registers (BSX=16).
#[inline(always)]
pub fn kernel_laplacian_simd_16<const N: usize, const N_HALO: usize>(
    halo: &HaloBlock<f32, 16, 18, N_HALO>,
    flags_block: &Block<u16, 16, N>,
    output: &mut Block<f32, 16, N>,
) {
    const HSX: usize = 18;
    const DY: usize = HSX;
    const DZ: usize = HSX * HSX;

    let mut out_idx = 0;

    // Hoist loop-invariant constants
    let coeff = f32x16::splat(-6.0);
    let fluid_mask_splat = u16x16::splat(FLUID_MASK);

    for z in 1..=16 {
        for y in 1..=16 {
            let halo_idx = z * DZ + y * DY + 1;

            // Unaligned loads from the contiguous staging buffer
            let c = f32x16::from_slice(&halo.data[halo_idx..halo_idx + 16]);
            let lx = f32x16::from_slice(&halo.data[halo_idx - 1..halo_idx + 15]);
            let rx = f32x16::from_slice(&halo.data[halo_idx + 1..halo_idx + 17]);
            let dy_up = f32x16::from_slice(&halo.data[halo_idx - DY..halo_idx - DY + 16]);
            let dy_dn = f32x16::from_slice(&halo.data[halo_idx + DY..halo_idx + DY + 16]);
            let dz_bk = f32x16::from_slice(&halo.data[halo_idx - DZ..halo_idx - DZ + 16]);
            let dz_fd = f32x16::from_slice(&halo.data[halo_idx + DZ..halo_idx + DZ + 16]);

            let neighbors = lx + rx + dy_up + dy_dn + dz_bk + dz_fd;
            let laplacian = c.mul_add(coeff, neighbors); // FMA moment

            // Grab original output data and cell flags
            let old_val = f32x16::from_slice(&output.data[out_idx..out_idx + 16]);
            let cell_flags = u16x16::from_slice(&flags_block.data[out_idx..out_idx + 16]);

            // Fluid cells must not overlap with the solid/air/void mask.
            let is_fluid = (cell_flags & fluid_mask_splat).simd_eq(u16x16::splat(0));

            // Cast Mask<i16, 16> to Mask<i32, 16> to select the f32x16 vectors.
            let final_val = is_fluid.cast::<i32>().select(laplacian, old_val);

            // Write aligned output
            final_val.copy_to_slice(&mut output.data[out_idx..out_idx + 16]);

            out_idx += 16;
        }
    }
}

#[cfg(test)]
mod laplacian_tests {
    use super::*;
    use crate::blockpool::Block;
    use crate::multires::halo::HaloBlock;

    const BSX: usize = 16;
    const N: usize = 4096;
    const HSX: usize = 18; // halo size
    const N_HALO: usize = HSX * HSX * HSX; // 5832

    /// Slow scalar 7-point Laplacian (reference)
    fn scalar_laplacian_7pt(
        halo: &HaloBlock<f32, BSX, HSX, N_HALO>,
        flags: &Block<u16, BSX, N>,
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
                    let flag = flags.data[out_idx];

                    // Same mask logic as the SIMD kernel
                    let is_fluid = (flag & FLUID_MASK) == 0;
                    out.data[out_idx] = if is_fluid { lap } else { out.data[out_idx] };

                    out_idx += 1;
                }
            }
        }
    }

    /// Build a HaloBlock filled with a known analytic field: f = x² + y² + z²
    /// (Laplacian of this field is constantly 6)
    fn fill_halo_quadratic(halo: &mut HaloBlock<f32, BSX, HSX, N_HALO>) {
        // Halo coordinates run [-1..16] relative to the interior block
        for z in 0..HSX {
            for y in 0..HSX {
                for x in 0..HSX {
                    let gx = (x as i32 - 1) as f32; // map to global-ish coords
                    let gy = (y as i32 - 1) as f32;
                    let gz = (z as i32 - 1) as f32;
                    let idx = z * HSX * HSX + y * HSX + x;
                    halo.data[idx] = gx * gx + gy * gy + gz * gz;
                }
            }
        }
    }

    #[test]
    fn test_lap_01_quadratic_field_laplacian_is_six() {
        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        fill_halo_quadratic(&mut halo);

        // all zeros = fluid
        let flags = Block::<u16, BSX, N>::new();
        let mut output = Block::<f32, BSX, N>::new();

        // Seed output with garbage so writes are detectable
        for v in output.data.iter_mut() {
            *v = 12345.0;
        }

        kernel_laplacian_simd_16(&halo, &flags, &mut output);

        // Every fluid cell must be ~6.0
        for &v in output.data.iter() {
            assert!((v - 6.0).abs() < 1e-4, "got {}", v);
        }
    }

    #[test]
    fn test_lap_02_mask_preserves_solid_cells() {
        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        fill_halo_quadratic(&mut halo);

        let mut flags = Block::<u16, BSX, N>::new();
        // Mark the first 128 cells as solid (CELL_SOLID = 1)
        for i in 0..128 {
            flags.data[i] = 1;
        }

        let mut output = Block::<f32, BSX, N>::new();

        // Put a sentinel value that must survive for solid cells
        for v in output.data.iter_mut() {
            *v = -999.0;
        }

        kernel_laplacian_simd_16(&halo, &flags, &mut output);

        // Solid region unchanged
        for i in 0..128 {
            assert_eq!(output.data[i], -999.0, "solid cell {} was overwritten", i);
        }

        // Fluid region still ~6.0
        for i in 128..N {
            assert!((output.data[i] - 6.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_lap_03_simd_matches_scalar_reference() {
        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        // Random-ish but deterministic field
        for (i, v) in halo.data.iter_mut().enumerate() {
            *v = ((i * 17) % 100) as f32 * 0.1;
        }

        let mut flags = Block::<u16, BSX, N>::new();
        // fill with fluid / solid / air
        for i in 0..N {
            flags.data[i] = match i % 5 {
                0 => 1,    // solid
                1 => 4,    // air
                2 => 4096, // void
                _ => 0,    // fluid
            };
        }

        let mut out_simd = Block::<f32, BSX, N>::new();
        let mut out_scalar = Block::<f32, BSX, N>::new();

        // Same initial values
        for i in 0..N {
            out_simd.data[i] = (i as f32) * 0.01;
            out_scalar.data[i] = out_simd.data[i];
        }

        kernel_laplacian_simd_16(&halo, &flags, &mut out_simd);
        scalar_laplacian_7pt(&halo, &flags, &mut out_scalar);

        for i in 0..N {
            let diff = (out_simd.data[i] - out_scalar.data[i]).abs();
            assert!(
                diff < 1e-5,
                "mismatch at {}: simd={} scalar={}",
                i,
                out_simd.data[i],
                out_scalar.data[i]
            );
        }
    }

    #[test]
    fn test_lap_04_air_and_void_also_masked() {
        // Exercise the other two bits of FLUID_MASK
        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        fill_halo_quadratic(&mut halo);

        let mut flags = Block::<u16, BSX, N>::new();
        flags.data[0] = 4; // CELL_AIR
        flags.data[1] = 4096; // CELL_VOID
        flags.data[2] = 0; // fluid

        let mut output = Block::<f32, BSX, N>::new();
        output.data[0] = 111.0;
        output.data[1] = 222.0;
        output.data[2] = 333.0;

        kernel_laplacian_simd_16(&halo, &flags, &mut output);

        assert_eq!(output.data[0], 111.0); // air preserved
        assert_eq!(output.data[1], 222.0); // void preserved
        assert!((output.data[2] - 6.0).abs() < 1e-4); // fluid updated
    }
}

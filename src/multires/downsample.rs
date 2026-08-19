//! Coarse/fine level transfers: downsampling and coarse ghost sampling.

use crate::multires::level::LevelData;
use rayon::prelude::*;

/// 2^3 box average downsample: each coarse voxel is the mean of its 8 fine
/// children. Mirrors the C++ `OPT_SIMPLE_AVERAGE` path of `downsampleChannel`
/// (`krnDownWidth = 2`, `krn = 1/8`). `src` is the finer level (block size
/// `BSX`), `dst` the coarser (block size `BSX / 2`); both share the same block
/// grid, so coarse block `bid` is the average of fine block `bid`. Only the
/// fine level's present blocks are visited (the sparse map), so the cost scales
/// with active blocks, not the virtual block count.
pub fn downsample_channel_avg<const BSX: usize, const N: usize, const BSC: usize, const NC: usize>(
    dst: &mut LevelData<BSC, NC>,
    src: &LevelData<BSX, N>,
) {
    debug_assert_eq!(BSX, BSC * 2, "coarse block size must be half the fine");
    debug_assert_eq!(dst.n_blocks, src.n_blocks, "levels share the block grid");

    let fine_log2 = BSX.trailing_zeros();

    // Materialize the coarse counterpart of every fine block that exists, and
    // resolve both raw pointers up front (stored as usize so the job vector is
    // Send for the parallel pass).
    let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
    for bid in src.active_block_ids() {
        dst.ensure_block(bid);
        let dp = dst.density_ptr_mut(bid) as usize;
        let sp = src.density_ptr(bid) as usize;
        jobs.push((bid, sp, dp));
    }

    jobs.into_par_iter().for_each(|(_bid, sp, dp)| {
        let sp = sp as *const f32;
        let dp = dp as *mut f32;
        debug_assert!(!sp.is_null() && !dp.is_null(), "downsample pointers must be live");
        let mut vid = 0usize;
        for cz in 0..BSC {
            for cy in 0..BSC {
                for cx in 0..BSC {
                    let fx = cx << 1;
                    let fy = cy << 1;
                    let fz = cz << 1;
                    let mut sum = 0.0f32;
                    for dz in 0..2usize {
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                let i = (fx + dx) | ((fy + dy) << fine_log2) | ((fz + dz) << (2 * fine_log2));
                                sum += unsafe { *sp.add(i) };
                            }
                        }
                    }
                    unsafe { *dp.add(vid) = sum / 8.0 };
                    vid += 1;
                }
            }
        }
    });
}

/// Read a coarse value at a fine-level voxel coordinate (the coarse ghost /
/// upsampling primitive used by the multires halo fill): the coarse grid holds
/// the same physical domain at half the resolution, so a fine coordinate maps to
/// `(x >> 1, y >> 1, z >> 1)`. Out-of-domain fine coordinates are clamped to the
/// coarse domain (C++ `clipGridCoords`).
#[inline(always)]
pub fn sample_coarse<const BSC: usize, const NC: usize>(
    coarse: &LevelData<BSC, NC>,
    gx: i32,
    gy: i32,
    gz: i32,
) -> f32 {
    let x = (gx >> 1).clamp(0, coarse.sx as i32 - 1) as usize;
    let y = (gy >> 1).clamp(0, coarse.sy as i32 - 1) as usize;
    let z = (gz >> 1).clamp(0, coarse.sz as i32 - 1) as usize;
    let coarse_log2 = BSC.trailing_zeros();
    let coarse_mask = BSC - 1;
    let bid = (x >> coarse_log2) + (y >> coarse_log2) * coarse.nx + (z >> coarse_log2) * coarse.nxy;
    let vid =
        (x & coarse_mask) | ((y & coarse_mask) << coarse_log2) | ((z & coarse_mask) << (2 * coarse_log2));
    unsafe { *coarse.density_ptr(bid).add(vid) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multires::Level;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn coarse_voxel(coarse: &LevelData<8, 512>, x: usize, y: usize, z: usize) -> f32 {
        sample_coarse::<8, 512>(coarse, (x * 2) as i32, (y * 2) as i32, (z * 2) as i32)
    }

    #[test]
    fn test_ds_01_constant_field_exact() {
        let mut g = crate::multires::MultiresGrid::create("t", 32, 32, 32, 16, 2, 6);
        let (l0, l1) = g.levels.split_at_mut(1);
        let fine = match &mut l0[0] {
            Level::B16(f) => f,
            _ => unreachable!(),
        };
        let coarse = match &mut l1[0] {
            Level::B8(c) => c,
            _ => unreachable!(),
        };
        for bid in 0..fine.n_blocks {
            fine.ensure_block(bid);
            let p = fine.density_ptr_mut(bid);
            for v in 0..N {
                unsafe { *p.add(v) = 0.5 };
            }
        }
        downsample_channel_avg::<16, 4096, 8, 512>(coarse, fine);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    assert!((coarse_voxel(coarse, x, y, z) - 0.5).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_ds_02_linear_gradient() {
        // f(x) = x downsamples to coarse[x] = (2x + 2x+1)/2 = 2x + 0.5.
        let mut g = crate::multires::MultiresGrid::create("t", 32, 16, 16, 16, 2, 6);
        let (l0, l1) = g.levels.split_at_mut(1);
        let fine = match &mut l0[0] {
            Level::B16(f) => f,
            _ => unreachable!(),
        };
        let coarse = match &mut l1[0] {
            Level::B8(c) => c,
            _ => unreachable!(),
        };
        for bid in 0..fine.n_blocks {
            fine.ensure_block(bid);
            let p = fine.density_ptr_mut(bid);
            for z in 0..BSX {
                for y in 0..BSX {
                    for x in 0..BSX {
                        let gx = (bid % fine.nx) * BSX + x;
                        unsafe { *p.add(x + y * BSX + z * BSX * BSX) = gx as f32 };
                    }
                }
            }
        }
        downsample_channel_avg::<16, 4096, 8, 512>(coarse, fine);
        for x in 0..16 {
            assert!((coarse_voxel(coarse, x, 0, 0) - (2.0 * x as f32 + 0.5)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ds_03_coarse_sample_matches_downsample() {
        let mut g = crate::multires::MultiresGrid::create("t", 32, 16, 16, 16, 2, 6);
        let (l0, l1) = g.levels.split_at_mut(1);
        let fine = match &mut l0[0] {
            Level::B16(f) => f,
            _ => unreachable!(),
        };
        let coarse = match &mut l1[0] {
            Level::B8(c) => c,
            _ => unreachable!(),
        };
        for bid in 0..fine.n_blocks {
            fine.ensure_block(bid);
            let p = fine.density_ptr_mut(bid);
            for z in 0..BSX {
                for y in 0..BSX {
                    for x in 0..BSX {
                        let gx = (bid % fine.nx) * BSX + x;
                        let gy = (bid / fine.nx) * BSX + y;
                        unsafe { *p.add(x + y * BSX + z * BSX * BSX) = (gx * gy + z) as f32 };
                    }
                }
            }
        }
        downsample_channel_avg::<16, 4096, 8, 512>(coarse, fine);
        for cz in 0..8i32 {
            for cy in 0..8i32 {
                for cx in 0..16i32 {
                    // sample_coarse at (2cx, 2cy, 2cz) == the coarse voxel at (cx, cy, cz)
                    let got = sample_coarse::<8, 512>(coarse, cx * 2, cy * 2, cz * 2);
                    let expect = coarse_voxel(coarse, cx as usize, cy as usize, cz as usize);
                    assert_eq!(got, expect);
                }
            }
        }
    }

    #[test]
    fn test_ds_04_out_of_domain_clamps() {
        let mut g = crate::multires::MultiresGrid::create("t", 16, 16, 16, 16, 2, 6);
        let coarse = match &mut g.levels[1] {
            Level::B8(c) => c,
            _ => unreachable!(),
        };
        coarse.ensure_block(0);
        let p = coarse.density_ptr_mut(0);
        unsafe {
            *p = 7.0; // coarse (0,0,0)
            *p.add(511) = 9.0; // coarse (7,7,7)
        }
        assert_eq!(sample_coarse::<8, 512>(coarse, -3, -3, -3), 7.0);
        assert_eq!(sample_coarse::<8, 512>(coarse, 100, 100, 100), 9.0);
    }
}

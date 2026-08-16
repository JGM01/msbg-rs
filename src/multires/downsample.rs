//! Coarse/fine level transfers: downsampling and coarse ghost sampling.

use crate::sparse_grid::SparseGrid;
use rayon::prelude::*;

/// 2^3 box average downsample: each coarse voxel is the mean of its 8 fine
/// children. Mirrors the C++ `OPT_SIMPLE_AVERAGE` path of `downsampleChannel`
/// (`krnDownWidth = 2`, `krn = 1/8`). `src` is the finer level (block size
/// `BSX`), `dst` the coarser (block size `BSX / 2`); both share the same block
/// grid, so coarse block `bid` is the average of fine block `bid`.
pub fn downsample_channel_avg<const BSX: usize, const N: usize, const BSC: usize, const NC: usize>(
    dst: &mut SparseGrid<f32, BSC, NC>,
    src: &SparseGrid<f32, BSX, N>,
) {
    debug_assert_eq!(BSX, BSC * 2, "coarse block size must be half the fine");
    debug_assert_eq!(dst.n_blocks, src.n_blocks, "levels share the block grid");

    let fine_log2 = BSX.trailing_zeros();

    // Materialize coarse blocks + resolve the fine source pointers.
    let mut src_data: Vec<usize> = vec![0; src.n_blocks];
    let mut dst_data: Vec<usize> = vec![0; dst.n_blocks];
    for bid in 0..src.n_blocks {
        src_data[bid] = match src.blockmap[bid] {
            Some(p) if p != src.empty_block && p != src.full_block => unsafe { (*p.as_ptr()).data.as_ptr() as usize },
            _ => 0,
        };
        dst.ensure_block(bid);
        dst_data[bid] = match dst.blockmap[bid] {
            Some(p) => unsafe { (*p.as_ptr()).data.as_mut_ptr() as usize },
            None => unreachable!("ensure_block left a block unmaterialized"),
        };
    }

    (0..dst.n_blocks).into_par_iter().for_each(|bid| {
        let sp = src_data[bid] as *const f32;
        let dp = dst_data[bid] as *mut f32;
        let mut vid = 0usize;
        for cz in 0..BSC {
            for cy in 0..BSC {
                for cx in 0..BSC {
                    let fx = cx << 1;
                    let fy = cy << 1;
                    let fz = cz << 1;
                    let mut sum = 0.0f32;
                    if sp.is_null() {
                        sum = 0.0;
                    } else {
                        for dz in 0..2usize {
                            for dy in 0..2usize {
                                for dx in 0..2usize {
                                    let i = (fx + dx) | ((fy + dy) << fine_log2) | ((fz + dz) << (2 * fine_log2));
                                    sum += unsafe { *sp.add(i) };
                                }
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
    coarse: &SparseGrid<f32, BSC, NC>,
    gx: i32,
    gy: i32,
    gz: i32,
) -> f32 {
    let x = (gx >> 1).clamp(0, coarse.sx as i32 - 1) as usize;
    let y = (gy >> 1).clamp(0, coarse.sy as i32 - 1) as usize;
    let z = (gz >> 1).clamp(0, coarse.sz as i32 - 1) as usize;
    coarse.get_voxel(x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use std::sync::Arc;

    #[test]
    fn test_ds_01_constant_field_exact() {
        // A constant fine field downsamples to the same constant.
        let fine_pool = Arc::new(BlockPool::<f32, 16, 4096>::new(8, 32));
        let mut fine = SparseGrid::new("f".into(), 32, 32, 32, 0.0, 1.0, fine_pool);
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    fine.set_voxel(x, y, z, 0.5);
                }
            }
        }
        let coarse_pool = Arc::new(BlockPool::<f32, 8, 512>::new(8, 32));
        let mut coarse = SparseGrid::new("c".into(), 16, 16, 16, 0.0, 1.0, coarse_pool);
        downsample_channel_avg::<16, 4096, 8, 512>(&mut coarse, &fine);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    assert!((coarse.get_voxel(x, y, z) - 0.5).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_ds_02_linear_gradient() {
        // f(x) = x downsamples to coarse[x] = (2x + 2x+1)/2 = 2x + 0.5.
        let fine_pool = Arc::new(BlockPool::<f32, 16, 4096>::new(8, 32));
        let mut fine = SparseGrid::new("f".into(), 32, 16, 16, 0.0, 1.0, fine_pool);
        for x in 0..32 {
            for z in 0..16 {
                for y in 0..16 {
                    fine.set_voxel(x, y, z, x as f32);
                }
            }
        }
        let coarse_pool = Arc::new(BlockPool::<f32, 8, 512>::new(8, 32));
        let mut coarse = SparseGrid::new("c".into(), 16, 8, 8, 0.0, 1.0, coarse_pool);
        downsample_channel_avg::<16, 4096, 8, 512>(&mut coarse, &fine);
        for x in 0..16 {
            assert!((coarse.get_voxel(x, 0, 0) - (2.0 * x as f32 + 0.5)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ds_03_coarse_sample_matches_downsample() {
        // "coarse sample == fine downsample": sampling the coarse field at a
        // coarse voxel via `sample_coarse` (coords >> 1) equals the average.
        let fine_pool = Arc::new(BlockPool::<f32, 16, 4096>::new(8, 32));
        let mut fine = SparseGrid::new("f".into(), 32, 16, 16, 0.0, 1.0, fine_pool);
        for x in 0..32 {
            for z in 0..16 {
                for y in 0..16 {
                    fine.set_voxel(x, y, z, (x * y + z) as f32);
                }
            }
        }
        let coarse_pool = Arc::new(BlockPool::<f32, 8, 512>::new(8, 32));
        let mut coarse = SparseGrid::new("c".into(), 16, 8, 8, 0.0, 1.0, coarse_pool);
        downsample_channel_avg::<16, 4096, 8, 512>(&mut coarse, &fine);

        // Fine coords (2*cx, 2*cy, 2*cz) map to coarse (cx, cy, cz).
        for cz in 0..8i32 {
            for cy in 0..8i32 {
                for cx in 0..16i32 {
                    let s = sample_coarse::<8, 512>(&coarse, cx * 2, cy * 2, cz * 2);
                    let expect = coarse.get_voxel(cx as usize, cy as usize, cz as usize);
                    assert_eq!(s, expect);
                }
            }
        }
    }

    #[test]
    fn test_ds_04_out_of_domain_clamps() {
        let coarse_pool = Arc::new(BlockPool::<f32, 8, 512>::new(8, 32));
        let mut coarse = SparseGrid::new("c".into(), 16, 16, 16, 0.0, 1.0, coarse_pool);
        coarse.set_voxel(0, 0, 0, 7.0);
        coarse.set_voxel(15, 15, 15, 9.0);
        assert_eq!(sample_coarse::<8, 512>(&coarse, -3, -3, -3), 7.0);
        assert_eq!(sample_coarse::<8, 512>(&coarse, 100, 100, 100), 9.0);
    }
}

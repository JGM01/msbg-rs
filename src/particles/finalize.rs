//! Finalize: convert the splatted per-voxel normalized squared-distance field
//! into a `[0, 1]` density ready for the smoother.
//!
//! Replicates the C++ `msbg_test_sparse` finalize loop (distSq → signed distance
//! → linear-step density, iso 0.5) but SIMD over x rows:
//! `f = 1 - clamp((sqrt(ratio*distSqMax) - rParticle + rParticle) / (2*rParticle+nbDist))`.
//! Untouched voxels (`ratio >= 0.999`) become density 0.

use rayon::prelude::*;

use super::SurfaceConfig;
use crate::channel::Quant;
use crate::math::simd::LANES;
use crate::sparse_grid::SparseGrid;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::{Select, Simd, StdFloat};

/// Threshold on the stored normalized distance ratio below which a voxel is
/// "touched" (C++ `distSq < 0.999f`).
pub const TOUCHED_RATIO: f32 = 0.999;

#[inline(always)]
fn block_data_ptr<D, const BSX: usize, const N: usize>(
    grid: &SparseGrid<D, BSX, N>,
    bid: usize,
) -> *mut D
where
    D: Copy + Default + Send + Sync,
{
    match grid.blockmap[bid] {
        Some(p) if p != grid.empty_block && p != grid.full_block => {
            unsafe { (*p.as_ptr()).data.as_mut_ptr() }
        }
        _ => std::ptr::null_mut(),
    }
}

/// Convert every active block's splat field into density in place.
pub fn finalize<D, const BSX: usize, const N: usize>(
    grid: &SparseGrid<D, BSX, N>,
    active: &[u32],
    cfg: &SurfaceConfig,
) where
    D: Quant + Copy + Default + Send + Sync,
{
    debug_assert_eq!(BSX % LANES, 0, "LANES must divide BSX");
    let r_scan = cfg.r_scan();
    let dist_sq_max = r_scan * r_scan;
    let r_particle = cfg.r_particle;
    let linstep_denom = 2.0 * r_particle + cfg.nb_dist;

    active.par_iter().for_each(|&bid| {
        let data = block_data_ptr(grid, bid as usize);
        if data.is_null() {
            return;
        }
        let mut vid = 0usize;
        for _vz in 0..BSX {
            for _vy in 0..BSX {
                for _vx in (0..BSX).step_by(LANES) {
                    let ratio = unsafe { D::dequant_chunk::<LANES>(data, vid) };
                    let touched = ratio.simd_lt(Simd::splat(TOUCHED_RATIO));
                    let dist_sq = ratio * Simd::splat(dist_sq_max);
                    let dist = dist_sq.sqrt() - Simd::splat(r_particle);
                    let t = (dist + Simd::splat(r_particle)) / Simd::splat(linstep_denom);
                    let t = t.simd_clamp(Simd::splat(0.0), Simd::splat(1.0));
                    let f = Simd::splat(1.0) - t;
                    let out = touched.select(f, Simd::splat(0.0));
                    unsafe { D::quant_chunk::<LANES>(data, vid, out) };
                    vid += LANES;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use crate::channel::Density;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn setup(sx: usize, sy: usize, sz: usize) -> SparseGrid<Density, BSX, N> {
        let pool = Arc::new(BlockPool::<Density, BSX, N>::new(64, 64));
        SparseGrid::new("finalize".into(), sx, sy, sz, Density(0), Density(u16::MAX), pool)
    }

    fn cfg() -> SurfaceConfig {
        SurfaceConfig {
            sx: 32,
            sy: 32,
            sz: 32,
            n_instances: 1,
            instance_scale_factor: 0.5,
            r_particle: 2.0,
            nb_dist: 2.0,
            n_smooth_iters: 0,
            smooth_dt: 0.05,
        }
    }

    fn get(grid: &SparseGrid<Density, BSX, N>, x: usize, y: usize, z: usize) -> f32 {
        grid.get_voxel(x, y, z).to_f32()
    }

    // A single particle: density 1 at the center, 2/3 at the surface
    // (dist = rParticle), ~1/3 just below the touched threshold, 0 beyond.
    #[test]
    fn finalize_01_single_particle() {
        let mut grid = setup(32, 32, 32);
        grid.set_voxel(8, 8, 8, Density(0)); // center (distSq ratio 0)
        let r = 16.0f32; // distSqMax
        grid.set_voxel(6, 8, 8, Density(((2.5f32 * 2.5) / r * 65535.0).round() as u16)); // dx=2.5
        grid.set_voxel(4, 8, 8, Density(65468)); // ratio ~0.99898, just touched
        let active = vec![0u32];
        finalize(&grid, &active, &cfg());

        // Center: sqrt(0)-2 = -2 -> f = 1 - clamp(0/6) = 1.
        assert!((get(&grid, 8, 8, 8) - 1.0).abs() < 1e-4);
        // dx=2.5: distSq ratio 0.390625 -> distSq=6.25 -> dist=2.5-2=0.5 -> t=2.5/6 -> f=1-2.5/6.
        let want = 1.0 - (0.5 + 2.0) / 6.0;
        assert!((get(&grid, 6, 8, 8) - want).abs() < 1e-4);
        // Ratio just below 0.999: f = 1 - sqrt(ratio*16)/6 ~ 1/3.
        let ratio = 65468.0f32 / 65535.0;
        let want = 1.0 - (ratio * r).sqrt() / 6.0;
        assert!((get(&grid, 4, 8, 8) - want).abs() < 1e-4, "got {} want {}", get(&grid, 4, 8, 8), want);
    }

    // Untouched voxels (stored u16::MAX) -> 0.
    #[test]
    fn finalize_02_untouched_is_zero() {
        let mut grid = setup(32, 32, 32);
        grid.set_voxel(3, 3, 3, Density(u16::MAX));
        let active = vec![0u32];
        finalize(&grid, &active, &cfg());
        assert_eq!(get(&grid, 3, 3, 3), 0.0);
    }

    // The 0.999 threshold edge: ratio just below stays touched, just above is 0.
    #[test]
    fn finalize_03_threshold_edge() {
        let mut grid = setup(32, 32, 32);
        let below = ((0.9989f32) * 65535.0).round() as u16;
        let above = ((0.9991f32) * 65535.0).round() as u16;
        grid.set_voxel(0, 0, 0, Density(below));
        grid.set_voxel(1, 0, 0, Density(above));
        let active = vec![0u32];
        finalize(&grid, &active, &cfg());
        assert!(get(&grid, 0, 0, 0) > 0.0, "just below threshold is touched");
        assert_eq!(get(&grid, 1, 0, 0), 0.0, "just above threshold is zero");
    }

    // Full block with mixed values; all stay in [0,1].
    #[test]
    fn finalize_04_mixed_block_in_range() {
        let mut grid = setup(16, 16, 16);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let ratio = ((x + y + z) % 17) as f32 / 17.0;
                    grid.set_voxel(x, y, z, Density((ratio * 65535.0).round() as u16));
                }
            }
        }
        let active = vec![0u32];
        finalize(&grid, &active, &cfg());
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let v = get(&grid, x, y, z);
                    assert!((0.0..=1.0).contains(&v), "({x},{y},{z}) = {v}");
                }
            }
        }
    }

    // Awkward: empty active list is a no-op.
    #[test]
    fn finalize_05_empty_active() {
        let grid = setup(16, 16, 16);
        finalize(&grid, &[], &cfg());
    }
}

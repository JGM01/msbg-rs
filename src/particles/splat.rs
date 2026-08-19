//! 8-color lock-free particle splatting with thread-local staging.
//!
//! The naive scatter (the C++ `msbg_test_sparse` path) does a per-voxel
//! `min`-RMW directly into the grid for every overlapping particle. For a
//! dense point cloud each voxel is touched by many particles, so the same
//! grid cache lines are read-modify-written dozens of times and the splat
//! becomes DRAM-latency bound (measured ~17 min for 16.7M particles on this
//! machine vs ~0.2 s for the staged path).
//!
//! Instead, each block's particle chunk is staged into a thread-local
//! `(BSX + 2·margin)³` buffer (`margin = ceil(rScan)`), where the min
//! accumulation runs in L1. The commit then writes each *touched* voxel to
//! the grid exactly once per contributing block. Voxels near block boundaries
//! are staged in up to 8 blocks' buffers (their margins) and committed in the
//! *other* blocks' color passes — the 8-coloring makes the whole thing
//! race-free with `rScan < BSX`, exactly as the direct scatter, but with the
//! RMW count reduced from `#overlapping particles` to `#contributing blocks`.

use std::cell::UnsafeCell;
use std::simd::cmp::SimdPartialOrd;
use std::simd::Simd;

use rayon::prelude::*;

use super::sort::Bucketed;
use super::{GridDims, SurfaceConfig};
use crate::channel::Quant;
use crate::sparse_grid::SparseGrid;

/// Precondition for the 8-color race-freedom: a particle's footprint along one
/// axis (`±(rScan + 0.5)`) must be narrower than half the gap between
/// same-color center blocks (`BSX`).
#[inline(always)]
fn check_r_scan(r_scan: f32, margin: usize) {
    debug_assert!(
        r_scan <= margin as f32,
        "rScan ({r_scan}) exceeds the staging margin ({margin}); raise the MSX const"
    );
    debug_assert!(
        r_scan + 0.5 < margin as f32 * 2.0,
        "rScan must be < BSX for the 8-color scheme to be race-free"
    );
}

/// Raw writable payload pointer for an allocated value block, or null.
#[inline(always)]
fn block_data_ptr<D, const BSX: usize, const N: usize>(
    grid: &SparseGrid<D, BSX, N>,
    bid: usize,
) -> *mut D
where
    D: Copy + Default + Send + Sync,
{
    grid.value_block_ptr_mut(bid)
}

/// Thread-local staging buffer for one block's splat.
struct SplatSlot<D, const MSX: usize> {
    staging: Box<[D]>,
}

struct SplatPool<D, const MSX: usize> {
    slots: Vec<UnsafeCell<SplatSlot<D, MSX>>>,
}

// Each slot is exclusively accessed by a single physical Rayon thread.
unsafe impl<D, const MSX: usize> Sync for SplatPool<D, MSX> {}

impl<D: Copy + Default, const MSX: usize> SplatPool<D, MSX> {
    fn new(num_threads: usize) -> Self {
        let slots = (0..=num_threads)
            .map(|_| UnsafeCell::new(SplatSlot {
                staging: vec![D::default(); MSX * MSX * MSX].into_boxed_slice(),
            }))
            .collect();
        Self { slots }
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut SplatSlot<D, MSX> {
        let idx = rayon::current_thread_index().unwrap_or(self.slots.len() - 1);
        debug_assert!(idx < self.slots.len());
        unsafe { &mut *self.slots[idx].get() }
    }
}

/// Splat every bucketed particle into `grid` (all active blocks pre-filled
/// with `D::full()`), via 8 color passes. `MSX` is the staging extent: a
/// multiple of 16 (for aligned SIMD rows) at least `BSX + 2·ceil(rScan)`
/// (`DEFAULT_MSX` for the demo radii).
pub fn splat<D, const BSX: usize, const N: usize, const MSX: usize>(
    grid: &SparseGrid<D, BSX, N>,
    bucketed: &Bucketed,
    cfg: &SurfaceConfig,
) where
    D: Quant + Copy + Default + Send + Sync,
{
    debug_assert_eq!(MSX % 8, 0, "MSX must be a multiple of 8 for SIMD8 staging");
    let margin = cfg.r_scan().ceil() as usize;
    debug_assert!(
        MSX >= BSX + 2 * margin,
        "MSX ({MSX}) too small for rScan {} (needs {})",
        cfg.r_scan(),
        BSX + 2 * margin
    );
    check_r_scan(cfg.r_scan(), margin);
    let r_scan = cfg.r_scan();
    let dist_sq_max = r_scan * r_scan;
    let dist_sq_max_inv = 1.0 / dist_sq_max;

    let dims = GridDims::new(grid.sx, grid.sy, grid.sz, BSX);

    let mut by_color: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
    for &bid in &bucketed.particle_blocks {
        let (bx, by, bz) = dims.coords(bid);
        let color = (bx & 1) | ((by & 1) << 1) | ((bz & 1) << 2);
        by_color[color].push(bid);
    }

    let pool = SplatPool::<D, MSX>::new(rayon::current_num_threads());

    for bucket in &by_color {
        bucket.par_iter().for_each(|&bid| {
            let start = bucketed.block_start[bid];
            let end = bucketed.block_start[bid + 1];
            let pts = &bucketed.positions[start..end];
            let slot = unsafe { pool.get_mut() };
            stage_and_commit::<D, BSX, N, MSX>(
                grid,
                &dims,
                slot,
                bid,
                pts,
                r_scan,
                dist_sq_max,
                dist_sq_max_inv,
                margin,
            );
        });
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn stage_and_commit<D, const BSX: usize, const N: usize, const MSX: usize>(
    grid: &SparseGrid<D, BSX, N>,
    dims: &GridDims,
    slot: &mut SplatSlot<D, MSX>,
    bid: usize,
    pts: &[[f32; 3]],
    r_scan: f32,
    dist_sq_max: f32,
    dist_sq_max_inv: f32,
    margin: usize,
) where
    D: Quant + Copy + Default + Send + Sync,
{
    let (bx, by, bz) = dims.coords(bid);
    let b0x = (bx * BSX) as i32;
    let b0y = (by * BSX) as i32;
    let b0z = (bz * BSX) as i32;
    let m = margin as i32;

    let staging = &mut slot.staging;
    staging.fill(D::full());

    // ---- Stage: accumulate every particle's footprint into the L1 buffer ----
    // 16-lane SIMD over each x-run; the buffer row is MSX-wide (multiple of 16)
    // so an aligned 16-lane chunk never crosses a row boundary.
    let x0s = b0x - m;
    let y0s = b0y - m;
    let z0s = b0z - m;
    let window_hi = (BSX as i32 + m) - 1;
    let staging_ptr = staging.as_mut_ptr();
    // Constant lane-index vectors, reused for every chunk.
    let lanes_f32 = Simd::<f32, 8>::from_array(std::array::from_fn(|i| i as f32));
    let lanes_i32 = Simd::<i32, 8>::from_array(std::array::from_fn(|i| i as i32));
    for &p in pts {
        let px0 = p[0] - 0.5;
        let py0 = p[1] - 0.5;
        let pz0 = p[2] - 0.5;

        // Footprint global voxel range, clipped to the staging window.
        let ix1 = (((p[0] - r_scan - 0.5).ceil() as i32).max(b0x - m)).min(b0x + window_hi);
        let ix2 = (((p[0] + r_scan - 0.5).floor() as i32).max(b0x - m)).min(b0x + window_hi);
        let iy1 = (((p[1] - r_scan - 0.5).ceil() as i32).max(b0y - m)).min(b0y + window_hi);
        let iy2 = (((p[1] + r_scan - 0.5).floor() as i32).max(b0y - m)).min(b0y + window_hi);
        let iz1 = (((p[2] - r_scan - 0.5).ceil() as i32).max(b0z - m)).min(b0z + window_hi);
        let iz2 = (((p[2] + r_scan - 0.5).floor() as i32).max(b0z - m)).min(b0z + window_hi);

        for iz in iz1..=iz2 {
            let dz = iz as f32 - pz0;
            let dz2 = dz * dz;
            let sz = (iz - z0s) as usize;
            for iy in iy1..=iy2 {
                let dy = iy as f32 - py0;
                let dyz = dz2 + dy * dy;
                let sy = (iy - y0s) as usize;
                let row = (sz * MSX + sy) * MSX;
                let o1 = (ix1 - x0s) as usize;
                let o2 = (ix2 - x0s) as usize;
                let mut c = o1 & !7;
                while c <= o2 {
                    // dx must match the scalar `ix as f32 - px0` exactly:
                    // `(cx + lane) - px0`, not `(cx - px0) + lane`.
                    let cx = (c as i32 + x0s) as f32;
                    let dx = (Simd::splat(cx) + lanes_f32) - Simd::splat(px0);
                    let dist_sq = dx * dx + Simd::splat(dyz);
                    let llo = o1.saturating_sub(c).min(15);
                    let lhi = o2.saturating_sub(c).min(15);
                    let in_run = lanes_i32.simd_ge(Simd::splat(llo as i32))
                        & lanes_i32.simd_le(Simd::splat(lhi as i32));
                    unsafe {
                        D::stage_chunk::<8>(
                            staging_ptr,
                            row + c,
                            dist_sq,
                            dist_sq_max,
                            dist_sq_max_inv,
                            in_run,
                        )
                    };
                    c += 8;
                }
            }
        }
    }

    // ---- Commit: write each touched staging voxel to its real block -------
    // The staging window decomposes into a 3×3×3 grid of regions (interior
    // 16³, six 4-deep faces, twelve edges, eight corners); each region maps to
    // a fixed neighbor block with contiguous x-runs, so a SIMD `commit_chunk`
    // amortizes one cache-line miss over W voxels.
    let bsx_log2 = BSX.trailing_zeros();
    let bsx_mask = BSX as i32 - 1;
    let bsx_i = BSX as i32;
    let margin_i = m;
    let mut neighbor = [0usize; 27];
    for dz in -1..=1i32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let bx = (bx as i32 + dx).clamp(0, dims.nx as i32 - 1);
                let by = (by as i32 + dy).clamp(0, dims.ny as i32 - 1);
                let bz = (bz as i32 + dz).clamp(0, dims.nz as i32 - 1);
                let idx = (((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)) as usize;
                neighbor[idx] = (bx as usize) + (by as usize) * dims.nx + (bz as usize) * dims.nxy;
            }
        }
    }

    for dz in -1..=1i32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                // Staging ranges for this region (interior = [margin, margin+BSX)).
                let (xlo, xhi) = region_range(dx, margin_i, bsx_i);
                let (ylo, yhi) = region_range(dy, margin_i, bsx_i);
                let (zlo, zhi) = region_range(dz, margin_i, bsx_i);
                let xlen = (xhi - xlo) as usize;

                // Global coords of the staging region's origin.
                let gx0 = b0x + xlo - m;
                let gy0 = b0y + ylo - m;
                let gz0 = b0z + zlo - m;
                // The neighbor this region maps into (clamped, may repeat).
                let nbid = neighbor[(((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)) as usize];
                if nbid == bid && !(dx == 0 && dy == 0 && dz == 0) {
                    // (unreachable: neighbors differ from the center block)
                    continue;
                }
                if gx0 < 0 || gy0 < 0 || gz0 < 0 {
                    continue;
                }
                if gx0 + xlen as i32 > dims.sx as i32
                    || gy0 + (yhi - ylo) > dims.sy as i32
                    || gz0 + (zhi - zlo) > dims.sz as i32
                {
                    continue;
                }
                let data = block_data_ptr(grid, nbid);
                if data.is_null() {
                    continue;
                }

                // Local in-neighbor coords of the region origin.
                let lx0 = gx0 & bsx_mask;
                let ly0 = gy0 & bsx_mask;
                let lz0 = gz0 & bsx_mask;

                for sz in zlo..zhi {
                    let vz = (lz0 + (sz - zlo)) << (2 * bsx_log2);
                    for sy in ylo..yhi {
                        let vy = (ly0 + (sy - ylo)) << bsx_log2;
                        let s_row = (sz * MSX as i32 + sy) * MSX as i32 + xlo;
                        let d_row = (lx0 | vy | vz) as usize;
                        let s_ptr = unsafe { staging.as_ptr().add(s_row as usize) };
                        let d_ptr = unsafe { data.add(d_row) };
                        if dx == 0 {
                            unsafe { D::commit_chunk::<16>(s_ptr, d_ptr) };
                        } else {
                            unsafe { D::commit_chunk::<4>(s_ptr, d_ptr) };
                        }
                    }
                }
            }
        }
    }
}

/// Staging range along one axis for region offset `d` (`-1` low margin,
/// `0` interior, `+1` high margin).
#[inline(always)]
fn region_range(d: i32, margin: i32, bsx: i32) -> (i32, i32) {
    match d {
        -1 => (0, margin),
        0 => (margin, margin + bsx),
        _ => (margin + bsx, 2 * margin + bsx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use crate::channel::Density;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;
    const MSX: usize = 32;

    fn setup(sx: usize, sy: usize, sz: usize) -> SparseGrid<Density, BSX, N> {
        let pool = Arc::new(BlockPool::<Density, BSX, N>::new(64, 64));
        SparseGrid::new("splat".into(), sx, sy, sz, Density(0), Density(u16::MAX), pool)
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

    fn fill_active(grid: &mut SparseGrid<Density, BSX, N>, active: &[usize]) {
        for &bid in active {
            grid.ensure_block(bid);
            grid.get_block_data_mut(bid)
                .unwrap()
                .fill(Density(u16::MAX));
        }
    }

    fn get(grid: &SparseGrid<Density, BSX, N>, x: usize, y: usize, z: usize) -> Density {
        grid.get_voxel(x, y, z)
    }

    fn do_splat(grid: &SparseGrid<Density, BSX, N>, pts: &[[f32; 3]], bids: Vec<usize>) {
        let bucketed =
            crate::particles::sort::bucket_by_block(pts.to_vec(), bids, grid.n_blocks);
        splat::<Density, BSX, N, MSX>(grid, &bucketed, &cfg());
    }

    // Happy path: single particle kernel shape (cell-centered offsets).
    #[test]
    fn splat_01_single_particle_shape() {
        let mut grid = setup(32, 32, 32);
        let pts = vec![[8.0f32, 8.0, 8.0]];
        let active = crate::particles::active::active_blocks(&pts, &GridDims::new(32, 32, 32, BSX), &cfg());
        fill_active(&mut grid, &active);
        do_splat(&grid, &pts, vec![0]);

        assert_eq!(get(&grid, 8, 8, 8), Density(3072)); // distSq 0.75
        let want = ((6.75f32 / 16.0) * 65535.0).round() as u16;
        assert_eq!(get(&grid, 5, 8, 8), Density(want));
        assert_eq!(get(&grid, 1, 8, 8), Density(u16::MAX));
    }

    // The case a 1-voxel splat halo gets wrong: a particle near a block
    // boundary writes voxels in the neighbor block.
    #[test]
    fn splat_02_boundary_crossing_particle() {
        let mut grid = setup(32, 32, 32);
        let pts = vec![[15.5f32, 8.0, 8.0]];
        let active = crate::particles::active::active_blocks(&pts, &GridDims::new(32, 32, 32, BSX), &cfg());
        assert!(active.contains(&1));
        fill_active(&mut grid, &active);
        do_splat(&grid, &pts, vec![0]);

        let want = ((1.5f32 / 16.0) * 65535.0).round() as u16;
        assert_ne!(get(&grid, 16, 8, 8), Density(u16::MAX));
        assert_eq!(get(&grid, 16, 8, 8), Density(want));
    }

    // Two overlapping particles: min is order-independent.
    #[test]
    fn splat_03_overlap_min() {
        let mut grid = setup(32, 32, 32);
        let pts = vec![[10.0f32, 8.0, 8.0], [12.0, 8.0, 8.0]];
        let active = crate::particles::active::active_blocks(&pts, &GridDims::new(32, 32, 32, BSX), &cfg());
        fill_active(&mut grid, &active);
        do_splat(&grid, &pts, vec![0, 0]);
        let forward = get(&grid, 10, 8, 8);

        let mut grid2 = setup(32, 32, 32);
        fill_active(&mut grid2, &active);
        do_splat(&grid2, &[pts[1], pts[0]], vec![0, 0]);
        assert_eq!(get(&grid2, 10, 8, 8), forward);
        for z in 5..12 {
            for y in 5..12 {
                for x in 5..16 {
                    assert_eq!(get(&grid2, x, y, z), get(&grid, x, y, z));
                }
            }
        }
    }

    // Domain-corner particle: the splat range clips to the domain.
    #[test]
    fn splat_04_domain_corner_clip() {
        let mut grid = setup(32, 32, 32);
        let pts = vec![[0.1f32, 0.1, 0.1]];
        let active = crate::particles::active::active_blocks(&pts, &GridDims::new(32, 32, 32, BSX), &cfg());
        fill_active(&mut grid, &active);
        do_splat(&grid, &pts, vec![0]);
        assert_ne!(get(&grid, 0, 0, 0), Density(u16::MAX));
    }

    // Radius cull: voxels beyond distSqMax are not written.
    #[test]
    fn splat_05_radius_cull() {
        let mut grid = setup(32, 32, 32);
        let pts = vec![[16.0f32, 16.0, 16.0]];
        let active = crate::particles::active::active_blocks(&pts, &GridDims::new(32, 32, 32, BSX), &cfg());
        fill_active(&mut grid, &active);
        do_splat(&grid, &pts, vec![grid.get_block_id(16, 16, 16)]);
        assert_eq!(get(&grid, 21, 16, 16), Density(u16::MAX));
        assert_eq!(get(&grid, 20, 16, 16), Density(u16::MAX));
    }

    // Quantize: closest voxel stores the smallest value, far voxels full.
    #[test]
    fn splat_06_quantize_endpoints() {
        let mut grid = setup(32, 32, 32);
        let pts = vec![[8.0f32, 8.0, 8.0]];
        let active = crate::particles::active::active_blocks(&pts, &GridDims::new(32, 32, 32, BSX), &cfg());
        fill_active(&mut grid, &active);
        do_splat(&grid, &pts, vec![0]);
        assert_eq!(get(&grid, 8, 8, 8), Density(3072));
        let want = ((12.75f32 / 16.0) * 65535.0).round() as u16;
        assert_eq!(get(&grid, 4, 8, 8), Density(want));
        assert_eq!(get(&grid, 0, 8, 8), Density(u16::MAX));
    }
}

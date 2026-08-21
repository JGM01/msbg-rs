//! Surface reconstruction from a particle point cloud.
//!
//! The pipeline mirrors the C++ demo's `msbg_test_sparse` (bunny-of-bunnies):
//! load a `.ply`, instantiate the base mesh at `n_instances` origins, bucket the
//! placed particles by their block, splat them into a `SparseGrid` density
//! channel with an 8-color race-free scatter, and convert the per-voxel
//! squared-distance field into a `[0, 1]` density ready for the step-8
//! [`Sweeper`](crate::solver::Sweeper).
//!
//! ```
//! use msbg_rs::blockpool::BlockPool;
//! use msbg_rs::particles::{reconstruct_surface, DEFAULT_MSX, SurfaceConfig};
//! use msbg_rs::channel::Density;
//! use std::sync::Arc;
//!
//! let cfg = SurfaceConfig::demo_case1(64);
//! let ply = br#"ply
//! format ascii 1.0
//! element vertex 4
//! property float x
//! property float y
//! property float z
//! end_header
//! 0 0 0
//! 1 0 0
//! 0 1 0
//! 0 0 1
//! "#;
//! let pool = Arc::new(BlockPool::<Density, 16, 4096>::new(8, 64));
//! let (grid, active) =
//!     reconstruct_surface::<Density, 16, 4096, DEFAULT_MSX>(&cfg, ply, pool).unwrap();
//! assert!(!active.is_empty());
//! ```

use std::io;
use std::sync::Arc;

use crate::blockpool::BlockPool;
use crate::channel::Quant;
use crate::math::simd::LANES;
use crate::math::stencil::StoreBack;
use crate::solver::{Fence, PdeParams, Stencil, Sweeper};
use crate::sparse_grid::SparseGrid;

pub mod active;
pub mod finalize;
pub mod sort;
pub mod splat;

/// Demo constants from `msbg_test_sparse` (rParticle=2, nbDist=2).
pub const DEMO_R_PARTICLE: f32 = 2.0;
pub const DEMO_NB_DIST: f32 = 2.0;

/// Staging extent for the splat: `BSX + 2·ceil(rScan)` (24 for the demo's
/// rScan = 4), a multiple of 8 so SIMD8 staging chunks stay inside one buffer
/// row. Raise it for larger radii.
pub const DEFAULT_MSX: usize = 24;

/// Voxel + block grid dimensions (the same fields `SparseGrid` derives).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridDims {
    pub sx: usize,
    pub sy: usize,
    pub sz: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nxy: usize,
    pub bsx: usize,
}

impl GridDims {
    pub fn new(sx: usize, sy: usize, sz: usize, bsx: usize) -> Self {
        debug_assert!(bsx.is_power_of_two());
        let log2 = bsx.trailing_zeros();
        let nx = (sx + bsx - 1) >> log2;
        let ny = (sy + bsx - 1) >> log2;
        let nz = (sz + bsx - 1) >> log2;
        let nxy = nx * ny;
        Self {
            sx,
            sy,
            sz,
            nx,
            ny,
            nz,
            nxy,
            bsx,
        }
    }

    #[inline(always)]
    pub fn n_blocks(&self) -> usize {
        self.nxy * self.nz
    }

    #[inline(always)]
    pub fn sxyz_max(&self) -> f32 {
        self.sx.max(self.sy).max(self.sz) as f32
    }

    #[inline(always)]
    pub fn sxyz_min(&self) -> f32 {
        self.sx.min(self.sy).min(self.sz) as f32
    }

    #[inline(always)]
    pub fn coords(&self, bid: usize) -> (usize, usize, usize) {
        (bid % self.nx, (bid / self.nx) % self.ny, bid / self.nxy)
    }
}

/// Configuration of one surface-reconstruction run.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceConfig {
    pub sx: usize,
    pub sy: usize,
    pub sz: usize,
    /// Number of mesh instances placed (the demo sets `nBasePoints`).
    pub n_instances: usize,
    /// Instance scale as a fraction of `min(sx, sy, sz)` (demo: 0.01 / 0.005).
    pub instance_scale_factor: f32,
    /// Particle radius in voxels (demo: 2).
    pub r_particle: f32,
    /// Falloff width past the radius (demo: 2).
    pub nb_dist: f32,
    /// Mean-curvature iterations (demo: 5).
    pub n_smooth_iters: usize,
    /// Mean-curvature time step (demo: 0.05 / 0.1).
    pub smooth_dt: f32,
}

impl SurfaceConfig {
    /// The demo's low-res bunny-of-bunnies placement.
    pub fn demo_case1(res: usize) -> Self {
        Self {
            sx: res,
            sy: res,
            sz: res,
            n_instances: 0, // filled in by `reconstruct_surface` (nBasePoints)
            instance_scale_factor: 0.01,
            r_particle: DEMO_R_PARTICLE,
            nb_dist: DEMO_NB_DIST,
            n_smooth_iters: 5,
            smooth_dt: 0.05,
        }
    }

    /// The demo's high-res bunny-of-bunnies placement.
    pub fn demo_case2(res: usize) -> Self {
        Self {
            n_instances: 0,
            instance_scale_factor: 0.005,
            smooth_dt: 0.1,
            ..Self::demo_case1(res)
        }
    }

    #[inline(always)]
    pub fn r_scan(&self) -> f32 {
        self.r_particle + self.nb_dist
    }
}

/// Placed particles plus the exact (footprint) active-block set.
pub struct Placed {
    pub positions: Vec<[f32; 3]>,
    pub bids: Vec<usize>,
    pub active: Vec<usize>,
}

/// Domain bounds used by the placement `isInDomainRange` check, replicating the
/// C++ `PARTICLE_QUANT_POS` path (`align24bitQuant` of `0.005` / `size-0.005`).
#[derive(Clone, Copy, Debug)]
pub struct DomainBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

fn align24(x: f64) -> f32 {
    let scale = 16_777_215.0 / 32_768.0;
    ((scale * x + 0.5).floor() / scale) as f32
}

impl DomainBounds {
    pub fn new(dims: &GridDims) -> Self {
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (k, size) in [dims.sx, dims.sy, dims.sz].into_iter().enumerate() {
            min[k] = align24(0.005);
            max[k] = align24(size as f64 - 0.005);
        }
        Self { min, max }
    }

    #[inline(always)]
    pub fn contains(&self, pos: &[f32; 3]) -> bool {
        (0..3).all(|k| pos[k] >= self.min[k] && pos[k] <= self.max[k])
    }
}

/// Footprint block range along one axis for a particle at `p`: the C++
/// `trunc(bpos ± rScan/bsx)` form (`bpos = p * scale2DestBlockGrid`), clipped
/// to `[0, n)`. Shared by placement (`sort::place`) and active-block
/// determination (`active::active_blocks`).
#[inline(always)]
pub(crate) fn footprint_axis(p: f32, r_scan_bsx: f32, bsx: usize, n: usize) -> (i32, i32) {
    let b = p / bsx as f32;
    let lo = ((b - r_scan_bsx).trunc() as i32).max(0);
    let hi = ((b + r_scan_bsx).trunc() as i32).min(n as i32 - 1);
    (lo, hi)
}

/// Run the full pipeline: load, place, bucket, splat, finalize.
///
/// Returns the density grid (active blocks filled with `D::full()`, then
/// splatted and finalized) and the active block list. Feed both to a
/// [`Sweeper`] for the mean-curvature smoothing phase. `MSX` is the splat
/// staging extent (`DEFAULT_MSX` for the demo radii).
pub fn reconstruct_surface<D, const BSX: usize, const N: usize, const MSX: usize>(
    cfg: &SurfaceConfig,
    ply: &[u8],
    pool: Arc<BlockPool<D, BSX, N>>,
) -> io::Result<(SparseGrid<D, BSX, N>, Vec<usize>)>
where
    D: Quant + Copy + Default + Send + Sync,
{
    let loaded = crate::io::ply::load_vertices(ply)?;
    if loaded.positions.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "PLY has no vertices"));
    }

    let dims = GridDims::new(cfg.sx, cfg.sy, cfg.sz, BSX);
    let span_max = {
        let mut m = 0.0f32;
        for k in 0..3 {
            m = m.max(loaded.bbox_max[k] - loaded.bbox_min[k]);
        }
        m
    };
    if span_max <= 0.0 || !span_max.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "degenerate point cloud (zero span)",
        ));
    }

    let placed = sort::place(
        &loaded.positions,
        &loaded.bbox_min,
        span_max,
        &dims,
        &DomainBounds::new(&dims),
        cfg,
    );
    let bucketed = sort::bucket_by_block(placed.positions, placed.bids);
    let active = placed.active;

    let mut grid = SparseGrid::new(
        "surface".into(),
        cfg.sx,
        cfg.sy,
        cfg.sz,
        D::empty(),
        D::full(),
        pool,
    );
    grid.ensure_blocks_parallel(&active);
    grid.fill_blocks_parallel(&active, D::full());
    splat::splat::<D, BSX, N, MSX>(&grid, &bucketed, cfg);
    finalize::finalize::<D, BSX, N>(&grid, &active, cfg);

    Ok((grid, active))
}

/// `reconstruct_surface` plus the demo's mean-curvature smoothing phase (step 8
/// [`Sweeper`]), producing the final surface density. `HSX` must be `BSX + 2`
/// for the mean-curvature halo; `MSX` is the splat staging extent.
pub fn reconstruct_and_smooth<D, const BSX: usize, const N: usize, const HSX: usize, const MSX: usize>(
    cfg: &SurfaceConfig,
    ply: &[u8],
    pool: Arc<BlockPool<D, BSX, N>>,
    num_threads: usize,
    fence: Fence,
) -> io::Result<(SparseGrid<D, BSX, N>, Vec<usize>)>
where
    D: Quant + StoreBack<LANES> + Copy + Default + Send + Sync,
{
    debug_assert_eq!(HSX, BSX + 2);
    let (grid, active) = reconstruct_surface::<D, BSX, N, MSX>(cfg, ply, pool)?;
    let sweeper = Sweeper::<D, BSX, N, HSX>::new(&grid, num_threads, fence);
    sweeper.sweep(
        &active,
        Stencil::MeanCurvature,
        &PdeParams {
            dt: cfg.smooth_dt,
            iterations: cfg.n_smooth_iters,
            do_constr_zero_one: true,
        },
    );
    Ok((grid, active))
}

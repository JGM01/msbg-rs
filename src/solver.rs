//! In-place 8-color PDE smoothing.
//!
//! The [`Sweeper`] is a use-case-independent primitive: it iterates an active
//! block list in 8 colors (blocks of one color never share a halo face, so they
//! can be updated in parallel without races), and for each block it gathers a
//! thread-local halo, runs a per-block stencil, and writes the result back into
//! the *same* grid block through the element type's [`StoreBack`]. Mean-
//! curvature, Laplacian, and bi-Laplacian are just stencils over this primitive
//! — the core stays solver-agnostic.
//!
//! ```
//! use msbg_rs::solver::{Fence, PdeParams, Stencil, Sweeper};
//! use msbg_rs::sparse_grid::SparseGrid;
//! use msbg_rs::blockpool::BlockPool;
//! use std::sync::Arc;
//!
//! let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(8, 64));
//! let grid = SparseGrid::new("phi".into(), 32, 32, 32, 0.0, 1.0, pool);
//! let active: Vec<u32> = (0..grid.n_blocks as u32).collect();
//!
//! let sweeper = Sweeper::<f32, 16, 4096, 18>::new(&grid, 1, Fence::Sfence);
//! let params = PdeParams { dt: 0.05, iterations: 4, do_constr_zero_one: true };
//! sweeper.sweep(&active, Stencil::MeanCurvature, &params);
//! ```

use std::cell::UnsafeCell;

use rayon::prelude::*;

use crate::math::boundary::BoundaryCondition;
use crate::math::gather::Dequant;
use crate::math::simd::LANES;
use crate::math::stencil::{store_fence, store_mfence, SimdRng, StoreBack};
use crate::multires::halo::HaloBlock;
use crate::multires::sort::morton3;
use crate::sparse_grid::SparseGrid;

use crate::math::bilaplacian::kernel_bilaplacian;
use crate::math::laplacian::kernel_laplacian;
use crate::math::meancurv::kernel_meancurv;

/// The per-block stencil to apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stencil {
    /// 7-point Laplacian smoothing.
    Laplacian,
    /// 19-tap mean-curvature flow.
    MeanCurvature,
    /// 25-tap bi-Laplacian smoothing.
    BiLaplacian,
}

/// Parameters for one solve: the timestep, the number of full 8-color sweeps,
/// and whether to clamp the result to `[0, 1]` after each update.
#[derive(Clone, Copy, Debug)]
pub struct PdeParams {
    pub dt: f32,
    pub iterations: usize,
    /// Clamp the result to `[0, 1]` after each update.
    pub do_constr_zero_one: bool,
}

/// Store fence policy between color passes. Only the `f32` path uses
/// non-temporal stores, so the fence is a no-op for `Density`/`Density8`.
///
/// `None` is **not** memory-safe for the `f32` path (the non-temporal store
/// bypasses the cache and a later regular load may read stale data). It exists
/// only to benchmark the cost of the store ordering against `Sfence`/`Mfence`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fence {
    /// Store-only fence (`_mm_sfence`).
    Sfence,
    /// Full fence (`_mm_mfence`).
    Mfence,
    /// No fence (benchmarking only; see the type docs).
    None,
}

impl Fence {
    #[inline(always)]
    fn apply(self) {
        match self {
            Fence::Sfence => store_fence(),
            Fence::Mfence => store_mfence(),
            Fence::None => {}
        }
    }
}

/// Thread-local sweep context: the halo staging buffer plus the PRNG for
/// stochastic-rounding stores.
struct Slot<const BSX: usize, const HSX: usize> {
    halo: HaloBlock<BSX, HSX>,
    rng: SimdRng<LANES>,
}

/// A pre-sized pool of thread-local slots, indexed by `rayon` worker index.
struct SlotPool<const BSX: usize, const HSX: usize> {
    slots: Vec<UnsafeCell<Slot<BSX, HSX>>>,
}

// Each slot is exclusively accessed by a single physical Rayon thread.
unsafe impl<const BSX: usize, const HSX: usize> Sync for SlotPool<BSX, HSX> {}

impl<const BSX: usize, const HSX: usize> SlotPool<BSX, HSX> {
    fn new(num_threads: usize) -> Self {
        // One extra slot for the calling thread: `current_thread_index()` is
        // `None` when a tiny bucket is processed inline on the thread that
        // drives the parallel iterator (rayon does not register it as a
        // worker). Workers use indices `0..num_threads`, so slot `num_threads`
        // is never claimed by a worker and is exclusive to the caller.
        let slots = (0..=num_threads)
            .map(|i| {
                UnsafeCell::new(Slot {
                    halo: HaloBlock::new(),
                    rng: SimdRng::seed((i as u32).wrapping_mul(0x9e37_79b9) ^ 0xdead_beef),
                })
            })
            .collect();
        Self { slots }
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut Slot<BSX, HSX> {
        let idx = rayon::current_thread_index().unwrap_or(self.slots.len() - 1);
        debug_assert!(
            idx < self.slots.len(),
            "SlotPool sized smaller than the active rayon pool"
        );
        unsafe { &mut *self.slots[idx].get() }
    }
}

/// In-place 8-color block sweeper over one [`SparseGrid`] channel.
///
/// `HSX` is the halo extent: `BSX + 2` for [`Stencil::Laplacian`] /
/// [`Stencil::MeanCurvature`] (1-voxel halo) and `BSX + 4` for
/// [`Stencil::BiLaplacian`] (2-voxel halo). `num_threads` must be at least the
/// active rayon pool's thread count.
pub struct Sweeper<'a, D, const BSX: usize, const N: usize, const HSX: usize>
where
    D: Copy + Default + Send + Sync,
{
    grid: &'a SparseGrid<D, BSX, N>,
    slots: SlotPool<BSX, HSX>,
    fence: Fence,
}

impl<'a, D, const BSX: usize, const N: usize, const HSX: usize> Sweeper<'a, D, BSX, N, HSX>
where
    D: Copy + Default + Send + Sync,
{
    pub fn new(grid: &'a SparseGrid<D, BSX, N>, num_threads: usize, fence: Fence) -> Self {
        Self {
            grid,
            slots: SlotPool::new(num_threads),
            fence,
        }
    }

    /// Run `params.iterations` full 8-color smoothing sweeps over `active`.
    ///
    /// `active` is the finest-level value-block list (the demo's `activeBlocks`).
    /// Blocks not in a writable (allocated) state are skipped.
    pub fn sweep(&self, active: &[u32], stencil: Stencil, params: &PdeParams)
    where
        D: Dequant<f32> + StoreBack<LANES>,
    {
        let buckets = build_color_buckets(self.grid, active);
        match stencil {
            Stencil::Laplacian => {
                debug_assert_eq!(HSX, BSX + 2, "Laplacian needs HSX = BSX + 2");
                self.run::<1, false, _>(&buckets, params, kernel_laplacian::<LANES, BSX, HSX, D>);
            }
            Stencil::MeanCurvature => {
                debug_assert_eq!(HSX, BSX + 2, "mean-curvature needs HSX = BSX + 2");
                self.run::<1, true, _>(&buckets, params, kernel_meancurv::<LANES, BSX, HSX, D>);
            }
            Stencil::BiLaplacian => {
                debug_assert_eq!(HSX, BSX + 4, "bi-Laplacian needs HSX = BSX + 4");
                self.run::<2, true, _>(&buckets, params, kernel_bilaplacian::<LANES, BSX, HSX, D>);
            }
        }
    }

    fn run<const HALO: usize, const FULL: bool, K>(
        &self,
        buckets: &[Vec<u32>; 8],
        params: &PdeParams,
        kernel: K,
    ) where
        K: Fn(&HaloBlock<BSX, HSX>, f32, bool, *mut D, &mut SimdRng<LANES>) + Sync,
        D: Dequant<f32> + StoreBack<LANES>,
    {
        for _ in 0..params.iterations {
            for color in 0..8 {
                let bucket = &buckets[color];
                bucket.par_iter().for_each(|&bid| {
                    let slot = unsafe { self.slots.get_mut() };
                    let out = self.block_data_ptr(bid);
                    if out.is_null() {
                        return;
                    }
                    slot.halo.fill::<{ HALO }, { FULL }, D, N>(
                        self.grid,
                        bid as usize,
                        BoundaryCondition::Neumann,
                    );
                    kernel(&slot.halo, params.dt, params.do_constr_zero_one, out, &mut slot.rng);
                    if D::USES_NT {
                        self.fence.apply();
                    }
                });
            }
        }
    }

    /// Raw writable data pointer for `bid`, or null if the block is a dummy or
    /// unallocated (not a value block).
    #[inline(always)]
    fn block_data_ptr(&self, bid: u32) -> *mut D {
        let bid = bid as usize;
        match self.grid.blockmap[bid] {
            Some(p) if p != self.grid.empty_block && p != self.grid.full_block => {
                unsafe { (*p.as_ptr()).data.as_mut_ptr() }
            }
            _ => std::ptr::null_mut(),
        }
    }
}

/// Convenience wrapper: build a [`Sweeper`] and run one solve.
pub fn apply_pde<D, const BSX: usize, const N: usize, const HSX: usize>(
    grid: &SparseGrid<D, BSX, N>,
    active: &[u32],
    stencil: Stencil,
    params: &PdeParams,
    num_threads: usize,
    fence: Fence,
) where
    D: Copy + Default + Send + Sync + Dequant<f32> + StoreBack<LANES>,
{
    let sweeper: Sweeper<'_, D, BSX, N, HSX> = Sweeper::new(grid, num_threads, fence);
    sweeper.sweep(active, stencil, params);
}

/// Split the active list into 8 color buckets (color = `bx&1 | (by&1)<<1 |
/// (bz&1)<<2`) and Morton-sort each bucket for cache locality. Blocks of one
/// color are separated by at least one block on some axis, so a 1- or 2-voxel
/// halo of two same-color blocks never overlaps — hence the parallel race-free
/// in-place update.
fn build_color_buckets<D, const BSX: usize, const N: usize>(
    grid: &SparseGrid<D, BSX, N>,
    active: &[u32],
) -> [Vec<u32>; 8]
where
    D: Copy + Default + Send + Sync,
{
    let mut buckets: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
    for &bid in active {
        let (bx, by, bz) = grid.get_block_coords_by_id(bid as usize);
        let color = (bx & 1) | ((by & 1) << 1) | ((bz & 1) << 2);
        buckets[color].push(bid);
    }
    for bucket in &mut buckets {
        bucket.sort_unstable_by_key(|&bid| {
            let (bx, by, bz) = grid.get_block_coords_by_id(bid as usize);
            morton3(bx as u32, by as u32, bz as u32)
        });
    }
    buckets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;
    const HSX: usize = 18;

    fn grid(sx: usize, sy: usize, sz: usize) -> SparseGrid<f32, BSX, N> {
        let pool = Arc::new(BlockPool::<f32, BSX, N>::new(64, 64));
        SparseGrid::new("solver_test".into(), sx, sy, sz, 0.0, 1.0, pool)
    }

    fn all_active<const BSX: usize, const N: usize, D: Copy + Default + Send + Sync>(
        g: &SparseGrid<D, BSX, N>,
    ) -> Vec<u32> {
        (0..g.n_blocks as u32).collect()
    }

    // Serial color-ordered Gauss-Seidel reference: processes blocks color 0..7
    // in order, gathering the halo from the *current* grid and writing back.
    fn serial_reference(
        g: &mut SparseGrid<f32, BSX, N>,
        active: &[u32],
        stencil: Stencil,
        params: &PdeParams,
    ) {
        let mut buckets: [Vec<u32>; 8] = std::array::from_fn(|_| Vec::new());
        for &bid in active {
            let (bx, by, bz) = g.get_block_coords_by_id(bid as usize);
            let color = (bx & 1) | ((by & 1) << 1) | ((bz & 1) << 2);
            buckets[color].push(bid);
        }
        let mut halo = HaloBlock::<BSX, HSX>::new();
        let mut rng = SimdRng::<LANES>::seed(0);
        for _ in 0..params.iterations {
            for color in 0..8 {
                for &bid in &buckets[color] {
                    let out = match g.blockmap[bid as usize] {
                        Some(p) if p != g.empty_block && p != g.full_block => unsafe {
                            (*p.as_ptr()).data.as_mut_ptr()
                        },
                        _ => continue,
                    };
                    match stencil {
                        Stencil::Laplacian => {
                            halo.fill::<1, false, f32, N>(g, bid as usize, BoundaryCondition::Neumann);
                            kernel_laplacian::<LANES, BSX, HSX, f32>(
                                &halo, params.dt, params.do_constr_zero_one, out, &mut rng,
                            );
                        }
                        Stencil::MeanCurvature => {
                            halo.fill::<1, true, f32, N>(g, bid as usize, BoundaryCondition::Neumann);
                            kernel_meancurv::<LANES, BSX, HSX, f32>(
                                &halo, params.dt, params.do_constr_zero_one, out, &mut rng,
                            );
                        }
                        Stencil::BiLaplacian => unreachable!("needs HSX=20"),
                    }
                }
            }
        }
    }

    #[test]
    fn test_solve_01_affine_field_identity() {
        let mut g = grid(32, 32, 32);
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    g.set_voxel(x, y, z, 2.0 * x as f32 + 3.0 * y as f32 - z as f32);
                }
            }
        }
        let active = all_active(&g);
        let sweeper = Sweeper::<f32, BSX, N, HSX>::new(&g, rayon::current_num_threads(), Fence::Mfence);
        let params = PdeParams { dt: 0.05, iterations: 1, do_constr_zero_one: false };
        sweeper.sweep(&active, Stencil::Laplacian, &params);

        // An affine field has zero Laplacian, so interior cells are exact fixed
        // points after one sweep. Domain-edge cells legitimately change: the
        // Neumann halo mirrors the edge voxel (not the affine extrapolation),
        // so their stencil is not zero — and that perturbation propagates one
        // cell inward per iteration, hence `iterations: 1` + interior check.
        for z in 1..31 {
            for y in 1..31 {
                for x in 1..31 {
                    let want = 2.0 * x as f32 + 3.0 * y as f32 - z as f32;
                    assert!((g.get_voxel(x, y, z) - want).abs() < 1e-4, "({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn test_solve_02_color_matches_serial_reference() {
        // Parallel 8-color must equal a serial color-ordered Gauss-Seidel
        // reference exactly (the coloring is a valid GS coloring).
        let mut g = grid(32, 32, 32);
        let mut g2 = grid(32, 32, 32);
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let v = ((x * 17 + y * 31 + z * 7) % 100) as f32 * 0.01;
                    g.set_voxel(x, y, z, v);
                    g2.set_voxel(x, y, z, v);
                }
            }
        }
        let active = all_active(&g);
        let params = PdeParams { dt: 0.05, iterations: 2, do_constr_zero_one: false };

        let sweeper = Sweeper::<f32, BSX, N, HSX>::new(&g, rayon::current_num_threads(), Fence::Mfence);
        sweeper.sweep(&active, Stencil::MeanCurvature, &params);
        serial_reference(&mut g2, &active, Stencil::MeanCurvature, &params);

        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let a = g.get_voxel(x, y, z);
                    let b = g2.get_voxel(x, y, z);
                    assert!((a - b).abs() < 1e-5, "({x},{y},{z}): {a} != {b}");
                }
            }
        }
    }

    #[test]
    fn test_solve_03_dummy_and_gap_active() {
        // Active list containing a full dummy and an unallocated (but valid)
        // block must be skipped without panic; real blocks still update.
        let mut g = grid(32, 32, 32);
        g.set_voxel(0, 0, 0, 0.5); // block 0
        g.set_voxel(16, 16, 16, 0.7); // block 7
        g.set_full_block(g.get_block_id(0, 16, 0)); // block 2 -> dummy full
        let active: Vec<u32> = vec![0, 7, g.get_block_id(0, 16, 0) as u32, 5];
        let params = PdeParams { dt: 0.05, iterations: 1, do_constr_zero_one: true };
        let sweeper = Sweeper::<f32, BSX, N, HSX>::new(&g, rayon::current_num_threads(), Fence::Mfence);
        sweeper.sweep(&active, Stencil::Laplacian, &params);
        // Just assert no panic and values remain in [0,1].
        assert!(g.get_voxel(0, 0, 0).is_finite());
        assert!((0.0..=1.0).contains(&g.get_voxel(0, 0, 0)));
    }

    #[test]
    fn test_solve_04_empty_active_is_noop() {
        let g = grid(32, 32, 32);
        let params = PdeParams { dt: 0.05, iterations: 2, do_constr_zero_one: true };
        let sweeper = Sweeper::<f32, BSX, N, HSX>::new(&g, rayon::current_num_threads(), Fence::Mfence);
        sweeper.sweep(&[], Stencil::Laplacian, &params);
    }

    #[test]
    fn test_solve_05_clamp_zero_one() {
        // Large dt on a large field drives values out of range; clamp01 keeps
        // them within [0,1].
        let mut g = grid(16, 16, 16);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    g.set_voxel(x, y, z, 100.0 + x as f32);
                }
            }
        }
        let active = all_active(&g);
        let params = PdeParams { dt: 1000.0, iterations: 1, do_constr_zero_one: true };
        let sweeper = Sweeper::<f32, BSX, N, HSX>::new(&g, rayon::current_num_threads(), Fence::Mfence);
        sweeper.sweep(&active, Stencil::MeanCurvature, &params);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let v = g.get_voxel(x, y, z);
                    assert!((0.0..=1.0).contains(&v), "clamp violated: {v}");
                }
            }
        }
    }

    #[test]
    fn test_solve_06_odd_iterations() {
        let mut g = grid(16, 16, 16);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    g.set_voxel(x, y, z, (x * y + z) as f32 / 255.0);
                }
            }
        }
        let active = all_active(&g);
        let params = PdeParams { dt: 0.05, iterations: 3, do_constr_zero_one: true };
        let sweeper = Sweeper::<f32, BSX, N, HSX>::new(&g, rayon::current_num_threads(), Fence::Mfence);
        sweeper.sweep(&active, Stencil::Laplacian, &params);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    assert!(g.get_voxel(x, y, z).is_finite());
                }
            }
        }
    }
}

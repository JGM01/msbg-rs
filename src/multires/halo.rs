use crate::math::boundary::{resolve_axis, BoundaryCondition};
use crate::math::gather::Dequant;
use crate::sparse_grid::SparseGrid;
use std::cell::UnsafeCell;

/// Thread-local staging buffer: a `BSX + 2*HALO` cube holding a block plus a
/// `HALO`-voxel halo on every face (and the edges/corners, when filled in full
/// mode). `HSX` must equal `BSX + 2*HALO`.
pub struct HaloBlock<const BSX: usize, const HSX: usize> {
    pub data: Box<[f32]>,
}

impl<const BSX: usize, const HSX: usize> HaloBlock<BSX, HSX> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            data: vec![0.0f32; HSX * HSX * HSX].into_boxed_slice(),
        }
    }

    /// Fill the halo from block `bid` of `grid`, dequantizing each element to
    /// `f32`.
    ///
    /// `HALO` is the halo width in voxels on each face (1 for the 7/19-tap
    /// stencils, 2 for the 25-tap bi-Laplacian). `FULL = true` fills the whole
    /// `HSX^3` cube (faces + edges + corners); `FULL = false` fills the center
    /// plus the six faces only (enough for the 7-point Laplacian) and leaves
    /// the edges/corners zero.
    #[inline]
    pub fn fill<const HALO: usize, const FULL: bool, D, const N: usize>(
        &mut self,
        grid: &SparseGrid<D, BSX, N>,
        bid: usize,
        bc: BoundaryCondition,
    ) where
        D: Dequant<f32>,
    {
        debug_assert_eq!(HSX, BSX + 2 * HALO, "HSX must equal BSX + 2*HALO");
        let bsx_log2 = BSX.trailing_zeros();
        let bsx_mask = BSX - 1;
        let bsx_i = BSX as i32;

        let nx = grid.nx as i32;
        let ny = grid.ny as i32;
        let nz = grid.nz as i32;
        let sx = grid.sx as i32;
        let sy = grid.sy as i32;
        let sz = grid.sz as i32;
        let empty_f32 = grid.empty_value().dequant();

        let bx0 = (bid % grid.nx) as i32;
        let by0 = ((bid / grid.nx) % grid.ny) as i32;
        let bz0 = (bid / grid.nxy) as i32;
        let x0 = bx0 * bsx_i;

        let empty_ptr = unsafe { (*grid.empty_block.as_ptr()).data.as_ptr() };
        let full_ptr = unsafe { (*grid.full_block.as_ptr()).data.as_ptr() };

        // Resolve the (2*HALO+1)^3 block neighborhood to raw data pointers once.
        // Fixed upper bound: HALO <= 2 -> 5^3 = 125 pointers.
        let n = 2 * HALO + 1;
        debug_assert!(n <= 5, "HALO must be <= 2");
        let halo_i = HALO as i32;
        let mut ptrs: [*const D; 125] = [empty_ptr; 125];
        for dz in 0..n {
            for dy in 0..n {
                for dx in 0..n {
                    let bx = bx0 + dx as i32 - halo_i;
                    let by = by0 + dy as i32 - halo_i;
                    let bz = bz0 + dz as i32 - halo_i;
                    let p = if bx >= 0 && bx < nx && by >= 0 && by < ny && bz >= 0 && bz < nz {
                        let b = (bx as usize) + (by as usize) * grid.nx + (bz as usize) * grid.nxy;
                        match grid.blockmap[b] {
                            Some(p) if p == grid.empty_block => empty_ptr,
                            Some(p) if p == grid.full_block => full_ptr,
                            Some(p) => unsafe { (*p.as_ptr()).data.as_ptr() },
                            None => empty_ptr,
                        }
                    } else {
                        empty_ptr
                    };
                    ptrs[dx + n * dy + n * n * dz] = p;
                }
            }
        }

        let dst = self.data.as_mut_ptr();
        let dy_stride = HSX;
        let dz_stride = HSX * HSX;

        // Read a voxel whose coords are already resolved into the domain.
        let read = |x: i32, y: i32, z: i32| -> f32 {
            let idx = ((x >> bsx_log2) - bx0 + halo_i) as usize
                + n * (((y >> bsx_log2) - by0 + halo_i) as usize)
                + n * n * (((z >> bsx_log2) - bz0 + halo_i) as usize);
            debug_assert!(idx < n * n * n);
            let vid = (x as usize & bsx_mask)
                | ((y as usize & bsx_mask) << bsx_log2)
                | ((z as usize & bsx_mask) << (2 * bsx_log2));
            unsafe { (*ptrs[idx].add(vid)).dequant() }
        };

        // The x-halo columns do not depend on the row; resolve them once.
        let mut rx_left = [None; HALO];
        let mut rx_right = [None; HALO];
        for j in 0..HALO {
            rx_left[j] = resolve_axis(x0 - halo_i + j as i32, sx - 1, bc);
            rx_right[j] = resolve_axis(x0 + bsx_i + j as i32, sx - 1, bc);
        }

        for z in 0..HSX {
            let gz = bz0 * bsx_i + z as i32 - halo_i;
            let rz = resolve_axis(gz, sz - 1, bc);
            for y in 0..HSX {
                let gy = by0 * bsx_i + y as i32 - halo_i;
                let ry = resolve_axis(gy, sy - 1, bc);
                let row = z * dz_stride + y * dy_stride;

                // Middle segment (halo x = HALO..=HALO+BSX-1): contiguous when
                // fully inside the domain.
                match (ry, rz) {
                    (Some(ryv), Some(rzv)) if x0 + bsx_i <= sx => {
                        let by = ryv >> bsx_log2;
                        let bz = rzv >> bsx_log2;
                        let idx = halo_i as usize
                            + n * ((by - by0 + halo_i) as usize)
                            + n * n * ((bz - bz0 + halo_i) as usize);
                        let p = ptrs[idx];
                        let vid = ((ryv as usize & bsx_mask) << bsx_log2)
                            | ((rzv as usize & bsx_mask) << (2 * bsx_log2));
                        unsafe {
                            D::copy_row(p.add(vid), dst.add(row + HALO), BSX);
                        }
                    }
                    (Some(ryv), Some(rzv)) => {
                        // Partial last block: resolve each x individually.
                        let by = ryv;
                        let bz = rzv;
                        unsafe {
                            for i in 0..BSX {
                                let gx = x0 + i as i32;
                                let v = match resolve_axis(gx, sx - 1, bc) {
                                    Some(rx) => read(rx, by, bz),
                                    None => empty_f32,
                                };
                                *dst.add(row + HALO + i) = v;
                            }
                        }
                    }
                    _ => unsafe {
                        for i in 0..BSX {
                            *dst.add(row + HALO + i) = empty_f32;
                        }
                    },
                }

                // Left/right halo columns. For faces-only they are only read
                // from center rows, so they are skipped elsewhere.
                if FULL || (y >= HALO && y < HALO + BSX && z >= HALO && z < HALO + BSX) {
                    unsafe {
                        for j in 0..HALO {
                            *dst.add(row + j) = match (rx_left[j], ry, rz) {
                                (Some(lx), Some(ly), Some(lz)) => read(lx, ly, lz),
                                _ => empty_f32,
                            };
                            *dst.add(row + HALO + BSX + j) = match (rx_right[j], ry, rz) {
                                (Some(rx), Some(ly), Some(lz)) => read(rx, ly, lz),
                                _ => empty_f32,
                            };
                        }
                    }
                }
            }
        }
    }
}

impl<const BSX: usize, const HSX: usize> Default for HaloBlock<BSX, HSX> {
    fn default() -> Self {
        Self::new()
    }
}

/// A pre-sized pool of thread-local halo blocks.
/// Designed to be indexed directly via `rayon::current_thread_index()`.
pub struct HaloBlockPool<const BSX: usize, const HSX: usize> {
    slots: Vec<UnsafeCell<HaloBlock<BSX, HSX>>>,
}

// Each slot is exclusively accessed by a single physical Rayon thread.
// Assert no reentrant parallel maps occur within the sweep.
unsafe impl<const BSX: usize, const HSX: usize> Sync for HaloBlockPool<BSX, HSX> {}

impl<const BSX: usize, const HSX: usize> HaloBlockPool<BSX, HSX> {
    /// Creates a pool sized to thread pool size.
    pub fn new(num_threads: usize) -> Self {
        let slots = (0..num_threads)
            .map(|_| UnsafeCell::new(HaloBlock::new()))
            .collect();
        Self { slots }
    }

    /// Retrieves the mutable halo buffer assigned to the current worker thread.
    ///
    /// # Safety
    ///
    /// Must only be called from inside a Rayon parallel context whose worker
    /// count matches (or is at most) the pool size: each slot is owned by
    /// exactly one worker. `current_thread_index` is scoped to whichever rayon
    /// pool is currently executing, so the pool must be sized for that pool.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self) -> &mut HaloBlock<BSX, HSX> {
        let idx = rayon::current_thread_index().expect("must run inside the Rayon thread pool");
        debug_assert!(
            idx < self.slots.len(),
            "HaloBlockPool sized smaller than the active rayon pool"
        );
        unsafe { &mut *self.slots[idx].get() }
    }
}

#[cfg(test)]
mod halo_tests {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use super::*;
    use crate::blockpool::BlockPool;
    use crate::channel::Density;
    use crate::math::boundary::BoundaryCondition;
    use crate::sparse_grid::SparseGrid;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;
    const HSX: usize = 18;
    const NEUMANN: BoundaryCondition = BoundaryCondition::Neumann;

    fn setup_grid(sx: usize, sy: usize, sz: usize) -> SparseGrid<f32, BSX, N> {
        let pool = Arc::new(BlockPool::new(64, 64));
        SparseGrid::new("halo_test".to_string(), sx, sy, sz, 0.0, 1.0, pool)
    }

    fn halo_at(halo: &HaloBlock<BSX, HSX>, x: usize, y: usize, z: usize) -> f32 {
        halo.data[z * HSX * HSX + y * HSX + x]
    }

    #[test]
    fn test_h_01_construction() {
        let halo = HaloBlock::<BSX, HSX>::new();
        assert_eq!(halo.data.len(), HSX * HSX * HSX);
        assert!(halo.data.iter().all(|&v| v == 0.0));
    }

    // Happy path: interior of a single block is copied verbatim.
    #[test]
    fn test_h_03_center_copy() {
        let mut grid = setup_grid(16, 16, 16);
        for z in 0..BSX {
            for y in 0..BSX {
                for x in 0..BSX {
                    grid.set_voxel(x, y, z, (x + y * 16 + z * 256) as f32);
                }
            }
        }
        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, f32, N>(&grid, 0, NEUMANN);
        for z in 0..BSX {
            for y in 0..BSX {
                for x in 0..BSX {
                    assert_eq!(halo_at(&halo, x + 1, y + 1, z + 1), (x + y * 16 + z * 256) as f32);
                }
            }
        }
    }

    // Happy path: a 3^3 grid — every face/edge/corner comes from the correct
    // neighbor block (each block stores its own bid).
    #[test]
    fn test_h_04_all_neighbors_full_fill() {
        let mut grid = setup_grid(48, 48, 48);
        for bid in 0..grid.n_blocks {
            let bx = bid % grid.nx;
            let by = (bid / grid.nx) % grid.ny;
            let bz = bid / grid.nxy;
            grid.set_voxel(bx * BSX, by * BSX, bz * BSX, bid as f32);
            if let Some(ptr) = grid.blockmap[bid]
                && ptr != grid.empty_block
                && ptr != grid.full_block
            {
                unsafe { (*ptr.as_ptr()).data.fill(bid as f32) };
            }
        }

        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, f32, N>(&grid, 13, NEUMANN);

        assert_eq!(halo_at(&halo, 1, 1, 1), 13.0); // center
        assert_eq!(halo_at(&halo, 0, 1, 1), 12.0); // left face
        assert_eq!(halo_at(&halo, 17, 1, 1), 14.0); // right face
        assert_eq!(halo_at(&halo, 1, 0, 1), 10.0); // bottom face
        assert_eq!(halo_at(&halo, 1, 17, 1), 16.0); // top face
        assert_eq!(halo_at(&halo, 1, 1, 0), 4.0); // back face
        assert_eq!(halo_at(&halo, 1, 1, 17), 22.0); // front face
        assert_eq!(halo_at(&halo, 0, 0, 0), 0.0); // corner (0,0,0)
        assert_eq!(halo_at(&halo, 17, 17, 17), 26.0); // corner (2,2,2)
        assert_eq!(halo_at(&halo, 0, 0, 1), 9.0); // edge (0,0,1) = bid 9
        assert_eq!(halo_at(&halo, 0, 17, 17), 24.0); // edge (0,2,2) = bid 24
    }

    // Boundary: Dirichlet reads the empty sentinel out-of-domain (not the edge).
    #[test]
    fn test_h_05_dirichlet_uses_empty() {
        let mut grid = setup_grid(16, 16, 16);
        grid.set_voxel(0, 0, 0, 42.0);
        grid.set_empty_value(7.0);

        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, f32, N>(&grid, 0, BoundaryCondition::Dirichlet);

        // Left face (global x = -1) must be the empty sentinel, not voxel 0.
        for z in 1..=BSX {
            for y in 1..=BSX {
                assert_eq!(halo_at(&halo, 0, y, z), 7.0);
            }
        }
    }

    // Boundary: Neumann mirrors the edge voxel out-of-domain (dmax=1, so
    // mirror == clamp == edge replicate).
    #[test]
    fn test_h_06_neumann_mirrors_edge() {
        let mut grid = setup_grid(16, 16, 16);
        fill_axis(&mut grid, |x| x as f32);

        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, f32, N>(&grid, 0, NEUMANN);

        // Left face (x=-1) mirrors x=0; right face (x=16) mirrors x=15.
        assert_eq!(halo_at(&halo, 0, 1, 1), 0.0);
        assert_eq!(halo_at(&halo, 17, 1, 1), 15.0);
    }

    fn fill_axis(grid: &mut SparseGrid<f32, BSX, N>, f: impl Fn(usize) -> f32) {
        for z in 0..grid.sz {
            for y in 0..grid.sy {
                for x in 0..grid.sx {
                    grid.set_voxel(x, y, z, f(x));
                }
            }
        }
    }

    // Awkward: domain-corner block has 7 missing neighbors; all resolve.
    #[test]
    fn test_h_07_domain_corner_block() {
        let mut grid = setup_grid(48, 48, 48);
        // Allocate only block (0,0,0) with a recognizable value; leave the rest
        // empty so the 7 missing neighbors hit dummy/out-of-domain paths.
        grid.set_voxel(0, 0, 0, 5.0);
        grid.set_empty_value(2.0);

        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, f32, N>(&grid, 0, BoundaryCondition::Dirichlet);

        // Interior = 5.0 (allocated block), out-of-domain/empty faces = 2.0.
        assert_eq!(halo_at(&halo, 1, 1, 1), 5.0);
        assert_eq!(halo_at(&halo, 0, 1, 1), 2.0); // x=-1 out of domain
        assert_eq!(halo_at(&halo, 1, 0, 1), 2.0); // y=-1
        assert_eq!(halo_at(&halo, 1, 1, 0), 2.0); // z=-1
        assert_eq!(halo_at(&halo, 17, 1, 1), 2.0); // x=16 -> empty block (1,0,0)
        assert_eq!(halo_at(&halo, 0, 0, 0), 2.0); // corner out-of-domain
    }

    // Awkward: partial last block. The block covers global x=16..31 but only
    // x=16 is in-domain; Neumann mirrors the rest.
    #[test]
    fn test_h_08_partial_last_block() {
        let mut grid = setup_grid(17, 16, 16);
        fill_axis(&mut grid, |x| x as f32);

        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, f32, N>(&grid, grid.get_block_id(16, 0, 0), NEUMANN);

        // Interior global x=16 -> 16.0; global 17 mirrors to 16; global 18 to 15.
        assert_eq!(halo_at(&halo, 1, 1, 1), 16.0);
        assert_eq!(halo_at(&halo, 2, 1, 1), 16.0);
        assert_eq!(halo_at(&halo, 3, 1, 1), 15.0);
        // Right face (global 32) mirrors to 1.
        assert_eq!(halo_at(&halo, 17, 1, 1), 1.0);
    }

    // Dummy neighbors: empty/full blocks read their sentinel values.
    #[test]
    fn test_h_09_empty_full_dummy_neighbors() {
        let mut grid = setup_grid(32, 16, 16);
        grid.set_voxel(0, 0, 0, 3.0); // allocate block 0
        grid.blockmap[1] = Some(grid.full_block); // right neighbor = full (1.0)
        grid.set_empty_value(2.0);

        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, f32, N>(&grid, 0, NEUMANN);

        // Right face = full_value (1.0); left face (out-of-domain) mirrors
        // edge voxel x=0 (3.0), not empty.
        assert_eq!(halo_at(&halo, 17, 1, 1), 1.0);
        assert_eq!(halo_at(&halo, 0, 1, 1), 3.0);
    }

    // Consistency: faces-only fill matches full fill on interior + faces, and
    // leaves edges/corners zero.
    #[test]
    fn test_h_10_faces_only_is_subset_of_full() {
        let mut grid = setup_grid(48, 48, 48);
        for bid in 0..grid.n_blocks {
            let bx = bid % grid.nx;
            let by = (bid / grid.nx) % grid.ny;
            let bz = bid / grid.nxy;
            grid.set_voxel(bx * BSX, by * BSX, bz * BSX, bid as f32);
            if let Some(ptr) = grid.blockmap[bid]
                && ptr != grid.empty_block
                && ptr != grid.full_block
            {
                unsafe { (*ptr.as_ptr()).data.fill(bid as f32) };
            }
        }

        let mut full = HaloBlock::<BSX, HSX>::new();
        full.fill::<1, true, f32, N>(&grid, 13, NEUMANN);
        let mut faces = HaloBlock::<BSX, HSX>::new();
        faces.fill::<1, false, f32, N>(&grid, 13, NEUMANN);

        for z in 0..HSX {
            for y in 0..HSX {
                for x in 0..HSX {
                    let x_halo = x == 0 || x == BSX + 1;
                    let y_halo = y == 0 || y == BSX + 1;
                    let z_halo = z == 0 || z == BSX + 1;
                    // Faces-only fills the middle everywhere and the x-halo
                    // columns only on center rows; x-halo on edge/corner rows
                    // is left zero.
                    if x_halo && (y_halo || z_halo) {
                        assert_eq!(faces.data[z * HSX * HSX + y * HSX + x], 0.0);
                    } else {
                        assert_eq!(
                            full.data[z * HSX * HSX + y * HSX + x],
                            faces.data[z * HSX * HSX + y * HSX + x]
                        );
                    }
                }
            }
        }
    }

    // Dequant: u16 density fill produces a [0,1] f32 halo.
    #[test]
    fn test_h_11_density_dequant() {
        let pool = Arc::new(BlockPool::<Density, BSX, N>::new(64, 64));
        let mut grid = SparseGrid::new(
            "dens".to_string(),
            32, 16, 16,
            Density(0),
            Density(u16::MAX),
            pool,
        );
        for x in 0..32 {
            grid.set_voxel(x, 0, 0, Density(((x as f32 / 31.0) * u16::MAX as f32).round() as u16));
        }

        let mut halo = HaloBlock::<BSX, HSX>::new();
        halo.fill::<1, true, Density, N>(&grid, 0, NEUMANN);

        // Interior voxel at global x=15 -> 15/31 ~ 0.4839.
        let v = halo_at(&halo, 16, 1, 1);
        assert!((v - 15.0 / 31.0).abs() < 1e-3, "got {v}");
    }

    // HALO=2 full fill: a 5^3 neighborhood resolves +-2 neighbors correctly.
    #[test]
    fn test_h_12_halo2_fill() {
        const HSX2: usize = BSX + 4;
        let mut grid = setup_grid(48, 48, 48);
        for bid in 0..grid.n_blocks {
            let bx = bid % grid.nx;
            let by = (bid / grid.nx) % grid.ny;
            let bz = bid / grid.nxy;
            grid.set_voxel(bx * BSX, by * BSX, bz * BSX, bid as f32);
            if let Some(ptr) = grid.blockmap[bid]
                && ptr != grid.empty_block
                && ptr != grid.full_block
            {
                unsafe { (*ptr.as_ptr()).data.fill(bid as f32) };
            }
        }

        // Block 13 = (1,1,1) in a 3^3 block grid (bid = bx + by*3 + bz*9).
        let mut halo = HaloBlock::<BSX, HSX2>::new();
        halo.fill::<2, true, f32, N>(&grid, 13, NEUMANN);

        let at = |x: usize, y: usize, z: usize| halo.data[z * HSX2 * HSX2 + y * HSX2 + x];

        // Halo index x maps to global x0 + (x - HALO); interior is x in [2, 18).
        assert_eq!(at(2, 2, 2), 13.0); // center (global 16) -> block 13
        assert_eq!(at(0, 2, 2), 12.0); // x=-2 (global 14) -> block (0,1,1) = 12
        assert_eq!(at(1, 2, 2), 12.0); // x=-1 (global 15) -> block 12
        assert_eq!(at(18, 2, 2), 14.0); // x=+1 (global 32) -> block (2,1,1) = 14
        assert_eq!(at(19, 2, 2), 14.0); // x=+2 (global 33) -> block 14
        assert_eq!(at(2, 0, 2), 10.0); // y=-2 -> block (1,0,1) = 10
        assert_eq!(at(2, 18, 2), 16.0); // y=+1 -> block (1,2,1) = 16
        assert_eq!(at(2, 2, 0), 4.0); // z=-2 -> block (1,1,0) = 4
        assert_eq!(at(2, 2, 18), 22.0); // z=+1 -> block (1,1,2) = 22
        assert_eq!(at(0, 0, 0), 0.0); // corner (-2,-2,-2) -> block (0,0,0) = 0
        assert_eq!(at(19, 19, 19), 26.0); // corner (+2,+2,+2) -> block (2,2,2) = 26
    }

    #[test]
    fn test_h_13_pool_distinct_per_thread() {
        use std::sync::Barrier;

        let n_threads = rayon::current_num_threads().max(2);
        let pool = Arc::new(HaloBlockPool::<BSX, HSX>::new(n_threads));
        let barrier = Arc::new(Barrier::new(n_threads));

        let addresses: Vec<usize> = (0..n_threads)
            .into_par_iter()
            .map(|_| {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                barrier.wait();
                let halo = unsafe { pool.get_mut() };
                halo as *const _ as usize
            })
            .collect();

        let mut sorted = addresses;
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            assert_ne!(w[0], w[1], "two threads received the same halo buffer");
        }
    }
}

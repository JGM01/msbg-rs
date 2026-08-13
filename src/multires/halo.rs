use crate::sparse_grid::SparseGrid;
use std::cell::UnsafeCell;

/// A thread-local staging buffer holding a central block and a 1-voxel
/// padding (halo) around all 6 faces for safe, branchless SIMD stencils.
/// `HSX` must be exactly `BSX + 2`.
pub struct HaloBlock<D, const BSX: usize, const HSX: usize, const N_HALO: usize>
where
    D: Copy + Default + Send + Sync,
{
    pub data: [D; N_HALO],
}

impl<D, const BSX: usize, const HSX: usize, const N_HALO: usize> HaloBlock<D, BSX, HSX, N_HALO>
where
    D: Copy + Default + Send + Sync,
{
    #[inline(always)]
    pub fn new() -> Self {
        assert_eq!(HSX, BSX + 2, "Halo dimensions must be exactly BSX + 2");
        assert_eq!(N_HALO, HSX * HSX * HSX, "N_HALO must equal HSX^3");
        Self {
            data:[D::default(); N_HALO],
        }
    }

    /// Resolves the 6 spatial neighbors of a block and copies them into the staging buffer.
    /// Missing neighbors smoothly substitute the grid's dummy blocks.
    #[inline]
    pub fn fill<const N: usize>(&mut self, grid: &SparseGrid<D, BSX, N>, bid: usize) {
        // Pre-computed constants at compile-time
        const DY: usize = 18;
        const DZ: usize = 18 * 18; // 324

        // Extract 3D block coordinates without repeated 64-bit division
        let nx = grid.nx;
        let nxy = grid.nxy;

        let bz = bid / nxy;
        let rem = bid % nxy;
        let by = rem / nx;
        let bx = rem % nx;

        let empty_ptr = unsafe { (*grid.empty_block.as_ptr()).data.as_ptr() };
        let full_ptr = unsafe { (*grid.full_block.as_ptr()).data.as_ptr() };

        let get_blk = |valid: bool, n_bid: usize| -> *const D {
            if !valid {
                return empty_ptr;
            }
            unsafe {
                match grid.blockmap.get_unchecked(n_bid) {
                    Some(ptr) if *ptr == grid.empty_block => empty_ptr,
                    Some(ptr) if *ptr == grid.full_block => full_ptr,
                    Some(ptr) => (*ptr.as_ptr()).data.as_ptr(),
                    None => empty_ptr,
                }
            }
        };

        let center_ptr = get_blk(true, bid);
        let left_ptr   = get_blk(bx > 0, bid.wrapping_sub(1));
        let right_ptr  = get_blk(bx + 1 < nx, bid + 1);
        let bot_ptr    = get_blk(by > 0, bid.wrapping_sub(nx));
        let top_ptr    = get_blk(by + 1 < grid.ny, bid + nx);
        let back_ptr   = get_blk(bz > 0, bid.wrapping_sub(nxy));
        let front_ptr  = get_blk(bz + 1 < grid.nz, bid + nxy);

        // 1. Center copy
        for z in 0..BSX {
            for y in 0..BSX {
                let halo_idx = (z + 1) * DZ + (y + 1) * DY + 1;
                let blk_idx = z * 256 + y * BSX;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        center_ptr.add(blk_idx),
                        self.data.as_mut_ptr().add(halo_idx),
                        BSX,
                    );
                }
            }
        }

        // 2. Left and Right faces
        for z in 0..BSX {
            for y in 0..BSX {
                let halo_idx_l = (z + 1) * DZ + (y + 1) * DY;
                let blk_idx_l = z * 256 + y * BSX + (BSX - 1);
                unsafe {
                    *self.data.get_unchecked_mut(halo_idx_l) = *left_ptr.add(blk_idx_l);
                }

                let halo_idx_r = (z + 1) * DZ + (y + 1) * DY + (BSX + 1);
                let blk_idx_r = z * 256 + y * BSX;
                unsafe {
                    *self.data.get_unchecked_mut(halo_idx_r) = *right_ptr.add(blk_idx_r);
                }
            }
        }

        // 3. Bottom and Top faces
        for z in 0..BSX {
            let halo_idx_b = (z + 1) * DZ + 1;
            let blk_idx_b = z * 256 + (BSX - 1) * BSX;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bot_ptr.add(blk_idx_b),
                    self.data.as_mut_ptr().add(halo_idx_b),
                    BSX,
                );
            }

            let halo_idx_t = (z + 1) * DZ + (BSX + 1) * DY + 1;
            let blk_idx_t = z * 256;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    top_ptr.add(blk_idx_t),
                    self.data.as_mut_ptr().add(halo_idx_t),
                    BSX,
                );
            }
        }

        // 4. Back and Front faces
        for y in 0..BSX {
            let halo_idx_bk = (y + 1) * DY + 1;
            let blk_idx_bk = (BSX - 1) * 256 + y * BSX;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    back_ptr.add(blk_idx_bk),
                    self.data.as_mut_ptr().add(halo_idx_bk),
                    BSX,
                );
            }

            let halo_idx_fd = (BSX + 1) * DZ + (y + 1) * DY + 1;
            let blk_idx_fd = y * BSX;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    front_ptr.add(blk_idx_fd),
                    self.data.as_mut_ptr().add(halo_idx_fd),
                    BSX,
                );
            }
        }
    }

    /// Simulates memory bandwidth of a `fill` operation for benchmarking.
    #[inline(always)]
    pub fn mock_fill_for_bench(&mut self) {
        // Force CPU to iterate and pull buffer into L1
        self.data.fill(D::default());
    }
}

/// A pre-sized pool of thread-local halo blocks.
/// Designed to be indexed directly via `rayon::current_thread_index()`.
pub struct HaloBlockPool<D, const BSX: usize, const HSX: usize, const N_HALO: usize>
where
    D: Copy + Default + Send + Sync,
{
    slots: Vec<UnsafeCell<HaloBlock<D, BSX, HSX, N_HALO>>>,
}

// Each slot is exclusively accessed by a single physical Rayon thread.
// Assert no reentrant parallel maps occur within the sweep.
unsafe impl<D, const BSX: usize, const HSX: usize, const N_HALO: usize> Sync for HaloBlockPool<D, BSX, HSX, N_HALO> where
    D: Copy + Default + Send + Sync
{
}

impl<D, const BSX: usize, const HSX: usize, const N_HALO: usize> HaloBlockPool<D, BSX, HSX, N_HALO>
where
    D: Copy + Default + Send + Sync,
{
    /// Creates a pool sized to thread pool size.
    pub fn new(num_threads: usize) -> Self {
        let slots = (0..num_threads)
            .map(|_| UnsafeCell::new(HaloBlock::new()))
            .collect();
        Self { slots }
    }

    /// Retrieves mutable halo buffer assigned to the current worker thread.
    ///
    /// Must only be called within a parallel context bounded by the pool size!
    #[inline(always)]
    pub unsafe fn get_mut(&self) -> &mut HaloBlock<D, BSX, HSX, N_HALO> {
        let idx = rayon::current_thread_index().expect("Must run inside the Rayon thread pool");
        unsafe { &mut *self.slots[idx].get() }
    }
}
#[cfg(test)]
mod halo_tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use crate::sparse_grid::SparseGrid;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;
    const HSX: usize = 18;
    const N_HALO: usize = HSX * HSX * HSX; // 5832

    fn setup_grid(sx: usize, sy: usize, sz: usize) -> SparseGrid<f32, BSX, N> {
        let pool = Arc::new(BlockPool::new(64, 64));
        SparseGrid::new("halo_test".to_string(), sx, sy, sz, 0.0, 1.0, pool)
    }

    #[test]
    fn test_h_01_construction() {
        let halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        assert_eq!(halo.data.len(), N_HALO);
        assert!(halo.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    #[should_panic(expected = "Halo dimensions must be exactly BSX + 2")]
    fn test_h_02_bad_hsx_panics() {
        let _ = HaloBlock::<f32, 16, 17, 4913>::new();
    }

    #[test]
    fn test_h_03_center_copy() {
        let mut grid = setup_grid(16, 16, 16); // single block
        // Fill the only block with a recognisable pattern
        for z in 0..BSX {
            for y in 0..BSX {
                for x in 0..BSX {
                    let v = (x + y * 16 + z * 256) as f32;
                    grid.set_voxel(x, y, z, v);
                }
            }
        }
        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        halo.fill(&grid, 0);

        // Interior must match the original block
        let dy = HSX;
        let dz = HSX * HSX;
        for z in 0..BSX {
            for y in 0..BSX {
                for x in 0..BSX {
                    let halo_idx = (z + 1) * dz + (y + 1) * dy + (x + 1);
                    let expected = (x + y * 16 + z * 256) as f32;
                    assert_eq!(halo.data[halo_idx], expected);
                }
            }
        }
    }

    #[test]
    fn test_h_04_boundary_missing_neighbors_use_empty() {
        // 1^3 block grid — every face is missing
        let mut grid = setup_grid(16, 16, 16);
        grid.set_voxel(0, 0, 0, 42.0); // force allocation of the single block
        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        halo.fill(&grid, 0);

        let dy = HSX;
        let dz = HSX * HSX;

        // Left face (x=0) of halo must be empty_value
        for z in 1..=BSX {
            for y in 1..=BSX {
                let idx = z * dz + y * dy + 0;
                assert_eq!(halo.data[idx], 0.0);
            }
        }

        // Right face (x=HSX-1)
        for z in 1..=BSX {
            for y in 1..=BSX {
                let idx = z * dz + y * dy + (HSX - 1);
                assert_eq!(halo.data[idx], 0.0);
            }
        }

        // Likewise for the other four faces (spot-check a few)
        assert_eq!(halo.data[1 * dy + 1], 0.0); // bottom
        assert_eq!(halo.data[(BSX + 1) * dy + 1], 0.0); // top
        assert_eq!(halo.data[1], 0.0); // back
        assert_eq!(halo.data[(BSX + 1) * dz + 1 * dy + 1], 0.0); // front
    }

    #[test]
    fn test_h_05_full_dummy_neighbor() {
        // 2x1x1 block grid so we have a real left/right pair
        let mut grid = setup_grid(32, 16, 16);
        // Allocate block 0 with a pattern, force block 1 to full_block
        for i in 0..N {
            // write via set_voxel on first block
            let x = i % BSX;
            let y = (i / BSX) % BSX;
            let z = i / (BSX * BSX);
            grid.set_voxel(x, y, z, i as f32);
        }
        grid.blockmap[1] = Some(grid.full_block);

        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        // Fill from block 0 — its right neighbor is the full dummy
        halo.fill(&grid, 0);

        let dy = HSX;
        let dz = HSX * HSX;

        // Right face of halo must be full_value (1.0)
        for z in 1..=BSX {
            for y in 1..=BSX {
                let idx = z * dz + y * dy + (HSX - 1);
                assert_eq!(halo.data[idx], 1.0);
            }
        }
    }

    #[test]
    fn test_h_06_empty_dummy_neighbor() {
        let mut grid = setup_grid(32, 16, 16);
        grid.set_voxel(0, 0, 0, 7.0); // allocate block 0
        grid.blockmap[1] = Some(grid.empty_block);

        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        halo.fill(&grid, 0);

        let dy = HSX;
        let dz = HSX * HSX;

        for z in 1..=BSX {
            for y in 1..=BSX {
                let idx = z * dz + y * dy + (HSX - 1);
                assert_eq!(halo.data[idx], 0.0);
            }
        }
    }

    #[test]
    fn test_h_07_all_six_faces() {
        // 3^3 block grid so the centre block has six real neighbours
        let mut grid = setup_grid(48, 48, 48);
        let center_bid = grid.get_block_id(16, 16, 16); // block (1,1,1)

        // Give every block a unique constant value = its bid
        for bid in 0..grid.n_blocks {
            let bx = bid % grid.nx;
            let by = (bid / grid.nx) % grid.ny;
            let bz = bid / grid.nxy;

            let base_x = bx * BSX;
            let base_y = by * BSX;
            let base_z = bz * BSX;

            // Allocate by writing one voxel
            grid.set_voxel(base_x, base_y, base_z, bid as f32);

            // Fill the whole block with the same value for easy checking
            if let Some(ptr) = grid.blockmap[bid] {
                if ptr != grid.empty_block && ptr != grid.full_block {
                    unsafe {
                        (*ptr.as_ptr()).data.fill(bid as f32);
                    }
                }
            }
        }

        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        halo.fill(&grid, center_bid);

        let dy = HSX;
        let dz = HSX * HSX;

        // Centre interior == center_bid
        assert_eq!(halo.data[1 * dz + 1 * dy + 1], center_bid as f32);

        // Left face (x=0) must come from bid-1
        let left_bid = center_bid - 1;
        assert_eq!(halo.data[1 * dz + 1 * dy + 0], left_bid as f32);

        // Right face
        let right_bid = center_bid + 1;
        assert_eq!(halo.data[1 * dz + 1 * dy + (HSX - 1)], right_bid as f32);

        // Bottom / top / back / front
        assert_eq!(
            halo.data[1 * dz + 0 * dy + 1],
            (center_bid - grid.nx) as f32
        );
        assert_eq!(
            halo.data[1 * dz + (HSX - 1) * dy + 1],
            (center_bid + grid.nx) as f32
        );
        assert_eq!(
            halo.data[0 * dz + 1 * dy + 1],
            (center_bid - grid.nxy) as f32
        );
        assert_eq!(
            halo.data[(HSX - 1) * dz + 1 * dy + 1],
            (center_bid + grid.nxy) as f32
        );
    }

    #[test]
    fn test_h_08_mock_fill() {
        let mut halo = HaloBlock::<f32, BSX, HSX, N_HALO>::new();
        // Put something non-zero first
        halo.data.fill(3.14);
        halo.mock_fill_for_bench();
        assert!(halo.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_h_09_pool_distinct_per_thread() {
        use std::sync::{Arc, Barrier};

        let n_threads = rayon::current_num_threads().max(2);
        let pool = Arc::new(HaloBlockPool::<f32, BSX, HSX, N_HALO>::new(n_threads));
        let barrier = Arc::new(Barrier::new(n_threads));

        let addresses: Vec<usize> = (0..n_threads)
            .into_par_iter()
            .map(|_| {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);

                // Wait until every worker has entered the closure
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

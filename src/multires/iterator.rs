use crate::sparse_grid::SparseGrid;
use std::simd::f32x8;

/// The payload yielded to free-standing SIMD kernels.
pub struct BlockSimdContext {
    pub bid: usize,
    /// 32-byte aligned raw pointer for SIMD reads/writes.
    pub data: *mut f32x8,
}

pub struct BlockIterator<'a, const BSX: usize, const N: usize> {
    grid: &'a SparseGrid<f32, BSX, N>,
    bids: Vec<usize>,
    pos: usize,
}

impl<'a, const BSX: usize, const N: usize> BlockIterator<'a, BSX, N> {
    pub fn new(grid: &'a SparseGrid<f32, BSX, N>) -> Self {
        let bids = grid.active_block_ids().collect();
        Self { grid, bids, pos: 0 }
    }
}

impl<'a, const BSX: usize, const N: usize> Iterator for BlockIterator<'a, BSX, N> {
    type Item = BlockSimdContext;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.bids.len() {
            let bid = self.bids[self.pos];
            self.pos += 1;

            let data_ptr = self.grid.value_block_ptr(bid) as *mut f32x8;
            if !data_ptr.is_null() {
                return Some(BlockSimdContext {
                    bid,
                    data: data_ptr,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod iterator_tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use crate::sparse_grid::SparseGrid;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn setup_grid(sx: usize, sy: usize, sz: usize) -> SparseGrid<f32, BSX, N> {
        let pool = Arc::new(BlockPool::new(32, 64));
        SparseGrid::new("iter_test".to_string(), sx, sy, sz, 0.0, 1.0, pool)
    }

    #[test]
    fn test_i_01_empty_grid_yields_nothing() {
        let mut grid = setup_grid(32, 32, 32);
        let mut iter = BlockIterator::new(&mut grid);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_i_02_only_dummies_are_skipped() {
        let mut grid = setup_grid(32, 32, 32);
        // Force a few slots to the sentinel blocks
        grid.set_empty_block(0);
        grid.set_full_block(1);
        grid.set_empty_block(2);

        let mut iter = BlockIterator::new(&mut grid);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_i_03_mixed_real_and_dummy() {
        let mut grid = setup_grid(48, 16, 16); // 3 blocks in X
        // Allocate only the middle block
        grid.set_voxel(16, 0, 0, 42.0);
        // Force the other two to dummies
        grid.set_empty_block(0);
        grid.set_full_block(2);

        let mut iter = BlockIterator::new(&mut grid);
        let ctx = iter.next().expect("should yield the real block");
        assert_eq!(ctx.bid, 1);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_i_04_data_pointer_valid_and_aligned() {
        let mut grid = setup_grid(16, 16, 16);
        // Write a recognisable pattern
        for i in 0..N {
            let x = i % BSX;
            let y = (i / BSX) % BSX;
            let z = i / (BSX * BSX);
            grid.set_voxel(x, y, z, i as f32);
        }

        let mut iter = BlockIterator::new(&mut grid);
        let ctx = iter.next().expect("one real block expected");
        assert_eq!(ctx.bid, 0);

        // Pointer must be 32-byte aligned (f32x8)
        let addr = ctx.data as usize;
        assert_eq!(addr % 32, 0, "SIMD pointer not 32-byte aligned");

        // And it must really point at the block data
        unsafe {
            let first = *ctx.data; // first f32x8
            // Just check the very first lane matches what we wrote
            assert_eq!(first[0], 0.0);
        }
    }

    #[test]
    fn test_i_05_exhaustion() {
        let mut grid = setup_grid(32, 16, 16); // two blocks
        grid.set_voxel(0, 0, 0, 1.0);
        grid.set_voxel(16, 0, 0, 2.0);

        let mut iter = BlockIterator::new(&mut grid);
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
        // Further calls stay None
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_i_06_yields_exactly_the_value_blocks() {
        let mut grid = setup_grid(48, 16, 16); // three blocks
        // Allocate 0 and 2, leave 1 empty
        grid.set_voxel(0, 0, 0, 10.0);
        grid.set_voxel(32, 0, 0, 30.0);

        // Map order, not sorted — collect and compare as a set.
        let iter = BlockIterator::new(&grid);
        let mut bids: Vec<usize> = iter.map(|ctx| ctx.bid).collect();
        bids.sort_unstable();
        assert_eq!(bids, vec![0, 2]);
    }
}

use crate::sparse_grid::SparseGrid;
use std::simd::f32x8;

/// The payload yielded to free-standing SIMD kernels.
pub struct BlockSimdContext {
    pub bid: usize,
    /// 32-byte aligned raw pointer for SIMD reads/writes.
    pub data: *mut f32x8,
}

pub struct BlockIterator<'a, const BSX: usize, const N: usize> {
    grid: &'a mut SparseGrid<f32, BSX, N>,
    current_bid: usize,
}

impl<'a, const BSX: usize, const N: usize> BlockIterator<'a, BSX, N> {
    pub fn new(grid: &'a mut SparseGrid<f32, BSX, N>) -> Self {
        Self {
            grid,
            current_bid: 0,
        }
    }
}

impl<'a, const BSX: usize, const N: usize> Iterator for BlockIterator<'a, BSX, N> {
    type Item = BlockSimdContext;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_bid < self.grid.n_blocks {
            let bid = self.current_bid;
            self.current_bid += 1;

            if let Some(ptr) = self.grid.blockmap[bid] {
                // Filter out the static sentinel blocks (Empty/Full)
                if ptr != self.grid.empty_block && ptr != self.grid.full_block {
                    // data is asserted to be at offset 0 and 64-byte aligned.
                    let data_ptr = unsafe { (*ptr.as_ptr()).data.as_mut_ptr() as *mut f32x8 };

                    return Some(BlockSimdContext {
                        bid,
                        data: data_ptr,
                    });
                }
            }
        }
        None
    }
}

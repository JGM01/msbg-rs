//! `MultiresGrid`: the multiresolution sparse block grid (C++ `MultiresSparseGrid`).

use crate::multires::blockinfo::BlockInfoStore;
use crate::multires::level::{level_from_block_size, Level};
use crate::multires::refinement::{
    compute_block_topology, regularize_refinement_map, BlockGridDims, RefinementMap, Topology,
};

/// Maximum number of resolution levels (C++ `MSBG_MAXRESLEVELS`).
pub const MAX_LEVELS: usize = 5;

/// A stack of resolution levels over one domain, plus per-block topology.
pub struct MultiresGrid {
    pub name: String,
    /// Number of resolution levels (`_nLevels`).
    pub n_levels: usize,
    /// Finest-level block size (power of two, ≤ 32).
    pub bsx0: usize,
    /// Finest-level voxel resolution.
    pub sx0: usize,
    pub sy0: usize,
    pub sz0: usize,
    /// Fine-coarse transition width in finest-level cells.
    pub dtrans_res: usize,
    /// Level-0 block-grid dimensions (identical across levels).
    pub dims: BlockGridDims,
    pub block_info: BlockInfoStore,
    pub levels: Vec<Level>,
}

impl MultiresGrid {
    /// C++ `MultiresSparseGrid::create`. `block_size0` is the finest-level block
    /// size; dimensions must be exact multiples of it (C++ asserts the same).
    pub fn create(
        name: &str,
        sx0: usize,
        sy0: usize,
        sz0: usize,
        block_size0: usize,
        n_levels: usize,
        dtrans_res: usize,
    ) -> Self {
        assert!(block_size0.is_power_of_two(), "block_size0 must be a power of two");
        assert!(block_size0 <= 32, "block_size0 must be <= 32");
        assert!((2..=MAX_LEVELS).contains(&n_levels), "n_levels must be in 2..={MAX_LEVELS}");
        assert!(
            sx0.is_multiple_of(block_size0) && sy0.is_multiple_of(block_size0) && sz0.is_multiple_of(block_size0),
            "domain dimensions must be exact multiples of block_size0"
        );

        let nx = sx0 / block_size0;
        let ny = sy0 / block_size0;
        let nz = sz0 / block_size0;
        let dims = BlockGridDims::new(nx, ny, nz);

        let mut levels = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let sx = sx0 >> l;
            let sy = sy0 >> l;
            let sz = sz0 >> l;
            let bsx = block_size0 >> l;
            levels.push(level_from_block_size(&format!("{name}:L{l}"), bsx, sx, sy, sz));
        }

        let block_info = BlockInfoStore::new(dims.n_blocks, n_levels);

        Self {
            name: name.to_string(),
            n_levels,
            bsx0: block_size0,
            sx0,
            sy0,
            sz0,
            dtrans_res,
            dims,
            block_info,
            levels,
        }
    }

    /// C++ `regularizeRefinementMap`: enforce the 2:1 refinement constraint in
    /// place, then `setRefinementMap`: compute block levels/flags, initialize
    /// cell flags, and fill `CH_DIST_FINECOARSE` (levelMg == 0, no obstacles).
    pub fn set_refinement_map(&mut self, map: &mut RefinementMap) -> Topology {
        regularize_refinement_map(&mut map.levels, &self.dims, self.n_levels);
        let topo = compute_block_topology(&mut self.block_info, map, &self.dims, self.n_levels, None);

        for level in 0..self.n_levels {
            self.levels[level].init_cell_flags(level, map, &self.dims, &self.block_info, None);
        }
        for level in 0..self.n_levels {
            self.levels[level].init_dist_fine_coarse(self.dtrans_res, map, &self.dims, &self.block_info);
            self.levels[level].propagate_dist_fine_coarse_full(level, self.dtrans_res, &self.dims, &self.block_info);
        }

        topo
    }

    /// Effective resolution level of `bid` at levelMg 0.
    #[inline(always)]
    pub fn block_level(&self, bid: usize) -> u8 {
        self.block_info.level0[bid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_01_create_dims() {
        let g = MultiresGrid::create("t", 64, 64, 64, 16, 3, 6);
        assert_eq!(g.n_levels, 3);
        assert_eq!(g.dims.nx, 4);
        assert_eq!(g.dims.n_blocks, 64);
        assert_eq!(g.levels[0].bsx(), 16);
        assert_eq!(g.levels[0].sx(), 64);
        assert_eq!(g.levels[1].bsx(), 8);
        assert_eq!(g.levels[1].sx(), 32);
        assert_eq!(g.levels[2].bsx(), 4);
        assert_eq!(g.levels[2].sx(), 16);
        // Block count is identical across levels.
        assert_eq!(g.levels[0].nx(), g.levels[2].nx());
    }

    #[test]
    fn test_grid_02_uniform_refinement_map() {
        let mut g = MultiresGrid::create("t", 64, 64, 64, 16, 3, 6);
        let mut map = RefinementMap::uniform(g.dims.n_blocks, 0);
        let topo = g.set_refinement_map(&mut map);
        assert_eq!(topo.blocks_fine_coarse[0], Vec::<u32>::new());
        for bid in 0..g.dims.n_blocks {
            assert_eq!(g.block_level(bid), 0);
        }
    }

    #[test]
    fn test_grid_03_fine_island_flags() {
        let mut g = MultiresGrid::create("t", 64, 64, 64, 16, 3, 6);
        let mut map = RefinementMap::uniform(g.dims.n_blocks, 1);
        map.levels[21] = 0; // block (1,1,1) in 4^3 grid
        g.set_refinement_map(&mut map);
        assert_eq!(g.block_level(21), 0);
        assert!(g.block_info.flags(21, 0).contains(crate::multires::blockinfo::BlockFlags::FINE_COARSE));
    }

    #[test]
    #[should_panic(expected = "block_size0 must be a power of two")]
    fn test_grid_04_non_pow2_block_size() {
        let _ = MultiresGrid::create("t", 64, 64, 64, 15, 3, 6);
    }

    #[test]
    #[should_panic(expected = "domain dimensions must be exact multiples")]
    fn test_grid_05_non_multiple_dims() {
        let _ = MultiresGrid::create("t", 65, 64, 64, 16, 3, 6);
    }
}

use std::{ptr::NonNull, sync::Arc};

use crate::blockpool::{Block, BlockPool};

/// Prevents accidental mutation of the shared `empty` or `full` dummy blocks.
pub enum BlockRef<'a, D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> {
    /// An allocated block containing mutable data
    Allocated(&'a mut Block<D, BSX, N>),
    /// A virtual block that is entirely empty
    Empty,
    /// A virtual block that is entirely full (e.g., solid obstacles)
    Full,
}

/// Sparse 3D spatial grid
pub struct SparseGrid<D, const BSX: usize, const N: usize>
where
    D: Copy + Default + Send + Sync,
{
    /// Diagnostic label.
    pub name: String,

    // Scalar Dimensions (in Voxels)
    pub sx: usize,
    pub sy: usize,
    pub sz: usize,

    // Block Grid Dimensions (in Blocks)
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Precomputed 2D slice stride (`nx * ny`) to save multiplications.
    pub nxy: usize,
    pub n_blocks: usize,

    /// Flat array mapping a 1D Block ID (bid) to an allocated memory block.
    /// `Option<NonNull>` is guaranteed by Rust to have the exact same 8-byte footprint as a C++ pointer.
    pub blockmap: Vec<Option<NonNull<Block<D, BSX, N>>>>,

    /// Memory pool for spatial regions.
    pub block_pool: Arc<BlockPool<D, BSX, N>>,

    /// Shared dummy block handles
    pub empty_block: NonNull<Block<D, BSX, N>>,
    pub full_block: NonNull<Block<D, BSX, N>>,

    pub empty_value: D,
    pub full_value: D,
}

impl<D, const BSX: usize, const N: usize> SparseGrid<D, BSX, N>
where
    D: Copy + Default + Send + Sync,
{
    // Compile-time math constants
    const BSX_LOG2: usize = BSX.trailing_zeros() as usize;
    const BSX_MASK: usize = BSX - 1;

    pub fn new(
        name: String,
        sx: usize,
        sy: usize,
        sz: usize,
        empty_value: D,
        full_value: D,
        block_pool: Arc<BlockPool<D, BSX, N>>,
    ) -> Self {
        // Run compile-time assertions before any allocations
        assert!(BSX.is_power_of_two(), "BSX must be a power of two");
        assert_eq!(N, BSX * BSX * BSX, "N must equal BSX^3");

        // Internalize dummy block allocations to guarantee they live
        // as long as this SparseGrid instance
        let empty_block = block_pool.alloc_block();
        let full_block = block_pool.alloc_block();

        // Populate the dummy blocks with their respective constant values
        unsafe {
            (*empty_block.as_ptr()).data.fill(empty_value);
            (*full_block.as_ptr()).data.fill(full_value);
        }

        let nx = (sx + BSX - 1) >> Self::BSX_LOG2;
        let ny = (sy + BSX - 1) >> Self::BSX_LOG2;
        let nz = (sz + BSX - 1) >> Self::BSX_LOG2;

        let nxy = nx * ny;
        let n_blocks = nxy * nz;

        Self {
            name,
            sx,
            sy,
            sz,
            nx,
            ny,
            nz,
            nxy,
            n_blocks,
            blockmap: vec![None; n_blocks],
            block_pool,
            empty_block,
            full_block,
            empty_value,
            full_value,
        }
    }

    /// Convert 3D voxel coordinates (x,y,z) into a 1D Block ID (bid).
    #[inline(always)]
    pub fn get_block_id(&self, x: usize, y: usize, z: usize) -> usize {
        let bx = x >> Self::BSX_LOG2;
        let by = y >> Self::BSX_LOG2;
        let bz = z >> Self::BSX_LOG2;

        bx + (by * self.nx) + (bz * self.nxy)
    }

    /// Convert 3D voxel coordinates (x,y,z) into a 1D Voxel-in-Block ID (vid).
    #[inline(always)]
    pub fn get_voxel_id(x: usize, y: usize, z: usize) -> usize {
        let vx = x & Self::BSX_MASK;
        let vy = y & Self::BSX_MASK;
        let vz = z & Self::BSX_MASK;

        vx | (vy << Self::BSX_LOG2) | (vz << (2 * Self::BSX_LOG2))
    }

    /// Safely retrieve a block from the map.
    #[inline(always)]
    pub fn get_block(&mut self, bid: usize) -> BlockRef<'_, D, BSX, N> {
        match self.blockmap[bid] {
            Some(ptr) if ptr == self.empty_block => BlockRef::Empty,
            Some(ptr) if ptr == self.full_block => BlockRef::Full,
            Some(mut ptr) => unsafe { BlockRef::Allocated(ptr.as_mut()) },
            None => BlockRef::Empty,
        }
    }

    /// Get a single voxel value without allocating memory.
    #[inline(always)]
    pub fn get_voxel(&mut self, x: usize, y: usize, z: usize) -> D {
        debug_assert!(x < self.sx && y < self.sy && z < self.sz);

        let bid = self.get_block_id(x, y, z);
        let vid = Self::get_voxel_id(x, y, z);

        match self.get_block(bid) {
            BlockRef::Allocated(block) => unsafe { *block.data.get_unchecked(vid) },
            BlockRef::Empty => self.empty_value,
            BlockRef::Full => self.full_value,
        }
    }

    /// Set a voxel value. If the block doesn't exist, lazily allocate it.
    #[inline(always)]
    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, val: D) {
        debug_assert!(x < self.sx && y < self.sy && z < self.sz);

        let bid = self.get_block_id(x, y, z);
        let vid = Self::get_voxel_id(x, y, z);

        let block_ptr = match self.blockmap[bid] {
            // Already allocated real block
            Some(ptr) if ptr != self.empty_block && ptr != self.full_block => ptr,
            // Lazy allocation trigger
            _ => {
                let new_block = self.block_pool.alloc_block();
                self.blockmap[bid] = Some(new_block);
                new_block
            }
        };

        unsafe {
            *(*block_ptr.as_ptr()).data.get_unchecked_mut(vid) = val;
        }
    }
}

// SparseGrid Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    /// Helper to instantiate a grid for testing
    fn setup_grid(sx: usize, sy: usize, sz: usize) -> SparseGrid<f32, BSX, N> {
        let pool = Arc::new(BlockPool::new(16, 16));

        SparseGrid::new("test_grid".to_string(), sx, sy, sz, 0.0, 1.0, pool)
    }

    #[test]
    fn test_grd_01_normal_api_usage() {
        let mut grid = setup_grid(64, 64, 64);

        // Write to a few different blocks
        grid.set_voxel(0, 0, 0, 42.0);
        grid.set_voxel(32, 10, 5, 7.5);
        grid.set_voxel(63, 63, 63, 99.9);

        // Read them back
        assert_eq!(grid.get_voxel(0, 0, 0), 42.0);
        assert_eq!(grid.get_voxel(32, 10, 5), 7.5);
        assert_eq!(grid.get_voxel(63, 63, 63), 99.9);

        // Unwritten voxel should return empty_value
        assert_eq!(grid.get_voxel(1, 1, 1), 0.0);
    }

    #[test]
    #[should_panic(expected = "BSX must be a power of two")]
    fn test_grd_02_panic_non_pow2_bsx() {
        // Capacity bumped to 2 blocks to safely house internal dummy blocks
        let pool = Arc::new(BlockPool::new(1, 2));

        // BSX = 15 is not a power of 2
        let _ =
            SparseGrid::<f32, 15, 3375>::new("bad_grid".to_string(), 30, 30, 30, 0.0, 1.0, pool);
    }

    #[test]
    #[should_panic(expected = "N must equal BSX^3")]
    fn test_grd_03_panic_mismatched_n() {
        // Capacity bumped to 2 blocks to safely house internal dummy blocks
        let pool = Arc::new(BlockPool::new(1, 2));

        // BSX = 16, but N = 100 instead of 4096
        let _ = SparseGrid::<f32, 16, 100>::new("bad_grid".to_string(), 32, 32, 32, 0.0, 1.0, pool);
    }

    #[test]
    #[should_panic]
    fn test_grd_04_panic_oob_x_get() {
        let mut grid = setup_grid(32, 32, 32);
        // sx is 32, so x=32 is out of bounds
        let _ = grid.get_voxel(32, 0, 0);
    }

    #[test]
    #[should_panic]
    fn test_grd_05_panic_oob_z_set() {
        let mut grid = setup_grid(32, 32, 32);
        // sz is 32, so z=50 is wildly out of bounds
        grid.set_voxel(0, 0, 50, 1.0);
    }

    #[test]
    fn test_grd_06_state_lazy_allocation() {
        let mut grid = setup_grid(32, 32, 32);

        let bid = grid.get_block_id(5, 5, 5);

        // State 1: Initially empty
        assert!(matches!(grid.get_block(bid), BlockRef::Empty));
        assert_eq!(grid.get_voxel(5, 5, 5), 0.0);

        // State Change: Write triggers lazy alloc
        grid.set_voxel(5, 5, 5, 123.4);

        // State 2: Now allocated
        assert!(matches!(grid.get_block(bid), BlockRef::Allocated(_)));
        assert_eq!(grid.get_voxel(5, 5, 5), 123.4);
    }

    #[test]
    fn test_grd_07_state_dummy_resolution() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(20, 20, 20); // Belongs to block (1,1,1)

        // Inject a full block marker manually (e.g., simulating a solid obstacle region)
        grid.blockmap[bid] = Some(grid.full_block);

        // Verify state resolution
        assert!(matches!(grid.get_block(bid), BlockRef::Full));

        // Verify it intercepts the read and returns `full_value` (1.0)
        assert_eq!(grid.get_voxel(20, 20, 20), 1.0);
        assert_eq!(grid.get_voxel(21, 25, 31), 1.0); // Anywhere in this block is full
    }

    #[test]
    fn test_grd_08_coordinate_mapping() {
        // Create a 32x32x32 voxel grid.
        // With BSX=16, this is a 2x2x2 block grid (nx=2, ny=2, nz=2).
        let grid = setup_grid(32, 32, 32);

        assert_eq!(grid.nx, 2);
        assert_eq!(grid.ny, 2);
        assert_eq!(grid.nz, 2);
        assert_eq!(grid.nxy, 4);
        assert_eq!(grid.n_blocks, 8);

        // Test 1: Origin
        assert_eq!(grid.get_block_id(0, 0, 0), 0);
        assert_eq!(SparseGrid::<f32, BSX, N>::get_voxel_id(0, 0, 0), 0);

        // Test 2: Just inside block 0 boundary
        // vx=15, vy=15, vz=15 -> vid = 15 | (15<<4) | (15<<8) = 15 | 240 | 3840 = 4095
        assert_eq!(grid.get_block_id(15, 15, 15), 0);
        assert_eq!(SparseGrid::<f32, BSX, N>::get_voxel_id(15, 15, 15), 4095);

        // Test 3: Block (1, 1, 1) -> bid = 1 + (1*2) + (1*4) = 7
        assert_eq!(grid.get_block_id(16, 16, 16), 7);
        // Local voxel is (0, 0, 0) inside the block
        assert_eq!(SparseGrid::<f32, BSX, N>::get_voxel_id(16, 16, 16), 0);

        // Test 4: Asymmetric block lookup
        // x=17 (bx=1), y=0 (by=0), z=31 (bz=1)
        // bid = 1 + (0*2) + (1*4) = 5
        assert_eq!(grid.get_block_id(17, 0, 31), 5);
        // vx = 1, vy = 0, vz = 15
        // vid = 1 | (0<<4) | (15<<8) = 1 | 0 | 3840 = 3841
        assert_eq!(SparseGrid::<f32, BSX, N>::get_voxel_id(17, 0, 31), 3841);
    }
}

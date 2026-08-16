use std::{ptr::NonNull, sync::Arc};

use crate::blockpool::{Block, BlockPool};

/// Thread-safe wrapper.
/// Sharing these pointers across threads is safe because
/// BlockPool + general architecture handle the synchronization.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct BlockPtr<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize>(
    pub NonNull<Block<D, BSX, N>>,
);

impl<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> PartialEq
    for BlockPtr<D, BSX, N>
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 // Just compare the memory addresses...
    }
}

unsafe impl<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> Send
    for BlockPtr<D, BSX, N>
{
}
unsafe impl<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> Sync
    for BlockPtr<D, BSX, N>
{
}

impl<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> BlockPtr<D, BSX, N> {
    #[inline(always)]
    pub fn as_ptr(self) -> *mut Block<D, BSX, N> {
        self.0.as_ptr()
    }
}

/// Immutable view of a block's value state.
pub enum BlockRef<'a, D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> {
    /// An allocated block containing data
    Allocated(&'a Block<D, BSX, N>),
    /// A virtual block that is entirely empty
    Empty,
    /// A virtual block that is entirely full (e.g., solid obstacles)
    Full,
}

/// Mutable view of a block's value state.
pub enum BlockRefMut<'a, D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> {
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
    pub blockmap: Vec<Option<BlockPtr<D, BSX, N>>>,

    /// Memory pool for spatial regions.
    pub block_pool: Arc<BlockPool<D, BSX, N>>,

    /// Shared dummy block handles
    pub empty_block: BlockPtr<D, BSX, N>,
    pub full_block: BlockPtr<D, BSX, N>,

    empty_value: D,
    full_value: D,
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
        let empty_block = BlockPtr(block_pool.alloc_block());
        let full_block = BlockPtr(block_pool.alloc_block());

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

    /// Immutably resolve a block's value state.
    #[inline(always)]
    pub fn get_block(&self, bid: usize) -> BlockRef<'_, D, BSX, N> {
        match self.blockmap[bid] {
            Some(ptr) if ptr == self.empty_block => BlockRef::Empty,
            Some(ptr) if ptr == self.full_block => BlockRef::Full,
            Some(ptr) => unsafe { BlockRef::Allocated(&*ptr.as_ptr()) },
            None => BlockRef::Empty,
        }
    }

    /// Mutably resolve a block's value state.
    #[inline(always)]
    pub fn get_block_mut(&mut self, bid: usize) -> BlockRefMut<'_, D, BSX, N> {
        match self.blockmap[bid] {
            Some(ptr) if ptr == self.empty_block => BlockRefMut::Empty,
            Some(ptr) if ptr == self.full_block => BlockRefMut::Full,
            Some(ptr) => unsafe { BlockRefMut::Allocated(&mut *ptr.as_ptr()) },
            None => BlockRefMut::Empty,
        }
    }

    /// True if `bid` holds an allocated (writable) data block.
    #[inline(always)]
    pub fn is_value_block(&self, bid: usize) -> bool {
        matches!(self.get_block(bid), BlockRef::Allocated(_))
    }

    /// True if `bid` is unallocated or the shared empty dummy.
    #[inline(always)]
    pub fn is_empty_block(&self, bid: usize) -> bool {
        matches!(self.get_block(bid), BlockRef::Empty)
    }

    /// True if `bid` is the shared full dummy.
    #[inline(always)]
    pub fn is_full_block(&self, bid: usize) -> bool {
        matches!(self.get_block(bid), BlockRef::Full)
    }

    /// Inverse of `get_block_id`: block id -> `(bx, by, bz)` block coords.
    #[inline(always)]
    pub fn get_block_coords_by_id(&self, bid: usize) -> (usize, usize, usize) {
        debug_assert!(bid < self.n_blocks);
        (bid % self.nx, (bid / self.nx) % self.ny, bid / self.nxy)
    }

    /// Immutable slice view of an allocated block's data; `None` for dummies.
    #[inline(always)]
    pub fn get_block_data(&self, bid: usize) -> Option<&[D]> {
        match self.get_block(bid) {
            BlockRef::Allocated(block) => Some(block.data.as_slice()),
            _ => None,
        }
    }

    /// Mutable slice view of an allocated block's data; `None` for dummies.
    #[inline(always)]
    pub fn get_block_data_mut(&mut self, bid: usize) -> Option<&mut [D]> {
        match self.get_block_mut(bid) {
            BlockRefMut::Allocated(block) => Some(block.data.as_mut_slice()),
            _ => None,
        }
    }

    /// Sentinel value for empty regions.
    #[inline(always)]
    pub fn empty_value(&self) -> D {
        self.empty_value
    }

    /// Sentinel value for full regions.
    #[inline(always)]
    pub fn full_value(&self) -> D {
        self.full_value
    }

    /// Set the empty sentinel; re-fills the shared empty dummy block.
    pub fn set_empty_value(&mut self, val: D) {
        self.empty_value = val;
        unsafe {
            (*self.empty_block.as_ptr()).data.fill(val);
        }
    }

    /// Set the full sentinel; re-fills the shared full dummy block.
    pub fn set_full_value(&mut self, val: D) {
        self.full_value = val;
        unsafe {
            (*self.full_block.as_ptr()).data.fill(val);
        }
    }

    /// Mark a block as entirely empty (no data allocated).
    #[inline(always)]
    pub fn set_empty_block(&mut self, bid: usize) {
        debug_assert!(bid < self.n_blocks);
        debug_assert!(!matches!(self.get_block(bid), BlockRef::Allocated(_)));
        self.blockmap[bid] = Some(self.empty_block);
    }

    /// Mark a block as entirely full (no data allocated).
    #[inline(always)]
    pub fn set_full_block(&mut self, bid: usize) {
        debug_assert!(bid < self.n_blocks);
        debug_assert!(!matches!(self.get_block(bid), BlockRef::Allocated(_)));
        self.blockmap[bid] = Some(self.full_block);
    }

    /// Get a single voxel value without allocating memory.
    #[inline(always)]
    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> D {
        debug_assert!(x < self.sx && y < self.sy && z < self.sz);
        self.get_voxel_unchecked(x, y, z)
    }

    /// `get_voxel` without the in-range assertion; caller proves `(x,y,z)`.
    #[inline(always)]
    pub fn get_voxel_unchecked(&self, x: usize, y: usize, z: usize) -> D {
        let bid = self.get_block_id(x, y, z);
        let vid = Self::get_voxel_id(x, y, z);
        self.get_value_in_block(bid, vid)
    }

    /// Read a voxel by (block, voxel-in-block) index.
    #[inline(always)]
    pub fn get_value_in_block(&self, bid: usize, vid: usize) -> D {
        debug_assert!(vid < N);
        match self.get_block(bid) {
            BlockRef::Allocated(block) => unsafe { *block.data.get_unchecked(vid) },
            BlockRef::Empty => self.empty_value,
            BlockRef::Full => self.full_value,
        }
    }

    /// Allocate a real data block for `bid` (initialized to `empty_value`) if it
    /// is currently unallocated or a dummy. Returns without effect if a value
    /// block already exists.
    #[inline(always)]
    pub fn ensure_block(&mut self, bid: usize) {
        debug_assert!(bid < self.n_blocks);
        if matches!(self.get_block(bid), BlockRef::Allocated(_)) {
            return;
        }
        let new_block = BlockPtr(self.block_pool.alloc_block());
        unsafe {
            (*new_block.as_ptr()).data.fill(self.empty_value);
        }
        self.blockmap[bid] = Some(new_block);
    }

    /// Set a voxel value. Allocates the block on demand (over `None` or a dummy).
    #[inline(always)]
    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, val: D) {
        debug_assert!(x < self.sx && y < self.sy && z < self.sz);
        let bid = self.get_block_id(x, y, z);
        let vid = Self::get_voxel_id(x, y, z);
        self.set_value_in_block(bid, vid, val);
    }

    /// Set a voxel by (block, voxel-in-block) index; allocates on demand.
    #[inline(always)]
    pub fn set_value_in_block(&mut self, bid: usize, vid: usize, val: D) {
        debug_assert!(vid < N);

        let block_ptr = match self.blockmap[bid] {
            Some(ptr) if ptr != self.empty_block && ptr != self.full_block => ptr,
            _ => {
                let new_block = BlockPtr(self.block_pool.alloc_block());
                unsafe {
                    (*new_block.as_ptr()).data.fill(self.empty_value);
                }
                self.blockmap[bid] = Some(new_block);
                new_block
            }
        };

        unsafe {
            *(*block_ptr.as_ptr()).data.get_unchecked_mut(vid) = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn setup_grid(sx: usize, sy: usize, sz: usize) -> SparseGrid<f32, BSX, N> {
        let pool = Arc::new(BlockPool::new(16, 16));

        SparseGrid::new("test_grid".to_string(), sx, sy, sz, 0.0, 1.0, pool)
    }

    #[test]
    fn test_grd_01_normal_api_usage() {
        let mut grid = setup_grid(64, 64, 64);

        // Write to some blocks
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
        let pool = Arc::new(BlockPool::new(1, 2));

        // BSX = 15 is not a power of 2
        let _ =
            SparseGrid::<f32, 15, 3375>::new("bad_grid".to_string(), 30, 30, 30, 0.0, 1.0, pool);
    }

    #[test]
    #[should_panic(expected = "N must equal BSX^3")]
    fn test_grd_03_panic_mismatched_n() {
        let pool = Arc::new(BlockPool::new(1, 2));

        // BSX = 16, but N = 100 instead of 4096
        let _ = SparseGrid::<f32, 16, 100>::new("bad_grid".to_string(), 32, 32, 32, 0.0, 1.0, pool);
    }

    #[test]
    #[should_panic]
    fn test_grd_04_panic_oob_x_get() {
        let grid = setup_grid(32, 32, 32);
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

        // Initially empty
        assert!(matches!(grid.get_block(bid), BlockRef::Empty));
        assert_eq!(grid.get_voxel(5, 5, 5), 0.0);

        // Write triggers lazy alloc
        grid.set_voxel(5, 5, 5, 123.4);

        // Now allocated
        assert!(matches!(grid.get_block(bid), BlockRef::Allocated(_)));
        assert_eq!(grid.get_voxel(5, 5, 5), 123.4);
    }

    #[test]
    fn test_grd_07_state_dummy_resolution() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(20, 20, 20); // Belongs to block (1,1,1)

        // Inject a full block marker (e.g., simulating a solid obstacle region)
        grid.set_full_block(bid);

        // Verify state resolution
        assert!(matches!(grid.get_block(bid), BlockRef::Full));

        // Verify it intercepts the read and returns `full_value` (1.0)
        assert_eq!(grid.get_voxel(20, 20, 20), 1.0);
        assert_eq!(grid.get_voxel(21, 25, 31), 1.0); // Anywhere in this block is full
    }

    #[test]
    fn test_grd_08_coordinate_mapping() {
        // Create a 32^3 voxel grid.
        // With BSX=16, this is a 2^3 block grid (nx=2, ny=2, nz=2).
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

    #[test]
    fn test_grd_09_partial_last_block() {
        // 17 voxels → 2 blocks in X (last block only has 1 valid voxel column)
        let mut grid = setup_grid(17, 16, 16);

        assert_eq!(grid.nx, 2);
        assert_eq!(grid.ny, 1);
        assert_eq!(grid.nz, 1);
        assert_eq!(grid.n_blocks, 2);

        // Write / read the single valid voxel that lives in the partial block
        grid.set_voxel(16, 0, 0, 42.0);
        assert_eq!(grid.get_voxel(16, 0, 0), 42.0);

        // Voxel that belongs to the first (full) block still works
        grid.set_voxel(15, 0, 0, 7.0);
        assert_eq!(grid.get_voxel(15, 0, 0), 7.0);

        // Unwritten voxel inside the partial block returns empty_value
        assert_eq!(grid.get_voxel(16, 5, 5), 0.0);
    }

    #[test]
    #[should_panic]
    fn test_grd_10_zero_size_grid_get() {
        let grid = setup_grid(0, 0, 0);
        // n_blocks == 0 → any access is out-of-bounds
        let _ = grid.get_voxel(0, 0, 0);
    }

    #[test]
    #[should_panic]
    fn test_grd_11_zero_size_grid_set() {
        let mut grid = setup_grid(0, 16, 16);
        grid.set_voxel(0, 0, 0, 1.0);
    }

    #[test]
    fn test_grd_12_ceiling_division_asymmetric() {
        // Dimensions that are not multiples of BSX and are different on each axis
        let grid = setup_grid(17, 33, 1);

        assert_eq!(grid.nx, 2); // ceil(17/16)
        assert_eq!(grid.ny, 3); // ceil(33/16)
        assert_eq!(grid.nz, 1); // ceil(1/16)
        assert_eq!(grid.nxy, 6);
        assert_eq!(grid.n_blocks, 6);

        // Spot-check a few block IDs near the edges
        assert_eq!(grid.get_block_id(0, 0, 0), 0);
        assert_eq!(grid.get_block_id(16, 0, 0), 1); // second X block
        assert_eq!(grid.get_block_id(0, 16, 0), 2); // second Y block
        assert_eq!(grid.get_block_id(0, 32, 0), 4); // third Y block
        assert_eq!(grid.get_block_id(16, 32, 0), 5); // last block
    }

    #[test]
    fn test_grd_13_set_voxel_replaces_dummy_empty() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(5, 5, 5);

        // Force the empty dummy into the map
        grid.set_empty_block(bid);
        assert!(matches!(grid.get_block(bid), BlockRef::Empty));
        assert_eq!(grid.get_voxel(5, 5, 5), 0.0);

        // A write must allocate a real block and replace the dummy
        grid.set_voxel(5, 5, 5, 123.4);
        assert!(matches!(grid.get_block(bid), BlockRef::Allocated(_)));
        assert_eq!(grid.get_voxel(5, 5, 5), 123.4);

        // The dummy pointer itself must no longer be present
        assert_ne!(grid.blockmap[bid], Some(grid.empty_block));
    }

    #[test]
    fn test_grd_14_set_voxel_replaces_dummy_full() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(20, 20, 20);

        grid.set_full_block(bid);
        assert!(matches!(grid.get_block(bid), BlockRef::Full));
        assert_eq!(grid.get_voxel(20, 20, 20), 1.0);

        grid.set_voxel(20, 20, 20, 55.5);
        assert!(matches!(grid.get_block(bid), BlockRef::Allocated(_)));
        assert_eq!(grid.get_voxel(20, 20, 20), 55.5);
        assert_ne!(grid.blockmap[bid], Some(grid.full_block));
    }

    #[test]
    fn test_grd_15_dummy_block_identity_and_alignment() {
        let grid = setup_grid(16, 16, 16);

        // Dummy blocks are distinct from each other
        assert_ne!(grid.empty_block, grid.full_block);

        // They are 64-byte aligned (inherited from the pool)
        let empty_addr = grid.empty_block.as_ptr() as usize;
        let full_addr = grid.full_block.as_ptr() as usize;
        assert_eq!(empty_addr % 64, 0);
        assert_eq!(full_addr % 64, 0);

        // And they are not null
        assert_ne!(empty_addr, 0);
        assert_ne!(full_addr, 0);
    }

    #[test]
    fn test_grd_16_partial_block_coordinate_mapping() {
        // 20^3 voxels → 2^3 blocks, last block on each axis is partial
        let grid = setup_grid(20, 20, 20);

        // Global (19,19,19) lives in block (1,1,1) with local (3,3,3)
        assert_eq!(grid.get_block_id(19, 19, 19), 7);
        assert_eq!(
            SparseGrid::<f32, BSX, N>::get_voxel_id(19, 19, 19),
            3 | (3 << 4) | (3 << 8) // 3 + 48 + 768 = 819
        );

        // Global (16,0,0) → block (1,0,0), local (0,0,0)
        assert_eq!(grid.get_block_id(16, 0, 0), 1);
        assert_eq!(SparseGrid::<f32, BSX, N>::get_voxel_id(16, 0, 0), 0);
    }

    #[test]
    fn test_grd_17_blockmap_length_matches_n_blocks() {
        // Awkward sizes
        for &(sx, sy, sz) in &[(1, 1, 1), (16, 16, 16), (17, 1, 33), (100, 50, 7)] {
            let grid = setup_grid(sx, sy, sz);
            assert_eq!(
                grid.blockmap.len(),
                grid.n_blocks,
                "blockmap length mismatch for {}×{}×{}",
                sx,
                sy,
                sz
            );
            assert_eq!(grid.n_blocks, grid.nx * grid.ny * grid.nz);
        }
    }

    #[test]
    fn test_grd_18_set_empty_value_refills_dummy() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(5, 5, 5);
        grid.set_empty_block(bid);

        grid.set_empty_value(-1.0);

        assert_eq!(grid.get_voxel(5, 5, 5), -1.0); // scalar path
        unsafe {
            // dummy block data must match the sentinel (halo gather reads it)
            assert_eq!((*grid.empty_block.as_ptr()).data[0], -1.0);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_grd_19_set_empty_block_over_allocated_panics() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(5, 5, 5);
        grid.set_voxel(5, 5, 5, 42.0); // allocate a real block
        grid.set_empty_block(bid); // must panic: would leak the pool slot
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_grd_20_get_value_in_block_vid_oob_panics() {
        let mut grid = setup_grid(16, 16, 16);
        grid.set_voxel(0, 0, 0, 1.0); // ensure block 0 is allocated
        let _ = grid.get_value_in_block(0, N); // vid == N out of bounds
    }

    #[test]
    fn test_grd_21_shared_ref_concurrent_read() {
        let mut grid = setup_grid(32, 32, 32);
        grid.set_voxel(1, 2, 3, 7.0);
        grid.set_voxel(10, 11, 12, 8.0);

        let grid_ref = &grid;
        let (a, b) = rayon::join(
            || grid_ref.get_voxel(1, 2, 3),
            || grid_ref.get_voxel(10, 11, 12),
        );
        assert_eq!(a, 7.0);
        assert_eq!(b, 8.0);
    }

    #[test]
    fn test_grd_22_block_state_predicates() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(5, 5, 5);

        assert!(grid.is_empty_block(bid));
        assert!(!grid.is_value_block(bid) && !grid.is_full_block(bid));

        grid.set_full_block(bid);
        assert!(grid.is_full_block(bid));
        assert!(!grid.is_value_block(bid) && !grid.is_empty_block(bid));

        grid.set_voxel(5, 5, 5, 1.0); // replaces the full dummy with an allocated block
        assert!(grid.is_value_block(bid));
        assert!(!grid.is_empty_block(bid) && !grid.is_full_block(bid));
    }

    #[test]
    fn test_grd_23_block_coords_round_trip() {
        let grid = setup_grid(64, 48, 32); // nx=4 ny=3 nz=2
        for bid in [0, 1, 5, grid.n_blocks - 1] {
            let (bx, by, bz) = grid.get_block_coords_by_id(bid);
            assert_eq!(grid.get_block_id(bx * BSX, by * BSX, bz * BSX), bid);
        }
    }

    #[test]
    fn test_grd_24_block_data_accessors() {
        let mut grid = setup_grid(32, 32, 32);

        let bid = grid.get_block_id(16, 16, 16); // block (1,1,1)
        assert!(grid.get_block_data(bid).is_none());
        assert!(grid.get_block_data_mut(bid).is_none());

        grid.set_voxel(16, 16, 16, 7.0);
        assert_eq!(grid.get_block_data(bid).unwrap()[0], 7.0);

        grid.get_block_data_mut(bid).unwrap()[0] = 9.0;
        assert_eq!(grid.get_voxel(16, 16, 16), 9.0);

        let fid = grid.get_block_id(0, 0, 0);
        grid.set_full_block(fid);
        assert!(grid.get_block_data(fid).is_none());
    }
}

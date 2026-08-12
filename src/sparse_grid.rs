use std::{
    ptr::NonNull,
    simd::{f32x4, i32x4, i32x8},
    sync::Arc,
};

use crate::blockpool::{Block, BlockPool};

/// Sparse 3D spatial grid
pub struct SparseGrid<D, const BSX: usize, const N: usize>
where
    D: Copy + Default + Send + Sync,
{
    // Pointers, Collections, & Heap References (8-Byte Alignment)
    /// Sparse block mapping table (Block ID -> NonNull Block).
    pub blockmap: Vec<Option<NonNull<Block<D, BSX, N>>>>,

    /// Memory pool for spatial regions.
    pub block_pool: Arc<BlockPool<D, BSX, N>>,

    /// Memory pool for uniform/constant dummy blocks.
    pub const_block_pool: Arc<BlockPool<D, BSX, N>>,

    /// Shared uniform region dummy block handles.
    pub full_block: NonNull<Block<D, BSX, N>>,
    pub empty_block: NonNull<Block<D, BSX, N>>,
    pub invalid_block: NonNull<Block<D, BSX, N>>,

    /// Flat storage allocation for when `is_dense_grid` is active.
    pub dense_data: Option<Vec<D>>,

    /// Diagnostic label.
    pub name: String,

    // Scalar Dimensions & Sizes (usize)
    pub sx: usize,
    pub sy: usize,
    pub sz: usize,

    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    pub n_blocks: usize,
    pub sxy_act: usize,
    pub n_tot_act_voxels: usize,
    pub max_bord_off: usize,

    // Precomputed SIMD Registers (16 & 32-Byte Alignments)
    pub dom_min_f: f32x4,
    pub dom_max_f: f32x4,
    pub dom_min2_f: f32x4,
    pub dom_max2_f: f32x4,

    pub dom_min_i: i32x4,
    pub dom_max_i: i32x4,

    pub strides_lerp_0246: i32x4,
    pub strides_lerp_1357: i32x4,
    pub strides3_lerp_0246: i32x4,
    pub strides3_lerp_1357: i32x4,
    pub grid_strides_blk: i32x4,

    pub strides_lerp_02461357: i32x8,
    pub strides3_lerp_02461357: i32x8,
    pub strides3_lerp_01234567: i32x8,

    // Fixed Buffers & Identifiers
    pub neigh_vox_offs: [usize; 8],

    pub empty_value: D,
    pub invalid_value: D,
    pub full_value: D,

    pub data_gen_count: u32,
    pub is_dense_grid: bool,
}

// SparseGrid Tests
#[test]
fn test_grd_02_option_nonnull_niche_optimization() {
    use std::mem::size_of;
    // Option<NonNull<T>> MUST match the size of a raw pointer (8 bytes on 64-bit)
    assert_eq!(
        size_of::<Option<NonNull<Block<f32, 16, 4096>>>>(),
        size_of::<*mut u8>()
    );
}

#[test]
fn test_grd_03_dummy_block_pointers() {
    let _pool = Arc::new(BlockPool::<f32, 16, 4096>::new(4, 16));
    let const_pool = Arc::new(BlockPool::<f32, 16, 4096>::new(4, 16));

    let full_block = const_pool.alloc_block();
    let empty_block = const_pool.alloc_block();
    let invalid_block = const_pool.alloc_block();

    // Verify dummy blocks are distinct
    assert_ne!(full_block.as_ptr(), empty_block.as_ptr());
    assert_ne!(empty_block.as_ptr(), invalid_block.as_ptr());

    // Verify safe dereference
    unsafe {
        assert_eq!((*full_block.as_ptr()).flags, 0);
        assert_eq!((*empty_block.as_ptr()).flags, 0);
        assert_eq!((*invalid_block.as_ptr()).flags, 0);
    }
}

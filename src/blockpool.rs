use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    ptr::NonNull,
    sync::{
        Arc, Mutex,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
    thread,
};

#[repr(align(64))]
pub struct Block<D, const BSX: usize, const N: usize>
where
    D: Copy + Default + Send + Sync,
{
    /// Array of voxels.
    pub data: [D; N],

    /// Block status metadata.
    pub flags: u16,

    /// Block has to be 64-bytes.
    _pad: [u8; 62],
}

impl<D, const BSX: usize, const N: usize> Block<D, BSX, N>
where
    D: Copy + Default + Send + Sync,
{
    #[inline(always)]
    pub fn new() -> Self {
        assert!(BSX.is_power_of_two(), "Block BSX must be a power of two");
        assert_eq!(N, BSX * BSX * BSX, "Block size N must equal BSX^3");

        Self {
            data: [D::default(); N],
            flags: 0,
            _pad: [0; 62],
        }
    }

    #[inline(always)]
    pub fn get_voxel(&self, vx: usize, vy: usize, vz: usize) -> D {
        debug_assert!(
            vx < BSX && vy < BSX && vz < BSX,
            "Voxel indices out of bounds"
        );
        let bsx_log2 = BSX.trailing_zeros() as usize;
        let index = vx | (vy << bsx_log2) | (vz << (bsx_log2 * 2));

        // This is safe due to above assert
        unsafe { *self.data.get_unchecked(index) }
    }
}

/// Lock-free & monotonic allocator
pub struct BlockPool<D, const BSX: usize, const N: usize>
where
    D: Copy + Default + Send + Sync,
{
    /// Atomic monotonic pointer for allocation tracking.
    next_free: AtomicUsize,

    /// Number of blocks per pre-allocated segment.
    blocks_per_seg: usize,

    /// Log2 shift factor for segment routing.
    blocks_per_seg_log2: u32,

    /// Bitmask for offset indexing inside segment.
    blocks_per_seg_mask: usize,

    /// Lock-free atomic segment pointer table.
    segments: Vec<AtomicPtr<Block<D, BSX, N>>>,

    /// Fallback lock for OS heap allocation/extensions.
    extend_lock: Mutex<()>,
}

impl<D, const BSX: usize, const N: usize> BlockPool<D, BSX, N>
where
    D: Copy + Default + Send + Sync,
{
    pub fn new(max_segments: usize, blocks_per_seg: usize) -> Self {
        assert!(
            blocks_per_seg.is_power_of_two(),
            "blocks_per_seg must be a power of two"
        );
        assert!(max_segments > 0, "max_segments must be greater than zero");

        let mut segments = Vec::with_capacity(max_segments);
        for _ in 0..max_segments {
            segments.push(AtomicPtr::new(std::ptr::null_mut()));
        }

        Self {
            next_free: AtomicUsize::new(0),
            blocks_per_seg,
            blocks_per_seg_log2: blocks_per_seg.trailing_zeros(),
            blocks_per_seg_mask: blocks_per_seg - 1,
            segments,
            extend_lock: Mutex::new(()),
        }
    }

    #[inline]
    pub fn alloc_block(&self) -> NonNull<Block<D, BSX, N>> {
        let index = self.next_free.fetch_add(1, Ordering::Relaxed);
        let seg_idx = index >> self.blocks_per_seg_log2;
        let block_idx = index & self.blocks_per_seg_mask;

        assert!(
            seg_idx < self.segments.len(),
            "BlockPool memory limit exhausted"
        );

        let mut seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        if seg_ptr.is_null() {
            seg_ptr = self.extend_pool(seg_idx);
        }

        unsafe {
            let block_ptr = seg_ptr.add(block_idx);
            NonNull::new_unchecked(block_ptr)
        }
    }

    #[cold]
    fn extend_pool(&self, seg_idx: usize) -> *mut Block<D, BSX, N> {
        let _guard = self.extend_lock.lock().unwrap();

        let existing_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        if !existing_ptr.is_null() {
            return existing_ptr;
        }

        // Create a memory layout for `blocks_per_seg` contiguous blocks
        let layout = Layout::array::<Block<D, BSX, N>>(self.blocks_per_seg).unwrap();

        // Ask OS for zeroed memory
        let raw_ptr = unsafe { alloc_zeroed(layout) } as *mut Block<D, BSX, N>;

        if raw_ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        self.segments[seg_idx].store(raw_ptr, Ordering::Release);
        raw_ptr
    }

    #[inline]
    pub fn reset(&self) {
        self.next_free.store(0, Ordering::Release);
    }
}

impl<D: Copy + Default + Send + Sync, const BSX: usize, const N: usize> Drop
    for BlockPool<D, BSX, N>
{
    fn drop(&mut self) {
        // Reconstruct the memory layout used during extend_pool
        let layout = Layout::array::<Block<D, BSX, N>>(self.blocks_per_seg)
            .expect("Failed to create memory layout for deallocation");

        for atomic_ptr in &self.segments {
            let ptr = atomic_ptr.load(Ordering::Relaxed);
            if !ptr.is_null() {
                unsafe {
                    // Deallocate the memory
                    dealloc(ptr as *mut u8, layout);
                }
            }
        }
    }
}

// Block Tests
#[test]
fn test_blk_01_construction() {
    let block: Block<f32, 16, 4096> = Block::new();
    assert_eq!(block.flags, 0);
    for voxel in block.data.iter() {
        assert_eq!(*voxel, 0.0f32);
    }
}

#[test]
#[should_panic(expected = "Block BSX must be a power of two")]
fn test_blk_02_non_pow2_bsx_panics() {
    // Should panic because BSX = 15 is not a power of 2
    let _ = Block::<f32, 15, 3375>::new();
}

#[test]
#[should_panic(expected = "Block size N must equal BSX^3")]
fn test_blk_03_mismatched_n_panics() {
    // Should panic because N (2048) != 16^3 (4096)
    let _ = Block::<f32, 16, 2048>::new();
}

#[test]
fn test_blk_04_memory_alignment() {
    let block: Block<f32, 16, 4096> = Block::new();
    let ptr = &block as *const _ as usize;
    // Verify 64-byte alignment
    assert_eq!(ptr % 64, 0, "Block memory address must be 64-byte aligned");
}

#[test]
fn test_blk_05_memory_layout_size() {
    use std::mem::size_of;
    // N * size_of(f32) + 2 (flags) + 62 (_pad) = 4096 * 4 + 64 = 16448 bytes
    let expected_size = 4096 * size_of::<f32>() + 64;
    assert_eq!(size_of::<Block<f32, 16, 4096>>(), expected_size);
}

#[test]
fn test_blk_06_to_08_indexing_math() {
    let mut block: Block<f32, 16, 4096> = Block::new();

    // Populate using data array directly
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                let idx = x + y * 16 + z * 256;
                block.data[idx] = idx as f32;
            }
        }
    }

    // BLK-06: Lower bound
    assert_eq!(block.get_voxel(0, 0, 0), 0.0);

    // BLK-07: Upper bound
    assert_eq!(block.get_voxel(15, 15, 15), 4095.0);

    // BLK-08: Arbitrary indexing match
    let (x, y, z) = (4, 7, 11);
    let expected_idx = x + y * 16 + z * 256; // 4 + 112 + 2816 = 2932
    assert_eq!(block.get_voxel(x, y, z), expected_idx as f32);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Voxel indices out of bounds")]
fn test_blk_09_debug_oob_panics() {
    let block: Block<f32, 16, 4096> = Block::new();
    // Should trigger debug_assert in debug mode
    let _ = block.get_voxel(16, 0, 0);
}

// Blockpool Tests
#[test]
fn test_pol_01_initialization() {
    let pool: BlockPool<f32, 16, 4096> = BlockPool::new(16, 64);
    // BlockPool created cleanly with empty segments table
    let _ptr = pool.alloc_block();
}

#[test]
#[should_panic(expected = "blocks_per_seg must be a power of two")]
fn test_pol_02_non_pow2_blocks_per_seg_panics() {
    let _ = BlockPool::<f32, 16, 4096>::new(16, 60);
}

#[test]
#[should_panic(expected = "max_segments must be greater than zero")]
fn test_pol_03_zero_max_segments_panics() {
    let _ = BlockPool::<f32, 16, 4096>::new(0, 64);
}

#[test]
fn test_pol_04_single_allocation_alignment() {
    let pool: BlockPool<f32, 16, 4096> = BlockPool::new(4, 16);
    let block_ptr: NonNull<Block<f32, 16, 4096>> = pool.alloc_block();

    let addr = block_ptr.as_ptr() as usize;
    assert_ne!(addr, 0, "Allocated pointer must not be null");
    assert_eq!(addr % 64, 0, "Allocated block must be 64-byte aligned");
}

#[test]
fn test_pol_05_monotonic_segment_boundary_crossing() {
    // 2 segments max, 4 blocks per segment = 8 total blocks capacity
    let pool: BlockPool<f32, 16, 4096> = BlockPool::new(2, 4);
    let mut pointers = Vec::new();

    // Allocate 8 blocks to cross segment 0 -> segment 1
    for _ in 0..8 {
        pointers.push(pool.alloc_block());
    }

    // Ensure all 8 pointers are unique
    for i in 0..pointers.len() {
        for j in (i + 1)..pointers.len() {
            assert_ne!(
                pointers[i].as_ptr(),
                pointers[j].as_ptr(),
                "Pool handed out duplicate block pointers"
            );
        }
    }
}

#[test]
#[should_panic(expected = "BlockPool memory limit exhausted")]
fn test_pol_06_oom_panics_when_exhausted() {
    // Capacity = 1 segment * 2 blocks = 2 blocks total
    let pool: BlockPool<f32, 16, 4096> = BlockPool::new(1, 2);
    let _ = pool.alloc_block();
    let _ = pool.alloc_block();

    // 3rd allocation exhausts pool and panics
    let _ = pool.alloc_block();
}

#[test]
fn test_pol_07_allocation_reset() {
    let pool: BlockPool<f32, 16, 4096> = BlockPool::new(2, 4);

    let first_pass_ptr = pool.alloc_block();
    let _ = pool.alloc_block();

    // Instant O(1) reset
    pool.reset();

    // Re-allocate: Should reuse segment 0 and return the same first pointer
    let second_pass_ptr = pool.alloc_block();
    assert_eq!(
        first_pass_ptr.as_ptr(),
        second_pass_ptr.as_ptr(),
        "Reset failed to reuse segment 0 from beginning"
    );
}

#[test]
fn test_pol_08_segment_memory_contiguity() {
    use std::mem::size_of;
    let pool: BlockPool<f32, 16, 4096> = BlockPool::new(1, 4);

    let b0 = pool.alloc_block().as_ptr() as usize;
    let b1 = pool.alloc_block().as_ptr() as usize;

    let expected_delta = size_of::<Block<f32, 16, 4096>>();
    assert_eq!(
        b1 - b0,
        expected_delta,
        "Adjacent blocks in segment must be contiguous"
    );
}

// Concurrency Tests
#[test]
fn test_con_01_multithreaded_concurrent_allocations() {
    // Shared pool across threads
    let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(32, 128));
    let num_threads = 8;
    let allocs_per_thread = 100;

    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let pool_clone = Arc::clone(&pool);
        handles.push(thread::spawn(move || {
            let mut thread_ptrs = Vec::with_capacity(allocs_per_thread);
            for _ in 0..allocs_per_thread {
                let ptr = pool_clone.alloc_block();
                let addr = ptr.as_ptr() as usize;
                assert_eq!(addr % 64, 0, "Thread allocated unaligned block");
                thread_ptrs.push(addr);
            }
            thread_ptrs
        }));
    }

    let mut all_addrs = Vec::new();
    for handle in handles {
        let thread_addrs = handle.join().unwrap();
        all_addrs.extend(thread_addrs);
    }

    // Verify no race conditions caused duplicate pointer allocation
    all_addrs.sort_unstable();
    for i in 0..(all_addrs.len() - 1) {
        assert_ne!(
            all_addrs[i],
            all_addrs[i + 1],
            "Race condition detected: duplicate block allocated!"
        );
    }
}

#[test]
fn test_con_02_segment_extension_race_condition() {
    // Pool with 16 segments of 1 block each, forcing extensions on every alloc
    let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(16, 1));
    let mut handles = Vec::new();

    for _ in 0..16 {
        let pool_clone = Arc::clone(&pool);
        handles.push(thread::spawn(move || {
            let ptr = pool_clone.alloc_block();
            ptr.as_ptr() as usize
        }));
    }

    for handle in handles {
        let _ = handle.join().unwrap();
    }
}

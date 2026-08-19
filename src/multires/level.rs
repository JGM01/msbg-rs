//! Per-resolution-level data, stored behind a closed `Level` enum over the
//! supported block sizes.
//!
//! One shared sparse `BlockMap` maps block id → one `LevelBlock` allocation
//! (one pointer per active block), replacing the step-1..10 design of nine
//! separate `SparseGrid` channels each with its own dense blockmap and pool.
//! The C++ `MultiresSparseGrid` keeps ~40 named `SparseGrid*` fields with a
//! dense `_blockmap` each; a multigrid sweep there is 9 map lookups + 9
//! scattered allocations per block, here it is 1 lookup + offset math.
//!
//! Layout note: the per-block *metadata* (cell flags + fine-coarse distance) is
//! co-resident so the solver's mask reads share cache lines, but the *density*
//! is materialized lazily into its own contiguous pool — flags-only blocks
//! (e.g. the coarse sea during `set_refinement_map`) don't pay a 16 KiB density
//! write, and the density stream the halo/solver gather stays contiguous.

use crate::blockmap::BlockMap;
use crate::blockpool::Pool;
use crate::channel::{CellFlags, DistFineCoarse};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::multires::blockinfo::CELL_EMPTY_VAL;

/// The scalar float "density" payload of a level block (`CH_FLOAT_1`), held in
/// its own contiguous pool so density-only sweeps stream without the metadata
/// stride.
#[repr(C, align(64))]
pub struct DensityBlock<const N: usize> {
    pub data: [f32; N],
}

/// The 2-phase MAC face channels (C++ `face_area` / `face_coeff`, three
/// directional components each), allocated lazily by [`LevelData::ensure_faces`]
/// so the single-phase steps (7..19) don't pay their ~6× f32 footprint per block.
#[repr(C, align(64))]
pub struct FaceBlock<const BSX: usize, const N: usize> {
    pub face_area: [[f32; N]; 3],
    pub face_coeff: [[f32; N]; 3],
}

impl<const BSX: usize, const N: usize> FaceBlock<BSX, N> {
    fn new() -> Self {
        Self {
            face_area: [[0.0; N]; 3],
            face_coeff: [[0.0; N]; 3],
        }
    }
}

/// One block of a resolution level: the per-block metadata (cell flags +
/// fine-coarse distance) co-resident, plus lazily-materialized density and
/// 2-phase face payloads.
#[repr(C, align(64))]
pub struct LevelBlock<const BSX: usize, const N: usize> {
    pub cell_flags: [CellFlags; N],
    pub dist_fine_coarse: [DistFineCoarse; N],
    /// Lazily-materialized density (step 7+); `None` reads as empty (0.0).
    pub density: Option<NonNull<DensityBlock<N>>>,
    /// Lazily-materialized 2-phase face payload (step 20).
    pub faces: Option<NonNull<FaceBlock<BSX, N>>>,
}

/// A shared, sendable handle to a [`LevelBlock`]. `NonNull` is deliberately
/// neither `Send` nor `Sync` (it is a raw pointer); concurrent access to a
/// level block is safe by construction — blocks are materialized in serial
/// phases and only *read* in the parallel sweeps (`density_ptr`/halo), while
/// `ensure_*` mutate via `&mut self`. Mirrors `sparse_grid::BlockPtr`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct LevelBlockPtr<const BSX: usize, const N: usize>(pub NonNull<LevelBlock<BSX, N>>);

impl<const BSX: usize, const N: usize> PartialEq for LevelBlockPtr<BSX, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

unsafe impl<const BSX: usize, const N: usize> Send for LevelBlockPtr<BSX, N> {}
unsafe impl<const BSX: usize, const N: usize> Sync for LevelBlockPtr<BSX, N> {}

/// Concrete data of one resolution level.
pub struct LevelData<const BSX: usize, const N: usize> {
    pub sx: usize,
    pub sy: usize,
    pub sz: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nxy: usize,
    pub n_blocks: usize,

    blocks: BlockMap<LevelBlockPtr<BSX, N>>,
    pool: Arc<Pool<LevelBlock<BSX, N>>>,
    density_pool: Arc<Pool<DensityBlock<N>>>,
    faces_pool: Arc<Pool<FaceBlock<BSX, N>>>,
    empty_density: NonNull<DensityBlock<N>>,
}

// All fields are Send/Sync except the read-only `empty_density` raw pointer
// (a zeroed f32 array, never mutated after construction); mutations go through
// `&mut self`, so sharing `&LevelData` across rayon workers is sound.
unsafe impl<const BSX: usize, const N: usize> Send for LevelData<BSX, N> {}
unsafe impl<const BSX: usize, const N: usize> Sync for LevelData<BSX, N> {}

impl<const BSX: usize, const N: usize> LevelData<BSX, N> {
    pub fn new(sx: usize, sy: usize, sz: usize) -> Self {
        let nx = sx / BSX;
        let ny = sy / BSX;
        let nz = sz / BSX;
        let n_blocks = nx * ny * nz;
        let max_segments = n_blocks / 4096 + 2;
        let pool = Arc::new(Pool::<LevelBlock<BSX, N>>::new(max_segments, 4096));
        let density_pool = Arc::new(Pool::<DensityBlock<N>>::new(max_segments, 4096));
        let faces_pool = Arc::new(Pool::<FaceBlock<BSX, N>>::new(max_segments, 4096));

        let empty_density = density_pool.alloc_block();
        unsafe {
            empty_density.as_ptr().write(DensityBlock { data: [0.0; N] });
        }

        Self {
            sx,
            sy,
            sz,
            nx,
            ny,
            nz,
            nxy: nx * ny,
            n_blocks,
            blocks: BlockMap::new(),
            pool,
            density_pool,
            faces_pool,
            empty_density,
        }
    }

    /// Allocate the per-block metadata for `bid` if absent (empty cell flags,
    /// zero fine-coarse distance, no density/faces). Returns without effect if
    /// present.
    #[inline(always)]
    pub fn ensure_block(&mut self, bid: usize) {
        debug_assert!(bid < self.n_blocks);
        if self.blocks.get(bid).is_some() {
            return;
        }
        let ptr = self.pool.alloc_block();
        unsafe {
            ptr.as_ptr().write(LevelBlock {
                cell_flags: [CellFlags(CELL_EMPTY_VAL); N],
                dist_fine_coarse: [DistFineCoarse(0); N],
                density: None,
                faces: None,
            });
        }
        self.blocks.insert(bid, LevelBlockPtr(ptr));
    }

    /// Materialize (once) the density payload for `bid` (zeroed).
    pub fn ensure_density(&mut self, bid: usize) {
        self.ensure_block(bid);
        let block = self.blocks.get(bid).expect("ensure_block materialized a block");
        if unsafe { (*block.0.as_ptr()).density }.is_none() {
            let d = self.density_pool.alloc_block();
            unsafe { d.as_ptr().write(DensityBlock { data: [0.0; N] }) };
            unsafe { (*block.0.as_ptr()).density = Some(d) };
        }
    }

    /// Materialize (once) the 2-phase face payload for `bid` and return it.
    pub fn ensure_faces(&mut self, bid: usize) -> NonNull<FaceBlock<BSX, N>> {
        debug_assert!(bid < self.n_blocks);
        self.ensure_block(bid);
        let block = self.blocks.get(bid).expect("ensure_block materialized a block");
        let faces = unsafe { (*block.0.as_ptr()).faces };
        match faces {
            Some(f) => f,
            None => {
                let f = self.faces_pool.alloc_block();
                unsafe { f.as_ptr().write(FaceBlock::new()) };
                unsafe { (*block.0.as_ptr()).faces = Some(f) };
                f
            }
        }
    }

    /// Raw density pointer for `bid`, resolving absent/not-yet-materialized to
    /// the zeroed empty dummy so reads are unconditional (the multires halo).
    #[inline(always)]
    pub fn density_ptr(&self, bid: usize) -> *const f32 {
        match self.blocks.get(bid) {
            Some(p) => match unsafe { (*p.0.as_ptr()).density } {
                Some(d) => unsafe { (*d.as_ptr()).data.as_ptr() },
                None => self.empty_density_ptr(),
            },
            None => self.empty_density_ptr(),
        }
    }

    /// Density pointer of the empty dummy (out-of-domain halo reads).
    #[inline(always)]
    pub fn empty_density_ptr(&self) -> *const f32 {
        unsafe { (*self.empty_density.as_ptr()).data.as_ptr() }
    }

    /// Raw writable density pointer for `bid`, materializing the density payload.
    #[inline(always)]
    pub fn density_ptr_mut(&mut self, bid: usize) -> *mut f32 {
        self.ensure_density(bid);
        let block = self.blocks.get(bid).expect("ensure_density materialized a block");
        unsafe { (*(*block.0.as_ptr()).density.expect("density materialized").as_ptr()).data.as_mut_ptr() }
    }

    /// Raw writable cell-flags pointer for an allocated block, else null.
    #[inline(always)]
    pub fn cell_flags_ptr_mut(&self, bid: usize) -> *mut CellFlags {
        match self.blocks.get(bid) {
            Some(p) => unsafe { (*p.0.as_ptr()).cell_flags.as_mut_ptr() },
            _ => std::ptr::null_mut(),
        }
    }

    /// Raw writable fine-coarse-distance pointer for an allocated block, else null.
    #[inline(always)]
    pub fn dfc_ptr_mut(&self, bid: usize) -> *mut DistFineCoarse {
        match self.blocks.get(bid) {
            Some(p) => unsafe { (*p.0.as_ptr()).dist_fine_coarse.as_mut_ptr() },
            _ => std::ptr::null_mut(),
        }
    }

    /// Iterate the ids of every materialized block (map order).
    pub fn active_block_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.blocks.iter().map(|(bid, _)| bid)
    }

    /// True if `bid` is a materialized (allocated) block.
    #[inline(always)]
    pub fn is_value_block(&self, bid: usize) -> bool {
        self.blocks.get(bid).is_some()
    }
}

macro_rules! level_enum {
    ($( $variant:ident => $bsx:literal, $n:literal; )+) => {
        /// A resolution level, type-erased over the block size. Block size halves
        /// each level (same block count across levels), so the concrete variant is
        /// determined by `blockSize0 >> level`.
        pub enum Level {
            $(
                $variant(LevelData<$bsx, $n>),
            )+
        }

        impl Level {
            pub fn bsx(&self) -> usize {
                match self { $( Self::$variant(_) => $bsx, )+ }
            }

            pub fn sx(&self) -> usize { match self { $( Self::$variant(l) => l.sx, )+ } }
            pub fn sy(&self) -> usize { match self { $( Self::$variant(l) => l.sy, )+ } }
            pub fn sz(&self) -> usize { match self { $( Self::$variant(l) => l.sz, )+ } }
            pub fn nx(&self) -> usize { match self { $( Self::$variant(l) => l.nx, )+ } }
            pub fn ny(&self) -> usize { match self { $( Self::$variant(l) => l.ny, )+ } }
            pub fn nz(&self) -> usize { match self { $( Self::$variant(l) => l.nz, )+ } }
        }
    };
}

level_enum! {
    B32 => 32, 32768;
    B16 => 16, 4096;
    B8  => 8,  512;
    B4  => 4,  64;
    B2  => 2,  8;
    B1  => 1,  1;
}

/// Construct the level matching `block_size`. Panics if `block_size` is not a
/// supported power of two (the `Level` enum's closed set).
pub fn level_from_block_size(block_size: usize, sx: usize, sy: usize, sz: usize) -> Level {
    match block_size {
        32 => Level::B32(LevelData::new(sx, sy, sz)),
        16 => Level::B16(LevelData::new(sx, sy, sz)),
        8 => Level::B8(LevelData::new(sx, sy, sz)),
        4 => Level::B4(LevelData::new(sx, sy, sz)),
        2 => Level::B2(LevelData::new(sx, sy, sz)),
        1 => Level::B1(LevelData::new(sx, sy, sz)),
        other => panic!("unsupported block size {other} (must be a power of two <= 32)"),
    }
}

/// Dispatch over the `Level` variants, binding the concrete `LevelData` to `$l`.
macro_rules! level_dispatch {
    (($self:expr), |$l:ident| $body:block) => {
        match $self {
            Level::B32($l) => $body,
            Level::B16($l) => $body,
            Level::B8($l) => $body,
            Level::B4($l) => $body,
            Level::B2($l) => $body,
            Level::B1($l) => $body,
        }
    };
}

use crate::multires::blockinfo::{BlockFlags, BlockInfoStore};
use crate::multires::refinement::{
    init_cell_flags, init_dist_fine_coarse, propagate_dist_fine_coarse_full, BlockGridDims,
    RefinementMap,
};

impl Level {
    pub fn init_cell_flags(
        &mut self,
        level: usize,
        map: &RefinementMap,
        dims: &BlockGridDims,
        store: &BlockInfoStore,
        block_flags: Option<&[BlockFlags]>,
    ) {
        level_dispatch!((self), |l| {
            init_cell_flags(l, level, map, dims, store, block_flags)
        })
    }

    pub fn init_dist_fine_coarse(
        &mut self,
        dtrans_res: usize,
        map: &RefinementMap,
        dims: &BlockGridDims,
        store: &BlockInfoStore,
    ) {
        level_dispatch!((self), |l| {
            init_dist_fine_coarse(l, dtrans_res, map, dims, store)
        })
    }

    pub fn propagate_dist_fine_coarse_full(
        &mut self,
        level: usize,
        dtrans_res: usize,
        dims: &BlockGridDims,
        store: &BlockInfoStore,
    ) {
        level_dispatch!((self), |l| {
            propagate_dist_fine_coarse_full(l, level, dtrans_res, dims, store)
        })
    }
}

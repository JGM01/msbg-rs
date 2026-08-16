//! Per-resolution-level data: a typed channel bundle plus the directional face
//! channels, stored behind a closed `Level` enum over the supported block sizes.

use crate::channel::{CellFlags, DistFineCoarse};
use crate::sparse_grid::SparseGrid;
use std::sync::Arc;

use crate::blockpool::BlockPool;

/// Concrete data of one resolution level. The scalar data channel is `f32`
/// (C++ `CH_FLOAT_1`, the float "density" channel the demo smooths); the cell
/// flags and fine-coarse distance channels are typed to their C++ element types.
pub struct LevelData<const BSX: usize, const N: usize> {
    pub sx: usize,
    pub sy: usize,
    pub sz: usize,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    pub density: SparseGrid<f32, BSX, N>,
    pub cell_flags: SparseGrid<CellFlags, BSX, N>,
    pub dist_fine_coarse: SparseGrid<DistFineCoarse, BSX, N>,
    /// Directional face channels (`face_area` / `face_coeff`, C++ MAC-staggered).
    pub face_area: [SparseGrid<f32, BSX, N>; 3],
    pub face_coeff: [SparseGrid<f32, BSX, N>; 3],
}

impl<const BSX: usize, const N: usize> LevelData<BSX, N> {
    pub fn new(name: &str, sx: usize, sy: usize, sz: usize) -> Self {
        let nx = sx / BSX;
        let ny = sy / BSX;
        let nz = sz / BSX;
        let n_blocks = nx * ny * nz;
        let max_segments = n_blocks / 4096 + 2;
        let density_pool = Arc::new(BlockPool::<f32, BSX, N>::new(max_segments, 4096));
        let flags_pool = Arc::new(BlockPool::<CellFlags, BSX, N>::new(max_segments, 4096));
        let dfc_pool = Arc::new(BlockPool::<DistFineCoarse, BSX, N>::new(max_segments, 4096));

        let density = SparseGrid::new(format!("{name}:DENS"), sx, sy, sz, 0.0, 1.0, density_pool);
        let cell_flags = SparseGrid::new(
            format!("{name}:FLG"),
            sx,
            sy,
            sz,
            CellFlags(crate::multires::blockinfo::CELL_EMPTY_VAL),
            CellFlags(0),
            flags_pool,
        );
        let dist_fine_coarse = SparseGrid::new(
            format!("{name}:FCDIST"),
            sx,
            sy,
            sz,
            DistFineCoarse(0),
            DistFineCoarse(0),
            dfc_pool,
        );

        let mut face_area = [(); 3].map(|_| None);
        let mut face_coeff = [(); 3].map(|_| None);
        for dir in 0..3 {
            let pool = Arc::new(BlockPool::<f32, BSX, N>::new(max_segments, 4096));
            let mut g = SparseGrid::new(format!("{name}:FAREA{dir}"), sx, sy, sz, 0.0, 1.0, pool);
            g.set_full_value(1.0);
            face_area[dir] = Some(g);

            let pool = Arc::new(BlockPool::<f32, BSX, N>::new(max_segments, 4096));
            let g = SparseGrid::new(format!("{name}:FCOEFF{dir}"), sx, sy, sz, 0.0, 0.0, pool);
            face_coeff[dir] = Some(g);
        }

        Self {
            sx,
            sy,
            sz,
            nx,
            ny,
            nz,
            density,
            cell_flags,
            dist_fine_coarse,
            face_area: face_area.map(|g| g.unwrap()),
            face_coeff: face_coeff.map(|g| g.unwrap()),
        }
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
pub fn level_from_block_size(name: &str, block_size: usize, sx: usize, sy: usize, sz: usize) -> Level {
    match block_size {
        32 => Level::B32(LevelData::new(name, sx, sy, sz)),
        16 => Level::B16(LevelData::new(name, sx, sy, sz)),
        8 => Level::B8(LevelData::new(name, sx, sy, sz)),
        4 => Level::B4(LevelData::new(name, sx, sy, sz)),
        2 => Level::B2(LevelData::new(name, sx, sy, sz)),
        1 => Level::B1(LevelData::new(name, sx, sy, sz)),
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

use crate::multires::refinement::{
    init_cell_flags, init_dist_fine_coarse, propagate_dist_fine_coarse_full, BlockGridDims,
    RefinementMap,
};
use crate::multires::blockinfo::{BlockFlags, BlockInfoStore};

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
            init_cell_flags(&mut l.cell_flags, level, map, dims, store, block_flags)
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
            init_dist_fine_coarse(&mut l.dist_fine_coarse, dtrans_res, map, dims, store)
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
            propagate_dist_fine_coarse_full(&mut l.dist_fine_coarse, level, dtrans_res, dims, store)
        })
    }
}

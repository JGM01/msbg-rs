//! Compact matrix-free pressure operators on adaptive MSBG cell layouts.
//!
//! A [`PressurePattern`] owns immutable topology while [`PressureOperator`]
//! owns face coefficients and the inverse diagonal. Solver vectors are dense
//! over active cells and carry the pattern identity that created them.
//!
//! ```
//! use msbg_rs::multires::{MultiresGrid, RefinementMap};
//! use msbg_rs::pressure::{PressureBoundary, PressureOperator, PressurePatternBuilder};
//!
//! let mut grid = MultiresGrid::create("p", 16, 16, 16, 16, 2, 2);
//! let mut refinement = RefinementMap::uniform(1, 0);
//! grid.set_refinement_map(&mut refinement);
//! let pattern = PressurePatternBuilder::new(
//!     &grid,
//!     &[0],
//!     [PressureBoundary::ClosedNeumann; 6],
//! ).build().unwrap();
//! let op = PressureOperator::constant(pattern.clone(), 1.0).unwrap();
//! let x = pattern.vector_with_value(1.0);
//! let mut y = pattern.vector();
//! op.apply(&x, &mut y).unwrap();
//! assert!(y.as_slice().iter().all(|v| *v == 0.0));
//! ```

use rayon::prelude::*;
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut, Range};
use std::ptr::NonNull;
use std::simd::num::SimdFloat;
use std::simd::{Simd, StdFloat};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::fluid::MacGrid;
use crate::math::simd::LANES;
use crate::math::stencil::{load, store_nt};
use crate::multires::MultiresGrid;
use crate::multires::sort::morton3;

static NEXT_PATTERN_ID: AtomicU64 = AtomicU64::new(1);
const NONE: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressureBoundary {
    ClosedNeumann,
    OpenDirichletZero,
    Periodic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DomainFace {
    XMin = 0,
    XMax = 1,
    YMin = 2,
    YMax = 3,
    ZMin = 4,
    ZMax = 5,
}

impl DomainFace {
    #[inline(always)]
    #[allow(dead_code)] // used by boundary_links construction / metrics
    fn axis(self) -> usize {
        self as usize / 2
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn is_min(self) -> bool {
        self as usize & 1 == 0
    }
}

const FACES: [DomainFace; 6] = [
    DomainFace::XMin,
    DomainFace::XMax,
    DomainFace::YMin,
    DomainFace::YMax,
    DomainFace::ZMin,
    DomainFace::ZMax,
];

#[derive(Clone, Debug, PartialEq)]
pub enum PressureError {
    EmptyActiveSet,
    InvalidActiveBlock(usize),
    DuplicateActiveBlock(usize),
    InvalidBoundaryPair {
        axis: usize,
    },
    InvalidSpacing(f32),
    InvalidActivityMask {
        expected: usize,
        actual: usize,
    },
    ZeroDiagonal {
        cell: usize,
    },
    InvalidCoefficient {
        axis: usize,
        cell: usize,
        value: f32,
    },
    CoefficientLength {
        axis: usize,
        expected: usize,
        actual: usize,
    },
    PatternMismatch,
    InvalidRelaxation {
        omega: f32,
        local_sweeps: usize,
    },
    InvalidPhaseParameters,
    UnsupportedMacLayout,
    InvalidTimeStep(f32),
    CorrectionLength {
        expected: usize,
        actual: usize,
    },
    PhaseLabelLength {
        expected: usize,
        actual: usize,
    },
}

impl Display for PressureError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyActiveSet => write!(f, "pressure active set is empty"),
            Self::InvalidActiveBlock(b) => write!(f, "active block {b} is outside the grid"),
            Self::DuplicateActiveBlock(b) => write!(f, "active block {b} appears more than once"),
            Self::InvalidBoundaryPair { axis } => {
                write!(f, "periodic boundaries must be paired on axis {axis}")
            }
            Self::InvalidSpacing(h) => {
                write!(f, "finest spacing must be finite and positive, got {h}")
            }
            Self::InvalidActivityMask { expected, actual } => {
                write!(f, "activity mask has {actual} bits, expected {expected}")
            }
            Self::ZeroDiagonal { cell } => {
                write!(f, "active pressure cell {cell} has a zero diagonal")
            }
            Self::InvalidCoefficient { axis, cell, value } => write!(
                f,
                "invalid beta coefficient {value} at axis {axis}, cell {cell}"
            ),
            Self::CoefficientLength {
                axis,
                expected,
                actual,
            } => write!(
                f,
                "axis {axis} has {actual} coefficients, expected {expected}"
            ),
            Self::PatternMismatch => write!(f, "pressure vector belongs to another pattern"),
            Self::InvalidRelaxation {
                omega,
                local_sweeps,
            } => write!(
                f,
                "omega must be in [0,1] and local_sweeps positive, got {omega} and {local_sweeps}"
            ),
            Self::InvalidPhaseParameters => write!(
                f,
                "phase densities and normalization parameters are invalid"
            ),
            Self::UnsupportedMacLayout => write!(
                f,
                "the MAC adapter requires a uniform finest-level pressure pattern"
            ),
            Self::InvalidTimeStep(dt) => {
                write!(f, "time step must be finite and positive, got {dt}")
            }
            Self::CorrectionLength { expected, actual } => write!(
                f,
                "divergence correction has {actual} values, expected {expected}"
            ),
            Self::PhaseLabelLength { expected, actual } => {
                write!(f, "phase labels have {actual} bits, expected {expected}")
            }
        }
    }
}

impl Error for PressureError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellActivityMask {
    bits: Vec<u64>,
    len: usize,
}

impl CellActivityMask {
    pub fn from_bools(active: &[bool]) -> Self {
        let mut bits = vec![0u64; active.len().div_ceil(64)];
        for (i, enabled) in active.iter().copied().enumerate() {
            if enabled {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self {
            bits,
            len: active.len(),
        }
    }

    pub fn from_words(bits: Vec<u64>, len: usize) -> Result<Self, PressureError> {
        if bits.len() != len.div_ceil(64) {
            return Err(PressureError::InvalidActivityMask {
                expected: len.div_ceil(64),
                actual: bits.len(),
            });
        }
        Ok(Self { bits, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    fn get(&self, i: usize) -> bool {
        self.bits[i / 64] >> (i % 64) & 1 != 0
    }
}

#[derive(Clone, Debug)]
pub struct PressureBlock {
    pub bid: usize,
    pub level: u8,
    pub block_size: usize,
    pub cell_start: usize,
    pub cell_end: usize,
    pub outer_color: usize,
    pub coords: [usize; 3],
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    block_rank: usize,
    local: [usize; 3],
    fine_origin: [usize; 3],
    level: u8,
}

#[derive(Clone, Copy, Debug)]
struct EdgePlan {
    a: usize,
    b: usize,
    source: usize,
    axis: usize,
    /// Direction of the edge from `a`: 0 = negative, 1 = positive. For
    /// in-block edges it is always + (the neighbor sits at local+1); for
    /// cross-block edges it is `-` when `a` is on the block's min face and `+`
    /// on the max face (this is what keeps the periodic-wrap direction right —
    /// origin comparison would mislabel the wrap as negative).
    dir: u8,
    geometric_scale: f32,
    transition: bool,
}

#[derive(Clone, Copy, Debug)]
struct BoundaryPlan {
    cell: usize,
    source: usize,
    axis: usize,
    face: DomainFace,
    geometric_scale: f32,
}

pub struct PressurePattern {
    id: u64,
    finest_spacing: f32,
    finest_block_size: usize,
    finest_dims: [usize; 3],
    boundaries: [PressureBoundary; 6],
    blocks: Vec<PressureBlock>,
    block_rank_by_bid: Vec<usize>,
    local_to_cell: Vec<Vec<usize>>,
    cells: Vec<Cell>,
    edges: Vec<EdgePlan>,
    boundary_links: Vec<BoundaryPlan>,
    #[cfg(feature = "cpp-parity")]
    row_edges: Vec<Vec<(usize, usize)>>,
    block_colors: [Vec<usize>; 2],
    transition_links: usize,
    /// Per-block halo face tables: for the six faces of a level-0 block, the
    /// `bs²` neighbor cell ids in `[-x,+x,-y,+y,-z,+z]` order (NONE = domain
    /// wall / inactive neighbor => the halo reads 0). Coarse blocks get an
    /// empty table (their cells mix 4 fine neighbors per face and use the CSR
    /// path instead).
    block_faces: Vec<Vec<usize>>,
    /// True for blocks whose cells are processed by the scalar CSR path
    /// (coarse blocks and any finest block whose size isn't a SIMD multiple).
    block_use_csr: Vec<bool>,
}

impl PressurePattern {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn blocks(&self) -> &[PressureBlock] {
        &self.blocks
    }

    pub fn boundaries(&self) -> &[PressureBoundary; 6] {
        &self.boundaries
    }

    pub fn transition_link_count(&self) -> usize {
        self.transition_links
    }

    pub fn link_count(&self) -> usize {
        self.edges.len()
    }

    pub fn boundary_link_count(&self, face: DomainFace) -> usize {
        self.boundary_links
            .iter()
            .filter(|link| link.face == face)
            .count()
    }

    pub fn finest_spacing(&self) -> f32 {
        self.finest_spacing
    }

    pub fn vector(&self) -> PressureVector {
        PressureVector {
            pattern_id: self.id,
            values: AlignedValues::new(self.len(), 0.0),
        }
    }

    pub fn vector_with_value(&self, value: f32) -> PressureVector {
        PressureVector {
            pattern_id: self.id,
            values: AlignedValues::new(self.len(), value),
        }
    }

    pub fn cell_index(&self, bid: usize, x: usize, y: usize, z: usize) -> Option<usize> {
        let rank = *self.block_rank_by_bid.get(bid)?;
        if rank == NONE {
            return None;
        }
        let bs = self.blocks[rank].block_size;
        if x >= bs || y >= bs || z >= bs {
            return None;
        }
        let cell = self.local_to_cell[rank][x + bs * (y + bs * z)];
        (cell != NONE).then_some(cell)
    }

    pub fn cell_volume(&self, cell: usize) -> f32 {
        let h = self.finest_spacing * (1usize << self.cells[cell].level) as f32;
        h * h * h
    }

    fn check_vector(&self, vector: &PressureVector) -> Result<(), PressureError> {
        if vector.pattern_id != self.id || vector.values.len() != self.len() {
            Err(PressureError::PatternMismatch)
        } else {
            Ok(())
        }
    }
}

pub struct PressurePatternBuilder<'a> {
    grid: &'a MultiresGrid,
    active_blocks: &'a [usize],
    boundaries: [PressureBoundary; 6],
    activity: Option<&'a CellActivityMask>,
    finest_spacing: f32,
}

impl<'a> PressurePatternBuilder<'a> {
    pub fn new(
        grid: &'a MultiresGrid,
        active_blocks: &'a [usize],
        boundaries: [PressureBoundary; 6],
    ) -> Self {
        Self {
            grid,
            active_blocks,
            boundaries,
            activity: None,
            finest_spacing: 1.0,
        }
    }

    pub fn activity_mask(mut self, activity: &'a CellActivityMask) -> Self {
        self.activity = Some(activity);
        self
    }

    pub fn finest_spacing(mut self, spacing: f32) -> Self {
        self.finest_spacing = spacing;
        self
    }

    pub fn build(self) -> Result<Arc<PressurePattern>, PressureError> {
        if self.active_blocks.is_empty() {
            return Err(PressureError::EmptyActiveSet);
        }
        if !self.finest_spacing.is_finite() || self.finest_spacing <= 0.0 {
            return Err(PressureError::InvalidSpacing(self.finest_spacing));
        }
        for axis in 0..3 {
            let lo = self.boundaries[2 * axis] == PressureBoundary::Periodic;
            let hi = self.boundaries[2 * axis + 1] == PressureBoundary::Periodic;
            if lo != hi {
                return Err(PressureError::InvalidBoundaryPair { axis });
            }
        }

        let mut seen = HashSet::with_capacity(self.active_blocks.len());
        let mut mask_base = HashMap::with_capacity(self.active_blocks.len());
        let mut expected_mask_len = 0usize;
        for &bid in self.active_blocks {
            if bid >= self.grid.dims.n_blocks {
                return Err(PressureError::InvalidActiveBlock(bid));
            }
            if !seen.insert(bid) {
                return Err(PressureError::DuplicateActiveBlock(bid));
            }
            mask_base.insert(bid, expected_mask_len);
            let bs = self.grid.bsx0 >> self.grid.block_level(bid);
            expected_mask_len += bs * bs * bs;
        }
        if let Some(mask) = self.activity
            && mask.len() != expected_mask_len
        {
            return Err(PressureError::InvalidActivityMask {
                expected: expected_mask_len,
                actual: mask.len(),
            });
        }

        let mut bids = self.active_blocks.to_vec();
        bids.sort_unstable_by_key(|&bid| {
            let (bx, by, bz) = self.grid.dims.coords(bid);
            (((bx + by + bz) & 1), morton3(bx, by, bz))
        });

        let mut blocks = Vec::with_capacity(bids.len());
        let mut block_rank_by_bid = vec![NONE; self.grid.dims.n_blocks];
        let mut local_to_cell = Vec::with_capacity(bids.len());
        let mut cells = Vec::new();
        let mut block_colors: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::new());

        for bid in bids {
            let rank = blocks.len();
            block_rank_by_bid[bid] = rank;
            let level = self.grid.block_level(bid);
            let bs = self.grid.bsx0 >> level;
            let (bx, by, bz) = self.grid.dims.coords(bid);
            let start = cells.len();
            let mut map = vec![NONE; bs * bs * bs];
            let base = mask_base[&bid];
            let scale = 1usize << level;
            for z in 0..bs {
                for y in 0..bs {
                    for x in 0..bs {
                        let local_index = x + bs * (y + bs * z);
                        if self.activity.is_some_and(|m| !m.get(base + local_index)) {
                            continue;
                        }
                        let cell = cells.len();
                        map[local_index] = cell;
                        cells.push(Cell {
                            block_rank: rank,
                            local: [x, y, z],
                            fine_origin: [
                                bx * self.grid.bsx0 + x * scale,
                                by * self.grid.bsx0 + y * scale,
                                bz * self.grid.bsx0 + z * scale,
                            ],
                            level,
                        });
                    }
                }
            }
            let color = (bx + by + bz) & 1;
            block_colors[color].push(rank);
            blocks.push(PressureBlock {
                bid,
                level,
                block_size: bs,
                cell_start: start,
                cell_end: cells.len(),
                outer_color: color,
                coords: [bx, by, bz],
            });
            local_to_cell.push(map);
        }
        if cells.is_empty() {
            return Err(PressureError::EmptyActiveSet);
        }

        let mut edges = Vec::new();
        let mut boundary_links = Vec::new();
        for (cell_index, &cell) in cells.iter().enumerate() {
            let block = &blocks[cell.block_rank];
            for axis in 0..3 {
                if cell.local[axis] + 1 < block.block_size {
                    let mut next = cell.local;
                    next[axis] += 1;
                    let other = local_cell(&local_to_cell[cell.block_rank], block.block_size, next);
                    if other != NONE {
                        edges.push(EdgePlan {
                            a: cell_index,
                            b: other,
                            source: cell_index,
                            axis,
                            dir: 1,
                            geometric_scale: self.finest_spacing * (1usize << cell.level) as f32,
                            transition: false,
                        });
                    }
                    continue;
                }

                let mut neighbor_coords = block.coords;
                let at_domain_max = neighbor_coords[axis] + 1
                    == [self.grid.dims.nx, self.grid.dims.ny, self.grid.dims.nz][axis];
                if at_domain_max {
                    match self.boundaries[2 * axis + 1] {
                        PressureBoundary::ClosedNeumann => continue,
                        PressureBoundary::OpenDirichletZero => {
                            boundary_links.push(BoundaryPlan {
                                cell: cell_index,
                                source: cell_index,
                                axis,
                                face: FACES[2 * axis + 1],
                                geometric_scale: self.finest_spacing
                                    * (1usize << cell.level) as f32,
                            });
                            continue;
                        }
                        PressureBoundary::Periodic => neighbor_coords[axis] = 0,
                    }
                } else {
                    neighbor_coords[axis] += 1;
                }
                let neighbor_bid = neighbor_coords[0]
                    + neighbor_coords[1] * self.grid.dims.nx
                    + neighbor_coords[2] * self.grid.dims.nxy;
                let neighbor_rank = block_rank_by_bid[neighbor_bid];
                if neighbor_rank == NONE {
                    continue;
                }
                append_cross_block_edges(
                    &blocks,
                    &local_to_cell,
                    &mut edges,
                    (cell_index, cell.block_rank, cell.local),
                    neighbor_rank,
                    axis,
                    self.finest_spacing,
                );
            }

            for axis in 0..3 {
                if cell.local[axis] != 0 || block.coords[axis] != 0 {
                    continue;
                }
                if self.boundaries[2 * axis] == PressureBoundary::OpenDirichletZero {
                    boundary_links.push(BoundaryPlan {
                        cell: cell_index,
                        source: cell_index,
                        axis,
                        face: FACES[2 * axis],
                        geometric_scale: self.finest_spacing * (1usize << cell.level) as f32,
                    });
                }
            }
        }

        #[cfg(feature = "cpp-parity")]
        let mut row_edges = {
            let mut re = vec![Vec::new(); cells.len()];
            for (edge_index, edge) in edges.iter().enumerate() {
                if edge.a == edge.b {
                    continue;
                }
                re[edge.a].push((edge.b, edge_index));
                re[edge.b].push((edge.a, edge_index));
            }
            re
        };
        #[cfg(not(feature = "cpp-parity"))]
        let _row_edges: Vec<Vec<(usize, usize)>> = Vec::new();
        let transition_links = edges.iter().filter(|e| e.transition).count();

        // Halo face tables for the block-SIMD kernel: each level-0 block's six
        // faces resolve to the cell ids read into its halo. Coarse blocks
        // (level > 0) have up to four fine neighbors per face and fall back to
        // the scalar CSR path. The grid enforces 2:1 block refinement, so a
        // level-0 face cell's neighbor is either a level-0 cell (exact origin
        // match) or a level-1 cell (origin rounded down to a 2-aligned cell).
        let mut origin_to_cell = HashMap::with_capacity(cells.len());
        for (cell, c) in cells.iter().enumerate() {
            origin_to_cell.insert(pack_origin(c.fine_origin), cell);
        }
        let domain = [self.grid.sx0, self.grid.sy0, self.grid.sz0];
        let mut block_faces = Vec::with_capacity(blocks.len());
        let mut block_use_csr = Vec::with_capacity(blocks.len());
        for (rank, block) in blocks.iter().enumerate() {
            if block.level > 0 {
                block_faces.push(Vec::new());
                block_use_csr.push(true);
                continue;
            }
            let bs = block.block_size;
            let scale = 1usize << block.level;
            let origin = [
                block.coords[0] * self.grid.bsx0,
                block.coords[1] * self.grid.bsx0,
                block.coords[2] * self.grid.bsx0,
            ];
            let mut faces = Vec::with_capacity(6 * bs * bs);
            for face in 0..6 {
                let axis = face / 2;
                let is_max = face & 1 == 1;
                let (a, b) = match axis {
                    0 => (1usize, 2usize),
                    1 => (0usize, 2usize),
                    _ => (0usize, 1usize),
                };
                for t in 0..bs * bs {
                    let ta = t % bs;
                    let tb = t / bs;
                    let mut coords = origin;
                    coords[a] += ta * scale;
                    coords[b] += tb * scale;
                    if is_max {
                        coords[axis] += bs * scale;
                    } else {
                        coords[axis] = coords[axis].saturating_sub(scale);
                    }
                    let on_wall = if is_max {
                        coords[axis] >= domain[axis]
                    } else {
                        coords[axis] == 0
                    };
                    if on_wall {
                        if self.boundaries[face] == PressureBoundary::Periodic {
                            // wrap to the opposite domain face
                            coords[axis] = if is_max {
                                coords[axis] - domain[axis]
                            } else {
                                domain[axis] - scale
                            };
                        } else {
                            faces.push(NONE);
                            continue;
                        }
                    }
                    // exact level-0 neighbor, else the containing level-1 cell.
                    let neighbor = origin_to_cell
                        .get(&pack_origin(coords))
                        .copied()
                        .or_else(|| {
                            let rounded = [coords[0] & !1, coords[1] & !1, coords[2] & !1];
                            (rounded != coords)
                                .then(|| origin_to_cell.get(&pack_origin(rounded)).copied())
                                .flatten()
                        })
                        .unwrap_or(NONE);
                    faces.push(neighbor);
                }
            }
            block_faces.push(faces);
            // An activity mask compacts a block's cells to its active subset,
            // so the dense fixed-offset SIMD kernel is only valid for fully
            // active level-0 blocks.
            let dense = !local_to_cell[rank].contains(&NONE);
            block_use_csr.push(!dense);
        }

        Ok(Arc::new(PressurePattern {
            id: NEXT_PATTERN_ID.fetch_add(1, Ordering::Relaxed),
            finest_spacing: self.finest_spacing,
            finest_block_size: self.grid.bsx0,
            finest_dims: [self.grid.sx0, self.grid.sy0, self.grid.sz0],
            boundaries: self.boundaries,
            blocks,
            block_rank_by_bid,
            local_to_cell,
            cells,
            edges,
            boundary_links,
            #[cfg(feature = "cpp-parity")]
            row_edges,
            block_colors,
            transition_links,
            block_faces,
            block_use_csr,
        }))
    }
}

/// Pack a cell's fine-origin `[x, y, z]` into a `u64` for the face-table
/// lookup. Fine coords are ≤ 32768 (paper domain), so 16 bits per axis suffice
/// with room to spare.
#[inline(always)]
fn pack_origin(o: [usize; 3]) -> u64 {
    (o[0] as u64) | ((o[1] as u64) << 21) | ((o[2] as u64) << 42)
}

#[inline(always)]
fn local_cell(map: &[usize], bs: usize, local: [usize; 3]) -> usize {
    map[local[0] + bs * (local[1] + bs * local[2])]
}

fn append_cross_block_edges(
    blocks: &[PressureBlock],
    local_to_cell: &[Vec<usize>],
    edges: &mut Vec<EdgePlan>,
    source: (usize, usize, [usize; 3]),
    neighbor_rank: usize,
    axis: usize,
    finest_spacing: f32,
) {
    let (cell_index, source_rank, source_local) = source;
    let source_block = &blocks[source_rank];
    let neighbor_block = &blocks[neighbor_rank];
    let delta = neighbor_block.level as i32 - source_block.level as i32;
    debug_assert!(
        delta.abs() <= 1,
        "pressure topology requires 2:1 block refinement"
    );

    if delta == 0 {
        let mut local = source_local;
        local[axis] = 0;
        let other = local_cell(
            &local_to_cell[neighbor_rank],
            neighbor_block.block_size,
            local,
        );
        if other != NONE {
            edges.push(EdgePlan {
                a: cell_index,
                b: other,
                source: cell_index,
                axis,
                dir: if source_local[axis] == 0 { 0 } else { 1 },
                geometric_scale: finest_spacing * (1usize << source_block.level) as f32,
                transition: false,
            });
        }
        return;
    }

    let fine_level = source_block.level.min(neighbor_block.level);
    let geometric_scale = (2.0 / 3.0) * finest_spacing * (1usize << fine_level) as f32;
    if delta == 1 {
        let mut local = source_local;
        for (tangent, coordinate) in local.iter_mut().enumerate() {
            if tangent != axis {
                *coordinate /= 2;
            }
        }
        local[axis] = 0;
        let other = local_cell(
            &local_to_cell[neighbor_rank],
            neighbor_block.block_size,
            local,
        );
        if other != NONE {
            edges.push(EdgePlan {
                a: cell_index,
                b: other,
                source: cell_index,
                axis,
                dir: if source_local[axis] == 0 { 0 } else { 1 },
                geometric_scale,
                transition: true,
            });
        }
    } else {
        let tangents: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
        for o1 in 0..2 {
            for o2 in 0..2 {
                let mut local = [0usize; 3];
                local[axis] = 0;
                local[tangents[0]] = 2 * source_local[tangents[0]] + o1;
                local[tangents[1]] = 2 * source_local[tangents[1]] + o2;
                let other = local_cell(
                    &local_to_cell[neighbor_rank],
                    neighbor_block.block_size,
                    local,
                );
                if other != NONE {
                    edges.push(EdgePlan {
                        a: cell_index,
                        b: other,
                        source: cell_index,
                        axis,
                        dir: if source_local[axis] == 0 { 0 } else { 1 },
                        geometric_scale,
                        transition: true,
                    });
                }
            }
        }
    }
}

struct AlignedValues {
    ptr: NonNull<f32>,
    len: usize,
}

impl AlignedValues {
    fn layout(len: usize) -> Layout {
        Layout::array::<f32>(len)
            .unwrap()
            .align_to(64)
            .unwrap()
            .pad_to_align()
    }

    fn new(len: usize, value: f32) -> Self {
        debug_assert!(len > 0);
        let layout = Self::layout(len);
        let raw = unsafe { alloc(layout) } as *mut f32;
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        debug_assert_eq!(ptr.as_ptr() as usize % 64, 0);
        for i in 0..len {
            debug_assert!(i < len);
            unsafe { ptr.as_ptr().add(i).write(value) };
        }
        Self { ptr, len }
    }

    fn as_slice(&self) -> &[f32] {
        debug_assert_eq!(self.ptr.as_ptr() as usize % 64, 0);
        debug_assert!(self.len > 0);
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        debug_assert_eq!(self.ptr.as_ptr() as usize % 64, 0);
        debug_assert!(self.len > 0);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Clone for AlignedValues {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.len, 0.0);
        output.as_mut_slice().copy_from_slice(self.as_slice());
        output
    }
}

impl std::fmt::Debug for AlignedValues {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl Deref for AlignedValues {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedValues {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for AlignedValues {
    fn drop(&mut self) {
        debug_assert_eq!(self.ptr.as_ptr() as usize % 64, 0);
        debug_assert!(self.len > 0);
        unsafe { dealloc(self.ptr.as_ptr().cast(), Self::layout(self.len)) };
    }
}

// The allocation is exclusively owned and contains only `f32`; mutation still
// requires `&mut AlignedValues`.
unsafe impl Send for AlignedValues {}
unsafe impl Sync for AlignedValues {}

#[derive(Clone, Debug)]
pub struct PressureVector {
    pattern_id: u64,
    values: AlignedValues,
}

impl PressureVector {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.values.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.values.as_mut_slice()
    }

    pub fn fill(&mut self, value: f32) {
        self.values.fill(value);
    }

    pub fn dot(&self, other: &Self) -> Result<f64, PressureError> {
        if self.pattern_id != other.pattern_id || self.values.len() != other.values.len() {
            return Err(PressureError::PatternMismatch);
        }
        Ok(self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum())
    }
}

#[derive(Clone, Debug)]
pub struct FaceCoefficients {
    pub right: [Vec<f32>; 3],
}

impl FaceCoefficients {
    pub fn constant(len: usize, beta: f32) -> Self {
        Self {
            right: std::array::from_fn(|_| vec![beta; len]),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoefficientFormat {
    #[default]
    F32,
    ExperimentalBf16,
}

#[derive(Clone, Debug)]
enum CoefficientStorage {
    F32([Vec<f32>; 3]),
    Bf16([Vec<u16>; 3]),
}

impl CoefficientStorage {
    #[inline(always)]
    fn get(&self, axis: usize, cell: usize) -> f32 {
        match self {
            Self::F32(v) => v[axis][cell],
            Self::Bf16(v) => f32::from_bits((v[axis][cell] as u32) << 16),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RelaxParams {
    omega: f32,
    local_sweeps: usize,
}

impl RelaxParams {
    pub fn new(omega: f32, local_sweeps: usize) -> Result<Self, PressureError> {
        if !omega.is_finite() || !(0.0..=1.0).contains(&omega) || local_sweeps == 0 {
            return Err(PressureError::InvalidRelaxation {
                omega,
                local_sweeps,
            });
        }
        Ok(Self {
            omega,
            local_sweeps,
        })
    }

    pub fn omega(self) -> f32 {
        self.omega
    }

    pub fn local_sweeps(self) -> usize {
        self.local_sweeps
    }
}

impl Default for RelaxParams {
    fn default() -> Self {
        Self {
            omega: 6.0 / 7.0,
            local_sweeps: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelaxMetrics {
    pub block_max: Vec<f32>,
    pub boundary_max: [Vec<f32>; 6],
}

impl RelaxMetrics {
    pub fn new(pattern: &PressurePattern) -> Self {
        Self {
            block_max: vec![0.0; pattern.blocks.len()],
            boundary_max: std::array::from_fn(|_| vec![0.0; pattern.blocks.len()]),
        }
    }

    fn reset(&mut self, pattern: &PressurePattern) -> Result<(), PressureError> {
        if self.block_max.len() != pattern.blocks.len()
            || self
                .boundary_max
                .iter()
                .any(|v| v.len() != pattern.blocks.len())
        {
            return Err(PressureError::PatternMismatch);
        }
        self.block_max.fill(0.0);
        for values in &mut self.boundary_max {
            values.fill(0.0);
        }
        Ok(())
    }
}

pub struct PressureOperator {
    pattern: Arc<PressurePattern>,
    coefficients: CoefficientStorage,
    edge_coefficients: Vec<f32>,
    boundary_coefficients: Vec<f32>,
    diagonal: Vec<f32>,
    inverse_diagonal: Vec<f32>,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    coeff_vals: Vec<f32>,
    /// Direction-major coefficients staged contiguously per SIMD block. This
    /// has the same 6-float/cell footprint as six domain arrays, but keeps a
    /// worker's complete block working set adjacent in cache.
    block_coefficients: Vec<Vec<f32>>,
    block_constant_coeff: Vec<bool>,
    /// Constant edge coefficient when every edge shares it (the C++ constant
    /// path skips coefficient loads entirely and just sums the 6 neighbors).
    const_coeff: Option<f32>,
    /// Per-thread halo scratch for the block-SIMD kernel (Sync: each slot is
    /// exclusively accessed by one rayon worker via `current_thread_index`).
    scratch: ScratchPool,
    format: CoefficientFormat,
}

impl PressureOperator {
    pub fn constant(pattern: Arc<PressurePattern>, beta: f32) -> Result<Self, PressureError> {
        let coefficients = FaceCoefficients::constant(pattern.len(), beta);
        Self::new(pattern, coefficients)
    }

    pub fn new(
        pattern: Arc<PressurePattern>,
        coefficients: FaceCoefficients,
    ) -> Result<Self, PressureError> {
        Self::with_format(pattern, coefficients, CoefficientFormat::F32)
    }

    pub fn with_format(
        pattern: Arc<PressurePattern>,
        coefficients: FaceCoefficients,
        format: CoefficientFormat,
    ) -> Result<Self, PressureError> {
        for axis in 0..3 {
            if coefficients.right[axis].len() != pattern.len() {
                return Err(PressureError::CoefficientLength {
                    axis,
                    expected: pattern.len(),
                    actual: coefficients.right[axis].len(),
                });
            }
            for (cell, &value) in coefficients.right[axis].iter().enumerate() {
                if !value.is_finite() || value < 0.0 {
                    return Err(PressureError::InvalidCoefficient { axis, cell, value });
                }
            }
        }
        // Constant-coefficient fast path: when every edge shares one value, the
        // SIMD kernel can skip the six per-cell coefficient loads and sum the
        // halo neighbors directly (the C++ constant `blockHasFaceCoeffs` branch
        // does the same). The edge value folds the per-level geometric scale.
        let const_coeff = {
            let first = coefficients.right[0].first().copied();
            first.filter(|&v| {
                coefficients
                    .right
                    .iter()
                    .all(|axis| axis.iter().all(|&c| c == v))
            })
        };
        let storage = match format {
            CoefficientFormat::F32 => CoefficientStorage::F32(coefficients.right),
            CoefficientFormat::ExperimentalBf16 => CoefficientStorage::Bf16(
                coefficients
                    .right
                    .map(|axis| axis.into_iter().map(f32_to_bf16_rne).collect()),
            ),
        };
        let edge_coefficients: Vec<f32> = pattern
            .edges
            .iter()
            .map(|edge| storage.get(edge.axis, edge.source) * edge.geometric_scale)
            .collect();
        let boundary_coefficients: Vec<f32> = pattern
            .boundary_links
            .iter()
            .map(|link| storage.get(link.axis, link.source) * link.geometric_scale)
            .collect();
        let mut diagonal = vec![0.0f32; pattern.len()];
        for (edge, &coefficient) in pattern.edges.iter().zip(&edge_coefficients) {
            if edge.a != edge.b {
                diagonal[edge.a] += coefficient;
                diagonal[edge.b] += coefficient;
            }
        }
        for (link, &coefficient) in pattern.boundary_links.iter().zip(&boundary_coefficients) {
            diagonal[link.cell] += coefficient;
        }
        let mut inverse_diagonal = Vec::with_capacity(diagonal.len());
        for (cell, &value) in diagonal.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(PressureError::ZeroDiagonal { cell });
            }
            inverse_diagonal.push(value.recip());
        }
        // Flat CSR (neighbors + signed coefficients in edge order, one
        // contiguous buffer) so the matvec/relax hot loops avoid per-row Vec
        // indirection; `row_edges` is retained for `relax_cpp_parity`/oracle.
        let mut counts = vec![0usize; pattern.len()];
        for edge in &pattern.edges {
            if edge.a != edge.b {
                counts[edge.a] += 1;
                counts[edge.b] += 1;
            }
        }
        let mut row_offsets = Vec::with_capacity(pattern.len() + 1);
        let mut total = 0usize;
        row_offsets.push(0);
        for &c in &counts {
            total += c;
            row_offsets.push(total);
        }
        let mut col_indices = vec![0usize; total];
        let mut coeff_vals = vec![0.0f32; total];
        let mut cursor = row_offsets[..pattern.len()].to_vec();
        for (edge_index, edge) in pattern.edges.iter().enumerate() {
            if edge.a == edge.b {
                continue;
            }
            let c = edge_coefficients[edge_index];
            col_indices[cursor[edge.a]] = edge.b;
            coeff_vals[cursor[edge.a]] = c;
            cursor[edge.a] += 1;
            col_indices[cursor[edge.b]] = edge.a;
            coeff_vals[cursor[edge.b]] = c;
            cursor[edge.b] += 1;
        }
        // Per-cell 6-direction coefficients for the block-SIMD stencil. Each
        // (cell, axis, side) has at most one edge (2:1 structure). The wall
        // side of a boundary cell has no edge and stays 0; the halo ghost is 0
        // there, so the 0 coefficient contributes nothing regardless of BC.
        let mut coeff_dir: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; pattern.len()]);
        for (edge_index, edge) in pattern.edges.iter().enumerate() {
            let c = edge_coefficients[edge_index];
            coeff_dir[edge.axis * 2 + edge.dir as usize][edge.a] = c;
            coeff_dir[edge.axis * 2 + (1 - edge.dir as usize)][edge.b] = c;
        }
        let block_coefficients: Vec<Vec<f32>> = pattern
            .blocks
            .iter()
            .enumerate()
            .map(|(rank, block)| {
                if pattern.block_use_csr[rank] {
                    return Vec::new();
                }
                let n = block.cell_end - block.cell_start;
                let mut staged = Vec::with_capacity(6 * n);
                for dir in &coeff_dir {
                    staged.extend_from_slice(&dir[block.cell_start..block.cell_end]);
                }
                staged
            })
            .collect();
        let expected_constant = const_coeff.map(|beta| beta * pattern.finest_spacing);
        let block_constant_coeff = block_coefficients
            .iter()
            .map(|weights| {
                expected_constant.is_some_and(|expected| {
                    !weights.is_empty()
                        && weights
                            .iter()
                            .all(|&weight| weight == 0.0 || weight == expected)
                })
            })
            .collect();
        // Match the C++ pressure kernel's padded halo rows: the interior starts
        // one SIMD vector into every row and y/z neighbors retain alignment.
        let bs = pattern.finest_block_size;
        let halo_hsx = (bs + 2 * LANES).next_multiple_of(LANES);
        let halo_len = halo_hsx * (bs + 2) * (bs + 2);
        let scratch = ScratchPool::new(rayon::current_num_threads(), halo_len);
        Ok(Self {
            pattern,
            coefficients: storage,
            edge_coefficients,
            boundary_coefficients,
            diagonal,
            inverse_diagonal,
            row_offsets,
            col_indices,
            coeff_vals,
            block_coefficients,
            block_constant_coeff,
            const_coeff,
            scratch,
            format,
        })
    }

    pub fn pattern(&self) -> &Arc<PressurePattern> {
        &self.pattern
    }

    pub fn coefficient_format(&self) -> CoefficientFormat {
        self.format
    }

    pub fn inverse_diagonal(&self) -> &[f32] {
        &self.inverse_diagonal
    }

    pub fn diagonal(&self) -> &[f32] {
        &self.diagonal
    }

    pub fn face_coefficient(&self, axis: usize, cell: usize) -> f32 {
        self.coefficients.get(axis, cell)
    }

    pub fn apply(&self, x: &PressureVector, dst: &mut PressureVector) -> Result<(), PressureError> {
        self.pattern.check_vector(x)?;
        self.pattern.check_vector(dst)?;
        if std::ptr::eq(x, dst) {
            return Err(PressureError::PatternMismatch);
        }
        let xv = x.values.as_slice();
        let out = dst.values.as_mut_ptr() as usize;
        let blocks = &self.pattern.blocks;
        let use_csr = &self.pattern.block_use_csr;
        let scratch = &self.scratch;
        let block_constant_coeff = &self.block_constant_coeff;
        let (offsets, cols, vals, diagonal, block_coefficients) = (
            &self.row_offsets,
            &self.col_indices,
            &self.coeff_vals,
            &self.diagonal,
            &self.block_coefficients,
        );
        let const_coeff = self.const_coeff;
        let finest_spacing = self.pattern.finest_spacing;
        let right_coefficients = self.direct_right_coefficients();
        let faces = &self.pattern.block_faces;
        (0..blocks.len()).into_par_iter().for_each(|rank| {
            let block = &blocks[rank];
            if use_csr[rank] || block.block_size % LANES != 0 {
                apply_csr_range(
                    offsets,
                    cols,
                    vals,
                    diagonal,
                    xv,
                    None,
                    out as *mut f32,
                    block.cell_start..block.cell_end,
                    None,
                    None,
                );
            } else {
                // SAFETY: each rayon worker owns its scratch slot exclusively.
                let halo = unsafe { scratch.get_mut() };
                unsafe {
                    let correct_constant_faces =
                        const_coeff.is_some() && !block_constant_coeff[rank];
                    apply_block_simd::<LANES>(
                        blocks,
                        faces,
                        diagonal,
                        block_coefficients,
                        const_coeff,
                        correct_constant_faces,
                        finest_spacing,
                        right_coefficients,
                        xv,
                        None,
                        out as *mut f32,
                        rank,
                        halo,
                        None,
                        None,
                    );
                }
            }
        });
        Ok(())
    }

    pub fn apply_and_dot(
        &self,
        x: &PressureVector,
        dst: &mut PressureVector,
    ) -> Result<f64, PressureError> {
        self.pattern.check_vector(x)?;
        self.pattern.check_vector(dst)?;
        if std::ptr::eq(x, dst) {
            return Err(PressureError::PatternMismatch);
        }
        let xv = x.values.as_slice();
        let out = dst.values.as_mut_ptr() as usize;
        let blocks = &self.pattern.blocks;
        let use_csr = &self.pattern.block_use_csr;
        let scratch = &self.scratch;
        let block_constant_coeff = &self.block_constant_coeff;
        let (offsets, cols, vals, diagonal, block_coefficients) = (
            &self.row_offsets,
            &self.col_indices,
            &self.coeff_vals,
            &self.diagonal,
            &self.block_coefficients,
        );
        let const_coeff = self.const_coeff;
        let finest_spacing = self.pattern.finest_spacing;
        let right_coefficients = self.direct_right_coefficients();
        let faces = &self.pattern.block_faces;
        // The dot is fused into the kernels (one pass, like the C++
        // LAPL_WITH_DOTPROD): each worker accumulates its blocks' partial sums.
        let total: f64 = (0..blocks.len())
            .into_par_iter()
            .fold(
                || 0.0f64,
                |mut dot, rank| {
                    let block = &blocks[rank];
                    if use_csr[rank] || block.block_size % LANES != 0 {
                        apply_csr_range(
                            offsets,
                            cols,
                            vals,
                            diagonal,
                            xv,
                            None,
                            out as *mut f32,
                            block.cell_start..block.cell_end,
                            Some(&mut dot),
                            None,
                        );
                    } else {
                        // SAFETY: each rayon worker owns its scratch slot.
                        let halo = unsafe { scratch.get_mut() };
                        unsafe {
                            let correct_constant_faces =
                                const_coeff.is_some() && !block_constant_coeff[rank];
                            apply_block_simd::<LANES>(
                                blocks,
                                faces,
                                diagonal,
                                block_coefficients,
                                const_coeff,
                                correct_constant_faces,
                                finest_spacing,
                                right_coefficients,
                                xv,
                                None,
                                out as *mut f32,
                                rank,
                                halo,
                                Some(&mut dot),
                                None,
                            );
                        }
                    }
                    dot
                },
            )
            .sum();
        Ok(total)
    }

    pub fn residual(
        &self,
        rhs: &PressureVector,
        x: &PressureVector,
        dst: &mut PressureVector,
    ) -> Result<f64, PressureError> {
        self.pattern.check_vector(rhs)?;
        self.pattern.check_vector(x)?;
        self.pattern.check_vector(dst)?;
        let (offsets, cols, vals, diagonal, block_coefficients) = (
            &self.row_offsets,
            &self.col_indices,
            &self.coeff_vals,
            &self.diagonal,
            &self.block_coefficients,
        );
        let blocks = &self.pattern.blocks;
        let use_csr = &self.pattern.block_use_csr;
        let faces = &self.pattern.block_faces;
        let scratch = &self.scratch;
        let block_constant_coeff = &self.block_constant_coeff;
        let const_coeff = self.const_coeff;
        let finest_spacing = self.pattern.finest_spacing;
        let right_coefficients = self.direct_right_coefficients();
        let xv = x.values.as_slice();
        let rv = rhs.values.as_slice();
        let out = dst.values.as_mut_ptr() as usize;
        // The `rv - matvec(x)` pass AND the norm2 reduction are fused into the
        // block kernels (one pass, like the C++ LAPL_CALC_RESIDUAL).
        let norm2: f64 = (0..blocks.len())
            .into_par_iter()
            .fold(
                || 0.0f64,
                |mut norm, rank| {
                    let block = &blocks[rank];
                    if use_csr[rank] || block.block_size % LANES != 0 {
                        apply_csr_range(
                            offsets,
                            cols,
                            vals,
                            diagonal,
                            xv,
                            Some(rv),
                            out as *mut f32,
                            block.cell_start..block.cell_end,
                            None,
                            Some(&mut norm),
                        );
                    } else {
                        // SAFETY: each rayon worker owns its scratch slot.
                        let halo = unsafe { scratch.get_mut() };
                        unsafe {
                            let correct_constant_faces =
                                const_coeff.is_some() && !block_constant_coeff[rank];
                            apply_block_simd::<LANES>(
                                blocks,
                                faces,
                                diagonal,
                                block_coefficients,
                                const_coeff,
                                correct_constant_faces,
                                finest_spacing,
                                right_coefficients,
                                xv,
                                Some(rv),
                                out as *mut f32,
                                rank,
                                halo,
                                None,
                                Some(&mut norm),
                            );
                        }
                    }
                    norm
                },
            )
            .sum();
        Ok(norm2)
    }

    pub fn relax(
        &self,
        pressure: &mut PressureVector,
        rhs: &PressureVector,
        params: RelaxParams,
        mut metrics: Option<&mut RelaxMetrics>,
    ) -> Result<(), PressureError> {
        self.pattern.check_vector(pressure)?;
        self.pattern.check_vector(rhs)?;
        let params = RelaxParams::new(params.omega, params.local_sweeps)?;
        if let Some(output) = metrics.as_deref_mut() {
            output.reset(&self.pattern)?;
        }
        let word_width = native_word_width(self.pattern.finest_block_size);
        // The shared pressure buffer is captured as a plain address (usize is
        // Sync); each worker re-derives the per-block disjoint &mut slice and
        // the shared read pointer inside the closure.
        let base = pressure.values.as_mut_ptr() as usize;
        let p_slice = pressure.values.as_slice();
        let rv = rhs.values.as_slice();
        let blocks = &self.pattern.blocks;
        let use_csr = &self.pattern.block_use_csr;
        let faces = &self.pattern.block_faces;
        let local_to_cell = &self.pattern.local_to_cell;
        let offsets = &self.row_offsets;
        let cols = &self.col_indices;
        let vals = &self.coeff_vals;
        let diagonal = &self.diagonal;
        let inverse_diagonal = &self.inverse_diagonal;
        let block_coefficients = &self.block_coefficients;
        let block_constant_coeff = &self.block_constant_coeff;
        let const_coeff = self.const_coeff;
        let finest_spacing = self.pattern.finest_spacing;
        let right_coefficients = self.direct_right_coefficients();
        let scratch = &self.scratch;
        let track_metrics = metrics.is_some();

        let process_block = |rank: usize| {
            let block = &blocks[rank];
            let stats = if use_csr[rank] || block.block_size % LANES != 0 {
                relax_csr_block(
                    base,
                    rv,
                    offsets,
                    cols,
                    vals,
                    &local_to_cell[rank],
                    diagonal,
                    inverse_diagonal,
                    block,
                    params,
                    word_width,
                )
            } else {
                // SAFETY: each rayon worker owns its scratch slot. The
                // non-metrics specialization omits all residual/max arithmetic.
                let halo = unsafe { scratch.get_mut() };
                unsafe {
                    let correct_constant_faces =
                        const_coeff.is_some() && !block_constant_coeff[rank];
                    if track_metrics {
                        relax_block_simd::<LANES, true>(
                            p_slice,
                            base,
                            rv,
                            blocks,
                            faces,
                            diagonal,
                            inverse_diagonal,
                            block_coefficients,
                            const_coeff,
                            correct_constant_faces,
                            finest_spacing,
                            right_coefficients,
                            rank,
                            halo,
                            params,
                            word_width,
                        )
                    } else {
                        relax_block_simd::<LANES, false>(
                            p_slice,
                            base,
                            rv,
                            blocks,
                            faces,
                            diagonal,
                            inverse_diagonal,
                            block_coefficients,
                            const_coeff,
                            correct_constant_faces,
                            finest_spacing,
                            right_coefficients,
                            rank,
                            halo,
                            params,
                            word_width,
                        )
                    }
                }
            };
            (rank, stats.0, stats.1)
        };

        // The 8-color schedule makes the blocks of one outer color halo-disjoint,
        // so their block-local GS sweeps are embarrassingly parallel and the
        // parallel result is bit-identical to the serial sweep. Cross-block
        // neighbor reads touch the opposite color, which is quiescent in this
        // wave. The per-block cell ranges partition `pressure`, so each worker
        // owns a disjoint &mut slice. `metrics` maxima are merged after each wave.
        if metrics.is_none() && rayon::current_num_threads() > 1 {
            let barrier = std::sync::Barrier::new(rayon::current_num_threads());
            rayon::broadcast(|ctx| {
                for outer_color in 0..2 {
                    let ranks = &self.pattern.block_colors[outer_color];
                    let start = ranks.len() * ctx.index() / ctx.num_threads();
                    let end = ranks.len() * (ctx.index() + 1) / ctx.num_threads();
                    for &rank in &ranks[start..end] {
                        process_block(rank);
                    }
                    if outer_color == 0 {
                        barrier.wait();
                    }
                }
            });
            return Ok(());
        }
        for outer_color in 0..2 {
            if let Some(output) = metrics.as_deref_mut() {
                let per_block: Vec<(usize, f32, [f32; 6])> = self.pattern.block_colors[outer_color]
                    .par_iter()
                    .map(|&rank| process_block(rank))
                    .collect();
                for (rank, bmax, face_max) in per_block {
                    output.block_max[rank] = bmax;
                    for face in 0..6 {
                        output.boundary_max[face][rank] = face_max[face];
                    }
                }
            } else {
                self.pattern.block_colors[outer_color]
                    .par_iter()
                    .for_each(|&rank| {
                        process_block(rank);
                    });
            }
        }
        Ok(())
    }

    /// Raw F32 right-face arrays can supply both directions on uniform finest
    /// grids: the negative weight is the preceding cell's positive weight.
    /// Transition edges carry a different geometric factor and use the staged
    /// six-direction representation instead.
    #[inline]
    fn direct_right_coefficients(&self) -> Option<[usize; 3]> {
        if self.pattern.transition_links != 0 {
            return None;
        }
        match &self.coefficients {
            CoefficientStorage::F32(axis) => {
                Some(std::array::from_fn(|d| axis[d].as_ptr() as usize))
            }
            CoefficientStorage::Bf16(_) => None,
        }
    }

    #[cfg(feature = "cpp-parity")]
    pub fn relax_cpp_parity(
        &self,
        pressure: &mut PressureVector,
        rhs: &PressureVector,
        params: RelaxParams,
    ) -> Result<(), PressureError> {
        self.pattern.check_vector(pressure)?;
        self.pattern.check_vector(rhs)?;
        let params = RelaxParams::new(params.omega, params.local_sweeps)?;
        let word_width = if cfg!(target_arch = "x86_64") {
            8.min(self.pattern.finest_block_size)
        } else {
            native_word_width(self.pattern.finest_block_size)
        };
        for outer_color in [1usize, 0] {
            for &rank in &self.pattern.block_colors[outer_color] {
                let block = &self.pattern.blocks[rank];
                for sweep in 0..params.local_sweeps {
                    let reverse = sweep & 1 == 0;
                    let final_snapshot = (sweep + 1 == params.local_sweeps)
                        .then(|| pressure.values.as_slice().to_vec());
                    let zs: Vec<usize> = if reverse {
                        (0..block.block_size).rev().collect()
                    } else {
                        (0..block.block_size).collect()
                    };
                    let ys = zs.clone();
                    let words: Vec<usize> = if reverse {
                        (0..block.block_size.div_ceil(word_width)).rev().collect()
                    } else {
                        (0..block.block_size.div_ceil(word_width)).collect()
                    };
                    for z in zs {
                        for &y in &ys {
                            for &word in &words {
                                let x0 = word * word_width;
                                let x1 = (x0 + word_width).min(block.block_size);
                                let mut updates = Vec::with_capacity(x1 - x0);
                                for x in x0..x1 {
                                    let cell = local_cell(
                                        &self.pattern.local_to_cell[rank],
                                        block.block_size,
                                        [x, y, z],
                                    );
                                    if cell == NONE {
                                        continue;
                                    }
                                    let read = final_snapshot
                                        .as_deref()
                                        .unwrap_or(pressure.values.as_slice());
                                    let old = read[cell];
                                    let mut neighbors = 0.0f32;
                                    for &(neighbor, edge) in &self.pattern.row_edges[cell] {
                                        neighbors += self.edge_coefficients[edge] * read[neighbor];
                                    }
                                    let gs = (rhs.values[cell] + neighbors)
                                        * self.inverse_diagonal[cell];
                                    updates.push((cell, old + params.omega * (gs - old)));
                                }
                                for (cell, value) in updates {
                                    pressure.values[cell] = value;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn energy(&self, x: &PressureVector) -> Result<f64, PressureError> {
        self.pattern.check_vector(x)?;
        let edge_energy: f64 = self
            .pattern
            .edges
            .iter()
            .zip(&self.edge_coefficients)
            .map(|(edge, &c)| {
                let d = x.values[edge.a] - x.values[edge.b];
                c as f64 * d as f64 * d as f64
            })
            .sum();
        let boundary_energy: f64 = self
            .pattern
            .boundary_links
            .iter()
            .zip(&self.boundary_coefficients)
            .map(|(link, &c)| {
                let v = x.values[link.cell];
                c as f64 * v as f64 * v as f64
            })
            .sum();
        Ok(edge_energy + boundary_energy)
    }
}

/// Scalar CSR matvec over a contiguous cell range (the coarse-block fallback).
fn apply_csr_range(
    offsets: &[usize],
    cols: &[usize],
    vals: &[f32],
    diagonal: &[f32],
    xv: &[f32],
    rv: Option<&[f32]>,
    out: *mut f32,
    range: Range<usize>,
    dot: Option<&mut f64>,
    norm2: Option<&mut f64>,
) {
    let mut block_dot = 0.0f64;
    let mut block_norm2 = 0.0f64;
    for row in range {
        let mut value = diagonal[row] * xv[row];
        for k in offsets[row]..offsets[row + 1] {
            value -= vals[k] * xv[cols[k]];
        }
        if let Some(rv) = rv {
            value = rv[row] - value;
        }
        if dot.is_some() {
            block_dot += xv[row] as f64 * value as f64;
        }
        if norm2.is_some() {
            block_norm2 += value as f64 * value as f64;
        }
        // SAFETY: `row` is within the caller's output buffer.
        unsafe { *out.add(row) = value };
    }
    if let Some(dot) = dot {
        *dot += block_dot;
    }
    if let Some(norm2) = norm2 {
        *norm2 += block_norm2;
    }
}

/// Assemble an x-padded halo for a level-0 block from the dense vectors
/// (interior + six face gathers) and run the 7-pt stencil with SIMD `W`-lane
/// x-runs, storing with a non-temporal store.
///
/// # Safety
///
/// `halo` must be at least `(bs+2)^3` floats and belong to the calling worker
/// exclusively; `out` must be writable over the block's cell range; the face
/// tables must be valid cell ids or NONE.
#[inline(always)]
unsafe fn apply_block_simd<const W: usize>(
    blocks: &[PressureBlock],
    block_faces: &[Vec<usize>],
    diagonal: &[f32],
    block_coefficients: &[Vec<f32>],
    const_coeff: Option<f32>,
    correct_constant_faces: bool,
    finest_spacing: f32,
    right_coefficients: Option<[usize; 3]>,
    xv: &[f32],
    rv: Option<&[f32]>,
    out: *mut f32,
    rank: usize,
    halo: &mut [f32],
    dot: Option<&mut f64>,
    norm2: Option<&mut f64>,
) {
    let block = &blocks[rank];
    let bs = block.block_size;
    let hsy = bs + 2;
    let hsx = (bs + 2 * W).next_multiple_of(W);
    let hsx2 = hsx * hsy;
    let xbase = W;
    let halo_len = hsx2 * hsy;
    let block_len = bs * bs * bs;
    debug_assert!(bs % W == 0 && halo.len() >= halo_len);
    let cell_start = block.cell_start;

    for z in 0..bs {
        for y in 0..bs {
            let src = cell_start + bs * (y + bs * z);
            let dst = xbase + hsx * (1 + y) + hsx2 * (1 + z);
            halo[dst..dst + bs].copy_from_slice(&xv[src..src + bs]);
        }
    }
    // Six explicit face fills (no per-cell branch/div/mod, unchecked indexing).
    // Slab t-order is (minor-tangent, major-tangent): the source ids advance
    // contiguously along the x-runs for the ±y/±z faces of same-level neighbors,
    // and the halo positions advance by a constant stride per face.
    unsafe { fill_block_faces(halo, &block_faces[rank], bs, xbase, hsx, hsx2, xv) };

    // SAFETY: ho spans [W, halo_len - W] and c + W <= pattern.len(); the raw
    // loads skip the slice bounds checks so LLVM can pipeline the (L2/L3)
    // diagonal loads.
    let hbase = halo.as_ptr();
    let db = diagonal.as_ptr();
    let coeff_scratch = &block_coefficients[rank];
    let k = const_coeff.map(|beta| Simd::<f32, W>::splat(beta * finest_spacing));
    let rvb = rv.map(|r| r.as_ptr());
    let mut block_dot = 0.0f64;
    let mut block_norm2 = 0.0f64;
    for z in 0..bs {
        for y in 0..bs {
            for x0 in (0..bs).step_by(W) {
                let c = cell_start + x0 + bs * (y + bs * z);
                let local = x0 + bs * (y + bs * z);
                let ho = xbase + x0 + hsx * (1 + y) + hsx2 * (1 + z);
                let center = unsafe { load(hbase, ho) };
                let mut acc = if let Some(k) = k {
                    unsafe {
                        let nx = load(hbase, ho - 1);
                        let px = load(hbase, ho + 1);
                        let ny = load(hbase, ho - hsx);
                        let py = load(hbase, ho + hsx);
                        let nz = load(hbase, ho - hsx2);
                        let pz = load(hbase, ho + hsx2);
                        let mut value = load(db, c) * center - k * (nx + px + ny + py + nz + pz);
                        if correct_constant_faces {
                            let staged =
                                |dir: usize| load(coeff_scratch.as_ptr(), dir * block_len + local);
                            if x0 == 0 {
                                value = (k - staged(0)).mul_add(nx, value);
                            }
                            if x0 + W == bs {
                                value = (k - staged(1)).mul_add(px, value);
                            }
                            if y == 0 {
                                value = (k - staged(2)).mul_add(ny, value);
                            }
                            if y + 1 == bs {
                                value = (k - staged(3)).mul_add(py, value);
                            }
                            if z == 0 {
                                value = (k - staged(4)).mul_add(nz, value);
                            }
                            if z + 1 == bs {
                                value = (k - staged(5)).mul_add(pz, value);
                            }
                        }
                        value
                    }
                } else {
                    unsafe {
                        let staged =
                            |dir: usize| load(coeff_scratch.as_ptr(), dir * block_len + local);
                        let (wnx, wpx, wny, wpy, wnz, wpz) = if let Some(right) = right_coefficients
                        {
                            let h = Simd::<f32, W>::splat(finest_spacing);
                            let rx = right[0] as *const f32;
                            let ry = right[1] as *const f32;
                            let rz = right[2] as *const f32;
                            (
                                if x0 == 0 {
                                    staged(0)
                                } else {
                                    load(rx, c - 1) * h
                                },
                                load(rx, c) * h,
                                if y == 0 {
                                    staged(2)
                                } else {
                                    load(ry, c - bs) * h
                                },
                                load(ry, c) * h,
                                if z == 0 {
                                    staged(4)
                                } else {
                                    load(rz, c - bs * bs) * h
                                },
                                load(rz, c) * h,
                            )
                        } else {
                            (
                                staged(0),
                                staged(1),
                                staged(2),
                                staged(3),
                                staged(4),
                                staged(5),
                            )
                        };
                        let mut weighted = load(db, c) * center;
                        weighted = (-wnx).mul_add(load(hbase, ho - 1), weighted);
                        weighted = (-wpx).mul_add(load(hbase, ho + 1), weighted);
                        weighted = (-wny).mul_add(load(hbase, ho - hsx), weighted);
                        weighted = (-wpy).mul_add(load(hbase, ho + hsx), weighted);
                        weighted = (-wnz).mul_add(load(hbase, ho - hsx2), weighted);
                        (-wpz).mul_add(load(hbase, ho + hsx2), weighted)
                    }
                };
                if let Some(rvb) = rvb {
                    acc = unsafe { load(rvb, c) } - acc;
                }
                if dot.is_some() {
                    block_dot += (center * acc).reduce_sum() as f64;
                }
                if norm2.is_some() {
                    block_norm2 += (acc * acc).reduce_sum() as f64;
                }
                unsafe { store_nt::<W>(out, c, acc) };
            }
        }
    }
    if let Some(dot) = dot {
        *dot += block_dot;
    }
    if let Some(norm2) = norm2 {
        *norm2 += block_norm2;
    }
}

/// Per-block scalar 8-color GS (the coarse-block relax fallback). Returns
/// `(block_max, boundary_max)` contrast-equalizing statistics.
#[allow(clippy::too_many_arguments)]
fn relax_csr_block(
    base: usize,
    rv: &[f32],
    offsets: &[usize],
    cols: &[usize],
    vals: &[f32],
    map: &[usize],
    diagonal: &[f32],
    inverse_diagonal: &[f32],
    block: &PressureBlock,
    params: RelaxParams,
    word_width: usize,
) -> (f32, [f32; 6]) {
    let bs = block.block_size;
    let p_block = unsafe {
        std::slice::from_raw_parts_mut(
            (base as *mut f32).add(block.cell_start),
            block.cell_end - block.cell_start,
        )
    };
    let mut bmax = 0.0f32;
    let mut face_max = [0.0f32; 6];
    for _ in 0..params.local_sweeps {
        for inner_color in 0..2 {
            for z in 0..bs {
                for y in 0..bs {
                    let mut word_x = 0usize;
                    let mut x0 = 0usize;
                    while x0 < bs {
                        let x1 = (x0 + word_width).min(bs);
                        if (word_x + y + z) & 1 == inner_color {
                            let mut updates = [(0usize, 0.0f32, 0.0f32); 16];
                            let mut n = 0usize;
                            for x in x0..x1 {
                                let local = x + bs * (y + bs * z);
                                let cell = map[local];
                                if cell == NONE {
                                    continue;
                                }
                                let old = p_block[local];
                                let mut neighbors = 0.0f32;
                                for k in offsets[cell]..offsets[cell + 1] {
                                    // SAFETY: cols[k] is a live cell id; cross-block
                                    // reads hit the opposite (quiescent) color.
                                    let p = base as *const f32;
                                    neighbors += vals[k] * unsafe { *p.add(cols[k]) };
                                }
                                let gs = (rv[cell] + neighbors) * inverse_diagonal[cell];
                                let next = old + params.omega * (gs - old);
                                let delta = (next - old).abs() * diagonal[cell];
                                updates[n] = (local, next, delta);
                                n += 1;
                            }
                            for (local, next, delta) in &updates[..n] {
                                p_block[*local] = *next;
                                bmax = bmax.max(*delta);
                                let lx = *local % bs;
                                let ly = (*local / bs) % bs;
                                let lz = *local / (bs * bs);
                                if lx == 0 {
                                    face_max[0] = face_max[0].max(*delta);
                                }
                                if lx + 1 == bs {
                                    face_max[1] = face_max[1].max(*delta);
                                }
                                if ly == 0 {
                                    face_max[2] = face_max[2].max(*delta);
                                }
                                if ly + 1 == bs {
                                    face_max[3] = face_max[3].max(*delta);
                                }
                                if lz == 0 {
                                    face_max[4] = face_max[4].max(*delta);
                                }
                                if lz + 1 == bs {
                                    face_max[5] = face_max[5].max(*delta);
                                }
                            }
                        }
                        x0 = x1;
                        word_x += 1;
                    }
                }
            }
        }
    }
    (bmax, face_max)
}

/// Per-block SIMD 8-color GS: assemble the halo once, run `local_sweeps`
/// word-red-black Gauss-Seidel passes on the halo (the C++
/// `RELAX_BLOCK_GAUSS_SEIDEL` pattern), then copy the interior back. Returns
/// `(block_max, boundary_max)` statistics.
///
/// # Safety
///
/// `halo` must be at least `(bs+2)^3` floats and belong to the calling worker;
/// `base` must be the live pressure buffer; cross-block halo reads hit the
/// opposite (quiescent) color.
#[allow(clippy::too_many_arguments)]
unsafe fn relax_block_simd<const W: usize, const TRACK_METRICS: bool>(
    p: &[f32],
    base: usize,
    rv: &[f32],
    blocks: &[PressureBlock],
    block_faces: &[Vec<usize>],
    diagonal: &[f32],
    inverse_diagonal: &[f32],
    block_coefficients: &[Vec<f32>],
    const_coeff: Option<f32>,
    correct_constant_faces: bool,
    finest_spacing: f32,
    right_coefficients: Option<[usize; 3]>,
    rank: usize,
    halo: &mut [f32],
    params: RelaxParams,
    word_width: usize,
) -> (f32, [f32; 6]) {
    let block = &blocks[rank];
    let bs = block.block_size;
    let hsy = bs + 2;
    let hsx = (bs + 2 * W).next_multiple_of(W);
    let hsx2 = hsx * hsy;
    let xbase = W;
    let halo_len = hsx2 * hsy;
    let block_len = bs * bs * bs;
    debug_assert!(bs % W == 0 && halo.len() >= halo_len);
    let cell_start = block.cell_start;

    for z in 0..bs {
        for y in 0..bs {
            let src = cell_start + bs * (y + bs * z);
            let dst = xbase + hsx * (1 + y) + hsx2 * (1 + z);
            halo[dst..dst + bs].copy_from_slice(&p[src..src + bs]);
        }
    }
    unsafe { fill_block_faces(halo, &block_faces[rank], bs, xbase, hsx, hsx2, p) };

    let mut bmax = 0.0f32;
    let mut face_max = [0.0f32; 6];
    let omega = params.omega;
    let n_words = bs.div_ceil(word_width);
    // Raw pointers (no slice bounds checks) so the GS word loop unrolls.
    let hbase = halo.as_mut_ptr();
    let db = diagonal.as_ptr();
    let idb = inverse_diagonal.as_ptr();
    let rvb = rv.as_ptr();
    let coeff_scratch = &block_coefficients[rank];
    let k = const_coeff.map(|beta| Simd::<f32, W>::splat(beta * finest_spacing));
    let single_sweep = params.local_sweeps == 1;
    for _ in 0..params.local_sweeps {
        // Lexicographic SIMD-word GS, matching the C++ block kernel. Blocks in
        // the active outer wave are halo-disjoint, so live in-block updates do
        // not compromise deterministic parallel execution.
        for z in 0..bs {
            for y in 0..bs {
                for word_x in 0..n_words {
                    let x0 = word_x * word_width;
                    let c = cell_start + x0 + bs * (y + bs * z);
                    let local = x0 + bs * (y + bs * z);
                    let ho = xbase + x0 + hsx * (1 + y) + hsx2 * (1 + z);
                    let center = unsafe { load(hbase, ho) };
                    let s = if let Some(k) = k {
                        unsafe {
                            let nx = load(hbase, ho - 1);
                            let px = load(hbase, ho + 1);
                            let ny = load(hbase, ho - hsx);
                            let py = load(hbase, ho + hsx);
                            let nz = load(hbase, ho - hsx2);
                            let pz = load(hbase, ho + hsx2);
                            let mut weighted = k * (nx + px + ny + py + nz + pz);
                            if correct_constant_faces {
                                let staged = |dir: usize| {
                                    load(coeff_scratch.as_ptr(), dir * block_len + local)
                                };
                                if x0 == 0 {
                                    weighted = (staged(0) - k).mul_add(nx, weighted);
                                }
                                if x0 + W == bs {
                                    weighted = (staged(1) - k).mul_add(px, weighted);
                                }
                                if y == 0 {
                                    weighted = (staged(2) - k).mul_add(ny, weighted);
                                }
                                if y + 1 == bs {
                                    weighted = (staged(3) - k).mul_add(py, weighted);
                                }
                                if z == 0 {
                                    weighted = (staged(4) - k).mul_add(nz, weighted);
                                }
                                if z + 1 == bs {
                                    weighted = (staged(5) - k).mul_add(pz, weighted);
                                }
                            }
                            weighted
                        }
                    } else {
                        unsafe {
                            let staged =
                                |dir: usize| load(coeff_scratch.as_ptr(), dir * block_len + local);
                            let (wnx, wpx, wny, wpy, wnz, wpz) =
                                if let Some(right) = right_coefficients {
                                    let h = Simd::<f32, W>::splat(finest_spacing);
                                    let rx = right[0] as *const f32;
                                    let ry = right[1] as *const f32;
                                    let rz = right[2] as *const f32;
                                    (
                                        if x0 == 0 {
                                            staged(0)
                                        } else {
                                            load(rx, c - 1) * h
                                        },
                                        load(rx, c) * h,
                                        if y == 0 {
                                            staged(2)
                                        } else {
                                            load(ry, c - bs) * h
                                        },
                                        load(ry, c) * h,
                                        if z == 0 {
                                            staged(4)
                                        } else {
                                            load(rz, c - bs * bs) * h
                                        },
                                        load(rz, c) * h,
                                    )
                                } else {
                                    (
                                        staged(0),
                                        staged(1),
                                        staged(2),
                                        staged(3),
                                        staged(4),
                                        staged(5),
                                    )
                                };
                            let mut weighted = Simd::<f32, W>::splat(0.0);
                            weighted = wnx.mul_add(load(hbase, ho - 1), weighted);
                            weighted = wpx.mul_add(load(hbase, ho + 1), weighted);
                            weighted = wny.mul_add(load(hbase, ho - hsx), weighted);
                            weighted = wpy.mul_add(load(hbase, ho + hsx), weighted);
                            weighted = wnz.mul_add(load(hbase, ho - hsx2), weighted);
                            wpz.mul_add(load(hbase, ho + hsx2), weighted)
                        }
                    };
                    let gs = unsafe { (load(rvb, c) + s) * load(idb, c) };
                    let next = center + Simd::splat(omega) * (gs - center);
                    // A one-sweep block update consumes the frozen halo and
                    // writes the final block directly (the C++ fast path).
                    // Multi-sweep calls retain live in-halo GS updates.
                    unsafe {
                        if single_sweep {
                            next.copy_to_slice(std::slice::from_raw_parts_mut(
                                (base as *mut f32).add(c),
                                W,
                            ));
                        } else {
                            next.copy_to_slice(std::slice::from_raw_parts_mut(hbase.add(ho), W));
                        }
                    }
                    if TRACK_METRICS {
                        let delta = (next - center).abs() * unsafe { load(db, c) };
                        let word_max = delta.reduce_max();
                        bmax = bmax.max(word_max);
                        if word_x == 0 {
                            face_max[0] = face_max[0].max(word_max);
                        }
                        if word_x + 1 == n_words {
                            face_max[1] = face_max[1].max(word_max);
                        }
                        if y == 0 {
                            face_max[2] = face_max[2].max(word_max);
                        }
                        if y + 1 == bs {
                            face_max[3] = face_max[3].max(word_max);
                        }
                        if z == 0 {
                            face_max[4] = face_max[4].max(word_max);
                        }
                        if z + 1 == bs {
                            face_max[5] = face_max[5].max(word_max);
                        }
                    }
                }
            }
        }
    }
    if !single_sweep {
        let p_block = unsafe {
            std::slice::from_raw_parts_mut((base as *mut f32).add(cell_start), bs * bs * bs)
        };
        for z in 0..bs {
            for y in 0..bs {
                let src = xbase + hsx * (1 + y) + hsx2 * (1 + z);
                p_block[bs * (y + bs * z)..bs * (y + bs * z) + bs]
                    .copy_from_slice(&halo[src..src + bs]);
            }
        }
    }
    (bmax, face_max)
}

/// Fill a `[bs+2]^3` halo's six face slabs from the dense source vector.
///
/// The ±y/±z faces read x-contiguous runs in `src` for same-level neighbors, so
/// a contiguity check turns each row into a single high-bandwidth copy; the ±x
/// faces are strided gathers and software-prefetch the source 8 ahead.
///
/// # Safety
///
/// `halo` must be `(bs+2)^3` floats; every non-NONE id in `block_faces` must be
/// `< src.len()` (guaranteed by the face-table construction).
#[inline(always)]
unsafe fn fill_block_faces(
    halo: &mut [f32],
    block_faces: &[usize],
    bs: usize,
    xbase: usize,
    hsx: usize,
    hsx2: usize,
    src: &[f32],
) {
    let slab = |face: usize| &block_faces[face * bs * bs..(face + 1) * bs * bs];
    let hp = halo.as_mut_ptr();
    let src_ptr = src.as_ptr();
    // ±x faces: strided source, prefetch ahead.
    for z in 0..bs {
        for (face, xoff) in [(0usize, xbase - 1), (1, xbase + bs)] {
            let mut ho = xoff + hsx * 1 + hsx2 * (1 + z);
            let mut t = z * bs;
            while t < z * bs + bs {
                if t + 8 < (z + 1) * bs {
                    let n = unsafe { *slab(face).get_unchecked(t + 8) };
                    if n != NONE {
                        unsafe { core::arch::x86_64::_mm_prefetch(src_ptr.add(n) as *const i8, 3) };
                    }
                }
                let nid = unsafe { *slab(face).get_unchecked(t) };
                // SAFETY: ho is within the halo; nid is validated at build time.
                unsafe {
                    *hp.add(ho) = if nid == NONE {
                        0.0
                    } else {
                        *src.get_unchecked(nid)
                    };
                }
                ho += hsx;
                t += 1;
            }
        }
    }
    // ±y/±z faces: contiguous-run copy fast path.
    for z in 0..bs {
        for (face, yoff) in [(2usize, 0usize), (3, bs + 1)] {
            let start = z * bs;
            let row = unsafe { slab(face).get_unchecked(start..start + bs) };
            let contiguous = row[0] != NONE && row[bs - 1] == row[0] + (bs - 1);
            if contiguous {
                let ho = xbase + hsx * yoff + hsx2 * (1 + z);
                halo[ho..ho + bs].copy_from_slice(&src[row[0]..row[0] + bs]);
            } else {
                let ho = xbase + hsx * yoff + hsx2 * (1 + z);
                for (i, &nid) in row.iter().enumerate() {
                    unsafe {
                        *hp.add(ho + i) = if nid == NONE {
                            0.0
                        } else {
                            *src.get_unchecked(nid)
                        };
                    }
                }
            }
        }
    }
    for y in 0..bs {
        for (face, zoff) in [(4usize, 0usize), (5, bs + 1)] {
            let start = y * bs;
            let row = unsafe { slab(face).get_unchecked(start..start + bs) };
            let contiguous = row[0] != NONE && row[bs - 1] == row[0] + (bs - 1);
            if contiguous {
                let ho = xbase + hsx * (1 + y) + hsx2 * zoff;
                halo[ho..ho + bs].copy_from_slice(&src[row[0]..row[0] + bs]);
            } else {
                let ho = xbase + hsx * (1 + y) + hsx2 * zoff;
                for (i, &nid) in row.iter().enumerate() {
                    unsafe {
                        *hp.add(ho + i) = if nid == NONE {
                            0.0
                        } else {
                            *src.get_unchecked(nid)
                        };
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn native_word_width(block_size: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    let native = if std::is_x86_feature_detected!("avx512f") {
        16
    } else if std::is_x86_feature_detected!("avx2") {
        8
    } else {
        4
    };
    #[cfg(target_arch = "aarch64")]
    let native = 4;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let native = 4;
    native.min(block_size).max(1)
}

/// Per-thread scratch buffers for the block-SIMD pressure kernels. Indexed via
/// `rayon::current_thread_index()` exactly like `HaloBlockPool`; each slot is
/// exclusively owned by one rayon worker, which makes `Sync` sound.
struct ScratchPool {
    slots: Vec<std::cell::UnsafeCell<AlignedValues>>,
}
unsafe impl Sync for ScratchPool {}
impl ScratchPool {
    fn new(num_threads: usize, len: usize) -> Self {
        let slots = (0..num_threads.max(1))
            .map(|_| std::cell::UnsafeCell::new(AlignedValues::new(len, 0.0)))
            .collect();
        Self { slots }
    }
    #[inline]
    unsafe fn get_mut(&self) -> &mut [f32] {
        // Fall back to slot 0 on non-worker threads (rayon runs tiny or
        // single-item iterations inline on the caller).
        let idx = rayon::current_thread_index().unwrap_or(0);
        debug_assert!(idx < self.slots.len());
        unsafe { (&mut *self.slots[idx].get()).as_mut_slice() }
    }
}

#[inline(always)]
fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

#[derive(Clone, Copy, Debug)]
pub struct PhaseFieldParams {
    pub liquid_density: f32,
    pub gas_density: f32,
    pub rho_tilde_0: f32,
    pub alpha_phi: f32,
}

/// Optional one-bit phase classification for pressure cells (`false` = gas,
/// `true` = liquid). Labels only snap faces whose two incident cells have the
/// same phase; mixed faces retain the continuous phase-field value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellPhaseLabels {
    bits: Vec<u64>,
    len: usize,
}

impl CellPhaseLabels {
    pub fn from_bools(liquid: &[bool]) -> Self {
        let mut bits = vec![0u64; liquid.len().div_ceil(64)];
        for (i, enabled) in liquid.iter().copied().enumerate() {
            if enabled {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self {
            bits,
            len: liquid.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn is_liquid(&self, cell: usize) -> bool {
        self.bits[cell / 64] >> (cell % 64) & 1 != 0
    }
}

impl PhaseFieldParams {
    pub fn new(
        liquid_density: f32,
        gas_density: f32,
        rho_tilde_0: f32,
        alpha_phi: f32,
    ) -> Result<Self, PressureError> {
        let params = Self {
            liquid_density,
            gas_density,
            rho_tilde_0,
            alpha_phi,
        };
        params.validate()?;
        Ok(params)
    }

    fn validate(self) -> Result<(), PressureError> {
        if !self.liquid_density.is_finite()
            || !self.gas_density.is_finite()
            || !self.rho_tilde_0.is_finite()
            || !self.alpha_phi.is_finite()
            || self.gas_density <= 0.0
            || self.liquid_density <= self.gas_density
            || self.rho_tilde_0 <= 0.0
            || self.alpha_phi <= 0.0
        {
            return Err(PressureError::InvalidPhaseParameters);
        }
        Ok(())
    }

    pub fn phase(self, raw_mass_density: f32) -> f32 {
        let threshold =
            (self.liquid_density / self.gas_density).ln() * self.gas_density * self.rho_tilde_0;
        if raw_mass_density < threshold {
            return 0.0;
        }
        ((raw_mass_density - threshold).max(0.0)
            / (self.alpha_phi * self.rho_tilde_0 * self.liquid_density))
            .sqrt()
            .min(1.0)
    }

    pub fn beta_from_phase(self, phase: f32) -> f32 {
        let phi = phase.clamp(0.0, 1.0);
        (self.gas_density + phi * (self.liquid_density - self.gas_density)).recip()
    }
}

/// Convert uniform finest-level MAC face masses directly into the three
/// canonical right-face `β = 1/ρ` streams. The paper's prose writes the density
/// interpolation in reversed `lerp` order; this uses its Eq. 7 classification,
/// where `φ=0` is gas and `φ=1` is liquid.
pub fn coefficients_from_mac_masses<const BSX: usize, const N: usize>(
    pattern: &PressurePattern,
    mac: &MacGrid<BSX, N>,
    params: PhaseFieldParams,
    labels: Option<&CellPhaseLabels>,
) -> Result<FaceCoefficients, PressureError> {
    params.validate()?;
    if pattern.finest_dims != [mac.sx(), mac.sy(), mac.sz()]
        || pattern.finest_block_size != BSX
        || pattern.cells.iter().any(|cell| cell.level != 0)
    {
        return Err(PressureError::UnsupportedMacLayout);
    }
    if let Some(labels) = labels
        && labels.len() != pattern.len()
    {
        return Err(PressureError::PhaseLabelLength {
            expected: pattern.len(),
            actual: labels.len(),
        });
    }

    let mut by_origin = HashMap::with_capacity(pattern.len());
    for (cell, metadata) in pattern.cells.iter().enumerate() {
        by_origin.insert(metadata.fine_origin, cell);
    }
    let mut right = std::array::from_fn(|_| vec![0.0f32; pattern.len()]);
    for (cell, metadata) in pattern.cells.iter().enumerate() {
        let [x, y, z] = metadata.fine_origin;
        for axis in 0..3 {
            let raw = mac.mass(axis).get_voxel(x, y, z);
            let mut phase = params.phase(raw);
            if let Some(labels) = labels {
                let mut neighbor_origin = metadata.fine_origin;
                let extent = pattern.finest_dims[axis];
                let neighbor = if neighbor_origin[axis] + 1 < extent {
                    neighbor_origin[axis] += 1;
                    by_origin.get(&neighbor_origin).copied()
                } else if pattern.boundaries[2 * axis + 1] == PressureBoundary::Periodic {
                    neighbor_origin[axis] = 0;
                    by_origin.get(&neighbor_origin).copied()
                } else {
                    None
                };
                if let Some(neighbor) = neighbor {
                    match (labels.is_liquid(cell), labels.is_liquid(neighbor)) {
                        (false, false) => phase = 0.0,
                        (true, true) => phase = 1.0,
                        _ => {}
                    }
                }
            }
            right[axis][cell] = params.beta_from_phase(phase);
        }
    }
    Ok(FaceCoefficients { right })
}

pub fn operator_from_mac_masses<const BSX: usize, const N: usize>(
    pattern: Arc<PressurePattern>,
    mac: &MacGrid<BSX, N>,
    params: PhaseFieldParams,
    labels: Option<&CellPhaseLabels>,
    format: CoefficientFormat,
) -> Result<PressureOperator, PressureError> {
    let coefficients = coefficients_from_mac_masses(&pattern, mac, params, labels)?;
    PressureOperator::with_format(pattern, coefficients, format)
}

/// Form the finite-volume pressure RHS directly from MAC velocities without an
/// intermediate divergence grid: `b=-h³/dt·(div(u*)+DIVCORR)`.
pub fn rhs_from_mac<const BSX: usize, const N: usize>(
    pattern: &PressurePattern,
    mac: &MacGrid<BSX, N>,
    dt: f32,
    divergence_correction: Option<&PressureVector>,
) -> Result<PressureVector, PressureError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PressureError::InvalidTimeStep(dt));
    }
    if pattern.finest_dims != [mac.sx(), mac.sy(), mac.sz()]
        || pattern.finest_block_size != BSX
        || pattern.cells.iter().any(|cell| cell.level != 0)
    {
        return Err(PressureError::UnsupportedMacLayout);
    }
    if let Some(correction) = divergence_correction {
        pattern.check_vector(correction)?;
    }
    let inverse_spacing = pattern.finest_spacing.recip();
    let inverse_dt = dt.recip();
    let mut rhs = pattern.vector();
    for (cell, metadata) in pattern.cells.iter().enumerate() {
        let [x, y, z] = metadata.fine_origin;
        let mut divergence = 0.0f32;
        for axis in 0..3 {
            let right = mac.velocity(axis).get_voxel(x, y, z);
            let left = match axis {
                0 if x > 0 => mac.velocity(axis).get_voxel(x - 1, y, z),
                1 if y > 0 => mac.velocity(axis).get_voxel(x, y - 1, z),
                2 if z > 0 => mac.velocity(axis).get_voxel(x, y, z - 1),
                _ => 0.0,
            };
            divergence += right - left;
        }
        divergence *= inverse_spacing;
        let correction = divergence_correction.map_or(0.0, |values| values.values[cell]);
        rhs.values[cell] = -pattern.cell_volume(cell) * inverse_dt * (divergence + correction);
    }
    Ok(rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use crate::multires::RefinementMap;
    use std::sync::Arc;

    const CLOSED: [PressureBoundary; 6] = [PressureBoundary::ClosedNeumann; 6];

    fn uniform_pattern(boundaries: [PressureBoundary; 6]) -> Arc<PressurePattern> {
        let mut grid = MultiresGrid::create("p", 16, 16, 16, 16, 2, 2);
        let mut map = RefinementMap::uniform(1, 0);
        grid.set_refinement_map(&mut map);
        PressurePatternBuilder::new(&grid, &[0], boundaries)
            .build()
            .unwrap()
    }

    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state as f32 / u32::MAX as f32
            })
            .collect()
    }

    #[test]
    fn pressure_01_closed_neumann_constant_nullspace() {
        let pattern = uniform_pattern(CLOSED);
        let op = PressureOperator::constant(pattern.clone(), 2.5).unwrap();
        let x = pattern.vector_with_value(7.0);
        let mut y = pattern.vector();
        op.apply(&x, &mut y).unwrap();
        assert!(y.as_slice().iter().all(|v| v.abs() < 2e-5));
        assert!(op.energy(&x).unwrap() < 1e-10);
    }

    #[test]
    fn pressure_02_symmetry_energy_and_fused_apply() {
        let pattern = uniform_pattern(CLOSED);
        let beta = FaceCoefficients {
            right: std::array::from_fn(|axis| {
                pseudo_random(pattern.len(), 10 + axis as u32)
                    .into_iter()
                    .map(|v| 0.001 + 2.0 * v)
                    .collect()
            }),
        };
        let op = PressureOperator::new(pattern.clone(), beta).unwrap();
        let mut x = pattern.vector();
        let mut y = pattern.vector();
        x.values.copy_from_slice(&pseudo_random(pattern.len(), 41));
        y.values.copy_from_slice(&pseudo_random(pattern.len(), 92));
        let mut ax = pattern.vector();
        let mut ay = pattern.vector();
        op.apply(&x, &mut ax).unwrap();
        op.apply(&y, &mut ay).unwrap();
        let xtay = x.dot(&ay).unwrap();
        let ytax = y.dot(&ax).unwrap();
        assert!((xtay - ytax).abs() <= 2e-3 * xtay.abs().max(1.0));
        let fused = op.apply_and_dot(&x, &mut ax).unwrap();
        assert!((fused - op.energy(&x).unwrap()).abs() <= 2e-3 * fused.abs().max(1.0));
        assert!(fused >= -1e-5);
    }

    #[test]
    fn pressure_03_affine_and_quadratic_interior() {
        let pattern = uniform_pattern(CLOSED);
        let op = PressureOperator::constant(pattern.clone(), 1.0).unwrap();
        let mut affine = pattern.vector();
        let mut quadratic = pattern.vector();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let cell = pattern.cell_index(0, x, y, z).unwrap();
                    affine.values[cell] = 2.0 * x as f32 - 3.0 * y as f32 + 0.5 * z as f32 + 1.0;
                    quadratic.values[cell] = (x * x + y * y + z * z) as f32;
                }
            }
        }
        let mut result = pattern.vector();
        op.apply(&affine, &mut result).unwrap();
        for z in 1..15 {
            for y in 1..15 {
                for x in 1..15 {
                    assert!(result.values[pattern.cell_index(0, x, y, z).unwrap()].abs() < 1e-4);
                }
            }
        }
        op.apply(&quadratic, &mut result).unwrap();
        for z in 1..15 {
            for y in 1..15 {
                for x in 1..15 {
                    let got = result.values[pattern.cell_index(0, x, y, z).unwrap()];
                    assert!(
                        (got + 6.0).abs() < 1e-4,
                        "quadratic at ({x},{y},{z}) = {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn pressure_04_dirichlet_and_periodic_boundaries() {
        let open = uniform_pattern([PressureBoundary::OpenDirichletZero; 6]);
        let open_op = PressureOperator::constant(open.clone(), 1.0).unwrap();
        let ones = open.vector_with_value(1.0);
        let mut out = open.vector();
        open_op.apply(&ones, &mut out).unwrap();
        let corner = open.cell_index(0, 0, 0, 0).unwrap();
        let interior = open.cell_index(0, 8, 8, 8).unwrap();
        assert_eq!(out.values[corner], 3.0);
        assert_eq!(out.values[interior], 0.0);

        let periodic = uniform_pattern([PressureBoundary::Periodic; 6]);
        let periodic_op = PressureOperator::constant(periodic.clone(), 1.0).unwrap();
        let x = periodic.vector_with_value(4.0);
        let mut y = periodic.vector();
        periodic_op.apply(&x, &mut y).unwrap();
        assert!(y.values.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn pressure_05_fine_coarse_pairs_are_symmetric_and_constant_preserving() {
        for levels in [vec![0, 1], vec![1, 0]] {
            let mut grid = MultiresGrid::create("adaptive", 32, 16, 16, 16, 2, 2);
            let mut map = RefinementMap::new(levels);
            grid.set_refinement_map(&mut map);
            let pattern = PressurePatternBuilder::new(&grid, &[0, 1], CLOSED)
                .build()
                .unwrap();
            assert_eq!(pattern.transition_link_count(), 16 * 16);
            let op = PressureOperator::constant(pattern.clone(), 1.0).unwrap();
            let constant = pattern.vector_with_value(3.0);
            let mut applied = pattern.vector();
            op.apply(&constant, &mut applied).unwrap();
            assert!(applied.values.iter().all(|v| v.abs() < 1e-5));

            let mut x = pattern.vector();
            let mut y = pattern.vector();
            x.values.copy_from_slice(&pseudo_random(pattern.len(), 7));
            y.values.copy_from_slice(&pseudo_random(pattern.len(), 8));
            let mut ax = pattern.vector();
            let mut ay = pattern.vector();
            op.apply(&x, &mut ax).unwrap();
            op.apply(&y, &mut ay).unwrap();
            let lhs = x.dot(&ay).unwrap();
            let rhs = y.dot(&ax).unwrap();
            assert!((lhs - rhs).abs() <= 2e-3 * lhs.abs().max(1.0));
            assert!(ax.values.iter().map(|&v| v as f64).sum::<f64>().abs() < 1e-3);
        }
    }

    #[test]
    fn pressure_06_activity_mask_and_validation() {
        let mut grid = MultiresGrid::create("p", 16, 16, 16, 16, 2, 2);
        let mut map = RefinementMap::uniform(1, 0);
        grid.set_refinement_map(&mut map);
        let mut active = vec![true; 4096];
        active[0] = false;
        let mask = CellActivityMask::from_bools(&active);
        let pattern =
            PressurePatternBuilder::new(&grid, &[0], [PressureBoundary::OpenDirichletZero; 6])
                .activity_mask(&mask)
                .build()
                .unwrap();
        assert_eq!(pattern.len(), 4095);
        assert_eq!(pattern.cell_index(0, 0, 0, 0), None);
        PressureOperator::constant(pattern, 1.0).unwrap();

        assert!(matches!(
            PressurePatternBuilder::new(&grid, &[], CLOSED).build(),
            Err(PressureError::EmptyActiveSet)
        ));
        assert!(matches!(
            RelaxParams::new(-0.1, 1),
            Err(PressureError::InvalidRelaxation { .. })
        ));
        assert!(RelaxParams::new(0.0, 1).is_ok());
        assert!(RelaxParams::new(1.0, 1).is_ok());
    }

    #[test]
    fn pressure_07_pattern_mismatch_and_fused_residual() {
        let a = uniform_pattern(CLOSED);
        let b = uniform_pattern(CLOSED);
        let op = PressureOperator::constant(a.clone(), 1.0).unwrap();
        let x = b.vector();
        let mut out = a.vector();
        assert_eq!(op.apply(&x, &mut out), Err(PressureError::PatternMismatch));

        let rhs = a.vector_with_value(2.0);
        let x = a.vector();
        let norm2 = op.residual(&rhs, &x, &mut out).unwrap();
        assert_eq!(norm2, 4.0 * a.len() as f64);
        assert!(out.values.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn pressure_08_relaxation_endpoints_metrics_and_convergence() {
        let pattern = uniform_pattern([PressureBoundary::OpenDirichletZero; 6]);
        let op = PressureOperator::constant(pattern.clone(), 1.0).unwrap();
        let rhs = pattern.vector_with_value(1.0);
        let mut p = pattern.vector();
        let mut metrics = RelaxMetrics::new(&pattern);
        op.relax(
            &mut p,
            &rhs,
            RelaxParams::new(0.0, 1).unwrap(),
            Some(&mut metrics),
        )
        .unwrap();
        assert!(p.values.iter().all(|&v| v == 0.0));
        op.relax(
            &mut p,
            &rhs,
            RelaxParams::new(1.0, 2).unwrap(),
            Some(&mut metrics),
        )
        .unwrap();
        assert!(p.values.iter().any(|&v| v > 0.0));
        assert!(metrics.block_max[0] > 0.0);
        let mut residual = pattern.vector();
        let before = rhs.dot(&rhs).unwrap();
        let after = op.residual(&rhs, &p, &mut residual).unwrap();
        assert!(after < before);
    }

    #[test]
    fn pressure_09_phase_conversion_labels_and_bf16() {
        assert!(PhaseFieldParams::new(1.0, 1.0, 1.0, 1.0).is_err());
        let params = PhaseFieldParams::new(1000.0, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(params.beta_from_phase(0.0), 1.0);
        assert!((params.beta_from_phase(1.0) - 0.001).abs() < 1e-8);

        let pattern = uniform_pattern(CLOSED);
        let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(8, 64));
        let mut mac = MacGrid::new(16, 16, 16, pool);
        mac.zero_blocks(&[0]);
        for axis in 0..3 {
            mac.mass_mut(axis).fill_blocks_parallel(&[0], 1000.0);
        }
        let labels = CellPhaseLabels::from_bools(&vec![true; pattern.len()]);
        let coefficients =
            coefficients_from_mac_masses(&pattern, &mac, params, Some(&labels)).unwrap();
        for z in 0..15 {
            for y in 0..15 {
                for x in 0..15 {
                    let cell = pattern.cell_index(0, x, y, z).unwrap();
                    assert!(
                        coefficients
                            .right
                            .iter()
                            .all(|axis| (axis[cell] - 0.001).abs() < 1e-8)
                    );
                }
            }
        }
        let op = PressureOperator::with_format(
            pattern,
            coefficients,
            CoefficientFormat::ExperimentalBf16,
        )
        .unwrap();
        assert_eq!(op.coefficient_format(), CoefficientFormat::ExperimentalBf16);
    }

    #[test]
    fn pressure_10_fused_mac_rhs_uses_physical_spacing() {
        let mut grid = MultiresGrid::create("p", 16, 16, 16, 16, 2, 2);
        let mut map = RefinementMap::uniform(1, 0);
        grid.set_refinement_map(&mut map);
        let pattern = PressurePatternBuilder::new(&grid, &[0], CLOSED)
            .finest_spacing(0.5)
            .build()
            .unwrap();
        let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(8, 64));
        let mut mac = MacGrid::new(16, 16, 16, pool);
        mac.zero_blocks(&[0]);
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    mac.velocity_mut(0).set_voxel(x, y, z, (x + 1) as f32);
                    mac.velocity_mut(1).set_voxel(x, y, z, (y + 1) as f32);
                    mac.velocity_mut(2).set_voxel(x, y, z, (z + 1) as f32);
                }
            }
        }
        let rhs = rhs_from_mac(&pattern, &mac, 0.25, None).unwrap();
        assert!(rhs.values.iter().all(|&v| (v + 3.0).abs() < 1e-6));
    }

    #[test]
    fn pressure_11_scalar_edge_oracle_covers_boundaries_and_inactive_cells() {
        for boundaries in [
            CLOSED,
            [PressureBoundary::OpenDirichletZero; 6],
            [PressureBoundary::Periodic; 6],
        ] {
            let mut grid = MultiresGrid::create("oracle", 16, 16, 16, 16, 2, 2);
            let mut map = RefinementMap::uniform(1, 0);
            grid.set_refinement_map(&mut map);
            let mut active = vec![true; 4096];
            for i in (37..4096).step_by(97) {
                active[i] = false;
            }
            let mask = CellActivityMask::from_bools(&active);
            let pattern = PressurePatternBuilder::new(&grid, &[0], boundaries)
                .activity_mask(&mask)
                .build()
                .unwrap();
            let coefficients = FaceCoefficients {
                right: std::array::from_fn(|axis| {
                    pseudo_random(pattern.len(), 100 + axis as u32)
                        .into_iter()
                        .map(|v| 0.01 + 10.0 * v)
                        .collect()
                }),
            };
            let op = PressureOperator::new(pattern.clone(), coefficients).unwrap();
            let mut x = pattern.vector();
            x.values.copy_from_slice(&pseudo_random(pattern.len(), 900));
            let mut expected = vec![0.0f32; pattern.len()];
            for (edge, &coefficient) in pattern.edges.iter().zip(&op.edge_coefficients) {
                let flux = coefficient * (x.values[edge.a] - x.values[edge.b]);
                expected[edge.a] += flux;
                expected[edge.b] -= flux;
            }
            for (link, &coefficient) in pattern.boundary_links.iter().zip(&op.boundary_coefficients)
            {
                expected[link.cell] += coefficient * x.values[link.cell];
            }
            let mut actual = pattern.vector();
            op.apply(&x, &mut actual).unwrap();
            for (&got, want) in actual.values.iter().zip(expected) {
                assert!((got - want).abs() <= 2e-5 * got.abs().max(1.0));
            }
            let mut repeated = pattern.vector();
            op.apply(&x, &mut repeated).unwrap();
            assert_eq!(actual.as_slice(), repeated.as_slice());
        }
    }

    #[test]
    fn pressure_12_rejects_isolated_cells_and_unpaired_periodic_faces() {
        let mut grid = MultiresGrid::create("invalid", 16, 16, 16, 16, 2, 2);
        let mut map = RefinementMap::uniform(1, 0);
        grid.set_refinement_map(&mut map);
        let mut one = vec![false; 4096];
        one[0] = true;
        let mask = CellActivityMask::from_bools(&one);
        let isolated = PressurePatternBuilder::new(&grid, &[0], CLOSED)
            .activity_mask(&mask)
            .build()
            .unwrap();
        assert!(matches!(
            PressureOperator::constant(isolated, 1.0),
            Err(PressureError::ZeroDiagonal { .. })
        ));

        let mut invalid = CLOSED;
        invalid[0] = PressureBoundary::Periodic;
        assert!(matches!(
            PressurePatternBuilder::new(&grid, &[0], invalid).build(),
            Err(PressureError::InvalidBoundaryPair { axis: 0 })
        ));
    }

    #[test]
    fn pressure_13_hybrid_smoother_beats_weighted_jacobi() {
        let pattern = uniform_pattern([PressureBoundary::OpenDirichletZero; 6]);
        let op = PressureOperator::constant(pattern.clone(), 1.0).unwrap();
        let mut rhs = pattern.vector();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let cell = pattern.cell_index(0, x, y, z).unwrap();
                    rhs.values[cell] = if (x + y + z) & 1 == 0 { 1.0 } else { -1.0 };
                }
            }
        }
        let params = RelaxParams::default();
        let mut hybrid = pattern.vector();
        let mut jacobi = pattern.vector();
        let mut ax = pattern.vector();
        for _ in 0..4 {
            op.relax(&mut hybrid, &rhs, params, None).unwrap();
            op.apply(&jacobi, &mut ax).unwrap();
            for cell in 0..pattern.len() {
                jacobi.values[cell] +=
                    params.omega * op.inverse_diagonal[cell] * (rhs.values[cell] - ax.values[cell]);
            }
        }
        let mut residual = pattern.vector();
        let hybrid_norm = op.residual(&rhs, &hybrid, &mut residual).unwrap();
        let jacobi_norm = op.residual(&rhs, &jacobi, &mut residual).unwrap();
        assert!(hybrid_norm < jacobi_norm, "{hybrid_norm} vs {jacobi_norm}");
    }

    #[test]
    fn pressure_14_parallel_relax_is_bit_identical_to_serial_colored_sweep() {
        let mut grid = MultiresGrid::create("p", 32, 32, 32, 16, 2, 2);
        let active: Vec<usize> = (0..grid.dims.n_blocks).collect();
        let mut map = RefinementMap::uniform(active.len(), 0);
        grid.set_refinement_map(&mut map);
        let pattern = PressurePatternBuilder::new(&grid, &active, CLOSED)
            .build()
            .unwrap();
        let op = PressureOperator::constant(pattern.clone(), 1.0).unwrap();
        let mut rhs = pattern.vector();
        rhs.values
            .copy_from_slice(&pseudo_random(pattern.len(), 0x91e1_0da5));
        let initial = pseudo_random(pattern.len(), 0x7f4a_7c15);
        let mut serial = pattern.vector();
        serial.values.copy_from_slice(&initial);
        let mut parallel = serial.clone();
        let params = RelaxParams::default();
        let serial_pool = crate::thread_pool::Pool::new(1);
        serial_pool.install(|| op.relax(&mut serial, &rhs, params, None).unwrap());
        let parallel_pool = crate::thread_pool::Pool::new(4);
        parallel_pool.install(|| op.relax(&mut parallel, &rhs, params, None).unwrap());
        assert_eq!(serial.as_slice(), parallel.as_slice());
    }
}

//! Refinement map and its regularization (`regularizeRefinementMap`) plus the
//! block-topology computation of `setRefinementMap`.

use crate::channel::{CellFlags, DistFineCoarse};
use crate::multires::blockinfo::{
    BlockFlags, BlockInfoStore, CELL_BLK_BORDER, CELL_COARSE_FINE, CELL_FINE_COARSE, CELL_VOID,
};
use crate::multires::level::LevelData;
use rayon::prelude::*;

/// Per-block finest-resolution level map. `levels[bid]` is the resolution level
/// (0 = finest) a block lives at.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefinementMap {
    pub levels: Vec<u8>,
}

impl RefinementMap {
    pub fn new(levels: Vec<u8>) -> Self {
        Self { levels }
    }

    /// A map with every block at `level` (C++ `create()` default: coarsest).
    pub fn uniform(n_blocks: usize, level: u8) -> Self {
        Self {
            levels: vec![level; n_blocks],
        }
    }
}

/// Block-grid dimensions at level 0 (identical across resolution levels).
#[derive(Clone, Copy, Debug)]
pub struct BlockGridDims {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nxy: usize,
    pub n_blocks: usize,
}

impl BlockGridDims {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            nx,
            ny,
            nz,
            nxy: nx * ny,
            n_blocks: nx * ny * nz,
        }
    }

    #[inline(always)]
    pub fn coords(&self, bid: usize) -> (usize, usize, usize) {
        (bid % self.nx, (bid / self.nx) % self.ny, bid / self.nxy)
    }

    #[inline(always)]
    pub fn in_range(&self, bx: i32, by: i32, bz: i32) -> bool {
        bx >= 0 && bx < self.nx as i32 && by >= 0 && by < self.ny as i32 && bz >= 0 && bz < self.nz as i32
    }
}

/// Enforce the 2:1 refinement constraint: neighboring blocks may not differ by
/// more than one level. Mirrors C++ `regularizeRefinementMap` (an `nLevels-2`
/// relaxation, in place when `map` is the output).
pub fn regularize_refinement_map(map: &mut [u8], dims: &BlockGridDims, n_levels: usize) {
    debug_assert_eq!(map.len(), dims.n_blocks);
    if n_levels <= 2 {
        return;
    }
    let mut prev = map.to_vec();
    for ll in 1..=(n_levels - 2) {
        let ll = ll as u8;
        map.par_iter_mut()
            .enumerate()
            .for_each(|(bid, out)| {
                let level = prev[bid];
                let mut level_out = level;
                if level >= ll {
                    let (bx, by, bz) = dims.coords(bid);
                    let mut level2 = n_levels as u8 - 1;
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let bx2 = bx as i32 + dx;
                                let by2 = by as i32 + dy;
                                let bz2 = bz as i32 + dz;
                                if !dims.in_range(bx2, by2, bz2) {
                                    continue;
                                }
                                let bid2 = bx2 as usize + by2 as usize * dims.nx + bz2 as usize * dims.nxy;
                                level2 = level2.min(prev[bid2]);
                            }
                        }
                    }
                    if level2 == ll - 1 {
                        level_out = ll;
                    }
                }
                *out = level_out;
            });
        prev.copy_from_slice(map);
    }
}

/// Block lists produced by `set_refinement_map` (levelMg == 0).
#[derive(Debug, Default, Clone)]
pub struct Topology {
    /// Blocks flagged `BLK_FINE_COARSE`, indexed by resolution level.
    pub blocks_fine_coarse: Vec<Vec<usize>>,
    /// Number of blocks flagged `BLK_EXISTS` (× voxels/block = active cells).
    pub n_act_blocks: u64,
}

/// Compute per-block effective level and flags for `levelMg == 0`, writing them
/// into `store`. Mirrors the block-flag part of C++ `setRefinementMap` (without
/// obstacles, which the demo/benchmark never pass).
///
/// `block_flags`, when `Some`, supplies the user block flags (C++ `blockFlags`);
/// its `BLK_EXISTS` bit drives the existing-block list and active-cell count.
pub fn compute_block_topology(
    store: &mut BlockInfoStore,
    map: &RefinementMap,
    dims: &BlockGridDims,
    n_levels: usize,
    block_flags: Option<&[BlockFlags]>,
) -> Topology {
    debug_assert_eq!(map.levels.len(), dims.n_blocks);
    debug_assert_eq!(store.level0.len(), dims.n_blocks);

    let nx = dims.nx;
    let ny = dims.ny;
    let nz = dims.nz;
    let nxy = dims.nxy;

    // Parallel: compute the (level, flags) pair for every block.
    let results: Vec<(u8, BlockFlags)> = (0..dims.n_blocks)
        .into_par_iter()
        .map(|bid| {
            let level = map.levels[bid];
            debug_assert!((level as usize) < n_levels);

            let mut flags = block_flags.map_or(BlockFlags(0), |f| f[bid]);
            let (bx, by, bz) = dims.coords(bid);

            if bx == 0 || bx == nx - 1 || by == 0 || by == ny - 1 || bz == 0 || bz == nz - 1 {
                flags.set(BlockFlags::DOM_BORDER);
            }

            let mut level2_max = 0u8;
            let mut level2_min = (n_levels - 1) as u8;
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let bx2 = bx as i32 + dx;
                        let by2 = by as i32 + dy;
                        let bz2 = bz as i32 + dz;
                        if !dims.in_range(bx2, by2, bz2) {
                            continue;
                        }
                        let bid2 = bx2 as usize + by2 as usize * nx + bz2 as usize * nxy;
                        let level2 = map.levels[bid2];
                        level2_max = level2_max.max(level2);
                        level2_min = level2_min.min(level2);
                    }
                }
            }

            debug_assert!(level2_max as i32 - level as i32 <= 1);
            debug_assert!(level as i32 - level2_min as i32 <= 1);

            if level2_max > level {
                flags.set(BlockFlags::FINE_COARSE);
            }
            if level2_min < level {
                flags.set(BlockFlags::COARSE_FINE);
            }

            (level, flags)
        })
        .collect();

    // Serial: write back to the store and collect the fine-coarse block lists
    // (each block can belong to only one resolution level).
    let mut block_lists: Vec<Vec<usize>> = vec![Vec::new(); n_levels];
    let mut n_act = 0u64;
    for (bid, (level, flags)) in results.into_iter().enumerate() {
        if flags.contains(BlockFlags::FINE_COARSE) {
            block_lists[level as usize].push(bid);
        }
        if flags.contains(BlockFlags::EXISTS) {
            n_act += 1;
        }
        store.level0[bid] = level;
        store.flags[0][bid] = flags;
    }

    Topology {
        blocks_fine_coarse: block_lists,
        n_act_blocks: n_act,
    }
}

/// Initialize the per-voxel cell flags of the level-`level` grid for every block
/// at that level. Mirrors the sparse (`levelMg == 0`) cell-flag init in C++
/// `setRefinementMap` (`msbg.cpp:1804-1883`), without the obstacle paths.
#[allow(clippy::needless_range_loop)]
pub fn init_cell_flags<const BSX: usize, const N: usize>(
    lv: &mut LevelData<BSX, N>,
    level: usize,
    map: &RefinementMap,
    dims: &BlockGridDims,
    store: &BlockInfoStore,
    block_flags: Option<&[BlockFlags]>,
) {
    let bsx = BSX as isize;
    let nbxy = dims.nxy as isize;
    let nbx = dims.nx as isize;
    let bxmax = dims.nx as isize - 1;
    let bymax = dims.ny as isize - 1;
    let bzmax = dims.nz as isize - 1;

    // Phase 1 (serial): materialize the blocks at this level.
    let bids: Vec<usize> = (0..dims.n_blocks)
        .filter(|&bid| store.level0[bid] as usize == level)
        .collect();
    for &bid in &bids {
        lv.ensure_block(bid);
    }

    // Phase 2 (parallel): fill each block's cell flags.
    bids.par_iter().for_each(|&bid| {
        let ptr = lv.cell_flags_ptr_mut(bid);
        if ptr.is_null() {
            return;
        }
        let (bx, by, bz) = dims.coords(bid);
        let bx = bx as isize;
        let by = by as isize;
        let bz = bz as isize;
        let lvl = store.level0[bid];
        let flags = store.flags(bid, 0);
        let is_res_border = flags.is_res_border();
        let exists = block_flags.is_some_and(|f| f[bid].contains(BlockFlags::EXISTS));

        let mut vid = 0usize;
        for vz in 0..bsx {
            let mut bstrid_z = 0isize;
            let mut flags_z = 0u16;
            if vz == 0 {
                flags_z |= CELL_BLK_BORDER;
                if bz != 0 {
                    bstrid_z = -nbxy;
                }
            } else if vz == bsx - 1 {
                flags_z |= CELL_BLK_BORDER;
                if bz != bzmax {
                    bstrid_z = nbxy;
                }
            }
            for vy in 0..bsx {
                let mut bstrid_y = 0isize;
                let mut flags_y = 0u16;
                if vy == 0 {
                    flags_y |= CELL_BLK_BORDER;
                    if by != 0 {
                        bstrid_y = -nbx;
                    }
                } else if vy == bsx - 1 {
                    flags_y |= CELL_BLK_BORDER;
                    if by != bymax {
                        bstrid_y = nbx;
                    }
                }
                for vx in 0..bsx {
                    let mut bstrid_x = 0isize;
                    let mut flags_x = 0u16;
                    if vx == 0 {
                        flags_x |= CELL_BLK_BORDER;
                        if bx != 0 {
                            bstrid_x = -1;
                        }
                    } else if vx == bsx - 1 {
                        flags_x |= CELL_BLK_BORDER;
                        if bx != bxmax {
                            bstrid_x = 1;
                        }
                    }

                    let mut f = flags_x | flags_y | flags_z;

                    if is_res_border {
                        // UPD_TRANS_RES_FLAGS: mark the 7 directional faces facing
                        // a resolution change (3 axes + 3 face diagonals + corner).
                        let mut upd = |bstrid: isize| {
                            if bstrid != 0 {
                                let bid2 = (bid as isize + bstrid) as usize;
                                let level2 = map.levels[bid2];
                                if level2 > lvl {
                                    f |= CELL_FINE_COARSE;
                                }
                                if level2 < lvl {
                                    f |= CELL_COARSE_FINE;
                                }
                            }
                        };
                        upd(bstrid_x);
                        upd(bstrid_y);
                        upd(bstrid_z);
                        upd(bstrid_x + bstrid_y);
                        upd(bstrid_x + bstrid_z);
                        upd(bstrid_y + bstrid_z);
                        upd(bstrid_x + bstrid_y + bstrid_z);
                    }

                    if !exists {
                        f |= CELL_VOID;
                    }

                    unsafe { *ptr.add(vid) = CellFlags(f) };
                    vid += 1;
                }
            }
        }
    });
}

/// Fill `CH_DIST_FINECOARSE` for `BLK_FINE_COARSE` blocks (fine blocks touching
/// coarser blocks): per-voxel distance to the nearest coarser neighbor block,
/// quantized `dist * 1024` and clamped to `dtrans_res`. Mirrors
/// `msbg.cpp:1892-1963`, but SIMD-vectorized across x (the C++ `Vec4f` path).
#[allow(clippy::needless_range_loop)]
pub fn init_dist_fine_coarse<const BSX: usize, const N: usize>(
    lv: &mut LevelData<BSX, N>,
    dtrans_res: usize,
    map: &RefinementMap,
    dims: &BlockGridDims,
    store: &BlockInfoStore,
) {
    use std::simd::num::{SimdFloat, SimdInt};
    use std::simd::{Simd, StdFloat};

    const W: usize = crate::math::simd::LANES;
    let bsx = BSX as f32;
    let dtrans = dtrans_res as f32;
    let scale = 1024.0f32;

    // Phase 1 (serial): materialize the FINE_COARSE blocks.
    let bids: Vec<usize> = (0..dims.n_blocks)
        .filter(|&bid| store.flags(bid, 0).contains(BlockFlags::FINE_COARSE))
        .collect();
    for &bid in &bids {
        lv.ensure_block(bid);
    }

    // Phase 2 (parallel): per voxel distance to the coarser neighbor boxes.
    bids.par_iter().for_each(|&bid| {
        let ptr = lv.dfc_ptr_mut(bid);
        if ptr.is_null() {
            return;
        }
        let lvl = store.level0[bid];
        let (bx, by, bz) = dims.coords(bid);

        // Gather coarser neighbors (level2 > lvl) and precompute their AABBs
        // once, hoisted out of the per-voxel loop.
        let mut boxes: [(f32, f32, f32, f32, f32, f32); 26] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); 26];
        let mut n_boxes = 0usize;
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let (bx2, by2, bz2) = (bx as i32 + dx, by as i32 + dy, bz as i32 + dz);
                    if !dims.in_range(bx2, by2, bz2) {
                        continue;
                    }
                    let bid2 = bx2 as usize + by2 as usize * dims.nx + bz2 as usize * dims.nxy;
                    if map.levels[bid2] <= lvl {
                        continue;
                    }
                    boxes[n_boxes] = (
                        bx2 as f32 * bsx,
                        by2 as f32 * bsx,
                        bz2 as f32 * bsx,
                        (bx2 + 1) as f32 * bsx,
                        (by2 + 1) as f32 * bsx,
                        (bz2 + 1) as f32 * bsx,
                    );
                    n_boxes += 1;
                }
            }
        }

        let mut vid = 0usize;
        for z1 in (bz * BSX)..((bz + 1) * BSX) {
            let pz = z1 as f32 + 0.5;
            for y1 in (by * BSX)..((by + 1) * BSX) {
                let py = y1 as f32 + 0.5;
                if BSX >= W {
                    // Hoist the y/z part of the box distance (invariant over the
                    // x chunks of this row).
                    let mut dy2dz2 = [0.0f32; 26];
                    for i in 0..n_boxes {
                        let (_, y0, z0, _, y1, z1) = boxes[i];
                        let dy = (y0 - py).max(0.0).max(py - y1);
                        let dz = (z0 - pz).max(0.0).max(pz - z1);
                        dy2dz2[i] = dy * dy + dz * dz;
                    }
                    let base = (bx * BSX) as f32 + 0.5;
                    let lane = Simd::<f32, W>::from_array(std::array::from_fn(|k| k as f32));
                    let mut px = Simd::<f32, W>::splat(base) + lane;
                    for _ in 0..(BSX / W) {
                        let mut d2 = Simd::<f32, W>::splat(f32::INFINITY);
                        for i in 0..n_boxes {
                            let (x0, _, _, x1, _, _) = boxes[i];
                            let dx = (Simd::splat(x0) - px)
                                .simd_max(Simd::splat(0.0))
                                .simd_max(px - Simd::splat(x1));
                            d2 = d2.simd_min(dx * dx + Simd::splat(dy2dz2[i]));
                        }
                        let dist = d2.sqrt().simd_min(Simd::splat(dtrans));
                        let q = dist * Simd::splat(scale);
                        // Values are in [0, dtrans*1024] (<= 6144), so the
                        // float->int conversion cannot overflow.
                        let i: Simd<i32, W> = unsafe { q.to_int_unchecked() };
                        let u: Simd<u16, W> = i.cast();
                        unsafe {
                            u.copy_to_slice(std::slice::from_raw_parts_mut(ptr.add(vid) as *mut u16, W));
                        }
                        vid += W;
                        px += Simd::splat(W as f32);
                    }
                } else {
                    for x1 in (bx * BSX)..((bx + 1) * BSX) {
                        let px = x1 as f32 + 0.5;
                        let mut d2 = f32::INFINITY;
                        for i in 0..n_boxes {
                            let (x0, y0, z0, x1b, y1b, z1b) = boxes[i];
                            let dx = (x0 - px).max(0.0).max(px - x1b);
                            let dy = (y0 - py).max(0.0).max(py - y1b);
                            let dz = (z0 - pz).max(0.0).max(pz - z1b);
                            d2 = d2.min(dx * dx + dy * dy + dz * dz);
                        }
                        let mut dist = d2.sqrt();
                        if dist > dtrans {
                            dist = dtrans;
                        }
                        unsafe { *ptr.add(vid) = DistFineCoarse((dist * scale) as u16) };
                        vid += 1;
                    }
                }
            }
        }
    });
}

/// Second distFineCoarse pass (`msbg.cpp:1995-2032`): mark blocks on the coarse
/// side of the interface (adjacent to a same-level `BLK_FINE_COARSE` block) as
/// full so the distance field's interpolation stencil is defined there too.
pub fn propagate_dist_fine_coarse_full<const BSX: usize, const N: usize>(
    lv: &mut LevelData<BSX, N>,
    level: usize,
    dtrans_res: usize,
    dims: &BlockGridDims,
    store: &BlockInfoStore,
) {
    let full = DistFineCoarse((dtrans_res * 1024) as u16);

    let blocks: Vec<usize> = (0..dims.n_blocks)
        .into_par_iter()
        .filter_map(|bid| {
            let lvl = store.level0[bid] as usize;
            if lvl > level || lvl + 1 < level {
                return None;
            }
            if lv.is_value_block(bid) {
                return None;
            }
            let (bx, by, bz) = dims.coords(bid);
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let (bx2, by2, bz2) = (bx as i32 + dx, by as i32 + dy, bz as i32 + dz);
                        if !dims.in_range(bx2, by2, bz2) {
                            continue;
                        }
                        let bid2 = bx2 as usize + by2 as usize * dims.nx + bz2 as usize * dims.nxy;
                        if store.level0[bid2] as usize == level
                            && store.flags(bid2, 0).contains(BlockFlags::FINE_COARSE)
                        {
                            return Some(bid);
                        }
                    }
                }
            }
            None
        })
        .collect();

    for bid in blocks {
        lv.ensure_block(bid);
        let p = lv.dfc_ptr_mut(bid);
        if !p.is_null() {
            unsafe {
                std::slice::from_raw_parts_mut(p, N).fill(full);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 4x4x4 block grid helper.
    fn dims() -> BlockGridDims {
        BlockGridDims::new(4, 4, 4)
    }

    #[test]
    fn test_ref_01_regularize_uniform_unchanged() {
        let d = dims();
        let mut map = vec![0u8; d.n_blocks];
        regularize_refinement_map(&mut map, &d, 3);
        assert_eq!(map, vec![0u8; d.n_blocks]);

        let mut map = vec![2u8; d.n_blocks];
        regularize_refinement_map(&mut map, &d, 3);
        assert_eq!(map, vec![2u8; d.n_blocks]);
    }

    // Happy path: a single fine block in a coarse sea stays legal (drops to the
    // regularization pass untouched), and its neighbors get flagged.
    #[test]
    fn test_ref_02_single_fine_block() {
        let d = dims();
        let mut map = vec![1u8; d.n_blocks];
        map[21] = 0; // block (1,1,1) in a 4^3 grid
        regularize_refinement_map(&mut map, &d, 3);

        let mut store = BlockInfoStore::new(d.n_blocks, 3);
        let topo = compute_block_topology(&mut store, &RefinementMap::new(map), &d, 3, None);

        // Center block is fine with a FINE_COARSE border toward its coarse neighbors.
        assert_eq!(store.level0[21], 0);
        assert!(store.flags(21, 0).contains(BlockFlags::FINE_COARSE));
        assert!(!store.flags(21, 0).contains(BlockFlags::COARSE_FINE));
        assert_eq!(topo.blocks_fine_coarse[0], vec![21]);

        // A far-away coarse block (3,3,3) has no resolution border.
        assert_eq!(store.level0[63], 1);
        assert!(!store.flags(63, 0).is_res_border());
    }

    // Boundary: a >1 level jump is regularized to a legal staircase.
    #[test]
    fn test_ref_03_regularize_large_jump() {
        let d = dims();
        let mut map = vec![2u8; d.n_blocks];
        map[21] = 0; // fine island in a level-2 sea: jump of 2
        regularize_refinement_map(&mut map, &d, 3);

        // Neighbors of the island must be lifted to level 1.
        let mut store = BlockInfoStore::new(d.n_blocks, 3);
        compute_block_topology(&mut store, &RefinementMap::new(map.clone()), &d, 3, None);

        for bid in 0..d.n_blocks {
            let (bx, by, bz) = d.coords(bid);
            // Check the 2:1 invariant across all in-range neighbor pairs.
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (bx2, by2, bz2) = (bx as i32 + dx, by as i32 + dy, bz as i32 + dz);
                        if !d.in_range(bx2, by2, bz2) || (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        let bid2 = bx2 as usize + by2 as usize * d.nx + bz2 as usize * d.nxy;
                        let diff = (map[bid] as i32 - map[bid2] as i32).abs();
                        assert!(diff <= 1, "level jump {diff} at {bid}->{bid2}");
                    }
                }
            }
        }
    }

    // Boundary: domain corner block gets BLK_DOM_BORDER.
    #[test]
    fn test_ref_04_dom_border() {
        let d = dims();
        let map = vec![0u8; d.n_blocks];
        let mut store = BlockInfoStore::new(d.n_blocks, 3);
        compute_block_topology(&mut store, &RefinementMap::new(map), &d, 3, None);
        assert!(store.flags(0, 0).contains(BlockFlags::DOM_BORDER));
        assert!(store.flags(d.n_blocks - 1, 0).contains(BlockFlags::DOM_BORDER));
        // Interior block (1,1,1) is not on the domain border.
        assert!(!store.flags(21, 0).contains(BlockFlags::DOM_BORDER));
    }

    // Boundary: n_levels == 1 and 2 must not panic in regularize.
    #[test]
    fn test_ref_05_small_level_counts() {
        let d = dims();
        let mut map = vec![0u8; d.n_blocks];
        regularize_refinement_map(&mut map, &d, 1);
        regularize_refinement_map(&mut map, &d, 2);
        assert_eq!(map, vec![0u8; d.n_blocks]);
    }

    // Awkward: in-place regularization with an all-fine map is idempotent.
    #[test]
    fn test_ref_06_idempotent() {
        let d = dims();
        let mut map = vec![0u8; d.n_blocks];
        regularize_refinement_map(&mut map, &d, 3);
        let once = map.clone();
        regularize_refinement_map(&mut map, &d, 3);
        assert_eq!(map, once);
    }
}

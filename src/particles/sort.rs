//! Particle placement and block bucketing.
//!
//! Placement replicates the C++ `msbg_test_sparse` loop exactly: each instance
//! `i` has an origin `0.2*sxyzMax + 0.6*sxyzMax*baseScale*(base[i]-bboxMin)`
//! and scale `sxyzMin*factor`; each base point is placed as
//! `origin + scale*baseScale*(base[j]-bboxMin)`, domain-checked, truncated, and
//! assigned to its block. Bucketing is an O(n) counting sort into per-block
//! contiguous position slices (no atomics, no per-block linked lists), then a
//! CSR start-offset array for the splat.

use rayon::prelude::*;

use super::{footprint_axis, DomainBounds, GridDims, SurfaceConfig};

/// Center block id of a truncated in-domain position.
#[inline(always)]
fn center_block(ipos: [i32; 3], dims: &GridDims, bsx_log2: u32) -> u32 {
    let bx = (ipos[0] >> bsx_log2) as usize;
    let by = (ipos[1] >> bsx_log2) as usize;
    let bz = (ipos[2] >> bsx_log2) as usize;
    (bx + by * dims.nx + bz * dims.nxy) as u32
}

/// Place every instance's particles, assign each to its center block, and
/// collect the footprint blocks inline (a single pass — the C++ path does the
/// same with atomics; here the footprint list is sorted/deduped at the end).
pub fn place(
    base: &[[f32; 3]],
    bbox_min: &[f32; 3],
    span_max: f32,
    dims: &GridDims,
    domain: &DomainBounds,
    cfg: &SurfaceConfig,
) -> super::Placed {
    let base_scale = 1.0 / span_max;
    let scale = dims.sxyz_min() * cfg.instance_scale_factor;
    let inst_scale = scale * base_scale;
    let sxyz_max = dims.sxyz_max();
    let bsx_log2 = dims.bsx.trailing_zeros();
    let r_scan_bsx = cfg.r_scan() / dims.bsx as f32;
    let n_instances = if cfg.n_instances == 0 {
        base.len()
    } else {
        cfg.n_instances
    };
    let bbox = *bbox_min;

    // One (positions, bids, footprint-blocks) accumulator per rayon worker.
    type Acc = (Vec<[f32; 3]>, Vec<u32>, Vec<u32>);
    let (positions, bids, mut footprint): Acc = (0..n_instances)
        .into_par_iter()
        .fold(
            || (Vec::with_capacity(base.len()), Vec::with_capacity(base.len()), Vec::new()),
            |(mut pos, mut bids, mut fp): Acc, i| {
                let bi = &base[i];
                // origin uses `sxyz_max * base_scale`; the instance offset is
                // `origin + inst_scale * (bj - bbox)` (non-fused, as the C++
                // expression tree; g++ may contract under -ffast-math but the
                // difftest tolerance absorbs the 1-ulp difference).
                let origin = [
                    0.2 * sxyz_max + 0.6 * sxyz_max * base_scale * (bi[0] - bbox[0]),
                    0.2 * sxyz_max + 0.6 * sxyz_max * base_scale * (bi[1] - bbox[1]),
                    0.2 * sxyz_max + 0.6 * sxyz_max * base_scale * (bi[2] - bbox[2]),
                ];
                for bj in base {
                    let p = [
                        origin[0] + inst_scale * (bj[0] - bbox[0]),
                        origin[1] + inst_scale * (bj[1] - bbox[1]),
                        origin[2] + inst_scale * (bj[2] - bbox[2]),
                    ];
                    if !domain.contains(&p) {
                        continue;
                    }
                    let ipos = [
                        p[0] as i32, // trunc toward zero == `truncate_to_int`
                        p[1] as i32,
                        p[2] as i32,
                    ];
                    if ipos[0] < 0
                        || ipos[1] < 0
                        || ipos[2] < 0
                        || ipos[0] >= dims.sx as i32
                        || ipos[1] >= dims.sy as i32
                        || ipos[2] >= dims.sz as i32
                    {
                        continue;
                    }
                    let bid = center_block(ipos, dims, bsx_log2);
                    pos.push(p);
                    bids.push(bid);
                    // Footprint blocks: trunc(bpos ± rScan/bsx) clipped to the
                    // block grid, the C++ `scale2DestBlockGrid` form.
                    let (x1, x2) = footprint_axis(p[0], r_scan_bsx, dims.bsx, dims.nx);
                    let (y1, y2) = footprint_axis(p[1], r_scan_bsx, dims.bsx, dims.ny);
                    let (z1, z2) = footprint_axis(p[2], r_scan_bsx, dims.bsx, dims.nz);
                    for bz_ in z1..=z2 {
                        for by_ in y1..=y2 {
                            for bx_ in x1..=x2 {
                                fp.push(
                                    (bx_ as usize + by_ as usize * dims.nx + bz_ as usize * dims.nxy)
                                        as u32,
                                );
                            }
                        }
                    }
                }
                (pos, bids, fp)
            },
        )
        .reduce(
            || (Vec::new(), Vec::new(), Vec::new()),
            |(mut a, mut b, mut c), (p, q, r)| {
                a.extend(p);
                b.extend(q);
                c.extend(r);
                (a, b, c)
            },
        );

    footprint.par_sort_unstable();
    footprint.dedup();

    super::Placed {
        positions,
        bids,
        active: footprint,
    }
}

/// Bucketed particles: positions grouped by center block, plus a CSR start
/// offset array (`block b` occupies `positions[start[b]..start[b+1]]`).
pub struct Bucketed {
    pub positions: Vec<[f32; 3]>,
    pub block_start: Vec<u32>,
    /// Blocks with at least one particle, sorted by block id.
    pub particle_blocks: Vec<u32>,
}

/// O(n) counting sort of `(positions, bids)` into block-major order, with
/// `particle_blocks` in ascending block-id order.
pub fn bucket_by_block(positions: Vec<[f32; 3]>, bids: Vec<u32>, n_blocks: usize) -> Bucketed {
    let n = positions.len();
    debug_assert_eq!(n, bids.len());

    let mut counts = vec![0u32; n_blocks];
    for &b in &bids {
        counts[b as usize] += 1;
    }

    let mut block_start = vec![0u32; n_blocks + 1];
    let mut acc = 0u32;
    for (b, c) in counts.iter().enumerate() {
        block_start[b] = acc;
        acc += c;
    }
    block_start[n_blocks] = acc;
    debug_assert_eq!(acc as usize, n);

    let mut ordered = vec![[0.0f32; 3]; n];
    let mut cursor = block_start[..n_blocks].to_vec();
    for j in 0..n {
        let b = bids[j] as usize;
        let out = cursor[b] as usize;
        ordered[out] = positions[j];
        cursor[b] += 1;
    }

    let particle_blocks: Vec<u32> = (0..n_blocks)
        .filter(|&b| counts[b] > 0)
        .map(|b| b as u32)
        .collect();

    Bucketed {
        positions: ordered,
        block_start,
        particle_blocks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(s: usize) -> GridDims {
        GridDims::new(s, s, s, 16)
    }

    fn cfg(n: usize, scale: f32) -> SurfaceConfig {
        SurfaceConfig {
            sx: 64,
            sy: 64,
            sz: 64,
            n_instances: n,
            instance_scale_factor: scale,
            r_particle: 2.0,
            nb_dist: 2.0,
            n_smooth_iters: 0,
            smooth_dt: 0.05,
        }
    }

    // A single instance: all base points land in-domain, same count out.
    #[test]
    fn place_01_single_instance_counts() {
        let base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.5], [0.25, 0.25, 0.75]];
        let d = dims(64);
        let p = place(&base, &[0.0, 0.0, 0.0], 1.0, &d, &DomainBounds::new(&d), &cfg(1, 0.5));
        assert_eq!(p.positions.len(), base.len());
        assert_eq!(p.bids.len(), base.len());
    }

    // A base point exactly at the domain edge after placement is dropped or
    // clamped identically to C++ (`trunc` then in-range check).
    #[test]
    fn place_02_domain_edge_clipping() {
        // Instance origin near 0.2*64=12.8; a base point pushed past 64 drops.
        let base = [[0.0, 0.0, 0.0], [0.999, 0.0, 0.0]];
        let d = dims(64);
        let p = place(&base, &[0.0, 0.0, 0.0], 1.0, &d, &DomainBounds::new(&d), &cfg(1, 100.0));
        // scale = 64*100 = 6400 voxels: the instance spans far beyond the grid.
        assert!(p.positions.len() < base.len());
    }

    // Degenerate base: span 0 is rejected by reconstruct_surface (tested there),
    // but `place` must not panic.
    #[test]
    fn place_03_zero_span_does_not_panic() {
        let base = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let d = dims(64);
        let p = place(&base, &[1.0, 2.0, 3.0], 0.0, &d, &DomainBounds::new(&d), &cfg(1, 0.5));
        // span_max = 0 -> base_scale = inf -> positions are inf; domain check
        // rejects them all. Callers guard span before calling.
        let _ = p;
    }

    // n_instances=0 means "one per base point" (the demo).
    #[test]
    fn place_04_zero_instances_means_all() {
        let base = [[0.0, 0.0, 0.0]];
        let d = dims(64);
        let p = place(&base, &[0.0, 0.0, 0.0], 1.0, &d, &DomainBounds::new(&d), &cfg(0, 0.5));
        assert_eq!(p.positions.len(), 1);
    }

    #[test]
    fn bucket_01_contiguous_sorted_per_block() {
        let positions = vec![[0.0; 3], [1.0; 3], [2.0; 3], [3.0; 3]];
        let bids = vec![1, 0, 1, 0];
        let b = bucket_by_block(positions, bids, 2);
        assert_eq!(b.block_start, vec![0, 2, 4]);
        // Block 0 holds positions 0,1 in order; block 1 holds 2,3.
        assert_eq!(b.particle_blocks, vec![0, 1]);
    }

    #[test]
    fn bucket_02_empty_block_slices() {
        let positions = vec![[5.0; 3]];
        let bids = vec![3];
        let b = bucket_by_block(positions, bids, 8);
        assert_eq!(b.block_start, vec![0, 0, 0, 0, 1, 1, 1, 1, 1]);
        assert_eq!(b.particle_blocks, vec![3]);
        assert_eq!(b.positions[0], [5.0; 3]);
    }

    #[test]
    fn bucket_03_all_same_block() {
        let positions = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
        let bids = vec![0, 0, 0];
        let b = bucket_by_block(positions, bids, 1);
        assert_eq!(b.block_start, vec![0, 3]);
        assert_eq!(b.positions.len(), 3);
    }

    #[test]
    fn bucket_04_radix_equals_reference_order() {
        // Counting sort must reproduce a stable reference ordering.
        let mut rng = 0x1234_5678u32;
        let next = |rng: &mut u32| {
            *rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *rng >> 8
        };
        let n = 100_000usize;
        let n_blocks = 64usize;
        let positions: Vec<[f32; 3]> = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
        let bids: Vec<u32> = (0..n).map(|_| next(&mut rng) % n_blocks as u32).collect();
        let b = bucket_by_block(positions.clone(), bids.clone(), n_blocks);

        // Reference: stable sort by bid.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| bids[i]);
        let mut expected = vec![[0.0; 3]; n];
        for (k, &i) in order.iter().enumerate() {
            expected[k] = positions[i];
        }
        assert_eq!(b.positions, expected);
        // block_start consistent with the reference.
        for blk in 0..n_blocks {
            let lo = order.iter().position(|&i| bids[i] as usize == blk).unwrap_or(n);
            let hi = order.iter().rposition(|&i| bids[i] as usize == blk).map_or(lo, |p| p + 1);
            assert_eq!(b.block_start[blk], lo as u32);
            assert_eq!(b.block_start[blk + 1], hi as u32);
        }
    }
}

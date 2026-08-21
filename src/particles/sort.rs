//! Particle placement and block bucketing.
//!
//! Placement replicates the C++ `msbg_test_sparse` loop exactly: each instance
//! `i` has an origin `0.2*sxyzMax + 0.6*sxyzMax*baseScale*(base[i]-bboxMin)`
//! and scale `sxyzMin*factor`; each base point is placed as
//! `origin + scale*baseScale*(base[j]-bboxMin)`, domain-checked, truncated, and
//! assigned to its block. Bucketing is an O(n) counting sort into per-block
//! contiguous position slices (no atomics, no per-block linked lists), then a
//! CSR start-offset array for the splat.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::blockmap::{BlockMap, BlockSet};

use super::{footprint_axis, DomainBounds, GridDims, SurfaceConfig};

/// Center block id of a truncated in-domain position.
#[inline(always)]
fn center_block(ipos: [i32; 3], dims: &GridDims, bsx_log2: u32) -> usize {
    let bx = (ipos[0] >> bsx_log2) as usize;
    let by = (ipos[1] >> bsx_log2) as usize;
    let bz = (ipos[2] >> bsx_log2) as usize;
    bx + by * dims.nx + bz * dims.nxy
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

    // One (positions, bids, footprint-set) accumulator per rayon worker. The
    // footprint is a sparse `BlockSet`, not a per-particle `Vec`, so the
    // active-block union scales with *occupied* blocks — at 32,768³/block-16 a
    // flat footprint Vec would be up to 8 × 1.29B entries (~82 GB) before the
    // dedup, where the set is ~25M keys (~0.2 GB).
    type Acc = (Vec<[f32; 3]>, Vec<usize>, BlockSet);
    let (positions, bids, footprint): Acc = (0..n_instances)
        .into_par_iter()
        .fold(
            || {
                (
                    Vec::with_capacity(base.len()),
                    Vec::with_capacity(base.len()),
                    BlockSet::new(),
                )
            },
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
                                fp.insert(
                                    bx_ as usize + by_ as usize * dims.nx + bz_ as usize * dims.nxy,
                                    (),
                                );
                            }
                        }
                    }
                }
                (pos, bids, fp)
            },
        )
        .reduce(
            || (Vec::new(), Vec::new(), BlockSet::new()),
            |(mut a, mut b, mut c), (p, q, r)| {
                a.extend(p);
                b.extend(q);
                for (k, _) in r.iter() {
                    c.insert(k, ());
                }
                (a, b, c)
            },
        );

    let mut active: Vec<usize> = footprint.iter().map(|(k, _)| k).collect();
    active.par_sort_unstable();

    super::Placed {
        positions,
        bids,
        active,
    }
}

/// Bucketed particles: positions grouped by center block (block-major order),
/// plus a compact CSR start-offset array. `particle_blocks[i]`'s particles
/// occupy `positions[starts[i]..starts[i+1]]`.
pub struct Bucketed {
    pub positions: Vec<[f32; 3]>,
    /// Blocks with at least one particle, sorted by block id.
    pub particle_blocks: Vec<usize>,
    /// CSR start offsets aligned to `particle_blocks` (len = blocks + 1).
    pub starts: Vec<usize>,
}

/// Sparse counting sort of `(positions, bids)` into block-major order.
///
/// Unlike the dense C++/first-cut approach (a `counts`/`block_start` array of
/// size `n_blocks` — 68.7 GB each at 32,768³/block-16), the histogram is keyed
/// by block id in an open-addressed `BlockMap`, so cost scales with *occupied*
/// blocks (~25M at paper scale, not 8.59B). The parallel scatter uses a
/// dense-by-rank `AtomicU32` cursor — rank ≤ occupied blocks, ~100 MB.
pub fn bucket_by_block(positions: Vec<[f32; 3]>, bids: Vec<usize>) -> Bucketed {
    let n = positions.len();
    debug_assert_eq!(n, bids.len());
    if n == 0 {
        return Bucketed {
            positions,
            particle_blocks: Vec::new(),
            starts: vec![0],
        };
    }

    let profile = std::env::var("MSBG_BUCKET_PROFILE").is_ok();
    let t0 = std::time::Instant::now();

    // Parallel sparse histogram. Explicit one-chunk-per-thread partitioning:
    // rayon's `fold` thief-splitting creates thousands of tiny accumulators
    // (measured ~2685 at 10M bids), turning the `reduce` merge into O(N·log).
    let n_threads = rayon::current_num_threads();
    let chunk = n.div_ceil(n_threads);
    let maps: Vec<BlockMap<u32>> = bids
        .par_chunks(chunk)
        .map(|chunk| {
            let mut m = BlockMap::with_capacity(chunk.len() / 64 + 16);
            for &b in chunk {
                m.update(b, 1, |v| v + 1);
            }
            m
        })
        .collect();

    // Merge the per-thread maps into ONE pre-sized map. Blocks are almost
    // disjoint across threads (instances are round-robin), so capacity ~= the
    // summed distinct counts; a `reduce` tree would instead grow each
    // intermediate accumulator from capacity 4, rehashing ~16.8M entries
    // through ~23 doublings (the perf hotspot: `BlockMap::insert` was ~50%).
    let total_cap = maps.iter().map(|m| m.len()).sum::<usize>() + 16;
    let mut counts = BlockMap::with_capacity(total_cap);
    for m in &maps {
        for (k, v) in m.iter() {
            counts.update(k, v, |old| old + v);
        }
    }
    let t1 = std::time::Instant::now();

    let pairs = counts.sorted_pairs();
    let t2 = std::time::Instant::now();
    let nblk = pairs.len();
    let mut particle_blocks = Vec::with_capacity(nblk);
    let mut starts = Vec::with_capacity(nblk + 1);
    let mut block_rank = BlockMap::with_capacity(nblk);
    starts.push(0usize);
    let mut acc = 0usize;
    for (i, (bid, count)) in pairs.into_iter().enumerate() {
        particle_blocks.push(bid);
        block_rank.insert(bid, i as u32);
        acc += count as usize;
        starts.push(acc);
    }
    debug_assert_eq!(acc, n);
    let t3 = std::time::Instant::now();

    // Parallel scatter into block-major order. Ranks own disjoint output
    // ranges (`starts[rank] + [0, count)`), so the atomic cursor is race-free
    // by construction; intra-block order is unspecified (the splat's `min`
    // reduction is order-independent).
    let mut ordered = vec![[0.0f32; 3]; n];
    let cursors: Vec<AtomicU32> = (0..nblk).map(|_| AtomicU32::new(0)).collect();
    let out_ptr = ordered.as_mut_ptr() as usize;
    positions
        .par_iter()
        .zip(bids.par_iter())
        .for_each(|(&pos, &bid)| {
            let rank = block_rank
                .get(bid)
                .unwrap_or_else(|| panic!("block {bid} missing from the histogram")) as usize;
            let off = cursors[rank].fetch_add(1, Ordering::Relaxed) as usize;
            // SAFETY: `starts[rank] + off` is a unique index per (rank, off);
            // ranges across ranks are disjoint, so no two writes alias.
            let out_ptr = out_ptr as *mut [f32; 3];
            unsafe {
                *out_ptr.add(starts[rank] + off) = pos;
            }
        });
    if profile {
        let t4 = std::time::Instant::now();
        eprintln!(
            "[bucket] n={n} nblk={nblk} hist={:.3}s sort={:.3}s rank={:.3}s scatter={:.3}s",
            t1.duration_since(t0).as_secs_f64(),
            t2.duration_since(t1).as_secs_f64(),
            t3.duration_since(t2).as_secs_f64(),
            t4.duration_since(t3).as_secs_f64(),
        );
    }

    Bucketed {
        positions: ordered,
        particle_blocks,
        starts,
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
        let b = bucket_by_block(positions, bids);
        assert_eq!(b.particle_blocks, vec![0, 1]);
        assert_eq!(b.starts, vec![0, 2, 4]);
        // Block 0 holds {pos[1], pos[3]} (unspecified order), block 1 {pos[0], pos[2]}.
        let mut b0 = b.positions[0..2].to_vec();
        b0.sort_unstable_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert_eq!(b0, vec![[1.0; 3], [3.0; 3]]);
        let mut b1 = b.positions[2..4].to_vec();
        b1.sort_unstable_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert_eq!(b1, vec![[0.0; 3], [2.0; 3]]);
    }

    #[test]
    fn bucket_02_sparse_single_occupied_block() {
        let positions = vec![[5.0; 3]];
        let bids = vec![3];
        let b = bucket_by_block(positions, bids);
        assert_eq!(b.particle_blocks, vec![3]);
        assert_eq!(b.starts, vec![0, 1]);
        assert_eq!(b.positions[0], [5.0; 3]);
    }

    #[test]
    fn bucket_03_all_same_block() {
        let positions = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
        let bids = vec![0, 0, 0];
        let b = bucket_by_block(positions, bids);
        assert_eq!(b.particle_blocks, vec![0]);
        assert_eq!(b.starts, vec![0, 3]);
        assert_eq!(b.positions.len(), 3);
    }

    #[test]
    fn bucket_04_empty_input() {
        let b = bucket_by_block(vec![], vec![]);
        assert!(b.positions.is_empty());
        assert!(b.particle_blocks.is_empty());
        assert_eq!(b.starts, vec![0]);
    }

    #[test]
    fn bucket_05_u64_scale_bids() {
        // Block ids straddling u32::MAX must round-trip (64-bit addressing).
        let bids = vec![
            u32::MAX as usize,
            0usize,
            u32::MAX as usize + 1,
            0x1_0000_0001,
            u32::MAX as usize,
        ];
        let positions: Vec<[f32; 3]> = bids.iter().map(|&b| [b as f32, 0.0, 0.0]).collect();
        let b = bucket_by_block(positions, bids);
        let mut expected = vec![
            0usize,
            u32::MAX as usize,
            u32::MAX as usize + 1,
            0x1_0000_0001,
        ];
        expected.sort_unstable();
        assert_eq!(b.particle_blocks, expected);
        let idx = b.particle_blocks.iter().position(|&k| k == u32::MAX as usize).unwrap();
        assert_eq!(b.starts[idx + 1] - b.starts[idx], 2);
    }

    #[test]
    fn bucket_06_sparse_equals_dense_reference() {
        // Equivalence oracle: randomized inputs match the dense counting sort
        // (kept below as the reference). Intra-block order is unspecified in
        // the sparse scatter, so contents are compared as multisets.
        let mut rng = 0x9e37_79b9u32;
        let next = |rng: &mut u32| {
            *rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *rng >> 8
        };
        for (n, n_blocks) in [(0usize, 1usize), (1, 1), (1000, 64), (10_000, 1024)] {
            let positions: Vec<[f32; 3]> = (0..n)
                .map(|i| [i as f32, (i % 7) as f32, (i % 13) as f32])
                .collect();
            let bids: Vec<usize> = (0..n).map(|_| (next(&mut rng) as usize) % n_blocks).collect();
            let dense = bucket_by_block_dense(positions.clone(), bids.clone(), n_blocks);
            let sparse = bucket_by_block(positions.clone(), bids.clone());

            assert_eq!(sparse.particle_blocks, dense.1, "particle_blocks differ at n={n}");
            assert_eq!(sparse.starts, dense.2, "starts differ at n={n}");

            for (i, &bid) in sparse.particle_blocks.iter().enumerate() {
                let (s_lo, s_hi) = (sparse.starts[i], sparse.starts[i + 1]);
                let (d_lo, d_hi) = (dense.2[i], dense.2[i + 1]);
                let mut sp = sparse.positions[s_lo..s_hi].to_vec();
                let mut dp = dense.0[d_lo..d_hi].to_vec();
                sp.sort_unstable_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
                dp.sort_unstable_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
                assert_eq!(sp, dp, "block {bid} contents differ at n={n}");
            }
        }
    }

    /// Dense counting sort (the pre-step-12 implementation), kept test-only as
    /// the equivalence oracle. Returns `(ordered, particle_blocks, compact_starts)`.
    fn bucket_by_block_dense(
        positions: Vec<[f32; 3]>,
        bids: Vec<usize>,
        n_blocks: usize,
    ) -> (Vec<[f32; 3]>, Vec<usize>, Vec<usize>) {
        let n = positions.len();
        let mut counts = vec![0usize; n_blocks];
        for &b in &bids {
            counts[b] += 1;
        }
        let mut block_start = vec![0usize; n_blocks + 1];
        let mut acc = 0usize;
        for (b, c) in counts.iter().enumerate() {
            block_start[b] = acc;
            acc += c;
        }
        block_start[n_blocks] = acc;

        let mut ordered = vec![[0.0f32; 3]; n];
        let mut cursor = block_start[..n_blocks].to_vec();
        for j in 0..n {
            let b = bids[j];
            let out = cursor[b];
            ordered[out] = positions[j];
            cursor[b] += 1;
        }

        let particle_blocks: Vec<usize> = (0..n_blocks).filter(|&b| counts[b] > 0).collect();
        let starts: Vec<usize> = particle_blocks
            .iter()
            .map(|&b| block_start[b])
            .chain(std::iter::once(block_start[n_blocks]))
            .collect();

        (ordered, particle_blocks, starts)
    }
}

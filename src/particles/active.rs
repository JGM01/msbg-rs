//! Active-block determination: the union of every placed particle's footprint
//! block range, replicated exactly from the C++ `msbg_test_sparse` marking
//! (`trunc(bpos ± rScan/bsx)` clipped to the block grid). Per-thread sparse
//! sets are merged and sorted — no atomics (C++ uses one `InterlockedIncrement`
//! per footprint block per particle into a dense `blockActive` array, which is
//! 68.7 GB at 32,768³/block-16), and no per-particle `Vec`.

use rayon::prelude::*;

use crate::blockmap::BlockSet;

use super::{footprint_axis, GridDims, SurfaceConfig};

/// Union of all particles' footprint blocks, sorted and deduplicated.
pub fn active_blocks(positions: &[[f32; 3]], dims: &GridDims, cfg: &SurfaceConfig) -> Vec<usize> {
    let r_scan_bsx = cfg.r_scan() / dims.bsx as f32;
    let chunk = dims.nxy;

    let set: BlockSet = positions
        .par_chunks(4096)
        .fold(BlockSet::new, |mut local: BlockSet, chunk_| {
            for p in chunk_ {
                let (x1, x2) = footprint_axis(p[0], r_scan_bsx, dims.bsx, dims.nx);
                let (y1, y2) = footprint_axis(p[1], r_scan_bsx, dims.bsx, dims.ny);
                let (z1, z2) = footprint_axis(p[2], r_scan_bsx, dims.bsx, dims.nz);
                for bz in z1..=z2 {
                    for by in y1..=y2 {
                        for bx in x1..=x2 {
                            let bid = (bx as usize) + (by as usize) * dims.nx + (bz as usize) * chunk;
                            local.insert(bid, ());
                        }
                    }
                }
            }
            local
        })
        .reduce(BlockSet::new, |mut a, b| {
            for (k, _) in b.iter() {
                a.insert(k, ());
            }
            a
        });

    let mut active: Vec<usize> = set.iter().map(|(k, _)| k).collect();
    active.par_sort_unstable();
    active
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(s: usize) -> GridDims {
        GridDims::new(s, s, s, 16)
    }

    fn cfg() -> SurfaceConfig {
        SurfaceConfig {
            sx: 64,
            sy: 64,
            sz: 64,
            n_instances: 1,
            instance_scale_factor: 0.5,
            r_particle: 2.0,
            nb_dist: 2.0,
            n_smooth_iters: 0,
            smooth_dt: 0.05,
        }
    }

    // A particle at the center of its block only touches its own block.
    #[test]
    fn active_01_center_particle_single_block() {
        let d = dims(64); // block 0 covers voxels 0..15
        let positions = [[8.0, 8.0, 8.0]];
        let a = active_blocks(&positions, &d, &cfg());
        assert_eq!(a, vec![0]);
    }

    // A particle near a block boundary touches both blocks (the case a
    // 1-voxel splat halo would get wrong).
    #[test]
    fn active_02_boundary_particle_two_blocks() {
        let d = dims(64); // blocks 0..3 per axis
        let positions = [[15.5, 8.0, 8.0]]; // in block 0, rScan=4 reaches block 1
        let a = active_blocks(&positions, &d, &cfg());
        assert_eq!(a, vec![0, 1]);
    }

    // Domain-corner particle: footprint clipped to the grid.
    #[test]
    fn active_03_domain_corner_clipping() {
        let d = dims(64);
        let positions = [[0.1, 0.1, 0.1]];
        let a = active_blocks(&positions, &d, &cfg());
        assert_eq!(a, vec![0]);
    }

    // Domain-edge particle near the far corner: clipping keeps bids valid.
    #[test]
    fn active_04_far_corner_clipping() {
        let d = dims(64); // nx=ny=nz=4, voxels 0..63
        let positions = [[63.0, 63.0, 63.0]];
        let a = active_blocks(&positions, &d, &cfg());
        let (bx, by, bz) = d.coords(a[0] as usize);
        assert_eq!((bx, by, bz), (3, 3, 3));
    }

    // Empty particle set -> empty active set.
    #[test]
    fn active_05_empty_is_empty() {
        let d = dims(64);
        let a = active_blocks(&[], &d, &cfg());
        assert!(a.is_empty());
    }

    // Dedup: two particles in the same block give one entry.
    #[test]
    fn active_06_dedup() {
        let d = dims(64);
        let positions = [[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]];
        let a = active_blocks(&positions, &d, &cfg());
        assert_eq!(a, vec![0]);
    }

    // Asymmetric grid dims do not alias block ids.
    #[test]
    fn active_07_asymmetric_dims() {
        let d = GridDims::new(32, 48, 16, 16); // nx=2 ny=3 nz=1
        let positions = [[17.0, 33.0, 8.0]]; // center block (1,2,0) = 5
        let a = active_blocks(&positions, &d, &cfg());
        // Footprint x: 17/16=1.0625 ± 0.25 -> trunc gives 0..1; y: 33/16=2.0625
        // ± 0.25 -> 1..2; z stays 0. Blocks {0,1}x{1,2}x{0} = 2,3,4,5.
        assert_eq!(a, vec![2, 3, 4, 5]);
    }

    // Block ids exceeding u32::MAX (64-bit addressing) must survive the sparse
    // set round-trip without truncation.
    #[test]
    fn active_08_u64_scale_bids() {
        let d = GridDims::new(1 << 21, 1 << 21, 1 << 21, 16); // 2^51 virtual blocks
        let positions = [[(1 << 21) as f32 - 8.0; 3]];
        let a = active_blocks(&positions, &d, &cfg());
        assert!(!a.is_empty());
        assert!(a.iter().any(|&b| b > u32::MAX as usize));
    }

    // The merged union must be identical regardless of how many rayon threads
    // split the work (no per-thread non-determinism leaks into the result).
    #[test]
    fn active_09_deterministic_across_thread_counts() {
        let d = dims(64);
        let positions: Vec<[f32; 3]> = (0..4096)
            .map(|i| {
                [
                    (i % 64) as f32 + 0.5,
                    ((i / 64) % 64) as f32 + 0.5,
                    (i / 4096) as f32 + 0.5,
                ]
            })
            .collect();
        let single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let multi = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        let a1 = single.install(|| active_blocks(&positions, &d, &cfg()));
        let a8 = multi.install(|| active_blocks(&positions, &d, &cfg()));
        assert_eq!(a1, a8);
    }

    // Equivalence oracle: the sparse union matches the dense footprint Vec +
    // sort/dedup reference (kept below) on randomized input.
    #[test]
    fn active_10_sparse_equals_dense_reference() {
        let mut rng = 0x1234_5678u32;
        let next = |rng: &mut u32| {
            *rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *rng >> 8
        };
        let d = dims(128);
        let positions: Vec<[f32; 3]> = (0..5000)
            .map(|_| [
                (next(&mut rng) % 128) as f32,
                (next(&mut rng) % 128) as f32,
                (next(&mut rng) % 128) as f32,
            ])
            .collect();
        assert_eq!(active_blocks(&positions, &d, &cfg()), active_blocks_dense(&positions, &d, &cfg()));
    }

    /// Dense footprint Vec + sort/dedup reference (the pre-step-12 path).
    fn active_blocks_dense(positions: &[[f32; 3]], dims: &GridDims, cfg: &SurfaceConfig) -> Vec<usize> {
        let r_scan_bsx = cfg.r_scan() / dims.bsx as f32;
        let mut merged: Vec<usize> = Vec::new();
        for p in positions {
            let (x1, x2) = footprint_axis(p[0], r_scan_bsx, dims.bsx, dims.nx);
            let (y1, y2) = footprint_axis(p[1], r_scan_bsx, dims.bsx, dims.ny);
            let (z1, z2) = footprint_axis(p[2], r_scan_bsx, dims.bsx, dims.nz);
            for bz in z1..=z2 {
                for by in y1..=y2 {
                    for bx in x1..=x2 {
                        merged.push((bx as usize) + (by as usize) * dims.nx + (bz as usize) * dims.nxy);
                    }
                }
            }
        }
        merged.par_sort_unstable();
        merged.dedup();
        merged
    }
}

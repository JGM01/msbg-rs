//! Active-block determination: the union of every placed particle's footprint
//! block range, replicated exactly from the C++ `msbg_test_sparse` marking
//! (`trunc(bpos ± rScan/bsx)` clipped to the block grid). Per-thread local
//! lists are sorted and merged — no atomics (C++ uses one `InterlockedIncrement`
//! per footprint block per particle).

use rayon::prelude::*;

use super::{footprint_axis, GridDims, SurfaceConfig};

/// Union of all particles' footprint blocks, sorted and deduplicated.
pub fn active_blocks(positions: &[[f32; 3]], dims: &GridDims, cfg: &SurfaceConfig) -> Vec<usize> {
    let r_scan_bsx = cfg.r_scan() / dims.bsx as f32;
    let chunk = dims.nxy;

    let mut merged: Vec<usize> = positions
        .par_chunks(4096)
        .fold(Vec::new, |mut local: Vec<usize>, chunk_| {
            for p in chunk_ {
                let (x1, x2) = footprint_axis(p[0], r_scan_bsx, dims.bsx, dims.nx);
                let (y1, y2) = footprint_axis(p[1], r_scan_bsx, dims.bsx, dims.ny);
                let (z1, z2) = footprint_axis(p[2], r_scan_bsx, dims.bsx, dims.nz);
                for bz in z1..=z2 {
                    for by in y1..=y2 {
                        for bx in x1..=x2 {
                            let bid = (bx as usize) + (by as usize) * dims.nx + (bz as usize) * chunk;
                            local.push(bid);
                        }
                    }
                }
            }
            local
        })
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

    merged.par_sort_unstable();
    merged.dedup();
    merged
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
}

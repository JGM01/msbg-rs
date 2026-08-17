//! Differential test for the step-9 surface-reconstruction pipeline against
//! the C++ baseline (`../MSBG/splattest.cpp`).
//!
//! Both sides run the REAL pipeline on `bun_zipper_res2.ply` (single instance,
//! 64-voxel bunny in a 256³ grid): load -> place -> active set -> splat ->
//! finalize -> 6 mean-curvature sweeps. The C++ binary prints two full level-0
//! u16 fields (after finalize, after smoothing); the golden samples below are
//! extracted from it, and the live check compares every voxel within a budget.
//!
//! The even iteration count (6) sidesteps the C++ `nMaxIter++` even-izing
//! quirk that would otherwise turn "5" into "6".

use msbg_rs::{
    blockpool::BlockPool,
    channel::Density,
    particles::{reconstruct_and_smooth, reconstruct_surface, DEFAULT_MSX, SurfaceConfig},
    solver::Fence,
    sparse_grid::SparseGrid,
};
use std::sync::Arc;

const BSX: usize = 16;
const N: usize = 4096;
const RES: usize = 256;

const CFG: SurfaceConfig = SurfaceConfig {
    sx: RES,
    sy: RES,
    sz: RES,
    n_instances: 1,
    instance_scale_factor: 0.25,
    r_particle: 2.0,
    nb_dist: 2.0,
    n_smooth_iters: 6,
    smooth_dt: 0.05,
};

/// Global voxel coordinates sampled for the golden check (surface + empty).
const COORDS: [(usize, usize, usize); 15] = [
    (130, 143, 127),
    (131, 143, 127),
    (132, 143, 127),
    (133, 143, 127),
    (134, 143, 127),
    (135, 143, 127),
    (140, 158, 123),
    (138, 159, 123),
    (139, 159, 123),
    (140, 159, 123),
    (141, 159, 123),
    (139, 155, 124),
    (10, 10, 10),
    (0, 0, 0),
    (255, 255, 255),
];

/// Field values at `COORDS`, produced by `../MSBG/build/splattest`.
const GOLDEN_A: [u16; 15] = [
    35775, 37972, 39126, 37459, 37071, 35845, 33105, 33327, 34642, 34692, 33286, 32815, 0, 0, 0,
];
const GOLDEN_B: [u16; 15] = [
    34658, 36685, 37514, 36981, 36022, 34815, 32891, 33417, 34277, 34338, 33486, 33641, 0, 0, 0,
];

const N_ACTIVE_GOLDEN: usize = 91;
const N_PARTICLES_GOLDEN: usize = 8171;

fn ply_path() -> String {
    std::env::var("MSBG_DATA_DIR")
        .unwrap_or_else(|_| "../MSBG/data".to_string())
        + "/bun_zipper_res2.ply"
}

fn load_ply() -> Vec<u8> {
    std::fs::read(ply_path()).unwrap()
}

fn voxel_at(g: &SparseGrid<Density, BSX, N>, x: usize, y: usize, z: usize) -> u16 {
    g.get_voxel(x, y, z).0
}

#[test]
fn splat_matches_cpp_golden_samples() {
    let ply = load_ply();

    // Field A: after splat + finalize.
    let cfg_a = SurfaceConfig { n_smooth_iters: 0, ..CFG };
    let (grid_a, active) = reconstruct_surface::<Density, BSX, N, DEFAULT_MSX>(
        &cfg_a,
        &ply,
        Arc::new(BlockPool::new(8, 64)),
    )
    .unwrap();
    assert_eq!(active.len(), N_ACTIVE_GOLDEN, "active block count");
    assert_eq!(grid_a.n_blocks, RES * RES * RES / 4096);
    for (k, &(x, y, z)) in COORDS.iter().enumerate() {
        assert!(voxel_at(&grid_a, x, y, z).abs_diff(GOLDEN_A[k]) <= 2, "field A at {x},{y},{z}");
    }

    // Field B: after splat + finalize + 6 mean-curvature sweeps.
    let (grid_b, active) = reconstruct_and_smooth::<Density, BSX, N, 18, DEFAULT_MSX>(
        &CFG,
        &ply,
        Arc::new(BlockPool::new(8, 64)),
        rayon::current_num_threads(),
        Fence::Sfence,
    )
    .unwrap();
    assert_eq!(active.len(), N_ACTIVE_GOLDEN);
    for (k, &(x, y, z)) in COORDS.iter().enumerate() {
        // The mean-curvature sweep propagates f32/FMA ulp differences (see
        // docs/refactor.md §8); allow the documented 2-ulp budget.
        assert!(voxel_at(&grid_b, x, y, z).abs_diff(GOLDEN_B[k]) <= 2, "field B at {x},{y},{z}");
    }
}

fn dump_full_field(g: &SparseGrid<Density, BSX, N>) -> Vec<u16> {
    let mut out = Vec::with_capacity(g.n_blocks * N);
    for bid in 0..g.n_blocks {
        match g.blockmap[bid] {
            Some(p) if p != g.empty_block && p != g.full_block => {
                let d = unsafe { (*p.as_ptr()).data };
                out.extend(d.iter().map(|x| x.0));
            }
            _ => out.extend(std::iter::repeat(0u16).take(N)),
        }
    }
    out
}

/// Budget: the field must match everywhere within 2 ulps, and the fraction of
/// voxels off by 1 ulp (the documented f32/FMA divergence) is bounded.
fn assert_within_budget(name: &str, rust: &[u16], cpp: &[u16]) {
    assert_eq!(rust.len(), cpp.len());
    let total = rust.len();
    let mut max_diff = 0u32;
    let mut n_off = 0usize;
    for (r, c) in rust.iter().zip(cpp) {
        let d = (*r as i32 - *c as i32).unsigned_abs();
        max_diff = max_diff.max(d);
        if d >= 1 {
            n_off += 1;
        }
    }
    let frac = n_off as f64 / total as f64;
    assert!(
        max_diff <= 2,
        "{name}: max voxel diff {max_diff} exceeds 2 ulps"
    );
    assert!(
        frac <= 0.001,
        "{name}: {n_off} voxels differ by >=1 ulp ({:.4}%) exceeds 0.1% budget",
        frac * 100.0
    );
}

#[test]
fn splat_against_cpp_binary_if_available() {
    let Ok(bin) = std::env::var("MSBG_CPP_SPLATTEST_BIN") else {
        eprintln!("skipping: set MSBG_CPP_SPLATTEST_BIN to the built C++ splattest");
        return;
    };
    let out = std::process::Command::new(&bin)
        .args([
            ply_path().as_str(),
            "256", "256", "256", "1", "0.25", "2.0", "2.0", "6", "0.05",
        ])
        .output()
        .expect("failed to run C++ splattest");
    assert!(
        out.status.success(),
        "C++ splattest failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let data = out.stdout;
    let nl = data.iter().position(|&b| b == b'\n').expect("no header line");
    let hdr: Vec<&str> = std::str::from_utf8(&data[..nl]).unwrap().split(' ').collect();
    let n_active_cpp: usize = hdr[0].parse().unwrap();
    let n_total_cpp: usize = hdr[1].parse().unwrap();
    let n_particles_cpp: usize = hdr[2].parse().unwrap();
    assert_eq!(n_particles_cpp, N_PARTICLES_GOLDEN);

    let rest = &data[nl + 1..];
    let half = rest.len() / 2;
    let cpp_a: Vec<u16> = rest[..half]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let cpp_b: Vec<u16> = rest[half..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let ply = load_ply();
    let cfg_a = SurfaceConfig { n_smooth_iters: 0, ..CFG };
    let (grid_a, active) = reconstruct_surface::<Density, BSX, N, DEFAULT_MSX>(
        &cfg_a,
        &ply,
        Arc::new(BlockPool::new(8, 64)),
    )
    .unwrap();
    assert_eq!(grid_a.n_blocks, n_total_cpp, "block count");
    assert_eq!(active.len(), n_active_cpp, "active block count");
    let rust_a = dump_full_field(&grid_a);
    assert_within_budget("field A (after finalize)", &rust_a, &cpp_a);

    let (grid_b, active_b) = reconstruct_and_smooth::<Density, BSX, N, 18, DEFAULT_MSX>(
        &CFG,
        &ply,
        Arc::new(BlockPool::new(8, 64)),
        rayon::current_num_threads(),
        Fence::Sfence,
    )
    .unwrap();
    assert_eq!(active_b.len(), n_active_cpp);
    let rust_b = dump_full_field(&grid_b);
    assert_within_budget("field B (after smoothing)", &rust_b, &cpp_b);
}

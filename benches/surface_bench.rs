//! Step-9/12 surface-reconstruction benchmark: the REAL `msbg_test_sparse`
//! pipeline phases (PLY parse, placement, sparse bucket, 8-color splat,
//! finalize, and end-to-end with 6 mean-curvature sweeps). Mirrors the C++
//! `benchmark.cpp` scenario H ("surface") — real library calls on both sides,
//! no mocks. Reports per-phase throughput + peak RSS (VmHWM).
//!
//! Sizes (per machine, `MSBG_BENCH_MACHINE=dell|macbook|aws`):
//! - small: `bun_zipper_res2.ply`, 256³, 1 instance — identical everywhere.
//! - dell big:    res2, 512³, 64 instances (~4.2M particles).
//! - macbook big: res2, 1024³, 8171 instances (~66.8M particles, testCase 1).
//! - aws big:     `bun_zipper.ply`, 4096³, 35947 instances (~1.29B particles).
//! - aws xbig:    `bun_zipper.ply`, 8192³, 35947 instances (~1.29B particles).
//! `MSBG_SURFACE_RES` / `MSBG_SURFACE_NINST` / `MSBG_SURFACE_PLY` override the
//! defaults (the 32,768³ paper run is `MSBG_SURFACE_RES=32768` on AWS).

use msbg_rs::{
    blockpool::BlockPool,
    channel::Density,
    particles::{
        finalize, reconstruct_and_smooth, sort, splat, DEFAULT_MSX, DomainBounds, GridDims,
        SurfaceConfig,
    },
    solver::Fence,
    sparse_grid::SparseGrid,
};
use std::{env, hint::black_box, sync::Arc, time::Instant};

const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;
const R_PARTICLE: f32 = 2.0;
const NB_DIST: f32 = 2.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Machine {
    Dell,
    Macbook,
    Aws,
}

fn machine() -> Machine {
    match env::var("MSBG_BENCH_MACHINE").as_deref() {
        Ok("dell") => Machine::Dell,
        Ok("macbook") => Machine::Macbook,
        Ok("aws") => Machine::Aws,
        Ok(other) => panic!("unknown MSBG_BENCH_MACHINE '{other}' (use dell|macbook|aws)"),
        Err(_) => {
            if cfg!(target_os = "macos") {
                Machine::Macbook
            } else {
                Machine::Dell
            }
        }
    }
}

fn scale() -> &'static str {
    static SCALE: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    SCALE.get_or_init(|| match env::var("MSBG_BENCH_SCALE").as_deref() {
        Ok("small") => "small",
        Ok("xbig") => "xbig",
        _ => "big",
    })
}

fn config() -> SurfaceConfig {
    let m = machine();
    let small = scale() == "small";
    let xbig = scale() == "xbig";
    let (res, n_inst, scale_factor) = match (m, small, xbig) {
        (_, true, _) => (256, 1, 0.25),
        (Machine::Dell, false, _) => (512, 64, 0.01),
        (Machine::Macbook, false, false) => (1024, 8171, 0.01),
        (Machine::Macbook, false, true) => (2048, 8171, 0.005),
        (Machine::Aws, false, false) => (4096, 35947, 0.005),
        (Machine::Aws, false, true) => (8192, 35947, 0.005),
    };
    let res = env::var("MSBG_SURFACE_RES")
        .map(|v| v.parse().unwrap())
        .unwrap_or(res);
    let n_inst = env::var("MSBG_SURFACE_NINST")
        .map(|v| v.parse().unwrap())
        .unwrap_or(n_inst);
    let scale_factor = env::var("MSBG_SURFACE_SCALE")
        .map(|v| v.parse().unwrap())
        .unwrap_or(scale_factor);
    SurfaceConfig {
        sx: res,
        sy: res,
        sz: res,
        n_instances: n_inst,
        instance_scale_factor: scale_factor,
        r_particle: R_PARTICLE,
        nb_dist: NB_DIST,
        n_smooth_iters: 6,
        smooth_dt: 0.05,
    }
}

fn ply_path() -> String {
    let name = env::var("MSBG_SURFACE_PLY").unwrap_or_else(|_| {
        if machine() == Machine::Aws && scale() != "small" {
            "bun_zipper.ply".to_string()
        } else {
            "bun_zipper_res2.ply".to_string()
        }
    });
    env::var("MSBG_DATA_DIR").unwrap_or_else(|_| "../MSBG/data".to_string()) + "/" + &name
}

fn thread_pool() -> &'static msbg_rs::thread_pool::Pool {
    static POOL: std::sync::OnceLock<msbg_rs::thread_pool::Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| msbg_rs::thread_pool::Pool::new(rayon::current_num_threads()))
}

fn iters() -> usize {
    if scale() == "small" {
        5
    } else {
        2
    }
}

/// Peak resident set size in MiB (Linux `/proc/self/status` VmHWM).
fn peak_rss_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Ok(kb) = rest.trim().trim_end_matches("kB").trim().parse::<f64>() {
                return kb / 1024.0;
            }
        }
    }
    0.0
}

fn report(name: &str, n_units: u64, dur: std::time::Duration) {
    let ms = dur.as_secs_f64() * 1000.0;
    let rate = n_units as f64 / dur.as_secs_f64();
    println!(
        "[Rust] {name}: {ms:.3} ms ({rate:.3} /s, rss={:.1} MiB)",
        peak_rss_mib()
    );
}

fn main() {
    println!("============================================================");
    println!(
        " msbg-rs Surface Reconstruction Benchmark (machine={:?}, scale={})",
        machine(),
        scale()
    );
    println!("============================================================\n");

    let cfg = config();
    let ply = std::fs::read(ply_path()).unwrap();
    thread_pool().install(|| {
        // ---- PLY parse ----------------------------------------------------
        let t = Instant::now();
        let loaded = msbg_rs::io::ply::load_vertices(&ply).unwrap();
        report(
            "surface_parse",
            loaded.positions.len() as u64,
            t.elapsed(),
        );
        let base = &loaded.positions;

        let dims = GridDims::new(cfg.sx, cfg.sy, cfg.sz, BSX);
        let span_max = (0..3)
            .map(|k| loaded.bbox_max[k] - loaded.bbox_min[k])
            .fold(0.0f32, f32::max);
        let domain = DomainBounds::new(&dims);

        // ---- Placement (positions + bids + sparse footprint union) ---------
        let placed = {
            let t = Instant::now();
            let placed = sort::place(base, &loaded.bbox_min, span_max, &dims, &domain, &cfg);
            report("surface_place", placed.positions.len() as u64, t.elapsed());
            println!(
                "    (placed particles {}, active blocks {})",
                placed.positions.len(),
                placed.active.len()
            );
            placed
        };

        // ---- Sparse bucket (block-major counting sort) ---------------------
        let (bucketed, active, n_placed) = {
            let t = Instant::now();
            let n_placed = placed.positions.len() as u64;
            let bucketed = sort::bucket_by_block(placed.positions, placed.bids);
            report("surface_bucket", n_placed, t.elapsed());
            (bucketed, placed.active, n_placed)
        };

        // ---- Grid allocation + fill (once) --------------------------------
        // Pool sized for the occupied block count (not a fixed small constant,
        // which would panic on the AWS paper-scale legs).
        let blocks_per_seg = 4096;
        let pool = Arc::new(BlockPool::new(
            active.len() / blocks_per_seg + 2,
            blocks_per_seg,
        ));
        let mut grid = SparseGrid::new(
            "surface".into(),
            cfg.sx,
            cfg.sy,
            cfg.sz,
            Density(0),
            Density(u16::MAX),
            pool,
        );
        grid.ensure_blocks_parallel(&active);
        grid.fill_blocks_parallel(&active, Density(u16::MAX));
        // ---- 8-color splat ------------------------------------------------
        {
            let it = iters();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..it {
                grid.fill_blocks_parallel(&active, Density(u16::MAX)); // reset to the pre-splat state
                let t = Instant::now();
                splat::splat::<Density, BSX, N, DEFAULT_MSX>(&grid, &bucketed, &cfg);
                total += t.elapsed();
            }
            report("surface_splat", n_placed, total / it as u32);
        }

        // ---- Finalize -----------------------------------------------------
        {
            let it = iters();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..it {
                let t = Instant::now();
                finalize::finalize::<Density, BSX, N>(&grid, &active, &cfg);
                total += t.elapsed();
            }
            let voxels = active.len() as u64 * N as u64;
            report("surface_finalize", voxels, total / it as u32);
        }

        // ---- End-to-end (parse+place+splat+finalize+6xMC) -----------------
        {
            let t = Instant::now();
            // Reuse the occupied-block count from the phase-by-phase pass above
            // (the e2e re-places to the same active set), sized like the grid.
            let blocks_per_seg = 4096;
            let (_g, _a) = reconstruct_and_smooth::<Density, BSX, N, HSX, DEFAULT_MSX>(
                &cfg,
                &ply,
                Arc::new(BlockPool::new(active.len() / blocks_per_seg + 2, blocks_per_seg)),
                rayon::current_num_threads(),
                Fence::Sfence,
            )
            .unwrap();
            black_box(&_g);
            report("surface_e2e", 1, t.elapsed());
        }

        println!("\nDone.");
    });
}

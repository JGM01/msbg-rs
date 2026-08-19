//! Step-9 surface-reconstruction benchmark: the REAL `msbg_test_sparse`
//! pipeline phases (PLY parse, placement+bucket, 8-color splat, finalize, and
//! end-to-end with 6 mean-curvature sweeps). Mirrors the C++ `benchmark.cpp`
//! scenario H ("surface") — real library calls on both sides, no mocks.
//!
//! Sizes:
//! - small: `bun_zipper_res2.ply`, 256³, 1 instance (8171 particles) — fast.
//! - big:   `bun_zipper_res2.ply`, 512³, 512 instances (~4.2M particles).
//!
//! The full paper-scale bunny-of-bunnies (1024³, 1.29G particles) needs
//! >20 GB RAM and is deferred (see docs/roadmap.md).

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

fn scale() -> &'static str {
    static SCALE: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    SCALE.get_or_init(|| match env::var("MSBG_BENCH_SCALE").as_deref() {
        Ok("small") => "small",
        _ => "big",
    })
}

fn config() -> SurfaceConfig {
    let small = scale() == "small";
    let res = env::var("MSBG_SURFACE_RES")
        .map(|v| v.parse().unwrap())
        .unwrap_or(if small { 256 } else { 512 });
    let n_inst = env::var("MSBG_SURFACE_NINST")
        .map(|v| v.parse().unwrap())
        .unwrap_or(if small { 1 } else { 64 });
    SurfaceConfig {
        sx: res,
        sy: res,
        sz: res,
        n_instances: n_inst,
        instance_scale_factor: if small { 0.25 } else { 0.01 },
        r_particle: R_PARTICLE,
        nb_dist: NB_DIST,
        n_smooth_iters: 6,
        smooth_dt: 0.05,
    }
}

fn ply_path() -> String {
    env::var("MSBG_DATA_DIR").unwrap_or_else(|_| "../MSBG/data".to_string()) + "/bun_zipper_res2.ply"
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

fn fill_active_full(grid: &mut SparseGrid<Density, BSX, N>, active: &[usize]) {
    for &bid in active {
        if let Some(data) = grid.get_block_data_mut(bid) {
            data.fill(Density(u16::MAX));
        }
    }
}

fn report(name: &str, n_units: u64, dur: std::time::Duration) {
    let ms = dur.as_secs_f64() * 1000.0;
    let rate = n_units as f64 / dur.as_secs_f64();
    println!("[Rust] {name}: {ms:.3} ms ({rate:.3} /s)");
}

fn main() {
    println!("============================================================");
    println!(" msbg-rs Step-9 Surface Reconstruction Benchmark (scale={})", scale());
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

        // ---- Placement + active set + bucket ------------------------------
        let (bucketed, active, n_placed) = {
            let t = Instant::now();
            let placed = sort::place(base, &loaded.bbox_min, span_max, &dims, &domain, &cfg);
            let active = placed.active;
            let n_placed = placed.positions.len() as u64;
            let bucketed =
                sort::bucket_by_block(placed.positions, placed.bids, dims.n_blocks());
            report("surface_place", n_placed, t.elapsed());
            println!("    (placed particles {}, active blocks {})", n_placed, active.len());
            (bucketed, active, n_placed)
        };

        // ---- Grid allocation + fill (once) --------------------------------
        // Pool capacity must cover the active set (n_blocks at most).
        let pool = Arc::new(BlockPool::new(16, 1024));
        let mut grid = SparseGrid::new(
            "surface".into(),
            cfg.sx,
            cfg.sy,
            cfg.sz,
            Density(0),
            Density(u16::MAX),
            pool,
        );
        for &bid in &active {
            grid.ensure_block(bid);
        }
        fill_active_full(&mut grid, &active);

        // ---- 8-color splat ------------------------------------------------
        {
            let it = iters();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..it {
                fill_active_full(&mut grid, &active); // reset to the pre-splat state
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
            let (_g, _a) = reconstruct_and_smooth::<Density, BSX, N, HSX, DEFAULT_MSX>(
                &cfg,
                &ply,
                Arc::new(BlockPool::new(16, 1024)),
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

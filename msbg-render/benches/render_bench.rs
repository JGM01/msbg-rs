//! Step-10 rendering benchmark: REAL `msbg-rs` reconstruction → REAL renderer
//! (`msbg-render`) on both the Rust side and the C++ `rendertest.cpp` harness.
//! No mocks — the density field is produced by the step-9 pipeline, and the
//! render consumes only the public `Sampler`/`SparseGrid` API.
//!
//! Sizes:
//! - small: `bun_zipper_res2.ply`, 256³, 1 instance, 640×480 frame.
//! - big:   `bun_zipper_res2.ply`, 512³, 64 instances, 1280×720 frame.
//!
//! Output lines are `[Rust] render_* ...` for side-by-side comparison with the
//! `[C++]` lines from `../MSBG/build/rendertest`.

use std::sync::Arc;
use std::time::Instant;
use std::{env, hint::black_box};

use msbg_render::{greyscale, raymarch, render_slice, Camera, RenderOptions, SliceAxis, turbo};
use msbg_rs::blockpool::BlockPool;
use msbg_rs::channel::{Density, Vec3};
use msbg_rs::math::Interpolation;
use msbg_rs::particles::{reconstruct_and_smooth, DEFAULT_MSX, SurfaceConfig};
use msbg_rs::solver::Fence;

const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;

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
        r_particle: 2.0,
        nb_dist: 2.0,
        n_smooth_iters: 6,
        smooth_dt: 0.05,
    }
}

fn ply_path() -> String {
    env::var("MSBG_DATA_DIR").unwrap_or_else(|_| "../../MSBG/data".to_string())
        + "/bun_zipper_res2.ply"
}

fn thread_pool() -> &'static msbg_rs::thread_pool::Pool {
    static POOL: std::sync::OnceLock<msbg_rs::thread_pool::Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| msbg_rs::thread_pool::Pool::new(rayon::current_num_threads()))
}

fn raymarch_res() -> (usize, usize) {
    let w = env::var("MSBG_RENDER_W").map(|v| v.parse().unwrap());
    let h = env::var("MSBG_RENDER_H").map(|v| v.parse().unwrap());
    match (w, h) {
        (Ok(w), Ok(h)) => (w, h),
        _ if scale() == "small" => (640, 480),
        _ => (1280, 720),
    }
}

fn report(name: &str, n_units: u64, dur: std::time::Duration) {
    let ms = dur.as_secs_f64() * 1000.0;
    let rate = n_units as f64 / dur.as_secs_f64();
    println!("[Rust] {name}: {ms:.3} ms ({rate:.3} /s)");
}

fn main() {
    println!("============================================================");
    println!(" msbg-render Step-10 Rendering Benchmark (scale={})", scale());
    println!("============================================================\n");

    let cfg = config();
    let ply = std::fs::read(ply_path()).unwrap();

    thread_pool().install(|| {
        let t0 = Instant::now();
        let (grid, _active) = reconstruct_and_smooth::<Density, BSX, N, HSX, DEFAULT_MSX>(
            &cfg,
            &ply,
            Arc::new(BlockPool::new(16, 1024)),
            rayon::current_num_threads(),
            Fence::Sfence,
        )
        .unwrap();
        println!("    (reconstruction {:.3} ms)\n", t0.elapsed().as_secs_f64() * 1000.0);

        // ---- O(N²) slices (3 planes, like the C++ getSlices2D) ------------
        {
            let sx = grid.sx as f32;
            let sy = grid.sy as f32;
            let sz = grid.sz as f32;
            let t = Instant::now();
            let a = render_slice::<Density, BSX, N, { Interpolation::Linear }>(
                &grid, SliceAxis::X, sx * 0.5, grid.sz, grid.sy,
                msbg_rs::math::BoundaryCondition::Clamp, greyscale,
            );
            let b = render_slice::<Density, BSX, N, { Interpolation::Linear }>(
                &grid, SliceAxis::Y, sy * 0.5, grid.sx, grid.sz,
                msbg_rs::math::BoundaryCondition::Clamp, turbo,
            );
            let c = render_slice::<Density, BSX, N, { Interpolation::Linear }>(
                &grid, SliceAxis::Z, sz * 0.5, grid.sx, grid.sy,
                msbg_rs::math::BoundaryCondition::Clamp, greyscale,
            );
            black_box((&a, &b, &c));
            let px = (grid.sz * grid.sy + grid.sx * grid.sz + grid.sx * grid.sy) as u64;
            report("render_slice", px, t.elapsed());
        }

        // ---- ESS-DDA isosurface raymarch -----------------------------------
        {
            let (w, h) = raymarch_res();
            let sxyz = grid.sx.max(grid.sy).max(grid.sz) as f32;
            let sx = grid.sx as f32;
            let sy = grid.sy as f32;
            let sz = grid.sz as f32;
            let cam = Camera::look_at(
                Vec3::new(0.5 * sx, 0.8 * sy, 0.7 * sz),
                Vec3::new(0.5 * sx, 0.5 * sy, 0.5 * sz),
                1.0,
            );
            let opts = RenderOptions {
                iso_level: 0.5,
                step_size: 1.0,
                max_dist: 2.0 * sxyz,
                background: image::Rgba([0, 0, 0, 255]),
                surface_color: [0.8, 0.6, 0.4],
                sun: Vec3::new(0.5 * sx, 5.0 * sy, 5.0 * sz),
                specular_strength: 0.3,
                shininess: 32.0,
                disable_ess: false,
            };

            let t = Instant::now();
            let img = raymarch::<Density, BSX, N>(&grid, &cam, w, h, &opts);
            black_box(&img);
            report("render_raymarch", (w * h) as u64, t.elapsed());

            let opts_off = RenderOptions { disable_ess: true, ..opts };
            let t = Instant::now();
            let img_off = raymarch::<Density, BSX, N>(&grid, &cam, w, h, &opts_off);
            black_box(&img_off);
            report("render_raymarch_ess_off", (w * h) as u64, t.elapsed());
        }

        println!("\nDone.");
    });
}

//! Paper-scale "bunny-of-bunnies" showcase: reconstruct the testCase-1/2
//! density field through the `msbg_rs` public API (with per-phase throughput +
//! peak-RSS reporting), then render a perspective raymarch + downsampled
//! orthogonal slices to PNG — all in one process, so the image is written
//! "during" the run rather than a second reconstruction.
//!
//! Usage:
//!   bunny_of_bunnies <ply> <sx> <sy> <sz> <nInst(0=auto)> <scale_factor> \
//!                    <out_dir> [render_w=1920] [render_h=1080] [bsx=16|32]
//!
//! AWS paper run (testCase 2, ~1.29B particles, ~100B active voxels at 32,768³):
//!   cargo run -p msbg-render --release --example bunny_of_bunnies -- \
//!       ../MSBG/data/bun_zipper.ply 32768 32768 32768 35947 0.005 \
//!       out_msbg_aws 3840 2160 16

use std::sync::Arc;
use std::time::Instant;

use image::Rgba;
use msbg_render::{raymarch, turbo, Camera, RenderOptions};
use msbg_rs::blockpool::BlockPool;
use msbg_rs::channel::{Density, Vec3};
use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sampler};
use msbg_rs::particles::{finalize, sort, splat, DomainBounds, GridDims, SurfaceConfig};
use msbg_rs::solver::{Fence, PdeParams, Stencil, Sweeper};
use msbg_rs::sparse_grid::SparseGrid;
use rayon::prelude::*;

fn arg<T: std::str::FromStr>(i: usize, default: Option<T>) -> T {
    match std::env::args().nth(i) {
        Some(s) => s.parse().ok().expect("bad arg"),
        None => default.expect("missing arg"),
    }
}

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
    println!("[Rust] {name}: {ms:.3} ms ({rate:.3} /s, rss={:.1} MiB)", peak_rss_mib());
}

/// Whole-domain downsampled Z-slice: sample `out_w × out_h` pixels across the
/// full field (not just the first `out_w` voxels, which is what the `render_slice`
/// helper does — it maps output pixels 1:1 to voxels).
fn slice_z_downsampled<const BSX: usize, const N: usize>(
    grid: &SparseGrid<Density, BSX, N>,
    out_w: usize,
    out_h: usize,
) -> image::RgbaImage {
    let sampler = Sampler::new(grid, GridAlignment::Corner, BoundaryCondition::Clamp);
    let mut buf = vec![0u8; out_w * out_h * 4];
    let sx = grid.sx as f32;
    let sy = grid.sy as f32;
    let sz = grid.sz as f32;
    buf.par_chunks_exact_mut(out_w * 4)
        .enumerate()
        .for_each(|(v, row)| {
            let y = (v as f32 + 0.5) * sy / out_h as f32;
            for (u, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let x = (u as f32 + 0.5) * sx / out_w as f32;
                let d = sampler.sample::<{ Interpolation::Linear }>(Vec3::new(x, y, 0.5 * sz));
                *px = turbo(d).0;
            }
        });
    image::RgbaImage::from_raw(out_w as u32, out_h as u32, buf).expect("slice size")
}

fn main() {
    let ply_path = arg::<String>(1, None);
    let sx = arg::<usize>(2, None);
    let sy = arg::<usize>(3, None);
    let sz = arg::<usize>(4, None);
    let n_inst = arg::<usize>(5, None);
    let scale_factor = arg::<f32>(6, None);
    let out_dir = arg::<String>(7, None);
    let render_w = arg::<usize>(8, Some(1920));
    let render_h = arg::<usize>(9, Some(1080));
    let bsx = arg::<usize>(10, Some(16));

    match bsx {
        16 => run::<16, 4096, 18, 24>(ply_path, sx, sy, sz, n_inst, scale_factor, out_dir, render_w, render_h),
        32 => run::<32, 32768, 34, 40>(ply_path, sx, sy, sz, n_inst, scale_factor, out_dir, render_w, render_h),
        other => panic!("unsupported bsx {other} (16 or 32)"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run<const BSX: usize, const N: usize, const HSX: usize, const MSX: usize>(
    ply_path: String,
    sx: usize,
    sy: usize,
    sz: usize,
    n_inst: usize,
    scale_factor: f32,
    out_dir: String,
    render_w: usize,
    render_h: usize,
) {
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let cfg = SurfaceConfig {
        sx,
        sy,
        sz,
        n_instances: n_inst,
        instance_scale_factor: scale_factor,
        r_particle: 2.0,
        nb_dist: 2.0,
        n_smooth_iters: 6,
        smooth_dt: 0.05,
    };
    let ply_bytes = std::fs::read(&ply_path).expect("read ply");

    let pool = msbg_rs::thread_pool::Pool::new(rayon::current_num_threads());
    pool.install(|| {
        // ---- Parse ---------------------------------------------------------
        let t = Instant::now();
        let loaded = msbg_rs::io::ply::load_vertices(&ply_bytes).expect("parse");
        report("parse", loaded.positions.len() as u64, t.elapsed());
        let base = &loaded.positions;

        let dims = GridDims::new(sx, sy, sz, BSX);
        let span_max = (0..3)
            .map(|k| loaded.bbox_max[k] - loaded.bbox_min[k])
            .fold(0.0f32, f32::max);
        let domain = DomainBounds::new(&dims);

        // ---- Placement -----------------------------------------------------
        let placed = {
            let t = Instant::now();
            let placed = sort::place(base, &loaded.bbox_min, span_max, &dims, &domain, &cfg);
            report("place", placed.positions.len() as u64, t.elapsed());
            placed
        };
        let n_placed = placed.positions.len() as u64;

        // ---- Bucket --------------------------------------------------------
        let bucketed = {
            let t = Instant::now();
            let bucketed = sort::bucket_by_block(placed.positions, placed.bids);
            report("bucket", n_placed, t.elapsed());
            bucketed
        };
        let active = placed.active;
        println!("    placed={} active_blocks={}", n_placed, active.len());

        // ---- Allocate + fill -------------------------------------------------
        let grid = {
            let t = Instant::now();
            // Pool sized for the *occupied* block count, not a fixed small-grid
            // constant — a tiny pool would panic on allocation.
            let blocks_per_seg = 4096; // 64 MB segments (C++ standard)
            let max_segments = active.len() / blocks_per_seg + 2;
            let grid_pool = Arc::new(BlockPool::<Density, BSX, N>::new(max_segments, blocks_per_seg));
            let mut grid = SparseGrid::new(
                "bunny_of_bunnies".into(),
                sx,
                sy,
                sz,
                Density(0),
                Density(u16::MAX),
                grid_pool,
            );
            grid.ensure_blocks_parallel(&active);
            grid.fill_blocks_parallel(&active, Density(u16::MAX));
            let voxels = active.len() as u64 * N as u64;
            report("alloc_fill", voxels, t.elapsed());
            grid
        };

        // ---- Splat ---------------------------------------------------------
        {
            let t = Instant::now();
            splat::splat::<Density, BSX, N, MSX>(&grid, &bucketed, &cfg);
            report("splat", n_placed, t.elapsed());
        }

        // ---- Finalize ------------------------------------------------------
        {
            let t = Instant::now();
            finalize::finalize::<Density, BSX, N>(&grid, &active, &cfg);
            let voxels = active.len() as u64 * N as u64;
            report("finalize", voxels, t.elapsed());
        }

        // ---- Mean-curvature smoothing --------------------------------------
        {
            let t = Instant::now();
            let sweeper = Sweeper::<Density, BSX, N, HSX>::new(&grid, rayon::current_num_threads(), Fence::Sfence);
            sweeper.sweep(
                &active,
                Stencil::MeanCurvature,
                &PdeParams { dt: 0.05, iterations: 6, do_constr_zero_one: true },
            );
            // unknowns = active_voxels × iterations, the paper's headline unit.
            let unknowns = active.len() as u64 * N as u64 * 6;
            report("mean_curvature", unknowns, t.elapsed());
        }

        // ---- Render --------------------------------------------------------
        {
            let t = Instant::now();
            let sxyz = (sx.max(sy).max(sz)) as f32;

            // Big-bunny world AABB: the instance origins trace the bunny shape
            // (`origin = 0.2*sxyz + 0.6*sxyz*base_scale*(bi - bbox_min)`).
            let base_scale = 1.0 / span_max;
            let wmap = |f: f32, k: usize| 0.2 * sxyz + 0.6 * sxyz * base_scale * (f - loaded.bbox_min[k]);
            let center = Vec3::new(
                wmap((loaded.bbox_min[0] + loaded.bbox_max[0]) * 0.5, 0),
                wmap((loaded.bbox_min[1] + loaded.bbox_max[1]) * 0.5, 1),
                wmap((loaded.bbox_min[2] + loaded.bbox_max[2]) * 0.5, 2),
            );
            let half = Vec3::new(
                0.3 * sxyz * base_scale * (loaded.bbox_max[0] - loaded.bbox_min[0]),
                0.3 * sxyz * base_scale * (loaded.bbox_max[1] - loaded.bbox_min[1]),
                0.3 * sxyz * base_scale * (loaded.bbox_max[2] - loaded.bbox_min[2]),
            );
            // Largest half-extent drives the framing (bunny is longest along x).
            let r = half.x().max(half.y()).max(half.z());

            // bun_zipper nose faces +x, up is +y; light it from front-top-right.
            let opts = RenderOptions {
                iso_level: 0.5,
                step_size: 1.0,
                max_dist: 3.0 * sxyz,
                background: Rgba([0, 0, 0, 255]),
                surface_color: [0.8, 0.6, 0.4],
                sun: Vec3::new(0.8, 0.6, 0.5) * (3.0 * sxyz),
                specular_strength: 0.3,
                shininess: 32.0,
                disable_ess: false,
            };

            // Shared 3/4 view direction: mostly side (+z shows the nose-to-tail
            // length), some front (+x), slightly above.
            let dir = Vec3::new(0.5, 0.25, 0.85);
            let dir = dir * (1.0 / dir.len());

            // Far shot: the whole bunny.
            let cam_far = Camera::look_at(center + dir * (r * 2.2), center, 1.4);
            raymarch::<Density, BSX, N>(&grid, &cam_far, render_w, render_h, &opts)
                .save(format!("{out_dir}/bunny_of_bunnies.png"))
                .expect("save far");

            // Close shot: individual bunnies on the near surface.
            let cam_close = Camera::look_at(center + dir * (r * 1.05), center, 2.6);
            raymarch::<Density, BSX, N>(&grid, &cam_close, render_w, render_h, &opts)
                .save(format!("{out_dir}/bunny_of_bunnies_close.png"))
                .expect("save close");

            let slice_w = 1024usize;
            let slice_h = 1024usize;
            slice_z_downsampled::<BSX, N>(&grid, slice_w, slice_h)
                .save(format!("{out_dir}/slice_z.png"))
                .expect("save slice");
            report("render", (2 * render_w * render_h + 2 * slice_w * slice_h) as u64, t.elapsed());
        }

        println!("done: images in {out_dir}");
    });
}

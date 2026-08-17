//! End-to-end step-10 demo: reconstruct the bunny density field through the
//! `msbg_rs` public API, then render orthogonal slices + a perspective
//! raymarch, writing PNGs.
//!
//! Usage: render_bunny <ply> <sx> <sy> <sz> <n_inst(0=auto)> <scale_factor> <out_dir>

use std::sync::Arc;

use image::Rgba;
use msbg_render::{greyscale, raymarch, render_slice, Camera, RenderOptions, SliceAxis, turbo};
use msbg_rs::blockpool::BlockPool;
use msbg_rs::channel::{Density, Vec3};
use msbg_rs::math::{BoundaryCondition, Interpolation};
use msbg_rs::particles::{reconstruct_and_smooth, DEFAULT_MSX, SurfaceConfig};
use msbg_rs::solver::Fence;

const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;

fn arg<T: std::str::FromStr>(i: usize) -> T {
    std::env::args().nth(i).expect("missing arg").parse().ok().expect("bad arg")
}

fn main() {
    let ply = arg::<String>(1);
    let sx = arg::<usize>(2);
    let sy = arg::<usize>(3);
    let sz = arg::<usize>(4);
    let n_inst = arg::<usize>(5);
    let scale_factor = arg::<f32>(6);
    let out_dir = arg::<String>(7);
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
    let ply_bytes = std::fs::read(&ply).expect("read ply");

    let pool = msbg_rs::thread_pool::Pool::new(rayon::current_num_threads());
    pool.install(|| {
        let (grid, _active) = reconstruct_and_smooth::<Density, BSX, N, HSX, DEFAULT_MSX>(
            &cfg,
            &ply_bytes,
            Arc::new(BlockPool::new(16, 1024)),
            rayon::current_num_threads(),
            Fence::Sfence,
        )
        .expect("reconstruct");

        // Orthogonal slices (one per axis), false-color + greyscale.
        let sx_f = grid.sx as f32;
        let sy_f = grid.sy as f32;
        let sz_f = grid.sz as f32;
        for (axis, w, h, off) in [
            (SliceAxis::X, grid.sz, grid.sy, sx_f * 0.5),
            (SliceAxis::Y, grid.sx, grid.sz, sy_f * 0.5),
            (SliceAxis::Z, grid.sx, grid.sy, sz_f * 0.5),
        ] {
            let img = render_slice::<Density, BSX, N, { Interpolation::Linear }>(
                &grid,
                axis,
                off,
                w,
                h,
                BoundaryCondition::Clamp,
                turbo,
            );
            img.save(format!("{out_dir}/slice_{axis:?}.png")).expect("save slice");
        }

        // Perspective raymarch (turntable-ish diagonal view).
        let sxyz = grid.sx.max(grid.sy).max(grid.sz) as f32;
        let cam = Camera::look_at(
            Vec3::new(0.5 * sx_f, 0.8 * sy_f, 0.7 * sz_f),
            Vec3::new(0.5 * sx_f, 0.5 * sy_f, 0.5 * sz_f),
            1.0,
        );
        let opts = RenderOptions {
            iso_level: 0.5,
            step_size: 1.0,
            max_dist: 2.0 * sxyz,
            background: Rgba([0, 0, 0, 255]),
            surface_color: [0.8, 0.6, 0.4],
            sun: Vec3::new(0.5 * sx_f, 5.0 * sy_f, 5.0 * sz_f),
            specular_strength: 0.3,
            shininess: 32.0,
            disable_ess: false,
        };
        let frame = raymarch::<Density, BSX, N>(&grid, &cam, 960, 540, &opts);
        frame.save(format!("{out_dir}/raymarch.png")).expect("save raymarch");

        // A single greyscale Z slice for a quick visual check.
        let grey = render_slice::<Density, BSX, N, { Interpolation::Linear }>(
            &grid,
            SliceAxis::Z,
            sz_f * 0.5,
            grid.sx,
            grid.sy,
            BoundaryCondition::Clamp,
            greyscale,
        );
        grey.save(format!("{out_dir}/slice_grey.png")).expect("save grey");
    });

    println!("done: images in {out_dir}");
}

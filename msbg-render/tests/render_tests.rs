//! Boundary & awkward-case test matrix for `msbg-render` (step 10).
//!
//! Only two happy-path tests; the rest exercise boundaries, degenerate grids,
//! and DDA edge cases. Synthetic analytic fields keep the suite fast and
//! deterministic (the real bunny is exercised by the bench + example).

use std::sync::Arc;

use image::Rgba;
use msbg_render::{greyscale, raymarch, render_slice, Camera, RenderOptions, SliceAxis};
use msbg_rs::blockpool::BlockPool;
use msbg_rs::channel::{Density8, Vec3};
use msbg_rs::math::{BoundaryCondition, Interpolation};
use msbg_rs::sparse_grid::SparseGrid;

const BSX: usize = 16;
const N: usize = 4096;
const IP_LINEAR: Interpolation = Interpolation::Linear;

fn grid<D: Copy + Default + Send + Sync>(
    sx: usize,
    sy: usize,
    sz: usize,
) -> SparseGrid<D, BSX, N> {
    let pool = Arc::new(BlockPool::new(8, 64));
    SparseGrid::new("t".into(), sx, sy, sz, D::default(), D::default(), pool)
}

fn fill_sphere<D: Copy + Default + Send + Sync>(
    g: &mut SparseGrid<D, BSX, N>,
    c: (f32, f32, f32),
    r: f32,
    inside: D,
) {
    for z in 0..g.sz {
        for y in 0..g.sy {
            for x in 0..g.sx {
                let dx = x as f32 - c.0;
                let dy = y as f32 - c.1;
                let dz = z as f32 - c.2;
                if (dx * dx + dy * dy + dz * dz).sqrt() <= r {
                    g.set_voxel(x, y, z, inside);
                }
            }
        }
    }
}

fn default_opts(g: &SparseGrid<f32, BSX, N>) -> RenderOptions {
    let sxyz = g.sx.max(g.sy).max(g.sz) as f32;
    RenderOptions {
        iso_level: 0.5,
        step_size: 1.0,
        max_dist: 2.0 * sxyz,
        background: Rgba([0, 0, 0, 255]),
        surface_color: [1.0, 1.0, 1.0],
        sun: Vec3::new(0.0, 0.0, 1.0),
        specular_strength: 0.0,
        shininess: 32.0,
        disable_ess: false,
    }
}

fn is_background(px: &Rgba<u8>) -> bool {
    px[0] == 0 && px[1] == 0 && px[2] == 0
}

// ---- happy paths ------------------------------------------------------------

#[test]
fn slice_01_sphere_center() {
    let mut g = grid::<f32>(48, 48, 48);
    fill_sphere(&mut g, (24.0, 24.0, 24.0), 12.0, 1.0f32);
    let img = render_slice::<f32, BSX, N, IP_LINEAR>(
        &g,
        SliceAxis::Z,
        24.0,
        48,
        48,
        BoundaryCondition::Clamp,
        greyscale,
    );
    assert_eq!((img.width(), img.height()), (48, 48));
    assert!(img.get_pixel(24, 24)[0] > 200, "center should be dense");
    assert_eq!(img.get_pixel(2, 2)[0], 0, "corner should be empty");
}

#[test]
fn raymarch_02_sphere_perspective() {
    let mut g = grid::<f32>(48, 48, 48);
    fill_sphere(&mut g, (24.0, 24.0, 24.0), 10.0, 1.0f32);
    let cam = Camera::look_at(Vec3::new(24.0, 24.0, 0.0), Vec3::new(24.0, 24.0, 24.0), 1.0);
    let img = raymarch::<f32, BSX, N>(&g, &cam, 64, 64, &default_opts(&g));
    assert_eq!((img.width(), img.height()), (64, 64));
    assert!(!is_background(img.get_pixel(32, 32)), "center ray should hit");
    let hits = img.pixels().filter(|p| !is_background(p)).count();
    assert!(hits > 0);
}

// ---- boundary / awkward -----------------------------------------------------

#[test]
fn slice_03_out_of_bounds_dirichlet_vs_clamp() {
    let mut g = grid::<f32>(16, 16, 16);
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                g.set_voxel(x, y, z, 1.0f32);
            }
        }
    }
    let dir = render_slice::<f32, BSX, N, IP_LINEAR>(
        &g,
        SliceAxis::Z,
        -100.0,
        16,
        16,
        BoundaryCondition::Dirichlet,
        greyscale,
    );
    let clamp = render_slice::<f32, BSX, N, IP_LINEAR>(
        &g,
        SliceAxis::Z,
        -100.0,
        16,
        16,
        BoundaryCondition::Clamp,
        greyscale,
    );
    // Dirichlet reads the empty sentinel (0) out-of-domain; Clamp replicates the
    // nearest edge slice (1.0).
    assert_eq!(dir.get_pixel(4, 4)[0], 0);
    assert_eq!(clamp.get_pixel(4, 4)[0], 255);

    // Far side, same contract.
    let dir_hi = render_slice::<f32, BSX, N, IP_LINEAR>(
        &g, SliceAxis::Z, 5000.0, 16, 16, BoundaryCondition::Dirichlet, greyscale,
    );
    assert_eq!(dir_hi.get_pixel(4, 4)[0], 0);
}

#[test]
fn raymarch_04_grazing_coplanar_ray() {
    let mut g = grid::<f32>(32, 32, 32);
    fill_sphere(&mut g, (16.0, 16.0, 16.0), 8.0, 1.0f32);
    // A ray exactly along a block boundary (y = 16.0, z = 16.0) with a
    // zero-component direction: `forward = (1, 0, 0)` makes the center ray
    // d = (1, 0, 0), exercising the `dir == 0` DDA axes.
    let cam = Camera {
        position: Vec3::new(2.0, 16.0, 16.0),
        forward: Vec3::new(1.0, 0.0, 0.0),
        right: Vec3::new(0.0, 0.0, 1.0),
        up: Vec3::new(0.0, 1.0, 0.0),
        focal_len: 1.0,
    };
    let img = raymarch::<f32, BSX, N>(&g, &cam, 64, 64, &default_opts(&g));
    // The center ray traverses the sphere along its equator -> a hit.
    assert!(!is_background(img.get_pixel(32, 32)), "coplanar ray must not NaN-skip");
}

#[test]
fn raymarch_05_camera_inside_density() {
    let mut g = grid::<f32>(32, 32, 32);
    fill_sphere(&mut g, (16.0, 16.0, 16.0), 12.0, 1.0f32);
    // Camera at the dense center: no surface is crossed looking outward, so the
    // near plane is not spuriously treated as a hit.
    let cam = Camera::look_at(Vec3::new(16.0, 16.0, 16.0), Vec3::new(24.0, 16.0, 16.0), 1.0);
    let img = raymarch::<f32, BSX, N>(&g, &cam, 64, 64, &default_opts(&g));
    assert!(is_background(img.get_pixel(32, 32)));
}

#[test]
fn raymarch_06_ess_equivalence() {
    // Hollow shell: empty interior forces a block-skip across the middle.
    let mut g = grid::<f32>(48, 48, 48);
    fill_sphere(&mut g, (24.0, 24.0, 24.0), 20.0, 1.0f32);
    for z in 0..48 {
        for y in 0..48 {
            for x in 0..48 {
                let dx = x as f32 - 24.0;
                let dy = y as f32 - 24.0;
                let dz = z as f32 - 24.0;
                if (dx * dx + dy * dy + dz * dz).sqrt() <= 12.0 {
                    g.set_voxel(x, y, z, 0.0f32);
                }
            }
        }
    }
    let cam = Camera::look_at(Vec3::new(24.0, 24.0, 0.0), Vec3::new(24.0, 24.0, 24.0), 1.0);
    let on = raymarch::<f32, BSX, N>(&g, &cam, 96, 96, &default_opts(&g));
    let off = raymarch::<f32, BSX, N>(
        &g,
        &cam,
        96,
        96,
        &RenderOptions { disable_ess: true, ..default_opts(&g) },
    );

    // Hit mask must match exactly; colors may shift by a few LSBs from the
    // sub-voxel difference in the interpolated surface position.
    let mut max_diff = 0u8;
    for (a, b) in on.pixels().zip(off.pixels()) {
        assert_eq!(is_background(a), is_background(b), "ESS changed the hit mask");
        for k in 0..3 {
            max_diff = max_diff.max(a[k].abs_diff(b[k]));
        }
    }
    assert!(max_diff <= 16, "ESS color divergence {max_diff} exceeds budget");
}

#[test]
fn raymarch_07_density8_sqrt_dequant() {
    // Physical density 0.5 stored sqrt-compressed: Density8(round(sqrt(0.5)*255)).
    let v = (0.5f32).sqrt() * 255.0;
    let mut g = grid::<Density8>(16, 16, 16);
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                g.set_voxel(x, y, z, Density8(v.round() as u8));
            }
        }
    }
    let img = render_slice::<Density8, BSX, N, IP_LINEAR>(
        &g, SliceAxis::Z, 8.0, 16, 16, BoundaryCondition::Clamp, greyscale,
    );
    let px = img.get_pixel(8, 8)[0];
    assert!((126..=130).contains(&px), "sqrt-space sample not squared: {px}");
}

#[test]
fn render_08_asymmetric_ribbon() {
    // 512 x 16 x 16 ribbon: 32 blocks along x, 1 block in y/z.
    let mut g = grid::<f32>(512, 16, 16);
    fill_sphere(&mut g, (256.0, 8.0, 8.0), 6.0, 1.0f32);
    let img = render_slice::<f32, BSX, N, IP_LINEAR>(
        &g, SliceAxis::Z, 8.0, 512, 16, BoundaryCondition::Clamp, greyscale,
    );
    assert_eq!((img.width(), img.height()), (512, 16));
    assert!(img.get_pixel(256, 8)[0] > 200);

    let cam = Camera::look_at(Vec3::new(256.0, 8.0, 0.0), Vec3::new(256.0, 8.0, 8.0), 1.0);
    let frame = raymarch::<f32, BSX, N>(&g, &cam, 64, 48, &default_opts(&g));
    assert_eq!((frame.width(), frame.height()), (64, 48));
}

#[test]
fn raymarch_09_ray_misses_grid() {
    let mut g = grid::<f32>(32, 32, 32);
    fill_sphere(&mut g, (16.0, 16.0, 16.0), 8.0, 1.0f32);
    // Camera looking away from the grid: no ray intersects the AABB.
    let cam = Camera::look_at(Vec3::new(16.0, 16.0, -20.0), Vec3::new(16.0, 16.0, -40.0), 1.0);
    let img = raymarch::<f32, BSX, N>(&g, &cam, 64, 64, &default_opts(&g));
    assert!(img.pixels().all(is_background));
}

#[test]
fn raymarch_11_origin_on_block_boundary() {
    // Camera x = 16 sits exactly on the block boundary between block 0 and 1,
    // and the starting block (z in [0,16)) is empty while the sphere lives in
    // block (1,1,1). A ray with a negative-x direction starts with t_max[x] == 0
    // (the boundary is "now"); the empty-block skip must advance into the
    // neighbor rather than killing the ray — otherwise every negative-x ray dies
    // and half the image is black.
    let mut g = grid::<f32>(32, 32, 32);
    fill_sphere(&mut g, (16.0, 16.0, 24.0), 8.0, 1.0f32);
    let cam = Camera::look_at(Vec3::new(16.0, 16.0, 0.0), Vec3::new(16.0, 16.0, 24.0), 1.0);
    let img = raymarch::<f32, BSX, N>(&g, &cam, 64, 64, &default_opts(&g));

    // Center ray (d = +z) hits the sphere.
    assert!(!is_background(img.get_pixel(32, 32)));
    // A right-of-center pixel: `right = cross(+z, +y) = -x`, so ndcX > 0 gives a
    // negative-x ray; it must still reach the sphere instead of dying at t = 0.
    assert!(
        !is_background(img.get_pixel(36, 32)),
        "negative-x ray killed at the origin block boundary"
    );
}

#[test]
fn render_10_all_empty_grid() {
    // No blocks allocated: every voxel is the empty sentinel.
    let g = grid::<f32>(32, 32, 32);
    let img = render_slice::<f32, BSX, N, IP_LINEAR>(
        &g, SliceAxis::Z, 16.0, 32, 32, BoundaryCondition::Clamp, greyscale,
    );
    assert!(img.pixels().all(|p| p[0] == 0));

    let cam = Camera::look_at(Vec3::new(16.0, 16.0, 0.0), Vec3::new(16.0, 16.0, 16.0), 1.0);
    let frame = raymarch::<f32, BSX, N>(&g, &cam, 64, 64, &default_opts(&g));
    assert!(frame.pixels().all(is_background));
}

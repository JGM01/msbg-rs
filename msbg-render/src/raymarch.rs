//! Empty-space-skipping isosurface raymarcher.
//!
//! The C++ `RaymarchRenderer` marches every ray with a fixed step and does a
//! per-sample `isEmptyBlock` early-out that returns `0` *without advancing* —
//! so an empty region is still micro-stepped one voxel at a time. This renderer
//! instead walks the `BSX³` block lattice with an Amanatides–Woo DDA and skips
//! an empty block in O(1) by jumping to the ray's exit plane of that block,
//! only fine-stepping inside value blocks (where the iso surface lives).
//!
//! Iso detection and shading use a unified linear sample + linear gradient
//! (the C++ splits trilinear detection vs cubic shading — a wart, not a
//! feature). Samples are taken in *physical* density space (`RenderElem`), so
//! `Density8`'s sqrt compression is undone before the `iso_level` comparison.

use image::{ImageBuffer, Rgba};
use msbg_rs::channel::Vec3;
use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sampler};
use msbg_rs::sparse_grid::SparseGrid;
use rayon::prelude::*;

use crate::camera::{normalize, Camera, Ray};
use crate::render_elem::RenderElem;

#[derive(Clone, Copy, Debug)]
pub struct RenderOptions {
    /// Physical-density threshold for the surface (`0.5` in the demo).
    pub iso_level: f32,
    /// Fine-march step inside value blocks, in voxels (`1.0` matches C++).
    pub step_size: f32,
    /// Maximum ray length in voxels (C++ `MAX_DIST · sxyzMax`).
    pub max_dist: f32,
    pub background: Rgba<u8>,
    pub surface_color: [f32; 3],
    /// Light position in voxel space.
    pub sun: Vec3,
    /// Blinn-Phong specular strength (`0` = pure Lambert, the C++ model).
    pub specular_strength: f32,
    pub shininess: f32,
    /// Disable block skipping — force the fixed-step march through every block
    /// (the reference used by the ESS-equivalence test).
    pub disable_ess: bool,
}

#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[inline(always)]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Ray / axis-aligned-box intersection, returning `(t_enter, t_exit)` with
/// `t_enter >= 0`. `None` when the ray misses the box. `dir == 0` axes are
/// handled by an inside/outside slab test (no division).
fn aabb_clip(o: [f32; 3], d: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> Option<(f32, f32)> {
    let mut t0 = f32::NEG_INFINITY;
    let mut t1 = f32::INFINITY;
    for i in 0..3 {
        if d[i].abs() < 1e-30 {
            if o[i] < bmin[i] || o[i] > bmax[i] {
                return None;
            }
        } else {
            let inv = 1.0 / d[i];
            let mut tn = (bmin[i] - o[i]) * inv;
            let mut tf = (bmax[i] - o[i]) * inv;
            if tn > tf {
                std::mem::swap(&mut tn, &mut tf);
            }
            t0 = fmax(t0, tn);
            t1 = fmin(t1, tf);
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((fmax(t0, 0.0), t1))
}

/// Block-lattice DDA state. `t` is the ray parameter of the current block's
/// entry; `t_max[i]` is the parameter of the next block-boundary crossing along
/// axis `i`.
struct BlockDda {
    b: [i32; 3],
    t: f32,
    t_max: [f32; 3],
    t_delta: [f32; 3],
    step: [i32; 3],
    nx: i32,
    ny: i32,
    nz: i32,
}

impl BlockDda {
    fn new(
        o: [f32; 3],
        d: [f32; 3],
        t0: f32,
        nx: usize,
        ny: usize,
        nz: usize,
        bsx: usize,
    ) -> Self {
        let bsxf = bsx as f32;
        let mut p = [0.0f32; 3];
        for i in 0..3 {
            p[i] = o[i] + d[i] * t0;
        }
        let mut b = [0i32; 3];
        b[0] = ((p[0] / bsxf).floor() as i32).clamp(0, nx as i32 - 1);
        b[1] = ((p[1] / bsxf).floor() as i32).clamp(0, ny as i32 - 1);
        b[2] = ((p[2] / bsxf).floor() as i32).clamp(0, nz as i32 - 1);
        let mut step = [0i32; 3];
        let mut t_max = [f32::INFINITY; 3];
        let mut t_delta = [f32::INFINITY; 3];
        for i in 0..3 {
            let pos = p[i];
            if d[i] > 0.0 {
                step[i] = 1;
                let next = (b[i] + 1) as f32 * bsxf;
                t_max[i] = (next - pos) / d[i];
                t_delta[i] = bsxf / d[i];
            } else if d[i] < 0.0 {
                step[i] = -1;
                let next = b[i] as f32 * bsxf;
                t_max[i] = (next - pos) / d[i];
                t_delta[i] = bsxf / (-d[i]);
            }
        }
        BlockDda {
            b,
            t: t0,
            t_max,
            t_delta,
            step,
            nx: nx as i32,
            ny: ny as i32,
            nz: nz as i32,
        }
    }

    #[inline(always)]
    fn bid(&self) -> usize {
        (self.b[0] + self.b[1] * self.nx + self.b[2] * self.nx * self.ny) as usize
    }

    #[inline(always)]
    fn in_bounds(&self) -> bool {
        self.b[0] >= 0
            && self.b[0] < self.nx
            && self.b[1] >= 0
            && self.b[1] < self.ny
            && self.b[2] >= 0
            && self.b[2] < self.nz
    }

    fn exit_t(&self) -> f32 {
        fmin(self.t_max[0], fmin(self.t_max[1], self.t_max[2]))
    }

    /// Advance `t` to the current block's exit plane and move one block along
    /// the exiting axis.
    fn advance(&mut self) {
        let (axis, texit) = if self.t_max[0] <= self.t_max[1] && self.t_max[0] <= self.t_max[2] {
            (0, self.t_max[0])
        } else if self.t_max[1] <= self.t_max[2] {
            (1, self.t_max[1])
        } else {
            (2, self.t_max[2])
        };
        self.t = texit;
        self.b[axis] += self.step[axis];
        self.t_max[axis] += self.t_delta[axis];
    }
}

#[inline(always)]
fn sample_physical<T, const BSX: usize, const N: usize>(
    sampler: &Sampler<T, BSX, N>,
    p: [f32; 3],
) -> f32
where
    T: RenderElem,
{
    T::physical(sampler.sample::<{ Interpolation::Linear }>(Vec3::new(p[0], p[1], p[2])))
}

/// Trace one ray; returns the isosurface hit position, or `None`.
fn trace<T, const BSX: usize, const N: usize>(
    grid: &SparseGrid<T, BSX, N>,
    sampler: &Sampler<T, BSX, N>,
    ray: &Ray,
    opts: &RenderOptions,
) -> Option<Vec3>
where
    T: RenderElem,
{
    let o = [ray.origin.x(), ray.origin.y(), ray.origin.z()];
    let d = [ray.direction.x(), ray.direction.y(), ray.direction.z()];
    let bmin = [0.0f32; 3];
    let bmax = [grid.sx as f32, grid.sy as f32, grid.sz as f32];

    let (t0, t1) = aabb_clip(o, d, bmin, bmax)?;
    let t_limit = fmin(t1, opts.max_dist);
    if t0 > t_limit {
        return None;
    }

    let mut dda = BlockDda::new(o, d, t0, grid.nx, grid.ny, grid.nz, BSX);
    let p = [o[0] + d[0] * t0, o[1] + d[1] * t0, o[2] + d[2] * t0];
    let mut prev_d = sample_physical(sampler, p);
    let mut prev_p = p;

    loop {
        if !dda.in_bounds() || dda.t >= t_limit {
            return None;
        }
        let bid = dda.bid();
        if !opts.disable_ess && grid.is_empty_block(bid) {
            let exit_t = dda.exit_t();
            if exit_t >= t_limit {
                return None;
            }
            // Skip the whole empty block. `advance()` handles a ray sitting
            // exactly on a block boundary (`t_max == t`): it moves to the next
            // block with no `t` change rather than stalling or dying.
            dda.advance();
            prev_d = 0.0;
            prev_p = [o[0] + d[0] * dda.t, o[1] + d[1] * dda.t, o[2] + d[2] * dda.t];
            continue;
        }

        let texit = fmin(dda.exit_t(), t_limit);
        while dda.t < texit {
            let tnext = fmin(dda.t + opts.step_size, texit);
            let pn = [o[0] + d[0] * tnext, o[1] + d[1] * tnext, o[2] + d[2] * tnext];
            let dn = sample_physical(sampler, pn);
            if prev_d < opts.iso_level && dn >= opts.iso_level {
                let alpha = (opts.iso_level - prev_d) / (dn - prev_d);
                return Some(Vec3::new(
                    prev_p[0] + (pn[0] - prev_p[0]) * alpha,
                    prev_p[1] + (pn[1] - prev_p[1]) * alpha,
                    prev_p[2] + (pn[2] - prev_p[2]) * alpha,
                ));
            }
            prev_d = dn;
            prev_p = pn;
            dda.t = tnext;
        }
        if dda.t >= t_limit {
            return None;
        }
        dda.advance();
    }
}

#[inline(always)]
fn shade<T, const BSX: usize, const N: usize>(
    sampler: &Sampler<T, BSX, N>,
    hit: Vec3,
    ray_dir: Vec3,
    opts: &RenderOptions,
) -> Rgba<u8>
where
    T: RenderElem,
{
    let grad = sampler.gradient::<{ Interpolation::Linear }>(hit);
    let normal = normalize(Vec3::new(-grad.x(), -grad.y(), -grad.z()));
    let light_dir = normalize(opts.sun - hit);
    let diff = normal.dot(light_dir).max(0.0);

    let mut r = opts.surface_color[0] * diff;
    let mut g = opts.surface_color[1] * diff;
    let mut b = opts.surface_color[2] * diff;
    if opts.specular_strength > 0.0 {
        let view = normalize(Vec3::new(-ray_dir.x(), -ray_dir.y(), -ray_dir.z()));
        let half = normalize(light_dir + view);
        let spec = normal.dot(half).max(0.0).powf(opts.shininess) * opts.specular_strength;
        r += spec;
        g += spec;
        b += spec;
    }

    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Rgba([to_u8(r), to_u8(g), to_u8(b), 255])
}

/// Render `width × height` perspective image of `grid` (Rayon over rows).
pub fn raymarch<T, const BSX: usize, const N: usize>(
    grid: &SparseGrid<T, BSX, N>,
    camera: &Camera,
    width: usize,
    height: usize,
    opts: &RenderOptions,
) -> ImageBuffer<Rgba<u8>, Vec<u8>>
where
    T: RenderElem,
{
    let sampler = Sampler::new(grid, GridAlignment::Corner, BoundaryCondition::Dirichlet);
    let mut buf = vec![0u8; width * height * 4];
    let wf = width as f32;
    let hf = height as f32;
    buf.par_chunks_exact_mut(width * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let yf = y as f32;
            for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let ray = camera.ray(x as f32, yf, wf, hf);
                *px = match trace(grid, &sampler, &ray, opts) {
                    Some(hit) => shade(&sampler, hit, ray.direction, opts).0,
                    None => opts.background.0,
                };
            }
        });
    ImageBuffer::from_raw(width as u32, height as u32, buf).expect("raymarch buffer size")
}

//! O(N²) orthogonal slice extraction.
//!
//! Unlike the C++ `getSlices2D` — which scans every voxel in the domain and
//! tests membership against the three slice planes — this iterates directly
//! over the output pixels and samples the field once per pixel (Rayon over
//! rows). A `512³` slice costs `512²` samples instead of `512³` scans.

use image::{ImageBuffer, Rgba};
use msbg_rs::channel::Vec3;
use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sampler};
use msbg_rs::sparse_grid::SparseGrid;
use rayon::prelude::*;

use crate::render_elem::RenderElem;

/// The axis normal to the slice plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceAxis {
    X,
    Y,
    Z,
}

/// Extract an orthogonal slice at plane coordinate `offset` (continuous voxel
/// space; pixel `(u, v)` maps to the plane's two free axes). `color` maps the
/// physical `[0, 1]` density to a pixel. `bc` selects out-of-domain behavior
/// for offsets past the grid edge.
pub fn render_slice<T, const BSX: usize, const N: usize, const IP: Interpolation>(
    grid: &SparseGrid<T, BSX, N>,
    axis: SliceAxis,
    offset: f32,
    out_w: usize,
    out_h: usize,
    bc: BoundaryCondition,
    color: fn(f32) -> Rgba<u8>,
) -> ImageBuffer<Rgba<u8>, Vec<u8>>
where
    T: RenderElem,
{
    let sampler = Sampler::new(grid, GridAlignment::Corner, bc);
    let mut buf = vec![0u8; out_w * out_h * 4];
    buf.par_chunks_exact_mut(out_w * 4)
        .enumerate()
        .for_each(|(v, row)| {
            let v = v as f32;
            for (u, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let u = u as f32;
                let pos = match axis {
                    SliceAxis::Z => Vec3::new(u, v, offset),
                    SliceAxis::Y => Vec3::new(u, offset, v),
                    SliceAxis::X => Vec3::new(offset, u, v),
                };
                let d = sampler.sample::<IP>(pos);
                *px = color(T::physical(d)).0;
            }
        });
    ImageBuffer::from_raw(out_w as u32, out_h as u32, buf).expect("slice buffer size")
}

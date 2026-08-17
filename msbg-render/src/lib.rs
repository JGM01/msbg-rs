//! Offline rendering over [`msbg_rs`] public API: 2D orthogonal slicing and a
//! 3D empty-space-skipping isosurface raymarcher.
//!
//! This crate depends only on `msbg_rs`'s `pub` surface (`SparseGrid`,
//! `Sampler`, the channel types) — it is the step-10 "end-user" of the library
//! and exists in part to prove that surface is usable from outside the crate.

#![feature(adt_const_params)]

pub mod camera;
pub mod colormap;
pub mod raymarch;
pub mod render_elem;
pub mod slice;

pub use camera::{Camera, Ray};
pub use colormap::{greyscale, turbo};
pub use raymarch::{raymarch, RenderOptions};
pub use render_elem::RenderElem;
pub use slice::{render_slice, SliceAxis};

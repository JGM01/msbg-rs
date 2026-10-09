//! Offline slices and isosurface raymarching through the [`msbg_rs`] public API.

#![feature(adt_const_params)]

pub mod camera;
pub mod colormap;
pub mod raymarch;
pub mod render_elem;
pub mod slice;

pub use camera::{Camera, Ray};
pub use colormap::{greyscale, turbo};
pub use raymarch::{RenderOptions, raymarch};
pub use render_elem::RenderElem;
pub use slice::{SliceAxis, render_slice};

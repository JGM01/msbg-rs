//! Rust port of *Multiresolution Sparse Block Grids* (SIGGRAPH 2025 phase-field
//! FLIP), with cross-platform SIMD (`std::simd`) instead of x86 intrinsics.
//!
//! Core building blocks:
//!
//! * [`blockpool`] — aligned blocks + a lock-free monotonic allocator.
//! * [`sparse_grid`] — single-resolution `SparseGrid<T>` over blocks.
//! * [`channel`] — typed data channels (density, pressure, velocity, ...).
//! * [`math`] — field sampling/interpolation and SIMD stencil kernels.
//! * [`multires`] — halo gather and block iteration.
//! * [`particles`] — step-9 surface reconstruction (PLY → splat → density).
//! * [`io`] — file-format I/O (PLY).
//! * [`thread_pool`] — rayon workers with FTZ/DAZ enabled.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use msbg_rs::blockpool::BlockPool;
//! use msbg_rs::channel::{Density, Vec3};
//! use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sampler};
//! use msbg_rs::sparse_grid::SparseGrid;
//!
//! let pool = Arc::new(BlockPool::<Density, 16, 4096>::new(1, 16));
//! let mut grid = SparseGrid::new("density".into(), 32, 32, 32, Density(0), Density(u16::MAX), pool);
//! grid.set_voxel(0, 0, 0, Density(u16::MAX));
//!
//! let sampler = Sampler::new(&grid, GridAlignment::Corner, BoundaryCondition::Clamp);
//! let d = sampler.sample::<{ Interpolation::Linear }>(Vec3::new(0.2, 0.0, 0.0));
//! assert!(d > 0.0);
//! ```

#![feature(portable_simd)]
#![feature(adt_const_params)]
#![feature(core_intrinsics)]

pub mod blockmap;
pub mod blockpool;
pub mod channel;
pub mod fluid;
pub mod io;
pub mod math;
pub mod multires;
pub mod particles;
pub mod solver;
pub mod sparse_grid;
pub mod thread_pool;

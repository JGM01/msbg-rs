# msbg-rs

Sparse block grids, SIMD kernels, and particle surface reconstruction in Rust.

This project implements **Multiresolution Sparse Block Grids**
([MSBG](https://github.com/tum-pbs/MSBG)) in Rust. The method comes from
Bernhard Braun, Jan Bender, and Nils Thuerey's SIGGRAPH 2025 paper,
[Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation](https://ge.in.tum.de/download/Adaptive_Phase_Field_FLIP_preprint.pdf).
The work focuses on memory layout, sparse indexing, and parallel execution.
Tests compare numerical results with the C++ reference.

![Surface reconstructed and rendered by the Rust implementation](bob_images/bunny_of_bunnies.png)

*Bunny-of-bunnies reconstruction from this implementation. A recorded run used
1.29 billion particles and over 100 billion active voxels in a 32,768³ domain
on a 256 GB EC2 instance.*

## Features

- Sparse and multiresolution grids with aligned block pools and typed channels.
- Trilinear and cubic B-spline sampling, gradients, and Hessians.
- SIMD Laplacian, mean-curvature, and bi-Laplacian stencils.
- Particle placement, sparse bucketing, eight-color splatting, and density smoothing.
- Orthogonal slices and isosurface raymarching with empty-space skipping.
- MAC-grid particle transfer, FLIP/PIC gathering, and divergence calculation.

The library is experimental. Surface reconstruction and rendering work.
The pressure solver and complete fluid-simulation timestep are planned.

## Build and test

Install [Rust through rustup](https://rustup.rs/). The repository selects
`nightly-2026-06-18` for portable SIMD and const-generic features.
Development and benchmarks have used Linux x86-64, macOS ARM64, and Windows x86-64.

```sh
cargo run --release --example sparse_field
cargo test --workspace --lib
cargo test -p msbg-render --test render_tests
cargo test --workspace --doc
```

These commands need no external data. The example constructs and samples a sparse field.

The full test suite also needs the reference bunny data:

```sh
git clone https://github.com/tum-pbs/MSBG.git ../MSBG
cargo test --workspace --locked
```

Set `MSBG_DATA_DIR` if the data is in another directory.
Live C++ comparisons also need compiled reference binaries.
See [development.md](docs/development.md) for details.

## Render the bunny

After obtaining the data:

```sh
cargo run -p msbg-render --release --example render_bunny -- \
  ../MSBG/data/bun_zipper_res2.ply 256 256 256 1 0.25 target/render-bunny
```

This writes `raymarch.png` and orthogonal slices to `target/render-bunny`.
Use this small workload before running larger reconstructions.

## Performance

Selected measurements from development:

| Workload | Baseline | Rust | Configuration |
| --- | --- | --- | --- |
| Reconstruction + six smoothing sweeps | C++: 394.2 ms | 149.4 ms | Ryzen 5500U, 512³, 522,944 particles |
| Particle splatting | C++: 379.6 ms | 110.6 ms | Same workload |
| Field allocation + initialization | Earlier Rust: 110.7 s | 16.3 s | EC2, approximately 215 GB field |
| Mean-curvature smoothing | — | Over 12 billion voxel updates/s | 32-core EC2, 32³ blocks, over 100B active voxels |

Reconstruction took 62% less time than C++ in the laptop comparison.
Each Rust/C++ pair ran on the same hardware and workload.
The values come from saved development notes; the raw timing logs were not retained.
The paper's published measurements are separate.

The implementation uses private particle buffers, sparse particle metadata,
cached halo neighbors, and parallel memory initialization.
See [benchmarks.md](docs/benchmarks.md) for commands, configurations, and comparison limits.

## Design and tests

Stencil kernels select SIMD widths for x86-64 and ARM64.
Quantized channels reduce storage. Eight-color scheduling supports in-place updates.
The x86 `f32` path uses streaming stores; ARM64 uses ordinary stores.

Tests cover analytic fields, boundaries, serial references, C++ goldens, and rendering.
Live C++ tests are optional. Numerical tolerances depend on the operation.
Storage and parallel kernels use `unsafe`; API safety needs further review.

- [Architecture](docs/refactor.md)
- [Development and tests](docs/development.md)
- [Benchmarks](docs/benchmarks.md)
- [Status and planned work](docs/roadmap.md)

The grid method, eight-color scheme, and reconstruction reference come from the MSBG authors.
This repository contains the Rust implementation and its storage and scheduling changes.
Bunny data comes from the Stanford dataset included in MSBG.

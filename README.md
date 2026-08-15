# msbg-rs

A Rust port of [MSBG](https://github.com/tum-pbs/MSBG/) — *Multiresolution
Sparse Block Grids*, the data structure behind a 2025 phase-field FLIP
paper. The port targets feature parity with the C++ library while improving the
public API and replacing x86-only intrinsics with `std::simd`, so it builds and
runs on macOS (arm64) as well as Linux. I would like it to run on Windows as 
well but that is secondary.

## Status

In progress, tracked dependency-by-dependency in [`docs/roadmap.md`](docs/roadmap.md).

## Quick start

Nightly Rust is required (pinned by `rust-toolchain.toml`). On Linux, the
`nix-shell` provides the toolchain:

```bash
nix-shell

cargo build
cargo test
cargo bench --bench allocator_benches
cargo bench --bench interp_bench
```

The C++ baseline msut live in `../MSBG` and is used for differential tests and
benchmark comparisons; see the two `AGENTS.md` files for how to build and run it.
If you do not have the original `MSBG` repository in the same directory as this
one, you will not be able to run many of the difftests & other comparison tools.

## Layout

- `src/blockpool.rs` — aligned blocks and a lock-free monotonic allocator
- `src/sparse_grid.rs` — single-level `SparseGrid<T>`
- `src/channel.rs` — typed data channels (density, pressure, velocity, ...)
- `src/math/` — field sampling/interpolation and SIMD stencil kernels
- `src/multires/` — halo gather and block iteration
- `src/thread_pool.rs` — rayon workers with FTZ/DAZ enabled

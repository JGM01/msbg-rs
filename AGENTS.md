# Working on msbg-rs

This is a Rust implementation of the SIGGRAPH 2025 MSBG grid method.
The C++ reference is optional. Rust builds and tests use the included fixtures.
Read `README.md` and the relevant source before editing.

## Build and test

Use rustup and the toolchain in `rust-toolchain.toml`.
Stable Rust cannot build the SIMD and const-generic code.
Nix is optional and Linux-specific.

```sh
cargo fmt --all -- --check
cargo test --workspace --lib --locked
cargo test -p msbg-render --test render_tests --locked
cargo test --workspace --doc --locked
```

The full suite uses the PLY fixtures in `data/`.
Tests and benchmarks resolve fixture paths from the crate location.
Use `MSBG_DATA_DIR` to override the fixture directory.
Live C++ tests return early when their binary variables are absent.
Do not count those returns as executed C++ comparisons.
See `docs/development.md` for details.

## Code map

- `src/blockpool.rs`, `src/blockmap.rs`, `src/sparse_grid.rs`: storage.
- `src/channel.rs`: channels and quantization.
- `src/math/`: sampling and SIMD kernels.
- `src/multires/`: topology, halos, levels, and restriction.
- `src/solver.rs`: single-resolution smoothing.
- `src/particles/`: placement, bucketing, and reconstruction.
- `src/fluid/`: MAC transfer and divergence.
- `msbg-render/`: rendering through the public core API.

## Code and prose

- Justify unsafe operations and state their invariants beside the code.
- Enforce safety-critical preconditions in release builds of safe public APIs.
- Review initialization, aliasing, lifetimes, alignment, padding, and parallel ownership.
- Do not use `std::thread_local!` inside Rayon closures.
- Choose explicit chunks, worker buffers, or fold/reduce to suit the workload.
  A fold accumulator can be created more than once per worker.
- Use `thread_pool::Pool` when x86 FTZ/DAZ settings matter.
- Comments should explain invariants and decisions.

## Benchmarks

Start with `MSBG_BENCH_SCALE=small`. Stress workloads can need hundreds of gigabytes.
Check output before timing.
Match hardware, data, precision, block size, scheduling, compiler features, and thread counts.
Some later C++ harnesses are absent from the current checkout.
Recover them before attempting those comparisons.

Retain raw results and the run configuration.
Separate memory estimates from RSS, and Rust measurements from paper results.
See `docs/benchmarks.md` for commands and recorded results.

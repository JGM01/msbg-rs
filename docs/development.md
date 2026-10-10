# Development and tests

## Toolchain

Use rustup. `rust-toolchain.toml` selects `nightly-2026-06-18` and rustfmt.
The crate uses portable SIMD and const-generic features.
Nix is optional and supplies Linux profiling tools.

```sh
cargo build --workspace --locked
cargo fmt --all -- --check
cargo doc --workspace --no-deps --locked
```

If builds use another Rust installation, check `rustc --version`, `cargo --version`, and `PATH`.
Use rustup for the compiler, documentation, and lint tools.

Local builds use `-C target-cpu=native`.
Binaries may require CPU features that are absent on other machines.
Record target features in benchmark results.

## Tests

```sh
cargo test --workspace --locked
```

The clone contains both bunny PLY files in `data/` and the C++ golden values in the tests.
No C++ checkout, binary, or data download is required.
Tests cover storage, kernels, boundaries, fluid transfer, rendering, and golden comparisons.

Tests and benchmarks resolve data paths from the crate location.
Set `MSBG_DATA_DIR` to use another fixture directory.
Examples take a PLY path as their first argument.
See [data/README.md](../data/README.md) for fixture provenance and use terms.

CI tests the Rust workspace on Linux, macOS, and Windows using the included data.
It does not run live C++ comparisons or benchmarks.

## C++ comparisons

Each integration-test file has a golden test and an optional live test:

| Test | Golden check | Live binary variable |
| --- | --- | --- |
| `difftest_cpp` | Exact FNV hash of grid reads/writes | `MSBG_CPP_DIFFTEST_BIN` |
| `difftest_interp` | Values, gradients, Hessians; tolerance 1e-4 | `MSBG_CPP_INTERTEST_BIN` |
| `difftest_stencil` | Stencil samples; tolerance 1e-4 | `MSBG_CPP_MEANCURVTEST_BIN` |
| `difftest_smoother` | Laplacian 1e-4, mean curvature 1e-3 | `MSBG_CPP_SMOOTHERTEST_BIN` |
| `difftest_splat` | Reconstructed and smoothed `u16` samples | `MSBG_CPP_SPLATTEST_BIN` |

Live tests return early if the binary variable is absent.
Use `-- --nocapture` to see the skip messages.

For live comparisons, obtain the development fork with the custom harnesses:

```sh
git clone https://github.com/JGM01/MSBG.git /path/to/MSBG
```

[The original repository](https://github.com/tum-pbs/MSBG) contains the research implementation.
[The development fork](https://github.com/JGM01/MSBG) also contains custom test and benchmark harnesses.
The development fork includes Linux build scripts for storage, interpolation, and stencil tests.
Build a matching harness and pass its absolute path:

```sh
MSBG_CPP_DIFFTEST_BIN=/path/to/MSBG/build/difftest \
  cargo test --test difftest_cpp -- --nocapture
```

The checkout can be anywhere. The environment variables select the binaries;
Rust tests do not depend on a sibling directory named `MSBG`.
The current development fork lacks the later smoother and reconstruction harness sources.
Recover the matching revision to run those live comparisons.

The live reconstruction test checks full fields.
It allows at most two `u16` code steps and at most 0.1% of entries to differ.
The golden test checks selected samples only.
These code-space limits do not specify relative physical error.

## Profiling and unsafe code

`profile-linux.sh` uses `perf` and Inferno.
`profile-macos.sh` uses the native `sample` tool and has limited validation.
Both write to `target/profile`.

```sh
cargo bench --bench allocator_benches --no-run
./profile-linux.sh halo_gather
```

For Miri, install the optional components with `rustup component add miri rust-src`.
Check support for the SIMD and architecture-specific operations used by the test.

Document unsafe invariants beside the operation.
Safe public APIs must enforce safety-critical preconditions in release builds.
Review layout, initialization, aliasing, lifetimes, and parallel ownership after changes.
Run numerical regression tests before measuring performance.

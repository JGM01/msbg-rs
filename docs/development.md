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

## Tests without external data

```sh
cargo test --workspace --lib --locked
cargo test -p msbg-render --test render_tests --locked
cargo test --workspace --doc --locked
cargo test --test difftest_cpp --test difftest_interp \
  --test difftest_stencil --test difftest_smoother --locked
```

These test storage, kernels, boundaries, fluid transfer, rendering, and embedded C++ goldens.

## Full suite

The reconstruction golden test needs the Stanford bunny fixture:

```sh
git clone https://github.com/tum-pbs/MSBG.git ../MSBG
cargo test --workspace --locked
```

`tests/difftest_splat.rs` reads `../MSBG/data/bun_zipper_res2.ply`.
Set `MSBG_DATA_DIR` if the file is elsewhere.
The golden test needs no C++ binary.

CI fetches the data at a fixed revision of the companion repository.
It runs Rust tests on Linux, macOS, and Windows.
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

The local C++ checkout has Linux build scripts for storage, interpolation, and stencil tests.
Build the binary there, then pass its absolute path:

```sh
MSBG_CPP_DIFFTEST_BIN=/absolute/path/to/MSBG/build/difftest \
  cargo test --test difftest_cpp -- --nocapture
```

The current checkout lacks the later smoother and reconstruction harness sources.
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

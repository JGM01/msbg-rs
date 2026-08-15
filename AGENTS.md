# AGENTS.md — msbg-rs (Rust port)

Rust port of *Multiresolution Sparse Block Grids* (SIGGRAPH 2025 phase-field
FLIP paper). The reference C++ implementation lives in `../MSBG` and is the
baseline for benchmarks and the differential test.

## Toolchain

**Nightly Rust is required** — `src/lib.rs` uses `#![feature(portable_simd)]`
and `#![feature(adt_const_params)]`.

```bash
cd /home/jacob/programming/msbg-rs
nix-shell   # provides rustup + gcc; rust-toolchain.toml pins `nightly`
```

`rustup` auto-installs/selects the pinned nightly the first time `cargo` runs.
All `rustc`/`cargo` invocations below assume this shell.

**Platform: portable.** Unlike the C++ baseline (`../MSBG`), this crate compiles
and runs on macOS (including arm64/Apple Silicon) as well as Linux x86-64, since
it uses `std::simd` (`portable_simd`) rather than x86-only intrinsics. The `nix`
shell here is Linux-specific, but `cargo build`/`test`/`bench` work on macOS with
a plain nightly toolchain.

## Layout

- `src/blockpool.rs` — aligned `Block` + lock-free monotonic `BlockPool`
- `src/sparse_grid.rs` — single-level `SparseGrid<T>`
- `src/multires/` — `halo.rs` (18³ gather), `iterator.rs`
- `src/math/` — `sample.rs` (interpolation), `bspline.rs`, `gather.rs`,
  `boundary.rs`, `laplacian.rs` (SIMD kernels)
- `src/thread_pool.rs` — rayon pool with FTZ/DAZ workers
- `benches/allocator_benches.rs` — criterion benches (scenarios A–E)
- `benches/interp_bench.rs` — field-sampling benches (scenario G)
- `tests/difftest_cpp.rs` — differential test against the C++ baseline
- `tests/difftest_interp.rs` — interpolation difftest vs `../MSBG/interptest.cpp`
- `docs/roadmap.md` — feature-parity plan; `docs/refactor.md` — design notes

## Build & test

```bash
cargo build
cargo test                      # unit tests + the difftest hash check
cargo bench --bench allocator_benches
cargo bench --bench interp_bench
```

## Benchmarks

Two sizes via `MSBG_BENCH_SCALE` (default = full):

```bash
MSBG_BENCH_SCALE=small cargo bench --bench allocator_benches   # fast debug cycles
cargo bench --bench allocator_benches                          # full; ~100k blocks
```

Full runs allocate ~100k active blocks — machines with < 7 GB RAM may OOM.
The benches mirror `../MSBG/benchmark.cpp` scenarios A–E; see `../MSBG/AGENTS.md`
§8 for the known asymmetries (Rust halo fill is a simplified copy, etc.).

`benches/interp_bench.rs` (field-sampling throughput) mirrors scenario G
(`../MSBG/benchmark.cpp interp`); see `docs/refactor.md` §7 for the
interpolation design notes and how the numbers compare.

## Differential test vs C++

`tests/difftest_cpp.rs` runs the same fixed write/read script as the C++
`difftest.cpp` and folds every voxel's f32 bits into an FNV-1a hash.

1. Build the C++ side (in the MSBG shell):
   ```bash
   cd ../MSBG && nix-shell && ./build_difftest.sh && exit
   ```
2. Run the live comparison (in this shell):
   ```bash
   MSBG_CPP_DIFFTEST_BIN="$PWD/../MSBG/build/difftest" cargo test --test difftest_cpp
   ```

Two tests run:
- `difftest_matches_cpp_hash` — compares the native hash to the hardcoded
  `CPP_HASH` (no C++ binary needed).
- `difftest_against_cpp_binary_if_available` — skipped unless
  `MSBG_CPP_DIFFTEST_BIN` is set; runs the binary and compares its output.

### Regenerating `CPP_HASH`

If the C++ grid layout changes, rebuild `difftest`, capture its output, and
paste it into the `CPP_HASH` const in `tests/difftest_cpp.rs`:

```bash
cd ../MSBG && nix-shell --run './build_difftest.sh && ./build/difftest'
```

### Interpolation difftest (`tests/difftest_interp.rs`)

Compares sampled value/gradient/Hessian against `../MSBG/interptest.cpp`
within 1e-4 (tolerance-based; not bit-exact). Golden values are hardcoded, so
no C++ binary is needed unless you want the live check:

```bash
cd ../MSBG && nix-shell && ./build_interptest.sh && exit
MSBG_CPP_INTERTEST_BIN="$PWD/../MSBG/build/interptest" cargo test --test difftest_interp
```

## Debugging

The shell ships both Rust-specific tooling (`cargo miri`, `cargo asm`,
`cargo bloat`, `cargo llvm-lines`, `cargo flamegraph`, `cargo geiger`,
`cargo nextest`) and the ELF-level tools (`gdb`, `perf`, `llvm-mca`, `pahole`,
`valgrind`, `rr`, `heaptrack`, `strace`) — the latter also work on the C++
binaries in `../MSBG`. See `.agents/skills/debugging/SKILL.md` for what each is
for and example invocations, and `.agents/skills/compare-cpp-rust/SKILL.md` for
the side-by-side C++-vs-Rust comparison workflow (spawn a subagent per repo,
aggregate IPC/cache/code-size/layout).
- `./profile-linux.sh <bench-group>` (Linux) / `./profile-macos.sh <bench-group>`
  profile one benchmark and write `target/profile/profile.txt` (flat top-N
  self-time) + `flamegraph.svg`.

## Conventions

- Unsafe is only acceptable when performance parity or improvement depends on it. Additionally,
  You must `debug_assert` the conditions that prove the Unsafe is fine.
- Don't use `std::thread_local!` inside rayon closures (use `.fold().reduce()`).
- FTZ/DAZ is enabled per-worker via `thread_pool::Pool` (see `docs/refactor.md`).
- No comments unless they explain *why*; keep names self-documenting.
- `//!` doc comments are API docs: state purpose and show a usage example —
  no roadmap status, C++-internal names, or "what we dropped/rejected" notes
  (that belongs in `docs/refactor.md` or `docs/roadmap.md`).

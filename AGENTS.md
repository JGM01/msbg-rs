# AGENTS.md — msbg-rs (Rust port)

Rust port of *Multiresolution Sparse Block Grids* (SIGGRAPH 2025 phase-field
FLIP paper). The reference C++ implementation lives in `../MSBG` and is the
baseline for benchmarks and the differential test.

## Toolchain

**Nightly Rust is required** — `src/lib.rs` uses `#![feature(portable_simd)]`.

```bash
cd /home/jacob/programming/msbg-rs
nix-shell   # provides rustup + gcc; rust-toolchain.toml pins `nightly`
```

`rustup` auto-installs/selects the pinned nightly the first time `cargo` runs.
All `rustc`/`cargo` invocations below assume this shell.

## Layout

- `src/blockpool.rs` — aligned `Block` + lock-free monotonic `BlockPool`
- `src/sparse_grid.rs` — single-level `SparseGrid<T>`
- `src/multires/` — `halo.rs` (18³ gather), `iterator.rs`
- `src/math/` — `laplacian.rs` (SIMD kernels), `interpolate.rs`
- `src/thread_pool.rs` — rayon pool with FTZ/DAZ workers
- `benches/allocator_benches.rs` — criterion benches (scenarios A–E)
- `tests/difftest_cpp.rs` — differential test against the C++ baseline
- `docs/roadmap.md` — feature-parity plan; `docs/refactor.md` — design notes

## Build & test

```bash
cargo build
cargo test                      # unit tests + the difftest hash check
cargo bench --bench allocator_benches
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

## Debugging

The shell ships both Rust-specific tooling (`cargo miri`, `cargo asm`,
`cargo bloat`, `cargo llvm-lines`, `cargo flamegraph`, `cargo geiger`,
`cargo nextest`) and the ELF-level tools (`gdb`, `perf`, `llvm-mca`, `pahole`,
`valgrind`, `rr`, `heaptrack`, `strace`) — the latter also work on the C++
binaries in `../MSBG`. See `.agents/skills/debugging/SKILL.md` for what each is
for and example invocations, and `.agents/skills/compare-cpp-rust/SKILL.md` for
the side-by-side C++-vs-Rust comparison workflow (spawn a subagent per repo,
aggregate IPC/cache/code-size/layout).

## Conventions

- Unsafe is confined to the allocator and inner SIMD loops; keep it there.
- Don't use `std::thread_local!` inside rayon closures (use `.fold().reduce()`).
- FTZ/DAZ is enabled per-worker via `thread_pool::Pool` (see `docs/refactor.md`).
- No comments unless they explain *why*; keep names self-documenting.

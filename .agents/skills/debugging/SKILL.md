---
name: debugging
description: Debug and profile the msbg-rs Rust library, and compare instruction-level behavior with the C++ MSBG baseline. Use when a test fails, when chasing a performance or correctness discrepancy against ../MSBG, or when auditing the unsafe code in blockpool/sparse_grid.
---

# msbg-rs debugging

All tools live in the nix-shell (see `shell.nix`); nightly Rust is pinned by
`rust-toolchain.toml`. Build/test with `cargo build` / `cargo test`.

## Tool cheat-sheet

| Tool | Purpose |
|------|---------|
| `cargo miri` | UB detection in unsafe code (blockpool/sparse_grid) |
| `cargo asm` (`cargo-show-asm`) | source-level disassembly of a function |
| `cargo bloat` | what's taking space in the binary |
| `cargo llvm-lines` | monomorphization / compile-time cost |
| `cargo flamegraph` | perf-based flame graph |
| `cargo geiger` | audit unsafe-code usage |
| `cargo nextest` | fast parallel test runner |
| `-Zprint-type-sizes` | struct layout (size/align/offsets) |
| `-Z sanitizer=address\|thread` | ASan / TSan on the unsafe code |
| `gdb` / `rust-gdb` | backtraces, breakpoints |
| `perf` | counters, `record`/`annotate`, false sharing (`c2c`) |
| `llvm-objdump --demangle` | disassembly (demangles Rust v0 names) |
| `llvm-cxxfilt` | demangle a single Rust symbol |
| `llvm-mca` | static instruction-throughput of a loop |
| `pahole` | struct layout from DWARF (cross-check vs C++) |
| `valgrind` | `cachegrind`/`callgrind` simulation |
| `rr` | record & replay a crash deterministically |
| `heaptrack` | heap profile |
| `strace` | syscalls, mmap, thread creation |

The ELF tools (`objdump`, `perf`, `llvm-mca`, `pahole`) also work on the C++
`../MSBG` binaries — comparisons in both directions.

## Examples

### UB in unsafe code (miri)
```bash
cargo miri test
# or a single test:
cargo miri test blockpool::block_tests::test_con_01_multithreaded_concurrent_allocations
```
First run compiles std under miri (slow); afterwards only the crate is checked.

### Source-level disassembly (cargo asm)
```bash
cargo asm --lib 'msbg_rs::blockpool::BlockPool<f32,16,4096>::alloc_block'
```
Note: `cargo asm` needs `--lib` (multiple targets) and won't show `#[inline]`
functions — use `llvm-objdump --demangle` (below) for those.

### Struct layout parity vs C++ (Rust side)
```bash
RUSTFLAGS="-Zprint-type-sizes" cargo build 2>&1 | grep -i -A6 'blockpool::Block'
# cross-check against the C++ side (in ../MSBG shell): pahole -C BlockPool
```

### Binary size / monomorphization
```bash
cargo bloat --release -n 10
cargo llvm-lines --release | head -30
```

### Flame graph of a benchmark
```bash
cargo flamegraph --bench allocator_benches
```

### Unsafe audit
```bash
cargo geiger
```

### Fast tests
```bash
cargo nextest run
```

### Sanitizers (ASan / TSan)
```bash
# full coverage needs std built with the sanitizer:
RUSTFLAGS="-Zsanitizer=address" cargo test -Zbuild-std --target x86_64-unknown-linux-gnu
RUSTFLAGS="-Zsanitizer=thread" cargo test -Zbuild-std --target x86_64-unknown-linux-gnu
```

### Backtrace / crash
```bash
BIN=$(find target -name 'allocator_benches-*' -type f -executable | head -1)
rust-gdb "$BIN"               # or plain gdb (rust-gdb adds Rust pretty-printers)
gdb -batch -ex run -ex bt "$BIN"
```

### Replay a flaky crash
```bash
# one-time: sudo sysctl kernel.perf_event_paranoid=1
rr record cargo test --test difftest_cpp
rr replay
```

### Hardware counters / false sharing
```bash
BIN=$(find target/release -name 'allocator_benches-*' -type f -executable | head -1)
perf stat -d "$BIN"
perf c2c record "$BIN" && perf c2c report
```

### Disassemble + demangle one function (works on the C++ binary too)
```bash
cargo bench --bench allocator_benches --no-run
BIN=$(find target/release -name 'allocator_benches-*' -type f -executable | head -1)
llvm-objdump -d --demangle "$BIN" | less
# demangle a single symbol:
nm "$BIN" | grep -oE '_R[A-Za-z0-9_$]+' | head -1 | xargs llvm-cxxfilt
```

### Static throughput of a hot loop
```bash
cargo rustc --lib --release -- --emit=asm
S=$(find target/release/build/msbg-rs -name '*.s' | head -1)
llvm-mca -mcpu=native -iterations=100 "$S"
```
`#[inline]` kernels have no standalone symbol — either `#[inline(never)]` them
temporarily, or extract the inlined loop from the caller's disassembly.

### Heap profile
```bash
BIN=$(find target/release -name 'allocator_benches-*' -type f -executable | head -1)
heaptrack "$BIN"
heaptrack_print heaptrack.allocator_benches.*.zst | head -30
```

## Notes

- `perf` is built for nixpkgs' kernel, which may be newer than the running one
  (`uname -r`); `perf stat`/`record` still work in practice.
- `pahole` needs debug info: build with `RUSTFLAGS="-C debuginfo=2"` (or prefer
  `-Zprint-type-sizes` on the Rust side).
- `cargo miri`, `cargo asm`, and `-Zprint-type-sizes` need nightly — already
  pinned via `rust-toolchain.toml`.
- Rust uses v0 symbol mangling — `nm -C`/`c++filt` won't demangle it; use
  `llvm-nm --demangle`, `llvm-cxxfilt`, or `llvm-objdump --demangle`.
- `#[inline]` kernels have no standalone symbol — `cargo asm`/`nm` can't see
  them; use `llvm-objdump --demangle` on the caller or `#[inline(never)]`.
- Known asymmetries vs C++ (simplified halo fill, fluid mask in the Laplacian
  kernel) are listed in `../MSBG/AGENTS.md` §8 — keep them in mind when a
  benchmark number looks off in either direction.

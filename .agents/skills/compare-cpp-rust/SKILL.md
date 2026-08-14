---
name: compare-cpp-rust
description: Compare this Rust msbg-rs library against the C++ MSBG baseline (../MSBG) — IPC/cache performance and code size. Use when asked whether the Rust refactor is faster or cleaner than C++, or why their numbers differ.
---

# Compare Rust msbg-rs vs C++ MSBG

> **Mindset**: the two implementations are *not meant to be identical*. This is
> a refactor intended to be **superior** — or equivalent where superiority
> doesn't apply. This comparison exists to **verify that**, not to enforce
> parity. Layout/architecture differences are expected; focus on IPC, cache,
> and code size, plus the `difftest` hash for correctness.

Orchestration: a parent agent builds both sides, then spawns **two parallel
`general` subagents** (one per repo). Each subagent runs only read-only analysis
and returns a structured markdown summary. The parent merges them into one
table and interprets the result.

## Step 0 — build (parent does this once)

```bash
cd /home/jacob/programming/msbg-rs && nix-shell --run 'cargo bench --bench allocator_benches --no-run'
cd /home/jacob/programming/MSBG && nix-shell --run './build_bench.sh && ./build_difftest.sh'
```

## Step 1 — spawn two subagents in parallel

Launch two `general` subagents simultaneously. Tell each one: "run the commands
in your block inside `nix-shell`, return the raw numbers and the command output
verbatim (no interpretation) as markdown."

### Subagent A — Rust (`msbg-rs`)

```bash
# --- layout (test build instantiates Block<f32,16,4096>; force a recompile so it prints)
cargo clean -p msbg-rs
RUSTFLAGS="-Zprint-type-sizes" cargo test --no-run 2>&1 \
  | grep -E 'print-type-size type: `(msbg_rs::)?blockpool::(Block|BlockPool)<f32, 16, 4096>`'

# --- IPC / cache (filter to one scenario; see gotchas below)
BIN=$(find target/release -name 'allocator_benches-*' -type f -executable | head -1)
MSBG_BENCH_SCALE=small perf stat -e cycles,instructions,cache-misses,cache-references \
  "$BIN" blockpool_hot_path

# --- code size (crate symbols only; criterion/clap dominate the raw dump)
llvm-nm --size-sort --demangle "$BIN" | grep -i msbg_rs | tail -10
```

### Subagent B — C++ (`MSBG`)

```bash
# --- layout (rebuild one TU with -g; the makefile objects have no debug info)
g++ -g -O3 -fopenmp -std=gnu++17 -m64 -DMI_WITH_64BIT -DMIMP_ON_LINUX -march=native \
  -I. -Iexternal -c src/blockpool.cpp -o /tmp/blockpool_dbg.o
pahole -C BlockPool /tmp/blockpool_dbg.o

# --- IPC / cache (small workload)
MSBG_BENCH_SCALE=small perf stat -e cycles,instructions,cache-misses,cache-references \
  ./build/bench_executable

# --- code size (text symbols only; skip the BSS noise tables)
nm --size-sort -C ./build/bench_executable | grep -E ' [TtW] ' | tail -10
```

## Step 2 — aggregate (parent)

Merge into one table and compute:

| Metric | Rust | C++ | How to read |
|--------|------|-----|-------------|
| IPC | instructions/cycles | instructions/cycles | higher = better ILP |
| miss rate | cache-misses/cache-references | same | lower = better locality |
| `Block` size/align | record it | record it | not meant to match — see below |
| `BlockPool` size | record it | record it | not meant to match — see below |
| hot-function code size | `llvm-nm` `msbg_rs` symbols | `nm` text symbols | smaller = denser codegen |

**Layout is intentionally different** — the Rust port is a redesign, not a copy.
Record both layouts only to *document* the divergence; don't reconcile them. The
Rust side is the target: it should be superior, or equivalent where superiority
isn't applicable.

## Gotchas

- **Nested `nix-shell` inherits the parent's `NIX_CFLAGS_*`/`NIX_LDFLAGS`.** If a
  subagent runs inside a shell whose buildInputs inject flags (e.g. `glibc`'s
  `-isystem`, which breaks `#include_next <stdlib.h>`), prefix the inner command
  with `env -u NIX_CFLAGS_COMPILE -u NIX_CFLAGS_COMPILE_FOR_TARGET -u NIX_CXXSTDLIB_COMPILE -u NIX_LDFLAGS`.
- **`nm -C` does not demangle Rust symbols** (v0 mangling); use `llvm-nm --demangle`,
  `llvm-cxxfilt`, or `llvm-objdump --demangle`.
- **`nm --size-sort` mixes data/BSS with code** — filter text symbols
  (`grep -E ' [TtW] '`) or large data tables dominate the code-size view.
- `-Zprint-type-sizes` only prints during (re)compilation, and only for types
  that are actually instantiated — `cargo clean -p <pkg>` and build the
  tests/benches (not just `cargo build`) if you get nothing.
- `#[inline]` kernels have no standalone symbol; use `llvm-objdump --demangle`
  on the caller, or `#[inline(never)]`.
- The criterion binary adds harness overhead to `perf stat`; for the cleanest
  Rust IPC use a minimal driver, or accept the caveat.
- Always `MSBG_BENCH_SCALE=small` for the comparison; full runs OOM on small boxes.

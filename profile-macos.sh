#!/usr/bin/env bash
# Profile one criterion benchmark scenario on macOS (no `perf` here).
# Uses the built-in `sample` sampler to emit target/profile/profile.txt.
# NOTE: untested best-effort — macOS sampling is a separate toolchain.
set -euo pipefail
cd "$(dirname "$0")"

scenario="${1:-blockpool_hot_path}"

BIN=$(find target/release -name 'allocator_benches-*' -type f -executable | head -1)
if [ -z "$BIN" ]; then
  echo "bench binary not found; run: cargo bench --bench allocator_benches --no-run" >&2
  exit 1
fi

mkdir -p target/profile

# `sample` samples a running process; launch the bench in the background first.
MSBG_BENCH_SCALE=small "$BIN" "$scenario" >/dev/null 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

sample "$pid" 5 -mayDie > target/profile/profile.txt

echo "Wrote target/profile/profile.txt"
sed -n '1,40p' target/profile/profile.txt

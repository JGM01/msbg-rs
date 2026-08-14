#!/usr/bin/env bash
# Profile one criterion benchmark scenario on Linux and emit:
#   target/profile/profile.txt    (perf report top-N self-time, machine-readable)
#   target/profile/flamegraph.svg (for humans)
# Requires the msbg-rs nix-shell (shell.nix) on Linux.
set -euo pipefail
cd "$(dirname "$0")"

scenario="${1:-blockpool_hot_path}"

BIN=$(find target/release -name 'allocator_benches-*' -type f -executable | head -1)
if [ -z "$BIN" ]; then
  echo "bench binary not found; run: cargo bench --bench allocator_benches --no-run" >&2
  exit 1
fi

mkdir -p target/profile

# Rust omits frame pointers by default, so dwarf (not fp) unwinding is required.
MSBG_BENCH_SCALE=small perf record --call-graph dwarf -F 999 \
  -o target/profile/profile.data -- "$BIN" "$scenario"

# -g none    flat table, no call-tree noise
# --no-children  self time (flamegraph "width"), not inclusive
# --percent-limit 2  drop functions under 2%
perf report -i target/profile/profile.data --stdio -g none --no-children \
  --percent-limit 2 --show-total-period > target/profile/profile.txt

perf script -i target/profile/profile.data | inferno-collapse-perf | inferno-flamegraph > target/profile/flamegraph.svg

echo "Wrote target/profile/profile.txt and target/profile/flamegraph.svg"
echo "--- profile.txt ---"
cat target/profile/profile.txt

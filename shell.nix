{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # toolchain
    rustup   # nightly toolchain is pinned by rust-toolchain.toml
    gcc      # linker for rustc (and for building the C++ difftest baseline)
    # ELF-level debug/profiling (work on the Rust binary too)
    gdb
    strace
    perf
    pahole
    llvmPackages.llvm   # llvm-mca, llvm-objdump --demangle, llvm-cxxfilt
    valgrind
    rr
    heaptrack
    # Rust-specific
    cargo-nextest      # fast parallel test runner
    cargo-show-asm     # `cargo asm` — source-level disassembly
    cargo-bloat        # what's taking space in the binary
    cargo-llvm-lines   # monomorphization/compile-time cost
    cargo-flamegraph   # perf-based flame graphs
    cargo-geiger       # unsafe-code audit
  ]; # NOTE: do NOT add `glibc` here. It breaks the C++ include order when
     # building the ../MSBG difftest baseline (see ../MSBG/AGENTS.md).

  shellHook = ''
    # rustup shims (rustc/cargo) live in ~/.cargo/bin, not in the nix store.
    export PATH="$HOME/.cargo/bin:$PATH"

    # rr / perf need the kernel to allow perf events; raise the limit if strict.
    if [ "$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 0)" -gt 1 ]; then
      echo "note: rr/perf need 'sudo sysctl kernel.perf_event_paranoid=1'"
    fi

    echo "msbg-rs environment loaded (rust nightly via rust-toolchain.toml)"
  '';
}

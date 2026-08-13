{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "rust-msbg-bench-env";

  # Native dependencies required to compile and link Rust binaries on Linux
  buildInputs = with pkgs; [
    rustup
    gcc       # The GNU Compiler Collection (Rust needs this to link the binary)
    glibc     # Standard C library
  ];

  shellHook = ''
    echo "========================================================"
    echo "🦀 Rust Benchmark Environment Loaded for NixOS!"
    echo "========================================================"
    echo "1. If this is your first time in this shell, install nightly:"
    echo "   rustup default nightly"
    echo ""
    echo "2. Run your benchmarks:"
    echo "   cargo bench"
    echo "========================================================"
  '';
}

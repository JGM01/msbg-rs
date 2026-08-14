// Differential test against the C++ SBG::SparseGrid<float> baseline.
//
// Both sides run the same fixed write/read script and fold every voxel's f32
// bit pattern into an FNV-1a hash. `CPP_HASH` was produced by MSBG/difftest.cpp
// (build it in the MSBG repo; set MSBG_CPP_DIFFTEST_BIN to the binary to also
// run it here and compare live output).
use msbg_rs::{blockpool::BlockPool, sparse_grid::SparseGrid};
use std::sync::Arc;

const BSX: usize = 16;
const N: usize = 4096;
const CPP_HASH: u64 = 0xba116b1aee638c84;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn native_hash() -> u64 {
    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(8, 4096));
    let mut grid = SparseGrid::<f32, BSX, N>::new("diff".into(), 17, 33, 5, 0.0, 1.0, pool);

    // Writes (avoid blocks (1,1,0) and (0,1,0), marked full/empty below).
    grid.set_voxel(0, 0, 0, 42.0);
    grid.set_voxel(16, 0, 0, 7.0);
    grid.set_voxel(0, 32, 0, 3.0);
    grid.set_voxel(16, 32, 4, 99.0);
    grid.set_voxel(5, 5, 2, 123.5);
    grid.set_voxel(3, 3, 3, 0.25);

    grid.set_full_block(grid.get_block_id(16, 16, 0));
    grid.set_empty_block(grid.get_block_id(0, 16, 0));

    let mut h = 0xcbf29ce484222325u64;
    for z in 0..5usize {
        for y in 0..33usize {
            for x in 0..17usize {
                h = fnv1a(h, &grid.get_voxel(x, y, z).to_bits().to_le_bytes());
            }
        }
    }
    h
}

#[test]
fn difftest_matches_cpp_hash() {
    assert_eq!(
        native_hash(),
        CPP_HASH,
        "native hash diverged from the C++ baseline"
    );
}

#[test]
fn difftest_against_cpp_binary_if_available() {
    let Ok(bin) = std::env::var("MSBG_CPP_DIFFTEST_BIN") else {
        eprintln!("skipping: set MSBG_CPP_DIFFTEST_BIN to the built C++ difftest");
        return;
    };

    let out = std::process::Command::new(bin)
        .output()
        .expect("failed to run C++ difftest binary");
    let hash = String::from_utf8_lossy(&out.stdout).trim().to_string();

    assert_eq!(
        hash,
        format!("{:016x}", CPP_HASH),
        "C++ binary hash differs"
    );
}

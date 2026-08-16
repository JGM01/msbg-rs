//! Differential test for the step-6 stencil kernels against the C++ baseline
//! (`../MSBG/meancurvtest.cpp`). Both sides smooth the same polynomial field
//! once (Laplacian `laplTyp=1`, mean-curvature `laplTyp=4`, bi-Laplacian
//! `laplTyp=2`) and the interior block is compared within 1e-4 tolerance
//! (f32 + FMA -> not bit-exact).

use msbg_rs::{
    blockpool::{Block, BlockPool},
    math::bilaplacian::kernel_bilaplacian,
    math::laplacian::kernel_laplacian,
    math::meancurv::kernel_meancurv,
    math::simd::LANES,
    math::stencil::MaskBlock,
    math::BoundaryCondition,
    multires::halo::HaloBlock,
    sparse_grid::{BlockPtr, SparseGrid},
};
use std::sync::Arc;

const BSX: usize = 16;
const N: usize = 4096;
const CHUNKS: usize = N / LANES;
const DT: f32 = 0.01;
const TOL: f32 = 1e-4;

fn field(x: i32, y: i32, z: i32) -> f32 {
    let u = (x - 24) as f32;
    let v = (y - 24) as f32;
    let w = (z - 24) as f32;
    let q = 0.01 * (u * u + 2.0 * v * v + 3.0 * w * w + u * v);
    let r = 0.001 * (u * u * u * u + v * v * v * v + w * w * w * w);
    q + r
}

fn build_grid() -> SparseGrid<f32, BSX, N> {
    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(8, 64));
    let mut grid = SparseGrid::new("meancurvtest".into(), 64, 64, 64, 0.0, 1.0, pool);
    for bz in 0..grid.nz {
        for by in 0..grid.ny {
            for bx in 0..grid.nx {
                let bid = bx + by * grid.nx + bz * grid.nxy;
                let ptr = BlockPtr(grid.block_pool.alloc_block());
                unsafe {
                    for z in 0..BSX {
                        for y in 0..BSX {
                            for x in 0..BSX {
                                let gx = bx * BSX + x;
                                let gy = by * BSX + y;
                                let gz = bz * BSX + z;
                                (*ptr.as_ptr()).data[x + y * BSX + z * BSX * BSX] =
                                    field(gx as i32, gy as i32, gz as i32);
                            }
                        }
                    }
                }
                grid.blockmap[bid] = Some(ptr);
            }
        }
    }
    grid
}

/// Run one kernel application on the interior block (1,1,1) == bid 21.
fn run_kernel(grid: &SparseGrid<f32, BSX, N>, lapl_typ: u8) -> [f32; N] {
    let flags = Block::<u16, BSX, N>::new();
    let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);
    let mut out = Block::<f32, BSX, N>::new();
    let bid = grid.get_block_id(16, 16, 16); // block (1,1,1)
    let bc = BoundaryCondition::Neumann;

    match lapl_typ {
        1 => {
            let mut halo = HaloBlock::<BSX, 18>::new();
            halo.fill::<1, false, f32, N>(grid, bid, bc);
            kernel_laplacian::<LANES, CHUNKS, BSX, 18, N>(&halo, &mask, DT, &mut out);
        }
        4 => {
            let mut halo = HaloBlock::<BSX, 18>::new();
            halo.fill::<1, true, f32, N>(grid, bid, bc);
            kernel_meancurv::<LANES, CHUNKS, BSX, 18, N>(&halo, &mask, DT, &mut out);
        }
        2 => {
            let mut halo = HaloBlock::<BSX, 20>::new();
            halo.fill::<2, true, f32, N>(grid, bid, bc);
            kernel_bilaplacian::<LANES, CHUNKS, BSX, 20, N>(&halo, &mask, DT, &mut out);
        }
        _ => unreachable!("unknown laplTyp"),
    }
    out.data
}

const SAMPLE_IDX: [usize; 10] = [0, 1, 2, 3, 15, 16, 255, 256, 272, 4095];

// Golden sample values produced by ../MSBG/build/meancurvtest.
const GOLDEN: [[f32; 10]; 3] = [
    // laplTyp 1
    [16.8713436, 14.9309492, 13.6057587, 12.7378674, 13.8075304, 14.7801056, 12.8503265, 14.709692, 12.6189442, 10.6895962],
    // laplTyp 4
    [16.7841225, 14.8576059, 13.5410128, 12.6785774, 13.7376575, 14.7077026, 12.7913237, 14.6377525, 12.5615482, 10.6455212],
    // laplTyp 2
    [16.7649345, 14.840188, 13.5253305, 12.664402, 13.7202902, 14.6902142, 12.7754869, 14.6202278, 12.5455322, 10.6308451],
];

fn assert_close(got: &[f32; N], want: &[f32], tag: &str) {
    let mut maxd = 0.0f32;
    let mut maxi = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > maxd {
            maxd = d;
            maxi = i;
        }
    }
    assert!(
        maxd <= TOL,
        "{tag}: max diff {maxd} at idx {maxi} (got {} want {})",
        got[maxi],
        want[maxi]
    );
}

#[test]
fn stencil_matches_cpp_golden_samples() {
    let grid = build_grid();
    for (t, lapl_typ) in [0usize, 1, 2].iter().enumerate() {
        let typ = [1u8, 4, 2][*lapl_typ];
        let out = run_kernel(&grid, typ);
        for (k, &idx) in SAMPLE_IDX.iter().enumerate() {
            let d = (out[idx] - GOLDEN[t][k]).abs();
            assert!(
                d <= TOL,
                "laplTyp {typ} idx {idx}: got {} want {} (diff {d})",
                out[idx],
                GOLDEN[t][k]
            );
        }
    }
}

#[test]
fn stencil_against_cpp_binary_if_available() {
    let Ok(bin) = std::env::var("MSBG_CPP_MEANCURVTEST_BIN") else {
        eprintln!("skipping: set MSBG_CPP_MEANCURVTEST_BIN to the built C++ meancurvtest");
        return;
    };
    let out = std::process::Command::new(&bin)
        .output()
        .expect("failed to run C++ meancurvtest binary");
    assert!(
        out.status.success(),
        "C++ meancurvtest failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();

    let grid = build_grid();
    let mut i = 0;
    for lapl_typ in [1u8, 4, 2] {
        // Header line is the laplTyp itself.
        assert_eq!(lines[i].trim(), lapl_typ.to_string(), "expected header {lapl_typ}");
        i += 1;
        let mut want = Vec::with_capacity(N);
        for _ in 0..N {
            want.push(lines[i].trim().parse::<f32>().expect("bad float in C++ output"));
            i += 1;
        }
        let got = run_kernel(&grid, lapl_typ);
        assert_close(&got, &want, &format!("live laplTyp {lapl_typ}"));
    }
}

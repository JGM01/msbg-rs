//! Differential test for the step-8 8-color in-place smoother against the C++
//! baseline (`../MSBG/smoothertest.cpp`). Both sides run the REAL solver — C++
//! `applyChannelPdeFast` with `-(laplTyp + OPT_8_COLOR_SCHEME)`, Rust
//! `Sweeper` — for 4 iterations over the same smooth field, and the full level-0
//! field is compared within tolerance (f32 + FMA -> not bit-exact).

use msbg_rs::{
    blockpool::BlockPool,
    solver::{Fence, PdeParams, Stencil, Sweeper},
    sparse_grid::SparseGrid,
};
use std::sync::Arc;

const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;
const N_BLOCKS: usize = 64; // 4^3
const N_VOXELS: usize = N_BLOCKS * N;
const DT: f32 = 0.01;
const ITERS: usize = 4;

fn field(x: f32, y: f32, z: f32) -> f32 {
    (0.1 * x).sin() * (0.08 * y).cos() * (0.06 * z).sin()
}

fn build_grid() -> SparseGrid<f32, BSX, N> {
    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(2, 64));
    let mut grid = SparseGrid::new("smoothertest".into(), 64, 64, 64, 0.0, 1.0, pool);
    for bz in 0..grid.nz {
        for by in 0..grid.ny {
            for bx in 0..grid.nx {
                let bid = bx + by * grid.nx + bz * grid.nxy;
                grid.ensure_block(bid);
                let data = grid.get_block_data_mut(bid).unwrap();
                for z in 0..BSX {
                    for y in 0..BSX {
                        for x in 0..BSX {
                            data[x + y * BSX + z * BSX * BSX] =
                                field((bx * BSX + x) as f32, (by * BSX + y) as f32, (bz * BSX + z) as f32);
                        }
                    }
                }
            }
        }
    }
    grid
}

fn run_solver(grid: &SparseGrid<f32, BSX, N>, stencil: Stencil) -> Vec<f32> {
    let active: Vec<usize> = (0..N_BLOCKS).collect();
    let sweeper = Sweeper::<f32, BSX, N, HSX>::new(grid, rayon::current_num_threads(), Fence::Sfence);
    let params = PdeParams { dt: DT, iterations: ITERS, do_constr_zero_one: false };
    sweeper.sweep(&active, stencil, &params);

    let mut out = Vec::with_capacity(N_VOXELS);
    for bid in 0..N_BLOCKS {
        out.extend_from_slice(grid.get_block_data(bid).unwrap());
    }
    out
}

/// Golden sample values (indices into the 262144-voxel level-0 field), produced
/// by `../MSBG/build/smoothertest`.
const SAMPLE_IDX: [usize; 12] = [0, 1, 15, 16, 255, 256, 272, 4095, 100_000, 200_000, 262_143, 131_072];

const GOLDEN: [[f32; 12]; 2] = [
    // laplTyp 1 (Laplacian)
    [6.35691917e-15, 7.16737929e-07, 7.1486691e-05, 6.27597627e-15, 9.39214442e-06, 7.16738782e-07, 7.1216067e-07, 0.283120126, -0.00017801064, -4.40651129e-05, -0.00321645429, 0.000175741952],
    // laplTyp 4 (mean-curvature)
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.282928258, 0.0, 0.0, -0.00317776599, 0.0],
];

/// Mean-curvature is compared at a looser tolerance than Laplacian: its mixed
/// partials are factored `0.25*(a-b-c+d)` (fewer FLOPs than C++'s two-stage
/// `0.5` form, see `docs/refactor.md`), and near the `gradMagSq > 1e-7` guard
/// cliff that ~1-ulp difference in `H` flips a voxel between `H = hnum/grad`
/// and `H = 0`. Four Gauss-Seidel iterations propagate the resulting
/// discontinuity into the neighborhood.
fn tol(stencil: Stencil) -> f32 {
    match stencil {
        Stencil::Laplacian => 1e-4,
        Stencil::MeanCurvature | Stencil::BiLaplacian => 1e-3,
    }
}

#[test]
fn smoother_matches_cpp_golden_samples() {
    for (t, stencil) in [Stencil::Laplacian, Stencil::MeanCurvature].iter().enumerate() {
        let grid = build_grid();
        let out = run_solver(&grid, *stencil);
        for (k, &idx) in SAMPLE_IDX.iter().enumerate() {
            let got = out[idx];
            let want = GOLDEN[t][k];
            assert!(
                (got - want).abs() <= tol(*stencil),
                "stencil {stencil:?} idx {idx}: got {got} want {want}"
            );
        }
    }
}

#[test]
fn smoother_against_cpp_binary_if_available() {
    let Ok(bin) = std::env::var("MSBG_CPP_SMOOTHERTEST_BIN") else {
        eprintln!("skipping: set MSBG_CPP_SMOOTHERTEST_BIN to the built C++ smoothertest");
        return;
    };
    let out = std::process::Command::new(&bin)
        .output()
        .expect("failed to run C++ smoothertest binary");
    assert!(
        out.status.success(),
        "C++ smoothertest failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines();

    for stencil in [Stencil::Laplacian, Stencil::MeanCurvature] {
        let header = lines.next().expect("expected laplTyp header").trim().to_string();
        let want_lapl = match stencil {
            Stencil::Laplacian => "1",
            Stencil::MeanCurvature => "4",
            Stencil::BiLaplacian => unreachable!(),
        };
        assert_eq!(header, want_lapl, "expected header {want_lapl}");

        let mut want = Vec::with_capacity(N_VOXELS);
        for _ in 0..N_VOXELS {
            want.push(lines.next().expect("truncated C++ output").trim().parse::<f32>().expect("bad float"));
        }

        let grid = build_grid();
        let got = run_solver(&grid, stencil);

        let mut maxd = 0.0f32;
        let mut maxi = 0;
        for i in 0..N_VOXELS {
            let d = (got[i] - want[i]).abs();
            if d > maxd {
                maxd = d;
                maxi = i;
            }
        }
        assert!(
            maxd <= tol(stencil),
            "stencil {stencil:?}: max diff {maxd} at idx {maxi} (got {} want {})",
            got[maxi],
            want[maxi]
        );
    }
}

#![cfg(feature = "cpp-parity")]

use msbg_rs::multires::{MultiresGrid, RefinementMap};
use msbg_rs::pressure::{PressureBoundary, PressureOperator, PressurePatternBuilder, RelaxParams};
use std::process::Command;

const MATVEC_GOLDEN: [f32; 8] = [
    -0.0179998875,
    -0.0179998875,
    -0.0179991722,
    -0.0180001259,
    -0.0180001259,
    -0.0179991722,
    -0.0180015564,
    -0.0180015564,
];

const RELAX_GOLDEN: [f32; 8] = [
    0.521749735,
    0.930204153,
    2.0442698,
    1.51345265,
    1.541785,
    2.84341812,
    5.52284098,
    7.48720264,
];

const SAMPLES: [[usize; 3]; 8] = [
    [2, 3, 4],
    [7, 11, 5],
    [14, 9, 13],
    [15, 8, 8],
    [16, 8, 8],
    [20, 17, 12],
    [25, 24, 21],
    [29, 28, 27],
];

fn close(actual: f32, expected: f32) {
    let tolerance = 1e-5 * (1.0 + expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs {expected}"
    );
}

fn rust_results() -> ([f32; 8], [f32; 8]) {
    let mut grid = MultiresGrid::create("pressuretest", 32, 32, 32, 16, 3, 6);
    let mut refinement = RefinementMap::uniform(grid.dims.n_blocks, 0);
    grid.set_refinement_map(&mut refinement);
    let active: Vec<usize> = (0..grid.dims.n_blocks).collect();
    let pattern =
        PressurePatternBuilder::new(&grid, &active, [PressureBoundary::OpenDirichletZero; 6])
            .build()
            .unwrap();
    let operator = PressureOperator::constant(pattern.clone(), 1.0).unwrap();
    let mut pressure = pattern.vector();
    let mut rhs = pattern.vector();
    for block in pattern.blocks() {
        let [bx, by, bz] = block.coords;
        for z in 0..block.block_size {
            for y in 0..block.block_size {
                for x in 0..block.block_size {
                    let gx = bx * 16 + x;
                    let gy = by * 16 + y;
                    let gz = bz * 16 + z;
                    let cell = pattern.cell_index(block.bid, x, y, z).unwrap();
                    pressure.as_mut_slice()[cell] = 0.002 * (gx * gx) as f32
                        + 0.003 * (gy * gy) as f32
                        + 0.004 * (gz * gz) as f32
                        + 0.01 * gx as f32
                        - 0.02 * gy as f32
                        + 0.03 * gz as f32
                        + 0.25;
                    rhs.as_mut_slice()[cell] =
                        0.1 + 0.001 * gx as f32 - 0.0005 * gy as f32 + 0.00025 * gz as f32;
                }
            }
        }
    }
    let mut applied = pattern.vector();
    operator.apply(&pressure, &mut applied).unwrap();
    let matvec = SAMPLES.map(|[x, y, z]| {
        let bid = x / 16 + (y / 16) * 2 + (z / 16) * 4;
        applied.as_slice()[pattern.cell_index(bid, x % 16, y % 16, z % 16).unwrap()]
    });
    operator
        .relax_cpp_parity(&mut pressure, &rhs, RelaxParams::new(6.0 / 7.0, 5).unwrap())
        .unwrap();
    let relaxed = SAMPLES.map(|[x, y, z]| {
        let bid = x / 16 + (y / 16) * 2 + (z / 16) * 4;
        pressure.as_slice()[pattern.cell_index(bid, x % 16, y % 16, z % 16).unwrap()]
    });
    (matvec, relaxed)
}

#[test]
fn pressure_matches_cpp_goldens() {
    let (matvec, relaxed) = rust_results();
    for (actual, expected) in matvec.into_iter().zip(MATVEC_GOLDEN) {
        close(actual, expected);
    }
    for (actual, expected) in relaxed.into_iter().zip(RELAX_GOLDEN) {
        close(actual, expected);
    }
}

#[test]
fn pressure_live_cpp_if_available() {
    let Ok(binary) = std::env::var("MSBG_CPP_PRESSURETEST_BIN") else {
        return;
    };
    let output = Command::new(binary).output().expect("run C++ pressuretest");
    assert!(output.status.success());
    let values: Vec<f32> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| line.parse().ok())
        .collect();
    assert_eq!(values.len(), 16);
    for (actual, expected) in values[..8].iter().copied().zip(MATVEC_GOLDEN) {
        close(actual, expected);
    }
    for (actual, expected) in values[8..].iter().copied().zip(RELAX_GOLDEN) {
        close(actual, expected);
    }
}

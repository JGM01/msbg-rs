//! Differential test against the C++ interpolation baseline
//! (`../MSBG/interptest.cpp`): golden values are compared within 1e-4
//! tolerance (the Rust port uses f32 + FMA, so it is not bit-exact).

use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sample};
use msbg_rs::sparse_grid::SparseGrid;
use msbg_rs::blockpool::BlockPool;
use msbg_rs::channel::Vec3;
use std::sync::Arc;

const BSX: usize = 16;
const N: usize = 4096;
const TOL: f32 = 1e-4;

fn field(x: f32, y: f32, z: f32) -> f32 {
    0.002 * x * x + 0.003 * y * y + 0.004 * z * z
        + 0.005 * x * y + 0.006 * x * z + 0.007 * y * z
        + 0.1 * x + 0.05 * y - 0.02 * z + 0.75
}

const POSITIONS: [[f32; 3]; 8] = [
    [5.3, 7.7, 3.2],
    [10.5, 12.25, 20.75],
    [15.1, 30.9, 10.4],
    [20.5, 21.25, 40.75],
    [25.25, 8.1, 33.3],
    [30.9, 15.4, 25.6],
    [35.5, 40.2, 12.8],
    [40.1, 25.5, 30.2],
];

// Per position: linear(value, gx, gy, gz), cubic(value, gx, gy, gz),
// hessian(fxx, fyy, fzz, fxy, fxz, fyz).
const GOLDEN: [[f32; 14]; 8] = [
    [2.35598993, 0.179700077, 0.143900022, 0.0936999992, 2.15639973, 0.171400011, 0.136099949, 0.0807999671, 0.0040001343, 0.00600012857, 0.00799955893, 0.00499999337, 0.00600012764, 0.00700006587],
    [8.12193775, 0.327749848, 0.322750449, 0.292750239, 7.65800142, 0.320249885, 0.312250018, 0.284250081, 0.00400029123, 0.00599963125, 0.00800015964, 0.00499996264, 0.00600003824, 0.00700035412],
    [12.8762102, 0.378900319, 0.381300569, 0.370900065, 12.3190031, 0.369800001, 0.37469995, 0.359598905, 0.0040007215, 0.00600041449, 0.00799880363, 0.00500013866, 0.00599988457, 0.00700033642],
    [25.1386871, 0.532750368, 0.56675148, 0.575748444, 24.308754, 0.525250375, 0.556249738, 0.567250311, 0.00399957877, 0.00600017421, 0.00799631327, 0.00499954633, 0.00600037538, 0.00700026378],
    [16.878685, 0.442300022, 0.460350513, 0.456200123, 16.2100258, 0.433800608, 0.44895038, 0.44410193, 0.00399971008, 0.00599987805, 0.00799714774, 0.00500036776, 0.00599958748, 0.00699947774],
    [19.2276192, 0.452600092, 0.476699442, 0.477198601, 18.5313587, 0.446700543, 0.467097908, 0.467500091, 0.00399891613, 0.00599860167, 0.00799880549, 0.00500041153, 0.00599968107, 0.0070006391],
    [27.5434208, 0.519800246, 0.560099423, 0.57439822, 26.7240982, 0.512300193, 0.549299717, 0.566301048, 0.00399957737, 0.00599710736, 0.00799883157, 0.00500020292, 0.00600039586, 0.00699900463],
    [32.0170708, 0.570700824, 0.614900231, 0.643100798, 31.1129017, 0.561600566, 0.605899572, 0.630201697, 0.00400260091, 0.00600098819, 0.00800082088, 0.00500041246, 0.00600062311, 0.00699989498],
];

fn build_grid() -> SparseGrid<f32, BSX, N> {
    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(16, 64));
    let mut g = SparseGrid::new("interp".into(), 48, 48, 48, 0.0f32, 1.0, pool);
    for z in 0..48 {
        for y in 0..48 {
            for x in 0..48 {
                g.set_voxel(x, y, z, field(x as f32, y as f32, z as f32));
            }
        }
    }
    g
}

fn sample_all(g: &SparseGrid<f32, BSX, N>, pos: Vec3) -> [f32; 14] {
    let lv = g.sample::<{ Interpolation::Linear }>(pos, GridAlignment::Corner, BoundaryCondition::Clamp);
    let lg = g.gradient::<{ Interpolation::Linear }>(pos, GridAlignment::Corner, BoundaryCondition::Clamp);
    let cv = g.sample::<{ Interpolation::CubicBSpline }>(pos, GridAlignment::Corner, BoundaryCondition::Clamp);
    let cg = g.gradient::<{ Interpolation::CubicBSpline }>(pos, GridAlignment::Corner, BoundaryCondition::Clamp);
    let h = g.hessian(pos, GridAlignment::Corner, BoundaryCondition::Clamp);
    [lv, lg.x(), lg.y(), lg.z(), cv, cg.x(), cg.y(), cg.z(), h.fxx, h.fyy, h.fzz, h.fxy, h.fxz, h.fyz]
}

fn assert_close(got: &[f32; 14], want: &[f32; 14], tag: &str) {
    for i in 0..14 {
        let d = (got[i] - want[i]).abs();
        assert!(d <= TOL, "{tag}[{i}]: got {} want {} (diff {d})", got[i], want[i]);
    }
}

#[test]
fn interp_matches_cpp_golden() {
    let g = build_grid();
    for (i, p) in POSITIONS.iter().enumerate() {
        let pos = Vec3::new(p[0], p[1], p[2]);
        assert_close(&sample_all(&g, pos), &GOLDEN[i], &format!("pos {i}"));
    }
}

#[test]
fn interp_against_cpp_binary_if_available() {
    let Ok(bin) = std::env::var("MSBG_CPP_INTERTEST_BIN") else {
        eprintln!("skipping: MSBG_CPP_INTERTEST_BIN not set");
        return;
    };
    let out = std::process::Command::new(&bin)
        .output()
        .expect("failed to run C++ interptest binary");
    assert!(out.status.success(), "C++ interptest failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);

    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let g = build_grid();
    for (i, p) in POSITIONS.iter().enumerate() {
        let line = lines.next().unwrap_or_else(|| panic!("C++ output ended early at pos {i}"));
        let nums: Vec<f32> = line
            .split_whitespace()
            .map(|t| t.parse::<f32>().expect("bad float in C++ output"))
            .collect();
        assert_eq!(nums.len(), 14, "pos {i}: expected 14 values");
        let want: [f32; 14] = nums.try_into().unwrap();
        let pos = Vec3::new(p[0], p[1], p[2]);
        assert_close(&sample_all(&g, pos), &want, &format!("live pos {i}"));
    }
}

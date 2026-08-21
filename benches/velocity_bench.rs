//! Step-13 velocity-transfer benchmark: the REAL P2G splat (staged 8-color
//! sum-splat to MAC faces) and the two G2P gathers (staggered `MacSampler` and
//! cell-centered `SampleVec3`). The cell-centered gather runs the same field and
//! positions as `../MSBG/velocitytest.cpp` (`interpolateVec3Float<IP_LINEAR>`),
//! so the two throughputs are directly comparable — real library calls, no mocks.
//!
//! The P2G splat has no C++ counterpart (the demo splats density only), so its
//! reference points are the step-9 density-splat numbers and the paper's
//! published ~62.6 Mparts/s MSBG P2G (Table 2).
//!
//! Knobs (defaults sized for the 5500U; raise for a bigger box):
//!   MSBG_VEL_RES=128      grid extent (voxels, one axis)
//!   MSBG_VEL_NSAMPLES=1000000
//!   MSBG_VEL_NPARTICLES=500000
//!   MSBG_VEL_ITERS=5

use msbg_rs::{
    blockpool::BlockPool,
    channel::{Vec3, Velocity},
    fluid::{mac::MacGrid, p2g::particles_to_grid, MacSampler, Particles, KIND_LIQUID},
    math::{BoundaryCondition, Interpolation, SampleVec3},
    sparse_grid::SparseGrid,
};
use std::{env, hint::black_box, sync::Arc, time::Instant};

const BSX: usize = 16;
const N: usize = 4096;
const MSX: usize = 24;
const R_P: f32 = 2.0;
const CLAMP: BoundaryCondition = BoundaryCondition::Clamp;

fn env_u(name: &str, dflt: usize) -> usize {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(dflt)
}

fn field(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(
        0.002 * x + 0.003 * y + 0.004 * z + 0.1,
        0.001 * x + 0.005 * y + 0.002 * z - 0.2,
        0.004 * x + 0.001 * y + 0.006 * z + 0.3,
    )
}

/// Deterministic interior positions (LCG), identical to `../MSBG/velocitytest.cpp`.
fn positions(n: usize, res: usize) -> Vec<Vec3> {
    let mut seed: u32 = 12_345;
    let mut rng = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed as f32 / u32::MAX as f32
    };
    (0..n)
        .map(|_| {
            Vec3::new(
                2.0 + rng() * (res - 4) as f32,
                2.0 + rng() * (res - 4) as f32,
                2.0 + rng() * (res - 4) as f32,
            )
        })
        .collect()
}

fn build_vec3_grid(res: usize) -> SparseGrid<Velocity, BSX, N> {
    let nblk = (res / BSX) * (res / BSX) * (res / BSX);
    let pool = Arc::new(BlockPool::<Velocity, BSX, N>::new(nblk / 4096 + 2, 4096));
    let mut g = SparseGrid::new("vel".into(), res, res, res, Velocity(Vec3::default()), Velocity(Vec3::default()), pool);
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                g.set_voxel(x, y, z, Velocity(field(x as f32, y as f32, z as f32)));
            }
        }
    }
    g
}

fn build_mac_grid(res: usize) -> MacGrid<BSX, N> {
    let nblk = (res / BSX) * (res / BSX) * (res / BSX);
    let pool = Arc::new(BlockPool::<f32, BSX, N>::new((6 * nblk).div_ceil(4096) + 2, 4096));
    let mut mac = MacGrid::new(res, res, res, pool);
    let active: Vec<usize> = (0..nblk).collect();
    mac.zero_blocks(&active);
    // Face-a field: u(i)=i+1 (face world-x), matching the step-13 convention.
    for a in 0..3 {
        for k in 0..res {
            for j in 0..res {
                for i in 0..res {
                    let v = match a {
                        0 => (i + 1) as f32,
                        1 => (j + 1) as f32,
                        _ => (k + 1) as f32,
                    };
                    mac.velocity_mut(a).set_voxel(i, j, k, v);
                }
            }
        }
    }
    mac
}

fn particle_cloud(n: usize, res: usize) -> Particles {
    let pos = positions(n, res);
    let mut seed: u32 = 99;
    let mut rng = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed as f32 / u32::MAX as f32
    };
    let velocities: Vec<[f32; 3]> = pos
        .iter()
        .map(|_| [rng() * 2.0 - 1.0, rng() * 2.0 - 1.0, rng() * 2.0 - 1.0])
        .collect();
    Particles {
        positions: pos.iter().map(|p| [p.x(), p.y(), p.z()]).collect(),
        velocities,
        kinds: vec![KIND_LIQUID; n],
        mass: vec![1.0; n],
    }
}

fn main() {
    let res = env_u("MSBG_VEL_RES", 128);
    let nsamples = env_u("MSBG_VEL_NSAMPLES", 1_000_000);
    let nparticles = env_u("MSBG_VEL_NPARTICLES", 500_000);
    let iters = env_u("MSBG_VEL_ITERS", 5);
    println!("============================================================");
    println!(" msbg-rs Velocity Transfer Benchmark (res={res})");
    println!("============================================================\n");

    // ---- Cell-centered Vec3 gather (vs C++ interpolateVec3Float) ---------
    {
        let g = build_vec3_grid(res);
        let p = positions(nsamples, res);
        let mut best = f64::MAX;
        let mut acc = 0.0f32;
        for _ in 0..iters {
            let t = Instant::now();
            acc = 0.0;
            for pos in &p {
                let v = g.sample_vec3::<{ Interpolation::Linear }>(*pos, msbg_rs::math::GridAlignment::CellCentered, CLAMP);
                acc += v.x() + v.y() + v.z();
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        black_box(acc);
        let ms = best * 1000.0;
        let mpairs = nsamples as f64 / best / 1e6;
        println!("[Rust] velocity_gather_vec3: {nsamples} samples in {ms:.3} ms ({mpairs:.2} Msamples/s, acc={acc:.6})");
    }

    // ---- Staggered MAC gather (step-13 G2P operator) ---------------------
    {
        let mac = build_mac_grid(res);
        let p = positions(nsamples, res);
        let sampler = MacSampler::new(&mac, CLAMP);
        let mut best = f64::MAX;
        let mut acc = 0.0f32;
        for _ in 0..iters {
            let t = Instant::now();
            acc = 0.0;
            for pos in &p {
                let v = sampler.sample::<{ Interpolation::Linear }>(*pos);
                acc += v.x() + v.y() + v.z();
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        black_box(acc);
        let ms = best * 1000.0;
        let mpairs = nsamples as f64 / best / 1e6;
        println!("[Rust] velocity_gather_mac:   {nsamples} samples in {ms:.3} ms ({mpairs:.2} Msamples/s, acc={acc:.6})");
    }

    // ---- P2G splat (mass + momentum to faces) ----------------------------
    {
        let nblk = (res / BSX) * (res / BSX) * (res / BSX);
        let pool = Arc::new(BlockPool::<f32, BSX, N>::new((6 * nblk).div_ceil(4096) + 2, 4096));
        let cloud = particle_cloud(nparticles, res);
        let mut mac = MacGrid::new(res, res, res, pool);
        let mut best = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            particles_to_grid::<BSX, N, MSX>(&mut mac, &cloud, R_P);
            best = best.min(t.elapsed().as_secs_f64());
        }
        black_box(&mac);
        let ms = best * 1000.0;
        let mparts = nparticles as f64 / best / 1e6;
        println!("[Rust] velocity_splat:        {nparticles} particles in {ms:.3} ms ({mparts:.2} Mparts/s)");
    }

    println!("\nDone.");
}

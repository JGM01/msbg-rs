use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use msbg_rs::blockpool::BlockPool;
use msbg_rs::channel::Vec3;
use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sample};
use msbg_rs::sparse_grid::SparseGrid;
use std::hint::black_box;
use std::sync::Arc;

const BSX: usize = 16;
const N: usize = 4096;
const CORNER: GridAlignment = GridAlignment::Corner;
const CLAMP: BoundaryCondition = BoundaryCondition::Clamp;
const NSAMPLES: u64 = 100_000;

fn field(x: f32, y: f32, z: f32) -> f32 {
    0.002 * x * x + 0.003 * y * y + 0.004 * z * z
        + 0.005 * x * y + 0.006 * x * z + 0.007 * y * z
        + 0.1 * x + 0.05 * y - 0.02 * z + 0.75
}

fn build_grid() -> SparseGrid<f32, BSX, N> {
    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(64, 4096));
    let mut g = SparseGrid::new("bench".into(), 96, 96, 96, 0.0f32, 1.0, pool);
    for z in 0..96 {
        for y in 0..96 {
            for x in 0..96 {
                g.set_voxel(x, y, z, field(x as f32, y as f32, z as f32));
            }
        }
    }
    g
}

/// Deterministic pseudo-random interior positions (LCG), identical on both sides.
fn positions() -> Vec<Vec3> {
    let mut seed: u32 = 12_345;
    let mut rng = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed as f32 / u32::MAX as f32
    };
    (0..NSAMPLES)
        .map(|_| {
            Vec3::new(2.0 + rng() * 92.0, 2.0 + rng() * 92.0, 2.0 + rng() * 92.0)
        })
        .collect()
}

fn bench_linear_value(c: &mut Criterion) {
    let g = build_grid();
    let p = positions();
    let mut group = c.benchmark_group("interp_linear_value");
    group.throughput(Throughput::Elements(NSAMPLES));
    group.bench_function("sample", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for pos in &p {
                acc += g.sample::<{ Interpolation::Linear }>(*pos, CORNER, CLAMP);
            }
            black_box(acc);
        })
    });
    group.finish();
}

fn bench_linear_grad(c: &mut Criterion) {
    let g = build_grid();
    let p = positions();
    let mut group = c.benchmark_group("interp_linear_grad");
    group.throughput(Throughput::Elements(NSAMPLES));
    group.bench_function("gradient", |b| {
        b.iter(|| {
            let mut acc = Vec3::default();
            for pos in &p {
                acc = acc + g.gradient::<{ Interpolation::Linear }>(*pos, CORNER, CLAMP);
            }
            black_box(acc);
        })
    });
    group.finish();
}

fn bench_cubic_grad(c: &mut Criterion) {
    let g = build_grid();
    let p = positions();
    let mut group = c.benchmark_group("interp_cubic_grad");
    group.throughput(Throughput::Elements(NSAMPLES));
    group.bench_function("gradient", |b| {
        b.iter(|| {
            let mut acc = Vec3::default();
            for pos in &p {
                acc = acc + g.gradient::<{ Interpolation::CubicBSpline }>(*pos, CORNER, CLAMP);
            }
            black_box(acc);
        })
    });
    group.finish();
}

fn bench_cubic_hess(c: &mut Criterion) {
    let g = build_grid();
    let p = positions();
    let mut group = c.benchmark_group("interp_cubic_hess");
    group.throughput(Throughput::Elements(NSAMPLES));
    group.bench_function("hessian", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for pos in &p {
                let h = g.hessian(*pos, CORNER, CLAMP);
                acc += h.fxx + h.fyy + h.fzz + h.fxy + h.fxz + h.fyz;
            }
            black_box(acc);
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_value,
    bench_linear_grad,
    bench_cubic_grad,
    bench_cubic_hess
);
criterion_main!(benches);

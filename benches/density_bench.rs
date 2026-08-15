use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use msbg_rs::channel::{Density, dequantize_density, quantize_density};

// One 4096-voxel block (the real per-block pipeline pattern) and a large
// contiguous buffer (pure memory-bandwidth bound).
const BLOCK: usize = 4096;
const LARGE: usize = 4 * 1024 * 1024;

fn bench_dequant(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_dequant");
    for &n in &[BLOCK, LARGE] {
        let src: Vec<Density> = (0..n).map(|i| Density((i * 26_543 + 1) as u16)).collect();
        let mut dst = vec![0.0f32; n];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| dequantize_density(&src, &mut dst));
        });
    }
    group.finish();
}

fn bench_quant(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_quantize");
    for &n in &[BLOCK, LARGE] {
        let src: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 / 999.0).collect();
        let mut dst = vec![Density(0); n];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| quantize_density(&src, &mut dst));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dequant, bench_quant);
criterion_main!(benches);

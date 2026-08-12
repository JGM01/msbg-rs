use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use msbg_rs::blockpool::BlockPool;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{hint::black_box, sync::Arc};

// 1 Block = ~16.06 KB
const BSX: usize = 16;
const N: usize = 4096;

fn bench_hot_path_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("blockpool_hot_path");

    // Benchmark 10k, 100k, and 250k blocks (Up to ~4 GB RAM allocation)
    for block_count in [10_000, 100_000, 250_000] {
        let blocks_per_seg = 4096; // 64 MB segments (MSBG C++ standard)
        let max_segments = (block_count / blocks_per_seg) + 2;

        let pool = BlockPool::<f32, BSX, N>::new(max_segments, blocks_per_seg);

        group.bench_with_input(
            BenchmarkId::new("single_thread", block_count),
            &block_count,
            |b, &count| {
                b.iter(|| {
                    for _ in 0..count {
                        let ptr = pool.alloc_block();
                        black_box(ptr);
                    }
                    pool.reset();
                });
            },
        );
    }
    group.finish();
}

fn bench_multithreaded_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("blockpool_contention");

    // Force small segments to stress segment creation lock races across all cores
    let num_threads = rayon::current_num_threads();
    let blocks_per_thread = 4096;
    let total_blocks = num_threads * blocks_per_thread;

    let blocks_per_seg = 256; // 1 MB segments to trigger frequent extend_pool calls
    let max_segments = (total_blocks / blocks_per_seg) + 16;

    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(max_segments, blocks_per_seg));

    group.bench_function(
        format!("{}_threads_{}_blocks", num_threads, total_blocks),
        |b| {
            b.iter(|| {
                (0..num_threads).into_par_iter().for_each(|_| {
                    let pool_clone = Arc::clone(&pool);
                    for _ in 0..blocks_per_thread {
                        let ptr = pool_clone.alloc_block();
                        black_box(ptr);
                    }
                });
                pool.reset();
            });
        },
    );

    group.finish();
}

fn bench_cold_extension(c: &mut Criterion) {
    let mut group = c.benchmark_group("blockpool_cold_alloc");

    // Benchmark the cost of creating and expanding a pool from scratch (including OS page allocations)
    group.bench_function("cold_alloc_10k_blocks", |b| {
        b.iter(|| {
            let pool = BlockPool::<f32, BSX, N>::new(16, 1024);
            for _ in 0..10_000 {
                let ptr = pool.alloc_block();
                black_box(ptr);
            }
            black_box(pool); // Drops and frees memory
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hot_path_scaling,
    bench_multithreaded_contention,
    bench_cold_extension
);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{hint::black_box, sync::Arc};

// Adjust imports based on your crate name
use msbg_rs::blockpool::BlockPool;

fn bench_single_thread_alloc(c: &mut Criterion) {
    c.bench_function("blockpool_alloc_single_thread_10k", |b| {
        // Create a pool of size 10_000 blocks
        let pool: BlockPool<f32, 16, 4096> = BlockPool::new(16, 1024);

        b.iter(|| {
            // Allocate 1_000 blocks monotonically
            for _ in 0..1024 {
                let ptr = pool.alloc_block();
                black_box(ptr);
            }
            // Reset for next iteration
            pool.reset();
        });
    });
}

fn bench_multithreaded_alloc(c: &mut Criterion) {
    c.bench_function("blockpool_alloc_8_threads_rayon_10k", |b| {
        // Pool capacity for ~8_000 allocations per iteration
        let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(64, 1024));

        // Setup thread pool with 8 threads
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        thread_pool.install(|| {
            b.iter(|| {
                // Execute 8 parallel tasks
                (0..8).into_par_iter().for_each(|_| {
                    let pool_clone = Arc::clone(&pool);
                    for _ in 0..1024 {
                        let ptr = pool_clone.alloc_block();
                        black_box(ptr);
                    }
                });

                // Reset pool for next iteration
                pool.reset();
            });
        });
    });
}

criterion_group!(
    benches,
    bench_single_thread_alloc,
    bench_multithreaded_alloc
);
criterion_main!(benches);

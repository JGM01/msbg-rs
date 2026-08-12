use criterion::{Criterion, criterion_group, criterion_main};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{hint::black_box, sync::Arc};

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

fn bench_multithreaded_alloc_static(c: &mut Criterion) {
    c.bench_function("blockpool_alloc_8_threads_static_10k", |b| {
        let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(64, 1024));

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        thread_pool.install(|| {
            b.iter(|| {
                // `with_min_len` and `with_max_len` forces static chunking across the threads.
                (0..8)
                    .into_par_iter()
                    .with_min_len(1)
                    .with_max_len(1)
                    .for_each(|_| {
                        let pool_clone = Arc::clone(&pool);
                        for _ in 0..1024 {
                            let ptr = pool_clone.alloc_block();
                            black_box(ptr);
                        }
                    });

                pool.reset();
            });
        });
    });
}

fn bench_multithreaded_alloc_scope(c: &mut Criterion) {
    c.bench_function("blockpool_alloc_8_threads_scope_10k", |b| {
        let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(64, 1024));

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        thread_pool.install(|| {
            b.iter(|| {
                // "scope" spawns 8 jobs onto the thread pool,
                // matching OpenMP static scheduling (mostly).
                rayon::scope(|s| {
                    for _ in 0..8 {
                        let pool_clone = Arc::clone(&pool);
                        s.spawn(move |_| {
                            for _ in 0..1024 {
                                let ptr = pool_clone.alloc_block();
                                black_box(ptr);
                            }
                        });
                    }
                });

                pool.reset();
            });
        });
    });
}

criterion_group!(
    benches,
    bench_single_thread_alloc,
    bench_multithreaded_alloc_static,
    bench_multithreaded_alloc_scope
);
criterion_main!(benches);

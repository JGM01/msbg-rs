use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use msbg_rs::{
    blockpool::{Block, BlockPool},
    math::laplacian::kernel_laplacian_simd_16,
    multires::halo::HaloBlockPool,
    sparse_grid::{BlockPtr, SparseGrid},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{hint::black_box, sync::Arc};

// 1 Block = ~16.06 KB
const BSX: usize = 16;
const N: usize = 4096;

/// Thin `Send` wrapper around a raw mutable pointer.
/// Thin `Send` wrapper around a raw mutable pointer.
struct SendPtr<T>(*mut T);

impl<T> Clone for SendPtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Same as `SendPtr` but for a read-only raw pointer.
struct SendConstPtr<T>(*const T);

impl<T> Clone for SendConstPtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendConstPtr<T> {}

unsafe impl<T> Send for SendConstPtr<T> {}
unsafe impl<T> Sync for SendConstPtr<T> {}

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
                        unsafe {
                            (*ptr.as_ptr()).flags = 0; // Force a RAM touch
                        }
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
                        unsafe {
                            (*ptr.as_ptr()).flags = 0; // Force a RAM touch
                        }
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
                unsafe {
                    (*ptr.as_ptr()).flags = 0; // Force a RAM touch
                }
                black_box(ptr);
            }
            black_box(pool); // Drops and frees memory
        });
    });

    group.finish();
}

fn bench_laplacian_halo_mocked(c: &mut Criterion) {
    let mut group = c.benchmark_group("laplacian_compute_throughput_mocked_fill");
    group.sample_size(20);
    for block_count in [1_000, 10_000, 50_000] {
        group.bench_with_input(
            BenchmarkId::new("rayon_sweep", block_count),
            &block_count,
            |b, &count| {
                let num_threads = rayon::current_num_threads();
                let halo_pool = HaloBlockPool::<f32, BSX, 18, 5832>::new(num_threads);

                let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); count];
                let flags_blocks: Vec<Block<u16, BSX, N>> = vec![Block::new(); count];

                // Static, persistent partition computed once — mirrors OpenMP's
                // `schedule(static)`: thread `tid` always owns the same range
                // on every invocation below, instead of Rayon's work-stealing
                // splitter deciding (potentially differently) each call.
                let chunk_size = count.div_ceil(num_threads);

                let out_ptr = SendPtr(output_blocks.as_mut_ptr());
                let flags_ptr = SendConstPtr(flags_blocks.as_ptr());

                b.iter(|| {
                    rayon::broadcast(|ctx| {
                        let out_ptr = out_ptr;
                        let flags_ptr = flags_ptr;

                        let tid = ctx.index();
                        let start = tid * chunk_size;
                        let end = (start + chunk_size).min(count);
                        if start >= end {
                            return;
                        }

                        // Direct slot lookup by broadcast index — no redundant
                        // current_thread_index() TLS read needed.
                        let halo = unsafe { halo_pool.get_mut_by_index(tid) };

                        for i in start..end {
                            // Mocking the memory bandwidth of gathering the neighborhood via memset
                            halo.mock_fill_for_bench();

                            unsafe {
                                let out_block = &mut *out_ptr.0.add(i);
                                let flags_block = &*flags_ptr.0.add(i);
                                kernel_laplacian_simd_16(halo, flags_block, out_block);
                            }
                        }
                    });
                    black_box(&mut output_blocks);
                });
            },
        );
    }
    group.finish();
}

fn bench_laplacian_halo_real_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("laplacian_compute_throughput_real_fill");
    group.sample_size(20);
    for block_count in [10_000, 50_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("rayon_sweep", block_count),
            &block_count,
            |b, &count| {
                let num_threads = rayon::current_num_threads();
                // HSX = 18, so N_HALO = 18^3 = 5832
                let halo_pool = HaloBlockPool::<f32, BSX, 18, 5832>::new(num_threads);

                let blocks_per_dim = (count as f64).cbrt().ceil() as usize;
                let sx = blocks_per_dim * BSX;
                let sy = blocks_per_dim * BSX;
                let sz = blocks_per_dim * BSX;

                let pool = Arc::new(BlockPool::<f32, BSX, N>::new(1024, 4096));
                let mut grid =
                    SparseGrid::new("bench_grid".to_string(), sx, sy, sz, 0.0, 1.0, pool);

                for bid in 0..count {
                    if bid < grid.n_blocks {
                        let new_block = BlockPtr(grid.block_pool.alloc_block());
                        grid.blockmap[bid] = Some(new_block);
                    }
                }

                let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); count];
                let flags_blocks: Vec<Block<u16, BSX, N>> = vec![Block::new(); count];

                // Static, persistent partition computed once outside the timed
                // closure — thread `tid` owns range [tid*chunk, (tid+1)*chunk)
                // on *every* call below, matching OpenMP's `schedule(static)`
                // + `omp_get_thread_num()` exactly, instead of leaving the
                // range-to-worker assignment to Rayon's work-stealing splitter
                // (which is free to reshuffle it between iterations since this
                // closure is invoked from Criterion's thread, not a Rayon
                // worker, forcing a fresh injection into the global queue
                // every call).
                let chunk_size = count.div_ceil(num_threads);

                let grid_ref = &grid; // shared ref: fill() only reads the grid
                let out_ptr = SendPtr(output_blocks.as_mut_ptr());
                let flags_ptr = SendConstPtr(flags_blocks.as_ptr());

                b.iter(|| {
                    rayon::broadcast(|ctx| {
                        let out_ptr = out_ptr;
                        let flags_ptr = flags_ptr;

                        let tid = ctx.index();
                        let start = tid * chunk_size;
                        let end = (start + chunk_size).min(count);
                        if start >= end {
                            return;
                        }

                        let halo = unsafe { halo_pool.get_mut_by_index(tid) };

                        for bid in start..end {
                            if grid_ref.blockmap[bid].is_some() {
                                halo.fill(grid_ref, bid);
                                unsafe {
                                    // Safe in practice: the static partition guarantees
                                    // every `bid` is owned by exactly one broadcast
                                    // closure, so no two threads ever alias the same
                                    // output/flags slot — the same disjointness OpenMP's
                                    // `schedule(static)` relies on implicitly.
                                    let out_block = &mut *out_ptr.0.add(bid);
                                    let flags_block = &*flags_ptr.0.add(bid);
                                    kernel_laplacian_simd_16(halo, flags_block, out_block);
                                }
                            }
                        }
                    });
                    black_box(&mut output_blocks);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    //bench_hot_path_scaling,
    //bench_multithreaded_contention,
    //bench_cold_extension,
    bench_laplacian_halo_mocked,
    bench_laplacian_halo_real_fill
);
criterion_main!(benches);

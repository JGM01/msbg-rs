use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use msbg_rs::{
    blockpool::{Block, BlockPool},
    math::laplacian::kernel_laplacian_simd_16,
    multires::halo::HaloBlockPool,
    sparse_grid::{BlockPtr, SparseGrid},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::{env, hint::black_box, sync::Arc};

// 1 Block = ~16.06 KB
const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;

// Approximate occupancy fraction of the spherical-shell active-block generator.
const SHELL_OCCUPANCY: f64 = 0.14;

/// `MSBG_BENCH_SCALE=small` runs reduced sizes for fast compile/debug cycles.
/// Anything else (or unset) runs the full stress sizes.
#[derive(Clone, Copy, PartialEq)]
enum Scale {
    Small,
    Full,
}

fn scale() -> Scale {
    match env::var("MSBG_BENCH_SCALE").as_deref() {
        Ok("small") => Scale::Small,
        _ => Scale::Full,
    }
}

fn sample_size(s: Scale) -> usize {
    match s {
        Scale::Small => 10,
        Scale::Full => 100,
    }
}

fn blockpool_hot_counts(s: Scale) -> Vec<usize> {
    match s {
        Scale::Small => vec![1_000, 10_000, 50_000],
        Scale::Full => vec![10_000, 100_000, 250_000],
    }
}

fn compute_only_counts(s: Scale) -> Vec<usize> {
    match s {
        Scale::Small => vec![200, 1_000, 5_000],
        Scale::Full => vec![1_000, 10_000, 50_000],
    }
}

fn active_targets(s: Scale) -> Vec<usize> {
    match s {
        Scale::Small => vec![1_000, 5_000, 10_000],
        Scale::Full => vec![10_000, 50_000, 100_000],
    }
}

/// Deterministic sparse occupancy: blocks whose center lies inside a spherical
/// shell, mimicking a surface region. Same formula is mirrored on the C++ side.
fn generate_active_blocks(nx: usize, ny: usize, nz: usize) -> Vec<usize> {
    let cx = nx as f64 * 0.5;
    let cy = ny as f64 * 0.5;
    let cz = nz as f64 * 0.5;
    let r_out = (nx.min(ny).min(nz)) as f64 * 0.5;
    let r_in = r_out * 0.9;

    let mut active = Vec::new();
    for bz in 0..nz {
        for by in 0..ny {
            for bx in 0..nx {
                let dx = bx as f64 + 0.5 - cx;
                let dy = by as f64 + 0.5 - cy;
                let dz = bz as f64 + 0.5 - cz;
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d >= r_in && d <= r_out {
                    active.push(bx + by * nx + bz * nx * ny);
                }
            }
        }
    }
    active
}

/// Build a sparse grid sized so the shell yields roughly `active_target`
/// active blocks. Active blocks are materialized and pre-filled with a fixed
/// value so gather reads are deterministic (no denormal garbage).
fn build_sparse_grid(active_target: usize) -> (Arc<SparseGrid<f32, BSX, N>>, Vec<usize>) {
    let full = (active_target as f64 / SHELL_OCCUPANCY).cbrt().ceil() as usize;
    let bpd = full.max(8);
    let sx = bpd * BSX;
    let sy = bpd * BSX;
    let sz = bpd * BSX;

    let blocks_per_seg = 4096; // 64 MB segments (MSBG C++ standard)
    let max_segments = (active_target * 3 / blocks_per_seg) + 4;
    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(max_segments, blocks_per_seg));
    let mut grid = SparseGrid::new("bench".to_string(), sx, sy, sz, 0.0, 1.0, pool);

    let active = generate_active_blocks(grid.nx, grid.ny, grid.nz);

    for &bid in &active {
        if grid.blockmap[bid].is_none() {
            let ptr = BlockPtr(grid.block_pool.alloc_block());
            unsafe {
                (*ptr.as_ptr()).data.fill(0.5);
            }
            grid.blockmap[bid] = Some(ptr);
        }
    }

    (Arc::new(grid), active)
}

fn bench_hot_path_scaling(c: &mut Criterion) {
    let s = scale();
    let mut group = c.benchmark_group("blockpool_hot_path");
    group.sample_size(sample_size(s));

    for block_count in blockpool_hot_counts(s) {
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
    let s = scale();
    let mut group = c.benchmark_group("blockpool_contention");
    group.sample_size(sample_size(s));

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
    let s = scale();
    let mut group = c.benchmark_group("blockpool_cold_alloc");
    group.sample_size(sample_size(s));

    let cold_blocks = match s {
        Scale::Small => 1_000,
        Scale::Full => 10_000,
    };

    // Benchmark the cost of creating and expanding a pool from scratch (including OS page allocations)
    group.bench_function(format!("cold_alloc_{}_blocks", cold_blocks), |b| {
        b.iter(|| {
            let pool = BlockPool::<f32, BSX, N>::new(16, 1024);
            for _ in 0..cold_blocks {
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

fn bench_laplacian_compute_only(c: &mut Criterion) {
    let s = scale();
    let mut group = c.benchmark_group("laplacian_compute_only");
    group.sample_size(sample_size(s));

    for block_count in compute_only_counts(s) {
        group.throughput(Throughput::Elements((block_count * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("mocked_fill", block_count),
            &block_count,
            |b, &count| {
                let num_threads = rayon::current_num_threads();
                let halo_pool = HaloBlockPool::<f32, BSX, HSX>::new(num_threads);

                let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); count];
                let flags_blocks: Vec<Block<u16, BSX, N>> = vec![Block::new(); count];

                b.iter(|| {
                    // Using zip so we can iterate both arrays together
                    output_blocks
                        .par_iter_mut()
                        .zip(flags_blocks.par_iter())
                        .for_each(|(out_block, flags_block)| {
                            let halo = unsafe { halo_pool.get_mut() };

                            // Mocking the memory bandwidth of gathering the neighborhood via memset
                            halo.mock_fill_for_bench();

                            kernel_laplacian_simd_16(halo, flags_block, out_block);
                        });
                    black_box(&mut output_blocks);
                });
            },
        );
    }
    group.finish();
}

fn bench_halo_gather(c: &mut Criterion) {
    let s = scale();
    let mut group = c.benchmark_group("halo_gather");
    group.sample_size(sample_size(s));

    for target in active_targets(s) {
        let (grid, active) = build_sparse_grid(target);
        let num_threads = rayon::current_num_threads();
        let halo_pool = HaloBlockPool::<f32, BSX, HSX>::new(num_threads);

        group.throughput(Throughput::Elements((active.len() * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("shell_fill", target),
            &active,
            |b, active| {
                b.iter(|| {
                    active.par_iter().for_each(|&bid| {
                        let halo = unsafe { halo_pool.get_mut() };
                        halo.fill(&grid, bid);
                        black_box(halo.data.as_ref());
                    });
                });
            },
        );
    }
    group.finish();
}

fn bench_laplacian_smoothing_e2e(c: &mut Criterion) {
    let s = scale();
    let mut group = c.benchmark_group("laplacian_smoothing_e2e");
    group.sample_size(sample_size(s));

    for target in active_targets(s) {
        let (grid, active) = build_sparse_grid(target);
        let n_active = active.len();
        let num_threads = rayon::current_num_threads();
        let halo_pool = HaloBlockPool::<f32, BSX, HSX>::new(num_threads);

        let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); n_active];
        let flags_blocks: Vec<Block<u16, BSX, N>> = vec![Block::new(); n_active];

        group.throughput(Throughput::Elements((n_active * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("shell_sweep", target),
            &active,
            |b, active| {
                b.iter(|| {
                    active
                        .par_iter()
                        .zip(output_blocks.par_iter_mut().zip(flags_blocks.par_iter()))
                        .for_each(|(&bid, (out_block, flags_block))| {
                            let halo = unsafe { halo_pool.get_mut() };

                            // True gather latency from 6 spatial boundaries across RAM
                            halo.fill(&grid, bid);

                            kernel_laplacian_simd_16(halo, flags_block, out_block);
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
    bench_hot_path_scaling,
    bench_multithreaded_contention,
    bench_cold_extension,
    bench_laplacian_compute_only,
    bench_halo_gather,
    bench_laplacian_smoothing_e2e
);
criterion_main!(benches);

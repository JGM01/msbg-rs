use criterion::{BenchmarkId, Criterion, Throughput};
use msbg_rs::{
    blockpool::{Block, BlockPool},
    math::BoundaryCondition,
    math::laplacian::kernel_laplacian,
    math::meancurv::kernel_meancurv,
    math::simd::LANES,
    math::stencil::MaskBlock,
    multires::halo::{HaloBlock, HaloBlockPool},
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
const CHUNKS: usize = N / LANES;

// Approximate occupancy fraction of the spherical-shell active-block generator.
const SHELL_OCCUPANCY: f64 = 0.14;

/// Target machine. Auto-detected from the OS; override with `MSBG_BENCH_MACHINE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Machine {
    Dell,
    Macbook,
    Windows,
}

/// Parsed `--scale`/`--machine` flags plus the benchmark FILTER.
struct Cli {
    scale: Option<String>,
    machine: Option<String>,
    filter: Option<String>,
}

fn cli() -> &'static Cli {
    static CLI: std::sync::OnceLock<Cli> = std::sync::OnceLock::new();
    CLI.get_or_init(parse_cli)
}

fn parse_cli() -> Cli {
    const VALUE_FLAGS: &[&str] = &[
        "--color",
        "--save-baseline",
        "--baseline",
        "--format",
        "--profile-time",
        "--sample-size",
        "--measurement-time",
        "--warm-up-time",
        "--nresamples",
        "--load-baseline",
        "--plotting-backend",
        "--output-format",
    ];

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut scale = None;
    let mut machine = None;
    let mut filter = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--scale" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                scale = Some(args[i + 1].clone());
                i += 1;
            }
        } else if a == "--machine" {
            if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                machine = Some(args[i + 1].clone());
                i += 1;
            }
        } else if a == "--bench" || a == "--test" {
            // cargo passes --bench; --test is the libtest-compat toggle that is ignored.
        } else if VALUE_FLAGS.contains(&a) {
            i += 1; // skip the flag's value
        } else if a.starts_with('-') {
            eprintln!("note: ignoring unsupported bench flag '{a}'");
        } else if filter.is_none() {
            filter = Some(a.to_string());
        }
        i += 1;
    }
    Cli {
        scale,
        machine,
        filter,
    }
}

fn machine() -> Machine {
    let raw = cli()
        .machine
        .clone()
        .or_else(|| env::var("MSBG_BENCH_MACHINE").ok());
    match raw.as_deref() {
        Some("dell") => Machine::Dell,
        Some("macbook") => Machine::Macbook,
        Some("windows") => Machine::Windows,
        Some(other) => panic!("unknown MSBG_BENCH_MACHINE '{other}' (use dell|macbook|windows)"),
        None => {
            if cfg!(target_os = "macos") {
                Machine::Macbook
            } else if cfg!(target_os = "windows") {
                Machine::Windows
            } else {
                Machine::Dell
            }
        }
    }
}

/// Benchmark scale. `small` is identical on every machine (cross-machine
/// comparison); `big` is the per-machine full stress; `xbig` is the aggressive
/// MacBook-only stress.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Size {
    Small,
    Big,
    XBig,
}

fn size() -> Size {
    let raw = cli().scale.clone().or_else(|| env::var("MSBG_BENCH_SCALE").ok());
    match raw.as_deref() {
        Some("small") => Size::Small,
        Some("big") | Some("full") => Size::Big,
        Some("xbig") => Size::XBig,
        Some(other) => panic!("unknown MSBG_BENCH_SCALE '{other}' (use small|big|xbig)"),
        None => Size::Big,
    }
}

fn resolve() -> (Machine, Size) {
    let m = machine();
    let s = size();
    if s == Size::XBig && m != Machine::Macbook {
        panic!("MSBG_BENCH_SCALE=xbig needs >=32GB RAM (MacBook); use big on this machine");
    }
    (m, s)
}

fn sample_size(s: Size) -> usize {
    match s {
        Size::Small => 10,
        Size::Big => 100,
        // xbig legs run ~0.6-1.8s each at 500k-750k active; 100 samples grinds
        // for minutes and thermal-throttles the M3 Pro, so keep it short.
        Size::XBig => 20,
    }
}

fn blockpool_hot_counts(m: Machine, s: Size) -> Vec<usize> {
    match (m, s) {
        (_, Size::Small) => vec![1_000, 10_000, 50_000],
        (Machine::Dell, Size::Big) => vec![10_000, 100_000, 250_000],
        // 9600X (6c/12t, 16 GB): same RAM envelope as the Dell, so same sizes.
        (Machine::Windows, Size::Big) => vec![10_000, 100_000, 250_000],
        (Machine::Macbook, Size::Big) => vec![100_000, 500_000, 1_000_000],
        (Machine::Macbook, Size::XBig) => vec![100_000, 1_000_000, 1_500_000],
        (Machine::Dell | Machine::Windows, Size::XBig) => unreachable!("xbig guarded in resolve()"),
    }
}

fn compute_only_counts(m: Machine, s: Size) -> Vec<usize> {
    match (m, s) {
        (_, Size::Small) => vec![200, 1_000, 5_000],
        (Machine::Dell, Size::Big) => vec![1_000, 10_000, 50_000],
        (Machine::Windows, Size::Big) => vec![1_000, 10_000, 50_000],
        (Machine::Macbook, Size::Big) => vec![1_000, 10_000, 50_000],
        (Machine::Macbook, Size::XBig) => vec![10_000, 50_000, 100_000],
        (Machine::Dell | Machine::Windows, Size::XBig) => unreachable!("xbig guarded in resolve()"),
    }
}

fn active_targets(m: Machine, s: Size) -> Vec<usize> {
    match (m, s) {
        (_, Size::Small) => vec![1_000, 5_000, 10_000],
        (Machine::Dell, Size::Big) => vec![10_000, 50_000, 100_000],
        // 16 GB caps the shell sweep around ~100k active (grid + output + flags).
        (Machine::Windows, Size::Big) => vec![10_000, 50_000, 100_000],
        (Machine::Macbook, Size::Big) => vec![50_000, 250_000, 500_000],
        (Machine::Macbook, Size::XBig) => vec![50_000, 250_000, 750_000],
        (Machine::Dell | Machine::Windows, Size::XBig) => unreachable!("xbig guarded in resolve()"),
    }
}

fn cold_block_count(s: Size) -> usize {
    match s {
        Size::Small => 1_000,
        Size::Big | Size::XBig => 10_000,
    }
}

fn voxel_size(s: Size) -> usize {
    match s {
        Size::Small => 64,
        Size::Big | Size::XBig => 128,
    }
}

fn print_banner(m: Machine, s: Size) {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    eprintln!(
        "[msbg-rs bench] machine={:?} scale={:?} arch={} os={} logical_cores={} rayon_threads={}",
        m,
        s,
        std::env::consts::ARCH,
        std::env::consts::OS,
        cores,
        rayon::current_num_threads(),
    );
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

/// Lazily-built rayon pool with FTZ/DAZ workers.
fn thread_pool() -> &'static msbg_rs::thread_pool::Pool {
    use std::sync::OnceLock;
    static POOL: OnceLock<msbg_rs::thread_pool::Pool> = OnceLock::new();
    POOL.get_or_init(|| msbg_rs::thread_pool::Pool::new(rayon::current_num_threads()))
}

fn bench_hot_path_scaling(c: &mut Criterion) {
    let (m, s) = resolve();
    let mut group = c.benchmark_group("blockpool_hot_path");
    group.sample_size(sample_size(s));

    for block_count in blockpool_hot_counts(m, s) {
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
    let (_, s) = resolve();
    let mut group = c.benchmark_group("blockpool_contention");
    group.sample_size(sample_size(s));

    let num_threads = rayon::current_num_threads();
    let blocks_per_thread = 4096;
    let total_blocks = num_threads * blocks_per_thread;

    let blocks_per_seg = 4096; // matches the C++ contention scenario
    let max_segments = (total_blocks / blocks_per_seg) + 16;

    let pool = Arc::new(BlockPool::<f32, BSX, N>::new(max_segments, blocks_per_seg));

    group.bench_function(
        format!("{}_threads_{}_blocks", num_threads, total_blocks),
        |b| {
            b.iter(|| {
                thread_pool().install(|| {
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
                });
                pool.reset();
            });
        },
    );

    group.finish();
}

fn bench_cold_extension(c: &mut Criterion) {
    let (_, s) = resolve();
    let mut group = c.benchmark_group("blockpool_cold_alloc");
    group.sample_size(sample_size(s));

    let cold_blocks = cold_block_count(s);

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
    let (m, s) = resolve();
    let mut group = c.benchmark_group("laplacian_compute_only");
    group.sample_size(sample_size(s));

    for block_count in compute_only_counts(m, s) {
        group.throughput(Throughput::Elements((block_count * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("kernel_only", block_count),
            &block_count,
            |b, &count| {
                // Real data: gather one halo from a real block once (gather is
                // measured separately in halo_gather), build the fluid mask
                // once, then time only the kernel over `count` fresh outputs.
                let (grid, active) = build_sparse_grid(1);
                let bid = active[0];
                let mut halo = HaloBlock::<BSX, HSX>::new();
                halo.fill::<1, false, f32, N>(&grid, bid, BoundaryCondition::Neumann);
                let flags = Block::<u16, BSX, N>::new();
                let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);
                let dt = 0.05;

                let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); count];

                b.iter(|| {
                    thread_pool().install(|| {
                        output_blocks
                            .par_iter_mut()
                            .for_each(|out_block| {
                                kernel_laplacian::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, dt, out_block);
                            });
                    });
                    black_box(&mut output_blocks);
                });
            },
        );
    }
    group.finish();
}

fn bench_mean_curvature_compute_only(c: &mut Criterion) {
    let (m, s) = resolve();
    let mut group = c.benchmark_group("mean_curvature_compute_only");
    group.sample_size(sample_size(s));

    for block_count in compute_only_counts(m, s) {
        group.throughput(Throughput::Elements((block_count * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("kernel_only", block_count),
            &block_count,
            |b, &count| {
                let (grid, active) = build_sparse_grid(1);
                let bid = active[0];
                let mut halo = HaloBlock::<BSX, HSX>::new();
                halo.fill::<1, true, f32, N>(&grid, bid, BoundaryCondition::Neumann);
                let flags = Block::<u16, BSX, N>::new();
                let mask = MaskBlock::<LANES, CHUNKS>::build(&flags);
                let dt = 0.05;

                let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); count];

                b.iter(|| {
                    thread_pool().install(|| {
                        output_blocks
                            .par_iter_mut()
                            .for_each(|out_block| {
                                kernel_meancurv::<LANES, CHUNKS, BSX, HSX, N>(&halo, &mask, dt, out_block);
                            });
                    });
                    black_box(&mut output_blocks);
                });
            },
        );
    }
    group.finish();
}

fn bench_halo_gather(c: &mut Criterion) {
    let (m, s) = resolve();
    let mut group = c.benchmark_group("halo_gather");
    group.sample_size(sample_size(s));

    for target in active_targets(m, s) {
        let (grid, active) = build_sparse_grid(target);
        let num_threads = rayon::current_num_threads();
        let halo_pool = HaloBlockPool::<BSX, HSX>::new(num_threads);

        group.throughput(Throughput::Elements((active.len() * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("shell_fill_full", target),
            &active,
            |b, active| {
                b.iter(|| {
                    thread_pool().install(|| {
                        active.par_iter().for_each(|&bid| {
                            let halo = unsafe { halo_pool.get_mut() };
                            halo.fill::<1, true, f32, N>(&grid, bid, BoundaryCondition::Neumann);
                            black_box(halo.data.as_ref());
                        });
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("shell_fill_faces", target),
            &active,
            |b, active| {
                b.iter(|| {
                    thread_pool().install(|| {
                        active.par_iter().for_each(|&bid| {
                            let halo = unsafe { halo_pool.get_mut() };
                            halo.fill::<1, false, f32, N>(&grid, bid, BoundaryCondition::Neumann);
                            black_box(halo.data.as_ref());
                        });
                    });
                });
            },
        );
    }
    group.finish();
}

fn bench_laplacian_smoothing_e2e(c: &mut Criterion) {
    let (m, s) = resolve();
    let mut group = c.benchmark_group("laplacian_smoothing_e2e");
    group.sample_size(sample_size(s));

    for target in active_targets(m, s) {
        let (grid, active) = build_sparse_grid(target);
        let n_active = active.len();
        let num_threads = rayon::current_num_threads();
        let halo_pool = HaloBlockPool::<BSX, HSX>::new(num_threads);

        let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); n_active];
        let flags_blocks: Vec<Block<u16, BSX, N>> = vec![Block::new(); n_active];
        let dt = 0.05;

        group.throughput(Throughput::Elements((n_active * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("shell_sweep", target),
            &active,
            |b, active| {
                b.iter(|| {
                    thread_pool().install(|| {
                        active
                            .par_iter()
                            .zip(output_blocks.par_iter_mut().zip(flags_blocks.par_iter()))
                            .for_each(|(&bid, (out_block, flags_block))| {
                                let halo = unsafe { halo_pool.get_mut() };

                                // True gather latency from 6 spatial boundaries across RAM
                                halo.fill::<1, false, f32, N>(&grid, bid, BoundaryCondition::Neumann);
                                let mask = MaskBlock::<LANES, CHUNKS>::build(flags_block);

                                kernel_laplacian::<LANES, CHUNKS, BSX, HSX, N>(halo, &mask, dt, out_block);
                            });
                    });
                    black_box(&mut output_blocks);
                });
            },
        );
    }
    group.finish();
}

fn bench_mean_curvature_shell_sweep(c: &mut Criterion) {
    let (m, s) = resolve();
    let mut group = c.benchmark_group("mean_curvature_shell_sweep");
    group.sample_size(sample_size(s));

    for target in active_targets(m, s) {
        let (grid, active) = build_sparse_grid(target);
        let n_active = active.len();
        let num_threads = rayon::current_num_threads();
        let halo_pool = HaloBlockPool::<BSX, HSX>::new(num_threads);

        let mut output_blocks: Vec<Block<f32, BSX, N>> = vec![Block::new(); n_active];
        let flags_blocks: Vec<Block<u16, BSX, N>> = vec![Block::new(); n_active];
        let dt = 0.05;

        group.throughput(Throughput::Elements((n_active * N) as u64));
        group.bench_with_input(
            BenchmarkId::new("shell_sweep", target),
            &active,
            |b, active| {
                b.iter(|| {
                    thread_pool().install(|| {
                        active
                            .par_iter()
                            .zip(output_blocks.par_iter_mut().zip(flags_blocks.par_iter()))
                            .for_each(|(&bid, (out_block, flags_block))| {
                                let halo = unsafe { halo_pool.get_mut() };

                                // Full 18^3 gather (faces + edges + corners)
                                halo.fill::<1, true, f32, N>(&grid, bid, BoundaryCondition::Neumann);
                                let mask = MaskBlock::<LANES, CHUNKS>::build(flags_block);

                                kernel_meancurv::<LANES, CHUNKS, BSX, HSX, N>(halo, &mask, dt, out_block);
                            });
                    });
                    black_box(&mut output_blocks);
                });
            },
        );
    }
    group.finish();
}

fn bench_voxel_access(c: &mut Criterion) {
    let (_, s) = resolve();
    let mut group = c.benchmark_group("voxel_access");
    group.sample_size(sample_size(s));

    let size = voxel_size(s);

    let mut grid: SparseGrid<f32, BSX, N> = SparseGrid::new(
        "voxel".to_string(),
        size,
        size,
        size,
        0.0,
        1.0,
        Arc::new(BlockPool::<f32, BSX, N>::new(16, 4096)),
    );

    // Mixed value states: full every 3rd block, allocated every 3rd+1, empty rest.
    for bz in 0..grid.nz {
        for by in 0..grid.ny {
            for bx in 0..grid.nx {
                let bid = bx + by * grid.nx + bz * grid.nxy;
                match bid % 3 {
                    0 => grid.set_full_block(bid),
                    1 => grid.set_voxel(bx * BSX, by * BSX, bz * BSX, 0.5),
                    _ => {}
                }
            }
        }
    }

    group.bench_function("get_sequential", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        acc += grid.get_voxel_unchecked(x, y, z);
                    }
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("set_allocated", |b| {
        b.iter(|| {
            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        grid.set_value_in_block(
                            grid.get_block_id(x, y, z),
                            SparseGrid::<f32, BSX, N>::get_voxel_id(x, y, z),
                            x as f32,
                        );
                    }
                }
            }
            black_box(&grid);
        });
    });
}

fn main() {
    let _ = thread_pool(); // eager build so warmup isn't charged
    print_banner(machine(), size());

    let mut criterion = Criterion::default();
    if let Some(f) = &cli().filter {
        criterion = criterion.with_filter(f.clone());
    }

    bench_hot_path_scaling(&mut criterion);
    bench_multithreaded_contention(&mut criterion);
    bench_cold_extension(&mut criterion);
    bench_voxel_access(&mut criterion);
    bench_laplacian_compute_only(&mut criterion);
    bench_mean_curvature_compute_only(&mut criterion);
    bench_halo_gather(&mut criterion);
    bench_laplacian_smoothing_e2e(&mut criterion);
    bench_mean_curvature_shell_sweep(&mut criterion);
}

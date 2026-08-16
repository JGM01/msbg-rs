//! Step-7 multires benchmarks: refinement-map setup, multires halo gather, and
//! downsampling. Mirrors the C++ `benchmark.cpp` "multires" scenario (real
//! library calls on both sides, no mocks).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use msbg_rs::blockpool::BlockPool;
use msbg_rs::multires::halo::{fill_multires, HaloBlockPool};
use msbg_rs::multires::{Level, MultiresGrid, RefinementMap};
use msbg_rs::sparse_grid::SparseGrid;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{env, hint::black_box, sync::Arc};

const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;
const SHELL_OCCUPANCY: f64 = 0.14;

fn scale() -> &'static str {
    static SCALE: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    SCALE.get_or_init(|| match env::var("MSBG_BENCH_SCALE").as_deref() {
        Ok("small") => "small",
        _ => "big",
    })
}

fn targets() -> Vec<usize> {
    if scale() == "small" {
        vec![500, 2_000, 5_000]
    } else {
        vec![5_000, 20_000, 50_000]
    }
}

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

fn thread_pool() -> &'static msbg_rs::thread_pool::Pool {
    use std::sync::OnceLock;
    static POOL: OnceLock<msbg_rs::thread_pool::Pool> = OnceLock::new();
    POOL.get_or_init(|| msbg_rs::thread_pool::Pool::new(rayon::current_num_threads()))
}

fn materialize_blocks<D: Copy + Default + Send + Sync, const B: usize, const N2: usize>(
    grid: &mut SparseGrid<D, B, N2>,
    bids: &[usize],
    val: D,
) {
    for &bid in bids {
        grid.ensure_block(bid);
        if let Some(p) = grid.blockmap[bid] {
            unsafe {
                (*p.as_ptr()).data.fill(val);
            }
        }
    }
}

/// Build a 3-level grid, set a spherical-shell refinement map (fine shell = 0,
/// sea = 2; `regularizeRefinementMap` inserts the level-1 ring), and materialize
/// the level-0 shell + level-1 ring density blocks. Returns the grid and the
/// fine (level-0) block set.
fn build_multires_grid(active_target: usize) -> (MultiresGrid, Vec<usize>) {
    let bpd = ((active_target as f64 / SHELL_OCCUPANCY).cbrt().ceil() as usize).max(8);
    let sx = bpd * BSX;
    let sy = bpd * BSX;
    let sz = bpd * BSX;

    let mut grid = MultiresGrid::create("bench", sx, sy, sz, BSX, 3, 6);

    let n_blocks = grid.dims.n_blocks;
    let mut map = RefinementMap::uniform(n_blocks, 2);
    let shell = generate_active_blocks(grid.dims.nx, grid.dims.ny, grid.dims.nz);
    for &bid in &shell {
        map.levels[bid] = 0;
    }
    grid.set_refinement_map(&mut map);

    let levels = grid.block_info.level0.clone();
    let fine: Vec<usize> = (0..n_blocks).filter(|&b| levels[b] == 0).collect();
    let ring: Vec<usize> = (0..n_blocks).filter(|&b| levels[b] == 1).collect();

    match &mut grid.levels[0] {
        Level::B16(lv) => materialize_blocks(&mut lv.density, &fine, 0.5),
        _ => unreachable!("level 0 block size"),
    }
    match &mut grid.levels[1] {
        Level::B8(lv) => materialize_blocks(&mut lv.density, &ring, 0.5),
        _ => unreachable!("level 1 block size"),
    }

    (grid, fine)
}

fn bench_set_refinement_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("multires_set_refinement_map");
    group.sample_size(30);

    for target in targets() {
        group.throughput(Throughput::Elements(target as u64));
        group.bench_with_input(BenchmarkId::new("topology", target), &target, |b, &t| {
            let bpd = ((t as f64 / SHELL_OCCUPANCY).cbrt().ceil() as usize).max(8);
            let mut grid =
                MultiresGrid::create("bench", bpd * BSX, bpd * BSX, bpd * BSX, BSX, 3, 6);
            let shell = generate_active_blocks(grid.dims.nx, grid.dims.ny, grid.dims.nz);
            let mut map = RefinementMap::uniform(grid.dims.n_blocks, 2);
            for &bid in &shell {
                map.levels[bid] = 0;
            }
            let seed = map.levels.clone();

            b.iter(|| {
                // Reset (regularize mutates in place) then run the full
                // set_refinement_map: regularize + topology + cell flags +
                // distFineCoarse.
                map.levels.copy_from_slice(&seed);
                let topo = grid.set_refinement_map(&mut map);
                black_box(&topo);
            });
        });
    }
    group.finish();
}

fn bench_multires_halo(c: &mut Criterion) {
    let mut group = c.benchmark_group("multires_halo_gather");
    group.sample_size(20);

    for target in targets() {
        group.throughput(Throughput::Elements((target as u64) * N as u64));
        group.bench_with_input(BenchmarkId::new("shell_fill", target), &target, |b, &_t| {
            let (grid, active) = build_multires_grid(target);
            let levels = grid.block_info.level0.clone();

            let (fine, coarse) = match (&grid.levels[0], &grid.levels[1]) {
                (Level::B16(l0), Level::B8(l1)) => (&l0.density, &l1.density),
                _ => unreachable!("level 0/1 block sizes"),
            };

            let pool = Arc::new(HaloBlockPool::<BSX, HSX>::new(rayon::current_num_threads()));
            b.iter(|| {
                thread_pool().install(|| {
                    active.par_iter().for_each(|&bid| {
                        let halo = unsafe { pool.get_mut() };
                        fill_multires::<BSX, HSX, 1, true, f32, N, 8, 512>(
                            halo, fine, coarse, &levels, bid,
                            msbg_rs::math::BoundaryCondition::Neumann,
                        );
                        black_box(&halo.data);
                    });
                });
            });
        });
    }
    group.finish();
}

fn bench_multires_downsample(c: &mut Criterion) {
    let mut group = c.benchmark_group("multires_downsample");
    group.sample_size(30);

    for target in targets() {
        let (grid, _active) = build_multires_grid(target);
        group.throughput(Throughput::Elements((grid.dims.n_blocks * 512) as u64));
        group.bench_with_input(BenchmarkId::new("avg_2x2x2", target), &target, |b, &_t| {
            let (fine, coarse) = match (&grid.levels[0], &grid.levels[1]) {
                (Level::B16(l0), Level::B8(l1)) => (&l0.density, &l1.density),
                _ => unreachable!("level 0/1 block sizes"),
            };
            let mut scratch = SparseGrid::<f32, 8, 512>::new(
                "scratch".into(),
                coarse.sx,
                coarse.sy,
                coarse.sz,
                0.0,
                1.0,
                Arc::new(BlockPool::new(coarse.n_blocks / 4096 + 2, 4096)),
            );
            for bid in 0..scratch.n_blocks {
                scratch.ensure_block(bid);
            }
            b.iter(|| {
                msbg_rs::multires::downsample::downsample_channel_avg::<16, 4096, 8, 512>(
                    &mut scratch, fine,
                );
                black_box(&scratch);
            });
        });
    }
    group.finish();
}

criterion_group!(
    multires,
    bench_set_refinement_map,
    bench_multires_halo,
    bench_multires_downsample
);
criterion_main!(multires);

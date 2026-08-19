use msbg_rs::{
    blockpool::{Block, BlockPool},
    channel::Density,
    math::boundary::BoundaryCondition,
    multires::halo::HaloBlockPool,
    solver::{Fence, PdeParams, Stencil, Sweeper},
    sparse_grid::SparseGrid,
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{hint::black_box, sync::Arc, time::Instant};

const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;

// Virtual Domain: 4096^3 voxels = 256^3 blocks
const DOMAIN_VOXELS: usize = 4096;
const GRID_BLOCKS_PER_AXIS: usize = DOMAIN_VOXELS / BSX;

/// Implicit Gyroid surface
fn gyroid(x: f32, y: f32, z: f32, freq: f32) -> f32 {
    let (sx, cx) = (x * freq).sin_cos();
    let (sy, cy) = (y * freq).sin_cos();
    let (sz, cz) = (z * freq).sin_cos();
    sx * cy + sy * cz + sz * cx
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("============================================================");
    println!(" msbg-rs 16-Bit Scale Stress Benchmark (M3 Pro 36GB)");
    println!(" Threads: {}", threads);
    println!(
        " Virtual Resolution: {0}x{0}x{0} ({1:.2} Billion Virtual Voxels)",
        DOMAIN_VOXELS,
        (DOMAIN_VOXELS as f64).powi(3) / 1e9
    );
    println!("============================================================\n");

    // 1. Domain Allocation (Now using 16-bit Density)
    println!("[1/4] Allocating 16-Bit Sparse Grid & Materializing Surface...");
    let start_setup = Instant::now();
    let blocks_per_seg = 4096;
    let max_segments = 1024; // 4.19 Million block capacity
    let pool = Arc::new(BlockPool::<Density, BSX, N>::new(
        max_segments,
        blocks_per_seg,
    ));

    let mut grid = SparseGrid::new(
        "hero_grid".to_string(),
        DOMAIN_VOXELS,
        DOMAIN_VOXELS,
        DOMAIN_VOXELS,
        Density::from_f32(0.0),
        Density::from_f32(1.0),
        pool,
    );

    let freq = 0.08f32;
    let thickness = 0.07f32; // Thick enough to hit 3 Billion voxels

    let mut candidate_blocks: Vec<usize> = (0..grid.nz)
        .into_par_iter()
        .flat_map(|bz| {
            let mut local = Vec::new();
            for by in 0..GRID_BLOCKS_PER_AXIS {
                for bx in 0..GRID_BLOCKS_PER_AXIS {
                    let cx = (bx * BSX + BSX / 2) as f32;
                    let cy = (by * BSX + BSX / 2) as f32;
                    let cz = (bz * BSX + BSX / 2) as f32;
                    let val = gyroid(cx, cy, cz, freq).abs();
                    if val < thickness {
                        local.push(bx + by * GRID_BLOCKS_PER_AXIS + bz * grid.nxy);
                    }
                }
            }
            local
        })
        .collect();

    let target_blocks = 750_000;
    if candidate_blocks.len() > target_blocks {
        candidate_blocks.truncate(target_blocks);
    }

    // Materialize active blocks in 16-bit
    for &bid in &candidate_blocks {
        let (bx, by, bz) = grid.get_block_coords_by_id(bid);
        grid.ensure_block(bid);
        let data = grid.get_block_data_mut(bid).unwrap();
        for z in 0..BSX {
            for y in 0..BSX {
                for x in 0..BSX {
                    let gx = (bx * BSX + x) as f32;
                    let gy = (by * BSX + y) as f32;
                    let gz = (bz * BSX + z) as f32;
                    let raw_val = gyroid(gx, gy, gz, freq);

                    // Map Gyroid [-1.5, 1.5] into Density [0.0, 1.0] range
                    let norm_val = (raw_val * 0.333 + 0.5).clamp(0.0, 1.0);
                    data[x + y * BSX + z * BSX * BSX] = Density::from_f32(norm_val);
                }
            }
        }
    }

    let n_active = candidate_blocks.len();
    let total_active_voxels = n_active * N;

    // RAM calc: Grid (u16) + Output (u16) + Flags (u16)
    let bytes_per_active = std::mem::size_of::<Block<Density, BSX, N>>() * 2
        + std::mem::size_of::<Block<u16, BSX, N>>();
    let active_gb = (n_active * bytes_per_active) as f64 / 1e9;
    let dense_equivalent_gb = ((DOMAIN_VOXELS as f64).powi(3) * 4.0) / 1e9;

    println!(
        "      Active Blocks:        {} ({:.2} Billion active voxels)",
        n_active,
        total_active_voxels as f64 / 1e9
    );
    println!(
        "      Active Grid Memory:   {:.2} GB (Dense Equivalent: {:.1} GB)",
        active_gb, dense_equivalent_gb
    );
    println!("      Setup Time:           {:.2?}", start_setup.elapsed());

    // 2. Prepare Working Buffers
    println!("\n[2/4] Initializing Stencil Buffers & Thread Pools...");
    let halo_pool = HaloBlockPool::<BSX, HSX>::new(threads);

    // 3. Benchmark: Halo Gather Throughput
    println!("\n[3/4] Benchmarking 16-bit Halo Gather Throughput...");
    let gather_iters = 3;
    let start_gather = Instant::now();
    for _ in 0..gather_iters {
        candidate_blocks.par_iter().for_each(|&bid| {
            let halo = unsafe { halo_pool.get_mut() };
            // Automatically dequantizes Density(u16) to f32 inside the L1 cache!
            halo.fill::<1, true, Density, N>(&grid, bid, BoundaryCondition::Neumann);
            black_box(halo.data.as_ref());
        });
    }
    let gather_duration = start_gather.elapsed() / gather_iters as u32;
    let gather_gvoxels = (total_active_voxels as f64 / gather_duration.as_secs_f64()) / 1e9;
    println!(
        "      Avg Gather Time:      {:.2?} ({:.2} Gvoxels/s)",
        gather_duration, gather_gvoxels
    );

    // 4. Benchmark: End-to-End Mean-Curvature PDE Smoothing Sweep
    println!("\n[4/4] Benchmarking 19-tap Mean Curvature PDE Solve...");
    let active: Vec<usize> = candidate_blocks.clone();
    let sweeper = Sweeper::<Density, BSX, N, HSX>::new(&grid, threads, Fence::Sfence);
    let params = PdeParams { dt: 0.025, iterations: 1, do_constr_zero_one: false };
    let pde_iters = 3;
    let start_pde = Instant::now();
    for _ in 0..pde_iters {
        sweeper.sweep(&active, Stencil::MeanCurvature, &params);
        black_box(&grid);
    }
    let pde_duration = start_pde.elapsed() / pde_iters as u32;
    let pde_gvoxels = (total_active_voxels as f64 / pde_duration.as_secs_f64()) / 1e9;
    println!(
        "      Avg PDE Solve Time:   {:.2?} ({:.2} Gvoxels/s)",
        pde_duration, pde_gvoxels
    );

    println!("\n============================================================");
    println!(" RESUME METRIC SUMMARY (16-BIT)");
    println!(" - Scaled to a 4096^3 virtual domain (68.7B virtual voxels)");
    println!(
        " - Simulated {:.2}B active voxels in {:.1} GB RAM (Dense eq: {:.0} GB)",
        total_active_voxels as f64 / 1e9,
        active_gb,
        dense_equivalent_gb
    );
    println!(
        " - Peak PDE Solver Throughput: {:.2} Gvoxels/sec (~{:.0} ms / full pass)",
        pde_gvoxels,
        pde_duration.as_secs_f64() * 1000.0
    );
    println!("============================================================");
}

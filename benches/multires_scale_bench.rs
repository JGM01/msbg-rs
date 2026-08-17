//! Multires memory-bound scale-stress benchmark (M3 Pro 36 GB).
//!
//! Builds a real `MultiresGrid` whose fine-level density shell is sized to a
//! target RAM budget (default ~28 GB), then times the three multires phases a
//! large sparse volume actually exercises:
//!
//!   1. `set_refinement_map` — topology + cell flags + fine-coarse distances
//!      (the "build" phase, materializes the per-block `cell_flags` channel).
//!   2. multires halo gather (`fill_multires`) over every fine block — the
//!      memory-bound inner loop a multires solver runs per block per sweep.
//!   3. `downsample_channel_avg` — the fine→coarse restriction pass.
//!
//! Working-set sanity: at 28 GB the live set is ~700× the M3's total cache, so
//! every pass is DRAM-streaming bound — this is the regime the paper's
//! 100-billion-voxel result lives in, and where the small-scale (cache-resident)
//! benches stop being predictive.
//!
//! Knobs:
//!   MSBG_STRESS_GB        target density+flags working set (default 28)
//!   MSBG_BENCH_SCALE=small  override the budget to 0.5 GB (validates on the
//!                          Dell 5500U, which cannot host the full run)
//!
//! Note: this is the multires *data structure + memory-bound primitives* we
//! have today, not the integrated multires solver (the `Sweeper` is
//! single-level; see step 8). It characterizes the substrate that solver will
//! sit on.

use msbg_rs::blockpool::BlockPool;
use msbg_rs::math::boundary::BoundaryCondition;
use msbg_rs::multires::halo::{fill_multires, HaloBlockPool};
use msbg_rs::multires::{MultiresGrid, RefinementMap, Level};
use msbg_rs::sparse_grid::SparseGrid;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{env, hint::black_box, sync::Arc, time::Instant};

const BSX: usize = 16;
const N: usize = 4096;
const HSX: usize = 18;
const SHELL_OCCUPANCY: f64 = 0.14;

// Per fine block, empirically calibrated (small-scale RSS): f32 density (16 KiB)
// + u16 cell_flags (8 KiB) + blockmaps (27 grids × 8 B/block) + BlockInfoStore +
// ring/coarse density+flags + allocator overhead. Not a paper constant — it is
// verified by the live RSS printout on each run.
const BYTES_PER_FINE_BLOCK: f64 = 64.0 * 1024.0;

fn target_gb() -> f64 {
    match env::var("MSBG_BENCH_SCALE").as_deref() {
        Ok("small") => 0.5,
        _ => env::var("MSBG_STRESS_GB")
            .map(|v| v.parse().unwrap_or(28.0))
            .unwrap_or(28.0),
    }
}

fn rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map(|pages| pages * 4096)
            .unwrap_or(0)
    }
    #[cfg(target_os = "macos")]
    {
        // `ps -o rss=` reports resident KB; std-only, no mach bindings needed.
        std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

fn gb(bytes: f64) -> f64 {
    bytes / 1e9
}

/// Spherical-shell block set (fine level), identical to `multires_bench`.
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

/// `bytes_per_fine_block` also covers the ring/coarse density+flags, which are
/// negligible (block sizes 8³/4³) relative to the fine shell.
fn blocks_per_axis_for(target_gb: f64) -> usize {
    let n_fine = (target_gb * 1e9 / BYTES_PER_FINE_BLOCK) as usize;
    ((n_fine as f64 / SHELL_OCCUPANCY).cbrt().ceil() as usize).max(8)
}

fn main() {
    let gb_target = target_gb();
    let bpd = blocks_per_axis_for(gb_target);
    let sx = bpd * BSX;

    println!("============================================================");
    println!(" msbg-rs MULTIRES Scale-Stress (memory-bound)");
    println!(" target working set: {gb_target:.1} GB");
    println!(" threads: {}", rayon::current_num_threads());
    println!("============================================================\n");

    // ---- [1] build + refinement map ----------------------------------------
    println!("[1/4] set_refinement_map (topology + cell flags + fine-coarse dist)");
    let t0 = Instant::now();
    let mut grid = MultiresGrid::create("stress", sx, sx, sx, BSX, 3, 6);
    let n_blocks = grid.dims.n_blocks;
    let mut map = RefinementMap::uniform(n_blocks, 2);
    let shell = generate_active_blocks(grid.dims.nx, grid.dims.ny, grid.dims.nz);
    for &bid in &shell {
        map.levels[bid] = 0;
    }
    let topo = grid.set_refinement_map(&mut map);
    let refine_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "      domain {sx}³ = {} blocks, {} fine shell blocks, {:.2?} ({:.0} kblocks/s)",
        n_blocks,
        shell.len(),
        t0.elapsed(),
        n_blocks as f64 / refine_ms * 1000.0
    );
    black_box(&topo);

    let levels = grid.block_info.level0.clone();
    let fine: Vec<usize> = (0..n_blocks).filter(|&b| levels[b] == 0).collect();
    let ring: Vec<usize> = (0..n_blocks).filter(|&b| levels[b] == 1).collect();

    // ---- [2] materialize density -------------------------------------------
    println!("\n[2/4] materialize fine (f32) + ring density");
    let t0 = Instant::now();
    match &mut grid.levels[0] {
        Level::B16(lv) => materialize(&mut lv.density, &fine, 0.5f32),
        _ => unreachable!(),
    }
    match &mut grid.levels[1] {
        Level::B8(lv) => materialize(&mut lv.density, &ring, 0.5f32),
        _ => unreachable!(),
    }
    let mat_gb = (fine.len() as f64 * (BSX * BSX * BSX * 4) as f64
        + ring.len() as f64 * (8 * 8 * 8 * 4) as f64)
        / 1e9;
    println!(
        "      {:.2} GB density, {:.2?} ({:.1} GB/s)",
        mat_gb,
        t0.elapsed(),
        mat_gb / t0.elapsed().as_secs_f64()
    );
    println!("      RSS now: {:.2} GB", gb(rss_bytes() as f64));

    // ---- [3] multires halo gather ------------------------------------------
    println!("\n[3/4] multires halo gather (fill_multires over fine shell)");
    let (fine_g, coarse_g) = match (&grid.levels[0], &grid.levels[1]) {
        (Level::B16(l0), Level::B8(l1)) => (&l0.density, &l1.density),
        _ => unreachable!(),
    };
    let halo_pool = Arc::new(HaloBlockPool::<BSX, HSX>::new(rayon::current_num_threads()));
    let iters = 3;
    let t0 = Instant::now();
    for _ in 0..iters {
        fine.par_iter().for_each(|&bid| {
            let halo = unsafe { halo_pool.get_mut() };
            fill_multires::<BSX, HSX, 1, true, f32, N, 8, 512>(
                halo,
                fine_g,
                coarse_g,
                &levels,
                bid,
                BoundaryCondition::Neumann,
            );
            black_box(&halo.data);
        });
    }
    let per = t0.elapsed() / iters as u32;
    let halo_gb = fine.len() as f64 * (BSX * BSX * BSX * 4) as f64 / 1e9;
    let halo_gvox = fine.len() as f64 * N as f64 / per.as_secs_f64() / 1e9;
    let halo_gbs = halo_gb / per.as_secs_f64();
    println!(
        "      avg per pass: {:.2?} ({:.2} GB/s, {:.2} Gvox/s)",
        per, halo_gbs, halo_gvox
    );

    // ---- [4] downsample -----------------------------------------------------
    println!("\n[4/4] downsample (fine -> coarse restriction)");
    let mut scratch = SparseGrid::<f32, 8, 512>::new(
        "scratch".into(),
        coarse_g.sx,
        coarse_g.sy,
        coarse_g.sz,
        0.0,
        1.0,
        Arc::new(BlockPool::new(coarse_g.n_blocks / 4096 + 2, 4096)),
    );
    for bid in 0..scratch.n_blocks {
        scratch.ensure_block(bid);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        msbg_rs::multires::downsample::downsample_channel_avg::<16, 4096, 8, 512>(
            &mut scratch,
            fine_g,
        );
        black_box(&scratch);
    }
    let per = t0.elapsed() / iters as u32;
    let ds_gb = fine.len() as f64 * (BSX * BSX * BSX * 4) as f64 / 1e9;
    let ds_gbs = ds_gb / per.as_secs_f64();
    println!(
        "      avg per pass: {:.2?} ({:.2} GB/s)",
        per, ds_gbs
    );

    println!("\n============================================================");
    println!(" SUMMARY");
    println!("  domain: {}³ ({} blocks, {} fine)", sx, n_blocks, fine.len());
    println!("  final RSS: {:.2} GB", gb(rss_bytes() as f64));
    println!("  set_refinement_map: {refine_ms:.0} ms ({n_blocks} blocks)");
    println!("  multires halo: {halo_gvox:.2} Gvox/s ({halo_gbs:.2} GB/s)");
    println!("  downsample:     {ds_gbs:.2} GB/s");
    println!("============================================================");
}

fn materialize<D: Copy + Default + Send + Sync, const B: usize, const N2: usize>(
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

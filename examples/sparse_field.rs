use std::sync::Arc;

use msbg_rs::blockpool::BlockPool;
use msbg_rs::channel::Vec3;
use msbg_rs::math::{BoundaryCondition, GridAlignment, Interpolation, Sampler};
use msbg_rs::sparse_grid::SparseGrid;

fn main() {
    let pool = Arc::new(BlockPool::<f32, 16, 4096>::new(1, 16));
    let mut grid = SparseGrid::new("field".into(), 64, 64, 64, 0.0, 1.0, pool);

    for z in 16..32 {
        for y in 16..32 {
            for x in 16..32 {
                grid.set_voxel(x, y, z, (x + 2 * y + 3 * z) as f32);
            }
        }
    }

    let sampler = Sampler::new(&grid, GridAlignment::Corner, BoundaryCondition::Clamp);
    let position = Vec3::new(20.5, 21.25, 22.75);
    let value = sampler.sample::<{ Interpolation::Linear }>(position);

    println!("Domain: 64³ voxels ({} virtual blocks)", grid.n_blocks);
    println!("Occupied blocks: {}", grid.active_block_ids().count());
    println!("Interpolated field value: {value:.2} (expected 131.25)");
    println!("Value in empty space: {}", grid.get_voxel(0, 0, 0));
}

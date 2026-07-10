use std::{marker::PhantomData, ops::Add};

// Memory indices for flat structures
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelId(pub usize);

/// Represents the smallest unit, a coordinate within a block.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Represents a coordinate in the global grid.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Add for GridCoord {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

/// Represents a specific block's location in the global grid.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

struct SparseGrid<T, const BSX_LOG2: u32> {
    _marker: PhantomData<T>,

    // Dimensions of the grid in BLOCKS.
    blocks_x: usize,
    blocks_y: usize,
    blocks_z: usize,

    // If block is empty, None.
    // Length is blocks_x * blocks_y * blocks_z
    blockmap: Vec<Option<BlockId>>,

    // For PDE solvers needing "padding" for
    // possible out-of-bounds in derivative calculation.
    pub border_offset: GridCoord,
}

impl<T, const BSX_LOG2: u32> SparseGrid<T, BSX_LOG2> {
    const BLOCK_DIM: usize = 1 << BSX_LOG2;
    const VOXELS_PER_BLOCK: usize = Self::BLOCK_DIM * Self::BLOCK_DIM * Self::BLOCK_DIM;

    /// Converts a block & voxel coordinate pair into the respective global grid coordinate.
    #[inline(always)]
    pub fn get_grid_coords_from_3d(block: BlockCoord, voxel: VoxelCoord) -> GridCoord {
        GridCoord {
            x: (block.x << BSX_LOG2) + voxel.x,
            y: (block.y << BSX_LOG2) + voxel.y,
            z: (block.z << BSX_LOG2) + voxel.z,
        }
    }

    /// Converts a block & voxel id pair into the respective global grid coordinates.
    #[inline(always)]
    pub fn get_grid_coords_from_1d(&self, bid: BlockId, vid: VoxelId) -> GridCoord {
        // Convert the ids into coords...
        let block_coord = self.get_block_coords_by_id(bid);
        let voxel_coord = Self::get_voxel_coords_by_id(vid);

        Self::get_grid_coords_from_3d(block_coord, voxel_coord)
    }

    /// Find the coordinate location of a voxel within a block by it's id.
    #[inline(always)]
    fn get_voxel_coords_by_id(vid: VoxelId) -> VoxelCoord {
        // Bitmasking to decode a voxel Id (0-4095) back into it's coordinate (0-15, 0-15, 0-15)
        let mask = (1 << BSX_LOG2) - 1;
        VoxelCoord {
            x: (vid.0 as i32) & mask,
            y: ((vid.0 as i32) >> BSX_LOG2) & mask,
            z: (vid.0 as i32) >> (BSX_LOG2 * 2),
        }
    }

    /// Converts a flat block id into a block coordinate.
    #[inline(always)]
    pub fn get_block_coords_by_id(&self, bid: BlockId) -> BlockCoord {
        // self.blocks_x and self.blocks_y are the dimensions of the grid IN BLOCKS.
        // So like if the domain is 1024 voxels wide, and blocks are 16^3, blocks_x = 64.

        let index = bid.0;
        let bx = self.blocks_x;
        let bxy = self.blocks_x * self.blocks_y;

        BlockCoord {
            x: (index % bx) as i32,
            y: ((index / bx) % self.blocks_y) as i32,
            z: (index / bxy) as i32,
        }
    }

    /// Returns the coordinate of the block that a given global grid coordinate is in.
    #[inline(always)]
    fn get_block_coords(&self, coord: GridCoord) -> BlockCoord {
        BlockCoord {
            x: coord.x >> BSX_LOG2,
            y: coord.y >> BSX_LOG2,
            z: coord.z >> BSX_LOG2,
        }
    }
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

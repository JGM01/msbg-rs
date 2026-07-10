use std::{marker::PhantomData, ops::Add};

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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

struct SparseGrid<T, const BSX_LOG2: u32> {
    _marker: PhantomData<T>,
}

impl<T, const BSX_LOG2: u32> SparseGrid<T, BSX_LOG2> {
    const BLOCK_DIM: usize = 1 << BSX_LOG2;
    const VOXELS_PER_BLOCK: usize = Self::BLOCK_DIM * Self::BLOCK_DIM * Self::BLOCK_DIM;

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

use crate::sparse_grid::SparseGrid;

pub trait Interpolate {
    /// Performs standard trilinear interpolation at the given global coordinates.
    fn interpolate_linear(&mut self, x: f32, y: f32, z: f32) -> f32;
}

impl<const BSX: usize, const N: usize> Interpolate for SparseGrid<f32, BSX, N> {
    #[inline]
    fn interpolate_linear(&mut self, x: f32, y: f32, z: f32) -> f32 {
        // Calculate base integer coordinates
        let ix = x.floor();
        let iy = y.floor();
        let iz = z.floor();

        // Calculate fractional interpolation weights
        let u = x - ix;
        let v = y - iy;
        let w = z - iz;

        let ix = ix as usize;
        let iy = iy as usize;
        let iz = iz as usize;

        // Fetch 8 corners.
        let c000 = self.get_voxel(ix, iy, iz);
        let c100 = self.get_voxel(ix + 1, iy, iz);
        let c010 = self.get_voxel(ix, iy + 1, iz);
        let c110 = self.get_voxel(ix + 1, iy + 1, iz);
        let c001 = self.get_voxel(ix, iy, iz + 1);
        let c101 = self.get_voxel(ix + 1, iy, iz + 1);
        let c011 = self.get_voxel(ix, iy + 1, iz + 1);
        let c111 = self.get_voxel(ix + 1, iy + 1, iz + 1);

        // Interpolate along X
        let c00 = c000 * (1.0 - u) + c100 * u;
        let c10 = c010 * (1.0 - u) + c110 * u;
        let c01 = c001 * (1.0 - u) + c101 * u;
        let c11 = c011 * (1.0 - u) + c111 * u;

        // Interpolate along Y
        let c0 = c00 * (1.0 - v) + c10 * v;
        let c1 = c01 * (1.0 - v) + c11 * v;

        // Interpolate along Z
        c0 * (1.0 - w) + c1 * w
    }
}

#[cfg(test)]
mod interpolate_tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn setup_grid(sx: usize, sy: usize, sz: usize) -> SparseGrid<f32, BSX, N> {
        let pool = Arc::new(BlockPool::new(32, 64));
        SparseGrid::new("interp_test".to_string(), sx, sy, sz, 0.0, 1.0, pool)
    }

    /// Reference scalar trilinear (identical math, no SIMD / grid indirection)
    #[allow(clippy::too_many_arguments)]
    fn scalar_trilinear(
        c000: f32,
        c100: f32,
        c010: f32,
        c110: f32,
        c001: f32,
        c101: f32,
        c011: f32,
        c111: f32,
        u: f32,
        v: f32,
        w: f32,
    ) -> f32 {
        let c00 = c000 * (1.0 - u) + c100 * u;
        let c10 = c010 * (1.0 - u) + c110 * u;
        let c01 = c001 * (1.0 - u) + c101 * u;
        let c11 = c011 * (1.0 - u) + c111 * u;
        let c0 = c00 * (1.0 - v) + c10 * v;
        let c1 = c01 * (1.0 - v) + c11 * v;
        c0 * (1.0 - w) + c1 * w
    }

    #[test]
    fn test_interp_01_exact_integer_coordinates() {
        let mut grid = setup_grid(32, 32, 32);
        grid.set_voxel(5, 7, 9, 42.0);

        // Query exactly on a voxel centre → must return the stored value
        let v = grid.interpolate_linear(5.0, 7.0, 9.0);
        assert!((v - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_interp_02_midpoint_same_block() {
        let mut grid = setup_grid(32, 32, 32);

        // Fill a 2×2×2 neighbourhood inside one block
        grid.set_voxel(4, 4, 4, 1.0);
        grid.set_voxel(5, 4, 4, 2.0);
        grid.set_voxel(4, 5, 4, 3.0);
        grid.set_voxel(5, 5, 4, 4.0);
        grid.set_voxel(4, 4, 5, 5.0);
        grid.set_voxel(5, 4, 5, 6.0);
        grid.set_voxel(4, 5, 5, 7.0);
        grid.set_voxel(5, 5, 5, 8.0);

        let expected = scalar_trilinear(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 0.5, 0.5);
        let v = grid.interpolate_linear(4.5, 4.5, 4.5);
        assert!(
            (v - expected).abs() < 1e-5,
            "got {} expected {}",
            v,
            expected
        );
    }

    #[test]
    fn test_interp_03_cross_block_boundary() {
        // Point that straddles two blocks in X (block 0 and block 1)
        let mut grid = setup_grid(32, 16, 16);

        grid.set_voxel(15, 0, 0, 10.0); // last voxel of block 0
        grid.set_voxel(16, 0, 0, 20.0); // first voxel of block 1

        // Mid-point between them
        let v = grid.interpolate_linear(15.5, 0.0, 0.0);
        assert!((v - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_interp_04_all_corners_unit_cube() {
        let mut grid = setup_grid(16, 16, 16);

        // Identity field: value = x + 2y + 4z  (easy closed form)
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    grid.set_voxel(x, y, z, (x + 2 * y + 4 * z) as f32);
                }
            }
        }

        // Sample at (0.3, 0.4, 0.7)
        let expected = 0.3 + 2.0 * 0.4 + 4.0 * 0.7; // 0.3 + 0.8 + 2.8 = 3.9
        let v = grid.interpolate_linear(0.3, 0.4, 0.7);
        assert!(
            (v - expected).abs() < 1e-5,
            "got {} expected {}",
            v,
            expected
        );
    }

    #[test]
    fn test_interp_05_empty_region_returns_empty_value() {
        let mut grid = setup_grid(32, 32, 32);
        // Never written → empty_value = 0.0
        let v = grid.interpolate_linear(10.5, 10.5, 10.5);
        assert!((v - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_interp_06_full_dummy_block() {
        let mut grid = setup_grid(32, 32, 32);
        let bid = grid.get_block_id(20, 20, 20);
        grid.blockmap[bid] = Some(grid.full_block); // full_value = 1.0

        // Any point inside that block must interpolate to 1.0
        let v = grid.interpolate_linear(20.3, 21.7, 22.1);
        assert!((v - 1.0).abs() < 1e-6);
    }
}

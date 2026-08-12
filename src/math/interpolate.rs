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

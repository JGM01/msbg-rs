//! The staggered MAC grid: three face-velocity and three face-mass scalar
//! channels over [`SparseGrid<f32>`].
//!
//! Convention: face-component `a`'s voxel `(i,j,k)` stores the face value of the
//! **right** face of cell `(i,j,k)`, located at the cell center
//! `(i+½, j+½, k+½) + ½·e_a`. So the face-a grid is the cell-centered grid
//! shifted by half a cell along axis `a`; the half-cell shift lives in the
//! sampling/staging coordinates, not the storage, keeping every grid cell-sized.
//! The right-most faces (`i = sx`, etc.) lie on the domain boundary and are the
//! caller's boundary condition (treated as 0 by [`MacGrid::divergence`]).
//!
//! Six scalar `f32` channels (not three `Vec3`) so the P2G splat and the
//! step-14 Poisson stencil both stream pure scalar lanes — the C++ `Vec3Float`
//! AoS layout forces 3-wide loads that block its own SIMD.

use std::sync::Arc;

use rayon::prelude::*;
use std::simd::cmp::SimdPartialOrd;
use std::simd::{Select, Simd};

use crate::blockpool::BlockPool;
use crate::sparse_grid::SparseGrid;

/// The three MAC face directions.
pub const X: usize = 0;
pub const Y: usize = 1;
pub const Z: usize = 2;

/// Staggered MAC grid: `velocity[a]` / `mass[a]` are the face-a fields.
pub struct MacGrid<const BSX: usize, const N: usize> {
    velocity: [SparseGrid<f32, BSX, N>; 3],
    mass: [SparseGrid<f32, BSX, N>; 3],
}

impl<const BSX: usize, const N: usize> MacGrid<BSX, N> {
    pub fn new(sx: usize, sy: usize, sz: usize, pool: Arc<BlockPool<f32, BSX, N>>) -> Self {
        // Velocity/mass have no "full" sentinel in single-phase FLIP; both
        // dummies read 0 so empty regions contribute zero mass/velocity.
        let mk = |name: &str| SparseGrid::new(name.to_string(), sx, sy, sz, 0.0, 0.0, pool.clone());
        Self {
            velocity: [mk("vel_u"), mk("vel_v"), mk("vel_w")],
            mass: [mk("mass_u"), mk("mass_v"), mk("mass_w")],
        }
    }

    #[inline(always)]
    pub fn sx(&self) -> usize {
        self.velocity[X].sx
    }
    #[inline(always)]
    pub fn sy(&self) -> usize {
        self.velocity[X].sy
    }
    #[inline(always)]
    pub fn sz(&self) -> usize {
        self.velocity[X].sz
    }
    #[inline(always)]
    pub fn nx(&self) -> usize {
        self.velocity[X].nx
    }
    #[inline(always)]
    pub fn ny(&self) -> usize {
        self.velocity[X].ny
    }
    #[inline(always)]
    pub fn nxy(&self) -> usize {
        self.velocity[X].nxy
    }

    #[inline(always)]
    pub fn velocity(&self, a: usize) -> &SparseGrid<f32, BSX, N> {
        &self.velocity[a]
    }

    #[inline(always)]
    pub fn velocity_mut(&mut self, a: usize) -> &mut SparseGrid<f32, BSX, N> {
        &mut self.velocity[a]
    }

    #[inline(always)]
    pub fn mass(&self, a: usize) -> &SparseGrid<f32, BSX, N> {
        &self.mass[a]
    }

    #[inline(always)]
    pub fn mass_mut(&mut self, a: usize) -> &mut SparseGrid<f32, BSX, N> {
        &mut self.mass[a]
    }

    /// Materialize (once) and zero every channel over `active` blocks. The P2G
    /// splat `+=` into these, so they must start zeroed (unlike the step-9
    /// density splat, which starts at `full()` and takes the `min`).
    pub fn zero_blocks(&mut self, active: &[usize]) {
        for a in 0..3 {
            self.velocity[a].ensure_blocks_parallel(active);
            self.mass[a].ensure_blocks_parallel(active);
            self.velocity[a].fill_blocks_parallel(active, 0.0);
            self.mass[a].fill_blocks_parallel(active, 0.0);
        }
    }

    /// Convert the accumulated momentum (in `velocity`) into face velocity
    /// `ũ* = P/M`, guarded against zero-mass faces.
    pub fn normalize_velocity(&mut self, active: &[usize]) {
        for a in 0..3 {
            let vel = &self.velocity[a];
            let mass = &self.mass[a];
            active.par_iter().for_each(|&bid| {
                let vp = vel.value_block_ptr_mut(bid);
                let mp = mass.value_block_ptr(bid);
                if vp.is_null() {
                    return;
                }
                let v = unsafe { std::slice::from_raw_parts_mut(vp, N) };
                let m = unsafe { std::slice::from_raw_parts(mp, N) };
                let mut i = 0;
                while i + 8 <= N {
                    let vv = Simd::<f32, 8>::from_slice(&v[i..i + 8]);
                    let mm = Simd::<f32, 8>::from_slice(&m[i..i + 8]);
                    let nz = mm.simd_gt(Simd::splat(0.0));
                    let safe = nz.select(mm, Simd::splat(1.0));
                    (nz.select(vv / safe, Simd::splat(0.0))).copy_to_slice(&mut v[i..i + 8]);
                    i += 8;
                }
                for j in i..N {
                    v[j] = if m[j] > 0.0 { v[j] / m[j] } else { 0.0 };
                }
            });
        }
    }

    /// Cell-centered divergence `∇·u* = Σ_a (u_a(right) − u_a(left))`. The
    /// leftmost faces (`i = 0` per axis) read 0 (the domain boundary); boundary
    /// conditions are a step-18 concern.
    pub fn divergence(
        &self,
        dst: &mut SparseGrid<f32, BSX, N>,
        active: &[usize],
    ) {
        dst.ensure_blocks_parallel(active);
        let (u, v, w) = (&self.velocity[X], &self.velocity[Y], &self.velocity[Z]);
        let sx = self.sx();
        let sy = self.sy();
        let sz = self.sz();
        active.par_iter().for_each(|&bid| {
            let dp = dst.value_block_ptr_mut(bid);
            if dp.is_null() {
                return;
            }
            let (bx, by, bz) = (bid % self.nx(), (bid / self.nx()) % self.ny(), bid / self.nxy());
            let (gx0, gy0, gz0) = (bx * BSX, by * BSX, bz * BSX);
            let (gx1, gy1, gz1) = ((gx0 + BSX).min(sx), (gy0 + BSX).min(sy), (gz0 + BSX).min(sz));
            let mut vid = 0usize;
            for z in gz0..gz1 {
                for y in gy0..gy1 {
                    for x in gx0..gx1 {
                        let ux1 = u.get_voxel(x, y, z);
                        let ux0 = if x == 0 { 0.0 } else { u.get_voxel(x - 1, y, z) };
                        let vy1 = v.get_voxel(x, y, z);
                        let vy0 = if y == 0 { 0.0 } else { v.get_voxel(x, y - 1, z) };
                        let wz1 = w.get_voxel(x, y, z);
                        let wz0 = if z == 0 { 0.0 } else { w.get_voxel(x, y, z - 1) };
                        unsafe { *dp.add(vid) = (ux1 - ux0) + (vy1 - vy0) + (wz1 - wz0) };
                        vid += 1;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn make_mac(s: usize) -> MacGrid<BSX, N> {
        let pool = Arc::new(BlockPool::<f32, BSX, N>::new(8, 64));
        MacGrid::new(s, s, s, pool)
    }

    fn active_all(s: usize) -> Vec<usize> {
        let dims = crate::particles::GridDims::new(s, s, s, BSX);
        (0..dims.n_blocks()).collect()
    }

    // Uniform field: divergence is zero in the interior (boundary faces read 0).
    #[test]
    fn div_10_uniform_and_linear() {
        let s = 32;
        let mut mac = make_mac(s);
        let active = active_all(s);
        mac.zero_blocks(&active);
        // Linear field: u(i)=i+1, v(j)=j+1, w(k)=k+1 -> div = 3 everywhere.
        for k in 0..s {
            for j in 0..s {
                for i in 0..s {
                    mac.velocity_mut(X).set_voxel(i, j, k, (i + 1) as f32);
                    mac.velocity_mut(Y).set_voxel(i, j, k, (j + 1) as f32);
                    mac.velocity_mut(Z).set_voxel(i, j, k, (k + 1) as f32);
                }
            }
        }
        let mut div = crate::sparse_grid::SparseGrid::new(
            "div".into(),
            s,
            s,
            s,
            0.0,
            0.0,
            mac.velocity(X).block_pool.clone(),
        );
        mac.divergence(&mut div, &active);
        for k in 0..s {
            for j in 0..s {
                for i in 0..s {
                    assert_eq!(div.get_voxel(i, j, k), 3.0, "div ({i},{j},{k})");
                }
            }
        }

        // Uniform field u=v=w=5 -> interior div 0.
        for k in 0..s {
            for j in 0..s {
                for i in 0..s {
                    mac.velocity_mut(X).set_voxel(i, j, k, 5.0);
                    mac.velocity_mut(Y).set_voxel(i, j, k, 5.0);
                    mac.velocity_mut(Z).set_voxel(i, j, k, 5.0);
                }
            }
        }
        mac.divergence(&mut div, &active);
        for k in 1..s {
            for j in 1..s {
                for i in 1..s {
                    assert_eq!(div.get_voxel(i, j, k), 0.0, "uniform div ({i},{j},{k})");
                }
            }
        }
    }
}

//! Grid-to-particles: staggered MAC velocity sampling + the FLIP/PIC blend.

use crate::channel::Vec3;
use crate::math::{BoundaryCondition, GridAlignment, Interpolation, Sampler};

use super::mac::{MacGrid, X, Y, Z};

/// Samples a staggered MAC velocity field at particle positions. Each face
/// component is a scalar trilinear/cubic sample at its half-cell offset:
/// face-a voxel `i` sits at `cell + ½·e_a`, so the query is `p − ½·e_a`.
pub struct MacSampler<'a, const BSX: usize, const N: usize> {
    samplers: [Sampler<'a, f32, BSX, N>; 3],
}

impl<'a, const BSX: usize, const N: usize> MacSampler<'a, BSX, N> {
    pub fn new(mac: &'a MacGrid<BSX, N>, bc: BoundaryCondition) -> Self {
        let samplers = [
            Sampler::new(mac.velocity(X), GridAlignment::CellCentered, bc),
            Sampler::new(mac.velocity(Y), GridAlignment::CellCentered, bc),
            Sampler::new(mac.velocity(Z), GridAlignment::CellCentered, bc),
        ];
        Self { samplers }
    }

    #[inline]
    pub fn sample<const IP: Interpolation>(&self, p: Vec3) -> Vec3 {
        Vec3::new(
            self.samplers[X].sample::<IP>(Vec3::new(p.x() - 0.5, p.y(), p.z())),
            self.samplers[Y].sample::<IP>(Vec3::new(p.x(), p.y() - 0.5, p.z())),
            self.samplers[Z].sample::<IP>(Vec3::new(p.x(), p.y(), p.z() - 0.5)),
        )
    }
}

/// FLIP/PIC blend (Eq. 12): the standard Bridson form
/// `u_new = α·(u_old + Δu) + (1−α)·u`, with `Δu = u − ũ*`. `u` is the
/// post-projection face velocity, `u_tilde` the pre-projection one (both
/// already sampled at the particle); `α = 0` is pure PIC (`u`), `α = 1` is pure
/// FLIP (`u_old + Δu`). Linear interpolation makes `I(u) − I(ũ*) = I(Δu)`, so
/// the two fields are sampled separately here and combined.
#[inline]
pub fn g2p_blend(u_old: Vec3, u: Vec3, u_tilde: Vec3, alpha: f32) -> Vec3 {
    let delta = u - u_tilde;
    (u_old + delta) * alpha + u * (1.0 - alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use crate::fluid::mac::{MacGrid, X, Y, Z};
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn make_mac(s: usize) -> MacGrid<BSX, N> {
        let pool = Arc::new(BlockPool::<f32, BSX, N>::new(8, 64));
        MacGrid::new(s, s, s, pool)
    }

    // FLIP/PIC blend endpoints: alpha=1 is pure FLIP (u_old + du), alpha=0 is
    // pure PIC (u).
    #[test]
    fn g2p_08_alpha_endpoints() {
        let u_old = Vec3::new(1.0, 2.0, 3.0);
        let u = Vec3::new(4.0, 5.0, 6.0);
        let u_tilde = Vec3::new(4.5, 4.5, 4.5);
        let delta = u - u_tilde;
        // alpha = 0 -> PIC: u_new = u.
        let pic = g2p_blend(u_old, u, u_tilde, 0.0);
        assert_eq!(pic, u);
        // alpha = 1 -> FLIP: u_new = u_old + (u - u_tilde).
        let flip = g2p_blend(u_old, u, u_tilde, 1.0);
        assert_eq!(flip, u_old + delta);
    }

    // Half-cell face offset: sampling at a face center returns that face value.
    #[test]
    fn g2p_11_face_center_exact() {
        let mut mac = make_mac(32);
        // u(i,j,k) = i+1 (the face's world-x), v = j+1, w = k+1.
        let active: Vec<usize> = (0..8).collect();
        mac.zero_blocks(&active);
        for a in 0..3 {
            for k in 0..32 {
                for j in 0..32 {
                    for i in 0..32 {
                        let val = match a {
                            X => (i + 1) as f32,
                            Y => (j + 1) as f32,
                            _ => (k + 1) as f32,
                        };
                        mac.velocity_mut(a).set_voxel(i, j, k, val);
                    }
                }
            }
        }
        let sampler = MacSampler::new(&mac, BoundaryCondition::Clamp);
        // p = (5, 8.5, 8.5) is the x-face center (i=4, j=8, k=8): got.x() must be
        // exactly u[4,8,8] = 5. In y/z the point is a cell center halfway between
        // faces, so those components interpolate to 8.5.
        let got = sampler.sample::<{ Interpolation::Linear }>(Vec3::new(5.0, 8.5, 8.5));
        assert_eq!(got.x(), 5.0);
        assert_eq!(got.y(), 8.5);
        assert_eq!(got.z(), 8.5);
    }

    // Out-of-domain sampling clamps (no panic, finite result).
    #[test]
    fn g2p_09_out_of_domain_clamp() {
        let mut mac = make_mac(32);
        let active: Vec<usize> = (0..8).collect();
        mac.zero_blocks(&active);
        for k in 0..32 {
            for j in 0..32 {
                for i in 0..32 {
                    mac.velocity_mut(X).set_voxel(i, j, k, 1.0);
                    mac.velocity_mut(Y).set_voxel(i, j, k, 2.0);
                    mac.velocity_mut(Z).set_voxel(i, j, k, 3.0);
                }
            }
        }
        let sampler = MacSampler::new(&mac, BoundaryCondition::Clamp);
        let got = sampler.sample::<{ Interpolation::Linear }>(Vec3::new(-5.0, -5.0, -5.0));
        assert!(got.x().is_finite() && got.y().is_finite() && got.z().is_finite());
        let got2 = sampler.sample::<{ Interpolation::Linear }>(Vec3::new(100.0, 100.0, 100.0));
        assert!(got2.x().is_finite());
    }
}

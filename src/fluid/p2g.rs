//! Particles-to-grid: the staged 8-color sum-splat of mass + momentum to MAC
//! faces.
//!
//! Mirrors the step-9 `splat` architecture — a thread-local `(BSX + 2·margin)³`
//! staging buffer whose accumulation stays in cache, then a 3×3×3 commit that
//! writes each touched voxel exactly once per contributing block — but with
//! three differences that make it the FLIP transfer rather than a min-SDF:
//!
//! * **sum** reduction (`+=`, mass and momentum accumulate) instead of `min`;
//! * a **cubic-falloff** kernel `w = max(1 − (d/r_p)², 0)³` (Eq. 6) instead of
//!   the `distSq` min-SDF, evaluated at the **face center** `cell + ½·e_a`; and
//! * **six** channels (3 face masses + 3 face momenta) staged two at a time per
//!   direction, sharing one weight evaluation.
//!
//! The face center is `½` past the cell center along its axis, so for direction
//! `a` the face distance is `d_face² = d_cell² + d_cell[a] + ¼`, and the face
//! footprint along axis `a` shifts by one extra half cell — both expressed
//! directly in the staging loop with no extra buffers.

use std::cell::UnsafeCell;
use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::SimdFloat;
use std::simd::{Select, Simd};

use rayon::prelude::*;

use crate::particles::GridDims;

use super::mac::{MacGrid, X, Y, Z};
use super::{BucketedParticles, Particles};

/// Precondition for the 8-color race-freedom: a particle's face footprint along
/// one axis (`±(r_p + 0.5)`) must be narrower than half the gap between
/// same-color center blocks (`BSX`).
#[inline(always)]
fn check_r_p(r_p: f32, margin: usize) {
    debug_assert!(r_p <= margin as f32, "r_p ({r_p}) exceeds the staging margin ({margin})");
    debug_assert!(
        r_p + 0.5 < margin as f32 * 2.0,
        "r_p must be < BSX for the 8-color scheme to be race-free"
    );
}

/// Two staging buffers (mass + momentum) for one face direction.
struct StageSlot<const MSX: usize> {
    mass: Box<[f32]>,
    mom: Box<[f32]>,
}

struct StagePool<const MSX: usize> {
    slots: Vec<UnsafeCell<StageSlot<MSX>>>,
}

// Each slot is exclusively accessed by a single physical Rayon thread.
unsafe impl<const MSX: usize> Sync for StagePool<MSX> {}

impl<const MSX: usize> StagePool<MSX> {
    fn new(num_threads: usize) -> Self {
        let slots = (0..=num_threads)
            .map(|_| {
                UnsafeCell::new(StageSlot {
                    mass: vec![0.0f32; MSX * MSX * MSX].into_boxed_slice(),
                    mom: vec![0.0f32; MSX * MSX * MSX].into_boxed_slice(),
                })
            })
            .collect();
        Self { slots }
    }

    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut StageSlot<MSX> {
        let idx = rayon::current_thread_index().unwrap_or(self.slots.len() - 1);
        debug_assert!(idx < self.slots.len());
        unsafe { &mut *self.slots[idx].get() }
    }
}

/// SIMD sum-commit: `dst[i] += src[i]` for `W` lanes, skipping entirely when
/// every source lane is zero (the block was not touched by this direction).
#[inline(always)]
unsafe fn commit_sum<const W: usize>(src: *const f32, dst: *mut f32) {
    let s = Simd::<f32, W>::from_slice(unsafe { std::slice::from_raw_parts(src, W) });
    if s.simd_eq(Simd::splat(0.0)).all() {
        return;
    }
    let d = Simd::<f32, W>::from_slice(unsafe { std::slice::from_raw_parts(dst, W) });
    unsafe { (d + s).copy_to_slice(std::slice::from_raw_parts_mut(dst, W)) };
}

/// Staging region range along one axis (the step-9 `region_range`).
#[inline(always)]
fn region_range(d: i32, margin: i32, bsx: i32) -> (i32, i32) {
    match d {
        -1 => (0, margin),
        0 => (margin, margin + bsx),
        _ => (margin + bsx, 2 * margin + bsx),
    }
}

/// Union of every particle's face-footprint blocks (a one-cell superset of the
/// per-direction face footprints, so the commit's 27-neighbor writes always
/// land in a materialized block).
pub fn active_blocks(positions: &[[f32; 3]], dims: &GridDims, r_p: f32) -> Vec<usize> {
    use crate::blockmap::BlockSet;
    let r_scan_bsx = (r_p + 1.0) / dims.bsx as f32;
    let set: BlockSet = positions
        .par_chunks(4096)
        .fold(BlockSet::new, |mut local, chunk| {
            for p in chunk {
                let (x1, x2) = crate::particles::footprint_axis(p[0], r_scan_bsx, dims.bsx, dims.nx);
                let (y1, y2) = crate::particles::footprint_axis(p[1], r_scan_bsx, dims.bsx, dims.ny);
                let (z1, z2) = crate::particles::footprint_axis(p[2], r_scan_bsx, dims.bsx, dims.nz);
                for bz in z1..=z2 {
                    for by in y1..=y2 {
                        for bx in x1..=x2 {
                            local.insert(
                                (bx as usize) + (by as usize) * dims.nx + (bz as usize) * dims.nxy,
                                (),
                            );
                        }
                    }
                }
            }
            local
        })
        .reduce(BlockSet::new, |mut a, b| {
            for (k, _) in b.iter() {
                a.insert(k, ());
            }
            a
        });
    let mut active: Vec<usize> = set.iter().map(|(k, _)| k).collect();
    active.par_sort_unstable();
    active
}

/// Full P2G transfer: bucket, zero the active blocks, splat mass + momentum to
/// faces, then normalize `ũ* = P/M`. `MSX` is the staging extent, a multiple of
/// 8 at least `BSX + 2·ceil(r_p)`.
pub fn particles_to_grid<const BSX: usize, const N: usize, const MSX: usize>(
    mac: &mut MacGrid<BSX, N>,
    particles: &Particles,
    r_p: f32,
) {
    particles.assert_consistent();
    let dims = GridDims::new(mac.sx(), mac.sy(), mac.sz(), BSX);
    let bsx_log2 = BSX.trailing_zeros();
    let bids: Vec<usize> = particles
        .positions
        .iter()
        .map(|p| {
            (p[0] as usize >> bsx_log2)
                + ((p[1] as usize >> bsx_log2) * dims.nx)
                + ((p[2] as usize >> bsx_log2) * dims.nxy)
        })
        .collect();
    let bucketed = super::bucket_particles(particles, &bids);
    let active = active_blocks(&particles.positions, &dims, r_p);

    mac.zero_blocks(&active);
    splat::<BSX, N, MSX>(mac, &bucketed, r_p);
    mac.normalize_velocity(&active);
}

/// Splat every bucketed particle's mass + momentum into the six face channels,
/// via 8 color passes per direction. `MSX` must be a multiple of 8 (aligned
/// SIMD rows) at least `BSX + 2·ceil(r_p)`.
pub fn splat<const BSX: usize, const N: usize, const MSX: usize>(
    mac: &MacGrid<BSX, N>,
    bucketed: &BucketedParticles,
    r_p: f32,
) {
    debug_assert_eq!(MSX % 8, 0, "MSX must be a multiple of 8 for SIMD8 staging");
    let margin = r_p.ceil() as usize;
    debug_assert!(
        MSX >= BSX + 2 * margin,
        "MSX ({MSX}) too small for r_p {r_p} (needs {})",
        BSX + 2 * margin
    );
    check_r_p(r_p, margin);
    let inv_r2 = 1.0 / (r_p * r_p);

    let dims = GridDims::new(mac.sx(), mac.sy(), mac.sz(), BSX);

    let mut by_color: [Vec<(usize, usize)>; 8] = std::array::from_fn(|_| Vec::new());
    for (i, &bid) in bucketed.particle_blocks.iter().enumerate() {
        let (bx, by, bz) = dims.coords(bid);
        let color = (bx & 1) | ((by & 1) << 1) | ((bz & 1) << 2);
        by_color[color].push((i, bid));
    }

    let pool = StagePool::<MSX>::new(rayon::current_num_threads());

    for a in [X, Y, Z] {
        for bucket in &by_color {
            bucket.par_iter().for_each(|&(i, bid)| {
                let start = bucketed.starts[i];
                let end = bucketed.starts[i + 1];
                let slot = unsafe { pool.get_mut() };
                stage_and_commit::<BSX, N, MSX>(
                    mac,
                    a,
                    &dims,
                    slot,
                    bid,
                    &bucketed.positions[start..end],
                    &bucketed.velocities[start..end],
                    &bucketed.mass[start..end],
                    r_p,
                    inv_r2,
                    margin,
                );
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn stage_and_commit<const BSX: usize, const N: usize, const MSX: usize>(
    mac: &MacGrid<BSX, N>,
    a: usize,
    dims: &GridDims,
    slot: &mut StageSlot<MSX>,
    bid: usize,
    pts: &[[f32; 3]],
    vels: &[[f32; 3]],
    mass: &[f32],
    r_p: f32,
    inv_r2: f32,
    margin: usize,
) {
    let (bx, by, bz) = dims.coords(bid);
    let b0x = (bx * BSX) as i32;
    let b0y = (by * BSX) as i32;
    let b0z = (bz * BSX) as i32;
    let m = margin as i32;

    let staging = &mut *slot;
    staging.mass.fill(0.0);
    staging.mom.fill(0.0);

    let x0s = b0x - m;
    let y0s = b0y - m;
    let z0s = b0z - m;
    let window_hi = (BSX as i32 + m) - 1;
    let sm_ptr = staging.mass.as_mut_ptr();
    let sv_ptr = staging.mom.as_mut_ptr();
    let lanes_f32 = Simd::<f32, 8>::from_array(std::array::from_fn(|i| i as f32));
    let lanes_i32 = Simd::<i32, 8>::from_array(std::array::from_fn(|i| i as i32));

    // Face offsets: axis `a` shifts by a full cell (½ cell-center + ½ face),
    // the other two axes keep the cell-center ½.
    let (ofx, ofy, ofz) = match a {
        X => (1.0f32, 0.5, 0.5),
        Y => (0.5, 1.0, 0.5),
        _ => (0.5, 0.5, 1.0),
    };

    for (pi, &p) in pts.iter().enumerate() {
        let m_p = mass[pi];
        let u_a = vels[pi][a];
        let scale_mass = m_p;
        let scale_mom = m_p * u_a;

        let px0 = p[0] - 0.5;
        let py0 = p[1] - 0.5;
        let pz0 = p[2] - 0.5;

        // Face footprint global voxel range, clipped to the staging window.
        let ix1 = (((p[0] - r_p - ofx).ceil() as i32).max(b0x - m)).min(b0x + window_hi);
        let ix2 = (((p[0] + r_p - ofx).floor() as i32).max(b0x - m)).min(b0x + window_hi);
        let iy1 = (((p[1] - r_p - ofy).ceil() as i32).max(b0y - m)).min(b0y + window_hi);
        let iy2 = (((p[1] + r_p - ofy).floor() as i32).max(b0y - m)).min(b0y + window_hi);
        let iz1 = (((p[2] - r_p - ofz).ceil() as i32).max(b0z - m)).min(b0z + window_hi);
        let iz2 = (((p[2] + r_p - ofz).floor() as i32).max(b0z - m)).min(b0z + window_hi);

        for iz in iz1..=iz2 {
            let dz = iz as f32 - pz0;
            let dz2 = dz * dz;
            let sz = (iz - z0s) as usize;
            for iy in iy1..=iy2 {
                let dy = iy as f32 - py0;
                let dyz = dz2 + dy * dy;
                let sy = (iy - y0s) as usize;
                let row = (sz * MSX + sy) * MSX;
                let o1 = (ix1 - x0s) as usize;
                let o2 = (ix2 - x0s) as usize;
                let mut c = o1 & !7;
                while c <= o2 {
                    let cx = (c as i32 + x0s) as f32;
                    let dx = (Simd::splat(cx) + lanes_f32) - Simd::splat(px0);
                    let dist_sq = dx * dx + Simd::splat(dyz);
                    let llo = o1.saturating_sub(c).min(15);
                    let lhi = o2.saturating_sub(c).min(15);
                    let in_run = lanes_i32.simd_ge(Simd::splat(llo as i32))
                        & lanes_i32.simd_le(Simd::splat(lhi as i32));

                    // Face distance: `d_face² = d_cell² + d_cell[a] + ¼`.
                    //
                    // Derivation: the face sits at `cell_center + ½·e_a` (the
                    // *right* face), so the face distance is `d_cell + ½·e_a` and
                    // `‖d_cell + ½·e_a‖² = ‖d_cell‖² + d_cell[a] + ¼` — the linear
                    // term `+d_cell[a]` has that sign because the offset is `+½`.
                    // (Had we stored the *left* face, `cell_center − ½·e_a`, the
                    // identity would flip to `− d_cell[a] + ¼`.)
                    let d_a = match a {
                        X => dx,
                        Y => Simd::splat(dy),
                        _ => Simd::splat(dz),
                    };
                    let dist_face2 = dist_sq + d_a + Simd::splat(0.25);
                    // Cubic-falloff kernel (Eq. 6).
                    let w = (Simd::splat(1.0) - dist_face2 * Simd::splat(inv_r2)).simd_max(Simd::splat(0.0));
                    let w = w * w * w;
                    let add_mass = in_run.select(w * Simd::splat(scale_mass), Simd::splat(0.0));
                    let add_mom = in_run.select(w * Simd::splat(scale_mom), Simd::splat(0.0));

                    unsafe {
                        let off = row + c;
                        let old_m =
                            Simd::<f32, 8>::from_slice(std::slice::from_raw_parts(sm_ptr.add(off), 8));
                        (old_m + add_mass).copy_to_slice(std::slice::from_raw_parts_mut(sm_ptr.add(off), 8));
                        let old_v =
                            Simd::<f32, 8>::from_slice(std::slice::from_raw_parts(sv_ptr.add(off), 8));
                        (old_v + add_mom).copy_to_slice(std::slice::from_raw_parts_mut(sv_ptr.add(off), 8));
                    }
                    c += 8;
                }
            }
        }
    }

    // ---- Commit: write each touched staging voxel to its real block -------
    let mass_grid = mac.mass(a);
    let vel_grid = mac.velocity(a);
    let bsx_log2 = BSX.trailing_zeros();
    let bsx_mask = BSX as i32 - 1;
    let bsx_i = BSX as i32;
    let margin_i = m;
    let mut neighbor = [0usize; 27];
    for dz in -1..=1i32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let bx = (bx as i32 + dx).clamp(0, dims.nx as i32 - 1);
                let by = (by as i32 + dy).clamp(0, dims.ny as i32 - 1);
                let bz = (bz as i32 + dz).clamp(0, dims.nz as i32 - 1);
                let idx = (((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)) as usize;
                neighbor[idx] = (bx as usize) + (by as usize) * dims.nx + (bz as usize) * dims.nxy;
            }
        }
    }

    for dz in -1..=1i32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let (xlo, xhi) = region_range(dx, margin_i, bsx_i);
                let (ylo, yhi) = region_range(dy, margin_i, bsx_i);
                let (zlo, zhi) = region_range(dz, margin_i, bsx_i);
                let xlen = (xhi - xlo) as usize;

                let gx0 = b0x + xlo - m;
                let gy0 = b0y + ylo - m;
                let gz0 = b0z + zlo - m;
                let nbid = neighbor[(((dz + 1) * 3 + (dy + 1)) * 3 + (dx + 1)) as usize];
                if gx0 < 0 || gy0 < 0 || gz0 < 0 {
                    continue;
                }
                if gx0 + xlen as i32 > dims.sx as i32
                    || gy0 + (yhi - ylo) > dims.sy as i32
                    || gz0 + (zhi - zlo) > dims.sz as i32
                {
                    continue;
                }
                let m_data = mass_grid.value_block_ptr_mut(nbid);
                let v_data = vel_grid.value_block_ptr_mut(nbid);
                if m_data.is_null() || v_data.is_null() {
                    continue;
                }

                let lx0 = gx0 & bsx_mask;
                let ly0 = gy0 & bsx_mask;
                let lz0 = gz0 & bsx_mask;

                for sz in zlo..zhi {
                    let vz = (lz0 + (sz - zlo)) << (2 * bsx_log2);
                    for sy in ylo..yhi {
                        let vy = (ly0 + (sy - ylo)) << bsx_log2;
                        let s_row = (sz * MSX as i32 + sy) * MSX as i32 + xlo;
                        let d_row = (lx0 | vy | vz) as usize;
                        let s_ptr = unsafe { staging.mass.as_ptr().add(s_row as usize) };
                        let d_ptr = unsafe { m_data.add(d_row) };
                        let sv_ptr = unsafe { staging.mom.as_ptr().add(s_row as usize) };
                        let dv_ptr = unsafe { v_data.add(d_row) };
                        if dx == 0 {
                            unsafe {
                                commit_sum::<16>(s_ptr, d_ptr);
                                commit_sum::<16>(sv_ptr, dv_ptr);
                            }
                        } else {
                            unsafe {
                                commit_sum::<4>(s_ptr, d_ptr);
                                commit_sum::<4>(sv_ptr, dv_ptr);
                            }
                        }
                    }
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use crate::channel::Vec3;
    use crate::fluid::mac::{MacGrid, X, Y, Z};
    use crate::fluid::{g2p::MacSampler, Particles, KIND_LIQUID};
    use crate::math::{BoundaryCondition, Interpolation};
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;
    const MSX: usize = 24;

    fn make_mac(s: usize) -> MacGrid<BSX, N> {
        let pool = Arc::new(BlockPool::<f32, BSX, N>::new(8, 64));
        MacGrid::new(s, s, s, pool)
    }

    fn particles(ps: &[[f32; 3]], vs: &[[f32; 3]], mass: &[f32]) -> Particles {
        Particles {
            positions: ps.to_vec(),
            velocities: vs.to_vec(),
            kinds: vec![KIND_LIQUID; ps.len()],
            mass: mass.to_vec(),
        }
    }

    /// Naive scalar reference: the exact Eq. 6 splat, used as the equivalence
    /// oracle for the staged SIMD path.
    fn reference(mac: &mut MacGrid<BSX, N>, ps: &Particles, r_p: f32) {
        let dims = crate::particles::GridDims::new(mac.sx(), mac.sy(), mac.sz(), BSX);
        let active = active_blocks(&ps.positions, &dims, r_p);
        mac.zero_blocks(&active);
        let inv_r2 = 1.0 / (r_p * r_p);
        for pi in 0..ps.len() {
            let p = ps.positions[pi];
            let m_p = ps.mass[pi];
            let u = ps.velocities[pi];
            for a in 0..3 {
                let (ofx, ofy, ofz) = match a {
                    X => (1.0f32, 0.5, 0.5),
                    Y => (0.5, 1.0, 0.5),
                    _ => (0.5, 0.5, 1.0),
                };
                let ix1 = ((p[0] - r_p - ofx).ceil() as i32).max(0);
                let ix2 = ((p[0] + r_p - ofx).floor() as i32).min(mac.sx() as i32 - 1);
                let iy1 = ((p[1] - r_p - ofy).ceil() as i32).max(0);
                let iy2 = ((p[1] + r_p - ofy).floor() as i32).min(mac.sy() as i32 - 1);
                let iz1 = ((p[2] - r_p - ofz).ceil() as i32).max(0);
                let iz2 = ((p[2] + r_p - ofz).floor() as i32).min(mac.sz() as i32 - 1);
                for k in iz1..=iz2 {
                    for j in iy1..=iy2 {
                        for i in ix1..=ix2 {
                            let fx = i as f32 + 0.5 + if a == X { 0.5 } else { 0.0 };
                            let fy = j as f32 + 0.5 + if a == Y { 0.5 } else { 0.0 };
                            let fz = k as f32 + 0.5 + if a == Z { 0.5 } else { 0.0 };
                            let dx = fx - p[0];
                            let dy = fy - p[1];
                            let dz = fz - p[2];
                            let d2 = dx * dx + dy * dy + dz * dz;
                            if d2 > r_p * r_p {
                                continue;
                            }
                            let mut w = 1.0 - d2 * inv_r2;
                            w = w.max(0.0);
                            let w = w * w * w;
                            let old_m = mac.mass(a).get_voxel(i as usize, j as usize, k as usize);
                            mac.mass_mut(a).set_voxel(
                                i as usize,
                                j as usize,
                                k as usize,
                                old_m + w * m_p,
                            );
                            let old_v = mac.velocity(a).get_voxel(i as usize, j as usize, k as usize);
                            mac.velocity_mut(a).set_voxel(
                                i as usize,
                                j as usize,
                                k as usize,
                                old_v + w * m_p * u[a],
                            );
                        }
                    }
                }
            }
        }
        mac.normalize_velocity(&active);
    }

    fn compare(a: &MacGrid<BSX, N>, b: &MacGrid<BSX, N>, tol: f32) {
        for dir in 0..3 {
            for k in 0..a.sz() {
                for j in 0..a.sy() {
                    for i in 0..a.sx() {
                        let am = a.mass(dir).get_voxel(i, j, k);
                        let bm = b.mass(dir).get_voxel(i, j, k);
                        assert!(
                            (am - bm).abs() <= tol,
                            "mass[{dir}] ({i},{j},{k}): {am} vs {bm}"
                        );
                        let av = a.velocity(dir).get_voxel(i, j, k);
                        let bv = b.velocity(dir).get_voxel(i, j, k);
                        assert!(
                            (av - bv).abs() <= tol,
                            "vel[{dir}] ({i},{j},{k}): {av} vs {bv}"
                        );
                    }
                }
            }
        }
    }

    // Happy path: the staged SIMD splat matches the naive Eq. 6 reference.
    #[test]
    fn p2g_01_single_particle_matches_reference() {
        let ps = particles(&[[8.0, 8.0, 8.0]], &[[1.0, 2.0, 3.0]], &[1.0]);
        let mut a = make_mac(32);
        let mut b = make_mac(32);
        particles_to_grid::<BSX, N, MSX>(&mut a, &ps, 2.0);
        reference(&mut b, &ps, 2.0);
        compare(&a, &b, 1e-4);
    }

    // Happy path: a rigidly-translating cloud round-trips exactly through
    // P2G + G2P (a constant field interpolates to itself).
    #[test]
    fn p2g_02_rigid_translation_roundtrip() {
        let u = Vec3::new(1.5, -2.0, 0.5);
        let ps = particles(
            &[[8.0, 8.0, 8.0], [12.0, 9.0, 7.0], [9.0, 14.0, 10.0]],
            &[[u.x(), u.y(), u.z()]; 3],
            &[1.0; 3],
        );
        let mut mac = make_mac(32);
        particles_to_grid::<BSX, N, MSX>(&mut mac, &ps, 2.0);
        let sampler = MacSampler::new(&mac, BoundaryCondition::Clamp);
        for p in ps.positions {
            let got = sampler.sample::<{ Interpolation::Linear }>(Vec3::new(p[0], p[1], p[2]));
            assert!((got.x() - u.x()).abs() < 1e-4, "x: {} vs {}", got.x(), u.x());
            assert!((got.y() - u.y()).abs() < 1e-4, "y: {} vs {}", got.y(), u.y());
            assert!((got.z() - u.z()).abs() < 1e-4, "z: {} vs {}", got.z(), u.z());
        }
    }

    // Boundary crossing: a particle near a block seam must splat into the
    // neighboring block.
    #[test]
    fn p2g_03_block_boundary_crossing() {
        let ps = particles(&[[15.9, 8.0, 8.0]], &[[1.0, 0.0, 0.0]], &[1.0]);
        let mut a = make_mac(32);
        let mut b = make_mac(32);
        particles_to_grid::<BSX, N, MSX>(&mut a, &ps, 2.0);
        reference(&mut b, &ps, 2.0);
        compare(&a, &b, 1e-4);
        // block (1,0,0) must have received mass.
        let any = (0..BSX).any(|i| a.mass(X).get_voxel(16 + i, 8, 8) > 0.0);
        assert!(any, "expected mass in the +x neighbor block");
    }

    // Domain corner: the face footprint clips, no panic, mass stays finite.
    #[test]
    fn p2g_04_domain_corner_clip() {
        let ps = particles(&[[0.1, 0.1, 0.1]], &[[1.0, 1.0, 1.0]], &[1.0]);
        let mut a = make_mac(32);
        let mut b = make_mac(32);
        particles_to_grid::<BSX, N, MSX>(&mut a, &ps, 2.0);
        reference(&mut b, &ps, 2.0);
        compare(&a, &b, 1e-4);
    }

    // Radius cull: a face beyond the support receives nothing, even though a
    // neighboring face is touched.
    #[test]
    fn p2g_05_radius_cull() {
        let ps = particles(&[[8.0, 8.0, 8.0]], &[[1.0, 0.0, 0.0]], &[1.0]);
        let mut mac = make_mac(32);
        particles_to_grid::<BSX, N, MSX>(&mut mac, &ps, 2.0);
        // x-face (i,j,k) at (i+1, j+.5, k+.5); the face at x = 8±2 is at the
        // support edge. Face (10, 8, 8) -> x=11, dist 3 > 2, must be zero.
        assert_eq!(mac.mass(X).get_voxel(10, 8, 8), 0.0);
        assert!(mac.mass(X).get_voxel(7, 8, 8) > 0.0);
    }

    // Overlap: two particles' contributions sum, order-independently.
    #[test]
    fn p2g_06_overlap_sum_order_independent() {
        let a_ps = particles(&[[8.0, 8.0, 8.0], [9.0, 8.0, 8.0]], &[[1.0, 0.0, 0.0]; 2], &[1.0; 2]);
        let b_ps = particles(&[[9.0, 8.0, 8.0], [8.0, 8.0, 8.0]], &[[1.0, 0.0, 0.0]; 2], &[1.0; 2]);
        let mut a = make_mac(32);
        let mut b = make_mac(32);
        particles_to_grid::<BSX, N, MSX>(&mut a, &a_ps, 2.0);
        particles_to_grid::<BSX, N, MSX>(&mut b, &b_ps, 2.0);
        compare(&a, &b, 1e-5);
    }

    // Zero-mass face: normalization yields velocity 0, not NaN.
    #[test]
    fn p2g_07_zero_mass_guard() {
        let ps = particles(&[[8.0, 8.0, 8.0]], &[[1.0, 0.0, 0.0]], &[1.0]);
        let mut mac = make_mac(32);
        particles_to_grid::<BSX, N, MSX>(&mut mac, &ps, 2.0);
        // A face far from the particle has zero mass and zero velocity.
        assert_eq!(mac.velocity(X).get_voxel(20, 20, 20), 0.0);
        assert!(mac.velocity(X).get_voxel(7, 8, 8).is_finite());
    }
}

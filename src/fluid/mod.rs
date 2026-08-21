//! MAC-grid fluid transfer (step 13): the particles-to-grid (P2G) splat of mass
//! + momentum to staggered faces, and the grid-to-particles (G2P) FLIP/PIC
//! gather.
//!
//! The paper's core unification (§3.2–3.6): a single cubic-falloff splat to
//! *cell faces* produces the face mass `M` — which is simultaneously the raw
//! phase-field density (Eq. 7) and the variable Poisson coefficient `β = 1/ρ`
//! (Eq. 9) — and the face momentum, whose ratio is the face velocity `ũ* = P/M`.
//! G2P is `u_new = α·u_old + I(Δu) + (1−α)·I(u)` (Eq. 12).
//!
//! The splat reuses the step-9 staged 8-color architecture (thread-local
//! staging + a 3×3×3 commit), but with a *sum* reduction, a *cubic-falloff*
//! weight, and a half-cell face offset per direction.

pub mod g2p;
pub mod mac;
pub mod p2g;

pub use g2p::{g2p_blend, MacSampler};
pub use mac::{MacGrid, X, Y, Z};
pub use p2g::particles_to_grid;

/// Particle kinds (Eq. 5: `m_p = ρ_kind · V`).
pub const KIND_LIQUID: u8 = 0;
pub const KIND_AIR: u8 = 1;

/// FLIP particles, stored structure-of-arrays so the P2G splat streams each
/// payload contiguously.
#[derive(Clone, Debug, Default)]
pub struct Particles {
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub kinds: Vec<u8>,
    /// Per-particle mass `m_p` (Eq. 5), precomputed by the caller.
    pub mass: Vec<f32>,
}

impl Particles {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Debug consistency check (debug builds only).
    pub fn assert_consistent(&self) {
        debug_assert_eq!(self.positions.len(), self.velocities.len());
        debug_assert_eq!(self.positions.len(), self.kinds.len());
        debug_assert_eq!(self.positions.len(), self.mass.len());
    }
}

/// Block-major particle payloads (the step-13 analogue of `particles::sort::Bucketed`).
#[derive(Clone, Debug, Default)]
pub struct BucketedParticles {
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub kinds: Vec<u8>,
    pub mass: Vec<f32>,
    /// Blocks with at least one particle, sorted by block id.
    pub particle_blocks: Vec<usize>,
    /// CSR start offsets aligned to `particle_blocks` (len = blocks + 1).
    pub starts: Vec<usize>,
}

/// Bucket particles into block-major order via the shared
/// [`crate::particles::sort::bucket_indices`] permutation (one counting sort
/// serves any particle layout).
pub fn bucket_particles(particles: &Particles, bids: &[usize]) -> BucketedParticles {
    particles.assert_consistent();
    debug_assert_eq!(bids.len(), particles.len());
    let plan = crate::particles::sort::bucket_indices(bids);
    let n = particles.len();
    let mut out = BucketedParticles {
        positions: vec![[0.0; 3]; n],
        velocities: vec![[0.0; 3]; n],
        kinds: vec![0; n],
        mass: vec![0.0; n],
        particle_blocks: plan.particle_blocks,
        starts: plan.starts,
    };
    for (i, &src) in plan.perm.iter().enumerate() {
        out.positions[i] = particles.positions[src];
        out.velocities[i] = particles.velocities[src];
        out.kinds[i] = particles.kinds[src];
        out.mass[i] = particles.mass[src];
    }
    out
}

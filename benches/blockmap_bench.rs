//! Step-11 blockmap lookup micro-bench.
//!
//! Compares three block-id→pointer lookup structures on the same key/value
//! distribution (a spherical-shell active set, the same shape the multires
//! benches use):
//!
//!   * `dense`  — the step-1..10 `Vec<Option<NonNull<Block>>>` layout
//!     (baseline, one array index per probe)
//!   * `hand`   — the new [`msbg_rs::blockmap::BlockMap`] (open addressing)
//!   * `hashbrown` — `hashbrown::HashMap` with the identical SplitMix hash
//!
//! Legs: `hits` (probe only active bids) and `mixed` (~50% active / 50% random).
//! Acceptance (step 11): the sparse probe within ~2× of the dense index.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hashbrown::HashMap;
use msbg_rs::blockmap::BlockMap;
use msbg_rs::blockpool::Block;
use std::hash::{BuildHasherDefault, Hasher};
use std::ptr::NonNull;

const BSX: usize = 16;
const N: usize = 4096;

// Mirror of `msbg_rs::blockmap::mix` (single odd-constant multiply); kept
// inline so the library hash stays private while the comparison is over the
// *same* hash.
#[inline(always)]
fn mix(key: usize) -> usize {
    (key as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) as usize
}

#[derive(Default)]
struct MixHasher(u64);

impl Hasher for MixHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = mix(self.0 as usize ^ b as usize) as u64;
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = mix(i) as u64;
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = mix(i as usize) as u64;
    }
}

type MixBuilder = BuildHasherDefault<MixHasher>;

fn shell_blocks(nx: usize, ny: usize, nz: usize) -> Vec<usize> {
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

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("blockmap_lookup");
    group.sample_size(60);

    for bpd in [8usize, 16, 24] {
        let n_blocks = bpd * bpd * bpd;
        let active = shell_blocks(bpd, bpd, bpd);

        // Dense baseline: exact step-1..10 layout.
        let mut dense: Vec<Option<NonNull<Block<f32, BSX, N>>>> = vec![None; n_blocks];
        // Hand-rolled sparse map.
        let mut hand: BlockMap<NonNull<Block<f32, BSX, N>>> = BlockMap::with_capacity(active.len() * 2);
        // hashbrown with the same SplitMix hash.
        let mut hb: HashMap<usize, NonNull<Block<f32, BSX, N>>, MixBuilder> =
            HashMap::with_capacity_and_hasher(active.len() * 2, MixBuilder::default());

        // Allocate one real block per active bid (same pointer used by all three).
        let pool = msbg_rs::blockpool::BlockPool::<f32, BSX, N>::new(active.len() / 4096 + 2, 4096);
        let mut ptrs = Vec::with_capacity(active.len());
        for &bid in &active {
            let p = pool.alloc_block();
            ptrs.push(p);
            dense[bid] = Some(p);
            hand.insert(bid, p);
            hb.insert(bid, p);
        }

        // Probe sequences: all-hits (active bids in a fixed pseudo-random order)
        // and mixed (interleave random in-range bids, ~50% misses).
        let mut hits: Vec<usize> = active.clone();
        let mut seed = 0x243f_6a88u64;
        let mut rng = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for i in (1..hits.len()).rev() {
            let j = rng() % (i + 1);
            hits.swap(i, j);
        }
        let mut mixed: Vec<usize> = Vec::with_capacity(hits.len());
        for (i, &h) in hits.iter().enumerate() {
            mixed.push(if i % 2 == 0 { h } else { rng() % n_blocks });
        }

        let hits_ref = &hits;
        let mixed_ref = &mixed;

        group.bench_with_input(BenchmarkId::new("dense/hits", bpd), &n_blocks, |b, _| {
            b.iter(|| {
                let mut acc = 0usize;
                for &bid in hits_ref {
                    acc = acc.wrapping_add(if dense[bid].is_some() { 1 } else { 0 });
                }
                std::hint::black_box(acc);
            });
        });
        group.bench_with_input(BenchmarkId::new("dense/mixed", bpd), &n_blocks, |b, _| {
            b.iter(|| {
                let mut acc = 0usize;
                for &bid in mixed_ref {
                    acc = acc.wrapping_add(if dense[bid].is_some() { 1 } else { 0 });
                }
                std::hint::black_box(acc);
            });
        });
        group.bench_with_input(BenchmarkId::new("hand/hits", bpd), &n_blocks, |b, _| {
            b.iter(|| {
                let mut acc = 0usize;
                for &bid in hits_ref {
                    acc = acc.wrapping_add(if hand.get(bid).is_some() { 1 } else { 0 });
                }
                std::hint::black_box(acc);
            });
        });
        group.bench_with_input(BenchmarkId::new("hand/mixed", bpd), &n_blocks, |b, _| {
            b.iter(|| {
                let mut acc = 0usize;
                for &bid in mixed_ref {
                    acc = acc.wrapping_add(if hand.get(bid).is_some() { 1 } else { 0 });
                }
                std::hint::black_box(acc);
            });
        });
        group.bench_with_input(BenchmarkId::new("hashbrown/hits", bpd), &n_blocks, |b, _| {
            b.iter(|| {
                let mut acc = 0usize;
                for &bid in hits_ref {
                    acc = acc.wrapping_add(if hb.get(&bid).is_some() { 1 } else { 0 });
                }
                std::hint::black_box(acc);
            });
        });
        group.bench_with_input(BenchmarkId::new("hashbrown/mixed", bpd), &n_blocks, |b, _| {
            b.iter(|| {
                let mut acc = 0usize;
                for &bid in mixed_ref {
                    acc = acc.wrapping_add(if hb.get(&bid).is_some() { 1 } else { 0 });
                }
                std::hint::black_box(acc);
            });
        });
    }
    group.finish();
}

criterion_group!(blockmap, bench_lookup);
criterion_main!(blockmap);

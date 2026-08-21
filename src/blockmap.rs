//! Sparse, block-id-keyed map (open addressing, linear probing).
//!
//! Replaces the dense per-grid `Vec<Option<BlockPtr>>` blockmap so lookup cost
//! scales with *active* blocks, not *virtual* blocks. At 32,768³/block-16 the
//! dense map is 68.7 GB per grid; this map is ~0.4 GB for the same domain.
//!
//! Keys are block ids (`usize`, 64-bit on all supported targets — 2048³ =
//! 8.59B blocks > `u32::MAX`). The table is SoA (`keys: Vec<usize>` +
//! `vals: Vec<MaybeUninit<V>>`) so a probe walks 8-byte key slots only, and
//! linear probing keeps the working set within one or two cache lines for
//! clustered block sets. Power-of-two capacity, ~3/4 load factor, tombstones
//! on delete (cleared on rehash).
//!
//! `V` is the payload pointer (a `BlockPtr` or an SoA level-block pointer);
//! `V: Copy` means entries never need dropping, so the `MaybeUninit` slots can
//! be left untouched.

use std::mem::MaybeUninit;

/// Vacant slot marker (no entry, end of a probe chain).
const EMPTY: usize = usize::MAX;
/// Deleted slot marker (entry removed; probe chains continue past it).
const TOMBSTONE: usize = usize::MAX - 1;

/// Multiplicative hash: one odd-constant multiply. Multiplication by an odd
/// constant is a bijection mod 2^k, so consecutive bid clusters (the low bits
/// are the fastest-varying block axis) spread uniformly across a power-of-two
/// table — ideal for linear probing. Far cheaper than a SplitMix finalizer on
/// the per-probe hot path (measured: ~3.5x faster probe).
#[inline(always)]
fn mix(key: usize) -> usize {
    (key as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) as usize
}

pub struct BlockMap<V: Copy> {
    keys: Vec<usize>,
    vals: Vec<MaybeUninit<V>>,
    len: usize,
    tombstones: usize,
    mask: usize,
}

impl<V: Copy> BlockMap<V> {
    pub fn new() -> Self {
        Self::with_capacity(4)
    }

    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(4).next_power_of_two();
        Self {
            keys: vec![EMPTY; cap],
            vals: vec![MaybeUninit::uninit(); cap],
            len: 0,
            tombstones: 0,
            mask: cap - 1,
        }
    }

    /// Number of live entries.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocated slot count (power of two).
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.keys.len()
    }

    #[inline(always)]
    fn slot(&self, key: usize) -> usize {
        mix(key) & self.mask
    }

    #[inline(always)]
    fn grow_threshold(&self) -> usize {
        self.capacity() * 3 / 4
    }

    #[inline]
    pub fn get(&self, key: usize) -> Option<V> {
        let mut i = self.slot(key);
        loop {
            let k = self.keys[i];
            if k == key {
                return Some(unsafe { self.vals[i].assume_init() });
            }
            if k == EMPTY {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline]
    pub fn insert(&mut self, key: usize, val: V) {
        if self.len + self.tombstones + 1 > self.grow_threshold() {
            self.grow();
        }
        self.insert_no_grow(key, val);
    }

    #[inline]
    fn insert_no_grow(&mut self, key: usize, val: V) {
        debug_assert!(key != EMPTY && key != TOMBSTONE, "block id collides with a sentinel");
        let mut i = self.slot(key);
        loop {
            let k = self.keys[i];
            if k == EMPTY || k == TOMBSTONE {
                if k == TOMBSTONE {
                    self.tombstones -= 1;
                }
                self.keys[i] = key;
                self.vals[i].write(val);
                self.len += 1;
                return;
            }
            if k == key {
                self.vals[i].write(val);
                return;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Remove `key`, returning its value if present. Leaves a tombstone so
    /// later entries in the same probe chain stay reachable.
    #[inline]
    pub fn remove(&mut self, key: usize) -> Option<V> {
        let mut i = self.slot(key);
        loop {
            let k = self.keys[i];
            if k == EMPTY {
                return None;
            }
            if k == key {
                self.keys[i] = TOMBSTONE;
                let v = unsafe { self.vals[i].assume_init() };
                self.len -= 1;
                self.tombstones += 1;
                return Some(v);
            }
            i = (i + 1) & self.mask;
        }
    }

    /// If `key` is present, replace its value with `f(old)` (single probe on
    /// the hit path); otherwise insert `default` (without calling `f`). Returns
    /// the new value. The sparse-histogram hot loop (`count[key] += 1`).
    #[inline]
    pub fn update(&mut self, key: usize, default: V, mut f: impl FnMut(V) -> V) -> V {
        debug_assert!(key != EMPTY && key != TOMBSTONE, "key collides with a sentinel");
        let mut i = self.slot(key);
        loop {
            let k = self.keys[i];
            if k == key {
                let old = unsafe { self.vals[i].assume_init() };
                let new = f(old);
                self.vals[i].write(new);
                return new;
            }
            if k == EMPTY {
                break;
            }
            i = (i + 1) & self.mask;
        }
        self.insert(key, default);
        default
    }

    /// All live `(key, value)` pairs sorted by key (for the k-way merges of the
    /// parallel sparse histogram).
    pub fn sorted_pairs(&self) -> Vec<(usize, V)>
    where
        V: Sync + Send,
    {
        use rayon::slice::ParallelSliceMut;
        let mut v: Vec<(usize, V)> = self.iter().collect();
        v.par_sort_unstable_by_key(|&(k, _)| k);
        v
    }

    /// Iterate live `(key, value)` pairs (map order, not sorted).
    pub fn iter(&self) -> impl Iterator<Item = (usize, V)> + '_ {
        (0..self.keys.len()).filter_map(|i| {
            let k = self.keys[i];
            if k != EMPTY && k != TOMBSTONE {
                Some((k, unsafe { self.vals[i].assume_init() }))
            } else {
                None
            }
        })
    }

    fn grow(&mut self) {
        let new_cap = self.capacity() * 2;
        let mut new = Self::with_capacity(new_cap);
        for (k, v) in self.iter() {
            new.insert_no_grow(k, v);
        }
        *self = new;
    }
}

impl<V: Copy> Default for BlockMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A sparse set of block ids (`V = ()` keeps the value array zero-cost). Used
/// for the active-block / footprint unions where only membership matters.
pub type BlockSet = BlockMap<()>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm_01_insert_get_overwrite() {
        let mut m = BlockMap::new();
        assert!(m.get(7).is_none());
        m.insert(7, 100u64);
        assert_eq!(m.get(7), Some(100));
        m.insert(7, 200);
        assert_eq!(m.get(7), Some(200));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_bm_02_growth_rehash_preserves_all() {
        let mut m = BlockMap::new();
        for i in 0..10_000usize {
            m.insert(i, (i * 3) as u64);
        }
        assert_eq!(m.len(), 10_000);
        for i in 0..10_000usize {
            assert_eq!(m.get(i), Some((i * 3) as u64));
        }
    }

    #[test]
    fn test_bm_03_collision_cluster_consecutive() {
        // Consecutive keys are the worst clustering case; all must resolve.
        let mut m = BlockMap::new();
        for i in 0..1000usize {
            m.insert(i, i as u64);
        }
        for i in 0..1000usize {
            assert_eq!(m.get(i), Some(i as u64));
        }
    }

    #[test]
    fn test_bm_04_delete_reinsert_same_key() {
        let mut m = BlockMap::new();
        m.insert(42, 1u64);
        assert_eq!(m.remove(42), Some(1));
        assert!(m.get(42).is_none());
        m.insert(42, 2);
        assert_eq!(m.get(42), Some(2));
    }

    #[test]
    fn test_bm_05_probe_chain_across_tombstone() {
        // Force collisions by inserting keys that share a slot, delete the
        // first, then prove the second is still reachable past the tombstone.
        let mut m = BlockMap::with_capacity(4);
        let base = 0usize;
        // Find two keys that collide at capacity 4.
        let mut colliding = None;
        let s0 = m.slot(base);
        for k in 1..1000usize {
            if m.slot(k) == s0 && k != base {
                colliding = Some(k);
                break;
            }
        }
        let k2 = colliding.expect("expected a colliding key at capacity 4");
        m.insert(base, 1u64);
        m.insert(k2, 2u64);
        assert_eq!(m.remove(base), Some(1));
        assert_eq!(m.get(k2), Some(2));
    }

    #[test]
    fn test_bm_06_load_factor_rehash_with_tombstones() {
        let mut m = BlockMap::with_capacity(8);
        for i in 0..6usize {
            m.insert(i, i as u64);
        }
        m.remove(0);
        m.remove(1);
        // Tombstones occupy slots and must not lose live entries on growth.
        m.insert(6, 6);
        m.insert(7, 7);
        m.insert(8, 8);
        for i in 2..9usize {
            assert_eq!(m.get(i), Some(i as u64), "key {i} lost");
        }
        assert!(m.get(0).is_none());
        assert!(m.get(1).is_none());
    }

    #[test]
    fn test_bm_07_empty_map_get_none() {
        let m: BlockMap<u64> = BlockMap::new();
        assert!(m.get(0).is_none());
        assert!(m.get(usize::MAX - 2).is_none());
        assert!(m.iter().next().is_none());
        assert!(m.is_empty());
    }

    #[test]
    fn test_bm_08_u64_scale_keys() {
        // Block ids straddling u32::MAX must round-trip (proves the 64-bit
        // key path the paper-scale domain needs).
        let mut m = BlockMap::new();
        for k in [0usize, u32::MAX as usize - 1, u32::MAX as usize, u32::MAX as usize + 1, 0x1_0000_0000] {
            m.insert(k, k as u64);
        }
        for k in [0usize, u32::MAX as usize - 1, u32::MAX as usize, u32::MAX as usize + 1, 0x1_0000_0000] {
            assert_eq!(m.get(k), Some(k as u64));
        }
    }

    #[test]
    fn test_bm_09_single_slot() {
        let mut m = BlockMap::with_capacity(1);
        m.insert(5, 9u64);
        assert_eq!(m.get(5), Some(9));
        assert!(m.get(6).is_none());
    }

    #[test]
    fn test_bm_10_iter_yields_all() {
        let mut m = BlockMap::new();
        for i in 0..100usize {
            m.insert(i * 7, i as u64);
        }
        let mut seen: Vec<(usize, u64)> = m.iter().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..100usize).map(|i| (i * 7, i as u64)).collect::<Vec<_>>());
    }

    #[test]
    fn test_bm_11_update_increment_and_insert() {
        let mut m = BlockMap::new();
        // Missing key -> default (f not applied).
        assert_eq!(m.update(7, 1u32, |v| v + 1), 1);
        assert_eq!(m.update(7, 1, |v| v + 1), 2);
        assert_eq!(m.update(7, 100, |v| v + 1), 3); // default ignored on hit
        assert_eq!(m.get(7), Some(3));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_bm_12_blockset_membership() {
        let mut s: BlockSet = BlockSet::new();
        assert!(s.get(5).is_none());
        s.insert(5, ());
        s.insert(5, ()); // idempotent
        assert!(s.get(5).is_some());
        assert_eq!(s.len(), 1);
        s.remove(5);
        assert!(s.get(5).is_none());
    }

    #[test]
    fn test_bm_13_sorted_pairs_order() {
        let mut m = BlockMap::new();
        for k in [40usize, 0, 20, u32::MAX as usize, 10] {
            m.insert(k, k as u64);
        }
        let p = m.sorted_pairs();
        let keys: Vec<usize> = p.iter().map(|&(k, _)| k).collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(keys.len(), 5);
    }
}

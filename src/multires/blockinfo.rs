//! Per-block topology: block flags, cell flags, and the SoA `BlockInfoStore`.

// ---------------------------------------------------------------------------
// Cell flags (C++ `msbg.h` enum at the `CELL_*` block).
// ---------------------------------------------------------------------------
pub const CELL_SOLID: u16 = 1 << 0;
pub const CELL_FIXED: u16 = 1 << 1;
pub const CELL_AIR: u16 = 1 << 2;
pub const CELL_AIR_BORDER: u16 = 1 << 3;
pub const CELL_EMPTY_U: u16 = 1 << 4;
pub const CELL_EMPTY_V: u16 = 1 << 5;
pub const CELL_PARTIAL_SOLID: u16 = 1 << 6;
pub const CELL_OUT_OF_DOMAIN: u16 = 1 << 7;
pub const CELL_BLK_BORDER: u16 = 1 << 8;
pub const CELL_COARSE_FINE: u16 = 1 << 9;
pub const CELL_FINE_COARSE: u16 = 1 << 10;
pub const CELL_EMPTY_C: u16 = 1 << 11;
pub const CELL_VOID: u16 = 1 << 12;
pub const CELL_BOUNDARY_ZONE: u16 = 1 << 13;
pub const CELL_TMP_MARK_2: u16 = 1 << 14;
pub const CELL_EMPTY_W: u16 = 1 << 15;

/// Sentinel cell-flag value for empty blocks (matches C++ `CELL_EMPTY_VAL`).
pub const CELL_EMPTY_VAL: u16 =
    CELL_VOID | CELL_EMPTY_U | CELL_EMPTY_V | CELL_EMPTY_W | CELL_EMPTY_C;

/// `!(cell & (CELL_SOLID | CELL_VOID))` — the C++ `CELL_IS_FLUID_` predicate.
#[inline(always)]
pub fn cell_is_fluid(cell: u16) -> bool {
    cell & (CELL_SOLID | CELL_VOID) == 0
}

// Block flags (C++ `BlockFlags` enum at `msbg.h:558`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BlockFlags(pub u16);

impl BlockFlags {
    pub const HAS_AIR: u16 = 1 << 0;
    pub const DOM_BORDER: u16 = 1 << 1;
    pub const COARSE_FINE: u16 = 1 << 2;
    pub const FINE_COARSE: u16 = 1 << 3;
    pub const EMPTY: u16 = 1 << 4;
    pub const NO_FLUID: u16 = 1 << 5;
    pub const ONLY_FLUID: u16 = 1 << 6;
    pub const BOUNDARY_ZONE: u16 = 1 << 7;
    pub const HAS_UNDEFINED: u16 = 1 << 8;
    pub const SLOW: u16 = 1 << 9;
    pub const TMP_MARK: u16 = 1 << 10;
    pub const FIXED: u16 = 1 << 11;
    pub const EXISTS: u16 = 1 << 12;
    pub const UNIFORM_EMPTY: u16 = 1 << 13;
    pub const HAS_DEFINED: u16 = 1 << 14;
    pub const CUTS_SOLID: u16 = 1 << 15;

    #[inline(always)]
    pub fn contains(self, bit: u16) -> bool {
        self.0 & bit != 0
    }

    #[inline(always)]
    pub fn set(&mut self, bit: u16) {
        self.0 |= bit;
    }

    #[inline(always)]
    pub fn reset(&mut self, bit: u16) {
        self.0 &= !bit;
    }

    /// `BLK_IS_RES_BORDER`: block sits on a fine-coarse or coarse-fine border.
    #[inline(always)]
    pub fn is_res_border(self) -> bool {
        self.0 & (Self::COARSE_FINE | Self::FINE_COARSE) != 0
    }
}

// SoA per-block info. C++ keeps an AoS `BlockInfo { uint16 level, flags }` per
// (levelMg, block); instead, store the finest-level (`levelMg == 0`) refinement
// once as `u8` and only the flags per multigrid level, so the hot sweeps walk
// 1 byte/block instead of 4.
pub struct BlockInfoStore {
    /// Finest-level refinement level per block (`levelMg == 0`). Coarser MG
    /// levels derive their effective level as `max(level0[bid], level_mg)`.
    pub level0: Vec<u8>,
    /// Block flags per multigrid level: `flags[level_mg][bid]`.
    pub flags: Vec<Vec<BlockFlags>>,
}

impl BlockInfoStore {
    /// C++ `create()` initializes every block to `level = n_levels - 1` (the
    /// coarsest resolution) with zero flags.
    pub fn new(n_blocks: usize, n_levels: usize) -> Self {
        Self {
            level0: vec![(n_levels - 1) as u8; n_blocks],
            flags: vec![vec![BlockFlags(0); n_blocks]; n_levels],
        }
    }

    /// Effective resolution level of `bid` at multigrid level `level_mg`.
    #[inline(always)]
    pub fn level(&self, bid: usize, level_mg: usize) -> u8 {
        (self.level0[bid] as usize).max(level_mg) as u8
    }

    #[inline(always)]
    pub fn flags(&self, bid: usize, level_mg: usize) -> BlockFlags {
        self.flags[level_mg][bid]
    }

    #[inline(always)]
    pub fn flags_mut(&mut self, bid: usize, level_mg: usize) -> &mut BlockFlags {
        &mut self.flags[level_mg][bid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bi_01_fluid_predicate() {
        assert!(cell_is_fluid(0));
        assert!(cell_is_fluid(CELL_AIR));
        assert!(!cell_is_fluid(CELL_SOLID));
        assert!(!cell_is_fluid(CELL_VOID));
        assert!(!cell_is_fluid(CELL_SOLID | CELL_VOID));
    }

    #[test]
    fn test_bi_02_empty_val_composition() {
        // CELL_EMPTY_VAL must be non-fluid and carry the empty_U/V/W/C bits.
        assert!(!cell_is_fluid(CELL_EMPTY_VAL));
        assert_eq!(CELL_EMPTY_VAL & CELL_VOID, CELL_VOID);
        assert_eq!(CELL_EMPTY_VAL & CELL_EMPTY_W, CELL_EMPTY_W);
    }

    #[test]
    fn test_bi_03_flag_bits_unique() {
        // Every block flag occupies a distinct bit.
        let bits = [
            BlockFlags::HAS_AIR,
            BlockFlags::DOM_BORDER,
            BlockFlags::COARSE_FINE,
            BlockFlags::FINE_COARSE,
            BlockFlags::EMPTY,
            BlockFlags::NO_FLUID,
            BlockFlags::ONLY_FLUID,
            BlockFlags::BOUNDARY_ZONE,
            BlockFlags::HAS_UNDEFINED,
            BlockFlags::SLOW,
            BlockFlags::TMP_MARK,
            BlockFlags::FIXED,
            BlockFlags::EXISTS,
            BlockFlags::UNIFORM_EMPTY,
            BlockFlags::HAS_DEFINED,
            BlockFlags::CUTS_SOLID,
        ];
        let mut seen = 0u16;
        for b in bits {
            assert_eq!(b & seen, 0, "block flag bit {b:#x} collides");
            seen |= b;
        }
        assert_eq!(seen, u16::MAX, "block flags should span all 16 bits");
    }

    #[test]
    fn test_bi_04_flag_helpers() {
        let mut f = BlockFlags(0);
        assert!(!f.contains(BlockFlags::EXISTS));
        f.set(BlockFlags::EXISTS);
        assert!(f.contains(BlockFlags::EXISTS));
        f.set(BlockFlags::FINE_COARSE);
        assert!(f.is_res_border());
        f.reset(BlockFlags::FINE_COARSE);
        assert!(!f.is_res_border());
        assert!(f.contains(BlockFlags::EXISTS));
    }

    #[test]
    fn test_bi_05_store_init_coarsest() {
        let store = BlockInfoStore::new(8, 3);
        assert_eq!(store.level0.len(), 8);
        assert_eq!(store.level0, vec![2u8; 8]);
        assert_eq!(store.flags.len(), 3);
        assert!(store.flags.iter().all(|f| f.iter().all(|&x| x == BlockFlags(0))));
    }

    #[test]
    fn test_bi_06_effective_level_caps_at_level_mg() {
        let mut store = BlockInfoStore::new(4, 3);
        store.level0[0] = 0;
        store.level0[1] = 1;
        store.level0[2] = 2;
        assert_eq!(store.level(0, 0), 0);
        assert_eq!(store.level(0, 1), 1);
        assert_eq!(store.level(0, 2), 2);
        assert_eq!(store.level(1, 0), 1);
        assert_eq!(store.level(1, 2), 2);
        assert_eq!(store.level(2, 1), 2);
    }
}

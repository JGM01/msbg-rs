//! Morton (Z-order) block-list sorting.

use crate::multires::BlockGridDims;

/// Spread the low 10 bits of `v` to every third bit (classic Morton interleave).
#[inline(always)]
fn spread(mut v: u32) -> u32 {
    v &= 0x3ff;
    v = (v | (v << 16)) & 0x0300_00ff;
    v = (v | (v << 8)) & 0x0300_f00f;
    v = (v | (v << 4)) & 0x030c_30c3;
    v = (v | (v << 2)) & 0x0924_9249;
    v
}

/// 30-bit 3D Morton code (10 bits per axis).
#[inline(always)]
pub fn morton3(x: u32, y: u32, z: u32) -> u32 {
    spread(x) | (spread(y) << 1) | (spread(z) << 2)
}

/// Sort a block list by a level-scaled Morton key so a coarse block and its
/// finer descendants are contiguous.
///
/// The key is computed on the coarsest lattice: every block shares the Morton
/// code of its coarsest-level ancestor, with coarser levels ordered first and
/// the fine offset (Morton-ordered) last. This makes the coarse block appear
/// directly before its ≤8 finer children — the index-space interleaving that a
/// cache-friendly sweep (and, later, temporal blocking) needs, without touching
/// physical block allocation.
///
/// C++ has the same idea (`sortBlockListMorton`) but it is `#ifdef`'d out.
pub fn sort_block_list_morton(
    blocks: &mut [u32],
    dims: &BlockGridDims,
    levels: &[u8],
    n_levels: usize,
) {
    debug_assert!(n_levels >= 1);
    let ref_shift = (n_levels - 1) as u32;
    let mask = (1u32 << ref_shift) - 1;

    blocks.sort_unstable_by_key(|&bid| {
        let bid = bid as usize;
        let (bx, by, bz) = dims.coords(bid);
        let lvl = levels[bid] as u32;

        let coarse = morton3(bx as u32 >> ref_shift, by as u32 >> ref_shift, bz as u32 >> ref_shift);
        // Coarser first: a block at the reference level gets priority 0.
        let priority = ref_shift - lvl;
        let fine = morton3(bx as u32 & mask, by as u32 & mask, bz as u32 & mask);

        ((coarse as u64) << 34) | ((priority as u64) << 30) | (fine as u64)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_01_zero() {
        assert_eq!(morton3(0, 0, 0), 0);
    }

    #[test]
    fn test_morton_02_unit_axes() {
        // Axis x contributes bit 0, y bit 1, z bit 2.
        assert_eq!(morton3(1, 0, 0), 0b001);
        assert_eq!(morton3(0, 1, 0), 0b010);
        assert_eq!(morton3(0, 0, 1), 0b100);
        assert_eq!(morton3(1, 1, 1), 0b111);
    }

    #[test]
    fn test_morton_03_bijective_low_bits() {
        for x in 0..8u32 {
            for y in 0..8u32 {
                for z in 0..8u32 {
                    // Distinct inputs -> distinct codes over 3 low bits.
                    let code = morton3(x, y, z);
                    assert_eq!(code & 0b111_111_111, code, "low bits collide");
                }
            }
        }
    }

    #[test]
    fn test_morton_04_sort_groups_coarse_first() {
        // 4x4x4 grid, 2 levels. Coarse block (1,1,1) at level 1 with its 8 fine
        // children at level 0 (the 2x2x2 blocks under it).
        let dims = BlockGridDims::new(4, 4, 4);
        let mut levels = vec![0u8; dims.n_blocks];
        levels[21] = 1; // block (1,1,1) is coarse
        let mut blocks: Vec<u32> = vec![];
        // All blocks under coarse cell (1,1,1): coarse block 21 + children
        // (2..4)^3 -> blocks (2,2,2)=42? Let's just collect the coarse block and its children.
        blocks.push(21);
        for bz in 2..4 {
            for by in 2..4 {
                for bx in 2..4 {
                    blocks.push((bx + by * 4 + bz * 16) as u32);
                }
            }
        }
        sort_block_list_morton(&mut blocks, &dims, &levels, 2);
        // The coarse block must come first.
        assert_eq!(blocks[0], 21, "coarse block should precede its fine children");
    }
}

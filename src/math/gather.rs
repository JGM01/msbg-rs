//! Stencil-window gather over a [`SparseGrid`], with inline dequantization.

use crate::math::boundary::{resolve_axis, BoundaryCondition};
use crate::sparse_grid::SparseGrid;

/// A window of `dmax <= 3` spans at most 2 blocks per axis (2^3 = 8).
const MAX_SLOTS: usize = 8;

/// Dequantize a stored element into the sampling domain (`f32` or `Vec3`).
pub trait Dequant<O>: Copy + Default + Send + Sync {
    fn dequant(self) -> O;

    /// Dequantize the element at `src + idx`, given the total element count
    /// `len` of the contiguous run `src` points into. The default copies the
    /// element and dequantizes it; vector types may override this to issue a
    /// single wider load that reads `Self`'s trailing bytes / the successor
    /// element, which is only in-bounds when a successor exists (`idx + 1 < len`).
    ///
    /// # Safety
    ///
    /// `src` must point at `len` valid `Self` elements and `idx < len`.
    ///
    /// **Contract for wide-load overrides.** An override that reads *past* the
    /// element (`idx`'s) allocation — e.g. a 16-byte load of a 12-byte `Vec3` —
    /// is only sound when `src` is the `data` field of a
    /// [`crate::blockpool::Block`], whose `flags: u16` + `_pad: [u8; 62]` tail
    /// is 64 bytes of initialized memory immediately after `data`. That
    /// guarantee is *by construction of the caller*, not a runtime check:
    /// [`gather_map`] is the sole caller of `dequant_at`, and it only ever
    /// passes [`SparseGrid::block_data_ptr`], which resolves real, full, and
    /// empty blocks alike to a `Block::data` pointer. The
    /// `debug_assert!(idx < len)` bounds the *index*; it does **not** detect a
    /// caller handing over an unpadded `[Self; N]` slice or stack array.
    #[inline]
    unsafe fn dequant_at(src: *const Self, idx: usize, len: usize) -> O {
        debug_assert!(idx < len);
        unsafe { (*src.add(idx)).dequant() }
    }

    /// Dequantize `n` contiguous elements from `src` into `dst`. The default is
    /// a scalar loop; `f32` overrides it with a memcpy.
    #[inline]
    fn copy_row(src: *const Self, dst: *mut O, n: usize) {
        for i in 0..n {
            unsafe { *dst.add(i) = (*src.add(i)).dequant(); }
        }
    }
}

/// Gather the `(dmax + 1)^3` stencil window anchored at `(ix0, iy0, iz0)`,
/// mapping each stored element through `Dequant::dequant` into the output.
/// `dmax` is 1 (linear) or 3 (cubic); `out` has length `(dmax + 1)^3`.
#[inline(always)]
pub fn gather_map<D, O, const BSX: usize, const N: usize>(
    grid: &SparseGrid<D, BSX, N>,
    ix0: i32,
    iy0: i32,
    iz0: i32,
    bc: BoundaryCondition,
    dmax: usize,
    out: &mut [O],
) where
    D: Dequant<O>,
    O: Copy,
{
    let bsx_log2 = BSX.trailing_zeros();
    let bsx_mask = BSX - 1;
    let span = (dmax + 1) as i32;
    debug_assert_eq!(out.len(), (dmax + 1) * (dmax + 1) * (dmax + 1));

    // Fast path: window entirely within one block and strictly in-domain
    // (the domain check also rejects partial trailing blocks).
    if ix0 >= 0
        && iy0 >= 0
        && iz0 >= 0
        && (ix0 as usize + dmax) < grid.sx
        && (iy0 as usize + dmax) < grid.sy
        && (iz0 as usize + dmax) < grid.sz
    {
        let vx0 = ix0 as usize & bsx_mask;
        let vy0 = iy0 as usize & bsx_mask;
        let vz0 = iz0 as usize & bsx_mask;
        if vx0 + dmax < BSX && vy0 + dmax < BSX && vz0 + dmax < BSX {
            let bid = ((ix0 as usize) >> bsx_log2)
                + ((iy0 as usize) >> bsx_log2) * grid.nx
                + ((iz0 as usize) >> bsx_log2) * grid.nxy;
            let ptr = grid.block_data_ptr(bid);
            let mut s = 0;
            for k in 0..span {
                for j in 0..span {
                    for i in 0..span {
                        let vid = (vx0 + i as usize)
                            | ((vy0 + j as usize) << bsx_log2)
                            | ((vz0 + k as usize) << (2 * bsx_log2));
                        out[s] = unsafe { D::dequant_at(ptr, vid, N) };
                        s += 1;
                    }
                }
            }
            return;
        }
    }

    let bx0 = ((ix0 >> bsx_log2).max(0).min(grid.nx as i32 - 1)) as usize;
    let by0 = ((iy0 >> bsx_log2).max(0).min(grid.ny as i32 - 1)) as usize;
    let bz0 = ((iz0 >> bsx_log2).max(0).min(grid.nz as i32 - 1)) as usize;

    let empty_ptr = unsafe { (*grid.empty_block.as_ptr()).data.as_ptr() };

    let mut slots: [*const D; MAX_SLOTS] = [empty_ptr; MAX_SLOTS];
    for sz in 0..2usize {
        for sy in 0..2usize {
            for sx in 0..2usize {
                let bxx = (bx0 + sx).min(grid.nx - 1);
                let byy = (by0 + sy).min(grid.ny - 1);
                let bzz = (bz0 + sz).min(grid.nz - 1);
                let bid = bxx + byy * grid.nx + bzz * grid.nxy;
                slots[sx | (sy << 1) | (sz << 2)] = grid.block_data_ptr(bid);
            }
        }
    }

    let sx_max = grid.sx as i32 - 1;
    let sy_max = grid.sy as i32 - 1;
    let sz_max = grid.sz as i32 - 1;
    let empty = grid.empty_value();

    let mut s = 0;
    for k in 0..span {
        let gz = iz0 + k;
        let rz = resolve_axis(gz, sz_max, bc);
        for j in 0..span {
            let gy = iy0 + j;
            let ry = resolve_axis(gy, sy_max, bc);
            for i in 0..span {
                let gx = ix0 + i;
                let rx = resolve_axis(gx, sx_max, bc);
                out[s] = match (rx, ry, rz) {
                    (Some(x), Some(y), Some(z)) => {
                        let x = x as usize;
                        let y = y as usize;
                        let z = z as usize;
                        let slot = ((x >> bsx_log2) - bx0)
                            | (((y >> bsx_log2) - by0) << 1)
                            | (((z >> bsx_log2) - bz0) << 2);
                        debug_assert!(slot < MAX_SLOTS);
                        let vid = (x & bsx_mask)
                            | ((y & bsx_mask) << bsx_log2)
                            | ((z & bsx_mask) << (2 * bsx_log2));
                        unsafe { D::dequant_at(slots[slot], vid, N) }
                    }
                    _ => empty.dequant(),
                };
                s += 1;
            }
        }
    }
}

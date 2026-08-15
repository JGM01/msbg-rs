//! Sampling configuration types and domain-boundary index resolution.

use std::marker::ConstParamTy;

/// How a stencil sample hitting the domain border is resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryCondition {
    /// Replicate the edge voxel.
    Clamp,
    /// Mirror (symmetric) reflection; the edge voxel is not repeated.
    Neumann,
    /// Out-of-domain samples read the grid's empty sentinel.
    Dirichlet,
}

/// Whether positions are node- or cell-centered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GridAlignment {
    /// Sample positions coincide with voxel centres (`OPT_IPCORNER`).
    Corner,
    /// Voxels live at half-integer positions; subtract 0.5 before flooring.
    CellCentered,
}

/// Interpolation stencil order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Interpolation {
    Linear,
    CubicBSpline,
}

/// Map stencil index `g` into `[0, max]`. Returns `None` only for `Dirichlet`
/// when `g` is out of range (caller substitutes the empty sentinel).
#[inline(always)]
pub fn resolve_axis(g: i32, max: i32, bc: BoundaryCondition) -> Option<i32> {
    if (0..=max).contains(&g) {
        return Some(g);
    }
    match bc {
        BoundaryCondition::Clamp => Some(g.clamp(0, max)),
        BoundaryCondition::Neumann => Some(reflect(g, max)),
        BoundaryCondition::Dirichlet => None,
    }
}

/// Symmetric mirror: `-1 -> 0`, `-2 -> 1`, `max+1 -> max`, `max+2 -> max-1`, ...
#[inline(always)]
fn reflect(g: i32, max: i32) -> i32 {
    let period = 2 * (max + 1);
    let r = g.rem_euclid(period);
    if r > max {
        period - 1 - r
    } else {
        r
    }
}

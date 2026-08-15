//! Typed data channels over `SparseGrid`.
//!
//! Step 3 of the port: a typed channel table replaces the C++ `int`-based
//! channel enum + `void** channelPointers` type-punning. Each channel carries
//! its element type at compile time; wrong-type access is a compile error.
//!
//! ```compile_fail
//! use msbg_rs::channel::{ChannelTable, Density};
//! use msbg_rs::sparse_grid::SparseGrid;
//! let mut t = ChannelTable::<16, 4096>::new();
//! // `get_pressure()` yields `SparseGrid<Pressure>`; ascribing `SparseGrid<Density>` must not compile.
//! let _: &SparseGrid<Density, 16, 4096> = t.get_pressure().unwrap();
//! ```
//!
//! TODO(step3): SIMD batch dequant — convert a whole `[Density; N]` block to
//! `[f32; N]` in one pass (C++ has `renderDensToFloat_simd8`).
//! TODO(step3): u8 density (`RSURF_8_BIT`), incl. the u8-only sqrt *encode*
//! (`renderDensFromFloat_` applies `doSqrtCompr` only when `sizeof(T)==1`).
//! TODO(step3): the remaining ~39 C++ channels (divergence, curvature, heat,
//! etc.) — each is one more `field: Type` line in `channel_table!`.
//! TODO(step3): `prepareDataAccess` / `resetChannel` / `protectChannel`
//! semantics (write-lock tokens) — needed by the Step 8 solvers.
//! TODO(step7): make the table per-level (multires); today it is single-level.
//! TODO(step3): shared byte-arena pool for lower peak DRAM — a separate
//! `BlockPool` redesign, orthogonal to this module.

use crate::sparse_grid::SparseGrid;
use std::ops::{Add, Mul, Sub};

/// Quantized scalar density, `u16` in `[0, 65535]`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Density(pub u16);

/// Per-voxel cell type flags (solid/air/void/fluid bitfield).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CellFlags(pub u16);

/// Scalar pressure field.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pressure(pub f32);

/// A 3-component float vector.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3(pub [f32; 3]);

/// Fluid velocity channel element.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Velocity(pub Vec3);

impl Density {
    /// Full-scale value used by the quantizer (`u16::MAX`).
    pub const MAX: f32 = u16::MAX as f32;

    /// Decode to linear `[0, 1]` float.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::MAX
    }

    /// Decode a sqrt-compressed value: `(v/max)^2`.
    #[inline(always)]
    pub fn to_f32_sqrt(self) -> f32 {
        let f = self.to_f32();
        f * f
    }

    /// Quantize a `[0, 1]` float (nearest, half away from zero).
    #[inline(always)]
    pub fn from_f32(f: f32) -> Self {
        debug_assert!(f.is_finite() && (0.0..=1.0).contains(&f));
        Density((f * Self::MAX).round() as u16)
    }

    /// Quantize with stochastic rounding: `rand` is a uniform `[0, 1)` draw.
    #[inline(always)]
    pub fn from_f32_sr(f: f32, rand: f32) -> Self {
        debug_assert!(f.is_finite() && (0.0..=1.0).contains(&f));
        debug_assert!((0.0..1.0).contains(&rand));
        let v = f * Self::MAX;
        let floor = v.floor();
        Density(if rand < v - floor { floor + 1.0 } else { floor } as u16)
    }
}

impl Vec3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3([x, y, z])
    }

    #[inline(always)]
    pub fn dot(self, o: Vec3) -> f32 {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }

    #[inline(always)]
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn add(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, s: f32) -> Vec3 {
        Vec3([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }
}

/// Declares the typed channel table. Each `field: Type` line adds:
/// a `ChannelId` variant, a table field, `get_/get_*_mut/set_*` accessors, and
/// a `ChannelRef`/`ChannelRefMut` variant. The element type must be a single
/// identifier (the newtypes above are).
#[rustfmt::skip]
macro_rules! channel_table {
    (
        $(
            $field:ident : $ty:ident,
        )+
    ) => {
        /// Runtime identifier of a channel (for iteration, reset, debug).
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum ChannelId {
            $(
                $ty,
            )+
        }

        impl ChannelId {
            pub fn name(self) -> &'static str {
                match self {
                    $(
                        Self::$ty => stringify!($ty),
                    )+
                }
            }
        }

        /// Borrowed view of one channel, type-erased to an enum for iteration.
        pub enum ChannelRef<'a, const BSX: usize, const N: usize> {
            $(
                $ty(&'a SparseGrid<$ty, BSX, N>),
            )+
        }

        /// Mutable borrowed view of one channel.
        pub enum ChannelRefMut<'a, const BSX: usize, const N: usize> {
            $(
                $ty(&'a mut SparseGrid<$ty, BSX, N>),
            )+
        }

        impl<'a, const BSX: usize, const N: usize> ChannelRef<'a, BSX, N> {
            pub fn id(&self) -> ChannelId {
                match self {
                    $(
                        Self::$ty(_) => ChannelId::$ty,
                    )+
                }
            }
        }

        impl<'a, const BSX: usize, const N: usize> ChannelRefMut<'a, BSX, N> {
            pub fn id(&self) -> ChannelId {
                match self {
                    $(
                        Self::$ty(_) => ChannelId::$ty,
                    )+
                }
            }
        }

        /// A typed, single-level channel table. Owns one `SparseGrid` per
        /// channel; grids are built externally and inserted with `set_*`.
        pub struct ChannelTable<const BSX: usize, const N: usize> {
            $(
                $field: Option<SparseGrid<$ty, BSX, N>>,
            )+
        }

        impl<const BSX: usize, const N: usize> ChannelTable<BSX, N> {
            pub fn new() -> Self {
                Self {
                    $(
                        $field: None,
                    )+
                }
            }

            $(
                paste::paste! {
                    pub fn [<get_ $field>](&self) -> Option<&SparseGrid<$ty, BSX, N>> {
                        self.$field.as_ref()
                    }

                    pub fn [<get_ $field _mut>](&mut self) -> Option<&mut SparseGrid<$ty, BSX, N>> {
                        self.$field.as_mut()
                    }

                    pub fn [<set_ $field>](&mut self, grid: SparseGrid<$ty, BSX, N>) {
                        self.$field = Some(grid);
                    }
                }
            )+

            pub fn get(&self, id: ChannelId) -> Option<ChannelRef<'_, BSX, N>> {
                match id {
                    $(
                        ChannelId::$ty => self.$field.as_ref().map(ChannelRef::$ty),
                    )+
                }
            }

            pub fn get_mut(&mut self, id: ChannelId) -> Option<ChannelRefMut<'_, BSX, N>> {
                match id {
                    $(
                        ChannelId::$ty => self.$field.as_mut().map(ChannelRefMut::$ty),
                    )+
                }
            }

            pub fn contains(&self, id: ChannelId) -> bool {
                match id {
                    $(
                        ChannelId::$ty => self.$field.is_some(),
                    )+
                }
            }

            pub fn len(&self) -> usize {
                0usize $( + usize::from(self.$field.is_some()) )+
            }

            pub fn is_empty(&self) -> bool {
                $( self.$field.is_none() )&&+
            }

            pub fn remove(&mut self, id: ChannelId) {
                match id {
                    $(
                        ChannelId::$ty => self.$field = None,
                    )+
                }
            }

            pub fn clear(&mut self) {
                $(
                    self.$field = None;
                )+
            }

            pub fn for_each<'a>(&'a self, mut f: impl FnMut(ChannelRef<'a, BSX, N>)) {
                $(
                    if let Some(g) = self.$field.as_ref() {
                        f(ChannelRef::$ty(g));
                    }
                )+
            }

            pub fn for_each_mut<'a>(&'a mut self, mut f: impl FnMut(ChannelRefMut<'a, BSX, N>)) {
                $(
                    if let Some(g) = self.$field.as_mut() {
                        f(ChannelRefMut::$ty(g));
                    }
                )+
            }

            pub fn iter<'a>(&'a self) -> impl Iterator<Item = ChannelRef<'a, BSX, N>> + 'a {
                let mut out = Vec::new();
                self.for_each(|c| out.push(c));
                out.into_iter()
            }

            pub fn iter_mut<'a>(
                &'a mut self,
            ) -> impl Iterator<Item = ChannelRefMut<'a, BSX, N>> + 'a {
                let mut out = Vec::new();
                self.for_each_mut(|c| out.push(c));
                out.into_iter()
            }
        }

        impl<const BSX: usize, const N: usize> Default for ChannelTable<BSX, N> {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

channel_table! {
    density:    Density,
    velocity:   Velocity,
    pressure:   Pressure,
    cell_flags: CellFlags,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockpool::BlockPool;
    use std::sync::Arc;

    const BSX: usize = 16;
    const N: usize = 4096;

    fn grid<D: Copy + Default + Send + Sync>(
        name: &str,
        empty: D,
        full: D,
    ) -> SparseGrid<D, BSX, N> {
        let pool = Arc::new(BlockPool::<D, BSX, N>::new(16, 16));
        SparseGrid::new(name.to_string(), 32, 32, 32, empty, full, pool)
    }

    // Boundary: dequant endpoints must map exactly.
    #[test]
    fn test_density_01_endpoints() {
        assert_eq!(Density(0).to_f32(), 0.0);
        assert_eq!(Density(u16::MAX).to_f32(), 1.0);
    }

    // Round-trip at awkward midpoints, within one quantization step.
    #[test]
    fn test_density_02_roundtrip() {
        assert_eq!(Density::from_f32(0.0), Density(0));
        assert_eq!(Density::from_f32(1.0), Density(u16::MAX));
        let v = Density::from_f32(0.5);
        assert!((v.to_f32() - 0.5).abs() < 1.0 / Density::MAX);
    }

    // sqrt-compression decode is the square of the linear decode.
    #[test]
    fn test_density_03_sqrt_decode() {
        let v = Density::from_f32(0.5).to_f32_sqrt();
        assert!((v - 0.25).abs() < 1e-3);
    }

    // Stochastic rounding: small `rand` rounds up, large `rand` rounds down.
    #[test]
    fn test_density_04_stochastic_round() {
        assert_eq!(Density::from_f32_sr(0.5, 0.0).0, 32768);
        assert_eq!(Density::from_f32_sr(0.5, 0.9).0, 32767);
    }

    // Out-of-range input must trip the debug assertion.
    #[test]
    #[should_panic]
    fn test_density_05_out_of_range_panics() {
        let _ = Density::from_f32(1.5);
    }

    // Awkward non-axis-aligned vector: len/dot/add/scale.
    #[test]
    fn test_vec3_01_non_axis_ops() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert_eq!(v.dot(v), 9.0);
        assert_eq!(v.len(), 3.0);
        assert_eq!(v + v, Vec3::new(2.0, 4.0, 4.0));
        assert_eq!(v * 2.0, Vec3::new(2.0, 4.0, 4.0));
    }

    // Empty table: nothing present.
    #[test]
    fn test_table_01_empty() {
        let t = ChannelTable::<BSX, N>::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!(!t.contains(ChannelId::Density));
        assert!(t.get_density().is_none());
    }

    // Set/remove one channel flips presence without touching others.
    #[test]
    fn test_table_02_set_remove() {
        let mut t = ChannelTable::<BSX, N>::new();
        t.set_density(grid("d", Density(0), Density(u16::MAX)));
        assert!(t.contains(ChannelId::Density));
        assert_eq!(t.len(), 1);
        assert!(t.get_pressure().is_none());
        t.remove(ChannelId::Density);
        assert!(t.is_empty());
    }

    // for_each visits exactly the set channels, with the right identity.
    #[test]
    fn test_table_03_for_each_visits_set() {
        let mut t = ChannelTable::<BSX, N>::new();
        t.set_density(grid("d", Density(0), Density(u16::MAX)));
        t.set_cell_flags(grid("cf", CellFlags(0), CellFlags(1)));

        let d_ptr = t.get_density().unwrap() as *const _;
        let mut seen = Vec::new();
        t.for_each(|c| {
            if c.id() == ChannelId::Density {
                assert_eq!(c.id(), ChannelId::Density);
                assert!(matches!(c, ChannelRef::Density(_)));
                // Same grid object we inserted.
                assert!(std::ptr::eq(
                    match c {
                        ChannelRef::Density(g) => g as *const _,
                        _ => unreachable!(),
                    },
                    d_ptr,
                ));
            }
            seen.push(c.id());
        });
        seen.sort_by_key(|id| id.name());
        assert_eq!(seen, vec![ChannelId::CellFlags, ChannelId::Density]);
    }

    // Mutation through get_*_mut is visible through get_*.
    #[test]
    fn test_table_04_get_mut_mutates() {
        let mut t = ChannelTable::<BSX, N>::new();
        t.set_pressure(grid("p", Pressure(0.0), Pressure(1.0)));
        t.get_pressure_mut().unwrap().set_voxel(3, 3, 3, Pressure(7.0));
        assert_eq!(t.get_pressure().unwrap().get_voxel(3, 3, 3), Pressure(7.0));
    }
}

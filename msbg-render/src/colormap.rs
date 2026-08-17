//! Scalar-density colormaps: greyscale and a polynomial `turbo`.

use image::Rgba;

#[inline(always)]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Linear greyscale: `0 → black`, `1 → white`.
pub fn greyscale(t: f32) -> Rgba<u8> {
    let v = to_u8(t);
    Rgba([v, v, v, 255])
}

/// Inigo Quilez's polynomial approximation of the Google `turbo` colormap.
#[allow(clippy::excessive_precision)] // GLSL float literals; extra digits carry no info in f32
pub fn turbo(t: f32) -> Rgba<u8> {
    let x = t.clamp(0.0, 1.0);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let x5 = x4 * x;

    let r = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3
        - 152.94239396 * x4 + 59.28637943 * x5;
    let g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3
        + 4.27729857 * x4 + 2.82956604 * x5;
    let b = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3
        - 89.90310912 * x4 + 27.34824973 * x5;

    Rgba([to_u8(r), to_u8(g), to_u8(b), 255])
}

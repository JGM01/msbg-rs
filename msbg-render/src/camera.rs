//! Pinhole camera + ray generation, native `Vec3` (no `glam`).
//!
//! Positions are in *voxel space* (the grid is `[0, sx]×[0, sy]×[0, sz]`); to
//! place the camera with the demo's normalized `[0,1]³` coordinates, scale each
//! component by the corresponding grid extent. The basis and NDC mapping match
//! the C++ `RaymarchRenderer::setCamera` / ray loop bit-for-bit.

use msbg_rs::channel::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    /// Zoom factor (dimensionless), the C++ `focalLen`.
    pub focal_len: f32,
}

pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )
}

pub(crate) fn normalize(v: Vec3) -> Vec3 {
    let len = v.len();
    if len > 1e-20 {
        v * (1.0 / len)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    }
}

impl Camera {
    /// Build the camera basis exactly like C++ (`worldUp = +Y`,
    /// `right = normalize(cross(forward, worldUp))`, `up = cross(right, forward)`).
    pub fn look_at(position: Vec3, look_at: Vec3, focal_len: f32) -> Self {
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let forward = normalize(look_at - position);
        let right = normalize(cross(forward, world_up));
        let up = cross(right, forward);
        Camera { position, forward, right, up, focal_len }
    }

    /// Primary ray for pixel `(x, y)` of a `width × height` image.
    pub fn ray(&self, x: f32, y: f32, width: f32, height: f32) -> Ray {
        let inv_w = 1.0 / width;
        let inv_h = 1.0 / height;
        let aspect = width / height;
        let ndc_x = ((x + 0.5) * inv_w - 0.5) * 2.0 * aspect;
        let ndc_y = (0.5 - (y + 0.5) * inv_h) * 2.0;
        let dir = normalize(self.forward * self.focal_len + self.right * ndc_x + self.up * ndc_y);
        Ray { origin: self.position, direction: dir }
    }
}

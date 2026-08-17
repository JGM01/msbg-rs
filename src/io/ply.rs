//! PLY point-cloud loading.
//!
//! The demo's `bun_zipper*.ply` files are `format ascii` with five float
//! properties per vertex (`x y z confidence intensity`) plus a `face` element
//! that is skipped entirely. We read the header once, then stream exactly
//! `element vertex N` payload lines, capturing only x/y/z.
//!
//! The parser is generic (`ply-rs` `PropertyAccess`): the crate's `peg` header
//! grammar is robust against malformed headers, and only the declared
//! properties are materialized, so faces and unused scalars cost nothing.
//!
//! Known `ply-rs` limitation: its ASCII payload grammar accepts `-4.2e3` but
//! not uppercase `E`, and requires digits after the decimal point (`.5` /
//! `5.` fail). The shipped bunny files use plain decimals; a hand-rolled
//! scanner is the fallback if a data source trips this.

use std::io::{self, BufRead, Cursor};

use ply_rs::parser::Parser;
use ply_rs::ply::{
    Encoding, Property, PropertyAccess, PropertyDef, PropertyType, ScalarType,
};

/// Vertices plus the axis-aligned bounding box computed during the scan.
#[derive(Debug, Clone, Default)]
pub struct LoadedParticles {
    pub positions: Vec<[f32; 3]>,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
}

/// Captures one vertex's x/y/z; all other properties are ignored.
#[derive(Debug, Clone, Copy, Default)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

impl PropertyAccess for Vertex {
    fn new() -> Self {
        Vertex::default()
    }

    fn set_property(&mut self, name: String, p: Property) {
        let v = match p {
            Property::Float(f) => Some(f),
            Property::Double(d) => Some(d as f32),
            _ => None,
        };
        if let Some(v) = v {
            match name.as_str() {
                "x" => self.x = v,
                "y" => self.y = v,
                "z" => self.z = v,
                _ => {}
            }
        }
    }
}

/// An x/y/z-only element definition (skip confidence/intensity tokens).
fn xyz_def() -> ply_rs::ply::ElementDef {
    use ply_rs::ply::Addable;
    let mut e = ply_rs::ply::ElementDef::new("vertex".to_string());
    e.count = 0;
    let scalar = |s: &str| PropertyDef::new(s.to_string(), PropertyType::Scalar(ScalarType::Float));
    e.properties.add(scalar("x"));
    e.properties.add(scalar("y"));
    e.properties.add(scalar("z"));
    e
}

/// Load all vertices from a PLY byte buffer.
///
/// Returns `Err` for a malformed header, a missing `vertex` element, or
/// unterminated payload. An empty `vertex` element yields `positions == []`
/// with `bbox_min > bbox_max` (matching the C++ `readVerticesFromPLY`
/// convention); callers must guard against a zero span.
pub fn load_vertices(bytes: &[u8]) -> io::Result<LoadedParticles> {
    let mut buf = Cursor::new(bytes);
    let parser = Parser::<Vertex>::new();
    let header = parser.read_header(&mut buf)?;

    let vertex_def = header
        .elements
        .get("vertex")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "PLY has no 'vertex' element"))?;
    let count = vertex_def.count;

    let mut positions = Vec::with_capacity(count);

    match header.encoding {
        Encoding::Ascii => {
            let def = xyz_def();
            let mut line = String::new();
            for _ in 0..count {
                line.clear();
                let n = buf.read_line(&mut line)?;
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "PLY payload ended before all vertices were read",
                    ));
                }
                let v = parser.read_ascii_element(&line, &def)?;
                positions.push([v.x, v.y, v.z]);
            }
        }
        Encoding::BinaryLittleEndian | Encoding::BinaryBigEndian => {
            let little = matches!(header.encoding, Encoding::BinaryLittleEndian);
            for _ in 0..count {
                let v = if little {
                    parser.read_little_endian_element(&mut buf, vertex_def)?
                } else {
                    parser.read_big_endian_element(&mut buf, vertex_def)?
                };
                positions.push([v.x, v.y, v.z]);
            }
        }
    }

    let mut bbox_min = [f32::INFINITY; 3];
    let mut bbox_max = [f32::NEG_INFINITY; 3];
    for p in &positions {
        for k in 0..3 {
            bbox_min[k] = bbox_min[k].min(p[k]);
            bbox_max[k] = bbox_max[k].max(p[k]);
        }
    }

    Ok(LoadedParticles {
        positions,
        bbox_min,
        bbox_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ply(vertices: &str) -> String {
        format!(
            "ply\nformat ascii 1.0\ncomment zipper output\nelement vertex {}\n\
             property float x\nproperty float y\nproperty float z\n\
             property float confidence\nproperty float intensity\n\
             element face 1\nproperty list uchar int vertex_indices\n\
             end_header\n{}",
            vertices.lines().count(),
            vertices
        )
    }

    #[test]
    fn ply_01_happy_5_props_and_faces() {
        let ply = sample_ply("0.1 0.2 0.3 0.9 0.5\n-1.0 2.0 3.5 0.5 0.5\n");
        let l = load_vertices(ply.as_bytes()).unwrap();
        assert_eq!(l.positions.len(), 2);
        assert_eq!(l.positions[0], [0.1, 0.2, 0.3]);
        assert_eq!(l.positions[1], [-1.0, 2.0, 3.5]);
        assert_eq!(l.bbox_min, [-1.0, 0.2, 0.3]);
        assert_eq!(l.bbox_max, [0.1, 2.0, 3.5]);
    }

    #[test]
    fn ply_02_scientific_notation() {
        // Lowercase `e` is supported by the peg grammar; uppercase `E` is not
        // (see the module docs). The shipped bunny files use plain decimals.
        let ply = sample_ply("1e-3 2.5e1 -4e-2 0.5 0.5\n");
        let l = load_vertices(ply.as_bytes()).unwrap();
        assert_eq!(l.positions[0], [1e-3, 25.0, -4e-2]);
    }

    #[test]
    fn ply_03_empty_vertex_element() {
        let ply = sample_ply("");
        let l = load_vertices(ply.as_bytes()).unwrap();
        assert!(l.positions.is_empty());
        assert!(l.bbox_min[0] > l.bbox_max[0]);
    }

    #[test]
    fn ply_04_no_face_element() {
        let ply = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n\
                   property float y\nproperty float z\nend_header\n1 2 3\n";
        let l = load_vertices(ply.as_bytes()).unwrap();
        assert_eq!(l.positions, vec![[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn ply_05_crlf_line_endings() {
        let ply = "ply\r\nformat ascii 1.0\r\nelement vertex 1\r\nproperty float x\r\n\
                   property float y\r\nproperty float z\r\nend_header\r\n4 5 6\r\n";
        let l = load_vertices(ply.as_bytes()).unwrap();
        assert_eq!(l.positions, vec![[4.0, 5.0, 6.0]]);
    }

    #[test]
    fn ply_06_missing_end_header_errors() {
        let ply = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n";
        assert!(load_vertices(ply.as_bytes()).is_err());
    }

    #[test]
    fn ply_07_truncated_payload_errors() {
        let ply = sample_ply("1 2 3\n4 5\n"); // second line has only 2 numbers
        assert!(load_vertices(ply.as_bytes()).is_err());
    }

    #[test]
    fn ply_08_malformed_float_errors() {
        let ply = sample_ply("1 2 notanumber 0.5 0.5\n");
        assert!(load_vertices(ply.as_bytes()).is_err());
    }

    #[test]
    fn ply_09_no_vertex_element_errors() {
        let ply = "ply\nformat ascii 1.0\nelement face 1\nproperty list uchar int v\nend_header\n";
        assert!(load_vertices(ply.as_bytes()).is_err());
    }

    #[test]
    fn ply_10_magic_number_missing() {
        let ply = "format ascii 1.0\nend_header\n";
        assert!(load_vertices(ply.as_bytes()).is_err());
    }
}

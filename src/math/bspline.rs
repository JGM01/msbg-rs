//! Uniform cubic B-spline reconstruction weights (Ruijters et al., "Efficient
//! GPU-Based Texture Interpolation using Uniform B-Splines", 2008).

/// Value weights `w[0..3]` at fractional coordinate `t in [0,1]`.
#[inline]
pub const fn cubic_weights(t: f32) -> [f32; 4] {
    let m = 1.0 - t;
    [
        (1.0 / 6.0) * m * m * m,
        2.0 / 3.0 - 0.5 * t * t * (2.0 - t),
        2.0 / 3.0 - 0.5 * m * m * (1.0 + t),
        (1.0 / 6.0) * t * t * t,
    ]
}

/// First-derivative weights `w'[0..3]`.
#[inline]
pub const fn cubic_deriv_weights(t: f32) -> [f32; 4] {
    let m = 1.0 - t;
    [
        -m * m / 2.0,
        0.5 * t * t - (2.0 - t) * t,
        m * (t + 1.0) - 0.5 * m * m,
        t * t / 2.0,
    ]
}

/// Second-derivative weights `w''[0..3]`.
#[inline]
pub const fn cubic_deriv2_weights(t: f32) -> [f32; 4] {
    [1.0 - t, 3.0 * t - 2.0, 1.0 - 3.0 * t, t]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    // Partition of unity across the whole interval.
    #[test]
    fn test_bspl_01_partition_of_unity() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let w = cubic_weights(t);
            assert!(close(w.iter().sum(), 1.0), "sum at t={t}: {:?}", w);
        }
    }

    // Derivative and 2nd-derivative weights sum to zero (constant field has
    // zero gradient/Hessian).
    #[test]
    fn test_bspl_02_deriv_sums_are_zero() {
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            assert!(close(cubic_deriv_weights(t).iter().sum(), 0.0), "d at {t}");
            assert!(close(cubic_deriv2_weights(t).iter().sum(), 0.0), "d2 at {t}");
        }
    }

    // Endpoints: w(t=0) and w(t=1) must be mirror images.
    #[test]
    fn test_bspl_03_endpoint_symmetry() {
        let w0 = cubic_weights(0.0);
        let w1 = cubic_weights(1.0);
        assert!(close(w0[0], w1[3]));
        assert!(close(w0[1], w1[2]));
        assert!(close(w0[2], w1[1]));
        assert!(close(w0[3], w1[0]));
    }
}

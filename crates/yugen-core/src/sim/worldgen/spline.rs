//! Piecewise-linear spline over a monotone-in-t control list.
//!
//! WHY THIS EXISTS AT ALL: `height = noise * amplitude` can only ever produce one
//! landform, because it is a linear map — the output distribution is the input
//! distribution, rescaled. Rolling bumps in, rolling bumps out. Every knob you
//! add (more octaves, more amplitude) makes the SAME shape rougher or bigger.
//!
//! Mapping the noise through an authored curve first decouples "how often" from
//! "how much". A long flat run in the curve turns a whole band of noise values
//! into one height — that band becomes a plain. A steep segment turns a narrow
//! band into a big height change — that becomes a coastal drop or a cliff face,
//! and it is RARE because the band is narrow. A second flat run above it is a
//! plateau. This is the Minecraft 1.18 trick, and it is the single reason the
//! generator can produce distinct landforms from one noise field.
//!
//! Linear rather than cubic on purpose: a cubic spline overshoots between
//! control points, and an overshoot in the continental curve means terrain that
//! dips back under sea level in the middle of a coastline. Corners in the height
//! function are invisible anyway once the relief term is added on top.

/// Control points as `(t, value)` pairs, sorted ascending by `t` and with no
/// duplicate `t`. Not validated at runtime — these are authored constants, and a
/// bad table is a compile-time-visible typo, not a user input.
pub type Spline = [(f64, f64)];

/// Evaluate `s` at `t`, clamping to the end values outside the control range.
///
/// The scan is linear rather than a binary search: every spline here has 6-12
/// control points and is evaluated once per COLUMN (not per cell), so the branch
/// predictor beats the extra arithmetic of a bisection.
#[inline]
pub fn spline(s: &Spline, t: f64) -> f64 {
    let n = s.len();
    if t <= s[0].0 {
        return s[0].1;
    }
    let last = s[n - 1];
    if t >= last.0 {
        return last.1;
    }
    for i in 1..n {
        let b = s[i];
        if t < b.0 {
            let a = s[i - 1];
            let span = b.0 - a.0;
            let u = (t - a.0) / span;
            return a.1 + (b.1 - a.1) * u;
        }
    }
    last.1
}

/// Slope of `s` at `t` (0 outside the control range). The terracing pass uses
/// this to detect "we are on a steep segment" — i.e. a cliff face — without
/// needing to re-derive where the segments are.
#[inline]
pub fn spline_slope(s: &Spline, t: f64) -> f64 {
    let n = s.len();
    if t <= s[0].0 || t >= s[n - 1].0 {
        return 0.0;
    }
    for i in 1..n {
        let b = s[i];
        if t < b.0 {
            let a = s[i - 1];
            return (b.1 - a.1) / (b.0 - a.0);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rising table with a flat run (a plain), a steep riser (a cliff) and a
    /// second flat run (a plateau) — the shape the module doc describes.
    const RISING: &Spline = &[(-1.0, 0.0), (-0.2, 0.1), (0.0, 5.0), (0.6, 5.2), (1.0, 9.0)];
    /// A falling table, because CONTINENTAL is one (rows grow downward).
    const FALLING: &Spline = &[(-1.0, 33.0), (-0.42, 5.0), (0.16, -13.0), (1.0, -36.0)];

    #[test]
    fn evaluating_outside_the_control_range_clamps_to_the_end_values() {
        assert_eq!(spline(RISING, -50.0), 0.0);
        assert_eq!(spline(RISING, -1.0), 0.0);
        assert_eq!(spline(RISING, 1.0), 9.0);
        assert_eq!(spline(RISING, 50.0), 9.0);
        assert_eq!(spline_slope(RISING, -50.0), 0.0);
        assert_eq!(spline_slope(RISING, 50.0), 0.0);
    }

    #[test]
    fn every_control_point_is_hit_exactly() {
        for s in [RISING, FALLING] {
            for &(t, v) in s {
                let got = spline(s, t);
                assert!((got - v).abs() < 1e-12, "spline({t}) = {got}, want {v}");
            }
        }
    }

    #[test]
    fn a_linear_spline_never_overshoots_its_control_points() {
        // THE reason this is linear and not cubic: a cubic would bulge past the
        // bracketing control values, and an overshoot in the continental curve
        // is terrain that dips back under sea level mid-coastline.
        for s in [RISING, FALLING] {
            for i in 1..s.len() {
                let (t0, v0) = s[i - 1];
                let (t1, v1) = s[i];
                let (lo, hi) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                for k in 0..=64 {
                    let t = t0 + (t1 - t0) * (k as f64 / 64.0);
                    let v = spline(s, t);
                    assert!(
                        v >= lo - 1e-12 && v <= hi + 1e-12,
                        "spline({t}) = {v} left [{lo},{hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn a_monotone_table_yields_a_monotone_curve() {
        let mut prev_r = f64::NEG_INFINITY;
        let mut prev_f = f64::INFINITY;
        for k in -1200..=1200 {
            let t = k as f64 / 1000.0;
            let r = spline(RISING, t);
            let f = spline(FALLING, t);
            assert!(r >= prev_r - 1e-12, "RISING dipped at {t}");
            assert!(f <= prev_f + 1e-12, "FALLING rose at {t}");
            prev_r = r;
            prev_f = f;
        }
    }

    #[test]
    fn the_slope_is_the_derivative_of_the_value() {
        // The terracing pass reads `spline_slope` as "how steep is this segment",
        // so it has to be the actual finite difference of `spline`.
        for s in [RISING, FALLING] {
            for k in -990..990 {
                let t = k as f64 / 1000.0;
                let h = 1e-6;
                // Stay off the control points, where the derivative jumps.
                if s.iter().any(|&(ct, _)| (ct - t).abs() < 1e-3) {
                    continue;
                }
                let fd = (spline(s, t + h) - spline(s, t - h)) / (2.0 * h);
                let got = spline_slope(s, t);
                assert!(
                    (fd - got).abs() < 1e-4,
                    "slope({t}) = {got}, finite diff {fd}"
                );
            }
        }
    }

    #[test]
    fn a_flat_run_maps_a_whole_band_of_noise_to_one_height() {
        // The mechanism the module doc is about: the plain.
        assert_eq!(spline_slope(RISING, 0.3), (5.2 - 5.0) / 0.6);
        let a = spline(RISING, 0.05);
        let b = spline(RISING, 0.55);
        assert!((a - b).abs() < 0.2, "the plain is not flat: {a} vs {b}");
        // ...and the cliff right below it moves 4.9 in a fifth of the domain.
        assert!(spline_slope(RISING, -0.1) > 20.0);
    }
}

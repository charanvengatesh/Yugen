//! The low-frequency world fields — the parameters that say WHAT KIND OF PLACE a
//! column is, before anything decides how high the ground is there.
//!
//! All four are pure functions of the absolute world column `wcx`, sampled as
//! horizontal slices of the 2D gradient field at fixed, mutually-irrational
//! anchors. Sampling a 2D field at a constant y instead of using `n1`/`fbm1`
//! costs the same and buys a much better 1D signal: `g2` is zero at every integer
//! x, so its 1D slice still has extrema at irrational positions, while 1D value
//! noise puts an extremum on every integer column and reads as a regular wave.
//!
//! WHY FOUR FIELDS AND NOT ONE TALLER fBm: a single heightmap can only trade
//! "smooth" against "rough". Splitting the decision into independent axes lets
//! the generator express combinations that a single field cannot reach at all —
//! a HIGH, FLAT place (plateau: high continentalness, high erosion) is a
//! different landform from a HIGH, ROUGH one (mountains: high continentalness,
//! low erosion), and both are different from a LOW, ROUGH one (fjord/badlands).
//! With one field, "high" and "rough" are the same number.
//!
//!   continentalness  ocean basin -> shelf -> coast -> inland -> highland
//!   erosion          how much relief the region is allowed at all
//!   peaks_valleys    the local ridged relief riding on top
//!   weirdness        a rarity gate: unusual landforms and chasms only where it spikes
//!
//! Climate (temperature/moisture/volcanism) deliberately stays in `sim::biomes`:
//! it selects a PALETTE, not a shape, it is consumed only by the biome mixer, and
//! it shares that file's tuning constants (CLIMATE_SPREAD, BLEND_WIDTH). Keeping
//! it there also keeps this module free of any dependency on the biome registry,
//! so `heightmap` can import both without a cycle.

use crate::config::WorldScale;
use crate::sim::noise::Noise;

// ---------------------------------------------------------------------------
// Frequencies
// ---------------------------------------------------------------------------
// Every period below is in LEGACY cells — the space on the far side of
// `WorldScale::coord`. At `WorldScale::LIVE` a world period is twice the number
// written here, which is the whole mechanism by which the world got bigger; the
// numbers themselves never move, so each one still means what its comment says.
//
// Chosen against the ~350-cell viewport (WINDOW_COLS) as it was at
// `WorldScale::LEGACY`: a field with period P is "one landform per P/350
// screens". At LIVE the viewport spans half as much world, so read every figure
// below as twice the screens it says. Anchors are large non-integer offsets so
// the four fields slice genuinely different rows of the gradient lattice — an
// integer offset would land on the same lattice row and correlate them.

/// ~1800 cells ~= 5 screens per continent lobe. Long enough to walk inland.
const CONT_FREQ: f64 = 0.00056;
const CONT_ANCHOR: f64 = 83.41;
const CONT_OCTAVES: u32 = 3;

/// ~1100 cells: erosion regions are smaller than continents, so a single
/// landmass can carry both a mountain spine and a flat basin.
const EROS_FREQ: f64 = 0.00091;
const EROS_ANCHOR: f64 = 1471.27;
const EROS_OCTAVES: u32 = 3;

/// ~170 cells for the coarsest ridge, with 4 octaves of detail on top. This is
/// the frequency you actually SEE as hills while walking.
const PV_FREQ: f64 = 0.0059;
const PV_ANCHOR: f64 = 2903.73;
const PV_OCTAVES: u32 = 4;

/// Ridged gain for [`peaks_valleys`] — slightly above `ridged2`'s 2.0 default so
/// the crests stay thin. In TypeScript this was a bare 4th argument at the call
/// site and the only place the gain was ever overridden.
const PV_GAIN: f64 = 2.1;

/// ~2400 cells. Rarity gate — must be slower than everything it gates or the
/// "rare" feature turns into a periodic one.
const WEIRD_FREQ: f64 = 0.00042;
const WEIRD_ANCHOR: f64 = 5417.19;
const WEIRD_OCTAVES: u32 = 2;

/// Domain warp applied to the column BEFORE the continental and erosion fields
/// are sampled. Both then bend along the same wandering line, so coastlines and
/// escarpments meander instead of running as straight noise contours — and,
/// because they share the warp, an escarpment tends to follow the coast rather
/// than cutting across it at a random angle.
///
/// Strength 90 against a 1800-cell continental period is a +/-5% coordinate
/// distortion: enough to break the contour, not enough to fold it back on itself
/// (a fold would make the height function non-monotone in a way the terracing
/// pass reads as noise).
const WARP_STRENGTH: f64 = 90.0;
const WARP_FREQ: f64 = 0.0013;
const WARP_ANCHOR: f64 = 6673.11;

/// Contrast gain applied to the raw fBm before it becomes a world field.
///
/// MEASURED, not assumed: a 2-3 octave gradient fBm slice does NOT fill [-1,1].
/// Over 40k columns these fields span about [-0.58, 0.55] with p50 ~= 0, so a
/// spline authored over [-1,1] — the obvious thing to write — never evaluates
/// its outer thirds at all. Mountains and abyssal trenches simply do not exist,
/// and a rarity gate at 0.7 fires exactly never (which is what happened to the
/// first draft's surface chasms: zero in 40,000 columns).
///
/// 1.8 stretches the observed body onto the full authored domain and clamps the
/// few percent that overshoot, so the tails of every spline become reachable and
/// "put this at t = 0.9" means what it looks like it means.
const FIELD_SPREAD: f64 = 1.8;

/// The clamp is written out longhand rather than delegated to `f64::clamp`,
/// matching the original ternary chain exactly. (`f64::clamp` also panics when
/// its bounds are ordered wrongly, which nothing here wants to risk.)
#[inline]
#[allow(clippy::manual_clamp)]
fn spread(v: f64) -> f64 {
    let s = v * FIELD_SPREAD;
    if s < -1.0 {
        -1.0
    } else if s > 1.0 {
        1.0
    } else {
        s
    }
}

/// The warped sampling coordinate for a column, in LEGACY cells. Pure in `wcx`.
///
/// This is the inward crossing for the whole field layer: the world column is
/// divided by the world scale exactly once, here, and everything downstream —
/// including [`continentalness`] and [`erosion`], which take the result — works
/// in legacy cells and needs no knowledge of the scale at all.
///
/// [`WARP_STRENGTH`] is deliberately NOT scaled. It is a displacement measured in
/// the same space as the coordinate it displaces, and that space is already on
/// the divided side of the crossing; scaling it too would apply the world scale
/// twice and bend the coastlines by ±10% of a doubled period instead of ±5% of
/// the authored one.
#[inline]
pub fn warped_column(noise: &Noise, wcx: i32, scale: WorldScale) -> f64 {
    let lx = scale.coord(wcx);
    lx + WARP_STRENGTH * noise.g2(lx * WARP_FREQ, WARP_ANCHOR)
}

/// Continentalness in [-1,1] after spreading. Low = ocean floor, ~-0.42 = the
/// waterline (see `heightmap`'s `CONTINENTAL`), high = highland. Pass the warped
/// column from [`warped_column`].
#[inline]
pub fn continentalness(noise: &Noise, wx: f64) -> f64 {
    spread(noise.gfbm2(wx * CONT_FREQ, CONT_ANCHOR, CONT_OCTAVES))
}

/// Erosion in ~[-1,1]. HIGH = heavily eroded, therefore FLAT (plains, valley
/// floors, sea bed). LOW = young rock, therefore steep (mountains, cliffs,
/// mesas). Sign follows the geological reading, not the visual one, so the
/// splines that consume it read correctly.
#[inline]
pub fn erosion(noise: &Noise, wx: f64) -> f64 {
    spread(noise.gfbm2(wx * EROS_FREQ, EROS_ANCHOR, EROS_OCTAVES))
}

/// Peaks-and-valleys in [0,1], RIDGED. This is the local relief profile, and it
/// is ridged rather than fBm on purpose: fBm relief is symmetric (as many pits
/// as peaks, all rounded), which reads as a lumpy blanket. Ridged relief has
/// sharp crests over broad smooth basins — the asymmetry real erosion produces,
/// and the thing that makes a skyline.
///
/// Unwarped: warping this too would double-bend the crests against the already
/// warped continental base and smear them into mush.
#[inline]
pub fn peaks_valleys(noise: &Noise, wcx: i32, scale: WorldScale) -> f64 {
    noise.ridged2(scale.coord(wcx) * PV_FREQ, PV_ANCHOR, PV_OCTAVES, PV_GAIN)
}

/// Weirdness in ~[-1,1] — a slow rarity field. Nothing reads it as a magnitude;
/// everything reads it as "am I past the threshold", so unusual landforms cluster
/// into a few regions per world instead of sprinkling uniformly. Surface chasms
/// and ravine density gate on this.
#[inline]
pub fn weirdness(noise: &Noise, wcx: i32, scale: WorldScale) -> f64 {
    spread(noise.gfbm2(scale.coord(wcx) * WEIRD_FREQ, WEIRD_ANCHOR, WEIRD_OCTAVES))
}

// ---------------------------------------------------------------------------
// Shared shaping helpers
// ---------------------------------------------------------------------------

/// Clamp to [0,1].
#[inline]
#[allow(clippy::manual_clamp)]
pub fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// Hermite ease of a value already normalised to [0,1]. The clamp is written out
/// rather than delegated to [`clamp01`]: this runs several times per generated
/// cell and the call showed up at 4% of chunk-generation time in a CPU profile.
#[inline]
#[allow(clippy::manual_clamp)]
pub fn smoothstep01(v: f64) -> f64 {
    let t = if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    };
    t * t * (3.0 - 2.0 * t)
}

/// Eased ramp from 0 at `a` to 1 at `b`. Works for a > b (a falling ramp).
#[inline]
pub fn smooth_ramp(a: f64, b: f64, v: f64) -> f64 {
    smoothstep01((v - a) / (b - a))
}

/// Linear interpolation. `t` is not clamped.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SEED, WorldScale};

    #[test]
    fn spread_widens_and_clamps() {
        assert_eq!(spread(0.0), 0.0);
        assert!((spread(0.5) - 0.9).abs() < 1e-12);
        assert_eq!(spread(0.9), 1.0);
        assert_eq!(spread(-0.9), -1.0);
    }

    #[test]
    fn every_field_stays_inside_its_advertised_range() {
        let n = Noise::new(SEED);
        for wcx in -2000..2000 {
            let wx = warped_column(&n, wcx, WorldScale::LEGACY);
            let c = continentalness(&n, wx);
            let e = erosion(&n, wx);
            let pv = peaks_valleys(&n, wcx, WorldScale::LEGACY);
            let w = weirdness(&n, wcx, WorldScale::LEGACY);
            assert!((-1.0..=1.0).contains(&c), "continentalness {c} at {wcx}");
            assert!((-1.0..=1.0).contains(&e), "erosion {e} at {wcx}");
            assert!((0.0..=1.0).contains(&pv), "peaks_valleys {pv} at {wcx}");
            assert!((-1.0..=1.0).contains(&w), "weirdness {w} at {wcx}");
        }
    }

    #[test]
    fn the_warp_never_exceeds_its_stated_strength() {
        // The doc claims a +/-5% distortion of a 1800-cell continental period.
        // That only holds if the displacement is bounded by WARP_STRENGTH.
        let n = Noise::new(SEED);
        for wcx in -3000..3000 {
            let d = warped_column(&n, wcx, WorldScale::LEGACY) - wcx as f64;
            assert!(d.abs() <= WARP_STRENGTH, "warp displaced {d} at {wcx}");
        }
    }

    #[test]
    fn the_spread_actually_reaches_the_authored_tails() {
        // The whole justification for FIELD_SPREAD: without it the fields never
        // leave their middle third and the splines' outer thirds are dead code.
        let n = Noise::new(SEED);
        let mut hi = 0usize;
        for wcx in -20000..20000 {
            let wx = warped_column(&n, wcx, WorldScale::LEGACY);
            if continentalness(&n, wx).abs() > 0.7 {
                hi += 1;
            }
        }
        assert!(hi > 0, "continentalness never reached the authored tail");
    }

    #[test]
    fn peaks_valleys_lives_where_the_relief_curve_is_authored() {
        // RELIEF_SHAPE (heightmap.rs) clusters its control points in
        // [0.30, 0.85] because that is where the measured field sits:
        // p05 = 0.21, p50 = 0.56, p95 = 0.83.
        let n = Noise::new(SEED);
        let mut s: Vec<f64> = (-20000..20000)
            .map(|x| peaks_valleys(&n, x, WorldScale::LEGACY))
            .collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = s[s.len() / 2];
        assert!((0.35..0.75).contains(&p50), "peaks_valleys p50 = {p50}");
    }

    #[test]
    fn shaping_helpers_agree_with_their_definitions() {
        assert_eq!(clamp01(-3.0), 0.0);
        assert_eq!(clamp01(3.0), 1.0);
        assert_eq!(smoothstep01(0.0), 0.0);
        assert_eq!(smoothstep01(1.0), 1.0);
        assert_eq!(smoothstep01(0.5), 0.5);
        // A falling ramp (a > b) still reads 0 at `a` and 1 at `b`.
        assert_eq!(smooth_ramp(10.0, 0.0, 10.0), 0.0);
        assert_eq!(smooth_ramp(10.0, 0.0, 0.0), 1.0);
        assert_eq!(smooth_ramp(10.0, 0.0, 20.0), 0.0);
        assert_eq!(lerp(2.0, 6.0, 0.25), 3.0);
    }
}

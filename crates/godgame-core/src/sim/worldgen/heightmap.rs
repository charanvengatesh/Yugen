//! The heightmap: absolute cell ROW of the ground surface for a world column.
//!
//! The old rule was `ANCHOR + fbm1(x*0.03, 4) * amplitude`. That is a linear map
//! of one noise field, so it can only ever produce one landform — rolling bumps,
//! everywhere, forever. This version composes three authored curves instead:
//!
//! ```text
//!   height = CONTINENTAL(continentalness)              // where the land sits
//!          + RELIEF(peaks_valleys) * EROSION(erosion)  // how much it moves
//! ```
//!
//! with the sampling column domain-warped first. The curves are what create
//! distinct places: a long flat run in CONTINENTAL is a coastal plain, the steep
//! segment right after it is an escarpment that is RARE because the noise band
//! feeding it is narrow, and EROSION independently decides whether the region is
//! allowed relief at all — so a high plateau and a mountain range come out of the
//! same continental value and read as completely different terrain.
//!
//! PURITY: every input is a function of `wcx` and the seed. Nothing here looks at
//! a neighbouring column's decision, so two chunks meeting at a seam compute the
//! identical ground line by computing it twice. The memo below is a pure memo —
//! it changes cost, never a value.

use crate::config::{SEA_LEVEL_Y, SHORE_BAND, SURFACE_AMPLITUDE, SURFACE_ANCHOR_Y};
use crate::sim::biomes::{ColumnProfile, height_params_at};
use crate::sim::noise::Noise;

use super::fields::{continentalness, erosion, lerp, peaks_valleys, smooth_ramp, warped_column};
use super::spline::{Spline, spline};

/// Re-exported so callers can clamp against the same [0,1] helper.
pub use super::fields::clamp01;

// ---------------------------------------------------------------------------
// Landform curves
// ---------------------------------------------------------------------------

/// continentalness -> row offset from [`SURFACE_ANCHOR_Y`] (POSITIVE IS DOWN).
///
/// Sea level is +6 from the anchor. Read the table against that number:
///   - `> +6`  submerged (worldgen fills the gap with water)
///   - `~ +6`  the waterline itself
///   - `< +6`  dry land
///
/// The segment from -0.50 to -0.34 spans 9 -> 1: that is the beach, and it is
/// DELIBERATELY the steepest low segment so the shoreline is narrow and you get a
/// proper coast rather than a hundred cells of tidal flat. The run from -0.02 to
/// +0.16 is the inland plain — the most common terrain in the world, because the
/// field's median sits in it and it is mapped to the least height change. Above
/// +0.4 the curve steepens again into highland and mountain: rare, because only
/// the field's tail reaches there. Measured against the SPREAD continentalness
/// (`worldgen::fields`), the waterline at -0.42 puts ~17% of columns under water.
const CONTINENTAL: &Spline = &[
    (-1.0, 33.0), // abyssal floor
    (-0.82, 25.0),
    (-0.62, 16.0),
    (-0.5, 9.0),  // shelf, still under water
    (-0.42, 5.0), // waterline
    (-0.34, 1.0), // beach berm
    (-0.2, -4.0),
    (-0.02, -9.0),
    (0.16, -13.0), // inland plain — the world's default
    (0.4, -18.0),
    (0.62, -25.0),
    (0.84, -32.0), // highland
    (1.0, -36.0),  // mountain root
];

/// erosion -> relief amplitude multiplier in [0,1].
///
/// Note the shape: it is NOT linear. Most of the field's mass sits near zero, and
/// that region maps to ~0.68 — enough relief for readable hills. Only the low tail
/// opens up to full amplitude, which is why mountain ranges are uncommon; only the
/// high tail collapses to 0.05, which is what makes a plain genuinely FLAT instead
/// of merely less bumpy. A linear map here would give every column mid-relief and
/// the world would read as uniform again.
const EROSION_RELIEF: &Spline = &[
    (-1.0, 1.0),
    (-0.55, 0.92),
    (-0.25, 0.82),
    (0.0, 0.68),
    (0.22, 0.44),
    (0.45, 0.2),
    (0.7, 0.09),
    (1.0, 0.05),
];

/// peaks_valleys (ridged, [0,1]) -> relief shape. Positive lifts the ground.
///
/// The negative tail matters as much as the positive one: without it, relief only
/// ever ADDS height and the terrain becomes a field of bumps sitting on a plane.
/// Letting the low band dig below the base carves valleys between the ridges, and
/// that is what puts inland lake basins under sea level without needing a
/// separate lake system. It is deliberately SHALLOWER than the positive side
/// (-0.55 against +1.3): a symmetric curve drowns a third of the world.
///
/// The control points cluster in [0.30, 0.85] because that is where the ridged
/// field actually LIVES: `peaks_valleys` measures p05 = 0.21, p50 = 0.56,
/// p95 = 0.83. Spreading the curve evenly over [0,1] — the obvious thing to
/// write — maps that whole bulk onto a few hundredths of the output and produces
/// a world flat to within two cells everywhere, which is exactly what the first
/// draft did.
const RELIEF_SHAPE: &Spline = &[
    (0.0, -0.55),
    (0.3, -0.4),
    (0.45, -0.18),
    (0.58, 0.1),
    (0.7, 0.5),
    (0.82, 0.92),
    (1.0, 1.3),
];

/// erosion -> terracing weight in [0,1].
///
/// Terracing quantises the height to a shelf grid. It peaks at MODERATELY low
/// erosion rather than at the extreme: fully un-eroded terrain is a jagged
/// mountain (stepping it just looks like a staircase), and fully eroded terrain
/// is a plain (there is nothing to step). The interesting case is in between —
/// a landform with real height that has been planed off — which is a mesa.
const TERRACE: &Spline = &[
    (-1.0, 0.42),
    (-0.62, 0.72),
    (-0.3, 0.6),
    (-0.05, 0.3),
    (0.25, 0.08),
    (0.6, 0.0),
];

/// Shelf height in cells. ~2 player heights: readable as a ledge, climbable.
const TERRACE_STEP: f64 = 6.0;
/// Phase field for the shelf grid, so terraces are not globally coplanar.
const TERRACE_PHASE_FREQ: f64 = 0.00071;
const TERRACE_PHASE_ANCHOR: f64 = 311.73;

/// Below this the terrace blend is skipped entirely — at a weight this small the
/// snap moves the ground line by well under a cell and the work is wasted.
const TERRACE_MIN_W: f64 = 0.01;

/// The relief term is scaled by this before [`SURFACE_AMPLITUDE`]. At 1.0 a
/// typical inland column gets +/-10 cells of local relief (a readable hill at 5px
/// cells) and a low-erosion mountain column stacks up to +30 on top of the
/// continental lift. Going higher makes peaks that the parallax and the lighting's
/// surface-row cache were never tuned for; going lower — the first draft used
/// 0.85 against a much shallower erosion curve — flattens the world to a table.
const RELIEF_SCALE: f64 = 1.0;

/// JavaScript's `Math.round`: halves go UP (toward +infinity), so `-2.5` rounds
/// to `-2`. Rust's `f64::round` goes away from zero and would give `-3`.
///
/// This is not pedantry. The ground row is `round(height)` and the terrace snap
/// is `round(h / step)`, both of which land on an exact half often enough at
/// negative rows and negative columns that using the wrong rule would tilt the
/// whole world west of the origin by a cell.
#[inline]
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

// ---------------------------------------------------------------------------
// Terrain parameters — the seam a feature pass reads
// ---------------------------------------------------------------------------

/// Everything the heightmap decided about a column.
///
/// The TypeScript version filled a module-level scratch record and handed out a
/// reference to it, so callers had to copy what they needed before the next call.
/// Eight `f64`-sized fields go back in registers here, so this is returned BY
/// VALUE and the aliasing hazard — and the comment warning about it — are gone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainParams {
    /// Warped sampling column — pass to `continentalness`/`erosion` to re-derive.
    pub wx: f64,
    pub continental: f64,
    pub erosion: f64,
    pub peaks_valleys: f64,
    /// Fractional ground row, before rounding.
    pub height: f64,
    /// Integer ground row — what [`Heightmap::surface_row_at`] returns.
    pub surf: i32,
    /// 1 where the column is at/near/below sea level, 0 well inland or uphill.
    pub shore: f64,
    /// True when the ground line is below sea level: this column is under water.
    pub submerged: bool,
}

/// The subset of [`TerrainParams`] chunk generation reads per column.
///
/// Replaces the `out` record the TypeScript `surfaceDetailAt` wrote through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceDetail {
    pub surf: i32,
    pub shore: f64,
    pub submerged: bool,
}

/// Shore weight for a ground row: 1 inside [`SHORE_BAND`] of sea level (and for
/// everything below it), easing to 0 over the band above. Drives the sand/gravel
/// beach cap. Continuous in the height, so the beach fades out along the coast
/// rather than ending on a column boundary.
#[inline]
pub fn shore_weight_at(surf: i32) -> f64 {
    if surf >= SEA_LEVEL_Y {
        return 1.0;
    }
    smooth_ramp(
        (SEA_LEVEL_Y - SHORE_BAND) as f64,
        (SEA_LEVEL_Y - 1) as f64,
        surf as f64,
    )
}

/// Snap `h` onto the shelf grid offset by `phase`. Split out of `compute_params`
/// only so a test can assert the thing that matters: the result minus the phase
/// is an exact multiple of [`TERRACE_STEP`].
#[inline]
fn terrace_snap(h: f64, phase: f64) -> f64 {
    js_round((h - phase) / TERRACE_STEP) * TERRACE_STEP + phase
}

/// Full terrain parameters for a column. `amp_scale`/`height_offset` come from
/// the blended biome profile so biome character survives the new pipeline: a
/// Swamp still flattens (amp_scale 0.45) and a Glacier still juts (1.45), but
/// they now modulate an authored landform instead of being the only shaping
/// there is.
fn compute_params(noise: &Noise, wcx: i32, amp_scale: f64, height_offset: f64) -> TerrainParams {
    let wx = warped_column(noise, wcx);
    let c = continentalness(noise, wx);
    let e = erosion(noise, wx);
    let pv = peaks_valleys(noise, wcx);

    let base = spline(CONTINENTAL, c);
    let relief_amp = spline(EROSION_RELIEF, e);
    // Relief is a LIFT, and rows grow downward, so it is subtracted.
    let relief = spline(RELIEF_SHAPE, pv) * relief_amp * SURFACE_AMPLITUDE as f64 * RELIEF_SCALE;

    let mut h = SURFACE_ANCHOR_Y as f64 + base + height_offset - relief * amp_scale;

    // --- Terracing ---------------------------------------------------------
    // Snap the height onto a shelf grid, blended in by erosion. The blend is what
    // keeps it from reading as a hard switch: at weight 0.6 a column is 60% of the
    // way onto its shelf, so shelf tops are near-flat while the risers between
    // them keep a little slope and the transition out of the terraced region is
    // invisible. Sea beds are excluded — stepped ocean floors look like a bug.
    let land_w = smooth_ramp((SEA_LEVEL_Y + 2) as f64, (SEA_LEVEL_Y - 10) as f64, h);
    let terrace_w = spline(TERRACE, e) * land_w;
    if terrace_w > TERRACE_MIN_W {
        let phase = TERRACE_STEP
            * (0.5 + 0.5 * noise.g2(wcx as f64 * TERRACE_PHASE_FREQ, TERRACE_PHASE_ANCHOR));
        let q = terrace_snap(h, phase);
        h = lerp(h, q, terrace_w);
    }

    // Round the SUM, not just the heightmap: `height_offset` is a weighted blend
    // of two biomes' offsets and is therefore fractional, and every caller
    // (trees, lighting, spawn) needs an integer cell row.
    let surf = js_round(h) as i32;
    TerrainParams {
        wx,
        continental: c,
        erosion: e,
        peaks_valleys: pv,
        height: h,
        surf,
        shore: shore_weight_at(surf),
        submerged: surf > SEA_LEVEL_Y,
    }
}

/// Terrain parameters at an absolute column. Pure in `wcx`. This is the seam a
/// feature/structure pass should read — it gets the ground line, whether the
/// column is coastal or submerged, and the raw landform fields, without
/// re-deriving any of the splines.
pub fn terrain_params_at(noise: &Noise, wcx: i32) -> TerrainParams {
    let p = height_params_at(noise, wcx);
    compute_params(noise, wcx, p.amp_scale, p.height_offset)
}

// ---------------------------------------------------------------------------
// Memo
// ---------------------------------------------------------------------------

/// Direct-mapped cache of [`Heightmap::surface_row_at`], keyed by column.
///
/// Decorators call `ctx.surface_at(wcx)` for the SAME columns repeatedly — the
/// tree pass alone probes every candidate origin and then several columns either
/// side of it — and lighting calls it per light column per frame. The full path
/// is one warp sample plus three multi-octave fBms plus a biome weighting; the
/// memo turns the repeat calls into an array read.
///
/// It is a PURE memo: the surface row is a function of (seed, wcx) alone, so a
/// hit returns exactly what a miss would have computed. Determinism is untouched
/// — the cache cannot change a value, only who pays for it. 4096 entries covers
/// ~11 chunks of columns, which is more than any single decorator pass touches.
///
/// `stamp` invalidates the whole table in O(1) when the seed changes, without
/// clearing 4096 entries or storing the `Noise` reference per slot.
///
/// # Why this is a struct and not a module global
///
/// The TypeScript version kept the three arrays, the `liveNoise` pointer and the
/// `stamp` at module scope. Rust has no safe equivalent that is also usable from
/// a rayon pool, and the brief forbids a `thread_local`. So the memo is OWNED:
/// every method that touches it takes `&mut self`, and each worker constructs its
/// own [`Heightmap`]. Worldgen's sampling functions stay `&self`-callable because
/// the pure paths ([`terrain_params_at`], [`shore_weight_at`]) are free functions
/// that never see the cache.
const MEMO_SIZE: usize = 4096;
const MEMO_MASK: i32 = MEMO_SIZE as i32 - 1;

/// The memoising front end to the heightmap. One per thread; cheap to build,
/// ~48 KiB of tables.
pub struct Heightmap {
    memo_col: Box<[i32]>,
    memo_row: Box<[i32]>,
    memo_stamp: Box<[u32]>,
    /// Identity of the `Noise` the live entries were computed against, as an
    /// address. Reproduces the TypeScript `noise !== liveNoise` reference test:
    /// worldgen holds one `Noise` for the life of a world, so a changed address
    /// means a changed seed.
    live_noise: Option<usize>,
    stamp: u32,
}

impl Default for Heightmap {
    fn default() -> Self {
        Self::new()
    }
}

impl Heightmap {
    /// An empty cache. Nothing is precomputed; the first call for any `Noise`
    /// bumps the stamp and invalidates every slot at once.
    pub fn new() -> Heightmap {
        Heightmap {
            memo_col: vec![0i32; MEMO_SIZE].into_boxed_slice(),
            memo_row: vec![0i32; MEMO_SIZE].into_boxed_slice(),
            memo_stamp: vec![0u32; MEMO_SIZE].into_boxed_slice(),
            live_noise: None,
            stamp: 0,
        }
    }

    /// Bump the generation counter if `noise` is not the one the live entries
    /// belong to. O(1) invalidation of all 4096 slots.
    #[inline]
    fn retire_if_new_seed(&mut self, noise: &Noise) {
        let id = noise as *const Noise as usize;
        if self.live_noise != Some(id) {
            self.live_noise = Some(id);
            self.stamp = self.stamp.wrapping_add(1);
        }
    }

    #[inline]
    fn store(&mut self, slot: usize, wcx: i32, surf: i32) {
        self.memo_col[slot] = wcx;
        self.memo_row[slot] = surf;
        self.memo_stamp[slot] = self.stamp;
    }

    /// Absolute cell row of the ground surface for world column `wcx`. Pure
    /// function of `wcx`, so adjacent chunks agree on the ground line across
    /// their seam.
    ///
    /// `col` is accepted for source compatibility and as a hint, but the value is
    /// derived from `height_params_at` either way: `ColumnProfile::amp_scale` and
    /// `height_params_at().amp_scale` are the same weighted sum computed by the
    /// same code path (see `biomes`' `height_from_weights`), so the two entry
    /// points cannot disagree in the last bit and round to different rows.
    pub fn surface_row_at(&mut self, noise: &Noise, wcx: i32, col: Option<&ColumnProfile>) -> i32 {
        self.retire_if_new_seed(noise);
        let slot = (wcx & MEMO_MASK) as usize;
        if self.memo_stamp[slot] == self.stamp && self.memo_col[slot] == wcx {
            return self.memo_row[slot];
        }

        let p = match col {
            Some(col) => compute_params(noise, wcx, col.amp_scale, col.height_offset),
            None => {
                let hp = height_params_at(noise, wcx);
                compute_params(noise, wcx, hp.amp_scale, hp.height_offset)
            }
        };

        self.store(slot, wcx, p.surf);
        p.surf
    }

    /// Fractional ground height and shore weight for a column, computed alongside
    /// the row. Chunk generation needs the shore weight for the beach cap and
    /// would otherwise pay for the whole heightmap twice; this returns both from
    /// one pass. Bypasses the memo ON READ on purpose — the caller is the one
    /// filling it — but still WRITES the row it just computed, so the decorator
    /// passes that follow over the same chunk hit a warm cache.
    pub fn surface_detail_at(
        &mut self,
        noise: &Noise,
        wcx: i32,
        col: &ColumnProfile,
    ) -> SurfaceDetail {
        let p = compute_params(noise, wcx, col.amp_scale, col.height_offset);

        self.retire_if_new_seed(noise);
        let slot = (wcx & MEMO_MASK) as usize;
        self.store(slot, wcx, p.surf);

        SurfaceDetail {
            surf: p.surf,
            shore: p.shore,
            submerged: p.submerged,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SEED;
    // Nothing in the shipping path needs the slope — the terracing pass reads it
    // through `TERRACE` — so it is imported here rather than at module scope.
    use super::super::spline::spline_slope;

    #[test]
    fn js_round_sends_halves_up_not_away_from_zero() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0); // f64::round would give -3
        assert_eq!(js_round(-2.6), -3.0);
        assert_eq!(js_round(0.5), 1.0);
        assert_eq!(js_round(-0.5), 0.0);
    }

    #[test]
    fn the_terrace_snap_lands_exactly_on_a_step_boundary() {
        // The whole point of terracing: the snapped height sits on the shelf
        // grid, which is TERRACE_STEP-periodic and offset by the phase.
        for pk in 0..13 {
            let phase = TERRACE_STEP * (pk as f64 / 12.0);
            for hk in -400..400 {
                let h = hk as f64 / 7.0;
                let q = terrace_snap(h, phase);
                let k = (q - phase) / TERRACE_STEP;
                assert!(
                    (k - k.round()).abs() < 1e-9,
                    "terrace_snap({h}, {phase}) = {q} is not on the grid"
                );
                // ...and it is the NEAREST such boundary.
                assert!(
                    (q - h).abs() <= TERRACE_STEP / 2.0 + 1e-9,
                    "terrace_snap({h}, {phase}) = {q} skipped a shelf"
                );
            }
        }
    }

    #[test]
    fn terracing_is_a_blend_between_the_raw_and_snapped_heights() {
        // The blend is why terracing does not read as a hard switch. At weight w
        // the height must land strictly between the raw value and the shelf.
        let raw = 51.3;
        let phase = 2.5;
        let q = terrace_snap(raw, phase);
        for k in 1..10 {
            let w = k as f64 / 10.0;
            let h = lerp(raw, q, w);
            let (lo, hi) = if raw < q { (raw, q) } else { (q, raw) };
            assert!(h >= lo - 1e-12 && h <= hi + 1e-12);
        }
    }

    #[test]
    fn the_continental_curve_puts_the_waterline_where_the_comment_says() {
        // "Sea level is +6 from the anchor" — the -0.42 control point is 5, i.e.
        // one cell of dry berm, and -0.5 is 9, i.e. submerged.
        let row_at = |t: f64| SURFACE_ANCHOR_Y + spline(CONTINENTAL, t) as i32;
        assert_eq!(row_at(-0.42), SEA_LEVEL_Y - 1);
        assert!(row_at(-0.5) > SEA_LEVEL_Y);
        assert!(row_at(-0.34) < SEA_LEVEL_Y);
        // The beach compresses 8 cells into 0.16 of the domain (a narrow coast),
        // while the inland plain — the world's default — spends 0.18 of the
        // domain on 4 cells. That ratio is what makes coasts rare and short and
        // plains common and flat.
        let beach = (spline(CONTINENTAL, -0.34) - spline(CONTINENTAL, -0.5)).abs() / 0.16;
        let plain = spline_slope(CONTINENTAL, 0.0).abs();
        assert!(
            beach >= 50.0,
            "the beach flattened out to {beach} cells/unit"
        );
        assert!(
            beach > 2.0 * plain,
            "beach {beach} is not steep against plain {plain}"
        );
    }

    #[test]
    fn shore_weight_saturates_below_sea_level_and_fades_above_it() {
        assert_eq!(shore_weight_at(SEA_LEVEL_Y), 1.0);
        assert_eq!(shore_weight_at(SEA_LEVEL_Y + 40), 1.0);
        assert_eq!(shore_weight_at(SEA_LEVEL_Y - SHORE_BAND), 0.0);
        assert_eq!(shore_weight_at(SEA_LEVEL_Y - 100), 0.0);
        // Monotone across the band, so the beach fades rather than switching.
        let mut prev = 0.0;
        for s in (SEA_LEVEL_Y - SHORE_BAND)..=SEA_LEVEL_Y {
            let w = shore_weight_at(s);
            assert!(w >= prev - 1e-12, "shore weight dipped at row {s}");
            prev = w;
        }
    }

    #[test]
    fn compute_params_is_pure_in_the_column() {
        // Two chunks meeting at a seam must compute the identical ground line by
        // computing it twice.
        let n = Noise::new(SEED);
        for wcx in -600..600 {
            let a = compute_params(&n, wcx, 1.0, 0.0);
            let b = compute_params(&n, wcx, 1.0, 0.0);
            assert_eq!(a, b, "compute_params disagreed with itself at {wcx}");
            assert_eq!(a.submerged, a.surf > SEA_LEVEL_Y);
            assert_eq!(a.surf, js_round(a.height) as i32);
        }
    }

    #[test]
    fn the_world_is_not_flat_and_not_a_cliff() {
        // The failure mode the whole spline pipeline exists to avoid: a world
        // flat to within two cells everywhere (the first draft), or one that
        // jumps so hard between adjacent columns that it reads as noise.
        let n = Noise::new(SEED);
        let rows: Vec<i32> = (-4000..4000)
            .map(|x| compute_params(&n, x, 1.0, 0.0).surf)
            .collect();
        let lo = *rows.iter().min().unwrap();
        let hi = *rows.iter().max().unwrap();
        assert!(hi - lo > 30, "the world spans only {} rows", hi - lo);
        let worst = rows.windows(2).map(|w| (w[1] - w[0]).abs()).max().unwrap();
        assert!(worst <= 8, "adjacent columns jumped {worst} rows");
        let submerged = rows.iter().filter(|&&r| r > SEA_LEVEL_Y).count();
        let frac = submerged as f64 / rows.len() as f64;
        assert!(
            (0.02..0.5).contains(&frac),
            "{:.1}% of the world is sea",
            frac * 100.0
        );
    }

    #[test]
    fn the_memo_returns_exactly_what_an_uncached_computation_would() {
        // The memo is a PURE memo. If a hit could differ from a miss in the last
        // bit, two chunks would disagree on a seam depending on visit order.
        let n = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let mut truth = Vec::new();
        for wcx in -500..500 {
            truth.push(terrain_params_at(&n, wcx).surf);
        }
        // Cold pass, then two warm passes, then interleaved.
        for pass in 0..3 {
            for (i, wcx) in (-500..500).enumerate() {
                assert_eq!(
                    hm.surface_row_at(&n, wcx, None),
                    truth[i],
                    "pass {pass} col {wcx}"
                );
            }
        }
        for (i, wcx) in (-500..500).enumerate().rev() {
            assert_eq!(hm.surface_row_at(&n, wcx, None), truth[i]);
        }
    }

    #[test]
    fn aliased_columns_do_not_collide_across_the_direct_mapped_table() {
        // 4096 slots, so `wcx` and `wcx + 4096` share one. The stored column tag
        // is what keeps the second from reading the first's row.
        let n = Noise::new(SEED);
        let mut hm = Heightmap::new();
        for wcx in 0..40 {
            let a = hm.surface_row_at(&n, wcx, None);
            let b = hm.surface_row_at(&n, wcx + MEMO_SIZE as i32, None);
            assert_eq!(a, terrain_params_at(&n, wcx).surf);
            assert_eq!(b, terrain_params_at(&n, wcx + MEMO_SIZE as i32).surf);
            // ...and the evicted entry recomputes correctly.
            assert_eq!(hm.surface_row_at(&n, wcx, None), a);
        }
    }

    #[test]
    fn a_new_seed_invalidates_the_whole_table_at_once() {
        let a = Noise::new(SEED);
        let b = Noise::new(SEED + 1);
        let mut hm = Heightmap::new();
        let mut differs = 0;
        for wcx in -200..200 {
            let ra = hm.surface_row_at(&a, wcx, None);
            let rb = hm.surface_row_at(&b, wcx, None);
            assert_eq!(rb, terrain_params_at(&b, wcx).surf, "stale entry at {wcx}");
            assert_eq!(hm.surface_row_at(&a, wcx, None), ra);
            if ra != rb {
                differs += 1;
            }
        }
        assert!(differs > 100, "two seeds produced the same coastline");
    }
}

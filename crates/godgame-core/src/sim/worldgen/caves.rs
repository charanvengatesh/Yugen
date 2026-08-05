//! Cave carving — a layered system, every layer a pure function of (wcx, wcy).
//!
//! The old carver was one fBm field thresholded at a constant: `fbm2(x,y,4) >
//! open`. That is isotropic by construction, so it can only produce isotropic
//! output — swiss cheese. Every hole is the same shape, they connect only by
//! accident, and there is nothing to explore because there is no structure to
//! follow. Three separate systems replace it, each producing a shape the others
//! cannot:
//!
//! * CHEESE — `gfbm2` thresholded, domain-warped. Big irregular CHAMBERS. This
//!   is the only isotropic layer and that is fine, because it is now the
//!   minority of the open volume rather than all of it. Domain warping is what
//!   stops the chambers reading as circles: it shears and folds their outlines
//!   into lobes and alcoves.
//!
//! * TUNNELS — thresholded ridged noise. THE important layer. In 2D the set
//!   `|noise| < t` is a thickened ZERO CONTOUR, and the zero contour of smooth
//!   gradient noise is a long, smooth, wandering, unbranching curve — which is
//!   exactly a tunnel. Ridged noise is `(1-|n|)^2`, so "near a ridge" IS "near
//!   the zero set", and thresholding it high carves that curve out of the rock.
//!
//!   NOTE ON THE 3D RECIPE: the famous Minecraft trick intersects TWO of these
//!   fields, because in 3D each one is a 2D sheet and only their intersection is
//!   a 1D worm. In 2D that reasoning inverts — each field is ALREADY a 1D curve,
//!   and intersecting two curves leaves isolated points. So the two independent
//!   fields here are UNIONED, giving two interleaved tunnel systems that cross
//!   each other; the intersection is still used, but to WIDEN the crossings into
//!   junction rooms, which is where the network reads as a network instead of
//!   two unrelated corridors.
//!
//!   Field A is vertically squashed (`TUN_FY > TUN_FX`), so its features are
//!   wider than they are tall and its zero contour trends HORIZONTAL — long
//!   walkable passages. Field B is squashed the other way and carries the
//!   vertical connections between levels.
//!
//! * RAVINES — a very low frequency ridged field with an extremely low VERTICAL
//!   frequency, so its ridge lines are near-vertical slabs ~130 columns apart.
//!   Gated hard by `weirdness` so most of the world has none, and windowed in
//!   depth so each one is a lens rather than an infinite slot. Where weirdness
//!   is highest the window opens above the topsoil and the ravine becomes a
//!   surface CHASM you can fall into.
//!
//! PERFORMANCE. Everything genuinely low-frequency — the warp offsets, the
//! openness bias, the ravine field, the tunnel-radius modulation, the liquid
//! table wobble, the strata bands — is sampled once per GEN_LATTICE cells into a
//! per-chunk scratch buffer and bilinearly interpolated per cell. Only the cheese
//! and tunnel fields, which have cell-scale detail, are evaluated per cell. The
//! lattice stride divides CHUNK_CELLS, so the lattice is GLOBALLY aligned: the
//! corner samples a chunk interpolates from at its right edge are the same
//! samples its right neighbour interpolates from at its left edge, bit for bit.
//! The optimisation is therefore invisible to chunk independence — which it would
//! NOT be if the lattice were anchored to the chunk instead of to world space.
//! Measured: 0.195 ms/chunk with every field sampled exactly per cell, 0.133 with
//! the lattice — a 32% saving over the whole generator.
//!
//! The tunnel and cheese fields deliberately stay per-cell. `ridged2` is
//! `(1-|g2|)^2`, so it has a KINK exactly on its zero set — which is exactly
//! where the passage threshold sits. Interpolating across that kink rounds the
//! crest off and pinches the tunnels shut; this is the one field in the generator
//! where the coarse lattice is not merely inaccurate but structurally wrong.
//!
//! # Port note
//!
//! The TypeScript held the lattice, the row parameters and the per-column record
//! in module-level mutable scratch, because allocating them per chunk was
//! measurable GC pressure. None of that survives into Rust: worldgen is called
//! through `&Noise` from a rayon pool, so shared mutable module state would be
//! either a data race or a lock. The lattice becomes a [`CaveLattice`] the caller
//! owns (one per worker), and the row parameters and column record are small
//! `Copy` structs passed by value — which costs nothing, since they live in
//! registers rather than on the heap.

use super::fields::{lerp, smooth_ramp, smoothstep01, weirdness};
use crate::config::{
    CAVE_SURFACE_FADE, CHUNK_CELLS, DEEP_DEPTH, GEN_LATTICE, UNDERWORLD_DEPTH, UNDERWORLD_FLOOR,
};
use crate::sim::biomes::ColumnProfile;
use crate::sim::noise::Noise;

/// Re-exported for callers that want the same clamp semantics.
pub use super::fields::clamp01;

/// What the carver decided about one cell.
///
/// The TypeScript used a plain numeric union rather than an enum, because
/// `isolatedModules` forbids `const enum` and a real TS enum would allocate a
/// runtime object for three integers used in the hottest loop in the generator.
/// Rust has no such tax: this is a one-byte value carrying the same
/// discriminants, and the exhaustive `match` on it is free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Carve {
    /// Rock. The layer pass decides which rock.
    Solid = 0,
    /// Open air.
    Air = 1,
    /// Open, but below the liquid table — the layer pass decides which liquid.
    Liquid = 2,
}

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

// Cheese chambers. FY slightly above FX so chambers are wider than tall, which
// reads as a room rather than a shaft. The warp is ~half a chamber radius: any
// less and the outlines stay round, any more and chambers fold into each other
// and the layer degenerates back into swiss cheese.
const CHEESE_FX: f64 = 0.03;
const CHEESE_FY: f64 = 0.038;
const CHEESE_WARP_AMP: f64 = 16.0;
const CHEESE_WARP_FREQ: f64 = 0.0075;
/// Thresholds are quoted against the MEASURED distribution of their field, not
/// against its theoretical range — gradient-noise fBm is nowhere near uniform and
/// a threshold picked by eye is off by an order of magnitude in open volume. For
/// `gfbm2c` (see gen/bench.ts `caveStats`, and the percentile sweep that tuned
/// these): p50 = 0.00, p95 = 0.32, p99 = 0.44, max ~ 0.63.
///
/// 0.425 is therefore past the 98th percentile: ~2% of cells at cavern depth open
/// as chamber, rising to ~5% at full depth as [`CHEESE_TD`] relaxes it. Chambers
/// are meant to be the MINORITY of the open volume — the tunnels below are what
/// connect the world, and cheese that dominates is exactly the swiss-cheese
/// failure this rewrite exists to remove.
const CHEESE_T0: f64 = 0.6;
/// How much the threshold relaxes by full depth — bigger caverns lower down.
const CHEESE_TD: f64 = 0.1;
/// How hard the low-frequency bias field pushes the threshold around.
const CHEESE_TB: f64 = 0.06;
/// Underworld cheese threshold — below the field's median, so the band is void.
const UNDERWORLD_T0: f64 = -0.16;

// Tunnels.
//
// FREQUENCY IS WIDTH. The open band around the zero contour has width
// ~ (1 - threshold) / |grad field|, and |grad field| is proportional to
// frequency — so halving the frequency doubles the passage width AND doubles the
// spacing between passages, leaving the open FRACTION unchanged while completely
// changing the read. The first draft ran at 0.023/0.049 and the caves came out
// as 1-2 cell scratches: the right amount of hole, shredded into cracks nobody
// can walk down. At 0.012 a passage is 4-7 cells — a corridor.
//
// The FY/FX ratio is what makes them horizontal: field A's features are ~4x
// wider than tall, so its zero contour has nowhere to go but sideways. Field B
// inverts the ratio and supplies the vertical connections between levels.
const TUN_FX: f64 = 0.0098;
const TUN_FY: f64 = 0.0215; // field A: squashed vertically -> horizontal passages
const TUN_BX: f64 = 0.017; // field B: the reverse aspect -> vertical connections
const TUN_BY: f64 = 0.0085;
/// Fraction of the cheese domain-warp vector applied to the tunnel coordinates.
///
/// Unwarped, a ridged zero contour is a smooth arc — geometric, and obviously
/// generated. A third of the warp bends it into a meander with switchbacks
/// without shredding it (a full-strength warp at this frequency folds the contour
/// back over itself and the passage pinches shut at every fold).
const TUN_WARP: f64 = 0.55;
/// Octaves per tunnel field. ONE, and that is a considered choice, not a cut
/// corner: at [`TUN_FX`] = 0.0098 the second octave sits at a 51-cell period,
/// which is smaller than a passage is long but larger than it is wide — so all it
/// does is chew the passage walls into a dashed line while doubling the cost of
/// the hottest sample in the generator. The domain warp above supplies the meander
/// that the second octave was supposed to; the walls get their fine irregularity
/// from the cheese and vein fields painted over them instead.
const TUN_OCT: u32 = 1;
/// Ridged threshold for a passage — the single biggest lever on how much of the
/// underground is open, and the one that is impossible to guess.
///
/// `ridged2(_, _, 2)` measures: p50 = 0.61, p85 = 0.80, p90 = 0.84, p95 = 0.89,
/// p99 = 0.95. Note how compressed the top end is: 0.80 opens 15% of cells and
/// 0.89 opens 5%. The first draft of this file used 0.80 for both fields and the
/// underground came out 83% open — rubble, not caves.
///
/// 0.925 puts each field near 3%; unioned, two fields plus junction widening plus
/// cheese and ravines land at ~16% open at cavern depth and ~21% at full depth
/// (measured by `caveStats`), which reads as a traversable network with real rock
/// between the passages.
const TUN_T0: f64 = 0.962;
/// Threshold relief by full depth: deeper passages are wider.
const TUN_TD: f64 = 0.018;
/// Modulation of the threshold by the low-frequency radius field, +/- this.
const TUN_TR: f64 = 0.01;
/// How far below the passage threshold BOTH fields must be for a junction room.
/// A crossing where each field is merely close to its own threshold gets opened,
/// which fattens the intersection into a chamber instead of a bare crossroads.
const TUN_JUNCTION: f64 = 0.018;
/// A biome/layer `caveScale` is a MULTIPLIER on openness, but these fields are
/// thresholded, not scaled — dividing the threshold by the scale (what the old
/// generator did) is wildly non-linear near the tail and turns caveScale 1.35
/// into "half the world is missing". Converting the multiplier into a threshold
/// SHIFT keeps its effect proportionate.
///
/// The two gains differ because the two fields have completely different spreads
/// at the top of their range: single-octave ridged noise packs p90-p99 into
/// 0.072, single-octave `g2` spreads the same span over 0.26. A shift that
/// doubles the cheese is a rounding error to the tunnels and vice versa — one
/// shared constant would silently make caveScale mean two different things.
const TUN_SCALE_GAIN: f64 = 0.03;
const CHEESE_SCALE_GAIN: f64 = 0.1;
/// Threshold penalty where the surface fade is fully closed. Larger than any
/// field's range, so `fade == 0` means "solid", not "nearly solid".
const FADE_BITE: f64 = 0.7;

// Ravines. FX gives ~135 columns between candidate slabs; FY is 7x lower still,
// so a slab's ridge line drifts only a few cells of x over 100 cells of y — the
// near-vertical read.
const RAV_FX: f64 = 0.0074;
const RAV_FY: f64 = 0.00105;
/// Threshold where no ravine is permitted at all. Nothing reaches this.
const RAV_T_CLOSED: f64 = 1.2;
/// Threshold in a fully ravine-prone region. Lower = wider slabs. p97 of the
/// ridged field is 0.915, so 0.93 is a thin slab even where fully enabled.
const RAV_T_OPEN: f64 = 0.87;
/// Weirdness at which ravines start.
///
/// Against the SPREAD weirdness field (see worldgen::fields): p73 and p93. So
/// about a quarter of columns permit a ravine at all and ~7% permit a full-width
/// one, which lands a chasm every few hundred columns rather than every few
/// thousand.
const RAV_W0: f64 = 0.12;
/// Weirdness at which ravines are fully enabled. See [`RAV_W0`].
const RAV_W1: f64 = 0.45;
/// Weirdness at which a ravine's top window starts opening above the topsoil.
///
/// p89 and p97 of the spread weirdness: breaching the topsoil is the rare case.
const CHASM_W0: f64 = 0.34;
/// Weirdness at which the top window is fully open. See [`CHASM_W0`].
const CHASM_W1: f64 = 0.66;
/// Depth window: a non-breaching ravine starts here.
const RAV_TOP_BURIED: f64 = 30.0;
/// Depth window: a chasm starts above ground.
const RAV_TOP_CHASM: f64 = -3.0;
/// Depth at which the ravine starts tapering shut at the bottom.
const RAV_TAPER_FROM: f64 = 120.0;
/// Depth by which the ravine has tapered fully shut. See [`RAV_TAPER_FROM`].
const RAV_TAPER_TO: f64 = 195.0;
/// How much the threshold is raised where the depth window is closed.
const RAV_WINDOW_BITE: f64 = 0.25;

// Liquid table: the depth below which an open cell holds the layer's pocket
// liquid instead of air. Per column, plus a lattice wobble, so the water line in
// a cave system is an undulating surface — dry chambers above it, flooded
// grottos below it, and half-drowned ones straddling it.
//
// The base sits at 150 rather than near the top of the cavern band on purpose:
// a table that shallow floods EVERY void below it and the whole underground
// becomes an ocean with rock in it (the first draft measured 87% of deep cells
// as liquid). At 150 +/- 55 the cavern band is mostly dry and walkable, the
// bottom of it starts to sump, and the deep band is where you need to swim.
// The layer's `pocketBias` shifts the table by +/-40 cells, so magma chambers
// run their lava high and dry layers keep their galleries dry.
const LIQ_BASE: f64 = 320.0;
const LIQ_WET_GAIN: f64 = 165.0;
const LIQ_WET_FREQ: f64 = 0.0011;
const LIQ_WET_ANCHOR: f64 = 733.19;
const LIQ_POCKET_BIAS_GAIN: f64 = 300.0;
/// Amplitude of the 2D wobble on the liquid table. Large on purpose: at +/-14 the
/// table is effectively a ruled horizontal line and everything below it is one
/// uniform sea, which is what the first draft rendered. At +/-52 the line is a
/// ragged surface that leaves dry galleries hanging above flooded sumps and puts
/// isolated pools well above the mean table — the read you want when you break
/// into a chamber and cannot tell in advance whether it is wet.
const LIQ_WOBBLE: f64 = 52.0;
const LIQ_WOBBLE_FREQ: f64 = 0.0062;
const LIQ_WOBBLE_ANCHOR: f64 = 2201.7;

/// Amplitude of the per-column shift applied to every DEPTH BAND boundary
/// (cavern -> deep -> underworld -> bedrock) and to the depth term that widens
/// caves.
///
/// Without it every band edge is a ruled horizontal line across the entire world,
/// which is the single most artificial thing a layered generator can render — you
/// can see the underworld's ceiling as a straight cut a screen wide. A slow +/-30
/// cell shift turns each one into a dipping, swelling interface, and because it
/// is a function of the column alone it costs one gradient sample per column and
/// cannot break chunk independence.
const BAND_SHIFT_AMP: f64 = 30.0;
const BAND_SHIFT_FREQ: f64 = 0.0023;
const BAND_SHIFT_ANCHOR: f64 = 4409.7;

// Low-frequency openness bias — the field that makes some REGIONS cavey and
// others near-solid, so exploring has a payoff gradient.
const BIAS_FX: f64 = 0.0021;
const BIAS_FY: f64 = 0.0034;

// Tunnel radius modulation along a passage's length.
const TUNMOD_FX: f64 = 0.0105;
const TUNMOD_FY: f64 = 0.0088;

// Strata: near-horizontal geological banding for the deep rock. Very low x
// frequency and a moderate y frequency gives bands ~30 cells thick that dip and
// swell over hundreds of columns — geology, not stripes. Consumed by layers.rs.
const STRATA_FX: f64 = 0.0024;
const STRATA_FY: f64 = 0.03;

// ---------------------------------------------------------------------------
// The low-frequency fields
// ---------------------------------------------------------------------------
//
// Each of these is read from exactly two places: once per lattice corner by
// [`CaveLattice::fill`], and once per cell by the `carve_exact` probe path. The
// TypeScript wrote each expression out twice; naming them here means an edit to
// one path cannot silently move the other, which is the one class of bug the
// "runtime branch, not two copies" note below is trying to prevent.

/// Domain-warp offset for the cheese chambers, and (scaled by [`TUN_WARP`]) the
/// meander applied to the tunnel coordinates.
#[inline]
fn cheese_warp(noise: &Noise, wcx: f64, wcy: f64) -> (f64, f64) {
    let wfx = wcx * CHEESE_WARP_FREQ;
    let wfy = wcy * CHEESE_WARP_FREQ;
    (
        CHEESE_WARP_AMP * noise.g2(wfx + 137.31, wfy - 41.77),
        CHEESE_WARP_AMP * noise.g2(wfx - 613.19, wfy + 917.53),
    )
}

/// Regional openness bias — cavey regions versus near-solid ones.
#[inline]
fn bias_field(noise: &Noise, wcx: f64, wcy: f64) -> f64 {
    noise.gfbm2(wcx * BIAS_FX + 55.5, wcy * BIAS_FY - 12.25, 2)
}

/// The near-vertical ravine slabs.
#[inline]
fn ravine_field(noise: &Noise, wcx: f64, wcy: f64) -> f64 {
    noise.ridged2(wcx * RAV_FX + 811.3, wcy * RAV_FY - 77.9, 2, 1.9)
}

/// Tunnel radius modulation along a passage's length.
#[inline]
fn tunmod_field(noise: &Noise, wcx: f64, wcy: f64) -> f64 {
    noise.g2(wcx * TUNMOD_FX - 301.7, wcy * TUNMOD_FY + 148.3)
}

/// 2D wobble on the liquid table.
#[inline]
fn liq_field(noise: &Noise, wcx: f64, wcy: f64) -> f64 {
    noise.gfbm2(
        wcx * LIQ_WOBBLE_FREQ + LIQ_WOBBLE_ANCHOR,
        wcy * LIQ_WOBBLE_FREQ,
        2,
    )
}

/// Near-horizontal geological banding.
///
/// ONE octave, deliberately: a second octave at twice the vertical frequency
/// chops the bands into blobs and the geology stops reading as layers. The x
/// frequency is 12x lower than the y frequency, which is what makes a band a
/// band rather than a patch.
#[inline]
fn strata_field(noise: &Noise, wcx: f64, wcy: f64) -> f64 {
    noise.g2(wcx * STRATA_FX - 909.1, wcy * STRATA_FY + 63.7)
}

// ---------------------------------------------------------------------------
// Per-chunk lattice
// ---------------------------------------------------------------------------

const STRIDE: usize = GEN_LATTICE;
const LAT: usize = CHUNK_CELLS as usize / STRIDE + 1;
const LAT_N: usize = LAT * LAT;

/// `lx >> 2` / `lx & 3` in [`CaveLattice::lat`] are hard-coded to STRIDE = 4; the
/// divide and modulo are in the innermost loop of the generator and the shifts
/// measurably beat them. This assertion keeps that honest if GEN_LATTICE ever
/// moves. (The TypeScript threw at module load; here it is a build error.)
const _: () = assert!(
    STRIDE == 4,
    "caves.rs: lattice sampling is specialised for GEN_LATTICE = 4"
);

/// One chunk's worth of coarse samples of every low-frequency cave field.
///
/// Seven planes of `LAT * LAT` = 81 corners each, bilinearly interpolated per
/// cell by [`CaveLattice::lat`]. The TypeScript held these as module-level
/// `Float32Array`s reused across every chunk; here the caller owns one per
/// worker thread, which keeps the "allocate once" property without the shared
/// mutable state that rayon would turn into a race.
///
/// The planes are `f32` while every sample that feeds them is `f64`. That mixed
/// precision is LOAD-BEARING, not an oversight: it halves the memory traffic in
/// the hottest loop in the generator, and every threshold downstream was tuned
/// against the rounded values. Widening the planes to `f64` moves boundary
/// outcomes. The only widening happens inside `lat`, at interpolation time.
#[derive(Clone)]
pub struct CaveLattice {
    warp_x: [f32; LAT_N],
    warp_y: [f32; LAT_N],
    bias: [f32; LAT_N],
    ravine: [f32; LAT_N],
    tunmod: [f32; LAT_N],
    liq: [f32; LAT_N],
    strata: [f32; LAT_N],
}

impl Default for CaveLattice {
    fn default() -> Self {
        Self::new()
    }
}

impl CaveLattice {
    /// A zeroed lattice. Useless until [`CaveLattice::fill`] runs over it.
    pub fn new() -> CaveLattice {
        CaveLattice {
            warp_x: [0.0; LAT_N],
            warp_y: [0.0; LAT_N],
            bias: [0.0; LAT_N],
            ravine: [0.0; LAT_N],
            tunmod: [0.0; LAT_N],
            liq: [0.0; LAT_N],
            strata: [0.0; LAT_N],
        }
    }

    /// Sample every low-frequency cave field onto the chunk's lattice. Call once
    /// per chunk before any [`CaveLattice::carve`]; skip it entirely when the
    /// chunk has no underground cells — that is where most of the saving comes
    /// from.
    ///
    /// `base_x` / `base_y` are the chunk's top-left cell in ABSOLUTE world
    /// coordinates. The lattice is anchored THERE, not to the chunk, and STRIDE
    /// divides CHUNK_CELLS, so neighbouring chunks share their edge corners bit
    /// for bit and the interpolation cannot produce a seam.
    pub fn fill(&mut self, noise: &Noise, base_x: i32, base_y: i32) {
        for gy in 0..LAT {
            let wcy = f64::from(base_y + (gy * STRIDE) as i32);
            for gx in 0..LAT {
                let wcx = f64::from(base_x + (gx * STRIDE) as i32);
                let i = gy * LAT + gx;

                let (wx, wy) = cheese_warp(noise, wcx, wcy);
                self.warp_x[i] = wx as f32;
                self.warp_y[i] = wy as f32;

                self.bias[i] = bias_field(noise, wcx, wcy) as f32;
                self.ravine[i] = ravine_field(noise, wcx, wcy) as f32;
                self.tunmod[i] = tunmod_field(noise, wcx, wcy) as f32;
                self.liq[i] = liq_field(noise, wcx, wcy) as f32;
                self.strata[i] = strata_field(noise, wcx, wcy) as f32;
            }
        }
    }

    /// Bilinear sample of one lattice plane at chunk-local cell (lx, ly).
    ///
    /// Hand-unrolled for STRIDE = 4: `lx >> 2` is the corner index and
    /// `(lx & 3) * 0.25` the fraction, so the innermost loop of the generator has
    /// no divide and no modulo in it. The `const _` assertion above is what keeps
    /// this honest. Widening from `f32` happens here and nowhere else.
    #[inline]
    fn lat(buf: &[f32; LAT_N], lx: i32, ly: i32) -> f64 {
        let i = (ly >> 2) as usize * LAT + (lx >> 2) as usize;
        let fx = f64::from(lx & 3) * 0.25;
        let fy = f64::from(ly & 3) * 0.25;
        let a = f64::from(buf[i]);
        let b = f64::from(buf[i + 1]);
        let top = a + (b - a) * fx;
        let c = f64::from(buf[i + LAT]);
        let d = f64::from(buf[i + LAT + 1]);
        let bot = c + (d - c) * fx;
        top + (bot - top) * fy
    }

    /// Interpolated strata value in ~[-1,1] at a chunk-local cell. Near-horizontal
    /// banding; layers.rs turns it into alternating rock so a dug shaft shows
    /// geology. Requires [`CaveLattice::fill`] for this chunk.
    #[inline]
    pub fn strata_at(&self, lx: i32, ly: i32) -> f64 {
        Self::lat(&self.strata, lx, ly)
    }
}

/// Lattice-free strata, for the arbitrary-coordinate probe path.
pub fn strata_exact(noise: &Noise, wcx: i32, wcy: i32) -> f64 {
    strata_field(noise, f64::from(wcx), f64::from(wcy))
}

// ---------------------------------------------------------------------------
// Per-column cave parameters
// ---------------------------------------------------------------------------

/// The column-invariant half of the cave decision, hoisted out of the row loop.
///
/// The TypeScript threaded a shared scratch record through `caveColumnAt` so the
/// row loop never allocated. Eight `f64`s, an `i32` and a `bool` are a `Copy`
/// struct here, returned by value — the same zero allocations without the
/// aliasing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaveColumn {
    /// Ground row (absolute).
    pub surf: i32,
    /// Topsoil thickness in cells — cave openness fades in below this.
    pub cap_t: f64,
    /// Multiplier on openness near the surface (blended biome caveScale).
    pub scale_surf: f64,
    /// Multiplier on openness in the deep (blended layer caveScale).
    pub scale_ug: f64,
    /// Depth below which open cells hold liquid.
    pub liquid_depth: f64,
    /// Ridged threshold a ravine must beat here. `RAV_T_CLOSED` means "never".
    pub ravine_t: f64,
    /// Depth at which this column's ravine window opens. Negative = surface chasm.
    pub ravine_top: f64,
    /// True when a ravine can reach the topsoil — the cap loop must test it.
    pub breaches: bool,
    /// Added to `depth` before any DEPTH BAND is tested, so band interfaces dip
    /// and swell instead of ruling straight lines across the world. Callers that
    /// select MATERIAL by band (layers.rs) must apply it too, or the rock would
    /// change at a different depth from the caves.
    pub band_shift: f64,
}

impl Default for CaveColumn {
    /// A zeroed `CaveColumn` for a caller to own — the "no caves here" column.
    fn default() -> CaveColumn {
        CaveColumn {
            surf: 0,
            cap_t: 0.0,
            scale_surf: 1.0,
            scale_ug: 1.0,
            liquid_depth: LIQ_BASE,
            ravine_t: RAV_T_CLOSED,
            ravine_top: RAV_TOP_BURIED,
            breaches: false,
            band_shift: 0.0,
        }
    }
}

/// Build the per-column cave parameters. Pure in `wcx`.
///
/// Everything gated on `weirdness` is gated CONTINUOUSLY (a ramp, not a
/// comparison) so a ravine does not get clipped in half by the column where a
/// boolean flipped — the slab's top just rises smoothly along its length.
pub fn cave_column_at(noise: &Noise, wcx: i32, surf: i32, col: &ColumnProfile) -> CaveColumn {
    let w = weirdness(noise, wcx);
    let x = f64::from(wcx);

    let liquid_depth = LIQ_BASE - LIQ_WET_GAIN * noise.g2(x * LIQ_WET_FREQ, LIQ_WET_ANCHOR)
        + col.ug_pocket_bias * LIQ_POCKET_BIAS_GAIN;

    let rav_w = smooth_ramp(RAV_W0, RAV_W1, w);
    let ravine_t = if rav_w <= 0.0 {
        RAV_T_CLOSED
    } else {
        lerp(RAV_T_CLOSED, RAV_T_OPEN, rav_w)
    };

    let chasm_w = smooth_ramp(CHASM_W0, CHASM_W1, w);
    let ravine_top = lerp(RAV_TOP_BURIED, RAV_TOP_CHASM, chasm_w);

    CaveColumn {
        surf,
        cap_t: col.cap_thickness,
        scale_surf: col.surf_cave_scale,
        scale_ug: col.ug_cave_scale,
        liquid_depth,
        ravine_t,
        ravine_top,
        breaches: rav_w > 0.0 && ravine_top < col.cap_thickness,
        band_shift: BAND_SHIFT_AMP * noise.g2(x * BAND_SHIFT_FREQ, BAND_SHIFT_ANCHOR),
    }
}

// ---------------------------------------------------------------------------
// Carving
// ---------------------------------------------------------------------------

/// Which band a row falls in — four band tests collapsed into one value, so the
/// per-cell path is: match on the mode, add one interpolated modulation term to a
/// precomputed threshold, compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Mode {
    /// Nothing can open this cell.
    Bedrock = 0,
    /// Inside the surface fade — only a ravine can open it.
    Crust = 1,
    /// Normal underground — tunnels, cheese, ravines.
    Normal = 2,
    /// One relaxed cheese field, everything open is lava.
    Underworld = 3,
}

/// ROW PARAMETERS — the depth-only half of the carve decision, resolved into a
/// MODE plus four thresholds before any noise is touched.
///
/// (This was tried as a per-column lookup table indexed by row. It is slower:
/// `depth = wcy - surf` and `surf` differs per column, so a chunk needs 32
/// distinct tables of 32 entries — exactly one entry per cell, no reuse, plus the
/// array traffic. Measured 0.144 vs 0.125 ms/chunk. Recording it here so nobody
/// "optimises" it back.)
///
/// The TypeScript returned these through five module-scope variables, because
/// passing five more numbers through the hottest call in the generator measurably
/// cost more than the context-slot loads did. The Rust equivalent is a `Copy`
/// struct passed by reference: no allocation, no shared mutable state, and the
/// same registers.
#[derive(Clone, Copy, Debug)]
struct Row {
    mode: Mode,
    rav_t: f64,
    tun_t: f64,
    ch_t: f64,
    liq_t: f64,
}

/// Depth-only carve parameters for one row of one column. Pure in (cc, depth).
#[inline]
fn compute_row(cc: &CaveColumn, depth: i32) -> Row {
    let d = f64::from(depth);
    // Band depth: the real depth plus this column's slow shift, so every band
    // interface is an undulating surface rather than a ruled line. The SURFACE
    // fade below deliberately uses the TRUE depth — the topsoil crust must be a
    // constant thickness under your feet wherever you stand.
    let bd = d + cc.band_shift;

    // Ravines are tested outside the surface fade — a chasm that stopped 22 cells
    // down would be a pit. The window multiplies the slab's width rather than
    // gating it, so it tapers to a point at top and bottom instead of ending flat.
    let mut rav_t = RAV_T_CLOSED;
    if cc.ravine_t < RAV_T_CLOSED && bd < f64::from(UNDERWORLD_FLOOR) {
        let win = smooth_ramp(cc.ravine_top - 2.0, cc.ravine_top + 5.0, d)
            * (1.0 - smooth_ramp(RAV_TAPER_FROM, RAV_TAPER_TO, d));
        if win > 0.02 {
            rav_t = cc.ravine_t + (1.0 - win) * RAV_WINDOW_BITE;
        }
    }

    // The three thresholds below are only meaningful in the mode that sets them;
    // the early returns leave them at zero, exactly as the TypeScript left them
    // at whatever the previous row wrote. Neither value is ever read.
    let mut row = Row {
        mode: Mode::Bedrock,
        rav_t,
        tun_t: 0.0,
        ch_t: 0.0,
        liq_t: 0.0,
    };

    if bd >= f64::from(UNDERWORLD_FLOOR) {
        return row; // the bottom of the world
    }

    // Depth normalised over the first DEEP_DEPTH cells and then held. Every
    // "deeper means more open" term rides on this.
    let deep = f64::from(DEEP_DEPTH);
    let dt = if bd >= deep {
        1.0
    } else if bd <= 0.0 {
        0.0
    } else {
        bd / deep
    };

    let fade = smoothstep01((d - cc.cap_t) / f64::from(CAVE_SURFACE_FADE));
    if fade <= 0.0 {
        row.mode = Mode::Crust;
        return row;
    }

    // Openness scale crossfades from the surface biome's to the underground
    // layer's over the same depth range the rock signature does, then becomes a
    // threshold SHIFT (see the gain constants). The fade adds its own penalty on
    // top rather than multiplying, so the two effects stay independent.
    let scale_adj = lerp(
        cc.scale_surf,
        cc.scale_ug,
        if dt < 1.0 {
            smoothstep01(d / 90.0)
        } else {
            1.0
        },
    ) - 1.0;
    let fade_adj = (1.0 - fade) * FADE_BITE;

    if bd >= f64::from(UNDERWORLD_DEPTH) {
        row.mode = Mode::Underworld;
        row.ch_t = UNDERWORLD_T0 - scale_adj * CHEESE_SCALE_GAIN + fade_adj;
        return row;
    }

    row.mode = Mode::Normal;
    row.tun_t = TUN_T0 - TUN_TD * dt - scale_adj * TUN_SCALE_GAIN + fade_adj;
    row.ch_t = CHEESE_T0 - CHEESE_TD * dt - scale_adj * CHEESE_SCALE_GAIN + fade_adj;
    // Liquid iff `bd > liquid_depth + wobble * LIQ_WOBBLE`, rearranged so the
    // per-cell test is one comparison against the interpolated wobble.
    row.liq_t = (bd - cc.liquid_depth) / LIQ_WOBBLE;
    row
}

/// Where `carve_core` reads its low-frequency inputs from: this chunk's lattice,
/// at a chunk-local cell.
#[derive(Clone, Copy)]
struct LatSite<'a> {
    lattice: &'a CaveLattice,
    lx: i32,
    ly: i32,
}

/// The per-cell half of the carve decision.
///
/// `site` selects where the five low-frequency inputs come from: `Some` for
/// bilinear interpolation of this chunk's lattice (the generation path), `None`
/// for an exact gradient sample (the [`carve_exact`] probe path). It is a runtime
/// branch rather than two copies of this function ON PURPOSE — the two paths must
/// never drift apart, and a perfectly-predicted branch costs less than the
/// divergence risk.
///
/// Every field is sampled AT THE POINT OF USE, so a solid cell in the crust costs
/// zero samples and a typical solid cell costs three rather than six.
fn carve_core(noise: &Noise, wcx: i32, wcy: i32, site: Option<LatSite<'_>>, r: &Row) -> Carve {
    let mode = r.mode;
    if mode == Mode::Bedrock {
        return Carve::Solid;
    }

    let x = f64::from(wcx);
    let y = f64::from(wcy);

    let mut open = false;
    if r.rav_t < RAV_T_CLOSED {
        let rav = match site {
            Some(s) => CaveLattice::lat(&s.lattice.ravine, s.lx, s.ly),
            None => ravine_field(noise, x, y),
        };
        open = rav > r.rav_t;
    }

    if !open {
        if mode == Mode::Crust {
            return Carve::Solid;
        }

        let (warp_x, warp_y) = match site {
            Some(s) => (
                CaveLattice::lat(&s.lattice.warp_x, s.lx, s.ly),
                CaveLattice::lat(&s.lattice.warp_y, s.lx, s.ly),
            ),
            None => cheese_warp(noise, x, y),
        };

        if mode == Mode::Underworld {
            // One big open lava sea broken by basalt spires. Cheese alone, at a
            // threshold below the field's median so most of the band is void —
            // tunnels would be invisible inside a cavern this large, and skipping
            // them saves two gradient samples per cell down here. The frequency
            // is scaled down so the remaining rock reads as islands, not gravel.
            let cheese = noise.g2(
                (x + warp_x) * CHEESE_FX * 0.7,
                (y + warp_y) * CHEESE_FY * 0.7,
            );
            let bias = match site {
                Some(s) => CaveLattice::lat(&s.lattice.bias, s.lx, s.ly),
                None => bias_field(noise, x, y),
            };
            if cheese <= r.ch_t + CHEESE_TB * bias {
                return Carve::Solid;
            }
            return Carve::Liquid;
        }

        // --- Tunnels ---------------------------------------------------------
        // Tested BEFORE cheese: they are the likeliest layer to open AND the
        // cheapest test (one gradient sample), so most open cells cost one sample
        // and never touch the other two fields.
        let tun_mod = match site {
            Some(s) => CaveLattice::lat(&s.lattice.tunmod, s.lx, s.ly),
            None => tunmod_field(noise, x, y),
        };
        let t = r.tun_t + TUN_TR * tun_mod;
        let tx = x + warp_x * TUN_WARP;
        let ty = y + warp_y * TUN_WARP;
        let ra = noise.ridged2(tx * TUN_FX + 311.7, ty * TUN_FY - 88.3, TUN_OCT, 2.0);
        // Field A opening is the common case and needs no further work — falling
        // through to the liquid table below IS the "open" outcome, so unlike the
        // TypeScript nothing writes `open` back here. Nothing reads it again.
        if ra <= t {
            let rb = noise.ridged2(tx * TUN_BX - 907.1, ty * TUN_BY + 512.9, TUN_OCT, 2.0);
            // Union for the passages; intersection (at a relaxed threshold) to
            // fatten the crossings into junction rooms.
            let tunnelled = rb > t || (ra > t - TUN_JUNCTION && rb > t - TUN_JUNCTION);
            if !tunnelled {
                // --- Cheese chambers -----------------------------------------
                let cheese = noise.g2((x + warp_x) * CHEESE_FX, (y + warp_y) * CHEESE_FY);
                let bias = match site {
                    Some(s) => CaveLattice::lat(&s.lattice.bias, s.lx, s.ly),
                    None => bias_field(noise, x, y),
                };
                if cheese <= r.ch_t + CHEESE_TB * bias {
                    return Carve::Solid;
                }
            }
        }
    } else if mode == Mode::Underworld {
        return Carve::Liquid;
    }

    // --- Liquid table --------------------------------------------------------
    // An undulating water/lava line rather than a flat depth test, so a cave
    // system has dry galleries, half-drowned ones and fully flooded sumps. What
    // the liquid actually IS is the layer's business (layers.rs) — water, acid or
    // lava depending on the underground layer here.
    let liq_wobble = match site {
        Some(s) => CaveLattice::lat(&s.lattice.liq, s.lx, s.ly),
        None => liq_field(noise, x, y),
    };
    if liq_wobble < r.liq_t {
        Carve::Liquid
    } else {
        Carve::Air
    }
}

impl CaveLattice {
    /// Carve one cell of the chunk currently loaded into this lattice. `lx`/`ly`
    /// are chunk-local; `wcx`/`wcy` absolute; `depth` measured from the column's
    /// surface.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn carve(
        &self,
        noise: &Noise,
        wcx: i32,
        wcy: i32,
        depth: i32,
        lx: i32,
        ly: i32,
        cc: &CaveColumn,
    ) -> Carve {
        let row = compute_row(cc, depth);
        let site = LatSite {
            lattice: self,
            lx,
            ly,
        };
        carve_core(noise, wcx, wcy, Some(site), &row)
    }
}

/// SEAM FOR THE FEATURE PASS. "Is this cell inside a cave?" at any absolute
/// coordinate, with no chunk lattice loaded and no row tables prepared — every
/// low-frequency field is evaluated exactly instead of interpolated.
///
/// Structures, mineshafts and dungeons need this to avoid spawning inside solid
/// rock (or inside a ravine). It is several times the cost of
/// [`CaveLattice::carve`] and is meant for the handful of probes a feature
/// placement does, not a full scan.
///
/// It can disagree with `carve` by at most the lattice interpolation error, on
/// cells within a hair of a threshold. That is intentional and safe: `carve` is
/// the authority on terrain, this is an oracle for placement, and a feature pass
/// must overwrite what it finds rather than assume it.
pub fn carve_exact(noise: &Noise, wcx: i32, wcy: i32, depth: i32, cc: &CaveColumn) -> Carve {
    let row = compute_row(cc, depth);
    carve_core(noise, wcx, wcy, None, &row)
}

/// SEAM FOR THE FEATURE PASS. Depth below the surface at which cave voids in this
/// column start holding liquid — i.e. how deep a shaft can go before it floods.
/// Cheap: one gradient sample.
pub fn liquid_table_at(noise: &Noise, wcx: i32) -> f64 {
    LIQ_BASE - LIQ_WET_GAIN * noise.g2(f64::from(wcx) * LIQ_WET_FREQ, LIQ_WET_ANCHOR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SEED;

    /// A column with caves everywhere and a ravine permitted, so the tests can
    /// exercise every branch without building a `ColumnProfile`.
    fn test_column() -> CaveColumn {
        CaveColumn {
            surf: 48,
            cap_t: 6.0,
            scale_surf: 1.0,
            scale_ug: 1.0,
            liquid_depth: LIQ_BASE,
            ravine_t: RAV_T_OPEN,
            ravine_top: RAV_TOP_BURIED,
            breaches: false,
            band_shift: 0.0,
        }
    }

    #[test]
    fn the_lattice_is_the_right_shape_for_stride_four() {
        assert_eq!(STRIDE, 4);
        assert_eq!(LAT, 9, "32 / 4 + 1");
        assert_eq!(LAT_N, 81);
        // The last cell of a chunk must still have a corner to its right and
        // below, or the unrolled bilerp reads out of bounds.
        let last = ((CHUNK_CELLS - 1) >> 2) as usize;
        assert!(last * LAT + last + LAT + 1 < LAT_N);
    }

    #[test]
    fn the_bilerp_reproduces_a_direct_sample_at_every_lattice_corner() {
        // The defining property of the optimisation: on a corner the bilinear
        // weights collapse to (1,0,0,0), so the interpolated value must be the
        // exact sample — modulo the `f32` the plane is stored in, which is the
        // one deliberate precision loss.
        let noise = Noise::new(SEED);
        let mut lattice = CaveLattice::new();
        let (base_x, base_y) = (64, 256);
        lattice.fill(&noise, base_x, base_y);

        for gy in 0..LAT - 1 {
            for gx in 0..LAT - 1 {
                let (lx, ly) = ((gx * STRIDE) as i32, (gy * STRIDE) as i32);
                let wcx = f64::from(base_x + lx);
                let wcy = f64::from(base_y + ly);

                let want = strata_field(&noise, wcx, wcy);
                let got = lattice.strata_at(lx, ly);
                assert!(
                    (want - got).abs() < 1e-6,
                    "strata corner ({lx},{ly}): {want} vs {got}"
                );

                let want = ravine_field(&noise, wcx, wcy);
                let got = CaveLattice::lat(&lattice.ravine, lx, ly);
                assert!(
                    (want - got).abs() < 1e-6,
                    "ravine corner ({lx},{ly}): {want} vs {got}"
                );

                // The warp is 16x the field, so its f32 error is 16x too.
                let (wx, wy) = cheese_warp(&noise, wcx, wcy);
                assert!((wx - CaveLattice::lat(&lattice.warp_x, lx, ly)).abs() < 1e-4);
                assert!((wy - CaveLattice::lat(&lattice.warp_y, lx, ly)).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn the_bilerp_stays_between_the_corners_it_interpolates() {
        let noise = Noise::new(SEED);
        let mut lattice = CaveLattice::new();
        lattice.fill(&noise, 0, 128);
        for ly in 0..CHUNK_CELLS {
            for lx in 0..CHUNK_CELLS {
                let i = (ly >> 2) as usize * LAT + (lx >> 2) as usize;
                let corners = [
                    lattice.strata[i],
                    lattice.strata[i + 1],
                    lattice.strata[i + LAT],
                    lattice.strata[i + LAT + 1],
                ];
                let lo = f64::from(corners.iter().copied().fold(f32::INFINITY, f32::min));
                let hi = f64::from(corners.iter().copied().fold(f32::NEG_INFINITY, f32::max));
                let v = lattice.strata_at(lx, ly);
                assert!(
                    v >= lo - 1e-9 && v <= hi + 1e-9,
                    "({lx},{ly}) {v} not in [{lo},{hi}]"
                );
            }
        }
    }

    #[test]
    fn neighbouring_chunks_share_their_edge_corners_exactly() {
        // This is the whole reason the lattice is anchored to WORLD space with a
        // stride that divides CHUNK_CELLS. If these ever disagreed, every chunk
        // boundary in the world would be a visible seam.
        let noise = Noise::new(SEED);
        let mut a = CaveLattice::new();
        let mut b = CaveLattice::new();
        a.fill(&noise, 0, 0);
        b.fill(&noise, CHUNK_CELLS, 0);

        for gy in 0..LAT {
            let right_of_a = gy * LAT + (LAT - 1);
            let left_of_b = gy * LAT;
            assert_eq!(a.strata[right_of_a], b.strata[left_of_b]);
            assert_eq!(a.ravine[right_of_a], b.ravine[left_of_b]);
            assert_eq!(a.warp_x[right_of_a], b.warp_x[left_of_b]);
            assert_eq!(a.warp_y[right_of_a], b.warp_y[left_of_b]);
            assert_eq!(a.bias[right_of_a], b.bias[left_of_b]);
            assert_eq!(a.tunmod[right_of_a], b.tunmod[left_of_b]);
            assert_eq!(a.liq[right_of_a], b.liq[left_of_b]);
        }

        // Vertically too.
        let mut c = CaveLattice::new();
        c.fill(&noise, 0, CHUNK_CELLS);
        for gx in 0..LAT {
            assert_eq!(a.strata[(LAT - 1) * LAT + gx], c.strata[gx]);
            assert_eq!(a.liq[(LAT - 1) * LAT + gx], c.liq[gx]);
        }
    }

    #[test]
    fn a_carved_cell_is_a_pure_function_of_its_coordinates() {
        let noise = Noise::new(SEED);
        let cc = test_column();
        let mut lattice = CaveLattice::new();
        lattice.fill(&noise, 0, 192);

        for ly in 0..CHUNK_CELLS {
            for lx in (0..CHUNK_CELLS).step_by(7) {
                let (wcx, wcy) = (lx, 192 + ly);
                let depth = wcy - cc.surf;
                let first = lattice.carve(&noise, wcx, wcy, depth, lx, ly, &cc);
                let again = lattice.carve(&noise, wcx, wcy, depth, lx, ly, &cc);
                assert_eq!(first, again, "carve is not pure at ({wcx},{wcy})");
                assert_eq!(
                    carve_exact(&noise, wcx, wcy, depth, &cc),
                    carve_exact(&noise, wcx, wcy, depth, &cc),
                    "carve_exact is not pure at ({wcx},{wcy})"
                );
            }
        }
    }

    #[test]
    fn every_band_carves_to_something_and_the_floor_holds() {
        let noise = Noise::new(SEED);
        let cc = test_column();
        let mut lattice = CaveLattice::new();

        let mut saw_solid = false;
        let mut saw_air = false;
        let mut saw_liquid = false;

        // Sweep the whole vertical extent of the world, a chunk at a time.
        let mut chunk_y = 0;
        while chunk_y < UNDERWORLD_FLOOR + 2 * CHUNK_CELLS {
            lattice.fill(&noise, 0, chunk_y);
            for ly in (0..CHUNK_CELLS).step_by(3) {
                for lx in (0..CHUNK_CELLS).step_by(5) {
                    let wcy = chunk_y + ly;
                    let depth = wcy - cc.surf;
                    let c = lattice.carve(&noise, lx, wcy, depth, lx, ly, &cc);
                    match c {
                        Carve::Solid => saw_solid = true,
                        Carve::Air => saw_air = true,
                        Carve::Liquid => saw_liquid = true,
                    }
                    // Below the floor there is only bedrock, forever.
                    if f64::from(depth) + cc.band_shift >= f64::from(UNDERWORLD_FLOOR) {
                        assert_eq!(c, Carve::Solid, "the world has a hole in its floor");
                    }
                    // Inside the topsoil the crust must be sealed unless a ravine
                    // opened it, and this column's ravine window starts at 30.
                    if depth >= 0 && f64::from(depth) < cc.cap_t {
                        assert_eq!(c, Carve::Solid, "the topsoil is a sponge at depth {depth}");
                    }
                }
            }
            chunk_y += CHUNK_CELLS;
        }

        assert!(
            saw_solid && saw_air && saw_liquid,
            "a whole carve class never occurred"
        );
    }

    #[test]
    fn the_probe_path_tracks_the_lattice_path() {
        // `carve_exact` may disagree with `carve` only by the lattice
        // interpolation error, on cells within a hair of a threshold. Anything
        // more means the two paths have structurally drifted.
        let noise = Noise::new(SEED);
        let cc = test_column();
        let mut lattice = CaveLattice::new();
        let base_y = 256;
        lattice.fill(&noise, 0, base_y);

        let mut disagreements = 0;
        let mut total = 0;
        for ly in 0..CHUNK_CELLS {
            for lx in 0..CHUNK_CELLS {
                let wcy = base_y + ly;
                let depth = wcy - cc.surf;
                total += 1;
                if lattice.carve(&noise, lx, wcy, depth, lx, ly, &cc)
                    != carve_exact(&noise, lx, wcy, depth, &cc)
                {
                    disagreements += 1;
                }
            }
        }
        assert!(
            disagreements * 20 < total,
            "{disagreements}/{total} cells disagree — the paths have drifted"
        );
    }

    #[test]
    fn the_liquid_table_stays_inside_its_advertised_swing() {
        let noise = Noise::new(SEED);
        for wcx in -500..500 {
            let d = liquid_table_at(&noise, wcx);
            assert!(
                (LIQ_BASE - LIQ_WET_GAIN..=LIQ_BASE + LIQ_WET_GAIN).contains(&d),
                "liquid table {d} at column {wcx}"
            );
        }
    }
}

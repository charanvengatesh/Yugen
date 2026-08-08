//! Ore.
//!
//! The terrain generator already paints each underground layer's three vein
//! tiers, but those are ordinary rock — crystal, obsidian, gravel — smeared
//! through the deep band by an fBm threshold. They give the rock a grain; they
//! do not give anyone a reason to dig HERE rather than THERE. This pass adds the
//! thing you prospect for.
//!
//! Three ideas carry it:
//!
//!   1. DEPTH IS VALUE. Every ore has a depth response — silent, then ramping
//!      in, then holding, then decaying to a thin residual tail. Coal is a
//!      surface fuel and is effectively gone by the deep line; gems only begin
//!      well below it. Descending changes what the rock is worth, continuously,
//!      with no banded seam anywhere.
//!
//!   2. DEPOSITS, NOT SPECKLE. A deposit is a lens, a streak or a pod grown from
//!      one lattice origin: a thing you can see from across a cavern, mine out,
//!      and feel you have exhausted. Salt-and-pepper ore is invisible and
//!      unsatisfying to work.
//!
//!   3. PLACE HAS CHARACTER. A low-frequency richness field carves the world
//!      into rich and barren districts, and the underground layer re-weights the
//!      whole table — magma runs gold and gem, geode hollows are gem country,
//!      flooded grottos are copper, fungal depths are coal and little else. A
//!      volcanic surface drags the entire column shallower, so a volcano is a
//!      shortcut into the deep economy.
//!
//! Everything below obeys the decorator contract in [`super`]: a deposit is a
//! pure function of its origin's absolute coordinates, so every chunk it touches
//! derives the identical deposit and paints only its own share. No neighbour
//! reads, no state carried between chunks, no PRNG.

use super::{DecorContext, Decorator, Lattice, origin_cells};
use crate::config::worldgen::{CAVERN_DEPTH, DEEP_DEPTH};
use crate::sim::biomes::{Biome, ColumnProfile, UG_COUNT, UndergroundLayerId};
use crate::sim::materials::{CellId, block};

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
const COAL: CellId = block::COAL_ORE;
const COPPER: CellId = block::COPPER_ORE;
const IRON: CellId = block::IRON_ORE;
const GOLD: CellId = block::GOLD_ORE;
const GEM: CellId = block::GEM_ORE;
const CRYSTAL: CellId = block::CRYSTAL;

// ---------------------------------------------------------------------------
// Origin lattice
// ---------------------------------------------------------------------------
// Ore is evaluated over the entire underground rather than a single surface
// row, so the lattice has to be genuinely sparse or the cost scales with the
// whole streamed volume. One candidate per 14x12 cells is ~6 candidates per
// 32x32 chunk before the reach margin, and the cheap gate below throws roughly
// half of those away without touching noise or the column profile.
const STRIDE_X: i32 = 14;
const STRIDE_Y: i32 = 12;
const PHASE_X: i32 = 5;
const PHASE_Y: i32 = 7;

/// Farthest any deposit reaches from its origin, and the number the whole shape
/// table is sized against. The binding case is the mother-lode streak: its
/// sweep centre travels at most `round(7 + 1.2)` = 8 cells from the origin
/// (half-length 7, wobble +-1.2), and the radius-2.2 disc around that centre
/// fills at most 2 cells (3 fails `dx*dx+dy*dy <= 4.84`), so 10. Every other
/// shape is smaller: normal streaks reach 8, geodes 6, lenses 5. 11 is one cell
/// of slack over the true bound, and keeping it there is what keeps the
/// candidate scan — which runs over the entire underground — small.
const REACH: i32 = 11;

// Independent hash streams for one origin. Offsets are pairwise distinct mod
// STRIDE_Y, so no origin's stream can ever land on another origin's stream.
const H_SET: i32 = 90001; // set-piece gate            (mod 12 == 1)
const H_SHAPE: i32 = 190007; // streak vs. lens        (mod 12 == 11)
const H_SIZE: i32 = 290009; // primary size roll       (mod 12 == 5)
const H_RAD: i32 = 490002; // secondary size roll      (mod 12 == 6)
const H_ANG: i32 = 390007; // streak bearing           (mod 12 == 7)
// Per-cell streams are offset on X instead, so they cannot collide with the
// per-origin streams above.
const H_RIM: i32 = 7331; // ragged deposit edge
const H_STUD: i32 = 5701; // gem studs inside a geode

// Low-frequency "is this district worth prospecting" field. One octave of value
// noise — cheap, and an extra octave buys nothing at this scale.
const RICH_FREQ: f64 = 0.007;
const RICH_ANCHOR_X: f64 = 611.3;
const RICH_ANCHOR_Y: f64 = 118.9;
const RICH_SWING: f64 = 0.45; // richness spans 0.55x .. 1.45x

/// Upper bound used to reject a candidate before any noise or profile work. Real
/// totals sit well under this (the depth responses barely overlap), so the clamp
/// is close to inert and the gate discards ~45% of candidates for one hash.
const WEIGHT_CEILING: f64 = 0.55;

// Shallowest depth that may hold ore at all, so deposits never chew up through
// the topsoil cap from below.
const MIN_DEPTH: i32 = 8;

/// `Math.round`, JavaScript's tie rule.
///
/// JS breaks halves toward +infinity; Rust's `f64::round` breaks them away from
/// zero. [`paint_streak`] rounds sweep offsets that are negative for half of
/// every streak, so without this a streak would be a cell off from the
/// TypeScript's on one end. [`crate::sim::worldgen::heightmap`] carries the same
/// helper for the same reason.
#[inline]
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

// ---------------------------------------------------------------------------
// The ore table
// ---------------------------------------------------------------------------

/// Which set-piece a deposit is promoted to, if any.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SetKind {
    None,
    /// Long, thick, unmistakable mother-lode streak.
    Lode,
    /// Crystal shell studded with gems.
    Geode,
}

struct OreDef {
    code: CellId,
    /// Diagnostics only — nothing in the pass branches on it, and the tests
    /// name an ore with it when they report which one misbehaved.
    #[allow(dead_code)]
    id: &'static str,
    /// Depth response, in cells below the column's surface: zero above `d0`,
    /// ramps to full over `d0..d1`, holds to `d2`, decays to `tail` by `d3` and
    /// stays there for the rest of the endless deep.
    d0: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    tail: f64,
    /// Chance a lattice origin becomes this ore where its response is full.
    peak: f64,
    /// Chance the deposit is a streak rather than a lens.
    vein_chance: f64,
    // Lens radii: horizontal then vertical, as base + span.
    rx0: f64,
    rx_span: f64,
    ry0: f64,
    ry_span: f64,
    // Streak half-length and sweep radius, as base + span.
    half0: f64,
    half_span: f64,
    rad0: f64,
    rad_span: f64,
    /// Chance this deposit is promoted to its set-piece instead.
    set_chance: f64,
    set_kind: SetKind,
}

/// "Never fades" — gold and gem hold their full weight into the endless deep.
const HUGE: f64 = 1e9;

/// Phase scale for a streak's sine wobble: a size roll in [0,1) is multiplied by
/// this to spread the wobble over a full period.
///
/// Deliberately 6.28 and NOT [`std::f64::consts::TAU`]. It is a hand-written
/// approximation in the TypeScript, and the difference — 0.0032 of a radian —
/// moves the wobble of every streak in the world. `clippy::approx_constant` is
/// right that this looks like a mistake and wrong that fixing it would be an
/// improvement.
#[allow(clippy::approx_constant)]
const PHASE_SCALE: f64 = 6.28;

const CAVERN: f64 = CAVERN_DEPTH as f64;
const DEEP: f64 = DEEP_DEPTH as f64;

/// Bands are written against [`CAVERN_DEPTH`] (110) and [`DEEP_DEPTH`] (210) so
/// the ore economy tracks the terrain's own bands if those ever move.
#[rustfmt::skip]
static ORES: &[OreDef] = &[
    // Coal — the shallow fuel. Everywhere in the first cavern, essentially gone
    // by the deep line. Flat lenticular seams, wider than they are tall, because
    // that is what a coal measure looks like and it reads instantly as "seam".
    // It burns, so a seam beside a magma pocket is a hazard as much as a prize.
    OreDef {
        code: COAL, id: "coal",
        d0: 9.0, d1: 20.0, d2: CAVERN * 0.7, d3: CAVERN + 60.0, tail: 0.045,
        peak: 0.26,
        vein_chance: 0.0,
        rx0: 2.2, rx_span: 1.8, ry0: 1.0, ry_span: 1.0,
        half0: 0.0, half_span: 0.0, rad0: 0.0, rad_span: 0.0,
        set_chance: 0.0, set_kind: SetKind::None,
    },
    // Copper — shallow-to-mid, and the streakiest of the metals. Overlaps coal at
    // the top of its range and iron at the bottom, so no depth feels empty.
    OreDef {
        code: COPPER, id: "copper",
        d0: 18.0, d1: 45.0, d2: DEEP - 30.0, d3: DEEP + 140.0, tail: 0.055,
        peak: 0.115,
        vein_chance: 0.55,
        rx0: 1.6, rx_span: 1.4, ry0: 1.4, ry_span: 1.3,
        half0: 2.2, half_span: 2.3, rad0: 1.0, rad_span: 0.5,
        set_chance: 0.0, set_kind: SetKind::None,
    },
    // Iron — the mid-depth staple, and the one ore that stays worth mining all
    // the way down thanks to a fat residual tail.
    OreDef {
        code: IRON, id: "iron",
        d0: CAVERN * 0.6, d1: CAVERN + 40.0, d2: DEEP + 250.0, d3: 900.0, tail: 0.35,
        peak: 0.14,
        vein_chance: 0.45,
        rx0: 1.8, rx_span: 1.5, ry0: 1.6, ry_span: 1.3,
        half0: 2.6, half_span: 2.6, rad0: 1.0, rad_span: 0.6,
        set_chance: 0.02, set_kind: SetKind::Lode,
    },
    // Gold — begins around the deep line and never fades. Small, tight, nuggety
    // pockets, faintly emissive, so a pocket is a glimmer in a dark shaft rather
    // than a wall of yellow. Rarely a mother-lode.
    OreDef {
        code: GOLD, id: "gold",
        d0: DEEP - 40.0, d1: DEEP + 150.0, d2: HUGE, d3: HUGE + 1.0, tail: 1.0,
        peak: 0.055,
        vein_chance: 0.25,
        rx0: 1.2, rx_span: 1.0, ry0: 1.1, ry_span: 0.9,
        half0: 2.2, half_span: 1.8, rad0: 1.0, rad_span: 0.4,
        set_chance: 0.035, set_kind: SetKind::Lode,
    },
    // Gem — the deepest and by a wide margin the rarest. Tiny, strongly emissive
    // pods, so finding one is an event rather than a resource tier. Its set-piece,
    // the geode, is the rarest bounded structure in the world.
    OreDef {
        code: GEM, id: "gem",
        d0: DEEP + 90.0, d1: DEEP + 420.0, d2: HUGE, d3: HUGE + 1.0, tail: 1.0,
        peak: 0.018,
        vein_chance: 0.0,
        rx0: 0.9, rx_span: 0.8, ry0: 0.9, ry_span: 0.7,
        half0: 0.0, half_span: 0.0, rad0: 0.0, rad_span: 0.0,
        set_chance: 0.06, set_kind: SetKind::Geode,
    },
];

const N_ORES: usize = ORES.len();

// ---------------------------------------------------------------------------
// Layer and biome flavour
// ---------------------------------------------------------------------------

/// Per-underground-layer multipliers, parallel to [`ORES`]. This is the lever
/// that makes the deep world worth reading: the same depth pays very differently
/// depending on which layer you are standing in.
///
/// Indexed by [`UndergroundLayerId::index`], so this array is in palette order
/// and must stay that way. The TypeScript keyed a record by layer id and fell
/// back to a `NEUTRAL` row of all-1s for an unlisted layer; every layer is
/// listed, here as there, so that fallback was already dead.
#[rustfmt::skip]
static LAYER_MULT: [[f64; N_ORES]; UG_COUNT] = [
    //             coal  copper  iron  gold   gem
    /* caverns */ [1.0,  1.0,   1.15,  1.0,   0.8],
    /* grottos */ [0.7,  1.9,    0.8,  0.7,   0.6], // water-worked copper
    /* magma   */ [0.35, 0.7,    1.2,  2.0,   1.7], // coal has long since burnt
    /* geode   */ [0.5,  0.6,    0.6,  0.9,   3.0], // gem country
    /* fungal  */ [1.6,  0.5,    0.4,  0.35,  0.3], // organic, metal-poor
    // Frozen ground gives up very little. Ore in the Rime Hollows is what the
    // ice happened to close around rather than what the rock made, so every
    // metal is scarce and the one thing it has more of than anywhere but the
    // geodes is crystal — which is what ice does to water given long enough.
    /* rime    */ [0.6,  0.45,   0.5,  0.4,   1.4],
];

#[inline]
fn mult_for(l: UndergroundLayerId) -> &'static [f64; N_ORES] {
    &LAYER_MULT[l.index()]
}

/// Weight a named surface biome carries in this column, in [0,1].
fn surf_weight(col: &ColumnProfile, b: Biome) -> f64 {
    if col.surf_a == b {
        return 1.0 - col.surf_t;
    }
    if col.surf_b == b {
        return col.surf_t;
    }
    0.0
}

#[inline]
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Depth response of one ore at `d` cells below the surface.
fn depth_weight(o: &OreDef, d: f64) -> f64 {
    if d < o.d0 {
        return 0.0;
    }
    if d < o.d1 {
        return (d - o.d0) / (o.d1 - o.d0);
    }
    if d <= o.d2 {
        return 1.0;
    }
    if d >= o.d3 {
        return o.tail;
    }
    1.0 + (o.tail - 1.0) * ((d - o.d2) / (o.d3 - o.d2))
}

// ---------------------------------------------------------------------------
// Per-column memo
// ---------------------------------------------------------------------------
// The lattice walks Y outermost, so each candidate column is revisited once per
// row of origins. Surface height and the column profile are pure in `wcx`, so
// memoising them inside a single decorate call is free.
//
// The TypeScript held this in three module-level arrays and cleared them on
// entry and exit so nothing survived between chunks. That discipline is not
// available here — chunks generate in parallel on a rayon pool, where a module
// static would be a data race and a `thread_local!` would be a per-thread
// determinism hazard wearing a disguise. So the table is a plain local owned by
// [`OreDecorator::decorate`] and threaded through as `&mut`: it cannot outlive
// one chunk, and the compiler is the one enforcing it.
const MEMO_N: usize = 8;

struct Memo {
    x: [i32; MEMO_N],
    surf: [i32; MEMO_N],
    prof: [Option<ColumnProfile>; MEMO_N],
}

impl Memo {
    fn new() -> Memo {
        Memo {
            x: [0; MEMO_N],
            surf: [0; MEMO_N],
            prof: [None; MEMO_N],
        }
    }

    /// Surface row and profile for `wcx`, computing them only on a miss.
    ///
    /// Direct-mapped on the low bits of the column, exactly as the TypeScript's
    /// `wcx & (MEMO_N - 1)` — and negative columns land in the same slots here
    /// as there, both being two's complement.
    fn get(&mut self, ctx: &mut DecorContext<'_>, wcx: i32) -> (i32, ColumnProfile) {
        let i = (wcx & (MEMO_N as i32 - 1)) as usize;
        if self.prof[i].is_none() || self.x[i] != wcx {
            self.x[i] = wcx;
            self.surf[i] = ctx.surface_at(wcx);
            self.prof[i] = Some(ctx.profile_at(wcx));
        }
        (self.surf[i], self.prof[i].expect("just filled"))
    }
}

// ---------------------------------------------------------------------------
// Deposit shapes
// ---------------------------------------------------------------------------

/// Elliptical lens with a ragged rim. Solid to 0.8 of the normalised radius,
/// then dissolving, so the deposit has a body and a fringe of flecks instead of
/// a drawn outline. The rim draw is keyed on absolute cell coords, so both sides
/// of a chunk boundary dissolve exactly the same cells.
fn paint_lens(ctx: &mut DecorContext<'_>, ox: i32, oy: i32, rx: f64, ry: f64, code: CellId) {
    let bx = rx.ceil() as i32 + 1;
    let by = ry.ceil() as i32 + 1;
    let ix2 = 1.0 / (rx * rx);
    let iy2 = 1.0 / (ry * ry);
    for dy in -by..=by {
        for dx in -bx..=bx {
            let d = f64::from(dx * dx) * ix2 + f64::from(dy * dy) * iy2;
            if d > 1.3 {
                continue;
            }
            if d > 0.8 && ctx.hash(ox + dx + H_RIM, oy + dy) < (d - 0.8) * 1.9 {
                continue;
            }
            ctx.plot_if_solid(ox + dx, oy + dy, code);
        }
    }
}

/// A streak: a disc of radius `rad` swept along a bearing through the origin,
/// bending on a slow sine so it snakes rather than ruling a line. Centred on the
/// origin, which is what makes [`REACH`] half the streak's length instead of all
/// of it.
#[allow(clippy::too_many_arguments)]
fn paint_streak(
    ctx: &mut DecorContext<'_>,
    ox: i32,
    oy: i32,
    half: f64,
    rad: f64,
    ang: f64,
    phase: f64,
    code: CellId,
) {
    let ax = ang.cos();
    let ay = ang.sin();
    let br = rad.ceil() as i32;
    let r2 = rad * rad;
    let steps = js_round(half) as i32;
    for t in -steps..=steps {
        let tf = f64::from(t);
        let w = (tf * 0.55 + phase).sin() * 1.2; // |wobble| <= 1.2, folded into REACH
        let px = ox + js_round(tf * ax - w * ay) as i32;
        let py = oy + js_round(tf * ay + w * ax) as i32;
        for dy in -br..=br {
            for dx in -br..=br {
                if f64::from(dx * dx + dy * dy) > r2 {
                    continue;
                }
                ctx.plot_if_solid(px + dx, py + dy, code);
            }
        }
    }
}

/// A geode: a crystal shell around an interior of crystal studded with gems.
/// Deliberately solid rather than hollow — carving air this deep could open into
/// a cave system or leave rock floating, and a glowing pocket you have to break
/// into reads better than one you fall through anyway.
fn paint_geode(ctx: &mut DecorContext<'_>, ox: i32, oy: i32, r: f64) {
    let b = r.ceil() as i32 + 1;
    let or2 = 1.0 / (r * r);
    let inner = r - 1.35;
    let ir2 = if inner > 0.9 {
        1.0 / (inner * inner)
    } else {
        0.0
    };
    for dy in -b..=b {
        for dx in -b..=b {
            let q = f64::from(dx * dx + dy * dy);
            let d = q * or2;
            if d > 1.15 {
                continue;
            }
            if d > 0.85 && ctx.hash(ox + dx + H_RIM, oy + dy) < (d - 0.85) * 3.0 {
                continue;
            }
            let mut code = CRYSTAL;
            if ir2 > 0.0 && q * ir2 < 1.0 {
                code = if ctx.hash(ox + dx + H_STUD, oy + dy) < 0.42 {
                    GEM
                } else {
                    CRYSTAL
                };
            }
            ctx.plot_if_solid(ox + dx, oy + dy, code);
        }
    }
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

fn decorate_origin(ctx: &mut DecorContext<'_>, memo: &mut Memo, wcx: i32, wcy: i32) {
    // One draw does double duty: the cheap reject, and the weighted pick. This
    // bails before any noise or profile work for the majority of candidates.
    let u = ctx.hash(wcx, wcy);
    if u >= WEIGHT_CEILING {
        return;
    }

    let (surf, col) = memo.get(ctx, wcx);
    let depth = wcy - surf;
    if depth < MIN_DEPTH {
        return;
    }
    let depth = f64::from(depth);

    // Surface flavour. Volcanism drags the whole column's economy upward: on a
    // volcano the deep ores surface early, which makes a volcano a shortcut worth
    // walking to. The other biomes only nudge the shallow end.
    let volc = surf_weight(&col, Biome::Volcanic);
    let organic = surf_weight(&col, Biome::Swamp) + surf_weight(&col, Biome::Jungle);
    let arid = surf_weight(&col, Biome::Desert);
    let eff_depth = depth * (1.0 + 0.55 * volc);

    // Layer flavour, faded in over the same band the terrain uses to hand a
    // column over from its biome rock to its layer rock — so ore character and
    // rock character change together instead of at two different lines.
    let lf = clamp01((depth - col.ug_fade_start) / (col.ug_fade_end - col.ug_fade_start));
    let t = col.ug_t;
    let m_top = mult_for(col.ug.top);
    let m_sec = mult_for(col.ug.second);

    // Prospecting districts: rich and barren country, ~140 cells across.
    let rich = 1.0
        + RICH_SWING
            * ctx.noise.n2(
                f64::from(wcx) * RICH_FREQ + RICH_ANCHOR_X,
                f64::from(wcy) * RICH_FREQ + RICH_ANCHOR_Y,
            );

    // Selection weights for the origin under consideration. Written and consumed
    // inside one candidate test, never read across candidates — a module-level
    // scratch array in the TypeScript, a stack local here.
    let mut weights = [0.0f64; N_ORES];
    let mut total = 0.0;
    for (i, o) in ORES.iter().enumerate() {
        let mut w = depth_weight(o, eff_depth);
        if w > 0.0 {
            let layer = m_top[i] * (1.0 - t) + m_sec[i] * t;
            w *= o.peak * rich * (1.0 + (layer - 1.0) * lf);
            if i == 0 {
                w *= 1.0 + 0.35 * organic - 0.4 * arid - 0.4 * volc;
            } else if i == 1 {
                w *= 1.0 + 0.2 * arid;
            } else if i >= 3 {
                w *= 1.0 + 0.3 * volc;
            }
            if w < 0.0 {
                w = 0.0;
            }
        }
        weights[i] = w;
        total += w;
    }
    // Near-inert in practice; it is what lets the gate above be an honest bound.
    if total > WEIGHT_CEILING {
        total = WEIGHT_CEILING;
    }
    if u >= total {
        return;
    }

    let mut pick = N_ORES - 1;
    let mut run = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        run += w;
        if u < run {
            pick = i;
            break;
        }
    }
    let ore = &ORES[pick];

    // Rare set-piece promotion.
    if ore.set_chance > 0.0 && ctx.hash(wcx, wcy + H_SET) < ore.set_chance {
        let s = ctx.hash(wcx, wcy + H_SIZE);
        if ore.set_kind == SetKind::Geode {
            paint_geode(ctx, wcx, wcy, 3.5 + s * 2.5); // radius 3.5 .. 6
            return;
        }
        // Mother-lode: half-length 6..7, swept at radius 1.6..2.2 — the widest and
        // longest thing this pass can make, and sized to sit inside REACH.
        let ang = ctx.hash(wcx, wcy + H_ANG) * std::f64::consts::PI * 2.0;
        paint_streak(
            ctx,
            wcx,
            wcy,
            6.0 + s,
            1.6 + s * 0.6,
            ang,
            s * PHASE_SCALE,
            ore.code,
        );
        return;
    }

    let s1 = ctx.hash(wcx, wcy + H_SIZE);
    let s2 = ctx.hash(wcx, wcy + H_RAD);
    if ore.vein_chance > 0.0 && ctx.hash(wcx, wcy + H_SHAPE) < ore.vein_chance {
        let ang = ctx.hash(wcx, wcy + H_ANG) * std::f64::consts::PI * 2.0;
        paint_streak(
            ctx,
            wcx,
            wcy,
            ore.half0 + s1 * ore.half_span,
            ore.rad0 + s2 * ore.rad_span,
            ang,
            s1 * PHASE_SCALE,
            ore.code,
        );
        return;
    }
    paint_lens(
        ctx,
        wcx,
        wcy,
        ore.rx0 + s1 * ore.rx_span,
        ore.ry0 + s2 * ore.ry_span,
        ore.code,
    );
}

/// Ore deposits over the whole underground, grown from a sparse 2D lattice of
/// origins. Depth sets what can appear, the underground layer and surface biome
/// re-weight it, and the richness field decides whether this district is worth
/// the shaft at all.
pub struct OreDecorator;

impl Decorator for OreDecorator {
    fn name(&self) -> &'static str {
        "ores"
    }

    fn reach_x(&self) -> i32 {
        REACH
    }

    fn reach_y(&self) -> i32 {
        REACH
    }

    fn decorate(&self, ctx: &mut DecorContext<'_>) {
        // Born empty and dropped at the end of this call, so nothing survives
        // between chunks — the TypeScript's clear-on-entry/clear-on-exit pair,
        // made structural.
        let mut memo = Memo::new();
        for (wcx, wcy) in origin_cells(
            ctx.base_x,
            ctx.base_y,
            REACH,
            REACH,
            Lattice {
                stride_x: STRIDE_X,
                stride_y: STRIDE_Y,
                phase_x: PHASE_X,
                phase_y: PHASE_Y,
            },
        ) {
            decorate_origin(ctx, &mut memo, wcx, wcy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::world::{CHUNK_CELLS, pmod};
    use crate::config::worldgen::SEED;
    use crate::sim::noise::Noise;
    use crate::sim::worldgen::heightmap::Heightmap;

    /// Ore only replaces solid rock, so a test chunk has to start as rock.
    const ROCK: CellId = block::STONE;

    fn chunk(noise: &Noise, hm: &mut Heightmap, base_x: i32, base_y: i32) -> Vec<CellId> {
        let mut out = vec![ROCK; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        {
            let mut ctx = DecorContext::new(noise, SEED, base_x, base_y, &mut out, hm);
            OreDecorator.decorate(&mut ctx);
        }
        out
    }

    /// A `w` x `h` cell canvas anchored at (x0, y0), tiled from chunks whose
    /// top-left corners start at (x0, y0).
    ///
    /// Moving (x0, y0) moves the CHUNK GRID, not the world: the decorator is a
    /// pure function of absolute coordinates, so two canvases over the same
    /// region under different alignments must agree cell for cell.
    fn canvas(noise: &Noise, x0: i32, y0: i32, w: i32, h: i32) -> Vec<CellId> {
        assert_eq!(w % CHUNK_CELLS, 0);
        assert_eq!(h % CHUNK_CELLS, 0);
        let mut hm = Heightmap::new();
        let mut buf = vec![ROCK; (w * h) as usize];
        let mut by = y0;
        while by < y0 + h {
            let mut bx = x0;
            while bx < x0 + w {
                let c = chunk(noise, &mut hm, bx, by);
                for ly in 0..CHUNK_CELLS {
                    for lx in 0..CHUNK_CELLS {
                        let cx = bx - x0 + lx;
                        let cy = by - y0 + ly;
                        buf[(cy * w + cx) as usize] = c[(ly * CHUNK_CELLS + lx) as usize];
                    }
                }
                bx += CHUNK_CELLS;
            }
            by += CHUNK_CELLS;
        }
        buf
    }

    #[test]
    fn the_tuned_numbers_are_the_tuned_ones() {
        assert_eq!((STRIDE_X, STRIDE_Y, PHASE_X, PHASE_Y), (14, 12, 5, 7));
        assert_eq!(REACH, 11);
        assert_eq!(MIN_DEPTH, 8);
        assert_eq!(N_ORES, 5);
        assert_eq!(
            (RICH_FREQ, RICH_ANCHOR_X, RICH_ANCHOR_Y, RICH_SWING),
            (0.007, 611.3, 118.9, 0.45)
        );
        assert_eq!(WEIGHT_CEILING, 0.55);
        assert_eq!(
            [H_SET, H_SHAPE, H_SIZE, H_RAD, H_ANG],
            [90001, 190007, 290009, 490002, 390007]
        );
        assert_eq!([H_RIM, H_STUD], [7331, 5701]);
    }

    #[test]
    fn the_per_origin_salts_are_distinct_mod_the_y_stride() {
        // The comment on the salts is a claim about collisions: two origins one
        // lattice row apart must not draw the same stream. That holds only while
        // the offsets stay pairwise distinct mod STRIDE_Y — and the residues the
        // TypeScript recorded are 1, 11, 5, 6, 7.
        let salts = [H_SET, H_SHAPE, H_SIZE, H_RAD, H_ANG];
        let residues: Vec<i32> = salts.iter().map(|&s| pmod(s, STRIDE_Y)).collect();
        assert_eq!(residues, vec![1, 11, 5, 6, 7]);
        for i in 0..residues.len() {
            for j in (i + 1)..residues.len() {
                assert_ne!(residues[i], residues[j], "salts {i} and {j} collide");
            }
        }
        // 0 is the gate's own stream (`ctx.hash(wcx, wcy)`), so no salt may land
        // on it either.
        assert!(!residues.contains(&0));
    }

    #[test]
    fn js_round_sends_halves_up_not_away_from_zero() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0); // f64::round would give -3
        assert_eq!(js_round(-2.6), -3.0);
        assert_eq!(js_round(-0.5), 0.0);
    }

    #[test]
    fn the_depth_response_is_continuous_and_bounded() {
        // A discontinuity here is a visible band in the rock. Walk every ore's
        // whole response and check it never jumps and never leaves [0,1].
        for o in ORES {
            let mut prev = depth_weight(o, 0.0);
            let mut d = 0.0;
            while d < 1500.0 {
                let w = depth_weight(o, d);
                assert!((0.0..=1.0).contains(&w), "{} weight {w} at depth {d}", o.id);
                assert!((w - prev).abs() < 0.05, "{} jumps at depth {d}", o.id);
                prev = w;
                d += 0.5;
            }
            assert_eq!(depth_weight(o, o.d0 - 1.0), 0.0, "{} starts early", o.id);
            assert_eq!(depth_weight(o, o.d1), 1.0, "{} never reaches full", o.id);
        }
    }

    #[test]
    fn every_ore_wins_somewhere_and_density_stays_under_the_ceiling() {
        // WEIGHT_CEILING is a declared upper bound the gate leans on: the summed
        // weights must genuinely stay under it, or the cheap reject is silently
        // discarding deposits it promised to keep. And an ore that never wins a
        // single origin is a content bug the eye would take a long time to find.
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let mut out = vec![ROCK; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let mut ctx = DecorContext::new(&noise, SEED, 0, 0, &mut out, &mut hm);
        let mut memo = Memo::new();

        let mut seen = [0usize; N_ORES];
        let mut candidates = 0usize;
        let mut over = 0usize;
        let mut deposits = 0usize;
        let mut wcy = 16;
        while wcy < 4000 {
            let mut wcx = -4000;
            while wcx < 4000 {
                candidates += 1;
                let (surf, col) = memo.get(&mut ctx, wcx);
                let depth = wcy - surf;
                if depth < MIN_DEPTH {
                    wcx += STRIDE_X;
                    continue;
                }
                let volc = surf_weight(&col, Biome::Volcanic);
                let organic = surf_weight(&col, Biome::Swamp) + surf_weight(&col, Biome::Jungle);
                let arid = surf_weight(&col, Biome::Desert);
                let eff = f64::from(depth) * (1.0 + 0.55 * volc);
                let lf = clamp01(
                    (f64::from(depth) - col.ug_fade_start) / (col.ug_fade_end - col.ug_fade_start),
                );
                let t = col.ug_t;
                let m_top = mult_for(col.ug.top);
                let m_sec = mult_for(col.ug.second);
                let rich = 1.0
                    + RICH_SWING
                        * ctx.noise.n2(
                            f64::from(wcx) * RICH_FREQ + RICH_ANCHOR_X,
                            f64::from(wcy) * RICH_FREQ + RICH_ANCHOR_Y,
                        );
                let mut total = 0.0;
                let mut w_each = [0.0f64; N_ORES];
                for (i, o) in ORES.iter().enumerate() {
                    let mut w = depth_weight(o, eff);
                    if w > 0.0 {
                        let layer = m_top[i] * (1.0 - t) + m_sec[i] * t;
                        w *= o.peak * rich * (1.0 + (layer - 1.0) * lf);
                        if i == 0 {
                            w *= 1.0 + 0.35 * organic - 0.4 * arid - 0.4 * volc;
                        } else if i == 1 {
                            w *= 1.0 + 0.2 * arid;
                        } else if i >= 3 {
                            w *= 1.0 + 0.3 * volc;
                        }
                        if w < 0.0 {
                            w = 0.0;
                        }
                    }
                    w_each[i] = w;
                    total += w;
                }
                if total > WEIGHT_CEILING {
                    // The clamp the pass applies, and the reason the cheap gate
                    // is an honest bound even where the weights overshoot.
                    over += 1;
                    total = WEIGHT_CEILING;
                }
                let u = ctx.hash(wcx, wcy);
                if u < total {
                    deposits += 1;
                    let mut run = 0.0;
                    for (i, &w) in w_each.iter().enumerate() {
                        run += w;
                        if u < run {
                            seen[i] += 1;
                            break;
                        }
                    }
                }
                wcx += STRIDE_X;
            }
            wcy += STRIDE_Y;
        }
        for (i, &n) in seen.iter().enumerate() {
            assert!(n > 0, "{} never wins a lattice origin", ORES[i].id);
        }
        // "Near-inert in practice" — the TypeScript's claim about the clamp,
        // measured. It fires on a fraction of a percent of candidates, where the
        // depth responses of two ores overlap in a rich district.
        let clamped = over as f64 / candidates as f64;
        assert!(
            clamped < 0.01,
            "the WEIGHT_CEILING clamp fires on {:.2}% of candidates — it is no \
             longer near-inert, and the gate is throwing deposits away",
            clamped * 100.0
        );
        let rate = deposits as f64 / candidates as f64;
        assert!(
            (0.01..WEIGHT_CEILING).contains(&rate),
            "{:.2}% of lattice origins become deposits",
            rate * 100.0
        );
    }

    #[test]
    fn a_deposit_straddling_a_seam_is_painted_identically_from_both_sides() {
        // THE contract test. The same deep region, generated under two different
        // chunk alignments: a lode split across a seam has to come out the same
        // from either side, or the world tears where the player did not walk.
        let noise = Noise::new(SEED);
        let a = canvas(&noise, 0, 256, 128, 128);
        let b = canvas(&noise, 16, 272, 96, 96);

        let mut ore_cells = 0;
        for y in 272..368 {
            for x in 16..112 {
                let va = a[((y - 256) * 128 + x) as usize];
                let vb = b[((y - 272) * 96 + (x - 16)) as usize];
                assert_eq!(va, vb, "chunk alignments disagree at ({x}, {y})");
                if va != ROCK {
                    ore_cells += 1;
                }
            }
        }
        assert!(
            ore_cells > 200,
            "only {ore_cells} cells of ore in 96x96 — the test proved nothing"
        );
    }

    #[test]
    fn nothing_is_painted_outside_the_declared_reach() {
        // Reach is the promise the generator scans against, and this pass has a
        // 2D lattice, so it has to hold on BOTH axes. Slide the chunk grid a cell
        // at a time in x and in y: a deposit that outran REACH would be clipped
        // by some offset and the canvases would part company.
        let noise = Noise::new(SEED);
        let base = canvas(&noise, 0, 256, 128, 128);
        for shift in [1, 7, 11, 23] {
            let sx = canvas(&noise, shift, 256, 96, 128);
            for y in 0..128 {
                for x in shift..(shift + 96) {
                    assert_eq!(
                        base[(y * 128 + x) as usize],
                        sx[(y * 96 + (x - shift)) as usize],
                        "x shift {shift} disagrees at ({x}, {})",
                        y + 256
                    );
                }
            }
            let sy = canvas(&noise, 0, 256 + shift, 128, 96);
            for y in shift..(shift + 96) {
                for x in 0..128 {
                    assert_eq!(
                        base[(y * 128 + x) as usize],
                        sy[((y - shift) * 128 + x) as usize],
                        "y shift {shift} disagrees at ({x}, {})",
                        y + 256
                    );
                }
            }
        }
    }

    #[test]
    fn the_memo_never_changes_an_answer() {
        // The memo is an optimisation over two pure functions. If it ever hands
        // back another column's numbers — a stale slot, a missed tag check — every
        // deposit in the chunk moves.
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let mut out = vec![ROCK; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let mut ctx = DecorContext::new(&noise, SEED, 0, 0, &mut out, &mut hm);
        let mut memo = Memo::new();
        // Deliberately adversarial: columns that collide in the same slot, walked
        // out of order and revisited.
        for wcx in [0, 8, 16, -8, 0, -16, 8, 3, 11, 3] {
            let (surf, col) = memo.get(&mut ctx, wcx);
            assert_eq!(surf, ctx.surface_at(wcx), "memo surface wrong at {wcx}");
            assert_eq!(col.wcx, wcx, "memo handed back another column at {wcx}");
        }
    }

    #[test]
    fn the_decorator_declares_itself() {
        assert_eq!(OreDecorator.name(), "ores");
        assert_eq!(OreDecorator.reach_x(), REACH);
        assert_eq!(OreDecorator.reach_y(), REACH);
    }
}

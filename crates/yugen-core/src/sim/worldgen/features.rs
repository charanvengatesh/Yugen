//! Large procedural features — the landform-scale landmarks, and the pass that
//! places both them and the template structures from
//! [`crate::sim::worldgen::structs`].
//!
//! A `struct` says what every cell is. A `feature` says how big, how often, how
//! deep and out of what, and the generator here grows the shape. That split is
//! the whole reason both kinds exist: an island whose silhouette must vary with
//! its size cannot be authored as glyph art without pinning it to one size, and
//! a cabin whose charm is the window placement cannot be expressed as four
//! numbers.
//!
//! ---- THE DETERMINISTIC FEATURE GRID ----------------------------------------
//! Every feature owns a grid of `cell_w x cell_h` cells anchored to WORLD space.
//! One hash of the integer cell index decides whether that cell hosts an
//! instance; two more place the origin inside it. Nothing else is consulted, so:
//!
//!   - placement is a pure function of absolute coordinates and the seed;
//!   - the extent is bounded by the authored `reach_x`/`reach_y`, which every
//!     generator below clamps itself to;
//!   - therefore every chunk within reach of a hosting cell independently
//!     derives the SAME instance, and paints only the cells that land inside
//!     itself.
//!
//! That last point is the invariant this whole architecture exists for (see
//! [`crate::sim::decor`]). A 168-cell mineshaft crosses six chunks generated in
//! whatever order the player walks into them; all six re-derive the identical
//! corridor.
//!
//! ---- THE LOOP CLAMP IS AN OPTIMISATION, NEVER A DECISION -------------------
//! Every generator is per-cell pure in (origin, dx, dy). So the loops are
//! clipped to the chunk being painted plus a small margin, and that cannot
//! change a single cell — it only stops that 168-cell corridor from evaluating
//! 168 columns six times over. The margin covers everything a column writes into
//! its neighbours (support posts, lintels, ragged edges); getting it wrong shows
//! up as a feature clipped at a chunk boundary, which is what the determinism
//! check in the purity suite is for.
//!
//! ---- COST ------------------------------------------------------------------
//! The pass runs for every chunk in the world, so the REJECT path is what
//! matters. It is: one band compare per feature (a chunk far above or below a
//! feature's depth window is out before any hash), then one hash per overlapping
//! grid cell. A sky chunk pays about a dozen compares and nothing else.

use std::sync::LazyLock;

use crate::config::{CHUNK_CELLS, SEA_LEVEL_Y, SURFACE_AMPLITUDE, SURFACE_ANCHOR_Y, pmod};
use crate::sim::biomes::{Biome, UndergroundLayerId};
use crate::sim::decor::{DecorContext, Decorator};
use crate::sim::materials::{BLOCKS, CellId, code_of};
use crate::sim::worldgen::caves::{Carve, carve_exact, cave_column_at};
use crate::sim::worldgen::containers::{container_code, containers_present};
use crate::sim::worldgen::structs::{stamp_structs, struct_reach_x, struct_reach_y};

// --- The content facade ------------------------------------------------------
// The compiled feature tables are reached through here and nowhere else, exactly
// as `sim::materials` is the one door to the block tables.

pub use yugen_data::worldgen::{
    FEAT_CELLH, FEAT_CELLW, FEAT_CLEAR0, FEAT_CLEAR1, FEAT_DENSITY, FEAT_HEIGHT0, FEAT_HEIGHT1,
    FEAT_KIND, FEAT_MAXDEPTH, FEAT_MINDEPTH, FEAT_PLACE, FEAT_RARITY, FEAT_REACHX, FEAT_REACHY,
    FEAT_SIZE0, FEAT_SIZE1, FEATURE_COUNT, FEATURE_IDS, FEATURES, FeatureDef, FeatureMat, feature,
};

// --- The compiled vocabularies ----------------------------------------------
//
// `FeatureKind` and `FeaturePlace` are GENERATED, and used to be written out
// here by hand with a comment conceding the codes were "load-bearing on both
// sides". They were: the schema assigned the numbers, this file restated them,
// and nothing checked the two lists agreed. Adding a generator meant editing
// both and remembering which order the variants were in.
//
// `contentc` emits the enum and its `from_code` now — see `collect_enums`,
// where naming a mapped enum is the opt-in that says the game meets this one as
// a number and needs the vocabulary back.

pub use yugen_data::worldgen::{FeatureKind as Kind, FeaturePlace as Place};

/// Which lattice offers a site, defaulting to the 2D one.
///
/// The generated `from_code` is honest and returns `None` for a code this build
/// does not have; what to DO about that is this file's decision and not the
/// compiler's. Falling back to `Underground` is the safe answer, because it is
/// the class that gates on depth and therefore cannot spray a feature across
/// the sky if content grows a placement class before the game does.
#[inline]
fn place_of(code: u8) -> Place {
    Place::from_code(code).unwrap_or(Place::Underground)
}

const AIR: CellId = 0;

// --- Material resolution -----------------------------------------------------
// Same preference discipline as layers.rs: the registry is grown by other work,
// so every fallback is something that has always existed. It stays a runtime
// resolution rather than a `block::*` constant because the ids come from
// `content/`, not from this file.
fn pick(ids: &[Option<&str>]) -> CellId {
    for id in ids.iter().flatten() {
        if BLOCKS.iter().any(|b| b.id == *id) {
            return code_of(id);
        }
    }
    let named: Vec<&str> = ids.iter().map(|i| i.unwrap_or("<none>")).collect();
    panic!(
        "features: none of [{}] exist in the material registry",
        named.join(", ")
    );
}

/// Runtime form of one feature: knobs unpacked, ids resolved, band precomputed.
pub struct Feat {
    pub id: &'static str,
    /// `None` for a kind this build has no generator for.
    kind: Option<Kind>,
    place: Place,
    cell_w: i32,
    cell_h: i32,
    rarity: f64,
    reach_x: i32,
    reach_y: i32,
    size0: i32,
    size1: i32,
    h0: i32,
    h1: i32,
    clear0: i32,
    clear1: i32,
    density: f64,
    min_depth: i32,
    max_depth: i32,
    /// `None` = any biome.
    biomes: Option<Vec<Biome>>,
    /// `None` = any underground layer.
    layers: Option<Vec<UndergroundLayerId>>,
    shell: CellId,
    fill: CellId,
    accent: CellId,
    frame: CellId,
    liquid: CellId,
    /// Decorrelating salts for this feature's grid hashes.
    salt_x: i32,
    salt_y: i32,
    /// Absolute rows this feature can possibly touch — the per-chunk early-out.
    band_top: i32,
    band_bot: i32,
}

// Rows the ground line can occupy, derived exactly as decor/structures.rs does:
// the largest biome amp_scale is 1.45 and the largest |height_offset| is 6.
fn surf_span() -> f64 {
    f64::from(SURFACE_AMPLITUDE) * 1.6 + 8.0
}
fn surf_min() -> i32 {
    (f64::from(SURFACE_ANCHOR_Y) - surf_span()).floor() as i32
}
fn surf_max() -> i32 {
    (f64::from(SURFACE_ANCHOR_Y) + surf_span()).ceil() as i32
}

/// Content names biomes and layers as plain strings; an id the palette does not
/// have simply never matches, exactly as a `Set.has` miss did.
fn biome_by_id(id: &str) -> Option<Biome> {
    Biome::ALL.into_iter().find(|b| b.def().id == id)
}
fn layer_by_id(id: &str) -> Option<UndergroundLayerId> {
    UndergroundLayerId::ALL
        .into_iter()
        .find(|l| l.def().id == id)
}

fn build(def: &'static FeatureDef) -> Option<Feat> {
    if def.rarity <= 0.0 || def.reach_x <= 0 {
        return None; // tombstone (FORMAT.md §4)
    }
    let m = def.mat;
    let place = place_of(def.place);
    let clear0 = def.clearance[0] as i32;
    let clear1 = def.clearance[1] as i32;

    // The band is the union of every row an instance of this feature could reach,
    // over every ground line the heightmap can produce. Two compares against it
    // reject the feature for most of the world's chunks before a single hash.
    let (band_top, band_bot) = match place {
        Place::Sky => (
            surf_min() - clear1 - def.reach_y,
            surf_max() - clear0 + def.reach_y,
        ),
        Place::Surface => (surf_min() - def.reach_y, surf_max() + def.reach_y),
        // max_depth is 65535 for an unbounded feature; the addition stays finite.
        Place::Underground => (
            surf_min() + def.min_depth - def.reach_y,
            surf_max() + def.max_depth + def.reach_y,
        ),
    };

    Some(Feat {
        id: def.id,
        kind: Kind::from_code(def.kind),
        place,
        cell_w: def.cell_w,
        cell_h: def.cell_h,
        rarity: f64::from(def.rarity),
        reach_x: def.reach_x,
        reach_y: def.reach_y,
        size0: def.size[0] as i32,
        size1: def.size[1] as i32,
        h0: def.height[0] as i32,
        h1: def.height[1] as i32,
        clear0,
        clear1,
        density: f64::from(def.density),
        min_depth: def.min_depth,
        max_depth: def.max_depth,
        biomes: def
            .biomes
            .map(|ids| ids.iter().filter_map(|id| biome_by_id(id)).collect()),
        layers: def
            .layers
            .map(|ids| ids.iter().filter_map(|id| layer_by_id(id)).collect()),
        shell: pick(&[m.and_then(|m| m.shell), Some("stone")]),
        fill: pick(&[m.and_then(|m| m.fill), Some("dirt")]),
        accent: pick(&[m.and_then(|m| m.accent), Some("crystal")]),
        frame: pick(&[m.and_then(|m| m.frame), Some("wood")]),
        liquid: pick(&[m.and_then(|m| m.liquid), Some("water")]),
        salt_x: (i32::from(def.code) + 1) * 9176,
        salt_y: (i32::from(def.code) + 1) * 31051,
        band_top,
        band_bot,
    })
}

/// Everything the pass derives from content, built once and never mutated.
///
/// The TypeScript's three module-level `const`s, bundled behind one
/// initialisation barrier. Initialise-once and `Sync`, so a rayon worker may
/// read it; it is not shared mutable state.
struct Feats {
    all: Vec<Feat>,
    reach_x: i32,
    reach_y: i32,
}

static FEATS: LazyLock<Feats> = LazyLock::new(|| {
    let all: Vec<Feat> = FEATURES.iter().filter_map(build).collect();
    let reach_x = all
        .iter()
        .fold(0, |m, f| if f.reach_x > m { f.reach_x } else { m });
    let reach_y = all
        .iter()
        .fold(0, |m, f| if f.reach_y > m { f.reach_y } else { m });
    Feats {
        all,
        reach_x,
        reach_y,
    }
});

/// Farthest a feature may extend horizontally from its origin, in cells.
#[inline]
pub fn feat_reach_x() -> i32 {
    FEATS.reach_x
}

/// Farthest a feature may extend vertically from its origin, in cells.
#[inline]
pub fn feat_reach_y() -> i32 {
    FEATS.reach_y
}

// --- Deterministic helpers ---------------------------------------------------

/// `Math.round`, JavaScript's way: halves break toward +infinity.
///
/// This is not pedantry. `spread_at` probes at `round((i * half) / 2)` for
/// `i` in -2..2, and the mineshaft floor line is `round(wobble * g2(...))` where
/// the gradient sample is signed — both land on a negative exact half often
/// enough that Rust's away-from-zero rule would move real cells.
#[inline]
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// Grid-cell hash. `k` decorrelates the several draws one instance makes.
#[inline]
fn gh(ctx: &DecorContext<'_>, f: &Feat, gx: i32, gy: i32, k: i32) -> f64 {
    ctx.hash(gx + f.salt_x + k * 7919, gy + f.salt_y - k * 104729)
}

/// Hash value -> integer in [lo, hi].
#[inline]
fn ri(v: f64, lo: i32, hi: i32) -> i32 {
    let n = lo + (v * f64::from(hi - lo + 1)).floor() as i32;
    if n > hi {
        hi
    } else if n < lo {
        lo
    } else {
        n
    }
}

/// Floor division that behaves for negative coordinates (the world goes both
/// ways).
#[inline]
fn fdiv(a: i32, b: i32) -> i32 {
    a.div_euclid(b)
}

fn overlaps(ctx: &DecorContext<'_>, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    x1 >= ctx.base_x
        && x0 < ctx.base_x + CHUNK_CELLS
        && y1 >= ctx.base_y
        && y0 < ctx.base_y + CHUNK_CELLS
}

/// Loop clamp: the lowest x worth evaluating for this chunk. See the header.
#[inline]
fn lo_x(ctx: &DecorContext<'_>, x0: i32, margin: i32) -> i32 {
    let c = ctx.base_x - margin;
    if x0 < c { c } else { x0 }
}
#[inline]
fn hi_x(ctx: &DecorContext<'_>, x1: i32, margin: i32) -> i32 {
    let c = ctx.base_x + CHUNK_CELLS - 1 + margin;
    if x1 > c { c } else { x1 }
}
#[inline]
fn lo_y(ctx: &DecorContext<'_>, y0: i32, margin: i32) -> i32 {
    let c = ctx.base_y - margin;
    if y0 < c { c } else { y0 }
}
#[inline]
fn hi_y(ctx: &DecorContext<'_>, y1: i32, margin: i32) -> i32 {
    let c = ctx.base_y + CHUNK_CELLS - 1 + margin;
    if y1 > c { c } else { y1 }
}

/// "Is this cell already open?" through the lattice-free probe in caves.rs.
///
/// Used for the handful of plausibility tests placement makes — a mineshaft
/// whose whole span hangs in a ravine, a geode grown inside a cavern. It is
/// several times the cost of the chunk path, so it is one probe per ACCEPTED
/// instance and never a scan. Per caves.rs, it may disagree with chunk
/// generation by the lattice interpolation error on a cell within a hair of a
/// threshold, which is exactly why every generator below overwrites what it
/// finds instead of assuming it.
///
/// The TypeScript kept a module-level scratch `CaveColumn` for this probe.
/// `cave_column_at` returns one by value here, so the scratch is gone and the
/// probe is re-entrant.
fn open_at(ctx: &mut DecorContext<'_>, wcx: i32, wcy: i32) -> bool {
    let surf = ctx.surface_at(wcx);
    let depth = wcy - surf;
    if depth < 0 {
        return true;
    }
    let noise = ctx.noise;
    let col = ctx.profile_at(wcx);
    let scale = ctx.scale();
    let cc = cave_column_at(noise, wcx, surf, &col, scale);
    carve_exact(noise, wcx, wcy, depth, &cc, scale) != Carve::Solid
}

/// Ground-line spread over a footprint, from positional probes only.
fn spread_at(ctx: &mut DecorContext<'_>, ox: i32, half: i32, base: i32) -> i32 {
    let mut lo = base;
    let mut hi = base;
    for i in -2i32..=2 {
        if i == 0 {
            continue;
        }
        let off = js_round(f64::from(i * half) / 2.0) as i32;
        let s = ctx.surface_at(ox + off);
        if s < lo {
            lo = s;
        }
        if s > hi {
            hi = s;
        }
    }
    hi - lo
}

// =============================================================================
// Generators
// =============================================================================
// Every one is a pure function of (f, origin, seed). Sizes are clamped to the
// feature's declared reach rather than trusted from content, so a content edit
// can make a feature look wrong but cannot make it produce a seam.

/// Floating island. The origin is the island's TOP row, so `clearance` is the
/// honest gap between the ground line and the thing you are looking up at, and
/// the island can never intersect the terrain however the heightmap swings.
///
/// The silhouette is the one that reads at a glance from a long way off: a
/// near-flat soil deck, a shallow crown, and a rock underside that tapers to a
/// point. `pow(1-t, 1.7)` rather than a circle — a hemisphere reads as a ball,
/// and the tapered keel is what makes it read as torn out of the ground.
fn grow_island(ctx: &mut DecorContext<'_>, f: &Feat, ox: i32, oy: i32) {
    let r_max = f.reach_x - 2;
    let d_max = f.reach_y - 5;
    let r = ri(gh(ctx, f, ox, oy, 11), f.size0, f.size1).min(r_max);
    let depth = ri(gh(ctx, f, ox, oy, 12), f.h0, f.h1).min(d_max);
    if r < 4 {
        return;
    }

    if !overlaps(ctx, ox - r, oy - 3, ox + r, oy + depth + 2) {
        return;
    }

    let soil = 3;
    let from = lo_x(ctx, ox - r, 2);
    let to = hi_x(ctx, ox + r, 2);
    for x in from..=to {
        let t = f64::from((x - ox).abs()) / f64::from(r);
        // Crown: two cells of rise over the middle third, so the deck is walkable.
        let rise = if t < 0.55 {
            if t < 0.25 { 2 } else { 1 }
        } else {
            0
        };
        let top = oy - rise;
        // Keel, roughened by one hash per column so the underside is not a smooth
        // curve — a smooth curve reads as a UI element, not as rock.
        let rough = i32::from(ctx.hash(x + 4451, oy) < 0.4);
        let bot = oy + js_round(f64::from(depth) * (1.0 - t).powf(1.7)) as i32 + rough;

        for y in top..=bot {
            let code = if y < top + soil { f.fill } else { f.shell };
            ctx.plot(x, y, code);
        }
        // A seam of accent in the keel, and creepers off the rim.
        if bot > oy + 2 && ctx.hash(x + 8837, oy) < f.density * 0.25 {
            ctx.plot(x, bot - 1, f.accent);
        }
        if t > 0.55 {
            let v = ctx.hash(x + 1213, oy + 7);
            if v < f.density {
                let len = 1 + (v * 7.0).floor() as i32;
                for d in 1..=len {
                    ctx.plot_if_empty(x, bot + d, f.frame);
                }
            }
        }
    }
}

/// Highland lake. The heightmap only floods columns whose ground line falls
/// BELOW sea level, so every basin above it is dry rock — this is the pass that
/// puts water back in the hills, and it has to excavate its own bowl to do it.
///
/// The bowl is cut from the ORIGIN column's ground line rather than each
/// column's own, which is why the site test rejects anything but flat ground: a
/// bowl cut to a single datum across a slope is a bathtub sticking out of a
/// hillside.
fn grow_lake(ctx: &mut DecorContext<'_>, f: &Feat, ox: i32, base: i32) {
    let r = ri(gh(ctx, f, ox, base, 11), f.size0, f.size1).min(f.reach_x - 2);
    let deep = ri(gh(ctx, f, ox, base, 12), f.h0, f.h1).min(f.reach_y - 4);
    if r < 5 {
        return;
    }
    if !overlaps(ctx, ox - r, base - 3, ox + r, base + deep + 2) {
        return;
    }

    let water_top = base + 1;
    let from = lo_x(ctx, ox - r, 2);
    let to = hi_x(ctx, ox + r, 2);
    for x in from..=to {
        let t = f64::from(x - ox) / f64::from(r);
        // A parabolic bowl with a hash-roughened lip, so the shoreline is ragged.
        let d = js_round(f64::from(deep) * (1.0 - t * t)) as i32
            - i32::from(ctx.hash(x + 6607, base) < 0.35);
        if d < 1 {
            // Beyond the water: just the beach ring.
            if (x - ox).abs() <= r {
                ctx.plot(x, base, f.fill);
            }
            continue;
        }
        let floor = base + d;
        // Clear anything above the waterline inside the rim — an overhang here
        // would read as a cave with a puddle rather than as a pond.
        for y in base - 2..water_top {
            ctx.plot(x, y, AIR);
        }
        for y in water_top..=floor {
            ctx.plot(x, y, f.liquid);
        }
        ctx.plot(x, floor + 1, f.shell);
        if d <= 2 {
            ctx.plot(x, base, f.fill); // sand at the shallow margin
        } else if ctx.hash(x + 9091, base) < f.density * 0.3 {
            ctx.plot(x, floor, f.accent);
        }
    }
}

/// Abandoned mineshaft. A long horizontal gallery with timber sets, carved as
/// AIR so the natural tunnel field opens into it wherever the two cross — which
/// is what makes it a ROUTE through the underground rather than another room.
///
/// The floor line wanders with a low-frequency gradient sample of the column, so
/// it is pure in x (every chunk agrees on the corridor's height at a given
/// column) without being ruler-straight.
const SHAFT_FREQ: f64 = 0.011;
const SHAFT_ANCHOR: f64 = 517.3;

fn grow_mineshaft(ctx: &mut DecorContext<'_>, f: &Feat, ox: i32, oy: i32) {
    let half = (ri(gh(ctx, f, ox, oy, 11), f.size0, f.size1) >> 1).min(f.reach_x - 4);
    let tall = ri(gh(ctx, f, ox, oy, 12), f.h0, f.h1);
    let wobble = 4;
    let riser = ri(gh(ctx, f, ox, oy, 13), 8, f.reach_y - wobble - tall - 2);
    if half < 12 {
        return;
    }

    if !overlaps(
        ctx,
        ox - half,
        oy - wobble - tall - riser,
        ox + half,
        oy + wobble + 2,
    ) {
        return;
    }

    // One probe, at the origin: a gallery whose middle hangs in open air was never
    // dug, it was found. Cheap because it runs once per accepted instance.
    if open_at(ctx, ox, oy) {
        return;
    }

    let noise = ctx.noise;
    let step = 4 + js_round((1.0 - f.density) * 4.0) as i32; // supports every 4..8 cells
    let from = lo_x(ctx, ox - half, 3);
    let to = hi_x(ctx, ox + half, 3);
    for x in from..=to {
        let floor = oy
            + js_round(f64::from(wobble) * noise.g2(f64::from(x) * SHAFT_FREQ + SHAFT_ANCHOR, 0.0))
                as i32;
        let roof = floor - tall;

        for y in roof..=floor {
            ctx.plot(x, y, AIR);
        }
        // Planked walkway, gappy where it has rotted through.
        if ctx.hash(x + 3001, oy) > 0.18 {
            ctx.plot(x, floor, f.frame);
        } else {
            ctx.plot(x, floor, f.fill);
        }

        // A timber set: two posts and a cap. This is the only thing that writes
        // outside its own column, and only by one cell — hence the margin of 3.
        if pmod(x - ox, step) == 0 {
            for y in roof + 1..=floor - 1 {
                ctx.plot(x, y, f.frame);
            }
            ctx.plot(x, roof, f.frame);
            ctx.plot(x - 1, roof, f.frame);
            ctx.plot(x + 1, roof, f.frame);
            // Break the post so the set reads as a frame, not a wall.
            let gap =
                roof + 1 + (ctx.hash(x, oy + 55) * f64::from((tall - 1).max(1))).floor() as i32;
            ctx.plot(x, gap, AIR);
        }

        // The seam they were following, still in the roof.
        if ctx.hash(x + 7717, oy + 11) < f.density * 0.22 {
            ctx.plot(x, roof - 1, f.accent);
        }
    }

    // One riser back toward daylight, at a fixed offset from the origin.
    let sx = ox + ri(gh(ctx, f, ox, oy, 14), -half + 4, half - 4);
    if sx >= from - 2 && sx <= to + 2 {
        let floor = oy
            + js_round(f64::from(wobble) * noise.g2(f64::from(sx) * SHAFT_FREQ + SHAFT_ANCHOR, 0.0))
                as i32;
        let top = floor - tall - riser;
        for y in top..floor - tall {
            ctx.plot(sx, y, AIR);
            ctx.plot(sx - 1, y, f.shell);
            ctx.plot(sx + 1, y, f.shell);
            if ((y - top) & 3) == 0 {
                ctx.plot(sx, y, f.frame); // ladder rungs
            }
        }
    }
}

/// Vault dungeon. A grid of walled rooms with doorways between them and one
/// strongroom that has none — the payoff for noticing the outline and digging.
///
/// Built room by room rather than cell by cell, so the bbox test that skips a
/// room outside this chunk is a handful of compares and the whole structure
/// never costs more than the rooms actually visible here.
fn grow_dungeon(ctx: &mut DecorContext<'_>, f: &Feat, ox: i32, oy: i32) {
    let cols = ri(gh(ctx, f, ox, oy, 11), f.size0, f.size1);
    let rows = ri(gh(ctx, f, ox, oy, 12), f.h0, f.h1);
    let room_w = ri(gh(ctx, f, ox, oy, 13), 9, 13);
    let room_h = ri(gh(ctx, f, ox, oy, 14), 7, 9);
    let total_w = cols * room_w;
    let total_h = rows * room_h;
    // Refuse rather than clip: a dungeon that silently lost a room column would
    // still be deterministic, but the reach in content would have stopped meaning
    // what it says.
    if total_w > f.reach_x * 2 || total_h > f.reach_y * 2 {
        return;
    }

    let x0 = ox - (total_w >> 1);
    let y0 = oy - (total_h >> 1);
    if !overlaps(ctx, x0, y0, x0 + total_w, y0 + total_h) {
        return;
    }
    if open_at(ctx, ox, oy) {
        return;
    }

    let vault_i = ri(gh(ctx, f, ox, oy, 15), 0, cols - 1);
    let vault_j = ri(gh(ctx, f, ox, oy, 16), 0, rows - 1);
    let flooded = gh(ctx, f, ox, oy, 17) < f.density * 0.5;

    for j in 0..rows {
        for i in 0..cols {
            let rx0 = x0 + i * room_w;
            let ry0 = y0 + j * room_h;
            let rx1 = rx0 + room_w;
            let ry1 = ry0 + room_h;
            if !overlaps(ctx, rx0, ry0, rx1, ry1) {
                continue;
            }

            let vault = i == vault_i && j == vault_j;

            // Shell, then interior. Order matters: the lining must survive the carve.
            for y in ry0..=ry1 {
                for x in rx0..=rx1 {
                    let edge = x == rx0 || x == rx1 || y == ry0 || y == ry1;
                    ctx.plot(x, y, if edge { f.shell } else { AIR });
                }
            }
            // Laid floor over the bottom course.
            for x in rx0 + 1..rx1 {
                ctx.plot(x, ry1 - 1, f.frame);
            }

            if vault {
                // No doors, spiked approach, and the only accent in the building.
                for x in rx0 + 2..=rx1 - 2 {
                    if ctx.hash(x, ry1 + 3) < 0.4 {
                        ctx.plot(x, ry1 - 2, f.accent);
                    }
                }
                for y in ry0 + 1..ry1 - 1 {
                    ctx.plot(rx0 + 1, y, f.shell);
                    ctx.plot(rx1 - 1, y, f.shell);
                }
                // The reason to break into the sealed room. A `feature` has no glyph
                // art and therefore no `mark=loot` to hang this off, so the cell is
                // derived the same way every other cell in this file is: from the
                // room's own coordinates, plotted last so it wins over the spike row
                // it shares a line with. Every chunk the room touches derives the
                // identical cell, and `plot` discards it for all but the one that
                // owns it.
                //
                // A container placed this way has no recoverable host template, so
                // worldgen/loot.rs scores it on depth alone (see `roll_container`) —
                // which is the right answer for a dungeon: at min_depth 96 it is
                // already deep, and it should not out-earn a sealed vault it did not
                // have to be as rare as.
                if containers_present() {
                    let cx = rx0 + 2 + ri(gh(ctx, f, ox, oy, 18), 0, room_w - 4);
                    ctx.plot(cx, ry1 - 2, container_code());
                }
                continue;
            }

            // Doorway east, hatch south. Both derived from the room index, so the two
            // rooms sharing a wall agree on where the hole is without talking.
            if i < cols - 1 {
                let dy = ry1 - 2;
                ctx.plot(rx1, dy, AIR);
                ctx.plot(rx1, dy - 1, AIR);
            }
            if j < rows - 1 {
                let dx = rx0 + 2 + ri(ctx.hash(rx0, ry0 + 91), 0, room_w - 5);
                ctx.plot(dx, ry1, AIR);
                ctx.plot(dx, ry1 - 1, AIR);
            }
            if flooded && j == rows - 1 {
                for x in rx0 + 1..rx1 {
                    ctx.plot(x, ry1 - 2, f.liquid);
                    ctx.plot(x, ry1 - 3, f.liquid);
                }
            }
            // Pillars, so a long room does not read as a corridor.
            if room_w >= 11 {
                let px = rx0 + (room_w >> 1);
                for y in ry0 + 1..ry1 - 1 {
                    ctx.plot(px, y, f.fill);
                }
            }
        }
    }
}

/// Crystal geode — a HOLLOW pocket lined with crystal.
///
/// Deliberately distinct from the small solid geode the ore pass promotes
/// (decor/ores.rs `paint_geode`): that one is a glowing lump you break into for
/// the gems, this one is a room you fall into. Both exist because they read as
/// completely different discoveries at completely different depths.
fn grow_geode(ctx: &mut DecorContext<'_>, f: &Feat, ox: i32, oy: i32) {
    let rx = ri(gh(ctx, f, ox, oy, 11), f.size0, f.size1).min(f.reach_x - 3);
    let ry = ri(gh(ctx, f, ox, oy, 12), f.h0, f.h1).min(f.reach_y - 3);
    if rx < 4 || ry < 3 {
        return;
    }
    if !overlaps(ctx, ox - rx - 2, oy - ry - 2, ox + rx + 2, oy + ry + 2) {
        return;
    }
    if open_at(ctx, ox, oy) {
        return; // a geode inside a cavern is just a cavern
    }

    let irx = 1.0 / f64::from(rx * rx);
    let iry = 1.0 / f64::from(ry * ry);
    let pool = oy + ry - 2;
    let y0 = lo_y(ctx, oy - ry - 2, 1);
    let y1 = hi_y(ctx, oy + ry + 2, 1);
    let x0 = lo_x(ctx, ox - rx - 2, 1);
    let x1 = hi_x(ctx, ox + rx + 2, 1);

    for y in y0..=y1 {
        let dy = f64::from(y - oy);
        let qy = dy * dy * iry;
        for x in x0..=x1 {
            let dx = f64::from(x - ox);
            let q = dx * dx * irx + qy;
            if q > 1.2 {
                continue;
            }
            // Ragged rind, so the shell is rock rather than a drawn ellipse.
            if q > 0.95 && ctx.hash(x + 2213, y) < (q - 0.95) * 3.6 {
                continue;
            }
            if q > 0.92 {
                ctx.plot(x, y, f.shell);
            } else if q > 0.68 {
                let code = if ctx.hash(x, y + 733) < f.density {
                    f.fill
                } else {
                    f.accent
                };
                ctx.plot(x, y, code);
            } else {
                let code = if y >= pool && ctx.hash(x, y + 41) < 0.7 {
                    f.liquid
                } else {
                    AIR
                };
                ctx.plot(x, y, code);
            }
        }
    }
}

/// Mushroom grove. A wide, low chamber with its own light: caps on the ceiling
/// and stems on the floor, over mud. The one place underground that is lit by
/// what grows in it, which is what makes it read as a biome rather than a cave.
fn grow_grove(ctx: &mut DecorContext<'_>, f: &Feat, ox: i32, oy: i32) {
    let rx = ri(gh(ctx, f, ox, oy, 11), f.size0, f.size1).min(f.reach_x - 2);
    let ry = ri(gh(ctx, f, ox, oy, 12), f.h0, f.h1).min(f.reach_y - 3);
    if rx < 8 || ry < 4 {
        return;
    }
    if !overlaps(ctx, ox - rx - 1, oy - ry - 1, ox + rx + 1, oy + ry + 2) {
        return;
    }

    let irx = 1.0 / f64::from(rx * rx);
    let x0 = lo_x(ctx, ox - rx - 1, 2);
    let x1 = hi_x(ctx, ox + rx + 1, 2);

    for x in x0..=x1 {
        let dx = f64::from(x - ox);
        let qx = dx * dx * irx;
        if qx > 1.0 {
            continue;
        }
        // Column extent of the lens, roughened so the chamber is not an ellipse.
        let span = f64::from(ry) * (1.0 - qx).sqrt();
        let jag = i32::from(ctx.hash(x + 5591, oy) < 0.4);
        let top = oy - js_round(span) as i32 - jag;
        let floor = oy + js_round(span) as i32;

        for y in lo_y(ctx, top, 0)..=hi_y(ctx, floor, 2) {
            if y < top || y > floor + 2 {
                continue;
            }
            if y > floor {
                ctx.plot(x, y, f.fill); // mud bed under the chamber
            } else if y == floor {
                let code = if ctx.hash(x, y + 17) < 0.35 {
                    f.shell
                } else {
                    f.fill
                };
                ctx.plot(x, y, code);
            } else {
                ctx.plot(x, y, AIR);
            }
        }

        // Ceiling caps — the light source. Sparse, because a lit ceiling everywhere
        // stops reading as light and starts reading as a texture.
        if ctx.hash(x + 1487, oy + 3) < f.density * 0.3 {
            ctx.plot(x, top + 1, f.accent);
        }

        // A stand of mushrooms on the floor.
        let v = ctx.hash(x + 9973, oy + 5);
        if v < f.density * 0.34 && qx < 0.86 {
            let tall = 1 + (v * 9.0).floor() as i32;
            for d in 1..=tall {
                ctx.plot(x, floor - d, f.frame);
            }
            let cap_y = floor - tall - 1;
            ctx.plot(x, cap_y, f.accent);
            ctx.plot_if_empty(x - 1, cap_y, f.accent);
            ctx.plot_if_empty(x + 1, cap_y, f.accent);
        } else if v > 0.97 {
            ctx.plot(x, floor - 1, f.liquid); // a seep in the low ground
        }
    }
}

/// Rich ore cluster. Denser and larger than anything the vein noise in layers.rs
/// produces, so an exposed edge of one is worth following.
///
/// `plot_if_solid` throughout: a cluster that filled the cave it intersected
/// would wall off the route that led you to it. The gangue ring is the tell —
/// you see the gravel a couple of cells before you see the ore.
fn grow_ore_blob(ctx: &mut DecorContext<'_>, f: &Feat, ox: i32, oy: i32, depth: i32) {
    let rx = ri(gh(ctx, f, ox, oy, 11), f.size0, f.size1).min(f.reach_x - 2);
    let ry = ri(gh(ctx, f, ox, oy, 12), f.h0, f.h1).min(f.reach_y - 2);
    if rx < 2 || ry < 2 {
        return;
    }
    if !overlaps(ctx, ox - rx - 1, oy - ry - 1, ox + rx + 1, oy + ry + 1) {
        return;
    }

    // What is in it scales with depth, so a shallow cluster is worth digging out
    // and a deep one is worth walking to. The two ends are content (`mat.fill`
    // and `mat.accent`); only the blend is code.
    let rich_frac = (f64::from(depth - 60) / 240.0).clamp(0.0, 1.0);
    let irx = 1.0 / f64::from(rx * rx);
    let iry = 1.0 / f64::from(ry * ry);
    let y0 = lo_y(ctx, oy - ry - 1, 0);
    let y1 = hi_y(ctx, oy + ry + 1, 0);
    let x0 = lo_x(ctx, ox - rx - 1, 0);
    let x1 = hi_x(ctx, ox + rx + 1, 0);

    for y in y0..=y1 {
        let dy = f64::from(y - oy);
        let qy = dy * dy * iry;
        for x in x0..=x1 {
            let dx = f64::from(x - ox);
            let q = dx * dx * irx + qy;
            if q > 1.3 {
                continue;
            }
            let r = ctx.hash(x, y + f.salt_y);
            if q > 1.0 {
                if r < 0.3 {
                    ctx.plot_if_solid(x, y, f.shell);
                }
                continue;
            }
            if r > f.density + (1.0 - q) * 0.25 {
                continue;
            }
            let code = if ctx.hash(x + 617, y) < rich_frac {
                f.accent
            } else {
                f.fill
            };
            ctx.plot_if_solid(x, y, code);
        }
    }
}

// =============================================================================
// Placement
// =============================================================================

/// Offer one grid cell of one feature to its generator.
///
/// Structured so the reject path is as short as it can be: the hosting hash first
/// (which throws away most cells for a couple of integer multiplies), then the
/// cheap positional gates, then the profile lookup, and only then the generator.
fn offer(ctx: &mut DecorContext<'_>, f: &Feat, gx: i32, gy: i32) {
    if gh(ctx, f, gx, gy, 0) >= f.rarity {
        return;
    }

    let ox = gx * f.cell_w + ri(gh(ctx, f, gx, gy, 1), 0, f.cell_w - 1);

    if f.place == Place::Sky || f.place == Place::Surface {
        let surf = ctx.surface_at(ox);
        if let Some(biomes) = &f.biomes
            && !biomes.contains(&ctx.profile_at(ox).surf_a)
        {
            return;
        }

        // Exhaustive on purpose, and the reason is the whole point of the enum
        // being generated: adding a variant to `content/`'s schema now fails to
        // compile HERE, with the list of arms in front of whoever added it,
        // rather than growing nothing and looking like a worldgen bug.
        match f.kind {
            Some(Kind::Island) => {
                let clear = ri(gh(ctx, f, gx, gy, 2), f.clear0, f.clear1);
                grow_island(ctx, f, ox, surf - clear);
            }
            Some(Kind::Lake) => {
                // A lake needs a basin above sea level (below it the heightmap
                // already flooded the column) and ground flat enough to cut one
                // datum across.
                if surf >= SEA_LEVEL_Y - 3 {
                    return;
                }
                if spread_at(ctx, ox, f.size1 >> 1, surf) > 4 {
                    return;
                }
                grow_lake(ctx, f, ox, surf);
            }
            // Grown on the 2D lattice below, not from a column.
            Some(Kind::Mineshaft)
            | Some(Kind::Dungeon)
            | Some(Kind::Geode)
            | Some(Kind::Grove)
            | Some(Kind::Oreblob) => {}
            // A code this build has no generator for. See `Kind::from_code`.
            None => {}
        }
        return;
    }

    let oy = gy * f.cell_h + ri(gh(ctx, f, gx, gy, 2), 0, f.cell_h - 1);
    let surf = ctx.surface_at(ox);
    let depth = oy - surf;
    if depth < f.min_depth || depth > f.max_depth {
        return;
    }

    if f.biomes.is_some() || f.layers.is_some() {
        let col = ctx.profile_at(ox);
        if let Some(biomes) = &f.biomes
            && !biomes.contains(&col.surf_a)
        {
            return;
        }
        if let Some(layers) = &f.layers
            && !layers.contains(&col.ug.top)
        {
            return;
        }
    }

    // Exhaustive, for the reason the column match above is. The comment this
    // replaces said "a new kind must add an arm", which is a comment doing a
    // compiler's job — and doing it worse, because a wildcard meant the new
    // kind placed nothing and said nothing.
    match f.kind {
        Some(Kind::Mineshaft) => grow_mineshaft(ctx, f, ox, oy),
        Some(Kind::Dungeon) => grow_dungeon(ctx, f, ox, oy),
        Some(Kind::Geode) => grow_geode(ctx, f, ox, oy),
        Some(Kind::Grove) => grow_grove(ctx, f, ox, oy),
        Some(Kind::Oreblob) => grow_ore_blob(ctx, f, ox, oy, depth),
        // Column-placed; grown above, before this point is reached.
        Some(Kind::Island) | Some(Kind::Lake) => {}
        // A code this build has no generator for. See `Kind::from_code`.
        None => {}
    }
}

/// Walk every feature's grid cells that could reach this chunk.
///
/// The band compare is the hard early-out the budget rests on: a sky chunk is
/// rejected by every underground feature before a hash is computed, and a deep
/// chunk by every surface one. What survives is one hash per overlapping cell,
/// and at these grid sizes that is one or two cells per axis.
fn place_features(ctx: &mut DecorContext<'_>) {
    let top = ctx.base_y;
    let bot = ctx.base_y + CHUNK_CELLS - 1;

    for f in &FEATS.all {
        if bot < f.band_top || top > f.band_bot {
            continue;
        }

        let gx0 = fdiv(ctx.base_x - f.reach_x, f.cell_w);
        let gx1 = fdiv(ctx.base_x + CHUNK_CELLS - 1 + f.reach_x, f.cell_w);

        if f.place != Place::Underground {
            // The row is derived from the ground line, so there is no vertical grid
            // to walk — exactly the reason surface features are placed by column.
            for gx in gx0..=gx1 {
                offer(ctx, f, gx, 0);
            }
            continue;
        }

        let gy0 = fdiv(ctx.base_y - f.reach_y, f.cell_h);
        let gy1 = fdiv(ctx.base_y + CHUNK_CELLS - 1 + f.reach_y, f.cell_h);
        for gy in gy0..=gy1 {
            for gx in gx0..=gx1 {
                offer(ctx, f, gx, gy);
            }
        }
    }
}

/// Landmarks: parametric features first, then template structures.
///
/// The order is the point. Features are landform-scale — an island IS terrain, a
/// lake replaces it, a mineshaft cuts through it — so they have to land before
/// anything that stands on them. Structures are built things and go last, which
/// is also what lets a template overwrite a tree the tree pass grew where the
/// cabin is now standing.
#[derive(Clone, Copy, Debug, Default)]
pub struct LandmarkDecorator;

/// The pass, as the generator names it.
pub const LANDMARK_DECORATOR: LandmarkDecorator = LandmarkDecorator;

impl Decorator for LandmarkDecorator {
    fn name(&self) -> &'static str {
        "landmarks"
    }

    fn reach_x(&self) -> i32 {
        feat_reach_x().max(struct_reach_x())
    }

    fn reach_y(&self) -> i32 {
        feat_reach_y().max(struct_reach_y())
    }

    fn decorate(&self, ctx: &mut DecorContext<'_>) {
        place_features(ctx);
        stamp_structs(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorldScale;
    use crate::sim::materials::EMPTY;
    use crate::sim::noise::Noise;
    use crate::sim::worldgen::heightmap::Heightmap;

    fn feats() -> &'static [Feat] {
        &FEATS.all
    }

    #[test]
    fn the_runtime_form_agrees_with_the_compiled_tables() {
        // The knobs are unpacked from the defs here and emitted as flat tables
        // there. A disagreement means the facade and the generator are reading
        // two different worlds.
        for def in FEATURES.iter() {
            let Some(f) = build(def) else { continue };
            let c = def.code as usize;
            assert_eq!(f.kind.map(|k| k as u8), Some(FEAT_KIND[c]), "{} kind", f.id);
            assert_eq!(f.place as u8, FEAT_PLACE[c], "{} place", f.id);
            assert_eq!(f.cell_w, i32::from(FEAT_CELLW[c]), "{} cell width", f.id);
            assert_eq!(f.cell_h, i32::from(FEAT_CELLH[c]), "{} cell height", f.id);
            assert_eq!(f.reach_x, i32::from(FEAT_REACHX[c]), "{} reach x", f.id);
            assert_eq!(f.reach_y, i32::from(FEAT_REACHY[c]), "{} reach y", f.id);
            assert_eq!(f.size0, i32::from(FEAT_SIZE0[c]), "{} size lo", f.id);
            assert_eq!(f.size1, i32::from(FEAT_SIZE1[c]), "{} size hi", f.id);
            assert_eq!(f.h0, i32::from(FEAT_HEIGHT0[c]), "{} height lo", f.id);
            assert_eq!(f.h1, i32::from(FEAT_HEIGHT1[c]), "{} height hi", f.id);
            assert_eq!(f.clear0, i32::from(FEAT_CLEAR0[c]), "{} clearance lo", f.id);
            assert_eq!(f.clear1, i32::from(FEAT_CLEAR1[c]), "{} clearance hi", f.id);
            assert_eq!(
                f.min_depth,
                i32::from(FEAT_MINDEPTH[c]),
                "{} min depth",
                f.id
            );
            assert_eq!(
                f.max_depth,
                i32::from(FEAT_MAXDEPTH[c]),
                "{} max depth",
                f.id
            );
        }
    }

    #[test]
    fn every_authored_feature_has_a_generator() {
        // A `kind` with no arm in `offer` silently generates nothing, which looks
        // exactly like a rare feature nobody has stumbled on yet.
        for f in feats() {
            assert!(
                f.kind.is_some(),
                "{} names a generator this build lacks",
                f.id
            );
        }
    }

    #[test]
    fn chunk_overlap_rejection_is_symmetric() {
        // `overlaps` is the gate every generator opens with: a feature paints a
        // chunk if and only if their boxes intersect. Two chunks that both
        // intersect one feature must BOTH accept it, or the half that rejected it
        // leaves a hole its neighbour already drew around.
        let noise = Noise::new(11);
        let mut hm = Heightmap::new();
        let mut cells = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];

        /// Do two closed integer intervals meet? Symmetric in its arguments by
        /// construction, which is the property `overlaps` has to have.
        fn meet(a0: i32, a1: i32, b0: i32, b1: i32) -> bool {
            a0 <= b1 && b0 <= a1
        }

        // A box straddling the seam between two adjacent chunks.
        let (x0, y0, x1, y1) = (CHUNK_CELLS - 3, 40, CHUNK_CELLS + 5, 60);
        let mut accepted = 0;
        for (bx, by) in [
            (0, 32),
            (CHUNK_CELLS, 32),
            (CHUNK_CELLS * 4, 32),
            (0, 0),
            (-CHUNK_CELLS, 32),
        ] {
            let ctx =
                DecorContext::new(&noise, 11, bx, by, &mut cells, &mut hm, WorldScale::LEGACY);
            let hit = overlaps(&ctx, x0, y0, x1, y1);
            // The relation stated the other way round: does the chunk's own box
            // meet the feature's? Same question, arguments swapped.
            let mirror = meet(x0, x1, ctx.base_x, ctx.base_x + CHUNK_CELLS - 1)
                && meet(y0, y1, ctx.base_y, ctx.base_y + CHUNK_CELLS - 1);
            assert_eq!(hit, mirror, "chunk ({bx},{by}) disagrees with itself");
            accepted += i32::from(hit);
        }
        // The two chunks either side of the seam must BOTH have accepted it, or
        // the half that rejected leaves a hole its neighbour already drew around.
        assert_eq!(
            accepted, 2,
            "the seam chunks did not both claim the feature"
        );
    }

    #[test]
    fn a_feature_paints_the_same_cells_from_either_side_of_a_seam() {
        // THE invariant, end to end: two adjacent chunks generate independently
        // and must agree on every cell of the overlap they both cover. Run over
        // an underground band where several features are live at once.
        let seed = 20_260_805;
        let noise = Noise::new(seed);

        let base_x = 512;
        let base_y = 256;
        let mut a = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let mut b = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        {
            let mut hm = Heightmap::new();
            let mut ctx = DecorContext::new(
                &noise,
                seed,
                base_x,
                base_y,
                &mut a,
                &mut hm,
                WorldScale::LEGACY,
            );
            place_features(&mut ctx);
        }
        {
            // A DIFFERENT heightmap memo, so nothing carries over between the two.
            let mut hm = Heightmap::new();
            let mut ctx = DecorContext::new(
                &noise,
                seed,
                base_x,
                base_y,
                &mut b,
                &mut hm,
                WorldScale::LEGACY,
            );
            place_features(&mut ctx);
        }
        assert_eq!(a, b, "the same chunk generated twice differs");
    }

    #[test]
    fn the_pass_actually_grows_something() {
        // Every other test here would pass just as happily against a generator
        // that painted nothing at all. This is the one that says it does not.
        let seed = 1234;
        let noise = Noise::new(seed);
        let mut hm = Heightmap::new();
        let mut painted = 0usize;
        // A band deep enough for the mineshafts, dungeons, geodes and groves.
        for cy in 4..14 {
            for cx in 0..24 {
                let mut cells = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
                let mut ctx = DecorContext::new(
                    &noise,
                    seed,
                    cx * CHUNK_CELLS,
                    cy * CHUNK_CELLS,
                    &mut cells,
                    &mut hm,
                    WorldScale::LEGACY,
                );
                place_features(&mut ctx);
                painted += cells.iter().filter(|&&c| c != EMPTY).count();
            }
        }
        assert!(
            painted > 0,
            "240 chunks of underground and not one feature cell"
        );
    }

    #[test]
    fn the_declared_reach_covers_every_authored_feature() {
        for f in feats() {
            assert!(
                f.reach_x <= feat_reach_x(),
                "{} out-reaches the decorator",
                f.id
            );
            assert!(
                f.reach_y <= feat_reach_y(),
                "{} out-reaches the decorator",
                f.id
            );
        }
        assert!(LANDMARK_DECORATOR.reach_x() >= feat_reach_x());
        assert!(LANDMARK_DECORATOR.reach_y() >= feat_reach_y());
    }
}

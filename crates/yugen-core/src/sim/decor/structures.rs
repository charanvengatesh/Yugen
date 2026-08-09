//! Procedural structures — the built things. Ruined towers, henges, huts, wells
//! and stepped temples on the surface; tombs, pillared halls, cisterns, mines and
//! forges underground.
//!
//! THE RULE (see [`super`]): a structure is authored from a POSITIONAL ORIGIN and
//! every chunk it overlaps recomputes the WHOLE structure independently, painting
//! only its own share. Structures are the largest decorations in the world — a
//! tower is 28 cells tall and a temple 29 wide, so a single one routinely spans
//! six chunks that are generated in whatever order the player walks into them.
//! Nothing here may read a neighbour, keep state, or call a stateful PRNG: every
//! decision is drawn from [`DecorContext::hash`] on the origin, so all six chunks
//! derive the identical building and their shares line up exactly.
//!
//! Consequently EVERY generator below is a pure function of (origin, seed). The
//! only thing that varies per chunk is which `plot` calls land inside it — and
//! `plot` silently discards the rest. Generators do bail out early when their
//! bounding box misses the chunk, but the bbox itself is derived from the origin
//! hash alone, so the bail-out is an optimisation and never a decision.

use super::{DecorContext, Decorator, Lattice, origin_cells, origin_columns};
use crate::config::world::CHUNK_CELLS;
use crate::config::worldgen::{SURFACE_AMPLITUDE, SURFACE_ANCHOR_Y};
use crate::sim::biomes::Biome;
use crate::sim::materials::{CellId, block};

// --- Materials ---------------------------------------------------------------
// In the TypeScript these were resolved by string id through `codeOf`, never
// hardcoded, because codes were an implementation detail of materials.ts that
// other agents renumbered. The registry is compiled content here, so every id
// resolves at COMPILE time through `block::*` and a renumbering cannot go
// unnoticed. The local names survive because the palette tables read better in
// the vocabulary of masonry than in the vocabulary of the block registry.
const AIR: CellId = block::EMPTY;
const STONE: CellId = block::STONE;
const SANDSTONE: CellId = block::SANDSTONE;
const BASALT: CellId = block::BASALT;
const OBSIDIAN: CellId = block::OBSIDIAN;
const WOOD: CellId = block::WOOD;
const GLASS: CellId = block::GLASS;
const CRYSTAL: CellId = block::CRYSTAL;
const ICE: CellId = block::ICE;
const PACKED_ICE: CellId = block::PACKED_ICE;
const SNOW: CellId = block::SNOW;
const CLAY: CellId = block::CLAY;
const GRAVEL: CellId = block::GRAVEL;
const MUD: CellId = block::MUD;
const SAND: CellId = block::SAND;
const MOSS: CellId = block::MOSS;
const VINE: CellId = block::VINE;
const MUSHROOM_CAP: CellId = block::MUSHROOM_CAP;
const LAVA: CellId = block::LAVA;
const WATER: CellId = block::WATER;
const ACID: CellId = block::ACID;
const COAL_ORE: CellId = block::COAL_ORE;
const IRON_ORE: CellId = block::IRON_ORE;
const GOLD_ORE: CellId = block::GOLD_ORE;
const GEM_ORE: CellId = block::GEM_ORE;
const SPIKE: CellId = block::SPIKE;
const CONVEYOR: CellId = block::CONVEYOR_RIGHT;
const ASH: CellId = block::ASH;

// --- Lattice tuning ----------------------------------------------------------
// Landmarks, so: sparse. One surface candidate every SURF_STRIDE columns, of
// which SURF_DENSITY survive the rarity gate and some of those are then rejected
// for standing on broken ground — roughly one built thing per 350 columns walked.
const SURF_STRIDE: i32 = 96;
const SURF_PHASE: i32 = 41;
const SURF_DENSITY: f64 = 0.46;

// Underground candidates sit on a 2D lattice; the depth gate throws away every
// one that is not properly buried, so the effective density is lower again.
const SUB_STRIDE_X: i32 = 72;
const SUB_STRIDE_Y: i32 = 56;
const SUB_PHASE_X: i32 = 17;
const SUB_PHASE_Y: i32 = 23;
const SUB_DENSITY: f64 = 0.22;
/// Cells below the local surface before a chamber may appear.
const SUB_MIN_DEPTH: i32 = 44;

// --- Reach -------------------------------------------------------------------
// Honest maxima, enforced by the caps inside each generator.
//   horizontal: underground chamber, halfW 15 + a 5-cell entrance stub = 20.
//               (widest surface piece is the temple at halfW 14.)
//   vertical:   ruined tower, 26 tall + 2 rows of battlements = 28 above origin.
//               (deepest is the well at 24 below; the mine shaft rises 19.)
const SURF_REACH_X: i32 = 15;
const SURF_UP: i32 = 28;
const SURF_DOWN: i32 = 24;
const SUB_REACH_X: i32 = 20;
const SUB_REACH_Y: i32 = 20;

const REACH_X: i32 = if SURF_REACH_X > SUB_REACH_X {
    SURF_REACH_X
} else {
    SUB_REACH_X
};
const REACH_Y: i32 = {
    let a = if SURF_UP > SURF_DOWN {
        SURF_UP
    } else {
        SURF_DOWN
    };
    if a > SUB_REACH_Y { a } else { SUB_REACH_Y }
};

/// `Math.floor` for a band edge, at compile time. `f64::floor` is not const, and
/// an `as` cast truncates toward zero, which is the wrong rule below the origin.
const fn floor_i(v: f64) -> i32 {
    let t = v as i32;
    if v < 0.0 && (t as f64) != v { t - 1 } else { t }
}

/// `Math.ceil` for a band edge, at compile time. See [`floor_i`].
const fn ceil_i(v: f64) -> i32 {
    let t = v as i32;
    if v > 0.0 && (t as f64) != v { t + 1 } else { t }
}

// Rows the ground line can possibly occupy, with slack: the largest amp_scale in
// BIOMES is 1.45 and the largest |height_offset| is 6. Chunks outside the band
// (plus a structure's vertical reach) cannot contain a surface structure, which
// is what stops every deep chunk in the world paying for the surface scan.
const SURF_SPAN: f64 = SURFACE_AMPLITUDE as f64 * 1.6 + 8.0;
const BAND_TOP: i32 = floor_i(SURFACE_ANCHOR_Y as f64 - SURF_SPAN) - SURF_UP;
const BAND_BOT: i32 = ceil_i(SURFACE_ANCHOR_Y as f64 + SURF_SPAN) + SURF_DOWN;
/// No chamber can reach above this row, whatever the terrain does.
const SUB_BAND_TOP: i32 =
    floor_i(SURFACE_ANCHOR_Y as f64 - SURF_SPAN) + SUB_MIN_DEPTH - SUB_REACH_Y;

// --- Deterministic helpers ---------------------------------------------------

/// Salted positional hash. `salt` shifts the sample point on both axes by
/// coprime-ish strides, so the ~30 independent decisions one structure makes are
/// decorrelated while every one of them stays a pure function of the origin.
#[inline]
fn h(ctx: &DecorContext<'_>, x: i32, y: i32, salt: i32) -> f64 {
    ctx.hash(x + salt * 7919, y - salt * 104_729)
}

/// Hash value -> integer in [lo, hi].
///
/// The clamp chain is written out rather than deferred to `i32::clamp` because
/// call sites pass an EMPTY range: a chamber at its minimum half-height asks for
/// `ri(_, 2, 1)`. JavaScript's chain answered `hi` there; `clamp` panics.
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

/// Does this axis-aligned box touch the chunk being generated?
#[inline]
fn overlaps(ctx: &DecorContext<'_>, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    x1 >= ctx.base_x
        && x0 < ctx.base_x + CHUNK_CELLS
        && y1 >= ctx.base_y
        && y0 < ctx.base_y + CHUNK_CELLS
}

fn fill(ctx: &mut DecorContext<'_>, x0: i32, y0: i32, x1: i32, y1: i32, code: CellId) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            ctx.plot(x, y, code);
        }
    }
}

fn hline(ctx: &mut DecorContext<'_>, x0: i32, x1: i32, y: i32, code: CellId) {
    for x in x0..=x1 {
        ctx.plot(x, y, code);
    }
}

fn vline(ctx: &mut DecorContext<'_>, x: i32, y0: i32, y1: i32, code: CellId) {
    for y in y0..=y1 {
        ctx.plot(x, y, code);
    }
}

// --- Palettes ----------------------------------------------------------------

/// What a building in this climate is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Palette {
    /// Load-bearing masonry.
    wall: CellId,
    /// Laid floors, slabs, walkways.
    floor: CellId,
    /// Accent: lintels, altars, sarcophagi, linings.
    trim: CellId,
    /// Debris and weathered blocks.
    rubble: CellId,
    /// Overgrowth material, or `None` for climates where nothing grows. (`-1` in
    /// the TypeScript.)
    over: Option<CellId>,
    /// What pools inside a flooded one.
    fluid: CellId,
}

const fn pal(
    wall: CellId,
    floor: CellId,
    trim: CellId,
    rubble: CellId,
    over: Option<CellId>,
    fluid: CellId,
) -> Palette {
    Palette {
        wall,
        floor,
        trim,
        rubble,
        over,
        fluid,
    }
}

/// Masonry palette for a surface biome. `r` picks between regional variants.
fn surface_palette(biome: Biome, r: f64) -> Palette {
    match biome {
        Biome::Desert | Biome::Savanna => {
            if r < 0.25 {
                pal(CLAY, SANDSTONE, SANDSTONE, SAND, None, WATER)
            } else {
                pal(SANDSTONE, SANDSTONE, CLAY, SAND, None, WATER)
            }
        }
        Biome::Tundra => pal(STONE, PACKED_ICE, ICE, SNOW, None, WATER),
        Biome::Glacier => pal(PACKED_ICE, ICE, GLASS, SNOW, None, WATER),
        Biome::Volcanic => {
            if r < 0.3 {
                pal(OBSIDIAN, BASALT, OBSIDIAN, ASH, None, LAVA)
            } else {
                pal(BASALT, BASALT, OBSIDIAN, ASH, None, LAVA)
            }
        }
        Biome::Jungle => pal(STONE, MUD, WOOD, GRAVEL, Some(MOSS), WATER),
        Biome::Swamp => pal(STONE, WOOD, WOOD, MUD, Some(MOSS), ACID),
        // Plains and anything a future biome adds: honest grey stone, sometimes
        // timber-framed, occasionally mossed over where it has stood a while.
        _ => {
            if r < 0.3 {
                pal(
                    STONE,
                    WOOD,
                    WOOD,
                    GRAVEL,
                    if r < 0.12 { Some(MOSS) } else { None },
                    WATER,
                )
            } else {
                pal(
                    STONE,
                    STONE,
                    WOOD,
                    GRAVEL,
                    if r > 0.85 { Some(MOSS) } else { None },
                    WATER,
                )
            }
        }
    }
}

/// Masonry palette for a chamber, flavoured by what is overhead and by depth.
fn sub_palette(biome: Biome, depth: i32, r: f64) -> Palette {
    if depth > 200 || biome == Biome::Volcanic {
        return pal(BASALT, OBSIDIAN, OBSIDIAN, ASH, None, LAVA);
    }
    if biome == Biome::Glacier || biome == Biome::Tundra {
        return pal(PACKED_ICE, ICE, GLASS, SNOW, None, WATER);
    }
    if biome == Biome::Swamp {
        return pal(STONE, MUD, WOOD, MUD, Some(MOSS), ACID);
    }
    if biome == Biome::Jungle {
        return pal(STONE, CLAY, WOOD, GRAVEL, Some(MOSS), WATER);
    }
    if (biome == Biome::Desert || biome == Biome::Savanna) && depth < 130 {
        return pal(SANDSTONE, SANDSTONE, CLAY, SAND, None, WATER);
    }
    if r < 0.35 {
        pal(
            STONE,
            GRAVEL,
            WOOD,
            GRAVEL,
            if r < 0.15 { Some(MOSS) } else { None },
            WATER,
        )
    } else {
        pal(STONE, CLAY, WOOD, GRAVEL, None, WATER)
    }
}

/// What a cache is worth at this depth. Deeper is better; a little luck helps.
fn prize_for(depth: i32, r: f64) -> CellId {
    if r > 0.93 {
        return GEM_ORE;
    }
    if depth > 190 {
        return if r > 0.4 { GEM_ORE } else { GOLD_ORE };
    }
    if depth > 130 {
        return if r > 0.5 { GOLD_ORE } else { IRON_ORE };
    }
    if depth > 70 {
        return if r > 0.6 { IRON_ORE } else { COAL_ORE };
    }
    if r > 0.8 { IRON_ORE } else { COAL_ORE }
}

// --- Weathered masonry -------------------------------------------------------

/// One block of a wall, weathered. `wear` in [0,1] is the chance the block has
/// fallen out entirely; a few survivors are cracked to rubble or taken by moss.
/// Positional, so the same wall crumbles identically in every chunk that draws it.
fn masonry(ctx: &mut DecorContext<'_>, x: i32, y: i32, p: Palette, wear: f64, salt: i32) {
    let r = h(ctx, x, y, salt);
    if r < wear {
        return; // fallen away — leaves whatever the carve left behind
    }
    if let Some(over) = p.over
        && r > 0.93
    {
        ctx.plot(x, y, over);
    } else if r > 0.88 {
        ctx.plot(x, y, p.rubble);
    } else {
        ctx.plot(x, y, p.wall);
    }
}

/// A weathered horizontal run.
fn masonry_row(
    ctx: &mut DecorContext<'_>,
    x0: i32,
    x1: i32,
    y: i32,
    p: Palette,
    wear: f64,
    salt: i32,
) {
    for x in x0..=x1 {
        masonry(ctx, x, y, p, wear, salt);
    }
}

/// Bridge the gap between a structure's floor line and the ground under it. Every
/// footprint column gets its own plinth down to its own terrain height, which is
/// what stops a building floating over a dip. `surface_at` is pure, so the plinth
/// a neighbouring chunk computes for the same column is identical.
fn plinth(ctx: &mut DecorContext<'_>, x0: i32, x1: i32, base: i32, p: Palette, max_drop: i32) {
    for x in x0..=x1 {
        let s = ctx.surface_at(x);
        if s <= base {
            continue;
        }
        let to = if s > base + max_drop {
            base + max_drop
        } else {
            s
        };
        for y in base..to {
            ctx.plot(x, y, p.wall);
        }
    }
}

/// Hang creepers off an exposed ledge. Only into open air, never into rock.
fn drape(ctx: &mut DecorContext<'_>, x0: i32, x1: i32, y: i32, p: Palette, salt: i32) {
    if p.over.is_none() {
        return;
    }
    for x in x0..=x1 {
        let r = h(ctx, x, y, salt);
        if r > 0.42 {
            continue;
        }
        let len = 1 + (r * 9.0).floor() as i32;
        for d in 0..len {
            ctx.plot_if_empty(x, y + d, VINE);
        }
    }
}

// =============================================================================
// Surface structures
// =============================================================================

/// Ruined tower. The tall landmark — up to 28 cells of it, so it always spans at
/// least two chunk rows and usually two chunk columns as well.
///
/// Varies on: width, height, how much of the top has come down, wear of the
/// surviving stone, floor spacing, which side the door is on, whether anything
/// was left inside.
fn build_tower(ctx: &mut DecorContext<'_>, ox: i32, base: i32, p: Palette) {
    let half_w = ri(h(ctx, ox, 0, 3), 2, 5);
    let height = ri(h(ctx, ox, 0, 4), 12, 26);
    let collapse = if h(ctx, ox, 0, 5) < 0.62 {
        ri(h(ctx, ox, 0, 6), 1, 11)
    } else {
        0
    };
    let wear = 0.03 + h(ctx, ox, 0, 7) * 0.16;
    let top_y = base - height + collapse;

    let x0 = ox - half_w;
    let x1 = ox + half_w;
    if !overlaps(ctx, x0 - 1, top_y - 3, x1 + 1, base + 8) {
        return;
    }

    // Clear the whole shaft first: on a slope the uphill side would otherwise be
    // solid ground, and the tower would read as half-buried.
    fill(ctx, x0, top_y - 2, x1, base - 1, AIR);

    // Shell. Wear climbs towards the break line so the ruin frays at the top
    // instead of stopping at a ruled edge.
    for y in (top_y..=base - 1).rev() {
        let t = f64::from(y - top_y) / 6.0;
        let w = if t < 1.0 {
            wear + (1.0 - t) * 0.45
        } else {
            wear
        };
        masonry(ctx, x0, y, p, w, 8);
        masonry(ctx, x1, y, p, w, 8);
    }

    if collapse == 0 {
        // Intact: cap it and crenellate.
        masonry_row(ctx, x0, x1, top_y - 1, p, wear, 9);
        let mut x = x0;
        while x <= x1 {
            ctx.plot(x, top_y - 2, p.wall);
            x += 2;
        }
    }

    // Floors, each with a hatch to climb through.
    let gap = ri(h(ctx, ox, 0, 10), 4, 6);
    let mut fy = base - gap;
    while fy > top_y + 1 {
        let hole = x0 + 1 + ri(h(ctx, ox, fy, 11), 0, half_w * 2 - 2);
        for x in x0 + 1..=x1 - 1 {
            if x != hole {
                ctx.plot(x, fy, p.floor);
            }
        }
        // An arrow slit, alternating sides up the tower.
        let side = if h(ctx, ox, fy, 12) < 0.5 { x0 } else { x1 };
        ctx.plot(side, fy - 2, AIR);
        ctx.plot(side, fy - 3, AIR);
        fy -= gap;
    }

    // Door, and the ground floor laid with slabs.
    let door_side = if h(ctx, ox, 0, 13) < 0.5 { x0 } else { x1 };
    for y in (base - 3..=base - 1).rev() {
        ctx.plot(door_side, y, AIR);
    }
    hline(ctx, x0 + 1, x1 - 1, base, p.floor);

    // Something worth the climb, now and then.
    if h(ctx, ox, 0, 14) < 0.3 && half_w >= 3 {
        let prize = prize_for(20, h(ctx, ox, 0, 15));
        fill(ctx, ox - 1, base - 2, ox, base - 1, prize);
    }

    plinth(ctx, x0, x1, base, p, 8);
    drape(ctx, x0 - 1, x1 + 1, top_y, p, 16);
}

/// Abandoned hut. The small, common one — a shelter that reads as somebody having
/// lived here. Varies on width, height, roof pitch, how far it has collapsed and
/// whether anything is still inside.
fn build_hut(ctx: &mut DecorContext<'_>, ox: i32, base: i32, p: Palette) {
    let half_w = ri(h(ctx, ox, 0, 3), 4, 6);
    let height = ri(h(ctx, ox, 0, 4), 4, 6);
    let ruined = h(ctx, ox, 0, 5) < 0.35;
    let wear = if ruined {
        0.16 + h(ctx, ox, 0, 6) * 0.3
    } else {
        0.02 + h(ctx, ox, 0, 6) * 0.07
    };
    let pitched = h(ctx, ox, 0, 7) < 0.55;
    let peak = if pitched { 2 } else { 0 };

    let x0 = ox - half_w;
    let x1 = ox + half_w;
    let top_y = base - height;
    if !overlaps(ctx, x0 - 1, top_y - peak - 1, x1 + 1, base + 8) {
        return;
    }

    fill(ctx, x0, top_y - peak, x1, base - 1, AIR);

    for y in (top_y..=base - 1).rev() {
        masonry(ctx, x0, y, p, wear, 8);
        masonry(ctx, x1, y, p, wear, 8);
    }

    // Roof: a flat lintel run, or two slopes meeting over the middle.
    let roof_wear = if ruined { wear + 0.3 } else { wear };
    if pitched {
        for x in x0..=x1 {
            let d = if x < ox { x - x0 } else { x1 - x };
            let step = if d > peak { peak } else { d };
            masonry(ctx, x, top_y - step, p, roof_wear, 17);
            if step > 0 {
                masonry(ctx, x, top_y - step + 1, p, roof_wear + 0.25, 18);
            }
        }
    } else {
        masonry_row(ctx, x0, x1, top_y, p, roof_wear, 17);
    }

    // Laid floor, doorway, one window.
    hline(ctx, x0 + 1, x1 - 1, base, p.floor);
    let door_side = if h(ctx, ox, 0, 13) < 0.5 { x0 } else { x1 };
    ctx.plot(door_side, base - 1, AIR);
    ctx.plot(door_side, base - 2, AIR);
    let win = ox + ri(h(ctx, ox, 0, 19), -half_w + 2, half_w - 2);
    if !ruined {
        ctx.plot(win, top_y, p.trim);
    }

    // Furniture: a bench, a hearth, sometimes a stash under the boards.
    let stuff = h(ctx, ox, 0, 20);
    if stuff < 0.4 {
        fill(ctx, ox + 1, base - 1, ox + 2, base - 1, p.trim);
    } else if stuff < 0.6 {
        fill(ctx, ox - 1, base - 1, ox, base - 1, ASH);
    }
    if h(ctx, ox, 0, 21) < 0.22 {
        let prize = prize_for(10, h(ctx, ox, 0, 22));
        fill(ctx, ox - 1, base + 1, ox, base + 2, prize);
    }

    plinth(ctx, x0, x1, base, p, 8);
    drape(ctx, x0, x1, top_y - peak, p, 23);
}

/// What stands in the middle of a henge. (`0 obelisk, 1 altar, 2 basin` in the
/// TypeScript.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Centre {
    Obelisk,
    Altar,
    Basin,
}

/// Standing stones. Seen from the side a henge is a row of monoliths, some
/// toppled, some capped with lintels; the middle carries an obelisk, an altar, or
/// — where the ground is volcanic — a lava basin.
///
/// Every stone stands on ITS OWN column's surface, not on the origin's, so the
/// row follows the ground instead of hovering over it.
fn build_henge(ctx: &mut DecorContext<'_>, ox: i32, base: i32, p: Palette) {
    let n = ri(h(ctx, ox, 0, 3), 3, 5);
    let spacing = ri(h(ctx, ox, 0, 4), 4, 6);
    let span = (n - 1) * spacing;
    let start_x = ox - (span >> 1);
    let lintels = h(ctx, ox, 0, 5) < 0.45;
    let centre = match ri(h(ctx, ox, 0, 6), 0, 2) {
        0 => Centre::Obelisk,
        1 => Centre::Altar,
        _ => Centre::Basin,
    };

    if !overlaps(ctx, start_x - 2, base - 20, start_x + span + 2, base + 8) {
        return;
    }

    let mut prev_x = 0;
    let mut prev_top = 0;
    for i in 0..n {
        let sx = start_x + i * spacing;
        let r = h(ctx, sx, 0, 7);
        let s = ctx.surface_at(sx);
        if r < 0.22 {
            // Toppled: a slab lying where it fell.
            let len = ri(h(ctx, sx, 0, 8), 3, 5);
            for d in 0..len {
                let gx = sx + d;
                let gs = ctx.surface_at(gx);
                ctx.plot(gx, gs - 1, p.wall);
            }
            prev_top = 0;
            continue;
        }
        let sh = ri(r, 4, 9);
        let top = s - sh;
        vline(ctx, sx, top, s - 1, p.wall);
        if h(ctx, sx, 0, 9) < 0.5 {
            vline(ctx, sx + 1, top + 1, s - 1, p.wall);
        }
        if lintels && prev_top != 0 && sx - prev_x <= 6 {
            let ly = if top > prev_top { top } else { prev_top };
            hline(ctx, prev_x, sx, ly - 1, p.trim);
        }
        prev_x = sx;
        prev_top = top;
    }

    match centre {
        Centre::Obelisk => {
            // Obelisk: tapered, and the tallest thing for a long way.
            let oh = ri(h(ctx, ox, 0, 10), 9, 18);
            for d in 0..oh {
                let y = base - 1 - d;
                // The TypeScript wrote the taper as two nested tests,
                // `d > oh - 4 ? 0 : d > oh * 0.5 ? 0 : 1`. Both narrow to the same
                // single column, so they are ORed here rather than nested — same
                // result, and it does not trip `clippy::if_same_then_else`.
                let w = if d > oh - 4 || f64::from(d) > f64::from(oh) * 0.5 {
                    0
                } else {
                    1
                };
                fill(ctx, ox - w, y, ox + w, y, p.wall);
            }
            fill(ctx, ox - 2, base - 1, ox + 2, base - 1, p.trim);
        }
        Centre::Altar => {
            // Altar: a slab on two legs.
            hline(ctx, ox - 2, ox + 2, base - 3, p.trim);
            vline(ctx, ox - 2, base - 2, base - 1, p.wall);
            vline(ctx, ox + 2, base - 2, base - 1, p.wall);
            if h(ctx, ox, 0, 11) < 0.4 {
                let prize = prize_for(30, h(ctx, ox, 0, 12));
                ctx.plot(ox, base - 4, prize);
            }
        }
        Centre::Basin => {
            // Basin: sunk into the ground, lined, and full of whatever this climate
            // pools. In Volcanic that is lava, which makes it a shrine worth respecting.
            fill(ctx, ox - 3, base, ox + 3, base + 4, p.wall);
            fill(ctx, ox - 2, base, ox + 2, base + 3, AIR);
            fill(ctx, ox - 2, base + 1, ox + 2, base + 3, p.fluid);
            vline(ctx, ox - 3, base - 4, base - 1, p.trim);
            vline(ctx, ox + 3, base - 4, base - 1, p.trim);
        }
    }

    drape(ctx, start_x, start_x + span, base - 10, p, 24);
}

/// Well. Small above ground, but it drops a lined shaft 10–22 cells down with
/// water (or worse) at the bottom and sometimes a cache cut into the side — a
/// short, self-contained descent that always crosses a chunk boundary.
fn build_well(ctx: &mut DecorContext<'_>, ox: i32, base: i32, p: Palette) {
    let depth = ri(h(ctx, ox, 0, 3), 10, 22);
    let rim = ri(h(ctx, ox, 0, 4), 2, 3);
    let wear = 0.02 + h(ctx, ox, 0, 5) * 0.14;
    let has_alcove = h(ctx, ox, 0, 6) < 0.5;

    if !overlaps(ctx, ox - 3, base - rim - 4, ox + 5, base + depth + 2) {
        return;
    }

    // Shaft and its lining.
    fill(ctx, ox - 1, base, ox + 1, base + depth, AIR);
    vline(ctx, ox - 2, base, base + depth + 1, p.wall);
    vline(ctx, ox + 2, base, base + depth + 1, p.wall);
    hline(ctx, ox - 2, ox + 2, base + depth + 1, p.wall);

    // Rim above ground.
    fill(ctx, ox - 2, base - rim, ox + 2, base - 1, AIR);
    for y in (base - rim..=base - 1).rev() {
        masonry(ctx, ox - 2, y, p, wear, 8);
        masonry(ctx, ox + 2, y, p, wear, 8);
    }
    // Windlass: two posts and a beam.
    if h(ctx, ox, 0, 7) < 0.6 {
        vline(ctx, ox - 2, base - rim - 3, base - rim - 1, WOOD);
        vline(ctx, ox + 2, base - rim - 3, base - rim - 1, WOOD);
        hline(ctx, ox - 2, ox + 2, base - rim - 4, WOOD);
    }

    // Water at the bottom.
    let fluid_top = base + depth - ri(h(ctx, ox, 0, 8), 2, 5);
    fill(ctx, ox - 1, fluid_top, ox + 1, base + depth, p.fluid);

    if has_alcove {
        let ay = base + depth - ri(h(ctx, ox, 0, 9), 6, 9);
        fill(ctx, ox + 2, ay - 2, ox + 5, ay, p.wall);
        fill(ctx, ox + 2, ay - 1, ox + 4, ay, AIR);
        let prize = prize_for(depth + 30, h(ctx, ox, 0, 10));
        fill(ctx, ox + 3, ay, ox + 4, ay, prize);
    }

    plinth(ctx, ox - 2, ox + 2, base, p, 4);
}

/// Stepped temple. The biggest surface piece — up to 29 wide and 23 tall, so it
/// reliably straddles four chunks. A sealed tomb sits UNDER it holding the prize,
/// which is the payoff for noticing the thing and digging.
///
/// Varies on tier count, tier height, base width, inset per tier, how ruined the
/// upper tiers are, and (in wet climates) how far the jungle has taken it back.
fn build_temple(ctx: &mut DecorContext<'_>, ox: i32, base: i32, p: Palette) {
    let tiers = ri(h(ctx, ox, 0, 3), 3, 5);
    let tier_h = ri(h(ctx, ox, 0, 4), 3, 4);
    let base_half = ri(h(ctx, ox, 0, 5), 9, 14);
    let inset = ri(h(ctx, ox, 0, 6), 2, 3);
    let wear = 0.02 + h(ctx, ox, 0, 7) * 0.12;

    let x0 = ox - base_half;
    let x1 = ox + base_half;
    let top_y = base - tiers * tier_h;
    if !overlaps(ctx, x0, top_y - 3, x1, base + 11) {
        return;
    }

    fill(ctx, x0, top_y - 3, x1, base - 1, AIR);

    // Tiers, each smaller and more broken than the one below it.
    let mut hw = base_half;
    let mut t = 0;
    while t < tiers && hw >= 2 {
        let ty1 = base - t * tier_h - 1;
        let ty0 = base - (t + 1) * tier_h;
        let tw = wear + f64::from(t) * 0.05;
        for y in (ty0..=ty1).rev() {
            masonry_row(ctx, ox - hw, ox + hw, y, p, tw, 8 + t);
        }
        // Hollow the tier out a little so it is masonry, not a solid heap.
        if hw >= 5 && t < tiers - 1 && ty1 - ty0 >= 2 {
            fill(ctx, ox - hw + 2, ty0 + 1, ox + hw - 2, ty1 - 1, AIR);
            hline(ctx, ox - hw + 2, ox + hw - 2, ty1, p.floor);
        }
        drape(ctx, ox - hw, ox + hw, ty1, p, 25 + t);
        hw -= inset;
        t += 1;
    }

    // Shrine on the summit.
    fill(ctx, ox - 1, top_y - 2, ox + 1, top_y - 1, p.trim);
    if h(ctx, ox, 0, 14) < 0.35 {
        ctx.plot(ox, top_y - 3, CRYSTAL);
    }

    // The sealed tomb beneath. No door: you dig in, or you do not get it.
    let ch_w = ri(h(ctx, ox, 0, 15), 3, 5);
    let ch_h = ri(h(ctx, ox, 0, 16), 3, 4);
    let cy0 = base + 3;
    let cy1 = cy0 + ch_h;
    fill(ctx, ox - ch_w - 1, cy0 - 1, ox + ch_w + 1, cy1 + 1, p.trim);
    fill(ctx, ox - ch_w, cy0, ox + ch_w, cy1, AIR);
    // Sarcophagus, and the grave goods behind it.
    fill(ctx, ox - 2, cy1 - 1, ox + 1, cy1, p.wall);
    let prize = prize_for(base + 60, h(ctx, ox, 0, 17));
    fill(ctx, ox + ch_w - 1, cy1 - 1, ox + ch_w, cy1, prize);
    // Whoever built it did not want visitors.
    for x in ox - ch_w..=ox + ch_w {
        if h(ctx, x, cy1, 18) < 0.22 {
            ctx.plot(x, cy1, SPIKE);
        }
    }

    plinth(ctx, x0, x1, base, p, 8);
}

// =============================================================================
// Underground structures
// =============================================================================

/// Cut a lit alcove into a wall and stock it. Extends 2 cells past the shell.
fn alcove(ctx: &mut DecorContext<'_>, x: i32, y: i32, dir: i32, p: Palette, prize: CellId) {
    let xa = x;
    let xb = x + dir * 2;
    let x0 = if xa < xb { xa } else { xb };
    let x1 = if xa < xb { xb } else { xa };
    fill(ctx, x0 - 1, y - 3, x1 + 1, y + 1, p.trim);
    fill(ctx, x0, y - 2, x1, y, AIR);
    fill(ctx, x0, y, x1, y, prize);
    ctx.plot(x + dir, y - 2, p.trim);
}

/// Which face an underground chamber wears. (The TypeScript drew a number:
/// `0` tomb, `1` pillared hall, `2` cistern, `3` mine, `4` forge.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChamberKind {
    Tomb,
    PillaredHall,
    Cistern,
    Mine,
    Forge,
}

/// Underground chamber. One generator, five faces — which one you get is drawn
/// from the origin, and every one of them varies again in size, wear, flooding,
/// hazards and what it is holding.
///
/// Chambers CARVE their own interior and then line it, so they read as rooms cut
/// out of the rock rather than rooms full of it, whatever the cave field was
/// doing here.
fn build_chamber(ctx: &mut DecorContext<'_>, ox: i32, oy: i32, depth: i32, p: Palette) {
    let half_w = ri(h(ctx, ox, oy, 3), 6, 15);
    let half_h = ri(h(ctx, ox, oy, 4), 4, 9);
    let wear = 0.02 + h(ctx, ox, oy, 5) * 0.14;
    let kind_r = h(ctx, ox, oy, 6);
    // Forges only where the rock is already angry; mines only where people could
    // plausibly have got to.
    let kind = if p.fluid == LAVA && kind_r < 0.3 {
        ChamberKind::Forge
    } else if depth < 110 && kind_r < 0.28 {
        ChamberKind::Mine
    } else if kind_r < 0.5 {
        ChamberKind::Tomb
    } else if kind_r < 0.78 {
        ChamberKind::PillaredHall
    } else {
        ChamberKind::Cistern
    };

    let stub_dir = if h(ctx, ox, oy, 7) < 0.5 { -1 } else { 1 };
    let stub_len = if kind == ChamberKind::Tomb {
        0
    } else {
        ri(h(ctx, ox, oy, 8), 3, 5)
    };
    let shaft = if kind == ChamberKind::Mine {
        ri(h(ctx, ox, oy, 9), 6, 10)
    } else {
        0
    };

    let x0 = ox - half_w;
    let x1 = ox + half_w;
    let y0 = oy - half_h;
    let y1 = oy + half_h;
    // The cache goes in the wall OPPOSITE the way in, so you cross whatever the
    // room is holding to reach it — and so the entrance tunnel cannot carve
    // straight through the alcove it would otherwise share a wall with.
    let a_dir = -stub_dir;
    let a_x = if a_dir < 0 { x0 } else { x1 };
    let margin = (if stub_len > 3 { stub_len } else { 3 }) + 2;
    if !overlaps(ctx, x0 - margin, y0 - shaft, x1 + margin, y1) {
        return;
    }

    // Carve, then line. Order matters: the lining must survive the carve.
    fill(ctx, x0 + 1, y0 + 1, x1 - 1, y1 - 1, AIR);
    masonry_row(ctx, x0, x1, y0, p, wear, 10);
    masonry_row(ctx, x0, x1, y1, p, wear * 0.4, 11);
    for y in y0..=y1 {
        masonry(ctx, x0, y, p, wear, 12);
        masonry(ctx, x1, y, p, wear, 12);
    }
    // Laid floor over the bottom course.
    hline(ctx, x0 + 1, x1 - 1, y1 - 1, p.floor);

    match kind {
        ChamberKind::Tomb => {
            // Tomb: sealed, occupied, trapped, and holding the good stuff.
            let sw = ri(h(ctx, ox, oy, 13), 2, 4);
            fill(ctx, ox - sw, y1 - 3, ox + sw, y1 - 2, p.trim);
            fill(ctx, ox - sw + 1, y1 - 3, ox + sw - 1, y1 - 3, p.wall);
            for x in x0 + 2..=x1 - 2 {
                if h(ctx, x, y1, 14) < 0.16 {
                    ctx.plot(x, y1 - 2, SPIKE);
                }
            }
            // Urn niches down both walls.
            let mut y = y0 + 2;
            while y <= y1 - 3 {
                if h(ctx, ox, y, 15) < 0.5 {
                    ctx.plot(x0 + 1, y, p.rubble);
                }
                if h(ctx, ox, y, 16) < 0.5 {
                    ctx.plot(x1 - 1, y, p.rubble);
                }
                y += 3;
            }
            let prize = prize_for(depth, h(ctx, ox, oy, 17));
            alcove(ctx, a_x, y1 - 2, a_dir, p, prize);
        }
        ChamberKind::PillaredHall => {
            // Pillared hall: a natural cavern somebody shored up. Crystals on the
            // ceiling, mushrooms in the damp corners.
            let step = ri(h(ctx, ox, oy, 18), 4, 6);
            let mut x = x0 + step;
            while x <= x1 - step {
                for y in y0 + 1..=y1 - 1 {
                    masonry(ctx, x, y, p, wear * 0.5, 19);
                }
                ctx.plot(x - 1, y0 + 1, p.trim);
                ctx.plot(x + 1, y0 + 1, p.trim);
                x += step;
            }
            for x in x0 + 2..=x1 - 2 {
                let r = h(ctx, x, y0, 20);
                if r < 0.12 {
                    ctx.plot(x, y0 + 1, CRYSTAL);
                    if r < 0.05 {
                        ctx.plot(x, y0 + 2, CRYSTAL);
                    }
                } else if r > 0.94 && p.over.is_some() {
                    ctx.plot(x, y1 - 2, MUSHROOM_CAP);
                }
            }
            let prize = prize_for(depth, h(ctx, ox, oy, 21));
            alcove(ctx, a_x, y1 - 2, a_dir, p, prize);
        }
        ChamberKind::Cistern => {
            // Flooded cistern: brick tank, liquid to a line, a walkway with gaps in it
            // above. Getting to the alcove means crossing whatever is in the tank.
            let level = y1 - 1 - ri(h(ctx, ox, oy, 22), 2, half_h - 1);
            fill(ctx, x0 + 1, level, x1 - 1, y1 - 1, p.fluid);
            let wy = level - 2;
            if wy > y0 + 1 {
                for x in x0 + 1..=x1 - 1 {
                    if h(ctx, x, wy, 23) > 0.22 {
                        ctx.plot(x, wy, p.floor);
                    }
                }
            }
            // Inflow channels through the ceiling.
            let mut x = x0 + 3;
            while x <= x1 - 3 {
                if h(ctx, x, y0, 24) < 0.5 {
                    ctx.plot(x, y0, AIR);
                }
                x += 5;
            }
            let prize = prize_for(depth, h(ctx, ox, oy, 25));
            alcove(ctx, a_x, level - 1, a_dir, p, prize);
        }
        ChamberKind::Mine => {
            // Mine: timber sets every few metres, a rail along the floor, ore left in
            // the walls, and a shaft heading back up towards daylight.
            let step = ri(h(ctx, ox, oy, 26), 4, 6);
            let mut x = x0 + step;
            while x <= x1 - step {
                vline(ctx, x, y0 + 2, y1 - 2, WOOD);
                ctx.plot(x + 1, y0 + 1, WOOD);
                ctx.plot(x - 1, y0 + 1, WOOD);
                x += step;
            }
            hline(ctx, x0 + 2, x1 - 2, y0 + 1, WOOD);
            for x in x0 + 2..=x1 - 2 {
                if h(ctx, x, y1, 27) > 0.25 {
                    ctx.plot(x, y1 - 1, CONVEYOR);
                }
            }
            // The seam they were following, still in the wall.
            let seam = prize_for(depth, h(ctx, ox, oy, 28));
            for y in y0 + 2..=y1 - 2 {
                if h(ctx, x0, y, 29) < 0.55 {
                    ctx.plot(x0, y, seam);
                }
                if h(ctx, x1, y, 30) < 0.35 {
                    ctx.plot(x1, y, seam);
                }
            }
            // Vertical shaft with a timbered collar.
            let sx = ox + ri(h(ctx, ox, oy, 31), -half_w + 3, half_w - 3);
            fill(ctx, sx - 1, y0 - shaft, sx + 1, y0, AIR);
            vline(ctx, sx - 2, y0 - shaft, y0, p.wall);
            vline(ctx, sx + 2, y0 - shaft, y0, p.wall);
            let mut y = y0 - shaft;
            while y <= y0 {
                hline(ctx, sx - 1, sx + 1, y, WOOD);
                y += 3;
            }
        }
        ChamberKind::Forge => {
            // Forge: a lava basin sunk into the floor, an anvil block, ash everywhere,
            // and the best cache in the room sitting right next to the heat.
            let bw = ri(h(ctx, ox, oy, 32), 2, 4);
            fill(ctx, ox - bw - 1, y1 - 3, ox + bw + 1, y1 - 1, OBSIDIAN);
            fill(ctx, ox - bw, y1 - 3, ox + bw, y1 - 2, LAVA);
            fill(ctx, x0 + 1, y0 + 1, x0 + 2, y0 + 2, p.trim);
            for x in x0 + 1..=x1 - 1 {
                if h(ctx, x, y1, 33) < 0.3 {
                    ctx.plot(x, y1 - 2, ASH);
                }
            }
            // Chimney.
            let cx = ox + ri(h(ctx, ox, oy, 34), -2, 2);
            fill(ctx, cx - 1, y0, cx + 1, y0, AIR);
            let prize = prize_for(depth + 60, h(ctx, ox, oy, 35));
            alcove(ctx, a_x, y1 - 2, a_dir, p, prize);
        }
    }

    // Way in — a stub of tunnel that a natural cave may or may not have found.
    if stub_len > 0 {
        let dy = oy + ri(h(ctx, ox, oy, 36), -half_h + 2, half_h - 3);
        let ex = if stub_dir < 0 { x0 } else { x1 };
        for d in 0..=stub_len {
            let x = ex + stub_dir * d;
            fill(ctx, x, dy - 1, x, dy + 1, AIR);
            ctx.plot(x, dy + 2, p.floor);
            ctx.plot(x, dy - 2, p.wall);
        }
    }
}

// =============================================================================
// Passes
// =============================================================================

/// Which surface structure stands here. (The TypeScript drew a number:
/// `0` tower, `1` hut, `2` henge, `3` well, `4` temple.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurfaceKind {
    Tower,
    Hut,
    Henge,
    Well,
    Temple,
}

/// Which surface structure stands here, given the climate and a hash draw.
fn pick_surface_kind(biome: Biome, r: f64) -> SurfaceKind {
    match biome {
        // temple / henge / tower
        Biome::Desert | Biome::Savanna => {
            if r < 0.42 {
                SurfaceKind::Temple
            } else if r < 0.72 {
                SurfaceKind::Henge
            } else {
                SurfaceKind::Tower
            }
        }
        // overgrown temple / tower / hut
        Biome::Jungle => {
            if r < 0.45 {
                SurfaceKind::Temple
            } else if r < 0.75 {
                SurfaceKind::Tower
            } else {
                SurfaceKind::Hut
            }
        }
        Biome::Swamp => {
            if r < 0.45 {
                SurfaceKind::Hut
            } else if r < 0.8 {
                SurfaceKind::Henge
            } else {
                SurfaceKind::Tower
            }
        }
        // shelter / stones / tower
        Biome::Glacier | Biome::Tundra => {
            if r < 0.5 {
                SurfaceKind::Hut
            } else if r < 0.75 {
                SurfaceKind::Henge
            } else {
                SurfaceKind::Tower
            }
        }
        // obsidian shrine / tower
        Biome::Volcanic => {
            if r < 0.55 {
                SurfaceKind::Henge
            } else {
                SurfaceKind::Tower
            }
        }
        // tower / hut / well / henge
        _ => {
            if r < 0.28 {
                SurfaceKind::Tower
            } else if r < 0.55 {
                SurfaceKind::Hut
            } else if r < 0.78 {
                SurfaceKind::Well
            } else {
                SurfaceKind::Henge
            }
        }
    }
}

/// Surface pass. Origins are columns — the y of a surface structure is DERIVED
/// from `surface_at(origin column)`, never scanned for, which is why no vertical
/// lattice is needed here and why a chunk two rows above the ground still paints
/// the top of a tower correctly.
fn decorate_surface(ctx: &mut DecorContext<'_>) {
    if ctx.base_y + CHUNK_CELLS <= BAND_TOP || ctx.base_y >= BAND_BOT {
        return;
    }

    for ox in origin_columns(ctx.base_x, SURF_REACH_X, SURF_STRIDE, SURF_PHASE) {
        if h(ctx, ox, 0, 1) >= SURF_DENSITY {
            continue; // rarity gate, one hash
        }

        let base = ctx.surface_at(ox);
        // Nothing is built on a cliff. Four sparse probes across the core of the
        // footprint — cheap, and purely positional so every chunk agrees. (The wide
        // pieces cope with the rest via `plinth`, which follows the ground column by
        // column, so this only has to reject genuinely broken sites.)
        let mut lo = base;
        let mut hi = base;
        let mut d = -8;
        while d <= 8 {
            if d != 0 {
                let s = ctx.surface_at(ox + d);
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
            }
            d += 4;
        }
        if hi - lo > 8 {
            continue;
        }

        let col = ctx.profile_at(ox);
        let biome = col.surf_a;
        let p = surface_palette(biome, h(ctx, ox, 0, 2));
        match pick_surface_kind(biome, h(ctx, ox, 0, 40)) {
            SurfaceKind::Tower => build_tower(ctx, ox, base, p),
            SurfaceKind::Hut => build_hut(ctx, ox, base, p),
            SurfaceKind::Henge => build_henge(ctx, ox, base, p),
            SurfaceKind::Well => build_well(ctx, ox, base, p),
            SurfaceKind::Temple => build_temple(ctx, ox, base, p),
        }
    }
}

/// Underground pass. A genuine 2D origin lattice, gated on depth.
fn decorate_underground(ctx: &mut DecorContext<'_>) {
    if ctx.base_y + CHUNK_CELLS < SUB_BAND_TOP {
        return;
    }

    for (ox, oy) in origin_cells(
        ctx.base_x,
        ctx.base_y,
        SUB_REACH_X,
        SUB_REACH_Y,
        Lattice {
            stride_x: SUB_STRIDE_X,
            stride_y: SUB_STRIDE_Y,
            phase_x: SUB_PHASE_X,
            phase_y: SUB_PHASE_Y,
        },
    ) {
        if h(ctx, ox, oy, 0) >= SUB_DENSITY {
            continue; // rarity gate, one hash
        }
        let depth = oy - ctx.surface_at(ox);
        if depth < SUB_MIN_DEPTH {
            continue; // must be properly buried
        }
        let col = ctx.profile_at(ox);
        let p = sub_palette(col.surf_a, depth, h(ctx, ox, oy, 2));
        build_chamber(ctx, ox, oy, depth, p);
    }
}

/// Structures. Two independent origin lattices — columns on the surface, a 2D
/// grid underground — both scanned `reach` past every chunk edge so a building
/// that overhangs into this chunk from outside still paints its share here.
pub struct StructureDecorator;

impl Decorator for StructureDecorator {
    fn name(&self) -> &'static str {
        "structures"
    }

    fn reach_x(&self) -> i32 {
        REACH_X
    }

    fn reach_y(&self) -> i32 {
        REACH_Y
    }

    fn decorate(&self, ctx: &mut DecorContext<'_>) {
        decorate_surface(ctx);
        decorate_underground(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorldScale;
    use crate::config::worldgen::SEED;
    use crate::sim::biomes::column_profile_at;
    use crate::sim::materials::EMPTY;
    use crate::sim::noise::Noise;
    use crate::sim::worldgen::heightmap::Heightmap;

    const CELLS: usize = (CHUNK_CELLS * CHUNK_CELLS) as usize;

    /// A background code that is not `EMPTY`, so the `AIR` a structure carves
    /// registers as a change. `plot_if_empty` never fires against it, which is why
    /// the reach sweep also runs a pass over an `EMPTY` background.
    const SENTINEL: CellId = block::BOUNCE;

    /// Run `build` over one chunk-sized window and hand back what it painted.
    fn paint(
        noise: &Noise,
        hm: &mut Heightmap,
        base_x: i32,
        base_y: i32,
        background: CellId,
        build: &mut dyn FnMut(&mut DecorContext<'_>),
    ) -> Vec<CellId> {
        let mut out = vec![background; CELLS];
        let mut ctx = DecorContext::new(
            noise,
            SEED,
            base_x,
            base_y,
            &mut out,
            hm,
            WorldScale::LEGACY,
        );
        build(&mut ctx);
        out
    }

    #[inline]
    fn at(chunk: &[CellId], base_x: i32, base_y: i32, wcx: i32, wcy: i32) -> CellId {
        chunk[((wcy - base_y) * CHUNK_CELLS + (wcx - base_x)) as usize]
    }

    /// Every surface builder, keyed off the enum so the two sweeps below cannot
    /// drift apart.
    fn build_surface(
        ctx: &mut DecorContext<'_>,
        kind: SurfaceKind,
        ox: i32,
        base: i32,
        p: Palette,
    ) {
        match kind {
            SurfaceKind::Tower => build_tower(ctx, ox, base, p),
            SurfaceKind::Hut => build_hut(ctx, ox, base, p),
            SurfaceKind::Henge => build_henge(ctx, ox, base, p),
            SurfaceKind::Well => build_well(ctx, ox, base, p),
            SurfaceKind::Temple => build_temple(ctx, ox, base, p),
        }
    }

    const EVERY_KIND: [SurfaceKind; 5] = [
        SurfaceKind::Tower,
        SurfaceKind::Hut,
        SurfaceKind::Henge,
        SurfaceKind::Well,
        SurfaceKind::Temple,
    ];

    #[test]
    fn the_placement_constants_are_the_tuned_ones() {
        // These ARE where the built world stands. A typo here moves every landmark,
        // and — worse — moves it somewhere the TypeScript does not.
        assert_eq!(SURF_STRIDE, 96);
        assert_eq!(SURF_PHASE, 41);
        assert_eq!(SURF_DENSITY, 0.46);
        assert_eq!(SUB_STRIDE_X, 72);
        assert_eq!(SUB_STRIDE_Y, 56);
        assert_eq!(SUB_PHASE_X, 17);
        assert_eq!(SUB_PHASE_Y, 23);
        assert_eq!(SUB_DENSITY, 0.22);
        assert_eq!(SUB_MIN_DEPTH, 44);
        assert_eq!(REACH_X, 20);
        assert_eq!(REACH_Y, 28);
        // The bands, as floor/ceil of the same expression the TypeScript used.
        // `24 * 1.6 + 8` is 46.400000000000006 in f64 and in JavaScript alike —
        // spelt as the expression rather than as 46.4 so the two agree bit for bit.
        assert_eq!(SURF_SPAN, 24.0 * 1.6 + 8.0);
        assert_eq!(BAND_TOP, 1 - 28);
        assert_eq!(BAND_BOT, 95 + 24);
        assert_eq!(SUB_BAND_TOP, 1 + 44 - 20);
    }

    #[test]
    fn ri_answers_the_high_end_of_an_inverted_range() {
        // `ri(_, 2, 1)` happens for real: a chamber at half_h 4 asks for
        // `ri(_, -half_h + 2, half_h - 3)` = `ri(_, -2, 1)`, and the tomb's niche
        // walks invert at the small end too. JavaScript's clamp chain answered
        // `hi`; `i32::clamp` would panic.
        assert_eq!(ri(0.0, 2, 1), 1);
        assert_eq!(ri(0.999, 2, 1), 1);
        assert_eq!(ri(0.0, 3, 7), 3);
        assert_eq!(ri(0.999_999, 3, 7), 7);
        assert_eq!(ri(0.5, 0, 1), 1);
        assert_eq!(ri(0.5, -4, 4), 0);
    }

    /// Origins a real surface pass would actually build on: past the rarity gate
    /// and not on a cliff.
    fn live_surface_origins(noise: &Noise, hm: &mut Heightmap, want: usize) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        let mut scratch = vec![EMPTY; CELLS];
        let mut ox = SURF_PHASE - 200 * SURF_STRIDE;
        while out.len() < want {
            let gate = {
                let ctx =
                    DecorContext::new(noise, SEED, 0, 0, &mut scratch, hm, WorldScale::LEGACY);
                h(&ctx, ox, 0, 1)
            };
            if gate < SURF_DENSITY {
                let base = hm.surface_row_at(noise, ox, None, WorldScale::LEGACY);
                let mut lo = base;
                let mut hi = base;
                for d in [-8, -4, 4, 8] {
                    let s = hm.surface_row_at(noise, ox + d, None, WorldScale::LEGACY);
                    lo = lo.min(s);
                    hi = hi.max(s);
                }
                if hi - lo <= 8 {
                    out.push((ox, base));
                }
            }
            ox += SURF_STRIDE;
            assert!(
                ox < SURF_PHASE + 4000 * SURF_STRIDE,
                "no live origins found"
            );
        }
        out
    }

    #[test]
    fn a_structure_is_painted_identically_from_either_side_of_a_boundary() {
        // The contract, tested head on: the SAME structure, authored from the same
        // origin, drawn into two windows offset from each other. Every cell they
        // share must come out identical, or a building straddling a chunk edge
        // disagrees with itself and the seam appears only when the player walks in
        // from the wrong side.
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let origins = live_surface_origins(&noise, &mut hm, 24);

        for &(ox, base) in &origins {
            for offset in [7, 13, 16, 25] {
                for kind in EVERY_KIND {
                    let mut run = |ctx: &mut DecorContext<'_>| {
                        let p = surface_palette(Biome::Plains, h(ctx, ox, 0, 2));
                        build_surface(ctx, kind, ox, base, p);
                    };
                    let (ax, ay) = (ox - 20, base - 20);
                    let (bx, by) = (ax + offset, ay);
                    let a = paint(&noise, &mut hm, ax, ay, SENTINEL, &mut run);
                    let b = paint(&noise, &mut hm, bx, by, SENTINEL, &mut run);
                    for wcy in ay..ay + CHUNK_CELLS {
                        for wcx in bx..ax + CHUNK_CELLS {
                            assert_eq!(
                                at(&a, ax, ay, wcx, wcy),
                                at(&b, bx, by, wcx, wcy),
                                "{kind:?} at origin {ox} disagrees at ({wcx}, {wcy}) \
                                 between windows based at {ax} and {bx}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_whole_pass_agrees_across_two_overlapping_windows() {
        // As above, but through the real passes, so the origin SCANS are on trial
        // and not just the builders.
        //
        // The compared strip is the shared region shrunk by one column at each end:
        // two windows 16 apart scan candidate sets that differ at the far edges, and
        // a candidate only one of them scanned may legitimately paint into the
        // outermost column of the overlap (see the henge note in
        // `nothing_is_painted_outside_the_declared_reach`).
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let d = StructureDecorator;

        for by in [-64i32, 0, 32, 96, 160, 320] {
            for bx in [-1024i32, -96, 0, 41, 288, 4096] {
                let mut run = |ctx: &mut DecorContext<'_>| d.decorate(ctx);
                let a = paint(&noise, &mut hm, bx, by, SENTINEL, &mut run);
                let b = paint(&noise, &mut hm, bx + 16, by, SENTINEL, &mut run);
                for wcy in by..by + CHUNK_CELLS {
                    for wcx in bx + 17..=bx + 30 {
                        assert_eq!(
                            at(&a, bx, by, wcx, wcy),
                            at(&b, bx + 16, by, wcx, wcy),
                            "chunk ({bx}, {by}) and its half-shifted twin disagree at \
                             ({wcx}, {wcy})"
                        );
                    }
                }
            }
        }
    }

    /// Every cell one structure touches, gathered by tiling windows over a box far
    /// larger than the declared reach.
    fn painted_by(
        noise: &Noise,
        hm: &mut Heightmap,
        ox: i32,
        oy: i32,
        build: &mut dyn FnMut(&mut DecorContext<'_>),
    ) -> Vec<(i32, i32)> {
        let span = 2 * CHUNK_CELLS;
        let mut hits = Vec::new();
        // Two backgrounds: the sentinel catches carved AIR, EMPTY catches the vines
        // `plot_if_empty` hangs. Neither alone sees the whole footprint.
        for background in [SENTINEL, EMPTY] {
            let mut by = oy - span;
            while by <= oy + span {
                let mut bx = ox - span;
                while bx <= ox + span {
                    let out = paint(noise, hm, bx, by, background, build);
                    for (i, &c) in out.iter().enumerate() {
                        if c != background {
                            hits.push((bx + i as i32 % CHUNK_CELLS, by + i as i32 / CHUNK_CELLS));
                        }
                    }
                    bx += CHUNK_CELLS;
                }
                by += CHUNK_CELLS;
            }
        }
        hits
    }

    #[test]
    fn nothing_is_painted_outside_the_declared_reach() {
        // An under-declared reach shows up in the finished world as buildings
        // clipped at a chunk edge, and only from certain approach directions.
        // Assert it structurally instead: sweep windows over a box two chunks
        // bigger than the reach in every direction and check nothing landed outside
        // the declared box around the origin.
        //
        // NOTE: this is the DECLARED reach (REACH_X = 20), not the surface pass's
        // own scan reach (SURF_REACH_X = 15). A maximal henge — 5 stones at spacing
        // 6, so `start_x = ox - 12` and the last stone at `ox + 12` — can drop a
        // toppled slab 4 cells further, to `ox + 16`. That is inside the decorator's
        // declared reach but one cell outside the surface scan's: a latent
        // single-cell seam carried over faithfully from the TypeScript, not one
        // introduced by this port.
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();

        for &(ox, base) in &live_surface_origins(&noise, &mut hm, 12) {
            for kind in EVERY_KIND {
                for biome in [Biome::Plains, Biome::Jungle, Biome::Volcanic] {
                    let mut run = |ctx: &mut DecorContext<'_>| {
                        let p = surface_palette(biome, h(ctx, ox, 0, 2));
                        build_surface(ctx, kind, ox, base, p);
                    };
                    let hits = painted_by(&noise, &mut hm, ox, base, &mut run);
                    assert!(!hits.is_empty(), "{kind:?} at {ox} painted nothing at all");
                    for (x, y) in hits {
                        assert!(
                            (x - ox).abs() <= REACH_X,
                            "{kind:?} at {ox} painted x={x}, {} past REACH_X",
                            (x - ox).abs() - REACH_X
                        );
                        assert!(
                            (y - base).abs() <= REACH_Y,
                            "{kind:?} at {ox} (base {base}) painted y={y}, {} past REACH_Y",
                            (y - base).abs() - REACH_Y
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_chamber_stays_inside_the_underground_reach() {
        // The underground pass scans exactly SUB_REACH_X/SUB_REACH_Y past each edge,
        // so a chamber overrunning either is a seam rather than a cosmetic issue.
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();

        let mut tested = 0;
        let mut oy = SUB_PHASE_Y + 2 * SUB_STRIDE_Y;
        while oy < SUB_PHASE_Y + 6 * SUB_STRIDE_Y {
            let mut ox = SUB_PHASE_X - 8 * SUB_STRIDE_X;
            while ox < SUB_PHASE_X + 8 * SUB_STRIDE_X {
                let depth = oy - hm.surface_row_at(&noise, ox, None, WorldScale::LEGACY);
                if depth >= SUB_MIN_DEPTH {
                    let biome = column_profile_at(&noise, ox, WorldScale::LEGACY).surf_a;
                    let mut run = |ctx: &mut DecorContext<'_>| {
                        let p = sub_palette(biome, depth, h(ctx, ox, oy, 2));
                        build_chamber(ctx, ox, oy, depth, p);
                    };
                    let hits = painted_by(&noise, &mut hm, ox, oy, &mut run);
                    assert!(!hits.is_empty(), "chamber at ({ox}, {oy}) painted nothing");
                    for (x, y) in hits {
                        assert!(
                            (x - ox).abs() <= SUB_REACH_X,
                            "chamber at ({ox}, {oy}) painted x={x}"
                        );
                        assert!(
                            (y - oy).abs() <= SUB_REACH_Y,
                            "chamber at ({ox}, {oy}) painted y={y}"
                        );
                    }
                    tested += 1;
                }
                ox += SUB_STRIDE_X;
            }
            oy += SUB_STRIDE_Y;
        }
        assert!(
            tested > 20,
            "only {tested} chambers were deep enough to test"
        );
    }

    #[test]
    fn the_bands_gate_out_the_chunks_that_cannot_hold_anything() {
        // The band tests are the only reason a chunk 5000 cells down does not pay
        // for the surface scan. If they ever gate out a chunk a structure DOES
        // reach, buildings lose their tops and bottoms instead.
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let d = StructureDecorator;

        for by in [-4096, -1024, BAND_TOP - CHUNK_CELLS, BAND_BOT, 8192] {
            let mut run = |ctx: &mut DecorContext<'_>| decorate_surface(ctx);
            let out = paint(&noise, &mut hm, 0, by, SENTINEL, &mut run);
            assert!(
                out.iter().all(|&c| c == SENTINEL),
                "the surface pass painted into gated-out chunk row {by}"
            );
        }

        // And a chunk on the ground line does get built in, sometimes.
        let mut any = false;
        let mut bx = -4096;
        while bx < 4096 {
            let mut run = |ctx: &mut DecorContext<'_>| d.decorate(ctx);
            let out = paint(&noise, &mut hm, bx, 32, SENTINEL, &mut run);
            if out.iter().any(|&c| c != SENTINEL) {
                any = true;
                break;
            }
            bx += CHUNK_CELLS;
        }
        assert!(
            any,
            "no structure was built anywhere along 8k columns of surface"
        );
    }
}

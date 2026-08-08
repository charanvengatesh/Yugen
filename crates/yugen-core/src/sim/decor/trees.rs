//! Flora.
//!
//! One decorator, nine silhouettes, keyed off the column's biome mix. Everything
//! here obeys the [`super`] contract: a plant is authored from its origin COLUMN
//! and every chunk that the plant can touch recomputes it in full and keeps only
//! the cells that land inside itself. So every decision — species, height, lean,
//! canopy radius, which leaf on which cell — is a [`DecorContext::hash`] of
//! absolute coordinates and nothing else. No PRNG, no state, no neighbour reads.
//!
//! The shapes, roughly:
//!
//! ```text
//!   broadleaf  conifer   jungle      acacia    mangrove  cactus  snag  shroom
//!     (##)       /\     .:###:.     ~~~~~~~      (##)      |      \|/    (@@)
//!    (####)     /##\   ::#####::   ((######))   (####)    ,|.      |      ||
//!     |##|     /####\  ..#|##|#..     \  /       \|/       |      /|      ||
//!      ||       ##||#   ,-|#|-,       |  |        |        |       |      ||
//! ```
//!
//! Material codes come from [`block`], resolved at COMPILE time. The TypeScript
//! resolved them through `codeOf` at module load, with the same rule stated as a
//! warning — never hardcode a numeric code, the registry owns those. Here the
//! registry is compiled content and a bad id is a build error.

use super::{DecorContext, Decorator, origin_columns};
use crate::config::STEP_UP_CELLS;
use crate::config::world::CHUNK_CELLS;
use crate::config::worldgen::{SURFACE_AMPLITUDE, SURFACE_ANCHOR_Y, WorldScale};
use crate::sim::biomes::{BIOME_COUNT, BIOMES, Biome, ColumnProfile};
use crate::sim::materials::{CellId, block};

// --- Palette -----------------------------------------------------------------
const WOOD: CellId = block::WOOD;
const LEAF: CellId = block::LEAVES;
const LEAF_PINE: CellId = block::LEAVES_PINE;
const LEAF_JUNGLE: CellId = block::LEAVES_JUNGLE;
const LEAF_AUTUMN: CellId = block::LEAVES_AUTUMN;
const CACTUS: CellId = block::CACTUS;
const CAP: CellId = block::MUSHROOM_CAP;
const STEM: CellId = block::MUSHROOM_STEM;
const MOSS: CellId = block::MOSS;
const VINE: CellId = block::VINE;
const SNOW: CellId = block::SNOW;
const ASH: CellId = block::ASH;

// --- Salts -------------------------------------------------------------------
// Every independent random decision needs its own salt; two decisions drawn from
// the same (wcx, salt) pair would be perfectly correlated and every tree in the
// world would, say, lean in the direction implied by its height.
const S_GATE: i32 = 101; // does anything grow in this column at all
const S_MIX: i32 = 211; // which of the two blended biomes governs
const S_SPECIES: i32 = 307;
const S_H: i32 = 409; // height
const S_LEAN: i32 = 503; // lean magnitude / direction (+1)
const S_RX: i32 = 601; // canopy half-width, and per-species shape flags
const S_RY: i32 = 701; // canopy half-height
const S_NB: i32 = 809; // branch / arm count
const S_BR: i32 = 907; // + k: branch height along the trunk
const S_BRD: i32 = 1009; // + k: branch direction
const S_BRL: i32 = 1103; // + k: branch length
const S_ROOT: i32 = 1201; // + k: root flares and prop roots
const S_LOBE: i32 = 1301; // canopy side lobes
const S_PAL: i32 = 1409; // palette coin-flips
const S_VINE: i32 = 1511; // vine seeding
const S_BLOB: i32 = 1709; // + 137k: per-cell canopy dither
const S_CLUT: i32 = 1801; // ground clutter gate
const S_CMIX: i32 = 1901;
const S_CKIND: i32 = 2003;
/// Which plane a tree stands in. See [`BEHIND_SHARE`].
const S_DEPTH: i32 = 2101;

// --- Reach -------------------------------------------------------------------
// The widest thing here is a jungle tree: 2 cells of lean plus a crown of 6, or
// a 4-cell limb tipped with a 2-cell leaf clump — 8 either way. The tallest is
// also jungle: a 22-cell trunk under a crown reaching 3 more cells above it.
// Under-declaring either is what clips a tree at a chunk seam, so these two are
// the numbers to change if a shape below ever grows.
const REACH_X: i32 = 8;
const REACH_Y: i32 = 26;
/// Deepest a root or a seeded moss cell reaches below the surface row.
const ROOT_DEPTH: i32 = 3;

// Candidate lattices, in AUTHORED cells — every one of them is multiplied by the
// raster factor at the scan, so the spacing grows with the trees standing in it.
// Trees are sparse and expensive, clutter is dense and cheap, so they get
// separate strides — and clutter, being 2 authored cells wide at most, gets a far
// smaller scan window.
const TREE_STRIDE: i32 = 3;
const TREE_PHASE: i32 = 1;
const CLUTTER_STRIDE: i32 = 2;
const CLUTTER_PHASE: i32 = 0;
const CLUTTER_REACH: i32 = 2;

/// Share of trees that stand BEHIND the play plane.
///
/// Drawn into the chunk's background plane, which nothing collides with and the
/// automata never touch — so these are trees you walk straight through. A third
/// is enough that every stretch of woodland has a lane through it while the
/// forest still reads as a forest, and it is a positional hash rather than a
/// second lattice so the choice stays a pure function of the column.
///
/// They are not free scenery: the light solver counts a filled back cell as a
/// wall, so a stand of them shades the ground beneath.
const BEHIND_SHARE: f64 = 0.35;

/// No biome grows on more than this fraction of its candidate columns.
const MAX_TREE_DENSITY: f64 = 0.56;
/// Likewise for ground cover. Both let the gate reject before any noise work.
const MAX_CLUTTER_DENSITY: f64 = 0.62;

/// `Math.round`, JavaScript's tie rule.
///
/// JS breaks halves toward +infinity; Rust's `f64::round` breaks them away from
/// zero, so `-2.5` rounds to `-2` there and `-3` here. Every rounded quantity in
/// this file is a trunk lean or a limb rise, and leans go NEGATIVE — a tree
/// leaning left would sit a cell off from the TypeScript's without this.
/// [`crate::sim::worldgen::heightmap`] carries the same helper for the same
/// reason.
#[inline]
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

// --- Vertical guard ----------------------------------------------------------
// Decorators run for EVERY chunk, including the hundreds of purely underground
// ones a streaming world pages in. Flora only ever exists in a band around the
// surface, and that band is bounded by constants: the heightmap is
// `round(ANCHOR + heightOffset + fbm1 * ampScale * AMPLITUDE)` with both biome
// terms drawn from BIOMES. Derive the band once and skip the whole pass for any
// chunk outside it. Pure arithmetic on constants — no neighbour reads, nothing
// order-dependent.
//
// The TypeScript ran this at module load into four `const`s. `BIOMES` is a
// `static`, which a `const` cannot read; the loop is eight float comparisons and
// runs once per chunk, which is nothing beside the candidate scan it guards.
fn flora_band() -> (i32, i32) {
    let mut max_amp_scale = 0.0f64;
    let mut min_height_offset = 0.0f64;
    let mut max_height_offset = 0.0f64;
    for b in &BIOMES {
        if b.amp_scale > max_amp_scale {
            max_amp_scale = b.amp_scale;
        }
        if b.height_offset < min_height_offset {
            min_height_offset = b.height_offset;
        }
        if b.height_offset > max_height_offset {
            max_height_offset = b.height_offset;
        }
    }
    // fbm1 is normalised to ~[-1,1] and the sum is rounded, so allow a little slop.
    let surf_swing = max_amp_scale * f64::from(SURFACE_AMPLITUDE) + 3.0;
    let highest_surface = (f64::from(SURFACE_ANCHOR_Y) + min_height_offset - surf_swing).floor();
    let lowest_surface = (f64::from(SURFACE_ANCHOR_Y) + max_height_offset + surf_swing).ceil();
    // Every row above is a LEGACY row and every reach is a legacy length, so the
    // whole band crosses out to world rows together. Getting this wrong does not
    // draw a wrong tree — it draws none at all, because the early-out above
    // decides the chunk is nowhere near the surface.
    let scale = WorldScale::LIVE;
    let k = scale.raster();
    (
        scale.row(highest_surface as i32) - REACH_Y * k,
        scale.row(lowest_surface as i32) + ROOT_DEPTH * k,
    )
}

// --- Species -----------------------------------------------------------------
const SP_BROADLEAF: u8 = 0;
const SP_CONIFER: u8 = 1;
const SP_JUNGLE: u8 = 2;
const SP_ACACIA: u8 = 3;
const SP_MANGROVE: u8 = 4;
const SP_CACTUS: u8 = 5;
const SP_SNAG: u8 = 6;
const SP_SHROOM: u8 = 7;
const SP_SHRUB: u8 = 8;

/// How much surface wobble a species tolerates under its feet, in cells.
const SLOPE_LIMIT: [i32; 9] = [1, 1, 1, 1, 2, 2, 2, 2, 3];

/// What one biome grows.
struct Flora {
    /// Fraction of tree-candidate columns that grow something.
    density: f64,
    /// Fraction of clutter-candidate columns that grow ground cover.
    clutter: f64,
    /// Species, paired with its RAW weight.
    ///
    /// The TypeScript's `flora()` helper normalised these into cumulative cuts
    /// at module load. [`species_for`] does the identical arithmetic in the
    /// identical order at lookup time, because a `static` initialiser cannot run
    /// a loop — and it runs only for candidates that already passed the density
    /// gate.
    weights: &'static [(u8, f64)],
}

/// The species whose cumulative cut first exceeds `r`.
///
/// Normalises the raw weights exactly as the TypeScript's `flora()` did —
/// sequential sum for the total, then a running `acc += w / total` — so the cuts
/// are bit-for-bit the ones it compared against. The last cut is forced to 1 so
/// a rounding shortfall can never drop a column on the floor.
fn species_for(f: &Flora, r: f64) -> u8 {
    let mut total = 0.0;
    for w in f.weights {
        total += w.1;
    }
    let last = f.weights.len() - 1;
    let mut acc = 0.0;
    for (i, &(sp, w)) in f.weights.iter().enumerate() {
        acc += w / total;
        let cut = if i == last { 1.0 } else { acc };
        if r < cut {
            return sp;
        }
    }
    f.weights[last].0
}

/// What grows where. Density is per candidate column at [`TREE_STRIDE`], so 0.5
/// with a stride of 3 means a plant every ~6 columns — canopies overlap and the
/// jungle closes over your head; savanna's 0.17 is a tree every ~18 columns.
///
/// Note this deliberately ignores [`crate::sim::biomes::BiomeDef::trees`]: that
/// flag only says "grows temperate forest", and a desert full of cacti or a
/// volcanic field of burnt snags is still flora. Density does the thinning
/// instead.
///
/// Indexed by [`Biome::index`], so this array is in palette order and must stay
/// that way. The TypeScript keyed a record by biome id and fell back to a `BARE`
/// entry — `flora(0.1, 0.15, [[SP_SHRUB, 1]])` — for an unlisted biome. Every
/// biome is listed, here as there, so that fallback was already dead; a total
/// array says so.
#[rustfmt::skip]
static FLORA: [Flora; BIOME_COUNT] = [
    // plains
    Flora { density: 0.33, clutter: 0.5, weights: &[
        (SP_BROADLEAF, 0.7), (SP_SHRUB, 0.23), (SP_SNAG, 0.05), (SP_SHROOM, 0.02),
    ] },
    // desert
    Flora { density: 0.26, clutter: 0.2, weights: &[
        (SP_CACTUS, 0.76), (SP_SNAG, 0.14), (SP_SHRUB, 0.1),
    ] },
    // tundra
    Flora { density: 0.3, clutter: 0.22, weights: &[
        (SP_CONIFER, 0.8), (SP_SNAG, 0.12), (SP_SHRUB, 0.08),
    ] },
    // swamp
    Flora { density: 0.34, clutter: 0.58, weights: &[
        (SP_MANGROVE, 0.5), (SP_SHROOM, 0.31), (SP_SHRUB, 0.19),
    ] },
    // volcanic
    Flora { density: 0.07, clutter: 0.1, weights: &[
        (SP_SNAG, 0.88), (SP_SHRUB, 0.12),
    ] },
    // glacier
    Flora { density: 0.08, clutter: 0.12, weights: &[
        (SP_CONIFER, 0.58), (SP_SNAG, 0.42),
    ] },
    // jungle
    Flora { density: 0.5, clutter: 0.6, weights: &[
        (SP_JUNGLE, 0.6), (SP_SHRUB, 0.26), (SP_SHROOM, 0.14),
    ] },
    // savanna
    Flora { density: 0.17, clutter: 0.36, weights: &[
        (SP_ACACIA, 0.6), (SP_SHRUB, 0.32), (SP_SNAG, 0.08),
    ] },
    // badlands -- barer than the desert, which at least has cacti. What stands
    // here is mostly dead: snags first, then the scrub that can live on the
    // little water the clay holds.
    Flora { density: 0.12, clutter: 0.18, weights: &[
        (SP_SNAG, 0.62), (SP_SHRUB, 0.38),
    ] },
    // cinderveld -- what a fire leaves standing: snags nearly alone, the odd
    // shrub in a hollow the embers missed. Barest ground in the game bar the
    // volcano itself.
    Flora { density: 0.09, clutter: 0.14, weights: &[
        (SP_SNAG, 0.8), (SP_SHRUB, 0.2),
    ] },
    // mirefen -- a moor is not bare, it is LOW: dense shrub and shroom cover
    // with almost nothing tall enough to call a tree.
    Flora { density: 0.3, clutter: 0.55, weights: &[
        (SP_SHRUB, 0.55), (SP_SHROOM, 0.35), (SP_SNAG, 0.1),
    ] },
];

#[inline]
fn flora_of(b: Biome) -> &'static Flora {
    &FLORA[b.index()]
}

// --- Palette helpers ---------------------------------------------------------

/// Autumn comes in bands, not per tree — a low-frequency 1D sample means whole
/// stretches of forest turn at once and the eye reads it as a region rather than
/// as noise.
fn autumn_band(ctx: &DecorContext<'_>, wcx: i32) -> bool {
    ctx.noise.n1(f64::from(wcx) * 0.0075 + 41.7) > 0.3
}

/// Canopy material for a biome, with a little within-biome variety.
fn canopy_of(ctx: &DecorContext<'_>, b: Biome, wcx: i32) -> CellId {
    match b {
        Biome::Jungle => LEAF_JUNGLE,
        Biome::Tundra | Biome::Glacier => LEAF_PINE,
        // Dark, sodden green with the odd sickly-bright crown.
        Biome::Swamp => {
            if ctx.hash(wcx, S_PAL) < 0.22 {
                LEAF_JUNGLE
            } else {
                LEAF_PINE
            }
        }
        Biome::Savanna => {
            if ctx.hash(wcx, S_PAL) < 0.55 {
                LEAF_AUTUMN
            } else {
                LEAF
            }
        }
        _ => {
            if autumn_band(ctx, wcx) {
                LEAF_AUTUMN
            } else {
                LEAF
            }
        }
    }
}

// --- Geometry primitives -----------------------------------------------------

/// A filled ellipse of `code` with a ragged rim, painted over air only so canopy
/// never eats terrain. The rim fade is a per-CELL hash of absolute coordinates,
/// which is what lets two chunks split one canopy and still agree cell for cell.
/// One authored cell, in world cells.
///
/// Every number in [`SHAPES`] and every inline dimension below is authored
/// against the 1x world the trees were drawn for, and crosses to world cells
/// here. This is the same "authored units in, world cells out" discipline
/// `WorldScale` uses for terrain, and it is what keeps the tuned numbers
/// readable: a 6-cell trunk is still written `6`.
#[inline]
fn a(ctx: &DecorContext<'_>, authored: i32) -> i32 {
    authored * ctx.raster()
}

/// As [`a`], but never returns less than one cell — for thicknesses, which
/// must not round away to nothing.
#[inline]
fn a1(ctx: &DecorContext<'_>, authored: i32) -> i32 {
    let v = a(ctx, authored);
    if v < 1 { 1 } else { v }
}

/// What a species is shaped like, in AUTHORED cells.
///
/// These were forty-odd magic numbers spread across nine drawing functions. They
/// are gathered here because the world got four times finer and every one of
/// them had to be found and reconsidered — which is a thing you can do to a
/// table and cannot reasonably do to numbers buried in nine `if` ladders.
///
/// `trunk_w`/`limb_w` are NEW. Until the world scaled, every primitive here drew
/// exactly one cell per step, and that one-cell-per-step limit is the whole
/// reason the trees were nearest-upscaled instead of drawn: there was no way to
/// express a thick trunk. A trunk one authored cell wide is now `raster()` cells
/// wide and can taper along its height.
#[derive(Clone, Copy)]
struct TreeShape {
    /// Trunk height range, authored cells.
    height: (i32, i32),
    /// Trunk width at the root and at the crown, authored cells.
    trunk_w: (i32, i32),
    /// Maximum lean over the full height, authored cells.
    lean_max: i32,
    /// Crown half-extents, authored cells: (rx range, ry range).
    crown_rx: (i32, i32),
    crown_ry: (i32, i32),
    /// Limb reach and thickness, authored cells.
    limb_len: (i32, i32),
    limb_w: i32,
    /// How far the base flares either side, authored cells. 0 for no flare.
    root_flare: i32,
}

impl TreeShape {
    /// Pick a value in `range` from a positional hash. Inclusive at both ends.
    #[inline]
    fn pick(ctx: &DecorContext<'_>, wcx: i32, salt: i32, range: (i32, i32)) -> i32 {
        let (lo, hi) = range;
        if hi <= lo {
            return lo;
        }
        lo + (ctx.hash(wcx, salt) * f64::from(hi - lo + 1)) as i32
    }
}

/// Indexed by the `SP_*` species codes.
static SHAPES: [TreeShape; 9] = [
    // SP_BROADLEAF — a rounded canopy on a stout trunk.
    TreeShape {
        height: (6, 10),
        trunk_w: (1, 1),
        lean_max: 2,
        crown_rx: (3, 4),
        crown_ry: (2, 4),
        limb_len: (2, 3),
        limb_w: 1,
        root_flare: 1,
    },
    // SP_CONIFER — tall, straight, tapering to a spire. Drawn by its own taper
    // scan rather than a crown blob, so the crown fields are unused.
    TreeShape {
        height: (9, 16),
        trunk_w: (1, 1),
        lean_max: 0,
        crown_rx: (0, 0),
        crown_ry: (0, 0),
        limb_len: (0, 0),
        limb_w: 1,
        root_flare: 1,
    },
    // SP_JUNGLE — the tallest thing that grows, on buttress roots.
    TreeShape {
        height: (14, 22),
        trunk_w: (3, 2),
        lean_max: 2,
        crown_rx: (4, 6),
        crown_ry: (3, 4),
        limb_len: (3, 4),
        limb_w: 1,
        root_flare: 2,
    },
    // SP_ACACIA — a flat slab crown on a forked trunk.
    TreeShape {
        height: (7, 10),
        trunk_w: (1, 1),
        lean_max: 1,
        crown_rx: (3, 4),
        crown_ry: (1, 2),
        limb_len: (2, 3),
        limb_w: 1,
        root_flare: 1,
    },
    // SP_MANGROVE — kinked, on prop roots, with a drooping canopy.
    TreeShape {
        height: (6, 10),
        trunk_w: (1, 1),
        lean_max: 3,
        crown_rx: (3, 4),
        crown_ry: (2, 2),
        limb_len: (2, 3),
        limb_w: 1,
        root_flare: 0,
    },
    // SP_CACTUS — a column with risers. No crown, no flare.
    TreeShape {
        height: (4, 8),
        trunk_w: (2, 2),
        lean_max: 0,
        crown_rx: (0, 0),
        crown_ry: (0, 0),
        limb_len: (1, 2),
        limb_w: 2,
        root_flare: 0,
    },
    // SP_SNAG — a dead trunk with bare 45-degree limbs.
    TreeShape {
        height: (4, 9),
        trunk_w: (1, 1),
        lean_max: 2,
        crown_rx: (0, 0),
        crown_ry: (0, 0),
        limb_len: (1, 3),
        limb_w: 1,
        root_flare: 1,
    },
    // SP_SHROOM — a stem under a hard-edged cap.
    TreeShape {
        height: (3, 9),
        trunk_w: (1, 1),
        lean_max: 1,
        crown_rx: (2, 4),
        crown_ry: (1, 2),
        limb_len: (0, 0),
        limb_w: 1,
        root_flare: 0,
    },
    // SP_SHRUB — one stem, one blob.
    TreeShape {
        height: (1, 1),
        trunk_w: (1, 1),
        lean_max: 0,
        crown_rx: (1, 2),
        crown_ry: (1, 2),
        limb_len: (0, 0),
        limb_w: 1,
        root_flare: 0,
    },
];

/// The trunk's own parameters, bundled because there are now five of them and
/// only the first two vary between most call sites.
///
/// The old free-function form took eight positional arguments and its doc
/// argued that bundling would hide which call site varied what. That argument
/// dies at ten: every field is named at the literal that builds it, one line
/// each, which reads better than eight bare values in a row.
struct Trunk {
    code: CellId,
    /// Width at the root and at the crown, in WORLD cells. Tapers between them.
    base_w: i32,
    tip_w: i32,
    kink_dir: i32,
    kink_at: i32,
}

fn leaf_blob(
    ctx: &mut DecorContext<'_>,
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    code: CellId,
    variant: i32,
) {
    let ax = if rx < 1 { 1 } else { rx };
    let ay = if ry < 1 { 1 } else { ry };
    let salt = S_BLOB + variant * 137;
    for dy in -ay..=ay {
        let y = cy + dy;
        let ny = f64::from(dy) / f64::from(ay);
        let yy = ny * ny;
        for dx in -ax..=ax {
            let nx = f64::from(dx) / f64::from(ax);
            let d = nx * nx + yy;
            if d > 1.05 {
                continue;
            }
            if d > 0.42 && ctx.hash(cx + dx, y + salt) < (d - 0.42) * 0.95 {
                continue;
            }
            ctx.plot_if_empty(cx + dx, y, code);
        }
    }
}

/// A trunk `h` cells tall and `t.base_w` wide at the root, tapering to
/// `t.tip_w` at the crown, leaning `lean` cells over its full height with an
/// optional kink above `t.kink_at`. Returns the x of its topmost centre cell.
///
/// Diagonal steps are doubled up so a leaning trunk stays a connected column —
/// that rule predates the width and still applies, now per edge column rather
/// than to the single cell there used to be.
///
/// A `kink_at` of 0 disables the kink. Note the jungle passes a `kink_dir` of 0
/// WITH a live `kink_at`, so its kink adds nothing; that is what the TypeScript
/// did and it is left alone.
fn trunk_up(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32, h: i32, lean: i32, t: &Trunk) -> i32 {
    let mut prev = wcx;
    let mut x = wcx;
    for i in 1..=h {
        x = wcx
            + js_round(f64::from(i * lean) / f64::from(h)) as i32
            + if t.kink_at > 0 && i >= t.kink_at {
                t.kink_dir
            } else {
                0
            };
        // Width eases from base to tip over the height. A trunk that keeps its
        // root width all the way up reads as a pillar, which is what four times
        // the cells made the un-tapered version look like.
        let w = taper_w(t.base_w, t.tip_w, i, h);
        span(ctx, x, surf - i, w, t.code);
        if i > 1 && prev != x {
            // The doubled diagonal step, at full width.
            span(ctx, prev, surf - i, w, t.code);
        }
        prev = x;
    }
    x
}

/// Width at height `i` of `h`, eased from `base` to `tip`.
#[inline]
fn taper_w(base: i32, tip: i32, i: i32, h: i32) -> i32 {
    if h <= 1 {
        return base;
    }
    let t = f64::from(i - 1) / f64::from(h - 1);
    let w = f64::from(base) + (f64::from(tip) - f64::from(base)) * t;
    let w = js_round(w) as i32;
    if w < 1 { 1 } else { w }
}

/// A horizontal run of `w` cells centred on `x`.
///
/// Biased so an even width puts the extra cell on the RIGHT, consistently — an
/// alternating bias makes a tapering trunk wobble by a cell as it narrows.
#[inline]
fn span(ctx: &mut DecorContext<'_>, x: i32, y: i32, w: i32, code: CellId) {
    let left = (w - 1) / 2;
    for dx in 0..w {
        ctx.plot(x - left + dx, y, code);
    }
}

/// A root flare: the trunk widening where it meets the ground, `spread` cells
/// either side, tallest against the trunk and dying out to nothing.
///
/// Replaces two hand-plotted cells per species. At one authored cell of spread
/// that is roughly what it still draws; at four world cells it is a buttress.
///
/// Two rules, both learned the hard way when the first version had neither:
///
/// It grows UP the trunk (`surf - 1 - k`), not down into the ground. Downward
/// it is invisible — it is underground — and it was also filling the rows the
/// body walks on.
///
/// Its height is capped at [`STEP_UP_CELLS`], so a flare is always something the
/// body can stride over. Uncapped, a buttress on a big tree is a wall two body
/// heights tall across the only gap between two trunks, and `walkable_spawn`
/// reported exactly that: a spawn with zero walkable cells in EITHER direction.
/// A tree may be an obstacle; its roots may not.
fn root_flare(ctx: &mut DecorContext<'_>, x: i32, surf: i32, spread: i32, code: CellId) {
    for d in 1..=spread {
        let rise = (spread - d + 1).min(STEP_UP_CELLS);
        for k in 0..rise {
            ctx.plot(x - d, surf - 1 - k, code);
            ctx.plot(x + d, surf - 1 - k, code);
        }
    }
}

/// Trunk x at `i` cells above the ground — for hanging branches off a lean.
fn trunk_x_at(wcx: i32, i: i32, h: i32, lean: i32) -> i32 {
    wcx + js_round(f64::from(i * lean) / f64::from(h)) as i32
}

/// A limb from (x, y) running `dir` horizontally and `rise` cells up over `len`,
/// `w` cells thick, thinning to one cell at the tip.
///
/// The taper is what stops a thick limb reading as a plank: a branch is widest
/// where it leaves the trunk.
#[allow(clippy::too_many_arguments)]
fn limb(
    ctx: &mut DecorContext<'_>,
    x: i32,
    y: i32,
    dir: i32,
    len: i32,
    rise: i32,
    w: i32,
    code: CellId,
) {
    for i in 1..=len {
        let ly = y - js_round(f64::from(i * rise) / f64::from(len)) as i32;
        let t = taper_w(w, 1, i, len);
        for k in 0..t {
            ctx.plot(x + dir * i, ly + k, code);
        }
    }
}

/// Seed a vine. Vines grow downward on their own at runtime (see the `growth`
/// block on the material), so a two or three cell stub is enough to become a
/// proper hanging strand once the chunk is simulated.
fn hang_vine(ctx: &mut DecorContext<'_>, x: i32, y: i32, len: i32) {
    for i in 0..len {
        ctx.plot_if_empty(x, y + i, VINE);
    }
}

/// A lean in [-max, max], zero-biased so most trees stand roughly upright.
fn lean_of(ctx: &DecorContext<'_>, wcx: i32, max: i32) -> i32 {
    let r = ctx.hash(wcx, S_LEAN);
    let dir = if ctx.hash(wcx, S_LEAN + 1) < 0.5 {
        -1
    } else {
        1
    };
    let mut mag = 0;
    if r > 0.52 {
        mag = 1;
    }
    if r > 0.84 {
        mag = 2;
    }
    if r > 0.95 {
        mag = 3;
    }
    dir * if mag > max { max } else { mag }
}

// --- Species -----------------------------------------------------------------

/// Plains oak: fat rounded crown on a short trunk, lobed rather than circular.
fn draw_broadleaf(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32, b: Biome) {
    let sh = SHAPES[SP_BROADLEAF as usize];
    let h = a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height));
    let lean = a(ctx, lean_of(ctx, wcx, sh.lean_max));
    let leaf = canopy_of(ctx, b, wcx);
    let tx = trunk_up(
        ctx,
        wcx,
        surf,
        h,
        lean,
        &Trunk {
            code: WOOD,
            base_w: a1(ctx, sh.trunk_w.0),
            tip_w: a1(ctx, sh.trunk_w.1),
            kink_dir: 0,
            kink_at: 0,
        },
    );

    // The trunk meets the ground as a flare rather than a stick.
    if ctx.hash(wcx, S_ROOT) < 0.45 {
        root_flare(ctx, wcx, surf, a1(ctx, sh.root_flare), WOOD);
    }

    // A pair of limbs angling up into the crown.
    let branch_at = a(ctx, 3) + (ctx.hash(wcx, S_BR) * f64::from(h - a(ctx, 4))) as i32;
    let bx = trunk_x_at(wcx, branch_at, h, lean);
    let ll = a(ctx, TreeShape::pick(ctx, wcx, S_BR + 7, sh.limb_len));
    let lw = a1(ctx, sh.limb_w);
    if ctx.hash(wcx, S_BRD) < 0.7 {
        limb(ctx, bx, surf - branch_at, -1, ll, ll, lw, WOOD);
    }
    if ctx.hash(wcx, S_BRD + 1) < 0.7 {
        limb(ctx, bx, surf - branch_at, 1, ll, ll, lw, WOOD);
    }

    let rx = a(ctx, TreeShape::pick(ctx, wcx, S_RX, sh.crown_rx));
    let ry = a(ctx, TreeShape::pick(ctx, wcx, S_RY, sh.crown_ry));
    let cy = surf - h;
    leaf_blob(ctx, tx, cy, rx, ry, leaf, 0);
    // Side lobes break the circle so no two crowns read as the same stamp.
    let lobe = a1(ctx, 2);
    if ctx.hash(wcx, S_LOBE) < 0.72 {
        leaf_blob(ctx, tx - rx + lobe, cy + lobe, lobe, lobe, leaf, 1);
    }
    if ctx.hash(wcx, S_LOBE + 1) < 0.72 {
        leaf_blob(ctx, tx + rx - lobe, cy + lobe, lobe, lobe, leaf, 2);
    }
    if ctx.hash(wcx, S_LOBE + 2) < 0.5 {
        leaf_blob(ctx, tx, cy - ry + lobe, lobe, lobe, leaf, 3);
    }

    // Leaf litter at the foot, same colour as the crown above it.
    let litter = a(ctx, 2);
    for dx in -litter..=litter {
        if ctx.hash(wcx + dx, S_CKIND) < 0.28 {
            ctx.plot_if_empty(wcx + dx, surf - 1, leaf);
        }
    }
}

/// Tundra / glacier spruce: narrow, tiered, snow-loaded.
fn draw_conifer(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32, b: Biome) {
    let sh = SHAPES[SP_CONIFER as usize];
    let icy = b == Biome::Glacier;
    // Glaciers stunt them; tundra grows the tall narrow ones.
    let h = if icy {
        a(ctx, 6 + (ctx.hash(wcx, S_H) * 5.0) as i32)
    } else {
        a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height))
    };
    let top = surf - h;
    let trunk = &Trunk {
        code: WOOD,
        base_w: a1(ctx, sh.trunk_w.0),
        tip_w: a1(ctx, sh.trunk_w.1),
        kink_dir: 0,
        kink_at: 0,
    };
    trunk_up(ctx, wcx, surf, h, 0, trunk);
    root_flare(ctx, wcx, surf, a1(ctx, sh.root_flare), WOOD);

    let bare = a(ctx, 1 + (ctx.hash(wcx, S_RX) * 2.0) as i32); // bare ankle
    let taper = 0.26 + ctx.hash(wcx, S_RY) * 0.2;
    let snowy = icy || ctx.hash(wcx, S_PAL) < 0.5;
    // Spire: a point rather than a single cell, now that there are cells to spare.
    let spire = a1(ctx, 1);
    for k in 0..spire {
        ctx.plot_if_empty(wcx, top - 1 - k, LEAF_PINE);
    }

    // The tier pitch and the radius cap are lengths and scale; the taper is a
    // RATIO of radius to height and does not. Getting that backwards makes a
    // spruce either a needle or a cone with no boughs at all.
    let tier = a1(ctx, 3);
    let rmax = a1(ctx, 3);
    for y in top..=surf - 1 - bare {
        let d = y - top;
        let mut r = (f64::from(d) * taper).floor() as i32;
        let on_tier = d % tier == tier - 1;
        if on_tier {
            r += a1(ctx, 1); // tier flare — the sawtooth edge of a spruce
        }
        if r > rmax {
            r = rmax;
        }
        if r < 1 {
            ctx.plot_if_empty(wcx, y, LEAF_PINE);
            continue;
        }
        for dx in -r..=r {
            // Rim dither is per CELL, so at four times the resolution a bough
            // edge frays instead of stepping. This is the whole reason the
            // species draw at native rate rather than being upscaled.
            if dx.abs() > r - a1(ctx, 1) && ctx.hash(wcx + dx, y + S_BLOB) < 0.4 {
                continue;
            }
            ctx.plot_if_empty(wcx + dx, y, LEAF_PINE);
        }
        // Snow sits on the outer edge of a flare, where a real bough would hold it.
        if snowy && on_tier {
            for k in 0..a1(ctx, 1) {
                if ctx.hash(wcx - r + k, y + S_PAL) < 0.6 {
                    ctx.plot_if_empty(wcx - r + k, y - 1, SNOW);
                }
                if ctx.hash(wcx + r - k, y + S_PAL) < 0.6 {
                    ctx.plot_if_empty(wcx + r - k, y - 1, SNOW);
                }
            }
        }
    }
}

/// Jungle emergent: tall, buttressed, several tiers of limb, vines off all of it.
fn draw_jungle(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32) {
    let sh = SHAPES[SP_JUNGLE as usize];
    let lw = a1(ctx, sh.limb_w);
    let h = a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height));
    let lean = a(ctx, lean_of(ctx, wcx, sh.lean_max));
    let kink_at = 4 + (ctx.hash(wcx, S_RX + 1) * f64::from(h - 6)) as i32;
    let tx = trunk_up(
        ctx,
        wcx,
        surf,
        h,
        lean,
        &Trunk {
            code: WOOD,
            base_w: a1(ctx, sh.trunk_w.0),
            tip_w: a1(ctx, sh.trunk_w.1),
            kink_dir: 0,
            kink_at,
        },
    );

    // Buttress roots — the thing that makes a jungle trunk read as enormous.
    // The flare is the base of it; the two probabilistic wings widen it further
    // and are what stop every jungle trunk meeting the ground identically.
    root_flare(ctx, wcx, surf, a1(ctx, sh.root_flare), WOOD);
    if ctx.hash(wcx, S_ROOT + 2) < 0.4 {
        root_flare(ctx, wcx, surf, a1(ctx, sh.root_flare + 1), WOOD);
    }

    let limbs = 2 + (ctx.hash(wcx, S_NB) * 3.0) as i32; // 2..4
    for k in 0..limbs {
        let at = 5 + (ctx.hash(wcx, S_BR + k) * f64::from(h - 7)) as i32;
        let bx = trunk_x_at(wcx, at, h, lean);
        let by = surf - at;
        let dir = if ctx.hash(wcx, S_BRD + k) < 0.5 {
            -1
        } else {
            1
        };
        let len = a(ctx, TreeShape::pick(ctx, wcx, S_BRL + k, sh.limb_len));
        limb(ctx, bx, by, dir, len, 2, lw, WOOD);
        let tip_x = bx + dir * len;
        let tip_y = by - 2;
        let tip_r = a1(ctx, 2);
        leaf_blob(ctx, tip_x, tip_y, tip_r, tip_r, LEAF_JUNGLE, 4 + k);
        if ctx.hash(tip_x, S_VINE + k) < 0.55 {
            let vlen = 2 + (ctx.hash(tip_x, S_VINE) * 3.0) as i32;
            hang_vine(ctx, tip_x, tip_y + 3, vlen);
        }
    }

    let rx = a(ctx, TreeShape::pick(ctx, wcx, S_RX, sh.crown_rx));
    let ry = a(ctx, TreeShape::pick(ctx, wcx, S_RY, sh.crown_ry));
    let cy = surf - h + a1(ctx, 1);
    let lobe = a1(ctx, 2);
    leaf_blob(ctx, tx, cy, rx, ry, LEAF_JUNGLE, 0);
    leaf_blob(ctx, tx - rx + lobe, cy + lobe, lobe, lobe, LEAF_JUNGLE, 1);
    leaf_blob(ctx, tx + rx - lobe, cy + lobe, lobe, lobe, LEAF_JUNGLE, 2);

    // Curtain of vines off the underside of the crown.
    let mut dx = -rx;
    while dx <= rx {
        if ctx.hash(tx + dx, S_VINE + 5) < 0.4 {
            let vlen = 2 + (ctx.hash(tx + dx, S_VINE) * 3.0) as i32;
            hang_vine(ctx, tx + dx, cy + ry + 1, vlen);
        }
        dx += 2;
    }
}

/// Savanna acacia: long bare trunk, high fork, flat wind-shorn crown.
fn draw_acacia(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32, b: Biome) {
    let sh = SHAPES[SP_ACACIA as usize];
    let lw = a1(ctx, sh.limb_w);
    let h = a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height));
    let lean = a(ctx, lean_of(ctx, wcx, sh.lean_max));
    let leaf = canopy_of(ctx, b, wcx);
    let tx = trunk_up(
        ctx,
        wcx,
        surf,
        h,
        lean,
        &Trunk {
            code: WOOD,
            base_w: a1(ctx, sh.trunk_w.0),
            tip_w: a1(ctx, sh.trunk_w.1),
            kink_dir: 0,
            kink_at: 0,
        },
    );
    let fork_y = surf - h;

    let spread = a(ctx, TreeShape::pick(ctx, wcx, S_BRL, sh.limb_len));
    limb(ctx, tx, fork_y, -1, spread, 1, lw, WOOD);
    limb(ctx, tx, fork_y, 1, spread, 1, lw, WOOD);
    if ctx.hash(wcx, S_NB) < 0.4 {
        ctx.plot(tx, fork_y - 1, WOOD);
    }

    // The crown is a slab, not a ball: two cells thick in the middle, one at the
    // rim, which is the whole read of an acacia from a distance.
    let half = a(ctx, TreeShape::pick(ctx, wcx, S_RX, sh.crown_rx));
    let cy = fork_y - 2;
    for dx in -half..=half {
        let thick = if dx.abs() > half - 2 { 1 } else { 2 };
        for k in 0..thick {
            if ctx.hash(tx + dx, cy - k + S_BLOB) < 0.1 {
                continue;
            }
            ctx.plot_if_empty(tx + dx, cy - k, leaf);
        }
    }
}

/// Swamp mangrove: kinked leaning trunk on prop roots, droopy mossy canopy.
fn draw_mangrove(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32, b: Biome) {
    let sh = SHAPES[SP_MANGROVE as usize];
    let h = a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height));
    let lean = a(ctx, lean_of(ctx, wcx, sh.lean_max));
    let kink_at = 2 + (ctx.hash(wcx, S_RX + 1) * f64::from(h - 3)) as i32;
    let kink_dir = if lean >= 0 { 1 } else { -1 };
    let leaf = canopy_of(ctx, b, wcx);
    let tx = trunk_up(
        ctx,
        wcx,
        surf,
        h,
        lean,
        &Trunk {
            code: WOOD,
            base_w: a1(ctx, sh.trunk_w.0),
            tip_w: a1(ctx, sh.trunk_w.1),
            kink_dir,
            kink_at,
        },
    );

    // Prop roots: diagonal legs that walk down into the mud.
    for k in 0..3 {
        if ctx.hash(wcx, S_ROOT + k) > 0.62 {
            continue;
        }
        let dir = match k {
            0 => -1,
            1 => 1,
            _ => kink_dir,
        };
        let len = a(ctx, TreeShape::pick(ctx, wcx, S_ROOT + 8 + k, sh.limb_len));
        for i in 1..=len {
            ctx.plot(wcx + dir * i, surf - (len - i), WOOD);
        }
    }

    let rx = a(ctx, TreeShape::pick(ctx, wcx, S_RX, sh.crown_rx));
    let cy = surf - h;
    leaf_blob(
        ctx,
        tx,
        cy,
        rx,
        a(ctx, TreeShape::pick(ctx, wcx, S_RY, sh.crown_ry)),
        leaf,
        0,
    );
    // Droop: the canopy sags at both ends.
    let mut side = -1;
    while side <= 1 {
        let ex = tx + side * (rx - 1);
        let drop = 1 + (ctx.hash(ex, S_LOBE) * 3.0) as i32;
        for i in 1..=drop {
            ctx.plot_if_empty(ex, cy + 2 + i, leaf);
        }
        let vlen = 2 + (ctx.hash(ex, S_VINE) * 4.0) as i32;
        hang_vine(ctx, ex, cy + 3 + drop, vlen);
        side += 2;
    }
    // Moss on the ground under it — it spreads itself from here.
    ctx.plot(wcx, surf, MOSS);
}

/// Desert saguaro, or a squat barrel cactus.
fn draw_cactus(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32) {
    let sh = SHAPES[SP_CACTUS as usize];
    if ctx.hash(wcx, S_SPECIES + 1) < 0.34 {
        let w = i32::from(ctx.hash(wcx, S_RX) >= 0.5);
        let t = 1 + (ctx.hash(wcx, S_RY) * 2.0) as i32;
        for dx in -w..=w {
            for dy in 1..=t {
                ctx.plot(wcx + dx, surf - dy, CACTUS);
            }
        }
        return;
    }
    let h = a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height));
    for i in 1..=h {
        ctx.plot(wcx, surf - i, CACTUS);
    }
    let r_arm = ctx.hash(wcx, S_NB);
    let arms = if r_arm < 0.34 {
        0
    } else if r_arm < 0.78 {
        1
    } else {
        2
    };
    for k in 0..arms {
        let dir = if k == 0 {
            if ctx.hash(wcx, S_BRD) < 0.5 { -1 } else { 1 }
        } else if ctx.hash(wcx, S_BRD) < 0.5 {
            1
        } else {
            -1
        };
        let span = if h - 3 < 1 { 1 } else { h - 3 };
        let at = 2 + (ctx.hash(wcx, S_BR + k) * f64::from(span)) as i32;
        let out = 1 + i32::from(ctx.hash(wcx, S_BRL + k) >= 0.5); // 1..2
        for i in 1..=out {
            ctx.plot(wcx + dir * i, surf - at, CACTUS);
        }
        let rise = 2 + (ctx.hash(wcx, S_RY + k) * 2.0) as i32;
        for j in 1..=rise {
            ctx.plot(wcx + dir * out, surf - at - j, CACTUS);
        }
    }
}

/// A dead tree: bare forked trunk. Charred and ash-strewn in the volcanic.
fn draw_snag(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32, b: Biome) {
    let sh = SHAPES[SP_SNAG as usize];
    let lw = a1(ctx, sh.limb_w);
    let h = a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height));
    let lean = a(ctx, lean_of(ctx, wcx, sh.lean_max));
    trunk_up(
        ctx,
        wcx,
        surf,
        h,
        lean,
        &Trunk {
            code: WOOD,
            base_w: a1(ctx, sh.trunk_w.0),
            tip_w: a1(ctx, sh.trunk_w.1),
            kink_dir: 0,
            kink_at: 0,
        },
    );
    for k in 0..3 {
        if ctx.hash(wcx, S_BR + k) > 0.62 {
            continue;
        }
        let span = if h - 2 < 1 { 1 } else { h - 2 };
        let at = 2 + (ctx.hash(wcx, S_BR + 4 + k) * f64::from(span)) as i32;
        let dir = if ctx.hash(wcx, S_BRD + k) < 0.5 {
            -1
        } else {
            1
        };
        let len = 1 + (ctx.hash(wcx, S_BRL + k) * 3.0) as i32; // 1..3
        limb(
            ctx,
            trunk_x_at(wcx, at, h, lean),
            surf - at,
            dir,
            len,
            len,
            lw,
            WOOD,
        );
    }
    if b == Biome::Volcanic {
        for dx in -2..=2 {
            if ctx.hash(wcx + dx, S_CKIND) < 0.45 {
                ctx.plot_if_empty(wcx + dx, surf - 1, ASH);
            }
        }
    }
}

/// Giant mushroom: pale stem, glowing cap, gilled rim. Damp ground only.
fn draw_shroom(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32) {
    let sh = SHAPES[SP_SHROOM as usize];
    let h = a(ctx, TreeShape::pick(ctx, wcx, S_H, sh.height));
    let lean = a(ctx, lean_of(ctx, wcx, sh.lean_max));
    let tx = trunk_up(
        ctx,
        wcx,
        surf,
        h,
        lean,
        &Trunk {
            code: STEM,
            base_w: a1(ctx, sh.trunk_w.0),
            tip_w: a1(ctx, sh.trunk_w.1),
            kink_dir: 0,
            kink_at: 0,
        },
    );
    let r = a(ctx, TreeShape::pick(ctx, wcx, S_RX, sh.crown_rx));
    let cap_y = surf - h - 1;
    // The cap is a DOME, not a slab. It used to be two hard rows, which at four
    // times the width read as a dinner plate on a stick; the rows now scale with
    // the radius and the shoulder is rounded, so the silhouette is a mushroom at
    // any size.
    let rows = a1(ctx, TreeShape::pick(ctx, wcx, S_RY, sh.crown_ry) + 1);
    for dy in 0..rows {
        // Full width at the rim, drawing in toward the crown.
        let t = f64::from(dy) / f64::from(rows);
        let rr = r - (f64::from(r) * t * t) as i32;
        for dx in -rr..=rr {
            ctx.plot_if_empty(tx + dx, cap_y - dy, CAP);
        }
    }
    if r >= a1(ctx, 3) {
        // Gills under the overhanging rim, hanging the depth of the cap.
        for k in 0..a1(ctx, 1) {
            ctx.plot_if_empty(tx - r + k, cap_y + 1, STEM);
            ctx.plot_if_empty(tx + r - k, cap_y + 1, STEM);
        }
    }
    if ctx.hash(wcx, S_ROOT) < 0.5 {
        ctx.plot(wcx, surf, MOSS);
    }
}

/// Undergrowth bush — the cheap thing that stops a biome floor reading as bare.
fn draw_shrub(ctx: &mut DecorContext<'_>, wcx: i32, surf: i32, b: Biome) {
    let sh = SHAPES[SP_SHRUB as usize];
    let leaf = match b {
        Biome::Desert | Biome::Savanna => LEAF_AUTUMN,
        Biome::Volcanic => ASH,
        Biome::Tundra | Biome::Glacier => LEAF_PINE,
        Biome::Jungle => LEAF_JUNGLE,
        _ => canopy_of(ctx, b, wcx),
    };
    let r = a(ctx, TreeShape::pick(ctx, wcx, S_RX, sh.crown_rx));
    // A stub of stem so the bush is not floating.
    span(ctx, wcx, surf - 1, a1(ctx, sh.trunk_w.0), WOOD);
    leaf_blob(ctx, wcx, surf - 1 - r, r, r, leaf, 0);
}

// --- Passes ------------------------------------------------------------------

/// Consider one tree-candidate column and, if it takes, grow the plant.
fn grow_tree(ctx: &mut DecorContext<'_>, wcx: i32) {
    // Cheapest possible rejection first: one integer hash, no noise, no profile.
    // At a stride of 3 this throws away roughly half the candidates before any of
    // the expensive per-column climate work runs.
    let gate = ctx.hash(wcx, S_GATE);
    if gate >= MAX_TREE_DENSITY {
        return;
    }

    let col: ColumnProfile = ctx.profile_at(wcx);
    let t = col.surf_t;
    let fa = flora_of(col.surf_a);
    let fb = flora_of(col.surf_b);
    // Blending the two densities IS the treeline fade: a forest handing over to a
    // bare biome thins continuously across the blend band instead of ending on a
    // line, and two wooded biomes meeting stay wooded throughout.
    if gate >= fa.density * (1.0 - t) + fb.density * t {
        return;
    }

    // Which biome's flora this particular plant belongs to. Positional, so the
    // boundary band is a genuine mix of both species lists.
    let use_b = t > 0.0 && ctx.hash(wcx, S_MIX) < t;
    let biome = if use_b { col.surf_b } else { col.surf_a };
    let sp = species_for(if use_b { fb } else { fa }, ctx.hash(wcx, S_SPECIES));

    let surf = ctx.surface_at(wcx);
    let west = ctx.surface_at(wcx - 1);
    let east = ctx.surface_at(wcx + 1);
    let slope = (west - surf).abs() + (east - surf).abs();
    if slope > SLOPE_LIMIT[sp as usize] {
        return; // nothing roots on a cliff
    }

    // A share of trees stand in the BACKGROUND plane. Selected here, around the
    // whole draw, so a tree is wholly in one plane or the other — half a trunk in
    // front of its own canopy would read as a rendering fault, not as depth.
    let behind = ctx.hash(wcx, S_DEPTH) < BEHIND_SHARE;
    if behind {
        ctx.draw_behind();
    }
    match sp {
        SP_BROADLEAF => draw_broadleaf(ctx, wcx, surf, biome),
        SP_CONIFER => draw_conifer(ctx, wcx, surf, biome),
        SP_JUNGLE => draw_jungle(ctx, wcx, surf),
        SP_ACACIA => draw_acacia(ctx, wcx, surf, biome),
        SP_MANGROVE => draw_mangrove(ctx, wcx, surf, biome),
        SP_CACTUS => draw_cactus(ctx, wcx, surf),
        SP_SNAG => draw_snag(ctx, wcx, surf, biome),
        SP_SHROOM => draw_shroom(ctx, wcx, surf),
        _ => draw_shrub(ctx, wcx, surf, biome),
    }
    ctx.in_front();
}

/// Ground cover: one to three cells, but dense enough that the surface stops
/// reading as a bare line. Runs after the tree pass and only over air, so it
/// never eats a trunk.
fn grow_clutter(ctx: &mut DecorContext<'_>, wcx: i32) {
    let gate = ctx.hash(wcx, S_CLUT);
    if gate >= MAX_CLUTTER_DENSITY {
        return;
    }

    let col: ColumnProfile = ctx.profile_at(wcx);
    let t = col.surf_t;
    let fa = flora_of(col.surf_a);
    let fb = flora_of(col.surf_b);
    if gate >= fa.clutter * (1.0 - t) + fb.clutter * t {
        return;
    }

    let use_b = t > 0.0 && ctx.hash(wcx, S_CMIX) < t;
    let b = if use_b { col.surf_b } else { col.surf_a };
    let surf = ctx.surface_at(wcx);
    let r = ctx.hash(wcx, S_CKIND);

    match b {
        Biome::Jungle | Biome::Swamp => {
            if r < 0.34 {
                // Toadstool: two cells, and the cap glows.
                ctx.plot_if_empty(wcx, surf - 1, STEM);
                ctx.plot_if_empty(wcx, surf - 2, CAP);
            } else if r < 0.62 {
                ctx.plot(wcx, surf, MOSS);
                let leaf = if b == Biome::Jungle {
                    LEAF_JUNGLE
                } else {
                    LEAF_PINE
                };
                ctx.plot_if_empty(wcx, surf - 1, leaf);
            } else {
                ctx.plot(wcx, surf, MOSS);
                if r > 0.86 {
                    hang_vine(ctx, wcx, surf - 2, 2);
                }
            }
        }
        Biome::Desert => {
            if r < 0.45 {
                ctx.plot_if_empty(wcx, surf - 1, LEAF_AUTUMN); // dry brush
            } else if r < 0.6 {
                ctx.plot(wcx, surf - 1, CACTUS); // a nub of prickly pear
            }
        }
        Biome::Tundra | Biome::Glacier => {
            if r < 0.5 {
                ctx.plot_if_empty(wcx, surf - 1, LEAF_PINE);
            } else if r < 0.8 {
                ctx.plot_if_empty(wcx, surf - 1, SNOW);
            }
        }
        Biome::Volcanic => {
            if r < 0.6 {
                ctx.plot_if_empty(wcx, surf - 1, ASH);
            }
        }
        Biome::Savanna => {
            if r < 0.55 {
                ctx.plot(wcx, surf, MOSS);
            } else if r < 0.8 {
                ctx.plot_if_empty(wcx, surf - 1, LEAF_AUTUMN);
            }
        }
        _ => {
            // Plains and anything unlisted: grass tufts that creep outward at
            // runtime, with the odd scatter of fallen leaves.
            if r < 0.62 {
                ctx.plot(wcx, surf, MOSS);
            } else if r < 0.82 {
                let leaf = if autumn_band(ctx, wcx) {
                    LEAF_AUTUMN
                } else {
                    LEAF
                };
                ctx.plot_if_empty(wcx, surf - 1, leaf);
            }
        }
    }
}

/// Flora. Two positional passes — trees on a sparse lattice, ground cover on a
/// dense one — both scanning `reach` past the chunk edge so an overhanging crown
/// from the next chunk over still paints its share here.
pub struct TreeDecorator;

impl Decorator for TreeDecorator {
    fn name(&self) -> &'static str {
        "trees"
    }

    fn reach_x(&self) -> i32 {
        REACH_X * WorldScale::LIVE.raster()
    }

    fn reach_y(&self) -> i32 {
        REACH_Y * WorldScale::LIVE.raster()
    }

    fn decorate(&self, ctx: &mut DecorContext<'_>) {
        // Flora lives in a bounded band around the surface. Every underground chunk
        // in a streaming world would otherwise pay for a full candidate scan to
        // paint nothing at all.
        let (top_row, bot_row) = flora_band();
        if ctx.base_y + CHUNK_CELLS - 1 < top_row || ctx.base_y > bot_row {
            return;
        }
        let k = ctx.raster();
        // The stride SCALES with the trees, and getting this wrong is what made
        // the first 4x forest a wall: trunks were still planted every 3 cells
        // while each tree grew four times wider, so a woodland became a solid
        // fence of trunks with no gap a body could walk into.
        //
        // Scaled, the trees-per-landform ratio is the one the biome densities
        // were tuned against — a plains oak every dozen cells rather than every
        // three — and the gaps between them are walkable again.
        for wcx in origin_columns(ctx.base_x, REACH_X * k, TREE_STRIDE * k, TREE_PHASE) {
            grow_tree(ctx, wcx);
        }
        for wcx in origin_columns(
            ctx.base_x,
            CLUTTER_REACH * k,
            CLUTTER_STRIDE * k,
            CLUTTER_PHASE,
        ) {
            grow_clutter(ctx, wcx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorldScale;
    use crate::config::world::pmod;
    use crate::config::worldgen::SEED;
    use crate::sim::materials::EMPTY;
    use crate::sim::noise::Noise;
    use crate::sim::worldgen::heightmap::Heightmap;

    /// Paint one chunk at (base_x, base_y) into a fresh buffer.
    fn chunk(
        noise: &Noise,
        hm: &mut Heightmap,
        base_x: i32,
        base_y: i32,
        scale: WorldScale,
    ) -> (Vec<CellId>, Vec<CellId>) {
        let mut out = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let mut back = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        {
            // Both planes, because a share of trees stand behind the play plane
            // and the seam and reach contracts apply to them exactly as much:
            // a back-plane canopy clipped at a chunk edge is just as wrong, and
            // nothing else in the suite would ever look at it.
            let mut ctx = DecorContext::new(noise, SEED, base_x, base_y, &mut out, hm, scale)
                .with_back(&mut back);
            TreeDecorator.decorate(&mut ctx);
        }
        (out, back)
    }

    /// The chunk-aligned row a flora canvas should start at, so the ground line
    /// runs through the middle of it.
    ///
    /// The ground is at row ~35 at LEGACY and ~145 at LIVE, so a canvas pinned to
    /// y = 0 sees a forest at one scale and empty sky at the other. Deriving it
    /// is what lets the same contract test run at both — and the `painted > 200`
    /// floor at the end of the seam test is what would have caught a canvas
    /// looking at the wrong place, as it did when this was first written.
    fn band_y0(noise: &Noise, scale: WorldScale) -> i32 {
        let mut hm = Heightmap::new();
        let surf = hm.surface_row_at(noise, 64, None, scale);
        // Two chunks of headroom above the ground for the crowns.
        ((surf / CHUNK_CELLS) - 2) * CHUNK_CELLS
    }

    /// A `w` x `h` cell canvas anchored at (x0, y0), tiled from chunks whose
    /// top-left corners start at (x0, y0).
    ///
    /// Moving (x0, y0) moves the CHUNK GRID, not the world: the decorator is a
    /// pure function of absolute coordinates, so two canvases over the same
    /// region under different alignments must agree cell for cell.
    fn canvas(noise: &Noise, x0: i32, y0: i32, w: i32, h: i32, scale: WorldScale) -> Vec<CellId> {
        assert_eq!(w % CHUNK_CELLS, 0);
        assert_eq!(h % CHUNK_CELLS, 0);
        let mut hm = Heightmap::new();
        // Front and back merged, front winning, so one comparison covers both
        // planes. A tree that changed PLANE between two chunk alignments still
        // shows up, because it would have to move cells to do it.
        let mut buf = vec![EMPTY; (w * h) as usize];
        let mut by = y0;
        while by < y0 + h {
            let mut bx = x0;
            while bx < x0 + w {
                let (front, back) = chunk(noise, &mut hm, bx, by, scale);
                let c: Vec<CellId> = front
                    .iter()
                    .zip(back.iter())
                    .map(|(&f, &b)| if f != EMPTY { f } else { b })
                    .collect();
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
        // These ARE the forest. A typo in a salt is a different world, and one
        // plausible enough that nobody notices for a month.
        assert_eq!(
            [
                S_GATE, S_MIX, S_SPECIES, S_H, S_LEAN, S_RX, S_RY, S_NB, S_BR, S_BRD, S_BRL,
                S_ROOT, S_LOBE, S_PAL, S_VINE, S_BLOB, S_CLUT, S_CMIX, S_CKIND
            ],
            [
                101, 211, 307, 409, 503, 601, 701, 809, 907, 1009, 1103, 1201, 1301, 1409, 1511,
                1709, 1801, 1901, 2003
            ]
        );
        assert_eq!((REACH_X, REACH_Y, ROOT_DEPTH), (8, 26, 3));
        assert_eq!((TREE_STRIDE, TREE_PHASE), (3, 1));
        assert_eq!((CLUTTER_STRIDE, CLUTTER_PHASE, CLUTTER_REACH), (2, 0, 2));
        assert_eq!(MAX_TREE_DENSITY, 0.56);
        assert_eq!(MAX_CLUTTER_DENSITY, 0.62);
    }

    #[test]
    fn js_round_sends_halves_up_not_away_from_zero() {
        // A tree leaning left rounds negative halves. Rust's own `round` would
        // put those cells one column off from the TypeScript's.
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0); // f64::round would give -3
        assert_eq!(js_round(-2.6), -3.0);
        assert_eq!(js_round(-0.5), 0.0);
    }

    #[test]
    fn every_biome_grows_something_and_the_weights_partition() {
        // A biome whose weight list does not reach 1 drops columns on the floor;
        // one whose species are out of range indexes SLOPE_LIMIT out of bounds.
        for b in Biome::ALL {
            let f = flora_of(b);
            assert!(f.density > 0.0 && f.density <= MAX_TREE_DENSITY, "{b:?}");
            assert!(f.clutter > 0.0 && f.clutter <= MAX_CLUTTER_DENSITY, "{b:?}");
            let mut seen: Vec<u8> = vec![];
            for r in 0..1000 {
                let sp = species_for(f, f64::from(r) / 1000.0);
                assert!((sp as usize) < SLOPE_LIMIT.len(), "{b:?} species {sp}");
                if !seen.contains(&sp) {
                    seen.push(sp);
                }
            }
            assert_eq!(
                seen.len(),
                f.weights.len(),
                "{b:?} never picks some species"
            );
            // The last cut is exactly 1, so a value just under 1 always resolves.
            let last = f.weights[f.weights.len() - 1].0;
            assert_eq!(species_for(f, 1.0 - f64::EPSILON), last, "{b:?}");
        }
    }

    #[test]
    fn the_flora_band_is_the_typescripts_and_is_slightly_stale() {
        // The vertical guard is an optimisation that must never skip a chunk a
        // tree could reach into — and it very nearly does not.
        //
        // Its derivation says the heightmap is
        // `round(ANCHOR + heightOffset + fbm1 * ampScale * AMPLITUDE)`. It has
        // not been that for a while: `heightmap::compute_params` builds a
        // continental spline with an erosion-scaled relief term and a terrace
        // snap, and that reaches a little higher than the old fBm did. This is
        // ported as-is, because the TypeScript computes exactly these two rows
        // from exactly these constants and a "corrected" band would generate a
        // different world.
        //
        // What that costs, measured over 120k columns: the ground line spans
        // rows -21..85, comfortably inside the band, so no chunk holding a TRUNK
        // is ever skipped. About 0.8% of columns sit high enough that a 26-cell
        // crown would reach above FLORA_TOP_ROW, and on those the chunk holding
        // the topmost leaves is skipped and the crown is clipped. Two numbers to
        // watch: if the ground line itself escapes the band, whole trees start
        // vanishing, and if the clip rate climbs, the mountains are going bald.
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let (top, bot) = flora_band();
        // The TypeScript band was (-22, 93) in legacy rows. It is written against
        // the raster factor rather than as the world rows it happens to be, so
        // this keeps asserting the ORIGINAL claim — the band is still the one the
        // TypeScript computed — at whatever scale the world is drawn.
        let k = WorldScale::LIVE.raster();
        assert_eq!(
            (top, bot),
            (-22 * k, 93 * k),
            "the band is not the TypeScript's"
        );

        let mut clipped = 0;
        let mut n = 0;
        let mut wcx = -20_000;
        while wcx < 20_000 {
            let surf = hm.surface_row_at(&noise, wcx, None, WorldScale::LEGACY);
            assert!(surf >= top, "the ground line at {wcx} is above the band");
            assert!(
                surf + ROOT_DEPTH <= bot,
                "a root at {wcx} sinks below the band"
            );
            if surf - REACH_Y < top {
                clipped += 1;
            }
            n += 1;
            wcx += 7;
        }
        let rate = f64::from(clipped) / f64::from(n);
        assert!(
            rate < 0.02,
            "{:.1}% of columns can have their crown clipped by the vertical guard",
            rate * 100.0
        );
    }

    #[test]
    fn density_stays_inside_the_declared_ceiling() {
        // Both gates are declared upper bounds the pass leans on to reject before
        // touching noise. Measure the realised rate over a long sweep: it must sit
        // under the ceiling, and must not be zero (a degenerate hash would
        // silently empty the world).
        let noise = Noise::new(SEED);
        let mut hm = Heightmap::new();
        let mut out = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
        let ctx = DecorContext::new(&noise, SEED, 0, 32, &mut out, &mut hm, WorldScale::LEGACY);

        let mut tree_candidates = 0;
        let mut trees = 0;
        let mut clutter_candidates = 0;
        let mut clutter = 0;
        let mut wcx = -30_000;
        while wcx < 30_000 {
            if pmod(wcx, TREE_STRIDE) == pmod(TREE_PHASE, TREE_STRIDE) {
                tree_candidates += 1;
                let gate = ctx.hash(wcx, S_GATE);
                let col = ctx.profile_at(wcx);
                let t = col.surf_t;
                let d = flora_of(col.surf_a).density * (1.0 - t) + flora_of(col.surf_b).density * t;
                assert!(
                    d <= MAX_TREE_DENSITY,
                    "blended density {d} exceeds the gate"
                );
                if gate < d {
                    trees += 1;
                }
            }
            if pmod(wcx, CLUTTER_STRIDE) == pmod(CLUTTER_PHASE, CLUTTER_STRIDE) {
                clutter_candidates += 1;
                let gate = ctx.hash(wcx, S_CLUT);
                let col = ctx.profile_at(wcx);
                let t = col.surf_t;
                let c = flora_of(col.surf_a).clutter * (1.0 - t) + flora_of(col.surf_b).clutter * t;
                assert!(
                    c <= MAX_CLUTTER_DENSITY,
                    "blended clutter {c} exceeds the gate"
                );
                if gate < c {
                    clutter += 1;
                }
            }
            wcx += 1;
        }
        let tree_rate = f64::from(trees) / f64::from(tree_candidates);
        let clutter_rate = f64::from(clutter) / f64::from(clutter_candidates);
        assert!(
            (0.05..MAX_TREE_DENSITY).contains(&tree_rate),
            "trees grow on {:.1}% of candidate columns",
            tree_rate * 100.0
        );
        assert!(
            (0.05..MAX_CLUTTER_DENSITY).contains(&clutter_rate),
            "clutter grows on {:.1}% of candidate columns",
            clutter_rate * 100.0
        );
    }

    #[test]
    fn a_tree_straddling_a_seam_is_painted_identically_from_both_sides() {
        seam_agrees_at(WorldScale::LEGACY);
    }

    /// The same contract at the scale the GAME runs at.
    ///
    /// Worth its own test rather than a loop inside the one above: for the whole
    /// life of these tests every canvas ran at LEGACY, where the raster factor is
    /// 1 and the scaled drawing path is a no-op — so the path the player actually
    /// sees had never been exercised by anything. A guard that only ever runs at
    /// the identity is not guarding the shipping code.
    #[test]
    fn a_tree_straddling_a_seam_agrees_at_the_live_scale() {
        seam_agrees_at(WorldScale::LIVE);
    }

    fn seam_agrees_at(scale: WorldScale) {
        // THE contract test. Generate the same world region twice under two
        // different chunk alignments; every cell they share must agree. A tree
        // whose origin falls outside one alignment's scan window shows up here as
        // a canopy clipped on one side and whole on the other.
        let noise = Noise::new(SEED);
        let y0 = band_y0(&noise, scale);
        // Aligned grid, 128x128 centred on the ground line.
        let a = canvas(&noise, 0, y0, 128, 128, scale);
        // Same world, chunk grid shifted by half a chunk on both axes.
        let b = canvas(&noise, 16, y0 + 16, 96, 96, scale);

        let mut painted = 0;
        for y in 16..112 {
            for x in 16..112 {
                let va = a[(y * 128 + x) as usize];
                let vb = b[((y - 16) * 96 + (x - 16)) as usize];
                assert_eq!(va, vb, "chunk alignments disagree at ({x}, {y})");
                if va != EMPTY {
                    painted += 1;
                }
            }
        }
        assert!(
            painted > 200,
            "only {painted} cells of flora in 96x96 — the test proved nothing"
        );
    }

    #[test]
    fn nothing_is_painted_outside_the_declared_reach() {
        reach_holds_at(WorldScale::LEGACY);
    }

    /// The reach promise at the scale the game runs at. See
    /// [`a_tree_straddling_a_seam_agrees_at_the_live_scale`] for why this is a
    /// separate test and not a loop.
    #[test]
    fn the_declared_reach_holds_at_the_live_scale() {
        reach_holds_at(WorldScale::LIVE);
    }

    fn reach_holds_at(scale: WorldScale) {
        // Reach is the promise the generator scans against. Paint a canvas from
        // aligned chunks, then re-paint it from chunks offset by one cell at a
        // time: if any decoration extended further than REACH_X from its origin
        // column, some offset would clip it and the canvases would part company.
        let noise = Noise::new(SEED);
        let y0 = band_y0(&noise, scale);
        let base = canvas(&noise, 0, y0, 128, 128, scale);
        for shift in [1, 5, 8, 13, 31] {
            let s = canvas(&noise, shift, y0, 96, 128, scale);
            for y in 32..96 {
                for x in shift..(shift + 96) {
                    assert_eq!(
                        base[(y * 128 + x) as usize],
                        s[(y * 96 + (x - shift)) as usize],
                        "shift {shift} disagrees at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn the_decorator_declares_itself() {
        assert_eq!(TreeDecorator.name(), "trees");
        // The declared reach is the AUTHORED reach times the raster factor: a
        // tree drawn k times bigger overhangs k times further, and a decorator
        // that declared less than it paints grows seams at chunk boundaries.
        let k = WorldScale::LIVE.raster();
        assert_eq!(TreeDecorator.reach_x(), REACH_X * k);
        assert_eq!(TreeDecorator.reach_y(), REACH_Y * k);
    }
}

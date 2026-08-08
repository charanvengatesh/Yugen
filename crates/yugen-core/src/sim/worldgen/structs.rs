//! Template structures — the facade over `yugen-data`'s compiled struct
//! tables and the pass that stamps them.
//!
//! The generated module is authored glyph art plus placement metadata. Turning
//! that into something a per-chunk loop can stamp costs a string scan per cell,
//! which is why it happens exactly once, HERE, at first use: every template
//! becomes a flat array of slot indices plus three small parallel tables, and
//! stamping a 21x15 pyramid is then an indexed read and a `plot` per cell with
//! no string work at all. FORMAT.md §5's rule that generated modules are reached
//! only through a facade is what makes that possible without every caller
//! paying.
//!
//! CHUNK INDEPENDENCE (see [`crate::sim::decor`] — the rule is not optional). A
//! template is placed from a POSITIONAL ORIGIN and every chunk it can touch
//! recomputes the whole placement independently:
//!
//!   - which lattice cell is a candidate      — position only
//!   - whether it survives the density gate   — ctx.hash(origin)
//!   - which template it draws                — ctx.hash(origin), weighted
//!   - how tall the repeated section is       — ctx.hash(origin)
//!   - whether it is mirrored                 — ctx.hash(origin)
//!   - where its top-left corner lands        — origin + anchor + surface_at(origin)
//!
//! Nothing consults a neighbour, nothing is carried between calls, and `plot`
//! silently discards everything outside the chunk being generated. A 21-wide
//! pyramid spanning three chunk columns is therefore derived three times,
//! identically, and the three shares line up exactly.
//!
//! ## What changed on the way over from TypeScript
//!
//! Four module-level mutable scratch bindings (`ELIGIBLE`, `MARK_CTX`, `SITE`
//! and the site pool) are gone. Each was safe there only because JavaScript is
//! single-threaded; chunk generation here runs on a rayon pool, so every one of
//! them is now a stack local or a returned value. The only statics left are
//! [`LazyLock`] tables that are written once and never again.

use std::sync::LazyLock;

use crate::config::{
    CAVERN_DEPTH, CHUNK_CELLS, SEA_LEVEL_Y, SURFACE_AMPLITUDE, SURFACE_ANCHOR_Y, UNDERWORLD_DEPTH,
    pmod,
};
use crate::sim::biomes::{Biome, ColumnProfile, column_profile_at};
use crate::sim::decor::{DecorContext, Lattice, origin_cells, origin_columns};
use crate::sim::materials::{BLOCKS, CellId, code_of};
use crate::sim::noise::Noise;
use crate::sim::worldgen::containers::container_code;
use crate::sim::worldgen::heightmap::{Heightmap, shore_weight_at};

// --- The content facade ------------------------------------------------------
// Hand-written code reaches the compiled struct tables through here and never
// through `yugen_data::structs` directly, exactly as `sim::materials` is the
// one door to the block tables. That indirection is the seam that lets the
// generated module change shape without touching a call site.

pub use yugen_data::structs::{
    NEVER, STRUCT_COUNT, STRUCT_FLAGS, STRUCT_H, STRUCT_IDS, STRUCT_MAXH, STRUCT_PLACE,
    STRUCT_REP_FROM, STRUCT_REP_MAX, STRUCT_REP_MIN, STRUCT_REP_ROWS, STRUCT_W, STRUCTS, Sband,
    Sflag, StructBands, StructDef,
};

// --- Codes mirrored from crates/contentc/src/schemas/structure.rs -------------
// The schema maps these enums to numbers so the placement loop compares
// integers. The values are load-bearing on both sides; they are written out here
// rather than imported because `contentc` is the compiler and must not be
// dragged into the game.

/// Where in the world a template may stand.
///
/// The TypeScript named only the four codes it compared against and left 3 and 4
/// as bare shifts in a bitmask; a real enum costs nothing here and makes
/// `(1 << 3) | (1 << 4)` legible as "underground or cavern".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Place {
    Surface = 0,
    Shore = 1,
    Floating = 2,
    Underground = 3,
    Cavern = 4,
    Underworld = 5,
}

impl Place {
    /// The code the compiled `STRUCT_PLACE` table holds.
    ///
    /// Total: an unknown code reads as [`Place::Surface`], which is the schema's
    /// own default and the one class that cannot place a template underground.
    #[inline]
    fn from_code(c: u8) -> Place {
        match c {
            1 => Place::Shore,
            2 => Place::Floating,
            3 => Place::Underground,
            4 => Place::Cavern,
            5 => Place::Underworld,
            _ => Place::Surface,
        }
    }

    /// This class's bit in a placement mask.
    #[inline]
    fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// Which cell of the template the placement origin refers to. `bottom_center` is
/// the default because almost everything is authored standing on the ground and
/// the ground line is the one coordinate placement actually derives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Anchor {
    BottomCenter = 0,
    BottomLeft = 1,
    Center = 2,
    TopCenter = 3,
    TopLeft = 4,
}

impl Anchor {
    /// The code the compiled `STRUCT_ANCHOR` table holds. Total, defaulting to
    /// the schema's own default.
    #[inline]
    fn from_code(c: u8) -> Anchor {
        match c {
            1 => Anchor::BottomLeft,
            2 => Anchor::Center,
            3 => Anchor::TopCenter,
            4 => Anchor::TopLeft,
            _ => Anchor::BottomCenter,
        }
    }
}

/// What a later pass should do with this cell. `None` is just masonry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Mark {
    None = 0,
    Loot = 1,
    Spawn = 2,
}

impl Mark {
    /// The code a compiled `StructGlyph` holds. Total, defaulting to "just
    /// masonry" — an unrecognised mark leaves the authored cell as authored.
    #[inline]
    fn from_code(c: u8) -> Mark {
        match c {
            1 => Mark::Loot,
            2 => Mark::Spawn,
            _ => Mark::None,
        }
    }
}

// --- Glyph slots -------------------------------------------------------------

/// A glyph's index into a template's parallel slot tables.
///
/// Slot 0 and 1 are the two reserved glyphs, so the stamp loop can branch on a
/// single integer instead of re-testing characters. This is a newtype rather
/// than an enum because the set is open by construction: every legend row a
/// template declares takes the next slot, so only the two reserved values can be
/// named.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot(u8);

impl Slot {
    /// Leave whatever is already in the cell. `_`, a space, and past the end of
    /// a short row.
    pub const KEEP: Slot = Slot(0);
    /// Authored open air. `.`.
    pub const AIR: Slot = Slot(1);

    #[inline]
    fn index(self) -> usize {
        self.0 as usize
    }
}

const AIR: CellId = 0;

/// Does the block registry have this id?
fn present(id: &str) -> bool {
    BLOCKS.iter().any(|b| b.id == id)
}

/// The blocks the mark pass writes, and whether it has anything to write.
///
/// The TypeScript resolved these at module load into three consts; they depend
/// on the block registry, so they cannot be `const` here either.
struct Marks {
    /// Block per mark kind, indexed by the [`Mark`] discriminant. Slot 0
    /// ([`Mark::None`]) is air and is never read; an air entry means "leave
    /// whatever the stamp already wrote", which is how the pass stays inert
    /// while the content half is missing.
    block: [CellId; 3],
    /// True when at least one mark kind has a block to place. The pass's hard
    /// gate.
    active: bool,
}

static MARKS: LazyLock<Marks> = LazyLock::new(|| {
    // The block a `mark=spawn` glyph becomes, or air.
    //
    // A mark can only ever become a BLOCK. Chunk generation's whole output is an
    // array of cell codes — there is no side channel for "and put a mob here",
    // and inventing one would mean worldgen handing state to the mob system,
    // which spawns from live population pressure around the player and owns that
    // decision entirely. So a den mark is honoured the only way it can be: it
    // becomes a spawner block if content has one, and otherwise leaves the
    // authored air exactly as it is today. One guarded lookup at load, nothing at
    // all per chunk.
    let spawner_code: CellId = if present("spawner") {
        code_of("spawner")
    } else {
        AIR
    };
    let container = container_code();
    Marks {
        block: [AIR, container, spawner_code],
        active: container != AIR || spawner_code != AIR,
    }
});

/// Resolve a legend row's block id. The compiler already validated it against
/// the block registry (that is the whole point of `ref?(block)`), so a miss here
/// means the generated module and the material registry were built from
/// different content — worth failing loudly at load rather than painting a hole.
fn legend_code(struct_id: &str, glyph: &str, id: Option<&str>) -> CellId {
    let Some(id) = id else { return AIR };
    assert!(
        present(id),
        "structs: '{struct_id}' glyph '{glyph}' names unknown block '{id}'"
    );
    code_of(id)
}

/// One template, flattened for stamping. `rows` is `h * w` slot indices in
/// reading order; short authored rows are padded with `keep`, which is why a
/// template can be authored ragged without the stamper caring.
#[derive(Clone, Debug)]
pub struct Template {
    pub id: &'static str,
    /// The code the compiled tables index by. Kept so a caller that recovered a
    /// template from a query can name it back to content.
    pub code: u16,
    pub place: Place,
    pub anchor: Anchor,
    pub w: i32,
    pub h: i32,
    /// Height with the repeatable slice fully expanded — the vertical extent.
    pub max_h: i32,
    pub sink: i32,
    pub clearance: i32,
    pub flatness: i32,
    pub mirror: bool,
    pub rarity: f64,
    pub weight: f64,
    pub bands: Sband,
    pub min_depth: i32,
    pub max_depth: i32,
    /// `None` = any biome.
    pub biomes: Option<Vec<Biome>>,
    rows: Vec<Slot>,
    slot_code: Vec<CellId>,
    slot_mark: Vec<Mark>,
    /// Does any slot carry a mark at all? Precomputed so the mark pass is one
    /// boolean test for a template that has nothing for it to do, instead of a
    /// second scan over the body to discover that.
    pub has_mark: bool,
    slot_soft: Vec<bool>,
    rep_from: i32,
    rep_rows: i32,
    rep_min: i32,
    rep_max: i32,
}

/// Content names biomes as plain strings (biomes are hand-written, so a forward
/// `ref` that fails the build would be worse than one that does not). An id the
/// palette does not have simply never matches, exactly as a `Set.has` miss did.
fn biome_by_id(id: &str) -> Option<Biome> {
    Biome::ALL.into_iter().find(|b| b.def().id == id)
}

fn build(def: &'static StructDef) -> Option<Template> {
    let body = def.body;
    // An empty body is a tombstone (FORMAT.md §4): its code stays reserved so old
    // saves resolve, but there is nothing to stamp.
    if body.is_empty() || def.weight <= 0.0 {
        return None;
    }

    // The TypeScript measured rows in UTF-16 code units. Glyph art is ASCII by
    // construction (the schema reserves `.`, `_` and space and legends declare
    // single characters), and asserting it here is what lets the stamper index
    // bytes and stay in lockstep with that measurement.
    for row in body {
        assert!(
            row.is_ascii(),
            "structs: '{}' body row is not ASCII: {row:?}",
            def.id
        );
    }

    let mut w = 0usize;
    for row in body {
        if row.len() > w {
            w = row.len();
        }
    }
    let h = body.len();

    // Byte-indexed rather than a map: the key is one ASCII character and this is
    // read `w * h` times per template.
    let mut slot_of = [Slot::KEEP; 256];
    slot_of[b'_' as usize] = Slot::KEEP;
    slot_of[b' ' as usize] = Slot::KEEP;
    slot_of[b'.' as usize] = Slot::AIR;

    // Slot 0's code is never read — the stamp loop skips `KEEP` before it looks
    // one up. The TypeScript stored -1 there as a tripwire; air is the same
    // thing said in a `u16`.
    let mut slot_code: Vec<CellId> = vec![AIR, AIR];
    let mut slot_mark: Vec<Mark> = vec![Mark::None, Mark::None];
    let mut slot_soft: Vec<bool> = vec![false, false];
    for g in def.legend {
        assert!(
            g.c.len() == 1 && g.c.is_ascii(),
            "structs: '{}' legend glyph {:?} is not a single ASCII character",
            def.id,
            g.c
        );
        slot_of[g.c.as_bytes()[0] as usize] = Slot(slot_code.len() as u8);
        slot_code.push(legend_code(def.id, g.c, g.block));
        slot_mark.push(Mark::from_code(g.mark));
        slot_soft.push(g.soft);
    }

    let mut rows = vec![Slot::KEEP; w * h];
    for (r, line) in body.iter().enumerate() {
        let bytes = line.as_bytes();
        for c in 0..w {
            // Past the end of a short row is `keep`, which is also what a space
            // means, so trailing whitespace in the source cannot change the shape.
            rows[r * w + c] = if c < bytes.len() {
                slot_of[bytes[c] as usize]
            } else {
                Slot::KEEP
            };
        }
    }

    let rep_rows = def.repeat.map_or(0, |r| r.rows);
    let rep_max = def.repeat.map_or(1, |r| r.times[1] as i32);

    let has_mark = slot_mark.iter().any(|m| *m != Mark::None);
    Some(Template {
        id: def.id,
        code: def.code,
        place: Place::from_code(def.place),
        anchor: Anchor::from_code(def.anchor),
        w: w as i32,
        h: h as i32,
        max_h: h as i32 + rep_rows * (rep_max - 1),
        sink: def.sink,
        clearance: def.clearance,
        flatness: def.flatness,
        mirror: Sflag::from_bits_truncate(u32::from(STRUCT_FLAGS[def.code as usize]))
            .contains(Sflag::MIRROR),
        rarity: f64::from(def.rarity),
        weight: f64::from(def.weight),
        bands: band_bits(def.bands),
        min_depth: def.min_depth,
        max_depth: def.max_depth,
        biomes: def
            .biomes
            .map(|ids| ids.iter().filter_map(|id| biome_by_id(id)).collect()),
        rows,
        slot_code,
        slot_mark,
        has_mark,
        slot_soft,
        rep_from: def.repeat.map_or(0, |r| r.from),
        rep_rows,
        rep_min: def.repeat.map_or(1, |r| r.times[0] as i32),
        rep_max,
    })
}

fn band_bits(bands: &[StructBands]) -> Sband {
    let mut bits = Sband::empty();
    for b in bands {
        bits |= match b {
            StructBands::Surface => Sband::SURFACE,
            StructBands::Shallow => Sband::SHALLOW,
            StructBands::Cavern => Sband::CAVERN,
            StructBands::Deep => Sband::DEEP,
            StructBands::Underworld => Sband::UNDERWORLD,
        };
    }
    bits
}

/// Everything the pass derives from content, built once and never mutated.
///
/// The TypeScript had these as seven module-level `const`s. They are bundled
/// here because they are derived from each other and because a single
/// [`LazyLock`] is one initialisation barrier instead of seven.
pub struct Templates {
    /// Two lattices, because the two derive their row differently: a surface
    /// template gets its row from `surface_at(origin column)` and therefore
    /// needs no vertical lattice at all, while a buried one needs a real 2D grid
    /// gated on depth.
    column: Vec<Template>,
    lattice: Vec<Template>,
    reach_x: i32,
    reach_y: i32,
    col_band_top: i32,
    col_band_bot: i32,
    sub_band_top: i32,
}

// --- Declared extent ---------------------------------------------------------
// Honest maxima over what was actually authored, so adding a bigger template
// widens the decorator's reach automatically instead of silently clipping.
fn max_over(list: &[Template], f: impl Fn(&Template) -> i32) -> i32 {
    let mut m = 0;
    for t in list {
        let v = f(t);
        if v > m {
            m = v;
        }
    }
    m
}

fn half_w(t: &Template) -> i32 {
    (t.w >> 1) + 1
}

static TEMPLATES: LazyLock<Templates> = LazyLock::new(|| {
    let all: Vec<Template> = STRUCTS.iter().filter_map(build).collect();
    let column: Vec<Template> = all
        .iter()
        .filter(|t| (t.place as u8) <= Place::Floating as u8)
        .cloned()
        .collect();
    let lattice: Vec<Template> = all
        .iter()
        .filter(|t| (t.place as u8) > Place::Floating as u8)
        .cloned()
        .collect();

    let reach_x = max_over(&column, half_w).max(max_over(&lattice, half_w));
    // Rows a column-placed template may occupy above its column's ground line.
    let column_up = max_over(&column, |t| t.max_h + t.clearance);
    // Rows it may occupy below it — `sink` pushes a ruin into the ground.
    let column_down = max_over(&column, |t| t.sink + 2);
    let reach_y = column_up
        .max(column_down)
        .max(max_over(&lattice, |t| t.max_h));

    // Rows the ground line can possibly occupy, with the same slack
    // decor/structures.rs derives (largest biome amp_scale is 1.45, largest
    // |height_offset| is 6). A chunk outside the band plus the templates' own
    // vertical extent cannot contain one, which is what keeps every sky and deep
    // chunk in the world from paying for the column scan.
    let surf_span = f64::from(SURFACE_AMPLITUDE) * 1.6 + 8.0;
    let anchor = f64::from(SURFACE_ANCHOR_Y);
    let surf_min = (anchor - surf_span).floor() as i32;
    let surf_max = (anchor + surf_span).ceil() as i32;

    Templates {
        column,
        lattice,
        reach_x,
        reach_y,
        col_band_top: surf_min - column_up,
        col_band_bot: surf_max + column_down,
        sub_band_top: surf_min + SUB_MIN_DEPTH - reach_y,
    }
});

/// The compiled templates and everything derived from them.
///
/// Initialised on first call and immutable thereafter, so it is safe to reach
/// from a rayon worker: this is the Rust spelling of "TypeScript module load",
/// not shared mutable state.
#[inline]
pub fn templates() -> &'static Templates {
    &TEMPLATES
}

/// Farthest a template may extend horizontally from its origin, in cells.
#[inline]
pub fn struct_reach_x() -> i32 {
    TEMPLATES.reach_x
}

/// Farthest a template may extend vertically from its origin, in cells.
#[inline]
pub fn struct_reach_y() -> i32 {
    TEMPLATES.reach_y
}

// --- Lattices ----------------------------------------------------------------
// Deliberately out of phase with the lattices in decor/structures.rs (96/41 and
// 72x56 at 17,23): two landmark passes that shared an origin grid would build on
// top of each other every time both fired.
const COL_STRIDE: i32 = 112;
const COL_PHASE: i32 = 53;
const COL_DENSITY: f64 = 0.5;

const SUB_STRIDE_X: i32 = 88;
const SUB_STRIDE_Y: i32 = 64;
const SUB_PHASE_X: i32 = 29;
const SUB_PHASE_Y: i32 = 37;
const SUB_DENSITY: f64 = 0.24;
/// Nothing is buried shallower than this — a "buried" ruin in the topsoil is not.
const SUB_MIN_DEPTH: i32 = 40;

// --- Deterministic helpers ---------------------------------------------------

/// `Math.round`, JavaScript's way: halves break toward +infinity.
///
/// Rust's `f64::round` breaks away from zero, so `-2.5` rounds to `-3` there and
/// to `-2` in the original. Both `site_spread` here and the generators in
/// `features.rs` round negative halves, so the difference is a real one-cell
/// divergence rather than a theoretical one.
#[inline]
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// Salted positional hash — the only source of randomness a template may use.
#[inline]
fn h<Q: SiteQuery + ?Sized>(q: &Q, x: i32, y: i32, salt: i32) -> f64 {
    q.hash(x + salt * 6151, y - salt * 88651)
}

fn overlaps(ctx: &DecorContext<'_>, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    x1 >= ctx.base_x
        && x0 < ctx.base_x + CHUNK_CELLS
        && y1 >= ctx.base_y
        && y0 < ctx.base_y + CHUNK_CELLS
}

/// Depth band bit for a depth below the local surface. Matches the band
/// vocabulary in the struct schema and the depth constants the rock selection in
/// layers.rs already bands against, so `bands cavern` on a template means the
/// same slice of world the cavern ROCK occupies.
fn band_at(depth: i32) -> Sband {
    if depth < 8 {
        Sband::SURFACE
    } else if depth < 40 {
        Sband::SHALLOW
    } else if depth < CAVERN_DEPTH {
        Sband::CAVERN
    } else if depth < UNDERWORLD_DEPTH {
        Sband::DEEP
    } else {
        Sband::UNDERWORLD
    }
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

// --- What resolution is allowed to know --------------------------------------

/// The read-only half of [`DecorContext`] that site resolution actually uses.
///
/// The TypeScript's loot module fabricated a whole `DecorContext` with no-op
/// `plot`s and a `peek` that returned air, because the only thing it needed —
/// `markAt` — took one. That fake is not expressible here (a real
/// [`DecorContext`] borrows the chunk it is painting), and it was never the
/// honest shape anyway: resolution genuinely reads three things and writes
/// nothing. So the three are a trait, [`DecorContext`] implements it, and
/// [`StructQuery`] below is the standalone implementation a query uses.
pub trait SiteQuery {
    /// Surface row for an absolute column. `&mut` because the implementations
    /// memoise; it is a read.
    fn surface_at(&mut self, wcx: i32) -> i32;
    /// Full climate/biome/layer profile for an absolute column.
    fn profile_at(&self, wcx: i32) -> ColumnProfile;
    /// Stable hash in [0,1) from any two integers.
    fn hash(&self, x: i32, y: i32) -> f64;
}

impl SiteQuery for DecorContext<'_> {
    #[inline]
    fn surface_at(&mut self, wcx: i32) -> i32 {
        DecorContext::surface_at(self, wcx)
    }
    #[inline]
    fn profile_at(&self, wcx: i32) -> ColumnProfile {
        DecorContext::profile_at(self, wcx)
    }
    #[inline]
    fn hash(&self, x: i32, y: i32) -> f64 {
        DecorContext::hash(self, x, y)
    }
}

/// A [`SiteQuery`] with no chunk behind it — for asking where a template stands
/// long after the chunk that held it was evicted.
///
/// Owns its `Noise` and its own heightmap memo, so it is `Send` and a caller may
/// keep one per thread. Build it once per seed: `Noise::new` fills a permutation
/// table, and a chest break must not pay for that.
pub struct StructQuery {
    noise: Noise,
    heightmap: Heightmap,
}

impl StructQuery {
    /// A query context for one world seed.
    pub fn new(seed: u32) -> StructQuery {
        StructQuery {
            noise: Noise::new(seed),
            heightmap: Heightmap::new(),
        }
    }

    /// The noise this query resolves against.
    #[inline]
    pub fn noise(&self) -> &Noise {
        &self.noise
    }
}

impl SiteQuery for StructQuery {
    #[inline]
    fn surface_at(&mut self, wcx: i32) -> i32 {
        // `None` for the profile: let the memo do its job rather than paying for
        // a profile this caller does not have.
        self.heightmap.surface_row_at(&self.noise, wcx, None)
    }
    #[inline]
    fn profile_at(&self, wcx: i32) -> ColumnProfile {
        column_profile_at(&self.noise, wcx)
    }
    #[inline]
    fn hash(&self, x: i32, y: i32) -> f64 {
        self.noise.hash2(x, y)
    }
}

// --- Weighted pick -----------------------------------------------------------

/// Pick one template from `list` by weight, among those eligible at this site.
///
/// Eligibility is tested BEFORE the weighted draw rather than after, so a
/// biome-locked template (a pyramid) does not consume the site and leave nothing
/// — a desert gets pyramids and obelisks at the same rate a plains gets cabins
/// and towers, instead of being thinned out by every template it cannot host.
/// The lists are single digits long and this only runs for a candidate that has
/// already passed the density gate, so the two scans are free.
///
/// The TypeScript kept the eligibility bitmap in a module-scope `Uint8Array`.
/// Here it is a stack array bounded by the registry, which is the same zero
/// allocations without the shared state.
fn pick_template(
    list: &[Template],
    biome: Biome,
    depth: i32,
    band: Sband,
    place_mask: u32,
    r: f64,
) -> Option<&Template> {
    let mut eligible = [false; STRUCT_COUNT];
    let mut total = 0.0;
    for (i, t) in list.iter().enumerate() {
        let ok = (place_mask & t.place.bit()) != 0
            && (t.bands.is_empty() || t.bands.intersects(band))
            && depth >= t.min_depth
            && depth <= t.max_depth
            && t.biomes.as_ref().is_none_or(|b| b.contains(&biome));
        eligible[i] = ok;
        if ok {
            total += t.weight;
        }
    }
    if total <= 0.0 {
        return None;
    }

    let mut x = r * total;
    let mut last: Option<&Template> = None;
    for (i, t) in list.iter().enumerate() {
        if !eligible[i] {
            continue;
        }
        last = Some(t);
        x -= t.weight;
        if x < 0.0 {
            return Some(t);
        }
    }
    last // float slop only
}

// --- Stamping ----------------------------------------------------------------

/// Body row that expanded row `r` draws from. The repeatable slice is a row
/// REMAP, never an expanded copy: a 22-cell tower and a 10-cell one run the same
/// loop over the same array and neither allocates, which matters because every
/// chunk a tower touches re-derives the whole tower.
fn src_row(t: &Template, r: i32, times: i32) -> i32 {
    if t.rep_rows == 0 {
        return r;
    }
    if r < t.rep_from {
        return r;
    }
    let span = t.rep_rows * times;
    if r < t.rep_from + span {
        return t.rep_from + (r - t.rep_from) % t.rep_rows;
    }
    r - span + t.rep_rows
}

/// Expanded row count, and the top-left cell the body is written from.
///
/// These three used to be inlined in both `stamp` and `each_mark`, which is a
/// standing invitation for the two to drift apart — and the moment they do, a
/// loot pass that recovers mark positions from the origin recovers the WRONG
/// cells while every determinism check still passes, because both halves are
/// individually consistent. One definition, four callers.
fn height_of(t: &Template, times: i32) -> i32 {
    t.h + t.rep_rows * (times - 1)
}

fn origin_x(t: &Template, ox: i32) -> i32 {
    let left = t.anchor == Anchor::BottomLeft || t.anchor == Anchor::TopLeft;
    if left { ox } else { ox - (t.w >> 1) }
}

fn origin_y(t: &Template, oy: i32, height: i32) -> i32 {
    let base = match t.anchor {
        Anchor::BottomCenter | Anchor::BottomLeft => oy - (height - 1),
        Anchor::Center => oy - (height >> 1),
        _ => oy,
    };
    base + t.sink
}

/// Stamp `t` with its anchor cell at (ox, oy). Pure in (t, ox, oy, times,
/// mirrored) — the only thing that varies per chunk is which `plot` calls land.
fn stamp(ctx: &mut DecorContext<'_>, s: &StructSite) {
    let t = s.t;
    let height = height_of(t, s.times);
    let x0 = origin_x(t, s.ox);
    let y0 = origin_y(t, s.oy, height);

    if !overlaps(ctx, x0, y0, x0 + t.w - 1, y0 + height - 1) {
        return;
    }

    let w = t.w;
    for r in 0..height {
        let base = src_row(t, r, s.times) * w;
        let wy = y0 + r;
        for c in 0..w {
            let col = if s.mirrored { w - 1 - c } else { c };
            let slot = t.rows[(base + col) as usize];
            if slot == Slot::KEEP {
                continue;
            }
            let code = t.slot_code[slot.index()];
            if t.slot_soft[slot.index()] {
                ctx.plot_if_empty(x0 + c, wy, code);
            } else {
                ctx.plot(x0 + c, wy, code);
            }
        }
    }
}

/// Every marked cell of a placed template, in absolute coordinates.
///
/// Exists so a later loot or spawn pass can recover exactly where a template's
/// caches and dens are WITHOUT reading the world back: it re-derives them from
/// the same origin, the same hashes and the same row remap the stamp used, which
/// is the only way such a pass can stay chunk-independent. `apply_marks` below
/// is that pass.
///
/// The TypeScript hoisted its callback into a module-scope closure over a
/// module-scope `DecorContext` to avoid allocating one per placement. A Rust
/// closure captures by reference and does not allocate, so the scratch is gone.
pub fn each_mark(s: &StructSite, mut f: impl FnMut(i32, i32, Mark)) {
    let t = s.t;
    let height = height_of(t, s.times);
    let x0 = origin_x(t, s.ox);
    let y0 = origin_y(t, s.oy, height);

    for r in 0..height {
        let base = src_row(t, r, s.times) * t.w;
        for c in 0..t.w {
            let col = if s.mirrored { t.w - 1 - c } else { c };
            let mark = t.slot_mark[t.rows[(base + col) as usize].index()];
            if mark != Mark::None {
                f(x0 + c, y0 + r, mark);
            }
        }
    }
}

/// The mark at one absolute cell of a placed template, or [`Mark::None`].
///
/// The inverse of [`each_mark`], and O(1) rather than O(w*h): the row remap is a
/// forward-only function of `r`, so recovering the source row for a KNOWN row is
/// one call to `src_row`, not a search. This is what makes the open-time query
/// below cheap enough to run on a mouse click without a second thought.
fn mark_in_site(s: &StructSite, wcx: i32, wcy: i32) -> Mark {
    let t = s.t;
    let height = height_of(t, s.times);
    let r = wcy - origin_y(t, s.oy, height);
    if r < 0 || r >= height {
        return Mark::None;
    }
    let c = wcx - origin_x(t, s.ox);
    if c < 0 || c >= t.w {
        return Mark::None;
    }
    let col = if s.mirrored { t.w - 1 - c } else { c };
    t.slot_mark[t.rows[(src_row(t, r, s.times) * t.w + col) as usize].index()]
}

// --- The mark pass -----------------------------------------------------------
// Marked glyphs are authored as air (they are the empty cell a cache or a den
// occupies), so until something claims them a vault generates as an empty room.
// This pass claims them: every marked cell becomes the block its mark names.
//
// It is a SECOND traversal of the body rather than a branch inside `stamp` on
// purpose. `each_mark` is the only place that knows how to turn (origin, times,
// mirrored) into mark coordinates, and the open-time query below has to use that
// same knowledge from the other direction; folding the logic into the stamp loop
// would give the two halves separate copies of it. The cost is a second pass
// over a template that (a) has marks at all and (b) actually overlaps this
// chunk, which is a few hundred array reads on the handful of chunks per
// thousand that contain a structure.
//
// Determinism is inherited, not re-argued: the pass is driven by exactly the
// origin, `times` and `mirrored` the stamp used, so every chunk a template
// touches derives the identical mark set and `plot` discards the rest.

fn apply_marks(ctx: &mut DecorContext<'_>, s: &StructSite) {
    let marks = &*MARKS;
    if !marks.active || !s.t.has_mark {
        return;
    }
    let height = height_of(s.t, s.times);
    let x0 = origin_x(s.t, s.ox);
    let y0 = origin_y(s.t, s.oy, height);
    if !overlaps(ctx, x0, y0, x0 + s.t.w - 1, y0 + height - 1) {
        return;
    }

    each_mark(s, |wcx, wcy, mark| {
        let code = marks.block[mark as usize];
        if code != AIR {
            ctx.plot(wcx, wcy, code);
        }
    });
}

// --- Passes ------------------------------------------------------------------

/// Ground-line spread across a template's footprint. Five probes, positional.
fn site_spread<Q: SiteQuery + ?Sized>(q: &mut Q, ox: i32, half: i32, base: i32) -> i32 {
    let mut lo = base;
    let mut hi = base;
    for i in -2i32..=2 {
        if i == 0 {
            continue;
        }
        let off = js_round(f64::from(i * half) / 2.0) as i32;
        let s = q.surface_at(ox + off);
        if s < lo {
            lo = s;
        }
        if s > hi {
            hi = s;
        }
    }
    hi - lo
}

/// One resolved placement: everything `stamp`, `apply_marks` and the open-time
/// query need, and nothing that depends on which chunk asked.
///
/// Splitting resolution out from stamping is what makes a chest's contents
/// derivable from its coordinates alone. [`resolve_column_site`] /
/// [`resolve_lattice_site`] touch `surface_at`, `profile_at` and `hash` — all
/// pure in absolute coordinates — and never `base_x`/`base_y`. So the same call
/// made by the generator while painting a chunk and by the game while the player
/// is prising a lid off, two hours and a chunk eviction later, returns the same
/// site.
///
/// The TypeScript rewrote one module-level scratch object per candidate. This is
/// a `Copy` struct returned by value: the same zero allocations, and nothing
/// shared between rayon workers.
#[derive(Clone, Copy, Debug)]
pub struct StructSite {
    pub t: &'static Template,
    pub ox: i32,
    pub oy: i32,
    pub times: i32,
    pub mirrored: bool,
}

/// Does a template stand on column `ox`, and if so, which and how?
///
/// The row is DERIVED from the origin column's ground line, never searched for,
/// which is what lets a chunk two rows above the terrain paint the top of a
/// 22-cell tower correctly without knowing anything about the chunk below it.
pub fn resolve_column_site<Q: SiteQuery + ?Sized>(q: &mut Q, ox: i32) -> Option<StructSite> {
    if h(q, ox, 0, 1) >= COL_DENSITY {
        return None; // one hash rejects half of them
    }

    let base = q.surface_at(ox);
    // Which placement classes this column can host at all. A shore template needs
    // the beach band; nothing at all is built on the sea floor.
    let mut place_mask = Place::Floating.bit();
    if base <= SEA_LEVEL_Y {
        place_mask |= Place::Surface.bit();
        if shore_weight_at(base) >= 0.6 {
            place_mask |= Place::Shore.bit();
        }
    }

    let col = q.profile_at(ox);
    let t = pick_template(
        &TEMPLATES.column,
        col.surf_a,
        0,
        Sband::SURFACE,
        place_mask,
        h(q, ox, 0, 2),
    )?;
    if h(q, ox, 0, 3) >= t.rarity {
        return None;
    }
    // Nothing is built on a cliff. The probes are positional, so every chunk that
    // re-derives this template rejects (or accepts) the site identically.
    if t.place != Place::Floating && site_spread(q, ox, t.w >> 1, base) > t.flatness {
        return None;
    }

    let times = if t.rep_rows == 0 {
        1
    } else {
        ri(h(q, ox, 0, 4), t.rep_min, t.rep_max)
    };
    let mirrored = t.mirror && h(q, ox, 0, 5) < 0.5;
    Some(StructSite {
        t,
        ox,
        oy: base - t.clearance,
        times,
        mirrored,
    })
}

/// Does a buried template sit at lattice cell (ox, oy)?
pub fn resolve_lattice_site<Q: SiteQuery + ?Sized>(
    q: &mut Q,
    ox: i32,
    oy: i32,
) -> Option<StructSite> {
    if h(q, ox, oy, 6) >= SUB_DENSITY {
        return None; // one hash rejects three in four
    }

    let depth = oy - q.surface_at(ox);
    if depth < SUB_MIN_DEPTH {
        return None;
    }
    // The underworld is its own world: templates authored for it are the only
    // ones allowed down there, and none of them are allowed above it.
    let place_mask = if depth >= UNDERWORLD_DEPTH {
        Place::Underworld.bit()
    } else {
        Place::Underground.bit() | Place::Cavern.bit()
    };

    let col = q.profile_at(ox);
    let t = pick_template(
        &TEMPLATES.lattice,
        col.surf_a,
        depth,
        band_at(depth),
        place_mask,
        h(q, ox, oy, 7),
    )?;
    if h(q, ox, oy, 8) >= t.rarity {
        return None;
    }

    let times = if t.rep_rows == 0 {
        1
    } else {
        ri(h(q, ox, oy, 9), t.rep_min, t.rep_max)
    };
    let mirrored = t.mirror && h(q, ox, oy, 10) < 0.5;
    Some(StructSite {
        t,
        ox,
        oy,
        times,
        mirrored,
    })
}

/// Column pass — surface, shore and floating templates.
fn stamp_column_places(ctx: &mut DecorContext<'_>) {
    let tm = &*TEMPLATES;
    if ctx.base_y + CHUNK_CELLS <= tm.col_band_top || ctx.base_y >= tm.col_band_bot {
        return;
    }

    for ox in origin_columns(ctx.base_x, tm.reach_x, COL_STRIDE, COL_PHASE) {
        let Some(s) = resolve_column_site(ctx, ox) else {
            continue;
        };
        stamp(ctx, &s);
        apply_marks(ctx, &s);
    }
}

/// Buried pass — a real 2D origin lattice, gated on depth.
fn stamp_lattice_places(ctx: &mut DecorContext<'_>) {
    let tm = &*TEMPLATES;
    if ctx.base_y + CHUNK_CELLS < tm.sub_band_top {
        return;
    }

    let cells = origin_cells(
        ctx.base_x,
        ctx.base_y,
        tm.reach_x,
        tm.reach_y,
        Lattice {
            stride_x: SUB_STRIDE_X,
            stride_y: SUB_STRIDE_Y,
            phase_x: SUB_PHASE_X,
            phase_y: SUB_PHASE_Y,
        },
    );
    for (ox, oy) in cells {
        let Some(s) = resolve_lattice_site(ctx, ox, oy) else {
            continue;
        };
        stamp(ctx, &s);
        apply_marks(ctx, &s);
    }
}

// --- The open-time query -----------------------------------------------------

/// What a query found: which template owns a cell, and what it marked it as.
#[derive(Clone, Copy, Debug)]
pub struct MarkHit {
    /// Borrowed from the immutable template table — safe to hold across further
    /// queries.
    pub template: &'static Template,
    pub mark: Mark,
}

/// Which template marked the cell at (wcx, wcy), if any — the inverse of the
/// whole placement pass.
///
/// THE POINT. A chest stores nothing. Its contents are rolled from its
/// coordinates at the moment it is opened, and they should be worth more when
/// the thing that put it there was rarer. That means recovering the HOST
/// TEMPLATE from a bare cell coordinate, and the only way to do that without
/// reading the world back (which would make the answer depend on generation
/// order, on chunk eviction, and on whether the player has already dug through
/// the wall) is to re-run placement around the point and ask each candidate
/// whether it claims it.
///
/// It is cheap because the lattices are coarse relative to the templates: an
/// origin can only be within `STRUCT_REACH` of the query, and the strides are
/// 112 for columns and 88x64 for the buried grid, so the scan considers one or
/// two candidates per axis and the density hash throws most of those out
/// immediately. The reverse lookup itself is O(1).
///
/// Candidates are visited in the same order [`stamp_structs`] paints them and
/// the LAST hit wins, so when two templates overlap the query agrees with the
/// block that actually survived to the grid.
pub fn mark_at<Q: SiteQuery + ?Sized>(q: &mut Q, wcx: i32, wcy: i32) -> Option<MarkHit> {
    let tm = &*TEMPLATES;
    let mut template: Option<&'static Template> = None;
    let mut mark = Mark::None;

    let from_x = wcx - tm.reach_x;
    let to_x = wcx + tm.reach_x;

    if !tm.column.is_empty() {
        let mut ox = from_x + pmod(COL_PHASE - from_x, COL_STRIDE);
        while ox <= to_x {
            if let Some(s) = resolve_column_site(q, ox) {
                let m = mark_in_site(&s, wcx, wcy);
                if m != Mark::None {
                    template = Some(s.t);
                    mark = m;
                }
            }
            ox += COL_STRIDE;
        }
    }

    if !tm.lattice.is_empty() {
        let from_y = wcy - tm.reach_y;
        let to_y = wcy + tm.reach_y;
        let mut oy = from_y + pmod(SUB_PHASE_Y - from_y, SUB_STRIDE_Y);
        while oy <= to_y {
            let mut ox = from_x + pmod(SUB_PHASE_X - from_x, SUB_STRIDE_X);
            while ox <= to_x {
                if let Some(s) = resolve_lattice_site(q, ox, oy) {
                    let m = mark_in_site(&s, wcx, wcy);
                    if m != Mark::None {
                        template = Some(s.t);
                        mark = m;
                    }
                }
                ox += SUB_STRIDE_X;
            }
            oy += SUB_STRIDE_Y;
        }
    }

    template.map(|template| MarkHit { template, mark })
}

/// Stamp this chunk's share of every template overlapping it.
pub fn stamp_structs(ctx: &mut DecorContext<'_>) {
    let tm = &*TEMPLATES;
    if !tm.column.is_empty() {
        stamp_column_places(ctx);
    }
    if !tm.lattice.is_empty() {
        stamp_lattice_places(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::materials::EMPTY;

    /// A chunk-sized canvas plus everything a `DecorContext` needs to exist.
    struct Canvas {
        noise: Noise,
        heightmap: Heightmap,
        cells: Vec<CellId>,
    }

    impl Canvas {
        fn new(seed: u32) -> Canvas {
            Canvas {
                noise: Noise::new(seed),
                heightmap: Heightmap::new(),
                cells: vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize],
            }
        }
    }

    #[test]
    fn the_compiled_geometry_tables_agree_with_the_bodies_they_were_derived_from() {
        // `w`, `h` and `max_h` are measured off the authored body here and
        // emitted by `contentc` there. If the two ever disagree the decorator's
        // declared reach stops matching what actually gets stamped.
        for def in STRUCTS.iter() {
            let Some(t) = build(def) else { continue };
            let c = def.code as usize;
            assert_eq!(t.w, i32::from(STRUCT_W[c]), "{} width", t.id);
            assert_eq!(t.h, i32::from(STRUCT_H[c]), "{} height", t.id);
            assert_eq!(t.max_h, i32::from(STRUCT_MAXH[c]), "{} max height", t.id);
            assert_eq!(
                t.rep_from,
                i32::from(STRUCT_REP_FROM[c]),
                "{} repeat from",
                t.id
            );
            assert_eq!(
                t.rep_rows,
                i32::from(STRUCT_REP_ROWS[c]),
                "{} repeat rows",
                t.id
            );
            assert_eq!(
                t.rep_min,
                i32::from(STRUCT_REP_MIN[c]),
                "{} repeat min",
                t.id
            );
            assert_eq!(
                t.rep_max,
                i32::from(STRUCT_REP_MAX[c]),
                "{} repeat max",
                t.id
            );
            assert_eq!(t.place as u8, STRUCT_PLACE[c], "{} place", t.id);
        }
    }

    #[test]
    fn every_template_survives_the_build_or_is_a_tombstone() {
        let tm = templates();
        assert!(!tm.column.is_empty(), "no column-placed templates compiled");
        assert!(
            !tm.lattice.is_empty(),
            "no lattice-placed templates compiled"
        );
        assert!(tm.reach_x > 0 && tm.reach_y > 0);
        assert!(tm.col_band_top < tm.col_band_bot);
        assert!(tm.sub_band_top > tm.col_band_top);
    }

    #[test]
    fn a_template_stamps_identically_from_two_different_chunk_origins() {
        // THE invariant. Two chunks that both overlap one template must agree on
        // every cell in the overlap, or the template shows a seam wherever the
        // player happens to have approached it from.
        let tm = templates();
        let t = &tm.column[0];
        let site = StructSite {
            t,
            ox: 1000,
            oy: 60,
            times: 1,
            mirrored: false,
        };

        let height = height_of(t, site.times);
        let x0 = origin_x(t, site.ox);
        let y0 = origin_y(t, site.oy, height);

        // Two chunk origins whose windows both cover the template's top-left.
        let mut a = Canvas::new(7);
        let mut b = Canvas::new(7);
        let (bax, bay) = (x0, y0);
        let (bbx, bby) = (x0 - CHUNK_CELLS + 4, y0 - CHUNK_CELLS + 4);

        {
            let mut ctx = DecorContext::new(&a.noise, 7, bax, bay, &mut a.cells, &mut a.heightmap);
            stamp(&mut ctx, &site);
        }
        {
            let mut ctx = DecorContext::new(&b.noise, 7, bbx, bby, &mut b.cells, &mut b.heightmap);
            stamp(&mut ctx, &site);
        }

        let mut compared = 0;
        for wy in y0..y0 + height {
            for wx in x0..x0 + t.w {
                let (lax, lay) = (wx - bax, wy - bay);
                let (lbx, lby) = (wx - bbx, wy - bby);
                let in_a = (0..CHUNK_CELLS).contains(&lax) && (0..CHUNK_CELLS).contains(&lay);
                let in_b = (0..CHUNK_CELLS).contains(&lbx) && (0..CHUNK_CELLS).contains(&lby);
                if !(in_a && in_b) {
                    continue;
                }
                compared += 1;
                assert_eq!(
                    a.cells[(lay * CHUNK_CELLS + lax) as usize],
                    b.cells[(lby * CHUNK_CELLS + lbx) as usize],
                    "chunks disagree at ({wx}, {wy})"
                );
            }
        }
        assert!(
            compared > 0,
            "the two chunks did not actually overlap the template"
        );
    }

    #[test]
    fn mirroring_and_the_row_remap_are_each_others_inverse() {
        // `each_mark` walks forward, `mark_in_site` walks back. If they ever
        // drift, loot lands on the wrong cells while every determinism check
        // still passes, because each half is individually consistent.
        let tm = templates();
        for t in tm.column.iter().chain(tm.lattice.iter()) {
            if !t.has_mark {
                continue;
            }
            for mirrored in [false, true] {
                for times in [t.rep_min, t.rep_max] {
                    let s = StructSite {
                        t,
                        ox: -37,
                        oy: 211,
                        times,
                        mirrored,
                    };
                    let mut seen = 0;
                    each_mark(&s, |wcx, wcy, mark| {
                        seen += 1;
                        assert_eq!(
                            mark_in_site(&s, wcx, wcy),
                            mark,
                            "{} at ({wcx},{wcy})",
                            t.id
                        );
                    });
                    assert!(seen > 0, "{} claims a mark but walks none", t.id);
                }
            }
        }
    }

    #[test]
    fn site_resolution_does_not_depend_on_which_context_asks() {
        // A `DecorContext` painting a chunk and a standalone `StructQuery` must
        // resolve the same site, or a chest's contents change when the chunk
        // holding it is evicted.
        let mut canvas = Canvas::new(4242);
        let mut q = StructQuery::new(4242);
        for ox in [-4096, -112, 0, 53, 1000, 100_000] {
            let via_ctx = {
                let mut ctx = DecorContext::new(
                    &canvas.noise,
                    4242,
                    ox,
                    0,
                    &mut canvas.cells,
                    &mut canvas.heightmap,
                );
                resolve_column_site(&mut ctx, ox).map(|s| (s.t.id, s.ox, s.oy, s.times, s.mirrored))
            };
            let via_query =
                resolve_column_site(&mut q, ox).map(|s| (s.t.id, s.ox, s.oy, s.times, s.mirrored));
            assert_eq!(via_ctx, via_query, "column {ox}");
        }
    }

    #[test]
    fn the_query_recovers_the_template_that_marked_a_cell() {
        // `mark_at` is the inverse of the whole placement pass, and the only
        // thing loot has to go on. If it ever stops finding a mark the stamp pass
        // really placed, every chest in the world silently drops to tier 0.
        let mut q = StructQuery::new(31_337);
        let mut checked = 0;
        // Walk the column lattice; every candidate is `COL_PHASE (mod COL_STRIDE)`.
        for k in -60..60 {
            let ox = COL_PHASE + k * COL_STRIDE;
            let Some(s) = resolve_column_site(&mut q, ox) else {
                continue;
            };
            if !s.t.has_mark {
                continue;
            }
            let mut marks: Vec<(i32, i32, Mark)> = Vec::new();
            each_mark(&s, |wcx, wcy, mark| marks.push((wcx, wcy, mark)));
            for (wcx, wcy, mark) in marks {
                let hit = mark_at(&mut q, wcx, wcy)
                    .unwrap_or_else(|| panic!("{} marked ({wcx},{wcy}) and lost it", s.t.id));
                // Another template may overlap and win; the mark that comes back
                // must at least be a real one placed by a real template.
                assert_ne!(hit.mark, Mark::None);
                if std::ptr::eq(hit.template, s.t) {
                    assert_eq!(hit.mark, mark, "{} at ({wcx},{wcy})", s.t.id);
                }
                checked += 1;
            }
        }
        assert!(checked > 0, "120 lattice cells and not one marked template");
    }

    #[test]
    fn the_pass_actually_stamps_something() {
        // Every other test here would pass just as happily against a pass that
        // painted nothing. This is the one that says it does not.
        let seed = 31_337;
        let noise = Noise::new(seed);
        let mut hm = Heightmap::new();
        let mut painted = 0usize;
        for cy in 0..6 {
            for cx in -24..24 {
                let mut cells = vec![EMPTY; (CHUNK_CELLS * CHUNK_CELLS) as usize];
                let mut ctx = DecorContext::new(
                    &noise,
                    seed,
                    cx * CHUNK_CELLS,
                    cy * CHUNK_CELLS,
                    &mut cells,
                    &mut hm,
                );
                stamp_structs(&mut ctx);
                painted += cells.iter().filter(|&&c| c != EMPTY).count();
            }
        }
        assert!(painted > 0, "288 surface chunks and not one template cell");
    }

    #[test]
    fn js_round_breaks_halves_toward_positive_infinity() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0); // Rust's `f64::round` would say -3
        assert_eq!(js_round(-2.6), -3.0);
        assert_eq!(js_round(0.0), 0.0);
    }

    #[test]
    fn ri_is_total_even_when_the_range_is_inverted() {
        assert_eq!(ri(0.0, 3, 7), 3);
        assert_eq!(ri(0.999_999, 3, 7), 7);
        // `hi < lo` cannot happen with the authored numbers, but the clamp order
        // is load-bearing if content ever makes it happen.
        assert_eq!(ri(0.5, 7, 3), 3);
    }
}

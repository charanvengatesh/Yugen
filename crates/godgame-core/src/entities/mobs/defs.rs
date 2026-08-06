//! The bestiary — a facade over `godgame_data::mobs`.
//!
//! Every creature is compiled from `content/mobs/*.toml` by `contentc`. This
//! module exists to do the two things a generated module deliberately cannot,
//! and to keep the names `brain.rs` / `system.rs` already speak. It mirrors
//! [`crate::sim::materials`], which is the same seam over the block tables:
//! hand-written code never imports `godgame_data::mobs` directly, it comes
//! through here.
//!
//! ---- WHY THE SCALING LIVES HERE AND NOT IN THE COMPILER --------------------
//! Every distance-bearing stat in `content/mobs/` is authored in the SAME
//! reference frame as [`crate::config::physics`]: the value that would feel
//! right on the 24px-tall character the player's physics were tuned against. It
//! is scaled to the creature's own body by `PHYS_SCALE * (bodyHeightPx /
//! PLAYER_H)`, which is exactly the rule `config::physics` uses (lengths,
//! velocities and accelerations scale with body height; durations and ratios do
//! not). The upshot is that a stat can be read directly against a player stat —
//! a walker at `speed 110` covers ground at roughly a third of the player's
//! `MAX_RUN_SPEED: 360` *relative to its own size*, and it stays true if
//! `PHYS_SCALE` or the player's size is ever retuned.
//!
//! That multiply depends on runtime constants AND on the sprite's measured
//! height, so the compiler emits the RAW authored numbers and the scaling
//! happens at load, here. Baking `PHYS_SCALE` into generated output would
//! silently desync the bestiary from the player the next time anyone retuned
//! movement. `tests/mob_regression.rs` is the standing gate on exactly that: it
//! compares the PRODUCTS verbatim against a snapshot taken before the content
//! compiler owned the bestiary.
//!
//! Radii (aggro, flee, ranged) are deliberately NOT body-scaled: they are about
//! how far away something notices you on screen, which is a property of the
//! camera, not of the creature's legs. They are plain world px.
//!
//! ---- ART IS NOT THE HITBOX -------------------------------------------------
//! `body_cells_*` (content) is the collision box; `art.cells_*` is the picture,
//! and the picture may be LARGER — wings, antennae, a plume. [`MobDef::w_px`] /
//! [`MobDef::h_px`] are the body; `art_w_px` / `art_h_px` / `art_pad_*` are the
//! draw rect. The overhang is centred horizontally and aligned to the FEET,
//! because a creature's feet are where it touches the world and the one edge
//! that must never drift.
//!
//! ---- HABITAT ---------------------------------------------------------------
//! A creature is eligible at a candidate cell when the depth band matches, the
//! depth is inside its `min_depth`/`max_depth` window, the dominant biome is in
//! its list (absent list = anywhere), and any extra predicate (`needs_lava`)
//! holds. Eligible defs are then picked by weight, so population composition
//! follows the terrain without any per-biome spawn table.
//!
//! # What the port dropped on the way in
//!
//! The TypeScript facade also CONSTRUCTED a `Sprite` per creature and resolved
//! `stateIds` off it, because `src/generated/*.gen.ts` imports nothing and a
//! `Sprite` rasterises canvases. This crate carries no renderer, so `sprite` and
//! `stateIds` did not come across, and neither did `bloodCss` — a canvas
//! fillStyle string. Everything the two draw calls READ is published as data
//! instead (`art_w_px`, `art_h_px`, `art_pad_x_px`, `art_pad_top_px`, `blood`,
//! `glow`), exactly as `player.rs` published what `Player.draw` read.
//!
//! The three rules the facade supplied to the sprite adapter are rules about
//! how the game draws creatures in general, so they move with the drawing —
//! except that two of them are stated here anyway because the SIMULATION
//! depends on them: [`POSE_FALLBACK_AIR_IS_MOVE`] and [`POSE_RATE_SCALE_IDLE`]
//! are what a brain's choice of [`MobPose`] means, and [`VARIANT_COUNT`] is
//! consumed by a spawn RNG draw whose stream position is load-bearing.

use std::sync::LazyLock;

use crate::config::{CAVERN_DEPTH, CELL_SIZE, DEEP_DEPTH, PHYS_SCALE, PLAYER_H};

pub use godgame_data::mobs::{
    Band, Dmg, MOB_BANDS, MOB_CODES, MOB_COUNT, MOB_FLAGS, MOB_IDS, MOB_IMMUNE, MOB_MAXDEPTH,
    MOB_MINDEPTH, MOB_WEIGHT, MobBrain, MobDrop, MobProjectile, Mobf, NEVER, mob,
};

/// Depth band relative to the LOCAL surface row, in cells.
///
/// The generated enum, renamed. The TypeScript spelled this `keyof typeof BAND`
/// — a string union whose members happened to match the bitmask's keys; here the
/// enum IS the vocabulary and [`band_bit`] is the one place it becomes a bit.
pub use godgame_data::mobs::MobSpecBands as MobBand;

/// A creature's ranged attack. Every field is a duration, a radius or damage —
/// none are body-scaled.
pub use godgame_data::mobs::MobSpecRanged as MobRanged;

use godgame_data::mobs::{MOBS, MobSpec};

/// Cells below the local surface at which the shallow band starts.
pub const BAND_SURFACE_MAX: i32 = 8;

/// Which band a depth below the local surface falls in.
///
/// `depth_cells` is an integer cell count in the TypeScript too (`wcy -
/// surfaceRowAt(...)`); it is typed as one here rather than as the float
/// `number` was, which is the only difference.
pub fn band_at_depth(depth_cells: i32) -> MobBand {
    if depth_cells < BAND_SURFACE_MAX {
        return MobBand::Surface;
    }
    if depth_cells < CAVERN_DEPTH {
        return MobBand::Shallow;
    }
    if depth_cells < DEEP_DEPTH {
        return MobBand::Cavern;
    }
    MobBand::Deep
}

/// Bitmask lookup for a band, so the spawn picker tests `MOB_BANDS[code] & bit`
/// instead of walking a list per candidate def. Bit order is fixed by
/// `crates/contentc/src/schemas/`.
pub const fn band_bit(band: MobBand) -> Band {
    match band {
        MobBand::Surface => Band::SURFACE,
        MobBand::Shallow => Band::SHALLOW,
        MobBand::Cavern => Band::CAVERN,
        MobBand::Deep => Band::DEEP,
    }
}

/// Which loop a creature is playing.
///
/// It was `"idle" | "move" | "air"` in the TypeScript, and then an index into a
/// dense `stateIds` array — the string had to become a small integer because
/// resolving a name to a sprite state was a map hit per mob per frame for a
/// value that cannot change after load. It is an enum here, which is that same
/// small integer with the names kept, and the renderer does the one array read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MobPose {
    Idle = 0,
    Move = 1,
    Air = 2,
}

/// A creature with no `air` sequence flies with its walk cycle.
///
/// A rule about how the game draws creatures in general, not a fact about any
/// one of them — letting content express it would give forty files forty
/// chances to disagree about what a missing state means. It is stated here
/// rather than in the renderer because it is the SEMANTICS of
/// [`MobPose::Air`]: a brain that sets it is not promising a distinct loop.
pub const POSE_FALLBACK_AIR_IS_MOVE: bool = true;

/// Mobs read their idle loop at this multiple of the authored rate.
///
/// The pre-migration `MobSprite` hard-coded exactly this (`pose === "idle" ?
/// fps * 0.5 : fps`); content does NOT pre-halve — an author writing `fps 7`
/// still means "seven, at the walking rate".
pub const POSE_RATE_SCALE_IDLE: f32 = 0.5;

/// How many tinted copies of a creature the art system bakes.
///
/// The length of the renderer's variant-tint table, restated on this side of
/// the boundary because `MobSystem::init` assigns `rand() * VARIANT_COUNT` to
/// every creature it spawns — the draw is part of the mob RNG's consumption
/// order, so the simulation cannot not know this number. A mob that baked fewer
/// than the full table would silently wrap several individuals onto one tint,
/// which is why content's own `variants` (schema default 1) is ignored here.
pub const VARIANT_COUNT: i32 = 3;

/// Reference gravity, matching the player's raw tunable in `config::physics`.
const RAW_GRAVITY: f32 = 2400.0;
/// Reference terminal fall speed, likewise.
const RAW_MAX_FALL: f32 = 1400.0;

/// Art footprint of a tombstoned id.
///
/// A tombstoned id (`content/FORMAT.md` §4) keeps its code forever so an old
/// save still resolves whatever it had spawned, but it carries no `art` group.
/// It is exactly 1x1 because the schema's tombstone defaults are
/// `body_cells_w 1` / `body_cells_h 1` — art and body agree, so the overhang
/// checks in `build` pass for a tombstone without needing a special case.
const REMOVED_ART_CELLS: (i32, i32) = (1, 1);

/// One creature, with the load-time arithmetic already done.
///
/// A wide record of `Copy` scalars, which is what the TypeScript's readonly
/// interface was; it lives in a build-once table and is handed out as
/// `&'static MobDef`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MobDef {
    /// Stable authoring name.
    pub id: &'static str,
    /// Stable numeric code — also the index into [`MOB_DEFS`] and every flat
    /// table.
    pub code: u16,
    /// Display name (bestiary UI, debug HUD).
    pub name: &'static str,
    /// Which movement/decision routine drives this creature.
    pub brain: MobBrain,

    /// COLLISION BOX width, in world px — `body_cells_w * CELL_SIZE`, always a
    /// whole number of cells. This is what the world collides with, what spawn
    /// footing is tested against, and what the burrower's breach tell is drawn
    /// the width of. It is NOT the size of the picture; see [`MobDef::art_w_px`].
    pub w_px: f32,
    /// COLLISION BOX height, in world px. See [`MobDef::w_px`].
    pub h_px: f32,
    /// Art rect width, in world px — `art.cells_w * CELL_SIZE`. DRAW ONLY.
    pub art_w_px: f32,
    /// Art rect height, in world px. DRAW ONLY.
    pub art_h_px: f32,
    /// Horizontal overhang of the art beyond the body, in world px. DRAW ONLY:
    /// the sprite is blitted at `(box.x - art_pad_x_px, box.y - art_pad_top_px)`.
    ///
    /// Centred, which is why an odd width difference is a load-time error — half
    /// a cell of pad would put sprite pixels off the cell grid.
    pub art_pad_x_px: f32,
    /// Vertical overhang, ALL on top, so the art's bottom row always lands on
    /// the body's bottom row. Feet stay planted; horns grow upward. DRAW ONLY.
    pub art_pad_top_px: f32,
    /// `PHYS_SCALE * (h_px / PLAYER_H)` — the factor already applied to every
    /// distance-bearing stat below. Anchored on the BODY height, never the
    /// art's: redrawing a creature with a taller plume must not make it walk
    /// faster. Exposed so anything the creature EMITS (a projectile's speed,
    /// say) can be authored in the same player reference frame and scaled the
    /// same way, instead of being tuned in raw px against one body size.
    pub body_scale: f32,

    /// Hit points at full health.
    pub max_health: f32,
    /// Health taken from the player per contact hit. 0 = harmless.
    pub contact_damage: f32,
    /// Seconds this creature must wait between contact hits.
    pub attack_cooldown: f32,
    /// Flat damage subtracted from every incoming hit. Combat floors the result
    /// at 1.
    pub armor: f32,
    /// Packed damage tags this creature ignores: `def.immune_mask & Dmg::FIRE`.
    pub immune_mask: Dmg,

    // --- px-space motion (already body-scaled) ---
    /// Patrol speed, px/s.
    pub speed: f32,
    /// Speed while alerted, px/s.
    pub chase_speed: f32,
    /// Initial upward velocity for a hop, ledge-leap or eruption, px/s. 0 =
    /// cannot leave the ground.
    pub jump_speed: f32,
    /// Downward acceleration, px/s^2. 0 = flight.
    pub gravity: f32,
    /// Terminal fall speed, px/s.
    pub max_fall: f32,
    /// Height a blocked horizontal move may be retried lifted by, in world px.
    pub step_up_max: f32,
    /// Impulse a hit puts into this creature, px/s — a FLOOR, not the value.
    pub knockback: f32,

    // --- perception, world px ---
    /// Horizontal notice radius for a hostile. 0 = never aggresses.
    pub aggro_px: f32,
    /// Vertical half-height of the notice box.
    pub aggro_y_px: f32,
    /// Horizontal panic radius. Non-zero makes the creature run AWAY and
    /// overrides `aggro_px`.
    pub flee_px: f32,
    /// Ranged attack, or `None` for a melee-only creature.
    pub ranged: Option<MobRanged>,

    /// Seconds between hops / dives / surfacings, depending on brain.
    pub beat: f32,
    /// Additive self-illumination drawn in the post-light overlay pass, 0..1.
    pub glow: f32,
    /// Skip the material-hazard probe outright — 'nothing the world is made of
    /// can hurt me', which is what a creature that LIVES in lava needs.
    pub lava_immune: bool,
    /// Particle colour for hits and death bursts — the creature's body tone.
    pub blood: [u8; 3],

    // --- rewards ---
    /// Experience awarded for a kill.
    pub xp: i32,
    /// Currency awarded for a kill, separate from `xp`.
    pub value: i32,
    /// What killing this yields. Rolled independently, so several entries can
    /// all drop.
    pub drops: &'static [MobDrop],

    // --- habitat ---
    /// Depth bands this may spawn in.
    pub bands: &'static [MobBand],
    /// First depth inside the bands, in cells below the local surface.
    pub min_depth: i32,
    /// Last depth this may spawn at. 65535 = no upper bound beyond `bands`.
    pub max_depth: i32,
    /// Dominant biome ids this may spawn in. `None` = any biome.
    pub biomes: Option<&'static [&'static str]>,
    /// Weight is multiplied by darkness at the surface, so these thin out in
    /// daylight.
    pub nocturnal: bool,
    /// Requires lava within a short radius of the spawn point.
    pub needs_lava: bool,
    /// Inclusive `[min, max]` count spawned together.
    pub pack_size: [f32; 2],
    /// Relative spawn weight among eligible defs. 0 = never spawns naturally.
    pub weight: f32,
}

/// Load-time build of one def from its compiled spec.
///
/// Panics on a geometry the art system cannot draw, which is the Rust spelling
/// of the TypeScript's `throw` — loud at load, matching the bake-time
/// strictness that already rejects a short row or a stray palette character.
/// Because [`MOB_DEFS`] is a `LazyLock`, "at load" means "on the first read of
/// the bestiary", which is the first thing any host does.
fn build(s: &'static MobSpec) -> MobDef {
    let (art_cells_w, art_cells_h) = match s.art {
        Some(a) => (a.cells_w, a.cells_h),
        None => REMOVED_ART_CELLS,
    };

    // --- geometry: the body is authored, the art rect is measured -------------
    let w_px = (s.body_cells_w * CELL_SIZE) as f32;
    let body_h_px = (s.body_cells_h * CELL_SIZE) as f32;
    let pad_cells_x = art_cells_w - s.body_cells_w;
    let pad_cells_y = art_cells_h - s.body_cells_h;
    // Art smaller than the body means part of the hitbox is invisible — the
    // player is hit by nothing.
    assert!(
        pad_cells_x >= 0 && pad_cells_y >= 0,
        "mobs::defs: {} art {}x{} is smaller than its {}x{} body — that is a hole in the hitbox",
        s.id,
        art_cells_w,
        art_cells_h,
        s.body_cells_w,
        s.body_cells_h
    );
    // The horizontal overhang is split evenly, so an odd difference would put
    // the sprite half a cell off the world grid — the exact mismatch the
    // one-pixel-one-cell invariant exists to prevent.
    assert!(
        pad_cells_x % 2 == 0,
        "mobs::defs: {} art is {} cells wider than its body; the overhang is centred, \
         so the difference must be even",
        s.id,
        pad_cells_x
    );

    // The dimensional rule from `config::physics`, re-anchored on the exported
    // PHYS_SCALE (which is itself tied to PLAYER_H) so mobs stay in step with
    // the player's feel automatically if either is retuned. BODY height, not art
    // height: this is the whole point of the split.
    //
    // WIDTH: this one multiply is done in `f64` and narrowed once per field,
    // which is the only place in the entities tier that leaves `f32`. It is
    // deliberate and it is not about precision for its own sake. `k` is a
    // recurring factor — nine fields are a product with it — so rounding it to
    // `f32` first and then multiplying nine times compounds one rounding into
    // nine, and `chase_speed` in particular lands an ulp away from the number
    // the TypeScript produced. Doing the arithmetic at the reference's width and
    // narrowing at the store makes the bestiary land on EXACTLY the original's
    // numbers, which is what lets `tests/mob_regression.rs` be an exact-equality
    // gate rather than a tolerance. Nothing downstream is `f64`: every field
    // below is `f32` and every consumer of them is too.
    let k = f64::from(PHYS_SCALE) * (f64::from(body_h_px) / f64::from(PLAYER_H));
    let g = f64::from(s.gravity_scale) * f64::from(RAW_GRAVITY) * k;
    /// Multiply an authored stat by the body scale and store it at the width the
    /// rest of the crate speaks.
    #[inline]
    fn scale(v: f32, k: f64) -> f32 {
        (f64::from(v) * k) as f32
    }

    MobDef {
        id: s.id,
        code: s.code,
        name: s.name,
        brain: s.brain,
        w_px,
        h_px: body_h_px,
        art_w_px: (art_cells_w * CELL_SIZE) as f32,
        art_h_px: (art_cells_h * CELL_SIZE) as f32,
        art_pad_x_px: ((pad_cells_x / 2) * CELL_SIZE) as f32,
        art_pad_top_px: (pad_cells_y * CELL_SIZE) as f32,
        body_scale: k as f32,
        max_health: s.max_health,
        contact_damage: s.contact_damage,
        attack_cooldown: s.attack_cooldown,
        armor: s.armor,
        immune_mask: Dmg::from_bits_truncate(MOB_IMMUNE[s.code as usize]),
        speed: scale(s.speed, k),
        // A creature that does not change gear defaults its chase speed to its
        // patrol speed, and the default is applied BEFORE the scale so the two
        // are the same product rather than two roundings of one.
        chase_speed: scale(s.chase_speed.unwrap_or(s.speed), k),
        jump_speed: scale(s.jump_speed, k),
        gravity: g as f32,
        max_fall: scale(RAW_MAX_FALL, k),
        // One cell, same ratio-to-body reasoning as the player's STEP_UP_CELLS.
        step_up_max: CELL_SIZE as f32,
        knockback: scale(260.0, k),
        aggro_px: s.aggro_px,
        aggro_y_px: s.aggro_y_px,
        flee_px: s.flee_px,
        ranged: s.ranged,
        beat: s.beat,
        glow: s.glow,
        lava_immune: s.lava_immune,
        blood: s.blood,
        xp: s.xp,
        value: s.value,
        drops: s.drops.unwrap_or(EMPTY_DROPS),
        bands: s.bands,
        min_depth: s.min_depth,
        max_depth: s.max_depth,
        biomes: s.biomes,
        nocturnal: s.nocturnal,
        needs_lava: s.needs_lava,
        pack_size: s.pack_size,
        weight: s.weight,
    }
}

/// The drop table a creature with no `drops` group gets.
const EMPTY_DROPS: &[MobDrop] = &[];

/// Every creature, index == code, tombstones included — the same shape
/// `BLOCKS` has, so a code read out of a save or an event indexes straight in.
/// A tombstone carries `weight: 0`, which is what keeps it out of the spawn
/// picker without needing a second array of "real" defs.
///
/// A `LazyLock` rather than a `const`: the body scale is a float product of a
/// config constant and a per-creature measurement, and the whole point of this
/// module is that the multiply happens at LOAD (see the header). Building it
/// once on first read is that, spelled in Rust.
pub static MOB_DEFS: LazyLock<Vec<MobDef>> = LazyLock::new(|| MOBS.iter().map(build).collect());

/// Index of a def by id, or `None` if there is no such creature.
pub fn def_index(id: &str) -> Option<usize> {
    MOB_CODES
        .iter()
        .find(|(name, _)| *name == id)
        .map(|(_, code)| *code as usize)
}

/// Def by stable id. Panics if the id is misspelled, which is what the
/// TypeScript's `throw` was — a mob id is a compile-time fact about content, not
/// user input.
pub fn def_by_id(id: &str) -> &'static MobDef {
    let code = def_index(id).unwrap_or_else(|| panic!("unknown mob id: {id}"));
    &MOB_DEFS[code]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_indexed_by_code() {
        assert_eq!(MOB_DEFS.len(), MOB_COUNT);
        for (i, d) in MOB_DEFS.iter().enumerate() {
            assert_eq!(d.code as usize, i, "{} is not at its own code", d.id);
            assert_eq!(d.id, MOB_IDS[i]);
        }
    }

    #[test]
    fn the_body_is_a_whole_number_of_cells_and_the_art_never_undercuts_it() {
        let cs = CELL_SIZE as f32;
        for d in MOB_DEFS.iter() {
            assert_eq!(d.w_px % cs, 0.0, "{} body width is not whole cells", d.id);
            assert_eq!(d.h_px % cs, 0.0, "{} body height is not whole cells", d.id);
            assert!(
                d.art_w_px >= d.w_px,
                "{} art is narrower than its body",
                d.id
            );
            assert!(
                d.art_h_px >= d.h_px,
                "{} art is shorter than its body",
                d.id
            );
            // The overhang is centred horizontally and all-on-top vertically.
            assert_eq!(d.art_pad_x_px * 2.0, d.art_w_px - d.w_px, "{}", d.id);
            assert_eq!(d.art_pad_top_px, d.art_h_px - d.h_px, "{}", d.id);
        }
    }

    #[test]
    fn body_scale_is_anchored_on_the_body_and_nothing_else() {
        // Restated at the width `build` does it at — see the WIDTH note there.
        // The claim is about the ANCHOR (the body height, never the art's), and
        // it would be a different claim if the two sides rounded differently.
        for d in MOB_DEFS.iter() {
            let want = f64::from(PHYS_SCALE) * (f64::from(d.h_px) / f64::from(PLAYER_H));
            assert_eq!(d.body_scale, want as f32, "{}", d.id);
        }
    }

    #[test]
    fn the_bands_a_def_lists_are_the_bits_the_flat_table_holds() {
        // The picker reads MOB_BANDS; everything else reads `def.bands`. They
        // are two spellings of one fact and must not drift.
        for d in MOB_DEFS.iter() {
            let from_list = d
                .bands
                .iter()
                .fold(Band::empty(), |acc, b| acc | band_bit(*b));
            assert_eq!(
                from_list.bits() as u8,
                MOB_BANDS[d.code as usize],
                "{} bands disagree",
                d.id
            );
        }
    }

    #[test]
    fn band_at_depth_walks_surface_to_deep() {
        assert_eq!(band_at_depth(0), MobBand::Surface);
        assert_eq!(band_at_depth(BAND_SURFACE_MAX - 1), MobBand::Surface);
        assert_eq!(band_at_depth(BAND_SURFACE_MAX), MobBand::Shallow);
        assert_eq!(band_at_depth(CAVERN_DEPTH - 1), MobBand::Shallow);
        assert_eq!(band_at_depth(CAVERN_DEPTH), MobBand::Cavern);
        assert_eq!(band_at_depth(DEEP_DEPTH - 1), MobBand::Cavern);
        assert_eq!(band_at_depth(DEEP_DEPTH), MobBand::Deep);
        assert_eq!(band_at_depth(i32::MAX), MobBand::Deep);
    }

    #[test]
    fn lookups_are_total() {
        assert_eq!(def_index("no_such_mob"), None);
        assert_eq!(def_index("grubling"), Some(0));
        assert_eq!(def_by_id("grubling").code, mob::GRUBLING);
    }
}

//! The one baked sprite. The player, item icons and every creature draw through
//! this.
//!
//! A 1:1 port of three TypeScript files, merged the way they were always meant
//! to be read together:
//!
//! | TypeScript | Here |
//! |---|---|
//! | `src/sprite/SpriteArt.ts` | [`Pose`], [`PlayMode`], [`SpriteClock`], [`SeqSpec`], [`SpriteArt`], [`split_frames`], [`pick_frame_index`] |
//! | `src/sprite/Sprite.ts` | [`VARIANT_TINT`], [`BakedSprite`] |
//! | `src/sprite/fromContent.ts` | [`ContentArt`], [`ContentSeq`], [`FromContentOpts`], [`sprite_art_from_content`] |
//!
//! Their headers are the design document for this module and most of what they
//! say is reproduced below, next to the code it explains. Three things are worth
//! reading before calling anything here.
//!
//! # ONE SPRITE PIXEL IS ONE WORLD CELL
//!
//! The invariant the whole system exists to protect. A frame is `cells_h` strings
//! of `cells_w` characters; it bakes to exactly `cells_w x cells_h` texels and is
//! drawn, nearest-neighbour, into a rect of `cells_w * CELL_SIZE` by
//! `cells_h * CELL_SIZE` world px. That is why creatures, the player and item
//! icons read at the same grain as the sand and stone they sit on instead of
//! looking like smooth stickers pasted onto a chunky world.
//!
//! Every size here is therefore a whole number of CELLS, never a scale factor: a
//! 1.1x mob would put sprite pixels at 5.5px against 5px cells and reintroduce
//! exactly the mismatch this system was built to kill. Cosmetic variation is
//! carried by TINT, which has no geometry — see [`VARIANT_TINT`].
//!
//! # RESOLVE POSES ONCE, AT LOAD
//!
//! [`BakedSprite::state_id`] maps a [`Pose`] to the dense sequence index, with
//! `fallback` already applied. Call it once when you build your facade and keep
//! the [`StateId`]; everything on the draw path takes the resolved id. The
//! TypeScript said this loudly because there `stateId` was a `Map<string, …>`
//! lookup and a per-frame call would have put a hash in the hottest loop the
//! renderer has. Here it is an array index into a 14-slot table, so the cost
//! argument is gone — but the CORRECTNESS argument is not: resolving at load is
//! what turns "content has no wall-slide sequence" into a startup failure instead
//! of a character that renders nothing on one wall, six months later.
//!
//! # BAKE ONCE
//!
//! The TypeScript rasterised every pose of every variant into a `<canvas>` in the
//! constructor, at module init, and `draw` was a single `drawImage`. Nothing was
//! allocated after construction. The same promise is kept here, in the shape Bevy
//! wants: [`SpritePlugin`] rasterises the whole sprite table once in `PreStartup`
//! and publishes [`SpriteAtlases`] — one `Handle<Image>` and one
//! `Handle<TextureAtlasLayout>` per sprite. A draw is then a tile index. Nothing
//! rasterises per frame, ever.
//!
//! # What the port changed
//!
//! - **Strings became enums.** The TypeScript keyed sequences by name; the
//!   compiler emits the same vocabulary as two closed enums (`SpriteSeqState`,
//!   twelve members, for the sprite table; `MobSpecSeqState`, three, for a mob's
//!   `art` group). [`Pose`] is their union, declared once here, and the two
//!   `From` impls are exhaustive matches — so a state added to a schema stops
//!   compiling until code decides what the thing has to be doing to reach it.
//!   That is the "red squiggle rather than an invisible player" property
//!   `PlayerArt.ts` wanted and could only approximate with a `satisfies` witness.
//! - **Throws became `Result`.** Every `throw new Error` is a [`SpriteError`].
//!   The failure POLICY is unchanged, and it is the point: malformed art is loud
//!   at load, never degraded, because a frame that bakes once bakes forever and a
//!   silent skip ships as a hole in a creature. [`SpritePlugin`] panics on one,
//!   which is what a module-init throw did.
//! - **`fromContent`'s mode check is gone**, because it cannot fail. It existed
//!   to narrow a generated `string` to the `PlayMode` union; the Rust tables
//!   already carry a closed enum, so the check is `impl From`.
//! - **`Sprite`'s runtime bounds checks are gone.** `draw` guarded
//!   `state < 0 || state >= specs.length` because it took a raw number. A
//!   [`StateId`] is opaque and only obtainable from the sprite that issued it,
//!   so the guard has nothing left to catch.
//! - **The bitmaps are one atlas per sprite, not one canvas per pose.** Same
//!   pixels, same dedup, one texture bind. See [`BakedSprite::image`].
//! - **`draw` / `drawGlow` / `drawStill` did not come across.** They were
//!   Canvas2D — `ctx.drawImage` plus a `"lighter"` composite for the damage
//!   flash and self-luminance. Here the caller spawns a Bevy entity and the
//!   engine draws it; what those three methods COMPUTED (which tile, flipped or
//!   not) is published as [`BakedSprite::tile`], [`BakedSprite::still_tile`] and
//!   [`BakedSprite::tile_uv`]. The additive passes belong to whoever owns the
//!   drawn thing, not here: `flash` is a tint on the same sprite, and the glow
//!   is a second entity above the light composite on an additive material — see
//!   [`crate::mobs`], which does both for creatures.
//!
//! # Naming
//!
//! The class is [`BakedSprite`] and not `Sprite`, because `bevy::prelude::Sprite`
//! is a component every consumer of this module also imports. The rename is
//! purely to keep `use` lists honest; it is the TypeScript's `Sprite` class.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::image::{TextureAtlas, TextureAtlasLayout};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use yugen_core::config::CELL_SIZE;
use yugen_data::mobs::{MobSpecArt, MobSpecSeq, MobSpecSeqMode, MobSpecSeqState};
use yugen_data::sprites::{
    SPRITE_COUNT, SPRITES, SpriteDef, SpriteSeq, SpriteSeqMode, SpriteSeqState,
};

// ---------------------------------------------------------------------------
// Tint — the one rendering decision content does not get to make
// ---------------------------------------------------------------------------

/// Per-individual colour tilts. THIS IS CODE, NOT ART.
///
/// Two of the same species standing side by side should not be pixel-identical,
/// and a multiplicative tilt on the palette is the cheapest honest way to get
/// that: applied once at bake time, so it costs nothing at draw time and cannot
/// drift the art off the cell grid the way a hue rotation or a per-pixel shader
/// would. Content declares `variants: n` and gets the first `n` of these; it does
/// not get to pick the numbers, because "how much may two of a kind differ" is
/// one decision for the whole game, not forty.
///
/// A module constant and not a `yugen-core/src/config` export: it describes ONE
/// algorithm — how a variant index becomes a palette — and nothing outside this
/// file can use it for anything else.
pub const VARIANT_TINT: [[f32; 3]; 3] = [[1.0, 1.0, 1.0], [1.12, 1.02, 0.9], [0.9, 0.99, 1.14]];

/// Ceiling for [`SpriteArt::variants`]. Asking for more is a loud construction
/// error, exactly as it was.
pub const VARIANT_COUNT: usize = VARIANT_TINT.len();

// ---------------------------------------------------------------------------
// The vocabulary
// ---------------------------------------------------------------------------

/// Every pose any drawable thing in the game can be in.
///
/// The union of the compiler's two host-specific state enums, declared here
/// because it is a RENDERING vocabulary: it is what `fallback` maps between and
/// what a facade names when it asks for a sequence. Content authors a state; code
/// decides what the set of states is.
///
/// The first twelve are the sprite table's (`SpriteSeqState`); [`Pose::Move`] and
/// [`Pose::Air`] are a mob's. [`Pose::Idle`] is shared, which is the whole reason
/// this is a union rather than two parallel systems — the borrowing rule "a mob
/// with no air pose uses its walk cycle" is `Air -> Move`, one pair of the same
/// type, and could not be spelled at all if the two vocabularies stayed apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pose {
    Idle,
    Run,
    Skid,
    Jump,
    DoubleJump,
    Fall,
    Land,
    Dash,
    WallSlide,
    Swim,
    Punch,
    Hurt,
    /// A creature's walk cycle. The mob vocabulary's, not the player's — the
    /// player's equivalent is [`Pose::Run`].
    Move,
    /// A creature off the ground.
    Air,
}

/// How many [`Pose`] variants there are — the width of a sprite's id table.
const POSE_COUNT: usize = 14;

impl Pose {
    /// Dense index into the id table. `as usize` on the enum would do it, but
    /// only silently; this is the one place the two are required to agree and the
    /// only place the cast lives.
    const fn index(self) -> usize {
        self as usize
    }
}

impl From<SpriteSeqState> for Pose {
    /// The sprite table's vocabulary. Exhaustive on purpose: a state added to
    /// `crates/contentc/src/schemas/` fails to compile here until this module
    /// says what it is.
    fn from(s: SpriteSeqState) -> Pose {
        match s {
            SpriteSeqState::Idle => Pose::Idle,
            SpriteSeqState::Run => Pose::Run,
            SpriteSeqState::Skid => Pose::Skid,
            SpriteSeqState::Jump => Pose::Jump,
            SpriteSeqState::DoubleJump => Pose::DoubleJump,
            SpriteSeqState::Fall => Pose::Fall,
            SpriteSeqState::Land => Pose::Land,
            SpriteSeqState::Dash => Pose::Dash,
            SpriteSeqState::WallSlide => Pose::WallSlide,
            SpriteSeqState::Swim => Pose::Swim,
            SpriteSeqState::Punch => Pose::Punch,
            SpriteSeqState::Hurt => Pose::Hurt,
        }
    }
}

impl From<MobSpecSeqState> for Pose {
    /// A mob's `art` vocabulary. Three poses, and `Idle` is the same pose the
    /// player's `Idle` is — see [`Pose`].
    fn from(s: MobSpecSeqState) -> Pose {
        match s {
            MobSpecSeqState::Idle => Pose::Idle,
            MobSpecSeqState::Move => Pose::Move,
            MobSpecSeqState::Air => Pose::Air,
        }
    }
}

/// How a sequence advances.
///
/// The set is closed on purpose — every mode below is something the game already
/// needed, and a new one is a code change reviewed against the whole system
/// rather than a field content can invent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlayMode {
    /// Single frame; ignores the clock entirely.
    Hold,
    /// Free-running off [`SpriteClock::state_t`].
    Loop,
    /// One-shot; holds the last frame forever.
    Once,
    /// Indexed by an externally-driven 0..1 cadence (the run cycle).
    Phase,
    /// Off [`SpriteClock::clock_t`], with a blink on the LAST frame.
    Ambient,
}

impl From<SpriteSeqMode> for PlayMode {
    fn from(m: SpriteSeqMode) -> PlayMode {
        match m {
            SpriteSeqMode::Hold => PlayMode::Hold,
            SpriteSeqMode::Loop => PlayMode::Loop,
            SpriteSeqMode::Once => PlayMode::Once,
            SpriteSeqMode::Phase => PlayMode::Phase,
            SpriteSeqMode::Ambient => PlayMode::Ambient,
        }
    }
}

impl From<MobSpecSeqMode> for PlayMode {
    fn from(m: MobSpecSeqMode) -> PlayMode {
        match m {
            MobSpecSeqMode::Hold => PlayMode::Hold,
            MobSpecSeqMode::Loop => PlayMode::Loop,
            MobSpecSeqMode::Once => PlayMode::Once,
            MobSpecSeqMode::Phase => PlayMode::Phase,
            MobSpecSeqMode::Ambient => PlayMode::Ambient,
        }
    }
}

/// The clock is OWNED AND MUTATED BY THE CALLER, never by the sprite.
///
/// That is the one design decision that made `draw` allocation-free at 60fps with
/// a screenful of mobs, and it survives the port for the same reason: the entity
/// already tracks these numbers for its own logic, so the sprite borrows the
/// struct instead of building state per draw. [`BakedSprite`] never keeps one and
/// never writes one.
///
/// Three fields, three different notions of time, and they are not
/// interchangeable.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpriteClock {
    /// Seconds since the current state was ENTERED. Resets on transition, which
    /// is what makes [`PlayMode::Once`] fire from its first frame every time.
    pub state_t: f32,
    /// Free-running seconds, never reset. Ambient loops read this so two idle
    /// creatures breathe out of sync with each other's state changes.
    pub clock_t: f32,
    /// 0..1, advanced from something physical (real horizontal speed, for the run
    /// cycle). Decoupling cadence from wall-clock is what stops the legs sliding
    /// when the body is moving at half speed.
    pub phase: f32,
}

/// One frame: one string per cell ROW, one character per cell COLUMN.
///
/// `.` or `0` are transparent; `1`-`9` index the palette. Borrowed from the
/// generated tables rather than copied — every frame in the game is a
/// `&'static str` in `yugen-data`, and the rasteriser reads each exactly once.
pub type Frame = Vec<&'static str>;

/// One playable sequence: the frames, and the rule for stepping through them.
#[derive(Clone, Debug, PartialEq)]
pub struct SeqSpec {
    pub frames: Vec<Frame>,
    pub mode: PlayMode,
    /// Frames per second. Ignored by [`PlayMode::Hold`] and [`PlayMode::Phase`].
    pub fps: f32,
    /// [`PlayMode::Ambient`] only: seconds between blinks. 0 = never blink, in
    /// which case NO frame is reserved and the sequence degenerates to a plain
    /// loop over all of them — leaving a permanently-invisible last frame would be
    /// a silent art bug, so the mode gives it back rather than hiding it.
    pub blink_every: f32,
    /// [`PlayMode::Ambient`] only: how long one blink lasts, in seconds.
    pub blink_for: f32,
}

/// A complete art table for one drawable thing — mob, player, or item icon.
///
/// Note what is NOT here: no per-state names baked into the type, no tint values,
/// no fallback rules authored alongside the art. Content declares geometry,
/// palette, frames and cadence; everything that is a *rendering decision* stays in
/// code where it can be changed for all forty-odd species at once.
#[derive(Clone, Debug, PartialEq)]
pub struct SpriteArt {
    pub cells_w: u32,
    pub cells_h: u32,
    /// Art pixels per world cell, per axis. 1 everywhere except an experiment:
    /// the drawn rect stays `cells * CELL_SIZE` world px, so a grain of 2 packs
    /// four texels into every cell of the SAME silhouette. See the schema's
    /// field doc for why this exists and why it must not quietly spread.
    pub grain: u32,
    /// `"#rrggbb"` per entry. Index 0 is the transparent slot and is never parsed
    /// — content writes `"."` there, and every existing art table does.
    pub pal: &'static [&'static str],
    /// How many tinted copies to bake. 1 = no variation.
    ///
    /// Content says HOW MANY, code says WHICH TILTS (see [`VARIANT_TINT`]). The
    /// player and 50 item icons have no crowd to distinguish, and at 1 they pay
    /// for exactly one bake per pose.
    pub variants: usize,
    /// Dense, in authoring order. `poses[i]` names `seqs[i]`.
    pub poses: Vec<Pose>,
    pub seqs: Vec<SeqSpec>,
    /// Runtime-supplied state borrowing, e.g. `[(Pose::Air, Pose::Move)]`. NOT
    /// authored.
    ///
    /// The facade that builds a `SpriteArt` from content supplies this, because
    /// "a mob with no air pose uses its walk cycle" is a rule about how the game
    /// animates creatures, not a fact about any one creature. Letting content
    /// invent borrowing rules would mean forty files each free to disagree about
    /// what happens when a state is missing.
    pub fallback: Vec<(Pose, Pose)>,
}

// ---------------------------------------------------------------------------
// Failure
// ---------------------------------------------------------------------------

/// Everything the TypeScript threw, as a value.
///
/// The policy is unchanged and is the point: frames are generated content, so a
/// frame that bakes once bakes forever, and a construction-time failure surfaces a
/// typo immediately instead of shipping it as a hole in a creature. An older
/// version of the rasteriser silently skipped bad input and that HID a real bug —
/// a short row made `charCodeAt` return `NaN`, the range test failed, and the
/// pixel was skipped, but only because the guard happened to catch `NaN`. Any
/// other stray character could have landed inside the range check as garbage.
/// Nothing but `.`, `0` and a legal palette digit gets past [`BakedSprite::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpriteError {
    /// `cells_w` or `cells_h` was zero or negative in the compiled record.
    Grid {
        who: String,
        cells_w: i32,
        cells_h: i32,
    },
    /// A palette entry is not `#rrggbb`.
    Palette { who: String, entry: String },
    /// A frame has the wrong number of rows for the art grid.
    FrameRows {
        who: String,
        frame: usize,
        rows: usize,
        expected: u32,
    },
    /// A row has the wrong number of characters for the art grid.
    RowWidth {
        who: String,
        row: usize,
        chars: usize,
        expected: u32,
    },
    /// A frame character is neither transparent nor a legal palette index.
    PaletteIndex {
        who: String,
        row: usize,
        col: usize,
        ch: char,
    },
    /// A sequence declared no frames at all.
    EmptySequence { who: String, pose: Pose },
    /// The record declared no sequences at all.
    NoSequences { who: String },
    /// The same pose was declared by two sequences.
    DuplicatePose { who: String, pose: Pose },
    /// `variants` is outside `1..=VARIANT_COUNT`.
    VariantCount { who: String, asked: usize },
    /// A `fallback` chain closed on itself.
    FallbackCycle { who: String, from: Pose },
    /// A `fallback` chain ran out without reaching an authored sequence.
    FallbackDangling { who: String, from: Pose, to: Pose },
}

impl std::fmt::Display for SpriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpriteError::Grid {
                who,
                cells_w,
                cells_h,
            } => write!(
                f,
                "{who}: art grid is {cells_w}x{cells_h} cells; both must be positive"
            ),
            SpriteError::Palette { who, entry } => {
                write!(f, "{who}: palette entry {entry:?} is not #rrggbb")
            }
            SpriteError::FrameRows {
                who,
                frame,
                rows,
                expected,
            } => write!(
                f,
                "{who}: frame {frame} has {rows} rows, expected cells_h={expected}"
            ),
            SpriteError::RowWidth {
                who,
                row,
                chars,
                expected,
            } => write!(
                f,
                "{who}: row {row} has {chars} chars, expected cells_w={expected}"
            ),
            SpriteError::PaletteIndex { who, row, col, ch } => write!(
                f,
                "{who}: row {row} col {col}: {ch:?} is not a palette index"
            ),
            SpriteError::EmptySequence { who, pose } => {
                write!(f, "{who}: sequence {pose:?} has no frames")
            }
            SpriteError::NoSequences { who } => write!(f, "{who}: sprite has no sequences"),
            SpriteError::DuplicatePose { who, pose } => {
                write!(f, "{who}: declares state {pose:?} twice")
            }
            SpriteError::VariantCount { who, asked } => write!(
                f,
                "{who}: asks for {asked} variants; the tint table has {VARIANT_COUNT}"
            ),
            SpriteError::FallbackCycle { who, from } => {
                write!(f, "{who}: fallback for {from:?} is a cycle")
            }
            SpriteError::FallbackDangling { who, from, to } => {
                write!(f, "{who}: fallback {from:?} -> {to:?} names no sequence")
            }
        }
    }
}

impl std::error::Error for SpriteError {}

// ---------------------------------------------------------------------------
// The pure animation model
// ---------------------------------------------------------------------------

/// Split one pose body into frames on blank lines, and validate the row count.
///
/// Frames are separated by a blank line so the source reads as a filmstrip. The
/// row count of every frame must match `cells_h` or the art and the collision box
/// disagree — the one error in a sprite table that renders as a plausible-looking
/// creature instead of a crash, which is why it fails here at load rather than
/// being noticed six months later. Row WIDTH is re-checked when [`BakedSprite`]
/// rasterises, because only the baker knows the palette.
pub fn split_frames(
    lines: &'static [&'static str],
    cells_h: u32,
    who: &str,
) -> Result<Vec<Frame>, SpriteError> {
    let mut out: Vec<Frame> = Vec::new();
    let mut cur: Frame = Vec::new();
    for line in lines {
        if line.is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        cur.push(line);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    for (i, frame) in out.iter().enumerate() {
        if frame.len() != cells_h as usize {
            return Err(SpriteError::FrameRows {
                who: who.to_string(),
                frame: i,
                rows: frame.len(),
                expected: cells_h,
            });
        }
    }
    Ok(out)
}

/// Resolve a sequence to a frame INDEX. Pure, integer-returning, no Bevy.
///
/// This is the entire animation model, and it is deliberately a free function:
/// pure, integer-returning, and therefore assertable without a texture, an app or
/// a GPU. [`BakedSprite::tile`] calls it and indexes an already-built array.
///
/// Every mode clamps its input to `>= 0` and wraps negatives back into range
/// rather than trusting the caller's clock, because a single frame of negative dt
/// (a window regaining focus, a debug scrub) would otherwise index out of the
/// array and blank the sprite for one frame — a bug that looks like a flicker and
/// reads like a GPU problem.
pub fn pick_frame_index(spec: &SeqSpec, clock: &SpriteClock) -> usize {
    let n = spec.frames.len();
    if n <= 1 {
        return 0;
    }

    match spec.mode {
        PlayMode::Hold => 0,

        // Cadence comes from the world (distance travelled), not from time, so a
        // half-speed run plays its contacts at half rate with no extra state.
        PlayMode::Phase => wrap_index(clock.phase * n as f32, n),

        // Holds the last frame: a one-shot that wrapped would replay its own
        // impact forever, which is precisely wrong for land/punch/double-jump.
        PlayMode::Once => {
            let raw = floor_to_i64(clock.state_t.max(0.0) * spec.fps);
            (n as i64 - 1).min(raw.max(0)) as usize
        }

        PlayMode::Ambient => ambient_index(spec, clock.clock_t),

        PlayMode::Loop => wrap_index(clock.state_t.max(0.0) * spec.fps, n),
    }
}

/// Idle: a slow breath with an occasional blink punched over the top.
///
/// The blink is the LAST frame by convention, so the breath frames are `0..n-2`
/// and adding a blink to an existing loop is an append rather than an index
/// rewrite. Both cycles are read off the SAME wrapped `t`, not off raw `clock_t`
/// — the breath therefore restarts in step with every blink instead of drifting
/// against it, which is what makes the whole idle read as one repeating gesture
/// rather than two unrelated ones beating against each other.
///
/// # Why this divides where every other mode multiplies
///
/// [`PlayMode::Loop`] and [`PlayMode::Once`] compute `floor(t * fps)`; this
/// computes `floor(t / beat)` with `beat = 1 / fps`. That is NOT an oversight.
/// The animation this replaces divided by a PERIOD (`IDLE_BREATH = 0.62`), so
/// dividing here is what makes the two exactly equivalent rather than 99.99%
/// equivalent: `t * (1/0.62)` and `t / 0.62` disagree at ties, and over an hour of
/// 60fps frame times the multiply form picks a different breath frame on eight of
/// them.
///
/// The TypeScript could close that gap completely, because in f64
/// `1/(1/0.62) === 0.62` exactly and its facade authored `fps: 1 / 0.62`. Here
/// the content compiler emits an `f32` (`1.6129032`) and the round trip is exact
/// only to about one part in `1e7`. The division form is still the right one and
/// is kept; what the port loses is the last bit of that equivalence, which is
/// invisible and is written down here rather than quietly dropped. See
/// `the_ambient_beat_divides_where_every_other_mode_multiplies`.
fn ambient_index(spec: &SeqSpec, clock_t: f32) -> usize {
    let n = spec.frames.len();
    let raw = clock_t.max(0.0);
    let period = spec.blink_every;
    let beat = 1.0 / spec.fps;

    // No blink period means no reserved frame: loop the whole strip. The NaN test
    // is not redundant — the TypeScript wrote this `!(period > 0)` precisely so
    // that a NaN period fell in HERE rather than dividing by it below, and
    // `period <= 0.0` alone would silently drop that half of the guard.
    if period.is_nan() || period <= 0.0 {
        return wrap_index(raw / beat, n);
    }

    let t = raw % period;
    if t < spec.blink_for {
        return n - 1;
    }

    let breaths = n - 1;
    if breaths <= 1 {
        return 0;
    }
    wrap_index(t / beat, breaths)
}

/// `floor(v)` wrapped into `0..n`, the way every mode above needs it.
///
/// The TypeScript spelled this `let i = Math.floor(v) % n; if (i < 0) i += n;`
/// three times over; `rem_euclid` is the same arithmetic with the sign correction
/// built in.
///
/// A non-finite `v` returns 0. In the TypeScript a `NaN` clock produced a `NaN`
/// index, `baked[…][NaN]` was `undefined`, and `draw` bailed — one blank frame.
/// Frame 0 is the authored neutral pose of every sequence in the game, so falling
/// back to it is strictly the better of the two, and it is the outcome the
/// clamping above was reaching for anyway.
fn wrap_index(v: f32, n: usize) -> usize {
    let i = v.floor();
    if !i.is_finite() {
        return 0;
    }
    (i as i64).rem_euclid(n as i64) as usize
}

/// `Math.floor` into an integer, saturating rather than wrapping on a huge input
/// and landing on 0 for a `NaN` one.
fn floor_to_i64(v: f32) -> i64 {
    let f = v.floor();
    if f.is_nan() { 0 } else { f as i64 }
}

// ---------------------------------------------------------------------------
// The baked sprite
// ---------------------------------------------------------------------------

/// A resolved sequence index, with `fallback` already applied.
///
/// Opaque, and only obtainable from [`BakedSprite::state_id`] on the sprite it
/// indexes — which is what let the TypeScript's `state < 0 || state >= len` guards
/// go. A `StateId` from one sprite used on another is a bug the way any index
/// confusion is; keep the one your facade resolved at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateId(u8);

impl StateId {
    /// The dense sequence index. For diagnostics, and for a facade that wants to
    /// store the whole resolved set as an array.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Tile index per `[variant][state][frame]`. Entries repeat where poses do.
///
/// A named alias because it is a field type, a local and a constructor result,
/// and three spellings of `Vec<Vec<Vec<u32>>>` is three chances to get the
/// nesting order wrong.
type TileGrid = Vec<Vec<Vec<u32>>>;

/// One rasterised sprite: the atlas pixels, the tile index per pose and frame,
/// and the cadence metadata that picks between them.
///
/// Constructed once. Nothing here rasterises, allocates or hashes after
/// [`BakedSprite::new`] returns — see the module header.
#[derive(Clone, Debug)]
pub struct BakedSprite {
    /// Art grid width in cells. `grain` texels per cell; see the module header.
    pub cells_w: u32,
    /// Art grid height in cells.
    pub cells_h: u32,
    /// Art pixels per cell, per axis. 1 for everything except a finer-grain
    /// experiment; the TEXEL dimensions of every tile are `cells * grain`.
    pub grain: u32,
    /// Art rect width in world px. For a mob this is also exactly the collision
    /// box.
    pub w_px: f32,
    /// Art rect height in world px.
    pub h_px: f32,
    /// How many tinted copies actually exist.
    pub variants: usize,
    /// Distinct tiles actually rasterised, and the atlas's tile count.
    ///
    /// Diagnostic as well as structural — it exists so the pose-sharing claim on
    /// [`BakedSprite::new`] is a measurable fact rather than an assertion, and so
    /// a regression that starts double-baking every sprite is visible instead of
    /// merely slow.
    pub bake_count: usize,

    /// RGBA8, `bake_count * cells_w` by `cells_h`. See [`BakedSprite::image`].
    pixels: Vec<u8>,
    tiles: TileGrid,
    /// Parallel to `tiles[v]` — cadence metadata, one copy for all variants.
    specs: Vec<SeqSpec>,
    /// Dense, in authoring order. Diagnostic; the lookup is `ids`.
    poses: Vec<Pose>,
    /// [`Pose`] -> dense state index, with `fallback` already applied.
    ids: [Option<StateId>; POSE_COUNT],
}

impl BakedSprite {
    /// Rasterise every pose of every variant, once.
    ///
    /// # The bake cache
    ///
    /// A pose repeated across sequences bakes ONCE, keyed on the frame's own text.
    /// The TypeScript's hand-written `PlayerSprite` got this by hand — a human
    /// noticed that run's passing pose is the idle exhale and assigned the same
    /// canvas to both. Content cannot do that: an author writes the same six
    /// characters in two bodies and has no way to say "this is the same bitmap".
    /// Keying on the text recovers the property automatically, for every sprite,
    /// without content knowing the cache exists. Construction-time only — nothing
    /// consults it at draw.
    ///
    /// The key includes the variant, because two tints of the same pose are two
    /// different bitmaps.
    ///
    /// # Validation
    ///
    /// Shape and every character are checked and anything illegal is an error.
    /// See [`SpriteError`] for why that is not negotiable.
    pub fn new(art: &SpriteArt, who: &str) -> Result<BakedSprite, SpriteError> {
        if art.cells_w == 0 || art.cells_h == 0 {
            return Err(SpriteError::Grid {
                who: who.to_string(),
                cells_w: art.cells_w as i32,
                cells_h: art.cells_h as i32,
            });
        }
        if art.variants < 1 || art.variants > VARIANT_COUNT {
            return Err(SpriteError::VariantCount {
                who: who.to_string(),
                asked: art.variants,
            });
        }
        debug_assert_eq!(
            art.poses.len(),
            art.seqs.len(),
            "a SpriteArt carries parallel pose and sequence lists"
        );

        let ids = build_ids(art, who)?;

        // Index 0 is the transparent slot and is never painted, so it is never
        // parsed either — content is free to write "." or "transparent" there, as
        // every existing art table does.
        let base: Vec<[u8; 3]> = art
            .pal
            .iter()
            .enumerate()
            .map(|(i, hex)| {
                if i == 0 {
                    Ok([0, 0, 0])
                } else {
                    parse_hex(hex, who)
                }
            })
            .collect::<Result<_, _>>()?;

        // `(variant, frame text) -> tile index`, and the tile pixels in tile
        // order. A `Frame` is a list of `&'static str`, so the key clones
        // pointers and not characters.
        let mut cache: HashMap<(usize, Frame), u32> = HashMap::new();
        let mut tile_pixels: Vec<Vec<u8>> = Vec::new();
        let mut tiles: TileGrid = Vec::with_capacity(art.variants);

        for (v, tint) in VARIANT_TINT.iter().enumerate().take(art.variants) {
            let pal: Vec<[u8; 3]> = base.iter().map(|c| tinted(*c, *tint)).collect();
            let mut per_state = Vec::with_capacity(art.seqs.len());
            for (s, seq) in art.seqs.iter().enumerate() {
                let mut out = Vec::with_capacity(seq.frames.len());
                for (f, frame) in seq.frames.iter().enumerate() {
                    let key = (v, frame.clone());
                    let tile = match cache.get(&key) {
                        Some(t) => *t,
                        None => {
                            let name = format!("{who}.{:?}[{f}]", art.poses[s]);
                            let px = bake_frame(
                                frame,
                                &pal,
                                art.cells_w * art.grain,
                                art.cells_h * art.grain,
                                &name,
                            )?;
                            let t = tile_pixels.len() as u32;
                            tile_pixels.push(px);
                            cache.insert(key, t);
                            t
                        }
                    };
                    out.push(tile);
                }
                per_state.push(out);
            }
            tiles.push(per_state);
        }

        Ok(BakedSprite {
            cells_w: art.cells_w,
            cells_h: art.cells_h,
            grain: art.grain,
            // The DRAWN rect is cells, not texels: a finer grain packs more
            // texels into the same world rectangle, which is the entire point.
            w_px: (art.cells_w * CELL_SIZE as u32) as f32,
            h_px: (art.cells_h * CELL_SIZE as u32) as f32,
            variants: art.variants,
            bake_count: tile_pixels.len(),
            pixels: assemble_strip(
                &tile_pixels,
                art.cells_w * art.grain,
                art.cells_h * art.grain,
            ),
            tiles,
            specs: art.seqs.clone(),
            poses: art.poses.clone(),
            ids,
        })
    }

    /// The dense sequence index for a pose, following `fallback`. `None` = absent.
    ///
    /// CALL THIS ONCE, AT LOAD, and pass the [`StateId`] to [`BakedSprite::tile`]
    /// forever after. The fallback map was applied at construction, so
    /// `state_id(Pose::Air)` on a sprite with no air sequence already returns
    /// `Move`'s index — there is no per-frame `art.air ?? art.move` left anywhere
    /// in the system.
    pub fn state_id(&self, pose: Pose) -> Option<StateId> {
        self.ids[pose.index()]
    }

    /// Which frame of a sequence a clock is showing. Pure; allocates nothing.
    pub fn frame_index(&self, state: StateId, clock: &SpriteClock) -> usize {
        pick_frame_index(&self.specs[state.index()], clock)
    }

    /// The atlas tile a clock is showing — what the TypeScript's `draw` resolved
    /// before its single `drawImage`.
    ///
    /// Feed it to [`SpriteAtlas::sprite_at`], or straight to a
    /// `TextureAtlas::index`. Facing is the caller's `Sprite::flip_x`: mirroring
    /// about the rect's own vertical axis is what the TypeScript's
    /// `translate(x + w, y); scale(-1, 1)` did by hand, and it is what makes the
    /// figure turn while staying exactly where it was — flipping about the origin
    /// would fling it across the screen.
    pub fn tile(&self, state: StateId, variant: usize, clock: &SpriteClock) -> usize {
        let f = self.frame_index(state, clock);
        self.tiles[self.wrap_variant(variant)][state.index()][f] as usize
    }

    /// Frame 0 of a state, no clock — item icons in a hotbar, bestiary portraits.
    ///
    /// A still caller has no clock to own and no business inventing one, and
    /// frame 0 is the authored neutral pose of every sequence in the game by
    /// convention. The TypeScript's `drawStill` was also always unflipped, because
    /// UI has no facing; here that is the caller leaving `flip_x` false.
    pub fn still_tile(&self, state: StateId, variant: usize) -> usize {
        self.tiles[self.wrap_variant(variant)][state.index()][0] as usize
    }

    /// A tile's sub-rectangle in the strip, as `(u0, v0, du, dv)`.
    ///
    /// The same tile [`BakedSprite::tile`] returns, addressed the other way. A
    /// [`Sprite`] takes an atlas INDEX and the engine does this arithmetic; a
    /// mesh takes uv, and the additive glow pass is a mesh because a sprite has
    /// no per-sprite blend state to make additive. Both draws must land on the
    /// same texels or the glow shows as a fringe, so the strip's layout is
    /// published here rather than recomputed by the one caller that needs it.
    ///
    /// `flip_x` mirrors by walking the tile backwards — a negative `du` from the
    /// far edge — which is the same mirror `Sprite::flip_x` performs and, like
    /// it, is about the tile's own axis rather than the texture's.
    pub fn tile_uv(&self, tile: usize, flip_x: bool) -> Vec4 {
        // A strip is one row, so a tile's width is its share of the whole and
        // `v` always spans it. `bake_count` cannot be zero — a sprite with no
        // sequences is a construction error — but a division is not worth the
        // asymmetry with the rest of this type's total functions.
        let du = 1.0 / self.bake_count.max(1) as f32;
        let u0 = tile as f32 * du;
        if flip_x {
            Vec4::new(u0 + du, 0.0, -du, 1.0)
        } else {
            Vec4::new(u0, 0.0, du, 1.0)
        }
    }

    /// The first authored sequence. What an icon with exactly one pose wants, and
    /// what `drawStill`'s `state = 0` default meant.
    pub fn first_state(&self) -> StateId {
        StateId(0)
    }

    /// The poses this sprite actually authored, dense and in authoring order.
    /// `poses()[id.index()]` names the sequence `id` resolves to.
    pub fn poses(&self) -> &[Pose] {
        &self.poses
    }

    /// The cadence metadata for a sequence.
    pub fn spec(&self, state: StateId) -> &SeqSpec {
        &self.specs[state.index()]
    }

    /// The atlas pixels: RGBA8, `bake_count * cells_w` by `cells_h`.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// The atlas texture size in texels.
    pub fn atlas_size(&self) -> UVec2 {
        UVec2::new(
            self.cells_w * self.grain * self.bake_count as u32,
            self.cells_h * self.grain,
        )
    }

    /// Variants are assigned from entity ids and hashes, so wrap rather than
    /// trust.
    fn wrap_variant(&self, variant: usize) -> usize {
        variant % self.variants
    }

    // -- The Bevy half ------------------------------------------------------

    /// The atlas as an image asset.
    ///
    /// # Why one strip and not one texture per pose
    ///
    /// One texture per pose is one bind per pose. A strip is one bind for the
    /// whole creature and turns "which frame" into an integer the engine already
    /// knows how to index. It stays a single ROW rather than a square grid because
    /// that makes a tile's x offset `tile * cells_w` and nothing else — the widest
    /// sprite in the game is four cells and the busiest has under forty tiles, so
    /// the strip is a couple of hundred texels against an 8192-texel limit on
    /// every target this runs on.
    ///
    /// `Rgba8UnormSrgb`, because the palette is authored as sRGB hex and a
    /// `Sprite`'s colour tint multiplies in linear — the same format
    /// [`crate::effects`] bakes its wash ramp in, for the same reason.
    ///
    /// `RENDER_WORLD` only: the bake happens once and nothing reads the pixels
    /// back, so the CPU copy is dropped after upload. That is the whole
    /// allocation story, and the contrast is `crate::cellmap`'s id texture, which
    /// is `MAIN_WORLD | RENDER_WORLD` precisely because it is rewritten every
    /// frame.
    ///
    /// Sampling is nearest, from `ImagePlugin::default_nearest()` in the binary.
    /// It has to be: adjacent tiles share an edge in the strip, and any filtering
    /// wider than a texel would bleed one pose into the next. It also has to be
    /// for the reason every sampler in this game is nearest — the art is pixel
    /// art, upscaled with hard edges.
    pub fn image(&self) -> Image {
        let size = self.atlas_size();
        Image::new(
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            self.pixels.clone(),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    /// The atlas layout: `bake_count` tiles across, one down.
    pub fn layout(&self) -> TextureAtlasLayout {
        TextureAtlasLayout::from_grid(
            UVec2::new(self.cells_w * self.grain, self.cells_h * self.grain),
            self.bake_count as u32,
            1,
            None,
            None,
        )
    }
}

/// Rasterise one frame into a fresh RGBA8 tile. The dimensions arrive in
/// TEXELS — `cells * grain` — and a frame's text must match them exactly.
fn bake_frame(
    frame: &Frame,
    pal: &[[u8; 3]],
    cells_w: u32,
    cells_h: u32,
    who: &str,
) -> Result<Vec<u8>, SpriteError> {
    if frame.len() != cells_h as usize {
        return Err(SpriteError::FrameRows {
            who: who.to_string(),
            frame: 0,
            rows: frame.len(),
            expected: cells_h,
        });
    }
    let mut out = vec![0u8; (cells_w * cells_h * 4) as usize];

    for (row, line) in frame.iter().enumerate() {
        // Counted rather than measured with `len()`, which is bytes: content is
        // ASCII so the two agree today, and this keeps agreeing if it ever is not.
        let chars = line.chars().count();
        if chars != cells_w as usize {
            return Err(SpriteError::RowWidth {
                who: who.to_string(),
                row,
                chars,
                expected: cells_w,
            });
        }
        for (col, ch) in line.chars().enumerate() {
            if ch == '.' || ch == '0' {
                continue;
            }
            // '0'..'9' -> 0..9. Any other character lands outside the range and is
            // rejected rather than painted with whatever colour was last set,
            // which is exactly the bug the original's header describes.
            let idx = (ch as u32).wrapping_sub('0' as u32) as usize;
            if idx < 1 || idx >= pal.len() {
                return Err(SpriteError::PaletteIndex {
                    who: who.to_string(),
                    row,
                    col,
                    ch,
                });
            }
            let c = pal[idx];
            let px = ((row * cells_w as usize) + col) * 4;
            out[px] = c[0];
            out[px + 1] = c[1];
            out[px + 2] = c[2];
            out[px + 3] = 255;
        }
    }
    Ok(out)
}

/// Lay the baked tiles out left to right into one RGBA8 strip.
fn assemble_strip(tiles: &[Vec<u8>], cells_w: u32, cells_h: u32) -> Vec<u8> {
    let strip_w = cells_w as usize * tiles.len();
    let mut out = vec![0u8; strip_w * cells_h as usize * 4];
    for (t, tile) in tiles.iter().enumerate() {
        for row in 0..cells_h as usize {
            let src = row * cells_w as usize * 4;
            let dst = (row * strip_w + t * cells_w as usize) * 4;
            let n = cells_w as usize * 4;
            out[dst..dst + n].copy_from_slice(&tile[src..src + n]);
        }
    }
    out
}

/// `#rrggbb` and nothing else.
fn parse_hex(hex: &str, who: &str) -> Result<[u8; 3], SpriteError> {
    let bad = || SpriteError::Palette {
        who: who.to_string(),
        entry: hex.to_string(),
    };
    let b = hex.as_bytes();
    if b.len() != 7 || b[0] != b'#' {
        return Err(bad());
    }
    let mut out = [0u8; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        let s = &hex[1 + i * 2..3 + i * 2];
        *slot = u8::from_str_radix(s, 16).map_err(|_| bad())?;
    }
    Ok(out)
}

/// One palette entry through one variant tilt.
///
/// Saturating, not wrapping: `0xff * 1.12` is 285, and letting that wrap would
/// turn the brightest highlight in a palette into a dark smear on exactly the
/// variant meant to be the warmest.
fn tinted(rgb: [u8; 3], mul: [f32; 3]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (rgb[i] as f32 * mul[i]).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Build the pose -> index table, resolving `fallback` chains at construction.
///
/// Authored poses always win: a fallback entry for a pose that genuinely exists is
/// ignored rather than shadowing it. A fallback that leads nowhere is an error —
/// it is a facade bug, and the alternative (returning `None` and drawing nothing)
/// is an invisible creature, which is the single hardest rendering bug to trace
/// back to its cause.
fn build_ids(art: &SpriteArt, who: &str) -> Result<[Option<StateId>; POSE_COUNT], SpriteError> {
    let mut ids: [Option<StateId>; POSE_COUNT] = [None; POSE_COUNT];
    for (i, pose) in art.poses.iter().enumerate() {
        if ids[pose.index()].is_some() {
            return Err(SpriteError::DuplicatePose {
                who: who.to_string(),
                pose: *pose,
            });
        }
        ids[pose.index()] = Some(StateId(i as u8));
    }

    for (from, first) in art.fallback.iter().copied() {
        if ids[from.index()].is_some() {
            continue; // authored wins
        }
        // Chase the chain until it lands on a real sequence. Bounded by a seen
        // mask so `[(A, B), (B, A)]` is an error instead of a hang at load. Only
        // fourteen poses exist, so the mask is a `u16` and the walk allocates
        // nothing.
        let mut seen: u16 = 1 << from.index();
        let mut cur = from;
        let mut next = Some(first);
        while let Some(step) = next {
            cur = step;
            if ids[step.index()].is_some() {
                break;
            }
            if seen & (1 << step.index()) != 0 {
                return Err(SpriteError::FallbackCycle {
                    who: who.to_string(),
                    from,
                });
            }
            seen |= 1 << step.index();
            next = art
                .fallback
                .iter()
                .find(|(f, _)| *f == step)
                .map(|(_, t)| *t);
        }
        match ids[cur.index()] {
            Some(id) => ids[from.index()] = Some(id),
            None => {
                return Err(SpriteError::FallbackDangling {
                    who: who.to_string(),
                    from,
                    to: first,
                });
            }
        }
    }
    Ok(ids)
}

// ---------------------------------------------------------------------------
// The content bridge
// ---------------------------------------------------------------------------

/// One compiled sequence, whichever host printed it.
///
/// The identical schema fragment is emitted under two names — `SpriteSeq` for the
/// standalone sprite kind and `MobSpecSeq` nested under a mob's `art` group — and
/// a future host would add a third. The TypeScript solved this by declaring the
/// shape structurally and relying on duck typing; Rust is nominal, so the same
/// idea is a trait with two trivial impls. Binding the bridge to one of the
/// concrete names would make it arbitrarily unusable by the other host, which is
/// the exact thing the original went out of its way to avoid.
pub trait ContentSeq {
    fn pose(&self) -> Pose;
    fn mode(&self) -> PlayMode;
    /// 0 means "unset" — see [`sprite_art_from_content`].
    fn fps(&self) -> f32;
    fn blink_every(&self) -> f32;
    fn blink_for(&self) -> f32;
    fn frames(&self) -> &'static [&'static str];
}

/// One compiled art record, whichever host printed it. See [`ContentSeq`].
pub trait ContentArt {
    type Seq: ContentSeq + 'static;
    fn cells_w(&self) -> i32;
    fn cells_h(&self) -> i32;
    /// Art pixels per world cell. Default 1 — the invariant — so a host whose
    /// compiled record predates the field, or never authors it, is unchanged.
    fn grain(&self) -> i32 {
        1
    }
    fn pal(&self) -> &'static [&'static str];
    /// The sprite-wide rate a sequence's `fps: 0` inherits.
    fn fps(&self) -> f32;
    fn variants(&self) -> i32;
    fn seq(&self) -> &'static [Self::Seq];
}

impl ContentSeq for SpriteSeq {
    fn pose(&self) -> Pose {
        self.state.into()
    }
    fn mode(&self) -> PlayMode {
        self.mode.into()
    }
    fn fps(&self) -> f32 {
        self.fps
    }
    fn blink_every(&self) -> f32 {
        self.blink_every
    }
    fn blink_for(&self) -> f32 {
        self.blink_for
    }
    fn frames(&self) -> &'static [&'static str] {
        self.frames
    }
}

impl ContentArt for SpriteDef {
    fn grain(&self) -> i32 {
        self.grain
    }
    type Seq = SpriteSeq;
    fn cells_w(&self) -> i32 {
        self.cells_w
    }
    fn cells_h(&self) -> i32 {
        self.cells_h
    }
    fn pal(&self) -> &'static [&'static str] {
        self.pal
    }
    fn fps(&self) -> f32 {
        self.fps
    }
    fn variants(&self) -> i32 {
        self.variants
    }
    fn seq(&self) -> &'static [SpriteSeq] {
        self.seq.unwrap_or(&[])
    }
}

impl ContentSeq for MobSpecSeq {
    fn pose(&self) -> Pose {
        self.state.into()
    }
    fn mode(&self) -> PlayMode {
        self.mode.into()
    }
    fn fps(&self) -> f32 {
        self.fps
    }
    fn blink_every(&self) -> f32 {
        self.blink_every
    }
    fn blink_for(&self) -> f32 {
        self.blink_for
    }
    fn frames(&self) -> &'static [&'static str] {
        self.frames
    }
}

impl ContentArt for MobSpecArt {
    fn grain(&self) -> i32 {
        self.grain.unwrap_or(1)
    }
    type Seq = MobSpecSeq;
    fn cells_w(&self) -> i32 {
        self.cells_w
    }
    fn cells_h(&self) -> i32 {
        self.cells_h
    }
    fn pal(&self) -> &'static [&'static str] {
        self.pal
    }
    /// A mob's art group makes `fps` optional where the sprite kind does not, so
    /// an absent rate is the same "unset" a sequence's `fps: 0` is. Defaulting to
    /// 0 rather than to 8 keeps the two spellings of unset behaving alike.
    fn fps(&self) -> f32 {
        self.fps.unwrap_or(0.0)
    }
    /// Absent means one copy, which is what content declaring nothing about
    /// crowds should mean.
    fn variants(&self) -> i32 {
        self.variants.unwrap_or(1)
    }
    fn seq(&self) -> &'static [MobSpecSeq] {
        self.seq.unwrap_or(&[])
    }
}

/// What the CALLER supplies that content is not allowed to.
#[derive(Clone, Copy, Debug, Default)]
pub struct FromContentOpts<'a> {
    /// State borrowing, e.g. `&[(Pose::Air, Pose::Move)]`. Chains are chased and
    /// validated by [`BakedSprite::new`], which fails on a cycle or a dangling
    /// target.
    ///
    /// A slice and not a map: there are at most a handful of rules, they are
    /// walked once at load, and a `HashMap` would be an allocation and a hash to
    /// beat a two-element linear scan.
    pub fallback: &'a [(Pose, Pose)],
    /// Per-pose playback-rate scale, applied AFTER the `fps: 0` inheritance.
    ///
    /// This exists for exactly one caller: mobs play their idle loop at half the
    /// authored rate, a rule the old `MobSprite` hard-coded. It is a property of
    /// how creatures read, not of any creature's art, so it stays out of content —
    /// and it is a multiplier rather than an override so a sequence that DID set
    /// its own rate is scaled rather than overwritten.
    pub rate_scale: &'a [(Pose, f32)],
    /// Override the baked variant count. Mobs pass [`VARIANT_COUNT`].
    pub variants: Option<usize>,
}

/// Build a runtime [`SpriteArt`] from a compiled content record.
///
/// This is the one adapter, and it exists once rather than three times because the
/// player facade, the item-icon registry and the mob facade all need the identical
/// transformation:
///
/// ```text
/// generated seq[]  ->  parallel poses[] + seqs[]  ->  SpriteArt
/// ```
///
/// Two things happen here that content deliberately does not get to express:
///
/// **`fps: 0` inherits the sprite-wide rate.** The schema defaults a sequence's
/// fps to 0 rather than to 8 so that "unset" is distinguishable from "deliberately
/// 8" — otherwise retuning a sprite's base rate would silently skip every sequence
/// that happened to have been written at the old default. The scale then applies to
/// whichever rate won, so a mob's idle is halved whether or not it named its own
/// fps.
///
/// **`fallback` is supplied by the CALLER, not authored.** See
/// [`FromContentOpts::fallback`].
///
/// Fails on malformed content rather than degrading: a frame with the wrong row
/// count or a sequence with no frames is a content bug that would otherwise show up
/// as a creature that silently animates wrong, which is far more expensive to track
/// down than a loud failure at load.
pub fn sprite_art_from_content<A: ContentArt>(
    art: &A,
    who: &str,
    opts: &FromContentOpts<'_>,
) -> Result<SpriteArt, SpriteError> {
    let (cw, ch) = (art.cells_w(), art.cells_h());
    if cw <= 0 || ch <= 0 {
        return Err(SpriteError::Grid {
            who: who.to_string(),
            cells_w: cw,
            cells_h: ch,
        });
    }
    let (cells_w, cells_h) = (cw as u32, ch as u32);
    // Clamped by the schema to 1..4; a 0 from a hand-built record would divide
    // the art out of existence, so it is floored here too.
    let grain = art.grain().max(1) as u32;

    let mut poses = Vec::new();
    let mut seqs = Vec::new();

    for s in art.seq() {
        let pose = s.pose();
        let frames = split_frames(s.frames(), cells_h * grain, &format!("{who}.{pose:?}"))?;
        if frames.is_empty() {
            return Err(SpriteError::EmptySequence {
                who: who.to_string(),
                pose,
            });
        }
        let base = if s.fps() == 0.0 { art.fps() } else { s.fps() };
        let scale = opts
            .rate_scale
            .iter()
            .find(|(p, _)| *p == pose)
            .map_or(1.0, |(_, k)| *k);

        poses.push(pose);
        seqs.push(SeqSpec {
            frames,
            mode: s.mode(),
            fps: base * scale,
            blink_every: s.blink_every(),
            blink_for: s.blink_for(),
        });
    }

    if seqs.is_empty() {
        return Err(SpriteError::NoSequences {
            who: who.to_string(),
        });
    }

    Ok(SpriteArt {
        cells_w,
        cells_h,
        grain,
        pal: art.pal(),
        variants: opts
            .variants
            .unwrap_or_else(|| art.variants().max(0) as usize),
        poses,
        seqs,
        fallback: opts.fallback.to_vec(),
    })
}

/// Sprite code for an authoring id, or `None`.
///
/// The sprite table is index == code, so this is the one string comparison in the
/// system and it is meant to be done ONCE, at load — `crate::ui` resolving an
/// item's icon id, a facade resolving `"player"`. Fifty-one entries with a length
/// check first, so it is cheap; a per-frame call would still be a string compare
/// in a draw loop for a value that never changes.
///
/// `None` is a first-class answer. `yugen_core::items::ITEM_ICONS` already
/// validated its ids against this same table, so a miss there means the generated
/// modules are out of step with each other — and it degrades to "no icon" rather
/// than reaching the renderer, exactly as the TypeScript did.
pub fn sprite_code_of(id: &str) -> Option<u16> {
    SPRITES.iter().find(|d| d.id == id).map(|d| d.code)
}

// ---------------------------------------------------------------------------
// The Bevy plumbing
// ---------------------------------------------------------------------------

/// One sprite's baked art plus the two handles a `Sprite` component needs.
#[derive(Clone, Debug)]
pub struct SpriteAtlas {
    /// The strip. One bind for the whole sprite.
    pub image: Handle<Image>,
    /// `bake_count` tiles across, one down.
    pub layout: Handle<TextureAtlasLayout>,
    /// The rasteriser's own output: tile lookup, cadence, geometry.
    pub baked: BakedSprite,
}

impl SpriteAtlas {
    /// A [`TextureAtlas`] pointing at one tile.
    pub fn atlas_at(&self, tile: usize) -> TextureAtlas {
        TextureAtlas {
            layout: self.layout.clone(),
            index: tile,
        }
    }

    /// A ready `Sprite` component showing one tile at one texel per world cell.
    ///
    /// `custom_size` is the art rect in world px — `cells * CELL_SIZE` — which is
    /// what keeps the blit at exactly one sprite pixel per world cell. A caller
    /// that wants squash-stretch (the player's landing squash, its rise stretch)
    /// overwrites `custom_size` afterwards; that is what the TypeScript's explicit
    /// `w`/`h` arguments to `draw` were for, and why they were not just `wPx`/`hPx`.
    pub fn sprite_at(&self, tile: usize, flip_x: bool) -> Sprite {
        Sprite {
            image: self.image.clone(),
            texture_atlas: Some(self.atlas_at(tile)),
            flip_x,
            custom_size: Some(Vec2::new(self.baked.w_px, self.baked.h_px)),
            ..default()
        }
    }

    /// Frame 0 of the first authored sequence, unflipped — the still an item icon
    /// or a bestiary portrait wants. The TypeScript's `drawStill` with both its
    /// defaults.
    pub fn still(&self) -> Sprite {
        self.sprite_at(self.baked.still_tile(self.baked.first_state(), 0), false)
    }
}

/// Every sprite in the table, rasterised and uploaded.
///
/// Indexed by sprite code, because that is what the table is: `SPRITES` is
/// index == code and so is this. [`SpriteAtlases::get`] takes an id for the callers
/// that hold one — `ITEM_ICONS` publishes ids, not codes — and is the linear scan
/// [`sprite_code_of`] describes, which belongs at load and not in a draw loop.
///
/// Every slot is `Some` in practice; the `Option` exists so a sprite that failed to
/// bake could in principle be reported without shifting every code after it. Today
/// one that fails to bake panics — see [`SpritePlugin`].
///
/// # Why this is `Clone`
///
/// So a consumer that cannot borrow the world can hold one — see [`crate::glue`],
/// where the HUD's icon source is a `Box<dyn IconAtlas>` with no access to `Res`.
/// The clone happens once, in `Startup`.
///
/// It costs **1744 bytes**, measured in `docs/PERF.md` rather than estimated.
/// This comment previously claimed "a few megabytes", which was wrong by roughly
/// a thousand times: it assumed the clone duplicated the baked CPU pixels, but by
/// `Startup` those are in `Assets<Image>` and a [`SpriteAtlas`] holds two handles
/// and six numbers. Threading an `Arc` through the plugin to save under 2 KB, once,
/// would be the worse trade — but for the opposite reason to the one recorded here
/// before.
#[derive(Resource, Default, Clone)]
pub struct SpriteAtlases {
    by_code: Vec<Option<SpriteAtlas>>,
}

impl SpriteAtlases {
    /// The atlas for a sprite code.
    pub fn by_code(&self, code: u16) -> Option<&SpriteAtlas> {
        self.by_code.get(code as usize)?.as_ref()
    }

    /// The atlas for an authoring id, or `None` if the table has no such sprite.
    ///
    /// Resolve once and keep the [`SpriteAtlas`] or the code — see
    /// [`sprite_code_of`].
    pub fn get(&self, id: &str) -> Option<&SpriteAtlas> {
        self.by_code(sprite_code_of(id)?)
    }

    /// How many sprites are loaded. Zero before [`SpritePlugin`]'s `PreStartup`
    /// bake, which is the state a test that never ran the schedule sees.
    pub fn count(&self) -> usize {
        self.by_code.iter().filter(|s| s.is_some()).count()
    }
}

/// Bakes the whole sprite table once, before anything asks for it.
///
/// # Why `PreStartup`
///
/// So that every facade built in `Startup` — `crate::ui`'s hotbar icons,
/// `crate::player_art`'s body — can take `Res<SpriteAtlases>` and find it
/// populated, with no `.after()` between them to get wrong. The bake is one pass
/// over 51 records of a few hundred texels each; it is not worth a loading state
/// and it is emphatically not worth doing lazily on a draw path.
///
/// # Why a panic is the right failure
///
/// Malformed art is a content bug the compiler let through. The TypeScript threw
/// from a module-level `new Sprite(...)`, which took the page down at load; this
/// panics out of `PreStartup`, which takes the app down at load. Both are the same
/// deliberate choice: a hole in a creature that ships is far more expensive than a
/// startup that refuses.
///
/// # Assets
///
/// `Assets<Image>` and `Assets<TextureAtlasLayout>` come from `ImagePlugin`, and
/// [`SpritePlugin::build`] registers them only if nothing already has. The guard
/// matters both ways: `init_asset` REPLACES the store rather than skipping, so
/// calling it unconditionally after `DefaultPlugins` would drop every image the app
/// had already loaded — and without it this plugin could not be driven by a
/// headless test, which is how the plumbing below is checked.
pub struct SpritePlugin;

impl Plugin for SpritePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Assets<Image>>() {
            app.init_asset::<Image>();
        }
        if !app
            .world()
            .contains_resource::<Assets<TextureAtlasLayout>>()
        {
            app.init_asset::<TextureAtlasLayout>();
        }
        app.init_resource::<SpriteAtlases>()
            .add_systems(PreStartup, bake_sprite_table);
    }
}

/// Rasterise and upload every record in `yugen_data::sprites::SPRITES`.
///
/// No `fallback`, no `rate_scale` and no `variants` override for anything in this
/// table: it holds the player, who authors every state it can reach and plays each
/// at the rate content asked for, and 50 item icons, which have one pose and no
/// crowd. Those are all decisions the CALLER makes, so a facade that wants
/// different ones calls [`sprite_art_from_content`] itself — this system is the
/// default, not the only door. Mob art wants all three and so is not baked here:
/// [`crate::mobs`] bakes the bestiary in the same `PreStartup`, through this same
/// bridge, supplying the three from `yugen-core` — see [`MobAtlases`].
///
/// [`MobAtlases`]: crate::mobs::MobAtlases
fn bake_sprite_table(
    mut atlases: ResMut<SpriteAtlases>,
    mut images: ResMut<Assets<Image>>,
    mut layouts: ResMut<Assets<TextureAtlasLayout>>,
) {
    let opts = FromContentOpts::default();
    let mut out = Vec::with_capacity(SPRITE_COUNT);
    for def in SPRITES.iter() {
        let art = sprite_art_from_content(def, def.id, &opts)
            .unwrap_or_else(|e| panic!("content/sprites: {e}"));
        let baked =
            BakedSprite::new(&art, def.id).unwrap_or_else(|e| panic!("content/sprites: {e}"));
        out.push(Some(SpriteAtlas {
            image: images.add(baked.image()),
            layout: layouts.add(baked.layout()),
            baked,
        }));
    }
    atlases.by_code = out;
}

/// THE ONE AXIS FLIP. Sim `(left, top)` and an extent, to a Bevy translation.
///
/// The simulation, the collision boxes, the cell grid and every art row are +y
/// DOWN — row 0 is the top of a creature. Bevy is +y UP. The conversion is a single
/// negation, and this is the single place in the sprite system that performs it, so
/// that no facade has to remember which way its own numbers point.
///
/// Nothing flips in the PIXELS. A texture's row 0 is already drawn at the top of a
/// quad, so a frame authored top-down arrives on screen top-down with no work;
/// inverting the atlas rows and then inverting the transform back would be two bugs
/// that cancel until one of them is fixed.
///
/// The top-left is rounded and the half-extents added afterwards, rather than the
/// centre being rounded, so an odd extent does not land the sprite's edges on half
/// pixels — the rule `crate::player::place_body` and `crate::mobs::place_mobs`
/// already snap with. A fractional edge is the one thing the low-res target cannot
/// forgive: see `crate::lowres`.
pub fn world_translation(left: f32, top: f32, w: f32, h: f32, z: f32) -> Vec3 {
    let left = left.round();
    let top = top.round();
    Vec3::new(left + w * 0.5, -(top + h * 0.5), z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::items::ITEM_ICONS;
    use yugen_data::sprites::sprite;

    /// A 2x2 art table with a two-colour palette, for the rasteriser tests.
    fn art(frames: Vec<Frame>, mode: PlayMode) -> SpriteArt {
        SpriteArt {
            cells_w: 2,
            cells_h: 2,
            grain: 1,
            pal: &[".", "#102030", "#ff8000"],
            variants: 1,
            poses: vec![Pose::Idle],
            seqs: vec![SeqSpec {
                frames,
                mode,
                fps: 10.0,
                blink_every: 0.0,
                blink_for: 0.1,
            }],
            fallback: Vec::new(),
        }
    }

    fn one_frame(rows: [&'static str; 2]) -> Frame {
        rows.to_vec()
    }

    /// The four RGBA bytes of one texel of a baked strip.
    fn texel(b: &BakedSprite, tile: usize, x: u32, y: u32) -> [u8; 4] {
        let w = b.atlas_size().x as usize;
        let i = ((y as usize * w) + tile * b.cells_w as usize + x as usize) * 4;
        [
            b.pixels()[i],
            b.pixels()[i + 1],
            b.pixels()[i + 2],
            b.pixels()[i + 3],
        ]
    }

    // -- Rasterising --------------------------------------------------------

    #[test]
    fn a_frame_bakes_one_texel_per_cell_at_the_authored_palette_colour() {
        let a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();

        assert_eq!(b.atlas_size(), UVec2::new(2, 2), "one texel per cell");
        assert_eq!(texel(&b, 0, 0, 0), [0x10, 0x20, 0x30, 255]);
        assert_eq!(texel(&b, 0, 1, 0), [0xff, 0x80, 0x00, 255]);
        assert_eq!(texel(&b, 0, 0, 1), [0xff, 0x80, 0x00, 255]);
        assert_eq!(texel(&b, 0, 1, 1), [0x10, 0x20, 0x30, 255]);
    }

    #[test]
    fn the_art_rect_is_a_whole_number_of_cells_in_world_px() {
        // The invariant the module header is about: geometry is always cells times
        // CELL_SIZE, never a scale factor.
        let a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.w_px, (2 * CELL_SIZE) as f32);
        assert_eq!(b.h_px, (2 * CELL_SIZE) as f32);
    }

    #[test]
    fn both_spellings_of_transparent_are_left_unpainted_and_fully_clear() {
        // '.' and '0' mean the same thing, and index 0 of the palette is never
        // parsed — which is why content is free to write "." there.
        let a = art(vec![one_frame([".0", "0."])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_eq!(
                texel(&b, 0, x, y),
                [0, 0, 0, 0],
                "({x},{y}) should be clear"
            );
        }
    }

    #[test]
    fn the_transparent_slot_is_never_parsed_as_a_colour() {
        // Slot 0 of every palette in the game is ".", which is not #rrggbb. If the
        // rasteriser parsed it, nothing in the game would bake at all.
        let a = art(vec![one_frame(["11", "11"])], PlayMode::Hold);
        assert!(BakedSprite::new(&a, "t").is_ok());
    }

    #[test]
    fn a_row_of_the_wrong_width_is_a_construction_error() {
        let a = art(vec![one_frame(["123", "12"])], PlayMode::Hold);
        assert_eq!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::RowWidth {
                who: "t.Idle[0]".into(),
                row: 0,
                chars: 3,
                expected: 2
            }
        );
    }

    #[test]
    fn a_frame_with_the_wrong_number_of_rows_is_a_construction_error() {
        let a = art(vec![vec!["12"]], PlayMode::Hold);
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::FrameRows {
                rows: 1,
                expected: 2,
                ..
            }
        ));
    }

    #[test]
    fn a_character_outside_the_palette_is_a_construction_error() {
        // The bug the original's header describes: 'a' is not a digit, and silently
        // skipping it would leave a hole in a creature that ships. '9' IS a digit
        // and is still out of range for a two-colour palette.
        for bad in ["a2", "92"] {
            let a = art(vec![one_frame([bad, "12"])], PlayMode::Hold);
            assert!(
                matches!(
                    BakedSprite::new(&a, "t").unwrap_err(),
                    SpriteError::PaletteIndex { row: 0, col: 0, .. }
                ),
                "{bad:?} should not bake"
            );
        }
    }

    #[test]
    fn a_palette_entry_that_is_not_six_hex_digits_is_a_construction_error() {
        for pal in [
            &[".", "#12345", "#ff8000"],
            &[".", "102030", "#ff8000"],
            &[".", "#nothex", "#ff8000"],
            &[".", "#1234567", "#ff8000"],
        ] {
            let mut a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
            a.pal = pal.as_slice();
            assert!(
                matches!(
                    BakedSprite::new(&a, "t").unwrap_err(),
                    SpriteError::Palette { .. }
                ),
                "{:?} should not parse",
                pal[1]
            );
        }
    }

    #[test]
    fn a_tiles_uv_rect_is_its_share_of_the_strip_and_mirrors_in_place() {
        // Three distinct frames, so the strip is three tiles wide.
        let mut a = art(vec![one_frame(["12", "21"])], PlayMode::Loop);
        a.seqs[0].frames = vec![
            one_frame(["12", "21"]),
            one_frame(["11", "22"]),
            one_frame(["22", "11"]),
        ];
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.bake_count, 3);

        // The middle tile is the middle third, and v always spans the strip.
        assert_eq!(
            b.tile_uv(1, false),
            Vec4::new(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0)
        );
        // Flipped, it starts at the far edge and walks back — the same texels in
        // the other order, so the figure turns without moving.
        let flipped = b.tile_uv(1, true);
        assert_eq!(flipped, Vec4::new(2.0 / 3.0, 0.0, -1.0 / 3.0, 1.0));
        let plain = b.tile_uv(1, false);
        assert_eq!(flipped.x + flipped.z, plain.x, "same span, other end");

        // Tiles tile: every one starts where the last one ended.
        assert_eq!(b.tile_uv(0, false).x, 0.0);
        assert_eq!(b.tile_uv(2, false).x + b.tile_uv(2, false).z, 1.0);
    }

    // -- Pose sharing and variants -----------------------------------------

    #[test]
    fn a_pose_repeated_across_sequences_bakes_once() {
        // The bake cache, and the reason `bake_count` is published: content has no
        // way to say "this is the same bitmap", so keying on the frame's own text
        // recovers the sharing PlayerSprite only had because a human hand-assigned
        // the same canvas twice.
        let neutral = one_frame(["12", "21"]);
        let other = one_frame(["11", "22"]);
        let mut a = art(vec![neutral.clone()], PlayMode::Loop);
        a.poses = vec![Pose::Idle, Pose::Run];
        a.seqs = vec![
            SeqSpec {
                frames: vec![neutral.clone(), other.clone()],
                mode: PlayMode::Loop,
                fps: 8.0,
                blink_every: 0.0,
                blink_for: 0.0,
            },
            // Run reuses idle's two poses, in the other order.
            SeqSpec {
                frames: vec![other, neutral],
                mode: PlayMode::Loop,
                fps: 8.0,
                blink_every: 0.0,
                blink_for: 0.0,
            },
        ];
        let b = BakedSprite::new(&a, "t").unwrap();

        assert_eq!(b.bake_count, 2, "four frames, two distinct poses");
        let idle = b.state_id(Pose::Idle).unwrap();
        let run = b.state_id(Pose::Run).unwrap();
        // Idle's frame 0 and run's frame 1 are the same bitmap, so the same tile.
        let late_run = SpriteClock {
            state_t: 0.2,
            ..default()
        };
        assert_eq!(b.frame_index(run, &late_run), 1);
        assert_eq!(b.still_tile(idle, 0), b.tile(run, 0, &late_run));
    }

    #[test]
    fn each_variant_is_the_same_pose_through_its_own_tint() {
        let mut a = art(vec![one_frame(["11", "11"])], PlayMode::Hold);
        a.variants = 3;
        let b = BakedSprite::new(&a, "t").unwrap();

        assert_eq!(b.bake_count, 3, "one pose, three tints, three tiles");
        let s = b.first_state();
        let v0 = texel(&b, b.still_tile(s, 0), 0, 0);
        let v1 = texel(&b, b.still_tile(s, 1), 0, 0);
        // Variant 0 is the identity tilt, so it is the authored colour exactly.
        assert_eq!(v0, [0x10, 0x20, 0x30, 255]);
        // Variant 1 is warmer: more red, less blue. That is the whole point of the
        // table — two of a kind must not be pixel-identical.
        assert_ne!(v0, v1);
        assert!(v1[0] > v0[0] && v1[2] < v0[2]);
    }

    #[test]
    fn a_variant_tilt_saturates_rather_than_wrapping_round_to_black() {
        // 0xff * 1.12 is 285 and 0xff * 1.02 is 260. Wrapping either would turn the
        // brightest highlight in a palette into a dark smear on exactly the variant
        // meant to be the warmest.
        let mut a = art(vec![one_frame(["22", "22"])], PlayMode::Hold);
        a.pal = &[".", "#102030", "#ffffff"];
        a.variants = 2;
        let b = BakedSprite::new(&a, "t").unwrap();
        let warm = texel(&b, b.still_tile(b.first_state(), 1), 0, 0);
        assert_eq!(
            [warm[0], warm[1]],
            [255, 255],
            "both channels this tilt brightens clamp at white"
        );
        // Blue is tilted DOWN by the same tint, so it is the one channel with
        // somewhere left to go — which is what makes the variant read warm at all.
        assert_eq!(warm[2], (255.0f32 * 0.9).round() as u8);
        assert_eq!(warm[3], 255);
    }

    #[test]
    fn asking_for_more_variants_than_the_tint_table_has_is_a_construction_error() {
        let mut a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        a.variants = VARIANT_COUNT + 1;
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::VariantCount { .. }
        ));
        a.variants = 0;
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::VariantCount { .. }
        ));
    }

    #[test]
    fn a_variant_index_from_an_entity_id_wraps_instead_of_indexing_out() {
        // Variants are assigned from entity ids and hashes, so the sprite wraps
        // rather than trusts.
        let mut a = art(vec![one_frame(["11", "11"])], PlayMode::Hold);
        a.variants = 3;
        let b = BakedSprite::new(&a, "t").unwrap();
        let s = b.first_state();
        assert_eq!(b.still_tile(s, 7), b.still_tile(s, 1));
        assert_eq!(b.still_tile(s, 300), b.still_tile(s, 0));
    }

    // -- The frame picker ---------------------------------------------------

    fn strip(n: usize) -> SeqSpec {
        SeqSpec {
            frames: (0..n).map(|_| one_frame(["12", "21"])).collect(),
            mode: PlayMode::Loop,
            fps: 10.0,
            blink_every: 0.0,
            blink_for: 0.1,
        }
    }

    #[test]
    fn a_single_frame_sequence_is_frame_zero_in_every_mode() {
        for mode in [
            PlayMode::Hold,
            PlayMode::Loop,
            PlayMode::Once,
            PlayMode::Phase,
            PlayMode::Ambient,
        ] {
            let mut s = strip(1);
            s.mode = mode;
            let clock = SpriteClock {
                state_t: 99.0,
                clock_t: 99.0,
                phase: 0.9,
            };
            assert_eq!(pick_frame_index(&s, &clock), 0, "{mode:?}");
        }
    }

    #[test]
    fn hold_ignores_the_clock_entirely() {
        let mut s = strip(4);
        s.mode = PlayMode::Hold;
        let clock = SpriteClock {
            state_t: 99.0,
            clock_t: 99.0,
            phase: 0.75,
        };
        assert_eq!(pick_frame_index(&s, &clock), 0);
    }

    #[test]
    fn loop_advances_with_the_state_clock_and_wraps() {
        let s = strip(4); // 10 fps
        for (t, want) in [
            (0.0, 0),
            (0.05, 0),
            (0.1, 1),
            (0.35, 3),
            (0.4, 0),
            (0.45, 0),
        ] {
            let clock = SpriteClock {
                state_t: t,
                ..default()
            };
            assert_eq!(pick_frame_index(&s, &clock), want, "t={t}");
        }
    }

    #[test]
    fn once_holds_its_last_frame_forever() {
        // A one-shot that wrapped would replay its own impact forever, which is
        // precisely wrong for land, punch and double-jump.
        let mut s = strip(3);
        s.mode = PlayMode::Once;
        for (t, want) in [(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 2), (100.0, 2)] {
            let clock = SpriteClock {
                state_t: t,
                ..default()
            };
            assert_eq!(pick_frame_index(&s, &clock), want, "t={t}");
        }
    }

    #[test]
    fn phase_indexes_off_distance_travelled_rather_than_time() {
        // The run cycle. A half-speed run plays its contacts at half rate with no
        // extra state, because the clock this reads is advanced by real horizontal
        // speed and not by dt — which is what stops the legs sliding.
        let mut s = strip(4);
        s.mode = PlayMode::Phase;
        for (p, want) in [(0.0, 0), (0.24, 0), (0.25, 1), (0.5, 2), (0.99, 3)] {
            let clock = SpriteClock {
                phase: p,
                state_t: 99.0,
                clock_t: 99.0,
            };
            assert_eq!(pick_frame_index(&s, &clock), want, "phase={p}");
        }
    }

    #[test]
    fn a_negative_clock_wraps_back_into_range_instead_of_blanking_the_sprite() {
        // One frame of negative dt — a window regaining focus, a debug scrub — must
        // not index out of the array. It reads as a flicker and looks like a GPU
        // problem.
        let mut s = strip(4);
        let clock = SpriteClock {
            state_t: -1.0,
            ..default()
        };
        assert_eq!(pick_frame_index(&s, &clock), 0, "loop clamps state_t at 0");

        s.mode = PlayMode::Phase;
        let back = SpriteClock {
            phase: -0.3,
            ..default()
        };
        // floor(-0.3 * 4) is -2, wrapped into 0..4 is 2.
        assert_eq!(pick_frame_index(&s, &back), 2);

        s.mode = PlayMode::Once;
        let neg = SpriteClock {
            state_t: -5.0,
            ..default()
        };
        assert_eq!(pick_frame_index(&s, &neg), 0);
    }

    #[test]
    fn a_non_finite_clock_falls_back_to_the_neutral_pose() {
        // The TypeScript produced a NaN index here, indexed `undefined`, and drew
        // nothing for one frame. Frame 0 is the authored neutral pose of every
        // sequence, so this is strictly the better of the two — see `wrap_index`.
        let mut s = strip(4);
        for mode in [
            PlayMode::Loop,
            PlayMode::Phase,
            PlayMode::Once,
            PlayMode::Ambient,
        ] {
            s.mode = mode;
            let clock = SpriteClock {
                state_t: f32::NAN,
                clock_t: f32::NAN,
                phase: f32::NAN,
            };
            assert_eq!(pick_frame_index(&s, &clock), 0, "{mode:?}");
        }
    }

    #[test]
    fn an_ambient_sequence_with_no_blink_period_loops_over_every_frame() {
        // Leaving a permanently-invisible last frame would be a silent art bug, so
        // the mode gives it back rather than hiding it.
        let mut s = strip(3);
        s.mode = PlayMode::Ambient;
        s.fps = 2.0; // beat 0.5s
        s.blink_every = 0.0;
        let seen: Vec<usize> = [0.0, 0.5, 1.0, 1.5]
            .iter()
            .map(|t| {
                pick_frame_index(
                    &s,
                    &SpriteClock {
                        clock_t: *t,
                        ..default()
                    },
                )
            })
            .collect();
        assert_eq!(seen, vec![0, 1, 2, 0]);
    }

    #[test]
    fn an_ambient_sequence_reserves_its_last_frame_for_the_blink() {
        // The blink is the LAST frame by convention, so adding one to an existing
        // loop is an append rather than an index rewrite.
        let mut s = strip(3);
        s.mode = PlayMode::Ambient;
        s.fps = 2.0; // beat 0.5s
        s.blink_every = 4.0;
        s.blink_for = 0.2;

        // Inside the blink window at the top of every period.
        for t in [0.0, 0.1, 4.0, 4.1] {
            let i = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t,
                    ..default()
                },
            );
            assert_eq!(i, 2, "t={t} is inside the blink");
        }
        // Outside it, the breath cycles over frames 0..n-2 only.
        for t in [0.3, 0.9, 1.4, 2.6, 3.9] {
            let i = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t,
                    ..default()
                },
            );
            assert!(
                i < 2,
                "the breath must not show the blink frame, t={t} gave {i}"
            );
        }
    }

    #[test]
    fn the_ambient_breath_restarts_in_step_with_every_blink() {
        // Both cycles read the SAME wrapped t, so the idle reads as one repeating
        // gesture rather than two unrelated ones beating against each other.
        let mut s = strip(3);
        s.mode = PlayMode::Ambient;
        s.fps = 2.0;
        s.blink_every = 4.0;
        s.blink_for = 0.2;
        for t in [0.3, 0.9, 1.4, 3.9] {
            let a = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t,
                    ..default()
                },
            );
            let b = pick_frame_index(
                &s,
                &SpriteClock {
                    clock_t: t + 4.0,
                    ..default()
                },
            );
            assert_eq!(a, b, "the breath should repeat with the period, t={t}");
        }
    }

    #[test]
    fn the_ambient_beat_divides_where_every_other_mode_multiplies() {
        // The old animation divided by a period of 0.62s, so the facade authors
        // `fps: 1/0.62` and this mode divides by `1/fps` to make the two
        // equivalent. In f64 that round trip was exact; the compiler emits f32, so
        // it is exact to about one part in 1e7 — the one place the port loses the
        // bit-for-bit equivalence the original had, worth eight frames an hour at
        // 60fps.
        let authored = SPRITES[sprite::PLAYER as usize].seq.unwrap()[0];
        assert_eq!(authored.state, SpriteSeqState::Idle);
        assert_eq!(authored.mode, SpriteSeqMode::Ambient);
        let beat = 1.0 / authored.fps;
        assert!(
            (beat - 0.62).abs() < 1e-6,
            "the player's idle beat should round-trip to 0.62s, was {beat}"
        );
    }

    // -- split_frames -------------------------------------------------------

    #[test]
    fn split_frames_reads_a_body_as_a_filmstrip_separated_by_blank_lines() {
        // Two blanks in a row, and a trailing one: a run of blank lines separates,
        // it does not add an empty frame, and neither does a body that ends on one.
        static BODY: [&str; 7] = ["12", "21", "", "", "11", "22", ""];
        let f = split_frames(&BODY, 2, "t").unwrap();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0], vec!["12", "21"]);
        assert_eq!(f[1], vec!["11", "22"]);
    }

    #[test]
    fn split_frames_rejects_a_frame_whose_row_count_disagrees_with_the_art_grid() {
        // The one error in a sprite table that renders as a plausible-looking
        // creature instead of a crash: the art and the collision box disagree.
        static BODY: [&str; 5] = ["12", "21", "", "11", "22"];
        assert!(matches!(
            split_frames(&BODY, 3, "t").unwrap_err(),
            SpriteError::FrameRows {
                frame: 0,
                rows: 2,
                expected: 3,
                ..
            }
        ));
    }

    #[test]
    fn split_frames_of_an_empty_body_is_no_frames_rather_than_one_empty_one() {
        static NONE: [&str; 0] = [];
        assert!(split_frames(&NONE, 2, "t").unwrap().is_empty());
    }

    // -- Fallback -----------------------------------------------------------

    fn two_state_art(poses: Vec<Pose>, fallback: Vec<(Pose, Pose)>) -> SpriteArt {
        let seq = SeqSpec {
            frames: vec![one_frame(["12", "21"])],
            mode: PlayMode::Hold,
            fps: 8.0,
            blink_every: 0.0,
            blink_for: 0.0,
        };
        SpriteArt {
            grain: 1,
            cells_w: 2,
            cells_h: 2,
            pal: &[".", "#102030", "#ff8000"],
            variants: 1,
            seqs: vec![seq; poses.len()],
            poses,
            fallback,
        }
    }

    #[test]
    fn an_unauthored_pose_borrows_the_sequence_the_renderer_names() {
        // "A mob with no air pose uses its walk cycle" — a rule about how the game
        // animates creatures, not a fact about any one creature.
        let a = two_state_art(vec![Pose::Idle, Pose::Move], vec![(Pose::Air, Pose::Move)]);
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Air), b.state_id(Pose::Move));
        assert_eq!(b.state_id(Pose::Air).unwrap().index(), 1);
    }

    #[test]
    fn a_fallback_chain_is_chased_until_it_lands_on_a_real_sequence() {
        let a = two_state_art(
            vec![Pose::Idle],
            vec![(Pose::Air, Pose::Move), (Pose::Move, Pose::Idle)],
        );
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Air), b.state_id(Pose::Idle));
        assert_eq!(b.state_id(Pose::Move), b.state_id(Pose::Idle));
    }

    #[test]
    fn an_authored_pose_beats_a_fallback_that_names_it() {
        let a = two_state_art(vec![Pose::Idle, Pose::Air], vec![(Pose::Air, Pose::Idle)]);
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Air).unwrap().index(), 1, "authored wins");
    }

    #[test]
    fn a_fallback_cycle_is_a_construction_error_rather_than_a_hang() {
        let a = two_state_art(
            vec![Pose::Idle],
            vec![(Pose::Air, Pose::Move), (Pose::Move, Pose::Air)],
        );
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::FallbackCycle {
                from: Pose::Air,
                ..
            }
        ));
    }

    #[test]
    fn a_fallback_that_names_no_sequence_is_a_construction_error() {
        // The alternative is returning None and drawing nothing, which is an
        // invisible creature — the single hardest rendering bug to trace.
        let a = two_state_art(vec![Pose::Idle], vec![(Pose::Air, Pose::Move)]);
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::FallbackDangling {
                from: Pose::Air,
                to: Pose::Move,
                ..
            }
        ));
    }

    #[test]
    fn a_pose_declared_twice_is_a_construction_error() {
        let a = two_state_art(vec![Pose::Idle, Pose::Idle], Vec::new());
        assert!(matches!(
            BakedSprite::new(&a, "t").unwrap_err(),
            SpriteError::DuplicatePose {
                pose: Pose::Idle,
                ..
            }
        ));
    }

    #[test]
    fn a_pose_the_sprite_never_authored_and_never_borrowed_is_absent() {
        let a = two_state_art(vec![Pose::Idle], Vec::new());
        let b = BakedSprite::new(&a, "t").unwrap();
        assert_eq!(b.state_id(Pose::Swim), None);
    }

    // -- The content bridge -------------------------------------------------

    #[test]
    fn a_sequence_fps_of_zero_inherits_the_sprite_wide_rate() {
        // 0 is "unset", not "deliberately 0" — otherwise retuning a sprite's base
        // rate would silently skip every sequence written at the old default.
        let def = &SPRITES[sprite::PLAYER as usize];
        let a = sprite_art_from_content(def, "player", &FromContentOpts::default()).unwrap();
        let authored = def.seq.unwrap();
        for (i, s) in authored.iter().enumerate() {
            let want = if s.fps == 0.0 { def.fps } else { s.fps };
            assert_eq!(a.seqs[i].fps, want, "{:?}", s.state);
        }
        assert!(
            authored.iter().any(|s| s.fps == 0.0),
            "the player record should exercise the inheritance path"
        );
    }

    #[test]
    fn the_rate_scale_multiplies_whichever_rate_won() {
        // A multiplier rather than an override, so a sequence that DID set its own
        // rate is scaled rather than overwritten — a mob's idle is halved either
        // way.
        let def = &SPRITES[sprite::PLAYER as usize];
        let plain = sprite_art_from_content(def, "player", &FromContentOpts::default()).unwrap();
        let scaled = sprite_art_from_content(
            def,
            "player",
            &FromContentOpts {
                rate_scale: &[(Pose::Idle, 0.5)],
                ..default()
            },
        )
        .unwrap();
        let i = plain.poses.iter().position(|p| *p == Pose::Idle).unwrap();
        assert_eq!(scaled.seqs[i].fps, plain.seqs[i].fps * 0.5);
        let r = plain.poses.iter().position(|p| *p == Pose::Run).unwrap();
        assert_eq!(
            scaled.seqs[r].fps, plain.seqs[r].fps,
            "unnamed poses are untouched"
        );
    }

    #[test]
    fn the_variant_override_beats_what_content_declared() {
        let def = &SPRITES[sprite::PLAYER as usize];
        assert_eq!(def.variants, 1, "the protagonist is not a crowd");
        let a = sprite_art_from_content(
            def,
            "player",
            &FromContentOpts {
                variants: Some(VARIANT_COUNT),
                ..default()
            },
        )
        .unwrap();
        assert_eq!(a.variants, VARIANT_COUNT);
    }

    #[test]
    fn every_sprite_in_the_compiled_table_bakes() {
        // The regression that matters: the whole shipping art set through the whole
        // rasteriser, where every failure mode above is loud.
        assert_eq!(SPRITES.len(), SPRITE_COUNT);
        for def in SPRITES.iter() {
            let art = sprite_art_from_content(def, def.id, &FromContentOpts::default())
                .unwrap_or_else(|e| panic!("{e}"));
            let baked = BakedSprite::new(&art, def.id).unwrap_or_else(|e| panic!("{e}"));
            assert!(baked.bake_count > 0, "{} baked nothing", def.id);
            assert_eq!(baked.cells_w, def.cells_w as u32);
            assert_eq!(baked.cells_h, def.cells_h as u32);
            assert_eq!(
                baked.pixels().len(),
                (baked.atlas_size().x * baked.atlas_size().y * 4) as usize
            );
        }
    }

    #[test]
    fn a_finer_grain_packs_more_texels_into_the_same_world_rect() {
        // Both halves of the invariant grain bends are asserted: the TEXELS
        // double per axis, and the DRAWN rect does not move a pixel — the whole
        // point is more resolution inside the same silhouette, and a grain that
        // changed `w_px` would be a resize wearing an experiment's name.
        //
        // Built here rather than read out of shipped content. It used to sample
        // the frostmite, which was the game's one grain-2 record; every body
        // sprite is grain 1 now, because BODY_SCALE doubled the cell box of each
        // one and halving the grain is what spreads the SAME characters over
        // twice the cells. A guard that can be switched off by a content edit is
        // not guarding the mechanism, so this constructs both grains itself and
        // pins the relationship between them.
        let coarse = art(vec![one_frame(["1.", ".2"])], PlayMode::Loop);

        let mut fine = art(vec![vec!["1..2", ".12.", ".21.", "2..1"]], PlayMode::Loop);
        fine.grain = 2;

        let baked_coarse = BakedSprite::new(&coarse, "coarse").expect("grain 1 bakes");
        let baked_fine = BakedSprite::new(&fine, "fine").expect("grain 2 bakes");

        assert_eq!(
            baked_fine.atlas_size().y,
            baked_coarse.atlas_size().y * 2,
            "texel height is cells * grain"
        );
        assert_eq!(
            baked_fine.atlas_size().x,
            baked_coarse.atlas_size().x * 2,
            "texel width is cells * grain"
        );
        assert_eq!(
            (baked_fine.w_px, baked_fine.h_px),
            (baked_coarse.w_px, baked_coarse.h_px),
            "the drawn rect is still cells * CELL_SIZE -- grain must never touch it"
        );
    }

    #[test]
    fn mob_art_bakes_through_the_same_bridge_as_the_sprite_table() {
        // The reason `ContentArt` is a trait: the identical schema fragment is
        // printed under two names, and binding the bridge to one would make it
        // unusable by the other host.
        let mut seen = 0;
        for spec in yugen_data::mobs::MOBS.iter() {
            let Some(art) = spec.art.as_ref() else {
                continue;
            };
            let a = sprite_art_from_content(
                art,
                spec.id,
                &FromContentOpts {
                    fallback: &[(Pose::Air, Pose::Move)],
                    rate_scale: &[(Pose::Idle, 0.5)],
                    variants: Some(VARIANT_COUNT),
                },
            )
            .unwrap_or_else(|e| panic!("{e}"));
            let baked = BakedSprite::new(&a, spec.id).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(baked.variants, VARIANT_COUNT);
            // Whatever a creature authored, the renderer's borrowing rule means it
            // always has something to draw in the air.
            assert!(
                baked.state_id(Pose::Air).is_some(),
                "{} has no air pose",
                spec.id
            );
            seen += 1;
        }
        assert!(seen > 0, "no mob in the table carries art");
    }

    #[test]
    fn a_record_with_no_sequences_is_a_construction_error() {
        // `MobSpecArt` makes `seq` optional, so this is a shape the tables can
        // actually hold — and a sprite with no sequences has nothing to draw.
        let art = MobSpecArt {
            grain: None,
            cells_w: 2,
            cells_h: 2,
            pal: &[".", "#102030"],
            fps: None,
            variants: None,
            seq: None,
        };
        assert!(matches!(
            sprite_art_from_content(&art, "t", &FromContentOpts::default()).unwrap_err(),
            SpriteError::NoSequences { .. }
        ));
    }

    // -- The id lookup ------------------------------------------------------

    #[test]
    fn every_item_icon_id_names_a_sprite_the_table_has() {
        // `ITEM_ICONS` validated its ids against this same table at compile time,
        // so a miss here means the two generated modules are out of step.
        let mut found = 0;
        for icon in ITEM_ICONS.iter().flatten() {
            assert!(sprite_code_of(icon).is_some(), "{icon} names no sprite");
            found += 1;
        }
        assert!(found > 0, "no item carries an icon");
    }

    #[test]
    fn a_sprite_id_the_table_does_not_have_degrades_to_no_icon() {
        assert_eq!(sprite_code_of("icon_not_a_thing"), None);
        assert_eq!(sprite_code_of(""), None);
        // And the resource says the same thing, so a dangling reference reaches the
        // flat-colour path rather than the renderer.
        let empty = SpriteAtlases::default();
        assert!(empty.get("icon_not_a_thing").is_none());
        assert!(empty.by_code(9999).is_none());
    }

    #[test]
    fn a_sprite_id_the_table_does_have_resolves_to_its_own_code() {
        assert_eq!(sprite_code_of("player"), Some(sprite::PLAYER));
        assert_eq!(sprite_code_of("icon_torch"), Some(sprite::ICON_TORCH));
        for def in SPRITES.iter() {
            assert_eq!(sprite_code_of(def.id), Some(def.code));
        }
    }

    // -- The atlas and the flip --------------------------------------------

    #[test]
    fn the_atlas_is_one_row_of_tiles_at_grain_texels_per_cell() {
        // The subject is the player, which is now a grain-2 sprite — so this
        // test is ALSO the check that a finer grain strides the atlas in
        // texels. It used to be called "...at_one_texel_per_cell" and assert
        // tile rects in cells; the day the player took `grain = 2` it went red,
        // which is exactly what it was for. Tile n starts n * (cells * grain)
        // texels across, and the strip is one tile tall — the property that
        // makes the offset arithmetic `tile * tile_w`.
        let def = &SPRITES[sprite::PLAYER as usize];
        let art = sprite_art_from_content(def, "player", &FromContentOpts::default()).unwrap();
        // The player is grain 1 under BODY_SCALE: doubling its cell box and
        // halving its grain is what spread the same characters over twice the
        // cells. The tile arithmetic below is the claim, and it holds at any
        // grain — so it is read off the record rather than pinned to a value.
        let g = art.grain;
        assert!(g >= 1, "grain is a positive texel rate");
        let b = BakedSprite::new(&art, "player").unwrap();
        let layout = b.layout();
        assert_eq!(layout.textures.len(), b.bake_count);
        assert_eq!(layout.size, b.atlas_size());
        let (tw, th) = (b.cells_w * b.grain, b.cells_h * b.grain);
        for (i, rect) in layout.textures.iter().enumerate() {
            assert_eq!(rect.min, UVec2::new(i as u32 * tw, 0));
            assert_eq!(rect.max - rect.min, UVec2::new(tw, th));
        }
    }

    #[test]
    fn the_image_asset_is_srgb_and_matches_the_baked_strip() {
        let a = art(vec![one_frame(["12", "21"])], PlayMode::Hold);
        let b = BakedSprite::new(&a, "t").unwrap();
        let img = b.image();
        assert_eq!(img.texture_descriptor.format, TextureFormat::Rgba8UnormSrgb);
        assert_eq!(img.texture_descriptor.size.width, b.atlas_size().x);
        assert_eq!(img.texture_descriptor.size.height, b.atlas_size().y);
    }

    #[test]
    fn the_one_axis_flip_puts_a_sim_top_left_rect_where_bevy_wants_its_centre() {
        // Sim +y is DOWN, Bevy +y is UP, and this is the only place in the sprite
        // system that knows it.
        let t = world_translation(100.0, 40.0, 10.0, 15.0, 0.5);
        assert_eq!(t.x, 105.0);
        assert_eq!(
            t.y, -47.5,
            "further down the sim is further down the screen"
        );
        assert_eq!(t.z, 0.5);
    }

    #[test]
    fn the_flip_snaps_the_corner_and_not_the_centre() {
        // An odd extent must not land the sprite's edges on half pixels; a
        // fractional edge is the one thing the low-res target cannot forgive.
        let t = world_translation(100.4, 40.6, 5.0, 15.0, 0.0);
        assert_eq!(t.x, 102.5, "corner rounded to 100, then half the extent");
        assert_eq!(t.y, -48.5);
    }

    // -- The plugin ---------------------------------------------------------

    /// A headless app with the sprite plugin and nothing else that draws.
    fn baked_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .add_plugins(SpritePlugin);
        app.update();
        app
    }

    #[test]
    fn the_plugin_bakes_the_whole_table_before_startup_runs() {
        // Headless: no window, no renderer, no GPU. `SpritePlugin` registers the
        // two asset stores itself when nothing else has, which is what makes this
        // possible — and the guard is what stops it wiping them when something has.
        let app = baked_app();

        let atlases = app.world().resource::<SpriteAtlases>();
        assert_eq!(atlases.count(), SPRITE_COUNT);

        let player = atlases
            .get("player")
            .expect("the player sprite is in the table");
        // 8, not 2: the character was redrawn at 4x5 and then BODY_SCALE spread
        // the same art over 8x10 cells. The number is asserted rather than
        // derived so that a bake reading the WRONG record still fails here —
        // `by_code` and `get` are two different lookups and this is the one place
        // both are checked against the same expectation.
        assert_eq!(player.baked.cells_w, 8);
        assert!(player.baked.state_id(Pose::WallSlide).is_some());
        assert_eq!(atlases.by_code(sprite::PLAYER).unwrap().baked.cells_w, 8);

        // The images really landed in the store, one per sprite and no more.
        assert_eq!(app.world().resource::<Assets<Image>>().len(), SPRITE_COUNT);
    }

    #[test]
    fn a_still_from_the_atlas_is_frame_zero_of_the_first_sequence_unflipped() {
        let app = baked_app();
        let atlases = app.world().resource::<SpriteAtlases>();
        let icon = atlases
            .get("icon_torch")
            .expect("icon_torch is in the table");

        let s = icon.still();
        assert!(!s.flip_x, "UI has no facing");
        assert_eq!(
            s.texture_atlas.as_ref().unwrap().index,
            icon.baked.still_tile(icon.baked.first_state(), 0)
        );
        assert_eq!(
            s.custom_size,
            Some(Vec2::new(icon.baked.w_px, icon.baked.h_px)),
            "one sprite pixel per world cell"
        );
        assert_eq!(s.image, icon.image);
    }

    #[test]
    fn a_facing_sprite_flips_about_its_own_axis_rather_than_the_origin() {
        // `flip_x` mirrors within the quad, which is what the TypeScript's
        // `translate(x + w, y); scale(-1, 1)` did by hand. The transform is
        // untouched, so the figure turns while staying exactly where it was.
        let app = baked_app();
        let atlases = app.world().resource::<SpriteAtlases>();
        let player = atlases.get("player").unwrap();
        let facing_left = player.sprite_at(0, true);
        let facing_right = player.sprite_at(0, false);
        assert!(facing_left.flip_x);
        assert!(!facing_right.flip_x);
        assert_eq!(facing_left.custom_size, facing_right.custom_size);
    }
}

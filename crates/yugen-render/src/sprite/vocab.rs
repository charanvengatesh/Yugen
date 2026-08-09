//! What a sprite IS, before anything draws one: the tint table, the pose and
//! play-mode vocabularies, and the error a bad one produces.
//!
//! Carved from [`super`]'s "Tint", "The vocabulary" and "Failure" sections, and
//! they belong together because they are the shared nouns — every other file in
//! this module is written in terms of a [`Pose`], a [`PlayMode`], a [`SeqSpec`]
//! or a [`SpriteError`], and none of them is written in terms of the others.
//!
//! The tint is here rather than in content because it is the one rendering
//! decision content does not get to make. No Bevy.

use yugen_data::mobs::{MobSpecSeqMode, MobSpecSeqState};
use yugen_data::sprites::{SpriteSeqMode, SpriteSeqState};

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
pub(super) const POSE_COUNT: usize = 14;

impl Pose {
    /// Dense index into the id table. `as usize` on the enum would do it, but
    /// only silently; this is the one place the two are required to agree and the
    /// only place the cast lives.
    pub(super) const fn index(self) -> usize {
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

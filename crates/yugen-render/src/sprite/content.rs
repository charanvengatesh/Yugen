//! The bridge from generated content tables to a [`SpriteArt`].
//!
//! [`super`]'s "The content bridge" section. Two shapes in `yugen-data` describe
//! authored art — a standalone sprite record and a mob's inline `art.` group —
//! and they are structurally identical without sharing a type, because
//! `contentc` emits each kind independently. The two traits here are what let
//! one baker read both.
//!
//! [`FromContentOpts`] is where the caller states the three decisions that
//! belong to the RENDERER rather than to content: the pose fallback, the rate
//! scale, and how many variants to bake. `crate::mobs` supplies a different set
//! from the standalone path, which is the whole reason they are arguments.

use yugen_data::mobs::{MobSpecArt, MobSpecSeq};
use yugen_data::sprites::{SPRITES, SpriteDef, SpriteSeq};

use super::anim::split_frames;
use super::vocab::{PlayMode, Pose, SeqSpec, SpriteArt, SpriteError};

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

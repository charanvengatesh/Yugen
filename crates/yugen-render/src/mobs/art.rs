//! Baking the bestiary: one strip texture per creature, resolved once.
//!
//! Carved out of [`super`]'s "The art" section, which is the argument this file
//! implements. Every creature draws its own authored sprite through
//! [`crate::sprite`], baked ONCE in `PreStartup` into [`MobAtlases`] with the
//! three poses a brain can ask for already resolved to a [`StateId`]. Nothing
//! rasterises per frame; a draw is a tile index, a flip and a translation.
//!
//! Three of the decisions in that bake are the RENDERER's and not content's, and
//! they are why mob art does not go through
//! [`bake_sprite_table`](crate::sprite): a creature with no `air` sequence flies
//! with its walk cycle, a creature's idle plays at half the authored rate, and
//! every creature bakes the whole variant tint table whatever its own `variants`
//! says. All three are stated in `yugen-core` as facts about the bestiary and
//! are spelt here as the three fields of a [`FromContentOpts`].
//!
//! The art rect is NOT the body box. A sprite may overhang its own hitbox, so
//! feet stay planted and horns grow upward — see [`art_origin`], which is the
//! only place that subtraction happens.

use bevy::image::TextureAtlasLayout;
use bevy::prelude::*;

use yugen_core::entities::mobs::defs::{POSE_FALLBACK_AIR_IS_MOVE, POSE_RATE_SCALE_IDLE};
use yugen_core::entities::mobs::{MobClock, MobDef, MobPose, VARIANT_COUNT as MOB_VARIANT_COUNT};
use yugen_data::mobs::{MOB_COUNT, MOBS, MobSpecArt, MobSpecSeq, MobSpecSeqMode, MobSpecSeqState};

use crate::sprite::{
    BakedSprite, FromContentOpts, Pose, SpriteAtlas, SpriteClock, StateId, VARIANT_COUNT,
    sprite_art_from_content,
};

// ---------------------------------------------------------------------------
// The baked bestiary
// ---------------------------------------------------------------------------

/// The one state-borrowing rule every creature in the game gets.
///
/// Supplied by this module and never authored, because "a mob with no air pose
/// flies with its walk cycle" is a rule about how the game animates creatures —
/// letting forty content files each express it would give forty chances to
/// disagree. `yugen-core` states the rule as a fact about [`MobPose::Air`]'s
/// SEMANTICS; the const assert below is what keeps the two from drifting apart
/// silently.
const POSE_FALLBACK: &[(Pose, Pose)] = &[(Pose::Air, Pose::Move)];

const _: () = assert!(
    POSE_FALLBACK_AIR_IS_MOVE,
    "the bestiary withdrew the air->move borrowing rule; POSE_FALLBACK must follow"
);

/// Mobs read their idle loop at half the authored rate.
///
/// A property of how creatures READ, not of any creature's art, so content does
/// not pre-halve — an author writing `fps 7` still means seven, at the walking
/// rate. It is a multiplier rather than an override so a sequence that named its
/// own rate is scaled rather than overwritten.
const POSE_RATE_SCALE: &[(Pose, f32)] = &[(Pose::Idle, POSE_RATE_SCALE_IDLE)];

/// The variant table the renderer bakes and the spawner assigns from are the
/// same table, or a crowd wraps several individuals onto one tint.
///
/// `MobSystem::init` draws `rand() * VARIANT_COUNT` for every creature it spawns,
/// so the number is part of the mob RNG's consumption order and the simulation
/// cannot not know it. This is the seam between the two copies.
const _: () = assert!(
    MOB_VARIANT_COUNT as usize == VARIANT_COUNT,
    "the bestiary and the tint table disagree about how many variants exist"
);

/// The art a tombstoned id draws: one magenta cell.
///
/// A tombstoned id (`content/FORMAT.md` §4) keeps its code forever so an old save
/// still resolves whatever it had spawned, but it carries no `art` group. Magenta
/// is exactly what you want to see if one is ever actually drawn, and it is 1x1
/// because the schema's tombstone defaults are `body_cells_w 1` / `body_cells_h
/// 1` — art and body agree, so `MobDef`'s overhang checks pass without a special
/// case.
///
/// Both `idle` and `move` are present even though they are the same frame:
/// [`POSE_FALLBACK`] is applied to every creature, and a fallback naming a
/// sequence that does not exist is a construction error.
static REMOVED_ART: MobSpecArt = MobSpecArt {
    grain: None,
    cells_w: 1,
    cells_h: 1,
    pal: &[".", "#ff00ff"],
    fps: Some(1.0),
    variants: Some(1),
    seq: Some(&REMOVED_SEQ),
};

/// The two held frames of [`REMOVED_ART`].
static REMOVED_SEQ: [MobSpecSeq; 2] = [
    MobSpecSeq {
        state: MobSpecSeqState::Idle,
        mode: MobSpecSeqMode::Hold,
        fps: 0.0,
        blink_every: 0.0,
        blink_for: 0.0,
        frames: &["1"],
    },
    MobSpecSeq {
        state: MobSpecSeqState::Move,
        mode: MobSpecSeqMode::Hold,
        fps: 0.0,
        blink_every: 0.0,
        blink_for: 0.0,
        frames: &["1"],
    },
];

/// One creature's art: the atlas, and its three poses resolved once.
///
/// RESOLVING AT LOAD is the rule [`crate::sprite`]'s header states and this is
/// where creatures obey it. The TypeScript kept the same array on its `MobDef`
/// and for the same reason — a brain sets a [`MobPose`], and turning that into a
/// sequence must not be a lookup on the draw path.
#[derive(Clone, Debug)]
pub struct MobArt {
    /// The strip texture, its layout, and the tile table.
    pub atlas: SpriteAtlas,
    /// Indexed by [`MobPose`]. `Air` has [`POSE_FALLBACK`] already applied.
    states: [StateId; POSE_COUNT],
}

/// How many poses a creature's brain can ask for. [`MobPose`] has three.
const POSE_COUNT: usize = 3;

impl MobArt {
    /// The resolved sequence for a pose.
    pub fn state(&self, pose: MobPose) -> StateId {
        self.states[pose as usize]
    }
}

/// Every creature's art, indexed by mob code.
///
/// Index == code, the same shape `MOB_DEFS` has, so a code off a `Mob` indexes
/// straight in. The `Option` carries the one honest absence: nothing today, since
/// a tombstone bakes [`REMOVED_ART`] rather than a hole, but a mob that failed to
/// bake could be reported without shifting every code after it.
#[derive(Resource, Default)]
pub struct MobAtlases {
    by_code: Vec<Option<MobArt>>,
}

impl MobAtlases {
    /// The art for a mob code, or `None` before [`bake_mob_art`] has run.
    pub fn by_code(&self, code: u16) -> Option<&MobArt> {
        self.by_code.get(code as usize)?.as_ref()
    }

    /// How many creatures are baked. Zero before `PreStartup`, which is the state
    /// a test that never ran the schedule sees.
    pub fn count(&self) -> usize {
        self.by_code.iter().filter(|a| a.is_some()).count()
    }
}

/// The simulation's pose vocabulary, in the renderer's.
///
/// Two enums rather than one because they answer different questions: `MobPose`
/// is what a brain decided the creature is DOING, [`Pose`] is what the sprite
/// system can draw. This match is exhaustive, so a fourth mob pose stops
/// compiling here until this module says what it looks like.
pub(super) fn pose_of(pose: MobPose) -> Pose {
    match pose {
        MobPose::Idle => Pose::Idle,
        MobPose::Move => Pose::Move,
        MobPose::Air => Pose::Air,
    }
}

/// The creature's own animation clock, in the sprite system's.
///
/// A copy of three floats, and deliberately not a `From` impl on either type:
/// [`MobClock`] is advanced by the fixed step next to the physics (`state_t` is
/// real simulation state — the burrower's tell reads it) and `yugen-core` may
/// not know what a [`SpriteClock`] is.
pub(super) fn clock_of(clock: &MobClock) -> SpriteClock {
    SpriteClock {
        state_t: clock.state_t,
        clock_t: clock.clock_t,
        phase: clock.phase,
    }
}

/// The art rect's top-left in SIM space, from the body box and the published pad.
///
/// The one place the two geometries meet. The body is what the world collides
/// with; the picture may be wider (wings) and taller (horns) and is offset off
/// the box's corner, never centred on it — the vertical pad is all on top so the
/// art's bottom row always lands on the body's bottom row.
///
/// The pad is a whole number of cells by construction (`MobDef`'s build asserts
/// the horizontal difference is even), which is why rounding can happen after the
/// subtraction here instead of before it as the original's
/// `Math.round(m.box.x) - d.artPadXPx` did: the two are the same number.
pub(super) fn art_origin(def: &MobDef, body_x: f32, body_y: f32) -> (f32, f32) {
    (body_x - def.art_pad_x_px, body_y - def.art_pad_top_px)
}

/// Rasterise and upload every creature in `yugen_data::mobs::MOBS`.
///
/// Panics on malformed art, which is the policy [`crate::sprite`]'s
/// [`SpriteError`](crate::sprite::SpriteError) exists to hold: a frame that bakes
/// once bakes forever, so a typo is loud at load instead of shipping as a hole in
/// a creature.
///
/// `PreStartup`, so anything built in `Startup` finds it populated with no
/// ordering edge to get wrong — the same promise
/// [`SpritePlugin`](crate::sprite::SpritePlugin) makes for the sprite table.
pub(super) fn bake_mob_art(
    mut atlases: ResMut<MobAtlases>,
    mut images: ResMut<Assets<Image>>,
    mut layouts: ResMut<Assets<TextureAtlasLayout>>,
) {
    let opts = FromContentOpts {
        fallback: POSE_FALLBACK,
        rate_scale: POSE_RATE_SCALE,
        variants: Some(VARIANT_COUNT),
    };

    let mut out = Vec::with_capacity(MOB_COUNT);
    for spec in MOBS.iter() {
        let baked = bake_one(spec.art.as_ref().unwrap_or(&REMOVED_ART), spec.id, &opts);
        let states = resolve_states(&baked, spec.id);
        out.push(Some(MobArt {
            atlas: SpriteAtlas {
                image: images.add(baked.image()),
                layout: images_layout(&mut layouts, &baked),
                baked,
            },
            states,
        }));
    }
    atlases.by_code = out;
}

/// One creature's art record through the content bridge and the rasteriser.
///
/// Split out of [`bake_mob_art`] so the bake is assertable without an `App`: it
/// is the half that has no Bevy in it, and every claim about pose sharing, rate
/// scaling and the variant table is a claim about what this returns.
fn bake_one(art: &MobSpecArt, who: &str, opts: &FromContentOpts<'_>) -> BakedSprite {
    let art =
        sprite_art_from_content(art, who, opts).unwrap_or_else(|e| panic!("content/mobs: {e}"));
    BakedSprite::new(&art, who).unwrap_or_else(|e| panic!("content/mobs: {e}"))
}

/// Add the strip's layout, keeping [`bake_mob_art`] to one expression per field.
fn images_layout(
    layouts: &mut Assets<TextureAtlasLayout>,
    baked: &BakedSprite,
) -> Handle<TextureAtlasLayout> {
    layouts.add(baked.layout())
}

/// Resolve all three poses once, or refuse to start.
///
/// `Idle` and `Move` must be authored; `Air` is allowed to be missing because
/// [`POSE_FALLBACK`] borrows it from `Move`, which is exactly why the failure
/// here can be stated as "every pose resolves" rather than as the original's
/// two-name special case. A pose that resolved to nothing would be an invisible
/// creature, which is the single hardest rendering bug to trace to its cause.
fn resolve_states(baked: &BakedSprite, who: &str) -> [StateId; POSE_COUNT] {
    let mut out = [baked.first_state(); POSE_COUNT];
    for pose in [MobPose::Idle, MobPose::Move, MobPose::Air] {
        out[pose as usize] = baked.state_id(pose_of(pose)).unwrap_or_else(|| {
            panic!("content/mobs: {who} has no {pose:?} sequence and nothing to borrow one from")
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::config::CELL_SIZE;
    use yugen_core::entities::mobs::MOB_DEFS;

    /// The options every creature is baked with. The bake system's, restated so
    /// the tests below assert about what the game actually loads.
    fn opts() -> FromContentOpts<'static> {
        FromContentOpts {
            fallback: POSE_FALLBACK,
            rate_scale: POSE_RATE_SCALE,
            variants: Some(VARIANT_COUNT),
        }
    }

    /// Every creature in the bestiary bakes, and its picture is the size the
    /// bestiary told the renderer to expect.
    ///
    /// This is the seam between two tables that are built from the same content
    /// by different code: `MobDef::art_w_px` is measured by `yugen-core` from
    /// `art.cells_w`, and `BakedSprite::w_px` is measured by the rasteriser from
    /// the frames it actually rasterised. If they ever disagree, the sprite is
    /// drawn into a rect it does not fill and every creature is subtly stretched.
    #[test]
    fn every_creature_bakes_at_the_art_rect_the_bestiary_published() {
        for spec in MOBS.iter() {
            let baked = bake_one(spec.art.as_ref().unwrap_or(&REMOVED_ART), spec.id, &opts());
            let def = &MOB_DEFS[spec.code as usize];
            assert_eq!(baked.w_px, def.art_w_px, "{}", spec.id);
            assert_eq!(baked.h_px, def.art_h_px, "{}", spec.id);
            assert_eq!(baked.variants, VARIANT_COUNT, "{}", spec.id);
        }
    }

    /// Every pose a brain can set resolves to a sequence, for every creature.
    ///
    /// The bake panics if one does not, so this is the assertion that the whole
    /// bestiary starts — and it is worth having as a test rather than as a
    /// startup surprise, because `resolve_states` is the only thing standing
    /// between a missing sequence and an invisible creature.
    #[test]
    fn every_pose_of_every_creature_resolves() {
        for spec in MOBS.iter() {
            let baked = bake_one(spec.art.as_ref().unwrap_or(&REMOVED_ART), spec.id, &opts());
            let states = resolve_states(&baked, spec.id);
            for pose in [MobPose::Idle, MobPose::Move, MobPose::Air] {
                assert!(
                    baked.poses().get(states[pose as usize].index()).is_some(),
                    "{} has no sequence for {pose:?}",
                    spec.id
                );
            }
        }
    }

    /// A creature with no `air` sequence flies with its walk cycle.
    ///
    /// Every creature in today's bestiary authors its own `air`, which is asserted
    /// here rather than assumed — the rule is not dead code, it is the thing that
    /// lets the FORTY-FIRST creature ship without one. The record that exercises
    /// it is the tombstone's, which authors `idle` and `move` and nothing else for
    /// exactly this reason.
    #[test]
    fn a_creature_with_no_air_pose_borrows_its_walk_cycle() {
        for spec in MOBS.iter() {
            let Some(art) = spec.art.as_ref() else {
                continue;
            };
            assert!(
                art.seq
                    .unwrap_or(&[])
                    .iter()
                    .any(|s| s.state == MobSpecSeqState::Air),
                "{} no longer authors air — this test now covers it directly",
                spec.id
            );
        }

        let baked = bake_one(&REMOVED_ART, "removed", &opts());
        let states = resolve_states(&baked, "removed");
        assert_eq!(
            states[MobPose::Air as usize],
            states[MobPose::Move as usize],
            "a creature with no air pose should fly with its walk cycle"
        );
        assert_ne!(
            states[MobPose::Air as usize],
            states[MobPose::Idle as usize],
            "the walk cycle specifically, not whichever sequence came first"
        );
    }

    /// Idle plays at half the authored rate; nothing else is touched.
    #[test]
    fn the_idle_loop_plays_at_half_the_authored_rate() {
        for spec in MOBS.iter() {
            let Some(art) = spec.art.as_ref() else {
                continue;
            };
            let baked = bake_one(art, spec.id, &opts());
            let states = resolve_states(&baked, spec.id);
            for seq in art.seq.unwrap_or(&[]) {
                // What the sequence would have played at with no scale: its own
                // rate, or the sprite-wide one it inherits when unset.
                let authored = if seq.fps == 0.0 {
                    art.fps.unwrap_or(0.0)
                } else {
                    seq.fps
                };
                let pose = match seq.state {
                    MobSpecSeqState::Idle => MobPose::Idle,
                    MobSpecSeqState::Move => MobPose::Move,
                    MobSpecSeqState::Air => MobPose::Air,
                };
                let want = if seq.state == MobSpecSeqState::Idle {
                    authored * POSE_RATE_SCALE_IDLE
                } else {
                    authored
                };
                assert_eq!(
                    baked.spec(states[pose as usize]).fps,
                    want,
                    "{} {:?}",
                    spec.id,
                    seq.state
                );
            }
        }
    }

    /// A tombstoned id draws one magenta cell rather than nothing.
    #[test]
    fn a_tombstoned_creature_bakes_a_magenta_cell() {
        let baked = bake_one(&REMOVED_ART, "removed", &opts());
        assert_eq!(baked.w_px, CELL_SIZE as f32);
        assert_eq!(baked.h_px, CELL_SIZE as f32);
        // Variant 0 is the identity tilt, so it is the authored colour exactly.
        let tile = baked.still_tile(baked.first_state(), 0);
        let px = &baked.pixels()[tile * 4..tile * 4 + 4];
        assert_eq!(px, [0xff, 0x00, 0xff, 0xff]);
        // And it resolves all three poses through the same fallback every real
        // creature uses, which is why it authors `move` as well as `idle`.
        resolve_states(&baked, "removed");
    }

    /// The picture hangs off the body box by the pad, never centred on it.
    ///
    /// This used to assert that every pad in the bestiary was ZERO, because it
    /// was: every creature was authored on a grid exactly its own hitbox. The
    /// 8x8 migration ended that — every creature is now drawn on the same 8x8
    /// canvas whatever its body is — so the claim has moved from "there is no
    /// overhang" to the thing that was always the real invariant: **wherever
    /// there is an overhang, it is split evenly across and carried entirely on
    /// top.**
    ///
    /// That is strictly stronger than the old assertion. The old one held
    /// vacuously — with every pad zero, an `art_origin` that centred vertically
    /// and one that planted the feet were indistinguishable. Now they are not,
    /// and this says which one is right against twenty real creatures instead of
    /// against one synthetic def.
    #[test]
    fn the_art_hangs_off_the_body_box_by_the_published_pad() {
        for def in MOB_DEFS.iter() {
            assert!(
                def.art_pad_x_px >= 0.0 && def.art_pad_top_px >= 0.0,
                "{}: art is smaller than its body box ({}, {})",
                def.id,
                def.art_pad_x_px,
                def.art_pad_top_px
            );
            // Centred across: the same overhang on each side.
            assert_eq!(
                def.art_w_px,
                def.w_px + 2.0 * def.art_pad_x_px,
                "{}: the horizontal overhang is not centred",
                def.id
            );
            // Feet planted: the whole vertical surplus sits above the art, so
            // the art's bottom row is the body's bottom row.
            assert_eq!(
                def.art_h_px,
                def.h_px + def.art_pad_top_px,
                "{}: the vertical overhang is not all on top",
                def.id
            );
        }

        // Two cells wider and one taller than its body: the horizontal overhang
        // splits evenly, the vertical is all on top.
        let def = MobDef {
            art_w_px: MOB_DEFS[0].w_px + 2.0 * CELL_SIZE as f32,
            art_h_px: MOB_DEFS[0].h_px + CELL_SIZE as f32,
            art_pad_x_px: CELL_SIZE as f32,
            art_pad_top_px: CELL_SIZE as f32,
            ..MOB_DEFS[0]
        };
        let (left, top) = art_origin(&def, 100.0, 200.0);

        assert_eq!(left, 100.0 - def.art_pad_x_px);
        assert_eq!(top, 200.0 - def.art_pad_top_px);
        // Feet planted: the art's bottom row is the body's bottom row.
        assert_eq!(top + def.art_h_px, 200.0 + def.h_px);
        // Centred: the same overhang either side.
        assert_eq!(left + def.art_w_px, 100.0 + def.w_px + def.art_pad_x_px);
    }
}

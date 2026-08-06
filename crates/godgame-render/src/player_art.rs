//! The player's art: which pose, how big, how far over, and how far behind.
//!
//! Ported from `src/entities/PlayerArt.ts` and from the half of
//! `Player.draw` that survived the crossing.
//!
//! # Two files became one, and why that is the right seam
//!
//! The TypeScript split this across two places for a reason that does not
//! survive the port:
//!
//! - `PlayerArt.ts` was a *facade*, and it existed because `sprites.gen.ts` held
//!   plain data that could not construct a class. Somebody had to bake the
//!   canvases at load, once, and that was it. It owned the art-grid arithmetic
//!   (`PLAYER_ART_PAD_X_PX`, `PLAYER_ART_PAD_TOP_PX`) and the state-name lookup
//!   table because they were the only two things derivable from the record.
//! - `Player.draw` owned the squash, the lean and the dash smear — arithmetic on
//!   the body's own velocity that happened to end in `ctx.drawImage`.
//!
//! Everything in both is the same computation: *given a body, where and how big
//! is the drawn figure*. The split was between "reads the art table" and "reads
//! the body", and in this tree both of those are inputs to one pure function. So
//! they are one module, and the one thing that was genuinely a renderer —
//! `ctx.save`, `ctx.transform`, `drawImage` — is not here at all. See
//! [`godgame_core::entities`]'s header, which is the other end of this: the
//! simulation publishes every field `draw` read and keeps not one pixel of
//! knowledge about what is done with them.
//!
//! # This module holds no Bevy
//!
//! [`PlayerFigure`] is a value, exactly as [`DayPhase`](crate::daynight::DayPhase)
//! is. That is not tidiness — it is what lets the squash extremes, the lean
//! clamp and the step-up ease be asserted in a unit test with no app, no window
//! and no GPU, which is where every one of those numbers can actually go wrong.
//! There is no plugin here, and adding one would be inventing a schedule for a
//! function that has no state to keep. The system that draws a figure belongs
//! next to whatever owns the rasteriser's entities.
//!
//! # Coordinates: this module is entirely in SIM space
//!
//! Every number that comes out of here has `+y` growing DOWNWARD, like the
//! simulation and like the TypeScript canvas both did. The flip to Bevy's `+y`
//! up happens in exactly one place per drawn thing, at the `Transform` write —
//! see `crate::player::place_body`, [`crate::lowres`] and [`crate::cellmap`],
//! which all do it on the last line and say so. Doing it here would put the
//! convention in two places and make the squash arithmetic read upside down.
//!
//! # Where this joins the rasteriser
//!
//! This section carried a SEAM note for as long as `crate::sprite` was
//! unwritten. Both joins it named are now made, and neither of them is made
//! HERE — which is why nothing in this file mentions an atlas, a tile or a
//! vertex:
//!
//! 1. **Pose indices**, in `crate::player::tile_uv`. Three hops, each one a
//!    crossing between vocabularies: [`seq_state`] from the simulation's
//!    hand-written [`AnimState`] to content's generated `SpriteSeqState`,
//!    `Pose::from` from content's to the rasteriser's, and
//!    `BakedSprite::state_id` to the dense sequence index. A pose the sheet does
//!    not author falls back to its first sequence there, because the wrong
//!    stance for a frame is a far cheaper failure than an invisible player. This
//!    module used to answer that question itself; the note under "Poses" says
//!    what stood there and why it went.
//! 2. **The blit**, in `crate::player::place_body`. A [`PlayerFigure`] becomes
//!    one to three quads in a single `Mesh2d` sampling one atlas tile —
//!    [`PlayerFigure::ghosts`] furthest-first, then the figure, because index
//!    order IS composite order inside a mesh. The lean was the hard part: Bevy's
//!    `Transform` is translate-rotate-scale and **cannot represent a shear**, so
//!    for one milestone it was computed here and discarded on the way to the
//!    screen. [`crate::shear`] settled it by putting
//!    [`PlayerFigure::shear_at`]'s displacement on the quad's top and bottom
//!    edges, which is EXACT and not an approximation because a shear is linear
//!    and so is vertex interpolation. The rotation about the feet this header
//!    once offered as the alternative was rejected on measurement, not taste: it
//!    tips the horizontals, and on a six-pixel character that is a foot hovering
//!    off the floor for a whole run cycle.
//!
//! What has not changed is that nothing here CALLS any of it. The art grid comes
//! from the compiled record and the geometry comes from the body, so every number
//! this module publishes is still derivable, and asserted, with no atlas and no
//! GPU.

use godgame_core::config::{CELL_SIZE, PLAYER_CELLS_H, PLAYER_CELLS_W, PLAYER_H, PLAYER_W, scaled};
use godgame_core::entities::{AnimState, Player};
use godgame_data::sprites::{SPRITES, SpriteDef, SpriteSeqState, sprite};

// ---------------------------------------------------------------------------
// The tuning that came back with the drawing
// ---------------------------------------------------------------------------
//
// These are the numbers `godgame-core`'s entity header says it dropped and
// deliberately did not restate: squash, smear and shear are the shape of a
// drawn sprite and not of a body. They are module constants and not `config`
// exports because each describes ONE algorithm — the one below — which is the
// tuning tier's own rule. `ACCEL_SMOOTH` is the counter-example that pins the
// boundary: the lean's low-pass runs on the fixed step, inside `update_anim`,
// so it stayed with the simulation.
//
// `scaled` is applied to exactly the three that are LENGTHS or velocities.
// `DASH_GHOST_STEP` is a distance in world px; `RISE_STRETCH_DIV` and `LEAN_DIV`
// are the px/s and px/s^2 a velocity and an acceleration are divided by, so they
// have to move with the body's scale or a smaller character would stretch and
// lean twice as hard as a larger one at the same *relative* speed. The rest are
// dimensionless ratios and must not be scaled.

/// Horizontal scale while dashing. Wider than tall, because the pose is already
/// leaning forward and the smear is meant to read as motion blur rather than as
/// a second contortion on top of it.
const DASH_SMEAR_X: f32 = 1.34;

/// Vertical scale while dashing. Deliberately not `1.0 / DASH_SMEAR_X`: a dash
/// is allowed to lose a little area, which reads as speed, where the rise
/// stretch below is not.
const DASH_SMEAR_Y: f32 = 0.78;

/// World px between consecutive dash after-images.
const DASH_GHOST_STEP: f32 = scaled(7.0);

/// Opacity of each after-image, NEAREST FIRST.
///
/// The trail is two deep and this array's length is what says so. It fades with
/// distance rather than being one flat ghost, because a single after-image at
/// any opacity reads as a second character, and two at these weights read as one
/// character that was recently somewhere else.
const DASH_GHOST_ALPHA: [f32; 2] = [0.3, 0.15];

/// Rise speed, in px/s, per unit of vertical stretch.
const RISE_STRETCH_DIV: f32 = scaled(2600.0);

/// Most the figure may SQUAT on a fast fall. Negative: a plummeting body gets
/// shorter and wider, which is the opposite of the rise.
const RISE_STRETCH_MIN: f32 = -0.16;

/// Most the figure may STRETCH on a fast rise.
const RISE_STRETCH_MAX: f32 = 0.34;

/// Stretch added at zero vertical speed.
///
/// Small and positive, so the apex of a jump is very slightly tall rather than
/// exactly the standing silhouette. Without it the figure snaps back to neutral
/// for the frame or two it hangs at the top, which is the one moment the eye is
/// most likely to be on it.
const RISE_STRETCH_BIAS: f32 = 0.04;

/// How much wider a full landing squash makes the figure.
const LAND_FATTEN: f32 = 0.45;

/// How much shorter a full landing squash makes the figure.
///
/// This and [`LAND_FATTEN`] are two independent knobs, NOT the area-preserving
/// pair the rise stretch uses — conserving area at this flatten would need a
/// fatten of 0.67, half again as wide as the one here. A landing is meant to
/// read as the body being driven INTO the floor, so the silhouette loses area
/// on the way down; the fatten only being slightly the larger of the two is
/// what stops that compression from reading as the character shrinking.
const LAND_FLATTEN: f32 = 0.4;

/// Horizontal acceleration, in px/s^2, per unit of shear.
const LEAN_DIV: f32 = scaled(40000.0);

/// Most the figure may shear, as an x offset per unit of height above the feet.
/// At 0.16 the head of a three-cell figure moves under half a cell.
const LEAN_MAX: f32 = 0.16;

/// Below this the lean is treated as no lean at all.
///
/// The smoothed acceleration is never exactly zero — it is a low-pass, and it
/// asymptotes — so without a deadband the figure would carry a sub-pixel shear
/// forever and a game spent standing still would pay for a transform every
/// frame.
const LEAN_DEADBAND: f32 = 0.005;

// ---------------------------------------------------------------------------
// The art grid
// ---------------------------------------------------------------------------

/// How the art rect sits on the collision box, read FROM the compiled record.
///
/// # Why this is derived and not declared
///
/// `SPRITE_CELLS_W/H` and `SPRITE_PAD_X/TOP` used to be hand-written constants
/// in the TypeScript's `config/physics.ts`, which meant the art grid existed in
/// two places: once as a number in code, once as the actual row and column count
/// of the art table. Any disagreement silently rescaled the sprite off the cell
/// grid — one art pixel stops being one world cell, the figure is drawn on a
/// finer grid than the terrain it stands on, and nothing anywhere reports it.
/// That is the single bug the whole content pipeline exists to make impossible,
/// so the grid is read from the record that carries the pixels and the two
/// cannot disagree.
///
/// [`PLAYER_CELLS_W`] and [`PLAYER_CELLS_H`] — the COLLISION BOX — stay in
/// `config`, untouched. That split is the point: content owns what a pixel looks
/// like, code owns what the body does. An artist may grow the art grid to hang a
/// scarf or a swung limb outside the silhouette and the hitbox does not move,
/// the physics does not retune, and nothing about how the character collides
/// changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArtGrid {
    /// Art rect width in world px. One art pixel is one world cell.
    pub w_px: f32,
    /// Art rect height in world px.
    pub h_px: f32,
    /// How far LEFT of the collision box the art rect starts.
    ///
    /// Half the horizontal surplus, so the figure does not drift sideways as the
    /// grid widens. A surplus of an odd number of cells therefore lands on a
    /// half cell, which is legal and intentional — the sprite still blits at one
    /// pixel per cell, it is only the box-relative origin that sits mid-cell,
    /// and forcing it to a whole cell would push the figure visibly off centre
    /// instead.
    pub pad_x_px: f32,
    /// How far ABOVE the collision box the art rect starts.
    ///
    /// All of the vertical surplus, not half: the art is aligned to the FEET, so
    /// a taller grid grows upward and the figure stays planted on the floor.
    pub pad_top_px: f32,
}

impl ArtGrid {
    /// The grid one compiled sprite record declares, against the player's box.
    pub fn from_def(def: &SpriteDef) -> ArtGrid {
        let cell = CELL_SIZE as f32;
        ArtGrid {
            w_px: def.cells_w as f32 * cell,
            h_px: def.cells_h as f32 * cell,
            pad_x_px: (def.cells_w - PLAYER_CELLS_W) as f32 / 2.0 * cell,
            pad_top_px: (def.cells_h - PLAYER_CELLS_H) as f32 * cell,
        }
    }

    /// The shipped player's grid.
    ///
    /// One static array read and four multiplies, not a hash lookup, so there is
    /// nothing here worth caching in a resource. At the authored 2x3 against a
    /// 2x3 box both pads are zero and the art exactly fills the hitbox.
    pub fn player() -> ArtGrid {
        ArtGrid::from_def(&SPRITES[sprite::PLAYER as usize])
    }
}

// ---------------------------------------------------------------------------
// Poses
// ---------------------------------------------------------------------------
//
// A `PoseSheet` trait, a `ContentPoses` implementation of it, a `PoseIds` table
// and a `MissingPose` error stood here. They were this module's own answer to
// "which frame set plays this pose", written while `crate::sprite` was still
// unwritten, and they answered it by scanning the compiled record's `seq` list
// for a matching state. `crate::sprite` landed with a better answer —
// `BakedSprite` keys its frame sets by `Pose`, resolves the authored `fallback`
// chains that a raw scan of the record cannot see, and rejects duplicate poses
// and fallback cycles at bake time — and `crate::player::tile_uv` has called
// that one on every drawn frame since. Nothing outside this file ever
// constructed a `PoseIds`, so the two answers never got the chance to disagree
// in a running game: the table was resolved by tests and by nobody else.
//
// What survives is the half that was never a stand-in. `seq_state` is the
// crossing from the simulation's vocabulary to content's, it is the first of the
// three hops the draw path actually takes, and it is still the only place the
// two vocabularies are asserted to be the same one.

/// Every pose the player can reach, as data.
///
/// The TypeScript got this list, and a compile-time proof that it was exactly
/// the state union, from `satisfies Record<AnimState, 1>`. Here the proof is
/// [`seq_state`]: it matches on [`AnimState`] with no wildcard arm, so adding a
/// thirteenth pose to the simulation fails to compile until somebody decides
/// which authored sequence it plays. This array is the runtime half — a `match`
/// is not something you can enumerate, and neither was the union.
///
/// Nothing on the draw path indexes it: `crate::player` crosses one pose at a
/// time, per frame, through [`seq_state`]. What needs the enumeration is the
/// content-coverage test below, which walks it to prove every pose the
/// simulation can enter has a sequence authored to play. That test is the only
/// thing between a dropped sequence and a character who silently stands there
/// idling through a wall slide — the draw path cannot report the gap, because
/// falling back to the sheet's first sequence draws SOMETHING and so degrades to
/// a plausible frame rather than to a visible fault.
pub const POSES: [AnimState; POSE_COUNT] = [
    AnimState::Idle,
    AnimState::Run,
    AnimState::Skid,
    AnimState::Jump,
    AnimState::DoubleJump,
    AnimState::Fall,
    AnimState::Land,
    AnimState::Dash,
    AnimState::WallSlide,
    AnimState::Swim,
    AnimState::Punch,
    AnimState::Hurt,
];

/// How many poses there are. See [`POSES`].
pub const POSE_COUNT: usize = 12;

/// The content vocabulary bridge, and the exhaustiveness witness.
///
/// [`AnimState`] is written by hand in `godgame-core` and [`SpriteSeqState`] is
/// generated from `content/sprites/player.toml`. This match is the one place the
/// two are asserted to be the same vocabulary, and it is the only reason a
/// renamed pose is a compile error on one side rather than a pose that silently
/// never plays.
pub const fn seq_state(pose: AnimState) -> SpriteSeqState {
    match pose {
        AnimState::Idle => SpriteSeqState::Idle,
        AnimState::Run => SpriteSeqState::Run,
        AnimState::Skid => SpriteSeqState::Skid,
        AnimState::Jump => SpriteSeqState::Jump,
        AnimState::DoubleJump => SpriteSeqState::DoubleJump,
        AnimState::Fall => SpriteSeqState::Fall,
        AnimState::Land => SpriteSeqState::Land,
        AnimState::Dash => SpriteSeqState::Dash,
        AnimState::WallSlide => SpriteSeqState::WallSlide,
        AnimState::Swim => SpriteSeqState::Swim,
        AnimState::Punch => SpriteSeqState::Punch,
        AnimState::Hurt => SpriteSeqState::Hurt,
    }
}

// ---------------------------------------------------------------------------
// The body, as the drawing sees it
// ---------------------------------------------------------------------------

/// Everything `Player.draw` reached into the player for, as one value.
///
/// This exists so that [`PlayerFigure`] is a pure function of numbers rather
/// than of a `Player`. That is not indirection for its own sake: a landing
/// squash of exactly 1.0, an acceleration past the lean clamp, and a
/// half-completed step-up are all states the body holds for two frames at a time
/// and none of them can be reached by construction. Being able to write one down
/// is the difference between testing the extremes and testing whatever the
/// physics happened to produce.
///
/// [`PlayerMotion::of`] is the only function in this module that touches a
/// `Player`, and every field it reads is an accessor
/// [`godgame_core::entities`] publishes for exactly this purpose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerMotion {
    /// Left edge of the COLLISION BOX, world px.
    pub x: f32,
    /// Top edge of the collision box, world px, `+y` down.
    pub y: f32,
    /// Vertical velocity, px/s. Negative is rising.
    pub vy: f32,
    /// Which way the body points: `1.0` or `-1.0`.
    pub facing: f32,
    /// Feet resting on something solid.
    pub on_ground: bool,
    /// Inside a dash right now.
    pub dashing: bool,
    /// Which way that dash is going.
    pub dash_dir: f32,
    /// Landing squash, decaying to zero.
    pub squash: f32,
    /// Smoothed horizontal acceleration, px/s^2.
    pub accel_x: f32,
    /// Residual visual offset from a step-up, px, easing to zero.
    pub step_up_visual: f32,
    /// The pose to play.
    pub pose: AnimState,
    /// Seconds since the pose was entered.
    pub state_t: f32,
    /// Free-running seconds, for ambient loops.
    pub clock_t: f32,
    /// Run cycle position in `[0, 1)`.
    pub phase: f32,
}

impl PlayerMotion {
    /// Gather what the drawing needs off a live body.
    pub fn of(p: &Player) -> PlayerMotion {
        PlayerMotion {
            x: p.x,
            y: p.y,
            vy: p.vy,
            facing: p.facing,
            on_ground: p.on_ground,
            dashing: p.dashing(),
            dash_dir: p.dash_dir(),
            squash: p.squash(),
            accel_x: p.accel_x(),
            step_up_visual: p.step_up_visual(),
            pose: p.anim(),
            state_t: p.anim_t(),
            clock_t: p.anim_clock_t(),
            phase: p.run_phase(),
        }
    }
}

impl Default for PlayerMotion {
    /// A body standing still at the world origin, facing right.
    ///
    /// The neutral figure: no squash, no lean, no smear, feet on the floor. It
    /// is the identity every extreme in this module's tests is written as a
    /// perturbation of.
    fn default() -> PlayerMotion {
        PlayerMotion {
            x: 0.0,
            y: 0.0,
            vy: 0.0,
            facing: 1.0,
            on_ground: true,
            dashing: false,
            dash_dir: 1.0,
            squash: 0.0,
            accel_x: 0.0,
            step_up_visual: 0.0,
            pose: AnimState::Idle,
            state_t: 0.0,
            clock_t: 0.0,
            phase: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// The figure
// ---------------------------------------------------------------------------

/// One drawn frame of the player: where the art goes, and how it is deformed.
///
/// Every field is derived from a [`PlayerMotion`] and an [`ArtGrid`] and from
/// nothing else, so this is a pure function of the body — which is what makes
/// the whole art model testable without a rasteriser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerFigure {
    /// Left edge of the deformed art rect, world px.
    pub x: f32,
    /// Top edge of the deformed art rect, world px, `+y` down.
    pub y: f32,
    /// Width of the deformed art rect, world px.
    pub w: f32,
    /// Height of the deformed art rect, world px.
    pub h: f32,

    /// Pose to play, in the SIMULATION's vocabulary. `crate::player::tile_uv`
    /// crosses it to a baked sequence; [`seq_state`] is the first hop.
    pub pose: AnimState,
    /// `-1.0` mirrors the art about the rect's OWN vertical axis, so the figure
    /// turns while staying exactly where it is.
    pub facing: f32,

    /// Seconds since the pose was entered.
    /// [`SpriteClock::state_t`](crate::sprite::SpriteClock::state_t).
    pub state_t: f32,
    /// Free-running seconds.
    /// [`SpriteClock::clock_t`](crate::sprite::SpriteClock::clock_t).
    pub clock_t: f32,
    /// Run cycle position in `[0, 1)`.
    /// [`SpriteClock::phase`](crate::sprite::SpriteClock::phase).
    pub phase: f32,

    /// Shear, as an x offset per unit of height ABOVE [`PlayerFigure::pivot_y`].
    /// Positive leans the figure to its right. Already clamped and deadbanded,
    /// so exactly `0.0` means no lean at all.
    pub lean: f32,
    /// Horizontal centre of the collision box — the shear's pivot.
    pub pivot_x: f32,
    /// The DRAWN feet: the box's sole plus the step-up residue. Both the shear
    /// and the squash pivot here, which is what keeps a deforming figure planted
    /// on the floor.
    pub pivot_y: f32,

    /// Direction of the dash that is smearing this figure, or `None`.
    pub smear_dir: Option<f32>,
}

/// One dash after-image: the same figure, dropped back along the dash and faded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ghost {
    /// Left edge, world px.
    pub x: f32,
    /// Top edge, world px. Always the figure's own — a ghost trails, it does not
    /// drift.
    pub y: f32,
    /// Opacity, `0..1`.
    pub alpha: f32,
}

impl PlayerFigure {
    /// The figure for a live body, on the shipped art grid.
    pub fn of(player: &Player) -> PlayerFigure {
        PlayerFigure::new(PlayerMotion::of(player), ArtGrid::player())
    }

    /// The whole art model.
    ///
    /// # Squash and stretch, all driven by real velocity
    ///
    /// An ordered chain, not a blend, and the order is the judgement:
    ///
    /// - **dashing** wins outright and smears horizontally. A dash is the one
    ///   pose already leaning forward, so a second deformation on top of the
    ///   smear reads as two different things happening to one body.
    /// - **landing** fattens and flattens, decaying with the squash timer the
    ///   simulation keeps.
    /// - **airborne** stretches tall in proportion to rise speed and squats
    ///   mildly on a fast fall, with `sx = 1/sy` so the silhouette keeps its
    ///   area. A landing squash deliberately does not (see [`LAND_FLATTEN`]) —
    ///   an impact is allowed to look heavy, a jump is not allowed to look fat.
    ///
    /// All three are anchored at the FEET, so the figure stays planted however
    /// it deforms. That is the `(h_px - h)` term, and it is the reason the drawn
    /// top moves and the drawn bottom does not.
    ///
    /// # The art rect may be larger than the collision box
    ///
    /// Width, height and both offsets come from [`ArtGrid`] — from the record
    /// that carries the pixels — so one art pixel is exactly one world cell.
    /// Drawing into the collision box instead would rescale the art onto a
    /// finer, non-square grid than the terrain it stands on.
    ///
    /// # The step-up
    ///
    /// [`PlayerMotion::step_up_visual`] is added to the art rect's top and to
    /// the shear pivot, and to nothing else. The collision box snaps up a whole
    /// cell the instant a step-up resolves; this is the residue that eases back
    /// to zero, so cresting rubble reads as a stride rather than a teleport. It
    /// moves the DRAWN figure and never the box — `crate::player::place_body`
    /// applies exactly the same term to the placeholder, for exactly this
    /// reason.
    pub fn new(m: PlayerMotion, grid: ArtGrid) -> PlayerFigure {
        let (sx, sy) = scale(&m);

        // The undeformed art rect: centred on the box horizontally, aligned to
        // its feet, carrying the step-up residue.
        let art_x = m.x - grid.pad_x_px;
        let art_y = m.y - grid.pad_top_px + m.step_up_visual;

        let w = grid.w_px * sx;
        let h = grid.h_px * sy;

        PlayerFigure {
            x: art_x + (grid.w_px - w) / 2.0,
            y: art_y + (grid.h_px - h), // feet stay put
            w,
            h,

            pose: m.pose,
            facing: m.facing,
            state_t: m.state_t,
            clock_t: m.clock_t,
            phase: m.phase,

            lean: lean(m.accel_x),
            pivot_x: m.x + PLAYER_W / 2.0,
            pivot_y: m.y + PLAYER_H + m.step_up_visual,

            smear_dir: m.dashing.then_some(m.dash_dir),
        }
    }

    /// Horizontal displacement the lean applies to a point at world `y`.
    ///
    /// This is `ctx.transform(1, 0, -lean, 1, 0, 0)` about the feet, written
    /// out: a point `d` px above the pivot moves `lean * d` px sideways, a point
    /// at the pivot does not move, and a point below it moves the other way.
    /// Braking — acceleration opposing motion — therefore leans the figure BACK,
    /// which is exactly what sells the skid.
    ///
    /// This is the number `crate::player::quad_of` puts on a quad's top and
    /// bottom edges, and [`crate::shear`] is the argument for why displacing four
    /// vertices reproduces the canvas transform EXACTLY rather than
    /// approximately. Because the pivot is the drawn sole, the bottom edge's
    /// displacement is zero and the feet stay on whatever whole pixel the snap
    /// put them on — which is why only the top edge moves there.
    #[inline]
    pub fn shear_at(&self, y: f32) -> f32 {
        -self.lean * (y - self.pivot_y)
    }

    /// True when the lean is worth paying a transform for. See [`LEAN_DEADBAND`].
    #[inline]
    pub fn leaning(&self) -> bool {
        self.lean != 0.0
    }

    /// The dash after-images, FURTHEST AND FAINTEST FIRST.
    ///
    /// Draw these before the figure itself and the trail composites back to
    /// front, which is the order the TypeScript's descending loop produced and
    /// the only one in which the nearest ghost ends up on top. Empty unless
    /// dashing, so a caller can iterate unconditionally.
    ///
    /// Each ghost carries the figure's already-deformed rect: an after-image is
    /// the same smeared silhouette a moment ago, not a differently shaped one.
    pub fn ghosts(&self) -> impl Iterator<Item = Ghost> {
        let figure = *self;
        (0..DASH_GHOST_ALPHA.len())
            .rev()
            .filter_map(move |i| figure.ghost(i))
    }

    /// After-image `i + 1` steps back along the dash, or `None` if not dashing.
    fn ghost(&self, i: usize) -> Option<Ghost> {
        let dir = self.smear_dir?;
        let back = (i + 1) as f32 * DASH_GHOST_STEP;
        Some(Ghost {
            x: self.x - dir * back,
            y: self.y,
            alpha: DASH_GHOST_ALPHA[i],
        })
    }
}

/// The squash/stretch pair. See [`PlayerFigure::new`] for why it is a chain.
fn scale(m: &PlayerMotion) -> (f32, f32) {
    if m.dashing {
        (DASH_SMEAR_X, DASH_SMEAR_Y)
    } else if m.squash > 0.0 {
        (1.0 + m.squash * LAND_FATTEN, 1.0 - m.squash * LAND_FLATTEN)
    } else if !m.on_ground {
        // `vy` is positive downward, so `-vy` is rise speed.
        let s = (-m.vy / RISE_STRETCH_DIV + RISE_STRETCH_BIAS)
            .clamp(RISE_STRETCH_MIN, RISE_STRETCH_MAX);
        let sy = 1.0 + s;
        (1.0 / sy, sy)
    } else {
        (1.0, 1.0)
    }
}

/// Smoothed acceleration to a clamped, deadbanded shear.
///
/// `f32::clamp` panics only on a NaN BOUND and both bounds here are constants,
/// so it cannot — the same reasoning [`crate::daynight`] writes down for its own.
fn lean(accel_x: f32) -> f32 {
    let lean = (accel_x / LEAN_DIV).clamp(-LEAN_MAX, LEAN_MAX);
    if lean.abs() > LEAN_DEADBAND {
        lean
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::config::STEP_DT;
    use godgame_core::entities::{Loadout, NoProjectiles};
    use godgame_core::input::Intent;
    use godgame_core::sim::grid::CellGrid;
    use godgame_core::sim::materials::{EMPTY, block};
    use godgame_core::sim::worldgen::SpawnPoint;

    /// One fixed step of a body that has nowhere to fire.
    ///
    /// Every test in this module is about locomotion or about art, never about
    /// arrows, so the [`Loadout`] deliberately carries the pool that REFUSES
    /// every shot: if one of these ever starts depending on a projectile, it
    /// fails rather than quietly firing into a pool nobody inspects.
    fn step(player: &mut Player, intent: Intent, grid: &CellGrid) {
        let mut nowhere = NoProjectiles;
        player.step(STEP_DT, intent, grid, &mut Loadout::new(&mut nowhere));
    }

    /// Row of the hand-built floor, so a test asserts against a surface whose
    /// height it knows rather than against whatever the worldgen produced.
    const FLOOR: i32 = 100;

    /// Fixed steps in one second of simulated time.
    const ONE_SECOND: u32 = (1.0 / STEP_DT) as u32;

    /// A 2x3 art grid on a 2x3 box: the shipped case, where both pads are zero.
    fn flush_grid() -> ArtGrid {
        ArtGrid {
            w_px: PLAYER_W,
            h_px: PLAYER_H,
            pad_x_px: 0.0,
            pad_top_px: 0.0,
        }
    }

    /// A synthetic record, for the overhang cases content has not authored yet.
    fn def_sized(cells_w: i32, cells_h: i32) -> SpriteDef {
        SpriteDef {
            cells_w,
            cells_h,
            ..SPRITES[sprite::PLAYER as usize]
        }
    }

    fn figure(m: PlayerMotion) -> PlayerFigure {
        PlayerFigure::new(m, flush_grid())
    }

    /// The drawn sole: where the feet actually land on screen.
    fn feet(f: &PlayerFigure) -> f32 {
        f.y + f.h
    }

    // --- The art grid -------------------------------------------------------

    #[test]
    fn the_shipped_art_grid_exactly_fills_the_collision_box() {
        // The authored 2x3 against the 2x3 box. Both pads are zero and the art
        // rect IS the hitbox, which is the state the game currently ships in —
        // and the state every geometry test below is written against.
        let grid = ArtGrid::player();
        assert_eq!(grid.w_px, PLAYER_W);
        assert_eq!(grid.h_px, PLAYER_H);
        assert_eq!(grid.pad_x_px, 0.0);
        assert_eq!(grid.pad_top_px, 0.0);
    }

    #[test]
    fn art_overhang_is_centred_horizontally_and_grows_upward() {
        // Two cells wider and two taller than the box: a scarf and a hat.
        let grid = ArtGrid::from_def(&def_sized(PLAYER_CELLS_W + 2, PLAYER_CELLS_H + 2));
        let cell = CELL_SIZE as f32;
        assert_eq!(grid.pad_x_px, cell, "half the surplus on each side");
        assert_eq!(grid.pad_top_px, 2.0 * cell, "ALL of it above the feet");
        assert_eq!(grid.w_px, PLAYER_W + 2.0 * cell);
        assert_eq!(grid.h_px, PLAYER_H + 2.0 * cell);
    }

    #[test]
    fn an_odd_horizontal_surplus_lands_the_origin_on_a_half_cell() {
        // Deliberate, and documented on `pad_x_px`: rounding to a whole cell
        // would push the figure visibly off centre, and the blit is still one
        // art pixel per world cell either way.
        let grid = ArtGrid::from_def(&def_sized(PLAYER_CELLS_W + 1, PLAYER_CELLS_H));
        assert_eq!(grid.pad_x_px, CELL_SIZE as f32 / 2.0);
    }

    #[test]
    fn a_grown_art_grid_moves_the_drawn_figure_and_not_the_body() {
        // The whole point of the content/code split: an artist may hang a limb
        // outside the silhouette and the hitbox must not notice.
        let m = PlayerMotion::default();
        let flush = PlayerFigure::new(m, flush_grid());
        let big = PlayerFigure::new(m, ArtGrid::from_def(&def_sized(4, 5)));

        assert_eq!(feet(&big), feet(&flush), "the feet moved");
        assert_eq!(big.pivot_x, flush.pivot_x, "the shear pivot moved");
        assert_eq!(big.pivot_y, flush.pivot_y, "the shear pivot moved");
        assert!(big.h > flush.h, "the art did not grow");
        // Centred: the surplus is split evenly around the box.
        let left = flush.x - big.x;
        let right = (big.x + big.w) - (flush.x + flush.w);
        assert!((left - right).abs() < 1e-6, "{left} left vs {right} right");
    }

    // --- Poses --------------------------------------------------------------

    #[test]
    fn every_pose_the_player_can_reach_is_authored_in_the_content_record() {
        // The TypeScript threw at module load for this. `crate::player` cannot
        // throw and does not want to: a pose the sheet has no sequence for falls
        // back to the sheet's FIRST sequence, which draws a plausible frame and
        // reports nothing. So a content change that drops a sequence has exactly
        // one thing standing in its way, and this is it.
        let seq = SPRITES[sprite::PLAYER as usize]
            .seq
            .expect("the player record has no sequences");
        for pose in POSES {
            let want = seq_state(pose);
            assert!(
                seq.iter().any(|s| s.state == want),
                "content/sprites/player.toml has no '{pose:?}' sequence, so a \
                 player who reaches it silently plays '{:?}' instead",
                seq[0].state
            );
        }
    }

    #[test]
    fn no_two_poses_cross_to_the_same_authored_sequence() {
        // This is `satisfies Record<AnimState, 1>` as a runtime assertion, and it
        // outlived the id table it was first written against. `crate::player`
        // crosses every drawn frame through `seq_state`, so a duplicated arm
        // there would make one pose permanently play another's frames — a bug
        // whose only symptom is that the character occasionally does the wrong
        // thing, which nothing else in the suite would notice.
        let mut seen = [false; POSE_COUNT];
        for pose in POSES {
            let i = seq_state(pose) as usize;
            assert!(i < POSE_COUNT, "{pose:?} crosses out of range at {i}");
            assert!(
                !seen[i],
                "{pose:?} shares a sequence state with another pose"
            );
            seen[i] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "a sequence state no pose ever reaches"
        );
    }

    // --- Squash and stretch -------------------------------------------------

    #[test]
    fn a_body_at_rest_is_drawn_at_exactly_its_own_size() {
        let f = figure(PlayerMotion::default());
        assert_eq!((f.x, f.y, f.w, f.h), (0.0, 0.0, PLAYER_W, PLAYER_H));
        assert_eq!(f.lean, 0.0);
        assert_eq!(f.smear_dir, None);
    }

    #[test]
    fn a_landing_squash_fattens_and_flattens_the_figure_about_its_feet() {
        let rest = figure(PlayerMotion::default());
        let f = figure(PlayerMotion {
            squash: 1.0,
            ..PlayerMotion::default()
        });

        assert!(f.w > rest.w, "a landing did not widen the figure");
        assert!(f.h < rest.h, "a landing did not flatten the figure");
        assert_eq!(feet(&f), feet(&rest), "the feet left the floor");
        // Still centred on the box: fattening must not shove the figure sideways.
        assert_eq!(f.x + f.w / 2.0, rest.x + rest.w / 2.0);
    }

    #[test]
    fn a_landing_compresses_the_silhouette_where_the_air_stretch_conserves_it() {
        // The one judgement call in the squash chain, and the easiest to
        // "correct" by accident. The rise stretch is exactly area-preserving
        // (`sx = 1/sy`); the landing pair is two independent knobs and is NOT,
        // because an impact should read as the body being driven into the floor
        // rather than as it bulging sideways to keep its volume. Conserving area
        // at this flatten would need a fatten of 0.67 against the tuned 0.45.
        let neutral = PLAYER_W * PLAYER_H;

        let landed = figure(PlayerMotion {
            squash: 1.0,
            ..PlayerMotion::default()
        });
        assert!(
            landed.w * landed.h < neutral,
            "the landing kept its area: {} vs {neutral}",
            landed.w * landed.h
        );

        let rising = figure(PlayerMotion {
            on_ground: false,
            vy: -600.0,
            ..PlayerMotion::default()
        });
        assert!(
            (rising.w * rising.h - neutral).abs() < 1e-3,
            "the air stretch stopped conserving area: {}",
            rising.w * rising.h
        );

        // And the compression is a widening one, not a shrink: the figure comes
        // out WIDER than neutral even as it loses area, which is what stops it
        // reading as the character simply getting smaller.
        assert!(landed.w > PLAYER_W, "the landing narrowed the figure");
    }

    #[test]
    fn the_squash_stays_planted_across_its_whole_range() {
        let rest = feet(&figure(PlayerMotion::default()));
        for squash in [0.001_f32, 0.25, 0.5, 0.75, 1.0] {
            let f = figure(PlayerMotion {
                squash,
                ..PlayerMotion::default()
            });
            assert!(
                (feet(&f) - rest).abs() < 1e-4,
                "squash {squash} lifted the feet to {}",
                feet(&f)
            );
        }
    }

    #[test]
    fn a_fast_rise_stretches_the_figure_tall_and_narrow_at_constant_area() {
        let f = figure(PlayerMotion {
            on_ground: false,
            vy: -600.0, // negative is up
            ..PlayerMotion::default()
        });
        assert!(f.h > PLAYER_H, "a rise did not stretch the figure");
        assert!(f.w < PLAYER_W, "a rise did not narrow the figure");
        assert!(
            (f.w * f.h - PLAYER_W * PLAYER_H).abs() < 1e-3,
            "the air stretch lost area: {}",
            f.w * f.h
        );
        assert_eq!(feet(&f), PLAYER_H, "the stretch grew downward");
    }

    #[test]
    fn a_fast_fall_squats_the_figure_rather_than_stretching_it() {
        let f = figure(PlayerMotion {
            on_ground: false,
            vy: 900.0,
            ..PlayerMotion::default()
        });
        assert!(f.h < PLAYER_H, "a plummet stretched the figure");
        assert!(f.w > PLAYER_W);
        assert_eq!(feet(&f), PLAYER_H, "the squat left the feet behind");
    }

    #[test]
    fn the_apex_of_a_jump_is_very_slightly_tall_rather_than_neutral() {
        // `RISE_STRETCH_BIAS`. Without it the figure snaps back to the standing
        // silhouette for the frame or two it hangs at the top, which is the one
        // moment the eye is most likely to be on it.
        let f = figure(PlayerMotion {
            on_ground: false,
            vy: 0.0,
            ..PlayerMotion::default()
        });
        assert!(f.h > PLAYER_H, "the apex was neutral");
        assert!(
            f.h < PLAYER_H * 1.1,
            "the apex is supposed to be subtle, was {}",
            f.h
        );
    }

    #[test]
    fn the_air_stretch_is_clamped_at_both_ends() {
        // An absurd velocity in either direction must land exactly on the bound,
        // because the bound is what stops a long fall from drawing a hairline.
        let up = figure(PlayerMotion {
            on_ground: false,
            vy: -1.0e6,
            ..PlayerMotion::default()
        });
        assert_eq!(up.h, PLAYER_H * (1.0 + RISE_STRETCH_MAX));

        let down = figure(PlayerMotion {
            on_ground: false,
            vy: 1.0e6,
            ..PlayerMotion::default()
        });
        assert_eq!(down.h, PLAYER_H * (1.0 + RISE_STRETCH_MIN));
        assert!(down.h > 0.0, "the clamp let the figure collapse");
    }

    #[test]
    fn the_deformations_resolve_in_one_fixed_order_rather_than_blending() {
        // Dash beats landing beats airborne. Each of these really can be true at
        // once — you can dash the frame you land, and you are still technically
        // airborne for the step the landing squash is set on.
        let all_at_once = PlayerMotion {
            dashing: true,
            squash: 1.0,
            on_ground: false,
            vy: -600.0,
            ..PlayerMotion::default()
        };
        let dashing = figure(all_at_once);
        assert_eq!(dashing.w, PLAYER_W * DASH_SMEAR_X);
        assert_eq!(dashing.h, PLAYER_H * DASH_SMEAR_Y);

        let landing = figure(PlayerMotion {
            dashing: false,
            ..all_at_once
        });
        assert_eq!(landing.w, PLAYER_W * (1.0 + LAND_FATTEN));

        let airborne = figure(PlayerMotion {
            dashing: false,
            squash: 0.0,
            ..all_at_once
        });
        assert!(airborne.h > PLAYER_H);
    }

    #[test]
    fn a_dash_smears_the_figure_wide_and_flat_and_keeps_it_on_the_floor() {
        let f = figure(PlayerMotion {
            dashing: true,
            ..PlayerMotion::default()
        });
        assert!(f.w > PLAYER_W && f.h < PLAYER_H);
        assert_eq!(feet(&f), PLAYER_H, "the smear lifted the feet");
    }

    // --- The lean -----------------------------------------------------------

    #[test]
    fn accelerating_right_shears_the_figure_forward_over_its_feet() {
        let f = figure(PlayerMotion {
            accel_x: LEAN_DIV * 0.5, // half of full lean, well inside the clamp
            ..PlayerMotion::default()
        });
        assert!(f.lean > 0.0);
        // A point at head height moves RIGHT; the feet do not move at all.
        assert!(f.shear_at(f.pivot_y - PLAYER_H) > 0.0, "the head lagged");
        assert_eq!(f.shear_at(f.pivot_y), 0.0, "the feet slid");
    }

    #[test]
    fn braking_leans_the_figure_back_which_is_what_sells_the_skid() {
        // Running right, accelerating LEFT: the body is still travelling right,
        // so the figure has to lie back against its own motion. This is the case
        // a lean driven by velocity instead of acceleration would get backwards,
        // and it is the whole reason the simulation smooths `dvx/dt`.
        let f = figure(PlayerMotion {
            accel_x: -LEAN_DIV * 0.5,
            ..PlayerMotion::default()
        });
        assert!(f.lean < 0.0);
        assert!(
            f.shear_at(f.pivot_y - PLAYER_H) < 0.0,
            "leaned into the skid"
        );
    }

    #[test]
    fn the_lean_is_clamped_so_no_acceleration_can_lay_the_figure_flat() {
        for accel in [1.0e7_f32, -1.0e7] {
            let f = figure(PlayerMotion {
                accel_x: accel,
                ..PlayerMotion::default()
            });
            assert_eq!(f.lean.abs(), LEAN_MAX, "lean escaped the clamp");
            // At the bound a three-cell figure's head moves under one cell.
            let head = f.shear_at(f.pivot_y - PLAYER_H).abs();
            assert!(head < CELL_SIZE as f32, "the head swung {head} px");
        }
    }

    #[test]
    fn a_negligible_acceleration_is_no_lean_at_all() {
        // The smoothed acceleration asymptotes rather than reaching zero, so
        // without the deadband a game spent standing still would pay for a
        // transform every frame and carry a permanent sub-pixel shear.
        let below = figure(PlayerMotion {
            accel_x: LEAN_DIV * LEAN_DEADBAND * 0.9,
            ..PlayerMotion::default()
        });
        assert_eq!(below.lean, 0.0);
        assert!(!below.leaning());

        let above = figure(PlayerMotion {
            accel_x: LEAN_DIV * LEAN_DEADBAND * 1.1,
            ..PlayerMotion::default()
        });
        assert!(above.leaning());
    }

    #[test]
    fn the_shear_pivots_on_the_drawn_feet_and_not_on_the_collision_box() {
        // Mid-step-up the two are a cell apart, and pivoting on the box would
        // swing the whole figure sideways for exactly as long as the ease lasts.
        let m = PlayerMotion {
            accel_x: LEAN_DIV,
            step_up_visual: CELL_SIZE as f32,
            ..PlayerMotion::default()
        };
        let f = figure(m);
        assert_eq!(f.pivot_y, m.y + PLAYER_H + m.step_up_visual);
        assert_eq!(f.shear_at(f.pivot_y), 0.0);
        assert_ne!(f.shear_at(m.y + PLAYER_H), 0.0, "pivoted on the box");
    }

    #[test]
    fn the_shear_is_linear_in_height_above_the_feet() {
        // It is a shear and not a rotation: the offset is proportional to height
        // and nothing about the figure's width changes with it.
        let f = figure(PlayerMotion {
            accel_x: LEAN_DIV * 0.5,
            ..PlayerMotion::default()
        });
        let one = f.shear_at(f.pivot_y - 1.0);
        for d in [2.0_f32, 5.0, 10.0, 15.0] {
            let got = f.shear_at(f.pivot_y - d);
            assert!(
                (got - one * d).abs() < 1e-4,
                "shear at {d} px up was {got}, not {}",
                one * d
            );
        }
    }

    // --- The dash trail -----------------------------------------------------

    #[test]
    fn a_figure_that_is_not_dashing_has_no_after_images() {
        let f = figure(PlayerMotion::default());
        assert_eq!(f.ghosts().count(), 0);
    }

    #[test]
    fn the_dash_trail_falls_behind_the_dash_and_fades_with_distance() {
        let f = figure(PlayerMotion {
            dashing: true,
            dash_dir: 1.0,
            ..PlayerMotion::default()
        });
        let trail: Vec<Ghost> = f.ghosts().collect();
        assert_eq!(trail.len(), DASH_GHOST_ALPHA.len());

        for g in &trail {
            assert!(g.x < f.x, "an after-image ran ahead of the figure");
            assert_eq!(g.y, f.y, "an after-image drifted vertically");
            assert!(g.alpha > 0.0 && g.alpha < 1.0);
        }
        // Evenly spaced, one `DASH_GHOST_STEP` apart, starting one step back.
        assert_eq!(trail[1].x, f.x - DASH_GHOST_STEP);
        assert_eq!(trail[0].x, f.x - 2.0 * DASH_GHOST_STEP);
    }

    #[test]
    fn the_after_images_come_out_furthest_first_so_the_nearest_lands_on_top() {
        // Composite order, not decoration: drawn the other way round the faint
        // far ghost would be painted over the solid near one, and the trail would
        // read as a stutter rather than as speed.
        let f = figure(PlayerMotion {
            dashing: true,
            ..PlayerMotion::default()
        });
        let trail: Vec<Ghost> = f.ghosts().collect();
        assert!(
            trail[0].alpha < trail[1].alpha,
            "the trail fades the wrong way"
        );
        assert!((f.x - trail[0].x).abs() > (f.x - trail[1].x).abs());
    }

    #[test]
    fn a_leftward_dash_trails_to_the_right() {
        let f = figure(PlayerMotion {
            dashing: true,
            dash_dir: -1.0,
            ..PlayerMotion::default()
        });
        for g in f.ghosts() {
            assert!(g.x > f.x, "the trail led the dash instead of following it");
        }
    }

    #[test]
    fn the_after_images_carry_the_same_smeared_silhouette_as_the_figure() {
        // A ghost is where the figure WAS, not a differently shaped thing. The
        // rect is the figure's own, which is why `Ghost` carries only an offset
        // and an alpha.
        let f = figure(PlayerMotion {
            dashing: true,
            ..PlayerMotion::default()
        });
        assert_eq!(f.w, PLAYER_W * DASH_SMEAR_X);
        assert!(f.ghosts().all(|g| g.y == f.y));
    }

    // --- The clocks and the facing ------------------------------------------

    #[test]
    fn the_figure_forwards_the_three_clocks_the_rasteriser_reads() {
        // Three different notions of time, and they are not interchangeable:
        // `state_t` resets on transition so a "once" pose fires from its first
        // frame, `clock_t` never resets so ambient loops stay out of phase with
        // state changes, and `phase` is driven by real speed so the legs do not
        // slide when the body is moving at half pace.
        let m = PlayerMotion {
            pose: AnimState::Run,
            state_t: 0.25,
            clock_t: 91.5,
            phase: 0.5,
            facing: -1.0,
            ..PlayerMotion::default()
        };
        let f = figure(m);
        assert_eq!(f.pose, AnimState::Run);
        assert_eq!((f.state_t, f.clock_t, f.phase), (0.25, 91.5, 0.5));
        assert_eq!(f.facing, -1.0);
    }

    #[test]
    fn facing_left_mirrors_the_art_without_moving_the_rect() {
        // `facing` is handed to the rasteriser, which mirrors about the rect's
        // OWN vertical axis. Nothing about the geometry changes here, and that
        // is the assertion — a flip that moved the rect would fling the figure
        // across the screen.
        let right = figure(PlayerMotion::default());
        let left = figure(PlayerMotion {
            facing: -1.0,
            ..PlayerMotion::default()
        });
        assert_eq!(
            (left.x, left.y, left.w, left.h),
            (right.x, right.y, right.w, right.h)
        );
    }

    // --- The step-up, against a real grid -----------------------------------

    /// A flat stone floor at [`FLOOR`] that rises one cell `ledge_ahead` columns
    /// right of the body: a step, not a wall.
    fn world_with_ledge(ledge_ahead: i32) -> (CellGrid, Player) {
        let mut grid = CellGrid::new(128, 128);
        for y in 0..grid.rows() {
            for x in 0..grid.cols() {
                grid.set(x, y, if y >= FLOOR { block::STONE } else { EMPTY });
            }
        }
        for x in (64 + ledge_ahead)..grid.cols() {
            grid.set(x, FLOOR - 1, block::STONE);
        }
        let spawn = SpawnPoint {
            x: (64 * CELL_SIZE) as f32,
            y: ((FLOOR - 8) * CELL_SIZE) as f32,
        };
        let player = Player::new(spawn);
        (grid, player)
    }

    fn run(player: &mut Player, grid: &CellGrid, intent: Intent, n: u32) {
        for i in 0..n {
            step(player, intent.for_substep(i), grid);
        }
    }

    #[test]
    fn cresting_a_ledge_eases_the_figure_up_rather_than_snapping_it() {
        // The flagship of the whole module, and the reason `step_up_visual`
        // exists at all. The COLLISION BOX teleports a whole cell the instant a
        // step-up resolves — `crate::player`'s own test asserts exactly that —
        // and if the drawing followed the box the figure would jump five px in
        // one frame and read as a hitch. What must happen instead is that the
        // drawn top climbs over several frames and never once moves DOWN.
        let (grid, mut player) = world_with_ledge(6);
        run(&mut player, &grid, Intent::default(), ONE_SECOND);
        assert!(player.on_ground, "the fixture never landed");

        let start = PlayerFigure::of(&player).y;
        let right = Intent {
            dir_x: 1.0,
            ..Intent::default()
        };

        let mut box_jump = 0.0_f32;
        let mut biggest_figure_jump = 0.0_f32;
        let mut previous = PlayerFigure::of(&player);

        for i in 0..ONE_SECOND {
            let before = player.y;
            step(&mut player, right.for_substep(i), &grid);
            box_jump = box_jump.max(before - player.y);

            let now = PlayerFigure::of(&player);
            biggest_figure_jump = biggest_figure_jump.max(previous.y - now.y);
            // Monotone: cresting must never drop the figure, or the ease reads
            // as a bounce.
            assert!(
                now.y <= previous.y + 1e-4,
                "the figure sank {} px while climbing",
                now.y - previous.y
            );
            previous = now;
        }

        let cell = CELL_SIZE as f32;
        assert_eq!(box_jump, cell, "the box did not step up a whole cell");
        assert_eq!(previous.y, start - cell, "the figure never caught up");
        assert!(
            biggest_figure_jump < cell,
            "the figure snapped {biggest_figure_jump} px in one frame — the whole \
             cell, which is the teleport this exists to hide"
        );
    }

    #[test]
    fn the_step_up_residue_moves_the_drawing_and_nothing_else() {
        // It is added to the art rect and to the shear pivot, and to no other
        // term. Getting that wrong would decouple the figure from its own lean.
        let cell = CELL_SIZE as f32;
        let plain = figure(PlayerMotion::default());
        let stepping = figure(PlayerMotion {
            step_up_visual: cell,
            ..PlayerMotion::default()
        });

        assert_eq!(stepping.y, plain.y + cell);
        assert_eq!(stepping.pivot_y, plain.pivot_y + cell);
        assert_eq!(stepping.x, plain.x, "the residue moved the figure sideways");
        assert_eq!((stepping.w, stepping.h), (plain.w, plain.h));
    }

    #[test]
    fn a_live_body_produces_the_same_figure_as_its_gathered_motion() {
        // `PlayerFigure::of` is only `PlayerMotion::of` plus `ArtGrid::player`,
        // and this is what says so — every test above works on hand-written
        // motions, and this is the one line that ties them to a real body.
        let (grid, mut player) = world_with_ledge(6);
        run(&mut player, &grid, Intent::default(), ONE_SECOND / 2);

        let direct = PlayerFigure::of(&player);
        let via_motion = PlayerFigure::new(PlayerMotion::of(&player), ArtGrid::player());
        assert_eq!(direct, via_motion);
    }

    #[test]
    fn the_figure_is_published_in_sim_space_with_y_growing_downward() {
        // The convention flip to Bevy's +y up happens at the `Transform` write
        // and nowhere else. A body one cell FURTHER DOWN the world must produce
        // a figure with a LARGER y here; anything else means the flip leaked in.
        let cell = CELL_SIZE as f32;
        let high = figure(PlayerMotion::default());
        let low = figure(PlayerMotion {
            y: cell,
            ..PlayerMotion::default()
        });
        assert_eq!(low.y, high.y + cell);
        assert!(feet(&low) > feet(&high));
    }
}

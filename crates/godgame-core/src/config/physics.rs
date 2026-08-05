//! Player body and locomotion.
//!
//! All rates are expressed per second and integrated with a fixed timestep (see
//! [`STEP_DT`]), so behaviour is frame-rate independent.
//!
//! Every distance-bearing number below is written as the value that felt right
//! for a `PHYS_TUNED_H`-tall character and then scaled to the current one. That
//! matters because the numbers are absolute px/s: shrinking the character
//! without touching them leaves it covering the same ground per second at a
//! fraction of the size, which reads as absurdly fast and floaty even though
//! nothing about the physics changed.
//!
//! The scaling rule is dimensional. Treat body height as the unit of length and
//! keep TIME fixed, so the action stays paced the same:
//!   - lengths scale by S
//!   - velocities (length/time) scale by S
//!   - accelerations (length/time^2) scale by S
//!   - durations, ratios and fractions are NOT scaled
//!
//! Under it, jump height in body-heights, run speed in bodies/second and time to
//! apex are all preserved exactly — the character feels identical, just smaller.
//!
//! [`scaled`] is public because combat and the ability mechanics are tuned in
//! the same reference frame and must apply the identical factor. Anything that
//! multiplies a velocity by its own private constant has forked the rule.

use super::world::CELL_SIZE;

/// Seconds per physics step.
pub const STEP_DT: f32 = 1.0 / 120.0;

/// Clamp on catch-up steps per frame, to avoid a spiral of death.
pub const MAX_STEPS_PER_FRAME: u32 = 5;

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------

/// Player collision box width, in cells.
///
/// The box being an exact whole number of cells is load-bearing for how the
/// character *reads*. The sprite is authored on a grid of logical pixels and
/// drawn into a box of this size, so if the box is not a whole number of cells,
/// one sprite pixel does not line up with one world cell and the character
/// renders at a visibly finer grain than the terrain it stands on. (It was once
/// 20x24 px = 4 x 4.8 cells against a 16x16 sprite, so sprite pixels were
/// 1.25 x 1.5 px — four times finer than a cell, and not even square, which
/// stretched the art horizontally.)
///
/// THE ART GRID IS NOT HERE, and that is the invariant, not a footnote. The
/// sprite's own padding is derived from the compiled sprite record that actually
/// carries the pixels, because declaring the grid here as a number AND in the
/// art table as a row count meant two sources of truth for one fact. Content
/// owns what a pixel looks like; code owns what the body does. An artist may
/// grow the art grid to hang a scarf or a swung limb outside the silhouette; the
/// surplus becomes overhang (centred horizontally, aligned to the feet) and the
/// hitbox, the physics tuned against it, and every constant scaled off
/// [`PLAYER_H`] all stay exactly where they were.
pub const PLAYER_CELLS_W: i32 = 2;

/// Player collision box height, in cells. See [`PLAYER_CELLS_W`].
pub const PLAYER_CELLS_H: i32 = 3;

/// Player collision box width, in world px.
pub const PLAYER_W: f32 = (PLAYER_CELLS_W * CELL_SIZE) as f32; // 10

/// Player collision box height, in world px.
pub const PLAYER_H: f32 = (PLAYER_CELLS_H * CELL_SIZE) as f32; // 15

// ---------------------------------------------------------------------------
// The dimensional scaling rule
// ---------------------------------------------------------------------------

/// Character height the raw numbers below were originally tuned against.
const PHYS_TUNED_H: f32 = 24.0;

/// Linear scale of the current character against that reference.
const PHYS_SIZE_SCALE: f32 = PLAYER_H / PHYS_TUNED_H;

/// Overall movement tempo — a pure FEEL knob, deliberately independent of body
/// size.
///
/// Turn this down to make the character cover less ground and jump less high
/// relative to itself; turn it up for a floatier, rangier game.
///
/// It is separate from the size scale on purpose. The size scale exists to keep
/// the feel CONSTANT as the character is resized; changing it would resize the
/// character rather than retune it. This is the one to turn when the movement is
/// simply too much.
///
/// Because velocities and accelerations are both multiplied by it, distances
/// scale by it while DURATIONS stay put: a lower tempo means shorter, tighter
/// jumps that still take the same time to peak, rather than sluggish ones.
const MOVE_TEMPO: f32 = 0.55;

/// Combined factor actually applied to every distance-bearing tunable.
pub const PHYS_SCALE: f32 = PHYS_SIZE_SCALE * MOVE_TEMPO;

/// Scale a length, velocity or acceleration. Never apply to a duration.
#[inline]
pub const fn scaled(v: f32) -> f32 {
    v * PHYS_SCALE
}

// ---------------------------------------------------------------------------
// Ground and air movement
// ---------------------------------------------------------------------------

/// Downward acceleration, px/s^2.
pub const GRAVITY: f32 = scaled(2400.0);

/// Terminal fall speed, px/s.
pub const MAX_FALL_SPEED: f32 = scaled(1400.0);

/// Ground acceleration, px/s^2.
pub const MOVE_ACCEL: f32 = scaled(6000.0);

/// In-air acceleration, px/s^2.
pub const AIR_ACCEL: f32 = scaled(3600.0);

/// Horizontal speed cap, px/s.
pub const MAX_RUN_SPEED: f32 = scaled(360.0);

/// Deceleration on normal ground, px/s^2.
pub const GROUND_FRICTION: f32 = scaled(5000.0);

/// Deceleration on ice, px/s^2 — slippery by being an order of magnitude lower.
pub const ICE_FRICTION: f32 = scaled(350.0);

/// Deceleration in air, px/s^2.
pub const AIR_FRICTION: f32 = scaled(900.0);

/// Initial upward velocity of a jump, px/s.
pub const JUMP_SPEED: f32 = scaled(780.0);

/// Fraction of upward velocity retained when the jump button is released early.
pub const JUMP_CUT: f32 = 0.45;

/// Seconds after leaving the ground during which a jump still works.
pub const COYOTE_TIME: f32 = 0.1;

/// Seconds a jump press is remembered before landing.
pub const JUMP_BUFFER: f32 = 0.12;

// ---------------------------------------------------------------------------
// Swimming
// ---------------------------------------------------------------------------

/// Velocity retained per SECOND while submerged.
///
/// The old model was a per-STEP multiplier of 0.86. At 120 Hz that is 0.86^120
/// per second — it annihilated any impulse in about a tenth of a second, which
/// is why water felt like treacle. Drag is now expressed per second and applied
/// as `drag.powf(dt)`, so it is timestep-independent and means what it says.
pub const LIQUID_DRAG: f32 = 0.06;

/// Gravity multiplier while swimming.
pub const LIQUID_GRAVITY_SCALE: f32 = 0.35;

/// Fraction of the body that must be in liquid before swim rules take over.
///
/// Below this you are WADING: normal walking and normal jumping, just damped.
/// Without this test a single water cell touching a foot flipped the player into
/// full swim mode, so puddles disabled jumping.
pub const SWIM_SUBMERGE_MIN: f32 = 0.55;

/// Upward acceleration while holding up or jump, px/s^2.
///
/// Swimming is a sustained stroke, not a one-shot impulse fired on a jump press
/// — which could only ever make the player bob upward one tap at a time.
pub const SWIM_ACCEL: f32 = scaled(1500.0);

/// Downward acceleration while holding down, px/s^2.
pub const SWIM_SINK_ACCEL: f32 = scaled(900.0);

/// Ascent cap while swimming, px/s.
pub const SWIM_MAX_UP: f32 = scaled(250.0);

/// Terminal sink rate while swimming, px/s.
pub const SWIM_MAX_DOWN: f32 = scaled(320.0);

/// Horizontal acceleration while swimming, px/s^2 (versus [`MOVE_ACCEL`] on land).
pub const SWIM_ACCEL_H: f32 = scaled(2200.0);

/// Horizontal speed cap while swimming, px/s.
pub const SWIM_MAX_SPEED_H: f32 = scaled(190.0);

/// Upward acceleration at full submersion, px/s^2.
///
/// Buoyancy pushes toward neutral at the surface so the player bobs and floats
/// rather than sinking like a stone, scaled by how submerged they are. Scaled
/// alongside [`GRAVITY`] — the float equilibrium is the RATIO of the two, so
/// scaling both keeps the player settling at the same 0.73 submersion
/// regardless of size.
pub const SWIM_BUOYANCY: f32 = scaled(1150.0);

/// Upward kick when a swim stroke breaks the surface, px/s.
///
/// So you can actually climb out of water instead of bobbing against the lip
/// forever.
pub const SWIM_OUT_BOOST: f32 = scaled(430.0);

/// Below this submersion a swim stroke counts as breaking the surface.
pub const SWIM_EXIT_SUBMERSION: f32 = 0.8;

// ---------------------------------------------------------------------------
// Climbing
// ---------------------------------------------------------------------------

/// Upward climb speed, px/s.
///
/// Ladders, ropes and vines — anything content tags `climb` — suspend gravity
/// and replace vertical movement outright, so these are speeds and not
/// accelerations: a climb that has to be accelerated into reads as ice, and the
/// whole point of a ladder is that it is the one place in a falling-sand world
/// where the player is in exact control of their altitude.
pub const CLIMB_SPEED_UP: f32 = scaled(150.0);

/// Downward climb speed, px/s.
///
/// Faster than up for the same reason it is in every game that has ever had a
/// ladder — sliding down is not the same effort as hauling yourself up — but not
/// so much faster that a descent is a fall.
pub const CLIMB_SPEED_DOWN: f32 = scaled(210.0);

/// Sideways speed while attached to a climbable, px/s.
///
/// Deliberately slow rather than zero: you must be able to shuffle onto an
/// adjacent rope or step off onto a ledge, but a ladder must never be a faster
/// way to cross a room than the floor is.
pub const CLIMB_SPEED_H: f32 = scaled(90.0);

/// Seconds after jumping off a ladder during which you cannot re-grab it.
///
/// Without this, jumping while inside the ladder's own cells re-mounts on the
/// very next step and the player is welded to it. Long enough to clear the rungs
/// at [`JUMP_SPEED`], short enough that a deliberate re-grab on the way past
/// still works.
pub const CLIMB_REMOUNT_LOCK: f32 = 0.22;

/// Upward velocity kept when a climb ends because the body left the last rung.
///
/// Cresting a ladder with `vy = 0` drops you straight back onto it, which reads
/// as the ladder refusing to let go. A small kick carries the feet over the lip.
pub const CLIMB_TOP_BOOST: f32 = scaled(190.0);

// ---------------------------------------------------------------------------
// Terrain traversal
// ---------------------------------------------------------------------------

/// Seconds of one-way-platform pass-through granted by a deliberate down input.
///
/// The player has to be UNSOLID for long enough that gravity carries the whole
/// body below the platform's top face, or the next step's resolve catches them
/// again and the drop silently fails. At [`GRAVITY`] and a 3-cell body that
/// takes about a tenth of a second; this is comfortably more, and short enough
/// that it cannot be held down as a general "ignore platforms" mode.
pub const DROP_THROUGH_TIME: f32 = 0.24;

/// Height, in cells, a blocked horizontal move may be retried lifted by.
///
/// Terrain here is loose grain, so a single settled sand cell is a wall to a
/// pure axis-resolved box — you stop dead on bumps you should walk over. When a
/// horizontal move is blocked, we retry it raised; if the raised box is clear,
/// the move is allowed and the player rises.
///
/// Sized RELATIVE TO THE PLAYER, not in absolute px. At 2 cells this was 10px
/// against a 50px-tall character — a fifth of its height, a natural stride. The
/// character is now 3 cells tall, so 2 cells would be two-thirds of its own
/// height and it would climb essentially any ledge shorter than itself, which
/// removes the reason to jump at all. One cell is a third of its height, which
/// is the same feel the original ratio had.
pub const STEP_UP_CELLS: i32 = 1;

/// Step-up height in world px.
pub const STEP_UP_MAX: f32 = (STEP_UP_CELLS * CELL_SIZE) as f32; // 5

/// Visual rise rate, px/s. The box snaps; the drawn sprite eases to hide it.
pub const STEP_UP_SMOOTH: f32 = scaled(420.0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scaling_rule_preserves_jump_height_in_body_heights() {
        // Time to apex is v/g, which the scale cancels out of entirely — that is
        // the whole claim the dimensional rule makes.
        let apex_t = JUMP_SPEED / GRAVITY;
        let unscaled_apex_t = 780.0 / 2400.0;
        assert!(
            (apex_t - unscaled_apex_t).abs() < 1e-6,
            "time to apex moved"
        );

        // Jump height in body-heights is v^2 / (2 g H). Substituting
        // v = 780*S, g = 2400*S, H = PHYS_TUNED_H*SIZE and S = SIZE*TEMPO, the
        // SIZE terms cancel and only the tempo survives:
        //
        //     h/H = 780^2 * TEMPO / (2 * 2400 * PHYS_TUNED_H)
        //
        // That cancellation IS the dimensional rule. Resizing the character
        // cannot move this number; only retuning the tempo can.
        let height_in_bodies = (JUMP_SPEED * JUMP_SPEED) / (2.0 * GRAVITY * PLAYER_H);
        let reference = (780.0f32 * 780.0 * MOVE_TEMPO) / (2.0 * 2400.0 * PHYS_TUNED_H);
        assert!(
            (height_in_bodies - reference).abs() < 1e-4,
            "jump height in body-heights moved: {height_in_bodies} vs {reference}"
        );
    }

    #[test]
    fn phys_scale_is_what_the_bestiary_expects() {
        // MobDefs multiplies authored creature speeds by this, and the mob
        // regression gate compares the products verbatim.
        assert!(
            (PHYS_SCALE - 0.34375).abs() < 1e-7,
            "PHYS_SCALE = {PHYS_SCALE}"
        );
    }
}

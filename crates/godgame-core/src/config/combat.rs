//! Combat — the floor the whole item ladder is measured against.
//!
//! A weapon item supplies `damage`, `knockback`, `swing_time` and
//! `projectile_speed` and OVERRIDES the corresponding number here. What stays in
//! this module is either the bare-handed case or a fact about how the game reads
//! that no single weapon gets to choose.
//!
//! Damage and durations are NOT body-scaled — a hit point is a hit point and a
//! second is a second. Knockback, reach and projectile speeds are lengths or
//! velocities, so they go through [`scaled`](super::physics::scaled) like
//! everything in `super::physics`.

use super::physics::scaled;

/// Player hit points at full health.
///
/// The scale every `damage` number in content is authored against — a 5-damage
/// fist is 20 hits, a 220/s lava pool is death in under half a second — so
/// moving it retunes the whole game's lethality at once rather than one hazard.
pub const MAX_HEALTH: f32 = 100.0;

// ---------------------------------------------------------------------------
// Melee
// ---------------------------------------------------------------------------

/// Bare-handed melee damage.
///
/// Keeps its `PUNCH_` name because it is the number the combat resolver has
/// always read; it is now the bare-hands case of a general rule rather than the
/// only rule there is.
pub const PUNCH_DAMAGE: f32 = 5.0;

/// Seconds between bare-handed swings. Fists are fast and weak, by design.
pub const PUNCH_SWING_TIME: f32 = 0.3;

/// px/s of impulse a bare fist puts into what it hits.
pub const PUNCH_KNOCKBACK: f32 = scaled(150.0);

/// How long a swing's damage window stays open, as a fraction of the swing time.
///
/// The window is NOT the whole swing: a two-thirds-of-a-second greatsword whose
/// hitbox is live the entire time would let a player stand still and grind a mob
/// down by holding the button, because every frame of the wind-up and the
/// recovery would count.
pub const SWING_WINDOW_FRAC: f32 = 0.45;

/// Ceiling on the damage window, in seconds.
///
/// Capping it means a slow weapon does not get a proportionally more forgiving
/// hit test than a fast one — the trade is damage for cadence, not damage for
/// cadence AND accuracy. Tuned against the `punch` sequence: 3 frames at 14fps,
/// contact on frame 1, i.e. the strike is on screen from 0.071s to 0.143s.
pub const SWING_WINDOW_MAX: f32 = 0.16;

/// Ceiling on the punch POSE, so a slow weapon does not freeze the figure
/// mid-strike.
pub const SWING_POSE_MAX: f32 = 0.3;

/// Bare-handed reach, in world px, measured OUTWARD from the body along facing.
///
/// The swing arc is the body box grown forward by this much. Bare hands get the
/// short one; anything held gets [`MELEE_REACH_WEAPON`], which is the mechanical
/// reason to hold a sword beyond the damage number. Both are lengths, so both
/// scale with the body — a character half the size has half the arm.
pub const MELEE_REACH_FIST: f32 = scaled(9.0);

/// Reach with a weapon held, in world px. See [`MELEE_REACH_FIST`].
pub const MELEE_REACH_WEAPON: f32 = scaled(20.0);

// ---------------------------------------------------------------------------
// Player projectiles
// ---------------------------------------------------------------------------

/// Downward acceleration on a player projectile, px/s^2.
///
/// A slight drop, so range is a real cost and a lobbed shot over a ledge is a
/// real skill. Small enough that point-blank aim is flat.
pub const SHOT_GRAVITY: f32 = scaled(420.0);

/// Seconds a player projectile lives.
///
/// The backstop that guarantees a shot cannot outlive its usefulness even flying
/// through open air; the streaming window would recycle it anyway (unloaded
/// cells read as solid), this just does it sooner.
pub const SHOT_LIFE: f32 = 2.2;

/// Fallback muzzle speed for a ranged weapon whose content omits the field, px/s.
pub const SHOT_SPEED_DEFAULT: f32 = scaled(620.0);

/// Impulse a projectile hit puts into what it hits, before any armour resistance.
pub const SHOT_KNOCKBACK: f32 = scaled(190.0);

//! Movement abilities and surface reactions.
//!
//! Split from `super::physics` on a clear line: physics is what a body always
//! does (gravity, friction, swimming, climbing), this is what it does because of
//! an ABILITY it has or a SURFACE it is standing on. Adding a dash or a
//! wall-jump belongs here; retuning how heavy the character feels does not.
//!
//! Surfaces are tagged in content (`block.surface = ice|bounce|conveyorL|
//! conveyorR|sticky`); the speeds those tags resolve to live here, so one number
//! retunes every block that carries the tag.

use super::physics::scaled;

/// Extra mid-air jumps. 1 = double jump.
pub const MAX_AIR_JUMPS: u32 = 1;

/// Horizontal dash velocity, px/s.
pub const DASH_SPEED: f32 = scaled(820.0);

/// Seconds a dash lasts.
pub const DASH_TIME: f32 = 0.14;

/// Seconds before dashing again.
pub const DASH_COOLDOWN: f32 = 0.55;

/// Max fall speed while wall-sliding, px/s.
pub const WALL_SLIDE_SPEED: f32 = scaled(140.0);

/// Horizontal kick off a wall, px/s.
pub const WALL_JUMP_PUSH: f32 = scaled(340.0);

/// Seconds of horizontal-input lockout after a wall jump.
pub const WALL_JUMP_LOCK: f32 = 0.14;

/// Launch speed off a `bounce` surface, px/s.
pub const BOUNCE_SPEED: f32 = scaled(1080.0);

/// Sideways carry on a `conveyor*` surface, px/s.
pub const CONVEYOR_SPEED: f32 = scaled(220.0);

/// Speed cap on a `sticky` surface, px/s.
pub const STICKY_MAX_SPEED: f32 = scaled(90.0);

/// Jump height multiplier out of a `sticky` surface.
pub const STICKY_JUMP_SCALE: f32 = 0.6;

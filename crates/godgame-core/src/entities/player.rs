//! The player.
//!
//! The movement feel — coyote time, jump buffer, variable jump, dash,
//! wall-slide/jump, ice/mud/liquid handling — is carried over unchanged from v2;
//! only the collision + environment sensing read the cell grid instead of tile
//! objects.
//!
//! # What the port changed, and what it deliberately did not
//!
//! **Nothing about the movement.** [`Player::step`] runs the same phases in the
//! same order, every constant is the same number, and every branch is the same
//! branch. `tests/player_golden.rs` replays 4 048 scripted steps against a
//! frozen dump of the TypeScript original to keep it that way.
//!
//! What did change is Rust-shaped:
//!
//! - `box` and `hitBox` were SHARED, MUTATED `AABB` instances, refreshed in
//!   place on every read so that reading them twice aliased. That existed to
//!   keep a 120 Hz hot path free of garbage. An [`Aabb`] is four floats and a
//!   `Copy` type, so [`Player::aabb`] and [`Player::hit_box`] return by value
//!   and the caution the TypeScript had to spend two paragraphs on is gone. The
//!   same goes for the module-level `SHOT` scratch: a [`ShotSpec`] is built on
//!   the stack per shot and allocates nothing.
//! - `AnimState` was a string union; it is an [`AnimState`] enum, and the two
//!   parallel `Record<AnimState, number>` tables are `match` arms on it, so a
//!   new state cannot be added without both answers.
//! - `PlayerEvent` was a string union; it is a [`PlayerEvent`] enum, and
//!   `drainEvents` — which allocated a fresh slice per drain — is
//!   [`Player::drain_events`], which drains into a buffer the caller owns.
//! - The player held a `Level` and read `level.grid` through a getter. Here the
//!   grid is a `&CellGrid` argument to the handful of methods that need one, and
//!   the only thing kept from the level is the spawn point `reset` reads. A
//!   `Player` that borrowed the level could not coexist with the automata
//!   mutating it, and the borrow would have been load-bearing for nothing: the
//!   player only ever read the grid.
//!
//! # Creative mode is a truce. This is a DELIBERATE DIVERGENCE, not a port.
//!
//! [`Player::untouchable`] has no counterpart in the TypeScript. Creative there
//! was an infinite build palette and nothing else: it changed what you could
//! PLACE and left the hazards and the creatures exactly as they were, so the
//! canonical way to survey a cavern was to fly a body that lava still ate.
//!
//! There is therefore no original to be faithful to here and nothing to check a
//! fixture against — which is precisely why it is written down. Everything else
//! in this file that differs from the original differs because Rust is not
//! JavaScript; this differs because the game now says something the original
//! never said. That is the same standard `godgame_render::particles` holds
//! itself to for its one knowing improvement, and it is held here for the
//! larger claim: a divergence in FEEL is easier to smuggle in than a divergence
//! in code, because nothing fails when you do.
//!
//! # Widths
//!
//! Everything here is `f32`, which is what [`crate::config`] and
//! [`crate::physics::collision`] are. The TypeScript necessarily ran the same
//! arithmetic in `f64` — JavaScript has no other number — so the parity suite
//! compares an `f32` port against `f64` references and finds a bounded,
//! documented drift rather than bit equality. See the head of
//! `tests/player_golden.rs` for the measured size of it. Widening this file
//! to `f64` to chase exactness would have made the player the one `f64` consumer
//! of an `f32` config and would have forked the width rule for a difference far
//! below a pixel.

use std::collections::VecDeque;

use crate::config::{
    AIR_ACCEL, AIR_FRICTION, BOUNCE_SPEED, CLIMB_REMOUNT_LOCK, CLIMB_SPEED_DOWN, CLIMB_SPEED_H,
    CLIMB_SPEED_UP, CLIMB_TOP_BOOST, CONVEYOR_SPEED, COYOTE_TIME, DASH_COOLDOWN, DASH_SPEED,
    DASH_TIME, DROP_THROUGH_TIME, GRAVITY, GROUND_FRICTION, ICE_FRICTION, JUMP_BUFFER, JUMP_CUT,
    JUMP_SPEED, LIQUID_DRAG, LIQUID_GRAVITY_SCALE, MAX_AIR_JUMPS, MAX_FALL_SPEED, MAX_HEALTH,
    MAX_RUN_SPEED, MELEE_REACH_FIST, MELEE_REACH_WEAPON, MOVE_ACCEL, PLAYER_H, PLAYER_W,
    PUNCH_DAMAGE, PUNCH_KNOCKBACK, PUNCH_SWING_TIME, SHOT_KNOCKBACK, SHOT_SPEED_DEFAULT,
    STEP_UP_MAX, STEP_UP_SMOOTH, STICKY_JUMP_SCALE, STICKY_MAX_SPEED, SWIM_ACCEL, SWIM_ACCEL_H,
    SWIM_BUOYANCY, SWIM_EXIT_SUBMERSION, SWIM_MAX_DOWN, SWIM_MAX_SPEED_H, SWIM_MAX_UP,
    SWIM_OUT_BOOST, SWIM_SINK_ACCEL, SWIM_SUBMERGE_MIN, SWING_POSE_MAX, SWING_WINDOW_FRAC,
    SWING_WINDOW_MAX, TILE_SIZE, WALL_JUMP_LOCK, WALL_JUMP_PUSH, WALL_SLIDE_SPEED, cell_at, scaled,
};
use crate::entities::projectiles::{SHOT_STYLE_ARROW, ShotSpec};
use crate::input::Intent;
use crate::physics::collision::{
    Aabb, NO_ONE_WAY, for_each_overlapped_cell, move_horizontal_stepped, one_way_under_feet,
    resolve_axis,
};
use crate::sim::coords::WorldCell;
use crate::sim::grid::CellGrid;
use crate::sim::materials::{
    BlockSurface, CellId, MAT_CLIMB, MAT_DAMAGE, MaterialState, mat_by_code,
};
use crate::sim::worldgen::SpawnPoint;

// ---------------------------------------------------------------------------
// The weapon, the ammo source and the projectile pool: three seams the player
// must not see through.
// ---------------------------------------------------------------------------

/// What the held item contributes to combat.
///
/// Structurally identical to the generated `ItemWeapon`, and deliberately NOT
/// that type. `Player` has no business depending on the item registry — it does
/// not know what an inventory is, what a slot is or what an item code means, and
/// the one thing it needs from that whole subsystem is four numbers.
///
/// TypeScript got the decoupling for free: the two types were structurally
/// identical, so `player.setWeapon(itemByCode(inv.held).weapon)` typechecked
/// with no conversion and no import. Rust is nominal, so the item layer builds
/// one of these instead — a three-line `From` on ITS side of the boundary. The
/// arrow still points one way, which was the whole point; only who writes the
/// adapter moved.
///
/// `ranged`, `projectile_speed` and `ammo` are the bow's fields; a melee weapon
/// leaves them at their defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerWeapon {
    /// Hit points one connecting swing or shot is worth.
    pub damage: f32,
    /// Impulse into the target, in the CONTENT's reference frame — the player
    /// body-scales it. `None` falls back to the bare-handed number.
    pub knockback: Option<f32>,
    /// Seconds between attacks. `None` or non-positive falls back to a fist's.
    pub swing_time: Option<f32>,
    /// Fires projectiles rather than swinging.
    pub ranged: bool,
    /// Muzzle speed, in the content's reference frame. `None` uses the default.
    pub projectile_speed: Option<f32>,
    /// Item ids this weapon can spend, best first. Empty = fires free.
    ///
    /// A LIST rather than one id because a `ref?` preference chain collapses to
    /// a single winner at compile time — it answers "which of these exists", not
    /// "which of these do you have". Ammo is the second question: a bow should
    /// fall back from fire arrows to plain ones at the moment of firing, based
    /// on what is actually in the pack.
    pub ammo: &'static [&'static str],
}

/// How the player spends ammo. Implemented by whoever owns the inventory;
/// [`spend`](AmmoSource::spend) returns how many were actually taken, which is
/// the exact signature `Inventory::remove` has once the id is resolved to a code.
///
/// A body handed no source fires freely — that is what makes this file testable
/// headlessly and what keeps a half-wired host playable rather than silently
/// unable to shoot.
///
/// # Why a trait and not the `Box<dyn FnMut + 'static>` this used to be
///
/// The TypeScript wrote `player.setAmmoSource((id, n) => inv.remove(...))`,
/// closing over the inventory instance. Transliterated, that becomes a boxed
/// closure stored on the `Player` — which must be `'static`, which forces the
/// captured inventory to be owned, which forces `Arc<Mutex<_>>` to share it with
/// the systems that also draw and edit it.
///
/// None of that is needed. The player spends ammo only while it is stepping, so
/// the source is borrowed for the length of the step and `Inventory` implements
/// this directly. See [`Loadout`].
pub trait AmmoSource {
    /// Spend `n` of the item with this authoring id. Returns how many were taken.
    fn spend(&mut self, id: &str, n: u32) -> u32;
}

// `ShotSpec` and `SHOT_STYLE_ARROW` used to be declared here, because `shoot`
// needed to name them and there was no pool yet to own them. Both now live in
// [`crate::entities::projectiles`], where the appearance tables the style
// indexes are, and are imported above.

/// The projectile pool, as the player needs it: one verb, `fire`.
///
/// A TRAIT rather than a concrete type because the body must not learn what a
/// shot IS, and deliberately ONE METHOD wide. It used to carry `update` and
/// `clear` as well, and both were mistakes of ownership rather than of interface:
/// `update` was called from the last line of [`Player::step`], which is what
/// forced the player to OWN a `Box<dyn Projectiles>` for life; `clear` was called
/// from [`Player::reset`] for the same reason. Both are now the host's, which
/// already holds the real pool and can name its concrete type.
///
/// Every method added past `fire` is one more thing the player could accidentally
/// reach for, and every one of them is a reason for the body to own the pool
/// instead of borrowing it.
pub trait Projectiles {
    /// Loose one projectile. Returns false if the pool is full.
    fn fire(&mut self, x: f32, y: f32, dir_x: f32, dir_y: f32, spec: ShotSpec) -> bool;
}

/// The pool a body gets when the host has not supplied one: shots are refused
/// and the rest of the player works.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProjectiles;

impl Projectiles for NoProjectiles {
    fn fire(&mut self, _x: f32, _y: f32, _dir_x: f32, _dir_y: f32, _spec: ShotSpec) -> bool {
        false
    }
}

/// What the body borrows from outside itself for the length of one step.
///
/// # Why this exists
///
/// [`Player`] used to OWN both halves: a `Box<dyn Projectiles>` and an
/// `Option<Box<dyn FnMut>>` ammo source, each installed once through a setter.
/// That is the TypeScript's shape — in JS a closure over the pool and the pack
/// costs nothing — and in Rust it is the root of three `Arc<Mutex<_>>` types, a
/// documented-but-unchecked lock ordering, and a `Player` that could not be
/// constructed before the things it fired into.
///
/// A body needs the pool and the pack only WHILE IT IS STEPPING. So they are
/// borrowed for exactly that long, which is a lifetime the caller already has,
/// and the `Player` goes back to being a plain value that owns only itself.
///
/// # Using one
///
/// ```ignore
/// let mut kit = Loadout::new(&mut arrows).with_ammo(&mut pack);
/// body.step(STEP_DT, intent, grid, &mut kit);
/// arrows.update(STEP_DT, grid, &mut hit_test);   // the very next line
/// ```
pub struct Loadout<'a> {
    /// Where a bow's arrows go.
    pub shots: &'a mut dyn Projectiles,
    /// What a bow spends. `None` fires freely — see [`AmmoSource`].
    pub ammo: Option<&'a mut dyn AmmoSource>,
}

impl<'a> Loadout<'a> {
    /// A pool to fire into and no ammo to spend.
    pub fn new(shots: &'a mut dyn Projectiles) -> Loadout<'a> {
        Loadout { shots, ammo: None }
    }

    /// Spend from this pack when the held weapon names ammo.
    pub fn with_ammo(mut self, ammo: &'a mut dyn AmmoSource) -> Loadout<'a> {
        self.ammo = Some(ammo);
        self
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// One-shot things that physically happened this step. Consumers call
/// [`Player::drain_events`] once per frame to spawn particles / shake; each
/// occurrence is emitted exactly once, and the buffer is capped so an undrained
/// player cannot grow it without bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlayerEvent {
    Land,
    Jump,
    DoubleJump,
    Dash,
    WallJump,
    Splash,
    Step,
    Hurt,
}

/// Oldest events are dropped past this if nobody drains.
const MAX_EVENTS: usize = 32;

// ---------------------------------------------------------------------------
// Animation tuning. Presentation-only knobs (frame selection, hold times,
// squash); the movement constants live in `crate::config::physics` and none of
// the values below feed back into the physics.
//
// The distance-bearing ones go through the shared `scaled` from
// `config::physics` for the same reason the physics tunables do: they are
// velocities and distances tuned against the reference character, and an
// unscaled landing threshold against scaled fall speeds silently changes which
// landings count as hard. Durations, ratios and cadences (Hz) are NOT scaled.
// ---------------------------------------------------------------------------

/// px/s of |vx| needed to start the run cycle …
const RUN_ENTER_SPEED: f32 = scaled(32.0);
/// … and the lower speed it takes to fall back to idle (hysteresis band).
const RUN_EXIT_SPEED: f32 = scaled(14.0);
/// px travelled per full 4-frame run cycle (2 footfalls).
const RUN_CYCLE_PX: f32 = scaled(144.0);
/// Cadence clamp, cycles/second: a trudge floor …
const RUN_CADENCE_MIN: f32 = 1.1;
/// … and a sprint ceiling.
const RUN_CADENCE_MAX: f32 = 3.6;
/// |vx| above which reversing input reads as a skid rather than a turn.
const SKID_SPEED: f32 = scaled(90.0);
/// |vy| deadband around the apex, so jump<->fall cannot flip on jitter.
const APEX_VY: f32 = scaled(45.0);
/// Landing recovery: only impacts faster than this crouch …
const LAND_IMPACT_MIN: f32 = scaled(260.0);
/// … and for how long.
const LAND_HOLD: f32 = 0.13;
/// How long the wall-slide pose survives losing wall contact.
const WALL_GRACE: f32 = 0.1;
/// Double-jump flip duration (4 frames at 12fps ~ 0.33s).
const DOUBLE_JUMP_HOLD: f32 = 0.34;
/// Punch pose duration for a bare fist (3 frames at 14fps = 0.214s, rounded up).
///
/// A held WEAPON overrides it with its own swing time, capped by
/// `SWING_POSE_MAX` — a slow greatsword should read as slow, but a pose held for
/// two thirds of a second is a figure frozen mid-strike rather than one swinging
/// heavily.
const PUNCH_TIME: f32 = 0.24;
/// Hurt pose duration …
const HURT_TIME: f32 = 0.3;
/// … and the minimum gap between repeat hurt events.
const HURT_REPEAT: f32 = 0.45;

/// The speed of descent that counts as a real landing rather than a stride over
/// a bump. Running over 1-cell rubble briefly clears `on_ground`, and that must
/// not fire a landing every stride.
const LAND_EVENT_MIN_VY: f32 = scaled(60.0);
/// Impact speed above which the landing squashes the sprite …
const SQUASH_MIN_VY: f32 = scaled(200.0);
/// … and the speed that squashes it fully.
const SQUASH_FULL_VY: f32 = scaled(900.0);

/// Low-pass on measured horizontal acceleration, for the drawn lean.
const ACCEL_SMOOTH: f32 = 0.3;

/// The pose a climb borrows.
///
/// There is no `Climb` sequence, and inventing one is a content change, not a
/// code one. `WallSlide` is not a stand-in here — it is the right frame: a
/// one-pixel-wide silhouette pressed flat against what it is holding, feet
/// tucked, looping slowly. That is a ladder as much as it is a wall, and the
/// alternative (`Swim`, the only other non-grounded loop) puts the body PRONE,
/// which reads as a character lying on a ladder.
const CLIMB_POSE: AnimState = AnimState::WallSlide;

/// The pose the sprite plays. Presentation only; never read back by the physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnimState {
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
}

impl AnimState {
    /// Preemption order (lower wins). A state can always be cut short by
    /// something more important; otherwise it must serve out its minimum hold.
    /// Together these are what stop boundary conditions from flickering frame to
    /// frame.
    const fn priority(self) -> u8 {
        match self {
            AnimState::Hurt => 0,
            AnimState::Dash => 1,
            AnimState::DoubleJump => 2,
            AnimState::Jump => 3,
            AnimState::Punch => 4,
            AnimState::Swim => 5,
            AnimState::WallSlide => 6,
            AnimState::Land => 7,
            AnimState::Fall => 8,
            AnimState::Skid => 9,
            AnimState::Run => 10,
            AnimState::Idle => 11,
        }
    }

    /// Seconds a state is held before a lower-priority state may replace it.
    const fn min_hold(self) -> f32 {
        match self {
            AnimState::Idle => 0.06,
            AnimState::Run => 0.09,
            AnimState::Skid => 0.12,
            AnimState::Jump => 0.07,
            AnimState::DoubleJump => 0.2,
            AnimState::Fall => 0.07,
            AnimState::Land => 0.1,
            AnimState::Dash => 0.05,
            AnimState::WallSlide => 0.1,
            AnimState::Swim => 0.08,
            AnimState::Punch => 0.1,
            AnimState::Hurt => 0.14,
        }
    }
}

/// `v` clamped, written out rather than using `f32::clamp`.
///
/// `f32::clamp` panics when `lo > hi` and propagates NaN differently; this is
/// the TypeScript's ternary, which is what every number here was tuned against.
#[inline]
fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `Math.sign`, which is NOT `f32::signum`.
///
/// `signum` answers 1.0 for `+0.0` and -1.0 for `-0.0`; JavaScript answers the
/// zero back. The friction branch is guarded so it cannot see zero, but the skid
/// test compares this against a direction that CAN be zero, and the two answers
/// disagree there.
#[inline]
fn sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// The player
// ---------------------------------------------------------------------------

/// The player.
pub struct Player {
    /// Top-left of the collision box, world px.
    pub x: f32,
    /// Top-left of the collision box, world px. Y grows downward.
    pub y: f32,
    /// Horizontal velocity, px/s.
    pub vx: f32,
    /// Vertical velocity, px/s.
    pub vy: f32,
    /// Which way the body is pointing: exactly `1.0` or `-1.0`.
    pub facing: f32,
    /// Hit points, clamped to `[0, MAX_HEALTH]` at the end of every step.
    pub health: f32,

    /// Nothing in the world may act on this body. Off unless a host turns it on.
    ///
    /// Named `untouchable` and NOT `invulnerable`, because it is stronger than
    /// invulnerability and the difference is the whole feature. Invulnerable is
    /// a well-worn word that means one thing — damage does not land — and a
    /// reader who saw it here would be entitled to expect a body that creatures
    /// still chase, still swarm and still shoot at, harmlessly. This flag also
    /// takes the body OUT OF PERCEPTION: [`MobTarget::targetable`] answers `!` of
    /// it, and a creature that cannot perceive you does not aggro, does not
    /// dive, does not fire. An unfamiliar word is the right cost for sending a
    /// reader to this paragraph.
    ///
    /// Two things stop when it is set, and it is worth being exact about which:
    ///
    /// - the material hazard drain in [`Player::overlap_effects`] — lava and
    ///   spikes — does not apply, and NEITHER DOES ITS REACTION. No hurt pose,
    ///   no [`PlayerEvent::Hurt`], no spent `hurt_cooldown`. A flash and a
    ///   flinch that cost nothing are a lie told to the player about their own
    ///   health bar, and the flinch is worse than the flash: the pose preempts
    ///   every other animation state, so a body standing in lava would stutter
    ///   permanently while taking no damage at all.
    /// - `MobTarget::take_damage` no-ops, which is the last line rather than the
    ///   first: `targetable` already stops every caller inside `MobSystem`, and
    ///   this catches the one that is written next year.
    ///
    /// What deliberately does NOT stop: everything the player does to the world.
    /// Digging, placing, swinging, shooting and killing all work exactly as they
    /// did, and creatures still spawn, wander, take damage and die. Creative is
    /// the world ignoring you, not the world stopping.
    ///
    /// [`Player::reset`] does not clear it — see the note there.
    ///
    /// [`MobTarget::targetable`]: crate::entities::mobs::MobTarget::targetable
    pub untouchable: bool,

    // Environment flags sensed pre-move each step from surrounding cells.
    /// Feet resting on something solid.
    pub on_ground: bool,
    /// The surface under the feet is tagged `ice`.
    pub on_ice: bool,
    /// True once past `SWIM_SUBMERGE_MIN` — swim rules replace ground rules.
    pub in_liquid: bool,
    /// 0..1 fraction of the body's cells that are liquid. Drives buoyancy/drag.
    pub submersion: f32,
    /// Any liquid contact at all, including ankle-deep. Drives the splash event.
    pub touching_liquid: bool,
    /// The surface under the feet is tagged `sticky`.
    pub sticky_this_step: bool,
    /// Sideways carry from a `conveyor*` surface, px/s.
    pub conveyor_vx: f32,
    /// Residual visual offset from a step-up, in px. The collision box snaps up
    /// a whole cell instantly; the drawn sprite carries this offset and eases it
    /// to zero so walking over rubble reads as a stride, not a teleport.
    step_up_visual: f32,

    // Jump / air state.
    /// Mid-air jumps left before the feet have to touch something.
    pub air_jumps: u32,
    coyote: f32,
    jump_buffer: f32,
    prev_jump_held: bool,

    // Dash state.
    dash_timer: f32,
    dash_cooldown: f32,
    dash_dir: f32,

    // Wall state.
    wall_dir: i32,
    wall_jump_lock: f32,

    // --- Climbing / one-way platforms ---
    /// True while attached to a ladder or rope: gravity is suspended.
    pub climbing: bool,
    /// Set pre-move each step: is the body overlapping anything climbable at all?
    on_climbable: bool,
    /// Seconds left of the lockout that stops a jump from instantly re-grabbing.
    climb_lock: f32,
    /// Seconds left of one-way pass-through granted by a deliberate down input.
    drop_through: f32,

    // --- Combat ---
    /// The held weapon, or `None` for bare hands. Pushed in by whoever owns the
    /// inventory (see [`Player::set_weapon`]); the player never reads an item
    /// registry.
    weapon: Option<PlayerWeapon>,
    /// Seconds until another swing/shot may start. The item's `swing_time`.
    swing_cd: f32,
    /// Seconds the current swing's damage window stays open. See
    /// [`Player::punching`].
    swing_live: f32,
    /// Rising-edge id, incremented once per swing.
    ///
    /// This is the whole reason a combat resolver does not have to track the
    /// player's previous punching state: it compares this id against the last
    /// one it damaged a given target with, so one swing lands on one creature
    /// exactly once no matter how many steps the window spans or how the frame
    /// rate moves.
    swing_seq: u32,

    // Landing squash for draw juice.
    squash: f32,
    /// Impact speed captured on the frame the player last landed (for juice).
    pub land_impact: f32,

    // --- Animation state (presentation only; never read back by the physics) ---
    anim_cur: AnimState,
    anim_t: f32,
    anim_clock_t: f32,
    run_phase: f32,
    /// Horizontal intent this step, for the skid test.
    input_dir: f32,
    prev_vx: f32,
    /// Smoothed dvx/dt, drives the directional lean.
    accel_x: f32,
    land_timer: f32,
    double_jump_timer: f32,
    punch_timer: f32,
    hurt_timer: f32,
    hurt_cooldown: f32,
    wall_grace: f32,
    prev_in_liquid: bool,
    events: VecDeque<PlayerEvent>,

    /// Where [`Player::reset`] puts the body.
    ///
    /// The TypeScript held the whole `Level` and read `level.spawn` through it.
    /// Two floats is all `reset` ever wanted, and keeping them means a `Player`
    /// does not borrow the world the automata is mutating.
    spawn: SpawnPoint,
}

impl Player {
    /// A fresh player at the level's spawn.
    ///
    /// There is no second constructor taking a pool: a body does not own one.
    /// What it fires into is a [`Loadout`] handed to [`Player::step`].
    pub fn new(spawn: SpawnPoint) -> Player {
        let mut p = Player {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            facing: 1.0,
            health: MAX_HEALTH,
            untouchable: false,
            on_ground: false,
            on_ice: false,
            in_liquid: false,
            submersion: 0.0,
            touching_liquid: false,
            sticky_this_step: false,
            conveyor_vx: 0.0,
            step_up_visual: 0.0,
            air_jumps: MAX_AIR_JUMPS,
            coyote: 0.0,
            jump_buffer: 0.0,
            prev_jump_held: false,
            dash_timer: 0.0,
            dash_cooldown: 0.0,
            dash_dir: 1.0,
            wall_dir: 0,
            wall_jump_lock: 0.0,
            climbing: false,
            on_climbable: false,
            climb_lock: 0.0,
            drop_through: 0.0,
            weapon: None,
            swing_cd: 0.0,
            swing_live: 0.0,
            swing_seq: 0,
            squash: 0.0,
            land_impact: 0.0,
            anim_cur: AnimState::Idle,
            anim_t: 0.0,
            anim_clock_t: 0.0,
            run_phase: 0.0,
            input_dir: 0.0,
            prev_vx: 0.0,
            accel_x: 0.0,
            land_timer: 0.0,
            double_jump_timer: 0.0,
            punch_timer: 0.0,
            hurt_timer: 0.0,
            hurt_cooldown: 0.0,
            wall_grace: 0.0,
            prev_in_liquid: false,
            events: VecDeque::new(),
            spawn,
        };
        p.reset();
        p
    }

    /// The player's collision box.
    ///
    /// The TypeScript returned a SHARED, MUTATED instance, refreshed in place on
    /// every read so that two reads aliased and neither could be retained. That
    /// bought a 120 Hz hot path — read several times per step by the player and
    /// by every mob testing against it — freedom from garbage. An [`Aabb`] is
    /// four floats and `Copy`, so this returns one and the caution is gone.
    #[inline]
    pub fn aabb(&self) -> Aabb {
        Aabb::new(self.x, self.y, PLAYER_W, PLAYER_H)
    }

    /// Out of health.
    #[inline]
    pub fn dead(&self) -> bool {
        self.health <= 0.0
    }

    /// The dash is off cooldown.
    #[inline]
    pub fn dash_ready(&self) -> bool {
        self.dash_cooldown <= 0.0
    }

    /// A dash is in progress.
    #[inline]
    pub fn dashing(&self) -> bool {
        self.dash_timer > 0.0
    }

    // -----------------------------------------------------------------------
    // COMBAT CONTRACT
    //
    // Damage resolution lives with whoever owns the targets (`MobSystem`), and
    // that stays true: the player does not know what a creature is. What CHANGED
    // is that it used to also own the numbers — a flat PUNCH_DAMAGE constant —
    // so the King's Sword hit exactly as hard as an empty hand and the entire
    // item ladder was cosmetic. Everything a swing is worth is published here
    // instead:
    //
    //   punching         the damage WINDOW is open (was: the pose is playing)
    //   swing_id         one id per swing; compare it to land one hit per target
    //   hit_box          the actual arc, already grown by the weapon's reach
    //   melee_damage     the weapon's damage, or a bare fist's
    //   melee_knockback  the weapon's impulse, or a bare fist's
    //
    // A resolver reads those five and applies its own armour, resistance and
    // immunity rules to them. Nothing here knows those rules exist.
    // -----------------------------------------------------------------------

    /// True while the current swing can connect.
    ///
    /// NOTE the change in meaning: this used to be "the punch pose is playing",
    /// which for a slow weapon would leave the hitbox live through the entire
    /// wind-up and recovery and let a player grind a creature down by standing
    /// still with the button held. It is now the strike window only
    /// (`SWING_WINDOW_FRAC` of the swing, capped at `SWING_WINDOW_MAX`), which
    /// is the interval the `punch` sequence actually has a fist out in.
    #[inline]
    pub fn punching(&self) -> bool {
        self.swing_live > 0.0
    }

    /// Rising-edge id of the current swing. See the note above.
    #[inline]
    pub fn swing_id(&self) -> u32 {
        self.swing_seq
    }

    /// Hit points one connecting swing is worth, before the target's armour.
    pub fn melee_damage(&self) -> f32 {
        match self.weapon {
            Some(w) if !w.ranged => w.damage,
            _ => PUNCH_DAMAGE,
        }
    }

    /// Impulse, px/s, a connecting swing puts into what it hits.
    pub fn melee_knockback(&self) -> f32 {
        match self.weapon {
            // Content authors knockback in the same reference frame as every
            // other velocity in the game (see `config::physics`), so it is
            // body-scaled here rather than being a raw px/s number tuned against
            // one particular character size.
            Some(w) if !w.ranged => w.knockback.map_or(PUNCH_KNOCKBACK, scaled),
            _ => PUNCH_KNOCKBACK,
        }
    }

    /// How far the swing reaches past the body, in world px.
    pub fn melee_reach(&self) -> f32 {
        match self.weapon {
            Some(w) if !w.ranged => MELEE_REACH_WEAPON,
            _ => MELEE_REACH_FIST,
        }
    }

    /// Seconds between swings — the item's `swing_time`, or a fist's cadence.
    pub fn swing_time(&self) -> f32 {
        match self.weapon.and_then(|w| w.swing_time) {
            Some(t) if t > 0.0 => t,
            _ => PUNCH_SWING_TIME,
        }
    }

    /// The swing arc: the body box grown forward along the facing by
    /// [`Player::melee_reach`].
    ///
    /// Shared and mutated in the TypeScript, exactly like `box` and for the same
    /// reason; returned by value here, for the same reason as [`Player::aabb`].
    pub fn hit_box(&self) -> Aabb {
        let reach = self.melee_reach();
        Aabb::new(
            if self.facing > 0.0 {
                self.x
            } else {
                self.x - reach
            },
            self.y,
            PLAYER_W + reach,
            PLAYER_H,
        )
    }

    /// Equip (or clear) the held weapon. Called by whoever owns the inventory
    /// when the selection changes.
    ///
    /// Swapping mid-swing does NOT cancel the swing already in the air: the
    /// damage was decided when it started, and re-reading it here would let a
    /// player start a fast swing and finish it with the greatsword's number.
    pub fn set_weapon(&mut self, w: Option<PlayerWeapon>) {
        self.weapon = w;
    }

    /// True when the held weapon fires projectiles rather than swinging.
    #[inline]
    pub fn ranged(&self) -> bool {
        self.weapon.is_some_and(|w| w.ranged)
    }

    /// Current animation state — the same value the sprite renders this frame.
    #[inline]
    pub fn anim(&self) -> AnimState {
        self.anim_cur
    }

    // --- What `draw` used to read -----------------------------------------
    // `Player.draw` was Canvas2D and stayed behind in the renderer. Everything
    // it reached into the player for is published here, so the drawing can be
    // rebuilt in `godgame-render` without this crate growing a pixel of
    // knowledge about how any of it looks.

    /// Seconds spent in the current animation state.
    #[inline]
    pub fn anim_t(&self) -> f32 {
        self.anim_t
    }

    /// Free-running seconds, for ambient loops.
    #[inline]
    pub fn anim_clock_t(&self) -> f32 {
        self.anim_clock_t
    }

    /// Run cycle position in `[0, 1)`. The two contact frames sit at 0.0 and 0.5.
    #[inline]
    pub fn run_phase(&self) -> f32 {
        self.run_phase
    }

    /// Landing squash, decaying to zero. Drives the fatten-and-flatten.
    #[inline]
    pub fn squash(&self) -> f32 {
        self.squash
    }

    /// Smoothed horizontal acceleration. Drives the directional lean.
    #[inline]
    pub fn accel_x(&self) -> f32 {
        self.accel_x
    }

    /// Which way the current dash is going, for the smear's after-images.
    #[inline]
    pub fn dash_dir(&self) -> f32 {
        self.dash_dir
    }

    /// Visual lag behind a step-up, in px, easing to zero. The sprite is drawn
    /// this far below the box so cresting rubble reads as a stride.
    #[inline]
    pub fn step_up_visual(&self) -> f32 {
        self.step_up_visual
    }

    /// Drain the one-shot events that occurred since the last call into `out`,
    /// clearing the buffer.
    ///
    /// The TypeScript returned a fresh slice per drain and a shared empty array
    /// for the common case, to keep the no-events path allocation-free. Draining
    /// into the caller's buffer makes every path allocation-free instead — the
    /// host keeps one `Vec` across frames and this only ever writes into it.
    ///
    /// Draining is what the caller is OBLIGED to do: see [`Player::emit`] for
    /// what happens to a buffer nobody empties.
    pub fn drain_events(&mut self, out: &mut Vec<PlayerEvent>) {
        out.extend(self.events.drain(..));
    }

    /// Attack with whatever is held: a swing, or a shot if the weapon is ranged.
    ///
    /// Returns false when the cadence has not come round yet, which is the ONLY
    /// gate — an attack is never refused for being mid-pose, because the pose
    /// length and the swing time are different numbers and letting the animation
    /// decide when the next hit may land is how a weapon ends up with a DPS
    /// nobody chose.
    pub fn attack(&mut self, aim_x: f32, aim_y: f32, kit: &mut Loadout<'_>) -> bool {
        if self.swing_cd > 0.0 {
            return false;
        }
        if self.ranged() {
            self.shoot(aim_x, aim_y, kit)
        } else {
            self.start_swing()
        }
    }

    /// Open a damage window and play the strike.
    ///
    /// The window is a FRACTION of the swing rather than a fixed time so that a
    /// slower weapon is genuinely slower to land as well as slower to repeat,
    /// and it is capped so a two-thirds-of-a-second greatsword does not also get
    /// a four-times-more-forgiving hit test than a dagger. See
    /// `config::combat`.
    fn start_swing(&mut self) -> bool {
        let swing = self.swing_time();
        self.swing_cd = swing;
        self.swing_live = (swing * SWING_WINDOW_FRAC).min(SWING_WINDOW_MAX);
        self.swing_seq += 1;
        self.punch_timer = if self.weapon.is_none() {
            PUNCH_TIME
        } else {
            swing.min(SWING_POSE_MAX)
        };
        true
    }

    /// Loose one projectile, spending ammo first.
    ///
    /// Ammo is spent BEFORE the shot is fired and the shot only happens if the
    /// spend succeeded, which is the ordering that cannot produce a free arrow
    /// under any interleaving. The cooldown is spent either way: a dry bow that
    /// let you retry every single frame would hammer the inventory scan at 120Hz
    /// for as long as the button was held.
    ///
    /// `aim_x`/`aim_y` are world coordinates when the host has a cursor; `(0, 0)`
    /// means "no aim" — the same sentinel [`Intent::has_aim`] tests — in which
    /// case the arrow flies flat along the facing.
    fn shoot(&mut self, aim_x: f32, aim_y: f32, kit: &mut Loadout<'_>) -> bool {
        let Some(w) = self.weapon else {
            return false;
        };
        self.swing_cd = self.swing_time();
        self.punch_timer = self.swing_cd.min(SWING_POSE_MAX);

        // Spend the first ammo the pack actually holds. `remove` returns how many
        // it took, so "do I have any" and "take one" are the same call — asking
        // first would be a second lookup and a window for the two answers to
        // disagree.
        if !w.ammo.is_empty()
            && let Some(source) = kit.ammo.as_deref_mut()
        {
            let mut spent = false;
            for id in w.ammo {
                if source.spend(id, 1) >= 1 {
                    spent = true;
                    break;
                }
            }
            // Out of every kind: the swing cooldown above still stands, so a dry
            // bow clicks at its own rate instead of letting you spam the empty
            // draw.
            if !spent {
                return false;
            }
        }

        // Launch from the chest rather than the feet or the box centre: an arrow
        // that leaves at ground level clips the first pebble in front of the
        // player, and one that leaves from the head reads as being fired over the
        // target.
        let ox = self.x + PLAYER_W * 0.5 + self.facing * PLAYER_W * 0.5;
        let oy = self.y + PLAYER_H * 0.35;
        let mut dx = self.facing;
        let mut dy = 0.0;
        if aim_x != 0.0 || aim_y != 0.0 {
            dx = aim_x - ox;
            dy = aim_y - oy;
            if dx == 0.0 && dy == 0.0 {
                dx = self.facing;
            }
        }

        let speed = scaled(w.projectile_speed.unwrap_or(0.0));
        let spec = ShotSpec {
            speed: if speed == 0.0 {
                SHOT_SPEED_DEFAULT
            } else {
                speed
            },
            damage: w.damage,
            knockback: w.knockback.map_or(SHOT_KNOCKBACK, scaled),
            r_px: 1.0,
            style: SHOT_STYLE_ARROW,
        };
        kit.shots.fire(ox, oy, dx, dy, spec)
    }

    /// Append an event, dropping the oldest if nobody is draining.
    ///
    /// EVICTS THE OLDEST rather than saturating, which is the opposite of what
    /// `MobSystem::push_rgb` does and is deliberate: a body emits a handful of
    /// events a second and the newest ones are the interesting ones, whereas 32
    /// creatures in a firefight emit in bursts and dropping the START of a burst
    /// would lose the kill that caused it.
    ///
    /// A [`VecDeque`] and not a `Vec`, because evicting the oldest is exactly
    /// what a deque's front is for. It was `Vec::remove(0)` — an O(n) shift of
    /// the whole buffer to drop one element, which is the shape you write when
    /// you are thinking in `Array.prototype.shift()`. At `MAX_EVENTS` = 32 the
    /// difference is not measurable; the reason to change it is that the deque
    /// SAYS the eviction policy, and a reader no longer has to infer a ring
    /// buffer from an index.
    fn emit(&mut self, e: PlayerEvent) {
        if self.events.len() >= MAX_EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(e);
    }

    /// Put the body back at the spawn point and clear the per-life state.
    ///
    /// [`Player::untouchable`] is NOT per-life state and is not cleared, on the
    /// same terms as the weapon: it is a MODE the host holds, mirrored onto the
    /// body from the build tool, and a respawn that silently dropped it would
    /// hand the mode back to a player who never asked for it — or, worse, be
    /// corrected a frame later by the host and read as a flicker.
    pub fn reset(&mut self) {
        let s = self.spawn;
        self.x = s.x + (TILE_SIZE as f32 - PLAYER_W) / 2.0;
        self.y = s.y + (TILE_SIZE as f32 - PLAYER_H);
        self.vx = 0.0;
        self.vy = 0.0;
        self.health = MAX_HEALTH;
        self.on_ground = false;
        self.coyote = 0.0;
        self.jump_buffer = 0.0;
        self.air_jumps = MAX_AIR_JUMPS;
        self.dash_timer = 0.0;
        self.dash_cooldown = 0.0;
        self.wall_dir = 0;
        self.wall_jump_lock = 0.0;
        self.climbing = false;
        self.on_climbable = false;
        self.climb_lock = 0.0;
        self.drop_through = 0.0;
        // The swing state resets but the weapon does NOT: equipment survives
        // death, and re-equipping is the inventory owner's decision, not a
        // respawn's.
        self.swing_cd = 0.0;
        self.swing_live = 0.0;
        // Arrows in flight belong to the life that fired them — but the body
        // does not own the pool, so DROPPING THEM IS THE CALLER'S, on the same
        // line as this reset. `godgame_render::glue::start_a_run` is the one
        // place in the tree that respawns, and it does both.

        // Animation/report state is respawn-local too: a fresh player should not
        // inherit a pose or a queue of stale events from the previous life.
        self.anim_cur = AnimState::Idle;
        self.anim_t = 0.0;
        self.run_phase = 0.0;
        self.input_dir = 0.0;
        self.prev_vx = 0.0;
        self.accel_x = 0.0;
        self.land_timer = 0.0;
        self.double_jump_timer = 0.0;
        self.punch_timer = 0.0;
        self.hurt_timer = 0.0;
        self.hurt_cooldown = 0.0;
        self.wall_grace = 0.0;
        self.prev_in_liquid = false;
        self.events.clear();
    }

    /// One fixed physics step.
    ///
    /// THE ORDER IS LOAD-BEARING and is the order the TypeScript ran in. Sensing
    /// happens after the timers and before any movement, so ice, mud, conveyors
    /// and buoyancy bite on the same step they are entered; `wall_dir` is cleared
    /// immediately before the collide that sets it, so it always describes THIS
    /// step; and `prev_jump_held` is latched after the vertical model has already
    /// used the previous value to decide whether to cut the jump.
    pub fn step(&mut self, dt: f32, intent: Intent, grid: &CellGrid, kit: &mut Loadout<'_>) {
        self.tick_timers(dt);
        // Held attacks auto-repeat at the weapon's own cadence; `attack` is the
        // one gate, so a queued press and a held button take exactly the same
        // path and a fast weapon does not need the player to out-click it.
        if intent.punch_queued || intent.punch_held {
            self.attack(intent.aim_x, intent.aim_y, kit);
        }
        self.start_dash(intent);
        self.sense_environment(grid);
        self.update_climb(intent, grid);
        self.update_drop_through(intent, grid);
        self.apply_horizontal(dt, intent);
        self.apply_vertical(dt, intent);
        self.wall_dir = 0;
        self.integrate_and_collide(dt, grid);
        self.overlap_effects(dt, grid);
        self.post_environment(dt);
        self.prev_jump_held = intent.jump_held;
        self.update_anim(dt);
    }

    /// Attach to, hold or let go of a ladder.
    ///
    /// Mounting is deliberately an INTENT, not a collision: standing in front of
    /// a ladder must not glue you to it, or every ladder becomes a wall you have
    /// to jump past. Pressing up or down is the grab, which is the rule every
    /// game with a ladder has converged on, and it is also what makes the release
    /// rules simple — there are only three, and each one is something the player
    /// did:
    ///
    ///   jumped            kick off, and a lockout stops the next step re-grabbing
    ///                     the rungs the body is still inside
    ///   ran out of rungs  the body no longer overlaps anything climbable; carry a
    ///                     little upward speed so cresting the top clears the lip
    ///                     instead of dropping you straight back on
    ///   entered liquid    swimming has its own vertical model and the two would
    ///                     fight over vy every step
    fn update_climb(&mut self, intent: Intent, grid: &CellGrid) {
        let up = intent.up;
        let down = intent.down;

        if self.climbing {
            // Jump takes precedence: it is the one release that should feel instant.
            if intent.jump_queued {
                self.climbing = false;
                self.climb_lock = CLIMB_REMOUNT_LOCK;
                return; // apply_vertical turns the buffered jump into a real one
            }
            if !self.on_climbable || self.in_liquid {
                self.climbing = false;
                // Only a climb that was going UP gets the lip boost. Letting go on
                // the way down should drop you, not launch you.
                if self.vy < 0.0 {
                    self.vy = self.vy.min(-CLIMB_TOP_BOOST);
                }
            }
            return;
        }

        if self.climb_lock > 0.0 || self.in_liquid || self.dash_timer > 0.0 {
            return;
        }
        // A fresh jump PRESS is always a jump, never a grab. `up` and the jump key
        // are the same physical binding (see `Intent::from_keys`), so without this
        // the first frame of every jump taken next to a ladder would mount it and
        // swallow the press. Holding the key AFTER that edge is what climbs —
        // which is also what makes grabbing a rope on the way past work.
        if intent.jump_queued {
            return;
        }
        if !self.on_climbable || !(up || down) {
            return;
        }
        // Pressing DOWN while standing on solid ground next to a ladder must not
        // grab it — that input is the drop-through, and the two would fight.
        //
        // The `standing_on_climbable` test distinguishes "standing at the foot of
        // a ladder" (pressing down should drop you through a platform, if there
        // is one) from "standing on a rung with more ladder below" (pressing down
        // should climb down). Without it, descending a ladder that passes through
        // a floor is impossible.
        if down && !up && self.on_ground && !self.standing_on_climbable(grid) {
            return;
        }

        self.climbing = true;
        self.vx = 0.0;
        self.vy = 0.0;
        self.air_jumps = MAX_AIR_JUMPS; // a ladder is a place to recover, like ground
    }

    /// Down + standing on a wooden platform = fall through it.
    ///
    /// The grant is a TIMER rather than a per-step test of the input, because the
    /// body has to be unsolid to the platform for long enough that gravity
    /// carries it fully below the top face; releasing the key half a step in
    /// would otherwise snap it back up and the drop would silently fail. It also
    /// means the input is a decision, not a mode: holding down through a stack of
    /// platforms falls through each in turn, but letting go stops on the next one.
    ///
    /// `on_ground` is cleared here rather than being left to the next step's
    /// collide so that the animation and the jump logic agree with the physics on
    /// the frame the drop starts — otherwise the first step of a drop is a
    /// grounded player with no floor, which reads as one frame of idle in mid-air.
    fn update_drop_through(&mut self, intent: Intent, grid: &CellGrid) {
        if self.climbing || !self.on_ground {
            return;
        }
        if !intent.down || intent.up || intent.jump_queued {
            return;
        }
        if !one_way_under_feet(grid, self.aabb()) {
            return;
        }
        self.drop_through = DROP_THROUGH_TIME;
        self.on_ground = false;
        self.coyote = 0.0; // dropping through is not a ledge you may still jump from
    }

    /// Is the cell directly under the feet itself climbable?
    ///
    /// Distinguishes "standing at the foot of a ladder" (pressing down should
    /// drop you through a platform, if there is one) from "standing on a rung
    /// with more ladder below" (pressing down should climb down). Without it,
    /// descending a ladder that passes through a floor is impossible.
    fn standing_on_climbable(&self, grid: &CellGrid) -> bool {
        let cx = cell_at(self.x + PLAYER_W / 2.0);
        let cy = cell_at(self.y + PLAYER_H + 1.0);
        let w = WorldCell::new(cx, cy);
        grid.is_loaded_world(w) && MAT_CLIMB[grid.get_world(w) as usize] == 1
    }

    /// Read surface + liquid state from surrounding cells BEFORE moving, so ice,
    /// mud, conveyors and buoyancy bite the same step.
    fn sense_environment(&mut self, grid: &CellGrid) {
        self.on_ice = false;
        self.in_liquid = false;
        self.sticky_this_step = false;
        let mut conv_r = false;
        let mut conv_l = false;
        let mut on_ice = false;
        let mut sticky = false;

        let body = self.aabb();
        // Surface directly under the feet.
        let foot = Aabb::new(body.x + 2.0, body.y + body.h - 1.0, body.w - 4.0, 3.0);
        for_each_overlapped_cell(grid, foot, |_cx, _cy, id| {
            match mat_by_code(id).surface {
                Some(BlockSurface::Ice) => on_ice = true,
                Some(BlockSurface::Sticky) => sticky = true,
                Some(BlockSurface::ConveyorR) => conv_r = true,
                Some(BlockSurface::ConveyorL) => conv_l = true,
                // `bounce` is read where it is acted on, in `integrate_and_collide`.
                _ => {}
            }
        });
        self.on_ice = on_ice;
        self.sticky_this_step = sticky;
        self.conveyor_vx =
            if conv_r { CONVEYOR_SPEED } else { 0.0 } - if conv_l { CONVEYOR_SPEED } else { 0.0 };

        // How much of the body is in liquid, not merely whether any of it is. A
        // boolean here meant one water cell touching a foot flipped the player into
        // full swim mode, so ankle-deep puddles disabled walking and jumping.
        let mut liquid_cells = 0u32;
        let mut total_cells = 0u32;
        let mut climbable = false;
        for_each_overlapped_cell(grid, body, |_cx, _cy, id: CellId| {
            total_cells += 1;
            if MaterialState::of(id) == MaterialState::Liquid {
                liquid_cells += 1;
            }
            // Folded into the liquid scan rather than being its own pass: it is the
            // same six cells, and the whole-box test is deliberate — a ladder is one
            // cell wide against a two-cell body, so a centre test would make grabbing
            // it depend on sub-cell alignment the player cannot see.
            if MAT_CLIMB[id as usize] == 1 {
                climbable = true;
            }
        });
        self.on_climbable = climbable;
        self.submersion = if total_cells > 0 {
            liquid_cells as f32 / total_cells as f32
        } else {
            0.0
        };
        self.touching_liquid = liquid_cells > 0;
        // Swim rules only take over past the threshold; below it you are wading —
        // normal ground movement, just damped.
        self.in_liquid = self.submersion >= SWIM_SUBMERGE_MIN;

        // Entering liquid is the splash moment (leaving it is not an event). Keyed
        // to first contact, not to full submersion, so the splash lands as the feet
        // break the surface rather than a beat later.
        if self.touching_liquid && !self.prev_in_liquid {
            self.emit(PlayerEvent::Splash);
        }
        self.prev_in_liquid = self.touching_liquid;
    }

    fn tick_timers(&mut self, dt: f32) {
        self.dash_cooldown = (self.dash_cooldown - dt).max(0.0);
        self.wall_jump_lock = (self.wall_jump_lock - dt).max(0.0);
        self.squash = (self.squash - dt * 6.0).max(0.0);
        // Ease the step-up offset away at a fixed rate so the visual catches up to
        // the box within a few frames regardless of how far it stepped.
        if self.step_up_visual > 0.0 {
            self.step_up_visual = (self.step_up_visual - STEP_UP_SMOOTH * dt).max(0.0);
        }
        self.land_timer = (self.land_timer - dt).max(0.0);
        self.double_jump_timer = (self.double_jump_timer - dt).max(0.0);
        self.punch_timer = (self.punch_timer - dt).max(0.0);
        self.swing_cd = (self.swing_cd - dt).max(0.0);
        self.swing_live = (self.swing_live - dt).max(0.0);
        self.climb_lock = (self.climb_lock - dt).max(0.0);
        self.drop_through = (self.drop_through - dt).max(0.0);
        self.hurt_timer = (self.hurt_timer - dt).max(0.0);
        self.hurt_cooldown = (self.hurt_cooldown - dt).max(0.0);
        self.wall_grace = (self.wall_grace - dt).max(0.0);
        if self.dash_timer > 0.0 {
            self.dash_timer -= dt;
            self.vx = self.dash_dir * DASH_SPEED;
            self.vy = 0.0;
        }
    }

    fn start_dash(&mut self, intent: Intent) {
        if intent.dash_queued && self.dash_cooldown <= 0.0 && self.dash_timer <= 0.0 {
            self.dash_timer = DASH_TIME;
            self.dash_dir = self.facing;
            self.dash_cooldown = DASH_COOLDOWN;
            self.vx = self.dash_dir * DASH_SPEED;
            self.vy = 0.0;
            self.emit(PlayerEvent::Dash);
        }
    }

    fn apply_horizontal(&mut self, dt: f32, intent: Intent) {
        if self.dash_timer > 0.0 {
            return;
        }
        let locked = self.wall_jump_lock > 0.0;
        let dir = if locked { 0.0 } else { intent.dir_x };
        self.input_dir = dir; // remembered for the skid/turnaround test

        // On a ladder, horizontal movement is a VELOCITY rather than an acceleration
        // with friction: you shuffle sideways onto the next rope or step off onto a
        // ledge, and you stop dead when you stop asking. Accelerating on a ladder
        // would let a player build up run speed while hanging in mid-air and carry it
        // off the end, which is a rope swing nobody designed.
        if self.climbing {
            self.vx = dir * CLIMB_SPEED_H;
            if dir != 0.0 {
                self.facing = if dir > 0.0 { 1.0 } else { -1.0 };
            }
            return;
        }
        let accel = if self.in_liquid {
            SWIM_ACCEL_H
        } else if self.on_ground {
            MOVE_ACCEL
        } else {
            AIR_ACCEL
        };

        if dir != 0.0 {
            self.vx += dir * accel * dt;
            self.facing = if dir > 0.0 { 1.0 } else { -1.0 };
        } else {
            let fric = if self.on_ice {
                ICE_FRICTION
            } else if self.on_ground {
                GROUND_FRICTION
            } else {
                AIR_FRICTION
            };
            let drop = fric * dt;
            if self.vx.abs() <= drop {
                self.vx = 0.0;
            } else {
                self.vx -= sign(self.vx) * drop;
            }
        }
        let cap_h = if self.in_liquid {
            SWIM_MAX_SPEED_H
        } else {
            MAX_RUN_SPEED
        };
        self.vx = (-cap_h).max(cap_h.min(self.vx));
        if self.sticky_this_step {
            self.vx = (-STICKY_MAX_SPEED).max(STICKY_MAX_SPEED.min(self.vx));
        }
    }

    fn apply_vertical(&mut self, dt: f32, intent: Intent) {
        if self.dash_timer > 0.0 {
            return;
        }

        // Climbing replaces the vertical model outright — no gravity, no terminal
        // velocity, no coyote time. Holding nothing HANGS: a ladder is the one place
        // in the game where the player's altitude is exactly what they asked for, and
        // that is the whole reason to put one in a cave.
        if self.climbing {
            let up = intent.up;
            let down = intent.down;
            self.vy = if up == down {
                0.0
            } else if up {
                -CLIMB_SPEED_UP
            } else {
                CLIMB_SPEED_DOWN
            };
            // The jump buffer still runs, so a press made just before grabbing (or
            // just before letting go) is not swallowed by the climb.
            if intent.jump_queued {
                self.jump_buffer = JUMP_BUFFER;
            } else {
                self.jump_buffer = (self.jump_buffer - dt).max(0.0);
            }
            self.coyote = COYOTE_TIME; // stepping off a ladder gets the same grace
            return;
        }

        if self.in_liquid {
            // Swimming: reduced gravity, opposed by buoyancy that scales with how
            // submerged the body is. Because buoyancy falls off as you break the
            // surface, the two balance near the waterline and the player floats and
            // bobs there instead of either sinking or being launched out.
            self.vy += GRAVITY * LIQUID_GRAVITY_SCALE * dt;
            self.vy -= SWIM_BUOYANCY * self.submersion * dt;

            // A sustained stroke you hold, not the old one-shot impulse. Holding up
            // climbs; holding down dives; neutral lets buoyancy settle you.
            if intent.jump_held {
                self.vy -= SWIM_ACCEL * dt;
            } else if intent.down {
                self.vy += SWIM_SINK_ACCEL * dt;
            }

            self.vy = (-SWIM_MAX_UP).max(SWIM_MAX_DOWN.min(self.vy));
        } else {
            self.vy += GRAVITY * dt;
            if self.vy > MAX_FALL_SPEED {
                self.vy = MAX_FALL_SPEED;
            }
        }

        self.coyote = if self.on_ground {
            COYOTE_TIME
        } else {
            (self.coyote - dt).max(0.0)
        };
        if self.on_ground {
            self.air_jumps = MAX_AIR_JUMPS;
        }
        if intent.jump_queued {
            self.jump_buffer = JUMP_BUFFER;
        } else {
            self.jump_buffer = (self.jump_buffer - dt).max(0.0);
        }

        if self.jump_buffer > 0.0 {
            if self.in_liquid {
                // Near the surface a stroke gets a kick, so you can climb out instead
                // of bobbing against the lip forever. Deep down the held stroke already
                // does the lifting and a tap should not fling you.
                if self.submersion < SWIM_EXIT_SUBMERSION {
                    self.vy = self.vy.min(-SWIM_OUT_BOOST);
                }
                self.jump_buffer = 0.0;
            } else if self.coyote > 0.0 {
                let scale = if self.sticky_this_step {
                    STICKY_JUMP_SCALE
                } else {
                    1.0
                };
                self.vy = -JUMP_SPEED * scale;
                self.jump_buffer = 0.0;
                self.coyote = 0.0;
                self.land_timer = 0.0; // a jump cancels any landing recovery
                self.emit(PlayerEvent::Jump);
            } else if self.wall_dir != 0 {
                self.wall_jump();
                self.jump_buffer = 0.0;
            } else if self.air_jumps > 0 {
                self.vy = -JUMP_SPEED;
                self.air_jumps -= 1;
                self.jump_buffer = 0.0;
                self.double_jump_timer = DOUBLE_JUMP_HOLD;
                self.emit(PlayerEvent::DoubleJump);
            }
        }

        if self.prev_jump_held && !intent.jump_held && self.vy < 0.0 {
            self.vy *= JUMP_CUT;
        }
    }

    fn wall_jump(&mut self) {
        self.vy = -JUMP_SPEED;
        self.vx = -(self.wall_dir as f32) * WALL_JUMP_PUSH;
        self.facing = if self.wall_dir > 0 { -1.0 } else { 1.0 };
        self.wall_jump_lock = WALL_JUMP_LOCK;
        self.wall_grace = 0.0; // kicking off the wall ends the slide pose immediately
        self.emit(PlayerEvent::WallJump);
    }

    fn integrate_and_collide(&mut self, dt: f32, grid: &CellGrid) {
        let was_grounded = self.on_ground;
        // Resolve X (with conveyor carry folded in), then Y, against solid cells.
        let body = self.aabb();
        let dx = (self.vx + self.conveyor_vx) * dt;
        // Step-up only applies with feet on (or just off) the ground: in mid-air it
        // would let the player ratchet up a sheer wall, and while dashing it would
        // fight the dash's straight line.
        // Climbing is excluded as well as mid-air: a ladder keeps `coyote` topped up
        // so that stepping off it still gets the grace period, and without this that
        // would let a climber ratchet sideways up a sheer wall a cell at a time.
        let may_step =
            (self.on_ground || self.coyote > 0.0) && self.dash_timer <= 0.0 && !self.climbing;
        // `resolve_axis` runs both axes in one call; passing a zero delta for the
        // other one takes neither of its clamp branches, which is exactly the
        // single-axis TypeScript call this replaces.
        let (rx_x, rx_y, rx_stepped, rx_hit_neg, rx_hit_pos) = if may_step {
            let r = move_horizontal_stepped(grid, body, dx, STEP_UP_MAX);
            (r.x, r.y, r.stepped, r.hit_left, r.hit_right)
        } else {
            let r = resolve_axis(grid, body, dx, 0.0, NO_ONE_WAY);
            (r.x, self.y, 0.0, r.hits.left, r.hits.right)
        };
        self.x = rx_x;
        if rx_stepped > 0.0 {
            self.y = rx_y;
            // Carry the rise as a visual offset that eases off, so cresting rubble
            // reads as a stride rather than the sprite jumping a cell.
            self.step_up_visual += rx_stepped;
            self.on_ground = true; // we are standing on what we just stepped onto
        }
        if rx_hit_pos {
            self.wall_dir = 1;
        } else if rx_hit_neg {
            self.wall_dir = -1;
        }
        if rx_hit_neg || rx_hit_pos {
            self.vx = 0.0;
        }

        let body2 = self.aabb();
        let dy = self.vy * dt;
        // One-way platforms block only a body descending onto them, and the test is
        // against where the feet WERE — so the value handed to the resolver is the
        // pre-move bottom edge. `NO_ONE_WAY` disables the whole mechanic for this
        // move, which is what a deliberate drop-through and a climb both want: the
        // first has been granted pass-through, and the second is threading a ladder
        // that runs through a platform's own cells.
        let one_way_from_y = if self.drop_through > 0.0 || self.climbing {
            NO_ONE_WAY
        } else {
            body2.bottom()
        };
        let ry = resolve_axis(grid, body2, 0.0, dy, one_way_from_y);
        self.y = ry.y;
        self.on_ground = ry.hits.bottom;

        // A wall touched this step keeps the slide pose alive for a moment after
        // contact is lost, so a rough cell wall cannot strobe wallSlide<->fall.
        if self.wall_dir != 0 && !ry.hits.bottom {
            self.wall_grace = WALL_GRACE;
        }

        // Landing juice: capture impact speed before vy is zeroed. The host reads
        // `land_impact` once to spawn dust/shake, then clears it.
        if ry.hits.bottom && !was_grounded {
            // Only a real descent counts: running over 1-cell bumps briefly clears
            // on_ground, and that must not fire a landing every stride.
            if self.vy > LAND_EVENT_MIN_VY {
                self.emit(PlayerEvent::Land);
            }
            self.double_jump_timer = 0.0; // the flip is over once the feet are down
            self.wall_grace = 0.0;
            if self.vy > LAND_IMPACT_MIN {
                self.land_timer = LAND_HOLD;
            }
            if self.vy > SQUASH_MIN_VY {
                self.squash = 1.0f32.min(self.vy / SQUASH_FULL_VY);
                self.land_impact = self.land_impact.max(self.vy);
            }
        }
        if ry.hits.bottom || ry.hits.top {
            self.vy = 0.0;
        }

        if ry.hits.bottom {
            self.air_jumps = MAX_AIR_JUMPS;
            // Bounce pad: launch off the surface just under the feet.
            //
            // NOTE, and this is a FAITHFULLY PORTED BUG, not a translation slip:
            // `cell_at` yields an ABSOLUTE cell and `CellGrid::get` takes a
            // WINDOW-LOCAL one. Every other cell read in this file goes through
            // `get_world`/`is_loaded_world`; this one does not, so the probe lands
            // `origin_cell_x` columns away from the cell the feet are actually on
            // and a bounce pad only fires when the window happens to be at the
            // origin. It is left exactly as it was because the parity fixture
            // freezes the original's behaviour, and changing it here would be a
            // gameplay change smuggled in under a port. Fix it in a commit that
            // says so, with the fixture regenerated deliberately.
            let cx = cell_at(self.x + PLAYER_W / 2.0);
            let cy = cell_at(self.y + PLAYER_H + 1.0);
            if mat_by_code(grid.get(cx, cy)).surface == Some(BlockSurface::Bounce) {
                self.vy = -BOUNCE_SPEED;
                self.on_ground = false;
            }
        }
    }

    fn overlap_effects(&mut self, dt: f32, grid: &CellGrid) {
        // An untouchable body does not sense the hazard at all, rather than
        // sensing it and subtracting zero. The scan below has exactly one
        // consumer — the drain and the flinch it triggers — so skipping it is
        // not an optimisation that could drift out of step with the rule; it IS
        // the rule. See `Player::untouchable` for why the flinch goes too.
        if self.untouchable {
            return;
        }

        // Highest damage among overlapped cells, applied once (spike / lava).
        let mut dmg = 0.0f32;
        for_each_overlapped_cell(grid, self.aabb(), |_cx, _cy, id| {
            let d = MAT_DAMAGE[id as usize];
            if d > dmg {
                dmg = d;
            }
        });
        if dmg > 0.0 {
            self.health -= dmg * dt;
            // Damage is continuous (lava/spikes tick every step), so the pose and the
            // event are rate-limited: one hurt reaction per HURT_REPEAT of contact.
            if self.hurt_cooldown <= 0.0 {
                self.hurt_cooldown = HURT_REPEAT;
                self.hurt_timer = HURT_TIME;
                self.emit(PlayerEvent::Hurt);
            }
        }
    }

    fn post_environment(&mut self, dt: f32) {
        // Drag scaled by submersion and converted from a per-second coefficient, so
        // it is timestep-independent. The old code multiplied by 0.86 every STEP at
        // 120Hz — effectively 0.86^120 per second, which killed any impulse in
        // about a tenth of a second and made water feel like setting concrete.
        if self.touching_liquid {
            let damp = LIQUID_DRAG.powf(dt * self.submersion);
            self.vx *= damp;
            self.vy *= damp;
        }

        // A ladder bolted to a wall is still a ladder: without the climb test the
        // wall-slide clamp would cap a descent at WALL_SLIDE_SPEED and climbing down
        // would mysteriously be slower in exactly the shafts ladders live in.
        if !self.on_ground && !self.climbing && self.wall_dir != 0 && self.vy > 0.0 {
            self.vy = self.vy.min(WALL_SLIDE_SPEED);
        }

        // No fall-death: the world is infinite in every direction, so there is no
        // bottom to fall off — only hazards (lava/spikes) drain health.
        self.health = 0.0f32.max(MAX_HEALTH.min(self.health));
    }

    /// Advance the animation: pick the state, hold it long enough that boundaries
    /// do not strobe, run the speed-driven cadence, and publish the clock the
    /// sprite reads. Called once per fixed step, so [`Player::anim`] is always the
    /// state the next draw will use.
    fn update_anim(&mut self, dt: f32) {
        self.anim_clock_t += dt;

        // Smoothed horizontal acceleration for the lean (measured, not a timer).
        let raw_accel = if dt > 0.0 {
            (self.vx - self.prev_vx) / dt
        } else {
            0.0
        };
        self.accel_x += (raw_accel - self.accel_x) * ACCEL_SMOOTH;
        self.prev_vx = self.vx;

        let next = self.anim_state();
        if next != self.anim_cur {
            let preempts = next.priority() < self.anim_cur.priority();
            if preempts || self.anim_t >= self.anim_cur.min_hold() {
                self.anim_cur = next;
                self.anim_t = 0.0;
                // Enter the run cycle on a contact frame so the first stride is clean.
                if next == AnimState::Run {
                    self.run_phase = 0.0;
                }
            }
        }
        self.anim_t += dt;
        self.advance_run_cycle(dt);
    }

    /// Run cadence is proportional to real horizontal speed — one full 4-frame
    /// cycle per `RUN_CYCLE_PX` travelled — clamped to a trudge floor and a sprint
    /// ceiling so mud-capped and conveyor-boosted speeds still read. The two
    /// contact frames sit at phase 0.0 and 0.5, so footstep events fire exactly on
    /// those crossings and always line up with the frame that plants a foot.
    fn advance_run_cycle(&mut self, dt: f32) {
        if self.anim_cur != AnimState::Run {
            return;
        }
        let cadence = clamp(
            self.vx.abs() / RUN_CYCLE_PX,
            RUN_CADENCE_MIN,
            RUN_CADENCE_MAX,
        );
        let prev = self.run_phase;
        let mut p = prev + cadence * dt;
        let wrapped = p >= 1.0;
        if wrapped {
            p -= p.floor();
        }
        self.run_phase = p;
        if wrapped {
            self.emit(PlayerEvent::Step); // lead foot plants
        } else if prev < 0.5 && p >= 0.5 {
            self.emit(PlayerEvent::Step); // trailing foot plants
        }
    }

    /// Pick the animation state from the current motion. This is an ordered chain,
    /// not a scoring pass: the first condition that matches wins, so overlapping
    /// situations (dashing while airborne, wall-sliding instead of falling,
    /// punching mid-run) resolve deterministically. Stability comes from three
    /// places — the priority/min-hold rules in `update_anim`, the run/idle
    /// hysteresis band here, and the apex deadband on jump<->fall.
    fn anim_state(&self) -> AnimState {
        if self.hurt_timer > 0.0 {
            return AnimState::Hurt;
        }
        if self.dash_timer > 0.0 {
            return AnimState::Dash;
        }
        if self.punch_timer > 0.0 {
            return AnimState::Punch;
        }
        if self.in_liquid {
            return AnimState::Swim;
        }
        // Above the air chain and below the reactive poses: a climber is neither
        // falling nor grounded, and every test in the air branch below would answer
        // wrongly for one (vy is an input on a ladder, not a consequence).
        if self.climbing {
            return CLIMB_POSE;
        }

        if !self.on_ground {
            if self.wall_grace > 0.0 && self.vy > 0.0 {
                return AnimState::WallSlide;
            }
            if self.double_jump_timer > 0.0 {
                return AnimState::DoubleJump;
            }
            if self.vy < -APEX_VY {
                return AnimState::Jump;
            }
            if self.vy > APEX_VY {
                return AnimState::Fall;
            }
            // apex deadband
            return if self.anim_cur == AnimState::Fall {
                AnimState::Fall
            } else {
                AnimState::Jump
            };
        }

        if self.land_timer > 0.0 {
            return AnimState::Land;
        }

        let speed = self.vx.abs();
        if self.input_dir != 0.0 && speed > SKID_SPEED && sign(self.vx) != self.input_dir {
            return AnimState::Skid;
        }
        let running = if self.anim_cur == AnimState::Run {
            speed > RUN_EXIT_SPEED
        } else {
            speed > RUN_ENTER_SPEED
        };
        if running {
            AnimState::Run
        } else {
            AnimState::Idle
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::materials::block;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    const SPAWN: SpawnPoint = SpawnPoint { x: 0.0, y: 0.0 };

    /// A bow, and the arrows it is allowed to spend.
    const ARROWS: &[&str] = &["arrow_fire", "arrow"];
    fn bow() -> PlayerWeapon {
        PlayerWeapon {
            damage: 9.0,
            knockback: Some(200.0),
            swing_time: Some(0.5),
            ranged: true,
            projectile_speed: Some(700.0),
            ammo: ARROWS,
        }
    }

    fn sword() -> PlayerWeapon {
        PlayerWeapon {
            damage: 22.0,
            knockback: Some(400.0),
            swing_time: Some(0.6),
            ..PlayerWeapon::default()
        }
    }

    /// Counts what was fired, so the ammo tests can tell "refused" from "fired".
    ///
    /// The counter is an `Arc` rather than a plain field because the pool is
    /// borrowed by the [`Loadout`] for the length of the attack; the assertions
    /// read their own handle instead of waiting for the borrow to end.
    #[derive(Default)]
    struct CountingPool(Arc<AtomicU32>);

    impl Projectiles for CountingPool {
        fn fire(&mut self, _x: f32, _y: f32, _dx: f32, _dy: f32, _spec: ShotSpec) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            true
        }
    }

    /// A pack with nothing in it, counting every kind it was asked for. This is
    /// the closure the ammo source used to be, given a name.
    struct EmptyPack(Arc<AtomicU32>);

    impl AmmoSource for EmptyPack {
        fn spend(&mut self, _id: &str, _n: u32) -> u32 {
            self.0.fetch_add(1, Ordering::Relaxed);
            0
        }
    }

    /// A pack holding only the PREFERRED kind, counting what it hands over.
    struct FireArrowsOnly(Arc<AtomicU32>);

    impl AmmoSource for FireArrowsOnly {
        fn spend(&mut self, id: &str, n: u32) -> u32 {
            if id == "arrow_fire" {
                self.0.fetch_add(n, Ordering::Relaxed);
                n
            } else {
                0
            }
        }
    }

    #[test]
    fn a_weapon_replaces_every_number_a_fist_supplies() {
        // The whole reason the combat contract exists: before it, the King's
        // Sword hit exactly as hard as an empty hand.
        let mut p = Player::new(SPAWN);
        assert_eq!(p.melee_damage(), PUNCH_DAMAGE);
        assert_eq!(p.melee_knockback(), PUNCH_KNOCKBACK);
        assert_eq!(p.melee_reach(), MELEE_REACH_FIST);
        assert_eq!(p.swing_time(), PUNCH_SWING_TIME);

        p.set_weapon(Some(sword()));
        assert_eq!(p.melee_damage(), 22.0);
        assert_eq!(
            p.melee_knockback(),
            scaled(400.0),
            "content authors knockback in the scaled reference frame"
        );
        assert_eq!(p.melee_reach(), MELEE_REACH_WEAPON);
        assert_eq!(p.swing_time(), 0.6);

        // A RANGED weapon contributes nothing to melee: swinging a bow is a
        // punch, and it must not land for the bow's arrow damage.
        p.set_weapon(Some(bow()));
        assert_eq!(p.melee_damage(), PUNCH_DAMAGE);
        assert_eq!(p.melee_knockback(), PUNCH_KNOCKBACK);
        assert_eq!(p.melee_reach(), MELEE_REACH_FIST);
    }

    #[test]
    fn a_non_positive_swing_time_falls_back_to_the_fist_cadence() {
        // A zero would be an infinite-DPS weapon, and content can write one.
        let mut p = Player::new(SPAWN);
        p.set_weapon(Some(PlayerWeapon {
            swing_time: Some(0.0),
            ..sword()
        }));
        assert_eq!(p.swing_time(), PUNCH_SWING_TIME);
    }

    #[test]
    fn the_hit_box_grows_forward_and_mirrors_with_the_facing() {
        let mut p = Player::new(SPAWN);
        p.x = 100.0;
        p.y = 50.0;

        let right = p.hit_box();
        assert_eq!(right.x, 100.0, "facing right, the arc starts at the body");
        assert_eq!(right.w, PLAYER_W + MELEE_REACH_FIST);
        assert_eq!(right.y, 50.0);
        assert_eq!(right.h, PLAYER_H);

        p.facing = -1.0;
        let left = p.hit_box();
        assert_eq!(left.x, 100.0 - MELEE_REACH_FIST);
        assert_eq!(
            left.right(),
            right.x + PLAYER_W,
            "the arc mirrors on the body"
        );
    }

    #[test]
    fn the_damage_window_is_a_fraction_of_the_swing_and_is_capped() {
        // Melee throughout, so the kit is only here to satisfy the signature: a
        // swing never reaches the pool.
        let mut none = NoProjectiles;
        let mut kit = Loadout::new(&mut none);

        let mut p = Player::new(SPAWN);
        assert!(!p.punching());

        assert!(p.attack(0.0, 0.0, &mut kit));
        assert_eq!(p.swing_id(), 1);
        assert!(p.punching());
        // A fist: 0.3 * 0.45 = 0.135, under the 0.16 cap.
        assert!((p.swing_live - PUNCH_SWING_TIME * SWING_WINDOW_FRAC).abs() < 1e-6);

        // Refused inside the cadence, and the id does not move.
        assert!(!p.attack(0.0, 0.0, &mut kit));
        assert_eq!(p.swing_id(), 1);

        // A slow weapon's window is clamped, so it does not also get a more
        // forgiving hit test than a dagger.
        let mut p = Player::new(SPAWN);
        p.set_weapon(Some(PlayerWeapon {
            swing_time: Some(2.0),
            ..sword()
        }));
        assert!(p.attack(0.0, 0.0, &mut kit));
        assert_eq!(p.swing_live, SWING_WINDOW_MAX);
        assert_eq!(p.punch_timer, SWING_POSE_MAX, "and so is the pose");
    }

    #[test]
    fn a_bow_with_no_ammo_source_fires_freely() {
        // What keeps a half-wired host playable, and this file headlessly
        // testable, rather than silently unable to shoot. A `Loadout` with no
        // ammo half is exactly that host.
        let fired = Arc::new(AtomicU32::new(0));
        let mut pool = CountingPool(Arc::clone(&fired));
        let mut p = Player::new(SPAWN);
        p.set_weapon(Some(bow()));
        assert!(p.attack(0.0, 0.0, &mut Loadout::new(&mut pool)));
        assert_eq!(fired.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_dry_bow_spends_its_cooldown_but_not_an_arrow() {
        // The ordering that cannot produce a free arrow, and the reason the
        // cooldown is spent anyway: a dry bow that let you retry every frame
        // would hammer the inventory scan at 120 Hz for as long as the button
        // was held.
        let fired = Arc::new(AtomicU32::new(0));
        let asked = Arc::new(AtomicU32::new(0));
        let mut pool = CountingPool(Arc::clone(&fired));
        let mut pack = EmptyPack(Arc::clone(&asked));
        let mut p = Player::new(SPAWN);
        p.set_weapon(Some(bow()));
        let mut kit = Loadout::new(&mut pool).with_ammo(&mut pack);

        assert!(!p.attack(0.0, 0.0, &mut kit), "no arrows, no shot");
        assert_eq!(fired.load(Ordering::Relaxed), 0);
        assert_eq!(asked.load(Ordering::Relaxed), 2, "both kinds were tried");
        assert!(p.swing_cd > 0.0, "the cadence still stands");
        assert!(
            !p.attack(0.0, 0.0, &mut kit),
            "so the next press is refused outright"
        );
        assert_eq!(asked.load(Ordering::Relaxed), 2, "without a second scan");
    }

    #[test]
    fn ammo_is_spent_best_first_and_the_search_stops_at_the_first_hit() {
        let fired = Arc::new(AtomicU32::new(0));
        let spent = Arc::new(AtomicU32::new(0));
        let mut pool = CountingPool(Arc::clone(&fired));
        // The pack holds the FIRST kind, which is the preferred one.
        let mut pack = FireArrowsOnly(Arc::clone(&spent));
        let mut p = Player::new(SPAWN);
        p.set_weapon(Some(bow()));

        assert!(p.attack(0.0, 0.0, &mut Loadout::new(&mut pool).with_ammo(&mut pack)));
        assert_eq!(fired.load(Ordering::Relaxed), 1);
        assert_eq!(
            spent.load(Ordering::Relaxed),
            1,
            "exactly one, of the good kind"
        );
    }

    #[test]
    fn the_event_buffer_is_capped_and_drops_the_oldest() {
        // An undrained player must not grow without bound; the newest events are
        // the ones a late consumer can still do something about.
        let mut p = Player::new(SPAWN);
        for _ in 0..MAX_EVENTS {
            p.emit(PlayerEvent::Step);
        }
        p.emit(PlayerEvent::Land);
        assert_eq!(p.events.len(), MAX_EVENTS);
        assert_eq!(p.events[MAX_EVENTS - 1], PlayerEvent::Land);
        assert_eq!(p.events[0], PlayerEvent::Step);

        let mut out = Vec::new();
        p.drain_events(&mut out);
        assert_eq!(out.len(), MAX_EVENTS);
        assert!(p.events.is_empty());
        p.drain_events(&mut out);
        assert_eq!(out.len(), MAX_EVENTS, "a second drain adds nothing");
    }

    #[test]
    fn reset_keeps_the_equipment_and_drops_the_life() {
        // Equipment survives death; re-equipping is the inventory owner's
        // decision, not a respawn's.
        let mut p = Player::new(SPAWN);
        p.set_weapon(Some(sword()));
        let mut none = NoProjectiles;
        p.attack(0.0, 0.0, &mut Loadout::new(&mut none));
        p.health = 3.0;
        p.emit(PlayerEvent::Hurt);

        p.reset();
        assert_eq!(p.melee_damage(), 22.0, "the sword is still held");
        assert_eq!(p.health, MAX_HEALTH);
        assert_eq!(p.swing_cd, 0.0);
        assert!(!p.punching());
        assert!(p.events.is_empty());
        assert_eq!(p.anim(), AnimState::Idle);
        assert_eq!(
            p.swing_id(),
            1,
            "the swing id does NOT reset: a resolver comparing ids across a \
             respawn must not see one it has already damaged with"
        );
    }

    /// A loaded window at the world origin, filled edge to edge with `id`, so a
    /// body standing anywhere inside it is overlapping that material.
    fn drowning_in(id: CellId) -> CellGrid {
        let mut g = CellGrid::new(crate::config::WINDOW_COLS, crate::config::WINDOW_ROWS);
        for cy in 0..crate::config::WINDOW_ROWS {
            for cx in 0..crate::config::WINDOW_COLS {
                g.set(cx, cy, id);
            }
        }
        g
    }

    /// A body standing in the middle of `grid`, one hazard tick already applied.
    fn one_tick_of_hazard(grid: &CellGrid, untouchable: bool) -> Player {
        let mut p = Player::new(SPAWN);
        p.x = 100.0;
        p.y = 100.0;
        p.untouchable = untouchable;
        p.overlap_effects(1.0 / 120.0, grid);
        p
    }

    #[test]
    fn an_untouchable_body_takes_no_hazard_damage_where_a_normal_one_does() {
        // Lava, which is the continuous per-tick drain rather than a discrete
        // hit, so it also proves the guard is on the DRAIN and not on some
        // event path that happens to sit next to it.
        let g = drowning_in(block::LAVA);

        let mortal = one_tick_of_hazard(&g, false);
        assert!(
            mortal.health < MAX_HEALTH,
            "the fixture did not hurt a normal body, so it proves nothing"
        );

        let spared = one_tick_of_hazard(&g, true);
        assert_eq!(spared.health, MAX_HEALTH);

        // And it is not a one-tick reprieve: a body left in the fire indefinitely
        // is still at full health.
        let mut p = one_tick_of_hazard(&g, true);
        for _ in 0..2_000 {
            p.overlap_effects(1.0 / 120.0, &g);
        }
        assert_eq!(p.health, MAX_HEALTH, "the drain leaked over time");
    }

    #[test]
    fn an_untouchable_body_does_not_flinch_at_damage_it_never_took() {
        // A hurt flash and a hurt pose with no health lost are a lie told to the
        // player about their own health bar — and the pose is the worse half,
        // because `AnimState::Hurt` preempts everything, so a body parked in
        // lava would stutter forever while taking nothing.
        let g = drowning_in(block::SPIKE);

        let mut mortal = one_tick_of_hazard(&g, false);
        let mut evs = Vec::new();
        mortal.drain_events(&mut evs);
        assert!(
            evs.contains(&PlayerEvent::Hurt),
            "the fixture did not flinch a normal body, so it proves nothing"
        );
        assert!(mortal.hurt_timer > 0.0 && mortal.hurt_cooldown > 0.0);

        let mut spared = one_tick_of_hazard(&g, true);
        evs.clear();
        spared.drain_events(&mut evs);
        assert!(!evs.contains(&PlayerEvent::Hurt), "a hurt with no hurt");
        assert_eq!(spared.hurt_timer, 0.0, "and no pose to go with it");
        assert_eq!(
            spared.hurt_cooldown, 0.0,
            "nor a spent cooldown, which would swallow the first real hit \
             after the truce ends"
        );
    }

    #[test]
    fn the_truce_is_off_by_default_and_a_respawn_does_not_end_it() {
        // Off by default is what keeps every existing suite — the parity replay
        // included — describing the same game it always did.
        let mut p = Player::new(SPAWN);
        assert!(!p.untouchable);

        // And it is a MODE the host holds, not per-life state: `reset` clears the
        // life, not the mode. See `Player::reset`.
        p.untouchable = true;
        p.health = 3.0;
        p.reset();
        assert_eq!(p.health, MAX_HEALTH);
        assert!(p.untouchable, "the respawn revoked a mode it does not own");
    }

    #[test]
    fn math_sign_is_not_signum_at_zero() {
        // The one place the two disagree, and the skid test reads it.
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(0.0f32.signum(), 1.0);
        assert_eq!(sign(-3.0), -1.0);
        assert_eq!(sign(3.0), 1.0);
    }
}

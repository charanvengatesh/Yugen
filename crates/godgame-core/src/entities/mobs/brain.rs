//! A single creature and the five brains that drive one.
//!
//! Mobs are POOLED — [`crate::entities::mobs::MobSystem`] preallocates every
//! instance and reuses dead slots, so spawning never allocates and the pool
//! length never changes. Nothing in here allocates per step either.
//!
//! Mobs READ the cell world and never write it: the sim is the only writer.
//! Collision goes exclusively through [`crate::physics::collision`], which
//! already treats an unloaded cell as solid — that is what stops a creature
//! walking off into ungenerated space when the streaming window slides.
//!
//! # What the port changed, and what it deliberately did not
//!
//! **Nothing about the behaviour.** The brains run the same phases in the same
//! order, every constant is the same number, every branch is the same branch,
//! and — because the decisions are randomised — every draw from the mob RNG
//! happens at the same point in the same stream.
//!
//! What did change is Rust-shaped:
//!
//! - `MobBrain` was a string union dispatched on with a `switch`; it is the
//!   generated [`MobBrain`] enum and a `match`, so a new brain cannot be added
//!   without this file answering for it.
//! - `MobPose` was `"idle" | "move" | "air"` collapsed to a small integer for
//!   the sprite's sake; it is the [`MobPose`] enum, which is that integer with
//!   its names kept.
//! - `surfaceRowNear` returned `NaN` for "no clean rock/air boundary in range";
//!   it returns [`Option<i32>`], so the one caller that must handle the absence
//!   cannot forget to (`Number.isNaN` is easy to write as `=== NaN`, which is
//!   always false).
//! - The module-scoped `PROBE` scratch rect and the `hazardPeak` /
//!   `hazardImmune` globals existed to keep the per-step path allocation-free
//!   in a language where `{x,y,w,h}` is a heap object. An [`Aabb`] is four
//!   floats and `Copy`, and a closure captures by reference, so all three are
//!   locals here and nothing allocates.
//! - **The RNG became a value.** See [`MobRng`].
//!
//! # Widths
//!
//! Everything here is `f32`, for the reason `player.rs` gives at length: this
//! crate's config and collider are `f32`, and widening one entity to `f64` to
//! chase a difference far below a pixel would fork the width rule. The mob RNG
//! keeps its `f64` scaling internally — see [`MobRng::rand`].

use crate::config::{CELL_SIZE, STEP_UP_REARM, cell_at};
use crate::physics::collision::{
    Aabb, NO_ONE_WAY, box_overlaps_solid, for_each_overlapped_cell, is_solid_cell,
    move_horizontal_stepped, resolve_axis,
};
use crate::sim::grid::CellGrid;
use crate::sim::materials::{BLOCKS, MAT_COUNT, MAT_DAMAGE};

use super::defs::{Dmg, MOB_DEFS, MobBrain, MobDef, MobPose};

/// How often a mob re-evaluates its situation. Physics still runs every step.
const DECIDE_INTERVAL: f32 = 0.12;
/// How often a mob samples the material it is standing in for damage.
const HAZARD_INTERVAL: f32 = 0.25;
/// Seconds a mob may spend engulfed in solid before it is culled (see
/// [`unbury`]).
const CRUSH_GRACE: f32 = 2.0;
/// How far below the local surface line a burrower rides while submerged.
const BURROW_DEPTH_CELLS: i32 = 2;
/// Seconds a burrower spends breaching before it actually erupts.
///
/// A creature that appears out of solid ground with no warning is an unfair hit
/// rather than an ambush; during the tell it rises to just under the rock line
/// and the host draws the disturbed surface above it, so the player has a beat
/// to move. Kept short — long enough to read, not long enough to trivialise.
pub const TELL_TIME: f32 = 0.45;
/// Seconds a patrolling walker stands at the lip of a drop before turning
/// round.
///
/// The old code reversed on the same frame it saw the gap, which read as the
/// creature bouncing off an invisible wall. Alerted walkers skip this entirely:
/// a hesitating chaser is a chaser you can outrun.
const LEDGE_HESITATE: f32 = 0.35;
/// How much higher than its target a flyer climbs before committing to a dive.
const DIVE_CLIMB_PX: f32 = 46.0;
/// Seconds a committed dive runs before the flyer pulls out and climbs again.
const DIVE_TIME: f32 = 0.7;

// ---------------------------------------------------------------------------
// The second RNG
// ---------------------------------------------------------------------------

/// 2^-32, the scale that turns a raw `u32` into a float in `[0, 1)`. Written
/// out rather than computed so it reads identically to the TypeScript literal.
const INV_2_POW_32: f64 = 2.3283064365386963e-10;

/// The creatures' random source — deliberately NOT the sim's.
///
/// [`crate::sim::rng::SimRng`] is consumed by the automata, and drawing from it
/// here would desync a replay of the world against the same seed. This is a
/// private xorshift32 with the same cost and a different default state.
///
/// The TypeScript held `rngState` as a module-level `let` and exported
/// `seedMobRng` / `rand` / `randRange` as free functions over it — a second
/// global mutable singleton, on top of the sim's. `SimRng` already showed what
/// that becomes in Rust, and this is the same answer: **the stream is a value,
/// threaded by `&mut`.** The arithmetic and, critically, the ORDER of
/// consumption are unchanged. A creature's patrol, its hop direction, its
/// wander phase, its variant tint and every spawn rejection all draw from one
/// stream in one fixed order, and moving the state from a module binding into a
/// struct field must not perturb a single draw.
#[derive(Clone, Debug)]
pub struct MobRng {
    state: u32,
}

impl MobRng {
    /// The state a fresh, unseeded stream starts from, and the fallback a zero
    /// seed lands on. Xorshift is stuck at zero, so it can never be that.
    pub const DEFAULT_STATE: u32 = 0x2545_f491;

    /// A stream at the default state, before any `seed` call.
    #[inline]
    pub fn new() -> MobRng {
        MobRng {
            state: MobRng::DEFAULT_STATE,
        }
    }

    /// A stream already reseeded to `seed`. See [`MobRng::seed`].
    #[inline]
    pub fn seeded(seed: u32) -> MobRng {
        let mut rng = MobRng::new();
        rng.seed(seed);
        rng
    }

    /// Reseed. A zero seed falls back to the default state, mirroring the
    /// TypeScript's `seed >>> 0 || 0x2545f491`.
    pub fn seed(&mut self, seed: u32) {
        self.state = if seed != 0 {
            seed
        } else {
            MobRng::DEFAULT_STATE
        };
        // Discard a few steps so nearby seeds don't produce correlated openings.
        for _ in 0..8 {
            self.next_u32();
        }
    }

    /// Next raw 32-bit value. The three shifts are the whole algorithm; in
    /// JavaScript they relied on `<<`/`>>>` being defined on a 32-bit view of a
    /// double, and here the type IS `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        self.state
    }

    /// Uniform float in `[0, 1)`.
    ///
    /// The `f64` multiply is the TypeScript's, kept at that width because it is
    /// the one place the raw `u32` is turned into a fraction and 24 bits of
    /// mantissa cannot represent the whole range; the result is then narrowed,
    /// because every number it feeds is `f32`. Doing the multiply in `f32`
    /// instead would quantise the stream itself.
    #[inline]
    pub fn rand(&mut self) -> f32 {
        (f64::from(self.next_u32()) * INV_2_POW_32) as f32
    }

    /// Uniform float in `[a, b)`.
    #[inline]
    pub fn rand_range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand()
    }
}

impl Default for MobRng {
    #[inline]
    fn default() -> MobRng {
        MobRng::new()
    }
}

// ---------------------------------------------------------------------------
// The creature
// ---------------------------------------------------------------------------

/// The sprite's animation clock, owned and mutated by the creature.
///
/// It lives here rather than in the renderer because it is advanced on the
/// fixed step, alongside the physics, and because one field of it is genuine
/// simulation state: `state_t` is what the burrower's breach tell is driven by.
///
/// `state_t` is deliberately NOT reset when [`Mob::pose`] changes, which is the
/// behaviour the pre-migration `poseT` had: a mob's poses are free-running
/// loops, and a walker that crosses the `|vx| > 4` threshold twice a second
/// would otherwise snap its walk cycle back to frame 0 each time and read as a
/// stutter. It is reset in exactly one place — a burrower's eruption, which is
/// a genuine restart. `clock_t` is advanced alongside it so an `ambient`
/// sequence (blinking) works the day some creature authors one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MobClock {
    /// Seconds spent in the current sequence.
    pub state_t: f32,
    /// Free-running seconds, for ambient loops.
    pub clock_t: f32,
    /// Sub-frame phase, for the `phase` playback mode.
    pub phase: f32,
}

/// One creature.
///
/// Every instance lives in the system's fixed pool and is reused; `active` is
/// what says whether the rest of the fields mean anything.
#[derive(Clone, Debug)]
pub struct Mob {
    /// This slot holds a live creature.
    pub active: bool,
    /// Set by the def on spawn; never read while inactive.
    ///
    /// The TypeScript wrote `def!: MobDef` — a non-null assertion saying "the
    /// pool holds a dead slot until `init` fills this in". Rust has no such
    /// hole, so a dead slot points at code 0. Reading it is meaningless in
    /// exactly the same way the `!` was documenting.
    pub def: &'static MobDef,
    /// COLLISION BOX, in world px.
    ///
    /// Not the art rect: a sprite may overhang it (see `MobDef::art_pad_x_px`).
    /// Everything physical — footing, collision, the burrower's breach tell —
    /// reads this; only the draw calls add the pad.
    pub body: Aabb,
    /// Horizontal velocity, px/s.
    pub vx: f32,
    /// Vertical velocity, px/s. Y grows downward.
    pub vy: f32,
    /// Which way the body is pointing: exactly `1.0` or `-1.0`.
    ///
    /// The TypeScript typed this `1 | -1` and Rust could too (a two-variant
    /// enum), but it is multiplied by a speed on nearly every line that reads
    /// it, so it is the same `f32` the player's `facing` is.
    pub facing: f32,
    /// Hit points.
    pub health: f32,
    /// Colour variant index, so two of a species are not pixel-identical.
    pub variant: i32,

    /// Feet resting on something solid.
    pub on_ground: bool,
    /// Which loop the sprite is playing. Set by the brains.
    pub pose: MobPose,
    /// The sprite's clock. See [`MobClock`].
    pub clock: MobClock,

    /// Countdown to the next AI decision (staggered per mob at spawn).
    pub decide_t: f32,
    /// Countdown to the next hop / dive / surfacing, whatever the brain uses.
    pub beat_t: f32,
    /// Countdown to the next material-hazard probe.
    pub hazard_t: f32,
    /// Horizontal travel since the last step-up, px, saturating at
    /// [`STEP_UP_REARM`]. Same gate as the player's, for the same reason: a
    /// patroller's `turn_on_wall` only fires when the mover reports BLOCKED,
    /// and an ungated step-up prevents "blocked" against exactly the ragged
    /// vertical faces a walker should turn at — so creatures climbed pillars
    /// and pines instead of turning. See [`STEP_UP_REARM`].
    pub step_rearm: f32,
    /// Countdown to the next contact hit.
    pub attack_cd: f32,
    /// Damage blink, 0..1, decayed every step.
    pub flash: f32,

    /// Chasing (hostile) or fleeing (passive) — set by `decide`.
    pub alerted: bool,
    /// A patrolling walker pauses on some beats instead of marching forever.
    pub idling: bool,

    /// Burrower only: travelling inside rock, invisible and harmless.
    pub buried: bool,
    /// Burrower only: seconds left in the breach tell. >0 means "about to
    /// erupt".
    pub tell_t: f32,
    /// Walker only: seconds left standing at a ledge before turning round.
    pub hesitate_t: f32,
    /// Flyer only: free-running phase for the wander.
    pub wander: f32,
    /// Flyer only: how fast that phase runs.
    pub wander_rate: f32,
    /// Flyer only: seconds left in a committed dive. 0 = climbing/circling.
    pub dive_t: f32,
    /// Flyer only: the point the current dive was aimed at, world px.
    pub dive_x: f32,
    /// Flyer only: the point the current dive was aimed at, world px.
    pub dive_y: f32,

    /// Countdown to the next ranged shot. Only meaningful when `def.ranged` is
    /// set.
    pub shot_cd: f32,
    /// Raised by [`step_mob`] when the creature wants to fire this step, and
    /// cleared by the system once the shot is placed. The projectile pool lives
    /// in the system, not on the creature, so the brain declares intent rather
    /// than reaching across into a pool it does not own.
    pub wants_shot: bool,

    /// Swing id already applied to this mob, so one swing lands once.
    ///
    /// The TypeScript used `-1` as "never hit"; `None` is that sentinel with
    /// the type system enforcing that it is never compared as a number.
    pub last_punch_id: Option<u32>,
    /// Seconds spent overlapping solid — a buried-alive escape hatch.
    pub crush_t: f32,
    /// Raised when the mob must be recycled regardless of distance.
    pub doomed: bool,
}

impl Mob {
    /// A dead slot. The pool is filled with these once, at construction.
    pub fn new() -> Mob {
        Mob {
            active: false,
            def: &MOB_DEFS[0],
            body: Aabb::new(0.0, 0.0, 0.0, 0.0),
            vx: 0.0,
            vy: 0.0,
            facing: 1.0,
            health: 0.0,
            variant: 0,
            on_ground: false,
            pose: MobPose::Idle,
            clock: MobClock::default(),
            decide_t: 0.0,
            beat_t: 0.0,
            step_rearm: STEP_UP_REARM,
            hazard_t: 0.0,
            attack_cd: 0.0,
            flash: 0.0,
            alerted: false,
            idling: false,
            buried: false,
            tell_t: 0.0,
            hesitate_t: 0.0,
            wander: 0.0,
            wander_rate: 1.0,
            dive_t: 0.0,
            dive_x: 0.0,
            dive_y: 0.0,
            shot_cd: 0.0,
            wants_shot: false,
            last_punch_id: None,
            crush_t: 0.0,
            doomed: false,
        }
    }

    /// Centre of the collision box, world px.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (
            self.body.x + self.body.w * 0.5,
            self.body.y + self.body.h * 0.5,
        )
    }
}

impl Default for Mob {
    fn default() -> Mob {
        Mob::new()
    }
}

// ---------------------------------------------------------------------------
// Hazard tags
// ---------------------------------------------------------------------------

/// Which damage tag each material's `MAT_DAMAGE` counts as, indexed by block
/// code. Built once on first read so the hazard probe stays a single indexed
/// read; the lookup is by id rather than by `code_of` so a block another agent
/// has renamed degrades to "untagged" instead of failing at load.
///
/// Untagged hazards (0) are never immune to anything, which is the right
/// default: a creature is immune to the things it is MADE of, and anything the
/// world grows that nobody has classified should still hurt.
static HAZARD_TAG: std::sync::LazyLock<Vec<u32>> = std::sync::LazyLock::new(|| {
    let mut out = vec![0u32; MAT_COUNT];
    let mut tag = |id: &str, bits: Dmg| {
        if let Some(i) = BLOCKS.iter().position(|b| b.id == id) {
            out[i] = bits.bits();
        }
    };
    tag("lava", Dmg::FIRE);
    tag("fire", Dmg::FIRE);
    tag("ember", Dmg::FIRE);
    tag("acid", Dmg::ACID);
    tag("spike", Dmg::IMPACT);
    out
});

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/// Exponential-ish approach: move `cur` a `rate * dt` fraction of the way to
/// `want`, clamped so a large `dt` cannot overshoot past the target.
#[inline]
fn approach(cur: f32, want: f32, rate: f32, dt: f32) -> f32 {
    let k = 1.0f32.min(rate * dt);
    cur + (want - cur) * k
}

/// Is there solid footing one cell ahead of, and one cell below, the feet?
fn ground_ahead(grid: &CellGrid, m: &Mob, dir: f32) -> bool {
    let ax = if dir > 0.0 {
        m.body.x + m.body.w + CELL_SIZE as f32 * 0.5
    } else {
        m.body.x - CELL_SIZE as f32 * 0.5
    };
    let ay = m.body.y + m.body.h + CELL_SIZE as f32 * 0.5;
    is_solid_cell(grid, cell_at(ax), cell_at(ay))
}

/// Topmost solid row at a column, searching a band around a guess.
///
/// `None` when there is no clean rock/air boundary in range — which includes
/// the case where the cells above are UNLOADED, since `is_solid_cell` reports
/// those as solid. That is exactly the answer a burrower needs: no boundary
/// means do not travel here.
fn surface_row_near(grid: &CellGrid, cx: i32, cy_guess: i32, span: i32) -> Option<i32> {
    ((cy_guess - span)..=(cy_guess + span))
        .find(|&cy| is_solid_cell(grid, cx, cy) && !is_solid_cell(grid, cx, cy - 1))
}

/// Flip a walker around and kill its momentum.
fn turn(m: &mut Mob) {
    m.facing = if m.facing > 0.0 { -1.0 } else { 1.0 };
    m.vx = 0.0;
}

/// Re-evaluate alert state. Cheap enough to run every step; run on a beat
/// anyway.
fn decide(m: &mut Mob, dx: f32, dy: f32) {
    let d = m.def;
    let near = |r: f32| r > 0.0 && dx.abs() < r && dy.abs() < d.aggro_y_px;
    m.alerted = if d.flee_px > 0.0 {
        near(d.flee_px)
    } else {
        near(d.aggro_px)
    };
}

// ---------------------------------------------------------------------------
// The step
// ---------------------------------------------------------------------------

/// Advance one creature by a fixed step. `tx`/`ty` are the player's CENTRE in
/// world px.
pub fn step_mob(m: &mut Mob, dt: f32, grid: &CellGrid, tx: f32, ty: f32, rng: &mut MobRng) {
    let d = m.def;
    m.clock.state_t += dt;
    m.clock.clock_t += dt;
    m.attack_cd = (m.attack_cd - dt).max(0.0);
    m.shot_cd = (m.shot_cd - dt).max(0.0);
    m.flash = (m.flash - dt * 5.0).max(0.0);
    m.beat_t -= dt;
    m.decide_t -= dt;

    let (cx, cy) = m.center();
    let dx = tx - cx;
    let dy = ty - cy;

    if m.decide_t <= 0.0 {
        m.decide_t = DECIDE_INTERVAL;
        decide(m, dx, dy);
    }

    // Ranged intent is brain-independent: a shooter is a shooter whether it
    // walks, hops or flies, and every brain already keeps `alerted` current. The
    // creature faces its shot so the sprite reads as aiming rather than as
    // firing sideways.
    if let Some(r) = d.ranged
        && m.alerted
        && !m.buried
        && m.shot_cd <= 0.0
        && dx.abs() < r.range
        && dy.abs() < r.range * 0.7
    {
        m.shot_cd = r.cooldown;
        m.wants_shot = true;
        m.facing = if dx > 0.0 { 1.0 } else { -1.0 };
    }

    match d.brain {
        MobBrain::Walker | MobBrain::Skitter => {
            step_walker(m, dt, grid, dx, d.brain == MobBrain::Skitter, rng);
        }
        MobBrain::Hopper => step_hopper(m, dt, grid, dx, rng),
        MobBrain::Flyer => step_flyer(m, dt, grid, dx, dy, rng),
        MobBrain::Burrower => step_burrower(m, dt, grid, dx, dy, rng),
    }

    if !m.buried {
        apply_hazards(m, dt, grid);
        unbury(m, dt, grid);
    }
}

// --- brains ----------------------------------------------------------------

/// Patrols, turns at ledges and walls, and either charges the player or bolts
/// from them. The ledge probe is what keeps a walker on its platform: without it
/// a chasing mob marches straight into the first pit and the world fills up with
/// creatures stuck at the bottom of holes.
fn step_walker(m: &mut Mob, dt: f32, grid: &CellGrid, dx: f32, flees: bool, rng: &mut MobRng) {
    let d = m.def;
    let toward = if dx > 0.0 { 1.0 } else { -1.0 };

    if m.alerted {
        m.facing = if flees { -toward } else { toward };
        m.idling = false;
        m.hesitate_t = 0.0; // a chase does not get to stop and think at the edge
    } else if m.beat_t <= 0.0 {
        m.beat_t = d.beat * rng.rand_range(0.6, 1.4);
        m.idling = rng.rand() < 0.28;
        if !m.idling && rng.rand() < 0.4 {
            m.facing = if m.facing > 0.0 { -1.0 } else { 1.0 };
        }
    }

    // Hesitation at a drop: stand still, look over the edge, THEN turn.
    // Reversing on the same frame the ledge probe fires reads as bouncing off
    // nothing; the pause is what makes the creature look like it noticed the
    // gap.
    if m.hesitate_t > 0.0 {
        m.hesitate_t -= dt;
        m.vx = approach(m.vx, 0.0, 18.0, dt);
        if m.hesitate_t <= 0.0 {
            turn(m);
        }
        move_grounded(m, dt, grid, true);
        m.pose = if m.on_ground {
            MobPose::Idle
        } else {
            MobPose::Air
        };
        return;
    }

    let speed = if m.alerted { d.chase_speed } else { d.speed };
    let want = if m.idling && !m.alerted {
        0.0
    } else {
        m.facing * speed
    };
    m.vx = approach(m.vx, want, 12.0, dt);

    // Ledge: refuse to step off. A jumper that is committed to a chase will hop
    // the gap instead of stopping, which is what makes a chase feel deliberate.
    if m.on_ground && !m.idling && !ground_ahead(grid, m, m.facing) {
        if d.jump_speed > 0.0 && m.alerted {
            m.vy = -d.jump_speed;
            m.on_ground = false;
        } else {
            m.hesitate_t = LEDGE_HESITATE * rng.rand_range(0.7, 1.3);
            m.vx = 0.0;
        }
    }

    move_grounded(m, dt, grid, true);
    m.pose = if !m.on_ground {
        MobPose::Air
    } else if m.vx.abs() > 4.0 {
        MobPose::Move
    } else {
        MobPose::Idle
    };
}

/// Grounded blob: sits, gathers, then commits to a single arc.
fn step_hopper(m: &mut Mob, dt: f32, grid: &CellGrid, dx: f32, rng: &mut MobRng) {
    let d = m.def;
    if m.on_ground {
        m.vx = approach(m.vx, 0.0, 14.0, dt);
        if m.beat_t <= 0.0 {
            m.beat_t = d.beat * rng.rand_range(0.7, 1.3);
            let dir = if m.alerted {
                if dx > 0.0 { 1.0 } else { -1.0 }
            } else if rng.rand() < 0.5 {
                1.0
            } else {
                -1.0
            };
            m.facing = dir;
            m.vy = -d.jump_speed;
            m.vx = dir * d.speed;
            m.on_ground = false;
        }
    }
    move_grounded(m, dt, grid, false);
    m.pose = if !m.on_ground {
        MobPose::Air
    } else if m.beat_t < 0.25 {
        MobPose::Move
    } else {
        MobPose::Idle
    };
}

/// Free flight. There is no gravity and no pathfinding: a flyer steers toward a
/// desired velocity and bounces off whatever it meets, which in a cave reads as
/// erratic fluttering. Solids and unloaded space both stop it, so it can neither
/// clip through rock nor wander out of the streamed window.
///
/// A hostile flyer does NOT home continuously. Steering at the player every
/// frame produces a creature that hovers just out of reach and slides sideways
/// as you walk, which reads as a cursor rather than as an animal. Instead it
/// alternates:
///
///   CLIMB  — hold station above and behind the player, gaining DIVE_CLIMB_PX
///            of height. This is the readable wind-up.
///   DIVE   — on the beat, snapshot where the player IS and commit to that point
///            at chase speed for DIVE_TIME seconds. It cannot re-aim mid-dive,
///            so sidestepping actually works and the attack has a counter.
///
/// Fleeing flyers keep the old continuous steering: panic has no wind-up.
fn step_flyer(m: &mut Mob, dt: f32, grid: &CellGrid, dx: f32, dy: f32, rng: &mut MobRng) {
    let d = m.def;
    m.wander += dt * m.wander_rate;
    let (cx, cy) = m.center();

    let want_x: f32;
    let want_y: f32;
    if m.alerted && d.contact_damage > 0.0 {
        if m.dive_t > 0.0 {
            m.dive_t -= dt;
            let ax = m.dive_x - cx;
            let ay = m.dive_y - cy;
            let len = 1.0f32.max((ax * ax + ay * ay).sqrt());
            want_x = (ax / len) * d.chase_speed;
            want_y = (ay / len) * d.chase_speed;
        } else {
            if m.beat_t <= 0.0 {
                m.beat_t = d.beat * rng.rand_range(0.8, 1.3);
                m.dive_t = DIVE_TIME;
                m.dive_x = cx + dx;
                m.dive_y = cy + dy;
                // The TypeScript assigned wantX/wantY here and then immediately
                // overwrote them with the climb below — a dead store, kept
                // faithfully by simply not writing it: the climb runs on the
                // beat frame too, which is what the original actually did.
            }
            // Climb: match the player's column with an offset that swings with
            // the wander, and sit DIVE_CLIMB_PX above. The offset is what makes
            // the approach an arc instead of a straight line.
            let ax = dx + m.wander.cos() * 40.0;
            let ay = dy - DIVE_CLIMB_PX;
            let len = 1.0f32.max((ax * ax + ay * ay).sqrt());
            want_x = (ax / len) * d.speed;
            want_y = (ay / len) * d.speed;
        }
    } else if m.alerted {
        let len = 1.0f32.max((dx * dx + dy * dy).sqrt());
        want_x = (-dx / len) * d.speed * 1.7;
        want_y = (-dy / len) * d.speed * 1.7;
    } else {
        want_x = m.wander.cos() * d.speed;
        want_y = (m.wander * 0.63).sin() * d.speed * 0.55;
    }

    // Steer hard into a dive and gently otherwise: the rate IS the bank.
    let rate = if m.dive_t > 0.0 { 11.0 } else { 5.0 };
    m.vx = approach(m.vx, want_x, rate, dt);
    m.vy = approach(m.vy, want_y, rate, dt);
    // A deadband, not a clamp: below 1px/s of drift the facing is LEFT ALONE, so
    // a flyer hovering on the spot keeps pointing the way it last flew instead
    // of flipping on every jitter of the steering.
    if m.vx.abs() > 1.0 {
        m.facing = if m.vx > 0.0 { 1.0 } else { -1.0 };
    }

    // One axis at a time, and `NO_ONE_WAY` on both: a bat should cross a
    // platform, not perch on it.
    let rx = resolve_axis(grid, m.body, m.vx * dt, 0.0, NO_ONE_WAY);
    m.body.x = rx.x;
    if rx.hit_x() {
        m.vx = -m.vx * 0.5;
        m.wander += std::f32::consts::PI;
        m.dive_t = 0.0; // a dive that ends in rock has ended
    }
    let ry = resolve_axis(grid, m.body, 0.0, m.vy * dt, NO_ONE_WAY);
    m.body.y = ry.y;
    if ry.hit_y() {
        m.vy = -m.vy * 0.5;
        m.wander += 1.7;
        m.dive_t = 0.0;
    }
    m.on_ground = false;
    m.pose = MobPose::Move;
}

/// Rides just under the rock line, then erupts.
///
/// While buried the mob is moved by hand rather than by collision — it is inside
/// solid, which is the one place the shared resolver cannot help. The safety is
/// [`surface_row_near`]: it only returns a burial depth where there is a real
/// rock/air boundary in the loaded window, so the worm cannot tunnel into
/// unstreamed space or surface inside a mountain. When it finds nothing it
/// simply stops advancing and brings its surfacing forward.
fn step_burrower(m: &mut Mob, dt: f32, grid: &CellGrid, dx: f32, dy: f32, rng: &mut MobRng) {
    let d = m.def;
    if m.buried {
        let dir = if dx > 0.0 { 1.0 } else { -1.0 };
        m.facing = dir;
        // During the tell it stops travelling and rises: the whole point is that
        // the player gets a fixed spot to step off, so a worm that kept tracking
        // during its own warning would make the warning useless.
        let telling = m.tell_t > 0.0;
        let next_x = if telling {
            m.body.x
        } else {
            m.body.x + dir * d.speed * dt
        };
        let probe_cx = cell_at(next_x + m.body.w * 0.5);
        let guess_cy = cell_at(m.body.y + m.body.h * 0.5);
        match surface_row_near(grid, probe_cx, guess_cy, 10) {
            None => {
                // Nowhere legal to swim to — surface at the next opportunity
                // instead of grinding against the edge of the world.
                if m.beat_t > 0.3 {
                    m.beat_t = 0.3;
                }
            }
            Some(sr) => {
                m.body.x = next_x;
                // Rise to just under the rock line while telling, so the body is
                // right where the surface disturbance the host draws says it is.
                let depth = if telling { 1 } else { BURROW_DEPTH_CELLS };
                let target_y = ((sr + depth) * CELL_SIZE) as f32;
                m.body.y = approach(m.body.y, target_y, if telling { 14.0 } else { 6.0 }, dt);
            }
        }

        if telling {
            m.tell_t -= dt;
            if m.tell_t <= 0.0 {
                erupt(m, grid, 1.0);
            }
            m.pose = MobPose::Idle;
            return;
        }

        let wants_ambush = m.beat_t <= 0.0 && dx.abs() < d.aggro_px && dy.abs() < d.aggro_y_px;
        // Nothing to ambush for a long while: surface anyway rather than sit as
        // a permanently invisible occupant of a pool slot.
        if wants_ambush {
            m.tell_t = TELL_TIME;
        } else if m.beat_t <= -6.0 {
            erupt(m, grid, 0.6);
        }
        m.pose = MobPose::Idle;
        return;
    }

    m.vx = approach(m.vx, m.facing * d.chase_speed * 0.5, 3.0, dt);
    move_grounded(m, dt, grid, false);
    m.pose = if m.on_ground {
        MobPose::Move
    } else {
        MobPose::Air
    };
    if m.beat_t <= 0.0 && m.on_ground {
        m.buried = true;
        m.beat_t = d.beat * rng.rand_range(0.7, 1.4);
        m.vx = 0.0;
        m.vy = 0.0;
    }
}

/// Break the surface. The worm is INSIDE rock while buried, so it has to be
/// lifted clear of the rock line in the same instant it stops being buried —
/// otherwise it spends its first frames above ground embedded in solid matter,
/// which the crush handler would then have to dig it out of. If no clean rock
/// line is in reach it stays down and tries again on the next beat.
fn erupt(m: &mut Mob, grid: &CellGrid, force: f32) {
    let d = m.def;
    let (mx, my) = m.center();
    let sr = match surface_row_near(grid, cell_at(mx), cell_at(my), 10) {
        Some(sr) => sr,
        None => {
            m.beat_t = 0.4;
            m.tell_t = 0.0;
            return;
        }
    };
    m.body.y = (sr * CELL_SIZE) as f32 - m.body.h;
    if box_overlaps_solid(grid, m.body) {
        m.beat_t = 0.4;
        m.tell_t = 0.0;
        return;
    }
    m.buried = false;
    m.tell_t = 0.0;
    m.vy = -d.jump_speed * force;
    m.vx = m.facing * d.chase_speed * force;
    m.on_ground = false;
    m.beat_t = 2.2; // seconds allowed above ground
    m.clock.state_t = 0.0;
}

// --- shared physics --------------------------------------------------------

/// X-then-Y axis resolution against the cell world, exactly as the player does
/// it. `turn_on_wall` is what separates a patroller (which reverses when it hits
/// a wall) from a blob in mid-arc (which just stops horizontally).
fn move_grounded(m: &mut Mob, dt: f32, grid: &CellGrid, turn_on_wall: bool) {
    let d = m.def;
    if m.vx != 0.0 {
        // The re-arm gate: a creature that has not travelled since its last
        // lift is offered no step-up, so a ragged face reads as BLOCKED and
        // `turn_on_wall` gets to do its job.
        let step = if m.step_rearm >= STEP_UP_REARM {
            d.step_up_max
        } else {
            0.0
        };
        let mv = move_horizontal_stepped(grid, m.body, m.vx * dt, step);
        // Restart at this frame's travel on a lift, never at zero — the same
        // overshoot credit the player's accumulator makes, for the same
        // deadlock; the argument lives in `player.rs`.
        let travel = (mv.x - m.body.x).abs();
        m.step_rearm = if mv.stepped > 0.0 {
            travel.min(STEP_UP_REARM)
        } else {
            (m.step_rearm + travel).min(STEP_UP_REARM)
        };
        m.body.x = mv.x;
        m.body.y = mv.y;
        if mv.blocked() {
            if turn_on_wall && m.on_ground && d.jump_speed > 0.0 && m.alerted {
                m.vy = -d.jump_speed;
                m.on_ground = false;
            } else if turn_on_wall {
                turn(m);
            } else {
                m.vx = 0.0;
            }
        }
    }

    // Pass the pre-move bottom edge so a GROUNDED creature lands on a one-way
    // platform instead of dropping through it. Walkers patrol ledges and hoppers
    // aim at them, so a mineshaft's plank floor being invisible to creatures
    // would read as a bug the moment anything walked over one.
    //
    // Deliberately only here, in the grounded mover. The flyer resolve omits it
    // — a bat should cross a platform, not perch on it — and a burrower travels
    // inside solid rock, where the question never arises.
    let bottom_before = m.body.bottom();
    m.vy = d.max_fall.min(m.vy + d.gravity * dt);
    let ry = resolve_axis(grid, m.body, 0.0, m.vy * dt, bottom_before);
    m.body.y = ry.y;
    if ry.hits.bottom {
        m.on_ground = true;
        m.vy = 0.0;
    } else {
        m.on_ground = false;
        if ry.hits.top {
            m.vy = 0.0;
        }
    }
}

/// Lava, acid and spikes hurt creatures too — sampled off the same cell tags.
fn apply_hazards(m: &mut Mob, dt: f32, grid: &CellGrid) {
    m.hazard_t -= dt;
    if m.hazard_t > 0.0 {
        return;
    }
    m.hazard_t = HAZARD_INTERVAL;
    // `lava_immune` is the blunt form: nothing the world is MADE of can hurt it.
    // `immune_mask` is the per-tag form and is applied inside the probe.
    if m.def.lava_immune {
        return;
    }
    let immune = m.def.immune_mask.bits();
    let mut peak = 0.0f32;
    for_each_overlapped_cell(grid, m.body, |_cx, _cy, id| {
        if HAZARD_TAG[id as usize] & immune != 0 {
            return;
        }
        let d = MAT_DAMAGE[id as usize];
        if d > peak {
            peak = d;
        }
    });
    if peak > 0.0 {
        m.health -= peak * HAZARD_INTERVAL;
        m.flash = 1.0;
    }
}

/// Escape hatch for a creature the WORLD has closed around it. This is a falling
/// sand game: a mob standing under a collapsing dune ends up genuinely inside
/// solid matter, which no amount of axis resolution can undo (the resolver only
/// stops you ENTERING solids). Lift it a cell at a time; if it is still engulfed
/// after CRUSH_GRACE seconds it is buried for good and gets recycled rather than
/// left vibrating inside a rock forever.
fn unbury(m: &mut Mob, dt: f32, grid: &CellGrid) {
    if !box_overlaps_solid(grid, m.body) {
        m.crush_t = 0.0;
        return;
    }
    m.crush_t += dt;
    for i in 1..=3 {
        let probe = Aabb {
            y: m.body.y - (i * CELL_SIZE) as f32,
            ..m.body
        };
        if !box_overlaps_solid(grid, probe) {
            m.body.y = probe.y;
            m.vy = 0.0;
            m.crush_t = 0.0;
            return;
        }
    }
    if m.crush_t > CRUSH_GRACE {
        m.doomed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::materials::block;

    /// One xorshift32 step, computed the long way, as an oracle.
    fn step(mut x: u32) -> u32 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    }

    #[test]
    fn the_mob_stream_is_a_xorshift32_from_its_own_default_state() {
        // Its own state, NOT the sim RNG's: two streams that started from the
        // same number would be one stream drawn twice.
        assert_ne!(
            MobRng::DEFAULT_STATE,
            crate::sim::rng::SimRng::DEFAULT_STATE
        );
        let mut want = MobRng::DEFAULT_STATE;
        let mut rng = MobRng::new();
        for i in 0..64 {
            want = step(want);
            assert_eq!(rng.next_u32(), want, "step {i} diverged");
        }
    }

    #[test]
    fn seeding_discards_exactly_eight_steps() {
        let mut oracle = MobRng { state: 12345 };
        for _ in 0..8 {
            oracle.next_u32();
        }
        assert_eq!(MobRng::seeded(12345).state, oracle.state);
    }

    #[test]
    fn a_zero_seed_falls_back_to_the_default_state() {
        let mut oracle = MobRng::new();
        for _ in 0..8 {
            oracle.next_u32();
        }
        assert_eq!(MobRng::seeded(0).state, oracle.state);
    }

    #[test]
    fn rand_stays_in_the_unit_interval_and_rand_range_in_its_own() {
        let mut rng = MobRng::seeded(0xdead_beef);
        for _ in 0..10_000 {
            let v = rng.rand();
            assert!((0.0..1.0).contains(&v), "rand() gave {v}");
            let r = rng.rand_range(-3.0, 7.0);
            assert!((-3.0..7.0).contains(&r), "rand_range gave {r}");
        }
    }

    #[test]
    fn hazard_tags_classify_the_five_materials_and_nothing_else() {
        // Block CODE order, which is content's, not the order the tags are
        // declared in — so the set is what is asserted.
        let mut tagged: Vec<&str> = BLOCKS
            .iter()
            .filter(|b| HAZARD_TAG[b.code as usize] != 0)
            .map(|b| b.id)
            .collect();
        tagged.sort_unstable();
        assert_eq!(tagged, vec!["acid", "ember", "fire", "lava", "spike"]);
        assert_eq!(
            HAZARD_TAG[block::LAVA as usize],
            Dmg::FIRE.bits(),
            "lava is fire"
        );
        assert_eq!(HAZARD_TAG[0], 0, "air is not a hazard");
    }

    #[test]
    fn approach_never_overshoots_however_long_the_step() {
        assert_eq!(approach(0.0, 10.0, 6.0, 1000.0), 10.0);
        assert_eq!(approach(10.0, 0.0, 6.0, 1000.0), 0.0);
        // And a zero-length step moves nothing.
        assert_eq!(approach(3.0, 10.0, 6.0, 0.0), 3.0);
    }

    // --- the brains, against a real grid ------------------------------------
    //
    // One test per brain, each asserting the thing that brain is FOR. They run
    // several hundred fixed steps because a brain's shape only shows over a beat
    // or two, and they run against a real `CellGrid` because every one of them
    // is a conversation with the collider.

    /// A grid whose window starts at the world origin, with solid rock from
    /// `floor_row` down. World cells and local cells coincide, so a test reads
    /// like the coordinates it writes.
    fn world(cols: i32, rows: i32, floor_row: i32) -> CellGrid {
        let mut g = CellGrid::new(cols, rows);
        for cy in floor_row..rows {
            for cx in 0..cols {
                g.set(cx, cy, block::STONE);
            }
        }
        g
    }

    /// A live creature of a species, at a world-px position.
    fn spawn(id: &str, x: f32, y: f32) -> Mob {
        let def = super::super::defs::def_by_id(id);
        let mut m = Mob::new();
        m.active = true;
        m.def = def;
        m.body = Aabb::new(x, y, def.w_px, def.h_px);
        m.health = def.max_health;
        // The one field `MobSystem::init` derives from the def rather than
        // zeroing; a burrower that started above ground would be testing the
        // wrong half of its brain.
        m.buried = def.brain == MobBrain::Burrower;
        m
    }

    /// The mobs' own fixed step.
    const DT: f32 = 1.0 / 60.0;
    /// Far enough away that nothing aggresses. The brains still run.
    const FAR: f32 = 100_000.0;

    #[test]
    fn a_walker_patrols_along_the_ground_and_never_ends_up_inside_it() {
        let g = world(64, 64, 40);
        let floor_y = 40.0 * CELL_SIZE as f32;
        let mut m = spawn("grubling", 100.0, floor_y - 10.0);
        let mut rng = MobRng::seeded(7);
        let start_x = m.body.x;
        let mut travelled = 0.0f32;

        for i in 0..900 {
            step_mob(&mut m, DT, &g, FAR, FAR, &mut rng);
            assert!(
                !box_overlaps_solid(&g, m.body),
                "walked into rock at step {i}"
            );
            assert!(
                m.body.bottom() <= floor_y + 0.001,
                "sank through the floor at step {i}: {}",
                m.body.y
            );
            travelled = travelled.max((m.body.x - start_x).abs());
        }
        assert!(
            m.on_ground,
            "a walker on a flat floor ends up standing on it"
        );
        assert_eq!(m.body.bottom(), floor_y, "feet exactly on the floor's face");
        assert!(
            travelled > 20.0,
            "a patrol that never patrolled: {travelled}"
        );
    }

    #[test]
    fn a_walker_hesitates_at_a_ledge_and_turns_instead_of_stepping_off() {
        // A floor that stops halfway. The creature starts on the solid half,
        // walking toward the gap; the ledge probe is what keeps it there.
        let mut g = CellGrid::new(64, 64);
        for cy in 40..64 {
            for cx in 0..30 {
                g.set(cx, cy, block::STONE);
            }
        }
        let floor_y = 40.0 * CELL_SIZE as f32;
        let mut m = spawn("grubling", 120.0, floor_y - 10.0);
        m.facing = 1.0; // at the gap
        let mut rng = MobRng::seeded(3);

        for i in 0..900 {
            step_mob(&mut m, DT, &g, FAR, FAR, &mut rng);
            assert!(
                m.body.right() <= 30.0 * CELL_SIZE as f32,
                "stepped off the ledge at step {i}: x={}",
                m.body.x
            );
        }
        assert!(m.on_ground);
    }

    #[test]
    fn a_hopper_leaves_the_ground() {
        let g = world(64, 64, 40);
        let floor_y = 40.0 * CELL_SIZE as f32;
        let mut m = spawn("slime", 100.0, floor_y - 10.0);
        let mut rng = MobRng::seeded(11);

        let mut airborne = false;
        let mut landed_again = false;
        let mut peak = 0.0f32;
        for _ in 0..900 {
            step_mob(&mut m, DT, &g, FAR, FAR, &mut rng);
            if m.on_ground {
                landed_again |= airborne;
            } else {
                airborne = true;
                peak = peak.max(floor_y - m.body.bottom());
            }
        }
        assert!(airborne, "a hopper that never hopped");
        assert!(
            peak > CELL_SIZE as f32,
            "the arc never cleared a cell: {peak}"
        );
        assert!(landed_again, "and it comes back down");
    }

    #[test]
    fn a_flyer_holds_its_altitude_and_never_touches_the_ground() {
        // No gravity is the whole brain: a bat 300px above a floor must still be
        // 300px above it after ten seconds of wandering.
        let g = world(64, 64, 60);
        let floor_y = 60.0 * CELL_SIZE as f32;
        let mut m = spawn("bat", 100.0, 100.0);
        let mut rng = MobRng::seeded(5);
        let start_y = m.body.y;

        for i in 0..900 {
            step_mob(&mut m, DT, &g, FAR, FAR, &mut rng);
            assert!(!m.on_ground, "a flyer reported footing at step {i}");
            assert!(
                m.body.bottom() < floor_y,
                "a flyer sank to the floor at step {i}"
            );
            assert!(
                !box_overlaps_solid(&g, m.body),
                "flew into rock at step {i}"
            );
        }
        // The wander is a bounded oscillation, not a drift: a brain that had
        // picked up gravity would be hundreds of px lower by now.
        assert!(
            (m.body.y - start_y).abs() < 120.0,
            "altitude drifted by {}",
            m.body.y - start_y
        );
    }

    #[test]
    fn a_burrower_surfaces_and_then_goes_back_under() {
        // With no target in reach there is no ambush, so it surfaces on the
        // `beatT <= -6` fallback rather than sitting as a permanently invisible
        // occupant of a pool slot — and then re-buries on its next beat.
        let g = world(128, 64, 40);
        let mut m = spawn("sandworm", 100.0, (40 + 2) as f32 * CELL_SIZE as f32);
        assert!(m.buried, "a burrower starts submerged");
        let mut rng = MobRng::seeded(13);

        let mut surfaced_at = None;
        let mut reburied_at = None;
        for i in 0..1800 {
            step_mob(&mut m, DT, &g, FAR, FAR, &mut rng);
            if surfaced_at.is_none() && !m.buried {
                surfaced_at = Some(i);
            }
            if surfaced_at.is_some() && reburied_at.is_none() && m.buried {
                reburied_at = Some(i);
            }
        }
        let up = surfaced_at.expect("a burrower that never came up");
        let down = reburied_at.expect("a burrower that never went back down");
        assert!(down > up, "re-buried before it surfaced");
        assert!(m.buried, "and it ends the run submerged");
    }

    #[test]
    fn a_crushed_creature_is_lifted_and_then_doomed() {
        // The escape hatch: a mob genuinely inside solid matter (a dune settled
        // on it) is lifted a cell at a time, and if there is nowhere to lift it
        // to it is marked for recycling rather than left vibrating in a rock.
        let mut g = CellGrid::new(64, 64);
        for cy in 0..64 {
            for cx in 0..64 {
                g.set(cx, cy, block::STONE);
            }
        }
        let mut m = spawn("grubling", 100.0, 100.0);
        let mut rng = MobRng::seeded(2);
        for _ in 0..(60 * 3) {
            step_mob(&mut m, DT, &g, FAR, FAR, &mut rng);
        }
        assert!(m.doomed, "solid rock in every direction must doom it");

        // And a creature with one clear cell above is lifted into it instead.
        let mut g2 = world(64, 64, 40);
        let mut m2 = spawn("grubling", 100.0, 40.0 * CELL_SIZE as f32);
        for cx in 0..64 {
            g2.set(cx, 40, block::STONE);
        }
        let before = m2.body.y;
        step_mob(&mut m2, DT, &g2, FAR, FAR, &mut rng);
        assert!(m2.body.y < before, "a buried creature must be lifted clear");
        assert!(!m2.doomed);
    }

    #[test]
    fn a_lava_bather_burns_unless_it_is_the_kind_that_lives_there() {
        let mut g = world(64, 64, 40);
        for cx in 0..64 {
            g.set(cx, 38, block::LAVA);
            g.set(cx, 39, block::LAVA);
        }
        let mut victim = spawn("grubling", 100.0, 38.0 * CELL_SIZE as f32);
        let full = victim.health;
        // The hazard probe is on its own quarter-second interval, so a couple of
        // seconds is several samples.
        for _ in 0..120 {
            apply_hazards(&mut victim, DT, &g);
        }
        assert!(victim.health < full, "lava must hurt a creature");

        // `lava_immune` is the blunt form: nothing the world is made of touches
        // it. The emberling is the creature that exists to say so.
        let mut native = spawn("emberling", 100.0, 38.0 * CELL_SIZE as f32);
        let native_full = native.health;
        for _ in 0..120 {
            apply_hazards(&mut native, DT, &g);
        }
        assert_eq!(native.health, native_full, "a fire creature does not burn");

        // The OTHER form: `immune_mask` is per-tag and is applied inside the
        // probe, so a creature that is fire-immune without being `lava_immune`
        // shrugs off lava and would still be hurt by acid.
        let mut moth = spawn("cinder_moth", 100.0, 38.0 * CELL_SIZE as f32);
        assert!(!moth.def.lava_immune, "the fixture wants the per-tag path");
        assert!(moth.def.immune_mask.contains(Dmg::FIRE));
        let moth_full = moth.health;
        for _ in 0..120 {
            apply_hazards(&mut moth, DT, &g);
        }
        assert_eq!(moth.health, moth_full, "fire immunity covers lava");
    }

    #[test]
    fn the_stream_replays_identically_for_a_seed() {
        // The whole reason the RNG is a value and not a global: two runs from one
        // seed must be the same run. Compared over a brain, not just over the
        // raw stream, so a draw that moved to a different point in the step is
        // caught too.
        let g = world(64, 64, 40);
        let floor_y = 40.0 * CELL_SIZE as f32;

        let run = || {
            let mut m = spawn("grubling", 100.0, floor_y - 10.0);
            let mut rng = MobRng::seeded(0xc0ff_ee11);
            let mut trace = Vec::new();
            for _ in 0..600 {
                step_mob(&mut m, DT, &g, FAR, FAR, &mut rng);
                trace.push((m.body.x.to_bits(), m.body.y.to_bits(), m.facing.to_bits()));
            }
            (trace, rng.state)
        };
        let (a, a_state) = run();
        let (b, b_state) = run();
        assert_eq!(a, b, "the same seed produced a different patrol");
        assert_eq!(a_state, b_state, "and left the stream in a different place");
    }

    #[test]
    fn surface_row_near_finds_the_boundary_and_refuses_unloaded_space() {
        let mut g = CellGrid::new(64, 64);
        for cx in 0..64 {
            for cy in 20..64 {
                g.set(cx, cy, block::STONE);
            }
        }
        assert_eq!(surface_row_near(&g, 10, 22, 10), Some(20));
        // Out of the search band entirely.
        assert_eq!(surface_row_near(&g, 10, 40, 5), None);
        // A column that is solid all the way up has no rock/air boundary; off
        // the window every cell reads solid, so there is none there either.
        assert_eq!(surface_row_near(&g, -5, 22, 10), None);
    }
}

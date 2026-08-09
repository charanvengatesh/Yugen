//! The mob system: spawning, population control, stepping, combat and
//! despawning. A host owns one of these and drives it exactly the way it drives
//! the particle system — one `update` in the play loop.
//!
//! ---- POPULATION ------------------------------------------------------------
//! The world is infinite and streams, so an unbounded creature list is a leak by
//! construction: walk east for two minutes and you would be paying to simulate
//! everything you ever passed. Three rules bound it, and the FIRST is structural
//! rather than policy:
//!
//!   1. The pool is a fixed array of [`MAX_MOBS`] instances allocated once.
//!      Spawning claims a dead slot; there is no path that grows it. The cap is
//!      therefore not something the code has to remember to enforce.
//!   2. Spawns land in the band of world just OUTSIDE the viewport (never in
//!      view, never in unloaded cells), on a timer, with rejection sampling.
//!   3. Anything that drifts past the despawn rectangle — or whose cell has left
//!      the streaming window — is recycled immediately.
//!
//! [`MAX_MOBS`] is 32. The reasoning: the streamed window is 352x256 cells and
//! the viewport at most 160x100 of them, so the spawn band covers roughly 40,000
//! cells; 32 creatures over that is about one per 1,200 cells, which reads as
//! inhabited without turning into a swarm. On cost, 32 mobs at 60Hz is ~4,000
//! cell reads per frame for collision.
//!
//! ---- COMBAT ----------------------------------------------------------------
//! The fence sits between the ATTACK and its EFFECT, and each side owns one.
//!
//! The player owns the attack: `hit_box` is the arc already grown by the held
//! weapon's reach and flipped by facing, `swing_id` increments once per swing,
//! `punching` is the damage WINDOW (not the pose), and `melee_damage` /
//! `melee_knockback` are the weapon's numbers. That has to live there — a sword
//! has reach and damage of its own, and this system has no idea what is held.
//!
//! This system owns what a hit DOES, because armour is the creature's property:
//! a mob whose box overlaps the arc, on a swing id it has not already been hit
//! by, takes `melee_damage` less its own `armor`. The reverse direction is
//! unchanged — a hostile mob overlapping the player subtracts from the player's
//! health on its own attack cooldown. Both emit events for the host to turn into
//! particles, flash and shake.
//!
//! [`MobSystem::hit_at`] is the same effect path reached from a point rather
//! than a box, so a player arrow lands with identical armour and knockback rules
//! to a sword.
//!
//! `armor` is a FLAT subtraction floored at 1 rather than a percentage. A
//! stoneback at armor 8 against a bare fist's 5 therefore takes the floor of 1
//! per swing: not invulnerable, just a creature that tells you to find a pick.
//! It applies to discrete hits only — a flat reduction against the per-tick
//! material hazard would make anything armoured immune to lava, which is what
//! `lava_immune` and `immune` exist to say deliberately.
//!
//! ---- PROJECTILES -----------------------------------------------------------
//! A `def.ranged` creature declares intent ([`Mob::wants_shot`]) and the system
//! owns the pool, for the same reason it owns the mob pool: a fixed array,
//! claimed and released, with no path that grows it. Shots integrate on the same
//! 60Hz sub-step as the creatures, and die on solid, on the player, or on a life
//! timer.
//!
//! # What the port changed
//!
//! - **The pool stayed a pool.** Not a `Vec<Option<Mob>>` and not an ECS: the
//!   slot semantics, the rolling cursor and the silently-saturating event and
//!   loot buffers are load-bearing, and the host is obliged to drain the latter
//!   two every frame.
//! - The whole of `draw`, `drawTell`, `drawGlow` and `drawShots` stayed behind —
//!   they were Canvas2D. Everything they read is published as an accessor, the
//!   way `player.rs` published what `Player.draw` read.
//! - `MobTarget` was a structural interface that `Player` satisfied for free. It
//!   is a [`MobTarget`] trait here, implemented for `Player` on this side of the
//!   boundary — the same arrow, with the adapter written down.
//! - `VIEW_W`/`VIEW_H` were module constants frozen from `window.innerWidth` at
//!   import time, so the spawn rectangles were constants too. A native window can
//!   be resized, so they are a [`SpawnRects`] value computed from a [`View`].
//! - Every loop over the pool is BY INDEX. That is not a translation slip: the
//!   body of each one calls back into `&mut self` (to fire, to bank loot, to
//!   push an event), which a live `&mut Mob` borrow would forbid.
//!
//! # A target the creatures may not perceive: a DELIBERATE DIVERGENCE
//!
//! [`MobTarget::targetable`] and everything that honours it are NEW. The
//! TypeScript's creative mode was an infinite build palette and nothing more —
//! its creatures chased, bit and shot a creative player exactly as they chased a
//! survival one — so there is no original behaviour being ported here and no
//! fixture that could catch a mistake in it. That is the reason it is called out
//! at the top of the file rather than left to be inferred from the code: this
//! tree is careful to distinguish what it INHERITED from what it DECIDED, and
//! `yugen_render::particles` sets the standard for how a decision gets said
//! out loud.
//!
//! What the divergence is, exactly: an untargetable player is invisible to the
//! creatures and to nothing else. They still spawn around it, wander, patrol,
//! burrow, take a sword to the face and die. See [`OUT_OF_RANGE_OFFSET`] for how
//! the invisibility is expressed to the brains without threading an `Option`
//! through all five of them.

use crate::config::{
    CELL_SIZE, CHUNK_CELLS, PLAYER_H, PLAYER_W, SEED, View, WINDOW_COLS, WINDOW_ROWS, WorldScale,
    cell_at,
};
use crate::entities::player::Player;
use crate::physics::collision::{Aabb, box_overlaps_solid, is_solid_cell};
use crate::sim::biomes::{BIOMES, biome_index_at};
use crate::sim::coords::WorldCell;
use crate::sim::grid::CellGrid;
use crate::sim::materials::{CellId, code_of};
use crate::sim::noise::Noise;
use crate::sim::worldgen::heightmap::Heightmap;
use crate::sim::worldgen::world_noise;

use super::brain::{Mob, MobRng, step_mob};
use super::defs::{
    Band, MOB_BANDS, MOB_DEFS, MOB_MAXDEPTH, MOB_MINDEPTH, MobBand, MobBrain, MobDef, MobPose,
    MobProjectile, VARIANT_COUNT, band_at_depth, band_bit,
};

/// Hard population cap. Also the pool length — see the header note.
pub const MAX_MOBS: usize = 32;

/// Mobs integrate at a fixed 60Hz (the player runs at 120Hz; mobs do not need
/// it).
const MOB_DT: f32 = 1.0 / 60.0;
/// Catch-up steps per frame, to avoid a spiral of death after a stall.
const MAX_MOB_STEPS: u32 = 3;

/// Seconds between spawn attempts.
const SPAWN_INTERVAL: f32 = 0.15;
/// Rejection-sampled candidate cells per attempt.
const SPAWN_TRIES: u32 = 12;
/// How far outside the viewport edge a spawn must be to be genuinely unseen.
const OFFSCREEN_MARGIN: f32 = 32.0;
/// Minimum gap between two freshly spawned creatures, world px.
const SPAWN_SEPARATION: f32 = 30.0;
/// Cells either side of the pack leader that a follower may be sampled at.
const PACK_SPREAD: f32 = 5.0;
/// Precomputed so the nocturnal test is a compare rather than a lookup.
const BAND_SURFACE: Band = band_bit(MobBand::Surface);

/// How far from the player the phantom target sits when the real one may not be
/// perceived ([`MobTarget::targetable`] is `false`). World px, on both axes.
///
/// [`step_mob`] takes the target's CENTRE and derives the `dx`/`dy` that drive
/// `decide` and every `near(aggro_px)` test in all five brains. Making that
/// target optional would mean an `Option<f32>` threaded through `step_mob`,
/// `decide`, `step_walker`, `step_hopper`, `step_flyer` and `step_burrower`, and
/// a new "no target" branch inside each — six new code paths, none of which the
/// game would run in the ninety-nine percent case, to express a state the brains
/// ALREADY HAVE A NAME FOR.
///
/// Because they do. Every creature in the game spends most of its life with the
/// player outside its aggro box; that is not an edge case being simulated, it is
/// the default condition of the world, exercised by every frame of
/// `two_systems_on_one_seed_spawn_the_same_world` and by every creature that
/// spawns off-screen and is culled before you ever see it. Putting the target
/// out there is not a sentinel smuggled into the arithmetic — it is the honest
/// input for "this creature has no one to chase".
///
/// It is an OFFSET from the player and not an absolute coordinate, because the
/// world is infinite: an absolute 1e6 would sit on top of a player who had
/// walked a million pixels east and turn the whole thing into an aggro magnet.
///
/// The value is 1e6 px, and both bounds matter:
///
/// - **Far enough.** The largest `aggroPx` in `content/mobs/` is 210, the
///   largest `aggroYPx` 140, the largest `ranged.range` 230. Every live creature
///   is inside the despawn rectangle, a couple of thousand px across at most, so
///   the smallest `dx` any brain can see here is ~1e6 — four orders of magnitude
///   past the widest perception in the game. Content would have to grow an aggro
///   radius of two hundred screen-widths to close it.
/// - **Near enough.** It is FINITE, so no `NaN` or infinity can enter the
///   arithmetic: the sharpest thing done to `dx` is `dx * dx` in the flyer's
///   steering, and 1e12 is twenty-six orders of magnitude inside `f32::MAX`.
///   Adding it to the player's own position survives `f32` rounding until the
///   player is past ~8e12 px from the origin, and the game's px coordinates stop
///   resolving whole pixels at 1.7e7 — so the offset outlives the coordinate
///   system it is added to by six orders of magnitude.
const OUT_OF_RANGE_OFFSET: f32 = 1.0e6;

/// Hard cap on projectiles in flight. Also the pool length — same rule as
/// [`MAX_MOBS`].
const MAX_SHOTS: usize = 24;
/// Hard cap on undrained loot entries.
const MAX_LOOT: usize = 32;
/// Hard cap on undrained events.
const MAX_EVENTS: usize = 48;

/// Cells scanned when a lava-dweller checks it is somewhere it belongs.
const LAVA_PROBE: [(i32, i32); 20] = [
    (4, 0),
    (-4, 0),
    (0, 4),
    (0, -4),
    (3, 3),
    (-3, 3),
    (3, -3),
    (-3, -3),
    (9, 0),
    (-9, 0),
    (0, 9),
    (0, -9),
    (7, 7),
    (-7, 7),
    (7, -7),
    (-7, -7),
    (14, 0),
    (-14, 0),
    (0, 14),
    (0, -14),
];

// ---------------------------------------------------------------------------
// The player, as the mobs need it
// ---------------------------------------------------------------------------

/// What the mob system needs to know about the player.
///
/// The melee fields all live on the PLAYER side, which is where they belong once
/// weapons are real: this system used to compute the swing itself — its own box,
/// its own id, a flat `PUNCH_DAMAGE` — which was fine while a punch was the only
/// attack in the game and wrong the moment a held sword had reach and damage of
/// its own. The player owns the weapon, so the player owns the arc; this side
/// only asks "did a new swing land on me, and for how much".
///
/// A trait rather than a concrete `&mut Player` for the reason `Projectiles` is
/// one: it was a structural interface in the TypeScript, satisfied by `Player`
/// with no import and no coupling, and fakeable by a test harness. Rust is
/// nominal, so the adapter is written down — three lines, on this side of the
/// fence.
pub trait MobTarget {
    /// Top-left of the collision box, world px.
    fn x(&self) -> f32;
    /// Top-left of the collision box, world px.
    fn y(&self) -> f32;
    /// Hit points. Zero or below means dead and un-hittable.
    fn health(&self) -> f32;
    /// Take `amount` hit points off.
    fn take_damage(&mut self, amount: f32);
    /// Whether the creatures may perceive and attack this target at all.
    ///
    /// Defaulted to `true`, which is the answer every target gave before this
    /// method existed, so adding it changed no implementor and no behaviour.
    /// That is deliberate and not merely convenient: a trait that is also a
    /// TEST HARNESS SEAM must stay cheap to satisfy, and a required method here
    /// would have made every fake in every test file answer a question it does
    /// not care about.
    ///
    /// It is a question about PERCEPTION, not about damage. A target that says
    /// `false` is not merely armoured — it is not there. See
    /// [`MobSystem::update`] for the three places that follow from it.
    fn targetable(&self) -> bool {
        true
    }
    /// Which way the body is pointing: exactly `1.0` or `-1.0`.
    fn facing(&self) -> f32;
    /// The damage WINDOW is open — not "the punch pose is playing".
    fn punching(&self) -> bool;
    /// One id per swing. Compared against `Mob::last_punch_id` to land one hit
    /// each.
    fn swing_id(&self) -> u32;
    /// The arc, already grown by the weapon's reach and flipped by facing.
    fn hit_box(&self) -> Aabb;
    /// Hit points one connecting swing is worth, before the target's armour.
    fn melee_damage(&self) -> f32;
    /// Impulse a connecting swing puts into what it hits.
    fn melee_knockback(&self) -> f32;
}

impl MobTarget for Player {
    fn x(&self) -> f32 {
        self.x
    }
    fn y(&self) -> f32 {
        self.y
    }
    fn health(&self) -> f32 {
        self.health
    }
    /// The belt to `targetable`'s braces. [`MobSystem`] already refuses to reach
    /// this for an untargetable player, so the guard is dead code today — and it
    /// is here anyway, because the next caller of `take_damage` will be written
    /// by someone who read the trait and not this file, and "the body cannot be
    /// hurt" should be true of the body rather than of one of its callers.
    fn take_damage(&mut self, amount: f32) {
        if self.untouchable {
            return;
        }
        // `1.0.max(damage - armour)`, which is the rule creatures' own `armor`
        // uses two hundred lines up — deliberately the same, so a player and a
        // mob wearing the same number are equally hard to hurt and the item
        // table can be read against the mob table without conversion.
        //
        // The floor matters more than the subtraction. Without it a player in
        // enough armour is not merely tough, they are unkillable by anything
        // below their number, and a whole band of the world stops being a
        // threat rather than becoming an easy one.
        self.health -= 1.0f32.max(amount - self.armour);
    }
    fn targetable(&self) -> bool {
        !self.untouchable
    }
    fn facing(&self) -> f32 {
        self.facing
    }
    fn punching(&self) -> bool {
        Player::punching(self)
    }
    fn swing_id(&self) -> u32 {
        Player::swing_id(self)
    }
    fn hit_box(&self) -> Aabb {
        Player::hit_box(self)
    }
    fn melee_damage(&self) -> f32 {
        Player::melee_damage(self)
    }
    fn melee_knockback(&self) -> f32 {
        Player::melee_knockback(self)
    }
}

// ---------------------------------------------------------------------------
// Spawn geometry
// ---------------------------------------------------------------------------

/// The streaming window's half-width, shrunk by the recenter hysteresis (a full
/// chunk of drift) plus a margin, so a spawn candidate is inside the loaded
/// window even at the worst-case moment just before the window shifts.
const SAFE_HX: f32 = ((WINDOW_COLS * CELL_SIZE) / 2 - (CHUNK_CELLS + 8) * CELL_SIZE) as f32;
/// The same, vertically.
const SAFE_HY: f32 = ((WINDOW_ROWS * CELL_SIZE) / 2 - (CHUNK_CELLS + 8) * CELL_SIZE) as f32;

/// Spawn/despawn geometry, in world px measured from the player.
///
/// Four nested rectangles, from the inside out: KEEP is the viewport plus a
/// margin and nothing may spawn inside it; SPAWN is the band candidates are
/// sampled from; DESPAWN is where anything that drifts out is recycled; SAFE is
/// the clamp that keeps SPAWN inside the streaming window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpawnRects {
    /// Half-width of the rectangle a spawn may never land inside.
    pub keep_hx: f32,
    /// Half-height of the same.
    pub keep_hy: f32,
    /// Half-width of the band candidates are sampled from.
    pub spawn_hx: f32,
    /// Half-height of the same.
    pub spawn_hy: f32,
    /// Half-width past which a live creature is recycled.
    pub despawn_hx: f32,
    /// Half-height past which a live creature is recycled.
    pub despawn_hy: f32,
}

impl SpawnRects {
    /// Derive the four rectangles from a viewport.
    ///
    /// In the TypeScript these were module constants, because `VIEW_W`/`VIEW_H`
    /// were frozen from `window.innerWidth` at import time. A native window can
    /// be resized, so the same arithmetic is a function of the surface.
    pub fn for_view(view: View) -> SpawnRects {
        let view_w = view.w as f32;
        let view_h = view.h as f32;
        let keep_hx = view_w / 2.0 + OFFSCREEN_MARGIN;
        let keep_hy = view_h / 2.0 + OFFSCREEN_MARGIN;
        let spawn_hx = (keep_hx + 40.0).max((view_w / 2.0 + 260.0).min(SAFE_HX));
        let spawn_hy = (keep_hy + 40.0).max((view_h / 2.0 + 180.0).min(SAFE_HY));
        SpawnRects {
            keep_hx,
            keep_hy,
            spawn_hx,
            spawn_hy,
            despawn_hx: spawn_hx + 160.0,
            despawn_hy: spawn_hy + 120.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Projectiles
// ---------------------------------------------------------------------------

/// Per-kind projectile behaviour.
///
/// `speed` is authored in the player's reference frame like every other velocity
/// in the bestiary and is multiplied by the SHOOTER's `body_scale`, so a big
/// creature throws proportionally harder without the table having to know how
/// big anything is. `life` is a duration and is not scaled. `chill` is seconds
/// of slow applied on hit, 0 for everything else.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectileSpec {
    /// Muzzle speed in the player's reference frame, px/s before body scaling.
    pub speed: f32,
    /// Half-extent drawn, in px.
    pub r_px: f32,
    /// Seconds it lives.
    pub life: f32,
    /// Colour, for the mote and its glow.
    pub rgb: [u8; 3],
    /// Seconds of slow applied on hit. See [`MobEventKind::PlayerChill`].
    pub chill: f32,
    /// Additive self-illumination, 0..1.
    pub glow: f32,
}

impl ProjectileSpec {
    /// The table, as a `match` on the generated enum rather than a record keyed
    /// by it — so a new projectile kind cannot be added without an answer here.
    pub const fn of(kind: MobProjectile) -> ProjectileSpec {
        match kind {
            MobProjectile::IceShard => ProjectileSpec {
                speed: 520.0,
                r_px: 2.0,
                life: 1.6,
                rgb: [159, 232, 255],
                chill: 1.6,
                glow: 0.5,
            },
            MobProjectile::Ember => ProjectileSpec {
                speed: 430.0,
                r_px: 2.0,
                life: 1.4,
                rgb: [255, 138, 60],
                chill: 0.0,
                glow: 0.8,
            },
            MobProjectile::Spore => ProjectileSpec {
                speed: 330.0,
                r_px: 2.0,
                life: 2.2,
                rgb: [200, 255, 122],
                chill: 0.0,
                glow: 0.35,
            },
            MobProjectile::Stone => ProjectileSpec {
                speed: 470.0,
                r_px: 2.0,
                life: 1.5,
                rgb: [150, 146, 140],
                chill: 0.0,
                glow: 0.0,
            },
        }
    }
}

/// One projectile. Pooled and mutated in place; never allocated after
/// construction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Shot {
    /// This slot holds a live projectile.
    pub active: bool,
    /// Position, world px (a point, not a box).
    pub x: f32,
    /// Position, world px.
    pub y: f32,
    /// Velocity, px/s.
    pub vx: f32,
    /// Velocity, px/s.
    pub vy: f32,
    /// Seconds left before it expires.
    pub life: f32,
    /// Health taken on a hit.
    pub damage: f32,
    /// Half-extent drawn, in px.
    pub r_px: f32,
    /// Seconds of slow applied on hit.
    pub chill: f32,
    /// Additive self-illumination, 0..1.
    pub glow: f32,
    /// Colour.
    pub rgb: [u8; 3],
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// What happened, for a host to turn into juice.
///
/// [`MobEventKind::PlayerChill`] is the hook for the ice-throwing sentinel. The
/// slow itself belongs to the player's movement model, which this system
/// deliberately does not reach into (see the COMBAT note above), so the shot
/// reports the DURATION and whoever owns the `Player` decides what a chill does.
/// A host that ignores kinds it does not handle pays nothing for it until then.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MobEventKind {
    /// The player took damage.
    PlayerHit,
    /// A creature took damage and survived.
    MobHurt,
    /// A creature was killed.
    MobDie,
    /// The player was chilled; `power` is the duration in seconds.
    PlayerChill,
}

/// A one-shot report. Pooled — never allocated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MobEvent {
    /// What happened.
    pub kind: MobEventKind,
    /// Where, world px.
    pub x: f32,
    /// Where, world px.
    pub y: f32,
    /// Particle colour — the creature's blood, or the projectile's tint.
    pub rgb: [u8; 3],
    /// Damage dealt, impact strength for a death, or seconds for a chill.
    pub power: f32,
}

/// One rolled drop from a kill.
///
/// Reported rather than spawned: there is no world item entity yet, and
/// inventing one here would put the item model in the mob system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MobLoot {
    /// Item id as authored in `content/mobs/`. May not exist in the registry
    /// yet.
    pub item: &'static str,
    /// How many.
    pub count: f32,
    /// Where it fell, world px.
    pub x: f32,
    /// Where it fell, world px.
    pub y: f32,
}

/// `Math.round`, which is NOT `f32::round`.
///
/// JavaScript rounds a half UP (`-0.5` is `-0`); Rust rounds a half AWAY from
/// zero (`-0.5` is `-1`). The two disagree only on an exact half, which a
/// uniform draw hits with probability zero — but "probability zero" is not
/// "never" over a stream that runs for hours, and this is a port.
#[inline]
fn js_round(v: f32) -> f32 {
    (v + 0.5).floor()
}

/// Do two boxes overlap?
#[inline]
fn overlaps(a: Aabb, b: Aabb) -> bool {
    a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y
}

// ---------------------------------------------------------------------------
// The system
// ---------------------------------------------------------------------------

/// The creatures, their projectiles, and everything that happens to them.
pub struct MobSystem {
    /// Fixed-length pool. Length never changes; `Mob::active` marks the live
    /// ones.
    pool: Vec<Mob>,
    /// Fixed-length projectile pool, same rule.
    shots: Vec<Shot>,
    noise: Noise,
    /// The surface-row memo the depth band is measured against. Owned rather
    /// than borrowed: it is a cache, and sharing one would make the mob system
    /// a writer of somebody else's scratch.
    heightmap: Heightmap,
    /// The world scale the terrain under this population was generated at — the
    /// depth bands below are authored in legacy cells and have to be compared
    /// against a legacy depth.
    scale: WorldScale,
    lava: CellId,

    /// Live creature count, maintained incrementally.
    ///
    /// It used to be a getter that walked the whole pool, and `update` consulted
    /// it every spawn tick — a pointless O(MAX_MOBS) scan several times a second
    /// for a number three call sites already know how to keep exact.
    live: usize,
    /// Rolling cursor for [`MobSystem::free_slot`].
    ///
    /// Claims and releases are uniformly spread over the pool, so starting the
    /// search where the last one finished turns a repeated O(n) scan from the
    /// front into an amortised O(1) step.
    slot_cursor: usize,

    accumulator: f32,
    spawn_timer: f32,
    /// Player centre at the moment of the current spawn attempt.
    spawn_px: f32,
    spawn_py: f32,

    events: Vec<MobEvent>,
    event_count: usize,
    loot: Vec<MobLoot>,
    loot_count: usize,
    xp_banked: i32,
    value_banked: i32,

    /// Scratch for weighted spawn selection — reused, never reallocated.
    cand_idx: Vec<usize>,
    cand_w: Vec<f32>,

    /// The creatures' own random source. See [`MobRng`].
    rng: MobRng,
    rects: SpawnRects,
}

impl MobSystem {
    /// An empty population, sized to a viewport.
    pub fn new(view: View) -> MobSystem {
        MobSystem {
            pool: (0..MAX_MOBS).map(|_| Mob::new()).collect(),
            shots: vec![Shot::default(); MAX_SHOTS],
            noise: world_noise(SEED),
            heightmap: Heightmap::new(),
            scale: WorldScale::LIVE,
            lava: code_of("lava"),
            live: 0,
            slot_cursor: 0,
            accumulator: 0.0,
            spawn_timer: 0.0,
            spawn_px: 0.0,
            spawn_py: 0.0,
            events: vec![
                MobEvent {
                    kind: MobEventKind::MobHurt,
                    x: 0.0,
                    y: 0.0,
                    rgb: [0, 0, 0],
                    power: 0.0,
                };
                MAX_EVENTS
            ],
            event_count: 0,
            loot: vec![
                MobLoot {
                    item: "",
                    count: 0.0,
                    x: 0.0,
                    y: 0.0,
                };
                MAX_LOOT
            ],
            loot_count: 0,
            xp_banked: 0,
            value_banked: 0,
            cand_idx: vec![0; MOB_DEFS.len()],
            cand_w: vec![0.0; MOB_DEFS.len()],
            rng: MobRng::seeded(SEED ^ 0x5bf0_3635),
            rects: SpawnRects::for_view(view),
        }
    }

    /// Re-derive the spawn rectangles after the window was resized.
    pub fn set_view(&mut self, view: View) {
        self.rects = SpawnRects::for_view(view);
    }

    /// The spawn/despawn geometry in force.
    #[inline]
    pub fn rects(&self) -> SpawnRects {
        self.rects
    }

    /// Live creature count.
    #[inline]
    pub fn count(&self) -> usize {
        self.live
    }

    /// The whole pool, live and dead — for the renderer, the HUD and the test
    /// harness. Skip anything whose `active` is false.
    #[inline]
    pub fn mobs(&self) -> &[Mob] {
        &self.pool
    }

    /// The projectile pool, same rule.
    #[inline]
    pub fn shots(&self) -> &[Shot] {
        &self.shots
    }

    /// Events raised since the last [`MobSystem::clear_events`].
    ///
    /// # This returns the VALID PREFIX, and that is the whole point
    ///
    /// It used to hand back the fixed backing store — all [`MAX_EVENTS`] of it —
    /// with a separate `event_count()` telling the caller how much of it meant
    /// anything. That is a C signature, a pointer and a length, and Rust has one
    /// type that carries both. The host drained the whole buffer, so ~14 phantom
    /// `MobHurt` events arrived every frame at (0, 0), each worth 0.08 trauma
    /// against a decay of 4/s: **the screen shake was pinned at maximum from the
    /// first frame of the game.** 896 tests and a purpose-built image gate were
    /// green, because a shaking camera still renders a varied, correctly-lit
    /// frame. [`MobSystem::loot`] had the identical bug and was surviving on a
    /// `count > 0` guard downstream.
    ///
    /// Slicing here does not merely catch that mistake — it makes it
    /// unrepresentable. There is no longer an invalid tail to hand anybody.
    #[inline]
    pub fn events(&self) -> &[MobEvent] {
        &self.events[..self.event_count]
    }

    /// How many events are valid. Identical to `events().len()`; kept because it
    /// reads better at a call site that only wants to know whether any arrived.
    #[inline]
    pub fn event_count(&self) -> usize {
        self.event_count
    }

    /// Drop everything reported so far. The host is obliged to call this every
    /// frame: the buffer SATURATES rather than growing, so an undrained system
    /// silently stops reporting instead of leaking.
    pub fn clear_events(&mut self) {
        self.event_count = 0;
    }

    /// Drops rolled since the last [`MobSystem::clear_loot`].
    ///
    /// The valid prefix, for the reason [`MobSystem::events`] spells out.
    #[inline]
    pub fn loot(&self) -> &[MobLoot] {
        &self.loot[..self.loot_count]
    }

    /// How many drops are valid. Identical to `loot().len()`.
    #[inline]
    pub fn loot_count(&self) -> usize {
        self.loot_count
    }

    /// Experience accumulated from kills, drained by the caller.
    #[inline]
    pub fn xp_banked(&self) -> i32 {
        self.xp_banked
    }

    /// Currency accumulated from kills, drained by the caller.
    #[inline]
    pub fn value_banked(&self) -> i32 {
        self.value_banked
    }

    /// Drop the rolled loot and the banked rewards. Same obligation as
    /// [`MobSystem::clear_events`].
    pub fn clear_loot(&mut self) {
        self.loot_count = 0;
        self.xp_banked = 0;
        self.value_banked = 0;
    }

    /// Wipe the population (level reload / respawn).
    pub fn clear(&mut self) {
        for m in &mut self.pool {
            m.active = false;
        }
        for s in &mut self.shots {
            s.active = false;
        }
        self.live = 0;
        self.event_count = 0;
        self.loot_count = 0;
        self.xp_banked = 0;
        self.value_banked = 0;
        self.accumulator = 0.0;
        self.spawn_timer = 0.0;
    }

    /// One frame.
    ///
    /// `dt` is real seconds; the creatures themselves integrate on a fixed 60Hz
    /// sub-step so their collision stays stable regardless of frame rate, capped
    /// at `MAX_MOB_STEPS` to avoid a spiral of death after a stall. `day` is the
    /// day/night factor (1 = full daylight) and only biases which species are
    /// eligible to spawn.
    ///
    /// # An untargetable player
    ///
    /// When [`MobTarget::targetable`] is `false` this frame, three things stop
    /// and nothing else does:
    ///
    ///   1. the brains are fed a target [`OUT_OF_RANGE_OFFSET`] away, so
    ///      `decide` never alerts, no flyer dives, no worm ambushes and no
    ///      shooter finds a firing solution;
    ///   2. the half of [`MobSystem::resolve_combat`] that resolves AGAINST the
    ///      player — contact damage — is skipped. The other half, the player's
    ///      own swing landing on a creature, runs untouched;
    ///   3. shots already in flight pass through the body instead of hitting it.
    ///
    /// Spawning, culling and the population budget all still read the player's
    /// REAL box, because the world is still built around where the player is. It
    /// simply stops noticing them. See the header for why that is a deliberate
    /// divergence and not a port.
    pub fn update(&mut self, dt: f32, grid: &CellGrid, player: &mut dyn MobTarget, day: f32) {
        // The player's box is snapshot at the top of the FRAME, not per
        // sub-step: the creatures move several times inside one `update` and the
        // player does not, so re-reading it would be reading the same numbers
        // three times.
        let player_box = Aabb::new(player.x(), player.y(), PLAYER_W, PLAYER_H);
        // Snapshot for the same reason, and it matters more here: a target that
        // flipped mid-frame could aggro a creature on sub-step one and disarm
        // the combat resolver on sub-step two, which is a state no reader of
        // this loop would predict.
        let targetable = player.targetable();
        // What the BRAINS are told the target is — the real centre, or a point
        // no aggro radius in the game can reach. The spawner and the culler
        // below deliberately do NOT use these.
        let (tx, ty) = if targetable {
            (player.x() + PLAYER_W * 0.5, player.y() + PLAYER_H * 0.5)
        } else {
            (
                player.x() + OUT_OF_RANGE_OFFSET,
                player.y() + OUT_OF_RANGE_OFFSET,
            )
        };

        self.accumulator += dt;
        let mut steps = 0;
        while self.accumulator >= MOB_DT && steps < MAX_MOB_STEPS {
            self.accumulator -= MOB_DT;
            steps += 1;
            for i in 0..self.pool.len() {
                if !self.pool[i].active {
                    continue;
                }
                step_mob(&mut self.pool[i], MOB_DT, grid, tx, ty, &mut self.rng);
                if self.pool[i].wants_shot {
                    self.pool[i].wants_shot = false;
                    self.fire(i, tx, ty);
                }
                self.resolve_combat(i, player, player_box, targetable);
            }
            self.step_shots(MOB_DT, grid, player, player_box, targetable);
        }
        if self.accumulator > MOB_DT * MAX_MOB_STEPS as f32 {
            self.accumulator = 0.0;
        }

        self.cull(grid, player_box);
        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 {
            self.spawn_timer = SPAWN_INTERVAL;
            if self.live < MAX_MOBS {
                self.try_spawn(grid, player_box, day);
            }
        }
    }

    // --- projectiles ---------------------------------------------------------

    /// Launch a shot from the creature's centre toward the player's centre.
    fn fire(&mut self, i: usize, tx: f32, ty: f32) {
        let Some(r) = self.pool[i].def.ranged else {
            return;
        };
        let Some(si) = self.shots.iter().position(|s| !s.active) else {
            // A full pool silently drops the shot. The cooldown has already been
            // spent, so a saturated screen thins the volume of fire instead of
            // queueing it up.
            return;
        };

        let p = ProjectileSpec::of(r.projectile);
        let (cx, cy) = self.pool[i].center();
        let ax = tx - cx;
        let ay = ty - cy;
        let len = 1.0f32.max((ax * ax + ay * ay).sqrt());
        let speed = p.speed * self.pool[i].def.body_scale;

        self.shots[si] = Shot {
            active: true,
            x: cx,
            y: cy,
            vx: (ax / len) * speed,
            vy: (ay / len) * speed,
            life: p.life,
            damage: r.damage,
            r_px: p.r_px,
            chill: p.chill,
            glow: p.glow,
            rgb: p.rgb,
        };
    }

    /// Integrate every live shot.
    ///
    /// Point-vs-cell against the world and point-vs-box against the player: a
    /// projectile is small enough that a swept AABB would only buy accuracy
    /// nobody can see at these speeds and this cell size.
    ///
    /// `targetable` gates the player test only. A shot at an untargetable player
    /// still flies, still dies on solid and still expires on its life timer — it
    /// simply passes through the body. That is the right behaviour for the frame
    /// creative is switched ON: the volley already in the air was aimed at a
    /// target that was fair game when it left the barrel, and deleting it would
    /// make the toggle a screen-clear.
    fn step_shots(
        &mut self,
        dt: f32,
        grid: &CellGrid,
        player: &mut dyn MobTarget,
        player_box: Aabb,
        targetable: bool,
    ) {
        for i in 0..self.shots.len() {
            if !self.shots[i].active {
                continue;
            }
            let s = &mut self.shots[i];
            s.life -= dt;
            s.x += s.vx * dt;
            s.y += s.vy * dt;

            if s.life <= 0.0 {
                s.active = false;
                continue;
            }
            // Unloaded cells read as solid, so this also stops a shot escaping
            // the streaming window — the same guarantee the mobs rely on.
            if is_solid_cell(grid, cell_at(s.x), cell_at(s.y)) {
                s.active = false;
                continue;
            }
            let (x, y, rgb, damage, chill) = (s.x, s.y, s.rgb, s.damage, s.chill);
            if targetable
                && player.health() > 0.0
                && x >= player_box.x
                && x <= player_box.x + PLAYER_W
                && y >= player_box.y
                && y <= player_box.y + PLAYER_H
            {
                s.active = false;
                player.take_damage(damage);
                self.push_rgb(MobEventKind::PlayerHit, x, y, rgb, damage);
                if chill > 0.0 {
                    self.push_rgb(MobEventKind::PlayerChill, x, y, rgb, chill);
                }
            }
        }
    }

    // --- combat --------------------------------------------------------------

    /// `targetable` gates ONE of the two directions. The player's swing always
    /// resolves — a creature has to be killable by a player the creatures cannot
    /// see, or creative would be a mode in which the world froze rather than one
    /// in which it ignored you — and only the contact damage coming back is
    /// skipped. An early return for `!targetable` would have quietly taken both.
    fn resolve_combat(
        &mut self,
        i: usize,
        player: &mut dyn MobTarget,
        player_box: Aabb,
        targetable: bool,
    ) {
        if self.pool[i].buried {
            return;
        }
        let d = self.pool[i].def;

        // Player's swing. The ARC and the DAMAGE are the player's — a held sword
        // has its own reach and its own numbers — but what a hit does to a
        // creature is still decided here, because armour is the creature's
        // property.
        if player.punching()
            && self.pool[i].last_punch_id != Some(player.swing_id())
            && overlaps(player.hit_box(), self.pool[i].body)
        {
            self.pool[i].last_punch_id = Some(player.swing_id());
            // Flat armour, floored at 1: heavy plate makes a fist nearly useless
            // without ever making the creature unkillable. See the header note.
            let dmg = 1.0f32.max(player.melee_damage() - d.armor);
            // Armour resists being MOVED as well as being hurt, on the same flat
            // curve, so a stoneback does not get punted around like a critter.
            // The creature's own `knockback` is a floor, not the value: a heavy
            // weapon should throw a critter further than a fist does, but no
            // weapon should make a bloat easier to shove than its own mass
            // allows.
            let kb = d.knockback.max(player.melee_knockback()) / (1.0 + d.armor * 0.12);
            let m = &mut self.pool[i];
            m.health -= dmg;
            m.flash = 1.0;
            m.alerted = true;
            m.vx = player.facing() * kb;
            m.vy = -kb * 0.45;
            m.on_ground = false;
            if m.health <= 0.0 {
                self.kill(i);
                return; // a corpse does not get to land a contact hit on the way out
            }
            let (cx, cy) = self.pool[i].center();
            self.push(MobEventKind::MobHurt, cx, cy, d, dmg);
        }

        // Contact damage, on the creature's own cooldown so a mob standing on
        // the player drains health at a readable rate instead of every step.
        //
        // `targetable` is tested FIRST, so an untargetable player does not even
        // spend the creature's `attack_cd`: nothing happened, so nothing should
        // have been consumed by it.
        if targetable
            && d.contact_damage > 0.0
            && self.pool[i].attack_cd <= 0.0
            && player.health() > 0.0
            && overlaps(player_box, self.pool[i].body)
        {
            let (cx, _) = self.pool[i].center();
            let m = &mut self.pool[i];
            m.attack_cd = d.attack_cooldown;
            let away = if cx < player_box.x + PLAYER_W * 0.5 {
                -1.0
            } else {
                1.0
            };
            m.vx = away * d.knockback * 0.6;
            player.take_damage(d.contact_damage);
            self.push(
                MobEventKind::PlayerHit,
                player_box.x + PLAYER_W * 0.5,
                player_box.y + PLAYER_H * 0.5,
                d,
                d.contact_damage,
            );
        }
    }

    /// Damage whatever live creature contains a point. Returns true if something
    /// was hit, which is what tells a projectile to die.
    ///
    /// This is the melee effect path reached from a point instead of a box, and
    /// it deliberately shares every rule with it: flat armour floored at 1,
    /// knockback divided by the same armour curve, the same `MobHurt` /
    /// `MobDie` events. An arrow that ignored armour would make the bow strictly
    /// better than any sword against exactly the creatures armour exists to
    /// gate.
    ///
    /// `dir_x`/`dir_y` are the projectile's direction, so knockback follows the
    /// shot rather than the shooter's facing — the one place this differs from a
    /// swing, and the difference is the point of shooting from range.
    ///
    /// Buried creatures are not hittable, matching the swing path: a burrower
    /// inside rock is not a target until it breaches.
    pub fn hit_at(
        &mut self,
        x: f32,
        y: f32,
        damage: f32,
        knockback: f32,
        dir_x: f32,
        dir_y: f32,
    ) -> bool {
        for i in 0..MAX_MOBS {
            let m = &self.pool[i];
            if !m.active || m.buried {
                continue;
            }
            let b = m.body;
            if x < b.x || x > b.x + b.w || y < b.y || y > b.y + b.h {
                continue;
            }

            let d = m.def;
            let dmg = 1.0f32.max(damage - d.armor);
            let kb = knockback / (1.0 + d.armor * 0.12);
            let m = &mut self.pool[i];
            m.health -= dmg;
            m.flash = 1.0;
            m.alerted = true;
            m.vx = dir_x * kb;
            m.vy = dir_y * kb - kb * 0.25; // a touch of lift, as a swing gives
            m.on_ground = false;
            if m.health <= 0.0 {
                self.kill(i);
                return true;
            }
            self.push(
                MobEventKind::MobHurt,
                b.x + b.w * 0.5,
                b.y + b.h * 0.5,
                d,
                dmg,
            );
            return true;
        }
        false
    }

    fn kill(&mut self, i: usize) {
        let d = self.pool[i].def;
        let (cx, cy) = self.pool[i].center();
        self.push(MobEventKind::MobDie, cx, cy, d, d.max_health);
        self.xp_banked += d.xp;
        self.value_banked += d.value;
        // Every entry rolls independently, so a creature can drop all of its
        // table or none of it — a single weighted pick would make the rare
        // entries feel like they were competing with the guaranteed ones.
        for drop in d.drops {
            if drop.chance < 1.0 && self.rng.rand() >= drop.chance {
                continue;
            }
            if self.loot_count >= MAX_LOOT {
                break;
            }
            let lo = drop.count[0];
            let hi = drop.count[1];
            let count = if hi <= lo {
                lo
            } else {
                lo + (self.rng.rand() * (hi - lo + 1.0)) as i32 as f32
            };
            self.loot[self.loot_count] = MobLoot {
                item: drop.item,
                count,
                x: cx,
                y: cy,
            };
            self.loot_count += 1;
        }
        self.retire(i);
    }

    /// The one place a live creature goes back to the pool. Keeps `live` exact.
    fn retire(&mut self, i: usize) {
        if !self.pool[i].active {
            return;
        }
        self.pool[i].active = false;
        self.live -= 1;
    }

    fn push(&mut self, kind: MobEventKind, x: f32, y: f32, d: &MobDef, power: f32) {
        self.push_rgb(kind, x, y, d.blood, power);
    }

    /// Append a report, or silently drop it if nobody has drained.
    ///
    /// SATURATES rather than evicting the oldest, which is the opposite of what
    /// `Player::emit` does and is deliberate: a player emits a handful of events
    /// a second and the newest ones are the interesting ones, whereas 32
    /// creatures in a firefight emit in bursts and dropping the START of a burst
    /// would lose the kill that caused it.
    fn push_rgb(&mut self, kind: MobEventKind, x: f32, y: f32, rgb: [u8; 3], power: f32) {
        if self.event_count >= MAX_EVENTS {
            return;
        }
        self.events[self.event_count] = MobEvent {
            kind,
            x,
            y,
            rgb,
            power,
        };
        self.event_count += 1;
    }

    // --- population ----------------------------------------------------------

    /// Recycle anything that is too far away, has left the streaming window, or
    /// has been crushed / killed by the world.
    ///
    /// The window check is the important one: `CellGrid::is_loaded_world` is the
    /// authority on what actually exists, and a mob outside it would be
    /// colliding against "solid" fail-safe cells.
    fn cull(&mut self, grid: &CellGrid, player_box: Aabb) {
        let px = player_box.x + PLAYER_W * 0.5;
        let py = player_box.y + PLAYER_H * 0.5;
        for i in 0..self.pool.len() {
            if !self.pool[i].active {
                continue;
            }
            if self.pool[i].health <= 0.0 {
                self.kill(i);
                continue;
            }
            if self.pool[i].doomed {
                self.retire(i);
                continue;
            }
            let (cx, cy) = self.pool[i].center();
            if (cx - px).abs() > self.rects.despawn_hx || (cy - py).abs() > self.rects.despawn_hy {
                self.retire(i);
                continue;
            }
            if !grid.is_loaded_world(WorldCell::new(cell_at(cx), cell_at(cy))) {
                self.retire(i);
            }
        }
    }

    /// One spawn attempt: up to `SPAWN_TRIES` rejection-sampled candidate cells.
    fn try_spawn(&mut self, grid: &CellGrid, player_box: Aabb, day: f32) {
        let px = player_box.x + PLAYER_W * 0.5;
        let py = player_box.y + PLAYER_H * 0.5;
        self.spawn_px = px;
        self.spawn_py = py;

        for _ in 0..SPAWN_TRIES {
            let ox = self
                .rng
                .rand_range(-self.rects.spawn_hx, self.rects.spawn_hx);
            let oy = self
                .rng
                .rand_range(-self.rects.spawn_hy, self.rects.spawn_hy);
            // Reject the viewport itself: creatures must never pop in on screen.
            if ox.abs() < self.rects.keep_hx && oy.abs() < self.rects.keep_hy {
                continue;
            }

            let wcx = cell_at(px + ox);
            let wcy = cell_at(py + oy);
            if !grid.is_loaded_world(WorldCell::new(wcx, wcy)) {
                continue;
            }

            let Some(def) = self.pick_def(grid, wcx, wcy, day) else {
                continue;
            };
            if !self.place(grid, def, wcx, wcy) {
                continue;
            }

            // Pack followers are sampled around the leader and go through the
            // exact same footing, separation and offscreen tests. A failed
            // follower is simply not placed, so a pack in awkward terrain
            // arrives smaller rather than half-buried in it.
            let lo = def.pack_size[0];
            let hi = def.pack_size[1];
            let want = if hi <= lo {
                lo
            } else {
                lo + (self.rng.rand() * (hi - lo + 1.0)) as i32 as f32
            };
            let mut k = 1.0f32;
            while k < want && self.live < MAX_MOBS {
                let dx = js_round(self.rng.rand_range(-PACK_SPREAD, PACK_SPREAD)) as i32;
                let dy = js_round(self.rng.rand_range(-2.0, 2.0)) as i32;
                self.place(grid, def, wcx + dx, wcy + dy);
                k += 1.0;
            }
            return;
        }
    }

    /// Weighted choice among the species whose habitat matches this cell.
    ///
    /// The depth band is measured against the LOCAL surface row (the world has
    /// no fixed top), and the biome is the dominant one at that column — the
    /// same two facts worldgen itself uses to decide what rock to put there.
    ///
    /// The band and depth-window tests are flat array reads (`MOB_BANDS` is a
    /// bitmask built by the compiler), so the common case — a def that does not
    /// live at this depth — costs two indexed reads and a compare, for every
    /// def, on every spawn attempt.
    fn pick_def(
        &mut self,
        grid: &CellGrid,
        wcx: i32,
        wcy: i32,
        day: f32,
    ) -> Option<&'static MobDef> {
        let surf = self
            .heightmap
            .surface_row_at(&self.noise, wcx, None, self.scale);
        let depth = wcy - surf;
        let band_mask = band_bit(band_at_depth(self.scale.depth(f64::from(depth)) as i32));
        let biome = BIOMES
            .get(biome_index_at(&self.noise, wcx, self.scale))
            .map_or("plains", |b| b.id);
        let clamped = depth.clamp(0, 0xffff);

        let mut total = 0.0f32;
        let mut n = 0usize;
        for (i, d) in MOB_DEFS.iter().enumerate() {
            if Band::from_bits_truncate(u32::from(MOB_BANDS[i])) & band_mask == Band::empty() {
                continue;
            }
            if clamped < i32::from(MOB_MINDEPTH[i]) || clamped > i32::from(MOB_MAXDEPTH[i]) {
                continue;
            }
            if let Some(biomes) = d.biomes
                && !biomes.contains(&biome)
            {
                continue;
            }
            if d.needs_lava && !self.lava_near(grid, wcx, wcy) {
                continue;
            }
            let mut w = d.weight;
            // Nocturnal species thin out in daylight rather than vanishing
            // outright, so daytime caves are still populated.
            if d.nocturnal && band_mask == BAND_SURFACE {
                w *= 0.15 + (1.0 - day) * 0.85;
            }
            if w <= 0.0 {
                continue;
            }
            total += w;
            self.cand_idx[n] = i;
            self.cand_w[n] = total;
            n += 1;
        }
        if n == 0 {
            return None;
        }

        let r = self.rng.rand() * total;
        for k in 0..n {
            if r <= self.cand_w[k] {
                return Some(&MOB_DEFS[self.cand_idx[k]]);
            }
        }
        Some(&MOB_DEFS[self.cand_idx[n - 1]])
    }

    fn lava_near(&self, grid: &CellGrid, wcx: i32, wcy: i32) -> bool {
        LAVA_PROBE.iter().any(|(dx, dy)| {
            let w = WorldCell::new(wcx + dx, wcy + dy);
            grid.is_loaded_world(w) && grid.get_world(w) == self.lava
        })
    }

    /// Validate and occupy a slot.
    ///
    /// A candidate is rejected unless the whole body (plus a cell of margin) is
    /// inside the loaded window and free of solids, and — for anything that
    /// walks — unless there is real footing within a few cells below it. Flyers
    /// need an air pocket instead. Rejecting rather than nudging is what
    /// guarantees the invariant that a mob is never inside rock.
    fn place(&mut self, grid: &CellGrid, def: &'static MobDef, wcx: i32, wcy: i32) -> bool {
        let mut cand = Aabb::new(
            (wcx * CELL_SIZE) as f32,
            (wcy * CELL_SIZE) as f32,
            def.w_px,
            def.h_px,
        );

        if !box_loaded(grid, cand) {
            return false;
        }

        let grounded = def.brain != MobBrain::Flyer;
        if grounded {
            // Fall to the first floor within 6 cells; if the body does not fit
            // above it, this position is not a place a creature can stand.
            let mut found = false;
            for step in 0..=6 {
                let foot_row = wcy + step;
                if !is_solid_cell(grid, wcx, foot_row) {
                    continue;
                }
                cand.y = (foot_row * CELL_SIZE) as f32 - def.h_px;
                if !box_overlaps_solid(grid, cand) && box_loaded(grid, cand) {
                    found = true;
                }
                break;
            }
            if !found {
                return false;
            }
            // A burrower starts submerged, so its final rect is a body-length
            // lower than where it would stand. Applied BEFORE the offscreen test
            // below, because that test has to see the rect the creature actually
            // occupies.
            if def.brain == MobBrain::Burrower {
                cand.y += (2 * CELL_SIZE) as f32;
            }
        } else if !pocket_free(grid, cand) {
            // A flyer wants room to move, not a coffin-sized hole.
            return false;
        }

        // The ground search above can slide the body a few cells down from the
        // sampled cell, so the offscreen test is re-run on the FINAL rect.
        // Sampling outside the viewport is not the same as landing outside it.
        if (cand.x + cand.w * 0.5 - self.spawn_px).abs() < self.rects.keep_hx
            && (cand.y + cand.h * 0.5 - self.spawn_py).abs() < self.rects.keep_hy
        {
            return false;
        }

        for o in &self.pool {
            if !o.active {
                continue;
            }
            if (o.body.x - cand.x).abs() < SPAWN_SEPARATION
                && (o.body.y - cand.y).abs() < SPAWN_SEPARATION
            {
                return false;
            }
        }

        let Some(slot) = self.free_slot() else {
            return false;
        };
        self.init(slot, def, cand.x, cand.y);
        true
    }

    fn free_slot(&mut self) -> Option<usize> {
        let n = self.pool.len();
        for k in 0..n {
            let i = (self.slot_cursor + k) % n;
            if !self.pool[i].active {
                self.slot_cursor = (i + 1) % n;
                return Some(i);
            }
        }
        None
    }

    fn init(&mut self, i: usize, def: &'static MobDef, x: f32, y: f32) {
        // Every field, every time: a reused slot that inherited one timer from
        // its previous occupant is the classic pool bug.
        if !self.pool[i].active {
            self.live += 1;
        }
        let facing = if self.rng.rand() < 0.5 { 1.0 } else { -1.0 };
        let variant = (self.rng.rand() * VARIANT_COUNT as f32) as i32;
        // Desynchronise the loops: a pack that spawned on one frame would
        // otherwise animate in lockstep, which reads as one creature drawn four
        // times.
        let state_t = self.rng.rand() * 2.0;
        // Stagger decisions so 32 creatures never all think on the same frame.
        let decide_t = self.rng.rand() * 0.12;
        let beat_t = def.beat * self.rng.rand_range(0.3, 1.2);
        let hazard_t = self.rng.rand() * 0.25;
        let wander = self.rng.rand() * std::f32::consts::PI * 2.0;
        let wander_rate = self.rng.rand_range(1.1, 2.2);
        // Stagger the first shot the same way decisions are staggered, so a pack
        // of shooters does not open with a perfectly synchronised volley.
        let shot_cd = match def.ranged {
            Some(r) => r.cooldown * self.rng.rand_range(0.2, 1.0),
            None => 0.0,
        };

        let m = &mut self.pool[i];
        m.active = true;
        m.def = def;
        m.body = Aabb::new(x, y, def.w_px, def.h_px);
        m.vx = 0.0;
        m.vy = 0.0;
        m.facing = facing;
        m.health = def.max_health;
        m.variant = variant;
        m.on_ground = false;
        m.pose = MobPose::Idle;
        m.clock.state_t = state_t;
        m.clock.clock_t = state_t;
        m.clock.phase = 0.0;
        m.decide_t = decide_t;
        m.beat_t = beat_t;
        m.hazard_t = hazard_t;
        m.attack_cd = 0.0;
        m.flash = 0.0;
        m.alerted = false;
        m.idling = false;
        m.buried = def.brain == MobBrain::Burrower;
        m.tell_t = 0.0;
        m.hesitate_t = 0.0;
        m.wander = wander;
        m.wander_rate = wander_rate;
        m.dive_t = 0.0;
        m.dive_x = 0.0;
        m.dive_y = 0.0;
        m.shot_cd = shot_cd;
        m.wants_shot = false;
        m.last_punch_id = None;
        m.crush_t = 0.0;
        m.doomed = false;
    }
}

/// Is the box, plus a cell of margin, entirely inside the streaming window?
fn box_loaded(grid: &CellGrid, b: Aabb) -> bool {
    let cx0 = cell_at(b.x) - 1;
    let cx1 = cell_at(b.x + b.w) + 1;
    let cy0 = cell_at(b.y) - 1;
    let cy1 = cell_at(b.y + b.h) + 1;
    grid.is_loaded_world(WorldCell::new(cx0, cy0))
        && grid.is_loaded_world(WorldCell::new(cx1, cy0))
        && grid.is_loaded_world(WorldCell::new(cx0, cy1))
        && grid.is_loaded_world(WorldCell::new(cx1, cy1))
}

/// A cell of clear air all round — enough for a flyer to start moving.
fn pocket_free(grid: &CellGrid, b: Aabb) -> bool {
    let cx0 = cell_at(b.x) - 1;
    let cx1 = cell_at(b.x + b.w - 0.001) + 1;
    let cy0 = cell_at(b.y) - 1;
    let cy1 = cell_at(b.y + b.h - 0.001) + 1;
    for cy in cy0..=cy1 {
        for cx in cx0..=cx1 {
            if is_solid_cell(grid, cx, cy) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MAX_HEALTH;
    use crate::entities::player::{Loadout, NoProjectiles, PlayerWeapon};
    use crate::sim::materials::block;
    use crate::sim::worldgen::SpawnPoint;

    /// One fixed step of a body with nowhere to fire.
    ///
    /// Every test in this module is about melee, contact damage or spawning —
    /// never about the player's arrows — so the [`Loadout`] deliberately carries
    /// the pool that REFUSES every shot. A test here that starts depending on a
    /// projectile fails rather than firing into a pool nobody inspects.
    fn step(p: &mut Player, dt: f32, grid: &CellGrid) {
        let mut nowhere = NoProjectiles;
        p.step(
            dt,
            crate::input::Intent::default(),
            grid,
            &mut Loadout::new(&mut nowhere),
        );
    }

    /// A swing from a body with nowhere to fire. See [`step`].
    fn swing(p: &mut Player) -> bool {
        let mut nowhere = NoProjectiles;
        p.attack(0.0, 0.0, &mut Loadout::new(&mut nowhere))
    }

    /// A window at the world origin with solid rock from `floor_row` down, so
    /// world cells and local cells coincide.
    fn world(floor_row: i32) -> CellGrid {
        let mut g = CellGrid::new(WINDOW_COLS, WINDOW_ROWS);
        for cy in floor_row..WINDOW_ROWS {
            for cx in 0..WINDOW_COLS {
                g.set(cx, cy, block::STONE);
            }
        }
        g
    }

    /// The row candidates are sampled at: within the 6-cell footing search of
    /// the floor, and far enough below the player's centre to clear the KEEP
    /// rectangle.
    const SAMPLE_ROW: i32 = 96;
    const FLOOR_ROW: i32 = 100;

    fn system() -> MobSystem {
        MobSystem::new(View::default())
    }

    #[test]
    fn the_pool_is_allocated_once_and_never_grows() {
        let sys = system();
        assert_eq!(sys.pool.len(), MAX_MOBS);
        assert_eq!(sys.shots.len(), MAX_SHOTS);
        assert_eq!(sys.events.len(), MAX_EVENTS);
        assert_eq!(sys.loot.len(), MAX_LOOT);
        assert_eq!(sys.count(), 0);
    }

    #[test]
    fn placing_more_creatures_than_the_pool_holds_simply_stops_placing() {
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let def = super::super::defs::def_by_id("grubling");

        // Ten cells apart, which clears SPAWN_SEPARATION with room to spare.
        let mut placed = 0;
        for k in 0..(MAX_MOBS * 2) {
            if sys.place(&g, def, 10 + 10 * k as i32, SAMPLE_ROW) {
                placed += 1;
            }
            assert!(sys.count() <= MAX_MOBS, "the pool grew past its cap");
            assert_eq!(sys.pool.len(), MAX_MOBS, "the pool itself grew");
        }
        assert_eq!(
            placed, MAX_MOBS,
            "every slot should have been claimable once"
        );
        assert_eq!(sys.count(), MAX_MOBS);
        assert_eq!(
            sys.pool.iter().filter(|m| m.active).count(),
            MAX_MOBS,
            "`live` and the flags must agree"
        );
    }

    #[test]
    fn a_retired_slot_is_reclaimed_and_the_cursor_rolls() {
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let def = super::super::defs::def_by_id("grubling");
        for k in 0..MAX_MOBS {
            assert!(sys.place(&g, def, 10 + 10 * k as i32, SAMPLE_ROW));
        }
        assert!(
            !sys.place(&g, def, 10 + 10 * MAX_MOBS as i32, SAMPLE_ROW),
            "a full pool refuses"
        );

        // Free two slots that are NOT next to the cursor, so the claim has to
        // wrap to find them — which is the whole point of the rolling search.
        sys.retire(3);
        sys.retire(20);
        assert_eq!(sys.count(), MAX_MOBS - 2);

        let a = sys.free_slot().expect("a freed slot must be claimable");
        // Claiming does not mark it live; `init` does. Take it properly.
        sys.init(a, def, 0.0, 0.0);
        let b = sys.free_slot().expect("and so must the second");
        sys.init(b, def, 0.0, 0.0);
        assert_ne!(a, b, "the cursor handed out the same slot twice");
        assert_eq!([a.min(b), a.max(b)], [3, 20]);
        assert_eq!(sys.count(), MAX_MOBS);
        assert!(sys.free_slot().is_none(), "and then there are none");
    }

    #[test]
    fn the_event_buffer_saturates_rather_than_growing() {
        let mut sys = system();
        for i in 0..(MAX_EVENTS * 4) {
            sys.push_rgb(MobEventKind::MobHurt, i as f32, 0.0, [1, 2, 3], 1.0);
        }
        assert_eq!(sys.event_count(), MAX_EVENTS);
        assert_eq!(sys.events().len(), MAX_EVENTS, "the buffer itself grew");
        // Saturating keeps the OLDEST, which is the opposite of `Player::emit`
        // and is deliberate — see `push_rgb`.
        assert_eq!(sys.events()[0].x, 0.0);
        assert_eq!(sys.events()[MAX_EVENTS - 1].x, (MAX_EVENTS - 1) as f32);

        sys.clear_events();
        assert_eq!(sys.event_count(), 0);
        sys.push_rgb(MobEventKind::MobDie, 9.0, 0.0, [0, 0, 0], 1.0);
        assert_eq!(sys.event_count(), 1);
    }

    #[test]
    fn the_loot_buffer_saturates_rather_than_growing() {
        let g = world(FLOOR_ROW);
        let mut sys = system();
        // A grubling drops chitin on every kill and grub_meat on some, so a full
        // pool of them rolls more entries than the buffer holds.
        let def = super::super::defs::def_by_id("grubling");
        for k in 0..MAX_MOBS {
            assert!(sys.place(&g, def, 10 + 10 * k as i32, SAMPLE_ROW));
        }
        for i in 0..MAX_MOBS {
            sys.kill(i);
        }
        assert_eq!(sys.count(), 0, "killing retires the slot");
        assert!(sys.loot_count() <= MAX_LOOT);
        assert_eq!(sys.loot_count(), MAX_LOOT, "32 guaranteed drops fill it");
        assert_eq!(sys.loot().len(), MAX_LOOT, "the buffer itself grew");
        assert_eq!(sys.xp_banked(), def.xp * MAX_MOBS as i32);
        assert_eq!(sys.value_banked(), def.value * MAX_MOBS as i32);

        sys.clear_loot();
        assert_eq!(sys.loot_count(), 0);
        assert_eq!(sys.xp_banked(), 0);
        assert_eq!(sys.value_banked(), 0);
    }

    #[test]
    fn clear_empties_the_population_without_touching_the_pool() {
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let def = super::super::defs::def_by_id("grubling");
        for k in 0..8 {
            assert!(sys.place(&g, def, 10 + 10 * k, SAMPLE_ROW));
        }
        sys.shots[0].active = true;
        sys.clear();
        assert_eq!(sys.count(), 0);
        assert_eq!(sys.pool.len(), MAX_MOBS);
        assert!(sys.pool.iter().all(|m| !m.active));
        assert!(sys.shots.iter().all(|s| !s.active));
    }

    #[test]
    fn one_swing_lands_on_one_creature_exactly_once() {
        // The `swing_id` de-dup: the damage window spans several 60Hz sub-steps,
        // and without the id a held sword would tick a mob down every one of
        // them.
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let def = super::super::defs::def_by_id("grubling");
        assert!(sys.place(&g, def, 40, SAMPLE_ROW));
        let slot = sys.pool.iter().position(|m| m.active).unwrap();
        let body = sys.pool[slot].body;

        let mut player = Player::new(SpawnPoint { x: 0.0, y: 0.0 });
        player.x = body.x;
        player.y = body.y;
        player.facing = 1.0;
        assert!(swing(&mut player), "the fist is off cooldown");
        assert!(player.punching());

        let player_box = Aabb::new(player.x, player.y, PLAYER_W, PLAYER_H);
        let before = sys.pool[slot].health;
        for _ in 0..10 {
            sys.resolve_combat(slot, &mut player, player_box, true);
        }
        let dealt = before - sys.pool[slot].health;
        let one_hit = 1.0f32.max(player.melee_damage() - def.armor);
        assert_eq!(dealt, one_hit, "one swing, one hit");

        // A NEW swing lands again.
        player.set_weapon(None);
        for _ in 0..40 {
            step(&mut player, 1.0 / 120.0, &g);
        }
        assert!(swing(&mut player));
        sys.resolve_combat(slot, &mut player, player_box, true);
        assert_eq!(before - sys.pool[slot].health, one_hit * 2.0);
    }

    #[test]
    fn armour_is_a_flat_subtraction_floored_at_one() {
        let g = world(FLOOR_ROW);
        let mut sys = system();
        // A stoneback's armour is far above a bare fist's damage, so the floor is
        // what it takes: not invulnerable, just a creature that wants a pick.
        let heavy = super::super::defs::def_by_id("stoneback");
        assert!(heavy.armor > 5.0, "the fixture assumes heavy plate");
        assert!(sys.place(&g, heavy, 40, SAMPLE_ROW));
        let slot = sys.pool.iter().position(|m| m.active).unwrap();
        let before = sys.pool[slot].health;
        let b = sys.pool[slot].body;
        assert!(sys.hit_at(b.x + 1.0, b.y + 1.0, 5.0, 100.0, 1.0, 0.0));
        assert_eq!(before - sys.pool[slot].health, 1.0);

        // And a miss is a miss.
        assert!(!sys.hit_at(b.x - 500.0, b.y, 5.0, 100.0, 1.0, 0.0));
    }

    #[test]
    fn a_buried_creature_is_not_a_target() {
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let worm = super::super::defs::def_by_id("sandworm");
        assert!(sys.place(&g, worm, 40, SAMPLE_ROW));
        let slot = sys.pool.iter().position(|m| m.active).unwrap();
        assert!(sys.pool[slot].buried, "a burrower spawns submerged");
        let b = sys.pool[slot].body;
        assert!(
            !sys.hit_at(b.x + 1.0, b.y + 1.0, 50.0, 100.0, 1.0, 0.0),
            "an arrow cannot hit rock-bound prey"
        );
        assert_eq!(sys.pool[slot].health, worm.max_health);
    }

    #[test]
    fn the_spawn_rectangles_nest_and_stay_inside_the_streaming_window() {
        for (w, h) in [(1000u32, 500u32), (2560, 1440), (640, 480)] {
            let r = SpawnRects::for_view(View::for_screen(w, h));
            assert!(r.spawn_hx > r.keep_hx, "{w}x{h}: nothing could ever spawn");
            assert!(r.spawn_hy > r.keep_hy, "{w}x{h}");
            assert!(r.despawn_hx > r.spawn_hx, "{w}x{h}: spawned then culled");
            assert!(r.despawn_hy > r.spawn_hy, "{w}x{h}");
            // The SAFE clamp is what keeps a candidate inside the loaded window
            // even at the worst-case moment just before it shifts.
            assert!(r.spawn_hx <= SAFE_HX.max(r.keep_hx + 40.0), "{w}x{h}");
            assert!(r.spawn_hy <= SAFE_HY.max(r.keep_hy + 40.0), "{w}x{h}");
        }
    }

    #[test]
    fn two_systems_on_one_seed_spawn_the_same_world() {
        // The second global RNG, threaded rather than shared: every spawn
        // rejection, every pack roll and every variant tint comes off one stream
        // in one order, so two systems driven identically must be identical.
        let g = world(FLOOR_ROW);
        let run = || {
            let mut sys = system();
            let mut player = Player::new(SpawnPoint { x: 0.0, y: 0.0 });
            player.x = 400.0;
            player.y = 400.0;
            for _ in 0..600 {
                sys.update(1.0 / 60.0, &g, &mut player, 1.0);
            }
            sys.pool
                .iter()
                .map(|m| {
                    (
                        m.active,
                        m.def.code,
                        m.body.x.to_bits(),
                        m.body.y.to_bits(),
                        m.variant,
                    )
                })
                .collect::<Vec<_>>()
        };
        let a = run();
        assert!(
            a.iter().any(|s| s.0),
            "the fixture spawned nothing, so it would prove nothing"
        );
        assert_eq!(a, run(), "the same seed produced a different population");
    }

    // --- the truce -----------------------------------------------------------

    /// A body, optionally one the creatures may not perceive.
    fn body(untouchable: bool) -> Player {
        let mut p = Player::new(SpawnPoint { x: 0.0, y: 0.0 });
        p.untouchable = untouchable;
        p
    }

    /// One grubling on the floor, and the slot it landed in. A walker with
    /// `contactDamage` 8 and `aggroPx` 130 — hostile enough to prove both halves
    /// of the truce.
    fn one_grubling(sys: &mut MobSystem, g: &CellGrid) -> usize {
        let def = super::super::defs::def_by_id("grubling");
        assert!(sys.place(g, def, 40, SAMPLE_ROW));
        sys.pool.iter().position(|m| m.active).unwrap()
    }

    /// The bare minimum a `MobTarget` was before `targetable` existed: every
    /// required method answered, and the new one left to its default.
    ///
    /// Written out rather than reusing `Player`, which DOES override it. The
    /// whole argument for defaulting the method is that a harness fake should
    /// not have to answer a question it has no opinion about, and this is the
    /// only thing that can check that argument still holds.
    #[derive(Default)]
    struct BareTarget {
        health: f32,
    }

    impl MobTarget for BareTarget {
        fn x(&self) -> f32 {
            0.0
        }
        fn y(&self) -> f32 {
            0.0
        }
        fn health(&self) -> f32 {
            self.health
        }
        fn take_damage(&mut self, amount: f32) {
            self.health -= amount;
        }
        fn facing(&self) -> f32 {
            1.0
        }
        fn punching(&self) -> bool {
            false
        }
        fn swing_id(&self) -> u32 {
            0
        }
        fn hit_box(&self) -> Aabb {
            Aabb::new(0.0, 0.0, 0.0, 0.0)
        }
        fn melee_damage(&self) -> f32 {
            0.0
        }
        fn melee_knockback(&self) -> f32 {
            0.0
        }
    }

    #[test]
    fn a_target_that_does_not_mention_targetable_is_still_a_target() {
        // The default has to be `true`, or adding the method would silently have
        // pacified the game for every implementor that predates it.
        assert!(BareTarget::default().targetable());
        assert!(body(false).targetable(), "and so is an ordinary player");
        assert!(!body(true).targetable());
    }

    #[test]
    fn an_untouchable_body_takes_no_contact_damage_from_a_creature() {
        let g = world(FLOOR_ROW);

        // The same fixture twice, differing only in the flag: the creature is
        // placed, the body is dropped on top of it, and one 60Hz step is run
        // through the real `update`, so what is under test is the wiring and not
        // just the guard.
        let bitten = |untouchable: bool| {
            let mut sys = system();
            let slot = one_grubling(&mut sys, &g);
            let b = sys.pool[slot].body;
            let mut p = body(untouchable);
            p.x = b.x;
            p.y = b.y;
            sys.update(MOB_DT, &g, &mut p, 1.0);
            (MAX_HEALTH - p.health, sys.pool[slot].attack_cd)
        };

        let (hurt, cd) = bitten(false);
        assert!(hurt > 0.0, "the fixture did not bite, so it proves nothing");
        assert!(cd > 0.0, "a bite spends the creature's cooldown");

        let (hurt, cd) = bitten(true);
        assert_eq!(hurt, 0.0);
        assert_eq!(
            cd, 0.0,
            "nothing happened, so nothing should have been consumed by it"
        );
    }

    #[test]
    fn an_untouchable_body_takes_no_projectile_damage_from_a_creature() {
        let g = world(FLOOR_ROW);

        // A shot planted stationary inside the body, so the only question the
        // step has to answer is whether it connects. Making a creature fire one
        // for real would test the aggro gate a second time instead of testing
        // the impact gate once.
        let shot_at = |untouchable: bool| {
            let mut sys = system();
            let mut p = body(untouchable);
            p.x = 200.0;
            p.y = 200.0;
            sys.shots[0] = Shot {
                active: true,
                x: p.x + PLAYER_W * 0.5,
                y: p.y + PLAYER_H * 0.5,
                vx: 0.0,
                vy: 0.0,
                life: 5.0,
                damage: 11.0,
                r_px: 1.0,
                chill: 0.0,
                glow: 0.0,
                rgb: [255, 0, 0],
            };
            sys.update(MOB_DT, &g, &mut p, 1.0);
            (MAX_HEALTH - p.health, sys.shots[0].active)
        };

        let (hurt, still_flying) = shot_at(false);
        assert_eq!(hurt, 11.0, "the fixture did not land, so it proves nothing");
        assert!(!still_flying, "a shot that lands is consumed");

        let (hurt, still_flying) = shot_at(true);
        assert_eq!(hurt, 0.0);
        assert!(
            still_flying,
            "a shot already in the air should pass through, not vanish: this is \
             a truce, not a screen-clear"
        );
    }

    #[test]
    fn creatures_do_not_aggro_on_an_untouchable_body_but_still_wander() {
        let g = world(FLOOR_ROW);

        // Five seconds: several `DECIDE_INTERVAL`s and several of the grubling's
        // 2.4s beats, so both the alert test and the wander have run repeatedly.
        let run = |untouchable: bool| {
            let mut sys = system();
            let slot = one_grubling(&mut sys, &g);
            let start = sys.pool[slot].body.x;
            let mut p = body(untouchable);
            // Standing on the creature, which is as far inside `aggroPx` as it
            // is possible to be.
            p.x = sys.pool[slot].body.x;
            p.y = sys.pool[slot].body.y;
            for _ in 0..300 {
                sys.update(MOB_DT, &g, &mut p, 1.0);
            }
            (
                sys.pool[slot].active,
                sys.pool[slot].alerted,
                (sys.pool[slot].body.x - start).abs(),
            )
        };

        let (alive, alerted, _) = run(false);
        assert!(alive && alerted, "the fixture failed to provoke a creature");

        let (alive, alerted, travelled) = run(true);
        assert!(!alerted, "a creature aggroed on a body it cannot perceive");
        assert!(alive, "the creature was culled rather than ignored");
        assert!(
            travelled > 0.0,
            "the creature stopped living: this is the world ignoring you, not \
             the world stopping"
        );
    }

    #[test]
    fn creatures_still_spawn_around_an_untouchable_body() {
        // The spawner and the culler read the player's REAL box, because the
        // world is still built around where the player is standing. If the
        // out-of-range offset ever leaked into either, the population would try
        // to form a million pixels away and this would find nothing.
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let mut p = body(true);
        p.x = 400.0;
        p.y = 400.0;
        for _ in 0..600 {
            sys.update(1.0 / 60.0, &g, &mut p, 1.0);
        }
        assert!(sys.count() > 0, "the world emptied out");
    }

    #[test]
    fn a_creature_is_still_killable_by_a_player_it_cannot_see() {
        // The half of `resolve_combat` that must NOT be gated. An early return
        // for an untargetable player would have taken this with it and made
        // creative a mode in which nothing can be fought.
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let slot = one_grubling(&mut sys, &g);
        let def = sys.pool[slot].def;

        let mut p = body(true);
        let b = sys.pool[slot].body;
        p.x = b.x;
        p.y = b.y;
        p.facing = 1.0;

        // A swing that hurts but does not kill, so both outcomes are observed.
        p.set_weapon(Some(PlayerWeapon {
            damage: def.max_health - 1.0 + def.armor,
            ..PlayerWeapon::default()
        }));
        assert!(swing(&mut p));
        sys.update(MOB_DT, &g, &mut p, 1.0);
        assert!(
            sys.pool[slot].active,
            "the first swing should not have killed"
        );
        assert!(
            sys.pool[slot].health < def.max_health,
            "an untouchable player could not hurt a creature"
        );

        // And the killing blow, on a fresh swing id.
        for _ in 0..120 {
            step(&mut p, 1.0 / 120.0, &g);
        }
        p.x = sys.pool[slot].body.x;
        p.y = sys.pool[slot].body.y;
        p.facing = 1.0;
        assert!(swing(&mut p));
        sys.update(MOB_DT, &g, &mut p, 1.0);
        assert!(
            !sys.pool[slot].active,
            "the creature survived a lethal swing"
        );
        assert_eq!(sys.xp_banked(), def.xp, "no kill was banked");
    }

    #[test]
    fn the_out_of_range_offset_is_past_every_perception_in_the_game() {
        // The constant's whole claim, checked against the compiled content
        // rather than against the numbers quoted in its doc comment — which is
        // the only version of this assertion that survives a content edit.
        let widest = MOB_DEFS
            .iter()
            .map(|d| {
                let ranged = d.ranged.map_or(0.0, |r| r.range);
                d.aggro_px.max(d.aggro_y_px).max(d.flee_px).max(ranged)
            })
            .fold(0.0f32, f32::max);
        assert!(widest > 0.0, "no creature perceives anything: bad fixture");
        assert!(
            OUT_OF_RANGE_OFFSET > widest * 1000.0,
            "the offset ({OUT_OF_RANGE_OFFSET}) is no longer orders of magnitude \
             past the widest perception in content ({widest})"
        );
        // Finite, and its SQUARE is finite: the flyer's steering squares it, and
        // an infinity there would put a NaN into every velocity in the pool.
        assert!(OUT_OF_RANGE_OFFSET.is_finite());
        assert!((OUT_OF_RANGE_OFFSET * OUT_OF_RANGE_OFFSET).is_finite());
    }

    #[test]
    fn the_fixed_step_never_runs_more_than_the_spiral_guard_allows() {
        let g = world(FLOOR_ROW);
        let mut sys = system();
        let mut player = Player::new(SpawnPoint { x: 0.0, y: 0.0 });
        player.x = 400.0;
        player.y = 400.0;
        // A ten-second stall. The accumulator is dumped rather than replayed, so
        // the next frame starts clean instead of spending minutes catching up.
        sys.update(10.0, &g, &mut player, 1.0);
        assert!(
            sys.accumulator <= MOB_DT * MAX_MOB_STEPS as f32,
            "the accumulator survived a stall: {}",
            sys.accumulator
        );
    }

    /// Armour subtracts from a hit, and never all of it.
    ///
    /// The same rule and the same floor creatures get, which is why this asserts
    /// against `d.armor`'s formula rather than a number typed twice.
    #[test]
    fn armour_reduces_a_hit_and_one_damage_always_gets_through() {
        let mut p = Player::new(SpawnPoint { x: 0.0, y: 0.0 });

        p.armour = 0.0;
        p.health = 100.0;
        p.take_damage(10.0);
        assert_eq!(p.health, 90.0, "no armour, no reduction");

        p.armour = 4.0;
        p.health = 100.0;
        p.take_damage(10.0);
        assert_eq!(p.health, 94.0, "four points off a ten-point hit");

        // The floor. Without it, enough armour is not toughness — it is
        // invulnerability to a whole band of the world at once.
        p.armour = 50.0;
        p.health = 100.0;
        p.take_damage(10.0);
        assert_eq!(p.health, 99.0, "one damage always gets through");
    }

    /// Creative still stops everything, armour or not. `untouchable` is checked
    /// first and is stronger than any number.
    #[test]
    fn untouchable_still_beats_armour_arithmetic() {
        let mut p = Player::new(SpawnPoint { x: 0.0, y: 0.0 });
        p.untouchable = true;
        p.armour = 0.0;
        p.health = 100.0;
        p.take_damage(999.0);
        assert_eq!(p.health, 100.0);
    }
}

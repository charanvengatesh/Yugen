//! The player's projectile pool.
//!
//! This is the same design as the mobs' shot pool, and deliberately so: a
//! fixed-length store allocated once, claimed and released, with NO path that
//! grows it. The cap is therefore structural rather than a rule the code has to
//! remember to enforce — which matters more here than it does for the mobs,
//! because the mobs' rate of fire is bounded by their own cooldowns and the
//! player's is bounded only by how fast a human can press a key.
//!
//! # Why SoA and not a pool of objects
//!
//! The mob pool holds shot structs; this holds parallel arrays. Both allocate
//! nothing at steady state, so the difference is not garbage — it is that
//! [`ProjectileSystem::update`] touches six contiguous arrays in index order,
//! and the whole live set is a few hundred bytes that stay in cache while a
//! screenful of terrain cells is being read past it. The ergonomic cost is the
//! same one the TypeScript paid: there is no shot object to hand around, so
//! everything that reads a shot reads it by index, here, in this file.
//! [`ProjectileSystem::shots`] is the one exception, and it exists for a
//! renderer that has no business knowing the layout.
//!
//! # Oldest-recycled, not dropped
//!
//! A full mob pool silently drops the shot: the cooldown is already spent, so a
//! saturated screen thins the volume of fire. The player's pool does the
//! opposite and recycles the OLDEST shot instead, because the player has just
//! spent AMMO — a real, countable resource out of the inventory — and an arrow
//! that vanishes because 24 others are in flight is an input that was silently
//! eaten. The oldest shot is the one furthest away and least likely to be being
//! watched.
//!
//! # What this file does not know
//!
//! It does not know what a mob is. Hits are resolved through [`ShotHitTest`], a
//! callback the owner supplies, for the same reason the mob system owns player
//! damage rather than reaching into [`Player`](super::player::Player): the
//! entity that owns the targets owns the rules for hurting them. Colour and size
//! are the projectile's own business; damage, armour and knockback resistance
//! are the target's.
//!
//! # What the port changed
//!
//! - `draw` and `drawGlow` were Canvas2D and did not come across; this crate
//!   carries no renderer. What they READ is published as [`ProjectileSystem::shots`]
//!   and the two style tables, so the same two passes can be reassembled in
//!   `godgame-render` without the pool gaining a pixel of knowledge about how it
//!   looks. `STYLE_CSS` — an `rgb(...)` string baked once at module load to keep
//!   allocation off the render path — has no analogue and is gone; a `[u8; 3]`
//!   was never the thing being cached.
//! - `hitTest`/`onImpact` were public nullable fields. They are private with
//!   setters, because a `Box<dyn FnMut>` is not `Copy` and a public field would
//!   invite a caller to swap one mid-`update`.
//! - The launch stamp was a `Float64Array`. It is `u64` here: the whole point of
//!   a stamp rather than an age is that it is exact and cannot tie, and an
//!   integer says that outright instead of relying on 2^53.

use core::fmt;

use crate::config::{SHOT_GRAVITY, SHOT_LIFE, cell_at};
use crate::physics::collision::is_solid_cell;
use crate::sim::grid::CellGrid;

use super::player::Projectiles;

/// Hard cap on player shots in flight. Also the array length — see the header.
///
/// 24 matches the mob side's own cap. At the fastest plausible bow cadence
/// (~4/s) and a [`SHOT_LIFE`] of 2.2s, about 9 are ever live; the headroom is
/// for the moment a player empties a quiver into a corridor, and the cap is what
/// stops that from being a frame-rate event.
pub const MAX_SHOTS: usize = 24;

/// The projectile style a player's arrow is drawn in.
///
/// Style is a small integer on the shot rather than a name, so a draw loop
/// indexes an array instead of hashing.
pub const SHOT_STYLE_ARROW: u8 = 0;

/// Appearance, per style: linear sRGB bytes, indexed by [`ShotSpec::style`].
const STYLE_RGB: [[u8; 3]; 1] = [
    [226, 214, 176], // arrow: pale fletched wood, readable against rock and sky
];

/// Additive overlay strength in the post-light pass, per style.
///
/// 0 = not self-luminous. Nothing the player fires glows yet, so the whole table
/// is zero and the glow pass is a no-op — it exists so that the first enchanted
/// bolt does not have to invent a render pass. A self-luminous projectile drawn
/// only in the world layer is multiplied away by the darkness mask in exactly
/// the cave where it is the one thing on screen the player has to react to.
const STYLE_GLOW: [f32; 1] = [0.0];

/// Colour for a style, clamped to a real entry the way [`ProjectileSystem::fire`]
/// clamps the style it stores.
#[inline]
pub fn style_rgb(style: u8) -> [u8; 3] {
    STYLE_RGB[style_index(style)]
}

/// Additive glow strength for a style. See [`STYLE_GLOW`].
#[inline]
pub fn style_glow(style: u8) -> f32 {
    STYLE_GLOW[style_index(style)]
}

/// A style number as an index into the tables, or the arrow's.
#[inline]
fn style_index(style: u8) -> usize {
    let i = style as usize;
    if i < STYLE_RGB.len() {
        i
    } else {
        SHOT_STYLE_ARROW as usize
    }
}

/// Everything a weapon contributes to one shot. Passed on every
/// [`ProjectileSystem::fire`] rather than stored, because the held item can
/// change between two shots and a cached spec would fire the previous bow's
/// arrow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShotSpec {
    /// px/s along the aim direction, already body-scaled by the caller.
    pub speed: f32,
    /// Hit points on contact.
    pub damage: f32,
    /// Impulse handed to [`ShotHitTest`] for the target to resist or apply.
    pub knockback: f32,
    /// Half-extent drawn, in px. 1 = a 2x2 mote, which is one cell wide at 5px.
    pub r_px: f32,
    /// Which of the pool's baked styles to draw.
    pub style: u8,
}

/// Ask the owner whether anything at this point was hit. Returns true if the
/// shot connected, in which case it dies.
///
/// The arguments are `(x, y, damage, knockback, dir_x, dir_y)`. `dir_x`/`dir_y`
/// are the shot's unit velocity, so the target can apply knockback along the
/// flight path rather than guessing from geometry.
///
/// A boxed `FnMut` rather than a plain `fn` pointer for the same reason
/// [`AmmoSource`](super::player::AmmoSource) is one: the TypeScript closed over
/// the creature system, and a bare function pointer cannot capture.
pub type ShotHitTest = Box<dyn FnMut(f32, f32, f32, f32, f32, f32) -> bool + Send + Sync>;

/// Called wherever a shot ended, for juice. The `bool` is `true` when it struck
/// a target and `false` when it struck the world.
pub type ShotImpact = Box<dyn FnMut(f32, f32, bool) + Send + Sync>;

/// One live shot, as a renderer needs it. See the header on `draw`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    /// Centre, world px.
    pub x: f32,
    /// Centre, world px. Y grows downward.
    pub y: f32,
    /// Half-extent, px. The drawn square is `2 * r_px` on a side.
    pub r_px: f32,
    /// Index into the style tables. See [`style_rgb`] and [`style_glow`].
    pub style: u8,
}

/// The player's arrows, in flight.
///
/// Fill a [`Player`](super::player::Player)'s
/// [`Projectiles`](super::player::Projectiles) seam with one of these — see
/// [`Player::with_projectiles`](super::player::Player::with_projectiles) — and
/// install a [`ShotHitTest`] from wherever the creatures live.
pub struct ProjectileSystem {
    // --- SoA. Parallel; index i is one shot. Allocated once, never resized. ---
    px: [f32; MAX_SHOTS],
    py: [f32; MAX_SHOTS],
    vx: [f32; MAX_SHOTS],
    vy: [f32; MAX_SHOTS],
    life: [f32; MAX_SHOTS],
    damage: [f32; MAX_SHOTS],
    knockback: [f32; MAX_SHOTS],
    radius: [f32; MAX_SHOTS],
    style: [u8; MAX_SHOTS],
    live: [bool; MAX_SHOTS],
    /// Launch order, for oldest-recycled. A monotonically increasing stamp
    /// rather than an age in seconds: ages would have to be compared as floats
    /// every eviction, and a stamp is exact, cheap, and cannot tie.
    stamp: [u64; MAX_SHOTS],
    next_stamp: u64,

    /// Live shot count, maintained incrementally — never a scan.
    count: usize,
    /// Rolling claim cursor. Claims and releases are spread evenly over the
    /// array, so starting where the last claim finished makes the search
    /// amortised O(1) instead of an O(n) walk from index 0 every time.
    cursor: usize,

    /// Owner-supplied target resolution. `None` = shots only ever hit the world.
    hit_test: Option<ShotHitTest>,
    /// Owner-supplied impact hook, for particles. `None` = no juice.
    on_impact: Option<ShotImpact>,
}

impl ProjectileSystem {
    /// An empty pool with no target resolution wired up.
    pub fn new() -> ProjectileSystem {
        ProjectileSystem {
            px: [0.0; MAX_SHOTS],
            py: [0.0; MAX_SHOTS],
            vx: [0.0; MAX_SHOTS],
            vy: [0.0; MAX_SHOTS],
            life: [0.0; MAX_SHOTS],
            damage: [0.0; MAX_SHOTS],
            knockback: [0.0; MAX_SHOTS],
            radius: [0.0; MAX_SHOTS],
            style: [SHOT_STYLE_ARROW; MAX_SHOTS],
            live: [false; MAX_SHOTS],
            stamp: [0; MAX_SHOTS],
            next_stamp: 1,
            count: 0,
            cursor: 0,
            hit_test: None,
            on_impact: None,
        }
    }

    /// Install the owner's target resolution. See [`ShotHitTest`].
    pub fn set_hit_test(&mut self, f: Option<ShotHitTest>) {
        self.hit_test = f;
    }

    /// Install the impact hook. See [`ShotImpact`].
    pub fn set_on_impact(&mut self, f: Option<ShotImpact>) {
        self.on_impact = f;
    }

    /// How many are in flight. For the debug HUD and the test harness.
    #[inline]
    pub fn live_count(&self) -> usize {
        self.count
    }

    /// Every live shot, for a renderer.
    ///
    /// This is what is left of `draw` and `drawGlow`, which were Canvas2D and
    /// stayed behind: a caller squares each of these off at `r_px`, colours it
    /// with [`style_rgb`], and — in a second, additive pass — overlays the ones
    /// whose [`style_glow`] is positive. Positions are NOT rounded here; that is
    /// a decision about which pixel grid the caller is drawing on.
    pub fn shots(&self) -> impl Iterator<Item = Shot> + '_ {
        (0..MAX_SHOTS).filter(|&i| self.live[i]).map(|i| Shot {
            x: self.px[i],
            y: self.py[i],
            r_px: self.radius[i],
            style: self.style[i],
        })
    }

    /// A free index, or the oldest live one. Never fails, never grows the arrays.
    fn claim(&mut self) -> usize {
        for k in 0..MAX_SHOTS {
            let i = (self.cursor + k) % MAX_SHOTS;
            if !self.live[i] {
                self.cursor = (i + 1) % MAX_SHOTS;
                self.live[i] = true;
                self.count += 1;
                return i;
            }
        }
        let mut oldest = 0;
        for i in 1..MAX_SHOTS {
            if self.stamp[i] < self.stamp[oldest] {
                oldest = i;
            }
        }
        oldest // already live; count and cursor stay as they are
    }

    /// Release a slot. Idempotent, because `update` can reach a shot that a
    /// previous branch already retired.
    fn retire(&mut self, i: usize) {
        if !self.live[i] {
            return;
        }
        self.live[i] = false;
        self.count -= 1;
    }

    /// Fire the impact hook, if there is one.
    fn impact(&mut self, x: f32, y: f32, hit: bool) {
        if let Some(f) = self.on_impact.as_mut() {
            f(x, y, hit);
        }
    }
}

impl Default for ProjectileSystem {
    fn default() -> ProjectileSystem {
        ProjectileSystem::new()
    }
}

impl fmt::Debug for ProjectileSystem {
    /// The callbacks are closures and the arrays are mostly dead slots; the only
    /// thing worth printing is how many shots are up.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectileSystem")
            .field("live_count", &self.count)
            .finish_non_exhaustive()
    }
}

impl Projectiles for ProjectileSystem {
    /// Launch one shot from `(x, y)` toward `(dir_x, dir_y)`, which need not be a
    /// unit vector — it is normalised here, so a caller may pass a raw delta to a
    /// target.
    ///
    /// Returns false only for a degenerate direction, which is a caller bug
    /// rather than a runtime condition: a full pool recycles instead of refusing
    /// (see the header), so "the shot was fired" is otherwise always true and the
    /// ammo the caller has already spent is never spent for nothing.
    fn fire(&mut self, x: f32, y: f32, dir_x: f32, dir_y: f32, spec: ShotSpec) -> bool {
        let len = dir_x.mul_add(dir_x, dir_y * dir_y).sqrt();
        // The NaN arm is not defensive padding: the TypeScript's `!(len > 0)`
        // rejected a NaN direction, and a plain `len <= 0.0` would let one
        // through to become a shot at NaN with no position and no way to die.
        if len <= 0.0 || len.is_nan() {
            return false;
        }

        let i = self.claim();
        let inv = spec.speed / len;
        self.px[i] = x;
        self.py[i] = y;
        self.vx[i] = dir_x * inv;
        self.vy[i] = dir_y * inv;
        self.life[i] = SHOT_LIFE;
        self.damage[i] = spec.damage;
        self.knockback[i] = spec.knockback;
        self.radius[i] = spec.r_px;
        self.style[i] = style_index(spec.style) as u8;
        self.stamp[i] = self.next_stamp;
        self.next_stamp += 1;
        true
    }

    /// Integrate every live shot. Point-vs-cell against the world and a point
    /// query against whatever [`ShotHitTest`] knows about — a projectile this
    /// small does not warrant a swept AABB, and at 5px cells the difference is
    /// not something a player can perceive.
    ///
    /// Gravity is applied BEFORE the position update, so a shot's arc is the same
    /// shape at any frame rate the fixed step is driven at.
    fn update(&mut self, dt: f32, grid: &CellGrid) {
        for i in 0..MAX_SHOTS {
            if !self.live[i] {
                continue;
            }

            self.life[i] -= dt;
            if self.life[i] <= 0.0 {
                self.retire(i);
                continue;
            }

            self.vy[i] += SHOT_GRAVITY * dt;
            self.px[i] += self.vx[i] * dt;
            self.py[i] += self.vy[i] * dt;
            let x = self.px[i];
            let y = self.py[i];

            // Unloaded cells read as solid, so this is also what stops a shot
            // escaping the streaming window — the same guarantee the mobs rely
            // on.
            if is_solid_cell(grid, cell_at(x), cell_at(y)) {
                self.retire(i);
                self.impact(x, y, false);
                continue;
            }

            if self.hit_test.is_none() {
                continue;
            }
            // Read the slot BEFORE borrowing the callback: the borrow of
            // `hit_test` has to end before `retire` can take `&mut self`.
            let damage = self.damage[i];
            let knockback = self.knockback[i];
            let vx = self.vx[i];
            let vy = self.vy[i];
            let len = vx.mul_add(vx, vy * vy).sqrt();
            let len = if len > 0.0 { len } else { 1.0 };
            let hit = self
                .hit_test
                .as_mut()
                .is_some_and(|t| t(x, y, damage, knockback, vx / len, vy / len));
            if hit {
                self.retire(i);
                self.impact(x, y, true);
            }
        }
    }

    /// Recycle everything (respawn, level reload).
    ///
    /// The stamp counter is deliberately NOT reset: it is a launch ORDER, and
    /// restarting it would make a shot fired after the clear compare as older
    /// than one fired before it if any survived a partial clear.
    fn clear(&mut self) {
        self.live = [false; MAX_SHOTS];
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// The shared handle
// ---------------------------------------------------------------------------

/// A pool the owner can still see after the player has taken it.
///
/// [`Player`](super::player::Player) owns its pool as a `Box<dyn Projectiles>`
/// and publishes it back only as `&dyn Projectiles`, which has no accessor for
/// the live set. That is the right shape for the PLAYER — it must not learn what
/// a shot IS — but it leaves two callers with nothing to hold:
///
///   - a renderer, which has to walk [`ProjectileSystem::shots`] to draw arrows;
///   - whoever owns the creatures, which has to install a [`ShotHitTest`] and is
///     not constructed until after the player is.
///
/// Widening the trait would fix it for both and is the wrong trade: `fire`,
/// `update` and `clear` are the three verbs the player needs, and every method
/// added past them is one more thing the player could accidentally reach for.
/// So the handle is shared instead — clone one into the player, keep one.
///
/// The `Mutex` is not defending against a second thread; there is one schedule
/// and it steps this from one system. It is what makes the handle `Sync` so the
/// player stays `Send + Sync`, and it is uncontended in the ordinary case. The
/// one ordering rule is on [`SharedPool::update`].
#[derive(Clone, Default)]
pub struct SharedPool(std::sync::Arc<std::sync::Mutex<ProjectileSystem>>);

impl SharedPool {
    /// An empty pool, with one handle to it.
    pub fn new() -> SharedPool {
        SharedPool::default()
    }

    /// Borrow the pool.
    ///
    /// Panics if a previous borrow panicked while holding it, which is the same
    /// contract every `Mutex` in this crate has: a poisoned pool is a bug, not a
    /// state to recover into.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, ProjectileSystem> {
        self.0.lock().expect("projectile pool poisoned")
    }
}

impl fmt::Debug for SharedPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SharedPool").field(&*self.lock()).finish()
    }
}

impl Projectiles for SharedPool {
    fn fire(&mut self, x: f32, y: f32, dir_x: f32, dir_y: f32, spec: ShotSpec) -> bool {
        self.lock().fire(x, y, dir_x, dir_y, spec)
    }

    /// One step of every shot, with the lock held across it.
    ///
    /// That is the ordering rule: a [`ShotHitTest`] installed on this pool runs
    /// INSIDE this call, so it must not reach back for a second handle to the
    /// same pool. Reaching for a different lock — the creatures, say — is fine
    /// and is the whole point, because nothing the creatures own locks a pool
    /// back.
    fn update(&mut self, dt: f32, grid: &CellGrid) {
        self.lock().update(dt, grid);
    }

    fn clear(&mut self) {
        self.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CELL_SIZE;
    use crate::sim::materials::{EMPTY, block};

    /// An all-air grid with a floor of stone across the bottom third.
    fn grid_with_floor(floor_row: i32) -> CellGrid {
        let mut g = CellGrid::new(64, 64);
        for y in 0..g.rows() {
            for x in 0..g.cols() {
                g.set(x, y, if y >= floor_row { block::STONE } else { EMPTY });
            }
        }
        g
    }

    fn arrow(speed: f32) -> ShotSpec {
        ShotSpec {
            speed,
            damage: 7.0,
            knockback: 100.0,
            r_px: 1.0,
            style: SHOT_STYLE_ARROW,
        }
    }

    #[test]
    fn a_fresh_pool_is_empty() {
        let p = ProjectileSystem::new();
        assert_eq!(p.live_count(), 0);
        assert_eq!(p.shots().count(), 0);
    }

    #[test]
    fn fire_normalises_the_direction_to_the_spec_speed() {
        let mut p = ProjectileSystem::new();
        // A raw delta, not a unit vector: 3-4-5, so the speed must come out of
        // the spec and not out of the length of what was passed in.
        assert!(p.fire(10.0, 20.0, 3.0, 4.0, arrow(500.0)));
        assert_eq!(p.live_count(), 1);
        assert_eq!(p.vx[0], 300.0);
        assert_eq!(p.vy[0], 400.0);
        assert_eq!((p.px[0], p.py[0]), (10.0, 20.0));
        assert_eq!(p.life[0], SHOT_LIFE);
    }

    #[test]
    fn a_degenerate_direction_is_refused_and_claims_nothing() {
        let mut p = ProjectileSystem::new();
        assert!(!p.fire(0.0, 0.0, 0.0, 0.0, arrow(500.0)));
        assert!(!p.fire(0.0, 0.0, f32::NAN, 0.0, arrow(500.0)));
        assert_eq!(p.live_count(), 0);
    }

    #[test]
    fn an_unknown_style_falls_back_to_the_arrow() {
        let mut p = ProjectileSystem::new();
        let mut spec = arrow(100.0);
        spec.style = 200;
        p.fire(0.0, 0.0, 1.0, 0.0, spec);
        assert_eq!(p.shots().next().unwrap().style, SHOT_STYLE_ARROW);
        assert_eq!(style_rgb(200), STYLE_RGB[0]);
        assert_eq!(style_glow(200), 0.0);
    }

    #[test]
    fn gravity_bends_the_arc_downward() {
        let g = grid_with_floor(1000); // no floor in range
        let mut p = ProjectileSystem::new();
        // Dead flat, so every bit of vy is gravity's doing.
        p.fire(100.0, 100.0, 1.0, 0.0, arrow(200.0));

        let dt = 1.0 / 120.0;
        p.update(dt, &g);
        assert_eq!(p.vy[0], SHOT_GRAVITY * dt);
        // Applied BEFORE the position update, so the first step already drops.
        assert!(p.py[0] > 100.0, "{}", p.py[0]);
        assert!((p.px[0] - (100.0 + 200.0 * dt)).abs() < 1e-3);
    }

    #[test]
    fn a_shot_expires_after_shot_life() {
        // Big and empty, with more headroom below the launch than SHOT_LIFE of
        // free fall covers: an unloaded cell reads as solid, so a grid that runs
        // out under the shot would kill it on the world instead of on the clock.
        let g = CellGrid::new(512, 512);
        let mut p = ProjectileSystem::new();
        // Straight up, so gravity is the only thing acting and it never meets a
        // cell — the life timer is what has to kill it.
        p.fire(1000.0, 600.0, 0.0, -1.0, arrow(50.0));

        let dt = 1.0 / 120.0;
        let steps = (SHOT_LIFE / dt).ceil() as i32;
        for _ in 0..steps - 2 {
            p.update(dt, &g);
        }
        assert_eq!(p.live_count(), 1, "died early");
        p.update(dt, &g);
        p.update(dt, &g);
        assert_eq!(p.live_count(), 0, "outlived SHOT_LIFE");
    }

    #[test]
    fn a_shot_dies_on_the_first_solid_cell_and_reports_the_world() {
        let floor_row = 40;
        let g = grid_with_floor(floor_row);
        let mut p = ProjectileSystem::new();

        let hits = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&hits);
        p.set_on_impact(Some(Box::new(move |x, y, hit| {
            sink.lock().unwrap().push((x, y, hit));
        })));

        // Straight down at the floor from well above it.
        p.fire(100.0, 10.0, 0.0, 1.0, arrow(400.0));
        let dt = 1.0 / 120.0;
        for _ in 0..240 {
            p.update(dt, &g);
            if p.live_count() == 0 {
                break;
            }
        }

        assert_eq!(p.live_count(), 0, "flew through the floor");
        let log = hits.lock().unwrap();
        assert_eq!(log.len(), 1);
        let (_, y, hit) = log[0];
        assert!(!hit, "a wall reported itself as a target");
        assert_eq!(cell_at(y), floor_row, "died a cell early or late");
    }

    #[test]
    fn an_unloaded_cell_stops_a_shot_leaving_the_window() {
        // A grid whose window is entirely air: the only thing out there is the
        // unloaded margin, which `is_solid_cell` reports as solid.
        let g = CellGrid::new(64, 64);
        let mut p = ProjectileSystem::new();
        p.fire(0.0, 100.0, -1.0, 0.0, arrow(400.0));
        for _ in 0..120 {
            p.update(1.0 / 120.0, &g);
        }
        assert_eq!(p.live_count(), 0);
    }

    #[test]
    fn the_hit_test_gets_the_flight_direction_and_kills_the_shot() {
        let g = grid_with_floor(1000);
        let mut p = ProjectileSystem::new();

        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&seen);
        p.set_hit_test(Some(Box::new(move |x, y, dmg, kb, dx, dy| {
            sink.lock().unwrap().push((x, y, dmg, kb, dx, dy));
            // Connect on the second query, so the first proves a miss is a miss.
            sink.lock().unwrap().len() == 2
        })));

        p.fire(100.0, 100.0, 1.0, 0.0, arrow(200.0));
        p.update(1.0 / 120.0, &g);
        assert_eq!(p.live_count(), 1, "a miss killed the shot");
        p.update(1.0 / 120.0, &g);
        assert_eq!(p.live_count(), 0, "a hit did not kill the shot");

        let log = seen.lock().unwrap();
        let (_, _, dmg, kb, dx, dy) = log[0];
        assert_eq!((dmg, kb), (7.0, 100.0));
        // Unit velocity: mostly +x, tipped down by one step of gravity.
        assert!((dx * dx + dy * dy - 1.0).abs() < 1e-5);
        assert!(dx > 0.99 && dy > 0.0, "{dx} {dy}");
    }

    #[test]
    fn a_full_pool_recycles_the_oldest_rather_than_refusing() {
        let mut p = ProjectileSystem::new();
        for i in 0..MAX_SHOTS {
            assert!(p.fire(i as f32, 0.0, 1.0, 0.0, arrow(100.0)));
        }
        assert_eq!(p.live_count(), MAX_SHOTS);

        // The 25th goes into slot 0 — the oldest — and the count does not move.
        assert!(p.fire(999.0, 0.0, 1.0, 0.0, arrow(100.0)));
        assert_eq!(p.live_count(), MAX_SHOTS, "the pool grew");
        assert_eq!(p.px[0], 999.0, "recycled something other than the oldest");
        assert_eq!(p.px[1], 1.0, "recycled more than one");
    }

    #[test]
    fn the_cursor_reuses_a_freed_slot_without_scanning_from_zero() {
        let g = grid_with_floor(1000);
        let mut p = ProjectileSystem::new();
        for _ in 0..MAX_SHOTS {
            p.fire(100.0, 100.0, 1.0, 0.0, arrow(100.0));
        }
        // Retire one in the middle and refill: the claim has to find it.
        p.retire(7);
        assert_eq!(p.live_count(), MAX_SHOTS - 1);
        p.fire(42.0, 0.0, 1.0, 0.0, arrow(100.0));
        assert_eq!(p.px[7], 42.0);
        assert_eq!(p.live_count(), MAX_SHOTS);
        // And nothing above claimed a second slot for it.
        p.update(1.0 / 120.0, &g);
        assert_eq!(p.live_count(), MAX_SHOTS);
    }

    #[test]
    fn clear_recycles_everything() {
        let mut p = ProjectileSystem::new();
        for _ in 0..5 {
            p.fire(0.0, 0.0, 1.0, 0.0, arrow(100.0));
        }
        p.clear();
        assert_eq!(p.live_count(), 0);
        assert_eq!(p.shots().count(), 0);
        // And the pool is usable again afterwards.
        assert!(p.fire(1.0, 2.0, 1.0, 0.0, arrow(100.0)));
        assert_eq!(p.live_count(), 1);
    }

    #[test]
    fn shots_reports_what_a_renderer_needs() {
        let mut p = ProjectileSystem::new();
        let mut spec = arrow(100.0);
        spec.r_px = 2.0;
        p.fire(33.0, 44.0, 1.0, 0.0, spec);
        let s: Vec<Shot> = p.shots().collect();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].x, 33.0);
        assert_eq!(s[0].y, 44.0);
        assert_eq!(s[0].r_px, 2.0);
        assert_eq!(style_rgb(s[0].style), [226, 214, 176]);
    }

    // --- The player's end of the seam ---------------------------------------

    /// The whole path: a held bow, one attack, an arrow in this pool.
    ///
    /// The interesting part is not that `fire` was reached — `player.rs` already
    /// tests that against a counting stub — but that the [`ShotSpec`] the player
    /// builds lands in THIS pool as a shot with the bow's speed on it.
    #[test]
    fn a_ranged_weapon_puts_a_real_shot_in_the_pool() {
        use crate::entities::player::{Player, PlayerWeapon};
        use crate::sim::worldgen::SpawnPoint;

        let shared = SharedPool::new();
        let spawn = SpawnPoint {
            x: (20 * CELL_SIZE) as f32,
            y: (20 * CELL_SIZE) as f32,
        };
        let mut player = Player::with_projectiles(spawn, Box::new(shared.clone()));
        player.set_weapon(Some(PlayerWeapon {
            damage: 11.0,
            knockback: None,
            swing_time: Some(0.5),
            ranged: true,
            projectile_speed: Some(600.0),
            ammo: &[],
        }));

        // No aim point: the arrow leaves flat along the facing, which `reset`
        // left at +1.
        assert!(player.attack(0.0, 0.0));
        let pool = shared.lock();
        assert_eq!(pool.live_count(), 1);
        assert_eq!(pool.damage[0], 11.0);
        assert!(pool.vx[0] > 0.0 && pool.vy[0] == 0.0);
        // Body-scaled, so not 600 exactly — but it is the bow's number scaled,
        // not the fallback.
        assert_ne!(pool.vx[0], crate::config::SHOT_SPEED_DEFAULT);
        assert!(pool.vx[0] > 100.0);
        // Launched from the leading edge of the box and from the chest, not from
        // the centre and not from the feet.
        assert_eq!(pool.px[0], player.x + crate::config::PLAYER_W);
        assert!(pool.py[0] > player.y && pool.py[0] < player.y + crate::config::PLAYER_H);
    }

    /// A respawn drops everything in flight — arrows belong to the life that
    /// fired them.
    #[test]
    fn reset_clears_the_pool_through_the_seam() {
        use crate::entities::player::Player;
        use crate::sim::worldgen::SpawnPoint;

        let shared = SharedPool::new();
        let mut player =
            Player::with_projectiles(SpawnPoint { x: 0.0, y: 0.0 }, Box::new(shared.clone()));
        shared.lock().fire(0.0, 0.0, 1.0, 0.0, arrow(100.0));
        assert_eq!(shared.lock().live_count(), 1);
        player.reset();
        assert_eq!(shared.lock().live_count(), 0);
    }
}

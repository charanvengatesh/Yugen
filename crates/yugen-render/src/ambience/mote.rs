//! One mote's life: how it is born, how it drifts, and the pool that holds it.
//!
//! [`super`]'s "The model" and "The pool". Fixed-length, allocated once,
//! recycled — the same trade [`crate::particles`] makes and for the same reason:
//! ambient life is cosmetic, so a full pool declines to add rather than
//! deleting something already on screen.
//!
//! Split from [`super::spawn`] on the line between deciding to emit and having
//! something to emit into.

use yugen_core::config::{CELL_SIZE, WorldScale};
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::materials::EMPTY;
use yugen_core::sim::noise::Noise;
use yugen_core::sim::rng::SimRng;
use yugen_core::sim::worldgen::heightmap::Heightmap;
use yugen_core::sim::worldgen::world_noise;

use yugen_core::sim::biomes::{Mix, biome_mix_at, underground_mix_at};

use super::scan::*;
use super::spawn::*;
use crate::daynight::DayPhase;

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// The ambience model: the climate fields, the eight accumulators, and the
/// stream that pays for the jitter.
///
/// Holds no Bevy and no pool. [`Ambience::update`] takes the pool it emits into,
/// which is what lets the whole thing be exercised without an `App`.
pub struct Ambience {
    /// The world's noise, for the climate fields. Same seed, same fields, same
    /// boundaries the terrain dithers on.
    noise: Noise,
    /// The memoising surface-row front end. Owned rather than borrowed because
    /// `surface_row_at` takes `&mut self` for its cache — see [`Heightmap`].
    heightmap: Heightmap,
    /// Ambience's own stream. NOT the automata's — see the module header.
    rng: SimRng,
    /// The seed the two above were built from, so a world change is detectable.
    seed: u32,
    /// The mood as of the last [`Ambience::resolve`].
    mood: Mood,
    /// Fractional spawn debt per emitter — carries the sub-mote remainder.
    pub(super) debt: [f32; EMITTER_COUNT],
}

impl Ambience {
    /// The ambience of the world grown from `seed`.
    pub fn new(seed: u32) -> Ambience {
        Ambience {
            noise: world_noise(seed),
            heightmap: Heightmap::new(),
            rng: SimRng::seeded(seed),
            seed,
            mood: Mood::default(),
            debt: [0.0; EMITTER_COUNT],
        }
    }

    /// The seed the climate fields were built from.
    #[inline]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Rebuild against `seed` if the world underneath has changed.
    ///
    /// A new world means new boundaries, so the debt and the resolved mood are
    /// dropped with the fields rather than carried into a place they no longer
    /// describe.
    pub fn follow_seed(&mut self, seed: u32) {
        if self.seed != seed {
            *self = Ambience::new(seed);
        }
    }

    /// The mood as of the last [`Ambience::resolve`].
    #[inline]
    pub fn mood(&self) -> Mood {
        self.mood
    }

    /// What one emitter currently owes, in fractional motes.
    #[inline]
    pub fn debt_of(&self, e: Emitter) -> f32 {
        self.debt[e.index()]
    }

    /// Sample the climate at the view centre and work out how far between
    /// "outdoors" and "underground" the view sits.
    ///
    /// One [`biome_mix_at`] plus one [`underground_mix_at`] per frame: a handful
    /// of two-octave fBm evaluations, the same cost the sky already pays.
    ///
    /// The port rounds the fractional camera column to a whole one, because the
    /// ported [`biome_mix_at`] takes an `i32` where the TypeScript took a
    /// `number` and multiplied it straight into a ~0.001 frequency. At that
    /// frequency half a column is under a thousandth of a noise unit, so the
    /// weights agree to several decimals; what it buys is that the mix here and
    /// the mix worldgen dithered the terrain with are the same call on the same
    /// integer column, which is the property the whole crossfade rests on.
    pub fn resolve(&mut self, cam_col: f32, cam_row: f32) -> Mood {
        let col = cam_col.round() as i32;
        let mut mood = Mood::default();

        // Depth below the LOCAL surface, not an absolute row.
        let surface = self
            .heightmap
            .surface_row_at(&self.noise, col, None, WorldScale::LIVE) as f32;
        mood.underground = smoothstep01((cam_row - surface - UG_FADE_START) / UG_FADE_SPAN);

        for (biome, w) in weights_of(&biome_mix_at(&self.noise, col, WorldScale::LIVE)) {
            mood.surface[biome.index()] = w;
            mood.weather[biome.def().atmo.weather as usize] += w;
        }
        for (layer, w) in weights_of(&underground_mix_at(&self.noise, col, WorldScale::LIVE)) {
            mood.layer[layer.index()] = w;
        }

        self.mood = mood;
        mood
    }

    /// Resolve the local mood and emit a frame's worth of ambience.
    ///
    /// Emission happens before integration, as it did in the TypeScript, where
    /// `Ambience.update` ran in the frame's sim block and `ParticleSystem.update`
    /// ran later in the same frame: a mote gets one step of motion before it is
    /// first drawn.
    pub fn update(&mut self, dt: f32, around: &Around, phase: &DayPhase, pool: &mut MotePool) {
        self.resolve(around.view.centre_col(), around.view.centre_row());
        let rates = spawn_rates(&self.mood, phase, around.hot.len());

        for e in Emitter::ALL {
            self.run(e, dt, rates[e.index()], around, pool);
        }
    }

    /// Accumulate fractional spawn debt and fire whole motes out of it.
    pub(super) fn run(
        &mut self,
        e: Emitter,
        dt: f32,
        rate: f32,
        around: &Around,
        pool: &mut MotePool,
    ) {
        let slot = e.index();
        if rate <= RATE_EPS {
            self.debt[slot] = 0.0;
            return;
        }
        let mut owed = (self.debt[slot] + rate * dt).min(DEBT_CAP);
        while owed >= 1.0 {
            self.spawn(e, around, pool);
            owed -= 1.0;
        }
        self.debt[slot] = owed;
    }

    /// Find this emitter a point and put one mote on it.
    ///
    /// Each kind picks by rejection sampling INSIDE THE VIEW RECT and gives up
    /// after a few tries; a frame that finds nowhere valid simply emits nothing.
    /// The debt is spent either way, which is deliberate — an emitter that
    /// carried its failures forward would fire a burst the moment the player
    /// walked past a gap in the rock.
    fn spawn(&mut self, e: Emitter, around: &Around, pool: &mut MotePool) {
        let found = match e {
            Emitter::Firefly | Emitter::Pollen | Emitter::Spore | Emitter::CaveDust => {
                self.pick_air(around)
            }
            Emitter::Sand => self.pick_ground(around),
            Emitter::Drip => self.pick_ceiling(around),
            Emitter::Glint => self.pick_wall(around),
            Emitter::Ember => self.pick_hot(around),
        };
        let Some((x, y)) = found else {
            return;
        };

        let mut spec = e.spec();
        if e == Emitter::Ember {
            let (base, range) = EMBER_GREEN;
            spec.color[1] = base.saturating_add(self.rng.rand_int(range) as u8);
        }
        pool.emit(x, y, &spec, &mut self.rng);
    }

    /// A uniform point inside the view, in world px.
    fn sample_point(&mut self, view: ViewRect) -> (f32, f32) {
        (
            view.x + unit(&mut self.rng) * view.w,
            view.y + unit(&mut self.rng) * view.h,
        )
    }

    /// An empty cell somewhere in view.
    fn pick_air(&mut self, around: &Around) -> Option<(f32, f32)> {
        for _ in 0..AIR_TRIES {
            let (x, y) = self.sample_point(around.view);
            let at = cell_of(x, y);
            if !around.grid.is_loaded_world(at) {
                continue;
            }
            if around.grid.get_world(at) == EMPTY {
                return Some((x, y));
            }
        }
        None
    }

    /// An air cell whose neighbour below is solid — the ground line.
    ///
    /// The returned y is one px ABOVE the air cell's top edge, so the grain
    /// skims the surface instead of starting somewhere up the column of air.
    fn pick_ground(&mut self, around: &Around) -> Option<(f32, f32)> {
        for _ in 0..SURFACE_TRIES {
            let (x, y) = self.sample_point(around.view);
            let at = cell_of(x, y);
            let below = WorldCell::new(at.x, at.y + 1);
            if !around.grid.is_loaded_world(at) || !around.grid.is_loaded_world(below) {
                continue;
            }
            if around.grid.get_world(at) != EMPTY || around.grid.get_world(below) == EMPTY {
                continue;
            }
            return Some((x, (at.y * CELL_SIZE - 1) as f32));
        }
        None
    }

    /// An air cell with solid directly above — a ceiling to drip from.
    fn pick_ceiling(&mut self, around: &Around) -> Option<(f32, f32)> {
        for _ in 0..SURFACE_TRIES {
            let (x, y) = self.sample_point(around.view);
            let at = cell_of(x, y);
            let above = WorldCell::new(at.x, at.y - 1);
            if !around.grid.is_loaded_world(at) || !around.grid.is_loaded_world(above) {
                continue;
            }
            if around.grid.get_world(at) != EMPTY || around.grid.get_world(above) == EMPTY {
                continue;
            }
            return Some((x, (at.y * CELL_SIZE) as f32));
        }
        None
    }

    /// An air cell touching solid on either side — a wall face to glint off.
    fn pick_wall(&mut self, around: &Around) -> Option<(f32, f32)> {
        for _ in 0..SURFACE_TRIES {
            let (x, y) = self.sample_point(around.view);
            let at = cell_of(x, y);
            if !around.grid.is_loaded_world(at) || around.grid.get_world(at) != EMPTY {
                continue;
            }
            let left = around.grid.get_world(WorldCell::new(at.x - 1, at.y));
            let right = around.grid.get_world(WorldCell::new(at.x + 1, at.y));
            if left == EMPTY && right == EMPTY {
                continue;
            }
            return Some((x, y));
        }
        None
    }

    /// A point just above one of the emissive cells the scan already found.
    ///
    /// No rejection sampling: the hot list is by construction a list of cells
    /// that exist and are in view, so this either has one or the rate was zero.
    fn pick_hot(&mut self, around: &Around) -> Option<(f32, f32)> {
        if around.hot.is_empty() {
            return None;
        }
        let k = self.rng.rand_int(around.hot.len() as i32) as usize;
        let at = around.hot[k];
        let jitter = (unit(&mut self.rng) - 0.5) * EMBER_JITTER_PX;
        Some((
            (at.x * CELL_SIZE) as f32 + jitter,
            (at.y * CELL_SIZE) as f32 - EMBER_LIFT_PX,
        ))
    }
}

/// Cell containing a world-px point. Floors, so it is correct west and north of
/// the origin as well.
#[inline]
pub(super) fn cell_of(x: f32, y: f32) -> WorldCell {
    WorldCell::new(
        (x / CELL_SIZE as f32).floor() as i32,
        (y / CELL_SIZE as f32).floor() as i32,
    )
}

/// A [`Mix`]'s entries paired with their individual normalised weights.
///
/// [`Mix::cum`] is the RUNNING total, so a weight is a difference of neighbours.
/// Undoing the running sum here rather than at each of the two call sites keeps
/// the "did I remember to subtract the previous one" question in one place.
pub(super) fn weights_of<T: Copy>(mix: &Mix<T>) -> impl Iterator<Item = (T, f32)> + '_ {
    let mut prev = 0.0f64;
    mix.items().iter().zip(mix.cum()).map(move |(item, cum)| {
        let w = cum - prev;
        prev = *cum;
        (*item, w as f32)
    })
}

/// The depth blend's easing curve — [`crate::daynight`] keeps the same two lines
/// private for its own.
///
/// `f32::clamp` rather than the TypeScript's ternary chain. The chain existed
/// because `clamp` panics on a NaN BOUND; both bounds here are literals, and on
/// a NaN INPUT the two agree — `clamp` propagates it and the chain's comparisons
/// both fail through to `v`.
#[inline]
pub(super) fn smoothstep01(v: f32) -> f32 {
    let t = v.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A uniform `[0, 1)` off the ambience stream.
///
/// The top 24 bits and not all 32: 24 is exactly what an `f32` mantissa holds,
/// so the quotient is exact and no two adjacent draws collapse onto one float.
/// [`SimRng`] divides in `f64` for the automata's parity with the TypeScript;
/// nothing here is compared against it, so the cheaper exact form is right.
#[inline]
pub(super) fn unit(rng: &mut SimRng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

/// A draw scaled into `range`.
#[inline]
pub(super) fn jittered(range: (f32, f32), t: f32) -> f32 {
    range.0 + (range.1 - range.0) * t
}

// ---------------------------------------------------------------------------
// The pool
// ---------------------------------------------------------------------------

/// One mote in flight.
///
/// Public because [`MotePool::motes`] hands the slice out for drawing, and a
/// draw pass that has to guess which fields exist is a draw pass that will guess
/// wrong once.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mote {
    /// World px, +x right.
    pub x: f32,
    /// World px, +y DOWN — the sim's convention, flipped once at draw time.
    pub y: f32,
    /// Velocity, px/s.
    pub vx: f32,
    /// Velocity, px/s, +y DOWN.
    pub vy: f32,
    /// Remaining seconds. `<= 0` means the slot is free.
    pub life: f32,
    /// Life at birth, for the fade envelope.
    pub max_life: f32,
    /// Square edge, world px.
    pub size: f32,
    /// Downward acceleration, px/s^2.
    pub gravity: f32,
    /// Per-second velocity damping.
    pub drag: f32,
    /// Drift acceleration, px/s^2. 0 is ballistic.
    pub wander: f32,
    /// Drift phase, advanced every step.
    pub phase: f32,
    /// Colour, 0-255 per channel.
    pub color: [u8; 3],
    /// Self-luminous. See the module header for what this does and does not do.
    pub glow: bool,
    /// Ease alpha in as well as out.
    pub fade_in: bool,
}

impl Mote {
    /// Whether this slot holds a live mote.
    #[inline]
    pub fn alive(&self) -> bool {
        self.life > 0.0
    }

    /// Alpha, 0..1.
    ///
    /// Every mote fades out linearly over its life; a `fade_in` mote also eases
    /// in over the first `1 / FADE_IN_RATE` of it, so ambient motes materialise
    /// instead of popping in front of the camera.
    pub fn alpha(&self) -> f32 {
        if !self.alive() {
            return 0.0;
        }
        let remaining = self.life / self.max_life;
        if !self.fade_in {
            return remaining;
        }
        remaining * ((1.0 - remaining) * FADE_IN_RATE).min(1.0)
    }
}

/// A fixed-capacity pool of motes.
///
/// Emitting reuses a dead slot instead of growing, and stepping touches every
/// slot but does work only on the live ones. Overflow is DROPPED rather than
/// grown into: ambience is cosmetic, so a saturated pool should quietly stop
/// adding motes instead of making the frame that saturated it the expensive one.
pub struct MotePool {
    slots: Vec<Mote>,
    /// Ring cursor: where the next scan for a free slot begins.
    cursor: usize,
}

impl MotePool {
    /// An empty pool of `capacity` slots.
    pub fn new(capacity: usize) -> MotePool {
        MotePool {
            slots: vec![Mote::default(); capacity],
            cursor: 0,
        }
    }

    /// Every slot, live and dead. The index is stable for the pool's lifetime,
    /// which is what lets the sprites be pooled one-to-one against it.
    #[inline]
    pub fn motes(&self) -> &[Mote] {
        &self.slots
    }

    /// How many slots there are.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// How many are live.
    pub fn live_count(&self) -> usize {
        self.slots.iter().filter(|m| m.alive()).count()
    }

    /// Put one mote at world px `(x, y)`, or drop it if the pool is saturated.
    ///
    /// Four draws, in a fixed order: angle, speed, life, wander phase. The order
    /// is not a parity contract with anything — it is fixed so a test can pin a
    /// spawn.
    pub fn emit(&mut self, x: f32, y: f32, spec: &EmitSpec, rng: &mut SimRng) {
        let Some(i) = self.allocate() else {
            return;
        };

        let angle = spec.angle.unwrap_or(UP) + (unit(rng) - 0.5) * spec.spread;
        let speed = spec.speed * jittered(SPEED_JITTER, unit(rng));
        let life = spec.life * jittered(LIFE_JITTER, unit(rng));

        self.slots[i] = Mote {
            x,
            y,
            vx: angle.cos() * speed,
            vy: angle.sin() * speed,
            life,
            max_life: life,
            size: spec.size,
            gravity: spec.gravity,
            drag: spec.drag,
            wander: spec.wander,
            phase: unit(rng) * core::f32::consts::TAU,
            color: spec.color,
            glow: spec.glow,
            fade_in: spec.fade_in,
        };
    }

    /// Integrate every live mote and free the ones that have aged out.
    pub fn update(&mut self, dt: f32) {
        for m in &mut self.slots {
            if !m.alive() {
                continue;
            }
            let remaining = m.life - dt;
            if remaining <= 0.0 {
                *m = Mote::default();
                continue;
            }
            m.life = remaining;

            // Symplectic-ish Euler: gravity, then wander, then drag, then
            // position — the TypeScript's order, and the order matters because
            // drag is applied to the velocity the accelerations just produced.
            let mut vx = m.vx;
            let mut vy = m.vy + m.gravity * dt;

            if m.wander > 0.0 {
                m.phase += dt * WANDER_RATE;
                vx += m.phase.cos() * m.wander * dt;
                vy += (m.phase * WANDER_Y_FREQ).sin() * m.wander * WANDER_Y_GAIN * dt;
            }

            if m.drag > 0.0 {
                // Frame-rate independent enough for something cosmetic, and
                // clamped at 0 so a long frame stops the mote rather than
                // reversing it.
                let k = (1.0 - m.drag * dt).max(0.0);
                vx *= k;
                vy *= k;
            }

            m.vx = vx;
            m.vy = vy;
            m.x += vx * dt;
            m.y += vy * dt;
        }
    }

    /// Free every slot.
    pub fn clear(&mut self) {
        for m in &mut self.slots {
            *m = Mote::default();
        }
        self.cursor = 0;
    }

    /// A free slot, scanning forward from the ring cursor. `None` when saturated.
    fn allocate(&mut self) -> Option<usize> {
        let n = self.slots.len();
        for _ in 0..n {
            let i = self.cursor;
            self.cursor = if self.cursor + 1 >= n {
                0
            } else {
                self.cursor + 1
            };
            if !self.slots[i].alive() {
                return Some(i);
            }
        }
        None
    }
}

impl Default for MotePool {
    fn default() -> MotePool {
        MotePool::new(POOL_CAPACITY)
    }
}

//! The pooled particle system: impact debris, dust, splashes, embers.
//!
//! Ported from `src/render/Particles.ts`.
//!
//! # The pool is the design
//!
//! Particles live in a fixed-length struct-of-arrays store allocated once.
//! Emitting reuses a dead slot instead of growing; the step and the draw touch
//! only live slots. Nothing on the per-frame path allocates.
//!
//! That property was load-bearing in the TypeScript for a reason that does not
//! survive the port — there is no garbage collector here to wake up mid-game —
//! and it is kept anyway, for the reason that does: the cap is STRUCTURAL. A
//! screen full of explosions cannot make this module cost more than a screen
//! with one spark on it, because there is no code path that would let it. This
//! is the same trade [`godgame_core::entities::projectiles`] documents at
//! length, at 85x the slot count.
//!
//! # Full pool: dropped, not recycled
//!
//! The player's arrow pool recycles its OLDEST slot when it saturates, because
//! the player has spent countable ammo and an arrow that silently vanished is an
//! input that was eaten. This pool does the opposite and DROPS the overflow,
//! because juice is cosmetic: under load the game stops adding new sparks, which
//! nobody can see, rather than deleting the oldest half of a burst that is
//! currently on screen, which everybody can. The two pools differ deliberately,
//! and the difference is a statement about what the contents are worth.
//!
//! # The sprite pool mirrors the particle pool
//!
//! [`ParticlesPlugin`] spawns [`MAX_PARTICLES`] hidden sprite entities at
//! startup and never spawns or despawns another. A particle dying toggles a
//! [`Visibility`], not an ECS archetype — the same pattern, and the same
//! reasoning, as [`crate::mobs`]'s creature placeholders. It matters more here:
//! a burst is fourteen particles arriving and leaving inside half a second, and
//! at that rate spawn/despawn would be a steady archetype churn for something
//! the player experiences as a puff of dust.
//!
//! # What the port changed
//!
//! - **The clock.** The TypeScript stepped particles once per rendered FRAME
//!   with the frame `dt`. Here they step in `FixedUpdate` at
//!   [`STEP_DT`]. That is the clock the things that emit them run on — a
//!   creature's death, a landing, an arrow's impact all arrive from the fixed
//!   step — so a particle now ages on the same clock as the event that made it,
//!   and its arc no longer changes shape with the frame rate.
//! - **Collision. This is a deliberate divergence from the original, not an
//!   oversight — it was raised and kept on purpose.** The TypeScript's particles
//!   were pure ballistics and fell through the floor; nothing in a Canvas2D
//!   `fillRect` cared. Here debris opts into the cell grid through
//!   [`EmitOpts::collide`], and the presets that represent MATTER —
//!   [`ParticleSystem::burst`], `dust`, `splash`, `puff`, `scrape` — have it ON.
//!   The ones that represent air or light (`trail`, `smear`, `ember`) stay
//!   ballistic, because an ember that stops at a ceiling reads as a bug and an
//!   ember that drifts through one reads as heat. See
//!   [`ParticleSystem::update`] for how the test is kept to one branch.
//!
//!   Every other deviation in this file exists because Canvas2D and wgpu are
//!   different machines. This one does not: the original could have done it and
//!   chose not to. It is the one place where this port knowingly plays better
//!   than the thing it is porting, and no parity suite covers particles, so
//!   nothing but this paragraph would tell you. Flipping `collide` to `false` on
//!   the five matter presets restores the original behaviour exactly and changes
//!   nothing else.
//! - **Randomness is a value.** `Math.random()` became [`JuiceRng`], threaded
//!   through the system it belongs to. Cosmetic randomness has no reason to be
//!   entropy-seeded, and making it a value is what lets every test below assert
//!   on a real burst instead of on a mock.
//! - **Two draw passes became two z levels.** `draw` and `drawGlow` were Canvas2D
//!   and did not come across. What separated them was the compositing mode, not
//!   the geometry; see below for what is left of that.
//!
//! # What the port dropped on the way in
//!
//! - **Additive compositing.** `drawGlow` set `globalCompositeOperation =
//!   "lighter"` so that self-luminous motes survived the lighting multiply. A
//!   Bevy [`Sprite`] alpha-blends and there is no per-sprite blend state to
//!   change without a custom material, so glowing particles are drawn as
//!   ordinary sprites at a higher z instead. The FLAG and the pass separation
//!   are kept intact ([`ParticleSystem::glow_count`], `GLOW_Z`) precisely so
//!   that the milestone which owns the darkness multiply — [`crate::light`] —
//!   can turn this into a real additive pass by swapping a material, not by
//!   reintroducing a concept.
//! - **`COLOR_CACHE`.** A `Map` of packed rgb to `rgb(r,g,b)` strings, so the
//!   draw loop did not allocate 2048 strings a frame for a palette of about six
//!   colours. There is no string here — a colour is three bytes — so the cache
//!   has nothing to cache. The "skip the assignment when the colour has not
//!   changed" trick went with it: a `Sprite`'s colour is a field, not a state
//!   machine.
//!
//! # Wiring
//!
//! Nothing calls [`ParticleSystem::emit`] yet. The two seams are already
//! draining the events that should feed it, and both say so in their own
//! comments:
//!
//! | Event source | Where it is drained | The call to add |
//! |---|---|---|
//! | [`MobEvent`] | `crate::mobs::step_creatures` | `particles.mob_event(e)` |
//! | [`PlayerEvent`](godgame_core::entities::player::PlayerEvent) | `crate::player::step_player` | one preset per arm — see [`ParticleSystem::puff`] |
//!
//! [`ParticleSystem::mob_event`] is one line because a [`MobEvent`] carries
//! everything a burst needs: where, what colour, how hard. A `PlayerEvent` is a
//! bare enum and carries none of that, so the caller supplies the feet position
//! and the tint of whatever is underfoot — which is exactly what `Game.ts` did,
//! and why the presets are public.

use bevy::prelude::*;

use godgame_core::config::{STEP_DT, cell_at};
use godgame_core::entities::mobs::{MobEvent, MobEventKind};
use godgame_core::physics::collision::is_solid_cell;
use godgame_core::sim::grid::CellGrid;

use crate::lowres::WORLD_LAYERS;
use crate::player::PlayerSet;
use crate::world::{SimSet, SimWorld};

// ---------------------------------------------------------------------------
// Tuning. Every number below describes ONE algorithm — how a particle moves,
// how it fades, or what one named burst looks like — so it lives here and not
// in `godgame-core`'s config, which is for numbers that describe the whole game.
// ---------------------------------------------------------------------------

/// Slots in the pool, and therefore sprite entities the plugin spawns.
///
/// The TypeScript's default capacity, unchanged. It is about eight simultaneous
/// full-strength bursts, which is more than a screen ever shows at once; the
/// headroom is for the frame a player lands in lava next to three dying slimes.
pub const MAX_PARTICLES: usize = 2048;

/// Where ordinary particles sit in z: over the shots (0.55), under the brush
/// preview (1.0).
///
/// Over the shots deliberately. Debris is what tells the player that the shot
/// connected, and a spark behind the arrow that made it is a frame of feedback
/// spent on nothing.
const PARTICLE_Z: f32 = 0.6;

/// Where self-luminous particles sit: over the ordinary ones.
///
/// This is the geometric half of `drawGlow`. See the header for the half that
/// did not come across.
const GLOW_Z: f32 = 0.7;

/// Speed jitter applied to every particle, as a fraction of [`EmitOpts::speed`].
///
/// A burst whose members all leave at exactly one speed reads as a rigid
/// starburst — the eye finds the ring. `0.6..1.0` is enough to break it up
/// without making the burst look under-powered.
const SPEED_JITTER: (f32, f32) = (0.6, 1.0);

/// Lifetime jitter, as a fraction of [`EmitOpts::life`]. The same argument as
/// [`SPEED_JITTER`], applied to when the burst DISAPPEARS: particles that all
/// died on one frame would pop out as a unit.
const LIFE_JITTER: (f32, f32) = (0.75, 1.25);

/// How fast a wandering particle's drift phase advances, radians/s.
const WANDER_RATE: f32 = 2.3;

/// Frequency multiplier on the vertical half of the wander.
///
/// Deliberately not a whole ratio: at 1.7 the x and y drifts never come back
/// into phase, so a mote traces an open figure rather than closing a loop and
/// visibly repeating.
const WANDER_Y_RATE: f32 = 1.7;

/// How much weaker the vertical wander is than the horizontal.
///
/// Drift reads as air movement, and air moves sideways. An equal-weight vertical
/// component makes a firefly look like it is being shaken.
const WANDER_Y_SCALE: f32 = 0.6;

/// Reciprocal of the fraction of life a [`EmitOpts::fade_in`] particle spends
/// easing in — `5.0` is one fifth.
///
/// Impact debris must appear instantly: it is a reaction, and a reaction that
/// fades in is a reaction that arrived late. Ambient motes must not, because
/// something materialising at full opacity in front of the camera is the one
/// thing that gives away that it was spawned rather than drifted in.
const FADE_IN_RECIP: f32 = 5.0;

/// How much speed a colliding particle keeps when it bounces off a cell.
///
/// Low: this is grit and blood, not rubber. Enough that debris visibly scatters
/// off the ground rather than sticking to it on first contact, little enough
/// that it settles within a few tenths of a second under its own drag.
const BOUNCE: f32 = 0.35;

/// Impact debris: a wide upward-biased spray in the material's colour.
const BURST: Preset = Preset {
    count: 14,
    opts: EmitOpts {
        speed: 220.0,
        spread: core::f32::consts::PI * 1.6,
        life: 0.5,
        gravity: 1600.0,
        size: 3.0,
        drag: 1.5,
        collide: true,
        ..EmitOpts::DEFAULT
    },
};

/// Landing puff: low grey kick-up that spreads sideways and settles fast.
const DUST: Preset = Preset {
    count: 10,
    opts: EmitOpts {
        color: [170, 165, 155],
        speed: 90.0,
        spread: core::f32::consts::PI * 0.9,
        life: 0.4,
        gravity: 300.0,
        size: 3.0,
        drag: 3.0,
        collide: true,
        ..EmitOpts::DEFAULT
    },
};

/// Dash trail: a slow, near-static smear that lingers where the player was.
///
/// Full-circle spread and heavy drag, so it expands a little and then simply
/// hangs: this is a record of where the body WAS, not a thing being thrown.
const TRAIL: Preset = Preset {
    count: 4,
    opts: EmitOpts {
        speed: 40.0,
        spread: core::f32::consts::TAU,
        life: 0.35,
        gravity: 0.0,
        size: 4.0,
        drag: 4.0,
        ..EmitOpts::DEFAULT
    },
};

/// Liquid splash: a tight, fast upward jet that arcs back down under gravity.
const SPLASH: Preset = Preset {
    count: 12,
    opts: EmitOpts {
        speed: 260.0,
        spread: core::f32::consts::PI * 0.7,
        life: 0.6,
        gravity: 1900.0,
        size: 2.0,
        drag: 0.5,
        collide: true,
        ..EmitOpts::DEFAULT
    },
};

/// Ground puff, before [`ParticleSystem::puff`] scales it by strength.
///
/// The fields this preset does NOT set are the ones strength drives: count,
/// speed, spread, life and size are all functions of it. What is fixed is the
/// character — a low kick-up that settles fast — which is the same whether it
/// came off a footstep or off a two-storey drop.
const PUFF: Preset = Preset {
    count: 2,
    opts: EmitOpts {
        gravity: 260.0,
        drag: 3.0,
        collide: true,
        ..EmitOpts::DEFAULT
    },
};

/// Wall-slide scrape: grit shed off the wall face, thrown back and downward.
const SCRAPE: Preset = Preset {
    count: 2,
    opts: EmitOpts {
        speed: 90.0,
        spread: 0.8,
        life: 0.3,
        gravity: 700.0,
        size: 2.0,
        drag: 2.0,
        collide: true,
        ..EmitOpts::DEFAULT
    },
};

/// Dash smear: a short streak of near-static motes thrown OPPOSITE the travel
/// direction, so it reads as displaced air rather than as exhaust.
const SMEAR: Preset = Preset {
    count: 5,
    opts: EmitOpts {
        speed: 120.0,
        spread: 0.9,
        life: 0.28,
        gravity: 0.0,
        size: 3.0,
        drag: 5.0,
        glow: true,
        ..EmitOpts::DEFAULT
    },
};

/// Rising fire ember: buoyant, warm orange, and deliberately not solid.
const EMBER: Preset = Preset {
    count: 3,
    opts: EmitOpts {
        color: [255, 150, 40],
        speed: 60.0,
        spread: core::f32::consts::PI * 0.5,
        life: 0.9,
        // Negative: heat floats. This is the one preset whose gravity points the
        // other way, and the sign IS the effect.
        gravity: -260.0,
        size: 2.0,
        drag: 1.0,
        ..EmitOpts::DEFAULT
    },
};

/// Strength of the puff a creature's non-fatal hit throws up.
const PUFF_MOB_HURT: f32 = 0.4;

/// Strength of the puff the player's own wound throws up. Harder than a
/// creature's, because the one the player has to notice is their own.
const PUFF_PLAYER_HIT: f32 = 0.5;

/// Strength of the puff under a creature's death, on top of its splash.
const PUFF_MOB_DIE: f32 = 0.85;

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// Bit flags packed into the per-particle `flags` byte.
///
/// A byte rather than three `bool` arrays: they are read together, on the same
/// slot, in the same branch, and three parallel arrays would be three cache
/// lines to answer one question.
mod flag {
    /// Draw over the lit frame instead of in the world layer. See the header.
    pub const GLOW: u8 = 1;
    /// Ease alpha in as well as out.
    pub const FADE_IN: u8 = 2;
    /// Test against the cell grid on every step.
    pub const COLLIDE: u8 = 4;
}

/// Per-emit tuning. Velocities are px/s, life is seconds.
///
/// A `Copy` value rather than the TypeScript's optional-field object literal:
/// [`EmitOpts::DEFAULT`] is what its `??` chain spelled, and struct update
/// syntax is what its partial literals did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitOpts {
    /// Base particle colour, 0-255 per channel.
    pub color: [u8; 3],
    /// Mean launch speed, px/s. Jittered per particle by [`SPEED_JITTER`].
    pub speed: f32,
    /// Random angular fan, radians. 0 = straight along [`EmitOpts::angle`].
    pub spread: f32,
    /// Lifetime in seconds, jittered per particle by [`LIFE_JITTER`].
    pub life: f32,
    /// Downward acceleration, px/s^2. Negative floats particles up.
    pub gravity: f32,
    /// Square edge in world px.
    pub size: f32,
    /// Per-second velocity damping. 0 = frictionless.
    pub drag: f32,
    /// Base launch angle in radians. The default is straight up.
    ///
    /// Up is NEGATIVE here: this is a sim-space angle, and sim +y is down. The
    /// one place the two conventions meet is [`place_particles`], and nothing
    /// between here and there flips a sign.
    pub angle: f32,
    /// Sinusoidal drift acceleration, px/s^2. 0 leaves the particle ballistic —
    /// this is the wander that separates a drifting mote from thrown debris.
    pub wander: f32,
    /// Draw over the finished lit frame instead of in the world layer.
    ///
    /// Ambient life — fireflies, embers, spores, glints — has to survive the
    /// lighting multiply to read as self-luminous, so it opts in here.
    pub glow: bool,
    /// Ease alpha in as well as out. See [`FADE_IN_RECIP`].
    pub fade_in: bool,
    /// Bounce off solid cells instead of passing through them.
    ///
    /// Not in the TypeScript at all; see the header. Off by default, because the
    /// cheapest correct answer for a mote of light is that the world is not
    /// there.
    pub collide: bool,
}

impl EmitOpts {
    /// What the TypeScript's `??` defaults spelled: a white, ballistic,
    /// frictionless particle launched straight up.
    pub const DEFAULT: EmitOpts = EmitOpts {
        color: [255, 255, 255],
        speed: 0.0,
        spread: 0.0,
        life: 0.0,
        gravity: 0.0,
        size: 1.0,
        drag: 0.0,
        angle: -core::f32::consts::FRAC_PI_2,
        wander: 0.0,
        glow: false,
        fade_in: false,
        collide: false,
    };
}

impl Default for EmitOpts {
    fn default() -> EmitOpts {
        EmitOpts::DEFAULT
    }
}

/// A named burst: how many particles, and what each one is.
///
/// The TypeScript kept the count as an argument and the rest as an object, so
/// each preset method restated both. Binding them means a preset is ONE
/// doc-commented constant that says what the effect is, and the methods below
/// are the three lines that tint it and fire it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    /// How many particles one call emits.
    pub count: usize,
    /// What each of them is.
    pub opts: EmitOpts,
}

impl Preset {
    /// The same burst in a different colour.
    ///
    /// Every preset that represents matter takes its colour from the material it
    /// came off — the falling-sand world already knows what you are standing in,
    /// and asking it is one array read — so tinting is the common case and a
    /// baked colour is the exception.
    #[inline]
    pub fn tinted(self, color: [u8; 3]) -> Preset {
        Preset {
            opts: EmitOpts { color, ..self.opts },
            ..self
        }
    }
}

/// One live particle, as a renderer needs it.
///
/// The alpha is already resolved — the fade envelope is the pool's business and
/// not the sprite's — and the position is NOT rounded, because which pixel grid
/// to snap to is a decision only the thing doing the drawing can make.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    /// Top-left of the drawn square, world px.
    pub x: f32,
    /// Top-left of the drawn square, world px. Y grows DOWN.
    pub y: f32,
    /// Square edge, world px.
    pub size: f32,
    /// Colour, 0-255 per channel.
    pub rgb: [u8; 3],
    /// Fade envelope, 0..1.
    pub alpha: f32,
    /// Whether this one belongs in the glow pass.
    pub glow: bool,
}

/// The render side's own random stream.
///
/// Deliberately not the sim's [`SimRng`](godgame_core::sim::rng::SimRng) and not
/// the creatures' [`MobRng`](godgame_core::entities::mobs::MobRng): both of those
/// are consumed in a fixed ORDER that a replay of the same seed has to reproduce
/// exactly, and a spark drawn from either would desync the world against a
/// change in how much juice happened to be on screen. Juice must never be able
/// to move a cell.
///
/// It is xorshift32 with a different default state, which is the same answer
/// `MobRng` gave to the same question one layer down.
///
/// Shared with [`crate::effects`], which shakes the screen on the same events
/// this pool bursts on and has the same reason not to reach for a sim stream.
#[derive(Clone, Copy, Debug)]
pub struct JuiceRng {
    state: u32,
}

impl JuiceRng {
    /// The state a fresh stream starts from. Xorshift is stuck at zero, so it
    /// can never be that.
    pub const DEFAULT_STATE: u32 = 0x9e37_79b9;

    /// A stream at the default state.
    #[inline]
    pub fn new() -> JuiceRng {
        JuiceRng {
            state: JuiceRng::DEFAULT_STATE,
        }
    }

    /// A stream reseeded to `seed`. A zero seed falls back to the default state.
    #[inline]
    pub fn seeded(seed: u32) -> JuiceRng {
        JuiceRng {
            state: if seed != 0 {
                seed
            } else {
                JuiceRng::DEFAULT_STATE
            },
        }
    }

    /// Next raw 32-bit value.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform float in `[0, 1)` — the `Math.random()` every call site used.
    ///
    /// The `f64` multiply is deliberate, and is `MobRng`'s: 24 bits of `f32`
    /// mantissa cannot represent the whole `u32` range, so doing the divide at
    /// `f32` would quantise the stream itself. The result narrows immediately,
    /// because everything it feeds is `f32`.
    #[inline]
    pub fn rand(&mut self) -> f32 {
        (f64::from(self.next_u32()) * 2.328_306_436_538_696_3e-10) as f32
    }

    /// Uniform float in `[a, b)`.
    #[inline]
    pub fn rand_range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.rand()
    }
}

impl Default for JuiceRng {
    #[inline]
    fn default() -> JuiceRng {
        JuiceRng::new()
    }
}

/// The particle pool.
///
/// Struct-of-arrays, allocated once, indexed by slot. See the header for why it
/// is this shape and why nothing grows it.
#[derive(Resource)]
pub struct ParticleSystem {
    // --- SoA. Parallel; index i is one particle. Never resized. ---
    px: [f32; MAX_PARTICLES],
    py: [f32; MAX_PARTICLES],
    vx: [f32; MAX_PARTICLES],
    vy: [f32; MAX_PARTICLES],
    /// Remaining seconds. `<= 0` is the whole definition of a dead slot — there
    /// is no separate live flag that could fall out of step with it.
    life: [f32; MAX_PARTICLES],
    /// Life at birth, for the fade envelope.
    max_life: [f32; MAX_PARTICLES],
    size: [f32; MAX_PARTICLES],
    gravity: [f32; MAX_PARTICLES],
    drag: [f32; MAX_PARTICLES],
    r: [u8; MAX_PARTICLES],
    g: [u8; MAX_PARTICLES],
    b: [u8; MAX_PARTICLES],
    /// Drift acceleration. 0 = ballistic, and the branch that skips the two trig
    /// calls tests exactly this.
    wander_a: [f32; MAX_PARTICLES],
    /// Drift phase, advanced by the step.
    phase: [f32; MAX_PARTICLES],
    flags: [u8; MAX_PARTICLES],

    /// Ring cursor: where the next scan for a free slot begins. Claims and
    /// deaths are spread evenly over the array, so resuming where the last claim
    /// finished is amortised O(1) against an O(n) walk from zero every time.
    cursor: usize,
    /// Live particles, maintained incrementally — never a scan.
    live: usize,
    /// Live particles carrying [`flag::GLOW`], so a caller can skip the glow
    /// pass outright. Kept from the TypeScript because the pass it guards is
    /// coming back; see the header.
    glowing: usize,
    /// The stream every jitter above is drawn from.
    rng: JuiceRng,
}

impl ParticleSystem {
    /// An empty pool.
    pub fn new() -> ParticleSystem {
        ParticleSystem {
            px: [0.0; MAX_PARTICLES],
            py: [0.0; MAX_PARTICLES],
            vx: [0.0; MAX_PARTICLES],
            vy: [0.0; MAX_PARTICLES],
            life: [0.0; MAX_PARTICLES],
            max_life: [0.0; MAX_PARTICLES],
            size: [0.0; MAX_PARTICLES],
            gravity: [0.0; MAX_PARTICLES],
            drag: [0.0; MAX_PARTICLES],
            r: [0; MAX_PARTICLES],
            g: [0; MAX_PARTICLES],
            b: [0; MAX_PARTICLES],
            wander_a: [0.0; MAX_PARTICLES],
            phase: [0.0; MAX_PARTICLES],
            flags: [0; MAX_PARTICLES],
            cursor: 0,
            live: 0,
            glowing: 0,
            rng: JuiceRng::new(),
        }
    }

    /// An empty pool whose jitter comes off `seed`, for a test that wants a
    /// different burst than the default stream's first one.
    pub fn seeded(seed: u32) -> ParticleSystem {
        let mut p = ParticleSystem::new();
        p.rng = JuiceRng::seeded(seed);
        p
    }

    /// How many particles are up.
    #[inline]
    pub fn live_count(&self) -> usize {
        self.live
    }

    /// How many of those are self-luminous. Zero means the glow pass has nothing
    /// to do.
    #[inline]
    pub fn glow_count(&self) -> usize {
        self.glowing
    }

    /// What is in slot `i`, or `None` if the slot is dead.
    ///
    /// Indexed rather than published as an iterator — which is what
    /// [`ProjectileSystem::shots`](godgame_core::entities::projectiles::ProjectileSystem::shots)
    /// does — because the consumer is a sprite POOL: entity `i` mirrors slot `i`
    /// for the lifetime of the run, so what it needs to ask is "what is in my
    /// slot", not "walk me through the live set". An iterator would deliver the
    /// answer in an order that has nothing to do with which entity is asking.
    #[inline]
    pub fn slot(&self, i: usize) -> Option<Particle> {
        let life = self.life[i];
        if life <= 0.0 {
            return None;
        }
        Some(Particle {
            x: self.px[i],
            y: self.py[i],
            size: self.size[i],
            rgb: [self.r[i], self.g[i], self.b[i]],
            alpha: self.alpha_of(i, life),
            glow: self.flags[i] & flag::GLOW != 0,
        })
    }

    /// Spawn `count` particles at world `(x, y)` with randomised velocity.
    ///
    /// Overflow is DROPPED, not recycled — see the header. The loop returns on
    /// the first refusal rather than continuing to try, because a pool that had
    /// no slot for particle three has none for particle four either.
    pub fn emit(&mut self, x: f32, y: f32, count: usize, opts: &EmitOpts) {
        let mut flags = 0u8;
        if opts.glow {
            flags |= flag::GLOW;
        }
        if opts.fade_in {
            flags |= flag::FADE_IN;
        }
        if opts.collide {
            flags |= flag::COLLIDE;
        }

        for _ in 0..count {
            let Some(i) = self.claim() else { return };

            // Fan the launch angle across the spread and jitter speed and life,
            // so a burst does not read as one rigid object.
            let a = opts.angle + (self.rng.rand() - 0.5) * opts.spread;
            let s = opts.speed * self.rng.rand_range(SPEED_JITTER.0, SPEED_JITTER.1);
            let l = opts.life * self.rng.rand_range(LIFE_JITTER.0, LIFE_JITTER.1);

            self.px[i] = x;
            self.py[i] = y;
            self.vx[i] = a.cos() * s;
            self.vy[i] = a.sin() * s;
            self.life[i] = l;
            self.max_life[i] = l;
            self.size[i] = opts.size;
            self.gravity[i] = opts.gravity;
            self.drag[i] = opts.drag;
            self.r[i] = opts.color[0];
            self.g[i] = opts.color[1];
            self.b[i] = opts.color[2];
            self.wander_a[i] = opts.wander;
            self.phase[i] = self.rng.rand() * core::f32::consts::TAU;
            self.flags[i] = flags;

            self.live += 1;
            if flags & flag::GLOW != 0 {
                self.glowing += 1;
            }
        }
    }

    /// Emit one whole [`Preset`].
    pub fn emit_preset(&mut self, x: f32, y: f32, preset: &Preset) {
        self.emit(x, y, preset.count, &preset.opts);
    }

    /// Impact debris in the material's own colour.
    pub fn burst(&mut self, x: f32, y: f32, color: [u8; 3]) {
        self.emit_preset(x, y, &BURST.tinted(color));
    }

    /// Landing puff, in the preset's own grey. See [`ParticleSystem::puff`] for
    /// the one that takes the colour of what is actually underfoot.
    pub fn dust(&mut self, x: f32, y: f32) {
        self.emit_preset(x, y, &DUST);
    }

    /// Dash trail.
    pub fn trail(&mut self, x: f32, y: f32, color: [u8; 3]) {
        self.emit_preset(x, y, &TRAIL.tinted(color));
    }

    /// Liquid splash.
    pub fn splash(&mut self, x: f32, y: f32, color: [u8; 3]) {
        self.emit_preset(x, y, &SPLASH.tinted(color));
    }

    /// Ground puff in the colour of whatever is underfoot.
    ///
    /// `strength` in `[0, 1]` scales count, speed, spread, life and size
    /// together, so one call covers everything from a footfall (0.15) to a hard
    /// landing (1) without four near-identical presets that would then have to
    /// be kept looking related by hand.
    pub fn puff(&mut self, x: f32, y: f32, color: [u8; 3], strength: f32) {
        // The floor is not defensive padding: a caller computing strength from a
        // touchdown speed can legitimately reach zero, and it should still be
        // visible that something landed.
        let s = strength.clamp(0.05, 1.0);
        let mut preset = PUFF.tinted(color);
        preset.count += (s * 12.0) as usize;
        preset.opts.speed = 50.0 + s * 110.0;
        preset.opts.spread = core::f32::consts::PI * (0.5 + s * 0.5);
        preset.opts.life = 0.25 + s * 0.35;
        // Two sizes, not a continuum: at 5px cells a 2px mote and a 3px mote are
        // the only two things a puff can usefully be.
        preset.opts.size = if s > 0.5 { 3.0 } else { 2.0 };
        self.emit_preset(x, y, &preset);
    }

    /// Wall-slide scrape. `dir` is the side the wall is on: positive is right.
    ///
    /// The grit leaves away from the wall and slightly up, because it is being
    /// shaved off a face the body is pressed against.
    pub fn scrape(&mut self, x: f32, y: f32, dir: f32, color: [u8; 3]) {
        let mut preset = SCRAPE.tinted(color);
        preset.opts.angle = if dir > 0.0 {
            -core::f32::consts::PI * 0.75
        } else {
            -core::f32::consts::PI * 0.25
        };
        self.emit_preset(x, y, &preset);
    }

    /// Dash smear. `dir` is the direction of travel; the motes go the other way.
    pub fn smear(&mut self, x: f32, y: f32, dir: f32, color: [u8; 3]) {
        let mut preset = SMEAR.tinted(color);
        preset.opts.angle = if dir > 0.0 {
            core::f32::consts::PI
        } else {
            0.0
        };
        self.emit_preset(x, y, &preset);
    }

    /// Rising fire ember.
    pub fn ember(&mut self, x: f32, y: f32) {
        self.emit_preset(x, y, &EMBER);
    }

    /// One creature event's worth of juice.
    ///
    /// This is the whole of the mobs-to-particles seam, and it is one call
    /// because a [`MobEvent`] already carries where, what colour and how hard.
    /// The colour comes off the creature's own blood tone, so a slime bursts
    /// green and an emberling bursts orange without this function owning a
    /// palette or a match on species.
    ///
    /// Add it to `crate::mobs::step_creatures`, which currently drains these
    /// events and throws them away. The shake and the screen flash that go with
    /// the same events are [`crate::effects::Feedback::mob_event`]'s half.
    pub fn mob_event(&mut self, e: MobEvent) {
        match e.kind {
            MobEventKind::PlayerHit => self.puff(e.x, e.y, e.rgb, PUFF_PLAYER_HIT),
            MobEventKind::MobHurt => self.puff(e.x, e.y, e.rgb, PUFF_MOB_HURT),
            MobEventKind::MobDie => {
                // Both, and in this order: the splash throws blood up and out,
                // the puff is the body coming down under it.
                self.splash(e.x, e.y, e.rgb);
                self.puff(e.x, e.y, e.rgb, PUFF_MOB_DIE);
            }
            // A chill is a status, not an impact. It has a duration in `power`
            // and no place to burst; when it gets a look it will be a tint on the
            // body, which is `crate::player`'s to draw and not a particle.
            MobEventKind::PlayerChill => {}
        }
    }

    /// Integrate every live particle and free the ones that have aged out.
    ///
    /// `grid` is what particles collide against. `None` means there is no world
    /// to hit — every particle is ballistic, which is the TypeScript's behaviour
    /// exactly, and is what the tests below run against so that a collision test
    /// is testing collision and nothing else.
    pub fn update(&mut self, dt: f32, grid: Option<&CellGrid>) {
        for i in 0..MAX_PARTICLES {
            let l = self.life[i];
            if l <= 0.0 {
                continue;
            }

            let nl = l - dt;
            if nl <= 0.0 {
                self.retire(i);
                continue;
            }
            self.life[i] = nl;

            // Symplectic-ish Euler: gravity, then wander, then drag, then
            // position. The order is the TypeScript's and it matters — drag
            // applied before gravity would let a heavy particle out-accelerate
            // its own damping on the first step.
            let mut vx = self.vx[i];
            let mut vy = self.vy[i] + self.gravity[i] * dt;

            // Two trig calls, and only for the ambient particles that ask.
            let w = self.wander_a[i];
            if w > 0.0 {
                let ph = self.phase[i] + dt * WANDER_RATE;
                self.phase[i] = ph;
                vx += ph.cos() * w * dt;
                vy += (ph * WANDER_Y_RATE).sin() * w * WANDER_Y_SCALE * dt;
            }

            let d = self.drag[i];
            if d > 0.0 {
                // Frame-rate independent enough for cosmetics, and clamped at
                // zero so a large `drag * dt` damps to a stop rather than
                // reversing the particle.
                let k = 1.0 - d * dt;
                let m = if k > 0.0 { k } else { 0.0 };
                vx *= m;
                vy *= m;
            }

            let (x, y) = (self.px[i], self.py[i]);
            let (mut nx, mut ny) = (x + vx * dt, y + vy * dt);

            if let Some(grid) = grid
                && self.flags[i] & flag::COLLIDE != 0
                // A particle ALREADY inside rock is not colliding with it — it
                // was emitted there, or the automata closed over it — and
                // testing it would pin it in place until it died. Skipping the
                // test lets it fly back out on its own.
                && !is_solid_cell(grid, cell_at(x), cell_at(y))
            {
                // One axis at a time, X then Y, which is the rule
                // `physics::collision` resolves bodies with. Resolving both at
                // once against a single sample makes a particle that clips a
                // corner bounce off a cell neither axis actually entered.
                if is_solid_cell(grid, cell_at(nx), cell_at(y)) {
                    nx = x;
                    vx = -vx * BOUNCE;
                }
                if is_solid_cell(grid, cell_at(nx), cell_at(ny)) {
                    ny = y;
                    vy = -vy * BOUNCE;
                }
            }

            self.vx[i] = vx;
            self.vy[i] = vy;
            self.px[i] = nx;
            self.py[i] = ny;
        }
    }

    /// Recycle everything — a respawn, a level reload.
    pub fn clear(&mut self) {
        self.life = [0.0; MAX_PARTICLES];
        self.flags = [0; MAX_PARTICLES];
        self.live = 0;
        self.glowing = 0;
    }

    /// Fade envelope: 1 at birth, 0 at death.
    ///
    /// Debris fades out linearly over its life. A [`flag::FADE_IN`] particle
    /// additionally eases in over the first fifth, so ambient motes materialise
    /// instead of popping.
    fn alpha_of(&self, i: usize, life: f32) -> f32 {
        let r = life / self.max_life[i];
        if self.flags[i] & flag::FADE_IN == 0 {
            return r;
        }
        let rise = (1.0 - r) * FADE_IN_RECIP;
        r * if rise < 1.0 { rise } else { 1.0 }
    }

    /// A free slot from the ring cursor, or `None` if the pool is saturated.
    fn claim(&mut self) -> Option<usize> {
        for _ in 0..MAX_PARTICLES {
            let i = self.cursor;
            self.cursor = if self.cursor + 1 >= MAX_PARTICLES {
                0
            } else {
                self.cursor + 1
            };
            if self.life[i] <= 0.0 {
                return Some(i);
            }
        }
        None
    }

    /// Return a slot to the pool. Only ever called on a live slot, from the one
    /// place a particle can die.
    fn retire(&mut self, i: usize) {
        self.life[i] = 0.0;
        if self.flags[i] & flag::GLOW != 0 {
            self.glowing -= 1;
        }
        self.flags[i] = 0;
        self.live -= 1;
    }
}

impl Default for ParticleSystem {
    fn default() -> ParticleSystem {
        ParticleSystem::new()
    }
}

impl core::fmt::Debug for ParticleSystem {
    /// The arrays are 2048 slots of mostly zeros; the only thing worth printing
    /// is how much is up.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ParticleSystem")
            .field("live", &self.live)
            .field("glowing", &self.glowing)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// The Bevy half
// ---------------------------------------------------------------------------

/// One pooled particle sprite. `slot` indexes the pool it mirrors.
#[derive(Component, Clone, Copy)]
pub struct ParticleSprite {
    /// Index into the [`ParticleSystem`] arrays.
    pub slot: usize,
}

/// The pool, its fixed step, and [`MAX_PARTICLES`] sprites to draw it with.
///
/// Deliberately not in [`GodGameRenderPlugin`](crate::GodGameRenderPlugin) yet:
/// until the two seams in the header are wired, adding it would spawn 2048
/// entities to draw nothing.
pub struct ParticlesPlugin;

impl Plugin for ParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleSystem>()
            .add_systems(Startup, spawn_pool)
            .add_systems(
                FixedUpdate,
                step_particles
                    // After the body and the creatures have both taken their
                    // step, so a particle emitted by either is integrated on the
                    // step it was born on rather than sitting still for one; and
                    // after the automata, so the cells it collides against are
                    // this step's and not the previous one's.
                    .after(PlayerSet::Step)
                    .after(SimSet::Simulate)
                    .run_if(resource_exists::<SimWorld>),
            )
            .add_systems(Update, place_particles);
    }
}

/// One hidden entity per slot, once. See the header on why they are pooled.
fn spawn_pool(mut commands: Commands) {
    for slot in 0..MAX_PARTICLES {
        commands.spawn((
            Sprite {
                custom_size: Some(Vec2::ZERO),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, PARTICLE_Z),
            ParticleSprite { slot },
            Visibility::Hidden,
            WORLD_LAYERS,
        ));
    }
}

/// One fixed step of every live particle, against the world it can hit.
fn step_particles(mut particles: ResMut<ParticleSystem>, world: Res<SimWorld>) {
    particles.update(STEP_DT, Some(&world.level.grid));
}

/// Put every sprite on its slot, or hide it.
///
/// The TypeScript drew `fillRect(round(px), round(py), s, s)`: the TOP-LEFT
/// rounded to a whole world pixel, so a square stays crisp against the cell art
/// instead of resampling. A Bevy sprite is positioned by its CENTRE, so the
/// round happens on the corner and the half-extent is added afterwards — the
/// same rule [`crate::mobs`] places a body box with.
fn place_particles(
    particles: Res<ParticleSystem>,
    mut sprites: Query<(
        &ParticleSprite,
        &mut Sprite,
        &mut Transform,
        &mut Visibility,
    )>,
) {
    for (which, mut sprite, mut transform, mut visibility) in &mut sprites {
        let Some(p) = particles.slot(which.slot) else {
            *visibility = Visibility::Hidden;
            continue;
        };
        *visibility = Visibility::Inherited;
        sprite.color = Color::srgb_u8(p.rgb[0], p.rgb[1], p.rgb[2]).with_alpha(p.alpha);
        sprite.custom_size = Some(Vec2::splat(p.size));
        let half = p.size * 0.5;
        transform.translation.x = p.x.round() + half;
        // +y is up in Bevy and down in the sim: the one convention flip, in the
        // one place, exactly as `crate::player`'s body placement does it.
        transform.translation.y = -(p.y.round() + half);
        // Glow particles sort above the rest. This is all that is left of
        // `drawGlow`'s second pass — see the header.
        transform.translation.z = if p.glow { GLOW_Z } else { PARTICLE_Z };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::config::CELL_SIZE;
    use godgame_core::sim::materials::{EMPTY, block};

    /// The step the plugin drives the pool at.
    const DT: f32 = STEP_DT;

    /// An all-air grid with a stone floor from `floor_row` down.
    fn grid_with_floor(floor_row: i32) -> CellGrid {
        let mut g = CellGrid::new(64, 64);
        for y in 0..g.rows() {
            for x in 0..g.cols() {
                g.set(x, y, if y >= floor_row { block::STONE } else { EMPTY });
            }
        }
        g
    }

    /// One preset under test: what it is called, the call that fires it, and how
    /// many particles it promises.
    type PresetCase = (&'static str, fn(&mut ParticleSystem), usize);

    /// Steps needed to outlive `life` at its longest jitter, plus one.
    fn steps_to_outlive(life: f32) -> u32 {
        (life * LIFE_JITTER.1 / DT).ceil() as u32 + 1
    }

    /// A long-lived, otherwise inert particle to hang one property off.
    fn probe(opts: EmitOpts) -> EmitOpts {
        EmitOpts { life: 5.0, ..opts }
    }

    /// A particle that does nothing but age. Most of what is tested below is
    /// the POOL, and a pool does not care what is in the slot.
    fn lasting(life: f32) -> EmitOpts {
        EmitOpts {
            life,
            ..EmitOpts::DEFAULT
        }
    }

    #[test]
    fn a_fresh_pool_is_empty() {
        let p = ParticleSystem::new();
        assert_eq!(p.live_count(), 0);
        assert_eq!(p.glow_count(), 0);
        assert!(p.slot(0).is_none());
        assert!(p.slot(MAX_PARTICLES - 1).is_none());
    }

    #[test]
    fn an_emit_fills_as_many_slots_as_it_was_asked_for() {
        let mut p = ParticleSystem::new();
        p.emit(10.0, 20.0, 7, &probe(EmitOpts::DEFAULT));
        assert_eq!(p.live_count(), 7);
        assert_eq!(
            (0..MAX_PARTICLES).filter(|&i| p.slot(i).is_some()).count(),
            7,
            "the live count and the live slots disagree"
        );
        let born = p.slot(0).unwrap();
        assert_eq!((born.x, born.y), (10.0, 20.0));
    }

    #[test]
    fn a_full_pool_drops_the_overflow_rather_than_growing() {
        // Deliberately NOT the projectile pool's oldest-recycled rule. See the
        // header: an arrow is spent ammo and a spark is not.
        let mut p = ParticleSystem::new();
        let opts = lasting(10.0);
        p.emit(0.0, 0.0, MAX_PARTICLES, &opts);
        assert_eq!(p.live_count(), MAX_PARTICLES);

        // The first particle is still the first particle afterwards: nothing was
        // evicted to make room for the overflow.
        let first = p.slot(0).expect("slot 0 died");
        p.emit(999.0, 999.0, 64, &opts);
        assert_eq!(p.live_count(), MAX_PARTICLES, "the pool grew");
        assert_eq!(p.slot(0), Some(first), "an overflow evicted a live slot");
    }

    #[test]
    fn a_particle_dies_when_its_life_runs_out_and_gives_its_slot_back() {
        let mut p = ParticleSystem::new();
        p.emit(0.0, 0.0, 1, &lasting(0.1));
        assert_eq!(p.live_count(), 1);

        for _ in 0..steps_to_outlive(0.1) {
            p.update(DT, None);
        }
        assert_eq!(p.live_count(), 0, "outlived its jittered life");
        assert!(p.slot(0).is_none());

        // And the slot is usable again, which is the whole point of the pool.
        p.emit(5.0, 6.0, 1, &probe(EmitOpts::DEFAULT));
        assert_eq!(p.live_count(), 1);
    }

    #[test]
    fn the_cursor_reuses_freed_slots_without_scanning_from_zero() {
        let mut p = ParticleSystem::new();
        // Fill the pool with sixteen short-lived particles and a long-lived
        // remainder, then age out only the first group.
        p.emit(0.0, 0.0, 16, &lasting(0.05));
        p.emit(0.0, 0.0, MAX_PARTICLES - 16, &lasting(100.0));
        assert_eq!(p.live_count(), MAX_PARTICLES);

        for _ in 0..steps_to_outlive(0.05) {
            p.update(DT, None);
        }
        let freed = MAX_PARTICLES - p.live_count();
        assert_eq!(freed, 16, "the short-lived ones did not all die");

        // Exactly `freed` more fit, and not one over: the scan found every
        // recycled slot and invented none.
        p.emit(1.0, 1.0, freed + 8, &lasting(100.0));
        assert_eq!(p.live_count(), MAX_PARTICLES);
    }

    #[test]
    fn gravity_pulls_debris_down_and_floats_an_ember_up() {
        // Sim +y is DOWN, so "down" is an increasing y and the ember's negative
        // gravity is what makes it rise. This is the one place the sign of that
        // convention is worth asserting on directly.
        // Launched flat, so every bit of vertical motion is the gravity's.
        let flat = EmitOpts {
            angle: 0.0,
            ..EmitOpts::DEFAULT
        };
        let mut heavy = ParticleSystem::new();
        heavy.emit(
            0.0,
            0.0,
            1,
            &probe(EmitOpts {
                gravity: 1000.0,
                ..flat
            }),
        );
        let mut light = ParticleSystem::new();
        light.emit(
            0.0,
            0.0,
            1,
            &probe(EmitOpts {
                gravity: -1000.0,
                ..flat
            }),
        );
        for _ in 0..12 {
            heavy.update(DT, None);
            light.update(DT, None);
        }
        assert!(heavy.slot(0).unwrap().y > 0.0, "gravity did not pull down");
        assert!(light.slot(0).unwrap().y < 0.0, "buoyancy did not lift");
    }

    #[test]
    fn drag_takes_speed_out_and_never_reverses_the_particle() {
        // A drag high enough that `1 - drag * dt` would go negative: the clamp is
        // what stops damping from becoming a reflection.
        let mut p = ParticleSystem::new();
        let damped = EmitOpts {
            speed: 400.0,
            angle: 0.0,
            drag: 500.0,
            ..probe(EmitOpts::DEFAULT)
        };
        p.emit(0.0, 0.0, 1, &damped);
        let x0 = p.slot(0).unwrap().x;
        p.update(DT, None);
        let x1 = p.slot(0).unwrap().x;
        p.update(DT, None);
        let x2 = p.slot(0).unwrap().x;
        assert!(x1 >= x0, "the first step went backwards");
        assert_eq!(x2, x1, "a fully damped particle kept moving");
    }

    #[test]
    fn alpha_fades_linearly_over_a_particles_life() {
        let mut p = ParticleSystem::new();
        p.emit(0.0, 0.0, 1, &lasting(1.0));
        let born = p.slot(0).unwrap().alpha;
        assert!((born - 1.0).abs() < 1e-6, "not opaque at birth: {born}");

        let max_life = p.max_life[0];
        for _ in 0..((max_life * 0.5 / DT) as u32) {
            p.update(DT, None);
        }
        let half = p.slot(0).unwrap().alpha;
        assert!((half - 0.5).abs() < 0.02, "half-life alpha was {half}");
    }

    #[test]
    fn a_fade_in_particle_eases_in_over_the_first_fifth_and_out_over_the_rest() {
        let mut p = ParticleSystem::new();
        let mote = EmitOpts {
            fade_in: true,
            ..lasting(1.0)
        };
        p.emit(0.0, 0.0, 1, &mote);
        // Born invisible — the whole point of the flag is not popping in.
        assert_eq!(p.slot(0).unwrap().alpha, 0.0);

        let max_life = p.max_life[0];
        let mut peak: f32 = 0.0;
        let mut peak_at = 0.0;
        for n in 0..((max_life / DT) as u32) {
            p.update(DT, None);
            if let Some(s) = p.slot(0)
                && s.alpha > peak
            {
                peak = s.alpha;
                peak_at = (n + 1) as f32 * DT / max_life;
            }
        }
        // It peaks at the end of the ramp-in — one fifth of the way through —
        // and falls from there.
        assert!(peak > 0.75, "never became visible enough: {peak}");
        assert!(
            (peak_at - 0.2).abs() < 0.05,
            "peaked at {peak_at} of its life, not a fifth in"
        );
    }

    #[test]
    fn the_glow_count_tracks_only_the_luminous_particles() {
        let mut p = ParticleSystem::new();
        let luminous = EmitOpts {
            glow: true,
            ..lasting(0.1)
        };
        p.emit(0.0, 0.0, 3, &luminous);
        p.emit(0.0, 0.0, 5, &lasting(100.0));
        assert_eq!(p.live_count(), 8);
        assert_eq!(p.glow_count(), 3);
        assert!(p.slot(0).unwrap().glow);
        assert!(!p.slot(3).unwrap().glow);

        // The counter comes back down when they die, or the pass it guards would
        // run forever over an empty set.
        for _ in 0..steps_to_outlive(0.1) {
            p.update(DT, None);
        }
        assert_eq!(p.glow_count(), 0);
        assert_eq!(p.live_count(), 5);
    }

    #[test]
    fn a_colliding_particle_bounces_off_a_solid_cell_instead_of_entering_it() {
        let floor = 10;
        let g = grid_with_floor(floor);
        let floor_y = (floor * CELL_SIZE) as f32;

        let mut p = ParticleSystem::new();
        // Straight down at the floor from two cells up, no drag and no gravity,
        // so the only thing that can change its velocity is the cell.
        let thrown = EmitOpts {
            speed: 300.0,
            angle: core::f32::consts::FRAC_PI_2,
            collide: true,
            ..lasting(10.0)
        };
        p.emit(20.0, floor_y - 10.0, 1, &thrown);
        let launch_vy = p.vy[0];
        assert!(launch_vy > 0.0, "not aimed at the floor");

        for _ in 0..60 {
            p.update(DT, Some(&g));
            if p.vy[0] < 0.0 {
                break;
            }
        }
        assert!(p.vy[0] < 0.0, "never bounced: vy {}", p.vy[0]);
        assert!(
            p.slot(0).unwrap().y < floor_y,
            "ended up inside the floor at {}",
            p.slot(0).unwrap().y
        );
        assert!(
            p.vy[0].abs() < launch_vy,
            "bounced back at least as fast as it arrived"
        );
    }

    #[test]
    fn a_ballistic_particle_passes_straight_through_rock() {
        // The TypeScript's behaviour, preserved for everything that does not opt
        // in. An ember stopping dead against a ceiling is the bug this avoids.
        let g = grid_with_floor(10);
        let floor_y = (10 * CELL_SIZE) as f32;

        let mut p = ParticleSystem::new();
        let thrown = EmitOpts {
            speed: 300.0,
            angle: core::f32::consts::FRAC_PI_2,
            ..lasting(10.0)
        };
        p.emit(20.0, floor_y - 10.0, 1, &thrown);
        for _ in 0..60 {
            p.update(DT, Some(&g));
        }
        assert!(
            p.slot(0).unwrap().y > floor_y,
            "a ballistic particle was stopped by the world"
        );
    }

    #[test]
    fn a_particle_emitted_inside_rock_is_not_trapped_by_it() {
        // Emission points come off a body's feet and a creature's centre, both of
        // which can be a pixel inside a cell. A collision test that pinned those
        // would turn every landing puff into a stationary clump.
        let g = grid_with_floor(10);
        let floor_y = (10 * CELL_SIZE) as f32;

        let mut p = ParticleSystem::new();
        let buried = EmitOpts {
            speed: 300.0,
            // Straight up, out of the rock it starts in.
            angle: -core::f32::consts::FRAC_PI_2,
            collide: true,
            ..lasting(10.0)
        };
        p.emit(20.0, floor_y + 6.0, 1, &buried);
        for _ in 0..60 {
            p.update(DT, Some(&g));
        }
        assert!(
            p.slot(0).unwrap().y < floor_y,
            "stayed buried at {}",
            p.slot(0).unwrap().y
        );
    }

    #[test]
    fn an_unloaded_cell_stops_a_colliding_particle_leaving_the_window() {
        // `is_solid_cell` reports unloaded cells as solid — the same guarantee
        // the shots and the mobs rely on. Debris outside the streaming window
        // would be drawn against cells nobody has generated.
        let g = CellGrid::new(64, 64);
        let mut p = ParticleSystem::new();
        let outbound = EmitOpts {
            speed: 400.0,
            angle: core::f32::consts::PI, // straight at the left edge
            collide: true,
            ..lasting(10.0)
        };
        p.emit(160.0, 160.0, 1, &outbound);
        for _ in 0..600 {
            p.update(DT, Some(&g));
        }
        assert!(p.slot(0).unwrap().x > 0.0, "escaped the window");
    }

    #[test]
    fn wander_costs_its_two_trig_calls_only_when_it_is_asked_for() {
        let mut still = ParticleSystem::new();
        still.emit(0.0, 0.0, 1, &probe(EmitOpts::DEFAULT));
        let phase0 = still.phase[0];
        still.update(DT, None);
        assert_eq!(still.phase[0], phase0, "a ballistic particle drifted");

        let mut drifting = ParticleSystem::new();
        let mote = EmitOpts {
            wander: 200.0,
            ..probe(EmitOpts::DEFAULT)
        };
        drifting.emit(0.0, 0.0, 1, &mote);
        let before = drifting.phase[0];
        drifting.update(DT, None);
        assert!((drifting.phase[0] - before - DT * WANDER_RATE).abs() < 1e-6);
        assert_ne!(drifting.vx[0], 0.0, "wander did not push sideways");
    }

    #[test]
    fn every_preset_emits_its_own_count_of_drawable_particles() {
        // One pool per preset, so a preset that quietly emitted nothing cannot
        // hide behind another one's particles.
        let cases: [PresetCase; 8] = [
            ("burst", |p| p.burst(0.0, 0.0, [1, 2, 3]), BURST.count),
            ("dust", |p| p.dust(0.0, 0.0), DUST.count),
            ("trail", |p| p.trail(0.0, 0.0, [1, 2, 3]), TRAIL.count),
            ("splash", |p| p.splash(0.0, 0.0, [1, 2, 3]), SPLASH.count),
            (
                "scrape",
                |p| p.scrape(0.0, 0.0, 1.0, [1, 2, 3]),
                SCRAPE.count,
            ),
            ("smear", |p| p.smear(0.0, 0.0, 1.0, [1, 2, 3]), SMEAR.count),
            ("ember", |p| p.ember(0.0, 0.0), EMBER.count),
            ("puff", |p| p.puff(0.0, 0.0, [1, 2, 3], 0.0), PUFF.count),
        ];
        for (name, emit, count) in cases {
            let mut p = ParticleSystem::new();
            emit(&mut p);
            assert_eq!(p.live_count(), count, "{name} emitted the wrong count");
            // And every one of them is drawable: a life that jittered to zero
            // would be a slot that is live and invisible forever.
            let born = p.slot(0).unwrap_or_else(|| panic!("{name} was born dead"));
            assert!(born.size > 0.0, "{name} emitted a zero-sized square");
            assert!(born.alpha > 0.0, "{name} emitted an invisible particle");
        }
    }

    #[test]
    fn a_tinted_preset_takes_the_callers_colour_and_keeps_everything_else() {
        let mut p = ParticleSystem::new();
        p.burst(0.0, 0.0, [12, 34, 56]);
        assert_eq!(p.slot(0).unwrap().rgb, [12, 34, 56]);
        assert_eq!(p.slot(0).unwrap().size, BURST.opts.size);
        // The one preset with a colour of its own does not take a caller's.
        let mut d = ParticleSystem::new();
        d.dust(0.0, 0.0);
        assert_eq!(d.slot(0).unwrap().rgb, DUST.opts.color);
    }

    #[test]
    fn a_puffs_strength_scales_the_whole_burst_together() {
        let mut soft = ParticleSystem::new();
        soft.puff(0.0, 0.0, [1, 2, 3], 0.15);
        let mut hard = ParticleSystem::new();
        hard.puff(0.0, 0.0, [1, 2, 3], 1.0);
        assert!(
            hard.live_count() > soft.live_count(),
            "a hard landing was no bigger than a footstep"
        );
        assert!(hard.slot(0).unwrap().size > soft.slot(0).unwrap().size);

        // Clamped at both ends: a zero-strength puff is still something touching
        // down, and an over-strength one is no bigger than a full landing.
        let mut zero = ParticleSystem::new();
        zero.puff(0.0, 0.0, [1, 2, 3], 0.0);
        assert!(zero.live_count() >= PUFF.count);
        let mut over = ParticleSystem::new();
        over.puff(0.0, 0.0, [1, 2, 3], 9.0);
        assert_eq!(over.live_count(), hard.live_count());
    }

    #[test]
    fn a_scrape_and_a_smear_leave_on_the_side_they_were_given() {
        let mut right = ParticleSystem::new();
        right.scrape(0.0, 0.0, 1.0, [1, 2, 3]);
        let mut left = ParticleSystem::new();
        left.scrape(0.0, 0.0, -1.0, [1, 2, 3]);
        // Grit is shaved off the wall and thrown AWAY from it: a wall on the
        // right throws left, and the other way round.
        assert!(right.vx[0] < 0.0, "grit went into the wall");
        assert!(left.vx[0] > 0.0, "grit went into the wall");
        // Both go upward, because they are being scraped off and not dropped.
        assert!(right.vy[0] < 0.0 && left.vy[0] < 0.0);

        let mut dash = ParticleSystem::new();
        dash.smear(0.0, 0.0, 1.0, [1, 2, 3]);
        assert!(
            dash.vx[0] < 0.0,
            "the smear followed the dash instead of trailing it"
        );
        assert_eq!(dash.glow_count(), SMEAR.count, "the smear stopped glowing");
    }

    #[test]
    fn a_creature_dying_puts_up_both_a_splash_and_a_puff_of_its_own_blood() {
        let at = |kind| MobEvent {
            kind,
            x: 40.0,
            y: 50.0,
            rgb: [90, 200, 80],
            power: 3.0,
        };

        let mut died = ParticleSystem::new();
        died.mob_event(at(MobEventKind::MobDie));
        assert!(
            died.live_count() > SPLASH.count,
            "a death was only a splash"
        );
        assert_eq!(
            died.slot(0).unwrap().rgb,
            [90, 200, 80],
            "not the creature's own blood"
        );

        // A hurt is one puff, and it is smaller than the death.
        let mut hurt = ParticleSystem::new();
        hurt.mob_event(at(MobEventKind::MobHurt));
        assert!(hurt.live_count() < died.live_count());

        // A chill has nowhere to burst and must not invent one.
        let mut chill = ParticleSystem::new();
        chill.mob_event(at(MobEventKind::PlayerChill));
        assert_eq!(chill.live_count(), 0);
    }

    #[test]
    fn clear_recycles_everything_and_leaves_the_pool_usable() {
        let mut p = ParticleSystem::new();
        let luminous = EmitOpts {
            glow: true,
            ..lasting(10.0)
        };
        p.emit(0.0, 0.0, 40, &luminous);
        assert_eq!(p.glow_count(), 40);
        p.clear();
        assert_eq!(p.live_count(), 0);
        assert_eq!(p.glow_count(), 0);
        assert!(p.slot(0).is_none());
        p.burst(1.0, 2.0, [4, 5, 6]);
        assert_eq!(p.live_count(), BURST.count);
    }

    #[test]
    fn the_jitter_stream_is_deterministic_and_is_not_the_creatures() {
        // Two pools built the same way produce the same burst, which is what
        // makes every test above an assertion rather than a sample.
        let mut a = ParticleSystem::new();
        let mut b = ParticleSystem::new();
        a.burst(10.0, 10.0, [1, 2, 3]);
        b.burst(10.0, 10.0, [1, 2, 3]);
        for i in 0..BURST.count {
            assert_eq!(a.vx[i], b.vx[i], "slot {i} diverged");
            assert_eq!(a.life[i], b.life[i], "slot {i} diverged");
        }

        // A different seed is a different burst — the stream is really being
        // consumed and not returning a constant.
        let mut c = ParticleSystem::seeded(12345);
        c.burst(10.0, 10.0, [1, 2, 3]);
        assert_ne!(a.vx[0], c.vx[0]);

        // And it is not the creatures' stream: sharing one would make a replay of
        // the world depend on how much juice happened to be on screen.
        assert_ne!(
            JuiceRng::DEFAULT_STATE,
            godgame_core::entities::mobs::MobRng::DEFAULT_STATE
        );
    }

    #[test]
    fn a_zero_seed_still_produces_a_stream() {
        // Xorshift is stuck at zero forever, so the fallback is not a nicety.
        let mut z = JuiceRng::seeded(0);
        assert_ne!(z.next_u32(), 0);
        let mut r = JuiceRng::new();
        for _ in 0..64 {
            let v = r.rand();
            assert!((0.0..1.0).contains(&v), "rand out of range: {v}");
        }
    }
}

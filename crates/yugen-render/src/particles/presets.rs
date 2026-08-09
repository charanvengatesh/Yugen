//! What a burst IS: the per-emit knobs, the named presets, and the tuning the
//! motion model reads.
//!
//! Split from [`super::system`] on the line between description and machinery.
//! Everything here is a `const` or a `Copy` value with no behaviour beyond
//! [`Preset::tinted`] — no pool, no clock, no grid. That is what makes a preset
//! reviewable as a statement about how an effect should LOOK, next to the
//! sentence saying why it looks that way, without the reader also holding the
//! stepping loop in their head.
//!
//! Every number here describes ONE algorithm — how a particle moves, how it
//! fades, or what one named burst is — so it lives beside the code it explains
//! rather than in `yugen-core`'s config, which is for numbers that describe the
//! whole game.
//!
//! The constants are `pub(super)` rather than private because [`super::system`]
//! fires them and [`super`] is where the module boundary now falls. They are
//! deliberately not `pub`: a tuning constant on the crate's surface would be an
//! invitation to read it from somewhere that cannot see this comment.

/// Speed jitter applied to every particle, as a fraction of [`EmitOpts::speed`].
///
/// A burst whose members all leave at exactly one speed reads as a rigid
/// starburst — the eye finds the ring. `0.6..1.0` is enough to break it up
/// without making the burst look under-powered.
pub(super) const SPEED_JITTER: (f32, f32) = (0.6, 1.0);

/// Lifetime jitter, as a fraction of [`EmitOpts::life`]. The same argument as
/// [`SPEED_JITTER`], applied to when the burst DISAPPEARS: particles that all
/// died on one frame would pop out as a unit.
pub(super) const LIFE_JITTER: (f32, f32) = (0.75, 1.25);

/// How fast a wandering particle's drift phase advances, radians/s.
pub(super) const WANDER_RATE: f32 = 2.3;

/// Frequency multiplier on the vertical half of the wander.
///
/// Deliberately not a whole ratio: at 1.7 the x and y drifts never come back
/// into phase, so a mote traces an open figure rather than closing a loop and
/// visibly repeating.
pub(super) const WANDER_Y_RATE: f32 = 1.7;

/// How much weaker the vertical wander is than the horizontal.
///
/// Drift reads as air movement, and air moves sideways. An equal-weight vertical
/// component makes a firefly look like it is being shaken.
pub(super) const WANDER_Y_SCALE: f32 = 0.6;

/// Reciprocal of the fraction of life a [`EmitOpts::fade_in`] particle spends
/// easing in — `5.0` is one fifth.
///
/// Impact debris must appear instantly: it is a reaction, and a reaction that
/// fades in is a reaction that arrived late. Ambient motes must not, because
/// something materialising at full opacity in front of the camera is the one
/// thing that gives away that it was spawned rather than drifted in.
pub(super) const FADE_IN_RECIP: f32 = 5.0;

/// How much speed a colliding particle keeps when it bounces off a cell.
///
/// Low: this is grit and blood, not rubber. Enough that debris visibly scatters
/// off the ground rather than sticking to it on first contact, little enough
/// that it settles within a few tenths of a second under its own drag.
pub(super) const BOUNCE: f32 = 0.35;

/// Impact debris: a wide upward-biased spray in the material's colour.
pub(super) const BURST: Preset = Preset {
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
pub(super) const DUST: Preset = Preset {
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
pub(super) const TRAIL: Preset = Preset {
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
pub(super) const SPLASH: Preset = Preset {
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
pub(super) const PUFF: Preset = Preset {
    count: 2,
    opts: EmitOpts {
        gravity: 260.0,
        drag: 3.0,
        collide: true,
        ..EmitOpts::DEFAULT
    },
};

/// Wall-slide scrape: grit shed off the wall face, thrown back and downward.
pub(super) const SCRAPE: Preset = Preset {
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
pub(super) const SMEAR: Preset = Preset {
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
pub(super) const EMBER: Preset = Preset {
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
pub(super) const PUFF_MOB_HURT: f32 = 0.4;

/// Strength of the puff the player's own wound throws up. Harder than a
/// creature's, because the one the player has to notice is their own.
pub(super) const PUFF_PLAYER_HIT: f32 = 0.5;

/// Strength of the puff under a creature's death, on top of its splash.
pub(super) const PUFF_MOB_DIE: f32 = 0.85;

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// Bit flags packed into the per-particle `flags` byte.
///
/// A byte rather than three `bool` arrays: they are read together, on the same
/// slot, in the same branch, and three parallel arrays would be three cache
/// lines to answer one question.
pub(super) mod flag {
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

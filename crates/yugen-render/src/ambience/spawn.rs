//! What the world is made of, as far as ambient life is concerned: the tuning,
//! the emitter table, and the mood that decides how much of each to spawn.
//!
//! [`super`]'s "Tuning", "Emitters" and "The mood" sections, kept together
//! because they are one argument in three parts — every number here is read by
//! ambience and by nothing else, the emitter table says which cells produce
//! which motes, and the mood turns a biome and a time of day into rates.
//!
//! No Bevy. A mood is a function of the world, and the whole point of ambient
//! life is that it is decided by where you are standing rather than by anything
//! the renderer knows.

use yugen_core::sim::biomes::{BIOME_COUNT, Biome, UG_COUNT, UndergroundLayerId, Weather};

use crate::daynight::DayPhase;

// ---------------------------------------------------------------------------
// Tuning — every number below is read by ambience and by nothing else
// ---------------------------------------------------------------------------

/// Depth, in cells below the LOCAL surface, at which the underground mood starts
/// taking over from the outdoor one.
///
/// Below the local surface and not an absolute row: the handover follows the
/// terrain, so standing at the bottom of a deep valley does not feel like a cave.
pub(super) const UG_FADE_START: f32 = 26.0;

/// Cells over which that handover completes.
///
/// Wide on purpose. The band is several screens tall, so a player walking down a
/// shaft sees pollen thin out and cave dust arrive over a descent long enough to
/// read as "going underground" rather than as a curtain.
pub(super) const UG_FADE_SPAN: f32 = 74.0;

/// Peak spawn rate in motes per second, at full weight, per [`Emitter`].
///
/// One table rather than eight scattered constants because they are ONE tuning
/// decision — the relative density of the eight kinds of ambient life — and
/// reading them side by side is the only way to judge it. Fireflies lead because
/// a swamp at night is otherwise the emptiest frame in the game; cave dust
/// trails because a cavern is supposed to feel still.
pub(super) const PEAK_RATE: [f32; EMITTER_COUNT] = [9.0, 5.0, 7.0, 7.0, 6.0, 5.0, 5.0, 3.5];

/// How much of a plains' pollen a savanna throws. Dry grass, not meadow.
pub(super) const POLLEN_SAVANNA: f32 = 0.6;

/// How much of a plains' pollen a jungle throws. Least of the three: a jungle
/// already has fireflies at night and need not be busy by day as well.
pub(super) const POLLEN_JUNGLE: f32 = 0.5;

/// Fraction of the daytime sand skim that still blows at midnight.
///
/// Not zero. A desert whose air goes perfectly still after dark reads as the
/// emitter having switched off, which is exactly the snap the whole crossfade
/// design exists to avoid.
pub(super) const SAND_NIGHT_FLOOR: f32 = 0.3;

/// Ember output over cold ground that merely happens to contain lava.
///
/// Embers are keyed to actual lava on screen rather than to the biome, so a
/// pocket you dug into embers exactly like a magma chamber does. The layer and
/// biome weights only decide how thickly, and this is the floor under that.
pub(super) const EMBER_BASE: f32 = 0.45;

/// Extra ember output at full magma-layer or volcanic-biome weight.
pub(super) const EMBER_TERRAIN_GAIN: f32 = 0.55;

/// Extra ember output at midnight. Embers read as embers when there is dark to
/// read them against.
pub(super) const EMBER_NIGHT_GAIN: f32 = 0.25;

/// How much of a swamp's fireflies the Mirefen carries. A moor at night is
/// marsh-light country, but colder and quieter than the swamp that anchors the
/// emitter.
pub(super) const FIREFLY_MIREFEN: f32 = 0.6;

/// How much of a grotto's drip the Scald condenses. Not a wet cave but a hot
/// one whose ceiling sweats — audible, sparser than real seepage.
pub(super) const DRIP_SCALD: f32 = 0.7;

/// How much of a geode's glint the Rime Hollows throw. Ice facets catch light
/// the same way crystal does, more dimly. This line is also a fix: Rime
/// shipped with NO emitter at all — the one layer whose mood changed nothing
/// on screen — and the gap survived precisely because nothing asserts a layer
/// is heard. `every_region_reaches_at_least_one_emitter` now does.
pub(super) const GLINT_RIME: f32 = 0.65;

/// How much MORE cave dust the Dust Hollows raise than the Stone Caverns that
/// anchor the emitter. Above 1 on purpose: dust is this layer's weather.
pub(super) const CAVEDUST_DUST_HOLLOWS: f32 = 1.6;

/// Hot cells in view at which the ember rate saturates.
///
/// Six is about one small pocket. Past that the rate is capped, so walking into
/// a magma chamber does not multiply the emission by the size of the lake.
pub(super) const EMBER_FULL_HOT: f32 = 6.0;

/// Horizontal spread, in world px, over which an ember leaves its lava cell.
pub(super) const EMBER_JITTER_PX: f32 = 18.0;

/// World px above the lava cell an ember starts at, so it does not spawn buried.
pub(super) const EMBER_LIFT_PX: f32 = 4.0;

/// Green channel an ember starts at, and how far above it one may land.
///
/// Red is pinned at 255 and blue at 60; only green varies, which walks the
/// colour along the orange ramp from deep to bright without ever leaving it.
pub(super) const EMBER_GREEN: (u8, i32) = (150, 60);

/// Most whole motes one emitter may owe after a single frame.
///
/// A stalled frame must not dump a hundred motes into the pool on the recovery
/// frame — that is a visible burst that reads as a bug, not as ambience.
pub(super) const DEBT_CAP: f32 = 3.0;

/// Below this rate an emitter is treated as off and forfeits its accumulated
/// debt, rather than trickling one mote out every few minutes at a weight that
/// has effectively gone to zero.
pub(super) const RATE_EPS: f32 = 0.001;

/// Rejection-sampling attempts for a plain air cell.
///
/// Deliberately few. A frame that finds nowhere valid simply emits nothing and
/// the emitter tries again next frame; spending longer looking for a gap would
/// cost more than the mote is worth.
pub(super) const AIR_TRIES: u32 = 4;

/// Rejection-sampling attempts for a point that must touch a surface — a ground
/// line, a ceiling, a wall face. Higher than [`AIR_TRIES`] because the target is
/// a one-cell-wide boundary rather than a volume.
pub(super) const SURFACE_TRIES: u32 = 6;

/// Range the nominal launch speed is scaled into, per mote.
///
/// A cohort that all left at the same speed looks like a rigid starburst.
pub(super) const SPEED_JITTER: (f32, f32) = (0.6, 1.0);

/// Range the nominal lifetime is scaled into, per mote. Wider than the speed
/// jitter and centred on 1, so a cohort dissolves raggedly instead of together.
pub(super) const LIFE_JITTER: (f32, f32) = (0.75, 1.25);

/// Radians per second the wander phase advances.
pub(super) const WANDER_RATE: f32 = 2.3;

/// Frequency ratio of the vertical wander to the horizontal one.
///
/// Deliberately not a whole number. Equal frequencies trace a closed ellipse and
/// the drift reads as a machine; an incommensurate ratio never closes, so the
/// path wanders.
pub(super) const WANDER_Y_FREQ: f32 = 1.7;

/// Vertical wander amplitude relative to horizontal. Under 1 because air moves
/// sideways more readily than it moves up.
pub(super) const WANDER_Y_GAIN: f32 = 0.6;

/// How fast a `fade_in` mote reaches full alpha, as a multiple of its life.
///
/// 5 means the rise takes the first fifth of the mote's life. Ambient motes must
/// materialise rather than pop into existence in front of the camera; impact
/// debris, which should appear instantly, does not set the flag.
pub(super) const FADE_IN_RATE: f32 = 5.0;

/// Motes alive at once before new spawns are dropped.
///
/// The TypeScript shared a 2048-slot pool with impact debris and measured
/// ambience's steady state at 40-90 of them. This pool serves ambience alone, so
/// it is sized at roughly three times that steady state: enough headroom that a
/// biome boundary crossing never truncates, small enough that the whole set is a
/// few pages.
pub const POOL_CAPACITY: usize = 256;

/// Most emissive cells the ember emitter will consider in one frame.
///
/// Mirrors the TypeScript light pass's `HOT_MAX`. The ember rate saturates at
/// [`EMBER_FULL_HOT`] anyway, so the cap costs nothing visible and bounds the
/// scan.
pub const HOT_MAX: usize = 64;

/// Highest emitter level in the content set, which normalises `light_emit` into
/// the 0..1 the light pass works in. From the TypeScript's `EMIT_MAX_LEVEL`.
pub(super) const EMIT_MAX_LEVEL: f32 = 15.0;

/// Gain applied to a block's `emissive` when it declares no `light_emit`.
///
/// A material that glows for the renderer but was never given a light level
/// still counts as hot, at a discount — that is how a fire cell qualifies
/// without the content having to state its brightness twice.
pub(super) const EMIT_FALLBACK_GAIN: f32 = 0.8;

/// Where a mote sits in z: over the terrain, under the creatures.
///
/// Under the creatures deliberately. Ambience is atmosphere and must never be
/// the thing obscuring something that can bite.
pub(super) const MOTE_Z: f32 = 0.42;

/// Where a self-luminous mote sits.
///
/// Above [`MOTE_Z`] and nothing more. The additive composite this flag exists
/// for belongs to the lighting pass — see the module header.
pub(super) const MOTE_GLOW_Z: f32 = 0.44;

// ---------------------------------------------------------------------------
// Emitters
// ---------------------------------------------------------------------------

/// How many weather kinds [`Weather`] has, and the width of [`Mood::weather`].
///
/// The discriminants ARE the indices, exactly as [`Biome`]'s are, so a weather
/// weight is looked up with `w as usize` and no side table.
/// `the_weather_enum_still_indexes_the_weight_table` is what holds that true.
pub const WEATHER_COUNT: usize = 5;

/// The eight things the world emits, and the eight fractional spawn
/// accumulators that pace them.
///
/// The discriminant is the index into [`PEAK_RATE`] and into the debt array. The
/// order is the TypeScript's `E_*` slot order, kept because it is the only thing
/// that makes the rate table readable next to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Emitter {
    /// Swamp and jungle, night only, drifting and self-luminous.
    Firefly = 0,
    /// Plains, savanna and jungle by day.
    Pollen = 1,
    /// Desert and savanna, skimming the ground line.
    Sand = 2,
    /// Off any lava cell on screen, whatever the biome.
    Ember = 3,
    /// The fungal depths, rising.
    Spore = 4,
    /// Off a grotto ceiling, straight down.
    Drip = 5,
    /// A facet of geode rock catching the light.
    Glint = 6,
    /// Cavern dust, settling slowly.
    CaveDust = 7,
}

/// How many emitters compete for the frame.
pub const EMITTER_COUNT: usize = 8;

impl Emitter {
    /// Every emitter, in slot order.
    pub const ALL: [Emitter; EMITTER_COUNT] = [
        Emitter::Firefly,
        Emitter::Pollen,
        Emitter::Sand,
        Emitter::Ember,
        Emitter::Spore,
        Emitter::Drip,
        Emitter::Glint,
        Emitter::CaveDust,
    ];

    /// Index into [`PEAK_RATE`] and into the debt array.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether this emitter's weight comes from the SURFACE mix.
    ///
    /// Embers belong to neither half — they are keyed to lava rather than to a
    /// biome — which is why this returns an [`Option`] rather than a `bool`.
    #[inline]
    pub const fn is_surface(self) -> Option<bool> {
        match self {
            Emitter::Firefly | Emitter::Pollen | Emitter::Sand => Some(true),
            Emitter::Spore | Emitter::Drip | Emitter::Glint | Emitter::CaveDust => Some(false),
            Emitter::Ember => None,
        }
    }

    /// The look of one of this emitter's motes.
    #[inline]
    pub const fn spec(self) -> EmitSpec {
        match self {
            Emitter::Firefly => FIREFLY,
            Emitter::Pollen => POLLEN,
            Emitter::Sand => SAND,
            Emitter::Ember => EMBER,
            Emitter::Spore => SPORE,
            Emitter::Drip => DRIP,
            Emitter::Glint => GLINT,
            Emitter::CaveDust => CAVEDUST,
        }
    }
}

/// Per-emit tuning. Velocities are px/s, life is seconds.
///
/// Field for field the TypeScript's `EmitOpts`, including the ones it declared
/// optional: `drag` and `wander` defaulted to 0 there and are written out here,
/// and `angle`'s `None` is its `undefined`. Kept identical so that the shared
/// particle system, when it lands, consumes this type unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitSpec {
    /// Base colour, 0-255 per channel.
    pub color: [u8; 3],
    /// Mean launch speed, px/s, before [`SPEED_JITTER`].
    pub speed: f32,
    /// Random angular fan, radians. 0 is straight along `angle`.
    pub spread: f32,
    /// Nominal lifetime in seconds, before [`LIFE_JITTER`].
    pub life: f32,
    /// Downward acceleration, px/s^2. Negative floats the mote up.
    pub gravity: f32,
    /// Square edge in world px.
    pub size: f32,
    /// Per-second velocity damping. 0 is frictionless.
    pub drag: f32,
    /// Base launch angle in radians. `None` means straight up.
    pub angle: Option<f32>,
    /// Sinusoidal drift acceleration, px/s^2. 0 leaves the mote ballistic —
    /// this is what separates a drifting mote from a thrown chip of debris.
    pub wander: f32,
    /// Draw additively over the lit frame rather than in the world layer. See
    /// the module header for what this currently does and does not do.
    pub glow: bool,
    /// Ease alpha in as well as out.
    pub fade_in: bool,
}

/// Straight up — the launch angle a spec that names none gets.
pub(super) const UP: f32 = -core::f32::consts::FRAC_PI_2;

/// A full circle of spread: the mote leaves in any direction at all.
pub(super) const ANY_DIRECTION: f32 = core::f32::consts::TAU;

/// Drifting, self-luminous, and slow enough to follow with the eye.
pub(super) const FIREFLY: EmitSpec = EmitSpec {
    color: [190, 255, 130],
    speed: 16.0,
    spread: ANY_DIRECTION,
    life: 4.5,
    gravity: 0.0,
    size: 2.0,
    drag: 0.9,
    angle: None,
    wander: 55.0,
    glow: true,
    fade_in: true,
};

/// Barely buoyant — hangs in the air rather than settling.
pub(super) const POLLEN: EmitSpec = EmitSpec {
    color: [226, 224, 178],
    speed: 12.0,
    spread: ANY_DIRECTION,
    life: 6.5,
    gravity: -3.0,
    size: 1.0,
    drag: 0.4,
    angle: None,
    wander: 14.0,
    glow: false,
    fade_in: true,
};

/// Near-horizontal and blowing right, so it reads as wind over the dunes.
pub(super) const SAND: EmitSpec = EmitSpec {
    color: [214, 188, 140],
    speed: 70.0,
    spread: 0.5,
    life: 2.6,
    gravity: 26.0,
    size: 1.0,
    drag: 0.25,
    angle: Some(-0.18),
    wander: 10.0,
    glow: false,
    fade_in: true,
};

/// Buoyant — rides the heat upward.
pub(super) const EMBER: EmitSpec = EmitSpec {
    color: [255, EMBER_GREEN.0, 60],
    speed: 34.0,
    spread: 1.1,
    life: 1.9,
    gravity: -46.0,
    size: 1.0,
    drag: 0.7,
    angle: Some(UP),
    wander: 22.0,
    glow: true,
    fade_in: true,
};

/// Rises through the caverns.
pub(super) const SPORE: EmitSpec = EmitSpec {
    color: [140, 255, 175],
    speed: 10.0,
    spread: ANY_DIRECTION,
    life: 5.5,
    gravity: -14.0,
    size: 1.0,
    drag: 0.5,
    angle: None,
    wander: 20.0,
    glow: true,
    fade_in: true,
};

/// Straight down off the ceiling. The life is long enough to leave the view and
/// short enough to give the slot back.
pub(super) const DRIP: EmitSpec = EmitSpec {
    color: [130, 185, 235],
    speed: 4.0,
    spread: 0.3,
    life: 1.2,
    gravity: 900.0,
    size: 1.0,
    drag: 0.0,
    angle: Some(core::f32::consts::FRAC_PI_2),
    wander: 0.0,
    glow: false,
    fade_in: false,
};

/// Sits on a rock face rather than floating — a facet catching the light.
pub(super) const GLINT: EmitSpec = EmitSpec {
    color: [200, 235, 255],
    speed: 0.0,
    spread: 0.0,
    life: 1.3,
    gravity: 0.0,
    size: 1.0,
    drag: 0.0,
    angle: None,
    wander: 0.0,
    glow: true,
    fade_in: true,
};

/// Settles slowly.
pub(super) const CAVEDUST: EmitSpec = EmitSpec {
    color: [158, 154, 148],
    speed: 6.0,
    spread: ANY_DIRECTION,
    life: 5.0,
    gravity: 16.0,
    size: 1.0,
    drag: 0.3,
    angle: None,
    wander: 8.0,
    glow: false,
    fade_in: true,
};

// ---------------------------------------------------------------------------
// The mood
// ---------------------------------------------------------------------------

/// The local climate, resolved: which biomes, which layers, how deep, what
/// weather.
///
/// Public and plainly constructible on purpose. Everything about which motes
/// appear is a pure function of this plus a [`DayPhase`], and separating the
/// noise sampling ([`Ambience::resolve`]) from the judgement ([`spawn_rates`])
/// is what makes "a cave never shows pollen" a two-line test rather than a world
/// generation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mood {
    /// Normalised surface-biome weights, indexed by [`Biome::index`]. Sums to 1.
    pub surface: [f32; BIOME_COUNT],
    /// Normalised underground-layer weights, indexed by
    /// [`UndergroundLayerId::index`]. Sums to 1.
    pub layer: [f32; UG_COUNT],
    /// Blended weather weights, indexed by `Weather as usize`.
    ///
    /// One of two published outputs that ambience itself does not consume:
    /// [`crate::weather`] reads it to crossfade the backdrop layer (the other is
    /// [`Mood::ambient`]). It is blended on the SAME weights rather than
    /// hard-switching to whichever biome happens to be dominant, so the sky and
    /// the motes agree about where the desert starts.
    pub weather: [f32; WEATHER_COUNT],
    /// 0 outdoors, 1 in the underground layers, smooth between.
    pub underground: f32,
}

impl Mood {
    /// Weight of one surface biome.
    #[inline]
    pub fn biome(&self, b: Biome) -> f32 {
        self.surface[b.index()]
    }

    /// Weight of one underground layer.
    #[inline]
    pub fn layer_of(&self, l: UndergroundLayerId) -> f32 {
        self.layer[l.index()]
    }

    /// Weight of one weather kind.
    #[inline]
    pub fn weather_of(&self, w: Weather) -> f32 {
        self.weather[w as usize]
    }

    /// The biome's additive ambient cast, 0..1 per channel.
    ///
    /// The second published output — [`crate::light`] adds it as a flat wash over
    /// the finished frame, which is what makes a tundra read cold and a magma
    /// chamber warm instead of every biome being lit the same colour.
    ///
    /// The colours are not invented here: each is the biome's own authored
    /// `atmo.ambient`, and they are summed over the SAME normalised weights the
    /// terrain is dithered with. That is the whole reason this lives on [`Mood`]
    /// rather than on a call to
    /// [`resolve_atmosphere`](yugen_core::sim::biomes::resolve_atmosphere),
    /// which would sample the noise a second time to arrive at the same number.
    /// Plains sends `[0, 0, 0]`, so a plains column is a no-op in the composite
    /// exactly as it was before anything wrote this at all.
    ///
    /// Surface weights only, and deliberately: the cast is the colour of the sky
    /// OVERHEAD, so a cave under a tundra is still cast cold. How dark that cave
    /// is belongs to the light pass's own depth term and is not this number's
    /// business.
    pub fn ambient(&self) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for b in Biome::ALL {
            let w = self.surface[b.index()];
            for (channel, cast) in out.iter_mut().zip(b.def().atmo.ambient) {
                *channel += cast as f32 * w;
            }
        }
        out
    }
}

/// Spawn rates in motes per second, for a resolved mood at a moment of the day.
///
/// Pure, and the whole of the module's judgement. `hot_count` is how many
/// emissive cells [`scan_hot`] found in view; everything else comes off the mood
/// and the clock.
///
/// The TypeScript ran embers down a separate path that zeroed their debt when
/// the hot list was empty, rather than through the shared rate check. Folding
/// them in is exact rather than merely equivalent: the ember rate with even one
/// hot cell is at least `PEAK x EMBER_BASE / EMBER_FULL_HOT`, about 0.5, which
/// never approaches [`RATE_EPS`] from above — so the only way the shared check
/// can fire for embers is the empty hot list the special case existed to handle.
pub fn spawn_rates(mood: &Mood, phase: &DayPhase, hot_count: usize) -> [f32; EMITTER_COUNT] {
    let surf = 1.0 - mood.underground;
    let ug = mood.underground;
    let day = phase.day;
    let night = phase.night;

    let mut out = [0.0f32; EMITTER_COUNT];

    // --- Surface -----------------------------------------------------------
    out[Emitter::Firefly.index()] = surf
        * night
        * (mood.biome(Biome::Swamp)
            + mood.biome(Biome::Jungle)
            + mood.biome(Biome::Mirefen) * FIREFLY_MIREFEN);
    // Plains declares no weather at all, so without pollen the friendliest biome
    // is also the emptiest.
    out[Emitter::Pollen.index()] = surf
        * day
        * (mood.biome(Biome::Plains)
            + mood.biome(Biome::Savanna) * POLLEN_SAVANNA
            + mood.biome(Biome::Jungle) * POLLEN_JUNGLE);
    out[Emitter::Sand.index()] = surf
        * (SAND_NIGHT_FLOOR + (1.0 - SAND_NIGHT_FLOOR) * day)
        * (mood.biome(Biome::Desert) + mood.biome(Biome::Savanna));

    // --- Depth -------------------------------------------------------------
    out[Emitter::Spore.index()] = ug * mood.layer_of(UndergroundLayerId::Fungal);
    out[Emitter::Drip.index()] = ug
        * (mood.layer_of(UndergroundLayerId::Grottos)
            + mood.layer_of(UndergroundLayerId::Scald) * DRIP_SCALD);
    out[Emitter::Glint.index()] = ug
        * (mood.layer_of(UndergroundLayerId::Geode)
            + mood.layer_of(UndergroundLayerId::Rime) * GLINT_RIME);
    out[Emitter::CaveDust.index()] = ug
        * (mood.layer_of(UndergroundLayerId::Caverns)
            + mood.layer_of(UndergroundLayerId::DustHollows) * CAVEDUST_DUST_HOLLOWS);

    // --- Embers ------------------------------------------------------------
    if hot_count > 0 {
        let heat = EMBER_BASE
            + EMBER_TERRAIN_GAIN
                * (ug
                    * (mood.layer_of(UndergroundLayerId::Magma)
                        + mood.layer_of(UndergroundLayerId::Scald))
                    + surf * (mood.biome(Biome::Volcanic) + mood.biome(Biome::Cinderveld)))
            + EMBER_NIGHT_GAIN * night;
        out[Emitter::Ember.index()] = heat * (hot_count as f32 / EMBER_FULL_HOT).min(1.0);
    }

    for (rate, peak) in out.iter_mut().zip(PEAK_RATE) {
        *rate *= peak;
    }
    out
}

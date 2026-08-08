//! Ambient life: the motes the WORLD emits, as opposed to the ones the player
//! kicks up.
//!
//! Ported from `src/render/ambience.ts`.
//!
//! Fireflies over a swamp at dusk, pollen on a plains morning, sand skimming a
//! dune, embers off a lava pocket, spores in the fungal depths, drips off a
//! grotto ceiling, glints in a geode, dust falling through a cavern.
//!
//! # How it blends
//!
//! Every emitter's spawn RATE is the product of three continuous weights and
//! nothing else:
//!
//! ```text
//! rate = peak x biome-or-layer weight x depth blend x time-of-day weight
//! ```
//!
//! The biome weight is the very same normalised weight out of [`biome_mix_at`]
//! that the world generator dithers terrain with, so ambience crossfades on
//! exactly the boundary the ground does — walk from jungle into desert and the
//! fireflies thin out over the same span the moss gives way to sand, because
//! both are being driven by the same number. Nothing snaps: a weight going to
//! zero just stops new spawns, and the motes already in flight live out their
//! seconds and fade. That also means a boundary needs no special case —
//! three-way junctions included, since [`Mix`] weights are continuous there too.
//!
//! # Cost discipline
//!
//! Emission is bounded three ways: spawn rates are per-second and fractional (an
//! emitter accumulates until it owes a whole mote), the weights across surface
//! emitters sum to at most 1 (they partition one mix) and likewise underground,
//! and spawn points are rejection-sampled INSIDE THE VIEW RECT only — never
//! across the streamed window. A handful of grid reads and at most a few emits
//! per frame; steady-state population is ~40-90 motes.
//!
//! # What the port changed
//!
//! **The randomness is seeded.** The TypeScript drew every spawn point, launch
//! angle and lifetime from `Math.random`, which is unseedable. That is not
//! available here: the tree has exactly one random source, [`SimRng`], and this
//! module uses it. The stream is deliberately its OWN — seeded from the world
//! seed, never shared with the automata's — because the automata's determinism
//! is order-dependent on a single global draw sequence, and a world whose
//! evolution depended on how many fireflies happened to be on screen would fail
//! `worldgen_purity`. Nothing about ambience needs to be reproducible for the
//! game's sake; it is reproducible so the tests below can assert what a swamp at
//! midnight actually emits instead of asserting a distribution.
//!
//! **The particles are here.** The TypeScript emitted into the shared
//! `ParticleSystem` that impact debris also used. `crate::particles` is not
//! written yet, so this module carries [`MotePool`] — the same fixed-capacity
//! pool, the same integrator, the same fade envelope, sized for ambience alone.
//! [`EmitSpec`] is `EmitOpts` field for field, so when the shared pool lands the
//! swap is a rename and a deletion, not a rewrite. The pool is an array of
//! structs rather than the TypeScript's parallel typed arrays: that layout
//! existed to keep the GC asleep, and there is no GC.
//!
//! **The hot list is found here.** Embers key off actual lava on screen rather
//! than off the biome, and the TypeScript got that list free from the lighting
//! pass, which had already visited every emissive cell. [`scan_hot`] walks the
//! view itself instead, on [`HOT_STRIDE_CELLS`], applying the light pass's emit
//! rule.
//!
//! It was written to mirror the light grid's stride exactly, so that lighting
//! could later hand it a hot list for free. That is no longer the plan: the
//! light grid now samples every cell, and following it would cost 16x for a
//! number that only sets an ember rate. See [`HOT_STRIDE_CELLS`].
//!
//! **The clock is read, never advanced.** [`WorldClock`] is owned and ticked by
//! [`crate::daynight::DayNightPlugin`], which is the only thing in the app that
//! calls `update` on it. This module, the sky and the light composite all take
//! `Res<WorldClock>` and read the phase. Ticking it here as well would run the
//! day at twice speed — see that plugin's header, which exists because all three
//! of these modules were written to tick a clock of their own.
//!
//! # What the port dropped on the way in
//!
//! **The camera shake.** The TypeScript sampled `cam.viewX/viewY`, which
//! included the transient shake offset `effects.ts` wrote each frame, so spawn
//! points jittered with the screen. `crate::effects` is a stub, and a shaken
//! spawn rect is a side effect of shake rather than a feature of ambience, so
//! [`ViewRect`] is built from [`WorldFocus`] alone. When effects lands, adding
//! the offset is one line in [`breathe`].
//!
//! **The additive glow pass.** A luminous mote was drawn a second time, in
//! `lighter` composite mode, over the finished lit frame — the only way a
//! firefly survives being multiplied down by the darkness mask. There is no
//! darkness mask yet. [`Mote::glow`] is carried and drives z-order only; the
//! additive composite is the lighting milestone's to add, and the flag it needs
//! is already on every mote.

use bevy::prelude::*;

use yugen_core::config::{CELL_SIZE, SEED, View};
use yugen_core::sim::biomes::{
    BIOME_COUNT, Biome, Mix, UG_COUNT, UndergroundLayerId, Weather, biome_mix_at,
    underground_mix_at,
};
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::{CellId, EMPTY, mat_by_code};
use yugen_core::sim::noise::Noise;
use yugen_core::sim::rng::SimRng;
use yugen_core::sim::worldgen::heightmap::Heightmap;
use yugen_core::sim::worldgen::world_noise;

use crate::daynight::{DayPhase, WorldClock};
use crate::light::BiomeAmbient;
use crate::lowres::{LowResTarget, WORLD_LAYERS};
use crate::weather::WeatherWeights;
use crate::world::{SimWorld, WorldFocus};

// ---------------------------------------------------------------------------
// Tuning — every number below is read by ambience and by nothing else
// ---------------------------------------------------------------------------

/// Depth, in cells below the LOCAL surface, at which the underground mood starts
/// taking over from the outdoor one.
///
/// Below the local surface and not an absolute row: the handover follows the
/// terrain, so standing at the bottom of a deep valley does not feel like a cave.
const UG_FADE_START: f32 = 26.0;

/// Cells over which that handover completes.
///
/// Wide on purpose. The band is several screens tall, so a player walking down a
/// shaft sees pollen thin out and cave dust arrive over a descent long enough to
/// read as "going underground" rather than as a curtain.
const UG_FADE_SPAN: f32 = 74.0;

/// Peak spawn rate in motes per second, at full weight, per [`Emitter`].
///
/// One table rather than eight scattered constants because they are ONE tuning
/// decision — the relative density of the eight kinds of ambient life — and
/// reading them side by side is the only way to judge it. Fireflies lead because
/// a swamp at night is otherwise the emptiest frame in the game; cave dust
/// trails because a cavern is supposed to feel still.
const PEAK_RATE: [f32; EMITTER_COUNT] = [9.0, 5.0, 7.0, 7.0, 6.0, 5.0, 5.0, 3.5];

/// How much of a plains' pollen a savanna throws. Dry grass, not meadow.
const POLLEN_SAVANNA: f32 = 0.6;

/// How much of a plains' pollen a jungle throws. Least of the three: a jungle
/// already has fireflies at night and need not be busy by day as well.
const POLLEN_JUNGLE: f32 = 0.5;

/// Fraction of the daytime sand skim that still blows at midnight.
///
/// Not zero. A desert whose air goes perfectly still after dark reads as the
/// emitter having switched off, which is exactly the snap the whole crossfade
/// design exists to avoid.
const SAND_NIGHT_FLOOR: f32 = 0.3;

/// Ember output over cold ground that merely happens to contain lava.
///
/// Embers are keyed to actual lava on screen rather than to the biome, so a
/// pocket you dug into embers exactly like a magma chamber does. The layer and
/// biome weights only decide how thickly, and this is the floor under that.
const EMBER_BASE: f32 = 0.45;

/// Extra ember output at full magma-layer or volcanic-biome weight.
const EMBER_TERRAIN_GAIN: f32 = 0.55;

/// Extra ember output at midnight. Embers read as embers when there is dark to
/// read them against.
const EMBER_NIGHT_GAIN: f32 = 0.25;

/// Hot cells in view at which the ember rate saturates.
///
/// Six is about one small pocket. Past that the rate is capped, so walking into
/// a magma chamber does not multiply the emission by the size of the lake.
const EMBER_FULL_HOT: f32 = 6.0;

/// Horizontal spread, in world px, over which an ember leaves its lava cell.
const EMBER_JITTER_PX: f32 = 18.0;

/// World px above the lava cell an ember starts at, so it does not spawn buried.
const EMBER_LIFT_PX: f32 = 4.0;

/// Green channel an ember starts at, and how far above it one may land.
///
/// Red is pinned at 255 and blue at 60; only green varies, which walks the
/// colour along the orange ramp from deep to bright without ever leaving it.
const EMBER_GREEN: (u8, i32) = (150, 60);

/// Most whole motes one emitter may owe after a single frame.
///
/// A stalled frame must not dump a hundred motes into the pool on the recovery
/// frame — that is a visible burst that reads as a bug, not as ambience.
const DEBT_CAP: f32 = 3.0;

/// Below this rate an emitter is treated as off and forfeits its accumulated
/// debt, rather than trickling one mote out every few minutes at a weight that
/// has effectively gone to zero.
const RATE_EPS: f32 = 0.001;

/// Rejection-sampling attempts for a plain air cell.
///
/// Deliberately few. A frame that finds nowhere valid simply emits nothing and
/// the emitter tries again next frame; spending longer looking for a gap would
/// cost more than the mote is worth.
const AIR_TRIES: u32 = 4;

/// Rejection-sampling attempts for a point that must touch a surface — a ground
/// line, a ceiling, a wall face. Higher than [`AIR_TRIES`] because the target is
/// a one-cell-wide boundary rather than a volume.
const SURFACE_TRIES: u32 = 6;

/// Range the nominal launch speed is scaled into, per mote.
///
/// A cohort that all left at the same speed looks like a rigid starburst.
const SPEED_JITTER: (f32, f32) = (0.6, 1.0);

/// Range the nominal lifetime is scaled into, per mote. Wider than the speed
/// jitter and centred on 1, so a cohort dissolves raggedly instead of together.
const LIFE_JITTER: (f32, f32) = (0.75, 1.25);

/// Radians per second the wander phase advances.
const WANDER_RATE: f32 = 2.3;

/// Frequency ratio of the vertical wander to the horizontal one.
///
/// Deliberately not a whole number. Equal frequencies trace a closed ellipse and
/// the drift reads as a machine; an incommensurate ratio never closes, so the
/// path wanders.
const WANDER_Y_FREQ: f32 = 1.7;

/// Vertical wander amplitude relative to horizontal. Under 1 because air moves
/// sideways more readily than it moves up.
const WANDER_Y_GAIN: f32 = 0.6;

/// How fast a `fade_in` mote reaches full alpha, as a multiple of its life.
///
/// 5 means the rise takes the first fifth of the mote's life. Ambient motes must
/// materialise rather than pop into existence in front of the camera; impact
/// debris, which should appear instantly, does not set the flag.
const FADE_IN_RATE: f32 = 5.0;

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
const EMIT_MAX_LEVEL: f32 = 15.0;

/// Gain applied to a block's `emissive` when it declares no `light_emit`.
///
/// A material that glows for the renderer but was never given a light level
/// still counts as hot, at a discount — that is how a fire cell qualifies
/// without the content having to state its brightness twice.
const EMIT_FALLBACK_GAIN: f32 = 0.8;

/// Where a mote sits in z: over the terrain, under the creatures.
///
/// Under the creatures deliberately. Ambience is atmosphere and must never be
/// the thing obscuring something that can bite.
const MOTE_Z: f32 = 0.42;

/// Where a self-luminous mote sits.
///
/// Above [`MOTE_Z`] and nothing more. The additive composite this flag exists
/// for belongs to the lighting pass — see the module header.
const MOTE_GLOW_Z: f32 = 0.44;

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
const UP: f32 = -core::f32::consts::FRAC_PI_2;

/// A full circle of spread: the mote leaves in any direction at all.
const ANY_DIRECTION: f32 = core::f32::consts::TAU;

/// Drifting, self-luminous, and slow enough to follow with the eye.
const FIREFLY: EmitSpec = EmitSpec {
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
const POLLEN: EmitSpec = EmitSpec {
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
const SAND: EmitSpec = EmitSpec {
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
const EMBER: EmitSpec = EmitSpec {
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
const SPORE: EmitSpec = EmitSpec {
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
const DRIP: EmitSpec = EmitSpec {
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
const GLINT: EmitSpec = EmitSpec {
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
const CAVEDUST: EmitSpec = EmitSpec {
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
    out[Emitter::Firefly.index()] =
        surf * night * (mood.biome(Biome::Swamp) + mood.biome(Biome::Jungle));
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
    out[Emitter::Drip.index()] = ug * mood.layer_of(UndergroundLayerId::Grottos);
    out[Emitter::Glint.index()] = ug * mood.layer_of(UndergroundLayerId::Geode);
    out[Emitter::CaveDust.index()] = ug * mood.layer_of(UndergroundLayerId::Caverns);

    // --- Embers ------------------------------------------------------------
    if hot_count > 0 {
        let heat = EMBER_BASE
            + EMBER_TERRAIN_GAIN
                * (ug * mood.layer_of(UndergroundLayerId::Magma)
                    + surf * mood.biome(Biome::Volcanic))
            + EMBER_NIGHT_GAIN * night;
        out[Emitter::Ember.index()] = heat * (hot_count as f32 / EMBER_FULL_HOT).min(1.0);
    }

    for (rate, peak) in out.iter_mut().zip(PEAK_RATE) {
        *rate *= peak;
    }
    out
}

// ---------------------------------------------------------------------------
// The view rectangle
// ---------------------------------------------------------------------------

/// The rectangle of world an emitter may spawn into, in world px, +y DOWN.
///
/// Spawn points are rejection-sampled inside this and nowhere else. The streamed
/// window is a good deal larger, and sampling it would spend most of the tries
/// on cells nobody can see.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewRect {
    /// Left edge, world px.
    pub x: f32,
    /// Top edge, world px, +y DOWN.
    pub y: f32,
    /// Width, world px.
    pub w: f32,
    /// Height, world px.
    pub h: f32,
}

impl ViewRect {
    /// The rect a `view`-sized viewport centred on world px `(cx, cy)` covers.
    ///
    /// [`WorldFocus`] is the CENTRE and the TypeScript camera's `x`/`y` were the
    /// top-left, which is the whole of the difference between this and the
    /// original's `cam.viewX`.
    pub fn centred_on(cx: f32, cy: f32, view: View) -> ViewRect {
        let w = view.w as f32;
        let h = view.h as f32;
        ViewRect {
            x: cx - w * 0.5,
            y: cy - h * 0.5,
            w,
            h,
        }
    }

    /// Centre column, in cells and fractional — where the climate is sampled.
    #[inline]
    pub fn centre_col(&self) -> f32 {
        (self.x + self.w * 0.5) / CELL_SIZE as f32
    }

    /// Centre row, in cells and fractional.
    #[inline]
    pub fn centre_row(&self) -> f32 {
        (self.y + self.h * 0.5) / CELL_SIZE as f32
    }
}

/// Everything outside the model that one frame of ambience reads.
///
/// Bundled rather than passed as three more arguments: they travel together,
/// they all answer "what does the world look like right now", and a `&mut self`
/// method taking six references is where an argument gets silently swapped.
pub struct Around<'a> {
    /// The streamed cells. Only cells inside [`Around::view`] are sampled.
    pub grid: &'a CellGrid,
    /// The view rectangle, world px, +y DOWN.
    pub view: ViewRect,
    /// Emissive cells found in view this frame — see [`scan_hot`].
    pub hot: &'a [WorldCell],
}

// ---------------------------------------------------------------------------
// The hot list
// ---------------------------------------------------------------------------

/// The light pass's emit level for a material, 0 when it is not an emitter.
///
/// Reproduces the TypeScript's `EMIT_LEVEL` table exactly: a declared
/// `light_emit` normalised by [`EMIT_MAX_LEVEL`] wins, and a material with none
/// falls back to its renderer `emissive` at [`EMIT_FALLBACK_GAIN`]. Air is never
/// an emitter — the TypeScript's table loop started at id 1 for the same reason.
pub fn emit_level(id: CellId) -> f32 {
    if id == EMPTY {
        return 0.0;
    }
    let def = mat_by_code(id);
    let declared = def.light_emit as f32 / EMIT_MAX_LEVEL;
    if declared > 0.0 {
        declared
    } else {
        def.emissive * EMIT_FALLBACK_GAIN
    }
}

/// Cells between hot-list samples.
///
/// **Deliberately its own number, not [`LIGHT_DOWNSCALE`].** It was that
/// constant, from when this stood in for a lighting pass that did not exist yet
/// and had to find the same cells the light grid would. Both facts have since
/// stopped being true: the lighting pass exists, and it does not sample on a
/// stride any more — `LIGHT_DOWNSCALE` went to 1 so light could sit on the art's
/// own grid.
///
/// Following it there cost 16x for nothing. This scan does not want a faithful
/// mirror of the solver; it wants a sparse "roughly where is it hot" for an
/// EMBER RATE, and it is capped at [`HOT_MAX`] anyway. Measured, the coupled
/// version went 1.26 us -> 17.69 us per frame, and worse than the cost: the cap
/// then covered a sixteenth of the area, so ember spawns bunched toward the
/// top-left of the view instead of spreading across it.
///
/// 4 is the value this scan was tuned at and is what keeps the ember rate where
/// it has always been.
const HOT_STRIDE_CELLS: i32 = 4;

/// Fill `out` with the emissive cells in view, capped at [`HOT_MAX`].
///
/// Strided by [`HOT_STRIDE_CELLS`] and sampling the stride's centre cell. A lava
/// pool one cell wide can therefore be missed, and it will be missed
/// CONSISTENTLY — the lattice is anchored to the world and not to the camera, so
/// a hot cell does not blink as the view scrolls a pixel. That consistency is
/// the property that matters here, not completeness: this feeds how many embers
/// drift up, not what is lit.
pub fn scan_hot(grid: &CellGrid, view: ViewRect, out: &mut Vec<WorldCell>) {
    out.clear();

    let step = HOT_STRIDE_CELLS;
    let half = step / 2;
    let cell = CELL_SIZE as f32;
    let lx0 = div_floor((view.x / cell).floor() as i32, step);
    let lx1 = div_floor(((view.x + view.w) / cell).ceil() as i32, step);
    let ly0 = div_floor((view.y / cell).floor() as i32, step);
    let ly1 = div_floor(((view.y + view.h) / cell).ceil() as i32, step);

    for ly in ly0..=ly1 {
        for lx in lx0..=lx1 {
            if out.len() >= HOT_MAX {
                return;
            }
            let at = WorldCell::new(lx * step + half, ly * step + half);
            if !grid.is_loaded_world(at) {
                continue;
            }
            if emit_level(grid.get_world(at)) > 0.0 {
                out.push(at);
            }
        }
    }
}

/// Floor division, so the light lattice is continuous west and north of the
/// world origin instead of folding at it.
#[inline]
fn div_floor(a: i32, b: i32) -> i32 {
    let q = a / b;
    if a % b != 0 && (a < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

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
    debt: [f32; EMITTER_COUNT],
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
        let surface = self.heightmap.surface_row_at(&self.noise, col, None) as f32;
        mood.underground = smoothstep01((cam_row - surface - UG_FADE_START) / UG_FADE_SPAN);

        for (biome, w) in weights_of(&biome_mix_at(&self.noise, col)) {
            mood.surface[biome.index()] = w;
            mood.weather[biome.def().atmo.weather as usize] += w;
        }
        for (layer, w) in weights_of(&underground_mix_at(&self.noise, col)) {
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
    fn run(&mut self, e: Emitter, dt: f32, rate: f32, around: &Around, pool: &mut MotePool) {
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
fn cell_of(x: f32, y: f32) -> WorldCell {
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
fn weights_of<T: Copy>(mix: &Mix<T>) -> impl Iterator<Item = (T, f32)> + '_ {
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
fn smoothstep01(v: f32) -> f32 {
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
fn unit(rng: &mut SimRng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

/// A draw scaled into `range`.
#[inline]
fn jittered(range: (f32, f32), t: f32) -> f32 {
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

// ---------------------------------------------------------------------------
// The plugin
// ---------------------------------------------------------------------------

/// The model, the pool, and the hot-cell scratch, in one resource.
///
/// One resource and not three because a frame mutates all three together and
/// Bevy's borrow checker is per-resource — the same reasoning
/// [`crate::world::SimWorld`] is built on.
#[derive(Resource)]
pub struct AmbientLife {
    /// The climate model and its accumulators.
    pub ambience: Ambience,
    /// The motes it has put up.
    pub motes: MotePool,
    /// This frame's emissive cells. A field rather than a local so the `Vec` is
    /// allocated once for the run and refilled in place.
    hot: Vec<WorldCell>,
}

impl AmbientLife {
    /// Ambience for the world grown from `seed`.
    pub fn new(seed: u32) -> AmbientLife {
        AmbientLife {
            ambience: Ambience::new(seed),
            motes: MotePool::default(),
            hot: Vec::with_capacity(HOT_MAX),
        }
    }
}

/// One pooled mote sprite. `slot` indexes [`MotePool::motes`].
#[derive(Component, Clone, Copy)]
pub struct MoteSprite {
    /// Index into the pool.
    pub slot: usize,
}

/// The ambient life the world emits, and the squares standing in for it.
pub struct AmbiencePlugin;

impl Plugin for AmbiencePlugin {
    fn build(&self, app: &mut App) {
        // Built here rather than in a startup system for the reason
        // `crate::mobs` builds its creatures here: the sprite pool is sized from
        // it, and both have to exist before any world does. `SEED` is the const
        // `crate::world::spawn_world` grows the world from; `follow_seed` is
        // what makes a runtime seed a non-event rather than a silent mismatch.
        app.insert_resource(AmbientLife::new(SEED))
            .add_systems(Startup, spawn_motes)
            // `PreUpdate` and not `Update`, which is where the mood is resolved:
            // both consumers read in `Update` and neither is nameable from here
            // — `crate::light`'s solve is a private system in no set, and
            // ordering against a system you cannot name is not expressible. A
            // whole schedule earlier is the edge that IS expressible, and Bevy
            // runs `PreUpdate` before `Update` every frame, so the write always
            // lands before both reads. The mood it publishes is therefore the
            // one resolved on the previous frame; that is a colour wash which
            // takes hundreds of columns of walking to change, so a frame of lag
            // in it is not a thing an eye can find.
            .add_systems(PreUpdate, publish_mood)
            .add_systems(
                Update,
                (breathe.run_if(resource_exists::<SimWorld>), place_motes).chain(),
            );
    }
}

/// Hand the resolved mood to the two passes that consume it.
///
/// Both targets are optional because this plugin has to stand on its own: a
/// caller may add [`AmbiencePlugin`] without the light composite or without the
/// weather backdrop, and a missing consumer is a system that publishes nothing,
/// not a panic.
fn publish_mood(
    life: Res<AmbientLife>,
    ambient: Option<ResMut<BiomeAmbient>>,
    weather: Option<ResMut<WeatherWeights>>,
) {
    let mood = life.ambience.mood();

    if let Some(mut ambient) = ambient {
        ambient.0 = mood.ambient();
    }
    if let Some(mut weather) = weather {
        // A move and not a translation: both vectors are five slots indexed by
        // `Weather as usize`, so the only thing that could ever put snow in the
        // embers slot is one of them changing width — and that is a type error
        // on this line rather than a silent mistint.
        weather.0 = mood.weather;
    }
}

/// One entity per pool slot, hidden, once.
///
/// Sized from the pool itself rather than from [`POOL_CAPACITY`] restated here,
/// so a pool that grows cannot leave slots undrawn.
fn spawn_motes(mut commands: Commands, life: Res<AmbientLife>) {
    for slot in 0..life.motes.capacity() {
        commands.spawn((
            Sprite {
                custom_size: Some(Vec2::ZERO),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, MOTE_Z),
            MoteSprite { slot },
            Visibility::Hidden,
            WORLD_LAYERS,
        ));
    }
}

/// Resolve the mood, emit a frame's worth of ambience, and step what is in
/// flight.
fn breathe(
    time: Res<Time>,
    world: Res<SimWorld>,
    focus: Res<WorldFocus>,
    target: Res<LowResTarget>,
    clock: Res<WorldClock>,
    mut life: ResMut<AmbientLife>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    // The view rect is built from the focus alone: the camera shake the
    // TypeScript folded in belongs to `crate::effects`, which is a stub.
    let view = ViewRect::centred_on(focus.x, focus.y, target.view);
    let phase = clock.0.phase();

    let AmbientLife {
        ambience,
        motes,
        hot,
    } = &mut *life;

    ambience.follow_seed(world.seed);
    scan_hot(&world.level.grid, view, hot);

    ambience.update(
        dt,
        &Around {
            grid: &world.level.grid,
            view,
            hot,
        },
        &phase,
        motes,
    );
    motes.update(dt);
}

/// Put every mote sprite on its slot, or hide it.
fn place_motes(
    life: Res<AmbientLife>,
    mut sprites: Query<(&MoteSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
) {
    let pool = life.motes.motes();

    for (which, mut sprite, mut transform, mut visibility) in &mut sprites {
        let m = &pool[which.slot];
        if !m.alive() {
            *visibility = Visibility::Hidden;
            continue;
        }
        *visibility = Visibility::Inherited;
        sprite.color = Color::srgb_u8(m.color[0], m.color[1], m.color[2]).with_alpha(m.alpha());
        sprite.custom_size = Some(Vec2::splat(m.size));
        transform.translation.z = if m.glow { MOTE_GLOW_Z } else { MOTE_Z };

        // A mote is a SQUARE with a corner, not a point with a half-extent: the
        // TypeScript filled a `size`-edge rect at the rounded position, so it is
        // the top-left that rounds and the centre that follows from it — the
        // rule `crate::mobs::place_mobs` snaps a body box with.
        let left = m.x.round();
        let top = m.y.round();
        transform.translation.x = left + m.size * 0.5;
        // +y is up in Bevy and down in the sim: the one convention flip.
        transform.translation.y = -(top + m.size * 0.5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::sim::materials::block;

    /// A mood that is one biome and nothing else, at the surface.
    fn all_surface(b: Biome) -> Mood {
        let mut mood = Mood::default();
        mood.surface[b.index()] = 1.0;
        mood
    }

    /// A mood that is one underground layer and nothing else, fully deep.
    fn all_deep(l: UndergroundLayerId) -> Mood {
        let mut mood = Mood::default();
        mood.layer[l.index()] = 1.0;
        mood.underground = 1.0;
        mood
    }

    /// Noon and midnight, the two ends every day/night gate is judged at.
    fn noon() -> DayPhase {
        DayPhase::at(0.5)
    }

    fn midnight() -> DayPhase {
        DayPhase::at(0.0)
    }

    /// A grid of nothing but air, big enough to hold a view.
    fn air_grid() -> CellGrid {
        CellGrid::new(256, 256)
    }

    /// A grid of nothing but stone.
    fn solid_grid() -> CellGrid {
        let mut g = CellGrid::new(256, 256);
        for y in 0..g.rows() {
            for x in 0..g.cols() {
                g.set(x, y, block::STONE);
            }
        }
        g
    }

    /// Air above `floor_row`, stone at and below it.
    fn grid_with_floor(floor_row: i32) -> CellGrid {
        let mut g = CellGrid::new(256, 256);
        for y in floor_row..g.rows() {
            for x in 0..g.cols() {
                g.set(x, y, block::STONE);
            }
        }
        g
    }

    /// The view the tests sample in — a whole number of cells, at the origin.
    fn view() -> ViewRect {
        ViewRect {
            x: 0.0,
            y: 0.0,
            w: (64 * CELL_SIZE) as f32,
            h: (64 * CELL_SIZE) as f32,
        }
    }

    fn around<'a>(grid: &'a CellGrid, hot: &'a [WorldCell]) -> Around<'a> {
        Around {
            grid,
            view: view(),
            hot,
        }
    }

    /// The mood at the first column of the real world that casts anything.
    ///
    /// Found rather than written down: which biome sits at which column is
    /// worldgen's business, and plains — the commonest biome, and the one the
    /// origin tends to land in — authored a black cast deliberately, so a
    /// hard-coded column would be testing the wrong thing the day the climate
    /// centroids moved.
    fn a_lit_column(a: &mut Ambience) -> Mood {
        /// How far to look before giving up. A whole climate region is a few
        /// hundred columns wide, so this is several of them.
        const SEARCH_COLS: i32 = 4000;

        for col in 0..SEARCH_COLS {
            let mood = a.resolve(col as f32, 0.0);
            if mood.ambient() != [0.0; 3] {
                return mood;
            }
        }
        panic!("none of the first {SEARCH_COLS} columns casts anything at all");
    }

    // --- Biome gating ------------------------------------------------------

    #[test]
    fn a_cave_biome_never_shows_surface_motes() {
        // Fully underground: whatever the biome overhead and whatever the hour,
        // the three surface emitters are off. `surf` is `1 - underground` and it
        // multiplies all three, which is the whole mechanism.
        let mut mood = all_deep(UndergroundLayerId::Caverns);
        mood.surface[Biome::Swamp.index()] = 1.0;

        for phase in [noon(), midnight()] {
            let r = spawn_rates(&mood, &phase, 0);
            assert_eq!(r[Emitter::Firefly.index()], 0.0);
            assert_eq!(r[Emitter::Pollen.index()], 0.0);
            assert_eq!(r[Emitter::Sand.index()], 0.0);
            assert!(r[Emitter::CaveDust.index()] > 0.0, "but the cave is alive");
        }
    }

    #[test]
    fn the_open_air_never_shows_cave_motes() {
        // And the mirror. A surface mood with every layer weight pinned high
        // still emits nothing underground, because `ug` is zero.
        let mut mood = all_surface(Biome::Plains);
        mood.layer = [1.0; UG_COUNT];

        let r = spawn_rates(&mood, &noon(), 0);
        for e in [
            Emitter::Spore,
            Emitter::Drip,
            Emitter::Glint,
            Emitter::CaveDust,
        ] {
            assert_eq!(r[e.index()], 0.0, "{e:?} leaked into the open air");
        }
    }

    #[test]
    fn only_the_wet_biomes_get_fireflies() {
        for b in Biome::ALL {
            let r = spawn_rates(&all_surface(b), &midnight(), 0);
            let want = b == Biome::Swamp || b == Biome::Jungle;
            assert_eq!(
                r[Emitter::Firefly.index()] > 0.0,
                want,
                "{b:?} fireflies at midnight"
            );
        }
    }

    #[test]
    fn only_the_dry_biomes_get_a_sand_skim() {
        for b in Biome::ALL {
            let r = spawn_rates(&all_surface(b), &noon(), 0);
            let want = b == Biome::Desert || b == Biome::Savanna;
            assert_eq!(r[Emitter::Sand.index()] > 0.0, want, "{b:?} sand at noon");
        }
    }

    #[test]
    fn a_jungle_pollens_less_than_a_plains_and_a_savanna_sits_between() {
        // The three weights are deliberately unequal — a plains has no weather
        // of its own and would otherwise be the emptiest frame in the game.
        let p = Emitter::Pollen.index();
        let plains = spawn_rates(&all_surface(Biome::Plains), &noon(), 0)[p];
        let savanna = spawn_rates(&all_surface(Biome::Savanna), &noon(), 0)[p];
        let jungle = spawn_rates(&all_surface(Biome::Jungle), &noon(), 0)[p];
        assert!(plains > savanna);
        assert!(savanna > jungle);
        assert!(jungle > 0.0);
    }

    // --- Day/night gating --------------------------------------------------

    #[test]
    fn fireflies_only_come_out_at_night() {
        let mood = all_surface(Biome::Swamp);
        assert_eq!(
            spawn_rates(&mood, &noon(), 0)[Emitter::Firefly.index()],
            0.0
        );
        assert!(spawn_rates(&mood, &midnight(), 0)[Emitter::Firefly.index()] > 0.0);
    }

    #[test]
    fn pollen_only_drifts_by_day() {
        let mood = all_surface(Biome::Plains);
        assert!(spawn_rates(&mood, &noon(), 0)[Emitter::Pollen.index()] > 0.0);
        assert_eq!(
            spawn_rates(&mood, &midnight(), 0)[Emitter::Pollen.index()],
            0.0
        );
    }

    #[test]
    fn the_dunes_are_never_completely_still() {
        // Sand is the one surface emitter with a floor rather than a gate. A
        // desert whose air switches off after dark reads as a bug.
        let mood = all_surface(Biome::Desert);
        let day = spawn_rates(&mood, &noon(), 0)[Emitter::Sand.index()];
        let night = spawn_rates(&mood, &midnight(), 0)[Emitter::Sand.index()];
        assert!(night > 0.0, "the desert went silent at night");
        assert!(day > night);
        assert!((night / day - SAND_NIGHT_FLOOR).abs() < 1e-5);
    }

    #[test]
    fn the_depths_do_not_care_what_time_it_is() {
        // No sunlight reaches a fungal cavern, and the rates say so. Embers are
        // excluded because they are lit by lava, not by the sun.
        for l in UndergroundLayerId::ALL {
            let mood = all_deep(l);
            assert_eq!(
                spawn_rates(&mood, &noon(), 0),
                spawn_rates(&mood, &midnight(), 0),
                "{l:?} changed with the hour"
            );
        }
    }

    // --- The depth handover ------------------------------------------------

    #[test]
    fn descending_trades_pollen_for_cave_dust_without_a_gap() {
        // Sweep the blend and check the two never both go to zero: there is no
        // depth at which the world stops emitting anything at all.
        let mut mood = Mood::default();
        mood.surface[Biome::Plains.index()] = 1.0;
        mood.layer[UndergroundLayerId::Caverns.index()] = 1.0;

        let mut surface_led = false;
        let mut deep_led = false;
        for i in 0..=100 {
            mood.underground = i as f32 / 100.0;
            let r = spawn_rates(&mood, &noon(), 0);
            let pollen = r[Emitter::Pollen.index()];
            let dust = r[Emitter::CaveDust.index()];
            assert!(
                pollen + dust > 0.0,
                "dead band at blend {}",
                mood.underground
            );
            surface_led |= pollen > dust;
            deep_led |= dust > pollen;
        }
        assert!(surface_led && deep_led, "the handover never happened");
    }

    #[test]
    fn every_rate_is_continuous_in_the_depth_blend() {
        // Nothing snaps. A weight going to zero must taper, because a mote
        // population that steps is the exact artefact the crossfade design is
        // for. The bound is generous — what is being caught is a jump, not
        // drift.
        let mut mood = Mood {
            surface: [1.0 / BIOME_COUNT as f32; BIOME_COUNT],
            layer: [1.0 / UG_COUNT as f32; UG_COUNT],
            ..Default::default()
        };

        let mut prev: Option<[f32; EMITTER_COUNT]> = None;
        for i in 0..=1000 {
            mood.underground = i as f32 / 1000.0;
            let now = spawn_rates(&mood, &DayPhase::at(0.3), 4);
            if let Some(before) = prev {
                for (e, (a, b)) in before.iter().zip(now.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 0.05,
                        "emitter {e} jumped {a} -> {b} at blend {}",
                        mood.underground
                    );
                }
            }
            prev = Some(now);
        }
    }

    #[test]
    fn the_depth_blend_is_flat_in_the_sky_and_saturated_far_below() {
        let mut a = Ambience::new(SEED);
        // A row well above any terrain: unambiguously outdoors.
        assert_eq!(a.resolve(0.0, -1000.0).underground, 0.0);
        // And one far past the end of the fade span.
        assert_eq!(a.resolve(0.0, 5000.0).underground, 1.0);
    }

    #[test]
    fn the_handover_follows_the_terrain_rather_than_an_absolute_row() {
        // The point of measuring depth below the LOCAL surface: two columns
        // whose ground lines differ must hand over at rows that differ by the
        // same amount, so a deep valley does not feel like a cave.
        let mut a = Ambience::new(SEED);
        let mut seen = Vec::new();
        for col in [0i32, 900, 1800, 2700] {
            // The row at which the blend first reaches half.
            let mut row = -400.0f32;
            while row < 4000.0 && a.resolve(col as f32, row).underground < 0.5 {
                row += 1.0;
            }
            seen.push(row);
        }
        assert!(
            seen.windows(2).any(|w| w[0] != w[1]),
            "the handover row never moved with the terrain: {seen:?}"
        );
    }

    // --- Embers ------------------------------------------------------------

    #[test]
    fn embers_stop_the_frame_the_last_lava_leaves_the_view() {
        let mood = all_deep(UndergroundLayerId::Magma);
        assert!(spawn_rates(&mood, &midnight(), 1)[Emitter::Ember.index()] > 0.0);
        assert_eq!(
            spawn_rates(&mood, &midnight(), 0)[Emitter::Ember.index()],
            0.0
        );
    }

    #[test]
    fn an_ember_emitter_with_no_lava_left_forfeits_its_debt() {
        // The TypeScript zeroed the debt on its own path. Folding embers into
        // the shared rate check has to preserve that, or the first lava cell to
        // come back into view fires a stored burst.
        let grid = air_grid();
        let hot = [WorldCell::new(3, 3)];
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();

        a.run(Emitter::Ember, 0.05, 4.0, &around(&grid, &hot), &mut pool);
        assert!(a.debt_of(Emitter::Ember) > 0.0);
        a.run(Emitter::Ember, 0.05, 0.0, &around(&grid, &[]), &mut pool);
        assert_eq!(a.debt_of(Emitter::Ember), 0.0);
    }

    #[test]
    fn embers_burn_thicker_over_a_magma_chamber_than_over_a_stone_cavern() {
        // Keyed to lava, weighted by terrain: the same pocket embers either way,
        // but the chamber embers harder.
        let e = Emitter::Ember.index();
        let magma = spawn_rates(&all_deep(UndergroundLayerId::Magma), &noon(), 3)[e];
        let caverns = spawn_rates(&all_deep(UndergroundLayerId::Caverns), &noon(), 3)[e];
        assert!(magma > caverns);
        assert!(caverns > 0.0, "a pocket you dug into must still ember");
    }

    #[test]
    fn the_ember_rate_saturates_once_a_pocket_is_hot() {
        let mood = all_deep(UndergroundLayerId::Magma);
        let e = Emitter::Ember.index();
        let full = spawn_rates(&mood, &noon(), EMBER_FULL_HOT as usize)[e];
        let flooded = spawn_rates(&mood, &noon(), HOT_MAX)[e];
        assert_eq!(full, flooded, "a lava lake must not scale the emission");
        assert!(spawn_rates(&mood, &noon(), 1)[e] < full);
    }

    #[test]
    fn embers_read_hotter_against_the_dark() {
        let mood = all_deep(UndergroundLayerId::Magma);
        let e = Emitter::Ember.index();
        assert!(spawn_rates(&mood, &midnight(), 3)[e] > spawn_rates(&mood, &noon(), 3)[e]);
    }

    #[test]
    fn a_volcanic_surface_embers_as_hard_as_a_magma_chamber() {
        // The two halves of the heat term are symmetric on purpose: a lava flow
        // on a volcanic slope must not look colder than the same flow in a cave.
        let e = Emitter::Ember.index();
        let mut above = all_surface(Biome::Volcanic);
        above.underground = 0.0;
        let below = all_deep(UndergroundLayerId::Magma);
        assert_eq!(
            spawn_rates(&above, &noon(), 4)[e],
            spawn_rates(&below, &noon(), 4)[e]
        );
    }

    // --- The debt accumulator ----------------------------------------------

    #[test]
    fn a_fractional_rate_accumulates_until_it_owes_a_whole_mote() {
        // Two motes a second, sixty-hertz frames: nothing for the first half
        // second, then one.
        let grid = air_grid();
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        let dt = 1.0 / 60.0;

        a.run(Emitter::Pollen, dt, 2.0, &around(&grid, &[]), &mut pool);
        assert_eq!(pool.live_count(), 0, "a thirtieth of a mote is not a mote");
        for _ in 0..29 {
            a.run(Emitter::Pollen, dt, 2.0, &around(&grid, &[]), &mut pool);
        }
        assert_eq!(pool.live_count(), 1);
    }

    #[test]
    fn a_stalled_frame_cannot_dump_more_than_three_motes_of_one_kind() {
        // The recovery frame after a stall is the one the player is most likely
        // to be looking at. It must not be the one that bursts.
        let grid = air_grid();
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        a.run(Emitter::Pollen, 10.0, 9.0, &around(&grid, &[]), &mut pool);
        assert_eq!(pool.live_count(), DEBT_CAP as usize);
    }

    #[test]
    fn an_emitter_that_switches_off_carries_no_debt_into_the_next_biome() {
        let grid = air_grid();
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        a.run(Emitter::Firefly, 0.1, 5.0, &around(&grid, &[]), &mut pool);
        assert!(a.debt_of(Emitter::Firefly) > 0.0);
        // Walked out of the swamp: the weight went to zero.
        a.run(Emitter::Firefly, 0.1, 0.0, &around(&grid, &[]), &mut pool);
        assert_eq!(a.debt_of(Emitter::Firefly), 0.0);
    }

    // --- Rejection sampling ------------------------------------------------

    #[test]
    fn nothing_spawns_inside_solid_rock() {
        // Solid to the horizon: the air pick has nowhere to land, gives up, and
        // the frame emits nothing rather than putting a firefly in the stone.
        let g = solid_grid();
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        for _ in 0..200 {
            a.run(Emitter::Firefly, 1.0, 9.0, &around(&g, &[]), &mut pool);
        }
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn nothing_spawns_where_the_world_is_not_streamed_in_yet() {
        // A view far outside a small window: every sample lands on an unloaded
        // cell, which must be rejected rather than read as air.
        let grid = CellGrid::new(8, 8);
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        let far = Around {
            grid: &grid,
            view: ViewRect {
                x: 10_000.0,
                y: 10_000.0,
                w: 320.0,
                h: 320.0,
            },
            hot: &[],
        };
        for _ in 0..200 {
            a.run(Emitter::Pollen, 1.0, 5.0, &far, &mut pool);
        }
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn a_grain_of_sand_sits_on_the_ground_line() {
        let floor = 32;
        let grid = grid_with_floor(floor);
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        for _ in 0..40 {
            a.run(Emitter::Sand, 1.0, 7.0, &around(&grid, &[]), &mut pool);
        }
        assert!(pool.live_count() > 0, "found no ground at all");
        // Every grain starts one px above the last air row: never in the stone
        // and never floating somewhere up the sky.
        let want = ((floor - 1) * CELL_SIZE - 1) as f32;
        for m in pool.motes().iter().filter(|m| m.alive()) {
            assert_eq!(m.y, want, "a grain spawned off the ground line");
        }
    }

    #[test]
    fn a_drip_hangs_from_a_ceiling_and_never_from_open_sky() {
        // A ceiling is stone with air UNDER it, which a plain floor grid has
        // nowhere — so the drip finds nothing and emits nothing.
        let open = grid_with_floor(32);
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        for _ in 0..40 {
            a.run(Emitter::Drip, 1.0, 5.0, &around(&open, &[]), &mut pool);
        }
        assert_eq!(pool.live_count(), 0, "a drip fell out of clear sky");

        // Now hollow a chamber out under the stone: row 40 is a ceiling.
        let mut cave = solid_grid();
        for y in 40..60 {
            for x in 0..64 {
                cave.set(x, y, EMPTY);
            }
        }
        let mut pool = MotePool::default();
        for _ in 0..60 {
            a.run(Emitter::Drip, 1.0, 5.0, &around(&cave, &[]), &mut pool);
        }
        assert!(pool.live_count() > 0, "found no ceiling in a cave");
        for m in pool.motes().iter().filter(|m| m.alive()) {
            assert_eq!(m.y, (40 * CELL_SIZE) as f32, "a drip started off-ceiling");
        }
    }

    #[test]
    fn a_glint_needs_rock_beside_it() {
        // Open air has no vertical faces, so the wall pick finds nothing; a
        // one-cell-wide shaft is nothing but faces.
        let open = air_grid();
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        for _ in 0..60 {
            a.run(Emitter::Glint, 1.0, 5.0, &around(&open, &[]), &mut pool);
        }
        assert_eq!(pool.live_count(), 0, "a glint off nothing at all");

        let mut shaft = solid_grid();
        for y in 0..shaft.rows() {
            shaft.set(10, y, EMPTY);
        }
        let mut pool = MotePool::default();
        for _ in 0..200 {
            a.run(Emitter::Glint, 1.0, 5.0, &around(&shaft, &[]), &mut pool);
        }
        assert!(pool.live_count() > 0, "found no wall in a shaft");
    }

    #[test]
    fn an_ember_leaves_from_a_cell_the_scan_actually_found() {
        let grid = air_grid();
        let hot = [WorldCell::new(20, 20)];
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        for _ in 0..20 {
            a.run(Emitter::Ember, 1.0, 7.0, &around(&grid, &hot), &mut pool);
        }
        assert!(pool.live_count() > 0);
        let lava_x = (20 * CELL_SIZE) as f32;
        let lava_y = (20 * CELL_SIZE) as f32;
        for m in pool.motes().iter().filter(|m| m.alive()) {
            assert!(
                (m.x - lava_x).abs() <= EMBER_JITTER_PX * 0.5,
                "an ember left from {} and its lava is at {lava_x}",
                m.x
            );
            assert_eq!(m.y, lava_y - EMBER_LIFT_PX, "an ember spawned buried");
        }
    }

    #[test]
    fn an_ember_is_always_some_shade_of_orange() {
        // Only the green channel varies, and it must stay inside the ramp — a
        // `u8` that wrapped would put a blue spark over the lava.
        let grid = air_grid();
        let hot = [WorldCell::new(20, 20)];
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        for _ in 0..60 {
            a.run(Emitter::Ember, 1.0, 7.0, &around(&grid, &hot), &mut pool);
        }
        let (base, range) = EMBER_GREEN;
        for m in pool.motes().iter().filter(|m| m.alive()) {
            assert_eq!(m.color[0], 255);
            assert_eq!(m.color[2], 60);
            assert!((base..base + range as u8).contains(&m.color[1]));
        }
    }

    // --- The hot scan ------------------------------------------------------

    #[test]
    fn only_emissive_cells_make_the_hot_list() {
        let mut g = solid_grid();
        let mut hot = Vec::new();
        scan_hot(&g, view(), &mut hot);
        assert!(hot.is_empty(), "stone is not hot");

        // Flood a band with lava. The scan strides, so it need not find every
        // cell — it must find some, and only lava ones.
        for y in 10..20 {
            for x in 0..64 {
                g.set(x, y, block::LAVA);
            }
        }
        scan_hot(&g, view(), &mut hot);
        assert!(!hot.is_empty());
        for at in &hot {
            assert!(emit_level(g.get_world(*at)) > 0.0);
        }
    }

    #[test]
    fn the_hot_list_is_capped_rather_than_unbounded() {
        let mut g = CellGrid::new(256, 256);
        for y in 0..g.rows() {
            for x in 0..g.cols() {
                g.set(x, y, block::LAVA);
            }
        }
        let mut hot = Vec::new();
        scan_hot(&g, view(), &mut hot);
        assert_eq!(hot.len(), HOT_MAX);
    }

    #[test]
    fn the_hot_scan_refills_its_buffer_instead_of_appending_to_it() {
        // The `Vec` is a resource field so the run allocates once. A second scan
        // of a cooled world must shrink the list, not extend it.
        let mut g = CellGrid::new(256, 256);
        for y in 0..40 {
            for x in 0..64 {
                g.set(x, y, block::LAVA);
            }
        }
        let mut hot = Vec::new();
        scan_hot(&g, view(), &mut hot);
        let first = hot.len();
        assert!(first > 0);
        for y in 0..40 {
            for x in 0..64 {
                g.set(x, y, EMPTY);
            }
        }
        scan_hot(&g, view(), &mut hot);
        assert!(hot.is_empty(), "the list still held {first} stale cells");
    }

    #[test]
    fn the_hot_scan_lattice_does_not_slide_with_the_camera() {
        // A hot cell must not blink as the view scrolls a pixel, which is why
        // the stride is anchored to the world and not to the view's left edge.
        let mut g = CellGrid::new(256, 256);
        for y in 0..40 {
            for x in 0..64 {
                g.set(x, y, block::LAVA);
            }
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        let base = view();
        scan_hot(&g, base, &mut a);
        scan_hot(
            &g,
            ViewRect {
                x: base.x + 1.0,
                ..base
            },
            &mut b,
        );
        // The two lists may differ at the edges as the rect slides, but the
        // cells they share must be the SAME cells, not shifted ones.
        assert!(a.iter().filter(|c| b.contains(c)).count() > a.len() / 2);
    }

    #[test]
    fn air_is_never_hot_however_the_table_reads() {
        assert_eq!(emit_level(EMPTY), 0.0);
        assert!(emit_level(block::LAVA) > 0.0);
    }

    // --- The pool ----------------------------------------------------------

    #[test]
    fn the_pool_drops_motes_instead_of_growing() {
        let mut pool = MotePool::new(8);
        let mut rng = SimRng::seeded(1);
        for _ in 0..64 {
            pool.emit(0.0, 0.0, &POLLEN, &mut rng);
        }
        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.live_count(), 8);
    }

    #[test]
    fn a_dead_mote_gives_its_slot_back() {
        let mut pool = MotePool::new(2);
        let mut rng = SimRng::seeded(1);
        pool.emit(0.0, 0.0, &GLINT, &mut rng);
        pool.emit(0.0, 0.0, &GLINT, &mut rng);
        assert_eq!(pool.live_count(), 2);
        // Longer than the longest jittered glint life.
        pool.update(GLINT.life * LIFE_JITTER.1 + 0.1);
        assert_eq!(pool.live_count(), 0);
        pool.emit(0.0, 0.0, &GLINT, &mut rng);
        assert_eq!(pool.live_count(), 1);
    }

    #[test]
    fn a_mote_fades_in_and_then_out() {
        // The envelope is the rise TIMES the decay, so a fade-in mote peaks at
        // 0.8 and never reaches full opacity — which is the TypeScript's
        // envelope exactly, and part of why ambient motes read as soft rather
        // than as sprites.
        let mut pool = MotePool::new(1);
        let mut rng = SimRng::seeded(1);
        pool.emit(0.0, 0.0, &POLLEN, &mut rng);
        let life = pool.motes()[0].max_life;

        let born = pool.motes()[0].alpha();
        assert!(born < 0.05, "a fade-in mote popped in at alpha {born}");

        // A fifth of the way through is the crest, where the rise saturates.
        pool.update(life * 0.2);
        let crest = pool.motes()[0].alpha();
        assert!((crest - 0.8).abs() < 1e-4, "crest was {crest}");

        // Past it, alpha is just the remaining fraction, falling to nothing.
        pool.update(life * 0.5);
        let late = pool.motes()[0].alpha();
        assert!(late < crest);
        pool.update(life * 0.2);
        assert!(pool.motes()[0].alpha() < late);
    }

    #[test]
    fn a_drip_appears_instantly_because_it_did_not_ask_to_fade_in() {
        let mut pool = MotePool::new(1);
        let mut rng = SimRng::seeded(1);
        pool.emit(0.0, 0.0, &DRIP, &mut rng);
        assert!((pool.motes()[0].alpha() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn buoyancy_is_the_sign_of_gravity_and_nothing_else() {
        // The risers rise and the settlers settle, in the sim's +y DOWN
        // convention. This is what catches the flip being applied in the model
        // rather than once at draw time.
        for spec in [SPORE, EMBER, POLLEN] {
            assert!(spec.gravity < 0.0, "a riser was given positive gravity");
        }
        for spec in [CAVEDUST, SAND, DRIP] {
            assert!(spec.gravity > 0.0, "a settler was given negative gravity");
        }
    }

    #[test]
    fn drag_stops_a_mote_rather_than_reversing_it() {
        // `1 - drag * dt` goes negative on a long enough frame, which without
        // the clamp would fling the mote backwards.
        let mut pool = MotePool::new(1);
        let mut rng = SimRng::seeded(1);
        let spec = EmitSpec {
            speed: 100.0,
            spread: 0.0,
            angle: Some(0.0),
            drag: 10.0,
            gravity: 0.0,
            wander: 0.0,
            life: 100.0,
            ..POLLEN
        };
        pool.emit(0.0, 0.0, &spec, &mut rng);
        pool.update(1.0);
        assert_eq!(pool.motes()[0].vx, 0.0);
        assert!(pool.motes()[0].x >= 0.0, "drag threw the mote backwards");
    }

    #[test]
    fn a_wandering_mote_does_not_retrace_its_own_path() {
        // The two drift frequencies are incommensurate on purpose: equal ones
        // would close into an ellipse and read as a machine.
        let mut pool = MotePool::new(1);
        let mut rng = SimRng::seeded(1);
        pool.emit(0.0, 0.0, &FIREFLY, &mut rng);
        let mut seen = Vec::new();
        for _ in 0..200 {
            pool.update(1.0 / 120.0);
            let m = pool.motes()[0];
            seen.push((m.x, m.y));
        }
        // No two sampled points coincide — a closed orbit would repeat.
        for (i, a) in seen.iter().enumerate() {
            for b in &seen[i + 1..] {
                assert!(
                    (a.0 - b.0).abs() > 1e-6 || (a.1 - b.1).abs() > 1e-6,
                    "the drift closed on itself at {a:?}"
                );
            }
        }
    }

    // --- Determinism -------------------------------------------------------

    #[test]
    fn two_ambiences_on_the_same_seed_agree_mote_for_mote() {
        // The reason the port routes `Math.random` through `SimRng` at all. Not
        // a gameplay requirement — a testability one.
        let grid = grid_with_floor(32);
        let hot = [WorldCell::new(4, 4)];

        let run = || {
            let mut a = Ambience::new(SEED);
            let mut pool = MotePool::default();
            for _ in 0..120 {
                a.update(
                    1.0 / 60.0,
                    &around(&grid, &hot),
                    &DayPhase::at(0.8),
                    &mut pool,
                );
                pool.update(1.0 / 60.0);
            }
            pool.motes().to_vec()
        };

        assert_eq!(run(), run());
    }

    #[test]
    fn a_new_world_starts_the_ambience_over() {
        let mut a = Ambience::new(SEED);
        a.debt = [0.5; EMITTER_COUNT];
        a.follow_seed(SEED);
        assert_eq!(a.debt_of(Emitter::Pollen), 0.5, "the same seed is a no-op");
        a.follow_seed(SEED + 1);
        assert_eq!(a.seed(), SEED + 1);
        assert_eq!(a.debt_of(Emitter::Pollen), 0.0, "stale debt survived");
    }

    // --- The resolved mood -------------------------------------------------

    #[test]
    fn the_biome_weights_are_a_partition_of_one() {
        // The property the cost discipline rests on: the surface emitters share
        // one mix, so their weights can never sum past 1 however many biomes
        // meet at a column.
        let mut a = Ambience::new(SEED);
        for col in (-4000..4000).step_by(97) {
            let mood = a.resolve(col as f32, -1000.0);
            let surf: f32 = mood.surface.iter().sum();
            let layer: f32 = mood.layer.iter().sum();
            assert!(
                (surf - 1.0).abs() < 1e-4,
                "surface weights at {col}: {surf}"
            );
            assert!(
                (layer - 1.0).abs() < 1e-4,
                "layer weights at {col}: {layer}"
            );
        }
    }

    #[test]
    fn the_weather_weights_ride_the_same_mix_the_biomes_do() {
        // `crate::weather`'s input. It must be the biome weights regrouped by
        // weather kind and nothing else — a hard switch to the dominant biome
        // would put the dust storm's edge somewhere the sand grains are not.
        let mut a = Ambience::new(SEED);
        for col in (-4000..4000).step_by(211) {
            let mood = a.resolve(col as f32, -1000.0);
            let total: f32 = mood.weather.iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "weather weights at {col}");

            let mut regrouped = [0.0f32; WEATHER_COUNT];
            for b in Biome::ALL {
                regrouped[b.def().atmo.weather as usize] += mood.biome(b);
            }
            for (got, want) in mood.weather.iter().zip(regrouped) {
                assert!((got - want).abs() < 1e-6, "weather at {col}");
            }
            // And the accessor agrees with the raw index.
            assert_eq!(mood.weather_of(Weather::Dust), mood.weather[1]);
        }
    }

    #[test]
    fn the_weather_enum_still_indexes_the_weight_table() {
        // `Weather as usize` is used as an array index with no side table, which
        // is only sound while the discriminants are a dense `0..WEATHER_COUNT`.
        let all = [
            Weather::None,
            Weather::Dust,
            Weather::Snow,
            Weather::Spores,
            Weather::Embers,
        ];
        assert_eq!(all.len(), WEATHER_COUNT);
        for (i, w) in all.into_iter().enumerate() {
            assert_eq!(w as usize, i);
        }
    }

    #[test]
    fn a_frame_of_real_world_ambience_puts_something_up() {
        // The end-to-end shape: real climate fields, a real grid, a real phase.
        // Sweeping columns rather than trusting one, because which biome sits at
        // a given column is worldgen's business and not this test's. The window
        // is dragged along with the view, exactly as the streamer drags it, so
        // the cells under the spawn rect stay loaded.
        let mut grid = grid_with_floor(32);
        let mut a = Ambience::new(SEED);
        let mut pool = MotePool::default();
        let mut any = false;
        for col in (0..6000).step_by(311) {
            pool.clear();
            grid.set_origin(col, 0);
            let v = ViewRect {
                x: (col * CELL_SIZE) as f32,
                y: 0.0,
                w: (64 * CELL_SIZE) as f32,
                h: (64 * CELL_SIZE) as f32,
            };
            for _ in 0..60 {
                a.update(
                    1.0 / 60.0,
                    &Around {
                        grid: &grid,
                        view: v,
                        hot: &[],
                    },
                    &noon(),
                    &mut pool,
                );
                pool.update(1.0 / 60.0);
            }
            any |= pool.live_count() > 0;
        }
        assert!(any, "a whole daytime surface sweep emitted nothing");
    }

    #[test]
    fn the_view_rect_is_centred_on_the_focus_and_not_cornered_at_it() {
        // The one real difference from the TypeScript camera, whose `x`/`y` were
        // the top-left. Getting this wrong offsets every spawn by half a screen.
        let rect = ViewRect::centred_on(1000.0, 500.0, View::for_screen(1280, 800));
        assert!((rect.centre_col() * CELL_SIZE as f32 - 1000.0).abs() < 1e-3);
        assert!((rect.centre_row() * CELL_SIZE as f32 - 500.0).abs() < 1e-3);
        assert!(rect.x < 1000.0 && rect.x + rect.w > 1000.0);
    }

    // --- The biome's ambient cast ------------------------------------------

    #[test]
    fn a_lone_biome_casts_exactly_the_colour_its_content_authored() {
        // The cast is not a palette this module invented. It is each biome's own
        // `atmo.ambient` — the same numbers `Game.ts` handed the light pass —
        // and weighted onto a single biome the sum has to come back as that
        // biome's entry, unchanged and unscaled.
        for b in Biome::ALL {
            let want = b.def().atmo.ambient.map(|c| c as f32);
            assert_eq!(all_surface(b).ambient(), want, "{b:?} cast");
        }
    }

    #[test]
    fn the_frozen_and_the_volcanic_are_not_lit_the_same_colour() {
        // The bug this test exists for: the composite read a cast that nothing
        // wrote, so it stayed at the black `Default` and every biome in the game
        // was lit identically. Two biomes at opposite ends of the palette have to
        // disagree, and neither may be black.
        let frozen = all_surface(Biome::Glacier).ambient();
        let hot = all_surface(Biome::Volcanic).ambient();

        assert_ne!(frozen, hot, "a glacier is lit like a magma chamber");
        assert!(frozen.iter().any(|c| *c > 0.0), "the glacier casts nothing");
        assert!(hot.iter().any(|c| *c > 0.0), "the volcano casts nothing");
        // And in the directions a player would name them, so a channel swap on
        // the way through is a failure rather than merely a different colour.
        assert!(frozen[2] > frozen[0], "the glacier does not read cold");
        assert!(hot[0] > hot[2], "the volcano does not read warm");
    }

    #[test]
    fn the_cast_crossfades_across_a_boundary_rather_than_switching() {
        // Half and half has to land on the midpoint. Anything that picked the
        // dominant biome instead would return one end or the other.
        let (a, b) = (Biome::Desert, Biome::Tundra);
        let mut half = Mood::default();
        half.surface[a.index()] = 0.5;
        half.surface[b.index()] = 0.5;

        let mid = half.ambient();
        let (ca, cb) = (all_surface(a).ambient(), all_surface(b).ambient());
        assert_ne!(mid, ca);
        assert_ne!(mid, cb);
        for i in 0..3 {
            assert!(
                (mid[i] - (ca[i] + cb[i]) * 0.5).abs() < 1e-6,
                "channel {i} is not the midpoint"
            );
        }
    }

    #[test]
    fn a_cavern_is_cast_by_the_sky_over_it_and_not_by_the_rock() {
        // Depth is the light pass's own term. A cavern under a tundra is still a
        // COLD cavern; darkening it here as well would put one handover in two
        // places and neither would be tunable on its own.
        let mut deep = all_surface(Biome::Tundra);
        deep.underground = 1.0;
        deep.layer[UndergroundLayerId::Caverns.index()] = 1.0;
        assert_eq!(deep.ambient(), all_surface(Biome::Tundra).ambient());
    }

    #[test]
    fn the_plugin_publishes_the_cast_and_the_weather_blend_to_their_consumers() {
        // The join itself, and the one thing the value tests above cannot reach:
        // both resources were declared, read by their pass, and written by
        // nobody, and every other test in this file passed throughout. So this
        // one boots the real plugin, hands it a resolved mood, and demands that
        // the resources the two passes read actually move.
        let mut app = App::new();
        app.add_plugins(AmbiencePlugin)
            .init_resource::<BiomeAmbient>()
            .init_resource::<WeatherWeights>();

        let mood = {
            let mut life = app.world_mut().resource_mut::<AmbientLife>();
            a_lit_column(&mut life.ambience)
        };
        app.update();

        assert_eq!(app.world().resource::<BiomeAmbient>().0, mood.ambient());
        assert_ne!(
            app.world().resource::<BiomeAmbient>().0,
            [0.0; 3],
            "the composite is still being sent black"
        );
        assert_eq!(app.world().resource::<WeatherWeights>().0, mood.weather);
    }

    #[test]
    fn every_emitter_belongs_to_exactly_one_half_of_the_world() {
        // The bookkeeping the cost argument depends on: three surface emitters
        // partition one mix, four underground emitters partition the other, and
        // embers stand outside both because they key off lava.
        let surface = Emitter::ALL
            .iter()
            .filter(|e| e.is_surface() == Some(true))
            .count();
        let deep = Emitter::ALL
            .iter()
            .filter(|e| e.is_surface() == Some(false))
            .count();
        let neither = Emitter::ALL
            .iter()
            .filter(|e| e.is_surface().is_none())
            .count();
        assert_eq!((surface, deep, neither), (3, 4, 1));
    }

    #[test]
    fn the_rate_table_lines_up_with_the_emitter_slots() {
        // `PEAK_RATE` is indexed by discriminant. If a variant is ever inserted
        // in the middle, every rate silently moves to the wrong emitter.
        assert_eq!(PEAK_RATE.len(), EMITTER_COUNT);
        for (i, e) in Emitter::ALL.into_iter().enumerate() {
            assert_eq!(e.index(), i);
        }
        // And the peaks are the TypeScript's, spot-checked at both ends.
        assert_eq!(PEAK_RATE[Emitter::Firefly.index()], 9.0);
        assert_eq!(PEAK_RATE[Emitter::CaveDust.index()], 3.5);
    }

    #[test]
    fn only_the_luminous_kinds_ask_for_the_glow_pass() {
        // The flag the lighting milestone will route. Fireflies, embers, spores
        // and glints are the four things that must survive the darkness
        // multiply; dust, pollen, sand and water must not.
        for e in Emitter::ALL {
            let want = matches!(
                e,
                Emitter::Firefly | Emitter::Ember | Emitter::Spore | Emitter::Glint
            );
            assert_eq!(e.spec().glow, want, "{e:?} glow");
        }
    }
}

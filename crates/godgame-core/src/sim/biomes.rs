//! Climate-driven biomes.
//!
//! The old model bucketed ONE low-frequency 1D field into N equal slices, which
//! forced the world into ordered vertical stripes: you could never walk from
//! Desert straight into Tundra, because the slice index had to pass through
//! everything in between.
//!
//! This model samples TWO independent low-frequency fields at the absolute world
//! column — `temperature` and `moisture` — and looks the pair up in a
//! Whittaker-style climate space. Each biome owns a centroid in that space; a
//! column belongs to whichever centroid is nearest. Adjacency is now a property
//! of the climate diagram, not of an array index, so regions come out organic
//! and any two climatically-neighbouring biomes can touch.
//!
//! A third, sparser field drives `volcanism`. Volcanic is deliberately NOT a
//! climate cell (there is no "hot enough to be lava" corner of a real Whittaker
//! diagram, and putting it in one would make it as common as everything else);
//! instead it enters the same nearest-centroid competition through a synthetic
//! distance derived from that sparse field, so it only wins on the rare columns
//! where volcanism spikes — and, because it competes in the same machinery, it
//! blends into its neighbours exactly like any other biome.
//!
//! EVERY function here is a pure function of the absolute world column and the
//! seed's noise. Nothing is stateful, nothing looks at a neighbouring column's
//! *decision* (only at continuous fields), so two chunks generated in any order
//! agree perfectly across their shared edge.
//!
//! # Scratch buffers, and why there are none
//!
//! The TypeScript kept module-level `DIST`/`WEIGHT`/`HEIGHT_SCRATCH` arrays and
//! returned shared mutable records, because allocating per column in a JS engine
//! is what made worldgen slow. None of that survives here: every scratch buffer
//! is a fixed-size array on the stack and every result is returned BY VALUE.
//! Chunk generation runs on a rayon pool against a shared `&Noise`, so a module
//! static — even one behind a lock or in a `thread_local!` — would be either a
//! data race or a per-thread copy of the same determinism hazard. Returning a
//! `[f64; 8]` in registers costs less than the JS version's aliasing did anyway.

use super::materials::{CellId, block};
use super::noise::Noise;

// ---------------------------------------------------------------------------
// Material resolution
// ---------------------------------------------------------------------------
// In the TypeScript, materials were referenced BY STRING ID and resolved once at
// module load through a `pick(...ids)` preference list: the first id present in
// the registry won, and the last entry was always something that had always been
// there. That way adding "gravel" or "permafrost" upgraded the world
// automatically, and not having them yet was merely a downgrade rather than a
// crash.
//
// The registry is compiled content here, so every id resolves at COMPILE time
// and a missing one is a build error rather than a silent downgrade — strictly
// better, and it makes the fallbacks dead. They are recorded here anyway,
// because they document what each material was standing in for:
//
//   packedIce  <- ice
//   mud        <- sticky
//   glass      <- sand
//   gravel     <- sandstone
//   wetSand    <- sand
//   permafrost <- frozenDirt, mud, sticky
//   clay       <- sticky, dirt
//
// Everything else was a hard reference with no fallback.

// ---------------------------------------------------------------------------
// Biome definitions
// ---------------------------------------------------------------------------

/// A colour as [r, g, b].
///
/// Backdrop colours are 0..255 per channel; `ambient` is an additive light tint
/// in 0..1 per channel. Both live in one type because both are summed by the
/// same weighted crossfade in [`resolve_atmosphere`].
pub type Rgb = [f64; 3];

/// `0xrrggbb` -> [r, g, b].
///
/// The TypeScript parsed `#rrggbb` strings once at load into an `ATMO_RGB` side
/// table, because `resolveAtmosphere` runs every frame. Here the parse happens
/// at compile time and the side table is unnecessary.
#[inline]
const fn hex(n: u32) -> Rgb {
    [
        ((n >> 16) & 255) as f64,
        ((n >> 8) & 255) as f64,
        (n & 255) as f64,
    ]
}

/// The particle kind a biome's sky throws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Weather {
    #[default]
    None,
    Dust,
    Snow,
    Spores,
    Embers,
}

/// Look-and-feel of a biome's backdrop, sampled and blended by the renderer.
#[derive(Clone, Copy, Debug)]
pub struct BiomeAtmosphere {
    /// gradient top (shallow)
    pub sky_top: Rgb,
    /// gradient top when the camera is underground
    pub sky_top_deep: Rgb,
    /// gradient bottom
    pub sky_bottom: Rgb,
    /// starfield colour
    pub star: Rgb,
    /// parallax hill silhouette
    pub hill: Rgb,
    /// additive light tint (0..1 per channel)
    pub ambient: Rgb,
    pub weather: Weather,
}

/// Everything a biome is.
#[derive(Clone, Copy, Debug)]
pub struct BiomeDef {
    pub id: &'static str,
    pub name: &'static str,
    /// Topsoil cap material (the surface layer).
    pub cap: CellId,
    /// Subsurface rock immediately under the cap (the *surface* rock signature).
    pub rock: CellId,
    /// Liquid that pools in the lower cavern pockets near the surface.
    pub pocket: CellId,
    /// Surface heightmap amplitude multiplier (flatter <1, jagged >1).
    pub amp_scale: f64,
    /// Cells the mean ground line is pushed down (+) or lifted up (-).
    pub height_offset: f64,
    /// Multiplier on the topsoil cap thickness.
    pub cap_scale: f64,
    /// Multiplier on shallow cave openness (>1 = more/bigger caves up high).
    pub cave_scale: f64,
    /// Whether this biome grows trees on flat ground.
    pub trees: bool,
    /// Position in climate space: [temperature, moisture], both 0..1. `None`
    /// marks a biome that does not compete on climate at all (Volcanic — see
    /// the volcanic branch of [`biome_dist_at`]).
    pub climate: Option<[f64; 2]>,
    /// Subtracted from this biome's climate distance. Positive = the biome
    /// spreads (wins columns further from its centroid); negative = it shrinks.
    /// The knob that tunes region sizes without moving centroids.
    pub bias: f64,
    pub atmo: BiomeAtmosphere,
}

/// Which biome.
///
/// The discriminant IS the index into [`BIOMES`], and that is load-bearing: the
/// palette order is the stable order a [`Mix`] iterates in, and
/// [`biome_index_at`] hands the index out to callers that key flat tables off
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Biome {
    #[default]
    Plains = 0,
    Desert = 1,
    Tundra = 2,
    Swamp = 3,
    Volcanic = 4,
    Glacier = 5,
    Jungle = 6,
    Savanna = 7,
}

/// How many biomes compete for a column.
pub const BIOME_COUNT: usize = 8;

impl Biome {
    /// Every biome, in stable palette order. Entry `i` describes `BIOMES[i]`.
    pub const ALL: [Biome; BIOME_COUNT] = [
        Biome::Plains,
        Biome::Desert,
        Biome::Tundra,
        Biome::Swamp,
        Biome::Volcanic,
        Biome::Glacier,
        Biome::Jungle,
        Biome::Savanna,
    ];

    /// This biome's definition — materials, height knobs, atmosphere.
    #[inline]
    pub const fn def(self) -> &'static BiomeDef {
        &BIOMES[self as usize]
    }

    /// Index into [`BIOMES`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Climate space, sketched (temperature ->, moisture ^):
///
/// ```text
///   wet  |  Glacier            Swamp            Jungle
///        |
///        |             Plains
///        |
///   dry  |  Tundra             Savanna          Desert
///        +------------------------------------------------
///          cold                                     hot
/// ```
///
/// Volcanic sits outside the diagram and is driven by its own sparse field.
#[rustfmt::skip]
pub static BIOMES: [BiomeDef; BIOME_COUNT] = [
    BiomeDef {
        id: "plains", name: "Plains",
        cap: block::DIRT, rock: block::STONE, pocket: block::WATER,
        amp_scale: 1.0, height_offset: 0.0, cap_scale: 1.0, cave_scale: 1.0, trees: true,
        climate: Some([0.48, 0.46]), bias: 0.03,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x141826), sky_top_deep: hex(0x0a0c16), sky_bottom: hex(0x2a2f42),
            star: hex(0xdfe6ff), hill: hex(0x1b2030), ambient: [0.0, 0.0, 0.0],
            weather: Weather::None,
        },
    },
    BiomeDef {
        id: "desert", name: "Desert",
        cap: block::SAND, rock: block::SANDSTONE, pocket: block::WATER,
        amp_scale: 0.7, height_offset: 2.0, cap_scale: 1.6, cave_scale: 0.8, trees: false,
        climate: Some([0.93, 0.08]), bias: 0.0,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x3a2f1e), sky_top_deep: hex(0x1a1408), sky_bottom: hex(0x6b4a2a),
            star: hex(0xffe6c0), hill: hex(0x3a2a18), ambient: [0.06, 0.04, 0.0],
            weather: Weather::Dust,
        },
    },
    BiomeDef {
        id: "tundra", name: "Tundra",
        cap: block::SNOW, rock: block::STONE, pocket: block::WATER,
        amp_scale: 1.1, height_offset: -2.0, cap_scale: 0.7, cave_scale: 1.0, trees: false,
        climate: Some([0.1, 0.28]), bias: 0.0,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x1c2634), sky_top_deep: hex(0x0c1420), sky_bottom: hex(0x3a4a5e),
            star: hex(0xffffff), hill: hex(0x223040), ambient: [0.02, 0.04, 0.08],
            weather: Weather::Snow,
        },
    },
    BiomeDef {
        id: "swamp", name: "Swamp",
        cap: block::MOSS, rock: block::STONE, pocket: block::ACID,
        amp_scale: 0.45, height_offset: 4.0, cap_scale: 1.0, cave_scale: 0.7, trees: true,
        climate: Some([0.5, 0.9]), bias: 0.0,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x16221a), sky_top_deep: hex(0x08120a), sky_bottom: hex(0x2a3a2a),
            star: hex(0xb8e0a0), hill: hex(0x16241a), ambient: [0.0, 0.05, 0.0],
            weather: Weather::Spores,
        },
    },
    BiomeDef {
        id: "volcanic", name: "Volcanic",
        cap: block::BASALT, rock: block::BASALT, pocket: block::LAVA,
        amp_scale: 1.35, height_offset: -4.0, cap_scale: 1.2, cave_scale: 1.2, trees: false,
        climate: None, bias: 0.0,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x2a1414), sky_top_deep: hex(0x140606), sky_bottom: hex(0x4a1e14),
            star: hex(0xffb090), hill: hex(0x2a1210), ambient: [0.10, 0.02, 0.0],
            weather: Weather::Embers,
        },
    },

    // --- New biomes the 2D climate space makes room for -----------------------
    BiomeDef {
        // Cold + wet: the corner the 1D model could never express (it only had
        // one "cold" slot). Sheet ice over a snow crust, deep and jagged.
        id: "glacier", name: "Glacier",
        cap: block::PACKED_ICE, rock: block::ICE, pocket: block::WATER,
        amp_scale: 1.45, height_offset: -6.0, cap_scale: 1.4, cave_scale: 1.3, trees: false,
        climate: Some([0.08, 0.85]), bias: -0.01,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x16202e), sky_top_deep: hex(0x080e18), sky_bottom: hex(0x4c6480), // fixed
            star: hex(0xe8f6ff), hill: hex(0x2c3f56), ambient: [0.04, 0.07, 0.12],
            weather: Weather::Snow,
        },
    },
    BiomeDef {
        // Hot + wet: dense canopy on deep loam, warm haze.
        id: "jungle", name: "Jungle",
        cap: block::MOSS, rock: block::DIRT, pocket: block::WATER,
        amp_scale: 0.9, height_offset: 1.0, cap_scale: 1.5, cave_scale: 0.85, trees: true,
        climate: Some([0.87, 0.82]), bias: 0.0,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x122418), sky_top_deep: hex(0x06120b), sky_bottom: hex(0x3c5c34),
            star: hex(0xd6ffa8), hill: hex(0x183020), ambient: [0.02, 0.07, 0.01],
            weather: Weather::Spores,
        },
    },
    BiomeDef {
        // Hot + semi-dry: hardpan and scattered trees between Desert and Plains.
        id: "savanna", name: "Savanna",
        cap: block::DIRT, rock: block::SANDSTONE, pocket: block::WATER,
        amp_scale: 0.6, height_offset: 1.0, cap_scale: 0.9, cave_scale: 0.9, trees: true,
        climate: Some([0.76, 0.3]), bias: 0.0,
        atmo: BiomeAtmosphere {
            sky_top: hex(0x2b2618), sky_top_deep: hex(0x14110a), sky_bottom: hex(0x6a5a32),
            star: hex(0xffeec0), hill: hex(0x33301c), ambient: [0.05, 0.04, 0.01],
            weather: Weather::Dust,
        },
    },
];

/// Index of Volcanic in [`BIOMES`] — it is the one biome not placed by climate.
pub const VOLCANIC_INDEX: usize = Biome::Volcanic as usize;

// ---------------------------------------------------------------------------
// Ecotones — transitional ground for specific adjacencies
// ---------------------------------------------------------------------------
// Keyed by the two biomes as an UNORDERED pair, so lookup is order-independent
// (the dither swaps which biome is "a" as you cross the seam). The TypeScript
// got that by sorting the two string ids into a single map key; a scan of twenty
// pairs, once per column, is cheaper than any keyed structure at this size.
//
// In the TypeScript the values were resolved through `pick`, so an id another
// agent had not added yet degraded to a sensible existing material instead of
// throwing. Here they are compiled constants.
#[rustfmt::skip]
static ECOTONES: &[(Biome, Biome, CellId)] = &[
    (Biome::Desert,  Biome::Plains,   block::GRAVEL),     //  dusty hardpan
    (Biome::Plains,  Biome::Savanna,  block::GRAVEL),
    (Biome::Desert,  Biome::Savanna,  block::GRAVEL),
    (Biome::Plains,  Biome::Swamp,    block::MUD),        //     churned mud
    (Biome::Jungle,  Biome::Swamp,    block::MUD),
    (Biome::Jungle,  Biome::Plains,   block::CLAY),
    (Biome::Plains,  Biome::Tundra,   block::PERMAFROST), // frozen ground
    (Biome::Swamp,   Biome::Tundra,   block::PERMAFROST),
    (Biome::Glacier, Biome::Tundra,   block::PACKED_ICE),
    (Biome::Glacier, Biome::Plains,   block::PERMAFROST),
    (Biome::Desert,  Biome::Swamp,    block::WET_SAND),   //  salt flat
    (Biome::Desert,  Biome::Jungle,   block::WET_SAND),
    (Biome::Savanna, Biome::Swamp,    block::CLAY),
    // Volcanic margins are scorched, whatever they border.
    (Biome::Plains,  Biome::Volcanic, block::ASH),
    (Biome::Desert,  Biome::Volcanic, block::ASH),
    (Biome::Swamp,   Biome::Volcanic, block::ASH),
    (Biome::Tundra,  Biome::Volcanic, block::ASH),
    (Biome::Jungle,  Biome::Volcanic, block::ASH),
    (Biome::Savanna, Biome::Volcanic, block::ASH),
    (Biome::Glacier, Biome::Volcanic, block::ASH),
];

/// Transitional cap material for an adjacency, or `None` when the two biomes are
/// the same or the pair has no ecotone. (The TypeScript returned `-1` for both
/// cases; `Option` says the same thing without borrowing a material code as a
/// sentinel.)
fn ecotone_for(a: Biome, b: Biome) -> Option<CellId> {
    if a == b {
        return None;
    }
    for &(x, y, mat) in ECOTONES {
        if (x == a && y == b) || (x == b && y == a) {
            return Some(mat);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Underground layers — vertical biomes
// ---------------------------------------------------------------------------

/// A vertical biome.
///
/// The character of the deep world is chosen from its OWN climate fields, at a
/// different frequency and phase from the surface. A Desert can therefore sit on
/// top of flooded grottos, and a Swamp on magma chambers — the surface no longer
/// dictates what is beneath it. The handover happens over a depth band whose
/// start wobbles with the column, so the interface is a wandering strata line
/// rather than a ruled horizontal cut.
#[derive(Clone, Copy, Debug)]
pub struct UndergroundLayer {
    pub id: &'static str,
    pub name: &'static str,
    /// Bulk rock of the layer.
    pub rock: CellId,
    /// Liquid pooling in the layer's lower cavities.
    pub pocket: CellId,
    /// Rare vein material (richest).
    pub vein_rich: CellId,
    /// Common vein material.
    pub vein_common: CellId,
    /// Material filling the "negative" vein band (soft/loose pockets).
    pub vein_soft: CellId,
    /// Multiplier on cave openness in this layer.
    pub cave_scale: f64,
    /// Added to the deep band's lava threshold (negative = more lava).
    pub pocket_bias: f64,
    /// Position in underground climate space: [heat, wetness], 0..1.
    pub climate: [f64; 2],
    pub bias: f64,
}

/// Which underground layer. Discriminant = index into [`UNDERGROUND_LAYERS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum UndergroundLayerId {
    #[default]
    Caverns = 0,
    Grottos = 1,
    Magma = 2,
    Geode = 3,
    Fungal = 4,
}

/// How many underground layers compete for a column.
pub const UG_COUNT: usize = 5;

impl UndergroundLayerId {
    /// Every layer, in stable palette order.
    pub const ALL: [UndergroundLayerId; UG_COUNT] = [
        UndergroundLayerId::Caverns,
        UndergroundLayerId::Grottos,
        UndergroundLayerId::Magma,
        UndergroundLayerId::Geode,
        UndergroundLayerId::Fungal,
    ];

    /// This layer's definition — rock, pocket liquid, veins.
    #[inline]
    pub const fn def(self) -> &'static UndergroundLayer {
        &UNDERGROUND_LAYERS[self as usize]
    }

    /// Index into [`UNDERGROUND_LAYERS`].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The deep world's palette, in stable order.
#[rustfmt::skip]
pub static UNDERGROUND_LAYERS: [UndergroundLayer; UG_COUNT] = [
    UndergroundLayer {
        id: "caverns", name: "Stone Caverns",
        rock: block::STONE, pocket: block::WATER,
        vein_rich: block::CRYSTAL, vein_common: block::OBSIDIAN, vein_soft: block::SAND,
        cave_scale: 1.0, pocket_bias: 0.0, climate: [0.5, 0.42], bias: 0.04,
    },
    UndergroundLayer {
        id: "grottos", name: "Flooded Grottos",
        rock: block::STONE, pocket: block::WATER,
        vein_rich: block::CRYSTAL, vein_common: block::GRAVEL, vein_soft: block::WET_SAND,
        // Negative pocket_bias raises the liquid table (see worldgen::caves): the
        // Flooded Grottos are the one layer that should actually be flooded.
        cave_scale: 1.35, pocket_bias: -0.1, climate: [0.32, 0.92], bias: 0.0,
    },
    UndergroundLayer {
        id: "magma", name: "Magma Chambers",
        rock: block::BASALT, pocket: block::LAVA,
        vein_rich: block::CRYSTAL, vein_common: block::OBSIDIAN, vein_soft: block::ASH,
        cave_scale: 1.1, pocket_bias: -0.14, climate: [0.93, 0.18], bias: 0.0,
    },
    UndergroundLayer {
        id: "geode", name: "Geode Hollows",
        rock: block::STONE, pocket: block::WATER,
        vein_rich: block::CRYSTAL, vein_common: block::CRYSTAL, vein_soft: block::GLASS,
        cave_scale: 0.85, pocket_bias: 0.1, climate: [0.12, 0.22], bias: -0.02,
    },
    UndergroundLayer {
        id: "fungal", name: "Fungal Depths",
        rock: block::DIRT, pocket: block::ACID,
        vein_rich: block::CRYSTAL, vein_common: block::MOSS, vein_soft: block::MUD,
        cave_scale: 0.95, pocket_bias: 0.06, climate: [0.72, 0.74], bias: 0.0,
    },
];

// ---------------------------------------------------------------------------
// Climate fields
// ---------------------------------------------------------------------------
// Three low-frequency fields sampled at the absolute column. Frequencies and
// noise-space anchors are mutually prime-ish so the fields are decorrelated:
// `fbm2` hashes (floor(x), floor(y)) through the permutation table, and each
// octave doubles both, so distinct anchors give genuinely independent slices.
const TEMP_FREQ: f64 = 0.0011;
const TEMP_ANCHOR: f64 = 41.7;
const MOIST_FREQ: f64 = 0.0017;
const MOIST_ANCHOR: f64 = 613.3;
const VOLC_FREQ: f64 = 0.00062;
const VOLC_ANCHOR: f64 = 907.1;
const CLIMATE_OCTAVES: u32 = 2;

/// Contrast gain applied to a raw fBm value before it becomes a 0..1 climate
/// coordinate. Value-noise fBm clusters hard around 0, so without this the world
/// would be nothing but the biomes near the middle of the diagram; too much of it
/// and the middle biomes starve instead.
const CLIMATE_SPREAD: f64 = 1.5;

/// Width, in climate-distance units, of the band around a boundary where two (or
/// more) biomes both contribute. Roughly 40-70 world cells of visible transition
/// at the climate frequencies above.
const BLEND_WIDTH: f64 = 0.085;

// Volcanic's synthetic distance: `VOLC_D0 - (volcanism - VOLC_T0) * VOLC_GAIN`,
// floored at 0, competing against real climate distances (median ~0.19).
//
// Volcanism gets NO contrast gain — climate wants a spread field so its corners
// are reachable, volcanism wants a raw bell so its tail stays thin. VOLC_T0 sits
// at roughly the field's 97th percentile, where Volcanic starts to draw level
// with the climate winner, and VOLC_GAIN makes it certain a little past that.
// Net effect: a couple of percent of columns, in isolated hotspots.
const VOLC_SPREAD: f64 = 0.5;
const VOLC_T0: f64 = 0.835;
const VOLC_D0: f64 = 0.19;
const VOLC_GAIN: f64 = 1.65;

#[inline]
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

#[inline]
fn smoothstep01(v: f64) -> f64 {
    let t = clamp01(v);
    t * t * (3.0 - 2.0 * t)
}

/// Raw fBm in ~[-1,1] -> a spread, clamped climate coordinate in [0,1].
#[inline]
fn shape01(v: f64, gain: f64) -> f64 {
    clamp01(0.5 + v * gain)
}

/// Temperature at an absolute world column, in [0,1].
#[inline]
pub fn temperature_at(noise: &Noise, wcx: i32) -> f64 {
    shape01(
        noise.fbm2(wcx as f64 * TEMP_FREQ, TEMP_ANCHOR, CLIMATE_OCTAVES),
        CLIMATE_SPREAD,
    )
}

/// Moisture at an absolute world column, in [0,1].
#[inline]
pub fn moisture_at(noise: &Noise, wcx: i32) -> f64 {
    shape01(
        noise.fbm2(wcx as f64 * MOIST_FREQ, MOIST_ANCHOR, CLIMATE_OCTAVES),
        CLIMATE_SPREAD,
    )
}

/// Volcanism at an absolute world column, in [0,1]. Sparser than the climate
/// fields, and deliberately un-spread — see [`VOLC_SPREAD`].
#[inline]
pub fn volcanism_at(noise: &Noise, wcx: i32) -> f64 {
    shape01(
        noise.fbm2(wcx as f64 * VOLC_FREQ, VOLC_ANCHOR, 2),
        VOLC_SPREAD,
    )
}

// ---------------------------------------------------------------------------
// Continuous mixing in climate space
// ---------------------------------------------------------------------------

/// Capacity of every scratch array below: the larger of the two palettes.
const POOL_MAX: usize = if BIOME_COUNT > UG_COUNT {
    BIOME_COUNT
} else {
    UG_COUNT
};

/// The set of palette entries contributing at a column, with normalised weights.
///
/// WHY A WEIGHT SET AND NOT JUST "NEAREST TWO": the obvious design is to take the
/// nearest and second-nearest centroid and lerp between them. That is continuous
/// across a simple two-way boundary, but it BREAKS at a three-way junction — the
/// runner-up's identity swaps while its weight is still large, so every blended
/// quantity jumps. Weighting every entry whose distance is within BLEND_WIDTH of
/// the winner, and letting each weight fall smoothly to exactly zero at that
/// cutoff, is continuous everywhere, junctions included.
///
/// [`Mix::items`] is in stable palette order and [`Mix::cum`] holds the running
/// total of the normalised weights, so a per-cell dither value in [0,1) can
/// select an entry by a single scan: the *proportion* of cells each entry gets
/// varies continuously with the weights, which is what makes the dithered
/// boundary seamless.
#[derive(Clone, Copy, Debug)]
pub struct Mix<T: Copy> {
    items: [T; POOL_MAX],
    cum: [f64; POOL_MAX],
    len: usize,
    /// Highest-weight entry.
    pub top: T,
    /// Second-highest-weight entry, or `top` when it stands alone.
    pub second: T,
    /// Normalised weight of `second` (0 when `top` stands alone). Range [0, 0.5].
    pub second_w: f64,
}

impl<T: Copy> Mix<T> {
    /// Contributing entries, in stable palette order. Length >= 1.
    #[inline]
    pub fn items(&self) -> &[T] {
        &self.items[..self.len]
    }

    /// Cumulative normalised weight, parallel to [`Mix::items`]. The last
    /// element is exactly 1.
    #[inline]
    pub fn cum(&self) -> &[f64] {
        &self.cum[..self.len]
    }

    /// How many entries contribute here. 1 means no active blend.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Never true — the nearest centroid always weighs 1, so a mix always has at
    /// least one entry. Present because a bare `len` invites the lint.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// See [`pick_from_mix`].
    #[inline]
    pub fn pick(&self, v: f64) -> T {
        pick_from_mix(self, v)
    }
}

/// The `WEIGHT` scratch buffer, as a value.
///
/// The TypeScript kept this as a module-level `Float64Array` that `mixOf`,
/// `heightFromWeights` and `heightParamsAt` all read after whoever ran last
/// wrote it — hence the "read the height params BEFORE `undergroundMixAt`
/// clobbers WEIGHT" ordering hazard inside `columnProfileAt`. Passing the buffer
/// by value deletes the hazard outright and keeps the arithmetic identical.
#[derive(Clone, Copy, Debug)]
struct Weights {
    w: [f64; POOL_MAX],
    n: usize,
    total: f64,
}

impl Weights {
    /// Fill `w[0..n)` from `dist[0..n)` and return them with their total.
    ///
    /// Split out of [`mix_of`] so [`height_params_at`] and [`column_profile_at`]
    /// can run the IDENTICAL arithmetic in the identical order. That is not
    /// tidiness: the two entry points both feed `surface_row_at`, whose result is
    /// rounded to a cell row, and a last-bit disagreement between them would let
    /// the same column round to two different ground rows depending on which
    /// caller asked. Same code, same order, same float.
    fn fill(dist: &[f64]) -> Weights {
        let n = dist.len();
        debug_assert!(n > 0 && n <= POOL_MAX);

        let mut d_min = f64::INFINITY;
        for &d in dist {
            if d < d_min {
                d_min = d;
            }
        }

        let mut w = [0.0f64; POOL_MAX];
        let mut total = 0.0;
        for i in 0..n {
            // 1 for the winner, easing to exactly 0 at BLEND_WIDTH of separation.
            let wi = smoothstep01(1.0 - (dist[i] - d_min) / BLEND_WIDTH);
            w[i] = wi;
            total += wi;
        }
        Weights { w, n, total }
    }
}

/// Build a [`Mix`] over `pool` from weights already filled for it.
fn mix_of<T: Copy + Default>(pool: &[T], wts: &Weights) -> Mix<T> {
    debug_assert_eq!(pool.len(), wts.n);
    let total = wts.total;

    let mut items = [T::default(); POOL_MAX];
    let mut cum = [0.0f64; POOL_MAX];
    let mut len = 0usize;
    let mut run = 0.0f64;
    let mut top_i: Option<usize> = None;
    let mut second_i: Option<usize> = None;
    for (i, &entry) in pool.iter().enumerate() {
        let w = wts.w[i];
        if w <= 0.0 {
            continue;
        }
        run += w / total;
        items[len] = entry;
        cum[len] = run;
        len += 1;
        if top_i.is_none_or(|t| w > wts.w[t]) {
            second_i = top_i;
            top_i = Some(i);
        } else if second_i.is_none_or(|s| w > wts.w[s]) {
            second_i = Some(i);
        }
    }
    cum[len - 1] = 1.0; // kill float drift so the last bucket always catches

    // The nearest centroid always weighs exactly 1, so `top_i` is always set.
    let top = pool[top_i.expect("the winning entry always has weight 1")];
    Mix {
        items,
        cum,
        len,
        top,
        second: match second_i {
            Some(s) => pool[s],
            None => top,
        },
        second_w: match second_i {
            Some(s) => wts.w[s] / total,
            None => 0.0,
        },
    }
}

/// Select an entry from a mix with a per-cell dither value in [0,1). Returns the
/// sole entry immediately in the common case of no active blend.
#[inline]
pub fn pick_from_mix<T: Copy>(mix: &Mix<T>, v: f64) -> T {
    let cum = mix.cum();
    let n = cum.len();
    if n == 1 {
        return mix.items[0];
    }
    for (&c, &item) in cum[..n - 1].iter().zip(mix.items().iter()) {
        if v < c {
            return item;
        }
    }
    mix.items[n - 1]
}

/// Weighted mean of a numeric field over a mix. Continuous in the weights.
fn mix_scalar<T: Copy>(mix: &Mix<T>, read: impl Fn(T) -> f64) -> f64 {
    let items = mix.items();
    if items.len() == 1 {
        return read(items[0]);
    }
    let cum = mix.cum();
    let mut sum = 0.0;
    let mut prev = 0.0;
    for i in 0..items.len() {
        sum += read(items[i]) * (cum[i] - prev);
        prev = cum[i];
    }
    sum
}

/// Every biome's climate distance at `wcx`, in palette order.
fn biome_dist_at(noise: &Noise, wcx: i32) -> [f64; BIOME_COUNT] {
    let temp = temperature_at(noise, wcx);
    let moist = moisture_at(noise, wcx);
    let volc = volcanism_at(noise, wcx);

    let mut dist = [0.0f64; BIOME_COUNT];
    for i in 0..BIOME_COUNT {
        match BIOMES[i].climate {
            None => {
                // Volcanic: not a climate cell. Its distance collapses as the
                // sparse volcanism field spikes, so it out-competes the climate
                // winner only on the rare columns where that happens — and
                // blends there like any other.
                let d = VOLC_D0 - (volc - VOLC_T0) * VOLC_GAIN;
                dist[i] = if d < 0.0 { 0.0 } else { d };
            }
            Some(c) => {
                let dx = temp - c[0];
                let dy = moist - c[1];
                dist[i] = (dx * dx + dy * dy).sqrt() - BIOMES[i].bias;
            }
        }
    }
    dist
}

/// Every underground layer's climate distance at `wcx`, in palette order.
fn underground_dist_at(noise: &Noise, wcx: i32) -> [f64; UG_COUNT] {
    // Independent fields: different frequencies AND different noise-space anchors
    // from the surface climate, so the deep world is genuinely decoupled from what
    // grows on top of it.
    let heat = shape01(
        noise.fbm2(wcx as f64 * 0.00083 + 12.5, 2311.7, CLIMATE_OCTAVES),
        CLIMATE_SPREAD,
    );
    let wet = shape01(
        noise.fbm2(wcx as f64 * 0.00131 - 31.25, 1487.3, CLIMATE_OCTAVES),
        CLIMATE_SPREAD,
    );

    let mut dist = [0.0f64; UG_COUNT];
    for i in 0..UG_COUNT {
        let c = UNDERGROUND_LAYERS[i].climate;
        let dx = heat - c[0];
        let dy = wet - c[1];
        dist[i] = (dx * dx + dy * dy).sqrt() - UNDERGROUND_LAYERS[i].bias;
    }
    dist
}

/// Surface biome mix for an absolute column. Pure in `wcx`.
pub fn biome_mix_at(noise: &Noise, wcx: i32) -> Mix<Biome> {
    let wts = Weights::fill(&biome_dist_at(noise, wcx));
    mix_of(&Biome::ALL, &wts)
}

/// Underground layer mix for an absolute column. Pure in `wcx`.
pub fn underground_mix_at(noise: &Noise, wcx: i32) -> Mix<UndergroundLayerId> {
    let wts = Weights::fill(&underground_dist_at(noise, wcx));
    mix_of(&UndergroundLayerId::ALL, &wts)
}

// ---------------------------------------------------------------------------
// Per-column profile — everything worldgen's vertical scan needs, computed once
// ---------------------------------------------------------------------------

/// All per-column terrain parameters, already blended. `generate_chunk` builds
/// one of these per world column and then runs a cheap vertical scan against it,
/// so the multi-octave climate work stays O(1) per COLUMN, never per cell.
#[derive(Clone, Copy, Debug)]
pub struct ColumnProfile {
    /// Absolute world column this profile describes.
    pub wcx: i32,

    /// Surface biomes contributing here, with their weights.
    pub surf: Mix<Biome>,
    /// Dominant surface biome (`surf.top`) — trees, weather kind, debug.
    pub surf_a: Biome,
    /// Runner-up surface biome.
    pub surf_b: Biome,
    /// Weight owed to `surf_b`, in [0, 0.5]. 0.5 exactly on a two-way boundary.
    pub surf_t: f64,
    /// Transitional cap material for the dominant adjacency, or `None` if the
    /// pair has no ecotone. (`-1` in the TypeScript.)
    pub eco_cap: Option<CellId>,

    // Continuous height params — no cliff at a boundary.
    pub amp_scale: f64,
    pub height_offset: f64,
    pub cap_thickness: f64,

    // Cave openness, blended surface-side and layer-side; mixed by depth at runtime.
    pub surf_cave_scale: f64,
    pub ug_cave_scale: f64,

    /// Underground layers contributing here, with their weights.
    pub ug: Mix<UndergroundLayerId>,
    /// Weight owed to the runner-up layer, in [0, 0.5].
    pub ug_t: f64,
    pub ug_pocket_bias: f64,

    /// Depth (below the surface) where the layer starts displacing the biome.
    pub ug_fade_start: f64,
    /// Depth by which the layer has fully displaced the biome's rock signature.
    pub ug_fade_end: f64,
}

/// Base topsoil thickness in cells; scaled per biome. Mirrors
/// [`crate::config::worldgen::TOPSOIL`].
const CAP_BASE: f64 = 6.0;

// Depth window (below the surface) over which the surface rock signature hands
// off to the underground layer. The start wobbles per column so the interface
// is a wandering strata line, not a ruled cut.
const UG_FADE_START: f64 = 22.0;
const UG_FADE_SPAN: f64 = 46.0;
const UG_FADE_JITTER: f64 = 13.0;
const UG_JITTER_FREQ: f64 = 0.013;
const UG_JITTER_ANCHOR: f64 = 317.4;

#[inline]
fn read_cap_scale(b: Biome) -> f64 {
    b.def().cap_scale
}

#[inline]
fn read_biome_cave(b: Biome) -> f64 {
    b.def().cave_scale
}

#[inline]
fn read_layer_cave(l: UndergroundLayerId) -> f64 {
    l.def().cave_scale
}

#[inline]
fn read_pocket_bias(l: UndergroundLayerId) -> f64 {
    l.def().pocket_bias
}

/// The blended surface height knobs for a column.
///
/// Two `f64`s, returned in registers. The TypeScript handed back a shared scratch
/// object under a "copy what you need before calling again" contract, for the
/// same reason it shared everything else: the lighting pass calls this for every
/// light column of every frame and could not afford an allocation. There is
/// nothing to allocate here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightParams {
    pub amp_scale: f64,
    pub height_offset: f64,
}

/// Weighted mean of `amp_scale` and `height_offset` straight out of the weights.
/// Allocation-free and — critically — the ONLY place those two numbers are
/// computed, so every caller gets the same float. Assumes `wts` was filled over
/// [`BIOMES`].
fn height_from_weights(wts: &Weights) -> HeightParams {
    let mut total = 0.0;
    let mut amp = 0.0;
    let mut off = 0.0;
    for (b, &w) in BIOMES.iter().zip(wts.w[..BIOME_COUNT].iter()) {
        if w <= 0.0 {
            continue;
        }
        total += w;
        amp += w * b.amp_scale;
        off += w * b.height_offset;
    }
    HeightParams {
        amp_scale: amp / total,
        height_offset: off / total,
    }
}

/// Build the fully-blended profile for absolute column `wcx`. Pure.
///
/// This is the one place in worldgen that produces a fresh value per column;
/// everything under it works in fixed-size arrays on the stack.
pub fn column_profile_at(noise: &Noise, wcx: i32) -> ColumnProfile {
    let wts = Weights::fill(&biome_dist_at(noise, wcx));
    let s = mix_of(&Biome::ALL, &wts);
    // Read the height params out of the SAME weights the surface mix came from.
    // (In the TypeScript this line had to run before `undergroundMixAt` clobbered
    // the shared WEIGHT buffer. Nothing can clobber it here — but the identity it
    // was protecting, that these are bit-for-bit the floats `height_params_at`
    // computes, is still the whole point.)
    let h = height_from_weights(&wts);
    let u = underground_mix_at(noise, wcx);
    let jitter = noise.n1(wcx as f64 * UG_JITTER_FREQ + UG_JITTER_ANCHOR) * UG_FADE_JITTER;
    let fade_start = UG_FADE_START + jitter;

    ColumnProfile {
        wcx,
        surf: s,
        surf_a: s.top,
        surf_b: s.second,
        surf_t: s.second_w,
        eco_cap: ecotone_for(s.top, s.second),

        amp_scale: h.amp_scale,
        height_offset: h.height_offset,
        cap_thickness: CAP_BASE * mix_scalar(&s, read_cap_scale),

        surf_cave_scale: mix_scalar(&s, read_biome_cave),
        ug_cave_scale: mix_scalar(&u, read_layer_cave),

        ug: u,
        ug_t: u.second_w,
        ug_pocket_bias: mix_scalar(&u, read_pocket_bias),

        ug_fade_start: fade_start,
        ug_fade_end: fade_start + UG_FADE_SPAN,
    }
}

/// Just the surface height parameters for a column — the cheap path for callers
/// (lighting, tree placement) that only need the ground line and would otherwise
/// pay for a whole profile.
///
/// Runs the same weighting as [`biome_mix_at`] and sums straight out of the
/// weights, skipping the mix entirely: the lighting pass calls this for every
/// light column of every frame.
pub fn height_params_at(noise: &Noise, wcx: i32) -> HeightParams {
    let wts = Weights::fill(&biome_dist_at(noise, wcx));
    height_from_weights(&wts)
}

// ---------------------------------------------------------------------------
// Atmosphere
// ---------------------------------------------------------------------------

/// Atmosphere colours, blended for the frame.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedAtmosphere {
    pub sky_top: Rgb,
    pub sky_top_deep: Rgb,
    pub sky_bottom: Rgb,
    pub star: Rgb,
    pub hill: Rgb,
    /// additive light tint (0..1 per channel)
    pub ambient: Rgb,
    pub weather: Weather,
}

#[inline]
fn add_scaled(dst: &mut Rgb, src: Rgb, w: f64) {
    dst[0] += src[0] * w;
    dst[1] += src[1] * w;
    dst[2] += src[2] * w;
}

/// Resolve the atmosphere at a column into numeric, frame-ready colours, blended
/// across a biome boundary.
///
/// This crossfades over the FULL weight set, not just the top two, so the sky is
/// continuous even where three climate regions meet — the same reason the terrain
/// blend uses a weight set (see [`Mix`]). `weather` stays the dominant biome's,
/// so the particle kind switches cleanly rather than dissolving into a mush.
pub fn resolve_atmosphere(noise: &Noise, wcx: i32) -> ResolvedAtmosphere {
    let mix = biome_mix_at(noise, wcx);
    let mut sky_top: Rgb = [0.0, 0.0, 0.0];
    let mut sky_top_deep: Rgb = [0.0, 0.0, 0.0];
    let mut sky_bottom: Rgb = [0.0, 0.0, 0.0];
    let mut star: Rgb = [0.0, 0.0, 0.0];
    let mut hill: Rgb = [0.0, 0.0, 0.0];
    let mut ambient: Rgb = [0.0, 0.0, 0.0];

    let mut prev = 0.0;
    for i in 0..mix.len() {
        let w = mix.cum()[i] - prev;
        prev = mix.cum()[i];
        let c = &mix.items()[i].def().atmo;
        add_scaled(&mut sky_top, c.sky_top, w);
        add_scaled(&mut sky_top_deep, c.sky_top_deep, w);
        add_scaled(&mut sky_bottom, c.sky_bottom, w);
        add_scaled(&mut star, c.star, w);
        add_scaled(&mut hill, c.hill, w);
        add_scaled(&mut ambient, c.ambient, w);
    }

    ResolvedAtmosphere {
        sky_top,
        sky_top_deep,
        sky_bottom,
        star,
        hill,
        ambient,
        weather: mix.top.def().atmo.weather,
    }
}

/// A backdrop colour as channel bytes, truncated.
///
/// The TypeScript exported `rgb(c)`, which built the CSS string `rgb(r,g,b)` for
/// a canvas fill. Nothing downstream of this renderer wants a string; the useful
/// half of that function was `c[0] | 0`, the truncation to a channel byte, which
/// is what this is.
#[inline]
pub fn rgb_u8(c: Rgb) -> [u8; 3] {
    [c[0] as u8, c[1] as u8, c[2] as u8]
}

// ---------------------------------------------------------------------------
// Back-compatible API
// ---------------------------------------------------------------------------

/// Dominant + runner-up biome and the weight owed to the runner-up.
#[derive(Clone, Copy, Debug)]
pub struct BiomePair {
    pub a: Biome,
    pub b: Biome,
    pub t: f64,
}

/// Dominant + runner-up biome and the weight owed to the runner-up. Kept at its
/// original shape for callers that only want "which two biomes am I between".
/// The crossfade is driven by climate distance rather than by how far a column
/// sits into a fixed-width band, and it reaches an exact 50/50 on a boundary.
/// Prefer [`biome_mix_at`] for anything that must stay continuous at a three-way
/// junction — `b`'s identity can swap there, [`Mix`] handles it.
pub fn biome_at(noise: &Noise, wcx: i32) -> BiomePair {
    let m = biome_mix_at(noise, wcx);
    BiomePair {
        a: m.top,
        b: m.second,
        t: m.second_w,
    }
}

/// Index of the DOMINANT biome at an absolute column.
pub fn biome_index_at(noise: &Noise, wcx: i32) -> usize {
    biome_mix_at(noise, wcx).top.index()
}

/// The biome at a palette index, or Plains if the index is out of range.
pub fn biome_at_index(i: usize) -> Biome {
    *Biome::ALL.get(i).unwrap_or(&Biome::ALL[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::worldgen::SEED;

    fn noise() -> Noise {
        Noise::new(SEED)
    }

    #[test]
    fn the_palette_index_is_the_enum_discriminant() {
        // Flat tables depend on this: `BIOMES[b as usize]` is `b.def()`, and
        // `biome_index_at` hands the index to callers that key off it.
        for (i, b) in Biome::ALL.iter().enumerate() {
            assert_eq!(b.index(), i);
            assert_eq!(b.def().id, BIOMES[i].id);
        }
        for (i, l) in UndergroundLayerId::ALL.iter().enumerate() {
            assert_eq!(l.index(), i);
            assert_eq!(l.def().id, UNDERGROUND_LAYERS[i].id);
        }
        assert_eq!(BIOMES[VOLCANIC_INDEX].id, "volcanic");
        // Volcanic is the ONE biome not placed by climate.
        assert_eq!(
            BIOMES.iter().filter(|b| b.climate.is_none()).count(),
            1,
            "exactly one biome may sit outside the climate diagram"
        );
    }

    #[test]
    fn the_climate_constants_are_the_tuned_ones() {
        // These ARE the world's biome layout. A typo here is a different planet.
        assert_eq!(TEMP_FREQ, 0.0011);
        assert_eq!(TEMP_ANCHOR, 41.7);
        assert_eq!(MOIST_FREQ, 0.0017);
        assert_eq!(MOIST_ANCHOR, 613.3);
        assert_eq!(VOLC_FREQ, 0.00062);
        assert_eq!(VOLC_ANCHOR, 907.1);
        assert_eq!(CLIMATE_OCTAVES, 2);
        assert_eq!(CLIMATE_SPREAD, 1.5);
        assert_eq!(BLEND_WIDTH, 0.085);
    }

    #[test]
    fn climate_coordinates_stay_in_the_unit_square() {
        let n = noise();
        let mut wcx = -30_000;
        while wcx < 30_000 {
            for v in [
                temperature_at(&n, wcx),
                moisture_at(&n, wcx),
                volcanism_at(&n, wcx),
            ] {
                assert!((0.0..=1.0).contains(&v), "climate escaped [0,1] at {wcx}");
            }
            wcx += 37;
        }
    }

    #[test]
    fn the_weights_sum_to_one() {
        // `cum` is a partition of [0,1]: strictly increasing, last element
        // exactly 1. `pick_from_mix` scanning it is a valid selection only if
        // that holds — a last element of 0.9999 would drop cells on the floor.
        let n = noise();
        let mut wcx = -20_000;
        while wcx < 20_000 {
            let s = biome_mix_at(&n, wcx);
            let u = underground_mix_at(&n, wcx);
            for cum in [s.cum(), u.cum()] {
                assert!(!cum.is_empty(), "an empty mix at {wcx}");
                assert_eq!(cum[cum.len() - 1], 1.0, "cum does not reach 1 at {wcx}");
                let mut prev = 0.0;
                for &c in cum {
                    assert!(c > prev, "cum is not increasing at {wcx}");
                    prev = c;
                }
            }
            wcx += 11;
        }
    }

    #[test]
    fn the_runner_up_never_out_weighs_the_winner() {
        let n = noise();
        let mut wcx = -20_000;
        while wcx < 20_000 {
            let m = biome_mix_at(&n, wcx);
            assert!(
                (0.0..=0.5 + 1e-12).contains(&m.second_w),
                "second_w = {} at {wcx}",
                m.second_w
            );
            if m.len() == 1 {
                assert_eq!(m.second_w, 0.0);
                assert_eq!(m.second, m.top);
            }
            wcx += 13;
        }
    }

    /// Spread of a per-entry number across a whole palette — the size of the
    /// jump a NON-blended lookup would make when the winner changes identity.
    fn span<T: Copy>(pool: &[T], read: impl Fn(T) -> f64) -> f64 {
        let lo = pool.iter().map(|&e| read(e)).fold(f64::INFINITY, f64::min);
        let hi = pool
            .iter()
            .map(|&e| read(e))
            .fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    }

    #[test]
    fn blending_is_continuous_across_a_boundary() {
        // The whole point of the weight set: every blended quantity is a
        // continuous function of the column, INCLUDING where the dominant biome
        // changes identity and where three regions meet. A jump here is a cliff
        // in the terrain and a seam in the ground materials.
        //
        // Tolerances are a FRACTION OF EACH QUANTITY'S PALETTE SPAN, because the
        // failure being tested for is a step of order the full span (Glacier's
        // height_offset is -6 and Desert's is +2 — an unblended swap between them
        // moves the ground line by 8 cells in one column). A continuous blend at
        // these climate frequencies moves a few percent of the span per column;
        // measured worst case over 120k columns is ~7%.
        let n = noise();
        let biomes = &Biome::ALL[..];
        let layers = &UndergroundLayerId::ALL[..];
        let tol = 0.15;
        let mut prev = column_profile_at(&n, -6001);
        let mut boundaries = 0;
        let mut worst: f64 = 0.0;
        for wcx in -6000..6000 {
            let p = column_profile_at(&n, wcx);
            if p.surf_a != prev.surf_a {
                boundaries += 1;
            }
            for (a, b, sp, what) in [
                (
                    p.amp_scale,
                    prev.amp_scale,
                    span(biomes, |b| b.def().amp_scale),
                    "amp_scale",
                ),
                (
                    p.height_offset,
                    prev.height_offset,
                    span(biomes, |b| b.def().height_offset),
                    "height_offset",
                ),
                (
                    p.cap_thickness,
                    prev.cap_thickness,
                    CAP_BASE * span(biomes, read_cap_scale),
                    "cap_thickness",
                ),
                (
                    p.surf_cave_scale,
                    prev.surf_cave_scale,
                    span(biomes, read_biome_cave),
                    "surf_cave_scale",
                ),
                (
                    p.ug_cave_scale,
                    prev.ug_cave_scale,
                    span(layers, read_layer_cave),
                    "ug_cave_scale",
                ),
                (
                    p.ug_pocket_bias,
                    prev.ug_pocket_bias,
                    span(layers, read_pocket_bias),
                    "ug_pocket_bias",
                ),
            ] {
                let step = (a - b).abs();
                worst = worst.max(step / sp);
                assert!(
                    step < tol * sp,
                    "{what} jumped by {step} ({:.1}% of its {sp} span) at column {wcx}",
                    100.0 * step / sp
                );
            }
            prev = p;
        }
        assert!(
            boundaries > 4,
            "the sweep crossed only {boundaries} biome boundaries, so it proved \
             nothing about continuity"
        );
        assert!(worst > 0.0, "nothing varied at all across 12k columns");
    }

    #[test]
    fn a_column_profile_is_a_pure_function_of_x() {
        // Chunk independence rests on this: no scratch survives a call, so
        // asking about column A after column B gives the same answer as asking
        // about it first.
        let n = noise();
        let cols = [-9973, 0, 1, 12, -1, 5000, 4999, -20000, 77, 77];
        let first: Vec<ColumnProfile> = cols.iter().map(|&c| column_profile_at(&n, c)).collect();
        let mut again: Vec<ColumnProfile> = cols
            .iter()
            .rev()
            .map(|&c| column_profile_at(&n, c))
            .collect();
        again.reverse();
        for (a, b) in first.iter().zip(again.iter()) {
            assert_eq!(a.wcx, b.wcx);
            assert_eq!(a.surf_a, b.surf_a);
            assert_eq!(a.surf_b, b.surf_b);
            assert_eq!(a.surf_t, b.surf_t);
            assert_eq!(a.eco_cap, b.eco_cap);
            assert_eq!(a.amp_scale, b.amp_scale);
            assert_eq!(a.height_offset, b.height_offset);
            assert_eq!(a.cap_thickness, b.cap_thickness);
            assert_eq!(a.surf_cave_scale, b.surf_cave_scale);
            assert_eq!(a.ug_cave_scale, b.ug_cave_scale);
            assert_eq!(a.ug_t, b.ug_t);
            assert_eq!(a.ug_pocket_bias, b.ug_pocket_bias);
            assert_eq!(a.ug_fade_start, b.ug_fade_start);
            assert_eq!(a.ug_fade_end, b.ug_fade_end);
            assert_eq!(a.surf.items(), b.surf.items());
            assert_eq!(a.surf.cum(), b.surf.cum());
            assert_eq!(a.ug.items(), b.ug.items());
            assert_eq!(a.ug.cum(), b.ug.cum());
        }
    }

    #[test]
    fn height_params_agrees_with_the_column_profile_bit_for_bit() {
        // Both feed `surface_row_at`, whose result is ROUNDED to a cell row. A
        // last-bit disagreement would let one column round to two different
        // ground rows depending on which caller asked, and the ground would tear
        // between a lit column and a generated one.
        let n = noise();
        let mut wcx = -20_000;
        while wcx < 20_000 {
            let p = column_profile_at(&n, wcx);
            let h = height_params_at(&n, wcx);
            assert_eq!(p.amp_scale.to_bits(), h.amp_scale.to_bits(), "at {wcx}");
            assert_eq!(
                p.height_offset.to_bits(),
                h.height_offset.to_bits(),
                "at {wcx}"
            );
            wcx += 7;
        }
    }

    #[test]
    fn a_dither_value_selects_in_proportion_to_the_weights() {
        // What makes a dithered boundary seamless: the SHARE of cells an entry
        // wins tracks its weight continuously.
        let n = noise();
        let mut wcx = 0;
        let m = loop {
            let m = biome_mix_at(&n, wcx);
            if m.len() >= 2 && m.second_w > 0.2 {
                break m;
            }
            wcx += 1;
            assert!(wcx < 100_000, "no blended column found");
        };
        let mut counts = [0usize; POOL_MAX];
        let steps = 100_000;
        for i in 0..steps {
            let v = i as f64 / steps as f64;
            let b = m.pick(v);
            let slot = m.items().iter().position(|&x| x == b).unwrap();
            counts[slot] += 1;
        }
        let mut prev = 0.0;
        for (i, (&c, &count)) in m.cum().iter().zip(counts.iter()).enumerate() {
            let want = c - prev;
            prev = c;
            let got = count as f64 / steps as f64;
            assert!((got - want).abs() < 1e-3, "bucket {i}: {got} vs {want}");
        }
    }

    #[test]
    fn ecotones_are_order_independent_and_never_self_referential() {
        for &a in &Biome::ALL {
            assert_eq!(ecotone_for(a, a), None, "{} borders itself", a.def().id);
            for &b in &Biome::ALL {
                assert_eq!(ecotone_for(a, b), ecotone_for(b, a));
            }
        }
        assert_eq!(
            ecotone_for(Biome::Desert, Biome::Plains),
            Some(block::GRAVEL)
        );
        // Volcanic margins are scorched, whatever they border.
        for &b in &Biome::ALL {
            if b != Biome::Volcanic {
                assert_eq!(
                    ecotone_for(Biome::Volcanic, b),
                    Some(block::ASH),
                    "{} does not scorch against Volcanic",
                    b.def().id
                );
            }
        }
    }

    #[test]
    fn volcanic_is_rare_but_real() {
        // Volcanic competes through a synthetic distance rather than a climate
        // cell precisely so it stays a couple of percent of columns in isolated
        // hotspots. If VOLC_T0 / VOLC_GAIN / VOLC_D0 drift, this is what moves.
        let n = noise();
        let span = 40_000;
        let mut volcanic = 0;
        for wcx in 0..span {
            if biome_mix_at(&n, wcx).top == Biome::Volcanic {
                volcanic += 1;
            }
        }
        let frac = f64::from(volcanic) / f64::from(span);
        assert!(
            (0.001..0.10).contains(&frac),
            "Volcanic covers {:.3}% of the world",
            frac * 100.0
        );
    }

    #[test]
    fn every_biome_wins_somewhere() {
        // A biome whose centroid is crowded out by its neighbours' biases never
        // appears in the world at all — a silent content bug.
        let n = noise();
        let mut seen = [false; BIOME_COUNT];
        let mut wcx = -60_000;
        while wcx < 60_000 {
            seen[biome_index_at(&n, wcx)] = true;
            wcx += 3;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "{} never wins a column", BIOMES[i].id);
        }
    }

    #[test]
    fn the_fade_window_is_below_the_surface_and_ordered() {
        let n = noise();
        let mut wcx = -10_000;
        while wcx < 10_000 {
            let p = column_profile_at(&n, wcx);
            assert!(p.ug_fade_start > 0.0, "fade starts above ground at {wcx}");
            assert!(p.ug_fade_end > p.ug_fade_start);
            assert!(p.cap_thickness > 0.0);
            wcx += 29;
        }
    }

    #[test]
    fn the_atmosphere_is_a_convex_blend_of_its_biomes() {
        // Weighted mean, not a sum: every channel has to stay inside the range
        // of the contributing biomes, or the sky blows out on a boundary.
        let n = noise();
        let mut wcx = -10_000;
        while wcx < 10_000 {
            let a = resolve_atmosphere(&n, wcx);
            let m = biome_mix_at(&n, wcx);
            for ch in 0..3 {
                let lo = m
                    .items()
                    .iter()
                    .map(|b| b.def().atmo.sky_top[ch])
                    .fold(f64::INFINITY, f64::min);
                let hi = m
                    .items()
                    .iter()
                    .map(|b| b.def().atmo.sky_top[ch])
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    a.sky_top[ch] >= lo - 1e-9 && a.sky_top[ch] <= hi + 1e-9,
                    "sky_top[{ch}] = {} outside [{lo}, {hi}] at {wcx}",
                    a.sky_top[ch]
                );
            }
            assert_eq!(a.weather, m.top.def().atmo.weather);
            wcx += 41;
        }
    }

    #[test]
    fn biome_at_index_is_total() {
        assert_eq!(biome_at_index(0), Biome::Plains);
        assert_eq!(biome_at_index(BIOME_COUNT - 1), Biome::Savanna);
        assert_eq!(biome_at_index(9999), Biome::Plains);
        let n = noise();
        let p = biome_at(&n, 0);
        assert_eq!(p.a, biome_at_index(biome_index_at(&n, 0)));
    }
}

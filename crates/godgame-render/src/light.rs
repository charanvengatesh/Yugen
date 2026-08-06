//! Coarse dynamic lighting, composited over the world.
//!
//! Ported from `src/render/light.ts`.
//!
//! # The model
//!
//! One light sample per [`LIGHT_DOWNSCALE`] sim cells — a 20 world-px light cell
//! — over the visible viewport plus a one-light-cell margin, so the small grid
//! upscales smoothly to the screen without a seam creeping in at the edges. The
//! look is three cheap layers stacked in order:
//!
//!   1. **skylight** — flood from the top; open sky is bright, and light decays
//!      as it descends through and behind solids so caves and the deep go dark;
//!   2. **emissive** — lava, fire and anything else declaring `lightEmit` smear
//!      glow into the light grid so their pockets read as lit underground;
//!   3. **composite** — the grid, blurred and upscaled, multiplied over the
//!      scene (a floor keeps caves moody, not pitch black), then additive bloom
//!      over emissive clusters, and finally a vignette and a depth wash.
//!
//! [`LightGrid`] is layers 1 and 2 and knows nothing about Bevy. It is a plain
//! struct with a `solve` that takes a [`CellGrid`] and gives back four grids of
//! floats, which is what makes every propagation rule below testable without a
//! window, a GPU or a frame. [`LightPlugin`] is layer 3 and the wiring.
//!
//! # What the port changed
//!
//! **The surface cache is gone.** The TypeScript kept a per-column cache of
//! `surfaceRowAt`, scrolled it with `copyWithin` as the camera moved, and
//! recomputed only the columns newly exposed at an edge — about forty lines of
//! bookkeeping, existing because `surfaceRowAt` was an uncached fBm evaluation
//! and the skylight pass called it once per column per frame. Here `Heightmap`
//! IS a 4096-slot memo keyed on the absolute column, so the cache would be a
//! cache in front of a cache. The columns are simply asked for, and a stationary
//! camera does no noise evaluation for the same reason it did not before.
//!
//! **The emitter census is taken here.** In the TypeScript the census was a
//! by-product of `paintCells`, which walked every visible cell anyway. The cell
//! pass is [`crate::cellmap`]'s shader now and walks nothing on the CPU, so
//! [`scan_emitters`] pays for the walk explicitly. It matters: the light grid
//! point-samples one cell in sixteen, so without it a one-cell torch is missed
//! fifteen times out of sixteen. The scan keeps the original's dedup rule (one
//! entry per 4-cell column group per row) and its [`EMIT_CENSUS_MAX`] cap, and
//! publishes two parallel `&[i32]`s so [`crate::cells::EmitterCensus`] can be
//! handed to [`LightGrid::add_census_emitters`] verbatim if the CPU blit ever
//! becomes the thing producing one.
//!
//! **The composite is fixed-function blending, not Canvas2D composite modes.**
//! `globalCompositeOperation = "multiply"` over an RGBA whose alpha is the
//! darkness becomes a texture of MULTIPLY FACTORS and a `Dst * Src` blend;
//! `"lighter"` becomes `One + One`. The arithmetic is identical — see
//! [`LightGrid::bake_shadow`] for the algebra — with one honest difference: a
//! GPU blends in linear light and Canvas2D blended sRGB bytes. Multiplying in
//! linear is BRIGHTER in the midtones than multiplying the encoded byte, so
//! caves come out a shade less crushed than the original's. Fixed-function
//! blending has nowhere else to put the operation, and linear is the correct
//! place for it; matching the original byte for byte would need a pass that
//! reads its own destination, which 2D has no way to do.
//!
//! **`f32` throughout.** The TypeScript was doubles because JavaScript has
//! nothing else. Nothing here is precision-critical: the whole output is
//! quantised to 8 bits per channel on a grid a sixteenth of the viewport's
//! resolution.
//!
//! # What the port dropped on the way in
//!
//! - **The profiler bracket.** `PROF.begin(P_LIGHT)` / `PROF.end` around the
//!   solve. There is no profiler in this tree, and a Bevy system is already a
//!   span the frame diagnostics can see.
//! - **`performance.now()` as the flicker clock.** `Time::elapsed_secs` is the
//!   same number from a source the app already owns and a test can fake.
//! - **The frame-stamp array.** The census dedup was a never-cleared `Int32Array`
//!   of frame counters, to avoid clearing a flag array once a frame. The flag
//!   array is ~1300 bytes; `fill(false)` on it is not worth a monotonic counter
//!   and the state it has to carry between frames.
//! - **The second flat `fillRect`.** The underworld glow and the biome ambient
//!   cast were two additive full-view fills. Adding two colours and adding their
//!   sum are the same operation, so they are one quad here.
//!
//! # The one convention flip
//!
//! Everything in this module is in the SIM's convention: +y is DOWN, world px,
//! absolute cells. The flip to Bevy's +y-up happens in [`place_quads`] and
//! [`place_bloom`] and nowhere else, exactly as it does in [`crate::cellmap`]'s
//! `follow_window`.

use bevy::asset::{RenderAssetUsages, uuid_handle};
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSampler;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, Extent3d,
    RenderPipelineDescriptor, SpecializedMeshPipelineError, TextureDimension, TextureFormat,
};
use bevy::shader::{Shader, ShaderRef};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey, Material2dPlugin};
use std::sync::LazyLock;

use godgame_core::config::{CELL_SIZE, LIGHT_DOWNSCALE, SEED, SURFACE_ANCHOR_Y, View};
use godgame_core::sim::grid::CellGrid;
use godgame_core::sim::materials::{
    CellId, EMPTY, MAT_B, MAT_COLLIDE, MAT_COUNT, MAT_EMISSIVE, MAT_G, MAT_LIGHT, MAT_R,
};
use godgame_core::sim::noise::Noise;
use godgame_core::sim::worldgen::heightmap::Heightmap;
use godgame_core::sim::worldgen::world_noise;

use crate::daynight::WorldClock;
use crate::lowres::{LowResTarget, WORLD_LAYERS};
use crate::world::{SimWorld, WorldFocus};

// --- Emitter tables ----------------------------------------------------------

/// The authored `lightEmit` level a material must declare to be at full scale.
///
/// The content format authors light as an integer 0..15 — the scale the block
/// schema validates against — so this is the divisor that turns a declared level
/// into the 0..1 weight the solver works in.
const EMIT_MAX_LEVEL: f32 = 15.0;

/// What a material that declares `emissive` but no `lightEmit` is worth.
///
/// A "this material is self-lit" fallback for blocks that predate `lightEmit`,
/// so nothing that used to glow stops glowing. Deliberately below full: a block
/// that never declared a light level never had one chosen for it.
const EMIT_FALLBACK_GAIN: f32 = 0.8;

/// How far an emitter's cast is pushed away from grey.
///
/// Light COLOUR is the block's own colour, re-saturated. A lava cell is
/// (235,110,35) and should cast orange, not white; a crystal is (152,112,232)
/// and should cast violet. Normalising by the max channel and then pushing past
/// the block's own saturation gives the emitted hue without also carrying the
/// block's brightness, which is what the level is for.
const EMIT_SATURATION: f32 = 1.35;

/// How many baked glow sprites the bloom pass chooses between.
const GLOW_COUNT: usize = 3;

/// Which prebaked glow sprite an emitter blooms with.
///
/// Three, because three is what the content set actually needs. The alternative
/// — tinting one sprite per draw — costs a filter or a scratch composite per
/// call, which is more than baking the three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlowHue {
    /// Lava, fire, embers.
    Warm,
    /// Crystal and the other cold-violet emitters.
    Violet,
    /// Fungal light.
    Green,
}

/// Every glow hue, in the order [`GlowHue::index`] numbers them.
pub const GLOW_HUES: [GlowHue; GLOW_COUNT] = [GlowHue::Warm, GlowHue::Violet, GlowHue::Green];

impl GlowHue {
    /// Index into the baked sprite set.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            GlowHue::Warm => 0,
            GlowHue::Violet => 1,
            GlowHue::Green => 2,
        }
    }

    /// The hot core and the halo the sprite fades through, 0..255.
    const fn stops(self) -> ([f32; 3], [f32; 3]) {
        match self {
            GlowHue::Warm => ([255.0, 240.0, 200.0], [255.0, 150.0, 60.0]),
            GlowHue::Violet => ([236.0, 224.0, 255.0], [150.0, 90.0, 240.0]),
            GlowHue::Green => ([226.0, 255.0, 226.0], [90.0, 220.0, 120.0]),
        }
    }
}

/// What one material contributes as a light source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    /// Emission strength, 0..1. Zero means "not a light source".
    pub level: f32,
    /// The hue the cast carries, each channel 0..1, normalised of brightness.
    pub rgb: [f32; 3],
    /// Which baked sprite this emitter blooms with.
    pub hue: GlowHue,
}

impl Emitter {
    /// A material that emits nothing.
    pub const NONE: Emitter = Emitter {
        level: 0.0,
        rgb: [0.0; 3],
        hue: GlowHue::Warm,
    };

    /// Whether this material is a light source at all.
    #[inline]
    pub fn emits(self) -> bool {
        self.level > 0.0
    }
}

/// Emitter properties per material, resolved once from the block data.
///
/// The light pass keys off `MAT_LIGHT` — the authored 0..15 `lightEmit` level —
/// rather than off `MAT_EMISSIVE` or a hard-coded list of lava and fire. That is
/// a deliberate contract with content: ANY block that declares `lightEmit` is a
/// light source here, at a radius and colour derived from its own declaration,
/// with no render-side edit. Placeable torches will light the world the day they
/// are added to `content/blocks/`, and so will a glowing ore or a lit furnace.
static EMITTERS: LazyLock<[Emitter; MAT_COUNT]> = LazyLock::new(|| {
    let mut table = [Emitter::NONE; MAT_COUNT];
    // Slot 0 is air and is never a light source; starting at 1 also keeps the
    // `MAT_EMISSIVE` fallback from reading air's zero as a decision.
    for (id, slot) in table.iter_mut().enumerate().skip(1) {
        let declared = f32::from(MAT_LIGHT[id]) / EMIT_MAX_LEVEL;
        let level = if declared > 0.0 {
            declared
        } else {
            MAT_EMISSIVE[id] * EMIT_FALLBACK_GAIN
        };
        if level <= 0.0 {
            continue;
        }

        let r = f32::from(MAT_R[id]);
        let g = f32::from(MAT_G[id]);
        let b = f32::from(MAT_B[id]);
        // A black block with a light level would divide by zero; flooring the
        // divisor at 1 makes its cast white rather than a NaN that would poison
        // the whole colour grid.
        let mx = r.max(g).max(b).max(1.0);
        let mean = (r + g + b) / 3.0 / mx;
        let sat = |c: f32| (mean + (c / mx - mean) * EMIT_SATURATION).min(1.0);
        let rgb = [sat(r), sat(g), sat(b)];

        // Hue family by which channel wins. The 0.95 slack means a cast that is
        // merely AS blue as it is red still counts as violet, which is what a
        // crystal wants, and the same for green. Blue is tested first, exactly
        // as the original's nested ternary did — a cast that is both would
        // otherwise depend on the order rather than on the content.
        let hue = if rgb[2] > rgb[0] * 0.95 {
            GlowHue::Violet
        } else if rgb[1] > rgb[0] * 0.95 {
            GlowHue::Green
        } else {
            GlowHue::Warm
        };

        *slot = Emitter { level, rgb, hue };
    }
    table
});

/// What a material emits. Out of range reads as air, which emits nothing.
#[inline]
pub fn emitter(id: CellId) -> Emitter {
    EMITTERS.get(id as usize).copied().unwrap_or(Emitter::NONE)
}

// --- Shadow hue and depth ----------------------------------------------------

/// The colour an unlit surface falls toward near the surface, 0..255.
///
/// Not black. The sky is a huge blue area light, so anything in shadow outdoors
/// is lit blue by it, and painting surface shadow as dark blue is why this is
/// the value that looks right at the top of the world.
const SHADOW_SKY: [f32; 3] = [24.0, 30.0, 48.0];

/// The colour an unlit surface falls toward in the deep, 0..255.
///
/// Five hundred cells down there is no sky to bounce and the correct answer is a
/// near-neutral that eats colour instead of adding one. Interpolating between
/// this and [`SHADOW_SKY`] by depth is the cheapest possible way to make
/// descending feel like going somewhere.
const SHADOW_DEEP: [f32; 3] = [14.0, 11.0, 12.0];

/// How much each channel of the shadow cools at full night, 0..255.
///
/// Blue loses most, so the shadow deepens toward the same neutral the deep is
/// already at rather than merely dimming.
const SHADOW_NIGHT_COOL: [f32; 3] = [6.0, 8.0, 10.0];

/// Cells below [`SURFACE_ANCHOR_Y`] over which depth walks from 0 to 1.
///
/// Everything depth drives — the shadow hue, the ambient floor, the vignette,
/// the underworld glow — is measured against this one ramp, so they stay in step
/// with each other as the player descends.
const DEPTH_RANGE_CELLS: f32 = 300.0;

/// Ambient floor so unlit caves stay readable rather than pure black.
const AMBIENT_FLOOR: f32 = 0.12;

/// How much of the ambient floor the deep takes away.
///
/// Caves keep a floor so they stay readable, but the floor itself falls with
/// depth: the deep is meant to be genuinely dark and lit by what you carry.
const AMBIENT_DEPTH_FALL: f32 = 0.05;

// --- Skylight ----------------------------------------------------------------

/// Open sky at midnight. Moonlight, not nothing.
const SKY_NIGHT: f32 = 0.22;

/// What full daylight adds on top of [`SKY_NIGHT`].
const SKY_DAY_GAIN: f32 = 0.78;

/// What a light cell of open air costs the flood.
///
/// LIGHT IS LOST BY PASSING THROUGH MATTER, NOT BY TRAVELLING. This was 0.94 —
/// a 6% loss per light cell of empty air. Over the 27 light rows a 1440p
/// viewport spans that compounds to 0.19, so the bottom of the screen sat at a
/// fifth of the brightness of the top with nothing in between them but sky. At
/// noon. It also made the flood depend on where the CAMERA happened to be rather
/// than on the world, so the same patch of ground brightened and dimmed as you
/// walked toward it, and it put visible vertical banding in open sky wherever
/// neighbouring columns had different surface heights.
///
/// Open air now costs almost nothing — a touch of haze, so a very tall shaft
/// still reads as deeper at the bottom.
const OPEN_DECAY: f32 = 0.99;

/// What a light cell of opaque rock costs the flood.
///
/// A cave is dark because there are forty cells of stone over it, which is the
/// reason it should be dark, and that is what makes a torch matter down there
/// while daylight stays flat and bright up top.
const SOLID_DECAY: f32 = 0.55;

/// Per-CELL decay used to seed a column whose top starts below the surface.
///
/// The streaming window's top may be far below the ground line. Seeding each
/// column's carry from how deep its top sample sits below that column's own
/// surface height keeps brightness a continuous function of absolute world
/// coordinates, so it stays seam-free as the window scrolls. Per sim cell rather
/// than per light cell because a surface height is a cell row.
const SEED_DECAY: f32 = 0.985;

// --- Emissive splats ---------------------------------------------------------

/// Flicker midpoint, so the swing lands on 0.76..1.0.
const FLICKER_BASE: f32 = 0.88;
/// Flicker amplitude.
const FLICKER_SWING: f32 = 0.12;
/// Flicker rate, radians per second.
///
/// The phase is hashed from the cell's absolute coords so neighbouring pockets
/// breathe out of step instead of pulsing as one slab.
const FLICKER_RATE: f32 = 5.5;

/// What the four neighbours of a scalar splat get, relative to its centre.
const SPLAT_EDGE: f32 = 0.42;

/// The declared level above which an emitter also throws a far ring.
///
/// A level-13 lava pool throws light two light cells (40 sim cells) and a
/// level-6 mushroom cap barely leaves its own. Before the reach scaled with the
/// declared level, every emitter had the same one-cell reach whatever it
/// declared, which is why a magma chamber and a glowing mushroom lit the same
/// volume.
const FAR_SPLAT_LEVEL: f32 = 0.55;

/// What the far ring gets, relative to the splat's centre.
const FAR_SPLAT_GAIN: f32 = 0.16;

/// How much of a splat's strength goes into the COLOUR grids.
///
/// Below the scalar gain on purpose: the cast should tint the surroundings, not
/// repaint them.
const COLOUR_GAIN: f32 = 0.62;

/// What the four neighbours of a colour splat get, relative to its centre.
const COLOUR_EDGE: f32 = 0.45;

/// Cell temperature below which residual heat casts no light at all.
///
/// Only the hot tail contributes, and only weakly, so this reads as warmth
/// bleeding out of a pocket rather than as a second light source.
const HEAT_THRESHOLD: f32 = 70.0;

/// Temperature span from [`HEAT_THRESHOLD`] to a full-strength heat glow.
///
/// 70 + 185 = 255, the top of the temperature plane, so the hottest possible
/// rock lands exactly at [`HEAT_GAIN`] and nothing has to clip.
const HEAT_RANGE: f32 = 185.0;

/// What the hottest possible non-emitting cell is worth as a light source.
const HEAT_GAIN: f32 = 0.3;

/// What the four neighbours of a heat splat get, relative to its centre.
const HEAT_EDGE: f32 = 0.5;

/// The hue residual heat casts, 0..1 per channel.
///
/// Warm for the same reason lava is: heat in rock glows red before it glows at
/// all.
const HEAT_RGB: [f32; 3] = [0.5, 0.16, 0.04];

/// Cap on the hot list the ambience layer seeds its embers from.
const HOT_MAX: usize = 64;

/// The eight offsets a strong emitter's far ring lands on.
const FAR_RING: [(i32, i32); 8] = [
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

/// The four the census pass uses.
///
/// Faithful to the original, which listed eight offsets in `addEmissive` and
/// four in `addCensusEmitters`. Almost certainly an oversight there rather than
/// a decision — but a census emitter is by definition a SMALL source, a torch
/// and not a magma chamber, and the narrower ring is the better answer for one.
/// Kept, and written down, rather than silently unified.
const FAR_RING_AXIAL: [(i32, i32); 4] = [(-2, 0), (2, 0), (0, -2), (0, 2)];

// --- Census ------------------------------------------------------------------

/// Cap on the emitter census.
///
/// Beyond this the extra emitters are simply not reported — the splat pass has a
/// bounded cost anyway, and the dedup below means the cap is only reached by a
/// view that is genuinely wall to wall light sources.
pub const EMIT_CENSUS_MAX: usize = 512;

/// Cells per census dedup group along a row.
///
/// One entry per group per row, so a wide lava surface cannot flood the list and
/// starve a torch on the far side of the view. Four is the light grid's own
/// downscale: two emitters closer together than this land in the same light cell
/// and would be deduplicated by the splat pass regardless.
const CENSUS_GROUP: i32 = LIGHT_DOWNSCALE;

// --- Bloom -------------------------------------------------------------------

/// Cells between bloom probes — one probe per ~40 world px.
const BLOOM_STRIDE_CELLS: i32 = LIGHT_DOWNSCALE * 2;

/// Hard cap on bloom sprites per frame, so a lava lake never floods the frame.
pub const BLOOM_BUDGET: usize = 120;

/// Glow sprite radius, in world px — one logical pixel of the low-res buffer.
const GLOW_RADIUS_PX: i32 = 64;

/// Alpha of the glow sprite's hot core.
const GLOW_CORE_ALPHA: f32 = 0.55;
/// Alpha where the core has finished handing over to the halo.
const GLOW_HALO_ALPHA: f32 = 0.28;
/// Fraction of the radius the core occupies.
const GLOW_CORE_STOP: f32 = 0.4;

/// Bloom flicker midpoint.
///
/// Bloom breathes on its own slower phase, decorrelated from the light grid's,
/// so the halo swells and settles instead of strobing with it.
const BLOOM_BASE: f32 = 0.72;
/// Bloom flicker amplitude.
const BLOOM_SWING: f32 = 0.28;
/// Bloom flicker rate, radians per second.
const BLOOM_RATE: f32 = 3.1;
/// How much of the cell's hash phase the bloom wave uses.
const BLOOM_PHASE_SCALE: f32 = 0.7;

// --- Vignette and washes -----------------------------------------------------

/// View px per vignette sample.
///
/// The vignette is a smooth radial ramp with no detail in it, so it is baked at
/// a sixteenth of the view's resolution and upscaled by the same linear sampler
/// the light grid uses. At 500x250 logical that is a 33x17 texture per frame.
const VIGNETTE_CELL: i32 = 16;

/// Vignette inner radius as a fraction of the view's short axis, at the surface.
const VIGNETTE_INNER: f32 = 0.35;
/// How much depth shrinks the bright core.
const VIGNETTE_INNER_DEPTH: f32 = 0.1;
/// How much night shrinks the bright core.
const VIGNETTE_INNER_NIGHT: f32 = 0.06;
/// Vignette outer radius as a fraction of the view's long axis.
const VIGNETTE_OUTER: f32 = 0.72;
/// Edge darkness at the surface in daylight.
const VIGNETTE_EDGE: f32 = 0.55;
/// How much depth lightens the edge — the deep is dark enough already.
const VIGNETTE_EDGE_DEPTH: f32 = 0.25;
/// How much night closes the edges in.
const VIGNETTE_EDGE_NIGHT: f32 = 0.1;
/// The colour the vignette's edge falls toward at the surface, 0..255.
const VIGNETTE_RGB: [f32; 3] = [40.0, 44.0, 64.0];
/// How much depth takes off the vignette's red and green, 0..255.
///
/// Blue is untouched, so the frame edge goes bluer as it goes deeper.
const VIGNETTE_RGB_DEPTH: [f32; 3] = [20.0, 20.0, 0.0];

/// Depth at which the underworld glow starts to ramp in, as a 0..1 fraction of
/// the depth range.
///
/// Below the magma line the rock itself is hot. A very faint additive floor
/// wash, over the last third of the depth range only, so the deep reads as lit
/// from below rather than merely dark.
///
/// **Not to be renamed back to `UNDERWORLD_DEPTH`.** That is the name of
/// `godgame_core::config::worldgen::UNDERWORLD_DEPTH`, which is an `i32` count
/// of CELLS below the surface, not a normalised fraction. This module already
/// does `use godgame_core::config::{…}`; the day that becomes a glob import, a
/// local of the same name would silently win and substitute 0.62 for 470, with
/// no error anywhere. `cargo xtask tuning` gates that collision, and this
/// constant is the one that made the rule earn its keep.
///
/// The TypeScript had the 0.62 as a bare literal, so the clash is a port
/// regression rather than something inherited.
const UNDERWORLD_GLOW_DEPTH: f32 = 0.62;
/// Strength of the underworld glow at full depth.
const UNDERWORLD_ALPHA: f32 = 0.20;
/// The underworld glow's colour, 0..255.
const UNDERWORLD_RGB: [f32; 3] = [96.0, 30.0, 14.0];

/// Strength of a biome's ambient cast.
///
/// A faint additive wash so a biome's mood (warm volcanic, cold tundra) colours
/// the whole scene. Plains sends `[0, 0, 0]`, which is a no-op.
const BIOME_AMBIENT_ALPHA: f32 = 0.6;

// --- The solver --------------------------------------------------------------

/// The frame-varying inputs to one solve.
///
/// Bundled rather than passed loose because every one of them is read by two or
/// three of the passes and the alternative is the same four arguments threaded
/// through four signatures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightFrame {
    /// Light-cell x of the grid's left edge, absolute.
    pub ox: i32,
    /// Light-cell y of the grid's top edge, absolute.
    pub oy: i32,
    /// The world clock's daylight weight, 0 night to 1 noon.
    ///
    /// It scales the open-sky flood ONLY: caves are already dark and should not
    /// get darker, but the surface must actually go dim at night for the cycle
    /// to read.
    pub day: f32,
    /// Seconds, for the flicker phase. Frame-constant.
    pub t: f32,
}

/// The light solver: four grids of floats over the visible world.
///
/// Bevy-free by construction. Everything the frame touches is allocated once in
/// [`LightGrid::new`]; `solve` refills the buffers and allocates nothing.
pub struct LightGrid {
    /// Grid width in light cells (view plus a one-cell margin each side).
    lw: i32,
    /// Grid height in light cells.
    lh: i32,

    /// Scalar light per light cell, 0..1.
    light: Vec<f32>,
    /// Ping-pong scratch for the separable blur.
    scratch: Vec<f32>,

    /// COLOURED light, accumulated separately from the scalar grid.
    ///
    /// The scalar grid drives the darkness MULTIPLY (how much of the scene
    /// survives at all); these three drive an ADDITIVE pass on top of it.
    /// Keeping them apart is what makes a torch-lit cave look different from a
    /// daylit surface rather than merely brighter: skylight contributes only to
    /// the scalar grid, so open ground is revealed in its own colours, while an
    /// emitter contributes to both, so rock near lava is revealed AND washed
    /// orange. One grid could not express that difference.
    lr: Vec<f32>,
    lg: Vec<f32>,
    lb: Vec<f32>,

    /// Whether anything wrote into the colour grids this frame.
    ///
    /// Residual HEAT casts colour without being an emitter, so the hot list
    /// alone would miss a wall that a fire has been licking; this flag is the
    /// honest answer to "is the coloured pass worth blurring and uploading".
    colour_dirty: bool,

    /// Per-light-cell dedup mask for the census splat, cleared per frame.
    seen: Vec<bool>,

    /// Emissive cells the last solve found, in absolute cells, capped at
    /// [`HOT_MAX`].
    ///
    /// The ambience layer seeds its embers from this instead of scanning the
    /// grid again: this pass already visits every light cell, so the list is a
    /// by-product that costs one push per hit. One frame stale, which nobody can
    /// see.
    hot: Vec<[i32; 2]>,

    /// The surface heightmap, for seeding a column that starts underground.
    heights: Heightmap,
    /// The noise the heightmap is evaluated against.
    noise: Noise,
}

impl LightGrid {
    /// A grid sized for `view`, reading the surface line of world `seed`.
    pub fn new(view: View, seed: u32) -> LightGrid {
        let (lw, lh) = grid_size(view);
        let n = (lw * lh) as usize;
        LightGrid {
            lw,
            lh,
            light: vec![0.0; n],
            scratch: vec![0.0; n],
            lr: vec![0.0; n],
            lg: vec![0.0; n],
            lb: vec![0.0; n],
            colour_dirty: false,
            seen: vec![false; n],
            hot: Vec::with_capacity(HOT_MAX),
            heights: Heightmap::new(),
            noise: world_noise(seed),
        }
    }

    /// Grid width in light cells.
    #[inline]
    pub const fn cols(&self) -> i32 {
        self.lw
    }

    /// Grid height in light cells.
    #[inline]
    pub const fn rows(&self) -> i32 {
        self.lh
    }

    /// Scalar light at a light cell, or 0 outside the grid.
    #[inline]
    pub fn light_at(&self, lx: i32, ly: i32) -> f32 {
        self.index(lx, ly).map_or(0.0, |i| self.light[i])
    }

    /// Coloured light at a light cell, or black outside the grid.
    #[inline]
    pub fn colour_at(&self, lx: i32, ly: i32) -> [f32; 3] {
        self.index(lx, ly)
            .map_or([0.0; 3], |i| [self.lr[i], self.lg[i], self.lb[i]])
    }

    /// Whether anything emitted into the colour grids on the last solve.
    #[inline]
    pub const fn colour_dirty(&self) -> bool {
        self.colour_dirty
    }

    /// Absolute cells of the emitters the last solve found.
    #[inline]
    pub fn hot(&self) -> &[[i32; 2]] {
        &self.hot
    }

    #[inline]
    fn index(&self, lx: i32, ly: i32) -> Option<usize> {
        if lx < 0 || ly < 0 || lx >= self.lw || ly >= self.lh {
            None
        } else {
            Some((ly * self.lw + lx) as usize)
        }
    }

    /// One frame's light: skylight, then emitters, then the census, then blur.
    ///
    /// The order is not arbitrary. Skylight WRITES the scalar grid — every cell,
    /// unconditionally, which is what makes it the only pass that needs no
    /// clear. The two emissive passes ADD to it. The blur is last because it is
    /// what turns a cross-shaped splat into a glow.
    ///
    /// `census_x` / `census_y` are the absolute cells of every emitter in view,
    /// shaped to take [`crate::cells::EmitterCensus`]'s two accessors directly.
    /// Pass empty slices to solve without one — the result is correct for bulk
    /// emitters and blind to single-cell ones, which is the whole reason the
    /// census exists.
    pub fn solve(
        &mut self,
        grid: &CellGrid,
        frame: LightFrame,
        census_x: &[i32],
        census_y: &[i32],
    ) {
        self.compute_skylight(grid, frame);
        self.add_emissive(grid, frame);
        self.add_census_emitters(grid, census_x, census_y, frame);
        self.blur();
    }

    /// Top-down skylight flood.
    ///
    /// Each column starts at full brightness above the world (open sky) and
    /// carries a running light value downward: open cells keep most of it, solid
    /// cells swallow it fast, so light attenuates behind and beneath rock and
    /// caves fall dark. One pass, column-major.
    pub fn compute_skylight(&mut self, grid: &CellGrid, frame: LightFrame) {
        let (lw, lh) = (self.lw, self.lh);
        let step = LIGHT_DOWNSCALE;
        let half = step / 2;
        let sky = SKY_NIGHT + SKY_DAY_GAIN * frame.day;
        let top_cy = frame.oy * step + half;

        // Resolve the window origin once and index the material plane directly:
        // `get_world` is a bounds test plus a translation per sample, and this
        // loop samples every light cell in the viewport.
        let gx0 = grid.origin_cell_x();
        let gy0 = grid.origin_cell_y();
        let cols = grid.cols();
        let rows = grid.rows();

        for lx in 0..lw {
            let cx = (frame.ox + lx) * step + half; // sample cell centre
            let below_surface = top_cy - self.heights.surface_row_at(&self.noise, cx, None);
            let mut carry = sky
                * if below_surface <= 0 {
                    1.0
                } else {
                    SEED_DECAY.powi(below_surface).min(1.0)
                };

            let gx = cx - gx0;
            let in_col = gx >= 0 && gx < cols;
            for ly in 0..lh {
                let gy = (frame.oy + ly) * step + half - gy0;
                // Unloaded cells read as air, exactly as `get_world` reports them.
                let solid = in_col
                    && gy >= 0
                    && gy < rows
                    && MAT_COLLIDE[grid.material[(gy * cols + gx) as usize] as usize] == 1;

                // STORE, THEN DECAY. A cell is lit by the light that REACHES it;
                // the occlusion it causes applies to what is behind it, not to
                // itself. Decaying first meant the topmost solid sample in every
                // column — the sunlit ground you are standing on — was already
                // darkened by 45% before it was ever written, so at noon the
                // surface came out the same value as rock four cells under it
                // and the whole world read as overcast dusk. The lit face of the
                // terrain is the thing the player looks at most; it has to be
                // the brightest solid in its column, and this is the ordering
                // that makes it so.
                self.light[(ly * lw + lx) as usize] = carry;
                carry *= if solid { SOLID_DECAY } else { OPEN_DECAY };
            }
        }
    }

    /// Smear emissive cells into the light grid so their pockets read as lit.
    ///
    /// One additive splat per emissive light cell with a small cross spread; the
    /// blur afterwards rounds it off. Two things make the glow LIVE rather than
    /// sit flat:
    ///
    ///   - a per-cell flicker on a phase hashed from the cell's absolute coords,
    ///     so neighbouring pockets breathe out of step instead of pulsing as one
    ///     slab;
    ///   - the heat field: a cell that is merely HOT (rock next to lava, a wall
    ///     a fire has been licking) contributes a weak glow of its own, so heat
    ///     visibly spreads into the surroundings and fades as it dissipates.
    ///
    /// The heat read is free: this loop already resolves the cell's flat index,
    /// so the temperature is one extra load per light cell.
    ///
    /// This is also the pass that CLEARS the colour grids and the hot list, so
    /// it must run before the census pass and not after.
    pub fn add_emissive(&mut self, grid: &CellGrid, frame: LightFrame) {
        let (lw, lh) = (self.lw, self.lh);
        let step = LIGHT_DOWNSCALE;
        let half = step / 2;

        self.lr.fill(0.0);
        self.lg.fill(0.0);
        self.lb.fill(0.0);
        self.colour_dirty = false;
        self.hot.clear();

        let gx0 = grid.origin_cell_x();
        let gy0 = grid.origin_cell_y();
        let cols = grid.cols();
        let rows = grid.rows();

        for ly in 0..lh {
            let cy = (frame.oy + ly) * step + half;
            let gy = cy - gy0;
            if gy < 0 || gy >= rows {
                continue;
            }
            let row = gy * cols;

            for lx in 0..lw {
                let cx = (frame.ox + lx) * step + half;
                let gx = cx - gx0;
                if gx < 0 || gx >= cols {
                    continue;
                }

                let gi = (row + gx) as usize;
                let e = emitter(grid.material[gi]);
                if e.emits() {
                    self.splat_emitter(e, (lx, ly), (cx, cy), frame.t, &FAR_RING);
                    continue;
                }

                let heat = f32::from(grid.temp[gi]);
                if heat > HEAT_THRESHOLD {
                    let v = ((heat - HEAT_THRESHOLD) / HEAT_RANGE) * HEAT_GAIN;
                    splat(&mut self.light, lw, lh, lx, ly, v, v * HEAT_EDGE);
                    self.splat_rgb(lx, ly, [v * HEAT_RGB[0], v * HEAT_RGB[1], v * HEAT_RGB[2]]);
                }
            }
        }
    }

    /// Splat the emitters the census found, deduplicated per light cell.
    ///
    /// The downscaled sampling loop above catches BULK emitters — a lava lake
    /// fills every block it is sampled in — but by construction cannot see a
    /// one-cell torch, because it only looks at one cell in sixteen. This pass
    /// covers the gap.
    ///
    /// Deduplicated per LIGHT CELL: several census entries commonly land in the
    /// same one (a 6-cell campfire, a torch beside a lantern), and splatting
    /// each of them would stack the same glow three or four times and blow out
    /// to white.
    ///
    /// The census reports WHERE, not WHAT, so the material is read back out of
    /// `grid`. A cell that stopped emitting between the scan and the splat
    /// therefore contributes nothing rather than a ghost.
    pub fn add_census_emitters(
        &mut self,
        grid: &CellGrid,
        census_x: &[i32],
        census_y: &[i32],
        frame: LightFrame,
    ) {
        let n = census_x.len().min(census_y.len());
        if n == 0 {
            return;
        }
        let step = LIGHT_DOWNSCALE;
        self.seen.fill(false);

        for i in 0..n {
            let (cx, cy) = (census_x[i], census_y[i]);
            let lx = cx.div_euclid(step) - frame.ox;
            let ly = cy.div_euclid(step) - frame.oy;
            let Some(li) = self.index(lx, ly) else {
                continue;
            };
            if self.seen[li] {
                continue;
            }
            self.seen[li] = true;

            let e = emitter(cell_at_world(grid, cx, cy));
            if !e.emits() {
                continue;
            }
            self.splat_emitter(e, (lx, ly), (cx, cy), frame.t, &FAR_RING_AXIAL);
        }
    }

    /// The scalar and coloured splat one emitter makes, plus its far ring.
    ///
    /// `ring` is what the two callers differ by and nothing else — see
    /// [`FAR_RING_AXIAL`] for why they differ at all.
    fn splat_emitter(
        &mut self,
        e: Emitter,
        (lx, ly): (i32, i32),
        (cx, cy): (i32, i32),
        t: f32,
        ring: &[(i32, i32)],
    ) {
        let (lw, lh) = (self.lw, self.lh);
        let v = e.level * flicker(t, cx, cy);
        splat(&mut self.light, lw, lh, lx, ly, v, v * SPLAT_EDGE);
        if e.level > FAR_SPLAT_LEVEL {
            let far = v * FAR_SPLAT_GAIN;
            for (dx, dy) in ring {
                add(&mut self.light, lw, lh, lx + dx, ly + dy, far);
            }
        }
        let cv = v * COLOUR_GAIN;
        self.splat_rgb(lx, ly, [e.rgb[0] * cv, e.rgb[1] * cv, e.rgb[2] * cv]);
        if self.hot.len() < HOT_MAX {
            self.hot.push([cx, cy]);
        }
    }

    /// Additive centre plus 4-neighbour spread into the three colour grids.
    fn splat_rgb(&mut self, lx: i32, ly: i32, rgb: [f32; 3]) {
        let (lw, lh) = (self.lw, self.lh);
        self.colour_dirty = true;
        for (plane, c) in [&mut self.lr, &mut self.lg, &mut self.lb]
            .into_iter()
            .zip(rgb)
        {
            splat(plane, lw, lh, lx, ly, c, c * COLOUR_EDGE);
        }
    }

    /// Two separable passes over the scalar grid and the three colour grids.
    ///
    /// The colour grids are only blurred when something actually emitted this
    /// frame. Standing on the surface in daylight — the common case — that is a
    /// single `fill`-cleared scan away from free, and the three extra blurs
    /// never run at all.
    pub fn blur(&mut self) {
        let (lw, lh) = (self.lw, self.lh);
        blur_one(&mut self.light, &mut self.scratch, lw, lh);
        if self.colour_dirty {
            blur_one(&mut self.lr, &mut self.scratch, lw, lh);
            blur_one(&mut self.lg, &mut self.scratch, lw, lh);
            blur_one(&mut self.lb, &mut self.scratch, lw, lh);
        }
    }

    /// Bake the darkness pass into RGBA multiply factors.
    ///
    /// # The algebra
    ///
    /// Canvas2D drew an RGBA whose alpha was the darkness with
    /// `globalCompositeOperation = "multiply"`, which is a source-over composite
    /// of a multiply blend:
    ///
    /// ```text
    /// out = (1 - a) * dst + a * (shadow * dst)
    ///     = dst * (1 - a + a * shadow)
    /// ```
    ///
    /// The bracket depends only on this module's own numbers, so it is evaluated
    /// here, per light cell, and written into RGB. The GPU then needs nothing
    /// but `dst * src` — see [`BLEND_MULTIPLY`] — and the texture is a plain
    /// non-sRGB `Rgba8Unorm` because these bytes are MULTIPLIERS and must not be
    /// transfer-function-decoded on the way in.
    ///
    /// Alpha is held at 255 so the destination's own alpha survives: the low-res
    /// buffer is sampled by [`crate::lowres`]'s blit sprite, and a pass that
    /// zeroed its alpha would erase the frame.
    ///
    /// `out` must be `cols() * rows() * 4` bytes.
    pub fn bake_shadow(&self, depth: f32, day: f32, out: &mut [u8]) {
        let shadow = shadow_tint(depth, day);
        let floor = ambient_floor(depth);
        for (px, &l) in out.chunks_exact_mut(4).zip(self.light.iter()) {
            let a = 1.0 - l.max(floor).min(1.0);
            for (dst, s) in px.iter_mut().zip(shadow) {
                *dst = unit_byte(1.0 - a + a * s);
            }
            px[3] = u8::MAX;
        }
    }

    /// Bake the coloured pass into RGBA additive increments.
    ///
    /// `"lighter"` at full alpha is `dst + src`, so the increment IS the texel
    /// and there is no algebra to do — only the clamp the original did on its
    /// way into an 8-bit `ImageData`.
    ///
    /// `out` must be `cols() * rows() * 4` bytes.
    pub fn bake_colour(&self, out: &mut [u8]) {
        for (i, px) in out.chunks_exact_mut(4).enumerate() {
            px[0] = unit_byte(self.lr[i]);
            px[1] = unit_byte(self.lg[i]);
            px[2] = unit_byte(self.lb[i]);
            // Additive blending leaves the destination alpha alone (see
            // `BLEND_ADD`), so this byte is never read. Held at 255 anyway so a
            // debug view of the texture is not an invisible rectangle.
            px[3] = u8::MAX;
        }
    }
}

/// Light-grid size for a view: the visible rect, rounded up, plus a one-cell
/// margin each side so the smooth upscale has something to interpolate toward
/// instead of clamping at the screen edge.
pub fn grid_size(view: View) -> (i32, i32) {
    let stride = light_stride_px();
    let ceil = |px: i32| (px.max(0) + stride - 1) / stride + 2;
    (ceil(view.w), ceil(view.h))
}

/// World px covered by one light cell on each axis.
pub const fn light_stride_px() -> i32 {
    CELL_SIZE * LIGHT_DOWNSCALE
}

/// Additive centre plus 4-neighbour spread, clamped to 1.
fn splat(buf: &mut [f32], lw: i32, lh: i32, lx: i32, ly: i32, core: f32, edge: f32) {
    add(buf, lw, lh, lx, ly, core);
    add(buf, lw, lh, lx - 1, ly, edge);
    add(buf, lw, lh, lx + 1, ly, edge);
    add(buf, lw, lh, lx, ly - 1, edge);
    add(buf, lw, lh, lx, ly + 1, edge);
}

/// Clamped additive write into a light cell. Out of range is a no-op.
fn add(buf: &mut [f32], lw: i32, lh: i32, x: i32, y: i32, v: f32) {
    if x < 0 || y < 0 || x >= lw || y >= lh {
        return;
    }
    let i = (y * lw + x) as usize;
    buf[i] = (buf[i] + v).min(1.0);
}

/// Horizontal then vertical `[1, 2, 1] / 4`, edges clamped to themselves.
///
/// `scratch` is the caller's ping-pong buffer; it comes in dirty and goes out
/// dirty, which is the whole point of it being owned by the grid rather than
/// allocated per pass.
fn blur_one(a: &mut [f32], scratch: &mut [f32], lw: i32, lh: i32) {
    let (w, h) = (lw as usize, lh as usize);
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let l = a[row + x.saturating_sub(1)];
            let r = a[row + (x + 1).min(w - 1)];
            scratch[row + x] = (l + a[row + x] * 2.0 + r) * 0.25;
        }
    }
    for y in 0..h {
        let row = y * w;
        let up = y.saturating_sub(1) * w;
        let down = (y + 1).min(h - 1) * w;
        for x in 0..w {
            a[row + x] = (scratch[up + x] + scratch[row + x] * 2.0 + scratch[down + x]) * 0.25;
        }
    }
}

/// The per-cell emissive flicker, 0.76..1.0.
#[inline]
fn flicker(t: f32, cx: i32, cy: i32) -> f32 {
    FLICKER_BASE + FLICKER_SWING * (t * FLICKER_RATE + hash_phase(cx, cy)).sin()
}

/// Cell coords to a stable phase in `[0, TAU)`.
///
/// An integer hash: no trig, no table — the point is only that two adjacent
/// pockets get decorrelated phases.
///
/// The first mix reproduces the original exactly. JavaScript's `*` on these
/// magnitudes is an exact `f64` product and `>>> 0` is a wrap to `u32`, which is
/// what an `i64` product cast to `u32` does. The SECOND mix does not, and cannot
/// — `(h ^ (h >>> 13)) * 1274126177` overflows `f64`'s 53-bit mantissa, so the
/// original was quietly rounding before it wrapped. `wrapping_mul` is the
/// operation that line was written to express, and the value it produces is a
/// decorrelation phase that nothing compares against anything.
#[inline]
fn hash_phase(x: i32, y: i32) -> f32 {
    let mut h = (i64::from(x) * 374_761_393 + i64::from(y) * 668_265_263) as u32;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (f32::from((h & 0xffff) as u16) / 65536.0) * std::f32::consts::TAU
}

/// The colour an unlit surface falls toward, 0..1 per channel.
///
/// The shadow's HUE moves with depth and time of day, and that single change
/// does most of the work of making the underground feel like a different place.
/// Near the surface an unlit face is in shadow but still lit by the sky, so its
/// shadow is blue; five hundred cells down there is no sky to bounce, so the
/// shadow goes to a dead neutral that swallows colour. At night the surface
/// shadow cools and deepens toward the same place.
pub fn shadow_tint(depth: f32, day: f32) -> [f32; 3] {
    let night = 1.0 - day;
    let mut out = [0.0f32; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let v =
            SHADOW_SKY[i] + (SHADOW_DEEP[i] - SHADOW_SKY[i]) * depth - night * SHADOW_NIGHT_COOL[i];
        // Floored to a whole 0..255 byte before use, as the original did: the
        // shadow hue is an 8-bit colour, and carrying more precision than that
        // into the multiply would put a value in it that no pixel could hold.
        *o = v.max(0.0).trunc() / 255.0;
    }
    out
}

/// The light a cave keeps however unlit it is.
pub fn ambient_floor(depth: f32) -> f32 {
    AMBIENT_FLOOR - depth * AMBIENT_DEPTH_FALL
}

/// Depth below the surface anchor, 0 at the surface to 1 deep.
///
/// Measured in ABSOLUTE cells from the view's centre so it does not jump as the
/// streaming window scrolls. The original derived the centre from the view's top
/// edge plus half its height; [`WorldFocus`] is already the centre, so the
/// arithmetic is one term shorter and says the same thing.
pub fn depth_at(centre_y_px: f32) -> f32 {
    let cells = centre_y_px / CELL_SIZE as f32 - SURFACE_ANCHOR_Y as f32;
    (cells / DEPTH_RANGE_CELLS).clamp(0.0, 1.0)
}

/// A 0..1 float as a 0..255 byte, clamped and truncated.
#[inline]
fn unit_byte(v: f32) -> u8 {
    if v >= 1.0 {
        u8::MAX
    } else if v <= 0.0 {
        0
    } else {
        (v * 255.0) as u8
    }
}

/// Material at an absolute cell, or air if the window does not hold it.
fn cell_at_world(grid: &CellGrid, cx: i32, cy: i32) -> CellId {
    let (gx, gy) = (cx - grid.origin_cell_x(), cy - grid.origin_cell_y());
    if gx < 0 || gy < 0 || gx >= grid.cols() || gy >= grid.rows() {
        EMPTY
    } else {
        grid.material[(gy * grid.cols() + gx) as usize]
    }
}

// --- The census scan ---------------------------------------------------------

/// Absolute cells of every emitting cell in a rect, deduplicated and capped.
///
/// See the module header for why this walk is paid here rather than taken as a
/// by-product of the cell blit. `x` and `y` are parallel and always the same
/// length, so they hand to [`LightGrid::add_census_emitters`] in exactly the
/// shape [`crate::cells::EmitterCensus`] publishes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmitterScan {
    x: Vec<i32>,
    y: Vec<i32>,
}

impl EmitterScan {
    /// Absolute cell X of each emitter found.
    pub fn x(&self) -> &[i32] {
        &self.x
    }
    /// Absolute cell Y of each emitter found.
    pub fn y(&self) -> &[i32] {
        &self.y
    }
    /// How many were found, at most [`EMIT_CENSUS_MAX`].
    pub fn len(&self) -> usize {
        self.x.len()
    }
    /// Whether the rect contained no emitters at all.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Walk every cell of the rect whose top-left is absolute cell `(ox, oy)` and
/// record the emitters, at most one per [`CENSUS_GROUP`] cells per row.
///
/// Refills `out` rather than returning a fresh one: this runs every frame over
/// up to ~16 000 cells, and those two vectors are the only allocation the whole
/// light pass would otherwise make.
pub fn scan_emitters(grid: &CellGrid, ox: i32, oy: i32, w: i32, h: i32, out: &mut EmitterScan) {
    out.x.clear();
    out.y.clear();

    let gx0 = grid.origin_cell_x();
    let gy0 = grid.origin_cell_y();
    let (cols, rows) = (grid.cols(), grid.rows());

    // Clip to the loaded window once, per axis, rather than testing every cell:
    // outside it there is nothing to find.
    let x0 = ox.max(gx0);
    let x1 = (ox + w).min(gx0 + cols);
    let y0 = oy.max(gy0);
    let y1 = (oy + h).min(gy0 + rows);

    for cy in y0..y1 {
        let row = (cy - gy0) * cols;
        let mut last_group = i32::MIN;
        for cx in x0..x1 {
            if MAT_LIGHT[grid.material[(row + cx - gx0) as usize] as usize] == 0 {
                continue;
            }
            let group = cx.div_euclid(CENSUS_GROUP);
            if group == last_group {
                continue;
            }
            last_group = group;
            out.x.push(cx);
            out.y.push(cy);
            if out.x.len() >= EMIT_CENSUS_MAX {
                return;
            }
        }
    }
}

// --- Bloom -------------------------------------------------------------------

/// A rectangle in world px, sim convention (+y down), given by its top-left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect2 {
    /// Left edge, world px.
    pub x: f32,
    /// Top edge, world px, +y DOWN.
    pub y: f32,
    /// Width, world px.
    pub w: f32,
    /// Height, world px.
    pub h: f32,
}

/// One additive glow sprite to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomProbe {
    /// World px, +x right — the sprite's centre.
    pub x: f32,
    /// World px, +y DOWN. The Bevy flip happens at draw time.
    pub y: f32,
    /// Which baked sprite to draw.
    pub hue: GlowHue,
    /// Strength, 0..1. Folds the emitter's declared level and its bloom flicker.
    pub alpha: f32,
}

/// Find the emissive clusters worth a glow sprite over the visible rect.
///
/// Walks visible cells on a coarse stride and stops at [`BLOOM_BUDGET`] so cost
/// is bounded no matter how much lava is on screen. Refills `out` for the same
/// reason [`scan_emitters`] does.
pub fn bloom_probes(grid: &CellGrid, view: Rect2, t: f32, out: &mut Vec<BloomProbe>) {
    out.clear();
    let stride = BLOOM_STRIDE_CELLS;
    let half = stride / 2;
    let wpx = (stride * CELL_SIZE) as f32;

    let cx0 = (view.x / wpx).floor() as i32;
    let cy0 = (view.y / wpx).floor() as i32;
    let cols = (view.w / wpx).ceil() as i32 + 1;
    let rows = (view.h / wpx).ceil() as i32 + 1;

    let gx0 = grid.origin_cell_x();
    let gy0 = grid.origin_cell_y();
    let (gw, gh) = (grid.cols(), grid.rows());

    for ry in 0..rows {
        let cy = (cy0 + ry) * stride + half;
        let gy = cy - gy0;
        if gy < 0 || gy >= gh {
            continue;
        }
        let row = gy * gw;
        for rx in 0..cols {
            let cx = (cx0 + rx) * stride + half;
            let gx = cx - gx0;
            if gx < 0 || gx >= gw {
                continue;
            }
            let e = emitter(grid.material[(row + gx) as usize]);
            if !e.emits() {
                continue;
            }
            // The halo's strength follows the emitter's declared level, so a
            // weak source gets a halo rather than the same flare as a lava lake.
            let wave = (t * BLOOM_RATE + hash_phase(cx, cy) * BLOOM_PHASE_SCALE).sin();
            out.push(BloomProbe {
                x: (cx * CELL_SIZE) as f32,
                y: (cy * CELL_SIZE) as f32,
                hue: e.hue,
                alpha: e.level * (BLOOM_BASE + BLOOM_SWING * wave),
            });
            if out.len() >= BLOOM_BUDGET {
                return;
            }
        }
    }
}

// --- Vignette ----------------------------------------------------------------

/// Vignette texture size for a view, in samples.
pub fn vignette_size(view: View) -> (i32, i32) {
    let ceil = |px: i32| (px.max(0) + VIGNETTE_CELL - 1) / VIGNETTE_CELL + 1;
    (ceil(view.w), ceil(view.h))
}

/// Bake the radial vignette into RGBA multiply factors.
///
/// The same algebra as [`LightGrid::bake_shadow`], over a gradient rather than a
/// grid. The original built a two-stop `createRadialGradient` and drew it with
/// `multiply`, and the inner stop is opaque WHITE — which multiplies to a no-op,
/// so the bright core is genuinely untouched and only the ramp toward the edge
/// does anything.
///
/// `out` must be `vignette_size(view)` texels of 4 bytes.
pub fn bake_vignette(view: View, depth: f32, day: f32, out: &mut [u8]) {
    let (cols, rows) = vignette_size(view);
    let night = 1.0 - day;
    let (w, h) = (view.w as f32, view.h as f32);

    let inner = VIGNETTE_INNER - depth * VIGNETTE_INNER_DEPTH - night * VIGNETTE_INNER_NIGHT;
    let edge = VIGNETTE_EDGE - depth * VIGNETTE_EDGE_DEPTH + night * VIGNETTE_EDGE_NIGHT;
    let r0 = w.min(h) * inner;
    let r1 = w.max(h) * VIGNETTE_OUTER;
    let span = (r1 - r0).max(f32::MIN_POSITIVE);

    let mut tint = [0.0f32; 3];
    for (i, o) in tint.iter_mut().enumerate() {
        *o = (VIGNETTE_RGB[i] - depth * VIGNETTE_RGB_DEPTH[i]).max(0.0) / 255.0;
    }

    for (i, px) in out.chunks_exact_mut(4).enumerate() {
        let sx = (i as i32 % cols) as f32 / (cols - 1).max(1) as f32 * w;
        let sy = (i as i32 / cols) as f32 / (rows - 1).max(1) as f32 * h;
        let r = ((sx - w * 0.5).powi(2) + (sy - h * 0.5).powi(2)).sqrt();
        let u = ((r - r0) / span).clamp(0.0, 1.0);
        // Both stops lerp together: colour from white toward the edge tint,
        // alpha from 1 toward `edge`.
        let a = 1.0 + (edge - 1.0) * u;
        for (dst, t) in px.iter_mut().zip(tint) {
            let cs = 1.0 + (t - 1.0) * u;
            *dst = unit_byte(1.0 - a + a * cs);
        }
        px[3] = u8::MAX;
    }
}

// --- The composite materials -------------------------------------------------

/// The one shader every light quad uses: sample, tint, hand it to the blend.
///
/// Held as a string rather than a `.wgsl` beside the module because it carries
/// no logic — all of the arithmetic is baked into the textures by
/// [`LightGrid::bake_shadow`], [`LightGrid::bake_colour`] and [`bake_vignette`],
/// which is what makes those decisions testable on the CPU. `cells.wgsl` earns
/// its own file by being a real shading model; four lines of passthrough does
/// not.
const LIGHT_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var light_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var light_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> tint: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(light_texture, light_sampler, mesh.uv) * tint;
}
"#;

/// Handle for [`LIGHT_WGSL`], inserted by [`LightPlugin`].
const LIGHT_SHADER: Handle<Shader> = uuid_handle!("6b1a0f2c-9d34-4a71-8e55-2c0f7b3d9a10");

/// `dst * src`, with the destination's alpha left alone.
///
/// The exact translation of Canvas2D's `multiply` composite once the
/// `1 - a + a * shadow` term is baked into the source — see
/// [`LightGrid::bake_shadow`].
const BLEND_MULTIPLY: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::Dst,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::Zero,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
};

/// `dst + src`. Canvas2D's `lighter`, with the destination's alpha left alone.
const BLEND_ADD: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::Zero,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
};

/// Overwrite the colour target's blend state with this pass's own.
///
/// `alpha_mode` puts the quad in the transparent phase with depth writes off,
/// which is what gets the ordering right; the blend state it picks there is then
/// replaced, because [`AlphaMode2d`] has three variants and none of them is
/// either of the two this module needs.
fn set_blend(descriptor: &mut RenderPipelineDescriptor, blend: BlendState) {
    if let Some(fragment) = descriptor.fragment.as_mut()
        && let Some(Some(target)) = fragment.targets.first_mut()
    {
        target.blend = Some(blend);
    }
}

/// Define a light material: one texture, one tint, one blend state.
///
/// A macro, and not one type carrying the blend in its bind-group data, because
/// `specialize` is a STATIC method — the blend has to be a property of the type
/// for the pipeline cache to key on it, and two named types say that more
/// plainly than a key struct would.
macro_rules! light_material {
    ($(#[$meta:meta])* $name:ident, $blend:expr) => {
        $(#[$meta])*
        #[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
        pub struct $name {
            /// The grid or sprite this quad samples. Sampled LINEARLY — the one
            /// place in this game that is not nearest — because the whole point
            /// of a light grid a sixteenth of the viewport's size is that it
            /// upscales smoothly.
            #[texture(0)]
            #[sampler(1)]
            pub texture: Handle<Image>,
            /// Multiplies the sampled texel. Canvas2D's `globalAlpha`, with the
            /// per-channel freedom the flat washes need.
            #[uniform(2)]
            pub tint: Vec4,
        }

        impl Material2d for $name {
            fn fragment_shader() -> ShaderRef {
                LIGHT_SHADER.into()
            }

            fn alpha_mode(&self) -> AlphaMode2d {
                AlphaMode2d::Blend
            }

            fn specialize(
                descriptor: &mut RenderPipelineDescriptor,
                _layout: &MeshVertexBufferLayoutRef,
                _key: Material2dKey<Self>,
            ) -> Result<(), SpecializedMeshPipelineError> {
                set_blend(descriptor, $blend);
                Ok(())
            }
        }
    };
}

light_material!(
    /// The darkness pass, and the vignette that follows it.
    LightShadowMaterial,
    BLEND_MULTIPLY
);

light_material!(
    /// The coloured light, the bloom sprites, and the flat washes.
    LightGlowMaterial,
    BLEND_ADD
);

// --- The plugin --------------------------------------------------------------

/// Where the darkness sits in z: over everything it darkens, under the UI.
///
/// [`crate::input`]'s brush cursor is at 1.0 and stays above all of these on
/// purpose — a cursor that dimmed with the cave it is pointing into would be
/// unusable in the one place a player most needs to aim.
const SHADOW_Z: f32 = 0.70;
/// The coloured light, immediately over the darkness it re-lights.
const COLOUR_Z: f32 = 0.71;
/// Bloom, over the coloured light it belongs to.
const BLOOM_Z: f32 = 0.72;
/// The vignette, over everything in the world.
const VIGNETTE_Z: f32 = 0.75;
/// The flat washes, last, exactly as the original drew them.
const WASH_Z: f32 = 0.76;

/// The biome's ambient cast, 0..1 per channel.
///
/// SEAM: which biome the camera is standing in is the ambience milestone's to
/// resolve. Until it lands this holds at black, which is what plains sends and
/// is a no-op in the composite. When it arrives it writes this resource and
/// nothing in this module changes.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct BiomeAmbient(pub [f32; 3]);

/// Everything the composite owns: the solver, its textures, and the frame's
/// derived geometry.
#[derive(Resource)]
pub struct LightPass {
    /// The solver. Public because the ambience layer reads [`LightGrid::hot`]
    /// and a debug overlay would read the grid itself.
    pub grid: LightGrid,
    /// The census the current frame solved against.
    pub census: EmitterScan,
    /// The bloom probes the current frame found.
    pub probes: Vec<BloomProbe>,

    /// The darkness multiply factors, one texel per light cell.
    shadow: Handle<Image>,
    /// The additive coloured light, one texel per light cell.
    colour: Handle<Image>,
    /// The radial vignette, in view space.
    vignette: Handle<Image>,
    /// One premultiplied glow sprite per [`GlowHue`].
    glows: [Handle<Image>; GLOW_COUNT],

    shadow_mat: Handle<LightShadowMaterial>,
    colour_mat: Handle<LightGlowMaterial>,
    vignette_mat: Handle<LightShadowMaterial>,
    wash_mat: Handle<LightGlowMaterial>,

    /// The view the textures above are currently sized for.
    view: View,
    /// World px of the light grid's top-left corner, sim convention.
    origin: Vec2,
    /// World px the light grid spans.
    extent: Vec2,
    /// World px of the view's centre, sim convention.
    centre: Vec2,
    /// Whether anything emitted this frame — the colour quad's visibility.
    colour_visible: bool,
}

/// Which composite quad an entity is.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightQuad {
    /// The darkness. Follows the light grid in world space.
    Shadow,
    /// The coloured light. Follows the light grid in world space.
    Colour,
    /// The radial vignette. Covers the view.
    Vignette,
    /// The underworld glow and the biome cast, summed. Covers the view.
    Wash,
}

/// One pooled bloom sprite. `slot` indexes [`LightPass::probes`].
#[derive(Component, Clone, Copy, Debug)]
pub struct BloomSprite {
    /// Index into the probe list.
    pub slot: usize,
}

/// The light solver, the composite quads, and the world clock that drives them.
///
/// Deliberately NOT added to [`crate::GodGameRenderPlugin`] here.
pub struct LightPlugin;

impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        // Inserted rather than embedded: `LIGHT_WGSL` is a string constant in
        // this file, so there is no asset path for `embedded_asset!` to take.
        // Bevy's own `ColorMaterialPlugin` reaches into the world at build time
        // for the same kind of reason.
        app.world_mut()
            .resource_mut::<Assets<Shader>>()
            .insert(&LIGHT_SHADER, Shader::from_wgsl(LIGHT_WGSL, file!()))
            .expect("the light shader handle is a uuid and cannot collide");

        app.add_plugins((
            Material2dPlugin::<LightShadowMaterial>::default(),
            Material2dPlugin::<LightGlowMaterial>::default(),
        ))
        .init_resource::<BiomeAmbient>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (solve_light, (place_quads, place_bloom).after(solve_light))
                .run_if(resource_exists::<SimWorld>)
                .run_if(resource_exists::<LowResTarget>)
                .run_if(resource_exists::<LightPass>),
        );
    }
}

/// Allocate every texture and material, and spawn the quads.
///
/// The glow sprites are baked here and never touched again — they are a pure
/// function of [`GlowHue::stops`]. The other three textures are rewritten every
/// frame, and RESIZED by [`solve_light`] on the first frame, because
/// [`LowResTarget`] is inserted by a command and so does not exist yet while
/// this runs. Sizing from [`View::default`] here and letting the first solve
/// correct it is the same trade [`crate::mobs`] makes with its spawn rectangles.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut shadow_materials: ResMut<Assets<LightShadowMaterial>>,
    mut glow_materials: ResMut<Assets<LightGlowMaterial>>,
) {
    let view = View::default();
    let (lw, lh) = grid_size(view);
    let (vw, vh) = vignette_size(view);

    let shadow = images.add(new_light_texture(lw, lh));
    let colour = images.add(new_light_texture(lw, lh));
    let vignette = images.add(new_light_texture(vw, vh));
    let glows = GLOW_HUES.map(|hue| images.add(new_glow_texture(hue)));
    // A single white texel, so the flat washes go through the same shader as
    // everything else and the tint uniform carries the whole signal.
    let white = images.add(new_light_texture(1, 1));

    let shadow_mat = shadow_materials.add(LightShadowMaterial {
        texture: shadow.clone(),
        tint: Vec4::ONE,
    });
    let vignette_mat = shadow_materials.add(LightShadowMaterial {
        texture: vignette.clone(),
        tint: Vec4::ONE,
    });
    let colour_mat = glow_materials.add(LightGlowMaterial {
        texture: colour.clone(),
        tint: Vec4::ONE,
    });
    let wash_mat = glow_materials.add(LightGlowMaterial {
        texture: white,
        tint: Vec4::ZERO,
    });

    let quad = meshes.add(Rectangle::default());
    commands.spawn(quad_bundle(
        &quad,
        MeshMaterial2d(shadow_mat.clone()),
        LightQuad::Shadow,
        SHADOW_Z,
    ));
    commands.spawn(quad_bundle(
        &quad,
        MeshMaterial2d(colour_mat.clone()),
        LightQuad::Colour,
        COLOUR_Z,
    ));
    commands.spawn(quad_bundle(
        &quad,
        MeshMaterial2d(vignette_mat.clone()),
        LightQuad::Vignette,
        VIGNETTE_Z,
    ));
    commands.spawn(quad_bundle(
        &quad,
        MeshMaterial2d(wash_mat.clone()),
        LightQuad::Wash,
        WASH_Z,
    ));

    // The bloom sprites are POOLED, on the same terms `crate::mobs`' creature
    // rectangles are: a fixed budget of entities, hidden when their slot is not
    // live. A lava lake scrolling into view must not churn the ECS's archetypes
    // once per probe per frame.
    let side = glow_side();
    for slot in 0..BLOOM_BUDGET {
        let material = glow_materials.add(LightGlowMaterial {
            texture: glows[0].clone(),
            tint: Vec4::ZERO,
        });
        commands.spawn((
            Mesh2d(quad.clone()),
            MeshMaterial2d(material),
            Transform::from_xyz(0.0, 0.0, BLOOM_Z).with_scale(Vec3::new(side, side, 1.0)),
            Visibility::Hidden,
            BloomSprite { slot },
            WORLD_LAYERS,
        ));
    }

    commands.insert_resource(LightPass {
        grid: LightGrid::new(view, SEED),
        census: EmitterScan::default(),
        probes: Vec::with_capacity(BLOOM_BUDGET),
        shadow,
        colour,
        vignette,
        glows,
        shadow_mat,
        colour_mat,
        vignette_mat,
        wash_mat,
        view,
        origin: Vec2::ZERO,
        extent: Vec2::ZERO,
        centre: Vec2::ZERO,
        colour_visible: false,
    });
}

/// The components every composite quad shares.
fn quad_bundle<M: Material2d>(
    mesh: &Handle<Mesh>,
    material: MeshMaterial2d<M>,
    which: LightQuad,
    z: f32,
) -> impl Bundle {
    (
        Mesh2d(mesh.clone()),
        material,
        // Placed by `place_quads` on the first frame; this only avoids one frame
        // of a unit square at the origin.
        Transform::from_xyz(0.0, 0.0, z),
        which,
        WORLD_LAYERS,
    )
}

/// The glow sprite's side, in world px.
fn glow_side() -> f32 {
    (GLOW_RADIUS_PX * 2) as f32
}

/// A blank `Rgba8Unorm` grid with a LINEAR sampler.
///
/// Not sRGB: every byte in these textures is a blend factor, and a transfer
/// function applied on the way in would silently change the arithmetic
/// [`LightGrid::bake_shadow`] worked out.
///
/// Linear sampling is the whole trick, and the reason the game's one
/// `default_nearest` sampler is overridden here. The grid is one texel per 20
/// world px; nearest sampling would put a hard 20px lattice over the entire
/// frame. The texel centres land exactly on the light cells' own sample centres,
/// so the interpolation is between the two values it should be between.
fn new_light_texture(w: i32, h: i32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: w.max(1) as u32,
            height: h.max(1) as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[u8::MAX; 4],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    image.sampler = ImageSampler::linear();
    image
}

/// Bake one radial glow: hot core, tinted halo, fully transparent rim.
///
/// PREMULTIPLIED. The sprite is only ever drawn additively, and an additive draw
/// adds `rgb * alpha` — so the texel that gets added IS the premultiplied value.
/// Storing it that way means the shader has nothing to do, the linear sampler
/// interpolates the right quantity, and the two gradient stops interpolate the
/// way Canvas2D's specification says a gradient must, which straight RGBA
/// between a white core and a saturated halo would not.
fn new_glow_texture(hue: GlowHue) -> Image {
    let (core, halo) = hue.stops();
    let r = GLOW_RADIUS_PX as f32;
    let side = (GLOW_RADIUS_PX * 2) as usize;
    let mut data = vec![0u8; side * side * 4];

    for (i, px) in data.chunks_exact_mut(4).enumerate() {
        let x = (i % side) as f32 - r + 0.5;
        let y = (i / side) as f32 - r + 0.5;
        let u = (x * x + y * y).sqrt() / r;
        if u >= 1.0 {
            continue;
        }
        // Two segments: core handing over to halo across `GLOW_CORE_STOP`, then
        // the halo fading to nothing over the rest of the radius.
        let (rgb, a) = if u <= GLOW_CORE_STOP {
            let k = u / GLOW_CORE_STOP;
            let mut c = [0.0f32; 3];
            for (ch, o) in c.iter_mut().enumerate() {
                *o = (core[ch] + (halo[ch] - core[ch]) * k) / 255.0;
            }
            (c, GLOW_CORE_ALPHA + (GLOW_HALO_ALPHA - GLOW_CORE_ALPHA) * k)
        } else {
            let k = (u - GLOW_CORE_STOP) / (1.0 - GLOW_CORE_STOP);
            let mut c = [0.0f32; 3];
            for (ch, o) in c.iter_mut().enumerate() {
                *o = halo[ch] / 255.0;
            }
            (c, GLOW_HALO_ALPHA * (1.0 - k))
        };
        for (dst, v) in px.iter_mut().zip(rgb) {
            *dst = unit_byte(v * a);
        }
        px[3] = u8::MAX;
    }

    let mut image = Image::new(
        Extent3d {
            width: side as u32,
            height: side as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    image.sampler = ImageSampler::linear();
    image
}

/// Everything the solve reads, bundled so the system stays five parameters.
#[derive(SystemParam)]
struct LightInputs<'w> {
    world: Res<'w, SimWorld>,
    focus: Res<'w, WorldFocus>,
    target: Res<'w, LowResTarget>,
    clock: Res<'w, WorldClock>,
    ambient: Res<'w, BiomeAmbient>,
    time: Res<'w, Time>,
}

/// One frame of light: resize if the window changed, scan, solve, bake, upload.
fn solve_light(
    inputs: LightInputs,
    mut pass: ResMut<LightPass>,
    mut images: ResMut<Assets<Image>>,
    mut shadow_materials: ResMut<Assets<LightShadowMaterial>>,
    mut glow_materials: ResMut<Assets<LightGlowMaterial>>,
) {
    let view = inputs.target.view;
    let cells = &inputs.world.level.grid;
    let day = inputs.clock.0.phase().day;
    let t = inputs.time.elapsed_secs();

    if pass.view != view {
        resize(&mut pass, view, &mut images);
        // The three grid textures just changed extent, so the bind groups that
        // hold them have to be rebuilt. Touching the materials is how a
        // `Material2d` asks for that.
        touch(&mut shadow_materials, &pass.shadow_mat.clone());
        touch(&mut shadow_materials, &pass.vignette_mat.clone());
        touch(&mut glow_materials, &pass.colour_mat.clone());
    }

    // The view rect in world px, sim convention. `WorldFocus` is its centre.
    let centre = Vec2::new(inputs.focus.x, inputs.focus.y);
    let rect = Rect2 {
        x: centre.x - view.w as f32 * 0.5,
        y: centre.y - view.h as f32 * 0.5,
        w: view.w as f32,
        h: view.h as f32,
    };
    let stride = light_stride_px() as f32;
    let frame = LightFrame {
        // Minus one for the margin the grid carries on every side.
        ox: (rect.x / stride).floor() as i32 - 1,
        oy: (rect.y / stride).floor() as i32 - 1,
        day,
        t,
    };
    let depth = depth_at(centre.y);

    // The census walks the visible cells at FULL resolution, plus a cell of
    // slack on each side so a torch straddling the edge still lights what is on
    // screen.
    let cell = CELL_SIZE as f32;
    scan_emitters(
        cells,
        (rect.x / cell).floor() as i32 - 1,
        (rect.y / cell).floor() as i32 - 1,
        (rect.w / cell).ceil() as i32 + 2,
        (rect.h / cell).ceil() as i32 + 2,
        &mut pass.census,
    );

    // Split the borrow: `solve` needs the grid mutably and the census by
    // reference, and they are two fields of the same resource.
    {
        let LightPass {
            grid,
            census,
            probes,
            ..
        } = &mut *pass;
        grid.solve(cells, frame, census.x(), census.y());
        bloom_probes(cells, rect, t, probes);
    }

    pass.origin = Vec2::new(frame.ox as f32 * stride, frame.oy as f32 * stride);
    pass.extent = Vec2::new(
        pass.grid.cols() as f32 * stride,
        pass.grid.rows() as f32 * stride,
    );
    pass.centre = centre;
    pass.colour_visible = pass.grid.colour_dirty();

    let (shadow, colour, vignette) = (
        pass.shadow.clone(),
        pass.colour.clone(),
        pass.vignette.clone(),
    );
    upload(&shadow, &mut images, |out| {
        pass.grid.bake_shadow(depth, day, out);
    });
    if pass.colour_visible {
        upload(&colour, &mut images, |out| pass.grid.bake_colour(out));
    }
    upload(&vignette, &mut images, |out| {
        bake_vignette(view, depth, day, out);
    });

    // The two flat washes are both additive fills over the whole view, so they
    // are one quad: adding two colours and adding their sum are the same
    // operation, and the second `fillRect` bought the original nothing.
    let mut wash = Vec3::ZERO;
    if depth > UNDERWORLD_GLOW_DEPTH {
        let k = (depth - UNDERWORLD_GLOW_DEPTH) / (1.0 - UNDERWORLD_GLOW_DEPTH);
        wash += Vec3::from_array(UNDERWORLD_RGB) / 255.0 * (k * UNDERWORLD_ALPHA);
    }
    let biome = inputs.ambient.0;
    if biome.iter().any(|c| *c > 0.0) {
        wash += Vec3::from_array(biome) * BIOME_AMBIENT_ALPHA;
    }
    if let Some(mut material) = glow_materials.get_mut(&pass.wash_mat) {
        material.tint = wash.extend(1.0);
    }
}

/// Rebuild the solver and every view-sized texture for a new window size.
///
/// In the TypeScript this could not happen: `VIEW_W`/`VIEW_H` were frozen from
/// `window.innerWidth` at import time and a resize did nothing until reload. A
/// native window can be dragged, so the grid is a function of the view here —
/// the same reason `crate::mobs`' spawn rectangles are.
fn resize(pass: &mut LightPass, view: View, images: &mut Assets<Image>) {
    pass.view = view;
    pass.grid = LightGrid::new(view, SEED);
    let (lw, lh) = grid_size(view);
    let (vw, vh) = vignette_size(view);
    for (handle, (w, h)) in [
        (pass.shadow.clone(), (lw, lh)),
        (pass.colour.clone(), (lw, lh)),
        (pass.vignette.clone(), (vw, vh)),
    ] {
        if let Some(mut image) = images.get_mut(&handle) {
            image.resize(Extent3d {
                width: w.max(1) as u32,
                height: h.max(1) as u32,
                depth_or_array_layers: 1,
            });
        }
    }
}

/// Mark an asset changed without editing it.
///
/// `Assets::get_mut` is the only way to say "this asset's dependents need
/// re-preparing", and its return value is deliberately discarded here.
fn touch<A: Asset>(assets: &mut Assets<A>, handle: &Handle<A>) {
    let _ = assets.get_mut(handle);
}

/// Refill a texture's bytes in place.
///
/// `Assets::get_mut` is what schedules the GPU upload, so this is only ever
/// called on a pass that actually has something new to say.
fn upload(handle: &Handle<Image>, images: &mut Assets<Image>, fill: impl FnOnce(&mut [u8])) {
    if let Some(mut image) = images.get_mut(handle)
        && let Some(data) = image.data.as_mut()
    {
        fill(data);
    }
}

/// Put the four composite quads where this frame's geometry says they go.
///
/// Bevy's +y is up and the sim's is down. This function and [`place_bloom`] are
/// the only two places in the module where that is true.
fn place_quads(
    pass: Res<LightPass>,
    mut quads: Query<(&LightQuad, &mut Transform, &mut Visibility)>,
) {
    let grid_centre = pass.origin + pass.extent * 0.5;
    for (which, mut transform, mut visibility) in &mut quads {
        let (centre, extent, shown) = match which {
            LightQuad::Shadow => (grid_centre, pass.extent, true),
            LightQuad::Colour => (grid_centre, pass.extent, pass.colour_visible),
            // The view quads are one px proud on each axis. The world camera is
            // snapped to whole pixels by `crate::lowres` and the focus is not,
            // so an exactly view-sized quad can leave a hairline of uncomposited
            // frame at an edge; a pixel of overhang costs nothing and cannot.
            LightQuad::Vignette | LightQuad::Wash => (
                Vec2::new(pass.centre.x.round(), pass.centre.y.round()),
                Vec2::new(pass.view.w as f32 + 2.0, pass.view.h as f32 + 2.0),
                true,
            ),
        };
        *visibility = if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        transform.translation.x = centre.x;
        transform.translation.y = -centre.y;
        transform.scale.x = extent.x.max(1.0);
        transform.scale.y = extent.y.max(1.0);
    }
}

/// Point every pooled bloom sprite at a probe, or hide its slot.
fn place_bloom(
    pass: Res<LightPass>,
    mut materials: ResMut<Assets<LightGlowMaterial>>,
    mut sprites: Query<(
        &BloomSprite,
        &MeshMaterial2d<LightGlowMaterial>,
        &mut Transform,
        &mut Visibility,
    )>,
) {
    for (sprite, handle, mut transform, mut visibility) in &mut sprites {
        let Some(probe) = pass.probes.get(sprite.slot) else {
            *visibility = Visibility::Hidden;
            continue;
        };
        *visibility = Visibility::Inherited;
        transform.translation.x = probe.x;
        transform.translation.y = -probe.y;
        if let Some(mut material) = materials.get_mut(handle.id()) {
            material.texture = pass.glows[probe.hue.index()].clone();
            // The strength rides in the tint: an additive draw of a
            // premultiplied sprite scaled by `alpha` is exactly Canvas2D's
            // `globalAlpha` with `lighter`, which is what the original did once
            // per `drawImage`.
            material.tint = Vec4::splat(probe.alpha.clamp(0.0, 1.0)).with_w(1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::sim::materials::block;

    /// A world of air with a stone floor from `floor_row` down, origin at 0.
    fn world(cols: i32, rows: i32, floor_row: i32) -> CellGrid {
        let mut g = CellGrid::new(cols, rows);
        for y in 0..rows {
            for x in 0..cols {
                g.set(x, y, if y >= floor_row { block::STONE } else { EMPTY });
            }
        }
        g
    }

    /// A world of air, so a test can place exactly what it means to test.
    fn air(cols: i32, rows: i32) -> CellGrid {
        CellGrid::new(cols, rows)
    }

    /// A solver over a fixed 16x16 light grid, so a test can name a cell.
    ///
    /// The view-derived size is incidental to every propagation rule below;
    /// pinning it keeps the assertions readable and independent of what window
    /// the test happens to run on.
    fn solver() -> LightGrid {
        let mut grid = LightGrid::new(View::default(), SEED);
        let n = 16 * 16;
        grid.lw = 16;
        grid.lh = 16;
        grid.light = vec![0.0; n];
        grid.scratch = vec![0.0; n];
        grid.lr = vec![0.0; n];
        grid.lg = vec![0.0; n];
        grid.lb = vec![0.0; n];
        grid.seen = vec![false; n];
        grid
    }

    fn noon(ox: i32, oy: i32) -> LightFrame {
        LightFrame {
            ox,
            oy,
            day: 1.0,
            t: 0.0,
        }
    }

    /// The first material declaring a light level above `min_level`.
    ///
    /// Read out of the content build rather than named, so this suite does not
    /// have to be edited every time a block is added.
    fn an_emitter(min_level: f32) -> CellId {
        (1..MAT_COUNT as CellId)
            .find(|&id| emitter(id).level > min_level)
            .expect("the content set has to contain at least one light source")
    }

    #[test]
    fn the_light_grid_covers_the_view_plus_a_margin_on_every_side() {
        let view = View::for_screen(1440, 900);
        let (lw, lh) = grid_size(view);
        let stride = light_stride_px();
        // Enough cells to cover the view...
        assert!((lw - 2) * stride >= view.w, "{lw} cells is too narrow");
        assert!((lh - 2) * stride >= view.h, "{lh} cells is too short");
        // ...and exactly one spare on each side, never two.
        assert!((lw - 3) * stride < view.w);
        assert!((lh - 3) * stride < view.h);
    }

    #[test]
    fn open_sky_at_noon_floods_the_whole_column_evenly() {
        // Air all the way down: every row of the column should land within a
        // hair of every other. This is the exact failure the 0.94 open decay
        // caused — the bottom of the screen at a fifth of the top's brightness
        // with nothing but sky in between.
        let g = air(64, 64);
        let mut light = solver();
        light.compute_skylight(&g, noon(0, 0));

        let top = light.light_at(4, 0);
        let bottom = light.light_at(4, light.rows() - 1);
        assert!(top > 0.9, "noon sky should be near full, was {top}");
        assert!(
            bottom > top * 0.8,
            "open air ate too much light: {top} at the top, {bottom} at the bottom"
        );
    }

    #[test]
    fn light_decays_far_faster_through_rock_than_through_air() {
        let g = world(64, 64, 0); // solid from the very top
        let mut light = solver();
        light.compute_skylight(&g, noon(0, 0));

        let first = light.light_at(4, 0);
        let fourth = light.light_at(4, 3);
        let expected = first * SOLID_DECAY.powi(3);
        assert!(
            (fourth - expected).abs() < 1e-4,
            "expected {expected} four light rows into rock, got {fourth}"
        );
        assert!(fourth < first * 0.2, "rock has to actually occlude");
    }

    #[test]
    fn a_solid_cell_is_lit_by_what_reaches_it_and_not_by_what_it_blocks() {
        // STORE, THEN DECAY. The sunlit ground the player stands on is the
        // brightest solid in its column; a pass that decayed first would make it
        // darker than the air above by the full solid factor, and the whole
        // world would read as overcast dusk at noon.
        let g = world(64, 64, 8);
        let mut light = solver();
        light.compute_skylight(&g, noon(0, 0));

        // Light rows sample cells 2, 6, 10, ... — row 2 is the first solid one.
        let air_above = light.light_at(4, 1);
        let ground = light.light_at(4, 2);
        assert!(
            ground > air_above * OPEN_DECAY - 1e-5,
            "the lit face of the terrain was darkened by its own occlusion: \
             {air_above} in the air above, {ground} on the ground"
        );
    }

    #[test]
    fn night_leaves_moonlight_rather_than_nothing() {
        let g = air(64, 64);
        let mut light = solver();
        light.compute_skylight(
            &g,
            LightFrame {
                day: 0.0,
                ..noon(0, 0)
            },
        );
        let midnight = light.light_at(4, 0);
        assert!(midnight > 0.0, "midnight went to pure black");
        assert!((midnight - SKY_NIGHT).abs() < 1e-5);

        light.compute_skylight(&g, noon(0, 0));
        assert!(
            light.light_at(4, 0) > midnight * 3.0,
            "noon has to be dramatically brighter than midnight"
        );
    }

    #[test]
    fn an_emitter_lights_its_own_cell_and_its_four_neighbours() {
        let id = an_emitter(0.0);
        let mut g = air(64, 64);
        // Light cell (4, 4) samples cell (18, 18) — the standard half-offset.
        g.set(18, 18, id);
        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));

        let centre = light.light_at(4, 4);
        assert!(centre > 0.0, "the emitter did not light its own cell");
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let edge = light.light_at(4 + dx, 4 + dy);
            assert!(
                (edge - centre * SPLAT_EDGE).abs() < 1e-4,
                "neighbour ({dx}, {dy}) got {edge}, expected {}",
                centre * SPLAT_EDGE
            );
        }
        // The diagonal is not part of the CROSS; it is only ever reached by a
        // strong emitter's far ring, and by the blur.
        let diagonal = if emitter(id).level > FAR_SPLAT_LEVEL {
            centre * FAR_SPLAT_GAIN
        } else {
            0.0
        };
        assert!((light.light_at(3, 3) - diagonal).abs() < 1e-4);
    }

    #[test]
    fn light_falls_off_to_nothing_at_the_radius() {
        // A strong emitter throws a far ring two light cells out and NOTHING
        // beyond it. Before the reach scaled with the declared level, a magma
        // chamber and a glowing mushroom lit the same volume.
        let strong = an_emitter(FAR_SPLAT_LEVEL);
        let mut g = air(64, 64);
        g.set(18, 18, strong);
        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));

        assert!(light.light_at(6, 4) > 0.0, "the far ring is missing");
        assert_eq!(
            light.light_at(7, 4),
            0.0,
            "the cast reached past its radius"
        );
        assert_eq!(light.light_at(4, 7), 0.0);

        // ...and a weak one does not throw the ring at all.
        let weak = (1..MAT_COUNT as CellId)
            .find(|&id| emitter(id).emits() && emitter(id).level <= FAR_SPLAT_LEVEL);
        if let Some(weak) = weak {
            let mut g = air(64, 64);
            g.set(18, 18, weak);
            let mut light = solver();
            light.add_emissive(&g, noon(0, 0));
            assert_eq!(
                light.light_at(6, 4),
                0.0,
                "a weak emitter must not throw the far ring"
            );
        }
    }

    #[test]
    fn an_emitters_cast_carries_its_own_hue_and_not_its_brightness() {
        // Every emitting material has exactly one channel at full: the hue is
        // normalised by its own max channel, so what survives is the colour and
        // not the block's brightness, which is what `level` is for.
        let mut seen = 0;
        for id in 1..MAT_COUNT as CellId {
            let e = emitter(id);
            if !e.emits() {
                continue;
            }
            seen += 1;
            let mx = e.rgb.iter().copied().fold(0.0f32, f32::max);
            assert!(
                (mx - 1.0).abs() < 1e-5,
                "material {id} casts {:?}, whose max channel is {mx}",
                e.rgb
            );
            assert!(e.rgb.iter().all(|c| (0.0..=1.0).contains(c)));
            assert!((0.0..=1.0).contains(&e.level));
        }
        assert!(seen > 0, "the content set declares no light sources at all");
    }

    #[test]
    fn residual_heat_glows_warm_but_only_above_the_threshold() {
        let mut g = air(64, 64);
        // Two light cells apart, so neither splat lands on the other's centre.
        let at_threshold = (18 * 64 + 18) as usize;
        let boiling = (18 * 64 + 38) as usize;
        g.temp[at_threshold] = HEAT_THRESHOLD as u8;
        g.temp[boiling] = u8::MAX;

        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));

        assert_eq!(
            light.light_at(4, 4),
            0.0,
            "a cell exactly at the threshold must not glow"
        );
        let glow = light.light_at(9, 4);
        assert!((glow - HEAT_GAIN).abs() < 1e-5, "hottest rock got {glow}");
        // And it is warm: more red than green, more green than blue.
        let [r, g_, b] = light.colour_at(9, 4);
        assert!(r > g_ && g_ > b, "heat cast {r},{g_},{b} — not warm");
    }

    #[test]
    fn light_never_exceeds_one_however_many_emitters_stack() {
        let id = an_emitter(0.0);
        let mut g = air(64, 64);
        // A solid block of emitter: every light cell samples one, and every
        // splat lands on its neighbours too.
        for y in 0..40 {
            for x in 0..40 {
                g.set(x, y, id);
            }
        }
        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));
        light.blur();

        for ly in 0..light.rows() {
            for lx in 0..light.cols() {
                let v = light.light_at(lx, ly);
                assert!((0.0..=1.0).contains(&v), "({lx}, {ly}) blew out to {v}");
            }
        }
    }

    #[test]
    fn the_colour_grids_stay_clean_when_nothing_emitted() {
        let g = world(64, 64, 8);
        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));
        assert!(
            !light.colour_dirty(),
            "plain stone and air marked the colour pass dirty"
        );
        assert_eq!(light.colour_at(4, 4), [0.0; 3]);
        assert!(light.hot().is_empty());

        // ...and go dirty the moment something does, even if it is only heat —
        // the case the hot list alone would miss.
        let mut warm = world(64, 64, 8);
        warm.temp[(18 * 64 + 18) as usize] = 200;
        light.add_emissive(&warm, noon(0, 0));
        assert!(light.colour_dirty());
        assert!(light.hot().is_empty(), "warm rock is not an emitter");
    }

    #[test]
    fn the_census_splat_is_deduplicated_per_light_cell() {
        let id = an_emitter(0.0);
        let mut g = air(64, 64);
        // Four cells that all fall in light cell (4, 4).
        for cx in 16..20 {
            g.set(cx, 18, id);
        }
        let frame = noon(0, 0);

        let mut four = solver();
        four.add_emissive(&air(64, 64), frame); // clear, no emitters sampled
        four.add_census_emitters(&g, &[16, 17, 18, 19], &[18, 18, 18, 18], frame);

        // The FIRST entry in the group wins, phase and all — the flicker is
        // hashed from the entry's own absolute cell, so which one survives the
        // dedup is observable and this pins it.
        let mut one = solver();
        one.add_emissive(&air(64, 64), frame);
        one.add_census_emitters(&g, &[16], &[18], frame);

        // One splat, not four: four stacked splats of the same source would
        // blow the cell out to white.
        assert!(
            (four.light_at(4, 4) - one.light_at(4, 4)).abs() < 1e-6,
            "the census stacked: {} against {}",
            four.light_at(4, 4),
            one.light_at(4, 4)
        );
        assert_eq!(four.hot().len(), 1, "one light cell, one hot entry");
        assert_eq!(four.hot()[0], [16, 18]);
    }

    #[test]
    fn a_census_entry_the_grid_no_longer_backs_is_dropped() {
        // The census says WHERE, the grid says WHAT. A torch dug out between the
        // scan and the splat must not leave a ghost light behind.
        let frame = noon(0, 0);
        let mut light = solver();
        light.add_emissive(&air(64, 64), frame);
        light.add_census_emitters(&air(64, 64), &[18], &[18], frame);
        assert_eq!(light.light_at(4, 4), 0.0);
        assert!(light.hot().is_empty());
    }

    #[test]
    fn a_census_entry_outside_the_grid_is_ignored() {
        let mut light = solver();
        let frame = noon(0, 0);
        let g = air(64, 64);
        light.add_emissive(&g, frame);
        // Far left, far right, far above, far below.
        light.add_census_emitters(&g, &[-400, 4000, 18, 18], &[18, 18, -400, 4000], frame);
        for ly in 0..light.rows() {
            for lx in 0..light.cols() {
                assert_eq!(light.light_at(lx, ly), 0.0, "({lx}, {ly}) was lit");
            }
        }
    }

    #[test]
    fn the_blur_conserves_a_flat_field_and_spreads_a_spike() {
        let mut light = solver();
        let (lw, lh) = (light.cols(), light.rows());
        light.light.fill(0.5);
        blur_one(&mut light.light, &mut light.scratch, lw, lh);
        for (i, v) in light.light.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "cell {i} drifted to {v} — the edge clamp is wrong"
            );
        }

        light.light.fill(0.0);
        let centre = (8 * lw + 8) as usize;
        light.light[centre] = 1.0;
        blur_one(&mut light.light, &mut light.scratch, lw, lh);
        assert!(light.light[centre] < 1.0, "the spike did not spread");
        assert!(light.light_at(7, 8) > 0.0 && light.light_at(9, 8) > 0.0);
        assert!(light.light_at(8, 7) > 0.0 && light.light_at(8, 9) > 0.0);
        // Two separable passes reach the diagonals too, which is what rounds a
        // cross-shaped splat into a glow.
        assert!(light.light_at(7, 7) > 0.0);
        assert_eq!(light.light_at(5, 8), 0.0, "it spread further than one cell");
    }

    #[test]
    fn the_shadow_hue_walks_from_sky_blue_to_neutral_with_depth() {
        let surface = shadow_tint(0.0, 1.0);
        let deep = shadow_tint(1.0, 1.0);
        // At the surface the shadow is BLUE — the sky bouncing into it.
        assert!(
            surface[2] > surface[0] * 1.5,
            "surface shadow {surface:?} is not sky-blue"
        );
        // In the deep it is a near-neutral that eats colour instead of adding.
        let spread = deep.iter().copied().fold(0.0f32, f32::max)
            - deep.iter().copied().fold(1.0f32, f32::min);
        assert!(spread < 0.02, "deep shadow {deep:?} still carries a hue");
        assert!(deep[2] < surface[2], "the deep did not lose the sky");
    }

    #[test]
    fn night_cools_the_shadow_and_the_clamp_never_lets_it_wrap() {
        let day = shadow_tint(0.0, 1.0);
        let night = shadow_tint(0.0, 0.0);
        for c in 0..3 {
            assert!(night[c] < day[c], "channel {c} did not deepen at night");
        }
        // The deep at midnight is where the subtraction bites hardest; it has to
        // floor at black, not wrap through it.
        for ch in shadow_tint(1.0, 0.0) {
            assert!((0.0..=1.0).contains(&ch), "shadow channel left 0..1: {ch}");
        }
    }

    #[test]
    fn the_ambient_floor_falls_with_depth_so_the_deep_is_darker() {
        assert_eq!(ambient_floor(0.0), AMBIENT_FLOOR);
        assert!(ambient_floor(1.0) < ambient_floor(0.0));
        assert!(ambient_floor(1.0) > 0.0, "the deep went to pure black");
    }

    #[test]
    fn depth_is_zero_at_the_surface_and_saturates_in_the_deep() {
        let anchor = (SURFACE_ANCHOR_Y * CELL_SIZE) as f32;
        let range = DEPTH_RANGE_CELLS * CELL_SIZE as f32;
        assert_eq!(depth_at(anchor), 0.0);
        assert_eq!(depth_at(anchor - 10_000.0), 0.0, "the sky is not deep");
        assert_eq!(depth_at(anchor + range * 2.0), 1.0);
        assert!((depth_at(anchor + range * 0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn the_darkness_bake_leaves_a_fully_lit_cell_untouched() {
        let mut light = solver();
        let mut out = vec![0u8; (light.cols() * light.rows() * 4) as usize];

        light.light.fill(1.0);
        light.bake_shadow(0.0, 1.0, &mut out);
        // A multiply factor of 1 is a no-op: the scene survives intact.
        assert_eq!(&out[0..4], &[255, 255, 255, 255]);

        // ...and a fully dark cell falls to the shadow hue and no further.
        light.light.fill(0.0);
        light.bake_shadow(0.0, 1.0, &mut out);
        let floor = ambient_floor(0.0);
        let tint = shadow_tint(0.0, 1.0);
        assert_eq!(out[2], unit_byte(floor + (1.0 - floor) * tint[2]));
        assert_eq!(out[3], 255, "the bake must never touch destination alpha");
    }

    #[test]
    fn the_darkness_bake_is_monotone_in_the_light() {
        // More light must never mean a darker frame, at any depth.
        let mut light = solver();
        let mut out = vec![0u8; (light.cols() * light.rows() * 4) as usize];
        let mut previous = 0u8;
        for step in 0..=10 {
            light.light.fill(step as f32 / 10.0);
            light.bake_shadow(0.5, 0.5, &mut out);
            assert!(
                out[0] >= previous,
                "step {step} went darker: {} after {previous}",
                out[0]
            );
            previous = out[0];
        }
        assert_eq!(previous, 255, "full light is not a full pass-through");
    }

    #[test]
    fn the_flicker_phase_decorrelates_neighbouring_cells() {
        // Two adjacent pockets must not breathe together; that is the whole
        // reason the phase is hashed rather than shared.
        let a = hash_phase(100, 100);
        let b = hash_phase(101, 100);
        let c = hash_phase(100, 101);
        assert!((a - b).abs() > 0.1, "{a} and {b} are in step");
        assert!((a - c).abs() > 0.1, "{a} and {c} are in step");
        // Stable and in range, including for the negative world coords the sim
        // reaches above and left of the origin.
        for (x, y) in [(0, 0), (-1, -1), (-99_999, 5), (7, -12_345)] {
            let h = hash_phase(x, y);
            assert!(
                (0.0..std::f32::consts::TAU).contains(&h),
                "({x},{y}) -> {h}"
            );
            assert_eq!(h, hash_phase(x, y));
        }
        // And the flicker it drives stays inside the band the look depends on.
        for t in 0..200 {
            let f = flicker(t as f32 * 0.05, 3, 9);
            assert!((0.76 - 1e-6..=1.0 + 1e-6).contains(&f), "flicker was {f}");
        }
    }

    #[test]
    fn the_hot_list_is_capped_and_reports_absolute_cells() {
        let id = an_emitter(0.0);
        let mut g = CellGrid::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                g.set(x, y, id);
            }
        }
        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));
        assert_eq!(light.hot().len(), HOT_MAX, "the hot list is not capped");
        // The first entry is the first sampled cell centre, in ABSOLUTE cells.
        let half = LIGHT_DOWNSCALE / 2;
        assert_eq!(light.hot()[0], [half, half]);

        // And it follows the WORLD, not the light grid's own indexing.
        let mut scrolled = solver();
        scrolled.add_emissive(&g, noon(3, 5));
        assert_eq!(
            scrolled.hot()[0],
            [3 * LIGHT_DOWNSCALE + half, 5 * LIGHT_DOWNSCALE + half]
        );
    }

    #[test]
    fn the_emitter_scan_finds_a_single_cell_torch_the_light_grid_would_miss() {
        let id = an_emitter(0.0);
        let mut g = air(64, 64);
        // Cell (17, 19) is NOT a light-grid sample point — the grid samples the
        // cell at offset 2 in each block of 4. This is the fifteen-in-sixteen
        // case the census exists for.
        g.set(17, 19, id);

        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));
        assert_eq!(
            light.light_at(4, 4),
            0.0,
            "the downscaled sampler should have missed it"
        );

        let mut scan = EmitterScan::default();
        scan_emitters(&g, 0, 0, 64, 64, &mut scan);
        assert_eq!(scan.len(), 1);
        assert_eq!((scan.x()[0], scan.y()[0]), (17, 19));

        light.add_census_emitters(&g, scan.x(), scan.y(), noon(0, 0));
        assert!(
            light.light_at(4, 4) > 0.0,
            "the census did not light the torch"
        );
    }

    #[test]
    fn the_emitter_scan_deduplicates_a_wide_surface_and_stops_at_the_cap() {
        let id = an_emitter(0.0);
        let mut g = air(256, 256);
        for x in 0..256 {
            g.set(x, 10, id);
        }
        let mut scan = EmitterScan::default();
        scan_emitters(&g, 0, 0, 256, 256, &mut scan);
        // One entry per group of CENSUS_GROUP cells, so a lava lake cannot
        // starve a torch on the far side of the view.
        assert_eq!(scan.len(), (256 / CENSUS_GROUP) as usize);

        for y in 0..256 {
            for x in 0..256 {
                g.set(x, y, id);
            }
        }
        scan_emitters(&g, 0, 0, 256, 256, &mut scan);
        assert_eq!(scan.len(), EMIT_CENSUS_MAX, "the census cap does not hold");
    }

    #[test]
    fn the_emitter_scan_clips_to_the_loaded_window() {
        let id = an_emitter(0.0);
        let mut g = air(64, 64);
        g.set(1, 1, id);
        let mut scan = EmitterScan::default();
        // A rect entirely off the left of the window finds nothing, and does not
        // index out of bounds doing it.
        scan_emitters(&g, -500, -500, 100, 100, &mut scan);
        assert!(scan.is_empty());
        // A rect straddling the window's corner finds what is inside it.
        scan_emitters(&g, -8, -8, 32, 32, &mut scan);
        assert_eq!(scan.len(), 1);
    }

    #[test]
    fn bloom_probes_stop_at_the_budget() {
        let id = an_emitter(0.0);
        let mut g = CellGrid::new(512, 512);
        for y in 0..512 {
            for x in 0..512 {
                g.set(x, y, id);
            }
        }
        let huge = Rect2 {
            x: 0.0,
            y: 0.0,
            w: 100_000.0,
            h: 100_000.0,
        };
        let mut probes = Vec::new();
        bloom_probes(&g, huge, 0.0, &mut probes);
        assert_eq!(probes.len(), BLOOM_BUDGET);
        for p in &probes {
            assert!((0.0..=1.0).contains(&p.alpha), "probe alpha {}", p.alpha);
        }

        // A world with nothing in it produces nothing to draw.
        bloom_probes(&air(64, 64), huge, 0.0, &mut probes);
        assert!(probes.is_empty(), "empty air bloomed");
    }

    #[test]
    fn the_glow_sprite_fades_to_nothing_at_its_rim() {
        let side = (GLOW_RADIUS_PX * 2) as usize;
        for hue in GLOW_HUES {
            let image = new_glow_texture(hue);
            let data = image.data.expect("the sprite is baked with its bytes");
            let at = |x: usize, y: usize| {
                let i = (y * side + x) * 4;
                [data[i], data[i + 1], data[i + 2]]
            };
            assert!(
                at(side / 2, side / 2).iter().any(|c| *c > 0),
                "{hue:?} has no hot core at all"
            );
            // The corner is outside the radius and the rim is at it: both add
            // nothing, which is what makes the sprite a circle and not a box.
            assert_eq!(at(0, 0), [0, 0, 0], "{hue:?} has a lit corner");
            assert_eq!(at(side - 1, side / 2), [0, 0, 0], "{hue:?} has a lit rim");
            // And it falls off monotonically along a radius.
            let mut previous = u16::MAX;
            for x in (side / 2..side).step_by(4) {
                let mean: u16 = at(x, side / 2).iter().map(|c| u16::from(*c)).sum::<u16>() / 3;
                assert!(mean <= previous, "{hue:?} brightens outward at x={x}");
                previous = mean;
            }
        }
    }

    #[test]
    fn the_vignette_leaves_the_centre_untouched_and_closes_in_at_the_edges() {
        let view = View::for_screen(1000, 500);
        let (cols, rows) = vignette_size(view);
        let mut day = vec![0u8; (cols * rows * 4) as usize];
        bake_vignette(view, 0.0, 1.0, &mut day);

        let at = |buf: &[u8], x: i32, y: i32| buf[((y * cols + x) * 4) as usize];
        assert_eq!(
            at(&day, cols / 2, rows / 2),
            255,
            "the bright core must be a no-op multiply"
        );
        assert!(at(&day, 0, 0) < 255, "the corner is not darkened at all");

        // Night closes the edges in further; depth LIGHTENS them, because the
        // deep is dark enough already without its frame closing too.
        let mut night = vec![0u8; day.len()];
        bake_vignette(view, 0.0, 0.0, &mut night);
        assert!(night[0] < day[0], "night did not close the edges in");
        let mut deep = vec![0u8; day.len()];
        bake_vignette(view, 1.0, 1.0, &mut deep);
        assert!(deep[0] > day[0], "the deep double-darkened its own edges");
    }

    #[test]
    fn a_solve_over_a_whole_window_stays_in_range_everywhere() {
        // The integration shape: skylight, emitters, census and blur in the
        // order `solve` runs them, over a world with sky, ground, a cave, a
        // torch in the cave and a wall the torch has warmed. Nothing may leave
        // 0..1 and nothing may be NaN.
        let id = an_emitter(0.0);
        let mut g = world(128, 128, 40);
        for y in 44..56 {
            for x in 20..60 {
                g.set(x, y, EMPTY);
            }
        }
        g.set(33, 50, id); // a torch, off the sample lattice on purpose
        g.temp[(50 * 128 + 34) as usize] = 220; // and a hot wall beside it

        let mut scan = EmitterScan::default();
        scan_emitters(&g, 0, 0, 128, 128, &mut scan);
        let mut light = solver();
        light.solve(&g, noon(0, 0), scan.x(), scan.y());

        for ly in 0..light.rows() {
            for lx in 0..light.cols() {
                let v = light.light_at(lx, ly);
                assert!(
                    v.is_finite() && (0.0..=1.0).contains(&v),
                    "({lx},{ly}) = {v}"
                );
                for c in light.colour_at(lx, ly) {
                    assert!(c.is_finite() && (0.0..=1.0).contains(&c));
                }
            }
        }
        assert!(light.colour_dirty(), "the torch cast no colour");
        assert!(!light.hot().is_empty(), "the torch is not in the hot list");

        // And both bakes of that solve produce a whole texture.
        let mut out = vec![0u8; (light.cols() * light.rows() * 4) as usize];
        light.bake_shadow(0.3, 1.0, &mut out);
        assert!(out.iter().any(|b| *b > 0));
        light.bake_colour(&mut out);
        assert!(out.chunks_exact(4).all(|px| px[3] == 255));
    }
}

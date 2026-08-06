//! Coarse dynamic lighting, composited over the world.
//!
//! Ported from `src/render/light.ts`.
//!
//! # The model
//!
//! One light sample per [`LIGHT_DOWNSCALE`] sim cells — a [`CELL_SIZE`] world-px
//! light cell, the same lattice the art is drawn on — over the visible viewport
//! plus a one-light-cell margin, so a camera sitting between two light cells
//! still has a texel to cover each screen edge. The look is three cheap layers
//! stacked in order:
//!
//!   1. **skylight** — flood from the top; open sky is bright, and light decays
//!      as it descends through and behind solids so caves and the deep go dark;
//!   2. **emissive** — lava, fire and anything else declaring `lightEmit` smear
//!      glow into the light grid so their pockets read as lit underground;
//!   3. **composite** — the grid, blurred and upscaled, multiplied over the
//!      scene (a floor keeps caves moody, not pitch black), then an additive
//!      bloom of the emissive field, and finally a vignette and a depth wash.
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
//! **The emitter census is taken here — when it is taken at all.** In the
//! TypeScript the census was a by-product of `paintCells`, which walked every
//! visible cell anyway. The cell pass is [`crate::cellmap`]'s shader now and
//! walks nothing on the CPU, so [`scan_emitters`] pays for the walk explicitly.
//! It existed because a light grid coarser than the cell grid point-samples one
//! cell in sixteen and so misses a one-cell torch fifteen times in sixteen. At
//! [`LIGHT_DOWNSCALE`] 1 there is no gap left for it to cover and it is skipped
//! outright — see [`CENSUS_NEEDED`], which is also why the largest single cost
//! in `docs/PERF.md`'s light table is no longer paid. The pass and its scan are
//! kept, tested and correct, because they are what any downscale above 1 needs.
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
//! quantised to 8 bits per channel and then to the cell lattice.
//!
//! # The bloom is NOT a port, and the thing it replaced is worth naming
//!
//! **Do not read the bloom below as what the original did.** It is not, and the
//! difference is the point of the pass.
//!
//! The TypeScript — and this port, faithfully, until the pass described here
//! replaced it — drew bloom as **stamped sprites**. `bloomProbes` walked the
//! visible cells on a coarse stride, and every emissive cell it landed on got a
//! prebaked radial glow `drawImage`d over it at `globalAlpha`: up to 120 of
//! them, each 64 px in radius, in a buffer whose whole width is 640. One sprite
//! was **a fifth of the screen across**, its core alpha was 0.55 and its halo
//! 0.28, and they were drawn with `lighter`. Over a lava lake — which is a
//! *field* of emissive cells, so the stride found one every 8 cells in both axes
//! — dozens of those discs overlapped, and additive blending of dozens of 0.55
//! cores does exactly what the arithmetic says it does: it clips to white and
//! takes the lava, the rock around it and half the frame with it. The pass had
//! no bound at all on what it could add to a pixel; it had a bound on how many
//! sprites it could draw, which is not the same thing and does not help.
//!
//! It was also the renderer's one unmeasurable cost. `place_bloom` called
//! `Assets::get_mut` on up to 120 materials **every frame** to retint them, and
//! every one of those marks an asset modified — a re-extract and a uniform
//! upload per sprite per frame, in the render world where `docs/PERF.md` §8.5
//! says outright that criterion cannot reach it.
//!
//! What is here instead is an actual bloom: **threshold, downsample, blur,
//! composite once.**
//!
//!   - **Threshold and downsample happen together, and for free.** The source is
//!     [`crate::cellmap`]'s cell-id texture — the same `R16Uint` plane the cell
//!     pass shades the world from, so the bloom is derived from what is actually
//!     drawn rather than from a second guess at where the lights are. One texel
//!     per cell IS a 5x downsample of the buffer, because a cell is
//!     [`CELL_SIZE`] px. The threshold is a 64-entry lookup baked on the CPU by
//!     [`bake_bloom_params`]: a material's contribution is its own emitted
//!     colour, in LINEAR light, weighted by a smooth knee on its authored
//!     `lightEmit`. Gold's derived level of 1/15 falls off the bottom of that
//!     knee and contributes nothing; lava's 13/15 contributes in full.
//!   - **The blur is one gather pass** over a
//!     [`BLOOM_RADIUS_CELLS`]-radius separable Gaussian, into a target sized in
//!     cells rather than pixels — ~166x106 texels for the largest view this game
//!     allows. It runs on its own camera at [`BLOOM_CAMERA_ORDER`], which is
//!     BEFORE the world camera's `-1`, so the frame the world camera composites
//!     is this frame's, not last frame's.
//!   - **The composite is one additive quad** at [`BLOOM_Z`], sampled linearly,
//!     in exactly the slot the sprites used to occupy.
//!
//! **It cannot blow out, and that is provable rather than tuned.** The gather
//! weights are normalised on the CPU to sum to one ([`bake_bloom_params`], and
//! there is a test), and every entry of the emit table is in 0..1, so the pass's
//! output is a convex combination of values in 0..1 and the most it can add to
//! any pixel is [`BLOOM_INTENSITY`]. A lava lake is the WORST case for the old
//! pass and the *tamest* case for this one: a solid field of emitters blurs to
//! itself, so the lake reads as evenly warm and the interesting gradient is at
//! its shore, which is where a glow belongs.
//!
//! Per frame this costs three `Transform` writes and no asset mutation at all —
//! the bloom's material tint is set once in [`setup`] and never touched again.
//! It also deleted `bloom_probes`, the 333 ns CPU scan that fed the sprites.
//!
//! [`scan_emitters`] never fed the bloom either — but the claim that used to
//! stand here, that it "is unaffected and still runs", stopped being true in the
//! same milestone and is worth correcting rather than deleting. It runs only
//! under [`CENSUS_NEEDED`], which is `LIGHT_DOWNSCALE > 1`, and the downscale is
//! one — so it is not in the frame at all. Its ~10.7 µs is measured by a bench
//! that exercises a path the shipping game does not take. See `docs/PERF.md` on
//! which benchmarks describe the frame and which describe an oracle.
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
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{RenderTarget, ScalingMode};
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSampler;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, Extent3d,
    RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages,
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

use crate::cellmap::{CellMap, MATERIAL_SLOTS};
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

/// What one material contributes as a light source.
///
/// No `hue` field any more. The sprite bloom this module used to draw picked
/// between three prebaked glow images by asking which channel of `rgb` won, and
/// that enum existed only to index them. The bloom is an image pass now and
/// carries the emitter's actual colour through to the frame, so quantising a
/// cast to one of three families is a question nothing asks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    /// Emission strength, 0..1. Zero means "not a light source".
    pub level: f32,
    /// The hue the cast carries, each channel 0..1, normalised of brightness.
    pub rgb: [f32; 3],
}

impl Emitter {
    /// A material that emits nothing.
    pub const NONE: Emitter = Emitter {
        level: 0.0,
        rgb: [0.0; 3],
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

        *slot = Emitter { level, rgb };
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

/// What one SIM CELL of open air costs the flood.
///
/// LIGHT IS LOST BY PASSING THROUGH MATTER, NOT BY TRAVELLING. Open air costs
/// almost nothing — a touch of haze, so a very tall shaft still reads as deeper
/// at the bottom. The per-light-cell figure this used to be written as was 0.94
/// once: a 6% loss per 20 px of empty air, which over the 27 light rows a 1440p
/// viewport spanned compounded to 0.19, so the bottom of the screen sat at a
/// fifth of the brightness of the top with nothing in between them but sky. At
/// noon. It also made the flood depend on where the CAMERA happened to be rather
/// than on the world, so the same patch of ground brightened and dimmed as you
/// walked toward it, and it put visible vertical banding in open sky wherever
/// neighbouring columns had different surface heights.
const OPEN_DECAY_PER_CELL: f32 = 0.997_490_6;

/// What one SIM CELL of opaque rock costs the flood.
///
/// A cave is dark because there are forty cells of stone over it, which is the
/// reason it should be dark, and that is what makes a torch matter down there
/// while daylight stays flat and bright up top.
const SOLID_DECAY_PER_CELL: f32 = 0.861_173_5;

/// What one SIM CELL of open air with a WALL behind it costs the flood.
///
/// Between [`OPEN_DECAY_PER_CELL`] and [`SOLID_DECAY_PER_CELL`], and the middle
/// term is the whole of the background wall plane's contribution to lighting.
///
/// A walled tunnel is not open sky: sunlight does not pour down a mineshaft the
/// way it pours into a canyon, because a mineshaft has three sides. But it is not
/// rock either — you are standing in it, and it has to be brighter than the stone
/// around it or there was no point digging.
///
/// So a shaft that reaches daylight through a wall-free column comes out brighter
/// than one that does not, and THAT DIFFERENCE IS THE FEATURE. `back == EMPTY`
/// means the sky is genuinely open above you; anything else means you are inside
/// the world looking at its back wall.
///
/// Closer to open than to solid on purpose. A tunnel is mostly air and the wall
/// is one surface at the back of it, not forty cells of rock in the way — biasing
/// this toward `SOLID` makes every corridor a pit and undoes the reason walls
/// were drawn at all.
const WALL_DECAY_PER_CELL: f32 = 0.962_0;

/// What one LIGHT CELL of open air costs the flood.
const OPEN_DECAY: f32 = over_a_light_cell(OPEN_DECAY_PER_CELL);

/// What one LIGHT CELL of walled air costs the flood. See [`WALL_DECAY_PER_CELL`].
const WALL_DECAY: f32 = over_a_light_cell(WALL_DECAY_PER_CELL);

/// The three decays are ordered, and the ordering is the rule.
///
/// Stated at compile time because it is the entire semantic content of the middle
/// term: a walled cell must be dimmer than open sky and brighter than rock. Get
/// it backwards and a tunnel is darker than the stone it was cut out of.
const _: () = assert!(
    SOLID_DECAY_PER_CELL < WALL_DECAY_PER_CELL && WALL_DECAY_PER_CELL < OPEN_DECAY_PER_CELL,
    "a walled cell must sit between open air and solid rock"
);

/// What one LIGHT CELL of opaque rock costs the flood.
const SOLID_DECAY: f32 = over_a_light_cell(SOLID_DECAY_PER_CELL);

/// A per-sim-cell survival fraction compounded over one light cell.
///
/// # Why the decays are not written as light-cell numbers any more
///
/// The flood carries one value per light cell and multiplies by a decay at each
/// step, so the decay is a per-STEP quantity — and the step is
/// [`LIGHT_DOWNSCALE`] cells wide. The two used to be written as the compounded
/// figures directly (0.99 open, 0.55 solid) with nothing tying them to the
/// stride, so when the stride went from 4 cells to 1 the same literals became a
/// FOUR TIMES FASTER loss per world px and every cave went black. Nothing in the
/// build would have said so; the constants were correct-looking numbers about a
/// distance that had silently changed under them.
///
/// A cell is the unit occlusion is quantised in — one cell is solid or it is not
/// — so the per-cell fraction is the thing that is actually a property of the
/// world, and the per-light-cell figure is a consequence of the sampling. This
/// also states the model the coarse sampler was always implying: the one cell a
/// light cell samples stands in for all [`LIGHT_DOWNSCALE`] of them, so its
/// decay applies that many times. The old 4-cell values are reproduced exactly
/// (0.997_490_6⁴ = 0.99, 0.861_173_5⁴ = 0.55), so this is a change of
/// EXPRESSION at the old stride and a retune only because the stride moved.
///
/// A `const fn`, so the compounding happens at compile time and the flood's hot
/// loop still multiplies by a literal. `f32::powi` is not `const`; a `while`
/// loop of multiplies is, and for an exponent this small it is also the exact
/// same arithmetic.
const fn over_a_light_cell(per_cell: f32) -> f32 {
    let mut out = 1.0;
    let mut n = 0;
    while n < LIGHT_DOWNSCALE {
        out *= per_cell;
        n += 1;
    }
    out
}

/// Per-CELL decay used to seed a column whose top starts below the surface.
///
/// The streaming window's top may be far below the ground line. Seeding each
/// column's carry from how deep its top sample sits below that column's own
/// surface height keeps brightness a continuous function of absolute world
/// coordinates, so it stays seam-free as the window scrolls.
///
/// **This is the one decay that takes no [`LIGHT_DOWNSCALE`] correction, and it
/// looks exactly like the two that do.** It is raised to a count of CELLS —
/// `below_surface` is the difference of two cell rows — not to a count of light
/// cells, because a surface height is a cell row. Rescaling it with the stride
/// would be four applications of a correction the exponent already carries, and
/// would light every deep window four times too brightly.
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

/// How far a splat's four arms reach, in SIM CELLS.
///
/// # Reach is a world distance, and it used to be written as a grid index
///
/// The splat is a cross: the centre, four arms, and — for a strong emitter — a
/// far ring outside them. All three offsets were literal light-cell steps of 1
/// and 2, which at four cells per sample meant 20 and 40 world px. At one cell
/// per sample the same literals mean 5 and 10, so every light in the game would
/// have kept its shape and lost three quarters of its reach: a lava lake stops
/// lighting the cave it is in and becomes a bright sticker on the floor. That is
/// what the first capture at this downscale actually looked like.
///
/// So the reaches are stated in cells and divided by the stride. Four cells is
/// the 20 px the arms have always covered, and it is about the distance at which
/// a glow still reads as coming from the block that cast it.
const SPLAT_REACH_CELLS: i32 = 4;

/// How far a strong emitter's far ring reaches, in SIM CELLS.
///
/// Twice [`SPLAT_REACH_CELLS`], as it has always been — the ring is the second
/// step out, not a different mechanism.
const FAR_REACH_CELLS: i32 = 8;

/// How far the blur that turns splats into glow spreads, in SIM CELLS.
///
/// Matched to [`SPLAT_REACH_CELLS`], and that is not a coincidence to be tidied
/// away: the blur's job is to fill the gap between a splat's centre and its
/// arms, so if it reaches less far than an arm does the cross stops being a
/// glow and starts being five dots with holes between them.
///
/// It is also the one thing here that must NOT be widened to hide the light
/// grid's lattice any more. At one sample per cell the lattice IS the art's, and
/// blurring it away is exactly the smooth wash this whole change exists to
/// remove.
const BLUR_REACH_CELLS: i32 = 4;

/// [`SPLAT_REACH_CELLS`] in light cells.
const SPLAT_REACH: i32 = in_light_cells(SPLAT_REACH_CELLS);

/// [`FAR_REACH_CELLS`] in light cells.
const FAR_REACH: i32 = in_light_cells(FAR_REACH_CELLS);

/// [`BLUR_REACH_CELLS`] in light cells — the blur's radius in taps per side.
const BLUR_REACH: i32 = in_light_cells(BLUR_REACH_CELLS);

/// How soft the light's own edges are against the art's, 0..1.
///
/// **The dial to reach for when the lighting reads wrong at the pixel level**,
/// and the one this module did not have. It is a fraction of a CELL: the light
/// steps from one cell's value to the next over a ramp this wide, and is flat
/// across the rest of the cell. 0 is a hard block edge, exactly a nearest fetch;
/// 1 is a full bilinear upscale, which at one texel per cell is a 5 px ramp.
/// [`LIGHT_WGSL`] is where it is applied and how.
///
/// # Why the answer is neither end
///
/// At 0 the light is drawn in exactly the art's blocks, and that is a real
/// problem rather than the goal: a lit cell and a differently-coloured BLOCK
/// become the same thing on screen, so the eye reads a bright patch on a wall as
/// masonry rather than as light falling on it. Light that is quantised exactly
/// like matter stops looking like light.
///
/// At 1 the light is a smooth field over the whole frame. That is what this
/// module used to do — and at four cells per sample it was a 20 px smear, the
/// airbrushed wash the underground was reported as having.
///
/// So: a ramp NARROWER than a cell. The light shares the art's lattice, which is
/// what stops it looking pasted on, and it crosses between cells over a fraction
/// of one, which is what keeps it distinguishable from the blocks it falls on.
/// Below about 0.2 the ramp is under a pixel at this cell size and the dial
/// stops doing anything the frame buffer can hold.
const LIGHT_SOFTNESS: f32 = 0.5;

/// [`LIGHT_SOFTNESS`], or a full bilinear upscale if the grid is coarser than
/// the art.
///
/// Snapping a sample to its texel centre reads as "the light belongs to this
/// block" only when a texel IS a block. At any [`LIGHT_DOWNSCALE`] above 1 the
/// same snap would quantise the light to a lattice nothing on screen is drawn
/// on — a hard 20 px grid at the old downscale of 4 — so the dial turns itself
/// off and the grid goes back to interpolating, which is the least-bad thing a
/// coarse grid can do. Derived rather than left to whoever changes the stride.
const LIGHT_SNAP: f32 = if LIGHT_DOWNSCALE == 1 {
    LIGHT_SOFTNESS
} else {
    1.0
};

const _: () = assert!(
    LIGHT_SOFTNESS >= 0.0 && LIGHT_SOFTNESS <= 1.0,
    "LIGHT_SOFTNESS is a fraction of a texel. Above 1 the shader would push a \
     sample past its own texel's edge and fetch a neighbour's neighbour, which \
     is a light field shifted off the world, not a softer one"
);

/// How far back one box pass of the blur looks, and how far forward.
///
/// A box of `BLUR_REACH + 1` samples. Two of them, offset against each other, is
/// the kernel — see [`blur_one`] for why the split is uneven when the reach is
/// odd. Split rather than one radius because an even-width box has no centre
/// sample and the pair only lands back on one if the second leans the other way.
const BLUR_BACK: usize = (BLUR_REACH / 2) as usize;

/// See [`BLUR_BACK`].
const BLUR_FWD: usize = (BLUR_REACH - BLUR_REACH / 2) as usize;

/// A distance in sim cells as a whole number of light cells.
///
/// Rounded UP, so a reach authored in cells can never collapse to zero and
/// silently delete the ring or the arm it describes; the const assert below is
/// what catches a stride that does not divide it cleanly, which would be a
/// reach that quietly grew instead.
const fn in_light_cells(cells: i32) -> i32 {
    let n = cells / LIGHT_DOWNSCALE;
    if n < 1 { 1 } else { n }
}

const _: () = assert!(
    SPLAT_REACH * LIGHT_DOWNSCALE == SPLAT_REACH_CELLS
        && FAR_REACH * LIGHT_DOWNSCALE == FAR_REACH_CELLS
        && BLUR_REACH * LIGHT_DOWNSCALE == BLUR_REACH_CELLS,
    "a reach converted to light cells no longer converts back to the distance \
     it was authored as — LIGHT_DOWNSCALE has to divide all three, or the grid \
     cannot express the reach they were tuned at"
);

/// What the four arms of a scalar splat get, relative to its centre.
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
///
/// Axial at [`FAR_REACH`] and diagonal at [`SPLAT_REACH`], which is what the
/// original's `(±2, 0)` and `(±1, ±1)` were saying: the diagonals sit one step
/// out and the axes two, so the ring is a rough circle rather than a square.
const FAR_RING: [(i32, i32); 8] = [
    (-FAR_REACH, 0),
    (FAR_REACH, 0),
    (0, -FAR_REACH),
    (0, FAR_REACH),
    (-SPLAT_REACH, -SPLAT_REACH),
    (SPLAT_REACH, -SPLAT_REACH),
    (-SPLAT_REACH, SPLAT_REACH),
    (SPLAT_REACH, SPLAT_REACH),
];

/// The four the census pass uses.
///
/// Faithful to the original, which listed eight offsets in `addEmissive` and
/// four in `addCensusEmitters`. Almost certainly an oversight there rather than
/// a decision — but a census emitter is by definition a SMALL source, a torch
/// and not a magma chamber, and the narrower ring is the better answer for one.
/// Kept, and written down, rather than silently unified.
const FAR_RING_AXIAL: [(i32, i32); 4] = [
    (-FAR_REACH, 0),
    (FAR_REACH, 0),
    (0, -FAR_REACH),
    (0, FAR_REACH),
];

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
/// starve a torch on the far side of the view. The light grid's own downscale:
/// two emitters closer together than this land in the same light cell and would
/// be deduplicated by the splat pass regardless.
const CENSUS_GROUP: i32 = LIGHT_DOWNSCALE;

/// Whether the census is worth taking at all at this [`LIGHT_DOWNSCALE`].
///
/// The census exists to find the emitters the downscaled sampler steps OVER. At
/// a downscale of 1 it steps over nothing: [`LightGrid::add_emissive`] visits
/// every cell under the grid, which is precisely the rect [`scan_emitters`]
/// walks, and it keys off a level that is a superset of the one the scan keys
/// off. Every census entry would therefore be a SECOND splat of a source already
/// splatted — and, since the scan stops at [`EMIT_CENSUS_MAX`], a second splat
/// of only the first 512 of them, which draws a hard seam across a lava lake at
/// the point the cap bites. Not a glow.
///
/// Skipping it also deletes `scan_emitters` from the frame, which `docs/PERF.md`
/// §8.3 names as the largest single cost in the light pass — the one place it
/// says to look first if the light ever has to shrink. Going to one sample per
/// cell is what makes the coarse grid's compensating scan redundant, so the
/// resolution increase buys the scan's whole cost back.
const CENSUS_NEEDED: bool = LIGHT_DOWNSCALE > 1;

// --- Bloom -------------------------------------------------------------------

/// Gather radius of the bloom blur, in CELLS.
///
/// A cell is [`CELL_SIZE`] px, so three cells is a halo that reaches 15 world px
/// past the lit surface — against the 64 px RADIUS of the sprite it replaced,
/// which was a fifth of the buffer's width per stamp. Fifteen px is roughly the
/// width of a player, which is the scale at which a glow still reads as
/// belonging to the thing that cast it rather than as fog over the frame.
///
/// **Hard-coded a second time in [`BLOOM_WGSL`]** as the loop bounds and the
/// tap-array length. WGSL has no way to import a Rust constant, so the const
/// assert below is the thing that stops the two drifting.
const BLOOM_RADIUS_CELLS: i32 = 3;

/// Materials the bloom's emit table has room for.
///
/// [`crate::cellmap::MATERIAL_SLOTS`] and not a second 64 of this module's own:
/// the table below is indexed by a texel of `cellmap`'s id texture, so the two
/// shaders must agree on the padding or a block added to `content/` would light
/// one pass and not the other.
const BLOOM_EMIT_SLOTS: usize = MATERIAL_SLOTS;

/// Emitted level below which a material contributes nothing to the bloom.
///
/// THE THRESHOLD, and it is on the content's declared `lightEmit` rather than on
/// the drawn pixel's luminance. That is deliberate and it is the better signal:
/// sunlit sand is one of the brightest things in the frame and must not bloom,
/// so a luminance threshold would have to sit above sand — at which point it is
/// above everything except lava anyway, and it would still bloom a white UI
/// panel. `lightEmit` says "this block is a light source" in the one place that
/// actually knows. Gold's derived 1/15 and the level-3 emitters fall below this
/// and are glints, not lamps.
const BLOOM_LEVEL_KNEE_LO: f32 = 0.25;

/// Emitted level at which a material contributes its colour in full.
///
/// The knee between it and [`BLOOM_LEVEL_KNEE_LO`] is smooth rather than a step
/// so that a level moving by one authored point cannot pop a whole cavern's glow
/// into existence. Against the content set as it stands: mushroom cap (6/15)
/// lands at 0.22, crystal (7/15) at 0.40, brazier (10/15) at 0.93, and lava,
/// fire and the torch (13..15) are all at 1.
const BLOOM_LEVEL_KNEE_HI: f32 = 0.75;

/// Standard deviation of the bloom's gather kernel, in cells.
///
/// Half [`BLOOM_RADIUS_CELLS`], which is the usual place to truncate a Gaussian:
/// the tap at the rim is `exp(-2)` of the centre, so the kernel is ~98% of the
/// untruncated one and the seam at the edge of the gather is invisible.
const BLOOM_SIGMA_CELLS: f32 = 1.5;

/// How much of the blurred emissive field reaches the frame.
///
/// THIS NUMBER IS A HARD CEILING, not a starting point for taste. The gather
/// weights sum to one and every emit-table entry is in 0..1, so the pass's
/// output is a convex combination of values in 0..1: this is the most the bloom
/// can add to any channel of any pixel, under any world, ever. Compare the pass
/// it replaced, whose 120 sprites at 0.55 core alpha could add 66 to a channel
/// and routinely added enough to clip.
///
/// It is low, and it does not need to be high, because the non-linearity does
/// the work: an additive amount in LINEAR light is worth far more over dark rock
/// than over daylit ground, where it lands in a value already near 1. A bloom
/// that shows up exactly where it should and nowhere else is what that buys, and
/// it is why this is tuned in linear rather than against the 0..255 bytes the
/// original was written in.
///
/// Judge this underground, never at the surface. The daylit case is exactly the
/// one where the value is invisible, so tuning against it will always say the
/// number is too small.
///
/// It was briefly halved to 0.15 in response to a report that the glow underground
/// was too strong and too smooth. That was the wrong knob, and the measurement is
/// worth recording so nobody reaches for it again: with this set to **0.0** the
/// reported haze was unchanged. What produced it was the COLOUR grid, which at
/// the time was one texel per four cells and interpolated across all twenty of
/// their pixels — not this pass. [`LIGHT_DOWNSCALE`] is 1 now and
/// [`LIGHT_SOFTNESS`] bounds the interpolation to a fraction of ONE cell, which
/// is what actually answered that report; [`COLOUR_GAIN`] is the strength dial
/// if the coloured cast ever needs one again.
const BLOOM_INTENSITY: f32 = 0.3;

/// Where the bloom's gather pass sits in camera order.
///
/// BEFORE `crate::lowres`' world camera at `-1`, so the texture the composite
/// quad samples was written by this frame's gather and not by last frame's. That
/// ordering is the whole reason the bloom can be a same-frame image pass without
/// a custom render-graph node: two cameras with different targets and different
/// orders are two render passes, and Bevy runs them in the order given.
const BLOOM_CAMERA_ORDER: isize = -2;

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
///
/// # This was 0.6, and 0.6 was the TypeScript's number in the wrong space
///
/// The original filled the frame with the biome's ambient colour at this alpha
/// in **sRGB**. This composites it additively in **linear**, and the two are not
/// the same operation anywhere except white — on a dark pixel, which is the
/// entire underground, linear addition lifts far harder.
///
/// Tundra authors `ambient = [0.04, 0.07, 0.12]`. Carried across at 0.6 that is
/// a linear `+0.072` on blue, which encodes to **76/255** added to a black cave
/// pixel. The original's sRGB fill added `0.6 * 0.12 = 18.4/255`. So the port was
/// **4.1x** too bright, on the blue channel, everywhere it was darkest.
///
/// You could see it: 900 px down, every rock face read the same blue-violet and
/// local albedo survived only inside the lava's own falloff. It looked like a
/// blue-lit cave rather than a dark one, and `HANDOFF.md` §8.4 had suspected the
/// cause without anyone doing the arithmetic.
///
/// 0.05 is the value that reproduces the original: it encodes to 17.9/255 against
/// its 18.4. Checked as a picture too, not just on paper — `tests/lit_scene.rs`
/// at 0.6, 0.15 and 0.05. At 0.05 the stone is stone again and the ore veins are
/// visible; 0.15 still hazes.
///
/// Judge any change to this UNDERGROUND, never at the surface, where the sky
/// swamps it. That is the same instruction [`BLOOM_INTENSITY`] carries, for the
/// same reason, after the same mistake was made with it.
const BIOME_AMBIENT_ALPHA: f32 = 0.05;

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
    /// One running column sum per grid column, for the blur's vertical pass.
    acc: Vec<f32>,

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
    /// The world seed [`LightGrid::heights`] and [`LightGrid::noise`] describe.
    /// See [`LightGrid::follow_seed`].
    seed: u32,
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
            acc: vec![0.0; lw.max(1) as usize],
            lr: vec![0.0; n],
            lg: vec![0.0; n],
            lb: vec![0.0; n],
            colour_dirty: false,
            seen: vec![false; n],
            hot: Vec::with_capacity(HOT_MAX),
            heights: Heightmap::new(),
            noise: world_noise(seed),
            seed,
        }
    }

    /// Rebuild the climate fields if the world underneath has changed.
    ///
    /// # Why this exists when nothing can currently call it usefully
    ///
    /// This grid keeps its own `Heightmap` and `Noise` so the skylight flood can
    /// seed a column that starts underground. `crate::ambience` keeps a second
    /// copy for its own reasons, and has always guarded itself this way; this one
    /// did not. It was built from the compile-time `SEED` at both production
    /// sites and never read `SimWorld::seed`.
    ///
    /// Nothing is wrong on screen today, because `build_world` is only ever
    /// called with that same constant. But it takes a seed — the signature is an
    /// open invitation — and the failure mode on the day someone adds a `--seed`
    /// flag or a new-world menu is silent: the light solver would flood sky down
    /// to one surface line while the terrain sat at another, and no test would
    /// notice. That is precisely the shape of the three bugs in `HANDOFF.md`
    /// §7.1, all of which degraded to plausible output.
    ///
    /// Only the seeded fields are rebuilt; the buffers do not depend on the seed.
    ///
    /// # The heightmap is REPLACED, and that is not belt-and-braces
    ///
    /// `Heightmap` invalidates its 4096-slot memo by comparing the ADDRESS of the
    /// `Noise` it is handed against the one its live entries belong to
    /// (`retire_if_new_seed`) — an O(1) trick that is exactly right for the
    /// worldgen path, where a new world means a new `ChunkGen` and so a new
    /// `Noise` at a new address.
    ///
    /// It is a trap here. `self.noise = world_noise(seed)` overwrites a field in
    /// place, so the address does not move, so the memo would go on serving the
    /// previous world's surface rows forever. Handing the new noise to a fresh
    /// `Heightmap` is what makes the invalidation fire.
    pub fn follow_seed(&mut self, seed: u32) {
        if self.seed != seed {
            self.seed = seed;
            self.noise = world_noise(seed);
            self.heights = Heightmap::new();
        }
    }

    /// The world seed the climate fields were built from.
    #[inline]
    pub const fn seed(&self) -> u32 {
        self.seed
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
                let loaded = in_col && gy >= 0 && gy < rows;
                let at = if loaded { (gy * cols + gx) as usize } else { 0 };
                let solid = loaded && MAT_COLLIDE[grid.material[at] as usize] == 1;
                // Open air with a wall behind it. Only asked where the front is
                // not solid, because rock in front of a wall is just rock.
                let walled = loaded && !solid && grid.back[at] != EMPTY;

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
                carry *= if solid {
                    SOLID_DECAY
                } else if walled {
                    WALL_DECAY
                } else {
                    OPEN_DECAY
                };
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
    /// A downscaled sampling loop catches BULK emitters — a lava lake fills
    /// every block it is sampled in — but by construction cannot see a one-cell
    /// torch, because it only looks at one cell in `LIGHT_DOWNSCALE` squared.
    /// This pass covers that gap. At a downscale of 1 there is no gap and no
    /// census is taken at all; see [`CENSUS_NEEDED`] for why feeding this one
    /// anyway would double every emitter rather than add nothing.
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
        blur_one(&mut self.light, &mut self.scratch, &mut self.acc, lw, lh);
        if self.colour_dirty {
            blur_one(&mut self.lr, &mut self.scratch, &mut self.acc, lw, lh);
            blur_one(&mut self.lg, &mut self.scratch, &mut self.acc, lw, lh);
            blur_one(&mut self.lb, &mut self.scratch, &mut self.acc, lw, lh);
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
/// margin each side.
///
/// The margin is what covers the camera sitting BETWEEN two light cells: the
/// grid's origin is floored to the lattice, so the view can hang up to one light
/// cell off each far edge, and without the spare column there the composite quad
/// would stop short of the screen. It also gives the upscale a value to
/// interpolate toward rather than clamping at the frame edge — a second reason
/// that used to be the stated one, and that now covers [`LIGHT_SOFTNESS`] of a
/// cell rather than a whole 20 px texel.
pub fn grid_size(view: View) -> (i32, i32) {
    let stride = light_stride_px();
    let ceil = |px: i32| (px.max(0) + stride - 1) / stride + 2;
    (ceil(view.w), ceil(view.h))
}

/// World px covered by one light cell on each axis.
pub const fn light_stride_px() -> i32 {
    CELL_SIZE * LIGHT_DOWNSCALE
}

/// Additive centre plus four arms at [`SPLAT_REACH`], clamped to 1.
///
/// Five writes whatever the stride: the arms move further out in light cells as
/// the grid gets finer, but there are still four of them. What fills the gap
/// between the centre and an arm is [`blur_one`], whose radius is the same
/// distance — that pairing is why a cross of five deltas reads as a glow and not
/// as five dots, and it is why the two reaches have to move together.
fn splat(buf: &mut [f32], lw: i32, lh: i32, lx: i32, ly: i32, core: f32, edge: f32) {
    add(buf, lw, lh, lx, ly, core);
    add(buf, lw, lh, lx - SPLAT_REACH, ly, edge);
    add(buf, lw, lh, lx + SPLAT_REACH, ly, edge);
    add(buf, lw, lh, lx, ly - SPLAT_REACH, edge);
    add(buf, lw, lh, lx, ly + SPLAT_REACH, edge);
}

/// Clamped additive write into a light cell. Out of range is a no-op.
fn add(buf: &mut [f32], lw: i32, lh: i32, x: i32, y: i32, v: f32) {
    if x < 0 || y < 0 || x >= lw || y >= lh {
        return;
    }
    let i = (y * lw + x) as usize;
    buf[i] = (buf[i] + v).min(1.0);
}

/// A triangle blur of radius [`BLUR_REACH`], edges clamped to themselves.
///
/// # Four box passes, and why not nine taps
///
/// A triangle kernel IS a box convolved with a box, so the whole blur is four
/// runs of a sliding window: two along the rows and two down the columns. At
/// radius 1 that is `[1, 2, 1] / 4` exactly, the kernel this pass has always
/// run; at radius 4 it is `[1..5..1] / 25`. The uneven `BLUR_BACK`/`BLUR_FWD`
/// split is what keeps the pair CENTRED when the box has an even width — the
/// first pass leans forward by half a sample and the second leans back by the
/// same half, and the two cancel. Getting that wrong shifts the entire light
/// field half a cell against the art, which is the one error this arrangement
/// can make and the reason the offsets are named rather than inlined.
///
/// The obvious implementation is `2 * BLUR_REACH + 1` taps per cell per axis,
/// and it was written that way first. It costs the reach: nine taps at radius 4
/// measured **171 µs** for one frame's four grids, 80% of the entire light
/// solve, and it would have got worse the moment anyone widened the glow. A
/// sliding window is O(1) in the radius — three float ops per cell per pass
/// whatever the reach — so the look and the cost are no longer the same dial.
///
/// `scratch` is the caller's ping-pong buffer and `acc` its column accumulator,
/// one entry per grid column. Both come in dirty and go out dirty, which is the
/// whole point of them being owned by the grid rather than allocated per pass:
/// the vertical window has to carry a running sum per column, and walking the
/// grid column by column to avoid that would read every cache line `h` times.
///
/// # This is now the ORACLE, not the shipping path
///
/// The blur runs on the GPU — `lightblur.wgsl`, two passes at
/// [`BLUR_CAMERA_ORDER`] — and [`LightGrid::solve`] no longer calls this. It is
/// kept, exported and still tested for exactly the reason `crate::cells`'
/// `paint_cells` is kept beside `cells.wgsl`: a shader nothing can diff against
/// is a shader nobody can change safely.
/// `tests/light_blur_matches_cpu.rs` compiles the shipping WGSL on a headless
/// adapter and compares it to this function, texel for texel.
pub fn blur_one(a: &mut [f32], scratch: &mut [f32], acc: &mut [f32], lw: i32, lh: i32) {
    let (w, h) = (lw as usize, lh as usize);
    box_rows(a, scratch, w, h, BLUR_BACK, BLUR_FWD);
    box_rows(scratch, a, w, h, BLUR_FWD, BLUR_BACK);
    box_cols(a, scratch, acc, w, h, BLUR_BACK, BLUR_FWD);
    box_cols(scratch, a, acc, w, h, BLUR_FWD, BLUR_BACK);
}

/// The blur's window as `(back, fwd)`, for whatever has to restate it.
///
/// Two things do: the uniform `lightblur.wgsl` reads its loop bounds from, and
/// the harness that diffs that shader against [`blur_one`]. Both get the numbers
/// from here rather than from a literal, so re-tuning [`BLUR_REACH_CELLS`] moves
/// the CPU pass, the GPU pass and the test together or moves none of them.
pub const fn blur_window() -> (u32, u32) {
    (BLUR_BACK as u32, BLUR_FWD as u32)
}

/// One box pass along the rows, window `[x - back, x + fwd]`, edges clamped.
///
/// The output is clamped to 0..1 as well as the window. A running sum adds and
/// subtracts the same values back out over a whole row, so it drifts by a few
/// ulps where a fresh sum of taps would not — enough to leave a `-1e-8` in a
/// grid every other pass in this module is entitled to assume is a fraction.
/// The clamp is two ops per cell and buys back an invariant.
fn box_rows(src: &[f32], dst: &mut [f32], w: usize, h: usize, back: usize, fwd: usize) {
    let inv = 1.0 / (back + fwd + 1) as f32;
    let last = w as isize - 1;
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        let at = |x: isize| row[x.clamp(0, last) as usize];

        let mut sum = 0.0;
        for d in 0..=(back + fwd) {
            sum += at(d as isize - back as isize);
        }
        for (x, out) in dst[y * w..y * w + w].iter_mut().enumerate() {
            *out = (sum * inv).clamp(0.0, 1.0);
            sum += at((x + fwd + 1) as isize) - at(x as isize - back as isize);
        }
    }
}

/// One box pass down the columns, window `[y - back, y + fwd]`, edges clamped.
///
/// Row-major throughout: `acc` holds one running sum per column, so the walk is
/// still linear over both buffers and the vertical pass costs what the
/// horizontal one does.
fn box_cols(
    src: &[f32],
    dst: &mut [f32],
    acc: &mut [f32],
    w: usize,
    h: usize,
    back: usize,
    fwd: usize,
) {
    let inv = 1.0 / (back + fwd + 1) as f32;
    let last = h as isize - 1;
    let row_at = |y: isize| (y.clamp(0, last) as usize) * w;

    acc[..w].fill(0.0);
    for d in 0..=(back + fwd) {
        let r = row_at(d as isize - back as isize);
        for (a, s) in acc[..w].iter_mut().zip(&src[r..r + w]) {
            *a += *s;
        }
    }
    for y in 0..h {
        let add = row_at((y + fwd + 1) as isize);
        let sub = row_at(y as isize - back as isize);
        let out = &mut dst[y * w..y * w + w];
        for (((o, a), plus), minus) in out
            .iter_mut()
            .zip(acc[..w].iter_mut())
            .zip(&src[add..add + w])
            .zip(&src[sub..sub + w])
        {
            *o = (*a * inv).clamp(0.0, 1.0);
            *a += *plus - *minus;
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

/// Everything the bloom's gather pass needs, and all of it is baked once.
///
/// The whole threshold-and-weight decision is a CPU function of the content
/// build ([`bake_bloom_params`]), for the same reason [`LightGrid::bake_shadow`]
/// is: a lookup table can be asserted about by a test on this machine, and an
/// expression buried in a fragment shader cannot. `BLOOM_WGSL` reads this and
/// does nothing but the sum.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct BloomParams {
    /// Per material: the LINEAR-light colour it contributes to the bloom,
    /// already multiplied by its threshold weight. `w` is unused padding —
    /// a WGSL uniform array pads a scalar to 16 bytes regardless, so a `vec4`
    /// is what an `array<f32>` would have cost anyway.
    pub emit: [Vec4; BLOOM_EMIT_SLOTS],
    /// The separable Gaussian's taps, `x` at the centre out to `w` at the rim.
    ///
    /// One `vec4` because [`BLOOM_RADIUS_CELLS`] is 3 and 3 + 1 is 4. Normalised
    /// so the 1D sum is one, which makes the 2D product kernel sum to one too —
    /// that is the identity the no-blowout guarantee rests on.
    pub taps: Vec4,
}

const _: () = assert!(
    BLOOM_RADIUS_CELLS == 3,
    "BLOOM_WGSL hard-codes a radius of 3 in its loop bounds and packs its four \
     taps into one vec4 — change both together or not at all"
);

const _: () = assert!(
    MAT_COUNT <= BLOOM_EMIT_SLOTS,
    "more materials than bloom emit slots — widen cellmap::MATERIAL_SLOTS, \
     BLOOM_WGSL and cellmap.wgsl together"
);

/// Bake the emit table and the gather weights the bloom pass runs on.
///
/// A pure function of the content build and of the four constants above, called
/// once in [`setup`] and never again. Two decisions live here rather than in the
/// shader:
///
///   - **The threshold.** A smooth knee on the authored `lightEmit`, so the
///     table entry for a material that is not a light source is exactly zero and
///     the gather adds literally nothing for it.
///   - **The transfer function.** [`Emitter::rgb`] is derived from the AUTHORED
///     sRGB bytes, because that is the space `cells::build_material_shades`
///     clamps in and the space a human picked `(235, 110, 35)` in. The bloom is
///     composited by a fixed-function ADD, and a GPU adds in linear light — so
///     the conversion has to happen, and here is the only place it can happen
///     once instead of per fragment. Skipping it would make every glow
///     noticeably more washed-out and less saturated than the block casting it,
///     which is the specific mistake this module's header already records the
///     multiply pass having to live with.
pub fn bake_bloom_params() -> BloomParams {
    let mut emit = [Vec4::ZERO; BLOOM_EMIT_SLOTS];
    for (id, slot) in emit.iter_mut().enumerate().take(MAT_COUNT) {
        let e = emitter(id as CellId);
        let weight = smooth_knee(e.level, BLOOM_LEVEL_KNEE_LO, BLOOM_LEVEL_KNEE_HI);
        if weight <= 0.0 {
            continue;
        }
        let linear = Color::srgb(e.rgb[0], e.rgb[1], e.rgb[2]).to_linear();
        *slot = Vec4::new(linear.red, linear.green, linear.blue, 0.0) * weight;
    }

    // Normalise the 1D taps to sum to one over the FULL kernel — the centre once
    // and every other ring twice, because `w[|d|]` is read on both sides.
    let mut taps = [0.0f32; BLOOM_RADIUS_CELLS as usize + 1];
    let denom = 2.0 * BLOOM_SIGMA_CELLS * BLOOM_SIGMA_CELLS;
    let mut total = 0.0;
    for (d, tap) in taps.iter_mut().enumerate() {
        let x = d as f32;
        *tap = (-(x * x) / denom).exp();
        total += *tap * if d == 0 { 1.0 } else { 2.0 };
    }
    for tap in &mut taps {
        *tap /= total;
    }

    BloomParams {
        emit,
        taps: Vec4::from_array(taps),
    }
}

/// Hermite ramp from 0 at `lo` to 1 at `hi`, flat outside.
///
/// `smoothstep` by another name. Written out rather than reached for because
/// `f32` has none and the two-line version is clearer than the clamp-and-fma
/// dance that would import one.
#[inline]
fn smooth_knee(v: f32, lo: f32, hi: f32) -> f32 {
    let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Bloom target size for a view, in texels — one texel per CELL.
///
/// Sized in cells and not in pixels: the cell grid is the resolution the
/// emissive field actually has, so this IS the downsample and it costs nothing
/// to take. [`BLOOM_RADIUS_CELLS`] of margin on every side so an emitter just
/// off the left of the screen still casts its halo onto the left of the screen —
/// without it, a lava lake would visibly switch its glow on as its first cell
/// crossed the edge.
pub fn bloom_size(view: View) -> (i32, i32) {
    let ceil = |px: i32| (px.max(0) + CELL_SIZE - 1) / CELL_SIZE + 2 * BLOOM_RADIUS_CELLS;
    (ceil(view.w), ceil(view.h))
}

/// Where the bloom's gather covers, in world px, sim convention.
///
/// Snapped to whole CELLS, which is the point of the function: one bloom texel
/// is one cell, and if the rect drifted by a fraction of a cell then every texel
/// would straddle two cells and the whole glow would crawl and shimmer as the
/// camera moved. `view` is the visible rect; the result is it, grown by the
/// gather margin and aligned down.
pub fn bloom_rect(view: Rect2, texels: (i32, i32)) -> Rect2 {
    let cell = CELL_SIZE as f32;
    let x0 = (view.x / cell).floor() as i32 - BLOOM_RADIUS_CELLS;
    let y0 = (view.y / cell).floor() as i32 - BLOOM_RADIUS_CELLS;
    Rect2 {
        x: (x0 * CELL_SIZE) as f32,
        y: (y0 * CELL_SIZE) as f32,
        w: (texels.0 * CELL_SIZE) as f32,
        h: (texels.1 * CELL_SIZE) as f32,
    }
}

// --- Vignette ----------------------------------------------------------------

/// How far `depth` or `day` must move before the vignette is baked again.
///
/// The bake is 1 530 radial samples — the largest of the three, and 5.8% of the
/// whole light stack — recomputed every frame from inputs that barely move.
/// `day` advances by 1/36 000 per frame at 120 Hz, and `depth` only as fast as
/// the player descends.
///
/// `the_vignette_cache_never_skips_a_visible_change` sweeps the input space and
/// requires two bakes this far apart to differ by at most one byte in any
/// channel — the smallest difference an `Rgba8Unorm` target can represent, so a
/// skipped frame cannot be a frame anyone could see.
///
/// **This is deliberately an order of magnitude under the measured limit.**
/// Bisecting the constant against that test: 0.02 still passes, 0.05 moves two
/// bytes. So the honest bound is somewhere in 0.02..0.05, and 0.002 buys nothing
/// in exchange for the caution — at 120 Hz `day` alone forces a re-bake every 72
/// frames here against every 720 at 0.02, which is the difference between
/// skipping 98.6% of the bakes and skipping 99.9%. Both round to "all of them".
///
/// Being 10x under a limit that was measured rather than assumed is the cheap
/// side of this trade. Do not tighten it for performance; there is none left to
/// win.
///
/// Skipping the bake also skips the `Assets::get_mut` that would schedule the
/// upload, which is the larger half of the saving and the same reason
/// `cellmap::upload_dirty_chunks` bails before touching its asset.
const VIGNETTE_REBAKE_EPS: f32 = 0.002;

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
/// almost no logic — all of the arithmetic is baked into the textures by
/// [`LightGrid::bake_shadow`], [`LightGrid::bake_colour`] and [`bake_vignette`],
/// which is what makes those decisions testable on the CPU. `cells.wgsl` earns
/// its own file by being a real shading model; a passthrough and a uv nudge does
/// not.
///
/// # The uv nudge IS the softness dial
///
/// `softness` is the one thing this shader decides, and it decides it by moving
/// the SAMPLE POINT rather than by changing the filter. Every one of these
/// textures is sampled linearly. Pulling the sample toward its own texel's
/// centre by `1 - softness` narrows the band over which the filter is allowed to
/// interpolate: at 0 every sample lands exactly on a texel centre and the result
/// is bit-for-bit a nearest fetch; at 1 nothing moves and it is a plain bilinear
/// upscale; in between, the light steps from cell to cell over a ramp
/// `softness` of a cell wide and is flat across the rest.
///
/// That is a continuous dial between "the light is drawn in the same blocks as
/// the art" and "the light is a smooth field over it", and it exists because
/// both extremes are wrong in the same picture: a fully snapped light is
/// indistinguishable from a differently-coloured BLOCK, and a fully smooth one
/// is the airbrushed wash this module spent a change removing. See
/// [`LIGHT_SOFTNESS`].
const LIGHT_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var light_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var light_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> tint: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> softness: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(light_texture));
    let texel = mesh.uv * dim;
    let centre = floor(texel) + vec2<f32>(0.5);
    let uv = mix(centre, texel, softness.x) / dim;
    return textureSample(light_texture, light_sampler, uv) * tint;
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
            /// The grid or ramp this quad samples.
            ///
            /// Every one of them carries a LINEAR sampler — the game's one
            /// `default_nearest` override, and no longer for the reason it was
            /// first taken. How hard-edged the result is belongs to `softness`
            /// below, which needs a filter underneath it that can interpolate at
            /// all; see [`new_light_texture`].
            #[texture(0)]
            #[sampler(1)]
            pub texture: Handle<Image>,
            /// Multiplies the sampled texel. Canvas2D's `globalAlpha`, with the
            /// per-channel freedom the flat washes need.
            #[uniform(2)]
            pub tint: Vec4,
            /// How far this quad's sampling is allowed to cross a texel edge,
            /// in `x`; see [`LIGHT_WGSL`] and [`LIGHT_SOFTNESS`]. `yzw` are
            /// unused — a WGSL uniform pads a scalar to 16 bytes regardless, so
            /// a `Vec4` costs what an `f32` would have and says the padding is
            /// deliberate.
            #[uniform(3)]
            pub softness: Vec4,
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
    /// The coloured light, the bloom composite, and the flat washes.
    LightGlowMaterial,
    BLEND_ADD
);

// --- The bloom's gather pass -------------------------------------------------

/// `src`, ignoring whatever was there. The gather owns its target outright.
///
/// Not [`AlphaMode2d::Opaque`], which would say the same thing by putting the
/// quad in the opaque phase: this module's other four materials are all
/// transparent-phase with an overridden blend, and one pass that reaches the
/// same place by a different mechanism is a thing a reader has to check.
const BLEND_REPLACE: BlendState = BlendState {
    color: BlendComponent::REPLACE,
    alpha: BlendComponent::REPLACE,
};

/// Threshold, blur, done — the entire bloom, in one gather over the cell plane.
///
/// Held inline for the same reason [`LIGHT_WGSL`] is: every decision worth
/// arguing about is in [`bake_bloom_params`], on the CPU, where a test can read
/// it. What is left is a bounds-checked double loop and a multiply-add, and a
/// `.wgsl` file beside the module would only put that four inches further from
/// the constants that shape it.
///
/// The loop bounds are `BLOOM_RADIUS_CELLS` and the tap array is that plus one;
/// a `const` assert beside [`BloomParams`] fails the build if either drifts.
///
/// `textureLoad` and no sampler at all. One texel is one cell — there is nothing
/// between two cells to interpolate, which is exactly the reasoning
/// [`crate::cellmap`]'s own id binding is written on. The SMOOTHING all happens
/// in the gather; the sampler that matters is the linear one on the way back
/// out, on the composite quad.
const BLOOM_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct BloomParams {
    emit: array<vec4<f32>, 64>,
    taps: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var cell_ids: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params: BloomParams;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let dim = vec2<i32>(textureDimensions(cell_ids));
    let centre = vec2<i32>(floor(mesh.uv * vec2<f32>(dim)));
    let w = array<f32, 4>(params.taps.x, params.taps.y, params.taps.z, params.taps.w);

    var sum = vec3<f32>(0.0);
    for (var dy = -3; dy <= 3; dy = dy + 1) {
        let y = centre.y + dy;
        if (y < 0 || y >= dim.y) {
            continue;
        }
        let wy = w[abs(dy)];
        for (var dx = -3; dx <= 3; dx = dx + 1) {
            let x = centre.x + dx;
            if (x < 0 || x >= dim.x) {
                continue;
            }
            let id = min(textureLoad(cell_ids, vec2<i32>(x, y), 0).r, 63u);
            sum = sum + params.emit[id].rgb * (wy * w[abs(dx)]);
        }
    }
    return vec4<f32>(sum, 1.0);
}
"#;

/// Handle for [`BLOOM_WGSL`], inserted by [`LightPlugin`].
const BLOOM_SHADER: Handle<Shader> = uuid_handle!("2f4c8d16-5b73-4e90-a1c2-7d6e8f0b3a45");

/// The bloom's gather pass: `cellmap`'s cell ids in, blurred emissive light out.
///
/// This is the only material in the module that is not a passthrough, and the
/// only one that reads a texture it does not own. That coupling to
/// [`crate::cellmap::CellMap`] is the deliberate part: sharing the id plane is
/// what makes this a bloom OF THE SCENE rather than a second, independently
/// drifting opinion about where the lights are.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct BloomMaterial {
    /// [`crate::cellmap::CellMap::ids`] — one `R16Uint` texel per cell of the
    /// streaming window, not a copy of it. `u_int`, so `textureLoad` only.
    #[texture(0, sample_type = "u_int")]
    pub ids: Handle<Image>,
    /// The baked threshold table and gather weights.
    #[uniform(1)]
    pub params: BloomParams,
}

impl Material2d for BloomMaterial {
    fn fragment_shader() -> ShaderRef {
        BLOOM_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        set_blend(descriptor, BLEND_REPLACE);
        Ok(())
    }
}

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
///
/// **Unmoved.** The sprites this pass replaced sat here, and the ordering around
/// it is load-bearing: after the darkness multiply and the coloured light,
/// because a bloom is light being added to a scene that has already been lit and
/// must not be dimmed by the multiply that darkened the rock it spills onto;
/// BEFORE the vignette and the washes, because the frame's edges have to be able
/// to close in over a glow the same as over anything else, and because a bloom
/// that survived the vignette would light the one part of the frame the vignette
/// exists to take away. Creature glow at 0.80 and the UI at 1.0 stay above it,
/// which is why a lava lake cannot bloom the health bar.
const BLOOM_Z: f32 = 0.72;

/// The layer the bloom's gather quad lives on, and the only thing on it.
///
/// A layer of its own so [`BLOOM_CAMERA_ORDER`]'s camera renders exactly one
/// quad and nothing else in the game can wander into the bloom source. It is
/// NOT [`WORLD_LAYERS`] for the reason that decides the shape of this whole
/// pass: everything the game draws is on `WORLD_LAYERS`, including the UI and
/// including the bloom's own composite quad, so a gather that read a re-render
/// of that layer would bloom the health bar and would feed its own output back
/// into its own input a frame later. Deriving the bloom from the cell plane
/// instead of from the finished framebuffer is what avoids both, and it is worth
/// being explicit that this is the reason rather than an accident.
const BLOOM_LAYERS: RenderLayers = RenderLayers::layer(2);
/// The vignette, over everything in the world.
const VIGNETTE_Z: f32 = 0.75;
/// The flat washes, last, exactly as the original drew them.
const WASH_Z: f32 = 0.76;

/// The biome's ambient cast, 0..1 per channel.
///
/// Written by [`crate::ambience`], which is the only thing that knows which
/// biome the camera is standing in. The values are each biome's authored
/// `atmo.ambient` from `godgame_core::sim::biomes`, blended over the same
/// normalised surface weights the atmosphere is resolved from — so a tundra
/// casts cold and a volcanic casts red, and nothing here invents a palette.
///
/// `Default` is black, which is what plains sends and is a no-op in the
/// composite. That default is a real fallback rather than a placeholder: an app
/// running the lighting without the ambience plugin composites correctly and
/// merely uncast.
///
/// This carried a SEAM note for two milestones saying the ambience layer had not
/// landed. It had — the resource was declared, read by the solve, and written by
/// nothing, so every biome lit identically and no test could see it. If you are
/// adding a resource that something else is expected to fill, that failure mode
/// is worth remembering: it degrades to plausible output.
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

    /// The darkness multiply factors, one texel per light cell.
    shadow: Handle<Image>,
    /// The additive coloured light, one texel per light cell.
    colour: Handle<Image>,
    /// The radial vignette, in view space.
    vignette: Handle<Image>,
    /// The blurred emissive field, one texel per CELL. Written by the gather
    /// camera at [`BLOOM_CAMERA_ORDER`], read by the composite quad.
    bloom: Handle<Image>,

    shadow_mat: Handle<LightShadowMaterial>,
    colour_mat: Handle<LightGlowMaterial>,
    vignette_mat: Handle<LightShadowMaterial>,
    wash_mat: Handle<LightGlowMaterial>,
    bloom_mat: Handle<LightGlowMaterial>,

    /// The view the textures above are currently sized for.
    view: View,
    /// `(view, depth, day)` the vignette currently on the GPU was baked from, or
    /// `None` before the first bake. See [`VIGNETTE_REBAKE_EPS`].
    vignette_baked: Option<(View, f32, f32)>,
    /// World px of the light grid's top-left corner, sim convention.
    origin: Vec2,
    /// World px the light grid spans.
    extent: Vec2,
    /// World px of the view's centre, sim convention.
    centre: Vec2,
    /// World px the bloom gather covers, snapped to cells, sim convention.
    bloom_rect: Rect2,
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
    /// The blurred emissive field. Covers [`LightPass::bloom_rect`].
    Bloom,
    /// The radial vignette. Covers the view.
    Vignette,
    /// The underworld glow and the biome cast, summed. Covers the view.
    Wash,
}

/// The camera that runs the bloom's gather into [`LightPass::bloom`].
#[derive(Component, Clone, Copy, Debug)]
pub struct BloomCamera;

/// The one quad [`BloomCamera`] renders: the cell plane, thresholded and blurred.
#[derive(Component, Clone, Copy, Debug)]
pub struct BloomSource;

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
        let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
        shaders
            .insert(&LIGHT_SHADER, Shader::from_wgsl(LIGHT_WGSL, file!()))
            .expect("the light shader handle is a uuid and cannot collide");
        shaders
            .insert(&BLOOM_SHADER, Shader::from_wgsl(BLOOM_WGSL, file!()))
            .expect("the bloom shader handle is a uuid and cannot collide");

        app.add_plugins((
            Material2dPlugin::<LightShadowMaterial>::default(),
            Material2dPlugin::<LightGlowMaterial>::default(),
            Material2dPlugin::<BloomMaterial>::default(),
        ))
        .init_resource::<BiomeAmbient>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                bind_cell_ids.run_if(resource_exists::<CellMap>),
                (solve_light, (place_quads, place_bloom).after(solve_light))
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<LowResTarget>),
            )
                .run_if(resource_exists::<LightPass>),
        );
    }
}

/// Allocate every texture and material, and spawn the quads.
///
/// The four view-sized textures are rewritten or re-rendered every frame, and
/// RESIZED by [`solve_light`] on the first frame, because [`LowResTarget`] is
/// inserted by a command and so does not exist yet while this runs. Sizing from
/// [`View::default`] here and letting the first solve correct it is the same
/// trade [`crate::mobs`] makes with its spawn rectangles. [`CellMap`] is
/// inserted by a command too, which is why the bloom's id binding is left blank
/// here and filled by [`bind_cell_ids`] rather than read out of the world.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut shadow_materials: ResMut<Assets<LightShadowMaterial>>,
    mut glow_materials: ResMut<Assets<LightGlowMaterial>>,
    mut bloom_materials: ResMut<Assets<BloomMaterial>>,
) {
    let view = View::default();
    let (lw, lh) = grid_size(view);
    let (vw, vh) = vignette_size(view);
    let (bw, bh) = bloom_size(view);

    let shadow = images.add(new_light_texture(lw, lh));
    let colour = images.add(new_light_texture(lw, lh));
    let vignette = images.add(new_light_texture(vw, vh));
    let bloom = images.add(new_bloom_target(bw, bh));
    // A single white texel, so the flat washes go through the same shader as
    // everything else and the tint uniform carries the whole signal.
    let white = images.add(new_light_texture(1, 1));

    // Which quads get the softness dial and which are pinned open. The two
    // light grids and the bloom are all one texel per CELL, so snapping their
    // samples toward a texel centre snaps them to the art's own blocks. The
    // vignette is one texel per 16 VIEW px and the wash is a single texel: there
    // is no lattice under either to snap to, and a dial that quantised the
    // vignette would band a ramp whose entire job is to be smooth.
    let snap = Vec4::splat(LIGHT_SNAP);
    let smooth = Vec4::ONE;

    let shadow_mat = shadow_materials.add(LightShadowMaterial {
        texture: shadow.clone(),
        tint: Vec4::ONE,
        softness: snap,
    });
    let vignette_mat = shadow_materials.add(LightShadowMaterial {
        texture: vignette.clone(),
        tint: Vec4::ONE,
        softness: smooth,
    });
    let colour_mat = glow_materials.add(LightGlowMaterial {
        texture: colour.clone(),
        tint: Vec4::ONE,
        softness: snap,
    });
    let wash_mat = glow_materials.add(LightGlowMaterial {
        texture: white,
        tint: Vec4::ZERO,
        softness: smooth,
    });
    // Set ONCE, here, and never touched again for the life of the process. The
    // strength of the bloom is a constant; the pass this replaced re-tinted up
    // to 120 materials every frame to say the same thing with a flicker on it.
    let bloom_mat = glow_materials.add(LightGlowMaterial {
        texture: bloom.clone(),
        tint: Vec4::new(BLOOM_INTENSITY, BLOOM_INTENSITY, BLOOM_INTENSITY, 1.0),
        // One texel per cell, like the two grids above, so the glow answers to
        // the same dial. It used to be pinned hard by a `nearest` sampler on the
        // texture; that is this value at 0, and there is no reason the bloom
        // should be the one layer whose hardness cannot be tuned with the rest.
        softness: snap,
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
        MeshMaterial2d(bloom_mat.clone()),
        LightQuad::Bloom,
        BLOOM_Z,
    ));
    commands.spawn(quad_bundle(
        &quad,
        MeshMaterial2d(wash_mat.clone()),
        LightQuad::Wash,
        WASH_Z,
    ));

    // The gather. ONE quad and ONE camera, in place of 120 pooled sprites and
    // the 120 material writes a frame that kept them tinted. The quad covers the
    // whole streaming window so its uv IS the cell plane's uv and the shader
    // needs no origin uniform at all; the camera crops that to the view, at one
    // texel per cell, which is the downsample.
    commands.spawn((
        Mesh2d(quad.clone()),
        MeshMaterial2d(bloom_materials.add(BloomMaterial {
            ids: Handle::default(),
            params: bake_bloom_params(),
        })),
        Transform::default(),
        BloomSource,
        BLOOM_LAYERS,
    ));
    commands.spawn((
        Camera2d,
        Camera {
            order: BLOOM_CAMERA_ORDER,
            // Black and not the sky: every texel of this target is an amount of
            // light to ADD, so "nothing here" has to be a literal zero.
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        RenderTarget::Image(bloom.clone().into()),
        // Fixed and not `WindowSize`, which would map one world px per texel and
        // shrink the gather to a fifth of the view. Corrected every frame by
        // `place_bloom`; this only avoids a frame at the wrong extent.
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: (bw * CELL_SIZE) as f32,
                height: (bh * CELL_SIZE) as f32,
            },
            ..OrthographicProjection::default_2d()
        }),
        Msaa::Off,
        BloomCamera,
        BLOOM_LAYERS,
    ));

    commands.insert_resource(LightPass {
        grid: LightGrid::new(view, SEED),
        census: EmitterScan::default(),
        vignette_baked: None,
        shadow,
        colour,
        vignette,
        bloom,
        shadow_mat,
        colour_mat,
        vignette_mat,
        wash_mat,
        bloom_mat,
        view,
        origin: Vec2::ZERO,
        extent: Vec2::ZERO,
        centre: Vec2::ZERO,
        bloom_rect: Rect2 {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        },
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

/// A blank `Rgba8Unorm` grid with a LINEAR sampler.
///
/// Not sRGB: every byte in these textures is a blend factor, and a transfer
/// function applied on the way in would silently change the arithmetic
/// [`LightGrid::bake_shadow`] worked out.
///
/// # Linear, and the game's one `default_nearest` override, for a NEW reason
///
/// The old reason was that the grid was one texel per 20 world px and nearest
/// would have laid a hard 20 px lattice over the frame. True, and no longer the
/// situation: a texel is a cell now. The reason it is still linear is that
/// [`LIGHT_WGSL`] needs a filter it can dial — it snaps the sample point toward
/// the texel centre by `1 - LIGHT_SOFTNESS`, and a NEAREST sampler would make
/// every value of that dial identical. The hardness of the light lives in one
/// tunable number instead of in a choice between two samplers, which is what
/// lets it be a fraction of a cell rather than all or nothing.
///
/// Used for the two light grids and, at a softness pinned to 1, for the vignette
/// and the washes: neither of those is on the cell lattice, so neither has
/// anything to snap to.
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

/// The bloom's render target: `w x h` CELLS, cleared and rewritten every frame.
///
/// `Rgba8Unorm`, for [`new_light_texture`]'s reason and one more of its own: the
/// gather writes LINEAR light, the composite ADDS it in linear, and an sRGB
/// target would encode on the way in and decode on the way out for no purpose
/// except to lose the darkest two stops of a pass whose whole output lives near
/// zero.
///
/// **`RENDER_ATTACHMENT`, which is what makes this a camera target rather than a
/// texture**, and `TEXTURE_BINDING`, which is what lets the composite quad read
/// it back. `COPY_DST` because [`Image::resize`] writes through it.
///
/// # The sampler, and why it stopped being the place this was decided
///
/// It was LINEAR first, on the reasoning that glow is the one quantity in a
/// frame that ought to be smooth. That reasoning was sound and the result was
/// wrong: a smoothly-interpolated haze over ore drawn in hard 5px blocks reads
/// as a photographic effect pasted onto pixel art, which is exactly the
/// complaint [`crate::sky`] answered by quantising the sun to the same grid.
/// So it became NEAREST, on the ground that one texel here is `CELL_SIZE` world
/// px — the SAME lattice the cells are drawn on — so snapping the glow to it
/// makes a lit block glow as a block rather than stamping a foreign grid.
///
/// Both of those are arguments about how far a glow may bleed past the block
/// that cast it, and the honest answer to that turned out not to be either 0 px
/// or a whole texel. It is [`LIGHT_SOFTNESS`], applied by [`LIGHT_WGSL`] on the
/// composite quad — so the sampler here is linear because that is the filter the
/// dial needs underneath it, and nearest is what the dial does at 0.
///
/// The composite still resolves into the 640x400 buffer and the buffer is still
/// blitted with the game's one nearest sampler, so nothing about the edges of
/// the art changes either way. What is tunable is whether the GLOW has edges.
fn new_bloom_target(w: i32, h: i32) -> Image {
    let size = Extent3d {
        width: w.max(1) as u32,
        height: h.max(1) as u32,
        depth_or_array_layers: 1,
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("godgame_bloom"),
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        sampler: ImageSampler::linear(),
        ..default()
    };
    image.resize(size);
    image
}

/// Point the bloom's gather at [`CellMap`]'s id plane, once.
///
/// Separate from [`setup`] and not merely late in it: `CellMap` is inserted by a
/// `Commands` call in another plugin's `Startup` system, so it does not exist
/// until that schedule's sync point — no ordering constraint inside `Startup`
/// can make it visible there. Guarded on inequality rather than on a `Local`
/// flag so it also survives `CellMap` being rebuilt, and so the common case is a
/// handle compare and no asset mutation at all.
fn bind_cell_ids(
    cellmap: Res<CellMap>,
    source: Single<&MeshMaterial2d<BloomMaterial>, With<BloomSource>>,
    mut materials: ResMut<Assets<BloomMaterial>>,
) {
    if materials
        .get(source.id())
        .is_some_and(|m| m.ids == cellmap.ids)
    {
        return;
    }
    if let Some(mut material) = materials.get_mut(source.id()) {
        material.ids = cellmap.ids.clone();
    }
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

    // Before anything reads the surface line. `crate::ambience::breathe` does the
    // same against the same resource; see `LightGrid::follow_seed`.
    pass.grid.follow_seed(inputs.world.seed);

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
        touch(&mut glow_materials, &pass.bloom_mat.clone());
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
    // screen. Only when the solve is coarser than that walk and can miss
    // something in it — see `CENSUS_NEEDED`. When it is not, `pass.census` is
    // never filled and stays the empty scan `solve` treats as "no census".
    if CENSUS_NEEDED {
        let cell = CELL_SIZE as f32;
        scan_emitters(
            cells,
            (rect.x / cell).floor() as i32 - 1,
            (rect.y / cell).floor() as i32 - 1,
            (rect.w / cell).ceil() as i32 + 2,
            (rect.h / cell).ceil() as i32 + 2,
            &mut pass.census,
        );
    }

    // Split the borrow: `solve` needs the grid mutably and the census by
    // reference, and they are two fields of the same resource.
    {
        let LightPass { grid, census, .. } = &mut *pass;
        grid.solve(cells, frame, census.x(), census.y());
    }

    pass.origin = Vec2::new(frame.ox as f32 * stride, frame.oy as f32 * stride);
    pass.extent = Vec2::new(
        pass.grid.cols() as f32 * stride,
        pass.grid.rows() as f32 * stride,
    );
    pass.centre = centre;
    // Geometry only. The bloom has NO per-frame CPU work beyond this line and no
    // asset write at all — the gather runs on the GPU from a texture the cell
    // pass already maintains, which is the whole difference from the sprite
    // stamping this replaced.
    pass.bloom_rect = bloom_rect(rect, bloom_size(view));
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
    // The vignette is re-baked only when it would actually differ. `upload` calls
    // `Assets::get_mut`, which is what schedules a GPU upload, so a frame that
    // skips this skips the copy as well as the 1 530 samples.
    let stale = match pass.vignette_baked {
        Some((v, d, dy)) => {
            v != view
                || (d - depth).abs() >= VIGNETTE_REBAKE_EPS
                || (dy - day).abs() >= VIGNETTE_REBAKE_EPS
        }
        None => true,
    };
    if stale {
        upload(&vignette, &mut images, |out| {
            bake_vignette(view, depth, day, out);
        });
        pass.vignette_baked = Some((view, depth, day));
    }

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
    // The texture below is reallocated, so whatever was baked into it is gone.
    // The `view` in the tuple would catch this on its own; stating it here means
    // a future resize that stopped changing the view still cannot serve a stale
    // vignette out of a freshly allocated buffer.
    pass.vignette_baked = None;
    let (lw, lh) = grid_size(view);
    let (vw, vh) = vignette_size(view);
    let (bw, bh) = bloom_size(view);
    for (handle, (w, h)) in [
        (pass.shadow.clone(), (lw, lh)),
        (pass.colour.clone(), (lw, lh)),
        (pass.vignette.clone(), (vw, vh)),
        // Resized like the rest. It is a camera TARGET as well as a sampled
        // texture, and Bevy sizes the attachment from the asset, so this one
        // line is also what keeps the gather camera's viewport correct.
        (pass.bloom.clone(), (bw, bh)),
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

/// Put the five composite quads where this frame's geometry says they go.
///
/// Bevy's +y is up and the sim's is down. This function and [`place_bloom`] are
/// the only two places in the module where that is true.
fn place_quads(
    pass: Res<LightPass>,
    mut quads: Query<(&LightQuad, &mut Transform, &mut Visibility)>,
) {
    let grid_centre = pass.origin + pass.extent * 0.5;
    let bloom = pass.bloom_rect;
    for (which, mut transform, mut visibility) in &mut quads {
        let (centre, extent, shown) = match which {
            LightQuad::Shadow => (grid_centre, pass.extent, true),
            LightQuad::Colour => (grid_centre, pass.extent, pass.colour_visible),
            // Exactly the rect the gather camera covered, to the world px. The
            // texel centres of the bloom target then land on cell centres in
            // this quad's uv, which is what makes the linear upscale interpolate
            // between the two cells it should be between rather than smearing a
            // half-cell offset across the frame.
            LightQuad::Bloom => (
                Vec2::new(bloom.x + bloom.w * 0.5, bloom.y + bloom.h * 0.5),
                Vec2::new(bloom.w, bloom.h),
                true,
            ),
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

/// Aim the bloom's gather: its camera at the view, its quad at the window.
///
/// **This function used to move 120 sprites and rewrite 120 materials.** It now
/// writes two transforms and, on a resize, one projection. Nothing here touches
/// an asset, which is the point: `docs/PERF.md` §8.5 flagged the 120
/// `Assets::get_mut` calls a frame as the one renderer cost it could not measure
/// because the cost was in the render-world extract, and the honest way to close
/// a measurement you cannot take is to delete the thing being measured.
///
/// Bevy's +y is up and the sim's is down; this and [`place_quads`] are the only
/// two places in the module where that is true.
fn place_bloom(
    pass: Res<LightPass>,
    world: Res<SimWorld>,
    camera: Single<(&mut Transform, &mut Projection), With<BloomCamera>>,
    source: Single<&mut Transform, (With<BloomSource>, Without<BloomCamera>)>,
) {
    let rect = pass.bloom_rect;
    let (mut camera_transform, mut projection) = camera.into_inner();
    camera_transform.translation.x = rect.x + rect.w * 0.5;
    camera_transform.translation.y = -(rect.y + rect.h * 0.5);

    // Written only when it moves — matched on rather than compared because
    // `ScalingMode` is not `PartialEq`. A `Projection` write is change detection
    // the render world acts on, and the view changes on a window drag and never
    // otherwise.
    if let Projection::Orthographic(ortho) = &mut *projection
        && !matches!(
            ortho.scaling_mode,
            ScalingMode::Fixed { width, height } if width == rect.w && height == rect.h
        )
    {
        ortho.scaling_mode = ScalingMode::Fixed {
            width: rect.w,
            height: rect.h,
        };
    }

    // The source quad is the whole streaming window, placed exactly as
    // `cellmap::follow_window` places the cell quad — same rect, same reasoning
    // about which edge is grid row 0. That is what lets the shader read its cell
    // coordinate straight out of `mesh.uv` with no origin uniform to keep in
    // step, and what makes the margin work: the gather camera can look at cells
    // outside the view because the quad it is looking at is bigger than it.
    let grid = &world.level.grid;
    let w = (grid.cols() * CELL_SIZE) as f32;
    let h = (grid.rows() * CELL_SIZE) as f32;
    let x = (grid.origin_cell_x() * CELL_SIZE) as f32;
    let y = (grid.origin_cell_y() * CELL_SIZE) as f32;
    let mut source = source.into_inner();
    source.translation = Vec3::new(x + w * 0.5, -(y + h * 0.5), 0.0);
    source.scale = Vec3::new(w, h, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::sim::coords::WorldCell;
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
        grid.acc = vec![0.0; 16];
        grid.lr = vec![0.0; n];
        grid.lg = vec![0.0; n];
        grid.lb = vec![0.0; n];
        grid.seen = vec![false; n];
        grid
    }

    /// The light grid must follow the world it is lighting, not a constant.
    ///
    /// This grid keeps its own `Heightmap` and `Noise` to seed the skylight
    /// flood, and it built them from the compile-time `SEED` at both production
    /// sites while `crate::ambience` — which keeps a second copy for its own
    /// reasons — guarded itself. Nothing was wrong on screen, because
    /// `build_world` is only ever called with that constant today.
    ///
    /// So this test is not defending a live bug. It is defending against the one
    /// that arrives the day someone adds a `--seed` flag: the solver would flood
    /// sky down to one surface line while the terrain sat at another, silently.
    /// The bug that motivated the whole frame-stability effort had exactly that
    /// shape, and 896 tests missed it.
    ///
    /// Asserting on the SURFACE ROW rather than on the seed field is deliberate,
    /// and it is what catches the real hazard. `Heightmap` retires its memo by
    /// comparing the ADDRESS of the noise it is given, so a `follow_seed` that
    /// overwrote `self.noise` in place and kept the heightmap would leave every
    /// memoised row live and stale — while passing any check on the seed field.
    /// Only reading the surface line back can see that.
    #[test]
    fn the_light_grid_follows_the_world_seed_it_is_given() {
        let mut grid = LightGrid::new(View::default(), SEED);
        let before = grid.heights.surface_row_at(&grid.noise, 0, None);

        // A seed it was not built with. `world_noise` is a pure function of it,
        // so a different seed is a different world with a different surface.
        grid.follow_seed(SEED + 1);
        assert_eq!(grid.seed(), SEED + 1, "the grid did not adopt the new seed");
        let after = grid.heights.surface_row_at(&grid.noise, 0, None);
        assert_ne!(
            before, after,
            "the surface line did not move, so either the noise or the heightmap \
             memo survived the seed change and the flood is lighting a world that \
             is no longer there"
        );

        // Idempotent, because it runs every frame: a re-seed to the value it
        // already holds must not throw the memo away and pay for it again.
        grid.follow_seed(SEED + 1);
        assert_eq!(grid.heights.surface_row_at(&grid.noise, 0, None), after);

        // And it goes back, so this is a mirror of the world and not a latch.
        grid.follow_seed(SEED);
        assert_eq!(grid.heights.surface_row_at(&grid.noise, 0, None), before);
    }

    /// The absolute cell a light cell samples, on either axis.
    ///
    /// Every test below that places a block "where the light grid will see it"
    /// goes through this. Written out rather than hard-coded because the answer
    /// moves with [`LIGHT_DOWNSCALE`], and a suite that pins cell 18 to light
    /// cell 4 is a suite that only tests one downscale.
    fn sample_cell(l: i32) -> i32 {
        l * LIGHT_DOWNSCALE + LIGHT_DOWNSCALE / 2
    }

    /// Whether the sampling loop looks at this cell coordinate at all.
    fn on_sample_lattice(c: i32) -> bool {
        c.rem_euclid(LIGHT_DOWNSCALE) == LIGHT_DOWNSCALE / 2
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

    /// A shaft with a wall behind it is dimmer than open sky and brighter than
    /// rock.
    ///
    /// This is the whole of the background wall plane's contribution to lighting,
    /// and it is the difference the feature exists to make readable: a shaft that
    /// broke through to daylight is genuinely open above you, and one that is
    /// still inside the world is not. `back == EMPTY` is that distinction, and
    /// this is where it becomes something you can see.
    ///
    /// Asserted as an ORDERING rather than against a number, so retuning
    /// `WALL_DECAY_PER_CELL` — which is a taste question and will move — cannot
    /// break it. What must never move is the ordering, and a `const _` beside the
    /// constants pins that at compile time as well.
    ///
    /// Worth stating what this does NOT claim: a deep cave is unaffected, because
    /// there is no skylight left down there to modulate. Walls matter to the
    /// flood near the surface, where a shaft can still reach the sky.
    #[test]
    fn a_walled_shaft_is_dimmer_than_open_sky_and_brighter_than_rock() {
        // Twelve cells down, in the same world-distance terms the rock test uses.
        const DEEP_CELLS: i32 = 12;
        let rows = DEEP_CELLS / LIGHT_DOWNSCALE;

        // Three columns of the same world, differing only in what is behind the
        // air: nothing, a wall, and solid rock in front.
        let open = air(64, 64);

        let mut walled = air(64, 64);
        for y in 0..walled.rows() {
            for x in 0..walled.cols() {
                walled.set_back_world(WorldCell::new(x, y), block::STONE);
            }
        }

        let rock = world(64, 64, 0);

        let sample = |g: &CellGrid| {
            let mut light = solver();
            light.compute_skylight(g, noon(0, 0));
            light.light_at(4, rows)
        };

        let open_lit = sample(&open);
        let walled_lit = sample(&walled);
        let rock_lit = sample(&rock);

        assert!(
            walled_lit < open_lit,
            "a walled shaft is not open sky — sunlight does not pour down a \
             mineshaft the way it pours into a canyon. open {open_lit}, walled \
             {walled_lit}"
        );
        assert!(
            walled_lit > rock_lit,
            "a walled shaft is brighter than the rock it was cut out of, or there \
             was no point digging it. walled {walled_lit}, rock {rock_lit}"
        );
    }

    /// A wall behind SOLID rock changes nothing.
    ///
    /// The flood asks about the wall only where the front is air. Rock in front of
    /// a wall is just rock, and if the two ever compounded, every cell in the
    /// world would be darker than it was before the back plane existed — the
    /// entire world dimming at once, which is the kind of change that reads as
    /// "the lighting was retuned" rather than as a bug.
    #[test]
    fn a_wall_behind_rock_does_not_darken_it_twice() {
        let plain = world(64, 64, 0);

        let mut backed = world(64, 64, 0);
        for y in 0..backed.rows() {
            for x in 0..backed.cols() {
                backed.set_back_world(WorldCell::new(x, y), block::STONE);
            }
        }

        let sample = |g: &CellGrid| {
            let mut light = solver();
            light.compute_skylight(g, noon(0, 0));
            (0..light.rows())
                .map(|r| light.light_at(4, r))
                .collect::<Vec<_>>()
        };

        assert_eq!(
            sample(&plain),
            sample(&backed),
            "putting a wall behind solid rock moved the skylight — the two \
             occlusions are compounding where only one applies"
        );
    }

    #[test]
    fn light_decays_far_faster_through_rock_than_through_air() {
        // Twelve CELLS of rock, however many light rows that happens to be.
        // Occlusion is a property of the world, so the assertion has to be about
        // a world distance: pinning it to a count of light rows is what let the
        // decays mean four different things at four different downscales.
        const DEEP_CELLS: i32 = 12;
        let rows = DEEP_CELLS / LIGHT_DOWNSCALE;

        let g = world(64, 64, 0); // solid from the very top
        let mut light = solver();
        light.compute_skylight(&g, noon(0, 0));

        let first = light.light_at(4, 0);
        let deep = light.light_at(4, rows);
        let expected = first * SOLID_DECAY.powi(rows);
        assert!(
            (deep - expected).abs() < 1e-4,
            "expected {expected} {DEEP_CELLS} cells into rock, got {deep}"
        );
        assert!(
            deep < first * 0.2,
            "rock has to actually occlude: {first} at the top, {deep} \
             {DEEP_CELLS} cells down"
        );
    }

    #[test]
    fn the_decays_are_the_same_loss_per_world_px_at_any_downscale() {
        // THE regression this file is one half of. `OPEN_DECAY` and
        // `SOLID_DECAY` are per light cell, so at four cells per sample they
        // were 0.99 and 0.55 — and those two literals, left alone, would have
        // meant four times the loss per world px the day the stride changed.
        // Compounding over twelve cells is the same number however the twelve
        // are grouped, and that is the invariant worth failing a build over.
        const CELLS: i32 = 12;
        let steps = CELLS / LIGHT_DOWNSCALE;
        for (per_light_cell, historical, name) in [
            (OPEN_DECAY, 0.99f32, "open"),
            (SOLID_DECAY, 0.55f32, "solid"),
        ] {
            let ours = per_light_cell.powi(steps);
            let theirs = historical.powi(CELLS / 4);
            assert!(
                (ours - theirs).abs() < 1e-4,
                "{name} air/rock lost {ours} over {CELLS} cells; the 4-cell \
                 stride this was tuned at lost {theirs}"
            );
        }
    }

    #[test]
    fn a_solid_cell_is_lit_by_what_reaches_it_and_not_by_what_it_blocks() {
        // STORE, THEN DECAY. The sunlit ground the player stands on is the
        // brightest solid in its column; a pass that decayed first would make it
        // darker than the air above by the full solid factor, and the whole
        // world would read as overcast dusk at noon.
        // The floor is placed ON the row light row 2 samples, so light row 1 is
        // the last air row and light row 2 is the first solid one at every
        // downscale.
        let g = world(64, 64, sample_cell(2));
        let mut light = solver();
        light.compute_skylight(&g, noon(0, 0));

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
        g.set(sample_cell(4), sample_cell(4), id);
        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));

        let centre = light.light_at(4, 4);
        let r = SPLAT_REACH;
        assert!(centre > 0.0, "the emitter did not light its own cell");
        for (dx, dy) in [(-r, 0), (r, 0), (0, -r), (0, r)] {
            let edge = light.light_at(4 + dx, 4 + dy);
            assert!(
                (edge - centre * SPLAT_EDGE).abs() < 1e-4,
                "arm ({dx}, {dy}) got {edge}, expected {}",
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
        assert!((light.light_at(4 - r, 4 - r) - diagonal).abs() < 1e-4);
    }

    #[test]
    fn light_falls_off_to_nothing_at_the_radius() {
        // A strong emitter throws a far ring two light cells out and NOTHING
        // beyond it. Before the reach scaled with the declared level, a magma
        // chamber and a glowing mushroom lit the same volume.
        let strong = an_emitter(FAR_SPLAT_LEVEL);
        let mut g = air(64, 64);
        g.set(sample_cell(4), sample_cell(4), strong);
        let mut light = solver();
        light.add_emissive(&g, noon(0, 0));

        assert!(
            light.light_at(4 + FAR_REACH, 4) > 0.0,
            "the far ring is missing"
        );
        assert_eq!(
            light.light_at(4 + FAR_REACH + 1, 4),
            0.0,
            "the cast reached past its radius"
        );
        assert_eq!(light.light_at(4, 4 + FAR_REACH + 1), 0.0);

        // ...and a weak one does not throw the ring at all.
        let weak = (1..MAT_COUNT as CellId)
            .find(|&id| emitter(id).emits() && emitter(id).level <= FAR_SPLAT_LEVEL);
        if let Some(weak) = weak {
            let mut g = air(64, 64);
            g.set(sample_cell(4), sample_cell(4), weak);
            let mut light = solver();
            light.add_emissive(&g, noon(0, 0));
            assert_eq!(
                light.light_at(4 + FAR_REACH, 4),
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
        // Five light cells apart, so neither splat lands on the other's centre
        // or in its far ring.
        let at = |lx: i32, ly: i32| (sample_cell(ly) * 64 + sample_cell(lx)) as usize;
        let at_threshold = at(4, 4);
        let boiling = at(9, 4);
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
        warm.temp[(sample_cell(4) * 64 + sample_cell(4)) as usize] = 200;
        light.add_emissive(&warm, noon(0, 0));
        assert!(light.colour_dirty());
        assert!(light.hot().is_empty(), "warm rock is not an emitter");
    }

    #[test]
    fn the_census_splat_is_deduplicated_per_light_cell() {
        let id = an_emitter(0.0);
        let mut g = air(64, 64);
        // Every cell of light cell (4, 4), listed twice. At a downscale of 1
        // there is only one such cell, and the rule being pinned — several
        // entries in one light cell make ONE splat — has to hold for that case
        // too, so the list is doubled rather than assumed to be four wide.
        let cy = sample_cell(4);
        let xs: Vec<i32> = (0..LIGHT_DOWNSCALE)
            .chain(0..LIGHT_DOWNSCALE)
            .map(|d| 4 * LIGHT_DOWNSCALE + d)
            .collect();
        let ys = vec![cy; xs.len()];
        for &cx in &xs {
            g.set(cx, cy, id);
        }
        let frame = noon(0, 0);

        let mut four = solver();
        four.add_emissive(&air(64, 64), frame); // clear, no emitters sampled
        four.add_census_emitters(&g, &xs, &ys, frame);

        // The FIRST entry in the group wins, phase and all — the flicker is
        // hashed from the entry's own absolute cell, so which one survives the
        // dedup is observable and this pins it.
        let mut one = solver();
        one.add_emissive(&air(64, 64), frame);
        one.add_census_emitters(&g, &xs[..1], &ys[..1], frame);

        // One splat, not four: four stacked splats of the same source would
        // blow the cell out to white.
        assert!(
            (four.light_at(4, 4) - one.light_at(4, 4)).abs() < 1e-6,
            "the census stacked: {} against {}",
            four.light_at(4, 4),
            one.light_at(4, 4)
        );
        assert_eq!(four.hot().len(), 1, "one light cell, one hot entry");
        assert_eq!(four.hot()[0], [xs[0], cy]);
    }

    #[test]
    fn a_census_entry_the_grid_no_longer_backs_is_dropped() {
        // The census says WHERE, the grid says WHAT. A torch dug out between the
        // scan and the splat must not leave a ghost light behind.
        let frame = noon(0, 0);
        let mut light = solver();
        let at = sample_cell(4);
        light.add_emissive(&air(64, 64), frame);
        light.add_census_emitters(&air(64, 64), &[at], &[at], frame);
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
        let at = sample_cell(4);
        light.add_census_emitters(&g, &[-400, 4000, at, at], &[at, at, -400, 4000], frame);
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
        blur_one(&mut light.light, &mut light.scratch, &mut light.acc, lw, lh);
        for (i, v) in light.light.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "cell {i} drifted to {v} — the edge clamp is wrong"
            );
        }

        light.light.fill(0.0);
        let centre = (8 * lw + 8) as usize;
        light.light[centre] = 1.0;
        blur_one(&mut light.light, &mut light.scratch, &mut light.acc, lw, lh);
        let r = BLUR_REACH;
        assert!(light.light[centre] < 1.0, "the spike did not spread");
        assert!(light.light_at(8 - r, 8) > 0.0 && light.light_at(8 + r, 8) > 0.0);
        assert!(light.light_at(8, 8 - r) > 0.0 && light.light_at(8, 8 + r) > 0.0);
        // Two separable passes reach the diagonals too, which is what rounds a
        // cross-shaped splat into a glow.
        assert!(light.light_at(8 - r, 8 - r) > 0.0);
        assert_eq!(
            light.light_at(8 - r - 1, 8),
            0.0,
            "it spread past the blur's radius"
        );

        // And the impulse response sums to one — the pass may move light around
        // but must never create or destroy any. The flat-field check above says
        // the same thing where the clamp hides it; this says it where a shifted
        // or mis-normalised box pair would show.
        let total: f32 = light.light.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "a unit spike blurred to {total}"
        );

        // The kernel is CENTRED. Two boxes of even width, both leaning the same
        // way, would blur perfectly well and put the whole light field half a
        // cell off the art.
        assert!(
            (light.light_at(8 - r, 8) - light.light_at(8 + r, 8)).abs() < 1e-6
                && (light.light_at(8, 8 - r) - light.light_at(8, 8 + r)).abs() < 1e-6,
            "the blur is lopsided: {:?}",
            (
                light.light_at(8 - r, 8),
                light.light_at(8 + r, 8),
                light.light_at(8, 8 - r),
                light.light_at(8, 8 + r)
            )
        );
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
    fn a_single_cell_torch_is_lit_wherever_it_sits_in_its_light_cell() {
        // THE contract the census exists to keep, stated so that it holds at
        // every downscale rather than at four. A coarse grid steps over fifteen
        // cells in sixteen and the census covers them; a grid of one sample per
        // cell steps over nothing and takes no census at all. Either way a torch
        // is lit, and the test that only knew the first arrangement said the
        // second was broken.
        let id = an_emitter(0.0);
        // The far corner of light cell (4, 4), and its sample centre. At a
        // downscale of 1 they are the same cell, which is the whole point.
        let corner = 4 * LIGHT_DOWNSCALE + LIGHT_DOWNSCALE - 1;
        for (cx, cy) in [(corner, corner), (sample_cell(4), sample_cell(4))] {
            let mut g = air(64, 64);
            g.set(cx, cy, id);
            let (lx, ly) = (
                cx.div_euclid(LIGHT_DOWNSCALE),
                cy.div_euclid(LIGHT_DOWNSCALE),
            );
            let frame = noon(0, 0);

            let mut light = solver();
            light.add_emissive(&g, frame);
            assert_eq!(
                light.light_at(lx, ly) > 0.0,
                on_sample_lattice(cx) && on_sample_lattice(cy),
                "the sampling loop found ({cx}, {cy}) exactly when it looks there"
            );

            if CENSUS_NEEDED {
                let mut scan = EmitterScan::default();
                scan_emitters(&g, 0, 0, 64, 64, &mut scan);
                assert_eq!(scan.len(), 1);
                assert_eq!((scan.x()[0], scan.y()[0]), (cx, cy));
                light.add_census_emitters(&g, scan.x(), scan.y(), frame);
            }

            assert!(
                light.light_at(lx, ly) > 0.0,
                "({cx}, {cy}) was never lit by any pass"
            );
        }
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
    fn the_bloom_kernel_sums_to_one_so_the_pass_cannot_blow_out() {
        // THE guarantee the sprite bloom did not have, and the only reason
        // `BLOOM_INTENSITY` can be read as a hard ceiling rather than as taste.
        // The shader reads `w[|d|]` on both sides of the centre, so the sum that
        // has to be one is the centre once and every other ring twice — and the
        // 2D kernel is the product of two of those, so it sums to one squared.
        let p = bake_bloom_params();
        let taps = p.taps.to_array();
        let full: f32 = taps[0] + 2.0 * (taps[1] + taps[2] + taps[3]);
        assert!((full - 1.0).abs() < 1e-6, "1D kernel sums to {full}");

        // And it is a Gaussian, not an accident: strictly falling outward, and
        // every tap positive so no pixel can be darkened by a light source.
        for w in taps.windows(2) {
            assert!(w[1] > 0.0 && w[1] < w[0], "taps not falling: {taps:?}");
        }
    }

    #[test]
    fn the_bloom_threshold_admits_lamps_and_rejects_glints() {
        let p = bake_bloom_params();
        let weight = |id: CellId| {
            let v = p.emit[id as usize];
            // The table entry is the emitter's linear colour times the knee, so
            // the largest channel recovers the knee for any saturated emitter —
            // `EMIT_SATURATION` drives at least one channel of every cast to 1.
            v.x.max(v.y).max(v.z)
        };

        // Lava declares 13/15 and is over the top of the knee: full weight.
        assert!(weight(block::LAVA) > 0.99, "lava is not blooming in full");
        // Gold ore derives a level of 1/15 from its `emissive`. It is a glint on
        // a wall, and a glint that bloomed would put a halo on every ore vein in
        // the game.
        assert_eq!(weight(block::GOLD_ORE), 0.0, "gold ore is blooming");
        // Air is not a light source and its slot has to be a literal zero: the
        // gather reads it for every empty cell in the kernel and adds it.
        assert_eq!(p.emit[EMPTY as usize], Vec4::ZERO, "air blooms");
        // Nothing in the table can push the convex combination above one.
        for (id, v) in p.emit.iter().enumerate() {
            assert!(
                v.to_array().iter().all(|c| (0.0..=1.0).contains(c)),
                "slot {id} is out of range: {v:?}"
            );
        }
    }

    #[test]
    fn the_bloom_rect_is_cell_aligned_and_covers_the_view_with_margin() {
        let view = View::for_screen(1000, 500);
        let texels = bloom_size(view);
        // A texel per cell, plus the gather margin on both sides.
        assert_eq!(texels.0, view.w / CELL_SIZE + 2 * BLOOM_RADIUS_CELLS);
        assert_eq!(texels.1, view.h / CELL_SIZE + 2 * BLOOM_RADIUS_CELLS);

        // Sub-cell camera motion must not move the rect, or the whole glow
        // crawls against the art as the player walks.
        let at = |x: f32| {
            bloom_rect(
                Rect2 {
                    x,
                    y: 0.0,
                    w: view.w as f32,
                    h: view.h as f32,
                },
                texels,
            )
        };
        let base = at(1000.0);
        assert_eq!(base.x, at(1000.0 + CELL_SIZE as f32 - 1.0).x);
        assert_eq!(base.x + CELL_SIZE as f32, at(1000.0 + CELL_SIZE as f32).x);
        assert_eq!(base.x % CELL_SIZE as f32, 0.0, "rect is off the cell grid");

        // And it really does contain the view it was asked about, margin and
        // all — an emitter this far off screen still reaches the screen.
        let margin = (BLOOM_RADIUS_CELLS * CELL_SIZE) as f32;
        assert!(base.x <= 1000.0 - margin, "left margin lost");
        assert!(base.x + base.w >= 1000.0 + view.w as f32 + margin, "right");
    }

    /// The vignette cache never skips a frame anybody could see.
    ///
    /// `VIGNETTE_REBAKE_EPS` is the whole of that promise, and this is what makes
    /// it a measurement rather than a guess: over the input space, two bakes that
    /// far apart must differ by at most ONE byte in any channel. One byte is the
    /// smallest difference the `Rgba8Unorm` target can represent, so a frame the
    /// cache skips cannot be a frame that would have looked different.
    ///
    /// The sweep is over both axes and both ends of each, because the terms are
    /// not symmetric: `depth` pulls the inner radius in and the edge alpha down
    /// AND desaturates the tint, while `day` only moves the two radii. A single
    /// midpoint probe would miss whichever one happens to be steepest at the ends.
    ///
    /// If this fails after a vignette retune, the constant is what moves — not the
    /// assertion.
    #[test]
    fn the_vignette_cache_never_skips_a_visible_change() {
        let view = View::for_screen(2560, 1440);
        let (cols, rows) = vignette_size(view);
        let n = (cols * rows * 4) as usize;

        let bake = |depth: f32, day: f32| {
            let mut buf = vec![0u8; n];
            bake_vignette(view, depth, day, &mut buf);
            buf
        };
        let worst = |a: &[u8], b: &[u8]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .expect("the vignette is not empty")
        };

        let mut seen_any_change = false;
        for &depth in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            for &day in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let base = bake(depth, day);

                // Both directions on both axes, clamped into range.
                for (dd, dy) in [
                    (VIGNETTE_REBAKE_EPS, 0.0),
                    (-VIGNETTE_REBAKE_EPS, 0.0),
                    (0.0, VIGNETTE_REBAKE_EPS),
                    (0.0, -VIGNETTE_REBAKE_EPS),
                    (VIGNETTE_REBAKE_EPS, VIGNETTE_REBAKE_EPS),
                ] {
                    let (d2, y2) = ((depth + dd).clamp(0.0, 1.0), (day + dy).clamp(0.0, 1.0));
                    let moved = worst(&base, &bake(d2, y2));
                    assert!(
                        moved <= 1,
                        "a skipped frame would have moved the vignette by {moved} \
                         bytes: depth {depth}->{d2}, day {day}->{y2}. \
                         VIGNETTE_REBAKE_EPS is too coarse"
                    );
                    if moved > 0 {
                        seen_any_change = true;
                    }
                }
            }
        }

        // The other half of the claim: the epsilon has to be small enough to be
        // near the resolution limit, not so small it is trivially satisfied. A
        // step this size should sometimes move a byte.
        assert!(
            seen_any_change,
            "no probe moved a single byte, so this test cannot tell a tight \
             epsilon from an absurdly small one"
        );

        // And a step well BEYOND the epsilon must be visible, or the cache would
        // be skipping real changes and this test would never notice.
        let far = worst(&bake(0.0, 1.0), &bake(0.5, 1.0));
        assert!(
            far > 1,
            "half the depth range moved the vignette by {far} bytes — either the \
             vignette barely depends on depth, or this test is measuring nothing"
        );

        // The headroom, stated. The constant sits an order of magnitude under
        // where this test starts failing (measured: 0.02 passes, 0.05 moves two
        // bytes), and that margin is the point — so a retune that made the
        // vignette 20x more sensitive to depth would still be caught here rather
        // than shipping as a cache that skips visible frames.
        let at_25x = worst(
            &bake(0.5, 0.5),
            &bake(0.5 + 25.0 * VIGNETTE_REBAKE_EPS, 0.5),
        );
        assert!(
            at_25x > 1,
            "twenty-five times the epsilon moved only {at_25x} bytes, so the \
             constant is no longer conservative — it is merely small"
        );
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
        // Laid out in LIGHT CELLS and converted, so the scene lands on the
        // solver's fixed 16x16 grid at any downscale instead of only at the one
        // the cell coordinates were written for.
        let step = LIGHT_DOWNSCALE;
        let n = 16 * step;
        let last = step - 1; // the far corner of a light cell, off the lattice
        let id = an_emitter(0.0);
        let mut g = world(n, n, 5 * step);
        for y in 6 * step..12 * step {
            for x in 2 * step..14 * step {
                g.set(x, y, EMPTY);
            }
        }
        let (tx, ty) = (4 * step + last, 9 * step + last);
        g.set(tx, ty, id); // a torch, as far off the sample lattice as it gets
        g.temp[((ty * n) + 6 * step + last) as usize] = 220; // a hot wall beside it

        let mut scan = EmitterScan::default();
        scan_emitters(&g, 0, 0, n, n, &mut scan);
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

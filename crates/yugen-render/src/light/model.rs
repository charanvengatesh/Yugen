//! What the world emits, absorbs and is shadowed by — before any of it is
//! solved or drawn.
//!
//! [`super`]'s "Emitter tables", "Shadow hue and depth", "Skylight", "Emissive
//! splats" and "Census" sections, plus the bloom and vignette tuning those
//! passes read. One argument in several parts: which materials give light, how
//! far underground turns a shadow from blue to black, how the sky reaches into
//! a cave mouth, and how a bright cell spreads into its neighbours.
//!
//! No Bevy. All of it is a function of the cell grid and the clock.

use std::sync::LazyLock;

use yugen_core::config::LIGHT_DOWNSCALE;
use yugen_core::sim::materials::{CellId, MAT_B, MAT_COUNT, MAT_EMISSIVE, MAT_G, MAT_LIGHT, MAT_R};

use crate::cellmap::MATERIAL_SLOTS;

// --- Emitter tables ----------------------------------------------------------

/// The authored `lightEmit` level a material must declare to be at full scale.
///
/// The content format authors light as an integer 0..15 — the scale the block
/// schema validates against — so this is the divisor that turns a declared level
/// into the 0..1 weight the solver works in.
pub(super) const EMIT_MAX_LEVEL: f32 = 15.0;

/// What a material that declares `emissive` but no `lightEmit` is worth.
///
/// A "this material is self-lit" fallback for blocks that predate `lightEmit`,
/// so nothing that used to glow stops glowing. Deliberately below full: a block
/// that never declared a light level never had one chosen for it.
pub(super) const EMIT_FALLBACK_GAIN: f32 = 0.8;

/// How far an emitter's cast is pushed away from grey.
///
/// Light COLOUR is the block's own colour, re-saturated. A lava cell is
/// (235,110,35) and should cast orange, not white; a crystal is (152,112,232)
/// and should cast violet. Normalising by the max channel and then pushing past
/// the block's own saturation gives the emitted hue without also carrying the
/// block's brightness, which is what the level is for.
pub(super) const EMIT_SATURATION: f32 = 1.35;

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
pub(super) static EMITTERS: LazyLock<[Emitter; MAT_COUNT]> = LazyLock::new(|| {
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
pub(super) const SHADOW_SKY: [f32; 3] = [24.0, 30.0, 48.0];

/// The colour an unlit surface falls toward in the deep, 0..255.
///
/// Five hundred cells down there is no sky to bounce and the correct answer is a
/// near-neutral that eats colour instead of adding one. Interpolating between
/// this and [`SHADOW_SKY`] by depth is the cheapest possible way to make
/// descending feel like going somewhere.
pub(super) const SHADOW_DEEP: [f32; 3] = [14.0, 11.0, 12.0];

/// How much each channel of the shadow cools at full night, 0..255.
///
/// Blue loses most, so the shadow deepens toward the same neutral the deep is
/// already at rather than merely dimming.
pub(super) const SHADOW_NIGHT_COOL: [f32; 3] = [6.0, 8.0, 10.0];

/// Cells below [`SURFACE_ANCHOR_Y`] over which depth walks from 0 to 1.
///
/// Everything depth drives — the shadow hue, the ambient floor, the vignette,
/// the underworld glow — is measured against this one ramp, so they stay in step
/// with each other as the player descends.
pub(super) const DEPTH_RANGE_CELLS: f32 = 300.0;

/// Ambient floor so unlit caves stay readable rather than pure black.
pub(super) const AMBIENT_FLOOR: f32 = 0.12;

/// How much of the ambient floor the deep takes away.
///
/// Caves keep a floor so they stay readable, but the floor itself falls with
/// depth: the deep is meant to be genuinely dark and lit by what you carry.
pub(super) const AMBIENT_DEPTH_FALL: f32 = 0.05;

// --- Skylight ----------------------------------------------------------------

/// Open sky at midnight. Moonlight, not nothing.
pub(super) const SKY_NIGHT: f32 = 0.22;

/// What full daylight adds on top of [`SKY_NIGHT`].
pub(super) const SKY_DAY_GAIN: f32 = 0.78;

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
pub(super) const OPEN_DECAY_PER_CELL: f32 = 0.997_490_6;

/// What one SIM CELL of opaque rock costs the flood.
///
/// A cave is dark because there are forty cells of stone over it, which is the
/// reason it should be dark, and that is what makes a torch matter down there
/// while daylight stays flat and bright up top.
pub(super) const SOLID_DECAY_PER_CELL: f32 = 0.861_173_5;

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
pub(super) const WALL_DECAY_PER_CELL: f32 = 0.962_0;

/// What one LIGHT CELL of open air costs the flood.
pub(super) const OPEN_DECAY: f32 = over_a_light_cell(OPEN_DECAY_PER_CELL);

/// What one LIGHT CELL of walled air costs the flood. See [`WALL_DECAY_PER_CELL`].
pub(super) const WALL_DECAY: f32 = over_a_light_cell(WALL_DECAY_PER_CELL);

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
pub(super) const SOLID_DECAY: f32 = over_a_light_cell(SOLID_DECAY_PER_CELL);

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
pub(super) const fn over_a_light_cell(per_cell: f32) -> f32 {
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
pub(super) const SEED_DECAY: f32 = 0.985;

// --- Emissive splats ---------------------------------------------------------

/// Flicker midpoint, so the swing lands on 0.76..1.0.
pub(super) const FLICKER_BASE: f32 = 0.88;
/// Flicker amplitude.
pub(super) const FLICKER_SWING: f32 = 0.12;
/// Flicker rate, radians per second.
///
/// The phase is hashed from the cell's absolute coords so neighbouring pockets
/// breathe out of step instead of pulsing as one slab.
pub(super) const FLICKER_RATE: f32 = 5.5;

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
pub(super) const SPLAT_REACH_CELLS: i32 = 4;

/// How far a strong emitter's far ring reaches, in SIM CELLS.
///
/// Twice [`SPLAT_REACH_CELLS`], as it has always been — the ring is the second
/// step out, not a different mechanism.
pub(super) const FAR_REACH_CELLS: i32 = 8;

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
pub(super) const BLUR_REACH_CELLS: i32 = 4;

/// [`SPLAT_REACH_CELLS`] in light cells.
pub(super) const SPLAT_REACH: i32 = in_light_cells(SPLAT_REACH_CELLS);

/// [`FAR_REACH_CELLS`] in light cells.
pub(super) const FAR_REACH: i32 = in_light_cells(FAR_REACH_CELLS);

/// [`BLUR_REACH_CELLS`] in light cells — the blur's radius in taps per side.
pub(super) const BLUR_REACH: i32 = in_light_cells(BLUR_REACH_CELLS);

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
pub(super) const LIGHT_SOFTNESS: f32 = 0.5;

/// [`LIGHT_SOFTNESS`], or a full bilinear upscale if the grid is coarser than
/// the art.
///
/// Snapping a sample to its texel centre reads as "the light belongs to this
/// block" only when a texel IS a block. At any [`LIGHT_DOWNSCALE`] above 1 the
/// same snap would quantise the light to a lattice nothing on screen is drawn
/// on — a hard 20 px grid at the old downscale of 4 — so the dial turns itself
/// off and the grid goes back to interpolating, which is the least-bad thing a
/// coarse grid can do. Derived rather than left to whoever changes the stride.
pub(super) const LIGHT_SNAP: f32 = if LIGHT_DOWNSCALE == 1 {
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
pub(super) const BLUR_BACK: usize = (BLUR_REACH / 2) as usize;

/// See [`BLUR_BACK`].
pub(super) const BLUR_FWD: usize = (BLUR_REACH - BLUR_REACH / 2) as usize;

/// A distance in sim cells as a whole number of light cells.
///
/// Rounded UP, so a reach authored in cells can never collapse to zero and
/// silently delete the ring or the arm it describes; the const assert below is
/// what catches a stride that does not divide it cleanly, which would be a
/// reach that quietly grew instead.
pub(super) const fn in_light_cells(cells: i32) -> i32 {
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
pub(super) const SPLAT_EDGE: f32 = 0.42;

/// The declared level above which an emitter also throws a far ring.
///
/// A level-13 lava pool throws light two light cells (40 sim cells) and a
/// level-6 mushroom cap barely leaves its own. Before the reach scaled with the
/// declared level, every emitter had the same one-cell reach whatever it
/// declared, which is why a magma chamber and a glowing mushroom lit the same
/// volume.
pub(super) const FAR_SPLAT_LEVEL: f32 = 0.55;

/// What the far ring gets, relative to the splat's centre.
pub(super) const FAR_SPLAT_GAIN: f32 = 0.16;

/// How much of a splat's strength goes into the COLOUR grids.
///
/// Below the scalar gain on purpose: the cast should tint the surroundings, not
/// repaint them.
pub(super) const COLOUR_GAIN: f32 = 0.62;

/// What the four neighbours of a colour splat get, relative to its centre.
pub(super) const COLOUR_EDGE: f32 = 0.45;

/// Cell temperature below which residual heat casts no light at all.
///
/// Only the hot tail contributes, and only weakly, so this reads as warmth
/// bleeding out of a pocket rather than as a second light source.
pub(super) const HEAT_THRESHOLD: f32 = 70.0;

/// Temperature span from [`HEAT_THRESHOLD`] to a full-strength heat glow.
///
/// 70 + 185 = 255, the top of the temperature plane, so the hottest possible
/// rock lands exactly at [`HEAT_GAIN`] and nothing has to clip.
pub(super) const HEAT_RANGE: f32 = 185.0;

/// What the hottest possible non-emitting cell is worth as a light source.
pub(super) const HEAT_GAIN: f32 = 0.3;

/// What the four neighbours of a heat splat get, relative to its centre.
pub(super) const HEAT_EDGE: f32 = 0.5;

/// The hue residual heat casts, 0..1 per channel.
///
/// Warm for the same reason lava is: heat in rock glows red before it glows at
/// all.
pub(super) const HEAT_RGB: [f32; 3] = [0.5, 0.16, 0.04];

/// Cap on the hot list the ambience layer seeds its embers from.
pub(super) const HOT_MAX: usize = 64;

/// The eight offsets a strong emitter's far ring lands on.
///
/// Axial at [`FAR_REACH`] and diagonal at [`SPLAT_REACH`], which is what the
/// original's `(±2, 0)` and `(±1, ±1)` were saying: the diagonals sit one step
/// out and the axes two, so the ring is a rough circle rather than a square.
pub(super) const FAR_RING: [(i32, i32); 8] = [
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
pub(super) const FAR_RING_AXIAL: [(i32, i32); 4] = [
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
pub(super) const CENSUS_GROUP: i32 = LIGHT_DOWNSCALE;

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
pub(super) const CENSUS_NEEDED: bool = LIGHT_DOWNSCALE > 1;

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
pub(super) const BLOOM_RADIUS_CELLS: i32 = 3;

/// Materials the bloom's emit table has room for.
///
/// [`crate::cellmap::MATERIAL_SLOTS`] and not a second 64 of this module's own:
/// the table below is indexed by a texel of `cellmap`'s id texture, so the two
/// shaders must agree on the padding or a block added to `content/` would light
/// one pass and not the other.
pub(super) const BLOOM_EMIT_SLOTS: usize = MATERIAL_SLOTS;

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
pub(super) const BLOOM_LEVEL_KNEE_LO: f32 = 0.25;

/// Emitted level at which a material contributes its colour in full.
///
/// The knee between it and [`BLOOM_LEVEL_KNEE_LO`] is smooth rather than a step
/// so that a level moving by one authored point cannot pop a whole cavern's glow
/// into existence. Against the content set as it stands: mushroom cap (6/15)
/// lands at 0.22, crystal (7/15) at 0.40, brazier (10/15) at 0.93, and lava,
/// fire and the torch (13..15) are all at 1.
pub(super) const BLOOM_LEVEL_KNEE_HI: f32 = 0.75;

/// Standard deviation of the bloom's gather kernel, in cells.
///
/// Half [`BLOOM_RADIUS_CELLS`], which is the usual place to truncate a Gaussian:
/// the tap at the rim is `exp(-2)` of the centre, so the kernel is ~98% of the
/// untruncated one and the seam at the edge of the gather is invisible.
pub(super) const BLOOM_SIGMA_CELLS: f32 = 1.5;

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
pub(super) const BLOOM_INTENSITY: f32 = 0.3;

/// Where the bloom's gather pass sits in camera order.
///
/// BEFORE `crate::lowres`' world camera at `-1`, so the texture the composite
/// quad samples was written by this frame's gather and not by last frame's. That
/// ordering is the whole reason the bloom can be a same-frame image pass without
/// a custom render-graph node: two cameras with different targets and different
/// orders are two render passes, and Bevy runs them in the order given.
pub(super) const BLOOM_CAMERA_ORDER: isize = -2;

// --- Vignette and washes -----------------------------------------------------

/// View px per vignette sample.
///
/// The vignette is a smooth radial ramp with no detail in it, so it is baked at
/// a sixteenth of the view's resolution and upscaled by the same linear sampler
/// the light grid uses. At 500x250 logical that is a 33x17 texture per frame.
pub(super) const VIGNETTE_CELL: i32 = 16;

/// Vignette inner radius as a fraction of the view's short axis, at the surface.
pub(super) const VIGNETTE_INNER: f32 = 0.35;
/// How much depth shrinks the bright core.
pub(super) const VIGNETTE_INNER_DEPTH: f32 = 0.1;
/// How much night shrinks the bright core.
pub(super) const VIGNETTE_INNER_NIGHT: f32 = 0.06;
/// Vignette outer radius as a fraction of the view's long axis.
pub(super) const VIGNETTE_OUTER: f32 = 0.72;
/// Edge darkness at the surface in daylight.
pub(super) const VIGNETTE_EDGE: f32 = 0.55;
/// How much depth lightens the edge — the deep is dark enough already.
pub(super) const VIGNETTE_EDGE_DEPTH: f32 = 0.25;
/// How much night closes the edges in.
pub(super) const VIGNETTE_EDGE_NIGHT: f32 = 0.1;
/// The colour the vignette's edge falls toward at the surface, 0..255.
pub(super) const VIGNETTE_RGB: [f32; 3] = [40.0, 44.0, 64.0];
/// How much depth takes off the vignette's red and green, 0..255.
///
/// Blue is untouched, so the frame edge goes bluer as it goes deeper.
pub(super) const VIGNETTE_RGB_DEPTH: [f32; 3] = [20.0, 20.0, 0.0];

/// Depth at which the underworld glow starts to ramp in, as a 0..1 fraction of
/// the depth range.
///
/// Below the magma line the rock itself is hot. A very faint additive floor
/// wash, over the last third of the depth range only, so the deep reads as lit
/// from below rather than merely dark.
///
/// **Not to be renamed back to `UNDERWORLD_DEPTH`.** That is the name of
/// `yugen_core::config::worldgen::UNDERWORLD_DEPTH`, which is an `i32` count
/// of CELLS below the surface, not a normalised fraction. This module already
/// does `use yugen_core::config::{…}`; the day that becomes a glob import, a
/// local of the same name would silently win and substitute 0.62 for 470, with
/// no error anywhere. `cargo xtask tuning` gates that collision, and this
/// constant is the one that made the rule earn its keep.
///
/// The TypeScript had the 0.62 as a bare literal, so the clash is a port
/// regression rather than something inherited.
pub(super) const UNDERWORLD_GLOW_DEPTH: f32 = 0.62;
/// Strength of the underworld glow at full depth.
///
/// **This is coupled to how bright the PALETTE is, and it is not obvious that it
/// would be.** The glow is a CONSTANT addition, so what it does to a frame
/// depends entirely on what it is added to. Halve the world's base luma and the
/// same wash is twice as loud in relative terms; every material in the deep
/// converges on the same red-brown and the distinctions authored into them stop
/// being visible.
///
/// That is exactly what happened. `content/PALETTE.md` took the palette's mean
/// luma from 128.7 to 95.5, and the ore chamber — the frame that exists to watch
/// ore-against-rock legibility, and the specific failure the 0.6 -> 0.05
/// `BIOME_AMBIENT_ALPHA` retune was about — lost 814 distinct colours and rose
/// from 7.7% to 9.6% single-colour dominance. The gold veins were washing out.
///
/// Measured on that frame, sweeping this alone:
///
/// ```text
///   alpha    0.20    0.14    0.10    0.06    0.00
///   distinct 2 797   3 259   3 698   4 385   6 496
///   stddev   23.18   24.40   25.45   26.80   30.47
/// ```
///
/// 0.10 is where the chamber passes its own PRE-repaint numbers (3 698 against
/// 3 611 distinct, 25.45 against 23.03 stddev) while the deep still visibly
/// reads as lit from below, which is the whole reason this constant exists. 0.00
/// scores better on every number and is wrong: it is not a dark cave any more,
/// it is a flat one. Judge this on the picture, underground, as with everything
/// else in this module.
pub(super) const UNDERWORLD_ALPHA: f32 = 0.10;
/// The underworld glow's colour, 0..255.
pub(super) const UNDERWORLD_RGB: [f32; 3] = [96.0, 30.0, 14.0];

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
pub(super) const BIOME_AMBIENT_ALPHA: f32 = 0.05;

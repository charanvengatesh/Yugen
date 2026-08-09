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

mod material;
mod model;
mod passes;
mod solver;

pub use material::*;
pub use model::*;
pub use passes::*;
pub use solver::*;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{RenderTarget, ScalingMode};
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::shader::Shader;
use bevy::sprite_render::{Material2d, Material2dPlugin};

use yugen_core::config::{CELL_SIZE, SEED, View};

use crate::cellmap::CellMap;

use crate::daynight::WorldClock;
use crate::lowres::{LowResTarget, WORLD_LAYERS};
use crate::world::{SimWorld, WorldFocus};

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
/// `atmo.ambient` from `yugen_core::sim::biomes`, blended over the same
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

impl LightPass {
    /// Solved light at a point in world px, or `None` if it is off the grid.
    ///
    /// The grid's origin is private and should stay that way — it is recomputed
    /// every frame from the focus and nothing outside this module has any
    /// business doing that arithmetic. This is the one question anybody actually
    /// wants to ask of it, and answering it here means the caller cannot get the
    /// stride or the margin wrong. `crate::debug`'s F3 panel is the only caller.
    ///
    /// `None` rather than 0 for off-grid, because "no light here" and "outside
    /// what was solved" are different answers and a panel that printed 0.000 for
    /// the second would be quietly wrong.
    pub fn light_at_world(&self, x: f32, y: f32) -> Option<f32> {
        let stride = light_stride_px() as f32;
        let lx = ((x - self.origin.x) / stride).floor() as i32;
        let ly = ((y - self.origin.y) / stride).floor() as i32;
        (lx >= 0 && ly >= 0 && lx < self.grid.cols() && ly < self.grid.rows())
            .then(|| self.grid.light_at(lx, ly))
    }
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
/// Deliberately NOT added to [`crate::YugenRenderPlugin`] here.
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
            label: Some("yugen_bloom"),
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
    use yugen_core::config::{CELL_SIZE, LIGHT_DOWNSCALE, SURFACE_ANCHOR_Y, View, WorldScale};
    use yugen_core::sim::coords::WorldCell;
    use yugen_core::sim::grid::CellGrid;
    use yugen_core::sim::materials::block;
    use yugen_core::sim::materials::{CellId, EMPTY, MAT_COUNT};

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
        let before = grid
            .heights
            .surface_row_at(&grid.noise, 0, None, WorldScale::LIVE);

        // A seed it was not built with. `world_noise` is a pure function of it,
        // so a different seed is a different world with a different surface.
        grid.follow_seed(SEED + 1);
        assert_eq!(grid.seed(), SEED + 1, "the grid did not adopt the new seed");
        let after = grid
            .heights
            .surface_row_at(&grid.noise, 0, None, WorldScale::LIVE);
        assert_ne!(
            before, after,
            "the surface line did not move, so either the noise or the heightmap \
             memo survived the seed change and the flood is lighting a world that \
             is no longer there"
        );

        // Idempotent, because it runs every frame: a re-seed to the value it
        // already holds must not throw the memo away and pay for it again.
        grid.follow_seed(SEED + 1);
        assert_eq!(
            grid.heights
                .surface_row_at(&grid.noise, 0, None, WorldScale::LIVE),
            after
        );

        // And it goes back, so this is a mirror of the world and not a latch.
        grid.follow_seed(SEED);
        assert_eq!(
            grid.heights
                .surface_row_at(&grid.noise, 0, None, WorldScale::LIVE),
            before
        );
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

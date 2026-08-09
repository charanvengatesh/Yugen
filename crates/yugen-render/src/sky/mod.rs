//! The parallax backdrop: gradient sky, twilight band, starfield, sun and moon,
//! and two hill ridges.
//!
//! Ported from `src/render/sky.ts`.
//!
//! Everything here is drawn in VIEW space — the TypeScript's screen space, with
//! the origin at the view's top-left and +y DOWN — and sits behind the world so
//! everything else paints over it. Star and hill positions are derived
//! deterministically from their index, so the layout is stable frame to frame and
//! costs nothing to keep.
//!
//! # Where the daytime palette comes from
//!
//! [`BiomeAtmosphere`](yugen_core::sim::biomes::BiomeAtmosphere) declares
//! NIGHT-side colours only, and that file belongs to the sim. So the daytime sky
//! is DERIVED from each biome's own horizon hue — `sky_bottom` is the brightest,
//! most saturated colour a biome declares — pushed toward a neutral daylight
//! target. Desert therefore stays warm and Glacier stays cold at noon without a
//! second palette existing anywhere, and because the input is
//! [`resolve_atmosphere`]'s already-blended weight set, crossing a biome boundary
//! crossfades at every hour of the day.
//!
//! # How it is drawn here
//!
//! The TypeScript had one `CanvasRenderingContext2D` and did five passes into it.
//! There is no immediate-mode context here, so each pass is one long-lived entity
//! parented to the world camera, sorted by z in the order the canvas painted them:
//!
//! | z | Entity | Canvas pass |
//! |---|---|---|
//! | [`GRADIENT_Z`] | a `cols x rows` image on a stretched sprite | the base gradient AND the twilight band |
//! | [`STARS_Z`] | one vertex-coloured mesh, 90 quads | the starfield |
//! | [`DISCS_Z`] | one vertex-coloured mesh, two block grids | sun and moon |
//! | [`RIDGES_Z`] | one vertex-coloured mesh, two block strips | the hill silhouettes |
//!
//! Parenting to the camera rather than following it in a system is what makes the
//! backdrop screen-locked with no ordering rule to get wrong: transform
//! propagation runs after every writer of the camera's transform, so the sky can
//! never be a frame behind the view it is meant to fill. It also inherits the
//! camera's pixel snap for free, which is what keeps a 1px star from shimmering.
//!
//! # The backdrop is drawn on the world's own pixel grid
//!
//! THIS IS A DELIBERATE DEPARTURE FROM THE ORIGINAL AND IT IS NOT A BUG. The
//! TypeScript drew this backdrop the way Canvas2D wants to be drawn: a
//! `createLinearGradient` ramp, `arc()` discs with a `createRadialGradient` falloff,
//! and a `lineTo` polyline for the hills. All three are SMOOTH — the gradient
//! resolves to one colour per device row, the discs are round with a continuous
//! radial falloff, and the hills are straight lines at whatever slope the sines
//! ask for.
//!
//! The world in front of them is not. Cells rasterise at
//! [`CELL_SIZE`] into a 640x400 buffer that is then upscaled with NEAREST, so
//! everything the player looks at has a hard 5px feature size. A smooth ramp
//! directly behind blocky terrain does not read as the same material; it reads as
//! a photograph someone pasted a sprite onto. So every backdrop element here is
//! rasterised onto [`SKY_PIXEL_PX`] blocks, which is [`CELL_SIZE`]: the gradient
//! becomes a grid of flat blocks rather than a ramp, the discs become block
//! circles rather than ring fans, and the ridges become block columns rather than
//! a polyline.
//!
//! The stars needed none of this. They were already rounding to a whole view pixel
//! — see [`place_stars`] — for precisely the reason everything else now does, and
//! they stay 1px, because a star is a point of light and a 5px star is a planet.
//!
//! Setting [`SKY_PIXEL_PX`] to 1 restores the smooth original almost exactly,
//! which is the intended way to look at what this bought.
//!
//! # What the port changed
//!
//! **The base gradient and the twilight band are one image.** The canvas painted
//! a two-stop vertical gradient and then a second, `lighter`-composited gradient
//! over the whole rect. Both are functions of y ALONE, so the two composite
//! exactly into one colour per row — [`sky_texel`] — and the row set is a small
//! texture stretched across the view. The arithmetic is identical to the canvas's
//! because it is done in sRGB, on the same premultiplied stops; only the SAMPLE
//! POSITIONS changed, from one per view row to one per [`SKY_PIXEL_PX`] block,
//! ordered-dithered within the block by [`BAYER`].
//!
//! **The discs are geometry, not a baked sprite.** `Sky.disc` prebaked a soft
//! radial sprite into an offscreen canvas and blitted it. A canvas radial gradient
//! IS a piecewise-linear ramp between its stops — see [`DISC_STOPS`] — so here
//! that ramp is evaluated on the CPU, once per block, by [`ring_at`]. It removes
//! the bake, the two offscreen canvases, and the blit's `x`/`y` culling test. It
//! also puts the ramp's interpolation back in sRGB where the canvas did it: the
//! ring fan this replaced handed the stops to the GPU as vertex colours and got
//! LINEAR-light interpolation between them, which moved the middle of a soft glow
//! by about a value step.
//!
//! **Additive is a blend state, not a composite op.** `globalCompositeOperation =
//! "lighter"` has no equivalent in Bevy's 2D materials — [`AlphaMode2d`] offers
//! opaque, mask and blend and nothing else. [`AdditiveMaterial`] is that missing
//! mode: an empty material on the default mesh2d shader whose only job is to
//! override the blend state in [`Material2d::specialize`]. `SrcAlpha, One, Add` is
//! exactly what `lighter` does. It carries no bindings and no shader of its own,
//! which is why it can be a dozen lines rather than a WGSL file.
//!
//! **The clock is ticked here.** [`DayNight`] has no plugin of its own and the sky
//! is its first and main consumer, so [`SkyPlugin`] owns the tick. If a day/night
//! plugin ever lands, this system moves to it — two clocks ticking one resource
//! would run the world at double speed, so there must only ever be one.
//!
//! # What the port dropped on the way in
//!
//! - **The frame scratch.** `TOP`/`BOT` were module-level arrays recomputed in
//!   place so the draw never allocated. A [`Gradient`] is six floats returned by
//!   value; there is nothing to reuse.
//! - **`performance.now()`.** Two passes read the wall clock directly and scaled
//!   milliseconds. Both now take [`Time::elapsed_secs`], which is the same number
//!   in seconds and, unlike a wall clock, stops when the app does.
//! - **The disc culling test.** `x < -r*2 || x > VIEW_W + r*2` skipped a blit that
//!   would land off-screen. Geometry off the edge of a viewport costs a clipped
//!   triangle, so the test would buy nothing back.

mod mesh;
mod model;

pub use mesh::AdditiveMaterial;
pub(crate) use mesh::{VertexBuf, dynamic_mesh, linear};
pub(crate) use model::wrap;
use model::*;
pub use model::{
    Disc, Gradient, HorizonGlow, MOON, Rgb, Ridge, STAR_COUNT, SUN, Star, StarField, depth_at,
    gradient, hash, horizon_glow, sky_texel,
};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::system::SystemParam;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::sprite_render::{AlphaMode2d, Material2dPlugin};

use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use yugen_core::config::{CELL_SIZE, SEED, View, WorldScale};
use yugen_core::sim::biomes::{ResolvedAtmosphere, resolve_atmosphere};
use yugen_core::sim::noise::Noise;
use yugen_core::sim::worldgen::world_noise;

use crate::daynight::{DayPhase, WorldClock};
use crate::lowres::{LowResTarget, WORLD_LAYERS, WorldCamera};
use crate::world::WorldFocus;

// ---------------------------------------------------------------------------
// The plugin
// ---------------------------------------------------------------------------

/// Where the base gradient sits: behind everything, the weather included.
pub const GRADIENT_Z: f32 = -100.0;

/// Stars, over the gradient.
pub const STARS_Z: f32 = -99.0;

/// Sun and moon, over the stars.
pub const DISCS_Z: f32 = -98.0;

/// The hill silhouettes, over everything else in the backdrop.
pub const RIDGES_Z: f32 = -97.0;

/// The world's noise field, built once.
///
/// [`resolve_atmosphere`] needs it every frame and building one is not free; the
/// TypeScript called `worldNoise(SEED)` inside its draw and relied on that
/// function memoising.
#[derive(Resource)]
pub struct SkyNoise(Noise);

/// The blended atmosphere at the camera, and how deep the camera is.
///
/// Published as a resource because it is a per-frame value with two consumers —
/// this module and [`crate::weather`] — and `Game.ts` resolved it once and handed
/// the same struct to both for the same reason. Sampling it twice would mean
/// running the biome mix twice a frame for one answer.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Atmosphere {
    /// Backdrop colours, blended across the biome boundary at the camera.
    pub resolved: ResolvedAtmosphere,
    /// 0 at the surface, 1 fully underground. See [`depth_at`].
    pub depth: f32,
}

/// The star seeds and the gradient's texture.
#[derive(Resource)]
pub struct Backdrop {
    /// sRGB, one texel per [`SKY_PIXEL_PX`] block, stretched over the view with a
    /// nearest sampler so one texel is exactly one block.
    pub gradient: Handle<Image>,
    /// The fixed star set.
    pub stars: StarField,
}

/// The sprite the base gradient is stretched over.
#[derive(Component)]
pub struct SkyGradient;

/// The mesh every star is a quad in.
#[derive(Component)]
pub struct SkyStars;

/// The mesh the sun and the moon are ring fans in.
#[derive(Component)]
pub struct SkyDiscs;

/// The mesh both hill ridges are strips in.
#[derive(Component)]
pub struct SkyRidges;

/// Everything a backdrop pass reads.
///
/// One [`SystemParam`] rather than five parameters on each of four systems: past
/// seven arguments a system stops being readable, and every pass here wants the
/// same five reads.
#[derive(SystemParam)]
pub struct Frame<'w> {
    /// The blended atmosphere and the camera's depth.
    pub atmo: Res<'w, Atmosphere>,
    /// The world clock.
    pub clock: Res<'w, WorldClock>,
    /// The low-res buffer, for its [`View`].
    pub target: Res<'w, LowResTarget>,
    /// The view centre in world px, +y DOWN.
    pub focus: Res<'w, WorldFocus>,
    /// The animation clock.
    pub time: Res<'w, Time>,
}

impl Frame<'_> {
    /// The logical buffer this frame is drawn into.
    pub fn view(&self) -> View {
        self.target.view
    }

    /// The world clock, sampled.
    pub fn phase(&self) -> DayPhase {
        self.clock.0.phase()
    }

    /// The view's TOP-LEFT in world px — the TypeScript's `camX, camY`.
    ///
    /// [`WorldFocus`] is the view CENTRE, so this is where the two conventions
    /// meet. Every parallax term in the backdrop is a fraction of this, and
    /// passing the centre instead would slide the whole thing by half a view's
    /// worth of parallax.
    pub fn cam(&self) -> Vec2 {
        let view = self.view();
        Vec2::new(
            self.focus.x - view.w as f32 * 0.5,
            self.focus.y - view.h as f32 * 0.5,
        )
    }

    /// Seconds since startup, for the twinkle and the drift.
    pub fn seconds(&self) -> f32 {
        self.time.elapsed_secs()
    }
}

/// The backdrop: the clock that drives it, the atmosphere it reads, and the four
/// passes that put it on screen.
pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        // [`crate::weather`] draws into the same layer with the same material and
        // either plugin may be added without the other, so whichever gets here
        // first registers it. Adding a plugin twice is a panic, not a no-op.
        if !app.is_plugin_added::<Material2dPlugin<AdditiveMaterial>>() {
            app.add_plugins(Material2dPlugin::<AdditiveMaterial>::default());
        }

        let noise = world_noise(SEED);
        let resolved = resolve_atmosphere(&noise, 0, WorldScale::LIVE);

        app.insert_resource(SkyNoise(noise))
            .insert_resource(Atmosphere {
                resolved,
                depth: 0.0,
            })
            // After every `Startup`, so the world camera this parents itself to
            // already exists: `LowResPlugin` spawns it there, and there is no
            // ordering label between two plugins' startup systems to hang this on.
            .add_systems(PostStartup, setup)
            .add_systems(
                Update,
                (
                    sample_atmosphere,
                    (paint_gradient, place_stars, place_discs, build_ridges),
                )
                    .chain()
                    .run_if(resource_exists::<Backdrop>),
            );
    }
}

/// Spawn the four backdrop passes as children of the world camera.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut blend: ResMut<Assets<ColorMaterial>>,
    mut additive: ResMut<Assets<AdditiveMaterial>>,
    camera: Single<Entity, With<WorldCamera>>,
) {
    let camera = *camera;
    let gradient = images.add(gradient_image(1, 1));

    commands.spawn((
        Sprite {
            image: gradient.clone(),
            // Resized to the view on the first paint; this only avoids one frame
            // of a one-pixel sky.
            custom_size: Some(Vec2::ONE),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, GRADIENT_Z),
        SkyGradient,
        ChildOf(camera),
        WORLD_LAYERS,
    ));

    // White and fully blending: all the colour is on the vertices, so one handle
    // serves every source-over pass in the backdrop and never has to be written.
    let source_over = blend.add(ColorMaterial {
        color: Color::WHITE,
        alpha_mode: AlphaMode2d::Blend,
        ..default()
    });

    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(source_over.clone()),
        Transform::from_xyz(0.0, 0.0, STARS_Z),
        SkyStars,
        // The geometry is rewritten in place every frame, so the bounding box Bevy
        // computed when it first saw the handle is stale from the second frame on.
        // Culling against it would blink the backdrop out.
        NoFrustumCulling,
        ChildOf(camera),
        WORLD_LAYERS,
    ));

    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(additive.add(AdditiveMaterial {})),
        Transform::from_xyz(0.0, 0.0, DISCS_Z),
        SkyDiscs,
        NoFrustumCulling,
        ChildOf(camera),
        WORLD_LAYERS,
    ));

    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(source_over),
        Transform::from_xyz(0.0, 0.0, RIDGES_Z),
        SkyRidges,
        NoFrustumCulling,
        ChildOf(camera),
        WORLD_LAYERS,
    ));

    commands.insert_resource(Backdrop {
        gradient,
        stars: StarField::new(),
    });
}

/// A `cols x rows` sRGB image: one texel per [`SKY_PIXEL_PX`] block of sky.
///
/// The sampler is pinned to nearest here rather than inherited from
/// `ImagePlugin::default_nearest()`. The binary does set that default, but this
/// texture is the one place in the backdrop where a linear sampler would not look
/// like a bug — it would look like the smooth gradient this file used to draw,
/// silently undoing the whole point of [`SKY_PIXEL_PX`]. Say it out loud instead.
fn gradient_image(cols: u32, rows: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: cols.max(1),
            height: rows.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    image.sampler = ImageSampler::nearest();
    image
}

/// Resolve the biome atmosphere and the camera's depth for this frame.
fn sample_atmosphere(focus: Res<WorldFocus>, noise: Res<SkyNoise>, mut atmo: ResMut<Atmosphere>) {
    let column = (focus.x / CELL_SIZE as f32).floor() as i32;
    atmo.resolved = resolve_atmosphere(&noise.0, column, WorldScale::LIVE);
    atmo.depth = depth_at(focus.y);
}

/// Rewrite the gradient's block grid and stretch it over the view.
///
/// The texture is one texel per block and the sprite is sized to a WHOLE number
/// of blocks, which is what makes every block exactly [`SKY_PIXEL_PX`] across
/// under the nearest sampler. Sizing it to the view instead would divide `view.w`
/// by a column count that does not divide it and scatter 4px and 6px blocks
/// through the sky. The overhang — under one block on each axis — falls outside
/// the view and is clipped.
fn paint_gradient(
    frame: Frame,
    backdrop: Res<Backdrop>,
    mut images: ResMut<Assets<Image>>,
    mut sprite: Single<&mut Sprite, With<SkyGradient>>,
) {
    let view = frame.view();
    let block = block_px();
    let cols = (view.w.max(1) as u32).div_ceil(block as u32) as usize;
    let rows = (view.h.max(1) as u32).div_ceil(block as u32) as usize;
    sprite.custom_size = Some(Vec2::new(cols as f32 * block, rows as f32 * block));

    let Some(mut image) = images.get_mut(&backdrop.gradient) else {
        return;
    };
    let size = image.texture_descriptor.size;
    if size.width as usize != cols || size.height as usize != rows {
        image.resize(Extent3d {
            width: cols as u32,
            height: rows as u32,
            depth_or_array_layers: 1,
        });
    }
    let Some(data) = image.data.as_mut() else {
        return;
    };

    let phase = frame.phase();
    let gradient = gradient(&frame.atmo.resolved, frame.atmo.depth, phase);
    let glow = horizon_glow(phase, frame.atmo.depth, view.h as f32);
    for row in 0..rows {
        // Every colour in a block row is a function of the dither phase alone, and
        // [`BAYER`] has only `BAYER_N` of those. So the ramp is evaluated four
        // times a row and the row is filled by repeating them, which keeps this at
        // roughly the `view.h` gradient evaluations a frame it cost when it was a
        // one-texel-wide strip rather than the `cols * rows` the grid implies.
        let mut phases = [[0u8; 4]; BAYER_N];
        for (bx, texel) in phases.iter_mut().enumerate() {
            let y = (row as f32 + block_sample(bx as i32, row as i32)) * block;
            *texel = sky_texel(&gradient, glow.as_ref(), y, view.h as f32);
        }
        for col in 0..cols {
            let at = (row * cols + col) * 4;
            data[at..at + 4].copy_from_slice(&phases[col % BAYER_N]);
        }
    }
}

/// Rebuild the starfield.
fn place_stars(
    frame: Frame,
    backdrop: Res<Backdrop>,
    mesh: Single<&Mesh2d, With<SkyStars>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buf: Local<VertexBuf>,
) {
    let Some(mut mesh) = meshes.get_mut(&mesh.0) else {
        return;
    };
    buf.clear();

    let visibility = StarField::visibility(frame.phase(), frame.atmo.depth);
    if visibility > STAR_CUTOFF {
        let view = frame.view();
        let cam = frame.cam();
        let seconds = frame.seconds();
        let color = rgb32(frame.atmo.resolved.star);
        for i in 0..STAR_COUNT {
            let star = backdrop.stars.star(i, cam, view, seconds, visibility);
            // Rounded to a whole view pixel: a canvas `fillRect` at a fractional
            // coordinate antialiases a 1px star across two, which at this buffer
            // size is a smear rather than a star.
            //
            // This one line is the oldest thing in the file and it is the argument
            // the rest of the backdrop now follows — see the module header. It is
            // also the one element that stays at 1px rather than moving to
            // `SKY_PIXEL_PX`: a star is a point of light, and a 5px one is a
            // planet.
            buf.rect(
                star.x.round(),
                star.y.round(),
                1.0,
                1.0,
                view,
                linear(color, star.alpha),
            );
        }
    }

    buf.write(&mut mesh);
}

/// Rebuild the sun and the moon.
fn place_discs(
    frame: Frame,
    mesh: Single<&Mesh2d, With<SkyDiscs>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buf: Local<VertexBuf>,
) {
    let Some(mut mesh) = meshes.get_mut(&mesh.0) else {
        return;
    };
    buf.clear();

    let view = frame.view();
    let phase = frame.phase();
    let open = 1.0 - frame.atmo.depth;
    for (disc, fx, fy, alpha) in [
        (SUN, phase.sun_x, phase.sun_y, phase.sun_a * open),
        (
            MOON,
            phase.moon_x,
            phase.moon_y,
            phase.moon_a * open * MOON_DIM,
        ),
    ] {
        if alpha <= DISC_CUTOFF {
            continue;
        }
        let centre = Vec2::new(fx * view.w as f32, fy * view.h as f32);
        push_disc(&mut buf, &disc, centre, alpha, view);
    }

    buf.write(&mut mesh);
}

/// One disc as a square block grid, centre outward.
///
/// A block circle, not a polygon. The disc used to be a 24-segment ring fan with
/// the stop colours on its vertices, which is round to within half a view pixel
/// and has a smooth radial falloff — a genuinely circular circle in a world made
/// of squares. Here each [`SKY_PIXEL_PX`] block inside the radius is one flat
/// quad whose colour is [`ring_at`] read at the block's own distance from the
/// centre, so the rim staircases and the halo falls off in visible steps, exactly
/// like the terrain does.
///
/// # The two snaps, and why they are different
///
/// The centre is snapped to a whole VIEW pixel and the lattice is then built out
/// from it, rather than the disc being pinned to the same global block grid the
/// gradient uses. Both were tried on paper and the global grid loses: a sun
/// pinned to a 5px grid crosses the view in 128 discrete hops over a 300-second
/// day, which is one visible jolt every two and a half seconds, and a jolting sun
/// is a worse artefact than a sun whose blocks are half a block out of phase with
/// the sky's. Nobody can see the phase. Everybody can see the jolt.
///
/// What the centre snap DOES buy is that every block edge lands on an integer
/// view pixel, so the blocks are all exactly [`SKY_PIXEL_PX`] wide and none of
/// them shimmers as the disc drifts. It is the same reasoning, and the same
/// `round`, that [`place_stars`] has always applied to a 1px star.
fn push_disc(buf: &mut VertexBuf, disc: &Disc, centre: Vec2, alpha: f32, view: View) {
    let rings = disc.rings(alpha);
    let block = block_px();
    let cx = centre.x.round();
    let cy = centre.y.round();
    // Indices run `-n..n`, so the centre is a block CORNER and the disc comes out
    // symmetric about it on both axes. Centring a block on the centre instead
    // would make the diameter an odd number of blocks and give the circle a spine.
    let n = (disc.radius / block).ceil() as i32;

    for row in -n..n {
        for col in -n..n {
            let dx = (col as f32 + 0.5) * block;
            let dy = (row as f32 + 0.5) * block;
            // The same dither as the gradient, on the radius instead of on y. The
            // halo's alpha falls by about a twentieth per block, which without
            // this reads as five concentric rings rather than one glow.
            let r = dx.hypot(dy) + block * (block_sample(col, row) - 0.5);
            let Some((color, a)) = ring_at(&rings, r) else {
                continue;
            };
            buf.rect(
                cx + col as f32 * block,
                cy + row as f32 * block,
                block,
                block,
                view,
                linear(color, a),
            );
        }
    }
}

/// Rebuild both hill ridges as columns of blocks.
///
/// The canvas walked the silhouette with `lineTo` every 20px and let the fill
/// draw whatever slope fell out, so a ridge was a polyline with smooth diagonal
/// edges. Here it is a run of [`SKY_PIXEL_PX`]-wide columns whose tops are
/// snapped to the same lattice — a staircase, which is what a hill drawn out of
/// cells looks like.
///
/// # The lattice lives in the ridge's space, not the screen's
///
/// This is the part that is easy to get wrong and looks terrible when you do.
/// Sampling at fixed SCREEN columns and snapping the height there means each
/// column's height creeps with the camera and pops to the next lattice step at
/// its own moment: the silhouette boils. So the columns are laid out in the
/// RIDGE's own space, where the height of column `j` is a constant, and the whole
/// strip is then translated onto the screen by a whole number of view px. The
/// silhouette is rigid and slides; no column ever changes height.
///
/// That is also why the parallax offset is rounded and `height_at` is called with
/// a camera of zero: the rounded offset IS the parallax, and passing it twice
/// would apply it twice.
fn build_ridges(
    frame: Frame,
    mesh: Single<&Mesh2d, With<SkyRidges>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buf: Local<VertexBuf>,
) {
    let Some(mut mesh) = meshes.get_mut(&mesh.0) else {
        return;
    };
    buf.clear();

    let view = frame.view();
    let w = view.w as f32;
    let h = view.h as f32;
    let day = frame.phase().day;
    let cam_x = frame.cam().x;
    let hill = rgb32(frame.atmo.resolved.hill);

    let block = block_px();
    for ridge in &RIDGES {
        let color = linear(ridge.tint(hill, day), 1.0);
        let shift = ridge.scroll(cam_x);
        // One column past each edge, so the partial column a shift that is not a
        // whole number of blocks leaves at the left never opens a gap.
        let first = (shift / block).floor() as i32;
        let last = ((shift + w) / block).ceil() as i32;
        for j in first..last {
            let ridge_x = j as f32 * block;
            let top = ridge.column_top(ridge_x, h);
            if top >= h {
                continue;
            }
            buf.rect(ridge_x - shift, top, block, h - top, view, color);
        }
    }

    buf.write(&mut mesh);
}

/// A sim-side colour as this module's.
///
/// `yugen-core` carries backdrop colours as `f64` because the biome blend that
/// produces them shares its arithmetic with the world generator, which is `f64`
/// throughout for parity. Nothing downstream of here needs that precision.
pub(crate) fn rgb32(c: yugen_core::sim::biomes::Rgb) -> Rgb {
    [c[0] as f32, c[1] as f32, c[2] as f32]
}

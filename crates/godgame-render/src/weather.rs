//! Per-biome weather particles: dust, snow, spores and embers.
//!
//! Ported from `src/render/weather.ts`.
//!
//! Drawn in VIEW space over the sky and under the world, on the same terms
//! [`crate::sky`] is drawn: one fixed, hash-positioned particle set is reused for
//! every kind, and the kind only changes how each particle moves and looks. There
//! is no per-frame allocation and the layout is stable frame to frame. Positions
//! animate off a clock plus a fraction of the camera (parallax), then wrap into
//! one view tile so a finite set blankets the whole screen.
//!
//! # Blending across a boundary
//!
//! [`resolve_atmosphere`](godgame_core::sim::biomes::resolve_atmosphere) hands
//! back only the DOMINANT biome's weather kind, so driving this layer off that
//! alone makes snow become dust between one step and the next. Instead it takes a
//! full weather-weight vector and DITHERS the particle set over it: every particle
//! owns a stable selector in `[0, 1)` and belongs to whichever kind's bucket that
//! selector falls in. As the weights slide, particles convert one at a time, so a
//! Desert/Tundra seam is a squall of mixed dust and snow that resolves as you walk
//! — the same dithered crossfade the terrain uses, for the same reason, and at no
//! extra cost. The set is still walked once per frame.
//!
//! # How it is drawn here
//!
//! Two meshes, rebuilt every frame and sorted by z against the backdrop:
//!
//! | z | Mesh | Kinds |
//! |---|---|---|
//! | [`HAZE_Z`] | vertex-coloured, source-over | dust, snow |
//! | [`GLOW_Z`] | vertex-coloured, [`AdditiveMaterial`] | spores, embers |
//!
//! That split IS the TypeScript's kind-major loop. It set `fillStyle` and the
//! composite mode at most four times a frame no matter how the particles were
//! distributed; here the same grouping is two draw calls, and for the same reason
//! — the state changes, not the particles, are what cost.
//!
//! # What the port changed
//!
//! **The weights are a resource, not an argument.** `Ambience` owned the blended
//! weight vector and `Game.ts` passed it down. [`WeatherWeights`] is that vector
//! as a resource, so the producer and the consumer no longer have to be called by
//! the same function. See the SEAM on [`follow_dominant_biome`] for what fills it
//! until the ambience layer lands.
//!
//! **Per-kind motion is a table, not a `switch`.** The four cases differed only in
//! eight numbers each, and written as a `match` those numbers sit in the middle of
//! control flow where nothing marks them as the tuning they are. [`MOTION`] is the
//! same values with names and reasons on them, and [`WeatherField::particle`] is
//! the one piece of arithmetic all four now share.
//!
//! **The phase stride is a whole turn.** [`PHASE_STRIDE`] was 6.28 and is now
//! `TAU`, on the TypeScript's own argument for why it did not matter — see the
//! constant. The particle phases move by a thousandth of a turn and nothing else
//! does.
//!
//! **The cumulative weights are not cached.** The TypeScript kept a `Float32Array`
//! for them and refilled it in place every frame, because allocating five floats a
//! frame in a browser was not free. [`WeatherField::buckets`] returns them on the
//! stack.
//!
//! # What the port dropped on the way in
//!
//! - **`performance.now()`.** Same as the sky: [`Time::elapsed_secs`], which is
//!   the same seconds and stops when the app does.
//! - **The `globalAlpha` / `globalCompositeOperation` save-and-restore.** There is
//!   no shared mutable context to hand back in the state it was found in.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::prelude::*;
use bevy::sprite_render::{AlphaMode2d, Material2dPlugin};

use godgame_core::config::View;
use godgame_core::sim::biomes::Weather as WeatherKind;

use crate::lowres::{WORLD_LAYERS, WorldCamera};
use crate::sky::{
    AdditiveMaterial, Atmosphere, Frame, Rgb, VertexBuf, dynamic_mesh, linear, rgb32, wrap,
};

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// The weather kinds, in the fixed order [`WeatherWeights`] is indexed by.
///
/// `None` is index 0 and is never drawn, but it is in the table and it keeps its
/// share of the weight: a half-clear sky has to be half EMPTY rather than double
/// the other kind. Dropping it would make a clearing storm thicken instead of
/// thin out.
pub const WEATHER_KINDS: [WeatherKind; WEATHER_KIND_COUNT] = [
    WeatherKind::None,
    WeatherKind::Dust,
    WeatherKind::Snow,
    WeatherKind::Spores,
    WeatherKind::Embers,
];

/// How many weather kinds there are.
pub const WEATHER_KIND_COUNT: usize = 5;

/// How many particles the field holds.
///
/// One set, shared by every kind. 120 is enough to read as weather across a whole
/// view and few enough that walking the set for each of four kinds is still one
/// pass over half a cache line's worth of floats.
pub const WEATHER_COUNT: usize = 120;

/// Below this weight a kind is skipped entirely.
///
/// A kind at a thousandth owns about a tenth of one particle, so the bucket is
/// almost always empty and the walk is pure cost.
const KIND_CUTOFF: f32 = 0.001;

/// Spread of the per-particle phase jitter, in radians.
///
/// Every sway below is `sin(t * rate + j * PHASE_STRIDE)`, where `j` is that
/// particle's fixed hash in `[0, 1)`. Multiplying by a full turn maps those hashes
/// across the whole cycle, so neighbouring particles sit at unrelated phases and
/// the field reads as weather rather than as one rigid sheet sliding about.
///
/// A full turn EXACTLY, where the TypeScript carried the rounded 6.28 it had
/// always had and said in as many words why it was not worth tightening: `j` is
/// already uniform, so the two spread the phases equally well and the difference
/// is not observable in a drifting particle field. That reasoning also says the
/// change is free, and a named constant that is visibly a rounded tau invites
/// exactly one question from every reader.
const PHASE_STRIDE: f32 = core::f32::consts::TAU;

/// Alpha floor every kind is drawn at, whatever the hour.
///
/// Weather never disappears at the wrong time of day, it just recedes: falling
/// weather is a daylight thing and glowing motes read at night, but both keep
/// [`VIS_FLOOR`] so a storm does not vanish at dusk.
const VIS_FLOOR: f32 = 0.45;

/// How much of a kind's visibility the time of day controls, above the floor.
const VIS_SWING: f32 = 0.55;

/// Pale haze tint for snow.
const SNOW_TINT: Rgb = [235.0, 240.0, 250.0];

/// Pale haze tint for dust — sand, not grey, so a desert squall reads as desert.
const DUST_TINT: Rgb = [210.0, 180.0, 130.0];

/// How one weather kind moves and looks.
///
/// Every field is per second or per view pixel, so a kind's behaviour can be read
/// off one row without following it through the arithmetic that applies it.
#[derive(Clone, Copy, Debug)]
struct Motion {
    /// Whether the kind TRAVELS horizontally. The sway is always on the other
    /// axis, which is what separates dust blowing across from snow coming down.
    horizontal: bool,
    /// Travel speed in view px/s at `j = 0`. Negative rises.
    speed: f32,
    /// How much of the particle's own hash is added to [`Motion::speed`], so a
    /// field has a spread of speeds rather than moving as a sheet.
    speed_spread: f32,
    /// Sway rate, radians per second.
    sway_rate: f32,
    /// Sway amplitude, view px.
    sway_amp: f32,
    /// Alpha floor.
    alpha: f32,
    /// How much of the particle's hash — or, for a pulsing kind, of the pulse —
    /// is added on top of [`Motion::alpha`].
    alpha_spread: f32,
    /// Fraction of the camera's motion this kind takes.
    parallax: f32,
    /// Hash above which a particle draws two px instead of one. Above 1 never.
    big_above: f32,
    /// Pulse rate in rad/s for a kind whose alpha breathes instead of being fixed
    /// per particle.
    pulse: Option<f32>,
}

/// Per-kind motion, indexed by [`WEATHER_KINDS`].
///
/// The four visible rows are the whole of what made the TypeScript's `switch` four
/// cases. Read down the `speed` column and the layer's design is legible in one
/// go: dust blows sideways, snow falls slowly, spores drift up slower still, and
/// embers rise faster than anything else on screen.
const MOTION: [Motion; WEATHER_KIND_COUNT] = [
    // None. Never drawn; present so the table indexes like every other one here.
    Motion {
        horizontal: false,
        speed: 0.0,
        speed_spread: 0.0,
        sway_rate: 0.0,
        sway_amp: 0.0,
        alpha: 0.0,
        alpha_spread: 0.0,
        parallax: 0.0,
        big_above: 2.0,
        pulse: None,
    },
    // Dust: drifts right on a mild parallax, bobbing gently. Always one px and
    // faint — it is haze, and a dune field of visible specks would read as rain.
    Motion {
        horizontal: true,
        speed: 18.0,
        speed_spread: 24.0,
        sway_rate: 0.6,
        sway_amp: 8.0,
        alpha: 0.18,
        alpha_spread: 0.18,
        parallax: 0.2,
        big_above: 2.0,
        pulse: None,
    },
    // Snow: falls, swaying across as it goes. The brightest and most solid of the
    // four, and the only one where a good fraction of the flakes are two px.
    Motion {
        horizontal: false,
        speed: 28.0,
        speed_spread: 34.0,
        sway_rate: 0.8,
        sway_amp: 12.0,
        alpha: 0.5,
        alpha_spread: 0.4,
        parallax: 0.25,
        big_above: 0.6,
        pulse: None,
    },
    // Spores: float up slowly on the widest sway of the four, and breathe rather
    // than hold an alpha. Slow, wide and pulsing is what makes them read as alive.
    Motion {
        horizontal: false,
        speed: -8.0,
        speed_spread: -10.0,
        sway_rate: 0.4,
        sway_amp: 16.0,
        alpha: 0.15,
        alpha_spread: 0.3,
        parallax: 0.2,
        big_above: 0.7,
        pulse: Some(1.0),
    },
    // Embers: rise fast on a tight, quick sway. The speed is the tell — anything
    // this quick reads as heat rather than as weather.
    Motion {
        horizontal: false,
        speed: -40.0,
        speed_spread: -50.0,
        sway_rate: 1.2,
        sway_amp: 10.0,
        alpha: 0.3,
        alpha_spread: 0.5,
        parallax: 0.25,
        big_above: 0.8,
        pulse: None,
    },
];

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// Everything the field needs to place itself for one frame.
///
/// A struct rather than six arguments: `paint` would otherwise be at the edge of
/// what is readable, and every field here comes from a different owner.
#[derive(Clone, Copy, Debug)]
pub struct WeatherFrame {
    /// Weight per kind, indexed by [`WEATHER_KINDS`]. Need not be normalised.
    pub weights: [f32; WEATHER_KIND_COUNT],
    /// The world clock's daylight weight, 0..1.
    pub day: f32,
    /// Animation clock, in seconds.
    pub seconds: f32,
    /// The view's TOP-LEFT in world px.
    pub cam: Vec2,
    /// The buffer being drawn into.
    pub view: View,
}

/// One particle, placed for a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drawn {
    /// Which kind it belongs to THIS frame. A particle can change kind as the
    /// weights slide; nothing about it is retained when it does.
    pub kind: WeatherKind,
    /// View x, already wrapped into `[0, view.w)`.
    pub x: f32,
    /// View y, already wrapped into `[0, view.h)`.
    pub y: f32,
    /// Side length in view px: one, or two for the larger flakes and motes.
    pub size: f32,
    /// Final alpha, time-of-day visibility included.
    pub alpha: f32,
}

/// The fixed particle set.
///
/// Four decorrelated hash streams per particle: a position, a phase and size
/// jitter, and the kind selector. All four are seeded once and never move — what
/// changes across a frame is only which bucket a selector falls in.
#[derive(Clone, Debug)]
pub struct WeatherField {
    x: [f32; WEATHER_COUNT],
    y: [f32; WEATHER_COUNT],
    jitter: [f32; WEATHER_COUNT],
    selector: [f32; WEATHER_COUNT],
}

impl WeatherField {
    /// Seed the field.
    pub fn new() -> WeatherField {
        let mut field = WeatherField {
            x: [0.0; WEATHER_COUNT],
            y: [0.0; WEATHER_COUNT],
            jitter: [0.0; WEATHER_COUNT],
            selector: [0.0; WEATHER_COUNT],
        };
        for i in 0..WEATHER_COUNT {
            let n = i as u32;
            field.x[i] = crate::sky::hash(n * 2 + 1);
            field.y[i] = crate::sky::hash(n * 5 + 2);
            field.jitter[i] = crate::sky::hash(n * 11 + 3);
            // Multiplied by a larger stride than the rest so the selector is
            // decorrelated from the POSITION: a selector that tracked x would
            // convert the field in a stripe down the screen rather than a scatter.
            field.selector[i] = crate::sky::hash(n * 29 + 17);
        }
        field
    }

    /// The cumulative selector buckets, or `None` for a sky with no weather in it.
    ///
    /// The last bucket is written as exactly 1 rather than left as the sum, so a
    /// selector just under 1 cannot fall off the end of the table through float
    /// drift and be silently dropped.
    pub fn buckets(weights: &[f32; WEATHER_KIND_COUNT]) -> Option<[f32; WEATHER_KIND_COUNT]> {
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mut cum = [0.0; WEATHER_KIND_COUNT];
        let mut run = 0.0;
        for (slot, w) in cum.iter_mut().zip(weights.iter()) {
            run += w / total;
            *slot = run;
        }
        cum[WEATHER_KIND_COUNT - 1] = 1.0;
        Some(cum)
    }

    /// Place every particle that is drawn this frame, kind-major.
    ///
    /// Kind-major because that is the order the two meshes want them in, and
    /// because it is the order that made the TypeScript's four state changes four
    /// rather than up to 120.
    pub fn paint(&self, frame: &WeatherFrame, out: &mut Vec<Drawn>) {
        out.clear();
        let Some(cum) = WeatherField::buckets(&frame.weights) else {
            return;
        };

        let w = frame.view.w as f32;
        let h = frame.view.h as f32;
        // From 1: `None` owns its share of the selectors and draws nothing with it.
        for k in 1..WEATHER_KIND_COUNT {
            if frame.weights[k] <= KIND_CUTOFF {
                continue;
            }
            let kind = WEATHER_KINDS[k];
            let (lo, hi) = (cum[k - 1], cum[k]);
            let vis = visibility(kind, frame.day);

            for i in 0..WEATHER_COUNT {
                let s = self.selector[i];
                if s < lo || s >= hi {
                    continue; // belongs to another kind this frame
                }
                let (x, y, size, alpha) = self.particle(i, k, frame);
                out.push(Drawn {
                    kind,
                    x: wrap(x, w),
                    y: wrap(y, h),
                    size,
                    alpha: alpha * vis,
                });
            }
        }
    }

    /// Particle `i` as kind `k`: `(view x, view y, size, alpha)`, unwrapped.
    ///
    /// The one piece of arithmetic all four kinds share. `travel` runs along the
    /// kind's axis and the sway crosses it; both take the same parallax, so a kind
    /// sits at one depth rather than shearing.
    fn particle(&self, i: usize, k: usize, frame: &WeatherFrame) -> (f32, f32, f32, f32) {
        let m = MOTION[k];
        let j = self.jitter[i];
        let t = frame.seconds;

        let travel = t * (m.speed + j * m.speed_spread);
        let sway = (t * m.sway_rate + j * PHASE_STRIDE).sin() * m.sway_amp;
        let (dx, dy) = if m.horizontal {
            (travel, sway)
        } else {
            (sway, travel)
        };

        let alpha = match m.pulse {
            Some(rate) => m.alpha + (t * rate + j * PHASE_STRIDE).sin().abs() * m.alpha_spread,
            None => m.alpha + j * m.alpha_spread,
        };

        (
            self.x[i] * frame.view.w as f32 + dx - frame.cam.x * m.parallax,
            self.y[i] * frame.view.h as f32 + dy - frame.cam.y * m.parallax,
            if j > m.big_above { 2.0 } else { 1.0 },
            alpha,
        )
    }
}

impl Default for WeatherField {
    fn default() -> WeatherField {
        WeatherField::new()
    }
}

/// Whether a kind glows in its own right rather than catching the light.
///
/// The lit kinds are additive and take the sky's own star colour; the rest are a
/// flat haze tint over the backdrop.
pub fn is_lit(kind: WeatherKind) -> bool {
    matches!(kind, WeatherKind::Spores | WeatherKind::Embers)
}

/// A kind's colour, given the atmosphere's starlight.
///
/// The lit kinds borrow `star` rather than declaring a colour, so an ember over
/// Volcanic and a spore in the fungal depths pick up their own biome's light —
/// the same value the starfield is drawn in, for the same reason.
pub fn kind_color(kind: WeatherKind, star: Rgb) -> Rgb {
    match kind {
        WeatherKind::Snow => SNOW_TINT,
        WeatherKind::Spores | WeatherKind::Embers => star,
        // Dust, and `None`, which is never drawn and so never asked.
        _ => DUST_TINT,
    }
}

/// How visible a kind is at this hour.
///
/// Falling weather is a daylight thing and glowing motes read at night, so the two
/// run opposite each other off the same floor and swing.
pub fn visibility(kind: WeatherKind, day: f32) -> f32 {
    let lit = if is_lit(kind) { 1.0 - day } else { day };
    VIS_FLOOR + VIS_SWING * lit
}

/// Where a kind sits in [`WEATHER_KINDS`].
pub fn kind_index(kind: WeatherKind) -> usize {
    match kind {
        WeatherKind::None => 0,
        WeatherKind::Dust => 1,
        WeatherKind::Snow => 2,
        WeatherKind::Spores => 3,
        WeatherKind::Embers => 4,
    }
}

// ---------------------------------------------------------------------------
// The plugin
// ---------------------------------------------------------------------------

/// Where the source-over kinds sit: over the whole backdrop, under the world.
pub const HAZE_Z: f32 = -50.0;

/// Where the additive kinds sit, over the haze.
pub const GLOW_Z: f32 = -49.0;

/// Weight per kind, indexed by [`WEATHER_KINDS`]. Need not be normalised.
///
/// SEAM: this is `Ambience.weatherW` — the dithered blend derived from the biome
/// mix at the camera. The ambience layer does not exist in this port yet, so
/// [`follow_dominant_biome`] fills it from the resolved atmosphere's single
/// dominant kind, which is exactly the snapping the dithering exists to fix. When
/// the ambience layer lands it writes this resource, that system is deleted, and
/// nothing else in this module changes.
#[derive(Resource, Clone, Copy, Debug)]
pub struct WeatherWeights(pub [f32; WEATHER_KIND_COUNT]);

impl Default for WeatherWeights {
    /// A clear sky.
    fn default() -> WeatherWeights {
        let mut w = [0.0; WEATHER_KIND_COUNT];
        w[0] = 1.0;
        WeatherWeights(w)
    }
}

/// The particle set and the frame's scratch list.
///
/// One resource because the two are used together and the list is scratch — it is
/// cleared and refilled every frame, and it lives here rather than in a `Local` so
/// that its allocation is visibly owned by something.
#[derive(Resource, Default)]
pub struct WeatherPool {
    /// The fixed particle set.
    pub field: WeatherField,
    /// This frame's placements, kind-major.
    pub drawn: Vec<Drawn>,
}

/// Which of the two meshes an entity is.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeatherLayer {
    /// Dust and snow, composited source-over.
    Haze,
    /// Spores and embers, composited additively.
    Glow,
}

impl WeatherLayer {
    /// Whether a kind belongs to this layer.
    fn holds(self, kind: WeatherKind) -> bool {
        is_lit(kind) == (self == WeatherLayer::Glow)
    }
}

/// The particle field and the two meshes it is drawn in.
///
/// Requires [`SkyPlugin`](crate::sky::SkyPlugin): the weather is drawn over the
/// sky, against the sky's [`Atmosphere`], on the sky's [`Frame`]. `Game.ts` handed
/// both layers the same resolved atmosphere for the same reason.
pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        // Registered by whichever of the two plugins is added first — see the same
        // guard in `SkyPlugin`. Adding a plugin twice is a panic, not a no-op.
        if !app.is_plugin_added::<Material2dPlugin<AdditiveMaterial>>() {
            app.add_plugins(Material2dPlugin::<AdditiveMaterial>::default());
        }

        app.init_resource::<WeatherWeights>()
            .init_resource::<WeatherPool>()
            // After every `Startup`, so the world camera these parent themselves
            // to already exists.
            .add_systems(PostStartup, setup)
            .add_systems(
                Update,
                (follow_dominant_biome, place_particles)
                    .chain()
                    .run_if(resource_exists::<Atmosphere>),
            );
    }
}

/// Spawn the two particle meshes as children of the world camera.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut blend: ResMut<Assets<ColorMaterial>>,
    mut additive: ResMut<Assets<AdditiveMaterial>>,
    camera: Single<Entity, With<WorldCamera>>,
) {
    let camera = *camera;

    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(blend.add(ColorMaterial {
            color: Color::WHITE,
            alpha_mode: AlphaMode2d::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, HAZE_Z),
        WeatherLayer::Haze,
        // Rewritten in place every frame, so the bounding box Bevy computed from
        // the first version of the mesh is stale immediately.
        NoFrustumCulling,
        ChildOf(camera),
        WORLD_LAYERS,
    ));

    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(additive.add(AdditiveMaterial {})),
        Transform::from_xyz(0.0, 0.0, GLOW_Z),
        WeatherLayer::Glow,
        NoFrustumCulling,
        ChildOf(camera),
        WORLD_LAYERS,
    ));
}

/// Put the whole weight on the dominant biome's kind.
///
/// SEAM — see [`WeatherWeights`]. This is the un-dithered stand-in: it snaps from
/// snow to dust in one step at a biome boundary, which is the exact behaviour the
/// selector dithering above was written to replace. Delete this system when
/// `Ambience` arrives to write the resource properly.
fn follow_dominant_biome(atmo: Res<Atmosphere>, mut weights: ResMut<WeatherWeights>) {
    let mut w = [0.0; WEATHER_KIND_COUNT];
    w[kind_index(atmo.resolved.weather)] = 1.0;
    weights.0 = w;
}

/// Rebuild both particle meshes.
fn place_particles(
    frame: Frame,
    weights: Res<WeatherWeights>,
    mut pool: ResMut<WeatherPool>,
    layers: Query<(&Mesh2d, &WeatherLayer)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buf: Local<VertexBuf>,
) {
    let view = frame.view();
    let star = rgb32(frame.atmo.resolved.star);
    let weather = WeatherFrame {
        weights: weights.0,
        day: frame.phase().day,
        seconds: frame.seconds(),
        cam: frame.cam(),
        view,
    };

    let WeatherPool { field, drawn } = &mut *pool;
    field.paint(&weather, drawn);

    for (mesh, layer) in &layers {
        let Some(mut mesh) = meshes.get_mut(&mesh.0) else {
            continue;
        };
        buf.clear();
        for particle in drawn.iter().filter(|p| layer.holds(p.kind)) {
            // Rounded to a whole view pixel, as the sky rounds its stars: a
            // fractional `fillRect` antialiases a 1px mote across two, which at
            // this buffer size is a smudge rather than a particle.
            buf.rect(
                particle.x.round(),
                particle.y.round(),
                particle.size,
                particle.size,
                view,
                linear(kind_color(particle.kind, star), particle.alpha),
            );
        }
        buf.write(&mut mesh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        View::for_screen(1440, 900)
    }

    fn frame(weights: [f32; WEATHER_KIND_COUNT]) -> WeatherFrame {
        WeatherFrame {
            weights,
            day: 1.0,
            seconds: 3.5,
            cam: Vec2::new(240.0, 80.0),
            view: view(),
        }
    }

    /// Weight entirely on one kind.
    fn only(kind: WeatherKind) -> [f32; WEATHER_KIND_COUNT] {
        let mut w = [0.0; WEATHER_KIND_COUNT];
        w[kind_index(kind)] = 1.0;
        w
    }

    fn count_of(drawn: &[Drawn], kind: WeatherKind) -> usize {
        drawn.iter().filter(|p| p.kind == kind).count()
    }

    #[test]
    fn the_kind_order_is_the_one_the_weights_are_indexed_by() {
        // `WeatherWeights` is a bare array and every producer and consumer of it
        // agrees on this order and nothing else. If the two ever disagree, snow
        // silently becomes spores.
        for (i, kind) in WEATHER_KINDS.iter().enumerate() {
            assert_eq!(kind_index(*kind), i, "{kind:?} moved");
        }
        assert_eq!(WEATHER_KINDS[0], WeatherKind::None, "clear is first");
    }

    #[test]
    fn an_unweighted_sky_draws_nothing_at_all() {
        assert!(WeatherField::buckets(&[0.0; WEATHER_KIND_COUNT]).is_none());
        let mut drawn = Vec::new();
        WeatherField::new().paint(&frame([0.0; WEATHER_KIND_COUNT]), &mut drawn);
        assert!(drawn.is_empty());
    }

    #[test]
    fn a_clear_sky_draws_nothing_even_though_none_owns_every_selector() {
        // `None` is weighted and therefore owns the whole particle set, and the
        // set is still walked. Nothing comes out of it, which is the difference
        // between "clear" and "no weather system".
        let mut drawn = Vec::new();
        WeatherField::new().paint(&frame(only(WeatherKind::None)), &mut drawn);
        assert!(drawn.is_empty());
    }

    #[test]
    fn the_weights_need_not_be_normalised() {
        let field = WeatherField::new();
        let mut unit = Vec::new();
        let mut scaled = Vec::new();
        let mut w = [0.0; WEATHER_KIND_COUNT];
        w[kind_index(WeatherKind::Dust)] = 0.25;
        w[kind_index(WeatherKind::Snow)] = 0.75;
        field.paint(&frame(w), &mut unit);
        field.paint(&frame(w.map(|v| v * 40.0)), &mut scaled);
        assert_eq!(unit, scaled);
    }

    #[test]
    fn one_kind_at_full_weight_owns_every_particle() {
        let field = WeatherField::new();
        let mut drawn = Vec::new();
        for kind in [
            WeatherKind::Dust,
            WeatherKind::Snow,
            WeatherKind::Spores,
            WeatherKind::Embers,
        ] {
            field.paint(&frame(only(kind)), &mut drawn);
            assert_eq!(drawn.len(), WEATHER_COUNT, "{kind:?} lost particles");
            assert_eq!(count_of(&drawn, kind), WEATHER_COUNT);
        }
    }

    #[test]
    fn a_half_clear_sky_really_is_half_empty() {
        // The reason `None` keeps a bucket. Drop it and a clearing storm would
        // thicken instead of thinning out, because the survivors would divide the
        // whole set between them.
        let mut w = [0.0; WEATHER_KIND_COUNT];
        w[kind_index(WeatherKind::None)] = 0.5;
        w[kind_index(WeatherKind::Snow)] = 0.5;
        let mut drawn = Vec::new();
        WeatherField::new().paint(&frame(w), &mut drawn);
        let half = WEATHER_COUNT / 2;
        assert!(
            drawn.len().abs_diff(half) < WEATHER_COUNT / 8,
            "half a sky of snow drew {} of {WEATHER_COUNT}",
            drawn.len()
        );
    }

    #[test]
    fn particles_convert_one_at_a_time_as_the_weights_slide() {
        // THE point of the dithering. A Desert/Tundra seam has to be a squall of
        // mixed dust and snow that resolves as you walk, not a switch.
        let field = WeatherField::new();
        let mut drawn = Vec::new();
        let mut snow_counts = Vec::new();

        for step in 0..=40 {
            let t = step as f32 / 40.0;
            let mut w = [0.0; WEATHER_KIND_COUNT];
            w[kind_index(WeatherKind::Dust)] = 1.0 - t;
            w[kind_index(WeatherKind::Snow)] = t;
            field.paint(&frame(w), &mut drawn);
            assert_eq!(drawn.len(), WEATHER_COUNT, "particles went missing at {t}");
            snow_counts.push(count_of(&drawn, WeatherKind::Snow));
        }

        assert_eq!(snow_counts[0], 0, "no snow in a pure dust storm");
        assert_eq!(*snow_counts.last().unwrap(), WEATHER_COUNT);
        for pair in snow_counts.windows(2) {
            assert!(pair[1] >= pair[0], "the crossfade went backwards");
            // 120 particles over 40 steps is three a step on average; nothing
            // should ever jump by a tenth of the field.
            assert!(
                pair[1] - pair[0] <= WEATHER_COUNT / 10,
                "{} particles converted in one step",
                pair[1] - pair[0]
            );
        }
    }

    #[test]
    fn a_three_way_seam_gives_every_kind_a_share() {
        let mut w = [0.0; WEATHER_KIND_COUNT];
        w[kind_index(WeatherKind::Dust)] = 0.4;
        w[kind_index(WeatherKind::Snow)] = 0.3;
        w[kind_index(WeatherKind::Embers)] = 0.3;
        let mut drawn = Vec::new();
        WeatherField::new().paint(&frame(w), &mut drawn);
        assert_eq!(drawn.len(), WEATHER_COUNT);
        for kind in [WeatherKind::Dust, WeatherKind::Snow, WeatherKind::Embers] {
            assert!(count_of(&drawn, kind) > 0, "{kind:?} got nothing");
        }
    }

    #[test]
    fn a_kind_below_the_cutoff_is_skipped_rather_than_drawn_as_a_speck() {
        let mut w = [0.0; WEATHER_KIND_COUNT];
        w[kind_index(WeatherKind::Snow)] = 1.0;
        w[kind_index(WeatherKind::Embers)] = KIND_CUTOFF;
        let mut drawn = Vec::new();
        WeatherField::new().paint(&frame(w), &mut drawn);
        assert_eq!(count_of(&drawn, WeatherKind::Embers), 0);
    }

    #[test]
    fn every_particle_lands_inside_the_view_however_far_the_camera_has_gone() {
        let field = WeatherField::new();
        let mut drawn = Vec::new();
        let view = view();
        for cam in [
            Vec2::ZERO,
            Vec2::new(1.0e6, -1.0e6),
            Vec2::new(-4.0e5, 9.0e5),
        ] {
            for seconds in [0.0, 7.5, 900.0] {
                let mut f = frame(only(WeatherKind::Snow));
                f.cam = cam;
                f.seconds = seconds;
                field.paint(&f, &mut drawn);
                for p in &drawn {
                    assert!(
                        p.x >= 0.0 && p.x < view.w as f32,
                        "x {} left the view at {cam:?} t={seconds}",
                        p.x
                    );
                    assert!(p.y >= 0.0 && p.y < view.h as f32, "y {} left it", p.y);
                }
            }
        }
    }

    /// How far a wrapped coordinate moved, taking the short way round.
    ///
    /// The field wraps into one view tile, so a particle that leaves the bottom
    /// reappears at the top and a raw subtraction reads that as a leap of a whole
    /// view in the wrong direction.
    fn moved(from: f32, to: f32, span: f32) -> f32 {
        (to - from + span * 0.5).rem_euclid(span) - span * 0.5
    }

    #[test]
    fn snow_falls_dust_blows_sideways_and_embers_rise() {
        // The four kinds differ in almost nothing else, so this is the assertion
        // that they are actually four kinds.
        let field = WeatherField::new();
        let view = view();
        let mut early = Vec::new();
        let mut late = Vec::new();

        for (kind, expect_down, expect_across) in [
            (WeatherKind::Snow, true, false),
            (WeatherKind::Dust, false, true),
            (WeatherKind::Embers, false, false),
            (WeatherKind::Spores, false, false),
        ] {
            let mut a = frame(only(kind));
            a.cam = Vec2::ZERO;
            a.seconds = 0.0;
            let mut b = a;
            b.seconds = 0.05;
            field.paint(&a, &mut early);
            field.paint(&b, &mut late);

            let dy: f32 = late
                .iter()
                .zip(early.iter())
                .map(|(l, e)| moved(e.y, l.y, view.h as f32))
                .sum();
            let dx: f32 = late
                .iter()
                .zip(early.iter())
                .map(|(l, e)| moved(e.x, l.x, view.w as f32))
                .sum();

            if expect_across {
                assert!(dx > 0.0, "{kind:?} should drift right, moved {dx}");
            } else if expect_down {
                assert!(dy > 0.0, "{kind:?} should fall, moved {dy}");
            } else {
                assert!(dy < 0.0, "{kind:?} should rise, moved {dy}");
            }
        }
    }

    #[test]
    fn embers_rise_faster_than_spores_drift_up() {
        // Speed is the whole difference between "heat" and "alive".
        let embers = MOTION[kind_index(WeatherKind::Embers)];
        let spores = MOTION[kind_index(WeatherKind::Spores)];
        assert!(embers.speed < spores.speed, "both rise; embers rise harder");
        assert!(spores.sway_amp > embers.sway_amp, "spores wander wider");
    }

    #[test]
    fn falling_weather_reads_in_daylight_and_glowing_motes_read_at_night() {
        for kind in [WeatherKind::Dust, WeatherKind::Snow] {
            assert!(visibility(kind, 1.0) > visibility(kind, 0.0), "{kind:?}");
        }
        for kind in [WeatherKind::Spores, WeatherKind::Embers] {
            assert!(visibility(kind, 0.0) > visibility(kind, 1.0), "{kind:?}");
        }
    }

    #[test]
    fn no_kind_ever_fades_out_completely() {
        // The floor: weather recedes at the wrong hour, it does not vanish.
        for kind in WEATHER_KINDS {
            for day in [0.0, 0.5, 1.0] {
                let v = visibility(kind, day);
                assert!(
                    (VIS_FLOOR..=VIS_FLOOR + VIS_SWING).contains(&v),
                    "{kind:?} at day={day} was {v}"
                );
            }
        }
    }

    #[test]
    fn spores_and_embers_are_the_lit_kinds_and_take_the_skys_own_starlight() {
        let star = [10.0, 200.0, 30.0];
        for kind in [WeatherKind::Spores, WeatherKind::Embers] {
            assert!(is_lit(kind));
            assert_eq!(kind_color(kind, star), star, "{kind:?} invented a colour");
        }
        for kind in [WeatherKind::Dust, WeatherKind::Snow] {
            assert!(!is_lit(kind));
            assert_ne!(kind_color(kind, star), star);
        }
    }

    #[test]
    fn snow_is_pale_and_dust_is_sandy() {
        let star = [0.0; 3];
        let snow = kind_color(WeatherKind::Snow, star);
        let dust = kind_color(WeatherKind::Dust, star);
        assert!(snow[2] > snow[0], "snow should read cool");
        assert!(dust[0] > dust[2], "dust should read warm");
        assert!(snow.iter().all(|c| *c > 200.0), "snow should be pale");
    }

    #[test]
    fn a_particle_is_never_larger_than_two_pixels() {
        let field = WeatherField::new();
        let mut drawn = Vec::new();
        for kind in [
            WeatherKind::Dust,
            WeatherKind::Snow,
            WeatherKind::Spores,
            WeatherKind::Embers,
        ] {
            field.paint(&frame(only(kind)), &mut drawn);
            for p in &drawn {
                assert!(p.size == 1.0 || p.size == 2.0, "{kind:?} drew {}", p.size);
            }
        }
        // Dust is the one kind that is never big: it is haze, and a dune field of
        // visible specks would read as rain.
        field.paint(&frame(only(WeatherKind::Dust)), &mut drawn);
        assert!(drawn.iter().all(|p| p.size == 1.0));
    }

    #[test]
    fn spores_pulse_and_every_other_kind_holds_a_fixed_alpha() {
        let field = WeatherField::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for kind in [
            WeatherKind::Dust,
            WeatherKind::Snow,
            WeatherKind::Spores,
            WeatherKind::Embers,
        ] {
            let mut early = frame(only(kind));
            early.seconds = 0.4;
            let mut late = early;
            late.seconds = 2.1;
            field.paint(&early, &mut a);
            field.paint(&late, &mut b);

            let moved = a
                .iter()
                .zip(b.iter())
                .any(|(x, y)| (x.alpha - y.alpha).abs() > 1.0e-4);
            assert_eq!(
                moved,
                kind == WeatherKind::Spores,
                "{kind:?} alpha behaviour is wrong"
            );
        }
    }

    #[test]
    fn the_phase_jitter_spreads_neighbours_over_the_whole_cycle() {
        // Without it the field sways as one rigid sheet. Two adjacent particles
        // must sit at unrelated phases, which is what `PHASE_STRIDE` buys.
        let field = WeatherField::new();
        let mut apart = 0;
        for i in 0..WEATHER_COUNT - 1 {
            let a = (field.jitter[i] * PHASE_STRIDE).sin();
            let b = (field.jitter[i + 1] * PHASE_STRIDE).sin();
            if (a - b).abs() > 0.3 {
                apart += 1;
            }
        }
        assert!(
            apart > WEATHER_COUNT / 2,
            "only {apart} of {WEATHER_COUNT} neighbours were out of phase"
        );
    }

    #[test]
    fn the_buckets_reach_exactly_one_so_no_selector_falls_off_the_end() {
        // The selectors are in [0,1) and the last bucket is written as 1 rather
        // than left as the accumulated sum. Without that a particle at 0.9999
        // would be silently dropped whenever the sum landed a ULP short.
        let mut w = [0.3; WEATHER_KIND_COUNT];
        w[2] = 0.1;
        let cum = WeatherField::buckets(&w).expect("weighted");
        assert_eq!(cum[WEATHER_KIND_COUNT - 1], 1.0);
        for pair in cum.windows(2) {
            assert!(pair[1] >= pair[0], "the buckets must not go backwards");
        }
    }

    #[test]
    fn the_selectors_are_decorrelated_from_the_positions() {
        // If a particle's selector tracked its x, the field would convert in a
        // stripe down the screen instead of a scatter. Compare the two rankings:
        // a correlated pair would agree far more often than half the time.
        let field = WeatherField::new();
        let mut agree = 0;
        for i in 0..WEATHER_COUNT - 1 {
            if (field.x[i] < field.x[i + 1]) == (field.selector[i] < field.selector[i + 1]) {
                agree += 1;
            }
        }
        let n = WEATHER_COUNT - 1;
        assert!(
            agree > n / 3 && agree < n * 2 / 3,
            "{agree} of {n} orderings agreed — the two streams are not independent"
        );
    }

    #[test]
    fn each_layer_takes_exactly_the_kinds_it_composites() {
        for kind in WEATHER_KINDS {
            assert_eq!(WeatherLayer::Glow.holds(kind), is_lit(kind));
            assert_ne!(
                WeatherLayer::Haze.holds(kind),
                WeatherLayer::Glow.holds(kind),
                "{kind:?} landed on both layers or neither"
            );
        }
    }

    #[test]
    fn a_clear_sky_is_the_default_until_the_ambience_layer_lands() {
        let w = WeatherWeights::default().0;
        assert_eq!(w[kind_index(WeatherKind::None)], 1.0);
        assert!(w.iter().skip(1).all(|v| *v == 0.0));
    }
}

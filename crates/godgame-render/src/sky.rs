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
//! [`BiomeAtmosphere`](godgame_core::sim::biomes::BiomeAtmosphere) declares
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
//! | [`GRADIENT_Z`] | a `1 x view.h` image on a stretched sprite | the base gradient AND the twilight band |
//! | [`STARS_Z`] | one vertex-coloured mesh, 90 quads | the starfield |
//! | [`DISCS_Z`] | one vertex-coloured mesh, two ring fans | sun and moon |
//! | [`RIDGES_Z`] | one vertex-coloured mesh, two strips | the hill silhouettes |
//!
//! Parenting to the camera rather than following it in a system is what makes the
//! backdrop screen-locked with no ordering rule to get wrong: transform
//! propagation runs after every writer of the camera's transform, so the sky can
//! never be a frame behind the view it is meant to fill. It also inherits the
//! camera's pixel snap for free, which is what keeps a 1px star from shimmering.
//!
//! # What the port changed
//!
//! **The base gradient and the twilight band are one image.** The canvas painted
//! a two-stop vertical gradient and then a second, `lighter`-composited gradient
//! over the whole rect. Both are functions of y ALONE, so the two composite
//! exactly into one colour per row — [`sky_texel`] — and the row set is a
//! `1 x view.h` texture stretched across the view. That is `view.h` texel writes
//! a frame instead of a full-screen fill, and the arithmetic is identical to the
//! canvas's because it is done in sRGB, on the same premultiplied stops.
//!
//! **The discs are geometry, not a baked sprite.** `Sky.disc` prebaked a soft
//! radial sprite into an offscreen canvas and blitted it. A canvas radial gradient
//! IS a piecewise-linear ramp between its stops, so the same thing here is a fan
//! of rings at the stop radii with the stop colours on their vertices — see
//! [`DISC_STOPS`]. It removes the bake, the two offscreen canvases, and the blit's
//! `x`/`y` culling test. The one difference is that the GPU interpolates between
//! rings in LINEAR light where the canvas interpolated in sRGB, which moves the
//! middle of a soft glow by roughly a value step. It is not visible on a 46px disc
//! and it is the price of not shipping a texture bake.
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

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::system::SystemParam;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, Extent3d,
    RenderPipelineDescriptor, SpecializedMeshPipelineError, TextureDimension, TextureFormat,
};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey, Material2dPlugin};

use godgame_core::config::{CELL_SIZE, SEED, SURFACE_ANCHOR_Y, View};
use godgame_core::sim::biomes::{ResolvedAtmosphere, resolve_atmosphere};
use godgame_core::sim::noise::Noise;
use godgame_core::sim::worldgen::world_noise;

use crate::daynight::{DayPhase, WorldClock};
use crate::lowres::{LowResTarget, WORLD_LAYERS, WorldCamera};
use crate::world::WorldFocus;

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// A backdrop colour, 0..255 per channel.
///
/// The same range the biome palettes are authored in, kept in `f32` rather than
/// bytes because every blend below is a lerp that would quantise badly at eight
/// bits and is truncated only once, on the way to a texel.
pub type Rgb = [f32; 3];

/// Neutral daylight target for the zenith.
///
/// A mid blue. Mixed at 0.45 against 1.25 of the biome's own horizon hue, so the
/// biome wins the tint and this only supplies the "it is daytime" lift.
const DAY_TOP: Rgb = [96.0, 140.0, 205.0];

/// Neutral daylight target for the horizon: near-white haze.
const DAY_BOT: Rgb = [200.0, 220.0, 240.0];

/// Sunrise/sunset zenith — a cool violet, because the sky opposite a low sun is
/// the one part of a real sunset that does NOT go warm.
const DUSK_TOP: Rgb = [64.0, 52.0, 96.0];

/// Sunrise/sunset horizon: hot orange.
const DUSK_BOT: Rgb = [242.0, 138.0, 74.0];

/// Below this the twilight band is skipped entirely.
///
/// The band is a full-view composite, so it is worth a compare not to run it for
/// the two thirds of the cycle where it would add less than a value step.
const GLOW_CUTOFF: f32 = 0.02;

/// Below this the starfield is skipped.
const STAR_CUTOFF: f32 = 0.01;

/// The twilight band's stops: `(offset, colour, alpha weight)`.
///
/// Blending the dusk colour into the whole gradient reddens the zenith as much as
/// the horizon, which is the one thing a real sunset never does — the warm light
/// is a BAND low in the sky, brightest where the sun is going down, and the sky
/// above it stays blue and then violet. This is that band.
///
/// The alphas are weights on the frame's glow, not absolute alphas, and the first
/// stop is fully transparent so the band fades IN from above. Below the last stop
/// a canvas gradient clamps, which is deliberate and load-bearing: the horizon
/// wash floods everything under the horizon line, not just the strip the stops
/// span.
const BAND_STOPS: [(f32, Rgb, f32); 3] = [
    (0.00, [240.0, 120.0, 60.0], 0.00),
    (0.55, [244.0, 132.0, 66.0], 0.30),
    (1.00, [255.0, 186.0, 110.0], 0.62),
];

/// Where the band's bottom sits, as a view fraction, when the sun is at the top of
/// its arc.
const BAND_HORIZON_BASE: f32 = 0.52;

/// How far the band's bottom tracks down the view as the sun sinks.
///
/// The glow is centred on the sun's own height so it follows the sun across the
/// horizon instead of sitting at a fixed line.
const BAND_HORIZON_TRACK: f32 = 0.30;

/// The band's height as a view fraction. Just under half the view: high enough to
/// read as sky, low enough to leave the zenith alone.
const BAND_SPAN: f32 = 0.42;

/// How many stars the field holds.
pub const STAR_COUNT: usize = 90;

/// Fraction of the view height the star seeds are spread over.
///
/// The upper 70% only — stars are seeded away from the horizon so the field does
/// not fight the hills. It is a seeding bias and not a hard ceiling: the parallax
/// wrap in [`StarField::star`] can carry any star anywhere.
const STAR_FIELD_H: f32 = 0.7;

/// Twinkle rate, radians per second.
///
/// The TypeScript wrote this as `performance.now() * 0.0016` — milliseconds, so
/// 1.6 rad/s, which is what this is.
const STAR_TWINKLE_RATE: f32 = 1.6;

/// Twinkle floor: the fraction of its own brightness a star never dips below.
const STAR_TWINKLE_BASE: f32 = 0.78;

/// Twinkle swing either side of [`STAR_TWINKLE_BASE`].
const STAR_TWINKLE_SWING: f32 = 0.22;

/// How much of the camera's motion the starfield takes. Distant, so almost none.
const STAR_PARALLAX: f32 = 0.1;

/// The sun's radius in view px.
const SUN_R: f32 = 46.0;

/// The moon's radius in view px. Smaller and cooler than the sun.
const MOON_R: f32 = 34.0;

/// The sun's hot white centre.
const SUN_CORE: Rgb = [255.0, 244.0, 214.0];

/// The sun's halo, which is where its warmth actually lives.
const SUN_HALO: Rgb = [255.0, 176.0, 92.0];

/// The moon's cold white centre.
const MOON_CORE: Rgb = [244.0, 248.0, 255.0];

/// The moon's halo: blue, the whole reason it does not read as a second sun.
const MOON_HALO: Rgb = [150.0, 170.0, 210.0];

/// How much of its own visibility the moon is drawn at.
///
/// The moon is the same soft disc as the sun and would read as a second sun at
/// full strength. Held back a tenth so it stays the dimmer of the two.
const MOON_DIM: f32 = 0.9;

/// Radial stops of a celestial disc: `(radius fraction, is-halo, alpha)`.
///
/// A hot core that holds most of its alpha to a third of the radius, a fast
/// hand-off to the tinted halo, then a long fade to nothing. `false` reads the
/// disc's core colour, `true` its halo.
const DISC_STOPS: [(f32, bool, f32); 4] = [
    (0.00, false, 0.95),
    (0.32, false, 0.70),
    (0.42, true, 0.28),
    (1.00, true, 0.00),
];

/// Segments in a disc's ring fan.
///
/// At the sun's 46px radius a 24-gon's edge bows in by `r * (1 - cos(pi/24))`,
/// under half a view pixel — round, at the only size this is ever drawn.
const DISC_SEGMENTS: usize = 24;

/// Below this a disc is not drawn at all.
const DISC_CUTOFF: f32 = 0.01;

/// The two hill ridges, far first.
///
/// Each is two summed sines, which is enough to stop the silhouette reading as a
/// repeating wave without being a heightmap. The far ridge sits higher, moves
/// slower and catches more of the sky's light; the near one is lower, faster and
/// nearly black, and the difference between the two IS the depth cue.
const RIDGES: [Ridge; 2] = [
    Ridge {
        parallax: 0.30,
        base: 0.70,
        f1: 0.006,
        f2: 0.017,
        a1: 26.0,
        a2: 12.0,
        shade: 1.0,
    },
    Ridge {
        parallax: 0.55,
        base: 0.82,
        f1: 0.011,
        f2: 0.023,
        a1: 18.0,
        a2: 9.0,
        shade: 0.55,
    },
];

/// How far apart a ridge is sampled, in view px.
///
/// The silhouette is straight-line interpolated between samples. 20px is coarse
/// enough to be cheap and fine enough that the shortest sine here (a period of
/// about 270px) never shows a facet.
const RIDGE_STEP_PX: f32 = 20.0;

/// Ridge brightness floor: what is left of a ridge's colour at midnight.
///
/// Both ridges sink toward black at night so the horizon reads as a silhouette
/// rather than as two grey bands.
const RIDGE_SHADE_FLOOR: f32 = 0.4;

/// How much of a ridge's colour daylight adds back on top of the floor.
const RIDGE_SHADE_DAY: f32 = 0.6;

/// Cells below [`SURFACE_ANCHOR_Y`] over which the sky darkens to fully
/// underground.
///
/// 300 cells is about six screens of descent: long enough that walking into a cave
/// mouth does not black out the sky, short enough that a real mining trip is dark
/// before the ore layers.
///
/// A module constant, not a `config` export: it describes ONE mapping, the one
/// from the camera's row to this module's `depth` parameter. In the TypeScript it
/// was an unnamed literal in `Game.ts` that existed only to feed this file.
const DEPTH_SPAN_CELLS: f32 = 300.0;

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// How far underground the view is: 0 at the surface anchor, 1 fully buried.
///
/// `focus_y` is the view centre in world px, +y DOWN.
pub fn depth_at(focus_y: f32) -> f32 {
    let row = focus_y / CELL_SIZE as f32;
    clamp01((row - SURFACE_ANCHOR_Y as f32) / DEPTH_SPAN_CELLS)
}

/// The base gradient's two ends, in 0..255.
///
/// Not clamped: the blends that build it can overshoot, and [`sky_texel`] is where
/// that is resolved — exactly as the canvas resolved it when an overshooting
/// `rgb()` string reached `fillStyle`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gradient {
    /// Zenith.
    pub top: Rgb,
    /// Horizon.
    pub bottom: Rgb,
}

/// The base gradient for a frame.
///
/// Three blends stack per channel: night to day by daylight, a twilight wash near
/// the horizon crossings, then a sink toward the biome's deep colour as the camera
/// descends — which is what keeps a cave underground-dark at noon.
///
/// The deep sink is weaker at the horizon (`depth * 0.9`) than at the zenith, so a
/// shallow cave still shows a rim of daylight at the bottom of the view rather
/// than going flat.
pub fn gradient(atmo: &ResolvedAtmosphere, depth: f32, phase: DayPhase) -> Gradient {
    let day = phase.day;
    let dusk = phase.twilight;
    let mut top = [0.0; 3];
    let mut bottom = [0.0; 3];

    for c in 0..3 {
        let sky_top = atmo.sky_top[c] as f32;
        let sky_bottom = atmo.sky_bottom[c] as f32;
        let deep = atmo.sky_top_deep[c] as f32;

        let day_top = clamp255(DAY_TOP[c] * 0.45 + sky_bottom * 1.25);
        let mut t = sky_top + (day_top - sky_top) * day;
        t += (DUSK_TOP[c] - t) * dusk * 0.45;
        top[c] = t + (deep - t) * depth;

        let day_bot = clamp255(DAY_BOT[c] * 0.5 + sky_bottom * 1.5);
        let mut b = sky_bottom + (day_bot - sky_bottom) * day;
        b += (DUSK_BOT[c] - b) * dusk * 0.6;
        bottom[c] = b + (deep - b) * depth * 0.9;
    }

    Gradient { top, bottom }
}

/// The warm wash low in the sky, when there is one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizonGlow {
    /// Overall strength, 0..1. Twilight, cut by how far underground the view is.
    pub strength: f32,
    /// View y where the band starts fading in.
    pub top: f32,
    /// View y where the band reaches full strength — and stays, all the way down.
    pub bottom: f32,
}

/// The band for a frame, or `None` when it would be invisible.
///
/// `depth` closes it down as the camera descends: a sunset seen from inside a
/// mountain is not a sunset.
pub fn horizon_glow(phase: DayPhase, depth: f32, view_h: f32) -> Option<HorizonGlow> {
    let strength = phase.twilight * (1.0 - depth);
    if strength <= GLOW_CUTOFF {
        return None;
    }
    let bottom = view_h * (BAND_HORIZON_BASE + BAND_HORIZON_TRACK * clamp01(phase.sun_y));
    Some(HorizonGlow {
        strength,
        top: bottom - view_h * BAND_SPAN,
        bottom,
    })
}

impl HorizonGlow {
    /// What the band ADDS at view row `y`, in 0..255 per channel.
    ///
    /// Premultiplied throughout, which is not a shortcut but the definition: a
    /// canvas gradient interpolates its stops premultiplied, and `lighter`
    /// composites premultiplied source onto the destination. Multiplying colour by
    /// alpha at the stops and lerping the product is therefore exactly both steps
    /// at once — and it is why the first stop's colour never appears anywhere in
    /// the result. Its alpha is zero, so it contributes nothing however it is
    /// written.
    pub fn add_at(&self, y: f32) -> Rgb {
        let span = self.bottom - self.top;
        let u = if span > 0.0 {
            clamp01((y - self.top) / span)
        } else {
            1.0
        };

        let mut lo = BAND_STOPS[0];
        for &hi in &BAND_STOPS[1..] {
            if u <= hi.0 {
                let t = if hi.0 > lo.0 {
                    clamp01((u - lo.0) / (hi.0 - lo.0))
                } else {
                    1.0
                };
                let mut out = [0.0; 3];
                for (slot, (lo_c, hi_c)) in out.iter_mut().zip(lo.1.iter().zip(hi.1.iter())) {
                    let a = lo_c * lo.2 * self.strength;
                    let b = hi_c * hi.2 * self.strength;
                    *slot = a + (b - a) * t;
                }
                return out;
            }
            lo = hi;
        }

        // Past the last stop, which a canvas gradient clamps to. `u` is already
        // clamped into [0,1] so this is only reached when the stop table's last
        // offset is below 1.
        let last = BAND_STOPS[BAND_STOPS.len() - 1];
        [
            last.1[0] * last.2 * self.strength,
            last.1[1] * last.2 * self.strength,
            last.1[2] * last.2 * self.strength,
        ]
    }
}

/// The finished sRGB texel for view row `y`.
///
/// `y` is a pixel CENTRE, so callers pass `row as f32 + 0.5` — a canvas gradient
/// samples the centre of the pixel it fills, and being half a texel out would tilt
/// the whole ramp.
///
/// Opaque: the backdrop is the bottom of the frame and there is nothing behind it
/// but the camera's clear colour.
pub fn sky_texel(gradient: &Gradient, glow: Option<&HorizonGlow>, y: f32, view_h: f32) -> [u8; 4] {
    let t = if view_h > 0.0 {
        clamp01(y / view_h)
    } else {
        0.0
    };
    let add = glow.map_or([0.0; 3], |g| g.add_at(y));
    let mut out = [0u8; 4];
    for c in 0..3 {
        let base = gradient.top[c] + (gradient.bottom[c] - gradient.top[c]) * t;
        // `as u8` saturates in Rust, which is the clamp the canvas applied when an
        // out-of-range `rgb()` string reached `fillStyle`.
        out[c] = (base + add[c]) as u8;
    }
    out[3] = 255;
    out
}

/// One star, placed for a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Star {
    /// View x, already wrapped into `[0, view.w)`.
    pub x: f32,
    /// View y, already wrapped into `[0, view.h)`.
    pub y: f32,
    /// Final alpha, twinkle and visibility included.
    pub alpha: f32,
}

/// The fixed star set.
///
/// Seeded once from [`hash`] and never touched again: positions live in a wrap
/// tile the size of the view, and the parallax scroll plus a modulo tiles that
/// finite set across the whole world.
#[derive(Clone, Debug)]
pub struct StarField {
    x: [f32; STAR_COUNT],
    y: [f32; STAR_COUNT],
    alpha: [f32; STAR_COUNT],
    phase: [f32; STAR_COUNT],
}

impl StarField {
    /// Seed the field.
    ///
    /// Seeds are view FRACTIONS rather than the TypeScript's pixels, because
    /// `VIEW_W`/`VIEW_H` were frozen at import there and are a resizable window
    /// here. A field in pixels would leave a bare strip down the side of a widened
    /// window until the next launch.
    pub fn new() -> StarField {
        let mut field = StarField {
            x: [0.0; STAR_COUNT],
            y: [0.0; STAR_COUNT],
            alpha: [0.0; STAR_COUNT],
            phase: [0.0; STAR_COUNT],
        };
        for i in 0..STAR_COUNT {
            // Decorrelated streams off one index, so the stars do not fall on a
            // visible lattice.
            let n = i as u32;
            field.x[i] = hash(n * 2 + 1);
            field.y[i] = hash(n * 7 + 3) * STAR_FIELD_H;
            field.alpha[i] = 0.25 + hash(n * 13 + 5) * 0.55;
            field.phase[i] = hash(n * 23 + 11) * core::f32::consts::TAU;
        }
        field
    }

    /// How visible the field is: night, closed down by depth.
    pub fn visibility(phase: DayPhase, depth: f32) -> f32 {
        phase.night * (1.0 - depth)
    }

    /// Star `i`, placed in view space.
    ///
    /// `cam` is the view's TOP-LEFT in world px and `seconds` the animation clock.
    pub fn star(&self, i: usize, cam: Vec2, view: View, seconds: f32, visibility: f32) -> Star {
        let w = view.w as f32;
        let h = view.h as f32;
        let twinkle = STAR_TWINKLE_BASE
            + STAR_TWINKLE_SWING * (seconds * STAR_TWINKLE_RATE + self.phase[i]).sin();
        Star {
            x: wrap(self.x[i] * w - cam.x * STAR_PARALLAX, w),
            y: wrap(self.y[i] * h - cam.y * STAR_PARALLAX, h),
            alpha: self.alpha[i] * visibility * twinkle,
        }
    }
}

impl Default for StarField {
    fn default() -> StarField {
        StarField::new()
    }
}

/// One sine-composed hill ridge.
#[derive(Clone, Copy, Debug)]
pub struct Ridge {
    /// Fraction of the camera's motion this ridge takes. Nearer means more.
    pub parallax: f32,
    /// Resting height as a view fraction, +y DOWN.
    pub base: f32,
    /// The slow sine's frequency, in radians per world px.
    pub f1: f32,
    /// The fast sine's frequency.
    pub f2: f32,
    /// The slow sine's amplitude, in view px.
    pub a1: f32,
    /// The fast sine's amplitude.
    pub a2: f32,
    /// Extra darkening on top of the day/night shade.
    pub shade: f32,
}

impl Ridge {
    /// The ridge's height at view x `sx`, in view px, +y DOWN.
    pub fn height_at(&self, sx: f32, cam_x: f32, view_h: f32) -> f32 {
        // World-ish x, so the ridge shifts with the camera rather than sliding
        // under a fixed wave.
        let wx = sx + cam_x * self.parallax;
        view_h * self.base + (wx * self.f1).sin() * self.a1 + (wx * self.f2).sin() * self.a2
    }

    /// This ridge's silhouette colour for a frame.
    ///
    /// Truncated per channel, which is the canvas's `| 0` and worth keeping: it is
    /// what stops the far ridge's shade rounding up into the near one's as the
    /// light falls at dusk.
    pub fn tint(&self, hill: Rgb, day: f32) -> Rgb {
        let shade = (RIDGE_SHADE_FLOOR + RIDGE_SHADE_DAY * day) * self.shade;
        [
            (hill[0] * shade).trunc(),
            (hill[1] * shade).trunc(),
            (hill[2] * shade).trunc(),
        ]
    }
}

/// A celestial disc: the soft radial sprite the sun and the moon both are.
#[derive(Clone, Copy, Debug)]
pub struct Disc {
    /// Radius in view px.
    pub radius: f32,
    /// Colour of the hot centre.
    pub core: Rgb,
    /// Colour of the tinted halo.
    pub halo: Rgb,
}

/// The sun.
pub const SUN: Disc = Disc {
    radius: SUN_R,
    core: SUN_CORE,
    halo: SUN_HALO,
};

/// The moon.
pub const MOON: Disc = Disc {
    radius: MOON_R,
    core: MOON_CORE,
    halo: MOON_HALO,
};

impl Disc {
    /// The disc's rings for a frame: `(radius in px, colour, alpha)`.
    ///
    /// `alpha` scales every stop, so a disc low on the horizon fades as a whole
    /// rather than shrinking.
    pub fn rings(&self, alpha: f32) -> [(f32, Rgb, f32); DISC_STOPS.len()] {
        let mut out = [(0.0, self.core, 0.0); DISC_STOPS.len()];
        for (slot, &(r, is_halo, a)) in out.iter_mut().zip(DISC_STOPS.iter()) {
            *slot = (
                r * self.radius,
                if is_halo { self.halo } else { self.core },
                a * alpha,
            );
        }
        out
    }
}

/// Cheap integer hash into `[0, 1)`. Deterministic per index, decent spread.
///
/// The TypeScript's second multiply overflows an `f64` mantissa — `h * 1274126177`
/// reaches about `2^62`, so its low ten bits are rounding noise by the time
/// `>>> 0` truncates them away. This is the hash that arithmetic was written to
/// be: wrapping `u32`, all 32 bits real. Star and particle LAYOUTS therefore
/// differ from the original run, which is the intended, undetectable kind of
/// difference — they were pseudo-random decoration in both.
pub fn hash(i: u32) -> f32 {
    let h = i.wrapping_mul(374_761_393);
    let h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (h & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Positive modulo, so a wrapped coordinate stays in `[0, m)`.
///
/// `f32::rem_euclid` rather than the TypeScript's `((v % m) + m) % m`: the same
/// answer in one operation, and it cannot round the second remainder back up onto
/// `m` the way the double negation can.
///
/// Shared with [`crate::weather`], which wraps its particle field into the same
/// view tile for the same reason.
#[inline]
pub(crate) fn wrap(v: f32, m: f32) -> f32 {
    if m <= 0.0 {
        return 0.0;
    }
    let r = v.rem_euclid(m);
    // `rem_euclid` adds the modulus back to a small negative remainder, and for a
    // coordinate a long way from the origin that sum can round UP onto the modulus
    // itself — which would put a star one pixel outside the buffer. Folding the
    // boundary back to zero costs a compare; widening the float costs everything.
    if r < m { r } else { 0.0 }
}

/// Clamp into `[0, 1]`.
///
/// `f32::clamp` rather than `daynight.rs`'s ternary chain: the bounds here are
/// literals, so the panic that function is guarding against cannot happen, and
/// `clamp` propagates a NaN input exactly as the chain does.
#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Clamp into a channel's `[0, 255]`.
#[inline]
fn clamp255(v: f32) -> f32 {
    v.clamp(0.0, 255.0)
}

// ---------------------------------------------------------------------------
// Drawing primitives
// ---------------------------------------------------------------------------

/// A 0..255 colour and an alpha, as the linear RGBA a vertex attribute wants.
///
/// Vertex colours reach the shader untouched and the render target does the
/// linear-to-sRGB conversion on write, so a colour authored in sRGB has to be
/// converted here or the whole backdrop comes out washed out.
pub(crate) fn linear(rgb: Rgb, alpha: f32) -> [f32; 4] {
    Color::srgb(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        .with_alpha(alpha)
        .to_linear()
        .to_f32_array()
}

/// View space — origin at the view's top-left, +y DOWN — into the backdrop's local
/// space, whose origin is the view CENTRE and whose +y is UP.
///
/// THIS IS THE ONE PLACE THE BACKDROP FLIPS. Every model function above works in
/// the TypeScript's screen space and every vertex below goes through here, so
/// there is exactly one sign to get wrong and it is on this line.
pub(crate) fn view_to_local(x: f32, y: f32, view: View) -> Vec2 {
    Vec2::new(x - view.w as f32 * 0.5, view.h as f32 * 0.5 - y)
}

/// Triangles under construction, reused across frames.
///
/// A `Local<VertexBuf>` per drawing system, so a per-frame rebuild allocates only
/// while the geometry is still growing — the same discipline the TypeScript's
/// preallocated typed arrays kept, in the one place this port still needs it.
#[derive(Default)]
pub(crate) struct VertexBuf {
    position: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    color: Vec<[f32; 4]>,
    index: Vec<u32>,
}

impl VertexBuf {
    /// Drop last frame's triangles, keeping the allocation.
    pub(crate) fn clear(&mut self) {
        self.position.clear();
        self.uv.clear();
        self.color.clear();
        self.index.clear();
    }

    /// Push a vertex in LOCAL space and return its index.
    pub(crate) fn vertex(&mut self, p: Vec2, color: [f32; 4]) -> u32 {
        let i = self.position.len() as u32;
        self.position.push([p.x, p.y, 0.0]);
        // Zero, and never read. `ColorMaterial`'s shader reads `mesh.uv`
        // unconditionally and that field only exists when the mesh declares the
        // attribute, so this is here to make the pipeline build, not to sample
        // anything.
        self.uv.push([0.0, 0.0]);
        self.color.push(color);
        i
    }

    /// Push a triangle from three existing vertices.
    pub(crate) fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.index.extend_from_slice(&[a, b, c]);
    }

    /// Push a quad from four corners in local space, wound in order.
    pub(crate) fn quad(&mut self, corners: [Vec2; 4], colors: [[f32; 4]; 4]) {
        let a = self.vertex(corners[0], colors[0]);
        let b = self.vertex(corners[1], colors[1]);
        let c = self.vertex(corners[2], colors[2]);
        let d = self.vertex(corners[3], colors[3]);
        self.tri(a, b, c);
        self.tri(a, c, d);
    }

    /// Push a flat-coloured rectangle given in VIEW space.
    ///
    /// This is `fillRect`: `(x, y)` is the TOP-LEFT and `w`/`h` extend right and
    /// DOWN, exactly as the canvas took them.
    pub(crate) fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, view: View, color: [f32; 4]) {
        self.quad(
            [
                view_to_local(x, y, view),
                view_to_local(x + w, y, view),
                view_to_local(x + w, y + h, view),
                view_to_local(x, y + h, view),
            ],
            [color; 4],
        );
    }

    /// Replace a mesh's geometry with what has been built.
    ///
    /// Takes the buffers rather than cloning them: every system here rebuilds from
    /// empty, and this is what keeps a full backdrop rebuild allocation-free once
    /// the vectors have reached their steady size.
    pub(crate) fn write(&mut self, mesh: &mut Mesh) {
        // A mesh with no vertices is not a mesh that draws nothing — it is a mesh
        // the renderer cannot allocate a slab for, and `bevy_render`'s mesh
        // allocator then reports
        // "Use-after-free: attempted to copy element data for an unallocated key"
        // once per empty mesh per frame. It is noisy rather than fatal, but it is
        // a real invariant being violated and it buried every other log line.
        //
        // Every buffer here legitimately empties: the stars are cut off in
        // daylight, the discs are both below the horizon twice a cycle, and a
        // weather layer is empty in a biome that has no dust. So instead of
        // asking four call sites to remember, one degenerate triangle stands in —
        // three coincident points at the origin at zero alpha. It allocates, it
        // rasterises no fragments, and it costs one triangle.
        if self.position.is_empty() {
            self.position.extend_from_slice(&[[0.0; 3]; 3]);
            self.uv.extend_from_slice(&[[0.0; 2]; 3]);
            self.color.extend_from_slice(&[[0.0; 4]; 3]);
            self.index.extend_from_slice(&[0, 1, 2]);
        }

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            core::mem::take(&mut self.position),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, core::mem::take(&mut self.uv));
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, core::mem::take(&mut self.color));
        mesh.insert_indices(Indices::U32(core::mem::take(&mut self.index)));
    }
}

/// A vertex-coloured triangle mesh, ready to be rewritten every frame.
///
/// It starts as the same degenerate triangle [`VertexBuf::write`] falls back to,
/// and for the same reason: these are spawned in `PostStartup` and the first
/// `place_*` does not run until the next frame, so an empty one here is an
/// unallocatable mesh for one frame at every launch.
pub(crate) fn dynamic_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0f32; 4]; 3])
    .with_inserted_indices(Indices::U32(vec![0, 1, 2]))
}

/// The material the canvas's `lighter` composite became.
///
/// Empty on purpose: it binds nothing, ships no shader, and rides the default
/// mesh2d pass, which returns the interpolated vertex colour and nothing else. The
/// whole material is [`Material2d::specialize`] — see the module header.
#[derive(Asset, TypePath, AsBindGroup, Clone, Copy, Debug, Default)]
pub struct AdditiveMaterial {}

impl Material2d for AdditiveMaterial {
    fn alpha_mode(&self) -> AlphaMode2d {
        // Blend, so the mesh is queued into the transparent phase and sorted by z
        // against the rest of the backdrop. The blend STATE that mode picks is
        // then replaced below; what is borrowed here is the sorting.
        AlphaMode2d::Blend
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = descriptor.fragment.as_mut()
            && let Some(Some(target)) = fragment.targets.first_mut()
        {
            target.blend = Some(BlendState {
                // `dst + src * srcAlpha`, which is canvas `lighter` exactly.
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                // The destination is the opaque backdrop; leave its alpha alone.
                alpha: BlendComponent {
                    src_factor: BlendFactor::Zero,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            });
        }
        Ok(())
    }
}

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
    /// `1 x view.h` sRGB: one texel per view row, stretched across the width.
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
        let resolved = resolve_atmosphere(&noise, 0);

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
    let gradient = images.add(gradient_image(1));

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

/// A `1 x rows` sRGB strip: one texel per view row.
fn gradient_image(rows: u32) -> Image {
    Image::new_fill(
        Extent3d {
            width: 1,
            height: rows.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    )
}

/// Resolve the biome atmosphere and the camera's depth for this frame.
fn sample_atmosphere(focus: Res<WorldFocus>, noise: Res<SkyNoise>, mut atmo: ResMut<Atmosphere>) {
    let column = (focus.x / CELL_SIZE as f32).floor() as i32;
    atmo.resolved = resolve_atmosphere(&noise.0, column);
    atmo.depth = depth_at(focus.y);
}

/// Rewrite the gradient strip and stretch it over the view.
fn paint_gradient(
    frame: Frame,
    backdrop: Res<Backdrop>,
    mut images: ResMut<Assets<Image>>,
    mut sprite: Single<&mut Sprite, With<SkyGradient>>,
) {
    let view = frame.view();
    let rows = view.h.max(1) as usize;
    sprite.custom_size = Some(Vec2::new(view.w as f32, view.h as f32));

    let Some(mut image) = images.get_mut(&backdrop.gradient) else {
        return;
    };
    if image.texture_descriptor.size.height as usize != rows {
        image.resize(Extent3d {
            width: 1,
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
        let texel = sky_texel(&gradient, glow.as_ref(), row as f32 + 0.5, view.h as f32);
        data[row * 4..row * 4 + 4].copy_from_slice(&texel);
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

/// One disc as a fan of rings, centre outward.
fn push_disc(buf: &mut VertexBuf, disc: &Disc, centre: Vec2, alpha: f32, view: View) {
    let rings = disc.rings(alpha);
    let step = core::f32::consts::TAU / DISC_SEGMENTS as f32;
    let at = |angle_sin: f32, angle_cos: f32, r: f32| {
        view_to_local(centre.x + angle_cos * r, centre.y + angle_sin * r, view)
    };

    for segment in 0..DISC_SEGMENTS {
        let (s0, c0) = (segment as f32 * step).sin_cos();
        let (s1, c1) = ((segment + 1) as f32 * step).sin_cos();

        // The innermost ring has zero radius, so its two edges are the same point:
        // that band is a triangle, not a quad.
        let (r, color, a) = rings[1];
        let hub = buf.vertex(at(0.0, 0.0, 0.0), linear(rings[0].1, rings[0].2));
        let near = linear(color, a);
        let n0 = buf.vertex(at(s0, c0, r), near);
        let n1 = buf.vertex(at(s1, c1, r), near);
        buf.tri(hub, n0, n1);

        for band in 1..rings.len() - 1 {
            let (r0, color0, a0) = rings[band];
            let (r1, color1, a1) = rings[band + 1];
            let inner = linear(color0, a0);
            let outer = linear(color1, a1);
            buf.quad(
                [
                    at(s0, c0, r0),
                    at(s1, c1, r0),
                    at(s1, c1, r1),
                    at(s0, c0, r1),
                ],
                [inner, inner, outer, outer],
            );
        }
    }
}

/// Rebuild both hill ridges.
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

    for ridge in &RIDGES {
        let color = linear(ridge.tint(hill, day), 1.0);
        let mut x0 = 0.0f32;
        let mut y0 = ridge.height_at(x0, cam_x, h);
        while x0 < w {
            // The last span is short whenever the view is not a whole number of
            // steps wide. The canvas closed its path straight to the bottom-right
            // corner there, which drew the same short last facet this does.
            let x1 = (x0 + RIDGE_STEP_PX).min(w);
            let y1 = ridge.height_at(x1, cam_x, h);
            buf.quad(
                [
                    view_to_local(x0, y0, view),
                    view_to_local(x1, y1, view),
                    view_to_local(x1, h, view),
                    view_to_local(x0, h, view),
                ],
                [color; 4],
            );
            x0 = x1;
            y0 = y1;
        }
    }

    buf.write(&mut mesh);
}

/// A sim-side colour as this module's.
///
/// `godgame-core` carries backdrop colours as `f64` because the biome blend that
/// produces them shares its arithmetic with the world generator, which is `f64`
/// throughout for parity. Nothing downstream of here needs that precision.
pub(crate) fn rgb32(c: godgame_core::sim::biomes::Rgb) -> Rgb {
    [c[0] as f32, c[1] as f32, c[2] as f32]
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::sim::biomes::Biome;

    /// A mesh handed to the renderer must never have zero vertices.
    ///
    /// This is a regression test for a real bug, and the bug is worth restating
    /// because nothing about it was visible from the game: five meshes here and
    /// in `crate::weather` are rewritten every frame, all five legitimately empty
    /// (no stars in daylight, both discs down, a biome with no dust), and an
    /// empty mesh made `bevy_render`'s allocator log
    /// "Use-after-free: attempted to copy element data for an unallocated key"
    /// **1612 times in a twelve-second run**. The frame still drew. Only the log
    /// said anything was wrong.
    #[test]
    fn an_empty_buffer_still_writes_an_allocatable_mesh() {
        let mut mesh = dynamic_mesh();
        assert_eq!(
            mesh.count_vertices(),
            3,
            "a freshly spawned mesh is empty for the frame before the first \
             place_* runs, so it has to carry the stand-in too"
        );

        // The buffer never had a single triangle pushed into it — the daylight
        // starfield case.
        let mut buf = VertexBuf::default();
        buf.clear();
        buf.write(&mut mesh);

        assert_eq!(
            mesh.count_vertices(),
            3,
            "an empty buffer must still leave something allocatable behind"
        );
        assert!(
            mesh.indices().is_some_and(|i| i.len() == 3),
            "and the indices have to match, or the draw call is malformed"
        );

        // It must also be invisible: this stands in for nothing, so it may not
        // put a pixel on screen.
        let Some(bevy::render::mesh::VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("the stand-in lost its vertex colours");
        };
        assert!(
            colors.iter().all(|c| c[3] == 0.0),
            "the stand-in triangle must be fully transparent, got {colors:?}"
        );
    }

    /// And a buffer with real geometry is not disturbed by the fallback.
    #[test]
    fn a_buffer_with_triangles_writes_exactly_those_triangles() {
        let mut mesh = dynamic_mesh();
        let mut buf = VertexBuf::default();
        buf.clear();
        buf.rect(0.0, 0.0, 2.0, 2.0, View::for_screen(1280, 800), [1.0; 4]);
        let want = buf.position.len();
        buf.write(&mut mesh);
        assert!(want > 0, "the fixture should have produced geometry");
        assert_eq!(mesh.count_vertices(), want, "the fallback must not fire");
    }

    const MIDNIGHT: f32 = 0.0;
    const SUNRISE: f32 = 0.25;
    const NOON: f32 = 0.5;
    const SUNSET: f32 = 0.75;

    /// A resolved atmosphere straight off one biome, with no blend in it.
    ///
    /// What every colour assertion below wants: a blended one would make the
    /// expected value a function of the world generator, and this module is not
    /// where that belongs.
    fn atmo_of(biome: Biome) -> ResolvedAtmosphere {
        let a = biome.def().atmo;
        ResolvedAtmosphere {
            sky_top: a.sky_top,
            sky_top_deep: a.sky_top_deep,
            sky_bottom: a.sky_bottom,
            star: a.star,
            hill: a.hill,
            ambient: a.ambient,
            weather: a.weather,
        }
    }

    fn plains() -> ResolvedAtmosphere {
        atmo_of(Biome::Plains)
    }

    fn luma(c: Rgb) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn the_sky_is_darkest_at_midnight_and_brightest_at_noon() {
        let atmo = plains();
        let night = gradient(&atmo, 0.0, DayPhase::at(MIDNIGHT));
        let noon = gradient(&atmo, 0.0, DayPhase::at(NOON));
        assert!(
            luma(noon.top) > luma(night.top),
            "noon zenith {:?} is not brighter than midnight's {:?}",
            noon.top,
            night.top
        );
        assert!(luma(noon.bottom) > luma(night.bottom));
    }

    #[test]
    fn the_night_sky_is_the_biomes_own_palette_untouched() {
        // At midnight `day` and `twilight` are both zero and the camera is at the
        // surface, so every blend in `gradient` collapses and what is left must be
        // exactly what the biome declared. This is the anchor the daytime
        // derivation hangs off: if it drifts, the whole palette has moved.
        let atmo = plains();
        let g = gradient(&atmo, 0.0, DayPhase::at(MIDNIGHT));
        assert_eq!(g.top, rgb32(atmo.sky_top));
        assert_eq!(g.bottom, rgb32(atmo.sky_bottom));
    }

    #[test]
    fn a_buried_camera_sees_the_biomes_deep_colour_and_no_sky_at_all() {
        let atmo = plains();
        let deep = gradient(&atmo, 1.0, DayPhase::at(NOON));
        assert_eq!(deep.top, rgb32(atmo.sky_top_deep), "the zenith sinks fully");
        // The horizon sinks at 0.9, deliberately: a rim of daylight survives at the
        // bottom of the view so a shallow cave does not read as flat black.
        assert!(
            luma(deep.bottom) > luma(deep.top),
            "the horizon should keep some light at full depth"
        );
    }

    #[test]
    fn the_daytime_sky_keeps_each_biomes_own_hue() {
        // The whole point of deriving day from the night palette: Desert must be
        // warm at noon and Glacier cold, with no second palette anywhere.
        let desert = gradient(&atmo_of(Biome::Desert), 0.0, DayPhase::at(NOON));
        let glacier = gradient(&atmo_of(Biome::Glacier), 0.0, DayPhase::at(NOON));
        assert!(
            desert.bottom[0] > desert.bottom[2],
            "desert noon should be warm, was {:?}",
            desert.bottom
        );
        assert!(
            glacier.bottom[2] > glacier.bottom[0],
            "glacier noon should be cold, was {:?}",
            glacier.bottom
        );
    }

    #[test]
    fn the_daylight_target_cannot_push_a_channel_past_full() {
        // `day_top` and `day_bot` multiply the biome's horizon by 1.25 and 1.5, so
        // a bright biome overshoots and the clamp is the only thing between that
        // and a colour that wraps when it is truncated.
        for biome in Biome::ALL {
            let g = gradient(&atmo_of(biome), 0.0, DayPhase::at(NOON));
            for c in 0..3 {
                assert!(
                    g.top[c] <= 255.0 && g.bottom[c] <= 255.0,
                    "{biome:?} overshot at noon: {g:?}"
                );
            }
        }
    }

    #[test]
    fn the_horizon_glow_only_burns_at_the_two_crossings() {
        assert!(horizon_glow(DayPhase::at(MIDNIGHT), 0.0, 450.0).is_none());
        assert!(horizon_glow(DayPhase::at(NOON), 0.0, 450.0).is_none());
        assert!(horizon_glow(DayPhase::at(SUNRISE), 0.0, 450.0).is_some());
        assert!(horizon_glow(DayPhase::at(SUNSET), 0.0, 450.0).is_some());
    }

    #[test]
    fn the_horizon_glow_is_shut_out_underground() {
        // A sunset seen from inside a mountain is not a sunset.
        assert!(horizon_glow(DayPhase::at(SUNSET), 1.0, 450.0).is_none());
        let shallow = horizon_glow(DayPhase::at(SUNSET), 0.2, 450.0).expect("still lit at 0.2");
        let surface = horizon_glow(DayPhase::at(SUNSET), 0.0, 450.0).expect("lit at the surface");
        assert!(shallow.strength < surface.strength);
    }

    #[test]
    fn the_glow_adds_nothing_above_the_band_and_floods_everything_below_it() {
        let glow = horizon_glow(DayPhase::at(SUNSET), 0.0, 450.0).expect("sunset is lit");
        assert_eq!(
            glow.add_at(glow.top - 50.0),
            [0.0; 3],
            "the sky above the band must be left alone"
        );
        assert_eq!(glow.add_at(glow.top), [0.0; 3], "the first stop is empty");

        // Below the last stop a canvas gradient clamps, and that is what floods the
        // ground with the horizon's warmth instead of banding it.
        let at_horizon = glow.add_at(glow.bottom);
        let below = glow.add_at(glow.bottom + 200.0);
        assert_eq!(at_horizon, below);
        assert!(at_horizon[0] > 0.0, "the horizon should be warm");
    }

    #[test]
    fn the_glow_warms_the_sky_rather_than_recolouring_it() {
        // Additive, so red gains most and blue least — the whole reason the band
        // exists rather than another lerp into the base gradient.
        let glow = horizon_glow(DayPhase::at(SUNRISE), 0.0, 450.0).expect("sunrise is lit");
        let add = glow.add_at(glow.bottom);
        assert!(
            add[0] > add[1] && add[1] > add[2],
            "not a warm wash: {add:?}"
        );
    }

    #[test]
    fn the_glow_tracks_the_sun_down_the_view_as_it_sets() {
        // The band is centred on the sun's own height, which is what makes it a
        // sunset rather than a stripe at a fixed line.
        let view_h = 450.0;
        let high = horizon_glow(DayPhase::at(0.28), 0.0, view_h).expect("just past sunrise");
        let low = horizon_glow(DayPhase::at(SUNRISE), 0.0, view_h).expect("at the crossing");
        assert!(low.bottom > high.bottom, "{low:?} vs {high:?}");
        // And it keeps its height whatever it does.
        assert!(((low.bottom - low.top) - view_h * BAND_SPAN).abs() < 1.0e-3);
    }

    #[test]
    fn the_gradient_runs_from_the_zenith_at_the_top_to_the_horizon_at_the_bottom() {
        let g = Gradient {
            top: [0.0, 0.0, 0.0],
            bottom: [255.0, 255.0, 255.0],
        };
        let top = sky_texel(&g, None, 0.5, 100.0);
        let bottom = sky_texel(&g, None, 99.5, 100.0);
        assert!(
            top[0] < 4,
            "the first row should be the zenith, was {top:?}"
        );
        assert!(
            bottom[0] > 251,
            "the last row should be the horizon, was {bottom:?}"
        );
        assert_eq!(top[3], 255, "the backdrop is opaque");
    }

    #[test]
    fn a_channel_that_overshoots_saturates_instead_of_wrapping() {
        // `gradient` is deliberately unclamped and the glow adds on top of it, so
        // this is the only clamp in the chain. Without a saturating cast a bright
        // sunset would wrap to black.
        let g = Gradient {
            top: [400.0, -20.0, 128.0],
            bottom: [400.0, -20.0, 128.0],
        };
        let texel = sky_texel(&g, None, 0.5, 1.0);
        assert_eq!([texel[0], texel[1], texel[2]], [255, 0, 128]);
    }

    #[test]
    fn the_stars_come_out_at_night_and_never_underground() {
        assert_eq!(StarField::visibility(DayPhase::at(NOON), 0.0), 0.0);
        assert_eq!(StarField::visibility(DayPhase::at(MIDNIGHT), 0.0), 1.0);
        assert_eq!(
            StarField::visibility(DayPhase::at(MIDNIGHT), 1.0),
            0.0,
            "buried is buried, whatever the hour"
        );
    }

    #[test]
    fn every_star_lands_inside_the_view_however_far_the_camera_has_gone() {
        let field = StarField::new();
        let view = View::for_screen(1440, 900);
        // Far from the origin in both directions, negative included: the wrap is
        // the only thing keeping a finite set covering an infinite world.
        for cam in [
            Vec2::ZERO,
            Vec2::new(1.0e6, 1.0e5),
            Vec2::new(-1.0e6, -1.0e5),
        ] {
            for i in 0..STAR_COUNT {
                let star = field.star(i, cam, view, 3.0, 1.0);
                assert!(
                    star.x >= 0.0 && star.x < view.w as f32,
                    "star {i} left the view at {cam:?}: x {}",
                    star.x
                );
                assert!(star.y >= 0.0 && star.y < view.h as f32);
            }
        }
    }

    #[test]
    fn stars_twinkle_around_their_own_brightness_without_ever_going_dark() {
        let field = StarField::new();
        let view = View::for_screen(1440, 900);
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for step in 0..400 {
            let star = field.star(0, Vec2::ZERO, view, step as f32 * 0.05, 1.0);
            min = min.min(star.alpha);
            max = max.max(star.alpha);
        }
        let base = field.alpha[0];
        assert!(min > 0.0, "a star should never blink fully out");
        assert!((min - base * (STAR_TWINKLE_BASE - STAR_TWINKLE_SWING)).abs() < 1.0e-3);
        assert!((max - base * (STAR_TWINKLE_BASE + STAR_TWINKLE_SWING)).abs() < 1.0e-3);
    }

    #[test]
    fn the_star_seeds_are_spread_and_not_on_a_lattice() {
        // Four decorrelated streams off one index. If the hash ever collapses,
        // every star lands in one place and the field stops being a field.
        let field = StarField::new();
        for i in 0..STAR_COUNT {
            assert!((0.0..1.0).contains(&field.x[i]));
            assert!((0.0..STAR_FIELD_H).contains(&field.y[i]));
            assert!((0.25..0.80).contains(&field.alpha[i]));
        }
        for i in 0..STAR_COUNT {
            for j in i + 1..STAR_COUNT {
                assert!(
                    (field.x[i] - field.x[j]).abs() > 1.0e-6
                        || (field.y[i] - field.y[j]).abs() > 1.0e-6,
                    "stars {i} and {j} are stacked"
                );
            }
        }
    }

    #[test]
    fn the_hash_spreads_over_the_whole_unit_interval() {
        let mut buckets = [0u32; 8];
        for i in 0..4096u32 {
            let v = hash(i);
            assert!((0.0..1.0).contains(&v), "hash({i}) = {v} escaped [0,1)");
            buckets[(v * 8.0) as usize] += 1;
        }
        // Even sampling would put 512 in each; within a third of that is a healthy
        // spread and outside it is a broken hash.
        for (b, n) in buckets.iter().enumerate() {
            assert!(*n > 340 && *n < 690, "bucket {b} held {n} of 4096");
        }
    }

    #[test]
    fn the_ridges_sink_toward_black_at_night() {
        let hill = rgb32(plains().hill);
        for ridge in &RIDGES {
            let noon = ridge.tint(hill, 1.0);
            let midnight = ridge.tint(hill, 0.0);
            assert!(
                luma(midnight) < luma(noon),
                "the ridge should darken at night"
            );
        }
    }

    #[test]
    fn the_near_ridge_is_lower_darker_and_scrolls_faster_than_the_far_one() {
        // The three cues that make the two read as depth rather than as two bands.
        let [far, near] = RIDGES;
        assert!(near.base > far.base, "+y is down: nearer means lower");
        assert!(near.shade < far.shade);
        assert!(near.parallax > far.parallax);
        let hill = rgb32(plains().hill);
        assert!(luma(near.tint(hill, 1.0)) < luma(far.tint(hill, 1.0)));
    }

    #[test]
    fn a_ridge_stays_within_its_own_amplitude_of_its_resting_height() {
        let view_h = 450.0;
        for ridge in &RIDGES {
            let rest = view_h * ridge.base;
            let swing = ridge.a1 + ridge.a2;
            for x in 0..2000 {
                let y = ridge.height_at(x as f32, 137.0, view_h);
                assert!(
                    (y - rest).abs() <= swing + 1.0e-3,
                    "ridge left its amplitude at x={x}: {y} vs {rest} +/- {swing}"
                );
            }
        }
    }

    #[test]
    fn a_ridge_scrolls_with_the_camera_at_its_own_rate() {
        // Moving the camera by `d` must move the ridge exactly as sampling `d *
        // parallax` further along it does. That equality IS the parallax; without
        // it the hills wobble instead of sliding.
        for ridge in &RIDGES {
            let a = ridge.height_at(100.0, 200.0, 450.0);
            let b = ridge.height_at(100.0 + 200.0 * ridge.parallax, 0.0, 450.0);
            assert!((a - b).abs() < 1.0e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn a_disc_is_a_hot_core_fading_to_nothing_at_its_rim() {
        for disc in [SUN, MOON] {
            let rings = disc.rings(1.0);
            assert_eq!(rings[0].0, 0.0, "the first ring is the centre point");
            assert_eq!(rings[rings.len() - 1].0, disc.radius, "the last is the rim");
            assert_eq!(rings[rings.len() - 1].2, 0.0, "and the rim is invisible");
            // Alpha falls monotonically from the centre out; anything else would
            // show as a ring rather than a glow.
            for pair in rings.windows(2) {
                assert!(pair[0].0 < pair[1].0, "rings must grow outward");
                assert!(pair[0].2 >= pair[1].2, "alpha must not rise outward");
            }
            assert_eq!(rings[0].1, disc.core);
            assert_eq!(rings[rings.len() - 1].1, disc.halo);
        }
    }

    #[test]
    fn a_disc_fades_as_a_whole_rather_than_shrinking() {
        let full = SUN.rings(1.0);
        let half = SUN.rings(0.5);
        for (a, b) in full.iter().zip(half.iter()) {
            assert_eq!(a.0, b.0, "the radius must not move with the fade");
            assert!((a.2 * 0.5 - b.2).abs() < 1.0e-6);
        }
    }

    #[test]
    fn the_moon_is_never_as_bright_or_as_warm_as_the_sun() {
        // Through `rings`, which is where the two are actually told apart: it is
        // the only thing the draw ever reads, and both discs go through it.
        let sun = SUN.rings(1.0);
        let moon = MOON.rings(MOON_DIM);
        assert!(moon[0].2 < sun[0].2, "the moon should be the dimmer disc");
        assert!(moon[3].0 < sun[3].0, "and the smaller one");
        // And the cold one, which is the other half of telling them apart at a
        // glance. The halo carries that, not the near-white core.
        let (sun_halo, moon_halo) = (sun[2].1, moon[2].1);
        assert!(moon_halo[2] > moon_halo[0], "the moon reads cold");
        assert!(sun_halo[0] > sun_halo[2], "the sun reads warm");
    }

    #[test]
    fn the_camera_is_at_the_surface_until_it_is_below_the_anchor() {
        let anchor = (SURFACE_ANCHOR_Y * CELL_SIZE) as f32;
        assert_eq!(depth_at(anchor), 0.0);
        assert_eq!(depth_at(anchor - 1000.0), 0.0, "the sky does not go deeper");
        assert_eq!(depth_at(anchor + DEPTH_SPAN_CELLS * CELL_SIZE as f32), 1.0);
        assert_eq!(depth_at(1.0e9), 1.0, "and it stops at fully buried");
        let half = depth_at(anchor + DEPTH_SPAN_CELLS * 0.5 * CELL_SIZE as f32);
        assert!((half - 0.5).abs() < 1.0e-4, "was {half}");
    }

    #[test]
    fn the_view_flip_is_the_only_thing_that_moves_a_point_between_the_two_spaces() {
        let view = View::for_screen(1440, 900);
        let w = view.w as f32;
        let h = view.h as f32;
        // The view's corners, in the TypeScript's screen space.
        assert_eq!(view_to_local(0.0, 0.0, view), Vec2::new(-w / 2.0, h / 2.0));
        assert_eq!(view_to_local(w, h, view), Vec2::new(w / 2.0, -h / 2.0));
        // And its centre, which is where the camera is.
        assert_eq!(view_to_local(w / 2.0, h / 2.0, view), Vec2::ZERO);
    }

    #[test]
    fn a_filled_rect_is_placed_from_its_top_left_with_y_running_down() {
        // `VertexBuf::rect` is the canvas `fillRect` this whole port is built on.
        // Getting its corner or its sign wrong would put the backdrop upside down
        // in a way no colour test would catch.
        let view = View::for_screen(1440, 900);
        let h = view.h as f32;
        let mut buf = VertexBuf::default();
        buf.rect(0.0, 0.0, 4.0, 2.0, view, [0.0; 4]);
        let top = buf.position[0][1];
        let bottom = buf.position[3][1];
        assert_eq!(top, h / 2.0, "the rect starts at the top of the view");
        assert!(bottom < top, "and extends downward");
        assert_eq!(top - bottom, 2.0, "by its height");
    }

    #[test]
    fn a_rebuilt_mesh_carries_every_attribute_the_pass_needs() {
        // `ColorMaterial`'s shader reads `mesh.uv` unconditionally and the default
        // mesh2d fragment returns magenta without vertex colours, so a mesh missing
        // either attribute fails at pipeline build time — a long way from here.
        let view = View::for_screen(1440, 900);
        let mut buf = VertexBuf::default();
        buf.rect(3.0, 4.0, 2.0, 2.0, view, [1.0, 1.0, 1.0, 1.0]);
        let mut mesh = dynamic_mesh();
        buf.write(&mut mesh);

        assert!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
        assert_eq!(mesh.count_vertices(), 4);
        assert_eq!(mesh.indices().map(Indices::len), Some(6));
    }

    #[test]
    fn a_vertex_buffer_is_emptied_by_a_write_and_reused_by_the_next_frame() {
        let view = View::for_screen(1440, 900);
        let mut buf = VertexBuf::default();
        let mut mesh = dynamic_mesh();
        buf.rect(0.0, 0.0, 1.0, 1.0, view, [1.0; 4]);
        buf.write(&mut mesh);
        assert_eq!(mesh.count_vertices(), 4);

        // A second frame that draws nothing must not leave LAST frame's geometry
        // behind. The systems clear before they build for exactly this reason,
        // and `write` taking the buffers is what makes the clear cheap.
        //
        // What it leaves is the stand-in triangle rather than literally nothing —
        // see `an_empty_buffer_still_writes_an_allocatable_mesh`. This test
        // originally asserted zero here, and zero is precisely what made the mesh
        // allocator log a use-after-free every frame. The invariant it is really
        // defending is "the square is gone", so that is what it checks.
        buf.clear();
        buf.write(&mut mesh);
        assert_eq!(
            mesh.count_vertices(),
            3,
            "the four-vertex square must be gone, replaced by the stand-in"
        );
        assert_eq!(mesh.indices().map(Indices::len), Some(3));
    }
}

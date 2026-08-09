//! The backdrop as arithmetic: the gradient, the glow, the stars, the ridges,
//! the discs, and the block grid they are all quantised onto.
//!
//! **No Bevy in this file, and that is the point of the boundary.** Everything
//! here is a function of the world clock, the camera's depth and the biome's
//! atmosphere, returning colours and positions. It is the half of [`super`] that
//! can be reasoned about — and tested — without a renderer, a window or an
//! `App`, which is why almost every test in the module lives here and asserts on
//! real numbers rather than on a captured frame.
//!
//! The three sections [`super`] named are kept in order: **Tuning**, which is
//! every authored number and the argument for it; **The model**, which turns
//! those into a [`Gradient`], a [`HorizonGlow`], a [`StarField`], a [`Ridge`] or
//! a [`Disc`]; and **The pixel grid**, which is the quantisation that makes the
//! backdrop read as part of a pixel game rather than as a smooth gradient
//! sitting behind one.

use bevy::math::Vec2;

use yugen_core::config::{CELL_SIZE, SURFACE_ANCHOR_Y, View};
use yugen_core::sim::biomes::ResolvedAtmosphere;

use crate::daynight::DayPhase;

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// The side of one backdrop pixel, in view px. **The knob for this whole file.**
///
/// [`CELL_SIZE`], because the backdrop's job is to read as the same MATERIAL as
/// the terrain in front of it and the terrain's feature size is one cell. Nothing
/// subtler than "match the world" survives contact with a nearest-neighbour
/// upscale: at 2x zoom a 4px backdrop block and a 5px terrain cell beat against
/// each other and the sky looks like a different game's asset.
///
/// The gradient, the discs and the ridges all take the same quantum, and that is
/// the non-obvious half of the decision. A gradient is the one element with a
/// case for a FINER grid — it is a very shallow ramp, so a coarse grid turns it
/// into a staircase of near-identical bands and the eye finds Mach edges in it
/// that are not really there. Two quanta in one backdrop would have been worse
/// than either: the sun would have sat on a grid the sky behind it did not share,
/// which is the exact mismatch this whole change exists to remove. [`BAYER`] is
/// what makes one quantum affordable — it breaks the staircase without making the
/// blocks any smaller. See [`SKY_DITHER`].
///
/// Set this to 1 and the backdrop goes back to the smooth Canvas2D original: one
/// texel per view row, a per-pixel disc falloff, a 1px ridge step. That is the
/// intended before/after, and it is a LOOKING setting rather than a shipping one
/// — every element here costs `(1/n)^2` of its geometry, so the sun alone goes
/// from about 280 quads to about 8500.
pub(super) const SKY_PIXEL_PX: i32 = CELL_SIZE;

/// How much of a block's own height the ordered dither is allowed to move the
/// sample by. 0 disables dithering; 1 spreads it over the full block.
///
/// Banding is the whole risk of quantising a sky. A block row of the daytime
/// gradient differs from the next by about one value step and a twilight one by
/// about five, and a hard edge between two near-identical colours is exactly what
/// the eye is best at inventing a line along.
///
/// Ordered dithering is the period-correct answer rather than a modern hack —
/// every hand-drawn EGA and Amiga sky is a Bayer ramp, because that is what you
/// do when you have big pixels and few colours, which is the situation this file
/// is now deliberately in. It costs nothing here: the dither does not add
/// resolution, it only decides WHERE INSIDE ITS OWN BLOCK a block samples the
/// smooth function underneath. The blocks stay exactly [`SKY_PIXEL_PX`] and stay
/// flat; the boundary between two bands stops being a straight line.
pub(super) const SKY_DITHER: f32 = 1.0;

/// The ordered-dither threshold matrix, in `0..N*N`.
///
/// The standard 4x4 Bayer matrix. At [`SKY_PIXEL_PX`] its tile is 20 view px —
/// four cells, which is a plausible size for a hand-placed dither cluster and
/// small enough not to read as a second pattern in the sky. An 8x8 would grade
/// more finely and tile at 40px; it is a drop-in replacement if the 4x4's texture
/// ever shows.
pub(super) const BAYER: [[u8; BAYER_N]; BAYER_N] =
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

/// Side of [`BAYER`].
pub(super) const BAYER_N: usize = 4;

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
pub(super) const DAY_TOP: Rgb = [96.0, 140.0, 205.0];

/// Neutral daylight target for the horizon: near-white haze.
pub(super) const DAY_BOT: Rgb = [200.0, 220.0, 240.0];

/// Sunrise/sunset zenith — a cool violet, because the sky opposite a low sun is
/// the one part of a real sunset that does NOT go warm.
pub(super) const DUSK_TOP: Rgb = [64.0, 52.0, 96.0];

/// Sunrise/sunset horizon: hot orange.
pub(super) const DUSK_BOT: Rgb = [242.0, 138.0, 74.0];

/// Below this the twilight band is skipped entirely.
///
/// The band is a full-view composite, so it is worth a compare not to run it for
/// the two thirds of the cycle where it would add less than a value step.
pub(super) const GLOW_CUTOFF: f32 = 0.02;

/// Below this the starfield is skipped.
pub(super) const STAR_CUTOFF: f32 = 0.01;

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
pub(super) const BAND_STOPS: [(f32, Rgb, f32); 3] = [
    (0.00, [240.0, 120.0, 60.0], 0.00),
    (0.55, [244.0, 132.0, 66.0], 0.30),
    (1.00, [255.0, 186.0, 110.0], 0.62),
];

/// Where the band's bottom sits, as a view fraction, when the sun is at the top of
/// its arc.
pub(super) const BAND_HORIZON_BASE: f32 = 0.52;

/// How far the band's bottom tracks down the view as the sun sinks.
///
/// The glow is centred on the sun's own height so it follows the sun across the
/// horizon instead of sitting at a fixed line.
pub(super) const BAND_HORIZON_TRACK: f32 = 0.30;

/// The band's height as a view fraction. Just under half the view: high enough to
/// read as sky, low enough to leave the zenith alone.
pub(super) const BAND_SPAN: f32 = 0.42;

/// How many stars the field holds.
pub const STAR_COUNT: usize = 90;

/// Fraction of the view height the star seeds are spread over.
///
/// The upper 70% only — stars are seeded away from the horizon so the field does
/// not fight the hills. It is a seeding bias and not a hard ceiling: the parallax
/// wrap in [`StarField::star`] can carry any star anywhere.
pub(super) const STAR_FIELD_H: f32 = 0.7;

/// Twinkle rate, radians per second.
///
/// The TypeScript wrote this as `performance.now() * 0.0016` — milliseconds, so
/// 1.6 rad/s, which is what this is.
pub(super) const STAR_TWINKLE_RATE: f32 = 1.6;

/// Twinkle floor: the fraction of its own brightness a star never dips below.
pub(super) const STAR_TWINKLE_BASE: f32 = 0.78;

/// Twinkle swing either side of [`STAR_TWINKLE_BASE`].
pub(super) const STAR_TWINKLE_SWING: f32 = 0.22;

/// How much of the camera's motion the starfield takes. Distant, so almost none.
pub(super) const STAR_PARALLAX: f32 = 0.1;

/// The sun's radius in view px.
pub(super) const SUN_R: f32 = 46.0;

/// The moon's radius in view px. Smaller and cooler than the sun.
pub(super) const MOON_R: f32 = 34.0;

/// The sun's hot white centre.
pub(super) const SUN_CORE: Rgb = [255.0, 244.0, 214.0];

/// The sun's halo, which is where its warmth actually lives.
pub(super) const SUN_HALO: Rgb = [255.0, 176.0, 92.0];

/// The moon's cold white centre.
pub(super) const MOON_CORE: Rgb = [244.0, 248.0, 255.0];

/// The moon's halo: blue, the whole reason it does not read as a second sun.
pub(super) const MOON_HALO: Rgb = [150.0, 170.0, 210.0];

/// How much of its own visibility the moon is drawn at.
///
/// The moon is the same soft disc as the sun and would read as a second sun at
/// full strength. Held back a tenth so it stays the dimmer of the two.
pub(super) const MOON_DIM: f32 = 0.9;

/// Radial stops of a celestial disc: `(radius fraction, is-halo, alpha)`.
///
/// A hot core that holds most of its alpha to a third of the radius, a fast
/// hand-off to the tinted halo, then a long fade to nothing. `false` reads the
/// disc's core colour, `true` its halo.
pub(super) const DISC_STOPS: [(f32, bool, f32); 4] = [
    (0.00, false, 0.95),
    (0.32, false, 0.70),
    (0.42, true, 0.28),
    (1.00, true, 0.00),
];

/// Below this a disc is not drawn at all.
pub(super) const DISC_CUTOFF: f32 = 0.01;

/// The two hill ridges, far first.
///
/// Each is two summed sines, which is enough to stop the silhouette reading as a
/// repeating wave without being a heightmap. The far ridge sits higher, moves
/// slower and catches more of the sky's light; the near one is lower, faster and
/// nearly black, and the difference between the two IS the depth cue.
pub(super) const RIDGES: [Ridge; 2] = [
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

/// Ridge brightness floor: what is left of a ridge's colour at midnight.
///
/// Both ridges sink toward black at night so the horizon reads as a silhouette
/// rather than as two grey bands.
pub(super) const RIDGE_SHADE_FLOOR: f32 = 0.4;

/// How much of a ridge's colour daylight adds back on top of the floor.
pub(super) const RIDGE_SHADE_DAY: f32 = 0.6;

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
pub(super) const DEPTH_SPAN_CELLS: f32 = 300.0;

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
/// `y` is a sample POSITION inside whatever the caller is filling, not a row
/// index. A canvas gradient samples the centre of the pixel it fills, and being
/// half a texel out would tilt the whole ramp; [`paint_gradient`] fills
/// [`SKY_PIXEL_PX`]-tall blocks rather than rows, so it passes a position inside
/// the block chosen by [`block_sample`], whose mean is that same centre.
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

    /// The top of the block column covering ridge-space x `ridge_x`, snapped to
    /// the backdrop's pixel grid.
    ///
    /// Takes a position in the RIDGE's own space rather than a screen x, and that
    /// is the whole point: this is a constant per column, so the silhouette a
    /// frame draws is the same set of columns translated, never a set of columns
    /// whose heights changed. See [`build_ridges`].
    pub fn column_top(&self, ridge_x: f32, view_h: f32) -> f32 {
        // Sampled at the column's CENTRE, so a column represents the hill across
        // its own width rather than at its left edge. At this feature size half a
        // block of phase error is half a cell of hill, which is visible.
        snap_to_block(self.height_at(ridge_x + block_px() * 0.5, 0.0, view_h))
    }

    /// How far along its own space this ridge has scrolled, in whole view px.
    ///
    /// Rounded, so the strip lands on the buffer's pixels and its blocks stay
    /// exactly [`SKY_PIXEL_PX`] wide instead of straddling a pixel boundary.
    /// Rounding to the BLOCK grid instead would make the hills jump a whole cell
    /// at a time, which for a layer this close reads as stutter rather than as
    /// parallax.
    pub fn scroll(&self, cam_x: f32) -> f32 {
        (cam_x * self.parallax).round()
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
pub(super) fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Clamp into a channel's `[0, 255]`.
#[inline]
pub(super) fn clamp255(v: f32) -> f32 {
    v.clamp(0.0, 255.0)
}

// ---------------------------------------------------------------------------
// The pixel grid
// ---------------------------------------------------------------------------

/// One backdrop block in view px, floored at one.
///
/// The floor is what makes `SKY_PIXEL_PX = 1` the "smooth original" setting
/// rather than a division by zero, and it is the only place the constant is read
/// as a length.
#[inline]
pub(super) fn block_px() -> f32 {
    SKY_PIXEL_PX.max(1) as f32
}

/// Where inside its own block the block at `(bx, by)` samples, as a fraction in
/// `(0, 1)`.
///
/// This is the entire dither. A block is one flat colour and stays one flat
/// colour; all that moves is which row of the smooth function underneath it took
/// that colour from. Two vertically adjacent block rows whose ideal colours
/// differ by a single value step therefore stop meeting along a straight line and
/// start interleaving over a 4-block-wide ramp — the staircase goes, the block
/// size does not.
///
/// The `+ 0.5` on the matrix entry centres the sixteen thresholds on 0.5, so the
/// MEAN sample position is the block centre. Without it the whole sky would sit
/// half a threshold high, which on a ramp this shallow is a real, if tiny, tilt.
///
/// `rem_euclid` rather than `%` because block indices are signed: the disc's
/// lattice is built outward from its own centre and half its columns are
/// negative.
pub(super) fn block_sample(bx: i32, by: i32) -> f32 {
    let n = BAYER_N as i32;
    let m = f32::from(BAYER[by.rem_euclid(n) as usize][bx.rem_euclid(n) as usize]);
    let t = (m + 0.5) / (BAYER_N * BAYER_N) as f32;
    0.5 + SKY_DITHER * (t - 0.5)
}

/// A view coordinate snapped to the nearest block edge.
#[inline]
pub(super) fn snap_to_block(v: f32) -> f32 {
    (v / block_px()).round() * block_px()
}

/// A [`Disc::rings`] ramp evaluated at radius `r`, or `None` where it adds
/// nothing.
///
/// The canvas's `createRadialGradient` was a piecewise-linear ramp between its
/// stops and this is that ramp, read on the CPU because a block is a flat colour
/// and something has to decide which one. Interpolating here rather than across
/// vertices also puts the lerp back in sRGB, where the canvas did it.
///
/// `None` past the rim AND at zero alpha: an additive draw at alpha zero adds
/// nothing at all, so the caller can drop the block instead of emitting a quad
/// that rasterises to a no-op. That is most of the bounding square of a disc.
pub(super) fn ring_at(rings: &[(f32, Rgb, f32)], r: f32) -> Option<(Rgb, f32)> {
    let last = rings.last()?;
    if r >= last.0 {
        return None;
    }
    let mut lo = *rings.first()?;
    for &hi in &rings[1..] {
        if r <= hi.0 {
            let t = if hi.0 > lo.0 {
                clamp01((r - lo.0) / (hi.0 - lo.0))
            } else {
                1.0
            };
            let alpha = lo.2 + (hi.2 - lo.2) * t;
            if alpha <= 0.0 {
                return None;
            }
            let mut color = [0.0; 3];
            for (slot, (lo_c, hi_c)) in color.iter_mut().zip(lo.1.iter().zip(hi.1.iter())) {
                *slot = lo_c + (hi_c - lo_c) * t;
            }
            return Some((color, alpha));
        }
        lo = hi;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::rgb32;
    use super::*;
    use yugen_core::sim::biomes::Biome;

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
    fn the_dither_matrix_is_a_real_ordered_dither() {
        // Every threshold once and only once is what MAKES a matrix an ordered
        // dither: it is the property that guarantees a block row visits all
        // sixteen sample positions and none of them twice, so the interleave is
        // even. A hand-edit that duplicates an entry silently clumps the pattern.
        let mut seen = [false; BAYER_N * BAYER_N];
        for row in BAYER {
            for m in row {
                let m = m as usize;
                assert!(m < seen.len(), "threshold {m} is outside 0..{}", seen.len());
                assert!(!seen[m], "threshold {m} appears twice");
                seen[m] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "the matrix has a hole in it");
    }

    #[test]
    fn a_dithered_block_samples_inside_itself_and_averages_to_its_own_centre() {
        // Two things, and both are load-bearing whatever `SKY_DITHER` is set to.
        //
        // Inside itself: a block that sampled past its own edge would be showing
        // the neighbouring band's colour, which is not dithering, it is being
        // wrong.
        //
        // Averaging to the centre: the dither must not move the ramp. On a sky
        // this shallow a half-threshold bias would tilt the whole gradient, and it
        // is the `+ 0.5` in `block_sample` that stops it.
        let mut total = 0.0;
        for by in 0..BAYER_N as i32 {
            for bx in 0..BAYER_N as i32 {
                let s = block_sample(bx, by);
                assert!(s > 0.0 && s < 1.0, "block ({bx},{by}) sampled at {s}");
                total += s;
            }
        }
        let mean = total / (BAYER_N * BAYER_N) as f32;
        assert!(
            (mean - 0.5).abs() < 1.0e-6,
            "the dither is biased: mean {mean}"
        );
    }

    #[test]
    fn the_dither_pattern_tiles_and_takes_negative_block_indices() {
        // The disc builds its lattice outward from its own centre, so half its
        // columns are negative. `%` would hand those back negative and index out
        // of the matrix; `rem_euclid` is what makes the tile continuous across the
        // origin instead of mirrored about it.
        let n = BAYER_N as i32;
        for by in -2 * n..2 * n {
            for bx in -2 * n..2 * n {
                assert_eq!(block_sample(bx, by), block_sample(bx + n, by + n));
            }
        }
    }

    #[test]
    fn the_dither_spreads_a_band_edge_over_more_than_one_colour() {
        // The reason this file dithers at all, asserted on the steepest ramp the
        // sky ever draws: a sunset's horizon glow, which climbs about five value
        // steps per block. Undithered that is sixteen hard horizontal lines
        // through the warmest part of the sky. Dithered, a block row spans more
        // than one colour and the lines interleave away.
        //
        // This asserts at the shipping `SKY_DITHER`. Turning the knob to zero is
        // meant to fail here — that is the knob doing what it says.
        let view_h = 400.0;
        let g = gradient(&plains(), 0.0, DayPhase::at(SUNSET));
        let glow = horizon_glow(DayPhase::at(SUNSET), 0.0, view_h).expect("sunset is lit");
        let block = block_px();

        let mut rows_with_a_mixed_edge = 0;
        for row in 0..(view_h / block) as i32 {
            let mut colours = std::collections::BTreeSet::new();
            for bx in 0..BAYER_N as i32 {
                let y = (row as f32 + block_sample(bx, row)) * block;
                colours.insert(sky_texel(&g, Some(&glow), y, view_h));
            }
            if colours.len() > 1 {
                rows_with_a_mixed_edge += 1;
            }
        }
        assert!(
            rows_with_a_mixed_edge > 20,
            "only {rows_with_a_mixed_edge} block rows dithered; the glow spans about \
             {} of them and every one of those should",
            (view_h * BAND_SPAN / block) as i32
        );
    }

    #[test]
    fn a_ridge_column_is_on_the_grid_and_does_not_move_with_the_camera() {
        // The boiling bug, in test form. If a column's height were a function of
        // where the camera is, then as the camera crept each column would pop to
        // the next lattice step at its own moment and the silhouette would seethe
        // instead of sliding. `column_top` takes ridge space precisely so it
        // cannot.
        let view_h = 400.0;
        let block = block_px();
        for ridge in &RIDGES {
            for j in -400..400 {
                let ridge_x = j as f32 * block;
                let top = ridge.column_top(ridge_x, view_h);
                assert!(
                    (top / block - (top / block).round()).abs() < 1.0e-3,
                    "column {j} topped at {top}, which is not on the {block}px grid"
                );
            }
            // And the scroll is what moves it: a camera `d` further along shifts
            // the strip by `scroll(d)` and changes nothing else.
            let far = ridge.scroll(1.0e4);
            assert!(far.fract() == 0.0, "the scroll must be whole px, was {far}");
            assert!(
                (far - 1.0e4 * ridge.parallax).abs() <= 0.5,
                "the rounding must not change the parallax rate"
            );
        }
    }

    #[test]
    fn a_ridge_silhouette_is_the_same_shape_wherever_the_camera_stands() {
        // The stronger form: run the column walk `build_ridges` runs, at two
        // camera positions far apart, and every column the two have in common must
        // come out at the same height. That is what "one rigid strip, translated"
        // means, and it is the property a screen-space lattice would not have.
        let (view_h, view_w) = (400.0, 640.0);
        let block = block_px();
        let walk = |ridge: &Ridge, cam_x: f32| {
            let shift = ridge.scroll(cam_x);
            let first = (shift / block).floor() as i32;
            let last = ((shift + view_w) / block).ceil() as i32;
            (first..last)
                .map(|j| (j, ridge.column_top(j as f32 * block, view_h)))
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        for ridge in &RIDGES {
            let near = walk(ridge, 0.0);
            // Far enough that the two strips overlap by only part of a view, so
            // the shared columns are a real intersection and not the whole set —
            // and near enough that the faster ridge, at parallax 0.55, has not
            // scrolled clean past the slower one's span.
            let far = walk(ridge, 800.0);
            let shared: Vec<_> = near.keys().filter(|j| far.contains_key(j)).collect();
            assert!(
                !shared.is_empty(),
                "the two walks should still overlap at parallax {}",
                ridge.parallax
            );
            for j in shared {
                assert_eq!(
                    near[j], far[j],
                    "column {j} changed height when the camera moved"
                );
            }
        }
    }

    #[test]
    fn a_disc_block_reads_the_same_ramp_the_ring_fan_drew() {
        // `ring_at` replaced GPU interpolation between ring vertices, so it has to
        // agree with the stops it interpolates or the sun changes colour.
        let rings = SUN.rings(1.0);
        let (core, a) = ring_at(&rings, 0.0).expect("the centre is lit");
        assert_eq!(core, SUN.core, "the centre is the core colour");
        assert!((a - rings[0].2).abs() < 1.0e-6);
        // On a stop, exactly that stop.
        for &(r, color, alpha) in &rings[..rings.len() - 1] {
            let (c, a) = ring_at(&rings, r).expect("a stop inside the rim is lit");
            assert_eq!(c, color, "the ramp drifted off its own stop at r={r}");
            assert!((a - alpha).abs() < 1.0e-6);
        }
        // And nothing at or past the rim, which is what lets the caller drop three
        // quarters of the bounding square instead of emitting invisible quads.
        assert!(ring_at(&rings, SUN.radius).is_none());
        assert!(ring_at(&rings, SUN.radius + 1.0).is_none());
    }

    #[test]
    fn a_disc_fades_outward_block_by_block() {
        // Monotone alpha, sampled at every block ring rather than at the four
        // stops: a block circle shows its ramp as visible rings, and one that rose
        // outward would read as a halo with a dark moat in it.
        for disc in [SUN, MOON] {
            let rings = disc.rings(1.0);
            let mut last = f32::MAX;
            let mut steps = 0;
            let mut r = 0.0;
            while r < disc.radius {
                if let Some((_, a)) = ring_at(&rings, r) {
                    assert!(a <= last + 1.0e-6, "alpha rose outward at r={r}");
                    last = a;
                    steps += 1;
                }
                r += block_px();
            }
            assert!(
                steps > 3,
                "a {}px disc should be more than {steps} blocks deep",
                disc.radius
            );
        }
    }

    #[test]
    fn the_backdrop_grid_matches_the_world_it_sits_behind() {
        // The claim the whole file is built on. If someone retunes `SKY_PIXEL_PX`
        // off `CELL_SIZE` this fails and points at the module header, which is
        // where the argument for keeping them equal is written down.
        assert_eq!(
            SKY_PIXEL_PX, CELL_SIZE,
            "the sky's pixel and the world's cell are meant to be the same size"
        );
        assert_eq!(block_px(), CELL_SIZE as f32);
        assert_eq!(snap_to_block(0.4 * CELL_SIZE as f32), 0.0);
        assert_eq!(snap_to_block(0.6 * CELL_SIZE as f32), CELL_SIZE as f32);
        assert_eq!(snap_to_block(-0.6 * CELL_SIZE as f32), -(CELL_SIZE as f32));
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
}

//! Bloom and vignette: the two passes applied after the light is solved.
//!
//! [`super`]'s "Bloom" and "Vignette" sections. Both take the solved grid and
//! produce something to composite over it — a bright-pass and blur for bloom, a
//! radial falloff and the flat washes for vignette — and both are separated
//! from the solver because they are decisions about how the light LOOKS rather
//! than about what it is.

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

use yugen_core::config::{CELL_SIZE, View};
use yugen_core::sim::materials::{CellId, MAT_COUNT};

use super::model::*;
use super::solver::*;

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
pub(super) fn smooth_knee(v: f32, lo: f32, hi: f32) -> f32 {
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
pub(super) const VIGNETTE_REBAKE_EPS: f32 = 0.002;

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

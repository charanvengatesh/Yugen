//! The cell rasteriser — one packed RGBA `u32` per visible cell.
//!
//! A 1:1 port of the TypeScript `src/render/ChunkCanvas.ts`. Despite that file's
//! name there is no per-chunk canvas cache in it, and there is none here: the
//! class it is named for is one screen-sized scratch buffer repainted every
//! frame. Its header, kept below, is why.
//!
//! # What is a screen-sized-in-cells scratch buffer
//!
//! Each frame it reads just the cells the camera can see straight out of the
//! (shared) material array into one ImageData, which the renderer then upscales
//! to the screen. Cost is fixed at ~one pixel per visible cell regardless of how
//! big the world is — the culling *is* the viewport crop, so there is no
//! per-chunk bookkeeping to keep in sync with the sim worker.
//!
//! There is deliberately no dirty-rect repaint here. It would need the worker's
//! per-chunk dirty mask to be visible on the main thread (it is not — the masks
//! are per-instance and the worker's are the live ones), which means another
//! shared buffer and another cross-thread invariant, to save part of a pass that
//! measures at a fraction of a millisecond for the whole viewport. The pass
//! below is instead made cheap enough that repainting all of it is the right
//! trade.
//!
//! THAT DECISION IS WHY MATERIAL TEXTURE, RIM LIGHT AND AMBIENT OCCLUSION ALL
//! LIVE IN THIS ONE LOOP. There is no chunk cache to hang them off — this pass
//! IS the repaint — so every one of them had to collapse into table reads that
//! ride along with the colour lookup already happening. None of them adds a
//! second walk of the grid, and none adds a material load: see the running
//! `next_id` / `depth_above` bookkeeping below.
//!
//! # What changed in the port
//!
//! Only the shape of the state, never the arithmetic:
//!
//!   - The immutable build-once tables (`TEX_A`, `TEX_B`, `TEX_OFF_*`, `PAT_D`,
//!     `SIN`, `IS_EMIT`, the shimmer id/phase lists) are `LazyLock` statics,
//!     which is exactly what a JS module-level `const` initialised at load is.
//!   - `SHADE32` is NOT, because [`CellShades::update_shimmer`] writes it every
//!     frame. A mutable global is not available without `unsafe` or interior
//!     mutability, and neither is wanted here, so the table is an owned
//!     [`CellShades`] the caller holds and passes in. This is the one signature
//!     change against the original.
//!   - The emitter census was a module static plus an `emitters()` accessor; it
//!     is returned from [`paint_cells`] instead, for the same reason.
//!   - `ViewBuffer` is not ported. It is the canvas/ImageData wrapper — the
//!     platform half — and this module is the pure half: grid in, one `u32` per
//!     cell out. That split is what lets the whole thing be diffed against the
//!     TypeScript byte for byte in `tests/cells_golden.rs`.
//!
//! # Bevy-free source, in a crate that is not
//!
//! This module deliberately does not touch Bevy: nothing it imports names it,
//! and the pass below is pure computation over `godgame-core` types — no `App`,
//! no window, no GPU, and nothing that would stop it living in `godgame-core`
//! today.
//!
//! IT DOES NOT, AND THAT COSTS SOMETHING. `tests/cells_golden.rs` is a test
//! target of `godgame-render`, and `godgame-render` depends on `bevy` — so the
//! parity suite links the whole engine in order to check a colour table. This is
//! exactly the arrangement the crate boundary exists to prevent, and which
//! `godgame-core`'s worldgen and noise suites do get: they never link a renderer.
//!
//! The fix is to move this module into `godgame-core`, and it HAS NOT BEEN DONE.
//! It is not a rename. The shader surface at the bottom of this file names
//! [`crate::cellmap`] and would have to point the other way across the crate
//! boundary; `cells_golden.rs` and its fixture move with the module, while
//! `shader_matches_cpu.rs` cannot follow it — that one wants a GPU and belongs
//! where the renderer is. That is a change made on its own, not one made in
//! passing. Until it happens, read "does not touch Bevy" as a property of the
//! SOURCE and not of what the test binary links.
//!
//! # Why CPU, when the frame is drawn by a shader
//!
//! [`paint_cells`] is a pure function — grid in, one `u32` per cell out — so it
//! can be diffed against the original exactly the way the worldgen port was.
//! The shader (`cells.wgsl`) now draws the frame, and this stays because having
//! a verified CPU reference is what turned "does the shader look right" into
//! "does the shader match this table". `tests/shader_matches_cpu.rs` renders the
//! shipping WGSL headlessly over a real worldgen window and requires every cell
//! of a non-animated material to be the exact byte this file packed.
//!
//! So this module is THE ORACLE, and it is also the readable statement of what
//! the shading is: the shader restates the arithmetic and nothing else, and
//! points back here by name for every reason. Do not delete it, and do not let
//! it drift — the scalars the shader cannot import are re-exported at the bottom
//! of this file and checked against the shader's own literals.

use std::sync::LazyLock;

use godgame_core::sim::grid::CellGrid;
use godgame_core::sim::materials::{
    MAT_B, MAT_COLORVAR, MAT_COUNT, MAT_EDGE, MAT_G, MAT_LIGHT, MAT_R, MAT_SHIMMER, MAT_TEXTURE,
    Tex,
};

// --- Texture patterns --------------------------------------------------------

/// Pattern INDICES, 0..7 — the tile arrays are index-addressed, one contiguous
/// 61x61 and 67x67 slab per pattern.
///
/// The content compiler emits `Tex` as one-hot BIT VALUES (flat = 1, speckle =
/// 2, grain = 4 ... glassy = 128) so that `MAT_TEXTURE[c] & (GRAIN | LAYERED)`
/// is a legal "either of these" test. Bits are the right shape for a mask and
/// the wrong shape for an array subscript, so `Pat` below is the same eight
/// patterns in the same order, as indices, and [`tex_index`] converts. The
/// declaration order is identical to the compiler's, so index == log2(bit) and
/// the two can never silently disagree — [`assert_tex_order`] checks exactly
/// that when the tables are first touched.
mod pat {
    pub const FLAT: usize = 0;
    pub const SPECKLE: usize = 1;
    pub const GRAIN: usize = 2;
    pub const CRYSTALLINE: usize = 3;
    pub const LAYERED: usize = 4;
    pub const FIBROUS: usize = 5;
    pub const MOLTEN: usize = 6;
    pub const GLASSY: usize = 7;
}
const TEX_COUNT: usize = 8;

/// One-hot bit -> index. `leading_zeros` makes this exact for every legal `Tex`
/// value (the TypeScript spelled it `31 - Math.clz32(bit)`).
fn tex_index(bit: u8) -> usize {
    if bit == 0 {
        pat::FLAT
    } else {
        (31 - u32::from(bit).leading_zeros()) as usize
    }
}

/// Guard the one assumption that connects this file to the content compiler:
/// that `pat` and `Tex` list the same patterns in the same order. If a pattern
/// is ever inserted in the middle of the compiler's enum, every block silently
/// renders with its neighbour's texture — a bug that looks like an art problem
/// and would be hunted for in the wrong file. Cheap to check once at load;
/// impossible to miss when it fires.
fn assert_tex_order() {
    const EXPECT: [(&str, Tex, usize); TEX_COUNT] = [
        ("flat", Tex::FLAT, pat::FLAT),
        ("speckle", Tex::SPECKLE, pat::SPECKLE),
        ("grain", Tex::GRAIN, pat::GRAIN),
        ("crystalline", Tex::CRYSTALLINE, pat::CRYSTALLINE),
        ("layered", Tex::LAYERED, pat::LAYERED),
        ("fibrous", Tex::FIBROUS, pat::FIBROUS),
        ("molten", Tex::MOLTEN, pat::MOLTEN),
        ("glassy", Tex::GLASSY, pat::GLASSY),
    ];
    for (name, bit, want) in EXPECT {
        let got = tex_index(bit.bits() as u8);
        assert!(
            got == want,
            "Tex/pat mismatch for \"{name}\": content bit {} maps to index {got}, \
             renderer expects {want}",
            bit.bits()
        );
    }
}

// --- Pattern tiles -----------------------------------------------------------

// THE DITHER REPEAT, AND WHAT WAS ACTUALLY WRONG WITH IT.
//
// The previous pass replaced a per-pixel hash with a 64x64 tile indexed by
// `(x & 63, y & 63)`, and flagged the 64-cell repeat as a risk. Rendering a flat
// 200x200 sand field and autocorrelating it horizontally confirms the repeat is
// real — r = 1.000 at a shift of exactly 64 cells — but it also turned up a
// worse, closer-range problem the tile size was hiding:
//
//   shift  8 -> r = 0.63      shift 21 -> r = 0.53      shift 56 -> r = 0.58
//
// That structure is not the tile. It is the HASH. The tile was seeded with
// `(imul(x, 374761393) ^ imul(y, 668265263)) >>> 8 & 63` — a multiply-xor with
// no avalanche step, so for consecutive x the sampled bits 8..13 march through a
// short arithmetic cycle. On screen a flat sand field reads as woven fabric: a
// hard 8-cell (40 px) pinstripe lattice, far more visible than the 64-cell
// repeat ever was.
//
// So the fix is two independent things, and both keep the LUT:
//
//  1. FINALIZE THE HASH. `hash2` below is a full xorshift-multiply finaliser
//     (the same construction `light.ts:hashPhase` already uses). This alone
//     removes every correlation peak above r = 0.1 inside the tile.
//  2. COPRIME TILES. Two tiles, 61x61 and 67x67, indexed independently and
//     SUMMED. The composite repeats only at lcm(61, 67) = 4087 cells — 20 435
//     screen px, an order of magnitude wider than any viewport, so there is no
//     shift at which the field can line up with itself.
//
// Summing (rather than the suggested XOR) matters for the structured patterns:
// XOR of two band fields is noise, whereas the sum of two band fields at
// coprime periods is still a band field, just one that never repeats. That is
// what lets `layered` produce genuine strata instead of horizontal static.
//
// Cost is unchanged in shape — two table reads instead of one, with the wrap
// done by a running counter and a compare rather than a modulo (61 and 67 are
// not powers of two, so `%` would be a real division in the hot loop).
const PA: i32 = 61;
const PB: i32 = 67;

/// Per-tile pattern amplitude, 0..31 each, so a summed sample is 0..62.
const TEX_LEVELS: i32 = 32;
/// Sum of the two tiles: 0..62, allocated 6 bits.
const PAT_BITS: u32 = 6;
/// 64 — the number of pattern levels a shade slice carries.
const PAT_LEVELS: usize = 1 << PAT_BITS;
/// 31 — the neutral, no-offset sample.
const PAT_MID: f64 = (TEX_LEVELS - 1) as f64;

/// Overall texture strength. 1.0 means one `MAT_COLORVAR` unit per sigma, which
/// is ~1.7x the old uniform dither's RMS. Set to 0.72 by eye against rendered
/// test fields: at 1.0 the coarse octaves stop reading as texture and start
/// reading as a second material, which is a different bug from the one being
/// fixed.
const TEX_GAIN: f64 = 0.72;

/// Target standard deviation of a single normalised tile, in its 0..31 range.
const TILE_SIGMA: f64 = 8.5;
/// …so the SUM of two independent tiles has this sigma in its 0..62 range.
const PAT_SIGMA: f64 = TILE_SIGMA * std::f64::consts::SQRT_2;

/// `(p - PAT_MID) / PAT_SIGMA` for every pattern level — the texture offset,
/// before the material's own colour variance scales it.
///
/// It is a constant, and the shimmer rebuild was recomputing it inside its inner
/// loop: one float DIVIDE per table entry, times 512 entries, times the nine
/// materials the content set now declares shimmer on. Four and a half thousand
/// divides a frame to re-derive sixty-four numbers that never change.
static PAT_D: LazyLock<[f64; PAT_LEVELS]> = LazyLock::new(|| {
    let mut d = [0.0f64; PAT_LEVELS];
    for (p, slot) in d.iter_mut().enumerate() {
        *slot = (p as f64 - PAT_MID) / PAT_SIGMA;
    }
    d
});

const TAU: f64 = std::f64::consts::PI * 2.0;

/// Integer hash with a real avalanche finaliser. The absence of these three
/// xorshift-multiply rounds is what put the 8-cell lattice on every flat field
/// in the game; they are not optional garnish.
///
/// Deliberately NOT `Noise::hash2` from `godgame-core`: that one is salted by a
/// world seed and uses a different finaliser (`0x2c1b3c6d` / `0x297a2d39`).
/// These tiles are world-independent, and swapping the constants would rebuild
/// every pattern.
fn hash2(x: i32, y: i32, seed: i32) -> f64 {
    let mut h = (x as u32).wrapping_mul(0x27d4_eb2d)
        ^ (y as u32).wrapping_mul(0x1656_67b1)
        ^ (seed as u32).wrapping_mul(0x9e37_79b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    f64::from(h) / 4294967296.0
}

/// ECMAScript `Math.round`: half rounds toward +Infinity, and a value strictly
/// below the halfway point never rounds up.
///
/// Rust's `f64::round` rounds half AWAY FROM ZERO, so it disagrees with JS on
/// every negative half-integer — `round(-0.5)` is -1 there and 0 here. The naive
/// `(v + 0.5).floor()` is wrong in the other direction: it rounds
/// `0.49999999999999994` up, because the add itself rounds to 0.5. Comparing the
/// fractional part is the only form that is right at both ends.
///
/// NOTE: the parity fixture cannot currently tell this apart from `f64::round`.
/// Both call sites feed it a sum of sines and blurred noise, and an exact tie in
/// such a value is measure-zero — swapping this for `v.round()` still passes
/// every case. It stays because the semantics, not the sample, are what is being
/// ported: change a wave harmonic or a quantisation step count and the tie
/// becomes reachable, and it would then be wrong silently.
fn js_round(v: f64) -> f64 {
    let f = v.floor();
    if v - f >= 0.5 { f + 1.0 } else { f }
}

/// ECMAScript `ToInt32`: truncate toward zero, then wrap modulo 2^32 into the
/// signed 32-bit range. This is what a JS `&` does to a float operand.
fn to_int32(v: f64) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    (v.trunc().rem_euclid(4294967296.0) as u32) as i32
}

/// Build one pattern tile of period `p`. Everything a pattern is made of has to
/// be exactly periodic in `p` or the tile seams:
///  - hashed white noise wraps for free, because the tile is indexed mod p,
///  - a sine at an INTEGER harmonic `sin(TAU * (kx*x + ky*y) / p)` wraps exactly,
///  - a box blur wraps if its window wraps.
///
/// There is no value-noise lattice here for that reason: 61 and 67 are prime, so
/// no coarse lattice divides them, and a non-dividing one would seam.
fn build_tile(tex: usize, p: i32, seed: i32, out: &mut [u8]) {
    let pu = p as usize;
    let n = pu * pu;
    let mut noise = vec![0.0f64; n];
    for y in 0..p {
        for x in 0..p {
            noise[(y * p + x) as usize] = hash2(x, y, seed) * 2.0 - 1.0;
        }
    }

    /// Integer harmonic wave — periodic in p by construction.
    fn wave(p: i32, x: i32, y: i32, kx: i32, ky: i32, ph: f64) -> f64 {
        ((TAU * f64::from(kx * x + ky * y)) / f64::from(p) + ph).sin()
    }

    // Wrapping box blurs of the white-noise field, built ONCE as whole fields
    // rather than sampled per pixel. Blurring white noise is how this file gets
    // scale without a value-noise lattice: a radius-r box average has most of
    // its energy at wavelengths around 2r, so stacking a few radii gives the
    // octaves a fBm would, and every one of them wraps because the window wraps.
    //
    // ANISOTROPY IS THE POINT. `bx` blurs along x only and leaves y sharp, which
    // is literally what bedded sediment looks like; `by` is its transpose and is
    // what makes wood read as fibre. That directionality is not available from
    // an isotropic noise function, which is why the patterns are built this way.
    fn blur_x(p: i32, src: &[f64], r: i32) -> Vec<f64> {
        let pu = p as usize;
        let mut dst = vec![0.0f64; pu * pu];
        let w = 1.0 / f64::from(r * 2 + 1);
        for y in 0..p {
            let row = (y * p) as usize;
            for x in 0..p {
                let mut s = 0.0;
                for d in -r..=r {
                    let mut k = x + d;
                    if k < 0 {
                        k += p;
                    } else if k >= p {
                        k -= p;
                    }
                    s += src[row + k as usize];
                }
                dst[row + x as usize] = s * w;
            }
        }
        dst
    }
    fn blur_y(p: i32, src: &[f64], r: i32) -> Vec<f64> {
        let pu = p as usize;
        let mut dst = vec![0.0f64; pu * pu];
        let w = 1.0 / f64::from(r * 2 + 1);
        for y in 0..p {
            for x in 0..p {
                let mut s = 0.0;
                for d in -r..=r {
                    let mut k = y + d;
                    if k < 0 {
                        k += p;
                    } else if k >= p {
                        k -= p;
                    }
                    s += src[(k * p + x) as usize];
                }
                dst[(y * p + x) as usize] = s * w;
            }
        }
        dst
    }
    // Isotropic octave: separable box blur, so cost is 2*(2r+1) per cell.
    let blob = |r: i32| blur_y(p, &blur_x(p, &noise, r), r);

    // Octaves. A box blur cuts variance by roughly sqrt(2r+1) per axis, so each
    // is renormalised to unit-ish scale before mixing; the final tile is
    // standardised anyway, but keeping the octaves comparable makes the mix
    // weights mean what they say.
    let bx3 = blur_x(p, &noise, 3); // horizontal smear, ~7 cells long
    let by3 = blur_y(p, &noise, 3); // vertical streak
    let blob1 = blob(1); // ~3-cell clumps
    let blob4 = blob(4); // ~9-cell blotches

    /// Reciprocal RMS of a field, so octaves can be mixed by weight not by luck.
    fn unit(f: &[f64]) -> f64 {
        let mut s = 0.0;
        for &v in f {
            s += v * v;
        }
        // `|| 1` in the original: a zero (or NaN) RMS must not divide.
        let rms = (s / f.len() as f64).sqrt();
        1.0 / if rms == 0.0 || rms.is_nan() { 1.0 } else { rms }
    }
    let kx3 = unit(&bx3);
    let ky3 = unit(&by3);
    let k1 = unit(&blob1);
    let k4 = unit(&blob4);

    /// Snap to `steps` plateaus — hard facet / stratum boundaries, not a ramp.
    fn quant(v: f64, steps: f64) -> f64 {
        let t = (v + 1.0) * 0.5;
        (js_round(t * steps) / steps) * 2.0 - 1.0
    }

    let mut raw = vec![0.0f64; n];
    for y in 0..p {
        for x in 0..p {
            let i = (y * p + x) as usize;
            let nz = noise[i];
            let mut v = 0.0;

            match tex {
                pat::FLAT => {
                    // Barely there. Enough to stop large panes banding, not
                    // enough to give a smooth material a texture it should not
                    // have.
                    v = blob1[i] * k1 * 0.18 + nz * 0.12;
                }

                pat::SPECKLE => {
                    // Isotropic mineral rock. NOT white noise: per-pixel noise
                    // at this amplitude reads as television static, which is
                    // precisely the "noise-coloured mush" this pass exists to
                    // remove. Rock has structure at several scales at once —
                    // blotches of a slightly different mineral, clumps within
                    // them, grit on top — so this is three octaves with the
                    // energy weighted toward the COARSE end. The fine octave is
                    // the only per-cell term and it carries less than a fifth of
                    // the variance.
                    v = blob4[i] * k4 * 0.5 + blob1[i] * k1 * 1.05 + nz * 0.45;
                }

                pat::GRAIN => {
                    // Anisotropic: noise smeared over 7 cells horizontally and
                    // left sharp vertically, so dirt and sand read as bedded
                    // sediment rather than static. The high-k y wave lays the
                    // beds down, and a coarse blotch octave keeps a long wall
                    // from looking like corduroy.
                    v = bx3[i] * kx3 * 1.15
                        + blob4[i] * k4 * 0.45
                        + wave(p, x, y, 0, 9, 0.0) * 0.28
                        + wave(p, x, y, 1, 17, 1.9) * 0.15;
                }

                pat::CRYSTALLINE => {
                    // Three interfering waves quantised into plateaus give
                    // angular facets with straight boundaries; the sparse tail
                    // of the hash adds the occasional bright glint sitting on a
                    // facet.
                    let f = wave(p, x, y, 3, 2, 0.0) * 0.9
                        + wave(p, x, y, -2, 3, 1.1) * 0.7
                        + wave(p, x, y, 5, 4, 2.3) * 0.45;
                    v = quant(f * 0.5, 5.0) * 0.95;
                    let glint = hash2(x, y, seed + 77);
                    if glint > 0.978 {
                        v = 1.6; // clamped on store — a hard specular pop
                    }
                }

                pat::LAYERED => {
                    // Horizontal banding for sandstone and the deep strata. The
                    // bands are quantised so they have edges, and given a very
                    // slight x tilt so a wide wall does not look like a ruled
                    // page.
                    let f = wave(p, x, y, 0, 6, 0.0) * 0.85
                        + wave(p, x, y, 0, 11, 2.1) * 0.4
                        + wave(p, x, y, 2, 0, 0.5) * 0.13;
                    // The grit rides the beds rather than the whole wall, so a
                    // stratum looks like packed sediment instead of a painted
                    // stripe.
                    v = quant(f * 0.55, 6.0) * 0.88 + bx3[i] * kx3 * 0.3 + nz * 0.12;
                }

                pat::FIBROUS => {
                    // The transpose of `grain`: noise smeared vertically, plus
                    // strong high-k x waves, so wood reads as long vertical
                    // fibre.
                    v = by3[i] * ky3 * 1.2
                        + wave(p, x, y, 7, 0, 0.0) * 0.42
                        + wave(p, x, y, 13, 1, 1.4) * 0.2;
                }

                pat::MOLTEN => {
                    // Cooling-crust cells: quantised blobs with hard boundaries
                    // that the shimmer animation crawls through (see
                    // `update_shimmer`). High contrast, because lava should be
                    // the brightest thing on screen.
                    let f = wave(p, x, y, 2, 3, 0.0) * 0.9
                        + wave(p, x, y, -3, 2, 1.7) * 0.6
                        + wave(p, x, y, 6, 5, 0.4) * 0.35;
                    v = quant(f * 0.55, 4.0) * 0.9 + blob1[i] * k1 * 0.35 + nz * 0.18;
                }

                pat::GLASSY => {
                    // Smooth, with rare narrow specular streaks along one
                    // diagonal — enough to say "this surface reflects" without
                    // giving glass grain.
                    v = wave(p, x, y, 1, 2, 0.0) * 0.3 + wave(p, x, y, 3, -1, 1.2) * 0.18;
                    let streak = wave(p, x, y, 4, 3, 0.6);
                    if streak > 0.965 {
                        v += 1.1;
                    }
                }

                _ => {}
            }

            raw[i] = v;
        }
    }

    // VARIANCE NORMALISATION. Each pattern is a different mix of noise, waves
    // and quantisation steps, so their raw spreads differ by 3-4x — `flat`
    // barely moves, `layered` slams between plateaus. Quantising them all on a
    // fixed scale would make the amplitude an accident of how a pattern happens
    // to be written rather than a property of the material, and `MAT_COLORVAR`
    // would stop meaning anything consistent.
    //
    // So each tile is standardised to a target sigma before quantising. Two
    // independent tiles are summed, so the composite sigma is
    // TILE_SIGMA * sqrt(2) ~= 12 in the 0..62 range: about 2.6 sigma to each
    // end, which fills the range without spending most of the table on clipped
    // extremes. `MAT_COLORVAR` is then applied against that known sigma in
    // `build_material_shades`, so one unit of declared colour variance means the
    // same visual swing for every pattern.
    //
    // Outliers (the crystalline glint, the glassy streak) clip to the ends on
    // purpose — a specular pop should saturate.
    let mut mean = 0.0;
    for &v in &raw {
        mean += v;
    }
    mean /= n as f64;
    let mut var_sum = 0.0;
    for &v in &raw {
        let d = v - mean;
        var_sum += d * d;
    }
    let sd = (var_sum / n as f64).sqrt();
    let sd = if sd == 0.0 || sd.is_nan() { 1.0 } else { sd };
    let scale = TILE_SIGMA / sd;

    let base = tex * n;
    let mid = f64::from(TEX_LEVELS - 1) * 0.5;
    for i in 0..n {
        let mut q = js_round(mid + (raw[i] - mean) * scale);
        if q < 0.0 {
            q = 0.0;
        } else if q > f64::from(TEX_LEVELS - 1) {
            q = f64::from(TEX_LEVELS - 1);
        }
        out[base + i] = q as u8;
    }
}

/// The 61x61 tile set, eight patterns, one contiguous slab each.
pub static TEX_A: LazyLock<Vec<u8>> = LazyLock::new(|| {
    assert_tex_order();
    let mut out = vec![0u8; TEX_COUNT * (PA * PA) as usize];
    for t in 0..TEX_COUNT {
        build_tile(t, PA, 0x1234 + t as i32 * 131, &mut out);
    }
    out
});

/// The 67x67 tile set. Coprime with [`TEX_A`] on purpose — see the note above.
pub static TEX_B: LazyLock<Vec<u8>> = LazyLock::new(|| {
    assert_tex_order();
    let mut out = vec![0u8; TEX_COUNT * (PB * PB) as usize];
    for t in 0..TEX_COUNT {
        build_tile(t, PB, 0x9e37 + t as i32 * 197, &mut out);
    }
    out
});

/// Element offset of a material's pattern within [`TEX_A`]. One L1 read.
static TEX_OFF_A: LazyLock<[i32; MAT_COUNT]> = LazyLock::new(|| {
    let mut off = [0i32; MAT_COUNT];
    for (id, slot) in off.iter_mut().enumerate() {
        *slot = tex_index(MAT_TEXTURE[id]) as i32 * PA * PA;
    }
    off
});

/// Element offset of a material's pattern within [`TEX_B`].
static TEX_OFF_B: LazyLock<[i32; MAT_COUNT]> = LazyLock::new(|| {
    let mut off = [0i32; MAT_COUNT];
    for (id, slot) in off.iter_mut().enumerate() {
        *slot = tex_index(MAT_TEXTURE[id]) as i32 * PB * PB;
    }
    off
});

// --- Edge / occlusion codes --------------------------------------------------

/// A cell's lighting class, 3 bits, laid out as `depth_above | (side_open << 2)`:
///
///   bits 0-1  depth_above — cells between this one and the nearest air ABOVE it
///             in the same column, capped at 3. 0 means the cell directly under
///             open air (the top face), 3 means "buried, no top light".
///   bit 2     side_open   — this cell touches air to the left or right.
///
/// WHY THIS AND NOT A 4-NEIGHBOUR STENCIL. A stencil needs `mat[i-cols]` and
/// `mat[i+cols]` per cell, which is two extra scattered loads a row apart —
/// cache-hostile, and the single most expensive thing that could be added to
/// this loop. `depth_above` gets the same information for free: the loop already
/// walks rows top-to-bottom, so a per-column running counter that resets on air
/// and SATURATES AT 3 IS the vertical neighbourhood, at one small-array
/// increment per cell. It is also strictly better looking than a binary top-edge
/// test, because it gives a two-cell falloff under every surface instead of a
/// single hard line, which is what makes terrain read as lit from above rather
/// than outlined.
///
/// The saturation is also what makes the eventual shader port possible: a
/// counter capped at 3 only ever needs to look three cells up, so the scan
/// becomes three texture taps at fixed offsets rather than a serial dependency
/// down the column.
///
/// `side_open` is likewise free: the loop reads each cell's material once and
/// carries it forward as `prev_id`/`next_id`, so the horizontal neighbourhood
/// costs a register, not a load.
///
/// Ambient occlusion falls out of the same code: class 3 (buried, no side) is
/// the darkest entry in the shade table, so the interior of a rock mass sinks
/// away from its lit faces without a second pass.
const EDGE_CODES: usize = 8;

/// Brightness offset per edge class, as a fraction of the material's `MAT_EDGE`.
/// Positive lifts (rim), negative sinks (occlusion).
const EDGE_GAIN: [f64; EDGE_CODES] = [
    0.62,  // 0: directly under air — the top face, full rim
    0.26,  // 1: one cell down — the falloff that reads as thickness
    0.07,  // 2: two cells down — nearly neutral
    -0.24, // 3: buried on all counted sides — ambient occlusion
    0.74,  // 4: top face AND a side open — an exposed corner, brightest
    0.36,  // 5: one down, side open
    0.19,  // 6: two down, side open — a vertical wall face catching side light
    0.08,  // 7: buried but side-open — a cliff face, lifted off the interior
];

// --- Shade table -------------------------------------------------------------

/// `3 = log2(EDGE_CODES)`.
const SHADE_SHIFT: u32 = PAT_BITS + 3;
/// 512 entries per material.
const SHADE_STRIDE: usize = 1 << SHADE_SHIFT;

/// Byte order of a 32-bit store into an RGBA byte buffer.
///
/// The original probed it at load with a one-element `Uint32Array` aliased as
/// bytes; there is no equivalent question to ask at runtime here, so it is
/// resolved at COMPILE time. Rust is never going to run this big-endian, but the
/// packing order is what the parity fixture holds, so it is spelled out rather
/// than assumed.
const LITTLE_ENDIAN: bool = cfg!(target_endian = "little");

/// `f64::clamp` is the same three comparisons the original's ternary chain was,
/// including on the edges that matter for a port: it leaves NaN as NaN (which
/// [`pack`] then turns into 0, exactly as `ToInt32(NaN)` does) and leaves -0.0
/// alone, because `-0.0 < 0.0` is false in both languages.
fn clamp255(v: f64) -> f64 {
    v.clamp(0.0, 255.0)
}

/// Pack three 0..255 channel values into one RGBA word, alpha forced opaque.
///
/// The channels arrive as FLOATS and are truncated toward zero, not rounded —
/// that is what a JS `|` / `<<` on a float operand does (ToInt32), and `as u32`
/// on a clamped non-negative `f64` is the same operation.
fn pack(r: f64, g: f64, b: f64) -> u32 {
    let (r, g, b) = (r as u32, g as u32, b as u32);
    if LITTLE_ENDIAN {
        (255 << 24) | (b << 16) | (g << 8) | r
    } else {
        (r << 24) | (g << 16) | (b << 8) | 255
    }
}

/// Finished RGBA pixel per (material, edge class, pattern sample). Air (code 0)
/// is left at 0 — fully transparent, so the sky shows through with no special
/// case in the loop.
///
/// 53 materials x 8 edge classes x 64 pattern levels x 4 bytes = 106 KB. Larger
/// than the old 12 KB table, but the WORKING SET is what matters and that is
/// per-material: a viewport showing six materials touches 6 x 2 KB = 12 KB of
/// it, the same as before. Everything the frame needs — the material's colour,
/// its texture amplitude, its rim strength, its occlusion, and (for emissive
/// blocks) this frame's shimmer — is folded in here at build time so the inner
/// loop is one indexed read.
///
/// This is owned rather than static because [`Self::update_shimmer`] rewrites it
/// every frame; see the module header.
#[derive(Clone, Debug)]
pub struct CellShades {
    shade32: Vec<u32>,
}

impl Default for CellShades {
    fn default() -> Self {
        Self::new()
    }
}

impl CellShades {
    /// Build every material's slice at rest (no shimmer lift).
    pub fn new() -> CellShades {
        let mut s = CellShades {
            shade32: vec![0u32; MAT_COUNT * SHADE_STRIDE],
        };
        for id in 1..MAT_COUNT {
            s.build_material_shades(id, 0.0, 0.0);
        }
        s
    }

    /// The raw table, for the parity suite and the eventual GPU upload.
    pub fn table(&self) -> &[u32] {
        &self.shade32
    }

    /// (Re)build one material's 512-entry slice. `lift` is an additive
    /// brightness in 0..255 units applied to the whole slice — 0 at rest, driven
    /// by the animation clock for shimmering materials.
    ///
    /// `warm` biases the lift toward red/orange as it rises, which is what stops
    /// an animated lava cell from simply going grey-bright: hot things shift
    /// hue, they do not just gain luminance.
    fn build_material_shades(&mut self, id: usize, lift: f64, warm: f64) {
        let r0 = f64::from(MAT_R[id]);
        let g0 = f64::from(MAT_G[id]);
        let b0 = f64::from(MAT_B[id]);
        // Texture amplitude comes off the authored colour variance, so a
        // material that declares itself uniform stays uniform whatever pattern
        // it carries.
        let amp = f64::from(MAT_COLORVAR[id]);
        let edge = f64::from(MAT_EDGE[id]) * (1.0 / 255.0);
        let base = id * SHADE_STRIDE;
        let pat_d = &*PAT_D;

        for (e, &gain) in EDGE_GAIN.iter().enumerate() {
            // Rim/AO is scaled to the material's own colour so a dark rock gets
            // a proportionate highlight rather than a grey wash: 34 units at
            // full MAT_EDGE, which is roughly a two-stop lift on mid-tone
            // terrain.
            let ed = gain * edge * 52.0;
            let e_base = base + (e << PAT_BITS);
            for (p, &pd) in pat_d.iter().enumerate() {
                // Pattern sample 0..62 -> signed swing over the material's own
                // variance. Divided by PAT_SIGMA, not by the range, so one unit
                // of MAT_COLORVAR is one standard deviation of colour swing
                // whatever pattern is in play.
                let d = pd * amp * TEX_GAIN;
                let t = d + ed + lift;
                self.shade32[e_base + p] = pack(
                    clamp255(r0 + t + lift * warm * 0.55),
                    clamp255(g0 + t + lift * warm * 0.12),
                    clamp255(b0 + t - lift * warm * 0.3),
                );
            }
        }
    }

    /// Advance the animated materials. `t` is seconds from a single global
    /// clock.
    ///
    /// WHY PALETTE CYCLING, AND NOT AN OVERLAY PASS.
    ///
    /// The two options for animating emissive cells were a second world-space
    /// pass that touches only emissive cells, or swapping the colour LUT per
    /// frame while the cell loop stays untouched. This is the LUT swap, for
    /// three reasons:
    ///
    ///  1. It costs nothing per cell. An overlay pass has to FIND the emissive
    ///     cells, which means a second walk of the visible grid every frame —
    ///     the exact cost this module exists to avoid. The LUT swap rebuilds
    ///     `SHIMMER_N x 512` entries regardless of how much lava is on screen:
    ///     about 2 500 table writes for the current content set, flat, whether
    ///     the view is one ember or a whole magma chamber.
    ///  2. It cannot force a repaint, because there is no cached repaint to
    ///     force. The requirement that animation must not dirty a chunk cache is
    ///     satisfied structurally here: the cell buffer is rebuilt every frame
    ///     anyway, so an animated palette is free rather than invalidating.
    ///  3. It animates the material's TEXTURE, not just its brightness. The lift
    ///     is modulated by the pattern index `p`, so the wave crawls through the
    ///     `molten` crust pattern instead of pulsing the whole pool in unison —
    ///     a lava lake visibly flows. An overlay pass drawn over finished pixels
    ///     could not do that without re-deriving the pattern it was drawn from.
    ///
    /// The one thing this cannot do is animate two cells of the SAME material at
    /// different phases by position alone. It does not need to: the pattern
    /// index is a function of position, so position-varying phase comes back
    /// through `p`.
    ///
    /// NOTE FOR THE SHADER MILESTONE: those ~2 500 palette entries rebuilt every
    /// frame are the single biggest CPU cost in the TypeScript renderer, and
    /// they exist ONLY because there was no cached repaint to invalidate — the
    /// whole trick is that a per-frame LUT rewrite is cheaper than finding the
    /// emissive cells twice. On the GPU the same animation is a uniform: `t`
    /// goes in, the shader evaluates the two sines per fragment, and this
    /// function disappears entirely. It is ported faithfully here because the
    /// CPU reference has to match the original before the shader can be checked
    /// against the CPU reference.
    pub fn update_shimmer(&mut self, t: f64) {
        let pat_d = &*PAT_D;
        let sin = &*SIN;
        let mut shimmer_lift = [0.0f64; PAT_LEVELS];

        for &id in SHIMMER_IDS.iter() {
            let amp = f64::from(MAT_SHIMMER[id]) * (1.0 / 255.0) * 46.0;
            let ph = SHIMMER_PHASE[id];
            let r0 = f64::from(MAT_R[id]);
            let g0 = f64::from(MAT_G[id]);
            let b0 = f64::from(MAT_B[id]);
            let tex_amp = f64::from(MAT_COLORVAR[id]) * TEX_GAIN;
            let edge = f64::from(MAT_EDGE[id]) * (1.0 / 255.0);
            let base = id * SHADE_STRIDE;

            // The lift depends on the pattern index but NOT on the edge class,
            // so it is resolved once for the 64 pattern levels and reused across
            // all 8 classes. Without this hoist the wave was evaluated 512 times
            // per material instead of 64 — measured at 0.084 ms/frame, more than
            // the entire cell blit, for a value that was identical eight times
            // over.
            let mut a0 = (t * 2.1 + ph) * SIN_SCALE;
            let mut a1 = (t * 0.77 + ph * 1.7) * SIN_SCALE;
            let d0 = 0.29 * SIN_SCALE;
            let d1 = 0.11 * SIN_SCALE;
            for slot in shimmer_lift.iter_mut() {
                // Phase advances with the pattern index: the bright band travels
                // across the crust rather than the whole surface flashing
                // together. Two rates beat against each other so the flow never
                // looks like a clean sawtooth.
                let w = sin[(to_int32(a0) & SIN_MASK) as usize] * 0.62
                    + sin[(to_int32(a1) & SIN_MASK) as usize] * 0.38;
                *slot = amp * (w * 0.5 + 0.5);
                a0 += d0;
                a1 += d1;
            }

            for (e, &gain) in EDGE_GAIN.iter().enumerate() {
                let ed = gain * edge * 52.0;
                let e_base = base + (e << PAT_BITS);
                for (p, &pd) in pat_d.iter().enumerate() {
                    let lift = shimmer_lift[p];
                    let s = pd * tex_amp + ed + lift;
                    self.shade32[e_base + p] = pack(
                        clamp255(r0 + s + lift * 0.55),
                        clamp255(g0 + s + lift * 0.12),
                        clamp255(b0 + s - lift * 0.3),
                    );
                }
            }
        }
    }
}

// --- Animated materials ------------------------------------------------------

/// Materials whose `MAT_SHIMMER` is worth animating, resolved once. Only these
/// slices are rebuilt per frame.
///
/// The floor is not zero. A shimmer of 24/255 peaks at a lift of about 4 units
/// of 8-bit colour, which is below what the eye resolves as motion on a 5px cell
/// and below the material's own texture contrast — so the weakest emitters (a
/// copper ore's faint glint, a cactus) would cost a full 512-entry table rebuild
/// every frame to produce an animation nobody can see. They keep their static
/// emissive colour and are simply not in this list.
const SHIMMER_MIN: u8 = 24;

static SHIMMER_IDS: LazyLock<Vec<usize>> = LazyLock::new(|| {
    (1..MAT_COUNT)
        .filter(|&id| MAT_SHIMMER[id] >= SHIMMER_MIN)
        .collect()
});

/// Per-material phase offset, so lava and crystal do not breathe in lockstep.
static SHIMMER_PHASE: LazyLock<[f64; MAT_COUNT]> = LazyLock::new(|| {
    let mut ph = [0.0f64; MAT_COUNT];
    for &id in SHIMMER_IDS.iter() {
        ph[id] = hash2(id as i32, 7, 3) * TAU;
    }
    ph
});

/// Sine table for the shimmer wave. A power-of-two size means the phase wraps
/// with a mask rather than a modulo, and — because ToInt32 wraps at 2^31, an
/// exact multiple of `SIN_SIZE` — the wrap stays phase-continuous however long
/// the session runs.
///
/// 2048 entries is ~0.18 degrees of resolution, far finer than a lift quantised
/// to 8-bit colour can express. `sin` is not cheap enough to call thousands of
/// times a frame for a value this coarse.
const SIN_SIZE: usize = 2048;
const SIN_MASK: i32 = SIN_SIZE as i32 - 1;
const SIN_SCALE: f64 = SIN_SIZE as f64 / TAU;

static SIN: LazyLock<[f64; SIN_SIZE]> = LazyLock::new(|| {
    let mut s = [0.0f64; SIN_SIZE];
    for (i, slot) in s.iter_mut().enumerate() {
        *slot = ((i as f64 / SIN_SIZE as f64) * TAU).sin();
    }
    s
});

// --- Emitter census ----------------------------------------------------------

/// Cap on the census. Beyond this the extra emitters are simply not reported —
/// the light pass has a fixed budget anyway.
const EMIT_MAX: usize = 512;

/// Absolute cell coords of the light-emitting cells a [`paint_cells`] walked
/// over.
///
/// WHY THE BLIT PUBLISHES THIS. The light grid is downscaled 4x, so it
/// point-samples one cell in sixteen. That is fine for a lava lake — any 4x4
/// block of it contains lava — and completely wrong for a TORCH, which is one or
/// two cells: a single-cell emitter is missed 15 times out of 16, so a placed
/// torch would light nothing, flicker on only when the camera scrolled onto a
/// lucky alignment, and be maddening to debug. Measured before this existed: a
/// cave with a torch, a lantern and a campfire in it reported `hotCount = 0`.
///
/// Finding them properly means visiting every visible cell — and there is
/// exactly one pass that already does that, this one. So the census is taken
/// here as a by-product. The per-cell cost is a single boolean test on a flag
/// that is only refreshed when the material RUN changes, which is the same trick
/// the shade-table hoist uses; on terrain, which is long runs of non-emitters,
/// it is a predictable never-taken branch.
///
/// Deduplicated to one entry per 4-cell column group per row, so a wide lava
/// surface cannot flood the list and starve a torch on the far side of the view.
///
/// The original published this as two module-level `Int32Array`s plus a count,
/// read through an `emitters()` accessor. Here it is RETURNED: a mutable global
/// would need `unsafe` or interior mutability, and neither is worth it for a
/// value whose contract was already "read it, do not retain it". The two `Vec`s
/// are one allocation each per call, reserved at `EMIT_MAX` so they never grow
/// mid-loop.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmitterCensus {
    x: Vec<i32>,
    y: Vec<i32>,
}

impl EmitterCensus {
    /// Absolute cell X of each emitter found.
    pub fn x(&self) -> &[i32] {
        &self.x
    }
    /// Absolute cell Y of each emitter found.
    pub fn y(&self) -> &[i32] {
        &self.y
    }
    /// How many were found (`<= EMIT_MAX`).
    pub fn len(&self) -> usize {
        self.x.len()
    }
    /// Whether the view contained no emitters at all.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// 1 where the material declares `lightEmit`. Resolved once.
static IS_EMIT: LazyLock<[bool; MAT_COUNT]> = LazyLock::new(|| {
    let mut e = [false; MAT_COUNT];
    for (id, slot) in e.iter_mut().enumerate().skip(1) {
        *slot = MAT_LIGHT[id] > 0;
    }
    e
});

// --- The blit ----------------------------------------------------------------

/// Positive modulo — JS `%` keeps the sign, and world cells go negative. Rust's
/// `%` keeps the sign too, so this is needed for the same reason.
fn pmod(v: i32, m: i32) -> i32 {
    let r = v % m;
    if r < 0 { r + m } else { r }
}

/// Fill `out` (a 32-bit RGBA buffer, `w` x `h` cells) with the material colours
/// for the cell rect whose top-left is ABSOLUTE cell (`ox`, `oy`), and return
/// the emitter census taken along the way.
///
/// Split out of the class so it can be measured headlessly — there is no canvas
/// in a node benchmark, and no window in a parity test, but this is where all
/// the per-pixel cost lives.
///
/// The window bounds are resolved per ROW rather than per pixel: a row either
/// misses the loaded window entirely (one fill and move on) or intersects it in
/// a contiguous span, and outside that span the pixels are transparent. The
/// inner loop over the span then needs no bounds test at all.
///
/// Per cell the loop does: one material load (each cell loaded exactly once, via
/// the `next_id` lookahead that also serves as the right-hand neighbour), two
/// pattern-tile loads, one `depth_above` load/store, and one shade-table load
/// ending in a single 32-bit store. The tile wrap is a compare-and-subtract on a
/// running counter, never a modulo — 61 and 67 are prime, so `%` in here would
/// be an integer division per pixel.
///
/// # Panics
///
/// If `out` is shorter than `w * h`.
pub fn paint_cells(
    out: &mut [u32],
    w: i32,
    h: i32,
    grid: &CellGrid,
    ox: i32,
    oy: i32,
    shades: &CellShades,
) -> EmitterCensus {
    assert!(w >= 0 && h >= 0, "viewport extent must be non-negative");
    let wu = w as usize;
    assert!(
        out.len() >= wu * h as usize,
        "output buffer is smaller than the {w}x{h} viewport"
    );

    let cols = grid.cols();
    let rows = grid.rows();
    let mat = &grid.material;
    let origin_x = grid.origin_cell_x();
    let origin_y = grid.origin_cell_y();

    let tex_a = &**TEX_A;
    let tex_b = &**TEX_B;
    let off_a_tab = &*TEX_OFF_A;
    let off_b_tab = &*TEX_OFF_B;
    let is_emit = &*IS_EMIT;
    let shade32 = &shades.shade32;

    // The original grew a module-level scratch array on demand and never
    // reallocated after the first frame. One `vec!` per call is the same cost
    // profile once the allocator has warmed, and it keeps the function pure.
    let mut d_above = vec![0u8; wu];

    // Local-x of the buffer's left edge, and the span of buffer columns that
    // lands inside the window. Constant for every row, so it is computed once.
    let lx0 = ox - origin_x;
    let mut span_start = -lx0; // buffer column where local_x reaches 0
    if span_start < 0 {
        span_start = 0;
    }
    let mut span_end = cols - lx0; // buffer column where local_x reaches cols
    if span_end > w {
        span_end = w;
    }

    // Seed the vertical run-lengths from the row ABOVE the buffer, so the top
    // row of the screen is shaded by what is really above it rather than
    // sprouting a spurious rim along the viewport edge every frame.
    {
        let seed_y = oy - 1 - origin_y;
        if seed_y >= 0 && seed_y < rows {
            let row_in = seed_y as isize * cols as isize + lx0 as isize;
            for lx in span_start..span_end {
                let id = mat[(row_in + lx as isize) as usize];
                d_above[lx as usize] = if id == 0 { 0 } else { 3 };
            }
        } else if span_start < span_end {
            d_above[span_start as usize..span_end as usize].fill(3);
        }
    }

    let mut emit = EmitterCensus {
        x: Vec::with_capacity(EMIT_MAX),
        y: Vec::with_capacity(EMIT_MAX),
    };

    // Tile row/column cursors. `pmod` once per row for the row bases; the column
    // cursors then walk with a compare-and-subtract.
    let col_a0 = pmod(ox + span_start, PA);
    let col_b0 = pmod(ox + span_start, PB);

    for ly in 0..h {
        let row_out = (ly as usize) * wu;
        let local_y = oy + ly - origin_y;
        if local_y < 0 || local_y >= rows || span_start >= span_end {
            out[row_out..row_out + wu].fill(0);
            // A row outside the window is air as far as the shading is
            // concerned, so the run-lengths must reset or the first loaded row
            // below would inherit a stale "buried" class and lose its top rim.
            if span_start < span_end {
                d_above[span_start as usize..span_end as usize].fill(0);
            }
            continue;
        }

        // Transparent margins either side of the loaded span.
        if span_start > 0 {
            out[row_out..row_out + span_start as usize].fill(0);
        }
        if span_end < w {
            out[row_out + span_end as usize..row_out + wu].fill(0);
        }

        let row_in = local_y as isize * cols as isize + lx0 as isize;
        // Pattern is keyed on the ABSOLUTE world cell so the texture is nailed
        // to the world and does not shimmer as the camera moves.
        let row_a = pmod(oy + ly, PA) * PA;
        let row_b = pmod(oy + ly, PB) * PB;
        let mut ca = col_a0;
        let mut cb = col_b0;

        // `prev_id` is the left neighbour, `next_id` the right — carried in
        // registers so the horizontal neighbourhood costs no loads. Out-of-span
        // is treated as SOLID (id 1) rather than air: a rim drawn down the edge
        // of the streaming window would be an artefact of where the window
        // happens to end.
        let mut prev_id = 1u16;
        let mut cur_id = mat[(row_in + span_start as isize) as usize];
        let mut next_id = if span_start + 1 < span_end {
            mat[(row_in + span_start as isize + 1) as usize]
        } else {
            1
        };

        // RUN HOISTING. The three per-material table reads — both tile base
        // offsets and the shade-table base — are pure functions of the material
        // id, and a scanline through terrain is made of LONG runs of one
        // material: a stone mass, a dirt bed, a lava pool. Caching them behind
        // an id compare turns three loads and two adds into one compare for
        // every cell after the first of each run, which measured as the
        // difference between this pass costing 2.6x the old flat blit and
        // costing 1.7x.
        let mut run_id: i32 = -1;
        let mut off_a: i32 = 0;
        let mut off_b: i32 = 0;
        let mut shade_base: usize = 0;
        let mut run_emit = false;
        let mut last_emit_col: i32 = -1; // 4-cell column group of the last entry

        for lx in span_start..span_end {
            let id = cur_id;

            if id == 0 {
                out[row_out + lx as usize] = 0;
                d_above[lx as usize] = 0; // air: everything below restarts its run
            } else {
                if i32::from(id) != run_id {
                    run_id = i32::from(id);
                    off_a = off_a_tab[id as usize] + row_a;
                    off_b = off_b_tab[id as usize] + row_b;
                    shade_base = (id as usize) << SHADE_SHIFT;
                    run_emit = is_emit[id as usize];
                }
                if run_emit && emit.x.len() < EMIT_MAX {
                    let col = lx >> 2;
                    if col != last_emit_col {
                        last_emit_col = col;
                        emit.x.push(ox + lx);
                        emit.y.push(oy + ly);
                    }
                }
                // Vertical class: saturating run length below the last air cell.
                let da = d_above[lx as usize];
                let side_open = if prev_id == 0 || next_id == 0 { 4u8 } else { 0 };
                let p = usize::from(tex_a[(off_a + ca) as usize])
                    + usize::from(tex_b[(off_b + cb) as usize]);
                out[row_out + lx as usize] =
                    shade32[shade_base | (usize::from(da | side_open) << PAT_BITS) | p];
                if da < 3 {
                    d_above[lx as usize] = da + 1;
                }
            }

            // Advance the coprime cursors and the material window.
            ca += 1;
            if ca == PA {
                ca = 0;
            }
            cb += 1;
            if cb == PB {
                cb = 0;
            }
            prev_id = id;
            cur_id = next_id;
            next_id = if lx + 2 < span_end {
                mat[(row_in + lx as isize + 2) as usize]
            } else {
                1
            };
        }
    }

    emit
}

// --- What the shader needs ---------------------------------------------------

// EVERYTHING BELOW EXISTS SO `cells.wgsl` NEED NOT RESTATE A SINGLE NUMBER
// FROM THIS FILE.
//
// `cells.wgsl` is where the shading lives — plain WGSL with no engine imports,
// which is what lets `tests/shader_matches_cpu.rs` compile it standalone.
// `cellmap.wgsl` is the Bevy-facing wrapper that `#import`s it and is not what
// these constants are about; naming that one here was a slip.
//
// The GPU pass in [`crate::cellmap`] is a translation of [`paint_cells`], and a
// translation is only trustworthy if both halves read the same constants. The
// tables (`TEX_A`, `TEX_B`, `CellShades::table`) upload verbatim; the scalars a
// shader cannot import are re-exported here and asserted equal to the WGSL's own
// literals by `tests/shader_matches_cpu.rs`, which parses `cells.wgsl` for them.
// A drift in either direction fails a test rather than quietly re-colouring the
// world.

/// Period of [`TEX_A`], in cells — 61.
///
/// Coprime with [`TEX_B_PERIOD`], which is the reason the composite pattern
/// repeats only at lcm(61, 67) = 4087 cells. The two tiles must stay SEPARATE
/// textures sampled at their own periods for that to hold; padding either to a
/// power of two would reinstate exactly the visible tiling this file's header
/// describes removing.
pub const TEX_A_PERIOD: i32 = PA;

/// Period of [`TEX_B`], in cells — 67. See [`TEX_A_PERIOD`].
pub const TEX_B_PERIOD: i32 = PB;

/// Patterns in each tile set — the slab count in [`TEX_A`] and [`TEX_B`].
pub const TEX_PATTERN_COUNT: usize = TEX_COUNT;

/// Edge classes per material in the shade table.
pub const SHADE_EDGE_CLASSES: usize = EDGE_CODES;

/// Pattern levels per edge class in the shade table — 64.
pub const SHADE_PATTERN_LEVELS: usize = PAT_LEVELS;

/// Entries per material in [`CellShades::table`] — 512.
pub const SHADE_MATERIAL_STRIDE: usize = SHADE_STRIDE;

/// `31.0` — the neutral, no-offset pattern sample.
pub const SHADE_PAT_MID: f64 = PAT_MID;

/// Standard deviation of a summed pattern sample, in its 0..62 range.
pub const SHADE_PAT_SIGMA: f64 = PAT_SIGMA;

/// Rim/AO scale: `EDGE_GAIN[class] * MAT_EDGE/255 * this` is the brightness
/// offset a class contributes. See `CellShades::build_material_shades`.
pub const SHADE_EDGE_SCALE: f64 = 52.0;

/// Brightness offset per edge class, as a fraction of the material's `MAT_EDGE`.
pub const SHADE_EDGE_GAIN: [f64; EDGE_CODES] = EDGE_GAIN;

/// Radians-to-table-index scale for the shimmer wave — `SIN_SIZE / TAU`.
pub const SHIMMER_SIN_SCALE: f64 = SIN_SCALE;

/// Entries in the shimmer sine table.
pub const SHIMMER_SIN_SIZE: usize = SIN_SIZE;

/// Which of the eight tile patterns a material's `MAT_TEXTURE` resolves to.
///
/// The blit folds this into `TEX_OFF_A`/`TEX_OFF_B` as a byte offset; a
/// shader wants the plain index, because its tiles are one texture per set with
/// the eight slabs stacked in y.
pub fn material_pattern(id: usize) -> usize {
    tex_index(MAT_TEXTURE[id])
}

/// Everything [`CellShades::update_shimmer`] needs about one material, in the
/// exact terms it uses them.
///
/// THIS STRUCT IS THE ENTIRE PER-FRAME CPU COST OF THE ANIMATION ON THE GPU.
/// `update_shimmer` rewrites ~2 500 packed table entries every frame; the shader
/// reads these four numbers out of a uniform that is written ONCE and evaluates
/// the wave per fragment instead. See the note at the end of `update_shimmer`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShimmerParams {
    /// Peak lift in 0..255 colour units. Zero for a material that does not
    /// animate — including one whose declared shimmer is below `SHIMMER_MIN`.
    pub amp: f64,
    /// Phase offset, so lava and crystal do not breathe in lockstep.
    pub phase: f64,
    /// `MAT_COLORVAR * TEX_GAIN` — the pattern's colour swing.
    pub tex_amp: f64,
    /// `MAT_EDGE / 255` — the rim/AO strength.
    pub edge: f64,
}

/// [`ShimmerParams`] for one material.
pub fn shimmer_params(id: usize) -> ShimmerParams {
    let animated = MAT_SHIMMER[id] >= SHIMMER_MIN;
    ShimmerParams {
        amp: if animated {
            f64::from(MAT_SHIMMER[id]) * (1.0 / 255.0) * 46.0
        } else {
            0.0
        },
        phase: SHIMMER_PHASE[id],
        tex_amp: f64::from(MAT_COLORVAR[id]) * TEX_GAIN,
        edge: f64::from(MAT_EDGE[id]) * (1.0 / 255.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tex_and_pat_agree_on_pattern_order() {
        assert_tex_order();
    }

    /// The re-exported scalars are the ones the blit actually uses, not copies
    /// that could drift from them.
    #[test]
    fn the_shader_surface_re_exports_the_blits_own_numbers() {
        assert_eq!(TEX_A_PERIOD, PA);
        assert_eq!(TEX_B_PERIOD, PB);
        assert_eq!(SHADE_MATERIAL_STRIDE, 1 << SHADE_SHIFT);
        assert_eq!(SHADE_EDGE_CLASSES * SHADE_PATTERN_LEVELS, SHADE_STRIDE);
        assert_eq!(SHADE_EDGE_GAIN, EDGE_GAIN);
        // The tile sets are exactly `TEX_PATTERN_COUNT` slabs of `p * p`.
        assert_eq!(TEX_A.len(), TEX_PATTERN_COUNT * (PA * PA) as usize);
        assert_eq!(TEX_B.len(), TEX_PATTERN_COUNT * (PB * PB) as usize);
    }

    /// `shimmer_params` must agree with `update_shimmer` about which materials
    /// animate — a mismatch would leave the shader animating a material the CPU
    /// oracle holds still, or the reverse.
    #[test]
    fn shimmer_params_animate_exactly_the_materials_update_shimmer_rewrites() {
        for id in 1..MAT_COUNT {
            let animated = shimmer_params(id).amp > 0.0;
            assert_eq!(
                animated,
                SHIMMER_IDS.contains(&id),
                "material {id}: shimmer_params says animated={animated}"
            );
        }
    }

    #[test]
    fn js_round_matches_ecmascript_at_the_awkward_points() {
        // Half rounds toward +Infinity, not away from zero.
        assert_eq!(js_round(0.5), 1.0);
        assert_eq!(js_round(-0.5), 0.0);
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        // The value that `(v + 0.5).floor()` gets wrong, because the add itself
        // rounds up to exactly 0.5.
        assert_eq!(js_round(0.499_999_999_999_999_94), 0.0);
    }

    #[test]
    fn to_int32_truncates_toward_zero_and_wraps() {
        assert_eq!(to_int32(3.9), 3);
        assert_eq!(to_int32(-3.9), -3);
        assert_eq!(to_int32(4294967296.0), 0);
        assert_eq!(to_int32(2147483648.0), i32::MIN);
        assert_eq!(to_int32(f64::NAN), 0);
    }

    #[test]
    fn every_tile_sample_lands_in_range() {
        for tile in [&**TEX_A, &**TEX_B] {
            assert!(tile.iter().all(|&v| v < TEX_LEVELS as u8));
        }
    }

    #[test]
    fn air_stays_transparent_and_solids_do_not() {
        let s = CellShades::new();
        assert_eq!(&s.table()[0..SHADE_STRIDE], &[0u32; SHADE_STRIDE][..]);
        assert!(s.table()[SHADE_STRIDE..].iter().any(|&v| v != 0));
    }

    #[test]
    fn shimmer_only_rewrites_the_materials_that_declare_it() {
        let rest = CellShades::new();
        let mut lit = rest.clone();
        lit.update_shimmer(1.25);
        for id in 1..MAT_COUNT {
            let a = &rest.table()[id * SHADE_STRIDE..(id + 1) * SHADE_STRIDE];
            let b = &lit.table()[id * SHADE_STRIDE..(id + 1) * SHADE_STRIDE];
            if SHIMMER_IDS.contains(&id) {
                continue;
            }
            assert_eq!(a, b, "material {id} moved without declaring shimmer");
        }
    }
}

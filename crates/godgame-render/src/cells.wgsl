#define_import_path godgame::cells

// The cell rasteriser, on the GPU.
//
// This is a translation of `cells.rs::paint_cells`, which is itself a 1:1 port
// of the TypeScript `ChunkCanvas.ts` and is verified against it byte for byte by
// `tests/ts_cells_parity.rs`. THE CPU PASS IS THE ORACLE, not a legacy path:
// `tests/shader_matches_cpu.rs` renders this module headlessly over a real
// worldgen window and diffs the framebuffer against `paint_cells` over the same
// grid. Read `cells.rs` for why any of the arithmetic below is what it is; this
// file only says how it maps onto a fragment.
//
// WHAT MOVED, AND WHY IT COULD.
//
//   - `depth_above` was a per-column running counter walked top-to-bottom, which
//     is a serial dependency down the column and the one thing that looks
//     unportable. It is not, because the counter SATURATES AT 3: a cell can only
//     be influenced by the three cells above it, so the scan is three
//     fixed-offset taps. `cells.rs::EDGE_CODES` calls that out as the reason the
//     shader port is possible at all.
//   - `side_open` was carried in `prev_id`/`next_id` registers. Here it is two
//     more taps, left and right.
//   - The two pattern tiles were flat arrays walked with a compare-and-subtract
//     cursor. Here they are two textures, still at their own COPRIME periods —
//     see `cell_pattern`.
//   - `SHADE32` was a 106 KB table the CPU indexed. Here it is a texture,
//     uploaded once. It is static for a given content build.
//   - `update_shimmer` REWROTE ~2 500 packed table entries EVERY FRAME on the
//     CPU. Here it is `clock.x` — one uniform — and `cell_shimmer_color` below.
//     That is the single biggest CPU win in the whole renderer: the TypeScript
//     paid a full palette rebuild per frame purely because there was no cached
//     repaint to invalidate, and on the GPU the same animation costs two sines
//     per fragment of lava and nothing at all anywhere else.
//
// Every scalar below is re-exported from `cells.rs` and asserted equal to these
// literals by `tests/shader_matches_cpu.rs`, which parses this file for them.
// Do not edit a number here without editing it there.

/// `cells::TEX_A_PERIOD`.
const PA: i32 = 61;
/// `cells::TEX_B_PERIOD`.
const PB: i32 = 67;
/// `cells::SHADE_PAT_MID` — the neutral, no-offset pattern sample.
const PAT_MID: f32 = 31.0;
/// `cells::SHADE_PAT_SIGMA` — sigma of a summed sample in its 0..62 range.
const PAT_SIGMA: f32 = 12.020815;
/// `cells::SHADE_EDGE_SCALE`.
const EDGE_SCALE: f32 = 52.0;
/// `cells::SHIMMER_SIN_SCALE` — radians to sine-table index.
const SIN_SCALE: f32 = 325.9493;
/// `cells::SHIMMER_SIN_SIZE`.
const SIN_SIZE: f32 = 2048.0;
/// `SHIMMER_SIN_SIZE - 1`, as the mask the phase wraps with.
const SIN_MASK: i32 = 2047;
const TAU: f32 = 6.2831855;

/// Everything the pass needs that is not a table — `cellmap::CellShadeParams`.
///
/// Declared here rather than beside the bindings so the game shader and the
/// parity harness cannot drift apart on its layout; only the `@group`/`@binding`
/// lines, which differ by construction, are stated twice.
struct CellShadeParams {
    /// (r, g, b, pattern index) per material, colour in 0..255.
    base: array<vec4<f32>, 64>,
    /// (shimmer amp, phase, tex amp, edge) per material. `amp == 0` means the
    /// material does not animate, which is what `cell_color` branches on.
    shim: array<vec4<f32>, 64>,
    /// (seconds, unused, unused, unused). This vec4 IS `update_shimmer`.
    clock: vec4<f32>,
    /// Absolute cell coordinate of texel (0, 0).
    origin: vec2<i32>,
};

/// `cells::SHADE_EDGE_GAIN`. A `var<private>` rather than a `const` because it
/// is indexed by the edge CLASS, which is a runtime value.
var<private> EDGE_GAIN: array<f32, 8> = array<f32, 8>(
    0.62,  // 0: directly under air — the top face, full rim
    0.26,  // 1: one cell down — the falloff that reads as thickness
    0.07,  // 2: two cells down — nearly neutral
    -0.24, // 3: buried on all counted sides — ambient occlusion
    0.74,  // 4: top face AND a side open — an exposed corner, brightest
    0.36,  // 5: one down, side open
    0.19,  // 6: two down, side open — a vertical wall face catching side light
    0.08,  // 7: buried but side-open — a cliff face, lifted off the interior
);

/// Positive modulo. `cells::pmod` — world cells go negative and both languages'
/// `%` keeps the sign.
fn cell_pmod(v: i32, m: i32) -> i32 {
    let r = v % m;
    return select(r, r + m, r < 0);
}

/// The material code at a WINDOW-LOCAL cell.
///
/// Outside the window reads as SOLID (id 1), never as air, on BOTH axes. That is
/// `paint_cells`' rule: `prev_id`/`next_id` are seeded to 1 outside the span, and
/// a seed row outside the grid seeds `d_above` to 3 — which is exactly what three
/// solid taps produce in `cell_edge_class`. A rim drawn down the edge of the
/// streaming window would be an artefact of where the window happens to end.
fn cell_id_at(ids: texture_2d<u32>, dims: vec2<i32>, p: vec2<i32>) -> u32 {
    if p.x < 0 || p.y < 0 || p.x >= dims.x || p.y >= dims.y {
        return 1u;
    }
    return textureLoad(ids, p, 0).r;
}

/// The 3-bit lighting class, `depth_above | (side_open << 2)`.
///
/// `depth_above` is the number of cells between this one and the nearest air
/// ABOVE it in the same column, CAPPED AT 3 — and the cap is what makes this a
/// shader at all. On the CPU it is a per-column counter that resets on air and
/// saturates; saturating at 3 means no cell can be influenced by anything more
/// than three rows up, so the serial scan collapses to the three taps below with
/// no dependency between fragments.
fn cell_edge_class(ids: texture_2d<u32>, dims: vec2<i32>, cell: vec2<i32>) -> u32 {
    var depth_above = 3u;
    if cell_id_at(ids, dims, cell + vec2<i32>(0, -1)) == 0u {
        depth_above = 0u;
    } else if cell_id_at(ids, dims, cell + vec2<i32>(0, -2)) == 0u {
        depth_above = 1u;
    } else if cell_id_at(ids, dims, cell + vec2<i32>(0, -3)) == 0u {
        depth_above = 2u;
    }
    let left = cell_id_at(ids, dims, cell + vec2<i32>(-1, 0));
    let right = cell_id_at(ids, dims, cell + vec2<i32>(1, 0));
    return depth_above | select(0u, 4u, left == 0u || right == 0u);
}

/// The summed pattern sample, 0..62.
///
/// THE TWO TILES KEEP THEIR OWN COPRIME PERIODS AND STAY SEPARATE TEXTURES.
/// 61 and 67 are why the composite repeats only at lcm(61, 67) = 4087 cells —
/// 20 435 screen px, wider than any viewport — so a flat sand field never lines
/// up with itself. Padding either to a power of two would reinstate exactly the
/// visible tiling `cells.rs` describes removing. Each texture is the eight
/// pattern slabs stacked in y, so the slab base is `pattern * p`.
///
/// Keyed on the ABSOLUTE world cell, so the texture is nailed to the world and
/// does not crawl as the camera moves.
fn cell_pattern(
    tex_a: texture_2d<u32>,
    tex_b: texture_2d<u32>,
    pattern: i32,
    world: vec2<i32>,
) -> u32 {
    let a = textureLoad(
        tex_a,
        vec2<i32>(cell_pmod(world.x, PA), pattern * PA + cell_pmod(world.y, PA)),
        0,
    ).r;
    let b = textureLoad(
        tex_b,
        vec2<i32>(cell_pmod(world.x, PB), pattern * PB + cell_pmod(world.y, PB)),
        0,
    ).r;
    return a + b;
}

/// `cells::clamp255` followed by `cells::pack`'s truncation, in 0..1 units.
///
/// The channels are TRUNCATED toward zero, not rounded — that is what a JS `|` on
/// a float operand does, and the CPU port spells it `as u32` on a clamped
/// non-negative value. `floor` is the same thing once the clamp has removed the
/// negatives.
fn cell_pack_channel(v: f32) -> f32 {
    return floor(clamp(v, 0.0, 255.0)) / 255.0;
}

/// The shimmer wave's sine, read the way `cells::update_shimmer` reads it.
///
/// The 2048-entry table is NOT uploaded: `sin` of the same index differs from the
/// tabulated value by well under an f32 ulp, and it is the INDEX TRUNCATION — the
/// `ToInt32` then mask — that the two implementations can actually disagree
/// about, so that part is reproduced exactly. `i32()` truncates toward zero in
/// WGSL exactly as ToInt32 does, and the mask of a negative index takes the same
/// low bits in both languages.
fn cell_sin_table(a: f32) -> f32 {
    let i = i32(a) & SIN_MASK;
    return sin(f32(i) / SIN_SIZE * TAU);
}

/// One animated material's colour, in place of a shade-table read.
///
/// This function IS `CellShades::update_shimmer`, evaluated for one (edge,
/// pattern) pair instead of for all 512 of them on the CPU. `shim` carries the
/// four numbers `cells::shimmer_params` resolves per material; `base` carries the
/// authored colour in 0..255.
///
/// The one deliberate departure: the CPU walks the 64 pattern levels with
/// `a += d`, and this multiplies. The series is the same; a single rounding
/// replaces `pattern` of them, which is strictly the more accurate of the two and
/// is far inside the f32/f64 gap that `tests/shader_matches_cpu.rs` measures.
///
/// MEASURED: byte-identical to `update_shimmer` for the first half hour of clock,
/// and one least-significant bit on a few per cent of lava beyond it — the point
/// where an f32 can no longer hold consecutive sine-table indices. The test has
/// the table.
fn cell_shimmer_color(base: vec4<f32>, shim: vec4<f32>, edge: u32, pattern: u32, t: f32) -> vec4<f32> {
    let amp = shim.x;
    let phase = shim.y;
    let tex_amp = shim.z;
    let edge_frac = shim.w;
    let p = f32(pattern);

    // Two rates beating against each other, with the phase advancing along the
    // pattern index so the bright band TRAVELS ACROSS THE CRUST rather than the
    // whole pool flashing together.
    let a0 = (t * 2.1 + phase) * SIN_SCALE + p * (0.29 * SIN_SCALE);
    let a1 = (t * 0.77 + phase * 1.7) * SIN_SCALE + p * (0.11 * SIN_SCALE);
    let w = cell_sin_table(a0) * 0.62 + cell_sin_table(a1) * 0.38;
    let lift = amp * (w * 0.5 + 0.5);

    let pd = (p - PAT_MID) / PAT_SIGMA;
    let ed = EDGE_GAIN[edge] * edge_frac * EDGE_SCALE;
    let s = pd * tex_amp + ed + lift;

    // The lift is biased toward red/orange as it rises: hot things shift hue,
    // they do not just gain luminance.
    return vec4<f32>(
        cell_pack_channel(base.x + s + lift * 0.55),
        cell_pack_channel(base.y + s + lift * 0.12),
        cell_pack_channel(base.z + s - lift * 0.3),
        1.0,
    );
}

/// The finished cell colour, in DISPLAY units — the same bytes `paint_cells`
/// packs, divided by 255.
///
/// It is display-space and not linear on purpose: `MAT_R`/`MAT_G`/`MAT_B` are
/// authored sRGB, and every clamp in `cells.rs` happens on those authored bytes.
/// Converting to linear before the clamp would move where the highlights
/// saturate. The caller converts on the way out if its target needs it.
///
/// `base` is `params.base[id]` — `(r0, g0, b0, pattern index)`; `shim` is
/// `params.shim[id]`, whose `x` is zero for every material that does not animate
/// and is therefore the branch between the two paths. `clock.x` is the animation
/// clock in seconds.
fn cell_color(
    ids: texture_2d<u32>,
    tex_a: texture_2d<u32>,
    tex_b: texture_2d<u32>,
    shade: texture_2d<f32>,
    cell: vec2<i32>,
    origin: vec2<i32>,
    id: u32,
    base: vec4<f32>,
    shim: vec4<f32>,
    clock: vec4<f32>,
) -> vec4<f32> {
    // Air is left fully transparent so the sky shows through with no special
    // case anywhere else — `paint_cells` stores a zero word for the same reason.
    if id == 0u {
        return vec4<f32>(0.0);
    }

    let dims = vec2<i32>(textureDimensions(ids));
    let edge = cell_edge_class(ids, dims, cell);
    let pattern = cell_pattern(tex_a, tex_b, i32(base.w), origin + cell);

    // A material that declares no shimmer reads the table `CellShades::new`
    // built; one that does has its slice recomputed here instead of on the CPU.
    // `update_shimmer` rewrites exactly this set of materials and no others.
    if shim.x > 0.0 {
        return cell_shimmer_color(base, shim, edge, pattern, clock.x);
    }
    // The static path: one indexed read, exactly as on the CPU. The shade
    // texture is `MAT_COUNT` rows of `8 edge classes x 64 pattern levels`.
    return textureLoad(shade, vec2<i32>(i32((edge << 6u) | pattern), i32(id)), 0);
}

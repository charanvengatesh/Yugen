// The light field's separable blur, on the GPU.
//
// The exact same kernel `light.rs`'s `blur_one` runs on the CPU, one axis per
// pass, over all four channels at once: R/G/B are the coloured light and A is
// the scalar light the darkness multiply is built from.
//
// # Why this is a nested pair of boxes and not one triangle
//
// A triangle blur IS a box convolved with a box, so the obvious shader is nine
// taps of `[1,2,3,4,5,4,3,2,1] / 25` in one loop. That is exactly right in the
// INTERIOR and wrong at the edges, and this grid is only ~162x92 with the view
// reaching to within a texel of its border, so the edges are on screen.
//
// The reason is that both implementations clamp out-of-range reads to the edge
// texel, and the CPU clamps them BETWEEN the two boxes: it replicates the first
// box's OUTPUT, not the original field. Replicating a value that is already a
// local average is not the same as averaging replicated values, so a fused
// triangle diverges across the outermost `back + fwd` texels of every edge.
//
// So the loops are nested, the outer one clamps its centre before the inner one
// runs, and the result matches `blur_one` everywhere rather than in the middle.
// It costs `(back + fwd + 1)^2` taps — 25 at the shipping reach — over a grid of
// ~15 000 texels, which is nothing on a GPU and was the entire point of moving.
//
// # The window is a uniform, not a constant
//
// `BLUR_BACK` and `BLUR_FWD` are derived from `BLUR_REACH_CELLS` and
// `LIGHT_DOWNSCALE` in `light.rs`. Restating them here as literals would put a
// second copy of that derivation in a file no `const` assert can reach — the
// trap `BLOOM_WGSL` documents and pays for. They arrive in `params` instead, so
// there is one source of truth and re-tuning the reach needs no edit here.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var src: texture_2d<f32>;

// `xy` is the window as `(back, fwd)`; `zw` is the axis step in texels, `(1, 0)`
// for the row pass and `(0, 1)` for the column pass. One material, two
// instances, and the only difference between the two passes is this vector.
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params: vec4<f32>;

/// One texel, with out-of-range reads clamped to the edge — `box_rows`' `at`.
fn tap(c: vec2<i32>, hi: vec2<i32>) -> vec4<f32> {
    return textureLoad(src, clamp(c, vec2(0), hi), 0);
}

/// The first box: the mean over `[c - back, c + fwd]` along `step`.
///
/// No clamp to 0..1 on the way out, and that is not an omission. The CPU pass
/// clamps here because its running sum adds and subtracts its way along a row
/// and drifts by a few ulps; this sums its taps fresh every texel, so there is
/// nothing to drift. The value is a mean of inputs already in 0..1 and so is in
/// 0..1 by construction — the clamp would be a no-op that costs two ops a tap.
fn inner(c: vec2<i32>, step: vec2<i32>, hi: vec2<i32>, back: i32, fwd: i32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var d = -back; d <= fwd; d = d + 1) {
        sum = sum + tap(c + step * d, hi);
    }
    return sum / f32(back + fwd + 1);
}

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(src));
    let hi = dims - vec2(1);
    let cell = clamp(vec2<i32>(floor(mesh.uv * vec2<f32>(dims))), vec2(0), hi);

    let back = i32(params.x);
    let fwd = i32(params.y);
    let step = vec2<i32>(i32(params.z), i32(params.w));

    // The second box leans back by exactly as much as the first leans forward,
    // which is what keeps the pair CENTRED when the window has an even width.
    // Getting this backwards shifts the whole light field half a cell against
    // the art — see `blur_one`, which passes the same two numbers the same way
    // round.
    var sum = vec4<f32>(0.0);
    for (var d = -fwd; d <= back; d = d + 1) {
        sum = sum + inner(clamp(cell + step * d, vec2(0), hi), step, hi, back, fwd);
    }
    return clamp(sum / f32(back + fwd + 1), vec4(0.0), vec4(1.0));
}

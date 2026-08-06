// Placeholder cell shading: one flat colour per material.
//
// This is the SEAM, not the destination. Everything the finished pass needs is
// already bound here — the per-cell material id, and a per-material lookup —
// so swapping in the real shading (`crate::cells`: material texture, edge
// light, ambient occlusion, shimmer) is a change to `fragment` and to what the
// lookup holds, not to the pipeline, the mesh, or the upload path.
//
// The 64 below is `cellmap::PALETTE_SLOTS`, asserted equal on the Rust side.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput

#ifdef SRGB_OUTPUT
#import bevy_render::color_operations::linear_to_srgb
#endif

// One linear-RGBA colour per material code. Slot 0 is air and is transparent,
// so the camera's clear colour is the sky.
struct CellPalette {
    colors: array<vec4<f32>, 64>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var cell_ids: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> palette: CellPalette;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // The quad spans the whole streaming window, so uv maps straight onto the
    // cell-id texture. `textureLoad`, not `textureSample`: a Uint texture has
    // no filtering, and one texel IS one cell — there is nothing to interpolate.
    let dims = vec2<f32>(textureDimensions(cell_ids));
    let cell = vec2<i32>(clamp(floor(mesh.uv * dims), vec2(0.0), dims - vec2(1.0)));
    let id = textureLoad(cell_ids, cell, 0).r;

    var color = palette.colors[min(id, 63u)];

#ifdef SRGB_OUTPUT
    color = vec4(linear_to_srgb(color.rgb), color.a);
#endif

    return color;
}

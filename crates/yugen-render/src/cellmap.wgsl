// The cell pass, as Bevy sees it: bindings, the quad's uv, and one call.
//
// All of the shading lives in `cells.wgsl`, which is plain WGSL with no engine
// imports in it — that is what lets `tests/shader_matches_cpu.rs` compile the
// same source against raw wgpu and diff the result against `cells.rs`. This file
// is the Bevy half and nothing else: if a line of arithmetic ever appears below,
// it is a line the parity harness is not checking.
//
// The 64 below is `cellmap::MATERIAL_SLOTS`, asserted equal on the Rust side.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import yugen::cells::{cell_color, CellShadeParams, GRAIN}
#import bevy_render::color_operations::srgb_to_linear

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var cell_ids: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var tex_a: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var tex_b: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var shade: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<uniform> params: CellShadeParams;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // The quad spans the whole streaming window, so uv maps straight onto the
    // cell-id texture. `textureLoad`, not `textureSample`: a Uint texture has
    // no filtering, and one texel IS one cell — there is nothing to interpolate.
    //
    // The FINE coordinate is quantised once and the cell DERIVED from it —
    // never a second `floor(uv * dims)`, because two independent float
    // truncations can disagree by an ulp exactly on a cell boundary and draw a
    // one-texel seam there. The pattern gets the fine grid; the id, the edge
    // class and the shade row stay per cell.
    let dims = vec2<f32>(textureDimensions(cell_ids));
    let fine = vec2<i32>(clamp(
        floor(mesh.uv * dims * f32(GRAIN)),
        vec2(0.0),
        dims * f32(GRAIN) - vec2(1.0),
    ));
    let cell = fine / GRAIN;
    let sub = fine - cell * GRAIN;
    let id = min(textureLoad(cell_ids, cell, 0).r, 63u);

    let color = cell_color(
        cell_ids,
        tex_a,
        tex_b,
        shade,
        cell,
        sub,
        params.origin,
        id,
        params.base[id],
        params.shim[id],
        params.clock,
    );

#ifdef SRGB_OUTPUT
    // The pipeline composites in sRGB and expects the shader to have encoded.
    // `cell_color` returns the authored sRGB bytes over 255 — already encoded —
    // so this is the branch with nothing to do.
    return color;
#else
    // The target is an `_Srgb` surface and the hardware encodes on write. Hand
    // it the linear value that encodes back to exactly the byte `paint_cells`
    // packs, or every colour in the game comes out washed out.
    return vec4<f32>(srgb_to_linear(color.rgb), color.a);
#endif
}

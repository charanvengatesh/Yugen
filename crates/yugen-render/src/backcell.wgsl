// The background wall pass, as Bevy sees it.
//
// The sibling of `cellmap.wgsl`, and deliberately as thin: all of the shading is
// `cells.wgsl`'s `cell_color`, imported unchanged. Not one literal of it is
// restated here, and not one line of it is edited — which is what keeps
// `tests/ts_cells_parity.rs` and `tests/shader_matches_cpu.rs` describing the
// shipping front pass rather than a fork of it.
//
// So the whole of this file's own arithmetic is: read the id, discard where the
// front covers us, and multiply by a tint.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import yugen::cells::{cell_color, CellShadeParams}
#import bevy_render::color_operations::srgb_to_linear

// Binding 0 is the BACK plane, and that placement is load-bearing rather than
// arbitrary. `cell_color` taps this texture's neighbours through
// `cell_edge_class` to decide rim light and ambient occlusion, so bound to the
// wall plane it computes the edges of the WALL — which is what a wall should
// have. It also means this pass is, by construction, the same function of a
// different plane, times a scalar.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var back_ids: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var tex_a: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var tex_b: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var shade: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<uniform> params: CellShadeParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var front_ids: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var<uniform> tint: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(back_ids));
    let cell = vec2<i32>(clamp(floor(mesh.uv * dims), vec2(0.0), dims - vec2(1.0)));

    // Where the front plane is not air it is FULLY opaque — `cell_color` returns
    // alpha 1 for every id but 0, and the game's transparency lives in the
    // material colour rather than in alpha. So the front quad already covers this
    // fragment completely and shading the wall under it is work thrown away.
    //
    // Bevy 2D has no depth prepass, so without this the back quad shades every
    // visible cell a second time — roughly doubling the cell fragment cost for
    // pixels nobody can see. With it the wall only shades where the front is air,
    // which is the only place it is visible.
    if textureLoad(front_ids, cell, 0).r != 0u {
        discard;
    }

    let id = min(textureLoad(back_ids, cell, 0).r, 63u);
    let color = cell_color(
        back_ids,
        tex_a,
        tex_b,
        shade,
        cell,
        params.origin,
        id,
        params.base[id],
        params.shim[id],
        params.clock,
    );

    // Air in the back plane is open sky and stays transparent, exactly as it does
    // in the front pass — `cell_color` has already returned a zero vec4, and the
    // tint keeps it zero.
    let walled = vec4<f32>(color.rgb * tint.rgb, color.a * tint.a);

#ifdef SRGB_OUTPUT
    return walled;
#else
    return vec4<f32>(srgb_to_linear(walled.rgb), walled.a);
#endif
}

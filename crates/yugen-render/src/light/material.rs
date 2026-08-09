//! The composite materials, and the quad each one is drawn on.
//!
//! [`super`]'s "The composite materials" and "The bloom's gather pass". The
//! solved light reaches the screen as a small stack of full-screen quads —
//! shadow, colour, bloom, vignette, wash — and this is where each one's shader
//! and blend state live.
//!
//! The z ordering between them is load-bearing and is stated where the
//! constants are: two quads at one depth sort arbitrarily, which is how a pass
//! becomes invisible in exactly the scene it exists for.

use bevy::asset::uuid_handle;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, RenderPipelineDescriptor,
    SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey};

use super::passes::*;

// --- The composite materials -------------------------------------------------

/// The one shader every light quad uses: sample, tint, hand it to the blend.
///
/// Held as a string rather than a `.wgsl` beside the module because it carries
/// almost no logic — all of the arithmetic is baked into the textures by
/// [`LightGrid::bake_shadow`], [`LightGrid::bake_colour`] and [`bake_vignette`],
/// which is what makes those decisions testable on the CPU. `cells.wgsl` earns
/// its own file by being a real shading model; a passthrough and a uv nudge does
/// not.
///
/// # The uv nudge IS the softness dial
///
/// `softness` is the one thing this shader decides, and it decides it by moving
/// the SAMPLE POINT rather than by changing the filter. Every one of these
/// textures is sampled linearly. Pulling the sample toward its own texel's
/// centre by `1 - softness` narrows the band over which the filter is allowed to
/// interpolate: at 0 every sample lands exactly on a texel centre and the result
/// is bit-for-bit a nearest fetch; at 1 nothing moves and it is a plain bilinear
/// upscale; in between, the light steps from cell to cell over a ramp
/// `softness` of a cell wide and is flat across the rest.
///
/// That is a continuous dial between "the light is drawn in the same blocks as
/// the art" and "the light is a smooth field over it", and it exists because
/// both extremes are wrong in the same picture: a fully snapped light is
/// indistinguishable from a differently-coloured BLOCK, and a fully smooth one
/// is the airbrushed wash this module spent a change removing. See
/// [`LIGHT_SOFTNESS`].
pub(super) const LIGHT_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var light_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var light_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> tint: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> softness: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(light_texture));
    let texel = mesh.uv * dim;
    let centre = floor(texel) + vec2<f32>(0.5);
    let uv = mix(centre, texel, softness.x) / dim;
    return textureSample(light_texture, light_sampler, uv) * tint;
}
"#;

/// Handle for [`LIGHT_WGSL`], inserted by [`LightPlugin`].
pub(super) const LIGHT_SHADER: Handle<Shader> =
    uuid_handle!("6b1a0f2c-9d34-4a71-8e55-2c0f7b3d9a10");

/// `dst * src`, with the destination's alpha left alone.
///
/// The exact translation of Canvas2D's `multiply` composite once the
/// `1 - a + a * shadow` term is baked into the source — see
/// [`LightGrid::bake_shadow`].
pub(super) const BLEND_MULTIPLY: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::Dst,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::Zero,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
};

/// `dst + src`. Canvas2D's `lighter`, with the destination's alpha left alone.
pub(super) const BLEND_ADD: BlendState = BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent {
        src_factor: BlendFactor::Zero,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
};

/// Overwrite the colour target's blend state with this pass's own.
///
/// `alpha_mode` puts the quad in the transparent phase with depth writes off,
/// which is what gets the ordering right; the blend state it picks there is then
/// replaced, because [`AlphaMode2d`] has three variants and none of them is
/// either of the two this module needs.
pub(super) fn set_blend(descriptor: &mut RenderPipelineDescriptor, blend: BlendState) {
    if let Some(fragment) = descriptor.fragment.as_mut()
        && let Some(Some(target)) = fragment.targets.first_mut()
    {
        target.blend = Some(blend);
    }
}

/// Define a light material: one texture, one tint, one blend state.
///
/// A macro, and not one type carrying the blend in its bind-group data, because
/// `specialize` is a STATIC method — the blend has to be a property of the type
/// for the pipeline cache to key on it, and two named types say that more
/// plainly than a key struct would.
macro_rules! light_material {
    ($(#[$meta:meta])* $name:ident, $blend:expr) => {
        $(#[$meta])*
        #[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
        pub struct $name {
            /// The grid or ramp this quad samples.
            ///
            /// Every one of them carries a LINEAR sampler — the game's one
            /// `default_nearest` override, and no longer for the reason it was
            /// first taken. How hard-edged the result is belongs to `softness`
            /// below, which needs a filter underneath it that can interpolate at
            /// all; see [`new_light_texture`].
            #[texture(0)]
            #[sampler(1)]
            pub texture: Handle<Image>,
            /// Multiplies the sampled texel. Canvas2D's `globalAlpha`, with the
            /// per-channel freedom the flat washes need.
            #[uniform(2)]
            pub tint: Vec4,
            /// How far this quad's sampling is allowed to cross a texel edge,
            /// in `x`; see [`LIGHT_WGSL`] and [`LIGHT_SOFTNESS`]. `yzw` are
            /// unused — a WGSL uniform pads a scalar to 16 bytes regardless, so
            /// a `Vec4` costs what an `f32` would have and says the padding is
            /// deliberate.
            #[uniform(3)]
            pub softness: Vec4,
        }

        impl Material2d for $name {
            fn fragment_shader() -> ShaderRef {
                LIGHT_SHADER.into()
            }

            fn alpha_mode(&self) -> AlphaMode2d {
                AlphaMode2d::Blend
            }

            fn specialize(
                descriptor: &mut RenderPipelineDescriptor,
                _layout: &MeshVertexBufferLayoutRef,
                _key: Material2dKey<Self>,
            ) -> Result<(), SpecializedMeshPipelineError> {
                set_blend(descriptor, $blend);
                Ok(())
            }
        }
    };
}

light_material!(
    /// The darkness pass, and the vignette that follows it.
    LightShadowMaterial,
    BLEND_MULTIPLY
);

light_material!(
    /// The coloured light, the bloom composite, and the flat washes.
    LightGlowMaterial,
    BLEND_ADD
);

// --- The bloom's gather pass -------------------------------------------------

/// `src`, ignoring whatever was there. The gather owns its target outright.
///
/// Not [`AlphaMode2d::Opaque`], which would say the same thing by putting the
/// quad in the opaque phase: this module's other four materials are all
/// transparent-phase with an overridden blend, and one pass that reaches the
/// same place by a different mechanism is a thing a reader has to check.
pub(super) const BLEND_REPLACE: BlendState = BlendState {
    color: BlendComponent::REPLACE,
    alpha: BlendComponent::REPLACE,
};

/// Threshold, blur, done — the entire bloom, in one gather over the cell plane.
///
/// Held inline for the same reason [`LIGHT_WGSL`] is: every decision worth
/// arguing about is in [`bake_bloom_params`], on the CPU, where a test can read
/// it. What is left is a bounds-checked double loop and a multiply-add, and a
/// `.wgsl` file beside the module would only put that four inches further from
/// the constants that shape it.
///
/// The loop bounds are `BLOOM_RADIUS_CELLS` and the tap array is that plus one;
/// a `const` assert beside [`BloomParams`] fails the build if either drifts.
///
/// `textureLoad` and no sampler at all. One texel is one cell — there is nothing
/// between two cells to interpolate, which is exactly the reasoning
/// [`crate::cellmap`]'s own id binding is written on. The SMOOTHING all happens
/// in the gather; the sampler that matters is the linear one on the way back
/// out, on the composite quad.
pub(super) const BLOOM_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct BloomParams {
    emit: array<vec4<f32>, 64>,
    taps: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var cell_ids: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> params: BloomParams;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let dim = vec2<i32>(textureDimensions(cell_ids));
    let centre = vec2<i32>(floor(mesh.uv * vec2<f32>(dim)));
    let w = array<f32, 4>(params.taps.x, params.taps.y, params.taps.z, params.taps.w);

    var sum = vec3<f32>(0.0);
    for (var dy = -3; dy <= 3; dy = dy + 1) {
        let y = centre.y + dy;
        if (y < 0 || y >= dim.y) {
            continue;
        }
        let wy = w[abs(dy)];
        for (var dx = -3; dx <= 3; dx = dx + 1) {
            let x = centre.x + dx;
            if (x < 0 || x >= dim.x) {
                continue;
            }
            let id = min(textureLoad(cell_ids, vec2<i32>(x, y), 0).r, 63u);
            sum = sum + params.emit[id].rgb * (wy * w[abs(dx)]);
        }
    }
    return vec4<f32>(sum, 1.0);
}
"#;

/// Handle for [`BLOOM_WGSL`], inserted by [`LightPlugin`].
pub(super) const BLOOM_SHADER: Handle<Shader> =
    uuid_handle!("2f4c8d16-5b73-4e90-a1c2-7d6e8f0b3a45");

/// The bloom's gather pass: `cellmap`'s cell ids in, blurred emissive light out.
///
/// This is the only material in the module that is not a passthrough, and the
/// only one that reads a texture it does not own. That coupling to
/// [`crate::cellmap::CellMap`] is the deliberate part: sharing the id plane is
/// what makes this a bloom OF THE SCENE rather than a second, independently
/// drifting opinion about where the lights are.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct BloomMaterial {
    /// [`crate::cellmap::CellMap::ids`] — one `R16Uint` texel per cell of the
    /// streaming window, not a copy of it. `u_int`, so `textureLoad` only.
    #[texture(0, sample_type = "u_int")]
    pub ids: Handle<Image>,
    /// The baked threshold table and gather weights.
    #[uniform(1)]
    pub params: BloomParams,
}

impl Material2d for BloomMaterial {
    fn fragment_shader() -> ShaderRef {
        BLOOM_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        set_blend(descriptor, BLEND_REPLACE);
        Ok(())
    }
}

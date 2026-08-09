//! The additive material a luminous creature is drawn through.
//!
//! [`super`]'s "The glow pass" is the argument; this is the shader behind it.
//! Nine species are self-luminous, and the darkness multiply that
//! [`crate::light`] composites over the world would erase every one of them in
//! exactly the cave where they are the brightest thing on screen. So they draw
//! twice: once into the world, and once more additively, above the composite.
//!
//! A material rather than a tint on the sprite, because a Bevy [`Sprite`]
//! alpha-blends and has no per-sprite blend state.
//! [`crate::sky::AdditiveMaterial`] is the same idea with no texture, and it is
//! what the glowing SHOTS use — a shot's glow is a flat rectangle and wants no
//! sampler at all.

use bevy::asset::uuid_handle;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, RenderPipelineDescriptor,
    SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey};

// ---------------------------------------------------------------------------
// The additive pass
// ---------------------------------------------------------------------------

/// The shader the creature glow quads sample the mob atlas with.
///
/// Held as a string rather than a `.wgsl` beside the module, on the same terms
/// [`crate::light`]'s is: it carries no shading model, only the tile lookup a
/// [`Sprite`] would have done from its `TextureAtlas`. `cells.wgsl` earns its own
/// file by being a real model; six lines of sampling does not.
///
/// The tile rect is a uniform because the quad is a unit square scaled by its
/// transform, which is what lets all 32 slots share one mesh — the geometry
/// never changes, only where in the strip it reads from.
pub(super) const MOB_GLOW_WGSL: &str = r#"
#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var glow_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var glow_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> tint: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> tile: vec4<f32>;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let uv = tile.xy + mesh.uv * tile.zw;
    let texel = textureSample(glow_texture, glow_sampler, uv);
    // Weighted by the texel's OWN alpha as well as the tint, so a transparent
    // cell adds nothing: the blend below ignores source alpha entirely, and a
    // sprite that is mostly holes would otherwise glow as a solid rectangle.
    return vec4<f32>(texel.rgb * texel.a * tint.rgb * tint.a, 1.0);
}
"#;

/// Handle for [`MOB_GLOW_WGSL`], inserted by [`MobsPlugin`].
pub(super) const MOB_GLOW_SHADER: Handle<Shader> =
    uuid_handle!("2f8c41d6-7b03-4e59-9a12-5d6e0c47af83");

/// `dst + src`, with the destination's alpha left alone.
///
/// Canvas2D's `lighter`. The source is already weighted by the pulse and by the
/// texel's own coverage in the shader above, so the blend takes the colour whole
/// rather than reading `SrcAlpha` — which is what
/// [`AdditiveMaterial`](crate::sky::AdditiveMaterial) does instead, because a
/// vertex-coloured quad has nowhere else to put the weight.
const BLEND_ADD: BlendState = BlendState {
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

/// One creature's second, additive pass: the same tile, added over the darkness.
///
/// A material and not a [`Sprite`] because a sprite alpha-blends and there is no
/// per-sprite blend state to change. It is the textured twin of
/// [`crate::sky::AdditiveMaterial`], which the glowing shots use — they are flat
/// rectangles and want no sampler.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct MobGlowMaterial {
    /// The creature's baked strip. Sampled NEAREST, from the atlas's own sampler:
    /// adjacent tiles share an edge, so anything wider than a texel would bleed
    /// one pose into the next.
    #[texture(0)]
    #[sampler(1)]
    pub atlas: Handle<Image>,
    /// Multiplies the sampled texel. `w` carries the pulse — see
    /// [`GLOW_PULSE_BASE`].
    #[uniform(2)]
    pub tint: Vec4,
    /// The tile's sub-rectangle in the strip, `(u0, v0, du, dv)`. From
    /// [`BakedSprite::tile_uv`], which is where the strip's layout is known.
    #[uniform(3)]
    pub tile: Vec4,
}

impl Material2d for MobGlowMaterial {
    fn fragment_shader() -> ShaderRef {
        MOB_GLOW_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        // Blend, so the quad is queued into the transparent phase and sorted by z
        // against everything else in the overlay. The blend STATE that mode picks
        // is then replaced below; what is borrowed here is the sorting.
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
            target.blend = Some(BLEND_ADD);
        }
        Ok(())
    }
}

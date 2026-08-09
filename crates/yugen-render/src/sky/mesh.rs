//! The one dynamic mesh the backdrop passes share, and the additive material
//! drawn through it.
//!
//! Stars, discs and ridges are all "some triangles in view space, in a colour",
//! so they are one [`VertexBuf`] rebuilt each frame rather than three meshes
//! with three materials. [`AdditiveMaterial`] is the untextured half of what
//! [`crate::mobs`]'s glow material does — a Bevy [`Sprite`] alpha-blends and has
//! no per-sprite blend state, and a star wants no sampler at all.
//!
//! Split from [`super::model`] on the line between deciding what to draw and
//! having somewhere to put it: nothing here knows what a horizon is.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, RenderPipelineDescriptor,
    SpecializedMeshPipelineError,
};
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey};

use yugen_core::config::View;

use super::model::Rgb;

// ---------------------------------------------------------------------------
// Drawing primitives
// ---------------------------------------------------------------------------

/// A 0..255 colour and an alpha, as the linear RGBA a vertex attribute wants.
///
/// Vertex colours reach the shader untouched and the render target does the
/// linear-to-sRGB conversion on write, so a colour authored in sRGB has to be
/// converted here or the whole backdrop comes out washed out.
pub(crate) fn linear(rgb: Rgb, alpha: f32) -> [f32; 4] {
    Color::srgb(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        .with_alpha(alpha)
        .to_linear()
        .to_f32_array()
}

/// View space — origin at the view's top-left, +y DOWN — into the backdrop's local
/// space, whose origin is the view CENTRE and whose +y is UP.
///
/// THIS IS THE ONE PLACE THE BACKDROP FLIPS. Every model function above works in
/// the TypeScript's screen space and every vertex below goes through here, so
/// there is exactly one sign to get wrong and it is on this line.
pub(crate) fn view_to_local(x: f32, y: f32, view: View) -> Vec2 {
    Vec2::new(x - view.w as f32 * 0.5, view.h as f32 * 0.5 - y)
}

/// Triangles under construction, reused across frames.
///
/// A `Local<VertexBuf>` per drawing system, so a per-frame rebuild allocates only
/// while the geometry is still growing — the same discipline the TypeScript's
/// preallocated typed arrays kept, in the one place this port still needs it.
#[derive(Default)]
pub(crate) struct VertexBuf {
    position: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    color: Vec<[f32; 4]>,
    index: Vec<u32>,
}

impl VertexBuf {
    /// Drop last frame's triangles, keeping the allocation.
    pub(crate) fn clear(&mut self) {
        self.position.clear();
        self.uv.clear();
        self.color.clear();
        self.index.clear();
    }

    /// Push a vertex in LOCAL space and return its index.
    pub(crate) fn vertex(&mut self, p: Vec2, color: [f32; 4]) -> u32 {
        let i = self.position.len() as u32;
        self.position.push([p.x, p.y, 0.0]);
        // Zero, and never read. `ColorMaterial`'s shader reads `mesh.uv`
        // unconditionally and that field only exists when the mesh declares the
        // attribute, so this is here to make the pipeline build, not to sample
        // anything.
        self.uv.push([0.0, 0.0]);
        self.color.push(color);
        i
    }

    /// Push a triangle from three existing vertices.
    pub(crate) fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.index.extend_from_slice(&[a, b, c]);
    }

    /// Push a quad from four corners in local space, wound in order.
    pub(crate) fn quad(&mut self, corners: [Vec2; 4], colors: [[f32; 4]; 4]) {
        let a = self.vertex(corners[0], colors[0]);
        let b = self.vertex(corners[1], colors[1]);
        let c = self.vertex(corners[2], colors[2]);
        let d = self.vertex(corners[3], colors[3]);
        self.tri(a, b, c);
        self.tri(a, c, d);
    }

    /// Push a flat-coloured rectangle given in VIEW space.
    ///
    /// This is `fillRect`: `(x, y)` is the TOP-LEFT and `w`/`h` extend right and
    /// DOWN, exactly as the canvas took them.
    pub(crate) fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, view: View, color: [f32; 4]) {
        self.quad(
            [
                view_to_local(x, y, view),
                view_to_local(x + w, y, view),
                view_to_local(x + w, y + h, view),
                view_to_local(x, y + h, view),
            ],
            [color; 4],
        );
    }

    /// Replace a mesh's geometry with what has been built.
    ///
    /// Takes the buffers rather than cloning them: every system here rebuilds from
    /// empty, and this is what keeps a full backdrop rebuild allocation-free once
    /// the vectors have reached their steady size.
    pub(crate) fn write(&mut self, mesh: &mut Mesh) {
        // A mesh with no vertices is not a mesh that draws nothing — it is a mesh
        // the renderer cannot allocate a slab for, and `bevy_render`'s mesh
        // allocator then reports
        // "Use-after-free: attempted to copy element data for an unallocated key"
        // once per empty mesh per frame. It is noisy rather than fatal, but it is
        // a real invariant being violated and it buried every other log line.
        //
        // Every buffer here legitimately empties: the stars are cut off in
        // daylight, the discs are both below the horizon twice a cycle, and a
        // weather layer is empty in a biome that has no dust. So instead of
        // asking four call sites to remember, one degenerate triangle stands in —
        // three coincident points at the origin at zero alpha. It allocates, it
        // rasterises no fragments, and it costs one triangle.
        if self.position.is_empty() {
            self.position.extend_from_slice(&[[0.0; 3]; 3]);
            self.uv.extend_from_slice(&[[0.0; 2]; 3]);
            self.color.extend_from_slice(&[[0.0; 4]; 3]);
            self.index.extend_from_slice(&[0, 1, 2]);
        }

        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            core::mem::take(&mut self.position),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, core::mem::take(&mut self.uv));
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, core::mem::take(&mut self.color));
        mesh.insert_indices(Indices::U32(core::mem::take(&mut self.index)));
    }
}

/// A vertex-coloured triangle mesh, ready to be rewritten every frame.
///
/// It starts as the same degenerate triangle [`VertexBuf::write`] falls back to,
/// and for the same reason: these are spawned in `PostStartup` and the first
/// `place_*` does not run until the next frame, so an empty one here is an
/// unallocatable mesh for one frame at every launch.
pub(crate) fn dynamic_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0f32; 4]; 3])
    .with_inserted_indices(Indices::U32(vec![0, 1, 2]))
}

/// The material the canvas's `lighter` composite became.
///
/// Empty on purpose: it binds nothing, ships no shader, and rides the default
/// mesh2d pass, which returns the interpolated vertex colour and nothing else. The
/// whole material is [`Material2d::specialize`] — see the module header.
#[derive(Asset, TypePath, AsBindGroup, Clone, Copy, Debug, Default)]
pub struct AdditiveMaterial {}

impl Material2d for AdditiveMaterial {
    fn alpha_mode(&self) -> AlphaMode2d {
        // Blend, so the mesh is queued into the transparent phase and sorted by z
        // against the rest of the backdrop. The blend STATE that mode picks is
        // then replaced below; what is borrowed here is the sorting.
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
            target.blend = Some(BlendState {
                // `dst + src * srcAlpha`, which is canvas `lighter` exactly.
                color: BlendComponent {
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                // The destination is the opaque backdrop; leave its alpha alone.
                alpha: BlendComponent {
                    src_factor: BlendFactor::Zero,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mesh handed to the renderer must never have zero vertices.
    ///
    /// This is a regression test for a real bug, and the bug is worth restating
    /// because nothing about it was visible from the game: five meshes here and
    /// in `crate::weather` are rewritten every frame, all five legitimately empty
    /// (no stars in daylight, both discs down, a biome with no dust), and an
    /// empty mesh made `bevy_render`'s allocator log
    /// "Use-after-free: attempted to copy element data for an unallocated key"
    /// **1612 times in a twelve-second run**. The frame still drew. Only the log
    /// said anything was wrong.
    #[test]
    fn an_empty_buffer_still_writes_an_allocatable_mesh() {
        let mut mesh = dynamic_mesh();
        assert_eq!(
            mesh.count_vertices(),
            3,
            "a freshly spawned mesh is empty for the frame before the first \
             place_* runs, so it has to carry the stand-in too"
        );

        // The buffer never had a single triangle pushed into it — the daylight
        // starfield case.
        let mut buf = VertexBuf::default();
        buf.clear();
        buf.write(&mut mesh);

        assert_eq!(
            mesh.count_vertices(),
            3,
            "an empty buffer must still leave something allocatable behind"
        );
        assert!(
            mesh.indices().is_some_and(|i| i.len() == 3),
            "and the indices have to match, or the draw call is malformed"
        );

        // It must also be invisible: this stands in for nothing, so it may not
        // put a pixel on screen.
        let Some(bevy::render::mesh::VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("the stand-in lost its vertex colours");
        };
        assert!(
            colors.iter().all(|c| c[3] == 0.0),
            "the stand-in triangle must be fully transparent, got {colors:?}"
        );
    }

    /// And a buffer with real geometry is not disturbed by the fallback.
    #[test]
    fn a_buffer_with_triangles_writes_exactly_those_triangles() {
        let mut mesh = dynamic_mesh();
        let mut buf = VertexBuf::default();
        buf.clear();
        buf.rect(0.0, 0.0, 2.0, 2.0, View::for_screen(1280, 800), [1.0; 4]);
        let want = buf.position.len();
        buf.write(&mut mesh);
        assert!(want > 0, "the fixture should have produced geometry");
        assert_eq!(mesh.count_vertices(), want, "the fallback must not fire");
    }

    #[test]
    fn the_view_flip_is_the_only_thing_that_moves_a_point_between_the_two_spaces() {
        let view = View::for_screen(1440, 900);
        let w = view.w as f32;
        let h = view.h as f32;
        // The view's corners, in the TypeScript's screen space.
        assert_eq!(view_to_local(0.0, 0.0, view), Vec2::new(-w / 2.0, h / 2.0));
        assert_eq!(view_to_local(w, h, view), Vec2::new(w / 2.0, -h / 2.0));
        // And its centre, which is where the camera is.
        assert_eq!(view_to_local(w / 2.0, h / 2.0, view), Vec2::ZERO);
    }

    #[test]
    fn a_rebuilt_mesh_carries_every_attribute_the_pass_needs() {
        // `ColorMaterial`'s shader reads `mesh.uv` unconditionally and the default
        // mesh2d fragment returns magenta without vertex colours, so a mesh missing
        // either attribute fails at pipeline build time — a long way from here.
        let view = View::for_screen(1440, 900);
        let mut buf = VertexBuf::default();
        buf.rect(3.0, 4.0, 2.0, 2.0, view, [1.0, 1.0, 1.0, 1.0]);
        let mut mesh = dynamic_mesh();
        buf.write(&mut mesh);

        assert!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
        assert_eq!(mesh.count_vertices(), 4);
        assert_eq!(mesh.indices().map(Indices::len), Some(6));
    }

    #[test]
    fn a_vertex_buffer_is_emptied_by_a_write_and_reused_by_the_next_frame() {
        let view = View::for_screen(1440, 900);
        let mut buf = VertexBuf::default();
        let mut mesh = dynamic_mesh();
        buf.rect(0.0, 0.0, 1.0, 1.0, view, [1.0; 4]);
        buf.write(&mut mesh);
        assert_eq!(mesh.count_vertices(), 4);

        // A second frame that draws nothing must not leave LAST frame's geometry
        // behind. The systems clear before they build for exactly this reason,
        // and `write` taking the buffers is what makes the clear cheap.
        //
        // What it leaves is the stand-in triangle rather than literally nothing —
        // see `an_empty_buffer_still_writes_an_allocatable_mesh`. This test
        // originally asserted zero here, and zero is precisely what made the mesh
        // allocator log a use-after-free every frame. The invariant it is really
        // defending is "the square is gone", so that is what it checks.
        buf.clear();
        buf.write(&mut mesh);
        assert_eq!(
            mesh.count_vertices(),
            3,
            "the four-vertex square must be gone, replaced by the stand-in"
        );
        assert_eq!(mesh.indices().map(Indices::len), Some(3));
    }

    #[test]
    fn a_filled_rect_is_placed_from_its_top_left_with_y_running_down() {
        // `VertexBuf::rect` is the canvas `fillRect` this whole port is built on.
        // Getting its corner or its sign wrong would put the backdrop upside down
        // in a way no colour test would catch.
        let view = View::for_screen(1440, 900);
        let h = view.h as f32;
        let mut buf = VertexBuf::default();
        buf.rect(0.0, 0.0, 4.0, 2.0, view, [0.0; 4]);
        let top = buf.position[0][1];
        let bottom = buf.position[3][1];
        assert_eq!(top, h / 2.0, "the rect starts at the top of the view");
        assert!(bottom < top, "and extends downward");
        assert_eq!(top - bottom, 2.0, "by its height");
    }
}

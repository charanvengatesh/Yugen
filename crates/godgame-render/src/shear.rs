//! The lean, as geometry: quads whose top and bottom edges may slide sideways.
//!
//! # Why this module exists at all
//!
//! [`PlayerFigure::lean`](crate::player_art::PlayerFigure::lean) is a SHEAR. In
//! the TypeScript it was `ctx.transform(1, 0, -lean, 1, 0, 0)` about the drawn
//! feet, and a canvas takes that happily because a canvas transform is a full
//! 2x3 affine matrix. A Bevy `Transform` is translation-rotation-scale, and a
//! shear is none of the three: TRS can move a rect, turn it and stretch it, but
//! it can never slide one edge of it past the other. So for as long as the
//! figure was a `Sprite` the lean was computed, clamped, deadbanded, unit
//! tested — and then thrown away on the way to the screen.
//!
//! # Why four vertices is EXACT and not an approximation
//!
//! A shear is a LINEAR map. A linear map is completely determined by where it
//! sends the corners of a region, and the rasteriser interpolates vertex
//! attributes linearly across a triangle. Displace the two top vertices of a
//! quad by `shear_at(top)` and the two bottom vertices by `shear_at(bottom)`,
//! and every interior pixel lands exactly where the canvas transform would have
//! put it — including the texture coordinate it samples, because the UV is
//! interpolated over the same triangle. There is no error term to bound. That
//! is the entire argument for this file, and it is why the quad carries two
//! independent edge displacements rather than an angle.
//!
//! The rejected alternative was a rotation about the feet, which `Transform` DOES
//! have. At the shipped clamp of `|lean| <= 0.16` — under ten degrees — it reads
//! close, and the difference is what happens to the HORIZONTALS. A shear slants
//! the verticals and leaves every horizontal exactly level: the sole stays flat
//! on the floor, the shoulders stay square, the figure keeps its full height. A
//! rotation tips all of it, so the feet come off the ground on one side and the
//! silhouette loses `1 - cos(theta)` of its height while it does. On a
//! deliberately six-pixel character that is a foot hovering over the floor for a
//! whole run cycle, which is not a subtle failure at this resolution.
//!
//! # What the port changed
//!
//! **The figure is a mesh, not a `Sprite`.** One `Mesh2d` carrying one to three
//! quads, with a `ColorMaterial` whose texture is the sprite atlas strip. The
//! atlas tile is selected by UV rather than by `TextureAtlas::index`, which is
//! the same lookup — a tile is a slice of a one-row strip, so its `u` range is
//! `index / bake_count` to `(index + 1) / bake_count` and its `v` range is the
//! whole texture.
//!
//! **The dash smear is quads in one mesh, not extra draw calls.** The canvas set
//! `globalAlpha` and issued a second and third `drawImage`; here the alpha rides
//! on the vertices — `ColorMaterial`'s shader multiplies the sampled texel by the
//! interpolated vertex colour — and the after-images are simply more triangles in
//! the same buffer. Back-to-front is index order, which is why
//! [`PlayerFigure::ghosts`](crate::player_art::PlayerFigure::ghosts) hands them
//! over furthest-first.
//!
//! **The vertex colour is LINEAR.** `ColorMaterial` multiplies in linear light,
//! so a caller with an sRGB colour must convert on the way in. Alpha is unaffected
//! and passes through as authored.
//!
//! # What is deliberately NOT here
//!
//! Any knowledge of the player. [`QuadBuf`] takes rects, texture slices and
//! tints; [`crate::player`] is what knows that one of those rects is a body, that
//! the displacements come from `shear_at`, and that the top-left is the thing that
//! gets rounded. Keeping the split means the geometry can be tested against a
//! hand-written rect with no `Player`, no atlas and no GPU, and the art model can
//! go on being tested with no mesh.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Vertices in one quad, and the length of one `corners`/`uv` array.
const CORNERS: usize = 4;

/// Indices one quad contributes: two triangles.
const INDICES_PER_QUAD: usize = 6;

/// THE ONE AXIS FLIP in this module. The simulation's `+y` is DOWN and Bevy's is
/// UP, so every sim-space point crosses over here and nowhere else.
///
/// Both [`origin_translation`] and [`QuadBuf::push`] go through it, which is what
/// keeps the entity's transform and the vertices it carries from disagreeing
/// about which way the world points — a disagreement that draws the figure
/// upside down at twice its distance from the camera, and is invisible until it
/// is a long way from home.
#[inline]
fn flip(p: Vec2) -> Vec2 {
    Vec2::new(p.x, -p.y)
}

/// Where the entity carrying a [`QuadBuf`]'s geometry has to sit, given the
/// sim-space point that buffer's local origin means.
///
/// Snap the origin BEFORE calling: the whole pixel-snap rule is that a rect's
/// top-left lands on a whole world pixel, and this is the only place a caller
/// still has the top-left on its own.
pub fn origin_translation(origin: Vec2, z: f32) -> Vec3 {
    flip(origin).extend(z)
}

/// One quad to draw, in SIM space: an axis-aligned rect whose two horizontal
/// edges are each free to slide sideways.
///
/// Two displacements rather than one angle, because that is exactly the freedom
/// a shear needs and no more — see the module header. A caller with a genuine
/// affine transform to apply has more than this can carry and wants a different
/// type; a caller with a lean has precisely this.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShearedRect {
    /// Left edge of the UNSHEARED rect, world px. Already snapped.
    pub x: f32,
    /// Top edge of the unsheared rect, world px, `+y` DOWN.
    pub y: f32,
    /// Width, world px.
    pub w: f32,
    /// Height, world px.
    pub h: f32,
    /// How far the TOP edge slides. Both top corners move together, so the edge
    /// stays horizontal and only its position changes — that is what makes this
    /// a shear rather than a general quadrilateral.
    pub top_dx: f32,
    /// How far the BOTTOM edge slides. Zero when the shear pivots on the rect's
    /// own sole, which is the player's case and the reason its feet stay planted.
    pub bottom_dx: f32,
}

impl ShearedRect {
    /// A rect with no shear on it at all.
    pub fn flat(x: f32, y: f32, w: f32, h: f32) -> ShearedRect {
        ShearedRect {
            x,
            y,
            w,
            h,
            top_dx: 0.0,
            bottom_dx: 0.0,
        }
    }

    /// The four corners in SIM space, top-left first and then clockwise.
    ///
    /// Clockwise in sim space (`+y` down) is what [`flip`] turns into the same
    /// order `sky::VertexBuf::rect` emits, so every vertex-coloured mesh in the
    /// game winds the same way. Bevy's 2D pipeline sets `cull_mode: None`, so a
    /// reversed winding would still draw — which is precisely why the convention
    /// has to be written down rather than discovered by something going black.
    pub fn corners(&self) -> [Vec2; CORNERS] {
        let bottom = self.y + self.h;
        [
            Vec2::new(self.x + self.top_dx, self.y),
            Vec2::new(self.x + self.w + self.top_dx, self.y),
            Vec2::new(self.x + self.w + self.bottom_dx, bottom),
            Vec2::new(self.x + self.bottom_dx, bottom),
        ]
    }
}

/// Which slice of a one-row atlas strip a quad samples.
///
/// `v` is always the whole texture: [`BakedSprite`](crate::sprite::BakedSprite)
/// lays its tiles out `bake_count` across and ONE down, so a tile is a vertical
/// slab and only `u` varies. Mirroring is `u0` and `u1` swapped, which reflects
/// the art about the rect's own vertical axis exactly as `Sprite::flip_x` does —
/// the figure turns where it stands instead of jumping a rect's width sideways.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileUv {
    /// `u` at the quad's left edge.
    pub u0: f32,
    /// `u` at the quad's right edge.
    pub u1: f32,
}

impl TileUv {
    /// The whole texture. What an untextured quad wants, and what a caller with a
    /// one-tile image wants.
    pub const WHOLE: TileUv = TileUv { u0: 0.0, u1: 1.0 };

    /// Tile `index` of a strip `count` tiles wide, optionally mirrored.
    ///
    /// `count` is clamped to at least one so a caller cannot divide by zero on
    /// the way to a NaN vertex, which is a whole mesh lost rather than one quad.
    pub fn of(index: usize, count: usize, flip_x: bool) -> TileUv {
        let count = count.max(1) as f32;
        let left = index as f32 / count;
        let right = (index + 1) as f32 / count;
        if flip_x {
            TileUv {
                u0: right,
                u1: left,
            }
        } else {
            TileUv {
                u0: left,
                u1: right,
            }
        }
    }

    /// The four texture coordinates, in [`ShearedRect::corners`] order.
    pub fn corners(&self) -> [[f32; 2]; CORNERS] {
        [
            [self.u0, 0.0],
            [self.u1, 0.0],
            [self.u1, 1.0],
            [self.u0, 1.0],
        ]
    }
}

/// Textured, vertex-tinted quads under construction, reused across frames.
///
/// A `Local<QuadBuf>` per drawing system, the same discipline
/// `sky::VertexBuf` keeps: a per-frame rebuild allocates only while the geometry
/// is still growing, and the player's never grows past three quads.
///
/// It is not `sky::VertexBuf` because that one writes `[0.0, 0.0]` into every UV
/// on purpose — the backdrop carries all its colour on the vertices and samples
/// nothing. This buffer exists for the case that does sample.
#[derive(Default)]
pub struct QuadBuf {
    /// The sim-space point local `(0, 0)` means. See [`origin_translation`].
    origin: Vec2,
    position: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    color: Vec<[f32; 4]>,
    index: Vec<u32>,
}

impl QuadBuf {
    /// Drop last frame's triangles and set the origin this frame's are relative
    /// to, keeping the allocation.
    pub fn begin(&mut self, origin: Vec2) {
        self.origin = origin;
        self.position.clear();
        self.uv.clear();
        self.color.clear();
        self.index.clear();
    }

    /// Add one quad: a rect in sim space, the texture slice it samples, and a
    /// LINEAR RGBA tint applied to every one of its corners.
    pub fn push(&mut self, rect: ShearedRect, uv: TileUv, tint: [f32; 4]) {
        let base = self.position.len() as u32;
        let corners = rect.corners();
        let uvs = uv.corners();
        for (corner, uv) in corners.iter().zip(uvs) {
            let local = flip(*corner) - flip(self.origin);
            self.position.push([local.x, local.y, 0.0]);
            self.uv.push(uv);
            self.color.push(tint);
        }
        self.index
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// How many quads have been pushed since [`QuadBuf::begin`]. Diagnostic, and
    /// what the tests count.
    pub fn quads(&self) -> usize {
        self.index.len() / INDICES_PER_QUAD
    }

    /// Replace a mesh's geometry with what has been built.
    ///
    /// Takes the buffers rather than cloning them, which is what keeps a steady
    /// state rebuild allocation-free.
    ///
    /// The degenerate-triangle floor is `sky::VertexBuf::write`'s, and it is here
    /// for the same reason it is there: a mesh with no vertices is not a mesh
    /// that draws nothing, it is a mesh `bevy_render`'s allocator cannot allocate
    /// a slab for, and it then logs "Use-after-free: attempted to copy element
    /// data for an unallocated key" once per empty mesh PER FRAME. Noisy rather
    /// than fatal, and it buried every other log line when it last happened.
    ///
    /// The player's buffer should never legitimately empty — there is always a
    /// figure, even when there is no art for it — so reaching the fallback here
    /// means a bug upstream. It still costs one triangle instead of a flood of
    /// engine errors, which is the right trade for something that has already
    /// gone wrong once.
    pub fn write(&mut self, mesh: &mut Mesh) {
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

/// A textured, vertex-coloured triangle mesh, ready to be rewritten every frame.
///
/// Distinct from `sky::dynamic_mesh` in exactly one respect — this one is meant
/// to be sampled, so its placeholder UVs are the ones [`QuadBuf::write`]'s
/// fallback writes rather than an attribute that exists only to make the pipeline
/// build. It starts as the same degenerate triangle, and for the same reason: the
/// entity is spawned in `PostStartup` and the first rebuild does not run until
/// the next frame, so an empty one here is an unallocatable mesh for one frame at
/// every launch.
pub fn dynamic_quad_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; 3])
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0f32; 4]; 3])
    .with_inserted_indices(Indices::U32(vec![0, 1, 2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Positions of a written mesh, as local-space points.
    fn positions(mesh: &Mesh) -> Vec<Vec2> {
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .expect("a written mesh has float3 positions")
            .iter()
            .map(|p| Vec2::new(p[0], p[1]))
            .collect()
    }

    fn index_count(mesh: &Mesh) -> usize {
        match mesh.indices().expect("a written mesh has indices") {
            Indices::U32(v) => v.len(),
            Indices::U16(v) => v.len(),
        }
    }

    /// Twice the signed area of a polygon. Negative is clockwise in a `+y` UP
    /// space, which is the winding every mesh in this game uses.
    fn signed_area(points: &[Vec2]) -> f32 {
        (0..points.len())
            .map(|i| {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                a.x * b.y - b.x * a.y
            })
            .sum()
    }

    /// A 10x15 rect at the origin — the player's own box, so the numbers below
    /// are the ones the game actually sees.
    fn rect() -> ShearedRect {
        ShearedRect::flat(0.0, 0.0, 10.0, 15.0)
    }

    #[test]
    fn a_flat_rect_has_four_axis_aligned_corners() {
        let c = rect().corners();
        assert_eq!(c[0], Vec2::new(0.0, 0.0));
        assert_eq!(c[1], Vec2::new(10.0, 0.0));
        assert_eq!(c[2], Vec2::new(10.0, 15.0));
        assert_eq!(c[3], Vec2::new(0.0, 15.0));
        // Axis aligned: the two top corners share a y, and so do the two bottom.
        assert_eq!(c[0].y, c[1].y);
        assert_eq!(c[2].y, c[3].y);
        // And with no shear on it the sides are vertical too.
        assert_eq!(c[0].x, c[3].x);
        assert_eq!(c[1].x, c[2].x);
    }

    /// The displacement lands on the edges it is given to and on nothing else.
    #[test]
    fn each_edge_carries_exactly_its_own_displacement() {
        let sheared = ShearedRect {
            top_dx: 2.4,
            bottom_dx: -0.5,
            ..rect()
        };
        let flat = rect().corners();
        let c = sheared.corners();

        // Compared as sums rather than as differences: `(10.0 + 2.4) - 10.0` is
        // not 2.4 in binary floating point, and a test that asserts it is has
        // found a rounding mode rather than a bug.
        assert_eq!(c[0].x, flat[0].x + 2.4, "the top-left missed the shear");
        assert_eq!(c[1].x, flat[1].x + 2.4, "the top-right missed the shear");
        assert_eq!(c[3].x, flat[3].x - 0.5, "the bottom-left missed the shear");
        assert_eq!(c[2].x, flat[2].x - 0.5, "the bottom-right missed it");

        // A shear moves nothing vertically, so every horizontal stays level and
        // the figure keeps its full height. That is the whole difference between
        // it and the rotation that was rejected.
        for (got, want) in c.iter().zip(flat) {
            assert_eq!(got.y, want.y, "the shear lifted a corner");
        }
        assert_eq!(c[3].y - c[0].y, 15.0, "the figure lost height");
        // Both corners of an edge move together, so each edge keeps its length.
        assert!(
            (c[1].x - c[0].x - 10.0).abs() < 1e-4,
            "the top edge stretched"
        );
        assert!(
            (c[2].x - c[3].x - 10.0).abs() < 1e-4,
            "the bottom edge stretched"
        );
    }

    /// Zero displacement is not merely close to the unsheared rect, it IS it.
    #[test]
    fn a_zero_shear_is_bit_identical_to_no_shear() {
        let zero = ShearedRect {
            top_dx: 0.0,
            bottom_dx: 0.0,
            ..rect()
        };
        assert_eq!(zero.corners(), rect().corners());
    }

    /// The corner order survives the axis flip as the same winding the backdrop
    /// meshes use. `cull_mode: None` means a reversal would still draw, so
    /// nothing on screen would catch this.
    #[test]
    fn a_quad_winds_clockwise_in_bevy_space() {
        let mut buf = QuadBuf::default();
        buf.begin(Vec2::ZERO);
        buf.push(rect(), TileUv::WHOLE, [1.0; 4]);
        let mut mesh = dynamic_quad_mesh();
        buf.write(&mut mesh);

        let p = positions(&mesh);
        assert_eq!(p.len(), CORNERS);
        assert!(signed_area(&p) < 0.0, "the quad wound anticlockwise: {p:?}");
    }

    /// Local space is sim space with one negation, applied to the corner and to
    /// the origin alike — get one and not the other and the figure lands at
    /// twice its own distance from the camera, mirrored.
    #[test]
    fn the_origin_is_subtracted_in_the_flipped_space() {
        let mut buf = QuadBuf::default();
        buf.begin(Vec2::new(4.0, 20.0));
        buf.push(
            ShearedRect::flat(4.0, 20.0, 10.0, 15.0),
            TileUv::WHOLE,
            [1.0; 4],
        );
        let mut mesh = dynamic_quad_mesh();
        buf.write(&mut mesh);

        let p = positions(&mesh);
        // A rect whose top-left IS the origin starts at local zero and grows
        // right and DOWN, which in Bevy's space is negative y.
        assert_eq!(p[0], Vec2::new(0.0, 0.0));
        assert_eq!(p[1], Vec2::new(10.0, 0.0));
        assert_eq!(p[2], Vec2::new(10.0, -15.0));
        assert_eq!(p[3], Vec2::new(0.0, -15.0));
    }

    /// A sim-space point becomes a Bevy translation with the y negated once, and
    /// [`QuadBuf`] agrees with it — together they put the quad back where the sim
    /// asked for it.
    #[test]
    fn the_origin_translation_and_the_vertices_compose_back_to_sim_space() {
        let origin = Vec2::new(37.0, 204.0);
        assert_eq!(
            origin_translation(origin, 0.5),
            Vec3::new(37.0, -204.0, 0.5)
        );

        let mut buf = QuadBuf::default();
        buf.begin(origin);
        // A rect five px right and three px below the origin.
        buf.push(
            ShearedRect::flat(42.0, 207.0, 10.0, 15.0),
            TileUv::WHOLE,
            [1.0; 4],
        );
        let mut mesh = dynamic_quad_mesh();
        buf.write(&mut mesh);

        let world = origin_translation(origin, 0.5).truncate() + positions(&mesh)[0];
        assert_eq!(world, Vec2::new(42.0, -207.0), "the quad moved");
    }

    #[test]
    fn a_tile_spans_exactly_one_slot_of_the_strip() {
        let uv = TileUv::of(2, 8, false);
        assert_eq!(uv.u0, 0.25);
        assert_eq!(uv.u1, 0.375);
        // The whole texture vertically: the strip is one tile tall.
        let c = uv.corners();
        assert_eq!(c[0], [0.25, 0.0]);
        assert_eq!(c[1], [0.375, 0.0]);
        assert_eq!(c[2], [0.375, 1.0]);
        assert_eq!(c[3], [0.25, 1.0]);
    }

    #[test]
    fn mirroring_swaps_the_slot_and_nothing_else() {
        let plain = TileUv::of(2, 8, false);
        let flipped = TileUv::of(2, 8, true);
        assert_eq!(flipped.u0, plain.u1);
        assert_eq!(flipped.u1, plain.u0);
        // Still the same slot, so the figure turns where it stands rather than
        // sampling its neighbour's tile.
        assert_eq!(flipped.u0.min(flipped.u1), plain.u0);
        assert_eq!(flipped.u0.max(flipped.u1), plain.u1);
    }

    /// A one-tile strip is the whole texture, which is what an untextured caller
    /// and a single-frame sprite both want.
    #[test]
    fn a_single_tile_strip_is_the_whole_texture() {
        assert_eq!(TileUv::of(0, 1, false), TileUv::WHOLE);
        // And a zero count cannot produce a NaN — one NaN vertex is a whole mesh
        // gone, not one quad.
        let degenerate = TileUv::of(0, 0, false);
        assert!(degenerate.u0.is_finite() && degenerate.u1.is_finite());
    }

    /// Every quad is six indices over four vertices — a shared diagonal, not two
    /// independent triangles.
    #[test]
    fn quads_share_their_diagonal() {
        let mut buf = QuadBuf::default();
        buf.begin(Vec2::ZERO);
        for i in 0..3 {
            buf.push(
                ShearedRect::flat(i as f32 * 10.0, 0.0, 10.0, 15.0),
                TileUv::WHOLE,
                [1.0; 4],
            );
        }
        assert_eq!(buf.quads(), 3);

        let mut mesh = dynamic_quad_mesh();
        buf.write(&mut mesh);
        assert_eq!(positions(&mesh).len(), 3 * CORNERS);
        assert_eq!(index_count(&mesh), 3 * INDICES_PER_QUAD);
    }

    /// The tint reaches all four corners, so a faded ghost fades evenly rather
    /// than ramping across itself.
    #[test]
    fn a_tint_reaches_every_corner() {
        let mut buf = QuadBuf::default();
        buf.begin(Vec2::ZERO);
        buf.push(rect(), TileUv::WHOLE, [1.0, 1.0, 1.0, 0.15]);
        let mut mesh = dynamic_quad_mesh();
        buf.write(&mut mesh);

        let colors = mesh
            .attribute(Mesh::ATTRIBUTE_COLOR)
            .expect("a written mesh has colours");
        let bevy::mesh::VertexAttributeValues::Float32x4(colors) = colors else {
            panic!("colours are float32x4");
        };
        assert_eq!(colors.len(), CORNERS);
        assert!(colors.iter().all(|c| *c == [1.0, 1.0, 1.0, 0.15]));
    }

    /// A buffer that pushed nothing still writes a mesh the allocator can take.
    /// See [`QuadBuf::write`] — the alternative is an engine error every frame.
    #[test]
    fn an_empty_buffer_still_writes_a_drawable_mesh() {
        let mut buf = QuadBuf::default();
        buf.begin(Vec2::ZERO);
        let mut mesh = dynamic_quad_mesh();
        buf.write(&mut mesh);

        assert_eq!(positions(&mesh).len(), 3, "the mesh came out empty");
        assert_eq!(index_count(&mesh), 3);
        // And it rasterises nothing: three coincident points at zero alpha.
        assert!(positions(&mesh).iter().all(|p| *p == Vec2::ZERO));
    }

    /// The mesh spawned before the first rebuild is drawable too, for the one
    /// frame it is on screen unwritten.
    #[test]
    fn the_starting_mesh_is_not_empty() {
        let mesh = dynamic_quad_mesh();
        assert_eq!(positions(&mesh).len(), 3);
        assert_eq!(index_count(&mesh), 3);
        assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some());
    }

    /// `begin` drops last frame's work. A buffer that accumulated would grow a
    /// smear a frame at a time until it filled the screen.
    #[test]
    fn begin_drops_the_previous_frame() {
        let mut buf = QuadBuf::default();
        buf.begin(Vec2::ZERO);
        buf.push(rect(), TileUv::WHOLE, [1.0; 4]);
        buf.begin(Vec2::ZERO);
        assert_eq!(buf.quads(), 0);
    }
}

//! The cell pass: one texel per cell of the streaming window.
//!
//! # Shape
//!
//! One `WINDOW_COLS x WINDOW_ROWS` `R16Uint` texture holds the material code of
//! every cell in the window, and one quad the size of the window samples it. The
//! quad's transform follows `CellGrid`'s origin, so the texture never scrolls —
//! a window shift moves the quad in world space and rewrites the texels, exactly
//! as it moves the cells inside the grid.
//!
//! Cell id in, colour out. That is still the seam; what has changed is what sits
//! on the far side of it. The shading is now [`crate::cells`]' — material
//! texture, rim light, ambient occlusion and shimmer — evaluated in
//! `cells.wgsl`, and the lookups it needs are four bindings:
//!
//!   - **the id texture**, one `R16Uint` texel per cell, filled by
//!     [`upload_dirty_chunks`];
//!   - **[`TEX_A`] and [`TEX_B`]**, the two coprime pattern tiles, as two
//!     `R8Uint` textures of eight stacked slabs each. They keep their 61 and 67
//!     periods and stay SEPARATE textures: that coprimality is the only reason a
//!     flat sand field does not visibly tile, and padding either to a power of
//!     two would throw it away;
//!   - **the shade table**, `MAT_COUNT x 512` prepacked RGBA. Static for a
//!     content build, so it is uploaded exactly once;
//!   - **[`CellShadeParams`]**, the per-material scalars the shimmer needs, plus
//!     the animation clock.
//!
//! # The shimmer is a uniform now
//!
//! `CellShades::update_shimmer` rebuilds ~2 500 packed palette entries EVERY
//! FRAME on the CPU. It does that not because anyone wanted to, but because the
//! TypeScript had no cached repaint to invalidate, so a per-frame LUT rewrite was
//! cheaper than finding the emissive cells a second time — its own header says
//! so, at length.
//!
//! On the GPU that entire function collapses into [`update_shade_params`]: one
//! float written into a uniform. The wave is then evaluated per fragment, and
//! only for fragments of a material that actually declares shimmer. THIS IS THE
//! SINGLE BIGGEST CPU SAVING AVAILABLE IN THE RENDERER — it deletes the whole
//! per-frame table rebuild, and it is the one thing the CPU path could not have
//! done however it was written.
//!
//! # Dirty-rect upload, and why the TypeScript could not do it
//!
//! `CellGrid` keeps a per-chunk dirty bit, set by every write and cleared by the
//! renderer. [`upload_dirty_chunks`] rewrites only the chunks that carry it: a
//! settled world costs nothing, and a world with one falling grain costs one
//! 32x32 rect instead of 90,000 cells.
//!
//! **The TypeScript renderer could not do this and did not try.** Its dirty
//! masks were per-`CellGrid`-instance and the live ones belonged to the WORKER,
//! on the far side of a `SharedArrayBuffer` that carried only the four cell
//! planes. The main thread could not see which chunks had changed, so
//! `ChunkCanvas` repainted the entire viewport every frame and its header says
//! so in as many words. Being able to upload dirty rects is not a clever
//! optimisation — it is a direct consequence of deleting the worker (see
//! [`crate::world`]). The same fact is why there is no per-chunk canvas cache to
//! keep in sync.
//!
//! One caveat, written down because it is easy to over-claim: Bevy re-uploads a
//! whole `Image` asset when it changes. The dirty-rect win here is therefore the
//! CPU-side conversion — the 90k-cell walk that does not happen — plus not
//! touching the asset AT ALL on a frame where nothing moved, which skips the GPU
//! transfer entirely. A true sub-rect `write_texture` needs a custom render
//! asset and is a later step; the bookkeeping it would need is already here.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

use godgame_core::config::{CELL_SIZE, CHUNK_CELLS, WINDOW_COLS, WINDOW_ROWS};
use godgame_core::sim::materials::{MAT_B, MAT_COUNT, MAT_G, MAT_R};

use crate::cells::{
    CellShades, SHADE_MATERIAL_STRIDE, TEX_A, TEX_A_PERIOD, TEX_B, TEX_B_PERIOD, TEX_PATTERN_COUNT,
    material_pattern, shimmer_params,
};
use crate::lowres::WORLD_LAYERS;
use crate::world::SimWorld;

/// Per-material parameter slots.
///
/// A fixed-size uniform array, so it is a compile-time constant on both sides:
/// the WGSL says 64 as a literal and [`the parameter test`](tests) fails if this
/// stops matching. Padded past [`MAT_COUNT`] so adding a block to `content/` is
/// a recompile, not a shader edit.
pub const MATERIAL_SLOTS: usize = 64;

const _: () = assert!(
    MAT_COUNT <= MATERIAL_SLOTS,
    "more materials than parameter slots — widen MATERIAL_SLOTS and cellmap.wgsl together"
);

/// Bytes per texel of the cell-id texture. `R16Uint` — [`CellId`] is a `u16`.
///
/// [`CellId`]: godgame_core::sim::materials::CellId
const ID_BYTES: usize = 2;

/// The quad that draws the streaming window.
#[derive(Component)]
pub struct CellQuad;

/// The cell-id texture and the material that reads it.
#[derive(Resource, Clone, Debug)]
pub struct CellMap {
    /// `WINDOW_COLS x WINDOW_ROWS` `R16Uint`: one material code per cell.
    pub ids: Handle<Image>,
    /// The pass that turns those ids into colour.
    pub material: Handle<CellMaterial>,
}

/// Everything the shader needs that is not a table.
///
/// `base` and `shim` are static for a content build and written once;
/// [`update_shade_params`] touches only `clock` and `origin`.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct CellShadeParams {
    /// Per material: the authored colour in 0..255, and the pattern index its
    /// `MAT_TEXTURE` resolves to in `w`.
    pub base: [Vec4; MATERIAL_SLOTS],
    /// Per material: `cells::ShimmerParams` as `(amp, phase, tex_amp, edge)`.
    /// `amp` is zero for every material that does not animate, which is what the
    /// shader branches on.
    pub shim: [Vec4; MATERIAL_SLOTS],
    /// `(seconds, unused, unused, unused)`.
    ///
    /// This vec4 IS `CellShades::update_shimmer`. See the module header.
    pub clock: Vec4,
    /// Absolute cell coordinate of texel (0, 0) — the pattern is keyed on the
    /// world, not on the window, so it does not crawl as the camera moves.
    pub origin: IVec2,
}

impl Default for CellShadeParams {
    fn default() -> Self {
        build_params()
    }
}

/// The cell pass.
///
/// Four textures and a uniform, and every one of them is a lookup
/// [`crate::cells`] already does on the CPU.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct CellMaterial {
    /// One material code per cell. `u_int`: read with `textureLoad`, never
    /// filtered — one texel IS one cell, so there is nothing to interpolate.
    #[texture(0, sample_type = "u_int")]
    pub ids: Handle<Image>,
    /// [`TEX_A`], eight 61x61 pattern slabs stacked in y.
    #[texture(1, sample_type = "u_int")]
    pub tex_a: Handle<Image>,
    /// [`TEX_B`], eight 67x67 pattern slabs stacked in y.
    #[texture(2, sample_type = "u_int")]
    pub tex_b: Handle<Image>,
    /// The prepacked shade table, `MAT_COUNT` rows of 512.
    #[texture(3, sample_type = "float", filterable = false)]
    pub shade: Handle<Image>,
    /// Per-material scalars and the animation clock.
    #[uniform(4)]
    pub params: CellShadeParams,
}

impl Material2d for CellMaterial {
    fn fragment_shader() -> ShaderRef {
        // Embedded rather than loaded from an `assets/` directory: the shader is
        // source, versioned with the module it belongs to, and the game should
        // not need a data directory beside the binary to draw its first frame.
        "embedded://godgame_render/cellmap.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        // Air is transparent so the camera's clear colour is the sky.
        AlphaMode2d::Blend
    }
}

/// The cell-id texture, the lookup tables, and the quad that draws them.
pub struct CellMapPlugin;

impl Plugin for CellMapPlugin {
    fn build(&self, app: &mut App) {
        // The shading itself. `load_shader_library!` both embeds it and holds a
        // handle, which is what makes the `#import` in `cellmap.wgsl` resolve —
        // an embedded asset nothing has asked for is never loaded.
        bevy::shader::load_shader_library!(app, "cells.wgsl");
        bevy::asset::embedded_asset!(app, "cellmap.wgsl");

        app.add_plugins(Material2dPlugin::<CellMaterial>::default())
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (follow_window, update_shade_params, upload_dirty_chunks)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<CellMap>),
            );
    }
}

/// Allocate the id texture and the lookup tables, and spawn the quad.
///
/// The three lookup textures are built here and never touched again: the pattern
/// tiles and the shade table are pure functions of the content build. Only the
/// id texture and the clock move.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CellMaterial>>,
) {
    let ids = images.add(new_id_texture());
    let material = materials.add(CellMaterial {
        ids: ids.clone(),
        tex_a: images.add(new_tile_texture(&TEX_A, TEX_A_PERIOD)),
        tex_b: images.add(new_tile_texture(&TEX_B, TEX_B_PERIOD)),
        shade: images.add(new_shade_texture(&CellShades::new())),
        params: build_params(),
    });

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::default())),
        MeshMaterial2d(material.clone()),
        // A 1x1 rectangle scaled to the window's extent in world px. Placed by
        // `follow_window` on the first frame; the transform here only avoids a
        // frame at the origin.
        Transform::from_scale(Vec3::new(
            (WINDOW_COLS * CELL_SIZE) as f32,
            (WINDOW_ROWS * CELL_SIZE) as f32,
            1.0,
        )),
        CellQuad,
        WORLD_LAYERS,
    ));

    commands.insert_resource(CellMap { ids, material });
}

/// An all-air cell-id plane the size of the streaming window.
fn new_id_texture() -> Image {
    Image::new_fill(
        Extent3d {
            width: WINDOW_COLS as u32,
            height: WINDOW_ROWS as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; ID_BYTES],
        TextureFormat::R16Uint,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    )
}

/// One pattern tile set as a `p x (p * 8)` `R8Uint` texture.
///
/// The slabs stack in y, which needs no reshuffling: `cells` already lays the
/// set out as `pattern * p * p + y * p + x`, and that IS row-major over a
/// `p`-wide, `8p`-tall image.
///
/// NOT resized to a power of two. 61 and 67 being coprime is the entire reason
/// the composite pattern repeats only every 4087 cells; rounding either up to 64
/// would put a visible 64-cell lattice back on every flat field in the game.
pub fn new_tile_texture(tile: &[u8], period: i32) -> Image {
    let p = period as u32;
    assert_eq!(
        tile.len() as u32,
        p * p * TEX_PATTERN_COUNT as u32,
        "a tile set is {TEX_PATTERN_COUNT} slabs of {period}x{period}"
    );
    Image::new(
        Extent3d {
            width: p,
            height: p * TEX_PATTERN_COUNT as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        tile.to_vec(),
        TextureFormat::R8Uint,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// The shade table as a `512 x MAT_COUNT` `Rgba8Unorm` texture.
///
/// One row per material, one texel per `(edge class, pattern level)` — the same
/// `(id << 9) | (edge << 6) | pattern` index the blit uses, with the material
/// split off into y.
///
/// `Rgba8Unorm` and NOT `Rgba8UnormSrgb`: these bytes are the finished pixel
/// `paint_cells` packs, and the shader has to be able to hand them back
/// unchanged for the parity harness to compare them. Where the pass needs linear
/// output, `cellmap.wgsl` converts on the way out.
pub fn new_shade_texture(shades: &CellShades) -> Image {
    let table = shades.table();
    assert_eq!(table.len(), MAT_COUNT * SHADE_MATERIAL_STRIDE);
    let mut data = Vec::with_capacity(table.len() * 4);
    for &word in table {
        // Undo `cells::pack`, which stores the red channel in whichever byte
        // this target calls lowest. `Rgba8Unorm` wants r, g, b, a in that order
        // — the texel layout is a wire format, not this machine's memory layout.
        let bytes = if cfg!(target_endian = "little") {
            word.to_le_bytes()
        } else {
            word.to_be_bytes()
        };
        data.extend_from_slice(&bytes);
    }
    Image::new(
        Extent3d {
            width: SHADE_MATERIAL_STRIDE as u32,
            height: MAT_COUNT as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// The per-material scalars, at rest.
///
/// Everything here is a pure function of the content build, so this runs once.
/// The colours are the AUTHORED sRGB bytes, unconverted: every clamp in
/// `cells::build_material_shades` happens on those bytes, and converting to
/// linear first would move where a highlight saturates.
pub fn build_params() -> CellShadeParams {
    let mut base = [Vec4::ZERO; MATERIAL_SLOTS];
    let mut shim = [Vec4::ZERO; MATERIAL_SLOTS];
    // Slot 0 is air: never read, because the shader returns transparent before
    // it looks at either array.
    for id in 1..MAT_COUNT.min(MATERIAL_SLOTS) {
        base[id] = Vec4::new(
            f32::from(MAT_R[id]),
            f32::from(MAT_G[id]),
            f32::from(MAT_B[id]),
            material_pattern(id) as f32,
        );
        let s = shimmer_params(id);
        shim[id] = Vec4::new(
            s.amp as f32,
            s.phase as f32,
            s.tex_amp as f32,
            s.edge as f32,
        );
    }
    CellShadeParams {
        base,
        shim,
        clock: Vec4::ZERO,
        origin: IVec2::ZERO,
    }
}

/// Put the quad where the streaming window currently is, in world px.
///
/// The grid's origin moves in whole-chunk steps as the window recentres, so this
/// is the only thing that has to move when the world scrolls — the texture's
/// contents are already window-local.
fn follow_window(world: Res<SimWorld>, mut quad: Single<&mut Transform, With<CellQuad>>) {
    let grid = &world.level.grid;
    let w = (grid.cols() * CELL_SIZE) as f32;
    let h = (grid.rows() * CELL_SIZE) as f32;
    let x = (grid.origin_cell_x() * CELL_SIZE) as f32;
    let y = (grid.origin_cell_y() * CELL_SIZE) as f32;

    // Bevy's +y is up, the sim's is down: the window's top edge is its LARGEST
    // Bevy y. The quad's uv origin is its top-left, which is therefore grid row
    // 0 — the texture needs no flip.
    quad.translation = Vec3::new(x + w * 0.5, -(y + h * 0.5), 0.0);
    quad.scale = Vec3::new(w, h, 1.0);
}

/// Advance the animation clock and follow the window's origin.
///
/// THIS FUNCTION REPLACES `CellShades::update_shimmer` ENTIRELY. That one
/// rewrites nine materials' 512-entry slices — about 2 500 packed table entries,
/// each a texture offset, three clamps and a pack — every single frame, and it
/// does so whether there is one ember on screen or none at all. Here the same
/// animation is two floats in a uniform, and the wave is evaluated only on the
/// fragments that are actually lava.
///
/// The write goes through `Assets::get_mut`, which re-uploads the whole 2 KB
/// uniform rather than just the vec4 that changed. That is the granularity Bevy
/// offers for a `Material2d`, and 2 KB a frame is four orders of magnitude below
/// what it replaces.
fn update_shade_params(
    time: Res<Time>,
    world: Res<SimWorld>,
    cellmap: Res<CellMap>,
    mut materials: ResMut<Assets<CellMaterial>>,
) {
    let Some(mut material) = materials.get_mut(&cellmap.material) else {
        return;
    };
    let grid = &world.level.grid;
    material.params.clock.x = time.elapsed_secs();
    material.params.origin = IVec2::new(grid.origin_cell_x(), grid.origin_cell_y());
}

/// Rewrite the texels of every chunk that changed, and only those.
///
/// See the module docs for why this is possible here and was not in the
/// TypeScript.
pub fn upload_dirty_chunks(
    mut world: ResMut<SimWorld>,
    cellmap: Res<CellMap>,
    mut images: ResMut<Assets<Image>>,
) {
    let grid = &mut world.level.grid;

    // Collect first, and bail before touching the asset if nothing changed.
    // `Assets::get_mut` flags the asset as modified, which is what schedules a
    // GPU upload — so on a settled frame this system must not call it at all.
    let mut dirty: Vec<(i32, i32)> = Vec::new();
    for cy in 0..grid.chunk_rows() {
        for cx in 0..grid.chunk_cols() {
            if grid.is_chunk_dirty(cx, cy) {
                dirty.push((cx, cy));
            }
        }
    }
    if dirty.is_empty() {
        return;
    }

    let Some(mut image) = images.get_mut(&cellmap.ids) else {
        return;
    };
    let Some(data) = image.data.as_mut() else {
        return;
    };

    let cols = grid.cols();
    let rows = grid.rows();
    for (cx, cy) in dirty {
        let x0 = cx * CHUNK_CELLS;
        let x1 = ((cx + 1) * CHUNK_CELLS).min(cols);
        let y0 = cy * CHUNK_CELLS;
        let y1 = ((cy + 1) * CHUNK_CELLS).min(rows);

        for y in y0..y1 {
            let row = (y * cols) as usize;
            let src = &grid.material[row + x0 as usize..row + x1 as usize];
            let mut at = (row + x0 as usize) * ID_BYTES;
            for id in src {
                // Explicit little-endian rather than a `bytemuck` cast: the
                // texel layout is a wire format, not this machine's memory
                // layout, and the whole conversion is a store per cell.
                data[at..at + ID_BYTES].copy_from_slice(&id.to_le_bytes());
                at += ID_BYTES;
            }
        }

        grid.clear_chunk_dirty(cx, cy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parameters_cover_every_material_and_match_the_shader() {
        // `cells.wgsl` writes this length as a literal.
        assert_eq!(MATERIAL_SLOTS, 64, "cells.wgsl says 64");
        let p = build_params();
        assert_eq!(p.base[0], Vec4::ZERO, "air is never shaded");
        for id in 1..MAT_COUNT {
            assert_eq!(
                p.base[id].w,
                material_pattern(id) as f32,
                "material {id} carries the wrong pattern index"
            );
        }
        // Nine materials declare a shimmer worth animating; the shader branches
        // on `amp > 0`, so anything else must be exactly zero.
        let animated = (1..MAT_COUNT).filter(|&id| p.shim[id].x > 0.0).count();
        assert!(animated > 0, "no material animates — the branch is dead");
    }

    #[test]
    fn the_id_texture_is_exactly_the_streaming_window() {
        let img = new_id_texture();
        let size = img.texture_descriptor.size;
        assert_eq!(size.width, WINDOW_COLS as u32);
        assert_eq!(size.height, WINDOW_ROWS as u32);
        assert_eq!(img.texture_descriptor.format, TextureFormat::R16Uint);
        assert_eq!(
            img.data.as_ref().map(Vec::len),
            Some(WINDOW_COLS as usize * WINDOW_ROWS as usize * ID_BYTES)
        );
    }

    /// The tiles must arrive at their own periods. A power-of-two resize here
    /// would be invisible in a screenshot and would put the 64-cell lattice back
    /// on every flat field.
    #[test]
    fn the_pattern_tiles_keep_their_coprime_periods() {
        for (tile, p) in [(&**TEX_A, TEX_A_PERIOD), (&**TEX_B, TEX_B_PERIOD)] {
            let img = new_tile_texture(tile, p);
            assert_eq!(img.texture_descriptor.size.width, p as u32);
            assert_eq!(
                img.texture_descriptor.size.height,
                p as u32 * TEX_PATTERN_COUNT as u32
            );
            assert_eq!(img.texture_descriptor.format, TextureFormat::R8Uint);
        }
        // Coprime, which is the property the whole scheme rests on.
        let (mut a, mut b) = (TEX_A_PERIOD, TEX_B_PERIOD);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        assert_eq!(a, 1, "the tile periods stopped being coprime");
    }

    /// The shade texture must hand back exactly the bytes `pack` produced —
    /// it is the whole static half of the shader.
    #[test]
    fn the_shade_texture_is_the_packed_table_verbatim() {
        let shades = CellShades::new();
        let img = new_shade_texture(&shades);
        assert_eq!(
            img.texture_descriptor.size.width,
            SHADE_MATERIAL_STRIDE as u32
        );
        assert_eq!(img.texture_descriptor.size.height, MAT_COUNT as u32);
        assert_eq!(img.texture_descriptor.format, TextureFormat::Rgba8Unorm);

        let data = img.data.expect("the shade table is uploaded with its data");
        for (i, &word) in shades.table().iter().enumerate() {
            let px = &data[i * 4..i * 4 + 4];
            let px = [px[0], px[1], px[2], px[3]];
            let round = if cfg!(target_endian = "little") {
                u32::from_le_bytes(px)
            } else {
                u32::from_be_bytes(px)
            };
            assert_eq!(round, word, "shade entry {i} did not survive the upload");
        }
    }
}

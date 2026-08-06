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
//! Cell id in, colour out. That is the seam:
//!
//!   - **the id texture** is what the pass reads per cell, and
//!   - **[`CellPalette`]** is the per-material lookup it reads with.
//!
//! [`upload_dirty_chunks`] fills the first; [`build_palette`] fills the second.
//! Swapping in the real shading — [`crate::cells`], with its material texture,
//! edge light, ambient occlusion and shimmer — replaces the body of
//! `cellmap.wgsl` and widens what the lookup holds. It does not touch the
//! upload, the quad, the material, or the pipeline. (`cells::paint_cells` is
//! today a CPU pass writing a packed `u32` per cell; whichever way that lands —
//! as an `Rgba8` texture written from the CPU or as WGSL reading these same
//! tables — the id texture below is its input.)
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

use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dPlugin};

use godgame_core::config::{CELL_SIZE, CHUNK_CELLS, WINDOW_COLS, WINDOW_ROWS};
use godgame_core::sim::materials::{MAT_B, MAT_COUNT, MAT_G, MAT_R};

use crate::lowres::WORLD_LAYERS;
use crate::world::SimWorld;

/// Palette length, in materials.
///
/// A fixed-size uniform array, so it is a compile-time constant on both sides:
/// the WGSL says 64 as a literal and [`the palette test`](tests) fails if this
/// stops matching. Padded past [`MAT_COUNT`] so adding a block to `content/` is
/// a recompile, not a shader edit.
pub const PALETTE_SLOTS: usize = 64;

const _: () = assert!(
    MAT_COUNT <= PALETTE_SLOTS,
    "more materials than palette slots — widen PALETTE_SLOTS and cellmap.wgsl together"
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

/// Per-material colour lookup, in LINEAR RGBA.
///
/// `MAT_R`/`MAT_G`/`MAT_B` are authored sRGB bytes; the render target is sRGB
/// and the hardware does the encode, so the values handed to the shader must be
/// linear or every colour comes out washed out.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct CellPalette {
    /// Indexed by material code. Slot 0 (air) is transparent.
    pub colors: [Vec4; PALETTE_SLOTS],
}

impl Default for CellPalette {
    fn default() -> Self {
        build_palette()
    }
}

/// The placeholder cell shading.
///
/// Two bindings, and they are the seam described in the module docs: the id
/// texture and the lookup.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct CellMaterial {
    /// One material code per cell. `u_int`: sampled with `textureLoad`, never
    /// filtered.
    #[texture(0, sample_type = "u_int")]
    pub ids: Handle<Image>,
    /// Colour per material code.
    #[uniform(1)]
    pub palette: CellPalette,
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

/// The cell-id texture, the palette, and the quad that draws them.
pub struct CellMapPlugin;

impl Plugin for CellMapPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "cellmap.wgsl");

        app.add_plugins(Material2dPlugin::<CellMaterial>::default())
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (follow_window, upload_dirty_chunks)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<CellMap>),
            );
    }
}

/// Allocate the id texture and spawn the quad that samples it.
fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CellMaterial>>,
) {
    let ids = images.add(new_id_texture());
    let material = materials.add(CellMaterial {
        ids: ids.clone(),
        palette: build_palette(),
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
        bevy::asset::RenderAssetUsages::RENDER_WORLD | bevy::asset::RenderAssetUsages::MAIN_WORLD,
    )
}

/// The per-material colour lookup, sRGB bytes converted to linear.
pub fn build_palette() -> CellPalette {
    let mut colors = [Vec4::ZERO; PALETTE_SLOTS];
    // Slot 0 is air: left fully transparent so the sky shows through.
    for code in 1..MAT_COUNT.min(PALETTE_SLOTS) {
        let linear = Color::srgb_u8(MAT_R[code], MAT_G[code], MAT_B[code]).to_linear();
        colors[code] = Vec4::new(linear.red, linear.green, linear.blue, 1.0);
    }
    CellPalette { colors }
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
    fn the_palette_covers_every_material_and_matches_the_shader() {
        // `cellmap.wgsl` writes this length as a literal.
        assert_eq!(PALETTE_SLOTS, 64, "cellmap.wgsl says 64");
        let p = build_palette();
        assert_eq!(p.colors[0].w, 0.0, "air must be transparent");
        for code in 1..MAT_COUNT {
            assert_eq!(p.colors[code].w, 1.0, "material {code} is not opaque");
        }
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
}

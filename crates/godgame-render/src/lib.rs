//! Every draw pass.
//!
//! The whole game renders into a low-resolution offscreen target at
//! `VIEW_W x VIEW_H` and is then blitted to the window with a nearest-neighbour
//! pass. That is not a shortcut — the art, and especially the 8-13px UI type,
//! is authored to be upscaled with hard pixel edges.
//!
//! # The plugins
//!
//! | Plugin | Owns |
//! |---|---|
//! | [`world::WorldSimPlugin`] | the owned `Level` / `WindowManager` / `Automata` and the 120 Hz fixed schedule |
//! | [`lowres::LowResPlugin`] | the offscreen buffer, the two cameras, the upscale blit |
//! | [`cellmap::CellMapPlugin`] | the cell-id texture, the shading tables, and the quad that draws the window |
//!
//! [`GodGameRenderPlugin`] is all three, which is what the binary wants.
//!
//! # The cell rasteriser, twice
//!
//! [`cells`] is the CPU rasteriser, ported byte-for-byte from the TypeScript and
//! verified against it by `tests/ts_cells_parity.rs`. [`cellmap`] is the same
//! pass in WGSL, and it is the one the frame actually runs.
//!
//! The CPU one stays, and not as dead weight. It is the ORACLE:
//! `tests/shader_matches_cpu.rs` renders the shader headlessly over a real
//! worldgen window and diffs the framebuffer against it. It is also the readable
//! statement of what the shading IS — `cells.rs` explains every constant the
//! shader merely uses, and `cells.wgsl` points back at it by name throughout.

use bevy::app::{PluginGroup, PluginGroupBuilder};

pub mod cellmap;
pub mod cells;
pub mod lowres;
pub mod world;

/// The whole render/sim stack: the owned world and its fixed schedule, the
/// low-res target, and the cell pass that draws the window into it.
pub struct GodGameRenderPlugin;

impl PluginGroup for GodGameRenderPlugin {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(world::WorldSimPlugin)
            .add(lowres::LowResPlugin)
            .add(cellmap::CellMapPlugin)
    }
}

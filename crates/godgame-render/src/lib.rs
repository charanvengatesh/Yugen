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
//! | [`input::InputPlugin`] | keyboard and mouse onto `Intent`, the dig/place brush, and who drives the view |
//! | [`player::PlayerPlugin`] | the body, its fixed step, the camera that follows it, and the figure on screen |
//! | [`mobs::MobsPlugin`] | the creatures, their fixed step, the arrow hit test, and both shot pools on screen |
//! | [`items::ItemsPlugin`] | the pack, the stacks on the floor, and the loot that becomes them |
//! | [`daynight::DayNightPlugin`] | the one world clock, and the daylight weight the spawner reads |
//! | [`sky::SkyPlugin`] | the gradient, the stars, the two discs, and the ridgeline |
//! | [`weather::WeatherPlugin`] | dust, snow, spores and embers drifting across the view |
//! | [`ambience::AmbiencePlugin`] | the biome's motes — what makes a cave feel unlike a forest |
//! | [`particles::ParticlesPlugin`] | the pooled debris, dust and sparks every impact throws |
//! | [`effects::EffectsPlugin`] | screen shake and the hit flash |
//! | [`light::LightPlugin`] | the skylight flood, the emissive splat, and the composite over all of it |
//! | [`sprite::SpritePlugin`] | the baked sprite atlases, from the compiled content tables |
//! | [`scenes::ScenesPlugin`] | menu, playing, game over |
//! | [`ui::UiPlugin`] | the bitmap font, the HUD, the hotbar and the screens |
//! | [`glue::GluePlugin`] | the joins between modules that must not know each other |
//!
//! [`input`] is in this crate and not in the binary because it is the other half
//! of the same boundary the rest of the crate is: `godgame-core` may not know
//! what a `KeyCode` is, so something between it and Bevy has to, and that thing
//! belongs with the other Bevy-facing translations rather than in a `main.rs`.
//!
//! [`GodGameRenderPlugin`] is all of them, which is what the binary wants.
//!
//! # The cell rasteriser, twice
//!
//! [`cells`] is the CPU rasteriser, ported byte-for-byte from the TypeScript and
//! verified against it by `tests/cells_golden.rs`. [`cellmap`] is the same
//! pass in WGSL, and it is the one the frame actually runs.
//!
//! The CPU one stays, and not as dead weight. It is the ORACLE:
//! `tests/shader_matches_cpu.rs` renders the shader headlessly over a real
//! worldgen window and diffs the framebuffer against it. It is also the readable
//! statement of what the shading IS — `cells.rs` explains every constant the
//! shader merely uses, and `cells.wgsl` points back at it by name throughout.

use bevy::app::{PluginGroup, PluginGroupBuilder};

pub mod ambience;
pub mod cellmap;
pub mod cells;
pub mod daynight;
pub mod debug;
pub mod effects;
pub mod glue;
pub mod input;
pub mod items;
pub mod light;
pub mod lowres;
pub mod mobs;
pub mod particles;
pub mod player;
pub mod player_art;
pub mod scenes;
pub mod shear;
pub mod sky;
pub mod sprite;
pub mod ui;
pub mod weather;
pub mod world;

/// The whole render/sim stack: the owned world and its fixed schedule, the
/// low-res target, the cell pass that draws the window into it, the input that
/// moves the view and edits the cells, and the body those inputs move.
pub struct GodGameRenderPlugin;

impl PluginGroup for GodGameRenderPlugin {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(world::WorldSimPlugin)
            .add(lowres::LowResPlugin)
            .add(cellmap::CellMapPlugin)
            .add(input::InputPlugin)
            .add(player::PlayerPlugin)
            .add(mobs::MobsPlugin)
            .add(items::ItemsPlugin)
            // The clock first: every atmosphere pass below reads `WorldClock`,
            // and exactly one thing in the app may advance it.
            .add(daynight::DayNightPlugin)
            .add(sky::SkyPlugin)
            .add(weather::WeatherPlugin)
            .add(ambience::AmbiencePlugin)
            .add(particles::ParticlesPlugin)
            .add(effects::EffectsPlugin)
            // Bakes in `PreStartup`, so anything in `Startup` can read the
            // atlases without an ordering edge.
            .add(sprite::SpritePlugin)
            .add(scenes::ScenesPlugin)
            // Composites over everything the world passes drew.
            .add(light::LightPlugin)
            // The overlay sits above the composite: the HUD is not in the world
            // and must not be dimmed by the world's darkness.
            //
            // The F3 panel gathers in `PreUpdate` and is DRAWN by `ui::compose`,
            // so it has to be installed before the plugin that reads its
            // readout. It draws nothing until somebody presses the key.
            .add(debug::DebugPlugin)
            .add(ui::UiPlugin)
            // Last of all — every seam it joins must exist before it reaches
            // across one. See `glue`.
            .add(glue::GluePlugin)
    }
}

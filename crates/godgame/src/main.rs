//! The binary: Bevy app, scene state, input, and the sim/render wiring.
//!
//! Everything of substance is in `godgame-render`'s four plugins (see
//! [`godgame_render`]). What is left here is the shell:
//!
//!   - the window, and `ImagePlugin::default_nearest()` — EVERY sampler in this
//!     game is nearest, because all of the art is pixel art and the whole frame
//!     is upscaled from a low-resolution buffer;
//!   - `--screenshot <path>`, which renders a few frames and writes a PNG. That
//!     exists so an agent, or CI, can prove the thing draws without a human at
//!     the keyboard;
//!   - `--edit <mode> <cx> <cy> <r>`, which stamps one brush stroke before the
//!     capture, so the same agent can prove the world CHANGES without a hand on
//!     the mouse. See [`StartupEdit`].
//!
//! The free camera used to live here. It is input, so it moved to
//! [`godgame_render::input`] along with the mouse and the brush; what is left is
//! genuinely only the shell.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PresentMode, WindowResolution};

use godgame_core::sim::edits::{EditMode, apply_brush};
use godgame_core::sim::materials::{CellId, EMPTY, code_of};
use godgame_render::GodGameRenderPlugin;
use godgame_render::world::{SimWorld, WorldFocus};

/// Frames to render before `--screenshot` captures.
///
/// The first frames have nothing in the cell texture yet — the upload runs in
/// `Update`, after `Startup` inserted the world — and the pipeline specialises
/// lazily. A handful of frames is enough for the shader to compile and the
/// first upload to land.
const SCREENSHOT_WARMUP_FRAMES: u32 = 30;

/// Default window size, in logical px. See [`check_default_window`].
const DEFAULT_WINDOW: (u32, u32) = (1280, 800);

/// Where a `--screenshot` run writes to, and how long until it does.
#[derive(Resource)]
struct ScreenshotRun {
    path: PathBuf,
    frames_left: u32,
    taken: bool,
}

/// One brush stroke stamped at load, from `--edit MODE CX CY R`.
///
/// The brush is a mouse verb, and a `--screenshot` run has no mouse. This is the
/// same stroke [`godgame_render::input`] would emit, taken from the command line
/// instead: it is what lets an agent or CI show that digging changes the world
/// and that the automata reacts to the hole, rather than assert it from a test
/// that never drew a pixel.
///
/// `cx`/`cy` are cells RELATIVE TO THE VIEW CENTRE, because that is the only
/// coordinate a caller knows without first reading the spawn point out of the
/// worldgen.
#[derive(Resource, Clone, Copy, Debug)]
struct StartupEdit {
    mode: EditMode,
    /// Material to place. Ignored when digging.
    mat: CellId,
    /// Cells right of the view centre.
    cx: i32,
    /// Cells below the view centre.
    cy: i32,
    /// Disc radius in cells.
    r: i32,
}

/// What the command line asked for, parsed once.
struct Args {
    screenshot: Option<PathBuf>,
    edit: Option<StartupEdit>,
}

fn main() -> AppExit {
    let args = parse_args();

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            // The one plugin setting here that is not a preference: the upscale
            // from the low-resolution buffer to the window must not filter.
            .set(ImagePlugin::default_nearest())
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "GodGame".into(),
                    resolution: WindowResolution::new(DEFAULT_WINDOW.0, DEFAULT_WINDOW.1),
                    present_mode: PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
    )
    .add_plugins(GodGameRenderPlugin);

    if let Some(edit) = args.edit {
        app.insert_resource(edit).add_systems(
            Update,
            stamp_startup_edit
                .run_if(resource_exists::<StartupEdit>)
                .run_if(resource_exists::<SimWorld>),
        );
    }

    if let Some(path) = args.screenshot {
        app.insert_resource(ScreenshotRun {
            path,
            frames_left: SCREENSHOT_WARMUP_FRAMES,
            taken: false,
        })
        .add_systems(Update, screenshot_then_exit);
    }

    app.run()
}

/// `--screenshot PATH` and `--edit MODE CX CY R`.
///
/// Hand-rolled rather than a dependency: two flags, both of them development
/// scaffolding, is not worth an argument parser in the tree.
fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut args = Args {
        screenshot: None,
        edit: None,
    };
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--screenshot" => {
                let Some(path) = argv.next() else { usage() };
                args.screenshot = Some(PathBuf::from(path));
            }
            "--edit" => {
                let mode = argv.next().unwrap_or_else(|| usage());
                let mat = if mode == "dig" {
                    EMPTY
                } else {
                    match code_of(&mode) {
                        EMPTY => {
                            eprintln!("--edit: {mode:?} is neither \"dig\" nor a block id");
                            usage()
                        }
                        code => code,
                    }
                };
                args.edit = Some(StartupEdit {
                    mode: if mat == EMPTY {
                        EditMode::Dig
                    } else {
                        EditMode::Place
                    },
                    mat,
                    cx: next_int(&mut argv),
                    cy: next_int(&mut argv),
                    r: next_int(&mut argv),
                });
            }
            other => {
                eprintln!("unknown argument {other:?}");
                usage()
            }
        }
    }
    args
}

/// The next argument as an integer, or the usage message.
fn next_int(argv: &mut impl Iterator<Item = String>) -> i32 {
    match argv.next().map(|a| a.parse()) {
        Some(Ok(n)) => n,
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage: godgame [--screenshot PATH] [--edit dig|BLOCK_ID CX CY R]");
    eprintln!("  --edit  one brush stroke at load; CX/CY are cells from the view centre");
    std::process::exit(2)
}

/// Stamp the `--edit` stroke once, the first frame the world exists.
///
/// This is deliberately the same [`apply_brush`] call the mouse path makes and
/// not a private shortcut: a flag that wrote cells its own way could pass while
/// the thing it is supposed to be demonstrating was broken.
fn stamp_startup_edit(
    mut commands: Commands,
    edit: Res<StartupEdit>,
    focus: Res<WorldFocus>,
    mut world: ResMut<SimWorld>,
) {
    let cx = godgame_core::config::cell_at(focus.x) + edit.cx;
    let cy = godgame_core::config::cell_at(focus.y) + edit.cy;
    apply_brush(&mut world.level.grid, edit.mode, cx, cy, edit.r, edit.mat);
    info!("--edit {:?} at cell ({cx}, {cy}) r={}", edit.mode, edit.r);
    // Once is the whole contract; removing the resource stops the system.
    commands.remove_resource::<StartupEdit>();
}

/// Render a few frames, write a PNG, then quit.
fn screenshot_then_exit(
    mut commands: Commands,
    mut run: ResMut<ScreenshotRun>,
    pending: Query<Entity, With<Screenshot>>,
    mut exit: MessageWriter<AppExit>,
) {
    if !run.taken {
        run.frames_left = run.frames_left.saturating_sub(1);
        if run.frames_left == 0 {
            let path = run.path.clone();
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
            run.taken = true;
        }
        return;
    }

    // `save_to_disk` writes the file synchronously from its observer, and the
    // screenshot entity is despawned once captured — so an empty query means
    // the PNG is on disk.
    if pending.is_empty() {
        exit.write(AppExit::Success);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::config::View;

    #[test]
    fn check_default_window() {
        // `View::for_screen` floors at zoom 2, so any window at or under
        // 1600x1000 physical px upscales by exactly 2 — a whole-pixel blit with
        // no resampling. The default is chosen to sit inside that.
        let v = View::for_screen(DEFAULT_WINDOW.0, DEFAULT_WINDOW.1);
        assert_eq!(v.zoom, 2.0);
        assert_eq!((v.w, v.h), (640, 400));
    }
}

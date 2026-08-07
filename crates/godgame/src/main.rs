//! The binary: Bevy app, scene state, input, and the sim/render wiring.
//!
//! Everything of substance is in `godgame-render`'s plugins (see
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
//!     the mouse. See [`StartupEdit`];
//!   - `--free-camera`, which starts with no player at all, and `--drive`,
//!     which runs the body right by itself so an agent can see it MOVE. See
//!     [`godgame_render::player::NoPlayer`].
//!
//! The free camera used to live here. It is input, so it moved to
//! [`godgame_render::input`] along with the mouse and the brush; what is left is
//! genuinely only the shell.

use std::path::PathBuf;

use bevy::app::{RunFixedMainLoop, RunFixedMainLoopSystems};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PresentMode, WindowResolution};

use godgame_core::sim::edits::{EditMode, apply_brush};
use godgame_core::sim::materials::{CellId, EMPTY, code_of};
use godgame_render::GodGameRenderPlugin;
use godgame_render::input::PlayerIntent;
use godgame_render::items::GroundItems;
use godgame_render::mobs::Creatures;
use godgame_render::player::{NoPlayer, PlayerBody};
use godgame_render::scenes::Scene;
use godgame_render::world::{SimWorld, WorldFocus};

/// Frames to render before `--screenshot` captures, by default.
///
/// The first frames have nothing in the cell texture yet — the upload runs in
/// `Update`, after `Startup` inserted the world — and the pipeline specialises
/// lazily. A handful of frames is enough for the shader to compile and the
/// first upload to land.
///
/// It is NOT enough for anything that has to happen in the world first. The
/// creatures spawn on a 0.15s timer and take a couple of seconds to populate a
/// screen, so a shot of them wants `--warmup` in the hundreds. That is a flag
/// rather than a bigger default because most captures only want the terrain and
/// should not pay two seconds for it.
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

/// Hold "run right" and tap jump on a timer, with no hand on the keyboard.
///
/// The same scaffolding [`StartupEdit`] is, for the same reason. `--screenshot`
/// proves the frame DRAWS; this is what proves the frame MOVES — that the body
/// runs, clears a ledge, and drags the streaming window along behind it — from
/// an agent or a CI job that cannot synthesise a keypress.
///
/// It writes the same [`PlayerIntent`] the keyboard writes, in the slot between
/// `PreUpdate` (where the keyboard is sampled) and the fixed loop (where the
/// body is stepped). So everything downstream of it — the substep edge masking,
/// the 120 Hz step, the camera ease, the recentre — is the path a player's hands
/// take, not a private one that could pass while the real one was broken.
#[derive(Resource, Clone, Copy, Debug)]
struct Autodrive {
    /// Seconds between jump taps. Non-positive never jumps.
    jump_every: f32,
    /// Seconds since the last tap.
    since: f32,
}

/// What the command line asked for, parsed once.
struct Args {
    screenshot: Option<PathBuf>,
    edit: Option<StartupEdit>,
    /// Start with no player, so WASD flies the view instead of moving a body.
    free_camera: bool,
    /// Start with the F3 panel up. A `--screenshot` run has no keyboard.
    debug_overlay: bool,
    /// Leave the title card immediately instead of waiting for a confirm key.
    play: bool,
    /// Run right by itself, jumping every this many seconds.
    drive: Option<f32>,
    /// Frames to render before `--screenshot` captures.
    warmup: u32,
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

    // Before the plugin group's `PostStartup` runs, which is the only thing
    // that reads it.
    if args.free_camera {
        app.insert_resource(NoPlayer);
    }

    // Leave the menu on the first frame.
    //
    // Without this every automated capture is a photograph of the TITLE CARD.
    // The world does render behind it — the menu is a dimming plate and a label
    // over a live scene — so the result looks enough like the game to be
    // accepted at a glance, and it took two screenshots this session before
    // anybody noticed the words "Press Enter or Space to start" across the
    // middle of them. `Scene::Menu` is the default and only a confirm KEY
    // advances it, which a headless run does not have.
    //
    // Implied by `--drive`, because a flag whose whole job is to move the body
    // is meaningless while the body has not been spawned.
    if args.play || args.drive.is_some() {
        app.add_systems(Startup, |mut next: ResMut<NextState<Scene>>| {
            next.set(Scene::Playing);
        });
    }

    if let Some(jump_every) = args.drive {
        app.insert_resource(Autodrive {
            jump_every,
            since: 0.0,
        })
        // `BeforeFixedMainLoop` is the one slot that is after the keyboard has
        // been read and before the body has been stepped. Anywhere in `Update`
        // would be a frame stale and would be overwritten by the next sample
        // before any fixed step ever saw it.
        .add_systems(
            RunFixedMainLoop,
            autodrive.in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
        )
        .add_systems(
            Update,
            drive_report
                .run_if(resource_exists::<PlayerBody>)
                .run_if(resource_exists::<SimWorld>),
        );
    }

    if let Some(edit) = args.edit {
        app.insert_resource(edit).add_systems(
            Update,
            stamp_startup_edit
                .run_if(resource_exists::<StartupEdit>)
                .run_if(resource_exists::<SimWorld>),
        );
    }

    // Started ON rather than toggled, because a headless `--screenshot` run has
    // nobody to press F3 and the panel is most useful in exactly those captures.
    if args.debug_overlay {
        app.insert_resource(godgame_render::debug::DebugOverlay(true));
    }

    if let Some(path) = args.screenshot {
        app.insert_resource(ScreenshotRun {
            path,
            frames_left: args.warmup,
            taken: false,
        })
        .add_systems(Update, screenshot_then_exit);
    }

    app.run()
}

/// `--screenshot PATH`, `--edit MODE CX CY R`, `--free-camera` and `--drive SECS`.
///
/// Hand-rolled rather than a dependency: four flags, all of them development
/// scaffolding, is not worth an argument parser in the tree.
fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let mut args = Args {
        screenshot: None,
        edit: None,
        free_camera: false,
        debug_overlay: false,
        play: false,
        drive: None,
        warmup: SCREENSHOT_WARMUP_FRAMES,
    };
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--free-camera" => args.free_camera = true,
            "--debug-overlay" => args.debug_overlay = true,
            "--play" => args.play = true,
            "--warmup" => args.warmup = next_int(&mut argv).max(1) as u32,
            "--drive" => {
                let secs = argv.next().unwrap_or_else(|| usage());
                args.drive = Some(match secs.parse::<f32>() {
                    Ok(s) if s.is_finite() => s,
                    _ => usage(),
                });
            }
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
    eprintln!(
        "usage: godgame [--screenshot PATH] [--warmup FRAMES] [--edit dig|BLOCK_ID CX CY R] [--free-camera] [--debug-overlay] [--play] [--drive SECS]"
    );
    eprintln!(
        "  --play           start the run at once; without it a capture photographs the menu"
    );
    eprintln!("  --debug-overlay  start with the F3 panel up (it has no keyboard in a capture)");
    eprintln!("  --edit         one brush stroke at load; CX/CY are cells from the view centre");
    eprintln!("  --free-camera  no player; WASD flies the view and streams the world");
    eprintln!("  --drive SECS   run right by itself, jumping every SECS (0 = never)");
    eprintln!(
        "  --warmup N     frames to render before --screenshot fires (default {SCREENSHOT_WARMUP_FRAMES}); creatures need a few hundred"
    );
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

/// Hold right, and raise a jump edge every [`Autodrive::jump_every`] seconds.
///
/// `jump_queued` is set for exactly one FRAME, not one step: the rising edge is
/// what the keyboard produces and what
/// [`Intent::for_substep`](godgame_core::input::Intent::for_substep) narrows to
/// one substep. Holding it true would be a different input from the one a
/// player can give, and it would pogo.
fn autodrive(time: Res<Time>, mut drive: ResMut<Autodrive>, mut intent: ResMut<PlayerIntent>) {
    intent.dir_x = 1.0;
    if drive.jump_every <= 0.0 {
        return;
    }
    drive.since += time.delta_secs();
    if drive.since >= drive.jump_every {
        drive.since = 0.0;
        intent.jump_queued = true;
    }
    // Held is what keeps a jump from being cut short a step after it starts; the
    // autodrive never releases early, so a tap here is a full-height jump.
    intent.jump_held = true;
    intent.up = true;
}

/// Once a second under `--drive`: where the body is, whether the window has
/// followed it, and how long a frame is taking.
///
/// The window origin is in the line because it is the streaming half of the
/// milestone and the one thing a screenshot cannot show: a body can run for a
/// mile in a window that never recentres, and the frame would look identical
/// right up to the moment it hit the unloaded margin and stopped dead.
fn drive_report(
    time: Res<Time>,
    body: Res<PlayerBody>,
    world: Res<SimWorld>,
    creatures: Res<Creatures>,
    ground: Res<GroundItems>,
    mut since: Local<f32>,
    mut frames: Local<u32>,
) {
    *since += time.delta_secs();
    *frames += 1;
    if *since < 1.0 {
        return;
    }
    let ms = *since * 1000.0 / *frames as f32;
    let grid = &world.level.grid;
    info!(
        "drive: body ({:.0}, {:.0}) vx {:>4.0} ground {} | window origin cell ({}, {}) | mobs {} drops {} | {ms:.2} ms/frame",
        body.x,
        body.y,
        body.vx,
        u8::from(body.on_ground),
        grid.origin_cell_x(),
        grid.origin_cell_y(),
        creatures.count(),
        ground.active(),
    );
    *since = 0.0;
    *frames = 0;
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

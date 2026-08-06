//! The binary: Bevy app, scene state, input, and the sim/render wiring.
//!
//! Everything of substance is in `godgame-render`'s three plugins (see
//! [`godgame_render`]). What is left here is the shell:
//!
//!   - the window, and `ImagePlugin::default_nearest()` — EVERY sampler in this
//!     game is nearest, because all of the art is pixel art and the whole frame
//!     is upscaled from a low-resolution buffer;
//!   - a free camera on WASD / arrow keys, so the world can be looked at before
//!     there is a player to look at it with;
//!   - `--screenshot <path>`, which renders a few frames and writes a PNG. That
//!     exists so an agent, or CI, can prove the thing draws without a human at
//!     the keyboard.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PresentMode, WindowResolution};

use godgame_core::config::MAX_RUN_SPEED;
use godgame_render::GodGameRenderPlugin;
use godgame_render::world::WorldFocus;

/// How much faster the free camera flies than the player runs.
///
/// The camera exists to inspect a streaming window 1760x1280 px across; at the
/// player's own speed, crossing it takes half a minute.
const CAMERA_SPEED_SCALE: f32 = 4.0;

/// Extra multiplier while shift is held.
const CAMERA_BOOST: f32 = 4.0;

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

fn main() -> AppExit {
    let shot = parse_args();

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
    .add_plugins(GodGameRenderPlugin)
    .add_systems(Update, fly_camera);

    if let Some(path) = shot {
        app.insert_resource(ScreenshotRun {
            path,
            frames_left: SCREENSHOT_WARMUP_FRAMES,
            taken: false,
        })
        .add_systems(Update, screenshot_then_exit);
    }

    app.run()
}

/// `--screenshot <path>`, and nothing else yet.
fn parse_args() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    let mut shot = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--screenshot" => {
                let Some(path) = args.next() else { usage() };
                shot = Some(PathBuf::from(path));
            }
            other => {
                eprintln!("unknown argument {other:?}");
                usage()
            }
        }
    }
    shot
}

fn usage() -> ! {
    eprintln!("usage: godgame [--screenshot PATH]");
    std::process::exit(2)
}

/// Fly the camera with WASD or the arrow keys; hold shift to go faster.
///
/// Writes [`WorldFocus`], which is both what the streaming window recentres on
/// and what the world camera follows — so this one system moves the view AND
/// pulls new chunks in behind it.
fn fly_camera(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut focus: ResMut<WorldFocus>) {
    let mut dir = Vec2::ZERO;
    if keys.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]) {
        dir.x -= 1.0;
    }
    if keys.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]) {
        dir.x += 1.0;
    }
    // +y is DOWN in world px, so "up" on the keyboard is negative.
    if keys.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]) {
        dir.y -= 1.0;
    }
    if keys.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]) {
        dir.y += 1.0;
    }
    if dir == Vec2::ZERO {
        return;
    }

    let boost = if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        CAMERA_BOOST
    } else {
        1.0
    };
    let step = dir.normalize() * MAX_RUN_SPEED * CAMERA_SPEED_SCALE * boost * time.delta_secs();
    focus.x += step.x;
    focus.y += step.y;
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

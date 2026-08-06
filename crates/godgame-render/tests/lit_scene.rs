//! A lit cave, captured, so that lighting changes can be looked at.
//!
//! # Why this exists
//!
//! `frame_capture.rs` boots the game at its spawn point, which is a snowy
//! surface in broad daylight. That is the right frame for asserting the game
//! draws at all, and it is the WORST frame for judging light: there are no
//! emitters in it, the sky washes out every additive pass, and the whole
//! lighting stack — the coloured splat, the bloom, the biome cast, the vignette
//! — is either invisible or clipped.
//!
//! So every attempt to evaluate a lighting change has had to build a lit scene
//! by hand, and the same scene has been built and thrown away three times now.
//! This is that scene, kept.
//!
//! # What it is not
//!
//! Not a gate. It asserts almost nothing, because there is no numeric property
//! of a cave worth failing a build over — "does this look right" is a question
//! for a person, and the honest thing is to hand them a picture rather than
//! invent a threshold that encodes one contributor's taste.
//!
//! What it DOES assert is that the scene it claims to have built is the scene it
//! captured: a dark room, with light in it, underground. If that stops being
//! true the PNG is worthless and silently so, which is the one failure here
//! worth catching.
//!
//! # Reading the picture
//!
//! Run it, then open the path it prints. Numbers alone mislead here — the two
//! things most often wrong with a light pass are its RESOLUTION (does the glow
//! sit on the same pixel grid as the art, or a coarser one) and its FALLOFF
//! SHAPE, and neither shows up in a mean or a colour count.
//!
//! ```text
//! cargo test -p godgame-render --test lit_scene -- --nocapture
//! ```

use std::path::PathBuf;

use bevy::app::PluginGroup;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::tasks::block_on;
use bevy::window::{WindowPlugin, WindowResolution};

use godgame_core::config::cell_at;
use godgame_core::sim::coords::WorldCell;
use godgame_core::sim::materials::{EMPTY, code_of};
use godgame_render::GodGameRenderPlugin;
use godgame_render::input::FocusDriver;
use godgame_render::lowres::LowResTarget;
use godgame_render::scenes::Scene;
use godgame_render::world::{SimWorld, WorldFocus};

/// How far below the spawn the room is carved, in world px.
///
/// Deep enough that the sky contributes nothing and the depth terms in the
/// vignette and the underworld glow have both engaged. Judging a light pass
/// against a frame the daylight still reaches tells you very little.
const DEPTH_PX: f32 = 900.0;

/// The room, in cells, measured from its centre.
const ROOM_HALF_W: i32 = 30;
/// See [`ROOM_HALF_W`].
const ROOM_HALF_H: i32 = 14;

/// Rows of lava along the floor of the room.
///
/// A pool rather than a point: one emitter shows the falloff, but only a broad
/// source shows whether the pass CLIPS — which is exactly what the sprite bloom
/// this rig was first built to judge did, over exactly this shape.
const LAVA_ROWS: i32 = 4;

/// Frames to let the world stream, settle and light before capturing.
const SETTLE_FRAMES: u32 = 60;

/// Frames to wait for the asynchronous screenshot readback.
const READBACK_FRAMES: u32 = 60;

#[test]
fn a_lit_cave_is_captured_for_a_human_to_look_at() {
    if !gpu_is_available() {
        println!("SKIPPED lit scene: no wgpu adapter on this machine");
        return;
    }

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(ImagePlugin::default_nearest())
            .set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: WindowResolution::new(1280, 800),
                    ..default()
                }),
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>(),
    )
    .add_plugins(GodGameRenderPlugin);
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    for _ in 0..SETTLE_FRAMES {
        app.update();
    }

    // Take the view off the body BEFORE moving it. `player::follow_player` eases
    // the focus back onto the body every frame and would drag the camera out of
    // the cave; it is gated on `FocusDriver::Player`, so flipping the driver is
    // what stops it. The body stays where it is — it is not what is being
    // photographed.
    let focus_x = app.world().resource::<WorldFocus>().x;
    let deep_y = app.world().resource::<WorldFocus>().y + DEPTH_PX;
    *app.world_mut().resource_mut::<FocusDriver>() = FocusDriver::FreeCamera;

    // Re-asserted every frame: the streamer recentres on the focus, and holding
    // it still for a while is what pulls the deep chunks in.
    for _ in 0..SETTLE_FRAMES {
        app.world_mut().resource_mut::<WorldFocus>().y = deep_y;
        app.update();
    }

    carve_and_light(&mut app, focus_x, deep_y);

    for _ in 0..SETTLE_FRAMES {
        app.world_mut().resource_mut::<WorldFocus>().y = deep_y;
        app.update();
    }

    let (air, floor) = sample_scene(&app, focus_x, deep_y);
    assert!(
        air,
        "the room was not carved — the capture is of solid rock and shows nothing"
    );
    assert!(
        floor,
        "the lava was not placed — the capture is of an unlit hole, which is not \
         a picture of a light pass"
    );

    let out = out_path();
    let canvas = app.world().resource::<LowResTarget>().canvas.clone();
    app.world_mut()
        .spawn(Screenshot::image(canvas))
        .observe(save_to_disk(out.clone()));
    for _ in 0..READBACK_FRAMES {
        app.world_mut().resource_mut::<WorldFocus>().y = deep_y;
        app.update();
    }

    assert!(
        out.exists(),
        "the readback never landed, so {} is stale or missing",
        out.display()
    );
    println!("lit cave written to {}", out.display());
}

/// Carve an air pocket at the focus and pour lava along its floor.
fn carve_and_light(app: &mut App, focus_x: f32, focus_y: f32) {
    let lava = code_of("lava");
    let (cx, cy) = (cell_at(focus_x), cell_at(focus_y));
    let mut world = app.world_mut().resource_mut::<SimWorld>();

    for dy in -ROOM_HALF_H..ROOM_HALF_H {
        for dx in -ROOM_HALF_W..ROOM_HALF_W {
            world
                .level
                .grid
                .set_world(WorldCell::new(cx + dx, cy + dy), EMPTY);
        }
    }
    for dy in (ROOM_HALF_H - LAVA_ROWS)..ROOM_HALF_H {
        for dx in -ROOM_HALF_W..ROOM_HALF_W {
            world
                .level
                .grid
                .set_world(WorldCell::new(cx + dx, cy + dy), lava);
        }
    }
}

/// `(the room is air, the floor is lava)` — that the scene is what it claims.
fn sample_scene(app: &App, focus_x: f32, focus_y: f32) -> (bool, bool) {
    let world = app.world().resource::<SimWorld>();
    let (cx, cy) = (cell_at(focus_x), cell_at(focus_y));
    let air = world.level.grid.get_world(WorldCell::new(cx, cy)) == EMPTY;
    let floor = world
        .level
        .grid
        .get_world(WorldCell::new(cx, cy + ROOM_HALF_H - 1))
        == code_of("lava");
    (air, floor)
}

/// Where the PNG lands. Inside `target/`, which is line 1 of `.gitignore`.
fn out_path() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("lit-scene");
    std::fs::create_dir_all(&dir).expect("target/ is writable");
    dir.join("cave.png")
}

/// Whether this machine has a GPU wgpu will talk to.
///
/// Asked BEFORE the app is built, because `RenderPlugin` panics rather than
/// returning an error when there is no adapter. Same discipline as
/// `shader_matches_cpu.rs`: no adapter means SKIP loudly, never pass quietly.
fn gpu_is_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_ok()
}

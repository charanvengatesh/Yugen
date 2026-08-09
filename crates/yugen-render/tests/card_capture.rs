//! Photographs of the two cards, for a human to look at.
//!
//! A rig, not a gate: it asserts only that each card drew something, and writes
//! the PNG where it can be opened. `control_frames` makes the same argument for
//! the world — there is no numeric property of a title card worth failing a
//! build over, but there is every reason to be able to see one.

mod common;

use bevy::prelude::*;
use yugen_render::scenes::{Paused, Scene};

#[test]
fn photograph_the_menu_and_the_pause_card() {
    if !common::gpu_is_available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let mut app = common::headless_game("Yūgen cards");
    app.finish();
    app.cleanup();

    // The menu, which is where `Scene` already starts.
    for _ in 0..8 {
        app.update();
    }
    let png = common::artefact_path(env!("CARGO_TARGET_TMPDIR"), "cards", "menu.png");
    common::capture_low_res(&mut app, &png);
    eprintln!("menu: {}", png.display());

    // Then a paused run.
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);
    for _ in 0..24 {
        app.update();
    }
    app.world_mut().insert_resource(Paused(true));
    app.update();
    let png = common::artefact_path(env!("CARGO_TARGET_TMPDIR"), "cards", "paused.png");
    common::capture_low_res(&mut app, &png);
    eprintln!("paused: {}", png.display());
}

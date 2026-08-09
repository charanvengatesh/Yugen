//! Photographs of every menu page, for a human to look at.
//!
//! A rig, not a gate, on `control_frames`' terms. What it DOES assert is that
//! each page reached the display list — a menu that silently drew nothing would
//! otherwise be a black screen nobody noticed until they opened it.

mod common;

use bevy::prelude::*;
use yugen_render::scenes::{Paused, Scene};
use yugen_render::ui::menu::{Nav, Page};
use yugen_render::ui::{UiFrame, UiPrim};

fn text_of(app: &App) -> String {
    app.world()
        .resource::<UiFrame>()
        .prims
        .iter()
        .filter_map(|p| match p {
            UiPrim::Text { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn photograph_every_menu_page() {
    if !common::gpu_is_available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let mut app = common::headless_game("Yūgen menus");
    app.finish();
    app.cleanup();
    // Warm up before photographing: the first frame has no sky, no terrain and
    // no baked atlases, so a capture of it is a black rectangle with a menu on
    // it — which is exactly what the first version of this test produced.
    for _ in 0..16 {
        app.update();
    }

    // The title card, which is where the game already starts.
    let png = common::artefact_path(env!("CARGO_TARGET_TMPDIR"), "menus", "title.png");
    common::capture_low_res(&mut app, &png);
    assert!(text_of(&app).contains("Singleplayer"), "{}", text_of(&app));
    eprintln!("title: {}", png.display());

    // Then a paused world, and the options pages over it.
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);
    for _ in 0..24 {
        app.update();
    }
    app.world_mut().insert_resource(Paused(true));
    app.update();
    assert!(text_of(&app).contains("Back to Game"), "{}", text_of(&app));
    let png = common::artefact_path(env!("CARGO_TARGET_TMPDIR"), "menus", "pause.png");
    common::capture_low_res(&mut app, &png);
    eprintln!("pause: {}", png.display());

    for (page, file, want) in [
        (Page::Options, "options.png", "Video..."),
        (Page::Video, "video.png", "Render Scale"),
        (Page::Interface, "interface.png", "Control Hints"),
        (Page::World, "world.png", "Day Length"),
        (Page::Controls, "controls.png", "Dash"),
        (Page::Worlds, "worlds.png", "Create New World"),
    ] {
        app.world_mut().resource_mut::<Nav>().push(page);
        app.update();
        let got = text_of(&app);
        assert!(got.contains(want), "{page:?} drew nothing useful: {got}");
        let png = common::artefact_path(env!("CARGO_TARGET_TMPDIR"), "menus", file);
        common::capture_low_res(&mut app, &png);
        eprintln!("{page:?}: {}", png.display());
        app.world_mut().resource_mut::<Nav>().pop();
    }
}

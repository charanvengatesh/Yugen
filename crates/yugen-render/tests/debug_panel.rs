//! The F3 panel is actually routed to the screen.
//!
//! # Why this exists
//!
//! `debug.rs` carried four tests and every one of them called the pure
//! `overlay()` directly. That proves the layout is right and proves nothing at
//! all about whether the panel is REACHED — the readout could go ungathered, the
//! layer could be missing from `ui::page::stack`, the plugin could be absent
//! from the group, and all four would still pass.
//!
//! `worldselect`'s header makes this argument about its own screen and calls a
//! capture test the thing that settles it. This is the F3 panel's, and it is the
//! first end-to-end coverage the panel has ever had: nothing in this directory
//! had ever set `DebugOverlay(true)`.
//!
//! It asserts against `UiFrame` rather than pixels because the panel's content
//! is text, and a text assertion that survives a font change is worth more here
//! than a pixel one that does not.

mod common;

use bevy::prelude::*;
use yugen_render::debug::{DebugOverlay, DebugReadout};
use yugen_render::ui::{UiFrame, UiPrim};

/// A headless app with its render plugins finished, on the playing screen.
///
/// `finish` and `cleanup` are what insert the render-stage resources the
/// overlay reads; without them the first `PreUpdate` runs against an app that is
/// still half-built. `frame_capture` does the same two calls for the same
/// reason.
fn ready(title: &str) -> App {
    let mut app = common::headless_game(title);
    app.finish();
    app.cleanup();
    // Off the title card: the panel draws over every screen, but the rows worth
    // asserting on are the ones a live world fills in.
    app.world_mut()
        .resource_mut::<NextState<yugen_render::scenes::Scene>>()
        .set(yugen_render::scenes::Scene::Playing);
    app
}

/// Every string the overlay put in this frame's display list.
fn frame_text(app: &App) -> String {
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
fn the_panel_reaches_the_display_list_when_it_is_switched_on() {
    let mut app = ready("debug-panel");
    app.update();

    let before = frame_text(&app);
    assert!(
        !before.contains("frame"),
        "the panel drew without being asked for: {before}"
    );

    app.world_mut().insert_resource(DebugOverlay(true));
    app.update();

    let after = frame_text(&app);
    // The one row that is there world or no world, which is why `overlay` puts
    // it first: a panel whose top line is blank looks broken rather than idle.
    assert!(
        after.contains("frame"),
        "DebugOverlay(true) did not put the panel on screen: {after}"
    );
    // And the rows that are new, which the pure tests cannot prove are wired to
    // anything live.
    for want in ["view", "draw"] {
        assert!(after.contains(want), "no `{want}` row reached the frame");
    }
}

#[test]
fn the_readout_is_gathered_even_while_the_panel_is_hidden() {
    // `gather` runs every frame regardless, deliberately: the frame-time average
    // has to keep converging while the panel is down, or the first number a
    // developer sees after pressing the key is a spike from the frame that
    // pressed it.
    let mut app = ready("debug-gather");
    for _ in 0..4 {
        app.update();
    }
    let r = app.world().resource::<DebugReadout>();
    assert!(
        r.frame_ms > 0.0,
        "the frame clock never started while hidden"
    );
    assert!(r.view.0 > 0 && r.view.1 > 0, "the view was never read");
}

#[test]
fn switching_the_panel_off_takes_it_back_off_the_screen() {
    let mut app = ready("debug-toggle-off");
    app.world_mut().insert_resource(DebugOverlay(true));
    app.update();
    assert!(frame_text(&app).contains("frame"));

    app.world_mut().insert_resource(DebugOverlay(false));
    app.update();
    assert!(
        !frame_text(&app).contains("frame"),
        "the panel stayed up after being switched off"
    );
}

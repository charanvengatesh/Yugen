//! The pause card exists, says what it does, and does not hide the world.

use yugen_core::config::View;
use yugen_render::ui::layout::Chrome;
use yugen_render::ui::{UiPrim, pause_at};

fn texts(prims: &[UiPrim]) -> Vec<String> {
    prims
        .iter()
        .filter_map(|p| match p {
            UiPrim::Text { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn the_pause_card_names_itself_and_the_key_that_dismisses_it() {
    let view = View::for_screen(1280, 720);
    let lines = texts(&pause_at(Chrome::of(view), view)).join(" | ");
    assert!(lines.contains("Paused"), "{lines}");
    assert!(
        lines.contains("Esc"),
        "the card must say how to leave: {lines}"
    );
    // The hint stack the HUD fades out has to be here, or fading it there just
    // deleted the controls from the game.
    assert!(lines.contains("craft"), "{lines}");
    assert!(lines.contains("dash"), "{lines}");
}

#[test]
fn the_pause_scrim_does_not_black_out_the_world() {
    let view = View::for_screen(1280, 720);
    let prims = pause_at(Chrome::of(view), view);
    let UiPrim::Rect { color, w, h, .. } = &prims[0] else {
        panic!("the card should open with its scrim");
    };
    assert_eq!((*w, *h), (view.w, view.h), "the scrim is not full-screen");
    let alpha = color.to_srgba().alpha;
    assert!(
        alpha > 0.0 && alpha < 0.8,
        "a pause scrim at {alpha} either does nothing or hides the run"
    );
}

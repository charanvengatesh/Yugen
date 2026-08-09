//! What is on screen, as an ordered stack of layers.
//!
//! # Why this replaces a `match`
//!
//! [`super::compose`] used to be a `match` over [`UiScreen`] with each screen's
//! layout written inline, plus one special case appended after it for the F3
//! panel because that panel belongs to no screen. Adding anything meant editing
//! the `match`, and adding anything that draws OVER an existing screen — a pause
//! card, a tooltip, an inventory — meant editing it in two places and hoping the
//! append order came out right.
//!
//! The screens were never mutually exclusive in the first place. The crafting
//! card already drew over the playing HUD, and the F3 panel already drew over
//! everything including the death card, deliberately: *"the one thing it must
//! never do is be hidden by the state you were trying to diagnose."* A stack is
//! what that already was; this makes it the thing the code says.
//!
//! # Adding a UI element
//!
//! Three edits, none of them to a system signature:
//!
//! 1. A variant on [`Layer`].
//! 2. Its arm in [`Layer::layout`], which is a pure `&PageCx -> Vec<UiPrim>`.
//! 3. A line in [`stack`] saying when it is up, and — because the vector is
//!    ordered back to front — what it draws over.
//!
//! If it needs something to read that [`PageCx`] does not carry, that is a
//! fourth edit, and the only one that touches Bevy. Everything else in this
//! module is pure arithmetic over borrowed data and is tested without an app.
//!
//! # Why an enum and not `Box<dyn Page>`
//!
//! A trait object would let a layer be registered from outside this crate,
//! which nothing wants, and would cost an allocation per layer per frame in the
//! one system that runs every frame for every screen. The enum is exhaustive,
//! so a new variant is a compile error in [`Layer::layout`] rather than a
//! surface that silently never draws — which is the failure `worldselect`'s
//! header warns about and the reason it argues for a capture test.

use yugen_core::config::View;
use yugen_core::interact::BuildTool;

use super::layout::Chrome;
use super::{IconAtlas, UiPrim, UiScreen};
use crate::items::Pack;
use crate::player::PlayerBody;

/// Everything any layer is allowed to read, borrowed for one frame.
///
/// One struct rather than an argument list per layer, so that adding a page
/// that needs the world clock does not change the signature of the seven pages
/// that do not. It borrows rather than owns: `compose` builds it from resources
/// it already holds, and no layer may mutate anything through it — a layout
/// function that could write is a layout function that can disagree with the
/// frame it is describing.
pub struct PageCx<'a> {
    /// The buffer this frame is being laid out for.
    pub view: View,
    /// Where the standing chrome has claimed space. See [`Chrome`].
    pub chrome: Chrome,
    /// The body, for health and dash. Absent until a world exists.
    pub body: Option<&'a PlayerBody>,
    /// The build tool, for the palette and the cursor feedback.
    pub tool: &'a BuildTool,
    /// The pack, for the hotbar.
    pub pack: &'a Pack,
    /// Where item art comes from.
    pub icons: &'a dyn IconAtlas,
    /// The transient line above the hotbar, and its alpha.
    pub toast: Option<(&'a str, f32)>,
    /// The crafting card's state.
    pub crafting: &'a crate::craftscreen::CraftingView,
    /// The world list.
    pub picker: &'a crate::worldselect::WorldPicker,
    /// XP the mob system has banked.
    pub xp: i32,
    /// The menu stack: which page, and where the cursor is.
    pub nav: &'a super::menu::Nav,
    /// What the options pages read and write.
    pub settings: &'a crate::settings::Settings,
    /// Whether a world exists behind the menu. Changes the shape of a page,
    /// never a value on one.
    pub in_game: bool,
    /// The saved worlds, for `Page::Worlds`.
    pub worlds: &'a [super::menu::WorldRow],
    /// Which world is one press from deletion, if any.
    pub confirming: Option<usize>,
    /// How brightly the health bar is flashing, `0.0..=1.0`.
    pub flash: f32,
    /// How opaque the control hints are, after the player's preference and the
    /// run's age have both had their say.
    pub hints: f32,
    /// Seconds since this run began, for chrome that fades out.
    pub run_age_s: f32,
    /// What the F3 panel would say. Gathered in `PreUpdate`, so it is this
    /// frame's.
    pub debug: &'a crate::debug::DebugReadout,
}

/// One drawable surface.
///
/// Ordered back to front by [`stack`], not by this enum's declaration order —
/// a variant's position here means nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Health, dash, and the stat row.
    Vitals,
    /// The bottom-left hotbar or creative palette, and the cursor note.
    Panel,
    /// The control hints, top right.
    Hints,
    /// The transient message line.
    Toast,
    /// The crafting card, over the world.
    Craft,
    /// The menu system: title, pause, and every options page. See
    /// [`super::menu`].
    Menu,
    /// The world list.
    WorldSelect,
    /// The death card.
    GameOver,
    /// The F3 panel, over everything.
    Debug,
}

impl Layer {
    /// This layer's prims, back to front within itself.
    pub fn layout(self, cx: &PageCx) -> Vec<UiPrim> {
        match self {
            Layer::Vitals => super::vitals(cx),
            Layer::Panel => super::build_hud(cx.tool, cx.pack, cx.icons, cx.view),
            Layer::Hints => super::hints(cx),
            Layer::Toast => match cx.toast {
                Some((text, alpha)) => super::toast(text, alpha, cx.view),
                None => Vec::new(),
            },
            Layer::Craft => crate::craftscreen::screen(cx.crafting, cx.view),
            Layer::Menu => {
                let rows = super::menu::page_rows(
                    cx.nav.top(),
                    &super::menu::MenuCx {
                        settings: cx.settings,
                        in_game: cx.in_game,
                        worlds: cx.worlds,
                        confirming: cx.confirming,
                    },
                );
                super::menu::screen(cx.nav.top(), &rows, cx.nav.focus(), cx.in_game, cx.view)
            }
            Layer::WorldSelect => crate::worldselect::screen(cx.picker, cx.view),
            Layer::GameOver => super::game_over(cx.view),
            Layer::Debug => crate::debug::overlay(cx.debug, cx.chrome),
        }
    }
}

/// The transient state that decides which optional layers are up.
///
/// A struct rather than four `bool` arguments, because four `bool`s at a call
/// site is four chances to transpose two of them and no way for the compiler to
/// notice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overlays {
    /// The crafting card is open.
    pub crafting: bool,
    /// The game is paused. See `scenes::Paused` for why this is a resource and
    /// not a `Scene` variant.
    pub paused: bool,
    /// The F3 panel is up.
    pub debug: bool,
    /// The control hints have not yet faded out.
    pub hints: bool,
}

/// The stack for the current state, back to front.
///
/// Written into `out` rather than returned so the caller can keep one buffer
/// across frames; `compose` runs every frame for every screen, and a `Vec` per
/// frame is a `Vec` per frame.
///
/// Everything is passed in rather than read from resources, because this
/// function IS the decision and it should be answerable in a test with no app.
pub fn stack(screen: UiScreen, on: Overlays, out: &mut Vec<Layer>) {
    out.clear();
    match screen {
        UiScreen::Menu => out.push(Layer::Menu),
        UiScreen::WorldSelect => out.push(Layer::WorldSelect),
        UiScreen::GameOver => out.push(Layer::GameOver),
        UiScreen::Playing => {
            // A menu REPLACES the HUD rather than covering it. The world keeps
            // drawing underneath — that is what makes pause a layer and not a
            // scene — but the health bar, the hotbar, the hints and any toast
            // go, because none of them is answering a question the player has
            // while a menu is open, and a toast fading out over an options row
            // is just two things written on top of each other.
            if on.paused {
                out.push(Layer::Menu);
            } else {
                out.extend([Layer::Vitals, Layer::Panel, Layer::Toast]);
                if on.hints {
                    out.push(Layer::Hints);
                }
                // Over the HUD, because it is a card the player opened and the
                // hotbar underneath it is not what they are looking at.
                if on.crafting {
                    out.push(Layer::Craft);
                }
            }
        }
    }

    // Last, and outside the `match`, because the panel is an instrument rather
    // than part of any screen: it is as useful over the death card as over the
    // world, and the one thing it must never do is be hidden by the state you
    // were trying to diagnose.
    if on.debug {
        out.push(Layer::Debug);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stacked(screen: UiScreen, on: Overlays) -> Vec<Layer> {
        let mut out = Vec::new();
        stack(screen, on, &mut out);
        out
    }

    #[test]
    fn a_card_screen_draws_one_card_and_no_hud() {
        for screen in [UiScreen::Menu, UiScreen::WorldSelect, UiScreen::GameOver] {
            let layers = stacked(screen, Overlays::default());
            assert_eq!(layers.len(), 1, "{screen:?} drew more than its card");
            assert!(!layers.contains(&Layer::Vitals), "{screen:?} drew the HUD");
        }
    }

    #[test]
    fn the_instrument_panel_is_on_top_of_every_screen_there_is() {
        // The property `debug`'s header argues for: it must never be hidden by
        // the state you were trying to diagnose.
        let on = Overlays {
            debug: true,
            crafting: true,
            paused: true,
            hints: true,
        };
        for screen in [
            UiScreen::Menu,
            UiScreen::WorldSelect,
            UiScreen::GameOver,
            UiScreen::Playing,
        ] {
            let layers = stacked(screen, on);
            assert_eq!(
                layers.last(),
                Some(&Layer::Debug),
                "{screen:?} drew something over the F3 panel"
            );
        }
    }

    #[test]
    fn a_menu_replaces_the_hud_but_not_the_world() {
        // The world is still drawn — that is the whole reason pause is a layer
        // and not a `Scene`, and `scenes::running` gates the stepping rather
        // than the drawing. What goes is the HUD: nothing on it answers a
        // question the player has with a menu open.
        let layers = stacked(
            UiScreen::Playing,
            Overlays {
                paused: true,
                hints: true,
                crafting: true,
                ..Overlays::default()
            },
        );
        assert_eq!(layers, vec![Layer::Menu], "{layers:?}");
    }

    #[test]
    fn the_hud_comes_back_when_the_menu_closes() {
        let layers = stacked(
            UiScreen::Playing,
            Overlays {
                hints: true,
                crafting: true,
                ..Overlays::default()
            },
        );
        for want in [Layer::Vitals, Layer::Panel, Layer::Hints, Layer::Craft] {
            assert!(layers.contains(&want), "{want:?} did not come back");
        }
        assert!(!layers.contains(&Layer::Menu));
    }

    #[test]
    fn faded_hints_leave_the_stack_entirely() {
        // Not drawn at zero alpha: a layer that contributes nothing should not
        // be costing the painter a pass over its prims every frame.
        let layers = stacked(UiScreen::Playing, Overlays::default());
        assert!(!layers.contains(&Layer::Hints));
    }

    #[test]
    fn nothing_but_playing_draws_the_pause_card() {
        // Pause is reachable only from a live world. `UiScreen::Menu` draws the
        // menu system anyway — it IS the title card — so the interesting cases
        // are the two screens that own their own input.
        for screen in [UiScreen::WorldSelect, UiScreen::GameOver] {
            let layers = stacked(
                screen,
                Overlays {
                    paused: true,
                    ..Overlays::default()
                },
            );
            assert!(
                !layers.contains(&Layer::Menu),
                "{screen:?} paused over its own card"
            );
        }
    }
}

//! Input as the simulation sees it: a per-step [`Intent`] and the binding table
//! that produces one.
//!
//! # What survived the port, and what did not
//!
//! The TypeScript `Input` class was a DOM object — `window.addEventListener`,
//! `KeyboardEvent.code`, a canvas bounding rect, a wheel accumulator. None of
//! that belongs in `yugen-core`, which carries no windowing, so the class
//! itself is gone. What it EXISTED to produce is here:
//!
//! - [`Intent`], the flat per-step struct the player reads. In the TypeScript
//!   this was declared in `Player.ts` and filled by `Player.readIntent`; it is
//!   input plumbing, not player behaviour, and it lives here so that `Player`
//!   never has to import a binding table.
//! - [`KEYS`], the semantic bindings, verbatim, as data.
//! - [`KeyState`], the two questions the old class answered — "is this held?"
//!   and "was this pressed this frame?". A host implements it over whatever
//!   keyboard it has; the mapping from Bevy's `ButtonInput<KeyCode>` onto it
//!   lives in the render crate, where Bevy is allowed to exist.
//!
//! The mouse half of the old class (canvas-space cursor, buttons, wheel steps)
//! went with the DOM. It served the build tool and the hotbar, both of which are
//! host-side concerns; what the simulation needs from a cursor is the world-space
//! aim point, which is two floats on [`Intent`].
//!
//! # Held vs just-pressed
//!
//! Keeping both is the whole point of the type. Jump and dash are EDGE
//! triggered — `jump_queued` is true on the frame the key went down and never
//! again while it is held, so leaning on the key does not pogo and does not
//! spend a dash a frame. Jump also has a held reading (`jump_held`) because
//! releasing early cuts the jump short, and a third reading (`up`) because the
//! same physical binding climbs a ladder. Collapsing any of those into one bool
//! breaks a mechanic.

/// Semantic key bindings, as `KeyboardEvent.code` names.
///
/// The code strings are kept exactly as the TypeScript had them even though
/// nothing here is a browser: they are the stable, layout-independent names for
/// physical keys, every host toolkit has a mapping to them, and keeping them
/// means the binding table is one table rather than one per backend.
pub struct KeyBindings {
    pub left: &'static [&'static str],
    pub right: &'static [&'static str],
    pub jump: &'static [&'static str],
    pub down: &'static [&'static str],
    pub dash: &'static [&'static str],
    pub punch: &'static [&'static str],
    pub debug: &'static [&'static str],
    pub confirm: &'static [&'static str],

    // --- Items -------------------------------------------------------------
    /// Hotbar slots, INDEX-ORDERED: `hotbar[i]` selects slot i. Digit0 is last
    /// because slot 10 lives under the `0` key, which is where the tenth key
    /// physically is on the row — not under a Digit10 that does not exist.
    pub hotbar: &'static [&'static str; 10],
    /// Consume the held consumable.
    pub use_item: &'static [&'static str],
    /// Craft the next affordable recipe; repeat to step through the set.
    pub craft: &'static [&'static str],
    /// Toggle the creative/debug palette. G for god, which is the whole game.
    pub creative: &'static [&'static str],
    /// Creative brush size. The wheel is spoken for by the hotbar now.
    pub brush_down: &'static [&'static str],
    pub brush_up: &'static [&'static str],
    /// Held: the brush acts on the BACKGROUND WALL plane. See
    /// [`Cursor::back`](crate::interact::Cursor::back).
    pub background: &'static [&'static str],
}

/// Semantic key bindings.
///
/// `use` is a Rust keyword, so the TypeScript `KEYS.use` is [`KeyBindings::use_item`];
/// every other field is the snake_case of its TypeScript name.
pub static KEYS: KeyBindings = KeyBindings {
    left: &["ArrowLeft", "KeyA"],
    right: &["ArrowRight", "KeyD"],
    jump: &["ArrowUp", "KeyW", "Space"],
    down: &["ArrowDown", "KeyS"],
    dash: &["ShiftLeft", "ShiftRight", "KeyK"],
    punch: &["KeyP"],
    debug: &["Backquote"],
    confirm: &["Enter", "Space"],

    hotbar: &[
        "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6", "Digit7", "Digit8", "Digit9",
        "Digit0",
    ],
    use_item: &["KeyF"],
    craft: &["KeyC"],
    creative: &["KeyG"],
    brush_down: &["BracketLeft", "Minus"],
    brush_up: &["BracketRight", "Equal"],
    // A modifier plus a letter, the shape `dash` uses. Shift is already dash and
    // Control is claimed by the window manager on every platform this runs on, so
    // Alt is the modifier and `B` for "background" is the one-handed alternative.
    background: &["AltLeft", "AltRight", "KeyB"],
};

/// The two questions the old `Input` class answered about the keyboard.
///
/// `is_down` is the held state; `was_pressed` is the one-shot rising edge for
/// this frame, which the host clears at the end of each rendered frame (the old
/// `clearFrame`). Implement this over whatever keyboard the host has and
/// [`Intent::from_keys`] does the rest.
pub trait KeyState {
    /// Is this key code currently held?
    fn is_down(&self, code: &str) -> bool;

    /// Did this key code go down this frame? Does not consume.
    fn was_pressed(&self, code: &str) -> bool;

    /// Any of the given codes currently held.
    fn any_down(&self, codes: &[&str]) -> bool {
        codes.iter().any(|c| self.is_down(c))
    }

    /// Any of the given codes pressed this frame.
    fn any_pressed(&self, codes: &[&str]) -> bool {
        codes.iter().any(|c| self.was_pressed(c))
    }
}

/// Per-step intent, sampled once per frame from the host's key state.
///
/// `Default` is "no input at all", which is also what a headless test wants.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Intent {
    /// -1 left, +1 right, 0 neither or both. The TypeScript carried `left` and
    /// `right` as separate bools and every consumer immediately computed
    /// `(right ? 1 : 0) - (left ? 1 : 0)`; that subtraction happens once, here.
    pub dir_x: f32,
    /// Rising edge this frame. Edge triggered so holding the key does not pogo.
    pub jump_queued: bool,
    /// Held; releasing early cuts the jump short.
    pub jump_held: bool,
    /// Rising edge this frame.
    pub dash_queued: bool,
    /// Rising edge this frame (starts a swing / looses a shot).
    pub punch_queued: bool,
    /// Held; auto-repeats attacks at the weapon's cadence.
    pub punch_held: bool,
    /// World-space aim point, when the host has one (the mouse). Ranged attacks
    /// fire toward it; without it they fire along the facing, which is what a
    /// keyboard-only session gets and is deliberately still playable.
    ///
    /// `(0, 0)` is the "no aim point" sentinel, exactly as the TypeScript's
    /// `intent.aimX ?? 0` produced. The world origin is deep in ungenerated sky
    /// and nothing ever legitimately aims at it.
    pub aim_x: f32,
    /// See [`Intent::aim_x`].
    pub aim_y: f32,
    /// Held; climbs up. Distinct from jump — see [`Intent::from_keys`].
    pub up: bool,
    /// Held; dives while swimming, climbs down, drops through platforms.
    pub down: bool,
}

impl Intent {
    /// Read frame-level input into a per-step Intent.
    ///
    /// `up` is the jump keys read as a HELD state, and that is not a mistake: the
    /// game binds up/W/Space to jump, so a ladder has to be climbed with the same
    /// keys or it needs a binding the player has never been taught. The two never
    /// collide because they are consumed in different modes — on a ladder, held-up
    /// climbs and a fresh PRESS (`jump_queued`) lets go; off one, held-up is
    /// ignored and only the press does anything.
    ///
    /// This is `Player.readIntent` from the TypeScript. It moved off `Player`
    /// because it is pure binding-table plumbing and the entity has no business
    /// knowing what a key code is.
    pub fn from_keys<K: KeyState + ?Sized>(keys: &K) -> Intent {
        let left = keys.any_down(KEYS.left);
        let right = keys.any_down(KEYS.right);
        Intent {
            dir_x: f32::from(right) - f32::from(left),
            jump_held: keys.any_down(KEYS.jump),
            jump_queued: keys.any_pressed(KEYS.jump),
            dash_queued: keys.any_pressed(KEYS.dash),
            punch_queued: keys.any_pressed(KEYS.punch),
            punch_held: keys.any_down(KEYS.punch),
            down: keys.any_down(KEYS.down),
            up: keys.any_down(KEYS.jump),
            aim_x: 0.0,
            aim_y: 0.0,
        }
    }

    /// The same intent as seen by substep `n` of a fixed-timestep frame.
    ///
    /// One rendered frame can drive several physics substeps, and a rising edge
    /// belongs to exactly one of them. Without this the queued flags stay true
    /// for every substep and a single tap of jump or dash fires up to
    /// `MAX_STEPS_PER_FRAME` times. Held flags are untouched — they describe a
    /// state, not an event.
    #[inline]
    pub fn for_substep(self, n: u32) -> Intent {
        Intent {
            jump_queued: self.jump_queued && n == 0,
            dash_queued: self.dash_queued && n == 0,
            punch_queued: self.punch_queued && n == 0,
            ..self
        }
    }

    /// Does the host have a cursor aimed somewhere? `(0, 0)` means it does not.
    #[inline]
    pub fn has_aim(self) -> bool {
        self.aim_x != 0.0 || self.aim_y != 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// A keyboard made of two sets, exactly like the class this replaced.
    #[derive(Default)]
    struct FakeKeys {
        held: HashSet<&'static str>,
        pressed: HashSet<&'static str>,
    }

    impl FakeKeys {
        /// A fresh press: down this frame, therefore also held.
        fn press(&mut self, code: &'static str) {
            self.held.insert(code);
            self.pressed.insert(code);
        }

        /// Held from an earlier frame: the edge is gone, the state remains.
        fn hold(&mut self, code: &'static str) {
            self.held.insert(code);
        }
    }

    impl KeyState for FakeKeys {
        fn is_down(&self, code: &str) -> bool {
            self.held.contains(code)
        }
        fn was_pressed(&self, code: &str) -> bool {
            self.pressed.contains(code)
        }
    }

    #[test]
    fn no_input_is_the_default_intent() {
        let k = FakeKeys::default();
        assert_eq!(Intent::from_keys(&k), Intent::default());
    }

    #[test]
    fn direction_is_the_difference_of_the_two_bindings() {
        let mut k = FakeKeys::default();
        k.hold("KeyD");
        assert_eq!(Intent::from_keys(&k).dir_x, 1.0);

        k.hold("KeyA");
        assert_eq!(
            Intent::from_keys(&k).dir_x,
            0.0,
            "both keys cancel, as the TypeScript subtraction did"
        );

        let mut k = FakeKeys::default();
        k.hold("ArrowLeft");
        assert_eq!(Intent::from_keys(&k).dir_x, -1.0);
    }

    #[test]
    fn jump_distinguishes_the_edge_from_the_hold() {
        let mut fresh = FakeKeys::default();
        fresh.press("Space");
        let i = Intent::from_keys(&fresh);
        assert!(i.jump_queued && i.jump_held);
        assert!(i.up, "the jump binding read as held is also `up`");

        let mut held = FakeKeys::default();
        held.hold("Space");
        let i = Intent::from_keys(&held);
        assert!(!i.jump_queued, "no edge on a key held from last frame");
        assert!(i.jump_held && i.up);
    }

    #[test]
    fn dash_and_punch_are_edge_triggered_too() {
        let mut k = FakeKeys::default();
        k.press("ShiftLeft");
        k.hold("KeyP");
        let i = Intent::from_keys(&k);
        assert!(i.dash_queued);
        assert!(i.punch_held);
        assert!(!i.punch_queued, "held from an earlier frame is not an edge");
    }

    #[test]
    fn a_rising_edge_belongs_to_exactly_one_substep() {
        let mut k = FakeKeys::default();
        k.press("Space");
        k.press("KeyK");
        let i = Intent::from_keys(&k);

        let first = i.for_substep(0);
        assert!(first.jump_queued && first.dash_queued);

        for n in 1..5 {
            let later = i.for_substep(n);
            assert!(!later.jump_queued, "substep {n} would double-jump");
            assert!(!later.dash_queued, "substep {n} would spend a second dash");
            assert!(later.jump_held, "held state is not an event");
        }
    }

    #[test]
    fn no_cursor_reads_as_no_aim() {
        let i = Intent::default();
        assert!(!i.has_aim());
        assert!(
            Intent {
                aim_x: 12.0,
                ..Intent::default()
            }
            .has_aim()
        );
    }

    #[test]
    fn the_hotbar_table_is_index_ordered_with_zero_last() {
        assert_eq!(KEYS.hotbar.len(), 10);
        assert_eq!(KEYS.hotbar[0], "Digit1");
        assert_eq!(KEYS.hotbar[9], "Digit0");
    }
}

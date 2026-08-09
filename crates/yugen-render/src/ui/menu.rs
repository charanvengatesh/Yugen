//! Menu screens: a navigation stack, four widgets, and the driving.
//!
//! # What this is
//!
//! The title card, the pause card and the options screens are one system
//! because they are one interaction: a stack of pages made of rows you move a
//! cursor through, where some rows go somewhere and some change a value.
//! Minecraft's menus have worked this way since Beta and it is worth copying
//! for the reason it survived — the player learns the controls once.
//!
//! # The stack is the whole design
//!
//! [`Nav`] is a `Vec<Page>` and Escape pops it. That single fact is what makes
//! Options reachable from the title card AND from the pause card without either
//! of them knowing about the other, and what makes "back" mean the right thing
//! in both. The alternative — a `previous: Page` field — is the same thing with
//! a depth of one, and it breaks the first time a sub-screen gets a sub-screen,
//! which Video already is.
//!
//! The stack never empties: [`Nav::pop`] on a lone page is ignored, because the
//! bottom of the stack is the screen the game is on and popping it would leave
//! nothing drawn and nothing to press.
//!
//! # Widgets are data, not objects
//!
//! A [`Row`] is a label plus a [`Control`], and a page is a `Vec<Row>` rebuilt
//! every frame from [`Settings`]. Nothing is retained, nothing is registered,
//! there is no callback and no `Box<dyn>`: pressing a row returns an [`Action`]
//! and the caller does it. That keeps the whole of the layout and the whole of
//! the input logic pure — [`page_rows`] is a function from settings to rows, and
//! every test below calls it directly with no app, no window and no GPU.
//!
//! # What is deliberately not here
//!
//! **A text field.** `worldselect`'s header already argues this: naming a world
//! means a caret, key repeat, an IME and a clipboard, and none of that is worth
//! it for a value the seed picker generates.
//!
//! **Key rebinding.** `KEYS` is a `&'static` table in `yugen-core`, which is
//! what makes it greppable and testable. Making it editable at runtime means
//! making it a resource, which every one of its readers would have to be
//! rewritten for. The Controls page lists the bindings and says what they are,
//! which is what a player opens it for nine times in ten.

use bevy::prelude::*;

use super::layout::Region;
use super::{Align, UiPrim, theme};
use crate::settings::{HintMode, RenderScale, Settings};
use yugen_core::config::View;

/// Which menu page is being shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    /// The title card.
    Title,
    /// The pause card, over a live world.
    Pause,
    /// The options index.
    Options,
    /// Everything about how it looks.
    Video,
    /// Everything about the overlay.
    Interface,
    /// Everything about the simulation.
    World,
    /// The binding table, read-only. See the module header.
    Controls,
    /// The world list: pick one, make one, delete one.
    ///
    /// This was `worldselect`'s own screen with its own keys and its own
    /// layout, and it was the last thing in the game that did not look or
    /// behave like the rest of the menus — from the title card's
    /// "Singleplayer" it read as a dead end, because Enter on an empty list
    /// does nothing and the only way forward was a letter printed in the
    /// footer. `worldselect` still owns the DIRECTORY: listing, creating and
    /// deleting are real file operations and they stay where they were.
    Worlds,
}

impl Page {
    /// The heading at the top of the card.
    pub fn title(self) -> &'static str {
        match self {
            Page::Title => "Yūgen",
            Page::Pause => "Paused",
            Page::Options => "Options",
            Page::Video => "Video",
            Page::Interface => "Interface",
            Page::World => "World",
            Page::Controls => "Controls",
            Page::Worlds => "Worlds",
        }
    }
}

/// The page stack. The last entry is what is on screen.
///
/// A resource rather than a field on a screen, because the pause card and the
/// title card both push onto the same stack and neither owns it.
#[derive(Resource, Clone, Debug)]
pub struct Nav {
    stack: Vec<Page>,
    /// Which row is focused, per depth, so going back restores the cursor.
    ///
    /// Not a nicety: coming out of Video to find the cursor on "Video" is the
    /// difference between adjusting two settings and hunting for your place
    /// twice.
    focus: Vec<usize>,
}

impl Default for Nav {
    fn default() -> Nav {
        Nav {
            stack: vec![Page::Title],
            focus: vec![0],
        }
    }
}

impl Nav {
    /// A stack showing exactly `page`.
    pub fn rooted(page: Page) -> Nav {
        Nav {
            stack: vec![page],
            focus: vec![0],
        }
    }

    /// What is on screen.
    pub fn top(&self) -> Page {
        *self.stack.last().expect("the stack is never empty")
    }

    /// Which row is focused on the current page.
    pub fn focus(&self) -> usize {
        *self.focus.last().expect("the stack is never empty")
    }

    /// How deep the stack is. One means the root.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Show `page`, remembering where the cursor was.
    pub fn push(&mut self, page: Page) {
        self.stack.push(page);
        self.focus.push(0);
    }

    /// Go back one, or do nothing at the root.
    ///
    /// Returns whether it moved, so the caller can tell "went back" from
    /// "Escape at the root", which on the pause card means resume.
    pub fn pop(&mut self) -> bool {
        if self.stack.len() <= 1 {
            return false;
        }
        self.stack.pop();
        self.focus.pop();
        true
    }

    /// Replace the whole stack.
    pub fn reset(&mut self, page: Page) {
        self.stack.clear();
        self.focus.clear();
        self.stack.push(page);
        self.focus.push(0);
    }

    /// Move the cursor by `delta`, wrapping, over `len` rows.
    pub fn step(&mut self, delta: i32, len: usize) {
        if len == 0 {
            return;
        }
        let n = len as i32;
        let f = self.focus.last_mut().expect("the stack is never empty");
        *f = (((*f as i32 + delta) % n + n) % n) as usize;
    }

    /// Put the cursor on `row`, for the mouse.
    pub fn focus_on(&mut self, row: usize) {
        *self.focus.last_mut().expect("the stack is never empty") = row;
    }
}

/// What a row does.
#[derive(Clone, Debug, PartialEq)]
pub enum Control {
    /// Go somewhere, or do something. Carries what.
    Press(Action),
    /// On or off.
    Toggle(bool),
    /// One of a short list, stepped through in place.
    Cycle {
        /// What the current choice is called.
        value: String,
        /// Which setting to step. See [`Action::Step`].
        of: Step,
    },
    /// A number between two ends.
    Slider {
        /// Where the handle is, `0.0..=1.0`.
        at: f32,
        /// What the current value is called.
        value: String,
        /// Which setting to drag.
        of: Slide,
    },
    /// Not interactive: a heading, or a binding being listed.
    Note(String),
}

impl Control {
    /// Can the cursor land on this?
    pub fn selectable(&self) -> bool {
        !matches!(self, Control::Note(_))
    }
}

/// What pressing a row does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Push a page.
    Open(Page),
    /// Pop one.
    Back,
    /// Make a world and land the cursor on it.
    NewWorld,
    /// Enter the world at this index of the list.
    PlayWorld(usize),
    /// Delete the world at this index. Only ever reached from a row that has
    /// already asked once — see [`Page::Worlds`].
    DeleteWorld(usize),
    /// Resume a paused world.
    Resume,
    /// Save and go back to the title card.
    Quit,
    /// Close the game.
    Exit,
    /// Flip a boolean setting.
    Flip(Flip),
    /// Step a cycled setting one place.
    Step(Step),
}

/// A boolean setting a row can flip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flip {
    Fullscreen,
    Vsync,
    DebugOverlay,
}

/// A cycled setting a row can step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Scale,
    Hints,
}

/// A continuous setting a row can drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slide {
    Particles,
    Shake,
    DayMinutes,
    Autosave,
}

impl Slide {
    /// The value's range.
    fn range(self) -> (f32, f32) {
        match self {
            Slide::Particles | Slide::Shake => (0.0, 1.0),
            Slide::DayMinutes => Settings::DAY_MINUTES,
            Slide::Autosave => Settings::AUTOSAVE_S,
        }
    }

    /// Read it out of `s`, as a fraction of its range.
    fn at(self, s: &Settings) -> f32 {
        let (lo, hi) = self.range();
        let v = match self {
            Slide::Particles => s.particles,
            Slide::Shake => s.shake,
            Slide::DayMinutes => s.day_minutes,
            Slide::Autosave => s.autosave_s,
        };
        ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
    }

    /// Write it into `s` from a fraction of its range.
    pub fn set(self, s: &mut Settings, at: f32) {
        let (lo, hi) = self.range();
        let v = lo + (hi - lo) * at.clamp(0.0, 1.0);
        match self {
            Slide::Particles => s.particles = v,
            Slide::Shake => s.shake = v,
            Slide::DayMinutes => s.day_minutes = v,
            Slide::Autosave => s.autosave_s = v,
        }
        s.clamp();
    }

    /// How far one key press moves it.
    ///
    /// A twentieth of the range, so twenty presses cross it — few enough to be
    /// quick and many enough to land on a value you meant.
    pub const KEY_STEP: f32 = 0.05;

    /// What the row prints beside the bar.
    fn value(self, s: &Settings) -> String {
        match self {
            Slide::Particles => percent(s.particles),
            Slide::Shake => {
                if s.shake <= 0.0 {
                    "Off".into()
                } else {
                    percent(s.shake)
                }
            }
            Slide::DayMinutes => format!("{:.0} min", s.day_minutes),
            Slide::Autosave => format!("{:.0}s", s.autosave_s),
        }
    }
}

/// `0.0..=1.0` as a percentage, or `Off` at nothing.
fn percent(v: f32) -> String {
    if v <= 0.0 {
        "Off".into()
    } else {
        format!("{:.0}%", v * 100.0)
    }
}

/// One line of a page.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// What it is called.
    pub label: String,
    /// What it does.
    pub control: Control,
    /// A dim value drawn on the right of a row that is also a button — a
    /// world's seed. `Control` is one thing or the other, and a world row
    /// needs both.
    pub aside: Option<String>,
}

impl Row {
    /// This row's text, kept, but pressable.
    ///
    /// A world row wants both: a value on the right (its seed) and an action
    /// when pressed. `Control` is one or the other, so the seed rides along in
    /// [`Row::note`] and this swaps the control for a press — which is why
    /// [`Row::aside`] exists rather than a sixth `Control` variant.
    fn into_press(self, action: Action) -> Row {
        let aside = match self.control {
            Control::Note(text) => Some(text),
            _ => None,
        };
        Row {
            label: self.label,
            control: Control::Press(action),
            aside,
        }
    }

    fn press(label: &str, action: Action) -> Row {
        Row {
            label: label.into(),
            control: Control::Press(action),
            aside: None,
        }
    }

    fn note(label: &str, text: &str) -> Row {
        Row {
            label: label.into(),
            control: Control::Note(text.into()),
            aside: None,
        }
    }

    fn toggle(label: &str, on: bool) -> Row {
        Row {
            label: label.into(),
            control: Control::Toggle(on),
            aside: None,
        }
    }

    fn cycle(label: &str, value: impl Into<String>, of: Step) -> Row {
        Row {
            label: label.into(),
            control: Control::Cycle {
                value: value.into(),
                of,
            },
            aside: None,
        }
    }

    fn slider(label: &str, of: Slide, s: &Settings) -> Row {
        Row {
            label: label.into(),
            control: Control::Slider {
                at: of.at(s),
                value: of.value(s),
                of,
            },
            aside: None,
        }
    }
}

/// One world, as the list needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldRow {
    /// What it is called.
    pub name: String,
    /// Its seed, shown beside the name — the one fact that tells two worlds
    /// with the same auto-generated name apart.
    pub seed: u32,
}

/// The world list in the shape [`page_rows`] wants it.
///
/// Mirrored from `worldselect::WorldPicker` by a system rather than built in
/// `compose`, which runs every frame for every screen and must not allocate a
/// `String` per world for a page that is usually not open.
#[derive(Resource, Clone, Debug, Default)]
pub struct WorldRows(pub Vec<WorldRow>);

/// Everything [`page_rows`] reads.
///
/// A struct rather than three arguments, so that a page needing a fourth thing
/// does not change the signature of the six that do not.
pub struct MenuCx<'a> {
    /// What the options pages show and edit.
    pub settings: &'a Settings,
    /// Whether a world exists behind the menu.
    pub in_game: bool,
    /// The saved worlds, newest first.
    pub worlds: &'a [WorldRow],
    /// Which world is one more press from being deleted, if any.
    pub confirming: Option<usize>,
}

/// The rows of `page`, given the current settings.
///
/// Pure, and the seam every test in this module goes through. `in_game` is
/// whether a world exists behind the menu, which is the only thing that changes
/// the SHAPE of a page rather than a value on it: the title card offers
/// "Singleplayer" and the pause card offers "Back to Game", and Options is the
/// same page from either.
pub fn page_rows(page: Page, cx: &MenuCx) -> Vec<Row> {
    let s = cx.settings;
    let in_game = cx.in_game;
    match page {
        Page::Title => vec![
            Row::press("Singleplayer", Action::Open(Page::Worlds)),
            Row::press("Options...", Action::Open(Page::Options)),
            Row::press("Quit Game", Action::Exit),
        ],
        Page::Pause => vec![
            Row::press("Back to Game", Action::Resume),
            Row::press("Options...", Action::Open(Page::Options)),
            Row::press("Save and Quit to Title", Action::Quit),
        ],
        Page::Options => vec![
            Row::press("Video...", Action::Open(Page::Video)),
            Row::press("Interface...", Action::Open(Page::Interface)),
            Row::press("World...", Action::Open(Page::World)),
            Row::press("Controls...", Action::Open(Page::Controls)),
            Row::press("Done", Action::Back),
        ],
        Page::Video => vec![
            Row::toggle("Fullscreen", s.fullscreen),
            Row::toggle("VSync", s.vsync),
            Row::cycle("Render Scale", s.scale.label(), Step::Scale),
            Row::slider("Particles", Slide::Particles, s),
            Row::slider("Screen Shake", Slide::Shake, s),
            Row::press("Done", Action::Back),
        ],
        Page::Interface => vec![
            Row::cycle("Control Hints", s.hints.label(), Step::Hints),
            Row::toggle("Debug Overlay", s.debug_overlay),
            Row::press("Done", Action::Back),
        ],
        Page::World => {
            let mut rows = vec![
                Row::slider("Day Length", Slide::DayMinutes, s),
                Row::slider("Autosave", Slide::Autosave, s),
            ];
            // Only true while there is a world to have been generated.
            if in_game {
                rows.push(Row::note("Seed", "shown on the F3 panel"));
            }
            rows.push(Row::press("Done", Action::Back));
            rows
        }
        Page::Worlds => {
            let mut rows: Vec<Row> = cx
                .worlds
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    if cx.confirming == Some(i) {
                        // The two-step delete, as a row rather than as a footer
                        // prompt. `worldselect`'s header is right that this is
                        // the only irreversible thing a player can do from a
                        // menu, so it keeps both steps and says which world.
                        Row {
                            label: format!("Delete {}?", w.name),
                            control: Control::Press(Action::DeleteWorld(i)),
                            aside: Some("X again".into()),
                        }
                    } else {
                        Row {
                            label: w.name.clone(),
                            control: Control::Note(format!("seed {}", w.seed)),
                            aside: None,
                        }
                        .into_press(Action::PlayWorld(i))
                    }
                })
                .collect();
            if rows.is_empty() {
                rows.push(Row::note("No worlds yet", "make one below"));
            }
            rows.push(Row::press("Create New World", Action::NewWorld));
            rows.push(Row::press("Done", Action::Back));
            rows
        }
        Page::Controls => {
            let mut rows: Vec<Row> = CONTROLS
                .iter()
                .map(|(what, keys)| Row::note(what, keys))
                .collect();
            rows.push(Row::press("Done", Action::Back));
            rows
        }
    }
}

/// The bindings, as the Controls page lists them.
///
/// Transcribed from `KEYS` rather than read from it, and that is a real cost:
/// the two can drift. `the_controls_page_lists_every_binding` is what stops it,
/// by counting them against the table. Reading `KEYS` directly would print
/// `KeyW` and `ArrowLeft` at the player, which is what the table stores and not
/// what anybody calls those keys.
const CONTROLS: &[(&str, &str)] = &[
    ("Move", "\u{2190} / \u{2192}"),
    ("Jump", "\u{2191}"),
    ("Descend", "\u{2193}"),
    ("Dash", "Shift"),
    ("Punch", "Z"),
    ("Dig", "Left Mouse"),
    ("Place", "Right Mouse"),
    ("Hotbar", "1 - 0, Wheel"),
    ("Use Item", "F"),
    ("Craft", "C"),
    ("Creative", "G"),
    ("Background Layer", "Alt / B"),
    ("Confirm", "Enter / Space"),
    ("Pause", "Esc"),
    ("Debug Panel", "F3 / `"),
];

/// The first row the cursor may land on, skipping any [`Control::Note`]s.
pub fn first_selectable(rows: &[Row]) -> usize {
    rows.iter()
        .position(|r| r.control.selectable())
        .unwrap_or(0)
}

/// The row `focus` should become after moving `delta`, skipping notes.
///
/// Walks rather than jumps, so a page that is all notes but for its "Done"
/// still lands on "Done" from either direction.
pub fn step_focus(rows: &[Row], focus: usize, delta: i32) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let n = rows.len() as i32;
    let mut at = focus as i32;
    for _ in 0..rows.len() {
        at = ((at + delta) % n + n) % n;
        if rows[at as usize].control.selectable() {
            return at as usize;
        }
    }
    focus
}

/// What a click or a press on `row` should do.
pub fn activate(row: &Row) -> Option<Action> {
    match &row.control {
        Control::Press(a) => Some(*a),
        Control::Toggle(_) => match row.label.as_str() {
            "Fullscreen" => Some(Action::Flip(Flip::Fullscreen)),
            "VSync" => Some(Action::Flip(Flip::Vsync)),
            "Debug Overlay" => Some(Action::Flip(Flip::DebugOverlay)),
            _ => None,
        },
        Control::Cycle { of, .. } => Some(Action::Step(*of)),
        Control::Slider { .. } | Control::Note(_) => None,
    }
}

/// Apply `action`'s effect on the settings, if it has one.
///
/// The navigation and world-state actions are the caller's business; this is
/// only the part that is a pure function of [`Settings`], which is the part
/// worth testing without an app.
pub fn apply_to_settings(action: Action, s: &mut Settings) -> bool {
    match action {
        Action::Flip(Flip::Fullscreen) => s.fullscreen = !s.fullscreen,
        Action::Flip(Flip::Vsync) => s.vsync = !s.vsync,
        Action::Flip(Flip::DebugOverlay) => s.debug_overlay = !s.debug_overlay,
        Action::Step(Step::Scale) => {
            let i = RenderScale::ALL
                .iter()
                .position(|v| *v == s.scale)
                .unwrap_or(0);
            s.scale = RenderScale::ALL[(i + 1) % RenderScale::ALL.len()];
        }
        Action::Step(Step::Hints) => {
            let i = HintMode::ALL
                .iter()
                .position(|v| *v == s.hints)
                .unwrap_or(0);
            s.hints = HintMode::ALL[(i + 1) % HintMode::ALL.len()];
        }
        _ => return false,
    }
    true
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// Width of a menu row, in buffer px.
const ROW_W: i32 = 240;

/// Height of a menu row.
const ROW_H: i32 = 18;

/// Blank px between one row and the next.
const ROW_GAP: i32 = 4;

/// Blank px between the heading's descender and the first row.
const HEAD_GAP: i32 = 12;

/// The style a page's heading is set in.
///
/// The game's name gets the display size; every other page is a heading.
fn head_style(page: Page) -> super::TextStyle {
    if page == Page::Title {
        theme::CARD_TITLE
    } else {
        theme::CARD_BODY
    }
}

/// Buffer px reserved above the first row for the heading.
///
/// Derived from the face rather than fixed, and that is the second time this
/// has mattered: at a constant 58 the descender of the `g` in "Yūgen" — set at
/// scale 6, so eighteen buffer pixels below its baseline — ran eight pixels
/// into the "Singleplayer" button. Departure Mono HAS descenders and the
/// authored face it replaced did not, so anything measured from a cap height
/// is measuring the wrong thing. `cell_h` is the whole glyph box.
fn head_h(page: Page) -> i32 {
    head_style(page).cell_h() + HEAD_GAP
}

/// Width of the bar in a slider row, in buffer px.
const SLIDER_W: i32 = 84;

/// Height of that bar.
const SLIDER_H: i32 = 5;

/// How many rows fit on this buffer at once.
///
/// Computed rather than fixed, because the buffer is a function of the window
/// and the Controls page is sixteen rows. At 640x360 — the buffer a 1280x720
/// window produces — a fixed count large enough for Controls would run the
/// short pages off the bottom, and one small enough for the short pages would
/// hide half the bindings. `worldselect` reached the same conclusion and this
/// is its `VISIBLE_ROWS` made dynamic.
pub fn visible_rows(view: View, page: Page) -> usize {
    let room = view.h - head_h(page) - 2 * theme::MARGIN;
    ((room / (ROW_H + ROW_GAP)).max(1)) as usize
}

/// The window of rows on screen: the first one, and how many.
///
/// Scrolls by keeping the focused row in view rather than by drawing a
/// scrollbar, which is `worldselect`'s argument too — a scrollbar is a widget
/// and this module has rectangles and text runs.
pub fn visible_window(view: View, page: Page, len: usize, focus: usize) -> (usize, usize) {
    let count = visible_rows(view, page).min(len);
    if count == 0 {
        return (0, 0);
    }
    // Far enough down that `focus` is the last visible row, then clamped so the
    // window never runs off the end of the list.
    let first = focus
        .saturating_sub(count - 1)
        .min(len.saturating_sub(count));
    (first, count)
}

/// Where visible slot `slot` of `count` sits, for a given buffer.
///
/// Public because the mouse needs it: hit-testing a row means asking the layout
/// where it drew that row, and a second copy of this arithmetic in the input
/// system is exactly how a menu ends up with a button that highlights one row
/// and activates another.
pub fn slot_rect(view: View, page: Page, count: usize, slot: usize) -> Region {
    let block = count as i32 * (ROW_H + ROW_GAP) - ROW_GAP;
    let head = head_h(page);
    let top = ((view.h - block + head) / 2).max(head);
    Region {
        x: (view.w - ROW_W.min(view.w - 2 * theme::MARGIN)) / 2,
        y: top + slot as i32 * (ROW_H + ROW_GAP),
        w: ROW_W.min(view.w - 2 * theme::MARGIN),
        h: ROW_H,
    }
}

/// The row under `(x, y)`, if the pointer is over one.
///
/// Takes the same `focus` the layout was drawn with, so the window it tests
/// against is the window that is on screen.
pub fn row_at(view: View, page: Page, len: usize, focus: usize, x: i32, y: i32) -> Option<usize> {
    let (first, count) = visible_window(view, page, len, focus);
    (0..count).find_map(|slot| {
        let r = slot_rect(view, page, count, slot);
        (x >= r.x && x < r.right() && y >= r.y && y < r.bottom()).then_some(first + slot)
    })
}

/// The slider bar inside `row`.
fn slider_rect(row: Region) -> Region {
    Region {
        x: row.right() - SLIDER_W - theme::PAD,
        y: row.cy() - SLIDER_H / 2,
        w: SLIDER_W,
        h: SLIDER_H,
    }
}

/// Where a click at `x` inside `row` puts a slider handle, `0.0..=1.0`.
pub fn slider_at(row: Region, x: i32) -> f32 {
    let bar = slider_rect(row);
    if bar.w <= 0 {
        return 0.0;
    }
    ((x - bar.x) as f32 / bar.w as f32).clamp(0.0, 1.0)
}

/// A whole menu page, as a display list.
///
/// Pure: same page, settings and focus in, same primitives out.
pub fn screen(page: Page, rows: &[Row], focus: usize, over_world: bool, view: View) -> Vec<UiPrim> {
    let mut out = Vec::with_capacity(rows.len() * 4 + 8);

    // A dimmer scrim over a live world than over nothing: the world is what the
    // player is coming back to. See `ui::pause_at`.
    out.push(UiPrim::rect(
        0,
        0,
        view.w,
        view.h,
        if over_world {
            theme::PLATE
        } else {
            theme::SCRIM
        },
    ));

    let (first, count) = visible_window(view, page, rows.len(), focus);
    let cx = view.w / 2;
    let head_y = if count > 0 {
        slot_rect(view, page, count, 0).y - head_h(page)
    } else {
        view.h / 2
    };

    // One card behind the heading and the rows.
    //
    // The scrim alone is not enough: a `Note` row has no plate of its own — it
    // is not a button and must not look like one — and eleven-pixel grey over a
    // snowfield seen through a 60% scrim is at the edge of legible. The card
    // gives every page one surface, which is also what makes a scrolling list
    // read as a list rather than as text floating in front of the game.
    if count > 0 {
        let first_r = slot_rect(view, page, count, 0);
        let last_r = slot_rect(view, page, count, count - 1);
        let top = first_r.y - head_h(page) + theme::UNIT;
        out.push(UiPrim::rect(
            first_r.x - theme::PAD,
            top,
            first_r.w + 2 * theme::PAD,
            last_r.bottom() - top + theme::PAD,
            theme::PLATE,
        ));
    }

    let style = head_style(page);
    out.push(UiPrim::text(
        page.title(),
        cx,
        style.baseline_from_top(head_y),
        Align::Centre,
        style,
        theme::INK,
    ));

    for slot in 0..count {
        let i = first + slot;
        let r = slot_rect(view, page, count, slot);
        let picked = i == focus && rows[i].control.selectable();
        out.extend(row_prims(&rows[i], r, picked));
    }

    // Say what is being hidden rather than hiding it silently — `worldselect`
    // makes the same argument for the same reason.
    if count < rows.len() {
        let last = slot_rect(view, page, count, count - 1);
        out.push(UiPrim::text(
            format!("{} of {}", first + count, rows.len()),
            last.right(),
            theme::CAPTION.baseline_from_top(last.bottom() + theme::UNIT),
            Align::Right,
            theme::CAPTION,
            theme::INK_MUTED,
        ));
    }
    out
}

/// One row.
fn row_prims(row: &Row, r: Region, picked: bool) -> Vec<UiPrim> {
    let mut out = Vec::with_capacity(5);
    let note = matches!(row.control, Control::Note(_));

    if !note {
        // The plate IS the button, as in every menu this is modelled on. The
        // selected one is lighter and carries a hairline, so selection is two
        // signals and neither of them is only hue — see `ui`'s selected slot.
        out.push(UiPrim::rect(
            r.x,
            r.y,
            r.w,
            r.h,
            if picked { theme::WELL } else { theme::PLATE },
        ));
        if picked {
            super::frame(&mut out, r.x, r.y, r.w, r.h, 1, theme::EDGE);
        }
    }

    let ink = if note {
        theme::INK_DIM
    } else if picked {
        theme::INK
    } else {
        theme::INK_FAINT
    };
    out.push(UiPrim::text(
        row.label.clone(),
        r.x + theme::PAD,
        theme::BODY.baseline_from_middle(r.cy()),
        Align::Left,
        theme::BODY,
        ink,
    ));

    if let Some(aside) = &row.aside {
        out.push(UiPrim::text(
            aside.clone(),
            r.right() - theme::PAD,
            theme::CAPTION.baseline_from_middle(r.cy()),
            Align::Right,
            theme::CAPTION,
            theme::INK_MUTED,
        ));
    }

    match &row.control {
        Control::Press(_) => {}
        Control::Note(text) => {
            out.push(UiPrim::text(
                text.clone(),
                r.right() - theme::PAD,
                theme::CAPTION.baseline_from_middle(r.cy()),
                Align::Right,
                theme::CAPTION,
                theme::INK_MUTED,
            ));
        }
        Control::Toggle(on) => {
            out.push(UiPrim::text(
                if *on { "On" } else { "Off" },
                r.right() - theme::PAD,
                theme::BODY.baseline_from_middle(r.cy()),
                Align::Right,
                theme::BODY,
                if *on { theme::ACCENT } else { theme::INK_MUTED },
            ));
        }
        Control::Cycle { value, .. } => {
            out.push(UiPrim::text(
                value.clone(),
                r.right() - theme::PAD,
                theme::BODY.baseline_from_middle(r.cy()),
                Align::Right,
                theme::BODY,
                theme::ACCENT,
            ));
        }
        Control::Slider { at, value, .. } => {
            let bar = slider_rect(r);
            out.push(UiPrim::rect(bar.x, bar.y, bar.w, bar.h, theme::PLATE));
            out.push(UiPrim::rect(
                bar.x,
                bar.y,
                super::js_round(bar.w as f32 * at.clamp(0.0, 1.0)),
                bar.h,
                if picked {
                    theme::ACCENT
                } else {
                    theme::ACCENT_SPENT
                },
            ));
            // The reading, left of the bar: a slider with no number on it is a
            // slider the player has to guess at.
            out.push(UiPrim::text(
                value.clone(),
                bar.x - theme::PAD,
                theme::CAPTION.baseline_from_middle(r.cy()),
                Align::Right,
                theme::CAPTION,
                theme::INK_DIM,
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        View::for_screen(1280, 720)
    }

    fn worlds() -> &'static [WorldRow] {
        static WORLDS: std::sync::OnceLock<Vec<WorldRow>> = std::sync::OnceLock::new();
        WORLDS.get_or_init(|| {
            vec![
                WorldRow {
                    name: "World 1".into(),
                    seed: 7,
                },
                WorldRow {
                    name: "World 2".into(),
                    seed: 8,
                },
            ]
        })
    }

    fn rows_with(page: Page, s: &Settings, in_game: bool) -> Vec<Row> {
        page_rows(
            page,
            &MenuCx {
                settings: s,
                in_game,
                worlds: worlds(),
                confirming: None,
            },
        )
    }

    fn rows(page: Page) -> Vec<Row> {
        rows_with(page, &Settings::default(), false)
    }

    /// Every page, so a new one cannot be added without the sweeps seeing it.
    const PAGES: [Page; 8] = [
        Page::Title,
        Page::Pause,
        Page::Options,
        Page::Video,
        Page::Interface,
        Page::World,
        Page::Controls,
        Page::Worlds,
    ];

    // --- The stack ----------------------------------------------------------

    #[test]
    fn options_returns_to_whichever_card_opened_it() {
        // The whole reason this is a stack. Both cards push the same page and
        // neither knows about the other.
        for root in [Page::Title, Page::Pause] {
            let mut nav = Nav::rooted(root);
            nav.push(Page::Options);
            nav.push(Page::Video);
            assert_eq!(nav.top(), Page::Video);
            assert!(nav.pop());
            assert_eq!(nav.top(), Page::Options);
            assert!(nav.pop());
            assert_eq!(nav.top(), root, "came back to the wrong card");
        }
    }

    #[test]
    fn the_stack_never_empties() {
        // Popping the root would leave nothing drawn and nothing to press.
        let mut nav = Nav::rooted(Page::Title);
        assert!(!nav.pop(), "the root was popped");
        assert_eq!(nav.top(), Page::Title);
        assert_eq!(nav.depth(), 1);
    }

    #[test]
    fn going_back_restores_the_cursor_rather_than_resetting_it() {
        let mut nav = Nav::rooted(Page::Options);
        nav.step(2, 5);
        assert_eq!(nav.focus(), 2);
        nav.push(Page::Video);
        assert_eq!(nav.focus(), 0, "a new page starts at the top");
        nav.pop();
        assert_eq!(nav.focus(), 2, "the cursor was lost going back");
    }

    // --- Focus --------------------------------------------------------------

    #[test]
    fn the_cursor_wraps_at_both_ends() {
        let r = rows(Page::Title);
        assert_eq!(step_focus(&r, 0, -1), r.len() - 1);
        assert_eq!(step_focus(&r, r.len() - 1, 1), 0);
    }

    #[test]
    fn the_cursor_skips_rows_that_do_nothing() {
        // The Controls page is fifteen notes and a Done. The cursor must land
        // on Done from both directions and never on a binding.
        let r = rows(Page::Controls);
        let done = r.len() - 1;
        assert!(matches!(r[0].control, Control::Note(_)));
        assert_eq!(first_selectable(&r), done);
        assert_eq!(step_focus(&r, done, 1), done, "wrapped onto a note");
        assert_eq!(step_focus(&r, done, -1), done, "wrapped onto a note");
    }

    // --- Actions ------------------------------------------------------------

    #[test]
    fn every_selectable_row_on_every_page_does_something() {
        // The failure this catches is a row that draws, highlights, accepts a
        // press and has no effect — the exact shape of menu furniture.
        for page in PAGES {
            for row in rows_with(page, &Settings::default(), true) {
                if !row.control.selectable() {
                    continue;
                }
                if matches!(row.control, Control::Slider { .. }) {
                    continue; // Sliders drag; they have no press.
                }
                assert!(
                    activate(&row).is_some(),
                    "{page:?} / {:?} is a button that does nothing",
                    row.label
                );
            }
        }
    }

    #[test]
    fn every_page_has_a_way_out() {
        // Every page reached FROM another one. The two ROOTS — Title and Pause
        // — are excluded on purpose: there is nowhere above them to go, which
        // `Nav::pop` says by returning false. Escape also pops, but a visible
        // way out is the discoverable one.
        for page in PAGES {
            if matches!(page, Page::Title | Page::Pause) {
                continue;
            }
            let r = rows_with(page, &Settings::default(), true);
            assert!(
                r.iter().any(|row| activate(row) == Some(Action::Back)),
                "{page:?} has no way back"
            );
        }
    }

    #[test]
    fn the_world_list_offers_a_way_to_make_one_even_when_it_is_empty() {
        // The dead end this page exists to remove: "Singleplayer" used to lead
        // to a list where Enter did nothing and the only way forward was a
        // letter printed in the footer.
        let s = Settings::default();
        let empty = page_rows(
            Page::Worlds,
            &MenuCx {
                settings: &s,
                in_game: false,
                worlds: &[],
                confirming: None,
            },
        );
        assert!(
            empty.iter().any(|r| activate(r) == Some(Action::NewWorld)),
            "an empty world list cannot make a world: {empty:?}"
        );
        assert!(
            empty.iter().any(|r| r.control.selectable()),
            "an empty world list has nothing the cursor can land on"
        );
    }

    #[test]
    fn each_world_row_plays_its_own_world() {
        // An off-by-one here would open somebody else's save.
        let r = rows(Page::Worlds);
        for (i, _) in worlds().iter().enumerate() {
            assert_eq!(activate(&r[i]), Some(Action::PlayWorld(i)), "row {i}");
            assert_eq!(r[i].label, worlds()[i].name);
        }
    }

    #[test]
    fn a_world_asks_before_it_is_deleted() {
        let s = Settings::default();
        // Unconfirmed, the row plays. Confirmed, and only that row deletes.
        let plain = rows(Page::Worlds);
        assert_eq!(activate(&plain[1]), Some(Action::PlayWorld(1)));

        let asked = page_rows(
            Page::Worlds,
            &MenuCx {
                settings: &s,
                in_game: false,
                worlds: worlds(),
                confirming: Some(1),
            },
        );
        assert_eq!(activate(&asked[1]), Some(Action::DeleteWorld(1)));
        assert!(asked[1].label.contains(&worlds()[1].name), "{:?}", asked[1]);
        assert_eq!(
            activate(&asked[0]),
            Some(Action::PlayWorld(0)),
            "asking about one world armed another"
        );
    }

    #[test]
    fn cycling_a_setting_returns_to_where_it_started() {
        let mut s = Settings::default();
        let first = s.scale;
        for _ in 0..RenderScale::ALL.len() {
            apply_to_settings(Action::Step(Step::Scale), &mut s);
        }
        assert_eq!(s.scale, first, "the scale cycle does not close");

        let first = s.hints;
        for _ in 0..HintMode::ALL.len() {
            apply_to_settings(Action::Step(Step::Hints), &mut s);
        }
        assert_eq!(s.hints, first, "the hint cycle does not close");
    }

    #[test]
    fn a_toggle_row_flips_the_setting_it_names() {
        let mut s = Settings::default();
        for (label, read) in [
            (
                "Fullscreen",
                (|s: &Settings| s.fullscreen) as fn(&Settings) -> bool,
            ),
            ("VSync", |s: &Settings| s.vsync),
            ("Debug Overlay", |s: &Settings| s.debug_overlay),
        ] {
            let page = if label == "Debug Overlay" {
                Page::Interface
            } else {
                Page::Video
            };
            let rows = rows_with(page, &s, false);
            let row = rows.iter().find(|r| r.label == label).expect(label);
            let before = read(&s);
            let action = activate(row).expect(label);
            assert!(apply_to_settings(action, &mut s), "{label} did nothing");
            assert_ne!(read(&s), before, "{label} did not flip");
        }
    }

    #[test]
    fn a_slider_reaches_both_ends_of_its_range_and_no_further() {
        let mut s = Settings::default();
        for slide in [
            Slide::Particles,
            Slide::Shake,
            Slide::DayMinutes,
            Slide::Autosave,
        ] {
            let (lo, hi) = slide.range();
            slide.set(&mut s, -1.0);
            assert!((slide.at(&s) - 0.0).abs() < 1e-6, "{slide:?} under-ran");
            slide.set(&mut s, 2.0);
            assert!((slide.at(&s) - 1.0).abs() < 1e-6, "{slide:?} over-ran");
            slide.set(&mut s, 0.5);
            let mid = lo + (hi - lo) * 0.5;
            let got = match slide {
                Slide::Particles => s.particles,
                Slide::Shake => s.shake,
                Slide::DayMinutes => s.day_minutes,
                Slide::Autosave => s.autosave_s,
            };
            assert!((got - mid).abs() < 1e-3, "{slide:?} midpoint is {got}");
        }
    }

    #[test]
    fn a_slider_at_zero_says_off_rather_than_nought_percent() {
        let mut s = Settings::default();
        Slide::Shake.set(&mut s, 0.0);
        assert_eq!(Slide::Shake.value(&s), "Off");
    }

    // --- Layout -------------------------------------------------------------

    #[test]
    fn no_two_rows_overlap_and_all_of_them_are_on_screen() {
        for v in [
            view(),
            View::for_screen(800, 600),
            View::for_screen(2560, 1440),
        ] {
            for page in [Page::Title, Page::Video, Page::Controls] {
                let r = rows(page);
                let (_, count) = visible_window(v, page, r.len(), r.len() - 1);
                let rects: Vec<Region> = (0..count)
                    .map(|slot| slot_rect(v, page, count, slot))
                    .collect();
                for (i, a) in rects.iter().enumerate() {
                    assert!(a.x >= 0 && a.y >= 0, "{page:?} slot {i} starts off screen");
                    assert!(
                        a.right() <= v.w && a.bottom() <= v.h,
                        "{page:?} slot {i} runs off a {}x{} buffer",
                        v.w,
                        v.h
                    );
                    for b in &rects[i + 1..] {
                        assert!(!a.overlaps(*b), "{page:?} rows overlap");
                    }
                }
            }
        }
    }

    #[test]
    fn a_long_page_scrolls_to_keep_the_focused_row_in_view() {
        // The Controls page is sixteen rows and does not fit a 640x360 buffer.
        let v = view();
        let len = rows(Page::Controls).len();
        assert!(
            visible_rows(v, Page::Controls) < len,
            "the test page now fits; pick a longer one"
        );
        for focus in 0..len {
            let (first, count) = visible_window(v, Page::Controls, len, focus);
            assert!(
                (first..first + count).contains(&focus),
                "focus {focus} is scrolled off screen"
            );
            assert!(first + count <= len, "the window ran past the end");
        }
    }

    #[test]
    fn the_pointer_finds_the_row_that_was_drawn_under_it() {
        // The failure this catches is a menu that highlights one row and
        // activates another, which is what a second copy of the layout
        // arithmetic in the input system always produces eventually.
        let v = view();
        let len = rows(Page::Controls).len();
        let focus = len - 1;
        let (first, count) = visible_window(v, Page::Controls, len, focus);
        for slot in 0..count {
            let r = slot_rect(v, Page::Controls, count, slot);
            assert_eq!(
                row_at(v, Page::Controls, len, focus, r.cx(), r.cy()),
                Some(first + slot)
            );
        }
        // And a click in the gutter between two rows hits neither.
        let r = slot_rect(v, Page::Controls, count, 0);
        assert_eq!(
            row_at(v, Page::Controls, len, focus, r.cx(), r.bottom()),
            None
        );
        assert_eq!(row_at(v, Page::Controls, len, focus, r.x - 4, r.cy()), None);
    }

    #[test]
    fn clicking_the_ends_of_a_slider_bar_reaches_the_ends_of_its_range() {
        let r = slot_rect(view(), Page::Video, 6, 3);
        let bar = slider_rect(r);
        assert_eq!(slider_at(r, bar.x), 0.0);
        assert_eq!(slider_at(r, bar.right()), 1.0);
        // And outside the bar clamps rather than extrapolating.
        assert_eq!(slider_at(r, bar.x - 500), 0.0);
        assert_eq!(slider_at(r, bar.right() + 500), 1.0);
    }

    #[test]
    fn the_controls_page_lists_every_binding() {
        // The transcription guard. `CONTROLS` is written out by hand so the
        // player reads "Shift" and not "ShiftLeft"; this counts it against the
        // table so a binding added to `KEYS` cannot go unlisted.
        use yugen_core::input::KEYS;
        let listed = CONTROLS.len();
        let bound = [
            KEYS.left,
            KEYS.jump,
            KEYS.down,
            KEYS.dash,
            KEYS.punch,
            KEYS.confirm,
            KEYS.use_item,
            KEYS.craft,
            KEYS.creative,
            KEYS.background,
            KEYS.debug,
            KEYS.pause,
        ]
        .len();
        // Twelve keyboard bindings, plus dig, place and the hotbar, which are
        // the mouse and the digit row.
        assert_eq!(listed, bound + 3, "the Controls page has drifted from KEYS");
    }

    #[test]
    fn no_menu_sets_a_character_the_face_cannot_draw() {
        // The same guard `ui` has, for the pages this module owns — the arrows
        // in the Controls list are the obvious hazard.
        let v = view();
        let s = Settings::default();
        for page in PAGES {
            let r = rows_with(page, &s, true);
            for prim in screen(page, &r, 0, true, v) {
                let UiPrim::Text { text, style, .. } = prim else {
                    continue;
                };
                for ch in text.chars() {
                    assert!(
                        style.face.chars().any(|c| c == ch),
                        "{page:?} sets {ch:?} (U+{:04X}), which {:?} cannot draw",
                        ch as u32,
                        style.face
                    );
                }
            }
        }
    }
}

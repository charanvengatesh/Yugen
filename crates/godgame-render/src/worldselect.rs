//! The world-select screen: pick a save, make one, delete one.
//!
//! # Why this is its own module
//!
//! Everything a save needs already exists — `godgame_core::sim::save` creates,
//! lists and deletes worlds, and `WorldSave` tells the world builder which one
//! to grow. What was missing is the only part a player can see: a way to have
//! more than one, without a command-line flag.
//!
//! It follows `crate::debug`'s shape rather than growing inside `crate::ui`,
//! and for the same reason. [`WorldPicker`] is plain data, [`screen`] is a pure
//! function from that data to [`UiPrim`]s, and the systems are the only part
//! that touches the world. So the layout — which is most of the code, and all of
//! the fiddly part — is tested headlessly with no GPU and no filesystem.
//!
//! # What it deliberately does not have
//!
//! **A text field.** Naming a world means a caret, a cursor, key repeat,
//! clipboard, IME, and a decision about what to do with a name that is all
//! spaces. That is a widget, not a screen, and `crate::ui` has no widgets — it
//! has rectangles and text runs. New worlds are named [`AUTO_NAME`] and a number
//! until somebody wants to build one; `save::create_world` already takes any
//! name and `slug_of` already survives whatever a field would produce.
//!
//! **A seed field**, for the same reason, and one more: a seed a player typed is
//! a thing they will want to share, which means it should be shown somewhere
//! after the fact too. `--seed` covers the case that exists today.

use bevy::prelude::*;
use godgame_core::config::{SEED, View};
use godgame_core::input::{KEYS, KeyState};
use godgame_core::sim::save::{WorldMeta, create_world, delete_world, list_worlds};

use crate::input::BevyKeys;
use crate::scenes::Scene;
use crate::ui::{Align, TextStyle, UiPrim, rgb, rgba};
use crate::world::WorldSave;

/// What a new world is called, before the number.
const AUTO_NAME: &str = "World";

/// Rows of the list drawn at once.
///
/// Eight fills the plate without crowding the hints under it. More than this
/// wants a scrollbar, and a scrollbar is a widget — see the module header. The
/// list scrolls by keeping the cursor in view instead, which needs no chrome.
const VISIBLE_ROWS: usize = 8;

// --- State -------------------------------------------------------------------

/// The screen's whole state. Plain data; no Bevy in the layout path.
#[derive(Resource, Clone, Debug, Default)]
pub struct WorldPicker {
    /// Where worlds live. Empty until the host sets it.
    pub root: std::path::PathBuf,
    /// Worlds, newest-played first — `list_worlds`' order.
    pub worlds: Vec<WorldMeta>,
    /// Which row is selected.
    pub cursor: usize,
    /// Whether the selected world is one more keypress from being deleted.
    ///
    /// A two-step delete, and not a nicety: this is the only irreversible thing
    /// a player can do from a menu, the key is next to the one that plays a
    /// world, and there is no undo behind it.
    pub confirming: bool,
    /// The last thing that went wrong, shown until something else happens.
    pub error: Option<String>,
}

impl WorldPicker {
    /// Re-read the directory, keeping the cursor on something real.
    pub fn refresh(&mut self) {
        self.worlds = list_worlds(&self.root);
        self.cursor = self.cursor.min(self.worlds.len().saturating_sub(1));
        self.confirming = false;
    }

    /// The selected world, if there is one.
    pub fn selected(&self) -> Option<&WorldMeta> {
        self.worlds.get(self.cursor)
    }

    /// First row drawn, so the cursor is always on screen.
    fn scroll(&self) -> usize {
        self.cursor.saturating_sub(VISIBLE_ROWS - 1)
    }
}

// --- Layout ------------------------------------------------------------------

const TITLE: TextStyle = TextStyle::for_px(24);
const ROW: TextStyle = TextStyle::for_px(13);
const HINT: TextStyle = TextStyle::for_px(11);
/// Height of one list row.
const ROW_H: i32 = 22;

/// The screen, as a display list. Pure.
pub fn screen(picker: &WorldPicker, view: View) -> Vec<UiPrim> {
    let mut out = Vec::with_capacity(VISIBLE_ROWS * 3 + 8);
    let (w, h) = (view.w, view.h);

    // The card. Dimming the whole buffer first is what `ui`'s menu and death
    // screens both do, and doing it differently here would read as a different
    // kind of screen rather than another one of them.
    out.push(UiPrim::rect(0, 0, w, h, rgba(0, 0, 0, 0.72)));
    out.push(UiPrim::text(
        "Worlds",
        w / 2,
        56,
        Align::Centre,
        TITLE,
        rgb(0xff, 0xff, 0xff),
    ));

    let list_w = (w - 96).min(460);
    let x = (w - list_w) / 2;
    let top = 84;

    if picker.worlds.is_empty() {
        out.push(UiPrim::text(
            "No worlds yet.",
            w / 2,
            top + ROW_H,
            Align::Centre,
            ROW,
            rgb(0xb4, 0xbe, 0xc8),
        ));
    }

    let from = picker.scroll();
    for (i, world) in picker
        .worlds
        .iter()
        .enumerate()
        .skip(from)
        .take(VISIBLE_ROWS)
    {
        let y = top + (i - from) as i32 * ROW_H;
        let picked = i == picker.cursor;
        if picked {
            out.push(UiPrim::rect(
                x,
                y,
                list_w,
                ROW_H - 2,
                rgba(0x50, 0x78, 0xa0, 0.85),
            ));
        }
        let ink = if picked {
            rgb(0xff, 0xff, 0xff)
        } else {
            rgb(0xc8, 0xd2, 0xdc)
        };
        out.push(UiPrim::text(
            world.name.clone(),
            x + 10,
            ROW.baseline_from_middle(y + ROW_H / 2 - 1),
            Align::Left,
            ROW,
            ink,
        ));
        // The seed on the same row, right-aligned. It is the one fact about a
        // world worth reading at a glance and the one a player might want to
        // write down.
        out.push(UiPrim::text(
            format!("seed {}", world.seed),
            x + list_w - 10,
            ROW.baseline_from_middle(y + ROW_H / 2 - 1),
            Align::Right,
            HINT,
            if picked {
                rgb(0xdc, 0xe6, 0xf0)
            } else {
                rgb(0x8c, 0x96, 0xa0)
            },
        ));
    }

    // More below than fits: say so, rather than letting the list look complete.
    if picker.worlds.len() > from + VISIBLE_ROWS {
        out.push(UiPrim::text(
            format!("+{} more", picker.worlds.len() - from - VISIBLE_ROWS),
            w / 2,
            top + VISIBLE_ROWS as i32 * ROW_H + 12,
            Align::Centre,
            HINT,
            rgb(0x8c, 0x96, 0xa0),
        ));
    }

    let footer = h - 44;
    let (line, ink) = if let Some(why) = &picker.error {
        (why.clone(), rgb(0xff, 0x8c, 0x78))
    } else if picker.confirming {
        (
            match picker.selected() {
                Some(w) => format!("Delete \"{}\" for good?  Y confirm   Esc cancel", w.name),
                None => "Nothing to delete.".to_string(),
            },
            rgb(0xff, 0xc8, 0x64),
        )
    } else {
        (
            "Enter play    N new world    X delete    Esc back".to_string(),
            rgb(0xb4, 0xbe, 0xc8),
        )
    };
    out.push(UiPrim::text(line, w / 2, footer, Align::Centre, HINT, ink));
    out
}

// --- Systems -----------------------------------------------------------------

/// Read the directory on the way in, so a world made last session is there.
fn refresh_on_enter(mut picker: ResMut<WorldPicker>) {
    picker.error = None;
    picker.refresh();
}

/// Drive the screen.
///
/// One system rather than one per key. They all mutate the same two fields and
/// the order between them matters — confirming a delete and pressing play are
/// the same physical key in two states — so splitting them would be splitting a
/// state machine across systems that Bevy is free to order either way.
fn keys(
    input: Res<ButtonInput<KeyCode>>,
    mut picker: ResMut<WorldPicker>,
    mut save: ResMut<WorldSave>,
    mut next: ResMut<NextState<Scene>>,
) {
    let bevy_keys = BevyKeys(&input);
    let confirm = bevy_keys.any_pressed(KEYS.confirm);

    if picker.confirming {
        // Only these two keys mean anything while a delete is pending. Anything
        // else is ignored rather than cancelling, so a stray keypress does not
        // quietly leave the player thinking they deleted something.
        if input.just_pressed(KeyCode::KeyY) || confirm {
            let Some(world) = picker.selected().cloned() else {
                picker.confirming = false;
                return;
            };
            match delete_world(picker.root.clone(), &world) {
                Ok(()) => picker.refresh(),
                Err(e) => {
                    picker.error = Some(format!("could not delete: {e}"));
                    picker.confirming = false;
                }
            }
        } else if input.just_pressed(KeyCode::Escape) {
            picker.confirming = false;
        }
        return;
    }

    if input.just_pressed(KeyCode::Escape) {
        next.set(Scene::Menu);
        return;
    }

    let n = picker.worlds.len();
    if n > 0 {
        // `KEYS.jump` is the UP binding — the game has no separate one, and
        // `Intent::from_keys` reads it the same way for climbing a ladder. A
        // menu that wanted its own up key would be teaching a second one.
        if bevy_keys.any_pressed(KEYS.jump) {
            picker.cursor = (picker.cursor + n - 1) % n;
            picker.error = None;
        }
        if bevy_keys.any_pressed(KEYS.down) {
            picker.cursor = (picker.cursor + 1) % n;
            picker.error = None;
        }
        if input.just_pressed(KeyCode::KeyX) {
            picker.confirming = true;
            picker.error = None;
            return;
        }
    }

    if input.just_pressed(KeyCode::KeyN) {
        let name = format!("{AUTO_NAME} {}", picker.worlds.len() + 1);
        match create_world(picker.root.clone(), &name, fresh_seed()) {
            Ok(made) => {
                picker.refresh();
                // Land on what was just made, wherever the recency sort put it.
                picker.cursor = picker
                    .worlds
                    .iter()
                    .position(|w| w.dir == made.dir)
                    .unwrap_or(0);
                picker.error = None;
            }
            Err(e) => picker.error = Some(format!("could not create: {e}")),
        }
        return;
    }

    if confirm && let Some(world) = picker.selected() {
        *save = WorldSave {
            dir: Some(world.dir.clone()),
            seed: world.seed,
        };
        next.set(Scene::Playing);
    }
}

/// A seed for a new world.
///
/// The wall clock, mixed. `godgame_core` has a deterministic RNG and this
/// deliberately does not use it: every stream in this game is seeded FROM a
/// world seed, and asking one of them to invent the next world's seed makes the
/// sequence of worlds a player gets a function of the first. The clock is the
/// only entropy available without a dependency, and one world per nanosecond is
/// not a collision anybody will meet.
fn fresh_seed() -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() ^ d.as_secs() as u32);
    // A round of xorshift, so two worlds made a millisecond apart do not get
    // neighbouring seeds — worldgen's low-frequency fields would give them
    // recognisably similar continents.
    let mut h = nanos ^ SEED;
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    h
}

/// The screen, its state and its keys.
pub struct WorldSelectPlugin;

impl Plugin for WorldSelectPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldPicker>()
            .add_systems(OnEnter(Scene::WorldSelect), refresh_on_enter)
            .add_systems(Update, keys.run_if(in_state(Scene::WorldSelect)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        View::for_screen(1280, 720)
    }

    fn picker(names: &[&str]) -> WorldPicker {
        WorldPicker {
            root: "/tmp/nowhere".into(),
            worlds: names
                .iter()
                .enumerate()
                .map(|(i, n)| WorldMeta {
                    name: (*n).to_string(),
                    seed: 1000 + i as u32,
                    dir: format!("/tmp/nowhere/{n}").into(),
                })
                .collect(),
            cursor: 0,
            confirming: false,
            error: None,
        }
    }

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
    fn an_empty_saves_folder_says_so_rather_than_drawing_nothing() {
        let text = texts(&screen(&picker(&[]), view())).join(" | ");
        assert!(text.contains("No worlds yet"), "{text}");
        assert!(
            text.contains("N new world"),
            "the way out is still offered: {text}"
        );
    }

    #[test]
    fn every_world_shows_its_name_and_its_seed() {
        let text = texts(&screen(&picker(&["Home", "The Deep"]), view())).join(" | ");
        assert!(text.contains("Home") && text.contains("The Deep"), "{text}");
        assert!(
            text.contains("seed 1000") && text.contains("seed 1001"),
            "{text}"
        );
    }

    /// A list longer than the plate scrolls to keep the cursor visible, and says
    /// how much is hidden — a list that looked complete and was not would be
    /// worse than one that scrolled badly.
    #[test]
    fn a_long_list_scrolls_with_the_cursor_and_admits_what_it_is_hiding() {
        let names: Vec<String> = (0..20).map(|i| format!("W{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut p = picker(&refs);

        let top = texts(&screen(&p, view())).join(" | ");
        assert!(top.contains("W0") && top.contains("W7"), "{top}");
        assert!(!top.contains("W8"), "only {VISIBLE_ROWS} rows fit: {top}");
        assert!(top.contains("+12 more"), "{top}");

        p.cursor = 19;
        let bottom = texts(&screen(&p, view())).join(" | ");
        assert!(
            bottom.contains("W19"),
            "the cursor must stay on screen: {bottom}"
        );
        assert!(
            !bottom.contains("| W0 "),
            "and the top scrolled away: {bottom}"
        );
        assert!(
            !bottom.contains("more"),
            "nothing is hidden at the end: {bottom}"
        );
    }

    /// The delete prompt names the world. "Delete this world?" over a list is
    /// how somebody deletes the wrong one.
    #[test]
    fn the_delete_prompt_names_what_it_is_about_to_destroy() {
        let mut p = picker(&["Home", "The Deep"]);
        p.cursor = 1;
        p.confirming = true;
        let text = texts(&screen(&p, view())).join(" | ");
        assert!(text.contains("Delete \"The Deep\" for good?"), "{text}");
        assert!(
            text.contains("Y confirm") && text.contains("Esc cancel"),
            "{text}"
        );
    }

    #[test]
    fn an_error_replaces_the_hints_rather_than_hiding_behind_them() {
        let mut p = picker(&["Home"]);
        p.error = Some("could not delete: permission denied".into());
        let text = texts(&screen(&p, view())).join(" | ");
        assert!(text.contains("permission denied"), "{text}");
        assert!(
            !text.contains("Enter play"),
            "the error takes the line: {text}"
        );
    }

    #[test]
    fn the_screen_stays_inside_the_buffer() {
        let v = view();
        let names: Vec<String> = (0..20).map(|i| format!("World number {i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut p = picker(&refs);
        p.cursor = 12;
        for prim in screen(&p, v) {
            if let UiPrim::Rect { x, y, w, h, .. } = prim {
                assert!(x >= 0 && y >= 0 && x + w <= v.w && y + h <= v.h, "{prim:?}");
            }
            if let UiPrim::Text { baseline, .. } = prim {
                assert!(baseline > 0 && baseline < v.h, "{prim:?}");
            }
        }
    }

    /// Two worlds made back to back must not get neighbouring seeds — worldgen's
    /// low-frequency fields would give them recognisably similar continents.
    #[test]
    fn two_fresh_seeds_are_not_neighbours() {
        let a = fresh_seed();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = fresh_seed();
        assert_ne!(a, b);
        assert!(a.abs_diff(b) > 1000, "{a} and {b} are too close together");
    }
}

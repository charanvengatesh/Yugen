//! The crafting screen: what you can make, what it costs, and why not.
//!
//! # What it replaces
//!
//! `C` used to craft the next affordable recipe and tell you what it made. That
//! is a fine key and a poor interface, and the difference is visible the moment
//! stations exist: given wood and stone, a player who wanted the workbench that
//! unlocks the whole tree gets a LADDER, because a ladder is affordable and
//! comes first in recipe order. Nothing was wrong; the player simply had no way
//! to say which of the affordable things they meant.
//!
//! So `C` opens this instead. It lists every recipe in the game — not only the
//! affordable ones — because the question a player actually has is *"what can I
//! make with this"* followed immediately by *"what would I need"*, and a list
//! that hides everything unaffordable can only answer the first.
//!
//! # Three states, not two
//!
//! A row is craftable, or it is not, and when it is not there are two entirely
//! different reasons: you cannot pay for it, or you are standing in the wrong
//! place. Those send a player to opposite ends of the game — one to a mine, one
//! to a workbench — so they are coloured and labelled differently.
//! `input::try_craft` already makes the same distinction in its toast; this is
//! that distinction with the numbers attached.
//!
//! # Shape
//!
//! `crate::debug`'s and `crate::worldselect`'s: [`CraftingView`] is plain data,
//! [`screen`] is a pure function from it to [`UiPrim`]s, and the systems are the
//! only part that touches the world. The layout is therefore unit-tested with no
//! GPU, no world and no inventory — which is most of the code and all of the
//! part that is fiddly to get right.
//!
//! # Not a Scene
//!
//! An overlay over `Scene::Playing` rather than a state of its own, because the
//! world keeps running behind it: sand keeps falling, creatures keep walking,
//! and a player who opens their recipes has not left the game. A Scene would
//! also mean `OnEnter`/`OnExit` and a second place the HUD decides what to draw.

use bevy::prelude::*;
use yugen_core::config::View;
use yugen_core::input::{KEYS, KeyState};
use yugen_core::items::crafting::{Reach, craft};
use yugen_core::items::registry::{ItemCode, Recipe, Station, item_by_code, recipes};
use yugen_core::items::{HOTBAR, Inventory};

use crate::input::BevyKeys;
use crate::interact_reach::stations_in_reach;
use crate::items::Pack;
use crate::player::PlayerBody;
use crate::scenes::Scene;
use crate::ui::{Align, TextStyle, Toast, UiPrim, rgb, rgba};
use crate::world::SimWorld;

/// Rows the panel is SIZED for.
///
/// Ten. The list is 45 recipes and growing, so it scrolls; ten fills the card
/// without crowding the ingredient line under each row, and the cursor is kept
/// in view rather than a scrollbar being drawn — a scrollbar is a widget and
/// `crate::ui` has rectangles and text runs.
///
/// How many are actually DRAWN is computed from the panel, which is clamped to
/// the buffer. On a short window that is fewer, and the scroll follows it.
const VISIBLE_ROWS: usize = 10;

/// Clearance kept between the panel and the edge of the buffer.
///
/// 40, which puts the panel clear of `ui::hud` at both ends: the health plate
/// and the control hints occupy the top, the hotbar and the held item's two
/// lines the bottom. The first version was a full-screen wash and the capture
/// showed station badges written across the control hints.
const CARD_MARGIN: i32 = 40;

// --- State -------------------------------------------------------------------

/// Why a recipe cannot be made right now, or that it can.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowState {
    /// Everything is here and the station is in reach.
    Ready,
    /// Affordable, but the station is somewhere else.
    WrongPlace,
    /// Not affordable, whatever else is true.
    CannotAfford,
}

/// One recipe, resolved against a pack and a place.
#[derive(Clone, Debug, PartialEq)]
pub struct CraftRow {
    /// What it makes, and how many.
    pub out: String,
    /// How many of the output, for the `x3` suffix.
    pub out_count: u16,
    /// Where it has to be made.
    pub station: Station,
    /// `(name, have, need)` per ingredient.
    pub needs: Vec<(String, u32, u32)>,
    /// Ready, or which of the two nots.
    pub state: RowState,
}

/// The screen's whole state. Plain data.
#[derive(Resource, Clone, Debug, Default)]
pub struct CraftingView {
    /// Whether the card is up.
    pub open: bool,
    /// Every recipe, in registry order.
    pub rows: Vec<CraftRow>,
    /// Selected row.
    pub cursor: usize,
    /// What the last craft attempt said, if anything.
    pub said: Option<String>,
}

impl CraftingView {
    /// First row drawn, so the cursor is always on screen.
    ///
    /// Takes how many rows the panel can actually fit rather than assuming
    /// [`VISIBLE_ROWS`]: the panel is clamped to the buffer, and on a short
    /// window it holds fewer. A scroll computed against a number of rows that
    /// are not drawn puts the cursor off the bottom.
    fn scroll_for(&self, fits: usize) -> usize {
        self.cursor.saturating_sub(fits.saturating_sub(1))
    }
}

/// Resolve every recipe against a pack and a reach.
///
/// A free function over its two inputs rather than a system, so the whole of
/// what the screen SAYS can be tested against an `Inventory` and a `Reach` with
/// no Bevy world at all — the same split `input::try_craft` gets, and for the
/// same reason: this is the part that can be wrong in a way no screenshot shows.
pub fn resolve(inv: &Inventory, reach: Reach) -> Vec<CraftRow> {
    recipes().iter().map(|r| row_of(inv, reach, r)).collect()
}

fn row_of(inv: &Inventory, reach: Reach, r: &Recipe) -> CraftRow {
    let needs: Vec<(String, u32, u32)> = (0..r.in_items.len())
        .map(|i| {
            let code = r.in_items[i];
            let need = u32::from(r.in_counts[i]);
            (name_of(code), inv.count_of(code), need)
        })
        .collect();

    // Affordability is asked of the INGREDIENTS only, so the two reasons stay
    // separate. `can_craft` folds the station in and would collapse them.
    let paid = needs.iter().all(|(_, have, need)| have >= need);
    let state = if !paid {
        RowState::CannotAfford
    } else if !reach.has(r.station) {
        RowState::WrongPlace
    } else {
        RowState::Ready
    };

    CraftRow {
        out: name_of(r.out),
        out_count: r.out_count,
        station: r.station,
        needs,
        state,
    }
}

fn name_of(code: ItemCode) -> String {
    item_by_code(code).name.to_string()
}

/// What a station is called on screen.
///
/// `hand` prints as nothing rather than as "Hand". A row that needs no station
/// should not carry a badge saying so — the badge exists to tell a player where
/// to go, and "go to your hands" is noise on two thirds of the list.
fn station_label(station: Station) -> &'static str {
    match station {
        Station::Hand => "",
        Station::Workbench => "workbench",
        Station::Furnace => "furnace",
        Station::Anvil => "anvil",
    }
}

// --- Layout ------------------------------------------------------------------

const TITLE: TextStyle = TextStyle::for_px(16);
const ROW: TextStyle = TextStyle::for_px(13);
const SMALL: TextStyle = TextStyle::for_px(10);
/// Height of one row: a name line and an ingredient line under it.
const ROW_H: i32 = 26;

/// The screen, as a display list. Pure.
///
/// A bounded PANEL rather than a full-screen wash, and that is not decoration.
/// The first version dimmed the whole buffer and laid rows across it, and the
/// capture showed why it is wrong: the station badges ran into `ui::hud`'s
/// control hints in the top right, and the footer ran into the held item's
/// description at the bottom. The HUD is still there and still readable, and a
/// card that ignores where it is gets written over by it.
pub fn screen(view_state: &CraftingView, view: View) -> Vec<UiPrim> {
    let mut out = Vec::with_capacity(VISIBLE_ROWS * 4 + 8);
    let (w, h) = (view.w, view.h);

    // The whole frame, dimmed. Everything below sits on the panel instead.
    out.push(UiPrim::rect(0, 0, w, h, rgba(0, 0, 0, 0.55)));

    let card_w = (w - 80).min(420);
    let card_h = (VISIBLE_ROWS as i32 * ROW_H + 52).min(h - 2 * CARD_MARGIN);
    let x = (w - card_w) / 2;
    // Clear of the HUD at both ends: `ui::hud`'s health plate and hint block
    // occupy the top, the hotbar and its item lines the bottom.
    let y = ((h - card_h) / 2).max(CARD_MARGIN);

    out.push(UiPrim::rect(
        x,
        y,
        card_w,
        card_h,
        rgba(0x14, 0x18, 0x1e, 0.96),
    ));
    // A hairline, so the panel has an edge against a bright or a dark world
    // rather than only against one of them.
    out.push(UiPrim::rect(x, y, card_w, 1, rgb(0x50, 0x5a, 0x66)));
    out.push(UiPrim::rect(
        x,
        y + card_h - 1,
        card_w,
        1,
        rgb(0x50, 0x5a, 0x66),
    ));

    out.push(UiPrim::text(
        "Crafting",
        x + card_w / 2,
        y + 18,
        Align::Centre,
        TITLE,
        rgb(0xff, 0xff, 0xff),
    ));

    let list_top = y + 28;
    let rows_fit = ((card_h - 52) / ROW_H).max(1) as usize;

    if view_state.rows.is_empty() {
        out.push(UiPrim::text(
            "No recipes.",
            x + card_w / 2,
            list_top + ROW_H,
            Align::Centre,
            ROW,
            rgb(0xb4, 0xbe, 0xc8),
        ));
    }

    let from = view_state.scroll_for(rows_fit);
    for (i, row) in view_state.rows.iter().enumerate().skip(from).take(rows_fit) {
        let ry = list_top + (i - from) as i32 * ROW_H;
        let picked = i == view_state.cursor;
        if picked {
            out.push(UiPrim::rect(
                x + 2,
                ry,
                card_w - 4,
                ROW_H - 2,
                rgba(0x3c, 0x60, 0x8c, 0.95),
            ));
        }

        // The output name carries the state in its colour, so the shape of the
        // list is readable before a single word of it is.
        let ink = match (picked, row.state) {
            (true, _) => rgb(0xff, 0xff, 0xff),
            (false, RowState::Ready) => rgb(0xa8, 0xe0, 0xa8),
            (false, RowState::WrongPlace) => rgb(0xe0, 0xc8, 0x8c),
            (false, RowState::CannotAfford) => rgb(0x78, 0x82, 0x8c),
        };
        let title = if row.out_count > 1 {
            format!("{} x{}", row.out, row.out_count)
        } else {
            row.out.clone()
        };
        out.push(UiPrim::text(title, x + 8, ry + 12, Align::Left, ROW, ink));

        // The station, right-aligned INSIDE the panel. Only when there is one.
        let badge = station_label(row.station);
        if !badge.is_empty() {
            out.push(UiPrim::text(
                badge.to_string(),
                x + card_w - 8,
                ry + 12,
                Align::Right,
                SMALL,
                match row.state {
                    RowState::WrongPlace => rgb(0xff, 0xc8, 0x64),
                    _ => rgb(0x78, 0x82, 0x8c),
                },
            ));
        }

        // Ingredients, with have/need. This is the line that answers "what
        // would I need", which is the second question every player has and the
        // one the old blind key could not answer at all.
        let costs = row
            .needs
            .iter()
            .map(|(name, have, need)| format!("{name} {have}/{need}"))
            .collect::<Vec<_>>()
            .join("   ");
        out.push(UiPrim::text(
            costs,
            x + 8,
            ry + 22,
            Align::Left,
            SMALL,
            if picked {
                rgb(0xdc, 0xe6, 0xf0)
            } else {
                rgb(0x64, 0x6e, 0x78)
            },
        ));
    }

    if view_state.rows.len() > from + rows_fit {
        out.push(UiPrim::text(
            format!("+{} more", view_state.rows.len() - from - rows_fit),
            x + card_w - 8,
            y + card_h - 22,
            Align::Right,
            SMALL,
            rgb(0x64, 0x6e, 0x78),
        ));
    }

    let (line, ink) = match (&view_state.said, view_state.rows.get(view_state.cursor)) {
        (Some(said), _) => (said.clone(), rgb(0xff, 0xc8, 0x64)),
        (None, Some(row)) => match row.state {
            RowState::Ready => ("Enter craft   C close".to_string(), rgb(0xb4, 0xbe, 0xc8)),
            RowState::WrongPlace => (
                format!("needs a {} — stand next to one", station_label(row.station)),
                rgb(0xff, 0xc8, 0x64),
            ),
            RowState::CannotAfford => ("not enough materials".to_string(), rgb(0x78, 0x82, 0x8c)),
        },
        (None, None) => ("C close".to_string(), rgb(0xb4, 0xbe, 0xc8)),
    };
    out.push(UiPrim::text(
        line,
        x + 8,
        y + card_h - 8,
        Align::Left,
        SMALL,
        ink,
    ));
    out
}

// --- Systems -----------------------------------------------------------------

/// Everything the screen reads and writes.
#[derive(bevy::ecs::system::SystemParam)]
struct Crafting<'w> {
    keys: Res<'w, ButtonInput<KeyCode>>,
    view: ResMut<'w, CraftingView>,
    pack: ResMut<'w, Pack>,
    toast: ResMut<'w, Toast>,
    body: Option<Res<'w, PlayerBody>>,
    world: Option<Res<'w, SimWorld>>,
}

/// Open, close, move and craft.
///
/// One system, for `worldselect`'s reason: these all mutate the same state and
/// the order between them matters — the key that opens the screen is the key
/// that closes it.
fn drive(mut c: Crafting) {
    let k = BevyKeys(&c.keys);
    let toggled = k.any_pressed(KEYS.craft);

    if !c.view.open {
        if toggled {
            c.view.open = true;
            c.view.said = None;
            refresh(&mut c);
        }
        return;
    }

    if toggled || c.keys.just_pressed(KeyCode::Escape) {
        c.view.open = false;
        return;
    }

    let n = c.view.rows.len();
    if n > 0 {
        if k.any_pressed(KEYS.jump) {
            c.view.cursor = (c.view.cursor + n - 1) % n;
            c.view.said = None;
        }
        if k.any_pressed(KEYS.down) {
            c.view.cursor = (c.view.cursor + 1) % n;
            c.view.said = None;
        }
    }

    if k.any_pressed(KEYS.confirm) {
        make_the_selected_thing(&mut c);
    }
}

/// Craft what the cursor is on, and say what happened either way.
fn make_the_selected_thing(c: &mut Crafting) {
    let at = c.view.cursor;
    let Some(r) = recipes().get(at) else { return };
    let reach = reach_of(c);

    // Through `craft`, which re-checks everything. The row's state is a VIEW of
    // the world as of the last refresh, and the world has kept running behind
    // this card — a player can walk off a workbench with the screen open. The
    // row is what to show; `craft` is what decides.
    if craft(&mut c.pack.0, r, reach) {
        let name = item_by_code(r.out).name;
        let said = if r.out_count > 1 {
            format!("crafted {}x {}", r.out_count, name)
        } else {
            format!("crafted {name}")
        };
        // Both places: the card, for somebody reading it, and the toast, for
        // when they close the card a moment later.
        c.toast.show(said.clone());
        c.view.said = Some(said);
    } else {
        c.view.said = Some(match c.view.rows.get(at).map(|row| row.state) {
            Some(RowState::WrongPlace) => "you are not standing near the station".to_string(),
            Some(RowState::CannotAfford) => "not enough materials".to_string(),
            // Neither, and the craft still failed: the only way left is a full
            // pack with nowhere to put the output, which `can_craft` refuses
            // rather than destroying the ingredients over.
            _ => "no room in the pack".to_string(),
        });
    }
    refresh(c);
}

fn reach_of(c: &Crafting) -> Reach {
    match (&c.world, &c.body) {
        (Some(world), Some(body)) => stations_in_reach(&world.level.grid, body.0.x, body.0.y),
        _ => Reach::HAND,
    }
}

/// Re-resolve every row against the pack and the place.
fn refresh(c: &mut Crafting) {
    let reach = reach_of(c);
    c.view.rows = resolve(&c.pack.0, reach);
    if !c.view.rows.is_empty() {
        c.view.cursor = c.view.cursor.min(c.view.rows.len() - 1);
    }
}

/// Keep the rows current while the card is open.
///
/// The world runs behind it: sand falls, a creature drops loot, the player walks
/// off a bench. A card showing what was true when it opened would be lying
/// within a second of being useful.
fn follow_the_world(mut c: Crafting) {
    if c.view.open {
        refresh(&mut c);
    }
}

/// The screen, its state and its keys.
pub struct CraftScreenPlugin;

impl Plugin for CraftScreenPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CraftingView>().add_systems(
            Update,
            (drive, follow_the_world)
                .chain()
                .run_if(in_state(Scene::Playing))
                .run_if(resource_exists::<Pack>),
        );
    }
}

/// Whether the card is up, for anything that has to stand aside while it is.
pub fn is_open(view: &CraftingView) -> bool {
    view.open
}

/// Hotbar keys the screen swallows while it is open.
///
/// Published so `crate::input` can stand aside rather than this module reaching
/// into it: `1`-`0` select a hotbar slot in the world and would otherwise fire
/// underneath the card.
pub const SWALLOWED_HOTBAR: usize = HOTBAR;

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::items::item_code_of;

    fn view() -> View {
        View::for_screen(1280, 720)
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

    /// A recipe that needs a station, and one that does not.
    fn a_station_recipe() -> &'static Recipe {
        recipes()
            .iter()
            .find(|r| r.station == Station::Workbench)
            .expect("some recipe wants a workbench")
    }

    fn stocked_for(r: &Recipe) -> Inventory {
        let mut inv = Inventory::new();
        for i in 0..r.in_items.len() {
            inv.add(r.in_items[i], u32::from(r.in_counts[i]));
        }
        inv
    }

    /// The distinction the whole screen exists for: three states, not two.
    #[test]
    fn a_row_says_which_of_the_two_nots_it_is() {
        let r = a_station_recipe();
        let paid = stocked_for(r);
        let broke = Inventory::new();

        let at = recipes().iter().position(|x| std::ptr::eq(x, r)).unwrap();

        let nowhere = resolve(&paid, Reach::HAND);
        assert_eq!(
            nowhere[at].state,
            RowState::WrongPlace,
            "paid for, wrong place"
        );

        let beside = resolve(&paid, Reach::HAND.with(Station::Workbench));
        assert_eq!(beside[at].state, RowState::Ready, "paid for, right place");

        let empty = resolve(&broke, Reach::HAND.with(Station::Workbench));
        assert_eq!(
            empty[at].state,
            RowState::CannotAfford,
            "right place, cannot pay — and this must NOT read as WrongPlace"
        );
    }

    /// Every recipe is listed, not only the ones you can make. The second
    /// question a player has is "what would I need", and a filtered list cannot
    /// answer it.
    #[test]
    fn the_list_is_every_recipe_in_the_game() {
        let rows = resolve(&Inventory::new(), Reach::HAND);
        assert_eq!(rows.len(), recipes().len());
        assert!(rows.len() > 20, "the registry has more than a handful");
        assert!(
            rows.iter().all(|r| r.state == RowState::CannotAfford),
            "an empty pack can afford nothing, and every row still appears"
        );
    }

    #[test]
    fn a_row_shows_what_it_costs_and_how_much_of_it_you_have() {
        let r = a_station_recipe();
        let mut inv = stocked_for(r);
        // Spend one, so a have/need pair is visibly short rather than equal.
        inv.remove(r.in_items[0], 1);
        let at = recipes().iter().position(|x| std::ptr::eq(x, r)).unwrap();

        let rows = resolve(&inv, Reach::HAND.with(Station::Workbench));
        assert_eq!(rows[at].state, RowState::CannotAfford);
        let (name, have, need) = rows[at].needs[0].clone();
        assert_eq!(have, u32::from(r.in_counts[0]) - 1);
        assert_eq!(need, u32::from(r.in_counts[0]));

        let state = CraftingView {
            open: true,
            rows,
            cursor: at,
            said: None,
        };
        let text = texts(&screen(&state, view())).join(" | ");
        assert!(
            text.contains(&format!("{name} {have}/{need}")),
            "the ingredient line has to carry have/need: {text}"
        );
    }

    /// The footer is the second half of the three-state distinction: it says
    /// what to DO about the row under the cursor.
    #[test]
    fn the_footer_tells_you_what_is_wrong_with_the_selected_row() {
        let r = a_station_recipe();
        let at = recipes().iter().position(|x| std::ptr::eq(x, r)).unwrap();

        let wrong_place = CraftingView {
            open: true,
            rows: resolve(&stocked_for(r), Reach::HAND),
            cursor: at,
            said: None,
        };
        let text = texts(&screen(&wrong_place, view())).join(" | ");
        assert!(
            text.contains("stand next to one"),
            "a player who has the materials must be sent to a station: {text}"
        );

        let broke = CraftingView {
            open: true,
            rows: resolve(&Inventory::new(), Reach::HAND.with(Station::Workbench)),
            cursor: at,
            said: None,
        };
        let text = texts(&screen(&broke, view())).join(" | ");
        assert!(
            text.contains("not enough materials"),
            "and a player who is in the right place must be sent to a mine: {text}"
        );
    }

    /// `hand` prints no badge. The badge exists to say where to go, and two
    /// thirds of the list needs nowhere.
    #[test]
    fn a_hand_recipe_carries_no_station_badge() {
        let hand = recipes()
            .iter()
            .position(|r| r.station == Station::Hand)
            .expect("some recipe is made by hand");
        let state = CraftingView {
            open: true,
            rows: resolve(&Inventory::new(), Reach::HAND),
            cursor: hand,
            said: None,
        };
        let text = texts(&screen(&state, view()));
        assert!(
            !text.iter().any(|t| t == "Hand"),
            "a hand recipe must not carry a badge: {text:?}"
        );
    }

    #[test]
    fn a_long_list_scrolls_with_the_cursor() {
        let rows = resolve(&Inventory::new(), Reach::HAND);
        let last = rows.len() - 1;
        let mut state = CraftingView {
            open: true,
            rows,
            cursor: 0,
            said: None,
        };

        let top = texts(&screen(&state, view())).join(" | ");
        assert!(top.contains("more"), "a 45-row list does not fit: {top}");

        state.cursor = last;
        let bottom = texts(&screen(&state, view())).join(" | ");
        let name = &state.rows[last].out;
        assert!(
            bottom.contains(name.as_str()),
            "the cursor must stay on screen: looking for {name:?}"
        );
        assert!(!bottom.contains("more"), "nothing is hidden at the end");
    }

    #[test]
    fn the_screen_stays_inside_the_buffer() {
        let v = view();
        let rows = resolve(&Inventory::new(), Reach::HAND);
        let mut state = CraftingView {
            open: true,
            rows,
            cursor: 0,
            said: Some("crafted something with a rather long name".into()),
        };
        for cursor in [0, 7, 20, state.rows.len() - 1] {
            state.cursor = cursor;
            for prim in screen(&state, v) {
                match prim {
                    UiPrim::Rect { x, y, w, h, .. } => {
                        assert!(x >= 0 && y >= 0 && x + w <= v.w && y + h <= v.h, "{prim:?}");
                    }
                    UiPrim::Text { baseline, .. } => {
                        assert!(baseline > 0 && baseline < v.h, "{prim:?}");
                    }
                    UiPrim::Icon { .. } => {}
                }
            }
        }
    }

    /// A craft that happened says so where the player is looking.
    #[test]
    fn a_message_replaces_the_hints_rather_than_hiding_behind_them() {
        let state = CraftingView {
            open: true,
            rows: resolve(&Inventory::new(), Reach::HAND),
            cursor: 0,
            said: Some("crafted 3x Ladder".into()),
        };
        let text = texts(&screen(&state, view())).join(" | ");
        assert!(text.contains("crafted 3x Ladder"), "{text}");
        assert!(!text.contains("Enter craft"), "the message takes the line");
    }

    /// The one thing `resolve` must not do: report a row as ready when the
    /// station is missing, which is what four milestones of ignoring the field
    /// looked like.
    #[test]
    fn nothing_is_ready_without_its_station() {
        let mut inv = Inventory::new();
        for r in recipes() {
            for i in 0..r.in_items.len() {
                inv.add(r.in_items[i], u32::from(r.in_counts[i]) * 4);
            }
        }
        let rows = resolve(&inv, Reach::HAND);
        for (row, r) in rows.iter().zip(recipes()) {
            if r.station != Station::Hand {
                assert_ne!(
                    row.state,
                    RowState::Ready,
                    "{} is ready with nothing in reach",
                    row.out
                );
            }
        }
        assert!(
            rows.iter().any(|r| r.state == RowState::Ready),
            "with a full pack the hand recipes must be ready, or this proves nothing"
        );
        let _ = item_code_of("wood_log");
    }
}

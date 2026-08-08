//! Bevy's keyboard and mouse, mapped onto what the engine understands.
//!
//! This is the whole of the `Input` class the TypeScript kept in `src/input/`.
//! That class was a DOM object and did not survive; what it PRODUCED did, and
//! it lives in two halves:
//!
//!   - [`yugen_core::input`] owns the binding table and [`Intent`]. No Bevy.
//!   - this module owns the translation: `ButtonInput<KeyCode>` into
//!     [`KeyState`], the pointer into a world-space [`Cursor`], and the wheel
//!     into a signed step whose meaning the tool's mode decides.
//!
//! # Held vs just-pressed, and why it is load-bearing
//!
//! `ButtonInput` answers both questions — `pressed` and `just_pressed` — and
//! [`BevyKeys`] hands both through unflattened. Jump and dash are EDGE
//! triggered: collapsing `just_pressed` into `pressed` makes a held jump key
//! pogo and a held dash key spend a dash every step, and the symptom (a
//! character that feels wrong) points nowhere near the cause.
//!
//! Bevy clears its just-pressed set once per rendered frame, but `FixedUpdate`
//! can run several times inside one. So the edge is read ONCE, in `PreUpdate`,
//! and [`Intent::for_substep`] is what hands it to exactly one fixed step —
//! which is `Game.ts`'s `jumpQueued: intent.jumpQueued && steps === 0`, moved
//! off the game loop and into a resource ([`FixedSubstep`]) that any fixed
//! system can read.
//!
//! # The brush writes the grid directly
//!
//! `Game.handleBuildInput` had two exits: `postMessage` to the sim worker, or a
//! direct `applyBrush` on the fallback. There is one owned grid here and one
//! writer, so [`swing_brush`] takes the direct path unconditionally. The one-tick
//! latency window the TypeScript documented (Game.ts:519-523) is not mitigated
//! here — it does not exist to mitigate.

use bevy::ecs::system::SystemParam;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::interact_reach::stations_in_reach;
use yugen_core::config::{CELL_SIZE, MAX_HEALTH, MAX_RUN_SPEED, PLAYER_H, PLAYER_W};
use yugen_core::input::{Intent, KEYS, KeyState};
use yugen_core::interact::{BuildTool, Cursor, Held, PALETTE_SLOTS};
use yugen_core::items::crafting::Reach;
use yugen_core::items::registry::{ItemCategory, ItemEffect};
use yugen_core::items::{HOTBAR, Inventory, craft, item_by_code, next_craftable, recipes};
use yugen_core::sim::edits::{EditMode, apply_brush};

use crate::cellmap::upload_dirty_chunks;
use crate::items::Pack;
use crate::lowres::{LowResTarget, WORLD_LAYERS};
use crate::player::PlayerBody;
use crate::ui::Toast;
use crate::world::{SimWorld, WorldFocus};

/// How much faster the free camera flies than the player runs.
///
/// The camera exists to inspect a streaming window 1760x1280 px across; at the
/// player's own speed, crossing it takes half a minute.
const CAMERA_SPEED_SCALE: f32 = 4.0;

/// Extra multiplier on the free camera while shift is held.
const CAMERA_BOOST: f32 = 4.0;

/// How solid the brush preview is over the cells it covers.
///
/// Low enough to read the terrain through, high enough to find on a bright
/// snowfield. A one-cell outline would be better and is the HUD milestone's job;
/// this is the geometry made visible, not the finished cursor.
const PREVIEW_ALPHA: f32 = 0.18;

/// Keyboard, mouse, the dig/place brush, and whatever is driving the view.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerIntent>()
            .init_resource::<FixedSubstep>()
            .init_resource::<CursorWorld>()
            .init_resource::<CursorOverride>()
            .init_resource::<FocusDriver>()
            .init_resource::<Tool>()
            .init_resource::<CraftCursor>()
            .add_systems(Startup, spawn_preview)
            .add_systems(PreUpdate, gather_intent.after(bevy::input::InputSystems))
            // The free camera runs first so the cursor is un-projected against
            // the view the frame will actually draw, and the brush runs before
            // the cell texture is uploaded so an edit lands on the same frame
            // the click did.
            .add_systems(
                Update,
                (fly_camera.run_if(resource_equals(FocusDriver::FreeCamera)),).before(track_cursor),
            )
            .add_systems(
                Update,
                (track_cursor, tool_keys, swing_brush, place_preview)
                    .chain()
                    .before(upload_dirty_chunks)
                    .run_if(resource_exists::<SimWorld>),
            )
            .add_systems(FixedLast, advance_substep);
    }
}

/// This frame's [`Intent`], read once from the keyboard.
///
/// A fixed-step consumer wants `intent.for_substep(substep.0)`, not this —
/// see [`FixedSubstep`].
#[derive(Resource, Clone, Copy, Debug, Default, Deref, DerefMut)]
pub struct PlayerIntent(pub Intent);

/// Which fixed step of the current rendered frame is running, counting from 0.
///
/// [`Intent::for_substep`] uses it to give a rising edge to exactly one step.
/// Reset when the intent is sampled, bumped at the end of every fixed step.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct FixedSubstep(pub u32);

/// The pointer in world px (+y DOWN), or `None` when it is off the window.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct CursorWorld(pub Option<Vec2>);

/// A pointer position supplied by something that is not a mouse.
///
/// `--script` needs one: the binary drives the whole run from a file, and a
/// headless or unfocused window reports no physical cursor at all, so
/// [`track_cursor`] would write `None` over anything the script had aimed and
/// [`swing_brush`] would return before digging a single cell.
///
/// An override consulted by [`track_cursor`] rather than a write that races it,
/// because there must stay exactly ONE place that decides where the pointer is.
/// A second writer ordered after this one would be a bug nobody could see:
/// whichever ran last would win, and both would look correct in isolation.
///
/// `None` — the default — means "there is no override", which is not the same as
/// `Some(None)`. Nothing sets that, and the distinction is why this is not just
/// another `CursorWorld`.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct CursorOverride(pub Option<Vec2>);

/// The dig/place tool. Bevy-side wrapper; all the behaviour is in the core type.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct Tool(pub BuildTool);

/// Where the next craft press starts scanning the recipe table.
///
/// `Game.craftCursor`. There is no crafting SCREEN — see the head of
/// [`yugen_core::items::crafting`] for why the authored station is validated
/// and then ignored — so one key has to reach every affordable recipe.
/// [`next_craftable`] scans from here and a successful craft parks the cursor
/// one past what it made, which turns repeated presses into a walk through the
/// affordable set instead of ten of the same bandage.
///
/// Nothing resets it when the pack changes, and nothing needs to: a recipe that
/// stops being affordable simply stops being returned, and the scan wraps.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct CraftCursor(pub usize);

/// Who moves [`WorldFocus`] — the view, and therefore the streaming window.
///
/// # Swapping the free camera for the player
///
/// This resource is the whole switch. When the player entity lands:
///
///   1. insert `FocusDriver::Player` where it spawns;
///   2. add one system that writes `WorldFocus` from the player's body,
///      `run_if(resource_equals(FocusDriver::Player))`;
///   3. delete nothing.
///
/// [`fly_camera`] stops running by itself, and a session with no player — a
/// screenshot run, a worldgen inspection — still flies, because the default is
/// the free camera and nothing sets the other value until there is a body to
/// follow.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FocusDriver {
    /// WASD flies the view. What you get while there is no player.
    #[default]
    FreeCamera,
    /// The player's body carries the view.
    Player,
}

/// The translucent quad drawn over the cells the next stroke would touch.
#[derive(Component)]
pub struct BrushPreview;

// --- Keyboard ---------------------------------------------------------------

/// [`KeyState`] over Bevy's keyboard.
///
/// The engine's binding table speaks `KeyboardEvent.code` names; this is the one
/// place those become `KeyCode`s. Borrowed rather than owned so a system can
/// wrap its `Res` for the length of one call and hold no state at all — which is
/// the point: the old class's `held`/`pressed` sets ARE `ButtonInput`, already
/// maintained by the engine, and keeping a second copy would only let the two
/// disagree.
pub struct BevyKeys<'a>(pub &'a ButtonInput<KeyCode>);

impl KeyState for BevyKeys<'_> {
    fn is_down(&self, code: &str) -> bool {
        key_code(code).is_some_and(|k| self.0.pressed(k))
    }

    fn was_pressed(&self, code: &str) -> bool {
        key_code(code).is_some_and(|k| self.0.just_pressed(k))
    }
}

/// Every `KeyboardEvent.code` name the binding table uses, and its Bevy key.
///
/// Bevy's `KeyCode` variants ARE the W3C code names, so this table is a spelling
/// of the identity function — but only for the codes below. A `KeyCode` cannot
/// be parsed from a string, and a table that silently returned `None` for a
/// typo'd binding would produce a key that does nothing and no error; the test
/// at the bottom is what makes that impossible.
const CODES: &[(&str, KeyCode)] = &[
    ("ArrowLeft", KeyCode::ArrowLeft),
    ("ArrowRight", KeyCode::ArrowRight),
    ("ArrowUp", KeyCode::ArrowUp),
    ("ArrowDown", KeyCode::ArrowDown),
    ("KeyA", KeyCode::KeyA),
    ("KeyB", KeyCode::KeyB),
    ("KeyC", KeyCode::KeyC),
    ("KeyD", KeyCode::KeyD),
    ("KeyF", KeyCode::KeyF),
    ("KeyG", KeyCode::KeyG),
    ("KeyK", KeyCode::KeyK),
    ("KeyP", KeyCode::KeyP),
    ("KeyS", KeyCode::KeyS),
    ("KeyW", KeyCode::KeyW),
    ("Space", KeyCode::Space),
    ("Enter", KeyCode::Enter),
    ("ShiftLeft", KeyCode::ShiftLeft),
    ("ShiftRight", KeyCode::ShiftRight),
    ("AltLeft", KeyCode::AltLeft),
    ("AltRight", KeyCode::AltRight),
    ("Backquote", KeyCode::Backquote),
    ("Escape", KeyCode::Escape),
    ("F3", KeyCode::F3),
    ("BracketLeft", KeyCode::BracketLeft),
    ("BracketRight", KeyCode::BracketRight),
    ("Minus", KeyCode::Minus),
    ("Equal", KeyCode::Equal),
    ("Digit0", KeyCode::Digit0),
    ("Digit1", KeyCode::Digit1),
    ("Digit2", KeyCode::Digit2),
    ("Digit3", KeyCode::Digit3),
    ("Digit4", KeyCode::Digit4),
    ("Digit5", KeyCode::Digit5),
    ("Digit6", KeyCode::Digit6),
    ("Digit7", KeyCode::Digit7),
    ("Digit8", KeyCode::Digit8),
    ("Digit9", KeyCode::Digit9),
];

/// The Bevy key for a code name, if the table has one.
///
/// A linear scan of ~30 entries, run a few dozen times a frame. A map would be
/// faster and would need a `LazyLock` and an allocation to save nanoseconds off
/// a path that runs once per frame per binding.
///
/// `pub` so a driver that has to SYNTHESISE a keypress can find the key a
/// binding names. `--script`'s `craft` verb is the caller: crafting is read from
/// `ButtonInput<KeyCode>` by `tool_keys` and has no representation in `Intent`,
/// so the only honest way to script it is to press the key the player would.
pub fn key_code(code: &str) -> Option<KeyCode> {
    CODES
        .iter()
        .find_map(|(name, key)| (*name == code).then_some(*key))
}

/// Sample the keyboard into [`PlayerIntent`], once per rendered frame.
///
/// The aim point is the cursor as of the previous frame's [`track_cursor`],
/// because that is the last time anyone knew where the view was. Aim decides
/// which way a shot leaves the bow; one frame of a moving camera is far below
/// the resolution of that decision, and reading it here rather than in `Update`
/// keeps every field of the intent sampled at the same instant.
fn gather_intent(
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Res<CursorWorld>,
    mut intent: ResMut<PlayerIntent>,
    mut substep: ResMut<FixedSubstep>,
) {
    let mut next = Intent::from_keys(&BevyKeys(&keys));
    if let Some(aim) = cursor.0 {
        next.aim_x = aim.x;
        next.aim_y = aim.y;
    }
    intent.0 = next;
    substep.0 = 0;
}

/// Count fixed steps within the current frame. See [`FixedSubstep`].
fn advance_substep(mut substep: ResMut<FixedSubstep>) {
    substep.0 = substep.0.saturating_add(1);
}

// --- The pointer ------------------------------------------------------------

/// Put the pointer in world px.
///
/// Three coordinate systems meet here and none of them is the next one's:
///
///   1. the window, in PHYSICAL px from its top-left — what winit reports;
///   2. the low-res buffer, `view.w x view.h` logical px, blitted centred on the
///      window at [`LowResTarget::blit_scale`];
///   3. world px, +y DOWN, which is what the sim and the brush speak.
///
/// The camera's own `Transform` is the bridge for the last hop rather than
/// [`WorldFocus`] directly, because [`crate::lowres`] snaps that focus to the
/// pixel grid before it draws. Reading the snapped value back means the cell the
/// cursor is over is the cell under the drawn crosshair, not the cell under an
/// unrounded number that is up to half a pixel away from it.
fn track_cursor(
    window: Single<&Window, With<PrimaryWindow>>,
    target: Res<LowResTarget>,
    camera: Single<&Transform, With<crate::lowres::WorldCamera>>,
    scripted: Res<CursorOverride>,
    mut cursor: ResMut<CursorWorld>,
) {
    // A driven pointer wins outright — see [`CursorOverride`]. It is already in
    // world px, so none of the three coordinate hops below apply to it.
    if let Some(at) = scripted.0 {
        cursor.0 = Some(at);
        return;
    }
    let Some(p) = window.physical_cursor_position() else {
        cursor.0 = None;
        return;
    };
    let physical = Vec2::new(
        window.physical_width() as f32,
        window.physical_height() as f32,
    );
    // Bevy's +y is up and the buffer's is down, so the camera's y comes back
    // negated — the same one-place convention flip `lowres::follow_focus` makes.
    let centre = Vec2::new(camera.translation.x, -camera.translation.y);
    cursor.0 = Some(cursor_world_px(p, physical, target.blit_scale(), centre));
}

/// Where a physical-pixel pointer lands in world px. See [`track_cursor`].
///
/// Split out from the system because it is the part that can be wrong in a way
/// no screenshot would show: an off-by-half here puts the dig one cell from the
/// crosshair and looks like a rendering bug.
fn cursor_world_px(pointer: Vec2, physical: Vec2, blit_scale: f32, centre: Vec2) -> Vec2 {
    centre + (pointer - physical * 0.5) / blit_scale
}

// --- The view ---------------------------------------------------------------

/// Fly the view with WASD or the arrow keys; hold shift to go faster.
///
/// Writes [`WorldFocus`], which is both what the streaming window recentres on
/// and what the world camera follows — so this one system moves the view AND
/// pulls new chunks in behind it.
///
/// It shares WASD with the player's own movement binding on purpose. The two
/// never run at once: [`FocusDriver`] gates this one off the moment there is a
/// body for those keys to move instead. Flying up is [`KEYS`]`.jump` rather than
/// a hand-listed W/ArrowUp, so space flies too — which is what jump does to a
/// player, and what the same binding read as held does on a ladder.
fn fly_camera(keys: Res<ButtonInput<KeyCode>>, time: Res<Time>, mut focus: ResMut<WorldFocus>) {
    let k = BevyKeys(&keys);
    let mut dir = Vec2::new(
        f32::from(k.any_down(KEYS.right)) - f32::from(k.any_down(KEYS.left)),
        // +y is DOWN in world px, so "up" on the keyboard is negative.
        f32::from(k.any_down(KEYS.down)) - f32::from(k.any_down(KEYS.jump)),
    );
    if dir == Vec2::ZERO {
        return;
    }
    dir = dir.normalize();

    let boost = if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
        CAMERA_BOOST
    } else {
        1.0
    };
    let step = dir * MAX_RUN_SPEED * CAMERA_SPEED_SCALE * boost * time.delta_secs();
    focus.x += step.x;
    focus.y += step.y;
}

// --- The brush --------------------------------------------------------------

/// Everything the survival half of [`tool_keys`] writes to.
///
/// Bundled for the reason [`Swinger`] is: four things, one question — "what is
/// the player carrying, and what have they just been told about it". Listing
/// them would also put the system at seven parameters, which is the count the
/// tree bundles at rather than argues about.
///
/// `body` is optional because `--free-camera` runs with no player at all. There
/// is still a pack to select from and craft into; there is just nobody to drink.
#[derive(SystemParam)]
struct Survival<'w> {
    pack: ResMut<'w, Pack>,
    toast: ResMut<'w, Toast>,
    cursor: ResMut<'w, CraftCursor>,
    body: Option<ResMut<'w, PlayerBody>>,
    /// For the stations within reach of the body — see
    /// [`crate::interact_reach`]. Read-only, and `Option` because a host may run
    /// the HUD without a world.
    world: Option<Res<'w, SimWorld>>,
    /// The crafting card, for whether it is up.
    ///
    /// Read here rather than the card disabling this system, because the same
    /// system also drives the hotbar and the brush: a card that took the whole
    /// of `tool_keys` out would freeze the wheel and the number keys as a side
    /// effect nobody asked for. Only the two keys the card owns stand aside.
    ///
    /// `Option` because a host may run input without the crafting plugin — the
    /// capture rigs do — and a missing card means no card is open.
    card: Option<Res<'w, crate::craftscreen::CraftingView>>,
}

/// The keyboard/wheel half of `Game.handleBuildInput`: mode, selection, size.
///
/// Runs before [`swing_brush`], and that order is the TypeScript's: a stroke
/// started on the same frame as a selection change places what you just picked.
///
/// The digits and the wheel mean different things per mode, and that is the
/// whole reason the branch exists rather than an accident of it: in creative
/// they drive the material palette, in survival they move the hotbar cursor.
/// Brush size lives on the brackets in BOTH modes, because the wheel is the
/// natural hotbar control and the hotbar is what a survival run actually uses.
fn tool_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    mut tool: ResMut<Tool>,
    mut player: Survival,
) {
    let k = BevyKeys(&keys);

    if k.any_pressed(KEYS.creative) {
        tool.toggle_creative();
        // `Game.say`, not a log line. This is the one mode switch a player can
        // make; it silently changes what every other key in this function does,
        // and until the HUD landed the only place to report it was stdout, which
        // the player is not reading. Every other branch below already talks
        // through the toast.
        let mode = if tool.creative {
            "creative on"
        } else {
            "creative off"
        };
        player.toast.show(mode);
    }

    // Read ONCE, above the branch, so the mode cannot decide whether the queue
    // is drained. Unread steps are dropped with the reader, which is the old
    // `clearFrame` dropping whatever no consumer took — and a frame whose
    // consumer happened to be the other branch must not leave a backlog for the
    // next one to spend.
    //
    // The sign is the DOM's, not Bevy's: the TypeScript accumulated
    // `Math.sign(deltaY)`, which is POSITIVE scrolling down, and BOTH consumers
    // below were written against that number. Bevy reports the opposite sign for
    // the same physical motion, hence the negation.
    //
    // So positive is a wheel rolled toward you, and it GROWS the brush and moves
    // the hotbar selection RIGHT, toward slot 9. On the brush that is faithful
    // rather than natural — which way a wheel grows a disc is a design decision
    // and not a porting one. On the hotbar it is both: scroll down, next slot is
    // what every game with a hotbar does, so the faithful port is also the one
    // that needs no explaining to a player.
    let steps: i32 = wheel.read().map(|w| -w.y.signum() as i32).sum();

    if tool.creative {
        for i in 0..PALETTE_SLOTS {
            if k.was_pressed(KEYS.hotbar[i]) {
                tool.select_index(i);
                info!("palette: {}", tool.selected_name());
            }
        }
        if steps != 0 {
            tool.add_brush(steps);
        }
    } else {
        let inv = &mut **player.pack;
        for i in 0..HOTBAR {
            if k.was_pressed(KEYS.hotbar[i]) {
                inv.select_slot(i);
            }
        }

        // Clamped to one slot a frame — `Math.sign(wheel)` in the original, and
        // deliberately NOT what the brush above gets. A brush size is a
        // magnitude and adding three to it is three steps of the same idea; a
        // ten-slot ring that jumps three places on one flick has lost the
        // player, who was looking at the hotbar and not at the wheel. Zero is
        // already a no-op inside `cycle`.
        inv.cycle(steps.signum());

        // `C` opens `crate::craftscreen` now rather than crafting blind, and
        // that module owns the key. `try_craft` below is still the whole of what
        // a craft IS and is still tested; the card calls `craft` directly and
        // this path is what a host without the screen plugin gets.
        if k.any_pressed(KEYS.craft) && !player.card.as_ref().is_some_and(|c| c.open) {
            // What is in reach RIGHT NOW, computed at the keypress rather than
            // cached: a player walks away from a bench, and a cached set would
            // let them keep crafting from it until something else invalidated
            // it. There is one scan per press and presses are rare.
            let reach = match (&player.world, &player.body) {
                (Some(world), Some(body)) => {
                    stations_in_reach(&world.level.grid, body.0.x, body.0.y)
                }
                // No body or no world: nothing to stand next to. Hand recipes
                // still work, which is what `Reach::HAND` means.
                _ => Reach::HAND,
            };
            try_craft(inv, &mut player.cursor.0, &mut player.toast, reach);
        }
        if k.any_pressed(KEYS.use_item)
            && !player.card.as_ref().is_some_and(|c| c.open)
            && let Some(body) = &mut player.body
        {
            try_use(inv, body, &mut player.toast);
        }
    }

    if k.any_pressed(KEYS.brush_down) {
        tool.add_brush(-1);
    }
    if k.any_pressed(KEYS.brush_up) {
        tool.add_brush(1);
    }
}

/// Craft the next affordable recipe and say what it was. `Game.tryCraft`.
///
/// A free function over the three things it touches rather than a system, so the
/// port can be tested against an [`Inventory`] and a [`Toast`] with no Bevy
/// world at all — the same split [`cursor_world_px`] gets, and for the same
/// reason: this is the part that can be wrong in a way no screenshot shows.
///
/// Both outcomes report. A key that silently does nothing is indistinguishable
/// from a key that is not bound, which is precisely the defect the HUD's
/// "C craft" hint had been advertising.
fn try_craft(inv: &mut Inventory, cursor: &mut usize, toast: &mut Toast, reach: Reach) {
    let Some(i) = next_craftable(inv, *cursor, reach) else {
        // Two different nothings, and telling them apart is the whole reason
        // stations are worth having in the UI at all. "You cannot afford
        // anything" and "you are standing in the wrong place" send a player to
        // opposite ends of the game, and a single message would send them to
        // the wrong one half the time.
        toast.show(if next_craftable(inv, *cursor, Reach::ALL).is_some() {
            "nothing craftable here — you need a station"
        } else {
            "nothing craftable"
        });
        return;
    };
    let r = &recipes()[i];
    // `next_craftable` has already run `can_craft` on this row and nothing can
    // have moved the pack in between, so this cannot fail today. Kept as the
    // original had it: the day `craft` grows a second refusal, the cursor must
    // not advance past a craft that did not happen.
    if !craft(inv, r, reach) {
        return;
    }
    *cursor = i + 1;
    let name = item_by_code(r.out).name;
    toast.show(if r.out_count > 1 {
        format!("crafted {}x {}", r.out_count, name)
    } else {
        format!("crafted {name}")
    });
}

/// The armour value of whatever is worn, or zero.
///
/// One place, so the body's number and the pack cannot disagree — which is the
/// failure mode a cached stat has, and the reason this is recomputed rather
/// than adjusted.
pub fn worn_armour(inv: &Inventory) -> f32 {
    inv.worn().map_or(0.0, |code| item_by_code(code).armour)
}

/// Consume the held item if it heals or buffs. `Game.tryUse`.
///
/// Effects are data and not code yet: the `effect` name is authored, compiled
/// and reported, and nothing applies it. That is exactly the state the
/// TypeScript left it in, and it stays there — the buff system is what turns a
/// name into a behaviour, and inventing one here would put a second, unauthored
/// effect table in the input module.
fn try_use(inv: &mut Inventory, body: &mut PlayerBody, toast: &mut Toast) {
    let Some(code) = inv.held() else {
        return;
    };
    let def = item_by_code(code);

    // Armour is USED by putting it on, which is why it shares the key rather
    // than getting one of its own. A player holding a jerkin and pressing the
    // key the HUD calls "use" means exactly one thing.
    if def.armour > 0.0 {
        let slot = inv.selected();
        if inv.equip(slot) {
            body.armour = worn_armour(inv);
            toast.show(format!("wearing {}", def.name));
        } else {
            // The only way `equip` refuses is no room for what came off.
            toast.show("no room to take that off");
        }
        return;
    }

    if def.category != ItemCategory::Consumable {
        return;
    }
    let Some(u) = def.r#use else {
        return;
    };

    let heal = u.heal.unwrap_or(0.0);
    if heal > 0.0 && body.health >= MAX_HEALTH {
        toast.show("already at full health");
        return;
    }

    let slot = inv.selected();
    inv.remove_at(slot, 1);
    if heal > 0.0 {
        body.health = MAX_HEALTH.min(body.health + heal);
    }

    // The original was `+${heal} hp`, with the effect appended when there was
    // one. It was written before a consumable with NO heal existed: the Emberward
    // Draught is pure `fireward`, and "+0 hp · fireward" leads with the one
    // number that is not the point. A heal-less use reports the effect alone.
    // Every other case is the original's line, character for character.
    let effect = u.effect.and_then(effect_name);
    toast.show(match (heal > 0.0, effect) {
        (true, Some(e)) => format!("+{heal} hp · {e}"),
        (false, Some(e)) => e.to_string(),
        (_, None) => format!("+{heal} hp"),
    });
}

/// The authored name of an effect, or `None` for "no effect at all".
///
/// [`ItemEffect::None`] and an absent `effect` field are the same thing to a
/// player and are folded together here, which is `def.use.effect ?? "none"` and
/// the `=== "none"` test that followed it, collapsed into one answer.
///
/// The match is exhaustive on purpose. A new effect in `content/items/` is then
/// a compile error in this file rather than a draught that consumes itself and
/// reports nothing.
fn effect_name(effect: ItemEffect) -> Option<&'static str> {
    match effect {
        ItemEffect::None => None,
        ItemEffect::Regen => Some("regen"),
        ItemEffect::Fireward => Some("fireward"),
        ItemEffect::Haste => Some("haste"),
        ItemEffect::Light => Some("light"),
    }
}

/// Who is swinging, and what they are holding.
///
/// Bundled rather than listed. A Bevy system's parameters are injected rather
/// than passed by a caller, so a long list is not the call-site burden the
/// `too_many_arguments` lint exists to catch — but three of these do answer one
/// question, and grouping them says that outright instead of silencing a lint
/// and leaving the reader to work out which three belong together.
///
/// `body` is optional because `--free-camera` runs with no player at all.
#[derive(SystemParam)]
struct Swinger<'w> {
    focus: Res<'w, WorldFocus>,
    body: Option<Res<'w, PlayerBody>>,
    /// Mutable because placing a block SPENDS one — see the bottom of
    /// [`swing_brush`]. It was `Res` while the pack was an `Arc<Mutex<_>>` and
    /// the write went through the lock, which is precisely the kind of hidden
    /// mutation that let Bevy schedule a writer alongside four readers.
    pack: ResMut<'w, Pack>,
}

/// Ask the tool for this frame's stroke and stamp it.
fn swing_brush(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    cursor: Res<CursorWorld>,
    mut actor: Swinger,
    mut tool: ResMut<Tool>,
    mut world: ResMut<SimWorld>,
) {
    let Swinger { focus, body, pack } = &mut actor;
    let Some(at) = cursor.0 else {
        return;
    };
    let swing = Cursor {
        x: at.x,
        y: at.y,
        dig: buttons.pressed(MouseButton::Left),
        place: buttons.pressed(MouseButton::Right),
        // HELD, not toggled — the modifier says "the other layer" for exactly as
        // long as you mean it, and there is no mode for the HUD to display or for
        // the hotbar to stay in sync with.
        back: BevyKeys(&keys).any_down(KEYS.background),
    };

    // Reach is measured from whoever is swinging: the body when there is one,
    // and the view itself under `--free-camera`, where there is no character to
    // measure from and creative's infinite reach is the only sensible answer.
    let (actor_x, actor_y) = match &body {
        Some(b) => (b.x + PLAYER_W * 0.5, b.y + PLAYER_H * 0.5),
        None => (focus.x, focus.y),
    };

    let held = pack.held().map(|code| Held {
        code,
        count: u32::from(pack.held_count()),
    });

    let Some(act) = tool.update(
        time.delta_secs(),
        swing,
        &world.level.grid,
        actor_x,
        actor_y,
        held,
    ) else {
        return;
    };

    // The direct path, and the only path. See the module docs.
    apply_brush(
        &mut world.level.grid,
        act.mode,
        act.cx,
        act.cy,
        act.r,
        act.mat,
    );

    // Pay for it. The tool sized the disc to what the stack could cover and
    // published the count rather than spending it itself — see `BuildTool::place`
    // — so this is the other half of that contract and must not be skipped.
    // Creative places for free, and `place_cost` is zero for a dig.
    if act.mode == EditMode::Place && !tool.creative {
        let cost = tool.place_cost();
        if cost > 0 {
            let inv = &mut **pack;
            let slot = inv.selected();
            inv.remove_at(slot, cost as u32);
        }
    }
}

/// A translucent quad the size of the disc the next stroke would cut.
fn spawn_preview(mut commands: Commands) {
    commands.spawn((
        Sprite {
            color: Color::srgba(1.0, 1.0, 1.0, PREVIEW_ALPHA),
            custom_size: Some(Vec2::splat(CELL_SIZE as f32)),
            ..default()
        },
        // Above the cell quad, which sits at z = 0.
        Transform::from_xyz(0.0, 0.0, 1.0),
        Visibility::Hidden,
        BrushPreview,
        WORLD_LAYERS,
    ));
}

/// Put the preview on the disc the tool reports, or hide it with the pointer.
///
/// The geometry is [`BuildTool::preview_bounds`] and is NOT recomputed here: a
/// second derivation of "which cells does radius r cover" is a second answer
/// waiting to disagree with `apply_brush`.
fn place_preview(
    tool: Res<Tool>,
    cursor: Res<CursorWorld>,
    preview: Single<(&mut Sprite, &mut Transform, &mut Visibility), With<BrushPreview>>,
) {
    let (mut sprite, mut transform, mut visibility) = preview.into_inner();
    if cursor.0.is_none() {
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Inherited;

    let (x, y, w, h) = tool.preview_bounds();
    sprite.custom_size = Some(Vec2::new(w, h));
    // Top-left plus half the extent is the centre a sprite is anchored at, and
    // world +y down is Bevy +y up.
    transform.translation.x = x + w * 0.5;
    transform.translation.y = -(y + h * 0.5);
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::config::View;

    /// Every binding the engine names has to resolve to a real key.
    ///
    /// A missing entry is silent at runtime — the key simply never fires — so
    /// this is the only thing standing between a typo and an unreachable verb.
    #[test]
    fn every_binding_maps_to_a_bevy_key() {
        let tables: [&[&str]; 12] = [
            KEYS.left,
            KEYS.right,
            KEYS.jump,
            KEYS.down,
            KEYS.dash,
            KEYS.punch,
            KEYS.debug,
            KEYS.confirm,
            KEYS.hotbar,
            KEYS.use_item,
            KEYS.craft,
            KEYS.creative,
        ];
        for table in tables {
            for code in table {
                assert!(key_code(code).is_some(), "no Bevy key for {code:?}");
            }
        }
        for code in KEYS.brush_down.iter().chain(KEYS.brush_up) {
            assert!(key_code(code).is_some(), "no Bevy key for {code:?}");
        }
    }

    #[test]
    fn the_code_table_has_no_duplicates() {
        for (i, (name, key)) in CODES.iter().enumerate() {
            for (other, other_key) in &CODES[i + 1..] {
                assert_ne!(name, other, "{name} listed twice");
                assert_ne!(key, other_key, "{name} and {other} are the same key");
            }
        }
    }

    #[test]
    fn an_unknown_code_is_none_rather_than_a_wrong_key() {
        assert_eq!(key_code("KeyQ"), None);
        assert_eq!(key_code(""), None);
        assert_eq!(key_code("keya"), None, "code names are case sensitive");
    }

    /// The pointer at the centre of the window is the cell the view is on.
    #[test]
    fn the_window_centre_is_the_camera_centre() {
        let physical = Vec2::new(1280.0, 800.0);
        let centre = Vec2::new(1000.0, -250.0);
        assert_eq!(
            cursor_world_px(physical * 0.5, physical, 2.0, centre),
            centre
        );
    }

    #[test]
    fn a_pointer_offset_is_divided_by_the_upscale() {
        let physical = Vec2::new(1280.0, 800.0);
        let centre = Vec2::ZERO;
        // 200 physical px right of centre at 2x is 100 world px, which at
        // CELL_SIZE 5 is 20 cells.
        let at = cursor_world_px(
            physical * 0.5 + Vec2::new(200.0, 0.0),
            physical,
            2.0,
            centre,
        );
        assert_eq!(at, Vec2::new(100.0, 0.0));
        assert_eq!(at.x as i32 / CELL_SIZE, 20);

        // And down the screen is +y in world px, the sim's convention.
        let below = cursor_world_px(physical * 0.5 + Vec2::new(0.0, 60.0), physical, 3.0, centre);
        assert_eq!(below, Vec2::new(0.0, 20.0));
    }

    /// One cell on screen is one cell to the brush.
    ///
    /// At the default window's 2x blit a cell is `CELL_SIZE * 2` physical px
    /// wide, and stepping the pointer by that much has to step the target cell
    /// by exactly one — anywhere on the screen, including left of the centre
    /// where the floor division changes sign.
    #[test]
    fn a_cell_on_screen_is_a_cell_to_the_brush() {
        let view = View::for_screen(1280, 800);
        assert_eq!(view.zoom, 2.0);
        let physical = Vec2::new(1280.0, 800.0);
        let cell_px = (CELL_SIZE * 2) as f32;

        let mut previous = None;
        for step in -4..=4 {
            let pointer = physical * 0.5 + Vec2::new(step as f32 * cell_px, 0.0);
            let at = cursor_world_px(pointer, physical, 2.0, Vec2::ZERO);
            let cell = yugen_core::config::cell_at(at.x);
            if let Some(prev) = previous {
                assert_eq!(cell - prev, 1, "step {step} moved by more than a cell");
            }
            previous = Some(cell);
        }
    }

    #[test]
    fn the_free_camera_is_the_default_driver() {
        assert_eq!(FocusDriver::default(), FocusDriver::FreeCamera);
    }

    // --- The mouse path, end to end -----------------------------------------
    //
    // A screenshot proves the brush draws; these prove the BUTTON reaches it.
    // They run the real system against a real `SimWorld`, so a broken resource,
    // a swapped button or a dropped `apply_brush` call fails here rather than in
    // a play session.

    use bevy::ecs::system::RunSystemOnce;
    use yugen_core::sim::automata::Automata;
    use yugen_core::sim::chunk_store::ChunkStore;
    use yugen_core::sim::grid::CellGrid;
    use yugen_core::sim::level::Level;
    use yugen_core::sim::materials::{EMPTY, block};
    use yugen_core::sim::window::WindowManager;
    use yugen_core::sim::worldgen::SpawnPoint;

    /// A world of solid stone with the cursor over its centre and a button held.
    fn clicking(button: MouseButton) -> World {
        const SEED: u32 = 1;
        let mut grid = CellGrid::new(64, 64);
        for y in 0..grid.rows() {
            for x in 0..grid.cols() {
                grid.set(x, y, block::STONE);
            }
        }

        let mut world = World::new();
        world.insert_resource(Time::<()>::default());
        let mut buttons = ButtonInput::<MouseButton>::default();
        buttons.press(button);
        world.insert_resource(buttons);

        // Cell (32, 32), dead centre of the grid, in world px.
        let at = Vec2::splat((32 * CELL_SIZE + CELL_SIZE / 2) as f32);
        world.insert_resource(CursorWorld(Some(at)));
        world.insert_resource(WorldFocus { x: at.x, y: at.y });
        // Creative EXPLICITLY, not by inheriting whatever `START_CREATIVE`
        // happens to be. These tests are about the brush reaching the grid at
        // all, so they want the mode that ignores the pack, reach and hardness —
        // and they should say so. They previously relied on the shipping default
        // being creative, and broke the day it flipped to survival, which is the
        // test depending on a decision it does not care about.
        //
        // The survival paths are tested where they live, in
        // `yugen_core::interact`, against a pack and a tool profile.
        let mut build = BuildTool::new();
        build.creative = true;
        world.insert_resource(Tool(build));
        world.insert_resource(Pack::default());
        // No keys held, so every stroke below is a FRONT-plane stroke. The
        // background modifier is tested where it is decided, in
        // `yugen_core::interact` and `yugen_core::sim::edits`; what these
        // check is the mouse-to-brush wiring.
        world.insert_resource(ButtonInput::<KeyCode>::default());
        world.insert_resource(SimWorld {
            level: Level::new(grid, SpawnPoint { x: at.x, y: at.y }),
            window: WindowManager::new(ChunkStore::new(SEED)),
            automata: Automata::new(),
            seed: SEED,
        });
        world
    }

    #[test]
    fn the_left_button_digs_the_cells_it_is_over() {
        let mut world = clicking(MouseButton::Left);
        let r = world.resource::<Tool>().brush;

        world.run_system_once(swing_brush).unwrap();

        let grid = &world.resource::<SimWorld>().level.grid;
        assert_eq!(grid.get(32, 32), EMPTY, "the centre survived the click");
        assert_eq!(grid.get(32 + r, 32), EMPTY, "the disc edge did not");
        assert_eq!(grid.get(32 + r + 1, 32), block::STONE, "and nothing beyond");
    }

    #[test]
    fn the_right_button_places_the_selected_material() {
        let mut world = clicking(MouseButton::Right);
        // Dig first, so there is somewhere for a placement to go: place fills
        // only empty cells, which on a solid grid is nowhere at all.
        {
            let grid = &mut world.resource_mut::<SimWorld>().level.grid;
            for y in 28..37 {
                for x in 28..37 {
                    grid.set(x, y, EMPTY);
                }
            }
        }
        let want = world.resource::<Tool>().selected();

        world.run_system_once(swing_brush).unwrap();

        let grid = &world.resource::<SimWorld>().level.grid;
        assert_eq!(grid.get(32, 32), want);
        // The corner of the cleared box is outside the disc, so the fill did
        // not reach it and it is still the air it was.
        assert_eq!(grid.get(28, 28), EMPTY);
        // And the stone beyond the cleared box was never a candidate: place
        // fills only empty cells.
        assert_eq!(grid.get(20, 32), block::STONE);
    }

    #[test]
    fn no_button_is_no_edit() {
        let mut world = clicking(MouseButton::Middle);
        world.run_system_once(swing_brush).unwrap();
        let grid = &world.resource::<SimWorld>().level.grid;
        assert!(
            grid.material.iter().all(|&m| m == block::STONE),
            "something wrote cells with nothing held"
        );
    }

    #[test]
    fn the_pointer_off_the_window_swings_at_nothing() {
        let mut world = clicking(MouseButton::Left);
        world.insert_resource(CursorWorld(None));
        world.run_system_once(swing_brush).unwrap();
        let grid = &world.resource::<SimWorld>().level.grid;
        assert!(grid.material.iter().all(|&m| m == block::STONE));
    }

    // --- The survival half of `tool_keys` -----------------------------------
    //
    // The seam these close was a HUD advertising three keys — the digits, C and
    // F — that reached nothing. So what is asserted throughout is that the key
    // ARRIVED: that the pack moved, that the toast said something. The rules it
    // arrived at (what `cycle` wraps to, what `craft` may spend) are tested
    // where they live, in `yugen_core::items`.

    use bevy::input::mouse::MouseScrollUnit;
    use bevy::input::touch::TouchPhase;
    use yugen_core::items::registry::{ITEM_DEFS, ItemDef};
    use yugen_core::items::{Recipe, item_code_of, recipes};

    /// A world with the survival half wired and the given keys down this frame.
    ///
    /// Survival EXPLICITLY, for the same reason [`clicking`] says creative
    /// explicitly: these tests are about which side of the branch a key lands
    /// on, so the side must not be whatever `START_CREATIVE` happens to be.
    fn pressing(codes: &[&str]) -> World {
        let mut world = World::new();
        let mut keys = ButtonInput::<KeyCode>::default();
        for code in codes {
            keys.press(key_code(code).expect("the test named an unbound key"));
        }
        world.insert_resource(keys);
        // `MessageReader` reads this resource whether or not anything wrote to
        // it, so a wheel-less test still has to have one.
        world.init_resource::<Messages<MouseWheel>>();

        let mut build = BuildTool::new();
        build.creative = false;
        world.insert_resource(Tool(build));
        world.insert_resource(Pack::default());
        world.insert_resource(Toast::default());
        world.insert_resource(CraftCursor::default());
        world
    }

    /// Roll the wheel by `y` in Bevy's sign — POSITIVE is away from you.
    fn scroll(world: &mut World, y: f32) {
        world.write_message(MouseWheel {
            unit: MouseScrollUnit::Line,
            x: 0.0,
            y,
            window: Entity::PLACEHOLDER,
            phase: TouchPhase::Moved,
        });
    }

    /// A player at full health, so `try_use` has somebody to heal.
    fn with_a_body(world: &mut World) {
        world.insert_resource(PlayerBody(yugen_core::entities::Player::new(SpawnPoint {
            x: 0.0,
            y: 0.0,
        })));
    }

    fn toast_of(world: &World) -> String {
        world
            .resource::<Toast>()
            .showing()
            .map(|(text, _)| text.to_owned())
            .unwrap_or_default()
    }

    /// The first consumable the content declares that actually restores health.
    fn a_healing_item() -> &'static ItemDef {
        ITEM_DEFS
            .iter()
            .find(|d| {
                d.category == ItemCategory::Consumable
                    && d.r#use.is_some_and(|u| u.heal.unwrap_or(0.0) > 0.0)
            })
            .expect("content declares no consumable that heals")
    }

    #[test]
    fn a_digit_selects_that_hotbar_slot_in_survival() {
        let mut world = pressing(&[KEYS.hotbar[4]]);
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Pack>().selected(), 4);
    }

    /// Digit 0 is the LAST slot, not the first. The binding table is
    /// index-ordered and this is the end of it that a naive `i` would get wrong.
    #[test]
    fn the_zero_key_selects_the_last_hotbar_slot() {
        let mut world = pressing(&[KEYS.hotbar[HOTBAR - 1]]);
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Pack>().selected(), HOTBAR - 1);
    }

    /// The creative branch must not have started writing the pack.
    #[test]
    fn a_digit_drives_the_palette_and_not_the_pack_in_creative() {
        let mut world = pressing(&[KEYS.hotbar[4]]);
        world.resource_mut::<Tool>().creative = true;
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Tool>().slot_index(), 4);
        assert_eq!(
            world.resource::<Pack>().selected(),
            0,
            "creative digits reached the inventory"
        );
    }

    /// Scrolling toward you steps the selection RIGHT and leaves the brush
    /// alone. See the sign argument in `tool_keys`.
    #[test]
    fn the_wheel_cycles_the_hotbar_in_survival_and_not_the_brush() {
        let mut world = pressing(&[]);
        let brush = world.resource::<Tool>().brush;
        scroll(&mut world, -1.0);
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Pack>().selected(), 1);
        assert_eq!(
            world.resource::<Tool>().brush,
            brush,
            "the survival wheel sized the brush"
        );
    }

    #[test]
    fn scrolling_away_from_you_steps_the_selection_back_and_wraps() {
        let mut world = pressing(&[]);
        scroll(&mut world, 1.0);
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Pack>().selected(), HOTBAR - 1);
    }

    /// A flick that reports three steps moves ONE slot — the original's
    /// `Math.sign` — while the same three steps size the brush by three.
    #[test]
    fn a_hard_flick_moves_one_slot_but_sizes_the_brush_by_three() {
        let mut world = pressing(&[]);
        for _ in 0..3 {
            scroll(&mut world, -1.0);
        }
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Pack>().selected(), 1);

        let mut world = pressing(&[]);
        world.resource_mut::<Tool>().creative = true;
        let brush = world.resource::<Tool>().brush;
        for _ in 0..3 {
            scroll(&mut world, -1.0);
        }
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Tool>().brush, brush + 3);
    }

    /// The brackets are the brush in survival too — the one binding the mode
    /// branch must NOT swallow.
    #[test]
    fn the_brackets_still_size_the_brush_in_survival() {
        let mut world = pressing(KEYS.brush_up);
        let brush = world.resource::<Tool>().brush;
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(world.resource::<Tool>().brush, brush + 1);
    }

    /// Stock `inv` with exactly what one craft of `r` costs.
    fn stock(inv: &mut Inventory, r: &Recipe) {
        for i in 0..r.in_items.len() {
            inv.add(r.in_items[i], r.in_counts[i] as u32);
        }
    }

    /// The first recipe a player can reach with no station: `recipes()[0]` is a
    /// furnace one, and this test predates stations entirely.
    fn first_hand_recipe() -> &'static Recipe {
        recipes()
            .iter()
            .find(|r| r.station == yugen_core::items::registry::Station::Hand)
            .expect("some recipe is made by hand")
    }

    #[test]
    fn the_craft_key_makes_the_first_affordable_recipe_and_says_so() {
        let mut world = pressing(KEYS.craft);
        // A HAND recipe, because this world has no station in it. Before
        // stations existed this was `recipes()[0]` and passed; that recipe
        // wants a furnace, so it now correctly refuses, and the test was
        // asserting the old behaviour rather than the intended one.
        let r = first_hand_recipe();
        stock(&mut world.resource_mut::<Pack>(), r);

        world.run_system_once(tool_keys).unwrap();

        let name = item_by_code(r.out).name;
        assert!(
            world.resource::<Pack>().count_of(r.out) >= r.out_count as u32,
            "{name} was never credited"
        );
        assert!(
            toast_of(&world).contains(name),
            "the toast did not name what was crafted"
        );
        let at = recipes().iter().position(|x| std::ptr::eq(x, r)).unwrap();
        assert_eq!(
            world.resource::<CraftCursor>().0,
            at + 1,
            "the cursor did not park past the recipe it made"
        );
    }

    /// The behaviour stations exist for, at the KEY rather than in the rule:
    /// pressing craft with the ingredients but no station makes nothing, and
    /// says which of the two nothings it was.
    #[test]
    fn the_craft_key_refuses_a_station_recipe_and_says_why() {
        let mut world = pressing(KEYS.craft);
        let r = recipes()
            .iter()
            .find(|r| r.station != yugen_core::items::registry::Station::Hand)
            .expect("some recipe wants a station");
        stock(&mut world.resource_mut::<Pack>(), r);

        world.run_system_once(tool_keys).unwrap();

        assert_eq!(
            world.resource::<Pack>().count_of(r.out),
            0,
            "a station recipe was crafted with no station in reach"
        );
        let said = toast_of(&world);
        assert!(
            said.contains("station"),
            "the player is standing in the wrong place and the message has to \
             say so rather than 'nothing craftable', which would send them off \
             to mine something they already have: {said:?}"
        );
    }

    #[test]
    fn the_craft_key_on_an_empty_pack_says_nothing_craftable() {
        let mut world = pressing(KEYS.craft);
        world.run_system_once(tool_keys).unwrap();
        assert_eq!(toast_of(&world), "nothing craftable");
        assert_eq!(world.resource::<CraftCursor>().0, 0);
    }

    #[test]
    fn the_use_key_consumes_one_and_heals() {
        let def = a_healing_item();
        let heal = def.r#use.and_then(|u| u.heal).unwrap();

        let mut world = pressing(KEYS.use_item);
        with_a_body(&mut world);
        world.resource_mut::<PlayerBody>().health = 1.0;
        world.resource_mut::<Pack>().add(def.code, 2);

        world.run_system_once(tool_keys).unwrap();

        assert_eq!(world.resource::<Pack>().count_of(def.code), 1);
        assert_eq!(world.resource::<PlayerBody>().health, 1.0 + heal);
        assert_eq!(toast_of(&world), format!("+{heal} hp"));
    }

    #[test]
    fn using_a_heal_at_full_health_refuses_rather_than_wasting_it() {
        let def = a_healing_item();
        let mut world = pressing(KEYS.use_item);
        with_a_body(&mut world);
        world.resource_mut::<PlayerBody>().health = MAX_HEALTH;
        world.resource_mut::<Pack>().add(def.code, 1);

        world.run_system_once(tool_keys).unwrap();

        assert_eq!(
            world.resource::<Pack>().count_of(def.code),
            1,
            "a full-health drink was swallowed anyway"
        );
        assert_eq!(toast_of(&world), "already at full health");
    }

    /// A buff with no heal reports the buff, not "+0 hp". The one place this
    /// port deliberately differs from `Game.tryUse` — see the note there.
    #[test]
    fn a_heal_less_draught_reports_its_effect_and_is_still_drunk() {
        let code = item_code_of("emberward_draught").expect("content lost the emberward draught");
        let mut world = pressing(KEYS.use_item);
        with_a_body(&mut world);
        world.resource_mut::<Pack>().add(code, 1);

        world.run_system_once(tool_keys).unwrap();

        assert_eq!(toast_of(&world), "fireward");
        assert_eq!(world.resource::<Pack>().count_of(code), 0);
    }

    #[test]
    fn the_use_key_on_something_you_cannot_eat_does_nothing() {
        let def = ITEM_DEFS
            .iter()
            .find(|d| d.category == ItemCategory::Material)
            .expect("content declares no material");
        let mut world = pressing(KEYS.use_item);
        with_a_body(&mut world);
        world.resource_mut::<Pack>().add(def.code, 3);

        world.run_system_once(tool_keys).unwrap();

        assert_eq!(world.resource::<Pack>().count_of(def.code), 3);
        assert_eq!(toast_of(&world), "", "a rock reported something");
    }

    /// Creative is the palette's mode and must reach neither verb — a `C` in
    /// creative is the same key that used to do nothing at all.
    #[test]
    fn neither_craft_nor_use_fires_in_creative() {
        let def = a_healing_item();
        let mut world = pressing(&[KEYS.craft[0], KEYS.use_item[0]]);
        world.resource_mut::<Tool>().creative = true;
        with_a_body(&mut world);
        world.resource_mut::<PlayerBody>().health = 1.0;
        world.resource_mut::<Pack>().add(def.code, 1);

        world.run_system_once(tool_keys).unwrap();

        assert_eq!(world.resource::<Pack>().count_of(def.code), 1);
        assert_eq!(world.resource::<PlayerBody>().health, 1.0);
        assert_eq!(toast_of(&world), "");
    }

    /// `--free-camera` runs survival with no body. Crafting still works; using
    /// has nobody to drink and must not panic reaching for one.
    #[test]
    fn the_use_key_with_no_body_at_all_is_survivable() {
        let def = a_healing_item();
        let mut world = pressing(KEYS.use_item);
        world.resource_mut::<Pack>().add(def.code, 1);

        world.run_system_once(tool_keys).unwrap();

        assert_eq!(world.resource::<Pack>().count_of(def.code), 1);
    }

    /// Every effect the content can name has a word for the toast, and the
    /// absent-effect and explicit-`none` spellings agree.
    #[test]
    fn every_authored_effect_has_a_name_and_none_has_none() {
        assert_eq!(effect_name(ItemEffect::None), None);
        for def in ITEM_DEFS.iter() {
            let Some(u) = def.r#use else { continue };
            let Some(e) = u.effect else { continue };
            if e == ItemEffect::None {
                continue;
            }
            assert!(
                effect_name(e).is_some(),
                "{} names an effect the toast cannot spell",
                def.id
            );
        }
    }
}

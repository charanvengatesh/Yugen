//! Bevy's keyboard and mouse, mapped onto what the engine understands.
//!
//! This is the whole of the `Input` class the TypeScript kept in `src/input/`.
//! That class was a DOM object and did not survive; what it PRODUCED did, and
//! it lives in two halves:
//!
//!   - [`godgame_core::input`] owns the binding table and [`Intent`]. No Bevy.
//!   - this module owns the translation: `ButtonInput<KeyCode>` into
//!     [`KeyState`], the pointer into a world-space [`Cursor`], and the wheel
//!     into brush steps.
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

use godgame_core::config::{CELL_SIZE, MAX_RUN_SPEED, PLAYER_H, PLAYER_W};
use godgame_core::input::{Intent, KEYS, KeyState};
use godgame_core::interact::{BuildTool, Cursor, Held, PALETTE_SLOTS};
use godgame_core::sim::edits::{EditMode, apply_brush};

use crate::cellmap::upload_dirty_chunks;
use crate::items::Pack;
use crate::lowres::{LowResTarget, WORLD_LAYERS};
use crate::player::PlayerBody;
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
            .init_resource::<FocusDriver>()
            .init_resource::<Tool>()
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

/// The dig/place tool. Bevy-side wrapper; all the behaviour is in the core type.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct Tool(pub BuildTool);

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
    ("Backquote", KeyCode::Backquote),
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
fn key_code(code: &str) -> Option<KeyCode> {
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
    mut cursor: ResMut<CursorWorld>,
) {
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

/// The keyboard/wheel half of `Game.handleBuildInput`: mode, palette, size.
///
/// Runs before [`swing_brush`], and that order is the TypeScript's: a stroke
/// started on the same frame as a palette change places what you just picked.
///
/// SEAM (items): the survival half of that function — hotbar selection,
/// crafting, consuming, and the wheel cycling the hotbar instead of the brush —
/// needs an inventory. Those keys are already named in [`KEYS`] (`hotbar`,
/// `craft`, `use_item`); they are left unbound rather than bound to something
/// invented.
fn tool_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    mut tool: ResMut<Tool>,
) {
    let k = BevyKeys(&keys);

    if k.any_pressed(KEYS.creative) {
        tool.toggle_creative();
        info!("creative {}", if tool.creative { "on" } else { "off" });
    }

    if tool.creative {
        for i in 0..PALETTE_SLOTS {
            if k.was_pressed(KEYS.hotbar[i]) {
                tool.select_index(i);
                info!("palette: {}", tool.selected_name());
            }
        }
    }

    // The wheel's sign is the DOM's, not Bevy's: the TypeScript accumulated
    // `Math.sign(deltaY)`, which is POSITIVE scrolling down, and fed it straight
    // to `addBrush` — so scrolling down grows the brush. Bevy reports the
    // opposite sign for the same physical motion, hence the negation. Faithful
    // rather than natural; which way a wheel grows a brush is a design decision
    // and not a porting one.
    //
    // Unread steps are dropped with the reader, which is the old `clearFrame`
    // dropping whatever no consumer took.
    let steps: i32 = wheel.read().map(|w| -w.y.signum() as i32).sum();
    if steps != 0 {
        tool.add_brush(steps);
    }
    if k.any_pressed(KEYS.brush_down) {
        tool.add_brush(-1);
    }
    if k.any_pressed(KEYS.brush_up) {
        tool.add_brush(1);
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
    pack: Res<'w, Pack>,
}

/// Ask the tool for this frame's stroke and stamp it.
fn swing_brush(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    cursor: Res<CursorWorld>,
    actor: Swinger,
    mut tool: ResMut<Tool>,
    mut world: ResMut<SimWorld>,
) {
    let Swinger { focus, body, pack } = &actor;
    let Some(at) = cursor.0 else {
        return;
    };
    let swing = Cursor {
        x: at.x,
        y: at.y,
        dig: buttons.pressed(MouseButton::Left),
        place: buttons.pressed(MouseButton::Right),
    };

    // Reach is measured from whoever is swinging: the body when there is one,
    // and the view itself under `--free-camera`, where there is no character to
    // measure from and creative's infinite reach is the only sensible answer.
    let (actor_x, actor_y) = match &body {
        Some(b) => (b.x + PLAYER_W * 0.5, b.y + PLAYER_H * 0.5),
        None => (focus.x, focus.y),
    };

    let held = {
        let inv = pack.lock();
        inv.held().map(|code| Held {
            code,
            count: u32::from(inv.held_count()),
        })
    };

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
            let mut inv = pack.lock();
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
    use godgame_core::config::View;

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
            let cell = godgame_core::config::cell_at(at.x);
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
    use godgame_core::sim::automata::Automata;
    use godgame_core::sim::chunk_store::ChunkStore;
    use godgame_core::sim::grid::CellGrid;
    use godgame_core::sim::level::Level;
    use godgame_core::sim::materials::{EMPTY, block};
    use godgame_core::sim::window::WindowManager;
    use godgame_core::sim::worldgen::SpawnPoint;

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
        // `godgame_core::interact`, against a pack and a tool profile.
        let mut build = BuildTool::new();
        build.creative = true;
        world.insert_resource(Tool(build));
        world.insert_resource(Pack::default());
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
}

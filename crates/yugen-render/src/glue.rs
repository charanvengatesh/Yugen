//! Where independently-written modules are joined.
//!
//! Everything here exists because two modules needed each other and neither
//! should own the other. `ui` draws item icons but must not know how a sprite is
//! baked; `sprite` bakes them but has never heard of a hotbar. `ui` shows a title
//! card but must stay assertable without registering a `States` type; `scenes`
//! owns the state machine but should not care what draws.
//!
//! The alternative is each module reaching into its neighbour, and the cost of
//! that is not hypothetical: three separate modules in this crate each grew their
//! own world clock, and wiring all three would have run the day at triple speed
//! (see [`crate::daynight::DayNightPlugin`]). Naming the seam and joining it in
//! one place is what keeps that from happening again.
//!
//! Each join here is a handful of lines. That is the point — a seam that needs
//! more than that is usually a boundary drawn in the wrong place.

use bevy::prelude::*;

use yugen_core::input::{KEYS, KeyState};

use yugen_core::config::STEP_DT;
use yugen_core::entities::{HitFn, Loadout, NoTargets, PlayerEvent};
use yugen_core::items::{Inventory, item_code_of};
use yugen_core::sim::save::read_run;

use crate::daynight::WorldClock;
use crate::input::{BevyKeys, FixedSubstep, PlayerIntent, Tool};
use crate::items::{GroundItems, Pack};
use crate::lowres::LowResTarget;
use crate::mobs::Creatures;
use crate::player::{ArrowPool, Juice, JuiceState, PlayerBody, PlayerSet, spend_step_events};
use crate::scenes::{Entry, Paused, Scene};
use crate::settings::Settings;
use crate::sprite::SpriteAtlases;
use crate::ui::layout::Region;
use crate::ui::menu::{self, Action, Control, Nav, Slide};
use crate::ui::{IconAtlas, Icons, RunAge, UiScreen};
use crate::world::{
    SaveNow, SimSet, SimWorld, WorldFocus, WorldSave, build_world_saved, restore_run,
};
use crate::worldselect::{AUTO_NAME, WorldPicker, fresh_seed};
use yugen_core::config::View;
use yugen_core::sim::save::{create_world, delete_world};

/// Lets the HUD draw a baked sprite without knowing what one is.
///
/// [`crate::ui`] declares [`IconAtlas`] and never names a sprite type;
/// [`crate::sprite`] publishes [`SpriteAtlases`] and never names a HUD. This is
/// the whole of the dependency between them, and it is deliberately a forward
/// with no judgement in it — the same shape as the hit-test and ammo closures in
/// [`crate::mobs`] and [`crate::items`].
///
/// A whole [`Sprite`] comes back rather than a `Handle<Image>` because a baked
/// icon is one TILE in a strip, so the atlas index is part of the answer. The
/// HUD overwrites only `custom_size` and `color`.
impl IconAtlas for SpriteAtlases {
    fn icon_cells(&self, id: &str) -> Option<(i32, i32)> {
        let baked = &self.get(id)?.baked;
        Some((baked.cells_w as i32, baked.cells_h as i32))
    }

    fn icon_sprite(&self, id: &str) -> Option<Sprite> {
        Some(self.get(id)?.still())
    }
}

/// The joins, as one plugin.
///
/// Added to the group AFTER everything it joins, so both halves of every seam
/// exist before anything reaches across one.
pub struct GluePlugin;

impl Plugin for GluePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, install_icons)
            // `.after(InputSystems)` for the reason `input::gather_intent`
            // states: that set is what repopulates `just_pressed`, and this
            // reads it.
            .add_systems(PreUpdate, escape_key.after(bevy::input::InputSystems))
            // After `escape_key`, so the frame that opens the pause card shows
            // it rather than driving it with the press that opened it.
            .add_systems(
                PreUpdate,
                (root_the_menu, drive_menu, mirror_worlds)
                    .chain()
                    .after(escape_key)
                    .run_if(menu_is_up),
            )
            .add_systems(
                FixedUpdate,
                step_the_body
                    .in_set(PlayerSet::Step)
                    .run_if(crate::scenes::running)
                    .after(SimSet::Stream)
                    .before(SimSet::Simulate)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<PlayerBody>)
                    // `PlayerPlugin` inserts both, so this cannot fail where the
                    // two above pass. It is stated anyway: a body with nowhere to
                    // put its arrows should not step at all rather than panic
                    // halfway through a frame.
                    .run_if(resource_exists::<ArrowPool>),
            )
            // Ordered after nothing in particular: the HUD reads `UiScreen` in
            // `compose`, and a frame's lag on a title card is not observable.
            // Chaining it would only add an ordering edge that has to be
            // maintained.
            .add_systems(
                Update,
                (follow_scene, death_ends_the_run, confirm_advances_the_scene)
                    .chain()
                    .run_if(resource_exists::<State<Scene>>),
            )
            .add_systems(
                Update,
                creative_is_a_truce
                    .run_if(resource_exists::<Tool>)
                    .run_if(resource_exists::<PlayerBody>),
            )
            .add_systems(OnEnter(Scene::Playing), start_a_run);
    }
}

/// One fixed step of the body, its arrows, and what those arrows hit.
///
/// # Why the whole step is here and not in `crate::player`
///
/// It reaches three modules at once — the pool it fires into, the pack it spends
/// from, the creatures its arrows resolve against — and this file exists for
/// exactly the case where several modules need each other and none should own
/// the others.
///
/// It used to be `player::step_player`, taking `ResMut<PlayerBody>` and nothing
/// else, because the other three were reachable through captured
/// `Arc<Mutex<_>>` handles installed at startup. That is what made the seams
/// locks: a `Box<dyn FnMut + 'static>` on the player had to own what it
/// captured, and Bevy could not hand a second borrow down
/// `system -> Player::step -> ProjectileSystem::update`. Asking for all four
/// borrows in one system's parameters is the thing Bevy is FOR, and it is
/// available the moment the call stack stops nesting.
///
/// # The order inside is load-bearing and is the order that used to be implicit
///
/// `Player::step` drove the pool from its own last line, so a shot fired this
/// step also integrated this step. That is preserved by making the pool's own
/// step the next statement — see [`ProjectileSystem::update`]'s header. Running
/// them the other way round would give an arrow one frame of free flight before
/// it could hit anything, and running the creatures first (they are ordered
/// `.after(PlayerSet::Step)`) would give a creature killed by an arrow one more
/// turn.
///
/// `intent.for_substep(n)` is `Game.ts`'s
/// `jumpQueued: intent.jumpQueued && steps === 0` — the rising edge goes to the
/// first substep of the frame and to no other, so a held jump key does not pogo
/// and a held dash key does not spend a dash every 8ms.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system's parameters are injected rather than passed, so the \
              count is not the call-site burden the lint exists to catch; four of \
              these are the four borrows whose absence is what forced the locks"
)]
fn step_the_body(
    mut body: ResMut<PlayerBody>,
    mut arrows: ResMut<ArrowPool>,
    creatures: Option<ResMut<Creatures>>,
    pack: Option<ResMut<Pack>>,
    world: Res<SimWorld>,
    intent: Res<PlayerIntent>,
    substep: Res<FixedSubstep>,
    mut juice: Juice,
    mut state: Local<JuiceState>,
    mut drained: Local<Vec<PlayerEvent>>,
) {
    let grid = &world.level.grid;
    let mut pack = pack;
    let mut creatures = creatures;

    // The body's armour, from whatever the pack says is worn. Recomputed here
    // rather than written when the equipment changes, and that is the point: a
    // stat cached at the moment of equipping is a stat that disagrees with the
    // pack the first time anything else touches it — a save restore, a death, a
    // creative reset. One line a frame, and the two cannot drift.
    if let Some(pack) = &pack {
        body.0.armour = crate::input::worn_armour(&pack.0);
    }

    // The pool and the pack, lent for the length of the call and no longer.
    // `Inventory` IS an `AmmoSource` and `ProjectileSystem` IS a `Projectiles`,
    // so there is no adapter here — which is the whole point of both traits
    // being one method wide.
    //
    // No pack means no ammo half, which fires FREELY rather than not at all. That
    // is `AmmoSource`'s own documented fallback and it is what keeps a host that
    // runs `PlayerPlugin` without `ItemsPlugin` playable instead of silently
    // unable to shoot — the same reason `start_a_run` takes every one of these as
    // an `Option`.
    {
        let mut kit = Loadout::new(&mut **arrows);
        if let Some(pack) = pack.as_deref_mut() {
            kit = kit.with_ammo(&mut **pack);
        }
        body.step(STEP_DT, intent.for_substep(substep.0), grid, &mut kit);
    }

    // The very next line, as `ProjectileSystem::update` requires. The creatures
    // are the world these shots are passing through; `MobSystem::hit_at` has the
    // `ShotWorld::hit` signature exactly, because that is the signature it was
    // shaped for, so the closure is a forward with no judgement of its own in it.
    //
    // No creatures means [`NoTargets`] — arrows still fly and still stop on rock,
    // there is simply nothing alive for them to find. A capture test that runs the
    // world and the body without `MobsPlugin` gets exactly that.
    match creatures.as_deref_mut() {
        Some(creatures) => {
            let creatures = &mut **creatures;
            arrows.update(
                STEP_DT,
                grid,
                &mut HitFn(|x, y, damage, knockback, dir_x, dir_y| {
                    creatures.hit_at(x, y, damage, knockback, dir_x, dir_y)
                }),
            );
        }
        None => arrows.update(STEP_DT, grid, &mut NoTargets),
    }

    drained.clear();
    body.drain_events(&mut drained);
    spend_step_events(&mut body, grid, &drained, &mut juice, &mut state);
}

/// Creative mode makes the body untouchable.
///
/// # This is a DELIBERATE DIVERGENCE, and the seam is where it is admitted
///
/// The TypeScript's creative mode was an infinite build palette and nothing
/// else — lava still ate you and everything with teeth still came for you. The
/// truce is new. `yugen_core::entities::player::Player::untouchable` and
/// `yugen_core::entities::mobs::MobTarget::targetable` each carry the full
/// argument; what is worth saying HERE is that the divergence is not a change to
/// either of those modules. Both were written knowing nothing about creative
/// mode: one exposes a flag, the other asks a question. The decision that
/// creative should ANSWER that question is this line, and it is the only line in
/// the tree that holds it.
///
/// # Why it is a mirror and not a shared field
///
/// [`Tool`] is the build brush: reach, palette, brush size, cadence.
/// [`PlayerBody`] is a simulated body with a health bar. Neither has any
/// business importing the other, and the alternative — `MobSystem` reaching for
/// a `Res<Tool>` to ask whether it is allowed to bite — would put a UI mode
/// inside a combat resolver, which is exactly the reach-into-your-neighbour this
/// module exists to prevent.
///
/// # Timing
///
/// Unordered in `Update`, like [`follow_scene`]. Bevy runs `FixedUpdate` before
/// `Update`, so a toggle is honoured by the creatures on the next frame's fixed
/// step at the latest — under 17ms, and the frame it costs is one in which a
/// creature that was already mid-lunge lands its blow. That reads as the hit
/// that was already coming, which is the more forgiving of the two ways to be
/// wrong; buying the frame back would mean an ordering edge into
/// [`crate::input`]'s private system chain, maintained forever, for something
/// nobody can perceive.
///
/// Written only when it differs, `set_if_neq` by hand: [`PlayerBody`] is touched
/// by the fixed step every frame anyway, and this system has no business adding
/// a change tick of its own on top.
fn creative_is_a_truce(tool: Res<Tool>, mut body: ResMut<PlayerBody>) {
    if body.untouchable != tool.creative {
        body.untouchable = tool.creative;
    }
}

/// A dead body ends the run.
///
/// `Game.updatePlaying`'s two lines: `if (this.player.dead) to(GameOver)`. It is
/// here rather than in [`crate::player`] for the same reason everything else in
/// this module is — the body knows it is dead and the scene machine knows what
/// that means, and neither should have to import the other.
///
/// Ordered before [`confirm_advances_the_scene`] so that the frame you die on
/// cannot also be the frame a held confirm key restarts you: the transition into
/// `GameOver` is queued first, and `confirm` reads the state it was in, not the
/// one it is going to.
fn death_ends_the_run(
    body: Option<Res<PlayerBody>>,
    scene: Res<State<Scene>>,
    mut next: ResMut<NextState<Scene>>,
    mut save: ResMut<SaveNow>,
) {
    if *scene.get() != Scene::Playing {
        return;
    }
    if body.is_some_and(|b| b.dead()) {
        // Ask for a save BEFORE the transition, so what reaches disk is the
        // world as the death left it. `docs/SAVE.md` states the rule: death is
        // the most durable moment in the game, and there is exactly one Quit.
        // Without this a player learns they can rewind a death by killing the
        // process, and once that is learnable it is the correct play — so the
        // affordance is removed rather than policed.
        //
        // When `docs/DEATH.md` lands, the corpse bag is spawned and the pack
        // cleared before this point, and the request needs no change: it will
        // already be writing the post-death state.
        save.request();
        next.set(Scene::GameOver);
    }
}

/// Start a run, or continue one: see [`Entry`].
///
/// This is `Game.loadLevel` plus the `reset` calls around it, and it still runs
/// on EVERY entry to [`Scene::Playing`]. What it no longer does is mean the same
/// thing every time.
///
/// A [`Entry::Restart`] is what this function always was: regenerate the world,
/// reset the body, clear everything loose, grant the starting kit, and then put
/// back whatever the save file holds. The first entry out of the menu and a
/// restart take the same path deliberately — a special case for "the first one"
/// is a second code path that only ever runs once and so is never really tested,
/// and it is safe because everything it clears is already empty then.
///
/// A [`Entry::Respawn`] is `docs/DEATH.md`: the body comes back and the WORLD
/// DOES NOT MOVE. No regenerate, no ground clear, no starting kit, no restore
/// from disk — the world in memory is already the one the player died in, and it
/// is the thing they are being sent back for. Reading the save here would be
/// worse than pointless: it would reload the state written a moment earlier by
/// `death_ends_the_run` and undo nothing, at the cost of pretending the two paths
/// are the same.
///
/// The restart path REGENERATES rather than repairs, so it discards the player's
/// excavation — see [`build_world`]. That is the original's behaviour, which
/// allocated a fresh `CellGrid` in `loadLevel`. It is now the behaviour of
/// starting a new run rather than of dying, which is the whole point of the
/// split.
///
/// `OnEnter` and not a system that watches the state: Bevy re-fires `OnEnter`
/// when a state is `set` to the value it already holds, which would rebuild the
/// world under a live player. [`crate::scenes`] documents that trap and
/// Everything a fresh run resets, plus where its world is kept. Bundled so the
/// system stays inside clippy's argument budget — the same `SystemParam` trick
/// `light`'s solve and `debug`'s gather use.
#[derive(bevy::ecs::system::SystemParam)]
struct RunState<'w> {
    body: Option<ResMut<'w, PlayerBody>>,
    arrows: Option<ResMut<'w, ArrowPool>>,
    creatures: Option<ResMut<'w, Creatures>>,
    ground: Option<ResMut<'w, GroundItems>>,
    pack: Option<ResMut<'w, Pack>>,
    save: Res<'w, WorldSave>,
    clock: Option<ResMut<'w, WorldClock>>,
}

/// [`confirm_advances_the_scene`] is the only thing that sets this state.
fn start_a_run(
    mut commands: Commands,
    mut focus: ResMut<WorldFocus>,
    mut run: RunState,
    mut paused: ResMut<Paused>,
    mut age: ResMut<RunAge>,
    mut entry: ResMut<Entry>,
) {
    // Read once and put back to the default in the same breath. The intent
    // belongs to ONE transition: leaving `Respawn` set would make the next
    // entry — a quit to the menu and a new game — silently keep the dead run's
    // world. Consuming it means a caller that forgets to speak gets `Restart`,
    // which is the safe reading. See `Entry`.
    let entering = std::mem::take(&mut *entry);

    // Either way the run is unpaused, with its hint clock back at full: a
    // restart must not inherit the faded hints of the run that just ended, and
    // an entry made while the pause card was up would resume into a stopped
    // world. A respawn wants both for the same reasons.
    *paused = Paused(false);
    *age = RunAge(0.0);

    if entering == Entry::Respawn {
        // Everything below this point either rebuilds the world or clears
        // something standing in it, and a respawn wants none of it. The body is
        // the only thing that comes back.
        if let Some(body) = &mut run.body {
            body.reset();
            *focus = WorldFocus {
                x: body.x,
                y: body.y,
            };
        }
        if let Some(arrows) = &mut run.arrows {
            // Arrows belong to the life that fired them, and that life is over
            // — the same argument the restart path makes below, and the one
            // reset a respawn does share.
            arrows.clear();
        }
        return;
    }

    let RunState {
        body,
        arrows,
        creatures,
        ground,
        pack,
        save,
        clock,
    } = &mut run;
    // Through `WorldSave` and not `build_world`, which is the unsaved path.
    // Missing this meant `--world` opened a directory, logged it, and then the
    // FIRST transition into `Playing` threw that world away and built an
    // unsaved one over the top. 22 chunks were "persisted" — into memory — and
    // not one file appeared on disk, while every log line said it had worked.
    let world = build_world_saved(save.seed, save.dir.as_deref());
    *focus = WorldFocus {
        x: world.level.spawn.x,
        y: world.level.spawn.y,
    };
    commands.insert_resource(world);

    // Each of these is optional because a caller may run `UiPlugin` and the
    // scene machine without the whole game — the capture test and the examples
    // both do. A missing one means that system is not in the app, not that
    // something failed.
    if let Some(body) = body {
        body.reset();
    }
    if let Some(arrows) = arrows {
        // Every arrow in flight belongs to the life that fired it. `reset` used
        // to do this, back when the body owned its pool; it does not own one now,
        // so the drop is here, beside the reset it belongs to.
        arrows.clear();
    }
    if let Some(creatures) = creatures {
        creatures.clear();
    }
    if let Some(ground) = ground {
        ground.clear();
    }
    if let Some(pack) = pack {
        give_starting_kit(pack);
    }

    // And then, if this world has been played before, put back what was left.
    //
    // LAST, after every reset above, and that order is the whole of it: a
    // restore that ran first would be undone by `body.reset()` and buried under
    // the starting kit, and it would look like the save had not been written.
    let Some(dir) = save.dir.as_deref() else {
        return;
    };
    let Some(loaded) = read_run(dir, save.seed) else {
        return;
    };
    info!(
        "world save: resuming a run — clock {:.2}, {} filled slots",
        loaded.clock_t,
        loaded.slots.len()
    );
    restore_run(
        &loaded,
        &mut focus,
        body.as_deref_mut(),
        pack.as_deref_mut(),
        clock.as_deref_mut(),
    );
}

/// What you wake up with.
///
/// `Game.giveStartingKit`, verbatim, and its reasoning is worth keeping: three
/// items and no blocks — the Traveler's Pick (which clears everything down to
/// copper), the Traveler Sword, and enough bandages to survive the first
/// mistake. Everything else in the game is something you dug up.
///
/// An id the registry does not have is skipped rather than fatal, on the same
/// terms as a creature's loot row: a content gap should cost you the item, not
/// the run. It cannot happen with these three — [`the_starting_kit_is_real_items`]
/// proves all three resolve — which is precisely why it is safe to be lenient.
fn give_starting_kit(inv: &mut Inventory) {
    const KIT: [(&str, u32); 3] = [("pick_traveler", 1), ("sword_traveler", 1), ("bandage", 3)];

    inv.clear();
    for (id, n) in KIT {
        if let Some(code) = item_code_of(id) {
            inv.add(code, n);
        }
    }
    // The pick, so the first thing you can do is dig.
    inv.select_slot(0);
}

/// Keep [`menu::WorldRows`] in step with the picker.
///
/// Only when the picker changes, which is on entering the world list and after
/// making or deleting one — so the clone is paid three times a session rather
/// than sixty times a second.
fn mirror_worlds(picker: Res<WorldPicker>, mut rows: ResMut<menu::WorldRows>) {
    if !picker.is_changed() {
        return;
    }
    rows.0 = picker
        .worlds
        .iter()
        .map(|w| menu::WorldRow {
            name: w.name.clone(),
            seed: w.seed,
        })
        .collect();
}

/// Is a menu on screen at all?
///
/// The title card always, and a paused world. Not the world list, which is its
/// own screen with its own keys — see `worldselect`.
fn menu_is_up(scene: Res<State<Scene>>, paused: Res<Paused>) -> bool {
    match scene.get() {
        Scene::Menu => true,
        Scene::Playing => paused.0,
        Scene::WorldSelect | Scene::GameOver => false,
    }
}

/// Keep [`Nav`] rooted at the card the current scene calls for.
///
/// Without this, opening Options from the pause card, quitting to the title and
/// pressing Escape would pop back into a pause card over a world that is no
/// longer there. The root follows the scene; the stack above it is the
/// player's.
fn root_the_menu(scene: Res<State<Scene>>, paused: Res<Paused>, mut nav: ResMut<Nav>) {
    let want = match scene.get() {
        Scene::Playing if paused.0 => menu::Page::Pause,
        Scene::Menu => menu::Page::Title,
        _ => return,
    };
    if nav.depth() == 1 && nav.top() != want {
        nav.reset(want);
    }
}

/// Everything the menus do: move, press, drag, and go back.
///
/// # Why this is one system and lives in `glue`
///
/// `ui::menu` owns the pages, the widgets and the arithmetic, and every bit of
/// that is pure — it can be driven in a test with no app. What it cannot own is
/// the CONSEQUENCES: "Singleplayer" is a scene change, "Back to Game" is a
/// resource, "Quit Game" is an `AppExit`, and "Fullscreen" is a window. Those
/// are four different halves of the game and joining them is what this module
/// is for.
///
/// The whole interaction is here rather than split across a keyboard system and
/// a mouse one, because both produce the same [`Action`]s and a second copy of
/// the action table is how a menu ends up doing one thing on click and another
/// on Enter.
#[allow(clippy::too_many_arguments)]
fn drive_menu(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    target: Option<Res<LowResTarget>>,
    mut nav: ResMut<Nav>,
    mut settings: ResMut<Settings>,
    mut picker: ResMut<WorldPicker>,
    mut save: ResMut<WorldSave>,
    mut paused: ResMut<Paused>,
    mut next: ResMut<NextState<Scene>>,
    scene: Res<State<Scene>>,
    mut exit: MessageWriter<AppExit>,
) {
    let in_game = *scene.get() == Scene::Playing;
    let worlds: Vec<menu::WorldRow> = picker
        .worlds
        .iter()
        .map(|w| menu::WorldRow {
            name: w.name.clone(),
            seed: w.seed,
        })
        .collect();
    let rows = menu::page_rows(
        nav.top(),
        &menu::MenuCx {
            settings: &settings,
            in_game,
            worlds: &worlds,
            confirming: picker.confirming.then_some(picker.cursor),
        },
    );
    if rows.is_empty() {
        return;
    }
    // Land the cursor somewhere it can act, before anything reads it.
    //
    // `Nav::push` starts a page at row 0, and row 0 is not always pressable: an
    // empty world list opens on "No worlds yet", and the Controls page is
    // fifteen bindings before its "Done". Pressing Enter then did nothing at
    // all, which is exactly how the world list looked like a dead end. Done
    // every frame rather than on push, because a page's rows change underneath
    // the cursor — deleting the last world is the case that proves it.
    if rows
        .get(nav.focus())
        .is_none_or(|r| !r.control.selectable())
    {
        nav.focus_on(menu::first_selectable(&rows));
    }

    let bevy_keys = BevyKeys(&keys);
    let view = target.as_ref().map(|t| t.view);

    // --- The pointer, in buffer pixels --------------------------------------
    //
    // The buffer is upscaled to fill the window, so this is one division. It is
    // `None` whenever there is no window, no cursor in it, or no target yet,
    // and every mouse branch below is skipped rather than guessing at (0, 0).
    let pointer = view.and_then(|view| {
        let window = windows.iter().next()?;
        let p = window.cursor_position()?;
        let (ww, wh) = (window.width(), window.height());
        (ww > 0.0 && wh > 0.0).then(|| {
            IVec2::new(
                (p.x / ww * view.w as f32) as i32,
                (p.y / wh * view.h as f32) as i32,
            )
        })
    });

    // Hover moves the cursor, as it does in every menu this is modelled on.
    if let (Some(view), Some(p)) = (view, pointer)
        && let Some(row) = menu::row_at(view, nav.top(), rows.len(), nav.focus(), p.x, p.y)
        && rows[row].control.selectable()
    {
        nav.focus_on(row);
    }

    // --- Moving -------------------------------------------------------------
    if bevy_keys.any_pressed(KEYS.jump) {
        let to = menu::step_focus(&rows, nav.focus(), -1);
        nav.focus_on(to);
    }
    if bevy_keys.any_pressed(KEYS.down) {
        let to = menu::step_focus(&rows, nav.focus(), 1);
        nav.focus_on(to);
    }

    // --- Dragging a slider --------------------------------------------------
    //
    // Held rather than just-pressed, so the handle follows the mouse. Left and
    // right nudge it by a step, which is the only way to set one precisely and
    // the only way to set one at all without a mouse.
    let focused = rows.get(nav.focus());
    if let Some(Control::Slider { of, at, .. }) = focused.map(|r| &r.control) {
        let (of, at) = (*of, *at);
        if let (Some(view), Some(p)) = (view, pointer)
            && buttons.pressed(MouseButton::Left)
            && let Some(row) = menu::row_at(view, nav.top(), rows.len(), nav.focus(), p.x, p.y)
            && row == nav.focus()
        {
            let r = drag_rect(view, nav.top(), rows.len(), nav.focus());
            of.set(&mut settings, menu::slider_at(r, p.x));
        }
        let mut nudge = 0.0;
        if bevy_keys.any_pressed(KEYS.left) {
            nudge -= Slide::KEY_STEP;
        }
        if bevy_keys.any_pressed(KEYS.right) {
            nudge += Slide::KEY_STEP;
        }
        if nudge != 0.0 {
            of.set(&mut settings, at + nudge);
        }
    }

    // --- Deleting -----------------------------------------------------------
    //
    // `X` on a world row asks; the row then becomes the confirmation and a
    // press carries it out. Two steps, as `worldselect` always had, because
    // this is the only irreversible thing a player can do from a menu.
    if nav.top() == menu::Page::Worlds && keys.just_pressed(KeyCode::KeyX) {
        let focus = nav.focus();
        if focus < picker.worlds.len() {
            picker.cursor = focus;
            picker.confirming = true;
        }
        return;
    }

    // --- Pressing -----------------------------------------------------------
    let clicked = pointer.is_some()
        && buttons.just_pressed(MouseButton::Left)
        && view.zip(pointer).is_some_and(|(view, p)| {
            menu::row_at(view, nav.top(), rows.len(), nav.focus(), p.x, p.y) == Some(nav.focus())
        });
    let pressed = bevy_keys.any_pressed(KEYS.confirm) || clicked;

    if pressed && let Some(action) = rows.get(nav.focus()).and_then(menu::activate) {
        do_menu_action(
            action,
            &mut Menu {
                nav: &mut nav,
                settings: &mut settings,
                paused: &mut paused,
                picker: &mut picker,
                save: &mut save,
                next: &mut next,
                exit: &mut exit,
            },
        );
    }

    // Going back is `escape_key`'s, and deliberately not also this system's —
    // see its header for what happened when both read the key.
}

/// Where the focused row was drawn, for the slider drag.
fn drag_rect(view: View, page: menu::Page, len: usize, focus: usize) -> Region {
    let (first, count) = menu::visible_window(view, page, len, focus);
    menu::slot_rect(view, page, count, focus.saturating_sub(first))
}

/// Carry out one menu [`Action`].
///
/// Split out so the navigation and the world actions can be tested through one
/// entry point — `menu::apply_to_settings` covers the settings half purely, and
/// this is the half that needs the app.
/// Everything [`do_menu_action`] can reach. Bundled to stay inside clippy's
/// argument budget, the way `debug::Sources` and `settings::Applied` are.
struct Menu<'a, 'w> {
    nav: &'a mut Nav,
    settings: &'a mut Settings,
    paused: &'a mut Paused,
    picker: &'a mut WorldPicker,
    save: &'a mut WorldSave,
    next: &'a mut NextState<Scene>,
    exit: &'a mut MessageWriter<'w, AppExit>,
}

/// Carry out one menu [`Action`].
fn do_menu_action(action: Action, m: &mut Menu) {
    if menu::apply_to_settings(action, m.settings) {
        return;
    }
    match action {
        Action::Open(page) => {
            // The world list is read off the disk on the way in, so a world
            // made or deleted in another run shows up without a restart.
            if page == menu::Page::Worlds {
                m.picker.refresh();
                m.picker.confirming = false;
            }
            m.nav.push(page);
        }
        Action::Back => {
            m.nav.pop();
        }
        Action::Resume => *m.paused = Paused(false),
        Action::NewWorld => {
            let name = format!("{AUTO_NAME} {}", m.picker.worlds.len() + 1);
            let root = m.picker.root.clone();
            // Permadeath is off until there is a UI to choose it at creation.
            // It is a world property, so this is the ONLY place it can be set,
            // and a menu row is a separate change.
            match create_world(root, &name, fresh_seed(), false) {
                Ok(made) => {
                    m.picker.refresh();
                    // Land on what was just made, wherever the recency sort put
                    // it, so the next press plays it.
                    m.picker.cursor = m
                        .picker
                        .worlds
                        .iter()
                        .position(|w| w.dir == made.dir)
                        .unwrap_or(0);
                    m.nav.focus_on(m.picker.cursor);
                    m.picker.error = None;
                }
                Err(e) => m.picker.error = Some(format!("could not create: {e}")),
            }
        }
        Action::PlayWorld(i) => {
            let Some(world) = m.picker.worlds.get(i) else {
                return;
            };
            m.picker.cursor = i;
            m.save.seed = world.seed;
            m.save.dir = Some(world.dir.clone());
            m.picker.confirming = false;
            m.nav.reset(menu::Page::Title);
            m.next.set(Scene::Playing);
        }
        Action::DeleteWorld(i) => {
            // Only ever reached from a row that already asked once — the row
            // itself is the confirmation, so by here the answer is yes.
            let Some(world) = m.picker.worlds.get(i).cloned() else {
                m.picker.confirming = false;
                return;
            };
            let root = m.picker.root.clone();
            match delete_world(root, &world) {
                Ok(()) => {
                    m.picker.refresh();
                    m.picker.error = None;
                }
                Err(e) => m.picker.error = Some(format!("could not delete: {e}")),
            }
            m.picker.confirming = false;
            m.nav.focus_on(0);
        }
        Action::Quit => {
            *m.paused = Paused(false);
            m.nav.reset(menu::Page::Title);
            m.next.set(Scene::Menu);
        }
        Action::Exit => {
            m.exit.write(AppExit::Success);
        }
        Action::Flip(_) | Action::Step(_) => {}
    }
}

/// `Esc`: open the pause card, go back a page, or resume.
///
/// # Why this is one system and not two
///
/// It was two, and the two fought over the same key press. `toggle_pause` set
/// `Paused` in `PreUpdate`, which made `menu_is_up` true, which let `drive_menu`
/// run LATER IN THE SAME FRAME and see the very same `just_pressed` Escape — so
/// it popped the stack it had just been given, found the root, and resumed.
/// Pressing Escape opened the pause card and closed it before a frame was
/// drawn, and the only visible symptom was that pause did not work at all.
///
/// Two systems reading one key and both acting on it is the bug. There is one
/// reader now, and the whole meaning of Escape is the match below:
///
///   - playing, not paused  -> open the pause card
///   - paused, above the root -> go back a page
///   - paused, at the root  -> resume
///   - on the title card    -> go back a page, or nothing at the root
///
/// `drive_menu` no longer looks at `KEYS.pause` at all.
fn escape_key(
    keys: Res<ButtonInput<KeyCode>>,
    scene: Res<State<Scene>>,
    mut paused: ResMut<Paused>,
    mut nav: ResMut<Nav>,
    mut save: ResMut<SaveNow>,
) {
    if !BevyKeys(&keys).any_pressed(KEYS.pause) {
        return;
    }
    match scene.get() {
        Scene::Playing => {
            if !paused.0 {
                paused.0 = true;
                // The moment before an alt-F4. Pausing is the closest thing to
                // a player saying "I am stopping now", and it costs one write.
                save.request();
                // Rooted here rather than left wherever the last menu was, so
                // Escape always opens ON the pause card.
                nav.reset(menu::Page::Pause);
            } else if !nav.pop() {
                paused.0 = false;
            }
        }
        // The title card has nowhere to go back to from its root, and `pop`
        // says so by returning false, which is exactly the right amount of
        // nothing to do.
        Scene::Menu => {
            nav.pop();
        }
        // `worldselect` owns its own keys, and the death card has one action.
        Scene::WorldSelect | Scene::GameOver => {}
    }
}

/// `Enter` or `Space` leaves the menu, and leaves the death screen.
///
/// This is `Game.updateMenu`/`updateGameOver`'s one line each, and it lives here
/// for a blunt reason: [`crate::ui`] draws a screen that says "Press Enter or
/// Space to start", [`crate::scenes`] owns the state machine that would honour
/// it, and neither can reach the other. Without this the title card is a promise
/// the build does not keep — the game boots into a menu nothing can dismiss.
///
/// [`KEYS`] is the ported binding table, so the keys are the game's, not this
/// module's opinion of them.
///
/// The respawn the TypeScript's `updateGameOver` did — `loadLevel()` before the
/// switch, rebuilding the world and refilling the pack — hangs off
/// `OnEnter(Scene::Playing)` and is [`start_a_run`], forty lines above. This is
/// only the key press, and it is deliberately the ONLY thing in the tree that
/// sets this state: see [`Scene`]'s docs on why Bevy re-fires `OnEnter` when a
/// state is `set` to the value it already holds, which would rebuild the world
/// under a live player.
fn confirm_advances_the_scene(
    keys: Res<ButtonInput<KeyCode>>,
    scene: Res<State<Scene>>,
    mut next: ResMut<NextState<Scene>>,
    mut entry: ResMut<Entry>,
) {
    if !BevyKeys(&keys).any_pressed(KEYS.confirm) {
        return;
    }
    match scene.get() {
        // The menu asks WHICH world before starting one. Game over does not:
        // dying and pressing confirm is a restart of the run you were in, and
        // sending the player back to a directory listing to do it would be a
        // different game.
        //
        // `drive_menu` owns the title card now, including what confirm does
        // on it. This must not also act, or pressing Enter on "Singleplayer"
        // would advance the scene twice.
        Scene::Menu => {}
        Scene::GameOver => {
            // The one place in the tree that says `Respawn`. Everything else
            // entering `Playing` — the world picker, `--play`, every capture
            // harness — leaves the default alone and gets a rebuilt world, which
            // is what each of them wants. See `scenes::Entry`.
            *entry = Entry::Respawn;
            next.set(Scene::Playing);
        }
        // `WorldSelect` reads confirm itself — see `crate::worldselect` — and
        // this must not also act on it, or picking a world would start the
        // PREVIOUS one on the same keypress.
        Scene::WorldSelect | Scene::Playing => {}
    }
}

/// Hand the HUD the baked atlas.
///
/// `Startup` is late enough: [`crate::sprite::SpritePlugin`] bakes in
/// `PreStartup` precisely so that any `Startup` system can take
/// `Res<SpriteAtlases>` without an `.after()` edge.
///
/// If this never ran, the HUD would still draw — [`Icons`] empty means every
/// swatch takes the flat-colour path, which is the state the TypeScript hotbar
/// actually shipped in. That is a real fallback rather than a broken frame,
/// which is why this is allowed to be a plain unordered system.
fn install_icons(mut commands: Commands, atlases: Res<SpriteAtlases>) {
    commands.insert_resource(Icons(Some(Box::new((*atlases).clone()))));
}

/// Mirror the scene machine onto the screen the HUD draws.
///
/// Two types rather than one, and [`UiScreen`]'s doc comment argues the case:
/// it defaults to `Playing` so a `UiPlugin` on its own draws a usable HUD rather
/// than a title card nothing can dismiss, while [`Scene`] defaults to `Menu`,
/// which is right for a scene machine. Collapsing them would force one of those
/// two defaults to be wrong.
fn follow_scene(scene: Res<State<Scene>>, mut screen: ResMut<UiScreen>) {
    let want = match scene.get() {
        Scene::Menu => UiScreen::Menu,
        Scene::WorldSelect => UiScreen::WorldSelect,
        Scene::Playing => UiScreen::Playing,
        Scene::GameOver => UiScreen::GameOver,
    };
    // `set_if_neq` semantics by hand: `UiScreen` is a plain resource, so writing
    // it every frame would mark it changed every frame for anything watching.
    if *screen != want {
        *screen = want;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use yugen_core::entities::player::Player;
    use yugen_core::interact::BuildTool;
    use yugen_core::sim::worldgen::SpawnPoint;

    #[test]
    fn every_scene_has_a_screen_and_the_mapping_is_total() {
        // The match in `follow_scene` is exhaustive by construction, so what is
        // worth pinning is that the two enums have not silently drifted apart —
        // a new `Scene` variant must be a compile error there, not a screen that
        // never appears.
        for (scene, want) in [
            (Scene::Menu, UiScreen::Menu),
            (Scene::WorldSelect, UiScreen::WorldSelect),
            (Scene::Playing, UiScreen::Playing),
            (Scene::GameOver, UiScreen::GameOver),
        ] {
            let got = match scene {
                Scene::Menu => UiScreen::Menu,
                Scene::WorldSelect => UiScreen::WorldSelect,
                Scene::Playing => UiScreen::Playing,
                Scene::GameOver => UiScreen::GameOver,
            };
            assert_eq!(got, want, "{scene:?} maps to the wrong screen");
        }
    }

    /// Press confirm in `scene`, and report what it asked for.
    fn confirm_from(scene: Scene) -> (Option<Scene>, Entry) {
        let mut world = World::new();
        let mut keys = ButtonInput::<KeyCode>::default();
        keys.press(KeyCode::Enter);
        world.insert_resource(keys);
        world.insert_resource(State::new(scene));
        world.insert_resource(NextState::<Scene>::default());
        world.insert_resource(Entry::default());
        world
            .run_system_once(confirm_advances_the_scene)
            .expect("the system runs");

        let next = match world.resource::<NextState<Scene>>() {
            NextState::Pending(s) | NextState::PendingIfNeq(s) => Some(*s),
            NextState::Unchanged => None,
        };
        (next, *world.resource::<Entry>())
    }

    #[test]
    fn only_the_death_card_asks_for_a_respawn() {
        // The whole of `docs/DEATH.md` step 1 rests on exactly one transition
        // declaring itself. If a second one ever starts saying `Respawn`, it
        // will keep a world that its caller expected to be rebuilt — and that
        // failure is invisible until somebody notices their old tunnels under a
        // new game.
        assert_eq!(
            confirm_from(Scene::GameOver),
            (Some(Scene::Playing), Entry::Respawn)
        );

        // And every other scene neither advances on confirm nor touches the
        // intent. `Menu` and `WorldSelect` are handled by their own modules, and
        // `Playing` has nothing to confirm.
        for scene in [Scene::Menu, Scene::WorldSelect, Scene::Playing] {
            let (next, entry) = confirm_from(scene);
            assert_eq!(next, None, "{scene:?} advanced the scene on confirm");
            assert_eq!(entry, Entry::Restart, "{scene:?} asked for a respawn");
        }
    }

    #[test]
    fn an_entry_that_says_nothing_rebuilds_the_world() {
        // The default is the DESTRUCTIVE reading on purpose. Eleven capture
        // harnesses, `--play` and the world picker all set `Playing` without
        // mentioning `Entry`, and every one of them wants a fresh world. A
        // caller that forgets gets the behaviour that was correct before the
        // split; the opposite default would leave a dead run's world standing
        // under a new game, which nothing would report.
        assert_eq!(Entry::default(), Entry::Restart);

        // `start_a_run` consumes it with `mem::take`, so the intent belongs to
        // one transition and cannot leak into the next entry.
        let mut entry = Entry::Respawn;
        assert_eq!(std::mem::take(&mut entry), Entry::Respawn);
        assert_eq!(entry, Entry::Restart, "the intent was not consumed");
    }

    #[test]
    fn the_starting_kit_is_real_items() {
        // `give_starting_kit` skips an id the registry does not have, which is
        // the right leniency for content that may lag code — but it means a
        // renamed item would silently hand you an empty pack. These three are
        // the run's whole opening position, so they get checked by name.
        for id in ["pick_traveler", "sword_traveler", "bandage"] {
            assert!(
                item_code_of(id).is_some(),
                "the starting kit names {id:?}, which content/items/ does not define"
            );
        }
    }

    #[test]
    fn waking_up_gives_a_pick_a_sword_and_bandages_and_nothing_else() {
        let mut inv = Inventory::default();
        // Something left over from a previous life, to prove the clear happens.
        let junk = item_code_of("bandage").expect("bandage is in the registry");
        inv.add(junk, 99);

        give_starting_kit(&mut inv);

        assert_eq!(inv.count_of(item_code_of("pick_traveler").unwrap()), 1);
        assert_eq!(inv.count_of(item_code_of("sword_traveler").unwrap()), 1);
        assert_eq!(
            inv.count_of(junk),
            3,
            "the previous life's 99 bandages must not carry over"
        );
        assert_eq!(inv.selected(), 0, "you wake up holding the pick");
    }

    #[test]
    fn a_restart_is_the_same_world_as_a_first_start() {
        // `start_a_run` regenerates rather than repairing, and the whole reason
        // that is safe is that a world is a pure function of its seed. If it ever
        // stops being one, a restart would silently drop you somewhere else.
        let first = build_world_saved(yugen_core::config::SEED, None);
        let again = build_world_saved(yugen_core::config::SEED, None);
        assert_eq!(first.level.spawn.x, again.level.spawn.x);
        assert_eq!(first.level.spawn.y, again.level.spawn.y);
        assert_eq!(first.seed, again.seed);
    }

    #[test]
    fn creative_makes_the_body_untouchable_and_survival_hands_it_back() {
        // The join, driven through a real app so what is under test is the
        // system and its run conditions rather than a copy of its two lines.
        let mut app = App::new();
        app.init_resource::<Tool>();
        app.insert_resource(PlayerBody(Player::new(SpawnPoint { x: 0.0, y: 0.0 })));
        app.add_systems(Update, creative_is_a_truce);

        assert!(
            !app.world().resource::<Tool>().creative,
            "survival by default"
        );
        app.update();
        assert!(
            !app.world().resource::<PlayerBody>().untouchable,
            "the default must not leak: every existing suite describes a world \
             where lava and teeth still work"
        );

        app.world_mut().resource_mut::<Tool>().toggle_creative();
        app.update();
        assert!(app.world().resource::<PlayerBody>().untouchable);

        // And it is a mirror, not a latch: leaving creative ends the truce.
        app.world_mut().resource_mut::<Tool>().toggle_creative();
        app.update();
        assert!(!app.world().resource::<PlayerBody>().untouchable);
    }

    #[test]
    fn the_truce_is_a_deliberate_divergence_with_no_original_to_match() {
        // Not a behaviour test — a claim about the two sides of the seam, which
        // is the thing that would rot silently. `Tool` is a build brush and
        // `Player` is a body; if either ever grew the other's concept, this join
        // would be dead code and the coupling it prevents would be back.
        //
        // The TypeScript's creative mode changed the PALETTE and nothing else.
        // Its default is the one thing that has to stay true, because everything
        // ported into this tree was measured against a world in which hazards
        // and creatures do not care what you are holding.
        assert!(
            !BuildTool::default().creative,
            "creative is not the default mode"
        );
        assert!(
            !Player::new(SpawnPoint { x: 0.0, y: 0.0 }).untouchable,
            "the body is mortal until this module says otherwise"
        );
    }

    #[test]
    fn the_two_defaults_are_deliberately_different() {
        // If these ever agree, one of the two arguments in `UiScreen`'s docs has
        // stopped being true and this join can be deleted. Until then the
        // difference is load-bearing: a bare `UiPlugin` draws a HUD, and a wired
        // scene machine opens on the menu.
        assert_eq!(UiScreen::default(), UiScreen::Playing);
        assert_eq!(Scene::default(), Scene::Menu);
    }
}

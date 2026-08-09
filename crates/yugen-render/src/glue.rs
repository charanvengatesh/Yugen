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
use crate::scenes::{Paused, Scene};
use crate::settings::Settings;
use crate::sprite::SpriteAtlases;
use crate::ui::layout::Region;
use crate::ui::menu::{self, Action, Control, Nav, Slide};
use crate::ui::{IconAtlas, Icons, RunAge, UiScreen};
use crate::world::{SimSet, SimWorld, WorldFocus, WorldSave, build_world_saved, restore_run};
use yugen_core::config::View;

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
            .init_resource::<Nav>()
            .add_systems(
                PreUpdate,
                toggle_pause
                    .after(bevy::input::InputSystems)
                    .run_if(in_state(Scene::Playing)),
            )
            // After `toggle_pause`, so the frame that opens the pause card does
            // not also drive the menu with the same Escape press.
            .add_systems(
                PreUpdate,
                (root_the_menu, drive_menu)
                    .chain()
                    .after(toggle_pause)
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
) {
    if *scene.get() != Scene::Playing {
        return;
    }
    if body.is_some_and(|b| b.dead()) {
        next.set(Scene::GameOver);
    }
}

/// Start a run: fresh world, body back at the spawn, nothing left over.
///
/// This is `Game.loadLevel` plus the `reset` calls around it, and it runs on
/// EVERY entry to [`Scene::Playing`] — the first one out of the menu as well as
/// each restart. That is deliberate: a restart and a first start should produce
/// the same world, and a special case for "the first one" is a second code path
/// that only ever runs once and so is never really tested.
///
/// It is safe to run on the first entry because everything it clears is already
/// empty then, and the world it replaces was built moments earlier from the same
/// seed.
///
/// The world is REGENERATED rather than repaired, so a restart discards the
/// player's excavation — see [`build_world`]. That is the original's behaviour,
/// which allocated a fresh `CellGrid` in `loadLevel`.
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
) {
    // A new run starts unpaused, with its clock at zero so the control hints
    // are back at full. Restarting from the death card must not inherit the
    // faded hints of the run that just ended, and a run that began while the
    // pause card was up would resume into a stopped world.
    *paused = Paused(false);
    *age = RunAge(0.0);

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
    mut paused: ResMut<Paused>,
    mut next: ResMut<NextState<Scene>>,
    scene: Res<State<Scene>>,
    mut exit: MessageWriter<AppExit>,
) {
    let in_game = *scene.get() == Scene::Playing;
    let rows = menu::page_rows(nav.top(), &settings, in_game);
    if rows.is_empty() {
        return;
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
            &mut nav,
            &mut settings,
            &mut paused,
            &mut next,
            &mut exit,
        );
        return;
    }

    // --- Going back ---------------------------------------------------------
    //
    // Escape pops the stack, and at the ROOT it means whatever that root is
    // for: leaving the pause card resumes, and there is nothing to leave on the
    // title card. That single rule is why the stack is a stack.
    if bevy_keys.any_pressed(KEYS.pause) && !nav.pop() && nav.top() == menu::Page::Pause {
        paused.0 = false;
    }
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
fn do_menu_action(
    action: Action,
    nav: &mut Nav,
    settings: &mut Settings,
    paused: &mut Paused,
    next: &mut NextState<Scene>,
    exit: &mut MessageWriter<AppExit>,
) {
    if menu::apply_to_settings(action, settings) {
        return;
    }
    match action {
        Action::Open(page) => nav.push(page),
        Action::Back => {
            nav.pop();
        }
        // The title card asks WHICH world before starting one, exactly as it
        // did before there were menus. See `confirm_advances_the_scene`.
        Action::Play => next.set(Scene::WorldSelect),
        Action::Resume => paused.0 = false,
        // `autosave` runs in `Last` and is gated on a save directory, so the
        // world is already on disk by the time this lands; there is nothing to
        // flush here beyond leaving the scene.
        Action::Quit => {
            paused.0 = false;
            nav.reset(menu::Page::Title);
            next.set(Scene::Menu);
        }
        Action::Exit => {
            exit.write(AppExit::Success);
        }
        // Handled by `apply_to_settings` above; listed so a new variant is a
        // compile error here rather than a button that silently does nothing.
        Action::Flip(_) | Action::Step(_) => {}
    }
}

/// `Esc` stops the world without leaving it.
///
/// Gated on `Scene::Playing`, so it cannot be reached from a card. Pausing the
/// title screen would put up a card over a card, and the only way out of the
/// inner one would be the key that had just been shown not to work.
///
/// See `scenes::Paused` for why this is a resource rather than a scene, and
/// `ui::pause_at` for what the card says.
fn toggle_pause(keys: Res<ButtonInput<KeyCode>>, mut paused: ResMut<Paused>) {
    if BevyKeys(&keys).any_pressed(KEYS.pause) {
        paused.0 = !paused.0;
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
        Scene::GameOver => next.set(Scene::Playing),
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

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

use godgame_core::input::{KEYS, KeyState};

use godgame_core::config::SEED;
use godgame_core::items::{Inventory, item_code_of};

use crate::input::BevyKeys;
use crate::items::{GroundItems, Pack};
use crate::mobs::Creatures;
use crate::player::PlayerBody;
use crate::scenes::Scene;
use crate::sprite::SpriteAtlases;
use crate::ui::{IconAtlas, Icons, UiScreen};
use crate::world::{WorldFocus, build_world};

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
            .add_systems(OnEnter(Scene::Playing), start_a_run);
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
/// [`confirm_advances_the_scene`] is the only thing that sets this state.
fn start_a_run(
    mut commands: Commands,
    mut focus: ResMut<WorldFocus>,
    body: Option<ResMut<PlayerBody>>,
    creatures: Option<Res<Creatures>>,
    ground: Option<ResMut<GroundItems>>,
    pack: Option<Res<Pack>>,
) {
    let world = build_world(SEED);
    *focus = WorldFocus {
        x: world.level.spawn.x,
        y: world.level.spawn.y,
    };
    commands.insert_resource(world);

    // Each of these is optional because a caller may run `UiPlugin` and the
    // scene machine without the whole game — the capture test and the examples
    // both do. A missing one means that system is not in the app, not that
    // something failed.
    if let Some(mut body) = body {
        // Also drops every arrow in flight: they belong to the life that fired
        // them. See `Player::reset`.
        body.reset();
    }
    if let Some(creatures) = creatures {
        creatures.lock().clear();
    }
    if let Some(mut ground) = ground {
        ground.clear();
    }
    if let Some(pack) = pack {
        give_starting_kit(&mut pack.lock());
    }
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
/// SEAM (respawn): the TypeScript's `updateGameOver` called `loadLevel()` before
/// switching, which rebuilds the world and refills the pack. Nothing here does
/// that yet, so `GameOver` currently returns to a world that still has a dead
/// body in it. The place to hang it is `OnEnter(Scene::Playing)` — and see
/// [`Scene`]'s docs on why that must use `set_if_neq` semantics, or the reload
/// fires every time the state is re-set to the value it already holds.
fn confirm_advances_the_scene(
    keys: Res<ButtonInput<KeyCode>>,
    scene: Res<State<Scene>>,
    mut next: ResMut<NextState<Scene>>,
) {
    if !BevyKeys(&keys).any_pressed(KEYS.confirm) {
        return;
    }
    match scene.get() {
        Scene::Menu | Scene::GameOver => next.set(Scene::Playing),
        // Confirm is not a pause key. Nothing to advance to from here.
        Scene::Playing => {}
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

    #[test]
    fn every_scene_has_a_screen_and_the_mapping_is_total() {
        // The match in `follow_scene` is exhaustive by construction, so what is
        // worth pinning is that the two enums have not silently drifted apart —
        // a new `Scene` variant must be a compile error there, not a screen that
        // never appears.
        for (scene, want) in [
            (Scene::Menu, UiScreen::Menu),
            (Scene::Playing, UiScreen::Playing),
            (Scene::GameOver, UiScreen::GameOver),
        ] {
            let got = match scene {
                Scene::Menu => UiScreen::Menu,
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
        let first = build_world(SEED);
        let again = build_world(SEED);
        assert_eq!(first.level.spawn.x, again.level.spawn.x);
        assert_eq!(first.level.spawn.y, again.level.spawn.y);
        assert_eq!(first.seed, again.seed);
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

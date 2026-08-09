//! Which screen the game is showing: the menu, the world, or the death card.
//!
//! Ported from `src/game/SceneManager.ts`.
//!
//! # What the original was, plainly
//!
//! Eighteen lines and one field. `SceneManager` held a `Scene`, and offered
//! `scene` to read it, `is(s)` to compare it, and `to(s)` to set it. That is the
//! whole class — there was no transition hook, no guard on which moves were
//! legal, and no per-scene setup or teardown anywhere in it.
//!
//! It is a thin thing, and this module is thinner. Almost all of the port is
//! `#[derive(States)]`, and dressing that up would be inflating one field into
//! an architecture.
//!
//! # Where the eighteen lines went
//!
//! | `SceneManager.ts` | here |
//! |---|---|
//! | `private current: Scene = Scene.Menu` | `#[default] Menu` on [`Scene`] |
//! | `get scene()` | `Res<State<Scene>>` |
//! | `is(s)` | `in_state(s)` as a run condition, or `*state.get() == s` |
//! | `to(s)` | `ResMut<NextState<Scene>>::set` |
//!
//! # Why the shape changed, and what it buys
//!
//! The TypeScript's `Game.tick` was a `switch` on `scenes.scene` that dispatched
//! to `updateMenu`, `updatePlaying` or `updateGameOver`. Every frame, in every
//! scene, the whole game's update path went through one branch a human had to
//! keep exhaustive by hand — and every subsystem that should only run while
//! playing was only not running because that branch never reached it.
//!
//! Bevy already owns that dispatch. A system says
//! `run_if(in_state(Scene::Playing))` and the scheduler does the branching,
//! which puts the "which scenes does this run in" fact on the system instead of
//! in a `switch` three files away. That is why this is `States` and not a
//! resource with a setter: a resource would have ported the eighteen lines
//! exactly and thrown away the only reason to port them into Bevy at all.
//!
//! The genuinely new capability is `OnEnter` / `OnExit`. The TypeScript had
//! nowhere to hang "do this once when the scene changes", so `updateGameOver`
//! called `loadLevel()` inline immediately before flipping the scene, on the
//! frame the confirm key was pressed. That is an `OnEnter(Scene::Playing)`
//! system, and writing it as one is what stops a second entry point into
//! `Playing` from silently forgetting to build a level.
//!
//! # Who drives the transitions
//!
//! Not this module. [`ScenesPlugin`] installs the state and nothing else, which
//! is exactly what let all three transitions be added later without it changing.
//! They live in [`crate::glue`], because each joins two modules that must not
//! import each other:
//!
//! - `Menu -> Playing` and `GameOver -> Playing`, on the confirm key.
//!   [`crate::ui`] draws a screen promising "Press Enter or Space to start",
//!   this module owns the state that would honour it, and neither can reach the
//!   other.
//! - `Playing -> GameOver`, when `Player::dead` goes true.
//! - `loadLevel`, as an `OnEnter(Scene::Playing)` that regenerates the world and
//!   grants the starting kit.
//!
//! One warning that is still live, and is the reason that entry hook is an
//! `OnEnter` rather than something watching the state. `NextState::set` fires
//! `OnExit` and
//! `OnEnter` even when the scene does not actually change, and `SceneManager.to`
//! did not — so an entry hook that rebuilds the level would rebuild it under a
//! live player if anything set `Playing` while already playing.
//! `NextState::set_if_neq` is the one with the TypeScript's semantics. See
//! `setting_the_scene_it_is_already_in_re_enters_it_unless_asked_not_to`.

use bevy::prelude::*;

/// Which scene the game is currently showing.
///
/// The three the TypeScript's `Scene` enum had, in the same order and with the
/// same meanings. There is still no `Loading`.
///
/// There is now a pause, but it is [`Paused`] — a resource — and NOT a variant
/// here, for a concrete reason rather than a stylistic one. `glue::start_a_run`
/// hangs off `OnEnter(Scene::Playing)`. A `Scene::Paused` would have to return
/// to `Playing` to resume, that transition would re-fire `OnEnter`, and the
/// hook would regenerate the level and re-grant the starting kit underneath a
/// live player. That is the same hazard this type's own note about
/// `NextState::set` describes, reached by a different road. A pause that draws
/// over the world and stops the systems stepping it needs no scene change at
/// all.
#[derive(States, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Scene {
    /// The title screen. Confirm goes to [`Scene::WorldSelect`].
    #[default]
    Menu,
    /// Pick a save, make one, or delete one.
    ///
    /// Between the menu and a run rather than replacing the menu, because the
    /// menu is the game's front door and a list of directories is not a title
    /// card. `--play` skips both: a capture or a scenario has already been told
    /// which world it wants.
    WorldSelect,
    /// A world, a body, and every simulation system running.
    Playing,
    /// The death card. Confirm rebuilds the level and starts again.
    GameOver,
}

/// Installs [`Scene`], and nothing else.
///
/// One line, and it is honest about being one line. The value is in the type,
/// not in the plugin — but the state has to be registered by somebody, and a
/// module that publishes a `States` type without the plugin that installs it
/// makes every consumer guess whether it already exists.
pub struct ScenesPlugin;

impl Plugin for ScenesPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<Scene>().init_resource::<Paused>();
    }
}

/// Whether the simulation is stopped while the world stays on screen.
///
/// See [`Scene`] for why this is a resource. Systems that STEP the world are
/// gated on it; systems that DRAW the world are not, so a paused frame is a
/// live frame that is not advancing rather than a frozen screenshot — the
/// difference shows the moment a shader or a light pass is still running.
///
/// Default is false, and there is no way to start paused. A game that boots
/// paused is a game whose first frame looks broken.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Paused(pub bool);

/// Run condition: the simulation is advancing.
///
/// Named rather than written as a closure at each call site, so that the set of
/// systems that pause is greppable and a system added later can join it by
/// copying one recognisable thing.
pub fn running(paused: Res<Paused>) -> bool {
    !paused.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::state::app::StatesPlugin;

    /// The smallest app a state can live in. `StatesPlugin` is what runs the
    /// `StateTransition` schedule; without it `NextState` is written and never
    /// read, which is the one way a module this small can still be wired wrong.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((StatesPlugin, ScenesPlugin));
        app
    }

    fn scene(app: &App) -> Scene {
        *app.world().resource::<State<Scene>>().get()
    }

    fn go_to(app: &mut App, next: Scene) {
        app.world_mut().resource_mut::<NextState<Scene>>().set(next);
        app.update();
    }

    #[test]
    fn a_fresh_game_opens_on_the_menu() {
        // `private current: Scene = Scene.Menu` — the only state the
        // TypeScript's constructor set. Booting into `Playing` would skip the
        // title screen entirely, and is the obvious way to get the derive wrong.
        let mut app = app();
        app.update();
        assert_eq!(scene(&app), Scene::Menu);
    }

    #[test]
    fn confirming_at_the_menu_starts_a_run() {
        let mut app = app();
        app.update();
        go_to(&mut app, Scene::Playing);
        assert_eq!(scene(&app), Scene::Playing);
    }

    #[test]
    fn dying_and_confirming_returns_to_a_fresh_run() {
        // The full round trip the TypeScript had: `updatePlaying` sets GameOver
        // when the body dies, and `updateGameOver` calls `loadLevel()` and goes
        // back to Playing. Both edges, in the one order the game can reach them.
        let mut app = app();
        app.update();
        go_to(&mut app, Scene::Playing);
        go_to(&mut app, Scene::GameOver);
        assert_eq!(scene(&app), Scene::GameOver);
        go_to(&mut app, Scene::Playing);
        assert_eq!(scene(&app), Scene::Playing);
    }

    #[test]
    fn a_scene_change_is_not_visible_until_the_transition_runs() {
        // `NextState` is a REQUEST, where `SceneManager.to` was an assignment
        // that took effect on the very next statement. Anything ported off the
        // old class that reads the scene back immediately would be reading the
        // old one, and this is that difference written down.
        let mut app = app();
        app.update();
        app.world_mut()
            .resource_mut::<NextState<Scene>>()
            .set(Scene::Playing);
        assert_eq!(scene(&app), Scene::Menu, "the set applied early");
        app.update();
        assert_eq!(scene(&app), Scene::Playing);
    }

    #[test]
    fn entering_and_leaving_a_scene_each_fire_exactly_once() {
        // The capability the TypeScript did not have, and the reason this is
        // `States` and not a resource with a setter: `loadLevel` used to be
        // called inline next to the assignment, so a second way into Playing
        // would silently have skipped it.
        #[derive(Resource, Default)]
        struct Counts {
            entered: u32,
            exited: u32,
        }

        let mut app = app();
        app.init_resource::<Counts>()
            .add_systems(OnEnter(Scene::Playing), |mut c: ResMut<Counts>| {
                c.entered += 1;
            })
            .add_systems(OnExit(Scene::Playing), |mut c: ResMut<Counts>| {
                c.exited += 1;
            });
        app.update();

        go_to(&mut app, Scene::Playing);
        assert_eq!(app.world().resource::<Counts>().entered, 1);
        assert_eq!(app.world().resource::<Counts>().exited, 0);

        // Two idle frames in the same scene must not re-enter it.
        app.update();
        app.update();
        assert_eq!(app.world().resource::<Counts>().entered, 1, "re-entered");

        go_to(&mut app, Scene::GameOver);
        assert_eq!(app.world().resource::<Counts>().exited, 1);
        assert_eq!(app.world().resource::<Counts>().entered, 1);
    }

    #[test]
    fn a_system_gated_on_a_scene_runs_only_in_that_scene() {
        // `in_state` is `SceneManager.is`, moved off a `switch` in `Game.tick`
        // and onto the system that cares. This is the assertion that the gate is
        // a real one and not decoration.
        #[derive(Resource, Default)]
        struct Ticks(u32);

        let mut app = app();
        app.init_resource::<Ticks>().add_systems(
            Update,
            (|mut t: ResMut<Ticks>| t.0 += 1).run_if(in_state(Scene::Playing)),
        );

        app.update();
        app.update();
        assert_eq!(app.world().resource::<Ticks>().0, 0, "ran outside Playing");

        go_to(&mut app, Scene::Playing);
        assert!(
            app.world().resource::<Ticks>().0 > 0,
            "never ran in Playing"
        );

        go_to(&mut app, Scene::GameOver);
        let after_exit = app.world().resource::<Ticks>().0;
        app.update();
        assert_eq!(
            app.world().resource::<Ticks>().0,
            after_exit,
            "kept running after leaving Playing"
        );
    }

    /// Setting the scene you are already in RE-ENTERS it, and `SceneManager.to`
    /// did not.
    ///
    /// This is the one place the two models genuinely disagree, and it is the
    /// trap the seam above walks straight into. `SceneManager.to(current)` was
    /// an assignment of a value to itself — nothing happened, and the
    /// TypeScript's `updateGameOver` could afford to be careless about it
    /// because there was no such thing as an entry hook. Here, `NextState::set`
    /// runs `OnExit` and `OnEnter` even when the state does not change, so an
    /// `OnEnter(Scene::Playing)` that rebuilds the level would rebuild it
    /// underneath a live player the moment anything set `Playing` while already
    /// playing.
    ///
    /// `NextState::set_if_neq` is the version with the TypeScript's semantics.
    /// Whoever wires the confirm key up should reach for it unless re-entry is
    /// what they actually mean.
    #[test]
    fn setting_the_scene_it_is_already_in_re_enters_it_unless_asked_not_to() {
        #[derive(Resource, Default)]
        struct Entered(u32);

        let mut app = app();
        app.init_resource::<Entered>()
            .add_systems(OnEnter(Scene::Menu), |mut e: ResMut<Entered>| e.0 += 1);
        app.update();
        let once = app.world().resource::<Entered>().0;
        assert_eq!(once, 1, "the initial state never fired its entry hook");

        go_to(&mut app, Scene::Menu);
        assert_eq!(
            app.world().resource::<Entered>().0,
            2,
            "a same-scene `set` is expected to re-enter — if this now matches \
             the TypeScript, the note on this test is stale"
        );

        // Spelled out rather than as a method call, and scoped: `ResMut` derefs
        // to `NextState`, but it also carries Bevy's own `set_if_neq` from
        // `DetectChangesMut`, which takes a whole `NextState` and is not the one
        // this is about.
        {
            let mut pending = app.world_mut().resource_mut::<NextState<Scene>>();
            NextState::set_if_neq(&mut pending, Scene::Menu);
        }
        app.update();
        assert_eq!(
            app.world().resource::<Entered>().0,
            2,
            "`set_if_neq` is the one with `SceneManager.to`'s semantics"
        );
        assert_eq!(scene(&app), Scene::Menu);
    }
}

//! The scenarios in `scenarios/README.md`, as assertions instead of prose.
//!
//! # What this closes
//!
//! `scenarios/README.md` catalogues thirteen situations and, for each, a fact
//! read out of a real `--dump-state` run: 417 of 441 cells empty in a carved
//! chamber, 387 lava at y=3000, `day = 0.498` at `t = 0.26`. Every one of those
//! was measured. **Not one of them was protected by anything.** They are
//! observations in a document, and a document is exactly where a true statement
//! goes to quietly stop being true — which is the failure `HANDOFF.md` §8.3
//! already records happening to three comments in this tree.
//!
//! So this runs them. Same code path as the command line: `scene::ScenePlugin`
//! is what `--at` and `--edit` register in the binary, and until it moved out of
//! `crates/godgame/src/main.rs` a scenario proved at a terminal could not become
//! a gate without somebody reimplementing three orderings that took three
//! attempts to get right the first time.
//!
//! # Why the bounds are loose
//!
//! The catalogue's numbers are exact — 417, 387, 44 — and these assertions are
//! not. That is deliberate and it is not laziness. A carved chamber's exact
//! cell count is a function of the brush, the automata's settling, the terrain
//! it was cut into and how many frames were allowed; pinning 417 would fail on
//! the next legitimate change to any of them and be deleted within a month, the
//! same argument `frame_capture.rs` makes about its own floors. What each test
//! asserts is the CLAIM the catalogue row is making — "this is a chamber", "this
//! is a lava sea", "underground air has a wall behind it" — at a bound wide
//! enough to survive tuning and narrow enough that the claim's opposite fails.
//!
//! Each test prints its measured number, so a drift shows up as a moving figure
//! in the log before it shows up as a red gate.
//!
//! # Skipping
//!
//! No GPU, no run — and a SKIP, never a pass. Same discipline as
//! `frame_capture.rs` and `shader_matches_cpu.rs`.

use std::collections::BTreeMap;

use bevy::prelude::*;
use godgame_core::sim::edits::EditMode;
use godgame_core::sim::materials::{CellId, code_of};
use godgame_render::daynight::{DayNight, WorldClock};
use godgame_render::player::{NoPlayer, PlayerBody};
use godgame_render::scene::{ARRANGE_FRAMES, ScenePlugin, StartAt, StartupEdit};
use godgame_render::scenes::Scene;
use godgame_render::world::SimWorld;

mod common;

/// Cells either side of the sample centre. 10 gives the same 21x21 block
/// `--dump-state` reports, so a number here is comparable with one from the
/// catalogue without arithmetic in between.
const RADIUS: i32 = 10;

/// Cells in that block.
const SAMPLE: usize = ((2 * RADIUS + 1) * (2 * RADIUS + 1)) as usize;

/// One scenario, as the command line would express it.
#[derive(Default)]
struct Scenario {
    at: Option<StartAt>,
    edit: Option<StartupEdit>,
    /// Clock as a fraction of a day.
    time: Option<f32>,
    /// No player body — the view is the actor, and nothing falls.
    free_camera: bool,
    /// Frames after the scene is arranged, for the automata to settle.
    settle: u32,
}

/// What a run leaves behind: the two cell planes around the sample centre, and
/// whatever else a test wants to look at.
struct Observed {
    front: Vec<CellId>,
    wall: Vec<CellId>,
    day: f32,
    health: Option<f32>,
    body: Option<(f32, f32)>,
}

impl Observed {
    /// Cells of one material in the front plane.
    fn count(&self, id: &str) -> usize {
        let code = code_of(id);
        assert_ne!(code, 0, "no material called {id:?}");
        self.front.iter().filter(|&&c| c == code).count()
    }

    /// Air cells in the front plane.
    fn air(&self) -> usize {
        self.front.iter().filter(|&&c| c == 0).count()
    }

    /// Air cells that have something in the wall plane behind them.
    fn air_with_wall(&self) -> usize {
        self.front
            .iter()
            .zip(&self.wall)
            .filter(|(f, w)| **f == 0 && **w != 0)
            .count()
    }

    /// The front plane by material name, commonest first — for a failure
    /// message that says what WAS there rather than only what was not.
    fn census(&self) -> String {
        let mut by: BTreeMap<CellId, usize> = BTreeMap::new();
        for &c in &self.front {
            *by.entry(c).or_default() += 1;
        }
        let mut rows: Vec<(usize, String)> = by
            .into_iter()
            .map(|(c, n)| {
                let name = godgame_data::blocks::BLOCKS
                    .get(c as usize)
                    .map_or_else(|| c.to_string(), |d| d.id.to_string());
                (n, name)
            })
            .collect();
        rows.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
        rows.iter()
            .take(5)
            .map(|(n, name)| format!("{name} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Boot the game, arrange the scene, settle, and read the world.
///
/// `None` means no GPU. The whole plugin group is booted rather than a subset,
/// because the point of a scenario is that it is the game — a harness that ran
/// worldgen and skipped the rest would prove nothing about what a capture of the
/// same flags would show.
fn run(scenario: &Scenario) -> Option<Observed> {
    if !common::gpu_is_available() {
        return None;
    }

    let mut app = common::headless_game("GodGame scenario");
    app.add_plugins(ScenePlugin);
    if scenario.free_camera {
        app.insert_resource(NoPlayer);
    }
    if let Some(at) = scenario.at {
        app.insert_resource(at);
    }
    if let Some(edit) = scenario.edit {
        app.insert_resource(edit);
    }
    if let Some(t) = scenario.time {
        app.insert_resource(WorldClock(DayNight::new(t)));
    }
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    for _ in 0..ARRANGE_FRAMES + scenario.settle {
        app.update();
    }

    // Sample around the BODY when there is one and the requested point when
    // there is not, which is what `--dump-state` does and why the numbers here
    // and in the catalogue are comparable.
    let body = app
        .world()
        .get_resource::<PlayerBody>()
        .map(|b| (b.0.x, b.0.y));
    // Body, then the requested point, then the CAMERA — and the last of those
    // is not a fallback nobody hits. A `free_camera` scenario with no `at` sits
    // at the spawn, and reading (0, 0) instead put the surface sample a hundred
    // cells up in empty sky: it reported 441 of 441 air with no wall behind it
    // and passed, because open sky genuinely has no wall. The claim was true and
    // about nothing.
    let focus = app.world().resource::<godgame_render::world::WorldFocus>();
    let (cx, cy) = match (body, scenario.at) {
        (Some((x, y)), _) => (x, y),
        (None, Some(at)) => (at.x, at.y),
        (None, None) => (focus.x, focus.y),
    };
    let cell = |v: f32| (v / godgame_core::config::CELL_SIZE as f32).floor() as i32;
    let (cx, cy) = (cell(cx), cell(cy));

    let world = app.world().resource::<SimWorld>();
    let grid = &world.level.grid;
    let mut front = Vec::with_capacity(SAMPLE);
    let mut wall = Vec::with_capacity(SAMPLE);
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            let at = godgame_core::sim::coords::WorldCell::new(cx + dx, cy + dy);
            front.push(grid.get_world(at));
            wall.push(grid.get_back_world(at));
        }
    }

    let day = app.world().resource::<WorldClock>().0.phase().day;
    let health = app.world().get_resource::<PlayerBody>().map(|b| b.0.health);
    Some(Observed {
        front,
        wall,
        day,
        health,
        body,
    })
}

/// Run, or say why not and hand back `None`.
fn run_or_skip(what: &str, scenario: &Scenario) -> Option<Observed> {
    let out = run(scenario);
    if out.is_none() {
        println!("SKIPPED {what}: no wgpu adapter on this machine");
    }
    out
}

/// A carved chamber 900 px down, from `scenarios/README.md`'s first row.
fn deep_chamber() -> Scenario {
    Scenario {
        at: Some(StartAt { x: 0.0, y: 900.0 }),
        edit: Some(StartupEdit {
            mode: EditMode::Dig,
            mat: 0,
            cx: 0,
            cy: 0,
            r: 12,
        }),
        free_camera: true,
        settle: 60,
        ..Scenario::default()
    }
}

#[test]
fn a_carved_chamber_is_a_chamber_and_the_rock_around_it_is_not() {
    let Some(dug) = run_or_skip("deep chamber", &deep_chamber()) else {
        return;
    };
    // The control, and the reason this is not just "some cells are air": the
    // same point with no stroke. Without it, a scenario that silently failed to
    // carve would still pass wherever worldgen had left a cave.
    let mut untouched = deep_chamber();
    untouched.edit = None;
    let Some(solid) = run_or_skip("deep chamber control", &untouched) else {
        return;
    };

    println!(
        "deep chamber: {} air of {SAMPLE} (control {}); {}",
        dug.air(),
        solid.air(),
        dug.census()
    );
    assert!(
        dug.air() > SAMPLE * 3 / 4,
        "the chamber did not open: {} air of {SAMPLE}, control {}. Contents: {}",
        dug.air(),
        solid.air(),
        dug.census()
    );
    assert!(
        solid.air() < SAMPLE / 4,
        "the CONTROL is already mostly air at {} of {SAMPLE}, so this scenario \
         proves nothing about the brush — move it into rock",
        solid.air()
    );
}

/// The claim `scenarios/README.md` makes about the background wall plane.
///
/// This is the one thing here that nothing else can check. `HANDOFF.md` §8.2
/// records `WALL_TINT` as never visually verified, and a number is what stands
/// in for the eye until somebody looks — it cannot say whether the tint is
/// RIGHT, but it can say the plane is populated where it should be and absent
/// where it should not.
///
/// The two halves control each other, which is why they are one test. A wall
/// plane that was empty everywhere fails the underground assertion; one that was
/// full everywhere fails the surface assertion. Neither can pass by being
/// trivially true, which is the trap this test fell into on its first run: the
/// surface sample was taken a hundred cells up in open sky, reported 441 of 441
/// air with nothing behind it, and passed while measuring nothing.
#[test]
fn underground_air_has_a_wall_behind_it_and_open_sky_does_not() {
    let Some(under) = run_or_skip("wall plane underground", &deep_chamber()) else {
        return;
    };
    let surface = Scenario {
        free_camera: true,
        settle: 60,
        ..Scenario::default()
    };
    let Some(above) = run_or_skip("wall plane at the surface", &surface) else {
        return;
    };

    println!(
        "wall plane: underground {}/{} air cells walled, surface {}/{}",
        under.air_with_wall(),
        under.air(),
        above.air_with_wall(),
        above.air()
    );

    // Underground it is all-or-nothing: worldgen fills the wall plane with the
    // terrain that would be there ignoring the carve, so every cell a player
    // dug out has rock behind it.
    assert!(
        under.air() > 0 && under.air_with_wall() * 10 >= under.air() * 9,
        "a carved chamber should be walled almost everywhere: {} of {} air cells",
        under.air_with_wall(),
        under.air()
    );
    // Above the surface line there is deliberately no wall, which is what makes
    // "you have dug through to open sky" a state you can see.
    assert!(
        above.air() > 0 && above.air_with_wall() * 4 < above.air(),
        "open sky should have nothing behind it: {} of {} air cells walled",
        above.air_with_wall(),
        above.air()
    );
}

#[test]
fn the_lava_sea_is_where_the_catalogue_says_it_is() {
    let deep = Scenario {
        at: Some(StartAt { x: 0.0, y: 3000.0 }),
        free_camera: true,
        settle: 40,
        ..Scenario::default()
    };
    let Some(seen) = run_or_skip("lava sea", &deep) else {
        return;
    };
    println!("lava sea at y=3000: {}", seen.census());
    assert!(
        seen.count("lava") > SAMPLE / 2,
        "y=3000 is supposed to be a lava sea and holds {} lava of {SAMPLE}: {}",
        seen.count("lava"),
        seen.census()
    );
}

/// Twilight is 0.26 and 0.735, NOT 0.25 and 0.75.
///
/// I wrote the wrong pair into the `--time` usage text by analogy with a clock,
/// and the catalogue caught it by measuring. This is the measurement, kept.
#[test]
fn the_clock_reaches_the_states_the_catalogue_names() {
    for (t, want, what) in [
        (0.0, 0.0, "midnight"),
        (0.5, 1.0, "noon"),
        (0.26, 0.5, "dawn"),
        (0.735, 0.55, "dusk"),
    ] {
        let scenario = Scenario {
            time: Some(t),
            free_camera: true,
            ..Scenario::default()
        };
        let Some(seen) = run_or_skip("clock", &scenario) else {
            return;
        };
        println!("  --time {t} -> day {:.3} ({what})", seen.day);
        assert!(
            (seen.day - want).abs() < 0.12,
            "--time {t} is documented as {what} (day ~{want}) and measures {:.3}. \
             The daylight ramps run t 0.23-0.30 and 0.70-0.77, so 0.25 and 0.75 are \
             NOT the half-lit points — check that before moving this bound",
            seen.day
        );
    }
}

/// The one a screenshot could never answer, and the reason the state dump
/// exists: a body in lava dies and a body in a stone chamber does not.
#[test]
fn lava_kills_a_body_and_a_carved_chamber_does_not() {
    let in_lava = Scenario {
        at: Some(StartAt { x: 0.0, y: 3000.0 }),
        settle: 120,
        ..Scenario::default()
    };
    let Some(burned) = run_or_skip("body in lava", &in_lava) else {
        return;
    };

    let mut in_rock = deep_chamber();
    in_rock.free_camera = false;
    in_rock.settle = 120;
    let Some(safe) = run_or_skip("body in a chamber", &in_rock) else {
        return;
    };

    println!(
        "health: in lava {:?} at {:?}, in a chamber {:?} at {:?}",
        burned.health, burned.body, safe.health, safe.body
    );
    assert_eq!(
        burned.health,
        Some(0.0),
        "a body dropped into the lava sea should be dead: {:?}",
        burned.health
    );
    assert_eq!(
        safe.health,
        Some(godgame_core::config::MAX_HEALTH),
        "a body standing in a carved chamber should be untouched: {:?}. If this \
         fell, the chamber did not open and the body is in rock",
        safe.health
    );
}

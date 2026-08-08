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
//! `crates/yugen/src/main.rs` a scenario proved at a terminal could not become
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
use yugen_core::config::WorldScale;
use yugen_core::input::KEYS;
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::edits::EditMode;
use yugen_core::sim::materials::{CellId, code_of};
use yugen_render::daynight::{DayNight, WorldClock};
use yugen_render::player::{NoPlayer, PlayerBody};
use yugen_render::scene::{ARRANGE_FRAMES, ScenePlugin, StartAt, StartWith, StartupEdit};
use yugen_render::scenes::Scene;
use yugen_render::world::SimWorld;

mod common;

/// Cells either side of the sample centre. 10 gives the same 21x21 block
/// `--dump-state` reports, so a number here is comparable with one from the
/// catalogue without arithmetic in between.
const RADIUS: i32 = 10;

/// Cells in that block.
const SAMPLE: usize = ((2 * RADIUS + 1) * (2 * RADIUS + 1)) as usize;

/// One scenario, as the command line would express it.
#[derive(Clone, Default)]
struct Scenario {
    /// Items to add on top of the starting kit — `--give`.
    give: Vec<(yugen_core::items::registry::ItemCode, u32)>,
    at: Option<StartAt>,
    edit: Option<StartupEdit>,
    /// Clock as a fraction of a day.
    time: Option<f32>,
    /// No player body — the view is the actor, and nothing falls.
    free_camera: bool,
    /// Frames after the scene is arranged, for the automata to settle.
    settle: u32,
    /// A block written straight into the grid beside the body, as
    /// `(id, dx, dy)` in cells from the body's centre.
    ///
    /// NOT `edit`, and the difference matters. `apply_brush` in `Place` mode
    /// only fills EMPTY cells — deliberately, so a player cannot paint over
    /// their own footing — and it aims from the VIEW CENTRE, which chases the
    /// body. Between those two a stroke meant to put a workbench beside the
    /// player lands on solid ground and is silently refused, which is exactly
    /// what happened here and looked for an hour like a broken station rule.
    ///
    /// The brush has its own tests. This one is about stations, so it writes
    /// the cell and moves on.
    place_block: Option<(&'static str, i32, i32)>,
    /// Put the crafting card's cursor on the row that makes this item, once the
    /// card is open.
    ///
    /// Standing in for the player scrolling to it. The cursor's own movement and
    /// scrolling have unit tests in `craftscreen`; what this test is about is
    /// the station rule and the key path, and forty `down` taps would be forty
    /// chances to test the wrong thing.
    select: Option<&'static str>,
    /// Keys to tap once the scene is arranged, in order, one per binding.
    ///
    /// Through `ButtonInput<KeyCode>` rather than by calling `try_craft`, for
    /// the reason `--script` presses the real button: a test that reached past
    /// the binding could pass while the binding, the mode branch and the card
    /// that now owns the key were all broken. Crafting is two keys now — `C`
    /// opens the card, confirm makes the selected row — and a test that knew
    /// only about the first would have gone green on the day the second
    /// stopped working.
    taps: Vec<&'static [&'static str]>,
}

/// What a run leaves behind: the two cell planes around the sample centre, and
/// whatever else a test wants to look at.
struct Observed {
    front: Vec<CellId>,
    wall: Vec<CellId>,
    day: f32,
    health: Option<f32>,
    /// Damage the body subtracts from a hit, from whatever is worn.
    armour: Option<f32>,
    body: Option<(f32, f32)>,
    pack: Vec<(yugen_core::items::registry::ItemCode, u16)>,
    /// Workbench cells anywhere in the streaming window, not just the sample.
    benches: usize,
    /// Stations the body can reach, as the game computes it.
    reach: String,
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

    /// How many of an item the pack holds.
    fn holds(&self, id: &str) -> u32 {
        let code = yugen_core::items::item_code_of(id).expect("no such item");
        self.pack
            .iter()
            .filter(|(c, _)| *c == code)
            .map(|(_, n)| u32::from(*n))
            .sum()
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
                let name = yugen_data::blocks::BLOCKS
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

    let mut app = common::headless_game("Yūgen scenario");
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
    if !scenario.give.is_empty() {
        app.insert_resource(StartWith(scenario.give.clone()));
    }
    if let Some(t) = scenario.time {
        app.insert_resource(WorldClock(DayNight::new(t)));
    }
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);

    // The craft key is pressed from a SYSTEM in `RunFixedMainLoop`, not from
    // outside the schedule, and that is not ceremony. `bevy_input` clears
    // `just_pressed` in `PreUpdate`, so a press written between `update()` calls
    // is gone before `input::tool_keys` reads it in `Update` — the key is down
    // and no edge ever arrives. The binary's `--script` driver presses in this
    // same slot for the same reason; anywhere earlier is a keypress the game
    // cannot see.
    if !scenario.taps.is_empty() {
        app.add_systems(
            RunFixedMainLoop,
            press_craft
                .in_set(bevy::app::RunFixedMainLoopSystems::BeforeFixedMainLoop)
                // Not until the scene is BUILT. `arrange_the_scene` removes both
                // resources when it is done, and it waits for the world to
                // stream — so a press before that lands in a world with no
                // workbench in it yet, and the test measures the wrong frame.
                // This cost a debugging round.
                // Only once the harness says the scene is ready. Inserting
                // `PressCraft` up front pressed the key on frame 1 — before the
                // block below was written — so the craft happened in a world
                // with no workbench in it and the test measured the wrong
                // frame. Twice.
                .run_if(resource_exists::<Tapping>),
        );
    }
    for _ in 0..ARRANGE_FRAMES {
        app.update();
    }

    // After the scene is arranged and the body has come to rest, so the offset
    // is measured from where the player actually IS.
    if let Some((id, dx, dy)) = scenario.place_block {
        let (bx, by) = app
            .world()
            .get_resource::<PlayerBody>()
            .map(|b| (b.0.x, b.0.y))
            .expect("place_block needs a body to measure from");
        let cx = ((bx + yugen_core::config::PLAYER_W * 0.5) / yugen_core::config::CELL_SIZE as f32)
            .floor() as i32;
        let cy = ((by + yugen_core::config::PLAYER_H * 0.5) / yugen_core::config::CELL_SIZE as f32)
            .floor() as i32;
        let code = yugen_core::sim::materials::code_of(id);
        assert_ne!(code, 0, "no block called {id:?}");
        app.world_mut()
            .resource_mut::<SimWorld>()
            .level
            .grid
            .set_world(WorldCell::new(cx + dx, cy + dy), code);
    }

    // Everything the scenario asked for is now in the world, so the key may
    // fire.
    if let Some(id) = scenario.select {
        app.insert_resource(SelectRow(id));
        app.add_systems(
            Update,
            select_row
                .run_if(resource_exists::<SelectRow>)
                // Before the tap that confirms, and after the one that opened
                // the card — which is what `Tapping` still holding an entry
                // means.
                .before(press_craft),
        );
    }
    if !scenario.taps.is_empty() {
        app.insert_resource(Tapping(scenario.taps.clone()));
    }
    for _ in 0..scenario.taps.len() as u32 * 2 + 2 + scenario.settle {
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
    let focus = app.world().resource::<yugen_render::world::WorldFocus>();
    let (cx, cy) = match (body, scenario.at) {
        (Some((x, y)), _) => (x, y),
        (None, Some(at)) => (at.x, at.y),
        (None, None) => (focus.x, focus.y),
    };
    let cell = |v: f32| (v / yugen_core::config::CELL_SIZE as f32).floor() as i32;
    let (cx, cy) = (cell(cx), cell(cy));

    let world = app.world().resource::<SimWorld>();
    let grid = &world.level.grid;
    let mut front = Vec::with_capacity(SAMPLE);
    let mut wall = Vec::with_capacity(SAMPLE);
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            let at = yugen_core::sim::coords::WorldCell::new(cx + dx, cy + dy);
            front.push(grid.get_world(at));
            wall.push(grid.get_back_world(at));
        }
    }

    let day = app.world().resource::<WorldClock>().0.phase().day;
    let health = app.world().get_resource::<PlayerBody>().map(|b| b.0.health);
    let armour = app.world().get_resource::<PlayerBody>().map(|b| b.0.armour);
    let pack = app
        .world()
        .get_resource::<yugen_render::items::Pack>()
        .map(|p| {
            (0..yugen_core::items::inventory::SLOT_COUNT)
                .filter_map(|i| p.0.stack_at(i))
                .collect()
        })
        .unwrap_or_default();
    let reach = match (app.world().get_resource::<PlayerBody>(), ()) {
        (Some(b), ()) => format!(
            "{:?}",
            yugen_render::interact_reach::stations_in_reach(
                &app.world().resource::<SimWorld>().level.grid,
                b.0.x,
                b.0.y
            )
        ),
        _ => "no body".to_string(),
    };
    let bench = yugen_core::sim::materials::code_of("workbench");
    let benches = world
        .level
        .grid
        .material
        .iter()
        .filter(|&&c| c == bench)
        .count();
    Some(Observed {
        front,
        wall,
        day,
        health,
        armour,
        body,
        pack,
        benches,
        reach,
    })
}

/// The item whose recipe row the cursor should sit on.
#[derive(Resource)]
struct SelectRow(&'static str);

/// Move the crafting cursor to that row, once the card is open.
fn select_row(
    mut commands: Commands,
    want: Res<SelectRow>,
    mut view: ResMut<yugen_render::craftscreen::CraftingView>,
) {
    if !view.open || view.rows.is_empty() {
        return;
    }
    let at = view
        .rows
        .iter()
        .position(|r| r.out == yugen_core::items::item_by_id(want.0).expect("item").name)
        .expect("no recipe makes that");
    view.cursor = at;
    commands.remove_resource::<SelectRow>();
}

/// Key taps still owed, in order.
#[derive(Resource)]
struct Tapping(Vec<&'static [&'static str]>);

/// Tap one binding per two frames: down, then up.
///
/// Down for one frame and up the next, which is what a keypress is. Held for
/// two it would raise one edge and then look like a key nobody let go of — and
/// the crafting card toggles on that edge, so a held key would open and close
/// it on alternate frames.
fn press_craft(mut owed: ResMut<Tapping>, mut keys: ResMut<ButtonInput<KeyCode>>) {
    // Anything still down goes up first, so two taps are two edges.
    let mut released = false;
    for binding in &owed.0 {
        if let Some(code) = binding
            .first()
            .and_then(|n| yugen_render::input::key_code(n))
            && keys.pressed(code)
        {
            keys.release(code);
            released = true;
        }
    }
    if released {
        return;
    }
    if owed.0.is_empty() {
        return;
    }
    let binding = owed.0.remove(0);
    let code = yugen_render::input::key_code(binding[0]).expect("a real key");
    keys.press(code);
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

/// Depth of the lava sea, in px, in the LEGACY world geometry these scenarios
/// were authored against — scaled at use.
///
/// The underworld begins at `UNDERWORLD_DEPTH` cells below the local surface and
/// that is a legacy depth, so a fixed px coordinate stops naming the underworld
/// the moment the world scales. At `WorldScale::LIVE` the unscaled 3000 px lands
/// in ordinary deep stone, which is what these two tests reported: 0 lava of 441,
/// and a body that stood in it at full health.
const LAVA_SEA_PX: f64 = 3000.0;

/// [`LAVA_SEA_PX`] in the world the game actually generates.
fn lava_sea_y() -> f32 {
    (LAVA_SEA_PX * WorldScale::LIVE.factor()) as f32
}

#[test]
fn the_lava_sea_is_where_the_catalogue_says_it_is() {
    let deep = Scenario {
        at: Some(StartAt {
            x: 0.0,
            y: lava_sea_y(),
        }),
        free_camera: true,
        settle: 40,
        ..Scenario::default()
    };
    let Some(seen) = run_or_skip("lava sea", &deep) else {
        return;
    };
    println!("lava sea at y={}: {}", lava_sea_y(), seen.census());
    assert!(
        seen.count("lava") > SAMPLE / 2,
        "y={} is supposed to be a lava sea and holds {} lava of {SAMPLE}: {}",
        lava_sea_y(),
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
        at: Some(StartAt {
            x: 0.0,
            y: lava_sea_y(),
        }),
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
        Some(yugen_core::config::MAX_HEALTH),
        "a body standing in a carved chamber should be untouched: {:?}. If this \
         fell, the chamber did not open and the body is in rock",
        safe.health
    );
}

/// The station rule, end to end through the real key.
///
/// This is `items::crafting`'s `Reach` and `interact_reach`'s scan and
/// `input::try_craft`'s branch, all at once, driven by the same `C` a player
/// presses — not one of them called directly. Each has its own unit test; none
/// of them can say the three are wired to each other.
///
/// The two runs differ by ONE thing: whether a workbench is standing next to
/// the body. Same seed, same ingredients, same script.
#[test]
fn an_anvil_needs_a_workbench_and_says_so_when_there_is_none() {
    let parts = || {
        vec![
            (
                yugen_core::items::item_code_of("iron_bar").expect("iron_bar"),
                3u32,
            ),
            (
                yugen_core::items::item_code_of("stone_chunk").expect("stone_chunk"),
                6,
            ),
        ]
    };

    let without = Scenario {
        give: parts(),
        // What a player does: `C` opens the card, confirm makes the row.
        taps: vec![KEYS.craft, KEYS.confirm],
        select: Some("anvil"),
        settle: 30,
        ..Scenario::default()
    };
    let Some(alone) = run_or_skip("anvil with no bench", &without) else {
        return;
    };

    let with = Scenario {
        give: parts(),
        taps: vec![KEYS.craft, KEYS.confirm],
        select: Some("anvil"),
        settle: 30,
        place_block: Some(("workbench", 2, 0)),
        ..Scenario::default()
    };
    let Some(beside) = run_or_skip("anvil beside a bench", &with) else {
        return;
    };

    println!(
        "  reach with bench: {} ({} benches); pack {:?}",
        beside.reach, beside.benches, beside.pack
    );
    println!(
        "anvil: without a bench {} (iron left {}), with one {} (iron left {})",
        alone.holds("anvil"),
        alone.holds("iron_bar"),
        beside.holds("anvil"),
        beside.holds("iron_bar")
    );
    assert_eq!(
        alone.holds("anvil"),
        0,
        "an anvil was crafted with no workbench in reach — the station field is \
         being ignored again, which is the state this tree was in for four \
         milestones"
    );
    assert_eq!(
        alone.holds("iron_bar"),
        3,
        "the ingredients were spent on a craft that did not happen"
    );
    assert_eq!(
        beside.holds("anvil"),
        1,
        "a workbench one cell away should be in reach: it is not, or the craft \
         key is not reaching `try_craft`"
    );
}

/// Armour is put on with the real key and reaches the body.
///
/// `crafting`'s rule and `Inventory::equip`'s swap and `Player::take_damage`'s
/// subtraction all have unit tests. None of them can say the key a player
/// presses reaches any of it — and the first attempt at this proved the point:
/// the script pressed `use` with the pickaxe selected, because `use` acts on the
/// HELD item and nothing had selected the armour. That is what the `hotbar` verb
/// is for.
#[test]
fn armour_is_worn_by_pressing_use_on_it_and_the_body_feels_it() {
    let plate = yugen_core::items::item_code_of("chitin_plate").expect("chitin_plate");
    let worn = Scenario {
        give: vec![(plate, 1)],
        // Slot 4 is the first free one after the starting kit's three.
        taps: vec![std::slice::from_ref(&KEYS.hotbar[3]), KEYS.use_item],
        settle: 20,
        ..Scenario::default()
    };
    let Some(dressed) = run_or_skip("wearing armour", &worn) else {
        return;
    };

    let mut carried = worn.clone();
    // The control: same plate, never put on. Carrying armour must do nothing.
    carried.taps = vec![std::slice::from_ref(&KEYS.hotbar[3])];
    let Some(carrying) = run_or_skip("carrying armour", &carried) else {
        return;
    };

    println!(
        "armour: worn {:?}, merely carried {:?}",
        dressed.armour, carrying.armour
    );
    assert_eq!(
        dressed.armour,
        Some(5.0),
        "the chitin plate is 5 points and the body did not get them — the key, \
         the equip, or `glue`'s line that copies it across is not connected"
    );
    assert_eq!(
        carrying.armour,
        Some(0.0),
        "armour in the pack must protect nobody"
    );
    assert_eq!(
        carrying.holds("chitin_plate"),
        1,
        "the control must still be carrying it"
    );
    assert_eq!(
        dressed.holds("chitin_plate"),
        0,
        "what is worn is out of the pack"
    );
}

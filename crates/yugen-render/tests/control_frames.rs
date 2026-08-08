//! The control set: one photograph of each place the art has to work, plus the
//! numbers that describe it.
//!
//! # Why this is a test and not a shell script
//!
//! `scenarios/README.md` documents this set as command lines against the real
//! binary — `--free-camera --at 0,900 --edit dig 0 0 12 --warmup 60` and so on.
//! Two things are wrong with that as the basis for judging an art change.
//!
//! The first is fatal in some environments and was found by hitting it: the
//! binary's `--screenshot` needs a window server, and without one it writes a
//! fully black PNG — the same 62 922 bytes at every position, while the paired
//! `--dump-state` files are perfectly correct. A capture that fails by producing
//! a plausible file is the worst way for a capture to fail. `capture_low_res`
//! reads `LowResTarget::canvas`, which is the buffer the whole game renders into
//! and needs no display at all.
//!
//! The second matters everywhere: the binary advances its clock on wall time, so
//! `scenarios/README.md` measures 54.8% of pixels moving between two runs of the
//! SAME build. Against that noise floor a real art change cannot be separated
//! from a rerun. `common::FRAME_DT` pins the delta here, so the harness is
//! bit-identical run to run and the control diff is exactly zero.
//!
//! # What it asserts, and what it does not
//!
//! Like `lit_scene.rs`, this is a rig rather than a gate: there is no numeric
//! property of a cave worth failing a build over, and a threshold invented here
//! would encode one contributor's taste as a build error.
//!
//! What it DOES assert is that each scene is the scene it claims to be — the
//! chamber is air, the lava frame has lava in it, the night frame is darker than
//! the day one. A rig whose subject quietly stopped being there would hand you a
//! picture of solid rock and say nothing, which is exactly how
//! `lit_scene::carve_and_light` once captured an empty hole for three milestones.
//!
//! # The numbers
//!
//! It prints `frame_capture`'s three floors for every frame — distinct colours,
//! luma mean and standard deviation, dominant-colour share. Those are the
//! quantitative read on "dark but still legible", and the art overhaul's commit
//! messages are supposed to carry them. Recording them BEFORE the repaint is
//! what makes the after-numbers mean anything.
//!
//! ```text
//! cargo test -p yugen-render --test control_frames -- --nocapture
//! ```

use std::path::PathBuf;

use bevy::prelude::*;

use yugen_core::config::WorldScale;
use yugen_core::config::cell_at;
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::materials::{EMPTY, code_of};
use yugen_render::daynight::{DayNight, WorldClock};
use yugen_render::input::FocusDriver;
use yugen_render::scenes::Scene;
use yugen_render::world::{SimWorld, WorldFocus, WorldSave};

mod common;
use common::{Frame, capture_low_res, error_count, error_sources, gpu_is_available, headless_game};

/// Frames to let the world stream and settle at a new focus.
///
/// The streamer recentres on the focus and the window has to be dragged there a
/// chunk at a time, so this is a settling time and not a formality: `scene.rs`
/// records `--warmup 0` and `1` producing untouched rock at a target only 900 px
/// away.
const SETTLE: u32 = 60;

/// Extra settling for the deep frames.
///
/// Below the surface the falling-sand automata is doing real work — lava spreads,
/// sand slumps — and a frame taken before it quiets is a frame of a world in
/// motion, which diffs against itself.
const DEEP_SETTLE: u32 = 140;

/// Half-extents of a carved chamber, in cells.
const ROOM_W: i32 = 30;
/// See [`ROOM_W`].
const ROOM_H: i32 = 14;

/// What to photograph.
struct Shot {
    /// File stem, and the name in the printed table.
    name: &'static str,
    /// World seed. `None` is the compiled default.
    seed: Option<u32>,
    /// Cycle position, `None` for the default start. 0 midnight, 0.5 noon.
    time: Option<f32>,
    /// Px BELOW the spawn to fly the camera, in the LEGACY world geometry these
    /// scenes were authored against. 0 keeps the surface frame.
    ///
    /// Multiplied by the world scale at use. The depth bands these shots aim at —
    /// the cavern, the ore band, the underworld's lava — are authored in legacy
    /// cells, so a fixed px depth stops naming the same band the moment the world
    /// scales. That is not hypothetical: at `WorldScale::LIVE` the unscaled 2900
    /// px put `lava-sea` a long way above the lava, and the rig caught it by
    /// checking the scene is the scene rather than by the numbers looking odd.
    depth: f32,
    /// Carve a room at the focus and floor it, so there is somewhere to stand.
    carve: bool,
    /// Pour lava along the floor of the carved room.
    ///
    /// The only way to get a POINT light in a dark space. Everything else in the
    /// set is either unlit or wall-to-wall emitter, and neither judges a bloom,
    /// a falloff or a coloured splat -- which is most of the lighting stack.
    lava_floor: bool,
}

/// The set.
///
/// Chosen so that every system the overhaul touches appears in at least one
/// frame, and each one is here for a reason worth stating:
///
///   - `surface` — the default read, and the only frame with a sky in it.
///   - `deep-chamber` — rock in near-dark. Where a desaturated palette either
///     keeps its internal contrast or turns into one flat mass.
///   - `lava-sea` — the bloom's worst case, and the frame that shows whether an
///     additive pass CLIPS.
///   - `ore-chamber` — ore against rock. `BIOME_AMBIENT_ALPHA` was retuned from
///     0.6 to 0.05 because the veins had become invisible; a dark repaint pushes
///     the same nerve from the other side, and this is the frame that tells you.
///   - `lit-chamber` — the frame the whole lighting stack is judged on: a pool
///     of lava in a carved room, deep enough that the sky contributes nothing.
///     Bloom, falloff shape and the coloured splat are invisible in every other
///     frame here, because the rest are either unlit or wall-to-wall emitter.
///   - `night` / `dusk` — the atmosphere's range. Dusk at 0.735 is the measured
///     half-lit point, not 0.75.
///   - `seed777` / `seed42` — a different biome and a different surface height,
///     so a palette tuned to snow is not mistaken for a palette.
static SHOTS: &[Shot] = &[
    Shot {
        name: "surface",
        seed: None,
        time: None,
        depth: 0.0,
        carve: false,
        lava_floor: false,
    },
    Shot {
        name: "night",
        seed: None,
        time: Some(0.0),
        depth: 0.0,
        carve: false,
        lava_floor: false,
    },
    Shot {
        name: "dusk",
        seed: None,
        time: Some(0.735),
        depth: 0.0,
        carve: false,
        lava_floor: false,
    },
    Shot {
        name: "deep-chamber",
        seed: None,
        time: None,
        depth: 900.0,
        carve: true,
        lava_floor: false,
    },
    Shot {
        name: "ore-chamber",
        seed: None,
        time: None,
        depth: 1500.0,
        carve: true,
        lava_floor: false,
    },
    Shot {
        name: "lava-sea",
        seed: None,
        time: None,
        depth: 2900.0,
        carve: false,
        lava_floor: false,
    },
    Shot {
        name: "lit-chamber",
        seed: None,
        time: None,
        depth: 900.0,
        carve: true,
        lava_floor: true,
    },
    Shot {
        name: "seed777",
        seed: Some(777),
        time: None,
        depth: 0.0,
        carve: false,
        lava_floor: false,
    },
    Shot {
        name: "seed42",
        seed: Some(42),
        time: None,
        depth: 0.0,
        carve: false,
        lava_floor: false,
    },
];

#[test]
fn the_control_set_is_captured_and_measured() {
    if !gpu_is_available() {
        println!("SKIPPED control frames: no wgpu adapter on this machine");
        return;
    }

    let mut rows: Vec<String> = Vec::new();
    let mut problems: Vec<String> = Vec::new();
    let mut lumas: Vec<(&str, f64)> = Vec::new();

    // A dead render pipeline still produces a plausible PNG — the sky draws,
    // the UI draws, and the terrain silently is not there. That frame measures
    // fine and photographs fine, which is exactly the failure this rig exists
    // to refuse. So: any engine ERROR during the set (shader compile failures
    // land in `bevy_render`'s pipeline cache as ERROR lines) voids every number
    // above it.
    let errors_before = error_count();

    for shot in SHOTS {
        let (frame, air, lava) = capture(shot);
        let (mean, stddev) = frame.luma_spread();
        let (_, share) = frame.dominant();
        let distinct = frame.colours.len();
        lumas.push((shot.name, mean));

        // Each scene is what it claims. A rig that silently photographs solid
        // rock is worse than no rig: it produces a file, and the file looks like
        // a result.
        if shot.carve && !air {
            problems.push(format!(
                "{}: the chamber was not carved — this is a picture of solid rock",
                shot.name
            ));
        }
        if (shot.name == "lava-sea" || shot.lava_floor) && !lava {
            problems.push(format!(
                "{}: no lava within the room at this depth, so the bloom's worst \
                 case is not in the frame that exists to show it",
                shot.name
            ));
        }

        rows.push(format!(
            "{:<13} luma {mean:6.2} +/- {stddev:5.2}   distinct {distinct:5}   \
             dominant {:5.1}%   {}",
            shot.name,
            share * 100.0,
            frame.artefact(),
        ));
    }

    println!("\nCONTROL SET\n{}", rows.join("\n"));

    // The one cross-frame claim worth making, and the cheapest guard against a
    // day/night mix-up in the rig itself: midnight must be darker than the
    // default morning. If this inverts, `time` is not reaching the clock and
    // every "night" frame in the set is a second daylight frame.
    let day = lumas.iter().find(|(n, _)| *n == "surface").unwrap().1;
    let night = lumas.iter().find(|(n, _)| *n == "night").unwrap().1;
    if night >= day {
        problems.push(format!(
            "night ({night:.2}) is not darker than the morning default ({day:.2}) \
             — the clock is not being set and the night frames are daylight"
        ));
    }

    let errors = error_count() - errors_before;
    if errors > 0 {
        let by_source: Vec<String> = error_sources()
            .iter()
            .map(|(target, n)| format!("{target}: {n}"))
            .collect();
        problems.push(format!(
            "the engine logged {errors} ERROR lines while the set was captured \
             — every number above is a measurement of a broken renderer:\n  {}",
            by_source.join("\n  ")
        ));
    }

    assert!(problems.is_empty(), "{}", problems.join("\n  "));
}

/// Boot, place, settle, photograph. Returns the frame and the two scene facts.
fn capture(shot: &Shot) -> (Frame, bool, bool) {
    let mut app = headless_game("Yūgen control frames");
    if let Some(seed) = shot.seed {
        app.insert_resource(WorldSave { dir: None, seed });
    }
    if let Some(t) = shot.time {
        app.insert_resource(WorldClock(DayNight::new(t)));
    }
    app.finish();
    app.cleanup();
    app.world_mut()
        .resource_mut::<NextState<Scene>>()
        .set(Scene::Playing);
    for _ in 0..SETTLE {
        app.update();
    }

    // Off the body BEFORE moving the view: `player::follow_player` eases the
    // focus back onto the body every frame and is gated on `FocusDriver::Player`,
    // so flipping the driver is what lets the camera stay where it is put.
    let focus_x = app.world().resource::<WorldFocus>().x;
    let depth_px = shot.depth * WorldScale::LIVE.factor() as f32;
    let target_y = app.world().resource::<WorldFocus>().y + depth_px;
    if depth_px > 0.0 {
        *app.world_mut().resource_mut::<FocusDriver>() = FocusDriver::FreeCamera;
        for _ in 0..DEEP_SETTLE {
            app.world_mut().resource_mut::<WorldFocus>().y = target_y;
            app.update();
        }
    }

    if shot.carve {
        carve(&mut app, focus_x, target_y, shot.lava_floor);
        for _ in 0..SETTLE {
            app.world_mut().resource_mut::<WorldFocus>().y = target_y;
            app.update();
        }
    }

    let (air, lava) = sample(&app, focus_x, target_y);
    let png = out_path(shot.name);
    let image = capture_low_res(&mut app, &png);
    (Frame::from_image(&image, png), air, lava)
}

/// Carve a room at the focus and lay a stone floor under it.
///
/// The floor is stated rather than hoped for, which is the fix `lit_scene.rs`
/// records: 900 px down, whatever worldgen left below the carve is as likely to
/// be a void as rock, and over a void the room drains into the dark.
fn carve(app: &mut App, focus_x: f32, focus_y: f32, lava_floor: bool) {
    let stone = code_of("stone");
    let (cx, cy) = (cell_at(focus_x), cell_at(focus_y));
    let mut world = app.world_mut().resource_mut::<SimWorld>();
    for dy in -ROOM_H..ROOM_H {
        for dx in -ROOM_W..ROOM_W {
            world
                .level
                .grid
                .set_world(WorldCell::new(cx + dx, cy + dy), EMPTY);
        }
    }
    for dx in (-ROOM_W - 1)..=ROOM_W {
        world
            .level
            .grid
            .set_world(WorldCell::new(cx + dx, cy + ROOM_H), stone);
    }
    // A POOL, not a point. One emitter shows the falloff; only a broad source
    // shows whether the pass clips, which is what `lit_scene.rs` was built to
    // catch and what it caught.
    if lava_floor {
        let lava = code_of("lava");
        for dy in (ROOM_H - 4)..ROOM_H {
            for dx in -ROOM_W..ROOM_W {
                world
                    .level
                    .grid
                    .set_world(WorldCell::new(cx + dx, cy + dy), lava);
            }
        }
    }
}

/// `(the focus cell is air, there is lava within the room)`.
fn sample(app: &App, focus_x: f32, focus_y: f32) -> (bool, bool) {
    let world = app.world().resource::<SimWorld>();
    let (cx, cy) = (cell_at(focus_x), cell_at(focus_y));
    let air = world.level.grid.get_world(WorldCell::new(cx, cy)) == EMPTY;
    let lava = code_of("lava");
    let mut found = false;
    for dy in -ROOM_H..ROOM_H {
        for dx in -ROOM_W..ROOM_W {
            if world.level.grid.get_world(WorldCell::new(cx + dx, cy + dy)) == lava {
                found = true;
            }
        }
    }
    (air, found)
}

/// Where the PNGs land. Inside `target/`, which is line 1 of `.gitignore`:
/// these are artefacts to look at, not fixtures to commit.
fn out_path(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("control-frames");
    std::fs::create_dir_all(&dir).expect("target/ is writable");
    dir.join(format!("{name}.png"))
}

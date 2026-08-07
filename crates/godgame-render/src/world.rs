//! The owned world, and the schedule that ticks it.
//!
//! # This module is the replacement for the Web Worker
//!
//! It is the single biggest architectural change in the port, so it is written
//! down here rather than left to be inferred.
//!
//! The TypeScript ran the cellular automata on a `Worker`. The grid's four
//! planes lived in a `SharedArrayBuffer` that both threads held a typed-array
//! view onto, plus a small shared control block for the window origin. That
//! bought parallelism and cost:
//!
//!   - **Tearing.** The worker wrote cells on its own `setInterval` at `SIM_HZ`,
//!     unsynchronised with `requestAnimationFrame`. The renderer could and did
//!     read a grid halfway through a sweep.
//!   - **Two owners of one window.** Both threads built a `WindowManager` over
//!     the same cells, which is the entire reason `attach()` existed (see
//!     [`godgame_core::sim::window`]).
//!   - **Invisible dirty masks.** Each `CellGrid` instance had its own
//!     per-chunk dirty bits and the worker's were the live ones. The main
//!     thread could not see them, so the renderer had no choice but to repaint
//!     the whole viewport every frame. [`crate::cellmap`] is what that
//!     restriction cost, undone.
//!   - **Cross-origin isolation.** `SharedArrayBuffer` needs COOP/COEP headers,
//!     which is a deployment constraint on a single-player game.
//!
//! Here there is ONE `CellGrid`, owned by one resource, mutated through `&mut`
//! from one schedule. No shared memory, no message hop, no `unsafe`. The sim
//! cannot tear against the renderer because the renderer reads it in a later
//! system of the same frame.
//!
//! # The clock
//!
//! [`bevy::time::Fixed`] runs at **120 Hz**, which is
//! [`STEP_DT`](godgame_core::config::STEP_DT) — the player's physics step. The
//! automata ticks on every SECOND fixed step, which is
//! [`SIM_HZ`](godgame_core::config::SIM_HZ) = 60. One clock, two rates, an
//! exact 2:1 ratio: the player never integrates against a stale grid and the
//! sim never runs twice between two player steps.
//!
//! `Time<Virtual>`'s max delta is clamped to
//! [`MAX_STEPS_PER_FRAME`](godgame_core::config::MAX_STEPS_PER_FRAME) steps, so
//! a stalled frame drops simulation time instead of trying to catch up forever.

use std::time::Duration;

use bevy::prelude::*;

use godgame_core::config::{MAX_STEPS_PER_FRAME, SEED, STEP_DT, cell_at};
use godgame_core::sim::automata::Automata;
use godgame_core::sim::chunk_store::{ChunkPersistence, ChunkStore};
use godgame_core::sim::grid::CellGrid;
use godgame_core::sim::level::{Level, window_size};
use godgame_core::sim::save::DiskChunkPersistence;
use godgame_core::sim::window::WindowManager;
use godgame_core::sim::worldgen::{SPAWN_COL, walkable_spawn};

/// System sets inside [`FixedUpdate`], in run order.
///
/// Named so that everything which steps between the two can say so rather than
/// infer it. The slot exists because a stepper reads the grid the streamer just
/// recentred and writes the focus the next stream will read, and that ordering
/// is not recoverable from the code once it is wrong.
///
/// Four things hang there now: [`crate::player`]'s body (which also drives the
/// arrow pool from inside `Player::step`), [`crate::mobs`]' creatures,
/// [`crate::items`]' dropped stacks, and [`crate::particles`]. All of them order
/// against [`crate::player::PlayerSet::Step`] rather than against each other.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimSet {
    /// Recentre the streaming window on [`WorldFocus`].
    Stream,
    /// Advance the automata, on even steps only.
    Simulate,
}

/// The whole simulated world, in one resource.
///
/// One resource rather than three because the three are mutated together every
/// tick and Bevy's borrow checker is per-resource: `WindowManager::recenter`
/// takes `&mut CellGrid`, which lives inside `Level`, so splitting them would
/// only move the problem into a system parameter conflict.
#[derive(Resource)]
pub struct SimWorld {
    /// The streaming window and its spawn point.
    pub level: Level,
    /// Streams chunks through [`SimWorld::level`]'s grid.
    pub window: WindowManager,
    /// The falling-sand automata's cross-tick state.
    pub automata: Automata,
    /// The world seed everything above was built from.
    pub seed: u32,
}

/// Where the camera — and later the player — is, in world px.
///
/// The streaming window recentres on this and the world camera follows it. It
/// is a resource rather than a component so the binary's input code and the
/// streamer do not have to agree on an entity.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct WorldFocus {
    /// World px, +x right.
    pub x: f32,
    /// World px, +y DOWN — the sim's convention, not Bevy's.
    pub y: f32,
}

/// Fixed steps elapsed. Its parity is what makes the sim 60 Hz on a 120 Hz clock.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct FixedStep(pub u64);

impl FixedStep {
    /// Whether the automata ticks on this step.
    #[inline]
    pub const fn is_sim_step(self) -> bool {
        self.0.is_multiple_of(2)
    }
}

/// The owned world, the 120 Hz clock, and the streaming that keeps the window
/// under the camera.
pub struct WorldSimPlugin;

impl Plugin for WorldSimPlugin {
    fn build(&self, app: &mut App) {
        // 120 Hz, from the player's own step, not a literal. `from_seconds`
        // rather than `from_hz` so the reciprocal is taken once in f64 rather
        // than round-tripping an f32 1/120 back through a division.
        app.insert_resource(Time::<Fixed>::from_seconds(f64::from(STEP_DT)))
            .init_resource::<WorldFocus>()
            // Default is `None` — unsaved, which is what every milestone before
            // this one did. The binary's `--world` overwrites it before Startup.
            .init_resource::<WorldSave>()
            .init_resource::<FixedStep>()
            .add_systems(Startup, (clamp_catch_up, spawn_world).chain())
            .configure_sets(FixedUpdate, (SimSet::Stream, SimSet::Simulate).chain())
            .add_systems(
                FixedUpdate,
                (
                    stream_window.in_set(SimSet::Stream),
                    simulate.in_set(SimSet::Simulate),
                )
                    .run_if(resource_exists::<SimWorld>),
            )
            // `Last`, so a frame's edits are already in the grid, and gated on a
            // save directory existing so an unsaved run pays nothing at all.
            .add_systems(
                Last,
                autosave
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(|save: Res<WorldSave>| save.0.is_some()),
            );
    }
}

/// Flush the world to disk on an interval, and once more on the way out.
///
/// Two triggers rather than one, because neither is enough on its own. An
/// interval alone loses up to [`AUTOSAVE_EVERY_S`] of digging every time somebody
/// quits normally, which is most of the time. Quit alone loses EVERYTHING if the
/// process dies without getting there — a crash, a kill, a lid closing on a
/// laptop that never wakes — which is rarer and much worse.
///
/// The exit path reads `AppExit` in `Last` rather than hooking a shutdown
/// callback: Bevy writes that message and then finishes the frame, so this is
/// still a normal system with normal access to the world, and there is exactly
/// one code path that saves.
fn autosave(
    time: Res<Time>,
    mut world: ResMut<SimWorld>,
    mut since: Local<f32>,
    exiting: MessageReader<AppExit>,
) {
    *since += time.delta_secs();
    let leaving = !exiting.is_empty();
    if !leaving && *since < AUTOSAVE_EVERY_S {
        return;
    }
    *since = 0.0;
    let SimWorld { level, window, .. } = &mut *world;
    window.flush(&level.grid);
    info!(
        "AUTOSAVE fired leaving={leaving} persisted={}",
        window.store().persisted_len()
    );
}

/// Cap how much simulation a single slow frame may try to catch up on.
fn clamp_catch_up(mut virt: ResMut<Time<Virtual>>) {
    virt.set_max_delta(Duration::from_secs_f32(
        STEP_DT * MAX_STEPS_PER_FRAME as f32,
    ));
}

/// Seconds between autosaves while a world with a save directory is running.
///
/// Thirty. The flush is cheap — only chunks that differ from worldgen are
/// written, and only the ones marked diverged reach the disk — and the cost of
/// getting this wrong is asymmetric: a save that is thirty seconds stale after a
/// crash is a annoyance, and one that never happened is the run.
const AUTOSAVE_EVERY_S: f32 = 30.0;

/// Where a world's chunk edits are kept, if anywhere.
///
/// Inserted by the host BEFORE `Startup` — `--world DIR` on the binary — and
/// absent by default, which is the behaviour every previous milestone had: edits
/// survive eviction and revisiting but not the process.
///
/// A resource and not an argument to [`build_world`], because there are TWO
/// places a world is built — `spawn_world` here and `glue::start_a_run` on every
/// entry to `Scene::Playing` — and a path threaded through one of them is a path
/// the other silently drops. It did: the first wiring passed the directory to
/// `spawn_world` only, so opening the game with `--world` logged the save
/// directory and then immediately replaced that world with an unsaved one. Both
/// read this resource now.
#[derive(Resource, Clone, Debug, Default)]
pub struct WorldSave(pub Option<std::path::PathBuf>);

/// Generate the world and fill the streaming window once, at load.
///
/// This is the expensive startup step — `WindowManager::init` generates all
/// `WINDOW_CHUNKS_X * WINDOW_CHUNKS_Y` chunks — and it is deliberately
/// synchronous. A loading screen is a later milestone's problem; correctness of
/// the first frame is this one's.
fn spawn_world(mut commands: Commands, mut focus: ResMut<WorldFocus>, save: Res<WorldSave>) {
    let world = build_world_saved(SEED, save.0.as_deref());
    *focus = WorldFocus {
        x: world.level.spawn.x,
        y: world.level.spawn.y,
    };
    commands.insert_resource(world);
}

/// Generate a world from a seed, with its streaming window already filled.
///
/// Split out of [`spawn_world`] because a respawn needs exactly this and must
/// not reimplement it: the TypeScript's `Game.loadLevel` built the grid, the
/// window and the automata together, and a second copy of that sequence is how
/// the two silently drift apart. `crate::glue` calls it when a run restarts.
///
/// A fresh grid rather than a cleared one, which is what makes a restart discard
/// the player's excavation — the same thing `loadLevel` did by allocating a new
/// `CellGrid`. The world is a pure function of the seed, so what comes back is
/// the same terrain, minus every hole that was dug in it.
pub fn build_world(seed: u32) -> SimWorld {
    build_world_saved(seed, None)
}

/// [`build_world`], with somewhere to keep the player's edits.
///
/// `None` is the in-memory backend every milestone before this one used: edits
/// survive eviction and revisiting, and die with the process. `Some(dir)` puts
/// them on disk, so the hole you dug is there when you come back.
///
/// A failure to open the directory falls back to memory and says so rather than
/// refusing to start. A player who mistyped a path wants to play; what they must
/// not get is a world that silently pretends to save.
pub fn build_world_saved(seed: u32, save: Option<&std::path::Path>) -> SimWorld {
    // `walkable_spawn` and not `spawn_point`: the latter asks the heightmap
    // where the land is and the heightmap knows nothing about the trees the
    // decorators put on it. Every leaf material is authored `collides = true`,
    // so a canopy is a wall, and a third of seeds measured put the body inside
    // one — two of the first twenty-four with nothing clear on either side. The
    // body could not move at all. See `walkable_spawn`.
    let spawn = walkable_spawn(seed, SPAWN_COL);

    let (cols, rows) = window_size();
    let mut grid = CellGrid::new(cols, rows);

    let store = match save {
        Some(dir) => match DiskChunkPersistence::open(dir.join("chunks")) {
            Ok(disk) => {
                info!(
                    "world save: {} ({} chunks on disk)",
                    dir.display(),
                    disk.len()
                );
                ChunkStore::with_persistence(seed, Box::new(disk))
            }
            Err(e) => {
                error!(
                    "world save: cannot use {}: {e} — playing unsaved",
                    dir.display()
                );
                ChunkStore::new(seed)
            }
        },
        None => ChunkStore::new(seed),
    };
    let mut window = WindowManager::new(store);
    window.init(&mut grid, cell_at(spawn.x), cell_at(spawn.y));

    let mut automata = Automata::new();
    automata.seed(seed);

    info!(
        "world {seed} ready: spawn ({:.0}, {:.0}), window {cols}x{rows} cells",
        spawn.x, spawn.y
    );

    SimWorld {
        level: Level::new(grid, spawn),
        window,
        automata,
        seed,
    }
}

/// Keep the streaming window centred on [`WorldFocus`].
///
/// Cheap to call every step: `recenter` early-returns unless the focus has
/// drifted a whole chunk past the window centre.
fn stream_window(mut world: ResMut<SimWorld>, focus: Res<WorldFocus>) {
    let SimWorld { level, window, .. } = &mut *world;
    window.recenter(&mut level.grid, cell_at(focus.x), cell_at(focus.y));
}

/// Advance the automata on even fixed steps — 60 Hz on a 120 Hz clock.
fn simulate(mut world: ResMut<SimWorld>, mut step: ResMut<FixedStep>) {
    step.0 = step.0.wrapping_add(1);
    if !step.is_sim_step() {
        return;
    }
    let SimWorld {
        level, automata, ..
    } = &mut *world;
    automata.simulate(&mut level.grid);
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::config::SIM_HZ;

    #[test]
    fn every_second_fixed_step_is_a_sim_step() {
        let fixed_hz = (1.0 / STEP_DT).round() as u32;
        assert_eq!(
            fixed_hz,
            SIM_HZ * 2,
            "the 2:1 fixed:sim ratio this module ticks on is wrong"
        );

        let sim_steps = (0..fixed_hz)
            .filter(|i| FixedStep(u64::from(*i)).is_sim_step())
            .count();
        assert_eq!(sim_steps as u32, SIM_HZ, "sim rate is not SIM_HZ");
    }
}

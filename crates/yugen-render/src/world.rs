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
//!     [`yugen_core::sim::window`]).
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
//! [`STEP_DT`](yugen_core::config::STEP_DT) — the player's physics step. The
//! automata ticks on every SECOND fixed step, which is
//! [`SIM_HZ`](yugen_core::config::SIM_HZ) = 60. One clock, two rates, an
//! exact 2:1 ratio: the player never integrates against a stale grid and the
//! sim never runs twice between two player steps.
//!
//! `Time<Virtual>`'s max delta is clamped to
//! [`MAX_STEPS_PER_FRAME`](yugen_core::config::MAX_STEPS_PER_FRAME) steps, so
//! a stalled frame drops simulation time instead of trying to catch up forever.

use std::time::Duration;

use bevy::prelude::*;

use crate::daynight::{DayNight, WorldClock};
use crate::items::Pack;
use crate::player::PlayerBody;
use yugen_core::config::{MAX_STEPS_PER_FRAME, SEED, STEP_DT, cell_at};
use yugen_core::items::inventory::SLOT_COUNT;
use yugen_core::sim::automata::Automata;
use yugen_core::sim::chunk_store::{ChunkPersistence, ChunkStore};
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::level::{Level, window_size};
use yugen_core::sim::save::{BodyState, DiskChunkPersistence, RunState, write_run};
use yugen_core::sim::window::WindowManager;
use yugen_core::sim::worldgen::{SPAWN_COL, walkable_spawn};

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
                    .run_if(|save: Res<WorldSave>| save.dir.is_some()),
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
    run: RunSources,
    mut since: Local<f32>,
    exiting: MessageReader<AppExit>,
) {
    *since += time.delta_secs();
    if exiting.is_empty() && *since < AUTOSAVE_EVERY_S {
        return;
    }
    *since = 0.0;

    let SimWorld { level, window, .. } = &mut *world;
    window.flush(&level.grid);

    let Some(dir) = run.save.dir.as_deref() else {
        return;
    };
    if let Err(e) = write_run(dir, &run.snapshot(world.seed)) {
        // A warning and not a panic. The terrain is already on disk by this
        // point, so a lost run file costs the player their position and pack and
        // not their world — and taking the process down on the way out would be
        // a strange way to report it.
        warn!("world save: could not write the run file: {e}");
    }
}

/// The parts of a run that are not terrain, on the way out and on the way in.
///
/// Every field optional for `glue`'s reason: a host may run the scene machine
/// without the whole game — the capture rigs do — and a missing resource means
/// that system is not in the app rather than that something failed.
#[derive(bevy::ecs::system::SystemParam)]
pub struct RunSources<'w> {
    save: Res<'w, WorldSave>,
    body: Option<Res<'w, PlayerBody>>,
    pack: Option<Res<'w, Pack>>,
    clock: Option<Res<'w, WorldClock>>,
}

impl RunSources<'_> {
    /// What to write.
    fn snapshot(&self, seed: u32) -> RunState {
        RunState {
            seed,
            clock_t: self.clock.as_ref().map_or(0.0, |c| c.0.t()),
            body: self.body.as_ref().map(|b| BodyState {
                x: b.0.x,
                y: b.0.y,
                vx: b.0.vx,
                vy: b.0.vy,
                facing: b.0.facing,
                health: b.0.health,
                untouchable: b.0.untouchable,
            }),
            slots: self.pack.as_ref().map_or_else(Vec::new, |p| {
                (0..SLOT_COUNT)
                    .filter_map(|i| p.0.stack_at(i).map(|(code, n)| (i as u16, code, n)))
                    .collect()
            }),
            selected: self.pack.as_ref().map_or(0, |p| p.0.selected() as u16),
            worn: self.pack.as_ref().and_then(|p| p.0.worn()),
        }
    }
}

/// Move the body and the camera to a point, and let the world stream to it.
///
/// The window is NOT recentred here. `stream_window` does that every frame from
/// the focus, so writing the focus is the whole of it — and reaching into the
/// window as well would be a second thing to keep in step with the first.
pub fn place_body(
    at: crate::scene::StartAt,
    focus: &mut WorldFocus,
    body: Option<&mut PlayerBody>,
) {
    focus.x = at.x;
    focus.y = at.y;
    if let Some(body) = body {
        body.0.x = at.x;
        body.0.y = at.y;
        // Whatever it was doing before it was moved, it is not doing now. A body
        // teleported mid-fall keeps its downward velocity and is through the
        // floor before the first frame is drawn.
        body.0.vx = 0.0;
        body.0.vy = 0.0;
    }
}

/// Put a loaded run back into the live resources.
///
/// Public because the thing that builds a world and the thing that resets a run
/// are in different modules — `crate::glue` owns the second — and a restore that
/// lived in only one of them would be a restore the other silently skipped. That
/// is the shape of the bug this module already had once, with `WorldSave`.
pub fn restore_run(
    run: &RunState,
    focus: &mut WorldFocus,
    body: Option<&mut PlayerBody>,
    pack: Option<&mut Pack>,
    clock: Option<&mut WorldClock>,
) {
    if let (Some(body), Some(b)) = (body, run.body) {
        body.0.x = b.x;
        body.0.y = b.y;
        body.0.vx = b.vx;
        body.0.vy = b.vy;
        body.0.facing = b.facing;
        body.0.health = b.health;
        body.0.untouchable = b.untouchable;
        // The camera goes where the BODY is, not where the spawn is. Without
        // this the world streams in around a point the player is not at and the
        // first frame looks at somewhere else entirely.
        focus.x = b.x;
        focus.y = b.y;
    }
    if let Some(pack) = pack {
        // Cleared first: the starting kit has already been handed out by the
        // time this runs, and a restore that only wrote the saved slots would
        // leave a second pick in whatever slot the kit used.
        pack.0.clear();
        for (slot, code, n) in &run.slots {
            pack.0.put_at(*slot as usize, *code, *n);
        }
        pack.0.select_slot(run.selected as usize);
        // What was worn goes back on. Through `put_at`-style restoration rather
        // than `equip`, which would take the piece out of a pack slot it is not
        // in — the saved pack and the saved armour are two separate facts and
        // restoring one must not consume the other.
        if let Some(code) = run.worn {
            pack.0.wear_restored(code);
        }
    }
    if let Some(clock) = clock {
        clock.0 = DayNight::new(run.clock_t);
    }
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
#[derive(Resource, Clone, Debug)]
pub struct WorldSave {
    /// Where this world's chunks and run file live, or `None` to play unsaved.
    pub dir: Option<std::path::PathBuf>,
    /// The seed to grow it from.
    ///
    /// **The first thing in this tree to make the seed a RUNTIME value.** Every
    /// production path built its world from the `SEED` constant, which is why
    /// `light::LightGrid::follow_seed` and `ambience`'s equivalent were written
    /// against a hazard nobody could reach: both keep a private `Noise` and
    /// `Heightmap`, and a world grown from a different seed would have left the
    /// lighting flooding sky to one surface line while the terrain sat at
    /// another, silently. They guard themselves, so this is safe to turn on —
    /// and it is now worth having a test that says so rather than a comment.
    pub seed: u32,
}

impl Default for WorldSave {
    fn default() -> WorldSave {
        WorldSave {
            dir: None,
            seed: SEED,
        }
    }
}

/// Generate the world and fill the streaming window once, at load.
///
/// This is the expensive startup step — `WindowManager::init` generates all
/// `WINDOW_CHUNKS_X * WINDOW_CHUNKS_Y` chunks — and it is deliberately
/// synchronous. A loading screen is a later milestone's problem; correctness of
/// the first frame is this one's.
fn spawn_world(mut commands: Commands, mut focus: ResMut<WorldFocus>, save: Res<WorldSave>) {
    let world = build_world_saved(save.seed, save.dir.as_deref());
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
    use yugen_core::config::SIM_HZ;

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

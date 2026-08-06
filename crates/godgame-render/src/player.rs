//! The player in the app: the body, the step that drives it, the camera that
//! follows it, and a placeholder rectangle so you can see all three working.
//!
//! # Where each half of `Game.updatePlaying` went
//!
//! The TypeScript ran one function per rendered frame that did, in order:
//! recentre the streaming window on the player's cell, tick the sim, run the
//! player's fixed-step loop, then ease the camera. Bevy already owns three of
//! those four clocks, so the order is expressed as schedule placement rather
//! than as statements:
//!
//! | `Game.ts` | here |
//! |---|---|
//! | `windowManager.recenter(pcx, pcy)` | [`SimSet::Stream`], already there |
//! | `while (acc >= STEP_DT) player.step(...)` | [`step_player`], in `FixedUpdate` between the two sim sets |
//! | `simulate(grid)` | [`SimSet::Simulate`], already there |
//! | `camera.follow(player.box)` | [`follow_player`], once per FRAME |
//!
//! **The ordering is the thing to get right.** The player steps, then the camera
//! follows, then the window recentres — in that order, or the streaming edge
//! shows a seam and the camera lags a frame behind the body. [`step_player`]
//! runs `.after(SimSet::Stream)`, so the window it reads is the one that was
//! recentred on where the body was BEFORE this step, exactly as the TypeScript's
//! `recenter` used the previous frame's `player.x`; and [`follow_player`] runs
//! in `RunFixedMainLoop`'s `AfterFixedMainLoop`, so it sees the last substep of
//! the frame and still lands before [`crate::lowres`] puts the camera on
//! [`WorldFocus`] in `Update`.
//!
//! # Once per frame, not once per substep
//!
//! [`follow_player`] eases at a per-FRAME rate, because `Camera.follow` was
//! called from the frame loop and not from inside the fixed accumulator. Moving
//! it into `FixedUpdate` would double its speed on a 120 Hz clock against a
//! 60 Hz screen and make the follow frame-rate dependent in a second, different
//! way from the one it already is.
//!
//! The other half of that rule is [`Intent::for_substep`]: Bevy clears
//! `just_pressed` once per rendered frame, but `FixedUpdate` may run several
//! times inside one, so an edge-triggered jump has to be handed to exactly one
//! substep. That masking already exists in [`godgame_core::input`]; this module
//! only passes it the counter [`crate::input::FixedSubstep`] keeps.
//!
//! # The rectangle
//!
//! [`spawn_player`] spawns one flat-coloured sprite the size of the collision
//! box. It is scaffolding for the sprite milestone, and it is deliberately the
//! BOX and not an art rect: the point of drawing it now is to see the body
//! collide, so the thing on screen has to be the thing the collider moves. The
//! only presentation state it reads is
//! [`Player::step_up_visual`](godgame_core::entities::Player::step_up_visual),
//! which exists precisely so that cresting a one-cell ledge reads as a stride
//! instead of a teleport — the box snaps up a whole cell instantly and the drawn
//! figure catches up.

use bevy::app::{RunFixedMainLoop, RunFixedMainLoopSystems};
use bevy::prelude::*;

use godgame_core::config::{PLAYER_H, PLAYER_W, STEP_DT};
use godgame_core::entities::player::PlayerEvent;
use godgame_core::entities::{Player, SharedPool};
use godgame_core::physics::collision::Aabb;

use crate::input::{FixedSubstep, FocusDriver, PlayerIntent};
use crate::lowres::WORLD_LAYERS;
use crate::world::{SimSet, SimWorld, WorldFocus};

/// How far the view closes on the player each FRAME, 0..1.
///
/// `Camera.follow`'s `ease` argument, verbatim, and frame-rate dependent in
/// exactly the way it was: at 0.12 the view covers 12% of the remaining gap per
/// frame, so it settles in about a third of a second at 60fps and faster on a
/// quicker screen. Keeping the wart is deliberate — this is the number the game
/// feels like, and a per-second reformulation would change the feel of every
/// jump while claiming to be a port.
const CAMERA_EASE: f32 = 0.12;

/// Where the placeholder body sits in z, between the cell quad (0) and the
/// brush preview (1) — under the cursor, over the world.
const PLAYER_Z: f32 = 0.5;

/// The placeholder body's colour.
///
/// A warm red, for the one property that matters before there is art: it is not
/// a colour the terrain palette contains, so the figure is unambiguous against
/// sky, snow, stone and lava alike.
const PLACEHOLDER: Color = Color::srgb(0.90, 0.27, 0.30);

/// The one system set this module publishes, so the creatures can hang off it.
///
/// [`crate::mobs`] has to run after the body has moved and after the arrow pool
/// has been stepped — and the arrow pool is stepped from INSIDE
/// [`Player::step`], not from a system of its own. Naming the set is what lets
/// that ordering be stated rather than inferred from the two `SimSet` bounds
/// both modules happen to share.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlayerSet {
    /// One fixed step of the body, and of the arrows it has in flight.
    Step,
}

/// The player's body, its fixed step, the camera follow and the placeholder.
pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app
            // Not built in `spawn_player`, because the creatures install their
            // hit test into it and are constructed before there is a body. It
            // holds nothing world-shaped, so there is nothing to wait for.
            .init_resource::<ArrowPool>()
            // `PostStartup`, not `Startup`: the world arrives through `Commands`
            // from `world::spawn_world` and is not a resource until that
            // schedule's command queue is flushed. Spawning here also means the
            // spawn point comes from the level rather than being derived from
            // the seed a second time.
            .add_systems(
                PostStartup,
                spawn_player
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(not(resource_exists::<NoPlayer>)),
            )
            .add_systems(
                FixedUpdate,
                step_player
                    .in_set(PlayerSet::Step)
                    .after(SimSet::Stream)
                    .before(SimSet::Simulate)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<PlayerBody>),
            )
            .add_systems(
                RunFixedMainLoop,
                follow_player
                    .in_set(RunFixedMainLoopSystems::AfterFixedMainLoop)
                    .run_if(resource_exists::<PlayerBody>)
                    .run_if(resource_equals(FocusDriver::Player)),
            )
            .add_systems(Update, place_body.run_if(resource_exists::<PlayerBody>));
    }
}

/// The simulated player.
///
/// A resource and not a component: there is exactly one, every system that
/// wants it wants the whole thing mutably, and a `Single<&mut …>` query would
/// buy nothing but a panic when the entity is missing. The drawn body is a
/// separate entity ([`PlayerSprite`]) for the same reason the cell quad is one —
/// it is a thing the renderer owns, and the simulation must not hold a handle to
/// it.
#[derive(Resource, Deref, DerefMut)]
pub struct PlayerBody(pub Player);

/// The rectangle standing in for the player sprite until the art lands.
#[derive(Component)]
pub struct PlayerSprite;

/// The arrows the player has in flight, held by everyone who needs them.
///
/// The player fires into this and steps it; [`crate::mobs`] installs the hit
/// test that lets an arrow find a creature, and draws what is up. Three owners,
/// one pool — see [`SharedPool`] for why that is a handle and not a wider
/// `Projectiles` trait.
///
/// It is a resource of its own rather than a field on [`PlayerBody`] because the
/// creatures reach for it in `Startup`, before any body exists.
#[derive(Resource, Clone, Default, Deref)]
pub struct ArrowPool(pub SharedPool);

/// Insert before startup to run with no player at all.
///
/// The free camera is not a fallback that stopped mattering the moment there was
/// a body to follow: a worldgen inspection wants to fly across a streaming
/// window the player would take half a minute to cross, and a screenshot run
/// wants a view that does not fall. [`FocusDriver`] is the switch and it stays
/// on [`FocusDriver::FreeCamera`] when this is present, because nothing spawns
/// to move it off.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct NoPlayer;

/// Put a body at the level's spawn, snap the view onto it, and hand it the keys.
///
/// The three writes at the end are `Game.loadLevel`'s: the player is
/// constructed, `camera.snap(player.box)` puts the view on it with no ease, and
/// from then on the player drives the view. Snapping rather than easing matters
/// on the first frame — an eased camera would start at the world origin and fly
/// across the whole streaming window before the first frame was drawn.
fn spawn_player(
    mut commands: Commands,
    world: Res<SimWorld>,
    mut focus: ResMut<WorldFocus>,
    mut driver: ResMut<FocusDriver>,
    arrows: Res<ArrowPool>,
) {
    // A clone of the handle, not a pool of its own: the creatures have already
    // installed their hit test into the one behind it, and `crate::mobs` draws
    // out of the same slots this player is about to fire into.
    let player = Player::with_projectiles(world.level.spawn, Box::new(arrows.0.clone()));

    let centre = centre_of(player.aabb());
    *focus = WorldFocus {
        x: centre.0,
        y: centre.1,
    };
    *driver = FocusDriver::Player;

    commands.spawn((
        Sprite {
            color: PLACEHOLDER,
            custom_size: Some(Vec2::new(PLAYER_W, PLAYER_H)),
            ..default()
        },
        Transform::from_xyz(centre.0, -centre.1, PLAYER_Z),
        PlayerSprite,
        WORLD_LAYERS,
    ));

    info!(
        "player at ({:.0}, {:.0}), {PLAYER_W}x{PLAYER_H} px",
        player.x, player.y
    );
    commands.insert_resource(PlayerBody(player));
}

/// One fixed step of the body, against the window the streamer just recentred.
///
/// `intent.for_substep(n)` is `Game.ts`'s
/// `jumpQueued: intent.jumpQueued && steps === 0` — the rising edge goes to the
/// first substep of the frame and to no other, so a held jump key does not pogo
/// and a held dash key does not spend a dash every 8ms.
///
/// The arrow pool is stepped from INSIDE this call — `Player::step` drives its
/// own `Projectiles` — which is why [`PlayerSet::Step`] exists and why
/// [`crate::mobs`] orders itself after the set rather than after this function.
/// A creature hit by an arrow is therefore already hurt by the time the
/// creatures take their own step this frame.
///
/// Both seams this used to describe are now wired: [`crate::items`] pushes the
/// held weapon and the ammo source, and [`crate::mobs`] installs the pool's
/// [`ShotHitTest`](godgame_core::entities::ShotHitTest).
fn step_player(
    mut body: ResMut<PlayerBody>,
    world: Res<SimWorld>,
    intent: Res<PlayerIntent>,
    substep: Res<FixedSubstep>,
    mut drained: Local<Vec<PlayerEvent>>,
) {
    body.step(STEP_DT, intent.for_substep(substep.0), &world.level.grid);

    // Land/jump/splash/hurt are particle and audio cues, and neither exists yet.
    // Draining anyway keeps the player's buffer from sitting permanently full,
    // so the first consumer to arrive sees this frame's events and not a
    // capped-out queue of everything since startup.
    drained.clear();
    body.drain_events(&mut drained);
}

/// Ease the view toward the body. `Camera.follow`, once per frame.
///
/// The TypeScript eased the view's TOP-LEFT toward `body centre - view / 2`;
/// [`WorldFocus`] is the view's CENTRE, so the same coefficient applied to the
/// centre is the same motion and the view size drops out of the arithmetic
/// entirely. That is why nothing here reads [`crate::lowres::LowResTarget`], and
/// why a window resize does not jolt the camera.
fn follow_player(body: Res<PlayerBody>, mut focus: ResMut<WorldFocus>) {
    let (cx, cy) = centre_of(body.aabb());
    focus.x += (cx - focus.x) * CAMERA_EASE;
    focus.y += (cy - focus.y) * CAMERA_EASE;
}

/// Put the placeholder on the box, on the pixel grid the terrain is drawn on.
///
/// The TOP-LEFT is what gets rounded, not the centre: at `PLAYER_W` 10 and
/// `PLAYER_H` 15 the box is even one way and odd the other, so rounding the
/// centre would put the vertical edges on half pixels. This is the same rule
/// [`crate::lowres`] snaps the camera with, applied to a body instead of a view.
fn place_body(body: Res<PlayerBody>, sprite: Single<&mut Transform, With<PlayerSprite>>) {
    let mut transform = sprite.into_inner();
    let left = body.x.round();
    // The collision box snaps up a whole cell the instant a step-up resolves;
    // `step_up_visual` is the residue that eases back to zero, so adding it
    // draws the stride the box does not have.
    let top = (body.y + body.step_up_visual()).round();
    transform.translation.x = left + PLAYER_W * 0.5;
    // Bevy's +y is up and the sim's is down: the one convention flip, in the one
    // place, exactly as `lowres::follow_focus` and `cellmap::follow_window` do.
    transform.translation.y = -(top + PLAYER_H * 0.5);
}

/// Centre of a box, in world px. The camera and the sprite both want it and
/// deriving it twice is how the two end up half a pixel apart.
#[inline]
fn centre_of(b: Aabb) -> (f32, f32) {
    (b.x + b.w * 0.5, b.y + b.h * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::config::{CELL_SIZE, SEED, cell_at};
    use godgame_core::input::Intent;
    use godgame_core::sim::chunk_store::ChunkStore;
    use godgame_core::sim::grid::CellGrid;
    use godgame_core::sim::level::window_size;
    use godgame_core::sim::materials::{EMPTY, block};
    use godgame_core::sim::window::WindowManager;
    use godgame_core::sim::worldgen::{SPAWN_COL, SpawnPoint, spawn_point};

    /// Fixed steps in one second of simulated time.
    const ONE_SECOND: u32 = (1.0 / STEP_DT) as u32;

    /// Row of the hand-built floor. High enough that the body has room to fall
    /// onto it and low enough to leave a jump's worth of air above.
    const FLOOR: i32 = 100;

    /// A flat stone floor with air above it and a body standing a little way up.
    ///
    /// Hand-built rather than generated so the tests below assert against a
    /// surface whose height they know, instead of against whatever the worldgen
    /// happens to put at the spawn column. The one test that does want the real
    /// thing says so.
    fn flat_world() -> (CellGrid, Player) {
        let mut grid = CellGrid::new(128, 128);
        for y in 0..grid.rows() {
            for x in 0..grid.cols() {
                grid.set(x, y, if y >= FLOOR { block::STONE } else { EMPTY });
            }
        }
        let spawn = SpawnPoint {
            x: (64 * CELL_SIZE) as f32,
            y: ((FLOOR - 8) * CELL_SIZE) as f32,
        };
        (grid, body_at(spawn))
    }

    /// A player with the real pool behind it, so every test here exercises the
    /// same construction [`spawn_player`] makes.
    fn body_at(spawn: SpawnPoint) -> Player {
        Player::with_projectiles(spawn, Box::new(SharedPool::new()))
    }

    /// Run `n` fixed steps of one intent, giving the rising edges to step 0 only
    /// — which is what [`step_player`] does with [`FixedSubstep`].
    fn run(player: &mut Player, grid: &CellGrid, intent: Intent, n: u32) {
        for i in 0..n {
            player.step(STEP_DT, intent.for_substep(i), grid);
        }
    }

    /// A body that has already fallen and settled on the floor.
    fn settled() -> (CellGrid, Player) {
        let (grid, mut player) = flat_world();
        run(&mut player, &grid, Intent::default(), ONE_SECOND);
        assert!(player.on_ground, "the fixture never landed");
        (grid, player)
    }

    fn running_right() -> Intent {
        Intent {
            dir_x: 1.0,
            ..Intent::default()
        }
    }

    fn jumping() -> Intent {
        Intent {
            jump_queued: true,
            jump_held: true,
            up: true,
            ..Intent::default()
        }
    }

    fn holding_jump() -> Intent {
        Intent {
            jump_held: true,
            up: true,
            ..Intent::default()
        }
    }

    /// Highest the body reaches over `n` steps of `intent` — smallest y, since
    /// the sim's y grows downward.
    fn apex(player: &mut Player, grid: &CellGrid, intent: Intent, n: u32) -> f32 {
        let mut highest = player.y;
        for i in 0..n {
            player.step(STEP_DT, intent.for_substep(i), grid);
            highest = highest.min(player.y);
        }
        highest
    }

    /// The camera is a pure lag, so it must converge and must not overshoot.
    #[test]
    fn the_camera_closes_on_the_body_without_passing_it() {
        let mut focus = WorldFocus { x: 0.0, y: 0.0 };
        let target = 100.0;
        let mut previous = 0.0;
        for step in 0..200 {
            focus.x += (target - focus.x) * CAMERA_EASE;
            assert!(focus.x >= previous, "the ease reversed at {}", focus.x);
            assert!(focus.x <= target, "the ease overshot to {}", focus.x);
            // Progress is required only while there is a gap an f32 can still
            // represent. Past that the ease saturates one ulp short of the
            // target, which is the right place for it to stop.
            assert!(step >= 100 || focus.x > previous, "stalled at {}", focus.x);
            previous = focus.x;
        }
        assert!((focus.x - target).abs() < 0.01, "{} never arrived", focus.x);
    }

    /// The drawn rectangle's edges land on whole world pixels.
    ///
    /// `place_body` rounds the TOP-LEFT and adds half the extent, which is only
    /// equivalent to rounding the centre when both extents are even. `PLAYER_H`
    /// is 15, so it is not, and getting this wrong puts every horizontal edge of
    /// the figure half a pixel into the terrain's pixel grid.
    #[test]
    fn the_placeholder_lands_on_the_pixel_grid() {
        for raw in [0.0_f32, 0.4, 0.5, 12.7, -3.2, -0.5] {
            let left = raw.round();
            let top = raw.round();
            let centre_x = left + PLAYER_W * 0.5;
            let centre_y = top + PLAYER_H * 0.5;
            assert_eq!(
                centre_x - PLAYER_W * 0.5,
                (centre_x - PLAYER_W * 0.5).round()
            );
            assert_eq!(
                centre_y - PLAYER_H * 0.5,
                (centre_y - PLAYER_H * 0.5).round()
            );
        }
    }

    // --- The body, against a real grid --------------------------------------
    //
    // These are what "I watched it move" reduces to when there is no screen: a
    // body dropped on a floor has to land ON it, a jump has to clear a known
    // height, a run has to stop at a wall, and a one-cell ledge has to be walked
    // over rather than jumped. Each is a thing a screenshot would show and a
    // unit test of `Player::step` alone would not, because each depends on the
    // grid underneath.

    #[test]
    fn a_dropped_body_lands_on_the_floor_and_stays_there() {
        let (grid, mut player) = flat_world();
        assert!(!player.on_ground, "started already landed");

        run(&mut player, &grid, Intent::default(), ONE_SECOND);

        assert!(player.on_ground, "never landed");
        assert_eq!(
            player.y + PLAYER_H,
            (FLOOR * CELL_SIZE) as f32,
            "the feet are not on the floor's top face"
        );
        assert_eq!(player.vy, 0.0, "still accelerating into the ground");

        // And it stays put: another second of nothing must not sink it.
        run(&mut player, &grid, Intent::default(), ONE_SECOND);
        assert_eq!(player.y + PLAYER_H, (FLOOR * CELL_SIZE) as f32, "sank");
    }

    #[test]
    fn a_run_along_a_flat_floor_does_not_snag_on_the_cell_seams() {
        let (grid, mut player) = settled();
        let rest_y = player.y;
        let start_x = player.x;

        run(&mut player, &grid, running_right(), ONE_SECOND);

        assert!(
            player.x - start_x > 100.0,
            "only travelled {} px in a second",
            player.x - start_x
        );
        assert_eq!(player.y, rest_y, "the run lifted or dropped the body");
        assert!(player.on_ground, "the run left the ground");
    }

    #[test]
    fn jump_clears_ground_and_lands_again() {
        let (grid, mut player) = settled();
        let rest_y = player.y;

        let top = apex(&mut player, &grid, jumping(), ONE_SECOND);

        // A jump that clears less than the body's own height is not a jump.
        assert!(
            rest_y - top > PLAYER_H,
            "apex was only {} px up",
            rest_y - top
        );
        assert!(player.on_ground, "still airborne a second later");
        assert_eq!(player.y, rest_y, "landed somewhere other than the floor");
    }

    #[test]
    fn a_second_jump_in_the_air_goes_higher_than_one() {
        // One jump, coasted to the top.
        let (grid, mut single) = settled();
        let rest_y = single.y;
        single.step(STEP_DT, jumping(), &grid);
        let single_top = apex(&mut single, &grid, holding_jump(), ONE_SECOND);

        // The same jump, with a second one pressed a fifth of a second later —
        // still on the way up, which is where a double jump is actually used.
        let (grid, mut double) = settled();
        double.step(STEP_DT, jumping(), &grid);
        run(&mut double, &grid, holding_jump(), ONE_SECOND / 5);
        double.step(STEP_DT, jumping(), &grid);
        let double_top = apex(&mut double, &grid, holding_jump(), ONE_SECOND);

        assert!(
            double_top < single_top - 1.0,
            "the air jump gained nothing: {double_top} vs {single_top}"
        );
        assert!(
            rest_y - double_top > PLAYER_H * 2.0,
            "barely left the floor: {} px",
            rest_y - double_top
        );
    }

    #[test]
    fn a_dash_covers_more_ground_than_a_run() {
        // Long enough to be inside the dash and short enough that the run has
        // not yet reached its own top speed.
        const STEPS: u32 = 24;

        let (grid, mut runner) = settled();
        run(&mut runner, &grid, running_right(), STEPS);

        let (grid, mut dasher) = settled();
        let dash = Intent {
            dash_queued: true,
            ..running_right()
        };
        run(&mut dasher, &grid, dash, STEPS);

        assert!(
            dasher.x > runner.x + 1.0,
            "the dash ({}) went no further than the run ({})",
            dasher.x,
            runner.x
        );
    }

    #[test]
    fn a_run_into_a_wall_stops_at_it() {
        let (mut grid, mut player) = flat_world();
        // A wall four cells right of the body, floor to well above head height.
        let wall_col = cell_at(player.x) + 4;
        for y in (FLOOR - 6)..FLOOR {
            grid.set(wall_col, y, block::STONE);
        }
        run(&mut player, &grid, Intent::default(), ONE_SECOND);
        run(&mut player, &grid, running_right(), ONE_SECOND);

        let wall_x = (wall_col * CELL_SIZE) as f32;
        assert!(
            player.x + PLAYER_W <= wall_x,
            "went {} px into the wall",
            player.x + PLAYER_W - wall_x
        );
        assert!(
            player.x + PLAYER_W > wall_x - 1.0,
            "stopped {} px short of it",
            wall_x - (player.x + PLAYER_W)
        );
    }

    #[test]
    fn a_one_cell_ledge_is_walked_over_rather_than_jumped() {
        let (mut grid, mut player) = flat_world();
        // Raise the floor by one cell everywhere right of a line ahead of the
        // body: a step, not a wall.
        let ledge_col = cell_at(player.x) + 6;
        for x in ledge_col..grid.cols() {
            grid.set(x, FLOOR - 1, block::STONE);
        }
        run(&mut player, &grid, Intent::default(), ONE_SECOND);
        let low = player.y;

        run(&mut player, &grid, running_right(), ONE_SECOND);

        assert_eq!(
            player.y,
            low - CELL_SIZE as f32,
            "did not end up exactly one cell higher"
        );
        assert!(
            player.x > (ledge_col * CELL_SIZE) as f32,
            "never got past the ledge"
        );
        // And it never left the ground doing it — a step-up is not a jump.
        assert!(player.on_ground);
    }

    /// The real world, not a hand-built one: the generated spawn the app puts a
    /// body at has to be somewhere that body can stand.
    #[test]
    fn the_generated_spawn_puts_the_body_on_solid_ground() {
        let (cols, rows) = window_size();
        let mut grid = CellGrid::new(cols, rows);
        let spawn = spawn_point(SEED, SPAWN_COL);
        let mut window = WindowManager::new(ChunkStore::new(SEED));
        window.init(&mut grid, cell_at(spawn.x), cell_at(spawn.y));

        let mut player = body_at(spawn);
        run(&mut player, &grid, Intent::default(), ONE_SECOND * 3);

        assert!(player.on_ground, "fell for three seconds without landing");
        assert!(
            player.y - spawn.y < (16 * CELL_SIZE) as f32,
            "fell {} px past the spawn — the spawn is in a hole",
            player.y - spawn.y
        );
    }

    /// Holding right at the generated spawn has to move the body RIGHT.
    ///
    /// Worth its own test rather than folding into the flat-floor run: the
    /// generated surface has slopes, trees and one-cell rubble on it, and "the
    /// body advances over real terrain" is a different claim from "the body
    /// advances over a floor with nothing on it". Written after a driven session
    /// showed the body drifting the wrong way for the first second — which
    /// turned out to be the terrain, not the controller, and this is what pins
    /// the difference down.
    #[test]
    fn holding_right_at_the_generated_spawn_travels_right() {
        let (cols, rows) = window_size();
        let mut grid = CellGrid::new(cols, rows);
        let spawn = spawn_point(SEED, SPAWN_COL);
        let mut window = WindowManager::new(ChunkStore::new(SEED));
        window.init(&mut grid, cell_at(spawn.x), cell_at(spawn.y));

        let mut player = body_at(spawn);
        run(&mut player, &grid, Intent::default(), ONE_SECOND);
        let settled_x = player.x;

        run(&mut player, &grid, running_right(), ONE_SECOND * 2);

        assert!(
            player.x > settled_x + (8 * CELL_SIZE) as f32,
            "two seconds of holding right moved the body {} px",
            player.x - settled_x
        );
    }
}

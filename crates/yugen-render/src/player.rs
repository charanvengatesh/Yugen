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
//! | `while (acc >= STEP_DT) player.step(...)` | [`crate::glue::step_the_body`], in `FixedUpdate` between the two sim sets |
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
//! substep. That masking already exists in [`yugen_core::input`]; this module
//! only passes it the counter [`crate::input::FixedSubstep`] keeps.
//!
//! # The figure is a mesh
//!
//! [`spawn_player`] spawns one `Mesh2d`, not a `Sprite`, and [`place_body`]
//! rebuilds its geometry every frame. That is not gratuitous: the figure's LEAN
//! is a shear, a Bevy `Transform` is TRS and cannot represent one, and a quad
//! with its top edge displaced can — exactly, because a shear is linear. See
//! [`crate::shear`] for the argument in full. The same mesh carries the dash
//! after-images as extra quads, which is what the canvas got from three
//! `drawImage` calls at three `globalAlpha`s.
//!
//! The fallback when the atlas has no player art is still a flat rectangle, and
//! it is still deliberately the collision BOX and not the art rect: the point of
//! drawing it is to see the body collide, so the thing on screen has to be the
//! thing the collider moves. The only presentation state it reads is
//! [`Player::step_up_visual`](yugen_core::entities::Player::step_up_visual),
//! which exists precisely so that cresting a one-cell ledge reads as a stride
//! instead of a teleport — the box snaps up a whole cell instantly and the drawn
//! figure catches up.

use bevy::app::{RunFixedMainLoop, RunFixedMainLoopSystems};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::sprite_render::AlphaMode2d;

use yugen_core::config::{PLAYER_H, PLAYER_W, STEP_DT, cell_at};
use yugen_core::entities::player::PlayerEvent;
use yugen_core::entities::{AnimState, Player, ProjectileSystem};
use yugen_core::physics::collision::Aabb;
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::{CellId, EMPTY, MAT_B, MAT_G, MAT_R, MaterialState};

use crate::effects::Feedback;
use crate::input::FocusDriver;
use crate::lowres::WORLD_LAYERS;
use crate::particles::{EmitOpts, ParticleSystem};
use crate::player_art::{ArtGrid, PlayerFigure, PlayerMotion, seq_state};
use crate::shear::{QuadBuf, ShearedRect, TileUv, dynamic_quad_mesh, origin_translation};
use crate::sprite::{Pose, SpriteAtlas, SpriteAtlases, SpriteClock};
use crate::world::{SimWorld, WorldFocus};

/// The player's sprite code in the compiled table.
use yugen_data::sprites::sprite::PLAYER as SPRITE_PLAYER;

/// How far the view closes on the player each FRAME, 0..1.
///
/// `Camera.follow`'s `ease` argument, verbatim, and frame-rate dependent in
/// exactly the way it was: at 0.12 the view covers 12% of the remaining gap per
/// frame, so it settles in about a third of a second at 60fps and faster on a
/// quicker screen. Keeping the wart is deliberate — this is the number the game
/// feels like, and a per-second reformulation would change the feel of every
/// jump while claiming to be a port.
const CAMERA_EASE: f32 = 0.12;

/// Where the body sits in z, between the cell quad (0) and the brush preview
/// (1) — under the cursor, over the world.
const PLAYER_Z: f32 = 0.5;

/// The placeholder body's colour.
///
/// A warm red, for the one property that matters before there is art: it is not
/// a colour the terrain palette contains, so the figure is unambiguous against
/// sky, snow, stone and lava alike.
const PLACEHOLDER: Color = Color::srgb(0.90, 0.27, 0.30);

/// The tint on the figure itself: untouched art at full opacity.
///
/// The dash after-images are the same white at [`Ghost::alpha`]'s opacity, so
/// every quad in the mesh differs only in its alpha channel — the atlas already
/// carries the colour.
///
/// [`Ghost::alpha`]: crate::player_art::Ghost::alpha
const OPAQUE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Dust over empty space, where there is no cell to take a colour from.
const DEFAULT_DUST: [u8; 3] = [170, 165, 155];

/// A splash whose liquid could not be sampled. See [`liquid_color`].
const DEFAULT_SPLASH: [u8; 3] = [120, 170, 230];

/// Touchdown speed that counts as a full-strength landing.
///
/// The event carries no magnitude, so the strength is `land_impact` over this.
/// It is a raw velocity compared against `Player::land_impact`, which is already
/// body-scaled, so it must not be scaled again here.
const LAND_IMPACT_DIV: f32 = 1400.0;

/// Strength of a landing the body did not consider hard enough to record.
///
/// Not zero: a soft landing still scuffs the ground. `land_impact` is only
/// written past a threshold, so zero there means "soft", not "no reading".
const LAND_IMPACT_SOFT: f32 = 0.12;

/// Puff strength for a footfall — the quietest of the three.
const PUFF_STEP: f32 = 0.16;

/// Puff strength for a jump: more than a step, less than a hard landing.
const PUFF_JUMP: f32 = 0.3;

/// How far above the feet the air-jump ring is centred, in px.
const AIR_JUMP_RISE: f32 = 4.0;

/// Motes in the air-jump ring.
const AIR_JUMP_COUNT: usize = 8;

/// The air jump's ring of displaced air.
///
/// A wide spread and a downward angle, because an air jump has no ground to kick
/// off — what it displaces is the air under the body. Pale and glowing so it
/// reads against terrain as well as sky.
const AIR_JUMP: EmitOpts = EmitOpts {
    color: [225, 235, 255],
    speed: 130.0,
    spread: std::f32::consts::PI * 0.6,
    life: 0.3,
    gravity: 120.0,
    size: 2.0,
    drag: 4.0,
    angle: std::f32::consts::FRAC_PI_2,
    glow: true,
    ..EmitOpts::DEFAULT
};

/// The dash's spark colour — the one hue the player throws that is not sampled.
///
/// A dash displaces no material, so there is nothing to take a colour from; this
/// is the same warm white the player's own art uses for its spark pixel.
const DASH_SPARK: [u8; 3] = [255, 240, 180];

/// Seconds between scrape puffs while sliding down a wall.
///
/// The slide is a continuous state, not an event, so it needs its own cadence or
/// it would emit once per fixed step — 120 puffs a second.
const SCRAPE_INTERVAL: f32 = 0.05;

/// The one system set this module publishes, so the creatures can hang off it.
///
/// [`crate::mobs`] has to run after the body has moved AND after the arrow pool
/// has been stepped, and those are two statements of one system —
/// [`crate::glue::step_the_body`]. Naming the set is what lets that ordering be
/// stated rather than inferred from the two `SimSet` bounds both modules happen
/// to share, and it is why the set outlived the move of the system into
/// `glue.rs`: the members can change without a single `.after()` elsewhere doing
/// so.
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

/// The drawn figure: one mesh of one to three quads. See [`crate::shear`].
///
/// Still named for the sprite it used to be, because it is still the same thing
/// in every way that matters to the rest of the tree — one entity the renderer
/// owns, holding the player's art and nothing the simulation may reach.
#[derive(Component)]
pub struct PlayerSprite;

/// The asset stores the drawn figure is built out of.
///
/// A [`SystemParam`] and not three arguments: [`spawn_player`] already takes six
/// things, and three more would put it past the argument count clippy refuses —
/// which is the right complaint, since "the mesh, its material and the atlas the
/// material samples" is plainly one idea and not three.
#[derive(SystemParam)]
struct FigureAssets<'w> {
    /// The baked sprite table. `None` for the player is the fallback case.
    atlases: Res<'w, SpriteAtlases>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<ColorMaterial>>,
}

/// The arrows the player has in flight.
///
/// A PLAIN resource. It was an `Arc<Mutex<ProjectileSystem>>` because the hit
/// test was a `Box<dyn FnMut + 'static>` stored on the pool and had to capture
/// the creatures; nothing captures anything now, so the pool is simply owned
/// here and Bevy hands it out. See
/// [`ShotWorld`](yugen_core::entities::ShotWorld) for the argument.
///
/// The upshot for the scheduler is the point: `ResMut<ArrowPool>` tells Bevy the
/// truth about who writes this, so the draw systems that only read it can
/// actually run in parallel — where five systems previously took a `Res` and
/// then serialised themselves on a lock the scheduler could not see.
///
/// It is a resource of its own rather than a field on [`PlayerBody`] because
/// [`crate::glue`] steps it beside the body, and `--free-camera` runs with a pool
/// and no body at all.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct ArrowPool(pub ProjectileSystem);

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
    mut art: FigureAssets,
) {
    // A body owns nothing it fires into. `ArrowPool` is its own resource and is
    // lent to the body one step at a time by `crate::glue::step_the_body`.
    let player = Player::new(world.level.spawn);

    let centre = centre_of(player.aabb());
    *focus = WorldFocus {
        x: centre.0,
        y: centre.1,
    };
    *driver = FocusDriver::Player;

    // The baked art if the atlas has it, and the old flat rectangle if it does
    // not. The fallback is not defensive padding: it is the state every
    // milestone before the sprite one shipped in, and a missing sprite should
    // cost you the art and not the body.
    //
    // ONE material either way, and it is never rewritten. White and blending, so
    // the shader returns `vertex colour * sampled texel` — which is the atlas
    // untouched for the figure, the atlas faded for a dash after-image, and the
    // placeholder colour on its own when there is no texture bound at all.
    let material = art.materials.add(ColorMaterial {
        color: Color::WHITE,
        alpha_mode: AlphaMode2d::Blend,
        texture: art.atlases.by_code(SPRITE_PLAYER).map(|a| a.image.clone()),
        ..default()
    });

    // The mesh's origin, not the body's centre: local `(0, 0)` is the drawn art
    // rect's snapped top-left and every vertex is an offset from it. Nothing is
    // on screen until `place_body` rewrites both — the mesh here is the
    // degenerate placeholder — but a transform that meant something else for one
    // frame is a transform somebody reads and believes.
    let start = PlayerFigure::of(&player);

    commands.spawn((
        Mesh2d(art.meshes.add(dynamic_quad_mesh())),
        MeshMaterial2d(material),
        Transform::from_translation(origin_translation(snapped(start.x, start.y), PLAYER_Z)),
        PlayerSprite,
        // The geometry is rewritten in place every frame, so the bounding box
        // Bevy computed when it first saw the handle is stale from the second
        // frame on — and it was computed around a degenerate triangle at the
        // world origin, so culling against it would hide the player everywhere
        // except spawn. Same reason the backdrop meshes carry it.
        NoFrustumCulling,
        WORLD_LAYERS,
    ));

    info!(
        "player at ({:.0}, {:.0}), {PLAYER_W}x{PLAYER_H} px",
        player.x, player.y
    );
    commands.insert_resource(PlayerBody(player));
}

/// Spend a step's events into dust, smear, shake and flash.
///
/// Called by [`crate::glue::step_the_body`], which is the system that owns the
/// fixed step now — the step reaches three modules at once (the pool, the pack,
/// the creatures) and the composition root is where the tree puts a join like
/// that. What stayed here is everything that is only about the BODY: the juice
/// mapping below, and the two `Local`s it remembers between steps.
pub(crate) fn spend_step_events(
    body: &mut Player,
    grid: &CellGrid,
    events: &[PlayerEvent],
    juice: &mut Juice,
    state: &mut JuiceState,
) {
    spawn_player_juice(body, grid, events, juice, state);
}

/// Everything a step's events are spent into.
///
/// Bundled for the reason [`crate::input`]'s `Swinger` is: a Bevy system's
/// parameters are injected rather than passed, so a long list is not the
/// call-site burden `too_many_arguments` exists to catch — but these two are one
/// idea (make the step visible and felt) and saying so beats listing them.
///
/// [`Feedback`] is itself a `SystemParam`, which is what lets this nest.
#[derive(SystemParam)]
pub(crate) struct Juice<'w> {
    particles: ResMut<'w, ParticleSystem>,
    feedback: Feedback<'w>,
}

/// The two things the juice remembers between steps.
///
/// Both are edge detectors rather than state anyone else can observe, which is
/// why they are a `Local` and not a resource: nothing outside this file has any
/// business reading "was the body in liquid last step".
#[derive(Default)]
pub(crate) struct JuiceState {
    /// Whether the body was submerged last step, for the belt-and-braces splash.
    prev_in_liquid: bool,
    /// Counts down between wall-slide scrape puffs, so a slide does not emit one
    /// every step at 120 Hz.
    scrape_timer: f32,
}

/// Turn a step's events into dust, smear, shake and flash.
///
/// `Game.spawnPlayerJuice`, and the mapping is the original's throughout: which
/// event throws which preset, at which strength, in what colour. None of it is
/// invented here — see the `case` arms in `Game.ts`.
///
/// # Why the colours are sampled and not fixed
///
/// A landing kicks up the ground it landed ON, and a splash is the colour of the
/// pool being entered, so both read the cell rather than carrying a palette. Over
/// empty space the sample falls back to a neutral grey ([`DEFAULT_DUST`]) — which
/// is a real case, because a wall-jump samples a cell to the side that may be
/// air by the time the event is drained.
///
/// # Why it runs on the fixed step
///
/// It is called from [`step_player`], with the events of the step that just ran.
/// The TypeScript called it once per FRAME, which meant a frame containing two
/// substeps merged both steps' events and sampled the ground once, at the body's
/// final position. Here each step's events are spent against the position that
/// step ended at, which is where they actually happened.
fn spawn_player_juice(
    body: &mut Player,
    grid: &CellGrid,
    events: &[PlayerEvent],
    juice: &mut Juice,
    state: &mut JuiceState,
) {
    let Juice {
        particles,
        feedback,
    } = juice;
    let JuiceState {
        prev_in_liquid,
        scrape_timer,
    } = state;
    let feet_x = body.x + PLAYER_W / 2.0;
    let feet_y = body.y + PLAYER_H;
    let mut splashed = false;

    for &e in events {
        match e {
            PlayerEvent::Land => {
                // The event carries no magnitude; `land_impact` holds the
                // touchdown speed and the body only records it past a threshold,
                // so a zero means a SOFT landing rather than a missing reading.
                let impact = if body.land_impact > 0.0 {
                    (body.land_impact / LAND_IMPACT_DIV).min(1.0)
                } else {
                    LAND_IMPACT_SOFT
                };
                particles.puff(
                    feet_x,
                    feet_y,
                    ground_color(grid, feet_x, feet_y, 1),
                    impact,
                );
                feedback.land(impact);
                body.land_impact = 0.0;
            }
            PlayerEvent::Step => {
                let c = ground_color(grid, feet_x, feet_y, 1);
                particles.puff(feet_x, feet_y, c, PUFF_STEP);
            }
            PlayerEvent::Jump => {
                let c = ground_color(grid, feet_x, feet_y, 1);
                particles.puff(feet_x, feet_y, c, PUFF_JUMP);
            }
            PlayerEvent::DoubleJump => {
                // An air jump has no ground to kick off, so it is a flat ring of
                // displaced air rather than a tinted puff.
                particles.emit(feet_x, feet_y - AIR_JUMP_RISE, AIR_JUMP_COUNT, &AIR_JUMP);
            }
            PlayerEvent::Dash => {
                particles.smear(feet_x, body.y + PLAYER_H / 2.0, body.facing, DASH_SPARK);
                feedback.dash();
            }
            PlayerEvent::WallJump => {
                // Sampled on the side the body pushed OFF, which is behind it.
                let wall_x = feet_x - body.facing * PLAYER_W * 0.5;
                let mid_y = body.y + PLAYER_H * 0.5;
                let c = ground_color(grid, feet_x - body.facing * PLAYER_W, mid_y, 0);
                particles.scrape(wall_x, mid_y, -body.facing.signum(), c);
            }
            PlayerEvent::Splash => {
                particles.splash(feet_x, feet_y, liquid_color(grid, feet_x, feet_y));
                splashed = true;
            }
            PlayerEvent::Hurt => {
                // Continuous damage — standing in lava — arrives as a
                // rate-limited `Hurt` rather than one per step, so the screen
                // pulses instead of pinning solid red.
                feedback.hurt();
            }
        }
    }

    // --- Continuous states ---------------------------------------------------

    if body.anim() == AnimState::Dash || body.dashing() {
        particles.trail(feet_x, body.y + PLAYER_H / 2.0, DASH_SPARK);
    }

    *scrape_timer -= STEP_DT;
    if body.anim() == AnimState::WallSlide && *scrape_timer <= 0.0 {
        *scrape_timer = SCRAPE_INTERVAL;
        let wall_x = feet_x + body.facing * (PLAYER_W * 0.5 + 2.0);
        let y = body.y + PLAYER_H * 0.6;
        particles.scrape(wall_x, y, body.facing, ground_color(grid, wall_x, y, 0));
    }

    // Liquid entry is reported as an event; this edge is the belt-and-braces
    // path and is suppressed whenever the event did arrive.
    if !splashed && body.in_liquid && !*prev_in_liquid {
        particles.splash(feet_x, feet_y, liquid_color(grid, feet_x, feet_y));
    }
    *prev_in_liquid = body.in_liquid;
}

/// Colour of the cell `dy_cells` below a world point, or a neutral grey.
fn ground_color(grid: &CellGrid, x: f32, y: f32, dy_cells: i32) -> [u8; 3] {
    let cell = WorldCell::new(cell_at(x), cell_at(y) + dy_cells);
    match grid.get_world(cell) {
        EMPTY => DEFAULT_DUST,
        m => mat_rgb(m),
    }
}

/// Colour of the liquid being entered, so a splash matches the pool.
///
/// A non-liquid cell falls back to the default blue rather than tinting the
/// splash with stone: the event means the body entered a liquid, and a sample
/// that disagrees means the surface moved between the step and the drain.
fn liquid_color(grid: &CellGrid, x: f32, y: f32) -> [u8; 3] {
    let cell = WorldCell::new(cell_at(x), cell_at(y));
    let m = grid.get_world(cell);
    if m == EMPTY || MaterialState::of(m) != MaterialState::Liquid {
        return DEFAULT_SPLASH;
    }
    mat_rgb(m)
}

/// A material's authored colour.
fn mat_rgb(m: CellId) -> [u8; 3] {
    let i = m as usize;
    [MAT_R[i], MAT_G[i], MAT_B[i]]
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

/// Rebuild the figure's quads, on the pixel grid the terrain is drawn on.
///
/// The geometry is [`PlayerFigure`]'s, not this function's: the art rect is
/// wider and taller than the collision box (the sprite has padding the collider
/// must not have), it squashes and stretches with real velocity, and it carries
/// the step-up residue that makes cresting a ledge read as a stride rather than
/// a teleport. All of that is a pure function of the body and is tested in
/// [`crate::player_art`] without a GPU; what is left here is the write.
///
/// The TOP-LEFT is what gets rounded, not the centre: an art rect can be even
/// one way and odd the other, so rounding the centre would put edges on half
/// pixels. Same rule [`crate::lowres`] snaps the camera with, applied to a body.
/// It is rounded ONCE, into the mesh's origin, and every quad is an offset from
/// there — so the ghosts cannot land half a pixel off the figure they trail.
///
/// # The lean
///
/// [`PlayerFigure::lean`] is a SHEAR and a Bevy `Transform` is TRS, so a
/// `Sprite` could not carry it and for one milestone it was computed and then
/// discarded. It is carried here instead, by [`quad_of`], as a displacement on
/// the quad's top and bottom edges — which reproduces the canvas's
/// `ctx.transform(1, 0, -lean, 1, 0, 0)` exactly rather than approximately,
/// because a shear is linear and the rasteriser interpolates linearly. See
/// [`crate::shear`] for why that is exact, and for why the rotation about the
/// feet that `Transform` DOES offer was rejected.
///
/// The one thing the snap and the shear have to agree about is the feet. The
/// shear pivots on [`PlayerFigure::pivot_y`], which is the drawn rect's own
/// sole, so the bottom edge's displacement is zero and the feet stay exactly on
/// the whole pixel the snap put them on. Only the top edge moves, and it moves
/// by a fraction of a pixel — rounding THAT would quantise a lean whose whole
/// range is under three pixels down to three positions.
fn place_body(
    body: Res<PlayerBody>,
    atlases: Res<SpriteAtlases>,
    mut meshes: ResMut<Assets<Mesh>>,
    parts: Single<(&mut Transform, &Mesh2d), With<PlayerSprite>>,
    mut buf: Local<QuadBuf>,
) {
    let (mut transform, mesh2d) = parts.into_inner();
    let Some(mut mesh) = meshes.get_mut(&mesh2d.0) else {
        return;
    };

    // `PlayerFigure::of` unrolled, because the fallback below wants the motion
    // the figure was derived from and deriving it twice is how the placeholder
    // and the art end up describing different frames of the same body.
    let motion = PlayerMotion::of(&body);
    let figure = PlayerFigure::new(motion, ArtGrid::player());

    let origin = match atlases.by_code(SPRITE_PLAYER) {
        Some(atlas) => push_figure(&mut buf, &figure, tile_uv(atlas, &figure)),
        None => push_box(&mut buf, &motion),
    };
    transform.translation = origin_translation(origin, PLAYER_Z);
    buf.write(&mut mesh);
}

/// The figure and its dash after-images, and the sim-space point they are
/// relative to.
///
/// Ghosts first and the figure last, because
/// [`PlayerFigure::ghosts`](crate::player_art::PlayerFigure::ghosts) yields them
/// furthest-and-faintest-first and index order IS composite order inside one
/// mesh. Draw them the other way round and the nearest after-image sits on top
/// of the body it is trailing.
///
/// Every quad shares the figure's tile and its deformed extent: an after-image is
/// the same smeared silhouette a moment ago, not a differently shaped one. They
/// share the shear too — the canvas's `save`/`restore` wrapped all three draws.
fn push_figure(buf: &mut QuadBuf, figure: &PlayerFigure, uv: TileUv) -> Vec2 {
    let origin = snapped(figure.x, figure.y);
    buf.begin(origin);
    for ghost in figure.ghosts() {
        buf.push(
            quad_of(figure, ghost.x, ghost.y),
            uv,
            [1.0, 1.0, 1.0, ghost.alpha],
        );
    }
    buf.push(quad_of(figure, figure.x, figure.y), uv, OPAQUE);
    origin
}

/// The flat rectangle for a world with no player art, and its origin.
///
/// The collision BOX, at one quad, unsheared and unsquashed — see the module
/// header. Deforming it would defeat the only thing it is for, which is to show
/// you where the collider actually is; the lean and the squash are art, and a
/// missing sprite is supposed to cost you the art.
///
/// It does carry [`PlayerMotion::step_up_visual`], because that is not art
/// either: it is the residue of a step-up the box resolved instantly, and
/// [`PlayerFigure::new`] applies exactly the same term for exactly this reason.
fn push_box(buf: &mut QuadBuf, motion: &PlayerMotion) -> Vec2 {
    let origin = snapped(motion.x, motion.y + motion.step_up_visual);
    buf.begin(origin);
    buf.push(
        ShearedRect::flat(origin.x, origin.y, PLAYER_W, PLAYER_H),
        TileUv::WHOLE,
        // Linear, because `ColorMaterial` multiplies in linear light and the
        // constant is authored in sRGB.
        LinearRgba::from(PLACEHOLDER).to_f32_array(),
    );
    origin
}

/// One quad of the figure: the rect snapped to the pixel grid, with the lean on
/// its two horizontal edges.
///
/// `x`/`y` are a top-left in sim space — the figure's own, or a ghost's. The
/// displacements come from [`PlayerFigure::shear_at`], so the number on the
/// vertex is the number the art model published and there is no second copy of
/// the shear formula anywhere in the renderer.
///
/// The shear is evaluated at the figure's OWN edges and the snap is then applied
/// to the finished rect, in that order and not the other one. The snap is a
/// whole-pixel translation of a drawn thing; the shear is a property of the body.
/// Shearing the snapped rect instead would put `pivot_y` up to half a pixel away
/// from the drawn sole, so the feet would slide by a fraction of the lean every
/// time the body crossed a pixel boundary — and would hand a ghost a different
/// shear from the figure it trails on any frame the two rounded apart.
fn quad_of(figure: &PlayerFigure, x: f32, y: f32) -> ShearedRect {
    let snap = snapped(x, y);
    ShearedRect {
        x: snap.x,
        y: snap.y,
        w: figure.w,
        h: figure.h,
        top_dx: figure.shear_at(y),
        bottom_dx: figure.shear_at(y + figure.h),
    }
}

/// A sim-space top-left on the world pixel grid. The whole snap rule, once.
#[inline]
fn snapped(x: f32, y: f32) -> Vec2 {
    Vec2::new(x.round(), y.round())
}

/// Which slice of the atlas strip this frame of the figure samples.
///
/// The pose is resolved through the same two hops the `Sprite` path used —
/// [`seq_state`] to cross from the simulation's vocabulary to content's, then
/// [`Pose`] to reach the baked sheet. A pose the sheet does not author falls back
/// to its first sequence: the wrong stance for a frame is a far cheaper failure
/// than an invisible player, and for the shipped sprite it cannot happen at all
/// because every pose is authored.
///
/// The art is authored facing right, so a left-facing body mirrors it — which for
/// a mesh is the two `u` coordinates swapped, exactly the reflection about the
/// rect's own vertical axis that `Sprite::flip_x` performed.
fn tile_uv(atlas: &SpriteAtlas, figure: &PlayerFigure) -> TileUv {
    let state = atlas
        .baked
        .state_id(Pose::from(seq_state(figure.pose)))
        .unwrap_or_else(|| atlas.baked.first_state());
    let clock = SpriteClock {
        state_t: figure.state_t,
        clock_t: figure.clock_t,
        phase: figure.phase,
    };
    TileUv::of(
        atlas.baked.tile(state, 0, &clock),
        atlas.baked.bake_count,
        figure.facing < 0.0,
    )
}

/// Centre of a box, in world px. The snap on spawn and the ease every frame
/// both want it, and deriving it twice is how the two end up half a pixel apart.
#[inline]
fn centre_of(b: Aabb) -> (f32, f32) {
    (b.x + b.w * 0.5, b.y + b.h * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::config::{CELL_SIZE, SEED, cell_at};
    use yugen_core::entities::{Loadout, NoProjectiles};
    use yugen_core::input::Intent;
    use yugen_core::sim::chunk_store::ChunkStore;
    use yugen_core::sim::grid::CellGrid;
    use yugen_core::sim::level::window_size;
    use yugen_core::sim::materials::{EMPTY, block};
    use yugen_core::sim::window::WindowManager;
    use yugen_core::sim::worldgen::{SPAWN_COL, SpawnPoint, spawn_point};

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
        Player::new(spawn)
    }

    /// One fixed step of a body that has nowhere to fire.
    ///
    /// Every test in this module is about locomotion or about the camera, never
    /// about arrows, so the [`Loadout`] deliberately carries the pool that
    /// REFUSES every shot: if one of these ever starts depending on a
    /// projectile, it fails rather than quietly firing into a pool nobody
    /// inspects. The real pool is stepped by [`crate::glue::step_the_body`], one
    /// line after the body, and is tested there.
    fn step(player: &mut Player, intent: Intent, grid: &CellGrid) {
        let mut nowhere = NoProjectiles;
        player.step(STEP_DT, intent, grid, &mut Loadout::new(&mut nowhere));
    }

    /// Run `n` fixed steps of one intent, giving the rising edges to step 0 only
    /// — which is what [`step_player`] does with [`FixedSubstep`].
    fn run(player: &mut Player, grid: &CellGrid, intent: Intent, n: u32) {
        for i in 0..n {
            step(player, intent.for_substep(i), grid);
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
            step(player, intent.for_substep(i), grid);
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

    // --- The drawn figure ---------------------------------------------------
    //
    // `crate::player_art` proves the shear MATHS and `crate::shear` proves the
    // QUAD; what is left for here is the join — that the displacement reaching a
    // vertex is the one `shear_at` published, that the snap survives being
    // routed through a mesh origin, and that the mesh is never empty.

    /// A figure with a given lean and no other deformation.
    fn leaning_figure(accel_x: f32) -> PlayerFigure {
        PlayerFigure::new(
            PlayerMotion {
                x: 100.4,
                y: 200.7,
                accel_x,
                ..PlayerMotion::default()
            },
            ArtGrid::player(),
        )
    }

    /// Local-space vertices of whatever was last pushed into a buffer.
    fn vertices(buf: &mut QuadBuf) -> Vec<Vec2> {
        let mut mesh = dynamic_quad_mesh();
        buf.write(&mut mesh);
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .expect("a written mesh has float3 positions")
            .iter()
            .map(|p| Vec2::new(p[0], p[1]))
            .collect()
    }

    /// Every drawn rect's TOP-LEFT lands on a whole world pixel.
    ///
    /// The rule is round the top-left, not the centre: that is only the same
    /// thing when both extents are even, and `PLAYER_H` is 15. Getting it wrong
    /// puts every horizontal edge of the figure half a pixel into the terrain's
    /// grid, which the low-res target cannot forgive.
    #[test]
    fn the_figure_lands_on_the_pixel_grid() {
        for raw in [0.0_f32, 0.4, 0.5, 12.7, -3.2, -0.5] {
            let figure = PlayerFigure::new(
                PlayerMotion {
                    x: raw,
                    y: raw,
                    ..PlayerMotion::default()
                },
                ArtGrid::player(),
            );
            let quad = quad_of(&figure, figure.x, figure.y);
            assert_eq!(quad.x, quad.x.round(), "the left edge is on a half pixel");
            assert_eq!(quad.y, quad.y.round(), "the top edge is on a half pixel");
        }
    }

    /// The quad's vertices carry the displacement `shear_at` specifies, at the
    /// y each of them actually sits at.
    #[test]
    fn the_quads_vertices_carry_the_shear_the_figure_published() {
        let figure = leaning_figure(4000.0);
        assert!(figure.leaning(), "the fixture is not leaning");

        let quad = quad_of(&figure, figure.x, figure.y);
        assert_eq!(quad.top_dx, figure.shear_at(figure.y));
        assert_eq!(quad.bottom_dx, figure.shear_at(figure.y + figure.h));

        // The head lags the feet, which is what a lean IS.
        assert!(quad.top_dx.abs() > 0.0, "the top edge did not move");
        let corners = quad.corners();
        assert_eq!(corners[0].x, quad.x + quad.top_dx);
        assert_eq!(corners[3].x, quad.x + quad.bottom_dx);
    }

    /// The shear pivots on the drawn sole, so the feet stay exactly on the whole
    /// pixel the snap put them on however hard the body leans.
    #[test]
    fn the_lean_never_moves_the_feet_off_the_pixel_grid() {
        for accel in [-9000.0_f32, -4000.0, 0.0, 4000.0, 9000.0] {
            let figure = leaning_figure(accel);
            let quad = quad_of(&figure, figure.x, figure.y);
            assert_eq!(quad.bottom_dx, 0.0, "the feet slid at accel {accel}");
            let corners = quad.corners();
            assert_eq!(corners[3].x, corners[3].x.round(), "left foot off the grid");
            assert_eq!(corners[2].x, corners[2].x.round(), "right foot off it");
        }
    }

    /// A body with no lean draws the same axis-aligned quad it would have drawn
    /// with no shear in the renderer at all.
    #[test]
    fn a_zero_lean_is_the_unsheared_quad_exactly() {
        let figure = leaning_figure(0.0);
        assert!(!figure.leaning(), "the fixture is leaning");

        let quad = quad_of(&figure, figure.x, figure.y);
        assert_eq!(
            quad,
            ShearedRect::flat(figure.x.round(), figure.y.round(), figure.w, figure.h)
        );
    }

    /// A dash draws its after-images and then itself, and every quad shares the
    /// figure's shear — the canvas's `save`/`restore` wrapped all three draws.
    #[test]
    fn a_dash_draws_its_ghosts_behind_the_figure() {
        let figure = PlayerFigure::new(
            PlayerMotion {
                dashing: true,
                dash_dir: 1.0,
                accel_x: 4000.0,
                on_ground: true,
                ..PlayerMotion::default()
            },
            ArtGrid::player(),
        );
        let ghosts: Vec<_> = figure.ghosts().collect();
        assert!(!ghosts.is_empty(), "a dash with no smear");

        let mut buf = QuadBuf::default();
        let origin = push_figure(&mut buf, &figure, TileUv::WHOLE);
        assert_eq!(origin, Vec2::new(figure.x.round(), figure.y.round()));
        assert_eq!(
            buf.quads(),
            ghosts.len() + 1,
            "a ghost or the body is missing"
        );

        let v = vertices(&mut buf);
        // The figure is the LAST quad, so it composites over its own trail.
        let body = &v[v.len() - 4..];
        assert_eq!(body[0], Vec2::new(figure.shear_at(figure.y), 0.0));

        // And the trail runs backwards along the dash: each quad is further
        // behind than the one before it, faintest first.
        let lefts: Vec<f32> = v.chunks(4).map(|q| q[3].x).collect();
        assert!(
            lefts.windows(2).all(|w| w[0] < w[1]),
            "the trail is not furthest-first: {lefts:?}"
        );
    }

    /// A body that is not dashing draws exactly one quad.
    #[test]
    fn a_still_body_is_one_quad() {
        let mut buf = QuadBuf::default();
        push_figure(&mut buf, &leaning_figure(0.0), TileUv::WHOLE);
        assert_eq!(buf.quads(), 1);
    }

    /// The fallback is the collision BOX, undeformed — see [`push_box`]. It also
    /// carries the step-up residue, because that is the body moving and not the
    /// art.
    #[test]
    fn the_fallback_draws_the_collision_box_and_nothing_else() {
        let motion = PlayerMotion {
            x: 100.4,
            y: 200.7,
            accel_x: 9000.0,
            squash: 1.0,
            step_up_visual: 3.0,
            ..PlayerMotion::default()
        };
        let mut buf = QuadBuf::default();
        let origin = push_box(&mut buf, &motion);

        assert_eq!(origin, Vec2::new(100.0, 204.0), "the step-up was dropped");
        assert_eq!(buf.quads(), 1);

        let v = vertices(&mut buf);
        assert_eq!(v[0], Vec2::ZERO, "the box is not at its own origin");
        assert_eq!(v[1], Vec2::new(PLAYER_W, 0.0), "not the box's width");
        assert_eq!(v[3], Vec2::new(0.0, -PLAYER_H), "not the box's height");
        // Unsheared and unsquashed: the point of it is to show where the collider
        // is, and a leaning squashed box is not where the collider is.
        assert_eq!(v[0].x, v[3].x, "the placeholder was sheared");
    }

    /// Whatever happens, the mesh has triangles in it. An empty one makes Bevy's
    /// allocator log a use-after-free every frame — see `QuadBuf::write`.
    #[test]
    fn the_mesh_is_never_empty() {
        let mut buf = QuadBuf::default();
        for figure in [leaning_figure(0.0), leaning_figure(9000.0)] {
            push_figure(&mut buf, &figure, TileUv::WHOLE);
            assert!(vertices(&mut buf).len() >= 3);
        }
        push_box(&mut buf, &PlayerMotion::default());
        assert!(vertices(&mut buf).len() >= 3);

        // Even a buffer nothing was pushed into, which is a bug upstream rather
        // than a state this module can reach — and still must not spam the log.
        buf.begin(Vec2::ZERO);
        assert_eq!(vertices(&mut buf).len(), 3);
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
        step(&mut single, jumping(), &grid);
        let single_top = apex(&mut single, &grid, holding_jump(), ONE_SECOND);

        // The same jump, with a second one pressed a fifth of a second later —
        // still on the way up, which is where a double jump is actually used.
        let (grid, mut double) = settled();
        step(&mut double, jumping(), &grid);
        run(&mut double, &grid, holding_jump(), ONE_SECOND / 5);
        step(&mut double, jumping(), &grid);
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

//! The creatures — their art, their two shot pools, and the additive pass that
//! keeps a luminous one visible inside the darkness that is multiplied over it.
//!
//! This module is the host [`MobSystem`] was written to be driven by. The
//! simulation half of the milestone is in `yugen-core` and knows nothing about
//! Bevy; what is here is the four wires that half deliberately left hanging:
//!
//! | Wire | Where it goes |
//! |---|---|
//! | the fixed step | [`step_creatures`], after [`PlayerSet::Step`] |
//! | the player, as a target | [`PlayerBody`], which is a [`MobTarget`] |
//! | the arrow hit test | [`crate::glue::step_the_body`], forwarding to [`MobSystem::hit_at`] |
//! | the viewport | [`follow_view`], from [`LowResTarget`] |
//!
//! [`MobTarget`]: yugen_core::entities::mobs::MobTarget
//!
//! # The creatures used to be behind a lock. They are not any more.
//!
//! [`Creatures`] was an `Arc<Mutex<MobSystem>>`, because the arrow hit test was a
//! `Box<dyn FnMut + 'static>` installed ONCE on the pool and fired from inside
//! `ProjectileSystem::update`, which was itself called from the last line of
//! `Player::step`, from a system holding `ResMut<PlayerBody>` and nothing else.
//! There was no point in that call stack where Bevy could hand a second borrow
//! down, so the closure had to CAPTURE the creatures — and `'static` forces the
//! capture to be owned, owned forces `Arc`, shared-mutable forces `Mutex`.
//!
//! Every step of that is downstream of one decision: that the pool was stepped
//! from inside `Player::step`. It is stepped from [`crate::glue`] now, one line
//! after the body, so the host holds both borrows and passes the creatures in as
//! a [`ShotWorld`](yugen_core::entities::ShotWorld). Nothing captures.
//!
//! What that bought, beyond deleting three `Arc<Mutex<_>>` types:
//!
//!   - **The scheduler sees the truth.** Five systems in this module took
//!     `Res<Creatures>` and mutated through the lock. Bevy read that as five
//!     shared borrows and ran them concurrently on the multithreaded executor;
//!     the lock then serialised them at runtime, invisibly. They are `Res` and
//!     `ResMut` honestly now, so the four that only read genuinely run at once.
//!   - **A lock-order inversion is gone.** [`place_shots`] took the creature lock
//!     and then the arrow lock; the hit-test path took them the other way round.
//!     Nothing deadlocked only because Bevy happens to run `RunFixedMainLoop` and
//!     `Update` as separate schedules — an accident of the engine's layout, not an
//!     invariant anyone had stated, and not something the single-threaded test
//!     named for it could ever have caught.
//!
//! # The art
//!
//! Every creature draws its own authored sprite, through [`crate::sprite`]. The
//! bestiary is baked ONCE by [`bake_mob_art`], in `PreStartup`, into
//! [`MobAtlases`] — one strip texture per mob code, with the three poses a brain
//! can ask for already resolved to a [`StateId`]. Nothing rasterises per frame;
//! a draw is a tile index, a flip and a translation.
//!
//! Three of the decisions in that bake are the RENDERER's and not content's, and
//! they are the reason mob art does not go through
//! [`bake_sprite_table`](crate::sprite): a creature with no `air` sequence flies
//! with its walk cycle, a creature's idle plays at half the authored rate, and
//! every creature bakes the whole variant tint table whatever its own `variants`
//! says. All three are stated in `yugen-core` as facts about the bestiary
//! ([`POSE_FALLBACK_AIR_IS_MOVE`], [`POSE_RATE_SCALE_IDLE`],
//! [`VARIANT_COUNT`](yugen_core::entities::mobs::VARIANT_COUNT)) and are spelt
//! here as the three fields of a [`FromContentOpts`].
//!
//! The art rect is NOT the body box. A sprite may overhang its own hitbox — the
//! pad is on [`MobDef::art_pad_x_px`] and [`MobDef::art_pad_top_px`], centred
//! horizontally and all on top vertically, so feet stay planted and horns grow
//! upward. See [`art_origin`], which is the only place that subtraction happens.
//!
//! Shots stay flat rectangles, because that is what they are: the original drew
//! one `fillRect` per shot from the projectile table's colour, and there is no
//! art to draw instead.
//!
//! # The glow pass
//!
//! Nine species are self-luminous, and the darkness multiply that
//! [`crate::light`] composites over the world would erase every one of them in
//! exactly the cave where they are the brightest thing on screen. So they draw
//! TWICE: once into the world, and once more additively, ABOVE the composite.
//! That is what `MobSystem.drawGlow` was, and the ordering is the whole point —
//! `Game.ts` ran it in the overlay callback, after `light.render`.
//!
//! The additive half is [`MobGlowMaterial`] on a quad, because a Bevy [`Sprite`]
//! alpha-blends and has no per-sprite blend state. [`crate::sky::AdditiveMaterial`]
//! is the same idea with no texture, and it is what the glowing SHOTS use, since
//! a shot's glow is a flat rectangle and wants no sampler at all.
//!
//! # The pools
//!
//! The sprite entities are POOLED, not spawned and despawned. Both simulation
//! pools are fixed-length and every slot is one entity for the lifetime of the
//! run; a slot that is not live is hidden. That is the same trade the pools
//! themselves make, for the same reason — a creature dying should not touch the
//! ECS's archetypes, and at 32 mobs the whole set fits in a cache line's worth of
//! transforms. The glow quads are a second pool on the same terms, on the same
//! terms [`crate::light`]'s bloom sprites are.

mod art;
mod material;

pub use art::{MobArt, MobAtlases};
pub use material::MobGlowMaterial;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::prelude::*;
use bevy::shader::Shader;
use bevy::sprite_render::{AlphaMode2d, Material2dPlugin};

use yugen_core::config::{CELL_SIZE, STEP_DT, View};
use yugen_core::entities::mobs::{MAX_MOBS, MobDef, MobEvent, MobSystem, TELL_TIME};
use yugen_core::entities::projectiles::{MAX_SHOTS as MAX_ARROWS, style_glow, style_rgb};

use crate::effects::Feedback;
use crate::lowres::{LowResTarget, WORLD_LAYERS};
use crate::particles::ParticleSystem;
use crate::player::{ArrowPool, PlayerBody, PlayerSet};
use crate::sky::{AdditiveMaterial, VertexBuf, dynamic_mesh, linear};
use crate::sprite::world_translation;
use crate::world::{SimSet, SimWorld};

use art::{art_origin, bake_mob_art, clock_of};
use material::{MOB_GLOW_SHADER, MOB_GLOW_WGSL};

/// Where the creatures sit in z: under the player, over the cell quad.
///
/// Under the player deliberately, and not as a tie-break: `Game.ts` drew
/// `mobs.draw` before `player.draw` into the same world layer, so a creature
/// standing on the body was always behind it. The one that must stay legible is
/// the one being steered.
const MOB_Z: f32 = 0.45;

/// Where shots sit — over everything that can be hit by one.
const SHOT_Z: f32 = 0.55;

/// Where a luminous creature's second, additive pass sits.
///
/// ABOVE the whole light composite, which ends at 0.76 with the flat washes, and
/// above the particles' own glow at 0.7 — the order `Game.ts` ran its overlay
/// callback in, particles first and then creatures. Below the UI at 1.0, because
/// a HUD dimmed by the cave behind it is the bug the overlay layer exists to
/// avoid in the first place.
const MOB_GLOW_Z: f32 = 0.80;

/// A glowing shot's additive pass, immediately over the creatures'.
const SHOT_GLOW_Z: f32 = 0.81;

/// Where a burrower's breach tell sits: just UNDER the creatures.
///
/// The tell marks the ground a creature is about to come through, so anything
/// standing on that ground should occlude it. It draws in place of the (still
/// buried, still invisible) sprite, so in practice it never overlaps the
/// creature it belongs to — but a second creature walking over the spot must
/// cover it, and that is what the ordering buys.
const TELL_Z: f32 = 0.44;

/// The tell's alpha at the moment it appears, before the countdown has run.
///
/// `alpha = TELL_ALPHA_BASE + TELL_ALPHA_SWELL * t`, straight from the original,
/// where `t` runs 0 -> 1 as the countdown empties. It starts faint enough to read
/// as a disturbance rather than as a marker and finishes nearly solid, so the
/// warning gets more urgent exactly as the creature gets closer.
const TELL_ALPHA_BASE: f32 = 0.35;
/// How much alpha the tell gains over its countdown. See [`TELL_ALPHA_BASE`].
const TELL_ALPHA_SWELL: f32 = 0.5;

/// Radians per second of the churn that lifts alternate cells of the tell.
///
/// Driven by the creature's OWN clock, so two burrowers surfacing side by side
/// churn out of step. Fast enough that the row boils rather than sliding as one
/// block, which is the whole difference between "the ground is breaking" and "a
/// rectangle is moving".
const TELL_CHURN_RATE: f32 = 26.0;

/// The self-luminance pulse: `BASE + SWING * sin(state_t * RATE + wander)`.
///
/// A slow breath on the glow so an emberling reads as a live fire rather than a
/// lamp. `wander` is per-creature and free-running, which is what stops a nest of
/// them pulsing in unison. Three module constants and not a `config` export: they
/// describe ONE drawing algorithm, and `yugen-core`'s mob header says so
/// explicitly — the tuning only the draw code read stayed with the draw code.
const GLOW_PULSE_BASE: f32 = 0.8;
/// Half the peak-to-peak of the pulse. See [`GLOW_PULSE_BASE`].
const GLOW_PULSE_SWING: f32 = 0.2;
/// Radians per second of the pulse. See [`GLOW_PULSE_BASE`].
const GLOW_PULSE_RATE: f32 = 3.0;

/// How far a glowing shot's additive square overhangs the shot itself, in px.
///
/// One pixel of bloom around a mote, so the glow reads as light coming off the
/// thing rather than as the thing being a brighter square. Straight from the
/// original's `const r = s.rPx + 1`.
const SHOT_GLOW_PAD_PX: f32 = 1.0;

/// The daylight factor, 0..1, that biases which species may spawn.
///
/// Nocturnal creatures thin out to 15% weight in full daylight rather than
/// vanishing, so this changes the mix and never the fact of spawning.
///
/// Written by [`crate::daynight::advance_clock`], which mirrors the world
/// clock's own daylight weight into it once a frame. This module never writes it
/// and never learns what a clock is — which is the seam working: when the
/// lighting milestone landed, nothing here changed.
///
/// A host that runs `MobPlugin` without `DayNightPlugin` gets
/// [`Daylight::default`] forever, which is a real state (permanent noon) and not
/// a broken one.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Daylight(pub f32);

impl Default for Daylight {
    /// Full daylight — the state a world with no clock is permanently in.
    fn default() -> Daylight {
        Daylight(1.0)
    }
}

/// Every creature in the world.
///
/// A PLAIN resource. See the module header for what it used to be and why it
/// stopped needing to be that.
#[derive(Resource, Deref, DerefMut)]
pub struct Creatures(pub MobSystem);

// ---------------------------------------------------------------------------
// The pools
// ---------------------------------------------------------------------------

/// One pooled creature sprite. `slot` indexes [`MobSystem::mobs`].
#[derive(Component, Clone, Copy)]
pub struct MobSprite {
    /// Index into [`MobSystem::mobs`].
    pub slot: usize,
}

/// One pooled creature GLOW quad, parallel to [`MobSprite`].
///
/// A second component and not a child of the sprite: the two live at opposite
/// ends of the frame — one under the light composite, one over it — and a
/// parent-child pair would tie their transforms together for no gain.
#[derive(Component, Clone, Copy)]
pub struct MobGlow {
    /// Index into [`MobSystem::mobs`].
    pub slot: usize,
}

/// One pooled shot rectangle, from either pool.
#[derive(Component, Clone, Copy)]
pub struct ShotSprite {
    /// Which pool the slot indexes.
    pub pool: ShotPool,
    /// Index into that pool's live set.
    pub slot: usize,
}

/// The one mesh every glowing shot in the frame is drawn into.
///
/// One entity and not a pool, because this pass has no per-shot state to hold:
/// it is a list of flat rectangles rebuilt from empty every frame, which is what
/// a [`VertexBuf`] is for.
#[derive(Component, Clone, Copy)]
pub struct ShotGlow;

/// The one mesh every burrower's breach tell is drawn into.
#[derive(Component)]
pub struct BreachTell;

/// Which of the two shot pools a [`ShotSprite`] mirrors.
///
/// They are drawn by one system because they are the same rectangle, and kept
/// distinct because they are stepped by different owners and carry different
/// colour: a creature's shot is tinted per projectile kind, and the player's is
/// one of the pool's baked styles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShotPool {
    /// [`MobSystem::shots`] — thrown rocks, spat acid, ice.
    Creature,
    /// [`ArrowPool`] — what the player's bow put up.
    Player,
}

/// The creatures, their fixed step, the art and the glow.
///
/// Not the arrow hit test: that is a borrow [`crate::glue`] hands the pool once
/// per step, not a thing installed here. See the module header.
pub struct MobsPlugin;

impl Plugin for MobsPlugin {
    fn build(&self, app: &mut App) {
        // Built in `build` rather than in a startup system so the resource
        // exists before any schedule runs — `spawn_pool` reads its shot count in
        // `Startup` to size the sprite pool, and the view it is sized to here is
        // a placeholder that `follow_view` corrects on the first frame.
        let creatures = Creatures(MobSystem::new(View::for_screen(1, 1)));

        // `init_asset` REPLACES the store rather than skipping, so calling it
        // unconditionally after `DefaultPlugins` would drop every image the app
        // had already loaded. The guard is `crate::sprite`'s, for its reasons.
        if !app.world().contains_resource::<Assets<Image>>() {
            app.init_asset::<Image>();
        }
        if !app
            .world()
            .contains_resource::<Assets<TextureAtlasLayout>>()
        {
            app.init_asset::<TextureAtlasLayout>();
        }

        // Inserted rather than embedded: `MOB_GLOW_WGSL` is a string constant in
        // this file, so there is no asset path for `embedded_asset!` to take.
        // `crate::light` does the same with its own.
        app.world_mut()
            .resource_mut::<Assets<Shader>>()
            .insert(&MOB_GLOW_SHADER, Shader::from_wgsl(MOB_GLOW_WGSL, file!()))
            .expect("the mob glow shader handle is a uuid and cannot collide");

        app.add_plugins(Material2dPlugin::<MobGlowMaterial>::default());
        // Registered by whichever plugin is added first — see the same guard in
        // `SkyPlugin` and `WeatherPlugin`. Adding a plugin twice is a panic.
        if !app.is_plugin_added::<Material2dPlugin<AdditiveMaterial>>() {
            app.add_plugins(Material2dPlugin::<AdditiveMaterial>::default());
        }

        app.insert_resource(creatures)
            .init_resource::<Daylight>()
            .init_resource::<MobAtlases>()
            .add_systems(PreStartup, bake_mob_art)
            .add_systems(Startup, spawn_pool)
            .add_systems(
                FixedUpdate,
                step_creatures
                    .after(PlayerSet::Step)
                    .before(SimSet::Simulate)
                    .run_if(crate::scenes::running)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<PlayerBody>),
            )
            .add_systems(
                Update,
                (
                    // `follow_view` WRITES the creature system and the four
                    // placers read it. With everything behind one mutex Bevy saw
                    // five shared borrows, ran them concurrently and let the lock
                    // sort it out at runtime; stated honestly, the write is
                    // ordered first and the four reads genuinely run at once.
                    follow_view,
                    (
                        place_mobs,
                        place_mob_glow,
                        place_breach_tells,
                        place_shots,
                        place_shot_glow,
                    ),
                )
                    .chain(),
            );
    }
}

/// One entity per simulation slot, hidden, once.
///
/// Sized from the pools themselves rather than from a constant restated here, so
/// a pool that grows cannot leave slots undrawn.
///
/// Every glow quad shares one unit-square mesh and owns one material, which is
/// the split [`crate::light`]'s bloom sprites make: the geometry is identical for
/// all of them and the tile, the tint and the texture are not.
fn spawn_pool(
    mut commands: Commands,
    creatures: Res<Creatures>,
    atlases: Res<MobAtlases>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut glow: ResMut<Assets<MobGlowMaterial>>,
    mut additive: ResMut<Assets<AdditiveMaterial>>,
    mut blend: ResMut<Assets<ColorMaterial>>,
) {
    let mob_shots = creatures.shots().len();
    let quad = meshes.add(Rectangle::default());
    // A real texture from the first frame, so the bind group builds rather than
    // being retried every frame until a creature happens to be luminous. Code 0
    // is also what a dead slot's `def` points at.
    let first = atlases.by_code(0).map(|a| a.atlas.image.clone());

    for slot in 0..MAX_MOBS {
        commands.spawn((
            Sprite::default(),
            Transform::from_xyz(0.0, 0.0, MOB_Z),
            MobSprite { slot },
            Visibility::Hidden,
            WORLD_LAYERS,
        ));
        commands.spawn((
            Mesh2d(quad.clone()),
            MeshMaterial2d(glow.add(MobGlowMaterial {
                atlas: first.clone().unwrap_or_default(),
                tint: Vec4::ZERO,
                tile: Vec4::new(0.0, 0.0, 1.0, 1.0),
            })),
            Transform::from_xyz(0.0, 0.0, MOB_GLOW_Z),
            MobGlow { slot },
            Visibility::Hidden,
            WORLD_LAYERS,
        ));
    }

    for (pool, n) in [
        (ShotPool::Creature, mob_shots),
        (ShotPool::Player, MAX_ARROWS),
    ] {
        for slot in 0..n {
            commands.spawn((
                Sprite {
                    custom_size: Some(Vec2::ZERO),
                    ..default()
                },
                Transform::from_xyz(0.0, 0.0, SHOT_Z),
                ShotSprite { pool, slot },
                Visibility::Hidden,
                WORLD_LAYERS,
            ));
        }
    }

    // Source-over, NOT additive: the tell is opaque ground being shoved up, and
    // the original drew it with `globalAlpha` and a `fillStyle`. Additive would
    // make it glow, which would read as the creature rather than as the dirt.
    // `ColorMaterial` multiplies by the vertex colour, so the per-cell alpha still
    // comes from the buffer.
    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(blend.add(ColorMaterial {
            color: Color::WHITE,
            alpha_mode: AlphaMode2d::Blend,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, TELL_Z),
        BreachTell,
        NoFrustumCulling,
        WORLD_LAYERS,
    ));

    commands.spawn((
        Mesh2d(meshes.add(dynamic_mesh())),
        MeshMaterial2d(additive.add(AdditiveMaterial {})),
        Transform::from_xyz(0.0, 0.0, SHOT_GLOW_Z),
        ShotGlow,
        // Rewritten in place every frame, so the bounding box Bevy computed from
        // the first version of the mesh is stale immediately.
        NoFrustumCulling,
        WORLD_LAYERS,
    ));
}

/// One fixed step of every creature, against the body that just moved.
///
/// The ordering is the point. [`PlayerSet::Step`] has already integrated the
/// body AND stepped the arrows in flight — so a creature an arrow reached this
/// frame is already hurt, or already dead and recycled, before it takes a turn.
/// Running the two the other way round would give every creature one free step
/// after the shot that killed it landed.
///
/// The events are drained for the same reason [`crate::player`] drains the
/// player's: they are particle and audio cues, neither exists yet, and a buffer
/// nobody empties is a buffer that is permanently full by the time the first
/// consumer arrives.
pub(crate) fn step_creatures(
    mut creatures: ResMut<Creatures>,
    world: Res<SimWorld>,
    mut body: ResMut<PlayerBody>,
    day: Res<Daylight>,
    mut particles: ResMut<ParticleSystem>,
    mut feedback: Feedback,
    mut drained: Local<Vec<MobEvent>>,
) {
    creatures.update(STEP_DT, &world.level.grid, &mut **body, day.0);

    // Copied out before any of it is spent, so the juice below is not holding a
    // `ResMut` borrow of the creatures while it works. Each event is one hit,
    // death or chill this tick — a handful at most, and the `Local` keeps the
    // allocation across frames.
    //
    // `events()` returns the VALID PREFIX. It used to return the whole fixed
    // backing store with a separate count, and this call site took all of it:
    // ~14 phantom `MobHurt` events per frame at (0, 0), each worth 0.08 trauma
    // against a decay of 4/s, which pinned the screen shake at maximum from the
    // moment the game opened. `MobSystem::events` now slices, so the mistake is
    // not merely fixed here — it cannot be written anywhere.
    drained.clear();
    drained.extend_from_slice(creatures.events());
    creatures.clear_events();

    for &e in drained.iter() {
        // Every event carries its own position and the creature's blood tone, so
        // neither of these has to learn what a creature is.
        particles.mob_event(e);
        feedback.mob_event(e.kind);
    }
}

/// Keep the spawn and despawn rectangles on the current viewport.
///
/// In the TypeScript these were module constants frozen from `window.innerWidth`
/// at import time, so this had nowhere to live. A native window can be resized
/// and [`LowResTarget::view`] changes when it is, which is the whole reason
/// `SpawnRects` is a function of a `View` here — see
/// [`SpawnRects::for_view`](yugen_core::entities::mobs::SpawnRects::for_view).
fn follow_view(
    mut creatures: ResMut<Creatures>,
    target: Res<LowResTarget>,
    mut last: Local<Option<View>>,
) {
    if *last == Some(target.view) {
        return;
    }
    *last = Some(target.view);
    creatures.set_view(target.view);
}

/// Put every creature's sprite on its art rect, or hide its slot.
///
/// Buried creatures are hidden rather than drawn dark: a burrower inside rock is
/// not hittable by a swing or a shot, and drawing something the player cannot
/// touch is the more misleading of the two. (The original drew the breach TELL in
/// its place while `tell_t` is running — the countdown and its duration are
/// published on `Mob::tell_t` and `TELL_TIME`, and nothing here reads them yet.)
///
/// The whole [`Sprite`] is rewritten rather than patched field by field. Change
/// detection fires either way — a `Mut` deref is what marks it — so the version
/// that cannot forget a field is the better one.
fn place_mobs(
    creatures: Res<Creatures>,
    atlases: Res<MobAtlases>,
    mut sprites: Query<(&MobSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
) {
    let pool = creatures.mobs();

    for (which, mut sprite, mut transform, mut visibility) in &mut sprites {
        let m = &pool[which.slot];
        let art = atlases.by_code(m.def.code);
        let (Some(art), true) = (art, m.active && !m.buried) else {
            *visibility = Visibility::Hidden;
            continue;
        };
        *visibility = Visibility::Inherited;

        // The variant picks the TILE, not a colour: the tint is multiplied into
        // the palette at bake time, so two of a species standing side by side are
        // two different bitmaps and cost nothing at draw. See `VARIANT_TINT`.
        let tile = art.atlas.baked.tile(
            art.state(m.pose),
            m.variant.max(0) as usize,
            &clock_of(&m.clock),
        );
        // `facing` is exactly 1.0 or -1.0, and a mirror about the rect's own
        // vertical axis is what turns the figure without moving it.
        *sprite = art.atlas.sprite_at(tile, m.facing < 0.0);
        sprite.color = flashed(m.flash);

        let (left, top) = art_origin(m.def, m.body.x, m.body.y);
        transform.translation = world_translation(left, top, m.def.art_w_px, m.def.art_h_px, MOB_Z);
    }
}

/// Add a luminous creature back over the darkness that was multiplied over it.
///
/// The same tile, the same rect, the same flip as [`place_mobs`] put in the world
/// layer — this pass differs only in WHEN it runs and HOW it composites. Anything
/// else here would show up as a fringe, which is why the geometry goes through
/// the same two helpers rather than being recomputed.
fn place_mob_glow(
    creatures: Res<Creatures>,
    atlases: Res<MobAtlases>,
    mut materials: ResMut<Assets<MobGlowMaterial>>,
    mut quads: Query<(
        &MobGlow,
        &MeshMaterial2d<MobGlowMaterial>,
        &mut Transform,
        &mut Visibility,
    )>,
) {
    let pool = creatures.mobs();

    for (which, material, mut transform, mut visibility) in &mut quads {
        let m = &pool[which.slot];
        let art = atlases.by_code(m.def.code);
        let lit = m.active && !m.buried && m.def.glow > 0.0;
        let (Some(art), true, Some(mut material)) = (art, lit, materials.get_mut(&material.0))
        else {
            *visibility = Visibility::Hidden;
            continue;
        };
        *visibility = Visibility::Inherited;

        let tile = art.atlas.baked.tile(
            art.state(m.pose),
            m.variant.max(0) as usize,
            &clock_of(&m.clock),
        );
        material.atlas = art.atlas.image.clone();
        material.tile = art.atlas.baked.tile_uv(tile, m.facing < 0.0);
        material.tint = Vec4::new(1.0, 1.0, 1.0, m.def.glow * pulse(m.clock.state_t, m.wander));

        let (left, top) = art_origin(m.def, m.body.x, m.body.y);
        transform.translation =
            world_translation(left, top, m.def.art_w_px, m.def.art_h_px, MOB_GLOW_Z);
        // The mesh is a unit square, so the art rect IS the scale. One mesh for
        // all 32 slots, and creatures are not all the same size.
        transform.scale = Vec3::new(m.def.art_w_px, m.def.art_h_px, 1.0);
    }
}

/// Put every shot rectangle on its slot, from whichever pool owns it.
fn place_shots(
    creatures: Res<Creatures>,
    arrows: Res<ArrowPool>,
    mut sprites: Query<(&ShotSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
) {
    // Collected once, not per sprite: the player's pool publishes its live set
    // as an iterator over claimed slots, which is not indexable and whose order
    // is the pool's, not a slot number's.
    let live: Vec<_> = arrows.shots().collect();
    let mob_shots = creatures.shots();

    for (which, mut sprite, mut transform, mut visibility) in &mut sprites {
        let drawn = match which.pool {
            ShotPool::Creature => {
                let s = &mob_shots[which.slot];
                s.active.then(|| {
                    (
                        s.x,
                        s.y,
                        s.r_px,
                        Color::srgb_u8(s.rgb[0], s.rgb[1], s.rgb[2]),
                    )
                })
            }
            ShotPool::Player => live.get(which.slot).map(|s| {
                let rgb = style_rgb(s.style);
                (s.x, s.y, s.r_px, Color::srgb_u8(rgb[0], rgb[1], rgb[2]))
            }),
        };

        let Some((x, y, r_px, color)) = drawn else {
            *visibility = Visibility::Hidden;
            continue;
        };
        *visibility = Visibility::Inherited;
        sprite.color = color;
        sprite.custom_size = Some(Vec2::splat(r_px * 2.0));
        // A shot is a POINT with a half-extent, not a box with a corner, so it
        // is the centre that rounds here and not the top-left.
        transform.translation.x = x.round();
        transform.translation.y = -y.round();
    }
}

/// Draw the breach tell for every burrower about to surface.
///
/// # What this is, and why buried creatures were invisible without it
///
/// A burrower spends most of its approach underground, where [`place_mobs`]
/// hides it — drawing something the player cannot hit would be the more
/// misleading of the two options. But it then erupts underneath you with no
/// warning at all, which is not a difficulty choice anyone made. The original
/// always drew this and the port simply never got to it: `Mob::tell_t` and
/// [`TELL_TIME`] have been published, simulated and unused since M6.
///
/// `MobSystem.drawTell`, and the shape is the whole point: a row of ground the
/// width of the BODY, jittering one cell above where the creature is about to
/// come out. It says WHERE and roughly WHEN without showing the creature early.
///
/// # Three details that are load-bearing
///
/// **The body width, not the art width.** The tell marks the ground the creature
/// will break through, and that is the footprint of the thing, not the span of
/// whatever it has drawn above its shoulders. `MobDef::w_px` is the collision
/// box; `art_w_px` would be wrong and would look right most of the time, because
/// every mob in current content has zero art padding.
///
/// **Cell-quantised, deliberately.** Every cell of the row is `CELL_SIZE` square
/// and lands on a whole-pixel boundary. A smooth marker over a chunky world reads
/// as UI — as something the game is telling you — rather than as dirt moving.
///
/// **The lift is per cell and comes off the creature's own clock**, so the row
/// churns instead of sliding as one block, and two burrowers surfacing side by
/// side are out of step with each other.
fn place_breach_tells(
    creatures: Res<Creatures>,
    mut meshes: ResMut<Assets<Mesh>>,
    quad: Query<&Mesh2d, With<BreachTell>>,
    mut buf: Local<VertexBuf>,
) {
    let Ok(quad) = quad.single() else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&quad.0) else {
        return;
    };

    buf.clear();
    for m in creatures.mobs() {
        // Only a BURIED creature tells. One that has already broken the surface
        // is drawn as itself, and a tell under it would be a second announcement
        // of something the player can now see.
        if !m.active || !m.buried || m.tell_t <= 0.0 {
            continue;
        }
        push_breach_tell(
            &mut buf,
            m.body.x,
            m.body.y,
            m.def,
            m.tell_t,
            m.clock.state_t,
        );
    }
    buf.write(&mut mesh);
}

/// One creature's tell, as a row of cell-sized quads. See [`place_breach_tells`].
fn push_breach_tell(
    buf: &mut VertexBuf,
    body_x: f32,
    body_y: f32,
    def: &MobDef,
    tell_t: f32,
    state_t: f32,
) {
    // 0 when the countdown starts, 1 as it runs out — so the tell swells and
    // rises as the creature arrives rather than fading away from it.
    let t = 1.0 - (tell_t / TELL_TIME).clamp(0.0, 1.0);
    let alpha = TELL_ALPHA_BASE + TELL_ALPHA_SWELL * t;
    let colour = linear(
        [
            f32::from(def.blood[0]),
            f32::from(def.blood[1]),
            f32::from(def.blood[2]),
        ],
        alpha,
    );

    let cell = CELL_SIZE as f32;
    // One cell ABOVE the body's top edge: the ground about to break, not the
    // creature under it.
    let top = body_y.round() - cell;
    let left = body_x.round();

    let mut c = 0.0;
    while c < def.w_px {
        // A square wave, not a sine: the cell is either lifted a whole cell or
        // not at all. Half-lifted cells would put the row back on a sub-cell
        // grid, which is the quantisation this is deliberately keeping.
        let raised = (state_t * TELL_CHURN_RATE + c).sin() > 0.0;
        // Truncated, matching the original's `| 0`. `t` is non-negative here, so
        // this floors.
        let lift = if raised { (t * cell).trunc() } else { 0.0 };
        let y = top - lift;
        // The one convention flip: +y is up in Bevy and down in the sim.
        buf.quad(
            [
                Vec2::new(left + c, -y),
                Vec2::new(left + c + cell, -y),
                Vec2::new(left + c + cell, -(y + cell)),
                Vec2::new(left + c, -(y + cell)),
            ],
            [colour; 4],
        );
        c += cell;
    }
}

/// Rebuild the one mesh every glowing shot in the frame is drawn into.
///
/// A shot from a luminous creature drawn only in the world layer is multiplied
/// away by the darkness mask in exactly the cave where it is the one thing on
/// screen the player has to react to. The player's arrows go through the same
/// pass and contribute nothing today — every style's [`style_glow`] is 0 — which
/// is deliberate: the first enchanted bolt should not have to invent a render
/// pass, and the loop that would have to be written then is the loop that is
/// here now.
fn place_shot_glow(
    creatures: Res<Creatures>,
    arrows: Res<ArrowPool>,
    mut meshes: ResMut<Assets<Mesh>>,
    quad: Query<&Mesh2d, With<ShotGlow>>,
    mut buf: Local<VertexBuf>,
) {
    let Ok(quad) = quad.single() else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&quad.0) else {
        return;
    };

    buf.clear();
    for s in creatures.shots().iter().filter(|s| s.active) {
        push_shot_glow(&mut buf, s.x, s.y, s.r_px, s.rgb, s.glow);
    }
    for s in arrows.shots() {
        push_shot_glow(
            &mut buf,
            s.x,
            s.y,
            s.r_px,
            style_rgb(s.style),
            style_glow(s.style),
        );
    }
    buf.write(&mut mesh);
}

/// One glowing shot's additive square, in world space.
///
/// Nothing is pushed for a shot that does not glow, so the mesh a frame with no
/// luminous shots in it hands over is empty — which [`VertexBuf::write`] already
/// knows how to make drawable.
///
/// The rounding matches [`place_shots`]: a shot is a point with a half-extent, so
/// the CENTRE is what lands on the pixel grid and the padded square grows either
/// side of it.
fn push_shot_glow(buf: &mut VertexBuf, x: f32, y: f32, r_px: f32, rgb: [u8; 3], glow: f32) {
    if glow <= 0.0 {
        return;
    }
    let r = r_px + SHOT_GLOW_PAD_PX;
    let (left, top) = ((x.round() - r), (y.round() - r));
    let side = r * 2.0;
    let rgb = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32];
    // The one convention flip: +y is up in Bevy and down in the sim.
    let color = linear(rgb, glow.min(1.0));
    buf.quad(
        [
            Vec2::new(left, -top),
            Vec2::new(left + side, -top),
            Vec2::new(left + side, -(top + side)),
            Vec2::new(left, -(top + side)),
        ],
        [color; 4],
    );
}

/// The self-luminance pulse for a creature. See [`GLOW_PULSE_BASE`].
fn pulse(state_t: f32, wander: f32) -> f32 {
    GLOW_PULSE_BASE + GLOW_PULSE_SWING * (state_t * GLOW_PULSE_RATE + wander).sin()
}

/// A creature's damage blink, as a tint.
///
/// The original blitted the SAME bitmap a second time under `lighter` at
/// `globalAlpha = flash`, which on an opaque texel is `src + src * flash` — that
/// is, `src * (1 + flash)`. A tint is exactly that product, so the blink costs
/// one colour instead of a second draw call per creature.
///
/// Multiplying rather than mixing toward white is the faithful half: the original
/// brightened a creature's OWN colours, so a blue slime flashes bright blue and
/// only saturates to white where it was already near it. That saturation is the
/// render target's — `Bgra8UnormSrgb` clamps on write, which is the same ceiling
/// the canvas composite hit.
///
/// The factor is applied through `Color::srgb`, whose transfer function is very
/// nearly the one the canvas was compositing in. `sRGB -> linear` is close enough
/// to a power law that multiplying the linear value by `to_linear(k)` and
/// multiplying the sRGB value by `k` agree to well under a code point, which is
/// what makes this the same brightening and not merely a similar one.
fn flashed(flash: f32) -> Color {
    let k = 1.0 + flash.clamp(0.0, 1.0);
    Color::srgb(k, k, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::config::CELL_SIZE;
    use yugen_core::entities::HitFn;
    use yugen_core::entities::mobs::MOB_DEFS;
    use yugen_core::entities::projectiles::{SHOT_STYLE_ARROW, ShotSpec, ShotWorld};
    use yugen_core::sim::grid::CellGrid;
    use yugen_core::sim::materials::{EMPTY, block};

    #[test]
    fn the_creatures_are_born_with_no_view_and_take_the_first_one() {
        // The placeholder view the plugin constructs with is deliberately
        // degenerate, so a `follow_view` that never ran would be obvious rather
        // than plausible.
        let tiny = MobSystem::new(View::for_screen(1, 1)).rects();
        let real = MobSystem::new(View::for_screen(1280, 800)).rects();
        assert!(real.spawn_hx > tiny.spawn_hx);
        assert!(real.despawn_hy > tiny.despawn_hy);
    }

    #[test]
    fn a_resize_moves_the_spawn_rectangles() {
        let mut mobs = MobSystem::new(View::for_screen(1280, 800));
        let before = mobs.rects();
        mobs.set_view(View::for_screen(2560, 1440));
        let after = mobs.rects();
        assert!(after.keep_hx > before.keep_hx, "a wider window keeps more");
        assert!(after.despawn_hx > before.despawn_hx);
    }

    #[test]
    fn full_daylight_is_the_default_until_the_lighting_milestone() {
        assert_eq!(Daylight::default().0, 1.0);
    }

    // -- The bake -----------------------------------------------------------

    // -- The draw -----------------------------------------------------------

    /// The hit blink brightens the creature's own colours and stops at doubling.
    #[test]
    fn the_hit_flash_brightens_and_tops_out_at_twice_the_body_colour() {
        assert_eq!(
            flashed(0.0).to_srgba(),
            Srgba::WHITE,
            "no flash is no tint at all"
        );
        let full = flashed(1.0).to_srgba();
        assert_eq!([full.red, full.green, full.blue], [2.0, 2.0, 2.0]);
        // A clock that overshot cannot brighten past the original's `min(1, flash)`.
        assert_eq!(flashed(4.0), flashed(1.0));
        // And it is monotone in between, so the blink reads as a decay.
        assert!(flashed(0.5).to_srgba().red < full.red);
        assert!(flashed(0.5).to_srgba().red > 1.0);
    }

    /// The glow pulse never goes dark and never exceeds the authored strength.
    #[test]
    fn the_glow_pulse_stays_inside_its_band() {
        let mut low = f32::MAX;
        let mut high = f32::MIN;
        for i in 0..2000 {
            let p = pulse(i as f32 * 0.01, 1.7);
            low = low.min(p);
            high = high.max(p);
        }
        assert!(low >= GLOW_PULSE_BASE - GLOW_PULSE_SWING);
        assert!(high <= GLOW_PULSE_BASE + GLOW_PULSE_SWING);
        // The band is a breath, not a blink: it never reaches zero, so a
        // luminous creature is never invisible on a trough.
        assert!(low > 0.5);
        // Two creatures with different wander phases are not in step.
        assert_ne!(pulse(1.0, 0.0), pulse(1.0, 1.0));
    }

    /// A shot that does not glow puts nothing in the additive pass.
    #[test]
    fn only_a_luminous_shot_reaches_the_additive_pass() {
        let mut buf = VertexBuf::default();
        push_shot_glow(&mut buf, 10.0, 20.0, 2.0, [255, 0, 0], 0.0);
        // `write` stands a degenerate triangle in for an empty buffer, so three
        // vertices is what "nothing was pushed" looks like on the other side.
        assert_eq!(vertices_written(&mut buf), 3);

        let mut buf = VertexBuf::default();
        push_shot_glow(&mut buf, 10.0, 20.0, 2.0, [255, 0, 0], 0.4);
        // Read off the POSITION attribute, not `count_vertices()`. This assertion
        // used to take the latter, which is the mesh's SMALLEST attribute — and
        // the `Rectangle` these are built on brings a four-element `NORMAL` that
        // `write` never touches. So it read 4 here whether one quad had been
        // pushed or a hundred, and could not have caught the case it names.
        assert_eq!(vertices_written(&mut buf), 4, "one quad");
    }

    /// The player's arrows are wired into the glow pass and glow nothing yet.
    ///
    /// Both halves matter: the pass has to be there for the first enchanted bolt,
    /// and it has to be silent today or every arrow would smear light across the
    /// overlay.
    #[test]
    fn the_players_arrows_are_in_the_glow_pass_and_contribute_nothing() {
        assert_eq!(style_glow(SHOT_STYLE_ARROW), 0.0);
        let mut buf = VertexBuf::default();
        push_shot_glow(
            &mut buf,
            0.0,
            0.0,
            1.0,
            style_rgb(SHOT_STYLE_ARROW),
            style_glow(SHOT_STYLE_ARROW),
        );
        let mut mesh = Mesh::from(Rectangle::default());
        buf.write(&mut mesh);
        assert_eq!(mesh.count_vertices(), 3, "the degenerate stand-in");
    }

    /// Vertices a [`VertexBuf`] actually wrote, read off the position attribute.
    ///
    /// NOT `Mesh::count_vertices()`, which returns the SMALLEST attribute length
    /// on the mesh. These tests build on `Mesh::from(Rectangle::default())`, which
    /// arrives carrying a four-element `ATTRIBUTE_NORMAL` that `VertexBuf::write`
    /// never replaces — so `count_vertices()` saturates at 4 the moment the buffer
    /// holds one quad, and reports 4 for one quad and for a hundred alike.
    fn vertices_written(buf: &mut VertexBuf) -> usize {
        let mut mesh = Mesh::from(Rectangle::default());
        buf.write(&mut mesh);
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) => p.len(),
            _ => panic!("a written mesh always carries float3 positions"),
        }
    }

    // -- The breach tell ----------------------------------------------------

    /// A burrower's tell is as wide as its BODY, not as wide as its art.
    ///
    /// This is the one detail of `drawTell` that would look correct in every
    /// frame anyone captured today and still be wrong: every mob in current
    /// content authors zero art padding, so `w_px` and `art_w_px` are equal for
    /// all of them. The day a creature grows horns, a tell drawn from the art
    /// rect would mark ground the creature is not going to come through.
    ///
    /// So the test builds a def whose art is deliberately wider than its body and
    /// counts the quads. `MOB_DEFS[0]` supplies every other field, because what is
    /// under test is which WIDTH is read and nothing else.
    #[test]
    fn the_tell_is_the_width_of_the_body_and_not_of_the_art() {
        let mut def = MOB_DEFS[0];
        let cell = CELL_SIZE as f32;
        def.w_px = cell * 3.0;
        def.art_w_px = cell * 7.0;

        let mut buf = VertexBuf::default();
        // Halfway through the countdown, so nothing is degenerate.
        push_breach_tell(&mut buf, 0.0, 0.0, &def, TELL_TIME * 0.5, 0.0);
        assert_eq!(
            vertices_written(&mut buf),
            3 * 4,
            "three body cells wide, four vertices each — an art-width tell would \
             be seven"
        );
    }

    /// The tell swells as the countdown empties, and never goes out of range.
    ///
    /// The alpha is the whole of the "roughly WHEN": it has to be faint at the
    /// start and nearly solid at the end, or the warning carries no urgency. It
    /// also has to stay inside 0..1, because `linear` feeds it straight to a
    /// vertex colour.
    #[test]
    fn the_tell_swells_as_the_creature_arrives() {
        let alpha_at = |tell_t: f32| {
            let t = 1.0 - (tell_t / TELL_TIME).clamp(0.0, 1.0);
            TELL_ALPHA_BASE + TELL_ALPHA_SWELL * t
        };

        let fresh = alpha_at(TELL_TIME);
        let landing = alpha_at(0.0);
        assert!((fresh - TELL_ALPHA_BASE).abs() < 1e-6, "starts at the base");
        assert!(
            (landing - (TELL_ALPHA_BASE + TELL_ALPHA_SWELL)).abs() < 1e-6,
            "ends fully swollen"
        );
        assert!(landing > fresh, "the tell gets louder, not quieter");

        // Clamped on both sides: a countdown longer than TELL_TIME (a mob reset
        // mid-tell) must not push the alpha under the base or over 1.
        for t in [-1.0, 0.0, TELL_TIME * 0.5, TELL_TIME, TELL_TIME * 4.0] {
            let a = alpha_at(t);
            assert!((0.0..=1.0).contains(&a), "alpha {a} out of range at {t}");
        }
    }

    /// The row sits one cell ABOVE the body, and every cell lands on the grid.
    ///
    /// Both halves are the difference between "the ground is breaking" and "a
    /// rectangle is sliding": the tell marks the ground the creature is under,
    /// and it is quantised so it reads as dirt rather than as UI.
    #[test]
    fn the_tell_sits_a_cell_above_the_body_on_whole_pixels() {
        let mut def = MOB_DEFS[0];
        let cell = CELL_SIZE as f32;
        def.w_px = cell * 2.0;

        // A deliberately fractional body position: the original rounds before it
        // draws, and a tell on half-pixels shimmers under the nearest-neighbour
        // upscale.
        let mut buf = VertexBuf::default();
        push_breach_tell(&mut buf, 10.4, 64.7, &def, TELL_TIME, 0.0);
        let mut mesh = Mesh::from(Rectangle::default());
        buf.write(&mut mesh);

        let Some(bevy::mesh::VertexAttributeValues::Float32x3(pos)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("the tell mesh carries positions");
        };
        for p in pos {
            assert_eq!(p[0], p[0].round(), "x {} is not a whole pixel", p[0]);
            assert_eq!(p[1], p[1].round(), "y {} is not a whole pixel", p[1]);
        }

        // At a full countdown nothing is lifted yet, so the top edge is exactly
        // one cell above the rounded body top. Bevy's +y is up, the sim's is
        // down, so the highest world row is the largest local y.
        let top = pos.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        assert_eq!(
            top,
            -(64.7f32.round() - cell),
            "the tell is not one cell above the body"
        );
    }

    // -- The seam -----------------------------------------------------------

    /// The arrow-to-creature seam: the six floats go out, the answer comes back.
    ///
    /// This replaces a test called
    /// `an_arrow_in_flight_takes_the_creature_lock_and_gives_it_back`, whose
    /// whole subject was a `Mutex` on the creatures taken from inside the pool's
    /// own step while the pool's lock was held. Nothing captures anything now, so
    /// there is no lock to leak and no ordering to invert — and it is worth
    /// recording that the old test could never have caught the inversion it was
    /// named for anyway, because it was single-threaded and a deadlock needs two.
    ///
    /// What survives is the property that always mattered and that the old test
    /// explicitly did NOT check: the shot's position, damage, knockback and unit
    /// velocity reach the target's resolver unchanged, a `true` retires the shot,
    /// and a `false` leaves it flying. That is exactly the closure
    /// [`crate::glue::step_the_body`] builds.
    #[test]
    fn an_arrow_asks_the_creatures_and_believes_the_answer() {
        /// Records what the pool asked, and answers whatever it was told to.
        struct Spy {
            answer: bool,
            asked: Vec<(f32, f32, f32, f32)>,
        }

        impl ShotWorld for Spy {
            fn hit(&mut self, x: f32, y: f32, damage: f32, kb: f32, dx: f32, dy: f32) -> bool {
                // The direction is handed over as a UNIT vector, so a target can
                // apply knockback along the flight path. Asserted here rather
                // than trusted, because normalising is the pool's job.
                let len = dx.hypot(dy);
                assert!((len - 1.0).abs() < 1e-4, "direction was not normalised");
                self.asked.push((x, y, damage, kb));
                self.answer
            }
        }

        let grid = grid_with_floor(60);

        // A miss: nothing up there to hit, so the shot keeps flying.
        let mut arrows = ArrowPool::default();
        let mut miss = Spy {
            answer: false,
            asked: Vec::new(),
        };
        assert!(arrows.fire(60.0, 60.0, 1.0, 0.0, arrow(200.0)));
        for _ in 0..10 {
            arrows.update(STEP_DT, &grid, &mut miss);
        }
        assert_eq!(arrows.live_count(), 1, "a miss must not retire the shot");
        assert_eq!(miss.asked.len(), 10, "every step asks exactly once");
        let (_, _, damage, knockback) = miss.asked[0];
        assert_eq!(damage, 7.0, "the spec's damage reached the target");
        assert_eq!(knockback, 100.0, "the spec's knockback reached the target");

        // A hit: the shot dies on the step it connects.
        let mut arrows = ArrowPool::default();
        let mut hit = Spy {
            answer: true,
            asked: Vec::new(),
        };
        assert!(arrows.fire(60.0, 60.0, 1.0, 0.0, arrow(200.0)));
        arrows.update(STEP_DT, &grid, &mut hit);
        assert_eq!(arrows.live_count(), 0, "a hit retires the shot");
        assert_eq!(hit.asked.len(), 1);
    }

    /// An empty creature system answers "nothing hit", which is what lets the
    /// seam be exercised at all before anything has spawned.
    ///
    /// `MobSystem::hit_at` has the [`ShotWorld::hit`] signature exactly, which is
    /// why `crate::glue` can forward to it with a closure that has no judgement
    /// of its own in it. If that signature ever drifts, this stops compiling —
    /// which is the point of writing it down as a call rather than as a comment.
    #[test]
    fn the_creature_resolver_has_the_shape_the_seam_forwards_to() {
        let mut creatures = Creatures(MobSystem::new(View::for_screen(1280, 800)));
        let mut seam = HitFn(|x, y, damage, knockback, dx, dy| {
            creatures.hit_at(x, y, damage, knockback, dx, dy)
        });
        assert!(
            !seam.hit(0.0, 0.0, 7.0, 100.0, 1.0, 0.0),
            "an empty population cannot be hit"
        );
    }

    fn grid_with_floor(floor_row: i32) -> CellGrid {
        let mut g = CellGrid::new(64, 64);
        for y in 0..g.rows() {
            for x in 0..g.cols() {
                g.set(x, y, if y >= floor_row { block::STONE } else { EMPTY });
            }
        }
        g
    }

    fn arrow(speed: f32) -> ShotSpec {
        ShotSpec {
            speed,
            damage: 7.0,
            knockback: 100.0,
            r_px: 1.0,
            style: SHOT_STYLE_ARROW,
        }
    }
}

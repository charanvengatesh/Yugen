//! The creatures, the two shot pools, and the rectangles standing in for all of
//! them.
//!
//! This module is the host [`MobSystem`] was written to be driven by. The
//! simulation half of the milestone is in `godgame-core` and knows nothing about
//! Bevy; what is here is the four wires that half deliberately left hanging:
//!
//! | Wire | Where it goes |
//! |---|---|
//! | the fixed step | [`step_creatures`], after [`PlayerSet::Step`] |
//! | the player, as a target | [`PlayerBody`], which is a [`MobTarget`] |
//! | the arrow hit test | [`install_hit_test`], into [`ArrowPool`] |
//! | the viewport | [`follow_view`], from [`LowResTarget`] |
//!
//! # Why the creatures are behind a lock
//!
//! [`Creatures`] is an `Arc<Mutex<MobSystem>>` and not a plain resource, and the
//! reason is the hit test. A [`ShotHitTest`] is a `Box<dyn FnMut + 'static>`
//! installed ONCE, and it fires from inside `ProjectileSystem::update` — which
//! is itself called from inside `Player::step`, from a system that holds
//! `ResMut<PlayerBody>` and nothing else. There is no point in that call stack
//! where Bevy could hand a borrow of a second resource down, so the closure has
//! to CAPTURE the creatures. The TypeScript closed over the mob system for
//! exactly the same reason; this is that closure, with the ownership written
//! down.
//!
//! The lock is never contended: one schedule, and the two systems that take it
//! run sequentially. The ordering rule is one-way and holds trivially — the
//! arrow pool's step reaches for the creatures, and nothing the creatures own
//! ever reaches for the arrow pool.
//!
//! # The rectangles
//!
//! Every drawn thing here is a flat rectangle, on the same terms
//! [`crate::player`]'s is: the body box for a creature, so that what is on
//! screen is what the collider moves, and a square of `2 * r_px` for a shot.
//! Colour is the creature's own [`MobDef::blood`] — the body tone the particle
//! system was going to use — so the bestiary reads as a bestiary and not as
//! thirty identical boxes, without this module inventing a palette that the art
//! milestone would only have to throw away.
//!
//! The sprite entities are POOLED, not spawned and despawned. Both simulation
//! pools are fixed-length and every slot is one entity for the lifetime of the
//! run; a slot that is not live is hidden. That is the same trade the pools
//! themselves make, for the same reason — a creature dying should not touch the
//! ECS's archetypes, and at 32 mobs the whole set fits in a cache line's worth of
//! transforms.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use godgame_core::config::{STEP_DT, View};
use godgame_core::entities::mobs::{MAX_MOBS, MobEvent, MobSystem};
use godgame_core::entities::projectiles::{MAX_SHOTS as MAX_ARROWS, style_rgb};

use crate::lowres::{LowResTarget, WORLD_LAYERS};
use crate::player::{ArrowPool, PlayerBody, PlayerSet};
use crate::world::{SimSet, SimWorld};

/// Where the creatures sit in z: under the player, over the cell quad.
///
/// Under the player deliberately. A creature standing on the same cells as the
/// body is the moment the placeholder rectangles are hardest to read, and the
/// one that must stay legible is the one being steered.
const MOB_Z: f32 = 0.45;

/// Where shots sit — over everything that can be hit by one.
const SHOT_Z: f32 = 0.55;

/// How bright a creature goes on the frame it is hit, 0..1.
///
/// `Mob::flash` is the countdown the simulation already keeps for this; the
/// TypeScript's draw read it to lerp the sprite toward white. There is no sprite
/// yet, so the same signal lerps the placeholder's colour instead. It is here
/// and not in `src/config` because it describes one drawing algorithm.
const FLASH_TO_WHITE: f32 = 0.75;

/// The daylight factor, 0..1, that biases which species may spawn.
///
/// Nocturnal creatures thin out to 15% weight in full daylight rather than
/// vanishing, so this changes the mix and never the fact of spawning.
///
/// SEAM (M7): the day/night cycle is the lighting milestone's, and until it
/// lands this holds at [`Daylight::default`]. When it arrives it writes this
/// resource and nothing in this module changes.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Daylight(pub f32);

impl Default for Daylight {
    /// Full daylight — the state a world with no clock is permanently in.
    fn default() -> Daylight {
        Daylight(1.0)
    }
}

/// The creature system, behind the handle its own hit test has to capture.
///
/// See the module header for why this is shared rather than a plain resource.
#[derive(Resource, Clone)]
pub struct Creatures(Arc<Mutex<MobSystem>>);

impl Creatures {
    /// Borrow the creatures.
    ///
    /// Panics on a poisoned lock, which is the same contract
    /// [`godgame_core::entities::SharedPool`] has: a poisoned system is a bug,
    /// not a state to recover into.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, MobSystem> {
        self.0.lock().expect("creature system poisoned")
    }
}

/// One pooled placeholder. `slot` indexes the simulation pool it mirrors.
#[derive(Component, Clone, Copy)]
pub struct MobSprite {
    /// Index into [`MobSystem::mobs`].
    pub slot: usize,
}

/// One pooled shot placeholder, from either pool.
#[derive(Component, Clone, Copy)]
pub struct ShotSprite {
    /// Which pool the slot indexes.
    pub pool: ShotPool,
    /// Index into that pool's live set.
    pub slot: usize,
}

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

/// The creatures, their fixed step, the arrow hit test, and the placeholders.
pub struct MobsPlugin;

impl Plugin for MobsPlugin {
    fn build(&self, app: &mut App) {
        // Built here rather than in a startup system because `install_hit_test`
        // has to capture it and the player has to be able to fire into a pool
        // that already resolves against it, both before any world exists.
        let creatures = Creatures(Arc::new(Mutex::new(MobSystem::new(View::for_screen(1, 1)))));

        app.insert_resource(creatures)
            .init_resource::<Daylight>()
            .add_systems(Startup, (install_hit_test, spawn_placeholders))
            .add_systems(
                FixedUpdate,
                step_creatures
                    .after(PlayerSet::Step)
                    .before(SimSet::Simulate)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<PlayerBody>),
            )
            .add_systems(Update, (follow_view, place_mobs, place_shots));
    }
}

/// Teach the player's arrows how to find a creature.
///
/// This is the whole of the mobs-to-projectiles seam. `MobSystem::hit_at` has
/// the [`ShotHitTest`](godgame_core::entities::ShotHitTest) signature exactly —
/// `(x, y, damage, knockback, dir_x, dir_y) -> bool` — because that is what the
/// signature was shaped for, so the closure is a lock and a forward and has no
/// judgement of its own in it. Every rule about who can be hurt, by how much,
/// through what armour, lives on the creature that owns the target.
fn install_hit_test(creatures: Res<Creatures>, arrows: Res<ArrowPool>) {
    let creatures = creatures.clone();
    arrows.lock().set_hit_test(Some(Box::new(
        move |x, y, damage, knockback, dir_x, dir_y| {
            creatures
                .lock()
                .hit_at(x, y, damage, knockback, dir_x, dir_y)
        },
    )));
}

/// One entity per simulation slot, hidden, once.
///
/// Sized from the pools themselves rather than from a constant restated here, so
/// a pool that grows cannot leave slots undrawn.
fn spawn_placeholders(mut commands: Commands, creatures: Res<Creatures>) {
    let mob_shots = creatures.lock().shots().len();

    for slot in 0..MAX_MOBS {
        commands.spawn((
            placeholder(MOB_Z),
            MobSprite { slot },
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
                placeholder(SHOT_Z),
                ShotSprite { pool, slot },
                Visibility::Hidden,
                WORLD_LAYERS,
            ));
        }
    }
}

/// A hidden, zero-sized rectangle waiting for a slot to become live.
fn placeholder(z: f32) -> (Sprite, Transform) {
    (
        Sprite {
            custom_size: Some(Vec2::ZERO),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, z),
    )
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
fn step_creatures(
    creatures: Res<Creatures>,
    world: Res<SimWorld>,
    mut body: ResMut<PlayerBody>,
    day: Res<Daylight>,
    mut drained: Local<Vec<MobEvent>>,
) {
    let mut creatures = creatures.lock();
    creatures.update(STEP_DT, &world.level.grid, &mut **body, day.0);

    drained.clear();
    drained.extend_from_slice(creatures.events());
    creatures.clear_events();
}

/// Keep the spawn and despawn rectangles on the current viewport.
///
/// In the TypeScript these were module constants frozen from `window.innerWidth`
/// at import time, so this had nowhere to live. A native window can be resized
/// and [`LowResTarget::view`] changes when it is, which is the whole reason
/// `SpawnRects` is a function of a `View` here — see
/// [`SpawnRects::for_view`](godgame_core::entities::mobs::SpawnRects::for_view).
fn follow_view(
    creatures: Res<Creatures>,
    target: Res<LowResTarget>,
    mut last: Local<Option<View>>,
) {
    if *last == Some(target.view) {
        return;
    }
    *last = Some(target.view);
    creatures.lock().set_view(target.view);
}

/// Put every creature placeholder on its body box, or hide its slot.
///
/// Buried creatures are hidden rather than drawn dark: a burrower inside rock is
/// not hittable by a swing or a shot, and drawing something the player cannot
/// touch is the more misleading of the two.
fn place_mobs(
    creatures: Res<Creatures>,
    mut sprites: Query<(&MobSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
) {
    let creatures = creatures.lock();
    let pool = creatures.mobs();

    for (slot, mut sprite, mut transform, mut visibility) in &mut sprites {
        let m = &pool[slot.slot];
        if !m.active || m.buried {
            *visibility = Visibility::Hidden;
            continue;
        }
        *visibility = Visibility::Inherited;
        sprite.color = flashed(m.def.blood, m.flash);
        sprite.custom_size = Some(Vec2::new(m.body.w, m.body.h));
        // Top-left rounded, not the centre, so an odd extent does not land the
        // edges on half pixels — the rule `crate::player::place_body` snaps with.
        let left = m.body.x.round();
        let top = m.body.y.round();
        transform.translation.x = left + m.body.w * 0.5;
        // +y is up in Bevy and down in the sim: the one convention flip.
        transform.translation.y = -(top + m.body.h * 0.5);
    }
}

/// Put every shot placeholder on its slot, from whichever pool owns it.
fn place_shots(
    creatures: Res<Creatures>,
    arrows: Res<ArrowPool>,
    mut sprites: Query<(&ShotSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
) {
    let creatures = creatures.lock();
    let arrows = arrows.lock();
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

/// A creature's body tone, lerped toward white by its hit flash.
fn flashed(blood: [u8; 3], flash: f32) -> Color {
    let base = Color::srgb_u8(blood[0], blood[1], blood[2]);
    let t = flash.clamp(0.0, 1.0) * FLASH_TO_WHITE;
    base.mix(&Color::WHITE, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::entities::player::Projectiles;
    use godgame_core::entities::projectiles::{SHOT_STYLE_ARROW, ShotSpec};
    use godgame_core::sim::grid::CellGrid;
    use godgame_core::sim::materials::{EMPTY, block};

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

    /// An arrow in flight takes the creature lock every step and gives it back.
    ///
    /// This is the deadlock test, and it is the half of the seam that only this
    /// crate can get wrong. The nesting is real — the pool's lock is held across
    /// `update`, and the installed closure takes the creatures' lock inside it —
    /// so a second handle to the WRONG one, or an ordering that ever ran the two
    /// the other way round, would hang here rather than fail an assertion.
    ///
    /// What it deliberately does not test is the damage. `hit_at` decides who is
    /// hittable, through what armour, for how much, and it is tested where those
    /// rules live; the closure this module installs forwards six floats and has
    /// no judgement of its own to check.
    #[test]
    fn an_arrow_in_flight_takes_the_creature_lock_and_gives_it_back() {
        let creatures = Creatures(Arc::new(Mutex::new(MobSystem::new(View::for_screen(
            1280, 800,
        )))));
        let arrows = ArrowPool::default();
        install_seam(&creatures, &arrows);

        let grid = grid_with_floor(60);
        arrows.lock().fire(60.0, 60.0, 1.0, 0.0, arrow(200.0));
        assert_eq!(arrows.lock().live_count(), 1);

        // Ten steps of open air. No creature is up, so every one of them runs
        // the closure, locks the creatures, finds nothing and returns false.
        for _ in 0..10 {
            arrows.lock().update(STEP_DT, &grid);
        }
        assert_eq!(
            arrows.lock().live_count(),
            1,
            "the shot hit nothing, so it should still be up"
        );

        // And the creatures are borrowable afterwards — the lock was released,
        // not leaked into the closure.
        assert_eq!(creatures.lock().count(), 0);
    }

    /// `install_hit_test` without the `Res` wrappers, so the test can call it.
    ///
    /// Duplicated rather than refactored into one function the system also
    /// calls: what is being tested is the closure the PLUGIN installs, and a
    /// shared helper would make the two identical by construction instead of by
    /// inspection. The three lines are the seam.
    fn install_seam(creatures: &Creatures, arrows: &ArrowPool) {
        let creatures = creatures.clone();
        arrows
            .lock()
            .set_hit_test(Some(Box::new(move |x, y, damage, knockback, dx, dy| {
                creatures.lock().hit_at(x, y, damage, knockback, dx, dy)
            })));
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

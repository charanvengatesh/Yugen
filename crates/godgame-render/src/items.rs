//! The pack, the stacks on the floor, and the four wires between them.
//!
//! `godgame-core`'s [`items`](godgame_core::items) layer is five modules that
//! know nothing about Bevy and nothing about each other beyond one arrow. This
//! module is the host that makes them a game loop:
//!
//! | Wire | From | To |
//! |---|---|---|
//! | mined cells | [`BuildTool`](godgame_core::interact::BuildTool)'s bag | [`GroundItems`] |
//! | killed creatures | [`MobSystem::loot`](godgame_core::entities::mobs::MobSystem::loot) | [`GroundItems`] |
//! | walked-over stacks | [`GroundItems`] | [`Pack`] |
//! | the held slot | [`Pack`] | [`Player::set_weapon`] |
//!
//! Both sources of loot are DRAINED here rather than pushed from there, and that
//! is the shape the two producers were written for: a creature reports a loot
//! row as a string and a position because the creature layer deliberately does
//! not import the item model, and the build tool fills a bag it never empties
//! because "does this yield become world stacks or go straight into a pack" is
//! the host's decision. This module is the only place that has heard of both
//! halves.
//!
//! # Why the pack is behind a lock
//!
//! The same reason [`crate::mobs`]'s creatures are: an
//! [`AmmoSource`](godgame_core::entities::AmmoSource) is a
//! `Box<dyn FnMut + 'static>` installed once on the player and called from deep
//! inside `Player::attack`, where Bevy cannot hand a borrow of a second resource
//! down. So the closure captures the pack. The TypeScript closed over the
//! inventory instance for exactly the same reason.
//!
//! Lock ordering is one-way and shallow: the ammo closure takes the pack and
//! nothing else, and no code the pack can reach ever takes the player.
//!
//! # The rectangles
//!
//! A dropped stack draws as a [`DROP_SIZE_PX`] square in the item's own colour,
//! which [`DropStack`] already caches at spawn. Icons are the sprite milestone's;
//! the colour is not a placeholder palette this module invented, it is the one
//! the item table authors.
//!
//! The count is not drawn. A stack of one and a stack of forty are the same
//! square until there is a font, and inventing a number here would be inventing
//! UI two milestones early.

use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use godgame_core::config::STEP_DT;
use godgame_core::items::{
    DROP_SIZE_PX, DropStack, HeldWeaponSync, Inventory, MAX_DROPS, WorldItems,
};

use crate::input::Tool;
use crate::lowres::WORLD_LAYERS;
use crate::mobs::Creatures;
use crate::player::{PlayerBody, PlayerSet};
use crate::world::{SimSet, SimWorld};

/// Where a dropped stack sits in z: over the terrain, under everything alive.
///
/// Under the creatures deliberately — a drop is scenery until it is walked over,
/// and it must never hide the thing that is trying to bite you.
const DROP_Z: f32 = 0.4;

/// The pack, behind the handle the ammo source has to capture.
///
/// See the module header for why this is shared rather than a plain resource.
#[derive(Resource, Clone, Default)]
pub struct Pack(Arc<Mutex<Inventory>>);

impl Pack {
    /// Borrow the pack.
    ///
    /// Panics on a poisoned lock, the same contract
    /// [`godgame_core::entities::SharedPool`] and [`Creatures`] have.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, Inventory> {
        self.0.lock().expect("inventory poisoned")
    }
}

/// The stacks lying on the floor.
///
/// A plain resource, unlike [`Pack`]: nothing captures it in a closure, because
/// every producer that feeds it is a system that can simply ask for it.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct GroundItems(pub WorldItems);

/// Tracks what the player was last told it was holding.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct HeldWeapon(pub HeldWeaponSync);

/// One pooled drop placeholder. `slot` indexes [`WorldItems`]'s pool.
#[derive(Component, Clone, Copy)]
pub struct DropSprite {
    /// Index into the drop pool. Stable for as long as the stack lives.
    pub slot: usize,
}

/// The pack, the floor, and everything that moves an item between the two.
pub struct ItemsPlugin;

impl Plugin for ItemsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Pack>()
            .init_resource::<GroundItems>()
            .init_resource::<HeldWeapon>()
            .add_systems(Startup, spawn_drop_placeholders)
            .add_systems(
                FixedUpdate,
                (collect_loot, magnetise)
                    .chain()
                    .after(PlayerSet::Step)
                    .before(SimSet::Simulate)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<PlayerBody>),
            )
            .add_systems(
                Update,
                (
                    // `resource_added` fires on the one frame after the body is
                    // inserted, which is the first frame there is a player to
                    // install anything on.
                    install_ammo_source.run_if(resource_added::<PlayerBody>),
                    sync_held_weapon.run_if(resource_exists::<PlayerBody>),
                    collect_dig_drops,
                    place_drops,
                ),
            );
    }
}

/// One entity per drop slot, hidden, once. See [`crate::mobs`] on pooling.
fn spawn_drop_placeholders(mut commands: Commands) {
    for slot in 0..MAX_DROPS {
        commands.spawn((
            Sprite {
                custom_size: Some(Vec2::splat(DROP_SIZE_PX)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, DROP_Z),
            Visibility::Hidden,
            DropSprite { slot },
            WORLD_LAYERS,
        ));
    }
}

/// Teach the player to spend arrows out of the pack.
///
/// [`Inventory::spend_by_id`] has the
/// [`AmmoSource`](godgame_core::entities::AmmoSource) signature exactly —
/// `(&str, u32) -> u32`, returning how many were actually spent — so this
/// closure is a lock and a forward with no judgement in it, the same shape
/// [`crate::mobs::install_hit_test`] has. Which ids count as ammo for a given
/// bow is the weapon's business and is already on `PlayerWeapon::ammo`.
fn install_ammo_source(mut body: ResMut<PlayerBody>, pack: Res<Pack>) {
    let pack = pack.clone();
    body.set_ammo_source(Some(Box::new(move |id, n| pack.lock().spend_by_id(id, n))));
}

/// Push the held slot at the player whenever the selection or the pack changes.
///
/// [`HeldWeaponSync`] is the guard, and it exists so this does not call
/// `set_weapon` sixty times a second with the same value: swapping a weapon
/// mid-swing is a real event with real rules (see [`Player::set_weapon`]), and a
/// host that re-pushed every frame would be firing that event constantly.
fn sync_held_weapon(mut body: ResMut<PlayerBody>, pack: Res<Pack>, mut sync: ResMut<HeldWeapon>) {
    if let Some(weapon) = sync.poll(&pack.lock()) {
        body.set_weapon(weapon);
    }
}

/// Turn this tick's kills into stacks on the floor.
///
/// A loot row names its item as an authoring id that the registry may not have —
/// the creature layer's own contract admits it — so [`WorldItems::spawn_loot`]
/// answers with a bool and a content gap silently drops nothing on the floor
/// rather than taking out the frame.
///
/// The buffer is cleared whether or not every row landed, for the reason
/// [`crate::mobs`] drains events: a report buffer nobody empties is permanently
/// full by the time anyone reads it.
fn collect_loot(creatures: Res<Creatures>, mut ground: ResMut<GroundItems>) {
    let mut creatures = creatures.lock();
    if creatures.loot_count() == 0 {
        return;
    }
    for (i, l) in creatures.loot().iter().enumerate() {
        let n = l.count.max(0.0) as u32;
        if n > 0 {
            // The slot index is the bob phase, so a pile from one kill does not
            // pulse in unison.
            ground.spawn_loot(l.item, n, l.x, l.y, i);
        }
    }
    creatures.clear_loot();
}

/// Empty the build tool's bag onto the floor at the cursor.
///
/// The tool rolls the drop table over the disc BEFORE the edit is emitted, which
/// is what lets a yield exist without the sim having to report what it removed.
/// All this does is decide where it lands: the stroke's centre, so a stack falls
/// out of the hole rather than out of the player.
fn collect_dig_drops(mut tool: ResMut<Tool>, mut ground: ResMut<GroundItems>) {
    if tool.drops().is_empty() {
        return;
    }
    // The centre of the disc the stroke just cut. `preview_bounds` is the same
    // geometry the cursor quad is drawn from, so a stack falls out of the hole
    // the player is looking at and not out of the player.
    let (x, y, w, h) = tool.preview_bounds();
    ground.spawn_bag(tool.drops_mut(), x + w * 0.5, y + h * 0.5);
}

/// Integrate the stacks, and let the player walk them into the pack.
///
/// Runs on the fixed step and after the body has moved, so the magnet pulls
/// toward where the player IS this tick rather than where it was last frame.
fn magnetise(
    world: Res<SimWorld>,
    body: Res<PlayerBody>,
    pack: Res<Pack>,
    mut ground: ResMut<GroundItems>,
) {
    ground.update(STEP_DT, &world.level.grid, body.x, body.y, &mut pack.lock());
}

/// Put every drop placeholder on its stack, or hide its slot.
fn place_drops(
    ground: Res<GroundItems>,
    mut sprites: Query<(&DropSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
) {
    // Indexed by slot, because the query hands entities back in whatever order
    // the archetype is stored in and `stacks()` yields only the live ones.
    let mut by_slot: [Option<DropStack>; MAX_DROPS] = [None; MAX_DROPS];
    for s in ground.stacks() {
        by_slot[s.slot] = Some(s);
    }

    for (which, mut sprite, mut transform, mut visibility) in &mut sprites {
        let Some(s) = by_slot[which.slot] else {
            *visibility = Visibility::Hidden;
            continue;
        };
        *visibility = Visibility::Inherited;
        sprite.color = Color::srgb_u8(s.color[0], s.color[1], s.color[2]);
        // A drop is a centre with no extent — see `WorldItems::update` — so the
        // centre is what rounds, as it does for a shot.
        transform.translation.x = s.x.round();
        // +y is up in Bevy and down in the sim.
        transform.translation.y = -s.y.round();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use godgame_core::items::{item_code_of, registry::ITEM_DEFS};

    #[test]
    fn the_ammo_seam_spends_out_of_the_pack() {
        // The closure `install_ammo_source` installs, without a Bevy app. What is
        // being checked is the forward and the shared handle: a closure that had
        // captured a COPY of the inventory would report the same number and leave
        // the pack full.
        let pack = Pack::default();
        let id = ITEM_DEFS[0].id;
        let code = item_code_of(id).expect("the first item is in the registry");
        pack.lock().add(code, 10);

        let captured = pack.clone();
        let mut ammo: godgame_core::entities::AmmoSource =
            Box::new(move |id, n| captured.lock().spend_by_id(id, n));

        assert_eq!(ammo(id, 3), 3);
        assert_eq!(pack.lock().count_of(code), 7, "the shared pack was spent");
    }

    #[test]
    fn spending_more_than_is_held_takes_only_what_is_there() {
        let pack = Pack::default();
        let id = ITEM_DEFS[0].id;
        let code = item_code_of(id).expect("the first item is in the registry");
        pack.lock().add(code, 2);

        let captured = pack.clone();
        let mut ammo: godgame_core::entities::AmmoSource =
            Box::new(move |id, n| captured.lock().spend_by_id(id, n));

        assert_eq!(ammo(id, 5), 2, "a partial spend reports what it took");
        assert_eq!(pack.lock().count_of(code), 0);
    }

    #[test]
    fn an_unknown_ammo_id_spends_nothing() {
        let pack = Pack::default();
        let captured = pack.clone();
        let mut ammo: godgame_core::entities::AmmoSource =
            Box::new(move |id, n| captured.lock().spend_by_id(id, n));
        assert_eq!(ammo("no_such_item_at_all", 1), 0);
    }

    #[test]
    fn the_held_weapon_is_pushed_once_per_change() {
        // The guard is the whole reason `sync_held_weapon` is cheap to run every
        // frame. An empty pack still pushes ONCE, because "bare hands" is a state
        // the player has to be told about exactly as much as a sword is.
        let pack = Pack::default();
        let mut sync = HeldWeaponSync::new();
        assert!(sync.poll(&pack.lock()).is_some(), "the first poll pushes");
        assert!(
            sync.poll(&pack.lock()).is_none(),
            "an unchanged pack pushes nothing"
        );

        let code = item_code_of(ITEM_DEFS[0].id).expect("the first item is in the registry");
        pack.lock().add(code, 1);
        assert!(
            sync.poll(&pack.lock()).is_some(),
            "a changed pack pushes again"
        );
    }

    #[test]
    fn every_drop_slot_has_a_placeholder() {
        // `place_drops` indexes `by_slot` with `DropStack::slot`, so the pool of
        // sprites has to be at least as long as the pool of stacks. If `MAX_DROPS`
        // ever drifted from the private `CAP` it mirrors, that index would panic
        // on the first drop into a high slot rather than here.
        let mut items = WorldItems::default();
        let code = item_code_of(ITEM_DEFS[0].id).expect("the first item is in the registry");
        for i in 0..MAX_DROPS {
            items.spawn(code, 1, i as f32, 0.0, i);
        }
        for s in items.stacks() {
            assert!(s.slot < MAX_DROPS, "slot {} is outside the pool", s.slot);
        }
    }
}

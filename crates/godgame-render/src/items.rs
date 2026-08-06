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
//! A dropped stack ON THE FLOOR draws as a [`DROP_SIZE_PX`] square in the item's
//! own colour, which [`DropStack`] already caches at spawn — not a placeholder
//! palette this module invented, but the one the item table authors. It stays a
//! square rather than an icon because a drop is 3px in a 640x400 buffer, where a
//! baked icon would be a smudge and its colour is the only thing that reads.
//!
//! The icon and the count both exist, in [`crate::ui`], where the same stack is
//! drawn INSIDE the pack at a size that can carry them: the HUD resolves an icon
//! through `IconAtlas` and sets the count in the bitmap font. That split is the
//! point — the same item is a coloured mote at world scale and a labelled icon at
//! UI scale, and neither drawing belongs in the other's module.

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

/// What the player is carrying.
///
/// A PLAIN resource. It was an `Arc<Mutex<Inventory>>` for one reason: the ammo
/// source was a `Box<dyn FnMut + 'static>` installed on the body, so it had to
/// capture the pack, so the pack had to be owned by the closure as well as by
/// this resource. [`AmmoSource`](godgame_core::entities::AmmoSource) is a trait
/// borrowed for the length of one step now, `Inventory` implements it, and there
/// is nothing left to capture.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct Pack(pub Inventory);

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
                    // AFTER the creatures have taken their step, or this drains
                    // last tick's kills and a drop lands a frame late. The two
                    // used to be unordered with respect to each other and to
                    // serialise on a mutex Bevy could not see, so which ran first
                    // was whatever the executor chose. Stated as `ResMut`, the
                    // ambiguity is real and the scheduler will not resolve it for
                    // us — which is correct, and the edge belongs here.
                    .after(crate::mobs::step_creatures)
                    .after(PlayerSet::Step)
                    .before(SimSet::Simulate)
                    .run_if(resource_exists::<SimWorld>)
                    .run_if(resource_exists::<PlayerBody>),
            )
            .add_systems(
                Update,
                (
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

/// Push the held slot at the player whenever the selection or the pack changes.
///
/// [`HeldWeaponSync`] is the guard, and it exists so this does not call
/// `set_weapon` sixty times a second with the same value: swapping a weapon
/// mid-swing is a real event with real rules (see [`Player::set_weapon`]), and a
/// host that re-pushed every frame would be firing that event constantly.
fn sync_held_weapon(mut body: ResMut<PlayerBody>, pack: Res<Pack>, mut sync: ResMut<HeldWeapon>) {
    if let Some(weapon) = sync.poll(&pack) {
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
fn collect_loot(mut creatures: ResMut<Creatures>, mut ground: ResMut<GroundItems>) {
    if creatures.loot().is_empty() {
        return;
    }
    // `loot()` returns the VALID PREFIX. It used to return the fixed backing
    // store with a separate `loot_count`, and this call site was surviving on
    // luck: a filler row has `count` 0, so the `n > 0` guard below was silently
    // doing the slicing the caller had forgotten. The identical mistake in the
    // event drain pinned the screen shake at maximum for two milestones. The
    // slice is the contract now; the guard is only about empty stacks.
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
    mut pack: ResMut<Pack>,
    mut ground: ResMut<GroundItems>,
) {
    ground.update(STEP_DT, &world.level.grid, body.x, body.y, &mut pack);
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
    use godgame_core::entities::AmmoSource;
    use godgame_core::items::{item_code_of, registry::ITEM_DEFS};

    #[test]
    fn the_ammo_seam_spends_out_of_the_pack() {
        // The seam is now `impl AmmoSource for Inventory` plus the borrow
        // `crate::glue::step_the_body` hands the body, so what is checked here is
        // that a `&mut dyn AmmoSource` pointing at THIS pack spends THIS pack.
        // There is no longer a captured copy that could report the right number
        // and leave the pack full — the borrow makes that unrepresentable — so
        // the test is cheaper and the property it guarded is now structural.
        let mut pack = Pack::default();
        let id = ITEM_DEFS[0].id;
        let code = item_code_of(id).expect("the first item is in the registry");
        pack.add(code, 10);

        let ammo: &mut dyn AmmoSource = &mut *pack;
        assert_eq!(ammo.spend(id, 3), 3);
        assert_eq!(pack.count_of(code), 7, "the real pack was spent");
    }

    #[test]
    fn spending_more_than_is_held_takes_only_what_is_there() {
        let mut pack = Pack::default();
        let id = ITEM_DEFS[0].id;
        let code = item_code_of(id).expect("the first item is in the registry");
        pack.add(code, 2);

        let ammo: &mut dyn AmmoSource = &mut *pack;
        assert_eq!(ammo.spend(id, 5), 2, "a partial spend reports what it took");
        assert_eq!(pack.count_of(code), 0);
    }

    #[test]
    fn an_unknown_ammo_id_spends_nothing() {
        // A content gap costs you the shot, not the run. The registry lookup is
        // inside `Inventory::spend_by_id`, so this is the one place an id that
        // does not resolve is allowed to be silent.
        let mut pack = Pack::default();
        let ammo: &mut dyn AmmoSource = &mut *pack;
        assert_eq!(ammo.spend("no_such_item_at_all", 1), 0);
    }

    #[test]
    fn the_held_weapon_is_pushed_once_per_change() {
        // The guard is the whole reason `sync_held_weapon` is cheap to run every
        // frame. An empty pack still pushes ONCE, because "bare hands" is a state
        // the player has to be told about exactly as much as a sword is.
        let mut pack = Pack::default();
        let mut sync = HeldWeaponSync::new();
        assert!(sync.poll(&pack).is_some(), "the first poll pushes");
        assert!(
            sync.poll(&pack).is_none(),
            "an unchanged pack pushes nothing"
        );

        let code = item_code_of(ITEM_DEFS[0].id).expect("the first item is in the registry");
        pack.add(code, 1);
        assert!(sync.poll(&pack).is_some(), "a changed pack pushes again");
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

//! What you are carrying, what a broken cell yields, and what you can make.
//!
//! The layer has five parts and the arrow between them points one way:
//!
//! | Module | Owns |
//! |---|---|
//! | [`registry`] | the facade over `godgame_data::items`, plus the three indexes the compiler cannot emit |
//! | [`inventory`] | 30 slots, a hotbar, and the revision counter that tells the host when something moved |
//! | [`drops`] | positional-hash drop rolls into a pooled bag |
//! | [`world_items`] | dropped stacks as falling, magnetised entities |
//! | [`crafting`] | spending an inventory against the recipe table |
//!
//! `inventory` reads `registry` for `ITEM_STACK`, so `registry` can never read
//! `inventory` back; `crafting` is where the two meet. That is the same shape
//! the TypeScript had and the reason `crafting.ts` was a separate file at all.
//!
//! # What the port dropped on the way in
//!
//! **All of the drawing.** `WorldItems.draw` was Canvas2D and this crate carries
//! no renderer, so it did not come across; everything it READ is published as an
//! accessor instead ([`world_items::WorldItems::stacks`],
//! [`world_items::DROP_SIZE_PX`]), the way `entities/player.rs` publishes what
//! `Player.draw` used to read. The two per-slot caches that existed only to feed
//! that loop went with it — see the note on [`world_items::WorldItems`].
//!
//! **The baked `Sprite`.** `ITEM_ICONS` in the TypeScript held `Sprite`
//! instances, rasterised into a canvas at module init. A `Sprite` is a renderer
//! object; here [`registry::ITEM_ICONS`] holds the sprite *id*, validated
//! against the sprite table so a dangling reference still degrades to "no icon"
//! exactly as it did, and `godgame-render` does the baking. The dedup-by-id map
//! the TypeScript needed went with the baking: sharing one `Sprite` across the
//! ~24 items that name `block_cube` is a decision for whoever owns the bitmap,
//! and a `&'static str` is already shared.

pub mod crafting;
pub mod drops;
pub mod inventory;
pub mod registry;
pub mod world_items;

pub use crafting::{can_craft, craft, next_craftable};
pub use drops::{DropBag, roll_cell_drops};
pub use inventory::{HOTBAR, HeldWeaponSync, Inventory, SLOT_COUNT};
pub use registry::{
    ITEM_DEFS, ITEM_FOR_BLOCK, ITEM_ICONS, ItemCode, NO_ITEM, Recipe, Station, item_by_code,
    item_by_id, item_code_of, item_color, item_for_block, places_block, recipes,
};
pub use world_items::{DROP_SIZE_PX, DropStack, MAX_DROPS, WorldItems};

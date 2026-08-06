//! The item registry — a facade over `godgame_data::items`, exactly as
//! [`crate::sim::materials`] is one over `godgame_data::blocks`.
//!
//! Everything below is compiled from `content/items/*.toml` by `contentc` (see
//! `content/FORMAT.md`). Nothing is parsed at startup: `ITEM_STACK`,
//! `ITEM_PLACES` and `ITEM_TAGS` are literal arrays baked into the generated
//! module.
//!
//! Hand-written code imports from HERE and never from `godgame_data` directly,
//! so that the derived indexes below sit in the same place as the raw registry
//! and cannot be bypassed. Each exists because the compiler genuinely cannot
//! emit it:
//!
//!   ITEM_FOR_BLOCK  the INVERSE of `places`, which the schema stores in one
//!                   direction only. Emitting it would need a table indexed by
//!                   block code inside the item schema — a different array
//!                   length from every other item table.
//!   RECIPES         the `craft` record[] flattened across every item, with ids
//!                   resolved to codes once so nothing downstream hashes strings.
//!   ITEM_ICONS      `icon` resolved against the sprite table, so a reference
//!                   that does not name a sprite reads as "no icon" rather than
//!                   reaching the renderer.
//!
//! `ITEM_PLACES` used to be built here too, for a reason that no longer holds:
//! the table context has grown a foreign-code lookup, so the item schema can
//! resolve a block id into a block code and the table is baked into the
//! generated module like every other. It is a straight re-export now.
//!
//! # `NO_ITEM` versus `Option`, per site
//!
//! [`NO_ITEM`] survives as the sentinel STORED IN FLAT TABLES —
//! [`ITEM_FOR_BLOCK`] is one `u16` a slot, and `[Option<u16>; MAT_COUNT]` would
//! be two (Rust has no niche to pack the discriminant into: every one of the
//! 65 536 bit patterns of a `u16` is a valid `u16`). That is the same trade
//! [`crate::sim::materials::CellId`] made and for the same reason.
//!
//! Every function that RETURNS one hands back an `Option` instead. The
//! TypeScript proved why: `Inventory.held` returned `-1` for "empty" while the
//! registry's sentinel was `0xffff`, and the one call site that bridged them —
//! `Game.syncHeldWeapon` — compared the wrong one of the two, so an empty
//! selected slot indexed `ITEM_DEFS[-1]` and would have thrown. The two spellings
//! of "nothing" could not be told apart by the type system. Here there is one
//! spelling at every boundary and the mismatch is not expressible.

use crate::entities::player::PlayerWeapon;
use crate::sim::materials::{CellId, MAT_COUNT};
use godgame_data::sprites::{SPRITE_COUNT, SPRITES};
use std::sync::OnceLock;

pub use godgame_data::items::{
    ITEM_COUNT, ITEM_IDS, ITEM_PLACES, ITEM_STACK, ITEM_TAGS, ITEMS as ITEM_DEFS, ItemCategory,
    ItemCraft, ItemCraftStation, ItemDef, ItemEffect, ItemTag, ItemTags, ItemTool, ItemUse,
    ItemWeapon, item,
};

/// An item's code. 0 is a REAL item, unlike block code 0.
///
/// A plain alias for the same reason [`CellId`] is one: it is the value stored
/// in every inventory slot, every pooled drop and every bag counter, and it is
/// indexed into flat tables in loops that run per pickup and per dig tick. A
/// newtype would buy a conversion at each of those and no new invariant — the
/// only invariant worth having here is "this is not [`NO_ITEM`]", and that is
/// what the `Option`-returning accessors below carry.
pub type ItemCode = u16;

/// "No item here." Item code 0 is a REAL item (`stone_chunk`), unlike block code
/// 0 which is air, so anything that needs an out-of-band empty value has to use
/// this rather than 0. Inventory slots use `count == 0` instead and never need
/// it; the reverse index below does.
pub const NO_ITEM: ItemCode = 0xffff;

/// Where a recipe can be made. `Hand` means anywhere.
pub type Station = ItemCraftStation;

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

/// Def for a code. Out-of-range reads item 0 rather than panicking, matching
/// [`crate::sim::materials::mat_by_code`] — a caller holding [`NO_ITEM`] has a
/// bug, and a swatch of the wrong colour reports it better than a crash inside a
/// draw loop.
#[inline]
pub fn item_by_code(code: ItemCode) -> &'static ItemDef {
    ITEM_DEFS.get(code as usize).unwrap_or(&ITEM_DEFS[0])
}

/// Def by stable id, or `None` if the id is misspelled.
///
/// The TypeScript threw here. A `Result`-shaped answer is the same information
/// without the frame-killing failure mode, and the two callers that genuinely
/// cannot continue can still `.expect()` at the point where that is true.
pub fn item_by_id(id: &str) -> Option<&'static ItemDef> {
    ITEM_DEFS.iter().find(|d| d.id == id)
}

/// Code for an item id — the counterpart of `code_of` in [`crate::sim::materials`].
///
/// This is a linear scan and belongs in load-time code only. In a hot path use
/// the generated constants — `item::STONE_CHUNK` is resolved at compile time.
pub fn item_code_of(id: &str) -> Option<ItemCode> {
    item_by_id(id).map(|d| d.code)
}

/// Colour of an item.
#[inline]
pub fn item_color(code: ItemCode) -> [u8; 3] {
    item_by_code(code).color
}

/// Cell id this item places, or 0 (air) when it places nothing.
///
/// `== 0` doubles as the "is this placeable" test, which is why no category
/// table is needed anywhere.
#[inline]
pub fn places_block(code: ItemCode) -> CellId {
    ITEM_PLACES[code as usize]
}

/// Does this item carry every one of these tags?
#[inline]
pub fn has_tags(code: ItemCode, tags: ItemTag) -> bool {
    ItemTag::from_bits_truncate(ITEM_TAGS[code as usize]).contains(tags)
}

// ---------------------------------------------------------------------------
// The reverse index
// ---------------------------------------------------------------------------

/// Block code -> the item that PLACES it, or [`NO_ITEM`].
///
/// This is the drop fallback: `block.drop` is optional, and the schema documents
/// an absent table as "the block drops its own item". Its own item is the one
/// whose `places` points back at it, which is a relation the item schema only
/// stores in the forward direction — hence the inversion here.
///
/// `u16` because block codes are `u16` in the cell grid. First writer wins, so
/// two items claiming one block is a content bug that degrades to "the earlier id
/// is the canonical drop" instead of a silent flip-flop.
///
/// Built by a `const fn` rather than at first use: the TypeScript paid one linear
/// pass at module init because that was the only moment it had. Rust has an
/// earlier one, so the table is in `.rodata` and the game never runs the loop at
/// all.
pub static ITEM_FOR_BLOCK: [ItemCode; MAT_COUNT] = build_item_for_block();

const fn build_item_for_block() -> [ItemCode; MAT_COUNT] {
    let mut map = [NO_ITEM; MAT_COUNT];
    let mut c = 0;
    while c < ITEM_COUNT {
        let block = ITEM_PLACES[c] as usize;
        if block != 0 && map[block] == NO_ITEM {
            map[block] = c as ItemCode;
        }
        c += 1;
    }
    map
}

/// The item a block yields when it has no explicit drop table, if any exists.
///
/// The `Option` is the point: "this block places nothing" (lava, fire, spikes)
/// and "this block drops item 0" are different answers and the sentinel spells
/// them the same way.
#[inline]
pub fn item_for_block(block: CellId) -> Option<ItemCode> {
    match ITEM_FOR_BLOCK.get(block as usize) {
        Some(&c) if c != NO_ITEM => Some(c),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

/// Sprite id per item code, or `None` -> the caller draws the flat `color`.
///
/// WHY A TABLE AND NOT A LOOKUP FUNCTION. The two consumers are a per-frame draw
/// loop over up to 96 world drops and a per-frame draw loop over 10 hotbar slots.
/// Both already hold the item code; an array index is the whole cost.
///
/// WHY `None` IS FIRST CLASS. `icon` is optional in the schema and every consumer
/// has a flat-colour path that predates art and still works. An item with no icon
/// is the NORMAL state, not a failure: icons land one at a time as they are
/// authored, and nothing regresses on the day the first one lands. An `icon` that
/// names a sprite which does not exist degrades to `None` too — the reference is
/// validated by the content compiler, so a miss here means the generated modules
/// are out of step with each other, and a black screen is a far worse way to
/// report that than a swatch that looks like it did last week.
///
/// WHAT THE PORT CHANGED. The TypeScript stored a baked `Sprite` and kept a
/// shared map so that the ~24 items naming `block_cube` pointed at ONE
/// rasterisation instead of 24 identical copies. Baking is a renderer job and
/// this crate has no renderer, so the table stops at the validated id; sharing is
/// automatic, because a `&'static str` is a pointer to the same bytes for every
/// item that names it, and any bitmap cache belongs next to whoever owns bitmaps.
/// A MALFORMED icon — wrong row count, a character outside the palette — is still
/// the sprite runtime's problem and still fails loudly there.
pub static ITEM_ICONS: [Option<&'static str>; ITEM_COUNT] = build_item_icons();

const fn build_item_icons() -> [Option<&'static str>; ITEM_COUNT] {
    let mut out = [None; ITEM_COUNT];
    let mut c = 0;
    while c < ITEM_COUNT {
        if let Some(id) = ITEM_DEFS[c].icon {
            let mut s = 0;
            while s < SPRITE_COUNT {
                if const_str_eq(SPRITES[s].id, id) {
                    out[c] = Some(id);
                    break;
                }
                s += 1;
            }
        }
        c += 1;
    }
    out
}

/// `a == b` for two `&str`, callable from a `const fn`. `str`'s own `PartialEq`
/// is not const, and the alternative is doing this resolution at first use.
const fn const_str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Icon sprite id for an item code, or `None` for the flat swatch.
#[inline]
pub fn item_icon(code: ItemCode) -> Option<&'static str> {
    ITEM_ICONS.get(code as usize).copied().flatten()
}

// ---------------------------------------------------------------------------
// Recipes
// ---------------------------------------------------------------------------

/// One recipe, flattened out of the `craft` record list on its output item.
///
/// Ingredients arrive from the DSL as two parallel lists (a `record[]` element is
/// one line of `key=value` and cannot nest a pair list — see the note in the item
/// schema). They are resolved to codes and counts once, here, so that nothing
/// downstream ever does a string lookup: [`crate::items::can_craft`] runs over
/// integers.
///
/// The two parallel `Vec`s are the TypeScript's two parallel `Uint16Array`s. A
/// `Vec<(ItemCode, u16)>` would have better locality, but the pairing is a
/// property of the DSL and not of this type, and restating it as one array keeps
/// the comment above true of the code below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    /// Item this produces.
    pub out: ItemCode,
    /// How many per craft.
    pub out_count: u16,
    /// Ingredient item codes.
    pub in_items: Vec<ItemCode>,
    /// Counts, parallel to `in_items`. A count the content omitted reads as 1.
    pub in_counts: Vec<u16>,
    pub station: Station,
}

/// Every recipe in the game, in item-code order then declaration order.
///
/// Built once on first use rather than at compile time, which is the one index
/// here that could not follow [`ITEM_FOR_BLOCK`] into `.rodata`: a `Recipe` owns
/// two variable-length lists and there is no const spelling for allocating them.
/// `OnceLock` is the direct analogue of the TypeScript's module-init IIFE — one
/// pass over 79 defs, once, with no lock taken on any read after it.
pub fn recipes() -> &'static [Recipe] {
    static RECIPES: OnceLock<Vec<Recipe>> = OnceLock::new();
    RECIPES.get_or_init(|| {
        let mut out: Vec<Recipe> = Vec::new();
        for def in ITEM_DEFS.iter() {
            let Some(craft) = def.craft else { continue };
            for r in craft {
                let n = r.r#in.len();
                let mut in_items = Vec::with_capacity(n);
                let mut in_counts = Vec::with_capacity(n);
                for (i, id) in r.r#in.iter().enumerate() {
                    // An unresolvable ingredient id degrades to code 0 rather
                    // than dropping the recipe, so a content typo makes ONE
                    // recipe wrong instead of removing it from the game
                    // silently.
                    in_items.push(item_code_of(id).unwrap_or(0));
                    // A short `n` list degrades to 1 rather than desyncing the
                    // two arrays.
                    in_counts.push(r.n.get(i).copied().unwrap_or(1).max(1) as u16);
                }
                out.push(Recipe {
                    out: def.code,
                    out_count: r.out.max(0) as u16,
                    in_items,
                    in_counts,
                    station: r.station,
                });
            }
        }
        out
    })
}

// ---------------------------------------------------------------------------
// The drop-id bridge, and the player adapter
// ---------------------------------------------------------------------------

/// Resolve a DROP-TABLE item id, which may be an `a|b|c` preference chain.
///
/// Mob loot and container loot author ids that the item registry may not have
/// yet, and may offer alternates: the first name that exists wins, and a chain
/// where nothing exists is a content gap rather than a crash. This is the whole
/// id -> code bridge for anything that drops loot, and it lives HERE rather than
/// in the creature layer, which deliberately does not import the item model.
///
/// The TypeScript memoised this in a `Map` because a kill is its hot path. There
/// is no memo here: the scan is 79 `&str` comparisons that stop at the first
/// match, and a global cache would need either a lock in the middle of a kill or
/// interior mutability shared across threads. If a profile ever disagrees, the
/// cache belongs to whoever owns the loot buffer, where it can be a plain field.
pub fn item_code_for_drop(id: &str) -> Option<ItemCode> {
    for part in id.split('|') {
        if let Some(code) = item_code_of(part) {
            return Some(code);
        }
    }
    None
}

/// What the item in a slot contributes to combat, in the shape the player speaks.
///
/// The three lines the player's doc comment asks for. `Player` has no business
/// depending on the item registry and does not; TypeScript got that decoupling
/// for free because the two types were structurally identical, and Rust is
/// nominal, so the conversion is written on THIS side of the boundary. The arrow
/// still points one way — only who writes the adapter moved.
impl From<ItemWeapon> for PlayerWeapon {
    fn from(w: ItemWeapon) -> PlayerWeapon {
        PlayerWeapon {
            damage: w.damage,
            knockback: w.knockback,
            swing_time: w.swing_time,
            // `ranged` carries a schema default, so it is materialised on every
            // item that has a weapon group at all and `unwrap_or(false)` never
            // actually fires. It is spelled out rather than `.unwrap()` because
            // a melee weapon reading as melee is the right answer to "unset".
            ranged: w.ranged.unwrap_or(false),
            projectile_speed: w.projectile_speed,
            ammo: w.ammo.unwrap_or(&[]),
        }
    }
}

/// The weapon stats for an item code, or `None` when it is not a weapon.
#[inline]
pub fn player_weapon(code: ItemCode) -> Option<PlayerWeapon> {
    item_by_code(code).weapon.map(PlayerWeapon::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::materials::BLOCKS;

    #[test]
    fn item_zero_is_a_real_item_and_no_item_is_not_a_code() {
        assert_eq!(ITEM_DEFS[0].code, 0);
        assert_eq!(ITEM_DEFS[0].id, "stone_chunk");
        assert!(ITEM_COUNT < NO_ITEM as usize, "NO_ITEM must be unreachable");
    }

    #[test]
    fn every_table_is_as_long_as_the_registry() {
        assert_eq!(ITEM_STACK.len(), ITEM_COUNT);
        assert_eq!(ITEM_PLACES.len(), ITEM_COUNT);
        assert_eq!(ITEM_TAGS.len(), ITEM_COUNT);
        assert_eq!(ITEM_IDS.len(), ITEM_COUNT);
        assert_eq!(ITEM_ICONS.len(), ITEM_COUNT);
        assert_eq!(ITEM_FOR_BLOCK.len(), MAT_COUNT);
    }

    #[test]
    fn lookups_are_total() {
        assert_eq!(item_by_code(9999).id, ITEM_DEFS[0].id);
        assert!(item_by_id("no_such_item").is_none());
        assert_eq!(item_code_of("no_such_item"), None);
        assert_eq!(item_code_of("stone_chunk"), Some(item::STONE_CHUNK));
    }

    /// The inversion is the whole reason this file builds an index, so both
    /// directions must agree for every placeable item AND every block that any
    /// item places.
    #[test]
    fn item_for_block_round_trips_for_every_placeable_block() {
        let mut seen = 0;
        for c in 0..ITEM_COUNT {
            let block = ITEM_PLACES[c];
            if block == 0 {
                continue;
            }
            seen += 1;
            let back = item_for_block(block).expect("a placed block must map back to an item");
            // First writer wins, so the answer is the LOWEST code that places
            // this block, not necessarily `c`.
            assert_eq!(
                places_block(back),
                block,
                "{} places {} but the reverse index answers {}",
                ITEM_DEFS[c].id,
                block,
                ITEM_DEFS[back as usize].id,
            );
            assert!(back as usize <= c, "first writer must win");
        }
        assert!(seen > 0, "content has no placeable items at all");

        // And nothing claims a block no item places.
        for (block, &owner) in ITEM_FOR_BLOCK.iter().enumerate() {
            if owner == NO_ITEM {
                assert!(
                    !ITEM_PLACES.contains(&(block as u16)) || block == 0,
                    "block {block} is placed by an item but has no reverse entry",
                );
            } else {
                assert_eq!(places_block(owner), block as CellId);
            }
        }
    }

    #[test]
    fn air_has_no_item() {
        // ITEM_PLACES uses 0 for "places nothing", so the inversion must never
        // claim air — otherwise digging empty space would yield stacks.
        assert_eq!(item_for_block(0), None);
        assert_eq!(BLOCKS[0].id, "empty");
    }

    #[test]
    fn icons_that_do_not_name_a_sprite_degrade_to_none() {
        for c in 0..ITEM_COUNT {
            match (ITEM_DEFS[c].icon, ITEM_ICONS[c]) {
                (None, resolved) => assert_eq!(resolved, None),
                (Some(id), Some(resolved)) => {
                    assert_eq!(resolved, id);
                    assert!(SPRITES.iter().any(|s| s.id == id));
                }
                (Some(id), None) => {
                    assert!(
                        !SPRITES.iter().any(|s| s.id == id),
                        "{id} exists but resolved to None",
                    );
                }
            }
        }
    }

    #[test]
    fn recipes_resolve_to_codes_and_never_ask_for_zero_of_anything() {
        let rs = recipes();
        assert!(!rs.is_empty(), "content declares no recipes");
        for r in rs {
            assert_eq!(r.in_items.len(), r.in_counts.len());
            assert!(r.out_count >= 1, "a recipe that yields nothing is a bug");
            for &n in &r.in_counts {
                assert!(n >= 1, "an ingredient count of 0 is free crafting");
            }
            for &c in &r.in_items {
                assert!((c as usize) < ITEM_COUNT);
            }
            // Declaration order: the output item's own defs come out in code
            // order, so the flattened list is sorted by output code.
            assert!((r.out as usize) < ITEM_COUNT);
        }
        assert!(rs.windows(2).all(|w| w[0].out <= w[1].out));
    }

    #[test]
    fn drop_ids_accept_a_preference_chain() {
        assert_eq!(item_code_for_drop("gem"), Some(item::GEM));
        assert_eq!(item_code_for_drop("nope|gem"), Some(item::GEM));
        assert_eq!(item_code_for_drop("nope|also_nope"), None);
        // First name that EXISTS wins, not first name that is listed.
        assert_eq!(item_code_for_drop("coal|gem"), Some(item::COAL));
    }

    #[test]
    fn the_player_adapter_carries_every_field_across() {
        let bow = item_by_code(item::BOW);
        let w = bow.weapon.expect("the bow must be a weapon");
        let pw = PlayerWeapon::from(w);
        assert_eq!(pw.damage, w.damage);
        assert_eq!(pw.knockback, w.knockback);
        assert_eq!(pw.swing_time, w.swing_time);
        assert!(pw.ranged, "the bow is ranged");
        assert_eq!(pw.projectile_speed, w.projectile_speed);
        assert!(!pw.ammo.is_empty(), "a bow with no ammo list fires free");

        let sword = player_weapon(item::SWORD_TRAVELER).expect("a sword is a weapon");
        assert!(!sword.ranged);
        assert!(sword.ammo.is_empty());

        assert_eq!(player_weapon(item::STONE_CHUNK), None);
    }
}

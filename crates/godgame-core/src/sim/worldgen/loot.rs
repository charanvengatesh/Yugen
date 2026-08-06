//! What comes out of a chest.
//!
//! ---- NO STORED STATE, AND THAT IS THE WHOLE DESIGN -------------------------
//! A chest holds nothing. It is one block code in the cell grid like any other,
//! and its contents are rolled from `hash(x, y, seed)` at the instant it is
//! broken. Nothing is written down, so:
//!
//!   - the chunk store can evict the chunk before the player ever reaches it and
//!     regenerate it later; the chest comes back holding the same three stacks.
//!   - There is no container persistence, no save format change, and no second
//!     source of truth that could disagree with the grid.
//!   - The sim stays the sole writer of cells. Opening a chest is a dig like any
//!     other; the loot is spawned as world items.
//!
//! The cost of that is that a chest cannot be a container you put things INTO,
//! and cannot be half-looted. Both are fine — it is a cache, not a backpack —
//! and both are recoverable later by promoting the block to a real entity
//! without touching a line of worldgen.
//!
//! ---- WHY THIS MODULE IS NOT PART OF GENERATION -----------------------------
//! It lives in `worldgen/` because it is the other half of the mark pass in
//! `worldgen/structs.rs` and shares its coordinate discipline exactly. But it is
//! a QUERY module, not a generation one: it reads the item registry, and it must
//! never be reached from the chunk-generation path. The dependency runs one way
//! — `structs.rs` knows nothing about this file.
//!
//! ---- WHAT MAKES A CHEST WORTH OPENING --------------------------------------
//! Two things, both recovered from the coordinate and neither stored:
//!
//!   DEPTH.  Read from `wcy - surface_at(wcx)`, against the same cut points
//!           `band_at` in structs.rs uses, so "cavern" here means the slice of
//!           world the cavern ROCK occupies.
//!   RARITY. The rarity of the TEMPLATE that placed the mark, recovered by
//!           `mark_at`. This is what makes a sealed vault (rarity 0.22) worth
//!           more than a surface cabin (0.4) at equal depth, and it is why the
//!           placement pass was split into resolve-then-stamp: without a
//!           re-derivable site there is nothing to ask.
//!
//! ---- WHAT CHANGED ON THE WAY OVER FROM TYPESCRIPT --------------------------
//! Two module-level mutable bindings are gone. The pooled `STACKS` array is now
//! a `Copy` value returned from the roll, and the cached fake `DecorContext` is
//! a [`StructQuery`] the caller owns — see [`roll_container_with`].

use std::sync::LazyLock;

use crate::config::{CAVERN_DEPTH, UNDERWORLD_DEPTH};
use crate::sim::worldgen::structs::{SiteQuery, StructQuery, mark_at};

// --- The content facade ------------------------------------------------------
// The compiled item registry is reached through here and nowhere else, exactly
// as `sim::materials` is the one door to the block tables.

pub use godgame_data::items::{ITEM_CODES, ITEM_COUNT, ITEM_IDS, item};

/// An item code. Item code 0 is a REAL item (`stone_chunk`), unlike block code 0
/// which is air.
pub type ItemId = u16;

/// "No item here." Because code 0 is a real item, anything that needs an
/// out-of-band empty value has to use this rather than 0.
pub const NO_ITEM: ItemId = 0xffff;

// --- Item resolution ---------------------------------------------------------

/// First id in a preference chain that the registry actually has, or
/// [`NO_ITEM`].
///
/// Same discipline as `pick` in features.rs and the fallbacks in layers.rs: the
/// registry is grown by other work, and a loot table that panicked on a renamed
/// id would take the whole generator down. Unlike those, a missing entry here is
/// survivable — it is dropped from the table at load and the remaining weights
/// simply redistribute.
fn item_code(ids: &str) -> ItemId {
    for id in ids.split('|') {
        if let Some((_, code)) = ITEM_CODES.iter().find(|(name, _)| *name == id) {
            return *code;
        }
    }
    NO_ITEM
}

// --- Tables ------------------------------------------------------------------

/// One weighted row of a loot table. Counts are inclusive.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Entry {
    code: ItemId,
    lo: i32,
    hi: i32,
    weight: f64,
}

/// Authoring shorthand: `(preference-chain, lo, hi, weight)`.
type Row = (&'static str, i32, i32, f64);

fn table(rows: &[Row]) -> Vec<Entry> {
    let mut out = Vec::new();
    for &(ids, lo, hi, weight) in rows {
        let code = item_code(ids);
        if code == NO_ITEM {
            continue; // registry does not have it (yet)
        }
        out.push(Entry {
            code,
            lo,
            hi,
            weight,
        });
    }
    out
}

/// The four tiers plus their weight sums, resolved once against the registry.
struct Tiers {
    rows: [Vec<Entry>; 4],
    /// Weight sums, computed once so a roll is one multiply and a walk.
    total: [f64; 4],
}

/// Four tiers, deliberately overlapping.
///
/// They are not a strict ladder where tier 3 replaces tier 0 — bandages and bars
/// appear at every depth, because a chest that can only ever contain the best
/// thing available makes the best thing available boring. What changes with tier
/// is the CEILING and the number of stacks: the tools and cores sit at the
/// bottom of the deep tables where they are a real find, and the bulk materials
/// stay to keep the median roll useful rather than insulting.
///
/// Every id is a preference chain, so a content rename degrades to the fallback
/// rather than silently emptying a tier.
static TIERS: LazyLock<Tiers> = LazyLock::new(|| {
    let rows = [
        // 0 — surface caches: a cabin's cupboard, a wreck's hold, a perch's stash.
        table(&[
            ("bandage", 1, 2, 20.0),
            ("wood_log", 3, 6, 16.0),
            ("flint", 2, 4, 12.0),
            ("coal", 2, 4, 12.0),
            ("copper_ore", 2, 4, 10.0),
            ("mushroom_stew", 1, 1, 8.0),
            ("sapling|pine_cone", 1, 2, 8.0),
            ("glass_pane", 2, 4, 6.0),
            ("hide|worm_hide", 1, 2, 6.0),
            ("gem", 1, 1, 2.0),
        ]),
        // 1 — shallow stashes: someone dug this far and meant to come back.
        table(&[
            ("copper_bar", 2, 4, 18.0),
            ("iron_ore", 2, 5, 16.0),
            ("bandage", 2, 3, 14.0),
            ("coal", 4, 8, 10.0),
            ("mushroom_stew|roast_grub", 1, 2, 8.0),
            ("crystal_shard", 1, 2, 6.0),
            ("bounce_pad", 1, 2, 6.0),
            ("pick_copper", 1, 1, 5.0),
            ("gem", 1, 1, 4.0),
            ("swiftwing_draught", 1, 1, 3.0),
        ]),
        // 2 — cavern hoards: shrines and grottoes, past the point of casual digging.
        table(&[
            ("iron_bar", 2, 4, 18.0),
            ("gold_ore", 2, 5, 14.0),
            ("crystal_shard", 2, 4, 12.0),
            ("gem", 1, 2, 10.0),
            ("emberward_draught", 1, 2, 8.0),
            ("swiftwing_draught", 1, 2, 8.0),
            ("bandage", 3, 5, 6.0),
            ("pearl_shard", 1, 2, 6.0),
            ("pick_iron", 1, 1, 6.0),
            ("ember_core", 1, 1, 4.0),
        ]),
        // 3 — vaults: sealed rooms and underworld ruins. The only place the late
        //     tools and the King's sword can be found rather than crafted.
        table(&[
            ("gold_bar", 2, 5, 16.0),
            ("gem", 2, 4, 14.0),
            ("pearl", 1, 2, 10.0),
            ("crystal_shard", 4, 8, 8.0),
            ("emberward_draught", 2, 3, 8.0),
            ("ember_core", 1, 2, 8.0),
            ("frozen_core", 1, 2, 8.0),
            ("pick_gold", 1, 1, 6.0),
            ("pick_gem", 1, 1, 3.0),
            ("sword_king", 1, 1, 2.0),
        ]),
    ];
    let total = std::array::from_fn(|i| rows[i].iter().map(|e| e.weight).sum());
    Tiers { rows, total }
});

/// Stacks per chest, by tier. A vault is worth the walk; a cupboard is not.
const PICKS_LO: [i32; 4] = [1, 2, 2, 3];
const PICKS_HI: [i32; 4] = [2, 3, 3, 4];

/// Pool size — the largest [`PICKS_HI`].
pub const MAX_CONTAINER_STACKS: usize = 4;

// --- Tier selection ----------------------------------------------------------

/// Depth contribution to the tier score, using the same cut points as `band_at`
/// in structs.rs (surface / shallow / cavern / deep / underworld). Fractional
/// rather than integral so that rarity can move a chest across a band boundary
/// instead of only ever nudging it within one.
fn depth_score(depth: i32) -> f64 {
    if depth < 8 {
        0.0
    } else if depth < 40 {
        0.6
    } else if depth < CAVERN_DEPTH {
        1.4
    } else if depth < UNDERWORLD_DEPTH {
        2.2
    } else {
        2.8
    }
}

/// Rarity contribution. `rarity` is the per-site acceptance probability, so LOW
/// is rare: 0.22 (sealed vault) earns +0.56, 0.4 (cabin) +0.2, 0.7 (underworld
/// ruin, which is common precisely because the underworld is hostile enough
/// already) loses 0.4. Scaled so the spread across everything authored is a
/// little under one full tier — rarity should be able to promote a chest, not
/// decide it on its own.
fn rarity_score(rarity: f64) -> f64 {
    (0.5 - rarity) * 2.0
}

// --- Rolling -----------------------------------------------------------------

/// One rolled stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LootStack {
    pub code: ItemId,
    pub count: i32,
}

/// The contents of one container.
///
/// The TypeScript wrote these into a module-level pool and returned a count, so
/// that rolling never allocated. A fixed-size `Copy` struct is the same zero
/// allocations without the shared mutable array — which matters here, because
/// this is reachable from any thread that can break a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LootRoll {
    stacks: [LootStack; MAX_CONTAINER_STACKS],
    len: usize,
}

impl LootRoll {
    /// The valid stacks. Empty when the tier resolved empty, which can only
    /// happen if content removed every item a tier names.
    #[inline]
    pub fn stacks(&self) -> &[LootStack] {
        &self.stacks[..self.len]
    }

    /// How many stacks are valid.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for LootRoll {
    fn default() -> LootRoll {
        LootRoll {
            stacks: [LootStack::default(); MAX_CONTAINER_STACKS],
            len: 0,
        }
    }
}

/// Salted positional hash, exactly as structs.rs salts its placement draws.
#[inline]
fn h(q: &StructQuery, x: i32, y: i32, salt: i32) -> f64 {
    q.noise().hash2(x + salt * 15731, y - salt * 51203)
}

/// Hash value -> integer in [lo, hi].
#[inline]
fn ri(v: f64, lo: i32, hi: i32) -> i32 {
    let n = lo + (v * f64::from(hi - lo + 1)).floor() as i32;
    if n > hi {
        hi
    } else if n < lo {
        lo
    } else {
        n
    }
}

/// Roll the contents of the container at cell (wcx, wcy).
///
/// Pure in (seed, wcx, wcy). Call it twice and get the same answer; call it
/// after the chunk has been evicted and regenerated and get the same answer.
///
/// This builds a [`StructQuery`] per call, which fills a noise permutation
/// table. A caller that opens more than one chest should keep a query and use
/// [`roll_container_with`]; the TypeScript did the same thing with a cached
/// context behind a module-level `let`, which is not a shape that survives a
/// thread pool.
pub fn roll_container(seed: u32, wcx: i32, wcy: i32) -> LootRoll {
    let mut q = StructQuery::new(seed);
    roll_container_with(&mut q, wcx, wcy)
}

/// As [`roll_container`], against a query the caller keeps.
///
/// The query is `&mut` only because it memoises the heightmap; nothing about the
/// roll depends on what was asked before it, which is what the regeneration test
/// pins down.
pub fn roll_container_with(q: &mut StructQuery, wcx: i32, wcy: i32) -> LootRoll {
    // Rarity of the template that put this mark here. A container with no
    // recoverable host — a feature-placed one, or a block the player carried down
    // and set themselves — is scored as the commonest thing there is, so it can
    // never out-earn a real structure at the same depth.
    let hit = mark_at(q, wcx, wcy);
    let rarity = hit.map_or(1.0, |h| h.template.rarity);
    let depth = wcy - q.surface_at(wcx);

    // The jitter is one-sided: it can only promote. A chest is allowed to be
    // luckier than its site, never poorer, so a vault always reads as a vault.
    let score = depth_score(depth) + rarity_score(rarity) + h(q, wcx, wcy, 1) * 0.9;
    let tier = (score.floor() as i32).clamp(0, TIERS.rows.len() as i32 - 1) as usize;

    let rows = &TIERS.rows[tier];
    let total = TIERS.total[tier];
    let mut roll = LootRoll::default();
    if rows.is_empty() || total <= 0.0 {
        return roll;
    }

    let picks = ri(h(q, wcx, wcy, 2), PICKS_LO[tier], PICKS_HI[tier]);
    for i in 0..picks {
        // Each pick draws independently, so a chest can hold two of the same
        // thing. That is a feature: it is how a table with one great entry
        // produces the occasional double rather than capping at one of everything.
        let mut x = h(q, wcx, wcy, 3 + i) * total;
        let mut e = rows[rows.len() - 1];
        for row in rows {
            x -= row.weight;
            if x < 0.0 {
                e = *row;
                break;
            }
        }
        roll.stacks[i as usize] = LootStack {
            code: e.code,
            count: ri(h(q, wcx, wcy, 11 + i), e.lo, e.hi),
        };
    }
    roll.len = picks as usize;
    roll
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chest_holds_the_same_thing_however_often_it_is_asked() {
        // THE property this whole module exists for: contents are a pure function
        // of (seed, x, y), so a chunk evicted and regenerated hands the player the
        // same cache it would have the first time.
        let seed = 0xC0FFEE;
        for (x, y) in [(0, 60), (-1234, 300), (77_777, 512), (12, 1), (-9, -40)] {
            let a = roll_container(seed, x, y);
            let b = roll_container(seed, x, y);
            assert_eq!(a, b, "({x}, {y}) rolled differently twice");
        }
    }

    #[test]
    fn a_shared_query_agrees_with_a_fresh_one_whatever_it_was_asked_before() {
        // The memo inside `StructQuery` must be invisible. If a warm heightmap
        // ever produced a different surface row from a cold one, a chest opened
        // after a walk across the world would hold something else.
        let seed = 4242;
        let mut warm = StructQuery::new(seed);
        // Warm the memo with unrelated columns, in an order nothing else uses.
        for x in [-500_000, 31, 999, -7, 200_000] {
            let _ = roll_container_with(&mut warm, x, 400);
        }
        for (x, y) in [(0, 96), (640, 220), (-320, 480)] {
            let cold = roll_container(seed, x, y);
            let hot = roll_container_with(&mut warm, x, y);
            assert_eq!(cold, hot, "({x}, {y}) depends on query history");
        }
    }

    #[test]
    fn a_different_seed_is_a_different_world() {
        // Not a strict requirement of the design, but if this ever held for every
        // cell the positional hash would have stopped depending on the seed.
        let mut differs = 0;
        for x in 0..64 {
            if roll_container(1, x * 7, 300) != roll_container(2, x * 7, 300) {
                differs += 1;
            }
        }
        assert!(differs > 0, "two seeds produced identical loot everywhere");
    }

    #[test]
    fn every_tier_resolved_against_the_registry() {
        for (i, rows) in TIERS.rows.iter().enumerate() {
            assert!(!rows.is_empty(), "tier {i} lost every item it names");
            assert!(TIERS.total[i] > 0.0, "tier {i} has no weight");
            for e in rows {
                assert_ne!(e.code, NO_ITEM);
                assert!(
                    e.lo <= e.hi,
                    "tier {i} entry {:?} has an inverted count",
                    e.code
                );
                assert!(e.weight > 0.0);
            }
        }
    }

    #[test]
    fn a_roll_never_exceeds_the_pool_and_names_real_items() {
        let seed = 99;
        for x in -40..40 {
            let roll = roll_container(seed, x * 13, 260);
            assert!(roll.len() <= MAX_CONTAINER_STACKS);
            assert!(roll.len() >= PICKS_LO[0] as usize);
            for s in roll.stacks() {
                assert!(
                    (s.code as usize) < ITEM_COUNT,
                    "code {} is not an item",
                    s.code
                );
                assert!(s.count >= 1, "a stack of {} is not worth spawning", s.count);
            }
        }
    }

    #[test]
    fn deeper_is_never_a_worse_tier_for_the_same_column() {
        // `depth_score` is monotone, and the jitter is one-sided, so the FLOOR of
        // what a depth can roll only ever rises.
        let mut last = -1.0;
        for depth in [0, 7, 8, 39, 40, 109, 110, 469, 470, 4000] {
            let s = depth_score(depth);
            assert!(s >= last, "depth {depth} scored below a shallower one");
            last = s;
        }
    }

    #[test]
    fn rarity_promotes_the_rare_and_demotes_the_common() {
        assert!(rarity_score(0.22) > rarity_score(0.4));
        assert!(rarity_score(0.4) > rarity_score(0.7));
        assert!(rarity_score(0.7) < 0.0);
        // A container with no host is scored as the commonest thing there is.
        assert!(rarity_score(1.0) <= rarity_score(0.7));
    }
}

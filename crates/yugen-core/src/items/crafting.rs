//! Spending an inventory against the recipe table.
//!
//! Separate from `registry.rs` only to keep the dependency arrow straight:
//! [`Inventory`] imports the registry for `ITEM_STACK`, so the registry cannot
//! import [`Inventory`] back. This module is where the two meet.
//!
//! # The station field is live now
//!
//! This file used to say: *"no station blocks exist yet, so this deliberately
//! ignores it... the alternative — hiding recipes behind furniture that cannot
//! be placed — would make the whole content graph unreachable and untestable."*
//! That was right, and it stopped being right the moment `workbench`, `furnace`
//! and `anvil` were authored. Every recipe in `content/items/*.toml` has always
//! declared a `station`; it was parsed, validated and thrown away.
//!
//! So [`can_craft`] takes the set of stations within reach. What that set IS is
//! not this module's business — it needs a grid and a body, and this file has
//! neither — so it arrives as a [`Reach`] and the host works it out.
//!
//! [`Reach::HAND`] is the empty set and is not a special case anywhere: `hand`
//! recipes are craftable from it because `hand` is in every set, which is the
//! whole of the rule. That is what keeps a player who has just spawned able to
//! make the workbench that unlocks the rest.

use super::inventory::Inventory;
use super::registry::{Recipe, Station, recipes};

/// The crafting stations a player can currently use.
///
/// A set rather than a single station, because a workbench and a furnace built
/// beside each other are both in reach and a player standing between them should
/// not have to pick one. Four possible members and a `u8` of flags; there is no
/// room for this to need anything cleverer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reach(u8);

impl Reach {
    /// Nothing but your hands. The state a player spawns in, and the reason
    /// `workbench`'s own recipe is a `hand` one.
    pub const HAND: Reach = Reach(0);

    /// Every station. Not something the world ever produces — it is for asking
    /// "would this be craftable ANYWHERE", which is how a caller tells "you
    /// cannot afford it" apart from "you are in the wrong place".
    pub const ALL: Reach = Reach(1 | 2 | 4);

    /// Add a station.
    #[must_use]
    pub const fn with(self, station: Station) -> Reach {
        Reach(self.0 | Self::bit(station))
    }

    /// Whether a recipe needing `station` can be made from here.
    ///
    /// `hand` is in EVERY set, including the empty one. It is not a station a
    /// player stands next to; it is the absence of a requirement, and treating
    /// it as a member would mean writing a special case at every call site.
    #[must_use]
    pub const fn has(self, station: Station) -> bool {
        matches!(station, Station::Hand) || self.0 & Self::bit(station) != 0
    }

    const fn bit(station: Station) -> u8 {
        match station {
            Station::Hand => 0,
            Station::Workbench => 1,
            Station::Furnace => 2,
            Station::Anvil => 4,
        }
    }
}

/// Does this inventory hold every ingredient, is a station in reach, and is
/// there room for the output?
pub fn can_craft(inv: &Inventory, r: &Recipe, reach: Reach) -> bool {
    if !reach.has(r.station) {
        return false;
    }
    for i in 0..r.in_items.len() {
        if !inv.has(r.in_items[i], r.in_counts[i] as u32) {
            return false;
        }
    }
    // Refuse rather than destroy the ingredients for an output with nowhere to go.
    !inv.full(r.out)
}

/// Pay for a recipe and credit the output. Returns false without touching the
/// inventory if it cannot be afforded — the check runs to completion before the
/// first removal, so a half-paid craft is not reachable.
pub fn craft(inv: &mut Inventory, r: &Recipe, reach: Reach) -> bool {
    if !can_craft(inv, r, reach) {
        return false;
    }
    for i in 0..r.in_items.len() {
        inv.remove(r.in_items[i], r.in_counts[i] as u32);
    }
    inv.add(r.out, r.out_count as u32);
    true
}

/// Index of the first affordable recipe at or after `from`, wrapping, or `None`.
///
/// `from` lets the caller step through the affordable set with repeated presses
/// instead of being pinned to whichever recipe happens to sort first — the same
/// "press it again to cycle" idiom the old build palette used.
pub fn next_craftable(inv: &Inventory, from: usize, reach: Reach) -> Option<usize> {
    let rs = recipes();
    let n = rs.len();
    if n == 0 {
        return None;
    }
    for k in 0..n {
        let i = (from + k) % n;
        if can_craft(inv, &rs[i], reach) {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every station in reach.
    ///
    /// The tests below predate stations and are about INGREDIENTS — whether an
    /// inventory can pay, whether a short one is refused, whether the scan
    /// wraps. Running them from a full workshop keeps them about that. The
    /// station rule has its own tests, at the bottom.
    const ALL: Reach = Reach::HAND
        .with(Station::Workbench)
        .with(Station::Furnace)
        .with(Station::Anvil);
    use crate::items::registry::{ITEM_STACK, ItemCode, Recipe, Station, item};

    /// A recipe with at least two distinct ingredients, so "short by one" can be
    /// tested against a real content row rather than a fabricated one.
    fn a_multi_ingredient_recipe() -> &'static Recipe {
        recipes()
            .iter()
            .find(|r| r.in_items.len() >= 2)
            .expect("content has no multi-ingredient recipe")
    }

    fn stock(inv: &mut Inventory, r: &Recipe, shortfall_on: Option<usize>) {
        for i in 0..r.in_items.len() {
            let want = r.in_counts[i] as u32;
            let give = if shortfall_on == Some(i) {
                want - 1
            } else {
                want
            };
            inv.add(r.in_items[i], give);
        }
    }

    #[test]
    fn exactly_enough_crafts_and_one_short_refuses() {
        let r = a_multi_ingredient_recipe();

        // Short by exactly one of the LAST ingredient: the check must run to
        // completion before anything is spent.
        let short_on = r.in_items.len() - 1;
        let mut inv = Inventory::new();
        stock(&mut inv, r, Some(short_on));
        let before: Vec<u32> = r.in_items.iter().map(|&c| inv.count_of(c)).collect();
        assert!(!can_craft(&inv, r, ALL));
        assert!(!craft(&mut inv, r, ALL), "one short must refuse");
        let after: Vec<u32> = r.in_items.iter().map(|&c| inv.count_of(c)).collect();
        assert_eq!(before, after, "a refused craft must spend nothing");
        assert_eq!(inv.count_of(r.out), 0);

        // Exactly enough: succeeds, spends everything, credits the output.
        let mut inv = Inventory::new();
        stock(&mut inv, r, None);
        assert!(can_craft(&inv, r, ALL));
        assert!(craft(&mut inv, r, ALL));
        for i in 0..r.in_items.len() {
            let left = inv.count_of(r.in_items[i]);
            // An ingredient that is also the output keeps what the craft paid.
            let expected = if r.in_items[i] == r.out {
                r.out_count as u32
            } else {
                0
            };
            assert_eq!(left, expected, "ingredient {i} was not spent exactly");
        }
        assert!(inv.count_of(r.out) >= r.out_count as u32);
    }

    #[test]
    fn a_craft_with_nowhere_to_put_the_output_is_refused_not_wasted() {
        // One ingredient, one output, a deliberately synthetic recipe so the
        // test does not depend on which content row happens to be simplest.
        let ingredient: ItemCode = item::COAL;
        let out: ItemCode = item::TORCH;
        let r = Recipe {
            out,
            out_count: 1,
            in_items: vec![ingredient],
            in_counts: vec![1],
            station: Station::Hand,
        };

        let mut inv = Inventory::new();
        // Fill every slot with the output item so nothing more of it fits, then
        // put the ingredient... nowhere. There is no room for it either, so
        // stock it first and fill around it.
        inv.add(ingredient, 1);
        inv.add(
            out,
            ITEM_STACK[out as usize] as u32 * (super::super::SLOT_COUNT as u32 - 1),
        );
        assert!(inv.full(out));
        assert!(inv.has(ingredient, 1));
        assert!(!can_craft(&inv, &r, ALL));
        assert!(!craft(&mut inv, &r, ALL));
        assert_eq!(inv.count_of(ingredient), 1, "the ingredient survived");
    }

    #[test]
    fn next_craftable_wraps_and_reports_nothing_for_an_empty_pack() {
        let inv = Inventory::new();
        assert_eq!(
            next_craftable(&inv, 0, ALL),
            None,
            "an empty pack affords nothing"
        );

        let r = a_multi_ingredient_recipe();
        let idx = recipes().iter().position(|x| std::ptr::eq(x, r)).unwrap();
        let mut inv = Inventory::new();
        stock(&mut inv, r, None);
        // Starting past it, the scan wraps around to find it.
        let found = next_craftable(&inv, idx + 1, ALL).expect("the scan must wrap");
        assert!(can_craft(&inv, &recipes()[found], ALL));
    }

    /// The rule the whole feature is: a recipe needs its station in reach.
    #[test]
    fn a_recipe_needs_its_station_and_hand_needs_nothing() {
        let bench = recipes()
            .iter()
            .find(|r| r.station == Station::Workbench)
            .expect("some recipe wants a workbench");
        let by_hand = recipes()
            .iter()
            .find(|r| r.station == Station::Hand)
            .expect("some recipe is made by hand");

        // Paid for in full, both of them, so the only thing left to refuse is
        // the station.
        let mut inv = Inventory::new();
        for r in [bench, by_hand] {
            for i in 0..r.in_items.len() {
                inv.add(r.in_items[i], u32::from(r.in_counts[i]));
            }
        }

        assert!(
            can_craft(&inv, by_hand, Reach::HAND),
            "a hand recipe must be craftable with nothing in reach — it is how a \
             player who has just spawned makes the workbench that unlocks the rest"
        );
        assert!(
            !can_craft(&inv, bench, Reach::HAND),
            "a workbench recipe must be refused with no workbench in reach"
        );
        assert!(
            can_craft(&inv, bench, Reach::HAND.with(Station::Workbench)),
            "and allowed with one"
        );
    }

    /// A station in reach unlocks its own recipes and nobody else's.
    #[test]
    fn one_station_does_not_stand_in_for_another() {
        let r = Reach::HAND.with(Station::Furnace);
        assert!(r.has(Station::Hand), "hand is in every set");
        assert!(r.has(Station::Furnace));
        assert!(!r.has(Station::Workbench));
        assert!(!r.has(Station::Anvil));
    }

    /// Two stations built beside each other are both usable, which is why this
    /// is a set and not a single station.
    #[test]
    fn standing_between_two_stations_reaches_both() {
        let r = Reach::HAND.with(Station::Workbench).with(Station::Anvil);
        assert!(r.has(Station::Workbench) && r.has(Station::Anvil));
        assert!(!r.has(Station::Furnace));
    }

    /// The scan skips what it cannot make HERE, rather than offering it and
    /// failing on the keypress.
    #[test]
    fn the_scan_only_offers_what_this_reach_can_make() {
        let bench = recipes()
            .iter()
            .find(|r| r.station == Station::Workbench)
            .expect("some recipe wants a workbench");
        let mut inv = Inventory::new();
        for i in 0..bench.in_items.len() {
            inv.add(bench.in_items[i], u32::from(bench.in_counts[i]));
        }
        let at = recipes()
            .iter()
            .position(|r| std::ptr::eq(r, bench))
            .unwrap();

        assert_ne!(
            next_craftable(&inv, at, Reach::HAND.with(Station::Workbench)),
            None,
            "with a bench in reach it is on offer"
        );
        assert_ne!(
            next_craftable(&inv, at, Reach::HAND),
            Some(at),
            "without one it must not be the thing `C` would make"
        );
    }
}

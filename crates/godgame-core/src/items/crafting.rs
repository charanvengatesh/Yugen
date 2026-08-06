//! Spending an inventory against the recipe table.
//!
//! Separate from `registry.rs` only to keep the dependency arrow straight:
//! [`Inventory`] imports the registry for `ITEM_STACK`, so the registry cannot
//! import [`Inventory`] back. This module is where the two meet.
//!
//! There is no crafting SCREEN in this pass. `content/items/*.toml` declares a
//! `station` per recipe, and no station blocks exist yet, so this deliberately
//! ignores it: everything a full inventory can pay for is craftable, and the
//! station field stays authored and validated for whoever builds the workbench.
//! The alternative — hiding recipes behind furniture that cannot be placed —
//! would make the whole content graph unreachable and untestable.

use super::inventory::Inventory;
use super::registry::{Recipe, recipes};

/// Does this inventory hold every ingredient, and is there room for the output?
pub fn can_craft(inv: &Inventory, r: &Recipe) -> bool {
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
pub fn craft(inv: &mut Inventory, r: &Recipe) -> bool {
    if !can_craft(inv, r) {
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
pub fn next_craftable(inv: &Inventory, from: usize) -> Option<usize> {
    let rs = recipes();
    let n = rs.len();
    if n == 0 {
        return None;
    }
    for k in 0..n {
        let i = (from + k) % n;
        if can_craft(inv, &rs[i]) {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(!can_craft(&inv, r));
        assert!(!craft(&mut inv, r), "one short must refuse");
        let after: Vec<u32> = r.in_items.iter().map(|&c| inv.count_of(c)).collect();
        assert_eq!(before, after, "a refused craft must spend nothing");
        assert_eq!(inv.count_of(r.out), 0);

        // Exactly enough: succeeds, spends everything, credits the output.
        let mut inv = Inventory::new();
        stock(&mut inv, r, None);
        assert!(can_craft(&inv, r));
        assert!(craft(&mut inv, r));
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
        assert!(!can_craft(&inv, &r));
        assert!(!craft(&mut inv, &r));
        assert_eq!(inv.count_of(ingredient), 1, "the ingredient survived");
    }

    #[test]
    fn next_craftable_wraps_and_reports_nothing_for_an_empty_pack() {
        let inv = Inventory::new();
        assert_eq!(
            next_craftable(&inv, 0),
            None,
            "an empty pack affords nothing"
        );

        let r = a_multi_ingredient_recipe();
        let idx = recipes().iter().position(|x| std::ptr::eq(x, r)).unwrap();
        let mut inv = Inventory::new();
        stock(&mut inv, r, None);
        // Starting past it, the scan wraps around to find it.
        let found = next_craftable(&inv, idx + 1).expect("the scan must wrap");
        assert!(can_craft(&inv, &recipes()[found]));
    }
}

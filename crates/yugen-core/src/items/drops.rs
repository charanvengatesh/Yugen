//! Turning broken cells into item stacks.
//!
//! WHERE THIS RUNS. In the tool layer, over the cells that are ABOUT to be
//! removed — never in the sim. [`crate::sim::edits`] is the single writer of
//! cells and it deliberately knows nothing about items; asking it what it just
//! destroyed would mean threading a report back out of the automata and a second
//! copy of the drop tables to read it with. The tool already has the brush rect
//! and read access to the grid, so it reads the cells first and applies the edit
//! second. See `BuildTool::dig`.
//!
//! The TypeScript had a second reason for that ordering — the sim lived in a
//! worker and "the worker's bundle does not carry the item registry" — and that
//! one is gone. One grid, one thread, one writer. The ordering survives on the
//! first reason alone, which was always the real one.
//!
//! DETERMINISM. Drops are rolled from a POSITIONAL HASH of the world cell, not
//! from a running stream like [`crate::sim::rng`]. Two reasons, and the second is
//! the real one:
//!
//!  1. The sim RNG is the automata's, and its determinism is order-dependent —
//!     one stream drawn in one fixed sweep order. A mining consumer sharing it
//!     would perturb every draw after it, and mining is not part of the sim's
//!     replay contract.
//!  2. A stream makes a drop depend on how many cells you happened to break
//!     first. A hash makes cell (x,y) always yield the same thing no matter what
//!     order — or how many times over a save/load — you get to it. That is the
//!     property that makes "I dug here and got a gem" reproducible from a seed,
//!     which is what determinism is actually for here.

use super::inventory::Inventory;
use super::registry::{ITEM_COUNT, ItemCode, item_code_of, item_for_block};
use crate::config::SEED;
use crate::sim::materials::{CellId, mat_by_code};

/// 2^-32, the scale that turns a raw `u32` into a float in `[0, 1)`. Written out
/// rather than computed so it reads identically to the TypeScript literal it came
/// from — the same constant [`crate::sim::rng`] uses, restated because these are
/// two independent algorithms that happen to share a scale.
const INV_2_POW_32: f64 = 2.3283064365386963e-10;

/// 32-bit integer hash of a world cell plus a salt (the drop-table row index, so
/// a block with three drop entries rolls three independent values).
///
/// The TypeScript spelled every multiply `Math.imul` because plain `*` on values
/// this large goes through doubles and loses the low bits that carry the
/// avalanche. `Math.imul` IS a wrapping 32-bit multiply, so [`u32::wrapping_mul`]
/// is the same operation with the workaround deleted; the sign difference between
/// them is invisible to `^` and `>>` on the bit patterns that follow.
fn hash32(x: i32, y: i32, salt: u32) -> u32 {
    let mut h = SEED ^ (x as u32).wrapping_mul(0x27d4_eb2d) ^ (y as u32).wrapping_mul(0x1656_67b1);
    h = (h ^ salt).wrapping_mul(0x9e37_79b1);
    h = (h ^ (h >> 15)).wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^ (h >> 15)
}

/// The hash as a float in `[0,1)`. One multiply, same scaling as `sim/rng.rs`.
///
/// `f64` and not `f32`: this is compared against a `chance` and multiplied by a
/// count range, and the TypeScript necessarily did both in doubles. Widening the
/// comparison is free — it happens once per drop-table row per broken cell, not
/// per pixel — and it keeps the roll bit-identical to the original rather than
/// bounded-close to it, which is what makes "dig here, get a gem" reproducible
/// ACROSS the two implementations and not just within this one.
fn unit(x: i32, y: i32, salt: u32) -> f64 {
    hash32(x, y, salt) as f64 * INV_2_POW_32
}

/// Accumulator for one dig tick's yield.
///
/// A brush of radius 4 breaks up to 49 cells that are almost always the same two
/// or three materials, so the natural output is "N of item X" rather than 49
/// separate stacks. Counts live in a dense array indexed by item code (79 slots,
/// allocated once for the lifetime of the game) and the codes that were actually
/// touched are tracked in a short list, so [`DropBag::clear`] is O(touched) rather
/// than O(ITEM_COUNT) and the whole thing allocates nothing per dig.
#[derive(Clone, Debug)]
pub struct DropBag {
    counts: [u32; ITEM_COUNT],
    touched: [ItemCode; ITEM_COUNT],
    n: usize,
}

impl Default for DropBag {
    fn default() -> DropBag {
        DropBag::new()
    }
}

impl DropBag {
    /// An empty bag. `const` so a tool can hold one in a `const` initialiser.
    pub const fn new() -> DropBag {
        DropBag {
            counts: [0; ITEM_COUNT],
            touched: [0; ITEM_COUNT],
            n: 0,
        }
    }

    /// How many distinct item kinds are in the bag.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Nothing was collected.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Item code of the `i`th distinct kind, in the order they were first added.
    #[inline]
    pub fn code_at(&self, i: usize) -> ItemCode {
        self.touched[i]
    }

    /// How many of the `i`th distinct kind.
    #[inline]
    pub fn count_at(&self, i: usize) -> u32 {
        self.counts[self.touched[i] as usize]
    }

    /// Every kind in the bag, for a caller that would rather not index.
    pub fn iter(&self) -> impl Iterator<Item = (ItemCode, u32)> + '_ {
        (0..self.n).map(|i| (self.code_at(i), self.count_at(i)))
    }

    pub fn add(&mut self, code: ItemCode, count: u32) {
        if count == 0 {
            return;
        }
        let c = code as usize;
        if self.counts[c] == 0 {
            self.touched[self.n] = code;
            self.n += 1;
        }
        self.counts[c] += count;
    }

    pub fn clear(&mut self) {
        for i in 0..self.n {
            self.counts[self.touched[i] as usize] = 0;
        }
        self.n = 0;
    }

    /// Push everything straight into an inventory, leaving behind whatever did
    /// not fit. Returns true if the bag is now empty. The world-item path uses
    /// the bag directly instead; this is the "creative-ish, no entities" shortcut
    /// and the fallback when there is nowhere to spawn.
    pub fn flush_into(&mut self, inv: &mut Inventory) -> bool {
        let mut write = 0;
        for i in 0..self.n {
            let code = self.touched[i];
            let left = inv.add(code, self.counts[code as usize]);
            self.counts[code as usize] = left;
            if left > 0 {
                self.touched[write] = code;
                write += 1;
            }
        }
        self.n = write;
        write == 0
    }
}

/// Roll the drop table for one cell of `block` at world cell (wx,wy) into `bag`.
///
/// An explicit `drop` list in `content/blocks/*.toml` wins. With no list the
/// block yields "its own item" per the block schema's doc for that field, which
/// is resolved through the block -> item reverse index; a block nothing places
/// (lava, fire, spikes) simply yields nothing, which is the correct reading of
/// "no item exists for this".
pub fn roll_cell_drops(block: CellId, wx: i32, wy: i32, bag: &mut DropBag) {
    let Some(table) = mat_by_code(block).drop else {
        if let Some(own) = item_for_block(block) {
            bag.add(own, 1);
        }
        return;
    };

    for (i, entry) in table.iter().enumerate() {
        // Salt by row so a block with several entries does not roll them all
        // identically off the same cell hash.
        let salt = (i as u32) * 2 + 1;
        let roll = unit(wx, wy, salt);
        let chance = entry.chance as f64;
        if chance < 1.0 && roll >= chance {
            continue;
        }

        let (lo, hi) = (entry.count[0], entry.count[1]);
        let n = if lo == hi {
            lo as i32
        } else {
            // `as i32` truncates toward zero, which is what the TypeScript's
            // `| 0` did to the same product.
            lo as i32 + (unit(wx, wy, salt + 1) * (hi - lo + 1.0) as f64) as i32
        };
        if n <= 0 {
            continue;
        }

        // An exact lookup, NOT the `a|b` chain `item_code_for_drop` walks: block
        // drop tables name one item, and a tombstoned or unresolved id is
        // skipped rather than guessed at.
        let Some(code) = item_code_of(entry.item) else {
            continue;
        };
        bag.add(code, n as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::registry::{ITEM_STACK, item};
    use crate::sim::materials::{BLOCKS, block};

    #[test]
    fn a_bag_dedups_by_kind_and_clears_in_touched_order() {
        let mut bag = DropBag::new();
        assert!(bag.is_empty());
        bag.add(item::COAL, 2);
        bag.add(item::STONE_CHUNK, 5);
        bag.add(item::COAL, 3);
        bag.add(item::GEM, 0); // zero is not a kind
        assert_eq!(bag.len(), 2);
        assert_eq!(bag.code_at(0), item::COAL);
        assert_eq!(bag.count_at(0), 5);
        assert_eq!(bag.code_at(1), item::STONE_CHUNK);
        assert_eq!(bag.count_at(1), 5);

        bag.clear();
        assert!(bag.is_empty());
        // The counts really were zeroed, not just the length reset.
        bag.add(item::COAL, 1);
        assert_eq!(bag.count_at(0), 1);
    }

    #[test]
    fn flushing_leaves_behind_only_what_did_not_fit() {
        let mut inv = Inventory::new();
        let cap = ITEM_STACK[item::STONE_CHUNK as usize] as u32;
        // Fill the pack with something that is not what we are about to flush,
        // leaving exactly one slot.
        inv.add(item::COAL, ITEM_STACK[item::COAL as usize] as u32 * 29);

        let mut bag = DropBag::new();
        bag.add(item::STONE_CHUNK, cap + 7);
        assert!(!bag.flush_into(&mut inv), "seven could not fit");
        assert_eq!(bag.len(), 1);
        assert_eq!(bag.count_at(0), 7);
        assert_eq!(inv.count_of(item::STONE_CHUNK), cap);

        // Free a whole slot and the rest goes in.
        inv.remove(item::COAL, ITEM_STACK[item::COAL as usize] as u32);
        assert!(bag.flush_into(&mut inv));
        assert!(bag.is_empty());
        assert_eq!(inv.count_of(item::STONE_CHUNK), cap + 7);
    }

    /// The whole point of hashing the coordinate instead of drawing from a
    /// stream: the same cell answers the same thing forever, in any order, any
    /// number of times.
    #[test]
    fn rolls_are_a_pure_function_of_block_and_coordinate() {
        let cells = [(0, 0), (17, -3), (-1024, 4096), (7, 7)];
        let blocks = [block::STONE, block::DIRT, block::GRAVEL, block::CRYSTAL];

        let mut first = Vec::new();
        for &b in &blocks {
            for &(x, y) in &cells {
                let mut bag = DropBag::new();
                roll_cell_drops(b, x, y, &mut bag);
                first.push(bag.iter().collect::<Vec<_>>());
            }
        }

        // Re-roll in the reverse order, into a REUSED bag, and get the same
        // answers back — order-independence and pool-independence at once.
        let mut bag = DropBag::new();
        let mut again = Vec::new();
        for &b in blocks.iter().rev() {
            for &(x, y) in cells.iter().rev() {
                bag.clear();
                roll_cell_drops(b, x, y, &mut bag);
                again.push(bag.iter().collect::<Vec<_>>());
            }
        }
        again.reverse();
        assert_eq!(first, again);
    }

    #[test]
    fn neighbouring_cells_do_not_all_roll_alike() {
        // A block with a chance-gated table must actually vary across the world,
        // or the hash has collapsed and every seam is all-or-nothing.
        let gated = BLOCKS
            .iter()
            .find(|b| {
                b.drop
                    .is_some_and(|t| t.iter().any(|e| e.chance > 0.0 && e.chance < 1.0))
            })
            .expect("no block has a chance-gated drop");

        let mut hits = 0;
        let mut misses = 0;
        let mut bag = DropBag::new();
        for x in 0..40 {
            for y in 0..40 {
                bag.clear();
                roll_cell_drops(gated.code, x, y, &mut bag);
                if bag.is_empty() {
                    misses += 1
                } else {
                    hits += 1
                }
            }
        }
        assert!(hits > 0 && misses > 0, "{} rolls do not vary", gated.id);
    }

    #[test]
    fn a_block_with_no_table_yields_the_item_that_places_it() {
        // Every block some item places must be recoverable by digging it.
        for def in BLOCKS.iter() {
            if def.drop.is_some() || def.code == 0 {
                continue;
            }
            let mut bag = DropBag::new();
            roll_cell_drops(def.code, 3, 5, &mut bag);
            match item_for_block(def.code) {
                Some(own) => {
                    assert_eq!(bag.len(), 1, "{} should drop its own item", def.id);
                    assert_eq!(bag.code_at(0), own);
                    assert_eq!(bag.count_at(0), 1);
                }
                // A block nothing places (lava, fire, spikes) yields nothing.
                None => assert!(bag.is_empty(), "{} invented a drop", def.id),
            }
        }
    }
}

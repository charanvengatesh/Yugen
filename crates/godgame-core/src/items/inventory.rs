//! The player's inventory: a fixed slot array, stored as two parallel arrays
//! rather than an array of `{ item, count }` records.
//!
//! WHY SoA. Every operation here is a linear slot scan — add walks looking for a
//! partial stack, `count_of` walks summing, the HUD walks drawing. With records
//! that is one pointer chase per slot and a garbage object per stack that is ever
//! created or destroyed; with two flat arrays it is two contiguous reads and the
//! whole inventory is 4 bytes a slot. Nothing in this file allocates at all, so a
//! hundred pickups a second cost nothing.
//!
//! SLOT EMPTINESS IS `count == 0`, never a sentinel item code. Item code 0 is a
//! real item (see [`NO_ITEM`](super::registry::NO_ITEM)), so a code-based empty
//! test would make `stone_chunk` unstorable — a bug that would only show up in
//! the one material the player picks up first.
//!
//! # What the port changed
//!
//! The two arrays and `selected` were public and mutable in the TypeScript.
//! They are private here, behind accessors, because of [`Inventory::revision`]:
//! a counter that must be bumped on every mutation is only trustworthy if
//! mutation has to go through this file. `Game.ts` reached in and wrote
//! `inv.selected = 0` directly at the start of a run; [`Inventory::select_slot`]
//! is that, and it deliberately does NOT bump the revision — see the note there.

use super::registry::{ITEM_STACK, ItemCode, item_code_of, player_weapon};
use crate::entities::player::{AmmoSource, PlayerWeapon};

/// Total slots. The first [`HOTBAR`] are the ones the HUD draws and the number
/// keys select; the rest is backpack the crafting code can still spend from.
///
/// 30 = 10 hotbar + 2 rows. Big enough that a deep run does not force a triage
/// decision every minute, small enough that "you are carrying too much" is a real
/// state the game can put you in.
pub const SLOT_COUNT: usize = 30;

/// Slots the hotbar shows, and the number keys reach.
pub const HOTBAR: usize = 10;

/// 30 slots, a selection, and a revision counter.
#[derive(Clone, Debug)]
pub struct Inventory {
    /// Item code per slot. Meaningless where `count[i] == 0`.
    item: [ItemCode; SLOT_COUNT],
    /// Stack size per slot. 0 = empty slot.
    count: [u16; SLOT_COUNT],
    /// Selected hotbar slot, `0..HOTBAR`. The held item.
    selected: usize,
    /// Bumped on every mutation. The HUD and the crafting scan use it to skip
    /// work on frames where nothing moved — cheaper than diffing 30 slots.
    revision: u32,
}

impl Default for Inventory {
    fn default() -> Inventory {
        Inventory::new()
    }
}

impl Inventory {
    /// An empty inventory with slot 0 selected.
    pub const fn new() -> Inventory {
        Inventory {
            item: [0; SLOT_COUNT],
            count: [0; SLOT_COUNT],
            selected: 0,
            revision: 0,
        }
    }

    // --- Reading -------------------------------------------------------------

    /// Bumped on every mutation, and on nothing else.
    ///
    /// This counter is not incidental. It is how the host avoids a per-frame
    /// item-definition lookup: the held weapon is only re-pushed to the player
    /// when the revision or the selected slot changed, so the steady-state cost
    /// of holding a sword is two integer compares a frame instead of a def deref
    /// and a struct copy. See [`HeldWeaponSync`].
    #[inline]
    pub fn revision(&self) -> u32 {
        self.revision
    }

    /// Selected hotbar slot, `0..HOTBAR`.
    #[inline]
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Item code in the selected slot, or `None` when it is empty.
    ///
    /// The TypeScript returned `-1` here while the registry's "nothing" was
    /// `0xffff`; the head of `registry.rs` has the story of what that cost.
    #[inline]
    pub fn held(&self) -> Option<ItemCode> {
        self.stack_at(self.selected).map(|(code, _)| code)
    }

    /// Stack size in the selected slot.
    #[inline]
    pub fn held_count(&self) -> u16 {
        self.count[self.selected]
    }

    /// What is in a slot, or `None` when it is empty. The HUD's read path.
    #[inline]
    pub fn stack_at(&self, slot: usize) -> Option<(ItemCode, u16)> {
        match self.count.get(slot) {
            Some(&0) | None => None,
            Some(&n) => Some((self.item[slot], n)),
        }
    }

    /// Put a stack in a slot outright, ignoring stacking rules.
    ///
    /// For a SAVE LOADER and nothing else, which is why it is not `add`. `add`
    /// implements the game's placement policy — merge into an existing stack,
    /// otherwise take the first free slot, respecting per-item stack limits —
    /// and that policy is exactly wrong when restoring: the slots are already
    /// decided, and re-deriving them would silently rearrange the player's
    /// inventory every time they loaded.
    ///
    /// A count of 0 empties the slot. Out-of-range slots are ignored rather than
    /// panicking, because the caller is reading a file.
    pub fn put_at(&mut self, slot: usize, code: ItemCode, n: u16) {
        if slot >= SLOT_COUNT {
            return;
        }
        self.item[slot] = code;
        self.count[slot] = n;
        self.revision = self.revision.wrapping_add(1);
    }

    /// First slot holding `code`, or `None`.
    ///
    /// The TypeScript hand-rolled this loop specifically to avoid `Array#find`,
    /// because it is called per dig tick and per craft check and `find` on a
    /// typed array allocates a closure frame per call in every JS engine that
    /// does not inline it. That reason does not survive the port: a `Range`'s
    /// `find` monomorphises to the same loop with the same three instructions a
    /// slot, so the adaptor is written the way it reads.
    pub fn find(&self, code: ItemCode) -> Option<usize> {
        (0..SLOT_COUNT).find(|&i| self.count[i] != 0 && self.item[i] == code)
    }

    /// Total of `code` across every slot.
    pub fn count_of(&self, code: ItemCode) -> u32 {
        let mut total = 0u32;
        for i in 0..SLOT_COUNT {
            if self.count[i] != 0 && self.item[i] == code {
                total += self.count[i] as u32;
            }
        }
        total
    }

    /// Cheaper than `count_of(code) >= n` — stops as soon as the answer is known.
    pub fn has(&self, code: ItemCode, n: u32) -> bool {
        let mut total = 0u32;
        for i in 0..SLOT_COUNT {
            if self.count[i] == 0 || self.item[i] != code {
                continue;
            }
            total += self.count[i] as u32;
            if total >= n {
                return true;
            }
        }
        false
    }

    /// True when nothing more of `code` will fit anywhere.
    pub fn full(&self, code: ItemCode) -> bool {
        let cap = ITEM_STACK[code as usize];
        for i in 0..SLOT_COUNT {
            if self.count[i] == 0 {
                return false;
            }
            if self.item[i] == code && self.count[i] < cap {
                return false;
            }
        }
        true
    }

    // --- Selection -----------------------------------------------------------

    /// Point the hotbar at slot `i`. Out of range, or already there, is a no-op.
    ///
    /// Selection does NOT bump [`Inventory::revision`], and that is deliberate:
    /// the revision means "the CONTENTS moved", and the host's held-weapon guard
    /// compares the selected slot separately. Folding selection into the counter
    /// would make every scroll of the wheel look like a pickup to anything else
    /// watching it.
    pub fn select_slot(&mut self, i: usize) {
        if i >= HOTBAR || i == self.selected {
            return;
        }
        self.selected = i;
    }

    /// Step the hotbar selection, wrapping. Used by the scroll wheel.
    pub fn cycle(&mut self, delta: i32) {
        if delta == 0 {
            return;
        }
        let h = HOTBAR as i32;
        self.selected = (((self.selected as i32 + delta) % h + h) % h) as usize;
    }

    // --- Mutation ------------------------------------------------------------

    /// Insert `n` of `code`. Fills partial stacks of the same item first (left to
    /// right, so the hotbar tops up before the backpack does), then empty slots.
    ///
    /// Returns what did NOT fit — the caller decides whether that is dropped back
    /// into the world or lost. Never panics, never resizes.
    pub fn add(&mut self, code: ItemCode, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let cap = ITEM_STACK[code as usize] as u32;
        let mut left = n;

        for i in 0..SLOT_COUNT {
            if left == 0 {
                break;
            }
            if self.count[i] == 0 || self.item[i] != code {
                continue;
            }
            let room = cap.saturating_sub(self.count[i] as u32);
            if room == 0 {
                continue;
            }
            let put = room.min(left);
            self.count[i] += put as u16;
            left -= put;
        }
        for i in 0..SLOT_COUNT {
            if left == 0 {
                break;
            }
            if self.count[i] != 0 {
                continue;
            }
            let put = cap.min(left);
            self.item[i] = code;
            self.count[i] = put as u16;
            left -= put;
        }

        if left != n {
            self.revision += 1;
        }
        left
    }

    /// Remove up to `n` of `code`, taking from the smallest stacks first so the
    /// inventory consolidates instead of fragmenting. Returns how many were
    /// actually removed.
    ///
    /// Two passes and no sort: the first finds the slot with the fewest of `code`,
    /// the second drains it. [`SLOT_COUNT`] is 30, so this is cheaper than any
    /// bookkeeping that would avoid it.
    pub fn remove(&mut self, code: ItemCode, n: u32) -> u32 {
        let mut left = n;
        while left > 0 {
            let mut best = None;
            // Above any `u16`, so the first matching slot always wins the first
            // compare — the TypeScript's `0x10000` for the same reason.
            let mut best_count = 0x1_0000u32;
            for i in 0..SLOT_COUNT {
                if self.count[i] == 0 || self.item[i] != code {
                    continue;
                }
                if (self.count[i] as u32) < best_count {
                    best_count = self.count[i] as u32;
                    best = Some(i);
                }
            }
            let Some(best) = best else { break };
            let take = best_count.min(left);
            self.count[best] -= take as u16;
            left -= take;
        }
        if left != n {
            self.revision += 1;
        }
        n - left
    }

    /// Remove up to `n` from one specific slot. Returns how many came out.
    pub fn remove_at(&mut self, slot: usize, n: u32) -> u32 {
        let Some(&have) = self.count.get(slot) else {
            return 0;
        };
        if have == 0 || n == 0 {
            return 0;
        }
        let take = (have as u32).min(n);
        self.count[slot] = have - take as u16;
        self.revision += 1;
        take
    }

    /// Exchange two slots outright. Both may be empty; that is a no-op that costs
    /// nothing — but it still bumps the revision, because the TypeScript's did
    /// and the counter's only contract is "at least as often as something moved".
    ///
    /// An out-of-range slot is refused rather than clamped. The TypeScript read
    /// and wrote past the end of the typed arrays, which silently did nothing on
    /// the read and dropped the write; refusing says the same thing without
    /// depending on that.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a == b || a >= SLOT_COUNT || b >= SLOT_COUNT {
            return;
        }
        self.item.swap(a, b);
        self.count.swap(a, b);
        self.revision += 1;
    }

    /// Empty every slot and reselect slot 0.
    pub fn clear(&mut self) {
        self.item.fill(0);
        self.count.fill(0);
        self.selected = 0;
        self.revision += 1;
    }

    /// Spend `n` of the item with this authoring id, returning how many were
    /// taken. An id the registry does not have spends nothing.
    ///
    /// This is the body of [`AmmoSource::spend`], which is the whole of the
    /// inventory-to-player seam. The TypeScript wrote
    /// `player.setAmmoSource((id, n) => inv.remove(itemCodeOf(id), n))` — a
    /// closure over the inventory instance, stored on the player. Rust cannot
    /// store that without first deciding who OWNS the inventory, and every
    /// answer to that question is an `Rc<RefCell<_>>` or an `Arc<Mutex<_>>`
    /// wrapped around a thing the host already holds perfectly well.
    ///
    /// So nothing is stored: the host lends its inventory to
    /// [`Player::step`](crate::entities::player::Player::step) through a
    /// [`Loadout`](crate::entities::player::Loadout), for exactly as long as the
    /// step runs. The seam is the impl below and this method.
    pub fn spend_by_id(&mut self, id: &str, n: u32) -> u32 {
        match item_code_of(id) {
            Some(code) => self.remove(code, n),
            None => 0,
        }
    }
}

/// The pack a bow spends from.
///
/// On ITS side of the boundary, exactly as the `From<&ItemWeapon>` adapter that
/// builds a [`PlayerWeapon`] is: `entities::player` states what it needs and
/// never imports the item registry, and `items` — which already knows both — is
/// what joins them.
impl AmmoSource for Inventory {
    fn spend(&mut self, id: &str, n: u32) -> u32 {
        self.spend_by_id(id, n)
    }
}

/// The revision guard: pushes the held item's weapon stats to the player when —
/// and only when — the selection actually changed.
///
/// `Player` never imports the item registry: `set_weapon` takes a plain shape, so
/// a weapon reaches it through the adapter in `registry.rs` and the physics layer
/// stays ignorant of items. That leaves someone having to NOTICE a change, and
/// [`Inventory`] already bumps its revision on every mutation, so the guard is
/// two integer compares per frame rather than a def lookup. Swapping slots
/// mid-swing is therefore also free of a stale-stat window: the next frame
/// re-reads it.
///
/// This was `Game.syncHeldWeapon` plus two fields on `Game`. It is here because
/// the fields and the method are one mechanism, and because the mechanism is the
/// entire reason `revision` exists — leaving it to the host to reinvent would
/// leave the counter looking like bookkeeping nobody reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeldWeaponSync {
    /// `None` forces a push on the first poll of a new run, which is what
    /// `heldRevision = -1` meant.
    last: Option<(u32, usize)>,
}

impl HeldWeaponSync {
    /// A guard that will push on its first poll.
    pub const fn new() -> HeldWeaponSync {
        HeldWeaponSync { last: None }
    }

    /// Forget what was last pushed, so the next poll pushes again. Call it when
    /// the player is rebuilt for a new run.
    pub fn reset(&mut self) {
        self.last = None;
    }

    /// `Some(w)` when the player should be told; the inner `Option` is the
    /// argument to `Player::set_weapon`, so `Some(None)` means "bare hands now".
    /// `None` means nothing changed and the caller does nothing.
    pub fn poll(&mut self, inv: &Inventory) -> Option<Option<PlayerWeapon>> {
        let now = (inv.revision(), inv.selected());
        if self.last == Some(now) {
            return None;
        }
        self.last = Some(now);
        Some(inv.held().and_then(player_weapon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::registry::item;

    /// The cap for an item known to stack, so the tests below read as intent
    /// rather than as a literal that content can move under them.
    fn cap(code: ItemCode) -> u32 {
        ITEM_STACK[code as usize] as u32
    }

    #[test]
    fn a_stack_fills_to_its_cap_before_overflowing_into_a_new_slot() {
        let code = item::STONE_CHUNK;
        let c = cap(code);
        assert!(c > 1, "this test needs a stackable item");

        let mut inv = Inventory::new();
        assert_eq!(inv.add(code, c + 1), 0, "two slots must hold cap + 1");
        assert_eq!(inv.stack_at(0), Some((code, c as u16)));
        assert_eq!(inv.stack_at(1), Some((code, 1)));
        assert_eq!(inv.count_of(code), c + 1);
    }

    #[test]
    fn a_partial_stack_tops_up_before_an_empty_slot_is_opened() {
        let code = item::STONE_CHUNK;
        let c = cap(code);
        let mut inv = Inventory::new();
        inv.add(code, c - 1);
        inv.add(code, 2);
        assert_eq!(inv.stack_at(0), Some((code, c as u16)));
        assert_eq!(inv.stack_at(1), Some((code, 1)));
    }

    #[test]
    fn an_unstackable_item_takes_one_slot_each() {
        let code = item::PICK_IRON;
        assert_eq!(cap(code), 1, "picks are meant to be stack=1");
        let mut inv = Inventory::new();
        assert_eq!(inv.add(code, 3), 0);
        for slot in 0..3 {
            assert_eq!(inv.stack_at(slot), Some((code, 1)));
        }
        assert_eq!(inv.stack_at(3), None);
    }

    #[test]
    fn a_full_inventory_hands_back_the_overflow() {
        let code = item::STONE_CHUNK;
        let c = cap(code);
        let total = c * SLOT_COUNT as u32;
        let mut inv = Inventory::new();
        assert_eq!(inv.add(code, total), 0);
        assert!(inv.full(code));
        assert_eq!(inv.add(code, 7), 7, "nothing fits, all seven come back");
        assert_eq!(inv.count_of(code), total);
    }

    #[test]
    fn removal_drains_the_smallest_stack_first() {
        let code = item::STONE_CHUNK;
        let c = cap(code) as u16;
        let mut inv = Inventory::new();
        inv.add(code, c as u32 + 3); // slot 0 = cap, slot 1 = 3
        assert_eq!(inv.remove(code, 2), 2);
        assert_eq!(inv.stack_at(0), Some((code, c)));
        assert_eq!(inv.stack_at(1), Some((code, 1)));
        // Drained past the small stack, the big one starts paying.
        assert_eq!(inv.remove(code, 5), 5);
        assert_eq!(inv.stack_at(0), Some((code, c - 4)));
        assert_eq!(inv.stack_at(1), None);
    }

    #[test]
    fn removal_stops_at_what_is_actually_there() {
        let code = item::COAL;
        let mut inv = Inventory::new();
        inv.add(code, 3);
        assert_eq!(inv.remove(code, 10), 3);
        assert_eq!(inv.count_of(code), 0);
    }

    /// The counter is load-bearing for the held-weapon guard, so "bumps on every
    /// mutation" and "bumps on nothing else" are both requirements.
    #[test]
    fn the_revision_bumps_on_every_mutation_and_only_then() {
        let code = item::STONE_CHUNK;
        let mut inv = Inventory::new();
        let mut r = inv.revision();

        let mut moved = |inv: &Inventory, what: &str| {
            assert_eq!(inv.revision(), r + 1, "{what} must bump the revision");
            r = inv.revision();
        };
        // Every mutating call, each exactly one bump.
        inv.add(code, 5);
        moved(&inv, "add");
        inv.remove(code, 1);
        moved(&inv, "remove");
        inv.remove_at(0, 1);
        moved(&inv, "remove_at");
        inv.swap(0, 1);
        moved(&inv, "swap");
        inv.clear();
        moved(&inv, "clear");

        // And every call that did not move anything, or that is not a mutation.
        inv.add(code, 0);
        inv.remove(code, 4); // nothing of `code` is left to take
        inv.remove_at(7, 3); // empty slot
        inv.remove_at(999, 3); // out of range
        inv.swap(2, 2); // same slot
        inv.swap(0, SLOT_COUNT); // out of range
        inv.select_slot(4);
        inv.cycle(3);
        inv.cycle(0);
        let _ = inv.held();
        let _ = inv.count_of(code);
        let _ = inv.has(code, 1);
        let _ = inv.full(code);
        let _ = inv.find(code);
        assert_eq!(inv.revision(), r, "nothing above moved a single item");

        // A full-but-refused add is not a mutation either.
        inv.add(code, cap(code) * SLOT_COUNT as u32);
        r = inv.revision();
        assert_eq!(inv.add(code, 1), 1);
        assert_eq!(inv.revision(), r, "a refused add moved nothing");
    }

    #[test]
    fn selection_wraps_in_both_directions_and_stays_in_the_hotbar() {
        let mut inv = Inventory::new();
        inv.cycle(-1);
        assert_eq!(inv.selected(), HOTBAR - 1);
        inv.cycle(1);
        assert_eq!(inv.selected(), 0);
        inv.cycle(HOTBAR as i32 * 3 + 2);
        assert_eq!(inv.selected(), 2);
        inv.select_slot(HOTBAR); // backpack is not selectable
        assert_eq!(inv.selected(), 2);
        inv.select_slot(9);
        assert_eq!(inv.selected(), 9);
    }

    #[test]
    fn held_is_none_for_an_empty_slot_and_never_a_sentinel() {
        let mut inv = Inventory::new();
        assert_eq!(inv.held(), None);
        assert_eq!(inv.held_count(), 0);
        // Item code 0 is a real item; holding it must not read as empty.
        inv.add(item::STONE_CHUNK, 1);
        assert_eq!(inv.held(), Some(0));
        assert_eq!(inv.held_count(), 1);
    }

    #[test]
    fn spend_by_id_is_the_ammo_source_body() {
        let mut inv = Inventory::new();
        inv.add(item::ARROW, 5);
        assert_eq!(inv.spend_by_id("arrow", 2), 2);
        assert_eq!(inv.count_of(item::ARROW), 3);
        assert_eq!(inv.spend_by_id("no_such_item", 2), 0);
    }

    #[test]
    fn the_held_weapon_guard_pushes_once_per_change() {
        let mut inv = Inventory::new();
        let mut sync = HeldWeaponSync::new();

        // First poll always pushes, even with nothing held.
        assert_eq!(sync.poll(&inv), Some(None));
        assert_eq!(sync.poll(&inv), None, "steady state costs nothing");

        inv.add(item::SWORD_TRAVELER, 1);
        let pushed = sync.poll(&inv).expect("a pickup changed the revision");
        assert_eq!(pushed, player_weapon(item::SWORD_TRAVELER));
        assert_eq!(sync.poll(&inv), None);

        // Moving the selection off the sword re-pushes bare hands, without the
        // revision having moved at all.
        let before = inv.revision();
        inv.select_slot(1);
        assert_eq!(inv.revision(), before);
        assert_eq!(sync.poll(&inv), Some(None));
        assert_eq!(sync.poll(&inv), None);

        sync.reset();
        assert!(sync.poll(&inv).is_some(), "reset forces one more push");
    }
}

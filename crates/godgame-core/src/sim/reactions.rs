//! Neighbour reactions — the chemistry layer that sits above plain movement.
//!
//! A reaction fires when a cell touches a partner material and rewrites one or
//! both cells (e.g. water hitting lava). The automata calls [`react`] before a
//! cell's movement rules; a true result means the cell changed and should not
//! also try to move this tick.
//!
//! ENGINE v2. Two things changed from the first cut:
//!
//!  1. A rule declares WHICH STATES may drive it (a bitmask) plus an optional
//!     per-tick probability. Previously the automata only ever called `react()`
//!     for Liquid cells, which hardcoded that policy into the caller and made
//!     powder/gas chemistry impossible. Now `react()` is offered every non-empty
//!     cell and the mask decides, so cross-state rules (sand+water) just work.
//!
//!  2. The Map is gone. Rules live in flat arrays keyed by
//!     `self * MAT_COUNT + other`, and `mat_reactive` (derived below) answers
//!     "can this material ever drive a reaction from its own state?" in ONE
//!     array read — so inert matter, which is nearly every cell, early-outs
//!     before the neighbour scan even starts.
//!
//! # What changed in the port
//!
//! The four typed arrays were module-level `const`s built by top-level
//! `addReaction` calls at import time. Rust has no import-time side effects, so
//! they are built once inside a [`LazyLock`] — [`RULES`] — the first time the
//! automata touches them. The table contents, the registration order and the
//! derived `mat_reactive` / `mat_passive` vectors are identical.
//!
//! `RX_PROB` stays `f32`, not `f64`. It is compared against an `f64` draw from
//! the RNG, and widening an `f32` to `f64` is exact — but the stored value must
//! be the `f32` the TypeScript's `Float32Array` held, or a probability like 0.2
//! rounds differently and a rule fires on a different tick.

use std::sync::LazyLock;

use super::grid::CellGrid;
use super::materials::{CellId, MAT_COUNT, MAT_STATE, MaterialState, block};
use super::rng::SimRng;

// Codes are resolved at compile time; the registry is fixed. (The TypeScript
// called `codeOf("water")` at module load for the same reason — it just had no
// way to do it any earlier.)
const WATER: CellId = block::WATER;
const LAVA: CellId = block::LAVA;
const STEAM: CellId = block::STEAM;
const OBSIDIAN: CellId = block::OBSIDIAN;
const SAND: CellId = block::SAND;
const GLASS: CellId = block::GLASS;
const SNOW: CellId = block::SNOW;
const ACID: CellId = block::ACID;
const STONE: CellId = block::STONE;
const DIRT: CellId = block::DIRT;
const SANDSTONE: CellId = block::SANDSTONE;
const ASH: CellId = block::ASH;
const EMBER: CellId = block::EMBER;
const WET_SAND: CellId = block::WET_SAND;
/// The "sticky" material is displayed as Mud.
///
/// Note this is `block::STICKY` (code 10) and NOT `block::MUD` (code 33) — the
/// registry has both, and the TypeScript's `codeOf("sticky")` resolved to the
/// former. Getting this wrong swaps one soil for another silently.
const MUD: CellId = block::STICKY;
const AIR: CellId = 0;

/// "Leave this side of the pair untouched."
pub const KEEP: i16 = -1;

// --- State masks -------------------------------------------------------------
// A rule's mask says which states are allowed to DRIVE it, tested against the
// state of the cell currently being stepped. Bit n = MaterialState n.

pub const ST_SOLID: u8 = 1 << (MaterialState::Solid as u8);
pub const ST_POWDER: u8 = 1 << (MaterialState::Powder as u8);
pub const ST_LIQUID: u8 = 1 << (MaterialState::Liquid as u8);
pub const ST_GAS: u8 = 1 << (MaterialState::Gas as u8);
pub const ST_ANY: u8 = ST_SOLID | ST_POWDER | ST_LIQUID | ST_GAS;

// --- Rule table --------------------------------------------------------------
// Structure-of-arrays indexed by `self * K + other`. K is the material count so
// the two codes pack into one index without collision, and the table grows
// automatically when materials are added.

/// Stride of the rule table: the material count.
const K: usize = MAT_COUNT;
const ENTRIES: usize = K * K;

/// What a matched rule does to one side of the pair.
///
/// The table stores this as a raw `i16` with `KEEP == -1` (a `Vec<Outcome>`
/// would be twice the width for a table of `MAT_COUNT^2` entries, and this one
/// is walked by the hot loop); [`Outcome::decode`] is the one place that knows
/// the sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// This side of the pair is left exactly as it was.
    Keep,
    /// This side is rewritten to the given material.
    Becomes(CellId),
}

impl Outcome {
    /// Read a stored table slot. Anything negative is [`Outcome::Keep`].
    #[inline]
    pub fn decode(code: i16) -> Outcome {
        if code == KEEP {
            Outcome::Keep
        } else {
            Outcome::Becomes(code as CellId)
        }
    }
}

/// The whole chemistry table, built once.
///
/// Field names are the TypeScript's array names so the two files read the same
/// way side by side.
pub struct Rules {
    /// Driving states allowed, per (self,other) pair. 0 = no rule registered.
    pub rx_mask: Vec<u8>,
    /// What the driving cell becomes (`KEEP` = unchanged).
    pub rx_self: Vec<i16>,
    /// What the neighbour becomes (`KEEP` = unchanged).
    pub rx_other: Vec<i16>,
    /// Per-tick probability; 1 means "always, don't even draw a random".
    pub rx_prob: Vec<f32>,
    /// 1 when this material appears as the DRIVING side of at least one rule
    /// whose mask includes the material's own state. This is the automata's
    /// early-out: stone is a participant (acid eats it) but never a driver, so
    /// stone stays as free as it was before `react()` was opened up to every
    /// cell.
    pub mat_reactive: Vec<u8>,
    /// 1 when this material can be the PASSIVE side of a rule that is certain
    /// to fire (probability 1) — i.e. some neighbour may rewrite it without it
    /// ever driving anything itself. Snow is the example: it never drives, but
    /// lava turns it to water on contact.
    ///
    /// The heat field consults this so a threshold transition cannot steal a
    /// cell out from under a contact rule (see [`reaction_pending`]). Only
    /// certain rules count; a stochastic rule shouldn't be able to block heat
    /// indefinitely.
    pub mat_passive: Vec<u8>,
}

impl Rules {
    /// Register a rule for the unordered pair {a,b}. Both orderings are stored
    /// so a cell reacts regardless of which side it sits on; `mask` and `prob`
    /// apply to whichever side is doing the driving.
    fn add(&mut self, a: CellId, b: CellId, a_becomes: i16, b_becomes: i16, mask: u8, prob: f32) {
        let ab = a as usize * K + b as usize;
        self.rx_mask[ab] = mask;
        self.rx_self[ab] = a_becomes;
        self.rx_other[ab] = b_becomes;
        self.rx_prob[ab] = prob;

        let ba = b as usize * K + a as usize;
        self.rx_mask[ba] = mask;
        self.rx_self[ba] = b_becomes;
        self.rx_other[ba] = a_becomes;
        self.rx_prob[ba] = prob;
    }
}

/// The chemistry table. Built on first touch, then read-only forever.
pub static RULES: LazyLock<Rules> = LazyLock::new(build_rules);

fn build_rules() -> Rules {
    let mut r = Rules {
        rx_mask: vec![0; ENTRIES],
        rx_self: vec![0; ENTRIES],
        rx_other: vec![0; ENTRIES],
        rx_prob: vec![0.0; ENTRIES],
        mat_reactive: vec![0; K],
        mat_passive: vec![0; K],
    };

    // --- The rules -----------------------------------------------------------
    // The original seven are declared with ST_LIQUID, which reproduces the old
    // "react() runs for Liquid cells only" policy EXACTLY: lava (liquid) still
    // drives lava+sand, sand (powder) still cannot, and so on.

    // Water quenches lava: water flashes to steam, lava freezes to obsidian.
    r.add(WATER, LAVA, STEAM as i16, OBSIDIAN as i16, ST_LIQUID, 1.0);
    // Lava fuses sand into glass on contact (lava keeps flowing).
    r.add(LAVA, SAND, LAVA as i16, GLASS as i16, ST_LIQUID, 1.0);
    // Lava melts snow to water; water dissolves snow it touches.
    r.add(LAVA, SNOW, LAVA as i16, WATER as i16, ST_LIQUID, 1.0);
    r.add(WATER, SNOW, WATER as i16, WATER as i16, ST_LIQUID, 1.0);
    // Acid etches through terrain it pools against (acid persists as it digs).
    r.add(ACID, STONE, ACID as i16, AIR as i16, ST_LIQUID, 1.0);
    r.add(ACID, DIRT, ACID as i16, AIR as i16, ST_LIQUID, 1.0);
    r.add(ACID, SANDSTONE, ACID as i16, AIR as i16, ST_LIQUID, 1.0);

    // --- Cross-state rules (only possible with the state mask) ---------------
    // Sand drinks the water it is dumped into and packs down into wet sand. The
    // water is NOT consumed, so a beach doesn't drain the sea it sits in.
    r.add(SAND, WATER, WET_SAND as i16, KEEP, ST_POWDER, 0.05);
    // Ash slakes into mud. Same reasoning: the water survives.
    r.add(ASH, WATER, MUD as i16, KEEP, ST_POWDER, 0.2);
    // A gas-driven rule: embers drifting into water hiss out as steam on contact.
    r.add(EMBER, WATER, STEAM as i16, KEEP, ST_GAS, 1.0);

    // --- Derived: which materials can ever drive a reaction ------------------
    for (a, &state) in MAT_STATE.iter().enumerate().skip(1) {
        let bit = 1u8 << state;
        let base = a * K;
        for b in 0..K {
            if r.rx_mask[base + b] & bit == 0 {
                continue;
            }
            r.mat_reactive[a] = 1;
            if r.rx_prob[base + b] >= 1.0 {
                r.mat_passive[b] = 1;
            }
        }
    }

    r
}

// The four orthogonal neighbours to probe (reactions are contact-only).
const NX: [i32; 4] = [0, 0, -1, 1];
const NY: [i32; 4] = [-1, 1, 0, 0];

/// Check the cell at (cx,cy) against its orthogonal neighbours and apply the
/// first matching reaction. Returns true if the cell reacted (so the caller
/// skips its movement this tick).
///
/// Callers gate on `mat_reactive` and already hold the driving cell's material,
/// so `self_id` is passed in rather than re-read. The window planes and
/// dimensions are pulled out of the grid ONCE per call instead of once per
/// neighbour: this runs for every water, lava, acid, sand and ash cell in an
/// awake region, every tick, and the four `grid.get` calls it used to make were
/// four bounds tests and eight property loads to fetch four bytes.
///
/// `rng` is threaded in because the stochastic rules draw from the ONE sim
/// stream, in sweep order — see [`super::rng`].
pub fn react(grid: &mut CellGrid, rng: &mut SimRng, cx: i32, cy: i32, self_id: CellId) -> bool {
    if self_id == 0 {
        return false;
    }

    let cols = grid.cols();
    let rows = grid.rows();

    let base = self_id as usize * K;
    let self_bit = 1u8 << MAT_STATE[self_id as usize];
    let rules = &*RULES;

    for n in 0..4 {
        let ox = cx + NX[n];
        let oy = cy + NY[n];
        if ox < 0 || oy < 0 || ox >= cols || oy >= rows {
            continue;
        }
        let other = grid.material[(oy * cols + ox) as usize];
        if other == 0 {
            continue;
        }

        let key = base + other as usize;
        // One read rejects both "no rule" (mask 0) and "wrong driving state".
        if rules.rx_mask[key] & self_bit == 0 {
            continue;
        }

        let p = rules.rx_prob[key];
        if p < 1.0 && !rng.chance(f64::from(p)) {
            continue;
        }

        let s = Outcome::decode(rules.rx_self[key]);
        let o = Outcome::decode(rules.rx_other[key]);
        if let Outcome::Becomes(id) = s {
            grid.set(cx, cy, id);
        }
        if let Outcome::Becomes(id) = o {
            grid.set(ox, oy, id);
        }
        return true;
    }
    false
}

/// True when a neighbour is standing by to drive a CERTAIN reaction against
/// this cell this tick.
///
/// The heat field calls this before melting anything, and it is what keeps the
/// two mechanisms from stepping on each other. Cells are stepped bottom-up, so
/// without it a snow cell sitting on lava would hit its (low) melting point and
/// turn to water a fraction of a tick before the lava below it was stepped —
/// and lava+water is a different rule, producing steam and obsidian instead of
/// the water that lava+snow promises. Deferring to the contact rule keeps the
/// original seven behaving exactly as they always did, while heat still melts
/// the same snow freely once nothing is touching it.
///
/// Callers gate on `mat_passive`, so this only ever runs for the handful of
/// materials that can actually be rewritten by a neighbour.
pub fn reaction_pending(grid: &CellGrid, cx: i32, cy: i32) -> bool {
    let self_id = grid.get(cx, cy);
    if self_id == 0 {
        return false;
    }

    let cols = grid.cols();
    let rows = grid.rows();
    let rules = &*RULES;

    for n in 0..4 {
        let ox = cx + NX[n];
        let oy = cy + NY[n];
        if ox < 0 || oy < 0 || ox >= cols || oy >= rows {
            continue;
        }
        let other = grid.material[(oy * cols + ox) as usize];
        if other == 0 {
            continue;
        }
        let key = other as usize * K + self_id as usize;
        if rules.rx_mask[key] & (1u8 << MAT_STATE[other as usize]) != 0 && rules.rx_prob[key] >= 1.0
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::grid::CellGrid;

    fn grid() -> CellGrid {
        CellGrid::new(16, 16)
    }

    #[test]
    fn every_rule_is_registered_in_both_orderings() {
        let r = &*RULES;
        // Water drives water+lava; lava drives lava+water. Both directions must
        // carry the mask, or the outcome depends on which cell the sweep
        // happened to reach first.
        let wl = WATER as usize * K + LAVA as usize;
        let lw = LAVA as usize * K + WATER as usize;
        assert_eq!(r.rx_mask[wl], ST_LIQUID);
        assert_eq!(r.rx_mask[lw], ST_LIQUID);
        assert_eq!(r.rx_self[wl], STEAM as i16);
        assert_eq!(r.rx_other[wl], OBSIDIAN as i16);
        // The mirrored entry swaps the two outcomes.
        assert_eq!(r.rx_self[lw], OBSIDIAN as i16);
        assert_eq!(r.rx_other[lw], STEAM as i16);
    }

    #[test]
    fn water_plus_lava_fires_from_either_side() {
        // Driven by the water.
        let mut g = grid();
        let mut rng = SimRng::new();
        g.set(4, 4, WATER);
        g.set(5, 4, LAVA);
        assert!(react(&mut g, &mut rng, 4, 4, WATER));
        assert_eq!(g.get(4, 4), STEAM);
        assert_eq!(g.get(5, 4), OBSIDIAN);

        // Driven by the lava — same products, mirrored.
        let mut g = grid();
        g.set(4, 4, WATER);
        g.set(5, 4, LAVA);
        assert!(react(&mut g, &mut rng, 5, 4, LAVA));
        assert_eq!(g.get(5, 4), OBSIDIAN);
        assert_eq!(g.get(4, 4), STEAM);
    }

    #[test]
    fn a_certain_reaction_draws_nothing_from_the_stream() {
        // Probability 1 must not consume a random, or every world downstream of
        // a lava lake shifts.
        let mut g = grid();
        let mut rng = SimRng::new();
        let before = rng.clone();
        g.set(4, 4, WATER);
        g.set(5, 4, LAVA);
        assert!(react(&mut g, &mut rng, 4, 4, WATER));
        assert_eq!(rng.next_u32(), before.clone().next_u32());
    }

    #[test]
    fn the_state_mask_gates_the_driver() {
        // Sand is a powder: it may drive sand+water, but NOT water+lava, and a
        // liquid may not drive the powder rule. This is the whole point of v2.
        let r = &*RULES;
        let sw = SAND as usize * K + WATER as usize;
        assert_eq!(r.rx_mask[sw], ST_POWDER);
        assert_eq!(r.rx_other[sw], KEEP, "the water must survive");
        assert_eq!(r.mat_reactive[SAND as usize], 1);
        // Stone participates (acid eats it) but never drives.
        assert_eq!(r.mat_reactive[STONE as usize], 0);
        assert_eq!(r.mat_passive[STONE as usize], 1);
        // Wet sand is only ever produced by a STOCHASTIC rule, so it must not
        // be marked passive — a 5% rule may not block the heat field forever.
        assert_eq!(r.mat_passive[WET_SAND as usize], 0);
    }

    #[test]
    fn keep_leaves_the_passive_side_alone() {
        let mut g = grid();
        let mut rng = SimRng::new();
        g.set(4, 4, SAND);
        g.set(4, 5, WATER);
        // 5% per tick; roll until it fires.
        let mut fired = false;
        for _ in 0..2000 {
            if react(&mut g, &mut rng, 4, 4, SAND) {
                fired = true;
                break;
            }
        }
        assert!(fired, "a 5% rule should fire within 2000 rolls");
        assert_eq!(g.get(4, 4), WET_SAND);
        assert_eq!(g.get(4, 5), WATER, "KEEP must leave the water alone");
    }

    #[test]
    fn reaction_pending_sees_only_certain_rules() {
        let mut g = grid();
        // Snow beside lava: lava+snow is certain, so the snow must defer.
        g.set(4, 4, SNOW);
        g.set(4, 5, LAVA);
        assert!(reaction_pending(&g, 4, 4));

        // Sand beside water: the rule is 5%, so nothing is "pending".
        let mut g = grid();
        g.set(4, 4, WATER);
        g.set(4, 5, SAND);
        assert!(!reaction_pending(&g, 4, 4));
    }

    #[test]
    fn a_cell_with_no_partner_does_not_react() {
        let mut g = grid();
        let mut rng = SimRng::new();
        g.set(4, 4, WATER);
        assert!(!react(&mut g, &mut rng, 4, 4, WATER));
        assert_eq!(g.get(4, 4), WATER);
    }
}

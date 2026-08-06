//! The mining and building tool — what a swing MEANS, with no engine in it.
//!
//! Ported from `src/interact/BuildTool.ts`. The class had two modes and they are
//! not variations on each other:
//!
//!  * **Survival** (the TypeScript default) — what you can do is decided
//!    entirely by the item in the selected hotbar slot. A pickaxe sets how hard
//!    a block you can break, how fast, how far away and how big a bite.
//!  * **Creative** — the infinite palette. A debug tool, and genuinely the
//!    fastest way to test terrain, reactions and worldgen.
//!
//! # What this module does NOT do: write cells
//!
//! It reads the grid to decide what a swing would produce and emits one
//! [`BrushAction`]. The caller hands that to
//! [`apply_brush`](crate::sim::edits::apply_brush). Keeping the decision and the
//! write apart is what made the TypeScript's single-writer rule expressible, and
//! it is still what keeps this file testable without a grid mutation in sight.
//!
//! # The one path
//!
//! In the TypeScript a `BrushAction` went one of two ways: `postMessage` to the
//! sim worker, or a direct `applyBrush` on the main-thread fallback. That
//! asymmetry is the whole reason `Game.ts` documented a one-tick latency window
//! (Game.ts:519-523) in which a dug cell was still there to be dug again.
//!
//! **There is no such window here.** One owned grid, one writer, one call:
//! `apply_brush` runs synchronously on the action this returns, so by the time
//! the next swing reads the grid the previous swing is already in it. Nothing in
//! this file compensates for a latency that no longer exists.
//!
//! # The radius-shrink trick
//!
//! [`apply_brush`](crate::sim::edits::apply_brush) takes a centre and a radius
//! and clears the whole disc; it has no per-cell veto. Emitting one r=0 edit per
//! breakable cell would be up to 49 edits per dig tick. So the tool SHRINKS the
//! radius until the disc contains nothing it is not allowed to break. The felt
//! behaviour is right — obsidian never breaks, and chipping next to it narrows
//! your bite to what fits around it — and it costs one edit either way.
//!
//! That reasoning was written when an edit was a worker message. It survives the
//! port for the second reason rather than the first: one edit per disc is also
//! one dirty-chunk pass per disc, and the shrink is what makes an unbreakable
//! block feel unbreakable instead of merely unchanged.

use crate::config::{BRUSH_MAX, BRUSH_MIN, BRUSH_RADIUS, CELL_SIZE};
use crate::sim::coords::WorldCell;
use crate::sim::edits::EditMode;
use crate::sim::grid::CellGrid;
use crate::sim::materials::{CellId, EMPTY, MAT_HARDNESS, block, mat_by_code};

/// Seconds between dig ticks at `dig_speed` 1.
///
/// One algorithm's cadence, so it lives here and not in `config`. Combined with
/// a tool's `brush_max` it is what spreads a dig from ~31 cells/s at the bottom
/// of the ladder to ~920 at the top — the 30x spread that makes a late-game
/// tunnel feel like a different verb from an early-game one.
const DIG_INTERVAL: f32 = 0.16;

/// Placing is not tool-gated, so it runs at a flat cadence.
const PLACE_INTERVAL: f32 = 0.1;

/// The mode a fresh tool starts in.
///
/// The TypeScript started in survival, because it had an inventory to start
/// with. Until the items milestone lands one there is nothing to place and bare
/// hands only clear soft cover, so a fresh tool starts creative and the game is
/// playable. Flip this to `false` the moment an inventory exists: the survival
/// path below is fully ported and is exactly what `false` selects.
const START_CREATIVE: bool = true;

/// A dig/place command for one frame.
///
/// `cx`,`cy` are ABSOLUTE world cells, which is what `apply_brush` wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrushAction {
    /// Dig clears the disc; place fills only its empty cells.
    pub mode: EditMode,
    /// Disc centre, world cell x.
    pub cx: i32,
    /// Disc centre, world cell y.
    pub cy: i32,
    /// Disc radius in cells. 0 is a single cell.
    pub r: i32,
    /// What to fill with. Ignored when digging.
    pub mat: CellId,
}

/// Where the pointer is and which of the two verbs it is asking for.
///
/// The TypeScript passed `{x, y, left, right}` in CANVAS space and added the
/// camera inside the tool. Here the position is already absolute world px: the
/// tool has no camera and should not learn about one, and un-projecting a
/// pointer is the host's job because only the host knows what a pointer is.
///
/// `dig`/`place` rather than `left`/`right` for the same reason — which physical
/// button means which verb is a binding, and bindings live with the host.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cursor {
    /// World px, +x right.
    pub x: f32,
    /// World px, +y DOWN — the sim's convention.
    pub y: f32,
    /// The dig verb is being held.
    pub dig: bool,
    /// The place verb is being held.
    pub place: bool,
}

/// The four numbers a held item contributes to digging.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToolProfile {
    /// Hardest material this tool can break. See `MAT_HARDNESS`.
    pub dig_power: f32,
    /// Multiplier on the dig cadence — 2 digs twice as often.
    pub dig_speed: f32,
    /// How far from the actor a swing may land, in cells, centre to centre.
    pub reach: f32,
    /// Largest disc radius this tool may cut.
    pub brush_max: i32,
}

/// What you get with nothing useful in hand.
///
/// `dig_power` 0.5 is calibrated against `content/blocks/`: it clears leaves
/// (0.2), moss/ash/snow (0.3), cactus (0.4) and sand (0.5), and stops at glass
/// (0.6) and dirt (1.0). So bare hands can strip a tree and scoop a beach but
/// cannot open the ground — exactly the amount of "go and find your pick" the
/// first minute of a run should have.
pub const HANDS: ToolProfile = ToolProfile {
    dig_power: 0.5,
    dig_speed: 0.6,
    reach: 4.0,
    brush_max: 1,
};

/// Creative ignores every gate; the numbers only have to be past every block.
pub const CREATIVE: ToolProfile = ToolProfile {
    dig_power: f32::INFINITY,
    dig_speed: 2.0,
    reach: f32::INFINITY,
    brush_max: BRUSH_MAX,
};

/// One family of placeable materials, behind one digit key.
pub struct PaletteGroup {
    /// What the HUD calls the family.
    pub name: &'static str,
    /// Members, in the order repeating the key steps through them.
    pub members: &'static [CellId],
}

/// Placeable palette for creative mode, as EIGHT GROUPS in HUD/number-key order.
///
/// Why groups and not a flat list: the registry has outgrown eight entries, but
/// the only palette input is `select_index(0..7)` from the eight digit keys.
/// Each digit owns a family and REPEATING the press steps through it — one key,
/// one shelf. Each group's cursor is remembered, so leaving and coming back
/// gives you the material you left on.
///
/// INVARIANT: exactly [`PALETTE_SLOTS`] groups.
///
/// The members are the generated `block::` constants rather than the
/// `matById("sand")` lookups the TypeScript did. Same table, resolved at compile
/// time, and a palette naming a block that content deleted stops being a silent
/// air slot and becomes a build error.
pub static PALETTE: [PaletteGroup; PALETTE_SLOTS] = [
    PaletteGroup {
        name: "Powder",
        members: &[block::SAND, block::WET_SAND, block::GRAVEL, block::ASH],
    },
    PaletteGroup {
        name: "Liquid",
        members: &[block::WATER, block::OIL, block::LAVA, block::ACID],
    },
    PaletteGroup {
        name: "Rock",
        members: &[
            block::STONE,
            block::SANDSTONE,
            block::BASALT,
            block::OBSIDIAN,
            block::GLASS,
            block::CRYSTAL,
        ],
    },
    PaletteGroup {
        name: "Soil",
        members: &[
            block::DIRT,
            block::CLAY,
            block::MUD,
            block::STICKY,
            block::PERMAFROST,
        ],
    },
    PaletteGroup {
        name: "Growth",
        members: &[block::WOOD, block::MOSS, block::VINE],
    },
    PaletteGroup {
        name: "Heat",
        members: &[block::FIRE, block::EMBER, block::SMOKE, block::STEAM],
    },
    PaletteGroup {
        name: "Cold",
        members: &[block::SNOW, block::ICE, block::PACKED_ICE],
    },
    PaletteGroup {
        name: "Gadget",
        members: &[
            block::BOUNCE,
            block::SPIKE,
            block::CONVEYOR_RIGHT,
            block::CONVEYOR_LEFT,
        ],
    },
];

/// How many palette groups there are, i.e. how many digit keys the host scans.
pub const PALETTE_SLOTS: usize = 8;

/// The dig/place tool: cursor state, palette, cadence, and the swing decision.
pub struct BuildTool {
    /// Debug/creative: infinite palette, no reach limit, no drops, no cost.
    pub creative: bool,
    /// Remembered member within each palette group.
    cursors: [usize; PALETTE_SLOTS],
    /// Which group the digit keys are currently pointing at.
    group: usize,
    /// Creative brush radius in cells, clamped to `BRUSH_MIN..=BRUSH_MAX`.
    pub brush: i32,
    /// Last cursor position in cell space — what the preview is drawn around.
    pub cursor_cx: i32,
    /// Last cursor position in cell space — what the preview is drawn around.
    pub cursor_cy: i32,
    /// Is the cursor inside the held tool's reach?
    pub in_reach: bool,
    /// Block code under the cursor (0 = air).
    pub target_block: CellId,
    /// True when the cursor is on a block this tool cannot break. Dig feedback.
    pub target_too_hard: bool,
    /// The profile actually in force — [`HANDS`], [`CREATIVE`], or a held tool's.
    pub profile: ToolProfile,
    dig_timer: f32,
    place_timer: f32,
    /// Cells the last [`BuildTool::place_radius`] would fill, i.e. items it costs.
    place_cost: i32,
}

impl Default for BuildTool {
    fn default() -> BuildTool {
        BuildTool {
            creative: START_CREATIVE,
            cursors: [0; PALETTE_SLOTS],
            group: 0,
            brush: BRUSH_RADIUS,
            cursor_cx: 0,
            cursor_cy: 0,
            in_reach: false,
            target_block: EMPTY,
            target_too_hard: false,
            profile: if START_CREATIVE { CREATIVE } else { HANDS },
            dig_timer: 0.0,
            place_timer: 0.0,
            place_cost: 0,
        }
    }
}

impl BuildTool {
    /// A fresh tool. See [`START_CREATIVE`] for which mode that is and why.
    pub fn new() -> BuildTool {
        BuildTool::default()
    }

    // --- Creative palette ----------------------------------------------------

    /// The material the creative brush would place.
    pub fn selected(&self) -> CellId {
        PALETTE[self.group].members[self.cursors[self.group]]
    }

    /// Display name of the selected material, for the HUD.
    pub fn selected_name(&self) -> &'static str {
        mat_by_code(self.selected()).name
    }

    /// One material per digit key — what the creative HUD's eight slots show.
    pub fn slots(&self) -> [CellId; PALETTE_SLOTS] {
        std::array::from_fn(|i| PALETTE[i].members[self.cursors[i]])
    }

    /// Which digit key is lit.
    pub fn slot_index(&self) -> usize {
        self.group
    }

    /// How far into the lit group's shelf the cursor is.
    pub fn member_index(&self) -> usize {
        self.cursors[self.group]
    }

    /// How long the lit group's shelf is.
    pub fn member_count(&self) -> usize {
        PALETTE[self.group].members.len()
    }

    /// Digit key `i` in creative mode.
    ///
    /// Pressing the slot you are already on steps to the next material in that
    /// group; pressing a different slot jumps to it and restores whatever you
    /// last had selected there.
    pub fn select_index(&mut self, i: usize) {
        if i >= PALETTE_SLOTS {
            return;
        }
        if i == self.group {
            self.cursors[i] = (self.cursors[i] + 1) % PALETTE[i].members.len();
        } else {
            self.group = i;
        }
    }

    /// Step one material forward through the whole palette, crossing groups.
    pub fn next(&mut self) {
        let g = self.group;
        let c = self.cursors[g] + 1;
        if c < PALETTE[g].members.len() {
            self.cursors[g] = c;
        } else {
            self.cursors[g] = 0;
            self.group = (g + 1) % PALETTE_SLOTS;
            self.cursors[self.group] = 0;
        }
    }

    /// Step one material backward through the whole palette, crossing groups.
    pub fn prev(&mut self) {
        let g = self.group;
        if self.cursors[g] > 0 {
            self.cursors[g] -= 1;
        } else {
            self.group = (g + PALETTE_SLOTS - 1) % PALETTE_SLOTS;
            self.cursors[self.group] = PALETTE[self.group].members.len() - 1;
        }
    }

    /// Grow or shrink the creative brush, clamped to `BRUSH_MIN..=BRUSH_MAX`.
    ///
    /// The clamp is the whole contract: a radius outside it either does nothing
    /// visible (below `BRUSH_MIN`) or stamps a disc bigger than the scratch the
    /// place search is sized for.
    pub fn add_brush(&mut self, delta: i32) {
        self.brush = (self.brush + delta).clamp(BRUSH_MIN, BRUSH_MAX);
    }

    /// Flip between creative and survival, resetting both cadences.
    pub fn toggle_creative(&mut self) {
        self.creative = !self.creative;
        self.dig_timer = 0.0;
        self.place_timer = 0.0;
    }

    // --- The preview ---------------------------------------------------------
    //
    // The renderer draws it, but WHERE it goes is this file's answer: the host
    // must not re-derive a disc from a radius and get a different disc from the
    // one `apply_brush` will cut.

    /// Radius the next stroke would use, before any per-cell veto.
    ///
    /// Survival can still shrink below this — see [`BuildTool::dig_radius`] —
    /// but a preview that flickered with every cell the cursor crossed would be
    /// worse than one that shows the bite you asked for.
    pub fn preview_radius(&self) -> i32 {
        self.brush.min(self.profile.brush_max)
    }

    /// Visit every cell of the preview disc.
    ///
    /// The disc test is `apply_brush`'s test, not an approximation of it.
    pub fn preview_cells(&self, mut f: impl FnMut(i32, i32)) {
        let r = self.preview_radius();
        let r2 = r * r;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r2 {
                    f(self.cursor_cx + dx, self.cursor_cy + dy);
                }
            }
        }
    }

    /// Bounding box of the preview disc in world px, as `(x, y, w, h)` with the
    /// origin at its top-left and +y DOWN.
    pub fn preview_bounds(&self) -> (f32, f32, f32, f32) {
        let r = self.preview_radius();
        let side = ((2 * r + 1) * CELL_SIZE) as f32;
        (
            ((self.cursor_cx - r) * CELL_SIZE) as f32,
            ((self.cursor_cy - r) * CELL_SIZE) as f32,
            side,
            side,
        )
    }

    // --- The frame -----------------------------------------------------------

    /// Decide this frame's action.
    ///
    /// Place wins over dig so a placement isn't instantly undug when both verbs
    /// are held. Nothing is written to the grid here; the caller passes the
    /// result to [`apply_brush`](crate::sim::edits::apply_brush).
    ///
    /// `actor_x`/`actor_y` is the centre of whoever is swinging, in world px —
    /// the player once there is one, the camera focus until then. Reach is
    /// measured centre to centre in cells, which is what the player reads off
    /// the screen: a radius, not a bounding box.
    ///
    /// SEAM (items): the TypeScript resolved `profile` from the held item's
    /// `tool` block, taking `dig_power` / `dig_speed` / `reach` / `brush_max`
    /// off the item registry. There is no registry yet, so survival gets
    /// [`HANDS`]. When the registry lands, that resolution is the only new code
    /// this function needs — everything downstream already reads `profile`.
    pub fn update(
        &mut self,
        dt: f32,
        cursor: Cursor,
        grid: &CellGrid,
        actor_x: f32,
        actor_y: f32,
    ) -> Option<BrushAction> {
        let cx = cell_of(cursor.x);
        let cy = cell_of(cursor.y);
        self.cursor_cx = cx;
        self.cursor_cy = cy;

        self.dig_timer -= dt;
        self.place_timer -= dt;

        self.profile = if self.creative { CREATIVE } else { HANDS };

        let pcx = cell_of(actor_x);
        let pcy = cell_of(actor_y);
        let ddx = (cx - pcx) as f32;
        let ddy = (cy - pcy) as f32;
        self.in_reach =
            self.creative || ddx * ddx + ddy * ddy <= self.profile.reach * self.profile.reach;

        let cell = WorldCell::new(cx, cy);
        self.target_block = if grid.is_loaded_world(cell) {
            grid.get_world(cell)
        } else {
            EMPTY
        };
        self.target_too_hard = !self.creative
            && self.target_block != EMPTY
            && hardness(self.target_block) > self.profile.dig_power;

        if !self.in_reach {
            return None;
        }
        if cursor.place {
            return self.place(cx, cy);
        }
        if cursor.dig {
            return self.dig(grid, cx, cy);
        }
        None
    }

    // --- Digging -------------------------------------------------------------

    /// SEAM (items): the TypeScript rolled the drop table over the same disc
    /// here, BEFORE emitting the edit, which is why drops could be computed
    /// without the sim reporting anything back. That loop stayed on the main
    /// thread only because "the worker's bundle does not carry" the item
    /// registry — a constraint that died with the worker. When drops land they
    /// go in this function against `dig_radius`'s result, with no main/worker
    /// split to work around and no double-roll window to reason about.
    fn dig(&mut self, grid: &CellGrid, cx: i32, cy: i32) -> Option<BrushAction> {
        if self.dig_timer > 0.0 {
            return None;
        }

        if self.creative {
            self.dig_timer = DIG_INTERVAL * 0.5;
            return Some(BrushAction {
                mode: EditMode::Dig,
                cx,
                cy,
                r: self.brush,
                mat: EMPTY,
            });
        }

        // Nothing here this tool can break — no edit, and no cooldown either, so
        // the next frame re-asks rather than pausing on a wall.
        let r = self.dig_radius(grid, cx, cy, self.profile.dig_power, self.profile.brush_max)?;

        self.dig_timer = DIG_INTERVAL / self.profile.dig_speed;
        Some(BrushAction {
            mode: EditMode::Dig,
            cx,
            cy,
            r,
            mat: EMPTY,
        })
    }

    /// Largest radius <= `r_max` whose disc contains no cell this tool cannot
    /// break, or `None` when there is nothing breakable in it at all.
    ///
    /// The trick is that a cell at squared distance `d2` first enters the disc at
    /// radius `ceil(sqrt(d2))`, so the largest radius that EXCLUDES it is one
    /// less than that. Taking the minimum over every unbreakable cell gives the
    /// answer in a single pass over the bounding box, with no per-radius rescan.
    ///
    /// `ceil(sqrt(d2))` is [`isqrt_ceil`] rather than a float round trip: the
    /// TypeScript's `Math.ceil(Math.sqrt(d2))` is exact for the integers it sees,
    /// but only because doubles are wide enough to hide the question.
    pub fn dig_radius(
        &self,
        grid: &CellGrid,
        cx: i32,
        cy: i32,
        dig_power: f32,
        r_max: i32,
    ) -> Option<i32> {
        let mut allowed = r_max;
        let mut first_breakable = i32::MAX;
        let r2 = r_max * r_max;

        for dy in -r_max..=r_max {
            for dx in -r_max..=r_max {
                let d2 = dx * dx + dy * dy;
                if d2 > r2 {
                    continue;
                }
                let cell = WorldCell::new(cx + dx, cy + dy);
                if !grid.is_loaded_world(cell) {
                    continue;
                }
                let m = grid.get_world(cell);
                if m == EMPTY {
                    continue;
                }

                let need = isqrt_ceil(d2);
                if hardness(m) > dig_power {
                    allowed = allowed.min(need - 1);
                } else {
                    first_breakable = first_breakable.min(need);
                }
            }
        }

        if first_breakable > allowed {
            return None;
        }
        Some(allowed)
    }

    // --- Placing -------------------------------------------------------------

    /// SEAM (items): survival placing spends a stack out of the hotbar, so both
    /// WHAT it puts down and how many cells it may cover come from the
    /// inventory. There is no inventory yet, so survival place is a no-op rather
    /// than an invented interface. [`BuildTool::place_radius`] is the entire
    /// grid-side half of that decision and needs nothing from the registry; it
    /// is public and tested, waiting for a caller that can say how many items
    /// are available and which block they turn into.
    fn place(&mut self, cx: i32, cy: i32) -> Option<BrushAction> {
        if self.place_timer > 0.0 {
            return None;
        }

        if self.creative {
            self.place_timer = PLACE_INTERVAL * 0.5;
            return Some(BrushAction {
                mode: EditMode::Place,
                cx,
                cy,
                r: self.brush,
                mat: self.selected(),
            });
        }

        None
    }

    /// Largest radius whose disc holds no more empty cells than `avail`, or
    /// `None` when even the centre cell cannot be paid for or is already
    /// occupied.
    ///
    /// `apply_brush` fills exactly the loaded EMPTY cells of the disc, so
    /// counting them here makes the charge and the effect agree by construction
    /// — one item spent per cell placed, with no reconciliation pass.
    /// [`BuildTool::place_cost`] carries the count out.
    ///
    /// The TypeScript kept a preallocated `Int32Array` ring buffer so the search
    /// allocated nothing per frame. `BRUSH_MAX + 2` `i32`s is 88 bytes; here it
    /// is a stack array, which is the same trick with the ownership question
    /// deleted.
    pub fn place_radius(&mut self, grid: &CellGrid, cx: i32, cy: i32, avail: i32) -> Option<i32> {
        self.place_cost = 0;
        if avail <= 0 {
            return None;
        }
        let r_max = self.profile.brush_max.min(BRUSH_MAX);

        // Cells that first enter the disc at radius k.
        let mut ring = [0i32; BRUSH_MAX as usize + 2];
        let r2 = r_max * r_max;
        for dy in -r_max..=r_max {
            for dx in -r_max..=r_max {
                let d2 = dx * dx + dy * dy;
                if d2 > r2 {
                    continue;
                }
                let cell = WorldCell::new(cx + dx, cy + dy);
                if !grid.is_loaded_world(cell) || !grid.is_empty_world(cell) {
                    continue;
                }
                ring[isqrt_ceil(d2) as usize] += 1;
            }
        }

        let mut cum = 0;
        let mut best = None;
        for (r, &entering) in ring.iter().enumerate().take(r_max as usize + 1) {
            cum += entering;
            if cum > avail {
                break;
            }
            if cum > 0 {
                best = Some(r as i32);
                self.place_cost = cum;
            }
        }
        best
    }

    /// Cells the last [`BuildTool::place_radius`] would fill — what it costs.
    pub fn place_cost(&self) -> i32 {
        self.place_cost
    }
}

/// Cell a world-pixel coordinate falls in.
///
/// `config::cell_at` says the same thing, but this is the tool's own reading of
/// a cursor and it must floor negatives — the world extends left of zero and
/// truncation would fold cell -1 onto cell 0.
#[inline]
fn cell_of(px: f32) -> i32 {
    crate::config::cell_at(px)
}

/// A material's hardness, or 0 for a code the registry does not have.
#[inline]
fn hardness(id: CellId) -> f32 {
    MAT_HARDNESS.get(id as usize).copied().unwrap_or(0.0)
}

/// `ceil(sqrt(n))` for a non-negative integer, exactly, in integers.
#[inline]
fn isqrt_ceil(n: i32) -> i32 {
    let root = n.isqrt();
    if root * root < n { root + 1 } else { root }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::edits::apply_brush;

    fn grid() -> CellGrid {
        CellGrid::new(64, 64)
    }

    fn creative() -> BuildTool {
        let mut t = BuildTool::new();
        t.creative = true;
        t.profile = CREATIVE;
        t
    }

    fn survival() -> BuildTool {
        let mut t = BuildTool::new();
        t.creative = false;
        t.profile = HANDS;
        t
    }

    /// Cursor at the centre of cell (cx, cy).
    fn at(cx: i32, cy: i32, dig: bool, place: bool) -> Cursor {
        Cursor {
            x: (cx * CELL_SIZE + CELL_SIZE / 2) as f32,
            y: (cy * CELL_SIZE + CELL_SIZE / 2) as f32,
            dig,
            place,
        }
    }

    #[test]
    fn ceil_sqrt_is_exact_on_and_around_every_square() {
        assert_eq!(isqrt_ceil(0), 0);
        assert_eq!(isqrt_ceil(1), 1);
        for root in 2..=40 {
            let sq = root * root;
            assert_eq!(isqrt_ceil(sq), root, "{sq}");
            // One short of a square still needs the whole radius; one past it
            // needs the next. Those two are the only cases the disc test cares
            // about, and getting either backwards moves every shrink by one.
            assert_eq!(isqrt_ceil(sq - 1), root, "{}", sq - 1);
            assert_eq!(isqrt_ceil(sq + 1), root + 1, "{}", sq + 1);
        }
    }

    #[test]
    fn the_brush_clamps_to_the_configured_range() {
        let mut t = creative();
        t.add_brush(-1000);
        assert_eq!(t.brush, BRUSH_MIN);
        t.add_brush(1000);
        assert_eq!(t.brush, BRUSH_MAX);
        // And a step from a clamped edge moves by exactly one.
        t.add_brush(-1);
        assert_eq!(t.brush, BRUSH_MAX - 1);
    }

    #[test]
    fn a_fresh_brush_starts_at_the_configured_radius() {
        assert_eq!(BuildTool::new().brush, BRUSH_RADIUS);
        assert!((BRUSH_MIN..=BRUSH_MAX).contains(&BRUSH_RADIUS));
    }

    #[test]
    fn every_palette_group_is_real_and_non_empty() {
        assert_eq!(PALETTE.len(), PALETTE_SLOTS);
        for g in &PALETTE {
            assert!(!g.members.is_empty(), "{} is empty", g.name);
            for &m in g.members {
                assert_ne!(m, EMPTY, "{} holds air", g.name);
                assert_eq!(mat_by_code(m).code, m, "{} holds an unknown code", g.name);
            }
        }
    }

    #[test]
    fn a_digit_key_jumps_between_groups_and_repeats_within_one() {
        let mut t = creative();
        let powder0 = t.selected();
        t.select_index(0); // already on group 0 -> step within it
        assert_ne!(t.selected(), powder0);
        assert_eq!(t.slot_index(), 0);

        t.select_index(2); // a different group -> jump, cursor 0
        assert_eq!(t.slot_index(), 2);
        assert_eq!(t.member_index(), 0);

        t.select_index(0); // back -> the member we left on, not a reset
        assert_eq!(t.member_index(), 1);

        t.select_index(PALETTE_SLOTS); // out of range is ignored
        assert_eq!(t.slot_index(), 0);
    }

    #[test]
    fn next_and_prev_walk_the_whole_palette_and_wrap() {
        let mut t = creative();
        let total: usize = PALETTE.iter().map(|g| g.members.len()).sum();
        let first = t.selected();
        for _ in 0..total {
            t.next();
        }
        assert_eq!(t.selected(), first, "a full lap should come home");
        t.prev();
        assert_eq!(t.slot_index(), PALETTE_SLOTS - 1);
        assert_eq!(
            t.member_index(),
            PALETTE[PALETTE_SLOTS - 1].members.len() - 1
        );
    }

    #[test]
    fn creative_digs_a_disc_of_the_brush_radius() {
        let mut t = creative();
        t.brush = 3;
        let mut g = grid();
        for y in 20..40 {
            for x in 20..40 {
                g.set(x, y, block::STONE);
            }
        }

        let act = t
            .update(0.0, at(30, 30, true, false), &g, 0.0, 0.0)
            .expect("creative dig always swings");
        assert_eq!(act.mode, EditMode::Dig);
        assert_eq!((act.cx, act.cy, act.r), (30, 30, 3));

        apply_brush(&mut g, act.mode, act.cx, act.cy, act.r, act.mat);
        assert_eq!(g.get(30, 30), EMPTY);
        assert_eq!(g.get(33, 30), EMPTY);
        assert_eq!(g.get(34, 30), block::STONE, "outside the disc");
    }

    #[test]
    fn the_cadence_gates_the_next_swing() {
        let mut t = creative();
        let mut g = grid();
        g.set(30, 30, block::STONE);

        assert!(
            t.update(0.0, at(30, 30, true, false), &g, 0.0, 0.0)
                .is_some()
        );
        assert!(
            t.update(0.0, at(30, 30, true, false), &g, 0.0, 0.0)
                .is_none(),
            "a second swing in the same instant must be refused"
        );
        assert!(
            t.update(1.0, at(30, 30, true, false), &g, 0.0, 0.0)
                .is_some(),
            "a second later it swings again"
        );
    }

    #[test]
    fn place_wins_over_dig_when_both_are_held() {
        let mut t = creative();
        let g = grid();
        let act = t
            .update(0.0, at(30, 30, true, true), &g, 0.0, 0.0)
            .expect("both held still swings");
        assert_eq!(act.mode, EditMode::Place);
        assert_eq!(act.mat, t.selected());
    }

    #[test]
    fn the_cursor_cell_floors_negative_world_px() {
        let mut t = creative();
        let g = grid();
        t.update(0.0, at(-3, -2, false, false), &g, 0.0, 0.0);
        assert_eq!((t.cursor_cx, t.cursor_cy), (-3, -2));
    }

    #[test]
    fn survival_reach_refuses_a_far_cursor() {
        let mut t = survival();
        let mut g = grid();
        g.set(30, 30, block::SAND);
        // HANDS reach is 4 cells; the actor sits at cell 0,0.
        assert!(
            t.update(0.0, at(30, 30, true, false), &g, 0.0, 0.0)
                .is_none()
        );
        assert!(!t.in_reach);

        // Same swing, actor next to it.
        let near = (30 * CELL_SIZE) as f32;
        assert!(
            t.update(0.0, at(30, 30, true, false), &g, near, near)
                .is_some()
        );
        assert!(t.in_reach);
    }

    #[test]
    fn bare_hands_stop_at_the_hardness_they_are_calibrated_for() {
        let mut t = survival();
        let mut g = grid();
        g.set(30, 30, block::DIRT); // hardness 1.0, over HANDS' 0.5
        let here = (30 * CELL_SIZE) as f32;
        assert!(
            t.update(0.0, at(30, 30, true, false), &g, here, here)
                .is_none()
        );
        assert!(t.target_too_hard, "and the HUD is told why");

        g.set(30, 30, block::SAND); // hardness 0.5, exactly at the limit
        let act = t
            .update(0.0, at(30, 30, true, false), &g, here, here)
            .expect("sand is diggable by hand");
        assert!(!t.target_too_hard);
        assert_eq!(act.r, HANDS.brush_max);
    }

    #[test]
    fn the_dig_radius_shrinks_around_what_it_cannot_break() {
        let t = survival();
        let mut g = grid();
        for y in 26..36 {
            for x in 26..36 {
                g.set(x, y, block::SAND);
            }
        }
        // Unbreakable at distance 2: it first enters the disc at r=2, so the
        // largest radius that excludes it is 1.
        g.set(32, 30, block::OBSIDIAN);
        assert_eq!(t.dig_radius(&g, 30, 30, HANDS.dig_power, 4), Some(1));

        // Adjacent: only r=0 is left.
        g.set(31, 30, block::OBSIDIAN);
        assert_eq!(t.dig_radius(&g, 30, 30, HANDS.dig_power, 4), Some(0));

        // On the cursor itself: nothing breakable is reachable at all.
        g.set(30, 30, block::OBSIDIAN);
        assert_eq!(t.dig_radius(&g, 30, 30, HANDS.dig_power, 4), None);
    }

    #[test]
    fn creative_dig_power_is_past_even_obsidian() {
        let t = creative();
        let mut g = grid();
        g.set(30, 30, block::OBSIDIAN);
        assert_eq!(t.dig_radius(&g, 30, 30, CREATIVE.dig_power, 2), Some(2));
    }

    #[test]
    fn an_empty_disc_is_not_a_swing() {
        let t = survival();
        let g = grid();
        assert_eq!(t.dig_radius(&g, 30, 30, HANDS.dig_power, 4), None);
    }

    #[test]
    fn the_place_radius_buys_exactly_what_it_fills() {
        let mut t = creative();
        let mut g = grid();

        // One item buys the centre cell and nothing more.
        assert_eq!(t.place_radius(&g, 30, 30, 1), Some(0));
        assert_eq!(t.place_cost(), 1);

        // r=1 is the centre plus its four orthogonal neighbours.
        assert_eq!(t.place_radius(&g, 30, 30, 5), Some(1));
        assert_eq!(t.place_cost(), 5);
        // Four short of the next ring, so it stays at r=1.
        assert_eq!(t.place_radius(&g, 30, 30, 12), Some(1));
        assert_eq!(t.place_cost(), 5);

        // Whatever it charges is what the brush actually fills.
        let r = t.place_radius(&g, 30, 30, 40).unwrap();
        let cost = t.place_cost();
        apply_brush(&mut g, EditMode::Place, 30, 30, r, block::STONE);
        let filled = g.material.iter().filter(|&&m| m == block::STONE).count();
        assert_eq!(filled as i32, cost);

        // The search does NOT stop at what it just filled: `apply_brush` skips
        // occupied cells, so the next stroke buys the annulus outside the disc
        // and pays for exactly that.
        let r2 = t.place_radius(&g, 30, 30, 40).unwrap();
        assert!(r2 > r);
        let before = filled as i32;
        apply_brush(&mut g, EditMode::Place, 30, 30, r2, block::STONE);
        let now = g.material.iter().filter(|&&m| m == block::STONE).count();
        assert_eq!(now as i32 - before, t.place_cost());
    }

    #[test]
    fn a_full_grid_has_nothing_to_sell() {
        let mut t = creative();
        let mut g = grid();
        for y in 0..g.rows() {
            for x in 0..g.cols() {
                g.set(x, y, block::STONE);
            }
        }
        assert_eq!(t.place_radius(&g, 30, 30, 1000), None);
        assert_eq!(t.place_cost(), 0);
    }

    #[test]
    fn no_items_buys_nothing() {
        let mut t = creative();
        let g = grid();
        assert_eq!(t.place_radius(&g, 30, 30, 0), None);
        assert_eq!(t.place_radius(&g, 30, 30, -1), None);
    }

    #[test]
    fn the_preview_disc_is_the_disc_the_brush_cuts() {
        let mut t = creative();
        t.brush = 3;
        let g = grid();
        t.update(0.0, at(30, 30, false, false), &g, 0.0, 0.0);

        let mut preview = Vec::new();
        t.preview_cells(|x, y| preview.push((x, y)));

        let mut g2 = grid();
        let act = BrushAction {
            mode: EditMode::Place,
            cx: 30,
            cy: 30,
            r: t.preview_radius(),
            mat: block::STONE,
        };
        apply_brush(&mut g2, act.mode, act.cx, act.cy, act.r, act.mat);
        for (x, y) in &preview {
            assert_eq!(g2.get(*x, *y), block::STONE, "preview missed ({x},{y})");
        }
        assert_eq!(
            g2.material.iter().filter(|&&m| m == block::STONE).count(),
            preview.len()
        );

        let (bx, by, bw, bh) = t.preview_bounds();
        assert_eq!((bx, by), ((27 * CELL_SIZE) as f32, (27 * CELL_SIZE) as f32));
        assert_eq!((bw, bh), ((7 * CELL_SIZE) as f32, (7 * CELL_SIZE) as f32));
    }

    #[test]
    fn survival_place_is_a_no_op_until_there_is_an_inventory() {
        let mut t = survival();
        let g = grid();
        let here = (30 * CELL_SIZE) as f32;
        assert!(
            t.update(0.0, at(30, 30, false, true), &g, here, here)
                .is_none(),
            "see the SEAM note on place()"
        );
    }

    #[test]
    fn toggling_creative_clears_both_cadences() {
        let mut t = creative();
        let mut g = grid();
        g.set(30, 30, block::STONE);
        t.update(0.0, at(30, 30, true, false), &g, 0.0, 0.0);
        t.toggle_creative();
        t.toggle_creative();
        assert!(
            t.update(0.0, at(30, 30, true, false), &g, 0.0, 0.0)
                .is_some(),
            "a mode flip is a fresh swing"
        );
    }

    #[test]
    fn a_brush_outside_the_loaded_window_reads_as_air() {
        let mut t = creative();
        let g = grid();
        t.update(0.0, at(-500, -500, false, false), &g, -2500.0, -2500.0);
        assert_eq!(t.target_block, EMPTY);
        assert!(!t.target_too_hard);
    }
}

//! Dropped item stacks: the two seconds between "that block broke" and "that is
//! in my bag".
//!
//! WHY THEY EXIST AT ALL rather than crediting the inventory the instant a cell
//! breaks. Two reasons, both mechanical rather than cosmetic:
//!
//!  - A full inventory has somewhere to put the overflow. Instant crediting has
//!    to either silently delete the drop or refuse the dig; a physical stack just
//!    sits on the floor until you make room.
//!  - Drops fall. Mining a seam from below drops the ore onto the ledge under it,
//!    which is information — you can see what a swing produced before you decide
//!    whether the next swing is worth it.
//!
//! This lives in `items/` and is driven by the host. It is deliberately NOT an
//! `entities/` mob: it has no brain, no health and no combat, and reusing the mob
//! pool for it would drag all three into a thing that is really just a falling
//! square with a pickup radius.
//!
//! Storage is SoA over a fixed pool, matching the mob system: no allocation per
//! drop, and a dead slot is one byte to test.

use super::drops::DropBag;
use super::inventory::Inventory;
use super::registry::{ItemCode, item_code_for_drop, item_color};
use crate::config::{PLAYER_H, PLAYER_W, cell_at};
use crate::sim::coords::WorldCell;
use crate::sim::grid::CellGrid;
use crate::sim::materials::MAT_COLLIDE;

/// Pool size. A max-tier brush breaks 49 cells of at most a handful of materials,
/// so a run of frantic digging produces single-digit stacks per tick; 96 is
/// roughly twenty seconds of that with nobody picking anything up. Past the cap
/// the oldest stack is recycled rather than the new one dropped, so the thing you
/// just mined is always the thing that survives.
///
/// Public because [`DropStack::slot`] is, and a host that pools one drawable per
/// slot — as `yugen-render` does — needs to know how many slots there are.
/// That is the same reason [`MAX_MOBS`](crate::entities::mobs::MAX_MOBS) and
/// [`MAX_SHOTS`](crate::entities::MAX_SHOTS) are public.
pub const MAX_DROPS: usize = 96;

/// The pool size, under the name the rest of this file reads it by.
const CAP: usize = MAX_DROPS;

// Named DROP_GRAVITY, not GRAVITY: an unqualified `GRAVITY` here is one import
// away from silently shadowing the player's `GRAVITY` from `config::physics`,
// and the two are deliberately different numbers.
/// px/s^2 — noticeably lighter than the player's, so drops float down.
const DROP_GRAVITY: f32 = 420.0;
/// Horizontal damping per second, so pops settle instead of sliding.
const DRAG: f32 = 3.4;
/// px/s of initial scatter, enough to separate two stacks visually.
const POP_SPEED: f32 = 46.0;
/// Fraction of vertical speed kept on landing.
const BOUNCE: f32 = 0.28;

/// Seconds before a stack will home. Stops a drop being eaten by the swing that
/// made it.
const ARM_TIME: f32 = 0.28;
/// Distance (px) at which a stack starts flying toward the player.
const MAGNET: f32 = 48.0;
/// Distance (px) at which it is absorbed.
const GRAB: f32 = 9.0;
const MAGNET_ACCEL: f32 = 900.0;

/// Seconds a stack survives untouched. Long enough to come back for it.
const LIFETIME: f32 = 180.0;

/// Drawn edge in px for an item with NO icon — still the size of most of the
/// game's drops until every item has art.
///
/// An item WITH an icon is drawn at `CELL_SIZE` px per icon pixel instead, so a
/// 2x2 icon is 10x10 world px. That is a deliberate visual choice and it is also
/// the only honest scale available: sprite art in this game is authored at one
/// art pixel per world cell — see the `PLAYER_CELLS_W` note in `config/physics`,
/// which exists precisely because drawing a sprite at a finer grain than the
/// terrain makes it read as a decal pasted over the world rather than a thing in
/// it. A drop at 3px would put a 1.5px icon pixel against a 5px cell: the exact
/// mismatch that note is about.
///
/// Position stays at whole-PIXEL rounding rather than snapping to the cell
/// lattice. The invariant that matters is grain (one icon pixel spans one cell
/// edge), not phase; snapping to a 5px lattice would quantise the ±0.8px bob to
/// nothing, and a magnetised drop would jerk across the screen in 5px steps.
///
/// Published rather than private because the drawing it describes now lives in
/// `yugen-render`, and this is the number that decides how big a swatch is.
pub const DROP_SIZE_PX: f32 = 3.0;

/// One live stack, as a renderer or a HUD reads it.
///
/// A by-value view over the SoA arrays: the pool stays SoA for the update loop,
/// and nothing outside this file has to know that. The `bob` phase the
/// TypeScript's draw loop computed from `age` and the slot index is not baked in
/// — both inputs are here, and the wobble is a drawing decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropStack {
    /// Pool slot. Stable while the stack lives; the draw loop's bob is keyed on
    /// it so a floor full of drops does not pulse in unison.
    pub slot: usize,
    /// Centre in world px. A drop has no extent — see [`WorldItems::update`].
    pub x: f32,
    /// Centre in world px, +y DOWN.
    pub y: f32,
    /// Seconds since it was spawned.
    pub age: f32,
    pub code: ItemCode,
    pub count: u16,
    /// The item's colour, cached at spawn — see [`WorldItems`].
    pub color: [u8; 3],
}

/// The pool of dropped stacks.
///
/// # What the port dropped
///
/// `draw` was Canvas2D and did not come across; [`WorldItems::stacks`] publishes
/// what it read. Two per-slot caches went with it:
///
///  - `fill`, the colour pre-rendered to an `rgb(r,g,b)` string. That existed
///    because the draw loop was building a fresh string 96 times a frame for a
///    value that cannot change after `spawn`. There are no strings in this
///    crate's draw path because there is no draw path.
///  - `icon`, the baked `Sprite` resolved at spawn. A `Sprite` is a renderer
///    object; `registry::item_icon(code)` is the one array read that cache was
///    saving, and the code is right there in [`DropStack`].
///
/// The r/g/b cache STAYS. It replaced an `ITEM_DEFS[code].color` deref in the
/// draw loop and was measured 7.2x faster; it is the numeric source of record,
/// and it is still one indexed read per stack rather than a def deref per frame.
#[derive(Clone, Debug)]
pub struct WorldItems {
    x: [f32; CAP],
    y: [f32; CAP],
    vx: [f32; CAP],
    vy: [f32; CAP],
    code: [ItemCode; CAP],
    count: [u16; CAP],
    age: [f32; CAP],
    live: [bool; CAP],
    /// Cached colour per slot so drawing never dereferences a def.
    r: [u8; CAP],
    g: [u8; CAP],
    b: [u8; CAP],
    /// Round-robin cursor, so a full pool recycles oldest-first.
    next: usize,
    /// Live count, for the HUD and for an early-out in update.
    active: usize,
}

impl Default for WorldItems {
    fn default() -> WorldItems {
        WorldItems::new()
    }
}

impl WorldItems {
    /// An empty pool.
    pub const fn new() -> WorldItems {
        WorldItems {
            x: [0.0; CAP],
            y: [0.0; CAP],
            vx: [0.0; CAP],
            vy: [0.0; CAP],
            code: [0; CAP],
            count: [0; CAP],
            age: [0.0; CAP],
            live: [false; CAP],
            r: [0; CAP],
            g: [0; CAP],
            b: [0; CAP],
            next: 0,
            active: 0,
        }
    }

    /// How many stacks are on the floor.
    #[inline]
    pub fn active(&self) -> usize {
        self.active
    }

    /// Nothing is on the floor.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.active == 0
    }

    /// Pool capacity — what `active` is measured against.
    pub const fn capacity() -> usize {
        CAP
    }

    /// Every live stack. The renderer's whole read of this module.
    pub fn stacks(&self) -> impl Iterator<Item = DropStack> + '_ {
        (0..CAP).filter(|&s| self.live[s]).map(move |s| DropStack {
            slot: s,
            x: self.x[s],
            y: self.y[s],
            age: self.age[s],
            code: self.code[s],
            count: self.count[s],
            color: [self.r[s], self.g[s], self.b[s]],
        })
    }

    pub fn clear(&mut self) {
        self.live.fill(false);
        self.active = 0;
        self.next = 0;
    }

    /// Empty a dig tick's bag into the world at (wx,wy) in PIXELS. Each item kind
    /// becomes one stack; the scatter is derived from the slot index rather than
    /// from a random draw, so two ticks that produced the same yield look the
    /// same.
    pub fn spawn_bag(&mut self, bag: &mut DropBag, wx: f32, wy: f32) {
        for i in 0..bag.len() {
            self.spawn(bag.code_at(i), bag.count_at(i), wx, wy, i);
        }
        bag.clear();
    }

    /// Put one stack on the floor.
    ///
    /// `phase` fans overlapping kinds apart. The TypeScript defaulted it to 0 and
    /// every call site passed one anyway; Rust has no default arguments and
    /// inventing an overload for a parameter nobody omitted would be worse than
    /// spelling the 0.
    pub fn spawn(&mut self, code: ItemCode, n: u32, wx: f32, wy: f32, phase: usize) {
        if n == 0 {
            return;
        }
        let s = self.alloc();
        // Fan the stacks out over a half-turn so overlapping kinds stay readable.
        let a = -std::f32::consts::FRAC_PI_2 + (phase as f32 - 1.0) * 0.6;
        self.x[s] = wx;
        self.y[s] = wy;
        self.vx[s] = a.cos() * POP_SPEED;
        self.vy[s] = a.sin() * POP_SPEED;
        self.code[s] = code;
        // A stack larger than a `u16` cannot exist: `ITEM_STACK` caps at 99 and
        // the bag that feeds this counts cells in a 49-cell disc.
        self.count[s] = n.min(u16::MAX as u32) as u16;
        self.age[s] = 0.0;

        let col = item_color(code);
        self.r[s] = col[0];
        self.g[s] = col[1];
        self.b[s] = col[2];
    }

    /// Spawn one loot stack named by an authoring id, which may be an `a|b`
    /// preference chain. Returns false when the registry has none of the names.
    ///
    /// This is the id -> code bridge for anything that drops loot by NAME rather
    /// than by code — mobs and containers. It lives here and not in the creature
    /// layer, which deliberately does not import the item model: a creature's
    /// loot row is a string and a position, and that is all this needs.
    ///
    /// A drop naming an item the registry does not have is a content gap, not a
    /// crash: the stack is skipped. Throwing would take out the frame, and the
    /// loot buffer's own contract admits ids that "may not exist in the registry
    /// yet".
    pub fn spawn_loot(&mut self, id: &str, n: u32, wx: f32, wy: f32, phase: usize) -> bool {
        match item_code_for_drop(id) {
            Some(code) => {
                self.spawn(code, n, wx, wy, phase);
                true
            }
            None => false,
        }
    }

    fn alloc(&mut self) -> usize {
        for i in 0..CAP {
            let s = (self.next + i) % CAP;
            if self.live[s] {
                continue;
            }
            self.live[s] = true;
            self.next = (s + 1) % CAP;
            self.active += 1;
            return s;
        }
        // Full: recycle the slot the cursor is on, which is the oldest survivor.
        let s = self.next;
        self.next = (s + 1) % CAP;
        s
    }

    /// Integrate, collide against solid cells, magnet toward the player, absorb.
    ///
    /// `player_x`/`player_y` is the player's TOP-LEFT corner in world px, i.e.
    /// `Player::x`/`Player::y`; the centre is derived here exactly as the
    /// TypeScript derived it. Two floats and not a `&Player`, because a drop has
    /// no business borrowing an entity for two numbers and the host may well be
    /// magnetising things toward something that is not the player.
    ///
    /// Collision is a point test against the cell the stack is moving into, not
    /// an AABB sweep: a drop is 3px inside a 5px cell grid, so the sweep would
    /// resolve to the same answer at four times the cost. Axes are stepped
    /// separately so a stack sliding along a floor does not catch on the seam
    /// between two cells.
    ///
    /// A DROP HAS NO SIZE, ONLY A PORTRAIT. Everything spatial in this method —
    /// the `solid` probe, the MAGNET radius, the GRAB radius — is measured from
    /// `(x, y)`, the stack's centre, and none of it reads the drawn extent.
    /// Giving items 10x10 icons therefore changed no physics and no pickup
    /// behaviour: the only visible consequence is that a resting icon may overlap
    /// the cell it sits on by a couple of px, which is what an object lying on a
    /// floor looks like. Deriving the probe from the icon instead would make
    /// pickup range depend on which art an item happens to have, which is a rule
    /// about art leaking into a rule about the game.
    pub fn update(
        &mut self,
        dt: f32,
        grid: &CellGrid,
        player_x: f32,
        player_y: f32,
        inv: &mut Inventory,
    ) {
        if self.active == 0 {
            return;
        }
        let px = player_x + PLAYER_W / 2.0;
        let py = player_y + PLAYER_H / 2.0;

        for s in 0..CAP {
            if !self.live[s] {
                continue;
            }

            self.age[s] += dt;
            if self.age[s] > LIFETIME {
                self.kill(s);
                continue;
            }

            let dx = px - self.x[s];
            let dy = py - self.y[s];
            let d2 = dx * dx + dy * dy;

            if self.age[s] > ARM_TIME && d2 < MAGNET * MAGNET {
                if d2 < GRAB * GRAB {
                    // Absorb only what fits. A stack the bag cannot take stays on
                    // the floor at its current size and will be re-offered next
                    // frame.
                    let left = inv.add(self.code[s], self.count[s] as u32);
                    if left == 0 {
                        self.kill(s);
                        continue;
                    }
                    self.count[s] = left as u16;
                }
                // Accelerate rather than teleport, so the arc reads as a pickup.
                let d = d2.sqrt();
                let d = if d == 0.0 { 1.0 } else { d };
                self.vx[s] += (dx / d) * MAGNET_ACCEL * dt;
                self.vy[s] += (dy / d) * MAGNET_ACCEL * dt;
            } else {
                self.vy[s] += DROP_GRAVITY * dt;
                self.vx[s] -= self.vx[s] * (DRAG * dt).min(1.0);
            }

            // --- X -----------------------------------------------------------
            let mut nx = self.x[s] + self.vx[s] * dt;
            if solid(grid, nx, self.y[s]) {
                nx = self.x[s];
                self.vx[s] = 0.0;
            }
            self.x[s] = nx;

            // --- Y -----------------------------------------------------------
            let mut ny = self.y[s] + self.vy[s] * dt;
            if solid(grid, self.x[s], ny) {
                ny = self.y[s];
                self.vy[s] = if self.vy[s] > 30.0 {
                    -self.vy[s] * BOUNCE
                } else {
                    0.0
                };
            }
            self.y[s] = ny;
        }
    }

    fn kill(&mut self, s: usize) {
        self.live[s] = false;
        self.active -= 1;
    }
}

/// Does the cell containing this world point block movement?
fn solid(grid: &CellGrid, x: f32, y: f32) -> bool {
    let cell = WorldCell::new(cell_at(x), cell_at(y));
    if !grid.is_loaded_world(cell) {
        return false;
    }
    MAT_COLLIDE[grid.get_world(cell) as usize] == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CELL_SIZE;
    use crate::items::registry::{ITEM_STACK, item};
    use crate::sim::materials::block;

    /// A grid with a floor of stone across the bottom half, and the world origin
    /// at cell (0,0) so world px and grid px agree.
    fn floored_grid(floor_cy: i32) -> CellGrid {
        let mut g = CellGrid::new(64, 64);
        g.set_origin(0, 0);
        for cy in floor_cy..64 {
            for cx in 0..64 {
                g.set_world(WorldCell::new(cx, cy), block::STONE);
            }
        }
        g
    }

    /// The player parked far enough away that nothing magnetises.
    const FAR: f32 = 10_000.0;

    #[test]
    fn a_dropped_stack_falls_and_comes_to_rest_on_the_terrain() {
        let floor_cy = 20;
        let grid = floored_grid(floor_cy);
        let floor_px = (floor_cy * CELL_SIZE) as f32;

        let mut w = WorldItems::new();
        let mut inv = Inventory::new();
        w.spawn(item::STONE_CHUNK, 4, 50.0, 10.0, 0);
        assert_eq!(w.active(), 1);

        // Three seconds at 120 Hz is far longer than the fall needs.
        for _ in 0..360 {
            w.update(1.0 / 120.0, &grid, FAR, FAR, &mut inv);
        }

        let stack = w.stacks().next().expect("the stack is still there");
        assert!(
            stack.y > floor_px - CELL_SIZE as f32 && stack.y <= floor_px,
            "came to rest at {} but the floor is at {floor_px}",
            stack.y,
        );
        // At rest, not still bouncing.
        let before = stack.y;
        w.update(1.0 / 120.0, &grid, FAR, FAR, &mut inv);
        let after = w.stacks().next().unwrap().y;
        assert!(
            (after - before).abs() < 0.01,
            "still moving: {before} -> {after}"
        );
        assert_eq!(inv.count_of(item::STONE_CHUNK), 0, "nobody was near it");
    }

    #[test]
    fn a_stack_is_absorbed_when_the_player_overlaps_it() {
        let grid = floored_grid(20);
        let mut w = WorldItems::new();
        let mut inv = Inventory::new();
        w.spawn(item::COAL, 3, 50.0, 90.0, 0);

        // Player centred on the drop. ARM_TIME must pass before it homes.
        let px = 50.0 - PLAYER_W / 2.0;
        let py = 90.0 - PLAYER_H / 2.0;
        w.update(1.0 / 120.0, &grid, px, py, &mut inv);
        assert_eq!(inv.count_of(item::COAL), 0, "armed too early");

        for _ in 0..120 {
            w.update(1.0 / 120.0, &grid, px, py, &mut inv);
            if w.is_empty() {
                break;
            }
        }
        assert!(w.is_empty(), "the stack was never absorbed");
        assert_eq!(inv.count_of(item::COAL), 3);
    }

    #[test]
    fn a_stack_the_pack_cannot_take_stays_on_the_floor_at_its_remainder() {
        let grid = floored_grid(20);
        let mut w = WorldItems::new();
        let mut inv = Inventory::new();
        let code = item::STONE_CHUNK;
        let cap = ITEM_STACK[code as usize] as u32;
        // 29 slots of coal + 1 slot of stone: exactly one stone slot of room.
        inv.add(item::COAL, ITEM_STACK[item::COAL as usize] as u32 * 29);
        inv.add(code, cap - 2);

        w.spawn(code, 5, 50.0, 90.0, 0);
        let px = 50.0 - PLAYER_W / 2.0;
        let py = 90.0 - PLAYER_H / 2.0;
        for _ in 0..120 {
            w.update(1.0 / 120.0, &grid, px, py, &mut inv);
        }
        assert_eq!(inv.count_of(code), cap, "the pack took what it could");
        assert_eq!(w.active(), 1, "the remainder stayed on the floor");
        assert_eq!(w.stacks().next().unwrap().count, 3);
    }

    #[test]
    fn a_bag_becomes_one_stack_per_kind_and_is_emptied() {
        let mut bag = DropBag::new();
        bag.add(item::COAL, 4);
        bag.add(item::IRON_ORE, 2);
        let mut w = WorldItems::new();
        w.spawn_bag(&mut bag, 30.0, 30.0);
        assert!(bag.is_empty());
        assert_eq!(w.active(), 2);
        let kinds: Vec<_> = w.stacks().map(|s| (s.code, s.count)).collect();
        assert!(kinds.contains(&(item::COAL, 4)));
        assert!(kinds.contains(&(item::IRON_ORE, 2)));
        // Colour is cached at spawn, not looked up per read.
        for s in w.stacks() {
            assert_eq!(s.color, crate::items::registry::item_color(s.code));
        }
    }

    #[test]
    fn a_full_pool_recycles_the_oldest_survivor_and_never_overcounts() {
        let mut w = WorldItems::new();
        for i in 0..WorldItems::capacity() + 10 {
            w.spawn(item::COAL, 1, i as f32, 0.0, i);
        }
        assert_eq!(w.active(), WorldItems::capacity());
        assert_eq!(w.stacks().count(), WorldItems::capacity());
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.stacks().count(), 0);
    }

    #[test]
    fn loot_named_by_id_resolves_and_a_content_gap_is_skipped_not_fatal() {
        let mut w = WorldItems::new();
        assert!(w.spawn_loot("slime_gel", 2, 10.0, 10.0, 0));
        assert!(!w.spawn_loot("no_such_item", 2, 10.0, 10.0, 1));
        assert!(w.spawn_loot("no_such_item|slime_gel", 1, 10.0, 10.0, 2));
        assert_eq!(w.active(), 2);
    }

    #[test]
    fn a_stack_expires_after_its_lifetime() {
        let grid = floored_grid(20);
        let mut w = WorldItems::new();
        let mut inv = Inventory::new();
        w.spawn(item::COAL, 1, 50.0, 10.0, 0);
        // One big step is enough: the age gate is a comparison, not an integral.
        w.update(LIFETIME + 1.0, &grid, FAR, FAR, &mut inv);
        assert!(w.is_empty());
        assert_eq!(
            inv.count_of(item::COAL),
            0,
            "it rotted, it was not collected"
        );
    }
}

//! Collision against the cell grid. Replaces the v2 tile-object list: instead of
//! gathering Tile objects and testing each, we sample the handful of cells the
//! player box overlaps straight out of the typed array — a bitmask-style lookup
//! that stays O(cells touched) no matter how large the world is.
//!
//! Movement is still resolved one axis at a time (X then Y) so the player never
//! snags on a seam, exactly like the v2 `resolveAxis`.
//!
//! ---- THREE KINDS OF CELL --------------------------------------------------
//! `MAT_COLLIDE`  blocks everything, from every direction. Rock.
//! `MAT_ONEWAY`   blocks a body only when it is DESCENDING onto the cell's top
//!                face. Wooden platforms: you jump up through them and land on
//!                them, and a deliberate down input drops you back through.
//! `MAT_CLIMB`    blocks nothing at all. Ladders and ropes are pure geometry to
//!                the collider; they are read by the player's movement model,
//!                which suspends gravity while attached (see Player.climbing).
//! The three are independent tables and a material may set more than one.
//!
//! # What changed in the port
//!
//! The TypeScript `resolveAxis(body, delta, axis, grid, oneWayFromY)` selected
//! its axis with a `"x" | "y"` string and returned one scalar. Here the axis is
//! implied by which of `dx` / `dy` is non-zero, and [`resolve_axis`] runs the
//! full X-then-Y sequence in one call: the per-axis clamps are unchanged and
//! still strictly independent, but the ORDER — the thing that keeps a body off
//! a flush seam — is now a property of the function rather than of every call
//! site remembering to write the two calls the right way round. Passing only
//! `dx` (or only `dy`) reproduces the single-axis TypeScript call exactly,
//! because a zero delta takes neither clamp branch.
//!
//! `oneWayFromY` had a TypeScript default of `Infinity`. Rust has no default
//! arguments, so it is an explicit parameter and [`NO_ONE_WAY`] is the value
//! that spells the old default.

use crate::config::{CELL_SIZE, cell_at};
use crate::sim::coords::WorldCell;
use crate::sim::grid::CellGrid;
use crate::sim::materials::{CellId, MAT_COLLIDE, MAT_ONEWAY};

/// Slack used to keep a box that ends exactly on a cell boundary out of the next
/// cell along. Without it a body flush against a wall reads as overlapping the
/// wall's column and is pushed back a cell every step.
const EPS: f32 = 1e-4;

/// The `one_way_from_y` that opts a move OUT of one-way platforms entirely.
///
/// This is the TypeScript default: with no previous bottom edge to compare
/// against, no platform's top face can ever be at or below it, so only rock
/// blocks. Horizontal moves and the mobs pass it, and so does a deliberate
/// drop-through.
pub const NO_ONE_WAY: f32 = f32::INFINITY;

/// An axis-aligned box in world pixels. `x`,`y` is the top-left corner; y grows
/// downward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Aabb {
    #[inline]
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Aabb {
        Aabb { x, y, w, h }
    }

    /// Right edge, exclusive.
    #[inline]
    pub fn right(self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge, exclusive. This is the line `one_way_from_y` is measured on.
    #[inline]
    pub fn bottom(self) -> f32 {
        self.y + self.h
    }
}

/// True if the material at an ABSOLUTE cell blocks the player. A cell outside the
/// loaded window counts as solid — a fail-safe so the player can never fall into
/// ungenerated space (the window follows the player, so this only ever concerns
/// the off-screen margin).
///
/// ONE-WAY PLATFORMS ARE NOT SOLID HERE, and callers must not add them: this is
/// the predicate for "is there rock at this cell", used by spawn placement, by the
/// step-up headroom probe and by the mobs. A platform that answered yes to it
/// would make a whole class of world unspawnable and would let the step-up ratchet
/// a body onto a platform it should have to jump to. The directional test lives in
/// `resolve_axis`, which is the only place that knows which way a body is moving.
#[inline]
pub fn is_solid_cell(grid: &CellGrid, wcx: i32, wcy: i32) -> bool {
    let w = WorldCell::new(wcx, wcy);
    if !grid.is_loaded_world(w) {
        return true;
    }
    MAT_COLLIDE[grid.get_world(w) as usize] == 1
}

/// The moving-body predicate: rock, plus one-way platforms the body is landing on.
///
/// `one_way_from_y` is the body's bottom edge BEFORE this move. A platform blocks
/// only when its top face is at or below that line, i.e. the body was already above
/// it and is coming down onto it. Everything else follows from that single test with
/// no branch on direction:
///
///   rising through one   its top is above the previous bottom  -> not solid
///   landing on one       its top is at or below it             -> solid
///   standing on one      top === previous bottom, exactly      -> solid (stays up)
///   walking sideways     callers pass no value                 -> not solid
///   dropping through     callers pass Infinity                 -> not solid
///
/// The value of [`NO_ONE_WAY`] (infinity) is what makes the one-way behaviour
/// strictly opt-in: every caller that does not care about platforms (the mobs, the
/// horizontal axis) keeps the old rock-only semantics exactly, without a flag.
#[inline]
fn blocks_body(grid: &CellGrid, wcx: i32, wcy: i32, one_way_from_y: f32) -> bool {
    let w = WorldCell::new(wcx, wcy);
    if !grid.is_loaded_world(w) {
        return true;
    }
    let id = grid.get_world(w) as usize;
    if MAT_COLLIDE[id] == 1 {
        return true;
    }
    if MAT_ONEWAY[id] != 1 {
        return false;
    }
    (wcy * CELL_SIZE) as f32 >= one_way_from_y - EPS
}

/// Cell span a box overlaps, as `(cx0, cx1, cy0, cy1)` inclusive.
///
/// Half-open in world space: the `- EPS` on the far edges is what stops a box
/// whose right edge lands exactly on a cell boundary from claiming the cell past
/// it.
#[inline]
fn cell_span(b: Aabb) -> (i32, i32, i32, i32) {
    (
        cell_at(b.x),
        cell_at(b.x + b.w - EPS),
        cell_at(b.y),
        cell_at(b.y + b.h - EPS),
    )
}

/// Result of a resolved move. Positions are the body's new top-left corner; the
/// four flags say which faces ran into something, which is what the caller zeroes
/// velocity against (`hit_bottom` is "on the ground", `hit_top` is "bonked a
/// ceiling").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolveHits {
    /// Hit a solid on the negative x side (left).
    pub left: bool,
    /// Hit a solid on the positive x side (right).
    pub right: bool,
    /// Hit a solid on the negative y side (top / ceiling).
    pub top: bool,
    /// Hit a solid on the positive y side (bottom / floor).
    pub bottom: bool,
}

/// New position plus the faces that were blocked. See [`resolve_axis`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolveResult {
    /// New x after resolving the horizontal part of the move.
    pub x: f32,
    /// New y after resolving the vertical part of the move.
    pub y: f32,
    /// Which faces ran into a solid.
    pub hits: ResolveHits,
}

impl ResolveResult {
    /// Blocked on either horizontal side.
    #[inline]
    pub fn hit_x(self) -> bool {
        self.hits.left || self.hits.right
    }

    /// Blocked on either vertical side.
    #[inline]
    pub fn hit_y(self) -> bool {
        self.hits.top || self.hits.bottom
    }

    /// The resolved box, keeping the input's size.
    #[inline]
    pub fn aabb(self, b: Aabb) -> Aabb {
        Aabb {
            x: self.x,
            y: self.y,
            ..b
        }
    }
}

/// Horizontal half of [`resolve_axis`]: move `body` by `delta` on x and push it
/// out of any solid column it would enter. Returns `(x, hit_neg, hit_pos)`.
///
/// One-ways are never solid here, which is why this takes no `one_way_from_y`:
/// a platform has an opinion about a body's feet, not about its flanks.
fn resolve_x(grid: &CellGrid, body: Aabb, delta: f32) -> (f32, bool, bool) {
    let mut px = body.x + delta;
    let mut hit_neg = false;
    let mut hit_pos = false;

    // Cell span the moved box overlaps (half-open: touching edges don't collide).
    let (cx0, cx1, cy0, cy1) = cell_span(Aabb { x: px, ..body });

    if delta > 0.0 {
        // Clamp to the left face of the nearest solid column on the right.
        let mut bound = f32::INFINITY;
        for cx in cx0..=cx1 {
            for cy in cy0..=cy1 {
                if is_solid_cell(grid, cx, cy) {
                    bound = bound.min((cx * CELL_SIZE) as f32);
                    break;
                }
            }
        }
        if bound.is_finite() {
            px = bound - body.w;
            hit_pos = true;
        }
    } else if delta < 0.0 {
        let mut bound = f32::NEG_INFINITY;
        for cx in (cx0..=cx1).rev() {
            for cy in cy0..=cy1 {
                if is_solid_cell(grid, cx, cy) {
                    bound = bound.max(((cx + 1) * CELL_SIZE) as f32);
                    break;
                }
            }
        }
        if bound.is_finite() {
            px = bound;
            hit_neg = true;
        }
    }

    (px, hit_neg, hit_pos)
}

/// Vertical half of [`resolve_axis`] — the only axis one-way platforms have an
/// opinion about. Returns `(y, hit_neg, hit_pos)`.
fn resolve_y(grid: &CellGrid, body: Aabb, delta: f32, one_way_from_y: f32) -> (f32, bool, bool) {
    let mut py = body.y + delta;
    let mut hit_neg = false;
    let mut hit_pos = false;

    // Cell span the moved box overlaps (half-open: touching edges don't collide).
    let (cx0, cx1, cy0, cy1) = cell_span(Aabb { y: py, ..body });

    if delta > 0.0 {
        let mut bound = f32::INFINITY;
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                if blocks_body(grid, cx, cy, one_way_from_y) {
                    bound = bound.min((cy * CELL_SIZE) as f32);
                    break;
                }
            }
        }
        if bound.is_finite() {
            py = bound - body.h;
            hit_pos = true;
        }
    } else if delta < 0.0 {
        let mut bound = f32::NEG_INFINITY;
        for cy in (cy0..=cy1).rev() {
            for cx in cx0..=cx1 {
                // A rising body cannot be blocked by a platform whose top face is above
                // its previous feet, so this resolves to the rock-only test in practice.
                // It is still `blocks_body` and not `is_solid_cell` so that the one rule
                // lives in one function rather than being restated as a direction check.
                if blocks_body(grid, cx, cy, one_way_from_y) {
                    bound = bound.max(((cy + 1) * CELL_SIZE) as f32);
                    break;
                }
            }
        }
        if bound.is_finite() {
            py = bound;
            hit_neg = true;
        }
    }

    (py, hit_neg, hit_pos)
}

/// Move `b` by `(dx, dy)` and push it out of any solid cell it would enter.
/// Steps are small (fixed timestep, bounded speed), so the box only ever overlaps
/// a thin band of cells — we clamp to the nearest blocking cell face on the side
/// it is moving toward.
///
/// The two axes are resolved INDEPENDENTLY and IN ORDER, x first: x is clamped
/// against the body's current row band, then y is clamped from the already-
/// corrected x. That is what keeps a body from snagging on the seam between two
/// flush cells, and it is the reason this is not a single swept test. Passing a
/// zero for one of the deltas skips that axis's clamp entirely, which is exactly
/// the old single-axis call.
///
/// `one_way_from_y` opts this move into one-way platforms: pass the body's bottom
/// edge as it was BEFORE the move (see `blocks_body`). It is meaningful on the y
/// axis only; on x, one-ways are never solid, which is why the player passes its
/// previous bottom edge here and the mobs pass [`NO_ONE_WAY`] — which keeps the
/// old rock-only behaviour exactly.
pub fn resolve_axis(
    grid: &CellGrid,
    b: Aabb,
    dx: f32,
    dy: f32,
    one_way_from_y: f32,
) -> ResolveResult {
    let (x, left, right) = resolve_x(grid, b, dx);
    let (y, top, bottom) = resolve_y(grid, Aabb { x, ..b }, dy, one_way_from_y);
    ResolveResult {
        x,
        y,
        hits: ResolveHits {
            left,
            right,
            top,
            bottom,
        },
    }
}

/// True if any cell the box covers is solid. Used as a headroom test.
pub fn box_overlaps_solid(grid: &CellGrid, b: Aabb) -> bool {
    let (cx0, cx1, cy0, cy1) = cell_span(b);
    for cy in cy0..=cy1 {
        for cx in cx0..=cx1 {
            if is_solid_cell(grid, cx, cy) {
                return true;
            }
        }
    }
    false
}

/// Result of [`move_horizontal_stepped`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepResult {
    /// New x after the move.
    pub x: f32,
    /// Y after any step-up (unchanged if the move did not need one).
    pub y: f32,
    /// Hit a solid on the negative side (left).
    pub hit_left: bool,
    /// Hit a solid on the positive side (right).
    pub hit_right: bool,
    /// How far the body was lifted to clear the obstruction, in px.
    pub stepped: f32,
}

impl StepResult {
    /// Blocked on either side — the move did not complete even after step-up.
    #[inline]
    pub fn blocked(self) -> bool {
        self.hit_left || self.hit_right
    }
}

/// Horizontal move with step-up.
///
/// A plain axis-resolved box stops dead against anything solid, which in a world
/// built from loose grain means a single settled sand cell is a wall. This tries
/// the move again from progressively higher offsets: if the body would clear the
/// obstruction when lifted by a cell or two, it takes the move and rises instead
/// of stopping. That is the difference between walking over rubble and being
/// pinned by it.
///
/// Only whole cells are tried, smallest lift first, so the body rises the least
/// it can. Each candidate is headroom-tested before use — otherwise stepping up
/// could shove the body into a ceiling, which is how "step-up" turns into
/// "clip through the floor above".
pub fn move_horizontal_stepped(grid: &CellGrid, b: Aabb, dx: f32, step_up_max: f32) -> StepResult {
    let (flat_x, flat_neg, flat_pos) = resolve_x(grid, b, dx);
    let flat = StepResult {
        x: flat_x,
        y: b.y,
        hit_left: flat_neg,
        hit_right: flat_pos,
        stepped: 0.0,
    };
    let blocked = flat_neg || flat_pos;
    if !blocked || dx == 0.0 || step_up_max <= 0.0 {
        return flat;
    }

    // Whole cells only, smallest first. The `+ EPS` is what lets a `step_up_max`
    // of exactly one cell try that one cell.
    let max_lifts = ((step_up_max + EPS) / CELL_SIZE as f32) as i32;
    for n in 1..=max_lifts {
        let lift = (n * CELL_SIZE) as f32;
        let probe = Aabb { y: b.y - lift, ..b };
        // The raised position must itself be free, or we would be stepping into
        // geometry rather than over it.
        if box_overlaps_solid(grid, probe) {
            continue;
        }

        let (raised_x, raised_neg, raised_pos) = resolve_x(grid, probe, dx);
        // Only accept a lift that actually buys the full move. A partial gain would
        // leave the body hovering mid-step for no benefit.
        if !raised_neg && !raised_pos {
            return StepResult {
                x: raised_x,
                y: probe.y,
                hit_left: false,
                hit_right: false,
                stepped: lift,
            };
        }
    }

    flat
}

/// Is the body standing on a one-way platform (and on NOTHING else)?
///
/// The "and nothing else" is the whole point: a platform laid over rock, or one
/// whose row also contains a stone cell under the player's other foot, must not
/// be droppable-through — the drop would be granted, the pass-through timer would
/// run, and the player would stand there wondering why the input did nothing while
/// being briefly unable to land on the platform at all. Requiring the entire
/// footprint to be platform makes the input either work or not exist.
///
/// Samples the row of cells one pixel below the feet, which is the same place
/// `resolve_axis` found the floor the player is resting on.
pub fn one_way_under_feet(grid: &CellGrid, b: Aabb) -> bool {
    let cy = cell_at(b.y + b.h + 1.0);
    let cx0 = cell_at(b.x);
    let cx1 = cell_at(b.x + b.w - EPS);
    let mut saw_platform = false;
    for cx in cx0..=cx1 {
        let w = WorldCell::new(cx, cy);
        if !grid.is_loaded_world(w) {
            return false;
        }
        let id = grid.get_world(w) as usize;
        if MAT_COLLIDE[id] == 1 {
            return false; // real rock down there too
        }
        if MAT_ONEWAY[id] == 1 {
            saw_platform = true;
        }
    }
    saw_platform
}

/// Visit every cell the box overlaps (cx, cy, material id). Used by the player to
/// sample surface tags (ice / conveyor / bounce) and damage (spike / lava) from
/// the cells it is standing in or touching.
pub fn for_each_overlapped_cell(grid: &CellGrid, b: Aabb, mut f: impl FnMut(i32, i32, CellId)) {
    // Absolute cell span of the box; cells outside the loaded window are skipped.
    let (cx0, cx1, cy0, cy1) = cell_span(b);
    for cy in cy0..=cy1 {
        for cx in cx0..=cx1 {
            let w = WorldCell::new(cx, cy);
            if !grid.is_loaded_world(w) {
                continue;
            }
            f(cx, cy, grid.get_world(w));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PLAYER_H, PLAYER_W, STEP_UP_MAX};
    use crate::sim::coords::WorldCell;
    use crate::sim::materials::{EMPTY, block};

    const CS: f32 = CELL_SIZE as f32;

    /// A grid whose window starts at the world origin, so world cells and local
    /// cells coincide and a test can read like the coordinates it writes.
    fn grid() -> CellGrid {
        CellGrid::new(64, 64)
    }

    /// A solid floor spanning the whole window at row `cy`.
    fn floor(g: &mut CellGrid, cy: i32) {
        for cx in 0..64 {
            g.set(cx, cy, block::STONE);
        }
    }

    fn player_at(x: f32, y: f32) -> Aabb {
        Aabb::new(x, y, PLAYER_W, PLAYER_H)
    }

    // --- The back plane is never solid ---------------------------------------

    /// A world made entirely of background wall is a world you fall through.
    ///
    /// # Why this is a test and not a comment
    ///
    /// The invariant holds structurally today: every entry point below reaches
    /// cells through `CellGrid::get_world` or `CellGrid::material`, and the back
    /// plane is only reachable through the separately-named `get_back*`. So a
    /// collision query cannot see a wall unless somebody types the word `back`.
    ///
    /// That is exactly the kind of guarantee that survives until the day it looks
    /// like a bug. A wall is drawn where the player can see it, so "why can I walk
    /// through that?" is a reasonable thing for someone to ask, and one `||
    /// grid.get_back(...)` in `is_solid_cell` would answer it — and turn every
    /// cave in the world into solid rock, because the wall plane is the terrain
    /// the carve removed. It would be a one-line change that makes the game
    /// unplayable in a way no existing test would notice.
    ///
    /// So: the whole plane filled, the front plane left empty, and every public
    /// way of asking "is this solid" required to say no.
    #[test]
    fn the_back_plane_is_never_solid() {
        let mut g = grid();
        for cy in 0..64 {
            for cx in 0..64 {
                g.set_back_world(WorldCell::new(cx, cy), block::STONE);
            }
        }
        assert_eq!(
            g.get_back(10, 10),
            block::STONE,
            "the fixture did not take — this test would pass on an empty plane"
        );
        assert_eq!(g.get(10, 10), EMPTY, "the FRONT plane must stay empty");

        assert!(!is_solid_cell(&g, 10, 10), "a wall is not a solid cell");
        assert!(
            !box_overlaps_solid(&g, player_at(10.0 * CS, 10.0 * CS)),
            "a body standing in a walled room is not overlapping anything"
        );
        assert!(
            !one_way_under_feet(&g, player_at(10.0 * CS, 10.0 * CS)),
            "a wall is not a platform"
        );

        // Nothing is reported as overlapped either — `for_each_overlapped_cell`
        // publishes the FRONT material, and a caller that got a wall id here
        // would treat it as matter.
        let mut seen = Vec::new();
        for_each_overlapped_cell(&g, player_at(10.0 * CS, 10.0 * CS), |_, _, id| {
            seen.push(id)
        });
        assert!(
            seen.iter().all(|&id| id == EMPTY),
            "the overlap walk reported {seen:?} inside a room whose front plane is \
             empty"
        );

        // And the whole point, end to end: a body falls straight through a world
        // made of nothing but wall, and is not stopped anywhere on the way.
        // Stopping short of the window's own floor: cells outside the window read
        // SOLID by design (see `cells_outside_the_window_read_solid`), so a fall
        // that runs off the bottom is stopped by the fail-safe rather than by a
        // wall, and would fail this test for the wrong reason. 64 cells is 320 px.
        let mut body = player_at(10.0 * CS, 0.0);
        for _ in 0..100 {
            let step = resolve_axis(&g, body, 0.0, 2.0, f32::INFINITY);
            assert!(
                !step.hit_y(),
                "a wall stopped a falling body at y {}",
                body.y
            );
            body = step.aabb(body);
        }
        assert!(
            body.y > 150.0,
            "the body did not actually travel, so nothing was proven"
        );
    }

    // --- The fail-safe -------------------------------------------------------

    #[test]
    fn cells_outside_the_window_read_solid() {
        let g = grid();
        assert!(is_solid_cell(&g, -1, 0), "west of the window");
        assert!(is_solid_cell(&g, 0, -1), "above the window");
        assert!(is_solid_cell(&g, 64, 0), "east of the window");
        assert!(is_solid_cell(&g, 0, 64), "below the window");
        assert!(
            !is_solid_cell(&g, 0, 0),
            "air inside the window is not solid"
        );
    }

    #[test]
    fn one_ways_are_not_solid_cells() {
        let mut g = grid();
        g.set(5, 5, block::PLATFORM);
        assert!(
            !is_solid_cell(&g, 5, 5),
            "is_solid_cell is rock only, by design"
        );
        assert!(!box_overlaps_solid(&g, Aabb::new(25.0, 25.0, CS, CS)));
    }

    #[test]
    fn climbables_are_not_solid_to_the_collider() {
        let mut g = grid();
        g.set(5, 5, block::LADDER);
        assert!(!is_solid_cell(&g, 5, 5));
        assert!(
            !blocks_body(&g, 5, 5, 0.0),
            "a ladder blocks nothing at all"
        );
    }

    // --- Resting on a boundary -----------------------------------------------

    #[test]
    fn a_box_resting_exactly_on_a_cell_boundary_neither_sinks_nor_pops() {
        let mut g = grid();
        floor(&mut g, 20);
        // Feet exactly on the floor's top face.
        let rest_y = 20.0 * CS - PLAYER_H;
        let b = player_at(10.0, rest_y);

        // Standing still: nothing moves and nothing is reported as a hit.
        let still = resolve_axis(&g, b, 0.0, 0.0, NO_ONE_WAY);
        assert_eq!(still.x, b.x);
        assert_eq!(still.y, b.y);
        assert!(!still.hit_x() && !still.hit_y());

        // Gravity for one step: clamped straight back to the same line, on the
        // ground. A sinking body would land below `rest_y`, a popping one above.
        let fall = resolve_axis(&g, b, 0.0, 1.0, NO_ONE_WAY);
        assert_eq!(fall.y, rest_y);
        assert!(fall.hits.bottom);
        assert!(!fall.hits.top);

        // And walking along it does not catch on the floor row.
        let walk = resolve_axis(&g, b, 2.0, 0.0, NO_ONE_WAY);
        assert_eq!(walk.x, b.x + 2.0);
        assert!(!walk.hit_x());
    }

    #[test]
    fn a_box_flush_against_a_wall_is_not_pushed_back() {
        let mut g = grid();
        for cy in 0..64 {
            g.set(20, cy, block::STONE);
        }
        // Right edge exactly on the wall column's left face.
        let b = player_at(20.0 * CS - PLAYER_W, 10.0);
        let r = resolve_axis(&g, b, 0.0, 0.0, NO_ONE_WAY);
        assert_eq!(r.x, b.x, "a flush box must not be shoved out of the wall");

        let into = resolve_axis(&g, b, 3.0, 0.0, NO_ONE_WAY);
        assert_eq!(into.x, b.x, "clamped back to the same flush position");
        assert!(into.hits.right);
    }

    // --- X then Y ------------------------------------------------------------

    #[test]
    fn walking_along_a_flush_seam_does_not_snag() {
        let mut g = grid();
        floor(&mut g, 20);
        let rest_y = 20.0 * CS - PLAYER_H;
        let mut b = player_at(10.0, rest_y);

        // Twenty steps of "run right, gravity down" over a perfectly flat floor
        // made of separate cells. A swept or Y-first resolve catches on the seam
        // between two of them and stops the body dead.
        for step in 0..20 {
            let r = resolve_axis(&g, b, 0.7, 1.0, NO_ONE_WAY);
            assert_eq!(r.x, b.x + 0.7, "snagged on the seam at step {step}");
            assert_eq!(r.y, rest_y, "left the floor at step {step}");
            b = r.aabb(b);
        }
        assert!((b.x - (10.0 + 0.7 * 20.0)).abs() < 1e-3);
    }

    #[test]
    fn a_zero_delta_axis_is_never_clamped() {
        let mut g = grid();
        floor(&mut g, 20);
        // Overlapping the floor already (a spawn inside geometry). With dy = 0 the
        // y axis takes neither branch, exactly like the TypeScript single-axis
        // call with delta 0 — the box is not teleported out.
        let b = player_at(10.0, 20.0 * CS - PLAYER_H + 2.0);
        let r = resolve_axis(&g, b, 0.0, 0.0, NO_ONE_WAY);
        assert_eq!(r.y, b.y);
        assert!(!r.hit_y());
    }

    // --- One-way platforms ---------------------------------------------------

    #[test]
    fn a_one_way_is_passable_from_below_and_solid_from_above() {
        let mut g = grid();
        for cx in 0..64 {
            g.set(cx, 20, block::PLATFORM);
        }
        let top_face = 20.0 * CS;

        // Rising through it: previous bottom is BELOW the platform's top face, so
        // the platform's top is above the previous feet and does not block.
        let rising = player_at(10.0, top_face + 2.0);
        let up = resolve_axis(&g, rising, 0.0, -4.0, rising.bottom());
        assert_eq!(up.y, rising.y - 4.0, "must pass up through a platform");
        assert!(!up.hits.top);

        // Descending onto it: previous bottom at or above the top face -> solid.
        let falling = player_at(10.0, top_face - PLAYER_H - 1.0);
        let down = resolve_axis(&g, falling, 0.0, 4.0, falling.bottom());
        assert_eq!(down.y, top_face - PLAYER_H, "must land on the platform");
        assert!(down.hits.bottom);

        // Standing on it: previous bottom EXACTLY on the top face -> still solid.
        let resting = player_at(10.0, top_face - PLAYER_H);
        let stay = resolve_axis(&g, resting, 0.0, 1.0, resting.bottom());
        assert_eq!(stay.y, top_face - PLAYER_H);
        assert!(stay.hits.bottom);

        // Dropping through: NO_ONE_WAY makes the same descent pass straight by.
        let drop = resolve_axis(&g, resting, 0.0, 4.0, NO_ONE_WAY);
        assert_eq!(drop.y, resting.y + 4.0);
        assert!(!drop.hits.bottom);

        // And a platform never blocks a horizontal move.
        let sideways = resolve_axis(&g, resting, 3.0, 0.0, resting.bottom());
        assert_eq!(sideways.x, resting.x + 3.0);
        assert!(!sideways.hit_x());
    }

    #[test]
    fn one_way_under_feet_needs_the_whole_footprint() {
        let mut g = grid();
        let top_face = 20.0 * CS;
        let b = player_at(10.0, top_face - PLAYER_H); // spans cells cx 2..=3
        assert!(!one_way_under_feet(&g, b), "air is not a platform");

        g.set(2, 20, block::PLATFORM);
        g.set(3, 20, block::PLATFORM);
        assert!(one_way_under_feet(&g, b));

        // One foot over rock: not droppable-through.
        g.set(3, 20, block::STONE);
        assert!(!one_way_under_feet(&g, b));

        // Partly platform, partly air is still a platform to stand off of — the
        // rule is "no rock anywhere and at least one platform".
        g.set(3, 20, EMPTY);
        assert!(one_way_under_feet(&g, b));
    }

    #[test]
    fn one_way_under_feet_is_false_outside_the_window() {
        let mut g = grid();
        g.set_origin(0, 0);
        // Feet one pixel above row 64, which is off the bottom of the window.
        let b = player_at(10.0, 64.0 * CS - PLAYER_H - 1.0);
        assert!(!one_way_under_feet(&g, b));
    }

    // --- Step-up -------------------------------------------------------------

    #[test]
    fn step_up_clears_exactly_step_up_max_and_refuses_one_cell_more() {
        let rest_y = 20.0 * CS - PLAYER_H;

        // One cell of rubble: cleared, and the body rises exactly one cell. The
        // body sits in cells 8..=9, the rubble is the cell in front of it.
        let mut g = grid();
        floor(&mut g, 20);
        g.set(10, 19, block::SAND);
        let b = player_at(8.0 * CS, rest_y);
        let one = move_horizontal_stepped(&g, b, 2.0, STEP_UP_MAX);
        assert_eq!(one.stepped, STEP_UP_MAX);
        assert_eq!(one.y, rest_y - STEP_UP_MAX);
        assert_eq!(one.x, b.x + 2.0);
        assert!(!one.blocked());

        // Two cells: refused. The body stops flush against the wall at its old y.
        let mut g2 = grid();
        floor(&mut g2, 20);
        g2.set(10, 19, block::SAND);
        g2.set(10, 18, block::SAND);
        let two = move_horizontal_stepped(&g2, b, 2.0, STEP_UP_MAX);
        assert_eq!(two.stepped, 0.0);
        assert_eq!(two.y, rest_y);
        assert_eq!(two.x, 10.0 * CS - PLAYER_W, "clamped to the obstruction");
        assert!(two.hit_right);
    }

    #[test]
    fn step_up_refuses_a_lift_with_no_headroom() {
        let mut g = grid();
        // Body in rows 16..=18; the rubble is in that band, so the flat move is
        // blocked. A ceiling one cell above: the raised probe overlaps it, so the
        // lift is rejected rather than clipping the body into the rock.
        g.set(10, 18, block::SAND);
        for cx in 0..64 {
            g.set(cx, 15, block::STONE);
        }
        let b = player_at(8.0 * CS, 16.0 * CS);
        let r = move_horizontal_stepped(&g, b, 2.0, STEP_UP_MAX);
        assert_eq!(r.stepped, 0.0);
        assert!(r.hit_right);
    }

    #[test]
    fn an_unblocked_move_never_steps() {
        let mut g = grid();
        floor(&mut g, 20);
        let b = player_at(8.0 * CS, 20.0 * CS - PLAYER_H);
        let r = move_horizontal_stepped(&g, b, 2.0, STEP_UP_MAX);
        assert_eq!(r.stepped, 0.0);
        assert_eq!(r.x, b.x + 2.0);
        assert!(!r.blocked());
    }

    #[test]
    fn a_zero_step_budget_is_a_plain_axis_resolve() {
        let mut g = grid();
        floor(&mut g, 20);
        g.set(10, 19, block::SAND);
        let b = player_at(8.0 * CS, 20.0 * CS - PLAYER_H);
        let r = move_horizontal_stepped(&g, b, 2.0, 0.0);
        assert_eq!(r.stepped, 0.0);
        assert!(r.hit_right);
    }

    #[test]
    fn step_up_will_not_ratchet_a_body_onto_a_platform() {
        // A one-way is not solid to `resolve_x`, so the flat move already
        // succeeds and there is nothing to step over. This is the invariant the
        // "is_solid_cell is rock only" comment protects.
        let mut g = grid();
        floor(&mut g, 20);
        g.set(10, 19, block::PLATFORM);
        let b = player_at(8.0 * CS, 20.0 * CS - PLAYER_H);
        let r = move_horizontal_stepped(&g, b, 2.0, STEP_UP_MAX);
        assert_eq!(r.stepped, 0.0);
        assert_eq!(r.x, b.x + 2.0);
        assert!(!r.blocked());
    }

    // --- Cell iteration ------------------------------------------------------

    #[test]
    fn for_each_overlapped_cell_visits_exactly_the_cells_touched() {
        let mut g = grid();
        g.set(2, 4, block::STONE);
        // A 10x15 box at (10, 20) covers cells x 2..=3, y 4..=6 — and NOT x 4 or
        // y 7, whose boundaries it only touches.
        let b = player_at(10.0, 20.0);
        let mut seen: Vec<(i32, i32, CellId)> = Vec::new();
        for_each_overlapped_cell(&g, b, |cx, cy, id| seen.push((cx, cy, id)));

        let expect: Vec<(i32, i32)> = (4..=6)
            .flat_map(|cy| (2..=3).map(move |cx| (cx, cy)))
            .collect();
        assert_eq!(
            seen.iter().map(|&(x, y, _)| (x, y)).collect::<Vec<_>>(),
            expect
        );
        assert_eq!(seen[0].2, block::STONE, "the id comes along");
    }

    #[test]
    fn for_each_overlapped_cell_handles_negative_coordinates() {
        // Slide the window west and north of the origin so negative world cells
        // are actually loaded — flooring, not truncation, is what makes this work.
        let mut g = CellGrid::new(64, 64);
        g.set_origin(-32, -32);
        g.set_world(WorldCell::new(-3, -3), block::STONE);

        // A 10x15 box at (-15, -15) has its far edges exactly on x = -5 and y = 0,
        // so it covers x -3..=-2 and y -3..=-1: the cells whose boundaries it only
        // touches (-1 on x, 0 on y) are excluded, on the negative side of the
        // origin just as on the positive. Truncating division would fold the whole
        // span one cell toward zero.
        let b = Aabb::new(-15.0, -15.0, PLAYER_W, PLAYER_H);
        let mut seen: Vec<(i32, i32, CellId)> = Vec::new();
        for_each_overlapped_cell(&g, b, |cx, cy, id| seen.push((cx, cy, id)));

        let expect: Vec<(i32, i32)> = (-3..=-1)
            .flat_map(|cy| (-3..=-2).map(move |cx| (cx, cy)))
            .collect();
        assert_eq!(
            seen.iter().map(|&(x, y, _)| (x, y)).collect::<Vec<_>>(),
            expect
        );
        assert_eq!(seen[0].2, block::STONE);
    }

    #[test]
    fn for_each_overlapped_cell_skips_unloaded_cells() {
        let g = grid();
        // Straddles the window's west edge: only the in-window column is visited.
        let b = Aabb::new(-CS, 20.0, PLAYER_W, CS);
        let mut seen: Vec<(i32, i32)> = Vec::new();
        for_each_overlapped_cell(&g, b, |cx, cy, _| seen.push((cx, cy)));
        assert_eq!(seen, vec![(0, 4)]);
    }

    #[test]
    fn collision_at_negative_world_coordinates_matches_the_positive_case() {
        let mut g = CellGrid::new(64, 64);
        g.set_origin(-32, -32);
        for cx in -32..0 {
            g.set_world(WorldCell::new(cx, -4), block::STONE);
        }
        let top_face = -4.0 * CS;
        let b = player_at(-40.0, top_face - PLAYER_H - 1.0);
        let r = resolve_axis(&g, b, 0.0, 4.0, NO_ONE_WAY);
        assert_eq!(r.y, top_face - PLAYER_H);
        assert!(r.hits.bottom);
    }
}

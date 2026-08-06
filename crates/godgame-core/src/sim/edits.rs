//! World edits (dig / place) as a single shared primitive.
//!
//! In the TypeScript both the sim worker and the main-thread fallback called
//! this, so the brush behaved identically either way. Keeping edits here (not in
//! the build tool) means the only writer of cells during play is whoever runs
//! the sim — no cross-thread write races.
//!
//! # What changed in the port
//!
//! `EDIT_DIG = 0` / `EDIT_PLACE = 1` were bare numbers threaded through a
//! `mode: number` parameter. They are an [`EditMode`] enum here, which is the
//! same two values with the "what does 2 mean?" question deleted.

use super::coords::WorldCell;
use super::grid::CellGrid;
use super::materials::{CellId, EMPTY};

/// What a brush stroke does to the cells it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditMode {
    /// Clear any cell under the brush to air.
    Dig,
    /// Fill only the EMPTY cells under the brush, leaving terrain alone.
    Place,
    /// Clear the background WALL under the brush, leaving the play plane alone.
    DigBack,
    /// Fill the empty background wall under the brush.
    PlaceBack,
}

impl EditMode {
    /// Whether this mode writes the background wall plane rather than the play
    /// plane.
    #[inline]
    pub const fn is_back(self) -> bool {
        matches!(self, EditMode::DigBack | EditMode::PlaceBack)
    }

    /// Whether this mode fills rather than clears.
    #[inline]
    pub const fn is_place(self) -> bool {
        matches!(self, EditMode::Place | EditMode::PlaceBack)
    }
}

/// Stamp a filled circle at (cx,cy).
///
/// Digging clears any cell to air; placing only fills empty cells so you can't
/// paint over existing terrain or the player's footing. `set_world` wakes the
/// cells so the sim picks them up.
///
/// `cx`,`cy` are ABSOLUTE world cells (from the cursor); the grid translates
/// them to the window and drops any that fall outside the loaded region.
pub fn apply_brush(grid: &mut CellGrid, mode: EditMode, cx: i32, cy: i32, r: i32, mat: CellId) {
    let r2 = r * r;
    let place = mode.is_place();
    let back = mode.is_back();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let cell = WorldCell::new(cx + dx, cy + dy);
            if !grid.is_loaded_world(cell) {
                continue;
            }
            // The two planes are asked the same question about their OWN
            // contents. A wall may be placed behind standing terrain — that is
            // the normal case, since the front plane is what you just dug out —
            // so the occupancy test has to be about the plane being written and
            // not about the play plane.
            let occupied = if back {
                grid.get_back_world(cell) != EMPTY
            } else {
                !grid.is_empty_world(cell)
            };
            if place && occupied {
                continue;
            }
            let id = if place { mat } else { EMPTY };
            if back {
                grid.set_back_world(cell, id);
            } else {
                grid.set_world(cell, id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::coords::WorldCell;
    use crate::sim::materials::EMPTY;
    use crate::sim::materials::block;

    /// A back stroke never touches the play plane, and a front stroke never
    /// touches the walls.
    ///
    /// This is the worst bug the wall brush can produce, and it is silent in the
    /// dangerous direction: a `DigBack` that wrote the front plane would delete
    /// the terrain out from under the player while they were redecorating, and
    /// nothing else in the tree would notice — the stroke lands where the cursor
    /// is, the disc is the right size, and the only thing wrong is WHICH ARRAY it
    /// went into.
    #[test]
    fn a_stroke_writes_one_plane_and_leaves_the_other_alone() {
        // Front solid, back empty — so a write to either is unambiguous.
        let mut g = grid();
        for (cx, cy) in disc(30, 30, 6) {
            g.set(cx, cy, block::STONE);
        }

        // Digging the BACK plane leaves the standing terrain exactly as it was.
        apply_brush(&mut g, EditMode::DigBack, 30, 30, 4, EMPTY);
        for (cx, cy) in disc(30, 30, 4) {
            assert_eq!(
                g.get(cx, cy),
                block::STONE,
                "a DigBack stroke cleared the play plane at ({cx},{cy}) — this \
                 would delete the ground under the player"
            );
        }

        // Placing into the BACK plane fills walls without adding any matter.
        apply_brush(&mut g, EditMode::PlaceBack, 30, 30, 3, block::DIRT);
        for (cx, cy) in disc(30, 30, 3) {
            assert_eq!(g.get_back(cx, cy), block::DIRT, "the wall was not placed");
            assert_eq!(
                g.get(cx, cy),
                block::STONE,
                "a PlaceBack stroke added matter to the play plane"
            );
        }

        // And the front verbs still ignore the walls entirely.
        apply_brush(&mut g, EditMode::Dig, 30, 30, 3, EMPTY);
        for (cx, cy) in disc(30, 30, 3) {
            assert_eq!(g.get(cx, cy), EMPTY, "the front dig did not happen");
            assert_eq!(
                g.get_back(cx, cy),
                block::DIRT,
                "a front Dig stroke removed the wall behind it — digging a tunnel \
                 must leave you a room, not a void"
            );
        }
    }

    /// A wall may be placed behind standing terrain.
    ///
    /// The occupancy test asks about the plane being WRITTEN, not about the play
    /// plane, and that distinction is the normal case rather than an edge one:
    /// the front plane is exactly what you just dug out, and a rule that refused
    /// to wall a cell with rock in front of it would refuse almost everywhere.
    #[test]
    fn a_wall_goes_in_behind_solid_ground() {
        let mut g = grid();
        for (cx, cy) in disc(20, 20, 5) {
            g.set(cx, cy, block::STONE);
        }

        apply_brush(&mut g, EditMode::PlaceBack, 20, 20, 3, block::DIRT);
        for (cx, cy) in disc(20, 20, 3) {
            assert_eq!(
                g.get_back(cx, cy),
                block::DIRT,
                "solid ground in front blocked a wall from being placed behind it"
            );
        }
    }

    /// Placing a wall does not paint over a wall that is already there.
    ///
    /// The same rule the front plane has, asked of its own plane: `Place` fills
    /// only what is empty, so a stroke cannot silently replace what a player put
    /// down earlier.
    #[test]
    fn placing_a_wall_does_not_overwrite_one() {
        let mut g = grid();
        g.set_back_world(WorldCell::new(10, 10), block::STONE);

        apply_brush(&mut g, EditMode::PlaceBack, 10, 10, 2, block::DIRT);
        assert_eq!(
            g.get_back(10, 10),
            block::STONE,
            "the existing wall was painted over"
        );
        assert_eq!(
            g.get_back(10, 11),
            block::DIRT,
            "its neighbours still filled"
        );
    }

    fn grid() -> CellGrid {
        CellGrid::new(64, 64)
    }

    /// Cells the brush should cover, by the same disc test.
    fn disc(cx: i32, cy: i32, r: i32) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    out.push((cx + dx, cy + dy));
                }
            }
        }
        out
    }

    #[test]
    fn place_fills_a_filled_circle_and_nothing_outside_it() {
        let mut g = grid();
        apply_brush(&mut g, EditMode::Place, 32, 32, 4, block::STONE);

        for (x, y) in disc(32, 32, 4) {
            assert_eq!(g.get(x, y), block::STONE, "({x},{y}) should be inside");
        }
        // The corners of the bounding box are outside the disc (16 + 16 > 16).
        assert_eq!(g.get(28, 28), EMPTY);
        assert_eq!(g.get(36, 36), EMPTY);
        // dx=4, dy=0 is exactly on the boundary and IS included.
        assert_eq!(g.get(36, 32), block::STONE);
        assert_eq!(g.get(37, 32), EMPTY);
    }

    #[test]
    fn place_refuses_to_paint_over_existing_terrain() {
        let mut g = grid();
        g.set(32, 32, block::STONE);
        apply_brush(&mut g, EditMode::Place, 32, 32, 3, block::SAND);
        assert_eq!(g.get(32, 32), block::STONE, "occupied cells are left alone");
        assert_eq!(g.get(33, 32), block::SAND);
    }

    #[test]
    fn dig_clears_anything_it_covers() {
        let mut g = grid();
        for (x, y) in disc(32, 32, 5) {
            g.set(x, y, block::STONE);
        }
        // The material argument is ignored when digging.
        apply_brush(&mut g, EditMode::Dig, 32, 32, 3, block::LAVA);
        for (x, y) in disc(32, 32, 3) {
            assert_eq!(g.get(x, y), EMPTY, "({x},{y}) should have been dug out");
        }
        assert_eq!(g.get(32, 27), block::STONE, "outside the brush, untouched");
    }

    #[test]
    fn a_brush_off_the_loaded_window_is_a_no_op_not_a_panic() {
        let mut g = grid();
        apply_brush(&mut g, EditMode::Place, -100, -100, 5, block::STONE);
        apply_brush(&mut g, EditMode::Dig, 10_000, 10_000, 5, block::STONE);
        assert!(g.material.iter().all(|&m| m == EMPTY));
    }

    #[test]
    fn radius_zero_touches_exactly_one_cell() {
        let mut g = grid();
        apply_brush(&mut g, EditMode::Place, 20, 20, 0, block::STONE);
        assert_eq!(g.get(20, 20), block::STONE);
        assert_eq!(g.material.iter().filter(|&&m| m != EMPTY).count(), 1);
    }
}

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
    let place = mode == EditMode::Place;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let cell = WorldCell::new(cx + dx, cy + dy);
            if !grid.is_loaded_world(cell) {
                continue;
            }
            if place && !grid.is_empty_world(cell) {
                continue;
            }
            grid.set_world(cell, if place { mat } else { EMPTY });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::materials::block;

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

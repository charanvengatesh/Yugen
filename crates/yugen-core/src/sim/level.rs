//! A loaded world: the streaming window grid plus the spawn point.

use super::grid::CellGrid;
use super::worldgen::SpawnPoint;
use crate::config::{CELL_SIZE, WINDOW_COLS, WINDOW_ROWS};

/// A loaded world: the streaming window grid plus the spawn point. The world is
/// infinite; the grid is a fixed-size window (see `WindowManager`) that follows
/// the player. `width_px`/`height_px` describe only the WINDOW, not the world,
/// so nothing should treat them as world bounds.
pub struct Level {
    /// The window. The sim borrows this mutably every tick.
    pub grid: CellGrid,
    /// Where a fresh player is placed, in world px.
    pub spawn: SpawnPoint,
    /// Window extent in world px — `grid.cols() * CELL_SIZE`.
    ///
    /// Private with an accessor, unlike the two fields above, because these are
    /// derived from the grid's dimensions: letting a caller set them
    /// independently is the only way they could ever disagree.
    width_px: i32,
    height_px: i32,
}

impl Level {
    pub fn new(grid: CellGrid, spawn: SpawnPoint) -> Level {
        let width_px = grid.cols() * CELL_SIZE;
        let height_px = grid.rows() * CELL_SIZE;
        Level {
            grid,
            spawn,
            width_px,
            height_px,
        }
    }

    /// Width of the WINDOW in world px. Not a world bound.
    #[inline]
    pub const fn width_px(&self) -> i32 {
        self.width_px
    }

    /// Height of the WINDOW in world px. Not a world bound.
    #[inline]
    pub const fn height_px(&self) -> i32 {
        self.height_px
    }
}

/// Streaming window size in cells — the fixed grid the sim/render address.
#[inline]
pub const fn window_size() -> (i32, i32) {
    (WINDOW_COLS, WINDOW_ROWS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_extent_describes_the_window_not_the_world() {
        let (cols, rows) = window_size();
        let level = Level::new(CellGrid::new(cols, rows), SpawnPoint { x: 0.0, y: 0.0 });
        assert_eq!(level.width_px(), WINDOW_COLS * CELL_SIZE);
        assert_eq!(level.height_px(), WINDOW_ROWS * CELL_SIZE);
        assert_eq!(level.grid.cols(), WINDOW_COLS);
        assert_eq!(level.grid.rows(), WINDOW_ROWS);
    }
}

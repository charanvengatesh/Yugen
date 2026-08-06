//! Pure coordinate math for the streaming world — the single place that knows how
//! world pixels, absolute cells, and chunks relate. Everything here handles
//! negative (unbounded) coordinates correctly.
//!
//!   worldPx / CELL_SIZE          → absolute cell (floored)
//!   floor(absoluteCell / CHUNK)  → chunk index
//!   local cell = absoluteCell - windowOriginCell
//!
//! # What moved, and what went away, in the port
//!
//! The TypeScript `coords.ts` owned its own `floorDiv` and a `pxToWorldCell`
//! that was a byte-for-byte duplicate of `CellGrid.cellAt`. Both of those are
//! world GEOMETRY rather than world math — they are defined by `CELL_SIZE` and
//! `CHUNK_CELLS` — so they live in [`crate::config::world`] as `const fn`s and
//! are re-exported here, deduplicated, for callers who want one import for all
//! their coordinate work.
//!
//! `chunkKey(x, y) -> "x,y"` is gone entirely. It existed only because a JS
//! `Map` cannot key on a pair; a Rust `HashMap<(i32, i32), _>` keys on the pair
//! directly, so the string never needs to exist and the chunk store never pays
//! for formatting or parsing it.

use crate::config::CHUNK_CELLS;

pub use crate::config::{cell_at, floor_div};

/// Chunk index owning an absolute cell coordinate.
#[inline]
pub const fn world_cell_to_chunk(wc: i32) -> i32 {
    floor_div(wc, CHUNK_CELLS)
}

// ---------------------------------------------------------------------------
// World vs window-local cells
// ---------------------------------------------------------------------------
// The grid stores a WINDOW into an unbounded world. Two coordinate spaces
// therefore address the very same cell, they are both a pair of `i32`, and
// mixing them up produces an offset that is zero at world origin and grows as
// the player walks — the single nastiest class of bug in the TypeScript build,
// which spent paragraphs of comment in `CellGrid.ts` and `WindowManager.ts`
// warning about it and still had to be careful at every call site.
//
// The two newtypes below put a compile error at the seam where the confusion
// actually happens: the world-facing API (`get_world` and friends) takes a
// [`WorldCell`] and nothing else, and the only way to obtain a [`LocalCell`] is
// `CellGrid::to_local`, which subtracts the origin. Both are `Copy` pairs of
// `i32` with public fields, so they cost nothing at runtime and destructure
// freely.
//
// They deliberately do NOT reach into the per-cell hot API (`get`, `set`,
// `swap`, `wake`, `idx`): that is the TypeScript's frozen contract, it is
// called several million times a second by the automata, and wrapping its
// scalars would put a constructor inside every inner loop for a distinction the
// automata — which lives entirely in local space — never has to make.

/// An ABSOLUTE cell coordinate in the unbounded world.
///
/// Both components may be negative: the world extends infinitely in every
/// direction and the origin is just where the first window happened to start.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldCell {
    pub x: i32,
    pub y: i32,
}

impl WorldCell {
    #[inline]
    pub const fn new(x: i32, y: i32) -> WorldCell {
        WorldCell { x, y }
    }

    /// The chunk this cell belongs to, in absolute chunk coordinates.
    #[inline]
    pub const fn chunk(self) -> (i32, i32) {
        (world_cell_to_chunk(self.x), world_cell_to_chunk(self.y))
    }
}

/// A WINDOW-LOCAL cell coordinate — an index into the loaded grid.
///
/// In range it is `0..cols` × `0..rows`; out of that range names a cell that is
/// simply not loaded right now. Produced by `CellGrid::to_local`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalCell {
    pub x: i32,
    pub y: i32,
}

impl LocalCell {
    #[inline]
    pub const fn new(x: i32, y: i32) -> LocalCell {
        LocalCell { x, y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CELL_SIZE;

    #[test]
    fn cells_and_chunks_round_trip_across_the_origin() {
        for wc in -80..80 {
            let ch = world_cell_to_chunk(wc);
            // The chunk's cell band must actually contain the cell — the
            // property truncating division breaks west of the origin.
            assert!(
                ch * CHUNK_CELLS <= wc && wc < (ch + 1) * CHUNK_CELLS,
                "cell {wc} landed in chunk {ch}"
            );
        }
        assert_eq!(world_cell_to_chunk(-1), -1);
        assert_eq!(world_cell_to_chunk(-CHUNK_CELLS), -1);
        assert_eq!(world_cell_to_chunk(-CHUNK_CELLS - 1), -2);
        assert_eq!(world_cell_to_chunk(0), 0);
    }

    #[test]
    fn pixels_floor_to_cells_on_both_sides_of_the_origin() {
        assert_eq!(cell_at(0.0), 0);
        assert_eq!(cell_at(CELL_SIZE as f32 - 0.1), 0);
        assert_eq!(cell_at(CELL_SIZE as f32), 1);
        assert_eq!(cell_at(-0.1), -1);
        assert_eq!(cell_at(-(CELL_SIZE as f32)), -1);
        assert_eq!(cell_at(-(CELL_SIZE as f32) - 0.1), -2);
    }

    #[test]
    fn world_cell_reports_its_chunk() {
        assert_eq!(WorldCell::new(0, 0).chunk(), (0, 0));
        assert_eq!(WorldCell::new(-1, -1).chunk(), (-1, -1));
        assert_eq!(
            WorldCell::new(CHUNK_CELLS, -CHUNK_CELLS).chunk(),
            (1, -1),
            "chunk index must floor, not truncate"
        );
    }
}

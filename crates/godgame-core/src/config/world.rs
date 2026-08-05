//! World geometry — the units and grids every other subsystem addresses the
//! world in.
//!
//! Nothing here is a feel knob. These are the definitions the sim, the renderer,
//! collision and worldgen all agree on, so changing one changes what a "cell" or
//! a "chunk" means everywhere at once. Physics and combat tunables scale off
//! these (see `super::physics`); they never redefine them.

/// Authoring tile edge, in world px. One tile expands into CELLS_PER_TILE^2 cells.
pub const TILE_SIZE: i32 = 50;

/// Cellular sim geometry: px per sim cell.
///
/// The world is a grid of small material cells; a level's TILE_SIZE authoring
/// tile expands into a CELLS_PER_TILE square of cells. Collision, rendering and
/// the Noita-style automata all address the world in cell space.
pub const CELL_SIZE: i32 = 5;

/// Cells along one edge of an authoring tile.
pub const CELLS_PER_TILE: i32 = TILE_SIZE / CELL_SIZE; // 10

/// Sim/render chunk edge, in cells. Chunks sleep when nothing moves in them.
pub const CHUNK_CELLS: i32 = 32;

/// Streaming window width, in chunks.
///
/// The simulated and rendered grid is a fixed NxM-chunk window that recenters on
/// the player as they walk; the world beyond it is generated on demand and held
/// in the chunk store. The window is sized to the viewport plus a margin so
/// shifts (which touch the window edge) stay off-screen: a viewport of roughly
/// 200x100 cells wants 11x8 chunks (352x256 cells). That is enough margin that
/// the player may roam a full chunk from the window centre (the shift
/// hysteresis) with the viewport still comfortably inside the window.
pub const WINDOW_CHUNKS_X: i32 = 11;

/// Streaming window height, in chunks. See [`WINDOW_CHUNKS_X`].
pub const WINDOW_CHUNKS_Y: i32 = 8;

/// Streaming window width, in cells.
pub const WINDOW_COLS: i32 = WINDOW_CHUNKS_X * CHUNK_CELLS; // 352

/// Streaming window height, in cells.
pub const WINDOW_ROWS: i32 = WINDOW_CHUNKS_Y * CHUNK_CELLS; // 256

/// Chunks farther than this (Chebyshev) from the window centre are evicted.
pub const EVICT_RADIUS_CHUNKS: i32 = 12;

/// Cellular sim rate, in ticks per second.
///
/// In the TypeScript build this was the rate of a `setInterval` on a worker
/// thread, unsynchronised with rendering. Here it is every second tick of a
/// 120 Hz fixed schedule, so the sim and the player step against the same clock
/// and the whole shared-memory tearing problem disappears.
pub const SIM_HZ: u32 = 60;

/// Cell coordinate a world-pixel coordinate falls in.
///
/// Floor division, not truncation: world px go negative and `-1 / 5` would
/// otherwise land in cell 0 alongside `+1 / 5`.
#[inline]
pub const fn cell_at(px: f32) -> i32 {
    (px / CELL_SIZE as f32).floor() as i32
}

/// Floor division for cell -> chunk conversion, correct for negative operands.
#[inline]
pub const fn floor_div(a: i32, b: i32) -> i32 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Positive modulo — `-1 % 32` is `31`, not `-1`.
///
/// Worldgen lattices and decorator phases are anchored in absolute world space
/// and the world extends infinitely in both directions, so every stride test has
/// to be sign-agnostic or the world becomes subtly different west of the origin.
#[inline]
pub const fn pmod(a: i32, m: i32) -> i32 {
    let r = a % m;
    if r < 0 { r + m } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_div_and_pmod_agree_across_the_origin() {
        for a in -70..70 {
            let q = floor_div(a, CHUNK_CELLS);
            let r = pmod(a, CHUNK_CELLS);
            assert!((0..CHUNK_CELLS).contains(&r), "pmod({a}) = {r}");
            assert_eq!(q * CHUNK_CELLS + r, a, "floor_div/pmod disagree at {a}");
        }
    }

    #[test]
    fn cell_at_floors_negatives() {
        assert_eq!(cell_at(0.0), 0);
        assert_eq!(cell_at(4.9), 0);
        assert_eq!(cell_at(5.0), 1);
        assert_eq!(cell_at(-0.1), -1);
        assert_eq!(cell_at(-5.0), -1);
        assert_eq!(cell_at(-5.1), -2);
    }
}

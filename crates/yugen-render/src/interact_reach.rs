//! Which crafting stations the player can reach, read off the world.
//!
//! # Why this is not in `yugen-core`
//!
//! `items::crafting::Reach` is the RULE — a recipe needs its station — and it
//! lives in core because it is arithmetic over a set. This is the ANSWER, and
//! answering it needs a grid, a body and a distance, none of which that module
//! has or should grow. Core states what is required; the host works out what is
//! available. The seam is a plain value passed down, which is this tree's
//! idiom for exactly this shape (`docs/HANDOFF.md` §4).
//!
//! # The radius is the tool's, not a new number
//!
//! A station is in reach when it is within [`STATION_REACH_CELLS`] of the body.
//! That is deliberately the same distance the starting pick can dig, because a
//! player already has a mental model of how far they can touch things and
//! giving crafting its own, different, invisible radius would be teaching a
//! second one for no reason.

use yugen_core::config::{CELL_SIZE, PLAYER_H, PLAYER_W};
use yugen_core::items::crafting::Reach;
use yugen_core::items::registry::Station;
use yugen_core::sim::coords::WorldCell;
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::{CellId, code_of};

/// How far a station may be from the body and still be usable, in cells.
///
/// Four, matching `pick_traveler`'s `reach`. See the module header.
pub const STATION_REACH_CELLS: i32 = 4;

/// The block behind each station, resolved once.
///
/// By AUTHORING ID and not by a hard-coded code: codes are stable — that is what
/// `content/ids.lock.json` and `registry_golden`'s prefix rule guarantee — but a
/// literal here would still be a second place the registry is written down, and
/// the one that goes stale silently. A station whose block is missing resolves
/// to 0 and simply never matches, which is the right behaviour for a build that
/// has had its content cut down.
fn station_blocks() -> [(Station, CellId); 3] {
    [
        (Station::Workbench, code_of("workbench")),
        (Station::Furnace, code_of("furnace")),
        (Station::Anvil, code_of("anvil")),
    ]
}

/// Every station within [`STATION_REACH_CELLS`] of a body at `(x, y)`.
///
/// `x`/`y` are the body's TOP-LEFT in world px, as `Player` stores them; the
/// scan is centred on the middle of the box, so a station at head height and one
/// at the feet are the same distance away. Measuring from the corner would make
/// reach depend on which way up the difference happened to be taken.
pub fn stations_in_reach(grid: &CellGrid, x: f32, y: f32) -> Reach {
    let cx = ((x + PLAYER_W * 0.5) / CELL_SIZE as f32).floor() as i32;
    let cy = ((y + PLAYER_H * 0.5) / CELL_SIZE as f32).floor() as i32;
    let blocks = station_blocks();

    let mut reach = Reach::HAND;
    let r = STATION_REACH_CELLS;
    for dy in -r..=r {
        for dx in -r..=r {
            let at = grid.get_world(WorldCell::new(cx + dx, cy + dy));
            if at == 0 {
                continue;
            }
            for (station, code) in blocks {
                if code != 0 && at == code {
                    reach = reach.with(station);
                }
            }
        }
    }
    reach
}

#[cfg(test)]
mod tests {
    use super::*;
    use yugen_core::sim::level::window_size;

    /// A grid with one block placed at a cell offset from the body's centre.
    fn grid_with(at: Option<(i32, i32, &str)>) -> (CellGrid, f32, f32) {
        let (cols, rows) = window_size();
        let mut grid = CellGrid::new(cols, rows);
        // Body centred well inside the window, so a negative offset is still a
        // real cell rather than a clamp.
        let (bx, by) = (100.0f32, 100.0f32);
        if let Some((dx, dy, id)) = at {
            let cx = ((bx + PLAYER_W * 0.5) / CELL_SIZE as f32).floor() as i32;
            let cy = ((by + PLAYER_H * 0.5) / CELL_SIZE as f32).floor() as i32;
            grid.set_world(WorldCell::new(cx + dx, cy + dy), code_of(id));
        }
        (grid, bx, by)
    }

    #[test]
    fn an_empty_world_reaches_nothing_but_your_hands() {
        let (grid, x, y) = grid_with(None);
        let reach = stations_in_reach(&grid, x, y);
        assert!(reach.has(Station::Hand));
        assert!(!reach.has(Station::Workbench));
        assert!(!reach.has(Station::Furnace));
        assert!(!reach.has(Station::Anvil));
    }

    #[test]
    fn a_bench_beside_the_body_is_in_reach_and_only_a_bench() {
        let (grid, x, y) = grid_with(Some((2, 0, "workbench")));
        let reach = stations_in_reach(&grid, x, y);
        assert!(reach.has(Station::Workbench));
        assert!(!reach.has(Station::Furnace), "one station is not another");
    }

    /// The boundary, both sides of it. A radius nobody tests is a radius that
    /// is off by one.
    #[test]
    fn the_edge_of_reach_is_where_it_says_it_is() {
        let (near, x, y) = grid_with(Some((STATION_REACH_CELLS, 0, "workbench")));
        assert!(
            stations_in_reach(&near, x, y).has(Station::Workbench),
            "a station exactly {STATION_REACH_CELLS} cells away is in reach"
        );

        let (far, x, y) = grid_with(Some((STATION_REACH_CELLS + 1, 0, "workbench")));
        assert!(
            !stations_in_reach(&far, x, y).has(Station::Workbench),
            "one cell further is not"
        );
    }

    #[test]
    fn two_stations_are_both_reached() {
        let (cols, rows) = window_size();
        let mut grid = CellGrid::new(cols, rows);
        let (bx, by) = (100.0f32, 100.0f32);
        let cx = ((bx + PLAYER_W * 0.5) / CELL_SIZE as f32).floor() as i32;
        let cy = ((by + PLAYER_H * 0.5) / CELL_SIZE as f32).floor() as i32;
        grid.set_world(WorldCell::new(cx - 2, cy), code_of("workbench"));
        grid.set_world(WorldCell::new(cx + 2, cy), code_of("anvil"));

        let reach = stations_in_reach(&grid, bx, by);
        assert!(reach.has(Station::Workbench) && reach.has(Station::Anvil));
        assert!(!reach.has(Station::Furnace));
    }
}

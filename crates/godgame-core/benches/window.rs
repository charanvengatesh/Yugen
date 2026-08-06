//! The streaming window shift, against the ordinary tick it has to hide behind.
//!
//! Ported from `src/perf/bench-window.ts`.
//!
//! # Why this is measured separately from the sim
//!
//! A shift is the one event in the sim that is NOT amortised over a frame. It
//! saves the trailing edge to the store, memmoves the survivors, generates and
//! blits the leading edge, and hands the automata a freshly-woken region — all
//! inside a single tick. If it costs multiples of a normal tick, walking sideways
//! feels like a stutter no matter how good the steady-state number is. A mean
//! over "ticks, some of which shifted" would bury exactly the cost the player
//! feels, so the two are timed as two benchmarks and the interesting figure is
//! their RATIO.
//!
//! # The plain tick here is the honest one
//!
//! `sim.rs` paints a deliberately unsettled world, because that is the only way
//! to see the automata do work. This one runs on a REAL streamed world: the
//! terrain worldgen produced, ticked until it has gone to sleep. That is the
//! world the game is actually in most of the time, and its tick is the floor the
//! shift is measured against. Both numbers are wanted, and neither substitutes
//! for the other.
//!
//! # What a step is
//!
//! The window walks two chunks at a time, because `WindowManager::recenter` has a
//! one-chunk dead zone and a one-chunk step would sometimes be a no-op — a bench
//! whose iterations silently do nothing half the time reports half the cost of
//! the thing it names. Walking always in the same direction keeps every step
//! landing on chunks the store has never seen, so each shift pays real
//! generation rather than a cache hit; a bench that walked back and forth would
//! measure the chunk cache after the second lap.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use godgame_core::config::{CHUNK_CELLS, SEED, SURFACE_ANCHOR_Y, WINDOW_COLS, WINDOW_ROWS};
use godgame_core::sim::automata::Automata;
use godgame_core::sim::chunk_store::ChunkStore;
use godgame_core::sim::grid::CellGrid;
use godgame_core::sim::window::WindowManager;

/// Ticks run after the first fill, before anything is timed.
///
/// `init` wakes the whole window, so the first ticks sweep all 90 112 cells and
/// are wildly unrepresentative of a streamed world at rest. This is long enough
/// for worldgen's own loose matter — sand on a slope, water in a cave — to come
/// to rest and the awake set to collapse to whatever genuinely keeps moving.
const SETTLE_TICKS: u32 = 240;

/// The cell row the window is centred on.
///
/// [`SURFACE_ANCHOR_Y`] is a CELL row, not a chunk row, and the first version of
/// this file multiplied it by [`CHUNK_CELLS`] — which put the window 1 536 cells
/// down, in the underworld, under a doc comment claiming it was at the surface.
/// The shift cost it reported was real, but it was the cost of streaming
/// underworld chunks, which generate differently and more cheaply than the
/// surface band the player spends most of a session walking across.
///
/// The surface is where terrain is most varied and generation slowest (see the
/// per-band figures in `worldgen.rs`), so a shift measured here is near the worst
/// case rather than below the average.
const CENTRE_CELL_Y: i32 = SURFACE_ANCHOR_Y;

/// A window centred on the surface, ticked until it has settled.
fn settled_world() -> (CellGrid, WindowManager, Automata, i32) {
    let mut grid = CellGrid::new(WINDOW_COLS, WINDOW_ROWS);
    let mut wm = WindowManager::new(ChunkStore::new(SEED));
    let mut sim = Automata::new();
    sim.seed(SEED);

    let start_x = 0;
    wm.init(&mut grid, start_x, CENTRE_CELL_Y);
    for _ in 0..SETTLE_TICKS {
        sim.simulate(&mut grid);
    }
    (grid, wm, sim, start_x)
}

/// Cells the last tick swept, as a share of the window. Printed for both
/// benchmarks: it is what makes the two numbers comparable, and the thing that
/// would explain a shift ratio moving without any shift code changing.
fn swept_pct(grid: &CellGrid) -> f64 {
    let swept: i64 = grid
        .awake_chunks()
        .iter()
        .map(|b| i64::from((b.x1 - b.x0).max(0)) * i64::from((b.y1 - b.y0).max(0)))
        .sum();
    swept as f64 / (f64::from(grid.cols()) * f64::from(grid.rows())) * 100.0
}

fn window(c: &mut Criterion) {
    // --- The ordinary tick ---------------------------------------------------
    // `recenter` is called and returns false: the player has not left the dead
    // zone. That call is part of every frame's cost whether or not it shifts, so
    // it belongs inside the baseline rather than being excluded to flatter it.
    {
        let (mut grid, mut wm, mut sim, x) = settled_world();
        let y = CENTRE_CELL_Y;
        println!(
            "settled world: awake {}/{} chunks, {:.2}% of window swept per tick",
            grid.awake_chunk_count(),
            grid.chunk_cols() * grid.chunk_rows(),
            swept_pct(&grid)
        );

        c.bench_function("plain tick (streamed world, at rest)", |b| {
            b.iter(|| {
                let moved = wm.recenter(&mut grid, x, y);
                sim.simulate(&mut grid);
                black_box(moved)
            });
        });

        assert!(
            !wm.recenter(&mut grid, x, y),
            "the baseline shifted the window — it is not measuring an ordinary tick"
        );
    }

    // --- The shift tick ------------------------------------------------------
    {
        let (mut grid, mut wm, mut sim, start_x) = settled_world();
        let y = CENTRE_CELL_Y;
        let mut x = start_x;
        let mut shifts: u64 = 0;
        let mut steps: u64 = 0;

        c.bench_function("shift tick (window walks 2 chunks)", |b| {
            b.iter(|| {
                x += CHUNK_CELLS * 2;
                steps += 1;
                let moved = wm.recenter(&mut grid, x, y);
                shifts += u64::from(moved);
                sim.simulate(&mut grid);
                black_box(moved)
            });
        });

        // Every step must have shifted. If some did not, the number above is a
        // blend of two very different costs and means nothing on its own.
        assert_eq!(
            shifts,
            steps,
            "{} of {steps} steps did not shift the window",
            steps - shifts
        );
        println!(
            "\nshift walk: {shifts} shifts over {} cells of travel, ending at chunk {:?}\n\
             after the walk: awake {}/{} chunks, {:.2}% of window swept per tick",
            x - start_x,
            wm.origin_chunk(),
            grid.awake_chunk_count(),
            grid.chunk_cols() * grid.chunk_rows(),
            swept_pct(&grid)
        );
    }

    // --- The shift alone -----------------------------------------------------
    // The shift tick above is two costs in one: the streaming work, and a sim
    // tick over the freshly-woken incoming edge. Those have completely different
    // fixes, so the ratio against the plain tick cannot be attributed without
    // splitting them. This walks identically and never ticks, so the difference
    // between the two benchmarks is the sim's share of a shift frame.
    //
    // Not ticking is a real divergence from the game and is the reason this is a
    // third measurement rather than a replacement for the second: matter that
    // would have settled between shifts stays in flight, so the incoming edges
    // this one generates are blitted into a slightly busier window than the
    // game's. It changes what `load_incoming` wakes, not what it costs.
    {
        let (mut grid, mut wm, _sim, start_x) = settled_world();
        let y = CENTRE_CELL_Y;
        let mut x = start_x;
        let mut shifts: u64 = 0;
        let mut steps: u64 = 0;

        c.bench_function("recenter only (window walks 2 chunks, no tick)", |b| {
            b.iter(|| {
                x += CHUNK_CELLS * 2;
                steps += 1;
                let moved = wm.recenter(&mut grid, x, y);
                shifts += u64::from(moved);
                black_box(moved)
            });
        });

        assert_eq!(
            shifts,
            steps,
            "{} of {steps} steps did not shift the window",
            steps - shifts
        );
        println!(
            "\nrecenter-only walk: {shifts} shifts over {} cells",
            x - start_x
        );
    }
}

criterion_group!(benches, window);
criterion_main!(benches);

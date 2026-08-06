//! Streaming an infinite world through a fixed-size window.
//!
//! # What changed in the port
//!
//! **`attach()` is gone.** The TypeScript had two `WindowManager`s over one
//! `SharedArrayBuffer` grid: the main thread built one, called `init` to fill the
//! window so the first frame was correct, and then the worker built a SECOND one
//! over the same cells with its own, empty, `ChunkStore`. `attach` was how that
//! second manager adopted a window it had not filled — it read the origin out of
//! the shared control block instead of choosing one, and it walked the whole
//! window seeding a pristine baseline copy of every resident chunk into its empty
//! store, because without a baseline the first shift would have had nothing to
//! diff the outgoing chunks against and would have marked the entire trailing
//! edge diverged.
//!
//! Every clause of that is about the two-owner dance. Here there is one grid,
//! one store and one schedule: `init` fills the window and, in doing so, leaves
//! `load_or_generate`'s freshly generated chunk in the store's hot cache — which
//! IS the pristine baseline `attach` had to reconstruct. A manager can never
//! observe a window it did not fill, so there is no state for `attach` to adopt
//! and nothing it could seed that `init` has not already seeded. Reloading a
//! world is `init` on a new manager, not an adoption. It is not ported.
//!
//! The other differences are mechanical: the shared control block is gone (see
//! [`CellGrid::set_origin`]), the profiler brackets around `recenter` are gone
//! with it, and `blit_in` is a row `copy_from_slice` rather than the TypeScript's
//! deliberate scalar loop — see the note on that function.

use super::chunk::Chunk;
use super::chunk_store::ChunkStore;
use super::coords::floor_div;
use super::grid::CellGrid;
use crate::config::{
    CHUNK_CELLS, EVICT_RADIUS_CHUNKS, WINDOW_CHUNKS_X, WINDOW_CHUNKS_Y, WINDOW_COLS, WINDOW_ROWS,
};

/// `i32::max` is not a `const fn`, and the resident radius below is a compile-time
/// property of the window geometry rather than something to recompute per shift.
const fn max2(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

/// See [`max2`].
const fn min2(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

/// Streams an infinite world through a fixed-size window (the CellGrid). The
/// window's top-left maps to absolute cell (originChunkX*CHUNK, originChunkY*CHUNK);
/// as the player walks, `recenter` shifts the window in whole-chunk steps so the
/// player stays near the centre.
///
/// A shift is incremental. The window moves by (dcx,dcy) chunks, so most of it is
/// the *same* world as before, just at a different offset:
///
///   1. save the chunks that fall off the trailing edge back to the ChunkStore,
///   2. memmove the surviving rectangle to its new offset inside the grid,
///   3. load only the chunks that appear along the leading edge.
///
/// For the usual one-chunk step that is 8 saves + 8 loads instead of 88 + 88, and
/// the survivors never round-trip through a Chunk object at all. A true ring
/// buffer would make step 2 free too, but that needs modular index math inside
/// CellGrid; the memmove is the best a linear layout allows.
///
/// # Ownership
///
/// The TypeScript took the grid and the store in its constructor and held both
/// for life. This holds the STORE — which is its private business, nothing else
/// in the engine touches it — and borrows the grid per call, because the grid is
/// owned by [`Level`](super::level::Level) and the sim borrows it mutably every
/// tick. A `&mut CellGrid` field would put a lifetime parameter on the window
/// manager and on everything that owns one, to express a borrow that the
/// single-threaded tick order already makes exclusive.
pub struct WindowManager {
    origin_chunk_x: i32,
    origin_chunk_y: i32,
    store: ChunkStore,
}

impl WindowManager {
    const HALF_X: i32 = WINDOW_CHUNKS_X >> 1;
    const HALF_Y: i32 = WINDOW_CHUNKS_Y >> 1;
    /// Chunks the player may drift from centre before the window shifts.
    const DEAD: i32 = 1;

    /// Chebyshev distance from the window centre to its farthest resident chunk.
    /// Store eviction is clamped to at least this, because a resident chunk's cache
    /// entry is a *stale* copy (kept only as the divergence baseline) and must
    /// never be flushed to persistence while the live cells sit in the grid.
    const RESIDENT_RADIUS: i32 = max2(
        max2(Self::HALF_X, WINDOW_CHUNKS_X - 1 - Self::HALF_X),
        max2(Self::HALF_Y, WINDOW_CHUNKS_Y - 1 - Self::HALF_Y),
    );

    /// A manager over a chunk store. The window is not positioned until
    /// [`WindowManager::init`].
    pub fn new(store: ChunkStore) -> WindowManager {
        WindowManager {
            origin_chunk_x: 0,
            origin_chunk_y: 0,
            store,
        }
    }

    /// The store this manager streams through. Exposed read-only for tests and
    /// diagnostics; the streaming path is the only thing that should drive it.
    #[inline]
    pub fn store(&self) -> &ChunkStore {
        &self.store
    }

    /// Mutable access to the store, for an editor or a tool that has to reach a
    /// chunk outside the window.
    #[inline]
    pub fn store_mut(&mut self) -> &mut ChunkStore {
        &mut self.store
    }

    /// Absolute chunk coordinate of the window's top-left chunk.
    #[inline]
    pub const fn origin_chunk(&self) -> (i32, i32) {
        (self.origin_chunk_x, self.origin_chunk_y)
    }

    /// Centre the window on a spawn cell and fill it. Call once at load.
    pub fn init(&mut self, grid: &mut CellGrid, spawn_cell_x: i32, spawn_cell_y: i32) {
        assert_window_sized(grid);
        let pcx = floor_div(spawn_cell_x, CHUNK_CELLS);
        let pcy = floor_div(spawn_cell_y, CHUNK_CELLS);
        self.set_origin(grid, pcx - Self::HALF_X, pcy - Self::HALF_Y);
        self.load_all(grid);
        grid.wake_all();
    }

    /// Keep the window centred on the player. Shifts (and returns true) only when
    /// the player has drifted more than DEAD chunks from the window centre, so a
    /// player loitering on a chunk boundary doesn't thrash the buffer.
    pub fn recenter(
        &mut self,
        grid: &mut CellGrid,
        player_cell_x: i32,
        player_cell_y: i32,
    ) -> bool {
        assert_window_sized(grid);
        let center_x = self.origin_chunk_x + Self::HALF_X;
        let center_y = self.origin_chunk_y + Self::HALF_Y;
        let pcx = floor_div(player_cell_x, CHUNK_CELLS);
        let pcy = floor_div(player_cell_y, CHUNK_CELLS);
        if (pcx - center_x).abs() <= Self::DEAD && (pcy - center_y).abs() <= Self::DEAD {
            return false;
        }

        // How far the window itself travels, in chunks.
        let dcx = pcx - center_x;
        let dcy = pcy - center_y;
        let new_origin_x = self.origin_chunk_x + dcx;
        let new_origin_y = self.origin_chunk_y + dcy;

        if dcx.abs() >= WINDOW_CHUNKS_X || dcy.abs() >= WINDOW_CHUNKS_Y {
            // Teleport-sized jump: nothing survives, so the old whole-window path is
            // already the cheapest correct one.
            self.save_all(grid);
            self.set_origin(grid, new_origin_x, new_origin_y);
            self.load_all(grid);
            grid.wake_all(); // nothing survived — everything is freshly loaded
        } else {
            // Order matters: save_outgoing reads the grid at the OLD offset and
            // names chunks off the OLD origin, and shift_overlap must run before
            // the origin moves for the same reason.
            self.save_outgoing(grid, dcx, dcy);
            Self::shift_overlap(grid, dcx, dcy);
            // The sim's per-chunk state describes the very cells shift_overlap just
            // moved, so it scrolls by the same amount rather than being thrown away.
            // `load_slot` then wakes each chunk it actually fills, which is the only
            // matter that genuinely needs to settle. Waking the whole window here
            // instead — as this used to — costs one tick that sweeps all 90k cells
            // twice, measured at ~55x an ordinary tick.
            grid.shift_chunks(dcx, dcy);
            self.set_origin(grid, new_origin_x, new_origin_y);
            self.load_incoming(grid, dcx, dcy);
        }

        self.store
            .evict_beyond(pcx, pcy, max2(EVICT_RADIUS_CHUNKS, Self::RESIDENT_RADIUS));
        grid.bump_shift_gen();
        true
    }

    fn set_origin(&mut self, grid: &mut CellGrid, ocx: i32, ocy: i32) {
        self.origin_chunk_x = ocx;
        self.origin_chunk_y = ocy;
        grid.set_origin(ocx * CHUNK_CELLS, ocy * CHUNK_CELLS);
    }

    // --- Whole-window paths (first fill, and jumps with no overlap) ------------

    fn load_all(&mut self, grid: &mut CellGrid) {
        for cj in 0..WINDOW_CHUNKS_Y {
            for ci in 0..WINDOW_CHUNKS_X {
                self.load_slot(grid, ci, cj);
            }
        }
    }

    fn save_all(&mut self, grid: &CellGrid) {
        for cj in 0..WINDOW_CHUNKS_Y {
            for ci in 0..WINDOW_CHUNKS_X {
                self.save_slot(grid, ci, cj);
            }
        }
    }

    // --- Incremental shift ----------------------------------------------------

    /// Half-open slot range, in OLD window coords, that survives a shift of `d`
    /// chunks along one axis. Slot `i` ends up at new slot `i - d`.
    const fn survivor_range(d: i32, span: i32) -> (i32, i32) {
        (max2(0, d), min2(span, span + d))
    }

    /// Write back only the chunks the shift pushes out of the window.
    fn save_outgoing(&mut self, grid: &CellGrid, dcx: i32, dcy: i32) {
        let (i_lo, i_hi) = Self::survivor_range(dcx, WINDOW_CHUNKS_X);
        let (j_lo, j_hi) = Self::survivor_range(dcy, WINDOW_CHUNKS_Y);
        for cj in 0..WINDOW_CHUNKS_Y {
            let row_survives = cj >= j_lo && cj < j_hi;
            for ci in 0..WINDOW_CHUNKS_X {
                if row_survives && ci >= i_lo && ci < i_hi {
                    continue; // stays resident
                }
                self.save_slot(grid, ci, cj);
            }
        }
    }

    /// Fill only the slots the shift left uncovered. Call AFTER `set_origin`.
    fn load_incoming(&mut self, grid: &mut CellGrid, dcx: i32, dcy: i32) {
        // Survivor ranges expressed in NEW window coords: subtract the shift.
        let (i_lo, i_hi) = Self::survivor_range(dcx, WINDOW_CHUNKS_X);
        let (j_lo, j_hi) = Self::survivor_range(dcy, WINDOW_CHUNKS_Y);
        for cj in 0..WINDOW_CHUNKS_Y {
            let row_covered = cj >= j_lo - dcy && cj < j_hi - dcy;
            for ci in 0..WINDOW_CHUNKS_X {
                if row_covered && ci >= i_lo - dcx && ci < i_hi - dcx {
                    continue;
                }
                self.load_slot(grid, ci, cj);
            }
        }
    }

    /// Move the surviving cells to where the new origin expects them. A cell at
    /// local (x,y) holds absolute (originCell + x, ...); after the origin advances
    /// by (dx,dy) cells that same cell must sit at (x - dx, y - dy).
    ///
    /// Source and destination overlap, so this is a memmove, not a copy:
    ///   - Horizontally, source and destination are the same row whenever dy is 0,
    ///     so `copy_within` (which is specified as a memmove) does the work rather
    ///     than a copy through a temporary.
    ///   - Vertically, row y is written by the pass that reads row y + dy. Walking
    ///     top-down when dy > 0 (destination row is above the source) and bottom-up
    ///     when dy < 0 guarantees every row is read before it is overwritten.
    ///
    /// `copy_within` handles the intra-row overlap for us; the row walk direction
    /// is the part no primitive can do, because the rows are separate calls.
    fn shift_overlap(grid: &mut CellGrid, dcx: i32, dcy: i32) {
        if dcx == 0 && dcy == 0 {
            return;
        }
        let cols = grid.cols();
        let rows = grid.rows();
        let dx = dcx * CHUNK_CELLS;
        let dy = dcy * CHUNK_CELLS;

        // Surviving cell rectangle in OLD local coords.
        let x0 = max2(0, dx);
        let x1 = min2(cols, cols + dx);
        let y0 = max2(0, dy);
        let y1 = min2(rows, rows + dy);
        if x0 >= x1 || y0 >= y1 {
            return;
        }

        let len = x1 - x0;
        if dy >= 0 {
            for y in y0..y1 {
                shift_row(grid, y, dx, dy, x0, len);
            }
        } else {
            for y in (y0..y1).rev() {
                shift_row(grid, y, dx, dy, x0, len);
            }
        }
    }

    // --- Single-slot save / load ----------------------------------------------

    fn load_slot(&mut self, grid: &mut CellGrid, ci: i32, cj: i32) {
        let chunk = self
            .store
            .load_or_generate(self.origin_chunk_x + ci, self.origin_chunk_y + cj);
        blit_in(grid, chunk, ci, cj);
        // blit_in writes the planes directly, so nothing on that path wakes the sim.
        // Freshly arrived matter has to settle, and the halo lets it flow across the
        // seam into whatever was already resident.
        grid.wake_chunk_slot(ci, cj);
    }

    fn save_slot(&mut self, grid: &CellGrid, ci: i32, cj: i32) {
        let cx = self.origin_chunk_x + ci;
        let cy = self.origin_chunk_y + cj;
        // The cached chunk holds this slot's cells as of the last blit-in, which is
        // what extract_out diffs against to decide divergence. A miss can only
        // happen if the chunk was dropped while resident; the all-zero fallback then
        // reads as "everything changed", which errs toward persisting — never
        // toward silently discarding an edit.
        let chunk = self.store.entry_mut(cx, cy);
        extract_out(chunk, grid, ci, cj);
    }
}

/// The window manager indexes the grid by window slot, not by the grid's own
/// chunk counts, so it only makes sense over a grid that IS the window. The
/// TypeScript relied on that silently; here a mismatch would be an out-of-range
/// row write, so it is stated once and checked.
#[inline]
fn assert_window_sized(grid: &CellGrid) {
    assert_eq!(
        (grid.cols(), grid.rows()),
        (WINDOW_COLS, WINDOW_ROWS),
        "WindowManager drives a grid that is exactly the streaming window"
    );
}

fn shift_row(grid: &mut CellGrid, y: i32, dx: i32, dy: i32, x0: i32, len: i32) {
    let cols = grid.cols() as usize;
    let src = (y as usize) * cols + x0 as usize;
    let dst = ((y - dy) as usize) * cols + (x0 - dx) as usize;
    let end = src + len as usize;
    grid.material.copy_within(src..end, dst);
    grid.flags.copy_within(src..end, dst);
    grid.aux.copy_within(src..end, dst);
    // Heat travels with the cells it belongs to. Survivors are moved rather than
    // re-blitted, so if this were omitted the thermal field would stay at the
    // old offset and every shift would smear hotspots sideways.
    grid.temp.copy_within(src..end, dst);
}

/// Copy a stored chunk's cells into window slot (ci,cj).
///
/// This is one `copy_from_slice` per plane per row. The TypeScript wrote it as an
/// explicit element loop instead, and said why: `dst.set(src.subarray(...))`
/// ALLOCATES a view object per call, so the natural form allocated four of them
/// per row — 128 short-lived views per chunk, ~1000 per one-chunk shift, all to
/// move 32 elements each. That is GC pressure created exactly at the moment the
/// frame can least afford it, and 32 elements is far too short for `set`'s
/// memmove to pay it back.
///
/// Neither half of that reasoning survives the port. A Rust slice range is a
/// pointer and a length in registers, it allocates nothing, and there is no GC to
/// pressure — so the row copy is both the clearer form and the faster one, and
/// `copy_from_slice` bottoms out in `memcpy` with the bounds checked once per row
/// instead of once per element. This is a case where the Rust is strictly better
/// and the only reason the TypeScript wasn't is gone.
fn blit_in(grid: &mut CellGrid, chunk: &Chunk, ci: i32, cj: i32) {
    let cols = grid.cols() as usize;
    let n = CHUNK_CELLS as usize;
    let base_lx = (ci * CHUNK_CELLS) as usize;
    let base_ly = (cj * CHUNK_CELLS) as usize;
    for ly in 0..n {
        let src = ly * n;
        let dst = (base_ly + ly) * cols + base_lx;
        grid.material[dst..dst + n].copy_from_slice(&chunk.material[src..src + n]);
        grid.flags[dst..dst + n].copy_from_slice(&chunk.flags[src..src + n]);
        grid.aux[dst..dst + n].copy_from_slice(&chunk.aux[src..src + n]);
        // Overwrites whatever heat the slot's previous occupant left behind. A
        // freshly generated chunk's plane is all-ambient, which is why nothing
        // needs to wipe the field on a shift.
        grid.temp[dst..dst + n].copy_from_slice(&chunk.temp[src..src + n]);
    }
}

/// Copy window slot (ci,cj) back out into a chunk for storage, marking the
/// chunk diverged if the window changed anything meaningful since it was
/// blitted in. That flag is what tells the ChunkStore this chunk can no longer
/// be regenerated from the seed.
fn extract_out(chunk: &mut Chunk, grid: &CellGrid, ci: i32, cj: i32) {
    let cols = grid.cols() as usize;
    let base_lx = (ci * CHUNK_CELLS) as usize;
    let base_ly = (cj * CHUNK_CELLS) as usize;
    let mut changed = false;
    for ly in 0..CHUNK_CELLS {
        let src_row = (base_ly + ly as usize) * cols + base_lx;
        if chunk.absorb_row(
            ly,
            &grid.material,
            &grid.flags,
            &grid.aux,
            &grid.temp,
            src_row,
        ) {
            changed = true;
        }
    }
    if changed {
        chunk.diverged = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::materials::CellId;
    use crate::sim::worldgen::ChunkGen;

    const SEED: u32 = 20260805;

    fn fresh_grid() -> CellGrid {
        CellGrid::new(WINDOW_COLS, WINDOW_ROWS)
    }

    /// A reference window built WITHOUT the window manager: every slot blitted
    /// straight from worldgen at the given origin. If the streaming path is
    /// correct, the live grid is always equal to this for the origin it is at.
    fn reference_materials(chunk_gen: &mut ChunkGen, ocx: i32, ocy: i32) -> Vec<CellId> {
        let cols = WINDOW_COLS as usize;
        let n = CHUNK_CELLS as usize;
        let mut out = vec![0 as CellId; cols * WINDOW_ROWS as usize];
        for cj in 0..WINDOW_CHUNKS_Y {
            for ci in 0..WINDOW_CHUNKS_X {
                let chunk = chunk_gen.generate(ocx + ci, ocy + cj);
                let base_lx = (ci * CHUNK_CELLS) as usize;
                let base_ly = (cj * CHUNK_CELLS) as usize;
                for ly in 0..n {
                    let dst = (base_ly + ly) * cols + base_lx;
                    out[dst..dst + n].copy_from_slice(&chunk[ly * n..ly * n + n]);
                }
            }
        }
        out
    }

    fn assert_matches_fresh(
        grid: &CellGrid,
        wm: &WindowManager,
        chunk_gen: &mut ChunkGen,
        what: &str,
    ) {
        let (ocx, ocy) = wm.origin_chunk();
        assert_eq!(
            grid.origin_cell_x(),
            ocx * CHUNK_CELLS,
            "grid origin disagrees with the manager's ({what})"
        );
        let want = reference_materials(chunk_gen, ocx, ocy);
        if grid.material != want {
            let bad = (0..want.len())
                .find(|&i| grid.material[i] != want[i])
                .expect("vectors differ");
            panic!(
                "{what}: window at chunk origin ({ocx},{ocy}) differs from a fresh \
                 generate at local cell ({}, {}) — got {}, want {}",
                bad % WINDOW_COLS as usize,
                bad / WINDOW_COLS as usize,
                grid.material[bad],
                want[bad]
            );
        }
    }

    #[test]
    fn init_centres_the_window_on_the_spawn_chunk_and_fills_it() {
        let mut grid = fresh_grid();
        let mut chunk_gen = ChunkGen::new(SEED);
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);

        assert_eq!(
            wm.origin_chunk(),
            (-(WINDOW_CHUNKS_X >> 1), -(WINDOW_CHUNKS_Y >> 1))
        );
        assert_matches_fresh(&grid, &wm, &mut chunk_gen, "after init");
        assert_eq!(
            wm.store().len(),
            (WINDOW_CHUNKS_X * WINDOW_CHUNKS_Y) as usize,
            "init leaves a pristine baseline for every resident chunk in the cache — \
             which is the job the TypeScript's `attach` existed to do"
        );
    }

    /// THE test. Walk the player far in every direction, including diagonally,
    /// and after every shift require the visible cells to be exactly what a fresh
    /// generate at that origin would produce. A wrong save/shift/load order, a
    /// wrong survivor range or a wrong memmove walk direction all show up here as
    /// a band of duplicated or stale cells.
    #[test]
    fn a_long_walk_leaves_the_window_identical_to_a_fresh_generate() {
        let mut grid = fresh_grid();
        let mut chunk_gen = ChunkGen::new(SEED);
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);

        let legs: [(i32, i32); 8] = [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ];
        let mut px = 0;
        let mut py = 0;
        for (lx, ly) in legs {
            for step in 0..6 {
                // Alternate a two-chunk stride with a one-chunk-plus stride so
                // both a single-chunk shift and a multi-chunk one are exercised.
                let stride = if step % 2 == 0 { 2 } else { 3 };
                px += lx * CHUNK_CELLS * stride;
                py += ly * CHUNK_CELLS * stride;
                assert!(
                    wm.recenter(&mut grid, px, py),
                    "a stride of {stride} chunks must shift"
                );
                assert_matches_fresh(
                    &grid,
                    &wm,
                    &mut chunk_gen,
                    &format!("leg ({lx},{ly}) step {step}"),
                );
            }
        }
    }

    #[test]
    fn a_teleport_refills_the_whole_window() {
        let mut grid = fresh_grid();
        let mut chunk_gen = ChunkGen::new(SEED);
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);

        // Far enough that nothing overlaps, in both axes at once.
        let far = CHUNK_CELLS * (WINDOW_CHUNKS_X + WINDOW_CHUNKS_Y) * 4;
        assert!(wm.recenter(&mut grid, far, -far));
        assert_matches_fresh(&grid, &wm, &mut chunk_gen, "after a teleport");

        // And back. The origin is a pure function of the player cell, so this
        // lands exactly where init did.
        assert!(wm.recenter(&mut grid, 0, 0));
        assert_matches_fresh(&grid, &wm, &mut chunk_gen, "after teleporting home");
    }

    #[test]
    fn a_shift_bumps_the_shift_generation_and_a_no_op_does_not() {
        let mut grid = fresh_grid();
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);
        let before = grid.shift_gen();

        assert!(!wm.recenter(&mut grid, 0, 0));
        assert_eq!(grid.shift_gen(), before);

        assert!(wm.recenter(&mut grid, CHUNK_CELLS * 4, 0));
        assert_eq!(grid.shift_gen(), before + 1);
    }

    /// The hysteresis exists so a player standing on a chunk seam doesn't shift
    /// the whole window back and forth once a frame. DEAD = 1, so the player may
    /// sit anywhere in the 3x3 chunk block around the centre.
    #[test]
    fn oscillating_on_a_chunk_boundary_does_not_thrash_the_window() {
        let mut grid = fresh_grid();
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);
        let origin = wm.origin_chunk();
        let gen_before = grid.shift_gen();

        // The centre chunk's cell band, and one cell either side of each seam.
        let (ocx, ocy) = origin;
        let cx0 = (ocx + (WINDOW_CHUNKS_X >> 1)) * CHUNK_CELLS;
        let cy0 = (ocy + (WINDOW_CHUNKS_Y >> 1)) * CHUNK_CELLS;
        for _ in 0..16 {
            for (dx, dy) in [
                (-1, -1),
                (0, 0),
                (CHUNK_CELLS, 0),
                (0, CHUNK_CELLS),
                (CHUNK_CELLS + 1, CHUNK_CELLS + 1),
                (-CHUNK_CELLS, -CHUNK_CELLS),
                (CHUNK_CELLS * 2 - 1, CHUNK_CELLS * 2 - 1),
            ] {
                assert!(
                    !wm.recenter(&mut grid, cx0 + dx, cy0 + dy),
                    "player at (+{dx},+{dy}) from the centre chunk is inside the dead zone"
                );
            }
        }
        assert_eq!(wm.origin_chunk(), origin);
        assert_eq!(grid.shift_gen(), gen_before);

        // Two chunks out is outside it, in either axis.
        assert!(wm.recenter(&mut grid, cx0 + CHUNK_CELLS * 2, cy0));
    }

    /// An edit made in the window has to survive being shifted out, evicted past
    /// the eviction radius, and walked back to — while its untouched neighbours
    /// regenerate bit-identically.
    #[test]
    fn an_edit_survives_eviction_while_pristine_chunks_regenerate() {
        let mut grid = fresh_grid();
        let mut chunk_gen = ChunkGen::new(SEED);
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);
        let home = wm.origin_chunk();

        // Edit one cell in the centre chunk. A material no generator produces, so
        // it cannot be confused with a coincidence.
        const MARKER: CellId = 4242;
        let lx = (WINDOW_CHUNKS_X >> 1) * CHUNK_CELLS + 5;
        let ly = (WINDOW_CHUNKS_Y >> 1) * CHUNK_CELLS + 7;
        let idx = (ly * WINDOW_COLS + lx) as usize;
        let pristine = grid.material[idx];
        assert_ne!(pristine, MARKER);
        grid.material[idx] = MARKER;

        // Walk away, far past the eviction radius, then come home.
        let far = CHUNK_CELLS * (EVICT_RADIUS_CHUNKS + WINDOW_CHUNKS_X + 4);
        assert!(wm.recenter(&mut grid, far, 0));
        assert!(wm.recenter(&mut grid, far * 2, 0));
        assert!(
            wm.store().persisted_len() >= 1,
            "the diverged chunk was written through on eviction"
        );
        assert!(wm.recenter(&mut grid, 0, 0));
        assert_eq!(wm.origin_chunk(), home);

        assert_eq!(
            grid.material[idx], MARKER,
            "the edit came back from persistence"
        );

        // Everything else is exactly a fresh generate: restore the one edited cell
        // and the whole window must match again.
        grid.material[idx] = pristine;
        assert_matches_fresh(&grid, &wm, &mut chunk_gen, "after the round trip");
    }

    /// Temperature is carried by the memmove, not re-blitted, so a hotspot must
    /// arrive at its new offset rather than staying put.
    #[test]
    fn heat_travels_with_the_cells_it_belongs_to() {
        let mut grid = fresh_grid();
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);

        // A cell well inside the surviving rectangle for a one-chunk shift east.
        let lx = WINDOW_COLS / 2;
        let ly = WINDOW_ROWS / 2;
        let before = (ly * WINDOW_COLS + lx) as usize;
        grid.temp[before] = 199;

        let center_cell_x = (wm.origin_chunk().0 + (WINDOW_CHUNKS_X >> 1)) * CHUNK_CELLS;
        assert!(wm.recenter(&mut grid, center_cell_x + CHUNK_CELLS * 2, 0));

        let after = (ly * WINDOW_COLS + (lx - CHUNK_CELLS * 2)) as usize;
        assert_eq!(grid.temp[after], 199, "the hotspot moved with its cells");
        assert_eq!(grid.temp[before], 0, "and did not stay behind");
    }

    /// A temperature-only difference must not latch `diverged` — the reason is
    /// spelled out on `Chunk::absorb_row`, and this is the level it matters at:
    /// nothing near a lava lake should be written through on eviction.
    #[test]
    fn a_hot_but_unedited_chunk_is_never_persisted() {
        let mut grid = fresh_grid();
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);

        // Heat every cell in the window, as a diffusing field would.
        for (i, t) in grid.temp.iter_mut().enumerate() {
            *t = (i % 251) as u8;
        }

        let far = CHUNK_CELLS * (EVICT_RADIUS_CHUNKS + WINDOW_CHUNKS_X + 4);
        assert!(wm.recenter(&mut grid, far, 0));
        assert!(wm.recenter(&mut grid, far * 2, 0));
        assert_eq!(
            wm.store().persisted_len(),
            0,
            "heat alone must never make a chunk non-regenerable"
        );
    }

    #[test]
    fn the_survivor_range_matches_the_shift_it_describes() {
        // No shift: everything survives.
        assert_eq!(WindowManager::survivor_range(0, 8), (0, 8));
        // Shift east by 3: old slots 3..8 survive, landing at new 0..5.
        assert_eq!(WindowManager::survivor_range(3, 8), (3, 8));
        // Shift west by 3: old slots 0..5 survive, landing at new 3..8.
        assert_eq!(WindowManager::survivor_range(-3, 8), (0, 5));
        // Past the span: an empty range.
        let (lo, hi) = WindowManager::survivor_range(8, 8);
        assert!(lo >= hi);
        let (lo, hi) = WindowManager::survivor_range(-8, 8);
        assert!(lo >= hi);
    }

    #[test]
    fn the_resident_radius_covers_the_farthest_resident_chunk() {
        // Eviction is clamped to at least this, so a resident chunk's stale cache
        // entry can never be flushed over its live cells.
        let half_x = WINDOW_CHUNKS_X >> 1;
        let half_y = WINDOW_CHUNKS_Y >> 1;
        for cj in 0..WINDOW_CHUNKS_Y {
            for ci in 0..WINDOW_CHUNKS_X {
                let d = (ci - half_x).abs().max((cj - half_y).abs());
                assert!(
                    d <= WindowManager::RESIDENT_RADIUS,
                    "slot ({ci},{cj}) is {d} chunks from centre, outside the resident radius"
                );
            }
        }
    }
}

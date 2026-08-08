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

use super::chunk::{Chunk, WindowRows};
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

    /// Window slots the current shift has to fill. Scratch for
    /// [`WindowManager::load_incoming`], reused across shifts.
    incoming_slots: Vec<(i32, i32)>,
    /// The same slots as absolute chunk coordinates, which is what
    /// [`ChunkStore::prefetch`] takes. Scratch, reused.
    incoming_chunks: Vec<(i32, i32)>,
    /// The window origin [`WindowManager::warm_ahead`] last generated for, so a
    /// player loitering on the dead-zone boundary does not re-walk the same
    /// coordinates every tick.
    warmed_for: Option<(i32, i32)>,
    /// The origin ONE MORE shift out that `warm_ahead` has covered, on a later
    /// idle tick than [`WindowManager::warmed_for`]. Two fields rather than a
    /// list because the ladder is exactly two rungs deep — see `warm_ahead` for
    /// why a third would warm chunks that mostly get evicted unused.
    warmed_next: Option<(i32, i32)>,
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
            incoming_slots: Vec::with_capacity((WINDOW_CHUNKS_X * WINDOW_CHUNKS_Y) as usize),
            incoming_chunks: Vec::with_capacity((WINDOW_CHUNKS_X * WINDOW_CHUNKS_Y) as usize),
            warmed_for: None,
            warmed_next: None,
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
        let (drift_x, drift_y) = (pcx - center_x, pcy - center_y);
        if drift_x.abs() <= Self::DEAD && drift_y.abs() <= Self::DEAD {
            // Inside the dead zone: no shift this tick, but the next one is
            // predictable and this is the only moment there is time to pay for it.
            self.warm_ahead(drift_x, drift_y);
            return false;
        }

        // How far the window itself travels, in chunks.
        let (dcx, dcy) = (drift_x, drift_y);
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
        // The window moved, so whatever was warmed for the old position describes
        // a shift that has now happened. The chunks themselves survive in the
        // store's cache — only the bookkeeping resets, and re-walking an edge
        // that is already cached is a contains_key sweep, not generation.
        self.warmed_for = None;
        self.warmed_next = None;
        true
    }

    /// Generate the chunks the NEXT shift will want, while there is still time.
    ///
    /// # Why this is worth doing at all
    ///
    /// A shift is the one event in the sim that is not amortised, and batching its
    /// generation across a thread pool already took it from 722 µs to 495. What is
    /// left is not throughput but LATENCY: the work still happens inside the single
    /// tick that crossed the dead zone, and a player walking sideways pays all of
    /// it at once, repeatedly.
    ///
    /// [`WindowManager::DEAD`] is exactly the lookahead needed to fix that, and it
    /// is already there for an unrelated reason — it exists so a player loitering
    /// on a chunk boundary does not thrash the buffer. A drift of one chunk means
    /// a shift is one chunk away; generating its incoming edge now moves the cost
    /// off the frame that can least afford it and onto one that is doing nothing.
    ///
    /// # What it predicts
    ///
    /// The shift that fires when the drift reaches `DEAD + 1` in whichever
    /// direction it is already going. That is the common case by a wide margin —
    /// a player walks — and being wrong is cheap: the chunks land in the store's
    /// cache, which is where a correct guess would have put them, and
    /// `evict_beyond` reclaims them on the next real shift like any other.
    ///
    /// It cannot predict a teleport, and does not try. Nothing survives one, so
    /// there is no edge to warm.
    fn warm_ahead(&mut self, drift_x: i32, drift_y: i32) {
        // Dead centre: no direction to guess, and the shift is at least two
        // chunks of travel away.
        if drift_x == 0 && drift_y == 0 {
            return;
        }
        let step = |d: i32| {
            if d == 0 {
                0
            } else {
                d.signum() * (Self::DEAD + 1)
            }
        };
        let (dcx, dcy) = (step(drift_x), step(drift_y));
        let ahead = (self.origin_chunk_x + dcx, self.origin_chunk_y + dcy);

        // One edge per idle tick, never two: the whole point of warming is to
        // move generation onto ticks that can afford it, and a tick that
        // generates two edges is the spike this function exists to remove.
        if self.warmed_for != Some(ahead) {
            self.warmed_for = Some(ahead);
            self.warm_edge(ahead, dcx, dcy);
            return;
        }

        // The shift after next, on a LATER idle tick. A walking player has a
        // couple of hundred idle ticks per shift, so the second rung costs
        // nothing extra in practice — but it is what a SPRINT hits: consecutive
        // shifts arriving with too few idle ticks between them to warm each
        // edge as it comes. Two rungs cover that burst; a third would warm
        // chunks two full shifts from being wanted, which `evict_beyond`
        // reclaims mostly unused the moment the player turns.
        let next = (self.origin_chunk_x + 2 * dcx, self.origin_chunk_y + 2 * dcy);
        if self.warmed_next != Some(next) {
            self.warmed_next = Some(next);
            self.warm_edge(next, dcx, dcy);
        }
    }

    /// Generate one predicted incoming edge into the store's cache: the chunks a
    /// shift of `(dcx, dcy)` will load when the window origin reaches `origin`.
    fn warm_edge(&mut self, origin: (i32, i32), dcx: i32, dcy: i32) {
        self.incoming_slots(dcx, dcy);
        self.incoming_chunks.clear();
        self.incoming_chunks.extend(
            self.incoming_slots
                .iter()
                .map(|&(ci, cj)| (origin.0 + ci, origin.1 + cj)),
        );
        self.store.prefetch(&self.incoming_chunks);
    }

    /// Window slots a shift of `(dcx, dcy)` would leave uncovered, into
    /// [`WindowManager::incoming_slots`].
    ///
    /// Shared by the shift itself and by [`WindowManager::warm_ahead`], so the
    /// edge that is warmed is by construction the edge that will be loaded. Two
    /// copies of this walk would be two chances to warm the wrong sixteen chunks
    /// and never notice, because the result would still be correct — just slow.
    fn incoming_slots(&mut self, dcx: i32, dcy: i32) {
        let (i_lo, i_hi) = Self::survivor_range(dcx, WINDOW_CHUNKS_X);
        let (j_lo, j_hi) = Self::survivor_range(dcy, WINDOW_CHUNKS_Y);
        self.incoming_slots.clear();
        for cj in 0..WINDOW_CHUNKS_Y {
            let row_covered = cj >= j_lo - dcy && cj < j_hi - dcy;
            for ci in 0..WINDOW_CHUNKS_X {
                if row_covered && ci >= i_lo - dcx && ci < i_hi - dcx {
                    continue;
                }
                self.incoming_slots.push((ci, cj));
            }
        }
    }

    fn set_origin(&mut self, grid: &mut CellGrid, ocx: i32, ocy: i32) {
        self.origin_chunk_x = ocx;
        self.origin_chunk_y = ocy;
        grid.set_origin(ocx * CHUNK_CELLS, ocy * CHUNK_CELLS);
    }

    // --- Whole-window paths (first fill, and jumps with no overlap) ------------

    /// Fill every slot in the window.
    ///
    /// The generates run first, as one parallel batch, and the blit-in stays
    /// serial. That split is deliberate: the generates are pure and independent
    /// (see [`ChunkStore::prefetch`]), whereas `blit_in` writes into one `&mut
    /// CellGrid` and `wake_chunk_slot` touches a halo that crosses slot
    /// boundaries — so the copy is a `memcpy` per row against work that costs
    /// orders of magnitude more per chunk.
    ///
    /// This is the load path and the teleport path. A one-chunk shift goes
    /// through [`Self::load_incoming`] and is untouched.
    fn load_all(&mut self, grid: &mut CellGrid) {
        let mut want = Vec::with_capacity((WINDOW_CHUNKS_X * WINDOW_CHUNKS_Y) as usize);
        for cj in 0..WINDOW_CHUNKS_Y {
            for ci in 0..WINDOW_CHUNKS_X {
                want.push((self.origin_chunk_x + ci, self.origin_chunk_y + cj));
            }
        }
        self.store.prefetch(&want);

        for cj in 0..WINDOW_CHUNKS_Y {
            for ci in 0..WINDOW_CHUNKS_X {
                self.load_slot(grid, ci, cj);
            }
        }
    }

    /// Write the whole live window back to the store, then push every diverged
    /// chunk through to persistence.
    ///
    /// The one call a durable backend needs and the trait cannot imply.
    /// `ChunkStore` is a write-back cache: a chunk reaches persistence when the
    /// window shifts far enough to evict it, which means a player who digs a
    /// hole and quits where they stand has written nothing at all. `ChunkStore`'s
    /// own `flush` has carried a comment since the port saying it is "the hook a
    /// durable backend needs on shutdown or manual save"; this is the half of it
    /// that lives up here, because the store cannot see the LIVE grid and the
    /// freshest edits are in exactly that.
    ///
    /// Cheap when nothing changed: `save_slot` only writes a chunk whose
    /// contents differ from what worldgen would produce, and `flush` only
    /// persists the ones marked diverged.
    pub fn flush(&mut self, grid: &CellGrid) {
        self.save_all(grid);
        self.store_mut().flush();
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
    ///
    /// # The batch is what makes a shift affordable
    ///
    /// This used to walk the incoming edge calling [`Self::load_slot`] one chunk
    /// at a time, so every generate ran serially on the calling thread — while
    /// [`ChunkStore::prefetch`], which already had a rayon path, was reached only
    /// from [`Self::load_all`] (the whole-window fill at startup and after a
    /// teleport). The shift is the most expensive event in the game and it is not
    /// amortised: save the trailing edge, memmove the survivors, generate and blit
    /// the leading edge, all inside one tick. It was paying for that generation
    /// on one core.
    ///
    /// Collecting the coordinates first and handing them over in ONE call takes
    /// the parallel path every time: [`WindowManager::DEAD`] is 1, so a recenter
    /// moves at least two chunks along an axis, and two columns of an 11x8 window
    /// is 16 chunks against a `PREFETCH_MIN_PARALLEL` of 8.
    ///
    /// It is behaviour-preserving by construction rather than by inspection —
    /// `prefetch` gives each worker its own [`ChunkGen`](super::worldgen::ChunkGen)
    /// and `worldgen_purity`'s `parallel_matches_serial` requires the result to be
    /// byte-identical to the serial one. Measured on an M3 Pro, same criterion
    /// session: `recenter only` 550.7 µs -> 307.5 µs, and the whole `shift tick`
    /// 686.9 µs -> 471.1 µs.
    ///
    /// Beyond this it is a LATENCY problem rather than a throughput one, and the
    /// stronger fix is to prefetch the leading edge before the dead zone is
    /// crossed — [`Self::recenter`]'s one-chunk dead zone is exactly that
    /// lookahead, and it is free. Not done.
    fn load_incoming(&mut self, grid: &mut CellGrid, dcx: i32, dcy: i32) {
        // The same walk `warm_ahead` uses, so the edge that was warmed is by
        // construction the edge that is loaded. Both buffers are the manager's own
        // scratch, reused across shifts: a walking player shifts the window every
        // few dozen cells of travel, and two heap allocations per shift is exactly
        // the kind of small, regular waste that the rest of this file goes out of
        // its way to avoid.
        self.incoming_slots(dcx, dcy);

        self.incoming_chunks.clear();
        self.incoming_chunks.extend(
            self.incoming_slots
                .iter()
                .map(|&(ci, cj)| (self.origin_chunk_x + ci, self.origin_chunk_y + cj)),
        );
        self.store.prefetch(&self.incoming_chunks);

        // Now warm, so each of these is a cache hit and a memcpy per row rather
        // than a heightmap, a cave field and three decorator passes.
        //
        // `take` and put back, because `load_slot` needs `&mut self` and the loop
        // is reading a field of the same `self`. The `Vec` is moved out and
        // returned with its allocation intact, so this costs three pointer writes
        // and no allocation — which is what a scratch buffer on a method that also
        // mutates is, spelt honestly.
        let slots = std::mem::take(&mut self.incoming_slots);
        for &(ci, cj) in &slots {
            self.load_slot(grid, ci, cj);
        }
        self.incoming_slots = slots;
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
    // Walls travel with the cells they are behind, for the reason `temp` does
    // below: a plane left at the old offset smears sideways by the shift on every
    // recentre, and a backdrop is exactly the sort of thing nobody would notice
    // sliding until it had slid a long way.
    grid.back.copy_within(src..end, dst);
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
        grid.back[dst..dst + n].copy_from_slice(&chunk.back[src..src + n]);
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
            &WindowRows {
                material: &grid.material,
                flags: &grid.flags,
                aux: &grid.aux,
                temp: &grid.temp,
                back: &grid.back,
            },
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
    use crate::sim::coords::WorldCell;
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

    /// Walls survive a walk away and back, exactly as matter does.
    ///
    /// # The trap this exists for
    ///
    /// A plane added to `CellGrid` has to be threaded through FOUR places or the
    /// world quietly corrupts: `blit_in`, `extract_out`, `shift_row`, and
    /// `Chunk`. Miss `shift_row` and the plane stays at the old offset while
    /// everything else moves, so every recentre smears it sideways by the shift.
    /// Miss `extract_out` or the divergence vote and the edit is simply gone the
    /// moment the chunk is evicted.
    ///
    /// `a_long_walk_leaves_the_window_identical_to_a_fresh_generate` cannot catch
    /// any of that: it compares `grid.material` against a fresh generate, and the
    /// back plane is neither generated nor material. Nothing else looks at this
    /// plane at all yet, which is exactly why it needs its own test now — an
    /// invisible plane that is already broken is the worst thing to inherit.
    ///
    /// The walls are written by hand because worldgen does not produce them yet.
    /// That is the point: this pins the STREAMING, and it will keep pinning it
    /// when the generator starts filling the plane.
    #[test]
    fn the_back_plane_survives_a_walk_away_and_back() {
        let mut grid = fresh_grid();
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);

        // A diagonal stripe of distinct wall ids across the middle of the window,
        // so a smear along either axis lands the wrong id rather than merely the
        // wrong count. Ids are arbitrary non-zero material codes; nothing reads
        // them yet, and a wall is a `CellId` like any other.
        let origin_x = grid.origin_cell_x();
        let origin_y = grid.origin_cell_y();
        let mut wrote: Vec<(WorldCell, CellId)> = Vec::new();
        for i in 0..40i32 {
            let w = WorldCell::new(origin_x + 20 + i, origin_y + 20 + i);
            let id = (i + 1) as CellId;
            grid.set_back_world(w, id);
            wrote.push((w, id));
        }
        for &(w, id) in &wrote {
            assert_eq!(grid.get_back_world(w), id, "the write did not land");
        }

        // Far enough that the stripe leaves the window entirely and has to come
        // back from the store rather than from a survivor memmove.
        let far = CHUNK_CELLS * (WINDOW_CHUNKS_X + WINDOW_CHUNKS_Y) * 2;
        assert!(wm.recenter(&mut grid, far, far));
        for &(w, _) in &wrote {
            assert!(
                !grid.is_loaded_world(w),
                "the stripe was supposed to leave the window"
            );
        }

        // And back to where it was written.
        assert!(wm.recenter(&mut grid, 0, 0));
        assert_eq!(
            (grid.origin_cell_x(), grid.origin_cell_y()),
            (origin_x, origin_y),
            "the origin is a pure function of the player cell, so this is the \
             window the stripe was written into"
        );
        for &(w, id) in &wrote {
            assert_eq!(
                grid.get_back_world(w),
                id,
                "a wall at {w:?} came back as the wrong id — the plane was \
                 dropped, smeared or never persisted"
            );
        }
    }

    /// A short shift keeps the walls under the cells they belong to.
    ///
    /// The round trip above goes through the STORE. This one never leaves the
    /// window, so it exercises `shift_row`'s memmove and nothing else — the one
    /// path where a forgotten plane slides against the rest of the world instead
    /// of vanishing outright, which is far harder to notice.
    #[test]
    fn a_short_shift_moves_the_walls_with_the_cells() {
        let mut grid = fresh_grid();
        let mut wm = WindowManager::new(ChunkStore::new(SEED));
        wm.init(&mut grid, 0, 0);

        let origin_x = grid.origin_cell_x();
        let origin_y = grid.origin_cell_y();
        let marks: Vec<(WorldCell, CellId)> = (0..12i32)
            .map(|i| {
                (
                    WorldCell::new(origin_x + 100 + i * 3, origin_y + 100),
                    (i + 1) as CellId,
                )
            })
            .collect();
        for &(w, id) in &marks {
            grid.set_back_world(w, id);
        }

        // Two chunks along, which is the smallest shift `recenter` will make and
        // the one a walking player produces constantly.
        assert!(wm.recenter(&mut grid, CHUNK_CELLS * 2, 0));
        for &(w, id) in &marks {
            assert_eq!(
                grid.get_back_world(w),
                id,
                "a wall moved relative to the world across a survivor memmove"
            );
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

//! The world as a dense grid of material cells.
//!
//! This is the single shared data structure — collision, rendering and the
//! Noita-style automata all address it.
//!
//! # What changed in the port
//!
//! The TypeScript ran the sim on a Web Worker and the grid's four cell planes
//! lived in `SharedArrayBuffer`s with NO synchronisation of any kind: the main
//! thread read cells while the worker wrote them, and a torn cell was accepted
//! as cosmetically harmless. That is legal in JavaScript and undefined
//! behaviour in Rust, so none of it survives.
//!
//! Here the sim is a system on the fixed 120 Hz schedule (every second tick, so
//! [`SIM_HZ`](crate::config::SIM_HZ) = 60) that holds `&mut CellGrid`. The grid
//! is a plain owned struct of `Vec`s — no shared buffers, no atomics, no
//! `unsafe` — and the compiler enforces what the TypeScript merely hoped for:
//! nothing reads the planes while the sim is writing them. Consequently:
//!
//!   - `CellGrid.shared()` / `CellGrid.fromShared()` are gone. There is one
//!     grid, and whoever wants it borrows it.
//!   - the `CellBuffers` / `SharedCellSABs` plumbing is gone with them.
//!   - the shared control block (an `Int32Array` over a SAB, slots
//!     `CTRL_ORIGIN_X` / `CTRL_ORIGIN_Y` / `CTRL_SHIFT_GEN`) is three plain
//!     `i32` fields. It existed to let two threads agree on the window origin
//!     without a message hop; with one owner there is nothing to agree with.

use bitflags::bitflags;

use super::coords::{LocalCell, WorldCell};
use super::materials::{CellId, EMPTY};
use crate::config::CHUNK_CELLS;

bitflags! {
    /// Per-cell flag bits (stored in `flags`). The automata owns MOVED (so a
    /// cell is only processed once per tick); BURNING marks a cell that is
    /// currently on fire. Bits 4+ are free for feature code.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct CellFlags: u8 {
        const MOVED = 1 << 0;
        const BURNING = 1 << 1;
    }
}

impl CellFlags {
    /// The subset of flag bits that are per-tick sim bookkeeping rather than
    /// world state. Anything comparing two versions of a cell to decide whether
    /// the world actually changed (see `Chunk::absorb_row`) must mask these out
    /// first, or a cell the automata merely swept past would read as an edit.
    ///
    /// Only MOVED qualifies today: it is set as a cell acts and cleared by the
    /// next tick's pre-pass, so it says nothing durable. BURNING is deliberately
    /// NOT here — a burning cell stays burning across ticks, its aux holds the
    /// burn timer, and losing the bit would silently put out fires across a
    /// window shift.
    ///
    /// Add new transient bits to this mask when you define them; that is the
    /// whole point of it living next to the definitions instead of at each
    /// consumer.
    pub const VOLATILE_MASK: CellFlags = CellFlags::MOVED;
}

/// A per-chunk ACTIVITY BOX: an INCLUSIVE cell rectangle in window-local coords.
///
/// The TypeScript kept the four corners in four parallel `Int32Array`s (eight
/// in total, counting the double buffer), which meant four strided loads from
/// four different cache lines to read one chunk's box. One `Vec` of this struct
/// is one allocation and one cache line per chunk instead, and it makes the
/// empty sentinel a single named value rather than four magic literals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Box2 {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Box2 {
    /// The "nothing here" box: lo above hi, so any union with a real rect wins
    /// on every axis and any emptiness test sees it immediately.
    pub const EMPTY: Box2 = Box2 {
        x0: i32::MAX,
        y0: i32::MAX,
        x1: -1,
        y1: -1,
    };

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.x1 < self.x0 || self.y1 < self.y0
    }

    /// Grow to cover an inclusive rect.
    #[inline]
    fn union(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        if x0 < self.x0 {
            self.x0 = x0;
        }
        if y0 < self.y0 {
            self.y0 = y0;
        }
        if x1 > self.x1 {
            self.x1 = x1;
        }
        if y1 > self.y1 {
            self.y1 = y1;
        }
    }

    /// Slide by (-dx, -dy) cells, or stay empty if it already was.
    #[inline]
    fn translated(self, dx: i32, dy: i32) -> Box2 {
        if self.is_empty() {
            Box2::EMPTY
        } else {
            Box2 {
                x0: self.x0 - dx,
                y0: self.y0 - dy,
                x1: self.x1 - dx,
                y1: self.y1 - dy,
            }
        }
    }
}

/// One awake chunk's sweep job, as handed to the automata.
///
/// `x0..x1` / `y0..y1` are HALF-OPEN window-local cell bounds — the accumulated
/// activity box, not the whole chunk — matching the exclusive-hi convention the
/// TypeScript callback used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AwakeChunk {
    /// Row-major chunk slot index, `chunk_cy * chunk_cols + chunk_cx`.
    pub slot: usize,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

/// The world as a dense grid of material cells. This is the single shared data
/// structure — collision, rendering and the Noita-style automata all address it.
///
/// Storage is flat `Vec`s (no per-cell objects), indexed row-major by
/// `cy * cols + cx`. Two coarse chunk masks track work:
///   - `dirty`  — a chunk changed since the renderer last drew it (render cull).
///   - `awake`  — a chunk had motion last tick and must be simulated again.
///
/// Chunks that hold nothing moving go to sleep, so an idle world costs almost
/// nothing to simulate or redraw.
///
/// FROZEN CONTRACT: the public method signatures below are depended on by the
/// sim, render, physics and tool modules. Add to it, don't reshape it.
pub struct CellGrid {
    /// Width / height of the world in cells.
    cols: i32,
    rows: i32,
    /// Width / height of the world in chunks.
    chunk_cols: i32,
    chunk_rows: i32,

    /// Material id per cell (0 = empty).
    ///
    /// The four planes are public because the renderer, the collision query and
    /// the chunk blitter all address cells in bulk, exactly as they did through
    /// the TypeScript's `readonly` typed-array fields — the binding is fixed,
    /// the contents are not. The DIMENSIONS above are private with accessors,
    /// which is the other half of that same `readonly`: resizing a plane or
    /// moving `cols` out from under it would desynchronise every index in the
    /// struct at once.
    pub material: Vec<CellId>,
    /// Bit flags per cell (see [`CellFlags`]).
    pub flags: Vec<CellFlags>,
    /// Generic per-cell scratch counter — burn timer, gas lifetime, growth generation.
    pub aux: Vec<u16>,
    /// Per-cell temperature, 0..255, where 0 is ambient. Emitters (lava/fire/ember)
    /// hold themselves at a floor, everything else diffuses toward its neighbours
    /// and decays back to ambient. Only the automata writes it — see the heat pass
    /// in `automata`, which runs over awake chunks only so a settled world does
    /// no thermal work at all.
    pub temp: Vec<u8>,

    /// Window origin: this grid is a WINDOW into an unbounded world, and its
    /// top-left cell maps to absolute cell (`origin_cell_x`, `origin_cell_y`).
    ///
    /// In the TypeScript these two lived in a `SharedArrayBuffer` control block
    /// so the worker could publish the origin and the main thread could read it
    /// without a message hop. There is one owner here, so they are plain fields.
    origin_cell_x: i32,
    origin_cell_y: i32,
    /// Bumped once per completed window shift. Anything caching a world→local
    /// translation compares this to know its cache is stale.
    shift_gen: i32,

    /// Per-chunk: needs redraw. Renderer clears after blitting.
    dirty: Vec<bool>,
    /// Per-chunk: simulate this tick. Swapped from `awake_next` each tick.
    awake_this: Vec<bool>,
    /// Per-chunk: motion happened, so wake next tick.
    awake_next: Vec<bool>,

    /// Per-chunk ACTIVITY BOX: the inclusive cell rectangle inside each chunk
    /// that actually needs sweeping, in absolute (window-local) cell coords.
    ///
    /// Sleep used to be all-or-nothing per chunk, which meant one falling grain
    /// bought a full 32x32 = 1024-cell sweep — twice, once per automata pass.
    /// Since the automata already calls `wake` on every cell it touches, the
    /// same calls can accumulate a bounding box for free, and the sweep shrinks
    /// to the part of the chunk where something is going on. In a world that is
    /// mostly settled with a few active fronts this is the difference between
    /// "cost scales with the number of awake chunks" and "cost scales with the
    /// amount of moving matter", which is the behaviour a falling-sand sim
    /// actually wants.
    ///
    /// The box carries a ONE-CELL HALO around whatever was woken (see
    /// [`CellGrid::wake_rect`]), because every rule in the automata is a
    /// nearest-neighbour rule: a cell that changed can only affect, and be
    /// affected by, cells one step away. The halo is what replaces the old "a
    /// cell on a chunk border also wakes the neighbour chunk" special case — a
    /// halo that crosses a chunk edge lands in the neighbour's box by
    /// construction, including diagonally.
    ///
    /// Double-buffered exactly like the awake masks: `box_this` is what this
    /// tick sweeps, `box_next` is what the current tick's writes are
    /// accumulating.
    box_this: Vec<Box2>,
    box_next: Vec<Box2>,
}

impl CellGrid {
    /// Build a grid of `cols` x `rows` cells, all empty.
    ///
    /// The TypeScript constructor also took optional shared cell buffers and a
    /// shared control array; see the module docs for why neither exists here.
    pub fn new(cols: i32, rows: i32) -> CellGrid {
        assert!(cols > 0 && rows > 0, "grid must have positive extent");
        let n = (cols as usize) * (rows as usize);
        // Round up: a window that is not a whole number of chunks still needs a
        // slot for its ragged last column/row. (`i32::div_ceil` is still
        // unstable; `cols`/`rows` are asserted positive above, so the classic
        // form is exact here.)
        let chunk_cols = (cols + CHUNK_CELLS - 1) / CHUNK_CELLS;
        let chunk_rows = (rows + CHUNK_CELLS - 1) / CHUNK_CELLS;
        let c = (chunk_cols as usize) * (chunk_rows as usize);

        CellGrid {
            cols,
            rows,
            chunk_cols,
            chunk_rows,
            material: vec![EMPTY; n],
            flags: vec![CellFlags::empty(); n],
            aux: vec![0; n],
            temp: vec![0; n],
            origin_cell_x: 0,
            origin_cell_y: 0,
            shift_gen: 0,
            dirty: vec![false; c],
            awake_this: vec![false; c],
            awake_next: vec![false; c],
            box_this: vec![Box2::EMPTY; c],
            // `box_next` starts at the "empty" sentinel — the accumulating
            // boxes are always reset, never zeroed, or chunk (0,0) would think
            // cell (0,0) was active from the first tick.
            box_next: vec![Box2::EMPTY; c],
        }
    }

    /// Width of the world in cells.
    #[inline]
    pub const fn cols(&self) -> i32 {
        self.cols
    }
    /// Height of the world in cells.
    #[inline]
    pub const fn rows(&self) -> i32 {
        self.rows
    }
    /// Width of the world in chunks.
    #[inline]
    pub const fn chunk_cols(&self) -> i32 {
        self.chunk_cols
    }
    /// Height of the world in chunks.
    #[inline]
    pub const fn chunk_rows(&self) -> i32 {
        self.chunk_rows
    }

    // --- World ↔ window-local coordinates -------------------------------------
    // The grid stores a window; automata/get/set address it in LOCAL coords, while
    // render/collision/edits/spawn work in ABSOLUTE world coords and translate
    // through the origin below.

    #[inline]
    pub const fn origin_cell_x(&self) -> i32 {
        self.origin_cell_x
    }
    #[inline]
    pub const fn origin_cell_y(&self) -> i32 {
        self.origin_cell_y
    }

    /// Move the window's top-left to an absolute CELL coordinate.
    ///
    /// The window manager calls this as part of a shift, after
    /// [`CellGrid::shift_chunks`] has scrolled the per-chunk bookkeeping — the
    /// order matters, because the shift is expressed in the OLD frame.
    #[inline]
    pub fn set_origin(&mut self, cell_x: i32, cell_y: i32) {
        self.origin_cell_x = cell_x;
        self.origin_cell_y = cell_y;
    }

    /// How many window shifts have completed. Cache-invalidation stamp.
    #[inline]
    pub const fn shift_gen(&self) -> i32 {
        self.shift_gen
    }

    /// Record that a window shift finished.
    #[inline]
    pub fn bump_shift_gen(&mut self) {
        self.shift_gen = self.shift_gen.wrapping_add(1);
    }

    #[inline]
    pub const fn world_to_local_x(&self, wcx: i32) -> i32 {
        wcx - self.origin_cell_x
    }
    #[inline]
    pub const fn world_to_local_y(&self, wcy: i32) -> i32 {
        wcy - self.origin_cell_y
    }

    /// Translate an absolute cell into a window-local one. The only way to get a
    /// [`LocalCell`] out of a [`WorldCell`], which is the point of the types.
    #[inline]
    pub const fn to_local(&self, w: WorldCell) -> LocalCell {
        LocalCell::new(self.world_to_local_x(w.x), self.world_to_local_y(w.y))
    }

    /// Is an absolute cell currently inside the loaded window?
    #[inline]
    pub const fn is_loaded_world(&self, w: WorldCell) -> bool {
        let l = self.to_local(w);
        self.in_bounds(l.x, l.y)
    }

    /// Material at an absolute cell (0 if outside the window).
    #[inline]
    pub fn get_world(&self, w: WorldCell) -> CellId {
        let l = self.to_local(w);
        self.get(l.x, l.y)
    }

    #[inline]
    pub fn is_empty_world(&self, w: WorldCell) -> bool {
        self.get_world(w) == EMPTY
    }

    /// Set an absolute cell (no-op if outside the window).
    #[inline]
    pub fn set_world(&mut self, w: WorldCell, id: CellId) {
        let l = self.to_local(w);
        self.set(l.x, l.y, id);
    }

    // --- Cell coordinate helpers ---------------------------------------------

    /// Flat index for a cell. Caller guarantees in-bounds.
    #[inline]
    pub const fn idx(&self, cx: i32, cy: i32) -> usize {
        (cy * self.cols + cx) as usize
    }

    #[inline]
    pub const fn in_bounds(&self, cx: i32, cy: i32) -> bool {
        cx >= 0 && cy >= 0 && cx < self.cols && cy < self.rows
    }

    // The pixel→cell conversion the TypeScript exposed as the static
    // `CellGrid.cellAt` is `crate::config::cell_at` (re-exported by
    // `sim::coords`): it is a property of `CELL_SIZE`, not of any one grid, and
    // it was already duplicated verbatim in `coords.ts`.

    // --- Cell access ----------------------------------------------------------

    /// Material id at a cell, or 0 (empty) when out of bounds.
    #[inline]
    pub fn get(&self, cx: i32, cy: i32) -> CellId {
        if !self.in_bounds(cx, cy) {
            return EMPTY;
        }
        self.material[self.idx(cx, cy)]
    }

    #[inline]
    pub fn is_empty(&self, cx: i32, cy: i32) -> bool {
        self.get(cx, cy) == EMPTY
    }

    /// Set a cell's material and wake it for sim + render. Out-of-bounds is a
    /// no-op. Clears per-cell flags/aux so a reused cell starts clean.
    ///
    /// `temp` is deliberately NOT cleared: temperature belongs to the LOCATION,
    /// not to the matter occupying it. Ice melting into water must hand the water
    /// the heat that melted it (otherwise it would instantly refreeze the
    /// gradient), and a cell placed inside a furnace should be hot immediately.
    /// Matter that MOVES carries its heat with it — see [`CellGrid::swap`].
    pub fn set(&mut self, cx: i32, cy: i32, id: CellId) {
        if !self.in_bounds(cx, cy) {
            return;
        }
        let i = self.idx(cx, cy);
        self.material[i] = id;
        self.flags[i] = CellFlags::empty();
        self.aux[i] = 0;
        self.wake(cx, cy);
    }

    /// Swap two cells (their material, flags, aux and temperature move together)
    /// and wake both. The automata's primitive move — sand falling is a swap with
    /// the empty cell below it. Heat rides along so a blob of molten matter stays
    /// hot as it flows.
    pub fn swap(&mut self, ax: i32, ay: i32, bx: i32, by: i32) {
        if !self.in_bounds(ax, ay) || !self.in_bounds(bx, by) {
            return;
        }
        let ia = self.idx(ax, ay);
        let ib = self.idx(bx, by);

        self.material.swap(ia, ib);
        self.flags.swap(ia, ib);
        self.aux.swap(ia, ib);
        self.temp.swap(ia, ib);

        // One rect, not two wakes. The two cells of a swap are always adjacent
        // (orthogonally or diagonally), and the halo of their bounding rect is
        // exactly the union of their two individual halos — so this marks the same
        // cells while walking the chunk-clip loop once instead of twice. Swaps are
        // the single most frequent write in the tick, so that halving is real.
        self.wake_rect(ax.min(bx), ay.min(by), ax.max(bx), ay.max(by));
    }

    // --- Chunk waking ---------------------------------------------------------

    /// Flag the chunk holding this cell as dirty (redraw) and mark the cell — plus
    /// its one-cell halo — as needing simulation next tick.
    ///
    /// The halo does two jobs at once. It is what lets a neighbour REACT to this
    /// write (fire igniting the wood beside it, a grain falling into the hole a
    /// liquid just left, heat stepping outward one cell per tick), and, where the
    /// halo spills over a chunk edge, it is what keeps a flow crossing a chunk seam
    /// instead of stalling at the border — the job the old explicit border test
    /// did, now falling out of the geometry for free, diagonals included.
    #[inline]
    pub fn wake(&mut self, cx: i32, cy: i32) {
        self.wake_rect(cx, cy, cx, cy);
    }

    /// Mark an INCLUSIVE cell rectangle (plus its one-cell halo) for simulation.
    ///
    /// Multi-cell moves need this. A liquid spreading sideways jumps several cells
    /// in one step, and the cells it vacated along the way are every bit as much a
    /// change as its endpoints: matter resting on top of that run must be woken or
    /// it hangs in the air until something unrelated happens to wake its chunk.
    /// [`CellGrid::wake`] is just the 1x1 case.
    pub fn wake_rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        // Grow by the halo and clamp to the world in one step; an entirely
        // out-of-bounds rect collapses and is rejected by the lo>hi test.
        // (Saturating, so a caller passing an extreme sentinel cannot overflow
        // its way past the clamp — JavaScript's doubles could not.)
        let rx0 = x0.saturating_sub(1).max(0);
        let ry0 = y0.saturating_sub(1).max(0);
        let rx1 = x1.saturating_add(1).min(self.cols - 1);
        let ry1 = y1.saturating_add(1).min(self.rows - 1);
        if rx0 > rx1 || ry0 > ry1 {
            return;
        }

        // Both ends are clamped non-negative above, so truncating division is
        // floor division here and no `floor_div` is needed in the hot path.
        let chx0 = rx0 / CHUNK_CELLS;
        let chx1 = rx1 / CHUNK_CELLS;
        let chy0 = ry0 / CHUNK_CELLS;
        let chy1 = ry1 / CHUNK_CELLS;

        let cc = self.chunk_cols;
        for chy in chy0..=chy1 {
            // Clip the rect to this chunk row's cell band.
            let c_lo = chy * CHUNK_CELLS;
            let c_hi = c_lo + CHUNK_CELLS - 1;
            let by0 = ry0.max(c_lo);
            let by1 = ry1.min(c_hi);
            let row_base = chy * cc;

            for chx in chx0..=chx1 {
                let d_lo = chx * CHUNK_CELLS;
                let d_hi = d_lo + CHUNK_CELLS - 1;
                let bx0 = rx0.max(d_lo);
                let bx1 = rx1.min(d_hi);

                let k = (row_base + chx) as usize;
                self.awake_next[k] = true;
                // Redraw rides along with the wake. It over-reports by the halo (a
                // border write dirties the neighbour too), which is harmless: `dirty` is
                // a render cull hint, and the failure mode of over-reporting is a
                // redundant repaint rather than a stale one.
                self.dirty[k] = true;
                self.box_next[k].union(bx0, by0, bx1, by1);
            }
        }
    }

    /// Wake one whole chunk slot — what a freshly blitted-in chunk needs.
    ///
    /// `blit_in` writes the cell planes directly rather than through
    /// [`CellGrid::set`], so nothing along that path wakes anything. This is the
    /// hook that replaces the blanket `wake_all` a shift used to do: only the
    /// leading-edge chunks that were actually loaded get woken, and
    /// [`CellGrid::wake_rect`]'s halo spills one cell into the neighbouring
    /// survivors so matter can flow straight across the new seam.
    pub fn wake_chunk_slot(&mut self, chunk_cx: i32, chunk_cy: i32) {
        let x0 = chunk_cx * CHUNK_CELLS;
        let y0 = chunk_cy * CHUNK_CELLS;
        let x1 = (x0 + CHUNK_CELLS).min(self.cols) - 1;
        let y1 = (y0 + CHUNK_CELLS).min(self.rows) - 1;
        if x1 < x0 || y1 < y0 {
            return;
        }
        self.wake_rect(x0, y0, x1, y1);
    }

    /// Wake every chunk (e.g. after a full-world edit or first load settle).
    ///
    /// This deliberately does NOT touch `temp`. It used to: the stored Chunk had no
    /// temperature plane, so after a window shift the temperatures sitting at each
    /// index belonged to the slot's OLD occupants and had to be cleared. Chunk now
    /// carries `temp` through both the blit-in and extract-out paths, and the
    /// incremental shift memmoves it alongside the other planes, so every cell in
    /// the window is covered by either a survivor move or a fresh blit. The heat
    /// field is therefore already correct for the new contents by the time this is
    /// called, and wiping it would throw away a real thermal field every time the
    /// player walked far enough to recentre.
    pub fn wake_all(&mut self) {
        self.awake_next.fill(true);
        self.dirty.fill(true);
        // Every chunk's activity box opens to the whole chunk, clipped to the world
        // so the ragged right/bottom edge of a window that isn't a whole number of
        // chunks never hands the sweep an out-of-range column.
        let last_x = self.cols - 1;
        let last_y = self.rows - 1;
        for chy in 0..self.chunk_rows {
            for chx in 0..self.chunk_cols {
                let k = (chy * self.chunk_cols + chx) as usize;
                let x0 = chx * CHUNK_CELLS;
                let y0 = chy * CHUNK_CELLS;
                self.box_next[k] = Box2 {
                    x0,
                    y0,
                    x1: (x0 + CHUNK_CELLS - 1).min(last_x),
                    y1: (y0 + CHUNK_CELLS - 1).min(last_y),
                };
            }
        }
    }

    // --- Sim tick iteration ---------------------------------------------------

    /// Promote the chunks woken last tick into the set to simulate now, and reset
    /// the accumulator. Call once before iterating awake chunks each sim tick.
    pub fn begin_sim_tick(&mut self) {
        std::mem::swap(&mut self.awake_this, &mut self.awake_next);
        self.awake_next.fill(false);

        std::mem::swap(&mut self.box_this, &mut self.box_next);
        self.box_next.fill(Box2::EMPTY);
    }

    /// The chunks to simulate this tick, in ROW-MAJOR SLOT ORDER, each with the
    /// ACTIVE cell bounds inside it — the accumulated activity box, not the whole
    /// chunk. The caller runs the automata over those cells and calls
    /// [`CellGrid::wake`] on anything that moves, seeding the next tick's boxes.
    ///
    /// THE ORDER IS DETERMINISM-CRITICAL. The automata consumes one global RNG
    /// stream as it sweeps, so the sequence in which chunks are visited decides
    /// which cell gets which random number; change it and an identical world
    /// evolves differently. The flat slot index IS the row-major order (slot =
    /// `chunk_cy * chunk_cols + chunk_cx`), so a single pass over the mask
    /// reproduces the TypeScript's nested `chy`/`chx` loops exactly, and there is
    /// no ordering decision left implicit anywhere else.
    ///
    /// This returns a collected list rather than taking a callback — the TypeScript
    /// `forEachAwakeChunk(cb)` shape — because the callback needs `&mut CellGrid`
    /// to actually simulate, and it cannot have that while an iterator borrows the
    /// masks. Snapshotting first is not a behaviour change: the callback only ever
    /// writes `awake_next` / `box_next` (via `wake`), never the `*_this` buffers
    /// being iterated, which is exactly why the masks are double-buffered.
    pub fn awake_chunks(&self) -> Vec<AwakeChunk> {
        let mut out = Vec::new();
        self.awake_chunks_into(&mut out);
        out
    }

    /// [`CellGrid::awake_chunks`] into a caller-owned buffer, so the sim schedule
    /// does not allocate once per tick.
    pub fn awake_chunks_into(&self, out: &mut Vec<AwakeChunk>) {
        out.clear();
        for (slot, &awake) in self.awake_this.iter().enumerate() {
            if !awake {
                continue;
            }
            let b = self.box_this[slot];
            let x1 = b.x1 + 1;
            if b.x0 >= x1 {
                continue; // awake but empty box — nothing to sweep
            }
            out.push(AwakeChunk {
                slot,
                x0: b.x0,
                y0: b.y0,
                x1,
                y1: b.y1 + 1,
            });
        }
    }

    /// Chunks scheduled for simulation this tick — instrumentation only.
    pub fn awake_chunk_count(&self) -> usize {
        self.awake_this.iter().filter(|&&a| a).count()
    }

    // --- Window shift ---------------------------------------------------------

    /// Scroll the per-chunk bookkeeping to follow a window shift of (dcx,dcy)
    /// CHUNKS, so a shift no longer costs a full-window wake.
    ///
    /// The window shift already memmoves the surviving cells to their new offset
    /// (see `WindowManager::shift_overlap`). Everything the sim knows about those
    /// cells — which chunks were awake, and the activity box inside each — is
    /// still perfectly valid; it is just indexed by a slot that has moved. Scroll
    /// it by the same amount and a survivor that was mid-flow keeps flowing, while
    /// a survivor that was asleep stays asleep.
    ///
    /// Without this the only correct option is [`CellGrid::wake_all`], and that
    /// turns every window shift into one tick that sweeps all 90k cells of the
    /// window twice — measured at ~55x an ordinary tick, i.e. exactly the hitch the
    /// incremental shift was built to avoid, reintroduced one layer up. Slots the
    /// shift leaves uncovered are cleared here and woken by the loader as they are
    /// filled.
    ///
    /// Boxes hold window-LOCAL cell coordinates, and the translation is exact: a
    /// source chunk's cell span maps onto the destination chunk's span one-to-one,
    /// so a plain subtraction needs no re-clipping.
    ///
    /// This touches the per-chunk side tables ONLY. The cell planes are moved by
    /// the window manager, which owns the chunk store the outgoing cells are saved
    /// to and the incoming cells are loaded from.
    pub fn shift_chunks(&mut self, dcx: i32, dcy: i32) {
        if dcx == 0 && dcy == 0 {
            return;
        }
        let cc = self.chunk_cols;
        let cr = self.chunk_rows;
        let n = (cc * cr) as usize;

        // Everything is visually somewhere else now, whatever else is true.
        self.dirty.fill(true);

        if dcx <= -cc || dcx >= cc || dcy <= -cr || dcy >= cr {
            // Nothing survives. The caller refills the whole window and wakes it.
            self.awake_this.fill(false);
            self.awake_next.fill(false);
            self.box_this.fill(Box2::EMPTY);
            self.box_next.fill(Box2::EMPTY);
            return;
        }

        let dx = dcx * CHUNK_CELLS;
        let dy = dcy * CHUNK_CELLS;
        // Slot d reads slot d + delta. Walk toward the source so a slot is always
        // read before it is overwritten — the same memmove discipline the cell
        // planes use, on the chunk index instead of the cell index.
        let delta = dcy * cc + dcx;
        if delta > 0 {
            for d in 0..n {
                self.shift_slot(d, dcx, dcy, dx, dy);
            }
        } else {
            for d in (0..n).rev() {
                self.shift_slot(d, dcx, dcy, dx, dy);
            }
        }
    }

    /// One slot of [`CellGrid::shift_chunks`]: pull slot `(i+dcx, j+dcy)` into
    /// slot `d`, or clear `d` if that source is off the window.
    fn shift_slot(&mut self, d: usize, dcx: i32, dcy: i32, dx: i32, dy: i32) {
        let cc = self.chunk_cols;
        let cr = self.chunk_rows;
        let i = d as i32 % cc;
        let j = d as i32 / cc;
        let si = i + dcx;
        let sj = j + dcy;

        if si < 0 || sj < 0 || si >= cc || sj >= cr {
            self.awake_this[d] = false;
            self.awake_next[d] = false;
            self.box_this[d] = Box2::EMPTY;
            self.box_next[d] = Box2::EMPTY;
            return;
        }

        let s = (sj * cc + si) as usize;
        self.awake_this[d] = self.awake_this[s];
        self.awake_next[d] = self.awake_next[s];
        // Translate the boxes from slot `s` to slot `d`, minus (dx,dy) cells.
        self.box_this[d] = self.box_this[s].translated(dx, dy);
        self.box_next[d] = self.box_next[s].translated(dx, dy);
    }

    // --- Render dirty API -----------------------------------------------------

    #[inline]
    pub fn is_chunk_dirty(&self, chunk_cx: i32, chunk_cy: i32) -> bool {
        self.dirty[(chunk_cy * self.chunk_cols + chunk_cx) as usize]
    }

    #[inline]
    pub fn clear_chunk_dirty(&mut self, chunk_cx: i32, chunk_cy: i32) {
        self.dirty[(chunk_cy * self.chunk_cols + chunk_cx) as usize] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4x4 chunks = 128x128 cells: enough to have interior chunks, chunk seams
    /// and world edges all in one grid.
    fn grid() -> CellGrid {
        CellGrid::new(4 * CHUNK_CELLS, 4 * CHUNK_CELLS)
    }

    fn slot(g: &CellGrid, chx: i32, chy: i32) -> usize {
        (chy * g.chunk_cols + chx) as usize
    }

    fn next_box(g: &CellGrid, chx: i32, chy: i32) -> Box2 {
        g.box_next[slot(g, chx, chy)]
    }

    // --- wake_rect ------------------------------------------------------------

    #[test]
    fn wake_grows_a_one_cell_halo() {
        let mut g = grid();
        g.wake(10, 12);
        assert_eq!(
            next_box(&g, 0, 0),
            Box2 {
                x0: 9,
                y0: 11,
                x1: 11,
                y1: 13
            }
        );
        assert!(g.awake_next[slot(&g, 0, 0)]);
        assert!(g.is_chunk_dirty(0, 0));
    }

    #[test]
    fn halo_clips_to_the_world_edge() {
        let mut g = grid();
        g.wake(0, 0);
        assert_eq!(
            next_box(&g, 0, 0),
            Box2 {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1
            },
            "the halo may not step off the west/north edge"
        );

        let mut g = grid();
        let (lx, ly) = (g.cols() - 1, g.rows() - 1);
        g.wake(lx, ly);
        assert_eq!(
            next_box(&g, g.chunk_cols() - 1, g.chunk_rows() - 1),
            Box2 {
                x0: lx - 1,
                y0: ly - 1,
                x1: lx,
                y1: ly
            }
        );
    }

    #[test]
    fn a_wake_on_a_chunk_corner_spills_into_all_four_neighbours() {
        // The cell at the bottom-right corner of chunk (0,0). Its halo reaches
        // one cell into (1,0), (0,1) and — diagonally — (1,1). This is the
        // whole reason the explicit chunk-border wake could be deleted.
        let mut g = grid();
        let e = CHUNK_CELLS - 1;
        g.wake(e, e);

        assert_eq!(
            next_box(&g, 0, 0),
            Box2 {
                x0: e - 1,
                y0: e - 1,
                x1: e,
                y1: e
            }
        );
        assert_eq!(
            next_box(&g, 1, 0),
            Box2 {
                x0: CHUNK_CELLS,
                y0: e - 1,
                x1: CHUNK_CELLS,
                y1: e
            }
        );
        assert_eq!(
            next_box(&g, 0, 1),
            Box2 {
                x0: e - 1,
                y0: CHUNK_CELLS,
                x1: e,
                y1: CHUNK_CELLS
            }
        );
        assert_eq!(
            next_box(&g, 1, 1),
            Box2 {
                x0: CHUNK_CELLS,
                y0: CHUNK_CELLS,
                x1: CHUNK_CELLS,
                y1: CHUNK_CELLS
            },
            "the diagonal neighbour must be woken too"
        );
        for (chx, chy) in [(2, 0), (0, 2), (2, 2), (3, 3)] {
            assert!(
                next_box(&g, chx, chy).is_empty(),
                "chunk ({chx},{chy}) is out of halo reach"
            );
        }
    }

    #[test]
    fn boxes_never_leave_their_own_chunk() {
        let mut g = grid();
        // A rect that crosses every seam in the grid, plus a couple of single
        // cells right on the seams.
        g.wake_rect(3, 5, g.cols() - 4, g.rows() - 4);
        g.wake(CHUNK_CELLS, CHUNK_CELLS - 1);
        g.wake(0, g.rows() - 1);

        for chy in 0..g.chunk_rows() {
            for chx in 0..g.chunk_cols() {
                let b = next_box(&g, chx, chy);
                if b.is_empty() {
                    continue;
                }
                let lo_x = chx * CHUNK_CELLS;
                let lo_y = chy * CHUNK_CELLS;
                assert!(
                    b.x0 >= lo_x && b.x1 < lo_x + CHUNK_CELLS,
                    "chunk ({chx},{chy}) box {b:?} escaped its column band"
                );
                assert!(
                    b.y0 >= lo_y && b.y1 < lo_y + CHUNK_CELLS,
                    "chunk ({chx},{chy}) box {b:?} escaped its row band"
                );
                assert!(
                    b.x1 < g.cols() && b.y1 < g.rows(),
                    "chunk ({chx},{chy}) box {b:?} escaped the world"
                );
            }
        }
    }

    #[test]
    fn a_fully_out_of_bounds_rect_wakes_nothing() {
        let mut g = grid();
        g.wake_rect(-50, -50, -10, -10);
        g.wake_rect(g.cols() + 4, 0, g.cols() + 9, 4);
        assert_eq!(g.awake_chunk_count(), 0);
        assert!(g.awake_next.iter().all(|&a| !a));
        assert!(g.box_next.iter().all(|b| b.is_empty()));
        assert!(g.dirty.iter().all(|&d| !d));
    }

    #[test]
    fn wake_rect_unions_rather_than_replaces() {
        let mut g = grid();
        g.wake(10, 10);
        g.wake(4, 20);
        assert_eq!(
            next_box(&g, 0, 0),
            Box2 {
                x0: 3,
                y0: 9,
                x1: 11,
                y1: 21
            }
        );
    }

    #[test]
    fn wake_chunk_slot_covers_the_whole_chunk_and_its_seam() {
        let mut g = grid();
        g.wake_chunk_slot(1, 1);
        assert_eq!(
            next_box(&g, 1, 1),
            Box2 {
                x0: CHUNK_CELLS,
                y0: CHUNK_CELLS,
                x1: 2 * CHUNK_CELLS - 1,
                y1: 2 * CHUNK_CELLS - 1
            }
        );
        // One cell of halo lands in each orthogonal neighbour, so matter flows
        // across the new seam instead of stalling on it.
        assert_eq!(next_box(&g, 0, 1).x1, CHUNK_CELLS - 1);
        assert_eq!(next_box(&g, 2, 1).x0, 2 * CHUNK_CELLS);
    }

    #[test]
    fn wake_all_clips_the_ragged_edge_of_a_partial_chunk() {
        // A window that is not a whole number of chunks: the last chunk column
        // and row are partial.
        let mut g = CellGrid::new(CHUNK_CELLS + 5, CHUNK_CELLS + 3);
        g.wake_all();
        assert_eq!(g.chunk_cols(), 2);
        assert_eq!(
            next_box(&g, 1, 1),
            Box2 {
                x0: CHUNK_CELLS,
                y0: CHUNK_CELLS,
                x1: g.cols() - 1,
                y1: g.rows() - 1
            }
        );
        assert!(g.awake_next.iter().all(|&a| a));
    }

    // --- cells ----------------------------------------------------------------

    #[test]
    fn set_clears_flags_and_aux_but_keeps_temperature() {
        let mut g = grid();
        let i = g.idx(5, 5);
        g.flags[i] = CellFlags::BURNING;
        g.aux[i] = 77;
        g.temp[i] = 200;

        g.set(5, 5, 3);
        assert_eq!(g.material[i], 3);
        assert_eq!(g.flags[i], CellFlags::empty());
        assert_eq!(g.aux[i], 0);
        assert_eq!(
            g.temp[i], 200,
            "temperature belongs to the location, not the matter"
        );
        assert!(g.awake_next[slot(&g, 0, 0)]);
    }

    #[test]
    fn set_out_of_bounds_is_a_no_op() {
        let mut g = grid();
        g.set(-1, 5, 9);
        g.set(5, g.rows(), 9);
        assert!(g.material.iter().all(|&m| m == EMPTY));
        assert_eq!(g.awake_chunk_count(), 0);
    }

    #[test]
    fn swap_moves_all_four_planes_together() {
        let mut g = grid();
        let (ax, ay) = (10, 10);
        let (bx, by) = (10, 11);
        let ia = g.idx(ax, ay);
        let ib = g.idx(bx, by);

        g.material[ia] = 7;
        g.flags[ia] = CellFlags::BURNING;
        g.aux[ia] = 300;
        g.temp[ia] = 180;
        g.material[ib] = 2;
        g.flags[ib] = CellFlags::MOVED;
        g.aux[ib] = 4;
        g.temp[ib] = 9;

        g.swap(ax, ay, bx, by);

        assert_eq!((g.material[ia], g.material[ib]), (2, 7));
        assert_eq!(
            (g.flags[ia], g.flags[ib]),
            (CellFlags::MOVED, CellFlags::BURNING)
        );
        assert_eq!((g.aux[ia], g.aux[ib]), (4, 300));
        assert_eq!(
            (g.temp[ia], g.temp[ib]),
            (9, 180),
            "heat rides along with the matter that moved"
        );
    }

    #[test]
    fn swap_wakes_the_halo_of_the_pair_in_one_rect() {
        let mut g = grid();
        // A diagonal swap, given in the "wrong" order on both axes to prove the
        // min/max bounding is what is being woken.
        g.swap(21, 31, 20, 30);
        assert_eq!(
            next_box(&g, 0, 0),
            Box2 {
                x0: 19,
                y0: 29,
                x1: 22,
                y1: 31
            }
        );
        // (20,30)'s halo reaches y=31 in chunk row 0 only; (21,31)'s reaches
        // y=32, which is chunk row 1.
        assert_eq!(
            next_box(&g, 0, 1),
            Box2 {
                x0: 19,
                y0: CHUNK_CELLS,
                x1: 22,
                y1: CHUNK_CELLS
            }
        );
    }

    #[test]
    fn swap_out_of_bounds_is_a_no_op() {
        let mut g = grid();
        g.set(0, 0, 5);
        g.begin_sim_tick();
        g.swap(0, 0, -1, 0);
        assert_eq!(g.get(0, 0), 5);
        assert_eq!(
            g.awake_chunk_count(),
            1,
            "only the earlier set woke a chunk"
        );
        assert!(g.awake_next.iter().all(|&a| !a));
    }

    // --- double-buffered awake masks -----------------------------------------

    #[test]
    fn a_wake_is_simulated_on_the_next_tick_not_this_one() {
        let mut g = grid();
        g.wake(40, 40); // chunk (1,1)
        assert!(
            g.awake_chunks().is_empty(),
            "a wake seeds NEXT tick; this tick's set was decided already"
        );

        g.begin_sim_tick();
        let awake = g.awake_chunks();
        assert_eq!(awake.len(), 1);
        assert_eq!(
            awake[0],
            AwakeChunk {
                slot: slot(&g, 1, 1),
                x0: 39,
                y0: 39,
                x1: 42,
                y1: 42
            },
            "bounds are half-open"
        );

        // Nothing woke during the tick, so the world settles.
        g.begin_sim_tick();
        assert!(g.awake_chunks().is_empty());
        assert_eq!(g.awake_chunk_count(), 0);
    }

    #[test]
    fn a_wake_issued_during_a_tick_is_not_lost_by_the_swap() {
        let mut g = grid();
        g.wake(40, 40);
        g.begin_sim_tick();
        assert_eq!(g.awake_chunks().len(), 1);

        // ...the automata runs and moves something in a different chunk.
        g.wake(100, 100); // chunk (3,3)

        g.begin_sim_tick();
        let awake = g.awake_chunks();
        assert_eq!(awake.len(), 1, "the in-tick wake survived the buffer swap");
        assert_eq!(awake[0].slot, slot(&g, 3, 3));
        assert_eq!((awake[0].x0, awake[0].y0), (99, 99));
    }

    #[test]
    fn awake_chunks_are_visited_in_row_major_slot_order() {
        // The automata pulls from one global RNG stream in this order, so it is
        // part of the simulation's identity, not an implementation detail.
        let mut g = grid();
        for (chx, chy) in [(3, 3), (0, 2), (2, 0), (1, 1)] {
            g.wake(chx * CHUNK_CELLS + 5, chy * CHUNK_CELLS + 5);
        }
        g.begin_sim_tick();
        let slots: Vec<usize> = g.awake_chunks().iter().map(|a| a.slot).collect();
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        assert_eq!(slots, sorted);
        assert_eq!(
            slots,
            vec![
                slot(&g, 2, 0),
                slot(&g, 1, 1),
                slot(&g, 0, 2),
                slot(&g, 3, 3)
            ]
        );
    }

    #[test]
    fn an_awake_chunk_with_an_empty_box_is_skipped() {
        let mut g = grid();
        let k = slot(&g, 1, 1);
        g.awake_next[k] = true; // awake, but nothing accumulated
        g.begin_sim_tick();
        assert_eq!(g.awake_chunk_count(), 1);
        assert!(g.awake_chunks().is_empty());
    }

    #[test]
    fn awake_chunks_into_reuses_the_callers_buffer() {
        let mut g = grid();
        let mut buf = vec![AwakeChunk {
            slot: 99,
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
        }];
        g.wake(5, 5);
        g.begin_sim_tick();
        g.awake_chunks_into(&mut buf);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].slot, 0);
    }

    // --- shift_chunks ---------------------------------------------------------

    /// Give every chunk slot a distinguishable box and awake state.
    fn seed_slots(g: &mut CellGrid) {
        for chy in 0..g.chunk_rows() {
            for chx in 0..g.chunk_cols() {
                let k = (chy * g.chunk_cols() + chx) as usize;
                let x0 = chx * CHUNK_CELLS + 1;
                let y0 = chy * CHUNK_CELLS + 2;
                g.box_this[k] = Box2 {
                    x0,
                    y0,
                    x1: x0 + 3,
                    y1: y0 + 4,
                };
                g.box_next[k] = Box2 {
                    x0: x0 + 1,
                    y0,
                    x1: x0 + 2,
                    y1: y0 + 1,
                };
                g.awake_this[k] = (chx + chy) % 2 == 0;
                g.awake_next[k] = chx == chy;
            }
        }
    }

    #[test]
    fn shift_chunks_scrolls_the_bookkeeping_on_both_axes_and_both_signs() {
        for (dcx, dcy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 1), (-1, -2)] {
            let mut before = grid();
            seed_slots(&mut before);
            let mut g = grid();
            seed_slots(&mut g);
            g.shift_chunks(dcx, dcy);

            let (cc, cr) = (g.chunk_cols(), g.chunk_rows());
            for chy in 0..cr {
                for chx in 0..cc {
                    let d = (chy * cc + chx) as usize;
                    let (si, sj) = (chx + dcx, chy + dcy);
                    if si < 0 || sj < 0 || si >= cc || sj >= cr {
                        assert!(!g.awake_this[d], "({dcx},{dcy}) slot {d} uncovered");
                        assert!(!g.awake_next[d]);
                        assert_eq!(g.box_this[d], Box2::EMPTY);
                        assert_eq!(g.box_next[d], Box2::EMPTY);
                        continue;
                    }
                    let s = (sj * cc + si) as usize;
                    assert_eq!(g.awake_this[d], before.awake_this[s], "({dcx},{dcy})");
                    assert_eq!(g.awake_next[d], before.awake_next[s], "({dcx},{dcy})");
                    let (dx, dy) = (dcx * CHUNK_CELLS, dcy * CHUNK_CELLS);
                    assert_eq!(g.box_this[d], before.box_this[s].translated(dx, dy));
                    assert_eq!(g.box_next[d], before.box_next[s].translated(dx, dy));
                    // The translation must land the box inside its NEW chunk.
                    let b = g.box_this[d];
                    let lo_x = chx * CHUNK_CELLS;
                    let lo_y = chy * CHUNK_CELLS;
                    assert!(b.x0 >= lo_x && b.x1 < lo_x + CHUNK_CELLS, "({dcx},{dcy})");
                    assert!(b.y0 >= lo_y && b.y1 < lo_y + CHUNK_CELLS, "({dcx},{dcy})");
                }
            }
            assert!(
                g.dirty.iter().all(|&d| d),
                "a shift moves everything visually"
            );
        }
    }

    #[test]
    fn shift_chunks_never_reads_a_slot_it_already_overwrote() {
        // The walk direction is the whole content of this test: with the wrong
        // one, a shift smears the first slot across the row.
        for (dcx, dcy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (3, 2), (-2, -3)] {
            let mut g = grid();
            let (cc, cr) = (g.chunk_cols(), g.chunk_rows());
            for (k, b) in g.box_this.iter_mut().enumerate() {
                *b = Box2 {
                    x0: k as i32,
                    y0: 0,
                    x1: k as i32,
                    y1: 0,
                };
            }
            g.shift_chunks(dcx, dcy);
            for chy in 0..cr {
                for chx in 0..cc {
                    let d = (chy * cc + chx) as usize;
                    let (si, sj) = (chx + dcx, chy + dcy);
                    if si < 0 || sj < 0 || si >= cc || sj >= cr {
                        continue;
                    }
                    let s = sj * cc + si;
                    assert_eq!(
                        g.box_this[d].x0,
                        s - dcx * CHUNK_CELLS,
                        "({dcx},{dcy}) slot {d} took the wrong source"
                    );
                }
            }
        }
    }

    #[test]
    fn shift_chunks_leaves_the_cell_planes_alone() {
        // The cells are memmoved by the window manager, which owns the chunk
        // store the outgoing ones are saved to. This is bookkeeping only.
        let mut g = grid();
        for (i, m) in g.material.iter_mut().enumerate() {
            *m = (i % 500) as CellId;
        }
        for (i, t) in g.temp.iter_mut().enumerate() {
            *t = (i % 251) as u8;
        }
        let material = g.material.clone();
        let temp = g.temp.clone();
        g.shift_chunks(1, -1);
        assert_eq!(g.material, material);
        assert_eq!(g.temp, temp);
    }

    #[test]
    fn shift_chunks_by_zero_is_a_no_op() {
        let mut g = grid();
        seed_slots(&mut g);
        let boxes = g.box_this.clone();
        g.shift_chunks(0, 0);
        assert_eq!(g.box_this, boxes);
        assert!(g.dirty.iter().all(|&d| !d), "no shift, no repaint");
    }

    #[test]
    fn a_shift_past_the_window_clears_everything() {
        for (dcx, dcy) in [(4, 0), (-4, 0), (0, 4), (0, -9), (99, 99)] {
            let mut g = grid();
            seed_slots(&mut g);
            g.shift_chunks(dcx, dcy);
            assert!(g.awake_this.iter().all(|&a| !a), "({dcx},{dcy})");
            assert!(g.awake_next.iter().all(|&a| !a), "({dcx},{dcy})");
            assert!(g.box_this.iter().all(|b| b.is_empty()), "({dcx},{dcy})");
            assert!(g.box_next.iter().all(|b| b.is_empty()), "({dcx},{dcy})");
            assert!(g.dirty.iter().all(|&d| d), "({dcx},{dcy})");
        }
    }

    #[test]
    fn a_survivor_mid_flow_keeps_flowing_across_a_shift() {
        // The end-to-end point of shift_chunks: a chunk that was awake before
        // the shift is still awake after it, at its new slot, with its box.
        let mut g = grid();
        g.wake(70, 70); // chunk (2,2)
        g.begin_sim_tick();
        assert_eq!(g.awake_chunks()[0].slot, slot(&g, 2, 2));

        g.shift_chunks(1, 1);
        let awake = g.awake_chunks();
        assert_eq!(awake.len(), 1);
        assert_eq!(awake[0].slot, slot(&g, 1, 1));
        assert_eq!(
            (awake[0].x0, awake[0].y0),
            (69 - CHUNK_CELLS, 69 - CHUNK_CELLS)
        );
    }

    // --- world coordinates ----------------------------------------------------

    #[test]
    fn world_coordinates_round_trip_across_the_origin() {
        let mut g = grid();
        // Put the window west and north of the world origin, so local 0 is a
        // negative world cell and the translation has to survive the sign.
        g.set_origin(-100, -50);
        assert_eq!(g.origin_cell_x(), -100);
        assert_eq!(g.origin_cell_y(), -50);

        for (wx, wy) in [(-100, -50), (-1, -1), (0, 0), (5, 7), (-60, 20)] {
            let w = WorldCell::new(wx, wy);
            let l = g.to_local(w);
            assert_eq!((l.x, l.y), (wx + 100, wy + 50));
            assert_eq!(g.world_to_local_x(wx), l.x);
            assert_eq!(g.world_to_local_y(wy), l.y);
            assert!(g.is_loaded_world(w));
            assert!(g.is_empty_world(w));
            g.set_world(w, 12);
            assert_eq!(g.get_world(w), 12);
            assert_eq!(g.get(l.x, l.y), 12);
            assert!(!g.is_empty_world(w));
        }
    }

    #[test]
    fn cells_outside_the_window_read_empty_and_write_nowhere() {
        let mut g = grid();
        g.set_origin(-100, -50);
        let outside = [
            WorldCell::new(-101, 0), // one west of the window
            WorldCell::new(0, -51),  // one north of it
            WorldCell::new(28, 0),   // one east (origin + cols)
            WorldCell::new(0, 78),   // one south (origin + rows)
        ];
        for w in outside {
            assert!(!g.is_loaded_world(w), "{w:?}");
            assert_eq!(g.get_world(w), EMPTY, "{w:?}");
            assert!(g.is_empty_world(w), "{w:?}");
            g.set_world(w, 42);
            assert_eq!(g.get_world(w), EMPTY, "{w:?} must not have been written");
        }
        assert!(g.material.iter().all(|&m| m == EMPTY));
    }

    #[test]
    fn moving_the_origin_reinterprets_the_same_cells() {
        let mut g = grid();
        g.set(0, 0, 5);
        assert_eq!(g.get_world(WorldCell::new(0, 0)), 5);
        g.set_origin(CHUNK_CELLS, 0);
        assert_eq!(
            g.get_world(WorldCell::new(0, 0)),
            EMPTY,
            "world (0,0) is no longer the window's top-left"
        );
        assert_eq!(g.get_world(WorldCell::new(CHUNK_CELLS, 0)), 5);
    }

    #[test]
    fn shift_gen_counts_completed_shifts() {
        let mut g = grid();
        assert_eq!(g.shift_gen(), 0);
        g.bump_shift_gen();
        g.bump_shift_gen();
        assert_eq!(g.shift_gen(), 2);
    }

    // --- indexing -------------------------------------------------------------

    #[test]
    fn idx_is_row_major_and_in_bounds_is_exclusive() {
        let g = grid();
        assert_eq!(g.idx(0, 0), 0);
        assert_eq!(g.idx(3, 2), 2 * g.cols() as usize + 3);
        assert!(g.in_bounds(0, 0));
        assert!(g.in_bounds(g.cols() - 1, g.rows() - 1));
        assert!(!g.in_bounds(-1, 0));
        assert!(!g.in_bounds(0, -1));
        assert!(!g.in_bounds(g.cols(), 0));
        assert!(!g.in_bounds(0, g.rows()));
    }

    #[test]
    fn chunk_counts_round_up_for_a_partial_window() {
        let g = CellGrid::new(CHUNK_CELLS + 1, 2 * CHUNK_CELLS);
        assert_eq!(g.chunk_cols(), 2);
        assert_eq!(g.chunk_rows(), 2);
    }

    #[test]
    fn dirty_is_set_by_writes_and_cleared_by_the_renderer() {
        let mut g = grid();
        assert!(!g.is_chunk_dirty(1, 1));
        g.set(CHUNK_CELLS + 4, CHUNK_CELLS + 4, 1);
        assert!(g.is_chunk_dirty(1, 1));
        g.clear_chunk_dirty(1, 1);
        assert!(!g.is_chunk_dirty(1, 1));
    }

    #[test]
    fn volatile_mask_holds_moved_and_not_burning() {
        assert!(CellFlags::VOLATILE_MASK.contains(CellFlags::MOVED));
        assert!(!CellFlags::VOLATILE_MASK.contains(CellFlags::BURNING));
        let durable = (CellFlags::MOVED | CellFlags::BURNING) - CellFlags::VOLATILE_MASK;
        assert_eq!(durable, CellFlags::BURNING);
    }
}

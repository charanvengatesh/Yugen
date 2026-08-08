//! A detached tile of the world — the unit the chunk store holds and a
//! persistence backend reads and writes.
//!
//! # What changed in the port
//!
//! `ChunkSnapshot` existed in the TypeScript as a methodless interface so it
//! would survive `structuredClone` across a `postMessage` and could be handed to
//! IndexedDB `put()` verbatim. Neither constraint survives here — there is one
//! thread and no browser storage — but the type stays, because the reason it is
//! *useful* is unchanged: it is the narrow plain-data shape the
//! [`ChunkPersistence`](super::chunk_store::ChunkPersistence) boundary speaks,
//! with no `diverged` bit and no behaviour attached.

use super::grid::CellFlags;
use super::materials::{CellId, EMPTY};
use crate::config::CHUNK_CELLS;

/// The plain-data form of a chunk: exactly what a persistence backend stores and
/// returns. Kept free of methods and identity so a backend can own it outright.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkSnapshot {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub material: Vec<CellId>,
    pub flags: Vec<CellFlags>,
    pub aux: Vec<u16>,
    pub temp: Vec<u8>,
    pub back: Vec<CellId>,
}

/// A detached chunk of the world: one CHUNK_CELLS² tile of cell data, addressed
/// by its absolute chunk coordinate. This is the serializable unit the ChunkStore
/// holds when a chunk is outside the live window (and what the persistence
/// backend reads/writes). Layout mirrors the live grid's flat arrays so blitting
/// in/out of the window is a plain slice copy.
///
/// A chunk also carries `diverged`: whether its contents still match what
/// `generate_chunk` would produce for this coordinate. Pristine chunks are free
/// to throw away — they regenerate identically from the seed — while diverged
/// ones must be persisted or the player's edits and the sim's results are lost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    chunk_x: i32,
    chunk_y: i32,

    /// Material id per cell.
    pub material: Vec<CellId>,
    /// Flag bits per cell.
    pub flags: Vec<CellFlags>,
    /// Per-cell scratch counter.
    pub aux: Vec<u16>,
    /// Per-cell temperature, carried so a window shift doesn't reset the thermal
    /// field. Note this plane does NOT participate in the divergence test — see
    /// [`Chunk::absorb_row`].
    pub temp: Vec<u8>,
    /// Background wall material per cell (0 = no wall).
    ///
    /// The plane behind the play plane: visible, diggable, and never solid. It
    /// carries a material id and nothing else — no flags, no aux, no temp — for
    /// the reason [`crate::sim::grid::CellGrid::back`] gives.
    ///
    /// Unlike [`Chunk::temp`], this DOES vote on divergence. See
    /// [`Chunk::absorb_row`].
    pub back: Vec<CellId>,

    /// True once this chunk's contents have been observed to differ from its
    /// pristine generated state (a player edit, or the sim actually moving a
    /// cell). Sticky: a chunk that happens to wander back to its generated state
    /// stays marked, which costs a little memory but can never lose data.
    ///
    /// Public in both directions because the two owners genuinely both write it:
    /// the window manager sets it from `extract_out`, and the store reads it to
    /// decide what to persist.
    pub diverged: bool,
}

/// The window planes one [`Chunk::absorb_row`] reads, borrowed together.
///
/// Five slices that are one idea — the live window, as the absorb sees it — and
/// passing them loose put `absorb_row` at eight arguments, which is both past
/// what clippy will allow and past what a reader can keep straight. Bundling
/// them also means a plane added to [`CellGrid`](crate::sim::grid::CellGrid) is
/// a field added HERE, in one place, rather than a sixth positional slice
/// threaded through every call site by hand.
///
/// Borrowed rather than owned: the absorb reads the window and never keeps it.
pub struct WindowRows<'a> {
    pub material: &'a [CellId],
    pub flags: &'a [CellFlags],
    pub aux: &'a [u16],
    pub temp: &'a [u8],
    pub back: &'a [CellId],
}

impl Chunk {
    /// Cells in one chunk.
    pub const SIZE: usize = (CHUNK_CELLS * CHUNK_CELLS) as usize;

    /// An all-empty chunk at an absolute chunk coordinate.
    pub fn new(chunk_x: i32, chunk_y: i32) -> Chunk {
        Chunk {
            chunk_x,
            chunk_y,
            material: vec![EMPTY; Chunk::SIZE],
            flags: vec![CellFlags::empty(); Chunk::SIZE],
            aux: vec![0; Chunk::SIZE],
            temp: vec![0; Chunk::SIZE],
            back: vec![EMPTY; Chunk::SIZE],
            diverged: false,
        }
    }

    /// Absolute chunk X. Immutable: a chunk's coordinate is its identity, and it
    /// is the key it is filed under in the store's cache.
    #[inline]
    pub const fn chunk_x(&self) -> i32 {
        self.chunk_x
    }

    /// Absolute chunk Y. See [`Chunk::chunk_x`].
    #[inline]
    pub const fn chunk_y(&self) -> i32 {
        self.chunk_y
    }

    /// Overwrite the material plane with a freshly generated one.
    ///
    /// flags / aux / temp are left zeroed — a freshly generated chunk has no sim
    /// state. The chunk stays pristine, which is the whole point: it can be
    /// dropped and regenerated bit-identically.
    pub fn set_generated_material(&mut self, material: &[CellId]) {
        self.material.copy_from_slice(material);
    }

    /// Overwrite the background wall plane with a freshly generated one.
    ///
    /// Separate from [`Chunk::set_generated_material`] rather than folded into
    /// it, so that a generator which produces only the front plane — every caller
    /// today — stays a caller that cannot accidentally leave the back plane
    /// holding another chunk's walls.
    pub fn set_generated_back(&mut self, back: &[CellId]) {
        self.back.copy_from_slice(back);
    }

    /// Copy one row of window cells into this chunk, reporting whether anything
    /// that matters to the world changed. This is the single place divergence is
    /// detected: the chunk still holds the cells as of the last blit *out of* the
    /// store, so comparing on the way back in tells us exactly whether the window
    /// changed them. [`CellFlags::VOLATILE_MASK`] bits are masked out — they flip
    /// every tick on matter that is merely flowing past and say nothing about
    /// stored state.
    ///
    /// `back` DOES vote, and the contrast with `temp` below is the whole reason
    /// this is worth stating. A wall is placed or dug by the player and by nothing
    /// else — no automata pass touches the plane — so a pristine chunk's back
    /// plane is bit-stable across a shift, and a difference can only mean a real
    /// edit. A chunk that did not mark itself diverged would lose that edit the
    /// moment it was evicted.
    ///
    /// `temp` is copied but deliberately does NOT vote on divergence, for two
    /// reasons:
    ///
    ///   1. It would be true of almost everything. Heat is a continuously-relaxing
    ///      field: it diffuses outward from every emitter and decays back toward
    ///      ambient, settling into a fixed point the automata only stops revisiting
    ///      once per-tick swings fall under HEAT_WAKE_DELTA — which permits a
    ///      standing ±1 flutter. So every chunk within diffusion range of a lava
    ///      lake would report "changed" on essentially every shift, latch the
    ///      sticky `diverged` bit, and be written through to persistence on
    ///      eviction. That is exactly the population the MAX_PERSISTED_CHUNKS LRU
    ///      is sized to exclude, and it would thrash the cap for no benefit.
    ///
    ///   2. It is derivable, unlike the other planes. A pristine chunk's material
    ///      comes back bit-identical from the seed, and its heat re-establishes
    ///      itself within a few dozen ticks of becoming resident again, because the
    ///      emitters that produced it regenerate too and every freshly-blitted
    ///      chunk is awake. Dropping temp for an untouched chunk costs a brief warm
    ///      -up; dropping a player's edit would be data loss. Only the second is
    ///      worth spending the persistence budget on.
    ///
    /// The plane is still copied through on every absorb, so heat round-trips
    /// intact across a window shift (the resident cache entry is the baseline the
    /// next blit-in reads), and a chunk that diverges for a real reason carries its
    /// thermal field into the snapshot for free.
    ///
    /// Unlike `blit_in`, this stays a scalar element loop in Rust too: it is a
    /// compare-and-copy, not a copy, and the comparison is the reason the
    /// function exists.
    ///
    /// The flag comparison goes through `bits()` rather than `bitflags`' `!`
    /// operator on purpose. `!` on a flags type is `from_bits_truncate(!bits)`,
    /// which drops every bit the type does not name — and bits 4+ are documented
    /// as free for feature code. Masking on the raw byte, as the TypeScript's
    /// `~FLAG_VOLATILE_MASK` did, keeps an unnamed feature bit voting on
    /// divergence instead of silently discarding it.
    pub fn absorb_row(&mut self, ly: i32, src: &WindowRows<'_>, src_offset: usize) -> bool {
        let base = (ly * CHUNK_CELLS) as usize;
        let n = CHUNK_CELLS as usize;
        let mut changed = false;
        for i in 0..n {
            let s = src_offset + i;
            let d = base + i;
            let m = src.material[s];
            let f = src.flags[s];
            let a = src.aux[s];
            let bk = src.back[s];
            if self.material[d] != m
                || self.aux[d] != a
                || self.back[d] != bk
                || ((self.flags[d] ^ f).bits() & !CellFlags::VOLATILE_MASK.bits()) != 0
            {
                changed = true;
            }
            self.material[d] = m;
            self.flags[d] = f;
            self.aux[d] = a;
            self.temp[d] = src.temp[s];
            self.back[d] = bk;
        }
        changed
    }

    /// Detached copy for the persistence backend to own.
    pub fn snapshot(&self) -> ChunkSnapshot {
        ChunkSnapshot {
            chunk_x: self.chunk_x,
            chunk_y: self.chunk_y,
            material: self.material.clone(),
            flags: self.flags.clone(),
            aux: self.aux.clone(),
            temp: self.temp.clone(),
            back: self.back.clone(),
        }
    }

    /// Rebuild a chunk from persisted data. A snapshot only ever exists because
    /// the chunk diverged, so the restored chunk inherits that state — it must
    /// never fall back to worldgen again.
    pub fn from_snapshot(snap: &ChunkSnapshot) -> Chunk {
        let mut chunk = Chunk::new(snap.chunk_x, snap.chunk_y);
        chunk.material.copy_from_slice(&snap.material);
        chunk.flags.copy_from_slice(&snap.flags);
        chunk.aux.copy_from_slice(&snap.aux);
        chunk.temp.copy_from_slice(&snap.temp);
        chunk.back.copy_from_slice(&snap.back);
        chunk.diverged = true;
        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row of window-shaped source planes, `CHUNK_CELLS` wide.
    ///
    /// Owned, because a test mutates them between absorbs; [`WindowRows`] borrows
    /// them back for each call through [`rows`].
    struct Sources {
        material: Vec<CellId>,
        flags: Vec<CellFlags>,
        aux: Vec<u16>,
        temp: Vec<u8>,
        back: Vec<CellId>,
    }

    fn row_sources() -> Sources {
        let n = CHUNK_CELLS as usize;
        Sources {
            material: vec![0; n],
            flags: vec![CellFlags::empty(); n],
            aux: vec![0; n],
            temp: vec![0; n],
            back: vec![0; n],
        }
    }

    fn rows(s: &Sources) -> WindowRows<'_> {
        WindowRows {
            material: &s.material,
            flags: &s.flags,
            aux: &s.aux,
            temp: &s.temp,
            back: &s.back,
        }
    }

    #[test]
    fn a_fresh_chunk_is_pristine_and_empty() {
        let c = Chunk::new(-3, 7);
        assert_eq!((c.chunk_x(), c.chunk_y()), (-3, 7));
        assert!(!c.diverged);
        assert_eq!(c.material.len(), Chunk::SIZE);
        assert!(c.material.iter().all(|&m| m == EMPTY));
    }

    #[test]
    fn absorbing_identical_cells_reports_no_change() {
        let mut c = Chunk::new(0, 0);
        let s = row_sources();
        assert!(!c.absorb_row(0, &rows(&s), 0));
    }

    #[test]
    fn material_and_aux_changes_vote() {
        let mut c = Chunk::new(0, 0);
        let mut s = row_sources();
        s.material[3] = 42;
        assert!(c.absorb_row(0, &rows(&s), 0));
        // Absorbed, so the same row is now the baseline and does not re-report.
        assert!(!c.absorb_row(0, &rows(&s), 0));
        s.aux[9] = 5;
        assert!(c.absorb_row(0, &rows(&s), 0));
    }

    #[test]
    fn volatile_flag_bits_do_not_vote_but_durable_ones_do() {
        let mut c = Chunk::new(0, 0);
        let mut s = row_sources();

        s.flags[0] = CellFlags::MOVED;
        assert!(
            !c.absorb_row(0, &rows(&s), 0),
            "MOVED is per-tick bookkeeping and must not mark a chunk diverged"
        );
        assert_eq!(c.flags[0], CellFlags::MOVED, "but it is still copied");

        s.flags[0] = CellFlags::BURNING;
        assert!(
            c.absorb_row(0, &rows(&s), 0),
            "BURNING is durable world state and must mark a chunk diverged"
        );
    }

    /// The comment on `absorb_row` is load-bearing: a chunk sitting next to a
    /// lava lake sees its temperature plane flutter every single tick, and if
    /// that voted, every such chunk would latch `diverged` and be persisted.
    #[test]
    fn temp_is_copied_but_never_votes() {
        let mut c = Chunk::new(0, 0);
        let mut s = row_sources();
        for (i, cell) in s.temp.iter_mut().enumerate() {
            *cell = (i as u8).wrapping_mul(7).wrapping_add(1);
        }
        assert!(
            !c.absorb_row(0, &rows(&s), 0),
            "a temperature-only difference must not count as divergence"
        );
        assert_eq!(
            &c.temp[..s.temp.len()],
            &s.temp[..],
            "temp still rides along"
        );
    }

    /// A wall placed or dug is a real edit and must mark the chunk diverged.
    ///
    /// The contrast with `temp` is the whole reason this is a separate test.
    /// `temp` is copied and deliberately does not vote, because it is a
    /// continuously-relaxing field that would be true of almost everything. The
    /// back plane is the opposite: no automata pass touches it, so it is
    /// bit-stable across a shift on a pristine chunk, and a difference can only
    /// mean the player put it there.
    ///
    /// A chunk that copied the plane without voting would look completely correct
    /// until the moment it was evicted, and then silently lose the wall.
    #[test]
    fn a_wall_edit_alone_marks_the_chunk_diverged() {
        let mut c = Chunk::new(0, 0);
        let mut s = row_sources();

        assert!(!c.absorb_row(0, &rows(&s), 0), "baseline is clean");

        s.back[6] = 77;
        assert!(
            c.absorb_row(0, &rows(&s), 0),
            "a wall appeared and nothing else changed — if this does not vote, \
             the edit is lost on eviction"
        );
        assert_eq!(c.back[6], 77, "and it is copied, not merely counted");

        // Absorbed, so the same row is now the baseline and does not re-report.
        assert!(!c.absorb_row(0, &rows(&s), 0));

        // Digging one out is an edit too, in the other direction.
        s.back[6] = 0;
        assert!(
            c.absorb_row(0, &rows(&s), 0),
            "removing a wall is as much an edit as placing one"
        );
    }

    #[test]
    fn absorb_reads_the_row_at_the_given_source_offset() {
        let n = CHUNK_CELLS as usize;
        // Two rows of source; row 1 is the interesting one.
        let mut material = vec![0 as CellId; n * 2];
        material[n + 4] = 9;
        let s = Sources {
            material,
            flags: vec![CellFlags::empty(); n * 2],
            aux: vec![0u16; n * 2],
            temp: vec![0u8; n * 2],
            back: vec![0 as CellId; n * 2],
        };

        let mut c = Chunk::new(0, 0);
        assert!(c.absorb_row(1, &rows(&s), n));
        assert_eq!(c.material[n + 4], 9, "landed in chunk row 1");
        assert!(c.material[..n].iter().all(|&v| v == 0), "row 0 untouched");
    }

    #[test]
    fn a_snapshot_round_trips_and_restores_as_diverged() {
        let mut c = Chunk::new(4, -2);
        c.material[17] = 99;
        c.temp[17] = 200;
        c.diverged = true;

        let snap = c.snapshot();
        let back = Chunk::from_snapshot(&snap);
        assert_eq!(back.chunk_x(), 4);
        assert_eq!(back.chunk_y(), -2);
        assert_eq!(back.material, c.material);
        assert_eq!(back.temp, c.temp);
        assert!(
            back.diverged,
            "a snapshot only exists because the chunk diverged"
        );
    }

    #[test]
    fn a_snapshot_is_detached_from_the_chunk_it_came_from() {
        let mut c = Chunk::new(0, 0);
        let snap = c.snapshot();
        c.material[0] = 123;
        assert_eq!(snap.material[0], EMPTY);
    }
}

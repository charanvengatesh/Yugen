//! The "game file system": the three tiers a chunk can be found in.
//!
//! # What changed in the port
//!
//! The TypeScript keyed both the hot cache and the persistence map on
//! `chunkKey(x, y) -> "x,y"`. That string existed for exactly one reason: a JS
//! `Map` cannot key on a pair. A Rust `HashMap<(i32, i32), _>` keys on the pair
//! directly, so `chunkKey` is gone (see [`super::coords`]) and every lookup on
//! the streaming path saves a `format!` and the allocation behind it.
//!
//! The LRU is the other change. `MemoryChunkPersistence` leaned on JS `Map`
//! iteration being insertion-ordered, with a `delete` + `set` to touch an entry;
//! Rust's `HashMap` promises no order at all, so recency is tracked explicitly.
//! See [`MemoryChunkPersistence`].

use std::collections::{BTreeMap, HashMap};

use rayon::prelude::*;

use super::chunk::{Chunk, ChunkSnapshot};
use super::worldgen::ChunkGen;

/// How many chunks [`ChunkStore::prefetch`] wants to see before it reaches for a
/// thread pool.
///
/// A `par_iter` costs a job split and a set of wakeups whether or not there is
/// work behind it. The two callers are the whole-window fills — 88 chunks each —
/// so anything below "more than a window shift's worth" is a batch that arrived
/// mostly cached and is cheaper to finish on this thread.
const PREFETCH_MIN_PARALLEL: usize = 8;

/// Upper bound on retained diverged chunks. At CHUNK_CELLS² = 1024 cells × 6
/// bytes (material 2 + flags 1 + aux 2 + temp 1) a snapshot is ~6 KB, so 2048
/// chunks is ~12 MB — roughly a 45×45-chunk region of fully-edited world, far
/// more than a player touches in a session. The temp plane rides along in the
/// snapshot but does not itself mark a chunk diverged, so it changes the size of
/// a snapshot without changing how many are retained. Past the cap the
/// least-recently-used diverged
/// chunk is dropped and that patch of world reverts to its generated state;
/// losing the oldest edit is preferable to an unbounded heap.
pub const MAX_PERSISTED_CHUNKS: usize = 2048;

/// Where diverged chunks live once they fall out of the hot cache. Deliberately
/// the narrowest possible surface (read / write / len) so swapping the in-memory
/// implementation for a file or database backend is a new type and nothing else.
///
/// `read` takes `&mut self` because reading is a TOUCH: it makes the entry the
/// most recently used one, which is the whole reason the default backend can
/// bound its memory. It returns an owned snapshot rather than a borrow so a
/// backend that has to go to disk — which necessarily materialises a fresh value
/// — fits the same signature.
///
/// **There is no durable save behind this trait, and this port does not add
/// one.** The TypeScript said the same thing (its `flush` had zero callers) and
/// the boundary is kept clean for the same reason: so adding one later is a new
/// implementation of these three methods and no change anywhere else.
/// `Send + Sync` is a bound on the TRAIT, not a detail of one backend. The whole
/// world — grid, window manager, store — is owned by the host application's
/// scheduler, which may move it between threads and has to be able to prove it.
/// Stating it here rather than at each `Box<dyn ...>` site means a future disk
/// backend cannot quietly make the world unmovable. It costs the in-memory
/// backend nothing: it is already both.
pub trait ChunkPersistence: Send + Sync {
    /// The stored snapshot for a chunk coordinate, if any. Touches it.
    fn read(&mut self, chunk_x: i32, chunk_y: i32) -> Option<ChunkSnapshot>;
    /// Store a snapshot, replacing any previous one for the same coordinate.
    fn write(&mut self, snap: ChunkSnapshot);
    /// How many snapshots are retained.
    fn len(&self) -> usize;
    /// Whether anything is retained. (Rust convention; the TypeScript had only
    /// a `size` getter.)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One retained snapshot plus its place in the recency order.
struct Entry {
    seq: u64,
    snap: ChunkSnapshot,
}

/// Default backend: an LRU map held in the sim's heap. Survives eviction and
/// revisiting, but not a process restart — nothing here is durable.
///
/// # The LRU
///
/// The TypeScript got its LRU for free from `Map`: iteration is insertion-
/// ordered, so `delete` + `set` moved an entry to the back and `keys().next()`
/// was the least-recently-used one. `HashMap` gives no such order, so recency is
/// a second index: a monotonically increasing sequence number per touch, and a
/// `BTreeMap<seq, key>` whose first entry is by construction the LRU one.
///
/// That is O(log n) per touch against the JS O(1), on a path that runs a handful
/// of times per window shift, and it costs one `u64` and one map node per
/// retained chunk. The alternative — an intrusive doubly-linked list over an
/// arena, for O(1) — is roughly four times the code and the same number of cache
/// misses, for an operation that happens ~10 times a second at most. Written
/// here rather than pulled in as a crate because `godgame-core`'s dependency
/// list is deliberately three entries long.
///
/// `seq` is a `u64` and is never reused. At one touch per nanosecond it wraps
/// after 584 years, so it is not treated as a wrapping counter.
pub struct MemoryChunkPersistence {
    capacity: usize,
    entries: HashMap<(i32, i32), Entry>,
    /// Recency order, least-recently-used first.
    order: BTreeMap<u64, (i32, i32)>,
    next_seq: u64,
}

impl Default for MemoryChunkPersistence {
    fn default() -> Self {
        MemoryChunkPersistence::new(MAX_PERSISTED_CHUNKS)
    }
}

impl MemoryChunkPersistence {
    /// A backend holding at most `capacity` snapshots.
    pub fn new(capacity: usize) -> MemoryChunkPersistence {
        MemoryChunkPersistence {
            capacity,
            entries: HashMap::new(),
            order: BTreeMap::new(),
            next_seq: 0,
        }
    }

    /// The cap this backend was built with.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    fn bump(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }
}

impl ChunkPersistence for MemoryChunkPersistence {
    fn read(&mut self, chunk_x: i32, chunk_y: i32) -> Option<ChunkSnapshot> {
        let key = (chunk_x, chunk_y);
        // Peek the current sequence number first: `bump` needs `&mut self` and
        // the entry borrow would still be live if it happened after.
        let old_seq = self.entries.get(&key)?.seq;
        let seq = self.bump();
        self.order.remove(&old_seq);
        self.order.insert(seq, key);
        let entry = self.entries.get_mut(&key).expect("just looked it up");
        entry.seq = seq;
        // Cloned, not moved out: a read leaves the snapshot in place, exactly as
        // the TypeScript's touch-and-return did. The caller copies it into a
        // `Chunk` either way, so nothing is saved by handing over ownership, and
        // dropping the stored copy would silently make a read destructive.
        Some(entry.snap.clone())
    }

    fn write(&mut self, snap: ChunkSnapshot) {
        let key = (snap.chunk_x, snap.chunk_y);
        if let Some(old) = self.entries.remove(&key) {
            self.order.remove(&old.seq);
        }
        let seq = self.bump();
        self.order.insert(seq, key);
        self.entries.insert(key, Entry { seq, snap });
        while self.entries.len() > self.capacity {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The "game file system": a hot cache of chunks keyed by chunk coordinate,
/// backed by a persistence layer, backed by deterministic generation.
///
///   cache hit          → the live chunk object
///   persistence hit    → a chunk that diverged and was evicted; restore it
///   miss               → generate from the seed
///
/// The cache is a write-back cache: a diverged chunk is flushed to persistence at
/// the moment it would otherwise be dropped (`evict_beyond`), not on every save,
/// so a chunk the player is walking around in costs no copies. Pristine chunks
/// are never written — they regenerate identically, which is what keeps memory
/// bounded no matter how far the player walks.
pub struct ChunkStore {
    cache: HashMap<(i32, i32), Chunk>,
    persistence: Box<dyn ChunkPersistence>,
    /// The generator, held rather than re-created per chunk.
    ///
    /// The TypeScript called a free `generateChunk(x, y, seed)`, which is still
    /// available here — but it builds a whole `Noise` and a whole heightmap memo
    /// per call, and the streaming path is the bulk caller worldgen's `ChunkGen`
    /// exists for. Same output, since worldgen is pure by contract and the memo
    /// is keyed on the noise identity.
    chunk_gen: ChunkGen,
}

impl ChunkStore {
    /// A store for one world seed, with the default in-memory backend.
    pub fn new(seed: u32) -> ChunkStore {
        ChunkStore::with_persistence(seed, Box::new(MemoryChunkPersistence::default()))
    }

    /// A store for one world seed with an explicit persistence backend.
    pub fn with_persistence(seed: u32, persistence: Box<dyn ChunkPersistence>) -> ChunkStore {
        ChunkStore {
            cache: HashMap::new(),
            persistence,
            chunk_gen: ChunkGen::new(seed),
        }
    }

    /// The world seed this store generates against.
    #[inline]
    pub fn seed(&self) -> u32 {
        self.chunk_gen.seed()
    }

    /// The cached chunk for a coordinate, if it is in the hot cache.
    pub fn get(&self, chunk_x: i32, chunk_y: i32) -> Option<&Chunk> {
        self.cache.get(&(chunk_x, chunk_y))
    }

    /// Put a chunk into the hot cache, replacing any entry for its coordinate.
    pub fn put(&mut self, chunk: Chunk) {
        self.cache.insert((chunk.chunk_x(), chunk.chunk_y()), chunk);
    }

    /// The cached chunk for a coordinate, inserting an empty one if absent.
    ///
    /// This is the shape the TypeScript's `saveSlot` wanted and could not have:
    /// it wrote `store.get(cx, cy) ?? new Chunk(cx, cy)`, mutated the result, and
    /// called `store.put` to file it — a redundant re-insert whenever the chunk
    /// was already cached, which is the overwhelmingly common case. One entry
    /// lookup replaces get + construct + put.
    ///
    /// A miss can only happen if the chunk was dropped while resident; the
    /// all-zero chunk then reads as "everything changed" to `absorb_row`, which
    /// errs toward persisting — never toward silently discarding an edit.
    pub fn entry_mut(&mut self, chunk_x: i32, chunk_y: i32) -> &mut Chunk {
        self.cache
            .entry((chunk_x, chunk_y))
            .or_insert_with(|| Chunk::new(chunk_x, chunk_y))
    }

    /// Return the chunk for a coordinate: from cache, else from persistence, else
    /// freshly generated. An untouched chunk always takes the last branch, so it
    /// comes back bit-identical however many times it is evicted and revisited.
    pub fn load_or_generate(&mut self, chunk_x: i32, chunk_y: i32) -> &Chunk {
        let key = (chunk_x, chunk_y);
        if !self.cache.contains_key(&key) {
            let chunk = match self.persistence.read(chunk_x, chunk_y) {
                Some(snap) => Chunk::from_snapshot(&snap),
                None => {
                    let mut chunk = Chunk::new(chunk_x, chunk_y);
                    chunk.set_generated_material(&self.chunk_gen.generate(chunk_x, chunk_y));
                    // flags / aux / temp start zeroed — a freshly generated
                    // chunk has no sim state.
                    chunk
                }
            };
            self.cache.insert(key, chunk);
        }
        // Two lookups on the miss path rather than one, because the borrow
        // checker will not let a `get_mut` borrow survive the `insert` that the
        // miss needs. The hit path — the one that runs on every load — is one
        // `contains_key` plus one `get`, both O(1) on an integer pair key.
        self.cache.get(&key).expect("just inserted")
    }

    /// Populate the cache for a whole batch of coordinates at once, generating
    /// the ones that have to be generated ACROSS A RAYON POOL.
    ///
    /// [`Self::load_or_generate`] is a one-at-a-time call and stays that way: it
    /// runs on the streaming path, where a window shift wants eight chunks and
    /// the pool would cost more than it saved. This is the other shape — filling
    /// a whole window at once, which happens on load and on a teleport-sized
    /// jump — and there the 88 generates are the single most expensive thing the
    /// engine does before the first frame.
    ///
    /// THE ONLY REASON THIS IS SAFE IS THAT WORLDGEN IS PURE. A chunk is a
    /// function of `(chunk_x, chunk_y, seed)` and nothing else: there is no
    /// static with interior mutability and no thread local anywhere in
    /// `worldgen`, and the heightmap memo and cave lattice that would otherwise
    /// have been module state are OWNED by [`ChunkGen`] for exactly this reason.
    /// `worldgen_purity.rs` asserts it — check 8 generates the same set across a
    /// pool and requires the result to be byte-identical to the serial one, and
    /// `map_init` is what makes that a real test rather than a formality.
    ///
    /// So each worker gets its own `ChunkGen`, which is why `self.chunk_gen` is
    /// untouched here. Two workers building the same chunk would produce the
    /// same bytes; they simply never have to.
    ///
    /// Persistence reads stay on this thread. They are a `HashMap` hit that also
    /// TOUCHES the LRU order, so they are inherently serial, and they are also
    /// the cheap branch — a stored chunk costs a memcpy where a generated one
    /// costs a heightmap, a cave field and three decorator passes.
    pub fn prefetch(&mut self, coords: &[(i32, i32)]) {
        let mut missing: Vec<(i32, i32)> = Vec::with_capacity(coords.len());
        for &key in coords {
            if self.cache.contains_key(&key) {
                continue;
            }
            match self.persistence.read(key.0, key.1) {
                Some(snap) => {
                    self.cache.insert(key, Chunk::from_snapshot(&snap));
                }
                None => missing.push(key),
            }
        }

        if missing.len() < PREFETCH_MIN_PARALLEL {
            for (cx, cy) in missing {
                self.load_or_generate(cx, cy);
            }
            return;
        }

        let seed = self.seed();
        let generated: Vec<Chunk> = missing
            .par_iter()
            .map_init(
                || ChunkGen::new(seed),
                |chunk_gen, &(cx, cy)| {
                    let mut chunk = Chunk::new(cx, cy);
                    chunk.set_generated_material(&chunk_gen.generate(cx, cy));
                    // flags / aux / temp start zeroed — a freshly generated
                    // chunk has no sim state.
                    chunk
                },
            )
            .collect();
        for chunk in generated {
            self.cache.insert((chunk.chunk_x(), chunk.chunk_y()), chunk);
        }
    }

    /// Drop cached chunks farther than `radius` (Chebyshev) from a centre chunk.
    /// Diverged chunks are written through to persistence on the way out; pristine
    /// ones are simply forgotten.
    ///
    /// `radius` must cover every chunk currently resident in the window, or a live
    /// chunk's stale cache entry would be persisted over its real contents — the
    /// WindowManager guarantees this.
    ///
    /// The evicted coordinates are collected and SORTED before the write-through
    /// pass. The TypeScript iterated its cache in insertion order, so the recency
    /// order it handed the LRU was deterministic; `HashMap` iteration order is
    /// unspecified and varies run to run, and letting that decide which edits
    /// survive the cap would make the world non-reproducible. One evict pass
    /// produces at most a few hundred writes against a cap of 2048, so the two
    /// orders are indistinguishable in practice — but only one of them is
    /// deterministic.
    pub fn evict_beyond(&mut self, center_chunk_x: i32, center_chunk_y: i32, radius: i32) {
        let mut evicting: Vec<(i32, i32)> = self
            .cache
            .keys()
            .copied()
            .filter(|&(cx, cy)| {
                (cx - center_chunk_x).abs() > radius || (cy - center_chunk_y).abs() > radius
            })
            .collect();
        evicting.sort_unstable();
        for key in evicting {
            if let Some(c) = self.cache.remove(&key)
                && c.diverged
            {
                self.persistence.write(c.snapshot());
            }
        }
    }

    /// Write every diverged cached chunk through to persistence without dropping
    /// it. Nothing calls this yet; it is the hook a durable backend needs on
    /// shutdown or manual save so in-flight edits aren't lost.
    ///
    /// It had zero callers in the TypeScript too, and it still has none because
    /// the only backend is in-memory: flushing to a map that dies with the
    /// process buys nothing. Kept because it is the one method a durable backend
    /// needs that the trait alone does not imply.
    pub fn flush(&mut self) {
        let mut keys: Vec<(i32, i32)> = self.cache.keys().copied().collect();
        keys.sort_unstable(); // deterministic recency order — see `evict_beyond`
        for key in keys {
            let c = &self.cache[&key];
            if c.diverged {
                self.persistence.write(c.snapshot());
            }
        }
    }

    /// Chunks held in the hot cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the hot cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Diverged chunks held by the persistence backend.
    #[inline]
    pub fn persisted_len(&self) -> usize {
        self.persistence.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::worldgen::generate_chunk;

    const SEED: u32 = 12345;

    fn snap_at(x: i32, y: i32, marker: u16) -> ChunkSnapshot {
        let mut c = Chunk::new(x, y);
        c.material[0] = marker;
        c.snapshot()
    }

    #[test]
    fn a_missing_chunk_is_generated_and_matches_worldgen_exactly() {
        let mut store = ChunkStore::new(SEED);
        let want = generate_chunk(3, -2, SEED);
        assert_eq!(store.load_or_generate(3, -2).material, want);
        assert!(!store.get(3, -2).expect("cached").diverged);
    }

    /// A batch fetched across the pool must be indistinguishable from the same
    /// chunks fetched one at a time — including the mixed case, where some are
    /// already cached and some come back from persistence rather than from the
    /// generator.
    ///
    /// `worldgen_purity.rs` already proves the generator itself is
    /// order-independent and thread-safe. What this adds is the store's own
    /// bookkeeping: that `prefetch` puts every chunk in the cache, keyed
    /// correctly, and never lets a parallel generate overwrite a persisted edit.
    #[test]
    fn a_prefetched_batch_is_identical_to_fetching_one_at_a_time() {
        let coords: Vec<(i32, i32)> = (-3..=3).flat_map(|x| (0..4).map(move |y| (x, y))).collect();
        assert!(
            coords.len() > PREFETCH_MIN_PARALLEL,
            "the pool path is taken"
        );

        let mut serial = ChunkStore::new(SEED);
        let mut batched = ChunkStore::new(SEED);
        // A stored edit and an already-resident chunk, so the batch has all
        // three source tiers in it.
        for store in [&mut serial, &mut batched] {
            store.put(Chunk::new(0, 0));
            store.persistence.write(snap_at(1, 1, 4242));
        }

        for &(cx, cy) in &coords {
            serial.load_or_generate(cx, cy);
        }
        batched.prefetch(&coords);

        assert_eq!(batched.len(), serial.len());
        for &(cx, cy) in &coords {
            let want = serial.get(cx, cy).expect("serial cached every chunk");
            let got = batched
                .get(cx, cy)
                .unwrap_or_else(|| panic!("prefetch missed chunk ({cx},{cy})"));
            assert_eq!(
                got.material, want.material,
                "chunk ({cx},{cy}) differs between the batched and serial paths"
            );
            assert_eq!((got.chunk_x(), got.chunk_y()), (cx, cy), "wrong cache key");
        }
        assert_eq!(
            batched.get(1, 1).expect("persisted").material[0],
            4242,
            "a parallel generate overwrote a persisted edit"
        );
    }

    /// Below the threshold `prefetch` is the serial path, and must still be
    /// correct — it is the same function, so a caller cannot end up with a
    /// half-filled cache because its batch was small.
    #[test]
    fn a_small_prefetch_still_fills_the_cache() {
        let coords: Vec<(i32, i32)> = (0..PREFETCH_MIN_PARALLEL as i32 - 1)
            .map(|x| (x, 9))
            .collect();
        let mut store = ChunkStore::new(SEED);
        store.prefetch(&coords);
        for (cx, cy) in coords {
            assert_eq!(
                store.get(cx, cy).expect("cached").material,
                generate_chunk(cx, cy, SEED)
            );
        }
    }

    #[test]
    fn the_held_generator_agrees_with_the_one_shot_free_function() {
        // The store holds a `ChunkGen` across chunks instead of building one per
        // call; worldgen's purity contract is what makes that legal, so assert it
        // at this boundary rather than trusting it.
        let mut store = ChunkStore::new(SEED);
        for (cx, cy) in [(0, 0), (5, 5), (-1, 3), (0, 0), (5, 5)] {
            assert_eq!(
                store.load_or_generate(cx, cy).material,
                generate_chunk(cx, cy, SEED),
                "chunk ({cx},{cy}) differed from a cold generate"
            );
            store.evict_beyond(1000, 1000, 0); // force a regenerate next time
        }
    }

    #[test]
    fn a_pristine_chunk_is_forgotten_on_eviction_and_regenerates_identically() {
        let mut store = ChunkStore::new(SEED);
        let first = store.load_or_generate(0, 0).material.clone();
        store.evict_beyond(50, 50, 1);
        assert_eq!(store.len(), 0, "pristine chunk dropped from the cache");
        assert_eq!(store.persisted_len(), 0, "and never written through");
        assert_eq!(store.load_or_generate(0, 0).material, first);
    }

    #[test]
    fn a_diverged_chunk_survives_eviction_and_comes_back_changed() {
        let mut store = ChunkStore::new(SEED);
        let pristine = store.load_or_generate(0, 0).material.clone();
        {
            let c = store.entry_mut(0, 0);
            c.material[7] = 4242;
            c.diverged = true;
        }

        store.evict_beyond(50, 50, 1);
        assert_eq!(store.len(), 0);
        assert_eq!(
            store.persisted_len(),
            1,
            "diverged chunks are written through"
        );

        let back = store.load_or_generate(0, 0);
        assert_eq!(back.material[7], 4242, "the edit came back");
        assert!(back.diverged, "and it is still known to be non-regenerable");
        assert_ne!(back.material, pristine);
    }

    #[test]
    fn eviction_uses_chebyshev_distance() {
        let mut store = ChunkStore::new(SEED);
        for cy in -2..=2 {
            for cx in -2..=2 {
                store.load_or_generate(cx, cy);
            }
        }
        assert_eq!(store.len(), 25);
        store.evict_beyond(0, 0, 1);
        assert_eq!(store.len(), 9, "the 3x3 Chebyshev ball survives");
        assert!(store.get(1, 1).is_some(), "the diagonal is within radius 1");
        assert!(store.get(2, 0).is_none());
    }

    #[test]
    fn flush_writes_diverged_chunks_without_dropping_them() {
        let mut store = ChunkStore::new(SEED);
        store.load_or_generate(0, 0);
        store.load_or_generate(1, 0);
        store.entry_mut(1, 0).diverged = true;

        store.flush();
        assert_eq!(store.len(), 2, "nothing was dropped");
        assert_eq!(
            store.persisted_len(),
            1,
            "only the diverged one was written"
        );
    }

    // --- The LRU --------------------------------------------------------------

    #[test]
    fn the_lru_evicts_in_insertion_order_and_honours_its_cap() {
        let mut p = MemoryChunkPersistence::new(3);
        for i in 0..3 {
            p.write(snap_at(i, 0, 1));
        }
        assert_eq!(p.len(), 3);

        p.write(snap_at(3, 0, 1));
        assert_eq!(p.len(), 3, "the cap is honoured");
        assert!(p.read(0, 0).is_none(), "the oldest write was dropped");
        assert!(p.read(1, 0).is_some());
        assert!(p.read(3, 0).is_some());
    }

    #[test]
    fn a_read_touches_an_entry_so_it_outlives_older_ones() {
        let mut p = MemoryChunkPersistence::new(3);
        for i in 0..3 {
            p.write(snap_at(i, 0, 1));
        }
        p.read(0, 0); // 0 is now the most recently used, 1 is the least
        p.write(snap_at(9, 0, 1));

        assert!(p.read(0, 0).is_some(), "touched, so it survived");
        assert!(p.read(1, 0).is_none(), "untouched and oldest, so it went");
        assert!(p.read(2, 0).is_some());
        assert!(p.read(9, 0).is_some());
    }

    #[test]
    fn a_read_is_not_destructive() {
        let mut p = MemoryChunkPersistence::new(3);
        p.write(snap_at(0, 0, 77));
        assert_eq!(p.read(0, 0).expect("present").material[0], 77);
        assert_eq!(p.len(), 1, "reading leaves the snapshot in place");
        assert_eq!(p.read(0, 0).expect("still present").material[0], 77);
    }

    #[test]
    fn rewriting_a_key_replaces_rather_than_accumulates() {
        let mut p = MemoryChunkPersistence::new(3);
        p.write(snap_at(0, 0, 1));
        p.write(snap_at(0, 0, 2));
        assert_eq!(p.len(), 1);
        assert_eq!(p.read(0, 0).expect("present").material[0], 2);
    }

    #[test]
    fn negative_coordinates_are_distinct_keys() {
        // The string key `"${x},${y}"` this replaced was fine here too, but it is
        // worth pinning that the pair key keeps -1,2 and 1,-2 apart.
        let mut p = MemoryChunkPersistence::new(8);
        p.write(snap_at(-1, 2, 1));
        p.write(snap_at(1, -2, 2));
        assert_eq!(p.len(), 2);
        assert_eq!(p.read(-1, 2).expect("present").material[0], 1);
        assert_eq!(p.read(1, -2).expect("present").material[0], 2);
    }

    #[test]
    fn the_default_backend_uses_the_documented_cap() {
        assert_eq!(
            MemoryChunkPersistence::default().capacity(),
            MAX_PERSISTED_CHUNKS
        );
    }
}

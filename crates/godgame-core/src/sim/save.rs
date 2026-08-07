//! Durability: the on-disk format, and the chunk backend that speaks it.
//!
//! # What this closes
//!
//! `ChunkPersistence`'s own doc comment has said, since the port, that **"there
//! is no durable save behind this trait, and this port does not add one"**, and
//! that the boundary was kept narrow so "adding one later is a new
//! implementation of these three methods and no change anywhere else". This is
//! that implementation, and the claim held: nothing above the trait changed.
//!
//! # Why the format is hand-rolled
//!
//! `godgame-core` depends on `godgame-data`, `bitflags` and `rayon`, and that
//! list is short on purpose — it is the crate the purity suite and the benches
//! link. A serialisation framework would be pulled into all of them, and would
//! buy nothing here: a chunk is five flat arrays of fixed length over primitive
//! integers. There is no graph, no optionality, no versioned struct with fields
//! coming and going. `bin/worldgen-dump.rs` makes the same case for its PNG
//! encoder and `json.rs` for its writer.
//!
//! What a hand-rolled format DOES have to do, and this one does:
//!
//!   - **say what it is** — [`MAGIC`], so a wrong file is refused rather than
//!     interpreted;
//!   - **say what version it is** — [`VERSION`], so a format change is a clean
//!     refusal instead of a world full of garbage;
//!   - **say how big its arrays are** — the cell count is in the header, so a
//!     future [`CHUNK_CELLS`] cannot silently reinterpret old saves;
//!   - **never half-write** — see [`DiskChunkPersistence::write`].
//!
//! # One file per chunk
//!
//! Rather than one file with an index. A chunk is ~8 KB and there are at most
//! [`MAX_PERSISTED_CHUNKS`] of them, so this trades a few thousand small files
//! for three properties worth more than the inodes: a read is an `open` at a
//! known path with no index to consult, a write cannot corrupt a chunk it is not
//! writing, and a file that goes bad costs exactly one chunk of edits rather
//! than the world.
//!
//! [`MAX_PERSISTED_CHUNKS`]: super::chunk_store::MAX_PERSISTED_CHUNKS

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::config::CHUNK_CELLS;
use crate::sim::chunk::ChunkSnapshot;
use crate::sim::chunk_store::ChunkPersistence;
use crate::sim::grid::CellFlags;
use crate::sim::materials::CellId;

/// File signature. `GGCH` — GodGame CHunk.
///
/// Four bytes at offset zero, checked on every read. Without it, pointing the
/// loader at the wrong directory produces a world assembled out of whatever
/// those bytes happened to mean, which is far worse than a refusal: it looks
/// like a corrupted save rather than a mistake.
const MAGIC: [u8; 4] = *b"GGCH";

/// Format version. Bump on ANY layout change, including adding a plane.
///
/// A reader that meets a version it does not know refuses the file and the chunk
/// regenerates from the seed. That is the right failure: a pristine chunk is
/// exactly what worldgen would produce, so an unreadable save costs the player
/// the edits in that chunk and nothing else.
const VERSION: u16 = 1;

/// Header bytes before the planes: magic, version, cell count, coordinates.
const HEADER: usize = 4 + 2 + 2 + 4 + 4;

/// Cells in one chunk's plane.
const CELLS: usize = (CHUNK_CELLS * CHUNK_CELLS) as usize;

/// Bytes one encoded chunk occupies: header plus `u16`, `u8`, `u16`, `u8`, `u16`
/// planes.
const ENCODED: usize = HEADER + CELLS * (2 + 1 + 2 + 1 + 2);

// --- The format --------------------------------------------------------------

/// Encode a snapshot. Always exactly [`ENCODED`] bytes.
///
/// Little-endian throughout, stated once here rather than at each field: the
/// only machines this runs on are little-endian, and picking the native order
/// would make the file's meaning depend on where it was written — which is the
/// one thing a save format may not do.
pub fn encode_chunk(snap: &ChunkSnapshot) -> Vec<u8> {
    let mut out = Vec::with_capacity(ENCODED);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(CHUNK_CELLS as u16).to_le_bytes());
    out.extend_from_slice(&snap.chunk_x.to_le_bytes());
    out.extend_from_slice(&snap.chunk_y.to_le_bytes());

    for v in &snap.material {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &snap.flags {
        out.push(v.bits());
    }
    for v in &snap.aux {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&snap.temp);
    for v in &snap.back {
        out.extend_from_slice(&v.to_le_bytes());
    }
    debug_assert_eq!(out.len(), ENCODED);
    out
}

/// Decode a snapshot, or say why not.
///
/// Every rejection is a `None` rather than a panic. This runs on bytes that came
/// off a disk somebody else's program may also have written to, and the only
/// safe reading of "these bytes are not a chunk" is to regenerate the chunk.
pub fn decode_chunk(bytes: &[u8]) -> Option<ChunkSnapshot> {
    if bytes.len() != ENCODED || bytes[..4] != MAGIC {
        return None;
    }
    if u16::from_le_bytes([bytes[4], bytes[5]]) != VERSION {
        return None;
    }
    // The cell count is checked rather than assumed: an old save written when
    // CHUNK_CELLS was a different number would otherwise be read as this one's
    // planes at the wrong stride, which is garbage that parses.
    if u16::from_le_bytes([bytes[6], bytes[7]]) != CHUNK_CELLS as u16 {
        return None;
    }

    let mut at = 8;
    let i32_at = |at: &mut usize| {
        let v = i32::from_le_bytes(bytes[*at..*at + 4].try_into().expect("4 bytes"));
        *at += 4;
        v
    };
    let chunk_x = i32_at(&mut at);
    let chunk_y = i32_at(&mut at);

    let u16s = |at: &mut usize| {
        let v: Vec<u16> = bytes[*at..*at + CELLS * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        *at += CELLS * 2;
        v
    };
    let material: Vec<CellId> = u16s(&mut at);

    let flags: Vec<CellFlags> = bytes[at..at + CELLS]
        .iter()
        // `from_bits_truncate` and not `from_bits`: an unknown bit in a save
        // written by a newer build is a flag this build does not have, and
        // dropping it is right. The cell keeps its material, which is the part
        // the player put there.
        .map(|b| CellFlags::from_bits_truncate(*b))
        .collect();
    at += CELLS;

    let aux = u16s(&mut at);
    let temp = bytes[at..at + CELLS].to_vec();
    at += CELLS;
    let back: Vec<CellId> = u16s(&mut at);

    Some(ChunkSnapshot {
        chunk_x,
        chunk_y,
        material,
        flags,
        aux,
        temp,
        back,
    })
}

// --- The backend -------------------------------------------------------------

/// Chunks on disk, one file each, under a directory owned by one world.
///
/// The counterpart to `MemoryChunkPersistence`, and the first thing behind
/// `ChunkPersistence` that survives a process. There is no LRU here and that is
/// deliberate: the cap `MemoryChunkPersistence` enforces exists to bound RAM,
/// and this does not hold anything in RAM. A world that has been dug through for
/// a long time should keep every edit, not the most recent two thousand.
pub struct DiskChunkPersistence {
    dir: PathBuf,
    /// Files known to exist, so [`len`](ChunkPersistence::len) is not a
    /// directory scan per call.
    known: usize,
    /// Failures since the last check. See [`DiskChunkPersistence::errors`].
    errors: usize,
}

impl DiskChunkPersistence {
    /// Open (creating if needed) a world's chunk directory.
    ///
    /// Returns the error rather than swallowing it: failing to create the
    /// directory means nothing will ever be saved, and that is worth refusing to
    /// start over rather than discovering three hours in.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<DiskChunkPersistence> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let known = fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "chunk"))
            .count();
        Ok(DiskChunkPersistence {
            dir,
            known,
            errors: 0,
        })
    }

    /// How many reads or writes have failed.
    ///
    /// A count and not a log, because this crate has no logger and should not
    /// grow one. It exists so the host can notice a full or read-only disk and
    /// tell the player, rather than the save quietly not happening — which is
    /// the failure mode a save system must not have.
    pub fn errors(&self) -> usize {
        self.errors
    }

    /// `<dir>/<x>_<y>.chunk`.
    ///
    /// Coordinates in decimal with a `_` separator, negatives included, so the
    /// name is greppable and a human can find the chunk they are standing in
    /// from the F3 panel. A hash would be shorter and would make that
    /// impossible.
    fn path_of(&self, chunk_x: i32, chunk_y: i32) -> PathBuf {
        self.dir.join(format!("{chunk_x}_{chunk_y}.chunk"))
    }
}

impl ChunkPersistence for DiskChunkPersistence {
    fn read(&mut self, chunk_x: i32, chunk_y: i32) -> Option<ChunkSnapshot> {
        let path = self.path_of(chunk_x, chunk_y);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            // A missing file is the normal case — most chunks were never
            // edited — and is not an error. Anything else is.
            Err(e) if e.kind() == io::ErrorKind::NotFound => return None,
            Err(_) => {
                self.errors += 1;
                return None;
            }
        };
        let snap = decode_chunk(&bytes);
        if snap.is_none() {
            self.errors += 1;
        }
        // The coordinate is checked against the NAME as well as being read from
        // the file: a chunk whose contents say somewhere else would otherwise be
        // blitted into the wrong place in the world, which looks like corrupted
        // terrain rather than a bad file.
        snap.filter(|s| s.chunk_x == chunk_x && s.chunk_y == chunk_y)
    }

    /// Write a snapshot, atomically.
    ///
    /// Through a temporary file and a rename, which is atomic on every platform
    /// this runs on. A direct write that was interrupted — a crash, a full disk,
    /// a lid closing — would leave a truncated file that decodes to `None`, and
    /// the chunk would silently revert to pristine terrain. The player would
    /// lose the edits they made an hour ago rather than the ones they made a
    /// second ago, which is the worse of the two and the harder to explain.
    fn write(&mut self, snap: ChunkSnapshot) {
        let path = self.path_of(snap.chunk_x, snap.chunk_y);
        let existed = path.exists();
        let tmp = path.with_extension("chunk.tmp");
        let bytes = encode_chunk(&snap);
        if fs::write(&tmp, &bytes).is_err() || fs::rename(&tmp, &path).is_err() {
            self.errors += 1;
            let _ = fs::remove_file(&tmp);
            return;
        }
        if !existed {
            self.known += 1;
        }
    }

    fn len(&self) -> usize {
        self.known
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_snapshot(chunk_x: i32, chunk_y: i32) -> ChunkSnapshot {
        // Values that vary per cell and per plane, so a plane written into the
        // wrong offset cannot round-trip by accident.
        ChunkSnapshot {
            chunk_x,
            chunk_y,
            material: (0..CELLS).map(|i| (i % 53) as CellId).collect(),
            flags: (0..CELLS)
                .map(|i| CellFlags::from_bits_truncate((i % 4) as u8))
                .collect(),
            aux: (0..CELLS).map(|i| (i * 7 % 65535) as u16).collect(),
            temp: (0..CELLS).map(|i| (i % 251) as u8).collect(),
            back: (0..CELLS).map(|i| (i % 31) as CellId).collect(),
        }
    }

    /// A directory this test owns, emptied first so a previous run cannot make
    /// a later one pass.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("godgame-save-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_chunk_survives_a_round_trip_plane_for_plane() {
        let snap = a_snapshot(-7, 12);
        let back = decode_chunk(&encode_chunk(&snap)).expect("decodes");
        assert_eq!(back, snap);
    }

    #[test]
    fn an_encoded_chunk_is_exactly_the_size_the_header_promises() {
        assert_eq!(encode_chunk(&a_snapshot(0, 0)).len(), ENCODED);
    }

    #[test]
    fn every_way_a_file_can_be_wrong_decodes_to_nothing() {
        let good = encode_chunk(&a_snapshot(1, 1));

        assert!(decode_chunk(&[]).is_none(), "empty");
        assert!(decode_chunk(&good[..good.len() - 1]).is_none(), "truncated");

        let mut longer = good.clone();
        longer.push(0);
        assert!(decode_chunk(&longer).is_none(), "trailing bytes");

        let mut wrong_magic = good.clone();
        wrong_magic[0] = b'X';
        assert!(decode_chunk(&wrong_magic).is_none(), "magic");

        let mut wrong_version = good.clone();
        wrong_version[4] = VERSION.wrapping_add(1) as u8;
        assert!(decode_chunk(&wrong_version).is_none(), "version");

        let mut wrong_cells = good.clone();
        wrong_cells[6] = (CHUNK_CELLS as u16).wrapping_add(1) as u8;
        assert!(decode_chunk(&wrong_cells).is_none(), "cell count");
    }

    /// An unknown flag bit is dropped rather than refusing the file. The cell
    /// keeps its material, which is the part the player put there.
    #[test]
    fn a_flag_bit_this_build_does_not_know_costs_the_flag_and_not_the_chunk() {
        let snap = a_snapshot(0, 0);
        let mut bytes = encode_chunk(&snap);
        let flags_at = HEADER + CELLS * 2;
        bytes[flags_at] = 0xff;
        let back = decode_chunk(&bytes).expect("still decodes");
        assert_eq!(back.material, snap.material);
        assert_eq!(back.flags[0], CellFlags::from_bits_truncate(0xff));
    }

    #[test]
    fn a_chunk_written_to_disk_comes_back_from_a_fresh_backend() {
        let dir = scratch("roundtrip");
        let snap = a_snapshot(3, -4);
        {
            let mut p = DiskChunkPersistence::open(&dir).expect("open");
            assert!(p.is_empty());
            p.write(snap.clone());
            assert_eq!(p.len(), 1);
        }
        // A DIFFERENT backend over the same directory — which is the whole
        // point, and what `MemoryChunkPersistence` cannot do.
        let mut fresh = DiskChunkPersistence::open(&dir).expect("reopen");
        assert_eq!(fresh.len(), 1, "the count survives the process");
        assert_eq!(fresh.read(3, -4).as_ref(), Some(&snap));
        assert_eq!(
            fresh.read(0, 0),
            None,
            "an unedited chunk was never written"
        );
        assert_eq!(fresh.errors(), 0);
    }

    #[test]
    fn rewriting_a_chunk_replaces_it_without_counting_it_twice() {
        let dir = scratch("rewrite");
        let mut p = DiskChunkPersistence::open(&dir).expect("open");
        p.write(a_snapshot(1, 1));
        let mut second = a_snapshot(1, 1);
        second.material[0] = 42;
        p.write(second.clone());
        assert_eq!(p.len(), 1);
        assert_eq!(p.read(1, 1), Some(second));
    }

    /// A file that says it is somewhere else is refused. Blitting it would show
    /// up as corrupted terrain rather than as a bad file.
    #[test]
    fn a_chunk_claiming_the_wrong_coordinate_is_refused() {
        let dir = scratch("misplaced");
        let mut p = DiskChunkPersistence::open(&dir).expect("open");
        fs::write(dir.join("5_5.chunk"), encode_chunk(&a_snapshot(9, 9))).expect("write");
        assert_eq!(p.read(5, 5), None);
    }

    /// The whole point, end to end and without a renderer: edit the live world,
    /// flush, and find the edit in a NEW process's world.
    ///
    /// This is the test the first wiring attempt needed and did not have. The
    /// store is a write-back cache — a chunk reaches persistence when the window
    /// evicts it — so a player who digs and quits where they stand writes
    /// nothing at all unless something flushes. Zero files appeared on disk and
    /// the run above looked like it had saved.
    #[test]
    fn an_edit_flushed_from_the_live_window_is_there_for_the_next_world() {
        use crate::config::CELL_SIZE;
        use crate::sim::grid::CellGrid;
        use crate::sim::level::window_size;
        use crate::sim::window::WindowManager;

        let dir = scratch("flush");
        let (cols, rows) = window_size();
        let (at_x, at_y) = (cols / 2, rows / 2);

        let (before, after) = {
            let mut grid = CellGrid::new(cols, rows);
            let mut w =
                WindowManager::new(super::super::chunk_store::ChunkStore::with_persistence(
                    4242,
                    Box::new(DiskChunkPersistence::open(&dir).expect("open")),
                ));
            w.init(&mut grid, at_x, at_y);
            let wc = crate::sim::coords::WorldCell::new(
                grid.origin_cell_x() + at_x,
                grid.origin_cell_y() + at_y,
            );
            let before = grid.get_world(wc);
            // Something that is definitely not what worldgen put there.
            let after = if before == 0 { 41 } else { 0 };
            grid.set_world(wc, after);
            w.flush(&grid);
            let _ = CELL_SIZE;
            (before, after)
        };
        assert_ne!(before, after, "the edit has to change something");

        // A fresh world over the same directory, exactly as a relaunch is.
        let mut grid = CellGrid::new(cols, rows);
        let mut w = WindowManager::new(super::super::chunk_store::ChunkStore::with_persistence(
            4242,
            Box::new(DiskChunkPersistence::open(&dir).expect("reopen")),
        ));
        w.init(&mut grid, at_x, at_y);
        let wc = crate::sim::coords::WorldCell::new(
            grid.origin_cell_x() + at_x,
            grid.origin_cell_y() + at_y,
        );
        assert_eq!(
            grid.get_world(wc),
            after,
            "the edit did not survive: nothing reached the disk, or nothing read it back"
        );
    }

    #[test]
    fn a_corrupt_file_is_counted_rather_than_crashing() {
        let dir = scratch("corrupt");
        let mut p = DiskChunkPersistence::open(&dir).expect("open");
        fs::write(dir.join("2_2.chunk"), b"not a chunk at all").expect("write");
        assert_eq!(p.read(2, 2), None);
        assert_eq!(p.errors(), 1, "a bad file is reported, not hidden");
    }
}

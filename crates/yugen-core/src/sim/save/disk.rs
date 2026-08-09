//! The backend: chunks on disk, one file each.
//!
//! The counterpart to `MemoryChunkPersistence` and the first thing behind
//! `ChunkPersistence` that survives a process. Everything here is about
//! surviving a bad disk rather than about the format — the atomic write, the
//! coordinate cross-check against the filename, and the error count that exists
//! so a full or read-only disk is something the host can notice rather than a
//! save that quietly did not happen.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::sim::chunk::ChunkSnapshot;
use crate::sim::chunk_store::ChunkPersistence;

use super::chunk::{decode_chunk, encode_chunk};

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
    fn errors(&self) -> usize {
        self.errors
    }

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
    use super::super::testing::{a_snapshot, scratch};
    use super::*;

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
            let mut w = WindowManager::new(crate::sim::chunk_store::ChunkStore::with_persistence(
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
        let mut w = WindowManager::new(crate::sim::chunk_store::ChunkStore::with_persistence(
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
    fn the_error_count_reaches_the_host_through_the_trait() {
        // The count existed and had no reader, which made a full or read-only
        // disk a save that quietly did not happen. It is on the trait now, so
        // the host can put it somewhere a player sees.
        let dir = scratch("errs-trait");
        let mut p = DiskChunkPersistence::open(&dir).expect("open");
        assert_eq!(ChunkPersistence::errors(&p), 0);

        std::fs::write(dir.join("0_0.chunk"), b"not a chunk").expect("clobber");
        assert!(p.read(0, 0).is_none());
        assert_eq!(
            ChunkPersistence::errors(&p),
            1,
            "the trait reports what the backend counted"
        );
    }

    #[test]
    fn a_backend_with_nowhere_to_fail_reports_no_errors() {
        // The in-memory backend answers honestly by not overriding the default:
        // it has no disk to be full and no volume to be read-only, so zero is
        // the truth rather than a stub.
        use crate::sim::chunk_store::MemoryChunkPersistence;
        let m = MemoryChunkPersistence::new(8);
        assert_eq!(ChunkPersistence::errors(&m), 0);
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

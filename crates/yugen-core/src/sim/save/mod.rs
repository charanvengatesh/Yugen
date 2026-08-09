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
//! `yugen-core` depends on `yugen-data`, `bitflags` and `rayon`, and that
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

mod bytes;
mod chunk;
mod disk;
mod legacy;
mod meta;
mod run;

pub use chunk::{decode_chunk, encode_chunk};
pub use disk::DiskChunkPersistence;
pub use meta::{WORLD_NAME_MAX, WorldMeta, create_world, delete_world, list_worlds, slug_of};
pub use run::{BodyState, RunState, decode_run, encode_run, read_run, run_path, write_run};

/// Fixtures two of the format modules need and neither owns.
///
/// The split forced exactly one structural addition, and this is it. A chunk
/// snapshot is the subject of `chunk`'s round-trip tests and the payload of
/// `disk`'s; a scratch directory is plumbing for `disk` and `meta` alike.
/// Duplicating either would leave two fixtures free to drift into disagreeing
/// about what a chunk looks like, which is the one thing a round-trip test may
/// not be uncertain about.
#[cfg(test)]
pub(super) mod testing {
    use std::path::PathBuf;

    use std::fs;

    use super::chunk::CELLS;
    use super::run::{BodyState, RunState};
    use crate::sim::chunk::ChunkSnapshot;
    use crate::sim::grid::CellFlags;
    use crate::sim::materials::CellId;

    pub(in crate::sim::save) fn a_snapshot(chunk_x: i32, chunk_y: i32) -> ChunkSnapshot {
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
    pub(in crate::sim::save) fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("yugen-save-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    pub(in crate::sim::save) fn a_run() -> RunState {
        RunState {
            seed: 2334,
            clock_t: 91.5,
            body: Some(BodyState {
                x: -1234.5,
                y: 678.25,
                vx: -12.0,
                vy: 3.5,
                facing: -1.0,
                health: 63.0,
                untouchable: true,
            }),
            slots: vec![(0, 41, 1), (2, 7, 99), (29, 13, 5)],
            selected: 2,
            worn: Some(77),
        }
    }
}

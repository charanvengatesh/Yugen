//! Retired readers. Nothing in here is ever edited again.
//!
//! # The rule
//!
//! When an envelope version moves, its reader is COPIED here and then left
//! alone. It does not get refactored alongside the format it no longer
//! describes, it does not gain a field, and it does not learn about a section.
//! Its only job is to turn bytes that were written years ago into today's
//! in-memory type, and the moment it starts tracking anything it stops being
//! able to do that.
//!
//! This is why migration is always *old reader → today's struct* and never a
//! chain of struct-to-struct steps. There is one [`RunState`] in the tree, so
//! there is one hop, however many versions have been and gone. A chain would
//! mean every intermediate shape had to be kept alive as a type too, and the
//! cost of a version bump would grow with the number of bumps that preceded it.
//!
//! # What a reader here is allowed to assume
//!
//! Only what its own version guaranteed. When a field is added to [`RunState`]
//! after this reader was retired, the reader supplies whatever the absence of
//! that field meant at the time — which is a decision to make deliberately and
//! write down, not a `Default::default()` to reach for. `decode_run_v2` below
//! has no such case yet; the first one that appears should be argued for in a
//! comment on the line that produces it.
//!
//! # Fixtures, not round-trips
//!
//! A retired reader cannot be tested by encoding and decoding, because nothing
//! encodes its version any more. It is tested against committed bytes —
//! `tests/fixtures/save/` — produced once by the build that still wrote them
//! and never regenerated. A fixture that gets regenerated has stopped being
//! evidence about the past and become a copy of the present.

use super::bytes::Reader;
use super::chunk::CELLS;
use super::meta::{MetaStamps, WORLD_NAME_MAX};
use super::run::{BodyState, RunState};
use crate::config::CHUNK_CELLS;
use crate::sim::chunk::ChunkSnapshot;
use crate::sim::grid::CellFlags;
use crate::sim::materials::CellId;

/// Read a version-1 chunk file.
///
/// Five planes at fixed offsets, in a fixed order, with no table in front of
/// them: magic, version, cell count, `chunk_x`, `chunk_y`, then `material`
/// (u16), `flags` (u8), `aux` (u16), `temp` (u8) and `back` (u16), each
/// `CELLS` long. Exactly `16 + CELLS * 8` bytes, always.
///
/// Version 2 put a table in front so a plane could be added without a bump. The
/// five planes themselves are byte-identical either side of that change, so this
/// reader is the old offsets and nothing else.
///
/// The cell-count check matters here as much as it does in the live reader: a
/// file written when `CHUNK_CELLS` was a different number would otherwise be
/// read at the wrong stride, which is garbage that parses rather than a refusal.
pub fn decode_chunk_v1(bytes: &[u8]) -> Option<ChunkSnapshot> {
    const HEADER_V1: usize = 4 + 2 + 2 + 4 + 4;
    const ENCODED_V1: usize = HEADER_V1 + CELLS * (2 + 1 + 2 + 1 + 2);

    if bytes.len() != ENCODED_V1 || bytes[..4] != *b"GGCH" {
        return None;
    }
    if u16::from_le_bytes([bytes[4], bytes[5]]) != 1 {
        return None;
    }
    if u16::from_le_bytes([bytes[6], bytes[7]]) != CHUNK_CELLS as u16 {
        return None;
    }

    let chunk_x = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let chunk_y = i32::from_le_bytes(bytes[12..16].try_into().ok()?);

    let mut at = HEADER_V1;
    let u16s = |at: &mut usize| -> Vec<u16> {
        let v = bytes[*at..*at + CELLS * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        *at += CELLS * 2;
        v
    };
    let material: Vec<CellId> = u16s(&mut at);
    let flags: Vec<CellFlags> = bytes[at..at + CELLS]
        .iter()
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

/// Read a version-2 run file.
///
/// The format `RUN_VERSION = 2` wrote, exactly: magic, version, seed, clock,
/// a body-present byte and its six floats plus the untouchable flag, the
/// selected slot, a worn-present byte and its code, then a slot count and that
/// many `(slot, code, count)` triples. Trailing bytes were refused and still
/// are.
///
/// Version 3 turned all of that into sections. Every field here has a section
/// there, so this is a pure re-shaping with nothing to invent — the reason it
/// is a migration at all rather than the refusal version 2 gave version 1.
pub fn decode_run_v2(bytes: &[u8]) -> Option<RunState> {
    let mut r = Reader { bytes, at: 0 };
    // Magic and version are re-read rather than skipped: this function is
    // reachable from a test with a raw fixture, not only from `decode_run`
    // after it has already checked them.
    if r.take(4)? != *b"GGRN" || r.u16()? != 2 {
        return None;
    }
    let seed = r.u32()?;
    let clock_t = r.f32()?;
    let body = match r.u8()? {
        0 => None,
        _ => Some(BodyState {
            x: r.f32()?,
            y: r.f32()?,
            vx: r.f32()?,
            vy: r.f32()?,
            facing: r.f32()?,
            health: r.f32()?,
            untouchable: r.u8()? != 0,
        }),
    };
    let selected = r.u16()?;
    let worn = match r.u8()? {
        0 => None,
        _ => Some(r.u16()?),
    };
    let n = r.u16()? as usize;
    let mut slots = Vec::with_capacity(n);
    for _ in 0..n {
        slots.push((r.u16()?, r.u16()?, r.u16()?));
    }
    (r.at == bytes.len()).then_some(RunState {
        seed,
        clock_t,
        body,
        slots,
        selected,
        worn,
    })
}

/// Read a version-1 identity file.
///
/// Magic, version, seed, name length, name. That is all version 1 recorded, so
/// every stamp version 2 adds has to be invented here — and what each absence is
/// taken to mean is a decision, not a default:
///
///   `created_at`   **0**, meaning unknown. Not "now": a world made last year
///                  would start claiming it was made the moment its owner first
///                  ran a build that could ask.
///   `last_played`  **0**, and the caller substitutes `run.save`'s mtime. That
///                  is exactly what version 1 did, so an unmigrated world keeps
///                  the ordering it has always had rather than jumping to the
///                  top or the bottom of the list.
///   `play_seconds` **0**. The time was never counted and cannot be recovered.
///   `permadeath`   **false**. It did not exist, so no world predating it was
///                  created under it — and guessing the other way would turn a
///                  player's existing world into one that deletes itself.
pub fn decode_meta_v1(bytes: &[u8]) -> Option<(String, u32, MetaStamps)> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != *b"GGWD" || r.u16()? != 1 {
        return None;
    }
    let seed = r.u32()?;
    let n = r.u16()? as usize;
    if n > WORLD_NAME_MAX {
        return None;
    }
    let name = std::str::from_utf8(r.take(n)?).ok()?.to_string();
    (r.at == bytes.len()).then_some((
        name,
        seed,
        MetaStamps {
            created_at: 0,
            last_played: 0,
            play_seconds: 0,
            permadeath: false,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A version-2 run, byte for byte, written by hand rather than by a
    /// function — which is the point. The bytes are the specification; a helper
    /// that built them would be a second implementation of the format, and the
    /// two could agree with each other while both being wrong about what is on
    /// somebody's disk.
    fn v2_bytes() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGRN");
        b.extend_from_slice(&2u16.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes()); // seed
        b.extend_from_slice(&12.5f32.to_le_bytes()); // clock
        b.push(1); // body present
        for v in [1.0f32, 2.0, 3.0, 4.0, -1.0, 63.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.push(1); // untouchable
        b.extend_from_slice(&3u16.to_le_bytes()); // selected
        b.push(1); // worn present
        b.extend_from_slice(&11u16.to_le_bytes()); // worn code
        b.extend_from_slice(&2u16.to_le_bytes()); // slot count
        for (slot, code, n) in [(0u16, 5u16, 9u16), (4, 6, 1)] {
            b.extend_from_slice(&slot.to_le_bytes());
            b.extend_from_slice(&code.to_le_bytes());
            b.extend_from_slice(&n.to_le_bytes());
        }
        b
    }

    #[test]
    fn a_version_two_run_still_loads_everything_it_held() {
        let run = decode_run_v2(&v2_bytes()).expect("a valid v2 run");
        assert_eq!(run.seed, 7);
        assert_eq!(run.clock_t, 12.5);
        let body = run.body.expect("a body");
        assert_eq!((body.x, body.y, body.vx, body.vy), (1.0, 2.0, 3.0, 4.0));
        assert_eq!((body.facing, body.health), (-1.0, 63.0));
        assert!(body.untouchable);
        assert_eq!(run.selected, 3);
        assert_eq!(run.worn, Some(11));
        assert_eq!(run.slots, vec![(0, 5, 9), (4, 6, 1)]);
    }

    #[test]
    fn the_live_reader_migrates_a_version_two_file_rather_than_refusing_it() {
        // The behaviour change this version exists for. Under version equality
        // this file decoded to `None` and cost the player their position and
        // their pack while the terrain beside it survived.
        let migrated = super::super::run::decode_run(&v2_bytes()).expect("v2 migrates");
        assert_eq!(migrated, decode_run_v2(&v2_bytes()).expect("same file"));
    }

    #[test]
    fn a_migrated_run_re_encodes_as_the_current_version() {
        // Migration is one hop to today's struct, so what gets written back is
        // an ordinary v3 file — the old shape does not survive the load.
        let migrated = super::super::run::decode_run(&v2_bytes()).expect("v2 migrates");
        let re = super::super::run::encode_run(&migrated);
        assert_eq!(&re[4..6], &3u16.to_le_bytes(), "re-encoded at v3");
        assert_eq!(
            super::super::run::decode_run(&re).expect("round trip"),
            migrated
        );
    }

    /// A version-1 chunk, byte for byte: header then five planes at fixed
    /// offsets, no table. Written out here for the same reason `v2_bytes` is —
    /// the bytes are the specification.
    fn v1_chunk_bytes() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGCH");
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&(CHUNK_CELLS as u16).to_le_bytes());
        b.extend_from_slice(&(-7i32).to_le_bytes());
        b.extend_from_slice(&12i32.to_le_bytes());
        for i in 0..CELLS {
            b.extend_from_slice(&((i % 53) as u16).to_le_bytes()); // material
        }
        b.extend(std::iter::repeat_n(1u8, CELLS)); // flags
        for i in 0..CELLS {
            b.extend_from_slice(&((i % 7) as u16).to_le_bytes()); // aux
        }
        b.extend((0..CELLS).map(|i| (i % 251) as u8)); // temp
        for i in 0..CELLS {
            b.extend_from_slice(&((i % 11) as u16).to_le_bytes()); // back
        }
        b
    }

    #[test]
    fn a_version_one_chunk_still_loads_every_plane_it_held() {
        let snap = decode_chunk_v1(&v1_chunk_bytes()).expect("a valid v1 chunk");
        assert_eq!((snap.chunk_x, snap.chunk_y), (-7, 12));
        assert_eq!(snap.material[52], 52);
        assert_eq!(snap.aux[6], 6);
        assert_eq!(snap.temp[250], 250);
        assert_eq!(snap.back[10], 10);
        assert!(snap.flags.iter().all(|f| f.bits() == 1));
    }

    #[test]
    fn the_live_reader_migrates_a_version_one_chunk_rather_than_refusing_it() {
        // The bug this closes is the quietest one in the tree. A refused chunk
        // regenerates, and a regenerated chunk is indistinguishable from one
        // nobody ever dug — so under version equality, adding a sixth plane
        // would have erased every edit in every world with no error anywhere.
        let bytes = v1_chunk_bytes();
        let migrated = super::super::chunk::decode_chunk(&bytes).expect("v1 migrates");
        assert_eq!(migrated, decode_chunk_v1(&bytes).expect("same file"));

        // And it comes back out as an ordinary v2 chunk, table and all.
        let re = super::super::chunk::encode_chunk(&migrated);
        assert_eq!(&re[4..6], &2u16.to_le_bytes(), "re-encoded at v2");
        assert_eq!(
            super::super::chunk::decode_chunk(&re).expect("round trip"),
            migrated
        );
    }

    #[test]
    fn no_truncation_of_a_version_one_chunk_decodes_to_something() {
        let full = v1_chunk_bytes();
        // Every 97th prefix rather than all 8208: the length check is a single
        // equality, so the property is uniform and sampling it keeps the suite
        // fast enough that nobody is tempted to delete it.
        for n in (0..full.len()).step_by(97) {
            assert!(
                decode_chunk_v1(&full[..n]).is_none(),
                "a {n}-byte prefix decoded to a chunk"
            );
        }
    }

    #[test]
    fn no_truncation_of_a_version_two_run_decodes_to_something() {
        // The property the retired reader has to keep as much as the live one:
        // every prefix of a valid file is refused, so a half-written file from
        // a build that predates the atomic rename cannot load as a shorter run.
        let full = v2_bytes();
        for n in 0..full.len() {
            assert!(
                decode_run_v2(&full[..n]).is_none(),
                "a {n}-byte prefix decoded to a run"
            );
        }
    }
}

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
use super::run::{BodyState, RunState};

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

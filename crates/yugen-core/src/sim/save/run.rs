//! The run file: `GGRN`, everything about a run that is not terrain.
//!
//! Terrain lives in the chunk files beside this; a world is the two together.
//! What belongs here is what a player would say out loud — where they are, what
//! they are carrying, what time it is — and deliberately not the per-frame
//! state a body accumulates. See [`BodyState`] for that argument and
//! `docs/SAVE.md` for the byte layout.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::bytes::Reader;

// --- The run -----------------------------------------------------------------

/// File signature for the run file. `GGRN` — Yūgen RuN.
const RUN_MAGIC: [u8; 4] = *b"GGRN";

/// Run-file envelope version, independent of the chunk's.
///
/// Two formats, two versions. A chunk and a run change for different reasons —
/// adding a cell plane and adding an equipment slot have nothing to do with each
/// other — and one shared number would force a world's terrain to be discarded
/// because its inventory format moved.
///
/// **2** added the worn armour slot, and refused a version-1 run rather than
/// reading it with a zero in it. That was the right call for one field and does
/// not survive being made five more times: `docs/DEATH.md` alone wants a respawn
/// point, a death cause, a death count, a corpse bag and a permadeath flag.
///
/// **3** is the last time this number should have to move. The payload is a list
/// of tagged sections, so a new field is a new section rather than a new
/// version — see [`encode_run`] for the rules and `docs/SAVE.md` for why they
/// are the rules. A version-2 run is now MIGRATED rather than refused, by
/// [`super::legacy::decode_run_v2`], which is what a version bump is supposed to
/// cost: a short function that lives forever, not a player's afternoon.
const RUN_VERSION: u16 = 3;

/// Sections this build knows how to read, in the order it writes them.
///
/// Ascending by tag bytes, which is also the order [`decode_run`] enforces.
/// Sorting by the tag rather than by a registry index is what lets a section
/// from a NEWER build slot in among these without breaking the ordering rule:
/// the reader can check `>` without knowing what the tag means.
mod tag {
    /// The world clock, `f32` seconds.
    pub const CLOK: [u8; 4] = *b"CLOK";
    /// The body: six `f32` and the untouchable flag. Absent = no body.
    pub const BODY: [u8; 4] = *b"BODY";
    /// The pack: selected slot, then the occupied slots.
    pub const PACK: [u8; 4] = *b"PACK";
    /// The worn item code. Absent = wearing nothing.
    pub const WORN: [u8; 4] = *b"WORN";
}

/// The body, as a save file sees it.
///
/// Deliberately not `Player`. That type carries a dozen fields of derived
/// per-frame state — coyote time, hurt cooldown, swing phase, pose — and none of
/// it should survive a reload: a body that loads mid-flinch, or with a dash
/// already half-spent, is carrying a moment that no longer exists. Only what a
/// player would say out loud is here.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyState {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub facing: f32,
    pub health: f32,
    pub untouchable: bool,
}

/// Everything about a run that is not terrain.
///
/// Terrain lives in the chunk files beside this; a world is the two together.
#[derive(Clone, Debug, PartialEq)]
pub struct RunState {
    /// The seed the world was grown from.
    ///
    /// Stored so a run file can be REFUSED when it does not belong to the world
    /// it was found next to. Without it, dropping a save into the wrong
    /// directory teleports the body into terrain generated from another seed,
    /// which reads as a corrupted world rather than as a mistake.
    pub seed: u32,
    /// The world clock, in seconds. Time of day is part of where you left off.
    pub clock_t: f32,
    /// The body, or `None` for a run with no player (`--free-camera`).
    pub body: Option<BodyState>,
    /// Occupied slots as `(slot, item code, count)`, and the selected slot.
    ///
    /// Slot indices are stored rather than a dense array, so an empty inventory
    /// is a few bytes and a diff of two saves points at the slot that changed.
    ///
    /// **Item CODES, not ids.** That is safe here and would not be in general:
    /// `content/ids.lock.json` pins id -> code so a reordered content file
    /// cannot renumber anything, and `registry_golden`'s prefix rule means a new
    /// item is appended above the boundary and never takes an existing code. If
    /// either of those ever stops being true, this needs ids and a
    /// [`RUN_VERSION`] bump — a silently reassigned code turns the player's gold
    /// into gravel with nothing to notice it.
    pub slots: Vec<(u16, u16, u16)>,
    /// The selected hotbar slot.
    pub selected: u16,
    /// The item code being worn, or `None`.
    ///
    /// Outside `slots`, because it is outside the pack — see
    /// `Inventory::equip`. Folding it in as a 31st slot would make every reader
    /// of `slots` know about a slot the pack does not have.
    pub worn: Option<u16>,
}

/// Encode a run.
///
/// Envelope, then a count, then that many sections of `(tag, length, payload)`.
/// Three rules make the format additive, and each one is checked on the way back
/// in by [`decode_run`]:
///
///   **Sections ascend by tag.** Which gives uniqueness and a canonical byte
///   order for free, and — because the comparison is on the tag rather than on
///   a registry position — lets a section from a newer build sort into place
///   without this build knowing what it is.
///
///   **A length precedes every payload.** So a reader that does not recognise a
///   tag can step over it exactly, rather than having to understand it or give
///   up on the file.
///
///   **An absent section is a documented default, never a refusal.** `CLOK` and
///   `PACK` are always written because they always have a value; `BODY` and
///   `WORN` are written only when they are `Some`, so presence carries the
///   optionality and there is no present-flag byte inside the payload.
///
/// Canonical for files THIS build writes: `encode(decode(bytes)) == bytes`. It
/// is deliberately not claimed for foreign files — one that omits `CLOK` reads
/// back as zero and re-encodes with the section present, which is a widening,
/// not a disagreement.
pub fn encode_run(run: &RunState) -> Vec<u8> {
    let mut sections: Vec<([u8; 4], Vec<u8>)> = Vec::new();

    if let Some(b) = run.body {
        let mut p = Vec::with_capacity(25);
        for v in [b.x, b.y, b.vx, b.vy, b.facing, b.health] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        p.push(u8::from(b.untouchable));
        sections.push((tag::BODY, p));
    }

    sections.push((tag::CLOK, run.clock_t.to_le_bytes().to_vec()));

    let mut pack = Vec::with_capacity(4 + run.slots.len() * 6);
    pack.extend_from_slice(&run.selected.to_le_bytes());
    pack.extend_from_slice(&(run.slots.len() as u16).to_le_bytes());
    for (slot, code, n) in &run.slots {
        pack.extend_from_slice(&slot.to_le_bytes());
        pack.extend_from_slice(&code.to_le_bytes());
        pack.extend_from_slice(&n.to_le_bytes());
    }
    sections.push((tag::PACK, pack));

    if let Some(code) = run.worn {
        sections.push((tag::WORN, code.to_le_bytes().to_vec()));
    }

    sections.sort_by_key(|(t, _)| *t);

    let mut out = Vec::with_capacity(32 + run.slots.len() * 6);
    out.extend_from_slice(&RUN_MAGIC);
    out.extend_from_slice(&RUN_VERSION.to_le_bytes());
    out.extend_from_slice(&run.seed.to_le_bytes());
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for (t, payload) in &sections {
        out.extend_from_slice(t);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
    }
    out
}

/// Decode a run, or `None` if these bytes are not one.
///
/// A version-2 file is handed to [`super::legacy::decode_run_v2`] rather than
/// refused. That is the whole point of the envelope: an old reader is a short
/// function kept forever, and migration is always old-reader-to-today rather
/// than a chain of struct-to-struct steps, because there is only ever one
/// in-memory [`RunState`].
pub fn decode_run(bytes: &[u8]) -> Option<RunState> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != RUN_MAGIC {
        return None;
    }
    match r.u16()? {
        RUN_VERSION => {}
        2 => return super::legacy::decode_run_v2(bytes),
        _ => return None,
    }

    let seed = r.u32()?;
    let count = r.u32()?;

    let mut clock_t = 0.0;
    let mut body = None;
    let mut selected = 0;
    let mut slots = Vec::new();
    let mut worn = None;
    let mut last: Option<[u8; 4]> = None;

    for _ in 0..count {
        let t: [u8; 4] = r.take(4)?.try_into().ok()?;
        // Strictly ascending: equal is a duplicate, less is out of order, and
        // both mean the file was not written by anything that agreed with this
        // format. Refusing keeps "two sections saying different things" from
        // ever being a state a reader has to have an opinion about.
        if last.is_some_and(|prev| t <= prev) {
            return None;
        }
        last = Some(t);

        let len = r.u32()? as usize;
        let payload = r.take(len)?;
        let mut p = Reader {
            bytes: payload,
            at: 0,
        };
        match t {
            tag::CLOK => clock_t = p.f32()?,
            tag::BODY => {
                body = Some(BodyState {
                    x: p.f32()?,
                    y: p.f32()?,
                    vx: p.f32()?,
                    vy: p.f32()?,
                    facing: p.f32()?,
                    health: p.f32()?,
                    untouchable: p.u8()? != 0,
                });
            }
            tag::PACK => {
                selected = p.u16()?;
                let n = p.u16()? as usize;
                slots = Vec::with_capacity(n);
                for _ in 0..n {
                    slots.push((p.u16()?, p.u16()?, p.u16()?));
                }
            }
            tag::WORN => worn = Some(p.u16()?),
            // A section this build does not know is stepped over by its declared
            // length and otherwise ignored. This is the clause that makes the
            // format additive: a save from a newer build loads here, minus
            // whatever it knew that this one does not.
            _ => p.at = payload.len(),
        }
        // A payload longer than the section says it is means the writer and this
        // reader disagree about what the tag means, which is exactly the case
        // rule 3 of `docs/SAVE.md` forbids.
        if p.at != payload.len() {
            return None;
        }
    }

    // Trailing bytes mean this is not the file it claims to be.
    (r.at == bytes.len()).then_some(RunState {
        seed,
        clock_t,
        body,
        slots,
        selected,
        worn,
    })
}

/// Where a world's run file lives, given its directory.
pub fn run_path(dir: impl AsRef<Path>) -> PathBuf {
    dir.as_ref().join("run.save")
}

/// Write a run beside its chunks, atomically, for [`DiskChunkPersistence::write`]'s
/// reason: a half-written run file would load as no run at all.
pub fn write_run(dir: impl AsRef<Path>, run: &RunState) -> io::Result<()> {
    let path = run_path(&dir);
    let tmp = path.with_extension("save.tmp");
    fs::write(&tmp, encode_run(run))?;
    fs::rename(&tmp, &path)
}

/// Read a run, if there is a valid one for `seed`.
///
/// `None` covers every way there might not be: no file, an unreadable one, a
/// format this build does not know, and a run belonging to a different world.
/// All four mean the same thing to the caller — start fresh — and distinguishing
/// them would only tempt somebody to load three of the four anyway.
pub fn read_run(dir: impl AsRef<Path>, seed: u32) -> Option<RunState> {
    let bytes = fs::read(run_path(&dir)).ok()?;
    decode_run(&bytes).filter(|r| r.seed == seed)
}

#[cfg(test)]
mod tests {
    use super::super::testing::{a_run, scratch};
    use super::*;

    #[test]
    fn a_run_survives_a_round_trip() {
        assert_eq!(decode_run(&encode_run(&a_run())), Some(a_run()));
    }

    #[test]
    fn a_run_with_no_body_round_trips_as_one() {
        let mut run = a_run();
        run.body = None;
        run.slots.clear();
        assert_eq!(decode_run(&encode_run(&run)), Some(run));
    }

    /// Every prefix of a valid file must decode to nothing rather than to a
    /// plausible half-run. This is what the bounds-checked reader buys.
    #[test]
    fn no_truncation_of_a_run_decodes_to_something() {
        let good = encode_run(&a_run());
        for n in 0..good.len() {
            assert!(
                decode_run(&good[..n]).is_none(),
                "prefix of {n} bytes decoded"
            );
        }
        let mut longer = good.clone();
        longer.push(0);
        assert!(decode_run(&longer).is_none(), "trailing bytes");
    }

    #[test]
    fn a_run_from_another_world_is_refused() {
        let dir = scratch("otherworld");
        fs::create_dir_all(&dir).expect("mkdir");
        write_run(&dir, &a_run()).expect("write");
        assert_eq!(read_run(&dir, 2334).as_ref(), Some(&a_run()));
        assert_eq!(
            read_run(&dir, 9999),
            None,
            "a different seed's run is not ours"
        );
        assert_eq!(read_run(dir.join("nope"), 2334), None, "no file at all");
    }

    /// The slots go back exactly where they were, which is the reason
    /// `Inventory::put_at` exists rather than the loader calling `add`.
    #[test]
    fn an_inventory_reloads_into_the_same_slots_it_left() {
        use crate::items::inventory::Inventory;
        let run = a_run();
        let mut inv = Inventory::new();
        for (slot, code, n) in &run.slots {
            inv.put_at(*slot as usize, *code, *n);
        }
        inv.select_slot(run.selected as usize);
        assert_eq!(inv.stack_at(0), Some((41, 1)));
        assert_eq!(inv.stack_at(1), None, "the gap stayed a gap");
        assert_eq!(inv.stack_at(2), Some((7, 99)));
        assert_eq!(inv.stack_at(29), Some((13, 5)));
        assert_eq!(inv.selected(), 2);
    }

    /// Build a v3 file from raw sections, so a test can write what no build
    /// would — out of order, repeated, or tagged with something from a future
    /// this one has not seen.
    fn v3_with(seed: u32, sections: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&RUN_MAGIC);
        out.extend_from_slice(&3u16.to_le_bytes());
        out.extend_from_slice(&seed.to_le_bytes());
        out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
        for (t, payload) in sections {
            out.extend_from_slice(t);
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        out
    }

    #[test]
    fn a_section_this_build_does_not_know_is_stepped_over() {
        // The clause the whole version exists for. A save written by a build
        // that has `docs/DEATH.md`'s respawn point in it must load here, minus
        // the respawn point — not be refused for containing a tag this build
        // has never heard of.
        //
        // `ZZZZ` sorts last on purpose in one case and `AAAA` first in the
        // other, so the skip is exercised both after a known section and before
        // one: a reader that only handled trailing unknowns would pass a test
        // with just the first.
        let clock = 9.5f32.to_le_bytes().to_vec();
        let bytes = v3_with(
            5,
            &[
                (*b"AAAA", vec![1, 2, 3]),
                (tag::CLOK, clock.clone()),
                (*b"ZZZZ", vec![7; 40]),
            ],
        );
        let run = decode_run(&bytes).expect("unknown sections are not fatal");
        assert_eq!(run.seed, 5);
        assert_eq!(run.clock_t, 9.5, "the known section still read");
        assert_eq!(run.body, None, "an absent BODY is no body");
        assert!(run.slots.is_empty(), "an absent PACK is an empty pack");
        assert_eq!(run.worn, None);
    }

    #[test]
    fn sections_out_of_order_or_repeated_are_refused() {
        // Ascending strictly is what makes the byte order canonical and makes
        // "two sections disagreeing about the clock" a state no reader has to
        // have an opinion about.
        let a = 1.0f32.to_le_bytes().to_vec();
        let b = 2.0f32.to_le_bytes().to_vec();
        assert!(
            decode_run(&v3_with(
                1,
                &[(tag::PACK, vec![0, 0, 0, 0]), (tag::CLOK, a.clone())]
            ))
            .is_none(),
            "descending tags were accepted"
        );
        assert!(
            decode_run(&v3_with(1, &[(tag::CLOK, a), (tag::CLOK, b)])).is_none(),
            "a repeated tag was accepted"
        );
    }

    #[test]
    fn a_payload_that_is_not_the_length_it_claims_is_refused() {
        // A writer and this reader disagreeing about what a tag means is the
        // one thing the section rules forbid outright, because silently reading
        // the first four bytes of a longer CLOK would be reinterpreting a
        // payload — see `docs/SAVE.md` rule 3.
        let mut too_long = 1.0f32.to_le_bytes().to_vec();
        too_long.push(0);
        assert!(decode_run(&v3_with(1, &[(tag::CLOK, too_long)])).is_none());
        assert!(decode_run(&v3_with(1, &[(tag::CLOK, vec![0, 0])])).is_none());
    }

    #[test]
    fn a_file_with_no_sections_at_all_is_a_run_at_every_default() {
        // Not a refusal. The defaults are the format's, written down here so a
        // later section cannot quietly change what its own absence means.
        let run = decode_run(&v3_with(42, &[])).expect("an empty section list is valid");
        assert_eq!(
            run,
            RunState {
                seed: 42,
                clock_t: 0.0,
                body: None,
                slots: Vec::new(),
                selected: 0,
                worn: None,
            }
        );
    }

    #[test]
    fn a_file_this_build_wrote_is_canonical() {
        // Sections are sorted on the way out, so one state has exactly one
        // encoding — which is what lets a save diff be read and this test be an
        // equality rather than a round-trip.
        for run in [
            a_run(),
            RunState {
                body: None,
                worn: None,
                slots: Vec::new(),
                ..a_run()
            },
        ] {
            let once = encode_run(&run);
            let twice = encode_run(&decode_run(&once).expect("round trip"));
            assert_eq!(once, twice, "re-encoding moved a byte");
        }
    }
}

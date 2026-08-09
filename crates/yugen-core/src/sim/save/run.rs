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

/// Run-file version, independent of [`VERSION`].
///
/// Two formats, two versions. A chunk and a run change for different reasons —
/// adding a cell plane and adding an equipment slot have nothing to do with each
/// other — and one shared number would force a world's terrain to be discarded
/// because its inventory format moved.
///
/// **2** adds the worn armour slot. A version-1 run is refused rather than read
/// with a zero in it: the difference between "wearing nothing" and "this file
/// predates armour" is invisible afterwards, and the terrain beside it survives
/// either way, so the cost of refusing is a body back at the spawn.
const RUN_VERSION: u16 = 2;

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
pub fn encode_run(run: &RunState) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + run.slots.len() * 6);
    out.extend_from_slice(&RUN_MAGIC);
    out.extend_from_slice(&RUN_VERSION.to_le_bytes());
    out.extend_from_slice(&run.seed.to_le_bytes());
    out.extend_from_slice(&run.clock_t.to_le_bytes());

    match run.body {
        Some(b) => {
            out.push(1);
            for v in [b.x, b.y, b.vx, b.vy, b.facing, b.health] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.push(u8::from(b.untouchable));
        }
        None => out.push(0),
    }

    out.extend_from_slice(&run.selected.to_le_bytes());
    match run.worn {
        Some(code) => {
            out.push(1);
            out.extend_from_slice(&code.to_le_bytes());
        }
        None => out.push(0),
    }
    out.extend_from_slice(&(run.slots.len() as u16).to_le_bytes());
    for (slot, code, n) in &run.slots {
        out.extend_from_slice(&slot.to_le_bytes());
        out.extend_from_slice(&code.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
    }
    out
}

/// Decode a run, or `None` if these bytes are not one.
pub fn decode_run(bytes: &[u8]) -> Option<RunState> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != RUN_MAGIC || r.u16()? != RUN_VERSION {
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
}

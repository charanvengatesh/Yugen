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

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::config::CHUNK_CELLS;
use crate::sim::chunk::ChunkSnapshot;
use crate::sim::chunk_store::ChunkPersistence;
use crate::sim::grid::CellFlags;
use crate::sim::materials::CellId;

/// File signature. `GGCH` — Yūgen CHunk.
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

/// A bounds-checked forward reader.
///
/// Every accessor returns `Option`, so a truncated file falls out as `None` at
/// the first short read instead of panicking somewhere deeper. The chunk format
/// does not need this — its length is fixed and checked once — and a run file's
/// is not.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let out = self.bytes.get(self.at..self.at + n)?;
        self.at += n;
        Some(out)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
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

// --- Worlds ------------------------------------------------------------------

/// File signature for a world's identity file. `GGWD` — Yūgen WorlD.
const META_MAGIC: [u8; 4] = *b"GGWD";

/// Identity-file version. Independent of the chunk and run versions, for the
/// reason [`RUN_VERSION`] gives.
const META_VERSION: u16 = 1;

/// Longest display name a world may have.
///
/// A limit exists so a name cannot be used to write an unbounded file or to
/// produce a menu row nothing can lay out. 48 is comfortably more than anybody
/// types and short enough to render at one of `ui`'s faces without wrapping.
pub const WORLD_NAME_MAX: usize = 48;

/// One saved world: what it is called, what it was grown from, and where it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldMeta {
    /// What the player called it. Free text, [`WORLD_NAME_MAX`] at most.
    pub name: String,
    /// The seed. Recorded at CREATION rather than inferred from the run file,
    /// because a world that has never been saved has no run file and would
    /// otherwise have no seed until the first autosave — at which point it
    /// would be whatever the loader happened to guess.
    pub seed: u32,
    /// The directory holding `world.meta`, `run.save` and `chunks/`.
    pub dir: PathBuf,
}

/// Turn a display name into a directory name.
///
/// Lowercase ASCII alphanumerics and `-`; everything else becomes `-`, runs
/// collapse, and the ends are trimmed. A name that survives none of that becomes
/// `world`.
///
/// This is a SECURITY boundary as much as a tidiness one. The name comes from a
/// text field, and a directory built by joining it raw would accept `..` and
/// `/` and write wherever the player typed. Rejecting instead of rewriting was
/// the alternative and is worse here: it turns naming a world after a place with
/// an apostrophe into an error message.
pub fn slug_of(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "world".to_string()
    } else {
        trimmed[..trimmed.len().min(WORLD_NAME_MAX)].to_string()
    }
}

fn encode_meta(name: &str, seed: u32) -> Vec<u8> {
    let name = &name[..name.len().min(WORLD_NAME_MAX)];
    let mut out = Vec::with_capacity(16 + name.len());
    out.extend_from_slice(&META_MAGIC);
    out.extend_from_slice(&META_VERSION.to_le_bytes());
    out.extend_from_slice(&seed.to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out
}

fn decode_meta(bytes: &[u8]) -> Option<(String, u32)> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != META_MAGIC || r.u16()? != META_VERSION {
        return None;
    }
    let seed = r.u32()?;
    let n = r.u16()? as usize;
    if n > WORLD_NAME_MAX {
        return None;
    }
    let name = std::str::from_utf8(r.take(n)?).ok()?.to_string();
    (r.at == bytes.len()).then_some((name, seed))
}

/// Create a world under `root`, in its own directory.
///
/// The directory is [`slug_of`] the name, with `-2`, `-3` and so on appended
/// until it is free — so two worlds called "Home" are two worlds rather than one
/// silently overwriting the other, which is the single worst thing a save menu
/// can do.
pub fn create_world(root: impl AsRef<Path>, name: &str, seed: u32) -> io::Result<WorldMeta> {
    let root = root.as_ref();
    fs::create_dir_all(root)?;
    let base = slug_of(name);
    let mut dir = root.join(&base);
    let mut n = 2;
    while dir.exists() {
        dir = root.join(format!("{base}-{n}"));
        n += 1;
    }
    fs::create_dir_all(&dir)?;
    let name = name[..name.len().min(WORLD_NAME_MAX)].to_string();
    fs::write(dir.join("world.meta"), encode_meta(&name, seed))?;
    Ok(WorldMeta { name, seed, dir })
}

/// Every world under `root`, most recently played first.
///
/// Ordered by the modification time of the run file, so the world you were last
/// in is the one already selected when the menu opens. A directory with no
/// readable `world.meta` is skipped rather than reported: the saves root is a
/// place a player may well have put something of their own, and a stray folder
/// is not an error.
pub fn list_worlds(root: impl AsRef<Path>) -> Vec<WorldMeta> {
    let Ok(entries) = fs::read_dir(root.as_ref()) else {
        return Vec::new();
    };
    let mut found: Vec<(Option<std::time::SystemTime>, WorldMeta)> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let dir = e.path();
            let bytes = fs::read(dir.join("world.meta")).ok()?;
            let (name, seed) = decode_meta(&bytes)?;
            let played = fs::metadata(run_path(&dir)).and_then(|m| m.modified()).ok();
            Some((played, WorldMeta { name, seed, dir }))
        })
        .collect();
    // Newest first; never played sorts last. The directory name breaks ties, so
    // the order is stable rather than whatever the filesystem enumerated.
    found.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.dir.file_name().cmp(&b.1.dir.file_name()))
    });
    found.into_iter().map(|(_, w)| w).collect()
}

/// Delete a world, and refuse anything that is not one.
///
/// Two guards, and both matter because this is the one irreversible thing in the
/// module. The directory must sit directly under `root`, so a `..` that survived
/// [`slug_of`] cannot walk out of the saves folder; and it must contain a
/// readable `world.meta`, so a `root` pointed at the wrong place deletes nothing
/// rather than everything in it.
pub fn delete_world(root: impl AsRef<Path>, world: &WorldMeta) -> io::Result<()> {
    let refuse = |why: &str| Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    if world.dir.parent() != Some(root.as_ref()) {
        return refuse("that directory is not in the saves folder");
    }
    if fs::read(world.dir.join("world.meta"))
        .ok()
        .and_then(|b| decode_meta(&b))
        .is_none()
    {
        return refuse("that directory is not a world");
    }
    fs::remove_dir_all(&world.dir)
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
        let dir = std::env::temp_dir().join(format!("yugen-save-test-{name}"));
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

    fn a_run() -> RunState {
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

    #[test]
    fn a_name_becomes_a_directory_that_cannot_escape_the_saves_folder() {
        assert_eq!(slug_of("Home"), "home");
        assert_eq!(slug_of("The Deep Below"), "the-deep-below");
        assert_eq!(slug_of("  spaced  out  "), "spaced-out");
        // The ones that matter: a name is text a player typed, and joining it
        // raw would write wherever they pointed it.
        assert_eq!(slug_of("../../etc/passwd"), "etc-passwd");
        assert_eq!(slug_of("/absolute"), "absolute");
        assert_eq!(slug_of("..").as_str(), "world");
        assert_eq!(slug_of("").as_str(), "world");
        assert_eq!(slug_of("💀💀💀").as_str(), "world");
        assert!(slug_of(&"x".repeat(200)).len() <= WORLD_NAME_MAX);
    }

    #[test]
    fn two_worlds_with_the_same_name_are_two_worlds() {
        let root = scratch("samename");
        let a = create_world(&root, "Home", 1).expect("first");
        let b = create_world(&root, "Home", 2).expect("second");
        assert_ne!(a.dir, b.dir, "the second must not land on the first");
        assert_eq!(a.name, b.name, "the DISPLAY name is allowed to collide");
        let listed = list_worlds(&root);
        assert_eq!(listed.len(), 2);
        assert_eq!(
            listed
                .iter()
                .map(|w| w.seed)
                .collect::<std::collections::BTreeSet<_>>(),
            [1, 2].into_iter().collect()
        );
    }

    /// The seed is recorded at creation, not inferred from a run file — a world
    /// that has never been played has no run file at all.
    #[test]
    fn a_world_knows_its_seed_before_it_has_ever_been_played() {
        let root = scratch("unplayed");
        let made = create_world(&root, "Fresh", 8675309).expect("create");
        assert!(!run_path(&made.dir).exists(), "nothing has been saved yet");
        assert_eq!(list_worlds(&root), vec![made]);
    }

    #[test]
    fn a_stray_folder_in_the_saves_root_is_skipped_rather_than_listed() {
        let root = scratch("stray");
        let real = create_world(&root, "Real", 1).expect("create");
        fs::create_dir_all(root.join("holiday-photos")).expect("mkdir");
        fs::write(root.join("notes.txt"), b"hello").expect("write");
        assert_eq!(list_worlds(&root), vec![real]);
    }

    #[test]
    fn deleting_refuses_anything_that_is_not_a_world_in_this_root() {
        let root = scratch("delete");
        let world = create_world(&root, "Doomed", 1).expect("create");

        // A directory outside the root, even if it looks like a world.
        let outside = scratch("delete-outside");
        let elsewhere = create_world(&outside, "Elsewhere", 1).expect("create");
        assert!(delete_world(&root, &elsewhere).is_err(), "escaped the root");
        assert!(elsewhere.dir.exists(), "and was not touched");

        // A directory under the root that is not a world.
        let notaworld = root.join("holiday-photos");
        fs::create_dir_all(&notaworld).expect("mkdir");
        let fake = WorldMeta {
            name: "nope".into(),
            seed: 0,
            dir: notaworld.clone(),
        };
        assert!(delete_world(&root, &fake).is_err(), "not a world");
        assert!(notaworld.exists(), "and was not touched");

        // The real thing.
        assert!(delete_world(&root, &world).is_ok());
        assert!(!world.dir.exists());
        assert_eq!(list_worlds(&root), vec![]);
    }

    #[test]
    fn the_most_recently_played_world_is_listed_first() {
        let root = scratch("recency");
        let old = create_world(&root, "Old", 1).expect("a");
        let new = create_world(&root, "New", 2).expect("b");
        write_run(&old.dir, &RunState { seed: 1, ..a_run() }).expect("play old");
        // A run file makes a world newer than one with none, whatever the
        // directory order happened to be.
        std::thread::sleep(std::time::Duration::from_millis(10));
        write_run(&new.dir, &RunState { seed: 2, ..a_run() }).expect("play new");
        assert_eq!(
            list_worlds(&root)
                .iter()
                .map(|w| w.name.as_str())
                .collect::<Vec<_>>(),
            vec!["New", "Old"]
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

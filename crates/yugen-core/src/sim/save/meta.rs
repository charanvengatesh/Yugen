//! A world's identity, and the directory it lives in: `GGWD`.
//!
//! The smallest of the three formats and the one with the most rules around it,
//! because this is where a display name typed by a player becomes a path on
//! disk. [`slug_of`] is a security boundary before it is a tidiness one, and
//! [`delete_world`] is the only irreversible thing in the module.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::bytes::Reader;
use super::run::run_path;

// --- Worlds ------------------------------------------------------------------

/// File signature for a world's identity file. `GGWD` — Yūgen WorlD.
const META_MAGIC: [u8; 4] = *b"GGWD";

/// Identity-file version. Independent of the chunk and run versions, for the
/// reason the run file's own version gives.
///
/// **2** adds the stamps: when the world was made, when it was last played, how
/// long it has been played for, and the flags byte that carries permadeath. A
/// version-1 file is migrated by [`super::legacy::decode_meta_v1`], which
/// supplies what version 1 did not record — see there for what each absence is
/// taken to mean.
const META_VERSION: u16 = 2;

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
    /// Unix seconds the world was created, or 0 if it predates the stamp.
    pub created_at: u64,
    /// Unix seconds it was last played.
    ///
    /// Recorded IN THE FILE rather than read off `run.save`'s mtime, which is
    /// what version 1 did. An mtime is destroyed by a backup, a `cp -r`, a
    /// restore or a checkout, and the world list orders by this — so the first
    /// time a player copies their saves to a new machine, version 1 shuffles
    /// the list into whatever order the copy happened to touch the files in.
    pub last_played: u64,
    /// Seconds of play, accumulated across sessions.
    pub play_seconds: u64,
    /// True if this world was created with permadeath on.
    ///
    /// A world property and deliberately not a setting. It changes what the
    /// files mean, and a global toggle would let a player turn it off after
    /// dying — which is the one moment it exists to matter at. Chosen once, at
    /// creation.
    pub permadeath: bool,
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

/// Fields a version-2 identity file carries, in order.
///
/// Flat rather than sectioned, unlike the run file. A run gains a field every
/// time the game gains a mechanic; this gains one about as often as the concept
/// of "a world" changes, which is to say almost never. Sections would be
/// machinery bought against a cost nobody is paying.
pub(super) struct MetaStamps {
    pub created_at: u64,
    pub last_played: u64,
    pub play_seconds: u64,
    pub permadeath: bool,
}

fn encode_meta(name: &str, seed: u32, s: &MetaStamps) -> Vec<u8> {
    let name = &name[..name.len().min(WORLD_NAME_MAX)];
    let mut out = Vec::with_capacity(40 + name.len());
    out.extend_from_slice(&META_MAGIC);
    out.extend_from_slice(&META_VERSION.to_le_bytes());
    out.extend_from_slice(&seed.to_le_bytes());
    out.extend_from_slice(&s.created_at.to_le_bytes());
    out.extend_from_slice(&s.last_played.to_le_bytes());
    out.extend_from_slice(&s.play_seconds.to_le_bytes());
    out.push(u8::from(s.permadeath));
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out
}

/// Now, as unix seconds. Zero if the clock is before the epoch, which is not a
/// state worth a `Result` — an unstamped world sorts last and that is all.
pub(super) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(super) fn decode_meta(bytes: &[u8]) -> Option<(String, u32, MetaStamps)> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(4)? != META_MAGIC {
        return None;
    }
    match r.u16()? {
        META_VERSION => {}
        1 => return super::legacy::decode_meta_v1(bytes),
        _ => return None,
    }
    let seed = r.u32()?;
    let stamps = MetaStamps {
        created_at: r.u64()?,
        last_played: r.u64()?,
        play_seconds: r.u64()?,
        permadeath: r.u8()? != 0,
    };
    let n = r.u16()? as usize;
    if n > WORLD_NAME_MAX {
        return None;
    }
    let name = std::str::from_utf8(r.take(n)?).ok()?.to_string();
    (r.at == bytes.len()).then_some((name, seed, stamps))
}

/// Create a world under `root`, in its own directory.
///
/// The directory is [`slug_of`] the name, with `-2`, `-3` and so on appended
/// until it is free — so two worlds called "Home" are two worlds rather than one
/// silently overwriting the other, which is the single worst thing a save menu
/// can do.
pub fn create_world(
    root: impl AsRef<Path>,
    name: &str,
    seed: u32,
    permadeath: bool,
) -> io::Result<WorldMeta> {
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
    let now = now_secs();
    let stamps = MetaStamps {
        created_at: now,
        last_played: now,
        play_seconds: 0,
        permadeath,
    };
    fs::write(dir.join("world.meta"), encode_meta(&name, seed, &stamps))?;
    Ok(WorldMeta {
        name,
        seed,
        dir,
        created_at: stamps.created_at,
        last_played: stamps.last_played,
        play_seconds: stamps.play_seconds,
        permadeath: stamps.permadeath,
    })
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
    let mut found: Vec<(u64, WorldMeta)> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let dir = e.path();
            let bytes = fs::read(dir.join("world.meta")).ok()?;
            let (name, seed, mut s) = decode_meta(&bytes)?;
            // A world that predates the stamp keeps the ordering it has always
            // had: version 1 sorted on `run.save`'s mtime, so an unmigrated
            // world still does rather than sinking to the bottom of the list.
            if s.last_played == 0 {
                s.last_played = fs::metadata(run_path(&dir))
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
            }
            Some((
                s.last_played,
                WorldMeta {
                    name,
                    seed,
                    dir,
                    created_at: s.created_at,
                    last_played: s.last_played,
                    play_seconds: s.play_seconds,
                    permadeath: s.permadeath,
                },
            ))
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

#[cfg(test)]
mod tests {
    use super::super::run::{RunState, write_run};
    use super::super::testing::{a_run, scratch};
    use super::*;

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
    fn a_version_one_world_keeps_the_order_it_has_always_had() {
        // Version 1 sorted the world list on `run.save`'s mtime. A migrated
        // world has no `last_played` stamp, so it must still sort on that
        // rather than reading as never-played and sinking to the bottom — which
        // is what a player would see as their worlds being shuffled by an
        // update.
        let root = scratch("metav1");
        let dir = root.join("old");
        std::fs::create_dir_all(&dir).expect("mkdir");

        let mut v1 = Vec::new();
        v1.extend_from_slice(b"GGWD");
        v1.extend_from_slice(&1u16.to_le_bytes());
        v1.extend_from_slice(&77u32.to_le_bytes());
        v1.extend_from_slice(&(4u16).to_le_bytes());
        v1.extend_from_slice(b"Home");
        std::fs::write(dir.join("world.meta"), &v1).expect("write v1");
        write_run(
            &dir,
            &RunState {
                seed: 77,
                ..a_run()
            },
        )
        .expect("play it");

        let found = list_worlds(&root);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "Home");
        assert_eq!(found[0].seed, 77);
        assert_eq!(found[0].created_at, 0, "unknown, not invented as now");
        assert_eq!(found[0].play_seconds, 0);
        assert!(
            !found[0].permadeath,
            "a world made before it cannot have it"
        );
        assert!(
            found[0].last_played > 0,
            "the mtime fallback did not fire, so the list order moved"
        );
    }

    #[test]
    fn a_new_world_records_when_it_was_made() {
        let root = scratch("metastamp");
        let w = create_world(&root, "Fresh", 3, true).expect("create");
        assert!(w.created_at > 0);
        assert_eq!(w.created_at, w.last_played);
        assert_eq!(w.play_seconds, 0);
        assert!(w.permadeath, "chosen at creation and nowhere else");
        assert_eq!(list_worlds(&root)[0], w, "and it survives a round trip");
    }

    #[test]
    fn two_worlds_with_the_same_name_are_two_worlds() {
        let root = scratch("samename");
        let a = create_world(&root, "Home", 1, false).expect("first");
        let b = create_world(&root, "Home", 2, false).expect("second");
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
        let made = create_world(&root, "Fresh", 8675309, false).expect("create");
        assert!(!run_path(&made.dir).exists(), "nothing has been saved yet");
        assert_eq!(list_worlds(&root), vec![made]);
    }

    #[test]
    fn a_stray_folder_in_the_saves_root_is_skipped_rather_than_listed() {
        let root = scratch("stray");
        let real = create_world(&root, "Real", 1, false).expect("create");
        fs::create_dir_all(root.join("holiday-photos")).expect("mkdir");
        fs::write(root.join("notes.txt"), b"hello").expect("write");
        assert_eq!(list_worlds(&root), vec![real]);
    }

    #[test]
    fn deleting_refuses_anything_that_is_not_a_world_in_this_root() {
        let root = scratch("delete");
        let world = create_world(&root, "Doomed", 1, false).expect("create");

        // A directory outside the root, even if it looks like a world.
        let outside = scratch("delete-outside");
        let elsewhere = create_world(&outside, "Elsewhere", 1, false).expect("create");
        assert!(delete_world(&root, &elsewhere).is_err(), "escaped the root");
        assert!(elsewhere.dir.exists(), "and was not touched");

        // A directory under the root that is not a world.
        let notaworld = root.join("holiday-photos");
        fs::create_dir_all(&notaworld).expect("mkdir");
        let fake = WorldMeta {
            created_at: 0,
            last_played: 0,
            play_seconds: 0,
            permadeath: false,
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
        let old = create_world(&root, "Old", 1, false).expect("a");
        let new = create_world(&root, "New", 2, false).expect("b");
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
}

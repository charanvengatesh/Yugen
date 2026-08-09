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
}

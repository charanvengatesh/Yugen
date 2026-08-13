//! The text a new record is born as.
//!
//! # The one place this tool writes prose
//!
//! Every other write in this crate puts lines where lines were: a redraw
//! replaces the lines a frame occupied and copies the paragraph above it
//! through untouched, because that paragraph is somebody's explanation and this
//! tool has no business rewriting it.
//!
//! An appended record has no paragraph yet. Nobody has written it, so composing
//! one is not overwriting anybody — it is the difference between editing prose
//! and authoring it, and only the first is forbidden. FORMAT.md §6 makes the
//! comments the most valuable thing in the file, which is an argument for
//! leaving a new record with something to say rather than with a bare header.
//!
//! What goes in it is deliberately small: what the record is, and the seed if a
//! generator made it. Not a description of the fields — the schema already
//! documents those, and restating them here would be a second copy to drift.
//!
//! # Terse by construction
//!
//! [`sound_record`] writes only the fields that differ from the schema's
//! defaults, so a new sound is born as short as a hand-written one. That is the
//! same rule [`crate::sound::Sound::changes`] enforces on every later save, and
//! having the two disagree would mean a record got longer the moment it was
//! created and shorter the first time it was edited.

use crate::procgen::sprite::Recipe;
use crate::sprite::{Frame, frames_body};
use crate::synth::Params;

/// A new sprite record, as the text a file should gain.
///
/// `seed` is written into the comment when a generator drew the frames. It is
/// the whole input to [`crate::procgen::sprite::generate`], so the line is not a
/// note about where the art came from — it is enough to produce it again.
pub fn sprite_record(
    id: &str,
    name: &str,
    cells: (u32, u32),
    pal: &[String],
    frames: &[Frame],
    seed: Option<u32>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {name}.\n"));
    if let Some(seed) = seed {
        out.push_str(&format!(
            "# Drawn by the generator at seed {seed:08x}; the same seed redraws it.\n"
        ));
    }
    out.push_str(&format!("[{id}]\n"));
    out.push_str(&format!("name = {name:?}\n"));
    out.push_str(&format!("cellsW = {}\n", cells.0));
    out.push_str(&format!("cellsH = {}\n", cells.1));
    let entries: Vec<String> = pal.iter().map(|c| format!("{c:?}")).collect();
    out.push_str(&format!("pal = [{}]\n", entries.join(", ")));
    out.push_str(&format!("[[{id}.seq]]\n"));
    out.push_str("frames = '''\n");
    for line in frames_body(frames) {
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str("'''\n");
    out
}

/// A new sound record, writing only what differs from the schema's defaults.
pub fn sound_record(id: &str, name: &str, p: &Params, note: Option<&str>) -> String {
    let d = Params::default();
    let mut out = String::new();
    out.push_str(&format!("# {name}.\n"));
    if let Some(note) = note {
        out.push_str(&format!("# {note}\n"));
    }
    out.push_str(&format!("[{id}]\n"));
    out.push_str(&format!("name = {name:?}\n"));

    // `wave`, `hz` and `seconds` are required by the schema, so they are written
    // whatever they are. Everything else earns its line by differing.
    out.push_str(&format!("wave = {:?}\n", p.wave.name()));
    out.push_str(&format!("hz = {}\n", float(p.hz)));
    if p.hz_to != p.hz {
        out.push_str(&format!("hzTo = {}\n", float(p.hz_to)));
    }
    out.push_str(&format!("seconds = {}\n", float(p.seconds)));
    for (key, v, def) in [
        ("attack", p.attack, d.attack),
        ("release", p.release, d.release),
        ("noise", p.noise, d.noise),
        ("gain", p.gain, d.gain),
    ] {
        if v != def {
            out.push_str(&format!("{key} = {}\n", float(v)));
        }
    }
    out
}

/// A float as TOML, always with a decimal point.
///
/// Restated from [`crate::sound`] rather than shared, for the reason its own
/// copy gives: `340` and `340.0` are an integer and a float to a TOML parser and
/// the schema says float. Two callers, four lines, and making it public would
/// put a formatting detail in the crate's surface.
fn float(v: f32) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Whether `id` is a legal record name.
///
/// FORMAT.md makes the table name the authoring id, and `contentc` derives a
/// SCREAMING const from it. Anything outside `[a-z0-9_]` either needs quoting in
/// TOML or produces a const name that does not compile, and both are failures
/// that happen later and read as something else.
pub fn is_legal_id(id: &str) -> bool {
    !id.is_empty()
        && id.starts_with(|c: char| c.is_ascii_lowercase())
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// The recipe's seed, if the frames came from a generator.
pub fn seed_of(recipe: Option<&Recipe>) -> Option<u32> {
    recipe.map(|r| r.seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sound;
    use crate::sprite;
    use crate::synth::Wave;

    fn frame(rows: &[&str]) -> Frame {
        rows.iter().map(|r| (*r).to_string()).collect()
    }

    #[test]
    fn a_new_sprite_record_reads_back_as_what_was_written() {
        // The round trip that matters: whatever this composes has to be
        // something `sprite::read` can open, or the editor creates records it
        // cannot then edit.
        let pal = vec![
            ".".to_string(),
            "#19242e".to_string(),
            "#e7b471".to_string(),
        ];
        let frames = vec![frame(&["1221", "2112", "1221", "2112"])];
        let text = sprite_record(
            "gulperling",
            "Gulperling",
            (4, 4),
            &pal,
            &frames,
            Some(0x3f2a),
        );

        text.parse::<toml::Table>().expect("valid TOML");
        let s = sprite::read(&text, "gulperling").expect("reads");
        assert_eq!(s.name, "Gulperling");
        assert_eq!((s.cells_w, s.cells_h), (4, 4));
        assert_eq!(s.pal, pal);
        assert_eq!(s.seqs.len(), 1);
        assert_eq!(s.seqs[0].frames, frames);
    }

    #[test]
    fn a_generated_sprite_carries_the_seed_that_drew_it() {
        // Not decoration: the seed is the whole input to the generator, so the
        // comment is enough to reproduce the art rather than merely to explain
        // where it came from.
        let pal = vec![".".to_string(), "#ffffff".to_string()];
        let frames = vec![frame(&["11", "11"])];
        let text = sprite_record("x", "X", (2, 2), &pal, &frames, Some(0x3f2a91c4));
        assert!(text.contains("3f2a91c4"), "{text}");

        let plain = sprite_record("x", "X", (2, 2), &pal, &frames, None);
        assert!(
            !plain.contains("seed"),
            "a hand-drawn record claims no seed"
        );
    }

    #[test]
    fn a_new_sound_record_reads_back_as_what_was_written() {
        let p = Params {
            wave: Wave::Saw,
            hz: 520.0,
            hz_to: 780.0,
            seconds: 0.07,
            attack: 0.05,
            release: 0.6,
            noise: 0.0,
            gain: 0.3,
            ..Params::default()
        };
        let text = sound_record("chime", "Chime", &p, None);
        text.parse::<toml::Table>().expect("valid TOML");
        let s = sound::read(&text, "chime").expect("reads");
        assert_eq!(s.name, "Chime");
        assert_eq!(s.params, p);
    }

    #[test]
    fn a_new_sound_writes_only_what_differs_from_the_defaults() {
        // A record born verbose and then edited terse would make the editor
        // look like it had two minds about the format. `Sound::changes` writes
        // only what moved on every later save; this is the same rule at birth.
        let text = sound_record("plain", "Plain", &Params::default(), None);
        assert!(!text.contains("attack"), "{text}");
        assert!(!text.contains("release"), "{text}");
        assert!(!text.contains("noise"), "{text}");
        assert!(!text.contains("gain"), "{text}");
        assert!(!text.contains("hzTo"), "an unswept sound writes no sweep");
        // The three the schema requires are always there.
        assert!(text.contains("wave = "));
        assert!(text.contains("hz = "));
        assert!(text.contains("seconds = "));
    }

    #[test]
    fn a_whole_number_is_still_written_as_a_float() {
        // `340` and `340.0` are different types to a TOML parser and the schema
        // says float, so a round number must not author an integer.
        let p = Params {
            hz: 340.0,
            seconds: 1.0,
            ..Params::default()
        };
        let text = sound_record("x", "X", &p, None);
        assert!(text.contains("hz = 340.0"), "{text}");
        assert!(text.contains("seconds = 1.0"), "{text}");
    }

    #[test]
    fn a_swept_sound_writes_its_sweep_and_an_unswept_one_does_not() {
        let swept = Params {
            hz: 300.0,
            hz_to: 900.0,
            ..Params::default()
        };
        assert!(sound_record("x", "X", &swept, None).contains("hzTo = 900.0"));

        let flat = Params {
            hz: 300.0,
            hz_to: 300.0,
            ..Params::default()
        };
        assert!(!sound_record("x", "X", &flat, None).contains("hzTo"));
    }

    #[test]
    fn an_id_has_to_be_something_contentc_can_name() {
        // `contentc` derives a SCREAMING const from the id. A capital, a dash or
        // a leading digit either needs quoting in TOML or produces a const name
        // that does not compile — both fail later and read as something else.
        for ok in ["a", "gulperling", "icon_block_cube", "mob2", "a_1"] {
            assert!(is_legal_id(ok), "{ok} should be legal");
        }
        for bad in ["", "Gulperling", "gulper-ling", "2mob", "gulper ling", "_x"] {
            assert!(!is_legal_id(bad), "{bad} should be refused");
        }
    }
}

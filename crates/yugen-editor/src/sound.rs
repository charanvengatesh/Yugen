//! Reading and writing a `sound` record: eight numbers and a waveform.
//!
//! # Why the sounds are the easy half and the delicate one
//!
//! A sound record is flat scalars — no `'''` bodies, no table arrays — so the
//! write side is [`crate::field::set_head_key`] eight times over and needs none
//! of the machinery the sprites needed. What it needs instead is restraint,
//! because of how these files are written:
//!
//! ```toml
//! [place]
//! name = "Place"
//! # A tone rather than noise, so building does not sound like digging in
//! # reverse. Placing is additive and reads better as a small definite click.
//! wave = "triangle"
//! hz = 340.0
//! ```
//!
//! Every record in `content/sounds/` carries a paragraph like that, and most
//! records write only the fields they mean: `[step]` never mentions `attack`,
//! because 0 is the default and an instant onset is what a footstep wants.
//!
//! Two rules follow, and [`Sound::changes`] is where both live:
//!
//! - **Only write what moved.** A save touches the keys whose values actually
//!   differ from what was read. Rewriting an untouched key would reformat its
//!   number — `1.6129032258064517` does not survive an `f32` round trip — and
//!   fill the diff with lines nobody changed.
//! - **A default stays unwritten until it is turned.** A record that omits
//!   `attack` keeps omitting it until somebody moves the slider, at which point
//!   the line is added. Materialising all eight fields on the first save would
//!   turn eight terse records into eight identical walls.

use std::collections::BTreeMap;

use crate::field::{FieldError, set_head_key};
use crate::splice::record_ids;
use crate::synth::{Params, Wave};

/// A sound record: what it is called, what it sounds like, and what it looked
/// like when it was read.
#[derive(Debug, Clone, PartialEq)]
pub struct Sound {
    pub id: String,
    pub name: String,
    /// The live parameters — what the sliders move and what the preview plays.
    pub params: Params,
    /// The parameters as read. The baseline [`Sound::changes`] diffs against, and
    /// the reason an untouched key is never rewritten.
    pub original: Params,
}

/// Everything that can stop a record being opened as a sound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoundError {
    Toml(String),
    NoSuchRecord(String),
    /// The record parses but is not a sound. How the browser filters — most of
    /// `content/` is blocks and items.
    NotASound(String),
    /// A required field is missing or the wrong type.
    Field {
        id: String,
        key: String,
    },
    /// `wave` names an oscillator that does not exist.
    Wave {
        id: String,
        wave: String,
    },
}

impl std::fmt::Display for SoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SoundError::Toml(e) => write!(f, "{e}"),
            SoundError::NoSuchRecord(id) => write!(f, "no record `[{id}]`"),
            SoundError::NotASound(id) => write!(f, "`[{id}]` is not a sound"),
            SoundError::Field { id, key } => write!(f, "`[{id}]`: `{key}` is missing or wrong"),
            SoundError::Wave { id, wave } => {
                write!(f, "`[{id}]`: no oscillator called `{wave}`")
            }
        }
    }
}

impl std::error::Error for SoundError {}

/// The schema's ranges, so the editor's sliders cannot author a value
/// `contentc` will reject.
///
/// Duplicated from `crates/contentc/src/schemas/sound.rs` rather than imported,
/// for the same reason the synthesiser is: this crate does not link the
/// compiler. The cost of being wrong is small and loud — a slider that stops
/// early, or a value the next `contentc` run refuses with a clear message — so
/// it does not want the machinery that a silent divergence would.
pub const HZ_RANGE: (f32, f32) = (20.0, 8000.0);
/// Total length. Short: this is feedback, not music.
pub const SECONDS_RANGE: (f32, f32) = (0.01, 2.0);

impl Sound {
    /// The keys whose values differ from what was read, as `key = value` lines.
    ///
    /// Ordered, so a save that moves three sliders inserts new keys in a stable
    /// order rather than whichever a hash map happened to yield.
    pub fn changes(&self) -> BTreeMap<&'static str, String> {
        let mut out = BTreeMap::new();
        let (p, o) = (&self.params, &self.original);

        if p.wave != o.wave {
            out.insert("wave", format!("wave = {:?}", p.wave.name()));
        }
        for (key, new, old) in [
            ("hz", p.hz, o.hz),
            ("hzTo", p.hz_to, o.hz_to),
            ("seconds", p.seconds, o.seconds),
            ("attack", p.attack, o.attack),
            ("release", p.release, o.release),
            ("noise", p.noise, o.noise),
            ("gain", p.gain, o.gain),
        ] {
            if new != old {
                out.insert(key, format!("{key} = {}", float(new)));
            }
        }
        out
    }

    /// True if anything would be written.
    pub fn dirty(&self) -> bool {
        !self.changes().is_empty()
    }

    /// Apply the changes to the file text, one splice per moved key.
    ///
    /// Sequential rather than batched because each splice may INSERT a line, and
    /// an insert moves every span below it — so each call re-finds its key in the
    /// text the previous one produced.
    pub fn write(&self, text: &str) -> Result<String, FieldError> {
        let mut out = text.to_string();
        for (key, line) in self.changes() {
            out = set_head_key(&out, &self.id, key, &line)?;
        }
        Ok(out)
    }
}

/// A float as TOML, always with a decimal point.
///
/// `340` and `340.0` are an integer and a float to a TOML parser, and the schema
/// declares these fields `float`. Rust's own `{}` prints `340` for a whole
/// number, which would author the wrong type and fail the next `contentc` run —
/// so the point is forced.
fn float(v: f32) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Every record in the file that is a sound, in file order.
pub fn sound_ids(text: &str) -> Vec<String> {
    record_ids(text)
        .into_iter()
        .filter(|id| !matches!(read(text, id), Err(SoundError::NotASound(_))))
        .collect()
}

/// Read one sound record, filling every unwritten field with its schema default.
pub fn read(text: &str, id: &str) -> Result<Sound, SoundError> {
    let table: toml::Table = text.parse().map_err(|e| SoundError::Toml(format!("{e}")))?;
    let record = table
        .get(id)
        .ok_or_else(|| SoundError::NoSuchRecord(id.to_string()))?;

    // `wave` is the probe: it is required on sounds and appears on no other kind,
    // so a record that has it is one and a record that does not is not.
    let Some(wave) = record.get("wave").and_then(|w| w.as_str()) else {
        return Err(SoundError::NotASound(id.to_string()));
    };
    let wave = Wave::parse(wave).ok_or_else(|| SoundError::Wave {
        id: id.to_string(),
        wave: wave.to_string(),
    })?;

    let num = |key: &str| record.get(key).and_then(|v| v.as_float()).map(|f| f as f32);
    let field = |key: &str| SoundError::Field {
        id: id.to_string(),
        key: key.to_string(),
    };

    let hz = num("hz").ok_or_else(|| field("hz"))?;
    let params = Params {
        wave,
        hz,
        // `hzTo` defaults to `hz` — "no sweep" — rather than to a constant. A
        // record that never mentions it means "stay where you started", and
        // defaulting it to anything else would invent a sweep nobody authored.
        hz_to: num("hzTo").unwrap_or(hz),
        seconds: num("seconds").ok_or_else(|| field("seconds"))?,
        attack: num("attack").unwrap_or(0.0),
        release: num("release").unwrap_or(0.5),
        noise: num("noise").unwrap_or(0.0),
        gain: num("gain").unwrap_or(0.7),
    };

    Ok(Sound {
        id: id.to_string(),
        name: record
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or(id)
            .to_string(),
        params,
        original: params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = "\
# What the body does.\n\
\n\
# Noise rather than a tone, because a footstep is a surface being disturbed.\n\
[step]\n\
name = \"Footstep\"\n\
wave = \"noise\"\n\
hz = 220.0\n\
hzTo = 160.0\n\
seconds = 0.05\n\
release = 0.8\n\
gain = 0.25\n\
\n\
# A tone rather than noise, so building does not sound like digging in reverse.\n\
[place]\n\
name = \"Place\"\n\
wave = \"triangle\"\n\
hz = 340.0\n\
seconds = 0.05\n\
";

    #[test]
    fn an_unwritten_field_reads_as_its_schema_default() {
        // `[step]` never mentions `attack` or `noise`, and what it MEANS by that
        // is 0 for both. A reader that left them undefined would preview a
        // different sound from the one the game plays.
        let s = read(FILE, "step").expect("reads");
        assert_eq!(s.params.wave, Wave::Noise);
        assert_eq!(s.params.attack, 0.0);
        assert_eq!(s.params.noise, 0.0);
        assert_eq!(s.params.release, 0.8, "and a written one is honoured");
        assert_eq!(s.name, "Footstep");
    }

    #[test]
    fn an_unwritten_sweep_means_no_sweep_and_not_a_constant() {
        // `[place]` omits `hzTo`, which means "stay at 340" — the schema defaults
        // it to `hz`. Defaulting it to anything fixed would author a sweep
        // nobody wrote.
        let s = read(FILE, "place").expect("reads");
        assert_eq!(s.params.hz, 340.0);
        assert_eq!(s.params.hz_to, 340.0);
    }

    #[test]
    fn nothing_is_written_until_something_moves() {
        // The rule that keeps these files terse: a save after opening a record
        // and touching nothing must produce no change at all.
        let mut s = read(FILE, "step").expect("reads");
        assert!(!s.dirty());
        assert_eq!(s.write(FILE).expect("writes"), FILE);

        s.params.gain = 0.4;
        assert!(s.dirty());
        assert_eq!(s.changes().len(), 1, "only the key that moved");
        assert_eq!(s.changes()["gain"], "gain = 0.4");
    }

    #[test]
    fn turning_a_default_up_adds_the_line_the_record_never_had() {
        // `[step]` has no `attack` line. Moving that slider has to ADD one, and a
        // tool that could only replace would do nothing at all here.
        let mut s = read(FILE, "step").expect("reads");
        s.params.attack = 0.25;
        let out = s.write(FILE).expect("writes");

        assert!(out.contains("attack = 0.25"));
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed["step"]["attack"].as_float(), Some(0.25));
        // The neighbour and every comment are untouched.
        assert!(out.contains("# Noise rather than a tone"));
        assert!(out.contains("# A tone rather than noise"));
        assert_eq!(parsed["place"]["hz"].as_float(), Some(340.0));
        assert_eq!(parsed["step"]["gain"].as_float(), Some(0.25));
    }

    #[test]
    fn a_whole_number_is_still_written_as_a_float() {
        // The schema declares these `float`, and `{}` on 300.0 prints `300` —
        // which is an integer to a TOML parser and a type error to `contentc`.
        let mut s = read(FILE, "step").expect("reads");
        s.params.hz = 300.0;
        assert_eq!(s.changes()["hz"], "hz = 300.0");
        let out = s.write(FILE).expect("writes");
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed["step"]["hz"].as_float(), Some(300.0));
    }

    #[test]
    fn a_record_that_is_not_a_sound_is_reported_as_such() {
        assert_eq!(sound_ids(FILE), vec!["step", "place"]);
        let blocks = "[stone]\nname = \"Stone\"\nhardness = 3.0\n";
        assert_eq!(
            read(blocks, "stone"),
            Err(SoundError::NotASound("stone".into()))
        );
        assert!(sound_ids(blocks).is_empty());
    }

    #[test]
    fn an_oscillator_that_does_not_exist_is_refused() {
        // The vocabulary is closed on purpose: each name is a branch of real DSP,
        // and `wave = "supersaw"` is a compile error until somebody writes it.
        let bad = FILE.replace("wave = \"noise\"", "wave = \"supersaw\"");
        assert_eq!(
            read(&bad, "step"),
            Err(SoundError::Wave {
                id: "step".into(),
                wave: "supersaw".into()
            })
        );
    }

    #[test]
    fn several_moved_keys_are_all_written() {
        let mut s = read(FILE, "step").expect("reads");
        s.params.wave = Wave::Sine;
        s.params.attack = 0.1;
        s.params.hz_to = 90.0;
        let out = s.write(FILE).expect("writes");

        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed["step"]["wave"].as_str(), Some("sine"));
        assert_eq!(parsed["step"]["attack"].as_float(), Some(0.1));
        assert_eq!(parsed["step"]["hzTo"].as_float(), Some(90.0));
        // `hzTo` was already written, so it moved in place rather than doubling.
        assert_eq!(out.matches("hzTo =").count(), 1);
        assert_eq!(out.matches("attack =").count(), 1);
    }
}

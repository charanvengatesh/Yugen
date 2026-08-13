//! Sound effects, as parameters rather than as recordings.
//!
//! # Why a synthesiser and not a folder of `.wav`
//!
//! Every other kind in `content/` is text that argues for itself: a block's
//! hardness sits next to the sentence saying why it is 3.0, and a diff shows the
//! number changing. A `.wav` is a binary blob that can only be diffed as "it is
//! different now", cannot be reviewed, cannot be tuned by hand, and cannot
//! explain itself in a comment. Putting one in this tree would make the audio
//! the only part of the game whose reasoning lives outside it.
//!
//! So a sound is a handful of numbers — a waveform, an envelope, a pitch sweep,
//! a noise mix — synthesised into PCM when the game loads. That is the sfxr
//! model, and it is the right one here for the same reason the sprites are
//! authored as digits: the whole game is small, deliberate values with an
//! argument attached.
//!
//! # What this kind is NOT
//!
//! Not music, and not streamed audio. Both want a file and a decoder, and both
//! are a different problem than "a footstep should tick". If either arrives it
//! arrives as its own kind with its own reasons, not by growing a `path` field
//! here.
//!
//! # The vocabulary is closed
//!
//! [`WAVE`] is a fixed set because each name is a branch of real DSP — the same
//! argument `block.texture` makes about drawing code. A content file may not
//! invent `wave = "supersaw"` and have it silently ignored; it is a compile
//! error until somebody writes the oscillator.

use crate::schema::{ArrayKind, Field, Schema, table};
use crate::value::Value;

/// Oscillators the synthesiser has. Closed — see the header.
///
/// `noise` is in the list rather than being only a mix amount because a sound
/// that is ENTIRELY noise is the common case (footsteps, digging, impacts), and
/// spelling it as `wave = "noise"` reads better than a square wave at 100% noise
/// mix.
const WAVE: &[(&str, i64)] = &[
    ("square", 0),
    ("saw", 1),
    ("sine", 2),
    ("triangle", 3),
    ("noise", 4),
];

fn enum_of(pairs: &[(&str, i64)]) -> String {
    format!(
        "enum({})",
        pairs.iter().map(|(k, _)| *k).collect::<Vec<_>>().join("|")
    )
}

pub fn schema() -> Schema {
    Schema {
        kind: "sound".into(),
        prefix: "SOUND".into(),
        iface: "SoundDef".into(),
        iface_prefix: "Sound".into(),

        fields: vec![
            (
                "name".into(),
                Field::new("string")
                    .doc("Display name — the settings screen and the debug HUD.")
                    .default_fn(|d| Some(Value::Str(d.str_of("id").to_string()))),
            ),
            (
                "wave".into(),
                Field::new(&enum_of(WAVE))
                    .doc("Which oscillator. Adding one needs code.")
                    // Named, so the synthesiser gets the enum and its `from_code`
                    // back out of the flat table — see SCHEMA-AUTHORING.md.
                    .alias("SoundWave")
                    .map(WAVE)
                    .required()
                    .hot(vec![table(
                        "SND_WAVE",
                        ArrayKind::U8,
                        0.0,
                        "Oscillator code.",
                        |d, _ctx| Some(d.num("wave")),
                    )]),
            ),
            (
                "hz".into(),
                Field::new("float")
                    .doc("Starting pitch in hertz.")
                    .required()
                    .between(20.0, 8000.0)
                    .hot(vec![table(
                        "SND_HZ",
                        ArrayKind::F32,
                        0.0,
                        "Starting pitch, Hz.",
                        |d, _ctx| Some(d.num("hz")),
                    )]),
            ),
            (
                "hzTo".into(),
                Field::new("float")
                    .doc(
                        "Pitch at the end of the sound. Equal to `hz` means no sweep; below it \
                         falls, above it rises.",
                    )
                    .default_fn(|d| Some(Value::Float(d.num("hz"))))
                    .between(20.0, 8000.0)
                    .hot(vec![table(
                        "SND_HZ_TO",
                        ArrayKind::F32,
                        0.0,
                        "Ending pitch, Hz.",
                        |d, _ctx| Some(d.num("hzTo")),
                    )]),
            ),
            (
                "seconds".into(),
                Field::new("float")
                    .doc("Total length. Short: this is feedback, not music.")
                    .required()
                    .between(0.01, 2.0)
                    .hot(vec![table(
                        "SND_SECONDS",
                        ArrayKind::F32,
                        0.0,
                        "Length in seconds.",
                        |d, _ctx| Some(d.num("seconds")),
                    )]),
            ),
            (
                "attack".into(),
                Field::new("float")
                    .doc(
                        "Fraction of the length spent rising to full volume. 0 is an instant \
                         onset, which is what an impact wants.",
                    )
                    .default_float(0.0)
                    .between(0.0, 1.0)
                    .hot(vec![table(
                        "SND_ATTACK",
                        ArrayKind::F32,
                        0.0,
                        "Attack, as a fraction of the length.",
                        |d, _ctx| Some(d.num("attack")),
                    )]),
            ),
            (
                "release".into(),
                Field::new("float")
                    .doc(
                        "Fraction of the length spent falling to silence. A sound with neither \
                         attack nor release clicks at both ends.",
                    )
                    .default_float(0.5)
                    .between(0.0, 1.0)
                    .hot(vec![table(
                        "SND_RELEASE",
                        ArrayKind::F32,
                        0.0,
                        "Release, as a fraction of the length.",
                        |d, _ctx| Some(d.num("release")),
                    )]),
            ),
            (
                "noise".into(),
                Field::new("chance")
                    .doc(
                        "How much white noise is mixed over the oscillator. Grit — a footstep on \
                         gravel against one on stone.",
                    )
                    .default_float(0.0)
                    .hot(vec![table(
                        "SND_NOISE",
                        ArrayKind::F32,
                        0.0,
                        "Noise mix, 0..1.",
                        |d, _ctx| Some(d.num("noise")),
                    )]),
            ),
            (
                "gain".into(),
                Field::new("chance")
                    .doc("Per-sound volume, before the player's own setting.")
                    .default_float(0.7)
                    .hot(vec![table(
                        "SND_GAIN",
                        ArrayKind::F32,
                        0.0,
                        "Per-sound gain, 0..1.",
                        |d, _ctx| Some(d.num("gain")),
                    )]),
            ),
            // ---- SHAPING -------------------------------------------------
            //
            // Four fields appended, never interleaved: field order drives
            // default-callback visibility and emission order, so inserting one
            // above `gain` would renumber nothing but would reorder the emitted
            // struct for no reason.
            //
            // EVERY ONE DEFAULTS TO ZERO AND ZERO MEANS OFF. That is not a
            // stylistic choice, it is the acceptance criterion for the whole
            // change: fourteen sounds exist and none of them author any of
            // these, so if a default here were audible, all fourteen would have
            // moved on the day this landed. `tests/pin/sound_synth.hex` is what
            // says they did not.
            //
            // The zeros sit OUTSIDE the ranges below, which is safe and
            // deliberate: `check` runs only on a value somebody actually wrote,
            // so an omitted key takes its default unvalidated. `hzTo` already
            // relies on the same asymmetry. The reading is "0 is off, and if you
            // write a number, write a real one".
            (
                "vibrato".into(),
                Field::new("chance")
                    .doc(
                        "Pitch wobble depth, as a fraction of the current pitch. 0 is none. \
                         A little is life in a held tone; a lot is a siren.",
                    )
                    .default_float(0.0)
                    .hot(vec![table(
                        "SND_VIBRATO",
                        ArrayKind::F32,
                        0.0,
                        "Vibrato depth, 0..1.",
                        |d, _ctx| Some(d.num("vibrato")),
                    )]),
            ),
            (
                "vibratoHz".into(),
                Field::new("float")
                    .doc(
                        "How fast the wobble is, in cycles per second. 0 with a non-zero \
                         depth is still no vibrato — both have to be set for either to do \
                         anything, which is why neither is required.",
                    )
                    .default_float(0.0)
                    .check(|v| {
                        let n = v.as_num().unwrap_or(0.0);
                        if (0.5..=40.0).contains(&n) {
                            None
                        } else {
                            Some("must be 0.5..40 Hz".to_string())
                        }
                    })
                    .hot(vec![table(
                        "SND_VIBRATO_HZ",
                        ArrayKind::F32,
                        0.0,
                        "Vibrato rate in Hz, 0 for none.",
                        |d, _ctx| Some(d.num("vibratoHz")),
                    )]),
            ),
            (
                "repeatHz".into(),
                Field::new("float")
                    .doc(
                        "How often the envelope and the pitch sweep restart, in cycles per \
                         second. 0 plays once. This is what turns one sweep into a stutter \
                         without authoring a second sound.",
                    )
                    .default_float(0.0)
                    .check(|v| {
                        let n = v.as_num().unwrap_or(0.0);
                        if (0.5..=60.0).contains(&n) {
                            None
                        } else {
                            Some("must be 0.5..60 Hz".to_string())
                        }
                    })
                    .hot(vec![table(
                        "SND_REPEAT_HZ",
                        ArrayKind::F32,
                        0.0,
                        "Restart rate in Hz, 0 to play once.",
                        |d, _ctx| Some(d.num("repeatHz")),
                    )]),
            ),
            (
                "lowpass".into(),
                Field::new("float")
                    .doc(
                        "One-pole low-pass cutoff in Hz. 0 is BYPASS — not a cutoff of \
                         zero, which would be silence. Takes the edge off a saw or a noise \
                         burst without dropping its gain.",
                    )
                    .default_float(0.0)
                    .check(|v| {
                        let n = v.as_num().unwrap_or(0.0);
                        if (20.0..=20000.0).contains(&n) {
                            None
                        } else {
                            Some("must be 20..20000 Hz".to_string())
                        }
                    })
                    .hot(vec![table(
                        "SND_LOWPASS",
                        ArrayKind::F32,
                        0.0,
                        "Low-pass cutoff in Hz, 0 to bypass.",
                        |d, _ctx| Some(d.num("lowpass")),
                    )]),
            ),
        ],

        tables: vec![],
        matrices: vec![],
        constants: vec![],

        // Enough to satisfy the required fields and nothing more. A tombstoned
        // sound should never be reached; if one is, it is a click at 440 Hz for
        // a hundredth of a second, which is audible as "something is wrong"
        // rather than silently absent.
        tombstone: vec![
            ("wave".into(), "\"sine\"".into()),
            ("hz".into(), "440.0".into()),
            ("seconds".into(), "0.01".into()),
        ],
    }
}

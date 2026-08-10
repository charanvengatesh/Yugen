//! Held to `yugen-render`'s answers, via `tests/pin/`.
//!
//! This crate carries its own rasteriser and its own synthesiser, and
//! `tests/pin/README.md` is the argument for why that is not fixable: linking
//! `yugen-render` means linking Bevy, and it would still be the wrong bytes,
//! because the renderer works from COMPILED content and an editor has to show a
//! file as it is being typed. An unsaved sound does not even have a code to look
//! up.
//!
//! What the duplication does not get to be is a second OPINION. The two crates
//! cannot call each other, so they meet at a golden file: `yugen-render` writes
//! it from its implementation, and this reads it.
//!
//! **Read-only, deliberately.** There is no bless switch here. A copy that could
//! rewrite the golden would only ever prove that it agreed with itself, which is
//! exactly the thing being guarded against. If one of these fails, the fix is in
//! this crate — unless the renderer moved on purpose, in which case that side is
//! blessed and this side is ported in the same commit.
//!
//! The cases below are declared to match the ones in
//! `yugen-render/src/sprite/baked.rs` and `yugen-render/src/sound/synth.rs`. They
//! are stated twice rather than parsed from a fixture because a fixture format
//! would need a parser in each crate, and those could drift in their own right.
//! If the cases ever diverge, both sides stop matching the golden and say so.

use std::path::PathBuf;

use yugen_editor::raster::raster;
use yugen_editor::synth::{self, Params, Wave};

/// The golden, as bytes. Whitespace in the file is the wrapping and is ignored.
fn golden(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/pin")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let digits: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    digits
        .chunks_exact(2)
        .map(|pair| {
            let s = std::str::from_utf8(pair).expect("hex is ASCII");
            u8::from_str_radix(s, 16).expect("the golden is hex")
        })
        .collect()
}

/// Where two byte strings first differ, for a message worth reading.
fn first_difference(want: &[u8], got: &[u8]) -> Option<usize> {
    want.iter()
        .zip(got)
        .position(|(a, b)| a != b)
        .or_else(|| (want.len() != got.len()).then_some(want.len().min(got.len())))
}

fn check(name: &str, got: &[u8]) {
    let want = golden(name);
    if let Some(at) = first_difference(&want, got) {
        panic!(
            "this crate's copy has drifted from yugen-render's: {name} differs at byte {at} \
             ({} bytes wanted, {} got).\n  \
             yugen-render is the authority — fix it HERE, or if the renderer changed on \
             purpose, bless there and port here in the same commit.",
            want.len(),
            got.len(),
        );
    }
}

// ---------------------------------------------------------------------------
// The rasteriser
// ---------------------------------------------------------------------------

/// Must match `yugen-render/src/sprite/baked.rs`'s `PIN_FRAME`.
const PIN_FRAME: [&str; 4] = [".01.", "1230", "3.21", ".11."];
/// Must match that file's `PIN_PAL`.
const PIN_PAL: [[u8; 3]; 4] = [
    [0, 0, 0],
    [0x1b, 0x12, 0x20],
    [0x3e, 0xc8, 0xb4],
    [0xff, 0xcf, 0x5c],
];

#[test]
fn the_raster_agrees_with_the_one_that_draws_the_game() {
    let frame: Vec<String> = PIN_FRAME.iter().map(|s| s.to_string()).collect();
    let got = raster(&frame, &PIN_PAL, 4, 4).expect("the pin case rasterises");
    assert_eq!(got.len(), 4 * 4 * 4);
    check("sprite_raster.hex", &got);
}

// ---------------------------------------------------------------------------
// The synthesiser
// ---------------------------------------------------------------------------

/// Must match `yugen-render/src/sound/synth.rs`'s `pin_case`.
fn pin_case() -> Params {
    Params {
        wave: Wave::Triangle,
        hz: 300.0,
        hz_to: 900.0,
        seconds: 0.02,
        attack: 0.3,
        release: 0.4,
        noise: 0.25,
        gain: 0.8,
    }
}

#[test]
fn the_synth_agrees_with_the_one_that_plays_the_game() {
    let pcm = synth::render(&pin_case());
    assert_eq!(pcm.len(), 882, "0.02 s at 44100");
    let got: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    check("sound_synth.hex", &got);
}

#[test]
fn the_sample_rate_is_the_one_the_game_renders_at() {
    // Not covered by the goldens above: a crate that rendered the same shape at
    // a different rate would produce a different NUMBER of samples, but the pin
    // asserts the length itself, so this states the constant outright.
    assert_eq!(synth::SAMPLE_RATE, 44_100);
}

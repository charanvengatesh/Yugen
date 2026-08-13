//! The extracted style study, held to the same rules as real content.
//!
//! `reference/` is a second content root — see its own header for why it is not
//! under `content/`. It is not compiled by `contentc` and nothing in the game
//! draws it, so without this file nothing would notice if it drifted out of the
//! format. It opens in the editor, and "opens in the editor" is a claim worth
//! checking: the whole point of keeping it in this format is that it can be read,
//! measured and copied from with the same tools.

use std::path::{Path, PathBuf};

use yugen_editor::procgen::ops;
use yugen_editor::raster::raster;
use yugen_editor::sprite::{art_ids, read};

fn sheet() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/yugen-editor always has two ancestors")
        .join("reference/sprites/chars.toml")
}

#[test]
fn every_extracted_character_reads_draws_and_is_eight_by_eight() {
    let text = std::fs::read_to_string(sheet()).expect("the study is on disk");
    let ids = art_ids(&text);
    assert_eq!(ids.len(), 54, "the sheet holds 54 characters");

    for id in ids {
        let sprite = read(&text, &id).unwrap_or_else(|e| panic!("[{id}]: {e}"));
        assert_eq!(
            (sprite.cells_w, sprite.cells_h),
            (8, 8),
            "[{id}] is not on the grid"
        );
        // Nine colours plus the transparent slot is what a one-digit frame
        // character can name. Five of these were quantised down to it.
        assert!(
            sprite.pal.len() <= 10,
            "[{id}] has {} palette entries",
            sprite.pal.len()
        );
        assert_eq!(sprite.pal[0], ".", "[{id}] slot 0 is not the placeholder");

        let colours = sprite
            .colours()
            .unwrap_or_else(|e| panic!("[{id}] palette: {e}"));
        for seq in &sprite.seqs {
            for frame in &seq.frames {
                raster(frame, &colours, sprite.texel_w(), sprite.texel_h())
                    .unwrap_or_else(|e| panic!("[{id}]: {e}"));
            }
        }
    }
}

#[test]
fn every_character_carries_a_derived_idle_and_run() {
    // The animation is not drawn — it is a function of the still frame, and that
    // is the claim worth keeping honest. If `anim::bob` or `anim::walk` were
    // retuned, regenerating this file would be the only way to keep it true, and
    // this is what says so.
    let text = std::fs::read_to_string(sheet()).expect("readable");
    for id in art_ids(&text) {
        let sprite = read(&text, &id).expect("reads");
        let states: Vec<&str> = sprite.seqs.iter().map(|s| s.state.as_str()).collect();
        assert_eq!(states, vec!["idle", "run"], "[{id}] poses");

        let idle = &sprite.seqs[0];
        let run = &sprite.seqs[1];
        assert_eq!(idle.frames.len(), 2, "[{id}] idle is a two-beat bob");
        assert_eq!(run.frames.len(), 4, "[{id}] run is a four-beat cycle");

        // Both cycles rest on the drawing they came from, so a paused sequence
        // looks like the art rather than like a half-step.
        assert_eq!(idle.frames[0], run.frames[0], "[{id}] the poses disagree");
        assert_eq!(run.frames[0], run.frames[2], "[{id}] run does not rest");
    }
}

#[test]
fn a_run_cycle_actually_moves_something() {
    // A four-frame sequence of four identical frames would satisfy every count
    // above and animate nothing. At least one beat has to differ from the rest
    // pose, or the walk generator silently did nothing.
    let text = std::fs::read_to_string(sheet()).expect("readable");
    let mut stepped = 0usize;
    for id in art_ids(&text) {
        let sprite = read(&text, &id).expect("reads");
        let run = &sprite.seqs[1];
        assert!(
            run.frames[1] != run.frames[0] || run.frames[3] != run.frames[0],
            "[{id}] run is four copies of the same drawing"
        );
        // Two separate foot groups give a real alternating step; one solid base
        // gives a hop, where the odd beat has nothing left to lift.
        if run.frames[3] != run.frames[0] {
            stepped += 1;
        }
    }
    assert!(
        stepped >= 25,
        "only {stepped} of 54 alternate feet; the sheet has 29 two-group bottoms"
    );
}

#[test]
fn every_extracted_character_stands_on_the_bottom_row() {
    // The extraction padded feet-anchored, which is the anchor both render hosts
    // use. A character floating above the bottom row would be one that had been
    // centred instead — and would sink into the ground if it were ever promoted
    // into `content/`.
    let text = std::fs::read_to_string(sheet()).expect("readable");
    for id in art_ids(&text) {
        let sprite = read(&text, &id).expect("reads");
        let frame = &sprite.seqs[0].frames[0];
        let bottom = frame.last().expect("eight rows");
        assert!(
            bottom.chars().any(|c| !ops::is_clear(c)),
            "[{id}] does not touch the bottom row"
        );
    }
}

#[test]
fn the_palettes_run_dark_to_light() {
    // Index 1 is the ink and the high indices are the highlights, which is how
    // the rest of the tree is authored — so a frame copied out of here into a
    // real record keeps meaning roughly the same thing.
    let text = std::fs::read_to_string(sheet()).expect("readable");
    for id in art_ids(&text) {
        let sprite = read(&text, &id).expect("reads");
        let colours = sprite.colours().expect("colours");
        let lum = |c: [u8; 3]| {
            0.2126 * f32::from(c[0]) + 0.7152 * f32::from(c[1]) + 0.0722 * f32::from(c[2])
        };
        let mut last = -1.0;
        for (i, c) in colours.iter().enumerate().skip(1) {
            let l = lum(*c);
            assert!(
                l >= last,
                "[{id}] index {i} is darker than the one before it"
            );
            last = l;
        }
    }
}

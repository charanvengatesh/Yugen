//! The splice, against every record actually in `content/`.
//!
//! The unit tests in `splice.rs` run on a fixture I wrote, which means they
//! prove the code does what I expected the files to look like. This proves it
//! against what they ARE — 300-odd records across seven kinds, including the
//! ones with `'''` bodies full of `#` and trailing spaces, the ones with table
//! arrays, and the ones whose comment paragraphs are longer than the record.
//!
//! The property is the strongest one available and needs no fixtures: replacing
//! a record with ITSELF must produce a byte-identical file. Anything the splice
//! gets wrong about boundaries, comments, blank lines or trailing newlines shows
//! up as a diff.

use std::path::{Path, PathBuf};

use yugen_editor::splice::{record_ids, record_lines, replace_record};

fn content_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/yugen-editor always has two ancestors")
        .join("content")
}

/// Every authored `*.toml` under `content/`, excluding the manifest.
fn content_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let root = content_dir();
    for kind in std::fs::read_dir(&root).expect("content/ exists").flatten() {
        if !kind.path().is_dir() {
            continue;
        }
        for f in std::fs::read_dir(kind.path())
            .expect("a kind dir")
            .flatten()
        {
            let p = f.path();
            if p.extension().is_some_and(|e| e == "toml") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn replacing_every_real_record_with_itself_changes_nothing() {
    let files = content_files();
    assert!(files.len() > 20, "found only {} content files", files.len());

    let mut records = 0usize;
    for path in &files {
        let text = std::fs::read_to_string(path).expect("readable");
        let ids = record_ids(&text);
        assert!(
            !ids.is_empty(),
            "{} has no records — the layout gate should have caught that",
            path.display()
        );

        for id in ids {
            let span = record_lines(&text, &id).expect("the id was just found");
            let lines: Vec<&str> = text.lines().collect();
            let same = lines[span.clone()].join("\n");

            let out = replace_record(&text, &id, &same)
                .unwrap_or_else(|e| panic!("{}: [{id}]: {e}", path.display()));
            assert_eq!(
                out,
                text,
                "{}: splicing [{id}] over itself changed the file",
                path.display()
            );
            records += 1;
        }
    }
    assert!(records > 250, "only walked {records} records");
}

#[test]
fn every_real_record_still_parses_after_a_round_trip() {
    // The splice is textual, so "identical" above is the real proof. This is the
    // weaker second opinion: whatever comes out is still the TOML that went in,
    // key for key, which is what `contentc` will read next.
    for path in content_files() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let before: toml::Table = text
            .parse()
            .unwrap_or_else(|e| panic!("{} is not TOML: {e}", path.display()));

        let Some(id) = record_ids(&text).into_iter().next() else {
            continue;
        };
        let span = record_lines(&text, &id).expect("found");
        let lines: Vec<&str> = text.lines().collect();
        let out = replace_record(&text, &id, &lines[span].join("\n")).expect("splices");

        let after: toml::Table = out.parse().expect("still TOML");
        assert_eq!(before, after, "{} changed meaning", path.display());
    }
}

#[test]
fn a_sprite_records_art_survives_being_spliced_over() {
    // `content/sprites/player.toml` is the hard case in the tree: one record,
    // twelve `[[player.seq]]` table arrays, and `'''` bodies where `#` is a
    // palette index and trailing spaces are part of the picture. A scanner that
    // treated `[[` as a record boundary would cut it apart, and a serialiser
    // would quietly reformat the art.
    let path = content_dir().join("sprites/player.toml");
    let text = std::fs::read_to_string(&path).expect("readable");
    assert_eq!(record_ids(&text), vec!["player"]);

    let span = record_lines(&text, "player").expect("found");
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        span.end - span.start > 400,
        "the span stopped early: {} lines",
        span.end - span.start
    );
    let out = replace_record(&text, "player", &lines[span].join("\n")).expect("splices");
    assert_eq!(out, text);
}

// ---------------------------------------------------------------------------
// The field splice, and the raster, against the same real files
// ---------------------------------------------------------------------------

use yugen_editor::field::{replace_frames, replace_head_key};
use yugen_editor::raster::raster;
use yugen_editor::sprite::{art_ids, frames_body, read};

/// Every record in `content/` that has art, as `(file, id)`.
fn art_records() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    for path in content_files() {
        let text = std::fs::read_to_string(&path).expect("readable");
        for id in art_ids(&text) {
            out.push((path.clone(), id));
        }
    }
    out
}

#[test]
fn rewriting_every_real_frames_body_with_itself_changes_nothing() {
    // The same property the record splice is held to, one level finer. This is
    // the one that matters for the editor: a redraw goes through
    // `replace_frames`, and anything it gets wrong about where a sequence ends
    // or where a `'''` body starts shows up here as a changed file.
    let records = art_records();
    assert!(
        records.len() > 20,
        "found only {} art records",
        records.len()
    );

    let mut sequences = 0usize;
    for (path, id) in &records {
        let text = std::fs::read_to_string(path).expect("readable");
        let sprite = read(&text, id).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

        for (i, seq) in sprite.seqs.iter().enumerate() {
            let body = frames_body(&seq.frames);
            let out = replace_frames(&text, id, &sprite.sub(), i, &body)
                .unwrap_or_else(|e| panic!("{}: [{id}] seq {i}: {e}", path.display()));
            assert_eq!(
                out,
                text,
                "{}: rewriting [{id}] sequence {i} ({}) with itself changed the file",
                path.display(),
                seq.state
            );
            sequences += 1;
        }
    }
    assert!(sequences > 60, "only walked {sequences} sequences");
}

#[test]
fn every_real_frame_rasterises() {
    // The editor draws every frame it opens, so a frame the raster refuses is a
    // record the editor cannot show. There are none today, and this is what says
    // so — including the grain-2 records, whose rows are `cellsW * grain` wide.
    let mut frames = 0usize;
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        let pal = sprite
            .colours()
            .unwrap_or_else(|e| panic!("{}: [{id}]: {e}", path.display()));

        for seq in &sprite.seqs {
            for frame in &seq.frames {
                let px = raster(frame, &pal, sprite.texel_w(), sprite.texel_h())
                    .unwrap_or_else(|e| panic!("{}: [{id}] {}: {e}", path.display(), seq.state));
                assert_eq!(px.len(), (sprite.texel_w() * sprite.texel_h() * 4) as usize);
                frames += 1;
            }
        }
    }
    assert!(frames > 100, "only rasterised {frames} frames");
}

#[test]
fn rewriting_every_real_palette_with_itself_changes_nothing() {
    // `pal` is the other thing the editor writes, and it is the head key that is
    // hardest to find: it sits above a wall of comment explaining what each
    // index is for, and on a mob it is spelled `art.pal`.
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");

        let lines: Vec<&str> = text.lines().collect();
        let head = yugen_editor::field::head_span(&text, &id).expect("has a head");
        let key = sprite.pal_key();
        let span = yugen_editor::field::key_span(&text, head, &key)
            .unwrap_or_else(|| panic!("{}: [{id}] has no `{key}`", path.display()));

        let out = replace_head_key(&text, &id, &key, lines[span.start])
            .unwrap_or_else(|e| panic!("{}: [{id}]: {e}", path.display()));
        assert_eq!(
            out,
            text,
            "{}: rewriting [{id}]'s `{key}` with itself changed the file",
            path.display()
        );
    }
}

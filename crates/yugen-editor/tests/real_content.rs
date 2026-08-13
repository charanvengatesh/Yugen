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

#[test]
fn rewriting_every_real_grid_with_itself_changes_nothing() {
    // `cellsW`/`cellsH` are the third thing the editor writes, and the newest:
    // a resize is the only edit that moves them. The property is the same one
    // the palette is held to, and it is worth its own test because the failure
    // mode is worse. A `pal` line this got wrong draws the wrong colour; a
    // `cellsH` line it got wrong authors a record whose frames no longer match
    // the grid they declare, and `bake_frame` answers that with a panic at
    // `PreStartup` rather than an error anybody can read.
    //
    // The spelling is prefixed, so this covers `art.cellsW` on twenty mobs as
    // well as the bare form on the sprites.
    let mut grids = 0usize;
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");

        for (name, n) in [("cellsW", sprite.cells_w), ("cellsH", sprite.cells_h)] {
            let key = format!("{}{name}", sprite.prefix);
            let line = format!("{key} = {n}");
            let out = replace_head_key(&text, &id, &key, &line)
                .unwrap_or_else(|e| panic!("{}: [{id}] {key}: {e}", path.display()));
            assert_eq!(
                out,
                text,
                "{}: rewriting [{id}]'s `{key}` with itself changed the file",
                path.display()
            );
            grids += 1;
        }
    }
    assert!(grids > 40, "only walked {grids} grid keys");
}

// ---------------------------------------------------------------------------
// The generators, against the same real files
// ---------------------------------------------------------------------------

use yugen_editor::procgen::anim;
use yugen_editor::procgen::ops;
use yugen_editor::procgen::resize::{Anchor, Mode, resize_sprite};

#[test]
fn appending_a_generated_record_to_every_real_file_keeps_it_parseable() {
    // Creating a record is the only write in this crate that ADDS rather than
    // replaces, so it is the only one the byte-identity property cannot cover.
    // What it can be held to instead: the file still parses, it gained exactly
    // one record, everything that was already there is still there, and the new
    // record reads back as what was written.
    //
    // Against every real file, because the interesting cases are the ones that
    // end oddly — a trailing `'''`, a comment with no newline after it, a file
    // whose last record is the player's four hundred lines.
    use yugen_editor::splice::append_record;
    use yugen_editor::template;

    let mut files = 0usize;
    for path in content_files() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let before = record_ids(&text);

        let r = Recipe {
            seed: 0x5eed,
            ..Recipe::default()
        };
        let pal = generate_pal(&r, Hsl::new(200.0, 0.55, 0.5));
        let record = template::sprite_record(
            "zz_probe_record",
            "Probe",
            (8, 8),
            &pal,
            &[generate(&r)],
            Some(r.seed),
        );
        let next = append_record(&text, &record);

        next.parse::<toml::Table>()
            .unwrap_or_else(|e| panic!("{} stopped being TOML: {e}", path.display()));

        let after = record_ids(&next);
        assert_eq!(
            after.len(),
            before.len() + 1,
            "{} gained {} records",
            path.display(),
            after.len() - before.len()
        );
        assert_eq!(
            &after[..before.len()],
            &before[..],
            "{} reordered the records that were already there",
            path.display()
        );
        assert_eq!(after.last().map(String::as_str), Some("zz_probe_record"));

        // And every original record still splices over itself byte-identically
        // in the new text — the append did not disturb a span above it.
        for id in &before {
            let span = record_lines(&next, id).expect("still there");
            let lines: Vec<&str> = next.lines().collect();
            let same = lines[span].join("\n");
            assert_eq!(
                replace_record(&next, id, &same).expect("splices"),
                next,
                "{}: [{id}] moved when a record was appended",
                path.display()
            );
        }

        let back = read(&next, "zz_probe_record").expect("the new record reads");
        assert_eq!((back.cells_w, back.cells_h), (8, 8));
        assert_eq!(back.pal, pal);
        files += 1;
    }
    assert!(files > 20, "only appended to {files} files");
}

#[test]
fn every_art_record_is_drawn_on_the_same_eight_by_eight_grid() {
    // The rule the migration bought, and the reason it was worth a redraw.
    //
    // One grid for every drawing in the game makes a class of operation TOTAL
    // instead of conditional: `ops::rotate_cw` returns `Some` for everything, a
    // frame can be pasted from any record into any other, and a generator with a
    // fixed footprint fits all of them. None of that was true while the tree was
    // 8x8 for eighty-seven records, 8x10 for the player and five other shapes
    // across the bestiary.
    //
    // There is no exception list, deliberately. An exception here would be a
    // hole rather than a note — every one of those operations would have to
    // start asking again.
    let mut records = 0usize;
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        assert_eq!(
            (sprite.cells_w, sprite.cells_h),
            (8, 8),
            "{}: [{id}] is {}x{} — every drawing in the game is 8x8",
            path.display(),
            sprite.cells_w,
            sprite.cells_h
        );
        records += 1;
    }
    assert!(records > 100, "only checked {records} art records");
}

#[test]
fn resizing_every_real_record_to_its_own_size_changes_nothing() {
    // The resize's version of "splice a record over itself". A migration is only
    // reviewable if running it twice is a no-op, and this is that claim against
    // every grid in the tree rather than against a fixture.
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        let cells = (sprite.cells_w, sprite.cells_h);

        for anchor in Anchor::ALL {
            for mode in Mode::ALL {
                let (out, report) = resize_sprite(&sprite, cells, anchor, mode);
                assert!(
                    report.lossless(),
                    "{}: [{id}] lost {} texels resizing to the size it already is",
                    path.display(),
                    report.total_lost()
                );
                assert_eq!(
                    out,
                    sprite,
                    "{}: [{id}] moved under a {} / {} resize to its own size",
                    path.display(),
                    anchor.name(),
                    mode.name()
                );
            }
        }
    }
}

#[test]
fn fitting_every_real_record_to_eight_by_eight_is_now_a_no_op() {
    // The migration, after the fact. Every record is already 8x8, so running the
    // fit again has to cost nothing — which is what makes `yugen-fit` safe to
    // re-run over the tree and what made the mob half of the migration a
    // provable visual no-op at the time.
    //
    // This is the weaker sibling of
    // `every_art_record_is_drawn_on_the_same_eight_by_eight_grid`, and it earns
    // its place by checking the TEXELS rather than the declaration: a record
    // that declared 8x8 while its frames had drifted would pass the other test
    // and fail this one.
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        let (_, report) = resize_sprite(&sprite, (8, 8), Anchor::FeetCentre, Mode::PadCrop);
        assert!(
            report.lossless(),
            "{}: [{id}] is {}x{} and fitting it to 8x8 dropped {} texels",
            path.display(),
            sprite.cells_w,
            sprite.cells_h,
            report.total_lost()
        );
    }
}

#[test]
fn a_resized_record_still_splices_back_into_its_file() {
    // The round trip the whole migration depends on: resize the model, splice
    // the frames and the two grid keys, re-read, and get back what was written.
    // If `replace_frames` and `replace_head_key` disagree about anything, the
    // file that results parses fine and draws wrong.
    // The target is 10x10 rather than 8x8 because the tree is 8x8 now: fitting a
    // record to the grid it already declares moves no lines, which would make
    // this pass without ever exercising the splice it exists to test. A pad to
    // something bigger rewrites every frame and both grid keys, which is exactly
    // what a real resize does.
    let mut checked = 0usize;
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        let (out, _) = resize_sprite(&sprite, (10, 10), Anchor::FeetCentre, Mode::PadCrop);

        let mut next = text.clone();
        for (i, seq) in out.seqs.iter().enumerate() {
            next = replace_frames(&next, &id, &out.sub(), i, &frames_body(&seq.frames))
                .unwrap_or_else(|e| panic!("{}: [{id}] seq {i}: {e}", path.display()));
        }
        for (name, n) in [("cellsW", out.cells_w), ("cellsH", out.cells_h)] {
            let key = format!("{}{name}", out.prefix);
            next = replace_head_key(&next, &id, &key, &format!("{key} = {n}"))
                .unwrap_or_else(|e| panic!("{}: [{id}] {key}: {e}", path.display()));
        }

        // It still parses, and reading it back gives the model that was written.
        next.parse::<toml::Table>()
            .unwrap_or_else(|e| panic!("{}: [{id}] no longer TOML: {e}", path.display()));
        let back = read(&next, &id)
            .unwrap_or_else(|e| panic!("{}: [{id}] no longer reads: {e}", path.display()));
        assert_eq!(
            back,
            out,
            "{}: [{id}] did not survive the splice",
            path.display()
        );
        checked += 1;
    }
    assert!(checked > 15, "only round-tripped {checked} resized records");
}
use yugen_editor::procgen::palette::Hsl;
use yugen_editor::procgen::sprite::{Recipe, Symmetry, generate, generate_pal};

#[test]
fn a_roll_into_any_real_record_fits_the_grid_that_record_declares() {
    // The generate panel's one safety property. It overwrites the recipe's
    // `w`/`h` from the open record every frame precisely so a roll cannot be a
    // different shape from the record it lands in — and this is that claim,
    // checked against every grid actually in the tree: 4x4, 6x4, 6x6, 8x4, 8x6,
    // 8x8 and the player's 8x10.
    //
    // Worth its own test because nothing downstream would catch it. `contentc`
    // validates no dimensions, so a mis-shaped roll would compile clean and
    // reach the player as a panic at `PreStartup`.
    let mut grids = std::collections::BTreeSet::new();
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        let (w, h) = (sprite.texel_w() as usize, sprite.texel_h() as usize);
        grids.insert((w, h));

        for symmetry in Symmetry::ALL {
            for seed in [1u32, 7, 4096] {
                let r = Recipe {
                    seed,
                    w,
                    h,
                    symmetry,
                    ..Recipe::default()
                };
                let f = generate(&r);
                assert_eq!(
                    f.len(),
                    h,
                    "{}: [{id}] roll was {} rows",
                    path.display(),
                    f.len()
                );
                for row in &f {
                    assert_eq!(
                        row.chars().count(),
                        w,
                        "{}: [{id}] roll had a {}-wide row",
                        path.display(),
                        row.chars().count()
                    );
                }
                // And the pair really is a pair: the palette generated beside a
                // roll has to answer every index the roll names.
                let pal = generate_pal(&r, Hsl::new(30.0, 0.6, 0.5));
                let colours = raster(
                    &f,
                    &yugen_editor::raster::palette(&pal).expect("colours"),
                    w as u32,
                    h as u32,
                );
                colours.unwrap_or_else(|e| panic!("{}: [{id}]: {e}", path.display()));
            }
        }
    }
    // One grid, now, and that is the migration's whole point — so this asserts
    // the uniformity rather than the variety it used to look for.
    assert_eq!(
        grids,
        std::collections::BTreeSet::from([(8usize, 8usize)]),
        "the tree is supposed to be one grid; found {grids:?}"
    );
}

#[test]
fn every_operator_preserves_the_shape_of_every_real_frame() {
    // The invariant `procgen::ops` is built around, held against what is
    // actually in the tree rather than against the 4x4 fixture its unit tests
    // use. The frames here are 4, 6, 8 and 10 rows tall, square and not, and
    // include the ones whose rows carry trailing spaces.
    //
    // The failure this catches is the expensive one. An operator that returned
    // a frame one row short would write a record whose art and whose `cellsH`
    // disagree, `contentc` would compile it without complaint — it validates no
    // dimensions — and the game would panic at `PreStartup` with a message
    // about a row count rather than about the button that was pressed.
    let mut checked = 0usize;
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        let (w, h) = (sprite.texel_w() as usize, sprite.texel_h() as usize);

        for seq in &sprite.seqs {
            for frame in &seq.frames {
                let mut outs = vec![
                    ops::mirror_x(frame),
                    ops::mirror_y(frame),
                    ops::fold_x(frame),
                    ops::shift(frame, 1, -1, false),
                    ops::shift(frame, -2, 3, true),
                    ops::outline(frame, '1'),
                    ops::auto_shade(frame, '2', '1', (-1, -1)),
                    ops::dither(frame, '1', '2', 0),
                    ops::flood(frame, 0, 0, '1'),
                    ops::remap(frame, &['.', '1', '2', '3', '4', '5', '6', '7', '8', '9']),
                    anim::blink(frame, '2', '1'),
                ];
                // `rotate_cw` is the one that is allowed to decline, and every
                // record that is not square today is exactly why.
                match ops::rotate_cw(frame) {
                    Some(r) => outs.push(r),
                    None => assert_ne!(
                        w,
                        h,
                        "{}: [{id}] is square and refused a rotate",
                        path.display()
                    ),
                }
                outs.extend(anim::bob(frame, 1));
                outs.extend(anim::squash(frame));

                for (i, out) in outs.iter().enumerate() {
                    assert_eq!(
                        out.len(),
                        h,
                        "{}: [{id}] {}: operator {i} changed the row count",
                        path.display(),
                        seq.state
                    );
                    for row in out {
                        assert_eq!(
                            row.chars().count(),
                            w,
                            "{}: [{id}] {}: operator {i} changed a row's width",
                            path.display(),
                            seq.state
                        );
                    }
                }
                checked += 1;
            }
        }
    }
    assert!(
        checked > 100,
        "only ran the operators over {checked} frames"
    );
}

#[test]
fn every_operator_leaves_a_frame_the_raster_still_accepts() {
    // Shape is necessary and not sufficient: an operator that wrote an index the
    // palette does not have keeps the shape exactly and draws magenta. This is
    // the second half, and it goes through the same rasteriser the canvas draws
    // with.
    //
    // The operators are given indices `1` and `2`, which every record in the
    // tree has — the smallest authored palette is three entries including the
    // transparent slot.
    for (path, id) in art_records() {
        let text = std::fs::read_to_string(&path).expect("readable");
        let sprite = read(&text, &id).expect("reads");
        let pal = sprite.colours().expect("a real palette is colours");
        assert!(pal.len() >= 3, "{}: [{id}] has no index 2", path.display());

        for seq in &sprite.seqs {
            for frame in &seq.frames {
                for (i, out) in [
                    ops::mirror_x(frame),
                    ops::outline(frame, '1'),
                    ops::auto_shade(frame, '2', '1', (-1, -1)),
                    ops::dither(frame, '1', '2', 1),
                    anim::blink(frame, '2', '1'),
                ]
                .iter()
                .enumerate()
                {
                    raster(out, &pal, sprite.texel_w(), sprite.texel_h()).unwrap_or_else(|e| {
                        panic!(
                            "{}: [{id}] {}: operator {i} produced art the raster refuses: {e}",
                            path.display(),
                            seq.state
                        )
                    });
                }
            }
        }
    }
}

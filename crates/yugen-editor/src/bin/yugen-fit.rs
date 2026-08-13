//! Resize every art record in a file, from the command line.
//!
//! ```text
//! cargo run -p yugen-editor --bin yugen-fit -- --to 8x8 content/mobs/*.toml
//! cargo run -p yugen-editor --bin yugen-fit -- --to 8x8 --dry-run content/sprites/player.toml
//! ```
//!
//! # Why a second binary and not twenty rounds of clicking
//!
//! The 8x8 migration touches every mob in the tree. Doing it through the window
//! would be twenty open-resize-save cycles with no artefact to review afterwards
//! except the diff. This is one command whose output is the review: it prints
//! what each record cost, and **it refuses by default to run any resize that
//! loses ink.**
//!
//! It is not a second implementation. It calls the same
//! [`resize_sprite`](yugen_editor::procgen::resize::resize_sprite) the dialog
//! calls and splices with the same `field::` functions `Editor::save` uses,
//! including the stale check and the temp-and-rename. A migration performed by a
//! tool that agreed with the editor only approximately would be the worst of
//! both.
//!
//! # What it will not do
//!
//! It does not write `content/ids.lock.json` and it does not run `contentc`,
//! for the reasons in `content/FORMAT.md` §9 — the same reasons the window does
//! not. Run `cargo run -p contentc` afterwards.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use yugen_editor::field::{replace_frames, replace_head_key};
use yugen_editor::procgen::resize::{Anchor, Mode, Report, resize_sprite};
use yugen_editor::sprite::{self, art_ids, frames_body};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut to: Option<(u32, u32)> = None;
    let mut anchor = Anchor::FeetCentre;
    let mut mode = Mode::PadCrop;
    let mut force = false;
    let mut dry_run = false;
    let mut paths: Vec<PathBuf> = Vec::new();

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--to" => {
                let Some(spec) = it.next() else {
                    return fail("--to wants a size, e.g. --to 8x8");
                };
                match parse_size(spec) {
                    Some(size) => to = Some(size),
                    None => return fail(&format!("`{spec}` is not a size like 8x8")),
                }
            }
            "--anchor" => {
                let Some(name) = it.next() else {
                    return fail("--anchor wants feet, centre or topleft");
                };
                anchor = match name.as_str() {
                    "feet" => Anchor::FeetCentre,
                    "centre" | "center" => Anchor::Centre,
                    "topleft" => Anchor::TopLeft,
                    other => return fail(&format!("unknown anchor `{other}`")),
                };
            }
            "--mode" => {
                let Some(name) = it.next() else {
                    return fail("--mode wants pad or scale");
                };
                mode = match name.as_str() {
                    "pad" => Mode::PadCrop,
                    "scale" => Mode::ScaleNearest,
                    other => return fail(&format!("unknown mode `{other}`")),
                };
            }
            // Losing ink is not a thing to do by accident, so it takes a word.
            "--force" => force = true,
            "--dry-run" => dry_run = true,
            other if other.starts_with('-') => {
                return fail(&format!("unknown option `{other}`"));
            }
            other => paths.push(PathBuf::from(other)),
        }
    }

    let Some(to) = to else {
        return fail("--to is required, e.g. --to 8x8");
    };
    if paths.is_empty() {
        return fail("give at least one file");
    }

    let mut touched = 0usize;
    let mut skipped = 0usize;
    for path in &paths {
        match fit(path, to, anchor, mode, force, dry_run) {
            Ok(n) => touched += n,
            Err(Skipped(n)) => skipped += n,
        }
    }

    if dry_run {
        println!("\n(dry run — nothing was written)");
    } else if touched > 0 {
        println!("\n{touched} record(s) written — run `cargo run -p contentc` to compile them");
    }
    if skipped > 0 {
        println!(
            "{skipped} record(s) skipped because they would lose ink; --force to do it anyway"
        );
        // Not a failure: skipping is the tool doing its job. The count is the
        // migration's to-do list.
    }
    ExitCode::SUCCESS
}

/// How many records were skipped for losing ink.
struct Skipped(usize);

fn fit(
    path: &Path,
    to: (u32, u32),
    anchor: Anchor,
    mode: Mode,
    force: bool,
    dry_run: bool,
) -> Result<usize, Skipped> {
    let Ok(original) = std::fs::read_to_string(path) else {
        eprintln!("{}: cannot read", path.display());
        return Ok(0);
    };

    let mut text = original.clone();
    let mut written = 0usize;
    let mut skipped = 0usize;

    for id in art_ids(&original) {
        // Read from the ORIGINAL, not from the running text: the spans a splice
        // uses are line offsets, and reading a half-spliced file to find the
        // next record would be reading a moving target. Each record's own splice
        // is applied to the accumulated text, which is safe because a record's
        // span does not move when a different record's frames are replaced with
        // frames of the same line count — and when it does, the re-read below
        // is what catches it.
        let sprite = match sprite::read(&text, &id) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{}: [{id}]: {e}", path.display());
                continue;
            }
        };

        if (sprite.cells_w, sprite.cells_h) == to {
            println!("{}: [{id}] already {}x{}", path.display(), to.0, to.1);
            continue;
        }

        let (out, report) = resize_sprite(&sprite, to, anchor, mode);
        print_report(path, &id, &sprite, to, &report);

        if !report.lossless() && !force {
            skipped += 1;
            continue;
        }

        let mut next = text.clone();
        let mut ok = true;
        for (i, seq) in out.seqs.iter().enumerate() {
            match replace_frames(&next, &id, &out.sub(), i, &frames_body(&seq.frames)) {
                Ok(t) => next = t,
                Err(e) => {
                    eprintln!("{}: [{id}] sequence {i}: {e}", path.display());
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            for (name, n) in [("cellsW", out.cells_w), ("cellsH", out.cells_h)] {
                let key = format!("{}{name}", out.prefix);
                match replace_head_key(&next, &id, &key, &format!("{key} = {n}")) {
                    Ok(t) => next = t,
                    Err(e) => {
                        eprintln!("{}: [{id}] {key}: {e}", path.display());
                        ok = false;
                        break;
                    }
                }
            }
        }
        if !ok {
            continue;
        }

        // Cheap proof that the splice said what the model meant, before anything
        // reaches the disk. This is the check the interactive tool gets for free
        // by re-reading on the next open.
        match sprite::read(&next, &id) {
            Ok(back) if back == out => {}
            Ok(_) => {
                eprintln!(
                    "{}: [{id}]: the splice did not round-trip — not written",
                    path.display()
                );
                continue;
            }
            Err(e) => {
                eprintln!(
                    "{}: [{id}]: spliced file no longer reads: {e}",
                    path.display()
                );
                continue;
            }
        }

        text = next;
        written += 1;
    }

    if written > 0 && !dry_run {
        // The same stale check and temp-and-rename `Editor::save` does: a
        // half-written content file is one `contentc` cannot read.
        match std::fs::read_to_string(path) {
            Ok(disk) if disk != original => {
                eprintln!(
                    "{}: changed on disk while this ran — not written",
                    path.display()
                );
                return Ok(0);
            }
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                return Ok(0);
            }
            _ => {}
        }
        let tmp = path.with_extension("toml.tmp");
        if let Err(e) = std::fs::write(&tmp, &text) {
            eprintln!("{}: {e}", tmp.display());
            return Ok(0);
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            eprintln!("{}: {e}", path.display());
            return Ok(0);
        }
    }

    if skipped > 0 {
        return Err(Skipped(skipped));
    }
    Ok(written)
}

fn print_report(
    path: &Path,
    id: &str,
    sprite: &yugen_editor::sprite::Sprite,
    to: (u32, u32),
    report: &Report,
) {
    let head = format!(
        "{}: [{id}] {}x{} -> {}x{}, {} frames",
        path.display(),
        sprite.cells_w,
        sprite.cells_h,
        to.0,
        to.1,
        report.frames
    );
    if report.lossless() {
        println!("{head} — nothing lost");
        return;
    }
    println!(
        "{head} — {} frame(s) LOSE {} texels:",
        report.lost.len(),
        report.total_lost()
    );
    for &(si, fi, n) in &report.lost {
        let pose = sprite.seqs.get(si).map_or("?", |s| s.state.as_str());
        println!("    {pose} frame {fi}: {n} texels");
    }
}

fn parse_size(spec: &str) -> Option<(u32, u32)> {
    let (w, h) = spec.split_once(['x', 'X'])?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

fn fail(why: &str) -> ExitCode {
    eprintln!("yugen-fit: {why}");
    eprintln!(
        "usage: yugen-fit --to WxH [--anchor feet|centre|topleft] [--mode pad|scale] [--force] [--dry-run] FILE..."
    );
    ExitCode::FAILURE
}

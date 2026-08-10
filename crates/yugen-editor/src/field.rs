//! Replace one FIELD inside a record, leaving the rest of the record alone.
//!
//! # Why [`crate::splice`] is not enough
//!
//! `splice` protects the file around a record. That is the right granularity for
//! a tool that rewrites a record wholesale, and it is the wrong granularity for a
//! pixel editor, because of what a real record looks like:
//!
//! ```toml
//! [player]
//! cellsW = 8
//!
//! # --- idle ---------------------------------------------------------------
//! # Ambient, not a loop: a slow breath with an occasional blink on top. The
//! # first two frames alternate at `fps` — exhale, then inhale.
//! [[player.seq]]
//! state = "idle"
//! frames = '''
//! ...
//! '''
//!
//! # --- run ----------------------------------------------------------------
//! [[player.seq]]
//! ```
//!
//! Those banners are INSIDE `[player]`. `content/sprites/player.toml` is roughly
//! two-thirds prose by line count, and nearly all of it sits between sequences —
//! the paragraph explaining why the trailing arm hangs in column 1 is the most
//! valuable thing in the file, and it is the thing a redraw most needs to keep.
//!
//! An editor that parses a record into a model and re-emits it destroys every one
//! of those lines. It would pass `splice`'s guard — the header still says
//! `[player]`, the file around it survives byte for byte — and still commit
//! exactly the loss FORMAT.md §6 and §9 exist to prevent. So the granularity has
//! to go one level finer: **find the art, replace the art, touch nothing else.**
//!
//! # What this module edits
//!
//! Line spans, not values. A frames body is `cellsH * grain` lines of digits and
//! a pixel edit changes one character of one of them, so the smallest honest edit
//! is "these lines become those lines" — and every byte outside them, comment or
//! key or delimiter, is copied through untouched.

use std::ops::Range;

use crate::splice::{code_mask, is_record_header, record_lines};

/// Everything wrong that a caller can do anything about.
#[derive(Debug, PartialEq, Eq)]
pub enum FieldError {
    /// No `[id]` header in the file.
    NoSuchRecord(String),
    /// Fewer than `index + 1` entries of that table-array in the record.
    NoSuchSeq {
        id: String,
        sub: String,
        index: usize,
        have: usize,
    },
    /// No `key = …` line where one was expected.
    NoSuchKey { id: String, key: String },
    /// The key exists but is not a `'''` body, so it has no art to replace.
    ///
    /// Its own variant rather than a `NoSuchKey` because the two want different
    /// fixes: one is a typo, the other is an editor pointed at a scalar.
    NotABody { id: String, key: String },
}

impl std::fmt::Display for FieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldError::NoSuchRecord(id) => write!(f, "no record `[{id}]` in this file"),
            FieldError::NoSuchSeq {
                id,
                sub,
                index,
                have,
            } => write!(
                f,
                "record `[{id}]` has {have} `[[{id}.{sub}]]` entries, wanted #{index}"
            ),
            FieldError::NoSuchKey { id, key } => write!(f, "no `{key} =` in `[{id}]`"),
            FieldError::NotABody { id, key } => {
                write!(f, "`{key}` in `[{id}]` is not a ''' body")
            }
        }
    }
}

impl std::error::Error for FieldError {}

/// The dotted path a sub-table header names, and whether it is a table-ARRAY.
///
/// `[[player.seq]]` is `("player.seq", true)`; `[player.heat]` is
/// `("player.heat", false)`. Both are inside a record — see
/// [`crate::splice::is_record_header`], which rejects exactly these — and this is
/// the function that tells them apart from each other.
fn sub_header(line: &str) -> Option<(&str, bool)> {
    let s = line.trim();
    let (inner, array) = match s.strip_prefix("[[").and_then(|r| r.strip_suffix("]]")) {
        Some(inner) => (inner, true),
        None => (s.strip_prefix('[')?.strip_suffix(']')?, false),
    };
    let path = inner.trim();
    let ok = path.contains('.')
        && path
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.');
    ok.then_some((path, array))
}

/// True for `key = …` at the start of a line, and false for a longer key that
/// merely starts with the same letters.
///
/// The `=` test is what separates `fps` from `fpsScale` and `blink` from
/// `blinkEvery` — both pairs exist in the sprite schema, so a prefix match here
/// would edit the wrong field with no error anywhere.
fn is_key(line: &str, key: &str) -> bool {
    let t = line.trim_start();
    match t.strip_prefix(key) {
        Some(rest) => rest.trim_start().starts_with('='),
        None => false,
    }
}

/// The record's own key lines: everything between its `[id]` header and its first
/// sub-table.
///
/// This is where `pal`, `cellsW`, `cellsH` and `grain` live. Bounding the search
/// here rather than over the whole record matters for `fps`, which the sprite
/// schema declares BOTH on the sprite and on each sequence: an unbounded search
/// for `fps =` inside `[player]` finds the first sequence's rate and edits that.
pub fn head_span(text: &str, id: &str) -> Result<Range<usize>, FieldError> {
    let record = record_lines(text, id).ok_or_else(|| FieldError::NoSuchRecord(id.to_string()))?;
    let lines: Vec<&str> = text.lines().collect();
    let code = code_mask(&lines);

    // The header sits after the record's comment paragraph, so it is found rather
    // than assumed to be at `record.start`.
    let header = (record.start..record.end)
        .find(|&i| code[i] && is_record_header(lines[i]) == Some(id))
        .ok_or_else(|| FieldError::NoSuchRecord(id.to_string()))?;

    let mut end = header + 1;
    while end < record.end && !(code[end] && sub_header(lines[end]).is_some()) {
        end += 1;
    }
    // The blank line and banner sitting above the first sequence introduce IT,
    // not the keys above them — the same rule this module applies in
    // `seq_spans`. Without the give-back, replacing `pal` on its last line would
    // be replacing a line inside `# --- idle ---`.
    while end > header + 1
        && (lines[end - 1].trim().is_empty() || lines[end - 1].trim_start().starts_with('#'))
    {
        end -= 1;
    }
    Ok(header + 1..end)
}

/// The line span of every `[[id.sub]]` entry, in file order.
///
/// `sub` is the dotted tail — `"seq"` for a standalone sprite, `"art.seq"` for a
/// mob's inline art. Taking it as an argument rather than hard-coding `seq` is
/// what lets one editor drive both hosts, which is the same reason
/// `yugen-render`'s content bridge is a trait rather than two functions.
///
/// A span starts at the `[[…]]` line, NOT at the banner above it. That is the
/// opposite of [`crate::splice::record_lines`]'s rule and deliberately so: this
/// module never replaces a whole entry, only a field inside one, so pulling the
/// prose into the span would only widen what a caller could damage.
pub fn seq_spans(text: &str, id: &str, sub: &str) -> Result<Vec<Range<usize>>, FieldError> {
    let record = record_lines(text, id).ok_or_else(|| FieldError::NoSuchRecord(id.to_string()))?;
    let lines: Vec<&str> = text.lines().collect();
    let code = code_mask(&lines);
    let want = format!("{id}.{sub}");

    let starts: Vec<usize> = (record.start..record.end)
        .filter(|&i| code[i] && sub_header(lines[i]) == Some((want.as_str(), true)))
        .collect();

    Ok(starts
        .iter()
        .map(|&start| {
            // To the next sub-table of ANY path, or the end of the record. Any
            // path, because a sequence is ended just as surely by a sibling group
            // as by the next sequence.
            let mut end = start + 1;
            while end < record.end && !(code[end] && sub_header(lines[end]).is_some()) {
                end += 1;
            }
            // Trailing blanks and the banner for the NEXT entry belong to what
            // comes after, so they are handed back.
            while end > start + 1
                && (lines[end - 1].trim().is_empty()
                    || lines[end - 1].trim_start().starts_with('#'))
            {
                end -= 1;
            }
            start..end
        })
        .collect())
}

/// The lines of `key = …` inside `within`, delimiters included for a `'''` body.
pub fn key_span(text: &str, within: Range<usize>, key: &str) -> Option<Range<usize>> {
    let lines: Vec<&str> = text.lines().collect();
    let code = code_mask(&lines);
    let start =
        (within.start..within.end.min(lines.len())).find(|&i| code[i] && is_key(lines[i], key))?;

    // An odd number of `'''` on the key's own line opens a body; the closing
    // delimiter is then the next line that carries one.
    if lines[start].matches("'''").count().is_multiple_of(2) {
        return Some(start..start + 1);
    }
    let mut end = start + 1;
    while end < lines.len() && lines[end].matches("'''").count().is_multiple_of(2) {
        end += 1;
    }
    Some(start..(end + 1).min(lines.len()))
}

/// The ART: the lines strictly between a body's `'''` delimiters.
///
/// This is the span a pixel editor owns. The `frames = '''` line and the closing
/// `'''` are the KEY's, not the art's, so they stay put and a replacement never
/// has to reproduce them — which is one fewer way to write a file that no longer
/// parses.
pub fn body_span(text: &str, within: Range<usize>, key: &str) -> Option<Range<usize>> {
    let span = key_span(text, within, key)?;
    (span.len() >= 2).then(|| span.start + 1..span.end - 1)
}

/// Replace lines `span` with `new`, one element per line.
///
/// Everything outside the span is copied through byte for byte, and the file's
/// trailing newline — or its absence — is taken from the original for the same
/// reason [`crate::splice::replace_record`] takes it: inventing one puts a
/// spurious hunk at the bottom of every diff the tool touches.
pub fn replace_lines(text: &str, span: Range<usize>, new: &[String]) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::with_capacity(text.len());
    for line in &lines[..span.start.min(lines.len())] {
        out.push_str(line);
        out.push('\n');
    }
    for line in new {
        out.push_str(line);
        out.push('\n');
    }
    for line in &lines[span.end.min(lines.len())..] {
        out.push_str(line);
        out.push('\n');
    }
    if !text.ends_with('\n') {
        out.pop();
    }
    out
}

/// Replace the art of sequence `index` in record `id`, and nothing else.
///
/// `frames` is the body as it should read: `cellsH * grain` rows per frame, a
/// blank line between frames, no `'''`. This is the one function the canvas
/// calls, and the whole module exists so that this call cannot take a comment
/// with it.
pub fn replace_frames(
    text: &str,
    id: &str,
    sub: &str,
    index: usize,
    frames: &[String],
) -> Result<String, FieldError> {
    let spans = seq_spans(text, id, sub)?;
    let span = spans.get(index).ok_or_else(|| FieldError::NoSuchSeq {
        id: id.to_string(),
        sub: sub.to_string(),
        index,
        have: spans.len(),
    })?;
    let art = body_span(text, span.clone(), "frames").ok_or_else(|| {
        // A `frames` key that is not a body is a different mistake from a missing
        // one, and the schema makes `frames` required, so tell them apart.
        if key_span(text, span.clone(), "frames").is_some() {
            FieldError::NotABody {
                id: id.to_string(),
                key: "frames".to_string(),
            }
        } else {
            FieldError::NoSuchKey {
                id: id.to_string(),
                key: "frames".to_string(),
            }
        }
    })?;
    Ok(replace_lines(text, art, frames))
}

/// Replace a scalar key on the record itself — `pal`, `cellsW`, `grain`.
///
/// `line` is the whole replacement line, `pal = [".", "#1b1220"]` and not just
/// the value, because the key's own spelling and spacing are part of what the
/// file looks like and this module has no opinion about either.
pub fn replace_head_key(text: &str, id: &str, key: &str, line: &str) -> Result<String, FieldError> {
    let head = head_span(text, id)?;
    let span = key_span(text, head, key).ok_or_else(|| FieldError::NoSuchKey {
        id: id.to_string(),
        key: key.to_string(),
    })?;
    Ok(replace_lines(text, span, &[line.to_string()]))
}

/// Set a head key whether or not the record already writes it.
///
/// # Why an insert is needed at all
///
/// Most schema fields carry a default, and `content/` leans on that hard: a
/// sound record that wants the standard release simply does not write
/// `release`, and `content/sounds/player.toml` omits `attack` on six of its
/// eight records. So a tool that turns a knob has to be able to ADD the line,
/// and a tool that can only replace would silently do nothing for exactly the
/// fields a designer is most likely to reach for first.
///
/// A new key goes at the END of the record's own keys, above whatever sub-tables
/// follow. That is the only position that is right without knowing the file:
/// inserting in schema order would need the schema, and inserting at the top
/// would push the line above the comment that explains the key below it.
pub fn set_head_key(text: &str, id: &str, key: &str, line: &str) -> Result<String, FieldError> {
    let head = head_span(text, id)?;
    match key_span(text, head.clone(), key) {
        Some(span) => Ok(replace_lines(text, span, &[line.to_string()])),
        // An empty span at `head.end` is an insert: `replace_lines` copies
        // everything before it, writes the line, then copies everything from the
        // same index on.
        None => Ok(replace_lines(text, head.end..head.end, &[line.to_string()])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped like `content/sprites/player.toml`: banners between sequences, a
    /// sprite-wide `fps` above a per-sequence one, and art that must survive.
    const FILE: &str = "\
# The file's own banner.\n\
\n\
# The player.\n\
[player]\n\
name = \"Player\"\n\
cellsW = 4\n\
cellsH = 2\n\
pal = [\".\", \"#1b1220\", \"#3ec8b4\"]\n\
fps = 8.0\n\
\n\
# --- idle -------------------------------------------------------------------\n\
# A slow breath. The trailing arm hangs in col 1 and it is the ONLY overhang\n\
# the resting figure spends.\n\
[[player.seq]]\n\
state = \"idle\"\n\
mode = \"ambient\"\n\
fps = 1.6\n\
frames = '''\n\
.11.\n\
1..1\n\
\n\
.22.\n\
2..2\n\
'''\n\
\n\
# --- run --------------------------------------------------------------------\n\
# Phase-driven, so footfalls land where the feet are.\n\
[[player.seq]]\n\
state = \"run\"\n\
frames = '''\n\
.12.\n\
1..2\n\
'''\n\
\n\
[icon]\n\
name = \"Icon\"\n\
";

    #[test]
    fn the_records_own_keys_stop_at_its_first_sequence() {
        // The failure this prevents is silent and specific: `fps` is declared on
        // the sprite AND on every sequence, so an unbounded search finds idle's
        // 1.6 and retunes the wrong thing.
        let head = head_span(FILE, "player").expect("found");
        let lines: Vec<&str> = FILE.lines().collect();
        assert_eq!(lines[head.start], "name = \"Player\"");
        assert_eq!(lines[head.end - 1], "fps = 8.0");
        let span = key_span(FILE, head, "fps").expect("found");
        assert_eq!(lines[span.start], "fps = 8.0");
    }

    #[test]
    fn a_sequence_span_stops_before_the_next_ones_banner() {
        // The banner belongs to the sequence BELOW it — the same rule `splice`
        // applies to records — so idle's span must not swallow run's paragraph.
        let spans = seq_spans(FILE, "player", "seq").expect("found");
        assert_eq!(spans.len(), 2);
        let lines: Vec<&str> = FILE.lines().collect();
        assert_eq!(lines[spans[0].start], "[[player.seq]]");
        assert_eq!(lines[spans[0].end - 1], "'''");
        assert!(
            (spans[0].start..spans[0].end).all(|i| !lines[i].contains("--- run ---")),
            "run's banner is not part of idle"
        );
    }

    #[test]
    fn the_art_span_is_the_digits_and_not_the_delimiters() {
        let spans = seq_spans(FILE, "player", "seq").expect("found");
        let art = body_span(FILE, spans[0].clone(), "frames").expect("found");
        let lines: Vec<&str> = FILE.lines().collect();
        assert_eq!(lines[art.start], ".11.");
        assert_eq!(lines[art.end - 1], "2..2");
        assert!(!lines[art.start..art.end].contains(&"'''"));
    }

    #[test]
    fn redrawing_one_frame_keeps_every_comment_in_the_record() {
        // The whole reason this module exists. A model-and-re-emit editor passes
        // `splice`'s guard and still deletes all four paragraphs below.
        let new: Vec<String> = [".33.", "3..3", "", ".22.", "2..2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = replace_frames(FILE, "player", "seq", 0, &new).expect("splices");

        assert!(out.contains("# The file's own banner."));
        assert!(out.contains("# --- idle ---"));
        assert!(out.contains("# the resting figure spends."));
        assert!(out.contains("# --- run ---"));
        assert!(out.contains("# Phase-driven, so footfalls land where the feet are."));
        // The edit landed, and only there.
        assert!(out.contains(".33.\n3..3\n"));
        assert!(!out.contains(".11.\n"));
        assert!(out.contains(".12.\n1..2\n"), "run's art is untouched");
        assert!(out.contains("fps = 1.6"), "idle's rate is untouched");
    }

    #[test]
    fn each_sequence_is_addressed_independently() {
        let new = vec!["....".to_string(), "....".to_string()];
        let out = replace_frames(FILE, "player", "seq", 1, &new).expect("splices");
        assert!(out.contains(".11.\n1..1\n"), "idle kept its art");
        assert!(!out.contains(".12.\n"), "run lost its art");
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed["player"]["seq"].as_array().expect("array").len(), 2);
    }

    #[test]
    fn a_sequence_that_is_not_there_is_an_error_and_not_a_silent_no_op() {
        let err = replace_frames(FILE, "player", "seq", 7, &[]).unwrap_err();
        assert_eq!(
            err,
            FieldError::NoSuchSeq {
                id: "player".into(),
                sub: "seq".into(),
                index: 7,
                have: 2
            }
        );
        assert!(replace_frames(FILE, "ghost", "seq", 0, &[]).is_err());
    }

    #[test]
    fn a_palette_edit_touches_one_line() {
        let out = replace_head_key(
            FILE,
            "player",
            "pal",
            "pal = [\".\", \"#1b1220\", \"#3ec8b4\", \"#ffcf5c\"]",
        )
        .expect("splices");
        assert!(out.contains("#ffcf5c"));
        assert!(out.contains("cellsW = 4"));
        assert!(out.contains("# --- idle ---"));
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed["player"]["pal"].as_array().expect("array").len(), 4);
    }

    #[test]
    fn a_key_the_record_does_not_write_yet_is_added_below_the_ones_it_does() {
        // The case that makes `set_head_key` necessary: most schema fields have
        // a default and `content/` leans on it, so the field a designer reaches
        // for first is usually the one with no line to replace.
        let out = set_head_key(FILE, "player", "grain", "grain = 2").expect("inserts");
        let lines: Vec<&str> = out.lines().collect();
        let at = lines.iter().position(|l| *l == "grain = 2").expect("added");
        assert_eq!(lines[at - 1], "fps = 8.0", "below the last key it had");
        assert!(
            lines[at + 1..].iter().any(|l| l.contains("--- idle ---")),
            "and above the banner introducing the first sequence"
        );
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed["player"]["grain"].as_integer(), Some(2));

        // A key that IS written is replaced in place rather than duplicated.
        let out = set_head_key(&out, "player", "grain", "grain = 4").expect("replaces");
        assert_eq!(out.matches("grain =").count(), 1);
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed["player"]["grain"].as_integer(), Some(4));
    }

    #[test]
    fn a_spliced_file_still_parses_and_still_has_its_neighbours() {
        let new = vec!["1111".to_string(), "1111".to_string()];
        let out = replace_frames(FILE, "player", "seq", 0, &new).expect("splices");
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed.len(), 2);
        assert!(parsed.contains_key("icon"));
        assert_eq!(
            parsed["player"]["seq"][0]["frames"].as_str(),
            Some("1111\n1111\n")
        );
    }
}

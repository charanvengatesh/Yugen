//! Replace one record in a content file, byte for byte, leaving the rest alone.
//!
//! # The shape of a record
//!
//! A record is a top-level table — `[stone]` — and everything up to the next
//! top-level table or the end of the file. "Top-level" is doing real work in
//! that sentence: `[[player.seq]]` and `[stone.heat]` are INSIDE a record, not
//! the start of a new one, and a scanner that treated any line starting with
//! `[` as a boundary would cut `player` in half at its first sequence.
//!
//! # Where a record's comments belong
//!
//! To the record BELOW them, and this is the decision the whole module turns on.
//! `content/` files are written with a paragraph above each entry explaining it:
//!
//! ```toml
//! # Peat: the Mirefen's cap — half-drowned moor soil, centuries of moss
//! # pressed flat. Holds water, burns badly, and is the one surface that
//! # slows a body walking over it.
//! [peat]
//! ```
//!
//! Replacing `peat` without its paragraph orphans three lines of prose above a
//! record they no longer describe. Deleting it takes the reasoning with it. So a
//! record's span starts at the first line of the comment block immediately above
//! its header, and a tool that rewrites a record is handed that prose to keep or
//! to change deliberately.
//!
//! A blank line ends the block. That is what separates a record's own paragraph
//! from a section banner further up — the files use one consistently, and
//! `terrain.toml` opens with a header comment that must never be swallowed by
//! the first record.

use std::ops::Range;

/// Everything wrong that a caller can do anything about.
#[derive(Debug, PartialEq, Eq)]
pub enum SpliceError {
    /// No `[id]` header in the file.
    NoSuchRecord(String),
    /// The replacement does not begin with the header it claims to be.
    ///
    /// Checked because the failure it prevents is silent: splicing text for
    /// `granite` into `peat`'s span leaves a file that parses, compiles, and has
    /// quietly renamed a record — which `ids.lock.json` then treats as one id
    /// deleted and another added.
    WrongHeader { want: String, got: String },
}

impl std::fmt::Display for SpliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpliceError::NoSuchRecord(id) => write!(f, "no record `[{id}]` in this file"),
            SpliceError::WrongHeader { want, got } => {
                write!(f, "replacement for `[{want}]` starts with `{got}`")
            }
        }
    }
}

impl std::error::Error for SpliceError {}

/// Which lines are CODE rather than the inside of a `'''` body.
///
/// Every scanner in this crate is line-based, and a line-based scanner that does
/// not know about literal bodies will read art as structure. FORMAT.md §5 makes
/// a body's contents arbitrary text — a structure template's legend is whatever
/// characters that structure chose — so nothing stops a body line from looking
/// exactly like `[stone]`. Reading one as a record header would cut a record in
/// half in the middle of its own art, and the resulting splice would be
/// syntactically fine and semantically shredded.
///
/// The rule is TOML's: `'''` toggles, and a line may open and close in one go
/// (`x = '''y'''`), so occurrences are counted rather than matched. The opening
/// and closing lines are both CODE — the delimiter belongs to the key, not to
/// the art — which is why the toggle is applied after the line is classified.
pub(crate) fn code_mask(lines: &[&str]) -> Vec<bool> {
    let mut mask = Vec::with_capacity(lines.len());
    let mut in_body = false;
    for line in lines {
        // Classify first: a line holding a delimiter is the key's, not the art's.
        mask.push(!in_body);
        let toggles = line.matches("'''").count();
        if toggles % 2 == 1 {
            in_body = !in_body;
        }
    }
    mask
}

/// True for a line that opens a TOP-LEVEL table: `[id]`, not `[a.b]` and not
/// `[[a.b]]`.
///
/// The dotted test is what keeps a record whole. A record's own sub-tables and
/// table-arrays are dotted by construction — FORMAT.md §2 and §4 — so a bare
/// name is the only thing that can start a new one.
pub(crate) fn is_record_header(line: &str) -> Option<&str> {
    let s = line.trim_end();
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    if inner.starts_with('[') || inner.contains('.') {
        return None;
    }
    let id = inner.trim();
    let ok = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    ok.then_some(id)
}

/// The line range a record occupies, comment paragraph included.
///
/// Returned as lines rather than bytes because everything a caller wants to do
/// with it — show it, count it, replace it — is line-shaped, and because a byte
/// range invites slicing a multi-byte character in half.
pub fn record_lines(text: &str, id: &str) -> Option<Range<usize>> {
    let lines: Vec<&str> = text.lines().collect();
    let code = code_mask(&lines);
    let header = lines
        .iter()
        .enumerate()
        .position(|(i, l)| code[i] && is_record_header(l) == Some(id))?;

    // Walk back over the record's own comment paragraph. A blank line stops it,
    // which is what keeps a file's opening banner out of its first record.
    let mut start = header;
    while start > 0 {
        let prev = lines[start - 1].trim();
        if prev.starts_with('#') {
            start -= 1;
        } else {
            break;
        }
    }

    // Forward to the next top-level header, or the end.
    let mut end = header + 1;
    while end < lines.len() && !(code[end] && is_record_header(lines[end]).is_some()) {
        end += 1;
    }
    // Give back any trailing blank lines: they separate this record from the
    // next and belong to neither, so leaving them out means a replacement
    // cannot accidentally eat or duplicate them.
    while end > header + 1 && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    Some(start..end)
}

/// Every top-level record id in the file, in order.
pub fn record_ids(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let code = code_mask(&lines);
    lines
        .iter()
        .enumerate()
        .filter(|(i, _)| code[*i])
        .filter_map(|(_, l)| is_record_header(l))
        .map(str::to_string)
        .collect()
}

/// Replace `id`'s record with `replacement`, returning the whole new file.
///
/// `replacement` is the record's full text — its comment paragraph, its header,
/// and its keys — exactly as it should appear. Everything outside the record's
/// span is preserved byte for byte, including the file's trailing newline or
/// lack of one.
pub fn replace_record(text: &str, id: &str, replacement: &str) -> Result<String, SpliceError> {
    let span = record_lines(text, id).ok_or_else(|| SpliceError::NoSuchRecord(id.to_string()))?;

    // The replacement must be the record it claims to be. See `WrongHeader`.
    let claimed = replacement
        .lines()
        .find_map(is_record_header)
        .unwrap_or_default();
    if claimed != id {
        return Err(SpliceError::WrongHeader {
            want: id.to_string(),
            got: claimed.to_string(),
        });
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::with_capacity(text.len() + replacement.len());
    for line in &lines[..span.start] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(replacement.trim_end_matches('\n'));
    out.push('\n');
    for line in &lines[span.end..] {
        out.push_str(line);
        out.push('\n');
    }
    // `str::lines` drops the information about whether the file ended with a
    // newline, so it is restored from the original rather than assumed. Every
    // file in `content/` ends with one; a tool that silently added or removed
    // one would put a spurious hunk at the bottom of every diff it touched.
    if !text.ends_with('\n') {
        out.pop();
    }
    Ok(out)
}

/// Append a record to the end of a file.
///
/// Appending rather than inserting in sorted position, deliberately: file order
/// feeds code assignment through `content/ids.lock.json`, and while the lock
/// makes reordering SAFE it also makes it pointless churn in a diff. A new
/// record goes at the bottom, where a reviewer expects to find it.
pub fn append_record(text: &str, replacement: &str) -> String {
    let mut out = text.trim_end_matches('\n').to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(replacement.trim_end_matches('\n'));
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = "\
# The ground itself. This banner belongs to the FILE and must never be\n\
# swallowed by the first record below it.\n\
\n\
# Stone is the datum every other hardness is quoted against.\n\
[stone]\n\
name = \"Stone\"\n\
hardness = 3.0\n\
heat.conduct = 56\n\
drop = [{ item = \"stone_chunk\", count = 1 }]\n\
\n\
# Peat: half-drowned moor soil. Burns badly.\n\
[peat]\n\
name = \"Peat\"\n\
hardness = 0.6\n\
\n\
[sand]\n\
name = \"Sand\"\n\
";

    #[test]
    fn a_dotted_table_does_not_start_a_new_record() {
        // The failure this prevents cuts `player` in half at its first
        // `[[player.seq]]`, which would splice a fragment of a sprite over the
        // rest of it.
        assert_eq!(is_record_header("[stone]"), Some("stone"));
        assert_eq!(is_record_header("[icon_pick_gem]"), Some("icon_pick_gem"));
        assert_eq!(is_record_header("[stone.heat]"), None);
        assert_eq!(is_record_header("[[player.seq]]"), None);
        assert_eq!(is_record_header("name = \"Stone\""), None);
        assert_eq!(is_record_header("# [stone]"), None);
        assert_eq!(record_ids(FILE), vec!["stone", "peat", "sand"]);
    }

    #[test]
    fn a_records_span_carries_the_paragraph_that_explains_it() {
        // The whole reason this module has an opinion. Replacing `peat` without
        // its comment orphans prose above a record it no longer describes.
        let span = record_lines(FILE, "peat").expect("found");
        let text: Vec<&str> = FILE.lines().collect();
        assert_eq!(
            text[span.start],
            "# Peat: half-drowned moor soil. Burns badly."
        );
        assert_eq!(text[span.end - 1], "hardness = 0.6");
    }

    #[test]
    fn the_files_own_banner_is_not_part_of_its_first_record() {
        // A blank line ends a comment block, which is what separates a section
        // banner from a record's own paragraph. Without that rule, editing
        // `stone` would rewrite the header of `terrain.toml`.
        let span = record_lines(FILE, "stone").expect("found");
        let text: Vec<&str> = FILE.lines().collect();
        assert_eq!(
            text[span.start],
            "# Stone is the datum every other hardness is quoted against."
        );
        assert!(
            FILE.lines()
                .take(span.start)
                .any(|l| l.contains("belongs to the FILE")),
            "the banner must sit above the span, untouched"
        );
    }

    #[test]
    fn everything_outside_the_record_survives_byte_for_byte() {
        let out = replace_record(
            FILE,
            "peat",
            "# Peat, retuned.\n[peat]\nname = \"Peat\"\nhardness = 0.9\n",
        )
        .expect("splices");
        // The neighbours, their comments and their multiline values are all
        // exactly as they were.
        assert!(out.contains("# The ground itself. This banner belongs to the FILE"));
        assert!(out.contains("# Stone is the datum every other hardness is quoted against."));
        assert!(out.contains("drop = [{ item = \"stone_chunk\", count = 1 }]"));
        assert!(out.contains("[sand]"));
        // The record itself changed, and its old comment went with it.
        assert!(out.contains("# Peat, retuned."));
        assert!(out.contains("hardness = 0.9"));
        assert!(!out.contains("half-drowned moor soil"));
        assert!(!out.contains("hardness = 0.6"));
        // And the file is still three records in the same order.
        assert_eq!(record_ids(&out), vec!["stone", "peat", "sand"]);
    }

    #[test]
    fn replacing_the_last_record_does_not_lose_the_trailing_newline() {
        let out = replace_record(FILE, "sand", "[sand]\nname = \"Sand\"\nhardness = 0.4\n")
            .expect("splices");
        assert!(out.ends_with("hardness = 0.4\n"));
        assert!(!out.ends_with("\n\n"), "no newline was invented either");
    }

    #[test]
    fn a_replacement_for_the_wrong_record_is_refused() {
        // The silent failure this exists to prevent: the file would still parse
        // and compile, and `ids.lock.json` would read it as one id deleted and
        // another added — which renumbers nothing but tombstones a live record.
        let err = replace_record(FILE, "peat", "[granite]\nname = \"Granite\"\n").unwrap_err();
        assert_eq!(
            err,
            SpliceError::WrongHeader {
                want: "peat".into(),
                got: "granite".into()
            }
        );
        assert!(replace_record(FILE, "basalt", "[basalt]\n").is_err());
    }

    #[test]
    fn a_spliced_file_still_parses_as_the_toml_it_was() {
        // The splice is textual, so nothing guarantees the result is valid TOML
        // except the replacement being valid. Checking it here means a caller
        // that builds malformed text finds out at the splice rather than at the
        // next `contentc` run.
        let out = replace_record(FILE, "stone", "[stone]\nname = \"Stone\"\nhardness = 4.0\n")
            .expect("splices");
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed["stone"]["hardness"].as_float(), Some(4.0));
        assert_eq!(parsed["peat"]["hardness"].as_float(), Some(0.6));
    }

    #[test]
    fn appending_puts_a_new_record_at_the_bottom() {
        // Bottom rather than sorted position: the lock makes reordering safe and
        // that does not make it free, and a reviewer looks for a new record at
        // the end of the diff.
        let out = append_record(FILE, "# Basalt.\n[basalt]\nname = \"Basalt\"\n");
        assert_eq!(record_ids(&out), vec!["stone", "peat", "sand", "basalt"]);
        assert!(out.ends_with("name = \"Basalt\"\n"));
        let parsed: toml::Table = out.parse().expect("still TOML");
        assert_eq!(parsed.len(), 4);
    }

    #[test]
    fn a_bracketed_line_inside_a_body_is_art_not_a_header() {
        // FORMAT.md §5: a body's contents are arbitrary text, and a structure
        // legend picks its own characters. Without the `'''` mask this scanner
        // reads the middle row as `[sand]`, ends the record there, and splices
        // over half a structure template.
        let art = "\
[vault]\n\
body = '''\n\
[stone]\n\
[sand]\n\
'''\n\
name = \"Vault\"\n\
\n\
[sand]\n\
name = \"Sand\"\n\
";
        assert_eq!(record_ids(art), vec!["vault", "sand"]);
        let span = record_lines(art, "vault").expect("found");
        assert_eq!(span, 0..6, "the record runs past its own art to `name`");
        // And the real `[sand]` is still findable as its own record.
        let out = replace_record(art, "sand", "[sand]\nname = \"Grit\"\n").expect("splices");
        assert!(out.contains("[stone]\n[sand]\n'''"), "the art survived");
        assert!(out.contains("Grit"));
    }

    #[test]
    fn a_multiline_body_in_a_neighbour_is_untouched() {
        // The case FORMAT.md §5 is about: `#` and trailing whitespace inside
        // `'''` are DATA. A serialiser would normalise both; a splice that does
        // not overlap the record cannot see them at all.
        let art = "[player]\nframes = '''\n..#..  \n.###.  \n'''\n\n[icon]\nname = \"Icon\"\n";
        let out = replace_record(art, "icon", "[icon]\nname = \"Icon 2\"\n").expect("splices");
        assert!(
            out.contains("..#..  \n"),
            "the glyph and its padding survived"
        );
        assert!(out.contains("Icon 2"));
    }
}

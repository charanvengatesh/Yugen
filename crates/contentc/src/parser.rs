//! Record parser: turns a token stream into flat `RawRecord`s keyed by DOTTED
//! path. Nesting is not built here — `heat.conduct` stays the literal key
//! `"heat.conduct"` — because the schema is what decides which dotted paths are
//! legal groups, and reconstructing objects before validation would let a typo
//! silently invent a group.
//!
//! Two structural rules are enforced at this layer because they are about the
//! shape of the file rather than the meaning of a field (FORMAT.md §2.4, §2.6):
//!  - a key may not be both a scalar and a group (`heat 4` + `heat.conduct 5`);
//!  - repeated keys are collected as an ordered occurrence list, and rejected
//!    later by the schema unless the field is a list-of-records. Silent
//!    last-wins clobbering is how content rots.

use crate::bail;
use crate::error::{ContentError, Loc, Result};
use crate::lexer::{Token, lex};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;

/// One occurrence of a key.
///
/// `text` is the rest-of-line value; `lines` is a heredoc body. Usually exactly
/// one is set, but BOTH are set for `art.seq state=run mode=phase |` — a
/// `record[]` element needs its `key=value` attributes and its pixel body on the
/// same occurrence, because they describe one thing (a sprite frame is its art
/// plus the state it belongs to) and splitting them across two keys would put
/// the attributes out of reach of the element the body builds. Neither being set
/// is impossible: a key line always produces at least an empty `text`.
#[derive(Clone, Debug)]
pub struct RawValue {
    pub text: Option<String>,
    pub lines: Option<Vec<String>>,
    pub loc: Loc,
}

/// Insertion-ordered map from dotted key to its occurrences.
///
/// The TS version got ordering free from `Map`. Order is load-bearing for error
/// reporting determinism — which of several bad keys is reported first must not
/// depend on a hash seed — so this is a `Vec` with linear lookup rather than a
/// `HashMap`. Records carry a few dozen fields at most.
#[derive(Clone, Debug, Default)]
pub struct FieldMap {
    entries: Vec<(String, Vec<RawValue>)>,
}

impl FieldMap {
    pub fn get(&self, key: &str) -> Option<&[RawValue]> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn push(&mut self, key: &str, value: RawValue) {
        if let Some((_, v)) = self.entries.iter_mut().find(|(k, _)| k == key) {
            v.push(value);
        } else {
            self.entries.push((key.to_string(), vec![value]));
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &[RawValue])> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct RawRecord {
    pub kind: String,
    pub id: String,
    pub loc: Loc,
    /// dotted key -> every occurrence, in file order.
    pub fields: FieldMap,
}

/// Lexically normalise a path the way node's `path.resolve` does — without
/// touching the filesystem, so a missing file still produces a sane message and
/// symlinks are not collapsed (which would break cycle detection against the
/// path the author actually wrote).
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        normalize(path)
    } else {
        normalize(&std::env::current_dir().unwrap_or_default().join(path))
    }
}

/// `path.relative(root, abs)` — for display only, so a plain strip-prefix with a
/// fallback to the absolute path is enough.
fn display_relative(root: &Path, abs: &Path) -> String {
    abs.strip_prefix(root)
        .unwrap_or(abs)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Lex a file and splice `+include`s, resolved relative to `root` (content/).
/// Cycles are a hard error rather than a silent truncation — an include cycle
/// means the author believes two files each own a definition.
fn lex_with_includes(file: &Path, root: &Path, stack: &mut Vec<PathBuf>) -> Result<Vec<Token>> {
    let abs = absolutize(file);
    if stack.contains(&abs) {
        let chain: Vec<String> = stack
            .iter()
            .chain(std::iter::once(&abs))
            .map(|p| display_relative(root, p))
            .collect();
        bail!("+include cycle: {}", chain.join(" -> "));
    }

    let rel_display = display_relative(root, &abs);
    let rel: Rc<str> = Rc::from(if rel_display.is_empty() {
        abs.to_string_lossy().to_string()
    } else {
        rel_display
    });

    let src = std::fs::read_to_string(&abs)
        .map_err(|e| ContentError::new(format!("cannot read {}: {e}", abs.display())))?;
    let tokens = lex(&src, &rel)?;

    let mut out = Vec::with_capacity(tokens.len());
    stack.push(abs.clone());
    for tk in tokens {
        if let Token::Include { path, .. } = &tk {
            // Resolved relative to content/ per FORMAT.md §2.7; a leading "./"
            // opts into "relative to the including file" for locally-shared
            // fragments.
            let base = if path.starts_with('.') {
                abs.parent().unwrap_or(root).to_path_buf()
            } else {
                root.to_path_buf()
            };
            let nested = lex_with_includes(&base.join(path), root, stack)?;
            out.extend(nested);
            continue;
        }
        out.push(tk);
    }
    stack.pop();

    Ok(out)
}

/// Parse one content file (plus its includes) into records.
pub fn parse_file(file: &Path, root: &Path) -> Result<Vec<RawRecord>> {
    let tokens = lex_with_includes(file, root, &mut Vec::new())?;
    to_records(tokens)
}

fn to_records(tokens: Vec<Token>) -> Result<Vec<RawRecord>> {
    let mut out: Vec<RawRecord> = Vec::new();

    for tk in tokens {
        match tk {
            Token::Include { .. } => continue, // already spliced
            Token::Record { kind, id, loc } => {
                out.push(RawRecord {
                    kind,
                    id,
                    loc,
                    fields: FieldMap::default(),
                });
            }
            Token::Key {
                key, value, loc, ..
            } => {
                let Some(cur) = out.last_mut() else {
                    bail!(&loc, "'{key}' appears before any @record header");
                };
                let value = RawValue {
                    text: Some(value),
                    lines: None,
                    loc,
                };
                check_id_agreement(cur, &key, &value)?;
                cur.fields.push(&key, value);
            }
            Token::Text {
                key,
                value,
                lines,
                loc,
                ..
            } => {
                let Some(cur) = out.last_mut() else {
                    bail!(&loc, "'{key}' appears before any @record header");
                };
                // A heredoc with nothing before its `|` leaves `text` UNSET
                // rather than empty, so a plain `art.idle |` is
                // indistinguishable from what it produced before inline values
                // existed. Downstream code uses that absence to tell "no scalar
                // was written here" from "the author wrote nothing", and the
                // `id` agreement check below depends on it.
                let value = RawValue {
                    text: if value.is_empty() { None } else { Some(value) },
                    lines: Some(lines),
                    loc,
                };
                check_id_agreement(cur, &key, &value)?;
                cur.fields.push(&key, value);
            }
        }
    }

    for rec in &out {
        check_group_conflicts(rec)?;
    }
    Ok(out)
}

/// A redundant `id` key must agree with the header (FORMAT.md §2.2).
fn check_id_agreement(cur: &RawRecord, key: &str, value: &RawValue) -> Result<()> {
    if key == "id"
        && let Some(text) = &value.text
        && *text != cur.id
    {
        bail!(
            &value.loc,
            "id '{text}' contradicts the record header '@{} {}'",
            cur.kind,
            cur.id
        );
    }
    Ok(())
}

/// `heat 4` and `heat.conduct 5` cannot both be meaningful — reject the pair.
fn check_group_conflicts(rec: &RawRecord) -> Result<()> {
    for key in rec.fields.keys() {
        let Some(dot) = key.find('.') else { continue };
        let parent = &key[..dot];
        if let Some(clash) = rec.fields.get(parent) {
            bail!(
                &clash[0].loc,
                "'{parent}' is assigned both as a value and as a group ('{key}')"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn recs(src: &str) -> Result<Vec<RawRecord>> {
        to_records(lex(src, &Rc::from("t.block")).unwrap())
    }

    #[test]
    fn repeated_keys_become_an_ordered_occurrence_list() {
        let r = recs("@sprite s\nseq a\nseq b\nseq c\n").unwrap();
        let occ = r[0].fields.get("seq").unwrap();
        assert_eq!(occ.len(), 3);
        let got: Vec<&str> = occ.iter().map(|v| v.text.as_deref().unwrap()).collect();
        assert_eq!(got, ["a", "b", "c"]);
    }

    #[test]
    fn a_scalar_and_a_group_on_the_same_name_is_rejected() {
        let e = recs("@block b\nheat 4\nheat.conduct 5\n").unwrap_err();
        assert!(e.message.contains("both as a value and as a group"), "{e}");
    }

    #[test]
    fn a_key_before_any_record_header_is_rejected() {
        let e = recs("stray 1\n@block b\n").unwrap_err();
        assert!(e.message.contains("before any @record header"), "{e}");
    }

    #[test]
    fn a_redundant_id_must_agree_with_the_header() {
        assert!(recs("@block stone\nid stone\n").is_ok());
        let e = recs("@block stone\nid granite\n").unwrap_err();
        assert!(e.message.contains("contradicts the record header"), "{e}");
    }

    #[test]
    fn a_bare_heredoc_leaves_text_unset_but_an_inline_one_sets_it() {
        let r = recs("@sprite s\nframes |\n  ab\nseq state=run |\n  cd\n").unwrap();
        let bare = &r[0].fields.get("frames").unwrap()[0];
        assert!(bare.text.is_none());
        assert_eq!(bare.lines.as_deref(), Some(&["ab".to_string()][..]));

        let inline = &r[0].fields.get("seq").unwrap()[0];
        assert_eq!(inline.text.as_deref(), Some("state=run"));
        assert_eq!(inline.lines.as_deref(), Some(&["cd".to_string()][..]));
    }

    #[test]
    fn dotted_keys_stay_flat() {
        let r = recs("@block b\nheat.conduct 56\n").unwrap();
        assert!(r[0].fields.contains("heat.conduct"));
        assert!(!r[0].fields.contains("heat"));
    }
}

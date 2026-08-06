//! The TOML front end: `content/**/*.toml` -> flat `RawRecord`s keyed by DOTTED
//! path.
//!
//! Nesting is deliberately NOT rebuilt here — `[stone]` with `heat.conduct = 56`
//! becomes the literal key `"heat.conduct"`, not a `heat` object — because the
//! schema is what decides which dotted paths are legal groups. Reconstructing
//! objects before validation would let a typo silently invent a group, which is
//! the exact class of bug the compiler exists to prevent.
//!
//! TOML does two of the old lexer's jobs for free and better: a duplicate key is
//! a parse error rather than a silent last-wins clobber, and a key that is both a
//! scalar and a table cannot be spelled at all. What TOML does not give us is a
//! source position for a value, so `LineIndex` below scans the raw text for one.

use crate::error::{Loc, Result};
use crate::{bail, err};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// One key's value, with where it was written.
#[derive(Clone, Debug)]
pub struct RawValue {
    pub value: toml::Value,
    pub loc: Loc,
}

/// Insertion-ordered map from dotted key to its value.
///
/// Order is load-bearing for error reporting determinism — which of several bad
/// keys is reported first must not depend on a hash seed — so this is a `Vec`
/// with linear lookup rather than a `HashMap`. Records carry a few dozen fields
/// at most.
#[derive(Clone, Debug, Default)]
pub struct FieldMap {
    entries: Vec<(String, RawValue)>,
}

impl FieldMap {
    pub fn get(&self, key: &str) -> Option<&RawValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn push(&mut self, key: &str, value: RawValue) {
        self.entries.push((key.to_string(), value));
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct RawRecord {
    /// From the DIRECTORY, not from the file — see FORMAT.md §1.
    pub kind: String,
    pub id: String,
    pub loc: Loc,
    /// dotted key -> value, in file order.
    pub fields: FieldMap,
}

/// `path.relative(root, abs)` — for display only, so a plain strip-prefix with a
/// fallback to the whole path is enough.
fn display_relative(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Parse one content file into records. `kind` comes from the directory.
pub fn parse_file(file: &Path, root: &Path, kind: &str) -> Result<Vec<RawRecord>> {
    let rel: Rc<str> = Rc::from(display_relative(root, file));
    let src =
        std::fs::read_to_string(file).map_err(|e| err!("cannot read {}: {e}", file.display()))?;
    parse_str(&src, &rel, kind)
}

/// The whole front end, with the filesystem left out of it.
pub fn parse_str(src: &str, rel: &Rc<str>, kind: &str) -> Result<Vec<RawRecord>> {
    let table: toml::Table = src.parse().map_err(|e| err!("{rel}: {e}"))?;
    let index = LineIndex::scan(src);

    let mut out = Vec::with_capacity(table.len());
    for (id, value) in &table {
        let loc = Loc::new(rel, index.record(id));
        let toml::Value::Table(body) = value else {
            bail!(&loc, "'{id}' must be a table — one table is one record");
        };
        let mut fields = FieldMap::default();
        flatten("", body, id, rel, &index, &mut fields);

        let rec = RawRecord {
            kind: kind.to_string(),
            id: id.clone(),
            loc,
            fields,
        };
        check_id_agreement(&rec)?;
        check_group_conflicts(&rec)?;
        out.push(rec);
    }
    Ok(out)
}

/// Walk a record's tables into dotted keys. A leaf is anything that is not a
/// table: an array stays whole, because `color`, `tags` and `drop` are all
/// arrays and the schema is what tells them apart.
fn flatten(
    prefix: &str,
    table: &toml::Table,
    id: &str,
    file: &Rc<str>,
    index: &LineIndex,
    out: &mut FieldMap,
) {
    for (k, v) in table {
        let key = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}.{k}")
        };
        match v {
            toml::Value::Table(t) => flatten(&key, t, id, file, index, out),
            _ => {
                let line = index.key(id, &key).unwrap_or_else(|| index.record(id));
                out.push(
                    &key,
                    RawValue {
                        value: v.clone(),
                        loc: Loc::new(file, line),
                    },
                );
            }
        }
    }
}

/// A redundant `id` key must agree with its table name (FORMAT.md §1).
fn check_id_agreement(rec: &RawRecord) -> Result<()> {
    let Some(raw) = rec.fields.get("id") else {
        return Ok(());
    };
    let toml::Value::String(text) = &raw.value else {
        bail!(&raw.loc, "'id' must be a string");
    };
    if *text != rec.id {
        bail!(
            &raw.loc,
            "id '{text}' contradicts the table name '[{}]'",
            rec.id
        );
    }
    Ok(())
}

/// `heat = 4` and `heat.conduct = 5` cannot both be meaningful.
///
/// TOML already refuses to spell this — it is a duplicate-key parse error — so
/// this is a belt on top of braces. It is kept because the rule is about what a
/// record MEANS, not about what the syntax happens to allow, and a future
/// front end that relaxes the syntax must not quietly relax the rule.
fn check_group_conflicts(rec: &RawRecord) -> Result<()> {
    for key in rec.fields.keys() {
        let Some(dot) = key.find('.') else { continue };
        let parent = &key[..dot];
        if let Some(clash) = rec.fields.get(parent) {
            bail!(
                &clash.loc,
                "'{parent}' is assigned both as a value and as a group ('{key}')"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Source positions
// ---------------------------------------------------------------------------

/// Where each key was written, so a schema error can point at a line.
///
/// `toml::Value` carries no spans, and "unknown block field 'hardnes'" without a
/// line number is most of a compiler's value thrown away. This is a scan of the
/// raw text rather than a second parser: a miss just falls back to the record's
/// own header line, so the worst a disagreement with the real parse can do is
/// report a coarser location. It can never change what compiles.
struct LineIndex {
    records: HashMap<String, usize>,
    keys: HashMap<(String, String), usize>,
}

impl LineIndex {
    fn scan(src: &str) -> Self {
        let mut records: HashMap<String, usize> = HashMap::new();
        let mut keys: HashMap<(String, String), usize> = HashMap::new();
        let mut record = String::new();
        let mut prefix = String::new();
        // `'''` bodies are art: a `#` in them is a glyph and a `[` is not a
        // header, so the scan has to know when it is inside one.
        let mut in_literal = false;

        for (i, raw) in src.lines().enumerate() {
            let line = i + 1;
            let opens = raw.matches("'''").count();
            if in_literal {
                if opens % 2 == 1 {
                    in_literal = false;
                }
                continue;
            }
            if opens % 2 == 1 {
                in_literal = true;
            }

            let t = raw.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }

            if let Some(path) = header(t) {
                let (head, rest) = match path.split_once('.') {
                    Some((h, r)) => (h, r),
                    None => (path, ""),
                };
                record = head.to_string();
                prefix = rest.to_string();
                records.entry(record.clone()).or_insert(line);
                if !rest.is_empty() {
                    keys.entry((record.clone(), rest.to_string()))
                        .or_insert(line);
                }
                continue;
            }

            if record.is_empty() {
                continue;
            }
            if let Some(key) = assignment(t) {
                let full = if prefix.is_empty() {
                    key.to_string()
                } else {
                    format!("{prefix}.{key}")
                };
                keys.entry((record.clone(), full)).or_insert(line);
            }
        }

        LineIndex { records, keys }
    }

    fn record(&self, id: &str) -> usize {
        self.records.get(id).copied().unwrap_or(1)
    }

    fn key(&self, id: &str, key: &str) -> Option<usize> {
        self.keys.get(&(id.to_string(), key.to_string())).copied()
    }
}

/// `[a.b]` / `[[a.b]]` -> `a.b`.
fn header(t: &str) -> Option<&str> {
    if let Some(rest) = t.strip_prefix("[[") {
        return rest.split("]]").next().map(str::trim);
    }
    t.strip_prefix('[')?.split(']').next().map(str::trim)
}

/// The key of a `key = value` line, dots included.
fn assignment(t: &str) -> Option<&str> {
    let eq = t.find('=')?;
    let key = t[..eq].trim();
    if key.is_empty() {
        return None;
    }
    let ok = key
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
    ok.then_some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recs(src: &str) -> Result<Vec<RawRecord>> {
        parse_str(src, &Rc::from("t.toml"), "block")
    }

    #[test]
    fn dotted_keys_stay_flat() {
        let r = recs("[b]\nheat.conduct = 56\n").unwrap();
        assert!(r[0].fields.contains("heat.conduct"));
        assert!(!r[0].fields.contains("heat"));
    }

    #[test]
    fn a_sub_table_flattens_the_same_way_as_a_dotted_key() {
        let r = recs("[b]\n[b.heat]\nconduct = 56\n").unwrap();
        assert!(r[0].fields.contains("heat.conduct"));
    }

    #[test]
    fn an_array_of_tables_stays_one_whole_value() {
        let r = recs("[b]\n[[b.seq]]\nstate = \"idle\"\n[[b.seq]]\nstate = \"run\"\n").unwrap();
        let v = &r[0].fields.get("seq").unwrap().value;
        assert_eq!(v.as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn a_redundant_id_must_agree_with_the_table_name() {
        assert!(recs("[stone]\nid = \"stone\"\n").is_ok());
        let e = recs("[stone]\nid = \"granite\"\n").unwrap_err();
        assert!(e.message.contains("contradicts the table name"), "{e}");
    }

    #[test]
    fn a_duplicate_key_is_a_parse_error_rather_than_a_silent_clobber() {
        let e = recs("[b]\nhardness = 1\nhardness = 2\n").unwrap_err();
        assert!(e.message.contains("duplicate"), "{e}");
    }

    #[test]
    fn keys_get_the_line_they_were_written_on() {
        let r = recs("# note\n\n[b]\nname = \"B\"\nheat.conduct = 5\n").unwrap();
        assert_eq!(r[0].loc.line, 3);
        assert_eq!(r[0].fields.get("name").unwrap().loc.line, 4);
        assert_eq!(r[0].fields.get("heat.conduct").unwrap().loc.line, 5);
    }

    #[test]
    fn a_bracket_inside_art_is_not_a_table_header() {
        let r = recs("[b]\nbody = '''\n[not a header]\n'''\nname = \"B\"\n").unwrap();
        assert_eq!(r[0].fields.get("name").unwrap().loc.line, 5);
    }
}

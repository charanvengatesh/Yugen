//! `content/ids.lock.json` — the file that makes a saved world survive a
//! content edit.
//!
//! Ids are assigned once and never reused; a deleted record leaves a tombstone.
//! Reading and writing are hand-rolled rather than pulled from a JSON crate for
//! one reason: the writer has to be byte-stable against the file the TypeScript
//! compiler produced, or the port would rewrite all 224 entries on its first run
//! and the diff would hide whether any code actually moved.

use crate::bail;
use crate::error::Result;
use std::collections::BTreeMap;

/// One kind's ids, ordered by code.
pub type KindLock = Vec<(String, u32)>;

/// Kinds sorted by name; within a kind, ids ordered by code.
#[derive(Clone, Debug, Default)]
pub struct Lock {
    kinds: BTreeMap<String, KindLock>,
}

impl Lock {
    pub fn get(&self, kind: &str, id: &str) -> Option<u32> {
        self.kinds
            .get(kind)?
            .iter()
            .find(|(k, _)| k == id)
            .map(|(_, c)| *c)
    }

    pub fn kind(&self, kind: &str) -> &[(String, u32)] {
        self.kinds.get(kind).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn set_kind(&mut self, kind: &str, entries: KindLock) {
        self.kinds.insert(kind.to_string(), entries);
    }

    pub fn has_kind(&self, kind: &str) -> bool {
        self.kinds.contains_key(kind)
    }

    /// Deterministic serialisation: kinds sorted, ids ordered by code.
    pub fn to_json(&self) -> String {
        let body: Vec<String> = self
            .kinds
            .iter()
            .map(|(k, entries)| {
                let mut sorted = entries.clone();
                sorted.sort_by_key(|(_, c)| *c);
                let rows: Vec<String> = sorted
                    .iter()
                    .map(|(id, c)| format!("    {}: {c}", quote(id)))
                    .collect();
                format!("  {}: {{\n{}\n  }}", quote(k), rows.join(",\n"))
            })
            .collect();
        format!("{{\n{}\n}}\n", body.join(",\n"))
    }

    /// Parse the two-level `{kind: {id: code}}` shape. Deliberately narrow: this
    /// reads exactly the file it writes, and anything else is a corrupt lock.
    pub fn from_json(src: &str) -> Result<Lock> {
        let mut lock = Lock::default();
        let b: Vec<char> = src.chars().collect();
        let mut i = 0usize;

        skip_ws(&b, &mut i);
        expect(&b, &mut i, '{')?;
        skip_ws(&b, &mut i);
        if peek(&b, i) == Some('}') {
            return Ok(lock);
        }

        loop {
            skip_ws(&b, &mut i);
            let kind = parse_string(&b, &mut i)?;
            skip_ws(&b, &mut i);
            expect(&b, &mut i, ':')?;
            skip_ws(&b, &mut i);
            expect(&b, &mut i, '{')?;

            let mut entries: KindLock = Vec::new();
            skip_ws(&b, &mut i);
            if peek(&b, i) == Some('}') {
                i += 1;
            } else {
                loop {
                    skip_ws(&b, &mut i);
                    let id = parse_string(&b, &mut i)?;
                    skip_ws(&b, &mut i);
                    expect(&b, &mut i, ':')?;
                    skip_ws(&b, &mut i);
                    let code = parse_u32(&b, &mut i)?;
                    entries.push((id, code));
                    skip_ws(&b, &mut i);
                    match peek(&b, i) {
                        Some(',') => i += 1,
                        Some('}') => {
                            i += 1;
                            break;
                        }
                        other => bail!("ids.lock.json: expected ',' or '}}', got {other:?}"),
                    }
                }
            }
            lock.kinds.insert(kind, entries);

            skip_ws(&b, &mut i);
            match peek(&b, i) {
                Some(',') => i += 1,
                Some('}') => break,
                other => bail!("ids.lock.json: expected ',' or '}}', got {other:?}"),
            }
        }
        Ok(lock)
    }
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn peek(b: &[char], i: usize) -> Option<char> {
    b.get(i).copied()
}

fn skip_ws(b: &[char], i: &mut usize) {
    while b.get(*i).is_some_and(|c| c.is_whitespace()) {
        *i += 1;
    }
}

fn expect(b: &[char], i: &mut usize, c: char) -> Result<()> {
    if peek(b, *i) != Some(c) {
        bail!("ids.lock.json: expected '{c}' at offset {i}");
    }
    *i += 1;
    Ok(())
}

fn parse_string(b: &[char], i: &mut usize) -> Result<String> {
    expect(b, i, '"')?;
    let mut out = String::new();
    while let Some(c) = peek(b, *i) {
        *i += 1;
        match c {
            '"' => return Ok(out),
            '\\' => {
                let Some(e) = peek(b, *i) else { break };
                *i += 1;
                out.push(match e {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    other => other,
                });
            }
            c => out.push(c),
        }
    }
    bail!("ids.lock.json: unterminated string")
}

fn parse_u32(b: &[char], i: &mut usize) -> Result<u32> {
    let start = *i;
    while b.get(*i).is_some_and(char::is_ascii_digit) {
        *i += 1;
    }
    if start == *i {
        bail!("ids.lock.json: expected a number at offset {start}");
    }
    let s: String = b[start..*i].iter().collect();
    s.parse::<u32>()
        .map_err(|e| crate::error::ContentError::new(format!("ids.lock.json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_byte_for_byte() {
        let src = "{\n  \"block\": {\n    \"air\": 0,\n    \"stone\": 1\n  },\n  \"item\": {\n    \"rock\": 0\n  }\n}\n";
        let lock = Lock::from_json(src).unwrap();
        assert_eq!(lock.get("block", "stone"), Some(1));
        assert_eq!(lock.get("item", "rock"), Some(0));
        assert_eq!(lock.to_json(), src);
    }

    #[test]
    fn an_empty_lock_is_legal() {
        let lock = Lock::from_json("{}").unwrap();
        assert!(!lock.has_kind("block"));
    }
}

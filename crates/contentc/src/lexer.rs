//! Line lexer for the content DSL (see `content/FORMAT.md` §2).
//!
//! The format is deliberately line-oriented: one key per line, no nesting
//! punctuation, no quoting unless a value would otherwise be ambiguous. That
//! makes a `.block` file diff cleanly and lets the lexer stay a single pass with
//! one piece of lookahead (heredocs). Nothing here knows what a `block` is —
//! the lexer only produces structural tokens; meaning is applied by the schema.
//!
//! The TS original leaned on four regexes. They are hand-written matchers here
//! rather than a `regex` dependency: the grammar is small enough that the
//! matchers are shorter than the patterns they replace, and `godgame-data` is
//! the only crate downstream of this one that ships in the game.

use crate::bail;
use crate::error::{Loc, Result};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Token {
    /// `@<kind> <id>` — opens a record.
    Record { kind: String, id: String, loc: Loc },
    /// `<key> <value>` — value is the raw rest-of-line, trimmed and unquoted.
    Key {
        key: String,
        value: String,
        indent: usize,
        loc: Loc,
    },
    /// `<key> [inline] |` followed by an indented verbatim block.
    ///
    /// `value` is the text before the trailing `|`, trimmed — empty for the bare
    /// `key |` form, and the element's `key=value` attributes for a `record[]`
    /// occurrence that carries both attributes and a body
    /// (`art.seq state=run mode=phase |`).
    Text {
        key: String,
        value: String,
        lines: Vec<String>,
        indent: usize,
        loc: Loc,
    },
    /// `+include <path>` at column 0.
    Include { path: String, loc: Loc },
}

impl Token {
    pub fn loc(&self) -> &Loc {
        match self {
            Token::Record { loc, .. }
            | Token::Key { loc, .. }
            | Token::Text { loc, .. }
            | Token::Include { loc, .. } => loc,
        }
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Key names may contain dots: `heat.conduct`. The dot is what makes a group.
fn is_key_char(c: char) -> bool {
    is_ident_char(c) || c == '.'
}

/// `#a0f` / `#49b077` — a colour literal, not the start of a comment.
///
/// Mirrors `/^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})(?![0-9a-fA-F])/`. The negative
/// lookahead is what the exact run-length test below reproduces: a run of 4, 5
/// or 7+ hex digits is not a colour, so `# 12345` stays a comment.
fn is_hex_color(s: &str) -> bool {
    let Some(rest) = s.strip_prefix('#') else {
        return false;
    };
    let n = rest.chars().take_while(char::is_ascii_hexdigit).count();
    n == 3 || n == 6
}

/// Strip a trailing `#` comment. A `#` only opens a comment when it is the first
/// non-space character or is preceded by whitespace, and is not inside a quoted
/// string.
///
/// The one refinement over that rule: a whitespace-preceded `#` that is followed
/// by a well-formed hex colour is data. FORMAT.md §2 lists both `# comment` and
/// `#rrggbb` as valid, so something has to break the tie, and a colour literal
/// is the only `#`-leading value the format has. Anything else with a leading
/// `#` (`name "Rank #1"`) has to be quoted. Heredoc bodies never reach this
/// function — they are verbatim, which is what makes `#` usable as sprite and
/// legend art.
pub fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    let first = line.find(|c: char| !c.is_whitespace());
    let mut it = line.char_indices();

    while let Some((i, c)) = it.next() {
        if c == '\\' && quoted {
            it.next(); // escaped char inside a string, skip it
            continue;
        }
        if c == '"' {
            quoted = !quoted;
            continue;
        }
        if c != '#' || quoted {
            continue;
        }
        if Some(i) == first {
            return ""; // a whole-line comment, whatever follows the '#'
        }
        if !line[..i]
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
        {
            continue;
        }
        if is_hex_color(&line[i..]) {
            continue;
        }
        return &line[..i];
    }
    line
}

/// Count leading spaces, counting a tab as one column (indent is only compared).
fn indent_of(line: &str) -> usize {
    line.chars().take_while(|&c| c == ' ' || c == '\t').count()
}

/// Remove one layer of surrounding double quotes, honouring `\"` escapes.
fn unquote(v: &str) -> String {
    let bytes = v.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"' {
        let inner = &v[1..v.len() - 1];
        let mut out = String::with_capacity(inner.len());
        let mut it = inner.chars();
        while let Some(c) = it.next() {
            if c == '\\' {
                // A lone trailing backslash has nothing to escape and survives
                // verbatim, matching `/\\(.)/g` needing a following character.
                match it.next() {
                    Some(n) => out.push(n),
                    None => out.push('\\'),
                }
            } else {
                out.push(c);
            }
        }
        out
    } else {
        v.to_string()
    }
}

/// `^@([A-Za-z_][A-Za-z0-9_]*)(?:\s+(\S+))?\s*$` against an already-trimmed line.
///
/// Returns `None` for a malformed header AND for a header with no id — the
/// caller reports both the same way, exactly as the TS `!m || !m[2]` test did.
fn match_record(body: &str) -> Option<(String, String)> {
    let rest = body.strip_prefix('@')?;
    let mut chars = rest.char_indices();
    let (_, c0) = chars.next()?;
    if !is_ident_start(c0) {
        return None;
    }
    let end = rest
        .char_indices()
        .skip(1)
        .find(|&(_, c)| !is_ident_char(c))
        .map_or(rest.len(), |(i, _)| i);
    let kind = &rest[..end];
    let tail = &rest[end..];

    // No id at all: `@block`. The optional group did not participate, so the
    // pattern's trailing `\s*$` has to swallow the rest — and then there is no
    // id to return.
    if tail.trim().is_empty() {
        return None;
    }
    if !tail.starts_with([' ', '\t']) {
        return None;
    }
    let id_part = tail.trim_start();
    // `(\S+)\s*$` — exactly one more token, then only whitespace.
    let id_end = id_part.find(char::is_whitespace).unwrap_or(id_part.len());
    if !id_part[id_end..].trim().is_empty() {
        return None; // `@block stone extra` — a third token is a malformed header
    }
    Some((kind.to_string(), id_part[..id_end].to_string()))
}

/// `^([A-Za-z_][A-Za-z0-9_.]*)(?:\s+([\s\S]*))?$` against an already-trimmed line.
fn match_key(body: &str) -> Option<(&str, &str)> {
    let mut chars = body.chars();
    if !is_ident_start(chars.next()?) {
        return None;
    }
    let end = body
        .char_indices()
        .skip(1)
        .find(|&(_, c)| !is_key_char(c))
        .map_or(body.len(), |(i, _)| i);
    let key = &body[..end];
    let tail = &body[end..];
    if tail.is_empty() {
        return Some((key, ""));
    }
    // The value must be separated from the key by whitespace. `foo=bar` stops
    // the key at `foo` and then fails here, which is what makes `=` unusable as
    // a top-level separator and keeps it available for `record[]` attributes.
    if !tail.starts_with([' ', '\t']) {
        return None;
    }
    Some((key, tail.trim()))
}

/// A trailing `|` opens a heredoc — but only when it stands alone as a token,
/// that is, it IS the whole value or is preceded by whitespace.
///
/// `|` is already load-bearing inside values: a `ref?` preference chain writes
/// `mat.accent ironOre|coalOre` and a struct legend writes
/// `block=goldOre|ironOre|coalOre`. Requiring the separator keeps the two
/// unambiguous with no lookahead and no escaping — `a|b|` is a chain whose last
/// candidate happens to be empty (the schema's problem, not the lexer's),
/// `a|b |` is a chain followed by a heredoc opener. Authors never have to think
/// about it because the whitespace is what they would type anyway.
fn is_heredoc(value: &str) -> bool {
    let Some(before) = value.strip_suffix('|') else {
        return false;
    };
    before.is_empty() || before.chars().next_back().is_some_and(char::is_whitespace)
}

/// Split like `/\r?\n/`.
fn split_lines(src: &str) -> Vec<&str> {
    src.split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect()
}

/// Tokenize one file. `file` is used only for error messages; includes are NOT
/// resolved here (the parser owns the filesystem), they surface as tokens.
pub fn lex(src: &str, file: &Rc<str>) -> Result<Vec<Token>> {
    let raw = split_lines(src);
    let mut out: Vec<Token> = Vec::new();

    let mut i = 0usize;
    while i < raw.len() {
        let raw_line = raw[i];
        let loc = Loc::new(file, i + 1);

        // `+include` is structural and must sit at column 0, so it is impossible
        // to mistake for a key inside a record.
        if raw_line.starts_with("+include") {
            let path = strip_comment(raw_line)["+include".len()..].trim();
            if path.is_empty() {
                bail!(&loc, "+include needs a path");
            }
            out.push(Token::Include {
                path: unquote(path),
                loc,
            });
            i += 1;
            continue;
        }

        let line = strip_comment(raw_line);
        let body = line.trim();
        if body.is_empty() {
            i += 1;
            continue; // blank or comment-only
        }

        if body.starts_with('@') {
            let Some((kind, id)) = match_record(body) else {
                bail!(&loc, "bad record header: {body}");
            };
            out.push(Token::Record { kind, id, loc });
            i += 1;
            continue;
        }

        let Some((key, value)) = match_key(body) else {
            bail!(&loc, "bad key line: {body}");
        };
        let indent = indent_of(line);

        // Heredoc: a standalone trailing `|` swallows every following line
        // indented deeper than the key line, verbatim (comments included —
        // sprite art may legitimately contain `#`). Anything before the `|`
        // stays with the token as its inline value, which is what lets one
        // `record[]` occurrence carry both its `key=value` attributes and its
        // body. A trailing `# comment` after the `|` has already been stripped
        // above, so it does not defeat the match.
        if is_heredoc(value) {
            let mut block: Vec<&str> = Vec::new();
            let mut j = i + 1;
            while j < raw.len() {
                let l = raw[j];
                if l.trim().is_empty() {
                    // A blank line is kept, not a terminator: it is the frame
                    // separator in sprite art.
                    block.push("");
                    j += 1;
                    continue;
                }
                if indent_of(l) <= indent {
                    break;
                }
                block.push(l);
                j += 1;
            }
            while block.last().is_some_and(|l| l.trim().is_empty()) {
                block.pop();
            }
            let min = block
                .iter()
                .filter(|l| !l.trim().is_empty())
                .map(|l| indent_of(l))
                .min()
                .unwrap_or(0);
            // Trailing whitespace is preserved: in sprite and structure art a
            // trailing run of glyphs-that-happen-to-be-spaces is data, not
            // formatting.
            let lines: Vec<String> = block
                .iter()
                .map(|l| {
                    if l.trim().is_empty() {
                        String::new()
                    } else {
                        l[min..].to_string()
                    }
                })
                .collect();

            // NOT unquoted: the inline segment is an attribute list, and
            // `split_pairs` in the schema does its own per-value quote handling.
            // Stripping a layer here would only ever damage a value it does not
            // understand.
            let inline = value[..value.len() - 1].trim().to_string();
            out.push(Token::Text {
                key: key.to_string(),
                value: inline,
                lines,
                indent,
                loc,
            });
            i = j;
            continue;
        }

        out.push(Token::Key {
            key: key.to_string(),
            value: unquote(value),
            indent,
            loc,
        });
        i += 1;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f() -> Rc<str> {
        Rc::from("t.block")
    }

    #[test]
    fn hex_colour_is_data_but_a_short_run_is_a_comment() {
        assert_eq!(strip_comment("color #49b077"), "color #49b077");
        assert_eq!(strip_comment("color #a0f"), "color #a0f");
        assert_eq!(strip_comment("color 1 2 3 # a note"), "color 1 2 3 ");
        // 5 hex digits is neither a 3- nor a 6-digit colour.
        assert_eq!(strip_comment("x #12345"), "x ");
        // First non-space `#` kills the whole line, colour or not.
        assert_eq!(strip_comment("   #a0f"), "");
        // Inside quotes a `#` is never a comment.
        assert_eq!(strip_comment(r#"name "Rank #1""#), r#"name "Rank #1""#);
    }

    #[test]
    fn heredoc_disambiguates_against_preference_chains() {
        assert!(!is_heredoc("ironOre|coalOre"));
        assert!(!is_heredoc("a|b|"));
        assert!(is_heredoc("a|b |"));
        assert!(is_heredoc("|"));
        assert!(is_heredoc("state=run mode=phase |"));
    }

    #[test]
    fn record_header_needs_exactly_one_id() {
        assert_eq!(
            match_record("@block stone"),
            Some(("block".into(), "stone".into()))
        );
        assert_eq!(match_record("@block"), None);
        assert_eq!(match_record("@block stone extra"), None);
        assert_eq!(match_record("@1bad id"), None);
    }

    #[test]
    fn key_line_requires_whitespace_before_the_value() {
        assert_eq!(match_key("heat.conduct 56"), Some(("heat.conduct", "56")));
        assert_eq!(match_key("solid"), Some(("solid", "")));
        assert_eq!(match_key("foo=bar"), None);
    }

    #[test]
    fn heredoc_body_is_dedented_and_keeps_blank_separators() {
        let src = "@sprite s\nframes |\n  .4\n  32\n\n  1.\n\nnext 1\n";
        let toks = lex(src, &f()).unwrap();
        let Token::Text { lines, value, .. } = &toks[1] else {
            panic!("expected text")
        };
        assert_eq!(value, "");
        assert_eq!(lines, &["\u{2e}4", "32", "", "1."]);
        // The trailing blank was popped, so the following key is a sibling.
        assert!(matches!(&toks[2], Token::Key { key, .. } if key == "next"));
    }

    #[test]
    fn heredoc_preserves_trailing_whitespace_as_art() {
        let src = "@struct s\nbody |\n  ab  \n  cd\n";
        let toks = lex(src, &f()).unwrap();
        let Token::Text { lines, .. } = &toks[1] else {
            panic!("expected text")
        };
        assert_eq!(lines, &["ab  ", "cd"]);
    }

    #[test]
    fn unquoting_happens_on_keys_but_not_on_heredoc_inline_values() {
        let toks = lex("@block b\nname \"Rank \\\"1\\\"\"\n", &f()).unwrap();
        let Token::Key { value, .. } = &toks[1] else {
            panic!("expected key")
        };
        assert_eq!(value, r#"Rank "1""#);
    }
}

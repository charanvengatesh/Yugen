//! A hand-rolled JSON writer, so a state dump costs no dependency.
//!
//! `godgame-core` depends on `godgame-data`, `bitflags` and `rayon`, and that
//! list is deliberately short: this is the crate the purity suite and the
//! benches link, so anything it pulls in is pulled into all of them.
//! `serde_json` is a DEV-dependency — it exists for the golden fixtures and the
//! tests below, and it is not in the shipping build. A `--dump-state out.json`
//! developer flag has to emit JSON from that shipping build, so it needs a
//! writer that adds nothing to the graph.
//!
//! `src/bin/worldgen-dump.rs` already made this argument for images: a PNG with
//! no compression is a fixed header, a zlib stream of STORED deflate blocks and
//! two checksums, which is cheaper than an image crate and cannot rot. Writing
//! JSON is the smaller version of the same case. The grammar is six value forms
//! and a comma rule; the only part with any teeth is string escaping, and that
//! is one `match` over a handful of code points. There is no schema to track, no
//! derive macro to keep in step with a struct, and nothing to break when the
//! ecosystem moves — which is the real saving, because a debug flag is exactly
//! the code nobody maintains.
//!
//! What this is NOT: a parser, a serialisation framework, or a general
//! `Serialize` replacement. It writes one document, forwards, once. Anything
//! that needs to read JSON back is test code and can keep using `serde_json`.

use std::fmt::Write as _;

/// A JSON document under construction.
///
/// The caller drives it with nested closures — [`object`](Json::object) and
/// [`array`](Json::array) open a container and hand back a builder scoped to it
/// — and the builder owns every comma, newline and indent. That shape is chosen
/// so a caller cannot emit invalid JSON by forgetting a separator: there is no
/// method that writes a comma, and the close brace is tied to the end of the
/// closure rather than to a call the caller might not make.
///
/// Output is pretty-printed with a two-space indent and ends in a newline. A
/// dump nobody reads is a dump nobody checks, and a dump on one line diffs
/// terribly.
pub struct Json {
    out: String,
    /// Open container count, and therefore the indent level.
    depth: usize,
    /// Does the container we are inside still hold nothing? Drives the comma.
    empty: bool,
    /// Has a key been written whose value is still owed? A value that answers a
    /// key must not indent itself — it belongs on the key's line.
    keyed: bool,
    /// Is the innermost open container an object? Saved and restored around
    /// nesting, so a bare value inside an object can be caught.
    in_object: bool,
}

impl Default for Json {
    fn default() -> Json {
        Json::new()
    }
}

impl Json {
    /// An empty document, ready for exactly one root value.
    pub fn new() -> Json {
        Json {
            out: String::new(),
            depth: 0,
            empty: true,
            keyed: false,
            in_object: false,
        }
    }

    /// Write an object; `f` populates it with [`field`](Json::field) calls.
    ///
    /// An object with no fields comes out as `{}` on one line rather than a
    /// brace pair straddling a blank line. The same goes for `[]`.
    pub fn object(&mut self, f: impl FnOnce(&mut Json)) {
        self.begin_value();
        self.out.push('{');
        self.enter(true, f);
        self.out.push('}');
    }

    /// Write an array; every value `f` emits becomes an element.
    pub fn array(&mut self, f: impl FnOnce(&mut Json)) {
        self.begin_value();
        self.out.push('[');
        self.enter(false, f);
        self.out.push(']');
    }

    /// Write one `"key": value` pair. Only legal inside [`object`](Json::object),
    /// and `f` must write exactly one value.
    ///
    /// Keys go through the same escaping as any other string, because a key in a
    /// state dump can be a material or item name that came out of `content/` and
    /// is therefore authored text rather than an identifier.
    pub fn field(&mut self, key: &str, f: impl FnOnce(&mut Json)) {
        debug_assert!(self.in_object, "field() called outside an object");
        debug_assert!(!self.keyed, "field() called while a value was still owed");
        self.separate();
        escape_into(&mut self.out, key);
        self.out.push_str(": ");
        self.keyed = true;
        f(self);
        debug_assert!(!self.keyed, "field(\"{key}\") wrote no value");
    }

    /// Write a string, escaped.
    pub fn str(&mut self, v: &str) {
        self.begin_value();
        escape_into(&mut self.out, v);
    }

    /// Write a signed integer.
    pub fn int(&mut self, v: i64) {
        self.begin_value();
        let _ = write!(self.out, "{v}");
    }

    /// Write an unsigned integer.
    ///
    /// Separate from [`int`](Json::int) rather than folded into it because cell
    /// counts, tick numbers and seeds are `u64` and the top of that range does
    /// not survive a cast to `i64`. JSON's grammar has one number type, so the
    /// split costs nothing in the output.
    pub fn uint(&mut self, v: u64) {
        self.begin_value();
        let _ = write!(self.out, "{v}");
    }

    /// Write a float — or `null`, if it is not finite.
    ///
    /// JSON cannot spell an infinity or a NaN, so every writer has to pick a
    /// lie. `crates/godgame-data/tests/registry.golden.json` picks the strings
    /// `"inf"` / `"-inf"`, and that is right for a FIXTURE: the baseline exists
    /// to prove a value matches the TypeScript exactly, so an infinity that came
    /// out of the registry must survive the file and come back distinguishable
    /// from every finite number. It has one reader, and that reader knows the
    /// convention.
    ///
    /// A state dump has the opposite audience. It is read by `jq`, by a diff, by
    /// whatever throwaway script someone points at it this afternoon, and all of
    /// those expect a number-shaped field to hold a number or `null`. A sentinel
    /// string turns a numeric column into a mixed one and breaks the tool
    /// several steps downstream of the bug. `null` is the value JSON already has
    /// for "there is nothing here", and a non-finite float in a simulation state
    /// IS the absence of a usable number — it means something upstream divided
    /// by zero, and the dump should say so rather than dress it up. Note that
    /// this is lossy: `+inf`, `-inf` and NaN are not told apart afterwards. That
    /// is acceptable for a debug dump and would not be for the fixture.
    ///
    /// Finite values are written with Rust's shortest round-tripping form, which
    /// keeps the `.0` on whole numbers so a float field never reads as an
    /// integer, and uses exponent notation instead of spelling `1e300` out in
    /// three hundred digits.
    pub fn float(&mut self, v: f64) {
        self.begin_value();
        if v.is_finite() {
            let _ = write!(self.out, "{v:?}");
        } else {
            self.out.push_str("null");
        }
    }

    /// Write `true` or `false`.
    pub fn bool(&mut self, v: bool) {
        self.begin_value();
        self.out.push_str(if v { "true" } else { "false" });
    }

    /// Write `null`.
    pub fn null(&mut self) {
        self.begin_value();
        self.out.push_str("null");
    }

    /// Finish the document and take the text, newline-terminated.
    ///
    /// Consuming `self` is the point: it makes "wrote the root value, then wrote
    /// another one" unrepresentable, and the trailing newline means the file
    /// ends the way every other text file in the tree does, so a diff does not
    /// open with a no-newline marker.
    pub fn finish(mut self) -> String {
        debug_assert_eq!(self.depth, 0, "finish() with a container still open");
        debug_assert!(!self.keyed, "finish() with a field value still owed");
        self.out.push('\n');
        self.out
    }

    /// Run a container's body one level deeper, restoring the enclosing
    /// container's state afterwards.
    fn enter(&mut self, is_object: bool, f: impl FnOnce(&mut Json)) {
        let outer_empty = std::mem::replace(&mut self.empty, true);
        let outer_object = std::mem::replace(&mut self.in_object, is_object);
        self.depth += 1;
        f(self);
        self.depth -= 1;
        if !self.empty {
            self.newline_indent();
        }
        self.empty = outer_empty;
        self.in_object = outer_object;
    }

    /// Position for a value: answer an owed key in place, otherwise open a new
    /// line in the enclosing container.
    fn begin_value(&mut self) {
        if self.keyed {
            self.keyed = false;
            return;
        }
        debug_assert!(
            !self.in_object,
            "a bare value inside an object — use field()"
        );
        if self.depth > 0 {
            self.separate();
        }
    }

    /// Comma if something came before, then a fresh indented line.
    fn separate(&mut self) {
        if !self.empty {
            self.out.push(',');
        }
        self.empty = false;
        self.newline_indent();
    }

    fn newline_indent(&mut self) {
        self.out.push('\n');
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
    }
}

/// Quote and escape a string into `out`.
///
/// The C0 range has to be escaped because the strings in a state dump are
/// authored text — material names, item names, whatever a `content/` TOML file
/// happened to contain — and one stray tab or newline in the middle of a name
/// produces a file that no parser will accept. `"` and `\` are the two
/// characters that would otherwise end or re-interpret the literal. The rest of
/// C0 has no short escape and goes out as `\u00XX`.
///
/// Non-ASCII is emitted as UTF-8 and NOT escaped. JSON is a UTF-8 format, `\u`
/// escaping every accented character would triple the size of the strings it
/// touches for no gain, and surrogate pairs are a whole class of bug that simply
/// does not arise if the writer never forms one.
fn escape_into(out: &mut String, s: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if c < ' ' => {
                let b = c as usize;
                out.push_str("\\u00");
                out.push(char::from(HEX[(b >> 4) & 0xf]));
                out.push(char::from(HEX[b & 0xf]));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The document every round-trip test below builds, so the builder is
    /// exercised through one non-trivial nesting rather than several toy ones.
    fn a_nested_document(j: &mut Json) {
        j.object(|j| {
            j.field("seed", |j| j.uint(1337));
            j.field("origin", |j| {
                j.array(|j| {
                    j.int(-8);
                    j.int(26);
                })
            });
            j.field("player", |j| {
                j.object(|j| {
                    j.field("name", |j| j.str("digger"));
                    j.field("health", |j| j.float(0.5));
                    j.field("grounded", |j| j.bool(true));
                    j.field("holding", |j| j.null());
                })
            });
            j.field("chunks", |j| {
                j.array(|j| {
                    j.object(|j| j.field("x", |j| j.int(0)));
                    j.object(|j| j.field("x", |j| j.int(1)));
                })
            });
        });
    }

    #[test]
    fn a_nested_document_parses_back_to_the_values_put_in() {
        let mut j = Json::new();
        a_nested_document(&mut j);
        let text = j.finish();

        let v: serde_json::Value = serde_json::from_str(&text).expect("output must parse");
        assert_eq!(v["seed"], 1337);
        assert_eq!(v["origin"][0], -8);
        assert_eq!(v["origin"][1], 26);
        assert_eq!(v["player"]["name"], "digger");
        assert_eq!(v["player"]["health"], 0.5);
        assert_eq!(v["player"]["grounded"], true);
        assert!(v["player"]["holding"].is_null());
        assert_eq!(v["chunks"].as_array().expect("an array").len(), 2);
        assert_eq!(v["chunks"][1]["x"], 1);
    }

    #[test]
    fn every_escape_case_survives_a_round_trip() {
        let awkward = "quote \" backslash \\ tab \t newline \n return \r backspace \u{08} \
                       formfeed \u{0c} slash / end";
        let mut j = Json::new();
        j.str(awkward);
        let text = j.finish();

        // The short escapes are used where they exist; a solidus is legal
        // unescaped and staying out of its way keeps the output readable.
        assert!(text.contains(r#"quote \" backslash \\"#), "{text}");
        assert!(text.contains(r"tab \t newline \n return \r"), "{text}");
        assert!(text.contains(r"backspace \b formfeed \f"), "{text}");
        assert!(text.contains("slash / end"), "{text}");

        let v: serde_json::Value = serde_json::from_str(&text).expect("output must parse");
        assert_eq!(v, serde_json::Value::String(awkward.to_string()));
    }

    #[test]
    fn a_control_character_survives_a_round_trip() {
        // 0x01 and 0x1f have no short escape, so they must take the \u00XX path.
        let raw = "before\u{01}middle\u{1f}after";
        let mut j = Json::new();
        j.object(|j| j.field("name", |j| j.str(raw)));
        let text = j.finish();

        assert!(text.contains("before\\u0001middle\\u001fafter"), "{text}");
        let v: serde_json::Value = serde_json::from_str(&text).expect("output must parse");
        assert_eq!(v["name"], raw);
    }

    #[test]
    fn non_ascii_text_is_emitted_as_utf8_rather_than_escaped() {
        let raw = "coloured lavå — 溶岩";
        let mut j = Json::new();
        j.str(raw);
        let text = j.finish();

        assert!(text.contains(raw), "non-ASCII must go out verbatim: {text}");
        assert!(
            !text.contains("\\u"),
            "nothing above ASCII gets escaped: {text}"
        );
        let v: serde_json::Value = serde_json::from_str(&text).expect("output must parse");
        assert_eq!(v, serde_json::Value::String(raw.to_string()));
    }

    #[test]
    fn a_non_finite_float_is_written_as_null() {
        let mut j = Json::new();
        j.array(|j| {
            j.float(f64::NAN);
            j.float(f64::INFINITY);
            j.float(f64::NEG_INFINITY);
            j.float(-0.0);
            j.float(1.5e300);
        });
        let text = j.finish();

        let v: serde_json::Value = serde_json::from_str(&text).expect("output must parse");
        assert!(v[0].is_null(), "NaN is null, not a sentinel string");
        assert!(v[1].is_null(), "+inf is null");
        assert!(v[2].is_null(), "-inf is null");
        assert_eq!(v[3].as_f64(), Some(-0.0), "a finite value is untouched");
        assert_eq!(v[4].as_f64(), Some(1.5e300));
        // The whole point of choosing null over "inf": no string leaks into a
        // column of numbers.
        assert!(!text.contains("inf"), "{text}");
        assert!(!text.contains("NaN"), "{text}");
    }

    #[test]
    fn a_whole_float_keeps_its_decimal_point() {
        let mut j = Json::new();
        j.float(3.0);
        assert_eq!(j.finish(), "3.0\n");
    }

    #[test]
    fn an_empty_object_and_an_empty_array_stay_on_one_line() {
        let mut j = Json::new();
        j.object(|j| {
            j.field("nothing", |j| j.object(|_| {}));
            j.field("nobody", |j| j.array(|_| {}));
        });
        assert_eq!(j.finish(), "{\n  \"nothing\": {},\n  \"nobody\": []\n}\n");

        let mut j = Json::new();
        j.object(|_| {});
        assert_eq!(j.finish(), "{}\n");
    }

    #[test]
    fn nesting_is_indented_two_spaces_per_level_and_ends_with_a_newline() {
        let mut j = Json::new();
        j.object(|j| {
            j.field("a", |j| {
                j.array(|j| {
                    j.int(1);
                    j.object(|j| j.field("b", |j| j.bool(false)));
                })
            });
        });
        let text = j.finish();

        assert_eq!(
            text,
            "{\n  \"a\": [\n    1,\n    {\n      \"b\": false\n    }\n  ]\n}\n"
        );
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn the_same_document_built_twice_is_byte_identical() {
        // The flag this feeds exists to diff two runs, so any variation in the
        // writer itself would make every diff useless.
        let mut a = Json::new();
        a_nested_document(&mut a);
        let mut b = Json::new();
        a_nested_document(&mut b);
        assert_eq!(a.finish().as_bytes(), b.finish().as_bytes());
    }
}

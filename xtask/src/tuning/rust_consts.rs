//! Reading constants back out of Rust source.
//!
//! # Why this parses instead of pattern-matching
//!
//! The TypeScript original did this with one regex anchored at column zero, and
//! it worked because TypeScript declares a constant one way. Rust does not: a
//! declaration wraps across lines when rustfmt decides it should, the value is
//! an arbitrary expression, and — the part that actually settles it — this tree
//! generates hundreds of `const` items from inside `bitflags!`. A regex sees
//! those as declarations and floods the index with `const DIGGABLE`, `const ORE`
//! and eighty other bit names that are identities, not tuned numbers.
//!
//! [`syn`] does not expand macros, so a macro body is one opaque token blob and
//! every one of those disappears for free, correctly, rather than by a
//! hand-maintained deny-list. The same parse gives the declared TYPE, which is
//! the honest way to answer "is this a number" — the RHS `CELL_SIZE * 4` says
//! nothing on its own, and `Rgb(20, 30, 40)` looks numeric and is not.
//!
//! # What it still cannot see
//!
//! Stated plainly, because an index that quietly omits things is worse than one
//! with known edges:
//!
//! - **Anything inside a macro invocation.** Deliberate for `bitflags!`, but it
//!   is a blanket rule: a tuned number declared inside any macro body is
//!   invisible here.
//! - **Constants declared inside a function body.** Only module-level items are
//!   collected, matching the original's column-zero anchor. A `const` inside an
//!   `fn` is scoped to that call and cannot be shadowed by a config import, so
//!   it is out of scope for both rules this enforces.
//! - **Associated constants in an `impl` block.** Same reason: `Foo::BAR` is
//!   reached through a type and never collides with a glob-imported name.
//! - **`cfg`-gated code.** Everything is read, whatever the cfg. A number behind
//!   `#[cfg(target_os = "windows")]` is indexed on macOS. That is the wanted
//!   behaviour for an index — it is a document, not a build.
//! - **The value as written, not as computed.** `PLAYER_W` indexes as
//!   `(PLAYER_CELLS_W * CELL_SIZE) as f32`, not `10`. Evaluating it would mean
//!   implementing const-eval; the expression is also more useful, because it
//!   shows what the number is derived from.

use proc_macro2::LineColumn;
use syn::spanned::Spanned;

/// One module-level constant, as the index reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Konst {
    pub name: String,
    /// The right-hand side as the author wrote it, with runs of whitespace
    /// collapsed so a wrapped declaration still fits one table cell.
    pub value: String,
    /// 1-based line of the declaration's name.
    pub line: usize,
    /// The one-line meaning, or empty. See [`doc_of`].
    pub doc: String,
    /// `pub`, i.e. part of the crate's surface. Tier 2 documents these and only
    /// these — a private helper in `config/` is an implementation detail of the
    /// module, not a knob a designer is being handed.
    pub exported: bool,
    /// The declared type is a numeric scalar, so this is a candidate knob rather
    /// than a table, a colour, an id or a string.
    pub tuned: bool,
}

/// The scalar types a tuned number can have.
///
/// A whitelist and not "anything that is not a container" because the tree is
/// full of newtypes over integers — `CellId`, `ItemCode`, `Biome` — that are
/// identities. Naming a `CellId` in the tuning index invites someone to turn it,
/// and the number it holds means nothing on its own.
const NUMERIC: &[&str] = &[
    "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
    "usize",
];

/// Every module-level `const` and `static` in one file, in source order.
pub fn scan(source: &str) -> Result<Vec<Konst>, syn::Error> {
    let file = syn::parse_file(source)?;
    let src = Source::new(source);
    let mut raw = Vec::new();
    collect(&file.items, &src, &mut raw);
    raw.sort_by_key(|r| r.konst.line);
    Ok(carry_docs(raw))
}

/// A `pub` item in `config/` with no doc comment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Undocumented {
    /// `const`, `fn`, `struct`, `field`, ... — for the error message.
    pub kind: &'static str,
    pub name: String,
    pub line: usize,
}

/// Every `pub` item in one file that carries no `///`.
///
/// Broader than the original's rule, which asked only about exported constants,
/// because this tree's own `CLAUDE.md` widened it to every `pub` item under
/// `config/` and a helper function with an unexplained return is exactly as
/// unturnable as an unexplained number.
///
/// `pub mod foo;` is exempt: the module is documented by the `//!` header of its
/// own file, which is not visible from the declaration site. An inline
/// `pub mod foo { .. }` has no such file and is not exempt.
pub fn undocumented_pub_items(source: &str) -> Result<Vec<Undocumented>, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut out = Vec::new();
    check_docs(&file.items, &mut out);
    out.sort_by_key(|u| u.line);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Collecting constants
// ---------------------------------------------------------------------------

struct Raw {
    konst: Konst,
    /// Line the declaration ends on, for the run-of-declarations doc rule.
    end_line: usize,
}

fn collect(items: &[syn::Item], src: &Source, out: &mut Vec<Raw>) {
    for item in items {
        match item {
            syn::Item::Const(c) => {
                if let Some(raw) = one(
                    src,
                    Decl {
                        attrs: &c.attrs,
                        vis: &c.vis,
                        ident: &c.ident,
                        ty: &c.ty,
                        eq: c.eq_token.span(),
                        semi: c.semi_token.span(),
                    },
                ) {
                    out.push(raw);
                }
            }
            syn::Item::Static(s) => {
                if let Some(raw) = one(
                    src,
                    Decl {
                        attrs: &s.attrs,
                        vis: &s.vis,
                        ident: &s.ident,
                        ty: &s.ty,
                        eq: s.eq_token.span(),
                        semi: s.semi_token.span(),
                    },
                ) {
                    out.push(raw);
                }
            }
            // An inline module's constants are still module-level, and a glob
            // import inside it shadows exactly the same way, so they are in
            // scope for both the index and the shadowing rule. Test modules are
            // not: fixtures are not tuning.
            syn::Item::Mod(m) => {
                if let (false, Some((_, inner))) = (is_cfg_test(&m.attrs), &m.content) {
                    collect(inner, src, out);
                }
            }
            _ => {}
        }
    }
}

/// The parts of a `const` and a `static` that this module treats identically.
struct Decl<'a> {
    attrs: &'a [syn::Attribute],
    vis: &'a syn::Visibility,
    ident: &'a syn::Ident,
    ty: &'a syn::Type,
    eq: proc_macro2::Span,
    semi: proc_macro2::Span,
}

fn one(src: &Source, d: Decl<'_>) -> Option<Raw> {
    let name = d.ident.to_string();
    if !is_screaming(&name) {
        return None;
    }
    let line = d.ident.span().start().line;
    Some(Raw {
        konst: Konst {
            name,
            // From just past the `=` to just before the `;`. Both are single
            // tokens, so their spans are exact — joining the whole expression's
            // span is the thing that can silently truncate.
            value: src.between(d.eq.end(), d.semi.start()),
            line,
            doc: doc_of(d.attrs, src, line),
            exported: is_pub(d.vis),
            tuned: is_numeric_scalar(d.ty),
        },
        end_line: d.semi.end().line,
    })
}

/// The one-line meaning of a declaration.
///
/// The `///` block wins, and within it the opening SENTENCE — not the nearest
/// line, and not the whole block. These blocks routinely run ten lines with the
/// derivation written out, so the line closest to the declaration is usually the
/// least useful sentence in it, and the whole block is a page, not a table cell.
///
/// Falling back to a trailing `// 800 logical px` and then to a plain `//` block
/// above is for tier 3, where a constant is often annotated rather than
/// documented. It cannot tell a `//` inside a string literal from a comment, so
/// a constant holding a URL would index its own tail as its meaning. No such
/// constant exists in this tree, and the failure is cosmetic if one appears.
fn doc_of(attrs: &[syn::Attribute], src: &Source, line: usize) -> String {
    let summary = doc_summary(attrs)
        .or_else(|| src.trailing_comment(line))
        .unwrap_or_else(|| src.comment_block_above(line));
    first_sentence(&summary).to_string()
}

/// A run of declarations under one comment all report it.
///
/// `SWIM_MAX_UP` / `SWIM_MAX_DOWN` are written as a pair under a single heading
/// and the second one has nothing of its own to say. Inheriting is better than
/// a blank cell, and only applies when the declarations are literally adjacent —
/// one blank line between them is the author saying they are separate things.
fn carry_docs(raw: Vec<Raw>) -> Vec<Konst> {
    let mut carried = String::new();
    let mut carried_end = 0usize;
    let mut out = Vec::with_capacity(raw.len());
    for r in raw {
        let mut k = r.konst;
        if k.doc.is_empty() && k.line == carried_end + 1 {
            k.doc.clone_from(&carried);
        }
        carried.clone_from(&k.doc);
        carried_end = r.end_line;
        out.push(k);
    }
    out
}

fn is_screaming(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_uppercase())
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn is_numeric_scalar(ty: &syn::Type) -> bool {
    let syn::Type::Path(p) = ty else {
        return false;
    };
    if p.qself.is_some() || p.path.segments.len() != 1 {
        return false;
    }
    let seg = &p.path.segments[0];
    let name = seg.ident.to_string();
    matches!(seg.arguments, syn::PathArguments::None) && NUMERIC.contains(&name.as_str())
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("cfg")
            && matches!(&a.meta, syn::Meta::List(l) if l.tokens.to_string().contains("test"))
    })
}

/// The opening PARAGRAPH of a doc comment, rewrapped onto one line.
///
/// The TypeScript took the first line and stopped, because over there the first
/// line was the summary. Rust doc comments are hard-wrapped at the same margin
/// as the code, so the summary sentence routinely spills onto a second line and
/// taking one line yields "Baseline magnification, so the world reads at a
/// decent size on a small" — a cell that ends mid-clause. Joining to the first
/// blank doc line reconstructs the sentence the author wrote and stops before
/// the derivation underneath it, which is the part the index is deliberately not
/// reprinting.
fn doc_summary(attrs: &[syn::Attribute]) -> Option<String> {
    let mut paragraph: Vec<String> = Vec::new();
    for line in doc_lines(attrs) {
        if line.is_empty() {
            if paragraph.is_empty() {
                continue;
            }
            break;
        }
        paragraph.push(line);
    }
    (!paragraph.is_empty()).then(|| paragraph.join(" "))
}

fn doc_lines(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut out = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let syn::Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(s),
            ..
        }) = &nv.value
        else {
            continue;
        };
        out.push(s.value().trim().to_string());
    }
    out
}

// ---------------------------------------------------------------------------
// The tier-2 documentation rule
// ---------------------------------------------------------------------------

fn check_docs(items: &[syn::Item], out: &mut Vec<Undocumented>) {
    for item in items {
        match item {
            syn::Item::Const(i) => want(&i.vis, &i.attrs, "const", &i.ident, out),
            syn::Item::Static(i) => want(&i.vis, &i.attrs, "static", &i.ident, out),
            syn::Item::Fn(i) => want(&i.vis, &i.attrs, "fn", &i.sig.ident, out),
            syn::Item::Type(i) => want(&i.vis, &i.attrs, "type", &i.ident, out),
            syn::Item::Trait(i) => want(&i.vis, &i.attrs, "trait", &i.ident, out),
            syn::Item::Union(i) => want(&i.vis, &i.attrs, "union", &i.ident, out),
            syn::Item::Struct(i) => {
                want(&i.vis, &i.attrs, "struct", &i.ident, out);
                if is_pub(&i.vis) {
                    for f in &i.fields {
                        if let Some(name) = &f.ident {
                            want(&f.vis, &f.attrs, "field", name, out);
                        }
                    }
                }
            }
            syn::Item::Enum(i) => {
                want(&i.vis, &i.attrs, "enum", &i.ident, out);
                if is_pub(&i.vis) {
                    for v in &i.variants {
                        // A variant of a public enum is public; it has no `vis`
                        // of its own to consult.
                        if doc_summary(&v.attrs).is_none() {
                            out.push(Undocumented {
                                kind: "variant",
                                name: format!("{}::{}", i.ident, v.ident),
                                line: v.ident.span().start().line,
                            });
                        }
                    }
                }
            }
            // Inherent impls only. A trait impl's methods carry the trait's
            // documentation and have no visibility to make public.
            syn::Item::Impl(i) if i.trait_.is_none() => {
                for it in &i.items {
                    match it {
                        syn::ImplItem::Fn(f) => want(&f.vis, &f.attrs, "fn", &f.sig.ident, out),
                        syn::ImplItem::Const(c) => {
                            want(&c.vis, &c.attrs, "const", &c.ident, out);
                        }
                        syn::ImplItem::Type(t) => want(&t.vis, &t.attrs, "type", &t.ident, out),
                        _ => {}
                    }
                }
            }
            syn::Item::Mod(m) if is_pub(&m.vis) && !is_cfg_test(&m.attrs) => {
                if let Some((_, inner)) = &m.content {
                    want(&m.vis, &m.attrs, "mod", &m.ident, out);
                    check_docs(inner, out);
                }
            }
            _ => {}
        }
    }
}

fn is_pub(vis: &syn::Visibility) -> bool {
    matches!(vis, syn::Visibility::Public(_))
}

fn want(
    vis: &syn::Visibility,
    attrs: &[syn::Attribute],
    kind: &'static str,
    ident: &syn::Ident,
    out: &mut Vec<Undocumented>,
) {
    if is_pub(vis) && doc_summary(attrs).is_none() {
        out.push(Undocumented {
            kind,
            name: ident.to_string(),
            line: ident.span().start().line,
        });
    }
}

// ---------------------------------------------------------------------------
// Source text, addressed by line and column
// ---------------------------------------------------------------------------

/// The file's text with a line index, so a `syn` span can be turned back into
/// the characters the author actually typed.
struct Source<'a> {
    text: &'a str,
    lines: Vec<&'a str>,
    line_starts: Vec<usize>,
}

impl<'a> Source<'a> {
    fn new(text: &'a str) -> Self {
        let mut line_starts = vec![0usize];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Source {
            text,
            lines: text.lines().collect(),
            line_starts,
        }
    }

    /// Byte offset of a span position.
    ///
    /// `LineColumn::column` counts CHARACTERS, not bytes, so this walks the line
    /// rather than adding the two together. Comments in this tree contain em
    /// dashes and arrows; getting that wrong would slice a value mid-codepoint
    /// and panic.
    fn byte_at(&self, lc: LineColumn) -> usize {
        let Some(&start) = self.line_starts.get(lc.line.saturating_sub(1)) else {
            return self.text.len();
        };
        let mut off = start;
        for (n, ch) in self.text[start..].chars().enumerate() {
            if n == lc.column || ch == '\n' {
                return off;
            }
            off += ch.len_utf8();
        }
        off
    }

    /// Source between two positions, with whitespace runs collapsed to one
    /// space so a declaration rustfmt wrapped over four lines is still one cell.
    fn between(&self, from: LineColumn, to: LineColumn) -> String {
        let (a, b) = (self.byte_at(from), self.byte_at(to));
        if a >= b || b > self.text.len() {
            return String::new();
        }
        self.text[a..b]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn trailing_comment(&self, line: usize) -> Option<String> {
        let text = *self.lines.get(line.checked_sub(1)?)?;
        let rest = text[text.find("//")? + 2..].trim();
        (!rest.is_empty() && !rest.starts_with('/')).then(|| rest.to_string())
    }

    /// First prose line of the contiguous `//` block directly above `line`.
    fn comment_block_above(&self, line: usize) -> String {
        let decl = line.saturating_sub(1).min(self.lines.len());
        let mut top = decl;
        while top > 0 && self.lines[top - 1].trim_start().starts_with("//") {
            top -= 1;
        }
        // Same paragraph rule as a `///` block: skip the rules and blank
        // delimiters, then take everything up to the next break.
        let mut paragraph: Vec<&str> = Vec::new();
        for raw in &self.lines[top..decl] {
            let p = raw
                .trim()
                .trim_start_matches('/')
                .trim_start_matches('!')
                .trim();
            if p.is_empty() || is_divider(p) {
                if paragraph.is_empty() {
                    continue;
                }
                break;
            }
            paragraph.push(p);
        }
        paragraph.join(" ")
    }
}

/// A `// -----` rule between sections says nothing about the next declaration.
///
/// Titled rules (`// --- Lattices ------`) count too. They read as prose to
/// anything looking for letters, and joining one to the paragraph below it
/// produces a cell that opens with forty dashes.
fn is_divider(s: &str) -> bool {
    s.chars().all(|c| matches!(c, '-' | '=' | '*'))
        || s.starts_with("---")
        || s.starts_with("===")
        || s.starts_with("***")
}

/// The first sentence of a summary, which is as much as a table cell can hold.
///
/// The opening PARAGRAPH is the wrong unit on its own: this codebase writes
/// paragraphs that run six hundred characters because the derivation of a number
/// belongs beside it, and pasting one into a Markdown row destroys the table for
/// every other row. The opening SENTENCE is the summary; the paragraph is the
/// argument, and the argument is why the index links to the line instead of
/// trying to replace it.
///
/// A period is only a full stop when a space follows it, which is what keeps
/// `0.0098` and `CHUNK_CELLS²` intact. The look-back for a second period two
/// characters earlier is for `e.g. ` and `i.e. `, which end a word and not a
/// sentence.
fn first_sentence(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if !matches!(b, b'.' | b'!' | b'?') || bytes.get(i + 1) != Some(&b' ') {
            continue;
        }
        if i >= 2 && bytes[i - 2] == b'.' {
            continue;
        }
        return s[..=i].trim_end();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(src: &str) -> Vec<String> {
        scan(src)
            .expect("parses")
            .into_iter()
            .filter(|k| k.tuned)
            .map(|k| k.name)
            .collect()
    }

    #[test]
    fn a_numeric_module_constant_is_indexed_with_its_value_line_and_doc() {
        let src = "/// Seconds per physics step.\npub const STEP_DT: f32 = 1.0 / 120.0;\n";
        let found = scan(src).expect("parses");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "STEP_DT");
        assert_eq!(found[0].value, "1.0 / 120.0");
        assert_eq!(found[0].line, 2);
        assert_eq!(found[0].doc, "Seconds per physics step.");
        assert!(found[0].exported);
        assert!(found[0].tuned);
    }

    #[test]
    fn constants_generated_inside_a_macro_body_are_invisible() {
        // This is the case that justifies parsing over matching: `bitflags!`
        // declares scores of `const` items that are bit identities, not knobs.
        let src = "bitflags! {\n    pub struct Tags: u32 {\n        const ORE = 1;\n        const DIGGABLE = 2;\n    }\n}\npub const REAL: f32 = 3.0;\n";
        assert_eq!(names(src), vec!["REAL"]);
    }

    #[test]
    fn a_newtype_over_an_integer_is_not_counted_as_a_tuned_number() {
        let src = "pub const MARKER: CellId = CellId(7);\npub const REACH: i32 = 7;\n";
        assert_eq!(names(src), vec!["REACH"]);
    }

    #[test]
    fn tables_colours_and_strings_are_not_counted_as_tuned_numbers() {
        let src = concat!(
            "const STOPS: [f32; 3] = [0.0, 0.5, 1.0];\n",
            "const SKY: Rgb = Rgb(10, 20, 30);\n",
            "const LABELS: &[&str] = &[\"a\"];\n",
            "const NAME: &str = \"a\";\n",
            "const ON: bool = true;\n",
            "const OCTAVES: u32 = 4;\n",
        );
        assert_eq!(names(src), vec!["OCTAVES"]);
    }

    #[test]
    fn a_constant_inside_a_function_body_is_out_of_scope() {
        let src = "fn f() {\n    const INNER: f32 = 1.0;\n}\npub const OUTER: f32 = 2.0;\n";
        assert_eq!(names(src), vec!["OUTER"]);
    }

    #[test]
    fn an_associated_constant_in_an_impl_block_is_out_of_scope() {
        let src = "struct S;\nimpl S {\n    pub const INNER: f32 = 1.0;\n}\n";
        assert!(names(src).is_empty());
    }

    #[test]
    fn a_constant_in_an_inline_module_is_in_scope_but_a_test_module_is_not() {
        let src = concat!(
            "pub mod inner {\n    pub const KEPT: f32 = 1.0;\n}\n",
            "#[cfg(test)]\nmod tests {\n    const DROPPED: f32 = 2.0;\n}\n",
        );
        assert_eq!(names(src), vec!["KEPT"]);
    }

    #[test]
    fn a_declaration_wrapped_over_several_lines_still_yields_one_value() {
        let src = "const WIDE: f32 =\n    (PLAYER_CELLS_W\n        * CELL_SIZE) as f32;\n";
        let found = scan(src).expect("parses");
        assert_eq!(found[0].value, "(PLAYER_CELLS_W * CELL_SIZE) as f32");
    }

    #[test]
    fn a_trailing_comment_documents_a_constant_that_has_no_doc_block() {
        let src = "const MAX_VIEW_W: f32 = 800.0; // 800 logical px\n";
        assert_eq!(scan(src).expect("parses")[0].doc, "800 logical px");
    }

    #[test]
    fn a_doc_block_beats_the_trailing_value_annotation() {
        let src =
            "/// Player collision box width, in world px.\npub const PLAYER_W: f32 = 10.0; // 10\n";
        assert_eq!(
            scan(src).expect("parses")[0].doc,
            "Player collision box width, in world px."
        );
    }

    #[test]
    fn the_meaning_is_the_opening_paragraph_not_the_line_nearest_the_declaration() {
        let src = "/// Cap on how much world is on screen.\n///\n/// ...which would let the camera see past the generated edge.\nconst MAX_VIEW_CELLS_W: i32 = 160;\n";
        assert_eq!(
            scan(src).expect("parses")[0].doc,
            "Cap on how much world is on screen."
        );
    }

    #[test]
    fn only_the_opening_sentence_of_a_long_paragraph_reaches_the_table() {
        let src = "/// Upper bound on retained diverged chunks. At 1024 cells a snapshot is\n/// ~6 KB, so 2048 chunks is ~12 MB — far more than a session touches.\nconst MAX_DIVERGED: usize = 2048;\n";
        assert_eq!(
            scan(src).expect("parses")[0].doc,
            "Upper bound on retained diverged chunks."
        );
    }

    #[test]
    fn a_decimal_point_does_not_end_a_sentence() {
        let src = "/// At 0.0098 the second octave sits at a 51-cell period, which is small.\nconst TUN_FX: f64 = 0.0098;\n";
        assert_eq!(
            scan(src).expect("parses")[0].doc,
            "At 0.0098 the second octave sits at a 51-cell period, which is small."
        );
    }

    #[test]
    fn an_abbreviation_does_not_end_a_sentence() {
        let src =
            "/// Applies to powders, e.g. sand and ash, and nothing else.\nconst K: f32 = 1.0;\n";
        assert_eq!(
            scan(src).expect("parses")[0].doc,
            "Applies to powders, e.g. sand and ash, and nothing else."
        );
    }

    #[test]
    fn a_summary_hard_wrapped_over_two_lines_is_rejoined_into_one_sentence() {
        let src = "/// Baseline magnification, so the world reads at a decent size on a small\n/// display rather than being a field of 5px specks.\nconst ZOOM_MIN: f32 = 2.0;\n";
        assert_eq!(
            scan(src).expect("parses")[0].doc,
            "Baseline magnification, so the world reads at a decent size on a small display rather than being a field of 5px specks."
        );
    }

    #[test]
    fn a_plain_comment_block_above_documents_a_constant_and_a_divider_does_not() {
        let src = "// ---------------------------\n// The swim envelope.\nconst SWIM_MAX_UP: f32 = 40.0;\n";
        assert_eq!(scan(src).expect("parses")[0].doc, "The swim envelope.");
    }

    #[test]
    fn a_titled_rule_is_a_divider_and_not_the_first_line_of_the_meaning() {
        let src = "// --- Lattices ------------------------------\n// Deliberately out of phase with decor/structures.rs.\nconst LAT_X: i32 = 96;\n";
        assert_eq!(
            scan(src).expect("parses")[0].doc,
            "Deliberately out of phase with decor/structures.rs."
        );
    }

    #[test]
    fn an_adjacent_declaration_inherits_the_comment_of_the_one_above_it() {
        let src = "// The swim envelope.\nconst SWIM_MAX_UP: f32 = 40.0;\nconst SWIM_MAX_DOWN: f32 = 60.0;\n";
        let found = scan(src).expect("parses");
        assert_eq!(found[1].doc, "The swim envelope.");
    }

    #[test]
    fn a_blank_line_stops_a_declaration_from_inheriting_the_comment_above() {
        let src = "// The swim envelope.\nconst SWIM_MAX_UP: f32 = 40.0;\n\nconst UNRELATED: f32 = 60.0;\n";
        let found = scan(src).expect("parses");
        assert_eq!(found[1].doc, "");
    }

    #[test]
    fn a_multibyte_comment_does_not_shift_the_extracted_value() {
        // Em dashes and arrows are everywhere in this tree's prose. Columns are
        // character offsets, so treating one as a byte offset slices wrong.
        let src = "/// Tempo — a feel knob, ratio → ratio.\npub const MOVE_TEMPO: f32 = 1.15;\n";
        assert_eq!(scan(src).expect("parses")[0].value, "1.15");
    }

    #[test]
    fn a_private_constant_is_indexed_but_not_marked_exported() {
        let src = "/// Baseline magnification.\nconst ZOOM_MIN: f32 = 2.0;\n";
        let found = scan(src).expect("parses");
        assert!(!found[0].exported);
        assert!(found[0].tuned);
    }

    #[test]
    fn a_static_assertion_named_underscore_is_not_a_constant() {
        let src = "const _: () = assert!(true);\npub const REAL: usize = 1;\n";
        assert_eq!(names(src), vec!["REAL"]);
    }

    #[test]
    fn a_lowercase_item_is_not_a_tuned_constant() {
        let src = "pub const lowercase_thing: f32 = 1.0;\npub const REAL: f32 = 2.0;\n";
        assert_eq!(names(src), vec!["REAL"]);
    }

    #[test]
    fn a_numeric_static_is_collected_alongside_constants() {
        let src = "/// A knob.\npub static TUNED: f32 = 1.0;\n";
        let found = scan(src).expect("parses");
        assert_eq!(found[0].name, "TUNED");
        assert!(found[0].tuned);
    }

    // -- the documentation rule ------------------------------------------

    fn undocumented(src: &str) -> Vec<String> {
        undocumented_pub_items(src)
            .expect("parses")
            .into_iter()
            .map(|u| format!("{} {}", u.kind, u.name))
            .collect()
    }

    #[test]
    fn an_undocumented_public_constant_is_reported() {
        assert_eq!(undocumented("pub const SEED: u32 = 1;\n"), ["const SEED"]);
    }

    #[test]
    fn a_private_constant_needs_no_doc_comment() {
        assert!(undocumented("const SEED: u32 = 1;\n").is_empty());
    }

    #[test]
    fn a_public_function_struct_field_and_enum_variant_all_need_doc_comments() {
        let src = concat!(
            "pub fn f() {}\n",
            "pub struct S {\n    pub w: i32,\n}\n",
            "pub enum E {\n    A,\n}\n",
        );
        assert_eq!(
            undocumented(src),
            ["fn f", "struct S", "field w", "enum E", "variant E::A"]
        );
    }

    #[test]
    fn a_documented_public_surface_reports_nothing() {
        let src = concat!(
            "/// A knob.\npub const SEED: u32 = 1;\n",
            "/// A viewport.\npub struct View {\n    /// Width.\n    pub w: i32,\n}\n",
            "impl View {\n    /// Fit it.\n    pub fn fit() {}\n}\n",
        );
        assert!(undocumented(src).is_empty(), "{:?}", undocumented(src));
    }

    #[test]
    fn a_module_declaration_is_exempt_because_its_own_file_header_documents_it() {
        assert!(undocumented("pub mod world;\n").is_empty());
    }

    #[test]
    fn an_inline_public_module_is_not_exempt_because_it_has_no_file_header() {
        assert_eq!(undocumented("pub mod world {}\n"), ["mod world"]);
    }

    #[test]
    fn a_trait_impl_method_is_exempt_because_the_trait_documents_it() {
        let src = "/// A viewport.\npub struct View;\nimpl Default for View {\n    fn default() -> Self {\n        View\n    }\n}\n";
        assert!(undocumented(src).is_empty());
    }
}

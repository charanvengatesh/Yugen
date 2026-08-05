//! The declarative schema system.
//!
//! A schema is data: a record of dotted field name -> descriptor. It is the
//! single source of truth for three things that must never disagree —
//! validation of the `.block` text, the shape of the emitted Rust struct, and
//! which flat tables get built. Adding a field to a schema is therefore the
//! whole job of adding a field to the game.
//!
//! Types are written in the notation of FORMAT.md §2 (`int`, `chance`,
//! `list<ref(block)>`, `enum(a|b|c)`) and parsed here, so the doc and the code
//! cannot drift.

use crate::error::{ContentError, Loc, Result};
use crate::parser::{RawRecord, RawValue};
use crate::value::{Def, Value};
use crate::{bail, err};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Type notation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeNode {
    Int,
    Float,
    Bool,
    Str,
    Enum(Vec<String>),
    Color,
    Range,
    Chance,
    Text,
    Ref { kind: String, chain: bool },
    List(Box<TypeNode>),
    Record,
}

pub fn parse_type(src: &str) -> Result<TypeNode> {
    let s = src.trim();
    let simple = match s {
        "int" => Some(TypeNode::Int),
        "float" => Some(TypeNode::Float),
        "bool" => Some(TypeNode::Bool),
        "string" => Some(TypeNode::Str),
        "color" => Some(TypeNode::Color),
        "range" => Some(TypeNode::Range),
        "chance" => Some(TypeNode::Chance),
        "text" => Some(TypeNode::Text),
        "record[]" => Some(TypeNode::Record),
        _ => None,
    };
    if let Some(t) = simple {
        return Ok(t);
    }

    if let Some(inner) = s.strip_prefix("list<").and_then(|r| r.strip_suffix('>'))
        && !inner.is_empty()
    {
        return Ok(TypeNode::List(Box::new(parse_type(inner)?)));
    }

    if let Some(inner) = s.strip_prefix("enum(").and_then(|r| r.strip_suffix(')'))
        && !inner.is_empty()
    {
        return Ok(TypeNode::Enum(
            inner.split('|').map(|v| v.trim().to_string()).collect(),
        ));
    }

    // `ref?(kind)` is a preference chain: `cap dirt|mud|stone` takes the first
    // id that exists, which keeps content additive.
    if let Some(rest) = s.strip_prefix("ref") {
        let (chain, rest) = match rest.strip_prefix('?') {
            Some(r) => (true, r),
            None => (false, rest),
        };
        if let Some(kind) = rest.strip_prefix('(').and_then(|r| r.strip_suffix(')'))
            && !kind.is_empty()
            && kind.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Ok(TypeNode::Ref {
                kind: kind.to_string(),
                chain,
            });
        }
    }

    Err(err!("unknown type notation: '{src}'"))
}

// ---------------------------------------------------------------------------
// Schema shape
// ---------------------------------------------------------------------------

/// Which flat array a hot field is packed into. Governs both the emitted Rust
/// element type and the saturation behaviour when a value overflows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayKind {
    U8,
    U16,
    U32,
    F32,
}

impl ArrayKind {
    pub fn rust_elem(self) -> &'static str {
        match self {
            ArrayKind::U8 => "u8",
            ArrayKind::U16 => "u16",
            ArrayKind::U32 => "u32",
            ArrayKind::F32 => "f32",
        }
    }
}

pub type TableFn = Box<dyn Fn(&Def, &TableCtx) -> Option<f64>>;
pub type RowFn = Box<dyn Fn(&Def, &TableCtx) -> Option<Vec<String>>>;
pub type DefaultFn = Box<dyn Fn(&Def) -> Option<Value>>;
pub type CheckFn = Box<dyn Fn(&Value) -> Option<String>>;

/// One flat array indexed by code (FORMAT.md §5).
pub struct HotArray {
    /// Exported symbol name, e.g. `MAT_DENSITY`. Frozen — hot paths import it.
    pub name: String,
    pub array: ArrayKind,
    /// Value every slot starts at, including tombstones and unset defs.
    pub fill: f64,
    pub doc: Option<String>,
    /// Per-def value; `None` leaves `fill` in place.
    pub value: TableFn,
}

/// A packed pair matrix, `NAME[a * COUNT + b]` (FORMAT.md §5).
pub struct Matrix {
    pub name: String,
    pub doc: Option<String>,
    /// Ids of the columns set for this def's row, or `None` for an empty row.
    pub row: RowFn,
}

/// A generated bit-constant set, emitted as a `bitflags!` type.
pub struct BitConstants {
    pub name: String,
    pub doc: Option<String>,
    pub values: Vec<String>,
}

/// Either a fixed value or one computed from the def built so far.
pub enum FieldDefault {
    Value(Value),
    Fn(DefaultFn),
}

#[derive(Default)]
pub struct Field {
    /// FORMAT.md §2 type notation.
    pub ty: String,
    /// Doc comment carried onto the emitted struct — say WHY, not what.
    pub doc: Option<String>,
    /// Value when the key is absent. A function receives the def built so far.
    pub default: Option<FieldDefault>,
    /// Absent + no default = compile error. Within a group, only when present.
    pub required: bool,
    /// ref only: an unknown id warns instead of failing the build.
    pub lenient: bool,
    /// enum only: resolve to this number on the def (e.g. state -> MaterialState).
    pub map: Option<Vec<(String, i64)>>,
    /// enum only: emit the variants as this named Rust enum instead of a
    /// synthesised name.
    pub alias: Option<String>,
    /// `record[]` only: the sub-fields of each element, in order.
    pub fields: Option<Vec<(String, Field)>>,
    /// `record[]` only: name of the emitted element struct.
    pub element: Option<String>,
    /// Post-coercion constraint. Returns a message to reject, `None` to pass.
    pub check: Option<CheckFn>,
    /// Also emit flat arrays for this field.
    pub hot: bool,
    /// The array(s) `hot` produces. A colour, for instance, produces three.
    pub hot_arrays: Vec<HotArray>,
}

impl Field {
    pub fn new(ty: &str) -> Self {
        Field {
            ty: ty.to_string(),
            ..Default::default()
        }
    }

    pub fn doc(mut self, d: &str) -> Self {
        self.doc = Some(d.to_string());
        self
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    pub fn default(mut self, v: Value) -> Self {
        self.default = Some(FieldDefault::Value(v));
        self
    }

    pub fn default_int(self, n: i64) -> Self {
        self.default(Value::Int(n))
    }

    pub fn default_float(self, n: f64) -> Self {
        self.default(Value::Float(n))
    }

    pub fn default_bool(self, b: bool) -> Self {
        self.default(Value::Bool(b))
    }

    pub fn default_str(self, s: &str) -> Self {
        self.default(Value::Str(s.to_string()))
    }

    pub fn default_fn(mut self, f: impl Fn(&Def) -> Option<Value> + 'static) -> Self {
        self.default = Some(FieldDefault::Fn(Box::new(f)));
        self
    }

    pub fn map(mut self, pairs: &[(&str, i64)]) -> Self {
        self.map = Some(pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect());
        self
    }

    pub fn alias(mut self, name: &str) -> Self {
        self.alias = Some(name.to_string());
        self
    }

    pub fn element(mut self, name: &str, fields: Vec<(String, Field)>) -> Self {
        self.element = Some(name.to_string());
        self.fields = Some(fields);
        self
    }

    pub fn check(mut self, f: impl Fn(&Value) -> Option<String> + 'static) -> Self {
        self.check = Some(Box::new(f));
        self
    }

    /// Reject a number outside an inclusive range — by far the most common
    /// constraint, so it gets a shorthand.
    pub fn between(self, lo: f64, hi: f64) -> Self {
        self.check(move |v| match v.as_num() {
            Some(n) if n < lo || n > hi => Some(format!("{n} is outside {lo}..{hi}")),
            _ => None,
        })
    }

    pub fn hot(mut self, arrays: Vec<HotArray>) -> Self {
        self.hot = true;
        self.hot_arrays = arrays;
        self
    }
}

/// Build one flat array. The closure gets the def and the table context.
pub fn table(
    name: &str,
    array: ArrayKind,
    fill: f64,
    doc: &str,
    value: impl Fn(&Def, &TableCtx) -> Option<f64> + 'static,
) -> HotArray {
    HotArray {
        name: name.to_string(),
        array,
        fill,
        doc: if doc.is_empty() {
            None
        } else {
            Some(doc.to_string())
        },
        value: Box::new(value),
    }
}

pub struct Schema {
    pub kind: String,
    /// Prefix for the generated registry symbols: `BLOCK_IDS`, `BLOCKS`, `BLOCK`.
    pub prefix: String,
    /// Name of the generated def struct, e.g. `BlockDef`.
    pub iface: String,
    /// Prefix for generated group and element structs, e.g. `Block` -> `BlockHeat`.
    pub iface_prefix: String,
    /// Dotted field name -> descriptor, in emission order.
    pub fields: Vec<(String, Field)>,
    /// Tables that read more than one field (`MAT_HEATTHRESH`) or a group's presence.
    pub tables: Vec<HotArray>,
    pub matrices: Vec<Matrix>,
    pub constants: Vec<BitConstants>,
    /// Raw field values used to synthesise the placeholder def for a tombstoned
    /// id (FORMAT.md §4). It only has to satisfy the schema's required fields —
    /// nothing should ever be looking at it, but old saves still index it.
    pub tombstone: Vec<(String, String)>,
}

impl Schema {
    pub fn field(&self, key: &str) -> Option<&Field> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, f)| f)
    }

    /// Every hot array a schema declares, field-level and derived, in order.
    pub fn hot_arrays(&self) -> Vec<&HotArray> {
        let mut out: Vec<&HotArray> = Vec::new();
        for (_, field) in &self.fields {
            if !field.hot {
                continue;
            }
            out.extend(field.hot_arrays.iter());
        }
        out.extend(self.tables.iter());
        out
    }
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

/// What a table or matrix callback may ask the compiler about the world.
pub struct TableCtx<'a> {
    /// Codes for the schema's own kind.
    pub codes: &'a HashMap<String, u32>,
    /// Codes for every kind, so a cross-kind lookup is well-defined.
    pub foreign: &'a HashMap<String, HashMap<String, u32>>,
    /// Bit index per declared constant set.
    pub bits: &'a HashMap<String, HashMap<String, u32>>,
    /// A callback cannot return an error through `Option<f64>`, so a rejected
    /// `bit()` is stashed here and raised by the emitter once the pass is done.
    pub error: RefCell<Option<ContentError>>,
}

/// Sentinel for "this threshold can never be reached".
pub const NEVER: f64 = 0xffff as f64;

impl<'a> TableCtx<'a> {
    pub fn new(
        codes: &'a HashMap<String, u32>,
        foreign: &'a HashMap<String, HashMap<String, u32>>,
        bits: &'a HashMap<String, HashMap<String, u32>>,
    ) -> Self {
        TableCtx {
            codes,
            foreign,
            bits,
            error: RefCell::new(None),
        }
    }

    /// Code for an id; 0 (empty) for anything unknown, matching the old facade.
    pub fn code(&self, id: &str) -> f64 {
        self.codes.get(id).copied().unwrap_or(0) as f64
    }

    /// Code for an id of ANOTHER kind — `foreign_code("block", "stone")` from
    /// inside an item table.
    ///
    /// `code` deliberately resolves against the schema's own kind only, so
    /// without this an item table asking for a block code silently got 0 (air).
    /// That is a table of nothing rather than a build failure, which is exactly
    /// the class of bug the compiler exists to prevent, so cross-kind lookups
    /// get their own verb. Returns 0 for an unknown kind or id, matching `code`.
    ///
    /// Well-defined for every kind because the driver assigns codes for ALL
    /// kinds before emitting ANY of them — see the pre-pass in `driver`.
    pub fn foreign_code(&self, kind: &str, id: &str) -> f64 {
        self.foreign
            .get(kind)
            .and_then(|m| m.get(id))
            .copied()
            .unwrap_or(0) as f64
    }

    /// Bit index of a declared constant value, e.g. `bit("TAG", "rock")`.
    pub fn bit(&self, set: &str, value: &str) -> f64 {
        match self.bits.get(set).and_then(|m| m.get(value)) {
            Some(&b) => b as f64,
            None => {
                let mut slot = self.error.borrow_mut();
                if slot.is_none() {
                    *slot = Some(err!("'{value}' is not a declared '{set}' constant"));
                }
                0.0
            }
        }
    }

    /// `1 << bit(set, value)`, folded over a list of values — the shape every
    /// tag, band and immunity bitfield wants.
    pub fn mask(&self, set: &str, values: &[&str]) -> f64 {
        let mut m: u32 = 0;
        for v in values {
            m |= 1u32 << (self.bit(set, v) as u32);
        }
        m as f64
    }

    pub fn take_error(&self) -> Option<ContentError> {
        self.error.borrow_mut().take()
    }
}

pub struct ResolveCtx<'a> {
    /// Known ids per kind, for `ref(kind)` validation. Missing kind = unchecked.
    pub registries: &'a HashMap<String, HashSet<String>>,
    pub warnings: RefCell<Vec<String>>,
}

impl<'a> ResolveCtx<'a> {
    pub fn new(registries: &'a HashMap<String, HashSet<String>>) -> Self {
        ResolveCtx {
            registries,
            warnings: RefCell::new(Vec::new()),
        }
    }

    fn registry(&self, kind: &str) -> Option<&HashSet<String>> {
        self.registries.get(kind)
    }

    fn warn(&self, message: String, loc: Option<&Loc>) {
        let text = match loc {
            Some(l) => format!("{l}: {message}"),
            None => message,
        };
        self.warnings.borrow_mut().push(text);
    }

    pub fn into_warnings(self) -> Vec<String> {
        self.warnings.into_inner()
    }
}

// ---------------------------------------------------------------------------
// Resolution: RawRecord -> Def
// ---------------------------------------------------------------------------

/// Split `heat.conduct` -> `(Some("heat"), "conduct")`; `density` -> `(None, "density")`.
fn split(key: &str) -> (Option<&str>, &str) {
    match key.find('.') {
        Some(i) => (Some(&key[..i]), &key[i + 1..]),
        None => (None, key),
    }
}

/// Validate a raw record against a schema and produce the nested def.
///
/// Groups (`heat`, `flammable`, `growth`) are emitted only when the source set
/// at least one of their keys, and sub-field defaults are NOT injected — a
/// material with no `heat` block has no `heat` group, exactly as the
/// hand-written registry did. Defaults still apply when the flat tables are
/// built, which is where they actually matter.
pub fn resolve_record(schema: &Schema, rec: &RawRecord, ctx: &ResolveCtx) -> Result<Def> {
    // Reject unknown keys outright: a typo'd key that is silently dropped is a
    // material that quietly stops behaving the way its file says it does.
    for key in rec.fields.keys() {
        if key == "id" || schema.field(key).is_some() {
            continue;
        }
        let loc = &rec.fields.get(key).unwrap()[0].loc;
        bail!(loc, "unknown {} field '{key}'", schema.kind);
    }

    let mut present: HashSet<&str> = HashSet::new();
    for key in rec.fields.keys() {
        if let (Some(g), _) = split(key) {
            present.insert(g);
        }
    }

    let mut def = Def::new();
    def.insert("id", Value::Str(rec.id.clone()));

    for (key, field) in &schema.fields {
        if key == "id" {
            continue;
        }
        let (group, leaf) = split(key);
        if let Some(g) = group
            && !present.contains(g)
        {
            continue;
        }

        let occ = rec.fields.get(key);
        let value = resolve_field(schema, key, field, occ, &def, rec, ctx)?;
        let Some(value) = value else { continue };

        match group {
            None => def.insert(leaf, value),
            Some(g) => {
                let mut obj = match def.get(g) {
                    Some(Value::Group(existing)) => existing.clone(),
                    _ => Def::new(),
                };
                obj.insert(leaf, value);
                def.insert(g, Value::Group(obj));
            }
        }
    }

    Ok(def)
}

fn apply_default(field: &Field, def: &Def) -> Option<Value> {
    match &field.default {
        None => None,
        Some(FieldDefault::Value(v)) => Some(v.clone()),
        Some(FieldDefault::Fn(f)) => f(def),
    }
}

fn resolve_field(
    schema: &Schema,
    key: &str,
    field: &Field,
    occ: Option<&[RawValue]>,
    def: &Def,
    rec: &RawRecord,
    ctx: &ResolveCtx,
) -> Result<Option<Value>> {
    let node = parse_type(&field.ty)?;

    if let Some(o) = occ
        && o.len() > 1
        && node != TypeNode::Record
    {
        bail!(
            &o[1].loc,
            "'{key}' is set {} times but is not a list-of-records",
            o.len()
        );
    }

    if node == TypeNode::Record {
        let Some(o) = occ else {
            return Ok(apply_default(field, def));
        };
        let mut out = Vec::with_capacity(o.len());
        for raw in o {
            out.push(resolve_sub_record(key, field, raw, ctx)?);
        }
        return Ok(Some(Value::Records(out)));
    }

    let Some(o) = occ else {
        let dflt = apply_default(field, def);
        if dflt.is_none() && field.required {
            bail!(&rec.loc, "{} '{}' is missing '{key}'", schema.kind, rec.id);
        }
        return Ok(dflt);
    };

    let raw = &o[0];
    if node == TypeNode::Text {
        let Some(lines) = &raw.lines else {
            bail!(&raw.loc, "'{key}' needs a '|' block");
        };
        return Ok(Some(Value::Text(lines.clone())));
    }
    let Some(text) = &raw.text else {
        bail!(&raw.loc, "'{key}' is not a text block field");
    };

    let value = coerce(&node, text, field, &raw.loc, ctx)?;
    if let Some(check) = &field.check
        && let Some(bad) = check(&value)
    {
        bail!(&raw.loc, "'{key}': {bad}");
    }
    Ok(Some(value))
}

/// `drop item=stone_chunk count=1..3 chance=50%` — one record on one line.
///
/// An element may also carry a heredoc body, which is how a sprite frame gets
/// both its attributes and its pixels on one occurrence:
///
/// ```text
/// art.seq  state=run mode=phase |
///   .4
///   32
/// ```
///
/// At most one sub-field may be declared `text`; it binds from the body, and
/// every other sub-field still comes from the `key=value` pairs with `enum`,
/// `required` and `default` checked exactly as before. Keeping the attributes in
/// pair syntax rather than inventing a nested mini-DSL inside the body is the
/// whole point — compile-time validation is what this compiler is for.
fn resolve_sub_record(key: &str, field: &Field, raw: &RawValue, ctx: &ResolveCtx) -> Result<Def> {
    let Some(fields) = &field.fields else {
        bail!("'{key}' is record[] but declares no fields");
    };
    let parts = split_pairs(raw.text.as_deref().unwrap_or(""), &raw.loc)?;
    let mut out = Def::new();

    for (k, _) in &parts {
        if !fields.iter().any(|(name, _)| name == k) {
            bail!(&raw.loc, "unknown '{key}' attribute '{k}'");
        }
    }

    for (k, sub) in fields {
        let node = parse_type(&sub.ty)?;
        let given = parts
            .iter()
            .find(|(name, _)| name == k)
            .map(|(_, v)| v.as_str());

        // A `text` sub-field is the body, never a pair. Writing it as `k=...` is
        // a misunderstanding worth naming rather than letting `coerce` reject it
        // with the generic "text fields need a '|' block".
        if node == TypeNode::Text {
            if given.is_some() {
                bail!(
                    &raw.loc,
                    "'{key}' attribute '{k}=' is a text field — write it as the '|' body"
                );
            }
            match &raw.lines {
                None => {
                    let dflt = apply_default(sub, &out);
                    if dflt.is_none() && sub.required {
                        bail!(&raw.loc, "'{key}' is missing its '{k}' '|' block");
                    }
                    if let Some(d) = dflt {
                        out.insert(k.clone(), d);
                    }
                }
                Some(lines) => out.insert(k.clone(), Value::Text(lines.clone())),
            }
            continue;
        }

        let Some(text) = given else {
            let dflt = apply_default(sub, &out);
            if dflt.is_none() && sub.required {
                bail!(&raw.loc, "'{key}' is missing '{k}='");
            }
            if let Some(d) = dflt {
                out.insert(k.clone(), d);
            }
            continue;
        };
        out.insert(k.clone(), coerce(&node, text, sub, &raw.loc, ctx)?);
    }

    Ok(out)
}

/// Tokenize `a=1 b="two words" c=3` respecting quotes.
///
/// Mirrors a global regex scan: text between matches is skipped rather than
/// rejected, and only a source that yields no pairs at all is an error.
fn split_pairs(src: &str, loc: &Loc) -> Result<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut consumed = 0usize;

    while i < bytes.len() {
        // A key must start at an identifier character.
        if !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i;
        while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'=' {
            i = j.max(start + 1);
            continue;
        }
        let name = &src[start..j];
        let mut k = j + 1;

        let value = if k < bytes.len() && bytes[k] == b'"' {
            // `"(?:[^"\\]|\\.)*"`
            let mut buf = String::new();
            k += 1;
            let mut closed = false;
            while k < bytes.len() {
                let c = bytes[k];
                if c == b'\\' && k + 1 < bytes.len() {
                    buf.push(src[k + 1..].chars().next().unwrap());
                    k += 1 + src[k + 1..].chars().next().unwrap().len_utf8();
                    continue;
                }
                if c == b'"' {
                    k += 1;
                    closed = true;
                    break;
                }
                let ch = src[k..].chars().next().unwrap();
                buf.push(ch);
                k += ch.len_utf8();
            }
            if !closed {
                // An unterminated quote never matched the pattern at all, so the
                // scan simply moves on from just after the key.
                i = j + 1;
                continue;
            }
            buf
        } else {
            // `\S+`
            let vs = k;
            while k < bytes.len() && !(bytes[k] as char).is_whitespace() {
                k += 1;
            }
            if k == vs {
                i = j + 1;
                continue;
            }
            src[vs..k].to_string()
        };

        consumed += k - start;
        out.push((name.to_string(), value));
        i = k;
    }

    if consumed == 0 && !src.trim().is_empty() {
        bail!(loc, "expected 'key=value' pairs, got '{src}'");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Scalar coercion
// ---------------------------------------------------------------------------

fn is_infinity(s: &str) -> Option<f64> {
    let t = s.trim();
    let (sign, rest) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    let lower = rest.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        Some(sign * f64::INFINITY)
    } else {
        None
    }
}

/// `Number(s)` with the pieces this format actually uses.
///
/// The empty string reads as 0, matching `Number("")`, because a bare key with
/// no value has always meant zero here rather than an error.
fn num(src: &str, loc: &Loc) -> Result<f64> {
    let s = src.trim();
    if let Some(v) = is_infinity(s) {
        return Ok(v);
    }
    if s.is_empty() {
        return Ok(0.0);
    }
    let parsed = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok().map(|n| n as f64)
    } else {
        s.parse::<f64>().ok()
    };
    match parsed {
        Some(n) if n.is_finite() => Ok(n),
        _ => Err(err!(loc, "'{src}' is not a number")),
    }
}

fn coerce(
    node: &TypeNode,
    text: &str,
    field: &Field,
    loc: &Loc,
    ctx: &ResolveCtx,
) -> Result<Value> {
    match node {
        TypeNode::Int => {
            let n = num(text, loc)?;
            if n.fract() != 0.0 {
                bail!(loc, "'{text}' is not an integer");
            }
            Ok(Value::Int(n as i64))
        }
        // Infinity is meaningful here: an unbreakable block's hardness.
        TypeNode::Float => match is_infinity(text) {
            Some(v) => Ok(Value::Float(v)),
            None => Ok(Value::Float(num(text, loc)?)),
        },
        TypeNode::Bool => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => Ok(Value::Bool(true)),
            "false" | "no" | "off" => Ok(Value::Bool(false)),
            _ => Err(err!(loc, "'{text}' is not a bool")),
        },
        TypeNode::Str => Ok(Value::Str(text.to_string())),
        TypeNode::Enum(values) => {
            let v = text.trim();
            if !values.iter().any(|x| x == v) {
                bail!(loc, "'{v}' must be one of: {}", values.join(" "));
            }
            match &field.map {
                Some(map) => {
                    let n = map
                        .iter()
                        .find(|(k, _)| k == v)
                        .map(|(_, n)| *n)
                        .unwrap_or(0);
                    Ok(Value::Int(n))
                }
                None => Ok(Value::Str(v.to_string())),
            }
        }
        TypeNode::Chance => {
            let t = text.trim();
            let v = match t.strip_suffix('%') {
                Some(pct) => num(pct, loc)? / 100.0,
                None => num(t, loc)?,
            };
            if !(0.0..=1.0).contains(&v) {
                bail!(loc, "chance '{text}' is outside 0..1");
            }
            Ok(Value::Float(v))
        }
        TypeNode::Range => {
            let t = text.trim();
            // `^(\S+)\.\.(\S+)$` — both halves must be whitespace-free, so
            // `1 .. 3` is not a range.
            if let Some(dots) = t.find("..") {
                let (a, b) = (&t[..dots], &t[dots + 2..]);
                if !a.is_empty()
                    && !b.is_empty()
                    && !a.chars().any(char::is_whitespace)
                    && !b.chars().any(char::is_whitespace)
                {
                    return Ok(Value::Range(num(a, loc)?, num(b, loc)?));
                }
            }
            let n = num(t, loc)?;
            Ok(Value::Range(n, n))
        }
        TypeNode::Color => color(text, loc),
        TypeNode::Ref { kind, chain } => reference(kind, *chain, text, field, loc, ctx),
        TypeNode::List(of) => {
            let items: Vec<&str> = text
                .split([' ', '\t', ','])
                .filter(|s| !s.is_empty())
                .collect();
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(coerce(of, it, field, loc, ctx)?);
            }
            Ok(Value::List(out))
        }
        TypeNode::Text => Err(err!(loc, "text fields need a '|' block")),
        TypeNode::Record => Err(err!(loc, "record[] fields cannot be inline")),
    }
}

fn color(text: &str, loc: &Loc) -> Result<Value> {
    let t = text.trim();
    if let Some(hex) = t.strip_prefix('#') {
        let d: Vec<char> = hex.chars().collect();
        let hx = |s: &str| u8::from_str_radix(s, 16).ok();
        if d.len() == 3 {
            let mut out = [0u8; 3];
            for (i, slot) in out.iter_mut().enumerate() {
                let pair: String = [d[i], d[i]].iter().collect();
                let Some(v) = hx(&pair) else {
                    bail!(loc, "'{t}' is not #rgb or #rrggbb")
                };
                *slot = v;
            }
            return Ok(Value::Color(out));
        }
        if d.len() == 6 {
            let mut out = [0u8; 3];
            for (i, slot) in out.iter_mut().enumerate() {
                let Some(v) = hx(&hex[i * 2..i * 2 + 2]) else {
                    bail!(loc, "'{t}' is not #rgb or #rrggbb")
                };
                *slot = v;
            }
            return Ok(Value::Color(out));
        }
        bail!(loc, "'{t}' is not #rgb or #rrggbb");
    }

    let parts: Vec<&str> = t
        .split([' ', '\t', ','])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 3 {
        bail!(loc, "colour needs 3 channels: '{t}'");
    }
    let mut out = [0u8; 3];
    for (i, p) in parts.iter().enumerate() {
        let n = num(p, loc)?;
        if !(0.0..=255.0).contains(&n) || n.fract() != 0.0 {
            bail!(loc, "colour channel '{n}' is not 0..255");
        }
        out[i] = n as u8;
    }
    Ok(Value::Color(out))
}

fn reference(
    kind: &str,
    chain: bool,
    text: &str,
    field: &Field,
    loc: &Loc,
    ctx: &ResolveCtx,
) -> Result<Value> {
    let candidates: Vec<&str> = if chain {
        text.split('|').map(str::trim).collect()
    } else {
        vec![text.trim()]
    };
    let first = candidates[0].to_string();

    let Some(known) = ctx.registry(kind) else {
        // The referenced registry does not exist yet (items are added by a
        // later pass). Emit the id verbatim so content can be authored ahead of
        // code.
        if !field.lenient {
            ctx.warn(
                format!("no '{kind}' registry to validate '{text}' against"),
                Some(loc),
            );
        }
        return Ok(Value::Str(first));
    };
    for c in &candidates {
        if known.contains(*c) {
            return Ok(Value::Str((*c).to_string()));
        }
    }

    if field.lenient {
        ctx.warn(format!("unknown {kind} '{text}'"), Some(loc));
        return Ok(Value::Str(first));
    }
    if chain {
        let last = candidates[candidates.len() - 1].to_string();
        ctx.warn(
            format!("no id in preference chain '{text}' exists; using '{last}'"),
            Some(loc),
        );
        return Ok(Value::Str(last));
    }
    Err(err!(loc, "unknown {kind} '{text}'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn loc() -> Loc {
        Loc {
            file: Rc::from("t.block"),
            line: 1,
        }
    }

    fn ctx_of(regs: &HashMap<String, HashSet<String>>) -> ResolveCtx<'_> {
        ResolveCtx::new(regs)
    }

    #[test]
    fn type_notation_round_trips() {
        assert_eq!(parse_type("int").unwrap(), TypeNode::Int);
        assert_eq!(
            parse_type("enum(a|b|c)").unwrap(),
            TypeNode::Enum(vec!["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(
            parse_type("ref(block)").unwrap(),
            TypeNode::Ref {
                kind: "block".into(),
                chain: false
            }
        );
        assert_eq!(
            parse_type("ref?(block)").unwrap(),
            TypeNode::Ref {
                kind: "block".into(),
                chain: true
            }
        );
        assert_eq!(
            parse_type("list<ref(item)>").unwrap(),
            TypeNode::List(Box::new(TypeNode::Ref {
                kind: "item".into(),
                chain: false
            }))
        );
        assert!(parse_type("wat").is_err());
    }

    #[test]
    fn colours_accept_both_notations() {
        assert_eq!(
            color("#a0f", &loc()).unwrap(),
            Value::Color([0xaa, 0x00, 0xff])
        );
        assert_eq!(
            color("#49b077", &loc()).unwrap(),
            Value::Color([0x49, 0xb0, 0x77])
        );
        assert_eq!(
            color("12, 34 56", &loc()).unwrap(),
            Value::Color([12, 34, 56])
        );
        assert!(color("#12345", &loc()).is_err());
        assert!(color("1 2", &loc()).is_err());
        assert!(color("1 2 300", &loc()).is_err());
    }

    #[test]
    fn chance_accepts_percent_and_fraction_but_not_out_of_range() {
        let regs = HashMap::new();
        let c = ctx_of(&regs);
        let f = Field::new("chance");
        let got = coerce(&TypeNode::Chance, "6%", &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Float(0.06));
        assert_eq!(
            coerce(&TypeNode::Chance, "0.5", &f, &loc(), &c).unwrap(),
            Value::Float(0.5)
        );
        assert!(coerce(&TypeNode::Chance, "150%", &f, &loc(), &c).is_err());
    }

    #[test]
    fn a_single_number_widens_to_a_range() {
        let regs = HashMap::new();
        let c = ctx_of(&regs);
        let f = Field::new("range");
        assert_eq!(
            coerce(&TypeNode::Range, "1..3", &f, &loc(), &c).unwrap(),
            Value::Range(1.0, 3.0)
        );
        assert_eq!(
            coerce(&TypeNode::Range, "4", &f, &loc(), &c).unwrap(),
            Value::Range(4.0, 4.0)
        );
    }

    #[test]
    fn infinity_survives_as_a_float() {
        let regs = HashMap::new();
        let c = ctx_of(&regs);
        let f = Field::new("float");
        let v = coerce(&TypeNode::Float, "inf", &f, &loc(), &c).unwrap();
        assert_eq!(v, Value::Float(f64::INFINITY));
        let v = coerce(&TypeNode::Float, "-Infinity", &f, &loc(), &c).unwrap();
        assert_eq!(v, Value::Float(f64::NEG_INFINITY));
    }

    #[test]
    fn a_preference_chain_takes_the_first_id_that_exists() {
        let mut regs = HashMap::new();
        regs.insert("block".to_string(), HashSet::from(["stone".to_string()]));
        let c = ctx_of(&regs);
        let f = Field::new("ref?(block)");
        let got = reference("block", true, "mud|stone", &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Str("stone".into()));
        assert!(c.warnings.borrow().is_empty());
    }

    #[test]
    fn a_chain_with_nothing_existing_warns_and_takes_the_last() {
        let mut regs = HashMap::new();
        regs.insert("block".to_string(), HashSet::new());
        let c = ctx_of(&regs);
        let f = Field::new("ref?(block)");
        let got = reference("block", true, "mud|slush", &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Str("slush".into()));
        assert_eq!(c.warnings.borrow().len(), 1);
    }

    #[test]
    fn a_strict_ref_to_an_unknown_id_is_fatal() {
        let mut regs = HashMap::new();
        regs.insert("block".to_string(), HashSet::new());
        let c = ctx_of(&regs);
        let f = Field::new("ref(block)");
        assert!(reference("block", false, "mud", &f, &loc(), &c).is_err());
    }

    #[test]
    fn pairs_respect_quotes_and_reject_junk_only_input() {
        let p = split_pairs(r#"item=stone count=1..3 name="two words""#, &loc()).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[2], ("name".to_string(), "two words".to_string()));
        assert!(split_pairs("no pairs here", &loc()).is_err());
        // An empty attribute list is legal — the body carries everything.
        assert!(split_pairs("", &loc()).unwrap().is_empty());
    }
}

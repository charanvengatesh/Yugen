//! The declarative schema system.
//!
//! A schema is data: a record of dotted field name -> descriptor. It is the
//! single source of truth for three things that must never disagree —
//! validation of the authored TOML, the shape of the emitted Rust struct, and
//! which flat tables get built. Adding a field to a schema is therefore the
//! whole job of adding a field to the game.
//!
//! Types are written in the notation of FORMAT.md §3 (`int`, `chance`,
//! `list<ref(block)>`, `enum(a|b|c)`) and parsed here, so the doc and the code
//! cannot drift.

use crate::error::{ContentError, Loc, Result};
use crate::toml_in::{RawRecord, RawValue};
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

    // `ref?(kind)` is a preference chain: `cap = ["dirt", "mud", "stone"]` takes
    // the first id that exists, which keeps content additive.
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

/// One flat array indexed by code (FORMAT.md §7).
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

/// A packed pair matrix, `NAME[a * COUNT + b]` (FORMAT.md §7).
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
    /// FORMAT.md §3 type notation.
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
    /// Field values used to synthesise the placeholder def for a tombstoned id
    /// (`content/ids.lock.json`), spelled as TOML source — `"\"empty\""`, `"[255, 0, 255]"`
    /// — so they are coerced by exactly the same code path as an authored file
    /// and cannot drift from it. It only has to satisfy the schema's required
    /// fields; nothing should ever be looking at it, but old saves still index it.
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
        let loc = &rec.fields.get(key).unwrap().loc;
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
    occ: Option<&RawValue>,
    def: &Def,
    rec: &RawRecord,
    ctx: &ResolveCtx,
) -> Result<Option<Value>> {
    let node = parse_type(&field.ty)?;

    if node == TypeNode::Record {
        let Some(raw) = occ else {
            return Ok(apply_default(field, def));
        };
        // Both spellings of FORMAT.md §4 — an inline array of inline tables and
        // a run of `[[id.key]]` blocks — parse to exactly this, which is why the
        // choice between them is purely about whether an entry has a body.
        let Some(entries) = raw.value.as_array() else {
            bail!(&raw.loc, "'{key}' must be an array of tables");
        };
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            out.push(resolve_sub_record(key, field, entry, &raw.loc, ctx)?);
        }
        return Ok(Some(Value::Records(out)));
    }

    let Some(raw) = occ else {
        let dflt = apply_default(field, def);
        if dflt.is_none() && field.required {
            bail!(&rec.loc, "{} '{}' is missing '{key}'", schema.kind, rec.id);
        }
        return Ok(dflt);
    };

    if node == TypeNode::Text {
        let Some(body) = raw.value.as_str() else {
            bail!(&raw.loc, "'{key}' needs a ''' body");
        };
        return Ok(Some(Value::Text(text_lines(body))));
    }

    let value = coerce(&node, &raw.value, field, &raw.loc, ctx)?;
    if let Some(check) = &field.check
        && let Some(bad) = check(&value)
    {
        bail!(&raw.loc, "'{key}': {bad}");
    }
    Ok(Some(value))
}

/// A `text` body: one entry per line of a TOML literal multiline string.
///
/// TOML has already dropped the newline immediately after the opening `'''`, so
/// the only one left to drop is the one the closing `'''` sits on — plus any
/// blank lines before it. Blank lines inside a body are FRAME SEPARATORS, and a
/// separator at the end separates nothing; the old heredoc trimmed them the same
/// way, which is why every frame count in the parity snapshot still matches.
fn text_lines(body: &str) -> Vec<String> {
    let mut lines: Vec<String> = body.split('\n').map(str::to_string).collect();
    while lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    lines
}

/// One `record[]` entry: `{ item = "stone_chunk", count = [1, 3], chance = 0.5 }`,
/// or the same table written as a `[[id.drop]]` block.
///
/// A sub-field declared `text` is the entry's multiline body — sprite `frames`
/// is the only shape in the tree — and is the reason the table-array spelling
/// exists at all. Every other sub-field is validated with `enum`, `required` and
/// `default` behaving exactly as they do on a top-level field, which is the
/// point: an element is a record, not a blob of attributes.
///
/// Sub-fields deliberately do NOT run their `check` constraint, matching what
/// the DSL front end did — the constraint is declared for top-level fields and
/// running it here would newly reject content the snapshot says is valid.
fn resolve_sub_record(
    key: &str,
    field: &Field,
    raw: &toml::Value,
    loc: &Loc,
    ctx: &ResolveCtx,
) -> Result<Def> {
    let Some(fields) = &field.fields else {
        bail!("'{key}' is record[] but declares no fields");
    };
    let Some(table) = raw.as_table() else {
        bail!(loc, "'{key}' entries must be tables");
    };
    let mut out = Def::new();

    for k in table.keys() {
        if !fields.iter().any(|(name, _)| name == k) {
            bail!(loc, "unknown '{key}' attribute '{k}'");
        }
    }

    for (k, sub) in fields {
        let node = parse_type(&sub.ty)?;
        let given = table.get(k);

        if node == TypeNode::Text {
            match given {
                None => {
                    let dflt = apply_default(sub, &out);
                    if dflt.is_none() && sub.required {
                        bail!(loc, "'{key}' is missing its '{k}' ''' body");
                    }
                    if let Some(d) = dflt {
                        out.insert(k.clone(), d);
                    }
                }
                Some(v) => {
                    let Some(body) = v.as_str() else {
                        bail!(loc, "'{key}' attribute '{k}' must be a ''' string");
                    };
                    out.insert(k.clone(), Value::Text(text_lines(body)));
                }
            }
            continue;
        }

        let Some(v) = given else {
            let dflt = apply_default(sub, &out);
            if dflt.is_none() && sub.required {
                bail!(loc, "'{key}' is missing '{k}'");
            }
            if let Some(d) = dflt {
                out.insert(k.clone(), d);
            }
            continue;
        };
        out.insert(k.clone(), coerce(&node, v, sub, loc, ctx)?);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Scalar coercion
// ---------------------------------------------------------------------------

/// What FORMAT.md calls a value, for a type-mismatch message.
fn spelling(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::Integer(_) => "an integer",
        toml::Value::Float(_) => "a float",
        toml::Value::Boolean(_) => "a bool",
        toml::Value::String(_) => "a string",
        toml::Value::Array(_) => "an array",
        toml::Value::Table(_) => "a table",
        toml::Value::Datetime(_) => "a datetime",
    }
}

fn wanted(v: &toml::Value, want: &str, loc: &Loc) -> ContentError {
    err!(loc, "expected {want}, got {}", spelling(v))
}

/// The numeric reading of a TOML scalar. An integer widens to a float, because
/// `fps = 8` and `fps = 8.0` are the same tuning decision written two ways.
fn num(v: &toml::Value, loc: &Loc) -> Result<f64> {
    match v {
        toml::Value::Integer(n) => Ok(*n as f64),
        toml::Value::Float(f) => Ok(*f),
        other => Err(wanted(other, "a number", loc)),
    }
}

/// Validate one typed TOML value against one schema type (FORMAT.md §3).
///
/// This is where the format stops being syntax and starts being content: every
/// range check, closed variant set and compile-time reference is enforced here
/// and nowhere else, so a value that reaches `Value` has already been proven to
/// mean something.
fn coerce(
    node: &TypeNode,
    raw: &toml::Value,
    field: &Field,
    loc: &Loc,
    ctx: &ResolveCtx,
) -> Result<Value> {
    match node {
        TypeNode::Int => match raw {
            toml::Value::Integer(n) => Ok(Value::Int(*n)),
            // `3.0` is a whole number spelled as a float; `3.5` is a mistake.
            toml::Value::Float(f) if f.fract() == 0.0 && f.is_finite() => Ok(Value::Int(*f as i64)),
            toml::Value::Float(f) => Err(err!(loc, "'{f}' is not an integer")),
            other => Err(wanted(other, "an int", loc)),
        },
        // Infinity is meaningful here — an unbreakable block's hardness — and
        // TOML spells it `inf`, so nothing special is needed to carry it.
        TypeNode::Float => Ok(Value::Float(num(raw, loc)?)),
        TypeNode::Bool => match raw {
            toml::Value::Boolean(b) => Ok(Value::Bool(*b)),
            other => Err(wanted(other, "a bool", loc)),
        },
        TypeNode::Str => match raw {
            toml::Value::String(s) => Ok(Value::Str(s.clone())),
            other => Err(wanted(other, "a string", loc)),
        },
        TypeNode::Enum(values) => {
            let Some(v) = raw.as_str() else {
                return Err(wanted(raw, "a string", loc));
            };
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
            let v = num(raw, loc)?;
            if !(0.0..=1.0).contains(&v) {
                bail!(loc, "chance '{v}' is outside 0..1");
            }
            Ok(Value::Float(v))
        }
        TypeNode::Range => match raw {
            // A bare number widens to `[n, n]`, so "exactly one" needs no
            // ceremony and every consumer still reads a pair.
            toml::Value::Integer(_) | toml::Value::Float(_) => {
                let n = num(raw, loc)?;
                Ok(Value::Range(n, n))
            }
            toml::Value::Array(a) if a.len() == 2 => {
                Ok(Value::Range(num(&a[0], loc)?, num(&a[1], loc)?))
            }
            toml::Value::Array(a) => Err(err!(loc, "a range is [min, max], got {} items", a.len())),
            other => Err(wanted(other, "a range", loc)),
        },
        TypeNode::Color => color(raw, loc),
        TypeNode::Ref { kind, chain } => reference(kind, *chain, raw, field, loc, ctx),
        TypeNode::List(of) => {
            let Some(items) = raw.as_array() else {
                return Err(wanted(raw, "an array", loc));
            };
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(coerce(of, it, field, loc, ctx)?);
            }
            Ok(Value::List(out))
        }
        TypeNode::Text => Err(err!(loc, "text fields need a ''' body")),
        TypeNode::Record => Err(err!(loc, "record[] fields need an array of tables")),
    }
}

/// `[96, 92, 84]` or `"#605c54"` — three channels, two spellings.
///
/// The hex form survived the move off the DSL because it is what an artist
/// pastes out of a colour picker, and the triple survived because it is what a
/// value tuned by hand against its neighbours looks like.
fn color(raw: &toml::Value, loc: &Loc) -> Result<Value> {
    if let Some(t) = raw.as_str() {
        let Some(hex) = t.strip_prefix('#') else {
            bail!(loc, "'{t}' is not #rgb or #rrggbb");
        };
        let d: Vec<char> = hex.chars().collect();
        let hx = |s: &str| u8::from_str_radix(s, 16).ok();
        let mut out = [0u8; 3];
        if d.len() == 3 {
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

    let Some(parts) = raw.as_array() else {
        return Err(wanted(raw, "a colour", loc));
    };
    if parts.len() != 3 {
        bail!(loc, "colour needs 3 channels: {} given", parts.len());
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

/// A `ref(kind)` is one id; a `ref?(kind)` is a preference chain, written as an
/// array, that takes the first id which exists.
///
/// The chain is what keeps content additive: a record may name a material a
/// later pass will add and degrade gracefully until it does. A single-candidate
/// chain stays a plain string, because most of them have only one candidate and
/// wrapping those in brackets would be noise.
fn reference(
    kind: &str,
    chain: bool,
    raw: &toml::Value,
    field: &Field,
    loc: &Loc,
    ctx: &ResolveCtx,
) -> Result<Value> {
    let candidates: Vec<&str> = match raw {
        toml::Value::String(s) => vec![s.as_str()],
        toml::Value::Array(a) if chain && !a.is_empty() => {
            let mut out = Vec::with_capacity(a.len());
            for c in a {
                let Some(s) = c.as_str() else {
                    return Err(wanted(c, "an id", loc));
                };
                out.push(s);
            }
            out
        }
        toml::Value::Array(_) if chain => bail!(loc, "a preference chain cannot be empty"),
        other => return Err(wanted(other, "an id", loc)),
    };
    // How the value is spelled back in a message: one id bare, a chain as the
    // array it was written as.
    let text = if candidates.len() == 1 {
        candidates[0].to_string()
    } else {
        format!("{candidates:?}")
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

    /// `v = <toml>` — the one-value document every test below coerces.
    fn v(src: &str) -> toml::Value {
        let t: toml::Table = format!("v = {src}").parse().unwrap();
        t["v"].clone()
    }

    #[test]
    fn colours_accept_both_notations() {
        assert_eq!(
            color(&v("\"#a0f\""), &loc()).unwrap(),
            Value::Color([0xaa, 0x00, 0xff])
        );
        assert_eq!(
            color(&v("\"#49b077\""), &loc()).unwrap(),
            Value::Color([0x49, 0xb0, 0x77])
        );
        assert_eq!(
            color(&v("[12, 34, 56]"), &loc()).unwrap(),
            Value::Color([12, 34, 56])
        );
        assert!(color(&v("\"#12345\""), &loc()).is_err());
        assert!(color(&v("[1, 2]"), &loc()).is_err());
        assert!(color(&v("[1, 2, 300]"), &loc()).is_err());
    }

    #[test]
    fn chance_is_a_fraction_and_is_range_checked() {
        let regs = HashMap::new();
        let c = ctx_of(&regs);
        let f = Field::new("chance");
        assert_eq!(
            coerce(&TypeNode::Chance, &v("0.06"), &f, &loc(), &c).unwrap(),
            Value::Float(0.06)
        );
        assert_eq!(
            coerce(&TypeNode::Chance, &v("1"), &f, &loc(), &c).unwrap(),
            Value::Float(1.0)
        );
        assert!(coerce(&TypeNode::Chance, &v("1.5"), &f, &loc(), &c).is_err());
    }

    #[test]
    fn a_single_number_widens_to_a_range() {
        let regs = HashMap::new();
        let c = ctx_of(&regs);
        let f = Field::new("range");
        assert_eq!(
            coerce(&TypeNode::Range, &v("[1, 3]"), &f, &loc(), &c).unwrap(),
            Value::Range(1.0, 3.0)
        );
        assert_eq!(
            coerce(&TypeNode::Range, &v("4"), &f, &loc(), &c).unwrap(),
            Value::Range(4.0, 4.0)
        );
        assert!(coerce(&TypeNode::Range, &v("[1, 2, 3]"), &f, &loc(), &c).is_err());
    }

    #[test]
    fn infinity_survives_as_a_float() {
        let regs = HashMap::new();
        let c = ctx_of(&regs);
        let f = Field::new("float");
        let got = coerce(&TypeNode::Float, &v("inf"), &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Float(f64::INFINITY));
        let got = coerce(&TypeNode::Float, &v("-inf"), &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Float(f64::NEG_INFINITY));
    }

    #[test]
    fn a_type_mismatch_is_rejected_rather_than_guessed_at() {
        let regs = HashMap::new();
        let c = ctx_of(&regs);
        assert!(coerce(&TypeNode::Int, &v("3.5"), &Field::new("int"), &loc(), &c).is_err());
        assert!(
            coerce(
                &TypeNode::Bool,
                &v("\"yes\""),
                &Field::new("bool"),
                &loc(),
                &c
            )
            .is_err()
        );
        assert!(coerce(&TypeNode::Str, &v("7"), &Field::new("string"), &loc(), &c).is_err());
        // An int is a legal spelling of a whole float, and the reverse.
        assert_eq!(
            coerce(&TypeNode::Float, &v("8"), &Field::new("float"), &loc(), &c).unwrap(),
            Value::Float(8.0)
        );
        assert_eq!(
            coerce(&TypeNode::Int, &v("8.0"), &Field::new("int"), &loc(), &c).unwrap(),
            Value::Int(8)
        );
    }

    #[test]
    fn a_preference_chain_takes_the_first_id_that_exists() {
        let mut regs = HashMap::new();
        regs.insert("block".to_string(), HashSet::from(["stone".to_string()]));
        let c = ctx_of(&regs);
        let f = Field::new("ref?(block)");
        let got = reference("block", true, &v("[\"mud\", \"stone\"]"), &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Str("stone".into()));
        assert!(c.warnings.borrow().is_empty());
    }

    #[test]
    fn a_chain_with_nothing_existing_warns_and_takes_the_last() {
        let mut regs = HashMap::new();
        regs.insert("block".to_string(), HashSet::new());
        let c = ctx_of(&regs);
        let f = Field::new("ref?(block)");
        let got = reference("block", true, &v("[\"mud\", \"slush\"]"), &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Str("slush".into()));
        assert_eq!(c.warnings.borrow().len(), 1);
    }

    #[test]
    fn a_single_candidate_chain_stays_a_plain_string() {
        let mut regs = HashMap::new();
        regs.insert("block".to_string(), HashSet::from(["stone".to_string()]));
        let c = ctx_of(&regs);
        let f = Field::new("ref?(block)");
        let got = reference("block", true, &v("\"stone\""), &f, &loc(), &c).unwrap();
        assert_eq!(got, Value::Str("stone".into()));
    }

    #[test]
    fn a_strict_ref_to_an_unknown_id_is_fatal() {
        let mut regs = HashMap::new();
        regs.insert("block".to_string(), HashSet::new());
        let c = ctx_of(&regs);
        let f = Field::new("ref(block)");
        assert!(reference("block", false, &v("\"mud\""), &f, &loc(), &c).is_err());
    }

    #[test]
    fn a_body_loses_only_its_closing_blank_lines() {
        assert_eq!(text_lines(".4\n32\n\n1.\n"), [".4", "32", "", "1."]);
        // Trailing spaces are transparent cells, not formatting.
        assert_eq!(text_lines("ab  \ncd\n"), ["ab  ", "cd"]);
        assert!(text_lines("").is_empty());
    }
}

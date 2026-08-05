//! The resolved value tree.
//!
//! The TS compiler resolved a record into `Record<string, unknown>` and let
//! duck typing carry it to the emitter. Rust needs the shape named, so this is
//! that shape — and having it named is the reason the emitter can decide a
//! field's Rust type from the value rather than re-deriving it from the schema.

use std::fmt;

/// A fully resolved record: nested, ordered, ready to be emitted.
///
/// Ordered because emission order is the schema's field declaration order, and
/// that order is what makes the generated source byte-stable across runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Def {
    entries: Vec<(String, Value)>,
}

impl Def {
    pub fn new() -> Self {
        Def::default()
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Reach into a group: `def.path("heat", "conduct")`.
    pub fn path(&self, group: &str, leaf: &str) -> Option<&Value> {
        match self.get(group) {
            Some(Value::Group(g)) => g.get(leaf),
            _ => None,
        }
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        if let Some((_, v)) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            *v = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -- typed accessors, for schema table callbacks ------------------------
    //
    // These are total on purpose: a table callback that asks for a field the
    // record does not carry gets the zero value, exactly as the TS callbacks got
    // `undefined` and coerced it. A missing REQUIRED field is impossible here
    // because resolution already failed.

    pub fn str_of(&self, key: &str) -> &str {
        match self.get(key) {
            Some(Value::Str(s)) => s,
            _ => "",
        }
    }

    pub fn num(&self, key: &str) -> f64 {
        self.get(key).and_then(Value::as_num).unwrap_or(0.0)
    }

    pub fn num_or(&self, key: &str, fallback: f64) -> f64 {
        self.get(key).and_then(Value::as_num).unwrap_or(fallback)
    }

    pub fn flag(&self, key: &str) -> bool {
        matches!(self.get(key), Some(Value::Bool(true)))
    }

    pub fn group_num(&self, group: &str, leaf: &str) -> f64 {
        self.path(group, leaf)
            .and_then(Value::as_num)
            .unwrap_or(0.0)
    }

    pub fn group_str(&self, group: &str, leaf: &str) -> &str {
        match self.path(group, leaf) {
            Some(Value::Str(s)) => s,
            _ => "",
        }
    }

    pub fn list_of(&self, key: &str) -> &[Value] {
        match self.get(key) {
            Some(Value::List(v)) => v,
            _ => &[],
        }
    }

    pub fn records_of(&self, key: &str) -> &[Def] {
        match self.get(key) {
            Some(Value::Records(v)) => v,
            _ => &[],
        }
    }

    /// Every string in a `list<...>` field, for tag and band bitfields.
    pub fn strings_of(&self, key: &str) -> Vec<&str> {
        self.list_of(key)
            .iter()
            .filter_map(|v| match v {
                Value::Str(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// A string, a `ref(kind)` id, or an unmapped `enum` variant.
    Str(String),
    /// `#rrggbb` or `r g b`, always three 0..=255 channels.
    Color([u8; 3]),
    /// `a..b`, or `n` widened to `[n, n]`.
    Range(f64, f64),
    /// A heredoc body, one entry per line.
    Text(Vec<String>),
    List(Vec<Value>),
    /// One or more `record[]` occurrences.
    Records(Vec<Def>),
    /// A dotted group: `heat.conduct` lands in the `heat` group.
    Group(Def),
}

impl Value {
    /// The numeric reading of a scalar. `true` is 1 so a bool field can feed a
    /// flat table without the schema restating the conversion.
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Float(f) => Some(*f),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Name of the type as FORMAT.md spells it, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::Str(_) => "string",
            Value::Color(_) => "color",
            Value::Range(..) => "range",
            Value::Text(_) => "text",
            Value::List(_) => "list",
            Value::Records(_) => "record[]",
            Value::Group(_) => "group",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{n}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Color([r, g, b]) => write!(f, "{r} {g} {b}"),
            Value::Range(a, b) => write!(f, "{a}..{b}"),
            Value::Text(l) => write!(f, "<{} lines>", l.len()),
            Value::List(v) => {
                let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
                write!(f, "{}", parts.join(" "))
            }
            Value::Records(v) => write!(f, "<{} records>", v.len()),
            Value::Group(g) => write!(f, "<group of {}>", g.len()),
        }
    }
}

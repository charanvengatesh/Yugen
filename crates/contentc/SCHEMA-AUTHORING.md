# Writing a schema in the Rust compiler

Reference for porting a `tools/contentc/schema/*.ts` schema to Rust. The API
lives in `crates/contentc/src/schema.rs` and the value tree in
`crates/contentc/src/value.rs` — read both before starting.

## The shape

```rust
use crate::schema::{ArrayKind, BitConstants, Field, Matrix, Schema, TableCtx, table};
use crate::value::{Def, Value};

pub fn schema() -> Schema {
    Schema {
        kind: "block".into(),
        prefix: "BLOCK".into(),          // BLOCK_IDS, BLOCKS, BLOCK_COUNT, block::STONE
        iface: "BlockDef".into(),        // the emitted struct name
        iface_prefix: "Block".into(),    // BlockHeat, BlockDrop, ...
        fields: vec![ /* (String, Field) in EMISSION ORDER */ ],
        tables: vec![ /* derived HotArrays that read >1 field */ ],
        matrices: vec![],
        constants: vec![],
        // Values spelled as TOML source, coerced by the same code as a real file.
        tombstone: vec![("name".into(), "\"(removed)\"".into()), /* ... */],
    }
}
```

## Field builders

`Field::new("<FORMAT.md type notation>")` then chain:

| TS | Rust |
|---|---|
| `doc: "..."` | `.doc("...")` |
| `required: true` | `.required()` |
| `lenient: true` | `.lenient()` |
| `default: 12` | `.default_int(12)` |
| `default: 0.5` | `.default_float(0.5)` |
| `default: false` | `.default_bool(false)` |
| `default: "flat"` | `.default_str("flat")` |
| `default: (d) => ...` | `.default_fn(\|d\| Some(Value::Str(d.str_of("id").into())))` — return `Option<Value>` |
| `map: STATE` | `.map(&[("empty", 0), ("solid", 1), ...])` |
| `tsAlias: "BlockTexture"` | `.alias("BlockTexture")` |

### Enums the game reads as a number

An `enum(...)` field emits a named Rust enum, and a field typed as one arrives
already typed on the def struct. A **mapped** enum does not: `.map(...)` turns
the variant into a code, so the def holds a number.

That is the right default — an enum nothing names would be dead — right up
until the field is also `hot`. Then it lands in a flat `u8` array, and whatever
loop reads that array has to turn the byte back into a variant. There is only
one place that can happen, and before `.alias(...)` opted into it, that place
was a hand-written copy of the same variant list somewhere in `yugen-core`,
with the codes load-bearing on both sides and nothing checking they agreed.

**Naming a mapped enum is how a schema says the game meets this one as a
number.** `contentc` then emits the enum *and* a total
`from_code(u8) -> Option<T>`, and the consumer imports both instead of restating
them:

```rust
Field::new(&enum_of(KIND))
    .alias("FeatureKind")   // <- emits the enum and its from_code
    .map(KIND)
    .required()
    .hot(vec![table("FEAT_KIND", ArrayKind::U8, 0.0, "Generator code.", ...)])
```

`from_code` returns `Option` and never defaults. A code this build does not know
is a variant added by content it has not caught up with; what to do about that
differs per field — a placement class falls back to the safest lattice, a
generator grows nothing — so the decision stays with the caller.
| `tsElement` + `fields` | `.element("BlockDrop", vec![("item".into(), Field::new("ref(item)").required()), ...])` |
| `check: (v) => msg \| undefined` | `.check(\|v\| ...Option<String>)`, or `.between(0.0, 255.0)` for a range |
| `hot: true` + `hotArray` | `.hot(vec![table(...)])` — always a `Vec`, even for one |

Type notation is FORMAT.md §3: `int`, `float`, `bool`, `string`, `color`,
`range`, `chance`, `text`, `record[]`, `list<T>`, `enum(a|b|c)`, `ref(kind)`,
`ref?(kind)`. How each is SPELLED in a content file is FORMAT.md's table; the
descriptor only names the type.

Two of them constrain how a record can be laid out, because TOML does:

- `text` is a `'''` literal multiline string. Declare it on a `record[]`
  sub-field and every entry has to be written as a `[[id.key]]` block, which
  TOML requires to come after the record's ordinary keys.
- `record[]` is an array of tables — `key = [{ ... }]` inline, or a run of
  `[[id.key]]` blocks. The compiler accepts both and cannot tell them apart, so
  the choice is purely about whether an entry carries a body.

## Tables

```rust
table("MAT_DENSITY", ArrayKind::F32, 0.0, "doc", |d, _ctx| Some(d.num("density")))
```

Signature: `table(name, array_kind, fill, doc, |def, ctx| -> Option<f64>)`.
`None` leaves `fill` in place. `ArrayKind` is `U8 | U16 | U32 | F32`
(TS `Uint8Array` / `Uint16Array` / `Uint32Array` / `Float32Array`).

**Note the fill is now a required positional argument**, where TS had it last
and optional. TS `fill` omitted means `0.0`.

## Reading the def inside a callback

`Def` accessors are total — a missing field reads as the zero value, matching
how the TS callbacks got `undefined` and coerced it:

| Need | Call |
|---|---|
| a number | `d.num("density")`, `d.num_or("density", 1.0)` |
| a string / ref id | `d.str_of("meltInto")` |
| a bool | `d.flag("flammable")` |
| inside a group | `d.group_num("heat", "conduct")`, `d.group_str("heat", "meltInto")` |
| does a group/field exist | `d.contains("heat")`, `d.path("heat", "meltAt").is_some()` |
| a `list<...>` of strings | `d.strings_of("tags")` -> `Vec<&str>` |
| a `record[]` | `d.records_of("drops")` -> `&[Def]`, each a `Def` |
| a colour channel | `match d.get("color") { Some(Value::Color(c)) => c[0], _ => 0 }` |
| a range half | `match d.get("size") { Some(Value::Range(a, _)) => *a, _ => 0.0 }` |

`Option`-returning `d.get(...)` / `d.path(...)` is what to use when the TS code
tested `=== undefined` to decide whether a table slot stays at `fill`.

## TableCtx

| TS | Rust |
|---|---|
| `ctx.code(id)` | `ctx.code(id)` -> `f64` |
| `ctx.foreignCode(kind, id)` | `ctx.foreign_code(kind, id)` -> `f64` |
| `ctx.NEVER` | `crate::schema::NEVER` (`0xffff`) |
| `ctx.bit(set, value)` | `ctx.bit(set, value)` -> `f64` (bit INDEX) |
| a fold of `1 << bit(...)` | `ctx.mask("TAG", &["rock", "diggable"])` -> `f64` |

`ctx.bit` on an undeclared value records an error the emitter raises later, so a
callback still returns a number rather than panicking.

## Bit constants

```rust
BitConstants {
    name: "TAG".into(),
    doc: Some("...".into()),
    values: TAGS.iter().map(|s| s.to_string()).collect(),
}
```

Emitted as a `bitflags!` type, so `Tag::ROCK` and `Tag::DIGGABLE` exist and
`MAT_TAGS[c] & Tag::ROCK.bits()` works. Max 32 values.

## Matrices

```rust
Matrix {
    name: "GROW_ONTO".into(),
    doc: Some("...".into()),
    row: Box::new(|d, _ctx| {
        let ids = d.strings_of("growth.onto");
        if ids.is_empty() { None } else { Some(ids.iter().map(|s| s.to_string()).collect()) }
    }),
}
```

## Rules

- **Preserve field order exactly.** It drives default-callback visibility and
  emission order.
- **Preserve every doc string.** They are the designer-facing surface and the
  tuning index reads them.
- **Preserve every table name verbatim** (`MAT_DENSITY`, `ITEM_STACK`, ...). Hot
  paths import these by name.
- Keep the explanatory comments from the TS file. They record why a constant or
  an ordering exists and are the highest-value thing in the port.
- Do not invent fields, tables or constants that the TS schema does not declare.

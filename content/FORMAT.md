# Content format

Authored content is TOML. One file per group of records, one directory per kind:

| Directory | What it declares |
|---|---|
| `blocks/` | materials: colour, state, hardness, damage, surface, light |
| `items/` | inventory items, weapons, tools, recipes |
| `mobs/` | the bestiary: brains, bodies, stats, spawn bands |
| `sprites/` | pixel art: palettes, frames, animation sequences |
| `structures/` | template-driven landmarks |
| `worldgen/` | parametric worldgen features |

The kind comes from the directory, not from anything in the file. Numeric ids
are pinned in `ids.lock.json` so a saved world does not change meaning when a
new record is added; ids are assigned once and never reused, and a deleted
record leaves a tombstone.

`cargo run -p contentc` compiles `content/` into `crates/yugen-data/src/`.
`cargo run -p contentc -- --check` is the gate that fails if the generated
tables are stale.

This file says how to author. **`PALETTE.md` says what to author**, for anything
with a colour in it: the ramps, the tonal bands, the accent allow-list, and the
arithmetic that turns `color`, `colorVar` and `edge` into a rendered pixel. A
block or sprite that compiles cleanly can still be wrong, and that is the file
that says how.

---

## 1. A record is a top-level table

The table name is the record's authoring id.

```toml
[stone]
name = "Stone"
state = "solid"
color = [96, 92, 84]
hardness = 3
```

Ids are the stable name that `ids.lock.json` pins and that every cross-reference
uses. Renaming one is a breaking change; adding one is not.

Unknown keys are a compile error, never a silent drop. A typo'd key that is
quietly ignored is a material that stops behaving the way its file says it does.

## 2. Groups are dotted keys

```toml
heat.conduct = 56
heat.meltAt = 50
heat.meltInto = "glass"
```

A group is emitted only when the source sets at least one of its keys, and group
defaults are NOT injected when the group is absent — a material with no `heat`
keys has no heat group at all.

TOML's own `[stone.heat]` sub-table syntax means the same thing and is accepted,
but the dotted form keeps a record readable as one block.

## 3. Types

| Type | Written as | Notes |
|---|---|---|
| int | `16` | must be a whole number |
| float | `0.5`, `inf`, `-inf` | `inf` is meaningful — an unbreakable block's hardness |
| bool | `true` / `false` | |
| string | `"Stone"` | |
| enum | `"layered"` | must be one of the declared variants |
| color | `[96, 92, 84]` or `"#605c54"` | three channels 0..255, or 3/6-digit hex |
| range | `[1, 3]` or `4` | a bare number widens to `[n, n]` |
| chance | `0.06` | a fraction in 0..1 |
| list | `["rock", "diggable"]` | |
| text | `'''…'''` | a literal multiline string — see §5 |
| record[] | array of tables — see §4 | |

### References

A `ref` field is the id of a record of another kind, and is **checked at compile
time**:

```toml
heat.meltInto = "glass"      # must be a real block id
```

A *preference chain* is an array, and takes the first id that exists. That is
what keeps content additive — a record can name a material that a later pass
will add, and degrade gracefully until it does:

```toml
accent = ["ironOre", "coalOre", "stone"]
```

If none of them exist the build warns and uses the last, rather than failing.

## 4. Lists of records

Two spellings, both meaning the same thing. Use the inline form for short
entries:

```toml
drop = [{ item = "stone_chunk", count = 1 }]
craft = [
    { in = ["wood_log", "coal"], n = [1, 1], out = 4, station = "hand" },
]
```

and the table-array form when an entry carries a multiline body:

```toml
[[player.seq]]
state = "run"
mode = "phase"
frames = '''
.4
32
1.
'''
```

A table array has to come after the record's ordinary keys — that is TOML's
rule, not ours.

## 5. Pixel art and glyph bodies

Sprite frames and structure templates are **literal multiline strings**, opened
and closed with `'''`. Literal, not basic (`"""`), for three reasons that all
matter here:

- **`#` is a glyph, not a comment.** Structure legends and sprite palettes use
  it, and inside `'''` it is data.
- **Trailing whitespace is data.** A trailing run of blank glyphs is part of the
  art. TOML preserves it inside a literal string; do not let an editor strip it.
- **Backslashes are literal.** No escape processing runs, so art means exactly
  what it looks like.

A leading newline immediately after the opening `'''` is trimmed by TOML, which
is why the art starts on the line below it.

Blank lines inside a frames body separate FRAMES — that is the filmstrip
convention, not formatting.

```toml
[[icon_pick_copper.seq]]
frames = '''
232
.1.
..1
'''
```

Within a frame body: `.` and `0` are transparent, `1`-`9` index the palette, and
each row must be exactly `cellsW` characters wide with exactly `cellsH` rows per
frame. Art is authored facing right; left is a draw-time flip.

## 6. Comments

`#` to end of line, standard TOML. Comments are the most valuable thing in these
files — they record why a number is what it is — so the compiler never
round-trips a file and never rewrites one.

---

## 7. What the compiler emits

One Rust module per kind under `crates/yugen-data/src/`, containing:

- a `struct` per record shape, plus an `enum` per closed variant set and a
  `bitflags` type per declared bit set;
- `<PREFIX>S`, every def, index == code, tombstones included;
- `<PREFIX>_IDS` and a `<prefix>::SCREAMING_NAME` constant per id, so
  `block::STONE` is a compile-time constant;
- flat arrays indexed by code for every field marked hot — `MAT_DENSITY`,
  `ITEM_STACK`, `MOB_WEIGHT`. A per-cell loop reads those, never the def table;
- packed pair matrices with an inline accessor.

**Never edit the generated modules.** Edit `content/` and recompile.

Which fields are hot, what the flat arrays are called, what the defaults and
constraints are, and which references are checked — all of that is declared in
`crates/contentc/src/schemas/`. Adding a field to the game is adding a row there.

---

## 8. History

Content was originally authored in a bespoke line-oriented DSL with its own
lexer and parser (`FORMAT-dsl-legacy.md`). TOML replaced it because it is a
standard the tooling already understands, and because the DSL's one genuinely
novel feature — a heredoc that keeps `#` and trailing whitespace as data — turns
out to be exactly what TOML's literal multiline string already does.

What did NOT change is everything the compiler does after parsing: the schema
descriptors, compile-time reference checking, the defaults and constraints, the
id lock and its tombstones, and the flat-array codegen. Those were always the
valuable part; the syntax never was.

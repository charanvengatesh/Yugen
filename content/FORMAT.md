# GodGame content format — the `contentc` contract

**Status: normative.** Everything in `content/` is authored in the plain-text DSL
described here and compiled to TypeScript by `tools/contentc/` at build time.
Nothing parses these files at runtime — the shipped bundle contains only
generated, pre-baked typed arrays. That is the whole point: authoring stays in
flat readable text files, and the game pays zero parse cost.

---

## 1. Why a custom format and not JSON/TS

- JSON has no comments, no heredocs for pixel art, and quotes every key.
- TS data modules can't be validated by a schema, can't auto-assign stable
  numeric codes, and can't emit the flat `Uint8Array` tables the hot loops want
  without doing the work at startup on every page load.
- The DSL is a strict superset of "one key per line", so a `.block` file diffs
  cleanly and a designer can add a material without touching a `.ts` file.

---

## 2. Lexical rules

Line-oriented. UTF-8. LF or CRLF.

```
# comment to end of line — only when '#' is the first non-space char,
# or is preceded by whitespace and is not inside a quoted string.

@block stone            # record header: '@' <kind> <id>

id        stone         # key + value; key is the first token,
name      "Stone"       # value is the rest of the line, trimmed
state     solid
color     96 92 84      # whitespace- or comma-separated list
heat.conduct 56         # dotted key = nested field
tags      rock diggable
```

Rules:

1. **Blank lines and comment-only lines are ignored.**
2. **`@<kind> <id>`** opens a record. A file may hold many records. The `id` in
   the header is authoritative; a redundant `id` key must match or it is an error.
3. **`<key> <value>`** — key is `[A-Za-z_][A-Za-z0-9_.]*`. Everything after the
   first run of whitespace, trimmed, is the raw value.
4. **Dotted keys** nest: `heat.conduct 56` → `{ heat: { conduct: 56 } }`.
   Assigning both `heat 4` and `heat.conduct 5` is an error.
5. **Heredoc**: a value ending in a standalone `|` starts a block. Every
   following line indented more than the key line is taken **verbatim** (leading
   indentation stripped to the block's minimum indent, trailing whitespace kept,
   blank lines preserved, `#` never treated as a comment). The block ends at the
   first line whose indent is ≤ the key's indent. Used for sprite art and
   structure templates.
   ```
   art.idle |
     .23
     111
   ```
   Anything before the `|` is the line's **inline value** and is parsed
   normally. On a `record[]` field that is how one element carries both its
   attributes and its body — see rule 6.
   ```
   art.seq  state=run mode=phase |
     .4
     32
     1.

     .4
     32
     .1
   ```
   **Standalone** means the `|` is the entire value or is preceded by
   whitespace. `|` is already meaningful *inside* a value — a `ref?` preference
   chain writes `mat.accent ironOre|coalOre` and a struct legend writes
   `block=goldOre|ironOre|coalOre` — so the separator is what tells the two
   apart, with no escaping and no lookahead. `a|b|` is a chain whose last
   candidate is empty; `a|b |` is a chain followed by a heredoc opener. A
   trailing `# comment` after the `|` is stripped before the block opens.
6. **Repeated key** appends into a list only where the schema says the field is
   a list-of-records (`frame`, `drop`, `layer`). Otherwise it is an error —
   silent last-wins clobbering is how content rots.

   A list-of-records element is written as `key=value` pairs on one line
   (`drop item=stone_chunk count=1..3 chance=50%`). If the element's schema
   declares one sub-field of type `text`, that sub-field is bound from the
   element's heredoc body instead of from a pair — writing it as `rows=…` is an
   error. Every other sub-field is still validated as a pair, so `enum`,
   `required` and `default` behave identically whether or not the element has a
   body. An element may repeat freely; each occurrence is one element, in file
   order.
7. **`+include <path>`** at column 0 splices another content file, resolved
   relative to `content/`. Used for shared palettes.

### Value types (resolved by schema, never by syntax)

| Type | Accepted forms | Notes |
|---|---|---|
| `int` / `float` | `56`, `-1.5`, `1e3` | |
| `bool` | `true` `false` `yes` `no` `on` `off` | |
| `string` | bare rest-of-line, or `"quoted"` | quote only when it has a `#` or leading/trailing space |
| `enum(a\|b\|c)` | bare token | compile error listing valid values on mismatch |
| `color` | `#rrggbb`, `#rgb`, or `r g b` (0–255) | always emitted as `[r,g,b]` |
| `ref(kind)` | a bare id | **validated at compile time** against that kind's registry; unknown id = build failure |
| `list<T>` | whitespace- or comma-separated | `tags rock, diggable` |
| `range` | `3..7` or a single number | emitted as `[min,max]` |
| `chance` | `0.06` or `6%` | emitted as 0..1 float |
| `text` | heredoc | array of lines |

A field marked `ref?` allows a **preference chain**: `cap dirt|mud|stone` picks
the first id that exists. This keeps content additive — referencing a block that
doesn't exist yet degrades instead of breaking the build.

---

## 3. Kinds, and where they live

| Kind | Path | Generates |
|---|---|---|
| `block` | `content/blocks/*.block` | `src/generated/blocks.gen.ts` |
| `item` | `content/items/*.item` | `src/generated/items.gen.ts` |
| `mob` | `content/mobs/*.mob` | `src/generated/mobs.gen.ts` |
| `sprite` | `content/sprites/*.sprite` | `src/generated/sprites.gen.ts` |
| `biome` | `content/biomes/*.biome` | `src/generated/biomes.gen.ts` |
| `layer` | `content/biomes/*.layer` | `src/generated/biomes.gen.ts` |
| `struct` | `content/structures/*.struct` | `src/generated/structs.gen.ts` |
| `feature` | `content/worldgen/*.feature` | `src/generated/worldgen.gen.ts` |
| `recipe` | `content/recipes/*.recipe` | `src/generated/items.gen.ts` |

Schemas live in `tools/contentc/schema/<kind>.ts` as data, and are the single
source of truth for both validation and the emitted TypeScript interface.

---

## 4. Stable numeric ids — `content/ids.lock.json`

Cells store a `Uint16` block code. Codes must never shift, or every saved chunk
in every existing world reinterprets its terrain.

```json
{ "block": { "empty": 0, "stone": 1, "ice": 2 },
  "item":  { "pick_copper": 1 },
  "mob":   { "grubling": 1 } }
```

Compiler rules:

- An id present in the lock keeps its code, forever.
- A new id is assigned the lowest free code and the lock file is **rewritten**
  (and committed).
- An id in the lock but absent from `content/` becomes a **tombstone**: its code
  stays reserved and a placeholder def is emitted so old saves still load.
- `contentc --check` fails if the lock would change — that's the CI gate.

---

## 5. Generated module shape (the API every consumer codes against)

Emission is uniform per kind. For `block`:

```ts
// src/generated/blocks.gen.ts  — GENERATED, DO NOT EDIT
export interface BlockDef { /* derived from schema/block.ts */ }
```

A `record[]` field emits a named element interface alongside the def interface,
whether the field is top-level or lives inside a dotted group. `art.seq`
declared as `record[]` with `tsElement: "SpriteFrames"` produces:

```ts
export interface SpriteFrames {
  state: "idle" | "run" | "jump";   // required enum attribute
  mode: "phase" | "loop";           // defaulted → always present, never optional
  fps: number;                      // required
  rows?: readonly string[];         // the heredoc body
}

export interface SpriteArt {
  cell?: number;
  seq?: readonly SpriteFrames[];
}
```

A sub-field is emitted non-optional when it is `required` **or** has a
`default` — a default is always materialised onto the def, so the `?` would be a
lie. Element interfaces are emitted once per distinct `tsElement` name, so two
fields may deliberately share one element shape. Key order inside the emitted
object literal follows the schema's `fields` declaration order, which is what
keeps a no-op rebuild byte-identical and `--check` meaningful.

```ts
export const BLOCK_IDS: readonly string[];              // index === code
export const BLOCKS: readonly BlockDef[];               // index === code, tombstones included
export const BLOCK_COUNT: number;                       // === BLOCKS.length

/** id -> code. Frozen literal object, so `BLOCK.stone` is a compile-time constant. */
export const BLOCK: { readonly [id: string]: number };

// Flat tables, index === code. Built by the COMPILER, not at startup.
export const MAT_STATE: Uint8Array;
export const MAT_DENSITY: Float32Array;
/* …one per hot-path scalar field declared in the schema… */

/** Pair matrices are emitted base64-packed and inflated once, cheaply. */
export const GROW_ONTO: Uint8Array;   // [grower * BLOCK_COUNT + target]
```

Non-hot-path fields stay on the def objects. **Rule of thumb the compiler
follows:** a schema field tagged `hot: true` gets a flat typed array; everything
else lives only on the def.

Hand-written code **never imports `src/generated/*` directly** except through the
facade modules (`src/sim/materials.ts`, `src/items/registry.ts`,
`src/entities/mobs/MobDefs.ts`, …), which re-export the generated symbols under
their existing names. That keeps the ~40 existing import sites untouched.

---

## 6. Build integration

- `tools/contentc/plugin.ts` is a Vite plugin: compiles on `buildStart`, watches
  `content/**` in dev and triggers HMR on the generated modules.
- `npm run content` runs the compiler standalone.
- `npm run build` = `content` → `tsc --noEmit` → `vite build`.
- Generated files **are committed** so a clean checkout typechecks without a
  pre-step.

---

## 7. Worked examples

### `content/blocks/stone.block`
```
@block stone
name      Stone
state     solid
color     96 92 84
colorVar  16
heat.conduct 56
hardness  3
tags      rock diggable
drop      item=stone_chunk count=1
```

### `content/mobs/slime.mob`
```
@mob slime
brain     hopper
maxHealth 30
speed     70
blood     #49b077
bands     shallow cavern
weight    1.0

art.cellsW 3
art.cellsH 2
art.pal   . #2c6f4a #49b077 #a8f0c4
art.fps   6
frame idle |
  .3.
  222
frame idle |
  ...
  212
frame move |
  ...
  222
```

### `content/structures/surface.struct`
```
@struct surface_cabin
name      "Woodcutter's Cabin"
place     surface
rarity    0.4
biomes    plains savanna jungle swamp
anchor    bottom_center
flatness  4
mirror    true
legend    c=W  block=wood
legend    c=G  block=glass
legend    c=L  block=goldOre|ironOre|coalOre  mark=loot
legend    c=X  mark=spawn
body |
  ___WWWWW___
  _WWWWWWWWW_
  _W.......W_
  _WGG.L.GGW_
  _...X....W_
  _WWWWWWWWW_
```

`.` = force air; `_` (or a space) = keep whatever terrain is there. Every other
glyph is declared by a `legend` row.

**`legend` is one repeated `record[]` row per glyph, not a single line of
`W=wood G=glass` pairs.** The one-line form would have to be a `list<string>`,
and §2's table only ref-validates a field declared `ref(kind)` — so a typo'd
block id would compile to a silently missing glyph instead of failing at the
line that named it. The row form also gives `mark` somewhere to live, and its
`block` value is a `ref?` preference chain like anywhere else.

`mark=loot|spawn` tags a glyph as a point of interest rather than a material;
those are surfaced as `SFLAG` bits and replayed from the structure's origin by
`eachMark()`, which is the seam a later loot pass hangs off.

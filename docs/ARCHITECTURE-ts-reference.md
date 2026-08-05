# Architecture

An infinite falling-sand sandbox on raw Canvas2D. No engine, no rendering
library, no p5 — the whole thing is TypeScript over `CanvasRenderingContext2D`
and a `SharedArrayBuffer`.

This document answers one question: **where does a new thing go?** It describes
the boundaries the code is organised by, not what each file does — for that, read
the file, they all carry a header.

---

## 1. The three layers

```
content/          authored data          →  compiled by tools/contentc
   ↓
src/generated/    compiler output           (never edited by hand)
   ↓
src/              the engine                (reads generated, never writes it)
```

The arrow only points one way. `src/` imports from `src/generated/`; nothing in
`src/` writes to `content/`, and nothing in `content/` knows a file path in
`src/`. That is what makes adding a block a content change rather than a code
change.

### `content/` — what things ARE

A custom line-oriented format (`content/FORMAT.md`), one directory per kind:

| Directory | Extension | What it declares |
|---|---|---|
| `blocks/` | `.block` | materials: colour, state, hardness, damage, surface, light |
| `items/` | `.item` | inventory items, weapons, tools, recipes |
| `mobs/` | `.mob` | the bestiary: brains, bodies, stats, spawn bands |
| `sprites/` | `.sprite` | pixel art: palettes, frames, animation sequences |
| `structures/` | `.struct` | template-driven landmarks |
| `worldgen/` | `.feature` | parametric worldgen features |

Numeric ids are pinned in `content/ids.lock.json` so a saved world does not
change meaning when a new block is added. Ids are assigned once and never reused;
a deleted record leaves a tombstone.

### `src/generated/` — compiler output

One module per kind, emitted by `tools/contentc`. Flat typed arrays for the hot
paths plus a def table for everything else. **Do not edit these files.** Run
`npm run content` after touching `content/`; `npm run content:check` is the CI
gate that fails if they are stale.

### `src/` — what things DO

The engine. Never restates a number that content already carries.

---

## 2. `src/` by directory

Ordered roughly by dependency depth — later ones import earlier ones, not the
other way round.

| Directory | Owns |
|---|---|
| `config/` | every cross-module tuned number. See §3. Depends on nothing but itself. |
| `game/` | `Game` (the loop, scene state, wiring), shared types |
| `sim/` | the cellular world: `CellGrid`, chunks, streaming window, automata, reactions, materials, biomes |
| `sim/gen/` | worldgen: heightmap, caves, layers, structures, features, loot |
| `sim/decor/` | what gets scattered on generated terrain: trees, ores, structures |
| `render/` | every draw pass: chunk canvas, light, sky, weather, particles, ambience, camera |
| `entities/` | the player, projectiles, player art |
| `entities/mobs/` | the living world: `MobSystem` pool, `Mob` brains, `MobDefs` facade |
| `items/` | inventory, registry, crafting, drops, dropped world items |
| `physics/` | AABB collision against the cell grid |
| `interact/` | the dig/place brush (`BuildTool`) |
| `input/` | keyboard and mouse |
| `ui/` | HUD, build HUD, menu and game-over screens — canvas draws, no DOM |
| `sprite/` | the sprite runtime: baking, palettes, animation clocks |
| `perf/` | benchmarks and the profiler. Not shipped logic. |

Two boundaries worth stating because they are load-bearing:

**The sim is the sole writer of cells.** The main thread only READS cells (for
rendering and collision) and POSTS edits. When the page is cross-origin isolated
the sim runs on a Web Worker over a `SharedArrayBuffer`; otherwise it ticks
inline. Both paths go through the same interface, so nothing downstream knows
which one it is on.

**Art and body are decoupled.** Content owns what a pixel looks like
(`.sprite`); code owns what a body does (`PLAYER_CELLS_W/H`, `bodyCellsW/H`). A
creature can be redrawn without moving a single tuned number — `npm run
verify:mobs` is the standing gate on exactly that.

---

## 3. Where a tuned number goes

Three tiers. Which one is decided by **what the number is about**, never by who
reads it.

| Tier | Where | It describes | Example |
|---|---|---|---|
| 1 | `content/` | one THING | a block's hardness, a mob's speed |
| 2 | `src/config/` | the WHOLE GAME | `CELL_SIZE`, `GRAVITY`, `SEA_LEVEL_Y` |
| 3 | a module constant | ONE ALGORITHM | an fBm octave count, a hysteresis band |

`src/config/` is split by domain, with `src/config/index.ts` as the barrel every
consumer imports:

| Module | Domain |
|---|---|
| `world` | cell/chunk/window geometry, sim rate. The units everything else uses. |
| `view` | canvas resolution and zoom. The only DOM-touching config module. |
| `worldgen` | where the ground, sea and depth bands sit |
| `physics` | the player's body and how it moves; owns `PHYS_SCALE` / `scaled` |
| `combat` | bare-handed melee and projectile facts weapons cannot override |
| `mechanics` | abilities (dash, wall jump) and surface reactions (ice, bounce) |
| `render` | render knobs shared across passes |
| `interact` | the dig/place brush |

**Tier 3 stays put.** Module constants are deliberately interleaved with the
prose that explains them in the context of the function below — `caves.ts` alone
carries 63 of them, each with its derivation written out. Hoisting them into
tier 2 would separate every number from its reason and rebuild the 483-line
constants monolith the config split exists to prevent.

Discoverability is solved by indexing rather than relocating: **`docs/TUNING.md`
lists every knob in all three tiers** with file, line, value and meaning. It is
generated — run `npm run tuning` after changing a constant.

### The dimensional scaling rule

Every distance-bearing tunable is written as the value that felt right for a
24px-tall character and multiplied by `scaled()` from `config/physics`. Lengths,
velocities and accelerations scale; **durations, ratios and fractions do not**.
Under that rule, resizing the character preserves jump height in body-heights,
run speed in bodies/second and time to apex exactly.

Anything that multiplies a velocity by its own private factor has forked the
rule. Import `scaled`.

---

## 4. `tools/`

| Path | What it is |
|---|---|
| `contentc/` | the content compiler: `lexer` → `parser` → `schema` → `emit` |
| `contentc/schema/` | one schema per kind. **Adding a field to the game is adding a row here.** |
| `contentc/plugin.ts` | Vite plugin, so `npm run dev` recompiles content on save |
| `tuning.mjs` | generates `docs/TUNING.md`; gates shadowing and tier-2 docs |
| `verify-mobs.mjs` | regression gate: a redraw must not move a tuned number |

A schema is data — a record of dotted field name → descriptor — and it is the
single source of truth for three things that must never disagree: validation of
the source text, the shape of the emitted TypeScript, and which typed arrays get
built.

---

## 5. Commands

| Command | Does |
|---|---|
| `npm run dev` | Vite dev server, content recompiled on save |
| `npm run build` | content → typecheck → bundle |
| `npm run check` | the full gate: content, types, tuning index, mob regression |
| `npm run content` | recompile `content/` into `src/generated/` |
| `npm run tuning` | regenerate `docs/TUNING.md` |
| `npm run verify:mobs` | assert no tuned creature number moved |

`npm run check` is what to run before committing.

---

## 6. Adding things — the short version

| To add… | Do this |
|---|---|
| a block, item, mob, sprite, structure | add a record under `content/`, run `npm run content` |
| a FIELD on one of those | add a row to the schema in `tools/contentc/schema/`, then use it |
| a game-wide knob | add it to the right `src/config/` module **with a doc comment**, run `npm run tuning` |
| a coefficient for one algorithm | a `const` in that module, next to the code, with its derivation |
| a render pass | a module in `src/render/`, owned and sequenced by `Game` (`Renderer` itself only paints the world buffer) |
| a mob brain | a case in `src/entities/mobs/Mob.ts`, selected by `brain` in `.mob` content |

The rule behind all six rows: **if it describes a thing, it is content; if it
describes the game, it is config; if it describes an algorithm, it lives with the
algorithm.**

# Terrain at grain 2 — how the world got twice the texture without moving a byte

This is the record of the P3 "render grain" change (commit `9c6b87c`): the
terrain's at-rest pattern now renders at **two texels per cell per axis** —
four sub-texels where one flat colour used to be — while the simulation, the
collision, the goldens and every piece of cell-keyed logic stayed exactly
where they were. It is written down because the interesting part is not the
feature; it is the discipline that let a whole-world visual change land with
**zero blessed fixtures**, and the one runtime failure that discipline did
not catch on its own.

The same architecture had already carried sprites, then mobs, then the
player (`grain` on the sprite records, render-only, cells stay
authoritative). Terrain was the last and hardest surface, because terrain is
the one place the pixels are pinned byte-for-byte by golden fixtures.

---

## 1. The constraint that shaped everything

Terrain pixels are guarded by three separate nets:

| Net | What it pins | What a careless grain change does to it |
|---|---|---|
| `cells_golden` | `paint_cells` output, byte-for-byte, 12 viewports + row hashes + a seeded-once `cover` mask | Every hash moves; the TS-authoritative `cover` would need a bless that its own design REFUSES |
| `shader_matches_cpu` | `cells.wgsl` == CPU blit, byte-identical, at rest and under shimmer | Any float drift or coordinate mismatch fails at pixel zero |
| `control_frames` | Luma / distinct-colour / dominant-share numbers per scene | Blends real change with noise unless before/after are honest |

The naive approach — change `paint_cells` to emit fine texels and re-bless —
would have destroyed the `cover` mask's authority (it is seeded-once,
TypeScript-era, and the blesser refuses to rewrite it, by design). So the
design goal was stated up front:

> **The G=1 path must remain byte-identical, provably, with the goldens left
> untouched. The fine path is a new, parallel output pinned by shader
> parity instead.**

## 2. The architecture: one implementation, two grains

`crates/yugen-render/src/cells.rs`:

- The body of `paint_cells` moved into `paint_grained<const G: i32>`.
- `paint_cells` became a thin wrapper calling `paint_grained::<1>` —
  monomorphisation folds the sub-texel loops away and compiles the G=1 case
  back to the original flat blit.
- New `paint_cells_fine` calls `paint_grained::<TEX_GRAIN>` (G=2), writing a
  `2w x 2h` buffer. This is the CPU oracle the shader is diffed against.
- `pub const TEX_GRAIN: i32 = 2` exported beside the tile periods.

**The acceptance gate for the refactor was `cells_golden` passing UNBLESSED
with `git diff --stat` on the fixture empty.** Green-unblessed is the proof
the wrapper is byte-identical; no amount of code review substitutes for it.

### Why the same tiles at doubled rate work

The fine sample is the same `TEX_A` (61-period) and `TEX_B` (67-period)
tiles read at doubled coordinate rate: `fine = cell * 2 + sub`. Because 2 is
invertible mod 61, mod 67 and mod 4087, each sub-column/sub-row is a
*permutation* of the tile rather than a subsampling of it:

- `p` stays in 0..62, so **no shade retune** was needed;
- no striping artefacts (a non-invertible stride would alias);
- the world-space repeat stays 4087 cells.

Per-cell structure was preserved in `paint_grained`: id runs, the `d_above`
depth array, side-openness and the emitter census all still compute **once
per cell**, with the G x G inner loops only writing texels. The row bases
became `row_a[sy] = pmod(G*(oy+ly)+sy, PA)*PA`, the column cursors advance
by `G` with one compare-subtract instead of a modulo — the no-modulo cursor
discipline of the original survived the genericity.

### What stayed keyed on the CELL, forever

Edge classes, census, air/cover, and the back-pass discard. Opacity is cell
geometry: a cell is wholly transparent or wholly opaque, and the grain may
never punch a sub-cell hole in the world. This is asserted, not hoped for —
see §4.

## 3. The shaders: quantise once, derive everything

`cells.wgsl` gained `const GRAIN: i32 = 2` and a `sub` parameter on
`cell_color`; the pattern samples at `(origin + cell) * GRAIN + sub`.

`cellmap.wgsl` and `backcell.wgsl` derive coordinates in one direction only:

```wgsl
let fine = vec2<i32>(clamp(floor(mesh.uv * dims * f32(GRAIN)), ...));
let cell = fine / GRAIN;
let sub  = fine - cell * GRAIN;
```

Never a second `floor(uv * dims)`. Two independent float truncations can
disagree by one ulp exactly on a cell boundary and draw a one-texel seam
there; deriving `cell` from `fine` by integer division makes the seam
impossible rather than unlikely.

## 4. The parity harness at the fine rate

`shader_matches_cpu.rs` was upgraded so the pinned thing is the *shipping*
arithmetic:

- The render target and readback went to `fw = w*g, fh = h*g` while the id
  texture stays `w x h` — that asymmetry IS the feature under test.
- The fragment shader derives `cell = fine / GRAIN` exactly like the game.
- `sweep` diffs against `paint_cells_fine`; the coverage threshold moved to
  >1.2M texels checked.
- The at-rest materials are **byte-identical** to the CPU oracle at the fine
  rate — all-integer arithmetic end to end is what makes "identical" a
  reasonable demand rather than a tolerance.
- New test `every_cells_texels_agree_in_alpha`: all four sub-texels of every
  cell agree in alpha. This is the GPU-side restatement of `cells_golden`'s
  `cover` mask — the harness has no cover hash, but it can assert the
  property the hash freezes.

### Kill tests (run during development, then reverted)

A guard that has never failed is a guess. Each planned failure mode was
injected and had to fail:

| Injected fault | Result |
|---|---|
| Fine coordinate off by one (`+ vec2(1,0)` in the pattern) | 214 963 of 359 908 cells differ — parity fails loudly |
| `sub` leaked into the edge class (`cell_edge_class(ids, dims, cell + sub)`) | 22 787 cells differ, concentrated on rims |
| `GRAIN = 3` in the wgsl only | Const-parity test fails **by name**: "cells.wgsl says GRAIN = 3, cells.rs says 2" |

The third required adding `("GRAIN", TEX_GRAIN)` to the const-parity expect
list — the test passed before that, which meant GRAIN simply wasn't being
asserted. A passing test proved nothing until its failure was demonstrated.

## 5. The bug the suites could not see, and what it taught the rig

First after-capture run: every scene's luma up ~12, distinct colours DOWN,
and `deep-chamber.png` showed **a night sky** — the camera was 900 px
underground and the frame had stars in it.

Cause: `cellmap.wgsl` and `backcell.wgsl` used `GRAIN` without importing it.
**naga_oil imports are item-scoped** — `#import yugen::cells::{cell_color,
CellShadeParams}` brings in exactly those two items, and a const used but
not imported is not a compile error in `cargo build` (shaders compile at
runtime, in the pipeline cache). The pipelines failed at runtime, the
terrain pass silently dropped out, and the sky rendered through where rock
should have been.

The parity harness couldn't catch it: it compiles `cells.wgsl` standalone
against raw wgpu, and `cells.wgsl` was correct. The two Bevy-side files are
exactly the "line of arithmetic the parity harness is not checking" their
own headers warn about.

Worse: **`control_frames` measured the corpse and passed.** Its scene checks
sample the SIM (is the chamber carved, is there lava), not the frame, so a
dead render pipeline produced plausible PNGs and a full table of numbers —
the exact "plausible file" failure its file header rants about.

Three fixes, each pinned:

1. The import: `#import yugen::cells::{cell_color, CellShadeParams, GRAIN}`
   in both files.
2. The structural test in `cellmap.rs` that requires backcell to be a CALLER
   of the shared shading now names the full import string, GRAIN included,
   with a comment explaining that a missing naga_oil import fails at
   runtime, not build time.
3. `control_frames` now counts engine ERROR lines (the harness's
   `error_count()` / `error_sources()` — which already existed and were
   simply never consulted) across the whole set and **voids every number**
   if any appeared: *"the engine logged N ERROR lines while the set was
   captured — every number above is a measurement of a broken renderer."*
   Kill-tested by dropping the import again: the rig fails with that
   message instead of producing a table.

## 6. The numbers

Control set before → after (luma mean / distinct colours / dominant share):

| scene | luma | distinct | dominant |
|---|---|---|---|
| surface | 94.71 → 94.41 | 5 456 → 5 976 | 1.7% → 1.7% |
| deep-chamber | 22.76 → 22.92 | 5 966 → 7 290 | 3.8% → 3.8% |
| ore-chamber | 43.87 → 43.82 | 3 764 → 4 393 | 11.0% → 10.2% |
| lava-sea | 168.95 → 168.97 | 29 583 → 36 471 | 1.7% → 1.7% |
| lit-chamber | 33.74 → 33.88 | 14 860 → 17 552 | 3.6% → 3.8% |
| seed777 | 96.41 → 96.59 | 6 707 → 8 125 | 1.7% → 1.7% |
| seed42 | 94.36 → 94.44 | 6 429 → 7 530 | 1.7% → 1.7% |

Exactly the prediction: luma within ±0.3 (the permutation argument holding —
same tile distribution, so the same mean), detail up 10–25%, dominant share
flat or down. If luma had moved, the change would not have been "the same
art, finer" — which is how the first, broken run was caught: +12 luma
everywhere was the sky bleeding through a dead terrain pass.

## 7. What to remember

- **Green-unblessed is the strongest acceptance gate this repo has.** A
  refactor that keeps a golden suite green without touching the fixture is
  proven byte-identical by the very net that would catch the lie.
- **Permutation, not subsampling.** Doubling a sample rate into a
  prime-period tile is safe exactly when the stride is invertible mod the
  period. Check the arithmetic before trusting the eyes.
- **Quantise once.** Any coordinate that exists at two resolutions must be
  derived (integer math) from a single float truncation, or boundaries
  seam.
- **naga_oil imports are item-scoped and fail at runtime.** A shader const
  used across files must be in the import list; the build succeeding means
  nothing. Structural tests should pin the import string itself.
- **A rig that can produce a plausible file from a broken subject must
  consult the engine's own error stream.** The counters existed; nothing
  read them. The cheapest guard is often already built.
- **Kill-test every guard, in both directions, before believing it.** Two
  of the five guards in this change passed vacuously on first writing (the
  GRAIN const check, and — in the follow-up viscosity work — an emitter
  test defeated by its own granted precondition). A test is evidence only
  after it has been watched failing.

## 8. The escape hatch

Flip `TEX_GRAIN` to 1 in `cells.rs` and `GRAIN` to 1 in `cells.wgsl` (the
const-parity test holds them in agreement). Everything else — wrapper,
harness, goldens — is already correct at G=1, because G=1 never stopped
being the pinned case.

**This hatch has been used, and not as a rollback.** `docs/WORLDSCALE.md` records
the change that made a cell half a world-feature unit, at which point a grain-2
texel at the old scale and a grain-1 texel at the new one are the same size, and
drawing two texels per cell became the same trick applied twice. Grain 2 was, in
hindsight, the render-only preview of halving the cell.

`paint_grained` and `paint_cells_fine` survive that retirement deliberately: the
parity harness still runs at a grain above 1, because
`every_cells_texels_agree_in_alpha` is vacuous when a cell has one sub-texel, and
§7 of this file is quite clear about what a guard that cannot fail is worth.

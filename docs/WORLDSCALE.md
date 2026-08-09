# The world got twice as big, and the render grain went home

This is the record of the change that made one cell half a world-feature unit
(`WorldScale::LIVE = 2.0`) and, in the same commit, put `TEX_GRAIN` back to 1.
It is the sequel to `GRAIN2.md`, and the short version is that **grain 2 was
always a preview of this**: it drew two texels per cell because the cell was too
coarse for the world, and the fix for a cell being too coarse is a smaller cell,
not a finer brush.

It is written down because the interesting part is not the feature. It is that a
change to every cell of the world landed with `player_golden` — 4 048 fixed
steps, no bless path, un-regenerable — never moving, and that the four rigs that
broke all broke by *detecting their own scene had stopped being their scene*.

---

## 1. The arithmetic the whole change rests on

Every frequency in `sim::worldgen` is *cycles per cell*. Every amplitude,
threshold and band depth is *cells*. The samplers all take `wcx`/`wcy` cell
indices cast to `f64` (`fields.rs`, `caves.rs`, `heightmap.rs`).

So scaling the world by `S` is not sixty edits. It is one transform, applied at
the boundary of the field layer:

> **divide the coordinate going in, multiply the cell length coming out.**

Nothing between those two crossings moves. `CONT_FREQ` still means an 1800-cell
continental lobe. `TERRACE_STEP` still means a six-cell shelf. `CAVERN_DEPTH`
still means the depth it says. Not one of them was touched.

The space on the far side of the crossing is called **legacy cells** throughout
worldgen. `WorldScale` carries the four crossings and nothing else:

| Method | Direction | For |
|---|---|---|
| `coord(wc)` | in | a world cell coordinate entering a noise sample |
| `depth(cells)` | in | a world depth meeting a band threshold |
| `len(cells)` | out | a legacy length becoming world cells |
| `row(legacy_row)` | out | an authored absolute row becoming a world row |

### The two things that must NOT be scaled

`WARP_STRENGTH` (`fields.rs`) and `CHEESE_WARP_AMP` (`caves.rs`) are
displacements *in noise space* — they are already on the divided side of the
crossing. Scaling them too applies the world scale twice and bends coastlines by
±10% of a doubled period instead of the ±5% of the authored one they were tuned
for. This is the single easiest way to get this change subtly wrong, because it
looks like a length and it is measured in cells.

## 2. Why it is a parameter and not a constant

`tests/player_golden.rs` replays 4 048 fixed steps against an arena stamped into
a world it pins by material hash. Its provenance is a TypeScript tool in a
repository that no longer exists. It has **no bless path**, and its header says
that if it fails, the port is wrong rather than the test.

A `const WORLD_SCALE` would have forced that fixture to be retired — the only
long-replay net over `Player::step`, and the only thing in the tree that can see
that coyote time is read one phase too late or that a jump cut is applied to an
already-clamped velocity. Those are ORDER bugs, and order is only visible in a
long replay.

So the scale is an argument. `player_golden` calls `generate_chunk_scaled` with
`WorldScale::LEGACY` — the identity — permanently, and the generator is free to
move underneath it. That is the entire justification for the parameter, and it
cost one argument on the worldgen call path.

## 3. The acceptance gate, which is the same one GRAIN2 used

The threading commit set `LIVE = 1.0` and shipped nothing else. Then:

```
cargo test -p yugen-core --test worldgen_golden   # 357 chunks
cargo test -p yugen-core --test player_golden     # 4 048 steps
cargo test -p yugen-core --test noise_golden
git diff --stat -- '*.golden.json'                # must be EMPTY
```

All green, fixtures untouched, whole workspace green. **Green-unblessed is the
strongest acceptance gate this repo has**: a refactor that keeps a golden suite
green without touching the fixture is proven byte-identical by the very net that
would catch the lie. Only then did `LIVE` go to 2.0.

Doing it in that order is what made the next section possible, because it meant
every failure afterwards was *the world changing*, not the refactor being wrong.

## 4. What actually broke, and why every break was the rigs working

### A real defect the identity commit could not expose

`SpawnProbe` generated its terrain at `ChunkGen::new` — LIVE — while the search
around it reasoned at the caller's scale. At the identity commit both were 1.0,
so it was invisible. At LIVE it measured walkable ground in a world the body was
not standing in. It now takes the scale it is asked for.

This is worth naming because it is the shape of every bug this kind of change
has: **two things that were the same number for so long that nobody noticed they
were two things.**

### Four rigs aimed at the world by a fixed coordinate

| Rig | Symptom |
|---|---|
| `control_frames` lava-sea | "no lava within the room at this depth" |
| `scenarios` lava sea | 0 lava of 441 cells |
| `scenarios` lava death | body stood in the lava at full health |
| `shader_matches_cpu` `depths` | shimmer branch: 5 960 animated texels, was >100 000 |

All four are the same defect: **a depth authored in the legacy geometry stops
naming the band it was chosen for the moment the world scales.** The underworld
begins `UNDERWORLD_DEPTH` cells below the *local surface*, and at LIVE that is
twice as far down in world cells. Every one now puts its authored depth through
the scale.

Not one of them was found by a number looking wrong. `control_frames` refused to
print a table for a lava frame with no lava in it, and the shimmer parity refused
to certify a branch exercised by 5 960 texels. That is exactly the
guard-against-a-plausible-file discipline `GRAIN2.md` §5 paid for, collecting.

### One empirical claim that genuinely changed

`walkable_spawn`'s both-sided claim went from 24/24 seeds to 22/24. The
guarantee the function makes — `SPAWN_WALK_CELLS` clear in at least one
direction — still holds for every seed, and the test now asserts that separately
from the preference, with the new number written down. Longer landforms mean a
spawn column is likelier to sit on a big hill's slope than on a small hill's flat
top. The two that fall back (seeds 6 and 22) are the honest cost.

## 5. Why the grain retired in the same commit

`GRAIN2.md` §8 kept the hatch open for exactly this, and the reasoning is one
line: in legacy-feature units, a grain-2 texel at the old scale and a grain-1
texel at the new one are **the same size**.

```
old cell   ####################
now cell   ##########
texel      ##########      <- grain 1, and 1/2 an old cell either way
```

Drawing two texels per cell on top of a halved cell would be the same trick
applied twice. `cells_golden` is the live pixel net again at grain 1, which is
what makes the flip safe rather than merely cheap.

`paint_grained<const G>` and `paint_cells_fine` **stay**. Two reasons, and the
second is the important one:

1. The hatch stays proven in both directions.
2. `every_cells_texels_agree_in_alpha` is **vacuous** at grain 1 — one sub-texel
   trivially agrees with itself. A guard that cannot fail is precisely what
   `GRAIN2.md` §7 warns about, so the harness keeps a grain at which it can.

## 6. What this did NOT change, which is most of the engine

`CELL_SIZE`, `CHUNK_CELLS`, `WINDOW_CHUNKS_X/Y`, the 352x256 streaming window,
`ZOOM_MIN`, `View`, `PLAYER_CELLS_W/H` and every px of physics are exactly where
they were. **The cell count per chunk, per window and per frame is unchanged, so
this costs nothing to simulate** — no 4x automata sweep, no bigger id texture, no
window resize, and `docs/PERF.md`'s budgets stand as measured.

What changed is the world under them: a chunk covers half the world it used to,
the viewport frames half as much of it, and the terrain carries twice the cell
detail per landform.

## 6b. The second pass: everything doubles relative to the body

The first pass left terrain at 2x with decor and structures at 1x, so trees read
small against the hills. The second pass took the world to **4x** and introduced
`BODY_SCALE = 2` for the player and the creatures. The gap between those two
numbers is the design: **at 4x world against 2x bodies, terrain, trees and
buildings are all twice the size they used to be relative to the player, and in
the proportion to each other they were authored in.**

Three kinds of thing scale three different ways, and keeping them apart is what
makes the change reviewable:

| Kind | How it scales | Example |
|---|---|---|
| **Lengths** | through `WorldScale` | cave radii, band depths, surface amplitude |
| **Rasters** | nearest-upscaled, `k x k` per authored cell | structure ASCII bodies |
| **Drawings** | by enlarging their pixels | trees, ore blobs, clutter |
| **Bodies** | by growing their cell footprint | player, mobs |

A raster cannot be scaled by a float — there is no such thing as 1.5 cells of
wall — so structures index the source at `c / k`, with **mirroring done in
destination space**: dividing first mirrors the block instead of the body and
shifts the template by `k-1` on every odd width. `stamp`, `each_mark` and the
O(1) `mark_in_site` all take the same factor, because a loot pass that recovers
mark positions from a different factor recovers the WRONG cells while every
determinism check still passes.

A tree is not a set of lengths either. It is fifty cell-space constants
describing a shape, so `DecorContext` grew an expansion anchor and `trees.rs`
goes on drawing a 1x tree from its trunk base. Threading a scale through every
one of those constants would have been fifty chances to get one wrong.

### Sprite art needed no re-authoring

Every body sprite carries a `grain`. Halving a record's grain while doubling its
`cellsW`/`cellsH` spreads the SAME characters over twice the cells per axis — so
the player went from 4x5 cells at grain 2 to 8x10 at grain 1 and **not one
character of art moved.**

The cost is that no grain-2 record ships any more, which quietly disarmed the
guard that pinned the grain mechanism (it sampled the frostmite). It now
synthesises both grains itself. A guard a content edit can switch off is not
guarding the mechanism.

### The one that had to be reverted

`STEP_UP_MAX` scales with the body and `STEP_UP_REARM` deliberately does not.
The reach belongs to the character. The re-arm is the horizontal run between
risers on the shallowest slope that must stay walkable, and **a riser is one CELL
tall whatever is climbing it.** Scaling it to `0.8 * STEP_UP_CELLS` cells was
tried: a 45-degree hill supplies a riser every one cell of travel, a re-arm at
1.6 cells never admits the second one, and the body climbed 2 cells in four
seconds instead of 12.

### player_golden was re-recorded, and is a weaker thing now

The body changed shape — `PLAYER_CELLS_W`/`H` went 2x3 to 4x6, which moves every
px in the replay and changes how the resolver snaps a blocked body to a cell face
— so no bless-free path existed. The fixture was re-recorded from this
implementation, which its own header forbade.

Be exact about the cost. Before: two implementations, written in different
languages from the same design, agreed on 4 048 steps — evidence the body is
CORRECT. After: this player compared against a recording of this player, which
can only say it has not CHANGED. `KNOWN_TIES` is now empty and has to be: all 46
entries were f64-versus-f32 ties against the TypeScript, and there is no second
implementation left to tie with. 20 240 of 20 240 discrete comparisons agree
exactly, which sounds like an improvement and is the opposite of one.

## 7. The known, deliberate debt

**Discharged in the second pass.** Decor and structures now scale — as rasters
and drawings rather than as lengths, per §6b. What follows is the record of why
they could not simply be multiplied, which is still the reason the mechanism
looks the way it does.

**Decor and structures do not scale by multiplication.**

Trees are procedural but authored at cell granularity (`trees.rs`: a 22-cell
trunk, a 6-cell crown, clutter 2 cells wide). Heights multiply cleanly; a
1-cell trunk becoming 2 cells is a redesign, not a multiply. Structure bodies in
`content/structures/*.toml` are ASCII cell art — a 13x9 shrine nearest-upscaled
to 26x18 is 2-cell-thick walls, which is a different building.

So they currently read half-size against the doubled terrain. That is visible in
`control-frames/surface.png`: the trees are small for the hillside. It was left
visible rather than half-fixed, because a partial scaling would be harder to see
and harder to undo than an obvious one.

Their *depth gates* needed no change at all — `minDepth`/`maxDepth` in the
structure TOMLs are legacy depths and the band tests run in legacy depth space.
That is a direct dividend of putting the crossing at the boundary instead of
scaling the constants.

## 8. What to remember

- **Divide the coordinate, multiply the length, and touch nothing in between.**
  A whole-world geometry change that edits sixty constants is a change nobody can
  review; one that edits four crossings is one anybody can.
- **A displacement measured in noise space is already scaled.** It looks like a
  length and it is spelled in cells, and scaling it applies the factor twice.
- **Prove the refactor at the identity before changing anything.** Every failure
  after that point is the world moving, which is a completely different debugging
  problem from the refactor being wrong — and you only get to tell them apart if
  you separated them in time.
- **A rig aimed at the world by a fixed coordinate has an invisible dependency on
  the world's geometry.** Four of them did here, and all four were named for the
  band they meant rather than the number they used.
- **A guard that says "this scene is not the scene you asked for" is worth more
  than a guard that says "this number moved."** Every one of the four was caught
  by the former.
- **The escape hatch you write for a change is the hatch the sequel uses.**
  `GRAIN2.md` §8 was written as a rollback and got used as a retirement.

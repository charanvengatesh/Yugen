# Scenarios

> **Five of these rows are now gated.**
> `crates/godgame-render/tests/scenarios.rs` runs the carved chamber, the lava
> sea, the wall plane, the clock and "lava kills a body" through
> `scene::ScenePlugin` — the same code path `--at` and `--edit` use — and
> asserts the claim each row makes. The numbers below are still exact
> measurements; the assertions are deliberately looser, because a chamber's
> exact cell count is a function of the brush, the automata's settling and the
> terrain it was cut into, and pinning 417 would fail on the next legitimate
> change and be deleted within a month.
>
> Everything NOT in that file is still an observation in a document, which is
> where a true statement goes to quietly stop being true. If you add a row, add
> the assertion.


A scenario is a named way of putting the game into a situation worth looking at.
There are two kinds and the difference is whether input is involved:

- **A command line.** Most situations are pure placement — a depth, a clock, one
  brush stroke — and those live in the tables below rather than in a file,
  because a file that holds no verbs holds nothing a flag did not already say.
- **A `.txt` script**, driven by `--script`. Only when the scene has to be
  *played* into existence: the pointer, the tool cadence, the drop tables and the
  pickup are things no placement flag reaches. There are two, and both earn it.

```
dig-a-shaft.txt            stand still and dig straight down
undercut-in-daylight.txt   sink a shaft at noon, then cut sideways off its floor
```

Every command below was run and its `--dump-state` JSON read before it was
written down. The numbers in the "what the dump says" column are from those runs.
They are written as `cargo run --release --`, which is the repo's convention;
`./target/release/godgame` is the same binary and is what these were actually
verified with.

`--dump-state` on its own **does not end the run** — it writes and carries on, by
design. Pair it with `--screenshot` (or a `--script`, which ends the run itself)
or the process sits there until you kill it.

---

## Two things that will catch you

### 1. `--at` and `--edit` compose, and `--edit` is relative to what `--at` set

`--edit`'s `CX CY` are cells from the **view centre**, not absolute cells, and
`--at` is what moves the view centre. So the same stroke means different places
depending on whether `--at` is present:

```
--edit dig 0 0 12                 carves at the spawn        (cell -127, 32)
--at 0,900 --edit dig 0 0 12      carves 900 px down         (cell 0, 180)
```

Both of those are the binary's own log line from a real run, on seed 2334. A
thousand cells apart, from the same four words.

The binary prints which one it did — `--edit Dig at cell (0, 180) r=12` — and
that line is the cheapest way to check you meant it. A cell is 5 world px, so
`CY` of `+12` is 60 px further down, and `+y` is DOWN.

**One stroke per run.** `--edit` is a single slot, not a list: passing it twice
silently keeps only the last one. `--edit dig 0 0 10 --edit lantern 0 0 3` logs
`Place at cell (0, 180) r=3` and never digs — the dump comes back with 243 cells
of stone still in the window. Multi-material scenes therefore need multiple runs
sharing a `--world` directory; see *A lit chamber* below.

### 2. The world has to stream to the new position, and the dump can beat it

`--at` moves the camera on the first frame. The world does not move with it: the
streaming window arrives over the next few frames, and nothing is stamped until
the grid actually holds the cell the stroke is aimed at. That wait is handled
inside the binary — you do not have to arrange it — but **`--warmup` can still
fire the dump before the carve lands**:

| `--warmup` | empty cells in the dumped window |
|---|---|
| 0 | 54 |
| 1 | 54 |
| 2 and above | 417 |

Same command otherwise (`--free-camera --at 0,900 --edit dig 0 0 12`). 54 is the
untouched rock; 417 is the chamber. Two frames is all it takes here, but the
failure is silent and looks like a stroke that did nothing, so the tables below
use comfortable warmups rather than minimal ones.

`--warmup` also **advances the clock**, which matters on the twilight ramp: 30
frames moved `clock.t` by +0.0011, 600 frames by +0.0169. A long settle at
`--time 0.26` is not a capture of `--time 0.26`.

---

## Underground

| Scenario | Command | What the dump says |
|---|---|---|
| **Deep chamber** | `cargo run --release -- --play --free-camera --at 0,900 --edit dig 0 0 12 --warmup 60 --dump-state out.json --screenshot out.png` | 417 of 441 cells empty. The control — same command with `--edit` removed — has 58. The carve is doing the work, not worldgen |
| **Deep chamber, with a body in it** | `cargo run --release -- --play --at 0,900 --edit dig 0 0 12 --warmup 120 --dump-state out.json --screenshot out.png` | Body ends at (0, 945): it is placed at 900 and falls 45 px down its own hole, so the dumped window recentres 10 cells lower. 297 empty. 5 mobs, all underground species — skeleton, sandworm, gulper, glimmerback, sentinel |
| **A lit chamber** (two runs) | `cargo run --release -- --play --world /tmp/lit --free-camera --at 0,900 --edit dig 0 0 10 --warmup 60 --dump-state a.json --screenshot a.png`<br>then<br>`cargo run --release -- --play --world /tmp/lit --free-camera --at 0,900 --edit lantern 0 0 3 --warmup 200 --dump-state b.json --screenshot b.png` | Run 1: 348 empty. Run 2 **into the same `--world`**: 319 empty and 29 `lantern`. 348 − 29 = 319 exactly, so the second run found the first run's chamber and filled 29 of its cells — which is the proof that `--world` persists a `--edit` stroke |
| **Ore chamber** | `cargo run --release -- --play --free-camera --at -500,1500 --edit dig 0 -7 9 --warmup 90 --dump-state out.json --screenshot out.png` | 186 empty, and 44 `gemOre` + 8 `ironOre` still standing in the chamber's floor and lower wall. The control at the same place with no `--edit` is 0 empty and 45 `gemOre`: the offset stroke (`CY = -7`) opens the room *above* the vein and leaves all but one cell of it intact |
| **Lava sea** | `cargo run --release -- --play --free-camera --at 0,3000 --warmup 40 --dump-state out.json --screenshot out.png` | 387 `lava` and 54 `basalt`. Zero empty cells — the window is entirely inside the melt, which makes it a light source and not a picture of one |
| **Lava-floored chamber** | `cargo run --release -- --play --free-camera --at 0,2620 --edit dig 0 -10 12 --warmup 200 --dump-state out.json --screenshot out.png` | 135 empty above 282 `lava`, in `ash` and `basalt` wall. The single scenario here that is a lit cave in one command: the stroke opens headroom at the top of the melt and the lava is its own floor, so there is nothing to drain |

The lava layer under the default seed 2334 was found by sweeping `--free-camera
--at X,Y` over x ∈ {−500, 0, 500} and y ∈ [1000, 5000] and counting block code
11 in the dumps. Not one of the nine samples at y ≤ 2000 contained any. It first
appears at y = 2500 (136 cells, threaded through 221 of basalt) and is lava-
bearing everywhere from y = 2700 to y = 3100, though not uniformly — 403, 440,
405, 279, 183, 260, 387, 441, 441 cells at 2700, 2750, 2800, 2850, 2900, 2950,
3000, 3050, 3100, so the 2850–2950 band is half basalt and the sea is solid again
below it. Below that it thins fast: 9 cells at y = 3500, none at 4000 or deeper,
where it is stone and ore again.

### The background wall plane

`cells.wall` non-zero where `cells.front` is 0 is what "you are inside something"
looks like in the dump, and the split is total rather than gradual:

| Where | air cells | of those, with a wall behind |
|---|---|---|
| Deep chamber, y = 900 | 417 | **417** |
| Lit chamber, y = 900 | 319 | **319** |
| Ore chamber, y = 1500 | 186 | **186** |
| Desert surface, seed 777 | 248 | **7** |
| Snow surface, seed 2334 | 282 | **0** |

Underground every air cell has a wall behind it, because every air cell was
carved out of something. On the surface almost none do, because the air above a
hillside is sky. The 7 in the desert row are worth looking at rather than
rounding to zero: they sit together at rows 8–10, columns 5–8 of the dumped
window, on the shoulder of a dune, and they are there because the background
dune's crest stands a cell or two higher than the foreground one. That is the
only case in this catalogue where the wall plane is visible in daylight without
anybody digging for it, which is what makes the desert the useful surface control
and the snow spawn — 0 of 282 — the useless one.

---

## Time of day

All four are the same spawn on seed 2334 with only `--time` changed, so
everything except the light is held constant — the front plane is 282 empty / 105
snow / 52 stone / 2 pine leaves in every one of them.

| Scenario | Command | What the dump says |
|---|---|---|
| **Night** | `cargo run --release -- --play --time 0.0 --warmup 600 --dump-state out.json --screenshot out.png` | `clock.day = 0.000`. 12 mobs: **7 bats and 5 grublings** |
| **Noon** (the control) | `cargo run --release -- --play --time 0.5 --warmup 600 --dump-state out.json --screenshot out.png` | `clock.day = 1.000`. 11 mobs: **1 bat**, 9 grublings, 1 critter |
| **Dawn** | `cargo run --release -- --play --time 0.26 --warmup 30 --dump-state out.json --screenshot out.png` | `clock.t = 0.2611`, `clock.day = 0.498` — the middle of the rising ramp |
| **Dusk** | `cargo run --release -- --play --time 0.735 --warmup 30 --dump-state out.json --screenshot out.png` | `clock.t = 0.7361`, `clock.day = 0.557` — the middle of the falling ramp |

Night and noon are worth running as a pair rather than separately. The mob counts
above are the nocturnal weighting made visible without a screenshot: the bat
share goes from 7-in-12 to 1-in-11 across nothing but the `--time` flag.

**Where the twilight values came from.** `docs/HANDOFF.md` §8.2 lists twilight as
never having been looked at, and calls the horizon glow the steepest ramp in
`sky.rs`. It is. Sampling `clock.day` at `--warmup 30` gives:

```
t     0.18  0.20  0.22  0.23  0.24  0.25  0.26  0.28  0.30
day   0.00  0.00  0.00  0.023 0.129 0.300 0.498 0.865 1.00

t     0.70  0.72  0.74  0.75  0.76  0.78  0.80
day   1.00  0.833 0.454 0.260 0.101 0.00  0.00
```

So the whole transition happens inside roughly `t = 0.23..0.30` and
`t = 0.70..0.77` — about 0.07 of a day each way, out of a day. The textbook
values sit low on their ramps rather than in the middle of them: `--time 0.25`
measures `day = 0.300` and `--time 0.75` measures `day = 0.260`, both nearer
night than half. **0.26 and 0.735 are the values that put the horizon glow at
roughly half strength** (0.498 and 0.557), and they are what the table uses.

---

## Surface and biome

Each is the default spawn for that seed, nothing else set.

| Scenario | Command | What the dump says |
|---|---|---|
| **Desert** | `cargo run --release -- --play --seed 777 --warmup 90 --dump-state out.json --screenshot out.png` | Spawn (−2430, 110). 176 `sand`, 11 `sandstone`, 6 `cactus`, no stone at all in frame |
| **Pine ridge** | `cargo run --release -- --play --seed 42 --warmup 90 --dump-state out.json --screenshot out.png` | Spawn (890, **40**) — by some way the highest spawn of the ten seeds sampled. 29 `leavesPine`, 11 `wood`, 51 `snow`, 344 empty. The best candidate found for HANDOFF §8.2's "the ridges need a hilltop", though whether the ridges are actually unoccluded here is a question for the picture, not the dump |
| **Coast** | `cargo run --release -- --play --seed 4242 --warmup 90 --dump-state out.json --screenshot out.png` | 72 `water`, 61 `sand`, 100 `sandstone`, 13 `ice` and 2 `wetSand` — a waterline, in one frame, with the wet/dry boundary material present |
| **Basalt flat** | `cargo run --release -- --play --seed 1 --warmup 90 --dump-state out.json --screenshot out.png` | 196 `basalt` and 245 empty, and **nothing else** — no soil, no plants, no mobs. The starkest surface of the ten sampled |

The other six seeds sampled (7, 101, 1234, 9001, 31337, and the default 2334) are
variations on dirt-and-sandstone temperate, pine-and-snow, or sandstone coast,
and none of them shows anything the four above do not.

---

## Scripts

| Scenario | Command | What the dump says |
|---|---|---|
| **Dig a shaft** | `cargo run --release -- --play --script scenarios/dig-a-shaft.txt --warmup 300 --dump-state out.json --screenshot out.png` | See the script's own header; it states its measured result against a `wait`-instead-of-`dig` control |
| **Dig a shaft, at noon** | `cargo run --release -- --play --time 0.5 --script scenarios/dig-a-shaft.txt --warmup 300 --dump-state out.json --screenshot out.png` | Body at (−615, 210). 154 air cells, **61** of them with a non-zero `cells.wall`, at `clock.day = 1.000` |
| **Undercut in daylight** | `cargo run --release -- --play --time 0.5 --script scenarios/undercut-in-daylight.txt --warmup 300 --dump-state out.json --screenshot out.png` | Same body position and therefore the same 21x21 window as the row above, so the two compare cell for cell: 179 air, **86** with a wall behind. 25 more cells of exposed wall plane. Reproduced exactly over two runs, `drops` at 8 both times |

The last two exist for `WALL_DECAY`. `docs/HANDOFF.md` §8.2 asks for "a daylight
surface rig with the scene forced to `Playing` and a shaft cut into a hillside"
and calls it a third `lit_scene`-shaped file. It is not a file any more; it is
`--time 0.5` in front of a script that already existed, plus one that cuts
sideways to give the rule more wall to act on.

---

## What did not work

**A lava pool poured into a carved room drains away, and this is not fixable with
`--edit`.** Carve a 10-cell room at y = 900 and pour `--edit lava 0 4 4` into it
through a shared `--world`, and after 200 frames the dump has **1 cell of lava
left** and 2 of `steam`. Laying a stone plug under the room first — a third run,
`--edit stone 0 14 9` — improves it to **4 cells of lava** and 5 of `glass`,
where the pool crossed some sand on the way down. Neither is a lit scene; both
are a photograph of an unlit hole.

This is the exact hazard `crates/godgame-render/tests/lit_scene.rs` documents in
its `carve_and_light` header: 900 px down, the floor is as likely to be a void as
rock, and the pool drains through it. That test solves it by *stating* the floor
rather than hoping for it, which `--edit`'s one-disc-per-run brush cannot do —
a disc big enough to seal the floor is big enough to refill the room. The two
working answers are in the tables above and neither is a lava pool: **lanterns**,
which do not flow (29 of them, still there after 200 frames), or **going to where
the lava already is** at y = 2620, where it is its own floor.

**A `--script` cannot light a cave either.** The `place` verb places the selected
hotbar item, and the starting inventory is `pick_traveler`, `sword_traveler` and
three `bandage`. There is no light source in it, so there is no scripted route to
a lit room.

**The undercut cannot be walked longer.** Adding `left 30f` between strokes moves
the body nowhere — it is standing in a one-cell shaft with rock on both sides —
and the extra strokes re-dig gone cells while snow slumps back in. Measured at
**66** exposed wall cells against the short script's 86. Recorded here, and in the
script's header, so nobody tries it a third time.

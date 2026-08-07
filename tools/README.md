# tools/

Small standalone scripts. Nothing here is part of the build; nothing here is run
by `cargo xtask check`.

## `pngdiff.py` / `pngwhere.py`

Compare two PNGs. `pngdiff` reports how much moved, `pngwhere` reports where.

```
python3 tools/pngdiff.py  a.png b.png
python3 tools/pngwhere.py a.png b.png
```

Pure standard library — no PIL, no numpy, no imagemagick, none of which is on the
machine this was written on. PNG is simple enough that a correct decoder for the
8-bit truecolour cases is forty lines, and a real pixel diff is worth far more
than "the hashes differ".

### What they are for

`tests/frame_stability.rs` asserts two bounds on how much an idle frame may
change. Those bounds are measured, not guessed, and these are what measured them
— including the region map that showed the movement was the sun's dithered halo
rather than something wrong.

### The rule they exist to enforce

**Never claim two frames are identical from looking at them.** That claim was
made once in this repo's history and was wrong: neither capture rig is bit-stable
by default, `lit_scene` varies by ~16% run to run, and the full binary by 54.8%,
because the world clock advances on wall time.

**Always diff against a same-tree control.** Two runs of the same build first,
then the build you care about. A difference only means something once you know
what zero looks like.

`tests/common::FRAME_DT` pins the frame delta, which takes `frame_capture` to
bit-identical across runs — so inside the test harness the control is now exactly
zero. Outside it, running the real binary, it is not.

## `statediff.py`

Diff two `--dump-state` JSONs. The workflow it serves is: run a scenario twice
with one thing changed, and read off what that change did.

```
./target/release/godgame --play --script scenarios/dig-a-shaft.txt \
    --warmup 300 --dump-state /tmp/dug.json
python3 tools/statediff.py /tmp/nodig.json /tmp/dug.json
```

Reports differing scalars, inventory by item id, a mob census, and — the point —
the cell layers. `--cells-only` and `--no-map` cut the output down. Exit status
is 0 if the dumps are equivalent and 1 if they differ, so it works as a shell
test; a file it cannot read is 2.

### The rule it exists to enforce

**The `cells` block is a window centred on the body, and the body moves.**
Comparing the two `front` arrays row by row therefore compares two different
places in the world. On the dig-a-shaft pair that reports 242 changed cells for
a dig that changed 53 — not a small error, but nonsense, because the whole
shifted band reads as changed against whatever sat nine rows above it.

So every cell is reconstructed to an absolute coordinate,
`(centre[0] - radius + i, centre[1] - radius + j)`, and only the overlap is
compared. On that pair the body falls 9 cells, the overlap is 252 of 441, and
the answer is 53: 43 snow and 10 stone to `empty`, against an inventory gain of
43 `snow_ball` and 18 `stone_chunk` — the other 8 chunks came out of the shaft
below the dumped window. Two windows that do not overlap at all say so rather
than reporting zero.

The ASCII map is there because the count cannot tell a dug shaft from a
landslide and the shape can.

Material names are parsed out of `crates/godgame-data/src/blocks.rs` at run
time. A table copied in here would go stale the first time a block was added,
and would fail no gate while doing it. Unknown codes print as numbers.

### The float tolerance

`1e-3`. Across four same-tree runs everything in a dump was bit-identical
except `clock.t`, which drifted by up to 5.6e-5 because the world clock advances
on wall time. The tolerance sits ~18x above that measured floor and far below
anything the sim can mean: cells are 5px, so a real positional change is at
least 1.0.

Mobs are reported as a census, never matched individually. `id` is the species,
not an identity, and nothing in a dump survives across runs to match them by.

## `palette_check.py`

Checks `content/blocks/*.toml` against the invariants in `content/PALETTE.md`.

```
python3 tools/palette_check.py             # exit 1 on a violation
python3 tools/palette_check.py --baseline  # report only, always exit 0
```

**Not yet a gate**, because the palette it describes has not landed: on the
current content it reports 25 violations, which is the work the repaint exists
to do. It joins `cargo xtask check` when the last band commit lands.

Two things about it are worth knowing before reading its output.

It measures **chroma**, `(max-min)/255`, and never HSL saturation. At extreme
lightness HSL-S is meaningless: `snow` reports S 58% at a chroma of 5.5, so an
HSL rule flags the least colourful material in the game and waves through
`copperOre`. This was got wrong once already.

The band-dependent invariants report **SKIPPED** rather than passing while
`BANDS` is empty. Band membership is authored, not inferred — inferring it lets
two bands' ranges overlap until a block is silently in whichever one was tested
first — and a check that has no data should say so rather than print a tick.

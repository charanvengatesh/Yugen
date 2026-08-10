# The pin

Two algorithms in this tree are written twice, and this is what stops the copies
drifting apart.

| authority | copy | what it does |
|---|---|---|
| `yugen-render/src/sprite/baked.rs::bake_frame` | `yugen-editor/src/raster.rs::raster` | frame digits to RGBA |
| `yugen-render/src/sound/synth.rs::render_params` | `yugen-editor/src/synth.rs::render` | eight floats to samples |

## Why they are written twice

`yugen-editor` cannot link `yugen-render`. That crate is Bevy — a windowing
stack, a render graph and a GPU — and pulling all of it in so a text grid can
become bytes would be absurd on its own. But the real reason is that it would
still be the **wrong** bytes: the renderer bakes and synthesises from *compiled*
content in `yugen-data`, and an editor has to show the file as it is being typed,
before `contentc` has run and while it may not even be valid. The sound case is
sharper still — an unsaved sound has no code, so there is nothing to pass
`render(code)` at all.

So the duplication is not laziness, and it is not fixable by refactoring. What it
needs is not one implementation but one **answer**.

## How the pin works

Neither crate can call the other, so they meet at a file. Each side builds the
same fixed case, runs its own implementation, and compares against the golden
here. If the two implementations ever disagree, one of them stops matching this
file and its test fails — which is the whole point, because the failure mode it
replaces is silent: the editor would draw a sprite or play a sound subtly unlike
the one the game does, and nothing would say so.

- `sprite_raster.hex` — RGBA8 of one 4x4 frame, row-major.
  Written by `yugen-render/src/sprite/baked.rs`, read by both.
- `sound_synth.hex` — the raw `f32` bits of one 0.02 s sound, in order.
  Written by `yugen-render/src/sound/synth.rs`, read by both.

The cases themselves are declared identically in both crates rather than parsed
from a fixture, because a fixture format would need a parser in each crate and
those could drift in their own right. The cases are six lines each; if they ever
diverge, both sides stop matching the golden and say so loudly.

## Regenerating

```sh
YUGEN_BLESS=1 cargo test -p yugen-render pin
```

The same switch the worldgen and player baselines use, and the same rule applies:
**blessing is not how you fix a failing pin.** A change here means the game's
rasteriser or synthesiser moved. If that was deliberate, bless it and port the
change to `yugen-editor` in the same commit. If it was not, the diff has found a
regression in the thing that draws and plays the whole game.

`yugen-render` is the authority and is the only crate that writes these files.
`yugen-editor` only ever reads them, so it can never bless its own copy into
agreement with itself.

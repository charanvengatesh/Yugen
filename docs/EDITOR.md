# The content editor

```text
cargo run -p yugen-editor                    # finds content/ by walking up
cargo run -p yugen-editor -- path/to/content # or point it somewhere
```

One window over everything in `content/` that is authored as numbers you have to
imagine. The browser lists both kinds and marks which is which:

- `▦` **art** — the twelve player poses, sixty-odd mob sequences, every item
  icon. A pixel canvas and a palette. Left button paints the selected index,
  right button erases, `⌘Z` undoes a stroke.
- `♪` **sound** — every record in `content/sounds/`. Knobs and a waveform scope.
  Space bar plays it.

`⌘S` saves, either way.

## Why it exists

A sprite in this game is digits inside a `'''` body and a sound is eight floats.
Both were authored that way — all of `content/sprites/player.toml` and all of
`content/sounds/` — and until this landed, redrawing a frame meant counting
characters in a text editor and retuning a sound meant editing a number, running
the game, and getting to the moment that plays it. That works, and it is why the
fleet is twenty-one creatures rather than fifty. The cost per change was the
thing limiting how much art and audio the game has.

## The one rule everything is built around

**The file is the document. The editor never regenerates it.**

A save does not serialise a model back to TOML. It takes the file exactly as it
was read and splices lines into it: the edited frames replace the lines those
frames occupied, and every other byte — the banner at the top, the paragraph
above each sequence, the palette table written out as a comment, the trailing
spaces that are part of the picture — is copied through untouched.

This is `content/FORMAT.md` §9, and it is not a stylistic preference. §5 makes
`#` and trailing whitespace *data* inside `'''`; §6 makes the comments the most
valuable thing in the file. `player.toml` is roughly two-thirds prose by line
count and nearly all of it sits *between* sequences, inside the record. A tool
that parsed the record and re-emitted it would delete every one of those lines
while leaving a file that still parses, still compiles, and still draws the same
character. That failure is silent, which is why the whole design bends around
preventing it.

Two consequences worth knowing:

- **An untouched sequence produces no diff.** A save rewrites every sequence in
  the record; the ones that did not change splice back byte-identical. So
  `git diff` after a session shows the frames that actually moved and nothing
  else. `crates/yugen-editor/tests/real_content.rs` proves the byte-identity
  against every art record in the tree, not against a fixture.
- **The editor refuses to save a file that changed on disk** since it was
  opened. It holds line spans into the text it read; applying them to a file
  somebody edited in the meantime would splice over whatever now occupies those
  lines. Reload, then redraw.

## What it does not do

It does not write `content/ids.lock.json` and it does not run the compiler. The
lock is what guarantees new content takes codes *above* the baseline boundary
instead of renumbering what is below, and renumbering is how a save file turns
the player's gold into gravel. So the status bar says what to run instead:

```text
cargo run -p contentc
```

Adding a **new** record is also not here yet — `splice::append_record` exists and
nothing in the UI calls it. That is deliberate for now: §0 says a new record goes
in the file `content/LAYOUT.toml` names for it, and when no row claims it the
right move is to add a file and a row, which is a human decision. A tool that
guessed would be guessing about filing.

## The canvas and the cell grid

The grid overlay draws a **heavier line every `grain` texels**, because that is
one world cell — the unit the simulation actually thinks in, what you dig and
what you stand on. At `grain = 1` every line is heavy and the two grids are the
same. At `grain = 2` the light lines are art texels and the heavy ones are cells,
which is the whole content of that field: a sprite is allowed to be finer than
the ground it stands on, because a sprite is never the thing you interact with.
See `docs/GRAIN2.md`.

## The rasteriser is written twice

`crates/yugen-editor/src/raster.rs` restates the rules in
`yugen-render/src/sprite/baked.rs::bake_frame`. The alternative was for the
editor to link `yugen-render`, which is Bevy — a windowing stack, a render graph
and a GPU, pulled in so a text grid can become bytes — and it would still be the
wrong bytes, because the renderer bakes from *compiled* content and an editor has
to draw the file as it is being typed, before `contentc` has run and while it may
not even be valid.

The honest statement of the risk: **nothing yet proves the two agree.** The rules
are small and have not changed since the TypeScript original, but that is a
reason to expect agreement, not a mechanism that enforces it. The mechanism would
be a shared golden fixture both crates rasterise and compare against, and it is
not written. Until it is, `bake_frame` is normative and `raster.rs` is the copy
that follows it.

## Sounds

The same window. A record marked `♪` in the browser opens knobs instead of a
canvas: the waveform picker, the two pitches, the length, the envelope, the noise
mix and the gain — every field the `sound` schema has, with its range. `Play` or
the space bar synthesises the current parameters and sends them to the speaker.

**This is the first time any of these sounds can actually be heard.** The
renderer's tests prove range, decoding and wiring; they do not prove a footstep
sounds like a footstep. That question needed ears and a knob, and this is the
knob.

Above the knobs is a scope drawing the samples the speaker is being handed —
not a picture of the parameters. The envelope clamp, the millisecond de-click
ramps on both ends and the clip are all visible there and none of them are
visible in the numbers.

### Only what moved gets written

The sounds have the same "no spurious diff" property as the sprites, arrived at
differently. `content/sounds/` records write only the fields they mean: `[step]`
never mentions `attack`, because 0 is the default and an instant onset is what a
footstep wants. So:

- A save writes only the keys whose values actually changed. Untouched keys are
  never rewritten, which matters because a number does not always survive an
  `f32` round trip.
- A field the record never wrote gets its line **added** when you first turn it,
  and stays absent until then. Materialising all eight fields on the first save
  would turn eight terse records into eight identical walls.

The panel lists exactly which `key = value` lines a save would write, so the rule
is visible rather than something you have to trust.

### The synthesiser is written twice, too

Same story as the rasteriser, and a stronger reason. `yugen-render`'s synth
renders from `yugen_data::sounds` — `static` arrays indexed by a compiled code —
so a sound that has not been through `contentc` has *nothing to pass it*. The
whole point of a knob is hearing the value before it is committed, and the game's
renderer cannot take an uncommitted value at all. Same caveat: nothing yet proves
the two agree, and `yugen-render`'s copy is normative.

Audio output is `rodio` with default features off — no decoders, because nothing
here decodes. The device is opened on the first `Play` rather than at startup, so
a machine with no sound still runs the editor for the half of its job that is
pixels.

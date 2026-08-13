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

It does not create FILES. Records, yes — `New…` appends one to a file you pick
from the list it already found. But the list is the point: §0 says a new record
goes in the file `content/LAYOUT.toml` names for it, and a file no row claims
fails the layout gate. Adding a row is a decision about how content is
organised, and a dialog that guessed would be guessing about filing.

The id is checked against every record in the **kind directory**, not just the
target file: `contentc` concatenates all of `content/mobs/`, so two files there
cannot both hold a `grubling`, and a per-file check would create a record that
compiles nowhere.

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

That risk used to be unmanaged, and this paragraph used to say so. It is managed
now: `tests/pin/` holds goldens both crates render and compare against, so the
copy cannot drift from the original without a test going red. `bake_frame` and
`yugen-render`'s `render_params` are still normative — the editor's versions are
the copies — but "normative" is now enforced rather than merely asserted. See
`tests/pin/README.md`.

## Generating instead of starting from nothing

The right-hand panel of an art record has four sections, and everything in them
is undoable (`⌘Z`) because a generator replaces a whole frame — far more than a
stroke does.

**Generate** rolls a whole silhouette from a seed, shown in hex and editable. The
seed is the entire input, so those four bytes are the drawing: type one back in
and you get it again, and a record created from a roll carries its seed in the
comment.

The pipeline is fitted against a reference sheet rather than guessed. Its
characters fill about 0.62 of their 8x8 and are bottom-weighted — they stand on
the ground. An earlier version filled 0.25 and was vertically symmetric, which
made every roll a small thing floating in a large box. What fixed it was not the
obvious knob: the cellular smoothing counted below-the-grid as empty, so it
eroded the feet. Counting the cell directly below as ground support does all of
it, and a per-row density bias that looked necessary turned out to earn nothing
once that was right, so it is not in the code.

**Operators** are `Frame -> Frame` and every one of them preserves the grid.
That invariant is why they can be buttons: a frame one row short is a record
whose art and whose `cellsH` disagree, which `contentc` compiles happily and the
renderer answers with a panic at `PreStartup`. `rotate ⟳` is the exception that
proves it — it refuses a non-square grid rather than resizing, which is why the
8x8 rule makes it total.

**Colours** generates ramps. Saturation moves only at the shadow end, toward a
middle value: measured off the reference, a saturated red loses 0.57 of its
saturation going dark while a near-grey *gains* 0.09, and one rule produces both.
Highlights hold their saturation, because a chalky highlight is the loudest tell
that a ramp was generated. `snap to reference` moves every colour to its nearest
of 64 reference colours — nearest in HSL with lightness weighted heaviest,
because lightness carries the form of an 8x8 drawing and hue does not.

**Animate** derives frames from the one on screen. `walk` is the interesting one:
at eight texels there is no limb to swing, so what reads as walking is *contact*
— which foot is down. It alternates runs of the bottom row, and a creature with
one unbroken bottom row gets a hop instead, which is the honest reading of
something with no legs.

## Resizing, and the two guards

`Resize…` is the only edit that moves `cellsW`/`cellsH`. It says what the change
would cost before you make it, naming every frame that would lose ink, and it
defaults to padding rather than scaling: padding leaves a creature
pixel-identical and scaling draws it at twice the size of the box the player can
actually hit.

Two of its checks disable the button rather than warn, because they guard
assertions in `MobDef::build` that fire as **panics at load**: art may never be
smaller than the body box, and the width difference must be even. `contentc`
validates neither, so this is the last place they can be a message next to a
button.

`yugen-fit` does the same thing from the command line, through the same code, for
when the answer is "all of them":

```text
cargo run -p yugen-editor --bin yugen-fit -- --to 8x8 --dry-run content/mobs/*.toml
```

It refuses by default to run any resize that loses ink.

## Sounds

The same window. A record marked `♪` in the browser opens knobs instead of a
canvas: the waveform picker, the two pitches, the length, the envelope, the noise
mix and the gain — every field the `sound` schema has, with its range. `Play` or
the space bar synthesises the current parameters and sends them to the speaker.

**This is the first time any of these sounds can actually be heard.** The
renderer's tests prove range, decoding and wiring; they do not prove a footstep
sounds like a footstep. That question needed ears and a knob, and this is the
knob.

Under the eight is a **Shaping** fold with four more — vibrato depth and rate, a
repeat rate, and a one-pole low-pass. Every one is off at 0, and off is an
explicit branch in the synthesiser rather than an identity multiply. That
distinction is the whole reason the fourteen sounds that predate these fields
still render bit-identically: a multiply by one and a filter coefficient of one
are the same synthesiser in algebra and different floats.

## Rolling a sound

A sound record's panel has categories — pickup, jump, hurt, blip, explosion,
laser, powerup. Each is a **region of the parameter space**, not a preset: press
one twice and get two different pickups that are both recognisably pickups.

`mutate` jitters what is loaded by a fraction of each field's range, and small is
the useful end — a roll finds the neighbourhood and a mutation finds the house. A
field that is OFF stays off, because "near this sound" should not mean "near this
sound plus a wobble".

Every roll is kept in a list you can audition without loading, because the roll
button's failure mode is not producing a bad sound — it is producing a good one
and then producing another. **None of it is persisted.** This editor writes
content files and nothing else; a favourite worth keeping becomes a record, which
is what `New…` is for.

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

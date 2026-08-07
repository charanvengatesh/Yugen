# Handoff

A port of a TypeScript/Canvas2D falling-sand sandbox game to Rust + Bevy +
wgpu, milestones M0 through M10. This document is for the person taking it over.

It is written to be useful rather than flattering. Where something is unverified,
it says so; where a decision is likely wrong, it says that too. **Read §7 and §8
before you read anything else** — the findings and the open items are the part
you cannot reconstruct from the code.

## Provenance, and where to be sceptical

The port was written by an AI agent (Claude), largely by fanning work out to
sub-agents, one file or module each. Practical consequences for a reviewer:

- **Doc comments are dense and argue their decisions.** This is deliberate and is
  the codebase's main asset. It is also the main risk: a confident comment can
  outlive the thing it describes. Several did (§8.3). Trust the code over the
  comment when they disagree, and fix the comment.
- **Every non-obvious number has a stated reason.** If you find one that does
  not, that is a defect worth treating as such.
- **The port's characteristic defect is a TypeScript shape written in Rust
  syntax.** A stored `Box<dyn FnMut + 'static>` where a borrowed trait would do; a
  pointer-plus-count pair where a slice would do; `Vec::remove(0)` where a
  `VecDeque` would do. Each is individually harmless-looking and each one cost
  something real — see §7.1 and §9.1. When something reads as competent but odd,
  ask what the JavaScript looked like.
- **The test suite is large (943 passing) and was written alongside the code**,
  so it encodes the same assumptions. Tests agreeing with the implementation is
  weaker evidence here than usual. The frozen parity suites (§6) are the
  exception and are the strongest evidence in the repo.

## 1. What it is

| | |
|---|---|
| Language / engine | Rust 2024, Bevy, wgpu |
| Toolchain | pinned by `rust-toolchain.toml` |
| Tests | **943 passing**, 0 failing |
| `unsafe` | **zero** (the one grep hit is the literal string in `contentc`'s Rust-keyword list) |
| `#[allow]` | 10, all pre-existing in `sim/worldgen`, `sim/decor`, `contentc`, generated data. `godgame-render` and `xtask` have none |
| Source | ~84k lines across six crates |

The original was at `../GodGame` (TypeScript). **It is no longer a reference for
anything.** Five test suites still carry fixtures dumped from it; they have been
renamed to `*_golden` and are now baselines of this project against itself. See
§6.

## 2. Layout, and the one boundary that matters

```
crates/contentc        content compiler: TOML -> generated Rust tables   (6.8k)
crates/godgame-data    the generated tables. Never hand-edit.           (10.0k)
crates/godgame-core    sim, entities, physics, items. NO BEVY.          (33.7k)
crates/godgame-render  every Bevy-facing thing                          (31.0k)
crates/godgame         the binary: window, CLI flags                    (0.4k)
xtask                  the gate runner and the tuning index             (2.0k)
```

The dependency arrow is one-way and `godgame-core` **may not know what a
`KeyCode` is**. That is the load-bearing constraint. What falls out of it:

- the sim is testable headlessly and benchable without a GPU;
- anything Bevy-shaped must be expressed as a seam (§4), which forces the
  boundary to stay explicit rather than eroding;
- `Player.draw`, `Projectiles.draw`, `WorldItems.draw` and the DOM `Input` class
  were amputated on the way in, and what they READ is published as accessors.

`docs/ARCHITECTURE.md` is the full version (445 lines) and is accurate as of M9.

## 3. Milestones

| | Delivered |
|---|---|
| **M0** | workspace scaffold |
| **M1** | `contentc` — content compiler |
| **M2** | config, noise, full worldgen + purity suite |
| **M3** | `CellGrid`, chunk streaming, falling-sand automata |
| **M4** | Bevy app, wgpu cell rasteriser, verified against a CPU oracle |
| **M5** | player, collision, input, build tool |
| **M6** | mobs, projectiles, items |
| **M7** | lighting and atmosphere (daynight, sky, weather, ambience, light, particles, effects) |
| **M8** | sprite atlas, authored bitmap font, HUD, scenes |
| **M9** | architecture doc, `cargo xtask check`, tuning index, perf pass |
| **M10** | quality pass — real bloom, pixel-grid sky, pixel-grid light, dead joins connected |

Two milestones changed the game rather than porting it, and both are documented
as deliberate divergences at the site:

- **Particle collision** (`particles.rs`). The original's debris fell through
  floors. Ours collides. This is the one place the port knowingly plays better
  than its source; no parity suite covers particles.
- **Creative is a truce** (`Player::untouchable`). The original's creative mode
  is only an infinite palette. Ours also makes the body untouchable — creatures
  neither aggro on it nor damage it, and hazards do not.

Also worth knowing: **content moved from a bespoke DSL to TOML** during M2
(commit `f9bead3`), compiled to Rust tables by `contentc`. `content/ids.lock.json`
pins id→code so a reordered file cannot silently renumber the world.

## 4. The seam pattern — the tree's main idiom

The same problem recurs: A needs B, and neither should own the other. It is
always solved the same way — **a small trait, declared on the side that needs the
answer, implemented on the side that has it, and borrowed for the length of the
call.** Never a direct import, and never a stored callback.

| Seam | Where |
|---|---|
| projectile hit test | `trait ShotWorld` -> `glue::step_the_body` -> `MobSystem::hit_at` |
| ammo source | `trait AmmoSource` -> `impl for Inventory` |
| the player's shot pool | `trait Projectiles` -> `impl for ProjectileSystem`, lent via `Loadout` |
| item icons | `glue::IconAtlas for SpriteAtlases` |
| scene -> HUD screen | `glue::follow_scene` |
| creature target | `MobTarget` trait |
| chunk persistence | `ChunkPersistence` trait |

**Why borrowed and not captured**: the first three used to be boxed `FnMut`s
stored on the caller, because Bevy cannot hand a second borrow down a call stack
that goes system -> `Player::step` -> `ProjectileSystem::update`. So the closure
captured — and `'static` forces owned, owned forces `Arc`, shared-mutable forces
`Mutex`. That call stack no longer nests: `glue::step_the_body` asks for all four
borrows in its parameters and steps the body and the pool on consecutive lines.
Three `Arc<Mutex<_>>` types went with it. See §9.1 and `ARCHITECTURE.md` §3.

**The rule to take away**: when A needs B for the duration of a call, pass B to
A's method. In a single-schedule game, an `Arc<Mutex<_>>` is usually a borrow
that has not been threaded far enough.

`crates/godgame-render/src/glue.rs` is the composition root and its header states
the principle. When you add a cross-module dependency, put it there.

## 5. Running and debugging

### Run

```
cargo run --release                      # play it
cargo run --release -- --free-camera     # no player; WASD flies the view
cargo run --release -- --drive 1.5       # runs right by itself, jumps every 1.5s
cargo run --release -- --screenshot out.png --warmup 400
cargo run --release -- --edit dig 0 20 6 --screenshot out.png
```

**Build release before launching the binary directly.** `./target/release/godgame`
does not rebuild. This cost real time during development: the binary sat frozen
at an older milestone while every test passed, because the tests build their own.
`cargo run` avoids it entirely.

In game: `G` toggles creative, `1`–`0` and the wheel drive the hotbar (survival)
or the palette (creative), `C` crafts, `F` consumes, Enter/Space leaves the menu,
**`F3` shows the debug panel.**

### The F3 panel — read this before adding a `println!`

`crates/godgame-render/src/debug.rs`. Frame time, seed, focus, cell, chunk,
depth, the resolved biome and underground layer with their weights, the mob
count, banked XP, and the material / wall plane / solved light under the pointer
— or under the CAMERA FOCUS when there is no pointer, which is what makes it
useful in a capture nobody is watching.

```
cargo run --release -- --debug-overlay --screenshot out.png --warmup 400
```

`--debug-overlay` starts it up, because a `--screenshot` run has no keyboard.

Two properties worth knowing. It samples **nothing the frame did not already
compute** — the biome weights are the mood `ambience` published this frame, the
light is the grid `light` already solved — so it cannot agree with itself and
disagree with the game. And its layout is a pure function of a plain
`DebugReadout` struct, so it is unit-tested with no GPU, no window and no world;
if you add a row, add it to those tests rather than to a screenshot.

It exists because of `§7.1`: three bugs whose shape was *a resource is declared,
read, and written by nothing*, one of which lit every biome identically for two
milestones. The panel is the cheapest instrument that would have shown all three.

### Saving: `--world DIR`

Without it nothing is durable, which is what every milestone up to this one did.
With it, `godgame_core::sim::save` puts each edited chunk in its own ~8 KB file
under `DIR/chunks/`, and the hole you dug is there when you come back.

`ChunkPersistence`'s doc has said since the port that "there is no durable save
behind this trait" and that the boundary was narrow so adding one later would be
"a new implementation of these three methods and no change anywhere else". That
held — `DiskChunkPersistence` is that implementation. What it did NOT imply, and
what cost two debugging rounds, is the rest of it:

- **The store is a write-back cache.** A chunk reaches persistence when the
  window shifts far enough to evict it, so a player who digs and quits where they
  stand writes nothing. `WindowManager::flush` is the missing half — `ChunkStore`
  already had a `flush` with zero callers and a comment saying it was "the hook a
  durable backend needs"; the store cannot see the LIVE grid, and the freshest
  edits are in exactly that. `world::autosave` calls it every 30 s and once more
  on `AppExit`.
- **A world is built in TWO places** — `world::spawn_world` at `Startup` and
  `glue::start_a_run` on every entry to `Scene::Playing`. Passing the directory
  to one of them means the other silently discards that world and builds an
  unsaved one over the top. It did. The logs said the save directory was open,
  22 chunks reported as persisted, and not one file existed.

Both are pinned by
`sim::save::an_edit_flushed_from_the_live_window_is_there_for_the_next_world`,
which is headless and needs no renderer. To check the whole path instead, dig
with a script and reload in a fresh process:

```
godgame --play --world /tmp/w --script scenarios/dig-a-shaft.txt --warmup 300 --dump-state a.json
godgame --play --world /tmp/w --warmup 300 --dump-state b.json     # a second process
```

Compare `cells` by ABSOLUTE coordinate — the window moves between runs, so
comparing them row by row compares two different places.

**Picking a world.** `Menu -> WorldSelect -> Playing`. The select screen lists
what is under the saves root (`--saves DIR`, default per-user data dir), makes
new worlds with a fresh random seed, and deletes them behind a named
confirmation. `--play` skips both screens: a capture or a scenario has already
been told which world it wants, via `--world` and `--seed`.

`crate::worldselect` follows `crate::debug`'s shape — plain data, a pure layout
function, systems only where the world is touched — so seven unit tests cover
every string and every bound with no GPU and no filesystem. The capture test
covers the one thing they cannot: that the screen is reached, composed and
painted, which a pure layout function nothing routes to would not be.

It has **no text field and no seed field**. That is a widget, and `crate::ui`
has rectangles and text runs. New worlds are "World N" until somebody builds
one; `create_world` already takes any name and `slug_of` already survives
whatever a field would produce.

**A world is two things.** `DIR/chunks/` is the terrain, `DIR/run.save` is
everything else: seed, world clock, the body, and the inventory slot by slot.
Both are written atomically, both refuse a file that is not theirs, and the run
file also refuses one whose seed does not match the world it was found beside —
dropping a save into the wrong directory would otherwise teleport the body into
terrain grown from a different seed, which reads as corruption rather than as a
mistake.

Two ordering facts, both of which cost a debugging round if forgotten:

- **The restore runs LAST in `glue::start_a_run`**, after the body reset and
  after the starting kit. Any earlier and `body.reset()` undoes it and the kit
  buries it, and it looks exactly like the save was never written.
- **`Inventory::put_at`, not `add`.** `add` implements the game's placement
  policy — merge, then first free slot — which is precisely wrong when the slots
  are already decided. Loading through `add` silently rearranges the player's
  pack every time.

### The spawn is chosen by `walkable_spawn`, not `spawn_point`

`spawn_point` asks the heightmap where the dry land is. The heightmap knows
nothing about the trees the decorators put on it afterwards, and every leaf
material in `content/blocks/flora.toml` is authored `collides = true` — so a
canopy is a wall. Measured over 24 seeds at `SPAWN_COL`, counting cells of
walkable ground either side of the body:

| | under 12 cells one side or both | zero on one side |
|---|---|---|
| `spawn_point` | **23 / 24** | 6, one of them zero on BOTH |
| `walkable_spawn` | 0 / 24 | 0 |

`spawn_point` is untouched and still what `worldgen_golden` pins — "where is the
land" and "where can a body stand" are different questions, the second needs
generated chunks, and folding it into the first would make a cheap pure function
expensive and move a baselined value for no terrain reason.

**It follows the ground, not a row**, and the first version did not. `spawn_point`
returns a point six cells ABOVE the surface, so a check run at that row measures
thin air: every column passed and the body still stopped dead after exactly the
number of cells the check had verified. Twice. If you change this, verify it by
walking the body in the game — `--script` plus `--dump-state` — and not by
reading the checker's own answer back.

What it buys is 12 cells (60 px) either way, which is room to stand up and pick a
direction. It does **not** promise an open world; the surface is genuinely broken
up by slopes steeper than the body's one-cell step, and getting further means
jumping or digging. What it rules out is starting the run entombed.

### Driving the game without a person: `--script` and `--dump-state`

```
cargo run --release -- --play --script scenarios/dig-a-shaft.txt \
    --warmup 300 --dump-state out.json --screenshot out.png
```

`--script` is a verb-per-line file (`right 2s`, `jump`, `aim 0 20`, `dig 3s`,
`hold punch 1s`, `wait 90f`) parsed by `godgame_core::script`. `--dump-state`
writes what the simulation believes — player, inventory, creatures, clock, and a
21x21 block of both cell planes around the body — as JSON.

Four things about them are load-bearing:

- **A scripted run is pinned to a fixed 60 Hz delta.** Without it the sim's
  accumulator clamps at `MAX_STEPS_PER_FRAME` and `right 2s` covers a different
  distance on a busy machine than an idle one. With it, three runs of the same
  script produced **byte-identical dumps**. `tests/common::FRAME_DT` pins the
  same thing for the capture rigs.
- **`--warmup` is the settle before the first verb, not padding.** Chunks stream
  over hundreds of frames and the body falls into whatever has arrived. Scripts
  starting at frame 0 finished in the wrong places.
- **`aim` is an offset from the BODY**, not a world point. A script's job is to
  move the body, so where it will be when a line runs is the output of every line
  above it and is not knowable when the file is written.
- **The script presses the real mouse buttons and the real pointer**, through
  `input::CursorOverride`. Nothing reaches into `apply_brush` behind the game's
  back, so a scenario that digs proves the reach check, the tool cadence and the
  hardness gate all work — the same argument `--edit` makes.

`scenarios/` holds committed scenario files. To check one did what it claims,
diff its dump against a control run with the verb removed; that is how the dig
above was confirmed to have removed exactly the two cells it aimed at.

### Jumping straight to a situation: `--at`, `--time`, `--edit`

```
godgame --play --at 0,900 --edit dig 0 0 12 --time 0.5 --warmup 300 --screenshot cave.png
```

That is a lit chamber 900 px underground at noon, in three flags. It is the
scene `lit_scene.rs` hand-builds in Rust, and whose header records that it was
built and thrown away three times before it stuck — because the default spawn is
a snowy surface in daylight where most of the lighting stack is invisible.

**The three flags are not independent, and `main::arrange_the_scene` is one
system rather than three because ordering them was not enough.** Three versions
looked right and were not:

1. Place the body, then carve. The body stands in solid rock and the collision
   resolver ejects it before the carve lands — 260 px of drift.
2. Carve, then place. `--edit` is relative to the VIEW CENTRE, still at the
   spawn, so `--at 0,900 --edit dig 0 0 12` carved at cell (-127, 32).
3. Aim the camera, carve, place, all on one frame. The camera moves instantly
   and the WORLD does not: the streaming window still covered the spawn, the
   brush was clipped away entirely, and the "cave" was solid stone.

So it waits for `stream_window` to bring the world to the new focus, and only
then carves and places, on one frame. If you add a flag that touches the world at
startup, put it in that system rather than beside it.

### Debug

Three rigs, in increasing order of how much they tell you:

```
cargo xtask check                                          # all five gates
cargo test -p godgame-render --test frame_capture -- --nocapture
cargo test -p godgame-render --test lit_scene -- --nocapture
cargo bench -p godgame-core -p godgame-render
```

**`frame_capture`** boots the whole plugin group headlessly with `WinitPlugin`
disabled, captures the low-res buffer, writes `target/tmp/frame-capture/lowres.png`,
and asserts the frame is not degenerate. No window, so **it works with the lid
shut** — which matters, because a closed laptop produces a black window
screenshot and that wasted an hour of debugging a bug that did not exist.

**`lit_scene`** carves a lava-lit cave 900px underground and writes
`target/tmp/lit-scene/cave.png`. It is **not a gate** — there is no numeric
property of a cave worth failing a build over. It exists because the default
spawn is a snowy surface in daylight where the entire lighting stack is invisible
or clipped, and every lighting change otherwise has to hand-build a scene. This
one was built and thrown away three times before it was kept.

**`graphify`** — there is a knowledge graph at `graphify-out/`. `graphify query
"<question>"` returns a scoped subgraph and is much cheaper than grepping. Run
`graphify update .` after structural changes.

### When something looks wrong on screen

Bisect by plugin. `GodGameRenderPlugin` in `lib.rs` is a `PluginGroup`; comment
one out and re-capture. That is how the smooth-glow complaint was traced to the
light grid rather than the bloom (§7.3).

## 6. The gates, and what each test kind catches

`cargo xtask check` runs five gates, cheapest first, fail-fast, with the child's
own output streamed. A failure names the reproduce command. Later gates are
reported `not run`, never as passing.

```
fmt      cargo fmt --all --check
tuning   docs/TUNING.md is current, config documented, no name shadows
content  generated tables are not stale
clippy   --workspace --all-targets -- -D warnings   (zero findings)
test     cargo test --workspace
```

`xtask` deliberately depends on **no workspace crate**: a gate runner that cannot
start when the sim crate is broken is useless exactly when you need it.

Test kinds, and what only each one can see:

| Suite | Catches |
|---|---|
| `registry_golden` (data) | 79 tables, 2629 slots, id->code mappings |
| `noise_golden` | every noise entry point |
| `worldgen_golden` | 357 chunks, cell for cell |
| `player_golden` | **ordering** bugs — 22 scripted cases, 4048 fixed steps. Coyote time read one phase late, `wall_dir` cleared after the collide. Invisible except in a long replay |
| `cells_golden` | the CPU rasteriser, pixel for pixel |
| `worldgen_purity` | a chunk is a pure function of `(chunk_x, chunk_y, seed)` |
| `shader_matches_cpu` | the WGSL agrees with the CPU oracle |
| `mob_regression` | 8 creatures x 25 fields, exact equality |
| `frame_capture` | the composite produces a varied, correctly-oriented image |

**The five golden baselines were the port's TypeScript parity suites. The port
is over; they are now this project's own regression net** and answer "has any of
this moved", not "does this match the original". Two rules survive the rename and
one is new:

- A baseline going red when you did not mean to move anything means the CODE is
  wrong. That has not changed.
- Changing a baseline is a deliberate act with a diff to read
  (`GODGAME_BLESS=1 cargo test -p godgame-data --test registry_golden`), in its
  own commit. It is never how a red test is made green.
- **New:** the baseline is a PREFIX. New blocks, items, mobs, sprites and
  structures are appended above the boundary and are not compared; they cannot
  renumber or overwrite anything below it. Before this, the suites asserted an
  exact record count and a single new block failed three of them, which is why
  the game could not grow. `crates/godgame-data/tests/registry_golden.rs` has the
  argument in full.

## 7. Findings

### 7.1 The bug class that matters here: pipes built, joins missed

Three separate defects with the same shape. Each degraded to *plausible* output,
which is why nothing caught them.

- **`MobSystem::events()` returns a fixed backing buffer**, and only the first
  `event_count` entries are valid — its doc says so. The host drained the whole
  buffer, so ~14 phantom `MobHurt` events arrived per frame at (0,0), each adding
  0.08 trauma against a decay of 4/s. **The screen shake was pinned at maximum
  from the first frame.** 896 tests and a purpose-built image gate were green:
  a shaking camera still renders a varied, correctly-lit frame. `collect_loot`
  had the identical bug and was surviving on luck.
- **`BiomeAmbient` was declared, read by the light solve, and written by
  nothing.** Every biome lit identically for two milestones.
- **`WeatherWeights` used an un-dithered stand-in**, so weather popped at biome
  boundaries instead of blending.

**Lesson for the reviewer**: when you add a resource that something else is
expected to fill, that failure mode is silent. A test that boots the real plugin
and demands the resource *move* is the one that catches it.

**The first of the three is now structurally impossible.** `MobSystem::events`
and `MobSystem::loot` return the VALID PREFIX — `&self.events[..self.event_count]`
— rather than the whole backing store plus a separate count. That signature was a
C API, a pointer and a length, transliterated into a language that has one type
carrying both. There is no longer an invalid tail to hand anybody, so the bug
cannot be written; `event_count()` survives only because it reads better at a call
site that just wants to know whether any arrived.

### 7.2 The capture gate is blind to motion

`frame_capture` proves a frame *is drawn*. It says nothing about whether the
frame is *stable over time*. Shake, flicker, and camera drift all pass it
cleanly — proven, not theorised, by 7.1. There is now an idle-trauma test, but a
general frame-to-frame comparison does not exist and is the single highest-value
test to add.

### 7.3 A complaint about "the bloom" was not the bloom

Reported: the underground glow was off-putting and too smooth. With
`BLOOM_INTENSITY = 0.0` the haze was **unchanged**. The cause was the coloured
light grid at `LIGHT_DOWNSCALE = 4` — a 20px texel, four times coarser than the
5px art, upscaled smoothly by design. Fixed in M10 by putting light on the cell
grid. The null result is recorded in `BLOOM_INTENSITY`'s doc so nobody reaches
for the same wrong knob.

### 7.4 Numbers that were wrong until measured

- The old bloom was **not a bloom**: up to 120 additive sprites, each 128px
  across in a 640x400 buffer. Nothing bounded what it could add to a pixel; the
  120 cap bounded sprite *count*. Over a lava lake it added +13.03 mean luminance
  and lifted 28.1% of pixels by >=32/255. The replacement adds +0.79 and 0.1%,
  which is *at* the shimmer's own frame-to-frame noise floor.
- `ParticleSystem::claim` scanned all 2048 slots to discover a full pool had no
  room: 240 emitters cost **411 µs (4.9% of a frame) spawning nothing**, paid
  exactly when the frame is busiest. `self.live` already knew. Now 801 ns.
- A doc comment claimed the `SpriteAtlases` clone wasted "a few megabytes".
  Measured: **1744 bytes**. Wrong by ~1000x. The conclusion happened to be
  right; the stated reason was invented.

### 7.5 Benchmarks lie by default

Three of the perf-pass benchmarks were wrong before they were right, and the
autopsies are kept in `docs/PERF.md` §7: a sim bench reading **13x too fast**
because criterion ran 843k ticks and the painted scene fell asleep; a window
bench measuring 1536 cells underground while its doc claimed the surface; a
particle "update" that was measuring emission. Each now carries an assertion
that would catch it again.

Also: **thermal drift across a session (11%) exceeds run-to-run variance
(3–5%)**. Two numbers in `docs/PERF.md` are only comparable if they came from the
same run.

## 8. Open items

### 8.1 Known-missing behaviour

- ~~**The burrower's breach tell (`drawTell`) is not drawn.**~~ **DONE.**
  `mobs::place_breach_tells` draws it on a source-over vertex-coloured mesh at
  `TELL_Z = 0.44`, just under the creatures. Three tests pin it, and one is worth
  knowing about: the row is the width of `MobDef::w_px`, the collision box —
  **not** `art_w_px`. Every mob in current content authors zero art padding, so
  the two are equal today and the wrong one would look right in every frame
  anyone has ever captured.
  **Not visually verified** — see §8.2. `MobSystem::place` is private, so staging
  a burrower mid-tell would mean widening core's API for a test.
- ~~**`particles.rs` glow draws as plain sprites at a higher z**~~ **DONE.**
  Luminous particles leave the sprite pool entirely for one `AdditiveMaterial`
  mesh at `PARTICLE_GLOW_Z = 0.79` — above the light composite (which ends at
  0.76), below the creature glow at 0.80, which is the order `Game.ts` ran its
  overlay callback in.
  **This fixed a live z-fight nobody had named.** The old `GLOW_Z` was `0.7`,
  which is *exactly* `light::SHADOW_Z` of `0.70`. Two quads at one depth sort
  arbitrarily, so whether a spark landed in front of the darkness multiply or
  behind it — invisible, the precise failure the glow pass exists to prevent —
  was undefined. Both orderings are now `const _: () = assert!(...)`, so they
  fail at compile rather than at test.
  Verified by eye: a 60-spark burst emitted into the lit cave reads as light
  over the composite. Note `lit_scene` has **no particles of its own** (probed:
  `live=0, glow=0`), so it cannot regress this on its own.
- **Mob art pads are all zero in current content**, and every mob authors its own
  `air` pose — so the art-rect padding and the pose fallback are correct but
  currently invisible. Tests assert the present state so the day either changes
  is loud.

### 8.2 Never visually verified by anyone

- **The ridges** — fully occluded by terrain in every captured frame. Needs a
  hilltop.
- **Twilight** — captures are mid-morning. The horizon glow is the steepest ramp
  in `sky.rs` and the main justification for its dithering, and no one has seen
  it.
- **Bloom while the camera scrolls.** The rect is cell-snapped specifically to
  stop the glow crawling and there is a test for the snap, but nobody watched it
  move.
- **`ui::paint` has never run on a real GPU in a test.** Its pure layer is
  exhaustively tested; the pooling, `ChildOf` parenting and atlas sampling are
  argued from code, not observed.
- **`WALL_DECAY`'s effect on screen.** The skylight rule for background walls is
  pinned by two unit tests, both fault-injected in each direction, and by a
  compile-time ordering assert. But nobody has SEEN it: the two capture rigs are a
  900 px-deep cave, where there is no skylight left to modulate and the rule
  correctly does nothing, and a menu-dimmed surface, which is too dark to read a
  shaft against. What is missing is a daylight surface rig with the scene forced
  to `Playing` and a shaft cut into a hillside — a third `lit_scene`-shaped file,
  and the obvious next thing to build for this feature.
- **The burrower's breach tell.** Drawn now, and its geometry is pinned by three
  tests, but nobody has watched one erupt. Staging it needs a burrower placed on
  demand and `MobSystem::place` is private; the honest options are to make that
  `pub(crate)`-plus-a-test-hook or to wait for a natural spawn with a long
  `--drive` run.

### 8.3 Documentation rot — check before trusting

Comments describing seams that have since been closed have been a recurring
problem. Known stale as of this writing:

- ~~`player_art.rs` — several `SEAM (sprite)` notes~~. Resolved: `ContentPoses`
  was confirmed dead in the game (its only constructions were the file's own
  tests) and deleted along with `PoseSheet`, `PoseIds` and `MissingPose`. The
  live path — `sprite::build_ids`, keyed by `Pose` with fallback-chain resolution
  and duplicate/cycle errors at bake time — is strictly better than the linear
  scan it replaced, which could not see fallbacks at all. Every `SEAM` note in
  the file was checked against the code; none survived as genuine.
- ~~`ui.rs` — `SEAM (sprite.rs)` and `SEAM (scenes.rs)` notes~~. Resolved: three
  notes, all closed, all rewritten to name the join in `glue.rs`. Two stale
  code sketches in those comments were deleted rather than corrected — a
  near-copy of a real `impl` is a copy that can rot, and one of them had already
  drifted (it lacked `follow_scene`'s write-only-when-different behaviour).
- ~~`mobs.rs` — `SEAM (M7)` on `Daylight`~~. Fixed: the note now names
  `daynight::advance_clock` as the writer.
- ~~`glue.rs` — `SEAM (respawn)`~~. Fixed: the note now names `start_a_run`,
  which is forty lines above it in the same file.

`caves.rs`'s two `SEAM FOR THE FEATURE PASS` notes are **genuine** and intended.

### 8.4 Tuning knobs a human should look at

| Knob | File | Why |
|---|---|---|
| `SKY_DITHER` | `sky.rs` | The halo needs it; the sun's core edge may be cleaner without. Try `0.0` |
| `LIGHT_SOFTNESS` | `light.rs` | 0.5 is a midpoint guess between "reads as masonry" and "reads as wash" |
| `COLOUR_GAIN` | `light.rs` | Strength dial. Its right value moved when the resolution changed and it has not been retuned since |
| ~~`BIOME_AMBIENT_ALPHA`~~ | `light.rs` | **DONE, 0.6 -> 0.05.** The suspicion here was right and the arithmetic makes it exact: tundra authors `[0.04, 0.07, 0.12]`, and 0.6 of that added in linear encodes to 76/255 on a black cave pixel where the original's sRGB fill added 18.4. **4.1x too bright, on blue, everywhere it was darkest.** 0.05 reproduces the original at 17.9/255. Verified as a picture at 0.6 / 0.15 / 0.05: at 0.05 the stone is stone and the ore veins are visible again |
| `START_CREATIVE` | `interact.rs` | Currently `false` (survival), matching the original |

## 9. Rust-specific work worth doing

Ordered by my estimate of value. These are leads, not verdicts — none has been
prototyped.

### 9.1 The `Arc<Mutex<_>>` seams — DONE

`Creatures`, `Pack` and `SharedPool` were all `Arc<Mutex<_>>` because a
`Box<dyn FnMut + 'static>` installed on the player had to capture them, and Bevy
cannot pass a second borrow down that call stack.

**All three are gone.** The root cause was one line — `Player::step` drove its own
projectile pool from its own last statement — and hoisting that out collapsed the
whole tower. See `docs/ARCHITECTURE.md` §3 for the full argument. What went:

- `SharedPool`, `Creatures(Arc<Mutex<_>>)`, `Pack(Arc<Mutex<_>>)`
- `ProjectileSystem::set_hit_test` / `set_on_impact`, `Player::set_ammo_source`
- `type ShotHitTest`, `type ShotImpact`, `Player::with_projectiles`
- `mobs::install_hit_test`, `items::install_ammo_source`
- the lock-ordering comment and the test named for it

What replaced them: `trait ShotWorld` and `trait AmmoSource`, both borrowed for
the length of one call; `Loadout<'a>`, which is what a body borrows while it
steps; and `glue::step_the_body`, which asks Bevy for all four borrows in its
parameters and steps the body and the pool on consecutive lines.

Two findings from doing it, neither of which §9.1 knew:

- **`Res<Creatures>` was lying to the scheduler.** Five systems took a shared
  borrow and mutated through the lock; Bevy ran them concurrently and the mutex
  serialised them at runtime. They are honest `Res`/`ResMut` now and the four
  read-only placers genuinely run in parallel.
- **There was a live lock-order inversion.** `place_shots` took creatures then
  arrows; the hit-test path took arrows then creatures. It could not deadlock only
  because Bevy runs `RunFixedMainLoop` and `Update` as separate schedules — an
  accident of engine layout, not a stated invariant. The "deadlock test" was
  single-threaded and could never have caught it.

One ordering that the locks had hidden is now stated: `items::collect_loot` is
`.after(mobs::step_creatures)`. It used to be unordered, so which ran first was
whatever the executor chose, and a drop could land a frame late.

### 9.2 The window shift — DONE

`ChunkStore::prefetch` had a rayon path and `WindowManager::load_incoming` never
reached it, loading the incoming edge one chunk at a time. It now collects the
incoming coordinates and hands them over in one call.

Measured on an M3 Pro, same criterion session (thermal drift across a session
exceeds run-to-run variance, so only same-run numbers are comparable):

| bench | before | after |
|---|---|---|
| `recenter only (window walks 2 chunks, no tick)` | 550.7 µs | 307.5 µs |
| `shift tick (window walks 2 chunks)` | 686.9 µs | 471.1 µs |

Behaviour-preserving by construction: each worker gets its own `ChunkGen` and
`worldgen_purity::parallel_matches_serial` requires byte-identity with the serial
path. `DEAD` is 1, so a recenter always moves at least two chunks along an axis —
16 chunks against a `PREFETCH_MIN_PARALLEL` of 8 — and the parallel path is taken
every time.

**Still open, and now the bigger half:** this is a *latency* problem, not a
throughput one. The stronger fix is to prefetch the leading edge BEFORE the dead
zone is crossed, off the critical path entirely. `recenter`'s one-chunk dead zone
is exactly that lookahead and it is free. Not prototyped.

### 9.3 The light blur — DONE, MEASURED, AND DELIBERATELY NOT SHIPPED

This was the largest item in `docs/PERF.md` for three milestones: the blur is
97.9 µs, 55% of the whole per-frame CPU render cost. It was built, verified,
measured against a whole frame, and **reverted**.

**The finding is worth more than the change would have been.** A new instrument,
`crates/godgame-render/tests/frame_cost.rs`, times whole headless frames. Three
builds, interleaved on a quiet machine, medians:

| build | median | vs base |
|---|---|---|
| the CPU blur, as shipped | 1313 µs | — |
| **the CPU blur DELETED entirely** | 1314 µs | **+1** |
| the blur moved to two GPU passes | 1550 µs | **+237** |

Deleting 98 µs of main-world CPU moved the frame by nothing. Bevy pipelines the
main world against the render app, so main-world CPU under the render app's cost
is free — and every number in `PERF.md`'s whole-frame CPU total is main-world
CPU. **That total is a budget, not a critical path.** The GPU version lost
237 µs, ~190 of it the two extra `Camera2d`s alone.

**What is in the tree and is not dead.** `lightblur.wgsl` and
`tests/light_blur_matches_cpu.rs` stay, in exactly the position `scan_emitters`
occupies: kept, tested, correct, not run. The harness also closes the gap this
section used to name — there IS an oracle for the light stack now, `blur_one`,
and the shader agrees with it to 9.1e-4, under a quarter of an 8-bit step. Its
fault injection is permanent rather than thrown away: a fused nine-tap triangle,
the kernel anybody would write first, is measured to be wrong by 10.3 8-bit steps
on the border, because the CPU clamps between its two boxes.

**If you come back to this**, the only version that could win is a render-graph
node inside an existing camera rather than two cameras of its own — and it would
win at most the 1 µs the middle row above says is there. Start from that row, not
from the 97.9 µs.

### 9.4 Dynamic dispatch on hot paths

`ShotWorld`, `AmmoSource`, `Projectiles` and `MobTarget` are all `dyn`. The hit
test in particular is called per live shot per fixed step, at 120 Hz. Probably
not measurable today (`MAX_SHOTS` is 24), but if projectile counts grow,
making `ProjectileSystem::update` generic over `W: ShotWorld` is a one-line
change now that the trait is a parameter rather than a stored field — which was
not true when it was a `Box<dyn FnMut>`.

### 9.5 Smaller

- `bake_vignette` is ~8.4 µs recomputed every frame from inputs that move very
  slowly. Cache it against depth and view size.
- `Player::emit` is a `VecDeque` now, not a `Vec` with `remove(0)`. At
  `MAX_EVENTS` = 32 the O(n) shift was never measurable; the point is that the
  deque SAYS the eviction policy instead of leaving a reader to infer a ring
  buffer from an index. `Array.prototype.shift()` is where that shape came from.
- `docs/PERF.md`'s light section is marked superseded but the *other* sections
  predate M10's changes; a fresh full run would be worth it.
- The whole tree is `f32` for parity with the original's `f64`-in-name-only
  arithmetic. Documented per site where precision was actually lost
  (`sprite.rs`'s `1/0.62` case, `hash_phase`'s mantissa overflow). Do not
  "upgrade" these to `f64` without reading those notes.

## 10. Process notes

Things that worked, for whoever continues:

- **Fan-out on disjoint files is safe; fan-out on one rendered image is not.**
  Three agents told to capture before/after all wrote the same PNG path and read
  each other's frames. File ownership held perfectly; the *visual* attribution in
  their reports did not.
- **`cargo fmt -p <crate>` formats the whole crate**, so concurrent agents write
  to each other's files even when scoped correctly.
- **A gate never seen to fail is unproven.** The `xtask check` runner was
  validated by deliberately breaking a test; the frame-capture gate by forcing
  the pixels black and confirming all three image assertions fail with
  actionable messages.
- **Tuning-index staleness caught real drift twice** during this session alone,
  including a genuine bug: `light.rs` declared a module-local `UNDERWORLD_DEPTH`
  (0.62, a normalised fraction) shadowing `config::worldgen::UNDERWORLD_DEPTH`
  (470, a cell count). One glob import away from silently substituting one for
  the other.

The single most useful habit: **look at the picture.** The gate stayed green
through the shake bug, a stale binary, and the light haze. All three were found
by a human opening the game.

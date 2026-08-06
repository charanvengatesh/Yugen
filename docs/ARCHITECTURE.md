# Architecture

An infinite falling-sand sandbox, native, on Bevy and wgpu. A port of a
TypeScript/Canvas2D original, which is still here as
`docs/ARCHITECTURE-ts-reference.md` — kept because it is the thing this port is
measured against, not because it describes this tree.

This document answers two questions: **where do the boundaries fall and why**,
and **where does a new thing go?** It does not describe what each file does —
every module carries a header, and those headers are the primary source. Where
one of them says something at length, this document points at it rather than
paraphrasing it badly.

---

## 1. The crate split

```
content/               authored TOML       ->  compiled by crates/contentc
   |
crates/godgame-data/   generated tables        (never edited by hand)
   |
crates/godgame-core/   the simulation          (no Bevy, no wgpu, no window)
   |
crates/godgame-render/ every Bevy-facing thing
   |
crates/godgame/        the binary
```

The arrow points one way, and `Cargo.toml` is what enforces it. `godgame-data`
depends on `bitflags` and nothing else. `godgame-core` depends on
`godgame-data`, `bitflags` and `rayon` — **not on `bevy`**. `godgame-render` is
the first crate in the graph that has heard of Bevy.

| Crate | Owns |
|---|---|
| `contentc` | the content compiler: TOML front end -> schema resolution -> Rust emitter |
| `godgame-data` | the compiled tables. Generated. Never hand-edited. |
| `godgame-core` | `config`, `sim`, `entities`, `items`, `physics`, `interact`, `input` |
| `godgame-render` | every draw pass, every plugin, every `KeyCode` |
| `godgame` | the window, `ImagePlugin::default_nearest()`, and the CLI flags |
| `xtask` | the repo gates |

### The `core` / `render` boundary is the load-bearing one

`godgame-core` may not know what a `KeyCode` is. That is not stylistic. Three
things fall out of it, and each one is a thing a test or a bench depends on:

- **The simulation is headless-testable.** `crates/godgame-core/tests/` runs a
  world, a player and 4 048 fixed steps without linking a renderer.
- **The benches never link a GPU.** `crates/godgame-core/benches/` measures
  worldgen and the sim, not Bevy startup.
- **The seams are forced into the open.** A thing that needs both sides has to
  be named, because it cannot simply reach across. That is §3.

The boundary shows up as a consistent amputation: `Player.draw`,
`Projectiles.draw`, `WorldItems.draw` and the whole DOM `Input` class did not
come across. What each of them *read* is published as an accessor, and the
drawing is reassembled in `godgame-render` — see the headers on
`crates/godgame-core/src/entities/mod.rs`, `items/mod.rs` and `input.rs`, each
of which lists exactly what it dropped and where the reader went.

`godgame-render::input` is in the render crate for the same reason and not in
`main.rs`: something has to translate `ButtonInput<KeyCode>` into
`godgame_core::input::KeyState`, and that translation belongs with the other
Bevy-facing translations.

---

## 2. What the port actually changed

Three real departures. These are the parts worth reading the source for.

### The Web Worker is gone

`crates/godgame-render/src/world.rs` carries the full argument in its header;
`sim/grid.rs` and `sim/window.rs` carry the consequences. In summary:

The TypeScript ran the automata on a `Worker`, with the grid's four cell planes
in a `SharedArrayBuffer` and **no synchronisation of any kind** — a torn cell
was accepted as cosmetically harmless. That is legal in JavaScript and undefined
behaviour in Rust, so none of it survives. Here there is ONE `CellGrid`, owned
by one resource, mutated through `&mut` from one schedule. No shared memory, no
message hop, no `unsafe`.

What that deletes, in order of how much code it was:

- **`WindowManager::attach`.** It existed only so a second `WindowManager`
  could adopt a window the first one had filled. One owner, nothing to adopt.
- **The shared control block.** An `Int32Array` with `CTRL_ORIGIN_X` /
  `CTRL_ORIGIN_Y` / `CTRL_SHIFT_GEN` slots is three plain `i32` fields.
- **`CellGrid.shared()` / `fromShared()`** and the `CellBuffers` plumbing.
- **The brush's two exits.** `Game.handleBuildInput` either posted to the worker
  or applied directly, and that asymmetry is why `Game.ts` documented a one-tick
  window in which a dug cell was still there to be dug again. There is one path
  now, so `crates/godgame-core/src/interact.rs` compensates for nothing.
- **Whole-viewport repaint.** This is the interesting one. The dirty masks were
  per-`CellGrid`-instance and the live ones were the *worker's*, on the far side
  of a buffer that carried only cells. The main thread could not see which
  chunks had changed, so it repainted everything, every frame. With one owner
  the masks are visible, and `cellmap::upload_dirty_chunks` rewrites only the
  chunks that carry a dirty bit. Dirty-rect upload is not a clever optimisation
  here — it is a direct consequence of deleting the worker.

The cost paid: no parallelism in the automata. The tick is a system on the fixed
schedule. Worldgen is where `rayon` is spent instead, which is safe because a
chunk is a pure function (§6).

### Canvas2D became wgpu, and the rasteriser exists twice

`crates/godgame-render/src/lib.rs`'s header states this; `cells.rs` and
`cellmap.rs` each argue their own half.

- **`cells.rs`** is the CPU rasteriser, ported byte-for-byte from
  `src/render/ChunkCanvas.ts`. It touches no Bevy — grid in, one packed RGBA
  `u32` per cell out — which is precisely what lets it be diffed against the
  original.
- **`cells.wgsl` / `cellmap.rs`** is the same pass on the GPU, and it is what
  the frame actually runs. One `R16Uint` texture holds the material code of
  every cell in the window; one quad samples it; the shading tables are three
  more bindings.

The CPU one stays as the **oracle**, not as dead weight:
`tests/shader_matches_cpu.rs` renders the shipping WGSL headlessly and diffs the
framebuffer against `paint_cells` over real terrain. It is also the readable
statement of what the shading *is* — `cells.rs` explains every constant the
shader merely uses, and `cells.wgsl` refers back to it by name.

The single largest CPU saving in the renderer came out of this: the TypeScript's
`updateShimmer` rebuilt ~2 500 packed palette entries every frame. On the GPU
that collapses to one float in a uniform, evaluated per fragment and only for
materials that declare shimmer.

The figure is a **mesh, not a `Sprite`**, and `shear.rs` is why: the player's
lean is a shear, a Bevy `Transform` is TRS, and TRS cannot represent one. Four
displaced vertices reproduce it *exactly* — a shear is a linear map and the
rasteriser interpolates linearly — which is a stronger claim than "close
enough". The rejected alternative (rotate about the feet) lifts the sole off the
floor by `1 - cos θ`, which on a six-pixel character is not subtle.

### Content moved from a bespoke DSL to TOML

The original had a custom line-oriented format and a lexer/parser to match. The
authoring format is now TOML (`content/FORMAT.md` is the normative spec), so
`contentc` has a `toml_in.rs` front end instead — and only the *parse* half:
nothing writes TOML back out, because a round-trip would rewrite the files and
strip the comments, which are the most valuable thing in them.

What did not change is the part that mattered: a schema is **data** — a record
of dotted field name to descriptor — and it remains the single source of truth
for validation of the source text, the shape of the emitted Rust, and which flat
tables get built. Adding a field to the game is still adding a row to a schema
under `crates/contentc/src/schemas/`. See `crates/contentc/SCHEMA-AUTHORING.md`.

Numeric ids are still pinned, in `content/ids.lock.json`; ids are assigned once,
never reused, and a deleted record leaves a tombstone.

---

## 3. The seam pattern

This is the tree's main idiom, and it is worth learning before reading anything
else. The problem recurs constantly: **A needs B, and neither should own the
other.** The answer is always the same shape — **a small trait, declared on the side that
needs the *answer*, implemented on the side that has it, and BORROWED for the
length of the call that needs it.**

`crates/godgame-render/src/glue.rs` states the principle in its header, and
states the cost of not following it: three separate modules in that crate each
grew their own world clock, and wiring all three would have run the day at
triple speed.

The instances:

| Seam | Declared as | Satisfied by |
|---|---|---|
| projectile hit test | `trait ShotWorld` (`core/entities/projectiles.rs`) | `glue::step_the_body`, forwarding to `MobSystem::hit_at` |
| ammo spend | `trait AmmoSource` (`core/entities/player.rs`) | `impl for Inventory` in `core/items/inventory.rs` |
| the player, as a target | `trait MobTarget` (`core/entities/mobs/system.rs`) | `impl for PlayerBody` in `render/mobs.rs` |
| the player's shot pool | `trait Projectiles` (`core/entities/player.rs`) | `impl for ProjectileSystem`; lent per step via `Loadout` |
| chunk persistence | `trait ChunkPersistence` (`core/sim/chunk_store.rs`) | `ChunkStore::with_persistence` |
| the icon atlas | `trait IconAtlas` (`render/ui.rs`) | `impl for SpriteAtlases` in `glue.rs` |
| which screen is up | `UiScreen`, a plain resource (`render/ui.rs`) | `glue::follow_scene`, mirroring `Scene` |

The last row is the pattern at its thinnest and shows why it is worth keeping:
`ui` must stay assertable without registering a `States` type, and `scenes` owns
the machine but should not care what draws. Six lines of mirror in `glue.rs`
buys both.

`MobTarget`'s own doc comment explains the general case better than a summary
can: it was a *structural* interface in TypeScript, satisfied by `Player` with
no import and no coupling. Rust is nominal, so the adapter has to be written
down — three lines, on the render side of the fence.

### Why nothing here captures, and what it cost to learn that

Two of these seams used to be `Box<dyn FnMut + 'static>` stored on the thing that
called them, installed once at startup — the TypeScript's shape, where a closure
over the mob system and a closure over the inventory are free. Transliterated,
that shape is expensive in a way that is worth spelling out, because it is the
single most instructive thing in this port:

> `'static` forces the capture to be **owned**. Owned forces `Arc`, because the
> host also holds it. Shared-mutable forces `Mutex`. Two locks taken in a nested
> call forces a **lock ORDER**, which is an invariant no compiler checks.

The nesting was real: a hit test fired from inside `ProjectileSystem::update`,
which was called from the last line of `Player::step`, which ran in a system
holding `ResMut<PlayerBody>` and nothing else. There was no point in that call
stack where Bevy could hand down a second borrow, so the closure had to capture —
and so `Creatures` was an `Arc<Mutex<MobSystem>>`, `Pack` an
`Arc<Mutex<Inventory>>`, and the pool an `Arc<Mutex<ProjectileSystem>>`.

Two things went wrong that the locks hid, and both are worth knowing:

- **The scheduler was lied to.** Five systems in `render/mobs.rs` took
  `Res<Creatures>` and mutated through the lock. Bevy read five shared borrows,
  ran them concurrently, and the mutex serialised them at runtime — invisibly,
  and exactly backwards from what you want.
- **The lock order was inverted.** `place_shots` took the creature lock and then
  the arrow lock; the hit-test path took them the other way round. Nothing
  deadlocked only because Bevy runs `RunFixedMainLoop` and `Update` as separate
  schedules. The test named for that hazard was single-threaded and could never
  have detected it.

Every step of that is downstream of ONE decision: that `Player::step` drove the
pool from its own last line. It does not. `glue::step_the_body` asks for all four
borrows in its parameters — which is what Bevy is for — steps the body, then
steps the pool on the very next line. Nothing captures, nothing locks, and the
`Loadout` the body borrows lives exactly as long as the call.

**The rule this leaves.** When A needs B for the duration of a call, pass B to
A's method. Reach for an owned handle only when A genuinely outlives the call,
and be suspicious when the answer is `Arc<Mutex<_>>` — in a single-schedule game
that is usually a borrow that has not been threaded far enough.

**`glue.rs` is added to the plugin group last**, so both halves of every seam
exist before anything reaches across one. Each join in it is a handful of lines
— that is the point. A seam needing more than that is usually a boundary drawn
in the wrong place.

The one genuinely new capability the port bought here is `OnEnter` / `OnExit`.
The TypeScript had nowhere to hang "do this once when the scene changes", so
`updateGameOver` called `loadLevel()` inline immediately before flipping the
scene. That is `glue::start_a_run` on `OnEnter(Scene::Playing)` now, and it runs
on *every* entry including the first — deliberately, because a special case for
"the first one" is a second code path that only ever runs once and is therefore
never really tested.

---

## 4. The schedule

One clock, two rates. `Time<Fixed>` runs at **120 Hz**, which is `STEP_DT`
(`config/physics.rs`) — the player's physics step. The automata ticks on every
SECOND fixed step, which is `SIM_HZ` = 60. An exact 2:1 ratio: the player never
integrates against a stale grid, and the sim never runs twice between two player
steps. `FixedStep::is_sim_step` is the parity check that implements it.

Inside `FixedUpdate`, in order:

| Set | System | Was |
|---|---|---|
| `SimSet::Stream` | `world::stream_window` | `windowManager.recenter(pcx, pcy)` |
| `PlayerSet::Step` | `player::step_player` | `while (acc >= STEP_DT) player.step(...)` |
| `SimSet::Simulate` | `world::simulate` | `simulate(grid)` |

and then, once per **frame** and not per substep, `player::follow_player` in
`RunFixedMainLoop`'s `AfterFixedMainLoop`.

This is `Game.updatePlaying`, decomposed. The ordering is the whole point, and
`player.rs`'s header is the place it is argued:

- `step_player` runs `.after(SimSet::Stream)`, so the window it reads was
  recentred on where the body was BEFORE this step — exactly as the
  TypeScript's `recenter` used the previous frame's `player.x`. Getting this
  backwards shows a seam at the streaming edge.
- `follow_player` eases at a per-FRAME rate because `Camera.follow` was called
  from the frame loop, not from inside the fixed accumulator. Moving it into
  `FixedUpdate` would double its speed on a 120 Hz clock and make the follow
  frame-rate dependent in a *second*, different way from the one it already is.
  (`CAMERA_EASE = 0.12` is frame-rate dependent, deliberately; a per-second
  reformulation would change the feel of every jump while claiming to be a
  port.)
- `PlayerSet::Step` exists as a named set because the arrow pool is stepped
  *inside* `Player::step`, not from a system of its own. `mobs`, `items` and
  `particles` all order themselves `.after(PlayerSet::Step)`, and none of them
  could have said that by naming `SimSet` bounds alone.

Edge-triggered input is the other half of the two-rate clock. Bevy clears
`just_pressed` once per rendered frame but `FixedUpdate` may run several times
inside one, so the edge is read once in `PreUpdate` and `Intent::for_substep`
hands it to exactly one substep. That is `Game.ts`'s
`jumpQueued && steps === 0`, moved into a resource.

`Time<Virtual>`'s max delta is clamped to `MAX_STEPS_PER_FRAME` (5) steps, so a
stalled frame drops simulation time rather than trying to catch up forever.

**The world clock is not in the sim.** `daynight.rs` ticks with the frame dt, on
the render side, because the automata must stay reproducible from
`(seed, edits)` and a wall clock feeding into it would make every chunk's
history depend on when you walked past. Its one consumer outside the draw passes
is creature spawning, which reads a *weight on which species may spawn* and
never anything that perturbs a cell. `effects.rs` states the same rule for
screenshake and the hit flash.

---

## 5. Where a tuned number goes

Three tiers, unchanged from the original in principle. Which tier is decided by
**what the number is about**, never by who reads it.

| Tier | Where | It describes |
|---|---|---|
| 1 | `content/` | one THING — a block's hardness, a mob's speed |
| 2 | `crates/godgame-core/src/config/` | the WHOLE GAME — `CELL_SIZE`, `GRAVITY` |
| 3 | a module constant | ONE ALGORITHM — an fBm octave count, a hysteresis band |

`config/` is split by domain: `world`, `view`, `worldgen`, `physics`, `combat`,
`mechanics`, `render`, `interact`. Every `pub` item under it must carry a doc
comment or `cargo xtask tuning` fails the build. Tier 3 stays put — hoisting
module constants into tier 2 would separate every number from its derivation
and rebuild the constants monolith the split exists to prevent. Discoverability
is solved by indexing, not by relocating: `docs/TUNING.md` is generated.

**The dimensional scaling rule survives verbatim.** Every distance-bearing
tunable in `config/physics.rs` is written as the value that felt right for a
`PHYS_TUNED_H`-tall character and multiplied by `scaled()`. Lengths, velocities
and accelerations scale; durations, ratios and fractions do not. Under that
rule, resizing the character preserves jump height in body-heights, run speed in
bodies/second and time to apex exactly. `scaled` is `pub` because combat and
the ability mechanics are tuned in the same frame; anything that multiplies a
velocity by its own private factor has forked the rule.

---

## 6. Testing strategy

There are six distinct kinds of check here and each catches something none of
the others can. This is the strongest part of the tree and the part most easily
weakened by accident.

**The frozen parity suites.** Four fixtures, produced by running the *original*
TypeScript under node and dumping its output. They are never regenerated from
the Rust side — that would turn the proof into a tautology. If one fails, the
port is wrong, not the test.

| Suite | Fixture | What only it can see |
|---|---|---|
| `godgame-data/tests/ts_parity.rs` | `ts-snapshot.json` | the compiler agrees: 224 id->code maps, 79 flat tables, 1 pair matrix |
| `godgame-core/tests/ts_noise_parity.rs` | `ts-noise.json` | the noise *primitives* agree, at every entry point, for four seeds |
| `godgame-core/tests/ts_worldgen_parity.rs` | `ts-worldgen.json` | the whole *pipeline* agrees — 357 chunks hashed, cell for cell |
| `godgame-render/tests/ts_cells_parity.rs` | `ts-cells.json` | the *rasteriser* agrees, pixel for pixel, over two real windows |

Noise parity says the primitives agree; worldgen parity says heightmap, biome
blend, cave lattice, depth bands, cap, shore, veins, strata and all four
decorator passes *composed in order* agree. It has already earned its keep: a
draft that floored the fractional band depth passed everything else and was
caught by the 67 cells out of 226 304 where the underworld crust ramp reads it
continuously.

`ts_player_parity.rs` (fixture `ts-player.json`, 22 scripted cases, 4 048 fixed
steps against an authored arena) is the same idea for locomotion, and its header
says precisely what it is for: coyote time read one phase too late, `wall_dir`
cleared after the collide instead of before, a jump cut applied to an
already-clamped velocity. Those are all **order**, and order is only visible in
a long replay. Unit tests of the collider cannot see any of them.

**`worldgen_purity.rs`** is the other half. It asserts the one property the
whole architecture rests on and which a single line can silently violate: a
chunk is a pure function of `(chunk_x, chunk_y, seed)`. A pure function of the
*wrong* thing passes every one of its checks, which is why the parity suite is
also needed; and a generator that matches the fixture but is order-dependent
would rearrange the world behind the player when a chunk store evicts and
regenerates. Check 8 has no TypeScript ancestor: chunks generated across a rayon
pool must be byte-identical to the serial result. That is the property the
owned-scratch design buys, and it is why worldgen contains no static with
interior mutability and no thread local.

**`shader_matches_cpu.rs`** closes the CPU/GPU chain. It compiles the shipping
WGSL, runs it on a headless adapter over a real worldgen window, reads the
framebuffer back and diffs it against `paint_cells`. Non-shimmer materials are a
pure table lookup on both sides and must be **byte identical** over ~180 000
cells; shimmer materials are recomputed per fragment in f32 against the CPU's
f64 and are bounded instead. This is what turns "does the shader look right"
into "does the shader match this verified table".

**`frame_capture.rs`** is the last link, and it catches the failure every suite
above is blind to: *the arithmetic can be perfect and the composite can still be
a black rectangle.* It boots the real `GodGameRenderPlugin` group headlessly,
lets the world live for 90 frames, captures `LowResTarget::canvas` — the actual
shipping buffer, not a re-render through a private path — writes the PNG where a
human can look at it, and asserts floors: varied, correctly oriented, not blank.
Deliberately *not* a golden image, because a strict pixel comparison would fail
on every legitimate tuning change and be deleted within a month.

Both GPU suites **skip loudly** on a machine with no usable adapter rather than
passing. A green tick from a machine that never rendered a frame is worse than a
red one, because a green tick is exactly the claim those files exist to make.

**`mob_regression.rs`** is the standing gate on the art/body split: a creature
may be redrawn, but not one tuned number may move. Comparison is exact, not
tolerant — a one-ulp difference fails, which is what lets it see a `PHYS_SCALE`
multiply that migrated from load time into the baked table.

---

## 7. The costs, plainly

Every one of these is a real trade with a visible consequence.

- **Caves are a shade less crushed than the original's.** A GPU blends in linear
  light; Canvas2D blended sRGB bytes. Multiplying in linear is brighter in the
  midtones. Fixed-function blending has nowhere else to put the operation, and
  linear is the correct place for it; matching byte for byte would need a pass
  that reads its own destination, which 2D has no way to do. See `light.rs`.
- **The automata lost its parallelism.** One owned grid, one writer, one
  schedule. The tearing, the two-owner window dance and the invisible dirty
  masks went with it, and dirty-rect upload came back — but the tick itself is
  serial now.
- **The player is `f32` where the original was `f64`.** JavaScript has one
  number type. Bit equality with the fixture is therefore impossible by
  construction, so `ts_player_parity.rs` runs two tiers: continuous quantities
  bounded (measured worst case 4.8e-4 px over 4 048 steps), discrete state
  exact.
- **The UI type is not the same type.** There was no bitmap font to port — all
  three TypeScript UI files called `ctx.fillText` with a system font, and the
  original's own header admits the deal ("text still upscales soft"). So the
  font in `ui.rs` is authored. What *is* ported faithfully is the geometry
  around the type and the measurement rules; the TypeScript's nine requested
  sizes collapse onto two faces and an integer scale, and the 10-to-14px band
  collapsing to one size is the biggest judgement call in the file. The
  hierarchy survives because the original never leant on size to carry it — it
  leant on alpha, and alpha ports exactly.

---

## 8. Where a new thing goes

| To add… | Do this |
|---|---|
| a block, item, mob, sprite, structure, worldgen feature | add a record under `content/`, run `cargo run -p contentc` |
| a FIELD on one of those | add a row to the schema in `crates/contentc/src/schemas/`, then use it — see `SCHEMA-AUTHORING.md` |
| a game-wide knob | a `pub` item in the right `crates/godgame-core/src/config/` module, **with a doc comment**, then `cargo xtask tuning` |
| a coefficient for one algorithm | a `const` in that module, next to the code, with its derivation written out |
| simulation behaviour | `godgame-core`. If you reach for `bevy::`, you are in the wrong crate. |
| a draw pass | a module + `Plugin` in `godgame-render`, added to `GodGameRenderPlugin` in `lib.rs` in paint order |
| a join between two modules that must not know each other | `glue.rs` — or a trait on the side that needs the answer, borrowed for the call |
| a mob brain | a case in `godgame-core/src/entities/mobs/brain.rs`, selected by `brain` in the `.toml` |
| anything the renderer must know about a body | an accessor on the core type. The simulation never gains a pixel of knowledge about how it looks. |

The rule behind all of it is the original's, and it still holds: **if it
describes a thing, it is content; if it describes the game, it is config; if it
describes an algorithm, it lives with the algorithm.** The port adds one more:
**if it needs Bevy, it is not in `godgame-core`.**

---

## 9. Gates

```
cargo xtask check                       # the full gate — run this before committing
cargo test --workspace                  # everything, including the parity suites
cargo run -p contentc                   # recompile content/ into godgame-data
cargo run -p contentc -- --check        # fail if the generated tables are stale
cargo clippy --workspace --all-targets  # must be zero findings
cargo fmt --all --check
```

Never hand-edit anything under `crates/godgame-data/src/` — edit `content/` and
recompile. `contentc` formats its own output, so `--check` is idempotent.

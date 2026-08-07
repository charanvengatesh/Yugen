# godgame-rs

An infinite falling-sand sandbox. A native Rust port of the TypeScript/Canvas2D
original, on Bevy and wgpu.

```
cargo run --release
```

## Layout

```
content/               authored TOML       ->  compiled by crates/contentc
   |
crates/godgame-data/   generated tables        (never edited by hand)
   |
crates/godgame-core/   the simulation          (no Bevy, no wgpu, no window)
   |
crates/godgame-render/ every draw pass
   |
crates/godgame/        the binary
xtask/                 the repo gates
```

The arrow points one way. `godgame-core` reads `godgame-data`; nothing in
`crates/` writes to `content/`. That is what makes adding a block a content
change rather than a code change.

`godgame-core` carries no Bevy, no wgpu and no windowing. That boundary is
load-bearing: it keeps the simulation headless-testable and means the worldgen
purity suite and the benches never link a renderer.

## Read next

- **`docs/ARCHITECTURE.md`** — the boundaries, why they fall where they do, what
  the port changed against the original, and where a new thing goes. Start here.
- `docs/TUNING.md` — generated index of every tuned number in the game.
- `content/FORMAT.md` — normative spec for the authoring format.
- `crates/contentc/SCHEMA-AUTHORING.md` — how to add a field to the game.
- `docs/ARCHITECTURE-ts-reference.md` — the TypeScript original's architecture.
  Kept because it is the reference this port is measured against, not because it
  describes this tree.

Every module carries a header explaining its own reasoning. Those headers are
the primary documentation; the files above are the map to them.

## Commands

| Command | Does |
|---|---|
| `cargo run --release` | play it |
| `cargo xtask check` | the full gate — run this before committing |
| `cargo test --workspace` | everything, including the parity suites |
| `cargo run -p contentc` | recompile `content/` into `godgame-data` |
| `cargo run -p contentc -- --check` | fail if the generated tables are stale |
| `cargo bench` | the worldgen and sim benchmarks |

Never hand-edit anything under `crates/godgame-data/src/` — edit `content/` and
recompile.

### Running it headless

The binary takes a few flags so an agent, or CI, can prove the thing draws
without a human at the keyboard:

| Flag | Does |
|---|---|
| `--screenshot <path>` | render a few frames, write a PNG, exit |
| `--warmup N` | frames to render before the capture fires |
| `--edit <mode> <cx> <cy> <r>` | stamp one brush stroke before capturing |
| `--free-camera` | start with no player at all |
| `--drive` | run the body by itself, so the figure moves |
| `--debug-overlay` | start with the F3 panel up (a capture has no keyboard) |
| `--play` | leave the title card at once; without it a capture photographs the menu |
| `--script FILE` | drive the whole run from a verb-per-line file, on a pinned 60 Hz clock |
| `--dump-state PATH` | write the simulation's state as JSON, for a diff rather than an eye |
| `--world DIR` | keep this world's edits in DIR; without it nothing is saved |

## Testing

Five golden baselines pin the content compiler, the noise primitives, the whole
worldgen pipeline, the player's locomotion and the cell rasteriser. They were
produced by running the *original* TypeScript under node, and while the port was
in progress they proved it faithful. **The port is over**, so they are now this
project's own regression net: they answer "has any of this moved" rather than
"does this match the original". On top of them sit a worldgen purity suite, a
GPU-against-CPU shader diff for two shaders, and a headless end-to-end frame
capture.

Two rules:

- **A baseline is a prefix, and a prefix cannot move.** New blocks, items and
  mobs are appended above the boundary; they cannot renumber or overwrite what is
  below. This is what makes new content possible — the suites used to assert an
  exact record count, and one new block failed three of them.
- **Blessing is deliberate.** `GODGAME_BLESS=1 cargo test -p godgame-data --test
  registry_golden` rewrites a baseline and leaves the diff to read. It is never
  how a red test is made green: if one goes red and you did not mean to move
  anything, the code is wrong.

`docs/ARCHITECTURE.md` §6 explains what each one catches that the others cannot.

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

## Testing

This is a port, and the suite is built around that fact. Four frozen fixtures —
produced by running the *original* TypeScript under node — pin the content
compiler, the noise primitives, the whole worldgen pipeline and the cell
rasteriser against it. They are never regenerated from the Rust side; that would
turn the proof into a tautology. On top of them sit a worldgen purity suite, a
player-locomotion replay, a GPU-against-CPU shader diff and a headless
end-to-end frame capture.

`docs/ARCHITECTURE.md` §6 explains what each one catches that the others cannot.
If a parity suite fails, the port is wrong, not the test.

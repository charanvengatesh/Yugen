# godgame-rs

An infinite falling-sand sandbox. A native Rust port of the TypeScript/Canvas2D
original, on Bevy and wgpu.

## Layout

```
content/            authored data      ->  compiled by crates/contentc
   |
crates/godgame-data/   compiler output     (never edited by hand)
   |
crates/godgame-core/   the engine          (reads generated, never writes it)
crates/godgame-render/ every draw pass
crates/godgame/        the binary
xtask/                 the repo gates
```

The arrow points one way. `godgame-core` reads `godgame-data`; nothing in
`crates/` writes to `content/`. That is what makes adding a block a content
change rather than a code change.

`godgame-core` carries no Bevy, no wgpu and no windowing. That boundary is
load-bearing: it keeps the simulation headless-testable and means the worldgen
purity suite and the benches never link a renderer.

## Commands

| Command | Does |
|---|---|
| `cargo run --release` | play it |
| `cargo run -p contentc` | recompile `content/` into `godgame-data` |
| `cargo run -p contentc -- --check` | fail if the generated tables are stale |
| `cargo xtask check` | the full gate |
| `cargo test` | worldgen purity, sim, and the TypeScript parity suite |
| `cargo bench` | sim tick, window shift, worldgen by depth band |

## Status

Ported so far:

- **M0** workspace, toolchain, content tree
- **M1** the content compiler — lexer, parser, schema resolution, code
  assignment, Rust emitter, and all six schemas. Every one of the 79 flat
  tables, the `GROW_ONTO` pair matrix and all 224 id->code mappings are verified
  byte-identical to the TypeScript build by `crates/godgame-data/tests/`.

Still to come: worldgen, the cell simulation, the wgpu renderer, the player,
mobs and items, lighting, and the UI.

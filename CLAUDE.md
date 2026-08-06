## Orientation

- `README.md` — the crate layout and what each one owns.
- `content/FORMAT.md` — normative spec for the authoring format (TOML).
- `crates/contentc/SCHEMA-AUTHORING.md` — how to add a field to the game.
- `docs/ARCHITECTURE-ts-reference.md` — the TypeScript original's architecture.
  Kept because it is the reference this port is measured against, not because it
  describes this tree.

Tuned numbers go in one of three tiers: `content/` describes one THING,
`crates/godgame-core/src/config/` describes the WHOLE GAME, a module constant
describes ONE ALGORITHM. Every `pub` item under `config/` must carry a doc
comment.

`godgame-core` carries no Bevy, no wgpu and no windowing. That boundary is
load-bearing: it keeps the simulation headless-testable and means the worldgen
purity suite and the benches never link a renderer. Do not break it.

## Gates

```
cargo test --workspace                  # everything, including the parity suites
cargo run -p contentc -- --check        # generated tables are not stale
cargo clippy --workspace --all-targets  # must be zero findings
cargo fmt --all --check
```

`cargo run -p contentc` regenerates `crates/godgame-data/` and formats its own
output, so `--check` is idempotent. Never hand-edit anything under
`crates/godgame-data/src/` — edit `content/` and recompile.

### The parity suites are the point

This is a port. Three test files compare against frozen artefacts of the
TypeScript original and must never be weakened to make a change pass:

- `crates/godgame-data/tests/ts_parity.rs` — 79 flat tables, 2629 slots, the
  pair matrix, 224 id->code mappings.
- `crates/godgame-core/tests/ts_noise_parity.rs` — every noise entry point.
- `crates/godgame-core/tests/ts_worldgen_parity.rs` — 357 chunks, cell for cell.

If one fails, the port is wrong, not the test. `crates/godgame-core/tests/worldgen_purity.rs`
is the other half: a chunk must be a pure function of `(chunk_x, chunk_y, seed)`.

## graphify

This project has a knowledge graph at `graphify-out/` with god nodes, community
structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when
  `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for
  relationships and `graphify explain "<concept>"` for focused concepts. These
  return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw
  grep output.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when
  query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current
  (AST-only, no API cost).

`graphify-out/` is gitignored — it is a local index, regenerate it rather than
committing it.

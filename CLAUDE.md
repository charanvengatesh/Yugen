## Orientation

- `README.md` — the crate layout and what each one owns.
- `content/FORMAT.md` — normative spec for the authoring format (TOML).
- `crates/contentc/SCHEMA-AUTHORING.md` — how to add a field to the game.
- `docs/ARCHITECTURE-ts-reference.md` — the TypeScript original's architecture.
  Kept because it is the reference this port is measured against, not because it
  describes this tree.

Tuned numbers go in one of three tiers: `content/` describes one THING,
`crates/yugen-core/src/config/` describes the WHOLE GAME, a module constant
describes ONE ALGORITHM. Every `pub` item under `config/` must carry a doc
comment.

`yugen-core` carries no Bevy, no wgpu and no windowing. That boundary is
load-bearing: it keeps the simulation headless-testable and means the worldgen
purity suite and the benches never link a renderer. Do not break it.

## Gates

```
cargo test --workspace                  # everything, including the parity suites
cargo run -p contentc -- --check        # generated tables are not stale
cargo clippy --workspace --all-targets  # must be zero findings
cargo fmt --all --check
```

`cargo run -p contentc` regenerates `crates/yugen-data/` and formats its own
output, so `--check` is idempotent. Never hand-edit anything under
`crates/yugen-data/src/` — edit `content/` and recompile.

### The golden baselines are the point

Five test files compare the build against a committed baseline. They were the
port's parity suites, frozen against the TypeScript original; **the port is over
and they are now this project's own regression net.** They answer "has any of
this moved", not "does this match the original".

- `crates/yugen-data/tests/registry_golden.rs` — 79 flat tables, 2629 slots,
  the pair matrix, 224 id->code mappings.
- `crates/yugen-core/tests/noise_golden.rs` — every noise entry point.
- `crates/yugen-core/tests/worldgen_golden.rs` — 357 chunks, cell for cell.
- `crates/yugen-core/tests/player_golden.rs` — 22 cases, 4048 fixed steps.
- `crates/yugen-render/tests/cells_golden.rs` — the blit, pixel for pixel.

**The baseline is a PREFIX and a prefix cannot move.** New blocks, items, mobs,
sprites and structures are APPENDED: they take codes above the boundary, are not
compared against anything, and cannot renumber or overwrite what is below. That
is what makes adding content possible at all — the old rule asserted an exact
record count and a workbench failed three suites.

**Changing a baseline is a deliberate act, never a way to fix a red test.**
`YUGEN_BLESS=1 cargo test -p yugen-data --test registry_golden` rewrites one
and leaves the diff in the tree; read it, and put it in its own commit. If a
baseline goes red and you did not mean to move anything, the code is wrong.

`crates/yugen-core/tests/worldgen_purity.rs` is the other half and has no
baseline at all: a chunk must be a pure function of `(chunk_x, chunk_y, seed)`.

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

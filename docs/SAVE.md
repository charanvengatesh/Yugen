# The save format

Normative. `crates/yugen-core/src/sim/save.rs` is the implementation; this is the
thing it has to agree with.

Two halves, and they are marked. **What it is** describes the format as it
stands today, field for field. **The rules** are policy — some of it is not yet
true of the code, and every clause that is not says so. A spec that quietly
described an intention as a fact would be worse than no spec.

---

## What it is

### The files

```
<data root>/yugen/
  options.txt                 player-level settings   (yugen-render/src/settings.rs)
  saves/
    <slug>/                   one world
      world.meta              GGWD  identity
      run.save                GGRN  the run
      chunks/<x>_<y>.chunk    GGCH  one diverged chunk each
```

The root comes from `crates/yugen/src/main.rs::saves_root()`: `$XDG_DATA_HOME`,
else `$HOME/.local/share`, `%APPDATA%` on Windows, `.` if the environment says
nothing — then `yugen/saves`. `--saves DIR` overrides it.

`<slug>` is `slug_of(name)`: lowercase ASCII alphanumerics and `-`, everything
else collapsed to `-`, ends trimmed, `world` if nothing survives, with `-2`,
`-3` appended until the directory is free. This is a **security boundary**, not
tidiness — the name comes from a text field, and joining it raw would accept
`..` and `/` and write wherever the player typed.

One file per chunk rather than one indexed file: a read is an `open` at a known
path, a write cannot corrupt a chunk it is not writing, and a file that goes bad
costs one chunk of edits rather than the world.

### Conventions

- **Little-endian throughout.** Stated once: picking native order would make a
  file's meaning depend on where it was written.
- **Magic first, version second.** A wrong file is refused, not interpreted.
- **Every rejection is a `None`, never a panic.** These bytes came off a disk
  another program can also write to.
- **Writes are atomic** — write `.tmp`, rename into place. A half-written run
  file would load as no run at all.

### `<x>_<y>.chunk` — `GGCH`, version 1, exactly 8208 bytes

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `GGCH` |
| 4 | 2 | version = 1 |
| 6 | 2 | `CHUNK_CELLS` (32) |
| 8 | 4 | `chunk_x`, i32 |
| 12 | 4 | `chunk_y`, i32 |
| 16 | 2048 | `material`, u16 × 1024 |
| 2064 | 1024 | `flags`, u8 × 1024 |
| 3088 | 2048 | `aux`, u16 × 1024 |
| 5136 | 1024 | `temp`, u8 × 1024 |
| 6160 | 2048 | `back`, u16 × 1024 |

The cell count is checked rather than assumed: a save written when
`CHUNK_CELLS` was a different number would otherwise read as this one's planes
at the wrong stride, which is garbage that parses. The coordinate in the file is
cross-checked against the one in the filename. `flags` uses
`CellFlags::from_bits_truncate`, so an unknown bit from a newer build is dropped
and the cell keeps its material.

### `run.save` — `GGRN`, version 2, variable length

In order: magic (4), version (2), `seed` u32, `clock_t` f32, body-present u8 —
and if 1, `x`, `y`, `vx`, `vy`, `facing`, `health` as f32 and `untouchable` as
u8 — then `selected` u16, worn-present u8 (and if 1, an item code u16), slot
count u16, and that many `(slot u16, code u16, count u16)` triples.

Trailing bytes after the last slot mean this is not the file it claims to be,
and it is refused.

`seed` is here so a run can be refused when it does not belong to the world it
was found beside. Slots hold **item codes, not ids** — safe only because
`content/ids.lock.json` pins id→code and `registry_golden`'s prefix rule forbids
renumbering. If either stops holding, this needs ids and a version bump: a
silently reassigned code turns the player's gold into gravel.

`BodyState` is deliberately not `Player`. Coyote time, hurt cooldown, swing
phase and pose are per-frame state, and a body that loads mid-flinch is carrying
a moment that no longer exists.

### `world.meta` — `GGWD`, version 1

Magic (4), version (2), `seed` u32, name length u16, name UTF-8 (≤ 48 bytes).
Trailing bytes are refused. The seed is recorded at **creation**, because a
world that has never been saved has no run file and would otherwise have no seed
until the first autosave.

`list_worlds` orders by the mtime of `run.save`, newest first, never-played
last, ties broken by directory name. A directory with no readable `world.meta`
is skipped rather than reported — the saves root is somewhere a player may have
put something of their own.

---

## The rules

### 1. Versioning has three levels

**This is policy. The code today implements only the first, and implements it as
a refusal rather than a migration.**

1. **The envelope version.** Bumping it is a hard event and the only thing that
   may ever cost a player a world. Old envelopes are read by *retired readers*
   that live forever — `decode_run_v2`, `decode_chunk_v1` — each a short
   function with a byte fixture pinning it. Migration is always *retired reader
   → today's in-memory struct*. There is never a chain of struct-to-struct
   migrations, because there is only ever one in-memory type.
2. **Sections inside the envelope are additive.** Adding a field is adding a
   section, not bumping a version. An unknown section is skipped by its declared
   length and counted; a known section that is absent takes a documented
   default. Neither is a refusal.
3. **Semantic reinterpretation is forbidden.** A section's payload may never
   change meaning. If `BODY` needs a seventh float it becomes `BOD2`, and
   `BODY`'s reader is retired rather than edited. This is the rule that makes
   rule 1 almost never fire.

Why this matters more than it looks: `docs/DEATH.md` alone adds a respawn point,
a death cause, a death count, a corpse bag and a permadeath flag. Under version
equality that is five bumps, each discarding the player's position and pack. And
the chunk version's current instruction — "bump on ANY layout change, including
adding a plane" — means adding one plane silently discards **every edit in every
world**, because a pristine chunk is exactly what worldgen produces and nothing
distinguishes it from a chunk whose file was refused.

### 2. What a save owns

- **Worldgen owns everything a chunk would regenerate to.** A chunk is a pure
  function of `(chunk_x, chunk_y, seed)`, and `worldgen_purity.rs` enforces it.
  **A missing chunk file is not missing data.**
- **The save owns divergence and intention**: cells the player changed, the
  body, the pack, the clock, the world's name and seed.
- **The save does not own transient population**: creatures, projectiles,
  particles, pose, coyote time. These are re-derived from world and clock on
  load. `BodyState`'s doc comment is the argument; it is policy, not a local
  decision, so that nobody adds mob persistence by accident.
- **One forced exception: dropped items.** *Not yet implemented.* `GroundItems`
  is cleared on load today, which falsifies `docs/DEATH.md` outright — a corpse
  bag that evaporates on reload is not a retrieval run.

### 3. Save cadence

**Policy; not yet implemented.** `docs/DEATH.md` raises the question and this is
the answer: **death is the most durable moment in the game, and there is exactly
one Quit.**

- One code path saves — `crates/yugen-render/src/world.rs::autosave` — and it
  stays that way. A `SaveNow` trigger extends its early-return rather than
  adding a second writer.
- Death raises `SaveNow` *before* the `GameOver` transition and *after* the
  corpse bag is spawned and the pack cleared, so what reaches disk is the
  post-death state.
- Also on entering pause, on any exit from `Scene::Playing`, and on leaving a
  world.
- **No "quit without saving" is ever offered.** The affordance is removed rather
  than policed. This is deliberate and is why the menu has one Quit.
- Permadeath belongs in `world.meta` flags, **not** `options.txt`: it changes
  what the files mean, and a global toggle would let a player flip it after
  dying.

The residual hole, stated rather than pretended away: a player can still `kill
-9` between the death and the flush. That window is one frame. Closing it
further would mean synchronous I/O on the death frame.

### 4. Known gaps

Each of these is a decision nobody wrote down, now written down.

| Not stored | Matters? |
|---|---|
| Dropped items (`GroundItems`) | **Yes** — blocks `docs/DEATH.md` |
| Respawn point, death cause, death count | **Yes** — `docs/DEATH.md` needs them |
| Permadeath flag | **Yes**, and it is a world property, not a setting |
| Created-at / last-played / play time | **Yes** — mtime is destroyed by a backup or a `cp -r` |
| Content stamp | As a **warning**, never a refusal |
| Creatures, projectiles, automata cross-tick state | No — re-derived, correctly |
| Brush radius, pause state, screen shake, coyote time | No — per-frame, correctly |

Two more worth naming:

- `DiskChunkPersistence::errors()` counts read and write failures and **nothing
  displays it**, so a read-only disk fails silently.
- `settings::path_beside(saves)` puts `options.txt` at `saves.parent()`, so
  `--saves /tmp/foo` writes `/tmp/options.txt`. The data root is the real
  concept and `saves/` is a subdirectory of it; the abstraction is inside-out.

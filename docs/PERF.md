# Performance

What was measured, on what, and what the numbers were.

The benchmarks are ports of the TypeScript original's `src/perf/` harnesses.
What carried over is the **judgement** in those files — which world states are
worth timing, which behaviours a sleep optimisation can break, which viewport
sizes matter — not their mechanism, which was a hand-rolled `hrtime` loop
because node has no criterion. Each bench module's header says what it ported,
what it deliberately did not, and why; this document is the numbers.

```
crates/godgame-core/benches/worldgen.rs   chunk generation by depth band   (pre-existing)
crates/godgame-core/benches/sim.rs        the cellular automata            (bench-sim.ts)
crates/godgame-core/benches/window.rs     the streaming window shift       (bench-window.ts)
crates/godgame-render/benches/render.rs   the per-frame CPU render work    (bench-render.ts)
```

---

## 1. The machine and the build

| | |
|---|---|
| CPU | Apple M3 Pro, 11 cores (5 performance + 6 efficiency) |
| Memory | 18 GB |
| OS | macOS 26.5.2 (Darwin 25.5.0) |
| Toolchain | rustc 1.97.1, pinned by `rust-toolchain.toml` |
| Profile | **release** — `opt-level = 3`, `lto = "thin"`, `codegen-units = 1` |

**Every number below is from a release build.** `cargo bench` selects the
`bench` profile, which inherits `release`; nothing here was measured at
`opt-level = 1`, and a debug-profile number for this codebase would be worthless
— the workspace manifest says outright that a chunk generate takes *seconds*
instead of microseconds at `opt-level = 0`.

Reproduce with:

```
cargo bench -p godgame-core -p godgame-render
```

### Run-to-run variance

Two complete back-to-back runs of the whole suite. Criterion's own
change-detection between them is the variance figure:

- **Most benchmarks: within ±3%.** The `light/*`, `cells/*` and `particles/*`
  numbers agreed between runs to better than 3% in every case.
- **Worst observed: ±5%**, on the widest `automata/swept-area scaling` point.
- **`light/pass/census` is the noisiest**, at 435–499 ns across runs (±13%).
  It is a 415 ns body, close enough to timer and cache noise that its absolute
  value should not be trusted to better than ±15%. It is also the smallest cost
  in the light pass by an order of magnitude, so this does not matter.

**Thermal drift across a session is larger than run-to-run variance and is not
noise.** The sim batch measured 37.7 ms on the first cold run of the session and
42.0 ms after ~30 minutes of continuous benching — an **11% spread** on the same
binary and the same input. Two numbers from this document should only be
compared against each other if they came from the same run. The paired runs
above were taken back to back, both warm, for that reason.

---

## 2. The budgets

The game runs a **120 Hz fixed step** (`STEP_DT = 1/120`) with the automata on
every second step (`SIM_HZ = 60`).

| Budget | Duration | What competes for it |
|---|---|---|
| One frame at 120 Hz | **8.333 ms** | render, physics, entities, and the sim on alternate steps |
| One sim tick at 60 Hz | **16.67 ms** | the automata |

Headline: **nothing measured is at or over budget.** The single most expensive
event in the game — a streaming window shift — is **8.7% of one frame**, and
everything else is under 1%. The numbers are reported against budget throughout
so that stays checkable rather than asserted.

---

## 3. The simulation

`crates/godgame-core/benches/sim.rs`. A hand-painted 352×256 window
(11×8 chunks of 32 cells), deliberately unsettled: mostly inert rock, a minority
of powder and liquid in motion, and fire/lava fronts driving the heat field, the
reaction table and the burn timers at once. Painted rather than generated
because a settled world costs almost nothing by design — that is the whole
chunk-sleep contract — and benching one would measure an iteration over an empty
awake mask.

### The tick

| | Run A | Run B | Per tick |
|---|---|---|---|
| 600 ticks, busy world | 42.046 ms | 42.013 ms | **70.1 µs** |

**70.1 µs against a 16.67 ms sim budget = 0.42%. 238× headroom.**

The batch is 600 ticks from a freshly-warmed world per criterion iteration, not
one world evolving across all of them — see §7 for why the obvious version of
this benchmark was wrong.

Activity decays across the measured batch as the scene settles, which is what
makes a single mean interpretable only alongside the curve:

```
swept cells as % of window:  t120=14.7%  t270=6.8%  t420=6.2%  t570=7.2%  t720=3.7%
world: 52 531 non-empty cells, 43/88 chunks awake at t120
digest: 53db6db8 at t120 -> e0438852 at t720
```

The digest is FNV-1a over all four cell planes. The automata is deterministic,
so for any change meant to be behaviour-preserving it must not move. It is also
the proof the timed loop was not optimised away: a deleted body cannot change a
digest, and the bench asserts it changed.

### Cost scales with swept area, linearly

The check that this benchmark measures what it names. Each iteration force-wakes
a full-height rect of the given width, so swept area is pinned rather than left
to decay:

| Forced rect | Cells swept | Run A | ns / swept cell |
|---|---|---|---|
| 1/8 window | 11 264 | 74.1 µs | 6.58 |
| 1/4 window | 22 528 | 143.4 µs | 6.36 |
| 1/2 window | 45 056 | 269.3 µs | 5.98 |
| full window | 90 112 | 540.2 µs | 6.00 |

**An 8× change in input produces a 7.3× change in time, and the per-cell cost is
flat to within 10%.** A benchmark the optimiser had elided, or one dominated by
fixed per-tick overhead, would be flat in the first column instead.

The ~6 ns per swept cell is the number to carry: the automata can sweep the
entire 90 112-cell window, twice over (both passes), in 540 µs — 3.2% of a sim
budget. There is no world state that puts this pass in trouble.

### Behavioural scenarios

Four small sealed worlds run to a full stop, checking things a sleep or
wake-coverage optimisation can plausibly break. All pass:

```
sandfall                 sand=600 levitating=0                    OK
water-level              water=336 levitating=0 surface-skew=0    OK
sand-on-draining-water   levitating=0                             OK
burn-out                 wood-left=0 fire=0                       OK
```

`sand-on-draining-water` is the one that matters most: a grain resting on liquid
that spreads out from under it in one multi-cell step. If the vacated span is
not woken, the grain hangs in mid-air forever.

---

## 4. The streaming window

`crates/godgame-core/benches/window.rs`. A real worldgen window centred on the
surface cell row, ticked 240 times until it has gone to sleep, then walked
sideways.

A shift is the one event in the sim that is **not** amortised over a frame: it
saves the trailing edge, memmoves the survivors, generates and blits the leading
edge, and hands the automata a freshly-woken region, all inside one tick.

| | Before the batch | Now | vs plain tick |
|---|---|---|---|
| Plain tick, streamed world at rest | 620 ns | 616 ns | 1× |
| `recenter` alone (walks 2 chunks) | 575.9 µs | **340.7 µs** | 553× |
| **Shift tick** (`recenter` + `simulate`) | **721.9 µs** | **494.6 µs** | **803×** |

**494.6 µs against an 8.333 ms frame = 5.9%**, down from 8.7%. Still the most
expensive single event in the game, and it now fits with 17× headroom.

The drop is `load_incoming` handing its whole incoming edge to
`ChunkStore::prefetch` in one call instead of generating chunk by chunk on the
calling thread. The "before" column is the same bench on the parent commit, run
back to back in one thermal window.

The back plane costs about 3% of that and is included above: generating walls
took `recenter` from 328.9 to 332.3 µs, because wherever the front plane is
already solid the wall is the same word and only carved cells pay an extra
`solid_at`.

Decomposition: `recenter` is **69%** of a shift tick; the sim tick over the
freshly-woken incoming edge is the other **~154 µs (31%)**. The parallel generate
shrank the numerator, so the sim's share of the event grew without the sim
getting slower.

Two chunks is not an arbitrary step size — it is the **minimum** shift the game
can perform. `WindowManager::recenter` has a one-chunk dead zone, so a drift of
one chunk returns `false` and a drift of two produces `dcx = 2`. Every real
shift loads two chunk columns × 8 rows = **16 chunks**.

**What is left here is latency, not throughput.** The generate is parallel now,
but it still happens inside the tick that crossed the dead zone. `recenter`'s
one-chunk dead zone is exactly the lookahead needed to start it a shift early,
and it is free. Not done.

The settled world is genuinely settled — **2 of 88 chunks awake, 0.05% of the
window swept per tick** — which is why the plain tick is 616 ns. After the walk
it is 34/88 chunks and 20.5%, because a shift wakes everything it loads.

---

## 5. Worldgen

`crates/godgame-core/benches/worldgen.rs`, pre-existing, re-run here so the
shift arithmetic above has a source. Per chunk, by depth band:

| Band | Run A | Run B |
|---|---|---|
| sky (above surface) | 8.69 µs | 8.96 µs |
| surface | 33.3 µs | 34.3 µs |
| cavern | 56.2 µs | 58.8 µs |
| deep | 61.1 µs | 61.7 µs |
| underworld | 28.7 µs | 29.6 µs |
| mixed (rows 0..24) | 47.3 µs | 48.0 µs |

The spread matters: a shift at cavern depth generates 16 chunks at 56 µs
(~900 µs) rather than 16 at 33 µs. That is still 11% of a frame, but it is the
worst case in the game and worth knowing.

---

## 6. The per-frame CPU render work

`crates/godgame-render/benches/render.rs`. **These are CPU-side costs, not a
frame time.** Nothing here measures the shader, the texture upload or the
compositor. The boundary is the same one the TypeScript drew with its canvas
stub, except that the port already draws it in the source: every pass benched
here is a free function or a Bevy-free struct that fills a buffer.

Viewport is `View::for_screen(2560, 1440)` — 162×92 cells, 14 904 pixels —
derived from the real zoom policy rather than hard-coded.

### The light pass, which is where the cost is

**Re-measured in full, one clean run, after M10, the window prefetch, the
background wall plane and its skylight rule.** Everything below is from that run;
the tables further down that were taken at `LIGHT_DOWNSCALE = 4` have been
replaced rather than annotated, because a superseded table nobody deletes is a
table somebody eventually quotes.

| Pass | Cost | Share of the solve |
|---|---|---|
| blur | **97.9 µs** | 68% |
| emissive splat | 23.6 µs | 16% |
| skylight flood | 18.0 µs | 13% |
| census splat | 3.7 µs | 3% |
| **`LightGrid::solve`, over lava (worst case)** | **143.6 µs** | |
| `LightGrid::solve`, at the walk midpoint | 133.2 µs | |
| `LightGrid::solve`, camera walking 6 px/frame | 133.9 µs | |

The three bakes, which are separate from the solve:

| Bake | Cost | Target |
|---|---|---|
| `bake_shadow` | 9.4 µs | 162×92 |
| `bake_colour` | 6.3 µs | 162×92 |
| `bake_vignette` | 9.8 µs | 51×30 |

**Whole light stack: ~169 µs worst case, 2.03% of an 8.33 ms frame.**

Two numbers moved for known reasons and are worth naming rather than filing as
noise:

- **`skylight` is up 7.5%** (16.7 → 18.0 µs). That is the background wall plane's
  third decay branch in `compute_skylight` — the flood now asks whether an air
  cell has a wall behind it. Paid once per light cell, and it is what makes a
  walled shaft read differently from open sky.
- **`census` reads 3.7 µs and is not in the frame.** See below.

### Which of these benchmarks describe the frame, and which describe an oracle

Three benches in this file measure code the shipping game does not run. They are
not stale and not wrong — every one of them guards something real — but adding
them to a frame budget would overstate it by about a third, and that mistake has
already been made once in this document's history.

| Bench | In the frame? | Why |
|---|---|---|
| `light/scans/scan_emitters` — 10.7 µs | **No** | Runs only under `CENSUS_NEEDED`, which is `LIGHT_DOWNSCALE > 1`. The downscale is 1. The pass is kept, tested and correct because it is what any coarser downscale needs. |
| `cells/update_shimmer` — 9.3 µs | **No** | `cellmap::update_shade_params` replaced it entirely: the shimmer is a `vec4` in a uniform and the wave is evaluated per fragment. The CPU version survives as `cells.rs`'s oracle and has one caller, a test. |
| `cells/paint_cells` — 24.2 µs at 2560×1440 | **No** | `cells.wgsl` draws the frame. `paint_cells` is the verified CPU reference `shader_matches_cpu` diffs against, and the function `ts_cells_parity` freezes. |

`light/pass/census` (3.7 µs) is the same story from the other side: the splat is
measured, and with no census to splat it does nothing in the frame.

### Still against walking

The scenario the TypeScript cared about — the surface-height cache should do no
worldgen work when nothing moves. The honest answer is that **motion is not what
drives this cost; content is:**

| | Run A | Run B |
|---|---|---|
| Camera still, over the lava sheet (worst case) | 8.41 µs | 8.62 µs |
| Camera still, at the walk's midpoint | 7.10 µs | 7.26 µs |
| Camera walking 6 px/frame | 7.30 µs | 7.50 µs |

Walking costs **3% more** than standing still at the same mean position. Moving
the camera from a lava sheet to open rock costs **18%**. The emissive census
under the light grid falls from 55 cells to 0 across the walk span, and that —
not the heightmap memo — is what the number tracks.

Getting to a comparison that says this took three attempts; see §7.

### The cell rasteriser

| | Cost | Per cell |
|---|---|---|
| `paint_cells`, headless 1000×500 (102×52) | 7.47 µs | 1.41 ns |
| `paint_cells`, 2560×1440 (162×92) | 24.2 µs | 1.63 ns |
| `update_shimmer` (palette rebuild) | 9.3 µs | view-independent |

**`paint_cells` is a cost the shipping frame does not pay.** It is the CPU
oracle that `tests/shader_matches_cpu.rs` diffs the WGSL pass against; `cellmap`
is what actually runs. The number bounds what the shading costs, and confirms
the 2.9× viewport increase produces a 3.2× time increase — it scales with pixels,
as a rasteriser should.

**`update_shimmer` is not a per-frame cost either, and this document used to say
it was.** `cellmap::update_shade_params` replaced it entirely: the shimmer clock
is one `vec4` in a uniform and the wave is evaluated per fragment, so the ~2 500
packed palette entries are no longer rebuilt at all. The CPU version survives as
`cells.rs`'s oracle and has exactly one caller, a test. The 9.3 µs is what the
replacement SAVED, not what the frame pays.

### Particles

| | Cost |
|---|---|
| `update`, full pool vs world (2 048 slots) | 9.6 µs |
| `emit` ×240, pool saturated | 779 ns |
| `claim`, one burst refused on a full pool | 3.3 ns |

4.7 ns per slot, and every one of the 2 048 slots is scanned whether or not it
holds a live particle. **0.12% of a frame.** See §8.1 for the emit path, which
was a different story until it was fixed.

Down from the 13.67 µs this table used to carry. That figure predates M10 and
several rounds of work either side of it; nothing was done to `update` itself, so
treat the difference as the two numbers having been taken on different days
rather than as an improvement anybody earned.

### Sprites (startup, not per frame)

| | Run A | Run B |
|---|---|---|
| Bake the whole sprite table (CPU half) | 49.21 µs | 49.96 µs |

51 sprites, 76 baked tiles, **1 744 bytes** of CPU pixels in total.

### Whole-frame CPU total

Only what the shipping frame actually runs — see the oracle table above for the
three benches deliberately left out.

| Pass | Cost |
|---|---|
| `LightGrid::solve` (worst case, over lava) | 143.6 µs |
| The three light bakes | 25.5 µs |
| Particle update (full 2 048-slot pool) | 9.6 µs |
| **Total** | **178.7 µs = 2.14% of an 8.333 ms frame** |

**The blur alone is 97.9 µs of that — 55% of the whole per-frame CPU render
cost, and more than everything else in the table put together.** It is already
the good algorithm: four sliding-window passes, separable, O(1) in the radius.
Written the obvious way at this resolution it measured 171 µs. There is no CPU
win left in it, which is why the remaining move is to the GPU — and why that is
the one item in this document worth a milestone rather than an afternoon.

The worst compound frame this suite can construct — a window shift, the CPU
render work, and a saturated particle emit all landing on one step — is
**~0.68 ms, 8.2% of a frame**, dominated by the 494.6 µs shift.

---

## 7. Benchmarks that were wrong, and what they said before they were fixed

Recorded because each of these produced a confident, plausible, wrong number,
and the failure modes are the ones that recur.

**The sim tick, measured over one evolving world.** The TypeScript ran 600 ticks
from a warm world and took the mean. Doing the same under `Criterion::iter`
looks identical and is not: criterion drives a sub-10 µs body ~843 000 times,
and by tick 843 000 the sand has all landed and the fires have all burnt out. It
reported **5.3 µs/tick** for a world that had been asleep for 99% of the run —
13× faster than the true 70.1 µs. Caught by an assertion on the swept-cell count
after the loop, which is now a permanent guard. Fixed by timing a fixed batch
from a rebuilt warm world per iteration.

**The window shift, measured 1 536 cells underground.** `SURFACE_ANCHOR_Y` is a
cell row; the first version multiplied it by `CHUNK_CELLS` and put the window in
the underworld under a doc comment claiming it was at the surface. It reported a
**472 µs** shift. The surface — where the player is — is **722 µs**, 53% more.
The number was real; the label was false, which is worse than no number.

**The particle update, which was measuring emission.** The refill condition was
`live_count() < full`, `full` was `MAX_PARTICLES`, and a single particle ageing
out satisfies it — so the refill ran every iteration and 240 emit calls were
timed under the name of an update. It reported **394 µs**; the update is
**13.7 µs**. Fixed with the TypeScript's every-64-frames counter, and the emit
cost now has its own benchmark, where it turned out to matter (§8).

**Light, still against walking — twice.** First version: one census computed
once and reused at every camera origin, so as the camera walked away its
emitters left the light grid, `add_census_emitters` early-outed, and walking got
steadily cheaper for a reason that existed only in the benchmark. Second
version, after giving each origin its own census: the still scenario sat on the
most emissive point in the scene and the walk moved away from it, so walking was
still "faster" — a content difference wearing a motion label. Only the third
version, which compares against a still camera at the walk's mean position,
measures motion at all. The emissive-count-by-origin table is printed by the
bench so this is visible rather than assumed.

**A `bloom_probes` throughput of 47.8 Gelem/s.** Inherited from the previous
benchmark in the criterion group, whose element count was 14 904 cells; bloom
samples on a stride the module keeps private. It now reports time per call and
no rate at all, because the only denominator this file could state would be a
guess at a constant it cannot see.

---

## 8. Performance findings — measured, not fixed

None of these were changed. Each is reported with the measurement.

### 8.1 `ParticleSystem::claim` is O(pool) on every refused emit — FIXED

> **Fixed after this report was written**, which is the outcome it was written
> for. `claim` now answers the saturated case from `self.live` before starting
> the scan, exactly as §8.1 recommends below. Re-measured on the same machine
> against criterion's stored baseline from the run described here:
>
> | | Before | After | |
> |---|---|---|---|
> | One `burst`, pool saturated (refused) | 1.733 µs | **3.33 ns** | ~520× |
> | 240 emitter calls, pool saturated | 411.0 µs | **801 ns** | **−99.807%** (p < 0.05) |
>
> The "4.9% of a frame doing nothing" below is now under 0.01%. The rest of the
> section is left exactly as written: the reasoning is what makes the fix
> legible, and the failure mode — paying a full scan to learn what a counter
> already knew — is worth recognising the next time it appears somewhere else.

`crates/godgame-render/src/particles.rs`, `claim()` around line 917.

`claim` walks the ring cursor over up to `MAX_PARTICLES` slots looking for a
dead one. On a **saturated** pool there is no dead one, so the walk runs to its
full 2 048 iterations and returns `None`, and `emit` returns having spawned
nothing. **Every refused emitter call pays a full pool scan to discover it had
nothing to do.**

| | Run A | Run B |
|---|---|---|
| One `burst`, pool saturated (refused, spawns nothing) | **1.733 µs** | 1.710 µs |
| One `burst`, pool empty (accepted) — after subtracting `clear` | **245 ns** | 244 ns |
| (`clear` alone, the baseline subtracted above) | 105.7 ns | 102.3 ns |
| 240 emitter calls, pool saturated | **411.0 µs** | 413.9 µs |

**A refused emit costs 7.1× an accepted one.** A frame in which the pool is full
and 240 emitters fire — an explosion over a splashing pool, which is exactly
when the pool *is* full — spends **411 µs, 4.9% of an 8.333 ms frame, doing
nothing at all.**

The refusal is stable and correct behaviour: overflow is dropped by design, and
the header says so. The cost of *detecting* the overflow is what is wrong. A
live count is already maintained (`self.live`), so the saturated case is
answerable in O(1) before the scan starts.

This is not currently over budget. It is reported because the cost is paid
precisely when the frame is already busiest, and because it grows with
`MAX_PARTICLES` — doubling the pool to make particles look better doubles the
cost of discovering there is no room in it.

### 8.2 The `SpriteAtlases` clone costs 1 744 bytes, not "a few megabytes" — COMMENT CORRECTED

> **The comment was rewritten after this report.** It now states the measured
> 1 744 bytes, and says plainly that the previous "few megabytes" was wrong by
> about a thousand times and had assumed the clone duplicated pixels that are in
> `Assets<Image>` by then. The conclusion is unchanged — declining the `Arc` is
> still right — but it now rests on the measurement instead of on a guess that
> happened to point the same way.

`crates/godgame-render/src/sprite.rs`, the doc comment on `SpriteAtlases`
(around line 1416) says the `Clone` derive duplicates the baked CPU pixels, that
this is "a few megabytes of waste for data already uploaded to the GPU", and
that "if the atlas ever grows a real budget this is the first thing to reach
for".

Measured, over the whole table:

```
51 sprites, 76 baked tiles, 1744 bytes of CPU pixels (1.7 KiB)
```

**The estimate is high by roughly three orders of magnitude.** The clone happens
once, in `Startup`, and copies under 2 KiB of pixels plus a handful of small
metadata vectors and handles.

The code's decision — decline to thread an `Arc` through the plugin for a
startup-only concern — is correct, and now measurably so. The comment's
justification is what is wrong, and a future reader following its advice would
spend effort on a non-problem. Left for the file's owner; this is a source
change and outside what this task may touch.

### 8.3 `scan_emitters` is the light pass's largest cost — RESOLVED, by deletion

> **No longer true, and no longer run.** At `LIGHT_DOWNSCALE = 1` the emissive
> splat visits exactly the rect this scan walked, so the census became a second
> splat of an already-splatted source — and past its 512 cap, of only the first
> 512, which drew a visible seam across a large lava lake. It is skipped
> entirely, which refunds the 11 µs below. The finding was correct when written;
> the fix was to make the pass unnecessary rather than faster.

Not a bug — a cost-distribution fact that contradicts the obvious guess.

At 162×92 cells, `scan_emitters` is **11.02 µs** against **8.41 µs** for the
entire four-pass `LightGrid::solve`. It runs at full cell resolution over
14 904 cells; the solve runs at quarter resolution over 1 050 light cells. Both
run every frame in `solve_light`.

Per cell it is already tight (0.74 ns — a table load and a compare). The cost is
the *extent*, not the inner loop, so anything done here would have to reduce how
much of the view is walked rather than how fast it is walked.

### 8.4 `bake_vignette` is 5.8% of the light stack and changes slowly

**9.8 µs** — a radial evaluation per sample over 51×30 samples, recomputed from
scratch every frame. The framing has changed since this was first written: at
`LIGHT_DOWNSCALE = 4` it was 30% of the light pass and the second largest cost in
it. At downscale 1 the blur dwarfs everything, so this is 5.8% of the light stack
and the largest of the three bakes.

Its inputs are `(view, depth, day)`. `view` changes only on a window resize;
`depth` and `day` both move continuously but *slowly* — `DAY_LENGTH_S` is 300
seconds, so `day` advances by 1/36 000 per frame at 120 Hz.

So it recomputes 1 530 samples every frame from inputs that have barely moved,
and a cache keyed on an epsilon over depth and day would skip nearly all of them.
Worth about 0.12% of a frame, which is small — but it is small, contained, and
does not need a shader.

### 8.5 `place_bloom` mutating up to 120 material assets per frame — not measured

`crates/godgame-render/src/light.rs`, `place_bloom` (around line 1982) calls
`materials.get_mut(handle.id())` once per pooled bloom sprite, up to
`BLOOM_BUDGET` = 120 of them, and `Assets::get_mut` marks the asset modified —
which queues a re-extract and a uniform re-upload per sprite per frame.

**This was flagged in the brief and I could not measure it honestly.** It is a
Bevy system whose cost is in the render-world extract and the GPU upload queue,
not in any function a criterion harness can call: `Assets<M>`, `Query` and the
render sub-app do not exist outside a running `App`. Benching it would require
booting the whole engine, at which point the measurement is dominated by
everything else in the frame and attributes nothing.

What can be said with a number: the CPU side that *feeds* it, `bloom_probes`,
costs **333 ns** and found **14** probes in this scene, well under the 120 cap.
The concern is real and unquantified; quantifying it needs a Tracy capture of a
running build (`bevy/trace_tracy`), not a criterion bench. That is the
replacement for `Profiler.ts` described in §9, and it is the right instrument
for this specific question.

---

## 9. What `Profiler.ts` became

It did not come across, and building a Rust copy of it would have been a
vestigial exercise. Here is what replaces it.

`Profiler.ts` was a **zero-allocation per-frame sampling instrument**: integer
scope ids indexing preallocated ring buffers, `PROF.begin(P_PASS2)` /
`PROF.end(...)` call sites compiled out by a `const PROFILE_BUILD` that esbuild
constant-folds away, and a p50/p95/max report at the end of a run. Every design
decision in it exists to solve one problem — **the browser gives you
`performance.now()` and nothing else**, so if you want to know what a frame
spent in the blur you have to build the instrument yourself, and it has to be
cheap enough not to perturb the thing it measures.

That problem does not exist here, and it splits into two that are already
solved:

**Offline, "what does this pass cost" → criterion.** Everything `Profiler.ts`
was pointed at in the two headless benches — `P_TICK`, `P_PASS1`, `P_PASS2`,
`P_RECENTER`, `P_PAINT`, `P_LIGHT`, `P_PARTICLES` — is a benchmark in this
document, measured better. Criterion gives a confidence interval, outlier
classification and automatic regression detection against the previous run;
`Profiler.ts` gave a p95 over a 512-sample ring. The scope list was, in effect,
a list of things that should have been benchmarks, and now is one.

**In-process, "what did *this* frame spend where" → Bevy's own diagnostics.**
This is the part criterion genuinely cannot do, and Bevy ships it:
`bevy::diagnostic::FrameTimeDiagnosticsPlugin` for frame time and FPS,
`SystemInformationDiagnosticsPlugin` for process CPU and memory, and — the real
replacement — the `trace` and `trace_tracy` features, which emit a span per
system per frame to a Tracy profiler with a timeline, per-frame drill-down and
statistics across a whole session. Bevy's scheduler already knows the boundaries
`Profiler.ts` had to be told about by hand, because a system *is* a scope.

Three things follow from the port that make the hand-rolled version pointless:

- **The `PROFILE_BUILD` compile-out has no analogue and needs none.** It existed
  so the shipping bundle carried no profiler. Bevy's tracing is a cargo feature;
  a default build has no spans in it at all, and the dead code is removed by the
  same mechanism that removes any unused feature — no `const false` trick, no
  call sites left in the source.
- **The zero-allocation ring buffer was solving a JavaScript problem.** Its
  header explains the design as avoiding a `Map<string, number[]>` — a string
  hash per call and an array push that might reallocate inside a loop being
  proven allocation-free. Neither hazard exists here.
- **The one thing it measured that nothing else does — profiler overhead itself
  (`bench-sim --overhead`) — is not a question worth asking any more,** because
  the instrument whose overhead it measured is gone.

The one capability genuinely lost is **a number for a specific frame in a
specific play session**, which is what §8.5 wants and cannot have from a
benchmark. The replacement is a Tracy capture of a running build, not a
reimplementation of the TypeScript instrument.

---

## 10. What was not ported, and why

Three of `bench-render.ts`'s measurements were deliberately dropped. Each is
explained at length in the bench module's header; in summary:

- **The two legacy blit kernels** (`paintCellsLegacy`, `paintCellsFlatLut`).
  Both were already retired in the TypeScript when that file was written, and
  existed to justify the revamp that replaced them. The Rust port only ever had
  the textured kernel. Porting two dead JavaScript functions would produce a
  ratio between two things this codebase does not contain.
- **The `fillStyle` string churn.** It measured building and GC-ing 2 048
  `rgb(r,g,b)` strings per frame, and exists because Canvas2D's only colour
  interface is a CSS string. There is no Canvas2D here and no string is built
  per particle. The cost is not reduced — it is absent.
- **The legacy skylight column.** It timed an un-memoised `surfaceRowAt` per
  light column per frame, to justify adding the memo. `Heightmap` shipped with
  the memo; there is no un-memoised version to compare against.

One scenario could **not** be reproduced faithfully and is reported as such:

- **A first-visit camera walk.** `bench-render.ts` advanced the camera
  `400 + f * 6` for 600 frames — 3 600 px, or 720 cells, across a 352-cell
  window — so two thirds of its run sampled outside the loaded grid, where every
  cell reads as air. Its "camera moving" figure was partly a measurement of an
  empty world. The port ping-pongs the camera across the ~910 px that stay
  inside the window, which keeps every sample on real terrain but leaves the
  heightmap memo (4 096 direct-mapped slots against a few hundred columns) warm.
  **The light numbers in §6 are therefore a player walking somewhere they have
  been, not a first visit.** The first-visit cost is a worldgen cost and is in
  §5, measured over never-repeated columns, rather than invented again here.

---

## 11. Conventions

- Zero `#[allow(...)]` added. There is not one in this tree and these files did
  not introduce the first.
- `cargo clippy -p godgame-core -p godgame-render --all-targets` is silent.
- No `unsafe`.
- `std::hint::black_box` on every timed body, and where black-boxing alone is
  not proof (the sim tick), an assertion on observable state after the run.

//! The per-frame CPU render work.
//!
//! Ported from `src/perf/bench-render.ts`. The scene it paints and the scenarios
//! it runs (a blit at two viewport sizes, the shimmer palette rebuild, light with
//! the camera still against the camera walking, a full particle pool) carry over
//! directly. Its mechanism does not, and neither do three of its measurements —
//! see "What did not come across" below.
//!
//! # What "CPU render work" means here, and what it excludes
//!
//! The TypeScript installed a stub `document`/canvas so it could run headless,
//! and argued that what the stub swallowed (`drawImage`, `putImageData`,
//! `fillRect`) was GPU and browser work the pass did not control, while what it
//! left running was the JavaScript that optimisation was responsible for.
//!
//! The same boundary exists here and needs no stub, because the port already
//! draws it in the source: every pass below is a free function or a Bevy-free
//! struct that fills a buffer, and the upload and the draw are somebody else's
//! problem in `cellmap`, `light`'s Bevy half and `particles`' sprite pool. So
//! these are honest CPU-side frame costs, and they are NOT a frame time. Nothing
//! here measures the shader, the upload, or the compositor — `cells.rs` is the
//! CPU ORACLE for a pass the GPU actually runs, and the numbers for it below
//! bound a cost the shipping frame does not pay.
//!
//! # The budget these are read against
//!
//! `STEP_DT` is 1/120, so a whole frame has 8.33 ms. Every number in this file
//! is per frame and competes for that, alongside the sim's own tick.
//!
//! # What did not come across
//!
//! **The two legacy blit kernels.** `bench-render.ts` reimplemented
//! `paintCellsLegacy` (a positional hash and an integer modulo per pixel) and
//! `paintCellsFlatLut` (the pre-revamp packed colour table) verbatim so a
//! before/after ran in one process on one JIT. Both were already retired in the
//! TypeScript when that file was written; the Rust port only ever had the
//! textured kernel. Porting two dead JavaScript functions into Rust would
//! produce a ratio between two things this codebase does not contain, which is
//! not a measurement of anything. The absolute cost of the live kernel is below;
//! the historical ratios stay in the TypeScript where they still mean something.
//!
//! **The `fillStyle` string churn.** That bench measured building and GC-ing
//! 2048 `rgb(r,g,b)` strings a frame, against a cache keyed on the packed
//! colour. It exists because Canvas2D's only colour interface is a CSS string.
//! There is no Canvas2D here and no string is built per particle — a colour is
//! four bytes in a vertex. The cost is not reduced, it is absent, and a Rust
//! benchmark of string formatting would measure `format!` rather than this game.
//!
//! **The legacy skylight column.** Same reason as the blit kernels: it timed a
//! pass that called `surfaceRowAt` per light column per frame with no memo, to
//! justify adding one. [`Heightmap`](godgame_core::sim::worldgen::heightmap::Heightmap)
//! shipped with the memo, so there is no un-memoised version to compare against.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use godgame_core::config::{CELL_SIZE, LIGHT_DOWNSCALE, SEED, View, WINDOW_COLS, WINDOW_ROWS};
use godgame_core::sim::grid::CellGrid;
use godgame_core::sim::materials::{CellId, block};
use godgame_data::sprites::SPRITES;

use godgame_render::cells::{CellShades, paint_cells};
use godgame_render::light::{
    EmitterScan, LightFrame, LightGrid, Rect2, bake_vignette, bloom_probes, grid_size,
    scan_emitters, vignette_size,
};
use godgame_render::particles::{MAX_PARTICLES, ParticleSystem};
use godgame_render::sprite::{BakedSprite, FromContentOpts, sprite_art_from_content};

/// Scene seed. Local to this file; see `godgame-core`'s `sim` bench for why a
/// benchmark's own stream is never the sim's.
const BENCH_SEED: u32 = 1337;

/// Camera speed for the walking scenarios, in world px per frame.
///
/// The TypeScript's `cam.snap({x: 400 + f * 6, ...})`. It is a little over a
/// cell per frame at [`CELL_SIZE`] 5, which is roughly a sprinting player, so it
/// is the fastest rate at which new content enters the view in normal play.
const WALK_PX_PER_FRAME: f32 = 6.0;

// --- The world ---------------------------------------------------------------

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(self.0) / 4_294_967_296.0
    }
}

/// The render bench's world — deliberately NOT the sim bench's.
///
/// Simpler, because nothing here cares whether the world is in motion: it cares
/// what is ON SCREEN. What it does care about is that the emissive paths have
/// real work, which is why the lava sheet and the scattered fire are here and
/// the suspended sand and the burning frames are not.
fn build_world(grid: &mut CellGrid) {
    let cols = grid.cols();
    let rows = grid.rows();
    let surface = (f64::from(rows) * 0.35) as i32;
    let mut rnd = Lcg(0x0123_4567);

    for y in surface..rows {
        for x in 0..cols {
            let cave = (f64::from(x) * 0.07 + f64::from(y) * 0.03).sin()
                + (f64::from(x) * 0.021 - f64::from(y) * 0.055).sin() * 1.3;
            if cave > 1.15 {
                continue;
            }
            let id = if y < surface + 8 {
                block::DIRT
            } else if rnd.next() < 0.06 {
                block::SAND
            } else {
                block::STONE
            };
            grid.material[(y * cols + x) as usize] = id;
        }
    }
    fill(grid, 20, surface - 20, 120, 20, block::WATER);
    fill(grid, 60, surface + 40, 50, 12, block::LAVA);

    // Scattered fire, so the emissive splat, the census and the bloom scan all
    // have single-cell sources to find rather than one convenient slab.
    for _ in 0..60 {
        let x = 20 + (rnd.next() * f64::from(cols - 40)) as i32;
        let y = surface + (rnd.next() * 60.0) as i32;
        if grid.material[(y * cols + x) as usize] == 0 {
            grid.material[(y * cols + x) as usize] = block::FIRE;
        }
    }
    grid.wake_all();
}

fn fill(grid: &mut CellGrid, x0: i32, y0: i32, w: i32, h: i32, id: CellId) {
    let cols = grid.cols();
    let rows = grid.rows();
    for y in y0.max(0)..(y0 + h).min(rows) {
        for x in x0.max(0)..(x0 + w).min(cols) {
            grid.material[(y * cols + x) as usize] = id;
        }
    }
}

fn world() -> CellGrid {
    let mut grid = CellGrid::new(WINDOW_COLS, WINDOW_ROWS);
    build_world(&mut grid);
    grid
}

/// How far the camera may travel before the view leaves the loaded window, in
/// world px, given a view `wide` cells across.
///
/// The camera PING-PONGS across this rather than walking away forever. That is a
/// real divergence from the TypeScript, which advanced `400 + f * 6` for 600
/// frames — 3600 px, or 720 cells, across a 352-cell window. Two thirds of that
/// run sampled outside the loaded grid, where every cell reads as air, so its
/// "camera moving" figure was partly a measurement of an empty world. Turning
/// round at the edge keeps every sample on real terrain.
///
/// What ping-ponging costs is the heightmap memo: 4096 direct-mapped slots
/// against a span of a few hundred columns means it is WARM after the first
/// traverse, so this measures a player walking somewhere they have been rather
/// than a first visit. The first-visit cost is a worldgen cost and is measured
/// by `godgame-core`'s `worldgen` bench, not invented again here.
fn walk_span_px(wide: i32) -> f32 {
    ((WINDOW_COLS - wide - 2).max(1) * CELL_SIZE) as f32
}

/// Ping-pong `f * step` into `[0, span)`.
fn pingpong(distance: f32, span: f32) -> f32 {
    let period = span * 2.0;
    let t = distance % period;
    if t < span { t } else { period - t }
}

// --- The cell blit -----------------------------------------------------------

/// The per-cell rasteriser, at the viewport sizes that matter.
///
/// Two sizes and not one, for the reason the TypeScript gave: the headless
/// default is a small window, and this pass has to hold up at the size a large
/// display asks for. Both are derived from [`View::for_screen`] rather than
/// hard-coded, so they track the zoom policy instead of restating a number it
/// could change out from under.
///
/// The camera pans a cell per frame so the texture and dither patterns are
/// resolved for fresh coordinates each time, as they are in play. A fixed origin
/// would let every iteration hit identical pattern offsets, which is both
/// unrepresentative and exactly the kind of accidental cache that makes a
/// benchmark faster than the code it stands for.
fn blit(c: &mut Criterion) {
    let grid = world();
    let shades = CellShades::new();
    let mut group = c.benchmark_group("cells/paint_cells");

    for (label, view) in [
        ("headless 1000x500", View::default()),
        ("display 2560x1440", View::for_screen(2560, 1440)),
    ] {
        let (w, h) = (view.cells_w(), view.cells_h());
        let cells = i64::from(w) * i64::from(h);
        let mut out = vec![0u32; (w * h) as usize];
        let mut f = 0i32;

        group.throughput(Throughput::Elements(cells.unsigned_abs()));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{label} ({w}x{h} cells)")),
            |b| {
                b.iter(|| {
                    f = (f + 1) & 255;
                    let census = paint_cells(&mut out, w, h, &grid, f, 40, &shades);
                    // The census is a real return value the light pass consumes;
                    // touching it stops the call being folded away as pure.
                    black_box(census.len())
                });
            },
        );
        println!("  {label}: {w}x{h} cells = {cells} pixels/frame");
    }
    group.finish();
}

/// The shimmer palette rebuild.
///
/// The whole per-frame cost of animated emissive blocks: no cells are visited,
/// only the shade table's emissive slices are rewritten. Cost is therefore FLAT
/// in how much lava is on screen, which is the entire point of doing it as a
/// palette cycle rather than an overlay pass — and this benchmark is the proof
/// of that claim, since it runs with no view at all.
fn shimmer(c: &mut Criterion) {
    let mut shades = CellShades::new();
    let mut t = 0.0f64;
    c.bench_function("cells/update_shimmer (view-independent)", |b| {
        b.iter(|| {
            t += 1.0 / 120.0;
            shades.update_shimmer(black_box(t));
            black_box(shades.table()[0])
        });
    });
}

// --- Light -------------------------------------------------------------------

/// The light solve, still and walking, and each pass in isolation.
///
/// Still against walking is the scenario the TypeScript cared about, because the
/// surface-height cache should do no worldgen work at all when nothing moves.
/// The per-pass breakdown is an addition: a single `solve` number says the light
/// costs what it costs, and says nothing about which of the four passes to look
/// at first if it ever stops fitting.
fn light(c: &mut Criterion) {
    let grid = world();
    let view = View::for_screen(2560, 1440);
    let (lw, lh) = grid_size(view);
    let stride = LIGHT_DOWNSCALE * CELL_SIZE;

    let (vw, vh) = (view.cells_w(), view.cells_h());
    let base_ox = 40 / LIGHT_DOWNSCALE;
    let base_oy = 40 / LIGHT_DOWNSCALE;

    let frame = |ox: i32| LightFrame {
        ox,
        oy: base_oy,
        day: 1.0,
        t: 0.0,
    };

    // ONE CENSUS PER ORIGIN, not one census reused at every origin.
    //
    // `solve_light` re-runs `scan_emitters` over the current view rect every
    // frame, so the census always tracks the camera. Holding one census fixed
    // while moving the frame — which this file did first — makes its emitters
    // fall outside the light grid as the camera walks away, so
    // `add_census_emitters` early-outs and the walking scenario gets steadily
    // cheaper for a reason that exists only in the benchmark. Precomputing one
    // per origin keeps the census faithful while still leaving the scan itself
    // outside the timed body, where it belongs: it has its own benchmark.
    let mut censuses: Vec<EmitterScan> = Vec::new();
    let ox_lo = base_ox;
    let ox_hi = base_ox + (walk_span_px(lw * LIGHT_DOWNSCALE) / stride as f32) as i32;
    for ox in ox_lo..=ox_hi {
        let mut scan = EmitterScan::default();
        scan_emitters(
            &grid,
            ox * LIGHT_DOWNSCALE,
            base_oy * LIGHT_DOWNSCALE,
            vw,
            vh,
            &mut scan,
        );
        censuses.push(scan);
    }
    let census_at = |ox: i32| &censuses[(ox - ox_lo).clamp(0, ox_hi - ox_lo) as usize];

    println!(
        "light grid: {lw}x{lh} cells at {stride}px  |  census by origin: ox={ox_lo}:{} \
         ox={ox_hi}:{}",
        census_at(ox_lo).len(),
        census_at(ox_hi).len()
    );
    assert!(
        !census_at(ox_lo).is_empty(),
        "the scene has no emitters in view — the emissive passes below would measure an early-out"
    );

    // How much of the solve cost is WHERE the camera is rather than whether it
    // is moving. `add_emissive` splats once per emissive light cell, so the cost
    // tracks how much lava and fire is under the grid — and the walking scenario
    // averages over a span of origins while the still one sits on exactly one.
    // Without this the two are not comparable, and the first run of this file
    // reported "walking is faster than standing still" with no way to see why.
    {
        let mut lg = LightGrid::new(view, SEED);
        print!("emissive cells under the light grid by origin:");
        for step in 0..6 {
            let ox = base_ox + step * 8;
            let cen = census_at(ox);
            lg.solve(&grid, frame(ox), cen.x(), cen.y());
            print!("  ox={ox}:{}(census {})", lg.hot().len(), cen.len());
        }
        println!();
    }

    let mut group = c.benchmark_group("light");
    group.throughput(Throughput::Elements(
        (i64::from(lw) * i64::from(lh)).unsigned_abs(),
    ));

    let span = walk_span_px(lw * LIGHT_DOWNSCALE);
    // The walk ping-pongs, so it spends equal time either side of the middle and
    // its mean origin is the midpoint. That is the origin the still scenario has
    // to sit at for the two to differ only in whether the camera moves.
    let mid_ox = base_ox + (span * 0.5 / stride as f32) as i32;

    // THREE still points, not one, and the third is the honest comparison.
    //
    // `base_ox` sits directly over the lava sheet — the most emissive place in
    // the scene, and the worst case for `add_emissive`. Standing there and
    // walking away from it makes walking look CHEAPER than standing still, which
    // is what the first run of this file reported, and it says nothing about
    // motion at all. See the emissive-by-origin table printed above.
    for (label, ox) in [
        ("camera still, over the lava (worst case)", base_ox),
        ("camera still, at the walk midpoint", mid_ox),
    ] {
        let mut lg = LightGrid::new(view, SEED);
        let mut t = 0.0f32;
        group.bench_function(
            BenchmarkId::from_parameter(format!("solve ({label})")),
            |b| {
                b.iter(|| {
                    t += 1.0 / 120.0;
                    let mut f = frame(ox);
                    f.t = t;
                    let cen = census_at(ox);
                    lg.solve(&grid, f, cen.x(), cen.y());
                    black_box(lg.light_at(1, 1))
                });
            },
        );
    }

    {
        let mut lg = LightGrid::new(view, SEED);
        let mut dist = 0.0f32;
        group.bench_function(
            BenchmarkId::from_parameter("solve (camera walking 6 px/frame)".to_string()),
            |b| {
                b.iter(|| {
                    dist += WALK_PX_PER_FRAME;
                    let px = pingpong(dist, span);
                    let ox = base_ox + (px / stride as f32) as i32;
                    let mut f = frame(ox);
                    f.t = dist / 120.0;
                    let cen = census_at(ox);
                    lg.solve(&grid, f, cen.x(), cen.y());
                    black_box(lg.light_at(1, 1))
                });
            },
        );
        println!(
            "  walking scenario pans over {span} px (ox {base_ox}..{}) before turning round; \
             midpoint ox={mid_ox}",
            base_ox + (span / stride as f32) as i32
        );
    }

    // The four passes `solve` is made of, so a number that grows can be
    // attributed without re-deriving this breakdown from scratch.
    for (name, which) in [
        ("skylight", 0u8),
        ("emissive", 1),
        ("census", 2),
        ("blur", 3),
    ] {
        let mut lg = LightGrid::new(view, SEED);
        // Each pass runs against a grid the earlier passes have already filled,
        // which is the state it sees in a frame. `blur` in particular reads what
        // the emissive passes wrote, and blurring a grid of zeroes is a
        // different memory access pattern from blurring a live one.
        let cen = census_at(base_ox);
        lg.solve(&grid, frame(base_ox), cen.x(), cen.y());
        group.bench_function(BenchmarkId::new("pass", name), |b| {
            b.iter(|| {
                let f = frame(base_ox);
                match which {
                    0 => lg.compute_skylight(&grid, f),
                    1 => lg.add_emissive(&grid, f),
                    2 => lg.add_census_emitters(&grid, cen.x(), cen.y(), f),
                    _ => lg.blur(),
                }
                black_box(lg.light_at(1, 1))
            });
        });
    }
    group.finish();

    // The two scans that feed the frame. Neither is inside `solve`; both run
    // every frame beside it.
    let mut scans = c.benchmark_group("light/scans");
    let mut scan = EmitterScan::default();
    scans.throughput(Throughput::Elements(
        (i64::from(vw) * i64::from(vh)).unsigned_abs(),
    ));
    scans.bench_function("scan_emitters (view rect)", |b| {
        b.iter(|| {
            scan_emitters(&grid, 40, 40, vw, vh, &mut scan);
            black_box(scan.len())
        });
    });

    scans.finish();

    let mut bloom = c.benchmark_group("light/bloom");
    let rect = Rect2 {
        x: (40 * CELL_SIZE) as f32,
        y: (40 * CELL_SIZE) as f32,
        w: view.w as f32,
        h: view.h as f32,
    };
    let mut probes = Vec::new();
    bloom_probes(&grid, rect, 0.0, &mut probes);
    println!("  bloom probes found: {}", probes.len());
    // No throughput on this one. `bloom_probes` samples on a stride the module
    // keeps private, so the only denominator this file could state would be a
    // guess at a constant it cannot see — and a per-element rate against a made-up
    // element count is worse than no rate at all. Time per call is the honest
    // figure, and the probe count above says what that call found.
    let mut t = 0.0f32;
    bloom.bench_function("bloom_probes (view rect)", |b| {
        b.iter(|| {
            t += 1.0 / 120.0;
            bloom_probes(&grid, rect, t, &mut probes);
            black_box(probes.len())
        });
    });
    bloom.finish();
}

// --- Particles ---------------------------------------------------------------

/// A full particle pool integrated against the world.
///
/// `Some(&grid)` and not `None`: collision is the expensive half of the update
/// and the frame always has a world.
///
/// # The refill has to be on a counter, not on a level
///
/// The first version of this refilled whenever `live_count()` had fallen below
/// the level the burst loop establishes. That level is [`MAX_PARTICLES`], the
/// pool is saturated by the burst loop, and a single particle ageing out is
/// enough to drop below it — so the refill ran EVERY iteration and the benchmark
/// reported the cost of 240 emit calls under the name of an update. It is on the
/// TypeScript's every-64-frames counter now, and the emit cost has its own
/// benchmark below rather than hiding inside this one.
fn particles(c: &mut Criterion) {
    let grid = world();
    let mut ps = ParticleSystem::seeded(BENCH_SEED);

    let refill = |ps: &mut ParticleSystem| {
        for i in 0..120 {
            ps.burst(100.0 + i as f32, 200.0, [200, 180, 120]);
            ps.splash(300.0 + i as f32, 220.0, [60, 120, 220]);
        }
    };
    refill(&mut ps);
    let full = ps.live_count();
    println!("particle pool: {full}/{MAX_PARTICLES} live after the refill burst");
    assert!(
        full > MAX_PARTICLES / 2,
        "the pool is only {full}/{MAX_PARTICLES} full — this is not the saturated case it claims"
    );

    let mut group = c.benchmark_group("particles");
    // Slots SCANNED, not particles integrated. `update` walks all
    // `MAX_PARTICLES` slots every call and skips the dead ones, so its cost is
    // bounded by the pool size rather than by the population — which is the
    // right denominator, and also why the live count printed afterwards varies
    // between runs without the timing moving.
    group.throughput(Throughput::Elements(MAX_PARTICLES as u64));
    let mut f: u32 = 0;
    let mut low = usize::MAX;
    group.bench_function("update (full pool, vs world)", |b| {
        b.iter(|| {
            f = f.wrapping_add(1);
            if f.is_multiple_of(64) {
                refill(&mut ps);
            }
            ps.update(1.0 / 120.0, Some(&grid));
            let n = ps.live_count();
            low = low.min(n);
            black_box(n)
        });
    });
    println!(
        "  population over the update run: {}..{full} live of {MAX_PARTICLES} slots, every one of \
         which is scanned on every call",
        low.min(full)
    );

    // The refill on its own: 240 emitter calls, which is what a violent frame
    // (an explosion over a splashing pool) actually asks for in one step.
    group.throughput(Throughput::Elements(240));
    group.bench_function("emit x240 (pool saturated)", |b| {
        b.iter(|| {
            refill(&mut ps);
            black_box(ps.live_count())
        });
    });
    group.finish();

    // --- Isolating the refusal ----------------------------------------------
    // The number above is large, and the reason is not the emitting. `claim`
    // walks the ring cursor over up to `MAX_PARTICLES` slots looking for a dead
    // one; on a SATURATED pool there is no dead one, so the walk runs to its
    // full 2048 iterations and returns `None`, and `emit` then returns having
    // spawned nothing. Every refused emitter call pays a full pool scan to
    // discover it had nothing to do.
    //
    // These two isolate that. One `burst` on a saturated pool changes no state —
    // nothing is spawned, nothing retires, the cursor returns to where it
    // started — so it is a stable steady-state measurement of pure refusal. The
    // same call on an empty pool is the cost when there IS room, and `clear` is
    // benched beside it because it has to run inside the empty-pool body to keep
    // the pool empty.
    let mut refusal = c.benchmark_group("particles/claim");
    {
        let mut full = ParticleSystem::seeded(BENCH_SEED);
        while full.live_count() < MAX_PARTICLES {
            let before = full.live_count();
            full.burst(10.0, 10.0, [1, 2, 3]);
            if full.live_count() == before {
                break;
            }
        }
        let saturated = full.live_count();
        refusal.bench_function("one burst, pool saturated (refused)", |b| {
            b.iter(|| {
                full.burst(10.0, 10.0, [1, 2, 3]);
                black_box(full.live_count())
            });
        });
        assert_eq!(
            full.live_count(),
            saturated,
            "a refused burst changed the pool — this is not measuring pure refusal"
        );
        println!("  saturated at {saturated}/{MAX_PARTICLES} live");
    }
    {
        let mut empty = ParticleSystem::seeded(BENCH_SEED);
        refusal.bench_function("clear only (the empty-pool baseline)", |b| {
            b.iter(|| {
                empty.clear();
                black_box(empty.live_count())
            });
        });
        refusal.bench_function("clear + one burst, pool empty (accepted)", |b| {
            b.iter(|| {
                empty.clear();
                empty.burst(10.0, 10.0, [1, 2, 3]);
                black_box(empty.live_count())
            });
        });
    }
    refusal.finish();
}

/// The three textures the light pass bakes on the CPU every frame.
///
/// Not part of [`LightGrid::solve`], and easy to forget for that reason — but
/// `solve_light` runs all three after it, every frame, and the vignette is a
/// radial evaluation per sample rather than a copy. They are the rest of the
/// light pass's real cost.
fn bakes(c: &mut Criterion) {
    let grid = world();
    let view = View::for_screen(2560, 1440);
    let (lw, lh) = grid_size(view);
    let (vw, vh) = vignette_size(view);

    let shades = CellShades::new();
    let (cw, ch) = (view.cells_w(), view.cells_h());
    let mut px = vec![0u32; (cw * ch) as usize];
    let census = paint_cells(&mut px, cw, ch, &grid, 40, 40, &shades);

    let mut lg = LightGrid::new(view, SEED);
    lg.solve(
        &grid,
        LightFrame {
            ox: 2,
            oy: 8,
            day: 1.0,
            t: 0.0,
        },
        census.x(),
        census.y(),
    );

    let mut shadow = vec![0u8; (lw * lh * 4) as usize];
    let mut colour = vec![0u8; (lw * lh * 4) as usize];
    let mut vignette = vec![0u8; (vw * vh * 4) as usize];
    println!("bake targets: shadow/colour {lw}x{lh}, vignette {vw}x{vh}");

    let mut group = c.benchmark_group("light/bake");
    group.bench_function("shadow", |b| {
        b.iter(|| {
            lg.bake_shadow(0.5, 1.0, &mut shadow);
            black_box(shadow[0])
        });
    });
    group.bench_function("colour", |b| {
        b.iter(|| {
            lg.bake_colour(&mut colour);
            black_box(colour[0])
        });
    });
    group.bench_function("vignette", |b| {
        b.iter(|| {
            bake_vignette(view, 0.5, 1.0, &mut vignette);
            black_box(vignette[0])
        });
    });
    group.finish();
}

// --- Sprites -----------------------------------------------------------------

/// The whole sprite table's CPU bake, and what the baked pixels weigh.
///
/// Not a per-frame cost: this runs once in `PreStartup`. It is here because
/// [`SpriteAtlases`](godgame_render::sprite::SpriteAtlases) derives `Clone` and
/// its own doc comment admits the clone duplicates the baked CPU pixels — "a few
/// megabytes of waste", written down so that if the atlas ever grows a budget
/// this is the first thing to reach for. That estimate had never been measured.
/// The footprint printed below is the measurement, and it is the number that
/// decides whether the `Arc` the comment declines to thread is worth threading.
///
/// Only the CPU half is timed. `SpriteAtlas` additionally uploads through
/// `Assets<Image>`, which needs a Bevy `App`, and a bench that booted one would
/// be timing asset-server plumbing rather than the rasteriser.
fn sprites(c: &mut Criterion) {
    let opts = FromContentOpts::default();
    let mut bytes = 0usize;
    let mut tiles = 0usize;
    for def in SPRITES.iter() {
        let art = sprite_art_from_content(def, def.id, &opts).expect("content/sprites");
        let baked = BakedSprite::new(&art, def.id).expect("content/sprites");
        bytes += baked.pixels().len();
        tiles += baked.bake_count;
    }
    println!(
        "sprite table: {} sprites, {tiles} baked tiles, {bytes} bytes of CPU pixels ({:.1} KiB) — \
         the amount every `SpriteAtlases` clone duplicates",
        SPRITES.len(),
        bytes as f64 / 1024.0
    );

    c.bench_function("sprite/bake whole table (startup, not per frame)", |b| {
        b.iter(|| {
            let mut n = 0usize;
            for def in SPRITES.iter() {
                let art = sprite_art_from_content(def, def.id, &opts).expect("content/sprites");
                let baked = BakedSprite::new(&art, def.id).expect("content/sprites");
                n += baked.pixels().len();
            }
            black_box(n)
        });
    });
}

criterion_group!(benches, blit, shimmer, light, bakes, particles, sprites);
criterion_main!(benches);

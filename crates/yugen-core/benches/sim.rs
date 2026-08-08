//! The cellular automata's hot loop, and the behavioural contracts around it.
//!
//! Ported from `src/perf/bench-sim.ts`. What carried over is the JUDGEMENT in
//! that file — which world states are worth timing and which behaviours a sleep
//! optimisation can plausibly break — not its mechanism, which was a hand-rolled
//! `hrtime` loop because node has no criterion.
//!
//! # Why the world is painted rather than generated
//!
//! The point is to measure the automata DOING WORK. A settled world costs almost
//! nothing by design — that is the entire chunk-sleep contract — so benching one
//! measures an iteration over an empty awake mask and nothing else. Worldgen
//! produces exactly such a world: rock that has never moved and never will.
//!
//! So the scene is painted by hand, in proportions chosen so every branch of the
//! movement pass is hit in roughly the ratio a real play session hits it: mostly
//! inert rock (the case the sweep must early-out on), a minority of powder and
//! liquid genuinely in motion, and a handful of fire/lava fronts driving the heat
//! field, the reaction table and the burn timers at once.
//!
//! # Two RNGs, deliberately
//!
//! The scene is built from a local LCG and never from [`SimRng`](yugen_core::sim::rng).
//! Drawing scene randomness from the sim's stream would advance it, changing what
//! the automata sees and making two runs incomparable. The sim stream is reseeded
//! to a fixed value before every world instead, so two runs of the same build are
//! bit-identical and a timing difference is a timing difference.
//!
//! # The digest
//!
//! FNV-1a over all four cell planes, printed before the timings. The automata is
//! deterministic (seeded RNG, positional hashing), so for any change meant to be
//! behaviour-preserving the digest after N ticks must be IDENTICAL to the
//! previous run's. A change that moves it has changed the simulation and has to
//! be justified as a deliberate trade-off rather than waved through.
//!
//! It is also the anti-optimiser guard this file needs. `black_box` stops the
//! compiler discarding the call; the digest is the proof the call did something,
//! because a world whose digest still equals its pre-tick digest was not
//! simulated at all.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use yugen_core::config::{CHUNK_CELLS, WINDOW_COLS, WINDOW_ROWS};
use yugen_core::sim::automata::Automata;
use yugen_core::sim::grid::CellGrid;
use yugen_core::sim::materials::{CellId, MaterialState, block};

/// Sim-stream seed. Fixed so a run is bit-identical to the last one.
///
/// Named `BENCH_SIM_SEED` rather than `SEED` so it cannot be read as the world
/// seed in `config::worldgen`, which is a different number with a different job —
/// the same care `worldgen.rs` takes with `BENCH_SEED`.
const BENCH_SIM_SEED: u32 = 1337;

/// Ticks run before anything is timed.
///
/// The TypeScript needed 120 to let V8 tier up. There is no interpreter here, so
/// this buys only the other half of what that warmup bought: the scene reaching
/// its steady state. The suspended sand has to land, the fires have to take hold.
/// A world one tick after painting is not the world this bench claims to measure.
const WARMUP_TICKS: u32 = 120;

// --- Scene RNG ---------------------------------------------------------------

/// The scene-building stream. Numerically the TypeScript's LCG, so the painted
/// world is the same world.
struct Lcg(u32);

impl Lcg {
    fn new() -> Lcg {
        Lcg(0x0123_4567)
    }

    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(self.0) / 4_294_967_296.0
    }
}

// --- Painting ----------------------------------------------------------------

/// Write a filled rect straight into the material plane, clipped to the grid.
///
/// Deliberately not [`CellGrid::set`]: that wakes chunks and grows activity
/// boxes cell by cell, which is thousands of redundant unions while painting a
/// world that is about to be woken wholesale anyway.
fn fill_rect(grid: &mut CellGrid, x0: i32, y0: i32, w: i32, h: i32, id: CellId) {
    let cols = grid.cols();
    let rows = grid.rows();
    for y in y0..y0 + h {
        if y < 0 || y >= rows {
            continue;
        }
        for x in x0..x0 + w {
            if x < 0 || x >= cols {
                continue;
            }
            grid.material[(y * cols + x) as usize] = id;
        }
    }
}

/// Paint the benchmark world. See the module header for why these proportions.
fn build_world(grid: &mut CellGrid) {
    let cols = grid.cols();
    let rows = grid.rows();
    let surface = (f64::from(rows) * 0.35) as i32;
    let mut rnd = Lcg::new();

    for y in surface..rows {
        for x in 0..cols {
            // Caves: two cheap sine bands, so the terrain is not a solid slab.
            // A solid slab hides the cost of everything that has to look at
            // empty space, which is most of the movement pass.
            let cave = (f64::from(x) * 0.07 + f64::from(y) * 0.03).sin()
                + (f64::from(x) * 0.021 - f64::from(y) * 0.055).sin() * 1.3;
            if cave > 1.15 {
                continue;
            }
            let id = if y < surface + 8 {
                block::DIRT
            } else if rnd.next() < 0.04 {
                block::GRAVEL
            } else {
                block::STONE
            };
            grid.material[(y * cols + x) as usize] = id;
        }
    }

    // Liquid bodies. Dropped into carved basins so they actually flow and level
    // rather than sitting in a pre-solved column.
    fill_rect(grid, 20, surface - 26, 90, 26, block::WATER);
    fill_rect(grid, 150, surface - 18, 60, 18, block::WATER);
    fill_rect(grid, 250, surface - 14, 70, 14, block::OIL);
    fill_rect(grid, 60, surface + 40, 40, 12, block::LAVA);
    fill_rect(grid, 230, surface + 60, 34, 10, block::LAVA);

    // Powder in flight: columns of sand suspended over open space. Guaranteed
    // motion for the whole run, which is what keeps a meaningful fraction of
    // chunks awake — see the stationarity note on `busy_tick`.
    for c in 0..14 {
        let x = 12 + c * 24;
        for y in 6..surface - 30 {
            if rnd.next() < 0.55 {
                grid.material[(y * cols + x) as usize] = block::SAND;
            }
            if rnd.next() < 0.25 {
                grid.material[(y * cols + x + 1) as usize] = block::SAND;
            }
        }
    }

    // Combustible structures: wood frames with a fire seed at the base. Burn
    // timers, contact ignition, smoke birth and the autoignition threshold all
    // run every tick here.
    for s in 0..6 {
        let bx = 30 + s * 52;
        let by = surface + 12;
        fill_rect(grid, bx, by, 14, 2, block::WOOD);
        fill_rect(grid, bx, by - 10, 2, 10, block::WOOD);
        fill_rect(grid, bx + 12, by - 10, 2, 10, block::WOOD);
        grid.material[((by + 1) * cols + bx + 6) as usize] = block::FIRE;
    }

    // Growth and melt fronts: moss creeps, ice sits near lava.
    for _ in 0..40 {
        let x = (rnd.next() * f64::from(cols - 2)) as i32;
        let y = surface + 4 + (rnd.next() * f64::from(rows - surface - 8)) as i32;
        if grid.material[(y * cols + x) as usize] == block::STONE {
            grid.material[(y * cols + x) as usize] = block::MOSS;
        }
    }
    fill_rect(grid, 100, surface + 44, 18, 8, block::ICE);

    grid.wake_all();
}

/// A painted world and a sim seeded to match it.
fn make_world() -> (CellGrid, Automata) {
    let mut grid = CellGrid::new(WINDOW_COLS, WINDOW_ROWS);
    let mut sim = Automata::new();
    sim.seed(BENCH_SIM_SEED);
    build_world(&mut grid);
    (grid, sim)
}

// --- Measurements taken off the world, not off the clock ---------------------

/// FNV-1a over every cell plane. The regression signal — see the module header.
fn digest(grid: &CellGrid) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    let mut mix = |v: u32| {
        h = (h ^ v).wrapping_mul(16_777_619);
    };
    for i in 0..grid.material.len() {
        mix(u32::from(grid.material[i]));
        mix(u32::from(grid.flags[i].bits()));
        mix(u32::from(grid.aux[i]));
        mix(u32::from(grid.temp[i]));
    }
    h
}

/// Non-empty cells — the sanity check that the world did not evaporate.
fn census(grid: &CellGrid) -> usize {
    grid.material.iter().filter(|&&m| m != 0).count()
}

/// Cells inside the activity boxes the last tick actually swept.
///
/// The cost driver the sleep design exists to shrink: an awake CHUNK is 1024
/// cells, but the box inside it is only the part where something is happening,
/// and the movement pass walks the box. Reported as a fraction of the window so
/// "how much of the world is alive" is one number.
fn swept_cells(grid: &CellGrid) -> i64 {
    grid.awake_chunks()
        .iter()
        .map(|b| i64::from((b.x1 - b.x0).max(0)) * i64::from((b.y1 - b.y0).max(0)))
        .sum()
}

// --- Behavioural scenarios ---------------------------------------------------

/// Small sealed worlds that DO reach rest, checked for the things a sleep or
/// wake-coverage optimisation can plausibly break.
///
/// The benchmark world never fully settles (fires relight, liquids keep finding
/// level), so a levitation count taken there is a snapshot that includes matter
/// legitimately in flight. These run to a full stop instead, so "a grain is in
/// the air" is unambiguous: nothing is moving, therefore anything unsupported is
/// stuck.
///
/// Each prints a verdict rather than asserting. This is a measuring instrument,
/// and a failure here is information the operator should read next to the
/// timings — not a panic that hides them.
struct Scenario {
    name: &'static str,
    ticks: u32,
    build: fn(&mut CellGrid),
    check: fn(&CellGrid) -> String,
}

/// Powder or liquid cells with nothing underneath them.
fn levitating_in(grid: &CellGrid, x0: i32, y0: i32, x1: i32, y1: i32) -> u32 {
    let cols = grid.cols();
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let id = grid.material[(y * cols + x) as usize];
            if id == 0 {
                continue;
            }
            if !matches!(
                MaterialState::of(id),
                MaterialState::Powder | MaterialState::Liquid
            ) {
                continue;
            }
            if grid.material[((y + 1) * cols + x) as usize] == 0 {
                n += 1;
            }
        }
    }
    n
}

fn count_of(grid: &CellGrid, id: CellId) -> usize {
    grid.material.iter().filter(|&&m| m == id).count()
}

/// Stone shell around a rectangle, so nothing escapes the test area.
fn boxed(grid: &mut CellGrid, x0: i32, y0: i32, x1: i32, y1: i32) {
    fill_rect(grid, x0, y0, x1 - x0, 1, block::STONE);
    fill_rect(grid, x0, y1 - 1, x1 - x0, 1, block::STONE);
    fill_rect(grid, x0, y0, 1, y1 - y0, block::STONE);
    fill_rect(grid, x1 - 1, y0, 1, y1 - y0, block::STONE);
}

fn verdict(ok: bool) -> &'static str {
    if ok { "OK" } else { "FAIL" }
}

const SCENARIOS: [Scenario; 9] = [
    Scenario {
        // A pile forms and every grain ends up supported. The basic wake
        // contract, and the cheapest thing that breaks when it is wrong.
        name: "sandfall",
        ticks: 900,
        build: |grid| {
            boxed(grid, 10, 10, 70, 70);
            fill_rect(grid, 20, 12, 30, 20, block::SAND);
        },
        check: |grid| {
            let lev = levitating_in(grid, 11, 11, 69, 69);
            let n = count_of(grid, block::SAND);
            format!(
                "sand={n} levitating={lev} {}",
                verdict(lev == 0 && n == 600)
            )
        },
    },
    Scenario {
        // Liquid finds its level across a wide basin. This is the path that
        // moves several cells at once, which is where a bounding-box wake can
        // under-cover.
        name: "water-level",
        ticks: 1200,
        build: |grid| {
            boxed(grid, 10, 80, 120, 130);
            fill_rect(grid, 12, 100, 12, 28, block::WATER);
        },
        check: |grid| {
            let lev = levitating_in(grid, 11, 81, 119, 129);
            let n = count_of(grid, block::WATER);
            // Surface flatness: highest water row on the left third against the
            // right third. A liquid that stopped spreading early is level
            // locally and stepped globally, which no levitation count catches.
            let cols = grid.cols();
            let top_in = |xa: i32, xb: i32| {
                (81..129)
                    .find(|&y| {
                        (xa..xb).any(|x| grid.material[(y * cols + x) as usize] == block::WATER)
                    })
                    .unwrap_or(i32::MAX)
            };
            let skew = (top_in(12, 40) - top_in(90, 119)).abs();
            format!(
                "water={n} levitating={lev} surface-skew={skew} {}",
                verdict(lev == 0 && n == 336 && skew <= 2)
            )
        },
    },
    Scenario {
        // THE case the activity box is most likely to get wrong: a grain resting
        // on liquid that spreads out from under it in one multi-cell step. If
        // the vacated span is not woken, the grain hangs in mid-air forever.
        name: "sand-on-draining-water",
        ticks: 900,
        build: |grid| {
            boxed(grid, 10, 140, 160, 190);
            fill_rect(grid, 12, 186, 146, 3, block::WATER);
            fill_rect(grid, 70, 170, 6, 16, block::WATER);
            fill_rect(grid, 70, 166, 6, 4, block::SAND);
        },
        check: |grid| {
            let lev = levitating_in(grid, 11, 141, 159, 189);
            format!("levitating={lev} {}", verdict(lev == 0))
        },
    },
    Scenario {
        // Fire has to consume its fuel and go out, leaving nothing burning.
        name: "burn-out",
        ticks: 1500,
        build: |grid| {
            boxed(grid, 200, 10, 300, 60);
            fill_rect(grid, 210, 55, 80, 4, block::WOOD);
            let cols = grid.cols();
            grid.material[(54 * cols + 250) as usize] = block::FIRE;
        },
        check: |grid| {
            let wood = count_of(grid, block::WOOD);
            let fire = count_of(grid, block::FIRE);
            format!(
                "wood-left={wood} fire={fire} {}",
                verdict(wood < 320 && fire == 0)
            )
        },
    },
    Scenario {
        // Oil poured onto a water pool must FLOAT: it is less dense, so it can
        // fall through air but not displace water downward. The layering — oil
        // band above, water band below — is the whole check, and it is the
        // baseline the viscosity work must not break.
        name: "oil-on-water",
        ticks: 1400,
        build: |grid| {
            boxed(grid, 10, 80, 90, 130);
            fill_rect(grid, 12, 110, 76, 18, block::WATER);
            fill_rect(grid, 30, 84, 20, 10, block::OIL);
        },
        check: |grid| {
            let lev = levitating_in(grid, 11, 81, 89, 129);
            let oil = count_of(grid, block::OIL);
            let cols = grid.cols();
            // Mean row of each liquid: floating means oil's mean row is ABOVE
            // (smaller than) water's by a clear margin.
            let mean_row = |id: CellId| {
                let (mut sum, mut n) = (0i64, 0i64);
                for y in 81..129 {
                    for x in 11..89 {
                        if grid.material[(y * cols + x) as usize] == id {
                            sum += i64::from(y);
                            n += 1;
                        }
                    }
                }
                if n == 0 { 0.0 } else { sum as f64 / n as f64 }
            };
            let sep = mean_row(block::WATER) - mean_row(block::OIL);
            format!(
                "oil={oil} levitating={lev} separation={sep:.1} {}",
                verdict(lev == 0 && oil == 200 && sep > 3.0)
            )
        },
    },
    Scenario {
        // A dam with a breach at its base. The first version of this check
        // demanded both chambers find ONE level, and it failed — correctly,
        // because this automata has NO HYDROSTATIC PRESSURE: water moves by
        // falling and by surface spreading, so nothing can push the far side
        // UP above the breach's own top. The honest equilibrium is "the right
        // chamber fills exactly to the breach top and everything comes to
        // rest", and that is what is asserted. (Pressure is a feature this
        // scenario would be the test for, the day someone builds it.)
        //
        // What it still catches: a liquid that goes to sleep mid-flow (the
        // wake bug) never fills the right chamber at all, and a viscosity gate
        // that strands cells shows up in the levitation count.
        name: "dam-break",
        ticks: 2400,
        build: |grid| {
            boxed(grid, 10, 80, 120, 130);
            // the dam, with a breach at the floor
            fill_rect(grid, 60, 81, 2, 44, block::STONE);
            // water on the left only, well above the breach
            fill_rect(grid, 12, 90, 46, 35, block::WATER);
        },
        check: |grid| {
            let lev = levitating_in(grid, 11, 81, 119, 129);
            let cols = grid.cols();
            let top_in = |xa: i32, xb: i32| {
                (81..129)
                    .find(|&y| {
                        (xa..xb).any(|x| grid.material[(y * cols + x) as usize] == block::WATER)
                    })
                    .unwrap_or(129)
            };
            let (l, r) = (top_in(12, 58), top_in(64, 118));
            // The breach spans rows 125..129; a filled right chamber surfaces
            // at the breach top.
            format!(
                "left-top={l} right-top={r} levitating={lev} {}",
                verdict(lev == 0 && r == 125 && l < 125)
            )
        },
    },
    Scenario {
        // A steam pocket sealed UNDER a water pool. The automata's own doc on
        // `rise_into` promises a gas can "displace a denser fluid above it";
        // the body only accepts EMPTY, so today the steam is trapped and this
        // prints FAIL. It is written against the PROMISED behaviour on
        // purpose: the gas-density fix flips it to OK, and anything that
        // breaks it afterwards re-prints the lie.
        name: "steam-under-water",
        ticks: 900,
        build: |grid| {
            boxed(grid, 10, 80, 60, 130);
            fill_rect(grid, 12, 100, 46, 20, block::WATER);
            // the pocket, sealed by a stone shelf below and water above
            fill_rect(grid, 30, 122, 6, 3, block::STEAM);
            fill_rect(grid, 12, 125, 46, 3, block::STONE);
        },
        check: |grid| {
            let cols = grid.cols();
            // Any steam above the pool's midline counts as escaped; total
            // steam reaching zero also counts (it condensed or vented).
            let mut above = 0;
            for y in 81..110 {
                for x in 11..59 {
                    if grid.material[(y * cols + x) as usize] == block::STEAM {
                        above += 1;
                    }
                }
            }
            let trapped = count_of(grid, block::STEAM);
            format!(
                "steam-above-pool={above} steam-total={trapped} {}",
                verdict(above > 0 || trapped == 0)
            )
        },
    },
    Scenario {
        // Tar is the viscosity system's extreme case, so this is ITS scenario:
        // a tar column dropped over a water pool must (a) still be moving well
        // after water would have levelled -- the ooze -- and (b) end up UNDER
        // the water, because it is denser and `flow_into` lets a heavy liquid
        // fall through a lighter one. A tar that levels as fast as water means
        // the gate is dead; a tar floating ON water means the density path
        // broke; tar cells stranded in the air mean a skipped tick failed to
        // wake -- three failure modes, one pour.
        name: "tar-through-water",
        ticks: 3000,
        build: |grid| {
            boxed(grid, 10, 80, 70, 130);
            fill_rect(grid, 12, 112, 56, 16, block::WATER);
            fill_rect(grid, 30, 84, 12, 8, block::TAR);
        },
        check: |grid| {
            let lev = levitating_in(grid, 11, 81, 69, 129);
            let tar = count_of(grid, block::TAR);
            let cols = grid.cols();
            let mean_row = |id: CellId| {
                let (mut sum, mut n) = (0i64, 0i64);
                for y in 81..129 {
                    for x in 11..69 {
                        if grid.material[(y * cols + x) as usize] == id {
                            sum += i64::from(y);
                            n += 1;
                        }
                    }
                }
                if n == 0 { 0.0 } else { sum as f64 / n as f64 }
            };
            let sink = mean_row(block::TAR) - mean_row(block::WATER);
            format!(
                "tar={tar} levitating={lev} sink={sink:.1} {}",
                verdict(lev == 0 && tar == 96 && sink > 2.0)
            )
        },
    },
    // The shear model's two claims, each pinned against its own failure mode.
    //
    // 100 ticks is chosen against both codes, not one: a 30-cell drop takes a
    // free-falling blob ~40 ticks, and under the old gate-everything rule tar
    // at 0.85 moved one cell per ~6.7 ticks — still ~170 ticks of air at the
    // century mark, so `airborne == 0` separates the two models cleanly. The
    // heap check is the other half: the blob lands 12 wide in a 58-wide box,
    // and levelling that spread through an 0.85 shear gate takes thousands of
    // ticks, so a still-ragged surface at t=100 proves the roll still gates
    // DEFORMATION. A wrong "fix" that ungated everything would land instantly
    // AND be pancake-flat by 100 — first check green, second red.
    Scenario {
        name: "tar-falls-fast-levels-slow",
        ticks: 100,
        build: |grid| {
            boxed(grid, 10, 80, 70, 130);
            fill_rect(grid, 30, 84, 12, 8, block::TAR);
        },
        check: |grid| {
            let airborne = levitating_in(grid, 11, 81, 69, 129);
            let tar = count_of(grid, block::TAR);
            let cols = grid.cols();
            // Surface skew: how many distinct rows hold the topmost tar of
            // some column. 1 = levelled flat; a fresh heap is many.
            let mut tops = std::collections::BTreeSet::new();
            for x in 11..69 {
                for y in 81..129 {
                    if grid.material[(y * cols + x) as usize] == block::TAR {
                        tops.insert(y);
                        break;
                    }
                }
            }
            let heap_rows = tops.len();
            format!(
                "tar={tar} airborne={airborne} heap-rows={heap_rows} {}",
                verdict(airborne == 0 && tar == 96 && heap_rows >= 3)
            )
        },
    },
];

fn run_scenarios() {
    println!("\nbehavioural scenarios (small sealed worlds run to rest):");
    for s in &SCENARIOS {
        let mut grid = CellGrid::new(WINDOW_COLS, WINDOW_ROWS);
        let mut sim = Automata::new();
        sim.seed(BENCH_SIM_SEED);
        (s.build)(&mut grid);
        grid.wake_all();
        for _ in 0..s.ticks {
            sim.simulate(&mut grid);
        }
        println!("  {:<26} {}", s.name, (s.check)(&grid));
    }
}

// --- The benchmarks ----------------------------------------------------------

/// The scene report: what the timed world contains, and what it did.
///
/// Printed before criterion runs so the numbers below it can be read against a
/// known world rather than an assumed one.
fn describe(grid: &mut CellGrid, sim: &mut Automata) {
    let before = digest(grid);
    for _ in 0..WARMUP_TICKS {
        sim.simulate(grid);
    }
    let after = digest(grid);
    assert_ne!(
        before, after,
        "the warmup changed nothing — the world is inert and every number below is meaningless"
    );

    let window = i64::from(grid.cols()) * i64::from(grid.rows());
    let swept = swept_cells(grid);
    println!(
        "world: {}x{} cells, {}x{} chunks ({CHUNK_CELLS}px), {} non-empty",
        grid.cols(),
        grid.rows(),
        grid.chunk_cols(),
        grid.chunk_rows(),
        census(grid)
    );
    println!(
        "digest after {WARMUP_TICKS} ticks: {after:08x}   awake chunks: {}/{}   swept: {swept} \
         cells/tick ({:.1}% of window)",
        grid.awake_chunk_count(),
        grid.chunk_cols() * grid.chunk_rows(),
        swept as f64 / window as f64 * 100.0
    );
}

/// How the scene's activity decays over the measured window.
///
/// This exists because the first attempt at this bench was WRONG in a way worth
/// leaving a guard against. It ran one evolving world under `Criterion::iter`,
/// as the TypeScript's 600-tick loop did — but criterion drives a sub-10 µs
/// body some 800 000 times, and by tick 800 000 the sand has all landed and the
/// fires have all burnt out. The result was a confident number for a world that
/// had been asleep for 99% of the run.
///
/// So the timed body is a fixed BATCH from a freshly-warmed world instead, and
/// this prints what "busy" decays to across that batch, so the mean below is
/// read against a known activity curve rather than an assumed steady state.
fn print_activity_curve() {
    let (mut grid, mut sim) = make_world();
    let window = i64::from(grid.cols()) * i64::from(grid.rows());
    print!("activity over the measured batch (swept cells as % of window):");
    for tick in 1..=WARMUP_TICKS + BATCH_TICKS {
        sim.simulate(&mut grid);
        if tick == WARMUP_TICKS
            || (tick > WARMUP_TICKS && (tick - WARMUP_TICKS).is_multiple_of(150))
        {
            print!(
                "  t{tick}={:.1}%",
                swept_cells(&grid) as f64 / window as f64 * 100.0
            );
        }
    }
    println!();
}

/// Ticks timed per criterion iteration.
///
/// The TypeScript's default run length, kept because it is the span over which
/// that scene's activity curve was judged reasonable — long enough that a fire
/// starts and finishes inside it, short enough that the world has not gone quiet
/// by the end. See [`print_activity_curve`].
const BATCH_TICKS: u32 = 600;

/// Does the tick cost actually track the amount of world it sweeps?
///
/// The brief every benchmark owes: prove the thing being timed is the thing
/// named. A tick the optimiser had elided, or a measurement dominated by fixed
/// per-tick overhead, would be FLAT across these — the whole point is that it is
/// not, and that the implied ns-per-swept-cell is roughly constant.
///
/// Each iteration force-wakes a full-height rect `num/den` of the window wide
/// before ticking, so the swept area is PINNED at that rect rather than left to
/// decay as the scene settles under a long criterion run. The `wake_rect` call is
/// inside the timing and is a handful of per-chunk unions — at the 1/8 point that
/// is the largest share it ever has, and it is still small against the sweep it
/// provokes.
///
/// The rect area is the throughput denominator, not a sampled `swept_cells`. The
/// forced rect is what every iteration is guaranteed to sweep; anything else
/// still moving is transient and dies away over the run, so a count taken before
/// the loop would be an over-estimate that shrinks as the loop proceeds — and a
/// denominator that drifts under the measurement produces a ns-per-cell figure
/// that says more about when it was sampled than about the code. The settled
/// value is printed afterwards so the two can be compared.
fn scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("automata/swept-area scaling");
    let mut settled = Vec::new();
    for (num, den) in [(1, 8), (1, 4), (1, 2), (1, 1)] {
        let (mut grid, mut sim) = make_world();
        for _ in 0..WARMUP_TICKS {
            sim.simulate(&mut grid);
        }
        let rows = grid.rows();
        let w = grid.cols() * num / den;
        let rect = i64::from(w) * i64::from(rows);

        group.throughput(Throughput::Elements(rect.unsigned_abs()));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{num}/{den} window")),
            |b| {
                b.iter(|| {
                    grid.wake_rect(0, 0, w, rows);
                    sim.simulate(black_box(&mut grid));
                });
            },
        );
        settled.push((num, den, rect, swept_cells(&grid)));
    }
    group.finish();

    println!("\nswept-area scaling denominators:");
    for (num, den, rect, actual) in settled {
        println!(
            "  {num}/{den} window: forced rect {rect} cells, actually swept {actual} at the end of \
             the run ({:+.1}%)",
            (actual - rect) as f64 / rect as f64 * 100.0
        );
    }
}

fn sim(c: &mut Criterion) {
    {
        let (mut grid, mut sim) = make_world();
        describe(&mut grid, &mut sim);
    }
    print_activity_curve();

    // A batch from a REBUILT warm world per iteration, not one world evolving
    // across all of them. `iter_batched` excludes the setup from the timing, so
    // what is measured is exactly ticks 121..720 of a fixed scene — the same
    // window the TypeScript timed, and stationary by construction rather than by
    // hope. The alternative is documented on `print_activity_curve`, along with
    // why it produced a number that was quietly meaningless.
    //
    // `LargeInput`: the setup allocates and warms a 90 112-cell world of four
    // planes, which criterion must not try to hold thousands of copies of.
    let mut group = c.benchmark_group("automata");
    // So criterion divides the batch out and reports per-TICK throughput
    // directly, rather than leaving a reader to divide by 600 by hand.
    group.throughput(Throughput::Elements(u64::from(BATCH_TICKS)));
    group.bench_function(
        BenchmarkId::new("busy world 352x256", format!("{BATCH_TICKS} ticks")),
        |b| {
            b.iter_batched_ref(
                || {
                    let (mut grid, mut sim) = make_world();
                    for _ in 0..WARMUP_TICKS {
                        sim.simulate(&mut grid);
                    }
                    (grid, sim)
                },
                |(grid, sim)| {
                    for _ in 0..BATCH_TICKS {
                        sim.simulate(black_box(grid));
                    }
                    // Cheap, and it is what proves the loop above was not
                    // discarded: a deleted body cannot produce a changed digest,
                    // and the assertion after the group checks one.
                    black_box(grid.awake_chunk_count())
                },
                BatchSize::LargeInput,
            );
        },
    );
    group.finish();

    // The whole-batch guard, run once outside the timing: the world at the end
    // of a batch must not be the world at the start of it.
    let (mut grid, mut sim) = make_world();
    for _ in 0..WARMUP_TICKS {
        sim.simulate(&mut grid);
    }
    let warm = digest(&grid);
    for _ in 0..BATCH_TICKS {
        sim.simulate(&mut grid);
    }
    println!(
        "\ndigest at t{WARMUP_TICKS}: {warm:08x}   at t{}: {:08x}",
        WARMUP_TICKS + BATCH_TICKS,
        digest(&grid)
    );
    assert_ne!(warm, digest(&grid), "the timed batch changed nothing");

    run_scenarios();
}

criterion_group!(benches, sim, scaling);
criterion_main!(benches);

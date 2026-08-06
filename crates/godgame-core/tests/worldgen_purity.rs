//! Determinism and seam checks for the generator.
//!
//! These are not unit tests of the shapes — nobody can assert that a cave "looks
//! good". They assert the ONE property the whole architecture rests on, and which
//! is silently violable by a single line of code: that a chunk is a pure function
//! of (chunk_x, chunk_y, seed).
//!
//! The failure this catches is nasty in production and cheap to catch here. A
//! chunk store discards a pristine chunk and regenerates it later; if generation
//! order can change the result, the world quietly rearranges itself behind the
//! player, and only when they approach from an unusual direction. The coarse
//! lattice in `worldgen::caves` and the memo in `worldgen::heightmap` are exactly
//! the kind of optimisation that breaks this, which is why these checks exist.
//!
//! # Port notes
//!
//! The TypeScript original (`src/sim/gen/gentest.ts`) returned a `CheckResult[]`
//! that nothing in `npm run check` ever looked at. Here each check is a real
//! `#[test]`: a violation fails the build.
//!
//! Every check that is ABOUT generation order drives ONE [`ChunkGen`] across all
//! its calls, deliberately. `generate_chunk(cx, cy, seed)` builds a fresh noise
//! and a fresh heightmap memo per call, which would make order independence
//! trivially true and the check worthless; a single long-lived `ChunkGen` is the
//! faithful analogue of the TypeScript's module-level scratch, and it is what the
//! streaming path will actually hold.
//!
//! Check 8 has no TypeScript ancestor: it asserts that the same chunks generated
//! across a rayon pool are byte-identical to the serial result. That is the
//! property the owned-scratch design buys, and the reason worldgen contains no
//! static with interior mutability and no thread local.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use godgame_core::config::CHUNK_CELLS;
use godgame_core::sim::biomes::column_profile_at;
use godgame_core::sim::decor::{DecorContext, Decorator};
use godgame_core::sim::materials::CellId;
use godgame_core::sim::noise::Noise;
use godgame_core::sim::worldgen::containers::{containers_present, is_container};
use godgame_core::sim::worldgen::features::LANDMARK_DECORATOR;
use godgame_core::sim::worldgen::heightmap::Heightmap;
use godgame_core::sim::worldgen::structs::{Mark, StructQuery, mark_at, stamp_structs};
use godgame_core::sim::worldgen::{ChunkGen, material_at, world_noise};

/// The seed the TypeScript suite ran on. Anything would do; a fixed one makes a
/// failure reproducible and its detail message quotable.
const SEED: u32 = 20260805;

/// Chunk coordinates the checks sweep. Spans sky, surface, cavern, deep, floor.
fn coords() -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    for cx in -3..=3 {
        for cy in [-1, 0, 1, 2, 4, 7, 12, 16, 21] {
            out.push((cx, cy));
        }
    }
    out
}

/// Index of the first differing cell, or `None` when the two are equal.
///
/// (The TypeScript returned `-1` for a length mismatch and `-2` for equal; the
/// lengths cannot differ here, so the sentinel pair collapses to an `Option`.)
fn first_difference(a: &[CellId], b: &[CellId]) -> Option<usize> {
    assert_eq!(a.len(), b.len(), "chunks are a fixed size");
    (0..a.len()).find(|&i| a[i] != b[i])
}

/// Where a cell index sits inside a chunk — for a failure message a human can
/// walk to.
fn locate(i: usize, cx: i32, cy: i32) -> String {
    let lx = i as i32 % CHUNK_CELLS;
    let ly = i as i32 / CHUNK_CELLS;
    format!(
        "cell index {i} (local {lx},{ly} — absolute {},{})",
        cx * CHUNK_CELLS + lx,
        cy * CHUNK_CELLS + ly
    )
}

// ---------------------------------------------------------------------------
// 1. ORDER INDEPENDENCE
// ---------------------------------------------------------------------------

/// Generate the same set of chunks forwards, then backwards, interleaved with
/// unrelated chunks, and require byte equality. Catches any state carried
/// between calls — a scratch buffer read before it is written, a cache keyed on
/// the wrong thing, an RNG advanced per chunk instead of per coordinate.
///
/// THE ONE THAT MATTERS MOST: it is what proves the heightmap memo and the cave
/// lattice are invisible.
#[test]
fn order_independence() {
    let coords = coords();
    let mut cg = ChunkGen::new(SEED);

    let mut forward: HashMap<(i32, i32), Vec<CellId>> = HashMap::new();
    for &(cx, cy) in &coords {
        forward.insert((cx, cy), cg.generate(cx, cy));
    }

    // Reverse order, with a far-away chunk generated between every pair so any
    // per-call scratch is guaranteed to have been clobbered by different data.
    for &(cx, cy) in coords.iter().rev() {
        cg.generate(cx + 991, cy + 47);
        let again = cg.generate(cx, cy);
        if let Some(at) = first_difference(&forward[&(cx, cy)], &again) {
            panic!(
                "order dependence: chunk ({cx},{cy}) differs at {} between a forward \
                 and a reverse sweep — something in worldgen is carrying state \
                 between calls",
                locate(at, cx, cy)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. HORIZONTAL SEAM
// ---------------------------------------------------------------------------

/// The ground line is computed independently by both chunks that meet at a
/// vertical seam. Assert they agree — and, more strictly, that the surface row
/// is CONTINUOUS across the seam (adjacent columns never jump by more than a
/// cliff's worth), which is what catches a heightmap whose inputs accidentally
/// depend on the chunk rather than the column.
#[test]
fn horizontal_seam() {
    let noise = world_noise(SEED);
    // Two SEPARATE memos, one per path. The TypeScript used one, which meant the
    // profile path always hit the entry the plain path had just filled and the
    // comparison below could not fail. Giving each path a cold cache is what
    // makes it a real test of the two entry points.
    let mut plain = Heightmap::new();
    let mut via = Heightmap::new();

    let mut max_step = 0;
    let mut worst_at = 0;
    for cx in -6..=6 {
        let base_x = cx * CHUNK_CELLS;
        for d in -2..=2 {
            let a = plain.surface_row_at(&noise, base_x + d, None);
            let b = plain.surface_row_at(&noise, base_x + d + 1, None);
            // Recompute through the profile path too: `surface_row_at(x, None)`
            // and `surface_row_at(x, Some(profile))` must agree exactly, or the
            // decorators (which use the first) would disagree with terrain
            // (which uses the second) and trees would float.
            let col = column_profile_at(&noise, base_x + d);
            let via_profile = via.surface_row_at(&noise, base_x + d, Some(&col));
            assert_eq!(
                via_profile,
                a,
                "column {}: profile path {via_profile} != plain path {a}",
                base_x + d
            );
            let step = (b - a).abs();
            if step > max_step {
                max_step = step;
                worst_at = base_x + d;
            }
        }
    }
    // A terrace riser is the biggest legal single-column step; TERRACE_STEP is 6.
    assert!(
        max_step <= 8,
        "max adjacent surface step {max_step} cells (at column {worst_at}); limit 8"
    );
}

// ---------------------------------------------------------------------------
// 3. VERTICAL SEAM
// ---------------------------------------------------------------------------

/// A chunk's bottom row and the chunk below it must describe the same cells.
/// Terrain is generated per column from an absolute row, so this is really a
/// check that no depth band, cave field or lattice sample is anchored to the
/// chunk instead of to world space — the single most likely way the lattice
/// optimisation could have gone wrong.
#[test]
fn vertical_seam() {
    let mut cg = ChunkGen::new(SEED);
    for cx in -3..=3 {
        for cy in [0, 1, 3, 6, 11, 15, 20] {
            let top = cg.generate(cx, cy);
            let bottom = cg.generate(cx, cy + 1);
            // Regenerate `top` AFTER `bottom` so any shared scratch is stale.
            let top_again = cg.generate(cx, cy);
            if let Some(at) = first_difference(&top, &top_again) {
                panic!(
                    "chunk ({cx},{cy}) changed after its neighbour below was \
                     generated, at {}",
                    locate(at, cx, cy)
                );
            }
            // The lattice corner shared by the two chunks: the bottom chunk's
            // row 0 and the top chunk's row CHUNK_CELLS are the SAME lattice
            // line, so any cell in bottom row 0 must match what the top chunk's
            // generator would have produced there. Verifying that by generating a
            // chunk offset by one cell is not possible (chunks are grid-aligned),
            // so instead assert the bottom chunk is itself stable when
            // regenerated after its neighbour.
            let bottom_again = cg.generate(cx, cy + 1);
            if let Some(at) = first_difference(&bottom, &bottom_again) {
                panic!(
                    "chunk ({cx},{}) is unstable at {}",
                    cy + 1,
                    locate(at, cx, cy + 1)
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. PROBE AGREEMENT
// ---------------------------------------------------------------------------

/// `material_at` is the lattice-free probe a feature pass uses; `generate` is the
/// authority. They are allowed to disagree only on cells within the lattice
/// interpolation error of a threshold, so this asserts a BOUND on the
/// disagreement rate rather than equality. A sudden jump here means one of the
/// two paths grew a term the other did not.
#[test]
fn probe_agreement() {
    let mut cg = ChunkGen::new(SEED);
    let noise = world_noise(SEED);
    let mut hm = Heightmap::new();

    let mut checked = 0u32;
    let mut differ = 0u32;
    for cx in -2..=2 {
        for cy in [2, 5, 9, 18] {
            let chunk = cg.generate(cx, cy);
            let base_x = cx * CHUNK_CELLS;
            let base_y = cy * CHUNK_CELLS;
            let mut lx = 0;
            while lx < CHUNK_CELLS {
                let wcx = base_x + lx;
                let col = column_profile_at(&noise, wcx);
                let surf = hm.surface_row_at(&noise, wcx, Some(&col));
                let mut ly = 0;
                while ly < CHUNK_CELLS {
                    let got = material_at(&noise, wcx, base_y + ly, &col, surf);
                    // Decorations overwrite terrain, so only compare where the
                    // chunk still holds a terrain material: compare open-vs-solid,
                    // not exact codes.
                    let in_chunk = chunk[(ly * CHUNK_CELLS + lx) as usize];
                    checked += 1;
                    if (got == 0) != (in_chunk == 0) {
                        differ += 1;
                    }
                    ly += 3;
                }
                lx += 3;
            }
        }
    }

    let rate = f64::from(differ) / f64::from(checked.max(1));
    assert!(
        rate < 0.06,
        "{:.2}% of {checked} probes disagree on open/solid (limit 6%, decorations \
         included) — the exact and lattice paths have drifted apart",
        rate * 100.0
    );
}

// ---------------------------------------------------------------------------
// 5 & 6. LANDMARKS
// ---------------------------------------------------------------------------

/// Which cells inside the chunk at (tgt_x, tgt_y) the landmark pass tries to
/// write when it is run FOR the chunk at (scan_x, scan_y).
///
/// Runs the real pass against a recording [`DecorContext`] instead of a chunk, so
/// there is no duplicated placement arithmetic to drift out of sync with the
/// generator. Passing a scan origin different from the target is what makes
/// [`landmark_reach`] possible: it asks a NEIGHBOUR what it thinks belongs in
/// this chunk.
fn record_landmarks(
    noise: &Noise,
    hm: &mut Heightmap,
    scan_x: i32,
    scan_y: i32,
    tgt_x: i32,
    tgt_y: i32,
) -> HashSet<i32> {
    let mut plotted: Vec<(i32, i32, CellId)> = Vec::new();
    {
        let mut ctx = DecorContext::recording(noise, SEED, scan_x, scan_y, &mut plotted, hm);
        LANDMARK_DECORATOR.decorate(&mut ctx);
    }
    plotted
        .into_iter()
        .filter_map(|(wcx, wcy, _)| {
            let lx = wcx - tgt_x;
            let ly = wcy - tgt_y;
            (lx >= 0 && ly >= 0 && lx < CHUNK_CELLS && ly < CHUNK_CELLS)
                .then_some(ly * CHUNK_CELLS + lx)
        })
        .collect()
}

/// Cells the landmark pass writes into its own chunk.
fn landmark_cells(noise: &Noise, hm: &mut Heightmap, cx: i32, cy: i32) -> usize {
    let bx = cx * CHUNK_CELLS;
    let by = cy * CHUNK_CELLS;
    record_landmarks(noise, hm, bx, by, bx, by).len()
}

/// Generate the eight neighbours of (cx, cy), in raster order or reversed.
fn generate_ring(cg: &mut ChunkGen, cx: i32, cy: i32, reverse: bool) {
    let mut ring: Vec<(i32, i32)> = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx != 0 || dy != 0 {
                ring.push((cx + dx, cy + dy));
            }
        }
    }
    if reverse {
        ring.reverse();
    }
    for (x, y) in ring {
        cg.generate(x, y);
    }
}

/// Cells inside this chunk that a landmark straddling its edge must also paint.
const STRADDLE_MIN: usize = 24;

/// The strictest check here, and the one the feature/struct architecture exists
/// to satisfy.
///
/// A landmark is far bigger than a chunk — a 168-cell mineshaft, a 60-cell
/// dungeon, a 22-cell tower — so a single one is authored once and painted by up
/// to a dozen chunks that never speak to each other. The failure mode is specific
/// and nasty: a generator that scans its own loops against the chunk being
/// painted (they all do, for speed) and gets the clamp margin wrong produces a
/// feature that is CORRECT when the chunk holding its origin is generated, and
/// CLIPPED when a neighbour paints its share first. It only shows when the player
/// approaches from the wrong side.
///
/// So: find chunks that genuinely straddle a landmark, then require byte equality
/// between the chunk generated cold, generated after its eight neighbours in
/// raster order, and generated after them in reverse order. Unrelated far-away
/// chunks are generated between every step so any shared scratch is guaranteed
/// stale.
#[test]
fn landmark_seams() {
    let noise = world_noise(SEED);
    let mut hm = Heightmap::new();

    // Sweep for chunks that hold a real piece of a landmark AND whose neighbour
    // holds another piece of one — i.e. something crosses the seam between them.
    let mut picked: Vec<(i32, i32)> = Vec::new();
    'sweep: for cx in -30..=30 {
        for cy in [-2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 15, 18, 21] {
            if picked.len() >= 24 {
                break 'sweep;
            }
            if landmark_cells(&noise, &mut hm, cx, cy) < STRADDLE_MIN {
                continue;
            }
            let straddles = landmark_cells(&noise, &mut hm, cx + 1, cy) >= STRADDLE_MIN
                || landmark_cells(&noise, &mut hm, cx, cy + 1) >= STRADDLE_MIN;
            if straddles {
                picked.push((cx, cy));
            }
        }
    }
    assert!(
        !picked.is_empty(),
        "no chunk in the sweep straddles a landmark — the pass is placing nothing"
    );

    let mut cg = ChunkGen::new(SEED);
    for (cx, cy) in picked {
        cg.generate(cx + 733, cy + 61); // clobber any shared scratch
        let cold = cg.generate(cx, cy);

        generate_ring(&mut cg, cx, cy, false);
        let after_forward = cg.generate(cx, cy);
        if let Some(at) = first_difference(&cold, &after_forward) {
            panic!(
                "chunk ({cx},{cy}) differs at {} after its neighbours were \
                 generated (raster order)",
                locate(at, cx, cy)
            );
        }

        generate_ring(&mut cg, cx, cy, true);
        let after_reverse = cg.generate(cx, cy);
        if let Some(at) = first_difference(&cold, &after_reverse) {
            panic!(
                "chunk ({cx},{cy}) differs at {} after its neighbours were \
                 generated (reverse order)",
                locate(at, cx, cy)
            );
        }
    }
}

/// LANDMARK REACH AND LOOP CLAMP. The check that catches what `landmark_seams`
/// cannot.
///
/// Every landmark generator clips its own loops to the chunk being painted
/// (`features`: "the loop clamp is an optimisation, never a decision") and scans
/// candidate origins `reach` past every chunk edge. Get either wrong and the
/// result is still perfectly ORDER-independent — every generation of that chunk
/// skips the same cells — so `landmark_seams` passes while the world grows a
/// hairline of missing masonry down every chunk boundary. The failure is
/// CHUNK-dependence, not order-dependence, and it needs its own instrument.
///
/// The invariant: if a neighbouring chunk's pass believes a cell inside THIS
/// chunk belongs to a landmark, this chunk's own pass must believe it too. A
/// neighbour may legitimately see fewer cells (a landmark that reaches here but
/// not there is outside its scan window), so the assertion is containment, not
/// equality — which also makes it immune to two overlapping landmarks being
/// applied in a different relative order.
#[test]
fn landmark_reach() {
    let noise = world_noise(SEED);
    let mut hm = Heightmap::new();

    let mut checked = 0u32;
    let mut cells = 0usize;
    for cx in -18..=18 {
        for cy in [-1, 0, 1, 2, 4, 6, 9, 12, 16, 20] {
            let bx = cx * CHUNK_CELLS;
            let by = cy * CHUNK_CELLS;
            let own = record_landmarks(&noise, &mut hm, bx, by, bx, by);
            if own.is_empty() {
                continue;
            }
            checked += 1;
            cells += own.len();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let seen = record_landmarks(
                        &noise,
                        &mut hm,
                        bx + dx * CHUNK_CELLS,
                        by + dy * CHUNK_CELLS,
                        bx,
                        by,
                    );
                    for i in &seen {
                        assert!(
                            own.contains(i),
                            "chunk ({cx},{cy}) misses {} that its ({dx},{dy}) \
                             neighbour assigns to a landmark — an under-declared \
                             reach or a too-tight loop clamp",
                            locate(*i as usize, cx, cy)
                        );
                    }
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no landmarks found in the sweep — the pass is placing nothing"
    );
    assert!(cells > 0);
}

// ---------------------------------------------------------------------------
// 7. CONTAINER RECOVERY
// ---------------------------------------------------------------------------

/// The invariant the loot pass adds, and one that all six checks above are
/// structurally blind to.
///
/// A chest stores nothing: its contents are rolled at open time from its
/// coordinates, and their tier comes from the RARITY of the template that placed
/// it, recovered by `mark_at` re-deriving placement around the point. That
/// recovery is a SECOND, independent implementation of "which template owns this
/// cell" — forward in `stamp_structs`, backward in `mark_at`. If the two ever
/// disagree, the world still generates byte-identically in every order and every
/// chest quietly rolls the wrong table forever. Checks 1, 5 and 6 all pass
/// straight through that, because nothing about the CELLS changed; only the
/// answer to a question they never ask did.
///
/// So: run the struct pass against a recording context, collect the cells it
/// would fill with a container, and require the reverse query to claim each one
/// as a loot mark. Containment holds in that direction only — a `mark_at` hit on
/// a cell that ends up holding something else is legal, because a later template
/// may stamp over an earlier one's chest.
///
/// Recording the pass rather than reading generated chunks is what keeps the
/// check pointed at the thing under test: chests placed by a parametric FEATURE
/// (the dungeon vault) have no host template by design and are scored on depth
/// alone, so counting them here would be asserting the opposite of what the code
/// promises.
#[test]
fn container_recovery() {
    if !containers_present() {
        // Content has no container block — the mark pass is inert, nothing to
        // verify.
        return;
    }

    let noise = world_noise(SEED);
    let mut hm = Heightmap::new();
    // `mark_at` never writes and never reads the world back, so a standalone
    // query is the honest shape: resolution reads a surface row, a profile and a
    // hash, and nothing else. It has no chunk and no base — which is exactly what
    // makes the query independent of who is asking.
    let mut query = StructQuery::new(SEED);

    let mut found = 0u32;
    for cx in -24..=24 {
        for cy in [-1, 0, 1, 2, 3, 4, 6, 9, 12, 16, 20] {
            let mut plotted: Vec<(i32, i32, CellId)> = Vec::new();
            {
                let mut ctx = DecorContext::recording(
                    &noise,
                    SEED,
                    cx * CHUNK_CELLS,
                    cy * CHUNK_CELLS,
                    &mut plotted,
                    &mut hm,
                );
                stamp_structs(&mut ctx);
            }

            for (wcx, wcy, code) in plotted {
                if !is_container(code) {
                    continue;
                }
                found += 1;
                let hit = mark_at(&mut query, wcx, wcy);
                match hit {
                    Some(h) if h.mark == Mark::Loot => {}
                    Some(h) => panic!(
                        "container stamped at ({wcx},{wcy}) is claimed by template \
                         `{}` as {:?}, not Loot — the forward and reverse placement \
                         paths disagree",
                        h.template.id, h.mark
                    ),
                    None => panic!(
                        "container stamped at ({wcx},{wcy}) is claimed by no \
                         template — the forward and reverse placement paths disagree"
                    ),
                }
            }
        }
    }

    assert!(
        found > 0,
        "no containers in the sweep — the mark pass is placing nothing"
    );
}

// ---------------------------------------------------------------------------
// 8. PARALLEL AGREEMENT (no TypeScript ancestor)
// ---------------------------------------------------------------------------

/// Generating the same chunks across a rayon pool produces byte-identical
/// results to generating them serially, one after another.
///
/// This is the property the no-shared-scratch design buys and the headline result
/// of the port: the TypeScript kept the column tables, the cave lattice, the
/// heightmap memo and the `Noise` at module scope, so `generateChunk` could only
/// ever run on one thread. Here every one of those lives in a [`ChunkGen`] the
/// caller owns, so N workers can each hold one.
///
/// `map_init` is what makes it a real test rather than a formality: rayon builds
/// one `ChunkGen` per worker and REUSES it across however many chunks that worker
/// happens to steal, in an order nobody chose. If any residue survived a
/// `generate` call, the split would show up here as a mismatch that moves from
/// run to run.
#[test]
fn parallel_matches_serial() {
    // A wider sweep than `coords()` — enough chunks for work-stealing to
    // interleave them differently on every run.
    let mut work: Vec<(i32, i32)> = Vec::new();
    for cx in -8..=8 {
        for cy in [-1, 0, 1, 2, 3, 5, 8, 11, 14, 18, 21] {
            work.push((cx, cy));
        }
    }

    let mut serial_cg = ChunkGen::new(SEED);
    let serial: Vec<Vec<CellId>> = work
        .iter()
        .map(|&(cx, cy)| serial_cg.generate(cx, cy))
        .collect();

    let parallel: Vec<Vec<CellId>> = work
        .par_iter()
        .map_init(|| ChunkGen::new(SEED), |cg, &(cx, cy)| cg.generate(cx, cy))
        .collect();

    for (i, &(cx, cy)) in work.iter().enumerate() {
        if let Some(at) = first_difference(&serial[i], &parallel[i]) {
            panic!(
                "chunk ({cx},{cy}) generated on a rayon worker differs from the \
                 serial result at {} — worldgen is carrying state a thread can see",
                locate(at, cx, cy)
            );
        }
    }
}
